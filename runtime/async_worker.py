"""
[批次 C3] runtime.async_worker — 异步沉淀 worker。

目标（批次 C3）：
    把 ON_EXIT_LOOP 中的记忆合并、技能评估等"沉淀"动作从主流程剥离到后台
    worker，使 ON_EXIT_LOOP 能快速返回，不阻塞调用方。

设计约束：
    - 不修改 Agent Loop（不改变钩子调度、不改变中间件执行语义）。
    - 异步化通过后台 worker（线程池 + 任务队列）实现，并附带一个最小事件总线，
      满足"事件总线或后台 worker"的交付要求。
    - 所有异步路径都必须有超时与降级：
        * flush(timeout) 带超时；
        * worker 提交失败或 worker 本身为 None 时，降级为**同步执行**，
          保证沉淀动作绝不丢失。

组件：
    - AsyncWorker:  后台线程池 worker（submit / flush / shutdown / pending 计数）。
    - EventBus:     线程安全的主题事件总线（publish / subscribe / subscribe_async）。
    - AsyncSedimentationMiddleware:  ON_EXIT_LOOP 时异步提交"记忆合并 + 技能评估"。

用法::

    from runtime.async_worker import AsyncWorker, AsyncSedimentationMiddleware

    worker = AsyncWorker(max_workers=2)
    mw = AsyncSedimentationMiddleware(worker=worker)
    runtime.register_middleware(mw)
    session = runtime.run(...)     # ON_EXIT_LOOP 立即返回，沉淀在后台跑
    mw.flush(timeout=5.0)          # 等待沉淀完成（供测试断言）
"""

from __future__ import annotations

import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Callable, Dict, List, Optional

from agent_loop import Context, HookResult, Middleware


# ============================================================================
# AsyncWorker —— 后台线程池 worker
# ============================================================================


class AsyncWorker:
    """后台线程池任务 worker。

    维护一个 ThreadPoolExecutor 与 pending future 队列，提供：
        - submit(fn, *args, **kwargs) 提交任务，立即返回 Future；
        - flush(timeout) 等待全部已提交任务完成（带超时）；
        - shutdown() 关闭线程池。

    线程安全：pending 队列操作由锁保护。
    """

    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers or 2)))
        self._lock = threading.Lock()
        self._pending: List[Future] = []
        self._submitted = 0
        self._completed = 0
        self._closed = False

    # ------------------------------------------------------------------
    # 提交 / 等待 / 关闭
    # ------------------------------------------------------------------

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """提交任务。worker 已关闭或线程池异常时降级同步执行并返回已完成的 Future。"""
        self._submitted += 1
        try:
            future = self._executor.submit(fn, *args, **kwargs)
        except Exception:
            # 降级：线程池不可用（如 shutdown 后）→ 同步执行，保证不丢失。
            return self._sync_future(fn, args, kwargs)
        with self._lock:
            self._pending.append(future)
        future.add_done_callback(self._on_done)
        return future

    def flush(self, timeout: Optional[float] = None) -> None:
        """等待所有已提交任务完成。

        对每个 pending future 以 timeout 等待；超时的任务被记录为可观测结果，
        但不抛出（保持"超时降级"语义——不阻塞调用方）。
        """
        with self._lock:
            pending = list(self._pending)
        for future in pending:
            try:
                future.result(timeout=timeout)
            except FutureTimeout:
                # 超时降级：不阻塞，留给后台继续
                continue
            except Exception:
                # 任务自身异常已由 _on_done 捕获计数，这里静默
                continue

    def shutdown(self, wait: bool = True) -> None:
        """关闭线程池（幂等）。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._executor.shutdown(wait=wait)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------

    @property
    def pending_count(self) -> int:
        """当前尚未完成的任务数。"""
        with self._lock:
            return sum(1 for f in self._pending if not f.done())

    @property
    def submitted_count(self) -> int:
        return self._submitted

    @property
    def completed_count(self) -> int:
        return self._completed

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _on_done(self, future: Future) -> None:
        with self._lock:
            self._completed += 1
            try:
                self._pending.remove(future)
            except ValueError:
                pass

    def _sync_future(self, fn: Callable[..., Any], args: tuple, kwargs: dict) -> Future:
        """构造一个已完成的 Future（同步执行降级路径）。"""
        future: Future = Future()
        try:
            result = fn(*args, **kwargs)
            future.set_result(result)
        except BaseException as exc:  # noqa: BLE001 - 同步降级需捕获所有异常
            future.set_exception(exc)
        return future


# ============================================================================
# EventBus —— 最小事件总线
# ============================================================================


class EventBus:
    """线程安全的最小事件总线。

    支持同步 subscribe 与异步 subscribe_async（经由 AsyncWorker 派发）。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._handlers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}

    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """注册同步处理器。"""
        with self._lock:
            self._handlers.setdefault(topic, []).append(handler)

    def subscribe_async(
        self,
        worker: AsyncWorker,
        topic: str,
        handler: Callable[[Dict[str, Any]], None],
    ) -> None:
        """注册异步处理器：事件由 worker 后台派发。"""
        def _dispatch(payload: Dict[str, Any]) -> None:
            worker.submit(handler, payload)

        self.subscribe(topic, _dispatch)

    def publish(self, topic: str, **payload: Any) -> List[Any]:
        """同步派发事件到所有处理器，返回每个处理器的返回值。"""
        with self._lock:
            handlers = list(self._handlers.get(topic, []))
        results: List[Any] = []
        for handler in handlers:
            try:
                results.append(handler(payload))
            except Exception:
                traceback.print_exc()
                results.append(None)
        return results


# ============================================================================
# AsyncSedimentationMiddleware —— 异步沉淀中间件
# ============================================================================


class AsyncSedimentationMiddleware(Middleware):
    """ON_EXIT_LOOP 时把记忆合并与技能评估异步提交到后台 worker。

    原本由 MemoryConsolidationMiddleware / SkillObservationMiddleware 在
    ON_EXIT_LOOP 同步执行的沉淀动作，这里改为提交到 AsyncWorker 后台执行，
    中间件本身立即返回，使 ON_EXIT_LOOP / ``runtime.run`` 不被阻塞。

    降级：
        - worker 为 None 或提交失败时，同步执行沉淀，保证动作不丢失。

    可观测：
        - ``submitted`` / ``completed`` 计数；
        - ``results`` 列表，每项 ``{"kind": str, "ok": bool}``。
    """

    def __init__(
        self,
        worker: Optional[AsyncWorker] = None,
        context_manager: Any = None,
        skill_system: Any = None,
        name: str = "async_sedimentation",
    ) -> None:
        super().__init__(name)
        self._worker = worker
        self._context_manager = context_manager
        self._skill_system = skill_system

        self.submitted = 0
        self.completed = 0
        self.results: List[Dict[str, Any]] = []

    def on_exit_loop(self, ctx: Context) -> HookResult:
        """把沉淀动作异步化，立即返回。"""
        cm = self._resolve_context_manager(ctx)
        skill = self._resolve_skill_system(ctx)
        namespace = self._resolve_namespace(ctx)
        task = self._resolve_task(ctx)

        if cm is not None:
            self._submit(
                kind="memory_consolidation",
                fn=lambda: self._run_consolidation(cm, namespace),
            )
        if skill is not None:
            self._submit(
                kind="skill_evaluation",
                fn=lambda: self._run_skill_evaluation(skill, task),
            )

        return HookResult.continue_()

    def flush(self, timeout: Optional[float] = None) -> None:
        """等待后台沉淀任务完成（供测试断言）。"""
        if self._worker is not None:
            self._worker.flush(timeout=timeout)

    def shutdown(self, wait: bool = True) -> None:
        """关闭内部 worker（若为本中间件创建）。"""
        if self._worker is not None:
            self._worker.shutdown(wait=wait)

    # ------------------------------------------------------------------
    # 沉淀执行
    # ------------------------------------------------------------------

    def _run_consolidation(self, cm: Any, namespace: Optional[str]) -> int:
        consolidator = getattr(cm, "_consolidate_episodic", None)
        if not callable(consolidator):
            return 0
        try:
            result = consolidator(namespace=namespace)
        except TypeError:
            try:
                result = consolidator(namespace)
            except Exception:
                return 0
        except Exception:
            return 0
        try:
            return int(result or 0)
        except (TypeError, ValueError):
            return 0

    def _run_skill_evaluation(self, skill: Any, task: str) -> bool:
        tracker = getattr(skill, "track_signal", None)
        if not callable(tracker):
            return False
        try:
            tracker()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # 提交路径（含降级）
    # ------------------------------------------------------------------

    def _submit(self, kind: str, fn: Callable[[], Any]) -> None:
        self.submitted += 1

        def _wrapped() -> Any:
            try:
                result = fn()
                self.results.append({"kind": kind, "ok": True})
                return result
            except BaseException:  # noqa: BLE001 - 记录后可观测
                self.results.append({"kind": kind, "ok": False})
                return None
            finally:
                self.completed += 1

        if self._worker is not None:
            try:
                self._worker.submit(_wrapped)
                return
            except Exception:
                # 降级为同步执行
                pass
        _wrapped()

    # ------------------------------------------------------------------
    # 依赖解析
    # ------------------------------------------------------------------

    def _resolve_context_manager(self, ctx: Context) -> Any:
        if self._context_manager is not None:
            return self._context_manager
        shared = getattr(ctx, "shared", None)
        if isinstance(shared, dict):
            return shared.get("context_manager")
        return None

    def _resolve_skill_system(self, ctx: Context) -> Any:
        if self._skill_system is not None:
            return self._skill_system
        shared = getattr(ctx, "shared", None)
        if isinstance(shared, dict):
            return shared.get("skill_system")
        return None

    def _resolve_namespace(self, ctx: Context) -> Optional[str]:
        shared = getattr(ctx, "shared", None)
        if isinstance(shared, dict):
            ns = shared.get("memory_namespace")
            if ns:
                return str(ns)
        return None

    @staticmethod
    def _resolve_task(ctx: Context) -> str:
        shared = getattr(ctx, "shared", None)
        if not isinstance(shared, dict):
            return ""
        session = shared.get("session")
        if session is None:
            return ""
        metadata = getattr(session, "metadata", None)
        if isinstance(metadata, dict):
            return str(metadata.get("task", "") or "")
        return ""


__all__ = ["AsyncWorker", "EventBus", "AsyncSedimentationMiddleware"]