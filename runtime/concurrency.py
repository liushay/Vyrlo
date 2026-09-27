"""
[批次 C3] runtime.concurrency — 子 Agent 并发调度。

目标（批次 C3）：
    把多个委派子 Agent 的**并行**执行能力放在 Runtime 层（不修改 Agent Loop）。

设计约束：
    - 不修改 Agent Loop：并发通过 ``ThreadPoolExecutor`` 在线程中各自调用
      Runtime 的委派执行逻辑（``runtime._run_delegation``）实现。
    - 所有并发路径都有超时与降级：
        * 每个任务可设 ``timeout``，超时记为 ``timed_out`` 而非阻塞；
        * 单个任务抛异常时不影响其他任务；
        * worker 数为 1 或任务数 <= 1 时自然退化为串行。
    - 无状态污染：Runtime 的委派深度 / active_ctx 均使用 ``threading.local``，
      线程间天然隔离；结果收集用锁保护，避免并发写同一共享容器。

组件：
    - TaskOutcome:           单个并发任务的结果。
    - ConcurrentScheduler:   通用并发调度器（线程池 + 超时 + 结果聚合）。
    - delegate_parallel:     高层便捷函数 —— 对一组委派请求并行调用
                             ``runtime._run_delegation``。

用法::

    from runtime.concurrency import ConcurrentScheduler, delegate_parallel

    sched = ConcurrentScheduler(max_workers=3)
    outcomes = sched.run_parallel([
        (lambda: slow_fn(1), None),
        (lambda: slow_fn(2), None),
        (lambda: slow_fn(3), None),
    ])
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

from runtime.agent_registry.interface import AgentSpec


# ============================================================================
# TaskOutcome —— 单个并发任务结果
# ============================================================================


@dataclass
class TaskOutcome:
    """单个并发任务的结果。

    Attributes:
        index:      任务在输入列表中的下标（保持顺序，便于调用方对齐）。
        success:    是否成功（无异常且未超时）。
        result:     任务返回值（出错 / 超时 / 被取消时为 None）。
        error:      任务抛出的异常（无则为 None）。
        elapsed:    任务实际耗时（秒）。
        timed_out:  是否因超时被放弃等待（线程可能仍在后台运行）。
    """

    index: int
    success: bool = True
    result: Any = None
    error: Optional[BaseException] = None
    elapsed: float = 0.0
    timed_out: bool = False


# ============================================================================
# ConcurrentScheduler —— 通用并发调度器
# ============================================================================


class ConcurrentScheduler:
    """基于线程池的并发调度器，带超时与结果聚合。

    Attributes:
        max_workers: 线程池大小（<=1 时行为等价串行）。
        history:     最近一次 ``run_parallel`` 的总耗时（秒）。
    """

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max(1, int(max_workers or 1))
        self.history: float = 0.0
        # 结果收集线程安全
        self._result_lock = threading.Lock()

    # ------------------------------------------------------------------
    # 并行调度
    # ------------------------------------------------------------------

    def run_parallel(
        self,
        tasks: Sequence[Tuple[Callable[[], Any], Optional[float]]],
    ) -> List[TaskOutcome]:
        """并行执行 ``tasks``，返回与输入等长的结果列表（保持顺序）。

        Args:
            tasks: ``[(fn, timeout_seconds_or_None), ...]`` 序列。

        Returns:
            与输入顺序一致的 ``TaskOutcome`` 列表。

        语义：
            - 每个任务独立捕获异常；
            - 超时任务不阻塞其他任务，标记 ``timed_out=True``；
            - 总耗时记录到 ``self.history``。
        """
        tasks = list(tasks)
        if not tasks:
            self.history = 0.0
            return []

        start = time.perf_counter()

        # 线程池共享，逐个 submit
        outcomes: List[Optional[TaskOutcome]] = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures: List[Tuple[int, Future]] = []
            for idx, (fn, _timeout) in enumerate(tasks):
                futures.append((idx, executor.submit(self._run_one, idx, fn)))

            for idx, future in futures:
                timeout = tasks[idx][1]
                self._record(
                    outcomes, idx, future, timeout=timeout,
                )

        self.history = time.perf_counter() - start
        return [o for o in outcomes if o is not None]

    # ------------------------------------------------------------------
    # 串行调度（对照基准）
    # ------------------------------------------------------------------

    def run_serial(
        self,
        tasks: Sequence[Tuple[Callable[[], Any], Optional[float]]],
    ) -> List[TaskOutcome]:
        """串行执行 ``tasks``（用于对比耗时基准）。"""
        tasks = list(tasks)
        if not tasks:
            self.history = 0.0
            return []

        start = time.perf_counter()
        outcomes: List[TaskOutcome] = []
        for idx, (fn, _timeout) in enumerate(tasks):
            t0 = time.perf_counter()
            outcome = TaskOutcome(index=idx)
            try:
                outcome.result = fn()
                outcome.success = True
            except BaseException as exc:  # noqa: BLE001 - 串行不外泄异常
                outcome.success = False
                outcome.error = exc
            finally:
                outcome.elapsed = time.perf_counter() - t0
            outcomes.append(outcome)

        self.history = time.perf_counter() - start
        return outcomes

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _run_one(self, idx: int, fn: Callable[[], Any]) -> Any:
        """在线程中执行单个任务（成功返回结果，异常通过 Future 传播）。"""
        return fn()

    def _record(
        self,
        outcomes: List[Optional[TaskOutcome]],
        idx: int,
        future: Future,
        timeout: Optional[float],
    ) -> None:
        """等待单个 future 并解析为 TaskOutcome（超时降级，不抛出）。"""
        t0 = time.perf_counter()
        outcome = TaskOutcome(index=idx)
        try:
            outcome.result = future.result(timeout=timeout)
            outcome.success = True
        except FutureTimeout:
            outcome.success = False
            outcome.timed_out = True
        except BaseException as exc:  # noqa: BLE001 - 捕获任务异常不外泄
            outcome.success = False
            outcome.error = exc
        finally:
            outcome.elapsed = time.perf_counter() - t0
        with self._result_lock:
            outcomes[idx] = outcome


# ============================================================================
# DelegationRequest —— 一次委派请求的定义
# ============================================================================


@dataclass
class DelegationRequest:
    """一次委派的完整输入。

    Attributes:
        spec:             目标 Agent。
        goal:             子 Agent 目标。
        success_criteria: 成功判定标准（透传给 Runtime._run_delegation）。
    """

    spec: AgentSpec
    goal: str
    success_criteria: str = ""


def delegate_parallel(
    runtime: Any,
    parent_ctx: Any,
    requests: Sequence[DelegationRequest],
    max_workers: int = 4,
    per_task_timeout: Optional[float] = None,
) -> List[TaskOutcome]:
    """对一组委派请求并行调用 ``runtime._run_delegation``。

    约束：
        - 不修改 Agent Loop / Runtime，只复用 Runtime 现成的委派执行逻辑。
        - Runtime 的委派深度 / active_ctx 均为线程局部，并行不互相污染。
        - 每个任务有超时；单个委派失败不影响其他委派。

    Args:
        runtime:          Runtime 实例（需有 ``_run_delegation`` 方法）。
        parent_ctx:       父 Context（子 Agent 的 sandbox / shared_state 继承来源）。
        requests:         委派请求序列。
        max_workers:      并发度。
        per_task_timeout: 每个委派的超时（None 表示不设超时）。

    Returns:
        ``TaskOutcome`` 列表，与 ``requests`` 顺序一致。
    """
    scheduler = ConcurrentScheduler(max_workers=max_workers)

    def make_task(req: DelegationRequest) -> Callable[[], Any]:
        def _task() -> Any:
            return runtime._run_delegation(
                parent_ctx, req.spec, req.goal, req.success_criteria
            )
        return _task

    tasks = [(make_task(req), per_task_timeout) for req in requests]
    return scheduler.run_parallel(tasks)


__all__ = [
    "TaskOutcome",
    "ConcurrentScheduler",
    "DelegationRequest",
    "delegate_parallel",
]