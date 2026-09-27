"""
[L3-2] MemoryLifecycleMiddleware — 记忆生命周期中间件。

职责边界（只做生命周期驱动，不做注入、不做压缩、不做预算判定）：
    - AFTER_ITERATION 钩子：每 N 轮调用一次 ContextManager.extract_memories()，
      触发 L5 记忆沉淀。默认 N=5，避免每轮都跑 LLM 抽取。
    - ON_EXIT_LOOP 钩子：调用 ContextManager.apply_decay()，触发 L2 衰减并归档
      强度过低的记忆。
    - 不写事件日志、不修改消息、不拦截工具、不做安全判定。

为什么需要本中间件：
    LayeredContextManager.extract_memories() / apply_decay() 虽然已实现，
    但没有任何中间件在运行期调用它们，实际运行中永远不执行。
    本中间件补齐"钩子 → ContextManager 方法"的驱动链路：

        AFTER_ITERATION (每 N 轮) → extract_memories()   # L5 沉淀
        ON_EXIT_LOOP              → apply_decay()        # L2 衰减 + 归档

注册钩子：
    - ON_ENTER_LOOP   : 复位 per-run 迭代计数（D009）
    - AFTER_ITERATION : 计数达到 N 的整数倍时触发 extract_memories
    - ON_EXIT_LOOP    : 触发 apply_decay，记录归档数量

[D009] 迭代计数为何放在 ctx.private：
    早期实现把 `_iteration` 作为中间件实例字段。中间件实例可跨多次 run 复用
    （Runtime.register_middleware 只注册一次），实例字段在第二个 run 不会复位，
    于是 `_iteration % extract_interval` 的整除判定被历史 run 的轮数污染——
    部分 run 会漏触发 extract_memories，统计口径随复用次数漂移。
    现在计数写入 `ctx.private[ITERATION_KEY]`：Context 每次 run 由 Runtime 新建，
    计数天然是 per-run 状态；同时在 ON_ENTER_LOOP 显式复位，使"同一 ctx 被复用
    跑多次 run"的场景也能正确从 0 开始。中间件实例本身保持无状态。
    实例上的 `_iteration` 仅作为"最近一次 run 的计数镜像"保留供观测，
    不参与任何逻辑判定。

依赖关系：
    MemoryLifecycleMiddleware -> ContextManager（公开方法 extract_memories / apply_decay）
    通过构造参数注入，或从 ctx.shared["context_manager"] 解析。

ctx.shared 约定键（写入）：
    "memory_lifecycle" :
        {
            "extract_calls":      extract_memories() 的调用次数,
            "extracted_traces":   累计沉淀的记忆条数,
            "decay_calls":        apply_decay() 的调用次数,
            "archived_traces":    累计归档的记忆条数,
            "last_extracted":     最近一次沉淀的条数,
            "last_archived":      最近一次归档的条数,
            "namespace":          命名空间,
        }

可独立启停：
    未提供 ContextManager 时所有钩子直接放行，不抛异常、不写入 shared。

[L3-2] 本文件为 L3 第二批中间件新增，未修改任何 L1 / L2 代码。
"""

from __future__ import annotations

from typing import Any, Optional

from agent_loop import Context, HookResult, Middleware


class MemoryLifecycleMiddleware(Middleware):
    """[L3-2] 记忆生命周期中间件。

    Attributes:
        extract_interval: 每多少轮迭代触发一次记忆沉淀（默认 5）。
        namespace:        长期记忆命名空间（None 时从 ctx.shared 解析）。
    """

    #: [D009] per-run 迭代计数在 ctx.private 中的键
    ITERATION_KEY = "_memory_lifecycle_iteration"

    def __init__(
        self,
        context_manager: Any = None,
        extract_interval: int = 5,
        namespace: Optional[str] = None,
        name: str = "memory_lifecycle",
    ) -> None:
        """
        Args:
            context_manager:  ContextManager 实例（需实现 extract_memories / apply_decay）。
            extract_interval: 每多少轮迭代触发一次记忆沉淀，默认 5。
                              <=0 时视为每轮都触发（退化为 interval=1）。
            namespace:        长期记忆命名空间。None 时从 ctx.shared["memory_namespace"] 解析。
            name:             中间件名称。
        """
        super().__init__(name)
        self._context_manager = context_manager
        # 防御：非正数退化为每轮触发，避免因配置失误导致沉淀永不执行
        self._extract_interval = extract_interval if extract_interval > 0 else 1
        self._namespace = namespace

        # [D009] 计数真值存放在 ctx.private（per-run 状态），实例不再持有计数器。
        # 此字段仅为"最近一次 run 的计数镜像"，供观测与断言使用，逻辑不读取它。
        self._iteration = 0

    @property
    def extract_interval(self) -> int:
        """当前生效的沉淀间隔。"""
        return self._extract_interval

    # ------------------------------------------------------------------
    # ON_ENTER_LOOP —— 复位 per-run 迭代计数（D009）
    # ------------------------------------------------------------------

    def on_enter_loop(self, ctx: Context) -> HookResult:
        """[D009] 每次 run 开始时把 per-run 迭代计数复位为 0。"""
        ctx.private[self.ITERATION_KEY] = 0
        self._iteration = 0
        return HookResult.continue_()

    # ------------------------------------------------------------------
    # AFTER_ITERATION —— 每 N 轮触发一次记忆沉淀（L5）
    # ------------------------------------------------------------------

    def after_iteration(self, ctx: Context) -> HookResult:
        """[L3-2] AFTER_ITERATION：每 N 轮调用一次 extract_memories()。

        [D009] 计数从 ctx.private 读取/写回，因此天然 per-run；
        中间件实例跨 run 复用不会让计数漂移。
        """
        iteration = int(ctx.private.get(self.ITERATION_KEY, 0)) + 1
        ctx.private[self.ITERATION_KEY] = iteration
        self._iteration = iteration  # 观测镜像

        # 未达到间隔整数倍：本轮不抽取，避免每轮跑 LLM
        if iteration % self._extract_interval != 0:
            return HookResult.continue_()

        cm = self._resolve_context_manager(ctx)
        if cm is None:
            # 无 ContextManager：静默放行，可独立启停
            return HookResult.continue_()

        namespace = self._resolve_namespace(ctx)
        extracted_count = self._call_extract(cm, namespace)

        state = self._state(ctx)
        state["extract_calls"] += 1
        state["last_extracted"] = extracted_count
        state["extracted_traces"] += extracted_count

        return HookResult.continue_()

    # ------------------------------------------------------------------
    # ON_EXIT_LOOP —— 触发记忆衰减（L2）
    # ------------------------------------------------------------------

    def on_exit_loop(self, ctx: Context) -> HookResult:
        """[L3-2] ON_EXIT_LOOP：调用 apply_decay() 并归档衰减过强的记忆。"""
        cm = self._resolve_context_manager(ctx)
        if cm is None:
            return HookResult.continue_()

        namespace = self._resolve_namespace(ctx)
        archived_count = self._call_decay(cm, namespace)

        state = self._state(ctx)
        state["decay_calls"] += 1
        state["last_archived"] = archived_count
        state["archived_traces"] += archived_count

        return HookResult.continue_()

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _resolve_context_manager(self, ctx: Context) -> Any:
        """[L3-2] 解析 ContextManager（构造注入 > ctx.shared["context_manager"]）。"""
        if self._context_manager is not None:
            return self._context_manager
        shared = getattr(ctx, "shared", None)
        if isinstance(shared, dict):
            return shared.get("context_manager")
        return None

    def _resolve_namespace(self, ctx: Context) -> Optional[str]:
        """[L3-2] 解析命名空间（构造参数 > ctx.shared["memory_namespace"]）。"""
        if self._namespace:
            return self._namespace
        shared = getattr(ctx, "shared", None)
        if isinstance(shared, dict):
            ns = shared.get("memory_namespace")
            if ns:
                return str(ns)
        return None

    def _state(self, ctx: Context) -> dict:
        """[L3-2] 获取（或初始化）ctx.shared["memory_lifecycle"] 统计字典。"""
        shared = ctx.shared
        state = shared.get("memory_lifecycle")
        if not isinstance(state, dict):
            state = {
                "extract_calls": 0,
                "extracted_traces": 0,
                "decay_calls": 0,
                "archived_traces": 0,
                "last_extracted": 0,
                "last_archived": 0,
                "namespace": self._resolve_namespace(ctx),
            }
            shared["memory_lifecycle"] = state
        return state

    def _call_extract(self, cm: Any, namespace: Optional[str]) -> int:
        """[L3-2] 调用 extract_memories，容错返回沉淀条数。"""
        extractor = getattr(cm, "extract_memories", None)
        if not callable(extractor):
            return 0
        try:
            traces = extractor(namespace=namespace)
        except TypeError:
            # 兼容只接受位置参数的实现
            try:
                traces = extractor(namespace)
            except Exception:
                return 0
        except Exception:
            return 0
        if isinstance(traces, list):
            return len(traces)
        return 0

    def _call_decay(self, cm: Any, namespace: Optional[str]) -> int:
        """[L3-2] 调用 apply_decay，容错返回归档条数。"""
        decay = getattr(cm, "apply_decay", None)
        if not callable(decay):
            return 0
        try:
            archived = decay(namespace=namespace)
        except TypeError:
            try:
                archived = decay(namespace)
            except Exception:
                return 0
        except Exception:
            return 0
        try:
            return int(archived or 0)
        except (TypeError, ValueError):
            return 0