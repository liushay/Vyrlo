"""
[L3] MemoryConsolidationMiddleware — 记忆合并中间件。

职责边界（只驱动 episodic→semantic 合并，不做注入、不做压缩、不做预算）：
    - ON_EXIT_LOOP 钩子：调用 ContextManager._consolidate_episodic(namespace)，
      把累积的情景记忆合并为语义记忆。
    - 不写事件日志、不修改消息、不拦截工具、不做安全判定。
    - 恒返回 continue_()。

为什么需要本中间件：
    LayeredContextManager._consolidate_episodic() 虽然已实现，
    但没有中间件在运行期驱动它。本中间件补齐"ON_EXIT_LOOP → 合并"的驱动链路。

注册钩子：
    - ON_EXIT_LOOP : 调用 _consolidate_episodic

依赖关系：
    MemoryConsolidationMiddleware -> ContextManager（_consolidate_episodic）
    通过构造参数注入，或从 ctx.shared["context_manager"] 解析。

ctx.shared 约定键（写入）：
    "memory_consolidation" :
        {
            "consolidate_calls":     调用次数,
            "consolidated_traces":   累计生成的 semantic 条数,
            "last_consolidated":     最近一次合并条数,
            "namespace":             命名空间,
        }

可独立启停：
    未提供 ContextManager 或未注入 llm_callable 时放行，不抛异常、不写入 shared。

[L3] 本文件为 L3 中间件新增，未修改任何 L1 / L2 代码。
"""

from __future__ import annotations

from typing import Any, Optional

from agent_loop import Context, HookResult, Middleware


class MemoryConsolidationMiddleware(Middleware):
    """[L3] 记忆合并中间件。

    Attributes:
        namespace: 长期记忆命名空间（None 时从 ctx.shared 解析）。
    """

    def __init__(
        self,
        context_manager: Any = None,
        namespace: Optional[str] = None,
        name: str = "memory_consolidation",
    ) -> None:
        """
        Args:
            context_manager: ContextManager 实例（需实现 _consolidate_episodic）。
            namespace:       长期记忆命名空间。None 时从 ctx.shared["memory_namespace"] 解析。
            name:            中间件名称。
        """
        super().__init__(name)
        self._context_manager = context_manager
        self._namespace = namespace

    # ------------------------------------------------------------------
    # ON_EXIT_LOOP —— 触发 episodic→semantic 合并
    # ------------------------------------------------------------------

    def on_exit_loop(self, ctx: Context) -> HookResult:
        """[L3] ON_EXIT_LOOP：调用 ContextManager._consolidate_episodic。"""
        cm = self._resolve_context_manager(ctx)
        namespace = self._resolve_namespace(ctx)

        if cm is None:
            return HookResult.continue_()

        consolidator = getattr(cm, "_consolidate_episodic", None)
        if not callable(consolidator):
            return HookResult.continue_()

        try:
            result = consolidator(namespace=namespace)
        except TypeError:
            try:
                result = consolidator(namespace)
            except Exception:
                result = 0
        except Exception:
            result = 0

        try:
            count = int(result or 0)
        except (TypeError, ValueError):
            count = 0

        state = self._state(ctx)
        state["consolidate_calls"] += 1
        state["last_consolidated"] = count
        state["consolidated_traces"] += count

        return HookResult.continue_()

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _resolve_context_manager(self, ctx: Context) -> Any:
        """[L3] 解析 ContextManager（构造注入 > ctx.shared["context_manager"]）。"""
        if self._context_manager is not None:
            return self._context_manager
        shared = getattr(ctx, "shared", None)
        if isinstance(shared, dict):
            return shared.get("context_manager")
        return None

    def _resolve_namespace(self, ctx: Context) -> Optional[str]:
        """[L3] 解析命名空间（构造参数 > ctx.shared["memory_namespace"]）。"""
        if self._namespace:
            return self._namespace
        shared = getattr(ctx, "shared", None)
        if isinstance(shared, dict):
            ns = shared.get("memory_namespace")
            if ns:
                return str(ns)
        return None

    def _state(self, ctx: Context) -> dict:
        """[L3] 获取（或初始化）ctx.shared["memory_consolidation"] 统计字典。"""
        shared = ctx.shared
        state = shared.get("memory_consolidation")
        if not isinstance(state, dict):
            state = {
                "consolidate_calls": 0,
                "consolidated_traces": 0,
                "last_consolidated": 0,
                "namespace": self._resolve_namespace(ctx),
            }
            shared["memory_consolidation"] = state
        return state