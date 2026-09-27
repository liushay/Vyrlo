"""
[L4] DelegationMiddleware — 委派拦截中间件。

职责边界（只做委派意图识别与拦截，不执行委派）：
    - BEFORE_TOOL 拦截 `delegate_to_agent` 工具调用。
    - 从 AgentRegistry 查找目标 AgentSpec：
        * 查不到 → 返回 skip_tool，并在 ctx.shared["delegation_rejected"] 记录原因。
        * 查到   → 写入 ctx.shared["pending_delegation"]，返回 skip_tool。
    - 实际委派由 Runtime 层在 run 循环中检测 pending_delegation 后递归执行。

依赖关系：
    DelegationMiddleware -> AgentRegistry（只读 get）
    通过构造参数注入，或从 ctx.shared["agent_registry"] 解析。

可独立启停：
    未提供 AgentRegistry 时不拦截 delegate_to_agent，而是放行（这样无注册表时
    该工具等同普通工具，不会被误拦）。

[L4] 本文件为多 Agent 编排新增，未修改任何 L1 / L2 代码。
"""

from __future__ import annotations

from typing import Any, Optional

from agent_loop import Context, HookResult, Middleware

from runtime.agent_registry.interface import AgentRegistry

DELEGATE_TOOL = "delegate_to_agent"


class DelegationMiddleware(Middleware):
    """[L4] 委派拦截中间件。"""

    def __init__(
        self,
        agent_registry: Optional[AgentRegistry] = None,
        name: str = "delegation",
    ) -> None:
        """
        Args:
            agent_registry: Agent 注册表。None 时从 ctx.shared["agent_registry"] 解析。
            name:           中间件名称。
        """
        super().__init__(name)
        self._agent_registry = agent_registry

    def before_tool(self, ctx: Context, tool_call: Any) -> HookResult:
        """[L4] BEFORE_TOOL：拦截 delegate_to_agent 调用并转为 pending_delegation。"""
        tool_name = self._tool_name(tool_call)
        if tool_name != DELEGATE_TOOL:
            return HookResult.continue_()

        registry = self._resolve_registry(ctx)
        if registry is None:
            # 无注册表：不拦截，退化为普通工具（skip 会误伤）
            return HookResult.continue_()

        args = self._tool_args(tool_call)
        target_agent = str(args.get("target_agent", ""))
        goal = str(args.get("goal", ""))
        success_criteria = str(args.get("success_criteria", ""))

        spec = registry.get(target_agent)
        if spec is None:
            ctx.shared.setdefault("delegation_rejected", []).append({
                "target_agent": target_agent,
                "reason": "unknown_agent",
            })
            return HookResult.skip_tool()

        ctx.shared["pending_delegation"] = {
            "target_agent": target_agent,
            "goal": goal,
            "success_criteria": success_criteria,
            "spec": spec,
        }
        return HookResult.skip_tool()

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _resolve_registry(self, ctx: Context) -> Optional[AgentRegistry]:
        if self._agent_registry is not None:
            return self._agent_registry
        shared = getattr(ctx, "shared", None)
        if isinstance(shared, dict):
            registry = shared.get("agent_registry")
            if registry is not None:
                return registry
        return None

    @staticmethod
    def _tool_name(tool_call: Any) -> str:
        if isinstance(tool_call, dict):
            return str(tool_call.get("name", ""))
        return str(getattr(tool_call, "name", ""))

    @staticmethod
    def _tool_args(tool_call: Any) -> dict:
        if isinstance(tool_call, dict):
            args = tool_call.get("arguments", tool_call.get("args", {}))
            return dict(args) if isinstance(args, dict) else {}
        args = getattr(tool_call, "arguments", getattr(tool_call, "args", {}))
        return dict(args) if isinstance(args, dict) else {}