"""
[L3-1] SafetyGuardMiddleware — 安全守卫中间件。

职责边界（只做工具准入判定，不执行工具、不记录日志）：
    - 在 BEFORE_TOOL 钩子中读取工具的准入信息：
        * registry 中 Tool.requires_approval（需要人工审批）
        * registry 中 Tool.metadata（如 {"dangerous": True, "risk_level": "high"}）
    - 高危工具返回 HookResult.skip_tool()（跳过本次调用）；
    - 普通工具返回 HookResult.continue_() 放行。
    - 白名单（allowed_tools）优先级最高：白名单内工具直接放行，忽略高危判定。
    - 不修改消息、不做预算判定、不写事件日志。

工具元数据来源（按优先级）：
    1. 构造函数注入的 tool_registry 的 get(name)
    2. ctx.shared["tool_registry"] 的 get(name)
    3. ctx.shared["current_tool_meta"]（由其它中间件预先注入的元数据）
    4. tool_call 自身携带的 metadata

白名单配置：
    通过构造函数 allowed_tools 传入（可迭代的工具名集合），
    或通过 ctx.shared["tool_whitelist"] 在运行时提供。

注册钩子：
    - BEFORE_TOOL : 高危工具 skip_tool，普通工具放行

依赖关系：
    SafetyGuardMiddleware -> runtime.tool_registry（只读 Tool.requires_approval / metadata）
    不依赖 EventLog / Session / ContextManager。

[L3-1] 本文件为 L3 第一批中间件新增，未修改任何 L1 / L2 代码。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from agent_loop import Context, HookResult, Middleware


class SafetyGuardMiddleware(Middleware):
    """[L3-1] 安全守卫中间件。

    Attributes:
        blocked: 已被拦截的工具记录列表（供断言与审计）。
    """

    #: 视为高危的风险等级取值
    HIGH_RISK_LEVELS = frozenset({"high", "critical", "severe"})

    def __init__(
        self,
        allowed_tools: Optional[Iterable[str]] = None,
        tool_registry: Any = None,
        name: str = "safety_guard",
    ) -> None:
        """
        Args:
            allowed_tools: 白名单工具名集合。白名单内工具一律放行。
            tool_registry: ToolRegistry 实例（需实现 get(name) -> Tool | None）。
            name:          中间件名称。
        """
        super().__init__(name)
        self._allowed_tools: set = set(allowed_tools or ())
        self._tool_registry = tool_registry

        self.blocked: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # BEFORE_TOOL —— 高危工具拦截
    # ------------------------------------------------------------------

    def before_tool(self, ctx: Context, tool_call: Dict[str, Any]) -> HookResult:
        """[L3-1] BEFORE_TOOL：高危工具跳过，白名单/普通工具放行。"""
        tool_name = self._tool_name(tool_call)
        if not tool_name:
            return HookResult.continue_()

        # 1) 白名单优先：直接放行
        if tool_name in self._effective_whitelist(ctx):
            return HookResult.continue_()

        # 2) 收集工具元数据
        meta = self._collect_meta(ctx, tool_call)
        requires_approval = bool(meta.get("requires_approval", False))
        tool_metadata = meta.get("metadata") or {}
        if not isinstance(tool_metadata, dict):
            tool_metadata = {}

        # 3) 高危判定
        reason = self._risk_reason(requires_approval, tool_metadata)
        if reason is None:
            return HookResult.continue_()

        # 4) 拦截：跳过当前工具
        self.blocked.append({"tool": tool_name, "reason": reason})
        ctx.shared.setdefault("safety_blocked", []).append(
            {"tool": tool_name, "reason": reason}
        )
        return HookResult.skip_tool()

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _effective_whitelist(self, ctx: Context) -> set:
        """[L3-1] 合并构造参数白名单与运行时白名单。"""
        allowed = set(self._allowed_tools)
        shared = getattr(ctx, "shared", None)
        if isinstance(shared, dict):
            runtime_whitelist = shared.get("tool_whitelist")
            if runtime_whitelist:
                try:
                    allowed.update(str(t) for t in runtime_whitelist)
                except TypeError:
                    pass
        return allowed

    @classmethod
    def _risk_reason(
        cls, requires_approval: bool, tool_metadata: Dict[str, Any]
    ) -> Optional[str]:
        """[L3-1] 判定工具是否高危，高危返回原因，否则返回 None。"""
        if requires_approval:
            return "requires_approval=True（需要人工审批）"

        if tool_metadata.get("dangerous") is True:
            return "metadata.dangerous=True"

        risk_level = str(tool_metadata.get("risk_level", "")).lower()
        if risk_level in cls.HIGH_RISK_LEVELS:
            return f"metadata.risk_level={risk_level}"

        if tool_metadata.get("requires_approval") is True:
            return "metadata.requires_approval=True"

        return None

    def _collect_meta(
        self, ctx: Context, tool_call: Dict[str, Any]
    ) -> Dict[str, Any]:
        """[L3-1] 收集工具准入所需的元数据。"""
        tool_name = self._tool_name(tool_call)

        # 来源 1/2：registry
        registry = self._tool_registry
        if registry is None:
            shared = getattr(ctx, "shared", None)
            if isinstance(shared, dict):
                registry = shared.get("tool_registry")

        if registry is not None:
            tool = None
            getter = getattr(registry, "get", None)
            if callable(getter):
                try:
                    tool = getter(tool_name)
                except Exception:
                    tool = None
            if tool is not None:
                return {
                    "requires_approval": getattr(tool, "requires_approval", False),
                    "metadata": getattr(tool, "metadata", {}) or {},
                }

        # 来源 3：其它中间件注入的元数据
        shared = getattr(ctx, "shared", None)
        if isinstance(shared, dict):
            injected = shared.get("current_tool_meta")
            if isinstance(injected, dict) and injected.get("name") in (None, tool_name):
                return {
                    "requires_approval": injected.get("requires_approval", False),
                    "metadata": injected.get("metadata", {}) or {},
                }

        # 来源 4：tool_call 自身携带
        if isinstance(tool_call, dict):
            return {
                "requires_approval": tool_call.get("requires_approval", False),
                "metadata": tool_call.get("metadata", {}) or {},
            }
        return {}

    @staticmethod
    def _tool_name(tool_call: Any) -> str:
        """[L3-1] 提取工具名（兼容 dict 与对象）。"""
        if isinstance(tool_call, dict):
            return str(tool_call.get("name", "") or "")
        return str(getattr(tool_call, "name", "") or "")