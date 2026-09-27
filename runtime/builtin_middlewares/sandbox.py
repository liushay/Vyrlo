"""
[L3] SandboxMiddleware — 沙箱中间件。

职责边界（只做沙箱生命周期管理，不修改消息、不做工具判定）：
    - ON_ENTER_LOOP：从 ctx.shared["session"].metadata 读取 agent_id，
      acquire 沙箱，写入 ctx.shared["sandbox"]；注册清理函数。
    - BEFORE_TOOL：把当前沙箱 id 写入 ctx.shared["current_sandbox_id"]，
      供 Runtime._tool_executor 读取。
    - ON_EXIT_LOOP：释放沙箱（release）。

依赖关系：
    SandboxMiddleware -> SandboxRegistry（acquire / release）
    通过构造参数注入。

子 Agent 共享沙箱：
    ctx.shared 已存在 "sandbox" 时（父级已 acquire），子 Agent 不再重新 acquire，
    直接继承父级 handle；同时通过 ContextVar 传递 sandbox_id 避免跨线程丢失。

可独立启停：
    未提供 SandboxRegistry 或无可用 provider 时放行，不抛异常。
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Optional

from agent_loop import Context, HookResult, Middleware

from runtime.sandbox.registry import SandboxRegistry

#: 跨线程传递的当前沙箱 id（为后续子 Agent 共享预留）
_current_sandbox_id: ContextVar = ContextVar("current_sandbox_id", default=None)


class SandboxMiddleware(Middleware):
    """[L3] 沙箱中间件 —— 管理沙箱的 acquire / release。"""

    def __init__(
        self,
        sandbox_registry: Optional[SandboxRegistry] = None,
        default_mode: str = "local",
        name: str = "sandbox",
    ) -> None:
        """
        Args:
            sandbox_registry: 沙箱注册表。为 None 时不启用沙箱（放行）。
            default_mode:     默认沙箱模式（"local" / "docker"）。
            name:             中间件名称。
        """
        super().__init__(name)
        self._registry = sandbox_registry
        self._default_mode = default_mode
        # 当前 run 由本中间件主动 acquire 的 handle（用于 ON_EXIT_LOOP 释放）
        self._owned_handle = None
        self._released = False

    # ------------------------------------------------------------------
    # ON_ENTER_LOOP —— acquire 沙箱
    # ------------------------------------------------------------------

    def on_enter_loop(self, ctx: Context) -> HookResult:
        """[L3] 为本次运行获取沙箱，写入 ctx.shared["sandbox"]。"""
        if self._registry is None:
            return HookResult.continue_()

        # 子 Agent 继承父级沙箱：已存在则不重新 acquire
        shared = ctx.shared
        existing = shared.get("sandbox")
        if existing is not None:
            self._owned_handle = None
            self._released = False
            self._set_sandbox_id(getattr(existing, "sandbox_id", None))
            return HookResult.continue_()

        agent_id = self._resolve_agent_id(ctx)
        handle = self._registry.acquire(agent_id, self._default_mode)
        if handle is None:
            # 无可用 provider，放行（沙箱可不启用）
            return HookResult.continue_()

        shared["sandbox"] = handle
        shared["current_sandbox_id"] = handle.sandbox_id
        self._owned_handle = handle
        self._released = False
        self._set_sandbox_id(handle.sandbox_id)

        # 注册清理（异常退出 / 注销中间件时释放）
        def _cleanup() -> None:
            self._release(handle)

        self.register_effect(_cleanup)
        return HookResult.continue_()

    # ------------------------------------------------------------------
    # BEFORE_TOOL —— 记录当前沙箱 id
    # ------------------------------------------------------------------

    def before_tool(self, ctx: Context, tool_call: Any) -> HookResult:
        """[L3] 把当前沙箱 id 写入 ctx.shared["current_sandbox_id"]。"""
        handle = ctx.shared.get("sandbox")
        if handle is not None:
            ctx.shared["current_sandbox_id"] = getattr(handle, "sandbox_id", None)
            self._set_sandbox_id(getattr(handle, "sandbox_id", None))
        return HookResult.continue_()

    # ------------------------------------------------------------------
    # ON_EXIT_LOOP —— 释放沙箱
    # ------------------------------------------------------------------

    def on_exit_loop(self, ctx: Context) -> HookResult:
        """[L3] 正常退出时释放由本中间件 acquire 的沙箱。"""
        if self._owned_handle is not None and not self._released:
            self._release(self._owned_handle)
        return HookResult.continue_()

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _release(self, handle: Any) -> None:
        """释放句柄（幂等，异常吞掉不阻断）。"""
        if self._released:
            return
        try:
            self._registry.release(handle, self._default_mode)
        except Exception:
            pass
        self._released = True
        self._owned_handle = None

    def _resolve_agent_id(self, ctx: Context) -> str:
        """[L3] 从 ctx.shared["session"].metadata 读取 agent_id。"""
        shared = ctx.shared
        session = shared.get("session")
        metadata = getattr(session, "metadata", None) if session is not None else None
        if isinstance(metadata, dict):
            agent_id = metadata.get("agent_id")
            if agent_id:
                return str(agent_id)
        return shared.get("agent_id") or "default-agent"

    def _set_sandbox_id(self, sandbox_id: Optional[str]) -> None:
        """[L3] 设置 ContextVar，供跨线程传递。"""
        try:
            _current_sandbox_id.set(sandbox_id)
        except Exception:
            pass