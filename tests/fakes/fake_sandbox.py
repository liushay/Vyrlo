"""
[测试] FakeSandboxProvider — 可控可用性 / 路径越界的沙箱。

实现 SandboxProvider 契约，支持：
    - available: 控制 is_available() 返回值（模拟 Docker 不可用）。
    - 每次 acquire 分配递增的 sandbox_id 并记录 release 调用。
    - 可通过 ``raise_on_translate`` 强制 translate_path 抛越界异常。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from runtime.sandbox.interface import SandboxHandle, SandboxProvider
from runtime.tool_registry.interface import ToolResult


class FakeSandboxProvider(SandboxProvider):
    """可控沙箱提供者。"""

    def __init__(self, available: bool = True) -> None:
        self.available = available
        self._counter = 0
        self.acquired: List[SandboxHandle] = []
        self.released: List[str] = []
        #: 为 True 时 translate_path 抛 ValueError（模拟路径越界）
        self.raise_on_translate: bool = False
        #: execute 时调用的回调（可返回 ToolResult）；默认成功。
        self._executed: List[Dict[str, Any]] = []

    def is_available(self) -> bool:
        return self.available

    def acquire(self, agent_id: str) -> SandboxHandle:
        self._counter += 1
        handle = SandboxHandle(
            sandbox_id=f"sb-{self._counter}",
            workdir=f"/tmp/fake-sandbox-{self._counter}",
            env={},
            metadata={"provider": "fake"},
        )
        self.acquired.append(handle)
        return handle

    def release(self, handle: SandboxHandle) -> None:
        self.released.append(handle.sandbox_id)

    def execute(
        self, handle: SandboxHandle, tool_name: str, args: Dict[str, Any]
    ) -> ToolResult:
        self._executed.append({"tool": tool_name, "args": args, "sandbox": handle.sandbox_id})
        return ToolResult.from_result(
            {"sandbox": handle.sandbox_id, "tool": tool_name},
            metadata={"elapsed_seconds": 0.001},
        )

    def translate_path(self, handle: SandboxHandle, path: str) -> str:
        if self.raise_on_translate:
            raise ValueError(f"路径越界: {path!r} 不在沙箱 workdir 内")
        return path


__all__ = ["FakeSandboxProvider"]