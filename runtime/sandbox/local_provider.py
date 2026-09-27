"""
[L3] LocalSandboxProvider — 本地沙箱提供者。

不隔离进程，只在受控 workdir 下执行。适用于单机开发与测试。

关键语义：
    - translate_path 强制将路径约束在 workdir 内，越界抛 ValueError。
    - 工具执行复用 InMemoryToolRegistry._execute_with_timeout（线程池 + 超时隔离）。
"""

from __future__ import annotations

import os
import tempfile
import uuid
from typing import Any, Dict

from runtime.sandbox.interface import SandboxHandle, SandboxProvider
from runtime.tool_registry.in_memory import _execute_with_timeout
from runtime.tool_registry.interface import ToolResult


class LocalSandboxProvider(SandboxProvider):
    """本地沙箱：受控 workdir 下执行，不隔离进程。"""

    def __init__(self, base_dir: str | None = None, registry: Any = None) -> None:
        """
        Args:
            base_dir: 沙箱根目录，None 时使用系统临时目录。
            registry: 用于 resolve 工具并取 fn 的 InMemoryToolRegistry。None 时
                      execute 无法执行需通过 registry 解析的工具（保留扩展）。
        """
        self._base_dir = base_dir or tempfile.gettempdir()
        self._registry = registry
        self._handles: Dict[str, SandboxHandle] = {}

    # ------------------------------------------------------------------
    # Provider 契约
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return True

    def acquire(self, agent_id: str) -> SandboxHandle:
        """为 agent 创建一个独立 workdir 沙箱。"""
        safe_id = re_safe(agent_id)
        workdir = os.path.join(self._base_dir, f"sandbox-{safe_id}-{uuid.uuid4().hex[:8]}")
        os.makedirs(workdir, exist_ok=True)
        handle = SandboxHandle(
            sandbox_id=workdir,
            workdir=workdir,
            env=dict(os.environ),
            metadata={"provider": "local"},
        )
        self._handles[handle.sandbox_id] = handle
        return handle

    def release(self, handle: SandboxHandle) -> None:
        """释放沙箱（本地实现仅清空注册表，不删除 workdir 以免误删产物）。"""
        self._handles.pop(handle.sandbox_id, None)

    def translate_path(self, handle: SandboxHandle, path: str) -> str:
        """把外部路径翻译到沙箱内绝对路径。

        强制路径必须在 workdir 内：规范化后若不在 workdir 前缀下则抛 ValueError。
        """
        workdir = os.path.abspath(handle.workdir)
        candidate = os.path.abspath(path) if os.path.isabs(path) else os.path.abspath(
            os.path.join(handle.workdir, path)
        )
        if candidate != workdir and not candidate.startswith(workdir + os.sep):
            raise ValueError(
                f"路径越界: {path!r} 不在沙箱 workdir 内 ({handle.workdir!r})"
            )
        return candidate

    def execute(
        self, handle: SandboxHandle, tool_name: str, args: Dict[str, Any]
    ) -> ToolResult:
        """在沙箱 workdir 内执行工具（复用 registry 的线程池超时执行）。"""
        fn = self._resolve_fn(tool_name)
        if fn is None:
            return ToolResult.from_error(f"沙箱内未找到工具 '{tool_name}'")

        # 切换当前工作目录到沙箱 workdir，执行后还原
        prev_cwd = os.getcwd()
        os.chdir(handle.workdir)
        try:
            return _execute_with_timeout(fn, args or {}, None)
        finally:
            os.chdir(prev_cwd)

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _resolve_fn(self, tool_name: str):
        """从 registry 解析工具的执行函数。"""
        if self._registry is None:
            return None
        tool = self._registry.get(tool_name)
        if tool is None:
            return None
        return tool.fn


def re_safe(value: str) -> str:
    """把字符串清洗为安全的路径片段。"""
    import re as _re
    return _re.sub(r"[^A-Za-z0-9_-]", "_", value or "agent")