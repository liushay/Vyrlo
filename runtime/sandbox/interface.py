"""
[L3] runtime.sandbox — 沙箱执行抽象。

定义工具在受控环境中执行所需的 Provider 契约与 Handle 数据模型。

职责边界：
    - 只定义接口与数据模型，不含具体执行逻辑。
    - 不依赖 Agent Loop / ToolRegistry 的具体实现（仅依赖 ToolResult 数据模型）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from runtime.tool_registry.interface import ToolResult


@dataclass
class SandboxHandle:
    """沙箱实例句柄。

    Attributes:
        sandbox_id: 沙箱唯一标识（本地 provider 时为 workdir 路径；docker 时为容器 ID）。
        workdir:    沙箱内的工作目录。
        env:        沙箱环境变量。
        metadata:   额外元数据（如 provider 名、容器名等）。
    """

    sandbox_id: str
    workdir: str
    env: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SandboxProvider(ABC):
    """沙箱提供者抽象接口。

    契约：
        - acquire(agent_id) 获取一个沙箱句柄。
        - release(handle) 释放沙箱。
        - execute(handle, tool_name, args) 在沙箱内执行工具。
        - translate_path(handle, path) 把外部路径翻译到沙箱内路径。
        - is_available() 判断当前环境是否可用（不可用时不应阻断装配）。
    """

    @abstractmethod
    def acquire(self, agent_id: str) -> SandboxHandle:
        """获取一个沙箱句柄。"""
        ...

    @abstractmethod
    def release(self, handle: SandboxHandle) -> None:
        """释放沙箱。"""
        ...

    @abstractmethod
    def execute(
        self, handle: SandboxHandle, tool_name: str, args: Dict[str, Any]
    ) -> ToolResult:
        """在沙箱内执行工具。"""
        ...

    @abstractmethod
    def translate_path(self, handle: SandboxHandle, path: str) -> str:
        """把外部路径翻译为沙箱内路径。"""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """当前环境是否可用。"""
        ...