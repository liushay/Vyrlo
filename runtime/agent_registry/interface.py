"""
[L4] runtime.agent_registry — 多 Agent 注册表。

定义 AgentSpec 数据模型与 AgentRegistry 抽象接口，
供 Runtime / DelegationMiddleware 查找可委派的目标 Agent。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AgentSpec:
    """一个可委派 Agent 的静态描述。

    Attributes:
        agent_id:           Agent 唯一标识。
        description:        人类可读描述。
        capabilities:       能力标签列表（供 find_by_capability 检索）。
        tool_whitelist:     该 Agent 可用的工具白名单（None 表示不限）。
        context_namespace:  该 Agent 使用的长期记忆命名空间。
        max_iterations:     该 Agent 的 Loop 最大迭代轮数。
        sandbox_mode:       沙箱模式（"local" / "docker"），默认 "local"。
    """

    agent_id: str
    description: str = ""
    capabilities: List[str] = field(default_factory=list)
    tool_whitelist: Optional[List[str]] = None
    context_namespace: str = "default"
    max_iterations: int = 5
    sandbox_mode: str = "local"


class AgentRegistry(ABC):
    """Agent 注册表抽象接口。"""

    @abstractmethod
    def register(self, spec: AgentSpec) -> None:
        """注册一个 Agent。"""
        ...

    @abstractmethod
    def unregister(self, agent_id: str) -> Optional[AgentSpec]:
        """注销 Agent，返回被移除的 spec（不存在返回 None）。"""
        ...

    @abstractmethod
    def get(self, agent_id: str) -> Optional[AgentSpec]:
        """按 ID 获取 Agent。"""
        ...

    @abstractmethod
    def list(self) -> List[AgentSpec]:
        """列出全部 Agent。"""
        ...

    @abstractmethod
    def find_by_capability(self, capability: str) -> List[AgentSpec]:
        """按能力标签查找 Agent。"""
        ...