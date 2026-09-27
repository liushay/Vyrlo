"""
[L4] InMemoryAgentRegistry — AgentRegistry 的内存实现。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from runtime.agent_registry.interface import AgentRegistry, AgentSpec


class InMemoryAgentRegistry(AgentRegistry):
    """基于字典的 Agent 注册表实现。"""

    def __init__(self) -> None:
        self._agents: Dict[str, AgentSpec] = {}

    def register(self, spec: AgentSpec) -> None:
        if spec.agent_id in self._agents:
            raise ValueError(f"Agent '{spec.agent_id}' 已存在")
        self._agents[spec.agent_id] = spec

    def unregister(self, agent_id: str) -> Optional[AgentSpec]:
        return self._agents.pop(agent_id, None)

    def get(self, agent_id: str) -> Optional[AgentSpec]:
        return self._agents.get(agent_id)

    def list(self) -> List[AgentSpec]:
        return list(self._agents.values())

    def find_by_capability(self, capability: str) -> List[AgentSpec]:
        return [spec for spec in self._agents.values() if capability in spec.capabilities]