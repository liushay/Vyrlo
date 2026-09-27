"""
[L4] runtime.agent_registry — 多 Agent 注册表。
"""

from runtime.agent_registry.interface import AgentRegistry, AgentSpec
from runtime.agent_registry.in_memory import InMemoryAgentRegistry

__all__ = [
    "AgentSpec",
    "AgentRegistry",
    "InMemoryAgentRegistry",
]