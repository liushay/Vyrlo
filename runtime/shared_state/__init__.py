"""
[L4] runtime.shared_state — 子 Agent 共享状态（最小版）。
"""

from runtime.shared_state.interface import SharedStateStore
from runtime.shared_state.in_memory import InMemorySharedStateStore

__all__ = [
    "SharedStateStore",
    "InMemorySharedStateStore",
]