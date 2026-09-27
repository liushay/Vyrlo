"""
[L4] runtime.shared_state — 子 Agent 共享状态（最小版）。

定义 SharedStateStore 抽象接口与 InMemorySharedStateStore 实现。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class SharedStateStore(ABC):
    """跨 Agent 共享状态存储抽象接口。"""

    @abstractmethod
    def set(self, namespace: str, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """写入一条共享状态（ttl 秒后过期，None 表示不过期）。"""
        ...

    @abstractmethod
    def get(self, namespace: str, key: str) -> Any:
        """读取一条共享状态，不存在或已过期返回 None。"""
        ...

    @abstractmethod
    def list_keys(self, namespace: str) -> List[str]:
        """列出某命名空间下所有有效 key。"""
        ...

    @abstractmethod
    def snapshot(self, namespace: str) -> Dict[str, Any]:
        """返回某命名空间的有效键值快照。"""
        ...