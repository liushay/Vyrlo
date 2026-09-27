"""
[L4] InMemorySharedStateStore — SharedStateStore 的内存实现。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from runtime.shared_state.interface import SharedStateStore


class InMemorySharedStateStore(SharedStateStore):
    """基于嵌套字典的内存共享状态存储（带 TTL 过期）。"""

    def __init__(self) -> None:
        # namespace -> key -> (value, expire_at)
        self._data: Dict[str, Dict[str, tuple]] = {}

    def set(self, namespace: str, key: str, value: Any, ttl: Optional[float] = None) -> None:
        expire_at = None
        if ttl is not None and ttl > 0:
            expire_at = time.time() + ttl
        bucket = self._data.setdefault(namespace, {})
        bucket[key] = (value, expire_at)

    def get(self, namespace: str, key: str) -> Any:
        bucket = self._data.get(namespace, {})
        entry = bucket.get(key)
        if entry is None:
            return None
        value, expire_at = entry
        if expire_at is not None and time.time() > expire_at:
            bucket.pop(key, None)
            return None
        return value

    def list_keys(self, namespace: str) -> List[str]:
        self._prune(namespace)
        return list(self._data.get(namespace, {}).keys())

    def snapshot(self, namespace: str) -> Dict[str, Any]:
        self._prune(namespace)
        return {
            k: v[0]
            for k, v in self._data.get(namespace, {}).items()
        }

    def _prune(self, namespace: str) -> None:
        """清理某命名空间下已过期的键。"""
        bucket = self._data.get(namespace)
        if not bucket:
            return
        now = time.time()
        expired = [
            k for k, (_, exp) in bucket.items()
            if exp is not None and now > exp
        ]
        for k in expired:
            bucket.pop(k, None)