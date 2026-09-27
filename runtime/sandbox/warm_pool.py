"""
[L3] runtime.sandbox.warm_pool — 沙箱预热池。

目标（批次 C3）：
    通过预先 ``acquire`` 一批沙箱句柄并缓存，使运行期 ``acquire`` 命中缓存
    时跳过 provider 慢启动（如 Docker 容器拉起），从而显著降低 acquire 开销。

设计约束：
    - 不修改 Agent Loop，也不修改 SandboxProvider 契约。
    - 对 Provider 透明：WarmSandboxPool 只包裹一个 SandboxProvider，对外暴露
      与 acquire / release 兼容的接口。
    - 线程安全：池的入队 / 出队用锁保护，支持并发 acquire/release。
    - 降级：provider.acquire 抛异常时不阻断（透传），命中/未命中统计可观测。

用法::

    from runtime.sandbox.warm_pool import WarmSandboxPool
    from runtime.sandbox.local_provider import LocalSandboxProvider

    pool = WarmSandboxPool(LocalSandboxProvider(), pool_size=4)
    pool.warm(4)                    # 预热 4 个句柄
    handle = pool.acquire("agent")  # 命中缓存，几乎零开销
    pool.release(handle)            # 归还池（池未满时复用，否则透传 release）
"""

from __future__ import annotations

import collections
import threading
from typing import Any, Deque, Dict, List, Optional

from runtime.sandbox.interface import SandboxHandle, SandboxProvider
from runtime.tool_registry.interface import ToolResult


class WarmSandboxPool:
    """沙箱预热池：把 Provider.acquire 的慢启动摊到预热阶段。

    Attributes:
        provider:      被包裹的沙箱提供者。
        pool_size:     池容量上限（达到上限后 release 直接透传，不再缓存）。
        hits:          acquire 命中缓存的次数（观测用）。
        misses:        acquire 未命中缓存的次数（观测用）。
        warm_count:    已预热的句柄数量（观测用）。
    """

    def __init__(
        self,
        provider: SandboxProvider,
        pool_size: int = 4,
    ) -> None:
        """
        Args:
            provider:  被包裹的沙箱提供者。
            pool_size: 池容量上限，<=0 时退化为直接透传 provider。
        """
        self.provider = provider
        self.pool_size = max(0, int(pool_size or 0))
        self._pool: Deque[SandboxHandle] = collections.deque()
        self._lock = threading.Lock()

        # 观测统计
        self.hits = 0
        self.misses = 0
        self.warm_count = 0

    # ------------------------------------------------------------------
    # 预热
    # ------------------------------------------------------------------

    def warm(self, n: Optional[int] = None) -> List[SandboxHandle]:
        """预热 ``n`` 个句柄并加入池。

        预热的句柄不绑定具体 agent（后续命中时直接复用 handle，agent 身份由
        调用方在 handle 之外维护）。

        Args:
            n: 预热数量；None 时使用 pool_size。

        Returns:
            本次预热产生的句柄列表。
        """
        if self.pool_size <= 0:
            return []
        target = int(n) if n is not None else self.pool_size
        created: List[SandboxHandle] = []
        for i in range(target):
            # 预热的 agent_id 用固定前缀，避免污染真实 agent 的句柄语义
            handle = self.provider.acquire(f"warm-pool-{i}")
            created.append(handle)
            self.warm_count += 1
        with self._lock:
            self._pool.extend(created)
        return created

    # ------------------------------------------------------------------
    # 获取 / 释放
    # ------------------------------------------------------------------

    def acquire(self, agent_id: str) -> SandboxHandle:
        """优先从预热池命中返回，否则透传 ``provider.acquire``。"""
        with self._lock:
            if self._pool:
                self.hits += 1
                return self._pool.popleft()
        # 未命中：走 slow path
        self.misses += 1
        return self.provider.acquire(agent_id)

    def release(self, handle: SandboxHandle) -> None:
        """归还句柄：池未满时缓存复用，否则透传 ``provider.release``。"""
        if self.pool_size <= 0:
            self.provider.release(handle)
            return
        with self._lock:
            if len(self._pool) < self.pool_size:
                self._pool.append(handle)
                return
        self.provider.release(handle)

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------

    def size(self) -> int:
        """当前池中缓存的句柄数。"""
        with self._lock:
            return len(self._pool)

    def hit_rate(self) -> float:
        """命中率 = hits / (hits + misses)，无 acquire 时返回 0.0。"""
        total = self.hits + self.misses
        if total <= 0:
            return 0.0
        return self.hits / total


__all__ = ["WarmSandboxPool"]