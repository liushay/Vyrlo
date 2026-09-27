"""
[L3] SandboxRegistry — 沙箱提供者注册表。

按名称注册 SandboxProvider，按 mode 选择可用 provider 并 acquire。

约束：
    - warm_pool 仅保留接口（可选），不实现完整预热逻辑。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from runtime.event_log import EVENT_FALLBACK, EventLog
from runtime.sandbox.interface import SandboxHandle, SandboxProvider


class SandboxRegistry:
    """沙箱提供者注册表。

    [D2] 降级语义：``acquire`` 按 ``fallback_modes`` 顺序选择可用 provider，
    默认 ``("docker", "local")``——Docker 不可用时自动降级到 Local；两者都
    不可用时返回 None（拒绝执行）。降级过程写入 ``EVENT_FALLBACK`` 事件。
    """

    #: 默认降级顺序（先 docker，后 local）
    DEFAULT_FALLBACK_MODES: List[str] = ["docker", "local"]

    def __init__(
        self,
        providers: Optional[Dict[str, SandboxProvider]] = None,
        fallback_modes: Optional[List[str]] = None,
        event_log: Optional[EventLog] = None,
    ) -> None:
        """
        Args:
            providers:      初始 provider 映射 {name: provider}。
            fallback_modes: 降级顺序列表；None 时用 DEFAULT_FALLBACK_MODES。
                            空列表表示"只用精确 mode，不降级"。
            event_log:      可选事件日志，用于记录降级事件。
        """
        self._providers: Dict[str, SandboxProvider] = dict(providers or {})
        self._fallback_modes: List[str] = (
            list(fallback_modes)
            if fallback_modes is not None
            else list(self.DEFAULT_FALLBACK_MODES)
        )
        self._event_log = event_log

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------

    def register(self, name: str, provider: SandboxProvider) -> None:
        """注册一个沙箱提供者。"""
        self._providers[name] = provider

    def get(self, name: str) -> Optional[SandboxProvider]:
        """按名称获取 provider。"""
        return self._providers.get(name)

    def list_modes(self) -> List[str]:
        """列出已注册的 mode 名。"""
        return list(self._providers.keys())

    # ------------------------------------------------------------------
    # 选择与获取
    # ------------------------------------------------------------------

    def acquire(self, agent_id: str, mode: str = "local") -> Optional[SandboxHandle]:
        """选择可用 provider 并 acquire 一个沙箱句柄。

        [D2] 选择规则（可配置降级）：
            - 先尝试精确 mode；不可用时按 ``fallback_modes`` 顺序降级
              （默认 docker → local）。
            - 只有"精确 mode"和"降级链"都不可用时才返回 None（拒绝执行）。

        Args:
            agent_id: agent 标识。
            mode:     期望的沙箱模式（如 "local"、"docker"）。

        Returns:
            SandboxHandle；无可用 provider 时返回 None。
        """
        order = self._resolve_order(mode)

        attempted: Optional[str] = None
        for candidate in order:
            provider = self._providers.get(candidate)
            if provider is None or not provider.is_available():
                attempted = candidate
                continue
            if attempted is not None and attempted != candidate:
                self._record_fallback(attempted, candidate)
            return provider.acquire(agent_id)

        return None

    def _resolve_order(self, mode: str) -> List[str]:
        """构建获取候选顺序：精确 mode 优先，其后追加 fallback_modes。"""
        order: List[str] = []
        if mode:
            order.append(mode)
        for candidate in self._fallback_modes:
            if candidate not in order:
                order.append(candidate)
        # 兜底：把不在降级链里的其余可用 provider 追加进来（保持旧"任意可用"语义）
        for candidate in self._providers.keys():
            if candidate not in order:
                order.append(candidate)
        return order

    def _record_fallback(self, from_mode: str, to_mode: str) -> None:
        """[D2] 把沙箱降级写入事件日志。"""
        if self._event_log is not None:
            try:
                self._event_log.emit(
                    EVENT_FALLBACK,
                    scope="sandbox",
                    **{"from": from_mode, "to": to_mode, "reason": "unavailable"},
                )
            except Exception:  # pragma: no cover - 日志失败不阻断
                pass

    def release(self, handle: SandboxHandle, mode: str = "local") -> None:
        """释放沙箱句柄。尽量找到对应的 provider。"""
        # 优先按 handle.metadata 记录的 provider 名；否则按 mode
        provider_name = handle.metadata.get("provider") if handle.metadata else mode
        # [D003 同类] 用 `is not None` 判定"是否取到"：若某个 Provider 实现
        # 了 __len__（如"活跃沙箱数 == 0"），`get(...) or ...` 会把它误判为
        # 未取到并静默替换成另一个 provider，导致 release 落到错误的沙箱上。
        provider = self._providers.get(provider_name)
        if provider is None:
            provider = self._select(mode)
        if provider is not None:
            provider.release(handle)

    def _select(self, mode: str) -> Optional[SandboxProvider]:
        """选择可用 provider：精确 mode 优先，其次任意可用。"""
        exact = self._providers.get(mode)
        if exact is not None and exact.is_available():
            return exact
        for provider in self._providers.values():
            if provider.is_available():
                return provider
        return None

    # ------------------------------------------------------------------
    # warm_pool 接口（预留）
    # ------------------------------------------------------------------

    def warm_pool(self, mode: str, n: int) -> List[SandboxHandle]:
        """预留：预先 acquire N 个 handle 组成池。

        此处返回空列表，不实现完整预热逻辑（见任务约束"不做 warm pool 完整实现"）。
        """
        return []

    def __len__(self) -> int:
        return len(self._providers)