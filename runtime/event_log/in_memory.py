"""
[L2-装配] InMemoryEventLog — EventLog 的默认内存后端。

特点：
    - 列表存储，O(1) 追加、O(n) 查询/聚合。
    - 可选 max_events 上限，超出时丢弃最旧事件，防止长会话内存膨胀。
    - 与 EventLog 基类行为一致（基类本身即为内存实现），
      显式命名以便装配方按语义选择后端。

用法::

    from runtime.event_log import InMemoryEventLog

    log = InMemoryEventLog(max_events=1000)
    log.emit_llm_call(model="gpt-4o", total_tokens=120, cost=0.001)
"""

from __future__ import annotations

from typing import List

from runtime.event_log.interface import Event, EventLog


class InMemoryEventLog(EventLog):
    """EventLog 的内存后端。

    继承基类已提供的内存实现，并显式覆写存储原语以明确语义边界：
    内存后端的事件全部驻留在进程内 `_events` 列表中，进程退出即丢失。

    Attributes:
        backend: 固定为 "memory"。
    """

    backend: str = "memory"

    def __init__(self, max_events: int = 0) -> None:
        """
        Args:
            max_events: 保留的最大事件条数，0 表示不限。
                        超出上限时丢弃最旧的事件（滑动窗口语义）。
        """
        super().__init__(max_events=max_events)

    # ------------------------------------------------------------------
    # 存储原语
    # ------------------------------------------------------------------

    def _append_event(self, event: Event) -> None:
        """追加事件到内存列表，并按 max_events 裁剪最旧事件。"""
        self._events.append(event)
        if self._max_events and len(self._events) > self._max_events:
            del self._events[: len(self._events) - self._max_events]

    def _iter_events(self) -> List[Event]:
        """返回内存事件的浅拷贝列表。"""
        return list(self._events)

    def _clear_events(self) -> None:
        """清空内存事件列表。"""
        self._events.clear()

    # ------------------------------------------------------------------
    # 内存后端专属辅助
    # ------------------------------------------------------------------

    @property
    def max_events(self) -> int:
        """当前配置的最大保留条数（0 表示不限）。"""
        return self._max_events

    def __repr__(self) -> str:
        return (
            f"<InMemoryEventLog events={len(self._events)} "
            f"max_events={self._max_events}>"
        )