"""
[L2-装配] event_log — EventLog 后端抽象包。

本包提供"运行期事实"的 append-only 记录基础设施，支持可插拔后端。

公开接口：
    - 事件模型：Event / LLMCallEvent / ToolCallEvent
    - 事件类型常量：EVENT_LLM_CALL / EVENT_TOOL_CALL / EVENT_LOOP_START /
                    EVENT_LOOP_EXIT / EVENT_ERROR
    - 后端：EventLog（接口契约 + 默认内存实现）
            InMemoryEventLog（显式内存后端）
            SQLiteEventLog（SQLite 持久化后端）
    - 工厂：create_event_log(backend, **kwargs)
    - 助手：resolve_event_log(...)

兼容性：
    既有代码（L3 五中间件、examples、tests）通过
    `from runtime.event_log import EventLog, resolve_event_log, EVENT_LLM_CALL ...`
    导入，本包保持这些符号的名字与语义不变。

用法::

    from runtime.event_log import create_event_log

    log = create_event_log("memory")                    # 默认内存后端
    log = create_event_log("sqlite", db_path="ev.db")   # SQLite 持久化后端
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

from runtime.event_log.interface import (
    EVENT_CIRCUIT_OPEN,
    EVENT_DELEGATION,
    EVENT_ERROR,
    EVENT_FALLBACK,
    EVENT_LLM_CALL,
    EVENT_LOOP_EXIT,
    EVENT_LOOP_START,
    EVENT_RETRY,
    EVENT_TOOL_CALL,
    EVENT_TYPES,
    Event,
    EventLog,
    LLMCallEvent,
    ToolCallEvent,
    event_from_record,
    event_to_record,
    resolve_event_log,
)
from runtime.event_log.in_memory import InMemoryEventLog
from runtime.event_log.sqlite_log import DEFAULT_SESSION_ID, SQLiteEventLog

__all__ = [
    # 事件模型
    "Event",
    "LLMCallEvent",
    "ToolCallEvent",
    # 事件类型常量
    "EVENT_LLM_CALL",
    "EVENT_TOOL_CALL",
    "EVENT_LOOP_START",
    "EVENT_LOOP_EXIT",
    "EVENT_ERROR",
    "EVENT_DELEGATION",
    "EVENT_TYPES",
    "EVENT_RETRY",
    "EVENT_FALLBACK",
    "EVENT_CIRCUIT_OPEN",
    # 后端
    "EventLog",
    "InMemoryEventLog",
    "SQLiteEventLog",
    "DEFAULT_SESSION_ID",
    # 工厂与助手
    "create_event_log",
    "resolve_event_log",
    # 序列化助手
    "event_to_record",
    "event_from_record",
]


#: 后端名称 → 后端类
_BACKENDS: Dict[str, Type[EventLog]] = {
    "memory": InMemoryEventLog,
    "in_memory": InMemoryEventLog,
    "sqlite": SQLiteEventLog,
    "sqlite3": SQLiteEventLog,
}


def create_event_log(backend: str = "memory", **kwargs: Any) -> EventLog:
    """创建指定后端的事件日志。

    Args:
        backend: 后端标识，支持 "memory"（默认）与 "sqlite"。
                 名称大小写不敏感，"-" 与 "_" 等价。
        **kwargs: 透传给后端构造函数的关键字参数。

            memory 后端：
                max_events (int): 保留的最大事件条数，0 表示不限。

            sqlite 后端：
                db_path (str):   数据库路径，默认 ":memory:"。
                max_events (int): 每个会话保留的最大事件条数，0 表示不限。
                session_id (str): 默认会话标识，默认 "default"。
                table (str):      事件表名，默认 "events"。

    Returns:
        EventLog 实例。

    Raises:
        ValueError: backend 不受支持时。

    Examples:
        >>> create_event_log("memory", max_events=100)
        <InMemoryEventLog events=0 max_events=100>
        >>> create_event_log("sqlite", db_path=":memory:")
        <SQLiteEventLog db=':memory:' table='events' events=0>
    """
    key = str(backend or "memory").strip().lower().replace("-", "_")
    backend_cls: Optional[Type[EventLog]] = _BACKENDS.get(key)
    if backend_cls is None:
        raise ValueError(
            f"不支持的 EventLog 后端: {backend!r}。"
            f"可选: {sorted(set(_BACKENDS.keys()))}"
        )
    return backend_cls(**kwargs)