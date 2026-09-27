"""
[L2-装配] SQLiteEventLog — EventLog 的 SQLite 持久化后端。

表结构（单表）::

    CREATE TABLE IF NOT EXISTS events (
        session_id   TEXT    NOT NULL,
        seq          INTEGER NOT NULL,
        event_type   TEXT    NOT NULL,
        timestamp    REAL    NOT NULL,
        iteration    INTEGER NOT NULL,
        payload_json TEXT    NOT NULL,
        PRIMARY KEY (session_id, seq)
    )

设计要点：
    - 单表结构：所有事件共用一张表，按 (session_id, seq) 主键唯一定位。
      seq 是"会话内单调递增序号"，保证同一会话内事件顺序可精确还原。
    - payload_json 存放除索引列（event_type / timestamp / iteration）之外的
      全部事件字段，读取时与索引列合并还原为具体事件对象。
    - threading.Lock 保护全部读写：sqlite3 连接以 check_same_thread=False 打开，
      由显式锁保证线程安全。
    - max_events（0 表示不限）按会话裁剪最旧事件，防止长会话表膨胀。

用法::

    from runtime.event_log import SQLiteEventLog

    log = SQLiteEventLog(db_path="runtime_events.db")
    log.emit_llm_call(session_id="s1", model="gpt-4o", total_tokens=120)
    log.close()
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Dict, List

from runtime.event_log.interface import (
    Event,
    EventLog,
    event_from_record,
    event_to_record,
)

#: 未显式提供 session_id 时使用的默认分组
DEFAULT_SESSION_ID = "default"

#: 索引列（其余字段进入 payload_json）
_INDEXED_COLUMNS = ("event_type", "timestamp", "iteration")


class SQLiteEventLog(EventLog):
    """EventLog 的 SQLite 持久化后端。

    Attributes:
        backend: 固定为 "sqlite"。
        db_path: SQLite 数据库路径（":memory:" 表示进程内临时库）。
    """

    backend: str = "sqlite"

    def __init__(
        self,
        db_path: str = ":memory:",
        max_events: int = 0,
        session_id: str = DEFAULT_SESSION_ID,
        table: str = "events",
    ) -> None:
        """
        Args:
            db_path:    SQLite 数据库文件路径，":memory:" 为内存库（默认）。
            max_events: 每个会话保留的最大事件条数，0 表示不限。
            session_id: 写入时使用的默认会话标识（emit 时可被 kwargs 覆盖）。
            table:      事件表名（允许隔离多套日志）。
        """
        # 不调用 super().__init__（基类初始化内存列表），
        # 但保留 _max_events 语义（用于按会话裁剪）。
        self._max_events: int = int(max_events or 0)
        self._default_session_id: str = session_id or DEFAULT_SESSION_ID
        self._table: str = table or "events"
        self._lock = threading.Lock()

        self.db_path: str = db_path or ":memory:"
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        """建表并创建按事件类型查询的索引。"""
        with self._lock:
            self._conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    session_id   TEXT    NOT NULL,
                    seq          INTEGER NOT NULL,
                    event_type   TEXT    NOT NULL,
                    timestamp    REAL    NOT NULL,
                    iteration    INTEGER NOT NULL,
                    payload_json TEXT    NOT NULL,
                    PRIMARY KEY (session_id, seq)
                )
                """
            )
            self._conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self._table}_type "
                f"ON {self._table} (event_type)"
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # 存储原语
    # ------------------------------------------------------------------

    def _append_event(self, event: Event) -> None:
        """将事件写入 SQLite，并按 max_events 裁剪最旧记录。

        事件对象上可携带 `session_id` 属性（非 dataclass 字段）用于分组；
        未携带时落到默认会话。
        """
        session_id = str(
            getattr(event, "session_id", None) or self._default_session_id
        )
        record = event_to_record(event)
        indexed = {key: record.pop(key, None) for key in _INDEXED_COLUMNS}

        event_type = str(indexed.get("event_type") or "")
        timestamp = float(indexed.get("timestamp") or time.time())
        iteration = int(indexed.get("iteration") or 0)
        payload_json = json.dumps(record, ensure_ascii=False, default=str)

        with self._lock:
            seq = self._next_seq(session_id)
            self._conn.execute(
                f"""
                INSERT INTO {self._table}
                    (session_id, seq, event_type, timestamp, iteration, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, seq, event_type, timestamp, iteration, payload_json),
            )
            self._trim_session(session_id)
            self._conn.commit()

    def _iter_events(self) -> List[Event]:
        """按 (session_id, seq) 顺序读出全部事件并还原为事件对象。"""
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT session_id, seq, event_type, timestamp, iteration, payload_json
                FROM {self._table}
                ORDER BY session_id ASC, seq ASC
                """
            ).fetchall()

        return [self._row_to_event(row) for row in rows]

    def _clear_events(self) -> None:
        """清空事件表。"""
        with self._lock:
            self._conn.execute(f"DELETE FROM {self._table}")
            self._conn.commit()

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _next_seq(self, session_id: str) -> int:
        """返回指定会话的下一个序号（会话内单调递增，从 1 开始）。"""
        cursor = self._conn.execute(
            f"SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq "
            f"FROM {self._table} WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        return int(row["next_seq"]) if row is not None else 1

    def _trim_session(self, session_id: str) -> None:
        """按 max_events 裁剪指定会话最旧的事件。"""
        if not self._max_events:
            return
        self._conn.execute(
            f"""
            DELETE FROM {self._table}
            WHERE session_id = ?
              AND seq <= (
                  SELECT MAX(seq) - ? FROM {self._table} WHERE session_id = ?
              )
            """,
            (session_id, self._max_events, session_id),
        )

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> Event:
        """将数据库行还原为事件对象（索引列 + payload_json 合并）。"""
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}

        record: Dict[str, Any] = dict(payload)
        record["event_type"] = row["event_type"]
        record["timestamp"] = row["timestamp"]
        record["iteration"] = row["iteration"]
        return event_from_record(record)

    # ------------------------------------------------------------------
    # SQLite 后端专属辅助
    # ------------------------------------------------------------------

    @property
    def max_events(self) -> int:
        """当前配置的每会话最大保留条数（0 表示不限）。"""
        return self._max_events

    def set_default_session(self, session_id: str) -> None:
        """切换后续写入使用的默认会话标识。

        Runtime 在每次 run() 前调用，使未显式携带 session_id 的事件
        （如 EventRecorder 写入的生命周期事件）归入当前会话分组。
        内存后端无此方法，Runtime 通过 hasattr 探测，属可选能力。
        """
        self._default_session_id = str(session_id or DEFAULT_SESSION_ID)

    def session_ids(self) -> List[str]:
        """列出库中出现过的全部会话标识。"""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT DISTINCT session_id FROM {self._table} "
                f"ORDER BY session_id ASC"
            ).fetchall()
        return [str(row["session_id"]) for row in rows]

    def close(self) -> None:
        """关闭数据库连接（幂等）。"""
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    def __enter__(self) -> "SQLiteEventLog":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"<SQLiteEventLog db={self.db_path!r} "
            f"table={self._table!r} events={self.count()}>"
        )