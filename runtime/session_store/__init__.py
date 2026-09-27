"""
[L2-装配] session_store — Session 持久化抽象与后端实现。

定位：
    为 Runtime 提供"会话状态可存取"的能力，使一次运行结束后可以按
    session_id 恢复（resume）。EventLog 负责"记录发生了什么"，
    SessionStore 负责"这次的会话状态存在哪里"。

公开接口：
    - SessionStore:         抽象接口（save / load / list_ids / delete / clear）
    - InMemorySessionStore: 内存后端（默认）
    - SQLiteSessionStore:   SQLite 持久化后端（单表存 Session.to_dict()）
    - create_session_store(backend, **kwargs): 工厂

抽象与兼容：
    - 接口方法在基类中以"内存后端"作为默认实现，因此 SessionStore() 可直接
      实例化（兼容旧代码 `SessionStore()` 的用法）。
    - 后端只需覆盖存储原语 _save / _load / _list_ids / _delete / _clear。

依赖关系：
    session_store  ->  session（序列化 / 反序列化 Session）
    runtime.py     ->  session_store（装配默认后端）
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Type

from runtime.session import Session

# ============================================================================
# SessionStore —— 接口契约（默认内存实现）
# ============================================================================


class SessionStore:
    """会话存储的接口契约 + 默认内存后端。

    设计约束：
    - 只做"按 session_id 存取会话状态"，不做任何业务决策。
    - list_ids 返回 ID 列表（浅拷贝），调用方无法通过返回值修改内部结构。
    - load 在不存在时返回 None，不抛异常。

    后端扩展方式：覆盖五个存储原语即可（见 _save / _load / _list_ids /
    _delete / _clear），其余行为自动复用。

    用法::

        store = SessionStore()
        store.save(session)
        store.load(session.session_id)
    """

    #: 后端标识（memory / sqlite）
    backend: str = "memory"

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}

    # ------------------------------------------------------------------
    # 存储原语（子类覆盖以替换后端）
    # ------------------------------------------------------------------

    def _save(self, session: Session) -> None:
        """原语：写入一条会话。"""
        self._sessions[session.session_id] = session

    def _load(self, session_id: str) -> Optional[Session]:
        """原语：按 ID 读取会话。"""
        return self._sessions.get(session_id)

    def _list_ids(self) -> List[str]:
        """原语：列出全部会话 ID。"""
        return list(self._sessions.keys())

    def _delete(self, session_id: str) -> bool:
        """原语：删除会话，返回是否删除成功。"""
        return self._sessions.pop(session_id, None) is not None

    def _clear(self) -> None:
        """原语：清空全部会话。"""
        self._sessions.clear()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def save(self, session: Session) -> str:
        """保存会话，返回 session_id。"""
        self._save(session)
        return session.session_id

    def load(self, session_id: str) -> Optional[Session]:
        """按 ID 加载会话，不存在时返回 None。"""
        if not session_id:
            return None
        return self._load(str(session_id))

    def list_ids(self) -> List[str]:
        """列出全部会话 ID。"""
        return self._list_ids()

    def delete(self, session_id: str) -> bool:
        """删除会话，返回是否删除成功。"""
        if not session_id:
            return False
        return self._delete(str(session_id))

    def clear(self) -> None:
        """清空全部会话。"""
        self._clear()

    # ------------------------------------------------------------------
    # 容器协议
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.list_ids())

    def __contains__(self, session_id: Any) -> bool:
        return self.load(str(session_id)) is not None

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} backend={self.backend!r} "
            f"sessions={len(self)}>"
        )


# ============================================================================
# InMemorySessionStore —— 内存后端
# ============================================================================


class InMemorySessionStore(SessionStore):
    """SessionStore 的内存后端。

    会话对象以引用方式驻留内存，进程退出即丢失。
    适合单进程、短生命周期场景，也是 Runtime 的默认后端。
    """

    backend: str = "memory"

    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return f"<InMemorySessionStore sessions={len(self._sessions)}>"


# ============================================================================
# SQLiteSessionStore —— SQLite 持久化后端
# ============================================================================


class SQLiteSessionStore(SessionStore):
    """SessionStore 的 SQLite 持久化后端。

    单表结构::

        CREATE TABLE IF NOT EXISTS sessions (
            session_id   TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL,
            updated_at   REAL NOT NULL
        )

    存的是 `Session.to_dict()` 的 JSON 序列化结果（不含事件明细），
    load 时用 `Session.from_dict()` 还原。threading.Lock 保护读写。

    Attributes:
        db_path: SQLite 数据库路径（":memory:" 表示进程内临时库）。
    """

    backend: str = "sqlite"

    def __init__(
        self,
        db_path: str = ":memory:",
        table: str = "sessions",
    ) -> None:
        """
        Args:
            db_path: SQLite 数据库文件路径，":memory:" 为内存库（默认）。
            table:   会话表名。
        """
        self.db_path: str = db_path or ":memory:"
        self._table: str = table or "sessions"
        self._lock = threading.Lock()

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    session_id   TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at   REAL NOT NULL
                )
                """
            )
            self._conn.commit()

    # ---- 存储原语 ----

    def _save(self, session: Session) -> None:
        payload = json.dumps(session.to_dict(), ensure_ascii=False, default=str)
        with self._lock:
            self._conn.execute(
                f"""
                INSERT INTO {self._table} (session_id, payload_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at   = excluded.updated_at
                """,
                (session.session_id, payload, time.time()),
            )
            self._conn.commit()

    def _load(self, session_id: str) -> Optional[Session]:
        with self._lock:
            row = self._conn.execute(
                f"SELECT payload_json FROM {self._table} WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            data = json.loads(row["payload_json"])
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        return Session.from_dict(data)

    def _list_ids(self) -> List[str]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT session_id FROM {self._table} ORDER BY session_id ASC"
            ).fetchall()
        return [str(row["session_id"]) for row in rows]

    def _delete(self, session_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                f"DELETE FROM {self._table} WHERE session_id = ?", (session_id,)
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def _clear(self) -> None:
        with self._lock:
            self._conn.execute(f"DELETE FROM {self._table}")
            self._conn.commit()

    # ---- 生命周期 ----

    def close(self) -> None:
        """关闭数据库连接（幂等）。"""
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    def __enter__(self) -> "SQLiteSessionStore":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"<SQLiteSessionStore db={self.db_path!r} "
            f"table={self._table!r} sessions={len(self.list_ids())}>"
        )


# ============================================================================
# 工厂
# ============================================================================

#: 后端名称 → 后端类
_BACKENDS: Dict[str, Type[SessionStore]] = {
    "memory": InMemorySessionStore,
    "in_memory": InMemorySessionStore,
    "sqlite": SQLiteSessionStore,
    "sqlite3": SQLiteSessionStore,
}


def create_session_store(backend: str = "memory", **kwargs: Any) -> SessionStore:
    """创建指定后端的会话存储。

    Args:
        backend: 后端标识，支持 "memory"（默认）与 "sqlite"。
                 名称大小写不敏感，"-" 与 "_" 等价。
        **kwargs: 透传给后端构造函数的关键字参数。

            memory 后端：无额外参数。

            sqlite 后端：
                db_path (str): 数据库路径，默认 ":memory:"。
                table (str):   会话表名，默认 "sessions"。

    Returns:
        SessionStore 实例。

    Raises:
        ValueError: backend 不受支持时。

    Examples:
        >>> create_session_store("memory")
        <InMemorySessionStore sessions=0>
        >>> create_session_store("sqlite", db_path=":memory:")
        <SQLiteSessionStore db=':memory:' table='sessions' sessions=0>
    """
    key = str(backend or "memory").strip().lower().replace("-", "_")
    backend_cls: Optional[Type[SessionStore]] = _BACKENDS.get(key)
    if backend_cls is None:
        raise ValueError(
            f"不支持的 SessionStore 后端: {backend!r}。"
            f"可选: {sorted(set(_BACKENDS.keys()))}"
        )
    return backend_cls(**kwargs)


__all__ = [
    "SessionStore",
    "InMemorySessionStore",
    "SQLiteSessionStore",
    "create_session_store",
]