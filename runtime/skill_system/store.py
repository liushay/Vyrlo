"""
[L5] skill_system/store — 技能存储（SkillStore）。

SQLite 后端，三张表：

- ``skills``：技能当前态（skill_id / name / text / version / created_at /
  updated_at）。
- ``skill_versions``：技能版本 DAG（每次变更记录一条，含 parent_version
  指向父版本）。
- ``skill_counters``：技能运行期计数器（selections / applied / completions /
  fallbacks）。

公开方法：

- create / get / update / list
- get_counters / bump_counter

版本 DAG 说明：
    每次 ``update`` / ``create_version`` 都会在 ``skill_versions`` 中追加一条
    记录，其 ``parent_version`` 指向前一版本号。初始创建的记录 ``parent_version``
    为 NULL（根节点）。这样可回溯技能完整的演化链，供 GitRatchet 回滚使用。
"""

from __future__ import annotations

import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

from runtime.skill_system.models import Skill, SkillCounters, SkillVersion

#: 计数器字段（skill_counters 表中的列）
_COUNTER_FIELDS = ("selections", "applied", "completions", "fallbacks")


class SkillStore:
    """技能的 SQLite 存储后端。

    Attributes:
        db_path: SQLite 数据库路径（":memory:" 表示进程内临时库）。
    """

    backend: str = "sqlite"

    def __init__(
        self,
        db_path: str = ":memory:",
        table_prefix: str = "skill",
    ) -> None:
        """
        Args:
            db_path:       SQLite 数据库文件路径，默认 ":memory:"。
            table_prefix:  表名前缀，默认 "skill"（生成 skills_<prefix> 等表）。
        """
        self.db_path: str = db_path or ":memory:"
        self._prefix: str = table_prefix or "skill"
        self._lock = threading.Lock()

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ------------------------------------------------------------------
    # 表名
    # ------------------------------------------------------------------

    @property
    def _skills_table(self) -> str:
        return f"{self._prefix}s"

    @property
    def _versions_table(self) -> str:
        return f"{self._prefix}_versions"

    @property
    def _counters_table(self) -> str:
        return f"{self._prefix}_counters"

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._skills_table} (
                    skill_id   TEXT PRIMARY KEY,
                    name       TEXT NOT NULL,
                    text       TEXT NOT NULL,
                    version    INTEGER NOT NULL DEFAULT 1,
                    score      REAL NOT NULL DEFAULT 0.0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            self._conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._versions_table} (
                    skill_id       TEXT NOT NULL,
                    version        INTEGER NOT NULL,
                    text           TEXT NOT NULL,
                    created_at     REAL NOT NULL,
                    score          REAL NOT NULL DEFAULT 0.0,
                    parent_version INTEGER,
                    PRIMARY KEY (skill_id, version)
                )
                """
            )
            self._conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._counters_table} (
                    skill_id    TEXT PRIMARY KEY,
                    selections  INTEGER NOT NULL DEFAULT 0,
                    applied     INTEGER NOT NULL DEFAULT 0,
                    completions INTEGER NOT NULL DEFAULT 0,
                    fallbacks   INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # create
    # ------------------------------------------------------------------

    def create(
        self,
        skill_id: str,
        name: str,
        text: str,
        version: int = 1,
        score: float = 0.0,
    ) -> Skill:
        """创建新技能（含根版本记录与零计数器）。

        Args:
            skill_id: 技能唯一标识。
            name:     技能名称。
            text:     技能文本。
            version:  初始版本号，默认 1。
            score:    初始评分，默认 0.0。

        Returns:
            创建的 Skill。

        Raises:
            ValueError: skill_id 已存在时。
        """
        now = time.time()
        with self._lock:
            exists = self._conn.execute(
                f"SELECT 1 FROM {self._skills_table} WHERE skill_id = ?",
                (skill_id,),
            ).fetchone()
            if exists is not None:
                raise ValueError(f"技能已存在: {skill_id!r}")

            self._conn.execute(
                f"""
                INSERT INTO {self._skills_table}
                    (skill_id, name, text, version, score, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (skill_id, name, text, int(version), float(score), now, now),
            )
            # 根版本：parent_version 为 NULL
            self._conn.execute(
                f"""
                INSERT INTO {self._versions_table}
                    (skill_id, version, text, created_at, score, parent_version)
                VALUES (?, ?, ?, ?, ?, NULL)
                """,
                (skill_id, int(version), text, now, float(score)),
            )
            self._conn.execute(
                f"INSERT INTO {self._counters_table} (skill_id) VALUES (?)",
                (skill_id,),
            )
            self._conn.commit()

        return Skill(
            skill_id=skill_id,
            name=name,
            text=text,
            version=int(version),
            score=float(score),
            counters=SkillCounters(),
        )

    # ------------------------------------------------------------------
    # get
    # ------------------------------------------------------------------

    def get(self, skill_id: str) -> Optional[Skill]:
        """按 ID 读取技能，不存在返回 None。"""
        with self._lock:
            row = self._conn.execute(
                f"SELECT * FROM {self._skills_table} WHERE skill_id = ?",
                (skill_id,),
            ).fetchone()
        if row is None:
            return None
        counters = self.get_counters(skill_id)
        return Skill(
            skill_id=row["skill_id"],
            name=row["name"],
            text=row["text"],
            version=int(row["version"]),
            score=float(row["score"]),
            counters=counters,
        )

    # ------------------------------------------------------------------
    # update / create_version
    # ------------------------------------------------------------------

    def update(
        self,
        skill_id: str,
        text: Optional[str] = None,
        name: Optional[str] = None,
        score: Optional[float] = None,
    ) -> Optional[Skill]:
        """更新技能文本 / 名称 / 评分，并生成一个新版本（DAG 追加节点）。

        版本号自动 +1，``parent_version`` 指向更新前的版本。若技能不存在，
        返回 None。

        Args:
            skill_id: 技能标识。
            text:     新文本（None 表示不变）。
            name:     新名称（None 表示不变）。
            score:    新评分（None 表示不变）。

        Returns:
            更新后的 Skill，或 None（不存在时）。
        """
        existing = self.get(skill_id)
        if existing is None:
            return None

        new_name = existing.name if name is None else name
        new_text = existing.text if text is None else text
        new_score = existing.score if score is None else float(score)
        new_version = existing.version + 1
        now = time.time()

        with self._lock:
            self._conn.execute(
                f"""
                UPDATE {self._skills_table}
                SET name = ?, text = ?, version = ?, score = ?, updated_at = ?
                WHERE skill_id = ?
                """,
                (new_name, new_text, new_version, new_score, now, skill_id),
            )
            self._conn.execute(
                f"""
                INSERT INTO {self._versions_table}
                    (skill_id, version, text, created_at, score, parent_version)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    skill_id,
                    new_version,
                    new_text,
                    now,
                    new_score,
                    existing.version,
                ),
            )
            self._conn.commit()

        return self.get(skill_id)

    def create_version(
        self,
        skill_id: str,
        text: str,
        score: Optional[float] = None,
    ) -> Optional[SkillVersion]:
        """显式追加一个技能版本（不动 skills 主表以外的计数器）。

        返回创建的 SkillVersion，或 None（技能不存在时）。
        """
        existing = self.get(skill_id)
        if existing is None:
            return None

        new_score = existing.score if score is None else float(score)
        new_version = existing.version + 1
        now = time.time()

        with self._lock:
            self._conn.execute(
                f"""
                INSERT INTO {self._versions_table}
                    (skill_id, version, text, created_at, score, parent_version)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (skill_id, new_version, text, now, new_score, existing.version),
            )
            self._conn.commit()

        return SkillVersion(
            skill_id=skill_id,
            version=new_version,
            text=text,
            created_at=now,
            score=new_score,
            parent_version=existing.version,
        )

    # ------------------------------------------------------------------
    # list / versions
    # ------------------------------------------------------------------

    def list(self) -> List[Skill]:
        """列出全部技能（含计数器）。"""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM {self._skills_table} ORDER BY skill_id ASC"
            ).fetchall()
        skills: List[Skill] = []
        for row in rows:
            skills.append(
                Skill(
                    skill_id=row["skill_id"],
                    name=row["name"],
                    text=row["text"],
                    version=int(row["version"]),
                    score=float(row["score"]),
                    counters=self.get_counters(row["skill_id"]),
                )
            )
        return skills

    def list_versions(self, skill_id: str) -> List[SkillVersion]:
        """按时间（版本号）升序返回某技能的全部版本记录。"""
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT * FROM {self._versions_table}
                WHERE skill_id = ?
                ORDER BY version ASC
                """,
                (skill_id,),
            ).fetchall()
        return [
            SkillVersion(
                skill_id=row["skill_id"],
                version=int(row["version"]),
                text=row["text"],
                created_at=float(row["created_at"]),
                score=float(row["score"]),
                parent_version=row["parent_version"],
            )
            for row in rows
        ]

    def get_version(self, skill_id: str, version: int) -> Optional[SkillVersion]:
        """读取某技能的指定版本，不存在返回 None。"""
        with self._lock:
            row = self._conn.execute(
                f"""
                SELECT * FROM {self._versions_table}
                WHERE skill_id = ? AND version = ?
                """,
                (skill_id, int(version)),
            ).fetchone()
        if row is None:
            return None
        return SkillVersion(
            skill_id=row["skill_id"],
            version=int(row["version"]),
            text=row["text"],
            created_at=float(row["created_at"]),
            score=float(row["score"]),
            parent_version=row["parent_version"],
        )

    # ------------------------------------------------------------------
    # counters
    # ------------------------------------------------------------------

    def get_counters(self, skill_id: str) -> SkillCounters:
        """读取技能计数器，不存在时返回全零计数器。"""
        with self._lock:
            row = self._conn.execute(
                f"SELECT * FROM {self._counters_table} WHERE skill_id = ?",
                (skill_id,),
            ).fetchone()
        if row is None:
            return SkillCounters()
        return SkillCounters(
            selections=int(row["selections"] or 0),
            applied=int(row["applied"] or 0),
            completions=int(row["completions"] or 0),
            fallbacks=int(row["fallbacks"] or 0),
        )

    def bump_counter(
        self,
        skill_id: str,
        counter: str,
        delta: int = 1,
    ) -> Optional[SkillCounters]:
        """对指定计数器累加 ``delta``，返回累加后的计数器。

        Args:
            skill_id: 技能标识。
            counter:  计数器名（selections / applied / completions / fallbacks）。
            delta:    增量，可为负（负值下限截断为 0）。

        Returns:
            更新后的 SkillCounters；技能不存在返回 None。

        Raises:
            ValueError: counter 非法时。
        """
        if counter not in _COUNTER_FIELDS:
            raise ValueError(f"未知计数器字段: {counter!r}")

        existing = self.get(skill_id)
        if existing is None:
            return None

        current = getattr(existing.counters, counter, 0) or 0
        new_value = max(0, int(current) + int(delta))

        with self._lock:
            self._conn.execute(
                f"""
                INSERT INTO {self._counters_table} (skill_id, {counter})
                VALUES (?, ?)
                ON CONFLICT(skill_id) DO UPDATE SET
                    {counter} = excluded.{counter}
                """,
                (skill_id, new_value),
            )
            self._conn.commit()

        return self.get_counters(skill_id)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def close(self) -> None:
        """关闭数据库连接（幂等）。"""
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    def __enter__(self) -> "SkillStore":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __len__(self) -> int:
        with self._lock:
            row = self._conn.execute(
                f"SELECT COUNT(*) AS c FROM {self._skills_table}"
            ).fetchone()
        return int(row["c"])

    def __contains__(self, skill_id: Any) -> bool:
        return self.get(str(skill_id)) is not None

    def __repr__(self) -> str:
        return (
            f"<SkillStore db={self.db_path!r} prefix={self._prefix!r} "
            f"skills={len(self)}>"
        )


__all__ = ["SkillStore"]