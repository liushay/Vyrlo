"""
上下文管理策略实现 — 五层记忆系统的全部策略实现。

[FIX-H2] SummaryInjection.inject 修复：移除死代码分支，明确契约只返回格式化文本。
[FIX-H6] SQLiteStore 修复：移除 FTS5 虚拟表，retrieve 用 ORDER BY strength DESC + Python BM25。
[FIX-L2] InMemoryStore/SQLiteStore 新增 get(trace_id) 方法。
[FIX-L3] InMemoryStore/SQLiteStore 新增 update_strengths_batch 方法。
[FIX-L5] LLMExtraction 新增去重逻辑。
"""

from __future__ import annotations

import json as json_mod
import logging
import math
import sqlite3
import threading
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from .interface import (
    CompressStrategy,
    DecayStrategy,
    ExtractionStrategy,
    InjectionStrategy,
    LongTermStore,
    Message,
    MemoryTrace,
    RetrievalStrategy,
    ShortTermStrategy,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 短期记忆策略
# ============================================================================


class SlidingWindow(ShortTermStrategy):
    """滑动窗口：保留最近 N 条消息。"""

    def __init__(self, max_messages: int = 20) -> None:
        self._max_messages = max_messages

    def manage(self, messages: List[Message], config: Dict[str, Any]) -> List[Message]:
        max_msgs = config.get("max_messages", self._max_messages)
        return messages[-max_msgs:]


class TokenBudget(ShortTermStrategy):
    """
    Token 预算：基于 token 数限制。

    [FIX-H3] 修复了截断逻辑——从末尾向前累积消息，
    直到接近 token 预算上限。
    """

    def __init__(self, max_tokens: int = 4096) -> None:
        self._max_tokens = max_tokens
        self._token_counter: Optional[Callable[[str], int]] = None

    def set_token_counter(self, counter: Callable[[str], int]) -> None:
        """设置 token 计数函数。"""
        self._token_counter = counter

    def _estimate_tokens(self, msg: Message) -> int:
        """估算消息的 token 数。"""
        if self._token_counter:
            return self._token_counter(msg.content)
        return max(1, len(msg.content) // 4)

    def manage(self, messages: List[Message], config: Dict[str, Any]) -> List[Message]:
        max_tokens = config.get("max_tokens", self._max_tokens)

        if not messages:
            return []

        # system 消息必须保留
        system_msgs = [m for m in messages if m.role == "system"]
        other_msgs = [m for m in messages if m.role != "system"]

        system_tokens = sum(self._estimate_tokens(m) for m in system_msgs)
        budget_remaining = max_tokens - system_tokens

        # 从末尾向前累积非 system 消息
        kept: List[Message] = []
        current_tokens = 0
        for msg in reversed(other_msgs):
            t = self._estimate_tokens(msg)
            if current_tokens + t > budget_remaining:
                break
            kept.insert(0, msg)
            current_tokens += t

        return system_msgs + kept


class SummaryBuffer(ShortTermStrategy):
    """摘要缓冲：压缩旧消息为摘要。"""

    def __init__(self, max_messages: int = 30) -> None:
        self._max_messages = max_messages

    def manage(self, messages: List[Message], config: Dict[str, Any]) -> List[Message]:
        max_msgs = config.get("max_messages", self._max_messages)
        if len(messages) <= max_msgs:
            return list(messages)

        system_msgs = [m for m in messages if m.role == "system"]
        regular_msgs = [m for m in messages if m.role != "system"]

        keep_recent = max_msgs - len(system_msgs) - 1
        if keep_recent < 1:
            keep_recent = 1

        recent = regular_msgs[-keep_recent:]
        old = regular_msgs[:-keep_recent]

        if old:
            summary_content = "对话历史摘要: " + "; ".join(
                f"[{m.role}] {m.content[:80]}" for m in old[-5:]
            )
            summary_msg = Message(role="system", content=summary_content)
            return system_msgs + [summary_msg] + recent

        return system_msgs + recent


# ============================================================================
# 长期记忆存储策略
# ============================================================================


class InMemoryStore(LongTermStore):
    """
    内存存储实现。

    [FIX-H6] 不使用 FTS5，检索用 Python 层打分。
    [FIX-L2] 新增 get(trace_id) 方法。
    [FIX-L3] 新增 update_strengths_batch 方法。
    """

    def __init__(self) -> None:
        self._traces: Dict[str, MemoryTrace] = {}
        self._archived: Dict[str, MemoryTrace] = {}

    def store(self, trace: MemoryTrace) -> None:
        self._traces[trace.trace_id] = trace

    def retrieve(
        self, query: str, namespace: str, top_k: int, **kwargs: Any
    ) -> List[MemoryTrace]:
        kind_filter = kwargs.get("kind")
        active = [
            t for t in self._traces.values()
            if (t.namespace == namespace or namespace == "all")
            and not t.forgotten
            and (kind_filter is None or t.kind == kind_filter)
        ]
        if not active:
            return []

        if not query.strip():
            active.sort(key=lambda t: t.strength, reverse=True)
            return active[:top_k]

        query_words = set(query.lower().split())
        scored: List[Tuple[float, MemoryTrace]] = []
        for t in active:
            content_words = set(t.content.lower().split())
            overlap = query_words & content_words
            tag_match = any(
                qw in tg.lower() for qw in query_words for tg in t.tags
            )
            score = len(overlap) * 2.0 + (3.0 if tag_match else 0)
            if score > 0:
                scored.append((score * t.strength, t))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:top_k]]

    @property
    def all_active(self) -> List[MemoryTrace]:
        """返回所有活跃记忆（过滤 forgotten=True）。"""
        return [t for t in self._traces.values() if not t.forgotten]

    def get(self, trace_id: str) -> Optional[MemoryTrace]:
        """[FIX-L2] O(1) 按 ID 获取。"""
        return self._traces.get(trace_id)

    def update_strength(self, trace_id: str, strength: float) -> None:
        if trace_id in self._traces:
            self._traces[trace_id] = self._traces[trace_id]._replace(
                strength=min(1.0, max(0.0, strength)),
            )

    def update_strengths_batch(self, updates: List[tuple]) -> None:
        """[FIX-L3] 批量更新。"""
        for trace_id, strength in updates:
            if trace_id in self._traces:
                self._traces[trace_id] = self._traces[trace_id]._replace(
                    strength=min(1.0, max(0.0, strength)),
                )

    def archive(self, trace_id: str) -> None:
        trace = self._traces.pop(trace_id, None)
        if trace:
            self._archived[trace_id] = trace._replace(strength=0.0)

    def forget(self, trace_id: str) -> None:
        self._traces.pop(trace_id, None)
        self._archived.pop(trace_id, None)

    def update_forgotten(self, trace_id: str, forgotten: bool = True) -> None:
        """[L3] 标记记忆为已遗忘（被合并后不再参与检索）。"""
        if trace_id in self._traces:
            self._traces[trace_id] = self._traces[trace_id]._replace(forgotten=forgotten)


class SQLiteStore(LongTermStore):
    """
    SQLite 持久化存储实现。

    [FIX-H6] 移除 FTS5 虚拟表。
    retrieve 先用 ORDER BY strength DESC 获取候选项，
    然后在 Python 层做 BM25 打分。
    [FIX-L2] 新增 get(trace_id) 方法。
    [FIX-L3] 新增 update_strengths_batch 方法。
    [FIX-P1-3] 线程安全：添加 threading.Lock 保护所有 DB 操作，
    启用 WAL 模式提高并发读性能。
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库表，启用 WAL 模式提升并发读性能。"""
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    full_content TEXT DEFAULT '',
                    namespace TEXT NOT NULL DEFAULT 'default',
                    strength REAL DEFAULT 1.0,
                    base_strength REAL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    last_recalled_at TEXT NOT NULL,
                    tags TEXT DEFAULT '[]',
                    kind TEXT DEFAULT 'episodic',
                    forgotten INTEGER DEFAULT 0,
                    metadata TEXT DEFAULT '{}',
                    archived INTEGER DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_traces_namespace ON traces(namespace);
                CREATE INDEX IF NOT EXISTS idx_traces_strength ON traces(strength DESC);
                CREATE INDEX IF NOT EXISTS idx_traces_archived ON traces(archived);
            """)
            # 兼容旧表：尝试添加新增列（失败代表列已存在或被旧版忽略）
            for column in ("base_strength REAL DEFAULT 1.0", "kind TEXT DEFAULT 'episodic'", "forgotten INTEGER DEFAULT 0"):
                try:
                    self._conn.execute(f"ALTER TABLE traces ADD COLUMN {column}")
                except sqlite3.OperationalError:
                    pass
            self._conn.commit()

    def store(self, trace: MemoryTrace) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO traces
                   (trace_id, content, full_content, namespace, strength, base_strength,
                    created_at, last_recalled_at, tags, kind, forgotten, metadata, archived)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
                (
                    trace.trace_id,
                    trace.content,
                    trace.full_content,
                    trace.namespace,
                    trace.strength,
                    trace.base_strength,
                    trace.created_at,
                    trace.last_recalled_at,
                    json_mod.dumps(trace.tags),
                    trace.kind,
                    int(bool(trace.forgotten)),
                    json_mod.dumps(trace.metadata),
                ),
            )
            self._conn.commit()

    def _row_to_trace(self, row: sqlite3.Row) -> MemoryTrace:
        """将数据库行转换为 MemoryTrace。兼容旧表中可能缺失的新列。"""
        def _col(name: str, default: Any) -> Any:
            try:
                return row[name]
            except (IndexError, KeyError):
                return default

        base_str = _col("base_strength", _col("strength", 1.0))
        kind = _col("kind", "episodic") or "episodic"
        forgotten = bool(_col("forgotten", 0))
        return MemoryTrace(
            trace_id=row["trace_id"],
            content=row["content"],
            full_content=row["full_content"],
            namespace=row["namespace"],
            strength=row["strength"],
            base_strength=base_str,
            created_at=row["created_at"],
            last_recalled_at=row["last_recalled_at"],
            tags=json_mod.loads(row["tags"] or "[]"),
            kind=kind,
            forgotten=forgotten,
            metadata=json_mod.loads(row["metadata"] or "{}"),
        )

    def retrieve(
        self, query: str, namespace: str, top_k: int, **kwargs: Any
    ) -> List[MemoryTrace]:
        """
        [FIX-H6] 检索逻辑：
        1. 从 traces 表按 strength DESC 获取活跃记忆
        2. 在 Python 层做 BM25 打分
        """
        kind_filter = kwargs.get("kind")
        with self._lock:
            cursor = self._conn.execute(
                """SELECT * FROM traces
                   WHERE namespace = ? AND archived = 0
                   ORDER BY strength DESC
                   LIMIT ?""",
                (namespace, max(top_k * 5, 50)),
            )
            rows = cursor.fetchall()
        candidates = [
            t for t in (self._row_to_trace(r) for r in rows)
            if not t.forgotten and (kind_filter is None or t.kind == kind_filter)
        ]

        if not candidates:
            return []

        if not query.strip():
            return candidates[:top_k]

        # BM25 打分（纯 Python 计算，无需持锁）
        k1 = float(kwargs.get("k1", 1.5))
        b = float(kwargs.get("b", 0.75))

        doc_lengths = [len(d.content.split()) for d in candidates]
        avgdl = sum(doc_lengths) / max(1, len(doc_lengths))
        query_terms = query.lower().split()
        doc_terms = [d.content.lower().split() for d in candidates]
        N = len(candidates)

        scores: List[Tuple[float, MemoryTrace]] = []
        for i, doc in enumerate(candidates):
            dl = doc_lengths[i]
            score = 0.0
            for term in query_terms:
                tf = doc_terms[i].count(term)
                if tf == 0:
                    continue
                df = sum(1 for dt in doc_terms if term in dt)
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * dl / avgdl)
                score += idf * numerator / denominator
            for term in query_terms:
                if any(term in tag.lower() for tag in doc.tags):
                    score += 3.0
            if score > 0:
                scores.append((score * doc.strength, doc))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scores[:top_k]]

    @property
    def all_active(self) -> List[MemoryTrace]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM traces WHERE archived = 0 ORDER BY strength DESC"
            )
            return [
                t for t in (self._row_to_trace(r) for r in cursor.fetchall())
                if not t.forgotten
            ]

    def get(self, trace_id: str) -> Optional[MemoryTrace]:
        """[FIX-L2] O(1) 按 ID 获取。"""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
            )
            row = cursor.fetchone()
        return self._row_to_trace(row) if row else None

    def update_strength(self, trace_id: str, strength: float) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE traces SET strength = ? WHERE trace_id = ?",
                (min(1.0, max(0.0, strength)), trace_id),
            )
            self._conn.commit()

    def update_strengths_batch(self, updates: List[tuple]) -> None:
        """[FIX-L3] 批量更新（单次事务）。"""
        with self._lock:
            self._conn.execute("BEGIN")
            for trace_id, strength in updates:
                self._conn.execute(
                    "UPDATE traces SET strength = ? WHERE trace_id = ?",
                    (min(1.0, max(0.0, strength)), trace_id),
                )
            self._conn.commit()

    def update_forgotten(self, trace_id: str, forgotten: bool = True) -> None:
        """[L3] 标记记忆为已遗忘（被合并后不再参与检索）。"""
        with self._lock:
            self._conn.execute(
                "UPDATE traces SET forgotten = ? WHERE trace_id = ?",
                (int(bool(forgotten)), trace_id),
            )
            self._conn.commit()

    def archive(self, trace_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE traces SET archived = 1, strength = 0.0 WHERE trace_id = ?",
                (trace_id,),
            )
            self._conn.commit()

    def forget(self, trace_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM traces WHERE trace_id = ?", (trace_id,))
            self._conn.commit()

    def close(self) -> None:
        """关闭数据库连接。"""
        with self._lock:
            self._conn.close()


class VectorStore(LongTermStore):
    """
    向量存储接口实现（占位）。

    当前为内存存储 + 关键词匹配，接口预留给向量检索扩展。
    """

    def __init__(self, embedding_fn: Optional[Callable] = None) -> None:
        self._embedding_fn = embedding_fn
        self._traces: Dict[str, MemoryTrace] = {}
        self._archived: Dict[str, MemoryTrace] = {}

    def store(self, trace: MemoryTrace) -> None:
        self._traces[trace.trace_id] = trace

    def retrieve(
        self, query: str, namespace: str, top_k: int, **kwargs: Any
    ) -> List[MemoryTrace]:
        filtered = [
            t for t in self._traces.values()
            if t.namespace == namespace and t.strength > 0
        ]
        if self._embedding_fn:
            query_words = set(query.lower().split())
            scored = []
            for t in filtered:
                content_words = set(t.content.lower().split())
                overlap = len(query_words & content_words)
                if overlap > 0:
                    scored.append((overlap * t.strength, t))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [t for _, t in scored[:top_k]]
        return filtered[:top_k]

    @property
    def all_active(self) -> List[MemoryTrace]:
        return list(self._traces.values())

    def get(self, trace_id: str) -> Optional[MemoryTrace]:
        """[FIX-L2] O(1) 按 ID 获取。"""
        trace = self._traces.get(trace_id)
        if trace:
            return trace
        return self._archived.get(trace_id)

    def update_strength(self, trace_id: str, strength: float) -> None:
        if trace_id in self._traces:
            self._traces[trace_id] = self._traces[trace_id]._replace(strength=strength)

    def update_strengths_batch(self, updates: List[tuple]) -> None:
        """[FIX-L3] 批量更新。"""
        for trace_id, strength in updates:
            if trace_id in self._traces:
                self._traces[trace_id] = self._traces[trace_id]._replace(strength=strength)

    def archive(self, trace_id: str) -> None:
        """[FIX-P0-5] 归档到独立字典，all_active 不再包含已归档记忆。"""
        trace = self._traces.pop(trace_id, None)
        if trace:
            self._archived[trace_id] = trace._replace(strength=0.0)

    def forget(self, trace_id: str) -> None:
        self._traces.pop(trace_id, None)
        self._archived.pop(trace_id, None)


# ============================================================================
# 压缩策略
# ============================================================================


class ThresholdCompress(CompressStrategy):
    """基于阈值的压缩：当消息数超过阈值时触发压缩。"""

    def __init__(self, threshold: int = 20) -> None:
        self._threshold = threshold

    def compress(self, messages: List[Message], **kwargs: Any) -> List[Message]:
        threshold = kwargs.get("threshold", self._threshold)
        if len(messages) <= threshold:
            return list(messages)
        system = [m for m in messages if m.role == "system"]
        recent = [m for m in messages if m.role != "system"][-10:]
        return system + recent


class LLMSummarize(CompressStrategy):
    """LLM 驱动的压缩策略。需要外部传入 summarizer。"""

    def __init__(self) -> None:
        self._summarizer: Optional[Callable] = None

    def set_summarizer(self, fn: Callable) -> None:
        self._summarizer = fn

    def compress(self, messages: List[Message], **kwargs: Any) -> List[Message]:
        summarizer = kwargs.get("summarizer", self._summarizer)
        if not summarizer or len(messages) < 5:
            return list(messages)

        text = "\n".join(f"[{m.role}] {m.content}" for m in messages[:-3])
        summary = summarizer(text[:4000])
        return [Message(role="system", content=f"摘要: {summary}")] + list(messages[-3:])


class NoCompress(CompressStrategy):
    """不压缩。"""

    def compress(self, messages: List[Message], **kwargs: Any) -> List[Message]:
        return list(messages)


# ============================================================================
# 衰减策略
# ============================================================================


class EbbinghausDecay(DecayStrategy):
    """
    基于艾宾浩斯遗忘曲线的衰减策略。

    公式: strength(t) = strength(0) * e^(-t / S)
    其中 S 是记忆稳定性常数。

    [FIX-H1] compute_decay 仅基于 created_at 和 current_time 计算瞬态值，
    不修改 trace 对象，不写回存储。
    """

    def __init__(self, stability: float = 86400.0, archive_threshold: float = 0.1) -> None:
        """
        Args:
            stability: 记忆稳定性常数（秒），默认 1 天。
            archive_threshold: 归档阈值。
        """
        self._stability = stability
        self._archive_threshold = archive_threshold

    def compute_decay(self, trace: MemoryTrace, current_time: datetime) -> float:
        """
        计算衰减后的强度（瞬态值，不写回）。

        [FIX-H1] 只计算不写回。衰减的写回统一由 LayeredContextManager._apply_decay() 处理。
        [FIX-P0-1] 使用 base_strength 而非当前 strength 作为衰减基准，
        避免多次 apply_decay 导致 exp(-elapsed/stability)^N 加速衰减。
        """
        created = datetime.fromisoformat(trace.created_at)
        elapsed = (current_time - created).total_seconds()
        # [FIX-P0-1] base_strength 是初始强度，不随衰减改变
        decayed = trace.base_strength * math.exp(-elapsed / self._stability)
        return max(0.0, decayed)

    def should_archive(self, trace: MemoryTrace, threshold: Optional[float] = None) -> bool:
        t = threshold or self._archive_threshold
        return trace.strength < t


# ============================================================================
# 检索策略
# ============================================================================


class KeywordRetrieval(RetrievalStrategy):
    """关键词检索。"""

    def search(
        self, query: str, documents: List[MemoryTrace], top_k: int, **kwargs: Any
    ) -> List[MemoryTrace]:
        if not documents:
            return []

        query_words = set(query.lower().split())
        scored: List[Tuple[float, MemoryTrace]] = []

        for doc in documents:
            doc_words = set(doc.content.lower().split())
            overlap = len(query_words & doc_words)
            tag_match = any(
                qw in tag.lower() for qw in query_words for tag in doc.tags
            )
            score = float(overlap) + (3.0 if tag_match else 0.0)
            if score > 0:
                scored.append((score * doc.strength, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:top_k]]


class BM25Retrieval(RetrievalStrategy):
    """
    BM25 检索策略。

    标准 BM25 算法实现，参数可调。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b

    def search(
        self, query: str, documents: List[MemoryTrace], top_k: int, **kwargs: Any
    ) -> List[MemoryTrace]:
        if not documents:
            return []

        k1 = kwargs.get("k1", self._k1)
        b = kwargs.get("b", self._b)

        doc_lengths = [len(d.content.split()) for d in documents]
        avgdl = sum(doc_lengths) / max(1, len(doc_lengths))

        query_terms = query.lower().split()
        doc_terms: List[List[str]] = [
            d.content.lower().split() for d in documents
        ]

        N = len(documents)
        scores: List[Tuple[float, MemoryTrace]] = []

        for i, doc in enumerate(documents):
            dl = doc_lengths[i]
            score = 0.0

            for term in query_terms:
                tf = doc_terms[i].count(term)
                if tf == 0:
                    continue
                df = sum(1 for dt in doc_terms if term in dt)
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * dl / avgdl)
                score += idf * numerator / denominator

            for term in query_terms:
                if any(term in tag.lower() for tag in doc.tags):
                    score += 3.0

            if score > 0:
                scores.append((score * doc.strength, doc))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scores[:top_k]]


# ============================================================================
# 注入策略
# ============================================================================


class SummaryInjection(InjectionStrategy):
    """
    摘要注入策略（L4）。

    [FIX-H2] 契约修正：
    - inject 只返回格式化后的记忆上下文文本，不负责替换 system prompt 中的占位符。
    - 占位符替换由 ContextManager.build_context 负责。
    - 移除了死代码分支（'placeholder in placeholder' 永远为真）。
    """

    def __init__(self, min_strength: float = 0.3) -> None:
        self._min_strength = min_strength

    def inject(
        self, traces: List[MemoryTrace], **kwargs: Any
    ) -> str:
        """
        生成记忆上下文字符串。

        [FIX-H2] 只返回格式化文本，不处理占位符替换。

        Args:
            traces: 检索到的记忆列表。

        Returns:
            格式化后的记忆上下文文本。
        """
        min_strength = kwargs.get("min_strength", self._min_strength)

        # 过滤低置信度记忆
        high_confidence = [t for t in traces if t.strength >= min_strength]

        if not high_confidence:
            return "（无相关记忆）"

        # 按命名空间分组
        by_ns: Dict[str, List[MemoryTrace]] = {}
        for t in high_confidence:
            by_ns.setdefault(t.namespace, []).append(t)

        parts: List[str] = ["## 相关记忆上下文\n"]
        for ns, ns_traces in by_ns.items():
            parts.append(f"### {ns}")
            for t in ns_traces:
                parts.append(f"- [{t.strength:.2f}] {t.content} (ref: {t.trace_id})")
            parts.append("")

        return "\n".join(parts)


# ============================================================================
# 沉淀策略
# ============================================================================


class LLMExtraction(ExtractionStrategy):
    """
    LLM 驱动的记忆沉淀策略 (L5)。

    从对话中抽取可长期保存的记忆。

    阶段：
    1. 抽取 (extract): 从消息中识别值得长期记忆的信息
    2. 编码 (encode): 生成 MemoryTrace 实例
    3. 入库 (store): 存入长期记忆存储

    每个阶段独立可替换。

    [FIX-L5] 新增去重逻辑：对 LLM 返回的结果做相似度校验。
    """

    def __init__(self, llm_callable: Optional[Any] = None) -> None:
        """
        Args:
            llm_callable: LLM 调用函数，签名 (prompt: str) -> str。
        """
        self._llm_callable = llm_callable

    def extract(
        self,
        messages: List[Message],
        existing_traces: List[MemoryTrace],
        **kwargs: Any,
    ) -> List[MemoryTrace]:
        """
        从对话中抽取可沉淀的记忆。

        如果没有 LLM callable，使用简单的启发式规则。
        """
        llm_callable = kwargs.get("llm_callable", self._llm_callable)
        namespace = kwargs.get("namespace", "default")

        if not messages:
            return []

        if llm_callable:
            return self._extract_with_llm(messages, existing_traces, llm_callable, namespace)
        else:
            return self._extract_heuristic(messages, existing_traces, namespace)

    @staticmethod
    def _is_duplicate(content: str, existing: List[MemoryTrace]) -> bool:
        """
        [FIX-L5] 去重检查：
        1. 精确匹配
        2. 前 50 字符前缀匹配
        3. Jaccard 相似度 > 0.8 视为重复
        """
        if not content:
            return True

        content_lower = content.lower().strip()
        content_words = set(content_lower.split())

        for ex in existing:
            ex_lower = ex.content.lower().strip()

            # 精确匹配
            if content_lower == ex_lower:
                return True

            # 前缀匹配
            if content_lower[:50] == ex_lower[:50]:
                return True

            # Jaccard 相似度
            ex_words = set(ex_lower.split())
            if content_words and ex_words:
                intersection = content_words & ex_words
                union = content_words | ex_words
                jaccard = len(intersection) / len(union)
                if jaccard > 0.8:
                    return True

        return False

    def _extract_with_llm(
        self,
        messages: List[Message],
        existing: List[MemoryTrace],
        llm_callable: Any,
        namespace: str,
    ) -> List[MemoryTrace]:
        """使用 LLM 进行记忆抽取。[FIX-L5] 加入去重。"""
        existing_summaries = "\n".join(
            f"- {t.content}" for t in existing[:10]
        ) or "（无已有记忆）"

        convo = "\n".join(
            f"[{m.role}] {m.content[:200]}" for m in messages[-20:]
        )

        prompt = f"""你是一个记忆抽取代理。请从以下对话中提取可长期保存的信息。

已有长期记忆：
{existing_summaries}

最近的对话：
{convo}

请以 JSON 数组格式输出抽取的记忆，每条包含：
- content: 记忆摘要（1-2句话）
- is_new: 是否为新的、不重复的信息 (true/false)
- importance: 重要性评分 (0.0-1.0)

示例输出：
[{{"content": "用户喜欢用 Python 做数据分析", "is_new": true, "importance": 0.8}}]

请只输出 JSON 数组，不要其他文字。"""

        try:
            response_text = llm_callable(prompt)

            # 尝试从响应中提取 JSON
            start = response_text.find("[")
            end = response_text.rfind("]") + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                items = json_mod.loads(json_str)
            else:
                return []

            # [FIX-L5] 去重创建
            traces: List[MemoryTrace] = []
            for item in items:
                content = item.get("content", "")
                if not item.get("is_new", True):
                    continue
                if self._is_duplicate(content, existing):
                    logger.debug("跳过重复记忆: %s", content[:50])
                    continue
                if self._is_duplicate(content, traces):
                    continue
                traces.append(MemoryTrace.create(
                    content=content,
                    namespace=namespace,
                    strength=item.get("importance", 0.5),
                    tags=item.get("tags", []),
                ))
            return traces

        except Exception as exc:
            logger.warning("LLM 记忆抽取失败: %s", exc)
            return []

    def _extract_heuristic(
        self,
        messages: List[Message],
        existing: List[MemoryTrace],
        namespace: str,
    ) -> List[MemoryTrace]:
        """
        启发式记忆抽取。[FIX-L5] 加入去重。

        规则：
        1. 用户消息包含"记住"、"记录"等关键词
        2. assistant 消息中包含决策、结论等
        """
        traces: List[MemoryTrace] = []
        existing_contents = {t.content for t in existing}

        remember_keywords = ["记住", "记录", "别忘了", "重要的是", "关键", "记住这"]
        decision_keywords = ["决定", "结论", "总结", "方案", "最终选择"]

        for msg in messages[-30:]:
            content = msg.content.strip()
            if not content or len(content) < 20:
                continue

            if msg.role == "user":
                if any(kw in content for kw in remember_keywords):
                    summary = content[:150]
                    if summary not in existing_contents:
                        if self._is_duplicate(summary, traces):
                            continue
                        traces.append(MemoryTrace.create(
                            content=summary,
                            namespace=namespace,
                            strength=0.7,
                            tags=["user_memory", "explicit"],
                        ))

            elif msg.role == "assistant":
                if any(kw in content for kw in decision_keywords):
                    summary = f"[决策] {content[:150]}"
                    if summary not in existing_contents:
                        if self._is_duplicate(summary, traces):
                            continue
                        traces.append(MemoryTrace.create(
                            content=summary,
                            namespace=namespace,
                            strength=0.5,
                            tags=["decision"],
                        ))

        return traces