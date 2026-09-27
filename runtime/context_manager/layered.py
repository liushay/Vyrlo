"""
LayeredContextManager — ContextManager 接口的默认实现。

实现五层记忆系统的完整逻辑。

[FIX-H1] 移除 retrieve 中的衰减逻辑。衰减统一由 apply_decay() 方法执行。
[FIX-L1] _apply_decay 现在使用 store.get() 直接获取（不再通过 retrieve 间接 O(N*M)）。
[FIX-H3] _manage_context_window 从末尾向前累积，修复截断方向。
[FIX-L4] _track_cost 现在记录每次调用的绝对成本，支持查询总成本。
[FIX-L6] 明确钩子映射：AFTER_ITERATION -> extract_memories，ON_EXIT_LOOP -> apply_decay。
[FIX-D1] set_markdown_root / sync_from_markdown 不再抛 NotImplementedError，降级为记录警告。
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from .interface import (
    ContextManager,
    ContextSnapshot,
    DecayStrategy,
    ExtractionStrategy,
    InjectionStrategy,
    LongTermStore,
    Message,
    MemoryTrace,
    RetrievalStrategy,
    ShortTermStrategy,
    CompressStrategy,
)
from .strategies import (
    InMemoryStore,
    SlidingWindow,
    NoCompress,
    EbbinghausDecay,
    BM25Retrieval,
    SummaryInjection,
    LLMExtraction,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# [FIX-L4] 成本追踪器
# ---------------------------------------------------------------------------


class CostTracker:
    """
    独立的内存成本追踪器。

    [FIX-L4] 记录每次调用的绝对成本，支持查询总成本和按模型分组。
    不依赖 ContextManager 的任何内部状态。
    """

    def __init__(self) -> None:
        self._costs: List[Dict[str, Any]] = []
        self._total: float = 0.0

    def track(self, model: str, cost: float, tokens: int = 0, **meta: Any) -> None:
        """[FIX-L4] 记录一次调用的绝对成本。"""
        entry: Dict[str, Any] = {
            "model": model,
            "cost": cost,
            "tokens": tokens,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **meta,
        }
        self._costs.append(entry)
        self._total += cost

    def get_total(self) -> float:
        """获取总成本。"""
        return self._total

    def get_by_model(self) -> Dict[str, float]:
        """按模型分组统计成本。"""
        groups: Dict[str, float] = {}
        for c in self._costs:
            model = c.get("model", "unknown")
            groups[model] = groups.get(model, 0.0) + c.get("cost", 0.0)
        return groups

    def snapshot(self) -> Dict[str, Any]:
        """返回当前成本快照。"""
        return {
            "total": self._total,
            "by_model": self.get_by_model(),
            "count": len(self._costs),
        }


# ============================================================================
# LayeredContextManager
# ============================================================================


class LayeredContextManager(ContextManager):
    """
    分层上下文管理器（默认实现）。

    五层记忆系统的完整编排：

    L1: MemoryTrace 数据模型（不可变）
    L2: 衰减与分层（DecayStrategy + 归档标记）
    L3: 检索（RetrievalStrategy + LongTermStore）
    L4: 注入（InjectionStrategy + system prompt 占位符替换）
    L5: 沉淀（ExtractionStrategy + 去重入库）

    [FIX-H1] 衰减不再内嵌在 retrieve 中，统一由 apply_decay() 执行。
    [FIX-L1] _apply_decay 使用 store.get() 直接获取记忆，不再通过 retrieve。
    [FIX-H3] _manage_context_window 从末尾向前累积。
    [FIX-L4] 使用独立的 CostTracker 记录成本。
    [FIX-L6] 明确钩子映射文档：
             - AFTER_ITERATION: extract_memories()
             - ON_EXIT_LOOP: apply_decay()
    [FIX-D1] markdown 方法降级为警告日志。
    """

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------

    def __init__(
        self,
        long_term_store: Optional[LongTermStore] = None,
        short_term_strategy: Optional[ShortTermStrategy] = None,
        compress_strategy: Optional[CompressStrategy] = None,
        decay_strategy: Optional[DecayStrategy] = None,
        retrieval_strategy: Optional[RetrievalStrategy] = None,
        injection_strategy: Optional[InjectionStrategy] = None,
        extraction_strategy: Optional[ExtractionStrategy] = None,
        max_working_memory: int = 50,
        llm_callable: Optional[Any] = None,
    ) -> None:
        """
        初始化分层上下文管理器。

        Args:
            long_term_store: 长期记忆存储 (L2-L3)。
            short_term_strategy: 短期记忆策略（窗口/TokenBudget等）。
            compress_strategy: 压缩策略。
            decay_strategy: 衰减策略 (L2)。
            retrieval_strategy: 检索策略 (L3)。
            injection_strategy: 注入策略 (L4)。
            extraction_strategy: 沉淀策略 (L5)。
            max_working_memory: 工作记忆最大消息数。
            llm_callable: 可选 LLM 调用函数 (prompt: str) -> str，
                          用于 episodic→semantic 记忆合并（_consolidate_episodic）。
        """
        # 存储与策略
        # [D003 同类] 用 `is not None` 判定"是否提供"：LongTermStore 的无参
        # 实现当前虽无 __len__，但一旦未来实现带 __len__（如空的向量库），
        # `or InMemoryStore()` 会静默丢弃调用方注入的空 store，重演 D003。
        self._long_term_store = (
            long_term_store if long_term_store is not None else InMemoryStore()
        )
        self._short_term_strategy = short_term_strategy or SlidingWindow()
        self._compress_strategy = compress_strategy or NoCompress()
        self._decay_strategy = decay_strategy or EbbinghausDecay()
        self._retrieval_strategy = retrieval_strategy or BM25Retrieval()
        self._injection_strategy = injection_strategy or SummaryInjection()
        self._extraction_strategy = extraction_strategy or LLMExtraction()

        # 工作记忆
        self._working_memory: List[Message] = []
        self._max_working_memory = max_working_memory

        # [FIX-L4] 独立的成本追踪器
        self._cost_tracker = CostTracker()

        # [FIX-L3] 待强化队列 — 批量处理策略
        self._strengthen_queue: List[str] = []
        self._strengthen_batch_size = 10
        self._strengthen_delta = 0.05

        # [FIX-D1] Markdown 真相源路径（可选）
        self._markdown_root: Optional[str] = None

        # [L3] episodic→semantic 合并用的 LLM 调用函数
        self._llm_callable = llm_callable

    # ------------------------------------------------------------------
    # 衰减管理 [FIX-H1 / FIX-L1]
    # ------------------------------------------------------------------

    def apply_decay(self, namespace: Optional[str] = None) -> int:
        """
        [FIX-H1 / FIX-L1] 对长期记忆应用衰减。

        衰减只在此方法中执行，不在 retrieve 中执行。
        此方法由 ON_EXIT_LOOP 钩子触发。

        实现：
        1. 获取所有活跃记忆（按命名空间过滤）
        2. 对每条记忆计算衰减值
        3. 批量写回
        4. 强度低于阈值的归档

        Returns:
            被归档的记忆数量。
        """
        ns = namespace or "default"
        current_time = datetime.now(timezone.utc)

        # 获取所有活跃记忆 — 直接通过 store 获取，不通过 retrieve
        if hasattr(self._long_term_store, 'all_active'):
            all_traces = self._long_term_store.all_active  # type: ignore[attr-defined]
        else:
            # 回退：通过 retrieve 获取（但这不是最优路径）
            all_traces = self._long_term_store.retrieve("", ns, 1000)

        # 按命名空间过滤
        traces = [t for t in all_traces if t.namespace == ns]

        if not traces:
            return 0

        # 计算衰减值
        strength_updates: List[Tuple[str, float]] = []
        to_archive: List[str] = []
        archived_count = 0

        for trace in traces:
            new_strength = self._decay_strategy.compute_decay(trace, current_time)

            if self._decay_strategy.should_archive(
                trace._replace(strength=new_strength)
            ):
                to_archive.append(trace.trace_id)
                archived_count += 1
            else:
                strength_updates.append((trace.trace_id, new_strength))

        # 批量归档
        for trace_id in to_archive:
            self._long_term_store.archive(trace_id)

        # 批量更新强度（使用 update_strengths_batch 避免 N 次写入）
        if strength_updates:
            self._long_term_store.update_strengths_batch(strength_updates)

        logger.debug(
            "应用衰减完成: 处理 %d 条记忆, 归档 %d 条",
            len(traces), archived_count,
        )
        return archived_count

    # ------------------------------------------------------------------
    # 记忆沉淀 [FIX-L6]
    # ------------------------------------------------------------------

    def extract_memories(self, namespace: Optional[str] = None) -> List[MemoryTrace]:
        """
        [FIX-L6] 从工作记忆中抽取可沉淀的记忆。

        钩子映射: AFTER_ITERATION → extract_memories()
        此方法由中间件在 AFTER_ITERATION 钩子中调用。

        Returns:
            新生成的 MemoryTrace 列表。
        """
        ns = namespace or "default"

        # 获取现有长期记忆用于去重
        existing = self._long_term_store.retrieve("", ns, 100)

        # 执行抽取
        new_traces = self._extraction_strategy.extract(
            messages=self._working_memory[-20:],
            existing_traces=existing,
            namespace=ns,
        )

        # 入库
        for trace in new_traces:
            self._long_term_store.store(trace)

        if new_traces:
            logger.info("沉淀 %d 条新记忆到命名空间 '%s'", len(new_traces), ns)

        return new_traces

    # ------------------------------------------------------------------
    # episodic→semantic 合并 [L3]
    # ------------------------------------------------------------------

    def _consolidate_episodic(self, namespace: Optional[str] = None) -> int:
        """[L3] episodic→semantic 合并。

        取命名空间下 kind="episodic" 且未 forgotten 的最旧 10 条记忆，
        经 LLM 合并为 1 条 semantic 记忆；原 10 条标记 forgotten=True。

        - 数量 < 10 时直接返回 0。
        - LLM 调用失败时不动任何 trace，返回 0。
        - 合并成功返回 1（新生成的 semantic 条数）。

        Args:
            namespace: 命名空间，None 时用 "default"。

        Returns:
            合并生成的 semantic 条数（0 或 1）。
        """
        if self._llm_callable is None:
            return 0

        ns = namespace or "default"

        # 取该 namespace 下 kind="episodic" 且未 forgotten 的 traces
        if hasattr(self._long_term_store, 'all_active'):
            all_traces = self._long_term_store.all_active  # type: ignore[attr-defined]
            episodic = [
                t for t in all_traces
                if t.namespace == ns and t.kind == "episodic" and not t.forgotten
            ]
        else:
            episodic = self._long_term_store.retrieve("", ns, 10000, kind="episodic")

        if len(episodic) < 10:
            return 0

        # 最旧 10 条（按 created_at 升序）
        oldest = sorted(episodic, key=lambda t: t.created_at)[:10]

        # 调 LLM 合并
        contents = "\n".join(f"- {t.content}" for t in oldest)
        prompt = (
            "将以下情景记忆合并为一条简洁的语义记忆摘要（只输出摘要文本，不要其他内容）：\n"
            f"{contents}"
        )
        try:
            summary = self._llm_callable(prompt)
        except Exception as exc:
            logger.warning("episodic 合并的 LLM 调用失败: %s", exc)
            return 0

        summary = (summary or "").strip()
        if not summary:
            return 0

        # 入库新 semantic trace
        semantic = MemoryTrace.create(
            content=summary,
            namespace=ns,
            kind="semantic",
            tags=["semantic", "consolidated"],
        )
        self.store(semantic)

        # 旧 10 条标记 forgotten
        updater = getattr(self._long_term_store, "update_forgotten", None)
        if callable(updater):
            for t in oldest:
                updater(t.trace_id, True)

        logger.info("合并 %d 条 episodic 为 1 条 semantic (%s)", len(oldest), ns)
        return 1

    # ------------------------------------------------------------------
    # 工作记忆
    # ------------------------------------------------------------------

    def append(self, message: Message) -> None:
        """追加消息到工作记忆。"""
        self._working_memory.append(message)
        self._manage_context_window()

    def get_working_memory(self) -> List[Message]:
        """获取当前工作记忆。"""
        return list(self._working_memory)

    def clear_working_memory(self) -> None:
        """清空工作记忆。"""
        self._working_memory.clear()

    def compress(self, strategy: Optional[str] = None) -> None:
        """
        压缩工作记忆。

        Args:
            strategy: 策略名。None 则使用默认策略。
        """
        if strategy == "threshold":
            from .strategies import ThresholdCompress
            s = ThresholdCompress()
        elif strategy == "llm_summarize":
            from .strategies import LLMSummarize
            s = LLMSummarize()
        else:
            s = self._compress_strategy

        self._working_memory = s.compress(self._working_memory)

    # ------------------------------------------------------------------
    # 长期记忆
    # ------------------------------------------------------------------

    def store(self, trace: MemoryTrace) -> None:
        """存储到长期记忆。"""
        self._long_term_store.store(trace)

    def retrieve(
        self,
        query: str,
        namespace: Optional[str] = None,
        top_k: int = 5,
        kind: Optional[str] = None,
    ) -> List[MemoryTrace]:
        """
        [FIX-H1] 从长期记忆检索（不执行衰减）。

        衰减的职责已经完全移出此方法，现在只做纯检索。

        Args:
            query:     检索查询。
            namespace: 命名空间。
            top_k:     返回条数。
            kind:      可选，记忆类型过滤（"episodic" / "semantic"）。
                       非 None 时只返回该 kind。
        """
        ns = namespace or "default"

        # 获取候选记忆列表
        if hasattr(self._long_term_store, 'all_active'):
            documents = [
                t for t in self._long_term_store.all_active  # type: ignore[attr-defined]
                if t.namespace == ns and (kind is None or t.kind == kind)
            ]
        else:
            documents = self._long_term_store.retrieve(
                "", ns, 1000, kind=kind,
            )

        # 使用检索策略搜索
        results = self._retrieval_strategy.search(query, documents, top_k)

        # 检索命中后，加入待强化队列（不立即更新）
        if results:
            trace_ids = [t.trace_id for t in results]
            self.enqueue_strengthen(trace_ids)

        return results

    def update_strength(self, trace_id: str, strength: float) -> None:
        """更新记忆强度。"""
        self._long_term_store.update_strength(trace_id, strength)

    def archive(self, trace_id: str) -> None:
        """归档记忆。"""
        self._long_term_store.archive(trace_id)

    def forget(self, trace_id: str) -> None:
        """删除记忆。"""
        self._long_term_store.forget(trace_id)

    # ------------------------------------------------------------------
    # 上下文构建
    # ------------------------------------------------------------------

    def build_context(
        self,
        system_prompt: str = "",
        namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        构建完整的 LLM 调用上下文。

        [FIX-H2] 负责 system prompt 中的 {{memory_context}} 占位符替换。

        分层结构：
        1. system prompt（含注入的长期记忆上下文）
        2. 工作记忆消息列表
        """
        ns = namespace or "default"

        # 构建记忆上下文
        memory_context = ""
        if "{{memory_context}}" in system_prompt:
            # 检索相关长期记忆
            # 使用工作记忆的最后一条用户消息作为查询
            query = ""
            for msg in reversed(self._working_memory):
                if msg.role == "user":
                    query = msg.content[-200:]
                    break

            if query:
                retrieved = self.retrieve(query, namespace=ns, top_k=5)
                memory_context = self._injection_strategy.inject(retrieved)

        # 替换占位符
        system_content = system_prompt.replace("{{memory_context}}", memory_context or "（无相关记忆）")

        # 构建消息列表
        messages: List[Dict[str, Any]] = []
        messages.append({"role": "system", "content": system_content})

        for msg in self._working_memory:
            messages.append(msg.to_dict())

        return messages

    # ------------------------------------------------------------------
    # 快照
    # ------------------------------------------------------------------

    def snapshot(self) -> ContextSnapshot:
        """生成当前上下文快照。"""
        if hasattr(self._long_term_store, 'all_active'):
            ltm = self._long_term_store.all_active  # type: ignore[attr-defined]
        else:
            ltm = []

        return ContextSnapshot(
            working_memory=list(self._working_memory),
            long_term_memory=ltm,
        )

    # ------------------------------------------------------------------
    # 待强化队列 [FIX-L3]
    # ------------------------------------------------------------------

    def enqueue_strengthen(self, trace_ids: List[str]) -> None:
        """
        将检索命中的 trace_id 加入待强化队列。

        这些 ID 将在 flush_strengthen() 时批量处理。

        [FIX-P1-1] 队列达到 batch_size 时自动触发 flush。
        """
        self._strengthen_queue.extend(trace_ids)
        # [FIX-P1-1] 达到阈值时自动刷新
        if len(self._strengthen_queue) >= self._strengthen_batch_size:
            self.flush_strengthen()

    def flush_strengthen(self) -> None:
        """
        [FIX-L3] 批量刷新强化队列。

        使用 store.get() 按 ID 获取 + update_strengths_batch() 批量提交。
        不再逐条更新或通过 retrieve 间接获取。
        """
        if not self._strengthen_queue:
            return

        # 统计每个 trace_id 的命中次数
        from collections import Counter
        hit_counts = Counter(self._strengthen_queue)
        self._strengthen_queue.clear()

        # [FIX-L2] 使用 store.get() 直接获取
        updates: List[Tuple[str, float]] = []
        for trace_id, count in hit_counts.items():
            trace = self._long_term_store.get(trace_id)
            if trace and trace.strength > 0:
                # 命中次数越多，强化越多
                delta = min(0.3, self._strengthen_delta * count)
                new_strength = min(1.0, trace.strength + delta)
                updates.append((trace_id, new_strength))

        if updates:
            # [FIX-L3] 批量提交
            self._long_term_store.update_strengths_batch(updates)
            logger.debug("批量强化 %d 条记忆", len(updates))

    # ------------------------------------------------------------------
    # 成本追踪 [FIX-L4]
    # ------------------------------------------------------------------

    def track_cost(self, model: str, cost: float, tokens: int = 0, **meta: Any) -> None:
        """[FIX-L4] 记录单次 LLM 调用的绝对成本。"""
        self._cost_tracker.track(model, cost, tokens, **meta)

    def get_total_cost(self) -> float:
        """获取总调用成本。"""
        return self._cost_tracker.get_total()

    def get_cost_snapshot(self) -> Dict[str, Any]:
        """获取成本统计快照。"""
        return self._cost_tracker.snapshot()

    # ------------------------------------------------------------------
    # Markdown 真相源 [FIX-D1]
    # ------------------------------------------------------------------

    def set_markdown_root(self, path: str) -> None:
        """
        设置 Markdown 文件根目录（真相源）。

        记录路径后，可通过 sync_from_markdown() 将该目录下的 .md 文件
        同步为长期记忆 trace。
        """
        self._markdown_root = path
        logger.info("Markdown 真相源路径已设置: %s", path)

    def sync_from_markdown(self) -> int:
        """
        [L3] 从 Markdown 文件同步记忆到长期记忆索引（真相源）。

        遍历 _markdown_root 下的所有 .md 文件，每个文件生成一条 MemoryTrace：
        - content 取文件前 200 字符摘要
        - full_content 存文件路径
        - namespace 用 f"markdown:{相对目录}"
        - tags 用文件名（小写、空格转横线）

        按 full_content 路径去重：已存在同路径 trace 的文件跳过。
        文件读取失败时记录警告并继续。未配置 _markdown_root 时返回 0。

        Returns:
            本次新同步的记忆条数。
        """
        root = self._markdown_root
        if not root:
            logger.debug("Markdown 真相源未配置，跳过同步")
            return 0
        if not os.path.isdir(root):
            logger.warning("Markdown 真相源目录不存在: %s", root)
            return 0

        # 去重：收集已入库 trace 的 full_content 路径集合
        existing_paths = self._existing_full_contents()

        synced = 0
        for dirpath, _dirnames, filenames in os.walk(root):
            for filename in filenames:
                if not filename.lower().endswith(".md"):
                    continue

                file_path = os.path.join(dirpath, filename)

                # 全路径去重：已存在同路径 trace 则跳过
                if file_path in existing_paths:
                    logger.debug("Markdown 文件已存在同路径记忆，跳过: %s", file_path)
                    continue

                content = self._read_markdown_summary(file_path)
                if content is None:
                    continue

                # 相对目录（根目录本身为 "." 时归一化为空）
                rel_dir = os.path.relpath(dirpath, root)
                if rel_dir == ".":
                    rel_dir = ""
                namespace = f"markdown:{rel_dir}" if rel_dir else "markdown:"

                # 文件名（去掉 .md 后缀）转 tags：小写、空格转横线
                stem = os.path.splitext(filename)[0]
                tags = [stem.lower().replace(" ", "-")]

                trace = MemoryTrace.create(
                    content=content,
                    namespace=namespace,
                    full_content=file_path,
                    tags=tags,
                )
                self.store(trace)
                existing_paths.add(file_path)
                synced += 1

        logger.info("Markdown 同步完成: %d 条新记忆", synced)
        return synced

    def _existing_full_contents(self) -> Set[str]:
        """收集已入库 trace 的 full_content 路径集合（用于 markdown 同步去重）。"""
        if hasattr(self._long_term_store, 'all_active'):
            traces = self._long_term_store.all_active  # type: ignore[attr-defined]
        else:
            traces = self._long_term_store.retrieve("", "all", 10000)
        return {t.full_content for t in traces if t.full_content}

    def _read_markdown_summary(self, file_path: str) -> Optional[str]:
        """读取 markdown 文件前 200 字符作为摘要。失败返回 None 并记录警告。"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Markdown 文件读取失败: %s (%s)", file_path, exc)
            return None
        return text[:200]

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _manage_context_window(self) -> None:
        """
        [FIX-H3] 管理上下文窗口大小。

        使用 token 预算模型：每条消息估算 token 数（字符数 / 4），
        从最新消息向前累积，超过 max_working_tokens 时截断旧消息。
        始终保留 system 消息。
        """
        # 估算 token 数（4 字符 ≈ 1 token 的保守估算）
        def _estimate(msg: Message) -> int:
            return max(1, (len(msg.content) + len(msg.role)) // 4)

        # 总 token 预算：消息数上限 × 平均每条 ~100 token
        max_tokens = self._max_working_memory * 100

        total_tokens = sum(_estimate(m) for m in self._working_memory)
        if total_tokens <= max_tokens:
            return

        # system 消息必须保留
        system_msgs = [m for m in self._working_memory if m.role == "system"]
        other_msgs = [m for m in self._working_memory if m.role != "system"]

        system_tokens = sum(_estimate(m) for m in system_msgs)
        budget_remaining = max_tokens - system_tokens
        if budget_remaining < 50:
            budget_remaining = 50

        # 从最新消息向前累积，保留在预算内的消息
        kept: List[Message] = []
        current_tokens = 0
        for msg in reversed(other_msgs):
            t = _estimate(msg)
            if current_tokens + t > budget_remaining:
                break
            kept.insert(0, msg)
            current_tokens += t

        self._working_memory = system_msgs + kept

    # ------------------------------------------------------------------
    # 属性访问（用于测试和监控）
    # ------------------------------------------------------------------

    @property
    def long_term_store(self) -> LongTermStore:
        return self._long_term_store

    @property
    def decay_strategy(self) -> DecayStrategy:
        return self._decay_strategy

    @property
    def injection_strategy(self) -> InjectionStrategy:
        return self._injection_strategy

    @property
    def extraction_strategy(self) -> ExtractionStrategy:
        return self._extraction_strategy

    @property
    def retrieval_strategy(self) -> RetrievalStrategy:
        return self._retrieval_strategy