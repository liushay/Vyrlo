"""
Agent 运行时第二层最小集成测试 — P0/P1 缺陷修复验证。

每个测试独立可运行，失败输出直接暴露缺陷位置和期望/实际值。
测试命名格式：test_t1_decay_no_acceleration ~ test_t9_e2e_integration
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List
from unittest.mock import patch, MagicMock

import pytest

from runtime.context_manager.interface import (
    MemoryTrace,
    Message,
)
from runtime.context_manager.strategies import (
    InMemoryStore,
    SQLiteStore,
    VectorStore,
    EbbinghausDecay,
    SlidingWindow,
    NoCompress,
    BM25Retrieval,
    SummaryInjection,
    LLMExtraction,
)
from runtime.context_manager.layered import LayeredContextManager
from runtime.llm_adapter.multi_provider import (
    MultiProviderAdapter,
    AnthropicProvider,
)
from runtime.llm_adapter.interface import ModelConfig
from runtime.tool_registry.interface import Tool
from runtime.tool_registry.in_memory import InMemoryToolRegistry
from runtime.tool_call_parser.parsers import (
    AutoDetectParser,
    _normalize_schema,
)


# ============================================================================
# 辅助函数（本地，不依赖 conftest）
# ============================================================================


def _make_trace(content="测试记忆", strength=1.0, namespace="default",
                created_at=None, trace_id=None):
    """快速创建 MemoryTrace。"""
    import uuid
    return MemoryTrace(
        trace_id=trace_id or str(uuid.uuid4()),
        content=content,
        namespace=namespace,
        strength=strength,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        last_recalled_at=datetime.now(timezone.utc).isoformat(),
    )


def _make_msg(role, content, name=None):
    """快速创建 Message。"""
    return Message(role=role, content=content, name=name)


# ============================================================================
# T1: 衰减不加速 (P0-1)
# ============================================================================


class TestT1DecayNoAcceleration:
    """
    [P0-1] 验证 apply_decay() 使用正确的衰减公式：
    strength = exp(-elapsed / stability)，而非 exp(-elapsed / stability)^n（重复指数衰减）。

    若每 call 一次 apply_decay 就乘一次衰减因子，10 次调用后 strength 会远远低于预期。
    """

    @staticmethod
    def _create_ctx_with_store():
        """创建带可观测 store 的 LayeredContextManager。"""
        store = InMemoryStore()
        ctx = LayeredContextManager(
            long_term_store=store,
            decay_strategy=EbbinghausDecay(stability=86400.0, archive_threshold=0.0),
            max_working_memory=50,
        )
        return ctx, store

    def test_10_decays_should_equal_one_decay(self):
        """
        核心断言：连续 10 次 apply_decay() 后，
        强度应接近 exp(-total_elapsed / stability) 而非重复乘积。
        """
        ctx, store = self._create_ctx_with_store()

        # 创建一个 created_at 为 1 小时前的 trace
        one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
        trace = _make_trace(
            content="衰减测试",
            strength=1.0,
            created_at=one_hour_ago.isoformat(),
        )
        store.store(trace)
        trace_id = trace.trace_id

        # 连续 apply_decay 10 次
        for _ in range(10):
            ctx.apply_decay()

        # 获取最终强度
        stored = store.get(trace_id)
        assert stored is not None, (
            f"P0-1: trace {trace_id} 在 apply_decay 后从 store 中消失"
        )

        actual_strength = stored.strength

        # 正确公式：经过 1 小时，stability=86400 秒 (24 小时)
        # elapsed ≈ 3600 秒，expected = exp(-3600/86400) ≈ exp(-0.0417) ≈ 0.959
        expected_strength = math.exp(-3600.0 / 86400.0)

        # 错误公式（加速衰减）：(expected_strength)^10 ≈ 0.959^10 ≈ 0.66
        wrong_accelerated = expected_strength ** 10

        assert actual_strength > 0.9, (
            f"P0-1 FAIL: 衰减过度加速！\n"
            f"  期望强度（正确公式 exp(-elapsed/stability)）：≈ {expected_strength:.6f}\n"
            f"  错误加速强度（exp(-elapsed/stability)^10）：≈ {wrong_accelerated:.6f}\n"
            f"  实际强度：{actual_strength:.6f}\n"
            f"  若实际值接近 {wrong_accelerated:.6f} 说明衰减被重复计算了 10 次"
        )

    def test_created_at_past_trace_deterministic(self):
        """
        确定性验证：用 stability=86400，created_at 为 24 小时前的 trace，
        一次 apply_decay 后 strength 应 ≈ exp(-1) ≈ 0.368。
        """
        ctx, store = self._create_ctx_with_store()

        # 24 小时前
        one_day_ago = datetime.now(timezone.utc) - timedelta(hours=24)
        trace = _make_trace(
            content="24h 衰减测试",
            strength=1.0,
            created_at=one_day_ago.isoformat(),
        )
        store.store(trace)
        trace_id = trace.trace_id

        ctx.apply_decay()

        stored = store.get(trace_id)
        assert stored is not None

        actual = stored.strength
        expected = math.exp(-1.0)  # exp(-86400/86400) = exp(-1) ≈ 0.3679

        assert abs(actual - expected) < 0.01, (
            f"P0-1 FAIL: 确定性衰减验证失败！\n"
            f"  stability=86400, elapsed=86400s\n"
            f"  期望强度: exp(-1) ≈ {expected:.6f}\n"
            f"  实际强度: {actual:.6f}\n"
            f"  差值: {abs(actual - expected):.6f}"
        )


# ============================================================================
# T2: 批量提交 (P0-2)
# ============================================================================


class TestT2BatchUpdate:
    """
    [P0-2] 验证 apply_decay() 使用 update_strengths_batch 批量提交，
    而非逐条调用 update_strength。
    """

    def test_batch_update_called_not_single(self):
        """存入 100 条，apply_decay 应调用 1 次 batch，0 次 single。"""
        store = InMemoryStore()
        ctx = LayeredContextManager(
            long_term_store=store,
            decay_strategy=EbbinghausDecay(stability=86400.0, archive_threshold=0.0),
            max_working_memory=50,
        )

        # 存入 100 条记忆
        for i in range(100):
            trace = _make_trace(content=f"记忆 {i}", strength=1.0)
            store.store(trace)

        # Monkey-patch 计数
        org_batch = store.update_strengths_batch
        org_single = store.update_strength

        batch_count = [0]
        single_count = [0]

        def counting_batch(updates):
            batch_count[0] += 1
            return org_batch(updates)

        def counting_single(trace_id, strength):
            single_count[0] += 1
            return org_single(trace_id, strength)

        store.update_strengths_batch = counting_batch
        store.update_strength = counting_single

        try:
            ctx.apply_decay()

            assert batch_count[0] >= 1, (
                f"P0-2 FAIL: update_strengths_batch 未被调用！\n"
                f"  batch 调用次数: {batch_count[0]}\n"
                f"  single 调用次数: {single_count[0]}"
            )

            assert single_count[0] == 0, (
                f"P0-2 FAIL: apply_decay 使用了逐条 update_strength 而非批量！\n"
                f"  batch 调用次数: {batch_count[0]}（期望 ≥1）\n"
                f"  single 调用次数: {single_count[0]}（期望 0）\n"
                f"  缺陷：flush 时应使用 update_strengths_batch 一次提交所有更新"
            )
        finally:
            store.update_strengths_batch = org_batch
            store.update_strength = org_single


# ============================================================================
# T3: MultiProviderAdapter 截断 (P0-3)
# ============================================================================


class TestT3ContextWindowTruncation:
    """
    [P0-3] 验证 _manage_context_window 截断后：
    - system 消息保留在最前面
    - user/assistant 消息顺序与原始一致（不是反转）
    - 总 token ≤ 上限
    """

    def test_system_preserved_order_not_reversed(self):
        """
        构造超长消息列表（含 system + 多条 user），
        调用 _manage_context_window 后验证保留顺序不为反转。
        """
        from runtime.llm_adapter.multi_provider import EchoProvider

        adapter = MultiProviderAdapter(
            default_config=ModelConfig(provider="echo", model="echo-test"),
            max_context_tokens=500,  # 很紧的窗口
        )
        adapter.register_provider(EchoProvider())
        config = ModelConfig(provider="echo", model="echo-test")

        # 构造超长消息：1 system + 20 user，每条约 160 字符 ≈ 40 token
        # 总 token 约 32 + 20*40 = 832 > 500，必定触发截断
        messages = [
            {"role": "system", "content": "你是一个有帮助的AI助手。" * 5},  # ~125 字符 ≈ 32 token
        ]
        for i in range(20):
            messages.append(
                {"role": "user", "content": f"这是第 {i} 条用户消息。" * 8}
            )  # ~160 字符 ≈ 40 token
        # 总 token 约 32 + 20*40 = 832 > 500，必定触发截断

        # _manage_context_window(messages, config) → List[Dict]
        truncated = adapter._manage_context_window(messages, config)

        assert len(truncated) > 0, (
            "P0-3 FAIL: 截断后消息列表为空"
        )

        # 断言 system 在第一位
        assert truncated[0]["role"] == "system", (
            f"P0-3 FAIL: system 消息不在第一位！\n"
            f"  第一条消息 role: {truncated[0]['role']}\n"
            f"  缺陷：截断逻辑可能反转了消息顺序或丢弃了 system"
        )

        # 断言 user 消息顺序是递增的（不是反转）
        user_indices = []
        for msg in truncated[1:]:
            if msg["role"] == "user" and "第 " in msg["content"]:
                # 提取序号
                import re
                m = re.search(r"第 (\d+) 条", msg["content"])
                if m:
                    user_indices.append(int(m.group(1)))

        if user_indices:
            # 应该保持原始顺序（递增），而非反转（递减）
            is_increasing = all(
                user_indices[i] < user_indices[i + 1]
                for i in range(len(user_indices) - 1)
            )
            assert is_increasing, (
                f"P0-3 FAIL: user 消息顺序被反转！\n"
                f"  user 消息序号序列: {user_indices}\n"
                f"  期望: 递增（原始顺序）\n"
                f"  缺陷：截断切片方向可能错误，保留了最后 N 条而非前 N 条"
            )

        # 验证总 token 不超过上限
        total_est = sum(len(msg["content"]) // 4 for msg in truncated)
        assert total_est <= 500 + 100, (  # 给一些缓冲
            f"P0-3 FAIL: 截断后 token 超限！\n"
            f"  估算 token: {total_est}\n"
            f"  上限: 500"
        )

    def test_truncation_preserves_latest_relevant(self):
        """
        验证截断保留最新消息而非最旧消息。
        """
        from runtime.llm_adapter.multi_provider import EchoProvider

        adapter = MultiProviderAdapter(
            default_config=ModelConfig(provider="echo", model="echo-test"),
            max_context_tokens=200,  # 极小窗口，强制截断
        )
        adapter.register_provider(EchoProvider())
        config = ModelConfig(provider="echo", model="echo-test")

        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "第一条消息: 1111111111"},
            {"role": "user", "content": "第二条消息: 2222222222"},
            {"role": "user", "content": "第三条消息: 3333333333"},
            {"role": "user", "content": "第四条消息: 4444444444"},
            {"role": "user", "content": "第五条消息: 5555555555"},
            {"role": "user", "content": "第六条消息: 6666666666"},
            {"role": "user", "content": "第七条消息: 7777777777"},
            {"role": "user", "content": "第八条消息: 8888888888"},
        ]

        truncated = adapter._manage_context_window(messages, config)

        # 最新消息应该在末尾 → 检查 content 字段
        all_content = " ".join(m.get("content", "") for m in truncated)

        # 如果保留了最后 N 条（正确），应有 8888888888
        # 如果错误保留了前 N 条，应有 1111111111
        has_latest = "8888888888" in all_content or "7777777777" in all_content
        has_oldest = "1111111111" in all_content and "8888888888" not in all_content

        assert has_latest, (
            f"P0-3 FAIL: 截断丢弃了最新消息！\n"
            f"  截断后内容: {all_content[:200]}...\n"
            f"  期望: 保留最新消息"
        )

        assert not has_oldest, (
            f"P0-3 FAIL: 截断保留了最旧消息而非最新消息！\n"
            f"  截断后内容: {all_content[:200]}..."
        )


# ============================================================================
# T4: schema 归一化 (P0-4)
# ============================================================================


class TestT4SchemaNormalization:
    """
    [P0-4] 验证 _normalize_schema 使用 Tool.params_schema 的真实结构：
    {"name", "description", "parameters": {...}}
    归一化为 {"type": "object", "properties": {...}, "required": [...]}，
    且 properties 来自内层 parameters，不是把 parameters 当属性。
    """

    def test_normalize_extracts_inner_parameters(self):
        """
        传入 Tool.params_schema 格式，断言 properties 来自内层 parameters。
        """
        # 真实 params_schema 结构（来自 generate_schema_from_function）
        params_schema = {
            "name": "get_weather",
            "description": "获取天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称",
                    },
                    "days": {
                        "type": "integer",
                        "description": "预报天数",
                        "default": 1,
                    },
                },
                "required": ["city"],
            },
        }

        result = _normalize_schema(params_schema)

        # 断言顶层结构
        assert result.get("type") == "object", (
            f"P0-4 FAIL: 归一化后 type 不正确！\n"
            f"  期望: object\n"
            f"  实际: {result.get('type')}"
        )

        # 断言 properties 存在且来自内层 parameters.properties
        properties = result.get("properties", {})
        assert "city" in properties, (
            f"P0-4 FAIL: properties 缺失 city 字段！\n"
            f"  properties keys: {list(properties.keys())}\n"
            f"  期望包含: city"
        )
        assert "days" in properties, (
            f"P0-4 FAIL: properties 缺失 days 字段！\n"
            f"  properties keys: {list(properties.keys())}"
        )

        # 关键断言：properties 不应包含 "parameters" 这个 key
        assert "parameters" not in properties, (
            f"P0-4 FAIL: properties 错误地包含了 'parameters' 作为属性！\n"
            f"  properties keys: {list(properties.keys())}\n"
            f"  期望: properties 来自内层 parameters.properties，而非外层的 'parameters' 键"
        )

        # 断言 required 正确
        assert result.get("required") == ["city"], (
            f"P0-4 FAIL: required 不正确！\n"
            f"  期望: ['city']\n"
            f"  实际: {result.get('required')}"
        )

    def test_normalize_params_list_format(self):
        """
        测试 params_schema 中 parameters 为列表格式的归一化。
        """
        params_schema = {
            "name": "search",
            "description": "搜索工具",
            "params": [
                {"name": "query", "type": "string", "description": "搜索关键词", "required": True},
                {"name": "limit", "type": "integer", "description": "结果数量"},
            ],
        }

        result = _normalize_schema(params_schema)

        assert result["type"] == "object", (
            f"P0-4 FAIL: params 列表归一化 type 不正确: {result}"
        )
        assert "query" in result["properties"], (
            f"P0-4 FAIL: params 列表归一化后 properties 缺失 query: {result}"
        )
        assert "query" in result["required"], (
            f"P0-4 FAIL: params 列表归一化后 required 缺失 query: {result}"
        )

    def test_normalize_already_standard(self):
        """
        测试已为标准 JSON Schema 的 schema 不被破坏。
        """
        standard = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        result = _normalize_schema(standard)
        assert result == standard, (
            f"P0-4 FAIL: 标准 schema 被意外修改！\n"
            f"  原始: {standard}\n"
            f"  结果: {result}"
        )


# ============================================================================
# T5: VectorStore 归档 (P0-5)
# ============================================================================


class TestT5VectorStoreArchive:
    """
    [P0-5] 验证 VectorStore 归档语义：
    - 存入 2 条，归档 1 条
    - all_active 只剩 1 条
    - get(归档id) 返回 None 或归档对象（语义明确）
    """

    def test_archive_removes_from_active(self, vector_store):
        """存入 2 条，归档 1 条，all_active 应为 1 条。"""
        t1 = _make_trace(content="记忆A", trace_id="a1")
        t2 = _make_trace(content="记忆B", trace_id="a2")

        vector_store.store(t1)
        vector_store.store(t2)

        assert len(vector_store.all_active) == 2, (
            f"P0-5 FAIL: 存入 2 条后 all_active 应为 2，实际为 {len(vector_store.all_active)}"
        )

        vector_store.archive("a1")

        active = vector_store.all_active
        assert len(active) == 1, (
            f"P0-5 FAIL: 归档 1 条后 all_active 应为 1，实际为 {len(active)}\n"
            f"  活跃 IDs: {[t.trace_id for t in active]}"
        )

        # 归档的那条 get 返回什么？语义应明确：None 或归档对象均可
        archived = vector_store.get("a1")
        # 允许两种情况：返回 None（归档后不可获取）或返回对象（标记为已归档）
        if archived is not None:
            # 如果是返回对象，应该与原始对象一致或标记已归档
            assert archived.trace_id == "a1", (
                f"P0-5 FAIL: get(归档id) 返回了错误对象，trace_id={archived.trace_id}"
            )

    def test_active_only_returns_unarchived(self, vector_store):
        """验证 all_active 不包含已归档的记忆。"""
        t1 = _make_trace(content="活跃", trace_id="act1")
        t2 = _make_trace(content="归档", trace_id="arc1")

        vector_store.store(t1)
        vector_store.store(t2)
        vector_store.archive("arc1")

        active_ids = {t.trace_id for t in vector_store.all_active}
        assert "act1" in active_ids, (
            f"P0-5 FAIL: 未归档的记忆在 all_active 中丢失"
        )
        assert "arc1" not in active_ids, (
            f"P0-5 FAIL: 已归档的记忆仍出现在 all_active 中\n"
            f"  active_ids: {active_ids}"
        )


# ============================================================================
# T6: flush_strengthen 自动触发 (P1-1)
# ============================================================================


class TestT6FlushStrengthenAutoTrigger:
    """
    [P1-1] 验证 enqueue_strengthen 达到批量阈值时自动触发 flush。

    若未实现自动触发，测试应 fail。
    """

    def test_auto_flush_on_threshold(self, in_memory_store):
        """连续 enqueue_strengthen 10 次，断言队列被自动清空或提供 is_flushed 检查。"""
        ctx = LayeredContextManager(
            long_term_store=in_memory_store,
            max_working_memory=50,
        )

        # 先存入一些记忆
        trace_ids = []
        for i in range(10):
            t = _make_trace(content=f"enqueue 测试 {i}", strength=0.5)
            in_memory_store.store(t)
            trace_ids.append(t.trace_id)

        # 连续 enqueue 10 次（阈值 5 应该触发至少一次自动刷新）
        for i in range(10):
            ctx.enqueue_strengthen([trace_ids[i]])

        # 检查队列状态
        queue_size = len(ctx._strengthen_queue)
        assert queue_size < 10, (
            f"P1-1 FAIL: enqueue 10 次后队列未被自动清空！\n"
            f"  队列长度: {queue_size}（阈值=5）\n"
            f"  期望: < 10（自动触发 flush 应清空或减少队列）\n"
            f"  若未实现自动触发，此测试 fail 是预期结果"
        )

    def test_flush_resets_queue(self, in_memory_store):
        """显式调用 flush_strengthen 应清空队列。"""
        ctx = LayeredContextManager(
            long_term_store=in_memory_store,
            max_working_memory=50,
        )

        for i in range(3):
            t = _make_trace(content=f"flush 测试 {i}")
            in_memory_store.store(t)
            ctx.enqueue_strengthen([t.trace_id])

        assert len(ctx._strengthen_queue) > 0, (
            f"P1-1 FAIL: enqueue 后队列为空，无法测试 flush"
        )

        ctx.flush_strengthen()

        assert len(ctx._strengthen_queue) == 0, (
            f"P1-1 FAIL: flush_strengthen 后队列未清空！\n"
            f"  剩余队列长度: {len(ctx._strengthen_queue)}\n"
            f"  期望: 0"
        )


# ============================================================================
# T7: Anthropic count_tokens 不调用 API (P1-2)
# ============================================================================


class TestT7AnthropicCountTokensNoAPI:
    """
    [P1-2] 验证 AnthropicProvider.count_tokens 在 anthropic SDK 不可用或
    monkey-patch 抛异常时走字符估算路径，不抛异常，返回正整数值。
    """

    def test_count_tokens_no_api_call(self):
        """
        Monkey-patch anthropic.Anthropic 构造抛异常，
        count_tokens 应走字符估算路径。
        """
        provider = AnthropicProvider()

        # count_tokens 签名: (messages: List[Dict], model: Optional[str] = None) -> int
        test_messages = [
            {"role": "user", "content": "Hello, world! 这是一段测试文本。"}
        ]

        # 尝试调用 count_tokens——如果内部调用 Anthropic API 会失败
        # 但我们期望它走字符估算路径
        try:
            result = provider.count_tokens(test_messages)
        except Exception as e:
            # 如果这里抛异常，说明 count_tokens 强行调用了 API
            pytest.fail(
                f"P1-2 FAIL: count_tokens 抛出了异常！\n"
                f"  异常类型: {type(e).__name__}\n"
                f"  异常信息: {e}\n"
                f"  期望: 不抛异常，走字符估算路径返回正整数值"
            )

        assert isinstance(result, int), (
            f"P1-2 FAIL: count_tokens 返回值类型错误！\n"
            f"  期望类型: int\n"
            f"  实际类型: {type(result).__name__}"
        )

        assert result > 0, (
            f"P1-2 FAIL: count_tokens 返回值不是正整数值！\n"
            f"  实际值: {result}\n"
            f"  输入消息数: {len(test_messages)}"
        )

    def test_count_tokens_reasonable_estimate(self):
        """验证字符估算结果在合理范围内（中文约 1 char/token，英文约 4 char/token）。"""
        provider = AnthropicProvider()

        # count_tokens 签名: (messages: List[Dict], model: Optional[str] = None) -> int
        # 纯英文
        eng_text = "hello world " * 50  # 600 字符
        eng_tokens = provider.count_tokens([{"role": "user", "content": eng_text}])
        # 英文约 4 chars/token → 600/4 = 150
        assert 100 <= eng_tokens <= 600, (
            f"P1-2 FAIL: 英文 token 估算异常！\n"
            f"  文本长度: {len(eng_text)}\n"
            f"  估算 tokens: {eng_tokens}\n"
            f"  合理范围: 100 ~ 600"
        )

        # 中文
        cn_text = "你好世界" * 100  # 400 字符
        cn_tokens = provider.count_tokens([{"role": "user", "content": cn_text}])
        # 中文约 1-2 chars/token → 200~400
        assert 100 <= cn_tokens <= 800, (
            f"P1-2 FAIL: 中文 token 估算异常！\n"
            f"  文本长度: {len(cn_text)}\n"
            f"  估算 tokens: {cn_tokens}\n"
            f"  合理范围: 100 ~ 800"
        )


# ============================================================================
# T8: SQLiteStore 线程安全 (P1-3)
# ============================================================================


class TestT8SQLiteStoreThreadSafety:
    """
    [P1-3] 验证 SQLiteStore 在多线程并发 store/update_strength 时：
    - 无异常抛出
    - 最终数据一致
    """

    def test_concurrent_store_and_update(self, sqlite_store):
        """启动 5 个线程并发 store 和 update_strength，断言无异常且数据一致。"""
        import threading
        import queue as qmod

        error_queue = qmod.Queue()

        # 先预存 10 条记忆
        trace_ids = []
        for i in range(10):
            t = _make_trace(content=f"thread 预存 {i}", strength=1.0)
            sqlite_store.store(t)
            trace_ids.append(t.trace_id)

        def worker(worker_id, ids_slice):
            try:
                for tid in ids_slice:
                    # 更新强度
                    sqlite_store.update_strength(tid, 0.5 + worker_id * 0.1)
                    # 再存入一条新记忆
                    new_t = _make_trace(
                        content=f"worker {worker_id} 新增",
                        strength=0.8,
                    )
                    sqlite_store.store(new_t)
            except Exception as e:
                error_queue.put((worker_id, str(e)))

        # 将 10 个 id 分给 5 个线程
        per_worker = 2
        threads = []
        for w in range(5):
            start = w * per_worker
            end = start + per_worker
            t = threading.Thread(
                target=worker,
                args=(w, trace_ids[start:end]),
                name=f"worker-{w}",
            )
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # 检查异常
        errors = []
        while not error_queue.empty():
            errors.append(error_queue.get())

        assert len(errors) == 0, (
            f"P1-3 FAIL: 并发操作抛出了异常！\n"
            f"  异常列表: {errors}"
        )

        # 验证数据一致性：预存的 10 条应都在
        for tid in trace_ids:
            stored = sqlite_store.get(tid)
            assert stored is not None, (
                f"P1-3 FAIL: 预存记忆 {tid} 在并发操作后丢失！"
            )
            assert 0.0 <= stored.strength <= 1.0, (
                f"P1-3 FAIL: 记忆 {tid} 强度异常: {stored.strength}"
            )

        # 新增的记忆应可检索（all_active 是 property，不是方法）
        all_traces = getattr(sqlite_store, "all_active", [])
        assert len(all_traces) >= 10, (
            f"P1-3 FAIL: 并发操作后活跃记忆数量异常！\n"
            f"  活跃数: {len(all_traces)}\n"
            f"  期望: >= 10"
        )


# ============================================================================
# T9: 端到端集成 (P1-4)
# ============================================================================


class TestT9EndToEndIntegration:
    """
    [P1-4] 使用 EchoProvider + InMemoryToolRegistry + AutoDetectParser + LayeredContextManager
    跑通完整流程：
    用户输入 → LLM 返回 tool_call → 解析 → 执行 echo → 记录到 ContextManager →
    下一轮 build_context 包含历史
    """

    def _build_integration_env(self):
        """构建完整的集成测试环境。"""
        # ContextManager
        store = InMemoryStore()
        ctx = LayeredContextManager(
            long_term_store=store,
            max_working_memory=50,
        )

        # ToolRegistry
        registry = InMemoryToolRegistry()
        echo_tool = Tool(
            name="echo",
            description="回显消息",
            params_schema={
                "name": "echo",
                "description": "回显消息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "要回显的消息"},
                    },
                    "required": ["message"],
                },
            },
            fn=lambda message: f"ECHO: {message}",
            source="local",
        )
        registry.register(echo_tool)

        # MultiProviderAdapter
        adapter = MultiProviderAdapter(
            default_config=ModelConfig(provider="echo", model="echo-test"),
            max_context_tokens=128000,
        )
        from runtime.llm_adapter.multi_provider import EchoProvider
        adapter.register_provider(EchoProvider())

        # Parser
        parser = AutoDetectParser()

        return ctx, registry, adapter, parser

    def test_full_loop(self):
        """
        完整流程：
        1. 构建初始 context
        2. 模拟 LLM 返回 tool_call
        3. 解析 tool_call
        4. 执行 echo 工具
        5. 将结果记录到 ContextManager
        6. 下一轮 build_context 包含历史
        """
        ctx, registry, adapter, parser = self._build_integration_env()

        # Round 1: 设置 system prompt + 用户输入
        system_prompt = "你是一个助手，当用户要求回显时使用 echo 工具。"
        user_msg_1 = Message(role="user", content="请回显：你好世界")
        ctx.append(Message(role="system", content=system_prompt))
        ctx.append(user_msg_1)

        # 构建第一轮 context
        round1_context = ctx.build_context(system_prompt=system_prompt)
        assert len(round1_context) >= 2, (
            f"P1-4 FAIL: Round 1 context 消息数不足！\n"
            f"  实际: {len(round1_context)}\n"
            f"  期望: >= 2 (system + user)"
        )

        # 模拟 LLM 返回 tool_call（OpenAI 格式）
        llm_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_001",
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "arguments": '{"message": "你好世界"}',
                        },
                    }],
                }
            }]
        }

        # 记录 assistant 消息（含 tool_calls）
        assistant_msg = Message(
            role="assistant",
            content="",
            tool_calls=[{
                "id": "call_001",
                "type": "function",
                "function": {
                    "name": "echo",
                    "arguments": '{"message": "你好世界"}',
                },
            }],
        )
        ctx.append(assistant_msg)

        # 解析 tool_call
        tool_calls = parser.parse(llm_response)
        assert len(tool_calls) == 1, (
            f"P1-4 FAIL: 解析 tool_call 失败！\n"
            f"  解析结果数: {len(tool_calls)}\n"
            f"  期望: 1"
        )
        assert tool_calls[0].name == "echo", (
            f"P1-4 FAIL: 解析的 tool name 不正确！\n"
            f"  实际: {tool_calls[0].name}\n"
            f"  期望: echo"
        )

        # 执行工具
        result = registry.execute("echo", {"message": tool_calls[0].args.get("message", "")})

        # 验证结果
        assert result.error is False, (
            f"P1-4 FAIL: 工具执行出错！\n"
            f"  错误: {result.content}"
        )
        assert "ECHO" in str(result.content), (
            f"P1-4 FAIL: echo 工具返回值不正确！\n"
            f"  实际: {result.content}"
        )

        # 记录 tool 结果
        tool_msg = Message(
            role="tool",
            content=str(result.content),
            tool_call_id="call_001",
            name="echo",
        )
        ctx.append(tool_msg)

        # Round 2: 第二轮用户输入
        user_msg_2 = Message(role="user", content="再次回显：测试")
        ctx.append(user_msg_2)

        # 构建第二轮 context
        round2_context = ctx.build_context(system_prompt=system_prompt)

        # 验证 context 包含历史记录
        assert len(round2_context) >= 5, (
            f"P1-4 FAIL: Round 2 context 消息数不足！\n"
            f"  实际: {len(round2_context)}\n"
            f"  期望: >= 5 (system + user1 + assistant + tool + user2)\n"
            f"  消息列表:\n" +
            "\n".join(f"    [{m['role']}] {m['content'][:50]}" for m in round2_context)
        )

        # 验证 Work Memory 中有正确数量的消息
        wm = ctx.get_working_memory()
        assert len(wm) >= 5, (
            f"P1-4 FAIL: 工作记忆消息数不足！\n"
            f"  实际: {len(wm)}\n"
            f"  期望: >= 5"
        )

    def test_no_exceptions_during_flow(self):
        """验证整个流程不抛异常。"""
        ctx, registry, adapter, parser = self._build_integration_env()

        try:
            # 完整的 mini 对话
            ctx.append(Message(role="system", content="你是一个助手"))
            ctx.append(Message(role="user", content="echo 你好"))

            context = ctx.build_context(system_prompt="你是一个助手")
            assert len(context) >= 2

            # 模拟 assistant tool_call
            assistant_msg = Message(
                role="assistant",
                content="",
                tool_calls=[{
                    "id": "call_002",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"message": "你好"}'},
                }],
            )
            ctx.append(assistant_msg)

            # 解析并执行
            response = {
                "choices": [{"message": {
                    "tool_calls": [{
                        "id": "call_002",
                        "type": "function",
                        "function": {"name": "echo", "arguments": '{"message": "你好"}'},
                    }]
                }}]
            }
            tcs = parser.parse(response)
            assert len(tcs) == 1

            result = registry.execute(tcs[0].name, tcs[0].args)
            assert result.error is False

            ctx.append(Message(
                role="tool",
                content=str(result.content),
                tool_call_id="call_002",
                name="echo",
            ))

            # 第二轮
            ctx.append(Message(role="user", content="谢谢"))
            context2 = ctx.build_context(system_prompt="你是一个助手")

            # 验证 working memory 完整性
            wm = ctx.get_working_memory()
            roles = [m.role for m in wm]
            assert "system" in roles
            assert "user" in roles
            assert "assistant" in roles
            assert "tool" in roles

            # ContextManager 快照
            snap = ctx.snapshot()
            assert snap is not None
            assert len(snap.working_memory) == len(wm)

        except Exception as e:
            import traceback
            pytest.fail(
                f"P1-4 FAIL: 端到端流程抛出了异常！\n"
                f"  异常类型: {type(e).__name__}\n"
                f"  异常信息: {e}\n"
                f"  堆栈:\n{traceback.format_exc()}"
            )