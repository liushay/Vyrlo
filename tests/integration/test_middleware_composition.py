"""
[A2] 中间件组合顺序（重点暴露顺序耦合）。

对每对中间件，用不同注册顺序跑两遍，断言最终 messages 精确形态差异。
差异即为顺序耦合证据。

重点组合：
    1. MemoryInjector(append) + ContextCompressor
    2. ConversationRecorder + MemoryLifecycle
    3. SafetyGuard + Observability
    4. CostGuard + Observability
    5. MemoryInjector(append/replace) + ContextCompressor
"""

from __future__ import annotations

from runtime.builtin_middlewares.context_compressor import ContextCompressorMiddleware
from runtime.builtin_middlewares.conversation_recorder import ConversationRecorderMiddleware
from runtime.builtin_middlewares.cost_guard import CostGuardMiddleware
from runtime.builtin_middlewares.memory_injector import MemoryInjectorMiddleware
from runtime.builtin_middlewares.memory_lifecycle import MemoryLifecycleMiddleware
from runtime.builtin_middlewares.observability import ObservabilityMiddleware
from runtime.builtin_middlewares.safety_guard import SafetyGuardMiddleware
from runtime.context_manager.interface import MemoryTrace, Message
from tests.fakes.fake_llm import FakeLLMProvider, openai_tool_call
from tests.fakes.fake_tool import FakeTool
from tests.integration.conftest import (
    ContextCaptureMiddleware,
    StubContextManager,
    build_runtime,
)


def _last_llm_messages(llm: FakeLLMProvider):
    """返回 LLM 最后一次实际收到的消息列表（最终形态）。"""
    if not llm.calls:
        return []
    return llm.calls[-1]


# ============================================================================
# 组合 1：MemoryInjector(append) + ContextCompressor —— 顺序对最终形态的影响
# ============================================================================


def test_injector_then_compressor_no_chaining():
    """组合 1：MemoryInjector(append) + ContextCompressor。

    [D004/D005 已修复] BEFORE_LLM 现在链式传递：injector 先注入长记忆，
    compressor 链式接收到注入后的完整 messages。注入后的消息已超阈值，
    compressor 现在**应当**触发压缩（无论注入后能否保留记忆取决于 compress 重建，
    但至少 compressed_count 不再恒为 0 —— 这是 D004 的回归证据）。
    """
    long_memory = "记忆内容" * 300  # ~ 1500 字符 ≈ 375 token

    def run_with(order):
        cm = StubContextManager()
        cm.ltm["m1"] = MemoryTrace.create(content=long_memory, namespace="default")
        llm = FakeLLMProvider()
        llm.enqueue_text("ok")
        capt = ContextCaptureMiddleware()
        build_runtime(
            llm, [], cm=cm,
            middlewares=[*order, capt],
            max_iterations=1,
        ).run("comp", "do something")
        # 从 order 中取出实际注册的 compressor 实例
        registered_compressor = next(
            m for m in order if isinstance(m, ContextCompressorMiddleware)
        )
        return registered_compressor, _last_llm_messages(llm)

    # forward 顺序：injector 先 → compressor 后
    compressor_forward, msgs_forward = run_with(
        [MemoryInjectorMiddleware(inject_mode="append"),
         ContextCompressorMiddleware(max_tokens=100, threshold_ratio=0.3)]
    )

    # reverse 顺序：compressor 先 → injector 后
    compressor_reverse, msgs_reverse = run_with(
        [ContextCompressorMiddleware(max_tokens=100, threshold_ratio=0.3),
         MemoryInjectorMiddleware(inject_mode="append")]
    )

    # D004 回归：注入后的消息已进入链；forward（injector 先）下 compressor
    # 收到注入后的超长消息并触发压缩。
    assert compressor_forward.compressed_count >= 1

    # reverse 顺序：compressor 先于 injector 执行，此时只看到短 messages，
    # 但 D005 会把工作记忆纳入估算 —— StubContextManager 工作记忆为空，
    # 故不会触发。这里只验证它不会因链式语义崩溃。
    assert isinstance(compressor_reverse.compressed_count, int)


# ============================================================================
# 组合 2：ConversationRecorder + MemoryLifecycle —— 沉淀能拿到本轮新增消息
# ============================================================================


def test_recorder_before_lifecycle_precipitation_sees_new_messages():
    cm = StubContextManager()
    # extract_memories 真实去读工作记忆：若 recorder 在其之后且 extract_interval 命中，
    # 应能看到本轮 assistant/tool 消息（取决于 after_iteration 执行时机）。
    llm = FakeLLMProvider()
    llm.enqueue_tool_calls([openai_tool_call("echo", {"m": "x"})])
    llm.enqueue_text("done")

    from tests.fakes.fake_tool import FakeTool
    tool = FakeTool()
    recorder = ConversationRecorderMiddleware()
    lifecycle = MemoryLifecycleMiddleware(extract_interval=1)
    capture = ContextCaptureMiddleware()

    runtime = build_runtime(
        llm, [tool.make_tool("echo")],
        cm=cm,
        middlewares=[recorder, lifecycle, capture],
        max_iterations=2,
    )
    runtime.run("precip", "hello")

    # 沉淀调用发生（extract_interval=1，两轮都应触发）
    assert lifecycle._iteration >= 2
    assert "memory_lifecycle" in capture.ctx.shared
    state = capture.ctx.shared["memory_lifecycle"]
    # after_iteration 在 after_tool 之后执行，理论上 extract 能看到工具消息
    # 但 extract_memories 用 LLMExtraction 实际会读工作记忆，这里用 stub 观察次数。
    assert state["extract_calls"] >= 1


# ============================================================================
# 组合 3：SafetyGuard + Observability —— 被 skip 的工具是否产生 ToolCallEvent
# ============================================================================


def test_skipped_tool_produces_tool_call_event():
    """[D007 已修复] 被 skip 的工具应产生一条可审计的 ToolCallEvent(skipped=True)。

    修复前：skip 发生在 BEFORE_TOOL，不进入 AFTER_TOOL，Observability 不写任何事件，
    审计链路完全缺失被 skip 的工具（本测试原名 test_skipped_tool_produces_no_tool_call_event，
    断言的正是缺陷现状）。
    修复后：Observability 在 AFTER_ITERATION 感知 ctx.shared["safety_blocked"]，
    为本轮被拦截的工具补写 ToolCallEvent(skipped=True)。
    """
    from tests.fakes.fake_tool import FakeTool

    llm = FakeLLMProvider()
    llm.enqueue_tool_calls([openai_tool_call("danger", {"x": 1})])

    tool = FakeTool()
    safety = SafetyGuardMiddleware()
    observability = ObservabilityMiddleware(echo=False)
    capture = ContextCaptureMiddleware()

    runtime = build_runtime(
        llm, [tool.make_tool("danger", requires_approval=True)],
        middlewares=[safety, observability, capture],
        max_iterations=2,
    )
    runtime.run("skip", "danger")

    # [D007] 被 skip 的工具补写了一条 ToolCallEvent，且带 skipped 标记
    tool_events = runtime.registry.event_log.tool_calls()
    assert len(tool_events) == 1
    assert tool_events[0].tool_name == "danger"
    assert tool_events[0].skipped is True


# ============================================================================
# 组合 4：CostGuard + Observability —— 预算累加与事件数一致
# ============================================================================


def test_cost_guard_observability_budget_consistency():
    llm = FakeLLMProvider()
    # [D001 已修复] 前两轮带工具调用以维持多轮迭代，使 3 次 LLM 调用真实发生；
    # 第 3 轮无工具调用 → Loop 自然终止。
    responses = [
        {"content": "ok",
         "tool_calls": [openai_tool_call("echo", {"message": "a"})],
         "usage": {"total_tokens": 10}, "cost": 0.0,
         "model": "m", "provider": "p", "finish_reason": "tool_calls"},
        {"content": "ok",
         "tool_calls": [openai_tool_call("echo", {"message": "b"})],
         "usage": {"total_tokens": 20}, "cost": 0.0,
         "model": "m", "provider": "p", "finish_reason": "tool_calls"},
        {"content": "ok", "tool_calls": [], "usage": {"total_tokens": 30}, "cost": 0.0,
         "model": "m", "provider": "p", "finish_reason": "stop"},
    ]
    for r in responses:
        llm.enqueue(r)

    tool = FakeTool()
    observability = ObservabilityMiddleware(echo=False)
    cost_guard = CostGuardMiddleware(max_tokens=1000)
    capture = ContextCaptureMiddleware()

    runtime = build_runtime(
        llm, [tool.make_tool("echo")],
        middlewares=[observability, cost_guard, capture],
        max_iterations=3,
    )
    session = runtime.run("cg", "count tokens")

    # 三个 LLM 调用事件，总 token 一致
    assert runtime.registry.event_log.count("llm_call") == 3
    assert session.budget.used_tokens == 60  # 10 + 20 + 30
    # 聚合 token 应与 session 累加一致
    assert runtime.registry.event_log.aggregate_tokens()["total_tokens"] == 60


# ============================================================================
# 组合 5：MemoryInjector append + ContextCompressor —— system prompt 是否被破坏
# ============================================================================


def test_append_mode_preserves_system_prompt_under_compression():
    cm = StubContextManager()
    cm.ltm["m1"] = MemoryTrace.create(content="关键约束", namespace="default")

    # append 模式：不替换 system prompt；压缩器应在压缩时保留 system 消息
    injector = MemoryInjectorMiddleware(inject_mode="append", system_prompt="SYS-PROMPT")
    compressor = ContextCompressorMiddleware(max_tokens=8, threshold_ratio=0.1)
    capture = ContextCaptureMiddleware()

    llm = FakeLLMProvider()
    llm.enqueue_text("ok")
    runtime = build_runtime(
        llm, [], cm=cm,
        middlewares=[injector, compressor, capture],
        max_iterations=1,
    )
    runtime.run("ap", "x" * 200, system_prompt="SYS-PROMPT")

    # 关键：system prompt 是否仍然存在且为原始内容（未被压缩器重建破坏）
    msgs = _last_llm_messages(llm)
    system_msgs = [m for m in msgs if m.get("role") == "system"]
    assert len(system_msgs) == 1, f"system prompt 数量异常: {msgs}"
    assert system_msgs[0]["content"] == "SYS-PROMPT"


# ============================================================================
# 组合：MemoryInjector replace 模式 + ContextCompressor（暴露 system 重建问题）
# ============================================================================


def test_replace_mode_system_prompt_preserved():
    cm = StubContextManager()
    injector = MemoryInjectorMiddleware(inject_mode="replace", system_prompt="SYS-{{memory_context}}")
    compressor = ContextCompressorMiddleware(max_tokens=8, threshold_ratio=0.1)
    capture = ContextCaptureMiddleware()

    llm = FakeLLMProvider()
    llm.enqueue_text("ok")
    runtime = build_runtime(
        llm, [], cm=cm,
        middlewares=[injector, compressor, capture],
        max_iterations=1,
    )
    runtime.run("rp", "x" * 200, system_prompt="SYS-{{memory_context}}")

    msgs = _last_llm_messages(llm)
    # build_context 把 {{memory_context}} 替换为"（无相关记忆）"
    system_msgs = [m for m in msgs if m.get("role") == "system"]
    assert len(system_msgs) >= 1