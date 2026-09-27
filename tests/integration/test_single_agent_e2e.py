"""
[A1] 单 Agent 端到端（记忆系统闭环）。

覆盖 runtime.run() → LLM → 工具 → 记忆写入 → 沉淀 → 衰减 → 合并。
不修改任何功能代码，只写测试断言并暴露缺陷。

断言口径：
    - session（budget / abort_flag）
    - event_log 的 loop_exit 事件（iterations / aborted）
    - ctx.shared 各统计键（通过 ContextCaptureMiddleware 捕获最终 ctx）
    - StubContextManager 工作记忆顺序与长期记忆条目 / forgotten 标记
"""

from __future__ import annotations

import pytest

from agent_loop import HookResult
from runtime.builtin_middlewares.conversation_recorder import ConversationRecorderMiddleware
from runtime.builtin_middlewares.cost_guard import CostGuardMiddleware
from runtime.builtin_middlewares.memory_consolidation import MemoryConsolidationMiddleware
from runtime.builtin_middlewares.memory_lifecycle import MemoryLifecycleMiddleware
from runtime.builtin_middlewares.observability import ObservabilityMiddleware
from runtime.builtin_middlewares.safety_guard import SafetyGuardMiddleware
from runtime.context_manager.interface import MemoryTrace, Message
from runtime.event_log.interface import EVENT_LOOP_EXIT
from tests.fakes.fake_llm import FakeLLMProvider, openai_tool_call
from tests.fakes.fake_tool import FakeTool
from tests.integration.conftest import (
    ContextCaptureMiddleware,
    StubContextManager,
    build_runtime,
)


def _loop_exit(event_log):
    return event_log.last(EVENT_LOOP_EXIT)


def _iterations(event_log) -> int:
    ev = _loop_exit(event_log)
    return int((ev.payload or {}).get("iterations", -1))


def _aborted(event_log) -> bool:
    ev = _loop_exit(event_log)
    return bool((ev.payload or {}).get("aborted", False))


# ============================================================================
# 场景 1：单轮对话无工具调用（暴露：Loop 无 stop 自然终止）
# ============================================================================


def test_single_round_no_tool_calls():
    llm = FakeLLMProvider()
    llm.enqueue_text("你好，有什么可以帮您？")

    tool = FakeTool()
    registry_tool = tool.make_tool("echo")

    capture = ContextCaptureMiddleware()
    runtime = build_runtime(
        llm, [registry_tool],
        middlewares=[ObservabilityMiddleware(echo=False), capture],
        max_iterations=3,
    )
    session = runtime.run("s1", "hello")

    # [D001 已修复] 无 tool_calls → Loop 自然终止，只跑 1 轮，不再空转至 max_iterations
    assert _iterations(runtime.registry.event_log) == 1
    # 只发生 1 次 LLM 调用（修复前为 3 次）
    assert len(llm.calls) == 1
    # 工具从未被调用
    assert tool.call_log == []
    # 自然结束不是中止：aborted / abort_flag 均为 False
    assert _aborted(runtime.registry.event_log) is False
    assert session.abort_flag is False


# ============================================================================
# 场景 2：单轮对话有 1 次工具调用
# ============================================================================


def test_single_round_one_tool_call():
    llm = FakeLLMProvider()
    llm.enqueue_tool_calls([openai_tool_call("echo", {"message": "hi"})])

    tool = FakeTool()
    registry_tool = tool.make_tool("echo")

    recorder = ConversationRecorderMiddleware()
    capture = ContextCaptureMiddleware()
    runtime = build_runtime(
        llm, [registry_tool],
        middlewares=[
            ObservabilityMiddleware(echo=False),
            recorder,
            capture,
        ],
        max_iterations=2,
    )
    session = runtime.run("s2", "echo hi")

    # 工具被调用一次
    assert len(tool.call_log) == 1
    assert tool.call_log[0]["tool"] == "echo"

    # 对话记录：1 assistant + 1 tool
    shared = capture.ctx.shared
    state = shared.get("conversation_recorded")
    assert state is not None
    assert state["assistant"] == 1
    assert state["tool"] == 1

    # 工作记忆顺序：assistant 在前，tool 在后
    cm = runtime.registry.context_manager
    working = cm.get_working_memory()
    assert [m.role for m in working] == ["assistant", "tool"]
    assert working[0].tool_calls is not None

    # 工具调用事件写入了 event_log
    assert runtime.registry.event_log.count("tool_call") == 1


# ============================================================================
# 场景 3：多轮对话（≥5 轮）有多次工具调用
# ============================================================================


def test_multi_round_multiple_tool_calls():
    llm = FakeLLMProvider()
    # 5 轮，每轮 1 次工具调用，全部 echo 成功
    for i in range(5):
        llm.enqueue_tool_calls([openai_tool_call("echo", {"message": f"m{i}"})])

    tool = FakeTool()
    registry_tool = tool.make_tool("echo")

    recorder = ConversationRecorderMiddleware()
    capture = ContextCaptureMiddleware()
    runtime = build_runtime(
        llm, [registry_tool],
        middlewares=[ObservabilityMiddleware(echo=False), recorder, capture],
        max_iterations=6,
    )
    runtime.run("s3", "echo 5 times")

    assert len(tool.call_log) == 5
    shared = capture.ctx.shared
    state = shared.get("conversation_recorded")
    assert state["assistant"] == 5
    assert state["tool"] == 5

    cm = runtime.registry.context_manager
    roles = [m.role for m in cm.get_working_memory()]
    # 交替 assistant/tool，共 10 条
    assert roles == ["assistant", "tool"] * 5


# ============================================================================
# 场景 4：工具调用失败（ToolResult.error=True）后继续
# ============================================================================


def test_tool_failure_continues():
    llm = FakeLLMProvider()
    llm.enqueue_tool_calls([openai_tool_call("echo", {"message": "x"})])  # 失败轮
    llm.enqueue_tool_calls([openai_tool_call("echo", {"message": "y"})])  # 成功轮

    tool = FakeTool()
    tool.enqueue("fail").enqueue("success")
    registry_tool = tool.make_tool("echo")

    recorder = ConversationRecorderMiddleware()
    capture = ContextCaptureMiddleware()
    runtime = build_runtime(
        llm, [registry_tool],
        middlewares=[ObservabilityMiddleware(echo=False), recorder, capture],
        max_iterations=3,
    )
    runtime.run("s4", "fail then success")

    # 两次工具都尝试执行（失败也走 registry.execute）
    assert len(tool.call_log) == 2

    # tool 消息仍被记录（错误结果也算一条 tool 消息）
    cm = runtime.registry.context_manager
    tool_messages = [m for m in cm.get_working_memory() if m.role == "tool"]
    assert len(tool_messages) == 2

    # event_log 记录两次 tool_call 事件，第一次 success=False
    tool_events = runtime.registry.event_log.tool_calls()
    assert len(tool_events) == 2
    assert tool_events[0].success is False
    assert tool_events[1].success is True


# ============================================================================
# 场景 5：LLM 返回空 content 无 tool_calls（正常结束）
# ============================================================================


def test_llm_empty_content():
    llm = FakeLLMProvider()
    llm.enqueue_empty()

    tool = FakeTool()
    recorder = ConversationRecorderMiddleware()
    capture = ContextCaptureMiddleware()
    runtime = build_runtime(
        llm, [tool.make_tool("echo")],
        middlewares=[ObservabilityMiddleware(echo=False), recorder, capture],
        max_iterations=2,
    )
    session = runtime.run("s5", "empty")

    # 空 content + 无 tool_calls 时 ConversationRecorder 跳过，不写入 assistant
    shared = capture.ctx.shared
    state = shared.get("conversation_recorded")
    # 空响应不产生 assistant 记录（缺陷：正常结束但无终止，且不写记忆）
    assert state is None or state["assistant"] == 0
    assert tool.call_log == []
    assert session.abort_flag is False


# ============================================================================
# 场景 6：LLM 返回非法 tool_calls（解析器降级，不崩溃）
# ============================================================================


def test_llm_illegal_tool_calls():
    llm = FakeLLMProvider()
    llm.enqueue_illegal_tool_calls()

    tool = FakeTool()
    capture = ContextCaptureMiddleware()
    runtime = build_runtime(
        llm, [tool.make_tool("echo")],
        middlewares=[ObservabilityMiddleware(echo=False), capture],
        max_iterations=2,
    )
    session = runtime.run("s6", "illegal")

    # 非法 tool_calls 被降级为空，不执行任何工具、不崩溃
    assert tool.call_log == []
    assert session.abort_flag is False


# ============================================================================
# 场景 7：触发 ContextCompressor（消息量超阈值）
# ============================================================================


def test_context_compressor_triggers():
    from runtime.builtin_middlewares.context_compressor import ContextCompressorMiddleware

    cm = StubContextManager()
    llm = FakeLLMProvider()
    llm.enqueue_text("ok")

    compressor = ContextCompressorMiddleware(max_tokens=100, threshold_ratio=0.5)
    capture = ContextCaptureMiddleware()
    runtime = build_runtime(
        llm, [],
        cm=cm,
        middlewares=[compressor, capture],
        max_iterations=2,
    )
    # 压缩器估算的是 Loop messages（不是工作记忆），所以必须靠超长 task 触发：
    # threshold = 100 * 0.5 = 50 token，task 长度 1000 字符 ≈ 250 token。
    runtime.run("s7", "x" * 1000)

    assert compressor.compressed_count >= 1
    assert "context_compressed" in capture.ctx.shared


# ============================================================================
# 场景 8：触发 CostGuard（token 超预算）
# ============================================================================


def test_cost_guard_aborts_on_budget():
    llm = FakeLLMProvider()
    # 每轮响应 total_tokens=100 且带一次工具调用（[D001 已修复] 需显式的继续信号
    # 才能进入下一轮），预算设为 150 → 第 2 轮即超限。
    for _ in range(5):
        llm.enqueue({
            "content": "ok",
            "tool_calls": [openai_tool_call("echo", {"message": "x"})],
            "usage": {"total_tokens": 100},
            "cost": 0.01,
            "model": "m",
            "provider": "p",
            "finish_reason": "tool_calls",
        })

    tool = FakeTool()
    capture = ContextCaptureMiddleware()
    observability = ObservabilityMiddleware(echo=False)
    cost_guard = CostGuardMiddleware(max_tokens=150)
    runtime = build_runtime(
        llm, [tool.make_tool("echo")],
        middlewares=[observability, cost_guard, capture],
        max_iterations=10,
    )
    session = runtime.run("s8", "budget", budget={"max_tokens": 150})

    assert session.abort_flag is True
    assert session.budget.used_tokens > 150
    assert cost_guard.abort_reason != ""


# ============================================================================
# 场景 9：触发 SafetyGuard（高危工具被 skip）
# ============================================================================


def test_safety_guard_skips_high_risk_tool():
    llm = FakeLLMProvider()
    llm.enqueue_tool_calls([openai_tool_call("danger", {"x": 1})])

    tool = FakeTool()
    # 高危工具：requires_approval=True
    danger_tool = tool.make_tool("danger", requires_approval=True)

    safety = SafetyGuardMiddleware()
    capture = ContextCaptureMiddleware()
    runtime = build_runtime(
        llm, [danger_tool],
        middlewares=[ObservabilityMiddleware(echo=False), safety, capture],
        max_iterations=2,
    )
    runtime.run("s9", "danger")

    # 高危工具被 skip，fn 从未执行
    assert tool.call_log == []
    assert safety.blocked != []
    blocked = capture.ctx.shared.get("safety_blocked")
    assert blocked is not None and len(blocked) == 1


# ============================================================================
# 场景 10：触发 MemoryConsolidation（episodic 累积 ≥10 条）
# ============================================================================


def test_memory_consolidation_triggers():
    cm = StubContextManager()
    cm.consolidate_result = 1  # 编程：合并生成 1 条 semantic

    llm = FakeLLMProvider()
    llm.enqueue_empty()

    consolidation = MemoryConsolidationMiddleware()
    capture = ContextCaptureMiddleware()
    runtime = build_runtime(
        llm, [],
        cm=cm,
        middlewares=[consolidation, capture],
        max_iterations=1,
    )
    runtime.run("s10", "consolidate")

    state = capture.ctx.shared.get("memory_consolidation")
    assert state is not None
    assert state["consolidate_calls"] == 1
    assert state["consolidated_traces"] == 1


# ============================================================================
# 辅助：记录 / 沉淀 / 衰减 / 合并联动（记忆系统闭环的最小验证）
# ============================================================================


def test_memory_lifecycle_iteration_resets_between_runs():
    """[D009 已修复] MemoryLifecycleMiddleware 的迭代计数跨 run 复位。

    修复前：`_iteration` 是中间件实例字段，同一实例跨 run 复用时第二次 run 的
    计数继续沿用上一轮（3 + 2 = 5），整除判定错位，部分 run 漏触发 extract_memories。
    修复后：计数真值存于 ctx.private，Context 每次 run 由 Runtime 新建，
    因此天然 per-run；实例上的 `_iteration` 只是"最近一次 run 的计数镜像"。
    """
    from runtime.builtin_middlewares.memory_lifecycle import MemoryLifecycleMiddleware

    lifecycle = MemoryLifecycleMiddleware(extract_interval=2)

    def one_run(sid, iters):
        llm = FakeLLMProvider()
        # [D001 已修复] 每轮注入一次工具调用，才能驱动 Loop 跑满 iters 轮
        for _ in range(iters):
            llm.enqueue_tool_calls([openai_tool_call("echo", {"message": "x"})])
        cm = StubContextManager()
        tool = FakeTool()
        build_runtime(
            llm, [tool.make_tool("echo")], cm=cm,
            middlewares=[lifecycle],
            max_iterations=iters,
        ).run(sid, "x")

    # 第一次 run 3 轮后，计数为 3
    one_run("d9-a", 3)
    after_first = lifecycle._iteration
    assert after_first == 3  # 等于本次 run 的迭代轮数

    # 第二次 run 2 轮，复用同一实例 → 计数应从 0 重新开始，结果为 2（不是 5）
    one_run("d9-b", 2)
    after_second = lifecycle._iteration
    assert after_second == 2, f"[D009] 跨 run 应复位，实际 {after_second}"


def test_memory_lifecycle_decay_linkage():
    from runtime.builtin_middlewares.memory_lifecycle import MemoryLifecycleMiddleware

    cm = StubContextManager()
    # 沉淀每次返回 1 条 episodic trace
    cm.extract_result = [
        MemoryTrace.create(content="remembered fact", namespace="default", kind="episodic")
    ]
    cm.decay_archived = 0

    llm = FakeLLMProvider()
    llm.enqueue_empty()

    lifecycle = MemoryLifecycleMiddleware(extract_interval=1)
    capture = ContextCaptureMiddleware()
    runtime = build_runtime(
        llm, [],
        cm=cm,
        middlewares=[lifecycle, capture],
        max_iterations=1,
    )
    runtime.run("s11", "lifecycle")

    state = capture.ctx.shared.get("memory_lifecycle")
    assert state is not None
    assert state["extract_calls"] >= 1
    assert state["decay_calls"] >= 1
    # 长期记忆里有沉淀出的 1 条 trace
    assert len(cm.ltm) == 1