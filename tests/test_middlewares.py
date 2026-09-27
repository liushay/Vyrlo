"""
[L3-1] L3 第一批中间件测试 — 每个中间件一个独立测试。

测试清单：
    T1 test_observability_middleware      : 事件写入 + 结构化日志 + 聚合指标
    T2 test_cost_guard_middleware         : 预算累加 + 超限 abort_loop
    T3 test_safety_guard_middleware       : 高危工具 skip_tool + 白名单放行
    T4 test_memory_injector_middleware    : build_context 结果注入 messages
    T5 test_context_compressor_middleware : 超限压缩 + 未超限放行
    T6 test_middlewares_independent_toggle: 每个中间件可单独启停，不影响其他

运行方式：python -m pytest tests/test_middlewares.py -v
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_loop import (  # noqa: E402
    AgentLoop,
    Context,
    ControlFlow,
    HookResult,
    Middleware,
)

from runtime.builtin_middlewares import (  # noqa: E402
    ContextCompressorMiddleware,
    CostGuardMiddleware,
    MemoryInjectorMiddleware,
    ObservabilityMiddleware,
    SafetyGuardMiddleware,
)
from runtime.context_manager.interface import MemoryTrace, Message  # noqa: E402
from runtime.context_manager.layered import LayeredContextManager  # noqa: E402
from runtime.event_log import (  # noqa: E402
    EVENT_LLM_CALL,
    EVENT_LOOP_EXIT,
    EVENT_TOOL_CALL,
    EventLog,
)
from runtime.session import Session  # noqa: E402
from runtime.tool_registry.in_memory import InMemoryToolRegistry  # noqa: E402
from runtime.tool_registry.interface import Tool, ToolResult  # noqa: E402


# ============================================================================
# 共用测试辅助
# ============================================================================


class _ToolCallStub(Middleware):
    """测试辅助：在 AFTER_LLM 中向 ctx 写入一个 tool_call。"""

    def __init__(
        self, tool_name: str, once: bool = True, name: str = "tool_call_stub"
    ) -> None:
        super().__init__(name)
        self._tool_name = tool_name
        self._once = once
        self._emitted = False

    def after_llm(self, ctx: Context, response: Any) -> HookResult:
        if self._once and self._emitted:
            return HookResult.continue_()
        self._emitted = True
        ctx.set_tool_calls([{"name": self._tool_name, "arguments": {"q": "x"}}])
        return HookResult.continue_()


def _make_registry() -> InMemoryToolRegistry:
    """构造含"普通工具 + 高危工具"的注册表。"""
    registry = InMemoryToolRegistry()
    registry.register(Tool(
        name="search",
        description="普通搜索工具",
        params_schema={"type": "object", "properties": {}},
        fn=lambda **kwargs: "search-ok",
    ))
    registry.register(Tool(
        name="rm_rf",
        description="高危删除工具",
        params_schema={"type": "object", "properties": {}},
        fn=lambda **kwargs: "deleted",
        requires_approval=True,
        metadata={"risk_level": "high"},
    ))
    return registry


def _mock_llm(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """带 usage / cost 的模拟 LLM 响应。"""
    return {
        "content": "我需要调用工具",
        "usage": {
            "prompt_tokens": 40,
            "completion_tokens": 20,
            "total_tokens": 60,
        },
        "cost": 0.003,
        "model": "mock-model",
        "provider": "mock",
    }


class _Recorder:
    """记录工具执行调用。"""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def __call__(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        self.calls.append(tool_name)
        return {"result": f"{tool_name}-ok", "metadata": {"elapsed_seconds": 0.01}}


def _run_loop(
    middlewares: List[Middleware],
    messages: List[Dict[str, Any]],
    max_iterations: int = 2,
    recorder: Any = None,
    ctx: Context = None,
) -> Context:
    """用给定中间件集合跑一次 AgentLoop。"""
    loop = AgentLoop(
        llm_call=_mock_llm,
        tool_executor=recorder if recorder is not None else _Recorder(),
        max_iterations=max_iterations,
    )
    for mw in middlewares:
        loop.register_middleware(mw)
    return loop.run(ctx if ctx is not None else Context(), messages)


# ============================================================================
# T1 — ObservabilityMiddleware
# ============================================================================


def test_observability_middleware():
    """T1: 事件写入 + 结构化 JSON 日志 + 性能指标聚合。"""
    log = EventLog()
    mw = ObservabilityMiddleware(event_log=log, echo=False)

    ctx = Context()

    # ---- AFTER_LLM：写入 LLMCallEvent ----
    response = {
        "content": "ok",
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        "cost": 0.002,
        "model": "gpt-4o-mini",
        "provider": "openai",
    }
    result = mw.after_llm(ctx, response)
    assert result.control == ControlFlow.CONTINUE, "可观测性中间件不得改变控制流"

    llm_events = log.llm_calls()
    assert len(llm_events) == 1
    assert llm_events[0].total_tokens == 150
    assert llm_events[0].model == "gpt-4o-mini"
    assert llm_events[0].cost == pytest.approx(0.002)
    assert llm_events[0].elapsed_ms >= 0

    # ---- AFTER_TOOL：写入 ToolCallEvent，优先采用结果中的精确耗时 ----
    tool_result = ToolResult(
        content="tool-ok",
        metadata={"elapsed_seconds": 0.12},
        confidence=1.0,
    )
    mw.after_tool(ctx, {"name": "search", "arguments": {"q": "x"}}, tool_result)

    tool_events = log.tool_calls()
    assert len(tool_events) == 1
    assert tool_events[0].tool_name == "search"
    assert tool_events[0].elapsed_ms == pytest.approx(120.0)
    assert tool_events[0].success is True

    # ---- ON_EXIT_LOOP：聚合指标 ----
    ctx2 = Context()
    mw.on_exit_loop(ctx2)
    summary = ctx2.shared["observability"]["summary"]
    assert summary["llm_calls"] == 1
    assert summary["tool_calls"] == 1
    assert summary["tokens"]["total_tokens"] == 150
    assert summary["cost"] == pytest.approx(0.002)
    assert summary["llm_elapsed"]["avg_ms"] > 0

    # ---- 结构化 JSON 日志可解析 ----
    assert len(mw.logs) == 3
    parsed = [json.loads(line) for line in mw.logs]
    assert [p["event"] for p in parsed] == [
        EVENT_LLM_CALL, EVENT_TOOL_CALL, EVENT_LOOP_EXIT,
    ]
    assert parsed[0]["total_tokens"] == 150
    assert parsed[2]["llm_calls"] == 1

    # ---- 事件日志 JSON Lines 导出 ----
    lines = log.to_json_lines().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event_type"] == EVENT_LLM_CALL


# ============================================================================
# T2 — CostGuardMiddleware
# ============================================================================


def test_cost_guard_middleware():
    """T2: 从事件/响应累加 token 与 cost，超预算返回 abort_loop。"""
    ctx = Context()

    # ---- 场景 A：无事件可用时，退化为直接读取 response ----
    session_a = Session(max_tokens=100)
    mw_a = CostGuardMiddleware(session=session_a)

    r1 = mw_a.after_llm(ctx, {"usage": {"total_tokens": 60}})
    assert r1.control == ControlFlow.CONTINUE
    assert session_a.budget.used_tokens == 60

    r2 = mw_a.after_llm(ctx, {"usage": {"total_tokens": 50}})
    assert r2.control == ControlFlow.ABORT_LOOP, "超预算应中止整个 Loop"
    assert session_a.budget.used_tokens == 110
    assert r2.payload["budget"]["used_tokens"] == 110
    assert "token 预算超限" in r2.payload["reason"]

    # ---- 场景 B：从 LLMCallEvent 累加（与 Observability 协作路径） ----
    session_b = Session(max_tokens=100)
    log_b = session_b.event_log
    mw_b = CostGuardMiddleware(session=session_b)

    log_b.emit_llm_call(total_tokens=80, cost=0.001, model="mock")
    r3 = mw_b.after_llm(ctx, None)
    assert r3.control == ControlFlow.CONTINUE
    assert session_b.budget.used_tokens == 80
    assert session_b.budget.used_cost == pytest.approx(0.001)

    log_b.emit_llm_call(total_tokens=30, cost=0.001, model="mock")
    r4 = mw_b.after_llm(ctx, None)
    assert r4.control == ControlFlow.ABORT_LOOP
    assert session_b.budget.used_tokens == 110

    # 同一条事件不得被重复累加（幂等）
    r5 = mw_b.after_llm(ctx, None)
    assert r5.control == ControlFlow.ABORT_LOOP
    assert session_b.budget.used_tokens == 110

    # ---- 场景 C：成本预算维度 ----
    session_c = Session(max_cost=0.01)
    mw_c = CostGuardMiddleware(session=session_c)
    r6 = mw_c.after_llm(ctx, {"usage": {"total_tokens": 1}, "cost": 0.02})
    assert r6.control == ControlFlow.ABORT_LOOP
    assert "成本预算超限" in r6.payload["reason"]

    # ---- 场景 D：不限预算 → 永不中止 ----
    session_d = Session()  # max_tokens=0, max_cost=0 → 不限
    mw_d = CostGuardMiddleware(session=session_d)
    for _ in range(20):
        assert mw_d.after_llm(ctx, {"usage": {"total_tokens": 1000}}).control == (
            ControlFlow.CONTINUE
        )
    assert session_d.budget.used_tokens == 20000


# ============================================================================
# T3 — SafetyGuardMiddleware
# ============================================================================


def test_safety_guard_middleware():
    """T3: 高危工具 skip_tool，普通工具与白名单放行。"""
    registry = _make_registry()
    mw = SafetyGuardMiddleware(allowed_tools={"search"}, tool_registry=registry)
    ctx = Context()

    # 普通工具（且在白名单内）→ 放行
    r1 = mw.before_tool(ctx, {"name": "search", "arguments": {}})
    assert r1.control == ControlFlow.CONTINUE

    # 高危工具（requires_approval=True 且 risk_level=high）→ 跳过
    r2 = mw.before_tool(ctx, {"name": "rm_rf", "arguments": {}})
    assert r2.control == ControlFlow.SKIP_CURRENT_TOOL, "高危工具应被跳过"
    assert len(mw.blocked) == 1
    assert mw.blocked[0]["tool"] == "rm_rf"
    assert "requires_approval" in mw.blocked[0]["reason"]
    assert ctx.shared["safety_blocked"][0]["tool"] == "rm_rf"

    # 白名单可覆盖高危判定（显式授权的工具放行）
    mw_allow = SafetyGuardMiddleware(
        allowed_tools={"rm_rf"}, tool_registry=registry
    )
    r3 = mw_allow.before_tool(Context(), {"name": "rm_rf", "arguments": {}})
    assert r3.control == ControlFlow.CONTINUE
    assert mw_allow.blocked == []

    # 运行时白名单（ctx.shared）同样生效
    ctx_rt = Context()
    ctx_rt.shared["tool_whitelist"] = ["rm_rf"]
    r4 = mw.before_tool(ctx_rt, {"name": "rm_rf", "arguments": {}})
    assert r4.control == ControlFlow.CONTINUE

    # 未注册工具（无元数据）→ 放行，不得误伤
    r5 = mw.before_tool(Context(), {"name": "unknown_tool", "arguments": {}})
    assert r5.control == ControlFlow.CONTINUE

    # metadata 高危标记（无需 requires_approval）→ 跳过
    mw_meta = SafetyGuardMiddleware()
    ctx_meta = Context()
    ctx_meta.shared["current_tool_meta"] = {
        "name": "deploy",
        "requires_approval": False,
        "metadata": {"dangerous": True},
    }
    r6 = mw_meta.before_tool(ctx_meta, {"name": "deploy", "arguments": {}})
    assert r6.control == ControlFlow.SKIP_CURRENT_TOOL

    # 在真实 Loop 中：高危工具未被真正执行
    recorder = _Recorder()
    _run_loop(
        [
            _ToolCallStub("rm_rf"),
            SafetyGuardMiddleware(tool_registry=registry),
        ],
        [{"role": "user", "content": "删除文件"}],
        max_iterations=2,
        recorder=recorder,
    )
    assert recorder.calls == [], f"高危工具不应被执行，实际: {recorder.calls}"

    # 对照：去掉安全守卫后，高危工具被执行（证明"可单独启停"）
    recorder2 = _Recorder()
    _run_loop(
        [_ToolCallStub("rm_rf")],
        [{"role": "user", "content": "删除文件"}],
        max_iterations=2,
        recorder=recorder2,
    )
    assert recorder2.calls == ["rm_rf"]


# ============================================================================
# T4 — MemoryInjectorMiddleware
# ============================================================================


def test_memory_injector_middleware():
    """T4: build_context 生成的含记忆消息通过 payload 注入。"""
    cm = LayeredContextManager()
    cm.store(MemoryTrace.create(
        content="User prefers Python async programming",
        namespace="default",
        strength=0.9,
        tags=["python"],
    ))

    system_prompt = "You are a coding assistant. 相关记忆:\n{{memory_context}}"
    mw = MemoryInjectorMiddleware(
        context_manager=cm, system_prompt=system_prompt, namespace="default",
        inject_mode="replace",
    )

    user_msg = {"role": "user", "content": "Python async programming preferences"}
    cm.append(Message(role="user", content=user_msg["content"]))

    ctx = Context()
    result = mw.before_llm(ctx, [user_msg])

    assert "messages" in result.payload, "必须通过 payload 注入消息"
    injected = result.payload["messages"]
    assert injected[0]["role"] == "system"
    assert "Python async programming" in injected[0]["content"], (
        "长期记忆应被注入 system prompt"
    )
    assert "{{memory_context}}" not in injected[0]["content"], "占位符应被替换"
    assert mw.last_injected_system == injected[0]["content"]
    assert ctx.shared["memory_injected"]["namespace"] == "default"

    # 原对话消息被保留（默认 use_working_memory=False）
    assert any(m.get("role") == "user" for m in injected)

    # 无 ContextManager 时放行，不抛异常（可独立启停）
    mw_no_cm = MemoryInjectorMiddleware()
    r2 = mw_no_cm.before_llm(Context(), [user_msg])
    assert r2.control == ControlFlow.CONTINUE
    assert r2.payload == {}

    # 从 ctx.shared 解析 ContextManager（不依赖构造函数注入）
    mw_shared = MemoryInjectorMiddleware(namespace="default", inject_mode="replace")
    ctx_shared = Context()
    ctx_shared.shared["context_manager"] = cm
    ctx_shared.shared["system_prompt_template"] = system_prompt
    r3 = mw_shared.before_llm(ctx_shared, [user_msg])
    assert "messages" in r3.payload
    assert "Python async programming" in r3.payload["messages"][0]["content"]


# ============================================================================
# T5 — ContextCompressorMiddleware
# ============================================================================


def test_context_compressor_middleware():
    """T5: 消息总量接近上限时压缩并替换消息，未超限时放行。"""
    # ---- 未超限：原样放行 ----
    mw_big = ContextCompressorMiddleware(max_tokens=100000)
    ctx_big = Context()
    msgs = [{"role": "user", "content": "短消息"}]
    r1 = mw_big.before_llm(ctx_big, msgs)
    assert r1.control == ControlFlow.CONTINUE
    assert r1.payload == {}, "未超限不得覆盖 messages"
    assert mw_big.compressed_count == 0
    assert "context_compressed" not in ctx_big.shared

    # ---- 超限：调用 ContextManager.compress 并替换消息 ----
    cm = LayeredContextManager(max_working_memory=50)
    for i in range(30):
        cm.append(Message(role="user", content=f"很长的历史消息 #{i} " + "x" * 200))
    cm.append(Message(role="system", content="你是助手"))

    working = cm.get_working_memory()
    before_len = len(working)
    assert before_len > 10, "前置条件：工作记忆足够长"

    mw = ContextCompressorMiddleware(
        context_manager=cm, max_tokens=200, threshold_ratio=0.5, strategy="threshold"
    )
    ctx = Context()
    result = mw.before_llm(ctx, [m.to_dict() for m in working])

    assert "messages" in result.payload, "超限必须替换 messages"
    compressed = result.payload["messages"]
    assert len(compressed) < before_len, (
        f"压缩后消息数应减少: {before_len} -> {len(compressed)}"
    )
    assert mw.compressed_count == 1
    info = ctx.shared["context_compressed"]
    assert info["before_tokens"] > info["threshold"]
    assert info["after_tokens"] < info["before_tokens"]

    # ---- 无 ContextManager 时走本地截断兜底，仍能压缩 ----
    mw_fallback = ContextCompressorMiddleware(
        max_tokens=20, threshold_ratio=0.5, keep_recent=1
    )
    many = [{"role": "user", "content": "y" * 200} for _ in range(20)]
    r3 = mw_fallback.before_llm(Context(), many)
    assert "messages" in r3.payload
    assert len(r3.payload["messages"]) == 1, "兜底路径保留 keep_recent 条"

    # ---- token 估算口径与 L2 一致（字符数 / 4） ----
    assert ContextCompressorMiddleware._estimate_messages(
        [{"role": "user", "content": "a" * 400}]
    ) == 101


# ============================================================================
# T6 — 独立启停隔离性
# ============================================================================


def test_middlewares_independent_toggle():
    """T6: 每个中间件可单独启停，互不影响。"""
    messages = [{"role": "user", "content": "开始任务"}]

    # ---- 基线：只有 tool_call 桩，五个中间件全关 ----
    ctx0 = Context()
    _run_loop([_ToolCallStub("search")], messages, ctx=ctx0)
    assert ctx0.shared.get("observability") is None
    assert ctx0.shared.get("safety_blocked") is None
    assert ctx0.shared.get("memory_injected") is None
    assert ctx0.shared.get("context_compressed") is None
    assert ctx0.hook_errors == [], "基线不应有钩子异常"

    # ---- 仅启用 Observability ----
    log1 = EventLog()
    session1 = Session(event_log=log1)
    ctx1 = Context()
    _run_loop(
        [ObservabilityMiddleware(event_log=log1, session=session1, echo=False),
         _ToolCallStub("search")],
        messages,
        ctx=ctx1,
    )
    assert ctx1.shared["observability"]["records"] >= 2, "应记录 LLM + 工具事件"
    assert log1.count(EVENT_LLM_CALL) >= 1
    assert log1.count(EVENT_TOOL_CALL) >= 1
    # 仅观测不累加预算
    assert session1.budget.used_tokens == 0
    assert ctx1.shared.get("safety_blocked") is None
    assert ctx1.shared.get("context_compressed") is None
    assert ctx1.hook_errors == []

    # ---- 仅启用 CostGuard ----
    session2 = Session(max_tokens=1000)
    ctx2 = Context()
    _run_loop(
        [CostGuardMiddleware(session=session2), _ToolCallStub("search")],
        messages,
        ctx=ctx2,
    )
    assert session2.budget.used_tokens == 120, "两轮 LLM 调用各 60 token"
    assert ctx2.shared.get("observability") is None
    assert ctx2.hook_errors == []

    # ---- 仅启用 SafetyGuard ----
    registry = _make_registry()
    ctx3 = Context()
    recorder3 = _Recorder()
    _run_loop(
        [SafetyGuardMiddleware(tool_registry=registry), _ToolCallStub("rm_rf")],
        messages,
        recorder=recorder3,
        ctx=ctx3,
    )
    assert len(ctx3.shared["safety_blocked"]) == 1
    assert recorder3.calls == []
    assert ctx3.shared.get("context_compressed") is None
    assert ctx3.hook_errors == []

    # ---- 仅启用 MemoryInjector ----
    cm4 = LayeredContextManager()
    cm4.store(MemoryTrace.create(
        content="User prefers Python async programming",
        namespace="default",
        strength=0.9,
    ))
    cm4.append(Message(role="user", content="Python async programming"))
    ctx4 = Context()
    ctx4.shared["context_manager"] = cm4
    ctx4.shared["system_prompt_template"] = "助手。记忆:\n{{memory_context}}"
    _run_loop(
        [MemoryInjectorMiddleware(namespace="default"), _ToolCallStub("search")],
        messages,
        ctx=ctx4,
    )
    assert "memory_injected" in ctx4.shared
    assert ctx4.shared.get("context_compressed") is None
    assert ctx4.shared.get("safety_blocked") is None
    assert ctx4.hook_errors == []

    # ---- 仅启用 ContextCompressor ----
    ctx5 = Context()
    long_messages = [{"role": "user", "content": "z" * 400} for _ in range(10)]
    _run_loop(
        [ContextCompressorMiddleware(max_tokens=50, threshold_ratio=0.8, keep_recent=1),
         _ToolCallStub("search")],
        long_messages,
        ctx=ctx5,
    )
    assert ctx5.shared["context_compressed"]["count"] >= 1
    assert ctx5.shared.get("memory_injected") is None
    assert ctx5.shared.get("safety_blocked") is None
    assert ctx5.hook_errors == []

    # ---- 五个中间件同时启用：各自生效且互不干扰 ----
    log_all = EventLog()
    session_all = Session(max_tokens=1000, event_log=log_all)
    cm_all = LayeredContextManager()
    cm_all.store(MemoryTrace.create(
        content="User prefers Python async programming",
        namespace="default",
        strength=0.9,
    ))
    cm_all.append(Message(role="user", content="Python async programming"))
    ctx_all = Context()
    ctx_all.shared["context_manager"] = cm_all
    ctx_all.shared["system_prompt_template"] = "助手。记忆:\n{{memory_context}}"
    recorder_all = _Recorder()
    _run_loop(
        [
            ObservabilityMiddleware(event_log=log_all, session=session_all, echo=False),
            CostGuardMiddleware(session=session_all),
            SafetyGuardMiddleware(tool_registry=_make_registry()),
            MemoryInjectorMiddleware(namespace="default"),
            ContextCompressorMiddleware(max_tokens=50, threshold_ratio=0.8),
            _ToolCallStub("rm_rf"),
        ],
        long_messages,
        recorder=recorder_all,
        ctx=ctx_all,
    )
    assert ctx_all.shared["observability"]["records"] >= 3, "日志输出"
    assert session_all.budget.used_tokens == 120, "成本累计"
    assert len(ctx_all.shared["safety_blocked"]) == 1, "安全拦截"
    assert "memory_injected" in ctx_all.shared, "记忆注入"
    assert "context_compressed" in ctx_all.shared, "压缩触发"
    assert recorder_all.calls == [], "高危工具未被实际执行"
    assert ctx_all.hook_errors == [], "不应有中间件钩子异常"