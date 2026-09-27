"""[L2-ASSEMBLY-TEST] L2 装配层端到端集成测试。

覆盖场景（5 个）：
    T1  Echo 管线跑通        —— EchoProvider + InMemoryToolRegistry + LayeredContextManager
                                经 ComponentRegistry 注入 Runtime 后可完整运行；
                                事件日志留下 loop_start / loop_exit；工作记忆含 user 消息。
    T2  Session 恢复         —— run 完成后从 SessionStore 取回会话，resume 返回同一会话，
                                session_id / budget / metadata 保持一致。
    T3  预算熔断             —— 装载 CostGuardMiddleware，budget={"max_tokens": 1} 时
                                session.abort_flag 置位 / budget.exceeded()，且 llm_call 有记录。
    T4  EventLog 后端一致性  —— memory 与 sqlite 后端写入同一事件序列后，
                                count / aggregate_tokens / aggregate_cost / summary 完全一致。
    T5  5 个 L3 中间件全装载 —— Observability + CostGuard + SafetyGuard + MemoryInjector
                                + ContextCompressor 协同运行，无异常、会话正常结束，
                                事件日志含 llm_call + tool_call + loop_start + loop_exit。

约束：
    - 仅使用标准库 + 项目已有模块；
    - 不依赖真实 LLM（统一基于 EchoProvider，无网络请求）；
    - 每个测试自建装配、自建会话 ID，彼此独立、可单独运行；
    - 不修改任何现有源文件；测试内仅使用少量显式标注的"测试脚手架"。

运行方式：python -m pytest tests/test_runtime_e2e.py -v
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_loop import Context, HookResult, Middleware  # noqa: E402

from runtime.builtin_middlewares import (  # noqa: E402
    ContextCompressorMiddleware,
    CostGuardMiddleware,
    MemoryInjectorMiddleware,
    ObservabilityMiddleware,
    SafetyGuardMiddleware,
)
from runtime.context_manager.interface import Message  # noqa: E402
from runtime.context_manager.layered import LayeredContextManager  # noqa: E402
from runtime.event_log import (  # noqa: E402
    EVENT_LLM_CALL,
    EVENT_LOOP_EXIT,
    EVENT_LOOP_START,
    EVENT_TOOL_CALL,
    InMemoryEventLog,
    create_event_log,
)
from runtime.event_log.interface import EventLog  # noqa: E402
from runtime.llm_adapter.interface import ModelConfig  # noqa: E402
from runtime.llm_adapter.multi_provider import (  # noqa: E402
    EchoProvider,
    MultiProviderAdapter,
)
from runtime.runtime import ComponentRegistry, Runtime  # noqa: E402
from runtime.session import Session  # noqa: E402
from runtime.tool_registry import (  # noqa: E402
    InMemoryToolRegistry,
    function_tool,
)


# ============================================================================
# 测试脚手架（仅测试内使用，不属于产品代码；不修改任何源文件）
# ============================================================================


@function_tool(
    name_override="echo",
    description_override="回显输入文本，用于验证工具注册与执行链路。",
)
def echo(text: str = "") -> str:
    """回显输入文本。

    Args:
        text: 要回显的文本。
    """
    return f"echo: {text}"


def make_echo_tool_registry() -> InMemoryToolRegistry:
    """构造仅注册一个 @function_tool 版 echo 工具的 InMemoryToolRegistry。"""
    registry = InMemoryToolRegistry()
    registry.register(echo)
    return registry


def make_echo_adapter(provider: Optional[EchoProvider] = None) -> MultiProviderAdapter:
    """构造以 EchoProvider 为唯一 Provider 的 MultiProviderAdapter。

    不产生任何网络请求：EchoProvider 直接回显输入消息。
    """
    adapter = MultiProviderAdapter(
        default_config=ModelConfig(provider="echo", model="echo-test"),
        max_context_tokens=128000,
    )
    adapter.register_provider(provider if provider is not None else EchoProvider())
    return adapter


class EchoToolCallProvider(EchoProvider):
    """在首轮调用中附带一次工具调用的 EchoProvider（测试脚手架）。

    产品内的 EchoProvider 永不产生 tool_calls，而 T5 需要验证
    "LLM 请求工具 → 解析 → 执行 → 记录 tool_call 事件" 的完整链路。
    因此以 EchoProvider 为基底，仅在首轮注入一个 OpenAI 形状的 tool_call，
    其余行为（无网络、无真实 LLM）完全沿用 EchoProvider。
    """

    TOOL_NAME = "echo"

    def call(self, messages: List[Dict[str, Any]], config: ModelConfig):
        self._calls = getattr(self, "_calls", 0) + 1
        response = super().call(messages, config)
        if self._calls == 1:
            response.tool_calls = [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": self.TOOL_NAME,
                    "arguments": json.dumps({"text": "hi"}),
                },
            }]
            response.finish_reason = "tool_calls"
        return response


class WorkingMemoryBridge(Middleware):
    """把 Loop 的 user 消息桥接到 ContextManager 工作记忆（测试脚手架）。

    依据 ARCHITECTURE.md 的契约："消息追加由中间件在 AFTER_LLM/AFTER_TOOL
    钩子中触发"。L3 的五个中间件均不负责追加消息，因此 T1 用这个最小的
    测试脚手架来验证：Runtime 把 ComponentRegistry.context_manager 正确
    暴露给中间件，且 Loop 中的 user 消息可以真正落到工作记忆里。
    """

    def __init__(self, context_manager: LayeredContextManager) -> None:
        super().__init__("test_working_memory_bridge")
        self._cm = context_manager
        self._synced = 0

    def before_llm(self, ctx: Context, messages: List[Dict[str, Any]]) -> HookResult:
        users = [
            m for m in (messages or [])
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        for msg in users[self._synced:]:
            self._cm.append(Message(role="user", content=str(msg.get("content", ""))))
        self._synced = len(users)
        return HookResult.continue_()


def capture_context(runtime: Runtime) -> Dict[str, Any]:
    """捕获 Runtime 本次运行结束后的 Context（测试脚手架）。

    Runtime.run 只返回 Session，不暴露 Context；AgentLoop 自身的全局异常
    保护会吞掉钩子异常并记录到 ctx.hook_errors。为了在 T5 断言"无异常"，
    这里包一层 _loop.run 以取回 Loop 返回的 Context。
    """
    holder: Dict[str, Any] = {}
    original_run = runtime._loop.run

    def _spy_run(ctx: Context, messages: List[Dict[str, Any]]) -> Context:
        result = original_run(ctx, messages)
        holder["ctx"] = result
        return result

    runtime._loop.run = _spy_run  # type: ignore[method-assign]
    return holder


def write_identical_sequence(log: EventLog) -> None:
    """向任意 EventLog 后端写入同一组确定性事件序列。"""
    log.emit_llm_call(
        iteration=1, model="mock-model", provider="mock",
        prompt_tokens=100, completion_tokens=50, total_tokens=150,
        cost=0.002, elapsed_ms=12.5, success=True,
    )
    log.emit_llm_call(
        iteration=1, model="mock-model", provider="mock",
        prompt_tokens=40, completion_tokens=10, total_tokens=50,
        cost=0.001, elapsed_ms=7.5, success=True,
    )
    log.emit_tool_call(
        iteration=1, tool_name="echo", arguments={"text": "hi"},
        success=True, elapsed_ms=3.0,
    )
    log.emit(EVENT_LOOP_START, session_id="s-e2e")


# ============================================================================
# T1 —— Echo 管线跑通
# ============================================================================


def test_t1_echo_pipeline_runs_end_to_end():
    """T1: EchoProvider + InMemoryToolRegistry + LayeredContextManager 经 Runtime 跑通。

    断言：
        - Runtime.run("s1", "hi") 返回可用的 Session 且被持久化
        - EventLog 含 loop_start / loop_exit 生命周期事件
        - ContextManager 工作记忆中确实包含了本轮 user 消息
    """
    event_log = InMemoryEventLog()
    context_manager = LayeredContextManager(max_working_memory=50)
    tool_registry = make_echo_tool_registry()

    # 组件经 ComponentRegistry 注入：EchoProvider / InMemoryToolRegistry / LayeredContextManager
    registry = ComponentRegistry(
        llm_adapter=make_echo_adapter(EchoProvider()),
        tool_registry=tool_registry,
        context_manager=context_manager,
        event_log=event_log,
    )
    runtime = Runtime(registry, max_iterations=1)
    # 测试脚手架：把 user 消息桥接到工作记忆（见 WorkingMemoryBridge 说明）
    runtime.register_middleware(WorkingMemoryBridge(context_manager))

    session = runtime.run("s1", "hi")

    # 1) Session 存在且被持久化
    assert isinstance(session, Session)
    assert session.session_id == "s1"
    assert runtime.get_session("s1") is session

    # 2) 生命周期事件齐备（由 EventRecorder 写入）
    assert event_log.count(EVENT_LOOP_START) == 1
    assert event_log.count(EVENT_LOOP_EXIT) == 1
    assert event_log.last(EVENT_LOOP_EXIT) is not None

    # 3) ContextManager 工作记忆含 user 消息
    working = context_manager.get_working_memory()
    assert len(working) >= 1, f"工作记忆不应为空: {working!r}"
    assert any(
        getattr(m, "role", "") == "user" and "hi" in str(getattr(m, "content", ""))
        for m in working
    ), f"工作记忆应包含 user 消息，实际: {working!r}"

    # 4) 装配一致性：Runtime 使用的正是注入的组件
    assert registry.context_manager is context_manager
    assert registry.tool_registry is tool_registry
    assert registry.llm_adapter.get_provider("echo") is not None

    # 5) @function_tool 装饰器确实生成了可注册/可执行的 Tool
    echo_tool = tool_registry.get("echo")
    assert echo_tool is not None
    assert echo_tool.name == "echo"
    assert echo_tool.fn is not None
    assert echo_tool.params_schema["parameters"]["properties"]["text"]["type"] == "string"
    assert tool_registry.execute("echo", {"text": "x"}).content == "echo: x"

    # T1 未装载 Observability，故不应出现 LLM 事件（事件类型由职责边界决定）
    assert event_log.count(EVENT_LLM_CALL) == 0
    assert event_log.count() == 2  # 仅 loop_start + loop_exit


# ============================================================================
# T2 —— Session 恢复
# ============================================================================


def test_t2_session_persisted_and_resumed():
    """T2: run 完成后取回会话 → resume 返回同一会话，核心状态一致。

    断言：session_id / budget / metadata 在 resume 前后一致。
    """
    registry = ComponentRegistry(
        llm_adapter=make_echo_adapter(),
        tool_registry=make_echo_tool_registry(),
        context_manager=LayeredContextManager(),
    )
    runtime = Runtime(registry, max_iterations=1)

    session = runtime.run(
        "s1", "hi", budget={"max_tokens": 1000, "max_cost": 2.0}
    )
    assert session.budget.max_tokens == 1000

    # 从 SessionStore 取出会话
    stored = runtime.get_session("s1")
    assert stored is not None
    assert stored is session
    assert stored.session_id == "s1"

    # 模拟"中断点"：在持久化的会话上补记已消耗用量
    stored.add_usage(tokens=42, cost=0.01)
    runtime.registry.session_store.save(stored)

    # resume 返回同一会话
    resumed = runtime.resume("s1")
    assert resumed is stored

    # 状态一致：session_id / budget / metadata
    assert resumed.session_id == "s1"
    assert resumed.budget.max_tokens == 1000
    assert resumed.budget.max_cost == pytest.approx(2.0)
    assert resumed.budget.used_tokens == 42
    assert resumed.budget.used_cost == pytest.approx(0.01)
    assert resumed.metadata["task"] == "hi"

    # resume 会重新绑定 registry 的事件日志
    assert resumed.event_log is registry.event_log

    # 二次 resume 幂等
    again = runtime.resume("s1")
    assert again is resumed
    assert again.budget.used_tokens == 42

    # 不存在的会话不可恢复
    with pytest.raises(KeyError):
        runtime.resume("missing")


# ============================================================================
# T3 —— 预算熔断
# ============================================================================


def test_t3_budget_circuit_breaker_aborts_loop():
    """T3: CostGuard 在超预算时中止 Loop，事件日志仍留痕。

    EchoProvider 单次调用即消耗 10+ token；预算上限设为 1，
    因此首轮 AFTER_LLM 即判定超限并 abort_loop。
    """
    event_log = InMemoryEventLog()
    registry = ComponentRegistry(
        llm_adapter=make_echo_adapter(),
        tool_registry=make_echo_tool_registry(),
        context_manager=LayeredContextManager(),
        event_log=event_log,
    )
    runtime = Runtime(registry, max_iterations=5)
    # Observability 先注册（写入 LLM 事件），CostGuard 后注册（读取并熔断）
    runtime.register_middleware(
        ObservabilityMiddleware(event_log=event_log, echo=False)
    )
    cost_guard = CostGuardMiddleware()
    runtime.register_middleware(cost_guard)

    session = runtime.run("s1", "hi", budget={"max_tokens": 1})

    # 熔断生效
    assert session.abort_flag is True
    assert session.budget.exceeded() is True
    assert cost_guard.abort_reason != ""

    # 事件日志有 LLM 调用记录
    llm_events = event_log.llm_calls()
    assert len(llm_events) >= 1
    assert event_log.count(EVENT_LLM_CALL) >= 1
    assert sum(e.total_tokens for e in llm_events) >= 10

    # 生命周期事件仍然完整（中止走的是正常退出路径）
    assert event_log.count(EVENT_LOOP_START) == 1
    assert event_log.count(EVENT_LOOP_EXIT) == 1


# ============================================================================
# T4 —— EventLog 后端一致性
# ============================================================================


def test_t4_event_log_backends_are_consistent():
    """T4: memory 与 sqlite 后端写同一序列后，全部聚合结果一致。"""
    memory_log = create_event_log("memory")
    sqlite_log = create_event_log("sqlite", db_path=":memory:")

    try:
        write_identical_sequence(memory_log)
        write_identical_sequence(sqlite_log)

        # 基础计数一致
        assert memory_log.count() == sqlite_log.count() == 4
        assert (
            memory_log.count(EVENT_LLM_CALL)
            == sqlite_log.count(EVENT_LLM_CALL)
            == 2
        )
        assert (
            memory_log.count(EVENT_TOOL_CALL)
            == sqlite_log.count(EVENT_TOOL_CALL)
            == 1
        )

        # 事件还原一致
        assert len(memory_log.llm_calls()) == len(sqlite_log.llm_calls())
        assert len(memory_log.tool_calls()) == len(sqlite_log.tool_calls())
        assert memory_log.last().event_type == sqlite_log.last().event_type
        assert (
            memory_log.tool_calls()[0].tool_name
            == sqlite_log.tool_calls()[0].tool_name
            == "echo"
        )
        assert (
            memory_log.tool_calls()[0].arguments
            == sqlite_log.tool_calls()[0].arguments
            == {"text": "hi"}
        )

        # token / cost 聚合一致
        assert memory_log.aggregate_tokens() == sqlite_log.aggregate_tokens()
        assert memory_log.aggregate_tokens()["total_tokens"] == 200
        assert memory_log.aggregate_cost() == pytest.approx(
            sqlite_log.aggregate_cost()
        )
        assert memory_log.aggregate_cost() == pytest.approx(0.003)

        # 耗时聚合一致
        assert (
            memory_log.aggregate_elapsed(EVENT_LLM_CALL)
            == sqlite_log.aggregate_elapsed(EVENT_LLM_CALL)
        )
        assert (
            memory_log.aggregate_elapsed(EVENT_TOOL_CALL)
            == sqlite_log.aggregate_elapsed(EVENT_TOOL_CALL)
        )

        # 完整 summary 一致
        assert memory_log.summary() == sqlite_log.summary()

        # JSON Lines 行数与事件类型序列一致
        memory_lines = memory_log.to_json_lines().splitlines()
        sqlite_lines = sqlite_log.to_json_lines().splitlines()
        assert len(memory_lines) == len(sqlite_lines) == 4
        assert [json.loads(l)["event_type"] for l in memory_lines] == [
            json.loads(l)["event_type"] for l in sqlite_lines
        ]
    finally:
        sqlite_log.close()


# ============================================================================
# T5 —— 五个 L3 中间件全部装载
# ============================================================================


def test_t5_all_l3_middlewares_loaded_and_cooperate():
    """T5: 五个 L3 中间件全部装载后协同运行，无异常且四类事件齐备。

    断言：
        - 无钩子异常、会话正常结束（未熔断）
        - EventLog 含 llm_call + tool_call + loop_start + loop_exit
    """
    event_log = InMemoryEventLog()
    context_manager = LayeredContextManager(max_working_memory=50)

    registry = ComponentRegistry(
        llm_adapter=make_echo_adapter(EchoToolCallProvider()),
        tool_registry=make_echo_tool_registry(),
        context_manager=context_manager,
        event_log=event_log,
    )
    runtime = Runtime(registry, max_iterations=2)
    captured = capture_context(runtime)

    observability = ObservabilityMiddleware(event_log=event_log, echo=False)
    cost_guard = CostGuardMiddleware()
    safety_guard = SafetyGuardMiddleware()
    memory_injector = MemoryInjectorMiddleware(namespace="default", inject_mode="replace")
    compressor = ContextCompressorMiddleware(
        context_manager=context_manager,
        max_tokens=100000,
        threshold_ratio=0.9,
        strategy="threshold",
    )
    for mw in (
        observability,
        cost_guard,
        safety_guard,
        memory_injector,
        compressor,
    ):
        runtime.register_middleware(mw)

    session = runtime.run(
        "s1",
        "test",
        system_prompt="你是助手 {{memory_context}}",
    )

    # 1) 无异常 + 正常结束
    ctx = captured["ctx"]
    assert ctx is not None
    assert ctx.hook_errors == [], f"不应有钩子异常: {ctx.hook_errors!r}"
    assert ctx.exit_errors == [], f"不应有退出异常: {ctx.exit_errors!r}"
    assert session.abort_flag is False
    assert session.budget.exceeded() is False
    assert runtime.get_session("s1") is session

    # 2) 事件日志四类事件齐备
    assert event_log.count(EVENT_LOOP_START) == 1
    assert event_log.count(EVENT_LOOP_EXIT) == 1
    assert event_log.count(EVENT_LLM_CALL) >= 1
    assert event_log.count(EVENT_TOOL_CALL) >= 1, "应至少记录一次工具调用事件"

    # 3) 中间件各自生效
    assert observability.records, "可观测性中间件应产出记录"
    assert len(ctx.shared.get("safety_blocked", [])) == 0  # echo 非高危，未被拦截
    assert memory_injector.last_injected_system != ""      # 记忆注入已执行
    assert "{{memory_context}}" not in memory_injector.last_injected_system

    # 4) 工具确实被执行过（通过事件日志的 tool_name 佐证）
    assert any(e.tool_name == "echo" for e in event_log.tool_calls())