"""
[L2-装配] Runtime 装配层集成测试。

覆盖范围：
    D. Runtime 装配（端到端 run / resume / 中间件 / Fiber / ctx.shared）
    E. L3 五中间件经 Runtime 装配后的协作
    F. 无回归兼容性（旧导入路径仍可用）

运行方式：python -m pytest tests/test_runtime_assembly.py -v
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Iterator, List, Optional

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
from runtime.context_manager.interface import MemoryTrace, Message  # noqa: E402
from runtime.context_manager.layered import LayeredContextManager  # noqa: E402
from runtime.event_log import (  # noqa: E402
    EVENT_LLM_CALL,
    EVENT_LOOP_EXIT,
    EVENT_LOOP_START,
    EVENT_TOOL_CALL,
    InMemoryEventLog,
    create_event_log,
)
from runtime.llm_adapter.interface import (  # noqa: E402
    LLMAdapter,
    LLMResponse,
    ModelConfig,
)
from runtime.runtime import ComponentRegistry, Runtime  # noqa: E402
from runtime.session import Session  # noqa: E402
from runtime.session_store import (  # noqa: E402
    InMemorySessionStore,
    SessionStore,
    create_session_store,
)
from runtime.tool_registry.in_memory import InMemoryToolRegistry  # noqa: E402
from runtime.tool_registry.interface import Tool, ToolResult  # noqa: E402


# ============================================================================
# 测试辅助：模拟 LLMAdapter 与工具
# ============================================================================


class ScriptedAdapter(LLMAdapter):
    """按剧本返回响应的模拟 LLMAdapter。

    剧本元素为 (tool_name, tool_args)；tool_name 为 None 表示本轮无工具调用。
    每轮附带固定的 usage / cost，便于断言预算累计。
    """

    def __init__(self, script: Optional[List[Any]] = None) -> None:
        self.script = list(script or [])
        self.turn = 0
        self.calls = 0
        self.last_messages: List[Dict[str, Any]] = []

    def _next(self) -> Any:
        if not self.script:
            return (None, None)
        index = min(self.turn, len(self.script) - 1)
        self.turn += 1
        return self.script[index]

    def call(
        self,
        messages: List[Dict[str, Any]],
        model_config: Optional[ModelConfig] = None,
        *,
        task_type: Optional[str] = None,
        context: Optional[Any] = None,
    ) -> LLMResponse:
        self.calls += 1
        self.last_messages = list(messages or [])
        tool_name, tool_args = self._next()

        tool_calls: List[Dict[str, Any]] = []
        if tool_name:
            tool_calls = [{
                "id": f"call_{self.calls}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(tool_args or {}),
                },
            }]

        return LLMResponse(
            content=f"第 {self.calls} 轮响应",
            tool_calls=tool_calls,
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            model="mock-model",
            provider="mock",
            cost=0.002,
            finish_reason="tool_calls" if tool_calls else "stop",
        )

    def stream(
        self, messages, model_config=None, *, task_type=None, context=None
    ) -> Iterator[str]:
        yield ""

    def count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        return sum(len(str(m.get("content", ""))) // 4 for m in messages or [])

    def get_cost(self, usage: Dict[str, int], model_config=None) -> float:
        return 0.002


def _make_registry() -> InMemoryToolRegistry:
    """注册普通工具 + 高危工具。"""
    registry = InMemoryToolRegistry()
    registry.register(Tool(
        name="search",
        description="普通检索工具",
        params_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        fn=lambda q="": f"检索: {q}",
    ))
    registry.register(Tool(
        name="rm_rf",
        description="高危删除工具",
        params_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        fn=lambda path="": f"已删除 {path}",
        requires_approval=True,
        metadata={"risk_level": "high", "dangerous": True},
    ))
    return registry


def _make_runtime(script=None, max_iterations=3, **overrides) -> Runtime:
    """构造一个装配完毕的 Runtime。

    overrides 可覆盖任意 ComponentRegistry 字段（如 context_manager / event_log）。
    """
    overrides.setdefault("context_manager", LayeredContextManager(max_working_memory=20))
    registry = ComponentRegistry(
        llm_adapter=ScriptedAdapter(script),
        tool_registry=_make_registry(),
        **overrides,
    )
    return Runtime(registry, max_iterations=max_iterations)


# ============================================================================
# D. Runtime 装配
# ============================================================================


def test_runtime_bare_run_without_middlewares():
    """D1: 无中间件时 Runtime 仍可完整运行（内核可剥离中间件）。"""
    runtime = _make_runtime(script=[("search", {"q": "deep agent"}), (None, None)])
    session = runtime.run("s-bare", "帮我检索", system_prompt="你是助手")

    assert isinstance(session, Session)
    assert session.session_id == "s-bare"
    assert session.metadata["task"] == "帮我检索"
    assert runtime.get_session("s-bare") is session


def test_runtime_defaults_are_wired():
    """D2: ComponentRegistry 未提供可选组件时使用默认实现。"""
    registry = ComponentRegistry(
        llm_adapter=ScriptedAdapter(),
        tool_registry=_make_registry(),
        context_manager=LayeredContextManager(),
    )
    assert registry.tool_call_parser is not None
    assert isinstance(registry.event_log, InMemoryEventLog)
    assert isinstance(registry.session_store, InMemorySessionStore)

    runtime = Runtime(registry, max_iterations=2)
    session = runtime.run("s-defaults", "空跑")
    assert session.session_id == "s-defaults"


def test_runtime_tool_call_parsing_and_execution():
    """D3: ToolCallParser 解析结果驱动 Loop 真实执行工具。"""
    executed: List[str] = []
    registry = ComponentRegistry(
        llm_adapter=ScriptedAdapter(script=[
            ("search", {"q": "hello"}),
            (None, None),
        ]),
        tool_registry=_make_registry(),
        context_manager=LayeredContextManager(),
    )
    original_execute = registry.tool_registry.execute

    def spy_execute(name: str, args: Dict[str, Any]) -> ToolResult:
        executed.append(name)
        return original_execute(name, args)

    registry.tool_registry.execute = spy_execute  # type: ignore[method-assign]

    runtime = Runtime(registry, max_iterations=3)
    ctx_holder: Dict[str, Any] = {}

    class _CtxCapture(Middleware):
        def on_exit_loop(self, ctx: Context) -> HookResult:
            ctx_holder["ctx"] = ctx
            return HookResult.continue_()

    runtime.register_middleware(_CtxCapture("capture"))
    runtime.run("s-tool", "检索 hello", system_prompt="你是助手")

    assert executed == ["search"], f"工具应被真实执行一次，实际: {executed}"

    assert registry.event_log.count(EVENT_LOOP_START) == 1
    assert registry.event_log.count(EVENT_LOOP_EXIT) == 1

    ctx = ctx_holder["ctx"]
    for key in ("event_log", "session", "context_manager", "tool_registry"):
        assert key in ctx.shared, f"ctx.shared 缺少约定键 {key}"
    assert ctx.shared["system_prompt_template"] == "你是助手"
    assert ctx.shared["event_log"] is registry.event_log


def test_runtime_budget_applies_and_accumulates():
    """D4: budget 参数生效；CostGuard 经 Runtime 累计预算。"""
    # [D001 已修复] 需每轮带一次工具调用才能跑满 max_iterations(3) 轮；
    # 剧本最后一项兜底复用，故 3 轮均有工具调用。
    runtime = _make_runtime(script=[("search", {"q": "x"}), ("search", {"q": "y"})])
    runtime.register_middleware(CostGuardMiddleware())

    # 有工具调用剧本 → Loop 跑满 max_iterations(3) 轮，每轮 150 token
    session = runtime.run("s-budget", "任务", budget={"max_tokens": 1000})
    assert session.budget.max_tokens == 1000
    assert session.budget.used_tokens == 450  # 3 轮 × 150

    # 复用同一 session_id 会延续已保存的用量（再跑 3 轮）
    session2 = runtime.run("s-budget", "任务", budget=1000)
    assert session2.session_id == "s-budget"
    assert session2.budget.used_tokens == 900


def test_runtime_resume_and_get_session():
    """D5: get_session / resume 语义与错误处理。"""
    store = create_session_store("sqlite", db_path=":memory:")
    runtime = _make_runtime(session_store=store)

    assert runtime.get_session("missing") is None
    with pytest.raises(KeyError):
        runtime.resume("missing")

    runtime.run("s-resume", "第一次")
    restored = runtime.resume("s-resume")
    assert restored.session_id == "s-resume"
    assert restored.event_log is runtime.registry.event_log

    runtime.close()


def test_runtime_uses_sqlite_event_log_backend():
    """D6: SQLite 事件日志后端在 Runtime 中按会话分组写入。"""
    log = create_event_log("sqlite", db_path=":memory:")
    runtime = _make_runtime(event_log=log)

    runtime.run("s-sqlite", "任务一")
    runtime.run("s-sqlite-2", "任务二")

    assert sorted(log.session_ids()) == ["s-sqlite", "s-sqlite-2"]
    assert log.count(EVENT_LOOP_START) == 2
    assert log.count(EVENT_LOOP_EXIT) == 2
    runtime.close()


def test_runtime_middleware_and_fiber_management():
    """D7: register_middleware / load_plugins / dispose_fiber 行为。"""

    class _RecorderMw(Middleware):
        def __init__(self) -> None:
            super().__init__("recorder_mw")
            self.entered = 0

        def on_enter_loop(self, ctx: Context) -> HookResult:
            self.entered += 1
            return HookResult.continue_()

    runtime = _make_runtime(script=[(None, None)])
    mw = _RecorderMw()
    runtime.register_middleware(mw)
    assert mw in runtime._middlewares

    runtime.run("s-mw", "任务")
    assert mw.entered == 1

    builtin_dir = os.path.join(
        os.path.dirname(__file__), "..", "runtime", "builtin_middlewares"
    )
    fiber = runtime.load_plugins([builtin_dir], fiber_id="fiber-1")
    assert fiber.fiber_id == "fiber-1"
    # Fiber 记录的是中间件实例名（name 属性），非类名
    assert "observability" in fiber.middleware_names
    assert "safety_guard" in fiber.middleware_names
    assert "fiber-1" in runtime._fibers

    runtime.dispose_fiber("fiber-1")
    runtime.dispose_fiber("fiber-1")  # 幂等
    assert "fiber-1" not in runtime._fibers


def test_runtime_uses_context_manager_from_shared():
    """D8: registry.context_manager 经 ctx.shared 传给中间件（无构造注入）。"""
    cm = LayeredContextManager()
    cm.store(MemoryTrace.create(
        content="User prefers Python async programming",
        namespace="default",
        strength=0.9,
    ))

    # 注意：build_context 以"工作记忆中最后一条 user 消息"作为检索 query，
    # 因此需追加匹配的记忆内容，检索才会命中长期记忆。
    cm.append(Message(role="user", content="Python async programming"))

    runtime = _make_runtime(script=[(None, None)], context_manager=cm)
    injector = MemoryInjectorMiddleware(namespace="default", inject_mode="replace")
    runtime.register_middleware(injector)

    runtime.run("s-cm", "任务", system_prompt="你是助手。记忆:\n{{memory_context}}")

    assert "Python async programming" in injector.last_injected_system


# ============================================================================
# E. L3 五中间件经 Runtime 装配的协作
# ============================================================================


def test_l3_middlewares_together_via_runtime():
    """E1: 五个 L3 中间件全部经 Runtime 装配 → 各自生效。"""
    event_log = InMemoryEventLog()
    cm = LayeredContextManager(max_working_memory=30)
    cm.store(MemoryTrace.create(
        content="User prefers Python async programming",
        namespace="default",
        strength=0.95,
    ))
    for i in range(24):
        cm.append(Message(role="user", content=f"历史 #{i}: " + "x" * 200))
    # 末条 user 消息作为检索 query，需匹配长期记忆内容
    cm.append(Message(role="user", content="Python async programming"))

    registry = ComponentRegistry(
        llm_adapter=ScriptedAdapter(script=[
            ("search", {"q": "deep agent"}),
            ("rm_rf", {"path": "/data"}),
            (None, None),
        ]),
        tool_registry=_make_registry(),
        context_manager=cm,
        event_log=event_log,
    )
    runtime = Runtime(registry, max_iterations=5)

    observability = ObservabilityMiddleware(event_log=event_log, echo=False)
    cost_guard = CostGuardMiddleware(max_tokens=100000)
    safety_guard = SafetyGuardMiddleware(allowed_tools={"search"})
    memory_injector = MemoryInjectorMiddleware(namespace="default", inject_mode="replace")
    compressor = ContextCompressorMiddleware(
        context_manager=cm, max_tokens=500, threshold_ratio=0.5, strategy="threshold"
    )
    for mw in (observability, cost_guard, safety_guard, memory_injector, compressor):
        runtime.register_middleware(mw)

    # 压缩中间件基于"本轮 messages"估算 token，故 task 需足够长以触发压缩
    long_task = "研究 deep agent 记忆架构并给出建议。" + "上下文压缩与记忆注入。" * 80
    session = runtime.run(
        "s-l3",
        long_task,
        system_prompt="你是研究助手。记忆:\n{{memory_context}}",
    )

    assert event_log.count(EVENT_LLM_CALL) >= 1
    assert event_log.count(EVENT_TOOL_CALL) >= 1
    assert observability.records

    assert session.budget.used_tokens > 0
    assert any(b["tool"] == "rm_rf" for b in safety_guard.blocked)
    assert "Python async programming" in memory_injector.last_injected_system
    assert compressor.compressed_count >= 1

    # EventRecorder 写入的 loop 级事件与 L3 事件类型不冲突
    assert event_log.count(EVENT_LOOP_START) == 1
    assert event_log.count(EVENT_LOOP_EXIT) == 1


def test_l3_middlewares_do_not_break_safety_execution():
    """E2: 经 Runtime 装配后，高危工具不被真实执行。"""
    executed: List[str] = []
    registry = ComponentRegistry(
        llm_adapter=ScriptedAdapter(script=[
            ("rm_rf", {"path": "/data"}),
            (None, None),
        ]),
        tool_registry=_make_registry(),
        context_manager=LayeredContextManager(),
    )
    original_execute = registry.tool_registry.execute

    def spy_execute(name: str, args: Dict[str, Any]) -> ToolResult:
        executed.append(name)
        return original_execute(name, args)

    registry.tool_registry.execute = spy_execute  # type: ignore[method-assign]

    runtime = Runtime(registry, max_iterations=3)
    safety = SafetyGuardMiddleware()
    runtime.register_middleware(safety)
    runtime.run("s-safety", "删除数据")

    assert executed == [], f"高危工具不应被执行: {executed}"
    assert any(b["tool"] == "rm_rf" for b in safety.blocked)


# ============================================================================
# F. 无回归兼容性
# ============================================================================


def test_package_exports_present():
    """F1: runtime 包与各子包的关键符号均可导入。"""
    import runtime
    import runtime.event_log as event_log_pkg
    import runtime.session_store as session_store_pkg
    import runtime.event_recorder as recorder_pkg
    import runtime.runtime as runtime_pkg

    assert hasattr(runtime_pkg, "Runtime")
    assert hasattr(runtime_pkg, "ComponentRegistry")
    assert hasattr(recorder_pkg, "EventRecorder")
    assert hasattr(session_store_pkg, "create_session_store")
    assert hasattr(event_log_pkg, "create_event_log")

    for name in ("event_log", "session", "session_store", "event_recorder", "runtime"):
        assert name in runtime.__all__


def test_old_event_log_module_path_replaced_by_package():
    """F2: 旧的单文件模块已由包取代（包优先，符号不变）。"""
    import runtime.event_log as pkg

    assert pkg.__file__.endswith(os.path.join("event_log", "__init__.py"))
    for symbol in (
        "EventLog", "resolve_event_log", "EVENT_LLM_CALL", "EVENT_TOOL_CALL",
        "EVENT_LOOP_START", "EVENT_LOOP_EXIT", "EVENT_ERROR",
        "Event", "LLMCallEvent", "ToolCallEvent",
    ):
        assert hasattr(pkg, symbol), f"缺少兼容符号 {symbol}"


def test_l3_middlewares_import_contract_unchanged():
    """F3: L3 五中间件仍可从原路径导入并使用（未修改其内部逻辑）。"""
    from runtime.event_log import EVENT_LLM_CALL as _e  # noqa: F401
    from runtime.session import Session as _s  # noqa: F401

    mw = SafetyGuardMiddleware()
    result = mw.before_tool(Context(), {"name": "unknown", "arguments": {}})
    assert result.control.name == "CONTINUE"


def test_session_store_base_class_importable_from_legacy_path():
    """F4: SessionStore 从 session 与 session_store 两处导入指向同一类。"""
    from runtime.session import SessionStore as Legacy
    from runtime.session_store import SessionStore as Canonical

    assert Legacy is Canonical