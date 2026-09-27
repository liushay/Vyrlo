"""
[B1/D002] 子 Agent Context 继承 —— ``Runtime.run`` 不得丢弃 ``child_ctx``。

背景（DEFECTS.md D002）：
    修复前 ``Runtime.run()`` 内部调用 ``_build_context()`` 重新生成 ctx，
    **完全丢弃** ``_run_delegation`` 传入的 ``child_ctx``，导致 sandbox /
    shared_state / agent_registry 的继承全部失效。

修复后：
    - ``run()`` 拆分为 ``run()``（顶层，构建 ctx）与 ``_run_with_ctx(ctx, messages, session)``
      （内部，在既有 ctx 上执行）。
    - ``_run_delegation`` 走 ``_run_with_ctx(child_ctx, ...)``，因此继承字段真实生效。

本文件覆盖：
    1. test_child_ctx_inherits_sandbox
    2. test_child_ctx_inherits_shared_state
    3. test_child_ctx_inherits_agent_registry

运行方式：
    python -m pytest tests/test_runtime_child_ctx.py -v
"""

from __future__ import annotations

from typing import List

from agent_loop import Context, HookResult, Middleware
from runtime.agent_registry.in_memory import InMemoryAgentRegistry
from runtime.agent_registry.interface import AgentSpec
from runtime.runtime import Runtime
from runtime.sandbox.registry import SandboxRegistry
from runtime.shared_state.in_memory import InMemorySharedStateStore
from tests.fakes.fake_llm import FakeLLMProvider
from tests.fakes.fake_sandbox import FakeSandboxProvider
from tests.integration.conftest import build_runtime


class _SeenCtxProbe(Middleware):
    """记录每次 on_enter_loop 实际生效的 ctx。"""

    def __init__(self) -> None:
        super().__init__("seen_ctx_probe")
        self.seen: List[Context] = []

    def on_enter_loop(self, ctx: Context) -> HookResult:
        self.seen.append(ctx)
        return HookResult.continue_()


def _prepare_runtime(**kwargs) -> tuple:
    """构造 Runtime + 探针 + 父 ctx，返回 (runtime, probe, parent_ctx, sub)。"""
    llm = FakeLLMProvider()
    llm.enqueue_empty()  # 子 Agent 单轮无工具调用 → 自然结束

    runtime = build_runtime(llm, [], max_delegation_depth=3, **kwargs)

    probe = _SeenCtxProbe()
    runtime.register_middleware(probe)

    parent_ctx = Context()
    parent_ctx.shared["session"] = type("S", (), {"session_id": "parent"})()
    parent_ctx.shared["event_log"] = runtime.registry.event_log
    return runtime, probe, parent_ctx, llm


# ============================================================================
# 1. sandbox 继承
# ============================================================================


def test_child_ctx_inherits_sandbox():
    """子 Agent 实际执行时，ctx.shared["sandbox"] 必须是父的 handle。"""
    sandbox = FakeSandboxProvider(available=True)
    registry = SandboxRegistry({"local": sandbox})
    agent_registry = InMemoryAgentRegistry()
    spec = AgentSpec(agent_id="child", description="child", sandbox_mode="local")
    agent_registry.register(spec)

    runtime, probe, parent_ctx, _ = _prepare_runtime(agent_registry=agent_registry)

    parent_handle = sandbox.acquire("parent")
    parent_ctx.shared["sandbox"] = parent_handle

    result = runtime._run_delegation(parent_ctx, spec, "goal", "ok")

    # 返回值携带的 child_ctx 含父 sandbox
    assert result["child_context"].shared.get("sandbox") is parent_handle
    # 子 Agent 运行时真正生效的 ctx 也含父 sandbox（D002 的核心修复点）
    assert probe.seen, "子 Agent 应至少执行一次 on_enter_loop"
    assert probe.seen[-1].shared.get("sandbox") is parent_handle


# ============================================================================
# 2. shared_state 继承
# ============================================================================


def test_child_ctx_inherits_shared_state():
    """子 Agent 实际执行时，ctx.shared["shared_state"] 必须是父的 store 实例。"""
    shared_state = InMemorySharedStateStore()
    shared_state.set("ns", "k", "v")

    agent_registry = InMemoryAgentRegistry()
    spec = AgentSpec(agent_id="child", description="child")
    agent_registry.register(spec)

    runtime, probe, parent_ctx, _ = _prepare_runtime(
        agent_registry=agent_registry,
        shared_state=shared_state,
    )
    # 父 ctx 也持有同一 store
    parent_ctx.shared["shared_state"] = shared_state

    result = runtime._run_delegation(parent_ctx, spec, "goal", "ok")

    assert result["child_context"].shared.get("shared_state") is shared_state
    assert probe.seen, "子 Agent 应至少执行一次 on_enter_loop"
    assert probe.seen[-1].shared.get("shared_state") is shared_state
    # 子 Agent 能读到父写入的共享状态
    assert probe.seen[-1].shared["shared_state"].get("ns", "k") == "v"


# ============================================================================
# 3. agent_registry 继承
# ============================================================================


def test_child_ctx_inherits_agent_registry():
    """子 Agent 实际执行时，ctx.shared["agent_registry"] 必须是父的注册表。

    这保证子 Agent 内的 DelegationMiddleware 能继续做嵌套委派解析。
    """
    agent_registry = InMemoryAgentRegistry()
    spec = AgentSpec(agent_id="child", description="child")
    other = AgentSpec(agent_id="grandchild", description="grandchild")
    agent_registry.register(spec)
    agent_registry.register(other)

    runtime, probe, parent_ctx, _ = _prepare_runtime(agent_registry=agent_registry)
    parent_ctx.shared["agent_registry"] = agent_registry

    result = runtime._run_delegation(parent_ctx, spec, "goal", "ok")

    assert result["child_context"].shared.get("agent_registry") is agent_registry
    assert probe.seen, "子 Agent 应至少执行一次 on_enter_loop"
    seen_registry = probe.seen[-1].shared.get("agent_registry")
    assert seen_registry is agent_registry
    # 子 Agent 能通过继承的注册表解析到其它 Agent
    assert seen_registry.get("grandchild") is not None


# ============================================================================
# 4. 顶层 run() 语义不变（回归保护）
# ============================================================================


def test_top_level_run_still_builds_own_context():
    """顶层 run() 仍自行构建 ctx（不受 D002 拆分影响）。"""
    llm = FakeLLMProvider()
    llm.enqueue_empty()
    runtime = build_runtime(llm, [], max_iterations=2)

    probe = _SeenCtxProbe()
    runtime.register_middleware(probe)

    session = runtime.run("top-level", "hello")

    assert session.session_id == "top-level"
    assert probe.seen, "顶层 run 也应执行 on_enter_loop"
    top_ctx = probe.seen[-1]
    # 顶层 ctx 的 session 是自身 session，不是子 session
    assert top_ctx.shared["session"] is session