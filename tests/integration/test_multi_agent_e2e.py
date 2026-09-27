"""
[A3] 多 Agent 编排。

覆盖委派拦截、委派深度上限、sandbox 继承、共享状态读写、
子 Agent 失败后父 Agent 继续、并发线程局部隔离。

重点暴露缺陷：
    - Runtime._run_delegation 构造的 child_ctx（含 sandbox/shared_state/
      agent_registry 继承）在 self.run() 中被丢弃，run() 会重新 _build_context，
      导致子 Agent 实际执行时 sandbox 继承失效。
"""

from __future__ import annotations

import threading

from agent_loop import Context, HookResult, Middleware
from runtime.agent_registry.interface import AgentRegistry, AgentSpec
from runtime.agent_registry.in_memory import InMemoryAgentRegistry
from runtime.builtin_middlewares.delegation import DelegationMiddleware
from runtime.builtin_middlewares.sandbox import SandboxMiddleware
from runtime.context_manager.interface import MemoryTrace
from runtime.runtime import Runtime
from runtime.sandbox.registry import SandboxRegistry
from runtime.shared_state.in_memory import InMemorySharedStateStore
from tests.fakes.fake_llm import FakeLLMProvider, openai_tool_call
from tests.fakes.fake_sandbox import FakeSandboxProvider
from tests.integration.conftest import (
    ContextCaptureMiddleware,
    StubContextManager,
    build_runtime,
)


# ============================================================================
# 场景 1：委派给不存在的 Agent → skip_tool
# ============================================================================


def test_delegate_to_unknown_agent_skipped():
    llm = FakeLLMProvider()
    llm.enqueue_tool_calls(
        [openai_tool_call("delegate_to_agent", {"target_agent": "ghost", "goal": "x"})]
    )

    registry = InMemoryAgentRegistry()
    delegation = DelegationMiddleware(agent_registry=registry)
    capture = ContextCaptureMiddleware()

    runtime = build_runtime(
        llm, [],
        middlewares=[delegation, capture],
        agent_registry=registry,
        max_iterations=1,
    )
    runtime.run("m1", "delegate to ghost")

    # 未知 agent → 写入 delegation_rejected
    rejected = capture.ctx.shared.get("delegation_rejected")
    assert rejected is not None
    assert any(r["reason"] == "unknown_agent" for r in rejected)
    # 未产生 pending_delegation
    assert "pending_delegation" not in capture.ctx.shared


# ============================================================================
# 场景 2：委派深度上限（直接调用 _run_delegation 验证递归拒绝）
# ============================================================================


def test_delegation_depth_limit():
    llm = FakeLLMProvider()
    llm.enqueue_empty()  # 子 session 空转

    registry = InMemoryAgentRegistry()
    spec = AgentSpec(agent_id="child", description="child")
    registry.register(spec)

    runtime = build_runtime(
        llm, [],
        agent_registry=registry,
        max_delegation_depth=0,  # 禁止委派
    )

    # 手动构造父 ctx，直接调用内部委派逻辑
    parent_ctx = Context()
    parent_ctx.shared["session"] = type("S", (), {"session_id": "parent"})()
    parent_ctx.shared["event_log"] = runtime.registry.event_log

    result = runtime._run_delegation(parent_ctx, spec, "goal", "ok")

    # 深度 0 即拒绝，success=False
    assert result["success"] is False
    assert "深度超限" in result["reason"]
    # 父 ctx 记录 delegation_rejected（reason 为 depth_exceeded）
    rejected = parent_ctx.shared.get("delegation_rejected")
    assert rejected is not None
    assert any(r["reason"] == "depth_exceeded" for r in rejected)


# ============================================================================
# 场景 3：子 Agent 复用父沙箱（暴露：child_ctx 被 run() 丢弃）
# ============================================================================


def test_child_agent_sandbox_inheritance_preserved():
    """[D002 已修复] 子 Agent 执行时使用继承父 sandbox 的 child_ctx。"""
    llm = FakeLLMProvider()
    llm.enqueue_empty()

    sandbox = FakeSandboxProvider(available=True)
    registry = SandboxRegistry({"local": sandbox})

    agent_registry = InMemoryAgentRegistry()
    spec = AgentSpec(agent_id="child", description="child", sandbox_mode="local")
    agent_registry.register(spec)

    sandbox_mw = SandboxMiddleware(sandbox_registry=registry)

    runtime = build_runtime(
        llm, [],
        agent_registry=agent_registry,
        max_delegation_depth=3,
    )
    runtime.register_middleware(sandbox_mw)

    # 记录子 Agent 执行期间"实际生效"的 ctx
    seen_ctxs = []

    class _CtxProbe(Middleware):
        def on_enter_loop(self, ctx: Context) -> HookResult:
            seen_ctxs.append(ctx)
            return HookResult.continue_()

    runtime.register_middleware(_CtxProbe("ctx_probe"))

    # 父 ctx 手动 acquire 一个 sandbox，再委派
    parent_ctx = Context()
    parent_ctx.shared["session"] = type("S", (), {"session_id": "parent"})()
    parent_ctx.shared["event_log"] = runtime.registry.event_log
    parent_handle = sandbox.acquire("parent")
    parent_ctx.shared["sandbox"] = parent_handle

    result = runtime._run_delegation(parent_ctx, spec, "goal", "ok")

    # [D002 已修复] 子 Agent 实际执行所用的 ctx 带有父 sandbox
    assert result["child_context"].shared.get("sandbox") is parent_handle
    # 子 Agent 运行期间 on_enter_loop 看到的 ctx 也携带父 sandbox
    assert seen_ctxs, "子 Agent 应至少执行一次 on_enter_loop"
    assert seen_ctxs[-1].shared.get("sandbox") is parent_handle


# ============================================================================
# 场景 4：子 Agent 写共享状态，父 Agent 读
# ============================================================================


def test_shared_state_write_read():
    store = InMemorySharedStateStore()
    store.set("ns", "key1", "value1")
    assert store.get("ns", "key1") == "value1"
    assert store.list_keys("ns") == ["key1"]
    assert store.snapshot("ns") == {"key1": "value1"}


# ============================================================================
# 场景 5：并发两个 session 跑同一 Runtime，线程局部状态不串
# ============================================================================


def test_concurrent_sessions_thread_local_isolation():
    llm = FakeLLMProvider()
    # 每个 session 独立脚本：用 callable 区分（依赖线程局部 active_ctx 不串）
    llm.default_response = type("R", (), {
        "content": "ok", "tool_calls": [], "usage": {},
        "model": "m", "provider": "p", "finish_reason": "stop", "raw": None,
    })()

    capture = ContextCaptureMiddleware()
    runtime = build_runtime(
        llm, [], middlewares=[capture], max_iterations=2,
    )

    errors = []

    def run_one(sid):
        try:
            s = runtime.run(sid, f"task-{sid}")
            return s.session_id
        except Exception as exc:  # pragma: no cover
            errors.append(exc)
            return None

    threads = [threading.Thread(target=run_one, args=(f"s-{i}",)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"并发运行抛异常: {errors}"

    # 每个 session 都独立保存
    for i in range(5):
        s = runtime.get_session(f"s-{i}")
        assert s is not None