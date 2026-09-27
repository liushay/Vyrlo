"""
[A2 补充] CostGuard 与 Observability 的注册顺序耦合。

CostGuardMiddleware.after_llm 优先从 event_log 的 LLMCallEvent 累加预算，
而 LLMCallEvent 由 ObservabilityMiddleware.after_llm 写入。若 CostGuard
注册在 Observability 之前，其 after_llm 先执行，此时事件尚未写入，便退化为
从 response 直接提取 usage；若 response 不携带 usage，预算将永远漏计。
"""

from __future__ import annotations

from runtime.builtin_middlewares.cost_guard import CostGuardMiddleware
from runtime.builtin_middlewares.observability import ObservabilityMiddleware
from tests.fakes.fake_llm import FakeLLMProvider
from tests.integration.conftest import ContextCaptureMiddleware, build_runtime


def _no_usage_response():
    # 模拟"仅事件里带用量、response 不带 usage"的 provider
    return {
        "content": "ok",
        "tool_calls": [],
        # 注意：无 usage 字段
        "model": "m",
        "provider": "p",
    }


def test_cost_guard_after_observability_counts_via_event():
    # observability 先注册：事件先写入，cost_guard 从事件累加
    llm = FakeLLMProvider()
    llm.enqueue({"content": "ok", "tool_calls": [], "usage": {"total_tokens": 10},
                 "model": "m", "provider": "p"})

    obs = ObservabilityMiddleware(echo=False)
    cg = CostGuardMiddleware(max_tokens=1000)
    capture = ContextCaptureMiddleware()
    runtime = build_runtime(llm, [], middlewares=[obs, cg, capture], max_iterations=1)
    session = runtime.run("cg-order-obs-first", "task")

    # observability 从 usage 提取 10 写入事件，cost_guard 从事件累加 → 10
    assert session.budget.used_tokens == 10


def test_cost_guard_before_observability_without_usage_loses_budget():
    # cost_guard 先注册：after_llm 先执行，事件尚未写入；response 无 usage → 漏洞
    llm = FakeLLMProvider()
    llm.enqueue(_no_usage_response())
    # 用带 usage 的事件注入模拟 observability 实际会写，但观察 cost_guard 的累积
    obs = ObservabilityMiddleware(echo=False)
    cg = CostGuardMiddleware(max_tokens=1000)
    capture = ContextCaptureMiddleware()
    runtime = build_runtime(llm, [], middlewares=[cg, obs, capture], max_iterations=1)
    session = runtime.run("cg-order-cg-first", "task")

    # 缺陷：response 无 usage，且 cost_guard 先于 observability 执行，
    # 既读不到事件（尚未写入），也无 usage 可兜底 → 预算漏计为 0。
    assert session.budget.used_tokens == 0