"""
[A5] 技能系统。

覆盖：
    1. 创建技能 → 选择 → 注入 → 观察 → 评估 → 计数更新的完整闭环。
    2. effective_rate 跌破阈值 → 触发进化。
    3. ScoreDeltaGate 拒绝分数下降的变异。
    4. GitRatchet 回滚退化。
    5. 不启用 skill_system 时零开销（不产生任何额外 LLM 调用）。
"""

from __future__ import annotations

from agent_loop import Context, Middleware
from runtime.builtin_middlewares.skill_injection import SkillInjectionMiddleware
from runtime.builtin_middlewares.skill_observation import SkillObservationMiddleware
from runtime.skill_system import SkillSystem
from runtime.skill_system.models import Skill
from runtime.skill_system.monitor import MetricMonitor
from runtime.skill_system.mutator import GitRatchet, ScoreDeltaGate
from runtime.skill_system.store import SkillStore
from tests.fakes.fake_llm import FakeLLMProvider
from tests.fakes.fake_tool import FakeTool
from tests.integration.conftest import ContextCaptureMiddleware, build_runtime


# ============================================================================
# 场景 1：创建 → 选择 → 计数更新闭环
# ============================================================================


def test_skill_full_cycle_counter_updates():
    store = SkillStore()
    # 先创建技能再构造门面（避免空 store 被 `store or SkillStore()` 丢弃）
    store.create("skill_a", "代码审查", "必须检查输入；必须输出结论", score=0.8)
    system = SkillSystem(store=store)

    skills = system.list_skills()
    assert len(skills) == 1

    # 选择：无 LLM → 纯质量回退，选中 skill_a，bump selections
    selected = system.select("做一些代码审查")
    assert len(selected) == 1
    assert selected[0].skill_id == "skill_a"

    counters = store.get_counters("skill_a")
    assert counters.selections == 1

    # 观察应用：bump applied
    system.observe_applied("skill_a", True)
    counters = store.get_counters("skill_a")
    assert counters.applied == 1
    assert counters.effective_rate() == 1.0


# ============================================================================
# 场景 2：effective_rate 跌破阈值 → 触发进化
# ============================================================================


def test_effective_rate_below_threshold_triggers_degrade():
    store = SkillStore()
    # 先创建技能（避免空 store 被 `store or SkillStore()` 丢弃）
    store.create("skill_b", "B", "text", score=0.8)

    # 用可编程 monitor 捕获 on_degrade 触发
    degraded = []
    monitor = MetricMonitor(
        threshold=0.5, min_selections=2,
        on_degrade=lambda skill, rate: degraded.append(skill.skill_id),
    )
    system = SkillSystem(store=store, monitor=monitor)

    # 2 次选择，0 次 applied → rate = 0.0 < 0.5，触发退化
    system.select("t")  # selections=1
    system.select("t")  # selections=2

    reports = system.check_metrics()
    assert len(reports) == 1
    assert reports[0].triggers is True
    # on_degrade 回调被触发（桥接到 evolve，若无 LLM 则静默）
    assert "skill_b" in degraded


# ============================================================================
# 场景 3：ScoreDeltaGate 拒绝分数下降的变异
# ============================================================================


def test_score_delta_gate_rejects_lower_score():
    # score_fn 返回固定低分，使变异分数低于当前 score
    gate = ScoreDeltaGate(score_fn=lambda candidate: 0.2, min_delta=0.0)

    skill = Skill(skill_id="s", name="n", text="old", score=0.8)
    result = gate.evaluate(skill, "new text")

    assert result.accepted is False
    assert "未提升" in result.reason


# ============================================================================
# 场景 4：GitRatchet 回滚退化
# ============================================================================


def test_git_ratchet_rollback_on_degrade():
    store = SkillStore()
    store.create("skill_c", "C", "good-text", score=0.9)
    ratchet = GitRatchet(store=store)

    skill = store.get("skill_c")
    ratchet.record(skill)  # last-known-good = score 0.9

    # 模拟退化：新分数 0.5 < 0.9
    assert ratchet.is_degraded(skill, 0.5) is True

    rolled = ratchet.rollback(skill, 0.5, reason="score degraded")
    assert rolled is not None
    # 回滚后 store 中技能恢复为 last-known-good text
    assert store.get("skill_c").text == "good-text"
    assert len(ratchet.rollbacks) == 1


# ============================================================================
# 场景 5：不启用 skill_system 时零开销（不产生额外 LLM 调用）
# ============================================================================


def test_no_skill_system_zero_overhead():
    llm = FakeLLMProvider()
    llm.enqueue_text("ok")

    tool = FakeTool()
    injection = SkillInjectionMiddleware()
    observation = SkillObservationMiddleware()
    capture = ContextCaptureMiddleware()

    # 不注入 skill_system（build_runtime 未传 skill_system）
    runtime = build_runtime(
        llm, [tool.make_tool("echo")],
        middlewares=[injection, observation, capture],
        max_iterations=1,
    )
    runtime.run("sk", "do something")

    # LLM 只被调用 1 次（Agent 主循环），skill 中间件零额外调用
    assert len(llm.calls) == 1
    # 无选中技能、无观察结果
    assert "skill_selected" not in capture.ctx.shared
    assert "skill_observed" not in capture.ctx.shared