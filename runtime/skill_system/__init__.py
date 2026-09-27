"""
[L5] skill_system — 技能自进化子系统。

这是一个**独立子系统**（非中间件），依赖批次 1-4 的基础设施：
EventLog / Session / Context / ToolRegistry / ContextManager，但通过
``ctx.shared`` 与 Runtime 解耦。

组成：
    - models     : Skill / SkillVersion / SkillCounters 数据类。
    - store      : SkillStore（SQLite 三表 + 版本 DAG）。
    - selector   : SkillSelector（quality filter + LLM hybrid）。
    - evaluator  : SkillJudgmentAnalyzer / TaskQualityJudge /
                   ResponseContractChecker / RuntimeTracker。
    - monitor    : MetricMonitor（effective_rate 阈值触发进化）。
    - mutator    : IVEFocuser / LLMMutator / ScoreDeltaGate / GitRatchet。

``SkillSystem`` 是把上述组件装配在一起的子系统门面，提供中间件与测试
使用的高层操作：list / select / observe / evolve。

LLM 依赖全部通过 ``llm_callable`` 注入，未注入时所有 LLM 判定保守降级，
绝不抛异常。
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from runtime.skill_system.models import (
    Skill,
    SkillCounters,
    SkillVersion,
)
from runtime.skill_system.store import SkillStore
from runtime.skill_system.selector import SkillSelector
from runtime.skill_system.evaluator import (
    ResponseContractChecker,
    RuntimeTracker,
    SkillJudgmentAnalyzer,
    TaskQualityJudge,
)
from runtime.skill_system.monitor import MetricMonitor
from runtime.skill_system.mutator import (
    Diagnosis,
    GitRatchet,
    IVEFocuser,
    LLMMutator,
    MutationResult,
    ScoreDeltaGate,
)


class SkillSystem:
    """技能自进化子系统门面。

    装配 store / selector / evaluator / monitor / mutator，向外提供：

    - ``list_skills()``：列出可用技能。
    - ``select(task)``：选择应注入的技能（并累加 selections 计数）。
    - ``observe_applied(skill_id, applied)``：记录技能是否被应用。
    - ``judge(task, response, skills)``：逐技能判定是否被应用。
    - ``evaluate_quality(task, response)``：四维质量打分。
    - ``check_contract(response, skill_text)``：契约合规检查。
    - ``evolve(skill_id, task, response)``：触发自进化闭环（变异 + 门控 + 棘轮）。

    所有 LLM 调用通过 ``llm_callable`` 注入，None 时保守降级。
    """

    def __init__(
        self,
        store: Optional[SkillStore] = None,
        selector: Optional[SkillSelector] = None,
        judgment_analyzer: Optional[SkillJudgmentAnalyzer] = None,
        quality_judge: Optional[TaskQualityJudge] = None,
        contract_checker: Optional[ResponseContractChecker] = None,
        tracker: Optional[RuntimeTracker] = None,
        monitor: Optional[MetricMonitor] = None,
        focuser: Optional[IVEFocuser] = None,
        mutator: Optional[LLMMutator] = None,
        gate: Optional[ScoreDeltaGate] = None,
        ratchet: Optional[GitRatchet] = None,
        llm_callable: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    ) -> None:
        """
        各组件可显式注入；未注入的组件用 ``llm_callable``（若提供）构造默认实现，
        否则构造不依赖 LLM 的空实现。

        Args:
            store:             技能存储。
            selector:          技能选择器。
            judgment_analyzer: 技能应用判定器。
            quality_judge:     任务质量打分器。
            contract_checker:  契约检查器。
            tracker:           applied-rate 趋势跟踪器。
            monitor:           指标监控器（退化触发进化）。
            focuser:           五问诊断器。
            mutator:           技能文本重写器。
            gate:              分数门控。
            ratchet:           棘轮回滚器。
            llm_callable:      可选 LLM 调用函数（未注入各组件时用于构造默认实现）。
        """
        self.llm_callable = llm_callable

        # [D003] 不能用 `store or SkillStore()`：SkillStore 定义了 __len__，
        # 空库（COUNT(*) == 0）时为 falsy，会把调用方显式注入的空 store
        # **替换**为内部新 store，导致后续对原 store 的写入全部丢失。
        # 必须用 `is not None` 判定"是否提供"。
        self.store = store if store is not None else SkillStore()
        self.selector = selector or SkillSelector(llm_callable=llm_callable)
        self.judgment_analyzer = judgment_analyzer or SkillJudgmentAnalyzer(llm_callable)
        self.quality_judge = quality_judge or TaskQualityJudge(llm_callable)
        self.contract_checker = contract_checker or ResponseContractChecker()
        self.tracker = tracker or RuntimeTracker()
        self.ratchet = ratchet or GitRatchet(store=self.store)

        # monitor 的退化回调默认桥接到 evolve
        self.monitor = monitor or MetricMonitor(on_degrade=self._on_degrade)
        # 若 monitor 是通过 on_degrade 构造的默认实例但调用方期望覆盖，保留传入
        if isinstance(monitor, MetricMonitor):
            self.monitor = monitor

        self.focuser = focuser or IVEFocuser(llm_callable)
        self.mutator = mutator or LLMMutator(llm_callable)
        # gate 需要 score_fn，默认用质量打分（基于 task 与文本）
        if gate is None:
            gate = ScoreDeltaGate(score_fn=self._default_score_fn)
        self.gate = gate

        # 对外可观测：最近一次进化结果
        self.last_mutation: Optional[MutationResult] = None
        self.evolutions: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 高层 API
    # ------------------------------------------------------------------

    def list_skills(self) -> List[Skill]:
        """列出全部技能。"""
        return self.store.list()

    def select(self, task: str, top_k: Optional[int] = None) -> List[Skill]:
        """选择应注入的技能，并对选中项累加 selections 计数。

        Returns:
            选中的技能列表（含更新后的计数器）。
        """
        skills = self.store.list()
        selected = self.selector.select(task, skills, top_k=top_k)
        for skill in selected:
            self.store.bump_counter(skill.skill_id, "selections", 1)
        # 返回更新后的技能视图
        return [self.store.get(s.skill_id) or s for s in selected]

    def judge(self, task: str, response: Any, skills: List[Skill]) -> Dict[str, bool]:
        """逐技能判定是否被应用，返回 {skill_id: applied_bool}。"""
        result: Dict[str, bool] = {}
        for skill in skills:
            result[skill.skill_id] = self.judgment_analyzer.judge(
                task, response, skill
            )
        return result

    def observe_applied(self, skill_id: str, applied: bool) -> None:
        """记录技能是否被应用（applied=True 时累加 applied 计数）。"""
        if applied:
            self.store.bump_counter(skill_id, "applied", 1)

    def evaluate_quality(self, task: str, response: Any, reference: Optional[str] = None):
        """四维质量打分，返回 QualityScores。"""
        return self.quality_judge.score(task, response, reference=reference)

    def check_contract(self, response: Any, skill_text: str) -> Dict[str, Any]:
        """契约合规检查。"""
        return self.contract_checker.check(response, skill_text)

    def observe_trend(self, applied: int, selections: int):
        """向 RuntimeTracker 记录一个样本。"""
        self.tracker.observe(applied, selections)

    def track_signal(self, baseline_rate: Optional[float] = None):
        """返回 applied-rate 趋势信号。"""
        return self.tracker.signal(baseline_rate)

    def check_metrics(self, skills: Optional[List[Skill]] = None):
        """检查技能 effective_rate 是否跌破阈值（触发进化）。"""
        # [D003 同类] `skills or ...` 会把"显式传入的空列表"误判为"未传"，
        # 退化成全量检查。改用 `is not None` 区分"未传"与"传空"。
        target = skills if skills is not None else self.store.list()
        return self.monitor.check_all(target)

    # ------------------------------------------------------------------
    # 自进化闭环
    # ------------------------------------------------------------------

    def evolve(
        self,
        skill_id: str,
        task: str,
        response: Any = None,
        score_fn: Optional[Callable[[Skill], float]] = None,
    ) -> Optional[MutationResult]:
        """触发一次自进化闭环：诊断 → 变异 → 门控 → 写回 / 棘轮回滚。

        Args:
            skill_id: 目标技能。
            task:     触发进化的任务描述。
            response: 代表性响应（用于诊断）。
            score_fn: 可选覆盖门控评分函数。

        Returns:
            MutationResult（被接受时），未通过时可能返回 MutationResult
            或 None（无候选文本时）。
        """
        skill = self.store.get(skill_id)
        if skill is None:
            return None

        # 1. 诊断
        diagnosis = self.focuser.diagnose(skill, task, response)

        # 2. 变异
        new_text = self.mutator.mutate(skill, diagnosis, task)
        if not new_text:
            return None

        # 3. 门控（可用外部 score_fn 覆盖）
        gate = self.gate
        if score_fn is not None:
            gate = ScoreDeltaGate(score_fn=score_fn)

        result = gate.evaluate(skill, new_text)
        self.last_mutation = result

        if result.accepted:
            # 写回 store，生成新版本
            updated = self.store.update(skill.skill_id, text=result.new_text, score=result.new_score)
            # 记录 last-known-good（分数提升）
            if updated is not None:
                self.ratchet.record(updated)
        else:
            # 未通过门控：分数下降 → 棘轮回滚检查
            self.ratchet.record(skill)
            degraded = self.ratchet.is_degraded(
                skill, result.new_score if result.new_score else skill.score
            )
            if degraded:
                self.ratchet.rollback(
                    skill,
                    result.new_score,
                    reason=result.reason,
                )

        self.evolutions.append({
            "skill_id": skill.skill_id,
            "accepted": result.accepted,
            "old_score": result.old_score,
            "new_score": result.new_score,
            "reason": result.reason,
        })
        return result

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _on_degrade(self, skill: Skill, rate: float) -> None:
        """MetricMonitor 退化回调：触发进化。"""
        try:
            self.evolve(skill.skill_id, task="", response=None)
        except Exception:
            pass

    def _default_score_fn(self, skill: Skill) -> float:
        """默认评分函数：用质量打分器评估技能文本自身。

        由于没有任务上下文，这里做一个启发式：技能文本越完整（长度适中、
        包含指令性关键词）分数越高，上限 1.0。这是纯文本启发，保底可用于
        测试；生产环境应注入真实 score_fn。
        """
        text = skill.text or ""
        parts = [p for p in re.split(r"[\n。；;]", text) if p.strip()]
        score = 0.2 + 0.1 * min(len(parts), 6)
        if any(k in text for k in ("必须", "禁止", "步骤", "输出", "输入")):
            score += 0.2
        return min(1.0, score)

    def close(self) -> None:
        """关闭底层存储。"""
        closer = getattr(self.store, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass


__all__ = [
    "Skill",
    "SkillCounters",
    "SkillVersion",
    "SkillStore",
    "SkillSelector",
    "SkillJudgmentAnalyzer",
    "TaskQualityJudge",
    "ResponseContractChecker",
    "RuntimeTracker",
    "MetricMonitor",
    "IVEFocuser",
    "Diagnosis",
    "LLMMutator",
    "ScoreDeltaGate",
    "MutationResult",
    "GitRatchet",
    "SkillSystem",
]