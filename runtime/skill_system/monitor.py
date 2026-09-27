"""
[L5] skill_system/monitor — 指标监控（MetricMonitor）。

监控技能的 ``effective_rate = applied / selections``。

- 读取技能计数器（通过注入的 ``SkillStore`` 或直接传入计数器集合）。
- 当 effective_rate 跌破 ``threshold`` 时触发进化流程（调用注入的
  ``on_degrade`` 回调）。

监控与进化解耦：本模块只负责"检测到退化"这一事实，具体如何进化由
mutator 完成。``on_degrade`` 回调由装配方注入（通常桥接到 mutator）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from runtime.skill_system.models import Skill, SkillCounters


@dataclass
class DegradationReport:
    """退化报告。

    Attributes:
        skill_id:       技能标识。
        name:           技能名称。
        effective_rate: 当前 effective_rate。
        threshold:      触发阈值。
        triggers:       是否触发进化。
    """

    skill_id: str
    name: str
    effective_rate: float
    threshold: float
    triggers: bool = False


class MetricMonitor:
    """技能 effective_rate 监控器。

    Attributes:
        threshold: effective_rate 退化阈值。
        min_selections: 至少被选择多少次才纳入监控（避免冷启动误报）。
        on_degrade: 可选回调，签名 ``on_degrade(skill, rate)``。
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_selections: int = 1,
        on_degrade: Optional[Callable[[Skill, float], None]] = None,
    ) -> None:
        """
        Args:
            threshold:      effective_rate 阈值，低于该值触发退化。
            min_selections: 最少 selections 数，低于此数不判定退化。
            on_degrade:     退化回调（接收 Skill 与 rate）。
        """
        self.threshold = float(threshold)
        self.min_selections = max(1, int(min_selections or 1))
        self._on_degrade = on_degrade
        # 已触发进化的技能（本轮监控内去重）
        self._triggered: Dict[str, bool] = {}

        # 对外可观测产物
        self.reports: List[DegradationReport] = []

    def check(self, skill: Skill) -> DegradationReport:
        """检查单个技能的 effective_rate，输出报告并可选触发回调。

        Args:
            skill: 待检查技能（含 counters）。

        Returns:
            DegradationReport。
        """
        selections = skill.counters.selections
        rate = skill.counters.effective_rate()

        triggers = (
            selections >= self.min_selections
            and rate < self.threshold
        )
        report = DegradationReport(
            skill_id=skill.skill_id,
            name=skill.name,
            effective_rate=rate,
            threshold=self.threshold,
            triggers=triggers,
        )
        self.reports.append(report)

        if triggers and self._on_degrade is not None and not self._triggered.get(skill.skill_id):
            self._triggered[skill.skill_id] = True
            try:
                self._on_degrade(skill, rate)
            except Exception:
                pass

        return report

    def check_all(self, skills: List[Skill]) -> List[DegradationReport]:
        """批量检查，返回报告列表。"""
        return [self.check(s) for s in skills]

    def reset(self) -> None:
        """清空已触发的去重记录与报告列表。"""
        self._triggered.clear()
        self.reports.clear()


__all__ = ["MetricMonitor", "DegradationReport"]