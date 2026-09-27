"""
[L5] skill_system/models — 技能数据模型。

定义技能系统的核心数据结构：

- ``Skill``：技能数据类，聚合 skill_id / name / text / version / score /
  counters。
- ``SkillVersion``：技能版本记录（版本 DAG 的一个节点）。
- ``SkillCounters``：技能运行期计数器（selections / applied / completions /
  fallbacks）。

本模块只包含纯数据类，不含任何存储、LLM 调用或流程控制逻辑，可被
store / selector / evaluator / monitor / mutator 安全复用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

#: 计数器字段名（用于类型文档与默认值对齐）
COUNTER_FIELDS = ("selections", "applied", "completions", "fallbacks")


@dataclass
class SkillCounters:
    """技能运行期计数器。

    Attributes:
        selections:  被选择次数（selector / injection 选中）。
        applied:     被判定为"实际应用"的次数（evaluator 判定）。
        completions: 完整完成任务的次数。
        fallbacks:   应用后发生回退 / 降级的次数。
    """

    selections: int = 0
    applied: int = 0
    completions: int = 0
    fallbacks: int = 0

    def to_dict(self) -> Dict[str, int]:
        """序列化为普通字典。"""
        return {
            "selections": self.selections,
            "applied": self.applied,
            "completions": self.completions,
            "fallbacks": self.fallbacks,
        }

    def bump(self, name: str, delta: int = 1) -> None:
        """对指定计数器累加 ``delta``。

        Args:
            name:  计数器名（selections / applied / completions / fallbacks）。
            delta: 累加增量，可为负。

        Raises:
            ValueError: 计数器名非法时。
        """
        if name not in COUNTER_FIELDS:
            raise ValueError(f"未知计数器字段: {name!r}")
        current = getattr(self, name, 0) or 0
        setattr(self, name, max(0, int(current) + int(delta)))

    def effective_rate(self) -> float:
        """计算有效应用率 = applied / selections。

        未发生选择时返回 0.0（避免除零）。
        """
        if self.selections <= 0:
            return 0.0
        return self.applied / self.selections


@dataclass
class SkillVersion:
    """技能版本 DAG 的一个节点。

    版本 DAG 通过 ``parent_version`` 指向前一版本，形成一条（或多条）
    演化链。``version=None`` 时表示根版本（初始创建）。

    Attributes:
        skill_id:       技能标识。
        version:        版本号（从 1 递增）。
        text:           该版本的技能文本。
        created_at:     创建时间（Unix 秒）。
        score:          该版本的评分（越高越好）。
        parent_version: 父版本号；根版本为 None。
    """

    skill_id: str
    version: int
    text: str
    created_at: float
    score: float = 0.0
    parent_version: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "text": self.text,
            "created_at": self.created_at,
            "score": self.score,
            "parent_version": self.parent_version,
        }


@dataclass
class Skill:
    """技能数据类（select / evaluate / evolve 的公共载体）。

    Attributes:
        skill_id:  技能唯一标识。
        name:      技能名称。
        text:      技能文本（提示词片段 / 指令）。
        version:   当前版本号。
        score:     当前评分（质量过滤器与 ScoreDeltaGate 的依据）。
        counters:  运行期计数器（默认全 0）。
    """

    skill_id: str
    name: str
    text: str
    version: int = 1
    score: float = 0.0
    counters: SkillCounters = field(default_factory=SkillCounters)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为普通字典（含 counters 展开）。"""
        data: Dict[str, Any] = {
            "skill_id": self.skill_id,
            "name": self.name,
            "text": self.text,
            "version": self.version,
            "score": self.score,
        }
        data["counters"] = self.counters.to_dict()
        return data

    def effective_rate(self) -> float:
        """有效应用率（applied / selections）。"""
        return self.counters.effective_rate()

    def __repr__(self) -> str:
        return (
            f"<Skill id={self.skill_id!r} name={self.name!r} "
            f"version={self.version} score={self.score:.4f}>"
        )


__all__ = [
    "Skill",
    "SkillVersion",
    "SkillCounters",
    "COUNTER_FIELDS",
]