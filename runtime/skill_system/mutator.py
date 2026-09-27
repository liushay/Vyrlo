"""
[L5] skill_system/mutator — 技能自进化（变异 + 棘轮回滚）。

包含四个组件：

1. ``IVEFocuser``：五问诊断，定位技能偏差来源。
2. ``LLMMutator``：基于诊断结论重写技能文本。
3. ``ScoreDeltaGate``：变异后分数更高才通过。
4. ``GitRatchet``：棘轮机制，记录 last-known-good，退化时自动回滚。

进化结果写回 ``SkillStore``，生成新版本（版本 DAG 追加节点）。

流程：``ScoreDeltaGate.mutate`` 编排 IVEFocuser → LLMMutator → 评估打分
→ 与当前分数比较 → 通过则写回；``GitRatchet`` 在判定退化时把技能回滚到
last-known-good 版本。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from runtime.skill_system.models import Skill, SkillVersion
from runtime.skill_system.store import SkillStore

#: LLM 调用签名：接收消息列表，返回字符串
LLMCallable = Callable[[List[Dict[str, str]]], str]
#: 评分函数签名：接收 Skill（变异候选），返回 0~1 的分数
ScoreFn = Callable[["Skill"], float]


def _call_llm(llm_callable: Optional[LLMCallable], prompt: str) -> Optional[str]:
    """安全调用 LLM，返回文本；未注入或异常时返回 None。"""
    if llm_callable is None:
        return None
    try:
        result = llm_callable([{"role": "user", "content": prompt}])
    except Exception:
        return None
    if isinstance(result, dict):
        return str(result.get("content", ""))
    return str(result or "")


# ============================================================================
# 1. IVEFocuser —— 五问诊断
# ============================================================================


@dataclass
class Diagnosis:
    """五问诊断结论。

    Attributes:
        input_bias:       输入处理偏差。
        validation_bias:  验证 / 校验偏差。
        environment_bias: 环境 / 上下文偏差。
        process_bias:     过程 / 步骤偏差。
        output_bias:      输出 / 格式偏差。
        summary:          诊断摘要（供 LLMMutator 使用）。
    """

    input_bias: str = ""
    validation_bias: str = ""
    environment_bias: str = ""
    process_bias: str = ""
    output_bias: str = ""
    summary: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "input_bias": self.input_bias,
            "validation_bias": self.validation_bias,
            "environment_bias": self.environment_bias,
            "process_bias": self.process_bias,
            "output_bias": self.output_bias,
            "summary": self.summary,
        }


class IVEFocuser:
    """五问诊断：从五个维度追问技能偏差。

    五个维度：输入 / 验证 / 环境 / 过程 / 输出。
    未注入 LLM 时返回空诊断（summary 为空），由上游决定跳过变异。
    """

    QUESTIONS = (
        ("input_bias", "输入", "技能是否对输入的理解或预处理存在偏差？"),
        ("validation_bias", "验证", "技能是否缺少对中间结果的校验或自检？"),
        ("environment_bias", "环境", "技能是否未适配运行环境或上下文约束？"),
        ("process_bias", "过程", "技能的执行步骤或顺序是否存在偏差？"),
        ("output_bias", "输出", "技能的输出格式或结论是否不够明确、不稳定？"),
    )

    def __init__(self, llm_callable: Optional[LLMCallable] = None) -> None:
        self._llm_callable = llm_callable

    def diagnose(self, skill: Skill, task: str, response: Any = None) -> Diagnosis:
        """诊断技能偏差，返回五问结论。

        Args:
            skill:    被诊断的技能。
            task:     代表性任务描述。
            response: 可选的代表性响应（用于判断何处偏差）。

        Returns:
            Diagnosis。
        """
        if self._llm_callable is None:
            return Diagnosis()

        prompt = (
            "你是技能诊断器。从五个维度诊断技能偏差，分别是：\n"
            "1) 输入 2) 验证 3) 环境 4) 过程 5) 输出。\n"
            "只输出 JSON 对象，形如："
            '{"input_bias":"...","validation_bias":"...",'
            '"environment_bias":"...","process_bias":"...","output_bias":"..."}\n'
            f"技能名称：{skill.name}\n"
            f"技能文本：{skill.text}\n"
            f"代表任务：{task}\n"
            + (f"代表响应：{response if isinstance(response, str) else ''}\n" if response else "")
            + "输出 JSON："
        )
        raw = _call_llm(self._llm_callable, prompt)
        parsed = self._parse_diagnosis(raw)

        summary = "；".join(f"{label}：{parsed[key]}" for _, key, label in [
            ("input_bias", "input_bias", "输入"),
            ("validation_bias", "validation_bias", "验证"),
            ("environment_bias", "environment_bias", "环境"),
            ("process_bias", "process_bias", "过程"),
            ("output_bias", "output_bias", "输出"),
        ] if parsed.get(key))
        return Diagnosis(
            input_bias=parsed.get("input_bias", ""),
            validation_bias=parsed.get("validation_bias", ""),
            environment_bias=parsed.get("environment_bias", ""),
            process_bias=parsed.get("process_bias", ""),
            output_bias=parsed.get("output_bias", ""),
            summary=summary,
        )

    @staticmethod
    def _parse_diagnosis(raw: Optional[str]) -> Dict[str, str]:
        """解析五问 JSON，失败返回空。"""
        if not raw:
            return {}
        text = raw.strip()
        import json
        import re
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                result = {}
                for key, _, _ in IVEFocuser.QUESTIONS:
                    val = data.get(key)
                    if isinstance(val, str):
                        result[key] = val
                return result
        except (json.JSONDecodeError, TypeError):
            pass
        # 兜底：正则抓 "key": "value"
        result = {}
        for key, _, _ in IVEFocuser.QUESTIONS:
            m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', text)
            if m:
                result[key] = m.group(1)
        return result


# ============================================================================
# 2. LLMMutator —— 重写技能文本
# ============================================================================


class LLMMutator:
    """基于诊断结论重写技能文本。"""

    def __init__(self, llm_callable: Optional[LLMCallable] = None) -> None:
        self._llm_callable = llm_callable

    def mutate(
        self,
        skill: Skill,
        diagnosis: Diagnosis,
        task: str,
    ) -> Optional[str]:
        """重写技能文本，返回新文本；无法重写时返回 None。

        Args:
            skill:     原技能。
            diagnosis: 五问诊断结论。
            task:      触发进化的任务描述。

        Returns:
            重写后的技能文本。
        """
        if self._llm_callable is None:
            return None

        prompt = (
            "你是技能自进化器。根据诊断结论重写技能文本，使其在下一次使用中"
            "偏差更小、更可靠。只输出重写后的技能文本，不要任何解释。\n"
            f"技能名称：{skill.name}\n"
            f"当前技能文本：{skill.text}\n"
            f"触发进化任务：{task}\n"
            f"诊断结论：{diagnosis.summary or '（无明确诊断，做保守改进）'}\n"
            "新技能文本："
        )
        raw = _call_llm(self._llm_callable, prompt)
        if not raw or not raw.strip():
            return None
        return raw.strip()


# ============================================================================
# 3. ScoreDeltaGate —— 分数门控
# ============================================================================


@dataclass
class MutationResult:
    """变异结果。

    Attributes:
        accepted:    是否通过门控。
        new_text:    新技能文本（通过时）。
        old_score:   当前分数。
        new_score:   变异后分数。
        skill_id:    技能标识。
        reason:      未通过的原因（通过时为空）。
    """

    accepted: bool
    skill_id: str
    old_score: float
    new_score: float = 0.0
    new_text: str = ""
    reason: str = ""


class ScoreDeltaGate:
    """变异门控：变异后分数严格更高才通过。

    ``min_delta`` 用于要求最小提升幅度（避免无意义微涨）。
    """

    def __init__(self, score_fn: ScoreFn, min_delta: float = 0.0) -> None:
        """
        Args:
            score_fn:  评分函数，接收候选 Skill，返回 0~1 分数。
            min_delta: 最小提升幅度，默认 0（严格更高即可）。
        """
        self._score_fn = score_fn
        self.min_delta = float(min_delta)

    def evaluate(self, skill: Skill, new_text: str) -> MutationResult:
        """评估变异是否通过。

        Args:
            skill:    原技能。
            new_text: 候选新文本。

        Returns:
            MutationResult。
        """
        if not new_text or new_text == skill.text:
            return MutationResult(
                accepted=False,
                skill_id=skill.skill_id,
                old_score=skill.score,
                reason="新文本为空或未变化",
            )

        candidate = Skill(
            skill_id=skill.skill_id,
            name=skill.name,
            text=new_text,
            version=skill.version + 1,
            score=skill.score,
            counters=skill.counters,
        )
        new_score = float(self._score_fn(candidate))
        old_score = float(skill.score)

        if new_score <= old_score + self.min_delta:
            reason = (
                f"变异分数未提升: {new_score:.4f} <= "
                f"{old_score + self.min_delta:.4f}"
            )
            return MutationResult(
                accepted=False,
                skill_id=skill.skill_id,
                old_score=old_score,
                new_score=new_score,
                new_text=new_text,
                reason=reason,
            )

        return MutationResult(
            accepted=True,
            skill_id=skill.skill_id,
            old_score=old_score,
            new_score=new_score,
            new_text=new_text,
        )


# ============================================================================
# 4. GitRatchet —— 棘轮机制
# ============================================================================


@dataclass
class RatchetEntry:
    """棘轮记录的 last-known-good 状态。

    Attributes:
        skill_id: 技能标识。
        version:  last-known-good 版本号。
        text:     last-known-good 文本。
        score:    last-known-good 分数。
    """

    skill_id: str
    version: int
    text: str
    score: float


class GitRatchet:
    """棘轮机制：记录 last-known-good，退化时回滚。

    维护一个内存中的 last-known-good 注册表（也可通过 store 读历史版本）。
    ``rollback`` 把技能写回 last-known-good 版本，生成新版本记录。
    """

    def __init__(self, store: Optional[SkillStore] = None) -> None:
        """
        Args:
            store: 可选 SkillStore；rollback 写回新版本时使用。
        """
        self._store = store
        self._entries: Dict[str, RatchetEntry] = {}

        # 对外可观测的回滚记录
        self.rollbacks: List[Dict[str, Any]] = []

    def record(self, skill: Skill) -> None:
        """把当前技能标记为 last-known-good。

        只有在当前分数不低于已记录版本时才会更新（保持棘轮只进不退）。
        """
        existing = self._entries.get(skill.skill_id)
        if existing is None or skill.score >= existing.score:
            self._entries[skill.skill_id] = RatchetEntry(
                skill_id=skill.skill_id,
                version=skill.version,
                text=skill.text,
                score=skill.score,
            )

    def get_known_good(self, skill_id: str) -> Optional[RatchetEntry]:
        """读取 last-known-good 快照。"""
        return self._entries.get(skill_id)

    def is_degraded(self, skill: Skill, new_score: float) -> bool:
        """判断新分数是否相对 last-known-good 退化。"""
        entry = self._entries.get(skill.skill_id)
        if entry is None:
            return False
        return float(new_score) < float(entry.score)

    def rollback(
        self,
        skill: Skill,
        new_score: float,
        reason: str = "",
    ) -> Optional[Skill]:
        """检测到退化后，回滚到 last-known-good 并写回 store。

        回滚逻辑：
            - 若没有 last-known-good 记录，则不做任何操作。
            - 若新分数低于 last-known-good，且 store 可用，则把技能文本 / 分数
              恢复为 last-known-good，生成新版本。

        Returns:
            回滚后的 Skill（可能为 None，当无记录或无需回滚时）。
        """
        entry = self._entries.get(skill.skill_id)
        if entry is None:
            return None

        if float(new_score) >= float(entry.score):
            return None

        self.rollbacks.append({
            "skill_id": skill.skill_id,
            "from_score": new_score,
            "to_score": entry.score,
            "to_version": entry.version,
            "reason": reason or "score degraded",
        })

        if self._store is not None:
            return self._store.update(
                skill.skill_id,
                text=entry.text,
                score=entry.score,
            )
        return Skill(
            skill_id=skill.skill_id,
            name=skill.name,
            text=entry.text,
            version=skill.version + 1,
            score=entry.score,
            counters=skill.counters,
        )


__all__ = [
    "IVEFocuser",
    "Diagnosis",
    "LLMMutator",
    "ScoreDeltaGate",
    "MutationResult",
    "GitRatchet",
    "RatchetEntry",
]