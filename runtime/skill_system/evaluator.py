"""
[L5] skill_system/evaluator — 技能评估。

包含四个评估组件：

1. ``SkillJudgmentAnalyzer``：逐技能逐任务判定"该技能是否被实际应用"。
2. ``TaskQualityJudge``：四维加权打分（准确度 0.50 / 完整性 0.35 /
   效率 0.05 / 深度 0.10）。
3. ``ResponseContractChecker``：从技能文本提取契约规则，检查响应合规。
4. ``RuntimeTracker``：跟踪 applied-rate 趋势，输出退化信号。

所有 LLM 判定均通过注入的 ``llm_callable`` 完成；未注入时返回保守默认值，
绝不抛异常。评估结果由调用方（SkillObservationMiddleware / 进化流程）
写回 ``skill_counters``。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from runtime.skill_system.models import Skill

#: 默认 LLM 调用签名：接收消息列表，返回字符串
LLMCallable = Callable[[List[Dict[str, str]]], str]

#: 任务质量四维权重
DEFAULT_WEIGHTS = {
    "accuracy": 0.50,
    "completeness": 0.35,
    "efficiency": 0.05,
    "depth": 0.10,
}


def _extract_text(response: Any) -> str:
    """从 LLM 响应中提取文本（兼容 str / dict / dataclass）。"""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        text = response.get("content", response.get("text", ""))
    else:
        text = getattr(response, "content", "") or getattr(response, "text", "")
    return str(text or "")


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
# 1. SkillJudgmentAnalyzer —— 技能应用判定
# ============================================================================


class SkillJudgmentAnalyzer:
    """逐技能逐任务判定技能是否被实际应用。

    LLM 判定未注入时，采用保守默认：**未应用**（False），
    避免凭空虚构 applied 信号污染计数器。
    """

    def __init__(self, llm_callable: Optional[LLMCallable] = None) -> None:
        self._llm_callable = llm_callable

    def judge(
        self,
        task: str,
        response: Any,
        skill: Skill,
        task_description: Optional[str] = None,
    ) -> bool:
        """判定 ``skill`` 是否被应用于解决 ``task`` 的 ``response``。

        Args:
            task:            任务描述。
            response:        LLM 响应（文本或 dict / dataclass）。
            skill:           被评估的技能。
            task_description: 可选，覆盖 task 的展示文本。

        Returns:
            是否被应用（bool）。
        """
        text = _extract_text(response)
        if not text:
            return False
        if self._llm_callable is None:
            return False

        desc = task_description or task
        prompt = (
            "你是技能应用判定器。判断指定技能是否被实际应用在给定回答中。\n"
            "只输出一个词：yes 或 no。\n"
            f"任务：{desc}\n"
            f"技能名称：{skill.name}\n"
            f"技能文本：{skill.text}\n"
            f"回答：{text[:2000]}\n"
            "输出："
        )
        raw = _call_llm(self._llm_callable, prompt)
        if raw is None:
            return False
        return "yes" in raw.strip().lower()


# ============================================================================
# 2. TaskQualityJudge —— 四维加权打分
# ============================================================================


@dataclass
class QualityScores:
    """四维打分结果。

    Attributes:
        accuracy:     准确度。
        completeness: 完整性。
        efficiency:   效率。
        depth:        深度。
        total:        加权总分（0~1）。
        weights:      本轮使用的权重。
    """

    accuracy: float = 0.0
    completeness: float = 0.0
    efficiency: float = 0.0
    depth: float = 0.0
    total: float = 0.0
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def to_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "completeness": self.completeness,
            "efficiency": self.efficiency,
            "depth": self.depth,
            "total": self.total,
        }


class TaskQualityJudge:
    """四维加权打分器。

    权重默认：准确度 0.50 / 完整性 0.35 / 效率 0.05 / 深度 0.10。
    LLM 未注入时返回全零分（保守）。
    """

    def __init__(
        self,
        llm_callable: Optional[LLMCallable] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self._llm_callable = llm_callable
        self._weights = dict(weights or DEFAULT_WEIGHTS)
        # 补齐缺失维度
        for key in DEFAULT_WEIGHTS:
            self._weights.setdefault(key, DEFAULT_WEIGHTS[key])

    def score(
        self,
        task: str,
        response: Any,
        reference: Optional[str] = None,
    ) -> QualityScores:
        """对响应做四维打分，返回 QualityScores。

        Args:
            task:      任务描述。
            response:  LLM 响应。
            reference: 可选参考答案 / 标准。

        Returns:
            QualityScores。
        """
        text = _extract_text(response)
        if not text or self._llm_callable is None:
            return QualityScores(weights=dict(self._weights))

        prompt = (
            "你是任务质量评估器。对回答在四个维度打分，每维 0.0~1.0。\n"
            "只输出 JSON 对象，形如："
            '{"accuracy": 0.9, "completeness": 0.8, "efficiency": 0.7, "depth": 0.6}\n'
            f"任务：{task}\n"
            + (f"参考答案：{reference}\n" if reference else "")
            + f"回答：{text[:2000]}\n"
            "输出 JSON："
        )
        raw = _call_llm(self._llm_callable, prompt)
        parsed = self._parse_scores(raw)
        total = (
            parsed.get("accuracy", 0.0) * self._weights.get("accuracy", 0.0)
            + parsed.get("completeness", 0.0) * self._weights.get("completeness", 0.0)
            + parsed.get("efficiency", 0.0) * self._weights.get("efficiency", 0.0)
            + parsed.get("depth", 0.0) * self._weights.get("depth", 0.0)
        )
        return QualityScores(
            accuracy=parsed.get("accuracy", 0.0),
            completeness=parsed.get("completeness", 0.0),
            efficiency=parsed.get("efficiency", 0.0),
            depth=parsed.get("depth", 0.0),
            total=min(1.0, max(0.0, total)),
            weights=dict(self._weights),
        )

    @staticmethod
    def _parse_scores(raw: Optional[str]) -> Dict[str, float]:
        """从 LLM 输出中解析四维分数，失败返回空。"""
        if not raw:
            return {}
        text = raw.strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return TaskQualityJudge._coerce(data)
        except (json.JSONDecodeError, TypeError):
            pass
        # 兜底：正则抓 key: value
        result: Dict[str, float] = {}
        for key in DEFAULT_WEIGHTS:
            m = re.search(rf'"{key}"\s*:\s*([\d.]+)', text)
            if not m:
                m = re.search(rf"\b{key}\b\s*[:=]\s*([\d.]+)", text, re.IGNORECASE)
            if m:
                try:
                    result[key] = min(1.0, max(0.0, float(m.group(1))))
                except ValueError:
                    pass
        return result

    @staticmethod
    def _coerce(data: Dict[str, Any]) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for key in DEFAULT_WEIGHTS:
            val = data.get(key)
            if val is None or isinstance(val, bool):
                continue
            try:
                result[key] = min(1.0, max(0.0, float(val)))
            except (TypeError, ValueError):
                continue
        return result


# ============================================================================
# 3. ResponseContractChecker —— 契约规则检查
# ============================================================================


class ResponseContractChecker:
    """从技能文本提取契约规则，并检查响应是否合规。

    契约规则格式约定（在技能文本中）：

    - ``必须包含："..."`` 或 ``必须包含：...`` → 响应必须包含该子串。
    - ``禁止包含："..."`` 或 ``禁止包含：...`` → 响应不得包含该子串。
    - ``长度 >= N`` / ``长度 <= N`` → 响应长度约束。

    未匹配到任何规则时返回空检查结果（合规视为 True）。
    """

    _MUST_PATTERN = re.compile(r"必须包含[:：]\s*[「『\"']?(.+?)[」』\"']?\s*(?:$|\n|；|;)")
    _FORBID_PATTERN = re.compile(r"禁止包含[:：]\s*[「『\"']?(.+?)[」』\"']?\s*(?:$|\n|；|;)")
    _MIN_LEN_PATTERN = re.compile(r"长度\s*>=\s*(\d+)")
    _MAX_LEN_PATTERN = re.compile(r"长度\s*<=\s*(\d+)")

    @classmethod
    def extract_rules(cls, skill_text: str) -> Dict[str, Any]:
        """从技能文本提取契约规则字典。

        Returns:
            {"must_contain": [...], "forbid_contain": [...],
             "min_len": Optional[int], "max_len": Optional[int]}
        """
        text = skill_text or ""
        must = [m.group(1).strip().strip('"').strip("'") for m in cls._MUST_PATTERN.finditer(text) if m.group(1).strip()]
        forbid = [m.group(1).strip().strip('"').strip("'") for m in cls._FORBID_PATTERN.finditer(text) if m.group(1).strip()]
        min_len: Optional[int] = None
        max_len: Optional[int] = None
        m_min = cls._MIN_LEN_PATTERN.search(text)
        if m_min:
            min_len = int(m_min.group(1))
        m_max = cls._MAX_LEN_PATTERN.search(text)
        if m_max:
            max_len = int(m_max.group(1))
        return {
            "must_contain": must,
            "forbid_contain": forbid,
            "min_len": min_len,
            "max_len": max_len,
        }

    def check(self, response: Any, skill_text: str) -> Dict[str, Any]:
        """检查响应是否符合技能文本中的契约规则。

        Returns:
            {"compliant": bool, "violations": [str], "rules": {...}}
        """
        rules = self.extract_rules(skill_text)
        text = _extract_text(response)
        violations: List[str] = []

        for sub in rules["must_contain"]:
            if sub and sub not in text:
                violations.append(f"缺少必需内容: {sub!r}")
        for sub in rules["forbid_contain"]:
            if sub and sub in text:
                violations.append(f"包含被禁止内容: {sub!r}")
        if rules["min_len"] is not None and len(text) < rules["min_len"]:
            violations.append(f"长度不足: {len(text)} < {rules['min_len']}")
        if rules["max_len"] is not None and len(text) > rules["max_len"]:
            violations.append(f"长度超限: {len(text)} > {rules['max_len']}")

        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "rules": rules,
        }


# ============================================================================
# 4. RuntimeTracker —— applied-rate 趋势 / 退化信号
# ============================================================================


@dataclass
class TrendSignal:
    """退化 / 趋势信号。

    Attributes:
        degraded:       是否退化。
        applied_rate:   当前 applied-rate。
        baseline_rate:  基线 applied-rate。
        delta:          当前与基线的差值。
        sample_count:   已跟踪的样本数。
        reason:         退化原因（未退化时为空）。
    """

    degraded: bool = False
    applied_rate: float = 0.0
    baseline_rate: float = 0.0
    delta: float = 0.0
    sample_count: int = 0
    reason: str = ""


class RuntimeTracker:
    """跟踪 applied-rate 趋势，输出退化信号。

    维护滑动窗口（``window_size``），每个样本为 (applied, selections)
    二元组。当窗口内 applied-rate 跌破 ``threshold`` 时输出退化信号。
    """

    def __init__(
        self,
        window_size: int = 10,
        threshold: float = 0.5,
        min_samples: int = 1,
    ) -> None:
        """
        Args:
            window_size: 滑动窗口大小。
            threshold:   applied-rate 退化阈值（低于该值判定退化）。
            min_samples: 至少累积多少个样本才输出退化判断。
        """
        self.window_size = max(1, int(window_size or 10))
        self.threshold = float(threshold)
        self.min_samples = max(1, int(min_samples or 1))
        self._samples: List[Tuple[int, int]] = []

    def observe(self, applied: int, selections: int) -> None:
        """记录一个样本 (applied, selections)。"""
        self._samples.append((int(applied), int(selections)))
        if len(self._samples) > self.window_size:
            self._samples = self._samples[-self.window_size:]

    def applied_rate(self) -> float:
        """当前窗口的 applied-rate = Σapplied / Σselections。"""
        total_applied = sum(a for a, _ in self._samples)
        total_selections = sum(s for _, s in self._samples)
        if total_selections <= 0:
            return 0.0
        return total_applied / total_selections

    def signal(self, baseline_rate: Optional[float] = None) -> TrendSignal:
        """输出趋势 / 退化信号。

        Args:
            baseline_rate: 可选基线 applied-rate；未提供时不计算 delta。

        Returns:
            TrendSignal。
        """
        rate = self.applied_rate()
        count = len(self._samples)
        base = baseline_rate if baseline_rate is not None else rate
        delta = rate - base

        degraded = False
        reason = ""
        if count >= self.min_samples and rate < self.threshold:
            degraded = True
            reason = f"applied-rate {rate:.3f} 低于阈值 {self.threshold:.3f}"

        return TrendSignal(
            degraded=degraded,
            applied_rate=rate,
            baseline_rate=base,
            delta=delta,
            sample_count=count,
            reason=reason,
        )

    def reset(self) -> None:
        """清空已跟踪样本。"""
        self._samples.clear()


__all__ = [
    "SkillJudgmentAnalyzer",
    "TaskQualityJudge",
    "QualityScores",
    "ResponseContractChecker",
    "RuntimeTracker",
    "TrendSignal",
    "DEFAULT_WEIGHTS",
]