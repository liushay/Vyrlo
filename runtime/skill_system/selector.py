"""
[L5] skill_system/selector — 技能选择（SkillSelector）。

从可用技能列表中，针对任务描述选出应注入的技能。

策略（两级）：
    1. **quality filter**：只保留 ``score > score_threshold`` 的候选。
    2. **LLM hybrid selection**：若注入 ``llm_callable``，则调用 LLM 从
       质量合格的候选里做语义匹配，返回选中项；若未注入 LLM，则按
       ``top_k`` 取质量最高的若干技能（纯质量回退）。

LLM 调用通过 ``llm_callable`` 注入，签名约定：

    def llm_callable(messages: List[Dict[str, str]]) -> str

返回文本需能被解析为技能 skill_id 列表（JSON 数组或纯文本逐行/逗号分隔）。
解析失败时安全降级为质量回退，绝不抛异常。

输出：选中的技能列表（Skill 对象）。
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from runtime.skill_system.models import Skill


class SkillSelector:
    """两级技能选择器。

    Attributes:
        top_k:  纯质量回退时返回的最大技能数。
    """

    def __init__(
        self,
        score_threshold: float = 0.0,
        top_k: int = 5,
        llm_callable: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    ) -> None:
        """
        Args:
            score_threshold: 质量过滤阈值（score > threshold 才进入候选）。
            top_k:           纯质量回退时返回的最大技能数。
            llm_callable:    可选的 LLM 调用函数。None 时跳过 LLM hybrid 选择。
        """
        self.score_threshold = float(score_threshold)
        self.top_k = max(1, int(top_k or 5))
        self._llm_callable = llm_callable

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def select(
        self,
        task: str,
        skills: List[Skill],
        top_k: Optional[int] = None,
    ) -> List[Skill]:
        """从 ``skills`` 中选出应注入的技能。

        Args:
            task:   任务描述。
            skills: 可用技能列表。
            top_k:  覆盖构造时设定的 top_k（可选）。

        Returns:
            选中的技能列表。
        """
        if not skills:
            return []

        # 1. quality filter
        candidates = self._quality_filter(skills)

        # 2. LLM hybrid 选择
        if candidates and self._llm_callable is not None:
            selected = self._llm_hybrid(task, candidates)
            if selected:
                return selected

        # 3. 纯质量回退
        return self._quality_fallback(candidates, top_k)

    # ------------------------------------------------------------------
    # 选择策略
    # ------------------------------------------------------------------

    def _quality_filter(self, skills: List[Skill]) -> List[Skill]:
        """按 score > threshold 过滤，并按 score 降序排序。"""
        filtered = [s for s in skills if float(s.score) > self.score_threshold]
        return sorted(filtered, key=lambda s: float(s.score), reverse=True)

    def _llm_hybrid(self, task: str, skills: List[Skill]) -> List[Skill]:
        """调用 LLM 做语义匹配，返回被选中的技能子集。

        任何异常都返回空列表，由调用方降级到质量回退。
        """
        try:
            raw = self._invoke_llm(task, skills)
        except Exception:
            return []

        ids = self._parse_ids(raw)
        if not ids:
            return []

        index = {s.skill_id: s for s in skills}
        selected: List[Skill] = []
        for skill_id in ids:
            skill = index.get(skill_id)
            if skill is not None and skill not in selected:
                selected.append(skill)
        return selected

    def _invoke_llm(self, task: str, skills: List[Skill]) -> str:
        """构造提示并调用注入的 LLM 函数。"""
        catalog = [
            {"skill_id": s.skill_id, "name": s.name, "text": s.text}
            for s in skills
        ]
        prompt = (
            "你是技能选择器。给定任务描述与候选技能，选出应注入的技能。\n"
            "只能从候选中选择，输出 JSON 数组，元素为 skill_id 字符串。\n"
            f"任务描述：{task}\n"
            f"候选技能：{json.dumps(catalog, ensure_ascii=False)}\n"
            "请输出 JSON 数组，例如：[\"skill_a\"]"
        )
        result = self._llm_callable(
            [{"role": "user", "content": prompt}]
        )
        if isinstance(result, str):
            return result
        # 兼容返回 dict / 对象的情况
        if isinstance(result, dict):
            return str(result.get("content", ""))
        return str(result)

    @staticmethod
    def _parse_ids(raw: str) -> List[str]:
        """从 LLM 返回文本中解析 skill_id 列表。"""
        text = str(raw or "").strip()
        if not text:
            return []

        # 优先按 JSON 数组解析
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(x) for x in data if x]
        except (json.JSONDecodeError, TypeError):
            pass

        # 尝试剥离前后杂讯（如模型包裹的说明文字），提取首个 JSON 数组
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end > start:
            try:
                data = json.loads(text[start:end + 1])
                if isinstance(data, list):
                    return [str(x) for x in data if x]
            except (json.JSONDecodeError, TypeError):
                pass

        # 最后：按逗号 / 换行拆分
        ids: List[str] = []
        for token in text.replace("\n", ",").split(","):
            tok = token.strip().strip('"').strip("'")
            if tok:
                ids.append(tok)
        return ids

    def _quality_fallback(
        self, skills: List[Skill], top_k: Optional[int]
    ) -> List[Skill]:
        """纯质量回退：取 score 最高的前 top_k 个。"""
        limit = self.top_k if top_k is None else max(1, int(top_k))
        return sorted(skills, key=lambda s: float(s.score), reverse=True)[:limit]


__all__ = ["SkillSelector"]