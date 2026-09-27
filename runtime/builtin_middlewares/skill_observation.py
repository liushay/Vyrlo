"""
[L5] SkillObservationMiddleware — 技能观察中间件。

职责边界：
    - AFTER_LLM：逐技能判定是否被应用（LLM 判定），写回 applied 计数，
      写 ``ctx.shared["skill_observed"]``；
    - ON_EXIT_LOOP：触发 TaskQualityJudge 打分 + RuntimeTracker 趋势跟踪，
      并输出退化信号。

依赖：
    - 从 ``ctx.shared["skill_system"]`` 解析 SkillSystem 门面；
    - 从 ``ctx.shared["skill_selected"]`` 读取上一轮选中的技能；
    - **未启用 skill_system 时静默放行**（所有钩子直接 continue）。

判定为"应用"的技能，通过 ``SkillSystem.observe_applied`` 写回
``skill_counters.applied``；质量分数通过 RuntimeTracker 记录 applied-rate
趋势样本（selections 与 applied）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent_loop import Context, HookResult, Middleware


def _extract_text(response: Any) -> str:
    """从 LLM 响应提取文本（兼容 dict / dataclass）。"""
    if isinstance(response, dict):
        return str(response.get("content", response.get("text", "")) or "")
    return str(getattr(response, "content", "") or getattr(response, "text", "") or "")


class SkillObservationMiddleware(Middleware):
    """[L5] 技能观察中间件。

    Attributes:
        observed: 本次运行观测到的技能应用结果列表（便于断言）。
        exit_summary: ON_EXIT_LOOP 时写入的汇总。
    """

    def __init__(self, name: str = "skill_observation") -> None:
        super().__init__(name)
        self.observed: List[Dict[str, Any]] = []
        self.exit_summary: Dict[str, Any] = {}

        # 累计 selections / applied（跨迭代）
        self._selections = 0
        self._applied = 0
        # 最近一次 LLM 响应文本（供 ON_EXIT_LOOP 打分）
        self._last_response_text = ""

    # ------------------------------------------------------------------
    # AFTER_LLM —— 技能应用判定
    # ------------------------------------------------------------------

    def after_llm(self, ctx: Context, response: Any) -> HookResult:
        skill_system = self._resolve_skill_system(ctx)
        if skill_system is None:
            return HookResult.continue_()

        selected = ctx.shared.get("skill_selected")
        if not selected or not selected.get("skills"):
            return HookResult.continue_()

        skills = selected.get("skills", [])
        self._selections += len(skills)

        # 重建 Skill 对象（从注入中间件写入的 to_dict 视图）
        skill_objects = self._rebuild_skills(skills)

        # 逐技能判定是否被应用
        judgments = skill_system.judge(self._resolve_task(ctx), response, skill_objects)
        applied_count = 0
        for skill_id, applied in judgments.items():
            skill_system.observe_applied(skill_id, applied)
            if applied:
                applied_count += 1
            self.observed.append({"skill_id": skill_id, "applied": applied})
        self._applied += applied_count

        # 记录本次响应文本，供 ON_EXIT_LOOP 打分
        response_text = _extract_text(response)
        if response_text:
            self._last_response_text = response_text

        # 写 skill_observed
        ctx.shared["skill_observed"] = {
            "judgments": judgments,
            "applied_count": applied_count,
            "selected_count": len(skills),
        }

        # 更新 RuntimeTracker 的 applied-rate 样本
        skill_system.observe_trend(self._applied, self._selections)

        return HookResult.continue_()

    # ------------------------------------------------------------------
    # ON_EXIT_LOOP —— 质量打分 + 趋势信号
    # ------------------------------------------------------------------

    def on_exit_loop(self, ctx: Context) -> HookResult:
        skill_system = self._resolve_skill_system(ctx)
        if skill_system is None:
            return HookResult.continue_()

        # TaskQualityJudge 打分（对最后一段响应）
        quality = None
        if self._last_response_text:
            quality = skill_system.evaluate_quality(
                self._resolve_task(ctx), self._last_response_text
            )
            quality_dict = quality.to_dict() if hasattr(quality, "to_dict") else None
        else:
            quality_dict = None

        # RuntimeTracker 趋势信号
        signal = skill_system.track_signal()

        self.exit_summary = {
            "selections": self._selections,
            "applied": self._applied,
            "quality": quality_dict,
            "signal": {
                "degraded": signal.degraded,
                "applied_rate": signal.applied_rate,
                "sample_count": signal.sample_count,
                "reason": signal.reason,
            },
        }

        # 写回 ctx.shared（供外部断言）
        observed = ctx.shared.setdefault("skill_observed", {})
        if isinstance(observed, dict):
            observed["exit_summary"] = self.exit_summary

        return HookResult.continue_()

    # ------------------------------------------------------------------
    # 依赖解析 / 辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_skill_system(ctx: Context) -> Any:
        shared = getattr(ctx, "shared", None)
        if not isinstance(shared, dict):
            return None
        return shared.get("skill_system")

    @staticmethod
    def _resolve_task(ctx: Context) -> str:
        shared = getattr(ctx, "shared", None)
        if not isinstance(shared, dict):
            return ""
        session = shared.get("session")
        if session is None:
            return ""
        metadata = getattr(session, "metadata", None)
        if isinstance(metadata, dict):
            return str(metadata.get("task", "") or "")
        return ""

    @staticmethod
    def _rebuild_skills(skills: List[Dict[str, Any]]) -> List[Any]:
        """从 dict 视图重建 Skill 对象（供 judge 使用）。"""
        from runtime.skill_system.models import Skill

        rebuilt = []
        for item in skills:
            if not isinstance(item, dict):
                rebuilt.append(item)
                continue
            rebuilt.append(Skill(
                skill_id=item.get("skill_id", ""),
                name=item.get("name", ""),
                text=item.get("text", ""),
                version=int(item.get("version", 1)),
                score=float(item.get("score", 0.0)),
            ))
        return rebuilt


__all__ = ["SkillObservationMiddleware"]