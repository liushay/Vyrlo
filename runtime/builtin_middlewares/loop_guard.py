"""
[批次 D2] LoopGuardMiddleware — Agent 循环检测中间件。

职责：
    在每一轮 AFTER_LLM 观测 Agent 的"行为指纹"（tool_calls 或文本），
    用 ``runtime.resilience.LoopDetector`` 检测连续重复行为；一旦判为死循环，
    返回 ``HookResult.abort_loop()`` 自动中止 Loop。

    与 CostGuard（预算熔断）不同：本中间件负责"行为重复"维度，两者可独立启停。

可配置：
    - ``threshold``：连续重复多少轮判定为循环（默认 3，满足"5 轮内识别"）。
    - ``detector``：可注入自定义 LoopDetector 实例。
    - ``max_iterations``：安全上限，迭代超过该值强制中止（可选，<=0 不启用）。

降级过程记录事件：
    - 检测到循环时向 ``ctx.shared["event_log"]``（若存在）写 ``EVENT_RETRY``
      语义外的 ``EVENT_LOOP_EXIT`` 不适用，这里复用 ``EVENT_RETRY`` 承载
      "loop_detected" 事实，或使用通用 ``EVENT_ERROR``。为不引入新类型，
      写入 ``event_log.emit(EVENT_ERROR, reason="loop_detected", ...)``。

依赖：
    LoopGuardMiddleware -> runtime.resilience.LoopDetector（纯工具）
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent_loop import Context, HookResult, Middleware

from runtime.event_log import EVENT_ERROR, EventLog, resolve_event_log
from runtime.resilience.circuit_breaker import LoopDetector


class LoopGuardMiddleware(Middleware):
    """[D2] Agent 循环检测中间件。

    Attributes:
        detected:           本次运行是否检测到循环。
        detected_fingerprint: 触发循环的指纹。
        iterations:         已观测的迭代轮次（per-run 计数）。
    """

    NAME = "loop_guard"

    def __init__(
        self,
        threshold: int = 3,
        detector: Optional[LoopDetector] = None,
        max_iterations: int = 0,
        name: Optional[str] = None,
        event_log: Optional[EventLog] = None,
    ) -> None:
        """
        Args:
            threshold:      连续重复多少轮判定为循环（默认 3）。
            detector:       可选，自定义 LoopDetector 实例。
            max_iterations: 安全迭代上限（<=0 不启用；>0 时超过即强制中止）。
            name:           中间件名称。
            event_log:      可选事件日志（检测到循环时记录）。
        """
        super().__init__(name or self.NAME)
        self._detector = detector or LoopDetector(threshold=threshold)
        self._max_iterations = max(0, int(max_iterations))
        self._event_log = event_log
        self._iteration = 0
        self.detected = False
        self.detected_fingerprint: Optional[str] = None
        self.abort_reason: str = ""

    # ------------------------------------------------------------------
    # ON_ENTER_LOOP —— 重置 per-run 状态
    # ------------------------------------------------------------------

    def on_enter_loop(self, ctx: Context) -> HookResult:
        self._detector.reset()
        self._iteration = 0
        self.detected = False
        self.detected_fingerprint = None
        self.abort_reason = ""
        return HookResult.continue_()

    # ------------------------------------------------------------------
    # AFTER_LLM —— 观测行为并检测循环
    # ------------------------------------------------------------------

    def after_llm(self, ctx: Context, response: Any) -> HookResult:
        # 已被其他中间件中止则不再检测
        if ctx.aborted:
            return HookResult.continue_()

        self._iteration += 1

        # 安全上限：超过 max_iterations 强制中止
        if self._max_iterations > 0 and self._iteration > self._max_iterations:
            reason = f"迭代超过安全上限 {self._max_iterations}"
            self.abort_reason = reason
            self._record(ctx, "max_iterations_exceeded")
            return HookResult.abort_loop({"reason": reason, "kind": "loop_guard"})

        tool_calls = ctx.get_tool_calls()
        content = self._extract_content(response)

        fingerprint = self._detector.observe(
            self._iteration, tool_calls=tool_calls, content=content
        )
        if fingerprint is not None:
            self.detected = True
            self.detected_fingerprint = fingerprint
            reason = (
                f"检测到循环：连续 {self._detector.detected_rounds} 轮行为重复"
            )
            self.abort_reason = reason
            self._record(ctx, "loop_detected", fingerprint=fingerprint)
            return HookResult.abort_loop({
                "reason": reason,
                "fingerprint": fingerprint,
                "kind": "loop_guard",
            })

        return HookResult.continue_()

    # ------------------------------------------------------------------
    # AFTER_ITERATION —— 兜底检测（覆盖"无 LLM 文本、仅工具循环"场景）
    # ------------------------------------------------------------------

    def after_iteration(self, ctx: Context) -> HookResult:
        # AFTER_LLM 已处理；此处仅保证无 LLM 响应的极端场景下也能记录
        return HookResult.continue_()

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_content(response: Any) -> str:
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            return str(response.get("content", "") or "")
        content = getattr(response, "content", None)
        return str(content) if content is not None else ""

    def _record(self, ctx: Context, reason: str, fingerprint: Optional[str] = None) -> None:
        """把循环 / 安全上限事件写入事件日志。"""
        log = resolve_event_log(event_log=self._event_log, ctx=ctx)
        try:
            log.emit(
                EVENT_ERROR,
                scope="loop_guard",
                reason=reason,
                fingerprint=fingerprint or "",
                iteration=self._iteration,
            )
        except Exception:  # pragma: no cover - 日志失败不阻断
            pass


__all__ = ["LoopGuardMiddleware"]