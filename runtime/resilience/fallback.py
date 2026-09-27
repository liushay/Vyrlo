"""
[批次 D2] runtime/resilience/fallback —— 降级策略。

设计目标：
    提供"主路径失败 → 备用路径"的可配置降级执行器，用于：
      - LLM 主模型失败 → 降级到备用模型（见 MultiProviderAdapter）；
      - 沙箱 Docker 不可用 → 降级到 Local。

约束（遵循批次 D2）：
    - 所有降级策略可配置：
        * ``fail_fast``：主路径失败后是否直接抛出，而不是尝试降级；
        * ``fallback_on_result``：结果判定函数，主路径"成功"则不降级；
        * ``on_fallback``：降级回调，用于记录事件；
        * ``chain``：显式降级链（可选，覆盖 list 的默认顺序）。
    - 降级过程通过 ``EventLog`` 记录 ``EVENT_FALLBACK`` 事件。
    - 纯类，不依赖任何中间件，可被 LLMAdapter / SandboxRegistry 复用。

公开符号：
    - FallbackError: 所有候选（含降级）均失败时抛出。
    - FallbackPolicy: 可配置的降级执行器。
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from runtime.event_log import EventLog, EVENT_FALLBACK

logger = logging.getLogger(__name__)


class FallbackError(Exception):
    """所有候选（主路径 + 降级路径）均失败时抛出。

    Attributes:
        attempts: 已尝试的候选数量。
        errors:   每个候选的失败错误（按尝试顺序，含首个失败候选）。
        last_error: 最后一个失败（可为 None，若失败来自"结果判定"）。
    """

    def __init__(
        self,
        attempted: int,
        errors: Optional[List[BaseException]] = None,
    ) -> None:
        self.attempted = attempted
        self.errors = errors or []
        self.last_error = self.errors[-1] if self.errors else None
        detail = (
            f"最后错误: {type(self.last_error).__name__}: {self.last_error}"
            if self.last_error is not None
            else "结果判定失败"
        )
        super().__init__(f"降级耗尽：{attempted} 个候选均失败。{detail}")


class FallbackPolicy:
    """可配置的降级执行器。

    语义：
        按 ``candidates`` 顺序依次尝试。第 0 个是主路径，其后是降级路径。
        每个候选 ``(key, operation)``：
            - ``operation()`` 抛异常 → 该候选失败，继续下一个候选；
            - ``operation()`` 正常返回 → 若提供了 ``fallback_on_result`` 且
              判定为 True（结果失败），则继续下一个候选；否则返回该结果。

        全部候选失败时抛 ``FallbackError``。

    Attributes:
        candidates:       候选列表 [(key, operation), ...]，key 用于日志与标记。
        fallback_on_result: 结果判定函数 ``(key, result) -> bool``。
                            为 None 时仅依赖异常；返回 True 表示需要降级。
        on_fallback:      降级发生时回调 ``(from_key, to_key, error) -> None``。
        fail_fast:        True 时主路径失败直接抛 ``FallbackError``，不降级
                          （等价于关闭降级——用于"拒绝执行"场景）。

    事件记录：
        每次发生降级（尝试下一个候选）都会 ``event_log.emit(EVENT_FALLBACK, ...)``，
        携带 ``from`` / ``to`` / ``reason`` 字段。
    """

    def __init__(
        self,
        candidates: Optional[List[Any]] = None,
        *,
        fallback_on_result: Optional[Callable[[Any, Any], bool]] = None,
        on_fallback: Optional[Callable[[Any, Any, Optional[BaseException]], None]] = None,
        fail_fast: bool = False,
        event_log: Optional[EventLog] = None,
        labels: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.candidates: List[Any] = list(candidates or [])
        self.fallback_on_result = fallback_on_result
        self.on_fallback = on_fallback
        self.fail_fast = bool(fail_fast)
        self._event_log = event_log
        self._labels = dict(labels or {})

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------

    def execute(self) -> Any:
        """按候选顺序执行，返回首个成功结果（见类 docstring）。

        Returns:
            某个候选的返回值。

        Raises:
            FallbackError: 所有候选均失败，或 ``fail_fast`` 且主路径失败时。
        """
        if not self.candidates:
            raise FallbackError(attempted=0)

        errors: List[BaseException] = []

        for idx, (key, operation) in enumerate(self.candidates):
            last_error: Optional[BaseException] = None
            try:
                result = operation()
            except Exception as exc:  # noqa: BLE001 - 降级器需捕获任意失败
                last_error = exc
                result = None
                if self.fail_fast:
                    raise FallbackError(attempted=1, errors=[exc]) from exc
            else:
                if self.fallback_on_result is None or not self.fallback_on_result(
                    key, result
                ):
                    return result

            # 走到这里：当前候选失败
            errors.append(last_error)  # type: ignore[arg-type]

            # 已经是最后一个候选 → 全部失败
            if idx == len(self.candidates) - 1:
                break

            # 尝试降级到下一个候选
            to_key, _ = self.candidates[idx + 1]
            self._record_fallback(key, to_key, last_error)
            if callable(self.on_fallback):
                try:
                    self.on_fallback(key, to_key, last_error)
                except Exception:  # pragma: no cover - 回调异常不阻断
                    logger.debug("on_fallback 回调异常，忽略", exc_info=True)

        raise FallbackError(attempted=len(self.candidates), errors=errors)

    # ------------------------------------------------------------------
    # 记录
    # ------------------------------------------------------------------

    def _record_fallback(
        self, from_key: Any, to_key: Any, error: Optional[BaseException]
    ) -> None:
        """把一次降级写入事件日志。"""
        payload: Dict[str, Any] = {
            "from": str(from_key),
            "to": str(to_key),
            "reason": (
                f"{type(error).__name__}: {error}" if error is not None else "result_failed"
            ),
        }
        payload.update(self._labels)

        if self._event_log is not None:
            try:
                self._event_log.emit(EVENT_FALLBACK, **payload)
            except Exception:  # pragma: no cover - 日志失败不阻断
                logger.debug("降级事件写入失败", exc_info=True)
        logger.warning("降级: %s → %s (%s)", from_key, to_key, payload["reason"])


__all__ = ["FallbackPolicy", "FallbackError"]