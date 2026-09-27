"""
[批次 D2] runtime/resilience/retry —— 指数退避重试策略。

设计目标：
    为 LLM 调用与（幂等）工具执行提供统一、可配置的重试能力。

约束（遵循批次 D2）：
    - 所有策略可配置（通过 ``RetryPolicy`` 数据类）。
    - 退避采用指数退避 + 可选抖动 + 上限，避免"风暴"式重试。
    - 降级 / 重试过程通过 ``EventLog`` 记录事件（``EVENT_RETRY``），
      同时用 ``logging`` 双写开发者日志。
    - 不依赖任何中间件：纯函数 / 纯类，可被 LLMAdapter / ToolRegistry / 沙箱复用。

公开符号：
    - RetryExhaustedError: 重试耗尽异常（携带尝试次数与最后错误）。
    - RetryPolicy: 可配置的重试策略（指数退避 + 上限）。
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

from runtime.event_log import EventLog, EVENT_RETRY

logger = logging.getLogger(__name__)


# ============================================================================
# 异常
# ============================================================================


class RetryExhaustedError(Exception):
    """重试耗尽：在最大尝试次数内仍未成功。

    Attributes:
        attempts:      已执行的尝试次数（含首次）。
        last_exception: 最后一次捕获的异常（若失败来自"结果判定"则为 None）。
        last_result:    最后一次操作结果（若失败来自"结果判定"）。
    """

    def __init__(
        self,
        attempts: int,
        last_exception: Optional[BaseException] = None,
        last_result: Any = None,
    ) -> None:
        self.attempts = attempts
        self.last_exception = last_exception
        self.last_result = last_result
        detail = (
            f"最后异常: {type(last_exception).__name__}: {last_exception}"
            if last_exception is not None
            else f"最后结果: {last_result!r}"
        )
        super().__init__(f"重试耗尽：{attempts} 次尝试均失败。{detail}")


# ============================================================================
# RetryPolicy
# ============================================================================


@dataclass
class RetryPolicy:
    """可配置的重试策略。

    Attributes:
        max_attempts:         总尝试次数（含首次），``<=0`` 视为 1。
        base_delay:           首次重试的退避秒数。
        max_delay:            单次退避的上限秒数。
        backoff_factor:       指数因子（默认 2.0 → 1s, 2s, 4s, ...）。
        jitter:               是否对退避加随机抖动（±20%），默认关闭以保行为可预测。
        retryable_exceptions: 可重试的异常类型元组；不在其中的异常立即传播。
        on_retry:             每次重试前的回调 ``(attempt, delay, error) -> None``。

    备注：
        - 当 ``should_retry`` 为 None 时，仅依赖异常判定（适合 LLM 调用）。
        - 提供 ``should_retry(result) -> bool`` 时，则按"结果是否失败"判定
          （适合工具返回 ToolResult(error=True) 的场景），异常仍按
          ``retryable_exceptions`` 处理。
    """

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    jitter: bool = False
    retryable_exceptions: Tuple[type, ...] = (Exception,)
    on_retry: Optional[Callable[[int, float, Optional[BaseException]], None]] = None

    # ------------------------------------------------------------------
    # 退避计算
    # ------------------------------------------------------------------

    def backoff(self, attempt: int) -> float:
        """计算第 ``attempt`` 次重试（attempt 从 1 开始）的退避秒数。"""
        delay = self.base_delay * (self.backoff_factor ** max(0, attempt - 1))
        delay = min(delay, self.max_delay)
        if self.jitter and delay > 0:
            delay = delay * random.uniform(0.8, 1.2)
        return max(0.0, delay)

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------

    def execute(
        self,
        operation: Callable[[], Any],
        *,
        should_retry: Optional[Callable[[Any], bool]] = None,
        event_log: Optional[EventLog] = None,
        labels: Optional[dict] = None,
    ) -> Any:
        """执行 ``operation``，按策略重试，返回首个成功结果。

        Args:
            operation:   无参可调用对象，返回任意结果。
            should_retry: 结果判定函数：返回 True 表示"结果失败，需要重试"。
                          为 None 时只依赖异常。
            event_log:   可选的事件日志；重试发生时写入 ``EVENT_RETRY``。
            labels:      重试事件的附加标签（如 provider / model / tool）。

        Returns:
            operation 的返回值。

        Raises:
            RetryExhaustedError: 最大尝试次数内仍未成功。
        """
        max_attempts = max(1, int(self.max_attempts))
        last_exception: Optional[BaseException] = None
        last_result: Any = None

        for attempt in range(1, max_attempts + 1):
            try:
                result = operation()
            except Exception as exc:  # noqa: BLE001 - 由 retryable_exceptions 精确控制
                if attempt >= max_attempts:
                    raise RetryExhaustedError(
                        attempts=attempt, last_exception=exc
                    ) from exc
                if not self._is_exception_retryable(exc):
                    raise
                last_exception = exc
            else:
                if should_retry is None or not should_retry(result):
                    return result
                last_result = result
                if attempt >= max_attempts:
                    raise RetryExhaustedError(
                        attempts=attempt, last_result=result
                    )

            # 进入重试等待
            delay = self.backoff(attempt)
            self._record(event_log, attempt, delay, last_exception, labels)
            if callable(self.on_retry):
                try:
                    self.on_retry(attempt, delay, last_exception)
                except Exception:  # pragma: no cover - 回调异常不阻断
                    logger.debug("on_retry 回调异常，忽略", exc_info=True)
            if delay > 0:
                time.sleep(delay)

        # 防御性：正常不应到达（循环内部已保证），保底抛出。
        raise RetryExhaustedError(
            attempts=max_attempts,
            last_exception=last_exception,
            last_result=last_result,
        )

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _is_exception_retryable(self, exc: BaseException) -> bool:
        """判断异常是否可重试。"""
        # 空元组视为"全部不可重试"
        return any(isinstance(exc, t) for t in self.retryable_exceptions)

    @staticmethod
    def _record(
        event_log: Optional[EventLog],
        attempt: int,
        delay: float,
        error: Optional[BaseException],
        labels: Optional[dict],
    ) -> None:
        """把一次重试记录为事件（EVENT_RETRY）。"""
        payload = {
            "attempt": attempt,
            "delay_seconds": round(delay, 4),
            "error": (
                f"{type(error).__name__}: {error}" if error is not None else ""
            ),
        }
        if labels:
            payload.update(labels)

        if event_log is not None:
            try:
                event_log.emit(EVENT_RETRY, **payload)
            except Exception:  # pragma: no cover - 日志失败不影响重试
                logger.debug("重试事件写入失败", exc_info=True)
        logger.warning("重试 (第 %d 次重试，延迟 %.3fs): %s", attempt, delay, payload["error"])


__all__ = ["RetryPolicy", "RetryExhaustedError"]