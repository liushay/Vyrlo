"""
[批次 D2] runtime/resilience —— 错误处理与降级策略。

本包提供三类可配置、可观测（通过 EventLog 记录事件）的韧性组件：

1. `retry.RetryPolicy`          —— 指数退避重试（上限可配）。
2. `fallback.FallbackPolicy`    —— 主路径失败后按候选链降级。
3. `circuit_breaker.CircuitBreaker`  —— 熔断器（连续失败短路）。
4. `circuit_breaker.LoopDetector`    —— Agent 重复行为循环检测。

设计约束（遵循批次 D2）：
    - 不改任何中间件的核心语义（本包为纯工具层）。
    - 所有策略可配置。
    - 降级 / 重试 / 熔断 / 循环过程均通过 ``runtime.event_log`` 记录事件
      （``EVENT_RETRY`` / ``EVENT_FALLBACK`` / ``EVENT_CIRCUIT_OPEN``）。

公开符号：
    RetryPolicy / RetryExhaustedError
    FallbackPolicy / FallbackError
    CircuitBreaker / CircuitOpenError
    LoopDetector / LoopDetectedError
"""

from runtime.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    LoopDetector,
    LoopDetectedError,
)
from runtime.resilience.fallback import FallbackError, FallbackPolicy
from runtime.resilience.retry import RetryExhaustedError, RetryPolicy

__all__ = [
    "RetryPolicy",
    "RetryExhaustedError",
    "FallbackPolicy",
    "FallbackError",
    "CircuitBreaker",
    "CircuitOpenError",
    "LoopDetector",
    "LoopDetectedError",
]