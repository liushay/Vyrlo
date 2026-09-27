"""
[批次 D2] runtime/resilience/circuit_breaker —— 熔断器与循环检测。

包含两个独立组件：

1. ``CircuitBreaker`` — 熔断器
   连续失败达到阈值后进入 OPEN 状态，在冷却期内直接拒绝调用
   （抛 ``CircuitOpenError``），避免对已知故障的下游持续施压。
   适用于 LLM 提供商多次失败后快速失败、短路降级。

2. ``LoopDetector`` — Agent 循环检测
   通过监听每轮迭代的"行为指纹"（LLM 产出的 tool_calls 或文本），
   检测 Agent 是否陷入重复行为；连续 ``threshold`` 轮指纹重复即判为死循环。
   默认阈值 3，配合 "5 轮内识别" 的验收目标（首次重复在第 3 轮即报警，
   加上首轮出现，总能在 5 轮内识别）。

约束（遵循批次 D2）：
    - 所有策略可配置（阈值 / 冷却 / 探测窗口）。
    - 熔断与循环检测过程通过 ``EventLog`` 记录事件
      （``EVENT_CIRCUIT_OPEN`` / ``EVENT_RETRY`` 等）。
    - 纯类，不依赖 Agent Loop 中间件协议（但提供适配：
      ``LoopGuardMiddleware`` 见 runtime/builtin_middlewares）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from runtime.event_log import EventLog, EVENT_CIRCUIT_OPEN

logger = logging.getLogger(__name__)


# ============================================================================
# 异常
# ============================================================================


class CircuitOpenError(Exception):
    """电路处于 OPEN 状态时拒绝调用。

    Attributes:
        opened_at:  进入 OPEN 状态的时间戳。
        retry_after: 距离可以再次尝试的秒数。
    """

    def __init__(self, opened_at: float, retry_after: float, failures: int) -> None:
        self.opened_at = opened_at
        self.retry_after = retry_after
        self.failures = failures
        super().__init__(
            f"电路已熔断（连续 {failures} 次失败，{retry_after:.1f}s 后可重试）"
        )


class LoopDetectedError(Exception):
    """检测到 Agent 陷入重复循环。

    Attributes:
        fingerprint: 触发检测的重复指纹。
        rounds:      检测发生时累计的连续重复轮数。
    """

    def __init__(self, fingerprint: str, rounds: int) -> None:
        self.fingerprint = fingerprint
        self.rounds = rounds
        super().__init__(f"检测到循环：连续 {rounds} 轮行为重复（指纹 {fingerprint!r}）")


# ============================================================================
# CircuitBreaker
# ============================================================================


class CircuitBreaker:
    """可配置的熔断器。

    Attributes:
        failure_threshold: 连续失败多少次后进入 OPEN。
        recovery_timeout:  OPEN 后等待多久进入 HALF_OPEN（秒）。
        event_log:         可选事件日志，进入 OPEN 时写 ``EVENT_CIRCUIT_OPEN``。

    状态机：CLOSED → (连续失败达阈值) → OPEN → (等待 recovery_timeout) →
    HALF_OPEN → (一次成功) → CLOSED；HALF_OPEN 失败则回到 OPEN。

    方法：``call(operation)`` 包装一次可失败操作；``protect`` 可用作装饰器。
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        event_log: Optional[EventLog] = None,
        name: str = "default",
    ) -> None:
        self.failure_threshold = max(1, int(failure_threshold))
        self.recovery_timeout = max(0.0, float(recovery_timeout))
        self._event_log = event_log
        self.name = name

        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._state = "CLOSED"  # CLOSED | OPEN | HALF_OPEN

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        with self._lock:
            return self._current_state()

    @property
    def failures(self) -> int:
        with self._lock:
            return self._failures

    @property
    def open(self) -> bool:
        return self.state == "OPEN"

    def _current_state(self) -> str:
        """在持有锁时计算当前状态（可能因超时从 OPEN 转 HALF_OPEN）。"""
        if self._state == "OPEN":
            elapsed = time.monotonic() - (self._opened_at or time.monotonic())
            if elapsed >= self.recovery_timeout:
                self._state = "HALF_OPEN"
        return self._state

    def _allow(self) -> None:
        """检查是否允许调用；不允许则抛 CircuitOpenError（调用者需持锁）。"""
        if self._state == "OPEN":
            elapsed = time.monotonic() - (self._opened_at or time.monotonic())
            if elapsed >= self.recovery_timeout:
                self._state = "HALF_OPEN"
            else:
                raise CircuitOpenError(
                    opened_at=self._opened_at or 0.0,
                    retry_after=self.recovery_timeout - elapsed,
                    failures=self._failures,
                )

    # ------------------------------------------------------------------
    # 调用
    # ------------------------------------------------------------------

    def call(self, operation: Callable[[], Any]) -> Any:
        """执行操作，受熔断保护。

        Raises:
            CircuitOpenError: 电路 OPEN 时拒绝调用。
            其它异常：操作自身抛出的异常（在 CLOSED / HALF_OPEN 下会累加失败并传播）。
        """
        with self._lock:
            self._allow()

        try:
            result = operation()
        except Exception as exc:  # noqa: BLE001 - 记录失败并传播
            self._record_failure()
            raise
        else:
            self._record_success()
            return result

    def protect(self, operation: Callable[[], Any]) -> Callable[[], Any]:
        """装饰器：把操作包装为受熔断保护的调用。"""

        def _wrapped(*_args: Any, **_kwargs: Any) -> Any:
            return self.call(lambda: operation(*_args, **_kwargs))

        return _wrapped

    # ------------------------------------------------------------------
    # 状态变更（内部）
    # ------------------------------------------------------------------

    def _record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == "HALF_OPEN" or self._failures >= self.failure_threshold:
                self._open()

    def _record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = "CLOSED"
            self._opened_at = None

    def _open(self) -> None:
        """进入 OPEN（内部持有锁）。"""
        was_open = self._state == "OPEN"
        self._state = "OPEN"
        self._opened_at = time.monotonic()
        if not was_open:
            self._emit_open()

    def _emit_open(self) -> None:
        if self._event_log is not None:
            try:
                self._event_log.emit(
                    EVENT_CIRCUIT_OPEN,
                    name=self.name,
                    failures=self._failures,
                    threshold=self.failure_threshold,
                    recovery_timeout=self.recovery_timeout,
                )
            except Exception:  # pragma: no cover
                logger.debug("熔断事件写入失败", exc_info=True)
        logger.warning(
            "电路熔断: %s（连续 %d 次失败，冷却 %.1fs）",
            self.name,
            self._failures,
            self.recovery_timeout,
        )

    def reset(self) -> None:
        """强制复位为 CLOSED（测试 / 手动恢复用）。"""
        with self._lock:
            self._failures = 0
            self._state = "CLOSED"
            self._opened_at = None


# ============================================================================
# LoopDetector
# ============================================================================


@dataclass
class LoopRecord:
    """一轮迭代的行为记录（用于循环检测）。"""

    iteration: int
    fingerprint: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    content: str = ""


class LoopDetector:
    """Agent 循环检测器。

    检测策略：
        对每一轮迭代，计算一个"行为指纹"：优先取 tool_calls 的
        (name, 规范化 arguments) 列表；无工具调用时取 LLM 文本内容的哈希。
        若最近连续 ``threshold`` 轮指纹相同，判为重复循环。

    何时调用：
        - ``observe(iteration, tool_calls, content) -> Optional[str]``：
          记录一轮行为；若检测到循环，返回触发指纹（同时置 ``detected`` 标志）。
        - ``detected``：是否已检测到循环。
        - ``reset()``：清空历史（每次 run 开始前调用）。

    可配置：
        - ``threshold``：连续重复多少轮判定为循环（默认 3 → 首轮出现 + 再 2 轮
          重复即可命中，满足"5 轮内识别"的验收目标）。
        - ``max_history``：保留的最大历史轮数（内存上限）。
    """

    def __init__(self, threshold: int = 3, max_history: int = 20) -> None:
        self.threshold = max(2, int(threshold))
        self.max_history = max(self.threshold, int(max_history))
        self._history: List[LoopRecord] = []
        self.detected: bool = False
        self.detected_fingerprint: Optional[str] = None
        self.detected_rounds: int = 0

    # ------------------------------------------------------------------
    # 观察
    # ------------------------------------------------------------------

    def observe(
        self,
        iteration: int,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        content: str = "",
    ) -> Optional[str]:
        """记录一轮行为，若检测到连续重复循环则返回触发指纹。

        Args:
            iteration:  当前迭代轮次。
            tool_calls: 本轮 LLM 产出的工具调用（Loop 内部规范格式）。
            content:    本轮 LLM 的文本回答（无工具调用时的行为来源）。

        Returns:
            触发循环的指纹；未触发时返回 None。
        """
        fingerprint = self.compute_fingerprint(tool_calls, content)
        record = LoopRecord(
            iteration=iteration,
            fingerprint=fingerprint,
            tool_calls=list(tool_calls or []),
            content=content,
        )

        # 仅当与最近一轮相同时才检查连续重复（否则直接截断重复链）
        if self._history and self._history[-1].fingerprint == fingerprint:
            run_length = self._repeat_run_length(fingerprint) + 1
        else:
            run_length = 1

        self._history.append(record)
        if len(self._history) > self.max_history:
            del self._history[: len(self._history) - self.max_history]

        if run_length >= self.threshold:
            self.detected = True
            self.detected_fingerprint = fingerprint
            self.detected_rounds = run_length
            logger.warning("检测到循环: 连续 %d 轮行为重复 (指纹 %r)", run_length, fingerprint)
            return fingerprint

        return None

    # ------------------------------------------------------------------
    # 指纹计算
    # ------------------------------------------------------------------

    @staticmethod
    def compute_fingerprint(
        tool_calls: Optional[List[Dict[str, Any]]],
        content: str = "",
    ) -> str:
        """计算行为指纹。

        优先使用工具调用（name + 规范化 arguments）；无工具调用时使用文本内容。
        """
        if tool_calls:
            parts = []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                name = str(tc.get("name", "") or "")
                args = LoopDetector._normalize_args(tc.get("arguments", {}))
                parts.append((name, args))
            if parts:
                canonical = json.dumps(parts, sort_keys=True, ensure_ascii=False)
                return f"tools:{hashlib.sha1(canonical.encode('utf-8')).hexdigest()}"
        # 无工具调用：基于文本内容
        text = str(content or "").strip()
        if text:
            return f"text:{hashlib.sha1(text.encode('utf-8')).hexdigest()}"
        return "empty"

    @staticmethod
    def _normalize_args(args: Any) -> Any:
        """规范化参数：字符串尝试 JSON 解析，dict 按 sorted keys 稳定化。"""
        if isinstance(args, str):
            try:
                return json.loads(args) if args.strip() else {}
            except (json.JSONDecodeError, TypeError):
                return args
        if isinstance(args, dict):
            return json.loads(json.dumps(args, sort_keys=True, ensure_ascii=False, default=str))
        return args

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _repeat_run_length(self, fingerprint: str) -> int:
        """计算历史末尾与 ``fingerprint`` 相同的连续轮数（不含本轮）。"""
        run = 0
        for record in reversed(self._history):
            if record.fingerprint == fingerprint:
                run += 1
            else:
                break
        return run

    def reset(self) -> None:
        """清空历史与检测状态。"""
        self._history.clear()
        self.detected = False
        self.detected_fingerprint = None
        self.detected_rounds = 0


__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "LoopDetector",
    "LoopDetectedError",
]