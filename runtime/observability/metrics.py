"""
[批次 D1] runtime/observability/metrics —— Prometheus 兼容指标。

设计目标（D1 三件套之一）：
    在不引入强制依赖的前提下，暴露 Prometheus 兼容的运行时指标；
    ``prometheus_client`` 未安装时自动降级，但不报错、不阻断业务。

实现策略：
    - 内置一个**零第三方依赖**的指标注册表（Counter / Gauge / Histogram），
      其 ``render()`` 直接输出 Prometheus text exposition 格式
      （``# HELP`` / ``# TYPE`` / ``name{labels} value``），因此即使没有
      ``prometheus_client``，也能被 Prometheus scrape，功能始终可用。
    - 若检测到 ``prometheus_client`` 已安装，则 ``prometheus_client_available``
      为 ``True``；此时仍走内置实现（保证行为一致、可测试），并额外暴露
      ``prometheus_registry()`` 用于把内置指标桥接进官方 Registry（可选）。
      没有官方库也不影响任何功能——这正是"降级为 no-op"的上界形态。

指标的录入不做任何业务决策，只读取钩子可观测到的事实：
    - LLM 调用 / token / 成本：来自 ``after_llm`` 的 response；
    - 工具调用 / 工具错误：来自 ``after_tool`` 的 tool_call / result；
    - 迭代轮次 / loop 耗时：来自 ``before_iteration`` / ``on_enter_loop`` /
      ``on_exit_loop``；
    - 记忆与技能指标：在 ``on_exit_loop`` 时读取其它中间件写入
      ``ctx.shared`` 的聚合状态（``memory_lifecycle`` / ``memory_injected`` /
      ``skill_selected`` / ``skill_observed``）。读到多少就累加多少，
      读不到则保持 0——不制造虚假数据。

公开指标（与任务书一一对应）：
    - llm_calls_total / llm_tokens_total / llm_cost_total
    - tool_calls_total / tool_errors_total
    - loop_iterations / loop_duration_seconds
    - memory_traces_total / memory_retrievals_total
    - skill_selections_total / skill_applied_total
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from agent_loop import Context, HookResult, Middleware

# ---------------------------------------------------------------------------
# 可选依赖探测
# ---------------------------------------------------------------------------

try:  # pragma: no cover - 探测逻辑依赖运行环境
    import prometheus_client  # noqa: F401

    _HAS_PROMETHEUS: bool = True
except Exception:  # pragma: no cover
    _HAS_PROMETHEUS = False

#: 是否已安装 ``prometheus_client``（未安装时降级，功能仍走内置实现）。
prometheus_client_available: bool = _HAS_PROMETHEUS

#: 默认直方图桶上界（秒）。
DEFAULT_BUCKETS: Tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
)


def _escape_label(value: str) -> str:
    """按 Prometheus 文本格式转义 label 值中的 ``\\`` / ``"`` / 换行。"""
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace('"', '\\"')
    )


# ============================================================================
# 指标模型
# ============================================================================


class _Metric:
    """指标基类：帮助文本 + 标签名列表。"""

    def __init__(self, name: str, help: str, labelnames: Iterable[str] = ()) -> None:
        self.name = name
        self.help = help
        self.labelnames: Tuple[str, ...] = tuple(labelnames)
        self._lock = threading.Lock()

    def _child(self, labels: Optional[Tuple[str, ...]]) -> Tuple[str, ...]:
        """归一化 labels 为内部 key。"""
        if not labels:
            return self.labelnames  # 全空标签等价；占位，与 Counter 的实际存储对齐
        return tuple(labels)


class Counter(_Metric):
    """单调递增计数器（支持标签）。"""

    def __init__(self, name: str, help: str, labelnames: Iterable[str] = ()) -> None:
        super().__init__(name, help, labelnames)
        self._values: Dict[Tuple[str, ...], float] = {}

    def _key(self, labels: Tuple[str, ...]) -> Tuple[str, ...]:
        return labels

    def inc(self, amount: float = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """``amount`` 增量累加。labels 传 dict 时按 ``name -> value`` 映射。"""
        key = self._resolve_key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + float(amount)

    def _resolve_key(self, labels: Optional[Dict[str, str]]) -> Tuple[str, ...]:
        if not labels:
            return ()
        return tuple(str(labels.get(k, "")) for k in self.labelnames)

    def value(self) -> float:
        """无标签聚合值（对 label 组合求和）。"""
        with self._lock:
            return sum(self._values.values())

    def samples(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {"labels": dict(zip(self.labelnames, key)) if self.labelnames else {},
                 "value": val}
                for key, val in self._values.items()
            ]


class Gauge(_Metric):
    """可增可减的仪表指标。"""

    def __init__(self, name: str, help: str, labelnames: Iterable[str] = ()) -> None:
        super().__init__(name, help, labelnames)
        self._values: Dict[Tuple[str, ...], float] = {}

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        key = self._resolve_key(labels)
        with self._lock:
            self._values[key] = float(value)

    def inc(self, amount: float = 1, labels: Optional[Dict[str, str]] = None) -> None:
        key = self._resolve_key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + float(amount)

    def dec(self, amount: float = 1, labels: Optional[Dict[str, str]] = None) -> None:
        self.inc(-float(amount), labels)

    def _resolve_key(self, labels: Optional[Dict[str, str]]) -> Tuple[str, ...]:
        if not labels:
            return ()
        return tuple(str(labels.get(k, "")) for k in self.labelnames)

    def value(self) -> float:
        with self._lock:
            return sum(self._values.values())

    def samples(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {"labels": dict(zip(self.labelnames, key)) if self.labelnames else {},
                 "value": val}
                for key, val in self._values.items()
            ]


class Histogram(_Metric):
    """累计直方图（线性桶 + _sum / _count）。"""

    def __init__(
        self,
        name: str,
        help: str,
        labelnames: Iterable[str] = (),
        buckets: Optional[Iterable[float]] = None,
    ) -> None:
        super().__init__(name, help, labelnames)
        self.buckets: Tuple[float, ...] = tuple(buckets or DEFAULT_BUCKETS)
        # 每 child 一个 (sum, count, bucket_counts)
        self._data: Dict[Tuple[str, ...], Tuple[float, float, Dict[float, float]]] = {}

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        key = self._resolve_key(labels)
        v = float(value)
        with self._lock:
            total, count, buckets = self._data.get(key, (0.0, 0.0, {}))
            for bound in self.buckets:
                if v <= bound:
                    buckets[bound] = buckets.get(bound, 0.0) + 1.0
            self._data[key] = (total + v, count + 1.0, buckets)

    def _resolve_key(self, labels: Optional[Dict[str, str]]) -> Tuple[str, ...]:
        if not labels:
            return ()
        return tuple(str(labels.get(k, "")) for k in self.labelnames)

    def sum(self) -> float:
        with self._lock:
            return sum(d[0] for d in self._data.values())

    def count(self) -> float:
        with self._lock:
            return sum(d[1] for d in self._data.values())

    def samples(self) -> List[Dict[str, Any]]:
        with self._lock:
            out: List[Dict[str, Any]] = []
            for key, (total, count, buckets) in self._data.items():
                out.append({
                    "labels": dict(zip(self.labelnames, key)) if self.labelnames else {},
                    "sum": total,
                    "count": count,
                    "buckets": {str(b): buckets.get(b, 0.0) for b in self.buckets},
                })
            return out


# ============================================================================
# MetricsRegistry —— 指标注册与 Prometheus 渲染
# ============================================================================


class MetricsRegistry:
    """零依赖的 Prometheus 兼容指标注册表。

    用法::

        registry = MetricsRegistry()
        c = registry.counter("llm_calls_total", "Number of LLM calls.")
        c.inc()
        registry.render()   # 输出 Prometheus text format
        registry.snapshot() # 输出字典，便于程序化断言
    """

    def __init__(self) -> None:
        self._metrics: Dict[str, _Metric] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    def counter(self, name: str, help: str, labelnames: Iterable[str] = ()) -> Counter:
        with self._lock:
            m = self._metrics.get(name)
            if m is None:
                m = Counter(name, help, labelnames)
                self._metrics[name] = m
            return m  # type: ignore[return-value]

    def gauge(self, name: str, help: str, labelnames: Iterable[str] = ()) -> Gauge:
        with self._lock:
            m = self._metrics.get(name)
            if m is None:
                m = Gauge(name, help, labelnames)
                self._metrics[name] = m
            return m  # type: ignore[return-value]

    def histogram(
        self,
        name: str,
        help: str,
        labelnames: Iterable[str] = (),
        buckets: Optional[Iterable[float]] = None,
    ) -> Histogram:
        with self._lock:
            m = self._metrics.get(name)
            if m is None:
                m = Histogram(name, help, labelnames, buckets)
                self._metrics[name] = m
            return m  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[_Metric]:
        return self._metrics.get(name)

    def names(self) -> List[str]:
        with self._lock:
            return list(self._metrics.keys())

    def get_value(self, name: str) -> float:
        """读取无标签聚合值（counter/gauge）。histogram 返回 ``sum``。"""
        m = self._metrics.get(name)
        if isinstance(m, Histogram):
            return m.sum()
        if isinstance(m, (Counter, Gauge)):
            return m.value()
        return 0.0

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """导出便于断言的字典结构（非 Prometheus 格式）。"""
        with self._lock:
            out: Dict[str, Any] = {}
            for name, m in sorted(self._metrics.items()):
                if isinstance(m, Histogram):
                    out[name] = {
                        "type": "histogram",
                        "help": m.help,
                        "samples": m.samples(),
                    }
                else:
                    out[name] = {
                        "type": "counter" if isinstance(m, Counter) else "gauge",
                        "help": m.help,
                        "samples": m.samples(),
                    }
            return out

    def render(self) -> str:
        """输出 Prometheus text exposition 格式。"""
        lines: List[str] = []
        with self._lock:
            for name in sorted(self._metrics.keys()):
                m = self._metrics[name]
                lines.append(f"# HELP {m.name} {m.help}")
                type_name = (
                    "histogram"
                    if isinstance(m, Histogram)
                    else "counter"
                    if isinstance(m, Counter)
                    else "gauge"
                )
                lines.append(f"# TYPE {m.name} {type_name}")

                if isinstance(m, Histogram):
                    for sample in m.samples():
                        base_labels = dict(sample["labels"])
                        count = sample["count"]
                        for bound in m.buckets:
                            le = f"{bound:g}"
                            bucket_labels = dict(base_labels)
                            bucket_labels["le"] = le
                            bucket_val = sample["buckets"].get(str(bound), 0.0)
                            lines.append(
                                f"{m.name}_bucket{self._format_labels(bucket_labels)} {bucket_val:g}"
                            )
                        inf_labels = dict(base_labels)
                        inf_labels["le"] = "+Inf"
                        lines.append(
                            f"{m.name}_bucket{self._format_labels(inf_labels)} {count:g}"
                        )
                        lines.append(
                            f"{m.name}_sum{self._format_labels(base_labels)} {sample['sum']:g}"
                        )
                        lines.append(
                            f"{m.name}_count{self._format_labels(base_labels)} {count:g}"
                        )
                else:
                    for sample in m.samples():
                        labels = self._format_labels(sample["labels"])
                        lines.append(f"{m.name}{labels} {sample['value']:g}")
        return "\n".join(lines) + ("\n" if lines else "")

    @classmethod
    def _format_labels(cls, labels: Dict[str, str]) -> str:
        if not labels:
            return ""
        parts = [f'{k}="{_escape_label(v)}"' for k, v in labels.items()]
        return "{" + ",".join(parts) + "}"


# ============================================================================
# 全局默认注册表 + 指标工厂
# ============================================================================

_default_registry = MetricsRegistry()
_default_lock = threading.Lock()


def get_registry() -> MetricsRegistry:
    """返回进程内默认指标注册表。"""
    return _default_registry


def reset() -> None:
    """重置默认注册表（测试隔离用）。"""
    global _default_registry
    with _default_lock:
        _default_registry = MetricsRegistry()


def counter(name: str, help: str, labelnames: Iterable[str] = ()) -> Counter:
    return _default_registry.counter(name, help, labelnames)


def histogram(
    name: str,
    help: str,
    labelnames: Iterable[str] = (),
    buckets: Optional[Iterable[float]] = None,
) -> Histogram:
    return _default_registry.histogram(name, help, labelnames, buckets)


# ---------------------------------------------------------------------------
# 预定义指标 —— 惰性工厂（首次 use 时才创建，避免 import 侧效应）
# ---------------------------------------------------------------------------


def llm_calls_total() -> Counter:
    return counter("llm_calls_total", "Total number of LLM calls.", ("model", "provider"))


def llm_tokens_total() -> Counter:
    return counter("llm_tokens_total", "Total number of tokens consumed by LLM calls.")


def llm_cost_total() -> Counter:
    return counter("llm_cost_total", "Total estimated cost of LLM calls (USD).")


def tool_calls_total() -> Counter:
    return counter("tool_calls_total", "Total number of tool calls.", ("tool",))


def tool_errors_total() -> Counter:
    return counter("tool_errors_total", "Total number of tool call errors.", ("tool",))


def loop_iterations() -> Counter:
    return counter("loop_iterations", "Total number of agent loop iterations.")


def loop_duration_seconds() -> Histogram:
    return histogram(
        "loop_duration_seconds",
        "Duration of a complete runtime.run() in seconds.",
        buckets=DEFAULT_BUCKETS,
    )


def memory_traces_total() -> Counter:
    return counter("memory_traces_total", "Total number of consolidated memory traces.")


def memory_retrievals_total() -> Counter:
    return counter("memory_retrievals_total", "Total number of memory retrievals.")


def skill_selections_total() -> Counter:
    return counter("skill_selections_total", "Total number of skill selections.")


def skill_applied_total() -> Counter:
    return counter("skill_applied_total", "Total number of skills judged as applied.")


def session_active() -> Gauge:
    return gauge("session_active", "Number of currently active sessions.")


def session_completed() -> Counter:
    return counter("session_completed", "Total number of completed sessions.")


# ============================================================================
# 响应 / 结果解析（兼容 dataclass 与 dict，与 ObservabilityMiddleware 对齐）
# ============================================================================


def _extract_usage(response: Any) -> Dict[str, int]:
    if isinstance(response, dict):
        usage = response.get("usage") or {}
    else:
        usage = getattr(response, "usage", None) or {}
    if not isinstance(usage, dict):
        usage = {}
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    total = int(usage.get("total_tokens", 0) or 0)
    if total <= 0:
        total = prompt + completion
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _extract_meta(response: Any) -> Dict[str, Any]:
    if isinstance(response, dict):
        return {
            "model": str(response.get("model", "") or ""),
            "provider": str(response.get("provider", "") or ""),
            "cost": 0.0,
            "error": str(response.get("error", "") or ""),
        }
    return {
        "model": str(getattr(response, "model", "") or ""),
        "provider": str(getattr(response, "provider", "") or ""),
        "cost": float(getattr(response, "cost", 0.0) or 0.0),
        "error": str(getattr(response, "error", "") or ""),
    }


def _tool_name(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        return str(tool_call.get("name", "") or "")
    return str(getattr(tool_call, "name", "") or "")


def _tool_error(result: Any) -> bool:
    if isinstance(result, dict):
        err = result.get("error")
        if isinstance(err, bool):
            return err
        return bool(err)
    return bool(getattr(result, "error", False))


# ============================================================================
# MetricsMiddleware —— 把运行时事实接入默认注册表
# ============================================================================


class MetricsMiddleware(Middleware):
    """[D1] 把 Agent Loop 运行事实映射到 Prometheus 指标。

    只观测、不改控制流、不改消息：所有钩子恒返回 ``continue_()``。
    依赖的数据全部来自钩子入参与 ``ctx.shared``（由其它中间件写入），
    无任何上游时，相应指标保持 0。
    """

    #: per-run 计时在一个中间件实例可跨多次 run 复用时要隔离，用 ContextVar。
    _RUN_START: Any = None

    def __init__(
        self,
        registry: Optional[MetricsRegistry] = None,
        name: str = "metrics",
    ) -> None:
        super().__init__(name)
        self.registry = registry or get_registry()
        self.enabled = True

    # ------------------------------------------------------------------

    def on_enter_loop(self, ctx: Context) -> HookResult:
        self._set_start(ctx, time.perf_counter())
        self.registry.gauge(
            "session_active", "Number of currently active sessions."
        ).inc(1)
        return HookResult.continue_()

    def before_iteration(self, ctx: Context) -> HookResult:
        self.registry.counter(
            "loop_iterations", "Total number of agent loop iterations."
        ).inc(1)
        return HookResult.continue_()

    def after_llm(self, ctx: Context, response: Any) -> HookResult:
        usage = _extract_usage(response)
        meta = _extract_meta(response)

        llm_calls = self.registry.counter(
            "llm_calls_total", "Total number of LLM calls.", ("model", "provider")
        )
        llm_calls.inc(1, {"model": meta["model"], "provider": meta["provider"]})

        llm_tokens = self.registry.counter(
            "llm_tokens_total", "Total number of tokens consumed by LLM calls."
        )
        llm_tokens.inc(usage["total_tokens"])

        llm_cost = self.registry.counter(
            "llm_cost_total", "Total estimated cost of LLM calls (USD)."
        )
        if meta["cost"]:
            llm_cost.inc(meta["cost"])
        return HookResult.continue_()

    def after_tool(
        self, ctx: Context, tool_call: Dict[str, Any], result: Any
    ) -> HookResult:
        name = _tool_name(tool_call)
        is_error = _tool_error(result)

        calls = self.registry.counter(
            "tool_calls_total", "Total number of tool calls.", ("tool",)
        )
        calls.inc(1, {"tool": name})

        if is_error:
            errors = self.registry.counter(
                "tool_errors_total", "Total number of tool call errors.", ("tool",)
            )
            errors.inc(1, {"tool": name})
        return HookResult.continue_()

    def on_exit_loop(self, ctx: Context) -> HookResult:
        start = self._get_start(ctx)
        if start is not None:
            self.registry.histogram(
                "loop_duration_seconds",
                "Duration of a complete runtime.run() in seconds.",
                buckets=DEFAULT_BUCKETS,
            ).observe(time.perf_counter() - start)

        self._sync_memory_skill_metrics(ctx)

        # 会话生命周期
        self.registry.gauge(
            "session_active", "Number of currently active sessions."
        ).dec(1)
        self.registry.counter(
            "session_completed", "Total number of completed sessions."
        ).inc(1)

        return HookResult.continue_()

    # ------------------------------------------------------------------
    # 记忆 / 技能指标同步（从其它中间件写入的 ctx.shared 状态读取）
    # ------------------------------------------------------------------

    def _sync_memory_skill_metrics(self, ctx: Context) -> None:
        shared = getattr(ctx, "shared", None)
        if not isinstance(shared, dict):
            return

        # 记忆沉淀：MemoryLifecycleMiddleware 写的累计条数
        lifecycle = shared.get("memory_lifecycle")
        if isinstance(lifecycle, dict):
            traces = int(lifecycle.get("extracted_traces", 0) or 0)
            if traces:
                self.registry.counter(
                    "memory_traces_total", "Total number of consolidated memory traces."
                ).inc(traces)

        # 记忆检索：MemoryInjectorMiddleware 注入成功即发生一次检索
        injected = shared.get("memory_injected")
        if isinstance(injected, dict) and injected.get("memory_injected"):
            self.registry.counter(
                "memory_retrievals_total", "Total number of memory retrievals."
            ).inc(1)

        # 技能选择：SkillInjectionMiddleware 写入
        selected = shared.get("skill_selected")
        if isinstance(selected, dict):
            count = int(selected.get("count", 0) or 0)
            if count:
                self.registry.counter(
                    "skill_selections_total", "Total number of skill selections."
                ).inc(count)

        # 技能应用：SkillObservationMiddleware 写入
        observed = shared.get("skill_observed")
        if isinstance(observed, dict):
            applied = int(observed.get("applied_count", 0) or 0)
            if applied:
                self.registry.counter(
                    "skill_applied_total", "Total number of skills judged as applied."
                ).inc(applied)

    # ------------------------------------------------------------------
    # per-run 计时（写入 ctx.private，跨 run 复用安全）
    # ------------------------------------------------------------------

    _TIMING_KEY = "_metrics_loop_start"

    def _set_start(self, ctx: Context, value: float) -> None:
        ctx.private[self._TIMING_KEY] = value

    def _get_start(self, ctx: Context) -> Optional[float]:
        return ctx.private.get(self._TIMING_KEY)


__all__ = [
    "MetricsRegistry",
    "Counter",
    "Gauge",
    "Histogram",
    "get_registry",
    "reset",
    "prometheus_client_available",
    "session_active",
    "session_completed",
    "MetricsMiddleware",
]
