"""
[批次 D1] runtime/observability/tracing —— OpenTelemetry 兼容的 trace。

设计目标（D1 三件套之一）：
    - 一次 ``runtime.run()`` 产生**一个根 span**；
    - 每次 LLM 调用、每次工具调用、每次委派产生一个**子 span**；
    - ``opentelemetry`` 未安装时降级为**内置 span 树记录器**（no-op 到 OTel，
      但 span 树仍可被程序化读取），绝不抛异常、不阻断业务。

实现策略（与 metrics.py 一致的"内置优先 + 可选官方兜底"）：
    - 内置 ``SpanRecorder``：零第三方依赖地以树形数据结构记录 span 的
      name / kind / attributes / 起止时间 / 父子关系，提供 ``to_dict()``
      供测试与诊断断言。**无论 OTel 是否安装，span 树都可观测**。
    - 可选 OTel 桥接：检测到 ``opentelemetry.sdk.trace`` 时，``TracingMiddleware``
      额外把每个 span 通过 SDK ``TracerProvider`` + ``SimpleSpanProcessor``
      导出到注入的 exporter（默认内置 ``InMemorySpanExporter``，测试友好）。
      未安装时该路径自动跳过，open-telemetry 相关符号全部为惰性导入。

span 结构约定（任务的"子 span"均为根 span 的直接 child）：
    runtime.run (root)
      ├─ llm_call    (每个 LLM 调用一次)
      ├─ tool_call   (每个工具调用一次)
      └─ delegation  (每次委派一次)
"""

from __future__ import annotations

import contextvars
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_loop import Context, HookResult, Middleware

# ---------------------------------------------------------------------------
# 可选依赖探测（惰性：只探测顶层命名空间可用性，具体 import 在使用处完成）
# ---------------------------------------------------------------------------

try:  # pragma: no cover - 依赖运行环境
    import opentelemetry.sdk.trace  # noqa: F401
    import opentelemetry.sdk.trace.export  # noqa: F401

    _HAS_OTEL: bool = True
except Exception:  # pragma: no cover
    _HAS_OTEL = False

#: 是否已安装 OpenTelemetry SDK（未安装时降级为内置 span 树记录器）。
otel_available: bool = _HAS_OTEL

#: 委派工具名，用于把委派识别为独立 span 类型。
DELEGATE_TOOL = "delegate_to_agent"


def _now_ns() -> int:
    """当前单调时钟的纳秒时间戳（OTel 使用纳秒 epoch，但这里仅需单调差值）。"""
    return time.perf_counter_ns()


# ============================================================================
# 内置 span 树记录器
# ============================================================================


@dataclass
class SpanRecord:
    """一次 span 的记录（树节点）。

    Attributes:
        name:       span 名称（如 "runtime.run" / "llm_call"）。
        kind:       span 类型（"root" / "llm" / "tool" / "delegation"）。
        attributes: 属性字典。
        parent:     父 SpanRecord（根 span 为 None）。
        start_ns:   开始时间戳（perf_counter_ns）。
        end_ns:     结束时间戳，``None`` 表示尚未结束。
        children:   子 span 列表。
    """

    name: str
    kind: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    parent: Optional["SpanRecord"] = None
    start_ns: int = field(default_factory=_now_ns)
    end_ns: Optional[int] = None
    children: List["SpanRecord"] = field(default_factory=list)

    # OTel 桥接的 span 对象（惰性创建，未启用 OTel 时恒为 None）
    _otel_span: Any = field(default=None, repr=False, compare=False)

    @property
    def duration_ns(self) -> Optional[int]:
        """span 时长（纳秒），未结束时为 None。"""
        if self.end_ns is None:
            return None
        return self.end_ns - self.start_ns

    def to_dict(self) -> Dict[str, Any]:
        """导出为 dict（递归包含子 span）。"""
        return {
            "name": self.name,
            "kind": self.kind,
            "attributes": dict(self.attributes),
            "duration_ns": self.duration_ns,
            "children": [child.to_dict() for child in self.children],
        }


class SpanRecorder:
    """零第三方依赖的 span 树记录器。

    线程安全通过 ``contextvars`` 为每个调用栈维护当前活动 span：
    ``start_span`` 把新 span 压入当前栈顶作为其父，``end`` 弹栈恢复父 span。
    这使得"根 span → 子 span"的父子关系在同步调用链中自然成立，
    也兼容委派递归（递归调用共享同一 contextvars 栈，但每次 start/end 配对）。

    用法::

        recorder = SpanRecorder()
        root = recorder.start_span("runtime.run", "root")
        child = recorder.start_span("llm_call", "llm")
        recorder.end(child)
        recorder.end(root)
        recorder.to_dict()
    """

    def __init__(self) -> None:
        # contextvars 在普通同步调用中退化为线程局部栈，足够满足本用途。
        self._current: contextvars.ContextVar[Optional[SpanRecord]] = (
            contextvars.ContextVar("span_recorder_current", default=None)
        )
        self._roots: List[SpanRecord] = []
        self._all: List[SpanRecord] = []

    # ------------------------------------------------------------------

    @property
    def roots(self) -> List[SpanRecord]:
        return list(self._roots)

    @property
    def spans(self) -> List[SpanRecord]:
        return list(self._all)

    def start_span(
        self,
        name: str,
        kind: str = "span",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> SpanRecord:
        """创建 span，以当前栈顶 span 为父；返回 SpanRecord。"""
        parent = self._current.get()
        record = SpanRecord(
            name=name,
            kind=kind,
            attributes=dict(attributes or {}),
            parent=parent,
        )
        if parent is not None:
            parent.children.append(record)
        else:
            self._roots.append(record)
        self._all.append(record)
        self._current.set(record)
        return record

    def end(self, record: SpanRecord, attributes: Optional[Dict[str, Any]] = None) -> SpanRecord:
        """结束 span，记录结束时间并恢复父 span。幂等。"""
        if record.end_ns is None:
            if attributes:
                record.attributes.update(attributes)
            record.end_ns = _now_ns()
            # 恢复父 span（仅当栈顶仍是本 span 时才弹栈，防御异常顺序）
            if self._current.get() is record:
                self._current.set(record.parent)
        return record

    def reset(self) -> None:
        """清空记录（测试隔离用）。"""
        self._roots.clear()
        self._all.clear()
        self._current.set(None)

    def to_dict(self) -> List[Dict[str, Any]]:
        """导出根 span 列表（递归含子 span）。"""
        return [root.to_dict() for root in self._roots]

    def names(self) -> List[str]:
        """列出所有 span 名称（便于快速断言）。"""
        return [span.name for span in self._all]


# ============================================================================
# OTel 桥接（可选）
# ============================================================================


class InMemorySpanExporter:
    """仅测试/诊断用的内存 span 导出器（兼容 OTel exporter 契约，无需 OTel 类）。"""

    def __init__(self) -> None:
        self._spans: List[Any] = []

    @property
    def spans(self) -> List[Any]:
        return list(self._spans)

    def export(self, spans: List[Any]) -> None:
        self._spans.extend(spans)

    def clear(self) -> None:
        self._spans.clear()


def _build_otel_bridge(exporter: Optional[Any] = None) -> Any:
    """惰性构建 OTel SDK 的 tracer / exporter 桥接；不可用时返回 None。

    返回一个 dict:
        {"tracer": ..., "exporter": ..., "processor": ..., "provider": ...}
    """
    if not otel_available:
        return None
    try:  # pragma: no cover - OTel 安装后的真实路径
        from opentelemetry import trace as otrace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter

        # 若注入的 exporter 不是官方 SpanExporter 子类，则包装为官方 exporter。
        target = exporter
        if target is None:
            target = InMemorySpanExporter()

        if not isinstance(target, SpanExporter):

            class _Adapter(SpanExporter):
                def __init__(self, inner: Any) -> None:
                    self._inner = inner

                def export(self, spans: Any) -> Any:
                    self._inner.export(spans)

                def shutdown(self) -> None:
                    closer = getattr(self._inner, "shutdown", None)
                    if callable(closer):
                        closer()

            target = _Adapter(target)

        processor = SimpleSpanProcessor(target)
        provider = TracerProvider(resource=Resource.create({"service.name": "deep-agent"}))
        provider.add_span_processor(processor)
        otrace.set_tracer_provider(provider)
        tracer = otrace.get_tracer("deep-agent")
        return {
            "tracer": tracer,
            "exporter": target,
            "processor": processor,
            "provider": provider,
        }
    except Exception:  # pragma: no cover
        return None


# ============================================================================
# TracingMiddleware —— 把 Agent Loop 事实映射为 span
# ============================================================================


class TracingMiddleware(Middleware):
    """[D1] 把 ``runtime.run()`` 的一次执行映射为 span 树。

    只观测、不改控制流、不改消息：所有钩子恒返回 ``continue_()``。

    每个 hook 消耗/产出：
        on_enter_loop  → 创建根 span ``runtime.run``
        after_llm      → 创建并立即结束子 span ``llm_call``
        before_tool    → 创建子 span（``delegate_to_agent`` 为 ``delegation``，
                          其余为 ``tool_call``），暂存供 ``after_tool`` 结束
        after_tool     → 结束暂存的 tool/委派 子 span
        on_exit_loop   → 结束根 span

    OTel 桥接：``otel_bridge`` 非 None 时，每个 span 同步创建 OTel span 并导出。
    """

    def __init__(
        self,
        recorder: Optional[SpanRecorder] = None,
        otel_exporter: Optional[Any] = None,
        enable_otel: Optional[bool] = None,
        name: str = "tracing",
    ) -> None:
        """
        Args:
            recorder:     内置 span 树记录器。None 时自建独立实例。
            otel_exporter: OTel exporter。None 时使用内置 ``InMemorySpanExporter``。
            enable_otel:  是否启用 OTel 路径。None 时按 ``otel_available`` 自动判断。
            name:         中间件名称。
        """
        super().__init__(name)
        self.recorder = recorder or SpanRecorder()
        use_otel = otel_available if enable_otel is None else bool(enable_otel)
        self.otel_bridge = _build_otel_bridge(otel_exporter) if use_otel else None

        # per-run 的根 span 与当前 tool span 存 ctx.private（跨 run 复用安全）
        self._ROOT_KEY = "_tracing_root"
        self._TOOL_KEY = "_tracing_tool"

    # ------------------------------------------------------------------

    def on_enter_loop(self, ctx: Context) -> HookResult:
        root = self.recorder.start_span(
            "runtime.run",
            "root",
            attributes=self._run_attrs(ctx),
        )
        if self.otel_bridge is not None:
            root._otel_span = self._start_otel_span(root)
        ctx.private[self._ROOT_KEY] = root
        return HookResult.continue_()

    def after_llm(self, ctx: Context, response: Any) -> HookResult:
        root = self._root(ctx)
        span = self.recorder.start_span(
            "llm_call",
            "llm",
            attributes=self._llm_attrs(response),
        )
        if self.otel_bridge is not None:
            span._otel_span = self._start_otel_span(span)
        self.recorder.end(span)
        if span._otel_span is not None:  # pragma: no cover
            span._otel_span.end()
        return HookResult.continue_()

    def before_tool(self, ctx: Context, tool_call: Dict[str, Any]) -> HookResult:
        name = self._tool_name(tool_call)
        if name == DELEGATE_TOOL:
            kind = "delegation"
            span_name = "delegation"
            attrs = self._delegation_attrs(tool_call)
            # 委派被 DelegationMiddleware skip，after_tool 不会触发，
            # 因此这里创建后立即结束。
            span = self.recorder.start_span(span_name, kind, attributes=attrs)
            if self.otel_bridge is not None:
                span._otel_span = self._start_otel_span(span)
            self.recorder.end(span)
            if span._otel_span is not None:  # pragma: no cover
                span._otel_span.end()
            return HookResult.continue_()

        kind = "tool"
        span_name = "tool_call"
        span = self.recorder.start_span(span_name, kind, attributes={"tool": name})
        if self.otel_bridge is not None:
            span._otel_span = self._start_otel_span(span)
        ctx.private[self._TOOL_KEY] = span
        return HookResult.continue_()

    def after_tool(
        self, ctx: Context, tool_call: Dict[str, Any], result: Any
    ) -> HookResult:
        span = ctx.private.get(self._TOOL_KEY)
        if span is not None:
            span.attributes.update(self._tool_result_attrs(result))
            self.recorder.end(span)
            if span._otel_span is not None:  # pragma: no cover
                span._otel_span.end()
            ctx.private.pop(self._TOOL_KEY, None)
        return HookResult.continue_()

    def on_exit_loop(self, ctx: Context) -> HookResult:
        root = self._root(ctx)
        if root is not None:
            self.recorder.end(root)
            if root._otel_span is not None:  # pragma: no cover
                root._otel_span.end()
            ctx.private.pop(self._ROOT_KEY, None)
        return HookResult.continue_()

    # ------------------------------------------------------------------
    # OTel span 创建（惰性，仅有桥接时调用）
    # ------------------------------------------------------------------

    def _start_otel_span(self, record: SpanRecord) -> Any:  # pragma: no cover
        """在 OTel tracer 上创建与内置记录平行的 span。"""
        if self.otel_bridge is None:
            return None
        try:
            tracer = self.otel_bridge["tracer"]
            parent = None
            if record.parent is not None and record.parent._otel_span is not None:
                from opentelemetry import trace as otrace

                parent = otrace.set_span_in_context(record.parent._otel_span)
            span = tracer.start_span(
                record.name,
                attributes=dict(record.attributes),
                context=parent,
            )
            return span
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _root(self, ctx: Context) -> Optional[SpanRecord]:
        return ctx.private.get(self._ROOT_KEY)

    @staticmethod
    def _tool_name(tool_call: Any) -> str:
        if isinstance(tool_call, dict):
            return str(tool_call.get("name", "") or "")
        return str(getattr(tool_call, "name", "") or "")

    @staticmethod
    def _run_attrs(ctx: Context) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {}
        shared = getattr(ctx, "shared", None)
        if isinstance(shared, dict):
            session = shared.get("session")
            if session is not None:
                attrs["session_id"] = str(
                    getattr(session, "session_id", "") or ""
                )
                metadata = getattr(session, "metadata", None)
                if isinstance(metadata, dict):
                    attrs["task"] = str(metadata.get("task", "") or "")
        return attrs

    @staticmethod
    def _llm_attrs(response: Any) -> Dict[str, Any]:
        if isinstance(response, dict):
            usage = response.get("usage") or {}
            model = response.get("model", "")
            provider = response.get("provider", "")
        else:
            usage = getattr(response, "usage", None) or {}
            model = getattr(response, "model", "") or ""
            provider = getattr(response, "provider", "") or ""
        if not isinstance(usage, dict):
            usage = {}
        return {
            "model": str(model or ""),
            "provider": str(provider or ""),
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }

    @staticmethod
    def _delegation_attrs(tool_call: Any) -> Dict[str, Any]:
        if isinstance(tool_call, dict):
            args = tool_call.get("arguments", tool_call.get("args", {}))
        else:
            args = getattr(tool_call, "arguments", getattr(tool_call, "args", {}))
        if not isinstance(args, dict):
            args = {}
        return {
            "target_agent": str(args.get("target_agent", "") or ""),
            "goal": str(args.get("goal", "") or "")[:200],
        }

    @staticmethod
    def _tool_result_attrs(result: Any) -> Dict[str, Any]:
        if isinstance(result, dict):
            err = result.get("error")
            if isinstance(err, bool):
                return {"error": err}
            return {"error": bool(err)}
        return {"error": bool(getattr(result, "error", False))}


__all__ = [
    "SpanRecord",
    "SpanRecorder",
    "InMemorySpanExporter",
    "otel_available",
    "TracingMiddleware",
    "DELEGATE_TOOL",
]