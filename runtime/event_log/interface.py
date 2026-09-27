"""
[L2-装配] EventLog 后端抽象与事件数据模型。

定位：
    本模块是 L2 装配层的基础设施契约层，为 L3 中间件提供
    "运行期事实"的 append-only 记录与只读查询能力。

职责边界：
    - EventLog 只做存储（写入 / 查询 / 聚合），不做任何决策。
    - 决策由消费方中间件完成（如 CostGuardMiddleware 读 LLMCallEvent 判断预算）。
    - 多后端可插拔：默认内存后端，可切换 SQLite 后端，接口完全一致。

抽象与兼容：
    - EventLog 声明完整的后端接口契约（append / emit / 查询 / 聚合）。
    - 接口方法在基类中以"内存后端"作为默认实现，因此 EventLog() 可直接实例化。
      这是为了兼容 L3 中间件（observability / cost_guard）中 `EventLog()` 的
      兜底用法——不修改任何 L3 代码。
    - InMemoryEventLog / SQLiteEventLog 分别提供显式命名的后端实现。
    - 后端只需覆盖三个存储原语：_append_event / _iter_events / _clear_events。

依赖关系：
    event_log  <-  session（Session 持有 EventLog）
    event_log  <-  event_recorder（记录 loop 级事件）
    event_log  <-  builtin_middlewares/observability（写入方）
    event_log  <-  builtin_middlewares/cost_guard（读取方）
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Iterator, List, Optional

# ============================================================================
# 事件类型常量
# ============================================================================

EVENT_LLM_CALL = "llm_call"
EVENT_TOOL_CALL = "tool_call"
EVENT_LOOP_START = "loop_start"
EVENT_LOOP_EXIT = "loop_exit"
EVENT_ERROR = "error"
EVENT_DELEGATION = "delegation"
EVENT_RETRY = "retry"
EVENT_FALLBACK = "fallback"
EVENT_CIRCUIT_OPEN = "circuit_open"

#: 全部内置事件类型（供校验 / 文档化使用）
EVENT_TYPES = (
    EVENT_LLM_CALL,
    EVENT_TOOL_CALL,
    EVENT_LOOP_START,
    EVENT_LOOP_EXIT,
    EVENT_ERROR,
    EVENT_DELEGATION,
    EVENT_RETRY,
    EVENT_FALLBACK,
    EVENT_CIRCUIT_OPEN,
)


# ============================================================================
# 事件数据模型
# ============================================================================


@dataclass
class Event:
    """通用事件。

    Attributes:
        event_type: 事件类型（见上方常量）。
        timestamp:  Unix 时间戳（秒）。
        iteration:  产生事件时所在的 Agent Loop 迭代轮次（0 表示未知）。
        payload:    自由格式的附加值。
    """

    event_type: str = ""
    timestamp: float = field(default_factory=time.time)
    iteration: int = 0
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """展开为扁平字典（payload 内容提升到顶层）。"""
        raw = asdict(self)
        payload = raw.pop("payload", {}) or {}
        merged: Dict[str, Any] = dict(payload)
        merged.update(raw)  # 事件自身字段优先级更高
        return merged

    def to_json(self) -> str:
        """序列化为单行 JSON（结构化日志格式）。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} type={self.event_type!r}>"


@dataclass
class LLMCallEvent(Event):
    """单次 LLM 调用事件（CostGuardMiddleware 的直接数据源）。"""

    event_type: str = EVENT_LLM_CALL
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    elapsed_ms: float = 0.0
    success: bool = True
    error: str = ""


@dataclass
class ToolCallEvent(Event):
    """单次工具调用事件。"""

    event_type: str = EVENT_TOOL_CALL
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    elapsed_ms: float = 0.0
    skipped: bool = False
    error: str = ""


#: 事件类型 → 具体事件类（持久化后端反序列化时使用）
_EVENT_CLASSES: Dict[str, type] = {
    EVENT_LLM_CALL: LLMCallEvent,
    EVENT_TOOL_CALL: ToolCallEvent,
}


def event_to_record(event: Event) -> Dict[str, Any]:
    """将事件展开为可持久化的完整字段字典（保留 payload 结构）。"""
    return asdict(event)


def event_from_record(record: Dict[str, Any]) -> Event:
    """从持久化字段字典重建事件对象（按 event_type 选择具体类型）。

    未知事件类型回退为通用 Event，保证前向兼容（旧数据不因新类型而报错）。
    """
    record = dict(record or {})
    event_type = str(record.get("event_type") or "")
    cls: type = _EVENT_CLASSES.get(event_type, Event)
    allowed = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in record.items() if k in allowed}
    kwargs.setdefault("event_type", event_type)
    return cls(**kwargs)  # type: ignore[arg-type]


# ============================================================================
# EventLog —— 后端接口契约（默认内存实现）
# ============================================================================


class EventLog:
    """append-only 事件日志的接口契约 + 默认内存后端。

    设计约束：
    - 只提供 append / 查询 / 聚合三类操作，不做任何流程控制。
    - 查询返回浅拷贝列表，调用方无法通过返回值修改内部列表。
    - max_events（0 表示不限）限制保留条数，防止长会话内存膨胀。

    后端扩展方式：覆盖三个存储原语即可（见 `_append_event` / `_iter_events`
    / `_clear_events`），其余接口与聚合逻辑自动复用。

    用法::

        log = EventLog()
        log.emit_llm_call(model="gpt-4o", total_tokens=120, cost=0.001)
        log.aggregate_tokens()
    """

    #: 后端标识（memory / sqlite），供工厂与自省使用
    backend: str = "memory"

    def __init__(self, max_events: int = 0) -> None:
        self._events: List[Event] = []
        self._max_events: int = int(max_events or 0)

    # ------------------------------------------------------------------
    # 存储原语（子类覆盖以替换后端）
    # ------------------------------------------------------------------

    def _append_event(self, event: Event) -> None:
        """原语：追加一条事件到底层存储（含 max_events 保留策略）。"""
        self._events.append(event)
        if self._max_events and len(self._events) > self._max_events:
            del self._events[: len(self._events) - self._max_events]

    def _iter_events(self) -> List[Event]:
        """原语：按写入顺序返回全部事件。"""
        return list(self._events)

    def _clear_events(self) -> None:
        """原语：清空底层存储。"""
        self._events.clear()

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def append(self, event: Event) -> Event:
        """追加一条事件，返回该事件本身。"""
        self._append_event(event)
        return event

    def emit(self, event_type: str, **payload: Any) -> Event:
        """构造并追加一条通用事件。"""
        return self.append(Event(event_type=event_type, payload=dict(payload)))

    def emit_llm_call(self, **kwargs: Any) -> LLMCallEvent:
        """构造并追加一条 LLM 调用事件。"""
        return self.append(LLMCallEvent(**kwargs))  # type: ignore[return-value]

    def emit_tool_call(self, **kwargs: Any) -> ToolCallEvent:
        """构造并追加一条工具调用事件。"""
        return self.append(ToolCallEvent(**kwargs))  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def all(self) -> List[Event]:
        """返回全部事件列表（浅拷贝）。"""
        return self._iter_events()

    def of_type(self, event_type: str) -> List[Event]:
        """按事件类型过滤。"""
        return [e for e in self._iter_events() if e.event_type == event_type]

    def llm_calls(self) -> List[LLMCallEvent]:
        """返回全部 LLM 调用事件。"""
        return [e for e in self._iter_events() if e.event_type == EVENT_LLM_CALL]  # type: ignore[misc]

    def tool_calls(self) -> List[ToolCallEvent]:
        """返回全部工具调用事件。"""
        return [e for e in self._iter_events() if e.event_type == EVENT_TOOL_CALL]  # type: ignore[misc]

    def last(self, event_type: Optional[str] = None) -> Optional[Event]:
        """返回最近一条事件（可按类型过滤）。"""
        for event in reversed(self._iter_events()):
            if event_type is None or event.event_type == event_type:
                return event
        return None

    def count(self, event_type: Optional[str] = None) -> int:
        """统计事件数量（可按类型过滤）。"""
        events = self._iter_events()
        if event_type is None:
            return len(events)
        return sum(1 for e in events if e.event_type == event_type)

    def clear(self) -> None:
        """清空全部事件。"""
        self._clear_events()

    # ------------------------------------------------------------------
    # 聚合
    # ------------------------------------------------------------------

    def aggregate_tokens(self) -> Dict[str, int]:
        """聚合所有 LLM 调用事件的 token 消耗。"""
        prompt = completion = total = calls = 0
        for event in self._iter_events():
            if event.event_type != EVENT_LLM_CALL:
                continue
            prompt += getattr(event, "prompt_tokens", 0) or 0
            completion += getattr(event, "completion_tokens", 0) or 0
            total += getattr(event, "total_tokens", 0) or 0
            calls += 1
        if total == 0:
            total = prompt + completion
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "calls": calls,
        }

    def aggregate_cost(self) -> float:
        """聚合所有 LLM 调用事件的成本。"""
        return sum(
            float(getattr(e, "cost", 0.0) or 0.0)
            for e in self._iter_events()
            if e.event_type == EVENT_LLM_CALL
        )

    def aggregate_elapsed(self, event_type: Optional[str] = None) -> Dict[str, float]:
        """聚合耗时指标：count / total_ms / avg_ms / max_ms。"""
        values: List[float] = []
        for event in self._iter_events():
            if event_type is not None and event.event_type != event_type:
                continue
            if hasattr(event, "elapsed_ms"):
                values.append(float(getattr(event, "elapsed_ms", 0.0) or 0.0))
        if not values:
            return {"count": 0, "total_ms": 0.0, "avg_ms": 0.0, "max_ms": 0.0}
        return {
            "count": len(values),
            "total_ms": round(sum(values), 3),
            "avg_ms": round(sum(values) / len(values), 3),
            "max_ms": round(max(values), 3),
        }

    def summary(self) -> Dict[str, Any]:
        """一次性返回完整聚合视图，供 ON_EXIT_LOOP 输出。"""
        return {
            "events": self.count(),
            "llm_calls": self.count(EVENT_LLM_CALL),
            "tool_calls": self.count(EVENT_TOOL_CALL),
            "tokens": self.aggregate_tokens(),
            "cost": round(self.aggregate_cost(), 6),
            "llm_elapsed": self.aggregate_elapsed(EVENT_LLM_CALL),
            "tool_elapsed": self.aggregate_elapsed(EVENT_TOOL_CALL),
        }

    def to_json_lines(self) -> str:
        """导出为 JSON Lines 文本（每行一个事件）。"""
        return "\n".join(e.to_json() for e in self._iter_events())

    # ------------------------------------------------------------------
    # 容器协议
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.count()

    def __iter__(self) -> Iterator[Event]:
        return iter(self._iter_events())

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} backend={self.backend!r} events={len(self)}>"


# ============================================================================
# 依赖解析助手
# ============================================================================


def resolve_event_log(
    event_log: Optional[EventLog] = None,
    session: Any = None,
    ctx: Any = None,
    fallback: Optional[EventLog] = None,
) -> EventLog:
    """解析中间件应使用的 EventLog。

    解析优先级：
        1. 显式传入的 event_log
        2. 显式传入的 session.event_log
        3. ctx.shared["event_log"]
        4. ctx.shared["session"].event_log
        5. fallback（或新建空的内存 EventLog）

    目的：让每个中间件都能独立启停——装配方无论用显式注入还是 ctx.shared
    约定提供依赖，中间件都可退化到可用状态而非抛异常。
    """
    if event_log is not None:
        return event_log
    if session is not None:
        session_log = getattr(session, "event_log", None)
        if session_log is not None:
            return session_log
    if ctx is not None:
        shared = getattr(ctx, "shared", None)
        if isinstance(shared, dict):
            shared_log = shared.get("event_log")
            if shared_log is not None:
                return shared_log
            shared_session = shared.get("session")
            if shared_session is not None:
                shared_session_log = getattr(shared_session, "event_log", None)
                if shared_session_log is not None:
                    return shared_session_log
    return fallback if fallback is not None else EventLog()