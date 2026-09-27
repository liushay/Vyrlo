"""
[L2-装配] EventRecorder — Agent Loop 生命周期事件记录器。

定位：
    这是 L2 装配层的**运行时基础设施**，不是生态中间件。
    它只负责把一个 Agent Loop 的"整体生命周期"事实写入 EventLog：
        ON_ENTER_LOOP  -> EVENT_LOOP_START
        ON_EXIT_LOOP   -> EVENT_LOOP_EXIT
        ON_LOOP_ERROR  -> EVENT_ERROR

与 L3 中间件的职责边界（严格不重叠）：
    - 本记录器**不**记录 LLM 调用事件（EVENT_LLM_CALL）——那是
      ObservabilityMiddleware 的职责。
    - 本记录器**不**记录工具调用事件（EVENT_TOOL_CALL）——同上。
    - 本记录器不做任何决策，恒不改变控制流，也不修改消息。
    因此它与 ObservabilityMiddleware 可以同时启用而互不干扰：
    两者写入的事件类型不相交。

设计约束：
    - 不实现 Middleware 接口、不注册到 AgentLoop 的中间件列表，
      以满足"不新增中间件"的约束。
    - 由 Runtime 显式在 loop.run() 前后调用 record_loop_start /
      record_loop_exit；异常路径由 Runtime 调用 record_error。

用法::

    recorder = EventRecorder(event_log=log)
    recorder.record_loop_start(ctx, session)
    ctx = loop.run(ctx, messages)
    recorder.record_loop_exit(ctx, session)
"""

from __future__ import annotations

from typing import Any, List, Optional

from runtime.event_log import (
    EVENT_ERROR,
    EVENT_LOOP_EXIT,
    EVENT_LOOP_START,
    Event,
    EventLog,
    resolve_event_log,
)


class EventRecorder:
    """把一个 Agent Loop 的生命周期事实写入 EventLog。

    Attributes:
        records: 已记录的生命周期事件列表（供程序化断言）。
    """

    def __init__(
        self,
        event_log: Optional[EventLog] = None,
        session: Any = None,
    ) -> None:
        """
        Args:
            event_log: 事件日志实例。为 None 时从 session / ctx.shared 解析。
            session:   会话实例（可选，用于解析日志与附带 session_id）。
        """
        self._event_log = event_log
        self._session = session
        self.records: List[Event] = []
        self._iterations = 0

    # ------------------------------------------------------------------
    # 生命周期记录
    # ------------------------------------------------------------------

    def record_loop_start(self, ctx: Any = None, session: Any = None) -> Event:
        """记录 Loop 进入事件（ON_ENTER_LOOP 语义）。"""
        self._iterations = 0
        return self._emit(
            EVENT_LOOP_START,
            ctx=ctx,
            session=session,
            payload={
                "session_id": self._session_id(ctx, session),
                "task": self._task(ctx, session),
            },
        )

    def record_loop_exit(self, ctx: Any = None, session: Any = None) -> Event:
        """记录 Loop 正常退出事件（ON_EXIT_LOOP 语义）。

        附带本轮的运行结果指标：迭代轮次、是否被中止、钩子/退出异常数。
        """
        return self._emit(
            EVENT_LOOP_EXIT,
            ctx=ctx,
            session=session,
            payload={
                "session_id": self._session_id(ctx, session),
                "iterations": self._iterations,
                "aborted": bool(getattr(ctx, "aborted", False)),
                "hook_errors": len(getattr(ctx, "hook_errors", []) or []),
                "exit_errors": len(getattr(ctx, "exit_errors", []) or []),
            },
        )

    def record_error(
        self,
        exception: BaseException,
        ctx: Any = None,
        session: Any = None,
    ) -> Event:
        """记录 Loop 级异常事件（ON_LOOP_ERROR 语义）。"""
        return self._emit(
            EVENT_ERROR,
            ctx=ctx,
            session=session,
            payload={
                "session_id": self._session_id(ctx, session),
                "error_type": type(exception).__name__,
                "error": str(exception),
            },
        )

    # ------------------------------------------------------------------
    # 迭代计数（Runtime 每轮迭代后可调用 on_iteration）
    # ------------------------------------------------------------------

    def bind(self, session: Any) -> None:
        """绑定当前会话（复用同一记录器跑多次运行时使用）。"""
        self._session = session

    def on_iteration(self) -> int:
        """递增并返回当前迭代轮次。"""
        self._iterations += 1
        return self._iterations

    @property
    def iterations(self) -> int:
        """当前运行已记录的迭代轮次。"""
        return self._iterations

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _emit(
        self,
        event_type: str,
        ctx: Any,
        session: Any,
        payload: dict,
    ) -> Event:
        """构造事件并写入日志，返回该事件。

        session_id 通过事件对象的动态属性携带，供 SQLite 后端按会话分组；
        该属性不属于 dataclass 字段，不会污染 to_dict()/to_json_lines() 输出。
        """
        log = self._resolve_log(ctx, session)
        event = Event(event_type=event_type, payload=dict(payload))

        sid = payload.get("session_id")
        if sid:
            # SQLite 后端按事件上的 session_id 属性分组；内存后端忽略该属性。
            event.session_id = str(sid)  # type: ignore[attr-defined]

        log.append(event)
        self.records.append(event)
        return event

    def _resolve_log(self, ctx: Any, session: Any) -> EventLog:
        """解析事件日志（显式注入 > session > ctx.shared > 兜底新建）。"""
        return resolve_event_log(
            event_log=self._event_log,
            session=session if session is not None else self._session,
            ctx=ctx,
        )

    def _resolve_session(self, ctx: Any, session: Any) -> Any:
        """解析会话（显式 > 绑定 > ctx.shared）。"""
        if session is not None:
            return session
        if self._session is not None:
            return self._session
        shared = getattr(ctx, "shared", None)
        if isinstance(shared, dict):
            return shared.get("session")
        return None

    def _session_id(self, ctx: Any, session: Any) -> str:
        """提取 session_id（取不到时返回空串）。"""
        resolved = self._resolve_session(ctx, session)
        return str(getattr(resolved, "session_id", "") or "")

    @staticmethod
    def _task(ctx: Any, session: Any) -> str:
        """提取任务描述（从 session.metadata 读取）。"""
        metadata = getattr(session, "metadata", None)
        if isinstance(metadata, dict):
            return str(metadata.get("task", "") or "")
        return ""

    def __repr__(self) -> str:
        return f"<EventRecorder records={len(self.records)}>"