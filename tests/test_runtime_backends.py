"""
[L2-装配] EventLog / SessionStore / EventRecorder 后端与基础设施测试。

运行方式：python -m pytest tests/test_runtime_backends.py -v
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_loop import Context  # noqa: E402

from runtime.event_log import (  # noqa: E402
    EVENT_ERROR,
    EVENT_LLM_CALL,
    EVENT_LOOP_EXIT,
    EVENT_LOOP_START,
    EVENT_TOOL_CALL,
    EventLog,
    InMemoryEventLog,
    LLMCallEvent,
    SQLiteEventLog,
    ToolCallEvent,
    create_event_log,
    resolve_event_log,
)
from runtime.event_recorder import EventRecorder  # noqa: E402
from runtime.session import Session  # noqa: E402
from runtime.session_store import (  # noqa: E402
    InMemorySessionStore,
    SessionStore,
    SQLiteSessionStore,
    create_session_store,
)


# ============================================================================
# A. EventLog 后端抽象
# ============================================================================


def test_event_log_memory_backend_full_interface():
    """A1: 内存后端需实现全部接口且语义正确。"""
    log = create_event_log("memory", max_events=5)
    assert isinstance(log, InMemoryEventLog)
    assert log.backend == "memory"

    log.emit_llm_call(
        model="gpt-4o", provider="openai",
        prompt_tokens=100, completion_tokens=50, total_tokens=150,
        cost=0.002, elapsed_ms=120.0,
    )
    log.emit_llm_call(model="gpt-4o", total_tokens=50, cost=0.001, elapsed_ms=80.0)
    log.emit_tool_call(tool_name="search", elapsed_ms=10.0, arguments={"q": "x"})
    log.emit(EVENT_LOOP_START, session_id="s1")

    # 查询接口
    assert log.count() == 4
    assert log.count(EVENT_LLM_CALL) == 2
    assert log.count(EVENT_TOOL_CALL) == 1
    assert len(log.llm_calls()) == 2
    assert len(log.tool_calls()) == 1
    assert len(log.of_type(EVENT_LLM_CALL)) == 2
    assert log.last().event_type == EVENT_LOOP_START
    assert log.last(EVENT_TOOL_CALL).tool_name == "search"
    assert isinstance(log.all()[0], LLMCallEvent)
    assert isinstance(log.of_type(EVENT_TOOL_CALL)[0], ToolCallEvent)

    # 聚合接口
    tokens = log.aggregate_tokens()
    assert tokens["prompt_tokens"] == 100
    assert tokens["completion_tokens"] == 50
    assert tokens["total_tokens"] == 200
    assert tokens["calls"] == 2
    assert log.aggregate_cost() == pytest.approx(0.003)

    llm_elapsed = log.aggregate_elapsed(EVENT_LLM_CALL)
    assert llm_elapsed["count"] == 2
    assert llm_elapsed["max_ms"] == pytest.approx(120.0)
    assert llm_elapsed["avg_ms"] == pytest.approx(100.0)

    summary = log.summary()
    assert summary["events"] == 4
    assert summary["llm_calls"] == 2
    assert summary["tokens"]["total_tokens"] == 200

    # JSON Lines 导出
    lines = log.to_json_lines().splitlines()
    assert len(lines) == 4
    parsed = json.loads(lines[0])
    assert parsed["event_type"] == EVENT_LLM_CALL
    assert parsed["total_tokens"] == 150

    # 容器协议与清空
    assert len(log) == 4
    assert len(list(iter(log))) == 4
    log.clear()
    assert len(log) == 0

    # max_events 滑动窗口
    for i in range(10):
        log.emit(EVENT_ERROR, seq=i)
    assert len(log) == 5
    assert log.last().payload["seq"] == 9


def test_event_log_sqlite_backend_matches_memory():
    """A2: SQLite 后端与内存后端行为一致（含会话分组与持久化）。"""
    log = create_event_log("sqlite", db_path=":memory:")
    assert isinstance(log, SQLiteEventLog)
    assert log.backend == "sqlite"

    log.set_default_session("s1")
    log.emit_llm_call(total_tokens=150, cost=0.002, model="m1", elapsed_ms=12.5)
    log.emit_tool_call(tool_name="search", elapsed_ms=3.0, arguments={"q": "abc"})

    log.set_default_session("s2")
    log.emit_llm_call(total_tokens=50, cost=0.001, model="m2")

    assert log.count() == 3
    assert log.count(EVENT_LLM_CALL) == 2
    assert log.count(EVENT_TOOL_CALL) == 1
    assert log.session_ids() == ["s1", "s2"]
    assert log.aggregate_tokens()["total_tokens"] == 200
    assert log.aggregate_cost() == pytest.approx(0.003)

    llm_events = log.llm_calls()
    assert all(isinstance(e, LLMCallEvent) for e in llm_events)
    assert llm_events[0].model == "m1"
    assert llm_events[0].elapsed_ms == pytest.approx(12.5)

    tool_event = log.tool_calls()[0]
    assert isinstance(tool_event, ToolCallEvent)
    assert tool_event.tool_name == "search"
    assert tool_event.arguments == {"q": "abc"}

    assert log.last().event_type == EVENT_LLM_CALL
    assert len(log.of_type(EVENT_LLM_CALL)) == 2

    for line in log.to_json_lines().splitlines():
        json.loads(line)

    log.clear()
    assert len(log) == 0
    log.close()


def test_event_log_sqlite_max_events_per_session():
    """A3: SQLite 后端按会话独立裁剪（不影响其它会话）。"""
    log = SQLiteEventLog(db_path=":memory:", max_events=3)
    log.set_default_session("a")
    for i in range(10):
        log.emit(EVENT_ERROR, seq=i, session_id="a")
    log.set_default_session("b")
    log.emit(EVENT_ERROR, seq=99, session_id="b")

    assert log.count() == 4
    a_events = [e for e in log.all() if e.payload.get("session_id") == "a"]
    assert len(a_events) == 3
    assert a_events[-1].payload["seq"] == 9
    log.close()


def test_event_log_factory_and_resolve():
    """A4: 工厂与依赖解析助手的解析优先级。"""
    assert isinstance(create_event_log(), InMemoryEventLog)
    assert isinstance(create_event_log("sqlite"), SQLiteEventLog)
    assert isinstance(create_event_log("in-memory"), InMemoryEventLog)
    with pytest.raises(ValueError):
        create_event_log("nonsense")

    explicit = InMemoryEventLog()
    session_log = InMemoryEventLog()
    shared_log = InMemoryEventLog()

    session = Session(event_log=session_log)
    ctx = Context()
    ctx.shared["event_log"] = shared_log

    assert resolve_event_log(event_log=explicit, session=session, ctx=ctx) is explicit
    assert resolve_event_log(session=session, ctx=ctx) is session_log
    assert resolve_event_log(ctx=ctx) is shared_log

    ctx2 = Context()
    ctx2.shared["session"] = Session(event_log=session_log)
    assert resolve_event_log(ctx=ctx2) is session_log

    fallback = InMemoryEventLog()
    assert resolve_event_log(fallback=fallback) is fallback
    assert isinstance(resolve_event_log(), EventLog)


def test_event_log_base_class_directly_instantiable():
    """A5: EventLog() 可直接实例化（不破坏 L3 中间件的兜底用法）。"""
    log = EventLog()
    log.emit_llm_call(total_tokens=10)
    assert log.count() == 1
    assert isinstance(log, EventLog)


# ============================================================================
# B. SessionStore
# ============================================================================


def test_session_store_memory_backend():
    """B1: 内存后端 save / load / list_ids / delete / clear。"""
    store = create_session_store("memory")
    assert isinstance(store, InMemorySessionStore)

    session = Session(session_id="s-1", max_tokens=100, metadata={"task": "t"})
    session.add_usage(tokens=30, cost=0.001)

    assert store.save(session) == "s-1"
    loaded = store.load("s-1")
    assert loaded is session
    assert loaded.budget.used_tokens == 30

    assert store.list_ids() == ["s-1"]
    assert len(store) == 1
    assert "s-1" in store

    assert store.load("missing") is None
    assert store.delete("missing") is False
    assert store.delete("s-1") is True
    assert store.list_ids() == []

    store.save(Session(session_id="s-2"))
    store.clear()
    assert store.list_ids() == []


def test_session_store_sqlite_backend_roundtrip():
    """B2: SQLite 后端序列化 / 反序列化往返，预算用量与元数据保留。"""
    store = create_session_store("sqlite", db_path=":memory:")
    assert isinstance(store, SQLiteSessionStore)

    session = Session(
        session_id="s-9", max_tokens=500, max_cost=1.5,
        metadata={"task": "research", "memory_namespace": "ns"},
    )
    session.add_usage(tokens=120, cost=0.25)
    store.save(session)

    loaded = store.load("s-9")
    assert loaded is not None
    assert loaded is not session
    assert loaded.session_id == "s-9"
    assert loaded.budget.max_tokens == 500
    assert loaded.budget.used_tokens == 120
    assert loaded.budget.used_cost == pytest.approx(0.25)
    assert loaded.metadata["task"] == "research"
    assert loaded.metadata["memory_namespace"] == "ns"

    loaded.add_usage(tokens=50)
    store.save(loaded)
    assert store.load("s-9").budget.used_tokens == 170
    assert store.list_ids() == ["s-9"]

    assert store.delete("s-9") is True
    assert store.load("s-9") is None
    store.close()


def test_session_store_factory_and_base_class():
    """B3: 工厂、非法后端、SessionStore() 直接实例化。"""
    assert isinstance(create_session_store(), InMemorySessionStore)
    assert isinstance(create_session_store("sqlite"), SQLiteSessionStore)
    with pytest.raises(ValueError):
        create_session_store("unknown")

    store = SessionStore()
    store.save(Session(session_id="x"))
    assert store.load("x").session_id == "x"


def test_legacy_session_import_path_still_works():
    """B4: `from runtime.session import SessionStore` 旧路径仍可用。"""
    from runtime.session import SessionStore as LegacyStore  # noqa: N814

    assert LegacyStore is SessionStore


# ============================================================================
# C. EventRecorder
# ============================================================================


def test_event_recorder_lifecycle_events():
    """C1: 记录 loop_start / loop_exit / error 三类生命周期事件。"""
    log = InMemoryEventLog()
    session = Session(session_id="s-rec", metadata={"task": "demo"})
    recorder = EventRecorder(event_log=log)
    ctx = Context()

    start = recorder.record_loop_start(ctx, session)
    assert start.event_type == EVENT_LOOP_START
    assert start.payload["session_id"] == "s-rec"
    assert start.payload["task"] == "demo"

    recorder.on_iteration()
    recorder.on_iteration()
    assert recorder.iterations == 2

    exit_event = recorder.record_loop_exit(ctx, session)
    assert exit_event.event_type == EVENT_LOOP_EXIT
    assert exit_event.payload["iterations"] == 2
    assert exit_event.payload["aborted"] is False
    assert exit_event.payload["hook_errors"] == 0

    err = recorder.record_error(RuntimeError("boom"), ctx, session)
    assert err.event_type == EVENT_ERROR
    assert err.payload["error_type"] == "RuntimeError"
    assert "boom" in err.payload["error"]

    assert log.count(EVENT_LLM_CALL) == 0
    assert log.count(EVENT_TOOL_CALL) == 0
    assert log.count() == 3
    assert len(recorder.records) == 3


def test_event_recorder_from_ctx_shared():
    """C2: 无显式注入时从 ctx.shared 解析日志与会话。"""
    log = InMemoryEventLog()
    session = Session(session_id="s-shared")
    ctx = Context()
    ctx.shared["event_log"] = log
    ctx.shared["session"] = session

    recorder = EventRecorder()
    recorder.record_loop_start(ctx)
    recorder.record_loop_exit(ctx)

    assert log.count(EVENT_LOOP_START) == 1
    assert log.count(EVENT_LOOP_EXIT) == 1
    assert log.of_type(EVENT_LOOP_START)[0].payload["session_id"] == "s-shared"


def test_event_recorder_sqlite_grouping():
    """C3: 生命周期事件在 SQLite 后端按会话分组。"""
    log = SQLiteEventLog(db_path=":memory:")
    log.set_default_session("sess-42")
    recorder = EventRecorder(event_log=log)
    session = Session(session_id="sess-42")

    recorder.record_loop_start(None, session)
    recorder.record_loop_exit(None, session)

    assert log.count() == 2
    assert log.session_ids() == ["sess-42"]
    log.close()