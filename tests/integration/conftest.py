"""
[测试] integration 包共享 fixture 与装配 helper。

提供统一的 Runtime 装配辅助，避免每个测试重复编写样板：
    - build_runtime: 用 fake 组件装配一个 Runtime，并注册中间件。
    - StubContextManager: 一个最小、可控、可断言的 ContextManager 实现。
    - ContextCaptureMiddleware: 捕获最终 ctx，用于断言 ctx.shared 统计键。

该 helper 只做测试装配，不修改任何功能代码。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from agent_loop import Context, HookResult, Middleware
from runtime.context_manager.interface import (
    ContextManager,
    ContextSnapshot,
    MemoryTrace,
    Message,
)
from runtime.event_log.interface import EventLog
from runtime.llm_adapter.interface import LLMAdapter
from runtime.runtime import ComponentRegistry, Runtime
from runtime.session import Session
from runtime.tool_registry.interface import Tool


class _StubInjection:
    """简单注入策略：把检索到的记忆拼成固定文本。"""

    def inject(self, traces):
        if not traces:
            return "（无相关记忆）"
        return "【记忆】" + "；".join(t.content for t in traces)


# ============================================================================
# 最小可控 ContextManager
# ============================================================================


class StubContextManager(ContextManager):
    """测试用可控 ContextManager。

    维护工作记忆与长期记忆，提供可配置的 extract_memories / apply_decay /
    _consolidate_episodic 行为。不涉及真实 LLM，便于精确断言。
    """

    def __init__(self) -> None:
        self._working: List[Message] = []
        self.ltm: Dict[str, MemoryTrace] = {}
        #: 注入策略（MemoryInjector append 模式会读取 injection_strategy.inject）
        self.injection_strategy = _StubInjection()
        self.extract_calls: int = 0
        self.decay_calls: int = 0
        self.consolidate_calls: int = 0
        # extract_memories 每次返回的记忆条数（可编程）
        self.extract_result: List[MemoryTrace] = []
        # apply_decay 每次归档条数（可编程）
        self.decay_archived: int = 0
        # _consolidate_episodic 每次合并条数（可编程）
        self.consolidate_result: int = 0

    # ---- 工作记忆 ----
    def append(self, message: Message) -> None:
        self._working.append(message)

    def get_working_memory(self) -> List[Message]:
        return list(self._working)

    def clear_working_memory(self) -> None:
        self._working.clear()

    def compress(self, strategy: Optional[str] = None) -> None:
        # 压缩：保留 system 与最后 2 条非 system 消息。
        system = [m for m in self._working if m.role == "system"]
        rest = [m for m in self._working if m.role != "system"]
        self._working = system + rest[-2:]

    # ---- 长期记忆 ----
    def store(self, trace: MemoryTrace) -> None:
        self.ltm[trace.trace_id] = trace

    def retrieve(
        self,
        query: str,
        namespace: Optional[str] = None,
        top_k: int = 5,
        kind: Optional[str] = None,
    ) -> List[MemoryTrace]:
        ns = namespace or "default"
        items = [
            t for t in self.ltm.values()
            if t.namespace == ns and not t.forgotten
            and (kind is None or t.kind == kind)
        ]
        return items[:top_k]

    def update_strength(self, trace_id: str, strength: float) -> None:
        if trace_id in self.ltm:
            self.ltm[trace_id] = self.ltm[trace_id]._replace(strength=strength)

    def archive(self, trace_id: str) -> None:
        if trace_id in self.ltm:
            self.ltm[trace_id] = self.ltm[trace_id]._replace(forgotten=True)

    def forget(self, trace_id: str) -> None:
        self.ltm.pop(trace_id, None)

    def build_context(
        self, system_prompt: str = "", namespace: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages.extend(m.to_dict() for m in self._working)
        return messages

    def snapshot(self) -> ContextSnapshot:
        return ContextSnapshot(
            working_memory=list(self._working),
            long_term_memory=list(self.ltm.values()),
        )

    def enqueue_strengthen(self, trace_ids: List[str]) -> None:
        pass

    def flush_strengthen(self) -> None:
        pass

    def set_markdown_root(self, path: str) -> None:
        pass

    def sync_from_markdown(self) -> int:
        return 0

    def extract_memories(self, namespace: Optional[str] = None) -> List[MemoryTrace]:
        self.extract_calls += 1
        for t in self.extract_result:
            self.store(t)
        return self.extract_result

    def apply_decay(self, namespace: Optional[str] = None) -> int:
        self.decay_calls += 1
        return self.decay_archived

    def _consolidate_episodic(self, namespace: Optional[str] = None) -> int:
        self.consolidate_calls += 1
        return self.consolidate_result


class ContextCaptureMiddleware(Middleware):
    """捕获最终 ctx 引用，供测试断言 ctx.shared 统计键。

    必须注册在中间件末尾（或至少在 ON_EXIT_LOOP 时仍被调度）。
    """

    def __init__(self, name: str = "context_capture") -> None:
        super().__init__(name)
        self.ctx: Optional[Context] = None

    def on_exit_loop(self, ctx: Context) -> HookResult:
        self.ctx = ctx
        return HookResult.continue_()


# ============================================================================
# 装配 helper
# ============================================================================


def build_runtime(
    llm: LLMAdapter,
    tools: List[Tool],
    cm: Optional[ContextManager] = None,
    middlewares: Optional[List[Middleware]] = None,
    *,
    max_iterations: int = 10,
    max_delegation_depth: int = 3,
    agent_registry: Any = None,
    shared_state: Any = None,
    skill_system: Any = None,
    event_log: Optional[EventLog] = None,
) -> Runtime:
    """装配一个 Runtime，注册工具，并按顺序注册中间件。"""
    from runtime.tool_registry.in_memory import InMemoryToolRegistry

    registry_tools = InMemoryToolRegistry()
    for t in tools:
        registry_tools.register(t)

    cm = cm or StubContextManager()

    comp = ComponentRegistry(
        llm_adapter=llm,
        tool_registry=registry_tools,
        context_manager=cm,
        event_log=event_log,
    )

    runtime = Runtime(
        comp,
        max_iterations=max_iterations,
        agent_registry=agent_registry,
        shared_state=shared_state,
        max_delegation_depth=max_delegation_depth,
        skill_system=skill_system,
    )

    for mw in middlewares or []:
        runtime.register_middleware(mw)

    return runtime


def make_session(session_id: str, task: str = "task", agent_id: Optional[str] = None) -> Session:
    """构造一个测试 Session。"""
    metadata = {"task": task}
    if agent_id is not None:
        metadata["agent_id"] = agent_id
    return Session(session_id=session_id, metadata=metadata)


@pytest.fixture
def stub_cm() -> StubContextManager:
    """提供一个空的可控 ContextManager。"""
    return StubContextManager()