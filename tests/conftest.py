"""
共用 fixture — 为 test_runtime_layer2.py 提供可复用的测试基础设施。

包含：
- 各类 Store 实例（InMemory、SQLite、Vector）
- LayeredContextManager（默认策略）
- MultiProviderAdapter（EchoProvider）
- Tool 与 InMemoryToolRegistry
- 辅助函数
"""

from __future__ import annotations

import math
import pytest
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

from runtime.context_manager.interface import (
    MemoryTrace,
    Message,
)
from runtime.context_manager.strategies import (
    InMemoryStore,
    SQLiteStore,
    VectorStore,
    EbbinghausDecay,
    SlidingWindow,
    NoCompress,
    BM25Retrieval,
    SummaryInjection,
    LLMExtraction,
)
from runtime.context_manager.layered import LayeredContextManager
from runtime.llm_adapter.multi_provider import (
    MultiProviderAdapter,
    EchoProvider,
    AnthropicProvider,
    OpenAIProvider,
)
from runtime.llm_adapter.interface import ModelConfig
from runtime.tool_registry.interface import Tool, ToolRegistry
from runtime.tool_registry.in_memory import InMemoryToolRegistry
from runtime.tool_call_parser.parsers import (
    AutoDetectParser,
    _normalize_schema,
)


# ============================================================================
# Store fixtures
# ============================================================================


@pytest.fixture
def in_memory_store():
    """提供一个空的 InMemoryStore 实例。"""
    return InMemoryStore()


@pytest.fixture
def sqlite_store():
    """提供一个内存 SQLiteStore 实例，测试后自动关闭。"""
    store = SQLiteStore(":memory:")
    yield store
    store.close()


@pytest.fixture
def vector_store():
    """提供一个空的 VectorStore 实例。"""
    return VectorStore()


# ============================================================================
# ContextManager fixtures
# ============================================================================


@pytest.fixture
def ctx_manager(in_memory_store):
    """提供一个使用 InMemoryStore 的 LayeredContextManager。"""
    return LayeredContextManager(
        long_term_store=in_memory_store,
        short_term_strategy=SlidingWindow(max_messages=50),
        compress_strategy=NoCompress(),
        decay_strategy=EbbinghausDecay(stability=86400.0),
        retrieval_strategy=BM25Retrieval(),
        injection_strategy=SummaryInjection(),
        extraction_strategy=LLMExtraction(),
        max_working_memory=50,
    )


@pytest.fixture
def ctx_manager_with_stability():
    """提供 stability=86400 的 LayeredContextManager，用于衰减测试。"""
    store = InMemoryStore()
    return LayeredContextManager(
        long_term_store=store,
        decay_strategy=EbbinghausDecay(stability=86400.0, archive_threshold=0.01),
        max_working_memory=50,
    )


# ============================================================================
# LLM Adapter fixtures
# ============================================================================


@pytest.fixture
def echo_provider():
    """提供一个 EchoProvider 实例。"""
    return EchoProvider()


@pytest.fixture
def multi_provider_adapter(echo_provider):
    """提供一个注册了 EchoProvider 的 MultiProviderAdapter。"""
    adapter = MultiProviderAdapter(
        default_config=ModelConfig(provider="echo", model="echo-test"),
        max_context_tokens=1000,  # 小窗口便于截断测试
    )
    adapter.register_provider(echo_provider)
    return adapter


@pytest.fixture
def multi_provider_adapter_large():
    """提供一个有大窗口的 MultiProviderAdapter，用于正常流程。"""
    adapter = MultiProviderAdapter(
        default_config=ModelConfig(provider="echo", model="echo-test"),
        max_context_tokens=128000,
    )
    adapter.register_provider(EchoProvider())
    return adapter


# ============================================================================
# Tool 与 ToolRegistry fixtures
# ============================================================================


@pytest.fixture
def echo_tool():
    """提供一个简单的 echo Tool，带有标准的 params_schema。"""
    return Tool(
        name="echo",
        description="回显输入的消息",
        params_schema={
            "name": "echo",
            "description": "回显输入的消息",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "要回显的消息",
                    },
                    "repeat": {
                        "type": "integer",
                        "description": "重复次数",
                        "default": 1,
                    },
                },
                "required": ["message"],
            },
        },
        fn=lambda message, repeat=1: message * repeat,
        source="local",
    )


@pytest.fixture
def tool_registry(echo_tool):
    """提供一个注册了 echo 工具的 InMemoryToolRegistry。"""
    registry = InMemoryToolRegistry()
    registry.register(echo_tool)
    return registry


# ============================================================================
# Parser fixtures
# ============================================================================


@pytest.fixture
def auto_detect_parser():
    """提供一个 AutoDetectParser 实例。"""
    return AutoDetectParser()


# ============================================================================
# 辅助函数
# ============================================================================


def make_trace(
    content: str = "测试记忆",
    strength: float = 1.0,
    namespace: str = "default",
    created_at: str | None = None,
    trace_id: str | None = None,
) -> MemoryTrace:
    """快速创建 MemoryTrace 的辅助函数。"""
    import uuid
    return MemoryTrace(
        trace_id=trace_id or str(uuid.uuid4()),
        content=content,
        namespace=namespace,
        strength=strength,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        last_recalled_at=datetime.now(timezone.utc).isoformat(),
    )


def make_message(role: str, content: str, name: str | None = None) -> Message:
    """快速创建 Message 的辅助函数。"""
    return Message(role=role, content=content, name=name)