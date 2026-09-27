"""
ContextManager 独立测试示例。

展示五层记忆系统的所有核心功能：
- 工作记忆管理
- 长期记忆存储与检索
- 记忆衰减
- 上下文构建（含记忆注入）
- 记忆沉淀
- 策略替换
- 快照

不需要任何其他组件即可运行。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from runtime.context_manager.interface import (
    ContextManager,
    ContextSnapshot,
    DecayStrategy,
    MemoryTrace,
    Message,
)
from runtime.context_manager.layered import LayeredContextManager
from runtime.context_manager.strategies import (
    BM25Retrieval,
    EbbinghausDecay,
    InMemoryStore,
    KeywordRetrieval,
    LLMExtraction,
    NoCompress,
    SlidingWindow,
    SQLiteStore,
    SummaryInjection,
    ThresholdCompress,
    TokenBudget,
)


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_basic_workflow() -> None:
    """测试基本工作流程。"""
    print_section("1. Basic Workflow")

    cm = LayeredContextManager()

    # Append messages using Message constructor
    cm.append(Message(role="assistant", content="Hello! How can I help you?"))
    cm.append(Message(role="user", content="Help me find resources on Python async programming"))

    # Get working memory
    working = cm.get_working_memory()
    print(f"Working memory messages: {len(working)}")
    for msg in working:
        print(f"  [{msg.role}] {msg.content[:50]}")

    # Build context
    ctx = cm.build_context(
        system_prompt="You are a helpful coding assistant. {{memory_context}}"
    )
    print(f"\nBuilt context messages: {len(ctx)}")
    for msg in ctx:
        print(f"  [{msg['role']}] {str(msg['content'])[:80]}...")


def test_long_term_memory() -> None:
    """Test long-term memory storage and retrieval."""
    print_section("2. Long-term Memory Storage & Retrieval")

    cm = LayeredContextManager()

    # Store memories
    traces = [
        MemoryTrace.create(content="User prefers Python 3.12+", namespace="preferences", strength=0.9),
        MemoryTrace.create(content="User is learning the FastAPI framework", namespace="learning", strength=0.8),
        MemoryTrace.create(content="User prefers async programming style", namespace="preferences", strength=0.7),
        MemoryTrace.create(content="User needs to search for resources", namespace="tasks", strength=0.6, tags=["search"]),
    ]
    for t in traces:
        cm.store(t)
    print(f"Stored {len(traces)} memory traces")

    # Retrieve
    results = cm.retrieve("Python async", namespace="preferences", top_k=3)
    print(f"\nRetrieve 'Python async' (namespace=preferences):")
    for r in results:
        print(f"  [{r.strength:.2f}] {r.content} (id={r.trace_id})")

    # Retrieve learning-related
    results = cm.retrieve("fastapi framework", namespace="learning", top_k=3)
    print(f"\nRetrieve 'FastAPI framework' (namespace=learning):")
    for r in results:
        print(f"  [{r.strength:.2f}] {r.content}")


def test_memory_injection() -> None:
    """Test memory injection into system prompt."""
    print_section("3. Memory Injection")

    cm = LayeredContextManager()

    # Query via working memory
    cm.append(Message(role="user", content="I want to write async code in Python"))

    # Store some long-term memories
    cm.store(MemoryTrace.create(
        content="User prefers using asyncio and trio libraries",
        namespace="preferences",
        strength=0.9,
        tags=["python", "async"],
    ))
    cm.store(MemoryTrace.create(
        content="User previously used aiohttp for web scraping",
        namespace="experience",
        strength=0.8,
    ))

    # Build context with memory injection
    ctx = cm.build_context(
        system_prompt="You are a coding assistant. Related memories:\n{{memory_context}}"
    )
    print("Context system prompt:")
    print(ctx[0]["content"])


def test_decay() -> None:
    """Test memory decay."""
    print_section("4. Memory Decay")

    cm = LayeredContextManager()

    # Create an "old" memory
    from datetime import datetime, timedelta, timezone
    old_trace = MemoryTrace(
        trace_id="old_memory_1",
        content="This is a very old memory",
        namespace="default",
        strength=1.0,
        created_at=(datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
        last_recalled_at=datetime.now(timezone.utc).isoformat(),
        tags=[],
    )
    cm.store(old_trace)
    print(f"Original strength: {old_trace.strength:.2f}")

    # Retrieval triggers decay computation
    results = cm.retrieve("old", top_k=5)
    if results:
        print(f"Strength after 30 days decay: {results[0].strength:.4f}")
    else:
        print("Memory decayed to 0 (possibly archived)")


def test_compression() -> None:
    """Test working memory compression."""
    print_section("5. Working Memory Compression")

    cm = LayeredContextManager(max_working_memory=10)

    # Add many messages
    for i in range(15):
        cm.append(Message(role="user", content=f"This is test message #{i} for testing compression"))
    cm.append(Message(role="assistant", content="Got it, I will process these messages"))

    print(f"Before compression: {len(cm.get_working_memory())} messages")

    # Threshold compression
    cm.compress(strategy="threshold")
    print(f"After threshold compression: {len(cm.get_working_memory())} messages")
    for msg in cm.get_working_memory():
        print(f"  [{msg.role}] {msg.content[:60]}")


def test_extraction() -> None:
    """Test memory extraction (heuristic)."""
    print_section("6. Memory Extraction (Heuristic)")

    cm = LayeredContextManager()

    # Add conversation with extractable content
    cm.append(Message(role="user", content="Remember: my most used programming languages are Python and Rust"))
    cm.append(Message(role="assistant", content="Recorded, your common languages are Python and Rust"))
    cm.append(Message(role="user", content="Decided to use SQLite as the default persistent storage solution"))
    cm.append(Message(role="assistant", content="OK, SQLite is a stable solution"))

    # Execute extraction
    new_traces = cm.extract_memories()
    print(f"Extracted {len(new_traces)} new memory traces")
    for t in new_traces:
        print(f"  [{t.strength:.2f}] {t.content} (tags={t.tags})")


def test_snapshot() -> None:
    """Test context snapshot."""
    print_section("7. Context Snapshot")

    cm = LayeredContextManager()
    cm.append(Message(role="user", content="Snapshot test message"))
    cm.store(MemoryTrace.create(content="Long-term memory content", namespace="default"))

    snap = cm.snapshot()
    print(f"Snapshot - working memory: {len(snap.working_memory)} items")
    print(f"Snapshot - long-term memory: {len(snap.long_term_memory)} items")
    print(f"Snapshot - namespace: {snap.namespace}")


def test_strategy_replacement() -> None:
    """Test strategy replacement."""
    print_section("8. Strategy Replacement")

    # Use SQLite + keyword retrieval + Ebbinghaus decay
    import tempfile
    db_path = os.path.join(tempfile.gettempdir(), "test_memory.db")

    cm = LayeredContextManager(
        long_term_store=SQLiteStore(db_path=db_path),
        retrieval_strategy=KeywordRetrieval(),
        decay_strategy=EbbinghausDecay(stability=3600),  # 1 hour half-life
        injection_strategy=SummaryInjection(min_strength=0.2),
        short_term_strategy=TokenBudget(),
        compress_strategy=NoCompress(),
    )

    print(f"Storage engine: {type(cm.long_term_store).__name__}")
    print(f"Retrieval strategy: {type(cm.retrieval_strategy).__name__}")
    print(f"Decay strategy: {type(cm.decay_strategy).__name__}")
    print(f"Injection strategy: {type(cm.injection_strategy).__name__}")

    # Verify functionality works
    cm.store(MemoryTrace.create(content="Strategy replacement test", namespace="test"))
    results = cm.retrieve("strategy", namespace="test")
    print(f"Retrieval results: {len(results)} items")

    # Cleanup
    try:
        os.remove(db_path)
    except OSError:
        pass


def test_l5_memory_system_overview() -> None:
    """Demonstrate the complete five-layer memory system flow."""
    print_section("9. Five-Layer Memory System Complete Flow")

    cm = LayeredContextManager()

    # L1: Data model - create MemoryTrace
    trace = MemoryTrace.create(
        content="User started learning AI Agent development in 2024",
        namespace="biography",
        strength=1.0,
        tags=["learning", "agent", "2024"],
    )
    print(f"L1 Data model: trace_id={trace.trace_id}, strength={trace.strength}")

    # Store to long-term memory
    cm.store(trace)

    # L2: Decay - test with short half-life
    cm._decay_strategy = EbbinghausDecay(stability=10)  # 10 seconds
    import time
    time.sleep(0.1)
    results = cm.retrieve("Agent development", top_k=5, namespace="biography")
    if results:
        print(f"L2 Decayed strength: {results[0].strength:.4f}")

    # L3: Retrieval
    cm._decay_strategy = EbbinghausDecay()  # Restore default
    cm._retrieval_strategy = BM25Retrieval(k1=1.5, b=0.75)
    results = cm.retrieve("AI", top_k=5, namespace="biography")
    if results:
        print(f"L3 BM25 retrieval: {results[0].content[:60]}")

    # L4: Injection
    cm.append(Message(role="user", content="I want to learn Agent development"))
    ctx = cm.build_context(
        system_prompt="You are an AI assistant. {{memory_context}}"
    )
    print(f"L4 Injected system prompt:")
    print(f"  {ctx[0]['content'][:150]}...")

    # L5: Extraction
    cm.append(Message(role="assistant", content="I suggest starting with LangGraph for Agent development"))
    new_traces = cm.extract_memories()
    print(f"L5 Extraction: {len(new_traces)} new memory traces")


if __name__ == "__main__":
    test_basic_workflow()
    test_long_term_memory()
    test_memory_injection()
    test_decay()
    test_compression()
    test_extraction()
    test_snapshot()
    test_strategy_replacement()
    test_l5_memory_system_overview()

    print_section("All Tests Passed [OK]")