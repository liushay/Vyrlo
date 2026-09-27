"""
ContextManager — Agent 消息历史与记忆管理。

核心抽象：
- ContextManager:      统一记忆管理接口
- MemoryTrace:         L1 不可变数据模型
- ContextSnapshot:     上下文快照

五层记忆系统：
  L1: 数据模型 (MemoryTrace, 原子操作)
  L2: 衰减与分层 (DecayStrategy, 活跃区/归档区)
  L3: 真相源与检索 (RetrievalStrategy, Markdown 文件)
  L4: 按需注入 (InjectionStrategy)
  L5: 显式沉淀 (ExtractionStrategy)

默认实现：
- LayeredContextManager: 五层记忆系统的默认实现
"""

from runtime.context_manager.interface import (
    ContextManager,
    MemoryTrace,
    ContextSnapshot,
    Message,
    # 策略接口
    ShortTermStrategy,
    LongTermStore,
    CompressStrategy,
    DecayStrategy,
    RetrievalStrategy,
    InjectionStrategy,
    ExtractionStrategy,
)
from runtime.context_manager.strategies import (
    # 短期策略
    SlidingWindow,
    TokenBudget,
    SummaryBuffer,
    # 长期策略
    InMemoryStore,
    SQLiteStore,
    VectorStore,
    # 压缩策略
    ThresholdCompress,
    LLMSummarize,
    NoCompress,
    # 衰减策略
    EbbinghausDecay,
    # 检索策略
    KeywordRetrieval,
    BM25Retrieval,
    # 注入策略
    SummaryInjection,
    # 沉淀策略
    LLMExtraction,
)
from runtime.context_manager.layered import LayeredContextManager

__all__ = [
    # 核心接口
    "ContextManager",
    "MemoryTrace",
    "ContextSnapshot",
    "Message",
    # 策略接口
    "ShortTermStrategy",
    "LongTermStore",
    "CompressStrategy",
    "DecayStrategy",
    "RetrievalStrategy",
    "InjectionStrategy",
    "ExtractionStrategy",
    # 默认实现
    "LayeredContextManager",
    # 短期策略
    "SlidingWindow",
    "TokenBudget",
    "SummaryBuffer",
    # 长期策略
    "InMemoryStore",
    "SQLiteStore",
    "VectorStore",
    # 压缩策略
    "ThresholdCompress",
    "LLMSummarize",
    "NoCompress",
    # 衰减策略
    "EbbinghausDecay",
    # 检索策略
    "KeywordRetrieval",
    "BM25Retrieval",
    # 注入策略
    "SummaryInjection",
    # 沉淀策略
    "LLMExtraction",
]