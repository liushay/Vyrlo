"""
ContextManager 抽象接口 — 五层记忆系统的接口定义。

设计原则：
1. 接口契约：定义所有组件必须实现的抽象方法
2. 策略接口独立可替换
3. 数据模型不可变
4. 工作记忆与长期记忆分离

五层记忆系统职责划分：
L1: 数据模型 — MemoryTrace 不可变数据结构
L2: 衰减与分层 — DecayStrategy 接口
L3: 检索 — RetrievalStrategy 接口 + LongTermStore 接口
L4: 注入 — InjectionStrategy 接口
L5: 沉淀 — ExtractionStrategy 接口

[FIX-D1] markdown 相关方法提升到 ContextManager 抽象接口，包含 set_markdown_root
和 sync_from_markdown，提供默认空实现（降级语义：标记为不支持，返回 0）。

[FIX-L2] LongTermStore 新增 get(trace_id) 方法，支持 O(1) 按 ID 获取。
[FIX-L3] LongTermStore 新增 update_strengths_batch(updates) 方法，支持批量提交。
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


# ============================================================================
# Message — 消息数据结构
# ============================================================================


@dataclass
class Message:
    """
    标准消息数据结构。

    兼容 OpenAI / Anthropic 等多种格式。

    [FIX-L10] to_dict 输出明确标注格式为 "openai"，AnthropicProvider.call 中做转换。
    [FIX-P2-2] 新增 timestamp 字段，记录消息创建时间。
    [FIX-P2-3] tool_calls 格式约定：
        - OpenAI 格式：List[{"id": str, "type": "function", "function": {"name": str, "arguments": str}}]
        - Anthropic 格式：List[{"id": str, "type": "tool_use", "name": str, "input": {}}]
        - 内部存储统一使用 OpenAI 格式，通过 to_dict(format) 按需转换。
    """

    role: str  # system, user, assistant, tool
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _parse_json_value(value: Any) -> Any:
        """[FIX-N3] 若值为 JSON 字符串，解析为 dict；否则原样返回。"""
        if isinstance(value, str) and value.strip():
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                pass
        return value if value is not None else {}

    def to_dict(self, format: str = "openai") -> Dict[str, Any]:
        """
        转换为字典格式。

        [FIX-L10] 支持两种格式：
        - "openai":   OpenAI 兼容格式，tool_calls 格式为 {"id", "type", "function"}
        - "anthropic": Anthropic 兼容格式，tool_calls 格式为 {"id", "type", "name", "input"}

        默认使用 OpenAI 格式（MultiProviderAdapter 中的 AnthropicProvider.call
        会自行将 OpenAI 格式转换为 Anthropic 原生格式）。
        """
        result: Dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.name:
            result["name"] = self.name

        if self.tool_calls:
            if format == "anthropic":
                # Anthropic 扁平格式: [{"id": ..., "type": "tool_use", "name": ..., "input": ...}]
                # 从 OpenAI function.name 提取函数名，兼容嵌套和扁平两种存储形式
                result["tool_calls"] = [
                    {
                        "id": tc.get("id", ""),
                        "type": "tool_use",
                        "name": (
                            tc.get("name", "")
                            or tc.get("function", {}).get("name", "")
                        ),
                        "input": self._parse_json_value(
                            tc.get("args", tc.get("input", tc.get("function", {}).get("arguments", {})))
                        ),
                    }
                    for tc in self.tool_calls
                ]
            else:
                # OpenAI 格式
                result["tool_calls"] = [
                    {
                        "id": tc.get("id", ""),
                        "type": tc.get("type", "function"),
                        "function": tc.get("function", {}),
                    }
                    for tc in self.tool_calls
                ]

        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result


# ============================================================================
# MemoryTrace — L1 数据模型（不可变）
# ============================================================================


# [FIX-L10] ToolExecutorFn 类型别名：明确返回契约
ToolExecutorFn = Any  # Callable[[Dict[str, Any]], ToolResult]

@dataclass(frozen=True)
class MemoryTrace:
    """
    L1 数据模型：不可变记忆轨迹。

    五个原子操作：
    - store: 创建新记忆轨迹
    - retrieve: 检索记忆
    - update_strength: 更新强度
    - archive: 归档
    - forget: 删除

    纯数据变换，不含 LLM 调用，不依赖任何存储引擎。

    Attributes:
        trace_id: 唯一标识。
        content: 记忆摘要（用于检索和注入）。
        full_content: 完整内容引用（路径或全文）。
        namespace: 命名空间。
        strength: 记忆强度 [0.0, 1.0]。
        created_at: 创建时间。
        last_recalled_at: 最后唤起时间。
        tags: 标签列表。
        metadata: 扩展元数据。
    """

    trace_id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    full_content: str = ""
    namespace: str = "default"
    strength: float = 1.0
    # [FIX-P0-1] base_strength 用于衰减计算基准值，不随 decay 改变
    base_strength: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_recalled_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tags: List[str] = field(default_factory=list)
    # 记忆抽象类型："episodic"（情景记忆） | "semantic"（语义记忆）
    kind: str = "episodic"
    # 是否已被合并/遗忘（forgotten=True 时不计入 all_active / retrieve）
    forgotten: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        content: str,
        namespace: str = "default",
        strength: float = 1.0,
        tags: Optional[List[str]] = None,
        full_content: str = "",
        kind: str = "episodic",
        forgotten: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "MemoryTrace":
        """工厂方法：创建新记忆轨迹。"""
        now = datetime.now(timezone.utc).isoformat()
        strength_val = min(1.0, max(0.0, strength))
        # [FIX-P0-1] base_strength 记录初始强度，用于衰减计算的绝对基准
        return cls(
            trace_id=str(uuid4()),
            content=content,
            full_content=full_content,
            namespace=namespace,
            strength=strength_val,
            base_strength=strength_val,
            created_at=now,
            last_recalled_at=now,
            tags=tags or [],
            kind=kind,
            forgotten=forgotten,
            metadata=metadata or {},
        )

    def _replace(self, **changes: Any) -> "MemoryTrace":
        """不可变替换辅助方法。"""
        d = {
            "trace_id": self.trace_id,
            "content": self.content,
            "full_content": self.full_content,
            "namespace": self.namespace,
            "strength": self.strength,
            "base_strength": self.base_strength,
            "created_at": self.created_at,
            "last_recalled_at": self.last_recalled_at,
            "tags": self.tags,
            "kind": self.kind,
            "forgotten": self.forgotten,
            "metadata": self.metadata,
        }
        d.update(changes)
        return MemoryTrace(**d)

    def to_dict(self) -> Dict[str, Any]:
        # [FIX-N1] 加入 base_strength，防止序列化丢失衰减基准
        return {
            "trace_id": self.trace_id,
            "content": self.content,
            "full_content": self.full_content,
            "namespace": self.namespace,
            "strength": self.strength,
            "base_strength": self.base_strength,
            "created_at": self.created_at,
            "last_recalled_at": self.last_recalled_at,
            "tags": self.tags,
            "kind": self.kind,
            "forgotten": self.forgotten,
            "metadata": self.metadata,
        }


# ============================================================================
# ContextSnapshot — 上下文快照
# ============================================================================


@dataclass
class ContextSnapshot:
    """
    上下文快照：包含当前工作记忆和长期记忆的完整状态。

    Attributes:
        working_memory: 工作记忆消息列表。
        long_term_memory: 长期记忆轨迹列表。
        namespace: 命名空间。
        timestamp: 快照时间戳。
    """

    working_memory: List[Message]
    long_term_memory: List[MemoryTrace]
    namespace: str = "default"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "working_memory": [m.to_dict() for m in self.working_memory],
            "long_term_memory": [t.to_dict() for t in self.long_term_memory],
            "namespace": self.namespace,
            "timestamp": self.timestamp,
        }


# ============================================================================
# 策略接口
# ============================================================================


class ShortTermStrategy(ABC):
    """短期记忆管理策略"""

    @abstractmethod
    def manage(self, messages: List[Message], config: Dict[str, Any]) -> List[Message]:
        """管理短期记忆，返回截断/压缩后的消息列表。"""
        ...


class LongTermStore(ABC):
    """长期记忆存储策略"""

    @abstractmethod
    def store(self, trace: MemoryTrace) -> None:
        """存储一条记忆轨迹。"""
        ...

    @abstractmethod
    def retrieve(
        self, query: str, namespace: str, top_k: int, **kwargs: Any
    ) -> List[MemoryTrace]:
        """检索记忆轨迹。"""
        ...

    @abstractmethod
    def get(self, trace_id: str) -> Optional[MemoryTrace]:
        """
        [FIX-L2] 按 trace_id 直接获取记忆，O(1)。

        用于 flush_strengthen 等场景，避免 O(N*M) 全量检索。
        """
        ...

    @abstractmethod
    def update_strength(self, trace_id: str, strength: float) -> None:
        """更新记忆强度。"""
        ...

    @abstractmethod
    def update_strengths_batch(self, updates: List[tuple]) -> None:
        """
        [FIX-L3] 批量更新记忆强度。

        Args:
            updates: List of (trace_id, new_strength) tuples。
        """
        ...

    @abstractmethod
    def archive(self, trace_id: str) -> None:
        """归档记忆。"""
        ...

    @abstractmethod
    def forget(self, trace_id: str) -> None:
        """删除记忆。"""
        ...


class CompressStrategy(ABC):
    """压缩策略"""

    @abstractmethod
    def compress(self, messages: List[Message], **kwargs: Any) -> List[Message]:
        """压缩短期记忆。"""
        ...


class DecayStrategy(ABC):
    """衰减策略"""

    @abstractmethod
    def compute_decay(self, trace: MemoryTrace, current_time: datetime) -> float:
        """
        计算衰减后的强度。

        重要：此方法只应基于 created_at 和 current_time 计算瞬态衰减值，
        不应累积写回，不应修改 trace 对象。
        """
        ...

    @abstractmethod
    def should_archive(self, trace: MemoryTrace, threshold: Optional[float] = None) -> bool:
        """判断是否应归档。"""
        ...


class RetrievalStrategy(ABC):
    """检索策略"""

    @abstractmethod
    def search(
        self, query: str, documents: List[MemoryTrace], top_k: int, **kwargs: Any
    ) -> List[MemoryTrace]:
        """执行检索。"""
        ...


class InjectionStrategy(ABC):
    """
    注入策略（L4）。

    [FIX-H2] 契约明确：inject 返回格式化后的记忆上下文文本，
    不负责替换 system prompt 中的占位符。
    占位符替换由 ContextManager.build_context 负责。
    """

    @abstractmethod
    def inject(
        self, traces: List[MemoryTrace], **kwargs: Any
    ) -> str:
        """
        生成记忆上下文字符串，用于注入 system prompt。

        Args:
            traces: 检索到的记忆列表。

        Returns:
            格式化后的记忆上下文文本。
        """
        ...


class ExtractionStrategy(ABC):
    """沉淀策略（L5）"""

    @abstractmethod
    def extract(
        self,
        messages: List[Message],
        existing_traces: List[MemoryTrace],
        **kwargs: Any,
    ) -> List[MemoryTrace]:
        """
        从对话中抽取可沉淀的记忆。

        此方法可能触发 LLM 调用，因此通过事件总线异步处理。

        Args:
            messages: 最近的对话消息。
            existing_traces: 已有的长期记忆，用于去重和合并。

        Returns:
            新生成的 MemoryTrace 列表。
        """
        ...


# ============================================================================
# ContextManager 抽象接口
# ============================================================================


class ContextManager(ABC):
    """
    Agent 上下文管理抽象接口。

    职责：
    - 工作记忆的增删改查
    - 长期记忆的存储与检索
    - 工作记忆压缩
    - 上下文快照
    - 分层上下文构建（system prompt + 工作记忆 + 长期记忆）

    可插拔设计：
    - 每种策略独立可替换。
    - 用户可替换整个 ContextManager 或单个策略。

    [FIX-D1] markdown 相关方法提升到抽象接口：
    - set_markdown_root(path): 设置 Markdown 真相源根目录
    - sync_from_markdown(): 从 Markdown 同步记忆，默认实现返回 0（降级语义）
    """

    # ----- 工作记忆 -----

    @abstractmethod
    def append(self, message: Message) -> None:
        """追加消息到工作记忆。"""
        ...

    @abstractmethod
    def get_working_memory(self) -> List[Message]:
        """获取当前工作记忆。"""
        ...

    @abstractmethod
    def clear_working_memory(self) -> None:
        """清空工作记忆。"""
        ...

    # ----- 工作记忆压缩 -----

    @abstractmethod
    def compress(self, strategy: Optional[str] = None) -> None:
        """压缩工作记忆。"""
        ...

    # ----- 长期记忆 -----

    @abstractmethod
    def store(self, trace: MemoryTrace) -> None:
        """存储到长期记忆。"""
        ...

    @abstractmethod
    def retrieve(
        self,
        query: str,
        namespace: Optional[str] = None,
        top_k: int = 5,
    ) -> List[MemoryTrace]:
        """从长期记忆检索。"""
        ...

    @abstractmethod
    def update_strength(self, trace_id: str, strength: float) -> None:
        """更新记忆强度。"""
        ...

    @abstractmethod
    def archive(self, trace_id: str) -> None:
        """归档记忆。"""
        ...

    @abstractmethod
    def forget(self, trace_id: str) -> None:
        """删除记忆。"""
        ...

    # ----- 上下文构建 -----

    @abstractmethod
    def build_context(
        self,
        system_prompt: str = "",
        namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        构建完整的 LLM 调用上下文。

        分层结构：
        1. system prompt（含注入的长期记忆）
        2. 工作记忆消息列表

        Args:
            system_prompt: 基础 system prompt 模板，可包含 {{memory_context}} 占位符。
            namespace: 长期记忆命名空间。

        Returns:
            LLM API 可用的消息列表。
        """
        ...

    # ----- 快照 -----

    @abstractmethod
    def snapshot(self) -> ContextSnapshot:
        """生成当前上下文快照。"""
        ...

    # ----- 待强化队列 -----

    @abstractmethod
    def enqueue_strengthen(self, trace_ids: List[str]) -> None:
        """将检索命中的 trace_id 加入待强化队列。"""
        ...

    @abstractmethod
    def flush_strengthen(self) -> None:
        """批量刷新强化队列。"""
        ...

    # ----- Markdown 真相源集成 [FIX-D1] -----

    @abstractmethod
    def set_markdown_root(self, path: str) -> None:
        """
        设置 Markdown 文件根目录（真相源）。

        默认实现标记为不支持，子类可覆盖。
        """
        ...

    @abstractmethod
    def sync_from_markdown(self) -> int:
        """
        从 Markdown 文件中同步记忆到索引。

        默认实现返回 0（降级语义：未配置真相源）。
        子类可覆盖实现实际同步逻辑。

        Returns:
            同步的记忆数量。
        """
        ...

    # ----- 记忆沉淀 [FIX-L6] -----

    @abstractmethod
    def extract_memories(self, namespace: Optional[str] = None) -> List[MemoryTrace]:
        """
        从工作记忆中抽取可沉淀的记忆。

        此方法由 AFTER_ITERATION 或 ON_EXIT_LOOP 钩子触发。

        Args:
            namespace: 命名空间。

        Returns:
            新生成的 MemoryTrace 列表。
        """
        ...

    # ----- 衰减应用 [FIX-H1/FIX-L1] -----

    @abstractmethod
    def apply_decay(self, namespace: Optional[str] = None) -> int:
        """
        对长期记忆应用衰减。

        衰减只在此方法中执行，不在 retrieve 中执行。
        此方法由定时或钩子触发（如 ON_EXIT_LOOP）。

        Returns:
            被归档的记忆数量。
        """
        ...