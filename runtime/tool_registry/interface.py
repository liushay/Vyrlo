"""
ToolRegistry 核心接口定义。

包含：
- Tool:           工具元数据模型（name、description、params_schema 等）
- ToolResult:     工具执行的标准化返回（content、sources、metadata、confidence）
- ToolRegistry:   工具注册表抽象接口
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ============================================================================
# Tool 元数据模型
# ============================================================================


@dataclass
class Tool:
    """
    工具的完整元数据描述。

    参考 OpenAI Agents SDK 的 FunctionTool 设计，
    支持从函数签名自动生成 schema，也支持手动构造。

    Attributes:
        name:               工具名称，全局唯一标识。
        description:        工具描述，供 LLM 理解用途。
        params_schema:      参数 JSON Schema，描述工具接受的参数格式。
        returns_schema:     返回值 JSON Schema（可选），描述工具返回值的格式。
        fn:                 实际执行函数。若不提供则工具不可执行（如外部来源）。
        requires_approval:  是否需要用户审批（默认 False）。
        is_idempotent:      是否幂等（默认 True），用于重试判定。
        timeout:            超时时间（秒），None 表示无超时限制。
        source:             工具来源标识（如 "local"、"mcp"、"external"）。
        metadata:           额外的自由格式元数据。
    """

    name: str
    description: str
    params_schema: Dict[str, Any]
    returns_schema: Optional[Dict[str, Any]] = None
    fn: Optional[Callable[..., Any]] = None
    requires_approval: bool = False
    is_idempotent: bool = True
    timeout: Optional[float] = None
    source: str = "local"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # [FIX-D2] executable 属性：指示工具是否可执行
    @property
    def executable(self) -> bool:
        """工具是否可执行（fn 不为 None）。"""
        return self.fn is not None

    def __repr__(self) -> str:
        return f"<Tool name={self.name!r} source={self.source!r}>"

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典，供序列化使用。"""
        return {
            "name": self.name,
            "description": self.description,
            "params_schema": self.params_schema,
            "returns_schema": self.returns_schema,
            "requires_approval": self.requires_approval,
            "is_idempotent": self.is_idempotent,
            "timeout": self.timeout,
            "source": self.source,
            "metadata": self.metadata,
        }


# ============================================================================
# ToolResult 标准化返回
# ============================================================================


@dataclass
class ToolResult:
    """
    工具执行的标准化返回结构。

    设计原则：
    - 工具异常不传播到 Loop，被包装为 ToolResult（error=True）。
    - content 保留原始返回值，sources 提供引用溯源。
    - confidence 用于下游归因与置信度评估。

    Attributes:
        content:    工具执行的原始结果（成功时）或错误信息（失败时）。
        error:      是否执行出错。
        sources:    引用的来源列表，如 URL、文档 ID 等。
        metadata:   元数据字典，如执行耗时、token 消耗等。
        confidence: 置信度（0.0-1.0），用于下游归因。
    """

    content: Any = None
    error: bool = False
    sources: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    @staticmethod
    def from_result(content: Any, **kwargs: Any) -> "ToolResult":
        """从成功结果构造 ToolResult。"""
        return ToolResult(content=content, error=False, **kwargs)

    @staticmethod
    def from_error(content: Any, **kwargs: Any) -> "ToolResult":
        """从错误结果构造 ToolResult，confidence 默认置 0。"""
        return ToolResult(content=content, error=True, confidence=0.0, **kwargs)


# ============================================================================
# ToolRegistry 抽象接口
# ============================================================================


class ToolRegistry(ABC):
    """
    工具注册表的抽象接口。

    职责：工具的注册、发现、schema 导出、执行。

    可插拔设计：
    - 用户可以实现此接口替换默认的 InMemoryToolRegistry。
    - 只需满足"注册 / 查找 / 导出 schema / 执行"的契约。

    与 Agent Loop 的交互契约：
    - Loop 不直接调用 ToolRegistry。
    - 中间件在 BEFORE_TOOL 钩子中通过 registry.get() 获取元数据。
    - 工具执行通过 registry.execute() 完成。
    """

    @abstractmethod
    def register(self, tool: Tool) -> None:
        """
        注册工具。

        支持三种来源：
        1. 装饰器自动转换的 Python 函数 → LocalFunctionSource
        2. 手动构造的 Tool 对象 → 直接调用
        3. 外部来源（MCP Server 的工具列表）→ MCPSource / ExternalAPISource

        Args:
            tool: 要注册的 Tool 实例。

        Raises:
            ValueError: 如果已存在同名工具。
        """
        ...

    @abstractmethod
    def unregister(self, name: str) -> Optional[Tool]:
        """
        注销工具。

        Args:
            name: 要注销的工具名称。

        Returns:
            被注销的 Tool 实例，若不存在则返回 None。
        """
        ...

    @abstractmethod
    def get(self, name: str) -> Optional[Tool]:
        """
        按名称获取工具。

        Args:
            name: 工具名称。

        Returns:
            Tool 实例，若不存在则返回 None。
        """
        ...

    @abstractmethod
    def list(self) -> List[Tool]:
        """
        列出所有已注册工具。

        Returns:
            已注册工具列表（无特定顺序保证）。
        """
        ...

    @abstractmethod
    def export_schemas(self, format: str = "openai") -> List[Dict[str, Any]]:
        """
        导出工具 schema 为 LLM function calling 格式。

        Args:
            format: 目标格式，支持 "openai"、"anthropic"、"json_schema"。

        Returns:
            对应格式的 schema 列表。
        """
        ...

    @abstractmethod
    def execute(
        self, name: str, args: Dict[str, Any], sandbox: Optional["SandboxHandle"] = None
    ) -> ToolResult:
        """
        执行工具，带超时隔离和错误边界。

        执行隔离保证：
        - 超时控制：通过 signal 或线程实现超时中断。
        - 错误边界：工具异常不传播到调用方，包装为 ToolResult(error=True)。
        - 结构化返回：统一返回 ToolResult。

        Args:
            name:    工具名称。
            args:    工具参数字典。
            sandbox: 可选的沙箱句柄。为 None 时走默认执行路径；
                     非 None 时委托给沙箱提供者执行。

        Returns:
            ToolResult 实例。工具执行异常时 error=True。
        """
        ...
