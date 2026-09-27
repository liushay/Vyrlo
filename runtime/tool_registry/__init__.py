"""
ToolRegistry — 工具的注册、发现、schema 导出、执行。

核心抽象：
- ToolRegistry:  工具注册表的抽象接口
- Tool:          工具元数据模型
- ToolResult:    工具执行的标准化返回
- ToolSource:    工具来源抽象（装饰器函数 / MCP / 外部 API）
- function_tool: 装饰器，自动从函数签名生成 Tool

默认实现：InMemoryToolRegistry
"""

from runtime.tool_registry.interface import (
    Tool,
    ToolResult,
    ToolRegistry,
)
from runtime.tool_registry.in_memory import InMemoryToolRegistry
from runtime.tool_registry.tool_source import (
    ToolSource,
    LocalFunctionSource,
    MCPSource,
    ExternalAPISource,
)
from runtime.tool_registry.decorator import function_tool
from runtime.tool_registry.schema import (
    generate_schema_from_function,
    export_to_openai_format,
    export_to_anthropic_format,
    export_to_json_schema,
)

__all__ = [
    # 核心模型
    "Tool",
    "ToolResult",
    # 接口
    "ToolRegistry",
    # 默认实现
    "InMemoryToolRegistry",
    # 工具来源
    "ToolSource",
    "LocalFunctionSource",
    "MCPSource",
    "ExternalAPISource",
    # 装饰器
    "function_tool",
    # Schema 工具
    "generate_schema_from_function",
    "export_to_openai_format",
    "export_to_anthropic_format",
    "export_to_json_schema",
]