"""
ToolCallParser — LLM 响应中的工具调用解析。

核心抽象：
- ToolCallParser:  解析器策略接口
- ToolCall:        标准化工具调用结构
- ToolCallValidationResult: 校验结果

默认实现：
- OpenAIParser:    解析 OpenAI 原生 function calling
- AnthropicParser: 解析 Anthropic tool use
- TextFallbackParser: 解析文本格式回退（Action/Action Input）

错误处理策略：
- ErrorStrategy: 可配置的错误处理策略（严格/宽松/回调）
"""

from runtime.tool_call_parser.interface import (
    ToolCall,
    ToolCallParser,
    ToolCallValidationResult,
    ToolCallError,
)
from runtime.tool_call_parser.parsers import (
    OpenAIParser,
    AnthropicParser,
    TextFallbackParser,
    AutoDetectParser,
)
from runtime.tool_call_parser.error_strategy import (
    ErrorStrategy,
    StrictErrorStrategy,
    LenientErrorStrategy,
    TemplateErrorStrategy,
    CallbackErrorStrategy,
)
from runtime.tool_call_parser.parser_strategy import (
    ParserWithStrategy,
)

__all__ = [
    # 核心模型
    "ToolCall",
    "ToolCallParser",
    "ToolCallValidationResult",
    "ToolCallError",
    # 默认解析器
    "OpenAIParser",
    "AnthropicParser",
    "TextFallbackParser",
    "AutoDetectParser",
    # 策略集成 [FIX-D3]
    "ParserWithStrategy",
    # 错误处理策略
    "ErrorStrategy",
    "StrictErrorStrategy",
    "LenientErrorStrategy",
    "TemplateErrorStrategy",
    "CallbackErrorStrategy",
]
