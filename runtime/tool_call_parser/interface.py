"""
ToolCallParser 核心接口定义。

包含：
- ToolCall:                    标准化工具调用结构
- ToolCallValidationResult:    校验结果
- ToolCallError:               解析/校验错误
- ToolCallParser:              解析器策略接口
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================================
# ToolCall — 标准化工具调用结构
# ============================================================================


@dataclass
class ToolCall:
    """
    标准化工具调用结构。

    无论 LLM 响应的原始格式是什么（OpenAI tool_calls、
    Anthropic content block、或文本 Action/Input），
    解析后统一为此结构。

    Attributes:
        id:         工具调用唯一标识（OpenAI/Anthropic 有，文本格式可为 None）。
        name:       工具名称。
        args:       工具参数字典。
        raw:        原始响应片段，用于调试和错误恢复。
        confidence: 解析置信度（0.0-1.0），文本回退时较低。
    """

    id: Optional[str]
    name: str
    args: Dict[str, Any]
    raw: Any = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典，供写入 Context.set_tool_calls()。

        键名与 Agent Loop 工具执行消费约定一致：使用 ``arguments``
        （而非 ``args``），与 ``Runtime._parse_tool_calls`` 的输出保持一致。
        """
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.args,
        }


# ============================================================================
# ToolCallValidationResult
# ============================================================================


@dataclass
class ToolCallValidationResult:
    """
    工具调用校验结果。

    Attributes:
        is_valid:   是否通过校验。
        errors:     错误列表，每项包含 field、message、expected、got。
        warnings:   警告列表。
    """

    is_valid: bool
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def success(warnings: Optional[List[Dict[str, Any]]] = None) -> "ToolCallValidationResult":
        return ToolCallValidationResult(is_valid=True, warnings=warnings or [])

    @staticmethod
    def failure(
        errors: List[Dict[str, Any]],
        warnings: Optional[List[Dict[str, Any]]] = None,
    ) -> "ToolCallValidationResult":
        return ToolCallValidationResult(is_valid=False, errors=errors, warnings=warnings or [])


# ============================================================================
# ToolCallError
# ============================================================================


@dataclass
class ToolCallError:
    """
    解析/校验阶段产生的错误。

    设计为可序列化，方便作为上下文传递给 LLM 以便修复。

    Attributes:
        type:       错误类型（"parse" | "validation" | "missing_tool"）。
        message:    人类可读的错误消息。
        raw_input:  触发错误的原始输入片段。
        field:      出错的字段名（校验错误时）。
        expected:   期望的类型或值。
        got:        实际得到的值。
    """

    type: str  # "parse" | "validation" | "missing_tool"
    message: str
    raw_input: Optional[Any] = None
    field: Optional[str] = None
    expected: Optional[Any] = None
    got: Optional[Any] = None

    def to_llm_feedback(self) -> Dict[str, Any]:
        """格式化为 LLM 可理解的反馈消息。"""
        return {
            "role": "tool",
            "content": self.message,
            "error_type": self.type,
            "field": self.field,
            "expected": str(self.expected) if self.expected else None,
            "got": str(self.got) if self.got else None,
        }


# ============================================================================
# ToolCallParser 抽象接口
# ============================================================================


class ToolCallParser(ABC):
    """
    工具调用解析器策略接口。

    职责：
    1. 解析 LLM 原始响应 → List[ToolCall]
    2. 校验参数是否符合 schema → ToolCallValidationResult
    3. 尝试修复不符合 schema 的参数 → Optional[ToolCall]
    4. 把错误格式化成 LLM 可理解的反馈 → Dict

    可插拔设计：
    - 每种响应格式对应一个解析器实现。
    - 用户可注册自定义解析策略。
    - 解析器通过 format 参数选择。
    """

    @abstractmethod
    def parse(self, response: Any, format: Optional[str] = None) -> List[ToolCall]:
        """
        解析 LLM 原始响应，返回标准化 ToolCall 列表。

        Args:
            response: LLM 响应（dict 或 str，取决于格式）。
            format:   显式指定格式，None 时自动检测。

        Returns:
            ToolCall 列表。若无工具调用则返回空列表。
        """
        ...

    @abstractmethod
    def validate(
        self, tool_call: ToolCall, schema: Dict[str, Any]
    ) -> ToolCallValidationResult:
        """
        校验工具调用参数是否符合 JSON Schema。

        Args:
            tool_call: 待校验的 ToolCall。
            schema:    JSON Schema 定义（params_schema 的 parameters 部分）。

        Returns:
            ToolCallValidationResult，包含是否通过和错误详情。
        """
        ...

    @abstractmethod
    def repair(
        self, tool_call: ToolCall, schema: Dict[str, Any], error: ToolCallError
    ) -> Optional[ToolCall]:
        """
        尝试修复不符合 schema 的参数。

        修复策略示例：
        - 类型转换（"123" → 123）
        - 补全缺失的必填字段（使用 default）
        - 修剪多余的未知字段

        Args:
            tool_call: 待修复的 ToolCall。
            schema:    目标 JSON Schema。
            error:     触发的错误。

        Returns:
            修复后的 ToolCall，若无法修复则返回 None。
        """
        ...

    @abstractmethod
    def format_error(self, error: ToolCallError) -> Dict[str, Any]:
        """
        将解析/校验错误格式化为 LLM 可理解的反馈。

        Args:
            error: ToolCallError 实例。

        Returns:
            格式化的字典，可直接作为 LLM 消息的一部分。
        """
        ...

    @property
    @abstractmethod
    def supported_format(self) -> str:
        """返回此解析器支持的格式标识（如 "openai"、"anthropic"、"text"）。"""
        ...