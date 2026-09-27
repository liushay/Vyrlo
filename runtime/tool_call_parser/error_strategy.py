"""
错误处理策略 — 解析/校验错误的可配置处理。

支持四种策略：
- StrictErrorStrategy:   任何错误立即抛出
- LenientErrorStrategy:  忽略错误，尽力继续
- TemplateErrorStrategy: 使用字符串模板格式化错误
- CallbackErrorStrategy: 使用回调函数自定义处理
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from runtime.tool_call_parser.interface import ToolCallError, ToolCallValidationResult


# ============================================================================
# ErrorStrategy 抽象接口
# ============================================================================


class ErrorStrategy(ABC):
    """
    错误处理策略抽象接口。

    每种策略决定了当解析或校验失败时，
    错误如何被报告、记录、或传递给 LLM。
    """

    @abstractmethod
    def handle_parse_error(self, error: ToolCallError) -> List[ToolCallError]:
        """
        处理解析错误。

        Args:
            error: 发生的解析错误。

        Returns:
            错误列表。返回空列表表示忽略此错误。
            Strict 模式下可抛出异常。
        """
        ...

    @abstractmethod
    def handle_validation_errors(
        self, errors: List[ToolCallError]
    ) -> List[ToolCallError]:
        """
        处理校验错误。

        Args:
            errors: 校验过程中收集的所有错误。

        Returns:
            过滤后的错误列表。
        """
        ...

    @abstractmethod
    def format_for_llm(self, errors: List[ToolCallError]) -> List[Dict[str, Any]]:
        """
        将错误列表格式化为 LLM 反馈消息。

        Args:
            errors: 要报告的错误列表。

        Returns:
            LLM 可理解的反馈消息列表。
        """
        ...


# ============================================================================
# StrictErrorStrategy — 严格模式
# ============================================================================


class StrictErrorStrategy(ErrorStrategy):
    """
    严格错误策略：任何解析/校验错误立即抛出异常。

    适用于开发调试阶段，快速暴露问题。
    """

    def __init__(self, error_type: Type[Exception] = ValueError) -> None:
        self._error_type = error_type

    def handle_parse_error(self, error: ToolCallError) -> List[ToolCallError]:
        raise self._error_type(f"[ParseError] {error.message}")

    def handle_validation_errors(self, errors: List[ToolCallError]) -> List[ToolCallError]:
        if errors:
            messages = "; ".join(e.message for e in errors)
            raise self._error_type(f"[ValidationError] {messages}")
        return errors

    def format_for_llm(self, errors: List[ToolCallError]) -> List[Dict[str, Any]]:
        # 严格模式下错误直接抛出，不会到达这里
        return [e.to_llm_feedback() for e in errors]


# ============================================================================
# LenientErrorStrategy — 宽松模式
# ============================================================================


class LenientErrorStrategy(ErrorStrategy):
    """
    宽松错误策略：忽略所有错误，尽力继续执行。

    适用于生产环境，优先保证流程不中断。
    """

    def handle_parse_error(self, error: ToolCallError) -> List[ToolCallError]:
        return []  # 忽略解析错误，不报告

    def handle_validation_errors(self, errors: List[ToolCallError]) -> List[ToolCallError]:
        return []  # 忽略校验错误，直接执行

    def format_for_llm(self, errors: List[ToolCallError]) -> List[Dict[str, Any]]:
        return []  # 不发送错误反馈给 LLM


# ============================================================================
# TemplateErrorStrategy — 模板模式
# ============================================================================


class TemplateErrorStrategy(ErrorStrategy):
    """
    模板错误策略：使用预定义的字符串模板格式化错误消息。

    模板变量：
    - {type}:  错误类型
    - {message}:  错误消息
    - {field}:  出错的字段名
    - {expected}:  期望的类型或值
    - {got}:  实际得到的值
    - {raw_input}:  原始输入

    用法：
        strategy = TemplateErrorStrategy(
            parse_template="工具调用解析失败: {message}",
            validation_template="参数 '{field}' 校验失败: 期望 {expected}，实际 {got}",
        )
    """

    def __init__(
        self,
        parse_template: str = "工具调用解析失败: {message}",
        validation_template: str = "参数校验失败: {message}",
    ) -> None:
        self.parse_template = parse_template
        self.validation_template = validation_template

    def handle_parse_error(self, error: ToolCallError) -> List[ToolCallError]:
        # 格式化后保留错误，以便传递给 LLM
        error.message = self._format(self.parse_template, error)
        return [error]

    def handle_validation_errors(self, errors: List[ToolCallError]) -> List[ToolCallError]:
        for error in errors:
            error.message = self._format(self.validation_template, error)
        return errors

    def _format(self, template: str, error: ToolCallError) -> str:
        """用错误字段填充模板。"""
        return template.format(
            type=error.type,
            message=error.message,
            field=error.field or "",
            expected=error.expected or "",
            got=error.got or "",
            raw_input=str(error.raw_input)[:200] if error.raw_input else "",
        )

    def format_for_llm(self, errors: List[ToolCallError]) -> List[Dict[str, Any]]:
        return [e.to_llm_feedback() for e in errors]


# ============================================================================
# CallbackErrorStrategy — 回调模式
# ============================================================================


class CallbackErrorStrategy(ErrorStrategy):
    """
    回调错误策略：使用用户提供的回调函数自定义错误处理。

    用法：
        def my_handler(errors: List[ToolCallError]) -> Dict[str, Any]:
            # 自定义逻辑：日志、告警、格式转换等
            return {"role": "user", "content": "请修正你的工具调用参数。"}

        strategy = CallbackErrorStrategy(on_error=my_handler)
    """

    def __init__(
        self,
        on_error: Callable[[List[ToolCallError]], Any],
    ) -> None:
        self._on_error = on_error

    def handle_parse_error(self, error: ToolCallError) -> List[ToolCallError]:
        self._on_error([error])
        return [error]

    def handle_validation_errors(self, errors: List[ToolCallError]) -> List[ToolCallError]:
        self._on_error(errors)
        return errors

    def format_for_llm(self, errors: List[ToolCallError]) -> List[Dict[str, Any]]:
        result = self._on_error(errors)
        if isinstance(result, dict):
            return [result]
        if isinstance(result, list):
            return result
        return [e.to_llm_feedback() for e in errors]


# ============================================================================
# 策略工厂：从配置字符串/对象创建策略
# ============================================================================


def create_error_strategy(config: Any) -> ErrorStrategy:
    """
    从配置创建错误处理策略。

    支持的配置格式：
    - True / "strict"  → StrictErrorStrategy
    - False / "lenient" → LenientErrorStrategy
    - str (模板)       → TemplateErrorStrategy(template=config)
    - callable         → CallbackErrorStrategy(on_error=config)
    - ErrorStrategy    → 直接返回
    - tuple(type, ...) → StrictErrorStrategy(error_type=type)
    """
    if isinstance(config, ErrorStrategy):
        return config

    if config is True or config == "strict":
        return StrictErrorStrategy()

    if config is False or config == "lenient":
        return LenientErrorStrategy()

    if isinstance(config, str):
        return TemplateErrorStrategy(parse_template=config, validation_template=config)

    if callable(config):
        return CallbackErrorStrategy(on_error=config)

    if isinstance(config, tuple):
        # (ExceptionType, ) → 严格模式 + 自定义异常类型
        exc_type = config[0] if config else ValueError
        return StrictErrorStrategy(error_type=exc_type)

    # 默认：严格模式
    return StrictErrorStrategy()