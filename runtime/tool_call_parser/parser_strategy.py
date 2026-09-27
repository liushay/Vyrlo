"""
[FIX-D3] ParserWithStrategy — ErrorStrategy 与 ToolCallParser 的集成桥梁。

将 ErrorStrategy 嵌入到 ToolCallParser 的完整解析链路中：
解析 → 错误处理 → 校验 → 错误处理 → 修复 → 二次校验。

Parser 负责格式解析，ErrorStrategy 负责错误的策略处置，
二者正交组合，各自接口不互相污染。

用法::

    wrapper = ParserWithStrategy(
        parser=AutoDetectParser(),
        strategy=TemplateErrorStrategy(),
    )
    tool_calls, errors = wrapper.parse_with_strategy(response, schemas)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from runtime.tool_call_parser.interface import (
    ToolCall,
    ToolCallParser,
    ToolCallError,
)
from runtime.tool_call_parser.error_strategy import (
    ErrorStrategy,
    StrictErrorStrategy,
)


class ParserWithStrategy:
    """ErrorStrategy 与 ToolCallParser 的集成包装器。

    提供完整的"解析 → 校验 → 修复 → 错误反馈"链路，
    使中间件可以一次调用完成工具调用提取的全流程。

    Attributes:
        parser:   底层解析器。
        strategy: 错误处理策略。
    """

    def __init__(
        self,
        parser: ToolCallParser,
        strategy: Optional[ErrorStrategy] = None,
    ) -> None:
        """初始化。

        Args:
            parser:   任意 ToolCallParser 实现。
            strategy: 错误处理策略；默认 StrictErrorStrategy。
        """
        self._parser = parser
        self._strategy: ErrorStrategy = (
            strategy if isinstance(strategy, ErrorStrategy)
            else StrictErrorStrategy()
        )

    # ---- 属性 ----

    @property
    def parser(self) -> ToolCallParser:
        """底层解析器。"""
        return self._parser

    @property
    def strategy(self) -> ErrorStrategy:
        """错误处理策略。"""
        return self._strategy

    # ---- 核心方法 ----

    def parse_with_strategy(
        self,
        response: Any,
        schemas: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> tuple:
        """完整解析链路。

        步骤：
        1. parser.parse()  — 从 LLM 响应提取 tool_calls
        2. 解析异常 → strategy.handle_parse_error()
        3. 逐个 parser.validate()  — 校验参数
        4. 校验失败 → parser.repair()*→ 二次校验
        5. 所有错误 → strategy.handle_validation_errors()
        6. strategy.format_for_llm()  — 生成 LLM 反馈

        Args:
            response: LLM 原始响应（dict / list / str）。
            schemas:  工具名 → params_schema 映射；不提供则不做校验。

        Returns:
            (valid_calls: List[ToolCall], errors_for_llm: List[Dict])
        """
        schemas = schemas or {}
        all_errors: List[ToolCallError] = []

        # ── 步骤 1：解析 ──
        try:
            tool_calls = self._parser.parse(response)
        except Exception as exc:
            parse_error = ToolCallError(
                type="parse",
                message=str(exc),
                raw_input=str(response)[:500],
            )
            handled = self._strategy.handle_parse_error(parse_error)
            all_errors.extend(handled)
            return [], self._strategy.format_for_llm(all_errors)

        if not tool_calls:
            return [], []

        # ── 步骤 2-4：校验 + 修复 ──
        valid_calls: List[ToolCall] = []
        for tc in tool_calls:
            schema = schemas.get(tc.name)
            if not schema:
                valid_calls.append(tc)
                continue

            validation = self._parser.validate(tc, schema)
            if validation.is_valid:  # [FIX-E1]
                valid_calls.append(tc)
                continue

            # 修复
            repaired = self._try_repair(tc, schema, validation.errors)
            if repaired is not None:
                re_validation = self._parser.validate(repaired, schema)
                if re_validation.is_valid:  # [FIX-E1]
                    valid_calls.append(repaired)
                    continue
                # 修复后仍无效
                for err_dict in re_validation.errors:
                    all_errors.append(ToolCallError(
                        type="validation",
                        message="修复后仍无效: %s" % err_dict.get("message", ""),
                        field=err_dict.get("field"),
                        expected=err_dict.get("expected"),
                        got=err_dict.get("got"),
                        raw_input=tc.raw,
                    ))
            else:
                for err_dict in validation.errors:
                    all_errors.append(ToolCallError(
                        type="validation",
                        message=err_dict.get("message", ""),
                        field=err_dict.get("field"),
                        expected=err_dict.get("expected"),
                        got=err_dict.get("got"),
                        raw_input=tc.raw,
                    ))

        # ── 步骤 5-6：过滤 + 格式化 ──
        filtered = self._strategy.handle_validation_errors(all_errors)
        errors_for_llm = self._strategy.format_for_llm(filtered)
        return valid_calls, errors_for_llm

    def parse(self, response: Any, format: Optional[str] = None) -> List[ToolCall]:
        """委托给底层 parser（跳过策略链路）。"""
        return self._parser.parse(response, format)

    # ---- 内部辅助 ----

    def _try_repair(
        self,
        tc: ToolCall,
        schema: Dict[str, Any],
        errors: List[Dict[str, Any]],
    ) -> Optional[ToolCall]:
        """逐个错误尝试修复，首次成功即返回。"""
        for error_dict in errors:
            tc_error = ToolCallError(
                type="validation",
                message=error_dict.get("message", ""),
                field=error_dict.get("field"),
                expected=error_dict.get("expected"),
                got=error_dict.get("got"),
                raw_input=tc.raw,
            )
            repaired = self._parser.repair(tc, schema, tc_error)
            if repaired is not None:
                return repaired
        return None