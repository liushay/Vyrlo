"""
ToolCallParser 具体实现。

包含四个解析器：
- OpenAIParser:      解析 OpenAI 原生 function calling 格式
- AnthropicParser:   解析 Anthropic tool use 格式
- TextFallbackParser: 解析文本格式回退（Action/Action Input）
- AutoDetectParser:  自动检测格式并委托给合适的解析器
"""

from __future__ import annotations

import json
import re
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from runtime.tool_call_parser.interface import (
    ToolCall,
    ToolCallParser,
    ToolCallError,
    ToolCallValidationResult,
)


# ============================================================================
# 基础校验逻辑（所有解析器共用）
# ============================================================================


def _normalize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    [FIX-L8] 将多种 schema 格式归一化为标准 JSON Schema。

    支持的输入格式：
    1. 标准 JSON Schema: {"type": "object", "properties": {...}, "required": [...]}
    2. params_schema 格式: {"params": {...}} 或 {"arguments": {...}}
    3. OpenAI function parameters: {"type": "object", "properties": {...}}

    归一化后始终包含 type、properties、required 三个键。

    [FIX-P0-4] 修复了 type 存在但 properties 缺失的边界情况。
    例如 input {"type": "object", "required": ["city"]} 现在被修复为
    {"type": "object", "properties": {}, "required": ["city"]}，
    不再落到回退分支返回空 schema。
    """
    if not schema:
        return {"type": "object", "properties": {}, "required": []}

    # [FIX-P0-4] type 存在时立即建立标准外壳，补全缺失的 properties / required
    if "type" in schema:
        result: Dict[str, Any] = {"type": schema["type"]}
        result["properties"] = dict(schema.get("properties", {}))
        result["required"] = list(schema.get("required", []))
        # 保留其他元信息（如 description、additionalProperties 等）
        for k, v in schema.items():
            if k not in ("type", "properties", "required"):
                result.setdefault(k, v)
        return result

    # 处理 params_schema 格式
    if "params" in schema:
        inner = schema["params"]
        if isinstance(inner, dict):
            return _normalize_schema(inner)
        # 如果是列表，转为 properties
        props: Dict[str, Any] = {}
        required_fields: List[str] = []
        if isinstance(inner, list):
            for param in inner:
                if isinstance(param, dict) and "name" in param:
                    pname = param["name"]
                    props[pname] = {
                        "type": param.get("type", "string"),
                        "description": param.get("description", ""),
                    }
                    if param.get("required", False):
                        required_fields.append(pname)
        return {"type": "object", "properties": props, "required": required_fields}

    # 处理 arguments 格式
    if "arguments" in schema:
        inner = schema["arguments"]
        if isinstance(inner, dict) and "type" in inner:
            return _normalize_schema(inner)

    # [FIX-P0-4] 处理 Tool params_schema 格式：外层是 {"name": ..., "description": ..., "parameters": {...}}
    # 实际 schema 定义在内层 parameters 中，递归提取
    if "parameters" in schema:
        inner = schema["parameters"]
        if isinstance(inner, dict) and "type" in inner:
            return _normalize_schema(inner)

    # 最后回退：尝试将外层本身当作 properties
    props = {}
    for key, value in schema.items():
        if isinstance(value, dict) and "type" in value:
            props[key] = value
    if props:
        return {"type": "object", "properties": props, "required": list(props.keys())}

    # 完全无法归一化，返回空 schema
    return {"type": "object", "properties": {}, "required": []}


def _validate_against_schema(
    args: Dict[str, Any], schema: Dict[str, Any]
) -> ToolCallValidationResult:
    """
    将参数字典与 JSON Schema 进行校验。

    [FIX-L8] 先归一化 schema，再进行校验。

    校验内容：
    1. 必填字段是否存在
    2. 字段类型是否匹配
    3. 是否有未知字段（仅产生警告）

    Args:
        args:   工具参数字典。
        schema: JSON Schema 对象（多种格式均接受）。

    Returns:
        ToolCallValidationResult。
    """
    # [FIX-L8] 归一化
    schema = _normalize_schema(schema)

    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    properties = schema.get("properties", {})
    required = schema.get("required", [])
    allowed_fields = set(properties.keys()) if properties else set()

    # 检查必填字段
    for field in required:
        if field not in args:
            errors.append({
                "field": field,
                "message": f"缺少必填字段 '{field}'",
                "expected": "required",
                "got": "missing",
            })

    # 检查字段类型
    for field, value in args.items():
        if field not in properties:
            warnings.append({
                "field": field,
                "message": f"未知字段 '{field}'，将保留但可能被忽略",
            })
            continue

        field_schema = properties[field]
        expected_type = field_schema.get("type")
        if expected_type and not _match_type(value, expected_type):
            errors.append({
                "field": field,
                "message": (
                    f"字段 '{field}' 类型不匹配: "
                    f"期望 {expected_type}，实际 {type(value).__name__}"
                ),
                "expected": expected_type,
                "got": type(value).__name__,
            })

    if errors:
        return ToolCallValidationResult.failure(errors, warnings)

    return ToolCallValidationResult.success(warnings)


def _match_type(value: Any, expected_type: str) -> bool:
    """检查 Python 值是否匹配 JSON Schema 类型。"""
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }
    expected = type_map.get(expected_type)
    if expected is None:
        return True  # 未知类型，放行
    return isinstance(value, expected)


def _basic_repair(
    args: Dict[str, Any],
    schema: Dict[str, Any],
    error: ToolCallError,
) -> Optional[Dict[str, Any]]:
    """
    基础修复逻辑：

    1. 类型转换：尝试将值转换为期望类型
    2. 补全缺失字段：如果 schema 中有 default，则使用
    3. 去除未知字段：移除不在 properties 中的字段
    """
    repaired = dict(args)
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    # 补全缺失字段
    for field in required:
        if field not in repaired:
            field_schema = properties.get(field, {})
            if "default" in field_schema:
                repaired[field] = field_schema["default"]
            else:
                return None  # 无法补全，放弃修复

    # 类型转换
    for field, value in list(repaired.items()):
        if field not in properties:
            continue
        expected_type = properties[field].get("type")
        if expected_type and not _match_type(value, expected_type):
            converted = _try_convert(value, expected_type)
            if converted is not None:
                repaired[field] = converted
            else:
                return None  # 无法转换，放弃修复

    return repaired


def _try_convert(value: Any, target_type: str) -> Any:
    """尝试将值转换为目标类型。"""
    type_converters = {
        "string": str,
        "integer": lambda v: int(float(v)),
        "number": float,
        "boolean": lambda v: str(v).lower() in ("true", "1", "yes"),
    }
    converter = type_converters.get(target_type)
    if converter is None:
        return None
    try:
        if target_type == "integer":
            return converter(value)
        if target_type == "boolean" and isinstance(value, str):
            return converter(value)
        return converter(value)
    except (ValueError, TypeError):
        return None


# ============================================================================
# OpenAIParser
# ============================================================================


class OpenAIParser(ToolCallParser):
    """
    解析 OpenAI 原生 function calling 格式。

    输入格式（OpenAI API 响应）：
    {
        "choices": [{
            "message": {
                "tool_calls": [
                    {
                        "id": "call_xxx",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Beijing\"}"
                        }
                    }
                ]
            }
        }]
    }
    """

    @property
    def supported_format(self) -> str:
        return "openai"

    def parse(self, response: Any, format: Optional[str] = None) -> List[ToolCall]:
        """
        从 OpenAI 格式响应中提取工具调用。

        支持两种输入格式：
        1. 完整的 API 响应 dict（包含 choices → message → tool_calls）
        2. 直接传入 message dict（包含 tool_calls 字段）
        """
        if isinstance(response, str):
            response = json.loads(response)

        # 尝试从完整响应中提取 message
        message = response
        if isinstance(response, dict):
            if "choices" in response:
                message = response["choices"][0].get("message", {})
            elif "message" in response:
                message = response["message"]

        if not isinstance(message, dict):
            return []

        tool_calls = message.get("tool_calls", [])
        if not tool_calls:
            return []

        result: List[ToolCall] = []
        for tc in tool_calls:
            func_info = tc.get("function", {})
            name = func_info.get("name", "")
            args_str = func_info.get("arguments", "{}")

            # 解析 JSON 参数
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = {}

            result.append(
                ToolCall(
                    id=tc.get("id"),
                    name=name,
                    args=args,
                    raw=tc,
                    confidence=1.0,
                )
            )

        return result

    def validate(self, tool_call: ToolCall, schema: Dict[str, Any]) -> ToolCallValidationResult:
        return _validate_against_schema(tool_call.args, schema)

    def repair(
        self, tool_call: ToolCall, schema: Dict[str, Any], error: ToolCallError
    ) -> Optional[ToolCall]:
        repaired_args = _basic_repair(tool_call.args, schema, error)
        if repaired_args is None:
            return None
        return ToolCall(
            id=tool_call.id,
            name=tool_call.name,
            args=repaired_args,
            raw=tool_call.raw,
            confidence=tool_call.confidence * 0.9,  # 降低置信度
        )

    def format_error(self, error: ToolCallError) -> Dict[str, Any]:
        return error.to_llm_feedback()


# ============================================================================
# AnthropicParser
# ============================================================================


class AnthropicParser(ToolCallParser):
    """
    解析 Anthropic tool use 格式。

    输入格式一（Anthropic API 原生响应）：
    {
        "content": [
            {"type": "text", "text": "..."},
            {
                "type": "tool_use",
                "id": "toolu_xxx",
                "name": "get_weather",
                "input": {"city": "Beijing"}
            }
        ]
    }

    [FIX-L9] 输入格式二（AnthropicProvider 扁平 tool_calls 列表）：
    [
        {
            "id": "toolu_xxx",
            "type": "tool_use",
            "name": "get_weather",
            "input": {"city": "Beijing"}
        }
    ]

    输入格式三（从 Context.tool_calls 读取的列表）：
    直接传入 List[dict]。
    """

    @property
    def supported_format(self) -> str:
        return "anthropic"

    def parse(self, response: Any, format: Optional[str] = None) -> List[ToolCall]:
        """从 Anthropic 格式响应中提取 tool_use 块。"""
        if isinstance(response, str):
            response = json.loads(response)

        # [FIX-L9] 格式一：Anthropic API 原生响应（dict，含 content 数组）
        if isinstance(response, dict):
            content = response.get("content", [])
            if isinstance(content, dict):
                content = [content]

            result: List[ToolCall] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    result.append(
                        ToolCall(
                            id=block.get("id"),
                            name=block.get("name", ""),
                            args=block.get("input", {}),
                            raw=block,
                            confidence=1.0,
                        )
                    )
            if result:
                return result

            # 没有 content 数组，尝试当作扁平 tool_calls dict 处理
            if "type" in response and response.get("type") == "tool_use":
                return [
                    ToolCall(
                        id=response.get("id"),
                        name=response.get("name", ""),
                        args=response.get("input", {}),
                        raw=response,
                        confidence=1.0,
                    )
                ]

            # 尝试 tool_calls 字段（可能在 Context 中）
            tool_calls = response.get("tool_calls", [])
            if isinstance(tool_calls, list):
                return self._parse_list(tool_calls)

        # [FIX-L9] 格式二/三：直接传入 tool_calls 列表
        if isinstance(response, list):
            return self._parse_list(response)

        return []

    def _parse_list(self, tool_calls: List[Any]) -> List[ToolCall]:
        """解析 AnthropicProvider 扁平的 tool_calls 列表。"""
        result: List[ToolCall] = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            # AnthropicProvider 格式: {"id": ..., "type": "tool_use", "name": ..., "input": ...}
            if tc.get("type") == "tool_use":
                result.append(
                    ToolCall(
                        id=tc.get("id"),
                        name=tc.get("name", ""),
                        args=tc.get("input", {}),
                        raw=tc,
                        confidence=1.0,
                    )
                )
            # OpenAI 格式兼容: {"id": ..., "type": "function", "function": {"name": ..., "arguments": ...}}
            elif tc.get("type") == "function":
                func = tc.get("function", {})
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {}
                result.append(
                    ToolCall(
                        id=tc.get("id"),
                        name=func.get("name", ""),
                        args=args,
                        raw=tc,
                        confidence=1.0,
                    )
                )
        return result

    def validate(self, tool_call: ToolCall, schema: Dict[str, Any]) -> ToolCallValidationResult:
        return _validate_against_schema(tool_call.args, schema)

    def repair(
        self, tool_call: ToolCall, schema: Dict[str, Any], error: ToolCallError
    ) -> Optional[ToolCall]:
        repaired_args = _basic_repair(tool_call.args, schema, error)
        if repaired_args is None:
            return None
        return ToolCall(
            id=tool_call.id,
            name=tool_call.name,
            args=repaired_args,
            raw=tool_call.raw,
            confidence=tool_call.confidence * 0.9,
        )

    def format_error(self, error: ToolCallError) -> Dict[str, Any]:
        return error.to_llm_feedback()


# ============================================================================
# TextFallbackParser
# ============================================================================


class TextFallbackParser(ToolCallParser):
    """
    解析文本格式回退（ReAct 风格 Action/Action Input）。

    输入格式：
        Action: get_weather
        Action Input: {"city": "Beijing"}

    也支持 JSON 代码块格式：
        ```json
        {"name": "get_weather", "arguments": {"city": "Beijing"}}
        ```
    """

    # ReAct 风格的 Action/Action Input 模式
    _ACTION_PATTERN = re.compile(
        r"Action\s*:\s*(.+?)\s*\n\s*Action\s*Input\s*:\s*(.+?)(?:\n|$)",
        re.DOTALL,
    )

    # JSON 代码块模式
    _JSON_BLOCK_PATTERN = re.compile(
        r"```(?:json)?\s*\n?(.*?)\n?```",
        re.DOTALL,
    )

    # 独立JSON对象模式: {"name": "xxx", "arguments": {...}}
    _JSON_TOOL_PATTERN = re.compile(
        r'\{\s*"name"\s*:\s*"(.*?)"\s*,\s*"arguments?"\s*:\s*(\{.*?\})\s*\}',
        re.DOTALL,
    )

    @property
    def supported_format(self) -> str:
        return "text"

    def parse(self, response: Any, format: Optional[str] = None) -> List[ToolCall]:
        """
        从纯文本中提取工具调用。

        策略优先级：
        1. 匹配 Action: / Action Input: 模式
        2. 匹配 JSON 代码块中的工具调用
        3. 匹配内联 JSON 对象 {"name": ..., "arguments": ...}
        """
        text = response
        if isinstance(response, dict):
            text = json.dumps(response)
        elif not isinstance(response, str):
            text = str(response)

        result: List[ToolCall] = []

        # 策略 1: Action / Action Input
        action_matches = self._ACTION_PATTERN.findall(text)
        for name, args_str in action_matches:
            args = self._parse_args(args_str.strip())
            result.append(
                ToolCall(
                    id=None,
                    name=name.strip(),
                    args=args,
                    raw={"action": name.strip(), "action_input": args_str.strip()},
                    confidence=0.7,  # 文本解析置信度较低
                )
            )

        if result:
            return result

        # 策略 2: JSON 代码块
        json_blocks = self._JSON_BLOCK_PATTERN.findall(text)
        for block in json_blocks:
            try:
                parsed = json.loads(block.strip())
                if isinstance(parsed, dict):
                    if "name" in parsed and ("arguments" in parsed or "args" in parsed):
                        args = parsed.get("arguments") or parsed.get("args", {})
                        result.append(
                            ToolCall(
                                id=parsed.get("id"),
                                name=parsed["name"],
                                args=args,
                                raw=parsed,
                                confidence=0.85,
                            )
                        )
                    elif "tool_calls" in parsed:
                        # 嵌套的 OpenAI 格式
                        for tc in parsed["tool_calls"]:
                            func = tc.get("function", {})
                            args_val = func.get("arguments", "{}")
                            if isinstance(args_val, str):
                                try:
                                    args_val = json.loads(args_val)
                                except json.JSONDecodeError:
                                    args_val = {}
                            result.append(
                                ToolCall(
                                    id=tc.get("id"),
                                    name=func.get("name", ""),
                                    args=args_val,
                                    raw=tc,
                                    confidence=0.85,
                                )
                            )
            except (json.JSONDecodeError, TypeError):
                continue

        if result:
            return result

        # 策略 3: 内联 JSON 对象
        json_matches = self._JSON_TOOL_PATTERN.findall(text)
        for name, args_str in json_matches:
            args = self._parse_args(args_str)
            result.append(
                ToolCall(
                    id=None,
                    name=name.strip(),
                    args=args,
                    raw={"json_match": name, "args_str": args_str},
                    confidence=0.6,
                )
            )

        return result

    def _parse_args(self, args_str: str) -> Dict[str, Any]:
        """解析参数字符串为字典。"""
        args_str = args_str.strip()
        # 尝试 JSON
        try:
            return json.loads(args_str)
        except (json.JSONDecodeError, TypeError):
            pass
        # 回退：当作 key=value 对
        if not args_str:
            return {}
        return {"raw_input": args_str}

    def validate(self, tool_call: ToolCall, schema: Dict[str, Any]) -> ToolCallValidationResult:
        return _validate_against_schema(tool_call.args, schema)

    def repair(
        self, tool_call: ToolCall, schema: Dict[str, Any], error: ToolCallError
    ) -> Optional[ToolCall]:
        repaired_args = _basic_repair(tool_call.args, schema, error)
        if repaired_args is None:
            return None
        return ToolCall(
            id=tool_call.id,
            name=tool_call.name,
            args=repaired_args,
            raw=tool_call.raw,
            confidence=tool_call.confidence * 0.85,
        )

    def format_error(self, error: ToolCallError) -> Dict[str, Any]:
        return error.to_llm_feedback()


# ============================================================================
# AutoDetectParser
# ============================================================================


class AutoDetectParser(ToolCallParser):
    """
    自动检测格式并委托给合适的解析器。

    检测逻辑：
    1. 如果是 dict 且包含 "choices" → OpenAI
    2. 如果是 dict 且包含 "content" 且其中有 "type": "tool_use" → Anthropic
    3. 否则 → TextFallback

    用户也可注册自定义格式检测器。
    """

    def __init__(self) -> None:
        self._parsers: Dict[str, ToolCallParser] = {
            "openai": OpenAIParser(),
            "anthropic": AnthropicParser(),
            "text": TextFallbackParser(),
        }
        self._detectors: List[Any] = [
            self._detect_openai,
            self._detect_anthropic,
        ]

    @property
    def supported_format(self) -> str:
        return "auto"

    def register_parser(self, format: str, parser: ToolCallParser) -> None:
        """
        注册自定义解析策略。

        Args:
            format: 格式标识。
            parser: ToolCallParser 实现。
        """
        self._parsers[format] = parser

    def register_detector(
        self, detector: Any
    ) -> None:
        """
        注册自定义格式检测器。

        Args:
            detector: 可调用对象，接收 response 返回 Optional[str]（格式标识）。
        """
        self._detectors.append(detector)

    def parse(self, response: Any, format: Optional[str] = None) -> List[ToolCall]:
        # 显式指定格式
        if format and format in self._parsers:
            return self._parsers[format].parse(response)

        # 自动检测
        for detector in self._detectors:
            detected = detector(response)
            if detected and detected in self._parsers:
                return self._parsers[detected].parse(response)

        # 回退到文本解析
        return self._parsers["text"].parse(response)

    def validate(self, tool_call: ToolCall, schema: Dict[str, Any]) -> ToolCallValidationResult:
        return _validate_against_schema(tool_call.args, schema)

    def repair(
        self, tool_call: ToolCall, schema: Dict[str, Any], error: ToolCallError
    ) -> Optional[ToolCall]:
        # [FIX-H5] 只调用一次 _basic_repair，不再重复调用
        repaired_args = _basic_repair(tool_call.args, schema, error)
        if repaired_args is None:
            return None
        return ToolCall(
            id=tool_call.id,
            name=tool_call.name,
            args=repaired_args,
            raw=tool_call.raw,
            confidence=tool_call.confidence * 0.9,
        )

    def format_error(self, error: ToolCallError) -> Dict[str, Any]:
        return error.to_llm_feedback()

    # ------------------------------------------------------------------
    # 自动检测逻辑
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_openai(response: Any) -> Optional[str]:
        """检测是否为 OpenAI 格式。"""
        if isinstance(response, dict):
            if "choices" in response:
                return "openai"
            if "tool_calls" in response.get("message", {}):
                return "openai"
            if "tool_calls" in response:
                items = response.get("tool_calls", [])
                if items and isinstance(items[0], dict) and "function" in items[0]:
                    return "openai"
        return None

    @staticmethod
    def _detect_anthropic(response: Any) -> Optional[str]:
        """检测是否为 Anthropic 格式。

        [S4] 支持三种形状：
            1. 原生响应：``{"content": [{"type": "tool_use", ...}, ...]}``
            2. AnthropicProvider 扁平 tool_calls：
               ``{"tool_calls": [{"type": "tool_use", ...}, ...]}``
               （此前未识别，导致被误判为文本而静默丢弃）
            3. 直接传入 ``[{"type": "tool_use", ...}]`` 列表

        注意：本检测器排在 ``_detect_openai`` 之后，OpenAI 形状
        （``tool_calls[i]["function"]``）已被前者截获，不会走到这里。
        """
        if isinstance(response, dict):
            content = response.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        return "anthropic"
            tool_calls = response.get("tool_calls")
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if isinstance(tc, dict) and tc.get("type") == "tool_use":
                        return "anthropic"
        elif isinstance(response, list):
            for tc in response:
                if isinstance(tc, dict) and tc.get("type") == "tool_use":
                    return "anthropic"
        return None
