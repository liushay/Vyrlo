"""
Schema 生成与导出工具。

功能：
- 从函数签名自动生成 JSON Schema（通过 inspect + type hints）
- 导出为 OpenAI / Anthropic / 通用 JSON Schema 格式
"""

from __future__ import annotations

import inspect
import typing
from typing import Any, Dict, List, Optional, get_origin, get_args


# ---------------------------------------------------------------------------
# Python 类型 → JSON Schema 类型映射
# ---------------------------------------------------------------------------

_TYPE_MAP: Dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}


def _py_type_to_json_type(py_type: type) -> str:
    """将 Python 类型映射为 JSON Schema 类型字符串。"""
    if py_type in _TYPE_MAP:
        return _TYPE_MAP[py_type]
    return "string"


def _resolve_annotation(annotation: Any) -> Dict[str, Any]:
    """
    将 Python 类型注解解析为 JSON Schema 片段。

    处理：
    - 基础类型（str, int, float, bool）
    - Optional[X] → anyOf [X, null]
    - List[X] → array of X
    - Dict[K, V] → object
    - Literal → enum
    """
    schema: Dict[str, Any] = {}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is None:
        # 基础类型
        schema["type"] = _py_type_to_json_type(annotation)
    elif origin is typing.Union or origin is typing.Optional:
        # Optional[X] → Union[X, None]
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            schema = _resolve_annotation(non_none[0])
        else:
            schema["anyOf"] = [_resolve_annotation(a) for a in args]
    elif origin is list or origin is typing.List:
        schema["type"] = "array"
        if args:
            schema["items"] = _resolve_annotation(args[0])
    elif origin is dict or origin is typing.Dict:
        schema["type"] = "object"
        if args:
            schema["additionalProperties"] = _resolve_annotation(args[1])
    elif origin is typing.Literal:
        schema["type"] = "string"
        schema["enum"] = list(args)
    else:
        schema["type"] = "string"

    return schema


def generate_schema_from_function(
    fn: typing.Callable[..., Any],
    *,
    description: Optional[str] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    从函数签名自动生成 JSON Schema。

    遍历函数参数的 type hints 和默认值，
    生成符合 JSON Schema 规范的参数定义。

    Args:
        fn:           目标函数。
        description:  工具描述（若不提供则从 docstring 提取首行）。
        name:         工具名称（若不提供则使用函数名）。

    Returns:
        JSON Schema 字典，包含 name、description、parameters。
    """
    sig = inspect.signature(fn)
    doc = inspect.getdoc(fn) or ""

    tool_name = name or fn.__name__
    tool_description = (
        description or doc.split("\n")[0].strip() if doc else f"执行 {tool_name}"
    )

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        prop_schema: Dict[str, Any] = {}

        # 解析类型注解
        if param.annotation is not inspect.Parameter.empty:
            prop_schema = _resolve_annotation(param.annotation)

        # 解析默认值
        if param.default is not inspect.Parameter.empty:
            prop_schema["default"] = param.default
        else:
            required.append(param_name)

        # 从 docstring 提取参数描述
        param_desc = _extract_param_description(doc, param_name)
        if param_desc:
            prop_schema["description"] = param_desc

        properties[param_name] = prop_schema

    return {
        "name": tool_name,
        "description": tool_description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def _extract_param_description(docstring: str, param_name: str) -> Optional[str]:
    """
    从 Google-style 或 reST-style docstring 中提取参数描述。

    支持格式：
    - :param param_name: description
    - Args: param_name: description
    """
    lines = docstring.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Google-style: param_name: description
        if stripped.startswith(f"{param_name}:"):
            return stripped[len(param_name) + 1:].strip()
        # reST-style: :param param_name: description
        if f":param {param_name}:" in stripped:
            idx = stripped.find(f":param {param_name}:")
            return stripped[idx + len(f":param {param_name}:"):].strip()
    return None


# ---------------------------------------------------------------------------
# 导出格式转换
# ---------------------------------------------------------------------------


def export_to_openai_format(tool: Any) -> Dict[str, Any]:
    """
    将 Tool 或 schema 字典导出为 OpenAI function calling 格式。

    格式：{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
    """
    schema = tool.params_schema if hasattr(tool, "params_schema") else tool
    name = tool.name if hasattr(tool, "name") else schema.get("name", "unknown")
    description = (
        tool.description
        if hasattr(tool, "description")
        else schema.get("description", "")
    )

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema.get("parameters", {}) if isinstance(schema, dict) else schema,
        },
    }


def export_to_anthropic_format(tool: Any) -> Dict[str, Any]:
    """
    将 Tool 或 schema 字典导出为 Anthropic tool use 格式。

    格式：{"name": ..., "description": ..., "input_schema": ...}
    """
    schema = tool.params_schema if hasattr(tool, "params_schema") else tool
    name = tool.name if hasattr(tool, "name") else schema.get("name", "unknown")
    description = (
        tool.description
        if hasattr(tool, "description")
        else schema.get("description", "")
    )

    params = schema.get("parameters", {}) if isinstance(schema, dict) else schema

    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": params.get("properties", {}),
            "required": params.get("required", []),
        },
    }


def export_to_json_schema(tool: Any) -> Dict[str, Any]:
    """
    将 Tool 或 schema 字典导出为通用 JSON Schema 格式。

    格式：{"title": ..., "description": ..., "type": "object", "properties": ...}
    """
    schema = tool.params_schema if hasattr(tool, "params_schema") else tool
    name = tool.name if hasattr(tool, "name") else schema.get("name", "unknown")
    description = (
        tool.description
        if hasattr(tool, "description")
        else schema.get("description", "")
    )

    params = schema.get("parameters", {}) if isinstance(schema, dict) else schema

    return {
        "title": name,
        "description": description,
        "type": "object",
        "properties": params.get("properties", {}),
        "required": params.get("required", []),
    }