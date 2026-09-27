"""
function_tool 装饰器 — 自动从函数签名生成 Tool。

参考 OpenAI Agents SDK 的 @function_tool 装饰器设计。
支持手动覆写元数据字段。
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Optional

from runtime.tool_registry.interface import Tool
from runtime.tool_registry.schema import generate_schema_from_function


def function_tool(
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    requires_approval: bool = False,
    is_idempotent: bool = True,
    timeout: Optional[float] = None,
    returns_schema: Optional[Dict[str, Any]] = None,
) -> Callable[[Callable[..., Any]], Tool]:
    """
    装饰器：将普通 Python 函数转换为 Tool。

    自动从函数签名和 docstring 生成 params_schema。
    同时保留原始函数引用（fn），供执行时使用。

    用法：
        @function_tool(name_override="my_tool", description_override="...")
        def my_tool(x: int, y: str = "default") -> str:
            '''工具描述（自动提取为 description）。'''
            ...

        # 可直接作为 Tool 使用：
        tool = my_tool  # my_tool 现在是一个 Tool 实例
        # 也可调用原始函数：
        result = tool.fn(x=1, y="hello")

    Args:
        name:                工具名称（兼容旧 API，等同于 name_override）。
        description:         工具描述（兼容旧 API，等同于 description_override）。
        name_override:       覆写自动生成的工具名称。
        description_override: 覆写自动生成的工具描述。
        requires_approval:   是否需要用户审批。
        is_idempotent:       是否幂等。
        timeout:             超时时间。
        returns_schema:      返回值 schema（可选）。

    Returns:
        一个 Tool 实例，替代原始函数对象。
    """
    _name = name_override or name
    _description = description_override or description

    def decorator(fn: Callable[..., Any]) -> Tool:
        # 从函数签名自动生成 schema
        schema = generate_schema_from_function(
            fn,
            name=_name,
            description=_description,
        )

        tool = Tool(
            name=schema["name"],
            description=schema["description"],
            params_schema=schema,
            returns_schema=returns_schema,
            fn=fn,
            requires_approval=requires_approval,
            is_idempotent=is_idempotent,
            timeout=timeout,
            source="local",
        )

        # 保留原始函数的元数据，方便调试
        functools.update_wrapper(tool, fn)

        return tool

    return decorator