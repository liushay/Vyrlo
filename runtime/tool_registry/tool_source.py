"""
ToolSource — 工具来源抽象。

支持三种来源：
- LocalFunctionSource: 装饰器自动转换的 Python 函数
- MCPSource: MCP 服务端工具列表
- ExternalAPISource: 外部 API 的工具列表
"""

from __future__ import annotations

import copy
import traceback
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from runtime.tool_registry.interface import Tool
from runtime.tool_registry.schema import generate_schema_from_function


# ============================================================================
# ToolSource 抽象接口
# ============================================================================


class ToolSource(ABC):
    """
    工具来源抽象接口。

    每种来源负责发现并返回 Tool 列表。
    用户可实现此接口来接入新的工具来源。
    """

    @abstractmethod
    def discover(self) -> List[Tool]:
        """
        发现并返回工具列表。

        对于外部来源（MCP、API），此方法可能涉及网络调用。
        对于本地来源，直接从模块中收集。
        """
        ...

    @abstractmethod
    def source_type(self) -> str:
        """返回来源类型标识（如 "local"、"mcp"、"external"）。"""
        ...


# ============================================================================
# LocalFunctionSource
# ============================================================================


class LocalFunctionSource(ToolSource):
    """
    本地函数工具来源。

    从模块或函数列表中收集被 @function_tool 装饰的函数。
    也支持原始函数：调用 register_function 时自动包装为 Tool。

    用法：
        source = LocalFunctionSource()
        source.register_function(my_func, name="my_tool")
        # 或从模块批量注册
        source.register_from_module(my_module)
    """

    def __init__(self) -> None:
        self._functions: Dict[str, Tool] = {}
        self._raw_functions: List[Callable[..., Any]] = []

    def source_type(self) -> str:
        return "local"

    def register_function(
        self,
        fn: Callable[..., Any],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        requires_approval: bool = False,
        is_idempotent: bool = True,
        timeout: Optional[float] = None,
    ) -> Tool:
        """
        注册一个普通 Python 函数，自动包装为 Tool。

        Args:
            fn:               目标函数。
            name:             工具名称（若不提供则使用函数名）。
            description:      工具描述（若不提供则从 docstring 提取）。
            requires_approval: 是否需要用户审批。
            is_idempotent:    是否幂等。
            timeout:          超时时间。

        Returns:
            生成的 Tool 实例。
        """
        schema = generate_schema_from_function(fn, name=name, description=description)
        tool = Tool(
            name=schema["name"],
            description=schema["description"],
            params_schema=schema,
            fn=fn,
            requires_approval=requires_approval,
            is_idempotent=is_idempotent,
            timeout=timeout,
            source="local",
        )
        self._functions[tool.name] = tool
        return tool

    def register_from_module(self, module: Any) -> List[Tool]:
        """
        从模块中批量注册 Tool 实例。

        扫描模块中所有 Tool 类型的对象（被 @function_tool 装饰的函数）。

        Args:
            module: Python 模块对象。

        Returns:
            注册的 Tool 列表。
        """
        import inspect

        registered: List[Tool] = []
        for _attr_name, obj in inspect.getmembers(module):
            if isinstance(obj, Tool):
                self._functions[obj.name] = obj
                registered.append(obj)
        return registered

    def discover(self) -> List[Tool]:
        """返回所有已注册的本地工具。"""
        return list(self._functions.values())


# ============================================================================
# MCPSource
# ============================================================================


class MCPSource(ToolSource):
    """
    MCP 服务端工具来源。

    通过 MCP 协议从外部服务端获取工具列表。
    工具的实际执行在 MCP 服务端完成，本地只保留元数据。

    注意：当前实现为接口预留，需要配合 MCP 客户端 SDK 使用。

    用法：
        source = MCPSource(server_url="http://localhost:8080")
        tools = source.discover()  # 远程获取工具列表
    """

    def __init__(
        self,
        server_url: str = "",
        *,
        transport: str = "stdio",
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            server_url: MCP 服务端 URL（transport="http" 时使用）。
            transport:  传输协议，"stdio" 或 "http"。
            command:    启动 MCP server 的命令（transport="stdio" 时使用）。
            args:       MCP server 的命令行参数。
        """
        self.server_url = server_url
        self.transport = transport
        self.command = command
        self.args = args or []
        self._cached_tools: List[Tool] = []

    def source_type(self) -> str:
        return "mcp"

    def discover(self) -> List[Tool]:
        """
        从 MCP 服务端获取工具列表。

        若缓存中有值则直接返回，否则发起工具发现请求。

        注：完整 MCP 实现需要 mcp 客户端库。
        当前提供占位实现，返回空列表。
        """
        if self._cached_tools:
            return list(self._cached_tools)

        # TODO: 实现 MCP 协议的工具发现
        # 通过 MCP 客户端建立连接 → list_tools → 转换为 Tool 对象
        # 工具的实际 fn 指向 MCP 远程调用
        return []

    def _make_remote_tool_fn(self, tool_name: str) -> Callable[..., Any]:
        """
        生成指向 MCP 远程调用的代理函数。

        注：完整实现需要 MCP 客户端运行时。
        """

        def _remote_call(**kwargs: Any) -> Any:
            # TODO: 通过 MCP 客户端调用 tool_name(kwargs)
            raise NotImplementedError(
                f"MCP 远程调用尚未实现: {tool_name}({kwargs})"
            )

        return _remote_call

    def cache_tools(self, tools: List[Tool]) -> None:
        """手动预填充工具缓存（用于测试或离线模式）。"""
        self._cached_tools = list(tools)


# ============================================================================
# ExternalAPISource
# ============================================================================


class ExternalAPISource(ToolSource):
    """
    外部 API 工具来源。

    从 OpenAPI/Swagger 规范或自定义工具注册端点发现工具。

    用法：
        source = ExternalAPISource(api_url="https://api.example.com/tools")
        tools = source.discover()
    """

    def __init__(
        self,
        api_url: str = "",
        *,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        spec: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            api_url: 工具注册 API 端点 URL。
            api_key: API 密钥。
            headers: 额外的 HTTP 请求头。
            spec:    OpenAPI 规范字典（离线模式）。
        """
        self.api_url = api_url
        self.api_key = api_key
        self.headers = headers or {}
        self.spec = spec
        self._cached_tools: List[Tool] = []

    def source_type(self) -> str:
        return "external"

    def discover(self) -> List[Tool]:
        """
        从外部 API 发现工具。

        若 spec 已提供则解析 OpenAPI 规范；
        否则向 api_url 发起 HTTP GET 获取工具列表。

        注：完整实现需要 httpx 或 requests 库。
        """
        if self._cached_tools:
            return list(self._cached_tools)

        if self.spec:
            return self._parse_openapi_spec(self.spec)

        # TODO: HTTP GET api_url → 解析工具列表
        return []

    def _parse_openapi_spec(self, spec: Dict[str, Any]) -> List[Tool]:
        """
        从 OpenAPI 3.x 规范中提取工具定义。

        将每个 path → operation 映射为一个 Tool。
        """
        tools: List[Tool] = []
        paths = spec.get("paths", {})

        for path, methods in paths.items():
            for method, operation in methods.items():
                if method not in ("get", "post", "put", "delete", "patch"):
                    continue
                if not isinstance(operation, dict):
                    continue

                op_id = operation.get("operationId", f"{method}_{path}")
                description = operation.get("description", operation.get("summary", ""))
                params_schema = self._operation_to_schema(method, path, operation)

                tool = Tool(
                    name=op_id,
                    description=description or f"{method.upper()} {path}",
                    params_schema=params_schema,
                    fn=None,  # 外部 API 工具不可在本地执行
                    source="external",
                    is_idempotent=method in ("get",),
                )
                tools.append(tool)

        return tools

    def _operation_to_schema(
        self, method: str, path: str, operation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """将 OpenAPI operation 转换为 JSON Schema 参数定义。"""
        properties: Dict[str, Any] = {}
        required: List[str] = []

        # 从 parameters 提取路径参数和查询参数
        for param in operation.get("parameters", []):
            name = param.get("name", "")
            if name:
                properties[name] = {
                    "type": param.get("schema", {}).get("type", "string"),
                    "description": param.get("description", ""),
                }
                if param.get("required", False):
                    required.append(name)

        # 从 requestBody 提取请求体 schema
        request_body = operation.get("requestBody", {})
        body_schema = request_body.get("content", {}).get("application/json", {}).get("schema", {})
        if body_schema:
            properties.update(body_schema.get("properties", {}))
            required.extend(body_schema.get("required", []))

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def cache_tools(self, tools: List[Tool]) -> None:
        """手动预填充工具缓存。"""
        self._cached_tools = list(tools)