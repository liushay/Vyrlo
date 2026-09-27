"""
InMemoryToolRegistry — ToolRegistry 接口的默认内存实现。

特点：
- 字典存储，O(1) 按名称查找
- 支持从 ToolSource 批量注册
- 执行隔离：超时控制 + 错误边界
"""

from __future__ import annotations

import atexit
import concurrent.futures
import logging
import time
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from runtime.resilience.retry import RetryPolicy
from runtime.tool_registry.interface import Tool, ToolRegistry, ToolResult
from runtime.tool_registry.tool_source import ToolSource

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查，避免与 sandbox 包形成循环导入
    from runtime.sandbox.interface import SandboxHandle, SandboxProvider
from runtime.tool_registry.schema import (
    export_to_openai_format,
    export_to_anthropic_format,
    export_to_json_schema,
)

logger = logging.getLogger(__name__)

# ============================================================================
# 执行隔离工具
# ============================================================================

# 全局线程池，用于在子线程中执行工具函数并施加超时控制
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=8, thread_name_prefix="tool_exec"
)

# 进程退出时回收线程池，避免长期运行下的线程资源泄漏。
# wait=False 不阻塞退出，已在执行中的任务由解释器随进程一并回收。
atexit.register(_EXECUTOR.shutdown, wait=False)


def _execute_with_timeout(
    fn: Any, args: Dict[str, Any], timeout: Optional[float]
) -> ToolResult:
    """
    在线程池中执行工具函数，带超时保护。

    设计理由：
    - 使用 ThreadPoolExecutor 而非 asyncio，因为工具函数可能是同步的。
    - 超时通过 Future.result(timeout) 实现。
    - 所有异常被捕获并包装为 ToolResult(error=True)，不传播到调用方。
    """
    start_time = time.monotonic()

    def _run() -> Any:
        return fn(**args)

    future = _EXECUTOR.submit(_run)

    try:
        if timeout is not None and timeout > 0:
            result = future.result(timeout=timeout)
        else:
            result = future.result()

        elapsed = time.monotonic() - start_time
        return ToolResult(
            content=result,
            error=False,
            sources=[],
            metadata={"elapsed_seconds": round(elapsed, 3)},
            confidence=1.0,
        )

    except concurrent.futures.TimeoutError:
        future.cancel()
        elapsed = time.monotonic() - start_time
        logger.warning(
            "工具执行超时（%.1fs > %.1fs）", elapsed, timeout
        )
        return ToolResult.from_error(
            f"工具执行超时：超过 {timeout}s 限制",
            metadata={"elapsed_seconds": round(elapsed, 3), "timeout": timeout},
        )

    except Exception as exc:
        elapsed = time.monotonic() - start_time
        tb = traceback.format_exc()
        logger.warning("工具执行异常: %s\n%s", exc, tb)
        return ToolResult.from_error(
            f"工具执行异常: {type(exc).__name__}: {exc}",
            metadata={"elapsed_seconds": round(elapsed, 3), "traceback": tb},
        )


# ============================================================================
# InMemoryToolRegistry
# ============================================================================


class InMemoryToolRegistry(ToolRegistry):
    """
    ToolRegistry 接口的默认内存实现。

    用法：
        registry = InMemoryToolRegistry()

        # 注册单个 Tool
        registry.register(my_tool)

        # 从 ToolSource 批量注册
        registry.register_from_source(local_source)

        # 查找
        tool = registry.get("my_tool")

        # 执行
        result = registry.execute("my_tool", {"x": 1, "y": "hello"})
    """

    def __init__(
        self,
        sandbox_provider: Optional[SandboxProvider] = None,
        tool_retry_policy: Optional[RetryPolicy] = None,
    ) -> None:
        """
        Args:
            sandbox_provider: 可选的沙箱提供者。为 None 时不启用沙箱
                              （向后兼容）。提供了沙箱句柄时 execute 委托给它。
            tool_retry_policy: [D2] 可选的重试策略。仅对**幂等**工具在返回
                              ``ToolResult(error=True)`` 时重试；非幂等工具
                              永不重试。为 None 时保持原行为（不重试）。
        """
        self._tools: Dict[str, Tool] = {}
        self._sandbox_provider = sandbox_provider
        self._tool_retry_policy = tool_retry_policy

    # ------------------------------------------------------------------
    # 注册 / 注销
    # ------------------------------------------------------------------

    def register(self, tool: Tool) -> None:
        """
        注册一个 Tool。

        若已存在同名工具则抛出 ValueError。
        这是有意为之：工具名称是全局唯一标识，不应静默覆盖。
        """
        if tool.name in self._tools:
            raise ValueError(
                f"工具 '{tool.name}' 已存在。请先调用 unregister() 或使用不同名称。"
            )
        self._tools[tool.name] = tool
        logger.debug("已注册工具: %s (来源: %s)", tool.name, tool.source)

    def unregister(self, name: str) -> Optional[Tool]:
        """
        注销工具。

        返回被注销的 Tool 实例，若不存在则返回 None。
        """
        tool = self._tools.pop(name, None)
        if tool:
            logger.debug("已注销工具: %s", name)
        return tool

    def register_from_source(self, source: ToolSource) -> int:
        """
        从 ToolSource 批量注册工具。

        遍历 source.discover() 返回的工具列表，逐一注册。

        Args:
            source: ToolSource 实例。

        Returns:
            成功注册的工具数量。

        Raises:
            ValueError: 当源中找到的工具与已注册工具重名时，
                       部分工具可能已先注册成功，不会被回滚。
        """
        tools = source.discover()
        count = 0
        for tool in tools:
            self.register(tool)
            count += 1
        logger.info(
            "从来源 '%s' 注册了 %d 个工具", source.source_type(), count
        )
        return count

    # ------------------------------------------------------------------
    # 查找
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[Tool]:
        """按名称获取工具。"""
        return self._tools.get(name)

    def list(self) -> List[Tool]:
        """列出所有已注册工具。"""
        return list(self._tools.values())

    # ------------------------------------------------------------------
    # Schema 导出
    # ------------------------------------------------------------------

    def export_schemas(self, format: str = "openai") -> List[Dict[str, Any]]:
        """
        导出所有工具 schema 为指定格式。

        Args:
            format: "openai" | "anthropic" | "json_schema"

        Returns:
            对应格式的 schema 列表。
        """
        _EXPORTERS = {
            "openai": export_to_openai_format,
            "anthropic": export_to_anthropic_format,
            "json_schema": export_to_json_schema,
        }

        exporter = _EXPORTERS.get(format)
        if exporter is None:
            raise ValueError(
                f"不支持的导出格式: '{format}'。可选: {list(_EXPORTERS.keys())}"
            )

        return [exporter(tool) for tool in self._tools.values()]

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------

    def execute(
        self, name: str, args: Dict[str, Any], sandbox: Optional[SandboxHandle] = None
    ) -> ToolResult:
        """
        执行工具，带超时隔离和错误边界。

        执行生命周期：
        1. 按名称查找工具 → 不存在则返回错误
        2. sandbox 非 None 时委托 sandbox_provider.execute
        3. 检查 fn 是否为 None → 是则返回错误（如外部 API 工具）
        4. 在线程池中执行 fn(**args)，带超时保护
        5. 所有异常被捕获，包装为 ToolResult(error=True)
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.from_error(
                f"工具 '{name}' 未注册。可用的工具: {list(self._tools.keys())}"
            )

        # 沙箱路径：委托 provider 执行（需注册 sandbox_provider）
        if sandbox is not None:
            if self._sandbox_provider is None:
                return ToolResult.from_error("未配置 sandbox_provider，无法在沙箱内执行工具")
            return self._sandbox_provider.execute(sandbox, name, args)

        if tool.fn is None:
            return ToolResult.from_error(
                f"工具 '{name}' 没有可执行的函数（来源: {tool.source}）。"
            )

        # [D2] 幂等工具失败重试；非幂等工具绝不重试。
        if self._tool_retry_policy is not None and tool.is_idempotent:
            return self._tool_retry_policy.execute(
                lambda: _execute_with_timeout(tool.fn, args, tool.timeout),
                should_retry=lambda r: bool(getattr(r, "error", False)),
            )

        return _execute_with_timeout(tool.fn, args, tool.timeout)
