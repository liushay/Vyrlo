"""
[测试] FakeTool — 可控成功/失败/超时的工具工厂。

配合 ``InMemoryToolRegistry`` 使用：通过 ``make_tool`` 生成带脚本化
行为的 ``Tool`` 对象，其 ``fn`` 绑定到本实例的脚本队列。

支持三种行为：
    - success: 正常返回，fn 返回 dict。
    - fail:    fn 抛出异常，由 registry 包装为 ToolResult(error=True)。
    - timeout: fn 阻塞，触发 registry 的超时隔离（需搭配很短的工具 timeout）。

同时记录每次执行的工具名与参数，供断言统计执行次数。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from runtime.tool_registry.interface import Tool


class FakeTool:
    """可编程工具行为容器。

    通过 ``script`` 队列逐次返回不同的行为；队列耗尽后使用 ``default``。
    """

    def __init__(self, default: str = "success") -> None:
        self.script: List[Any] = []          # "success" | "fail" | "timeout" | ("success", value)
        self.default: str = default
        self.call_log: List[Dict[str, Any]] = []

    def enqueue(self, behavior: str) -> "FakeTool":
        """入队一个行为。"""
        self.script.append(behavior)
        return self

    def enqueue_success_result(self, value: Any) -> "FakeTool":
        """入队一个返回指定 content 的成功行为。"""
        self.script.append(("success", value))
        return self

    def _next_behavior(self) -> Any:
        if self.script:
            return self.script.pop(0)
        return self.default

    def _fn(self, tool_name: str, timeout: Optional[float], **args: Any) -> Any:
        """实际被 registry 调用的 fn。"""
        self.call_log.append({"tool": tool_name, "args": args})
        behavior = self._next_behavior()

        if isinstance(behavior, tuple):
            kind, payload = behavior
        else:
            kind, payload = behavior, None

        if kind == "fail":
            raise RuntimeError(f"FakeTool[{tool_name}]: simulated failure")

        if kind == "timeout":
            # 阻塞一个远超 tool.timeout 的时长，触发 registry 超时隔离。
            sleep = max(float(timeout or 0.1) * 2, 0.05)
            time.sleep(sleep)
            return {"should": "not-return"}

        # success（含 tuple 中的成功结果）
        return payload if payload is not None else {"ok": True, "tool": tool_name}

    def make_tool(
        self,
        name: str,
        description: str = "",
        timeout: Optional[float] = None,
        requires_approval: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tool:
        """构造一个绑定到本实例行为的 Tool。"""
        return Tool(
            name=name,
            description=description or f"fake tool {name}",
            params_schema={
                "name": name,
                "description": description or f"fake tool {name}",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            fn=lambda **args: self._fn(name, timeout, **args),
            timeout=timeout,
            requires_approval=requires_approval,
            metadata=dict(metadata or {}),
            source="local",
        )


__all__ = ["FakeTool"]