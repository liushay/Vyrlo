"""
[测试] fakes 包 — 测试专用 Fake 组件。

包含 FakeLLMProvider（可编程 LLM）、FakeTool（可控成功/失败/超时）、
FakeSandbox（可控路径越界/不可用）。
"""

from tests.fakes.fake_llm import FakeLLMProvider, openai_tool_call
from tests.fakes.fake_tool import FakeTool
from tests.fakes.fake_sandbox import FakeSandboxProvider

__all__ = [
    "FakeLLMProvider",
    "openai_tool_call",
    "FakeTool",
    "FakeSandboxProvider",
]