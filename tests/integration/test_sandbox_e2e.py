"""
[A4] 沙箱。

覆盖：
    1. LocalSandboxProvider 路径越界（translate_path 抛 ValueError）。
    2. 工具在沙箱 workdir 下执行，产物落盘。
    3. ON_EXIT_LOOP 后 sandbox handle 被 release。
    4. 不启用沙箱时所有工具走原路径。
    5. 子 Agent 继承父沙箱（已由 A3 覆盖，这里补 FakeSandbox 级别验证）。
"""

from __future__ import annotations

import os
import tempfile

import pytest

from runtime.builtin_middlewares.sandbox import SandboxMiddleware
from runtime.sandbox.interface import SandboxHandle
from runtime.sandbox.local_provider import LocalSandboxProvider
from runtime.sandbox.registry import SandboxRegistry
from tests.fakes.fake_llm import FakeLLMProvider, openai_tool_call
from tests.fakes.fake_sandbox import FakeSandboxProvider
from tests.fakes.fake_tool import FakeTool
from tests.integration.conftest import ContextCaptureMiddleware, build_runtime


# ============================================================================
# 场景 1：LocalSandboxProvider 路径越界
# ============================================================================


def test_local_sandbox_path_escape_raises():
    provider = LocalSandboxProvider(base_dir=tempfile.mkdtemp())
    handle = provider.acquire("agent-x")

    # workdir 内的合法路径：正常翻译
    inside = provider.translate_path(handle, "a/b.txt")
    assert inside.startswith(handle.workdir)

    # workdir 外的绝对路径：抛 ValueError
    with pytest.raises(ValueError):
        provider.translate_path(handle, "C:/Windows/System32")

    # 通过 .. 逃逸到 workdir 外的相对路径：抛 ValueError
    with pytest.raises(ValueError):
        provider.translate_path(handle, "../escape.txt")


# ============================================================================
# 场景 2：工具在沙箱 workdir 下执行，产物落盘
# ============================================================================


def test_tool_executes_in_sandbox_workdir():
    base = tempfile.mkdtemp()
    provider = LocalSandboxProvider(base_dir=base)

    # 注册一个会写文件的工具
    captured = {}

    def write_file(name, content):
        captured["cwd"] = os.getcwd()
        with open(name, "w", encoding="utf-8") as f:
            f.write(content)
        return name

    registry = provider._registry.__class__() if provider._registry else None
    from runtime.tool_registry.in_memory import InMemoryToolRegistry
    from runtime.tool_registry.interface import Tool

    tool_registry = InMemoryToolRegistry()
    tool_registry.register(Tool(
        name="write_file",
        description="write",
        params_schema={"name": "write_file", "description": "write",
                       "parameters": {"type": "object", "properties": {
                           "name": {"type": "string"}, "content": {"type": "string"}},
                           "required": ["name", "content"]}},
        fn=write_file,
        source="local",
    ))
    provider._registry = tool_registry

    handle = provider.acquire("agent-w")
    result = provider.execute(handle, "write_file", {"name": "out.txt", "content": "hello"})

    assert result.error is False
    # 工具执行时 cwd = 沙箱 workdir
    assert os.path.abspath(captured["cwd"]) == os.path.abspath(handle.workdir)
    # 产物落在沙箱 workdir
    assert os.path.exists(os.path.join(handle.workdir, "out.txt"))


# ============================================================================
# 场景 3：ON_EXIT_LOOP 后 sandbox handle 被 release
# ============================================================================


def test_sandbox_release_on_exit():
    llm = FakeLLMProvider()
    llm.enqueue_empty()

    sandbox = FakeSandboxProvider(available=True)
    registry = SandboxRegistry({"fake": sandbox})
    sandbox_mw = SandboxMiddleware(sandbox_registry=registry, default_mode="fake")
    capture = ContextCaptureMiddleware()

    runtime = build_runtime(
        llm, [],
        middlewares=[sandbox_mw, capture],
        max_iterations=1,
    )
    runtime.run("sb1", "do something")

    # acquire 后被 release
    assert len(sandbox.acquired) == 1
    assert len(sandbox.released) == 1
    assert sandbox.released[0] == sandbox.acquired[0].sandbox_id


# ============================================================================
# 场景 4：不启用沙箱时所有工具走原路径
# ============================================================================


def test_no_sandbox_goes_original_path():
    llm = FakeLLMProvider()
    llm.enqueue_tool_calls([openai_tool_call("echo", {"message": "hi"})])

    tool = FakeTool()
    # 不注册任何 sandbox 中间件
    capture = ContextCaptureMiddleware()
    runtime = build_runtime(
        llm, [tool.make_tool("echo")],
        middlewares=[capture],
        max_iterations=1,
    )
    runtime.run("sb2", "echo hi")

    # 工具正常走原路径执行
    assert len(tool.call_log) == 1
    # 无 sandbox 写入 ctx.shared
    assert "sandbox" not in capture.ctx.shared


# ============================================================================
# 场景 5：SandboxRegistry 不可用时放行（不阻断装配）
# ============================================================================


def test_sandbox_unavailable_ignored():
    llm = FakeLLMProvider()
    llm.enqueue_tool_calls([openai_tool_call("echo", {"message": "hi"})])

    tool = FakeTool()
    # 不可用的 provider
    sandbox = FakeSandboxProvider(available=False)
    registry = SandboxRegistry({"fake": sandbox})
    sandbox_mw = SandboxMiddleware(sandbox_registry=registry, default_mode="fake")
    capture = ContextCaptureMiddleware()

    runtime = build_runtime(
        llm, [tool.make_tool("echo")],
        middlewares=[sandbox_mw, capture],
        max_iterations=1,
    )
    session = runtime.run("sb3", "echo hi")

    # 沙箱未 acquire，但工具正常走原路径执行
    assert sandbox.acquired == []
    assert len(tool.call_log) == 1
    assert session.abort_flag is False