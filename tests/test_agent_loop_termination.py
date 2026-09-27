"""
[B1/D001] AgentLoop 终止语义 ——「无 tool_calls 即自然结束」。

背景（DEFECTS.md D001）：
    修复前 AgentLoop 从不读取 finish_reason，也没有"不产生工具调用"的终止信号，
    单轮无 tool_calls 的对话会空转至 max_iterations，产生成倍的 LLM 调用。

修复后的默认行为：
    一轮迭代完成后，若 ctx 私有域中没有解析出的 tool_calls，视为 LLM 已给出
    最终答复，Loop 自然结束（**不**置 ctx.aborted，因为自然结束不是异常或主动中止）。

本文件覆盖：
    1. test_loop_terminates_when_no_tool_calls
    2. test_loop_continues_when_tool_calls_present
    3. test_loop_terminates_on_empty_content_no_tool_calls   （同时验证 D008 场景）

运行方式：
    python -m pytest tests/test_agent_loop_termination.py -v
"""

from __future__ import annotations

from typing import Any, Dict, List

from agent_loop import AgentLoop, Context, HookResult, Middleware


# ============================================================================
# 测试辅助
# ============================================================================


class _ToolCallBridge(Middleware):
    """把 dict 形状 LLM 响应中的 tool_calls 写入 ctx 私有域。

    纯 AgentLoop 场景下没有 Runtime 的 _ToolCallBridgeMiddleware，
    需要测试自行提供解析桥，才能驱动 Loop 的工具调用流程。
    """

    def __init__(self) -> None:
        super().__init__("_test_tool_call_bridge")

    def after_llm(self, ctx: Context, response: Any) -> HookResult:
        if isinstance(response, dict):
            tcs = response.get("tool_calls") or []
            if tcs:
                ctx.set_tool_calls([
                    {
                        "id": tc.get("id"),
                        "name": tc.get("name", ""),
                        "arguments": tc.get("arguments", {}) or {},
                    }
                    for tc in tcs
                ])
        return HookResult.continue_()


class _CountingMiddleware(Middleware):
    """统计各生命周期钩子被触发的次数。"""

    def __init__(self) -> None:
        super().__init__("_counting")
        self.after_iteration_calls = 0
        self.exit_loop_calls = 0

    def after_iteration(self, ctx: Context) -> HookResult:
        self.after_iteration_calls += 1
        return HookResult.continue_()

    def on_exit_loop(self, ctx: Context) -> HookResult:
        self.exit_loop_calls += 1
        return HookResult.continue_()


def _make_loop(llm_call, tool_executor=None, max_iterations: int = 10) -> AgentLoop:
    return AgentLoop(
        llm_call=llm_call,
        tool_executor=tool_executor or (lambda name, args: "ok"),
        max_iterations=max_iterations,
    )


# ============================================================================
# 1. 无 tool_calls → Loop 自然终止（只调用 LLM 一次）
# ============================================================================


def test_loop_terminates_when_no_tool_calls():
    """LLM 单轮给出最终答复（无 tool_calls）→ Loop 只跑 1 轮即结束。

    回归口径：D001 最小复现从 10 次 llm_call 变为 1 次。
    """
    calls: List[List[Dict[str, Any]]] = []

    def llm_call(messages):
        calls.append(messages)
        return {"content": "你好，有什么可以帮您？", "tool_calls": []}

    ctx = Context()
    loop = _make_loop(llm_call, max_iterations=10)
    loop.register_middleware(_ToolCallBridge())
    counting = _CountingMiddleware()
    loop.register_middleware(counting)

    loop.run(ctx, [{"role": "user", "content": "hello"}])

    # 只调用 LLM 一次，不空转
    assert len(calls) == 1, f"无 tool_calls 应只调用 LLM 1 次，实际 {len(calls)}"
    # 恰好一轮迭代
    assert counting.after_iteration_calls == 1
    assert counting.exit_loop_calls == 1
    # 自然结束不是"中止"，aborted 必须保持 False
    assert ctx.aborted is False


# ============================================================================
# 2. 有 tool_calls → Loop 继续下一轮，直到无 tool_calls 才终止
# ============================================================================


def test_loop_continues_when_tool_calls_present():
    """第 1 轮带 tool_calls → 继续；第 2 轮无 tool_calls → 终止。"""
    calls = {"n": 0}

    def llm_call(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "content": "",
                "tool_calls": [{"id": "c1", "name": "echo", "arguments": {"q": "hi"}}],
            }
        return {"content": "已完成", "tool_calls": []}

    executed: List[str] = []

    def tool_executor(name, args):
        executed.append(name)
        return "ok"

    ctx = Context()
    loop = _make_loop(llm_call, tool_executor, max_iterations=10)
    loop.register_middleware(_ToolCallBridge())
    counting = _CountingMiddleware()
    loop.register_middleware(counting)

    loop.run(ctx, [{"role": "user", "content": "echo hi"}])

    # 两轮：第 1 轮工具调用，第 2 轮收尾
    assert calls["n"] == 2, f"有 tool_calls 应继续到第 2 轮，实际 {calls['n']}"
    assert executed == ["echo"]
    assert counting.after_iteration_calls == 2
    assert ctx.aborted is False


# ============================================================================
# 3. 空 content + 无 tool_calls → 同样只跑 1 轮（D008 场景）
# ============================================================================


def test_loop_terminates_on_empty_content_no_tool_calls():
    """LLM 返回空 content 且无 tool_calls → Loop 亦正常终止，只跑 1 轮。

    该测试同时验证 D008：「LLM 返回空 content 且无 tool_calls 的『正常结束』」
    在 D001 修复后不再空转 —— 空响应与普通无工具响应共享同一条终止路径。
    """
    calls = {"n": 0}

    def llm_call(messages):
        calls["n"] += 1
        return {"content": "", "tool_calls": []}

    ctx = Context()
    loop = _make_loop(llm_call, max_iterations=10)
    loop.register_middleware(_ToolCallBridge())
    counting = _CountingMiddleware()
    loop.register_middleware(counting)

    loop.run(ctx, [{"role": "user", "content": "empty"}])

    # 空响应不再导致空转：只跑 1 轮、只调用一次 LLM
    assert calls["n"] == 1, f"空响应应只调用 LLM 1 次，实际 {calls['n']}"
    assert counting.after_iteration_calls == 1, "结束记录恰好一次"
    assert counting.exit_loop_calls == 1
    assert ctx.aborted is False
    # 无 tool_calls 残留（finally 中已清理）
    assert ctx.get_tool_calls() == []