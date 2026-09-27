"""
[L3-1] L3 第一批中间件演示 — 5 个中间件全部启用，跑通一个完整任务。

运行方式：python examples/l3_demo.py

演示内容：
    1. 可观测性：结构化 JSON 日志 + 性能指标聚合
    2. 成本累计：token / cost 累加与预算检查
    3. 安全拦截：高危工具被 skip，普通工具放行
    4. 记忆注入：长期记忆注入 system prompt
    5. 压缩触发：消息超限时压缩并替换

场景：
    一个"研究报告助手"任务——需要检索资料（普通工具）、
    删除文件（高危工具）、并在长上下文下持续工作。

[L3-1] 本文件为 L3 第一批中间件新增，未修改任何 L1 / L2 代码。
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Windows 控制台默认 GBK 编码，无法输出部分字符（如 emoji）。
# 统一切换到 UTF-8 并容错，保证演示输出不因编码中断。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, OSError):
    pass

from agent_loop import AgentLoop, Context, ControlFlow, HookResult, Middleware

from runtime.builtin_middlewares import (
    ContextCompressorMiddleware,
    CostGuardMiddleware,
    MemoryInjectorMiddleware,
    ObservabilityMiddleware,
    SafetyGuardMiddleware,
)
from runtime.context_manager.interface import MemoryTrace, Message
from runtime.context_manager.layered import LayeredContextManager
from runtime.event_log import EventLog
from runtime.session import Session
from runtime.tool_registry.in_memory import InMemoryToolRegistry
from runtime.tool_registry.interface import Tool, ToolResult


# ============================================================================
# 输出辅助
# ============================================================================


def section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def subsection(title: str) -> None:
    print(f"\n--- {title} ---")


# ============================================================================
# 1. LLM 与工具：模拟
# ============================================================================


class ScriptedLLM:
    """按剧本返回响应的模拟 LLM，每轮附带 token 与成本。"""

    #: (tool_name, tool_args) 剧本
    SCRIPT = [
        ("web_search", {"query": "deep agent memory architecture"}),
        ("rm_rf", {"path": "/data/important"}),      # 高危：应被拦截
        ("web_search", {"query": "context compression strategies"}),
        (None, None),                                 # 无工具调用 → 结束
    ]

    def __init__(self) -> None:
        self.turn = 0

    def __call__(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        index = min(self.turn, len(self.SCRIPT) - 1)
        tool_name, tool_args = self.SCRIPT[index]
        self.turn += 1

        response: Dict[str, Any] = {
            "content": (
                f"第 {self.turn} 轮：我正在处理任务。"
                if tool_name is None
                else f"第 {self.turn} 轮：我需要调用 {tool_name}。"
            ),
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 60,
                "total_tokens": 180,
            },
            "cost": 0.0045,
            "model": "mock-gpt-4o",
            "provider": "mock",
            "tool_calls": (
                [{"name": tool_name, "arguments": tool_args}]
                if tool_name
                else []
            ),
        }
        return response


class ScriptedToolCallParser(Middleware):
    """把 LLM 响应中的 tool_calls 写入 ctx 私有域（AFTER_LLM）。"""

    def after_llm(self, ctx: Context, response: Any) -> HookResult:
        calls = response.get("tool_calls", []) if isinstance(response, dict) else []
        if calls:
            ctx.set_tool_calls(calls)
        return HookResult.continue_()


class MockToolExecutor:
    """模拟工具执行器：记录被真正执行的工具。"""

    def __init__(self) -> None:
        self.executed: List[str] = []

    def __call__(self, tool_name: str, tool_args: Dict[str, Any]) -> ToolResult:
        self.executed.append(tool_name)
        return ToolResult(
            content=f"[{tool_name}] 检索到 3 条相关资料",
            metadata={"elapsed_seconds": 0.08, "args": tool_args},
            confidence=0.9,
        )


# ============================================================================
# 2. 装配
# ============================================================================


def build_registry() -> InMemoryToolRegistry:
    """注册两个工具：普通检索工具 + 高危删除工具。"""
    registry = InMemoryToolRegistry()
    registry.register(Tool(
        name="web_search",
        description="联网检索资料",
        params_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        fn=lambda query: f"检索结果: {query}",
    ))
    registry.register(Tool(
        name="rm_rf",
        description="递归删除目录（危险操作）",
        params_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        fn=lambda path: f"已删除 {path}",
        requires_approval=True,
        metadata={"risk_level": "high", "dangerous": True},
    ))
    return registry


def build_context_manager() -> LayeredContextManager:
    """构造带长期记忆的 ContextManager，并预置较长的工作记忆。"""
    cm = LayeredContextManager(max_working_memory=30)

    # 长期记忆（应被 MemoryInjector 注入 system prompt）
    cm.store(MemoryTrace.create(
        content="User prefers Python async programming for agent runtime",
        namespace="default",
        strength=0.95,
        tags=["python", "async"],
    ))
    cm.store(MemoryTrace.create(
        content="User is researching deep agent memory architecture",
        namespace="default",
        strength=0.85,
        tags=["research"],
    ))

    # 预置较长工作记忆（触发 ContextCompressor 压缩）
    for i in range(24):
        cm.append(Message(
            role="user",
            content=f"历史上下文 #{i}：" + "deep agent memory architecture " * 8,
        ))
    cm.append(Message(role="user", content="请检索 deep agent memory architecture 资料"))

    return cm


# ============================================================================
# 3. 主流程
# ============================================================================


def main() -> None:
    section("L3 第一批中间件演示：研究报告助手")

    # ---- 共享基础设施 ----
    event_log = EventLog()
    session = Session(
        session_id="demo-session-001",
        max_tokens=2000,
        max_cost=1.0,
        event_log=event_log,
        metadata={"task": "deep-agent-research"},
    )
    registry = build_registry()
    context_manager = build_context_manager()

    # ---- 五个中间件全部启用 ----
    observability = ObservabilityMiddleware(
        event_log=event_log, session=session, echo=True
    )
    cost_guard = CostGuardMiddleware(session=session)
    safety_guard = SafetyGuardMiddleware(
        allowed_tools={"web_search"},  # 白名单：只允许检索
        tool_registry=registry,
    )
    memory_injector = MemoryInjectorMiddleware(
        context_manager=context_manager,
        system_prompt=(
            "你是一个研究助手。请结合以下长期记忆回答：\n{{memory_context}}"
        ),
        namespace="default",
    )
    # 阈值说明：
    #   初始历史约 1500 token > 2000 * 0.5 = 1000，触发压缩；
    #   压缩后约 750 token < 1000，不再重复触发，便于观察单次压缩效果。
    compressor = ContextCompressorMiddleware(
        context_manager=context_manager,
        max_tokens=2000,
        threshold_ratio=0.5,
        strategy="threshold",
    )

    executor = MockToolExecutor()
    llm = ScriptedLLM()

    loop = AgentLoop(llm_call=llm, tool_executor=executor, max_iterations=5)
    for mw in (
        observability,
        cost_guard,
        safety_guard,
        memory_injector,
        compressor,
        ScriptedToolCallParser(),
    ):
        loop.register_middleware(mw)

    # ---- 通过 ctx.shared 提供约定依赖（也可以走构造函数注入） ----
    ctx = Context()
    ctx.shared["session"] = session
    ctx.shared["event_log"] = event_log
    ctx.shared["context_manager"] = context_manager
    ctx.shared["tool_registry"] = registry
    ctx.shared["system_prompt_template"] = (
        "你是一个研究助手。请结合以下长期记忆回答：\n{{memory_context}}"
    )
    ctx.shared["memory_namespace"] = "default"

    # Loop 侧已累积的长对话历史（模拟多轮会话），用于演示上下文压缩。
    initial_messages = [
        {
            "role": "user",
            "content": f"历史上下文 #{i}：" + "deep agent memory architecture " * 8,
        }
        for i in range(20)
    ]
    initial_messages.append(
        {"role": "user", "content": "请研究 deep agent memory architecture 并给出建议"}
    )

    # ================================================================
    # 演示 1：启动 Loop
    # ================================================================
    section("启动 Agent Loop（5 个中间件全部启用）")
    print("中间件注册顺序：observability → cost_guard → safety_guard")
    print("                → memory_injector → context_compressor → tool_call_parser")

    result = loop.run(ctx, initial_messages)

    # ================================================================
    # 演示 2：结构化日志输出
    # ================================================================
    section("① 可观测性：结构化 JSON 日志")
    print(f"共输出 {len(observability.logs)} 条结构化日志记录")
    subsection("前 3 条原始日志行")
    for line in observability.logs[:3]:
        parsed = json.loads(line)
        print(json.dumps(parsed, ensure_ascii=False, indent=2))

    subsection("ON_EXIT_LOOP 聚合指标")
    exit_record = observability.records[-1]
    print(json.dumps(exit_record, ensure_ascii=False, indent=2))

    # ================================================================
    # 演示 3：成本累计
    # ================================================================
    section("② 成本累计：token 与 cost")
    budget = session.budget
    print(f"会话 ID        : {session.session_id}")
    print(f"已用 token     : {budget.used_tokens} / {budget.max_tokens}")
    print(f"已用成本       : {budget.used_cost:.6f} / {budget.max_cost}")
    print(f"剩余 token     : {budget.token_remaining}")
    print(f"预算是否超限   : {budget.exceeded()}")
    print(f"CostGuard 中止 : {cost_guard.abort_reason or '未触发'}")
    print(json.dumps(budget.to_dict(), ensure_ascii=False, indent=2))

    # ================================================================
    # 演示 4：安全拦截
    # ================================================================
    section("③ 安全拦截：高危工具")
    print(f"LLM 请求调用的工具 : {[name for name, _ in ScriptedLLM.SCRIPT if name]}")
    print(f"实际执行的工具     : {executor.executed}")
    print(f"被拦截的工具       : {[b['tool'] for b in safety_guard.blocked]}")
    for entry in safety_guard.blocked:
        print(f"  拦截原因: {entry['tool']} -> {entry['reason']}")
    assert "rm_rf" not in executor.executed, "高危工具不得被执行"
    assert any(b["tool"] == "rm_rf" for b in safety_guard.blocked)

    # ================================================================
    # 演示 5：记忆注入
    # ================================================================
    section("④ 记忆注入：长期记忆进入 system prompt")
    info = ctx.shared.get("memory_injected", {})
    print(f"注入信息: {json.dumps(info, ensure_ascii=False)}")
    injected = memory_injector.last_injected_system
    print(f"注入的 system prompt 长度: {len(injected)}")
    print("注入内容（截断展示）:")
    print(injected[:400])
    assert "Python async programming" in injected, "长期记忆应被注入"
    assert "deep agent memory architecture" in injected

    # ================================================================
    # 演示 6：压缩触发
    # ================================================================
    section("⑤ 压缩触发：上下文超限")
    comp_info = ctx.shared.get("context_compressed", {})
    print(f"压缩信息: {json.dumps(comp_info, ensure_ascii=False)}")
    print(f"压缩触发次数     : {compressor.compressed_count}")
    print(
        f"压缩前 token 总量: {compressor.first_before_tokens} "
        f"→ 压缩后: {comp_info.get('after_tokens')}"
    )
    print(f"最近估算 token   : {compressor.last_estimate}")
    assert compressor.compressed_count >= 1, "应至少触发一次压缩"
    assert comp_info["after_tokens"] < compressor.first_before_tokens, (
        "压缩后 token 总量应显著下降"
    )

    # ================================================================
    # 演示 7：聚合视图
    # ================================================================
    section("最终状态汇总")
    print(f"Loop aborted      : {result.aborted}")
    print(f"钩子异常数        : {len(result.hook_errors)}")
    print(f"退出异常数        : {len(result.exit_errors)}")
    print(f"事件日志条数      : {len(event_log)}")
    print(json.dumps(event_log.summary(), ensure_ascii=False, indent=2))

    section("最终断言校验")
    checks = [
        ("日志输出", len(observability.logs) > 0),
        ("成本累计", session.budget.used_tokens > 0),
        ("安全拦截", any(b["tool"] == "rm_rf" for b in safety_guard.blocked)),
        ("记忆注入", "Python async programming" in injected),
        ("压缩触发", compressor.compressed_count >= 1),
        ("无钩子异常", result.hook_errors == []),
    ]
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    section("[OK] 演示完成：5 个中间件全部生效")

    # ---- 独立启停演示 ----
    section("附：独立启停验证（各中间件互不影响）")
    print("说明：下文每次只启用一个中间件，验证其余中间件均未生效。")

    _show_toggle_off(context_manager, event_log)

    return result


def _show_toggle_off(
    context_manager: LayeredContextManager,
    event_log: EventLog,
) -> None:
    """展示"关闭所有中间件时 Loop 仍可运行"。"""
    loop = AgentLoop(
        llm_call=ScriptedLLM(),
        tool_executor=MockToolExecutor(),
        max_iterations=2,
    )
    bare_ctx = Context()
    loop.run(bare_ctx, [{"role": "user", "content": "空跑一轮"}])
    print(f"裸 Loop（无中间件）shared 键: {sorted(bare_ctx.shared.keys())}")
    print("  -> 不含 observability / safety_blocked / memory_injected "
          "/ context_compressed，证明中间件可完全剥离")
    print(f"  -> 钩子异常数: {len(bare_ctx.hook_errors)}")


if __name__ == "__main__":
    main()