"""
[D012-FIX] ToolResultFeedbackMiddleware — 工具结果回灌到下一轮 messages。

== 缺陷背景（D012，P0/P1）==
B4 基线中 13 个失败样本同源于一个缺陷：工具执行结果**从不进入下一轮 LLM 调用的
messages**。根因（见 ``reconnaissance-B8.md``）：

1. ``agent_loop.py::_run_impl`` 中 ``tool_exec_result`` 只是一个局部变量，
   仅作为 ``AFTER_TOOL`` 的入参传入（``agent_loop.py:932-954``），此后即被丢弃；
2. ``AFTER_TOOL`` 返回值的 ``payload`` **从不被 Loop 读取**（Loop 只读 ``control``，
   ``agent_loop.py:955-961``）；D004 的链式传递**只对 BEFORE_LLM / BEFORE_TOOL 启用**
   （``agent_loop.py:707-714``），跨钩子（AFTER_TOOL → BEFORE_LLM）没有通道；
3. ``ConversationRecorderMiddleware`` 把 tool 消息写进 **ContextManager 工作记忆**
   （``conversation_recorder.py:116-135``），那是与 Loop ``messages`` 互不相通的另一条通道；
4. 跨轮 ``messages`` 仅在 ``BEFORE_LLM`` 阶段由 ``payload["messages"]`` 整体替换
   （``agent_loop.py:865-873``），没有任何一行把工具结果放进去。

== 修复策略（不改 Agent Loop）==
Loop 缺少「AFTER_TOOL payload → messages」的契约，但 ``ctx`` 对象在所有钩子间共享
（``_dispatch_hook`` 只 deepcopy 位置参数，不 copy ``ctx``，见 ``agent_loop.py:727``），
因此用 ``ctx.private`` 作为跨钩子桥即可，**无需修改 agent_loop.py**：

    AFTER_LLM  ── 记录 assistant 消息（content + tool_calls）到 ctx.private 的待发缓冲
         ↓
    AFTER_TOOL ── 记录 tool 消息 {"role":"tool","tool_call_id":...,"content":...} 到同一缓冲
         ↓
    ON_TOOL_ERROR ── 工具执行抛异常时，同样以 tool 消息形式记录 error 内容
         ↓
    BEFORE_LLM ── 弹出缓冲，按「assistant 在前、tool 在后」追加到 messages 末尾，
                  返回 HookResult(payload={"messages": new_messages})，
                  由 Loop 既有契约（agent_loop.py:872-873）采纳并跨轮保持。

== 为什么必须同时回灌 assistant 消息 ==
LLM API（OpenAI / Ollama OpenAI 兼容端点）约定：``tool`` 消息必须**紧跟**在携带
对应 ``tool_call_id`` 的 ``assistant`` 消息之后。只回灌 ``tool`` 消息会构造出
"没有前驱 assistant"的非法对话，被提供商拒绝。因此本中间件把「assistant(tool_calls)
+ 其对应的 tool 结果」作为一个不可分割的块整体回灌。

== 与控制流 / 其他中间件的关系 ==
- 恒返回 ``HookResult.continue_()``（BEFORE_LLM 有回灌时返回 payload，但不改 control）。
- **不**触碰 ContextManager 工作记忆：``ConversationRecorderMiddleware`` 的既有逻辑
  完全保留（工作记忆仍按原样累积，供 L5 沉淀使用）。
- ``priority`` 默认 -100：``_dispatch_hook`` 按 priority 降序执行，负值使本中间件在
  ``BEFORE_LLM`` 链中**最后**运行，从而把工具结果追加到其他消息改写中间件
  （MemoryInjector / ContextCompressor / SkillInjection）产出的 messages **之上**，
  最不容易被后续改写覆盖。

可独立启停：
    无任何工具调用时缓冲为空，所有钩子直接放行，对既有装配零行为影响。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agent_loop import Context, HookResult, Middleware


class ToolResultFeedbackMiddleware(Middleware):
    """[D012-FIX] 把工具执行结果回灌进下一轮 LLM 调用的 messages。

    缓冲结构（存于 ``ctx.private[PENDING_KEY]``，随 ctx 每 run 重置）：

        [
          {
            "assistant": {"role": "assistant", "content": ..., "tool_calls": [...]},
            "entries":   [{"call": <openai tool_call>, "id": ..., "fulfilled": bool}, ...],
            "tools":     [{"role": "tool", "tool_call_id": ..., "content": ...}, ...],
          },
          ...
        ]

    Attributes:
        feedback_rounds:   累计回灌的轮次数（便于外部断言）。
        feedback_messages: 累计回灌的消息条数（assistant + tool）。
    """

    #: ctx.private 约定键：本轮待回灌的 assistant/tool 消息块
    PENDING_KEY = "__tool_result_feedback_pending__"
    #: ctx.private 约定键：回灌轮次计数（per-run）
    ROUNDS_KEY = "__tool_result_feedback_rounds__"

    NAME = "tool_result_feedback"

    def __init__(
        self,
        name: Optional[str] = None,
        priority: int = -100,
        summarizer: Optional[Any] = None,
        summarize_threshold: int = 2000,
    ) -> None:
        """
        Args:
            name:     中间件名称，默认 "tool_result_feedback"。
            priority: 调度优先级。默认 -100，使本中间件在 BEFORE_LLM 链中最后运行，
                      把工具结果追加到其他消息改写中间件的输出之上。
            summarizer:           可选的大工具结果摘要器（[C2]）。
                                  签名 ``(text: str) -> str``，应由**便宜模型**实现，
                                  用于在回灌前压缩超大工具结果、保留关键信息。
                                  为 None 时回退为本地截断（不影响既有行为）。
            summarize_threshold:  工具结果 content 超过该字符数才触发摘要（[C2]）。
                                  默认 2000。此项是"如何用"成本优化项，
                                  非 B10 校准的六个阈值之一（不改变任何校准常量）。
        """
        super().__init__(name or self.NAME, priority=priority)
        self.feedback_rounds = 0
        self.feedback_messages = 0
        # [C2] 大工具结果摘要
        self._summarizer = summarizer
        self._summarize_threshold = max(0, int(summarize_threshold))
        # [C2] 观测：摘要次数与压缩前后字符数
        self.summarized_count = 0
        self.summarized_chars_before = 0
        self.summarized_chars_after = 0

    # ------------------------------------------------------------------
    # ON_ENTER_LOOP —— 显式清空 per-run 缓冲（覆盖同一 ctx 被复用的场景）
    # ------------------------------------------------------------------

    def on_enter_loop(self, ctx: Context) -> HookResult:
        """[D012-FIX] 每次 run 开始时清空待回灌缓冲与计数，避免跨 run 残留。

        正常情况下 Runtime 每次 run 新建 Context，天然隔离；本钩子额外保护
        "同一 ctx 被重复用于多次 run" 的场景（与 D009 的复位策略一致）。
        """
        ctx.private.pop(self.PENDING_KEY, None)
        ctx.private.pop(self.ROUNDS_KEY, None)
        return HookResult.continue_()

    # ------------------------------------------------------------------
    # AFTER_LLM —— 记录本轮的 assistant 消息（含 tool_calls）
    # ------------------------------------------------------------------

    def after_llm(self, ctx: Context, response: Any) -> HookResult:
        """[D012-FIX] 记录 assistant 消息，等待 AFTER_TOOL 补上工具结果。

        只消费 ``ctx.get_tool_calls()``（Loop 驱动工具流程的权威来源），
        与 ``_ToolCallBridgeMiddleware`` 的写入保持一致；无工具调用时不记录，
        返回 CONTINUE，对最终答复轮零行为。

        注意：``response`` 由 Loop deepcopy 后传入，不可依赖其对象身份。
        """
        calls = ctx.get_tool_calls()
        if not calls:
            return HookResult.continue_()

        entries: List[Dict[str, Any]] = []
        for call in calls:
            if not isinstance(call, dict):
                continue
            entries.append({
                "call": self._to_openai_tool_call(call),
                "id": self._call_id(call),
                "fulfilled": False,
            })

        if not entries:
            return HookResult.continue_()

        assistant = {
            "role": "assistant",
            "content": self._extract_content(response),
            "tool_calls": [e["call"] for e in entries],
        }

        self._pending(ctx).append({
            "assistant": assistant,
            "entries": entries,
            "tools": [],
        })
        return HookResult.continue_()

    # ------------------------------------------------------------------
    # AFTER_TOOL —— 记录工具结果（tool 消息）
    # ------------------------------------------------------------------

    def after_tool(
        self, ctx: Context, tool_call: Dict[str, Any], result: Any
    ) -> HookResult:
        """[D012-FIX] 把工具结果记录为 ``{"role": "tool", ...}`` 消息。

        ``result`` 为 ``ToolResult``（``runtime/tool_registry/interface.py``）。
        工具内部异常已在 registry 层被包装为 ``ToolResult(error=True)``，
        因此这里同样覆盖"工具失败"场景：error 文本就在 ``result.content`` 中，
        会以 tool 消息的 content 原样回灌。

        多个工具在同一轮内按 Loop 的遍历顺序依次触发本钩子，故 tool 消息顺序正确。
        """
        block = self._current_block(ctx)
        if block is None:
            return HookResult.continue_()

        call_id = self._call_id(tool_call)
        self._mark_fulfilled(block, call_id)

        block["tools"].append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": self._serialize_tool_result(result),
        })
        return HookResult.continue_()

    # ------------------------------------------------------------------
    # ON_TOOL_ERROR —— 工具执行抛异常时，同样以 tool 消息回灌 error
    # ------------------------------------------------------------------

    def on_tool_error(
        self, ctx: Context, tool_call: Dict[str, Any], exception: Exception
    ) -> HookResult:
        """[D012-FIX] ``_tool_executor`` 直接抛异常（非 ToolResult 包装）时回灌 error。

        与 ``after_tool`` 的分工：registry 正常返回（含 ToolResult(error=True)）走
        ``after_tool``；executor 本身抛出（如沙箱不可用）走本钩子。两条路径都保证
        error 信息以 tool 消息形式进入下一轮上下文。
        """
        block = self._current_block(ctx)
        if block is None:
            return HookResult.continue_()

        call_id = self._call_id(tool_call)
        self._mark_fulfilled(block, call_id)

        block["tools"].append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": f"工具执行异常: {type(exception).__name__}: {exception}",
        })
        return HookResult.continue_()

    # ------------------------------------------------------------------
    # BEFORE_LLM —— 把累积的 assistant + tool 消息追加到 messages
    # ------------------------------------------------------------------

    def before_llm(
        self, ctx: Context, messages: List[Dict[str, Any]]
    ) -> HookResult:
        """[D012-FIX] 弹出待回灌缓冲，按顺序追加到 messages 末尾。

        返回 ``HookResult(payload={"messages": new_messages})``，由 Loop 既有契约
        （``agent_loop.py:872-873``）采纳；由于 Loop 的 ``messages`` 变量跨轮存活，
        本次追加的内容会延续到后续所有轮。

        无待回灌内容时返回 CONTINUE，**不**改写 messages（保证零行为影响）。
        """
        blocks = ctx.private.get(self.PENDING_KEY)
        if not blocks:
            return HookResult.continue_()

        additions = self._build_additions(blocks)
        # 清空缓冲，避免重复回灌
        ctx.private.pop(self.PENDING_KEY, None)

        if not additions:
            return HookResult.continue_()

        new_messages = list(messages or [])
        new_messages.extend(additions)

        self.feedback_rounds += 1
        self.feedback_messages += len(additions)
        ctx.private[self.ROUNDS_KEY] = self.feedback_rounds

        # 观测键：供外部装配 / 测试断言回灌规模
        ctx.shared["tool_result_feedback"] = {
            "rounds": self.feedback_rounds,
            "messages": self.feedback_messages,
            "last_batch": len(additions),
        }
        return HookResult(payload={"messages": new_messages})

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _pending(self, ctx: Context) -> List[Dict[str, Any]]:
        """获取（或初始化）ctx.private 中的待回灌块列表。"""
        pending = ctx.private.get(self.PENDING_KEY)
        if not isinstance(pending, list):
            pending = []
            ctx.private[self.PENDING_KEY] = pending
        return pending

    @staticmethod
    def _current_block(ctx: Context) -> Optional[Dict[str, Any]]:
        """取最近一个尚未完成的回灌块（即本轮 assistant 对应的块）。"""
        pending = ctx.private.get(ToolResultFeedbackMiddleware.PENDING_KEY)
        if isinstance(pending, list) and pending:
            return pending[-1]
        return None

    @staticmethod
    def _mark_fulfilled(block: Dict[str, Any], call_id: Any) -> None:
        """按 tool_call_id 匹配并标记该块内对应的 tool_call 已产出结果。

        优先按 id 精确匹配；无 id（None/空）时退化为"第一个未完成项"，
        以兼容不使用 id 的简化 tool_call 形状。
        """
        entries = block.get("entries") or []
        for entry in entries:
            if not entry.get("fulfilled") and call_id and entry.get("id") == call_id:
                entry["fulfilled"] = True
                return
        for entry in entries:
            if not entry.get("fulfilled"):
                entry["fulfilled"] = True
                return

    def _build_additions(
        self, blocks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """把待回灌块展开为 flatten 的消息列表（assistant 在前，tool 在后）。

        仅发射**确实产出结果**（fulfilled）的 tool_call，避免把被 BEFORE_TOOL
        跳过的工具写进 assistant.tool_calls，构造出非法对话。

        [C2] 回灌前对超大工具结果做摘要（压缩注入量，成功率不降）。
        """
        additions: List[Dict[str, Any]] = []
        for block in blocks:
            tools = block.get("tools") or []
            if not tools:
                continue
            fulfilled = [
                e["call"] for e in (block.get("entries") or []) if e.get("fulfilled")
            ]
            if fulfilled:
                assistant = dict(block.get("assistant") or {})
                assistant["tool_calls"] = fulfilled
                assistant.setdefault("role", "assistant")
                additions.append(assistant)
            for tool_msg in tools:
                additions.append(self._maybe_summarize_tool_msg(tool_msg))
        return additions

    # ---- [C2] 大工具结果摘要 ----

    def _maybe_summarize_tool_msg(
        self, tool_msg: Dict[str, Any]
    ) -> Dict[str, Any]:
        """[C2] 对超大工具结果的 content 做摘要，保留关键信息。

        - 未达到 ``summarize_threshold`` 时原样返回（零行为影响）。
        - 有 ``summarizer``（便宜模型）时调用之；失败或为空结果时回退本地截断。
        - 无 ``summarizer`` 时回退本地截断（首尾各保留一半），保证语义退化安全。

        始终返回新 dict，不原地修改入参。
        """
        if not isinstance(tool_msg, dict):
            return tool_msg

        content = tool_msg.get("content")
        if not isinstance(content, str) or len(content) <= self._summarize_threshold:
            return tool_msg

        before = len(content)
        after = self._summarize_text(content)
        self.summarized_count += 1
        self.summarized_chars_before += before
        self.summarized_chars_after += len(after)

        new_msg = dict(tool_msg)
        new_msg["content"] = after
        return new_msg

    def _summarize_text(self, text: str) -> str:
        """[C2] 摘要一条超长工具结果文本。

        优先使用 ``summarizer``（便宜模型）；不可用、抛异常或返回空时，
        回退本地截断（前 40% + "…[已截断]…" + 后 40%），保证不丢头尾关键信息。
        """
        if self._summarizer is not None:
            try:
                summary = self._summarizer(text)
            except Exception:
                summary = None
            if isinstance(summary, str) and summary.strip():
                return summary.strip()
        # 回退：保留头尾，控制注入量
        half = self._summarize_threshold // 2
        if half <= 0:
            half = 1000
        if len(text) <= half * 2:
            return text
        return f"{text[:half]}\n…[工具结果已截断]…\n{text[-half:]}"

    # ---- tool_call / 结果 的规范化与序列化 ----

    @staticmethod
    def _call_id(tool_call: Any) -> Optional[str]:
        """提取 tool_call id（兼容 dict / 对象）。"""
        if isinstance(tool_call, dict):
            return tool_call.get("id") or tool_call.get("tool_call_id")
        return getattr(tool_call, "id", getattr(tool_call, "tool_call_id", None))

    @staticmethod
    def _to_openai_tool_call(call: Dict[str, Any]) -> Dict[str, Any]:
        """把 Loop 内部 tool_call（{id,name,arguments}）转成 OpenAI 形状。

        产出：``{"id", "type": "function", "function": {"name", "arguments"}}``。
        ``arguments`` 必须是 JSON 字符串（OpenAI / Ollama 兼容端点的约定）。
        """
        args = call.get("arguments", call.get("args", {}))
        args_str = args if isinstance(args, str) else json.dumps(
            args or {}, ensure_ascii=False
        )
        return {
            "id": call.get("id"),
            "type": "function",
            "function": {
                "name": str(call.get("name") or call.get("tool") or ""),
                "arguments": args_str,
            },
        }

    @staticmethod
    def _extract_content(response: Any) -> str:
        """从 LLM 响应中提取文本内容（兼容 LLMResponse / dict / str）。"""
        if response is None:
            return ""
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            if "choices" in response:
                choices = response.get("choices") or []
                if choices and isinstance(choices[0], dict):
                    message = choices[0].get("message", {})
                    if isinstance(message, dict):
                        return str(message.get("content") or "")
            return str(response.get("content") or "")
        content = getattr(response, "content", None)
        return str(content) if content is not None else ""

    @staticmethod
    def _serialize_tool_result(result: Any) -> str:
        """把工具结果序列化为字符串（与 ToolResult.content 保持一致）。

        - 直接字符串 → 原样；
        - dict → 取 ``content`` 字段（缺失则整体 JSON 化）；
        - 对象（含 ToolResult）→ 取 ``content`` 属性；
        - 其余 → ``str(result)``。

        工具失败时 ``result.content`` 已是 error 文本，因此同一条路径即可完成
        "error 信息以 tool 消息形式回灌"。
        """
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            content = result.get("content", result)
            if isinstance(content, str):
                return content
            return ToolResultFeedbackMiddleware._dumps_or_str(content)
        content = getattr(result, "content", None)
        if content is not None:
            if isinstance(content, str):
                return content
            return ToolResultFeedbackMiddleware._dumps_or_str(content)
        return str(result)

    @staticmethod
    def _dumps_or_str(value: Any) -> str:
        """JSON 优先，失败退化为 str。"""
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)


__all__ = ["ToolResultFeedbackMiddleware"]