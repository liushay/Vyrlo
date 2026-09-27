"""
[L3-1] ConversationRecorderMiddleware — 对话记录中间件。

职责边界（只写工作记忆，不做决策、不写事件日志、不修改消息）：
    - AFTER_LLM  钩子：把 LLM 响应（content + tool_calls）构造为
      Message(role="assistant", ...) 追加到 ContextManager 工作记忆。
    - AFTER_TOOL 钩子：把工具调用结果构造为 Message(role="tool", ...) 追加到
      ContextManager 工作记忆。
    - 恒返回 HookResult.continue_()，不介入流程控制。
    - 不写事件日志（那是 ObservabilityMiddleware 的职责），
      不修改消息，不做安全性/预算判定。

为什么需要本中间件：
    LayeredContextManager.extract_memories()（L5 沉淀）依赖工作记忆中累积的
    对话消息。此前没有任何中间件把 Agent Loop 的 messages / 工具结果写入
    ContextManager 工作记忆，导致 extract_memories 抽取到空列表，L5 沉淀空转。
    本中间件补齐"Actor Loop 消息 → 工作记忆"的写入链路：

        AFTER_LLM  → context_manager.append(assistant 消息)   # 记录模型回复
        AFTER_TOOL → context_manager.append(tool 消息)        # 记录工具结果

    与 MemoryLifecycleMiddleware 的分工：
        ConversationRecorderMiddleware  写工作记忆（入）
        MemoryLifecycleMiddleware       抽工作记忆（出，→ L5 沉淀）

注册钩子：
    - AFTER_LLM  : 记录 assistant 消息
    - AFTER_TOOL : 记录 tool 消息

依赖关系：
    ConversationRecorderMiddleware -> ContextManager（公开方法 append）
    通过构造参数注入，或从 ctx.shared["context_manager"] 解析。

ctx.shared 约定键（写入）：
    "conversation_recorded" :
        {
            "assistant": 累计记录的 assistant 消息条数,
            "tool":      累计记录的 tool 消息条数,
        }

可独立启停：
    未提供 ContextManager 时所有钩子直接放行，不抛异常、不写入 shared。

[L3-1] 本文件为 L3 中间件新增，未修改任何 L1 / L2 代码。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agent_loop import Context, HookResult, Middleware

from runtime.context_manager.interface import Message


class ConversationRecorderMiddleware(Middleware):
    """[L3-1] 对话记录中间件。

    把 LLM 回复与工具结果落入 ContextManager 工作记忆，作为 L5 沉淀的输入。

    Attributes:
        recorded: 总计记录的消息条数（assistant + tool），便于外部断言。
    """

    def __init__(
        self,
        context_manager: Any = None,
        name: str = "conversation_recorder",
    ) -> None:
        """
        Args:
            context_manager: ContextManager 实例（需实现 append）。
                             为 None 时从 ctx.shared["context_manager"] 解析。
            name:            中间件名称。
        """
        super().__init__(name)
        self._context_manager = context_manager
        self.recorded = 0

    # ------------------------------------------------------------------
    # AFTER_LLM —— 记录 assistant 回复
    # ------------------------------------------------------------------

    def after_llm(self, ctx: Context, response: Any) -> HookResult:
        """[L3-1] AFTER_LLM：把 LLM 回复写入工作记忆（assistant 消息）。"""
        content = self._extract_content(response)
        tool_calls = self._extract_tool_calls(response)

        # 无 content 且无 tool_calls 时跳过
        if not content and not tool_calls:
            return HookResult.continue_()

        cm = self._resolve_context_manager(ctx)
        if cm is None:
            # 无 ContextManager：静默放行，可独立启停
            return HookResult.continue_()

        msg = Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls or None,
        )
        self._append(cm, msg)

        state = self._state(ctx)
        state["assistant"] += 1
        self.recorded += 1

        return HookResult.continue_()

    # ------------------------------------------------------------------
    # AFTER_TOOL —— 记录工具调用结果
    # ------------------------------------------------------------------

    def after_tool(
        self, ctx: Context, tool_call: Dict[str, Any], result: Any
    ) -> HookResult:
        """[L3-1] AFTER_TOOL：把工具结果写入工作记忆（tool 消息）。"""
        cm = self._resolve_context_manager(ctx)
        if cm is None:
            return HookResult.continue_()

        msg = Message(
            role="tool",
            content=self._serialize_tool_result(result),
            tool_call_id=self._tool_call_id(tool_call),
        )
        self._append(cm, msg)

        state = self._state(ctx)
        state["tool"] += 1
        self.recorded += 1

        return HookResult.continue_()

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _resolve_context_manager(self, ctx: Context) -> Any:
        """[L3-1] 解析 ContextManager（构造注入 > ctx.shared["context_manager"]）。"""
        if self._context_manager is not None:
            return self._context_manager
        shared = getattr(ctx, "shared", None)
        if isinstance(shared, dict):
            return shared.get("context_manager")
        return None

    def _state(self, ctx: Context) -> dict:
        """[L3-1] 获取（或初始化）ctx.shared["conversation_recorded"] 统计字典。"""
        shared = ctx.shared
        state = shared.get("conversation_recorded")
        if not isinstance(state, dict):
            state = {"assistant": 0, "tool": 0}
            shared["conversation_recorded"] = state
        return state

    @staticmethod
    def _append(cm: Any, msg: Message) -> None:
        """[L3-1] 调用 ContextManager.append，容错处理。"""
        appender = getattr(cm, "append", None)
        if not callable(appender):
            return
        try:
            appender(msg)
        except Exception:
            return

    # ---- 响应解析（兼容 LLMResponse / dict / str / Anthropic content 块） ----

    @staticmethod
    def _extract_content(response: Any) -> str:
        """[L3-1] 从 LLM 响应中提取文本内容。"""
        if response is None:
            return ""
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            if "choices" in response:
                choices = response.get("choices") or []
                if choices:
                    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                    return ConversationRecorderMiddleware._content_to_str(message.get("content"))
            return ConversationRecorderMiddleware._content_to_str(response.get("content"))
        return ConversationRecorderMiddleware._content_to_str(getattr(response, "content", None))

    @staticmethod
    def _content_to_str(content: Any) -> str:
        """[L3-1] 归一化 content 为字符串（兼容 Anthropic content 块数组）。"""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(str(block))
            return "".join(parts)
        return str(content)

    @staticmethod
    def _extract_tool_calls(response: Any) -> Optional[List[Dict[str, Any]]]:
        """[L3-1] 从 LLM 响应中提取工具调用列表。"""
        if isinstance(response, dict):
            tcs = response.get("tool_calls")
            if tcs is None and "choices" in response:
                choices = response.get("choices") or []
                if choices and isinstance(choices[0], dict):
                    tcs = choices[0].get("message", {}).get("tool_calls")
            return tcs if isinstance(tcs, list) else None
        tcs = getattr(response, "tool_calls", None)
        return tcs if isinstance(tcs, list) else None

    @staticmethod
    def _serialize_tool_result(result: Any) -> str:
        """[L3-1] 把工具结果序列化为字符串。"""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            content = result.get("content", result)
            if isinstance(content, str):
                return content
            return _dumps_or_str(content)
        content = getattr(result, "content", None)
        if content is not None:
            if isinstance(content, str):
                return content
            return _dumps_or_str(content)
        return str(result)

    @staticmethod
    def _tool_call_id(tool_call: Any) -> Optional[str]:
        """[L3-1] 提取工具调用 ID。"""
        if isinstance(tool_call, dict):
            return tool_call.get("id") or tool_call.get("tool_call_id")
        return getattr(tool_call, "id", getattr(tool_call, "tool_call_id", None))


def _dumps_or_str(value: Any) -> str:
    """把任意值安全序列化为字符串（JSON 优先，失败退化为 str）。"""
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)