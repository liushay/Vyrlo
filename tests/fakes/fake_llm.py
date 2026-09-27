"""
[测试] FakeLLMProvider — 可编程 LLM 适配器。

实现 LLMAdapter 抽象接口，支持脚本化返回四种行为：
    - 正常：返回带 content / tool_calls 的 LLMResponse。
    - 空：返回空 content、无 tool_calls 的 LLMResponse。
    - 非法：返回结构非法的 tool_calls（解析器应降级或拒绝而不崩溃）。
    - 异常：第 N 次调用抛异常（模拟网络失败）。

所有返回 / 异常行为由测试通过 ``enqueue`` / ``fail_at`` 编程控制，
不引入任何运行时依赖，不触达外部网络。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Optional

from runtime.llm_adapter.interface import LLMAdapter, LLMResponse, ModelConfig


class FakeLLMProvider(LLMAdapter):
    """可编程 LLM 适配器（实现 LLMAdapter 契约）。"""

    def __init__(self, default_content: str = "") -> None:
        #: FIFO 脚本队列：每次 call 弹出一个响应构造器（或直接 LLMResponse）。
        self.script: List[Any] = []
        #: 记录每次调用的消息列表（便于断言 LLM 实际收到的输入）。
        self.calls: List[List[Dict[str, Any]]] = []
        #: 当脚本耗尽时使用的兜底响应。
        self.default_response: LLMResponse = LLMResponse(
            content=default_content, finish_reason="stop"
        )
        #: 第 N 次（1-based）调用抛异常；None 表示从不抛。
        self.fail_at: Optional[int] = None
        self._call_count = 0

        self._model = "fake-model"
        self._provider = "fake"

    # ------------------------------------------------------------------
    # 编程接口
    # ------------------------------------------------------------------

    def enqueue(self, response: Any) -> "FakeLLMProvider":
        """入队一个响应。``response`` 可为 LLMResponse 或构造 LLMResponse 的 callable。"""
        self.script.append(response)
        return self

    def enqueue_text(self, content: str) -> "FakeLLMProvider":
        """入队一个纯文本响应。"""
        return self.enqueue(
            LLMResponse(content=content, model=self._model, provider=self._provider,
                        usage={"total_tokens": max(1, len(content) // 4)})
        )

    def enqueue_empty(self) -> "FakeLLMProvider":
        """入队一个空响应（空 content、无 tool_calls，正常结束）。"""
        return self.enqueue(
            LLMResponse(content="", finish_reason="stop",
                        model=self._model, provider=self._provider)
        )

    def enqueue_tool_calls(self, tool_calls: List[Dict[str, Any]], content: str = "") -> "FakeLLMProvider":
        """入队一个含工具调用的响应。

        传入的 ``tool_calls`` 为 OpenAI function calling 形状:
            [{"id": "...", "type": "function",
              "function": {"name": "...", "arguments": "{\"...\": ...}"}}]
        """
        return self.enqueue(
            LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason="tool_calls",
                model=self._model,
                provider=self._provider,
                usage={"total_tokens": 10},
            )
        )

    def enqueue_illegal_tool_calls(self) -> "FakeLLMProvider":
        """入队一个结构非法的 tool_calls 响应（解析器应降级为空，不崩溃）。"""
        # raw 设置为一个既不是 OpenAI 也不是 Anthropic 的畸形结构，
        # 使 AutoDetectParser 各检测器都不命中，最终 TextFallback 返回空。
        malformed = LLMResponse(
            content="",
            tool_calls=[{"bogus": "field", "no_function": True}],
            finish_reason="tool_calls",
            model=self._model,
            provider=self._provider,
        )
        return self.enqueue(malformed)

    # ------------------------------------------------------------------
    # LLMAdapter 契约
    # ------------------------------------------------------------------

    def call(
        self,
        messages: List[Dict[str, Any]],
        model_config: Optional[ModelConfig] = None,
        *,
        task_type: Optional[str] = None,
        context: Optional[Any] = None,
    ) -> LLMResponse:
        """记录调用，按脚本或 fail_at 返回。"""
        self._call_count += 1
        self.calls.append(list(messages or []))

        if self.fail_at is not None and self._call_count == self.fail_at:
            raise RuntimeError("FakeLLMProvider: simulated network failure")

        if self.script:
            item = self.script.pop(0)
            if callable(item) and not isinstance(item, LLMResponse):
                item = item()
            if isinstance(item, LLMResponse):
                return item
            # 允许脚本项是普通 dict（方便构造）
            return LLMResponse(**item) if isinstance(item, dict) else LLMResponse(
                content=str(item), model=self._model, provider=self._provider
            )

        return self.default_response

    def stream(
        self,
        messages: List[Dict[str, Any]],
        model_config: Optional[ModelConfig] = None,
        *,
        task_type: Optional[str] = None,
        context: Optional[Any] = None,
    ) -> Iterator[str]:
        resp = self.call(messages, model_config, task_type=task_type, context=context)
        if resp.content:
            yield resp.content

    def count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        return sum(max(1, len(str(m.get("content", ""))) // 4) for m in messages or [])

    def get_cost(
        self, usage: Dict[str, int], model_config: Optional[ModelConfig] = None
    ) -> float:
        return float(usage.get("total_tokens", 0)) * 0.0001


# ----------------------------------------------------------------------
# 便捷构造 OpenAI function calling 形状的 tool_calls
# ----------------------------------------------------------------------

def openai_tool_call(name: str, arguments: Dict[str, Any], call_id: str = "call_1") -> Dict[str, Any]:
    """构造一个 OpenAI function calling 形状的工具调用条目。"""
    import json

    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


__all__ = ["FakeLLMProvider", "openai_tool_call"]