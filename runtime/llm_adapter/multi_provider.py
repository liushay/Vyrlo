"""
MultiProviderAdapter — LLMAdapter 的默认多提供商实现。

功能：
- 管理多个 Provider 插件
- 按任务类型路由到不同的模型配置
- 重试与降级（主模型失败 → 备选模型）
- 成本追踪写入 Context
- 上下文窗口管理（超限时截断）
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from runtime.llm_adapter.interface import (
    LLMAdapter,
    LLMResponse,
    ModelConfig,
    Provider,
)
from runtime.llm_adapter.window_sizes import resolve_window_size
from runtime.resilience.circuit_breaker import CircuitBreaker
from runtime.resilience.fallback import FallbackError, FallbackPolicy
from runtime.resilience.retry import RetryPolicy

logger = logging.getLogger(__name__)


# ============================================================================
# 内置 Provider 实现
# ============================================================================


class OpenAIProvider(Provider):
    """
    OpenAI 兼容 API 提供商。

    支持 OpenAI 官方 API 和兼容 OpenAI 接口的自定义端点
    （如 vLLM、Ollama、DeepSeek、本地模型等）。
    """

    # 定价表（美元 / 1M tokens）
    _PRICING: Dict[str, Tuple[float, float]] = {
        # (prompt_price, completion_price) per 1M tokens
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4-turbo": (10.00, 30.00),
        "gpt-4": (30.00, 60.00),
        "gpt-3.5-turbo": (0.50, 1.50),
        "o1-preview": (15.00, 60.00),
        "o1-mini": (1.10, 4.40),
        "deepseek-chat": (0.14, 0.28),
        "deepseek-coder": (0.14, 0.28),
    }

    @property
    def provider_name(self) -> str:
        return "openai"

    def call(self, messages: List[Dict[str, Any]], config: ModelConfig) -> LLMResponse:
        try:
            import openai

            client = openai.OpenAI(
                api_key=os.getenv(config.api_key_env, "sk-placeholder"),
                base_url=config.base_url or None,
            )

            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                **config.extra,
            )

            choice = response.choices[0]
            message = choice.message

            # 解析工具调用
            tool_calls: List[Dict[str, Any]] = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    })

            # 提取 usage
            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            finish_reason = choice.finish_reason or "stop"
            if tool_calls:
                finish_reason = "tool_calls"

            cost = self.get_cost(usage)

            return LLMResponse(
                content=message.content or "",
                tool_calls=tool_calls,
                usage=usage,
                model=config.model,
                provider=self.provider_name,
                cost=cost,
                finish_reason=finish_reason,
                raw=response.model_dump() if hasattr(response, "model_dump") else str(response),
            )

        except ImportError:
            logger.error("openai 库未安装。请运行: pip install openai")
            return LLMResponse(
                content="",
                provider=self.provider_name,
                finish_reason="error",
                cost=0.0,
            )
        except Exception as exc:
            logger.warning("OpenAI 调用失败: %s", exc)
            raise

    def stream(
        self, messages: List[Dict[str, Any]], config: ModelConfig
    ) -> Iterator[str]:
        try:
            import openai

            client = openai.OpenAI(
                api_key=os.getenv(config.api_key_env, "sk-placeholder"),
                base_url=config.base_url or None,
            )

            stream = client.chat.completions.create(
                model=config.model,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                stream=True,
                **config.extra,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except ImportError:
            logger.error("openai 库未安装。请运行: pip install openai")
        except Exception as exc:
            logger.warning("OpenAI 流式调用失败: %s", exc)

    def count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        估算 token 数。

        使用简单的字符数/4 估算。精确计算需 tiktoken 库。
        """
        try:
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base")
            total = 0
            for msg in messages:
                for key, value in msg.items():
                    if isinstance(value, str):
                        total += len(encoding.encode(value))
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                total += len(encoding.encode(str(item)))
                total += 4  # 每条消息的格式开销
            return total + 2  # 响应引导开销
        except ImportError:
            # 回退：字符数 / 4 粗略估算
            total_chars = sum(
                len(str(v)) if isinstance(v, str) else len(str(v))
                for msg in messages
                for v in msg.values()
            )
            return total_chars // 4 + len(messages) * 4

    def get_cost(self, usage: Dict[str, int]) -> float:
        if not usage:
            return 0.0

        model_key = self._find_pricing_key(usage)
        prompt_price, completion_price = self._PRICING.get(
            model_key, (1.0, 3.0)  # 默认定价
        )

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        cost = (
            prompt_tokens / 1_000_000 * prompt_price
            + completion_tokens / 1_000_000 * completion_price
        )
        return round(cost, 6)

    def _find_pricing_key(self, usage: Dict[str, int]) -> str:
        """在定价表中查找匹配的模型定价。"""
        model_name = str(usage.get("model", ""))
        for key in self._PRICING:
            if key in model_name.lower():
                return key
        return ""


# ============================================================================
# Anthropic 消息格式转换
# ============================================================================


def _openai_tool_call_to_anthropic_block(tc: Dict[str, Any]) -> Dict[str, Any]:
    """
    将单个工具调用转换为 Anthropic tool_use content 块。

    兼容两种存储形式：
    - OpenAI 嵌套：{"id", "type": "function", "function": {"name", "arguments"}}
    - Anthropic 扁平：{"id", "type": "tool_use", "name", "input"}

    arguments / input 若为 JSON 字符串会被解析为 dict（Anthropic 要求 input 为对象）。
    """
    if not isinstance(tc, dict):
        return {"type": "tool_use", "id": "", "name": "", "input": {}}

    fn = tc.get("function") or {}
    name = tc.get("name") or fn.get("name") or ""
    raw_input = tc.get("input", tc.get("args", fn.get("arguments", {})))

    # arguments 通常是 JSON 字符串，Anthropic 要求 input 为 JSON 对象
    if isinstance(raw_input, str):
        try:
            raw_input = json.loads(raw_input) if raw_input.strip() else {}
        except (json.JSONDecodeError, TypeError):
            raw_input = {}
    if not isinstance(raw_input, dict):
        raw_input = {}

    return {
        "type": "tool_use",
        "id": tc.get("id", ""),
        "name": name,
        "input": raw_input,
    }


def _to_anthropic_blocks(content: Any) -> List[Dict[str, Any]]:
    """
    把消息 content 归一化为 Anthropic content 块数组。

    - list：逐项归一化（非 dict 项包装为 text 块）。
    - 其他值（含 str / None）：包装为单个 {"type": "text", "text": ...} 块。

    始终返回新列表，不修改入参对象。
    """
    if isinstance(content, list):
        blocks: List[Dict[str, Any]] = []
        for block in content:
            if isinstance(block, dict):
                blocks.append(block)
            else:
                blocks.append({"type": "text", "text": str(block)})
        return blocks
    if content is None:
        return []
    return [{"type": "text", "text": str(content)}]


def _merge_adjacent_same_role(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    合并相邻的同角色消息，保证 Anthropic 要求的 user / assistant 严格交替。

    合并规则：
    - 相邻同角色消息：后者的 content 追加到前者的 content 列表中。
    - 若任一侧 content 是字符串，先转为 [{"type": "text", "text": ...}] 再合并。
    - role == "user" 时，合并结果中 tool_result 块统一前置，其余块保持原有
      相对顺序（符合 Anthropic 惯例：工具结果先于用户文本）。
    - 未发生合并的消息保持原样（content 仍为原字符串），单轮场景输出结构不变。

    Args:
        messages: 已转换的 Anthropic 格式消息列表。

    Returns:
        合并后的新列表（元素为新 dict，不修改入参）。
    """
    merged: List[Dict[str, Any]] = []

    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")

        if merged and role and role == merged[-1].get("role"):
            prev = merged[-1]
            blocks = _to_anthropic_blocks(prev.get("content")) + _to_anthropic_blocks(
                msg.get("content")
            )
            if role == "user":
                # tool_result 前置，其余保持相对顺序
                tool_results = [b for b in blocks if b.get("type") == "tool_result"]
                others = [b for b in blocks if b.get("type") != "tool_result"]
                prev["content"] = tool_results + others
            else:
                prev["content"] = blocks
            continue

        merged.append(dict(msg))

    return merged


def _convert_messages_for_anthropic(
    messages: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    将 OpenAI 格式消息列表转换为 Anthropic 原生格式。

    Message.to_dict() 输出的是 OpenAI 格式（tool_calls 为 {"id","type","function"}，
    tool 角色消息带 tool_call_id）。Anthropic Messages API 不接受该格式，直接透传
    会导致工具调用场景失败。本函数负责转换。

    转换规则：
    1. system 消息：内容提取为 system 参数（返回值的第 1 项），不进入 messages。
    2. assistant 消息含 tool_calls：转为 content 块数组，
       每块为 {"type": "tool_use", "id", "name", "input"}；
       若非空文本内容存在，保留为其前置的 {"type": "text", "text": ...} 块。
    3. tool 角色消息：转为
       {"role": "user", "content": [{"type": "tool_result", "tool_use_id", "content"}]}。
       **若 tool_use_id 取不到（既无 tool_call_id 也无 id），该 tool 消息会被跳过
       并记录 logger.warning**——因为 Anthropic 要求 tool_result.tool_use_id 必须
       命中对应的 tool_use.id，产出空 id 会让整个请求被 API 拒绝。跳过是"降级但
       可继续"的选择；如需快速失败，调用方可在转换前自行校验消息完整性。
    4. 其余消息：原样保留。
    5. 向后兼容：若消息的 content 已是 Anthropic 原生 content 块数组（list），
       则原样透传，不做二次转换。
    6. 后处理：合并相邻同角色消息。多轮工具调用会产生连续的 user 消息（每条
       tool 消息各自转为一个 user），而 Anthropic 要求 user / assistant 严格交替，
       否则直接拒绝请求。合并时 tool_result 块置于同一条 user 消息的前部。

    Args:
        messages: OpenAI 格式的消息列表。

    Returns:
        (system_text, anthropic_messages) 二元组。
        system_text 为拼接后的 system 文本（无 system 消息时为空字符串）。
    """
    system_parts: List[str] = []
    converted: List[Dict[str, Any]] = []

    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")

        # ---- 规则 1：system 消息提取为 system 参数 ----
        if role == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                if content:
                    system_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        system_parts.append(str(block.get("text", "")))
                    elif isinstance(block, str):
                        system_parts.append(block)
            continue

        # ---- 规则 3：tool 角色消息 → user + tool_result 块 ----
        if role == "tool":
            tool_use_id = msg.get("tool_call_id") or msg.get("id") or ""
            if not tool_use_id:
                # 空 tool_use_id 会产出非法 tool_result 块，导致整个请求被拒。
                # 默认策略：跳过该消息并记录警告（不阻断其余请求）。
                logger.warning(
                    "跳过缺少 tool_call_id 的 tool 消息（无法构造合法的 tool_result）: %r",
                    str(msg.get("content", ""))[:120],
                )
                continue
            result_content = msg.get("content", "")
            if not isinstance(result_content, str):
                result_content = json.dumps(result_content, ensure_ascii=False)
            converted.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_content,
                }],
            })
            continue

        # ---- 规则 5：已是 Anthropic 原生块数组，原样透传 ----
        if isinstance(msg.get("content"), list):
            converted.append(dict(msg))
            continue

        # ---- 规则 2：assistant 消息含 tool_calls → content 块数组 ----
        tool_calls = msg.get("tool_calls")
        if role == "assistant" and tool_calls:
            blocks: List[Dict[str, Any]] = []
            text = msg.get("content")
            if isinstance(text, str) and text:
                blocks.append({"type": "text", "text": text})
            for tc in tool_calls:
                blocks.append(_openai_tool_call_to_anthropic_block(tc))
            converted.append({"role": "assistant", "content": blocks})
            continue

        # ---- 规则 4：其余消息原样保留 ----
        converted.append(dict(msg))

    # ---- 规则 6：合并相邻同角色消息（Anthropic 要求严格交替） ----
    converted = _merge_adjacent_same_role(converted)

    return "\n".join(p for p in system_parts if p), converted


class AnthropicProvider(Provider):
    """Anthropic Claude API 提供商。"""

    _PRICING: Dict[str, Tuple[float, float]] = {
        "claude-3-5-sonnet": (3.00, 15.00),
        "claude-3-opus": (15.00, 75.00),
        "claude-3-haiku": (0.25, 1.25),
        "claude-3-sonnet": (3.00, 15.00),
    }

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def call(self, messages: List[Dict[str, Any]], config: ModelConfig) -> LLMResponse:
        try:
            import anthropic

            client = anthropic.Anthropic(
                api_key=os.getenv(config.api_key_env, "sk-placeholder"),
                base_url=config.base_url or None,
            )

            # 转换为 Anthropic 原生格式（system 分离 + 工具消息转换）
            system_msg, chat_messages = _convert_messages_for_anthropic(messages)

            response = client.messages.create(
                model=config.model,
                system=system_msg,
                messages=chat_messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                **config.extra,
            )

            # 提取内容
            content_text = ""
            tool_calls = []
            for block in response.content:
                if block.type == "text":
                    content_text += block.text
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "type": "tool_use",
                        "name": block.name,
                        "input": block.input,
                    })

            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

            return LLMResponse(
                content=content_text,
                tool_calls=tool_calls,
                usage=usage,
                model=config.model,
                provider=self.provider_name,
                cost=self.get_cost(usage),
                finish_reason="tool_calls" if tool_calls else "stop",
                raw=str(response),
            )

        except ImportError:
            logger.error("anthropic 库未安装。请运行: pip install anthropic")
            return LLMResponse(provider=self.provider_name, finish_reason="error")
        except Exception as exc:
            logger.warning("Anthropic 调用失败: %s", exc)
            raise

    def stream(
        self, messages: List[Dict[str, Any]], config: ModelConfig
    ) -> Iterator[str]:
        try:
            import anthropic

            client = anthropic.Anthropic(
                api_key=os.getenv(config.api_key_env, "sk-placeholder"),
                base_url=config.base_url or None,
            )

            # 转换为 Anthropic 原生格式（system 分离 + 工具消息转换）
            system_msg, chat_messages = _convert_messages_for_anthropic(messages)

            with client.messages.stream(
                model=config.model,
                system=system_msg,
                messages=chat_messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                **config.extra,
            ) as stream:
                for text in stream.text_stream:
                    yield text

        except ImportError:
            logger.error("anthropic 库未安装。")
        except Exception as exc:
            logger.warning("Anthropic 流式调用失败: %s", exc)

    def count_tokens(self, messages: List[Dict[str, Any]], model: Optional[str] = None) -> int:
        """
        [FIX-N2] 估算 token 数。保持可选 model 参数向后兼容。

        使用字符数/4 粗略估算，不调用 Anthropic API（避免用 dummy key 发注定失败的 HTTP 请求）。
        """
        total_chars = sum(
            len(str(v)) if isinstance(v, str) else len(str(v))
            for msg in messages
            for v in msg.values()
        )
        return total_chars // 4 + len(messages) * 4

    def get_cost(self, usage: Dict[str, int]) -> float:
        """
        [FIX-P1-2] 根据 usage 中携带的 model 名匹配定价，
        不再硬编码 claude-3-haiku。匹配不到时回退到 claude-3-haiku 默认价。
        """
        if not usage:
            return 0.0

        # 按 usage 中的 model 名查找定价
        model_name = str(usage.get("model", ""))
        matched_key = ""
        for key in self._PRICING:
            if key in model_name.lower():
                matched_key = key
                break

        # 回退：从 config 提供商的模型匹配（若 usage 没有 model 字段）
        if not matched_key:
            from runtime.llm_adapter.interface import ModelConfig
            # 尝试从 usage 中的 __model 或直接查找
            fallback_model = str(usage.get("__model", ""))
            for key in self._PRICING:
                if key in fallback_model.lower():
                    matched_key = key
                    break

        prompt_price, completion_price = self._PRICING.get(
            matched_key or "claude-3-haiku",
            (0.25, 1.25),
        )
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cost = (
            prompt_tokens / 1_000_000 * prompt_price
            + completion_tokens / 1_000_000 * completion_price
        )
        return round(cost, 6)


class EchoProvider(Provider):
    """
    Echo 提供商 — 用于测试，回显输入作为输出。

    适合在没有真实 LLM 时进行集成测试。
    """

    @property
    def provider_name(self) -> str:
        return "echo"

    def call(self, messages: List[Dict[str, Any]], config: ModelConfig) -> LLMResponse:
        last_msg = messages[-1]["content"] if messages else ""
        return LLMResponse(
            content=f"[Echo] 收到 {len(messages)} 条消息。最后一条: {last_msg[:200]}",
            model=config.model,
            provider=self.provider_name,
            usage={"prompt_tokens": len(last_msg)//4, "completion_tokens": 10, "total_tokens": len(last_msg)//4+10},
            cost=0.0,
            finish_reason="stop",
        )

    def stream(
        self, messages: List[Dict[str, Any]], config: ModelConfig
    ) -> Iterator[str]:
        last_msg = messages[-1]["content"] if messages else ""
        for word in last_msg.split()[:20]:
            yield word + " "

    def count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        return sum(len(str(m.get("content", ""))) // 4 for m in messages)

    def get_cost(self, usage: Dict[str, int]) -> float:
        return 0.0


# ============================================================================
# MultiProviderAdapter
# ============================================================================


@dataclass
class TaskRoute:
    """任务路由配置。"""
    task_type: str
    model_config: ModelConfig
    fallback_tasks: List[str] = field(default_factory=list)  # 降级任务类型


class MultiProviderAdapter(LLMAdapter):
    """
    多提供商 LLM 适配器。

    功能：
    1. 管理多个 Provider 插件
    2. 任务路由：根据 task_type 选择模型
    3. 重试：调用失败时自动重试
    4. 降级：主模型失败后切换到备用模型
    5. 成本追踪：调用完成后写入 Context
    6. 上下文窗口管理：超限时自动截断消息

    用法：
        adapter = MultiProviderAdapter(default_config=ModelConfig(
            provider="openai", model="gpt-4o-mini",
        ))
        adapter.register_provider(OpenAIProvider())
        adapter.register_provider(AnthropicProvider())

        # 注册任务路由
        adapter.add_task_route(TaskRoute(
            task_type="reasoning",
            model_config=ModelConfig(provider="openai", model="gpt-4o"),
            fallback_tasks=["default"],
        ))

        # 调用
        response = adapter.call(messages, task_type="reasoning", context=ctx)
    """

    def __init__(
        self,
        default_config: Optional[ModelConfig] = None,
        max_retries: int = 2,
        retry_delay: float = 1.0,
        max_context_tokens: int = 128000,
        retry_policy: Optional[RetryPolicy] = None,
        fallback_configs: Optional[List[ModelConfig]] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        event_log: Any = None,
    ) -> None:
        """
        Args:
            default_config:     默认模型配置（task_type 未匹配时使用）。
            max_retries:        最大重试次数（未显式提供 retry_policy 时生效）。
            retry_delay:        重试延迟（秒），随指数增长。
            max_context_tokens: 上下文窗口 token 上限，超限自动截断。
            retry_policy:       可配置的重试策略（指数退避 + 上限）。为 None 时
                                用 max_retries / retry_delay 构造等价策略。
            fallback_configs:   显式降级链（主模型失败后依次尝试）。
                                为 None 时沿用 task_route.fallback_tasks 的降级。
            circuit_breaker:    可选的熔断器。为 None 时不启用（保持默认行为）。
            event_log:          可选事件日志，用于记录重试 / 降级事件。
        """
        self._providers: Dict[str, Provider] = {
            "echo": EchoProvider(),
        }
        self._task_routes: Dict[str, TaskRoute] = {}
        self._default_config = default_config or ModelConfig(
            provider="echo", model="echo-test",
        )
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._max_context_tokens = max_context_tokens

        # [D2] 可配置的韧性策略
        self._retry_policy = retry_policy or RetryPolicy(
            max_attempts=max(1, max_retries + 1),
            base_delay=retry_delay,
        )
        self._fallback_configs: List[ModelConfig] = list(fallback_configs or [])
        self._circuit_breaker = circuit_breaker
        self._event_log = event_log

    # ------------------------------------------------------------------
    # Provider 注册
    # ------------------------------------------------------------------

    def register_provider(self, provider: Provider) -> None:
        """注册提供商插件。"""
        self._providers[provider.provider_name] = provider
        logger.info("已注册提供商: %s", provider.provider_name)

    def unregister_provider(self, name: str) -> Optional[Provider]:
        """注销提供商。"""
        return self._providers.pop(name, None)

    def get_provider(self, name: str) -> Optional[Provider]:
        """获取提供商。"""
        return self._providers.get(name)

    # ------------------------------------------------------------------
    # 任务路由
    # ------------------------------------------------------------------

    def add_task_route(self, route: TaskRoute) -> None:
        """注册任务路由。"""
        self._task_routes[route.task_type] = route
        logger.info(
            "已注册任务路由: %s → %s/%s",
            route.task_type,
            route.model_config.provider,
            route.model_config.model,
        )

    def remove_task_route(self, task_type: str) -> None:
        """移除任务路由。"""
        self._task_routes.pop(task_type, None)

    def _resolve_config(self, task_type: Optional[str] = None) -> ModelConfig:
        """根据任务类型解析模型配置。"""
        if task_type and task_type in self._task_routes:
            return self._task_routes[task_type].model_config
        return self._default_config

    def _get_fallback_config(
        self, config: ModelConfig, task_type: Optional[str] = None
    ) -> Optional[ModelConfig]:
        """获取降级配置。"""
        if task_type and task_type in self._task_routes:
            route = self._task_routes[task_type]
            for fallback_task in route.fallback_tasks:
                if fallback_task in self._task_routes:
                    return self._task_routes[fallback_task].model_config
        return None

    # ------------------------------------------------------------------
    # 调用接口
    # ------------------------------------------------------------------

    def call(
        self,
        messages: List[Dict[str, Any]],
        model_config: Optional[ModelConfig] = None,
        *,
        task_type: Optional[str] = None,
        context: Optional[Any] = None,
    ) -> LLMResponse:
        """单次 LLM 调用，带重试和降级。

        内核使用 runtime.resilience 策略：单个候选按 ``RetryPolicy`` 重试，
        候选间按 ``FallbackPolicy`` 降级（主模型 → 降级链）。全部失败时返回
        明确的错误响应（``finish_reason="error"``），而非让异常向上传播。

        成本追踪对最终成功的响应执行（重试 / 降级成功同样累计）。
        """
        config = model_config or self._resolve_config(task_type)

        # 上下文窗口管理：截断超限消息
        messages = self._manage_context_window(messages, config)

        chain = self._build_fallback_chain(config, task_type)

        candidates = [
            (
                self._provider_key(cfg),
                lambda cfg=cfg: self._call_provider_with_retry(messages, cfg),
            )
            for cfg in chain
        ]

        policy = FallbackPolicy(
            candidates=candidates,
            fallback_on_result=self._is_failed_response,
            event_log=self._event_log,
            labels={"scope": "llm"},
        )

        try:
            response = policy.execute()
        except FallbackError as exc:
            last = exc.last_error
            logger.error("LLM 调用全部失败（重试 + 降级均耗尽）: %s", last)
            return LLMResponse(
                content=f"[Error] LLM 调用失败: {last}",
                model=config.model,
                provider=config.provider,
                finish_reason="error",
                cost=0.0,
            )

        if context is not None:
            self._track_cost(context, response)
        return response

    # ------------------------------------------------------------------
    # [D2] 韧性策略内部实现
    # ------------------------------------------------------------------

    def _provider_key(self, cfg: ModelConfig) -> str:
        return f"{cfg.provider}/{cfg.model}"

    def _call_provider_with_retry(
        self, messages: List[Dict[str, Any]], cfg: ModelConfig
    ) -> LLMResponse:
        provider = self._providers.get(cfg.provider)
        if provider is None:
            raise ValueError(
                f"提供商 '{cfg.provider}' 未注册。"
                f"可用的: {list(self._providers.keys())}"
            )

        def _invoke() -> LLMResponse:
            if self._circuit_breaker is not None:
                return self._circuit_breaker.call(lambda: provider.call(messages, cfg))
            return provider.call(messages, cfg)

        return self._retry_policy.execute(
            _invoke,
            event_log=self._event_log,
            labels={"scope": "llm", "provider": cfg.provider, "model": cfg.model},
        )

    def _build_fallback_chain(
        self, config: ModelConfig, task_type: Optional[str]
    ) -> List[ModelConfig]:
        if self._fallback_configs:
            return [config] + [c for c in self._fallback_configs if c is not config]
        fallback = self._get_fallback_config(config, task_type)
        if fallback is not None:
            return [config, fallback]
        return [config]

    @staticmethod
    def _is_failed_response(cfg: ModelConfig, response: LLMResponse) -> bool:
        """判定响应是否失败（仅 ``finish_reason="error"`` 视为失败，触发降级）。"""
        return bool(getattr(response, "finish_reason", "") == "error")

    def _try_fallback_call(
        self,
        messages: List[Dict[str, Any]],
        config: ModelConfig,
        original_error: Optional[Exception] = None,
    ) -> LLMResponse:
        """尝试降级调用（保留向后兼容；新 call 已内联到 FallbackPolicy）。"""
        try:
            provider = self._providers.get(config.provider)
            if provider is None:
                return LLMResponse(
                    model=config.model,
                    provider=config.provider,
                    finish_reason="error",
                )
            return provider.call(messages, config)
        except Exception as exc:
            logger.error("降级调用也失败: %s", exc)
            return LLMResponse(
                model=config.model,
                provider=config.provider,
                finish_reason="error",
            )

    def stream(
        self,
        messages: List[Dict[str, Any]],
        model_config: Optional[ModelConfig] = None,
        *,
        task_type: Optional[str] = None,
        context: Optional[Any] = None,
    ) -> Iterator[str]:
        """流式调用。"""
        config = model_config or self._resolve_config(task_type)
        messages = self._manage_context_window(messages, config)

        try:
            provider = self._providers.get(config.provider)
            if provider is None:
                yield f"[Error] 提供商 '{config.provider}' 未注册。"
                return
            yield from provider.stream(messages, config)
        except Exception as exc:
            logger.error("流式调用失败: %s", exc)
            yield f"[Error] 调用失败: {exc}"

    # ------------------------------------------------------------------
    # Token 计数与成本
    # ------------------------------------------------------------------

    def count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """使用默认提供商估算 token 数。"""
        provider = self._providers.get(self._default_config.provider)
        if provider:
            return provider.count_tokens(messages)
        return sum(len(str(m.get("content", ""))) // 4 for m in messages)

    def get_cost(
        self,
        usage: Dict[str, int],
        model_config: Optional[ModelConfig] = None,
    ) -> float:
        config = model_config or self._default_config
        provider = self._providers.get(config.provider)
        if provider:
            return provider.get_cost(usage)
        return 0.0

    # ------------------------------------------------------------------
    # 上下文窗口管理
    # ------------------------------------------------------------------

    def _manage_context_window(
        self,
        messages: List[Dict[str, Any]],
        config: ModelConfig,
    ) -> List[Dict[str, Any]]:
        """
        确保消息不超出上下文窗口。

        策略：
        1. 估算消息 token 数
        2. 若未超限则直接返回
        3. 若超限则从最早的消息开始截断（保留 system 消息 + 最近的消息）
        """
        provider = self._providers.get(config.provider)
        if provider is None:
            return messages

        estimated = provider.count_tokens(messages)
        # 显式覆盖优先：config.extra["max_context_tokens"] 高于按模型查表
        explicit_limit = config.extra.get("max_context_tokens")
        if explicit_limit:
            effective_limit = int(explicit_limit)
        else:
            effective_limit = resolve_window_size(
                config.model, self._max_context_tokens
            )

        if estimated <= effective_limit:
            return messages

        logger.warning(
            "消息 token 数 %d 超出上限 %d，执行截断",
            estimated,
            effective_limit,
        )

        # [FIX-E2] 截断策略：保留 system 消息 + 最近的消息
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]

        system_tokens = provider.count_tokens(system_msgs)
        budget_remaining = effective_limit - system_tokens
        if budget_remaining < 50:
            budget_remaining = 50

        # 从最新消息向前累积，保持原始相对顺序（最早的在前）
        kept: List[Dict[str, Any]] = []
        current_tokens = 0
        for msg in reversed(other_msgs):
            t = provider.count_tokens([msg])
            if current_tokens + t > budget_remaining:
                break
            kept.insert(0, msg)
            current_tokens += t

        return system_msgs + kept

    # ------------------------------------------------------------------
    # 成本追踪
    # ------------------------------------------------------------------

    def _track_cost(self, context: Any, response: LLMResponse) -> None:
        """
        [FIX-L4] 将成本信息写入 Context.shared["cost_log"]，仅用于观测记录。

        职责边界（重要）：
            本方法仅用于观测记录，不作为预算依据。它只把每次调用的
            绝对成本追加到 ctx.shared["cost_log"]，不累加到 Session.budget，
            也不参与任何预算判定。

            预算是 CostGuardMiddleware 的唯一职责：它从 EventLog 消费
            LLMCallEvent 事件并累加到 Session.budget。两者数据同源但口径解耦，
            同时启用时不会出现"同一批调用被重复计数"的问题。

        写入 Context.shared 的字段（观测用途）：
        - last_cost:     最近一次调用的绝对成本
        - last_model:    最近一次调用的模型
        - cost_log:      所有调用的绝对成本列表（append-only，不参与预算累加）
        """
        try:
            shared = getattr(context, "shared", None)
            if isinstance(shared, dict):
                # 记录最近一次调用
                shared["last_cost"] = response.cost
                shared["last_model"] = response.model
                # 追加到成本日志
                cost_log = shared.setdefault("cost_log", [])
                cost_log.append({
                    "model": response.model,
                    "provider": response.provider,
                    "cost": response.cost,
                    "tokens": response.usage.get("total_tokens", 0),
                    "prompt_tokens": response.usage.get("prompt_tokens", 0),
                    "completion_tokens": response.usage.get("completion_tokens", 0),
                    "finish_reason": response.finish_reason,
                })
                return
        except (AttributeError, TypeError):
            pass

        try:
            # 尝试调用 set 方法
            if hasattr(context, "set"):
                context.set("last_cost", response.cost, "shared")
                context.set("last_model", response.model, "shared")
        except Exception:
            pass
