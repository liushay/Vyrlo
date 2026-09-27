"""
LLMAdapter 核心接口定义。

包含：
- LLMResponse:   标准 LLM 响应结构
- ModelConfig:   模型配置
- Provider:      提供商插件接口
- LLMAdapter:    统一调用接口
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


# ============================================================================
# LLMResponse — 标准响应结构
# ============================================================================


@dataclass
class LLMResponse:
    """
    标准 LLM 响应结构，统一不同提供商的返回格式。

    Attributes:
        content:    文本响应内容（若没有文本则为空字符串）。
        tool_calls: 工具调用列表（若有）。
        usage:      token 使用统计 {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}。
        model:      实际使用的模型名称。
        provider:   提供商名称。
        cost:       预估成本（美元）。
        finish_reason: 结束原因（"stop" | "tool_calls" | "length" | "error"）。
        raw:        原始响应（调试用）。
    """

    content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = ""
    provider: str = ""
    cost: float = 0.0
    finish_reason: str = "stop"
    raw: Any = None

    def has_tool_calls(self) -> bool:
        """是否包含工具调用。"""
        return len(self.tool_calls) > 0


# ============================================================================
# ModelConfig — 模型配置
# ============================================================================


@dataclass
class ModelConfig:
    """
    模型调用配置。

    Attributes:
        provider:   提供商标识（如 "openai"、"anthropic"、"local"）。
        model:      模型名称（如 "gpt-4o"、"claude-3-opus-20240229"）。
        api_key_env: API 密钥的环境变量名。
        base_url:    自定义 API 端点（可选，用于代理或本地模型）。
        max_tokens:  最大生成 token 数。
        temperature: 采样温度。
        top_p:       nucleus 采样参数。
        extra:       提供商特有的额外参数。
    """

    provider: str = "openai"
    model: str = "gpt-4o"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Provider 插件接口
# ============================================================================


class Provider(ABC):
    """
    LLM 提供商插件接口。

    每个提供商（OpenAI、Anthropic、本地模型 等）实现此接口。
    用户新增提供商只需实现此接口的四个方法。

    可用环境：
    - 运行时注入 Context 引用用于成本追踪。
    - 通过构造函数传递配置。
    """

    @abstractmethod
    def call(self, messages: List[Dict[str, Any]], config: ModelConfig) -> LLMResponse:
        """
        单次调用 LLM，返回标准响应。

        Args:
            messages: 消息列表 [{"role": "user/system/assistant/tool", "content": ...}]
            config:   模型配置。

        Returns:
            LLMResponse 实例。
        """
        ...

    @abstractmethod
    def stream(
        self, messages: List[Dict[str, Any]], config: ModelConfig
    ) -> Iterator[str]:
        """
        流式调用 LLM，返回 token 迭代器。

        Args:
            messages: 消息列表。
            config:   模型配置。

        Yields:
            每个增量 token 字符串。
        """
        ...

    @abstractmethod
    def count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        估算消息列表的 token 数。

        注：不同模型的 tokenizer 不同，此方法提供近似值。
        精确计算需要具体的 tokenizer 库。

        Args:
            messages: 消息列表。

        Returns:
            估算的 token 数。
        """
        ...

    @abstractmethod
    def get_cost(self, usage: Dict[str, int]) -> float:
        """
        根据 token 使用量计算成本。

        Args:
            usage: {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}

        Returns:
            预估成本（美元）。
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """返回提供商名称。"""
        ...


# ============================================================================
# LLMAdapter 抽象接口
# ============================================================================


class LLMAdapter(ABC):
    """
    LLM 适配器抽象接口。

    职责：
    - 统一多提供商的调用接口
    - 按任务类型路由到不同模型
    - 重试与降级
    - 成本追踪
    - 流式与非流式统一

    可插拔设计：
    - 用户可实现此接口替换默认的 MultiProviderAdapter。
    - 只需满足 call / stream / count_tokens / get_cost 契约。
    """

    @abstractmethod
    def call(
        self,
        messages: List[Dict[str, Any]],
        model_config: Optional[ModelConfig] = None,
        *,
        task_type: Optional[str] = None,
        context: Optional[Any] = None,
    ) -> LLMResponse:
        """
        单次 LLM 调用。

        Args:
            messages:     消息列表。
            model_config: 模型配置。若为 None 则根据 task_type 路由选择。
            task_type:    任务类型标识（如 "reasoning"、"summarize"、"coding"），
                         用于按任务路由到不同的模型配置。
            context:      Context 对象引用，用于成本追踪写入。

        Returns:
            LLMResponse 实例。
        """
        ...

    @abstractmethod
    def stream(
        self,
        messages: List[Dict[str, Any]],
        model_config: Optional[ModelConfig] = None,
        *,
        task_type: Optional[str] = None,
        context: Optional[Any] = None,
    ) -> Iterator[str]:
        """
        流式 LLM 调用。

        Args:
            messages:     消息列表。
            model_config: 模型配置。
            task_type:    任务类型标识。
            context:      Context 对象引用。

        Yields:
            每个增量 token 字符串。
        """
        ...

    @abstractmethod
    def count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """估算消息的 token 数。"""
        ...

    @abstractmethod
    def get_cost(
        self,
        usage: Dict[str, int],
        model_config: Optional[ModelConfig] = None,
    ) -> float:
        """根据使用量和模型配置计算成本。"""
        ...