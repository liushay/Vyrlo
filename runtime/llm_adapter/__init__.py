"""
LLMAdapter — 统一多提供商的 LLM 调用接口。

核心抽象：
- LLMAdapter:        统一调用接口
- LLMResponse:       标准响应结构
- ModelConfig:       模型配置
- Provider:          提供商插件接口

默认实现：
- MultiProviderAdapter: 多提供商适配器，支持任务路由、重试、降级
"""

from runtime.llm_adapter.interface import (
    LLMAdapter,
    LLMResponse,
    ModelConfig,
    Provider,
)
from runtime.llm_adapter.multi_provider import MultiProviderAdapter

__all__ = [
    "LLMAdapter",
    "LLMResponse",
    "ModelConfig",
    "Provider",
    "MultiProviderAdapter",
]