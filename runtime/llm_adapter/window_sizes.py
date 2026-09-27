"""
[L3] 模型上下文窗口大小查表。

为 MultiProviderAdapter._manage_context_window 提供"按模型名动态解析
上下文窗口上限"的能力，替代硬编码的 max_context_tokens。

约定：
- 键为模型名前缀（小写），值为 token 上限。
- 前缀匹配时"最长前缀优先"：gpt-4o-mini 先于 gpt-4o，避免短路到较短前缀。
- 无任何前缀匹配时返回调用方提供的 default。
"""

from __future__ import annotations

#: 常见模型前缀 → 上下文窗口 token 上限
WINDOW_SIZES: dict = {
    # OpenAI
    "gpt-4o-mini": 128000,
    "gpt-4o": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    "o1-preview": 128000,
    "o1-mini": 128000,
    # Anthropic
    "claude-3-5-sonnet": 200000,
    "claude-3-opus": 200000,
    "claude-3-haiku": 200000,
    "claude-3-sonnet": 200000,
    # DeepSeek
    "deepseek-chat": 64000,
    "deepseek-coder": 64000,
    # Echo（测试用，极小窗口便于触发截断路径）
    "echo": 4000,
}


def resolve_window_size(model: str, default: int) -> int:
    """按模型名前缀解析上下文窗口大小。

    匹配规则：最长前缀优先（如 ``gpt-4o-mini`` 先于 ``gpt-4o``），
    以保证更具体的模型名命中更精确的窗口配置。

    Args:
        model:   模型名（可为空字符串）。
        default: 无匹配时的回退窗口大小。

    Returns:
        解析出的 token 上限。
    """
    if not model:
        return default

    name = str(model).lower()
    matched_key = ""
    for key in WINDOW_SIZES:
        if key in name and len(key) > len(matched_key):
            matched_key = key

    if matched_key:
        return int(WINDOW_SIZES[matched_key])
    return default