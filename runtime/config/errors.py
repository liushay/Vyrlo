"""
[批次 D3] runtime.config.errors —— 配置加载 / 校验的异常类型。

提供语义明确的异常，使"配置文件错误时给出明确提示"这一验收点
具备稳定的对外契约（错误类型 + 可读消息）。
"""

from __future__ import annotations


class ConfigError(Exception):
    """配置系统的基础异常。"""


class ConfigFileNotFoundError(ConfigError):
    """配置文件不存在。"""


class ConfigParseError(ConfigError):
    """配置文件解析失败（YAML 语法错误等）。"""


class ConfigValidationError(ConfigError):
    """配置内容未通过 schema 校验。"""

    def __init__(self, message: str, errors: list | None = None) -> None:
        super().__init__(message)
        #: 详细校验错误列表，每项为人类可读的字符串。
        self.errors: list = errors or []

    def __str__(self) -> str:
        if self.errors:
            detail = "\n".join(f"  - {e}" for e in self.errors)
            return f"{super().__str__()}\n{detail}"
        return super().__str__()


class ConfigTypeError(ConfigError):
    """环境变量值无法转换为 schema 声明的目标类型。"""


__all__ = [
    "ConfigError",
    "ConfigFileNotFoundError",
    "ConfigParseError",
    "ConfigValidationError",
    "ConfigTypeError",
]