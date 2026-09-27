"""
[批次 D3] runtime.config —— 统一配置管理与校验。

子模块：
    - errors     : 配置异常类型。
    - schema     : 自包含 JSON Schema 校验器（无第三方依赖）。
    - loader     : YAML 加载 + 环境变量覆盖 + 迁移 + 校验的统一入口。
    - migrations : schema 迁移脚本注册表。

用法::

    from runtime.config import load_config
    from runtime.config.errors import ConfigValidationError

    cfg = load_config()            # 默认加载 config/default.yaml + config/schema.json
    cfg.get("llm.provider")        # "openai"
"""

from __future__ import annotations

from runtime.config import errors, loader, migrations, schema
from runtime.config.loader import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_SCHEMA_PATH,
    ENV_CONFIG_FILE,
    ENV_DEFAULT_CONFIG,
    ENV_PREFIX,
    ENV_SCHEMA_FILE,
    LoadedConfig,
    deep_merge,
    load_config,
    validate_against_schema,
)

__all__ = [
    "errors",
    "schema",
    "loader",
    "migrations",
    "load_config",
    "LoadedConfig",
    "deep_merge",
    "validate_against_schema",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_SCHEMA_PATH",
    "ENV_PREFIX",
    "ENV_CONFIG_FILE",
    "ENV_DEFAULT_CONFIG",
    "ENV_SCHEMA_FILE",
]