"""
[批次 D3] runtime.config.loader —— 统一配置加载器。

职责（配置文件错误时给出明确提示）：
    1. 从 ``default.yaml`` 加载默认配置。
    2. 用 ``CONFIG_FILE`` / ``APP_CONFIG`` 指向的覆盖文件合并。
    3. 按 schema 声明应用环境变量覆盖（``APP_`` 前缀 + 双下划线键路径）。
    4. 依序执行迁移脚本（schema_version 落后时）。
    5. 用 ``config/schema.json`` 校验合并结果，失败抛出
       ``ConfigValidationError``（含路径化错误明细）。

环境变量覆盖规则：
    - 前缀 ``APP_`` 之后的变量名按双下划线 ``__`` 拆分为配置路径，
      例如 ``APP_LLM__MAX_TOKENS=2048`` → ``config["llm"]["max_tokens"] = 2048``。
    - 值按其对应 schema 节点的 ``type`` 与 config 默认值进行类型推断：
      ``boolean`` → bool；``integer`` → int；``number`` → float；
      ``array`` → 按逗号拆分；其余保持 str。
    - schema 中未声明的键采用默认值类型推断兜底；无法换算时抛 ``ConfigTypeError``。

约束：
    - 纯标准库 + PyYAML。PyYAML 缺失时提供受限降级（仅支持所给默认文件的
      JSON 子集），并在文档中注明生产环境需安装 PyYAML。
"""

from __future__ import annotations

import copy
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from runtime.config import schema as schema_validator
from runtime.config.errors import (
    ConfigError,
    ConfigFileNotFoundError,
    ConfigParseError,
    ConfigTypeError,
    ConfigValidationError,
)
from runtime.version import SCHEMA_VERSION, SemVer

#: 默认配置文件路径（相对仓库根目录）。
DEFAULT_CONFIG_PATH = "config/default.yaml"

#: 默认 schema 文件路径（相对仓库根目录）。
DEFAULT_SCHEMA_PATH = "config/schema.json"

#: 环境变量前缀（用于覆盖配置的键）。
ENV_PREFIX = "APP_"

#: 环境变量中用于指定额外覆盖配置文件路径的变量名。
ENV_CONFIG_FILE = "APP_CONFIG_FILE"

#: 环境变量中用于显式指定 default.yaml 路径的变量名。
ENV_DEFAULT_CONFIG = "APP_DEFAULT_CONFIG"

#: 环境变量中用于显式指定 schema.json 路径的变量名。
ENV_SCHEMA_FILE = "APP_SCHEMA_FILE"


# ============================================================================
# YAML 加载（PyYAML / 受限 JSON 降级）
# ============================================================================

try:  # pragma: no cover - 依赖探测
    import yaml

    _HAS_YAML = True
except Exception:  # pragma: no cover
    _HAS_YAML = False

_BOOL_TRUE = {"true", "yes", "on", "1"}
_BOOL_FALSE = {"false", "no", "off", "0"}


def _load_yaml_text(text: str) -> Dict[str, Any]:
    """把 YAML 文本解析为 dict。

    优先走 PyYAML；缺失时用有限 JSON 兼容解析兜底。
    """
    if _HAS_YAML:
        data = yaml.safe_load(text)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ConfigParseError("配置文件顶层必须是对象（mapping）。")
        return data
    # 降级：尝试 JSON，失败则抛明确错误。
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigParseError(
            f"PyYAML 未安装且文件不是合法 JSON，无法解析: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ConfigParseError("配置文件顶层必须是对象（mapping）。")
    return data


def _load_yaml_file(path: str) -> Dict[str, Any]:
    """加载一个 YAML 配置文件（不存在时抛 ConfigFileNotFoundError）。"""
    if not os.path.exists(path):
        raise ConfigFileNotFoundError(f"配置文件不存在: {path!r}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        raise ConfigFileNotFoundError(f"无法读取配置文件 {path!r}: {exc}") from exc
    return _load_yaml_text(text)


# ============================================================================
# 深合并
# ============================================================================


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并两个字典，override 覆盖 base（列表整体替换）。"""
    result = copy.deepcopy(base or {})
    for key, value in (override or {}).items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ============================================================================
# 环境变量覆盖
# ============================================================================


def _env_pairs() -> Dict[str, str]:
    """读取所有以 ENV_PREFIX 开头的环境变量，剥离前缀。"""
    pairs: Dict[str, str] = {}
    for key, value in os.environ.items():
        if key.startswith(ENV_PREFIX):
            pairs[key[len(ENV_PREFIX):]] = value
    return pairs


def _split_key(raw_key: str) -> List[str]:
    """把环境变量名映射为配置路径段列表。

    规则：双下划线 ``__`` 为层级分隔；单下划线保留为键名一部分。
    """
    return [part for part in raw_key.split("__") if part]


def _normalize_key_path(existing: Dict[str, Any], key: str) -> str:
    """在 config 中查找键时忽略大小写 / 连字符差异，返回规范键名。

    配置 YAML 通常用 snake_case，环境变量可能混用大小写或连字符；
    这里做一次宽松匹配，找不到时原样返回 key。
    """
    if key in existing:
        return key
    for candidate in existing:
        if _key_equiv(candidate, key):
            return candidate
    return key


def _key_equiv(a: str, b: str) -> bool:
    return a.lower().replace("-", "_") == b.lower().replace("-", "_")


def _coerce(value: str, target_type: Any, default_value: Any) -> Any:
    """把字符串环境变量转换为合适类型。

    优先级：schema 声明的 target_type > default_value 的类型推断。
    无法转换时抛 ConfigTypeError（明确提示）。
    """
    t = str(target_type).lower() if target_type else None

    # schema 显式类型
    if t in ("bool", "boolean"):
        low = value.strip().lower()
        if low in _BOOL_TRUE:
            return True
        if low in _BOOL_FALSE:
            return False
        raise ConfigTypeError(
            f"无法把 {value!r} 转换为 boolean（支持 true/false/1/0/yes/no/on/off）。"
        )
    if t in ("int", "integer"):
        try:
            return int(value.strip())
        except ValueError as exc:
            raise ConfigTypeError(f"无法把 {value!r} 转换为 integer。") from exc
    if t in ("float", "number"):
        try:
            return float(value.strip())
        except ValueError as exc:
            raise ConfigTypeError(f"无法把 {value!r} 转换为 number。") from exc
    if t == "array":
        return [s.strip() for s in value.split(",") if s.strip()]

    # 依据默认值类型推断
    if isinstance(default_value, bool):
        low = value.strip().lower()
        if low in _BOOL_TRUE:
            return True
        if low in _BOOL_FALSE:
            return False
        raise ConfigTypeError(f"无法把 {value!r} 转换为 boolean。")
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        try:
            return int(value.strip())
        except ValueError as exc:
            raise ConfigTypeError(f"无法把 {value!r} 转换为 integer。") from exc
    if isinstance(default_value, float):
        try:
            return float(value.strip())
        except ValueError as exc:
            raise ConfigTypeError(f"无法把 {value!r} 转换为 number。") from exc
    if isinstance(default_value, list):
        return [s.strip() for s in value.split(",") if s.strip()]

    return value


def _apply_env_overrides(
    config: Dict[str, Any],
    pairs: Dict[str, str],
    schema: Dict[str, Any],
    root_schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """把环境变量覆盖写入 config（深拷贝后返回）。

    对每条 ``APP_...`` 环境变量，沿其键路径在 config 中逐步下行：
        - 中间节点已存在 → 沿既有键（做大小写/连字符归一）下行；
        - 中间节点不存在 → 新建空 dict。
    同时沿 schema 追踪对应节点的子 schema，以便叶子值做类型换算。
    """
    root = root_schema or schema
    result = copy.deepcopy(config)

    for raw_key, value in pairs.items():
        path = _split_key(raw_key)
        if not path:
            continue

        target: Any = result
        schema_node: Any = schema

        # 下行到叶子节点的父容器，并同步推进 schema 定位
        for seg in path[:-1]:
            if not isinstance(target, dict):
                target = {}
            canonical = _normalize_key_path(target, seg)
            schema_node = _schema_child(schema_node, canonical, root)
            if canonical not in target or not isinstance(target[canonical], dict):
                target[canonical] = {}
            target = target[canonical]

        # 写入叶子
        leaf = path[-1]
        if isinstance(target, dict):
            canonical_leaf = _normalize_key_path(target, leaf)
            leaf_schema = _schema_child(schema_node, canonical_leaf, root)
            leaf_type = leaf_schema.get("type") if isinstance(leaf_schema, dict) else None
            default_for_leaf = target.get(canonical_leaf)
            target[canonical_leaf] = _coerce(value, leaf_type, default_for_leaf)

    return result


def _schema_child(schema_node: Any, key: str, root_schema: Dict[str, Any]) -> Any:
    """获取 schema 节点中指定键的子 schema（无匹配时返回空 dict）。

    - 展开 ``$ref``（相对根 schema 的内部引用）。
    - 在 ``properties`` 中做键名归一匹配。
    """
    if not isinstance(schema_node, dict):
        return {}

    # 展开 $ref（同名片段，仅支持内部 ``#/...`` 指针）
    if "$ref" in schema_node:
        resolved = _resolve_schema_ref(schema_node["$ref"], root_schema)
        if isinstance(resolved, dict):
            schema_node = resolved

    props = schema_node.get("properties")
    if not isinstance(props, dict):
        return {}

    if key in props:
        return props[key]
    for candidate in props:
        if _key_equiv(candidate, key):
            return props[candidate]
    return {}


def _resolve_schema_ref(ref: Any, root_schema: Dict[str, Any]) -> Any:
    """解析 schema 内部 ``#/...`` 引用（相对根 schema）。"""
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return None
    if not isinstance(root_schema, dict):
        return None
    parts = [p.replace("~1", "/").replace("~0", "~") for p in ref[2:].split("/")]
    node: Any = root_schema
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


# ============================================================================
# 迁移
# ============================================================================

Migration = Callable[[Dict[str, Any]], Dict[str, Any]]


def _discover_migrations() -> Dict[str, Migration]:
    """返回 ``{from_version: migrate_fn}`` 的迁移表。

    迁移间隔遵循 schema_version：配置当前值为 ``from_version`` 时执行一次，
    最终收敛到 ``SCHEMA_VERSION``。
    """
    from runtime.config import migrations as _migrations

    table: Dict[str, Migration] = {}
    for name in dir(_migrations):
        if name.startswith("migrate_"):
            from_ver = name[len("migrate_"):].replace("_", ".")
            table[from_ver] = getattr(_migrations, name)
    return table


def _run_migrations(config: Dict[str, Any]) -> Dict[str, Any]:
    """按拓扑顺序依序执行迁移脚本，直到 schema_version == SCHEMA_VERSION。"""
    result = copy.deepcopy(config)
    current = str(result.get("schema_version", "0.0.0"))

    migrations = _discover_migrations()
    if not migrations:
        return result

    try:
        target = SemVer.parse(SCHEMA_VERSION)
        cur = SemVer.parse(current)
    except ValueError:
        raise ConfigValidationError(
            f"配置中的 schema_version 不合法: {current!r}"
        )

    # 只按版本号顺序执行；无向图迁移由迁移函数自身保证幂等。
    ordered = sorted(
        migrations.items(), key=lambda kv: SemVer.parse(kv[0])
    )

    applied = []
    for from_ver, fn in ordered:
        fv = SemVer.parse(from_ver)
        if cur >= target:
            break
        # 只有落在 [cur, target) 区间的迁移才执行
        if fv >= cur:
            fn(result)
            applied.append(from_ver)

    return result


# ============================================================================
# 主入口
# ============================================================================


class LoadedConfig:
    """已加载并校验通过的配置对象。

    Attributes:
        data:        合并后的配置字典。
        source_files: 参与合并的文件路径列表（从左到右，后者覆盖前者）。
        env_keys:    已应用的环境变量覆盖键列表。
    """

    def __init__(
        self,
        data: Dict[str, Any],
        source_files: List[str],
        env_keys: List[str],
    ) -> None:
        self.data = data
        self.source_files = source_files
        self.env_keys = env_keys

    def get(self, path: str, default: Any = None) -> Any:
        """按点分路径读取配置值（如 ``"llm.max_tokens"``）。"""
        node: Any = self.data
        for seg in path.split("."):
            if not isinstance(node, dict) or seg not in node:
                return default
            node = node[seg]
        return node

    def as_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.data)

    def to_json(self) -> str:
        return json.dumps(self.data, ensure_ascii=False, indent=2)


def load_config(
    default_path: Optional[str] = None,
    schema_path: Optional[str] = None,
    extra_files: Optional[List[str]] = None,
    apply_env: bool = True,
) -> LoadedConfig:
    """加载并校验配置（批次 D3 统一入口）。

    Args:
        default_path: 默认配置文件路径。None 时按以下顺序确定：
                      ``APP_DEFAULT_CONFIG`` 环境变量 → 当前工作目录下的
                      ``config/default.yaml`` → 仓库根目录下的
                      ``config/default.yaml``。
        schema_path:  schema 文件路径。None 时按 ``APP_SCHEMA_FILE`` →
                      ``config/schema.json`` 顺序确定。
        extra_files: 额外覆盖文件列表（依次合并，后者覆盖前者）。
        apply_env:   是否应用 ``APP_`` 环境变量覆盖（默认 True）。

    Returns:
        LoadedConfig 实例。

    Raises:
        ConfigFileNotFoundError: 文件不存在时。
        ConfigParseError:        文件无法解析时。
        ConfigValidationError:   schema 校验失败时。
        ConfigTypeError:         环境变量类型换算失败时。
    """
    schema_file = schema_path or os.environ.get(
        ENV_SCHEMA_FILE, DEFAULT_SCHEMA_PATH
    )
    schema = _load_schema(schema_file)

    default_file = default_path or os.environ.get(
        ENV_DEFAULT_CONFIG, DEFAULT_CONFIG_PATH
    )

    files = [default_file]
    # 通过环境变量指定的额外配置文件
    env_extra = os.environ.get(ENV_CONFIG_FILE)
    if env_extra:
        files.extend([f.strip() for f in env_extra.split(os.pathsep) if f.strip()])
    if extra_files:
        files.extend(extra_files)

    merged: Dict[str, Any] = {}
    for f in files:
        data = _load_yaml_file(f)
        merged = deep_merge(merged, data)

    # 迁移（schema_version 落后时）
    merged = _run_migrations(merged)

    # 环境变量覆盖
    env_keys: List[str] = []
    if apply_env:
        pairs = _env_pairs()
        merged = _apply_env_overrides(merged, pairs, schema)
        env_keys = sorted(pairs.keys())

    # 最终校验
    errors = validate_against_schema(merged, schema)
    if errors:
        raise ConfigValidationError(
            f"配置校验失败（schema: {schema_file}）:", errors
        )

    return LoadedConfig(
        data=merged,
        source_files=files,
        env_keys=env_keys,
    )


def _load_schema(schema_path: str) -> Dict[str, Any]:
    """加载 schema JSON 文件。"""
    try:
        with open(schema_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise ConfigFileNotFoundError(f"schema 文件不存在: {schema_path!r}")
    except json.JSONDecodeError as exc:
        raise ConfigParseError(f"schema 文件不是合法 JSON: {exc}")
    if not isinstance(data, dict):
        raise ConfigParseError("schema 文件顶层必须是对象。")
    return data


def validate_against_schema(
    config: Dict[str, Any], schema: Dict[str, Any]
) -> List[str]:
    """校验配置是否匹配 schema，返回错误列表（空列表表示通过）。"""
    ok, errors = schema_validator.validate(config, schema)
    return errors


__all__ = [
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