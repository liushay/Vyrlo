"""
[批次 D3] runtime.config.schema —— 自包含的 JSON Schema 校验器。

实现 draft-07 的一个**最小但实用**子集，用于校验合并后的配置字典
（YAML 加载 + 环境变量覆盖之后）。仅依赖标准库，不引入 ``jsonschema``。

支持的关键字：
    - ``type``（含 "integer" / "number" / "array" / "object" / "string" /
      "boolean" / "null"，以及字符串数组形式的联合类型）
    - ``properties`` / ``required`` / ``additionalProperties``
    - ``items``（单个子 schema，应用于全部数组元素）
    - ``enum`` / ``const``
    - ``minimum`` / ``maximum``（number / integer）
    - ``minLength`` / ``maxLength``（string）
    - ``minItems`` / ``maxItems``（array）
    - ``pattern``（string，正则）
    - ``oneOf`` / ``anyOf`` / ``allOf``
    - ``$ref``（仅内部 ``#/...`` JSON Pointer，用于共享片段）
    - ``$defs`` / ``definitions``（``$ref`` 的锚点容器）

校验结果是 ``(is_valid, errors)`` 二元组，其中 ``errors`` 为人类可读的
路径化错误列表（如 ``"llm.provider: 期望是 string 类型"``）。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

#: 用于 ``$ref`` 锚点容器的关键字（二者等价，draft-07 用 definitions）。
_REF_CONTAINERS = ("$defs", "definitions")

#: 受支持的类型关键字列表，用于报错时的可读提示。
_TYPE_NAMES = (
    "null", "boolean", "object", "array", "number", "integer", "string",
)


def validate(instance: Any, schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """校验 ``instance`` 是否匹配 ``schema``。

    Args:
        instance: 待校验的数据（任意 JSON 兼容值）。
        schema:   一个 JSON Schema 对象（dict）。

    Returns:
        ``(is_valid, errors)``。合法时 errors 为空列表。
    """
    validator = _Validator(schema)
    errors = validator.validate(instance)
    return (len(errors) == 0), errors


class _Validator:
    """单次校验会话，持有根 schema 与错误收集器。"""

    def __init__(self, root_schema: Dict[str, Any]) -> None:
        self._root: Dict[str, Any] = root_schema or {}

    def validate(self, instance: Any) -> List[str]:
        errors: List[str] = []
        self._walk(instance, self._root, "$", errors)
        return errors

    # ------------------------------------------------------------------
    # 核心分发
    # ------------------------------------------------------------------

    def _walk(
        self,
        instance: Any,
        schema: Any,
        path: str,
        errors: List[str],
    ) -> None:
        if not isinstance(schema, dict):
            # 裸 schema 视为无约束（宽松透传）
            return

        # $ref 优先展开（可与其它关键字并存，但按惯例 $ref 独占）
        if "$ref" in schema:
            resolved = self._resolve_ref(schema["$ref"])
            if resolved is None:
                errors.append(f"{path}: 无法解析 $ref: {schema['$ref']!r}")
                return
            self._walk(instance, resolved, path, errors)
            return

        # 组合关键字
        for key in ("allOf",):
            if key in schema:
                for sub in schema[key]:
                    self._walk(instance, sub, path, errors)
        if "anyOf" in schema:
            self._walk_any_of(instance, schema["anyOf"], path, errors)
        if "oneOf" in schema:
            self._walk_one_of(instance, schema["oneOf"], path, errors)

        # 类型约束
        self._check_type(instance, schema, path, errors)
        if instance is None:
            return

        # 枚举 / 常量
        if "enum" in schema and instance not in schema["enum"]:
            errors.append(
                f"{path}: 取值 {instance!r} 不在枚举 {schema['enum']!r} 中"
            )
        if "const" in schema and instance != schema["const"]:
            errors.append(f"{path}: 取值应为 {schema['const']!r}，实际为 {instance!r}")

        self._check_numeric(instance, schema, path, errors)
        self._check_string(instance, schema, path, errors)
        self._check_array(instance, schema, path, errors)
        self._check_object(instance, schema, path, errors)

    # ------------------------------------------------------------------
    # 类型检查
    # ------------------------------------------------------------------

    def _check_type(
        self,
        instance: Any,
        schema: Dict[str, Any],
        path: str,
        errors: List[str],
    ) -> None:
        expected = schema.get("type")
        if expected is None:
            return
        types = expected if isinstance(expected, list) else [expected]
        if not any(self._matches(t, instance) for t in types):
            if isinstance(expected, list):
                want = " / ".join(str(t) for t in expected)
            else:
                want = str(expected)
            errors.append(f"{path}: 期望类型为 {want}，实际为 {_type_of(instance)}")

    @staticmethod
    def _matches(type_name: Any, instance: Any) -> bool:
        t = str(type_name)
        if t == "integer":
            return isinstance(instance, int) and not isinstance(instance, bool)
        if t == "number":
            return isinstance(instance, (int, float)) and not isinstance(instance, bool)
        if t == "boolean":
            return isinstance(instance, bool)
        if t == "string":
            return isinstance(instance, str)
        if t == "array":
            return isinstance(instance, list)
        if t == "object":
            return isinstance(instance, dict)
        if t == "null":
            return instance is None
        return False

    # ------------------------------------------------------------------
    # 数值 / 字符串 / 数组 / 对象
    # ------------------------------------------------------------------

    def _check_numeric(
        self,
        instance: Any,
        schema: Dict[str, Any],
        path: str,
        errors: List[str],
    ) -> None:
        if not isinstance(instance, (int, float)) or isinstance(instance, bool):
            return
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: 数值应 >= {schema['minimum']}，实际为 {instance}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: 数值应 <= {schema['maximum']}，实际为 {instance}")

    def _check_string(
        self,
        instance: Any,
        schema: Dict[str, Any],
        path: str,
        errors: List[str],
    ) -> None:
        if not isinstance(instance, str):
            return
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(
                f"{path}: 字符串长度应 >= {schema['minLength']}，实际为 {len(instance)}"
            )
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(
                f"{path}: 字符串长度应 <= {schema['maxLength']}，实际为 {len(instance)}"
            )
        if "pattern" in schema:
            try:
                if not re.search(schema["pattern"], instance):
                    errors.append(
                        f"{path}: 字符串 {instance!r} 不匹配模式 {schema['pattern']!r}"
                    )
            except re.error:
                errors.append(f"{path}: schema 中的 pattern 非法: {schema['pattern']!r}")

    def _check_array(
        self,
        instance: Any,
        schema: Dict[str, Any],
        path: str,
        errors: List[str],
    ) -> None:
        if not isinstance(instance, list):
            return
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: 数组长度应 >= {schema['minItems']}，实际为 {len(instance)}")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: 数组长度应 <= {schema['maxItems']}，实际为 {len(instance)}")
        if "items" in schema:
            for idx, item in enumerate(instance):
                self._walk(item, schema["items"], f"{path}[{idx}]", errors)

    def _check_object(
        self,
        instance: Any,
        schema: Dict[str, Any],
        path: str,
        errors: List[str],
    ) -> None:
        if not isinstance(instance, dict):
            return

        properties = schema.get("properties") or {}
        required = schema.get("required") or []

        for key in required:
            if key not in instance:
                errors.append(f"{path}: 缺少必需字段 {key!r}")

        for key, value in instance.items():
            if key in properties:
                self._walk(value, properties[key], _join(path, key), errors)
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: 不允许额外字段 {key!r}")

        for key in properties:
            if key in instance and "default" in properties[key]:
                # default 只提示不强制（符合 JSON Schema 语义）
                pass

    # ------------------------------------------------------------------
    # 组合关键字
    # ------------------------------------------------------------------

    def _walk_any_of(
        self,
        instance: Any,
        schemas: Any,
        path: str,
        errors: List[str],
    ) -> None:
        if not isinstance(schemas, list):
            return
        for sub in schemas:
            sub_errors: List[str] = []
            self._walk(instance, sub, path, sub_errors)
            if not sub_errors:
                return
        errors.append(f"{path}: 不满足 anyOf 中的任一子 schema")

    def _walk_one_of(
        self,
        instance: Any,
        schemas: Any,
        path: str,
        errors: List[str],
    ) -> None:
        if not isinstance(schemas, list):
            return
        matches = 0
        for sub in schemas:
            sub_errors: List[str] = []
            self._walk(instance, sub, path, sub_errors)
            if not sub_errors:
                matches += 1
        if matches != 1:
            errors.append(
                f"{path}: oneOf 要求恰好匹配 1 个子 schema，实际匹配 {matches} 个"
            )

    # ------------------------------------------------------------------
    # $ref 解析
    # ------------------------------------------------------------------

    def _resolve_ref(self, ref: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(ref, str) or not ref.startswith("#/"):
            return None
        parts = [p.replace("~1", "/").replace("~0", "~") for p in ref[2:].split("/")]
        node: Any = self._root
        for part in parts:
            if not isinstance(node, dict):
                return None
            # 允许 $ref 穿越 $defs / definitions 容器
            if part not in node:
                return None
            node = node[part]
        return node if isinstance(node, dict) else None


def _type_of(instance: Any) -> str:
    if instance is None:
        return "null"
    if isinstance(instance, bool):
        return "boolean"
    if isinstance(instance, int):
        return "integer"
    if isinstance(instance, float):
        return "number"
    if isinstance(instance, str):
        return "string"
    if isinstance(instance, list):
        return "array"
    if isinstance(instance, dict):
        return "object"
    return type(instance).__name__


def _join(path: str, key: str) -> str:
    if path in ("", "$"):
        return f"$.{key}"
    return f"{path}.{key}"


__all__ = ["validate"]