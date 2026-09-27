"""
[批次 D1] runtime/observability/logging —— 结构化 JSON 日志。

设计目标：
    为 Agent 运行时提供**结构化 JSON 日志**的标准化配置。ObservabilityMiddleware
    已经输出了部分 JSON 行日志（one JSON per line），本模块在此基础上补齐：

    1. ``JsonFormatter``：标准 ``logging.Formatter`` 子类，把 ``LogRecord`` 渲染为
       单个 JSON 对象行，稳定保留 ``ts`` / ``level`` / ``logger`` / ``message``，
       并把 ``extra`` 中提供的结构化字段原样并入（不丢失）。
    2. ``get_json_logger(name, stream, level)``：返回一个配置了 ``JsonFormatter``
       的 ``logging.Logger``，开箱即用。
    3. ``StructuredLog``：轻量的结构化日志事件构造器（非必需，便于在钩子中
       直接组装 dict 交给 logger）。

约束：
    - 纯标准库实现，无第三方依赖。
    - 不改变任何业务逻辑，只提供日志输出工具。
    - 与 ObservabilityMiddleware 的 ``_emit`` 输出格式保持兼容
      （每行一个 JSON 对象，含 ``ts`` 字段）。
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, Dict

#: JSON 日志中时间字段的键（与 ObservabilityMiddleware 的 ``ts`` 对齐）。
TS_KEY = "ts"


class JsonFormatter(logging.Formatter):
    """把 ``LogRecord`` 渲染为一行 JSON 结构化日志。

    输出字段：
        - ``ts``     : 记录时间（Unix 秒，浮点，3 位小数），与 Observability 对齐。
        - ``level``  : 日志级别名（INFO / WARNING / ERROR ...）。
        - ``logger`` : logger 名称。
        - ``message``: 格式化后的消息文本。
        - ``extra`` 字段 : 所有通过 ``record.__dict__`` 传入的额外键，
          原样并入（跳过保留键，避免覆盖）。

    用法::

        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.info("llm call done", extra={"model": "gpt-4o", "tokens": 42})
    """

    def format(self, record: logging.LogRecord) -> str:
        """渲染 record 为一行 JSON。"""
        payload: Dict[str, Any] = {
            TS_KEY: round(getattr(record, "created", time.time()), 3),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # 并入 extra 提供的结构化字段（跳过保留键与日志框架内部键）。
        reserved = {
            TS_KEY,
            "level",
            "logger",
            "message",
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "taskName",
        }
        for key, value in record.__dict__.items():
            if key in reserved or key.startswith("_"):
                continue
            payload[key] = value

        # 异常信息合并：栈信息作为字段而非破坏 JSON 结构。
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def get_json_logger(
    name: str = "deep-agent",
    stream: Any = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """构造一个输出 JSON 行的 ``logging.Logger``。

    Args:
        name:   logger 名称。
        stream: 输出流，默认 ``sys.stdout``。
        level:  日志级别。

    Returns:
        已配置 ``JsonFormatter`` 的 logger。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # 幂等：避免重复添加 handler（同一 logger 多次调用不叠加重复输出）。
    if getattr(logger, "_json_handler", None) is not None:
        return logger

    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger._json_handler = handler  # type: ignore[attr-defined]
    return logger


class StructuredLog:
    """结构化日志事件构造器（可选糖）。

    便于在钩子中把若干字段组装成一个 dict 后交给 logger 输出，
    与 ``logger.info(msg, extra=...)`` 等价。

    用法::

        entry = StructuredLog(event="llm_call", model="gpt-4o")
        entry.set("tokens", 42)
        logger.info("llm call", extra=entry.as_dict())
    """

    def __init__(self, **fields: Any) -> None:
        self._fields: Dict[str, Any] = dict(fields)

    def set(self, key: str, value: Any) -> "StructuredLog":
        """设置一个结构化字段（链式）。"""
        self._fields[key] = value
        return self

    def as_dict(self) -> Dict[str, Any]:
        """返回字段副本。"""
        return dict(self._fields)

    def to_json(self) -> str:
        """序列化为 JSON 字符串。"""
        return json.dumps(self._fields, ensure_ascii=False, default=str)


__all__ = [
    "JsonFormatter",
    "get_json_logger",
    "StructuredLog",
    "TS_KEY",
]