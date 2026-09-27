"""
[批次 D1] runtime/observability —— metrics / traces / logs 三件套。

子模块：
    - metrics  : Prometheus 兼容指标（零依赖内置实现 + 可选官方桥接）。
    - tracing  : OpenTelemetry 兼容 span 树（内置记录器 + 可选 OTel 桥接）。
    - logging  : 结构化 JSON 日志（JsonFormatter / get_json_logger）。

所有能力均为**可选特性**：未安装 ``prometheus_client`` / ``opentelemetry`` 时
自动降级为内置实现或 no-op，不抛异常、不阻断业务。
"""

from __future__ import annotations

from runtime.observability import logging, metrics, tracing

__all__ = ["metrics", "tracing", "logging"]