"""
[批次 D3] serve —— Deep Agent HTTP 服务入口。

装配统一配置 + 依赖组件，启动零第三方依赖的 HTTP 服务，暴露：
    - ``GET /health``    : JSON 健康状态（LLM / 沙箱 / 存储依赖检查）
    - ``GET /metrics``   : Prometheus text format 指标
    - ``GET /``          : 服务说明

设计目标：
    - 仅用标准库 + PyYAML 即可运行（Prometheus / openai / anthropic 均为可选）。
    - 配置错误时**立即失败并打印路径化错误明细**，满足"D3 配置错误明确提示"。
    - LLM 默认回落到 echo provider（无外部依赖即可起完整环境）。

用法::

    python serve.py
    python serve.py --config config/default.yaml
    curl http://localhost:8000/health
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict

from runtime.config import load_config
from runtime.config.errors import ConfigError
from runtime.health import HealthChecker


def build_components(config: Dict[str, Any]) -> Dict[str, Any]:
    """根据配置组装 LLM / 沙箱 / 存储等依赖，返回供健康检查使用的组件字典。

    全部组装尽量宽松（echoprovider、local 沙箱恒可用），
    确保无外部服务时也能起完整环境并如实反映健康状态。
    """
    from runtime.llm_adapter.interface import ModelConfig
    from runtime.llm_adapter.multi_provider import (
        AnthropicProvider,
        EchoProvider,
        MultiProviderAdapter,
        OpenAIProvider,
    )

    llm_cfg = config.get("llm", {})
    provider = str(llm_cfg.get("provider", "echo"))

    adapter = MultiProviderAdapter(
        default_config=ModelConfig(
            provider=provider,
            model=str(llm_cfg.get("model", "echo")),
            api_key_env=str(llm_cfg.get("api_key_env", "OPENAI_API_KEY")),
            base_url=llm_cfg.get("base_url") or None,
            max_tokens=int(llm_cfg.get("max_tokens", 4096)),
            temperature=float(llm_cfg.get("temperature", 0.7)),
            top_p=float(llm_cfg.get("top_p", 1.0)),
        ),
        max_retries=int(llm_cfg.get("max_retries", 2)),
        retry_delay=float(llm_cfg.get("retry_delay", 1.0)),
        max_context_tokens=int(llm_cfg.get("max_context_tokens", 128000)),
    )
    adapter.register_provider(EchoProvider())
    adapter.register_provider(OpenAIProvider())
    adapter.register_provider(AnthropicProvider())

    # 沙箱：local + docker provider
    from runtime.sandbox.docker_provider import DockerSandboxProvider
    from runtime.sandbox.local_provider import LocalSandboxProvider
    from runtime.sandbox.registry import SandboxRegistry

    sandbox_cfg = config.get("sandbox", {})
    sandbox_registry = SandboxRegistry(
        providers={
            "local": LocalSandboxProvider(base_dir=sandbox_cfg.get("base_dir")),
            "docker": DockerSandboxProvider(),
        },
        fallback_modes=list(sandbox_cfg.get("fallback_modes", ["docker", "local"])),
    )

    # 存储：按 backend 组装 event_log + session_store
    storage_cfg = config.get("storage", {})
    backend = str(storage_cfg.get("backend", "sqlite"))
    db_path = str(storage_cfg.get("db_path", "data/deep_agent.db"))

    if backend == "sqlite":
        if db_path not in (":memory:", ""):
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        from runtime.event_log import create_event_log
        from runtime.session_store import create_session_store

        event_log = create_event_log(
            "sqlite",
            db_path=db_path,
            table=str(storage_cfg.get("event_table", "events")),
        )
        session_store = create_session_store(
            "sqlite",
            db_path=db_path,
            table=str(storage_cfg.get("session_table", "sessions")),
        )
    else:
        from runtime.event_log import create_event_log
        from runtime.session_store import create_session_store

        event_log = create_event_log("memory")
        session_store = create_session_store("memory")

    return {
        "llm_adapter": adapter,
        "sandbox_registry": sandbox_registry,
        "event_log": event_log,
        "session_store": session_store,
    }


class _DispatchHandler(BaseHTTPRequestHandler):
    """按路径分发到 health / metrics / 根说明。"""

    health_checker: HealthChecker

    def do_GET(self) -> None:  # noqa: N802
        from urllib.parse import urlparse

        path = urlparse(self.path).path

        if path in ("/health", "/healthz"):
            self._health()
        elif path == "/metrics":
            self._metrics()
        elif path == "/" or path == "":
            self._index()
        else:
            self._json({"error": "not_found"}, 404)

    def _health(self) -> None:
        try:
            result = self.health_checker.check()
            code = 200 if result["status"] == "ok" else 503
        except Exception as exc:  # pragma: no cover - 兜底
            result = {"status": "unhealthy", "error": str(exc)}
            code = 503
        self._json(result, code)

    def _metrics(self) -> None:
        body = self._render_metrics().encode("utf-8")
        self.send_response(200)
        self.send_header(
            "Content-Type", "text/plain; version=0.0.4; charset=utf-8"
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _render_metrics(self) -> str:
        try:
            from runtime.observability.metrics import get_registry

            return get_registry().render()
        except Exception:
            return ""

    def _index(self) -> None:
        self._json({
            "service": "deep-agent",
            "endpoints": ["/health", "/metrics"],
        }, 200)

    def _json(self, payload: Dict[str, Any], code: int) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


def run(host: str, port: int, checker: HealthChecker) -> None:
    """启动 HTTP 服务。"""
    _register_base_metrics()
    handler_cls = type(
        "_Handler", (_DispatchHandler,), {"health_checker": checker}
    )
    server = HTTPServer((host, port), handler_cls)
    print(f"[serve] deep-agent 已启动: http://{host}:{port}", flush=True)
    print(f"[serve] 健康检查: http://{host}:{port}/health", flush=True)
    print(f"[serve] 指标端点: http://{host}:{port}/metrics", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _register_base_metrics() -> None:
    """在默认 metrics registry 注册基础信息指标（版本 + 运行时长）。

    使 ``/metrics`` 在无业务流量时也有稳定输出，便于 Prometheus 校验目标
    存活；同时携带版本信息，供 Grafana 等按版本聚合。
    """
    import time

    from runtime import version
    from runtime.observability.metrics import get_registry

    registry = get_registry()
    gauge = registry.gauge("deep_agent_info", "Deep Agent build info.", ("version",))
    gauge.set(1, {"version": version.__version__})

    started = registry.gauge("deep_agent_start_time_seconds", "Process start time.")
    if started is None or started.value() == 0.0:
        started.set(time.time())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deep Agent HTTP 服务")
    parser.add_argument(
        "--config", default=None,
        help="配置文件路径（默认 config/default.yaml，或 APP_DEFAULT_CONFIG）",
    )
    parser.add_argument(
        "--schema", default=None,
        help="schema 文件路径（默认 config/schema.json，或 APP_SCHEMA_FILE）",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(
            default_path=args.config,
            schema_path=args.schema,
        )
    except ConfigError as exc:
        print(f"[serve] 配置加载失败，已中止启动：\n{exc}", file=sys.stderr)
        return 1

    components = build_components(cfg.as_dict())
    checker = HealthChecker(
        llm_adapter=components["llm_adapter"],
        sandbox_registry=components["sandbox_registry"],
        event_log=components["event_log"],
        session_store=components["session_store"],
        config=cfg.as_dict(),
    )

    server_cfg = cfg.get("server", {}) or {}
    host = str(server_cfg.get("host", "0.0.0.0"))
    port = int(server_cfg.get("port", 8000))
    run(host, port, checker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())