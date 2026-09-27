"""
[批次 D3] runtime.health —— 健康检查。

提供两类能力：
    1. ``HealthChecker``：可编程的健康检查器，判定服务整体状态与三类依赖
       （LLM / 沙箱 / 存储）的可用性，输出结构化结果。
    2. ``HealthHandler`` / ``MetricsHandler``：零第三方依赖的 HTTP 处理器
       （基于标准库 ``http.server``），暴露：
           - ``GET /health``  —— JSON 健康状态；
           - ``GET /metrics`` —— Prometheus text format 指标。
       二者都不引入 FastAPI / Flask，可直接嵌入任何部署形态，
       也可被 docker-compose 的 ``healthcheck`` 通过 ``curl`` 调用。

依赖检查语义（不发起真实业务调用，避免副作用）：
    - LLM：适配器可用、提供商可解析；配置了 API key（或提供商为 echo）。
    - 沙箱：至少一个 provider 的 ``is_available()`` 为真。
    - 存储：后端可读写（SQLite 用一次 SELECT 探测；内存后端恒为真）。

HTTP 响应约定：
    - 状态码：整体健康 200，否则 503。
    - 响应体：``{"status": "ok|degraded|unhealthy", "checks": {...}}``。
"""

from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict, Optional

#: 内部版本，避免在 health 中重复维护（与 runtime.version 解耦）。
try:  # pragma: no cover - 环境相关
    from runtime.version import __version__ as _RUNTIME_VERSION
except Exception:  # pragma: no cover
    _RUNTIME_VERSION = "unknown"


# ============================================================================
# 健康检查结果模型
# ============================================================================


class CheckResult:
    """单个依赖的检查结果。

    Attributes:
        name:    依赖名。
        healthy: 是否健康。
        detail:  附加信息（错误原因 / 元数据）。
    """

    def __init__(self, name: str, healthy: bool, detail: str = "") -> None:
        self.name = name
        self.healthy = healthy
        self.detail = detail

    def as_dict(self) -> Dict[str, Any]:
        return {
            "healthy": self.healthy,
            "detail": self.detail,
        }


class HealthChecker:
    """聚合依赖健康检查器。

    用法::

        checker = HealthChecker(llm_adapter=adapter, sandbox_registry=sandbox,
                                event_log=log, session_store=store)
        result = checker.check()
        result["status"]   # "ok" | "degraded" | "unhealthy"
    """

    LLM = "llm"
    SANDBOX = "sandbox"
    STORAGE = "storage"

    def __init__(
        self,
        llm_adapter: Any = None,
        sandbox_registry: Any = None,
        event_log: Any = None,
        session_store: Any = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            llm_adapter:      LLMAdapter 实例（兼容任何含 provider 注册表信息的对象）。
            sandbox_registry: SandboxRegistry / SandboxProvider 实例。
            event_log:        EventLog 实例（可选）。
            session_store:    SessionStore 实例（可选）。
            config:           已加载的配置字典（用于探测 API key 等）。
        """
        self._llm = llm_adapter
        self._sandbox = sandbox_registry
        self._event_log = event_log
        self._session_store = session_store
        self._config = config or {}

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def check(self) -> Dict[str, Any]:
        """执行所有依赖检查，返回聚合状态字典。"""
        checks = {
            self.LLM: self._check_llm().as_dict(),
            self.SANDBOX: self._check_sandbox().as_dict(),
            self.STORAGE: self._check_storage().as_dict(),
        }
        healthy = all(c["healthy"] for c in checks.values())
        degraded = any(c["healthy"] for c in checks.values())
        status = "ok" if healthy else ("degraded" if degraded else "unhealthy")

        return {
            "status": status,
            "version": _RUNTIME_VERSION,
            "timestamp": round(time.time(), 3),
            "checks": checks,
        }

    # ------------------------------------------------------------------
    # 依赖检查
    # ------------------------------------------------------------------

    def _check_llm(self) -> CheckResult:
        """LLM 依赖健康检查（不发起真实调用）。"""
        adapter = self._llm
        if adapter is None:
            return CheckResult(self.LLM, False, "llm_adapter 未注入")

        # 1. 至少具备可调用的 call 方法
        if not callable(getattr(adapter, "call", None)):
            return CheckResult(self.LLM, False, "llm_adapter 缺少 call 方法")

        # 2. 提供商可解析（优先读 config，其次是 default_config）
        provider = _config_value(self._config, "llm.provider") or _default_provider(adapter)
        if not provider:
            return CheckResult(self.LLM, False, "未配置 provider")

        # 3. API key 探针：echo / 本地模型无需 key
        if provider in ("echo",):
            return CheckResult(self.LLM, True, f"provider={provider}（无需 key）")

        api_key_env = _config_value(self._config, "llm.api_key_env") or "OPENAI_API_KEY"
        if not os.environ.get(api_key_env):
            return CheckResult(
                self.LLM,
                False,
                f"provider={provider} 但环境变量 {api_key_env} 未设置",
            )
        return CheckResult(self.LLM, True, f"provider={provider}")

    def _check_sandbox(self) -> CheckResult:
        """沙箱依赖健康检查（is_available 探测）。"""
        sandbox = self._sandbox
        if sandbox is None:
            # 未配置沙箱视为降级：报告不健康，但不致命（允许无沙箱运行）
            return CheckResult(self.SANDBOX, False, "沙箱未配置")

        # SandboxRegistry：列出全部 provider
        providers = {}
        if hasattr(sandbox, "list_modes"):
            for mode in sandbox.list_modes():
                p = sandbox.get(mode)
                if p is not None:
                    providers[mode] = p
        elif hasattr(sandbox, "is_available"):
            providers["default"] = sandbox

        available = []
        for name, p in providers.items():
            try:
                if callable(getattr(p, "is_available", None)) and p.is_available():
                    available.append(name)
            except Exception:
                continue

        if not available:
            return CheckResult(self.SANDBOX, False, "无可用沙箱 provider")
        return CheckResult(self.SANDBOX, True, f"可用: {', '.join(available)}")

    def _check_storage(self) -> CheckResult:
        """存储依赖健康检查（SQLite 探测 / 内存后端恒真）。"""
        # 优先探测 session_store（具备 SQLite 实现），否则 event_log
        backend = self._session_store or self._event_log
        if backend is None:
            return CheckResult(self.STORAGE, False, "存储后端未注入")

        # 内存后端：恒判定健康
        backend_name = getattr(backend, "backend", None)
        if backend_name in ("memory", "in_memory", None):
            return CheckResult(self.STORAGE, True, f"backend={backend_name or 'memory'}")

        # SQLite 后端：执行一次读探测
        db_path = getattr(backend, "db_path", None)
        try:
            conn = getattr(backend, "_conn", None)
            if conn is not None:
                conn.execute("SELECT 1")
            else:
                import sqlite3

                probe = sqlite3.connect(db_path or ":memory:")
                probe.execute("SELECT 1")
                probe.close()
        except Exception as exc:
            return CheckResult(self.STORAGE, False, f"存储探测失败: {exc}")
        return CheckResult(self.STORAGE, True, f"backend={backend_name}")


# ============================================================================
# 零依赖 HTTP 处理器
# ============================================================================


class HealthHandler(BaseHTTPRequestHandler):
    """零依赖的 ``/health`` HTTP 处理器。

    用法（配合标准库 http.server）::

        from http.server import HTTPServer
        server = HTTPServer(("0.0.0.0", 8000), handler_factory(checker))
        server.serve_forever()
    """

    def __init__(self, checker: HealthChecker, *args: Any, **kwargs: Any) -> None:
        self._checker = checker
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:  # noqa: N802 - 覆写标准库命名
        from urllib.parse import urlparse

        path = urlparse(self.path).path
        if path == "/health" or path == "/healthz":
            self._respond_health()
        else:
            self._respond_json({"status": "unhealthy", "error": "not_found"}, 404)

    def _respond_health(self) -> None:
        try:
            result = self._checker.check()
            code = 200 if result["status"] == "ok" else 503
        except Exception as exc:  # 检查器异常不应导致 500 空响应
            result = {
                "status": "unhealthy",
                "error": str(exc),
                "timestamp": round(time.time(), 3),
            }
            code = 503
        self._respond_json(result, code)

    def _respond_json(self, payload: Dict[str, Any], code: int) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """健康检查高频请求，默认静默，避免刷屏。"""  # noqa: D401
        pass


class MetricsHandler(BaseHTTPRequestHandler):
    """零依赖的 ``/metrics`` HTTP 处理器（Prometheus text format）。

    仅在 ``runtime.observability.metrics`` 存在指标时渲染；不可用则返回空体。
    """

    def do_GET(self) -> None:  # noqa: N802 - 覆写标准库命名
        body = self._render_metrics().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _render_metrics(self) -> str:
        try:
            from runtime.observability.metrics import get_registry

            return get_registry().render()
        except Exception:
            return ""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


def handler_factory(checker: HealthChecker):
    """返回一个可传给 ``HTTPServer`` 的 handler 工厂（闭包注入 checker）。"""

    class _Handler(HealthHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(checker, *args, **kwargs)

    return _Handler


# ============================================================================
# 辅助
# ============================================================================


def _default_provider(adapter: Any) -> str:
    config = getattr(adapter, "_default_config", None)
    return str(getattr(config, "provider", "") or "")


def _config_value(config: Dict[str, Any], path: str) -> Any:
    node: Any = config
    for seg in path.split("."):
        if not isinstance(node, dict) or seg not in node:
            return None
        node = node[seg]
    return node


__all__ = [
    "HealthChecker",
    "CheckResult",
    "HealthHandler",
    "MetricsHandler",
    "handler_factory",
]