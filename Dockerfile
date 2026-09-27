# ============================================================================
# Deep Agent 运行时镜像
# ============================================================================
# 构建：docker build -t deep-agent:latest .
# 说明：最小可运行集仅需 PyYAML；openai / anthropic / prometheus / otel 均为
#       可选依赖，未安装时运行时自动降级（见 runtime/observability/metrics.py）。
# ============================================================================

# ---- 运行时基础镜像 ----
FROM python:3.11-slim

# 运行时依赖（无 build 依赖；后置取消注释可追加可选 SDK）
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 非 root 运行，最小权限
RUN useradd --create-home --uid 10001 agent

WORKDIR /app

# 先拷贝依赖清单以利用 Docker 层缓存
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝应用代码与配置
COPY runtime/ ./runtime/
COPY agent_loop.py ./
COPY serve.py ./
COPY config/ ./config/

# 数据目录（SQLite 持久化 + 本地沙箱）
RUN mkdir -p /app/data && chown -R agent:agent /app/data

USER agent

EXPOSE 8000

# 健康检查：使用容器内置 python + curl 探测 /health 端点
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health >/dev/null || exit 1

# 默认以配置文件启动服务；LLM 默认回落 echo（无外部依赖即可起完整环境）
CMD ["python", "serve.py", "--config", "config/default.yaml"]