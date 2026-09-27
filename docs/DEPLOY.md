# 部署文档 — Vyrlo / Deep Agent v1.0.0

> **版本**：v1.0.0
> **最后更新**：2026-09-27 (Asia/Shanghai)
> **对应 tag**：`v1.0.0`

---

## 目录

1. [环境要求](#1-环境要求)
2. [安装步骤](#2-安装步骤)
3. [配置说明](#3-配置说明)
4. [启动与停止](#4-启动与停止)
5. [健康检查](#5-健康检查)
6. [常见问题排查](#6-常见问题排查)

---

## 1. 环境要求

### 1.1 硬件

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| CPU | 4 核 | 8 核+ |
| 内存 | 8 GB | 16 GB+（加载 7B 模型用） |
| 磁盘 | 20 GB 可用 | 50 GB+ SSD |
| 网络 | 无需外网（本地 Ollama） | 按需 |

### 1.2 软件

| 软件 | 版本 | 说明 |
|------|------|------|
| **Python** | 3.10+ | 运行环境 |
| **Ollama** | ≥0.2.0 | 本地 LLM 推理服务 |
| **pip** | 最新稳定版 | 包管理器 |
| **Docker**（可选） | ≥24.0 | Docker Sandbox + 容器化部署 |
| **Git** | ≥2.30 | 版本管理 |

### 1.3 主模型

**必须安装：** `qwen2.5:7b-instruct`（Ollama 格式，Q4_K_M 量化）

```bash
# 拉取主模型
ollama pull qwen2.5:7b-instruct

# 可选降级模型
ollama pull qwen2.5:3b
ollama pull qwen2:7b
ollama pull llama3.1:8b
```

> ⚠️ 严禁使用 DeepSeek 系列作为默认或 fallback 模型（原因见 `E6_DECISION.md` §5.2）

### 1.4 网络端口

| 端口 | 用途 | 类型 |
|------|------|------|
| 11434 | Ollama API | 内部（本机） |
| 8080 | 健康检查 + 服务 API | 对外 |
| 9090 | Prometheus metrics | 监控 |

---

## 2. 安装步骤

### 2.1 方式 A：pip 安装（开发/单机）

```bash
# 1. 克隆仓库
git clone <repo-url> deep-agent
cd deep-agent

# 2. 创建虚拟环境（推荐）
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证安装
python -c "from runtime.version import VERSION; print(VERSION)"
# 应输出: 1.0.0
```

### 2.2 方式 B：Docker 部署

```bash
# 1. 构建镜像
docker build -t deep-agent:v1.0.0 .

# 2. 检查镜像
docker images deep-agent:v1.0.0
```

### 2.3 首次运行验证

```bash
# 确保 Ollama 已启动
ollama serve
# 或在新终端验证
ollama list | grep qwen2.5

# 运行冒烟测试
python serve.py &
sleep 3
curl http://localhost:8080/health
# 预期: {"status": "ok", "version": "1.0.0"}
```

---

## 3. 配置说明

### 3.1 配置体系

```
config/
├── schema.json          # 配置 JSON Schema（校验定义）
└── default.yaml         # 默认配置（生产使用）
```

### 3.2 YAML 配置文件

> 完整配置参考：`config/default.yaml`（带注释）
> Schema 定义：`config/schema.json`

**完整配置项清单详见 `docs/API.md` §7。**

### 3.3 环境变量覆盖

所有 YAML 配置项均可通过环境变量覆盖，规则为：路径分隔符 `.` → `_`，全部大写，加前缀 `DEEP_AGENT_`。

| YAML 路径 | 环境变量 | 示例 |
|-----------|---------|------|
| `llm.model` | `DEEP_AGENT_LLM_MODEL` | `export DEEP_AGENT_LLM_MODEL=qwen2.5:14b` |
| `llm.base_url` | `DEEP_AGENT_LLM_BASE_URL` | `export DEEP_AGENT_LLM_BASE_URL=http://192.168.1.100:11434` |
| `runtime.max_iterations` | `DEEP_AGENT_RUNTIME_MAX_ITERATIONS` | `export DEEP_AGENT_RUNTIME_MAX_ITERATIONS=5` |
| `llm.max_tokens` | `DEEP_AGENT_LLM_MAX_TOKENS` | `export DEEP_AGENT_LLM_MAX_TOKENS=2048` |
| `observability.log_level` | `DEEP_AGENT_OBSERVABILITY_LOG_LEVEL` | `export DEEP_AGENT_OBSERVABILITY_LOG_LEVEL=DEBUG` |
| `storage.db_path` | `DEEP_AGENT_STORAGE_DB_PATH` | `export DEEP_AGENT_STORAGE_DB_PATH=/data/deep_agent.db` |

### 3.4 配置校验

启动时自动执行 schema 校验，校验失败将拒绝启动并打印错误详情。

手动校验命令：
```bash
python -c "
from runtime.config.loader import validate_config
result = validate_config('config/default.yaml')
print(result)
"
```

### 3.5 关键配置项（上线必检）

| 配置项 | 值 | 说明 |
|--------|----|------|
| `llm.model` | `qwen2.5:7b-instruct` | **必须**，唯一验证过全链路的主模型 |
| `llm.provider` | `ollama` | LLM 提供商 |
| `llm.base_url` | `http://localhost:11434` | Ollama API 端点 |
| `schema_version` | `"1.0.0"` | 配置版本号 |
| `runtime.max_iterations` | `10` | 单任务最大轮数 |
| `llm.max_tokens` | `4096` | 单轮输出上限 |
| `llm.temperature` | `0.7` | 生成温度 |

### 3.6 版本迁移

配置 schema 版本迁移由 `runtime/config/migrations.py` 管理。当前 `schema_version: "1.0.0"` 为初始版本，无需迁移。后续版本升级时，启动脚本会自动执行迁移。

---

## 4. 启动与停止

### 4.1 直接启动

```bash
# 启动 Ollama（如未运行）
ollama serve &

# 启动 Deep Agent 服务
python serve.py

# 指定配置
python serve.py --config config/production.yaml

# 后台运行
nohup python serve.py > logs/agent.log 2>&1 &
```

### 4.2 Docker Compose 启动

> 配置文件：`docker-compose.yml`

```bash
# 启动所有服务（Deep Agent + Ollama）
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止
docker-compose down
```

### 4.3 独立 Docker 启动

```bash
# 启动（连接到主机 Ollama）
docker run -d \
  --name deep-agent \
  -p 8080:8080 -p 9090:9090 \
  -e DEEP_AGENT_LLM_BASE_URL=http://host.docker.internal:11434 \
  -v $(pwd)/data:/app/data \
  deep-agent:v1.0.0
```

### 4.4 优雅停止

```bash
# 方式 1：SIGTERM（推荐）
kill -TERM $(cat agent.pid)

# 方式 2：Ctrl+C（前台运行时）

# Docker
docker stop deep-agent
```

---

## 5. 健康检查

### 5.1 端点

| 端点 | 说明 | 预期响应 |
|------|------|---------|
| `GET /health` | 服务健康状态 | `{"status": "ok", "version": "1.0.0"}` |
| `GET /metrics` | Prometheus 指标 | text/plain 格式 |

### 5.2 Kubernetes 探针示例

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 10
```

### 5.3 手动健康检查

```bash
# 基本检查
curl -s http://localhost:8080/health | jq .

# metrics 检查
curl -s http://localhost:9090/metrics | grep agent_task_total
```

---

## 6. 常见问题排查

### 6.1 启动失败

**问题：`schema validation failed`**

```
原因：config/default.yaml 与 config/schema.json 不匹配
解决：检查 config/default.yaml 中的 schema_version 是否为 "1.0.0"
      对照 config/schema.json 检查配置字段
```

**问题：`ModuleNotFoundError: No module named 'xxx'`**

```bash
# 重新安装依赖
pip install -r requirements.txt --upgrade
```

**问题：`Ollama connection refused`**

```bash
# 确保 Ollama 正在运行
ollama serve

# 检查 Ollama 端口
curl http://localhost:11434/api/tags
```

### 6.2 运行时问题

**问题：工具调用全部失败**

```
原因：模型不支持原生 function calling 格式
检查：模型是否为 qwen2.5:7b-instruct
      严禁使用 DeepSeek 系列（E3.5 验证 0/6 成功率）
      参考 E6_DECISION.md §5.2
```

**问题：write_file 输出内容不正确（自产占位符）**

```
原因：模型能力边界（F8 类，E1.5_REPORT.md）
现象：模型在 read→write 场景下用自产占位符替代真实内容
影响：约 12% 任务（6/50）
规避：换用更大模型（如 qwen2.5:14b 或更高）
```

**问题：任务执行超过预期时间**

```
# 检查 max_iterations 配置
# 配置路径: runtime.max_iterations，默认 10
# 可用环境变量覆盖:
export DEEP_AGENT_RUNTIME_MAX_ITERATIONS=10
```

**问题：内存持续增长**

```
E2 验证：100 次循环内存波动 5.66%，无泄漏
如果超出此范围，检查：
1. 是否有大型上下文导致窗口膨胀
2. event_log 后端是否定期清理
3. context_compressor 阈值是否合理
```

### 6.3 日志查看

```bash
# 查看服务日志
tail -f logs/agent.log

# 查看事件日志（SQLite）
sqlite3 data/deep_agent.db "SELECT * FROM events ORDER BY timestamp DESC LIMIT 50;"

# Docker 日志
docker logs -f deep-agent
```

### 6.4 回滚到上一版本

```bash
# 方案 A：Git 回滚
git checkout batch-E5-done

# 方案 B：配置回滚
git checkout batch-E5-done -- config/default.yaml

# 方案 C：Docker 回滚
docker-compose down
docker tag deep-agent:v1.0.0 deep-agent:v1.0.0-backup
docker tag deep-agent:v0.x deep-agent:latest
docker-compose up -d
```

> 完整回滚预案见 `ROLLBACK.md`。E5 验证：回滚耗时 0.74s，数据完整。

### 6.5 降级操作（E5 验证，3/3 通过）

**LLM 不可用：**
```bash
# 自动 fallback 链：qwen2.5:7b → qwen2.5:3b → qwen2:7b → llama3.1:8b
# 需确保降级模型均已 pull
ollama pull qwen2.5:3b qwen2:7b llama3.1:8b
```

**沙箱不可用：**
```yaml
# config/default.yaml
sandbox:
  provider: "local"  # 回退到本地执行
```

**存储不可用：**
```bash
export DEEP_AGENT_STORAGE_EVENT_LOG_BACKEND=memory
export DEEP_AGENT_STORAGE_SESSION_STORE_BACKEND=memory
```

---

## 附录：报告引用

| 报告 | 文件 | 与本部署文档相关的结论 |
|------|------|------------------------|
| E2 | `acceptance-e2/E2_REPORT.md` | 100 次 0 异常验证；内存波动 5.66%；environment.md 环境基线 |
| E3 | `acceptance-e3/E3_REPORT.md` | qwen2.5:7b + Ollama 配置的性能基线 |
| E3.5 | `acceptance-e3.5/E3.5_REPORT.md` | F1（Ollama 原生 FC）环境修复 |
| E3.6 | `acceptance-e3.6/check_env.py` | 环境就绪检查脚本 |
| E4 | `acceptance-e4/E4_REPORT.md` | 健康检查 + metrics 的可观测验证 |
| E5 | `acceptance-e5/E5_REPORT.md` | 回滚 0.74s；可配置 4/4；可降级 3/3；可扩展 3/3 |
| E6 | `E6_DECISION.md` | 上线环境与配置清单 |