# Vyrlo

可组合的 Agent 运行时框架——五层可插拔架构，中间件驱动，本地优先。

## 核心特性

- **五层可插拔架构**：Agent 内核 → 装配层 → 中间件层 → 多 Agent 编排 → 技能自进化，每层可独立替换
- **15 个内置中间件**：记忆管理、安全策略、可观测性、沙箱隔离、熔断重试、技能注入等，开箱即用
- **本地优先部署**：仅依赖 PyYAML + SQLite，无需外部服务即可运行完整环境
- **工具可扩展**：统一 ToolRegistry 接口，注册自定义工具只需一个装饰器
- **三通道可观测**：Prometheus 指标 + JSONL 事件日志 + Tracing span，故障定位分钟级
- **Docker 一键部署**：`docker compose up` 即可启动带健康检查、指标端点和 Grafana 仪表盘的服务

## 架构概览

```
┌──────────────────────────────────────────────┐
│  L5: 技能自进化系统                             │
│  SkillSystem = Store + Selector + Mutator      │
│  + Evaluator + Monitor                         │
├──────────────────────────────────────────────┤
│  L4: 多 Agent 编排                             │
│  AgentRegistry + SharedState + Delegation      │
│  + Orchestrator                                │
├──────────────────────────────────────────────┤
│  L3: 中间件层（15 个内置中间件）                  │
│  记忆 / 安全 / 可观测 / 隔离 / 弹性 / 技能注入    │
├──────────────────────────────────────────────┤
│  L2: 四基础组件 + 装配层                        │
│  LLMAdapter + ToolRegistry + ContextManager     │
│  + ToolCallParser → Runtime 装配               │
├──────────────────────────────────────────────┤
│  L1: Agent 内核                                │
│  AgentLoop + Context + Middleware              │
└──────────────────────────────────────────────┘
```

## 快速开始

### 环境要求

- Python 3.10+
- (可选) Ollama — 如需使用本地模型

### 安装与运行

```bash
# 克隆仓库
git clone https://github.com/liushay/Vyrlo.git
cd Vyrlo

# 安装依赖（仅 PyYAML 为必需）
pip install -r requirements.txt

# 启动服务（使用 echo provider，无需外部 API key）
python serve.py

# 验证服务
curl http://localhost:8000/health
```

预期输出：

```json
{"status": "ok", "components": {"llm": "healthy", "sandbox": "healthy", "storage": "healthy"}}
```

### 运行一个最小任务

```python
from runtime import Runtime
from runtime.llm_adapter.multi_provider import EchoProvider, MultiProviderAdapter
from runtime.llm_adapter.interface import ModelConfig

# 使用 echo provider (总是回显用户输入，无需 API key)
adapter = MultiProviderAdapter(
    default_config=ModelConfig(provider="echo", model="echo-test")
)
adapter.register_provider(EchoProvider())

rt = Runtime(llm_adapter=adapter)
result = rt.run(messages=[{"role": "user", "content": "Hello, Vyrlo!"}])
print(result["assistant"])
```

## 配置说明

配置文件位于 `config/default.yaml`，支持环境变量覆盖（`APP_` 前缀 + 双下划线层级分隔）。

```bash
# 使用自定义配置
python serve.py --config /path/to/config.yaml

# 通过环境变量覆盖 LLM provider
APP_LLM__PROVIDER=openai python serve.py
```

关键配置项：

| 配置路径 | 说明 | 默认值 |
|---------|------|--------|
| `llm.provider` | LLM 提供商 | `echo` |
| `sandbox.mode` | 沙箱模式 | `local` |
| `storage.backend` | 存储后端 | `sqlite` |
| `server.port` | HTTP 服务端口 | `8000` |

详见 `config/schema.json` 获取完整配置 schema。

## 扩展方式

### 添加中间件

继承 `Middleware` 基类，覆盖所需的生命周期钩子后注册即可：

```python
from runtime import Middleware, HookResult

class LoggingMiddleware(Middleware):
    async def before_llm_call(self, ctx, messages):
        print(f"[log] sending {len(messages)} messages to LLM")
        return HookResult.CONTINUE

# 运行时注册
rt.register_middleware(LoggingMiddleware())
```

支持的钩子：`before_iteration`、`after_iteration`、`before_llm_call`、`after_llm_call`、`before_tool`、`after_tool`、`on_error`。

### 添加工具

使用装饰器即可将任意函数注册为工具：

```python
from runtime.tool_registry import tool_registry

@tool_registry.register(name="weather", description="获取指定城市的天气")
def get_weather(city: str) -> dict:
    return {"city": city, "temperature": 22, "condition": "sunny"}
```

### 添加技能

技能系统支持动态加载、变异和评估，详见 `docs/USER_GUIDE.md` §4。

## 已知限制

当前版本的已知限制（模型能力边界、单点依赖、工具数量等）参见 [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)。

## 许可

MIT License — 详见 [LICENSE](LICENSE)。