# 用户指南 — Vyrlo / Deep Agent v1.0.0

> **版本**：v1.0.0
> **最后更新**：2026-09-27 (Asia/Shanghai)
> **对应 tag**：`v1.0.0`

---

## 目录

1. [快速开始（5 分钟跑通一个任务）](#1-快速开始5-分钟跑通一个任务)
2. [常用配置](#2-常用配置)
3. [如何接入自己的工具](#3-如何接入自己的工具)
4. [如何接入自己的模型](#4-如何接入自己的模型)
5. [如何查看运行日志与追踪](#5-如何查看运行日志与追踪)

---

## 1. 快速开始（5 分钟跑通一个任务）

### 前置条件（2 分钟）

```bash
# ① 确保已按 docs/DEPLOY.md 完成安装
# ② 确保 Ollama 运行且模型已拉取
ollama list | grep qwen2.5:7b-instruct

# ③ 启动服务
python serve.py &
sleep 3
curl http://localhost:8080/health
# 预期: {"status": "ok", "version": "1.0.0"}
```

### 编写第一个 Agent 任务（3 分钟）

创建一个脚本 `my_first_task.py`：

```python
"""我的第一个 Deep Agent 任务 — 5 分钟快速开始"""
from runtime.runtime import Runtime, ComponentRegistry

# 1. 准备组件（从 config/default.yaml 加载即可）
from runtime.config.loader import load_config
cfg = load_config("config/default.yaml")

from runtime.llm_adapter.multi_provider import MultiProviderAdapter
from runtime.tool_registry.in_memory import InMemoryToolRegistry
from runtime.context_manager.layered import LayeredContextManager
from runtime.tool_registry.decorator import tool

# 2. 定义工具
@tool(
    name="read_file",
    description="读取指定文件的内容",
    params_schema={
        "type": "object",
        "properties": {
            "filepath": {"type": "string", "description": "文件路径"},
        },
        "required": ["filepath"],
    },
)
def read_file(filepath: str) -> str:
    with open(filepath, encoding="utf-8") as f:
        return f.read()

@tool(
    name="write_file",
    description="写入内容到指定文件",
    params_schema={
        "type": "object",
        "properties": {
            "filepath": {"type": "string", "description": "文件路径"},
            "content": {"type": "string", "description": "要写入的内容"},
        },
        "required": ["filepath", "content"],
    },
)
def write_file(filepath: str, content: str) -> str:
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return f"已写入 {len(content)} 字符到 {filepath}"

# 3. 注册工具
tool_registry = InMemoryToolRegistry()
tool_registry.register(read_file)
tool_registry.register(write_file)

# 4. 组装 Runtime
llm = MultiProviderAdapter(default_provider="ollama")
context_manager = LayeredContextManager()
registry = ComponentRegistry(
    llm_adapter=llm,
    tool_registry=tool_registry,
    context_manager=context_manager,
)
runtime = Runtime(registry, max_iterations=5)

# 5. 提交任务！
session = runtime.run(
    session_id="quick_start_1",
    task="请读取 README.md 文件，然后将其前 3 行写入 demo/output.txt",
    system_prompt="你是一个文件操作助手。请逐步完成任务，每次只能调用一个工具。",
)

print(f"✅ 任务完成！session={session.session_id}")
print(f"   迭代轮数: {session.metadata.get('iterations', 'N/A')}")
```

运行：

```bash
python my_first_task.py
```

**预期输出：**
```
✅ 任务完成！session=quick_start_1
   迭代轮数: 2
```

**发生了什么：**
1. Agent 收到任务
2. 调用 `read_file("README.md")`
3. LLM 收到文件内容
4. 调用 `write_file("demo/output.txt", <前3行>)
5. 工具成功执行 → 任务完成

> 这是最简单的单文件 Agent 示例。生产环境推荐使用 `serve.py` + 健康检查 + metrics。

---

## 2. 常用配置

### 2.1 调整 agent 行为

```yaml
# config/default.yaml

runtime:
  max_iterations: 10   # 单任务最大工具调用轮数
  max_delegation_depth: 3  # 委派深度

llm:
  model: qwen2.5:7b-instruct  # 主模型
  max_tokens: 4096             # 单轮回复上限
  temperature: 0.7             # 越高越随机（0.0-2.0）
  max_retries: 2               # 调用失败重试次数
  retry_delay: 1.0             # 重试间隔（秒）
```

**何时调整：**

| 场景 | 调整项 | 推荐值 |
|------|--------|--------|
| 简单任务（单步完成） | `max_iterations` | `3` |
| 复杂分析任务 | `max_iterations` | `10-15` |
| 需要创意性输出 | `temperature` | `1.0-1.5` |
| 需要严格一致性 | `temperature` | `0.0-0.3` |
| 长文档模式 | `max_tokens` | `8192` |

### 2.2 启用/禁用中间件

```yaml
# config/default.yaml
middlewares:
  enabled:
    - conversation_recorder
    - memory_lifecycle
    - memory_injector
    - memory_consolidation
    - tool_result_feedback
    - context_compressor
    - loop_guard
    - cost_guard
    - observability
    - sandbox
    - safety_guard
    - skill_injection
    - skill_observation
    - delegation
    - orchestrator
```

**按需精简（加速调试）：**

```yaml
middlewares:
  enabled:
    - loop_guard
    - observability
    - sandbox
    - tool_result_feedback
```

### 2.3 切换模型

**方式 1：配置文件**
```yaml
llm:
  model: qwen2.5:14b  # 换更大模型
```

**方式 2：环境变量**
```bash
export DEEP_AGENT_LLM_MODEL=qwen2.5:14b
```

> ⚠️ 新模型必须支持 OpenAI native function calling（E3.5 验证）。严禁 DeepSeek 系列。

### 2.4 使用多模型降级

系统内置 fallback 链：`主模型 → qwen2.5:3b → qwen2:7b → llama3.1:8b`

```bash
# 拉取降级模型
ollama pull qwen2.5:3b
ollama pull qwen2:7b
ollama pull llama3.1:8b
```

当主模型不可用时，系统自动降级尝试下一个模型。E3.5 验证：fallback 有效。

---

## 3. 如何接入自己的工具

### 3.1 方式 A：装饰器注册（推荐）

```python
from runtime.tool_registry.decorator import tool

@tool(
    name="web_search",
    description="在互联网上搜索信息",
    params_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "max_results": {"type": "integer", "description": "最大结果数", "default": 5},
        },
        "required": ["query"],
    },
    timeout=30.0,
    requires_approval=False,
    is_idempotent=True,
)
def web_search(query: str, max_results: int = 5) -> dict:
    """执行 Web 搜索"""
    import requests
    resp = requests.get(f"https://api.example.com/search", params={"q": query, "n": max_results})
    return {"results": resp.json(), "total": len(resp.json())}

# 注册到 Runtime
tool_registry.register(web_search)
```

### 3.2 方式 B：手动构造 Tool 对象

```python
from runtime.tool_registry.schema import Tool

def my_custom_fn(x: int, y: int) -> int:
    return x + y

my_tool = Tool(
    name="add_numbers",
    description="计算两个数的和",
    params_schema={
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "第一个数"},
            "y": {"type": "integer", "description": "第二个数"},
        },
        "required": ["x", "y"],
    },
    fn=my_custom_fn,
    timeout=5.0,
    is_idempotent=True,
)
tool_registry.register(my_tool)
```

### 3.3 工具返回值格式

工具可以返回：
- `str` — 纯文本结果
- `dict` — 结构化结果
- `ToolResult(content=..., sources=[...], metadata={...}, confidence=0.9)` — 含可信度

```python
from runtime.tool_registry.schema import ToolResult

@tool(name="analyze_file", ...)
def analyze_file(filepath: str) -> ToolResult:
    content = open(filepath).read()
    lines = content.count('\n')
    return ToolResult(
        content=f"文件共 {lines} 行",
        sources=[filepath],
        metadata={"size_bytes": len(content), "lines": lines},
        confidence=1.0,
    )
```

### 3.4 工具来源（ToolSource）

工具注册表支持三种来源（`runtime/tool_registry/tool_source.py`）：

| 来源 | 说明 |
|------|------|
| `DECORATOR` | `@tool` 装饰器注册（最常用） |
| `MCP` | MCP 协议服务端工具 |
| `EXTERNAL_API` | 外部 API 包装 |

---

## 4. 如何接入自己的模型

### 4.1 方式 A：Ollama 加载新模型

```bash
# 拉取新模型
ollama pull qwen2.5:14b

# 在配置中指定
export DEEP_AGENT_LLM_MODEL=qwen2.5:14b
```

### 4.2 方式 B：实现自定义 Provider

```python
from runtime.llm_adapter.interface import Provider
from typing import List, Dict, Any, Iterator

class MyProvider(Provider):
    """接入自定义 LLM 服务"""

    @property
    def supports_function_calling(self) -> bool:
        return True  # 你的服务必须支持 function calling

    def call(self, messages: List[Dict], **kwargs) -> "LLMResponse":
        import openai
        client = openai.OpenAI(base_url="https://my-llm-api.example.com/v1")
        resp = client.chat.completions.create(
            model="my-model",
            messages=messages,
            tools=kwargs.pop("tools", None),
            max_tokens=kwargs.pop("max_tokens", 4096),
        )
        choice = resp.choices[0]
        from runtime.llm_adapter.interface import LLMResponse
        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=choice.message.tool_calls or [],
            raw=resp,
            usage={
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            },
            model=resp.model,
            provider="my_provider",
            cost=0.0,
        )

    def stream(self, messages, **kwargs) -> Iterator[str]:
        # 流式实现...
        ...

    def count_tokens(self, messages) -> int:
        # 近似：每个消息 100 tokens
        return sum(len(m["content"]) // 4 for m in messages if "content" in m)

    def get_cost(self, usage: dict) -> float:
        return 0.0  # 自有模型可能免费

# 注册到 MultiProviderAdapter
from runtime.llm_adapter.multi_provider import MultiProviderAdapter
adapter = MultiProviderAdapter(default_provider="my_provider")
adapter.register_provider("my_provider", MyProvider())
```

### 4.3 模型兼容性要求

| 要求 | 说明 |
|------|------|
| **OpenAI native function calling** | 响应中必须包含 `tool_calls` 数组 |
| **JSON 参数** | tool_call.arguments 必须为合法 JSON |
| **多轮对话支持** | 必须支持 messages 历史 |

> ⚠️ **不兼容的模型**：DeepSeek flash/chat（`<｜｜DSML｜｜ invoke>` 格式）、无原生 FC 能力的模型。

---

## 5. 如何查看运行日志与追踪

### 5.1 Prometheus Metrics

```bash
# 查看所有指标
curl -s http://localhost:9090/metrics

# 关键指标
curl -s http://localhost:9090/metrics | grep agent_task
curl -s http://localhost:9090/metrics | grep llm_request
curl -s http://localhost:9090/metrics | grep tool_call
```

**核心指标清单（详见 `docs/API.md` §6.5）：**

| 指标 | 含义 |
|------|------|
| `agent_task_total` | 任务总数 |
| `agent_task_success_rate` | 任务成功率 |
| `agent_task_duration_seconds` | 任务耗时分布（P50/P95/P99） |
| `llm_request_count` | LLM 调用次数 |
| `llm_token_usage` | Token 消耗 |
| `tool_call_count` | 工具调用次数 |
| `tool_call_success_rate` | 工具调用成功率 |

### 5.2 Grafana Dashboard

```bash
# 导入 Dashboard 模板
# 文件: examples/grafana-d1-observability.json

# Prometheus 数据源配置
# 文件: config/prometheus.yml
```

### 5.3 事件日志（JSONL）

```bash
# 查看某个 session 的事件
sqlite3 data/deep_agent.db "SELECT * FROM events WHERE session_id='quick_start_1' ORDER BY timestamp;"

# 或导出为 JSON
sqlite3 data/deep_agent.db "SELECT events FROM events WHERE session_id='quick_start_1';" | python -m json.tool
```

### 5.4 Tracing Span

> 源码：`runtime/observability/tracing.py`

每次 Runtime.run() 创建一个 `session` span，内含嵌套的：
- `iteration` span（每轮迭代）
- `llm_call` span（每次 LLM 调用）
- `tool_call` span（每次工具执行）
- `middleware` span（每个中间件钩子）

```bash
# 日志级 tracing 输出（DEBUG 级别）
export DEEP_AGENT_OBSERVABILITY_LOG_LEVEL=DEBUG
python serve.py
```

### 5.5 故障定位流程（E4 验证，2-3 分钟）

**标准排查步骤：**

1. **检查健康状态** → `GET /health`
2. **检查关键 metrics** → `GET /metrics`，关注 `agent_task_success_rate`、`middleware_errors`
3. **拉取事件日志** → 按 session_id 查询 SQLite event_log
4. **分析 tracing span** → 定位耗时/失败的 span
5. **回查 LLM 响应** → 从 event_log 提取 tool_call 链路

> E4 验证：3 次故障定位均在 2-3 分钟内完成（debug1: T10 单词计数幻觉 / debug2: T40 占位符幻觉 / debug3: T46 无 tool_call），证据见 `acceptance-e4/debugging/`

### 5.6 已知可观测性缺口（上线前须接受）

| # | 缺口 | P 级 | 规避方法 |
|---|------|:---:|------|
| G1 | LLM 请求/响应完整 payload 未记录 | P1 | 手动复现 + DEBUG 日志 |
| G2 | 中间件决策路径未记录 | P1 | 检查 event_log 中间接推断 |
| G3 | tool_result_feedback 注入内容未保留 | P1 | 检查下游 LLM 响应推断反馈是否生效 |
| G4 | LLM 响应 tool_calls 原始 JSON 未记录 | P2 | 额外启用 DEBUG 日志 |

> 来源：`LAUNCH_READY.md` §4.3

---

## 附录：报告引用

| 报告 | 文件 | 与本用户指南相关的结论 |
|------|------|------------------------|
| E1.5 | `E1.5_REPORT.md` | 工具使用模式（read/write/copy 等）；F8 模型能力边界 |
| E2 | `acceptance-e2/E2_REPORT.md` | environment.md 环境基线 |
| E3 | `acceptance-e3/E3_REPORT.md` | qwen2.5:7b 性能基线 P95/P50 = 1.99 |
| E3.5 | `acceptance-e3.5/E3.5_REPORT.md` | Ollama 原生 FC 的使用说明 |
| E4 | `acceptance-e4/E4_REPORT.md` | 故障定位流程 2-3 分钟验证；3 个 debug 案例 |
| E5 | `acceptance-e5/E5_REPORT.md` | 自定义工具/中间件/技能的注册示例 |
| E6 | `E6_DECISION.md` | 生产环境推荐配置 |