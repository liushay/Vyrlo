# API 文档 — Vyrlo / Deep Agent v1.0.0

> **版本**：v1.0.0
> **最后更新**：2026-09-27 (Asia/Shanghai)
> **对应 tag**：`v1.0.0`

---

## 目录

1. [Runtime（装配层）](#1-runtime装配层)
2. [L2 四组件接口](#2-l2-四组件接口)
   - [LLMAdapter](#21-llmadapter)
   - [ToolRegistry](#22-toolregistry)
   - [ToolCallParser](#23-toolcallparser)
   - [ContextManager](#24-contextmanager)
3. [L3 中间件基类与钩子签名](#3-l3-中间件基类与钩子签名)
4. [L4 多 Agent 接口](#4-l4-多-agent-接口)
5. [L5 技能系统接口](#5-l5-技能系统接口)
6. [基础设施接口](#6-基础设施接口)
7. [配置项清单](#7-配置项清单)

---

## 1. Runtime（装配层）

> 源码：`runtime/runtime.py`

### 1.0 数据结构

```python
@dataclass
class ComponentRegistry:
    """Runtime 构造时的依赖组件集合"""
    llm_adapter: LLMAdapter            # LLM 适配器（必需）
    tool_registry: ToolRegistry         # 工具注册表（必需）
    context_manager: Any                # 上下文管理器（必需）
    tool_call_parser: Optional[ToolCallParser] = None   # 默认 AutoDetectParser
    event_log: Optional[EventLog] = None                # 默认 create_event_log("memory")
    session_store: Optional[SessionStore] = None         # 默认 create_session_store("memory")
```

### 1.1 构造器

```python
class Runtime:
    def __init__(
        self,
        registry: ComponentRegistry,
        max_iterations: int = 10,
        agent_registry: Optional[AgentRegistry] = None,
        shared_state: Optional[SharedStateStore] = None,
        max_delegation_depth: int = 3,
        skill_system: Any = None,
    ) -> None
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `registry` | `ComponentRegistry` | 必需 | 组件注册表 |
| `max_iterations` | `int` | `10` | AgentLoop 最大迭代轮数 |
| `agent_registry` | `Optional[AgentRegistry]` | `None` | L4 Agent 注册表，None 不启用委派 |
| `shared_state` | `Optional[SharedStateStore]` | `None` | L4 共享状态存储 |
| `max_delegation_depth` | `int` | `3` | 委派深度上限 |
| `skill_system` | `Any` | `None` | L5 技能自进化子系统，None 时 skill 中间件静默放行 |

### 1.2 核心运行方法

```python
def run(
    self,
    session_id: str,
    task: str,
    system_prompt: str = "",
    budget: Optional[Any] = None,
) -> Session
```

运行一次 Agent 任务。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `session_id` | `str` | 必需 | 会话 ID（复用已保存的会话） |
| `task` | `str` | 必需 | 用户任务文本 |
| `system_prompt` | `str` | `""` | system prompt 模板（可含 `{{memory_context}}` 占位符） |
| `budget` | `Optional[Any]` | `None` | 预算：`Budget` 实例、`dict` 或 `int`（token 上限） |

**返回值：** `Session` — 运行结束后的会话对象（已保存到 SessionStore）

**示例：**

```python
from runtime.runtime import Runtime, ComponentRegistry
from runtime.llm_adapter.interface import LLMAdapter
from runtime.tool_registry.in_memory import InMemoryToolRegistry
from runtime.context_manager.layered import LayeredContextManager

registry = ComponentRegistry(
    llm_adapter=LLMAdapter(...),
    tool_registry=InMemoryToolRegistry(),
    context_manager=LayeredContextManager(),
)
runtime = Runtime(registry, max_iterations=8)
session = runtime.run("s1", "帮我计算 3+5", system_prompt="你是一个数学助手")
```

### 1.3 中间件管理

```python
def register_middleware(self, middleware: Middleware) -> None
```
注册 L3 中间件。

```python
def load_plugins(self, paths: List[str], fiber_id: Optional[str] = None) -> Fiber
```
加载插件并创建 Fiber。

```python
def dispose_fiber(self, fiber_id: str) -> None
```
销毁 Fiber 并回滚状态。

### 1.4 Agent 注册表管理

```python
def register_agent(self, spec: AgentSpec) -> None
```
注册一个可委派 Agent。

```python
def list_agents(self) -> List[AgentSpec]
```
列出已注册的 Agent。

### 1.5 Session 管理

```python
def get_session(self, session_id: str) -> Optional[Session]
```
从 SessionStore 读取会话，不存在时返回 None。

```python
def resume(self, session_id: str) -> Session
```
恢复已保存的会话。会话不存在时抛出 `KeyError`。

### 1.6 资源清理

```python
def close(self) -> None
```
关闭可关闭的后端（SQLite 事件日志/会话存储）。

---

## 2. L2 四组件接口

### 2.1 LLMAdapter

> 源码：`runtime/llm_adapter/interface.py`

```python
class LLMAdapter:
    def call(
        self, 
        messages: List[Dict[str, Any]], 
        context: Optional[Any] = None,
    ) -> LLMResponse
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `messages` | `List[Dict]` | OpenAI 格式的消息列表 |
| `context` | `Optional[Any]` | 可选上下文对象（用于成本追踪写入） |

**返回值 `LLMResponse`：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `content` | `str` | 文本响应 |
| `tool_calls` | `List[dict]` | 工具调用列表 |
| `raw` | `Any` | 原始响应 |
| `usage` | `dict` | token 使用统计 `{"prompt_tokens", "completion_tokens", "total_tokens"}` |
| `model` | `str` | 实际使用的模型名称 |
| `provider` | `str` | 提供商名称 |
| `cost` | `float` | 预估成本 |

```python
def stream(
    self,
    messages: List[Dict[str, Any]],
    model_config: Optional[Dict] = None,
) -> Iterator[str]
```

流式调用，返回 token 迭代器。

```python
def count_tokens(self, messages: List[Dict[str, Any]]) -> int
```

估算 token 数。

```python
def get_cost(self, usage: Dict[str, int]) -> float
```

计算调用成本。

---

**MultiProviderAdapter**（`runtime/llm_adapter/multi_provider.py`）：

```python
class MultiProviderAdapter:
    def __init__(self, default_provider: str = "ollama")
    def register_provider(self, name: str, provider: "Provider") -> None
    def call(self, messages, context=None) -> LLMResponse
    def stream(self, messages, model_config=None) -> Iterator[str]
```

**Provider 接口**（需实现的标准接口）：

```python
class Provider:
    @property
    def supports_function_calling(self) -> bool: ...
    
    def call(self, messages: List[dict], **kwargs) -> LLMResponse: ...
    def stream(self, messages: List[dict], **kwargs) -> Iterator[str]: ...
    def count_tokens(self, messages: List[dict]) -> int: ...
    def get_cost(self, usage: dict) -> float: ...
```

---

### 2.2 ToolRegistry

> 源码：`runtime/tool_registry/interface.py`

```python
class ToolRegistry:
    def register(self, tool: Tool) -> None
    def unregister(self, name: str) -> None
    def get(self, name: str) -> Tool
    def list(self) -> List[Tool]
    def export_schemas(self, format: str = "openai") -> List[dict]
    def execute(self, name: str, args: Dict[str, Any], sandbox: Any = None) -> ToolResult
```

**`register()` 方法：**
- `tool`: `Tool` 对象（由装饰器或手动构造）

**`execute()` 方法：**
- `name`: 工具名称
- `args`: 参数字典
- `sandbox`: 可选沙箱句柄（由 SandboxMiddleware 传入）

**`Tool` 数据模型（`runtime/tool_registry/schema.py`）：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | `str` | 工具名称 |
| `description` | `str` | 工具描述 |
| `params_schema` | `dict` | 参数 JSON Schema（从函数签名自动生成） |
| `returns_schema` | `Optional[dict]` | 返回值 schema |
| `requires_approval` | `bool` | 是否需要用户审批 |
| `is_idempotent` | `bool` | 是否幂等 |
| `timeout` | `Optional[float]` | 超时时间（秒） |

**`ToolResult` 结构化返回：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `content` | `Any` | 工具执行原始结果 |
| `sources` | `List[str]` | 引用来源列表 |
| `metadata` | `dict` | 元数据（如执行耗时） |
| `confidence` | `float` | 置信度 0.0-1.0 |

**装饰器注册（`runtime/tool_registry/decorator.py`）：**

```python
@tool(
    name="my_tool",
    description="我的自定义工具",
    params_schema={"type": "object", "properties": {...}},
    timeout=30.0,
    requires_approval=False,
)
def my_tool(arg1: str, arg2: int) -> str:
    ...
    return result
```

---

### 2.3 ToolCallParser

> 源码：`runtime/tool_call_parser/interface.py`

```python
class ToolCallParser:
    def parse(self, response: Any, format: Optional[str] = None) -> List[ToolCall]
    def validate(self, tool_call: ToolCall, schema: dict) -> ToolCallValidationResult
    def repair(self, tool_call: ToolCall, schema: dict, error: str) -> Optional[ToolCall]
    def format_error(self, error: Any) -> dict
```

**`parse()` 方法：**
- `response`: LLM 原始响应
- `format`: 可选格式（`"openai"` / `"anthropic"` / `"text"`），不指定时自动检测

**`ToolCall` 结构：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `Optional[str]` | 调用 ID |
| `name` | `str` | 工具名称 |
| `args` | `dict` | 参数字典 |

**支持的响应格式：**
- OpenAI 原生 function calling（`tool_calls` 数组）
- Anthropic tool use（`content` 块）
- 文本格式回退（`Action:` / `Action Input:` 模式）

**默认解析器：** `AutoDetectParser`（`runtime/tool_call_parser/parsers.py`）— 自动检测格式

**错误处理策略（`runtime/tool_call_parser/error_strategy.py`）：**

可配置为：
- 布尔值（严格/宽松）
- 字符串模板（自定义错误消息）
- 异常类型/元组（按类型区分处理）
- 回调函数（自定义逻辑）

---

### 2.4 ContextManager

> 源码：`runtime/context_manager/interface.py`

```python
class ContextManager:
    def append(self, message: Dict[str, Any]) -> None
    def get_working_memory(self) -> List[Dict[str, Any]]
    def compress(self, strategy: str = "threshold") -> None
    def store(self, key: str, value: Any, namespace: str = "default") -> None
    def retrieve(self, query: str, namespace: str = "default", top_k: int = 5) -> List[MemoryItem]
    def snapshot(self) -> Dict[str, Any]
```

**方法说明：**

| 方法 | 说明 |
|------|------|
| `append(message)` | 追加消息到工作记忆（短期窗口） |
| `get_working_memory()` | 获取当前工作记忆（经过窗口/压缩后） |
| `compress(strategy)` | 按指定策略压缩工作记忆 |
| `store(key, value, namespace)` | 存储到长期记忆 |
| `retrieve(query, namespace, top_k)` | 按 query 检索长期记忆，返回 top_k 条 |
| `snapshot()` | 生成当前上下文快照（用于调试/可观测） |

**`MemoryItem` 结构：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `content` | `str` | 记忆内容 |
| `namespace` | `str` | 命名空间 |
| `strength` | `float` | 记忆强度（用于衰减） |
| `created_at` | `float` | 创建时间戳 |
| `last_recalled_at` | `float` | 最后召回时间戳 |

**默认实现：** `LayeredContextManager`（`runtime/context_manager/layered.py`）

**五层记忆架构：**
- L1 数据模型：MemoryTrace（不可变）
- L2 衰减与分层：艾宾浩斯衰减 / 活跃区+归档区
- L3 真相源与检索：Markdown 文件 + BM25
- L4 按需注入：`{{memory_context}}` 占位符 + 摘要注入
- L5 显式沉淀：抽取→编码→入库 三阶段

---

## 3. L3 中间件基类与钩子签名

> 源码：`agent_loop.py`

### 3.1 Middleware 基类

```python
class Middleware:
    NAME: str   # 中间件唯一标识

    def before_iteration(self, ctx: Context) -> HookResult:
        """每轮迭代开始前触发"""
        return HookResult.continue_()

    def after_iteration(self, ctx: Context) -> HookResult:
        """每轮迭代结束后触发"""
        return HookResult.continue_()

    def before_llm(self, ctx: Context) -> HookResult:
        """每次 LLM 调用前触发（可修改 messages）"""
        return HookResult.continue_()

    def after_llm(self, ctx: Context, response: Any) -> HookResult:
        """每次 LLM 调用后触发（可解析/修改响应）"""
        return HookResult.continue_()

    def before_tool(self, ctx: Context, tool_call: dict) -> HookResult:
        """每次工具执行前触发（可校验/拦截）"""
        return HookResult.continue_()

    def after_tool(self, ctx: Context, tool_call: dict, result: Any) -> HookResult:
        """每次工具执行后触发（可记录/反馈）"""
        return HookResult.continue_()

    def on_exit_loop(self, ctx: Context) -> HookResult:
        """Agent 循环退出时触发（含正常/异常退出）"""
        return HookResult.continue_()
```

### 3.2 HookResult

```python
class HookResult:
    flow: ControlFlow

    @staticmethod
    def continue_() -> HookResult
        """继续执行后续中间件 → 对应 CONTROL FLOW: CONTINUE"""

    @staticmethod
    def abort(reason: str = "") -> HookResult
        """中止当前循环 → 对应 CONTROL FLOW: ABORT"""

    @staticmethod
    def pause(reason: str = "") -> HookResult
        """暂停（预留） → 对应 CONTROL FLOW: PAUSE"""
```

### 3.3 ControlFlow

```python
class ControlFlow(enum.Enum):
    CONTINUE = "continue"   # 继续
    ABORT = "abort"         # 中止
    PAUSE = "pause"         # 暂停
```

### 3.4 钩子执行顺序

```
BEFORE_ITERATION
  → BEFORE_LLM
    → [LLM 调用]
  → AFTER_LLM
  → BEFORE_TOOL (× N, 每个 tool_call 一次)
    → [工具执行]
  → AFTER_TOOL (× N)
  → AFTER_ITERATION
[循环直到终止]
ON_EXIT_LOOP
```

### 3.5 内置中间件清单（v1.0.0）

| 中间件类 | NAME | 重写的钩子 | 说明 |
|---------|------|-----------|------|
| `ObservabilityMiddleware` | `observability` | 全部 7 个 | metrics/日志/追踪 |
| `SandboxMiddleware` | `sandbox` | `before_tool` | 工具执行隔离 |
| `SafetyGuardMiddleware` | `safety_guard` | `before_tool` | 安全审查 |
| `ConversationRecorderMiddleware` | `conversation_recorder` | `after_iteration` | 对话记录 |
| `MemoryLifecycleMiddleware` | `memory_lifecycle` | `after_iteration`, `on_exit_loop` | 生命周期 |
| `MemoryInjectorMiddleware` | `memory_injector` | `before_llm`, `before_iteration` | 记忆注入 |
| `MemoryConsolidationMiddleware` | `memory_consolidation` | `after_iteration` | 记忆合并 |
| `ToolResultFeedbackMiddleware` | `tool_result_feedback` | `after_tool` | D012 反馈 |
| `ContextCompressorMiddleware` | `context_compressor` | `before_llm` | 上下文压缩 |
| `LoopGuardMiddleware` | `loop_guard` | `before_iteration`, `after_iteration` | 循环切止 |
| `CostGuardMiddleware` | `cost_guard` | `after_llm`, `after_iteration` | 成本控制 |
| `SkillInjectionMiddleware` | `skill_injection` | `before_llm` | 技能注入 |
| `SkillObservationMiddleware` | `skill_observation` | `after_tool` | 技能观测 |
| `DelegationMiddleware` | `delegation` | `before_tool` | 委派拦截 |
| `OrchestratorMiddleware` | `orchestrator` | `after_iteration` | 多 Agent 编排 |

---

## 4. L4 多 Agent 接口

### 4.1 AgentRegistry

> 源码：`runtime/agent_registry/interface.py`

```python
class AgentRegistry:
    def register(self, spec: AgentSpec) -> None
    def unregister(self, agent_id: str) -> None
    def get(self, agent_id: str) -> Optional[AgentSpec]
    def list(self) -> List[AgentSpec]
```

**`AgentSpec` 数据结构：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `agent_id` | `str` | Agent 标识 |
| `model` | `str` | 模型名 |
| `system_prompt` | `str` | system prompt |
| `tool_whitelist` | `Optional[List[str]]` | 工具白名单 |
| `context_namespace` | `str` | 上下文命名空间 |

### 4.2 SharedState

> 源码：`runtime/shared_state/interface.py`

```python
class SharedStateStore:
    def get(self, key: str) -> Optional[Any]
    def set(self, key: str, value: Any) -> None
    def delete(self, key: str) -> bool
    def list_keys(self, prefix: str = "") -> List[str]
    def clear(self) -> None
```

### 4.3 Delegation（委派）

通过 `ctx.shared` 约定的键协作：

| 键 | 类型 | 说明 |
|----|------|------|
| `pending_delegation` | `dict` | `DelegationMiddleware` 写入的待执行委派请求 |
| `delegation_results` | `List[dict]` | `_DelegationDispatcherMiddleware` 写入的执行结果 |
| `delegation_rejected` | `List[dict]` | 被拒绝的委派请求（原因 + 目标 Agent） |

---

## 5. L5 技能系统接口

> 源码：`runtime/skill_system/`

### 5.1 SkillStore

```python
class SkillStore:
    def add(self, skill: SkillTemplate) -> None
    def get(self, skill_id: str) -> Optional[SkillTemplate]
    def list(self) -> List[SkillTemplate]
    def update(self, skill: SkillTemplate) -> None
    def remove(self, skill_id: str) -> None
```

### 5.2 SkillSelector

```python
class SkillSelector:
    def select(self, task: str, top_k: int = 3) -> List[SkillTemplate]
```

### 5.3 SkillMutator

```python
class SkillMutator:
    def mutate(self, skill: SkillTemplate, feedback: dict) -> SkillTemplate
```

### 5.4 SkillEvaluator

```python
class SkillEvaluator:
    def evaluate(self, skill_id: str, execution_log: dict) -> float
```

### 5.5 SkillMonitor

```python
class SkillMonitor:
    def record_execution(self, skill_id: str, metrics: dict) -> None
    def get_stats(self, skill_id: str) -> dict
```

---

## 6. 基础设施接口

### 6.1 EventLog

> 源码：`runtime/event_log/interface.py`

```python
class EventLog:
    def emit(self, event_type: str, **kwargs) -> None
    def query(self, filters: Dict[str, Any] = None, limit: int = 100) -> List[Dict]
    def get_events(self, session_id: str = None) -> List[Dict]
    def close(self) -> None
```

**内置后端：**
- `InMemoryEventLog`（`runtime/event_log/in_memory.py`）—— 内存，测试用
- `SQLiteEventLog`（`runtime/event_log/sqlite_log.py`）—— SQLite，生产用

### 6.2 SessionStore

> 源码：`runtime/session_store/__init__.py`

```python
class SessionStore:
    def save(self, session: Session) -> None
    def load(self, session_id: str) -> Optional[Session]
    def delete(self, session_id: str) -> bool
    def close(self) -> None
```

### 6.3 EventRecorder

> 源码：`runtime/event_recorder.py`

```python
class EventRecorder:
    def __init__(self, event_log: EventLog)
    def bind(self, session: Session) -> None
    def record_loop_start(self, ctx: Context, session: Session) -> None
    def record_loop_exit(self, ctx: Context, session: Session) -> None
    def record_error(self, exc: BaseException, ctx: Context, session: Session) -> None
    def on_iteration(self) -> None   # 递增轮次计数
```

### 6.4 Session

> 源码：`runtime/session.py`

```python
class Session:
    session_id: str
    event_log: EventLog
    metadata: Dict[str, Any]
    budget: Budget
    context_snapshot: Optional[Dict[str, Any]]
```

**`Budget`：**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_tokens` | `int` | `0` | token 上限（0=不限制） |
| `max_cost` | `float` | `0.0` | 成本上限（0.0=不限制） |

### 6.5 可观测性

> 源码：`runtime/observability/`

**Metrics（`runtime/observability/metrics.py`）：**

| 指标 | 类型 | 说明 |
|------|------|------|
| `agent_task_total` | Counter | 任务总数 |
| `agent_task_success_rate` | Gauge | 任务成功率 |
| `agent_task_duration_seconds` | Histogram | 任务耗时分布 |
| `llm_request_count` | Counter | LLM 调用次数 |
| `llm_token_usage` | Counter | Token 消耗 |
| `tool_call_count` | Counter | 工具调用次数 |
| `tool_call_success_rate` | Gauge | 工具调用成功率 |
| `middleware_errors` | Counter | 中间件错误计数 |
| `session_count` | Gauge | 活跃 session 数 |

**健康检查（`runtime/health.py`）：**
- `GET /health` → `{"status": "ok", "version": "1.0.0"}`
- `GET /metrics` → Prometheus text format

**Tracing（`runtime/observability/tracing.py`）：**
- Span 类型：`session` / `iteration` / `llm_call` / `tool_call` / `middleware`
- 记录 duration、status、attributes

---

## 7. 配置项清单

> 配置 schema 定义：`config/schema.json`
> 默认配置文件：`config/default.yaml`
> 版本：`schema_version: "1.0.0"`

### 7.1 LLM 配置

| 配置项 | 路径 | 类型 | 默认值 | 说明 |
|--------|------|------|--------|------|
| 主模型 | `llm.model` | `str` | `"qwen2.5:7b-instruct"` | 主模型名称 |
| provider | `llm.provider` | `str` | `"ollama"` | LLM 提供商 |
| 基础 URL | `llm.base_url` | `str` | `"http://localhost:11434"` | API 端点 |
| max_tokens | `llm.max_tokens` | `int` | `4096` | 单轮输出上限 |
| temperature | `llm.temperature` | `float` | `0.7` | 生成温度 |
| max_retries | `llm.max_retries` | `int` | `2` | 调用重试次数 |
| retry_delay | `llm.retry_delay` | `float` | `1.0` | 重试间隔（秒） |
| max_context_tokens | `llm.max_context_tokens` | `int` | `128000` | 上下文窗口上限 |

### 7.2 降级模型

| 配置项 | 路径 | 类型 | 默认值 | 说明 |
|--------|------|------|--------|------|
| fallback 1 | `llm.fallback_models[0]` | `str` | `"qwen2.5:3b"` | 第一降级模型 |
| fallback 2 | `llm.fallback_models[1]` | `str` | `"qwen2:7b"` | 第二降级模型 |
| fallback 3 | `llm.fallback_models[2]` | `str` | `"llama3.1:8b"` | 第三降级模型 |

### 7.3 运行时配置

| 配置项 | 路径 | 类型 | 默认值 | 说明 |
|--------|------|------|--------|------|
| max_iterations | `runtime.max_iterations` | `int` | `10` | 单任务最大迭代轮数 |
| max_delegation_depth | `runtime.max_delegation_depth` | `int` | `3` | 委派深度上限 |
| schema_version | `schema_version` | `str` | `"1.0.0"` | 配置 schema 版本 |

### 7.4 存储配置

| 配置项 | 路径 | 类型 | 默认值 | 说明 |
|--------|------|------|--------|------|
| event_log 后端 | `storage.event_log_backend` | `str` | `"sqlite"` | `"memory"` / `"sqlite"` |
| 数据库路径 | `storage.db_path` | `str` | `"data/deep_agent.db"` | SQLite 文件路径 |
| session_store 后端 | `storage.session_store_backend` | `str` | `"sqlite"` | `"memory"` / `"sqlite"` |

### 7.5 可观测性配置

| 配置项 | 路径 | 类型 | 默认值 | 说明 |
|--------|------|------|--------|------|
| metrics 端口 | `observability.metrics_port` | `int` | `9090` | Prometheus metrics 端口 |
| 健康检查端口 | `observability.health_port` | `int` | `8080` | 健康检查端点端口 |
| 日志级别 | `observability.log_level` | `str` | `"INFO"` | 日志级别 |

### 7.6 中间件配置

| 配置项 | 路径 | 类型 | 默认值 | 说明 |
|--------|------|------|--------|------|
| 启用清单 | `middlewares.enabled` | `List[str]` | 全量 | 启用的中间件清单 |
| 内存压缩阈值 | `middlewares.context_compressor.threshold_tokens` | `int` | `100000` | 触发压缩的 token 阈值 |

### 7.7 沙箱配置

| 配置项 | 路径 | 类型 | 默认值 | 说明 |
|--------|------|------|--------|------|
| provider | `sandbox.provider` | `str` | `"local"` | `"local"` / `"docker"` |
| 工作目录 | `sandbox.work_dir` | `str` | `"./workspace"` | 沙箱工作目录 |
| 预热池大小 | `sandbox.warm_pool_size` | `int` | `0` | Docker 沙箱预热池大小 |

### 7.8 弹性配置

| 配置项 | 路径 | 类型 | 默认值 | 说明 |
|--------|------|------|--------|------|
| 重试次数 | `resilience.retry.max_attempts` | `int` | `3` | 通用重试次数 |
| 熔断阈值 | `resilience.circuit_breaker.failure_threshold` | `int` | `5` | 触发熔断的失败次数 |
| 熔断恢复时间 | `resilience.circuit_breaker.recovery_timeout` | `float` | `30.0` | 熔断后恢复等待（秒） |

---

## 附录：报告引用

| 报告 | 文件 | 与本 API 文档相关的结论 |
|------|------|------------------------|
| E1.5 | `E1.5_REPORT.md` | ToolCallParser 对 F8 类失败的处理行为 |
| E2 | `acceptance-e2/E2_REPORT.md` | Runtime 100 次运行 0 异常（验证稳定性） |
| E3 | `acceptance-e3/E3_REPORT.md` | LLMAdapter 性能数据 P95/P50 = 1.99 |
| E3.5 | `acceptance-e3.5/E3.5_REPORT.md` | MultiProviderAdapter 降级与 fallback 行为 |
| E4 | `acceptance-e4/E4_REPORT.md` | EventLog + ObservabilityMiddleware 的故障定位数据 |
| E5 | `acceptance-e5/E5_REPORT.md` | 配置变更/降级/扩展 API 的运维验证 |
| E6 | `E6_DECISION.md` | 所有接口的可上线确认 |