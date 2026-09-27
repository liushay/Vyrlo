# 架构文档 — Vyrlo / Deep Agent v1.0.0

> **版本**：v1.0.0
> **最后更新**：2026-09-27 (Asia/Shanghai)
> **对应 tag**：`v1.0.0`

---

## 目录

1. [概述](#1-概述)
2. [五层架构](#2-五层架构)
3. [依赖图](#3-依赖图)
4. [核心抽象](#4-核心抽象)
5. [数据流：一次 runtime.run() 的完整路径](#5-数据流一次-runtimerun-的完整路径)
6. [扩展点](#6-扩展点)

---

## 1. 概述

Deep Agent 是一个五层可插拔的 Agent 运行时框架。每一层对上层提供服务，对下层暴露接口；任意一层的实现均可替换，无需修改相邻层代码。

### 设计目标

- **可组合**：中间件、工具、Agent、技能均可独立注册与排列
- **可观测**：EventLog、ObservabilityMiddleware、Tracing 三层可观测性（E4 验证：3 次故障定位均 2-3 分钟内）
- **可运维**：回滚 0.74s、配置热加载、多级降级（E5 验证：可配置 4/4、可降级 3/3、可扩展 3/3）
- **弹性**：重试 / 熔断 / 降级 / 并发控制（C1-C3 批次验证）

### 技术栈

| 层面 | 技术 |
|------|------|
| 语言 | Python 3.10+ |
| 模型协议 | OpenAI function calling（原生） |
| 主模型 | qwen2.5:7b-instruct (Ollama, Q4_K_M) |
| 存储 | SQLite（事件日志/会话） |
| 可观测 | Prometheus metrics、JSONL 事件日志、Tracing span |
| 部署 | Docker Compose / 直接 Python 启动 |

---

## 2. 五层架构

```
┌─────────────────────────────────────────────────┐
│  L5: 技能自进化系统                                │
│  SkillSystem = Store + Selector + Mutator         │
│  + Evaluator + Monitor                            │
├─────────────────────────────────────────────────┤
│  L4: 多 Agent 编排                                │
│  AgentRegistry + SharedState + Delegation +       │
│  Orchestrator                                     │
├─────────────────────────────────────────────────┤
│  L3: 中间件层（15 个内置中间件）                     │
│  记忆 / 安全 / 可观测 / 隔离 / 弹性 / 技能注入 /     │
│  多 Agent 委派                                    │
├─────────────────────────────────────────────────┤
│  L2: 四基础组件 + 装配层                           │
│  LLMAdapter + ToolRegistry + ContextManager       │
│  + ToolCallParser  →  Runtime 装配                │
├─────────────────────────────────────────────────┤
│  L1: Agent 内核                                   │
│  AgentLoop + Context + Fiber + Middleware(基类)   │
│  + HookResult + ControlFlow                       │
└─────────────────────────────────────────────────┘
```

### L1 — Agent 内核

> 源码：`agent_loop.py`

L1 是框架的核心，提供 Agent 执行循环、上下文管理、中间件注册和 Fiber 生命周期管理。它不依赖任何 L2～L5 的具体实现。

**核心类：**

| 类 | 职责 |
|-----|------|
| `AgentLoop` | 执行循环：接收消息 → 调 LLM → 检查 tool_calls → 执行工具 → 循环直到终止。公开 `run(ctx, messages)`。通过注入的 `llm_call` 和 `tool_executor` 回调与外部交互。 |
| `Context` | 线程局部上下文：`.shared`（中间件约定的依赖字典）、`.set_tool_calls()` / `.get_tool_calls()`（内部桥接）。 |
| `Fiber` | 插件生命周期容器：包含一个中间件列表，支持 `activate()` / `deactivate()`。 |
| `Middleware` | 中间件基类：定义 7 个钩子（见下方） |
| `HookResult` | 钩子返回值：`continue_()` / `abort()` / `pause()` |
| `ControlFlow` | 枚举：`CONTINUE` / `ABORT` / `PAUSE` |

**7 个钩子执行顺序：**
```
BEFORE_ITERATION
  → BEFORE_LLM
    → [LLM 调用]
  → AFTER_LLM
  → BEFORE_TOOL (每轮可多次，与 tool_call 一一对应)
    → [工具执行]
  → AFTER_TOOL (每轮可多次)
  → AFTER_ITERATION
ON_EXIT_LOOP
```

### L2 — 四基础组件 + 装配层

> 源码：`runtime/` 目录，`ARCHITECTURE.md`（根目录）

详细设计见根目录 `ARCHITECTURE.md`，此处仅做分层摘要。

| 组件 | 接口 | 默认实现 | 职责 |
|------|------|---------|------|
| **LLMAdapter** | `runtime/llm_adapter/interface.py` | `MultiProviderAdapter` | 多提供商 LLM 调用：OpenAI/Ollama/Anthropic。成本追踪、重试、降级。 |
| **ToolRegistry** | `runtime/tool_registry/interface.py` | `InMemoryToolRegistry` | 工具注册、Schema 导出、执行与超时隔离。支持装饰器/MCP/外部 API 三种来源。 |
| **ContextManager** | `runtime/context_manager/interface.py` | `LayeredContextManager` | 五层记忆系统：短期窗口/长期存储/检索/衰减/注入/沉淀。 |
| **ToolCallParser** | `runtime/tool_call_parser/interface.py` | `AutoDetectParser` | 解析 LLM 响应中的工具调用，支持 OpenAI/Anthropic/文本回退三种格式。 |
| **Runtime** | `runtime/runtime.py` | — | 装配层：把四组件装配为可运行的 Agent。内部桥接中间件 `_ToolCallBridgeMiddleware` 在 AFTER_LLM 解析 tool_calls。 |

**Runtime 依赖组件注册表：**
```python
ComponentRegistry(
    llm_adapter=...,       # 必需
    tool_registry=...,      # 必需
    context_manager=...,    # 必需
    tool_call_parser=...,   # 可选，默认 AutoDetectParser
    event_log=...,          # 可选，默认内存 EventLog
    session_store=...,      # 可选，默认内存 SessionStore
)
```

### L3 — 中间件层

> 源码：`runtime/builtin_middlewares/`

L3 中间件是"即插即用"的功能增强模块。每个中间件继承 `Middleware` 基类，重写一个或多个钩子。

**内置中间件清单：**

| 中间件 | 钩子 | 职责 | 批次验证 |
|--------|------|------|:---:|
| `conversation_recorder` | AFTER_ITERATION | 记录每轮对话消息到 EventLog | B1 |
| `memory_lifecycle` | AFTER_ITERATION / ON_EXIT_LOOP | Session 生命周期管理、上下文快照 | B2 |
| `memory_injector` | BEFORE_ITERATION | 跨 session 记忆注入（`{{memory_context}}` 占位符） | B2 |
| `memory_consolidation` | AFTER_ITERATION | 记忆合并：抽取→编码→入库 | B2 |
| `tool_result_feedback` | AFTER_TOOL | 工具结果人类自然语言反馈注回上下文（D012 机制） | C2 |
| `context_compressor` | BEFORE_LLM | 上下文压缩防超窗口 | C2 |
| `loop_guard` | BEFORE_ITERATION / AFTER_ITERATION | 循环检测 + 冗余切止 | C1 |
| `cost_guard` | AFTER_LLM / AFTER_ITERATION | 成本上限控制（本地模型不触发） | C2 |
| `observability` | 全部 7 钩子 | metrics 计数、日志、追踪 | D1 |
| `sandbox` | BEFORE_TOOL | 工具执行隔离（LocalProvider / DockerProvider） | B3 |
| `safety_guard` | BEFORE_TOOL | 工具调用安全审查（白名单/参数校验） | B1 |
| `skill_injection` | BEFORE_LLM | 活跃技能注入到 system prompt | B5 |
| `skill_observation` | AFTER_TOOL | 技能执行观测、触发技能演进决策 | B5 |
| `delegation` | BEFORE_TOOL | 拦截 `delegate_to_agent` 工具调用，写入 `pending_delegation` | B4 |
| `orchestrator` | AFTER_ITERATION | 多 Agent 编排：根据任务描述选择最优 Agent | B4 |

**中间件执行顺序（优先级排序）：**

中间件按注册顺序执行，但可通过 `priority` 属性调整。当前装配层中，内部桥接中间件 (`__runtime_tool_call_bridge__`) 首先注册，保证 AFTER_LLM 中 tool_calls 最先被解析，后续中间件可修改。

### L4 — 多 Agent 编排

> 源码：`runtime/agent_registry/`、`runtime/shared_state/`、`runtime/builtin_middlewares/delegation.py`、`runtime/builtin_middlewares/orchestrator.py`

| 组件 | 接口 | 默认实现 | 职责 |
|------|------|---------|------|
| **AgentRegistry** | `runtime/agent_registry/interface.py` | `InMemoryAgentRegistry` | 注册/查询可委派 Agent 的规格（model、tool_whitelist、system_prompt） |
| **SharedState** | `runtime/shared_state/interface.py` | `InMemorySharedState` | 跨 Agent 共享状态（K/V 存储） |
| **Delegation** | 内置中间件 | — | 拦截 `delegate_to_agent` 工具调用，写入 `pending_delegation`，由 `_DelegationDispatcherMiddleware` 在 AFTER_ITERATION 执行委派 |
| **Orchestrator** | 内置中间件 | — | 根据任务描述自动选择最优子 Agent，编排执行流程 |

**委派流程：**
```
父 Agent 调用 delegate_to_agent(spec, goal)
  → DelegationMiddleware (BEFORE_TOOL) 拦截，写入 pending_delegation
  → _DelegationDispatcherMiddleware (AFTER_ITERATION) 检测标记
  → _run_delegation：构造子 Context（继承 sandbox/shared_state/agent_registry）
  → 子 AgentLoop 执行
  → 结果写回 parent_ctx.shared["delegation_results"]
```

委派深度上限：`max_delegation_depth = 3`（E5 可配置验证通过）。

### L5 — 技能自进化系统

> 源码：`runtime/skill_system/`

SkillSystem 是一套独立的技能生命周期管理子系统，负责技能的存储、选择、变异、评估与监控。

| 组件 | 文件 | 职责 |
|------|------|------|
| **SkillStore** | `runtime/skill_system/store.py` | 技能持久化存储（CRUD） |
| **SkillSelector** | `runtime/skill_system/selector.py` | 根据任务上下文选择最匹配的技能 |
| **SkillMutator** | `runtime/skill_system/mutator.py` | 基于执行反馈变异/优化技能模板 |
| **SkillEvaluator** | `runtime/skill_system/evaluator.py` | 评估技能执行效果 |
| **SkillMonitor** | `runtime/skill_system/monitor.py` | 监控技能执行指标 |

**与中间件的协作：**
- `skill_injection` 中间件在 BEFORE_LLM 注入活跃技能到 system prompt
- `skill_observation` 中间件在 AFTER_TOOL 观测技能执行 → 触发进化决策
- SkillSystem 作为可选组件注入 Runtime，None 时所有 skill 中间件静默放行（零开销）

---

## 3. 依赖图

### 3.1 层间依赖

```
L5 (SkillSystem)
 │
 ├──→ L3 (skill_injection, skill_observation)
 │
L4 (AgentRegistry, SharedState, Delegation, Orchestrator)
 │
 ├──→ L3 (delegation, orchestrator)
 │
L3 (Middlewares)
 │
 ├──→ L1 (Middleware 基类, Context.shared)
 ├──→ L2 (LLMAdapter, ToolRegistry, ContextManager, ToolCallParser — 通过 ctx.shared)
 │
L2 (Runtime + 四组件)
 │
 ├──→ L1 (AgentLoop, Context, Fiber, Middleware 基类)
 │
L1 (AgentLoop, Context, Fiber, Middleware)
 │
 └── [无外部依赖]
```

### 3.2 L2 内部依赖

```
Runtime ────→ LLMAdapter (llm_call 回调)
    │
    ├────→ ToolRegistry (tool_executor 回调)
    │
    ├────→ ToolCallParser (AFTER_LLM 钩子中解析 tool_calls)
    │
    ├────→ ContextManager (ctx.shared 注入)
    │
    ├────→ EventLog (EventRecorder 使用)
    │
    └────→ SessionStore (Session 持久化)
```

### 3.3 L3 中间件依赖（通过 ctx.shared 键）

```
observability ──→ event_log
sandbox ────────→ tool_registry (白名单)
memory_* ───────→ context_manager, session
safety_guard ───→ tool_registry
delegation ─────→ agent_registry, shared_state
orchestrator ───→ agent_registry
skill_* ────────→ skill_system
cost_guard ─────→ session.budget
loop_guard ─────→ (无, 仅跟踪轮次/内容)
```

---

## 4. 核心抽象

### AgentLoop

```python
class AgentLoop:
    def __init__(self, llm_call, tool_executor, max_iterations=10)
    def run(ctx: Context, messages: List[dict]) -> Context
    def register_middleware(middleware: Middleware)
    def load_plugins(paths: List[str], fiber_id: str) -> Fiber
    def dispose_fiber(fiber_id: str)
```

- `llm_call` 回调：接收 messages，返回 LLM 原始响应
- `tool_executor` 回调：接收 tool_name + tool_args，返回工具执行结果
- 主循环终止条件：无更多 tool_calls、ABORT 返回、max_iterations 耗尽

### Context

```python
class Context:
    shared: dict           # 中间件约定的依赖字典
    def set_tool_calls(calls: List[dict])
    def get_tool_calls() -> List[dict]
```

`ctx.shared` 约定的键（由 Runtime._build_context 注入）：

| 键 | 类型 | 描述 |
|----|------|------|
| `event_log` | EventLog | 事件日志后端 |
| `session` | Session | 当前会话对象 |
| `context_manager` | ContextManager | 上下文管理器 |
| `tool_registry` | ToolRegistry | 工具注册表 |
| `system_prompt_template` | str | system prompt 模板 |
| `memory_namespace` | str | 记忆命名空间 |
| `tool_whitelist` | List[str] | 工具白名单 |
| `agent_registry` | AgentRegistry | Agent 注册表（L4） |
| `shared_state` | SharedStateStore | 共享状态（L4） |
| `skill_system` | SkillSystem | 技能系统（L5） |
| `sandbox` | SandboxProvider | 沙箱句柄（SandboxMiddleware 写入） |

### Middleware

```python
class Middleware:
    NAME: str

    # 7 个钩子，每个返回 HookResult
    def before_iteration(self, ctx: Context) -> HookResult: ...
    def after_iteration(self, ctx: Context) -> HookResult: ...
    def before_llm(self, ctx: Context) -> HookResult: ...
    def after_llm(self, ctx: Context, response: Any) -> HookResult: ...
    def before_tool(self, ctx: Context, tool_call: dict) -> HookResult: ...
    def after_tool(self, ctx: Context, tool_call: dict, result: Any) -> HookResult: ...
    def on_exit_loop(self, ctx: Context) -> HookResult: ...
```

### Fiber

```python
class Fiber:
    fiber_id: str
    middlewares: List[Middleware]
    def activate()    # 注册所有中间件到 AgentLoop
    def deactivate()  # 从 AgentLoop 注销所有中间件
```

### HookResult

```python
class HookResult:
    flow: ControlFlow   # CONTINUE | ABORT | PAUSE
    @staticmethod
    def continue_() -> HookResult
    @staticmethod
    def abort(reason="") -> HookResult
    @staticmethod
    def pause(reason="") -> HookResult
```

### ControlFlow

```python
class ControlFlow(enum.Enum):
    CONTINUE = "continue"   # 继续当前循环
    ABORT = "abort"         # 终止当前循环
    PAUSE = "pause"         # 暂停（预留，当前未使用）
```

---

## 5. 数据流：一次 runtime.run() 的完整路径

```
1. 用户调用 runtime.run(session_id, task, system_prompt)

2. Runtime._prepare_session()
   ├── 从 SessionStore 加载或创建 Session
   └── 应用 budget 配置

3. Runtime._build_context()
   ├── 创建 Context 实例
   └── ctx.shared 注入所有约定的依赖键（event_log, session, context_manager,
       tool_registry, system_prompt_template, memory_namespace, tool_whitelist,
       agent_registry, shared_state, skill_system）

4. Runtime._build_messages()
   └── 构造 [system_prompt, user_task] 消息列表

5. Runtime._run_with_ctx()
   ├── EventRecorder.bind(session)
   ├── EventRecorder.record_loop_start()
   │
   ├── self._loop.run(ctx, messages) ────────────────────────────┐
   │                                                              │
   │   ┌─ [BEFORE_ITERATION 钩子] 所有中间件                     │
   │   │   ├── memory_injector: 注入跨 session 记忆              │
   │   │   ├── loop_guard: 循环检测                              │
   │   │   └── observability: 轮次计数                          │
   │   │                                                          │
   │   ├─ [BEFORE_LLM 钩子]                                      │
   │   │   ├── skill_injection: 注入活跃技能到 system prompt     │
   │   │   ├── context_compressor: 压缩上下文                    │
   │   │   └── memory_injector: 注入记忆摘要                    │
   │   │                                                          │
   │   ├─ [LLM 调用] Runtime._llm_call()                         │
   │   │   └── registry.llm_adapter.call(messages, context=ctx)  │
   │   │       ├── MultiProviderAdapter 选择 provider            │
   │   │       ├── 指数退避重试 (max_retries)                    │
   │   │       ├── 失败时降级 provider 链                        │
   │   │       └── 返回 LLMResponse (content + tool_calls)      │
   │   │                                                          │
   │   ├─ [AFTER_LLM 钩子]                                       │
   │   │   ├── __runtime_tool_call_bridge__:                     │
   │   │   │   └── ToolCallParser.parse(response)                │
   │   │   │       └── ctx.set_tool_calls(parsed) ← 写入私有域  │
   │   │   ├── observability: 记录 LLM 调用 metrics             │
   │   │   └── cost_guard: 累计 token 成本                      │
   │   │                                                          │
   │   ├─ [对每个 tool_call:                                     │
   │   │                                                          │
   │   │   ├─ [BEFORE_TOOL 钩子]                                 │
   │   │   │   ├── safety_guard: 白名单/参数校验                 │
   │   │   │   ├── sandbox: 设置沙箱隔离上下文                   │
   │   │   │   ├── delegation: 拦截 delegate_to_agent            │
   │   │   │   └── observability: 工具调用计数                  │
   │   │   │                                                      │
   │   │   ├─ [工具执行] Runtime._tool_executor()                │
   │   │   │   └── registry.tool_registry.execute(name, args,    │
   │   │   │       sandbox=sandbox)                              │
   │   │   │       ├── 参数校验（JSON Schema）                   │
   │   │   │       ├── asyncio.wait_for(timeout)                 │
   │   │   │       └── 返回 ToolResult                           │
   │   │   │                                                      │
   │   │   └─ [AFTER_TOOL 钩子]                                  │
   │   │       ├── tool_result_feedback: D012 自然语言反馈       │
   │   │       ├── skill_observation: 技能执行观测               │
   │   │       └── observability: 工具调用 metrics              │
   │   │                                                          │
   │   │  ]                                                      │
   │   │                                                          │
   │   ├─ [AFTER_ITERATION 钩子]                                 │
   │   │   ├── __runtime_delegation_dispatcher__:                │
   │   │   │   └── 检测 pending_delegation → _run_delegation()   │
   │   │   │       ├── _build_child_context (继承 sandbox 等)    │
   │   │   │       └── 递归 _run_with_ctx()                     │
   │   │   ├── conversation_recorder: 记录对话轮次               │
   │   │   ├── memory_lifecycle: 会话生命周期管理                │
   │   │   ├── memory_consolidation: 记忆合并（抽→编→入库）     │
   │   │   ├── loop_guard: 冗余检测                              │
   │   │   ├── orchestrator: 多 Agent 编排                      │
   │   │   └── observability: 轮次 metrics                     │
   │   │                                                          │
   │   └─ [循环直到：                                              │
   │       - 无更多 tool_calls                                   │
   │       - ABORT 返回                                          │
   │       - max_iterations 耗尽]                                │
   │                                                              │
   │   ┌─ [ON_EXIT_LOOP 钩子]                                    │
   │   │   ├── memory_lifecycle: 会话关闭沉淀记忆                │
   │   │   └── observability: 任务结束 metrics                  │
   │   │                                                          │
   │   └─ 返回 result_ctx                                        │
   └──────────────────────────────────────────────────────────────┘
   │
   ├── EventRecorder.record_loop_exit()
   └── session_store.save(session)
       └── 返回 session
```

---

## 6. 扩展点

### 6.1 如何添加中间件

```python
from runtime.builtin_middlewares import observability
from agent_loop import Middleware, HookResult

# 方式 1：使用内置中间件
runtime.register_middleware(observability.ObservabilityMiddleware())

# 方式 2：自定义中间件
class MyMiddleware(Middleware):
    NAME = "my_custom"

    def before_llm(self, ctx):
        print(f"[MY] 即将调用 LLM，当前消息数: {len(ctx.shared['context_manager'].get_working_memory())}")
        return HookResult.continue_()

runtime.register_middleware(MyMiddleware())
```

### 6.2 如何添加工具

```python
from runtime.tool_registry import tool

# 方式 1：装饰器注册
@tool(description="读取文件内容", params_schema={...})
def my_custom_tool(filepath: str) -> str:
    ...

# 方式 2：手动注册
from runtime.tool_registry.schema import Tool
tool_obj = Tool(name="my_tool", fn=my_custom_tool, ...)
registry.tool_registry.register(tool_obj)
```

### 6.3 如何添加技能

```python
from runtime.skill_system.models import SkillTemplate
from runtime.skill_system.store import SkillStore

# 注册技能模板
skill = SkillTemplate(
    name="web_search",
    description="使用搜索引擎查找信息",
    prompt_fragment="你可以使用 web_search 工具搜索最新信息...",
    trigger_keywords=["搜索", "查一下", "最新"],
)
store = runtime._skill_system.store
store.add(skill)
```

### 6.4 如何接入新模型

```python
from runtime.llm_adapter.interface import Provider

class MyProvider(Provider):
    def call(self, messages, **kwargs) -> LLMResponse:
        # 实现调用逻辑
        ...

    def stream(self, messages, **kwargs) -> Iterator[str]:
        ...

    def count_tokens(self, messages) -> int:
        ...

    def get_cost(self, usage) -> float:
        ...

# 注册到 MultiProviderAdapter
from runtime.llm_adapter.multi_provider import MultiProviderAdapter
adapter = MultiProviderAdapter()
adapter.register_provider("my_model", MyProvider(...))
```

### 6.5 EventLog 后端切换

```python
from runtime.event_log.sqlite_log import SQLiteEventLog
from runtime.event_log.in_memory import InMemoryEventLog

# SQLite（持久化，生产环境推荐）
event_log = SQLiteEventLog("data/deep_agent.db")

# 内存（测试/开发）
event_log = InMemoryEventLog()

# 注入 Runtime
registry = ComponentRegistry(
    ...,
    event_log=event_log,
)
```

### 6.6 配置 Schema 扩展

系统配置由 `config/schema.json` 校验，`config/default.yaml` 提供默认值。添加新配置项：

1. 在 `config/schema.json` 中定义新字段的 JSON Schema
2. 在 `config/default.yaml` 中提供默认值
3. 在对应的 Python 组件中通过 `ctx.shared` 或配置加载器读取

版本迁移由 `runtime/config/migrations.py` 管理，当前 `schema_version: "1.0.0"`。

---

## 附录：关键报告引用

| 报告 | 文件 | 关键结论 |
|------|------|---------|
| E1 基线 | `acceptance-e1/` | 50 任务 × qwen2.5:3b，原始成功率 76% |
| E1.5 分析 | `E1.5_REPORT.md` | 有效成功率 96%；F8 模型能力边界 |
| E2 可靠性 | `acceptance-e2/E2_REPORT.md` | 100 次 0 异常/0 污染/内存波动 5.66% |
| E3 性能 | `acceptance-e3/E3_REPORT.md` | P95/P50 = 1.99 ≤ 5 |
| E3.5 校准 | `acceptance-e3.5/E3.5_REPORT.md` | F1 修复 Ollama 环境有效 |
| E4 可观测 | `acceptance-e4/E4_REPORT.md` | 3 次故障定位 2-3 分钟 |
| E5 运维 | `acceptance-e5/E5_REPORT.md` | 回滚 0.74s / 可配置 4/4 / 可降级 3/3 / 可扩展 3/3 |
| E6 决策 | `E6_DECISION.md` | 五条标准全部通过，判定可上线 |