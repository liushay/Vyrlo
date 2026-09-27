"""
[L3-1] builtin_middlewares — L3 第一批内置中间件。

本包提供 5 个可独立启停的内置中间件，全部只依赖 EventLog / Session /
Context / ToolRegistry 的公开字段，不修改 L1 / L2 任何代码。

中间件职责边界（互不重叠）：

| 中间件                        | 注册钩子                  | 唯一职责                         | 依赖                            |
|-------------------------------|---------------------------|----------------------------------|---------------------------------|
| ObservabilityMiddleware       | AFTER_LLM / AFTER_TOOL /  | 写事件日志 + 结构化日志 + 聚合   | EventLog                        |
|                               | ON_EXIT_LOOP              | 性能指标（耗时/token/cost）      |                                 |
| CostGuardMiddleware           | AFTER_LLM                 | 累加 token/cost 到预算，超限中止 | EventLog, Session               |
| SafetyGuardMiddleware         | BEFORE_TOOL               | 高危工具跳过，白名单放行         | ToolRegistry（只读元数据）      |
| MemoryInjectorMiddleware      | BEFORE_LLM                | build_context 注入记忆消息       | ContextManager.build_context    |
| ContextCompressorMiddleware   | BEFORE_LLM                 | 超 token 上限时压缩并替换消息    | ContextManager.compress         |

依赖关系（有向，无环）：
    observability  ──写入──>  EventLog  ──读取──>  cost_guard
    observability, cost_guard  ──读写──>  Session.budget
    safety_guard   ──只读──>  ToolRegistry
    memory_injector, context_compressor  ──调用──>  ContextManager

关于 ctx.shared 约定键（用于解耦装配，使每个中间件可独立启停）：
    "event_log"              : EventLog 实例
    "session"                : Session 实例
    "context_manager"        : ContextManager 实例
    "tool_registry"          : ToolRegistry 实例
    "system_prompt_template" : system prompt 模板（含 {{memory_context}}）
    "memory_namespace"       : 长期记忆命名空间
    "tool_whitelist"         : 运行时工具白名单（可迭代）

加载方式：
    可由 AgentLoop.load_plugins 动态加载本目录（每个 .py 文件会被扫描），
    也可显式 import 后通过 register_middleware 注册。

[L3-1] 本文件为 L3 第一批中间件新增，未修改任何 L1 / L2 代码。
"""

from runtime.builtin_middlewares.context_compressor import ContextCompressorMiddleware
from runtime.builtin_middlewares.conversation_recorder import ConversationRecorderMiddleware
from runtime.builtin_middlewares.cost_guard import CostGuardMiddleware
from runtime.builtin_middlewares.delegation import DelegationMiddleware
from runtime.builtin_middlewares.memory_consolidation import MemoryConsolidationMiddleware
from runtime.builtin_middlewares.memory_injector import MemoryInjectorMiddleware
from runtime.builtin_middlewares.loop_guard import LoopGuardMiddleware
from runtime.builtin_middlewares.memory_lifecycle import MemoryLifecycleMiddleware
from runtime.builtin_middlewares.observability import ObservabilityMiddleware
from runtime.builtin_middlewares.orchestrator import OrchestratorMiddleware
from runtime.builtin_middlewares.safety_guard import SafetyGuardMiddleware
from runtime.builtin_middlewares.sandbox import SandboxMiddleware
from runtime.builtin_middlewares.skill_injection import SkillInjectionMiddleware
from runtime.builtin_middlewares.skill_observation import SkillObservationMiddleware
from runtime.builtin_middlewares.tool_result_feedback import (
    ToolResultFeedbackMiddleware,
)

__all__ = [
    "ObservabilityMiddleware",
    "CostGuardMiddleware",
    "SafetyGuardMiddleware",
    "MemoryInjectorMiddleware",
    "ContextCompressorMiddleware",
    "MemoryLifecycleMiddleware",
    "ConversationRecorderMiddleware",
    "MemoryConsolidationMiddleware",
    "SandboxMiddleware",
    "DelegationMiddleware",
    "OrchestratorMiddleware",
    "SkillInjectionMiddleware",
    "SkillObservationMiddleware",
    "ToolResultFeedbackMiddleware",
    "LoopGuardMiddleware",
]
