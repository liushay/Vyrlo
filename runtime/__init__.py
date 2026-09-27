"""
运行时基础组件层 — 为 Agent Loop 提供可插拔的基础设施。

本包分为两部分：

L2 四组件（执行组件）：
- tool_registry:    工具注册、发现、schema 导出、执行
- tool_call_parser: LLM 响应中的工具调用解析
- llm_adapter:      统一多提供商的 LLM 调用接口
- context_manager:  消息历史、工作记忆、长期记忆管理

L2 装配层（基础设施 + 装配）：
- event_log:       结构化事件日志（后端可插拔：memory / sqlite）
- session:         会话状态与预算（Session / Budget）
- session_store:   会话持久化（后端可插拔：memory / sqlite）
- event_recorder:  Agent Loop 生命周期事件记录器
- runtime:         Runtime 装配类（ComponentRegistry + Runtime）

[批次 D3] 部署与配置管理：
- version:         语义化版本（SemVer）与 schema 版本
- config:          统一配置加载 / 校验（YAML + 环境变量覆盖 + 迁移）
- health:          健康检查（/health 端点 + LLM / 沙箱 / 存储依赖检查）

用法::

    from runtime import Runtime, ComponentRegistry, create_event_log
    from runtime import Session, create_session_store

注意：为避免与既有代码的导入风格冲突，本包仅暴露子模块名（不做符号提升）。
具体符号请从对应子模块导入，例如 `from runtime.runtime import Runtime`。
"""

__all__ = [
    # L2 四组件
    "tool_registry",
    "tool_call_parser",
    "llm_adapter",
    "context_manager",
    # L2 装配层
    "event_log",
    "session",
    "session_store",
    "event_recorder",
    "runtime",
    # [D3] 部署与配置管理
    "version",
    "config",
    "health",
]
