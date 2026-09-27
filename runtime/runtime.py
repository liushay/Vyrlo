"""
[L2-装配] Runtime — Agent 运行时的总装配入口。

定位：
    Runtime 是 L2 装配层的"最后一公里"：把 L2 四组件（LLMAdapter /
    ToolRegistry / ContextManager / ToolCallParser）与两项基础设施
    （EventLog / SessionStore）装配为一个可直接运行的 Agent。

职责（只做装配与生命周期编排，不含业务决策）：
    1. 把 LLMAdapter 接入 AgentLoop 的 `llm_call` 回调。
    2. 把 ToolRegistry 接入 AgentLoop 的 `tool_executor` 回调。
    3. 把 ToolCallParser 的解析结果写入 ctx 私有域（供 Loop 驱动工具流程）。
    4. 通过 ctx.shared 暴露 L3 中间件约定的依赖键。
    5. 管理 Session 的创建、保存与恢复（SessionStore）。
    6. 记录 Loop 生命周期事件（EventRecorder）。

设计约束：
    - **不修改 L1 内核**：只调用 AgentLoop 的公开方法。
    - **不修改 L2 四组件内部逻辑**：只调用其公开接口。
    - L3 中间件由装配方通过 `register_middleware` / `load_plugins` 注入。
    - **唯一例外：[S2] 内部桥接中间件** `_ToolCallBridgeMiddleware`。它由 Runtime
      在构造时自动注册到 `after_llm` 钩子中，用于把 ToolCallParser 的解析结果
      写入 ctx 私有域（`__tool_calls__`）。之所以必须是中间件而非回调逻辑：
      `_llm_call` 在 AFTER_LLM **之前**执行，若在回调里就写入 tool_calls，
      AFTER_LLM 阶段的中间件（职责正是 tool_call 解析与校验）将看不到、也改不了
      它。该中间件为内部实现细节，不暴露给装配方，也不改变装配方注入中间件的既有路径。

用法::

    from runtime.runtime import Runtime, ComponentRegistry

    registry = ComponentRegistry(
        llm_adapter=adapter,
        tool_registry=tool_registry,
        context_manager=context_manager,
    )
    runtime = Runtime(registry, max_iterations=8)
    session = runtime.run("s-1", "帮我查一下北京天气", system_prompt="你是助手")
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent_loop import AgentLoop, Context, Fiber, HookResult, Middleware

from runtime.agent_registry.interface import AgentRegistry, AgentSpec
from runtime.event_log import EVENT_DELEGATION, EventLog, create_event_log
from runtime.event_recorder import EventRecorder
from runtime.llm_adapter.interface import LLMAdapter
from runtime.session import Budget, Session
from runtime.session_store import SessionStore, create_session_store
from runtime.shared_state.interface import SharedStateStore
from runtime.tool_call_parser.interface import ToolCallParser
from runtime.tool_call_parser.parsers import AutoDetectParser
from runtime.tool_registry.interface import ToolRegistry

# ============================================================================
# ComponentRegistry —— 组件注册表
# ============================================================================


@dataclass
class ComponentRegistry:
    """运行时依赖组件集合（装配的输入）。

    前三项为必需组件（L2 四组件中的三个执行组件），后三项有默认实现。

    Attributes:
        llm_adapter:       LLM 适配器（必需）。
        tool_registry:     工具注册表（必需）。
        context_manager:   上下文管理器（必需）。
        tool_call_parser:  工具调用解析器，默认 AutoDetectParser。
        event_log:         事件日志后端，默认 create_event_log("memory")。
        session_store:     会话存储后端，默认 create_session_store("memory")。
    """

    llm_adapter: LLMAdapter
    tool_registry: ToolRegistry
    context_manager: Any
    tool_call_parser: Optional[ToolCallParser] = None
    event_log: Optional[EventLog] = None
    session_store: Optional[SessionStore] = None

    def __post_init__(self) -> None:
        """用工厂补齐未提供的可选组件（避免可变默认值共享）。"""
        if self.tool_call_parser is None:
            self.tool_call_parser = AutoDetectParser()
        if self.event_log is None:
            self.event_log = create_event_log("memory")
        if self.session_store is None:
            self.session_store = create_session_store("memory")


# ============================================================================
# _ToolCallBridgeMiddleware —— Runtime 内部桥接中间件
# ============================================================================


class _ToolCallBridgeMiddleware(Middleware):
    """[内部] 把 LLM 响应解析为 tool_calls 并写入 ctx 私有域。

    为什么需要它：
        Loop 的 AFTER_LLM 钩子职责正是 "tool_call 解析与校验"，但
        `Runtime._llm_call` 是 AFTER_LLM **之前**执行的。若在回调里就写入
        tool_calls，AFTER_LLM 注册的中间件既看不到也改不了它。因此把解析
        从回调搬到本中间件，使 AFTER_LLM 阶段的中间件看到的是"解析后"的
        tool_calls，并可在钩子中覆盖它（后写入者胜出）。

    契约：
        - 输入 response，调用 `Runtime._parse_tool_calls(response)`；
        - 有结果时写入 `ctx.set_tool_calls(calls)`；
        - 无结果时不写入（保留 ctx 中的既有状态），返回 CONTINUE。

    可见性：
        - 由 Runtime 在构造时自动注册，**不对外暴露**，装配方无需也不应
          感知它；装配方通过 `Runtime.register_middleware` 注入的 L3 中间件
          注册在其之后，因此天然晚于本中间件执行。
    """

    #: 内部保留名：以双下划线包裹，避免与 L3 中间件重名冲突
    NAME = "__runtime_tool_call_bridge__"

    def __init__(self, runtime: "Runtime") -> None:
        super().__init__(self.NAME)
        self._runtime = runtime

    def after_llm(self, ctx: Context, response: Any) -> HookResult:
        """解析响应中的工具调用并写入 ctx 私有域约定键。"""
        calls = self._runtime._parse_tool_calls(response)
        if calls:
            ctx.set_tool_calls(calls)
        return HookResult.continue_()


class _IterationCounterMiddleware(Middleware):
    """[内部] 把 Agent Loop 的真实迭代轮次同步给 EventRecorder。

    为什么需要它：
        ``EventRecorder.record_loop_exit`` 会写入 ``iterations`` 指标，但
        Runtime 只调用一次 ``loop.run()``，无法感知 loop 内部的每一轮迭代。
        Agent Loop 的 ``BEFORE_ITERATION`` 钩子每轮迭代开始时都会触发，
        恰是轮次递增的准确时机。该中间件与 ``_ToolCallBridgeMiddleware``
        一样是内部实现细节，不对外暴露，也不改变装配方注入中间件的既有路径。
    """

    #: 内部保留名：以双下划线包裹，避免与 L3 中间件重名冲突
    NAME = "__runtime_iteration_counter__"

    def __init__(self, runtime: "Runtime") -> None:
        super().__init__(self.NAME)
        self._runtime = runtime

    def before_iteration(self, ctx: Context) -> HookResult:
        """每轮迭代开始时递增 EventRecorder 的轮次计数。"""
        self._runtime._recorder.on_iteration()
        return HookResult.continue_()


class _DelegationDispatcherMiddleware(Middleware):
    """[L4][内部] 在每一轮迭代结束后检测并执行委派。

    DelegationMiddleware 在 BEFORE_TOOL 拦截 ``delegate_to_agent`` 后写入
    ``ctx.shared["pending_delegation"]``，本中间件在 AFTER_ITERATION 检测到
    该标记后，委托 Runtime._run_delegation 递归执行目标 Agent，并把结果写回
    父 ctx.shared["delegation_results"]。随后清空标记，父 Loop 继续。
    """

    NAME = "__runtime_delegation_dispatcher__"

    def __init__(self, runtime: "Runtime") -> None:
        super().__init__(self.NAME)
        self._runtime = runtime

    def after_iteration(self, ctx: Context) -> HookResult:
        pending = ctx.shared.get("pending_delegation")
        if not pending:
            return HookResult.continue_()

        # 清空标记（避免重复触发）
        ctx.shared.pop("pending_delegation", None)

        spec = pending.get("spec")
        if spec is None:
            return HookResult.continue_()

        goal = str(pending.get("goal", ""))
        success_criteria = str(pending.get("success_criteria", ""))
        result = self._runtime._run_delegation(ctx, spec, goal, success_criteria)
        ctx.shared.setdefault("delegation_results", []).append(result)
        return HookResult.continue_()


# ============================================================================
# Runtime —— 装配与生命周期编排
# ============================================================================


class Runtime:
    """Agent 运行时装配类。

    把 L2 组件装配为一个可运行的 AgentLoop，并管理与持久化 Session。
    """

    def __init__(
        self,
        registry: ComponentRegistry,
        max_iterations: int = 10,
        agent_registry: Optional[AgentRegistry] = None,
        shared_state: Optional[SharedStateStore] = None,
        max_delegation_depth: int = 3,
        skill_system: Any = None,
    ) -> None:
        """
        Args:
            registry:            组件注册表（ComponentRegistry）。
            max_iterations:      AgentLoop 最大迭代轮数，防止无限循环。
            agent_registry:      可选，Agent 注册表（多 Agent 委派用）。None 时不启用委派。
            shared_state:        可选，共享状态存储（子 Agent 共享用）。None 时不启用。
            max_delegation_depth: 委派深度上限，默认 3。
            skill_system:        可选，技能自进化子系统（SkillSystem）。None 时
                                 所有 skill 中间件静默放行（零开销）。
        """
        self.registry = registry
        self.max_iterations = int(max_iterations or 10)

        # [L5] 技能自进化子系统（可选，None 时不启用）
        self._skill_system = skill_system

        # [L4] 多 Agent 编排：Agent 注册表 / 共享状态 / 委派深度
        self._agent_registry = agent_registry
        self._shared_state = shared_state
        # 注意：0 是有意义的（表示不允许任何委派），不能用 or 回退
        self._max_delegation_depth = (
            3 if max_delegation_depth is None else int(max_delegation_depth)
        )

        # 当前运行的 Context（供 llm_call 回调传入成本追踪）。
        # 使用线程局部存储，避免同一 Runtime 实例并发跑多个会话时互相覆盖。
        self._thread_local = threading.local()

        # [L4] 委派深度线程局部计数（递归委派时递增）
        self._delegation_depth = threading.local()

        # 单例 AgentLoop：回调通过方法绑定 self，便于复用与 Fiber 生命周期管理
        self._loop = AgentLoop(
            llm_call=self._llm_call,
            tool_executor=self._tool_executor,
            max_iterations=self.max_iterations,
        )
        # 装配方注入的中间件列表（不含内部桥接中间件）
        self._middlewares: List[Middleware] = []
        self._fibers: Dict[str, Fiber] = {}
        self._recorder = EventRecorder(event_log=registry.event_log)

        # [S2] 内部桥接中间件：必须在其他 AFTER_LLM 中间件之前注册，
        # 这样后续（装配方注入的）中间件才能看到并修改解析出的 tool_calls。
        self._tool_call_bridge = _ToolCallBridgeMiddleware(self)
        self._loop.register_middleware(self._tool_call_bridge)

        # [内部] 迭代计数中间件：通过 BEFORE_ITERATION 钩子把真实轮次同步给
        # EventRecorder，修复 record_loop_exit 中 iterations 恒为 0 的问题。
        self._iteration_counter = _IterationCounterMiddleware(self)
        self._loop.register_middleware(self._iteration_counter)

        # [L4] 委派调度中间件：在 AFTER_ITERATION 检测 pending_delegation 并执行委派
        self._delegation_dispatcher = _DelegationDispatcherMiddleware(self)
        self._loop.register_middleware(self._delegation_dispatcher)

    # ------------------------------------------------------------------
    # 回调一：llm_call —— 接入 LLMAdapter + ToolCallParser
    # ------------------------------------------------------------------

    @property
    def _active_ctx(self) -> Optional[Context]:
        """当前线程正在运行的 Context（兼容别名）。

        供测试脚手架 / 子 Agent 共享读取使用，等价于 ``_thread_local.active_ctx``。
        """
        return getattr(self._thread_local, "active_ctx", None)

    def _llm_call(self, messages: List[Dict[str, Any]]) -> Any:
        """AgentLoop 的 LLM 回调 —— 只调 LLM，不做 tool_call 解析。

        1. 委托 registry.llm_adapter.call() 获取标准响应。
        2. 直接返回响应；tool_call 解析与写入由 AFTER_LLM 钩子中的
           `_ToolCallBridgeMiddleware` 完成（见 [S2]）。

        为什么不在回调里解析：
            本回调在 AFTER_LLM 钩子**之前**执行。若在此写入 tool_calls，
            AFTER_LLM 阶段的中间件（其职责恰是 tool_call 解析与校验）既看不到
            也改不了它。故解析下沉到桥接中间件，保持"AFTER_LLM 是 tool_call
            解析的唯一入口"这一契约。
        """
        ctx = getattr(self._thread_local, "active_ctx", None)
        return self.registry.llm_adapter.call(messages, context=ctx)

    def _parse_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """解析 LLM 响应中的工具调用，返回 Loop 期望的规范列表。

        返回格式：``List[{"id": str|None, "name": str, "arguments": dict}]``。
        """
        parser = self.registry.tool_call_parser
        if parser is None:
            return []

        candidate = self._parser_input(response)
        if candidate is None:
            return []

        try:
            parsed = parser.parse(candidate)
        except Exception:
            return []

        calls: List[Dict[str, Any]] = []
        for tc in parsed or []:
            name = str(getattr(tc, "name", "") or "")
            if not name:
                continue
            args = getattr(tc, "args", None)
            if not isinstance(args, dict):
                args = {}
            calls.append({"id": getattr(tc, "id", None), "name": name, "arguments": args})
        return calls

    @staticmethod
    def _parser_input(response: Any) -> Any:
        """构造解析器可识别的输入（原生 dict/list 优先，否则规范化 tool_calls）。

        [S4] Anthropic 扁平 tool_calls 的归一化：
            AnthropicProvider 返回的 ``LLMResponse.raw`` 是 ``str``，而
            ``tool_calls`` 是扁平形状 ``[{"id", "type": "tool_use", "name",
            "input"}]``。此前统一包装为 ``{"tool_calls": [...]}`` 交给
            AutoDetectParser，但 ``_detect_anthropic`` 只认 ``content`` 数组中的
            ``tool_use`` 块、``_detect_openai`` 又要求 ``tool_calls[i]["function"]``，
            最终 fallback 到 TextFallbackParser，导致 Anthropic 提供商的工具调用
            被静默丢弃。现在识别出该形状后改构造原生 ``{"content": [...]}``，
            交给 AnthropicParser 的原生路径处理（不改变 OpenAI 形状的检测结果）。
        """
        raw = getattr(response, "raw", None)
        if isinstance(raw, (dict, list)):
            return raw

        if isinstance(response, dict):
            if "choices" in response:
                return response
            content = response.get("content")
            if isinstance(content, list):
                return response
            tcs = response.get("tool_calls")
            if tcs:
                normalized = Runtime._normalize_tool_calls(tcs)
                native = Runtime._as_anthropic_content(normalized)
                return native if native is not None else {"tool_calls": normalized}
            return response

        tcs = getattr(response, "tool_calls", None)
        if tcs:
            normalized = Runtime._normalize_tool_calls(tcs)
            native = Runtime._as_anthropic_content(normalized)
            return native if native is not None else {"tool_calls": normalized}
        return None

    @staticmethod
    def _as_anthropic_content(
        normalized: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """[S4] 把扁平 Anthropic tool_use 列表还原为原生 ``content`` 块形状。

        仅当首项是 ``{"type": "tool_use", ...}`` 时转换；OpenAI function
        calling 形状（``{"type": "function", "function": {...}}``）返回 None，
        继续走既有 ``{"tool_calls": [...]}`` 路径，从而保证检测结果不变。
        """
        if not normalized or not isinstance(normalized[0], dict):
            return None
        if normalized[0].get("type") != "tool_use":
            return None
        blocks = [dict(block) for block in normalized if isinstance(block, dict)]
        return {"content": blocks} if blocks else None

    @staticmethod
    def _normalize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
        """把工具调用列表规范化到 OpenAI function calling 形状。"""
        if not isinstance(tool_calls, (list, tuple)):
            return []

        normalized: List[Dict[str, Any]] = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                tc = {
                    "id": getattr(tc, "id", None),
                    "name": getattr(tc, "name", ""),
                    "arguments": getattr(tc, "arguments", getattr(tc, "args", {})),
                }
            # 已是 OpenAI / Anthropic 原生形状则透传
            if "function" in tc or tc.get("type") == "tool_use":
                normalized.append(tc)
                continue

            name = tc.get("name") or tc.get("tool") or ""
            args = tc.get("arguments", tc.get("args", {}))
            args_str = args if isinstance(args, str) else json.dumps(
                args or {}, ensure_ascii=False
            )
            normalized.append({
                "id": tc.get("id"),
                "type": "function",
                "function": {"name": name, "arguments": args_str},
            })
        return normalized

    # ------------------------------------------------------------------
    # 回调二：tool_executor —— 接入 ToolRegistry
    # ------------------------------------------------------------------

    def _tool_executor(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """AgentLoop 的工具执行回调，委托给 registry.tool_registry.execute()。

        若 ctx.shared["sandbox"] 存在（由 SandboxMiddleware 写入），
        则把沙箱句柄传给 execute，使工具在沙箱内执行。
        """
        sandbox = None
        ctx = getattr(self._thread_local, "active_ctx", None)
        if ctx is not None:
            shared = getattr(ctx, "shared", None)
            if isinstance(shared, dict):
                sandbox = shared.get("sandbox")

        if sandbox is None:
            # 无沙箱：走原路径（向后兼容，不传多余关键字参数）
            return self.registry.tool_registry.execute(tool_name, tool_args or {})
        return self.registry.tool_registry.execute(
            tool_name, tool_args or {}, sandbox=sandbox
        )

    # ------------------------------------------------------------------
    # 中间件 / 插件管理（转发到 AgentLoop）
    # ------------------------------------------------------------------

    def register_middleware(self, middleware: Middleware) -> None:
        """注册中间件（L3 中间件由装配方显式注入）。"""
        if middleware not in self._middlewares:
            self._middlewares.append(middleware)
        self._loop.register_middleware(middleware)

    # ------------------------------------------------------------------
    # Agent 注册表（L4 多 Agent 编排）
    # ------------------------------------------------------------------

    def register_agent(self, spec: AgentSpec) -> None:
        """注册一个可委派 Agent（转发到 agent_registry）。"""
        if self._agent_registry is None:
            from runtime.agent_registry.in_memory import InMemoryAgentRegistry
            self._agent_registry = InMemoryAgentRegistry()
        self._agent_registry.register(spec)

    def list_agents(self) -> List[AgentSpec]:
        """列出已注册的 Agent。"""
        if self._agent_registry is None:
            return []
        return self._agent_registry.list()

    def load_plugins(self, paths: List[str], fiber_id: Optional[str] = None) -> Fiber:
        """动态加载插件并创建 Fiber（转发到 AgentLoop.load_plugins）。"""
        fiber = self._loop.load_plugins(paths, fiber_id=fiber_id)
        self._fibers[fiber.fiber_id] = fiber
        return fiber

    def dispose_fiber(self, fiber_id: str) -> None:
        """销毁 Fiber 并回滚状态（转发到 AgentLoop.dispose_fiber）。"""
        self._fibers.pop(fiber_id, None)
        self._loop.dispose_fiber(fiber_id)

    # ------------------------------------------------------------------
    # Session 管理
    # ------------------------------------------------------------------

    def get_session(self, session_id: str) -> Optional[Session]:
        """从 SessionStore 读取会话，不存在时返回 None。"""
        return self.registry.session_store.load(session_id)

    def resume(self, session_id: str) -> Session:
        """恢复已保存的会话。

        Raises:
            KeyError: 会话不存在时。
        """
        session = self.registry.session_store.load(session_id)
        if session is None:
            raise KeyError(f"会话不存在，无法恢复: {session_id!r}")
        session.event_log = self.registry.event_log
        return session

    # ------------------------------------------------------------------
    # 委派执行（L4 多 Agent 编排）
    # ------------------------------------------------------------------

    def _run_delegation(
        self,
        parent_ctx: Context,
        spec: AgentSpec,
        goal: str,
        success_criteria: str,
    ) -> Dict[str, Any]:
        """递归执行一次委派：用目标 AgentSpec 构造子 Context + 子 Session。

        - 子 ctx.shared["sandbox"] 从父继承（批次 3 的 handle）。
        - 委派深度超限时拒绝并记录，不执行。
        - 委派链路写入 EVENT_DELEGATION 事件。

        Returns:
            委派结果字典。
        """
        depth = getattr(self._delegation_depth, "depth", 0)

        # 深度上限检查
        if depth >= self._max_delegation_depth:
            reason = f"委派深度超限 ({depth} >= {self._max_delegation_depth})"
            self._emit_delegation(parent_ctx, spec.agent_id, goal, ok=False, reason=reason)
            parent_ctx.shared.setdefault("delegation_rejected", []).append({
                "target_agent": spec.agent_id,
                "reason": "depth_exceeded",
            })
            return {
                "target_agent": spec.agent_id,
                "success": False,
                "reason": reason,
                "depth": depth,
            }

        child_session_id = f"{spec.agent_id}-delegated-{uuid.uuid4().hex[:8]}"
        system_prompt = f"你是子 Agent「{spec.agent_id}」，目标：{goal}"

        # 子 Session：沿用父 Runtime 的 event_log / session_store
        child_session = Session(
            session_id=child_session_id,
            event_log=self.registry.event_log,
            metadata={
                "task": goal,
                "parent_session_id": parent_ctx.shared["session"].session_id,
                "agent_id": spec.agent_id,
                "memory_namespace": spec.context_namespace,
            },
        )

        # 子 Context：继承父 sandbox、shared_state、agent_registry
        child_ctx = self._build_child_context(parent_ctx, child_session, system_prompt)

        # 递增委派深度并执行子 Agent
        old_depth = depth
        self._delegation_depth.depth = depth + 1
        try:
            # [D002] 必须直接使用 child_ctx 执行，而不是 self.run(session_id, ...)。
            # run() 会重新 _build_context() 覆盖 ctx，导致 sandbox / shared_state /
            # agent_registry 的继承全部失效。这里改走内部的 _run_with_ctx()。
            child_messages = self._build_messages(goal, system_prompt)
            child_result = self._run_with_ctx(child_ctx, child_messages, child_session)
        finally:
            self._delegation_depth.depth = old_depth

        ok = not child_result.abort_flag
        result = {
            "target_agent": spec.agent_id,
            "success": ok,
            "session_id": child_session_id,
            "goal": goal,
            "depth": depth + 1,
            "child_context": child_ctx,
        }

        self._emit_delegation(
            parent_ctx, spec.agent_id, goal,
            ok=ok, reason="" if ok else "child_aborted",
        )
        return result

    def _build_child_context(
        self,
        parent_ctx: Context,
        child_session: Session,
        system_prompt: str = "",
    ) -> Context:
        """构造子 Context，继承父的 sandbox、shared_state、agent_registry。

        与 ``_build_context`` 保持键集合一致，并额外继承父 ctx 的编排字段。
        """
        child_ctx = Context()
        shared = child_ctx.shared

        # 基础依赖
        shared["event_log"] = self.registry.event_log
        shared["session"] = child_session
        shared["context_manager"] = self.registry.context_manager
        shared["tool_registry"] = self.registry.tool_registry

        # [D002] 与 _build_context 对齐的可选键：否则子 Agent 改用 child_ctx 执行后
        # 会丢失 system prompt / 命名空间 / 白名单 / 技能子系统等能力。
        if system_prompt:
            shared["system_prompt_template"] = system_prompt

        child_metadata = getattr(child_session, "metadata", None) or {}
        namespace = child_metadata.get("memory_namespace")
        if namespace:
            shared["memory_namespace"] = namespace

        whitelist = child_metadata.get("tool_whitelist")
        if whitelist is not None:
            shared["tool_whitelist"] = whitelist

        if self._skill_system is not None:
            shared["skill_system"] = self._skill_system

        # 继承父 sandbox（批次 3 的 handle）
        parent_sandbox = parent_ctx.shared.get("sandbox")
        if parent_sandbox is not None:
            shared["sandbox"] = parent_sandbox

        # 共享状态
        if self._shared_state is not None:
            shared["shared_state"] = self._shared_state
        parent_shared_state = parent_ctx.shared.get("shared_state")
        if parent_shared_state is not None:
            shared["shared_state"] = parent_shared_state

        # Agent 注册表（供子 Agent 内的 DelegationMiddleware 使用）
        if self._agent_registry is not None:
            shared["agent_registry"] = self._agent_registry
        parent_agent_registry = parent_ctx.shared.get("agent_registry")
        if parent_agent_registry is not None:
            shared["agent_registry"] = parent_agent_registry

        return child_ctx

    def _emit_delegation(
        self,
        ctx: Context,
        target_agent: str,
        goal: str,
        ok: bool,
        reason: str = "",
    ) -> None:
        """写入一条委派事件。"""
        try:
            self.registry.event_log.emit(
                EVENT_DELEGATION,
                target_agent=target_agent,
                goal=goal,
                success=ok,
                reason=reason,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 主入口：run
    # ------------------------------------------------------------------

    def run(
        self,
        session_id: str,
        task: str,
        system_prompt: str = "",
        budget: Optional[Any] = None,
    ) -> Session:
        """运行一次 Agent 任务（顶层入口）。

        Args:
            session_id:    会话标识（同一 ID 重复运行会复用已保存的会话状态）。
            task:          用户任务文本。
            system_prompt: system prompt 模板（可含 {{memory_context}} 占位符）。
            budget:        可选预算。支持 Budget 实例、字典或整数（token 上限）。

        Returns:
            运行结束后的 Session（已保存到 SessionStore）。
        """
        session = self._prepare_session(session_id, task, budget)
        ctx = self._build_context(session, system_prompt)
        messages = self._build_messages(task, system_prompt)
        return self._run_with_ctx(ctx, messages, session)

    def _run_with_ctx(
        self,
        ctx: Context,
        messages: List[Dict[str, Any]],
        session: Session,
    ) -> Session:
        """在**已构建好的 ctx** 上执行一次 Agent Loop。

        [D002] 由 ``run()``（顶层，ctx 来自 ``_build_context``）与
        ``_run_delegation``（子 Agent，ctx 来自 ``_build_child_context``）共用。
        子 Agent 走本方法而非 ``run()``，才能保留 sandbox / shared_state /
        agent_registry 等继承字段。

        注意：本方法仍通过 ``self._loop.run(ctx, messages)`` 执行，
        以保持测试对 ``AgentLoop.run`` 的打桩有效。
        """
        # 共享 EventLog 后端按会话分组（SQLite 后端支持；内存后端忽略）
        setter = getattr(self.registry.event_log, "set_default_session", None)
        if callable(setter):
            setter(session.session_id)

        self._thread_local.active_ctx = ctx
        self._recorder.bind(session)
        self._recorder.record_loop_start(ctx, session)
        try:
            result_ctx = self._loop.run(ctx, messages)
        except BaseException as exc:  # pragma: no cover - Loop 内部已做全局兜底
            self._recorder.record_error(exc, ctx, session)
            self.registry.session_store.save(session)
            raise
        finally:
            self._thread_local.active_ctx = None

        self._recorder.record_loop_exit(result_ctx, session)
        self.registry.session_store.save(session)
        return session

    # ------------------------------------------------------------------
    # 内部装配辅助
    # ------------------------------------------------------------------

    def _prepare_session(
        self,
        session_id: str,
        task: str,
        budget: Optional[Any],
    ) -> Session:
        """创建或复用会话，并应用预算配置。"""
        session = self.registry.session_store.load(session_id)
        if session is None:
            session = Session(
                session_id=session_id,
                event_log=self.registry.event_log,
                metadata={"task": task},
            )

        # 会话持有与 registry 一致的 EventLog（恢复出的会话需重新绑定）
        session.event_log = self.registry.event_log
        session.metadata.setdefault("task", task)

        max_tokens, max_cost = self._budget_values(budget)
        if max_tokens:
            session.budget.max_tokens = max_tokens
        if max_cost:
            session.budget.max_cost = max_cost
        return session

    @staticmethod
    def _budget_values(budget: Optional[Any]) -> Any:
        """从 Budget 实例 / 字典 / 整数中提取 (max_tokens, max_cost)。"""
        if budget is None:
            return 0, 0.0
        if isinstance(budget, Budget):
            return budget.max_tokens, budget.max_cost
        if isinstance(budget, dict):
            return (
                int(budget.get("max_tokens", 0) or 0),
                float(budget.get("max_cost", 0.0) or 0.0),
            )
        if isinstance(budget, int):
            return int(budget), 0.0
        return 0, 0.0

    def _build_context(self, session: Session, system_prompt: str) -> Context:
        """构建 Context 并填充 L3 中间件约定的 ctx.shared 依赖键。"""
        ctx = Context()
        shared = ctx.shared

        # 约定的必备键（L3 中间件按此解析依赖）
        shared["event_log"] = self.registry.event_log
        shared["session"] = session
        shared["context_manager"] = self.registry.context_manager
        shared["tool_registry"] = self.registry.tool_registry

        if system_prompt:
            shared["system_prompt_template"] = system_prompt

        namespace = session.metadata.get("memory_namespace")
        if namespace:
            shared["memory_namespace"] = namespace

        whitelist = session.metadata.get("tool_whitelist")
        if whitelist is not None:
            shared["tool_whitelist"] = whitelist

        # [L4] 多 Agent 编排依赖（可选，None 时不写入）
        if self._agent_registry is not None:
            shared["agent_registry"] = self._agent_registry
        if self._shared_state is not None:
            shared["shared_state"] = self._shared_state

        # [L5] 技能自进化子系统（可选，None 时不写入 → skill 中间件静默放行）
        if self._skill_system is not None:
            shared["skill_system"] = self._skill_system

        return ctx

    @staticmethod
    def _build_messages(task: str, system_prompt: str) -> List[Dict[str, Any]]:
        """构建初始消息列表（可选 system + user）。"""
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": task})
        return messages

    # ------------------------------------------------------------------
    # 资源清理
    # ------------------------------------------------------------------

    def close(self) -> None:
        """关闭可关闭的后端（SQLite 事件日志 / 会话存储）。"""
        for backend in (self.registry.event_log, self.registry.session_store):
            closer = getattr(backend, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass

    def __repr__(self) -> str:
        return (
            f"<Runtime max_iterations={self.max_iterations} "
            f"middlewares={len(self._middlewares)} fibers={len(self._fibers)}>"
        )