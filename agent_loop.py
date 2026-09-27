"""
纯Python Agent运行时 — Agent-Loop 核心与状态模型

设计原则：
1. Agent Loop内核不含业务逻辑，全部横切能力通过中间件生命周期钩子实现。
2. 移除全部中间件后，Agent Loop仍可完整执行。
3. 状态模型区分三类状态：Agent私有状态、编排共享状态、持久化状态。
4. 中间件的状态修改可撤销：通过effect逆操作机制实现。

===== 本次修改记录 =====
[FEATURE] 新增 Fiber 生命周期容器类，代表一次插件加载的生命周期实例。
[FEATURE] Middleware 新增 effect(ctx, execute) 方法，将清理函数注册到 Fiber 的 disposables。
[FEATURE] AgentLoop 新增 load_plugins(paths, fiber_id) 方法，动态加载插件并创建 Fiber。
[FEATURE] AgentLoop 新增 dispose_fiber(fiber_id) 方法，销毁 Fiber 并回滚状态。
[FEATURE] 新增 _load_middleware_from_file / _load_middleware_from_dir 辅助函数。
[FEATURE] AgentLoop 新增 _unregister_middleware_no_undo 内部方法。

===== 本次修复记录 =====
[FIX-1] 消除 run 与 run_safe 的职责分裂：将全局异常保护合并进 run()，
        run_safe 降级为 run() 的别名。正常退出不触发 undo 栈。
[FIX-2] 修正 _dispatch_hook 对中间件钩子异常的升级策略：中间件钩子异常
        不再自动升级为 ABORT_LOOP，改为记录到 ctx._runtime["_hook_errors"]
        并打印 trace，然后继续执行后续中间件的同一钩子。
[FIX-3] 统一 ABORT_ITERATION 在各钩子中的语义：after_iteration 中
        ABORT_ITERATION 明确等同于 CONTINUE（本轮已结束）；before_tool /
        after_tool 中触发 ABORT_ITERATION 时在 ctx._runtime 中设置
        "_aborted_iteration" 标记供 after_iteration 中间件读取。
[FIX-4] tool_call 解析结果强制清理：工具循环体用 try/finally 包裹，
        ctx.clear_tool_calls() 在 finally 中执行，保证任何路径下都清理。
[FIX-5] run() 返回前检查 exit_errors：若非空则打印警告到 stderr。
[FIX-6] 统一 tool_calls 约定键名为 __tool_calls__（双下划线前后各两个）。
[FIX-7] _dispatch_hook docstring 明确 deepcopy 策略（性能换隔离）。
[FIX‑8] BUGFIX：before_iteration ABORT_ITERATION 未置位 aborted_iteration
[FIX‑9] BUGFIX：LLM异常路径未置位 aborted_iteration
[FIX‑10] BUGFIX：on_tool_error 返回HookResult被丢弃，控制流失效
[FEATURE] [C1] 钩子遍历短路：注册时预计算"每个钩子由哪些中间件实现"，
        调度时直接遍历该表，跳过基类默认实现的 getattr + 调用开销。
[FEATURE] [C1] 分层拷贝：Middleware 新增 mutates_args 属性（默认 False）。
        只读中间件仅做浅拷贝（保证顶层身份独立），改写型中间件做 deepcopy。
[FEATURE] [C1] 链式钩子引用复用：D004 链式传递中，未改写链式键的只读中间件
        复用同一份入参副本，避免逐个重建。
"""

from __future__ import annotations

import copy
import enum
import importlib
import importlib.util
import inspect
import os
import sys
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# 一、枚举定义
# ============================================================================


class HookType(enum.Enum):
    """生命周期钩子类型枚举"""

    ON_ENTER_LOOP = "ON_ENTER_LOOP"           # 鉴权、上下文加载、检查点恢复
    BEFORE_ITERATION = "BEFORE_ITERATION"      # 单轮迭代前置
    BEFORE_LLM = "BEFORE_LLM"                 # prompt注入、上下文压缩、rerank、敏感词过滤
    AFTER_LLM = "AFTER_LLM"                   # tool_call解析、格式校验、输出合规检查
    ON_LLM_ERROR = "ON_LLM_ERROR"             # LLM调用异常回调
    BEFORE_TOOL = "BEFORE_TOOL"               # 权限校验、白名单、审批拦截
    AFTER_TOOL = "AFTER_TOOL"                 # 结果清洗、错误捕获、单次审计
    ON_TOOL_ERROR = "ON_TOOL_ERROR"           # 工具执行异常回调
    AFTER_ITERATION = "AFTER_ITERATION"       # 单轮迭代后置
    ON_EXIT_LOOP = "ON_EXIT_LOOP"             # 状态持久化、trace上报、快照
    ON_LOOP_ERROR = "ON_LOOP_ERROR"           # Loop全局异常兜底


class ControlFlow(enum.Enum):
    """
    流程控制枚举，用于中间件做流程短路。

    语义说明：
    - CONTINUE:          继续执行当前流水线，不做任何短路。
    - SKIP_CURRENT_TOOL: 仅跳过本次工具调用，同一轮内后续工具继续执行。
                          仅在 BEFORE_TOOL 钩子中有意义，其余钩子返回此值等同于 CONTINUE。
    - ABORT_ITERATION:   终止本轮迭代，结束当前迭代，不再执行本轮剩余工具，
                          但 Loop 继续下一轮。
    - ABORT_LOOP:        终止整个 Agent Loop，进入正常退出流程（即触发 ON_EXIT_LOOP）。
    """

    CONTINUE = "CONTINUE"
    SKIP_CURRENT_TOOL = "SKIP_CURRENT_TOOL"
    ABORT_ITERATION = "ABORT_ITERATION"
    ABORT_LOOP = "ABORT_LOOP"


# ---------------------------------------------------------------------------
# 便于类型标注的 Callable 别名
# ---------------------------------------------------------------------------
EffectUndoFn = Callable[[], None]

# ============================================================================
# 二、HookResult 定义
# ============================================================================


@dataclass
class HookResult:
    """
    中间件钩子统一返回值。

    Attributes:
        control:  流程控制枚举，指示 Loop 下一步行为。
        payload:  可选字典，用于传递钩子处理后的数据。
                  约定 key：
                    - BEFORE_LLM  使用 key="messages" 返回更新后消息。
                    - BEFORE_TOOL 使用 key="tool_call" 返回更新后的工具调用。
                  payload 中的值必须是新对象，不得是入参对象的引用。
    """

    control: ControlFlow = ControlFlow.CONTINUE
    payload: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def continue_() -> "HookResult":
        """便捷构造：继续执行"""
        return HookResult(control=ControlFlow.CONTINUE)

    @staticmethod
    def skip_tool() -> "HookResult":
        """便捷构造：跳过当前工具"""
        return HookResult(control=ControlFlow.SKIP_CURRENT_TOOL)

    @staticmethod
    def abort_iteration() -> "HookResult":
        """便捷构造：终止本轮迭代"""
        return HookResult(control=ControlFlow.ABORT_ITERATION)

    @staticmethod
    def abort_loop(payload: Optional[Dict[str, Any]] = None) -> "HookResult":
        """便捷构造：终止整个 Loop"""
        return HookResult(control=ControlFlow.ABORT_LOOP, payload=payload or {})


# ============================================================================
# 三、Context 状态模型
# ============================================================================


@dataclass
class Context:
    """
    Agent 运行时上下文。

    状态域划分：
    ┌──────────────────────┬──────────────────────────────────────────┐
    │ 域                    │ 读写语义                                  │
    ├──────────────────────┼──────────────────────────────────────────┤
    │ private (私有状态)     │ Agent 内部使用，中间件通过钩子读写。        │
    │                       │ tool_call 解析结果存在此域约定键中。       │
    ├──────────────────────┼──────────────────────────────────────────┤
    │ shared (编排共享状态)  │ 跨 Agent / 编排器共享。                   │
    │                       │ effect 逆操作应谨慎使用，避免跨 Agent       │
    │                       │ 状态意外回滚。                            │
    ├──────────────────────┼──────────────────────────────────────────┤
    │ persistent (持久化)   │ 需要跨次运行持久化的状态。                  │
    │                       │ 中间件可在 ON_EXIT_LOOP 中将其落盘。       │
    ├──────────────────────┼──────────────────────────────────────────┤
    │ _runtime (运行时标记)  │ Loop 内部使用，外部不应直接写入。          │
    │                       │ 调用方可通过其查询运行状态。                │
    └──────────────────────┴──────────────────────────────────────────┘

    约定键（私有域）：
    - "__tool_calls__":  tool_call 解析结果，格式为 List[dict]（属于 private 域，
                         通过 get_tool_calls/set_tool_calls/clear_tool_calls 读写）。
                         由 AFTER_LLM 钩子中的中间件负责写入，
                         Loop 仅消费此键驱动工具调用流程。

    运行时标记字段说明（_runtime 域）：
    - "_hook_errors" (hook_errors property):  记录本次运行中所有中间件钩子抛出的
      非致命异常列表。跨 run() 调用累积，不自动清理——由调用方负责在合适的时机清空。
      每项为包含 middleware_name / hook_name / exception 的字典。
    - "_exit_errors" (exit_errors property):  记录本次运行中 ON_EXIT_LOOP 钩子
      抛出的异常列表。跨 run() 调用累积，不自动清理——由调用方负责在合适的时机清空。
      Loop 在 run() 返回前检查此字段，若非空则打印 stderr 警告。
    - "_aborted_iteration" (aborted_iteration property):  标记本轮迭代是否被
      ABORT_ITERATION 终止。该标记是每轮独立的，每轮迭代开始时由 Loop 重置为 False，
      本轮内若触发 ABORT_ITERATION 则置为 True。不是跨轮累积的。
    """

    private: Dict[str, Any] = field(default_factory=dict)
    shared: Dict[str, Any] = field(default_factory=dict)
    persistent: Dict[str, Any] = field(default_factory=dict)

    # ---- 运行时标记字段（Loop 内部使用） ----
    _runtime: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    # 便捷键名常量（FIX-6: 统一为双下划线命名 __tool_calls__）
    KEY_TOOL_CALLS: str = "__tool_calls__"

    # ---- 辅助方法 ----

    def get_tool_calls(self) -> List[Dict[str, Any]]:
        """从私有域约定键读取 tool_call 解析结果"""
        return self.private.get(self.KEY_TOOL_CALLS, [])

    def set_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> None:
        """写入 tool_call 解析结果到私有域约定键"""
        self.private[self.KEY_TOOL_CALLS] = tool_calls

    def clear_tool_calls(self) -> None:
        """清理 tool_call 解析结果"""
        self.private.pop(self.KEY_TOOL_CALLS, None)

    @property
    def aborted(self) -> bool:
        """查询 Loop 是否已被标记为终止"""
        return self._runtime.get("_aborted", False)

    @aborted.setter
    def aborted(self, value: bool) -> None:
        self._runtime["_aborted"] = value

    @property
    def exit_errors(self) -> List[Exception]:
        """ON_EXIT_LOOP 钩子中发生的异常列表，供调用方查询"""
        return self._runtime.setdefault("_exit_errors", [])

    @property
    def hook_errors(self) -> List[Dict[str, Any]]:
        """
        中间件钩子执行过程中发生的非致命异常列表，供调用方查询。

        [FIX-2] 每项为包含 middleware_name / hook_name / exception 的字典。
        """
        return self._runtime.setdefault("_hook_errors", [])

    @property
    def aborted_iteration(self) -> bool:
        """
        查询本轮迭代是否被 ABORT_ITERATION 终止。

        该标记是每轮独立的，不是跨轮累积的：每轮迭代开始时由 Loop 重置为 False，
        本轮内若触发 ABORT_ITERATION 则置为 True，供 after_iteration 中间件读取。

        [FIX-3] 在 before_tool/after_tool 中触发 ABORT_ITERATION 时置位。
        [FIX-8] 明确标记为每轮独立，并在每轮迭代开始时由 _run_impl 重置。
        """
        return self._runtime.get("_aborted_iteration", False)

    @aborted_iteration.setter
    def aborted_iteration(self, value: bool) -> None:
        self._runtime["_aborted_iteration"] = value


# ============================================================================
# 四、Fiber 生命周期容器
# ============================================================================
#
# [FEATURE] Fiber 代表一次插件加载的生命周期实例。
# 当通过 load_plugins 加载一组中间件时，会创建一个 Fiber 实例，
# 该 Fiber 持有这些中间件产生的所有 disposables（清理函数）。
# dispose_fiber 时逆序执行 disposables，实现状态回滚。


class Fiber:
    """
    一次插件加载的生命周期实例。

    Attributes:
        fiber_id:         Fiber 唯一标识。
        middleware_names: 该 Fiber 引入的中间件名称列表。
        _disposables:     处置栈（LIFO），存放清理函数。
        state:            生命周期状态：ACTIVE 或 DISPOSED。
    """

    def __init__(self, fiber_id: str, middleware_names: List[str]) -> None:
        self.fiber_id: str = fiber_id
        self.middleware_names: List[str] = list(middleware_names)
        self._disposables: List[Callable[[], None]] = []
        self.state: str = "ACTIVE"  # "ACTIVE" | "DISPOSED"

    def dispose(self) -> None:
        """
        逆序执行 _disposables，标记为 DISPOSED。

        幂等：多次调用不重复执行，不抛异常。
        """
        if self.state == "DISPOSED":
            return
        # 逆序执行（LIFO）
        for undo_fn in reversed(self._disposables):
            try:
                undo_fn()
            except Exception:
                traceback.print_exc()
        self._disposables.clear()
        self.state = "DISPOSED"

    def __repr__(self) -> str:
        return (
            f"<Fiber id={self.fiber_id!r} state={self.state!r} "
            f"middlewares={self.middleware_names!r}>"
        )


# ============================================================================
# 五、Middleware 协议基类
# ============================================================================


class Middleware:
    """
    中间件协议基类。

    子类仅需实现关心的钩子，其余钩子留空（默认返回 HookResult.continue_()）。

    Attributes:
        name: 中间件名称，用于日志和注销标识。
        priority: 调度优先级。数值越大，在同一钩子中的执行顺序越靠前。
                  用于解决跨中间件的顺序耦合（如 Observability 必须先于
                  CostGuard 写入事件）。同 priority 的中间件按注册顺序执行
                  （稳定排序）。默认 0。
        mutates_args: 是否可能原地改写入参（默认 False，即只读）。
                  [C1] 该属性决定 _dispatch_hook 的拷贝策略：
                    - False（只读）：仅做浅拷贝（copy.copy），保证每个中间件
                      拿到顶层身份独立的入参，但不递归复制嵌套结构；
                    - True（改写型）：做 deepcopy，提供完整隔离。
                  子类可声明为类属性，也可通过构造参数覆盖。
        _undo_stack: 该中间件注册的全部逆操作栈（后进先出）。
        _fiber:     当前关联的 Fiber 实例（由 AgentLoop.load_plugins 设置）。
    """

    #: [C1] 拷贝策略开关。默认 False —— 只读中间件不写钩子入参。
    mutates_args: bool = False

    def __init__(
        self,
        name: str = "",
        priority: int = 0,
        mutates_args: Optional[bool] = None,
    ) -> None:
        self.name: str = name or self.__class__.__name__
        self.priority: int = priority
        # [C1] 仅在显式传入时覆盖类属性，保证子类只需声明 `mutates_args = True`。
        if mutates_args is not None:
            self.mutates_args = bool(mutates_args)
        self._undo_stack: List[EffectUndoFn] = []
        self._fiber: Optional[Fiber] = None

    # ------------------------------------------------------------------
    # effect：将清理函数注册到当前关联的 Fiber 的 disposables 中
    # ------------------------------------------------------------------
    #
    # [FEATURE] effect 方法提供了一个与 Fiber 生命周期绑定的逆操作注册机制。
    # 与 register_effect（绑定到 Middleware 自身 undo_stack）不同，
    # effect 将清理函数绑定到当前 Fiber 实例的 _disposables 列表中。
    # 当 Fiber 被 dispose 时，这些清理函数会被逆序执行。

    def effect(
        self,
        ctx: Context,
        execute: Callable[["Context"], Optional[Callable[[], None]]],
    ) -> None:
        """
        执行 execute(ctx) 并将返回的清理函数注册到当前 Fiber 的 disposables。

        execute(ctx) 立即执行，返回一个可选的清理函数（undo function）。
        清理函数被追加到当前 Fiber 的 _disposables 列表末尾。
        当 Fiber 被 dispose 时，这些清理函数会按 LIFO 顺序逆序执行。

        Args:
            ctx:     运行时上下文。
            execute: 接收 ctx 的函数，返回一个清理函数（或 None）。

        Raises:
            RuntimeError: 如果当前中间件未关联 Fiber，或 Fiber 已 DISPOSED。
        """
        if self._fiber is None:
            raise RuntimeError(
                f"Middleware {self.name!r} 未关联任何 Fiber，"
                f"请通过 AgentLoop.load_plugins 加载中间件以绑定 Fiber。"
            )
        if self._fiber.state == "DISPOSED":
            raise RuntimeError(
                f"Fiber {self._fiber.fiber_id!r} 已 DISPOSED，"
                f"无法再注册 effect。"
            )

        cleanup = execute(ctx)
        if cleanup is not None:
            self._fiber._disposables.append(cleanup)

    # ------------------------------------------------------------------
    # register_effect：注册状态修改的逆操作
    # ------------------------------------------------------------------

    def register_effect(self, undo_fn: EffectUndoFn) -> None:
        """
        注册一个逆操作函数。

        中间件在钩子中修改状态后，调用此接口注册逆操作。
        effect 逆操作绑定到当前中间件实例。
        当中间件注销或 Loop 异常终止时，按注册逆序执行全部 undo_fn。
        单次钩子正常完成不会自动清除 effect。
        """
        self._undo_stack.append(undo_fn)

    # ------------------------------------------------------------------
    # 执行当前中间件的全部逆操作（逆序）
    # ------------------------------------------------------------------

    def _execute_undo_stack(self) -> None:
        """按注册逆序执行全部逆操作，执行后清空栈。"""
        # 逆序执行
        for undo_fn in reversed(self._undo_stack):
            try:
                undo_fn()
            except Exception:
                # undo 执行异常不阻断其他 undo，记录到 trace 中
                traceback.print_exc()
        self._undo_stack.clear()

    # ------------------------------------------------------------------
    # 生命周期钩子（默认实现：直接继续）
    # ------------------------------------------------------------------

    def on_enter_loop(self, ctx: Context) -> HookResult:
        return HookResult.continue_()

    def before_iteration(self, ctx: Context) -> HookResult:
        return HookResult.continue_()

    def before_llm(self, ctx: Context, messages: List[Dict[str, Any]]) -> HookResult:
        return HookResult.continue_()

    def after_llm(self, ctx: Context, response: Any) -> HookResult:
        return HookResult.continue_()

    def on_llm_error(self, ctx: Context, exception: Exception) -> HookResult:
        return HookResult.continue_()

    def before_tool(self, ctx: Context, tool_call: Dict[str, Any]) -> HookResult:
        return HookResult.continue_()

    def after_tool(
        self, ctx: Context, tool_call: Dict[str, Any], result: Any
    ) -> HookResult:
        return HookResult.continue_()

    def on_tool_error(
        self, ctx: Context, tool_call: Dict[str, Any], exception: Exception
    ) -> HookResult:
        return HookResult.continue_()

    def after_iteration(self, ctx: Context) -> HookResult:
        return HookResult.continue_()

    def on_exit_loop(self, ctx: Context) -> HookResult:
        return HookResult.continue_()

    def on_loop_error(self, ctx: Context, exception: Exception) -> HookResult:
        return HookResult.continue_()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"


# ---------------------------------------------------------------------------
# [C1] 钩子调度常量
# ---------------------------------------------------------------------------
#
# 钩子方法名全集。用于在注册时预计算"每个钩子由哪些中间件实现"，
# 使 _dispatch_hook 无需再对每个中间件做 getattr + 基类默认实现调用。

_HOOK_METHOD_NAMES: Tuple[str, ...] = (
    "on_enter_loop",
    "before_iteration",
    "before_llm",
    "after_llm",
    "on_llm_error",
    "before_tool",
    "after_tool",
    "on_tool_error",
    "after_iteration",
    "on_exit_loop",
    "on_loop_error",
)

#: [D004] 启用链式传递的钩子 → 其链式 payload 键。
#: 仅这两个钩子在 _dispatch_hook 中做"前一跳输出作为后一跳输入"的串联。
_CHAIN_KEYS: Dict[str, str] = {
    "before_llm": "messages",
    "before_tool": "tool_call",
}


# ============================================================================
# 六、AgentLoop 实现
# ============================================================================


class AgentLoop:
    """
    Agent Loop 运行时内核。

    不含业务逻辑：安全判断、诊断归因、上下文压缩、委派决策全部作为中间件实现。
    移除全部中间件后，Agent Loop 仍可完整执行。

    核心流程：
    1. ON_ENTER_LOOP  → 所有中间件钩子
    2. 迭代循环：
       a. BEFORE_ITERATION → 所有中间件钩子
       b. 构建消息，执行 BEFORE_LLM → 所有中间件钩子
       c. 调用 LLM（通过 _llm_call 注入）
       d. AFTER_LLM → 所有中间件钩子（tool_call 解析在此完成）
       e. 读取 ctx 私有域的 tool_calls，逐个执行：
          - BEFORE_TOOL → 中间件钩子
          - 执行工具（通过 _tool_executor 注入）
          - AFTER_TOOL → 中间件钩子
       f. AFTER_ITERATION → 所有中间件钩子
    3. ON_EXIT_LOOP → 所有中间件钩子（正常退出）
    4. 异常路径：
       - LLM 异常 → ON_LLM_ERROR，终止本轮，继续下一轮
       - 工具异常 → ON_TOOL_ERROR，跳过当前工具，继续同轮其他工具
       - 未捕获异常 → ON_LOOP_ERROR，终止 Loop，执行全局 undo，ON_EXIT_LOOP

    注意：
        - 中间件返回 ABORT_LOOP（控制流正常终止）**不会执行undo栈**；
          只有Python抛出未捕获异常才执行全部中间件undo回滚。
        - hook_errors / exit_errors 不会自动清空，调用方需要手动清理。
    """

    # LLM 调用签名：接收消息列表，返回 LLM 原始响应
    LLMCallFn = Callable[[List[Dict[str, Any]]], Any]

    # 工具执行签名：接收工具名称和参数字典，返回执行结果
    ToolExecutorFn = Callable[[str, Dict[str, Any]], Any]

    def __init__(
        self,
        llm_call: LLMCallFn,
        tool_executor: ToolExecutorFn,
        max_iterations: int = 10,
    ) -> None:
        """
        Args:
            llm_call:       LLM 调用函数，Loop 内核不关心 LLM 具体实现。
            tool_executor:  工具执行函数，接收 (tool_name, tool_args)，返回结果。
            max_iterations: 最大迭代轮数，防止无限循环。
        """
        self._llm_call = llm_call
        self._tool_executor = tool_executor
        self._max_iterations = max_iterations

        # 中间件有序注册表（保持注册顺序）
        self._middlewares: List[Middleware] = []
        # 名称→中间件映射，用于按名称注销
        self._middleware_map: Dict[str, Middleware] = {}

        # [FEATURE] Fiber 注册表：fiber_id → Fiber 实例
        self._fibers: Dict[str, Fiber] = {}

        # [C1] 钩子调度索引缓存：hook_name → [(middleware, bound_hook_fn), ...]。
        # 仅在中间件集合变化时重建（见 _invalidate_hook_index），
        # 避免每次 _dispatch_hook 重复做 sorted + getattr。
        self._hook_index_cache: Optional[
            Dict[str, List[Tuple[Middleware, Callable[..., "HookResult"]]]]
        ] = None

    # ------------------------------------------------------------------
    # [C1] 钩子调度索引
    # ------------------------------------------------------------------

    def _invalidate_hook_index(self) -> None:
        """[C1] 使钩子调度索引缓存失效（中间件集合变化时调用）。"""
        self._hook_index_cache = None

    def _hook_index(
        self,
    ) -> Dict[str, List[Tuple[Middleware, Callable[..., "HookResult"]]]]:
        """
        [C1] 构建（或返回缓存的）钩子调度索引。

        索引结构：hook_name → [(middleware, bound_hook_fn), ...]
            - 仅包含**真正实现了**该钩子的中间件，跳过基类默认实现；
            - 每个钩子内已按 priority 降序稳定排序（同 priority 保持注册顺序），
              与 D006 的执行顺序契约一致。

        为什么只含实现者：
            之前每次调度都要对全部中间件做 getattr 并调用基类默认实现
            （返回 HookResult.continue_()），既浪费调用开销，也让这些空实现
            进入链式传递的入参循环。预计算后 _dispatch_hook 直接遍历实现者。

        缓存失效：register_middleware / unregister_middleware /
            clear_middlewares / _unregister_middleware_no_undo 均会置空缓存。
        """
        cached = self._hook_index_cache
        if cached is not None:
            return cached

        # [D006] 按 priority 降序稳定排序（数值越大越先执行）
        ordered: List[Middleware] = sorted(
            self._middlewares,
            key=lambda mw: mw.priority,
            reverse=True,
        )

        index: Dict[str, List[Tuple[Middleware, Callable[..., "HookResult"]]]] = {}
        for hook_name in _HOOK_METHOD_NAMES:
            entries: List[Tuple[Middleware, Callable[..., "HookResult"]]] = []
            for mw in ordered:
                hook_fn = _resolve_hook_impl(mw, hook_name)
                if hook_fn is None:
                    continue
                entries.append((mw, hook_fn))
            index[hook_name] = entries

        self._hook_index_cache = index
        return index

    # ------------------------------------------------------------------
    # 中间件注册 / 注销 / 清空
    # ------------------------------------------------------------------

    def register_middleware(self, middleware: Middleware) -> None:
        """注册中间件。已存在同名中间件时先注销旧中间件。"""
        if middleware.name in self._middleware_map:
            self.unregister_middleware(middleware.name)
        self._middlewares.append(middleware)
        self._middleware_map[middleware.name] = middleware
        # [C1] 中间件集合变化 → 钩子索引失效
        self._invalidate_hook_index()

    def unregister_middleware(self, name: str) -> None:
        """
        注销指定名称的中间件。

        注销不只是从列表移除，同时执行该中间件注册的全部逆操作。
        运行时动态调用：已经进入执行流水线的钩子会完成当前流水线，
        下一轮迭代不再执行该中间件钩子。
        """
        mw = self._middleware_map.pop(name, None)
        if mw is None:
            return
        # 从有序列表中移除
        self._middlewares = [m for m in self._middlewares if m.name != name]
        # [C1] 中间件集合变化 → 钩子索引失效
        self._invalidate_hook_index()
        # 执行该中间件的全部逆操作
        mw._execute_undo_stack()

    def clear_middlewares(self) -> None:
        """清空全部中间件，并对每个中间件执行全部逆操作。"""
        for mw in reversed(self._middlewares):
            mw._execute_undo_stack()
        self._middlewares.clear()
        self._middleware_map.clear()
        # [C1] 中间件集合变化 → 钩子索引失效
        self._invalidate_hook_index()

    # ------------------------------------------------------------------
    # [FEATURE] load_plugins：动态加载插件并创建 Fiber
    # ------------------------------------------------------------------

    def load_plugins(
        self,
        paths: List[str],
        fiber_id: Optional[str] = None,
    ) -> Fiber:
        """
        扫描 paths 中的 Python 文件，动态 import，实例化 Middleware 子类，
        创建 Fiber 实例并关联中间件，注册到 AgentLoop，返回 Fiber。

        扫描逻辑：
        - 若 path 是 .py 文件，直接加载该模块。
        - 若 path 是目录，递归扫描目录下所有 .py 文件并加载。
        - 从每个模块中查找 Middleware 的子类（排除 Middleware 自身），
          实例化后通过 register_middleware 注册。

        Args:
            paths:    插件文件或目录路径列表。
            fiber_id: 可选，Fiber 标识。未提供时自动生成 UUID。

        Returns:
            创建的 Fiber 实例。

        Raises:
            FileNotFoundError: 若某个 path 不存在。
        """
        if fiber_id is None:
            fiber_id = str(uuid.uuid4())

        mw_instances: List[Middleware] = []

        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"插件路径不存在: {path!r}")

            if os.path.isfile(path) and path.endswith(".py"):
                _load_middleware_from_file(path, mw_instances)
            elif os.path.isdir(path):
                _load_middleware_from_dir(path, mw_instances)

        # 创建 Fiber
        mw_names = [mw.name for mw in mw_instances]
        fiber = Fiber(fiber_id=fiber_id, middleware_names=mw_names)

        # 将中间件的 _fiber 绑定到当前 Fiber
        for mw in mw_instances:
            mw._fiber = fiber
            self.register_middleware(mw)

        # 存入 _fibers
        self._fibers[fiber_id] = fiber

        return fiber

    # ------------------------------------------------------------------
    # [FEATURE] dispose_fiber：销毁 Fiber，回滚状态
    # ------------------------------------------------------------------

    def dispose_fiber(self, fiber_id: str) -> None:
        """
        销毁指定 Fiber：
        1. 逆序执行该 Fiber 的 _disposables（通过 Fiber.dispose() 实现）。
        2. 从 _fibers 注册表移除。
        3. 注销该 Fiber 引入的所有中间件，不重复执行它们的 undo 栈。

        幂等：多次调用不报错，不重复执行。

        Args:
            fiber_id: 要销毁的 Fiber 标识。
        """
        fiber = self._fibers.pop(fiber_id, None)
        if fiber is None:
            return

        # 1. 逆序执行 Fiber 的 disposables
        fiber.dispose()

        # 2. 注销该 Fiber 引入的所有中间件，但不执行其 undo 栈
        #    （因为 disposal 已通过 effect 的清理函数处理了回滚）
        for mw_name in fiber.middleware_names:
            self._unregister_middleware_no_undo(mw_name)

    # ------------------------------------------------------------------
    # 内部方法：注销中间件但不执行其 undo 栈
    # ------------------------------------------------------------------

    def _unregister_middleware_no_undo(self, name: str) -> None:
        """
        从注册表中移除指定名称的中间件，不执行其 undo 栈。

        用于 dispose_fiber 场景：Fiber 的 disposables 已经处理了状态回滚，
        不需要中间件自身的 undo_stack 再重复执行。
        """
        mw = self._middleware_map.pop(name, None)
        if mw is None:
            return
        self._middlewares = [m for m in self._middlewares if m.name != name]
        # [C1] 中间件集合变化 → 钩子索引失效
        self._invalidate_hook_index()

    # ------------------------------------------------------------------
    # 钩子调度 —— 遍历所有中间件，聚合 HookResult
    # ------------------------------------------------------------------

    def _dispatch_hook(
        self,
        hook_name: str,
        ctx: Context,
        args: tuple = (),
    ) -> HookResult:
        """
        遍历所有中间件，依次执行指定钩子。

        聚合策略：
        - payload 合并：后执行的中间件覆盖先执行中间件的同 key 值。
        - control 合并：取最高优先级（ABORT_LOOP > ABORT_ITERATION >
          SKIP_CURRENT_TOOL > CONTINUE）。

        执行顺序（D006）：
        - 按中间件 priority 降序稳定排序（同 priority 保持注册顺序）。
          用于解决跨中间件的顺序耦合（如 Observability 必须先于 CostGuard
          写入 EventLog，CostGuard 才能读到事件并正确累加预算）。

        链式传递（D004）：
        - 仅对 BEFORE_LLM / BEFORE_TOOL 启用。链式键分别为
          "messages" / "tool_call"。前一个中间件返回的链式键值将作为
          后一个中间件的同名入参，使多个改写 messages / tool_call 的中间件
          按注册顺序串联生效，而非"后写覆盖先写、只生效一个"。
        - 其他钩子保持原语义：每个中间件都收到原始入参的独立副本。

        拷贝策略（[C1] 分层拷贝，取代 FIX-7 的全量 deepcopy）：
        - 只读中间件（mutates_args=False）：仅做浅拷贝（copy.copy）。入参在
          **顶层对象身份**上相互独立（保持 FIX-7 的隔离契约），但不递归复制
          嵌套结构，因此不必为每次调度付出深拷贝代价。
        - 改写型中间件（mutates_args=True）：做 deepcopy，提供完整隔离。
        - 链式钩子（BEFORE_LLM / BEFORE_TOOL）的引用复用：链式键未被改写的
          连续只读中间件复用同一份入参副本，避免逐跳重建。

        [FIX-2] 异常策略：
        - 中间件钩子内部抛出的异常不会被自动升级为 ABORT_LOOP。异常被记录到
          ctx.hook_errors 并打印 traceback，继续执行后续中间件的同一钩子。
        - 如果中间件确实需要终止 Loop，应通过返回 HookResult.abort_loop()
          表达，而不是抛异常。

        Args:
            hook_name: 钩子方法名（如 "before_llm"）。
            ctx:       运行时上下文。
            args:      钩子的额外位置参数（如 messages、tool_call 等）。

        Returns:
            聚合后的 HookResult。
        """
        # [C1] 钩子遍历短路：直接取"实现了该钩子"的中间件列表。
        # 该列表已在 _hook_index 中按 priority 降序稳定排序（D006），
        # 且已剔除仅使用基类默认实现的中间件。
        implemented = self._hook_index().get(hook_name)
        if not implemented:
            return HookResult.continue_()

        aggregated = HookResult.continue_()

        # [D004] 链式传递：仅 BEFORE_LLM（messages）/ BEFORE_TOOL（tool_call）启用。
        chain_key: Optional[str] = _CHAIN_KEYS.get(hook_name)

        # 链式输入：初始为调用方传入的原始 args。仅当某中间件返回链式键的新值时
        # 才被替换，并作为下一跳的入参来源。
        working_args: tuple = args
        # [C1] 复用槽位：链式键未被改写时，连续只读中间件共用同一份入参副本。
        # 仅对链式钩子启用 —— 非链式钩子需保持"每个中间件一份独立副本"的原语义。
        reusable_copy: Optional[tuple] = None

        for mw, hook_fn in implemented:
            # [C1] 分层拷贝：改写型 deepcopy，只读型浅拷贝。
            if mw.mutates_args:
                # 改写型中间件获得完整隔离，并切断复用链
                args_copy = tuple(copy.deepcopy(a) for a in working_args)
                reusable_copy = None
            elif reusable_copy is not None:
                # 链式钩子且链式键未被改写 → 复用上一跳的入参副本
                args_copy = reusable_copy
            else:
                args_copy = tuple(copy.copy(a) for a in working_args)
                if chain_key is not None:
                    reusable_copy = args_copy

            try:
                result: HookResult = hook_fn(ctx, *args_copy)
            except Exception as exc:
                # [FIX-2] 中间件钩子异常不自动升级为 ABORT_LOOP。
                # 记录到 ctx.hook_errors 并打印 traceback，继续执行后续中间件。
                traceback.print_exc()
                ctx.hook_errors.append({
                    "middleware_name": mw.name,
                    "hook_name": hook_name,
                    "exception": exc,
                })
                # 继续执行后续中间件，不修改 control 和 payload
                continue

            # 合并 control：优先级从高到低
            aggregated.control = _max_control(aggregated.control, result.control)

            # 合并 payload：后执行覆盖先执行
            aggregated.payload.update(result.payload)

            # [D004] 链式传递：该钩子启用链式且中间件返回了链式键的新值时，
            # 将其作为下一跳的对应入参。
            if (
                chain_key is not None
                and result.payload
                and chain_key in result.payload
            ):
                working_args = (result.payload[chain_key],)
                # 入参已变更 → 先前的复用副本失效，下一跳需自建副本
                reusable_copy = None

            # ABORT_LOOP 信号一旦出现就不再继续执行后续中间件
            if aggregated.control == ControlFlow.ABORT_LOOP:
                break

        return aggregated

    # ------------------------------------------------------------------
    # 主循环（FIX-1: 合并 run_safe 的全局异常保护）
    # ------------------------------------------------------------------

    def run(self, ctx: Context, initial_messages: List[Dict[str, Any]]) -> Context:
        """
        启动 Agent Loop。

        内置全局异常保护：发生未捕获异常时完整执行
        ON_LOOP_ERROR → 全局 undo 栈 → ON_EXIT_LOOP。
        正常退出路径仅执行 ON_EXIT_LOOP，不触发 undo 栈。

        Args:
            ctx:              运行时上下文（外部预先构建好的 Context 实例）。
            initial_messages: 初始消息列表。

        Returns:
            运行结束后的 Context 实例（调用方可从中读取状态）。
        """
        try:
            return self._run_impl(ctx, initial_messages)
        except Exception as exc:
            # [FIX-1] 全局异常 → ON_LOOP_ERROR → undo 栈 → ON_EXIT_LOOP
            traceback.print_exc()
            ctx.aborted = True  # 置位终止标记，保证 ON_EXIT_LOOP 读到正确状态

            try:
                self._dispatch_hook("on_loop_error", ctx, (exc,))
            except Exception:
                traceback.print_exc()

            # 按中间件注册逆序执行全部 undo 栈
            self._execute_all_undo_stacks()

            # 执行 ON_EXIT_LOOP 收尾
            self._run_on_exit_loop(ctx)

            # [FIX-5] 检查 exit_errors，若非空打印警告
            self._warn_exit_errors(ctx)
            return ctx

    # ------------------------------------------------------------------
    # run_safe（FIX-1: 降级为 run() 的别名）
    # ------------------------------------------------------------------

    def run_safe(self, ctx: Context, initial_messages: List[Dict[str, Any]]) -> Context:
        """
        [FIX-1] run() 的别名，保留以兼容旧调用方。

        行为与 run() 完全相同：内置全局异常保护。
        """
        return self.run(ctx, initial_messages)

    # ------------------------------------------------------------------
    # _run_impl：主循环核心实现
    # ------------------------------------------------------------------

    def _run_impl(self, ctx: Context, initial_messages: List[Dict[str, Any]]) -> Context:
        """
        Agent Loop 主循环核心实现。

        不含 try/except 全局守卫——异常由 run() 统一捕获并进入异常退出路径。
        """
        # 重置终止标记
        ctx.aborted = False

        # ---- Phase 1: ON_ENTER_LOOP ----
        result = self._dispatch_hook("on_enter_loop", ctx)
        if result.control == ControlFlow.ABORT_LOOP:
            ctx.aborted = True
            self._run_on_exit_loop(ctx)
            self._warn_exit_errors(ctx)
            return ctx

        # ---- Phase 2: 迭代循环 ----
        messages = initial_messages
        iteration = 0

        while iteration < self._max_iterations and not ctx.aborted:
            iteration += 1

            # [FIX-8] 每轮迭代开始时重置 aborted_iteration 标记。
            # 该标记是每轮独立的，上一轮的残留值不得泄漏到本轮。
            # 重置后，本轮内若 before_tool/after_tool 触发 ABORT_ITERATION，
            # 仍会置为 True，供 after_iteration 中间件读取。
            ctx.aborted_iteration = False

            # --- BEFORE_ITERATION ---
            result = self._dispatch_hook("before_iteration", ctx)
            if result.control == ControlFlow.ABORT_LOOP:
                ctx.aborted = True
                break
            if result.control == ControlFlow.ABORT_ITERATION:
                ctx.aborted_iteration = True  # FIX‑8
                continue

            # --- BEFORE_LLM ---
            result = self._dispatch_hook("before_llm", ctx, (messages,))
            if result.control == ControlFlow.ABORT_LOOP:
                ctx.aborted = True
                break
            if result.control == ControlFlow.ABORT_ITERATION:
                continue
            # 使用中间件返回的更新后消息（若存在）
            if "messages" in result.payload:
                messages = result.payload["messages"]

            # --- LLM Call ---
            response = None
            llm_error: Optional[Exception] = None
            try:
                response = self._llm_call(messages)
            except Exception as exc:
                llm_error = exc

            if llm_error is not None:
                # LLM 调用异常 → 触发 ON_LLM_ERROR，终止本轮，继续下一轮
                err_result = self._dispatch_hook("on_llm_error", ctx, (llm_error,))
                if err_result.control == ControlFlow.ABORT_LOOP:
                    ctx.aborted = True
                    break
                ctx.aborted_iteration = True  # FIX‑9
                continue

            # --- AFTER_LLM ---
            result = self._dispatch_hook("after_llm", ctx, (response,))
            if result.control == ControlFlow.ABORT_LOOP:
                ctx.aborted = True
                break
            if result.control == ControlFlow.ABORT_ITERATION:
                continue

            # --- 消费私有域中的 tool_calls ---
            tool_calls = ctx.get_tool_calls()
            # 注意：Loop 不得从 LLM 原始响应中直接读取 tool_call 结构作为 fallback。
            # tool_call 解析由中间件（在 AFTER_LLM 钩子中）完成并写入 ctx 私有域。

            # [D001] 无 tool_calls 即视为 LLM 已给出最终答复（finish_reason=stop /
            # 空响应），本轮迭代结束后自然终止 Loop，不再空转至 max_iterations。
            # 该判定只消费 ctx 私有域（__tool_calls__），不读取原始 response 的
            # tool_call 结构，符合上面 852-853 行的约束。
            no_tool_calls = not tool_calls

            # [FIX-4] 用 try/finally 保证 tool_calls 在任何路径下都被清理
            try:
                for tc in tool_calls:
                    # --- BEFORE_TOOL ---
                    # 传入副本，禁止中间件原地篡改
                    tool_result = self._dispatch_hook("before_tool", ctx, (tc,))
                    if tool_result.control == ControlFlow.ABORT_LOOP:
                        ctx.aborted = True
                        break
                    if tool_result.control == ControlFlow.ABORT_ITERATION:
                        # [FIX-3] 记录 ABORT_ITERATION 标记，供 after_iteration 读取
                        ctx.aborted_iteration = True
                        break  # 跳出工具循环，进入下一轮
                    if tool_result.control == ControlFlow.SKIP_CURRENT_TOOL:
                        continue  # 跳过当前工具，继续同轮后续工具
                    # 使用中间件返回的更新后 tool_call（若存在）
                    effective_tc = tool_result.payload.get("tool_call", tc)

                    # --- 执行工具 ---
                    tool_name = effective_tc.get("name", "")
                    tool_args = effective_tc.get("arguments", {})
                    tool_exec_result = None
                    tool_error: Optional[Exception] = None
                    try:
                        tool_exec_result = self._tool_executor(tool_name, tool_args)
                    except Exception as exc:
                        tool_error = exc

                    if tool_error is not None:
                        # [FIX‑10] 接收on_tool_error返回的控制流
                        err_res = self._dispatch_hook("on_tool_error", ctx, (effective_tc, tool_error))
                        if err_res.control == ControlFlow.ABORT_LOOP:
                            ctx.aborted = True
                            break
                        if err_res.control == ControlFlow.ABORT_ITERATION:
                            ctx.aborted_iteration = True
                            break
                        # SKIP / CONTINUE，继续下一个工具
                        continue

                    # --- AFTER_TOOL ---
                    after_result = self._dispatch_hook(
                        "after_tool", ctx, (effective_tc, tool_exec_result)
                    )
                    if after_result.control == ControlFlow.ABORT_LOOP:
                        ctx.aborted = True
                        break
                    if after_result.control == ControlFlow.ABORT_ITERATION:
                        # [FIX-3] 记录 ABORT_ITERATION 标记，供 after_iteration 读取
                        ctx.aborted_iteration = True
                        break  # 跳出工具循环
            finally:
                # [FIX-4] 无论工具循环内部发生什么（正常完成、跳过、ABORT_LOOP、
                # ABORT_ITERATION、异常），都必须清理 tool_calls 解析结果。
                ctx.clear_tool_calls()

            if ctx.aborted:
                break

            # --- AFTER_ITERATION ---
            # [FIX-3] ABORT_ITERATION 在 after_iteration 中明确等同于 CONTINUE：
            # 本轮迭代已经结束，无剩余流程可跳过，Loop 自然流转到下一轮。
            # 中间件可通过 ctx.aborted_iteration 读取本轮是否被 ABORT_ITERATION 终止。
            result = self._dispatch_hook("after_iteration", ctx)
            if result.control == ControlFlow.ABORT_LOOP:
                ctx.aborted = True
                break
            # ABORT_ITERATION / CONTINUE 均视为本轮正常结束，不影响循环流转

            # [D001] 本轮没有任何 tool_calls → LLM 已给出最终答复，Loop 自然结束。
            # 注意：这里**不**置 ctx.aborted=True —— 自然结束不是异常/主动中止，
            # 置位会污染 ON_EXIT_LOOP 与 runtime 侧 `abort_flag` 的语义。
            if no_tool_calls:
                break

        # ---- Phase 3: ON_EXIT_LOOP ----
        self._run_on_exit_loop(ctx)

        # [FIX-5] 返回前检查 exit_errors，若非空打印警告
        self._warn_exit_errors(ctx)
        return ctx

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _run_on_exit_loop(self, ctx: Context) -> None:
        """
        执行 ON_EXIT_LOOP 钩子。

        ON_EXIT_LOOP 钩子中的异常不得静默吞掉，
        必须记录到 ctx 的运行时标记字段中，供调用方查询。

        与 _dispatch_hook 的通用异常处理策略不同：ON_EXIT_LOOP 中每个中间件的异
        常需要同时记录到 hook_errors（保持与 FIX-2 的一致性）和 exit_errors
        （满足规范要求，供调用方通过 ctx.exit_errors 查询）。
        """
        # [C1] 用预计算的钩子索引跳过未实现 on_exit_loop 的中间件，
        # 免去逐件 getattr。遍历顺序仍取 _middlewares（注册顺序）——
        # 本方法不走 _dispatch_hook，因此不受 priority 排序影响。
        exit_impls = dict(self._hook_index().get("on_exit_loop") or ())
        for mw in self._middlewares:
            hook_fn = exit_impls.get(mw)
            if hook_fn is None:
                continue
            try:
                hook_fn(ctx)
            except Exception as exc:
                traceback.print_exc()
                ctx.hook_errors.append({
                    "middleware_name": mw.name,
                    "hook_name": "on_exit_loop",
                    "exception": exc,
                })
                ctx.exit_errors.append(exc)

    def _execute_all_undo_stacks(self) -> None:
        """
        按中间件注册逆序，执行所有中间件的全部 undo 栈。

        仅在 Loop 异常终止时触发；正常退出不触发。
        """
        for mw in reversed(self._middlewares):
            mw._execute_undo_stack()

    def _warn_exit_errors(self, ctx: Context) -> None:
        """
        [FIX-5] 检查 ctx.exit_errors，若非空则打印警告到 stderr。

        不改变 ctx.exit_errors 的存储位置和类型，
        调用方仍可通过该字段查询完整异常列表。
        """
        if ctx.exit_errors:
            print(
                f"[AgentLoop WARNING] ON_EXIT_LOOP 钩子中存在 {len(ctx.exit_errors)} 个异常:",
                file=sys.stderr,
            )
            for i, exc in enumerate(ctx.exit_errors, 1):
                print(f"  [{i}] {type(exc).__name__}: {exc}", file=sys.stderr)


# ============================================================================
# 七、辅助函数
# ============================================================================


# ControlFlow 优先级映射（值越大优先级越高）
_CONTROL_PRIORITY: Dict[ControlFlow, int] = {
    ControlFlow.CONTINUE: 0,
    ControlFlow.SKIP_CURRENT_TOOL: 1,
    ControlFlow.ABORT_ITERATION: 2,
    ControlFlow.ABORT_LOOP: 3,
}


def _max_control(a: ControlFlow, b: ControlFlow) -> ControlFlow:
    """取两个 ControlFlow 中优先级更高的那个"""
    return a if _CONTROL_PRIORITY[a] >= _CONTROL_PRIORITY[b] else b


def _resolve_hook_impl(
    mw: Middleware,
    hook_name: str,
) -> Optional[Callable[..., HookResult]]:
    """
    [C1] 判断中间件 ``mw`` 是否真正实现了钩子 ``hook_name``。

    若仅继承 Middleware 的默认实现（返回 HookResult.continue_()），返回 None，
    使其在钩子遍历中被短路跳过，省去 getattr + 空调用开销。

    动态实现兼容：若中间件类定义了 ``__getattr__``（无法静态判定钩子是否存在），
    退化为 getattr 动态探测，保持与旧行为一致。

    Args:
        mw:        中间件实例。
        hook_name: 钩子方法名。

    Returns:
        绑定后的钩子函数；未实现时返回 None。
    """
    default_impl = Middleware.__dict__.get(hook_name)

    for base in type(mw).__mro__:
        if base is object:
            break
        impl = base.__dict__.get(hook_name)
        if impl is None:
            continue
        if base is Middleware:
            # 落到基类默认实现 → 该中间件未实现此钩子
            break
        if isinstance(impl, staticmethod):
            impl = impl.__func__
        if impl is default_impl:
            # 显式把基类默认实现赋给子类，等同于未实现
            break
        return getattr(mw, hook_name)

    # 动态属性兜底：类定义了 __getattr__ 时静态查不到钩子名
    if getattr(type(mw), "__getattr__", None) is not None:
        try:
            fn = getattr(mw, hook_name)
        except AttributeError:
            return None
        if callable(fn) and getattr(fn, "__func__", None) is not default_impl:
            return fn

    return None


# ---------------------------------------------------------------------------
# [FEATURE] 插件加载辅助函数
# ---------------------------------------------------------------------------


def _load_middleware_from_file(
    filepath: str,
    out_instances: List[Middleware],
) -> None:
    """
    从单个 .py 文件动态 import，查找 Middleware 子类并实例化。

    对于找到的每个 Middleware 子类（排除 Middleware 自身），
    创建实例并追加到 out_instances 列表中。
    """
    # 生成一个唯一的模块名，避免多次加载同一文件名冲突
    mod_name = (
        f"_plugin_{os.path.basename(filepath).replace('.py', '')}"
        f"_{uuid.uuid4().hex[:8]}"
    )

    spec = importlib.util.spec_from_file_location(mod_name, filepath)
    if spec is None or spec.loader is None:
        return

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 遍历模块中所有成员，查找 Middleware 的具体子类
    for _name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, Middleware) and obj is not Middleware:
            instance = obj()
            out_instances.append(instance)


def _load_middleware_from_dir(
    dirpath: str,
    out_instances: List[Middleware],
) -> None:
    """
    递归扫描目录下所有 .py 文件（排除 __pycache__ 等），
    调用 _load_middleware_from_file 加载每个文件。
    """
    for root, dirs, files in os.walk(dirpath):
        # 跳过 __pycache__ 等隐藏目录
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for filename in files:
            if filename.endswith(".py"):
                filepath = os.path.join(root, filename)
                _load_middleware_from_file(filepath, out_instances)