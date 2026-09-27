"""
[L3-1] ObservabilityMiddleware — 可观测性中间件。

职责边界（只做观测，不做决策）：
    - 是 EventLog 中 LLM / 工具调用事件的唯一写入方。
    - 输出结构化 JSON 日志（每行一个 JSON 对象）。
    - 聚合性能指标：每步耗时、token 消耗、成本。
    - 恒返回 HookResult.continue_()，不介入流程控制。
    - 不修改消息、不拦截工具、不做预算判定——那是其他中间件的职责。

注册钩子：
    - AFTER_LLM      : 记录一次 LLM 调用事件 + 输出一步耗时与 token
    - AFTER_TOOL     : 记录一次工具调用事件 + 输出工具耗时
    - AFTER_ITERATION: 补写被 SafetyGuard 等中间件 skip 的工具事件
                       （ToolCallEvent(skipped=True)），补全审计链路
    - ON_EXIT_LOOP   : 输出全量聚合指标（token / cost / 每步耗时）

[D007] 被 skip 的工具为何要补写事件：
    SafetyGuardMiddleware 在 BEFORE_TOOL 返回 SKIP_CURRENT_TOOL，被拦截的工具
    不会进入工具循环体，也就永远不会触发 AFTER_TOOL。结果是"工具被拦截"这一
    事实在 EventLog 中完全不可见，审计链路与 tool_calls 统计双双缺失。
    由于 Observability 是本项目 LLM / 工具事件的唯一写入方，这里在
    AFTER_ITERATION 读取 ctx.shared["safety_blocked"]（SafetyGuard 写入的约定键），
    为尚未记录过的每条拦截补写一条 ToolCallEvent(skipped=True)。
    补写是幂等的：用实例级游标记录"已补写条数"，且游标绑定当前 ctx，
    每次 run（新 Context）自动从 0 开始，不跨 run 漂移。

依赖关系：
    ObservabilityMiddleware -> runtime.event_log（写入并读取事件）
    可选读取 Session（仅用于附带 session_id）。

耗时口径：
    本中间件只注册 AFTER_* 钩子（不注册 BEFORE_*），因此单步耗时以
    相邻两次钩子触发的真实时间差作为近似值（step_ms）。若被观测对象
    自身提供 elapsed 字段（如 ToolResult.metadata["elapsed_seconds"]），
    则优先采用该精确值。

[L3-1] 本文件为 L3 第一批中间件新增，未修改任何 L1 / L2 代码。
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from agent_loop import Context, HookResult, Middleware

from runtime.event_log import (
    EVENT_LOOP_EXIT,
    EVENT_TOOL_CALL,
    EventLog,
    resolve_event_log,
)


class ObservabilityMiddleware(Middleware):
    """[L3-1] 可观测性中间件。

    Attributes:
        records: 结构化日志记录列表（供程序化断言使用）。
        logs:    已输出的日志文本行。
    """

    def __init__(
        self,
        event_log: Optional[EventLog] = None,
        session: Any = None,
        stream: Any = None,
        echo: bool = True,
        json_logging: bool = True,
        name: str = "observability",
    ) -> None:
        """
        Args:
            event_log:     事件日志实例。为 None 时从 ctx.shared / session 解析。
            session:       会话实例（可选，仅用于附带 session_id）。
            stream:        自定义输出流（需支持 write）。None 且 echo=True 时打印到 stdout。
            echo:          是否打印到 stdout。
            json_logging:  True 输出 JSON 行，False 输出 dict 的 str 形式。
            name:          中间件名称。
        """
        super().__init__(name, priority=10)
        self._event_log = event_log
        self._session = session
        self._stream = stream
        self._echo = echo
        self._json_logging = json_logging

        # 无外部依赖时的兜底日志，保证中间件可独立启停
        self._fallback_log = EventLog()
        self._iteration = 0

        # [D007] 被 skip 工具事件的补写游标：记录"已补写的 safety_blocked 条数"。
        # 与 ctx 绑定——换了新 Context（新 run）即从 0 重新开始，保证幂等且不跨 run 漂移。
        self._skipped_cursor = 0
        self._skipped_ctx: Optional[Context] = None
        # 使用单调高精度时钟：time.time() 在 Windows 上分辨率约 1~15ms，
        # 短间隔的两步会取到同一时间戳，导致耗时指标恒为 0。
        self._last_tick = time.perf_counter()

        # 对外可观测的产物
        self.records: List[Dict[str, Any]] = []
        self.logs: List[str] = []

    # ------------------------------------------------------------------
    # AFTER_LLM —— 记录一次 LLM 调用
    # ------------------------------------------------------------------

    def after_llm(self, ctx: Context, response: Any) -> HookResult:
        """[L3-1] AFTER_LLM：写入 LLMCallEvent 并输出结构化日志。"""
        log = self._resolve_log(ctx)
        session = self._resolve_session(ctx)

        self._iteration += 1
        elapsed_ms = self._tick()

        info = self._extract_llm(response)
        log.emit_llm_call(
            iteration=self._iteration,
            model=info["model"],
            provider=info["provider"],
            prompt_tokens=info["prompt_tokens"],
            completion_tokens=info["completion_tokens"],
            total_tokens=info["total_tokens"],
            cost=info["cost"],
            elapsed_ms=elapsed_ms,
            success=info["success"],
            error=info["error"],
        )

        record: Dict[str, Any] = {
            "event": "llm_call",
            "middleware": self.name,
            "iteration": self._iteration,
            "session_id": getattr(session, "session_id", ""),
            "model": info["model"],
            "provider": info["provider"],
            "prompt_tokens": info["prompt_tokens"],
            "completion_tokens": info["completion_tokens"],
            "total_tokens": info["total_tokens"],
            "cost": round(info["cost"], 8),
            "step_ms": elapsed_ms,
            "success": info["success"],
        }
        self._emit(record)
        return HookResult.continue_()

    # ------------------------------------------------------------------
    # AFTER_TOOL —— 记录一次工具调用
    # ------------------------------------------------------------------

    def after_tool(
        self, ctx: Context, tool_call: Dict[str, Any], result: Any
    ) -> HookResult:
        """[L3-1] AFTER_TOOL：写入 ToolCallEvent 并输出结构化日志。"""
        log = self._resolve_log(ctx)

        elapsed_ms = self._extract_tool_elapsed(result)
        step_ms = self._tick()
        if elapsed_ms <= 0.0:
            elapsed_ms = step_ms

        tool_name = self._tool_name(tool_call)
        success, error = self._tool_status(result)

        log.emit_tool_call(
            iteration=self._iteration,
            tool_name=tool_name,
            arguments=self._tool_arguments(tool_call),
            success=success,
            elapsed_ms=elapsed_ms,
            error=error,
        )

        record: Dict[str, Any] = {
            "event": EVENT_TOOL_CALL,
            "middleware": self.name,
            "iteration": self._iteration,
            "tool": tool_name,
            "success": success,
            "elapsed_ms": elapsed_ms,
            "step_ms": step_ms,
        }
        if error:
            record["error"] = error
        self._emit(record)
        return HookResult.continue_()

    # ------------------------------------------------------------------
    # AFTER_ITERATION —— 补写被 skip 的工具事件（D007）
    # ------------------------------------------------------------------

    def after_iteration(self, ctx: Context) -> HookResult:
        """[D007] AFTER_ITERATION：为被 skip 的工具补写 ToolCallEvent(skipped=True)。

        为什么放在 AFTER_ITERATION：
            BEFORE_TOOL 的 SKIP_CURRENT_TOOL 使该工具不进入工具循环体，
            AFTER_TOOL 永不触发；而 AFTER_ITERATION 每轮必然执行，
            且此时本轮所有 before_tool 拦截都已写入 ctx.shared["safety_blocked"]。

        幂等性：
            用实例级游标 _skipped_cursor 记录已补写条数；游标一旦发现 ctx 变更
            （新 run 复用同一中间件实例）即复位为 0，避免跨 run 重复补写或漏写。
        """
        blocked = ctx.shared.get("safety_blocked")
        if not isinstance(blocked, list) or not blocked:
            return HookResult.continue_()

        # ctx 变更（新 run）→ 复位游标
        if self._skipped_ctx is not ctx:
            self._skipped_ctx = ctx
            self._skipped_cursor = 0

        # 只补写尚未记录过的部分（本轮新增的拦截）
        pending = blocked[self._skipped_cursor:]
        if not pending:
            return HookResult.continue_()

        log = self._resolve_log(ctx)
        for entry in pending:
            self._emit_skipped_tool(ctx, log, entry)
        self._skipped_cursor = len(blocked)

        return HookResult.continue_()

    def _emit_skipped_tool(self, ctx: Context, log: EventLog, entry: Any) -> None:
        """[D007] 为单条 safety_blocked 记录补写 ToolCallEvent(skipped=True)。"""
        if isinstance(entry, dict):
            tool_name = str(entry.get("tool", "") or "")
            reason = str(entry.get("reason", "") or "")
        else:
            tool_name = str(getattr(entry, "tool", "") or "")
            reason = str(getattr(entry, "reason", "") or "")

        log.emit_tool_call(
            iteration=self._iteration,
            tool_name=tool_name,
            arguments={},
            success=False,
            elapsed_ms=0.0,
            skipped=True,
            error=reason,
        )

        record: Dict[str, Any] = {
            "event": EVENT_TOOL_CALL,
            "middleware": self.name,
            "iteration": self._iteration,
            "tool": tool_name,
            "success": False,
            "skipped": True,
            "elapsed_ms": 0.0,
        }
        if reason:
            record["error"] = reason
        self._emit(record)

    # ------------------------------------------------------------------
    # ON_EXIT_LOOP —— 输出聚合指标
    # ------------------------------------------------------------------
    def on_exit_loop(self, ctx: Context) -> HookResult:
        """[L3-1] ON_EXIT_LOOP：输出全量聚合指标（token / cost / 耗时）。"""
        log = self._resolve_log(ctx)
        session = self._resolve_session(ctx)
        summary = log.summary()

        record: Dict[str, Any] = {
            "event": EVENT_LOOP_EXIT,
            "middleware": self.name,
            "session_id": getattr(session, "session_id", ""),
            "total_events": summary["events"],
            "llm_calls": summary["llm_calls"],
            "tool_calls": summary["tool_calls"],
            "total_tokens": summary["tokens"]["total_tokens"],
            "prompt_tokens": summary["tokens"]["prompt_tokens"],
            "completion_tokens": summary["tokens"]["completion_tokens"],
            "total_cost": summary["cost"],
            "llm_avg_ms": summary["llm_elapsed"]["avg_ms"],
            "tool_avg_ms": summary["tool_elapsed"]["avg_ms"],
        }
        budget = getattr(session, "budget", None)
        if budget is not None and hasattr(budget, "to_dict"):
            record["budget"] = budget.to_dict()
        self._emit(record)

        # 聚合视图同时挂到 ctx.shared，供编排方读取
        ctx.shared.setdefault("observability", {})["summary"] = summary
        ctx.shared["observability"]["records"] = len(self.records)
        return HookResult.continue_()

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _resolve_log(self, ctx: Context) -> EventLog:
        """[L3-1] 解析事件日志（显式注入 > session > ctx.shared > 兜底）。"""
        return resolve_event_log(
            event_log=self._event_log,
            session=self._session,
            ctx=ctx,
            fallback=self._fallback_log,
        )

    def _resolve_session(self, ctx: Context) -> Any:
        """[L3-1] 解析会话（可选依赖，取不到时返回 None）。"""
        if self._session is not None:
            return self._session
        shared = getattr(ctx, "shared", None)
        if isinstance(shared, dict):
            return shared.get("session")
        return None

    def _tick(self) -> float:
        """[L3-1] 计算距上次钩子触发的时间差（毫秒），并刷新时间戳。

        与 _last_tick 的初始化口径保持一致，均使用 perf_counter：
        纳秒级单调时钟，能真实反映短间隔两步的耗时。
        """
        now = time.perf_counter()
        elapsed_ms = round((now - self._last_tick) * 1000, 4)
        self._last_tick = now
        return elapsed_ms

    def _emit(self, record: Dict[str, Any]) -> None:
        """[L3-1] 记录并输出一条结构化日志。"""
        record.setdefault("ts", round(time.time(), 3))
        self.records.append(record)

        if self._json_logging:
            line = json.dumps(record, ensure_ascii=False, default=str)
        else:
            line = str(record)
        self.logs.append(line)

        if self._stream is not None:
            try:
                self._stream.write(line + "\n")
            except Exception:
                pass
        elif self._echo:
            print(f"[{self.name}] {line}")

    # ---- 响应/结果解析（同时兼容 dataclass 与 dict） ----

    @staticmethod
    def _extract_llm(response: Any) -> Dict[str, Any]:
        """[L3-1] 从 LLM 响应中提取指标（兼容 LLMResponse 与 dict）。"""
        if isinstance(response, dict):
            usage = response.get("usage") or {}
            cost = response.get("cost") or 0.0
            model = response.get("model") or ""
            provider = response.get("provider") or ""
            error = response.get("error") or ""
        else:
            usage = getattr(response, "usage", None) or {}
            cost = getattr(response, "cost", 0.0) or 0.0
            model = getattr(response, "model", "") or ""
            provider = getattr(response, "provider", "") or ""
            error = getattr(response, "error", "") or ""

        if not isinstance(usage, dict):
            usage = {}

        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)
        if total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens

        return {
            "model": str(model),
            "provider": str(provider),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": float(cost or 0.0),
            "success": not bool(error),
            "error": str(error),
        }

    @staticmethod
    def _extract_tool_elapsed(result: Any) -> float:
        """[L3-1] 从工具结果中提取精确耗时（毫秒），取不到返回 0。"""
        if isinstance(result, dict):
            metadata = result.get("metadata")
        else:
            metadata = getattr(result, "metadata", None)
        if isinstance(metadata, dict) and "elapsed_seconds" in metadata:
            try:
                return round(float(metadata["elapsed_seconds"]) * 1000, 3)
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    @staticmethod
    def _tool_name(tool_call: Any) -> str:
        """[L3-1] 提取工具名（兼容 dict 与对象）。"""
        if isinstance(tool_call, dict):
            return str(tool_call.get("name", "") or "")
        return str(getattr(tool_call, "name", "") or "")

    @staticmethod
    def _tool_arguments(tool_call: Any) -> Dict[str, Any]:
        """[L3-1] 提取工具参数（兼容 dict 与对象）。"""
        if isinstance(tool_call, dict):
            args = tool_call.get("arguments", tool_call.get("args", {}))
        else:
            args = getattr(tool_call, "arguments", getattr(tool_call, "args", {}))
        return dict(args) if isinstance(args, dict) else {}

    @staticmethod
    def _tool_status(result: Any) -> Any:
        """[L3-1] 判定工具执行成败，返回 (success, error)。"""
        if isinstance(result, dict):
            error = result.get("error")
            if isinstance(error, bool):
                return (not error), ("" if not error else "tool_error")
            if error:
                return False, str(error)
            return True, ""
        is_error = bool(getattr(result, "error", False))
        if is_error:
            return False, str(getattr(result, "content", "tool_error"))[:200]
        return True, ""