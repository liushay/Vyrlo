"""
[L3-1] CostGuardMiddleware — 成本守卫中间件。

职责边界（只做预算判定，不做记录与展示）：
    - 在 AFTER_LLM 钩子中，从 LLMCallEvent 读取本次调用的 token 与 cost，
      累加到 Session.budget。
    - 一旦预算超限，返回 HookResult.abort_loop() 中止整个 Loop。
    - 不写事件日志（写入方是 ObservabilityMiddleware），不输出日志、不拦截工具。

预算是唯一权威来源：
    CostGuardMiddleware 是 Session.budget 的唯一权威累加方与判定方。
    它只从 EventLog 消费 LLMCallEvent 事件并累加，与 MultiProviderAdapter
    写入的 ctx.shared["cost_log"] 完全解耦——后者仅用于观测记录，不参与
    任何预算累加。因此两者同时启用时，同一批 LLM 调用只会在本中间件中
    累加一次，不会出现成本双重计数。

数据来源（按优先级）：
    1. Session.event_log 中最新的 LLMCallEvent（与 Observability 协作的推荐路径）
    2. 直接解析 after_llm 的 response 入参（独立启用时的兜底）

    这样设计让 CostGuard 可以脱离 Observability 独立启停：
    若没有可用的 LLMCallEvent，则退化到从 response 直接读取用量。

幂等性：
    [S5] 通过"已消费事件唯一标识集合"（_consumed_ids）避免同一条事件被
    重复累加。此前用 int 下标依赖 `log.llm_calls()` 长度单调递增，当
    SQLiteEventLog 配合 max_events > 0 时旧事件被裁剪、长度不再增长（甚至
    缩短），下标随之错位，导致后续事件漏计。

    标识选取（按后端分派，均为只读探测，不改后端类也不改 EventLog 接口）：
        1. SQLite 后端：使用持久化主键 ``(session_id, seq)``。SQLite 每轮读取
           都会重建事件对象，对象身份不稳定，而 ``seq`` 天然唯一。标识通过
           对底层表的只读查询获得，行数与 ``llm_calls()`` 不一致时保守回退。
        2. 其他后端：使用事件对象身份 ``id(event)``。内存后端持有事件对象的
           强引用，同一事件身份恒定，可稳定去重。**不使用时间戳等字段做内容
           指纹**——Windows 上 ``time.time()`` 分辨率约 15.6ms，快速连续调用
           会得到相同时间戳，指纹会把不同事件误判为同一条而漏计。
    集合上限：超过 _CONSUMED_LIMIT 时只保留最近 _CONSUMED_KEEP 条
    （list 维护顺序 + set 维护查找），避免长会话内存膨胀。

注册钩子：
    - AFTER_LLM : 累加用量 → 超预算则 abort_loop

依赖关系：
    CostGuardMiddleware -> runtime.event_log（读 LLMCallEvent）
    CostGuardMiddleware -> runtime.session（写 Session.budget）

[L3-1] 本文件为 L3 第一批中间件新增，未修改任何 L1 / L2 代码。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from agent_loop import Context, HookResult, Middleware

from runtime.event_log import EVENT_LLM_CALL, EventLog, resolve_event_log
from runtime.session import Session, resolve_session

# ============================================================================
# [S5] 已消费事件标识的容量约束
# ============================================================================

#: 已消费标识集合的上限：超过即裁剪
_CONSUMED_LIMIT = 1000

#: 裁剪时保留的最近标识条数（少于上限，避免每次调用都触发裁剪）
_CONSUMED_KEEP = 500

#: 表名合法性（SQLite 只读探测时防注入）
_SAFE_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class CostGuardMiddleware(Middleware):
    """[L3-1] 成本守卫中间件 —— Session.budget 的唯一权威来源。

    预算是唯一权威来源，与 MultiProviderAdapter 的 ctx.shared["cost_log"]
    解耦：本中间件从 EventLog 消费 LLMCallEvent 事件累加预算，cost_log 只
    承担观测记录职责。二者同时启用不会导致预算被重复累加。

    Attributes:
        abort_reason: 触发中止时的原因摘要（未触发则为空字符串）。
    """

    def __init__(
        self,
        session: Optional[Session] = None,
        event_log: Optional[EventLog] = None,
        max_tokens: int = 0,
        max_cost: float = 0.0,
        name: str = "cost_guard",
    ) -> None:
        """
        Args:
            session:    会话实例。为 None 时从 ctx.shared 解析。
            event_log:  事件日志实例。为 None 时从 session / ctx.shared 解析。
            max_tokens: 若提供且会话尚未设置，则作为 token 预算上限。
            max_cost:   若提供且会话尚未设置，则作为成本预算上限。
            name:       中间件名称。
        """
        super().__init__(name, priority=0)
        self._session = session
        self._event_log = event_log
        self._max_tokens = max_tokens
        self._max_cost = max_cost

        # 兜底：保证无外部依赖时可独立启停
        self._fallback_log = EventLog()
        self._fallback_session = Session(event_log=self._fallback_log)

        # [S5] 已消费事件的唯一标识：set 负责 O(1) 查重，list 维护插入顺序
        self._consumed_ids: set = set()
        self._consumed_order: List[Any] = []
        self._consumed_limit: int = _CONSUMED_LIMIT
        self._consumed_keep: int = _CONSUMED_KEEP

        self.abort_reason: str = ""

    # ------------------------------------------------------------------
    # AFTER_LLM —— 累加用量 + 预算判定
    # ------------------------------------------------------------------

    def after_llm(self, ctx: Context, response: Any) -> HookResult:
        """[L3-1] AFTER_LLM：累加本次调用用量，超预算则中止 Loop。"""
        session = self._resolve_session(ctx)
        log = self._resolve_log(ctx, session)

        self._apply_budget_config(session)

        # 已中止过则直接返回中止结果，避免重复累加与重复中止
        if getattr(session, "abort_flag", False):
            return self._abort(session, repeated=True)

        # [S5] 按唯一标识消费"尚未累加过"的事件。
        # 不再依赖列表下标：SQLite 后端在 max_events 裁剪下长度会停滞/缩短，
        # 下标会错位并漏计后续事件。
        new_events = self._collect_unconsumed(log, log.llm_calls())
        if new_events:
            for event in new_events:
                session.add_usage(
                    tokens=getattr(event, "total_tokens", 0) or 0,
                    cost=getattr(event, "cost", 0.0) or 0.0,
                )
        else:
            usage = self._extract_usage(response)
            if usage["total_tokens"] or usage["cost"]:
                session.add_usage(tokens=usage["total_tokens"], cost=usage["cost"])

        if session.budget.exceeded():
            return self._abort(session, repeated=False)

        return HookResult.continue_()

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _abort(self, session: Session, repeated: bool) -> HookResult:
        """[L3-1] 构造中止结果并记录原因。"""
        budget = session.budget
        if budget.max_tokens > 0 and budget.used_tokens > budget.max_tokens:
            reason = f"token 预算超限: {budget.used_tokens} >= {budget.max_tokens}"
        elif budget.max_tokens_p5 > 0 and budget.used_tokens > budget.max_tokens_p5:
            reason = f"token P5 分位阈值超限: {budget.used_tokens} >= {budget.max_tokens_p5}"
        elif budget.max_cost > 0 and budget.used_cost > budget.max_cost:
            reason = f"成本预算超限: {budget.used_cost:.6f} >= {budget.max_cost}"
        elif budget.max_cost_p5 > 0 and budget.used_cost > budget.max_cost_p5:
            reason = f"成本 P5 分位阈值超限: {budget.used_cost:.6f} >= {budget.max_cost_p5}"
        else:
            reason = "预算已超限"

        self.abort_reason = reason
        session.abort_flag = True
        return HookResult.abort_loop({
            "reason": reason,
            "budget": budget.to_dict(),
            "repeated": repeated,
        })

    def _apply_budget_config(self, session: Session) -> None:
        """[L3-1] 将构造参数中的预算应用到会话（仅在会话未设置时生效）。"""
        if self._max_tokens and not session.budget.max_tokens:
            session.budget.max_tokens = self._max_tokens
        if self._max_cost and not session.budget.max_cost:
            session.budget.max_cost = self._max_cost

    def _resolve_session(self, ctx: Context) -> Session:
        """[L3-1] 解析会话（显式注入 > ctx.shared > 兜底实例）。"""
        if self._session is not None:
            return self._session

        resolved = resolve_session(ctx=ctx)
        shared = getattr(ctx, "shared", None)
        if isinstance(shared, dict) and shared.get("session") is not None:
            return resolved

        # 无外部会话：统一落到兜底实例，保证同一次运行内用量可累积
        self._fallback_session.event_log = self._fallback_log
        return self._fallback_session

    def _resolve_log(self, ctx: Context, session: Session) -> EventLog:
        """[L3-1] 解析事件日志（显式注入 > 会话 > ctx.shared > 兜底）。"""
        return resolve_event_log(
            event_log=self._event_log,
            session=session,
            ctx=ctx,
            fallback=self._fallback_log,
        )

    @staticmethod
    def _extract_usage(response: Any) -> Dict[str, Any]:
        """[L3-1] 从 response 直接提取用量（兼容 dict 与 LLMResponse）。"""
        if isinstance(response, dict):
            usage = response.get("usage") or {}
            cost = response.get("cost") or 0.0
        else:
            usage = getattr(response, "usage", None) or {}
            cost = getattr(response, "cost", 0.0) or 0.0

        if not isinstance(usage, dict):
            usage = {}

        total = int(usage.get("total_tokens", 0) or 0)
        if total <= 0:
            total = int(usage.get("prompt_tokens", 0) or 0) + int(
                usage.get("completion_tokens", 0) or 0
            )
        return {"total_tokens": total, "cost": float(cost or 0.0)}

    # ------------------------------------------------------------------
    # [S5] 事件消费去重
    # ------------------------------------------------------------------

    def _collect_unconsumed(self, log: EventLog, events: Any) -> List[Any]:
        """[S5] 返回本次尚未消费过的事件，并把其标识登记进已消费集合。

        与旧下标方案的区别：下标假设"事件列表只增不减"，一旦后端裁剪旧事件
        就会错位（长度停滞甚至缩短），唯一标识则对增删都免疫。

        实现采用"尾部锚定 + 标识去重"两步：
            1. 为每个事件计算稳定唯一标识（见 ``_stable_keys``）；
            2. 从尾部向前找到第一个"已消费"的事件作为锚点，其后的区间即为
               本次新增；区间内再用标识兜底去重。
        锚点缺失（首次调用、或日志被清空重建）时退化为扫描全部事件——
        与旧实现首轮的 ``[0:]`` 行为一致。
        """
        events = list(events or [])
        if not events:
            return []

        keys = self._stable_keys(log, events)

        start = 0
        for idx in range(len(keys) - 1, -1, -1):
            if keys[idx] in self._consumed_ids:
                start = idx + 1
                break

        fresh: List[Any] = []
        for idx in range(start, len(keys)):
            key = keys[idx]
            if key in self._consumed_ids:
                continue
            self._consumed_ids.add(key)
            self._consumed_order.append(key)
            fresh.append(events[idx])
        self._trim_consumed()
        return fresh

    def _stable_keys(self, log: EventLog, events: List[Any]) -> List[Any]:
        """[S5] 为一组事件计算稳定唯一标识（与入参一一对应）。"""
        seq_keys = self._sqlite_seq_keys(log, len(events))
        if seq_keys is not None:
            return seq_keys
        return [self._event_key(event) for event in events]

    @staticmethod
    def _sqlite_seq_keys(log: EventLog, expected: int) -> Optional[List[Any]]:
        """[S5] 从 SQLite 后端只读地取回事件的 ``(session_id, seq)`` 标识。

        ``SQLiteEventLog`` 每轮查询都会重建事件对象，对象身份不稳定；而其持久化
        主键 ``(session_id, seq)`` 天然唯一且会话内单调递增。这里以"只读探测"
        的方式直接查询底层表——**不修改后端类、不修改 EventLog 接口**——按与
        ``llm_calls()`` 相同的排序取回标识，从而与入参事件列表一一对应。

        Returns:
            标识列表；若后端不是 SQLite、查询失败，或行数与 ``expected`` 不一致，
            返回 None，由调用方回退到对象身份方案。
        """
        if getattr(log, "backend", "") != "sqlite":
            return None

        conn = getattr(log, "_conn", None)
        table = getattr(log, "_table", None)
        if conn is None or not isinstance(table, str) or not _SAFE_TABLE_RE.match(table):
            return None

        sql = (
            f"SELECT session_id, seq FROM {table} WHERE event_type = ? "
            f"ORDER BY session_id ASC, seq ASC"
        )
        lock = getattr(log, "_lock", None)
        try:
            if lock is not None:
                with lock:
                    rows = conn.execute(sql, (EVENT_LLM_CALL,)).fetchall()
            else:  # pragma: no cover - 非法后端形态，保守回退
                rows = conn.execute(sql, (EVENT_LLM_CALL,)).fetchall()
            keys = [("seq", str(row["session_id"]), int(row["seq"])) for row in rows]
        except Exception:  # pragma: no cover - 后端形态变化时保守回退
            return None

        return keys if len(keys) == expected else None

    @staticmethod
    def _event_key(event: Any) -> Any:
        """[S5] 计算事件的稳定唯一标识（非 SQLite 后端的方案）。

        优先使用事件自带的 ``(session_id, seq)``；否则使用事件对象身份
        ``id(event)``：内存后端的 EventLog 持有事件对象的强引用，同一事件在同一
        列表生命周期内身份恒定，因此可稳定去重。**不使用时间戳等字段做内容
        指纹**——Windows 上 ``time.time()`` 分辨率约 15.6ms，快速连续的多次调用
        会得到相同时间戳，指纹会把不同事件误判为同一条而漏计。
        """
        session_id = getattr(event, "session_id", None)
        seq = getattr(event, "seq", None)
        if seq is not None:
            return ("seq", str(session_id or ""), int(seq))
        return ("obj", id(event))

    def _trim_consumed(self) -> None:
        """[S5] 已消费标识集合超过上限时，只保留最近 _consumed_keep 条。"""
        limit = self._consumed_limit
        if limit <= 0 or len(self._consumed_order) <= limit:
            return
        keep = max(int(self._consumed_keep), 0)
        drop = self._consumed_order[: len(self._consumed_order) - keep] if keep else list(
            self._consumed_order
        )
        if not drop:
            return
        del self._consumed_order[: len(drop)]
        for key in drop:
            self._consumed_ids.discard(key)
