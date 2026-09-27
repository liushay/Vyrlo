"""
[L3-1] Session — Agent 一次运行的会话状态（含 token/cost 预算）。

定位：
    本模块属于 L3 装配层，为中间件提供跨钩子共享的会话级状态。
    不修改 L1（agent_loop.py）与 L2（runtime 四组件）的任何代码。

职责边界：
    - Session 只做状态持有：会话标识、事件日志引用、预算、元数据。
    - Session 不做任何决策；是否中止由 CostGuardMiddleware 判定并返回 HookResult。
    - EventLog 负责"记录发生了什么"，Session 负责"这次运行的整体状态"。

与中间件的依赖关系（L3 内部）：
    session.py  ->  event_log.py（Session 持有 EventLog）
    session.py  <-  builtin_middlewares/cost_guard.py（累加 token/cost 到 Budget）
    session.py  <-  builtin_middlewares/observability.py（读预算做汇总输出）

[L3-1] 本文件为 L3 第一批中间件新增，未修改任何 L1 / L2 代码。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from runtime.event_log import EventLog, resolve_event_log

# ============================================================================
# [L3-1] Budget —— 预算
# ============================================================================


@dataclass
class Budget:
    """[L3-1] 会话预算。

    Attributes:
        max_tokens:     最大可用 token 数（<=0 表示不限）。
        max_cost:       最大可用成本（美元，<=0 表示不限）。
        max_tokens_p5:  token 的 P5 分位阈值（<=0 表示不启用）。
        max_cost_p5:    cost 的 P5 分位阈值（<=0.0 表示不启用）。
        used_tokens:    已消耗 token。
        used_cost:      已消耗成本。
    """

    max_tokens: int = 0
    max_cost: float = 0.0
    max_tokens_p5: int = 0
    max_cost_p5: float = 0.0
    used_tokens: int = 0
    used_cost: float = 0.0

    def add(self, tokens: int = 0, cost: float = 0.0) -> None:
        """[L3-1] 累加一次用量。"""
        self.used_tokens += int(tokens or 0)
        self.used_cost += float(cost or 0.0)

    @property
    def token_remaining(self) -> int:
        """[L3-1] 剩余 token（不限时返回 -1）。"""
        if self.max_tokens <= 0:
            return -1
        return self.max_tokens - self.used_tokens

    @property
    def cost_remaining(self) -> float:
        """[L3-1] 剩余成本（不限时返回 -1.0）。"""
        if self.max_cost <= 0:
            return -1.0
        return self.max_cost - self.used_cost

    def token_exceeded(self) -> bool:
        """[L3-1] token 是否超预算（严格超过上限才判定）。

        采用 ``>`` 而非 ``>=``：用量恰好等于上限时视为"已用满但未超"，
        只有真正越过上限才判定超限，避免在边界处提前中止。

        绝对阈值（max_tokens）与分位阈值（max_tokens_p5）任一命中即超限。
        """
        if self.max_tokens > 0 and self.used_tokens > self.max_tokens:
            return True
        if self.max_tokens_p5 > 0 and self.used_tokens > self.max_tokens_p5:
            return True
        return False

    def cost_exceeded(self) -> bool:
        """[L3-1] 成本是否超预算（严格超过上限才判定）。

        绝对阈值（max_cost）与分位阈值（max_cost_p5）任一命中即超限。
        """
        if self.max_cost > 0 and self.used_cost > self.max_cost:
            return True
        if self.max_cost_p5 > 0 and self.used_cost > self.max_cost_p5:
            return True
        return False

    def exceeded(self) -> bool:
        """[L3-1] 是否任一维度超预算。"""
        return self.token_exceeded() or self.cost_exceeded()

    def to_dict(self) -> Dict[str, Any]:
        """[L3-1] 序列化。"""
        return {
            "max_tokens": self.max_tokens,
            "max_cost": self.max_cost,
            "max_tokens_p5": self.max_tokens_p5,
            "max_cost_p5": round(self.max_cost_p5, 8),
            "used_tokens": self.used_tokens,
            "used_cost": round(self.used_cost, 8),
            "token_remaining": self.token_remaining,
            "cost_remaining": round(self.cost_remaining, 8),
            "exceeded": self.exceeded(),
        }


# ============================================================================
# [L3-1] Session —— 会话
# ============================================================================


class Session:
    """[L3-1] 一次 Agent 运行的会话状态。

    Usage::

        session = Session(max_tokens=10000, max_cost=1.0)
        session.add_usage(tokens=120, cost=0.0012)
        session.budget.exceeded()

    公开字段：session_id / event_log / budget / metadata / created_at。
    只做状态持有与只读派生属性，不做流程决策。
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        max_tokens: int = 0,
        max_cost: float = 0.0,
        event_log: Optional[EventLog] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.session_id: str = session_id or str(uuid.uuid4())
        self.event_log: EventLog = event_log if event_log is not None else EventLog()
        self.budget: Budget = Budget(max_tokens=max_tokens, max_cost=max_cost)
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.created_at: float = time.time()
        # 结算标记：CostGuard 首次判定超预算时置位，避免重复中止
        self.abort_flag: bool = False

    # ---- 用量累加 ----

    def add_usage(self, tokens: int = 0, cost: float = 0.0) -> Budget:
        """[L3-1] 累加一次用量到预算，返回预算对象。"""
        self.budget.add(tokens=tokens, cost=cost)
        return self.budget

    def exhausted(self) -> bool:
        """[L3-1] 会话预算是否已耗尽。"""
        return self.budget.exceeded()

    # ---- 派生视图 ----

    @property
    def elapsed_seconds(self) -> float:
        """[L3-1] 会话已持续秒数。"""
        return round(time.time() - self.created_at, 3)

    def snapshot(self) -> Dict[str, Any]:
        """[L3-1] 会话状态快照（供持久化或日志输出）。"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "elapsed_seconds": self.elapsed_seconds,
            "budget": self.budget.to_dict(),
            "event_summary": self.event_log.summary(),
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> Dict[str, Any]:
        """[L3-1] 简化序列化（不含事件明细）。"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "budget": self.budget.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """[L3-1] 从字典恢复会话（事件日志不恢复）。"""
        budget_data = data.get("budget", {}) or {}
        session = cls(
            session_id=data.get("session_id"),
            max_tokens=int(budget_data.get("max_tokens", 0) or 0),
            max_cost=float(budget_data.get("max_cost", 0.0) or 0.0),
            metadata=data.get("metadata", {}) or {},
        )
        session.budget.used_tokens = int(budget_data.get("used_tokens", 0) or 0)
        session.budget.used_cost = float(budget_data.get("used_cost", 0.0) or 0.0)
        session.created_at = float(data.get("created_at", session.created_at))
        return session

    def __repr__(self) -> str:
        return (
            f"<Session id={self.session_id!r} "
            f"tokens={self.budget.used_tokens}/{self.budget.max_tokens} "
            f"cost={self.budget.used_cost:.4f}/{self.budget.max_cost}>"
        )


# ============================================================================
# [L2-装配] 会话存储已迁移
# ============================================================================
#
# SessionStore / InMemorySessionStore / SQLiteSessionStore / create_session_store
# 已迁移到 runtime/session_store/ 包（L2 装配层）。本模块只保留
# Session / Budget / resolve_session，遵循"单一职责"划分。
#
# 兼容性：为不破坏既有 `from runtime.session import SessionStore` 的写法，
# 本模块提供惰性转发（见文件末尾 __getattr__）。


# ============================================================================
# [L3-1] 依赖解析助手
# ============================================================================


def resolve_session(
    session: Optional[Session] = None,
    ctx: Any = None,
    event_log: Optional[EventLog] = None,
) -> Session:
    """[L3-1] 解析中间件应使用的 Session。

    解析优先级：
        1. 显式传入的 session
        2. ctx.shared["session"]
        3. 用 event_log（或新建）构造一个临时 Session

    保证中间件在依赖缺失时退化为可用状态，从而支持独立启停。
    """
    if session is not None:
        return session
    if ctx is not None:
        shared = getattr(ctx, "shared", None)
        if isinstance(shared, dict):
            shared_session = shared.get("session")
            if shared_session is not None:
                return shared_session
    resolved_log = resolve_event_log(event_log=event_log, ctx=ctx)
    return Session(event_log=resolved_log)


# ============================================================================
# [L2-装配] 兼容转发：SessionStore 等已迁移到 runtime.session_store
# ============================================================================

#: 迁移到 runtime.session_store 的符号
_MOVED_TO_SESSION_STORE = frozenset({
    "SessionStore",
    "InMemorySessionStore",
    "SQLiteSessionStore",
    "create_session_store",
})


def __getattr__(name: str):
    """惰性转发已迁移的会话存储符号，保持旧导入路径可用。

    仅在 `from runtime.session import SessionStore` 等属性访问时触发，
    避免在模块导入期引入 runtime.session_store -> runtime.session 的循环依赖。
    """
    if name in _MOVED_TO_SESSION_STORE:
        from runtime import session_store as _store

        return getattr(_store, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
