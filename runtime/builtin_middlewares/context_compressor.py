"""
[L3-1] ContextCompressorMiddleware — 上下文压缩中间件。

职责边界（只做超限检测与压缩触发，不做记忆注入、不做预算判定）：
    - 在 BEFORE_LLM 钩子中估算本轮消息的总 token；
    - 当总量接近上限（超过 max_tokens * threshold_ratio）时，
      调用 ContextManager.compress() 压缩工作记忆并替换本轮消息；
    - 未超限时原样放行（返回 HookResult.continue_()，不覆盖 messages）。
    - 不负责注入记忆（那是 MemoryInjectorMiddleware 的职责），
      不做安全判定、不写事件日志。

token 估算口径：
    与 L2 保持一致——按"字符数 / 4"的保守估算逐条累加，
    避免为了计数引入 tokenizer 依赖。

消息替换语义（[D010] 已改为"以当前 messages 为基准"）：
    压缩是把工作记忆（ContextManager 内部列表）压短，然后重建本轮消息：
        [当前 messages 的 system 消息]
      + [压缩后的非 system 工作记忆]
      + [当前 messages 中"未被工作记忆覆盖"的新增消息]
    最后一项是 [D010] 的关键：MemoryInjector 在 append 模式下把记忆作为
    独立 user 消息追加到本轮 messages，这条消息**不进入工作记忆**；
    旧的"仅用工作记忆重建"会把注入的记忆整条丢掉。
    system 消息始终取自当前 messages（不取自工作记忆），因此 replace 模式下
    build_context 替换好的 system prompt 不会被重建逻辑改写。
    若 ContextManager 未提供工作记忆（get_working_memory 不可用或为空），
    则退化为"保留最新 keep_recent 条消息"的本地截断，
    保证中间件在无 ContextManager 时仍可独立工作。

注册钩子：
    - BEFORE_LLM : 估算 token → 超限则 compress 并替换 messages

依赖关系：
    ContextCompressorMiddleware -> ContextManager（公开方法 compress /
                                    get_working_memory）
    通过构造参数注入，或从 ctx.shared["context_manager"] 解析。

[L3-1] 本文件为 L3 第一批中间件新增，未修改任何 L1 / L2 代码。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent_loop import Context, HookResult, Middleware


class ContextCompressorMiddleware(Middleware):
    """[L3-1] 上下文压缩中间件。

    Attributes:
        compressed_count: 触发压缩的累计次数。
        last_estimate:    最近一次估算的 token 总数。
    """

    #: [C1] 本中间件在 BEFORE_LLM 返回压缩改写后的 messages，
    #: 声明为改写型 → _dispatch_hook 对其入参做 deepcopy 隔离。
    mutates_args: bool = True

    def __init__(
        self,
        context_manager: Any = None,
        max_tokens: int = 4096,
        threshold_ratio: float = 0.8,
        strategy: Optional[str] = None,
        keep_recent: int = 10,
        name: str = "context_compressor",
    ) -> None:
        """
        Args:
            context_manager: ContextManager 实例（需实现 compress）。
            max_tokens:      消息总 token 上限。
            threshold_ratio: 触发压缩的比例阈值（默认 0.8，即到达 80% 上限时压缩）。
            strategy:        传给 ContextManager.compress(strategy) 的策略名。
            keep_recent:     无 ContextManager 时本地保留的最新消息条数。
            name:            中间件名称。
        """
        super().__init__(name)
        self._context_manager = context_manager
        self._max_tokens = max_tokens
        self._threshold_ratio = threshold_ratio
        self._strategy = strategy
        self._keep_recent = keep_recent

        self.compressed_count = 0
        self.last_estimate = 0
        # 首次压缩前的 token 总量，用于观测"压缩总量从多少降到多少"
        self.first_before_tokens: Optional[int] = None
        # [C2] 只压缩旧消息（本地截断）的次数，不触发破坏性 cm.compress 重建
        self.light_compress_count = 0

    # ------------------------------------------------------------------
    # BEFORE_LLM —— 超限检测 + 压缩
    # ------------------------------------------------------------------

    def before_llm(
        self, ctx: Context, messages: List[Dict[str, Any]]
    ) -> HookResult:
        """[L3-1] BEFORE_LLM：消息总量接近上限时压缩并替换消息。

        [D005] 估算范围：
        - 优先估算本轮 Loop messages（在 D004 链式传递下，已包含 MemoryInjector
          等前置中间件注入后的完整消息）；
        - 当 Loop messages 未超限时，额外估算 ContextManager 工作记忆。长对话下
          工作记忆持续增长，仅估算极短的 Loop messages 会导致压缩永不触发。
        """
        current = list(messages or [])
        cm = self._resolve_context_manager(ctx)
        estimate = self._estimate_messages(current)
        self.last_estimate = estimate

        threshold = int(self._max_tokens * self._threshold_ratio)
        if estimate <= threshold:
            # [D005] Loop messages 未超限：把工作记忆纳入估算。
            working = self._working_memory_as_dicts(cm)
            working_estimate = self._estimate_messages(working)
            if working_estimate <= threshold:
                return HookResult.continue_()
            estimate = working_estimate

        # [C2] 压缩触发点的微调：先尝试"只压缩旧消息"（本地截断保留最近若干条），
        # 若轻量截断后已低于阈值，则不再调用 cm.compress 重建全部——
        # 避免破坏性重建（cm.compress 会改写整个工作记忆），从而降低压缩成本、
        # 提升 cache 命中（system + 最近消息字节不变）。阈值本身不变。
        light = self._light_compress(cm, current)
        light_tokens = self._estimate_messages(light)
        if light_tokens <= threshold:
            if self.first_before_tokens is None:
                self.first_before_tokens = estimate
            self.light_compress_count += 1
            self.compressed_count += 1
            ctx.shared["context_compressed"] = {
                "first_before_tokens": self.first_before_tokens,
                "before_tokens": estimate,
                "after_tokens": light_tokens,
                "threshold": threshold,
                "count": self.compressed_count,
                "mode": "light",
            }
            return HookResult(payload={"messages": light})

        compressed = self._compress(ctx, cm, current)
        after_tokens = self._estimate_messages(compressed)

        if self.first_before_tokens is None:
            self.first_before_tokens = estimate

        self.compressed_count += 1
        ctx.shared["context_compressed"] = {
            "first_before_tokens": self.first_before_tokens,
            "before_tokens": estimate,
            "after_tokens": after_tokens,
            "threshold": threshold,
            "count": self.compressed_count,
            "mode": "rebuild",
        }
        return HookResult(payload={"messages": compressed})

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _resolve_context_manager(self, ctx: Context) -> Any:
        """[L3-1] 解析 ContextManager（构造注入 > ctx.shared）。"""
        if self._context_manager is not None:
            return self._context_manager
        shared = getattr(ctx, "shared", None)
        if isinstance(shared, dict):
            return shared.get("context_manager")
        return None

    def _light_compress(
        self, cm: Any, current: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """[C2] 轻量压缩：只压缩旧消息（本地截断），不调用 cm.compress 重建全部。

        保留 system 消息 + 最近 ``keep_recent`` 条非 system 消息，
        与无 ContextManager 时的本地截断语义一致。此路径不触碰工作记忆，
        system 与最近消息的字节保持不变（cache 友好），且不引入重建开销。
        """
        system_msgs = [m for m in current if m.get("role") == "system"]
        current_rest = [m for m in current if m.get("role") != "system"]
        kept = (
            current_rest[-self._keep_recent :]
            if self._keep_recent > 0
            else current_rest
        )
        return system_msgs + kept

    def _compress(
        self,
        ctx: Context,
        cm: Any,
        current: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """[L3-1] 压缩工作记忆并重建本轮消息（[D010] 以当前 messages 为基准）。"""
        system_msgs = [m for m in current if m.get("role") == "system"]
        current_rest = [m for m in current if m.get("role") != "system"]

        if cm is not None:
            self._invoke_compress(cm)
            working = self._working_memory_as_dicts(cm)
            if working:
                working_rest = [m for m in working if m.get("role") != "system"]
                if working_rest:
                    # [D010] 追加当前 messages 中未被压缩后工作记忆覆盖的消息。
                    # 典型来源：append 模式注入的记忆 / 尚未入工作记忆的本轮 task。
                    extra = self._uncovered_messages(working_rest, current_rest)
                    return system_msgs + working_rest + extra

        # 退化路径：无 ContextManager 或工作记忆为空时，本地保留最新若干条
        kept = (
            current_rest[-self._keep_recent :]
            if self._keep_recent > 0
            else current_rest
        )
        return system_msgs + kept

    @staticmethod
    def _uncovered_messages(
        working_rest: List[Dict[str, Any]],
        current_rest: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """[D010] 返回当前 messages 中未被工作记忆覆盖的尾部消息。

        判定方式：以压缩后工作记忆的**末条消息**为锚点，在当前 messages 中
        从后向前查找同一条消息（按 role + content 比较）。
          - 找到：锚点之后的消息是工作记忆之后新增的（如 append 注入的记忆），
            返回它们；锚点及之前的消息已由工作记忆承载，不重复保留。
          - 未找到：两者不同源（例如工作记忆是另一段历史），此时不做覆盖，
            返回当前全部非 system 消息，宁可保守也不丢内容。
        """
        if not working_rest:
            return list(current_rest)
        if not current_rest:
            return []

        anchor = working_rest[-1]
        key = (anchor.get("role"), anchor.get("content"))
        for idx in range(len(current_rest) - 1, -1, -1):
            item = current_rest[idx]
            if (item.get("role"), item.get("content")) == key:
                return list(current_rest[idx + 1 :])
        return list(current_rest)

    def _invoke_compress(self, cm: Any) -> None:
        """[L3-1] 调用 ContextManager.compress，容错忽略失败。"""
        compressor = getattr(cm, "compress", None)
        if not callable(compressor):
            return
        try:
            if self._strategy is not None:
                compressor(self._strategy)
            else:
                try:
                    compressor()
                except TypeError:
                    compressor(None)
        except Exception:
            pass

    @staticmethod
    def _working_memory_as_dicts(cm: Any) -> List[Dict[str, Any]]:
        """[L3-1] 读取压缩后的工作记忆并统一转为 dict 形式。"""
        getter = getattr(cm, "get_working_memory", None)
        if not callable(getter):
            return []
        try:
            working = getter()
        except Exception:
            return []
        if not isinstance(working, list):
            return []

        result: List[Dict[str, Any]] = []
        for item in working:
            if isinstance(item, dict):
                result.append(item)
                continue
            to_dict = getattr(item, "to_dict", None)
            if callable(to_dict):
                try:
                    converted = to_dict()
                except Exception:
                    continue
                if isinstance(converted, dict):
                    result.append(converted)
        return result

    @staticmethod
    def _estimate_messages(messages: List[Dict[str, Any]]) -> int:
        """[L3-1] 按字符数/4 保守估算消息总 token（与 L2 口径一致）。"""
        total = 0
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content", "")
            role = msg.get("role", "")
            if not isinstance(content, str):
                content = str(content)
            total += max(1, (len(content) + len(str(role))) // 4)
        return total