"""
[L3-1] MemoryInjectorMiddleware — 记忆注入中间件。

职责边界（只做上下文构建与消息注入，不做压缩、不做预算判定）：
    - 在 BEFORE_LLM 钩子中调用 ContextManager.build_context()，
      生成包含长期记忆（L4 注入）与工作记忆的完整消息列表；
    - 通过 HookResult.payload["messages"] 返回给 Agent Loop 用于 LLM 调用。
    - 不压缩消息（那是 ContextCompressorMiddleware 的职责），
      不修改工具、不做安全判定、不写事件日志。

为什么用 build_context：
    LayeredContextManager.build_context(system_prompt, namespace) 的职责正是
    "system prompt（含 {{memory_context}} 占位符替换）+ 工作记忆消息列表"。
    本中间件只负责调用它并把结果交给 Loop，不重复实现注入逻辑。

与工作记忆的关系：
    Agent Loop 的 messages 是"本轮向 LLM 发送的消息"，
    ContextManager 的工作记忆是"跨轮累积的消息历史"。
    默认策略（use_working_memory=False）只注入 system prompt + 记忆，
    保留 Loop 传入的对话消息，避免与其它中间件的消息管理策略冲突；
    当 use_working_memory=True 时，完全以 build_context 的结果为准。

注册钩子：
    - BEFORE_LLM : build_context → HookResult.payload["messages"]

依赖关系：
    MemoryInjectorMiddleware -> ContextManager（公开方法 build_context）
    通过构造参数注入，或从 ctx.shared["context_manager"] 解析。

[L3-1] 本文件为 L3 第一批中间件新增，未修改任何 L1 / L2 代码。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent_loop import Context, HookResult, Middleware


class MemoryInjectorMiddleware(Middleware):
    """[L3-1] 记忆注入中间件。

    Attributes:
        last_injected_system: 最近一次注入的 system prompt 文本（便于断言）。
    """

    #: [C1] 本中间件在 BEFORE_LLM 返回改写后的 messages（注入记忆），
    #: 声明为改写型 → _dispatch_hook 对其入参做 deepcopy 隔离。
    mutates_args: bool = True

    def __init__(
        self,
        context_manager: Any = None,
        system_prompt: str = "",
        namespace: Optional[str] = None,
        use_working_memory: bool = False,
        inject_mode: str = "append",
        name: str = "memory_injector",
        top_k: int = 5,
    ) -> None:
        """
        Args:
            context_manager:    ContextManager 实例（需实现 build_context / retrieve）。
            system_prompt:      基础 system prompt 模板，可含 {{memory_context}} 占位符。
            namespace:          长期记忆命名空间。
            use_working_memory: True 时完全以 build_context 结果作为本轮消息；
                                False 时只注入 system prompt，保留原对话消息。
                                 仅在 inject_mode="replace" 时生效。
            inject_mode:        注入模式："append"（默认）| "replace"。
                                 - "append"：记忆文本作为独立的 user 消息追加到 messages
                                   末尾，system prompt 保持原样（不替换 {{memory_context}}），
                                   不破坏 system prompt 缓存（cache 友好）。
                                 - "replace"：沿用旧行为，build_context 替换 system prompt
                                   中的 {{memory_context}} 占位符（向后兼容）。
            top_k:              [C2] 记忆检索返回条数（默认 5，与历史行为一致）。
                                可通过降低 top_k 减少注入量（召回率不变）。
                                注意：top_k 是"如何用"维度的注入规模参数，
                                不是 B10 校准的六个阈值之一，不改变任何校准常量。
            name:               中间件名称。
        """
        super().__init__(name)
        self._context_manager = context_manager
        self._system_prompt = system_prompt
        self._namespace = namespace
        self._use_working_memory = use_working_memory
        self._inject_mode = inject_mode if inject_mode in ("append", "replace") else "append"
        self._top_k = max(1, int(top_k)) if isinstance(top_k, int) else 5

        self.last_injected_system: str = ""

        # [C2] system prompt 一致性守护：记录历次 BEFORE_LLM 观察到的 system
        # prompt 字节串，若多次调用间不一致说明有中间件改写了它（定位改写源）。
        self._seen_system_prompts: List[str] = []
        self._system_prompt_inconsistent = False

    # ------------------------------------------------------------------
    # BEFORE_LLM —— 构建含记忆的消息
    # ------------------------------------------------------------------

    def before_llm(
        self, ctx: Context, messages: List[Dict[str, Any]]
    ) -> HookResult:
        """[L3-1] BEFORE_LLM：按 inject_mode 注入记忆。

        - "append"：生成记忆文本作为独立 user 消息追加，system prompt 保持原样。
        - "replace"：沿用旧行为，build_context 替换 system prompt 占位符。
        """
        # [C2] 记录本轮观察到的 system prompt（用于一致性守护 / cache 命中诊断）
        self._observe_system_prompt(ctx, messages)

        cm = self._resolve_context_manager(ctx)
        if cm is None:
            # 无 ContextManager：不做任何注入，放行
            return HookResult.continue_()

        system_prompt = self._resolve_system_prompt(ctx)
        namespace = self._resolve_namespace(ctx)

        if self._inject_mode == "append":
            return self._inject_append(ctx, cm, system_prompt, namespace, messages)
        return self._inject_replace(ctx, cm, system_prompt, namespace, messages)

    def _inject_append(
        self,
        ctx: Context,
        cm: Any,
        system_prompt: str,
        namespace: Optional[str],
        messages: List[Dict[str, Any]],
    ) -> HookResult:
        """[L3] append 模式：记忆文本追加为独立 user 消息，不替换 system prompt。

        cache 友好：system prompt 每次都保持不变，不会因记忆变化而生成不同文本，
        从而可被上游（如 LLM 网关的 prompt cache）稳定复用。

        记忆文本生成按 build_context 的原逻辑：取本轮最后一条用户消息作为查询，
        调用 ContextManager.retrieve 检索，再用 injection_strategy.inject 格式化。
        """
        memory_text = self._build_memory_text(cm, namespace, messages)

        new_messages = list(messages or [])
        if memory_text:
            new_messages.append({"role": "user", "content": memory_text})

        self.last_injected_system = system_prompt
        ctx.shared["memory_injected"] = {
            "messages": len(new_messages),
            "system_prompt_len": len(system_prompt),
            "namespace": namespace,
            "inject_mode": "append",
            "memory_injected": bool(memory_text),
        }
        return HookResult(payload={"messages": new_messages})

    def _build_memory_text(
        self,
        cm: Any,
        namespace: Optional[str],
        messages: List[Dict[str, Any]],
    ) -> str:
        """[L3] 生成记忆文本（与 build_context 的检索/注入逻辑保持一致）。

        [C2] 检索条数使用构造参数 ``top_k``（默认 5），使注入量可调优：
        降低 top_k 可在召回率不变的前提下减少注入 prompt 的 token。
        """
        query = ""
        for msg in reversed(messages or []):
            if isinstance(msg, dict) and msg.get("role") == "user":
                query = str(msg.get("content", ""))[-200:]
                break
        if not query:
            return ""

        retriever = getattr(cm, "retrieve", None)
        if not callable(retriever):
            return ""
        try:
            retrieved = retriever(query, namespace=namespace, top_k=self._top_k)
        except TypeError:
            try:
                retrieved = retriever(query)
            except Exception:
                return ""
        except Exception:
            return ""

        injection = getattr(cm, "injection_strategy", None)
        inject = getattr(injection, "inject", None)
        if not callable(inject):
            return ""
        try:
            text = inject(retrieved)
        except Exception:
            return ""
        if text == "（无相关记忆）":
            return ""
        return str(text)

    def _inject_replace(
        self,
        ctx: Context,
        cm: Any,
        system_prompt: str,
        namespace: Optional[str],
        messages: List[Dict[str, Any]],
    ) -> HookResult:
        """[L3] replace 模式：沿用旧行为，build_context 替换 system prompt 占位符。"""
        built = self._build(cm, system_prompt, namespace)
        if not built:
            return HookResult.continue_()

        injected = self._merge(built, list(messages or []))

        # 记录注入的 system prompt，便于外部断言与审计
        for msg in built:
            if isinstance(msg, dict) and msg.get("role") == "system":
                self.last_injected_system = str(msg.get("content", ""))
                break

        ctx.shared["memory_injected"] = {
            "messages": len(injected),
            "system_prompt_len": len(self.last_injected_system),
            "namespace": namespace,
            "inject_mode": "replace",
        }
        return HookResult(payload={"messages": injected})

    # ------------------------------------------------------------------
    # [C2] system prompt 一致性守护
    # ------------------------------------------------------------------

    def _observe_system_prompt(
        self, ctx: Context, messages: List[Dict[str, Any]]
    ) -> None:
        """[C2] 记录本轮传入 messages 中的 system prompt 字节串。

        append 模式下 MemoryInjector **不改写** system prompt，因此这个观察点
        正好能捕获"是哪个中间件在多次调用间改写了 system prompt"：若这里看到的
        system 消息在不同调用间不一致，说明有**其他**中间件改写了它（本中间件的
        优先级较低，观察到的已是上游改写后的结果）。

        观测结果：
            - ``_seen_system_prompts``：去重后的 system prompt 文本序列。
            - ``_system_prompt_inconsistent``：是否出现过不一致（True 表示
              system prompt 在多次调用间发生了改写，会破坏 prompt cache 命中）。
            - ``ctx.shared["system_prompt_consistency"]``：per-run 快照。
        """
        system = next(
            (
                str(m.get("content", ""))
                for m in (messages or [])
                if isinstance(m, dict) and m.get("role") == "system"
            ),
            None,
        )
        if system is None:
            # 本轮无 system 消息（可能是空模板），不参与一致性判定
            return

        seen = self._seen_system_prompts
        if seen and system != seen[-1]:
            self._system_prompt_inconsistent = True
        seen.append(system)

        if getattr(ctx, "shared", None) is not None:
            ctx.shared["system_prompt_consistency"] = {
                "consistent": not self._system_prompt_inconsistent,
                "distinct_prompts": len(set(seen)),
                "observed_calls": len(seen),
            }

    @property
    def system_prompt_consistent(self) -> bool:
        """[C2] system prompt 在多次 BEFORE_LLM 调用间是否保持一致（cache 友好）。"""
        return not self._system_prompt_inconsistent

    @property
    def seen_system_prompts(self) -> List[str]:
        """[C2] 去重前的 system prompt 观察序列（用于定位改写源）。"""
        return list(self._seen_system_prompts)

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

    def _resolve_system_prompt(self, ctx: Context) -> str:
        """[L3-1] 解析 system prompt 模板（构造参数 > ctx.shared）。"""
        if self._system_prompt:
            return self._system_prompt
        shared = getattr(ctx, "shared", None)
        if isinstance(shared, dict):
            template = shared.get("system_prompt_template")
            if template:
                return str(template)
        return ""

    def _resolve_namespace(self, ctx: Context) -> Optional[str]:
        """[L3-1] 解析命名空间（构造参数 > ctx.shared）。"""
        if self._namespace:
            return self._namespace
        shared = getattr(ctx, "shared", None)
        if isinstance(shared, dict):
            ns = shared.get("memory_namespace")
            if ns:
                return str(ns)
        return None

    @staticmethod
    def _build(cm: Any, system_prompt: str, namespace: Optional[str]) -> List[Any]:
        """[L3-1] 调用 ContextManager.build_context，容错返回空列表。"""
        builder = getattr(cm, "build_context", None)
        if not callable(builder):
            return []
        try:
            result = builder(system_prompt=system_prompt, namespace=namespace)
        except TypeError:
            # 兼容只接受位置参数的实现
            try:
                result = builder(system_prompt)
            except Exception:
                return []
        except Exception:
            return []
        if isinstance(result, list):
            return result
        return []

    def _merge(
        self, built: List[Any], original: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """[L3-1] 合并 build_context 结果与 Loop 传入的消息。"""
        normalized = [m for m in built if isinstance(m, dict)]

        if self._use_working_memory:
            # 完全以 build_context 结果为准
            return normalized

        # 仅注入/替换 system 消息，保留原对话消息
        built_system = [m for m in normalized if m.get("role") == "system"]
        original_rest = [m for m in original if m.get("role") != "system"]
        return built_system + original_rest