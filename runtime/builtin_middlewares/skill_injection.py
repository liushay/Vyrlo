"""
[L5] SkillInjectionMiddleware — 技能注入中间件。

职责边界：
    - BEFORE_LLM 钩子中调用 SkillSelector 选择应注入的技能；
    - 把选中技能的文本注入到 prompt（system 消息 / 追加 user 消息）；
    - 写 ``ctx.shared["skill_selected"]``；
    - **未启用 skill_system 时静默放行**（ctx.shared["skill_system"] 缺失）。

依赖：
    - 从 ``ctx.shared["skill_system"]`` 解析 SkillSystem 门面；
    - 从 ``ctx.shared["session"]`` 读取任务描述（可选）。

注入方式（默认 append）：把选中技能文本拼接为一条 user 消息追加到 messages
末尾，保留原 system prompt（cache 友好，与 MemoryInjector 的 append 模式一致）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent_loop import Context, HookResult, Middleware


class SkillInjectionMiddleware(Middleware):
    """[L5] 技能注入中间件。

    Attributes:
        inject_mode: 注入模式："append"（默认）| "system"。
        last_selected: 最近一次选中的技能列表（便于断言）。
    """

    #: [C1] 本中间件在 BEFORE_LLM 返回注入技能后的 messages，
    #: 声明为改写型 → _dispatch_hook 对其入参做 deepcopy 隔离。
    mutates_args: bool = True

    def __init__(
        self,
        inject_mode: str = "append",
        name: str = "skill_injection",
    ) -> None:
        """
        Args:
            inject_mode: 注入模式。
                         - "append"：技能文本作为独立 user 消息追加到 messages。
                         - "system"：技能文本拼接到 system prompt 末尾。
            name:        中间件名称。
        """
        super().__init__(name)
        self._inject_mode = inject_mode if inject_mode in ("append", "system") else "append"
        self.last_selected: List[Any] = []

    def before_llm(
        self, ctx: Context, messages: List[Dict[str, Any]]
    ) -> HookResult:
        """[L5] BEFORE_LLM：选择并注入技能。

        未启用 skill_system（ctx.shared["skill_system"] 缺失）或未选中技能时，
        返回 continue，不做任何修改（零开销路径）。
        """
        skill_system = self._resolve_skill_system(ctx)
        if skill_system is None:
            return HookResult.continue_()

        task = self._resolve_task(ctx)
        selected = skill_system.select(task)
        self.last_selected = selected

        if not selected:
            ctx.shared["skill_selected"] = {"count": 0, "skills": []}
            return HookResult.continue_()

        # 写入选中的技能（供观察中间件与断言读取）
        ctx.shared["skill_selected"] = {
            "count": len(selected),
            "skills": [s.to_dict() for s in selected],
            "skill_ids": [s.skill_id for s in selected],
        }

        new_messages = self._inject(messages, selected)
        return HookResult(payload={"messages": new_messages})

    # ------------------------------------------------------------------
    # 注入
    # ------------------------------------------------------------------

    def _inject(self, messages: List[Dict[str, Any]], skills: List[Any]) -> List[Dict[str, Any]]:
        """按 inject_mode 注入技能文本。"""
        block = self._format_block(skills)
        if not block:
            return list(messages or [])

        if self._inject_mode == "system":
            return self._inject_system(messages, block)
        return self._inject_append(messages, block)

    @staticmethod
    def _format_block(skills: List[Any]) -> str:
        """把选中技能格式化为一段注入文本。"""
        parts = []
        for skill in skills:
            name = getattr(skill, "name", "") or ""
            text = getattr(skill, "text", "") or ""
            parts.append(f"【技能：{name}】\n{text}")
        return "\n\n".join(parts)

    def _inject_append(self, messages: List[Dict[str, Any]], block: str) -> List[Dict[str, Any]]:
        """追加为独立 user 消息。"""
        new_messages = list(messages or [])
        new_messages.append({"role": "user", "content": block})
        return new_messages

    def _inject_system(self, messages: List[Dict[str, Any]], block: str) -> List[Dict[str, Any]]:
        """把技能文本拼接到 system prompt 末尾。"""
        new_messages = list(messages or [])
        if new_messages and new_messages[0].get("role") == "system":
            original = new_messages[0].get("content", "")
            new_messages[0] = {
                "role": "system",
                "content": f"{original}\n\n{block}" if original else block,
            }
        else:
            new_messages.insert(0, {"role": "system", "content": block})
        return new_messages

    # ------------------------------------------------------------------
    # 依赖解析
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_skill_system(ctx: Context) -> Any:
        """从 ctx.shared 解析 SkillSystem，未启用返回 None。"""
        shared = getattr(ctx, "shared", None)
        if not isinstance(shared, dict):
            return None
        return shared.get("skill_system")

    @staticmethod
    def _resolve_task(ctx: Context) -> str:
        """从 session.metadata 解析任务描述，取不到返回空串。"""
        shared = getattr(ctx, "shared", None)
        if not isinstance(shared, dict):
            return ""
        session = shared.get("session")
        if session is None:
            return ""
        metadata = getattr(session, "metadata", None)
        if isinstance(metadata, dict):
            task = metadata.get("task", "")
            return str(task or "")
        return ""


__all__ = ["SkillInjectionMiddleware"]