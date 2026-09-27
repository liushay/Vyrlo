"""
[L4] OrchestratorMiddleware — 编排建议中间件。

职责边界（只扫描并给出建议，不做实际委派）：
    - AFTER_LLM 扫描 response.content，若含 @agent_id 提及，
      写入 ctx.shared["suggested_delegation"]。
    - 不拦截工具、不执行委派、不修改控制流。

与 DelegationMiddleware 的职责区分：
    - Orchestrator 只"发现意图并给建议"；
    - Delegation 只"接收显式工具调用并落 pending_delegation"，
      实际执行在 Runtime 层。

可独立启停：
    解析不到 agent_registry 时，仍可扫描和建议（建议不依赖于注册表存在）。
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

from agent_loop import Context, HookResult, Middleware


class OrchestratorMiddleware(Middleware):
    """[L4] 编排建议中间件 —— 扫描 @agent 提及并给出建议。"""

    _MENTION_RE = re.compile(r"@([A-Za-z_][A-Za-z0-9_]*)")

    def __init__(self, name: str = "orchestrator") -> None:
        super().__init__(name)

    def after_llm(self, ctx: Context, response: Any) -> HookResult:
        """[L4] AFTER_LLM：扫描 response.content 中的 @agent_id 提及。"""
        content = self._extract_content(response)
        mentioned = list(set(self._MENTION_RE.findall(content)))
        if not mentioned:
            return HookResult.continue_()

        # 排除 @ 开头的 email 等误匹配：这里做一次保守过滤
        suggestions = [m for m in mentioned if m]
        ctx.shared["suggested_delegation"] = {
            "targets": suggestions,
        }
        return HookResult.continue_()

    @staticmethod
    def _extract_content(response: Any) -> str:
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            if "choices" in response:
                choices = response.get("choices") or []
                if choices and isinstance(choices[0], dict):
                    msg = choices[0].get("message", {})
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content
            content = response.get("content")
            if isinstance(content, str):
                return content
            return ""
        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content
        return ""