"""
[L3] DockerSandboxProvider — Docker 沙箱提供者（接口预留）。

未安装 docker 时 is_available() 返回 False，不阻断装配。此处仅实现
接口契约、可用性探测与路径翻译的 Windows → WSL2 占位逻辑，execute 预留。

约束：不引入新依赖。
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any, Dict

from runtime.sandbox.interface import SandboxHandle, SandboxProvider
from runtime.tool_registry.interface import ToolResult


class DockerSandboxProvider(SandboxProvider):
    """Docker 沙箱提供者（接口预留）。

    说明：
        - acquire：创建或复用容器（预留，返回占位 handle）。
        - execute：通过 docker exec 调用（预留，返回友好错误）。
        - translate_path：Windows → WSL2 路径翻译（占位实现）。
    """

    IMAGE = "sandbox-runtime:latest"

    def is_available(self) -> bool:
        """检测 docker CLI 是否可用。"""
        return shutil.which("docker") is not None

    def acquire(self, agent_id: str) -> SandboxHandle:
        """创建或复用容器（预留）。"""
        # 预留：真实实现应 `docker run -d` 并记录容器 ID。
        return SandboxHandle(
            sandbox_id=f"docker-{agent_id}",
            workdir="/workspace",
            env={},
            metadata={"provider": "docker", "image": self.IMAGE},
        )

    def release(self, handle: SandboxHandle) -> None:
        """释放容器（预留）。"""
        # 预留：真实实现应 `docker rm -f {container_id}`。

    def execute(
        self, handle: SandboxHandle, tool_name: str, args: Dict[str, Any]
    ) -> ToolResult:
        """通过 docker exec 调用（预留）。"""
        # 预留：真实实现应组装 `docker exec ...` 并解析输出。
        return ToolResult.from_error(
            f"Docker 沙箱 execute 尚未实现（tool={tool_name!r}）"
        )

    def translate_path(self, handle: SandboxHandle, path: str) -> str:
        """Windows → WSL2 路径翻译（占位实现）。

        约定：把 Windows 盘符路径 ``C:\\...`` 翻译为 ``/mnt/c/...``；
        无法翻译时原样返回。
        """
        p = str(path or "")
        if len(p) >= 2 and p[1] == ":":
            drive = p[0].lower()
            rest = p[2:].replace("\\", "/")
            return f"/mnt/{drive}{rest}"
        return p.replace("\\", "/")