"""
[L3] runtime.sandbox — 沙箱执行抽象包。

公开：
    - SandboxHandle / SandboxProvider（接口与数据模型）
    - LocalSandboxProvider（本地受控 workdir 执行）
    - DockerSandboxProvider（接口预留，未安装 docker 时不可用）
    - SandboxRegistry（provider 注册表，按 mode 选择 acquire）
"""

from runtime.sandbox.interface import SandboxHandle, SandboxProvider
from runtime.sandbox.local_provider import LocalSandboxProvider
from runtime.sandbox.docker_provider import DockerSandboxProvider
from runtime.sandbox.registry import SandboxRegistry

__all__ = [
    "SandboxHandle",
    "SandboxProvider",
    "LocalSandboxProvider",
    "DockerSandboxProvider",
    "SandboxRegistry",
]