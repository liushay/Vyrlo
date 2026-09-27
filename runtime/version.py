"""
[批次 D3] runtime/version —— 语义化版本与 schema 版本管理。

提供：
    - ``__version__``  : 当前发行版的语义化版本（MAJOR.MINOR.PATCH）。
    - ``SemVer``       : 可解析、可比较的语义化版本值对象。
    - ``SCHEMA_VERSION`` : 配置 schema 的当前版本（用于触发迁移）。

约束：
    - 纯标准库实现，无第三方依赖。
    - 版本比较遵循 SemVer 2.0 的优先级规则（PATCH > MINOR > MAJOR）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple

#: 当前发行版语义化版本。发布时请同步更新（遵守语义化版本规则）。
__version__ = "1.0.0"

#: 配置 schema 当前版本号。配置文件中的 ``schema_version`` 低于此值时，
#: 由 ``runtime.config.migrations`` 依序执行迁移脚本。
SCHEMA_VERSION = "1.0.0"

_SEMVER_RE = re.compile(
    r"^[vV]?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


@dataclass(frozen=True)
class SemVer:
    """语义化版本值对象（不可变，可比较、可哈希）。"""

    major: int
    minor: int
    patch: int
    prerelease: Tuple[str, ...] = ()

    @classmethod
    def parse(cls, text: str) -> "SemVer":
        """解析语义化版本字符串。

        Raises:
            ValueError: 版本号不合法时。
        """
        m = _SEMVER_RE.match(str(text or "").strip())
        if not m:
            raise ValueError(f"不是合法的语义化版本: {text!r}")
        pre = m.group(4)
        return cls(
            major=int(m.group(1)),
            minor=int(m.group(2)),
            patch=int(m.group(3)),
            prerelease=tuple(pre.split(".")) if pre else (),
        )

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            base += "-" + ".".join(self.prerelease)
        return base

    def _key(self) -> Tuple[int, int, int, Tuple]:
        # 正式版优先于预发布版：无 prerelease 时用 (1,) 占位，确保 1.0.0 > 1.0.0-rc1
        pre = (1,) if not self.prerelease else (0,) + tuple(
            _int_or_str(p) for p in self.prerelease
        )
        return (self.major, self.minor, self.patch, pre)

    def __lt__(self, other: "SemVer") -> bool:
        return self._key() < other._key()

    def __le__(self, other: "SemVer") -> bool:
        return self._key() <= other._key()

    def __gt__(self, other: "SemVer") -> bool:
        return self._key() > other._key()

    def __ge__(self, other: "SemVer") -> bool:
        return self._key() >= other._key()


def _int_or_str(value: str):
    """预发布标识按 SemVer 规则比较：纯数字按数值，否则按字符串。"""
    if value.isdigit():
        return int(value)
    return value


def current_version() -> SemVer:
    """返回当前发行版的 ``SemVer`` 对象。"""
    return SemVer.parse(__version__)


def parse_version(text: str) -> SemVer:
    """解析任意版本字符串（``current_version`` 的别名，便于外部调用）。"""
    return SemVer.parse(text)


__all__ = [
    "__version__",
    "SCHEMA_VERSION",
    "SemVer",
    "current_version",
    "parse_version",
]