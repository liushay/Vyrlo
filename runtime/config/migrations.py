"""
[批次 D3] runtime.config.migrations —— 配置 schema 迁移脚本注册表。

约定：
    - 每个迁移函数命名为 ``migrate_<from_version>``，其中 ``<from_version>``
      使用下划线代替点号（例如 ``migrate_1_0_0`` 对应从 ``1.0.0`` 迁移）。
    - 函数就地修改传入的 ``config`` 字典（返回可选的新字典，返回 None 视为
      原地修改），并把 ``config["schema_version"]`` 提升到目标版本。
    - 迁移必须幂等：重复执行不会破坏数据。

当前 schema 为 ``1.0.0``（无历史版本），因此本模块暂无迁移函数。
新增迁移时，在此处追加函数即可被 ``runtime.config.loader`` 自动发现并
按版本号升序执行。
"""

from __future__ import annotations

from typing import Any, Dict

__all__: list = []