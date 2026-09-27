"""
[B1/D003] SkillSystem 注入语义 —— 空 store 不得被替换。

背景（DEFECTS.md D003）：
    修复前 ``SkillSystem.__init__`` 用 ``store or SkillStore()`` 判空。
    ``SkillStore`` 定义了 ``__len__``（``SELECT COUNT(*)``），空库时返回 0
    → 空 store 为 falsy → 调用方显式注入的空 store 被**替换**为内部新 store，
    后续对原 store 的写入全部丢失。

修复后：改用 ``store if store is not None else SkillStore()``。

本文件覆盖：
    1. test_empty_store_not_replaced
    2. test_writes_land_on_injected_store

运行方式：
    python -m pytest tests/test_skill_system_injection.py -v
"""

from __future__ import annotations

from runtime.skill_system import SkillSystem
from runtime.skill_system.store import SkillStore


# ============================================================================
# 1. 空 store 不被替换
# ============================================================================


def test_empty_store_not_replaced():
    """用空 SkillStore 构造 SkillSystem → 必须使用调用方传入的实例。"""
    injected = SkillStore()

    # 前置条件：空 store 的 __len__ 为 0（即 falsy，这正是缺陷触发条件）
    assert len(injected) == 0
    assert not injected

    system = SkillSystem(store=injected)

    # 修复点：门面持有的就是注入的实例本身，而非内部新建的 store
    assert system.store is injected
    # ratchet 也应绑定到同一个 store（默认构造时传入 self.store）
    assert system.ratchet._store is injected


# ============================================================================
# 2. 写入落到注入的 store 上
# ============================================================================


def test_writes_land_on_injected_store():
    """通过门面写入的技能必须落在注入的 store 上，而非被丢弃到内部新 store。"""
    injected = SkillStore()
    system = SkillSystem(store=injected)

    skill = system.store.create(
        "demo",
        "演示技能",
        "步骤：先读取输入，再输出结果。",
    )

    # 注入的 store 里能查到刚创建的技能
    assert len(injected) == 1
    assert injected.get(skill.skill_id) is not None
    # 门面视角与注入 store 视角一致
    assert [s.skill_id for s in system.list_skills()] == [skill.skill_id]

    # 计数写入也落在注入的 store 上
    system.observe_applied(skill.skill_id, applied=True)
    counters = injected.get_counters(skill.skill_id)
    assert counters.applied == 1


# ============================================================================
# 3. 同类问题：显式传入空列表不应退化为全量检查
# ============================================================================


def test_check_metrics_with_explicit_empty_list():
    """[D003 同类] check_metrics([]) 应检查空集，而非退化为全量检查。"""
    injected = SkillStore()
    system = SkillSystem(store=injected)

    # 制造一个待检查的技能
    system.store.create("s1", "技能一", "步骤：输出。")

    # 显式传空列表 → 应检查 0 个（而不是回退到 store.list() 全量）
    assert system.check_metrics([]) == []

    # 不传（None）→ 才使用全量
    assert len(system.check_metrics()) == 1