"""阶段N测试：验证依赖关系显式化。

验证内容：
1. _dependencies.py 声明的依赖图无循环
2. 拓扑排序有效
3. 实际源码中的 import 与声明的依赖一致（核心验证）
4. 不存在未声明的依赖（声明缺失）
5. 不存在多余的声明（声明冗余）
6. 传递依赖计算正确
"""

import re
from pathlib import Path

import pytest

from agent_whitepaper._dependencies import (
    STAGE_DEPENDENCIES,
    TRANSITIVE_DEPENDENCIES,
    get_dependents,
    get_transitive_dependencies,
    has_cycle,
    topological_order,
)

# ---------------------------------------------------------------------------
# 扫描实际 import 语句，构建实际依赖图
# ---------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# 匹配: from agent_whitepaper.stage_x.module import ...
# 或:   from agent_whitepaper.demos.module import ...
IMPORT_RE = re.compile(
    r"^\s*from\s+agent_whitepaper\.(stage_[a-k]|demos)\b",
    re.MULTILINE,
)

# 各阶段对应的目录名
STAGE_DIRS = {
    "stage_a": "stage_a",
    "stage_b": "stage_b",
    "stage_c": "stage_c",
    "stage_d": "stage_d",
    "stage_e": "stage_e",
    "stage_f": "stage_f",
    "stage_g": "stage_g",
    "stage_h": "stage_h",
    "stage_i": "stage_i",
    "stage_j": "stage_j",
    "stage_k": "stage_k",
    "demos": "demos",
}


def _scan_actual_dependencies() -> dict:
    """扫描源码，返回实际依赖图 {stage: set(deps)}。"""
    actual: dict = {}
    for stage, dirname in STAGE_DIRS.items():
        stage_dir = PACKAGE_ROOT / dirname
        deps = set()
        for py_file in stage_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
            text = py_file.read_text(encoding="utf-8")
            for match in IMPORT_RE.finditer(text):
                dep = match.group(1)
                if dep != stage:  # 排除自引用
                    deps.add(dep)
        actual[stage] = deps
    return actual


# ---------------------------------------------------------------------------
# 依赖图结构验证
# ---------------------------------------------------------------------------

class TestDependencyGraphStructure:
    """验证依赖图本身的结构正确性。"""

    def test_no_cycle(self):
        assert not has_cycle(), "依赖图存在循环依赖"

    def test_topological_order_complete(self):
        """拓扑排序应包含所有阶段。"""
        order = topological_order()
        assert set(order) == set(STAGE_DEPENDENCIES.keys())

    def test_topological_order_valid(self):
        """拓扑排序中，每个阶段的依赖必须出现在它之前。"""
        order = topological_order()
        position = {stage: i for i, stage in enumerate(order)}
        for stage, deps in STAGE_DEPENDENCIES.items():
            for dep in deps:
                assert position[dep] < position[stage], (
                    f"拓扑序无效: {dep} 应在 {stage} 之前"
                )

    def test_all_stages_declared(self):
        """所有已知阶段都应在依赖图中声明。"""
        expected_stages = set(STAGE_DIRS.keys())
        assert set(STAGE_DEPENDENCIES.keys()) == expected_stages

    def test_dependencies_are_known_stages(self):
        """声明的依赖必须是已知阶段。"""
        known = set(STAGE_DEPENDENCIES.keys())
        for stage, deps in STAGE_DEPENDENCIES.items():
            for dep in deps:
                assert dep in known, (
                    f"{stage} 声明了未知依赖: {dep}"
                )

    def test_no_self_dependency(self):
        """阶段不应依赖自身。"""
        for stage, deps in STAGE_DEPENDENCIES.items():
            assert stage not in deps, f"{stage} 依赖了自身"


# ---------------------------------------------------------------------------
# 核心：声明 vs 实际一致性验证
# ---------------------------------------------------------------------------

class TestDeclaredVsActual:
    """验证声明的依赖与实际 import 语句一致。"""

    @pytest.fixture(scope="class")
    def actual_deps(self):
        return _scan_actual_dependencies()

    def test_no_undeclared_dependencies(self, actual_deps):
        """实际 import 中不应存在未声明的依赖。

        即: actual ⊆ declared
        """
        for stage, actual in actual_deps.items():
            declared = set(STAGE_DEPENDENCIES[stage])
            undeclared = actual - declared
            assert not undeclared, (
                f"{stage} 实际导入了未声明的依赖: {undeclared}"
            )

    def test_no_redundant_declarations(self, actual_deps):
        """声明中不应存在实际未使用的依赖（允许少量冗余但应报告）。

        即: declared ⊆ actual （严格模式）
        """
        for stage, actual in actual_deps.items():
            declared = set(STAGE_DEPENDENCIES[stage])
            redundant = declared - actual
            assert not redundant, (
                f"{stage} 声明了但实际未导入的冗余依赖: {redundant}"
            )

    def test_dependency_graph_matches(self, actual_deps):
        """声明图与实际图完全一致。"""
        for stage in STAGE_DEPENDENCIES:
            declared = set(STAGE_DEPENDENCIES[stage])
            actual = actual_deps[stage]
            assert declared == actual, (
                f"{stage} 依赖不一致:\n"
                f"  声明: {sorted(declared)}\n"
                f"  实际: {sorted(actual)}\n"
                f"  未声明: {sorted(actual - declared)}\n"
                f"  冗余: {sorted(declared - actual)}"
            )


# ---------------------------------------------------------------------------
# 传递依赖验证
# ---------------------------------------------------------------------------

class TestTransitiveDependencies:
    """验证传递依赖计算。"""

    def test_stage_a_no_transitive(self):
        """基础层 stage_a 无传递依赖。"""
        assert get_transitive_dependencies("stage_a") == set()

    def test_stage_j_no_transitive(self):
        """独立安全层 stage_j 无传递依赖。"""
        assert get_transitive_dependencies("stage_j") == set()

    def test_stage_g_all_below(self):
        """集成层 stage_g 传递依赖包含所有底层阶段。"""
        trans = get_transitive_dependencies("stage_g")
        assert trans == {
            "stage_a", "stage_b", "stage_c",
            "stage_d", "stage_e", "stage_f",
        }

    def test_stage_k_transitive(self):
        """stage_k 传递依赖包含 stage_g 及其所有下层依赖。"""
        trans = get_transitive_dependencies("stage_k")
        assert "stage_g" in trans
        assert "stage_a" in trans  # 通过 stage_g 传递
        assert "stage_h" not in trans  # stage_k 不直接或间接依赖 stage_h

    def test_transitive_excludes_self(self):
        """传递依赖不应包含自身。"""
        for stage in STAGE_DEPENDENCIES:
            trans = get_transitive_dependencies(stage)
            assert stage not in trans

    def test_transitive_cache_consistent(self):
        """预计算的 TRANSITIVE_DEPENDENCIES 与实时计算一致。"""
        for stage in STAGE_DEPENDENCIES:
            assert TRANSITIVE_DEPENDENCIES[stage] == \
                get_transitive_dependencies(stage)


# ---------------------------------------------------------------------------
# 反向依赖验证
# ---------------------------------------------------------------------------

class TestReverseDependencies:
    """验证反向依赖（dependents）计算。"""

    def test_stage_a_has_many_dependents(self):
        """stage_a 被多个阶段直接依赖。"""
        deps = get_dependents("stage_a")
        assert "stage_b" in deps
        assert "stage_d" in deps
        assert "stage_e" in deps
        assert "stage_i" in deps

    def test_stage_j_no_dependents(self):
        """stage_j 不被任何阶段直接依赖。"""
        assert get_dependents("stage_j") == []

    def test_reverse_consistency(self):
        """反向依赖与正向依赖一致。"""
        for stage in STAGE_DEPENDENCIES:
            for dep in STAGE_DEPENDENCIES[stage]:
                assert stage in get_dependents(dep), (
                    f"{stage} 依赖 {dep}，但 {dep} 的 dependents 不含 {stage}"
                )


# ---------------------------------------------------------------------------
# 层次结构验证
# ---------------------------------------------------------------------------

class TestLayerStructure:
    """验证依赖层次结构符合设计。"""

    def test_base_layer_independent(self):
        """基础层（stage_a, stage_j）不依赖任何阶段。"""
        assert STAGE_DEPENDENCIES["stage_a"] == []
        assert STAGE_DEPENDENCIES["stage_j"] == []

    def test_stage_b_only_depends_on_a(self):
        assert STAGE_DEPENDENCIES["stage_b"] == ["stage_a"]

    def test_stage_g_depends_on_all_below(self):
        """集成层 stage_g 依赖 stage_a ~ stage_f。"""
        assert set(STAGE_DEPENDENCIES["stage_g"]) == {
            "stage_a", "stage_b", "stage_c",
            "stage_d", "stage_e", "stage_f",
        }

    def test_stage_h_depends_on_a_g_and_d(self):
        """stage_h 依赖 stage_a（便捷API）、stage_d（持久化）、stage_g（编排）。"""
        assert set(STAGE_DEPENDENCIES["stage_h"]) == {
            "stage_a", "stage_d", "stage_g"
        }

    def test_stage_k_depends_on_g(self):
        assert set(STAGE_DEPENDENCIES["stage_k"]) == {"stage_g"}
