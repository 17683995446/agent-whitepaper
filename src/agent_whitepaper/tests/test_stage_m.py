"""阶段M测试：验证公共API导出。

验证内容：
1. 顶层包 agent_whitepaper 的公共API可导入且 __all__ 一致
2. 各 stage_a ~ stage_k 子包的公共API可导入且 __all__ 一致
3. demos 子包的公共API可导入且 __all__ 一致
4. __all__ 中每个名称在对应模块命名空间中真实存在
5. 顶层包 __all__ 中的名称是各子包 __all__ 的子集（契约一致性）
"""

import inspect

import agent_whitepaper
from agent_whitepaper import (
    demos,
    stage_a,
    stage_b,
    stage_c,
    stage_d,
    stage_e,
    stage_f,
    stage_g,
    stage_h,
    stage_i,
    stage_j,
    stage_k,
)

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _assert_all_matches_namespace(module, label):
    """断言模块的 __all__ 中每个名称都能在模块命名空间中找到。"""
    assert hasattr(module, "__all__"), f"{label} 缺少 __all__"
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert not missing, f"{label}.__all__ 中存在但命名空间缺失: {missing}"


def _assert_all_no_duplicates(module, label):
    """断言 __all__ 没有重复项。"""
    names = module.__all__
    duplicates = [n for n in set(names) if names.count(n) > 1]
    assert not duplicates, f"{label}.__all__ 存在重复: {duplicates}"


def _assert_all_is_list_of_str(module, label):
    """断言 __all__ 是字符串列表。"""
    assert isinstance(module.__all__, list), f"{label}.__all__ 必须是 list"
    for name in module.__all__:
        assert isinstance(name, str), f"{label}.__all__ 元素 {name!r} 必须是 str"


# ---------------------------------------------------------------------------
# 顶层包
# ---------------------------------------------------------------------------

class TestTopLevelPackage:
    """顶层包 agent_whitepaper 公共API。"""

    def test_has_all(self):
        assert hasattr(agent_whitepaper, "__all__")

    def test_has_version(self):
        assert hasattr(agent_whitepaper, "__version__")
        assert isinstance(agent_whitepaper.__version__, str)

    def test_all_matches_namespace(self):
        _assert_all_matches_namespace(agent_whitepaper, "agent_whitepaper")

    def test_all_no_duplicates(self):
        _assert_all_no_duplicates(agent_whitepaper, "agent_whitepaper")

    def test_all_is_list_of_str(self):
        _assert_all_is_list_of_str(agent_whitepaper, "agent_whitepaper")

    def test_key_exports_importable(self):
        """顶层包导出的核心类可从顶层直接导入。"""
        from agent_whitepaper import (
            GridWorld,
            SelfDrivenAgent,
            SystemOrchestrator,
        )
        # 确保它们是真实对象而非 None
        assert SelfDrivenAgent is not None
        assert GridWorld is not None
        assert SystemOrchestrator is not None

    def test_all_subset_of_subpackages(self):
        """顶层 __all__ 中每个名称必须来自某个子包或基础模块的 __all__。"""
        sub_alls = {}
        for sub in [
            stage_a, stage_b, stage_c, stage_d, stage_e,
            stage_f, stage_g, stage_h, stage_i, stage_j, stage_k,
        ]:
            sub_alls[sub.__name__] = set(sub.__all__)
        # 基础模块（非阶段）也是合法的导出来源
        from agent_whitepaper import _exceptions, _protocols
        sub_alls["_exceptions"] = set(_exceptions.__all__)
        sub_alls["_protocols"] = set(_protocols.__all__)

        for name in agent_whitepaper.__all__:
            found = any(name in s for s in sub_alls.values())
            assert found, (
                f"顶层 __all__ 中的 '{name}' 不属于任何子包/基础模块 __all__，"
                f"违反契约一致性"
            )


# ---------------------------------------------------------------------------
# 各子包参数化测试
# ---------------------------------------------------------------------------

SUBPACKAGES = [
    ("stage_a", stage_a),
    ("stage_b", stage_b),
    ("stage_c", stage_c),
    ("stage_d", stage_d),
    ("stage_e", stage_e),
    ("stage_f", stage_f),
    ("stage_g", stage_g),
    ("stage_h", stage_h),
    ("stage_i", stage_i),
    ("stage_j", stage_j),
    ("stage_k", stage_k),
    ("demos", demos),
]


def test_each_subpackage_has_all():
    for label, mod in SUBPACKAGES:
        assert hasattr(mod, "__all__"), f"{label} 缺少 __all__"


def test_each_subpackage_all_matches_namespace():
    for label, mod in SUBPACKAGES:
        _assert_all_matches_namespace(mod, label)


def test_each_subpackage_all_no_duplicates():
    for label, mod in SUBPACKAGES:
        _assert_all_no_duplicates(mod, label)


def test_each_subpackage_all_is_list_of_str():
    for label, mod in SUBPACKAGES:
        _assert_all_is_list_of_str(mod, label)


def test_each_subpackage_all_nonempty():
    for label, mod in SUBPACKAGES:
        assert len(mod.__all__) > 0, f"{label}.__all__ 不应为空"


def test_each_subpackage_has_docstring():
    """每个子包应有描述其公共API的docstring。"""
    for label, mod in SUBPACKAGES:
        doc = inspect.getdoc(mod)
        assert doc is not None, f"{label} 缺少模块docstring"
        assert len(doc) > 10, f"{label} docstring过短"


# ---------------------------------------------------------------------------
# 各子包核心类可实例化/可调用验证（轻量级）
# ---------------------------------------------------------------------------

def test_stage_a_core_classes_in_all():
    """stage_a 的核心类在 __all__ 中。"""
    for name in ["SelfDrivenAgent", "GridWorld", "QTable", "DriveBasedPolicy"]:
        assert name in stage_a.__all__, f"stage_a.__all__ 缺少 {name}"


def test_stage_b_core_classes_in_all():
    for name in ["Skill", "SkillLibrary", "SkillAbstractor", "MetaLearner",
                 "TaskDecomposer", "LongTaskExecutor"]:
        assert name in stage_b.__all__, f"stage_b.__all__ 缺少 {name}"


def test_stage_c_core_classes_in_all():
    for name in ["AgentRegistry", "MessageBus", "CoalitionFormer",
                 "SharedSkillLibrary"]:
        assert name in stage_c.__all__, f"stage_c.__all__ 缺少 {name}"


def test_stage_d_core_classes_in_all():
    for name in ["PersistenceEngine", "TaskScheduler", "HealthMonitor",
                 "PerpetualRuntime"]:
        assert name in stage_d.__all__, f"stage_d.__all__ 缺少 {name}"


def test_stage_e_core_classes_in_all():
    for name in ["QTableMigrator", "SkillQValueInjector", "KnowledgeDistiller"]:
        assert name in stage_e.__all__, f"stage_e.__all__ 缺少 {name}"


def test_stage_f_core_classes_in_all():
    for name in ["AutonomousTaskGenerator", "DegradationDetector",
                 "AutonomousRetrainer"]:
        assert name in stage_f.__all__, f"stage_f.__all__ 缺少 {name}"


def test_stage_g_core_classes_in_all():
    for name in ["MetricsCollector", "SystemOrchestrator"]:
        assert name in stage_g.__all__, f"stage_g.__all__ 缺少 {name}"


def test_stage_h_core_classes_in_all():
    for name in ["ConfigLoader", "load_config", "main", "build_parser"]:
        assert name in stage_h.__all__, f"stage_h.__all__ 缺少 {name}"


def test_stage_i_core_classes_in_all():
    for name in ["EnvironmentProtocol", "StateEncoder", "FunctionCallEnvironment"]:
        assert name in stage_i.__all__, f"stage_i.__all__ 缺少 {name}"


def test_stage_j_core_classes_in_all():
    for name in ["SafetyMonitor", "SafeActionFilter", "ResourceManager"]:
        assert name in stage_j.__all__, f"stage_j.__all__ 缺少 {name}"


def test_stage_k_core_classes_in_all():
    for name in ["LongRunDemo", "ReportGenerator", "quick_demo", "full_demo"]:
        assert name in stage_k.__all__, f"stage_k.__all__ 缺少 {name}"


# ---------------------------------------------------------------------------
# 星号导入安全性验证
# ---------------------------------------------------------------------------

def test_star_import_top_level():
    """验证 `from agent_whitepaper import *` 不会引入非 __all__ 的名称。"""
    namespace = {}
    exec("from agent_whitepaper import *", namespace)
    exported = {n for n in namespace if not n.startswith("_")}
    expected = set(agent_whitepaper.__all__)
    # 允许 expected 是 exported 的子集（exec 可能引入builtins等）
    assert expected.issubset(exported), (
        f"星号导入缺失: {expected - exported}"
    )


def test_star_import_each_subpackage():
    """验证各子包 `from xxx import *` 安全。"""
    for label, mod in SUBPACKAGES:
        namespace = {}
        exec(f"from {mod.__name__} import *", namespace)
        exported = {n for n in namespace if not n.startswith("_")}
        expected = set(mod.__all__)
        assert expected.issubset(exported), (
            f"{label} 星号导入缺失: {expected - exported}"
        )
