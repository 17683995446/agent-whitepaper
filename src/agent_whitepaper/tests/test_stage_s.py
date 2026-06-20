"""阶段S测试：验证模块化接口抽象。

验证内容：
1. _protocols.py 定义了核心抽象接口
2. 具体类正确实现接口（isinstance 关系）
3. 接口方法都是 abstractmethod
4. Environment 升级为正式 ABC
5. 接口在顶层包导出
6. _protocols.py 无循环导入
"""

import inspect
from abc import ABC
from pathlib import Path

import pytest

from agent_whitepaper import (
    AgentProtocol,
    MemoryProtocol,
    PolicyProtocol,
    _protocols,
)
from agent_whitepaper.stage_a.agent import SelfDrivenAgent
from agent_whitepaper.stage_a.environment import Environment, GridWorld
from agent_whitepaper.stage_a.memory import EpisodicBuffer
from agent_whitepaper.stage_a.policy import QTable
from agent_whitepaper.stage_i.environment_interface import (
    EnvironmentProtocol,
    StateEncoder,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PACKAGE_ROOT


# ---------------------------------------------------------------------------
# 接口定义验证
# ---------------------------------------------------------------------------
class TestProtocolDefinitions:
    """验证抽象接口定义。"""

    def test_protocols_are_abc(self):
        """所有协议都是 ABC 子类。"""
        for proto in [PolicyProtocol, MemoryProtocol, AgentProtocol]:
            assert issubclass(proto, ABC), f"{proto.__name__} 不是 ABC"

    def test_policy_protocol_abstract_methods(self):
        """PolicyProtocol 有正确的抽象方法。"""
        abstract_methods = PolicyProtocol.__abstractmethods__
        assert "get" in abstract_methods
        assert "best_action" in abstract_methods
        assert "update" in abstract_methods

    def test_memory_protocol_abstract_methods(self):
        """MemoryProtocol 有正确的抽象方法。"""
        abstract_methods = MemoryProtocol.__abstractmethods__
        assert "add" in abstract_methods
        assert "sample" in abstract_methods
        assert "get_recent_episodes" in abstract_methods
        assert "__len__" in abstract_methods

    def test_agent_protocol_abstract_methods(self):
        """AgentProtocol 有正确的抽象方法。"""
        abstract_methods = AgentProtocol.__abstractmethods__
        assert "perceive_and_act" in abstract_methods
        assert "run_episode" in abstract_methods

    def test_protocol_methods_are_abstractmethod(self):
        """协议方法都用 @abstractmethod 装饰。"""
        for proto in [PolicyProtocol, MemoryProtocol, AgentProtocol]:
            for name, method in inspect.getmembers(proto, predicate=inspect.isfunction):
                if name.startswith("_") and name != "__len__":
                    continue
                # 检查是否是抽象方法
                assert getattr(method, "__isabstractmethod__", False), (
                    f"{proto.__name__}.{name} 不是 @abstractmethod"
                )

    def test_cannot_instantiate_protocols(self):
        """抽象接口不能被实例化。"""
        for proto in [PolicyProtocol, MemoryProtocol, AgentProtocol]:
            with pytest.raises(TypeError):
                proto()

    def test_module_has_all(self):
        """模块有 __all__ 声明。"""
        assert hasattr(_protocols, "__all__")
        assert "PolicyProtocol" in _protocols.__all__
        assert "MemoryProtocol" in _protocols.__all__
        assert "AgentProtocol" in _protocols.__all__

    def test_module_has_docstring(self):
        """模块有 docstring 说明第一性原理。"""
        assert _protocols.__doc__ is not None
        assert "第一性原理" in _protocols.__doc__


# ---------------------------------------------------------------------------
# 具体类实现验证
# ---------------------------------------------------------------------------
class TestConcreteImplementations:
    """验证具体类正确实现接口。"""

    def test_qtable_implements_policy_protocol(self):
        """QTable 实现 PolicyProtocol。"""
        assert issubclass(QTable, PolicyProtocol)

    def test_episodic_buffer_implements_memory_protocol(self):
        """EpisodicBuffer 实现 MemoryProtocol。"""
        assert issubclass(EpisodicBuffer, MemoryProtocol)

    def test_self_driven_agent_implements_agent_protocol(self):
        """SelfDrivenAgent 实现 AgentProtocol。"""
        assert issubclass(SelfDrivenAgent, AgentProtocol)

    def test_qtable_isinstance_policy_protocol(self):
        """QTable 实例是 PolicyProtocol 的实例。"""
        qt = QTable(n_states=10, n_actions=4)
        assert isinstance(qt, PolicyProtocol)

    def test_episodic_buffer_isinstance_memory_protocol(self):
        """EpisodicBuffer 实例是 MemoryProtocol 的实例。"""
        buf = EpisodicBuffer(capacity=100)
        assert isinstance(buf, MemoryProtocol)

    def test_qtable_no_abstract_methods_remaining(self):
        """QTable 实现了所有抽象方法（无剩余抽象方法）。"""
        assert len(QTable.__abstractmethods__) == 0

    def test_episodic_buffer_no_abstract_methods_remaining(self):
        """EpisodicBuffer 实现了所有抽象方法。"""
        assert len(EpisodicBuffer.__abstractmethods__) == 0

    def test_self_driven_agent_no_abstract_methods_remaining(self):
        """SelfDrivenAgent 实现了所有抽象方法。"""
        assert len(SelfDrivenAgent.__abstractmethods__) == 0


# ---------------------------------------------------------------------------
# Environment ABC 升级验证
# ---------------------------------------------------------------------------
class TestEnvironmentABC:
    """验证 Environment 升级为正式 ABC。"""

    def test_environment_is_abc(self):
        """Environment 是 ABC 子类。"""
        assert issubclass(Environment, ABC)

    def test_environment_has_abstract_methods(self):
        """Environment 有抽象方法。"""
        abstract_methods = Environment.__abstractmethods__
        assert "reset" in abstract_methods
        assert "step" in abstract_methods
        assert "action_space" in abstract_methods
        assert "observation_dim" in abstract_methods

    def test_cannot_instantiate_environment(self):
        """Environment 不能被直接实例化。"""
        with pytest.raises(TypeError):
            Environment()

    def test_gridworld_implements_environment(self):
        """GridWorld 实现 Environment。"""
        assert issubclass(GridWorld, Environment)
        assert len(GridWorld.__abstractmethods__) == 0

    def test_gridworld_isinstance_environment(self):
        """GridWorld 实例是 Environment 的实例。"""
        env = GridWorld(size=5)
        assert isinstance(env, Environment)

    def test_environment_no_raise_not_implemented(self):
        """Environment 方法不再使用 raise NotImplementedError。"""
        env_file = SRC_ROOT / "stage_a" / "environment.py"
        content = env_file.read_text(encoding="utf-8")
        # Environment 类定义部分不应有 raise NotImplementedError
        #（GridWorld 的 step 中有 raise ValueError，这是合法的）
        lines = content.splitlines()
        in_environment_class = False
        for i, line in enumerate(lines):
            if line.startswith("class Environment"):
                in_environment_class = True
            elif line.startswith("class ") and in_environment_class:
                break
            elif in_environment_class and "raise NotImplementedError" in line:
                pytest.fail(
                    f"Environment 类中仍有 raise NotImplementedError: line {i+1}"
                )


# ---------------------------------------------------------------------------
# stage_i 接口验证（已有接口）
# ---------------------------------------------------------------------------
class TestStageIInterfaces:
    """验证 stage_i 已有的抽象接口。"""

    def test_environment_protocol_is_abc(self):
        """EnvironmentProtocol 是 ABC。"""
        assert issubclass(EnvironmentProtocol, ABC)

    def test_state_encoder_is_abc(self):
        """StateEncoder 是 ABC。"""
        assert issubclass(StateEncoder, ABC)

    def test_gridworld_not_environment_protocol(self):
        """GridWorld 不直接继承 EnvironmentProtocol（stage_a 不依赖 stage_i）。

        这是依赖方向的设计：stage_i 依赖 stage_a，而非反向。
        GridWorld 满足 EnvironmentProtocol 的鸭子类型契约，
        但不通过继承正式声明（避免循环依赖）。
        """
        assert not issubclass(GridWorld, EnvironmentProtocol)


# ---------------------------------------------------------------------------
# 顶层导出验证
# ---------------------------------------------------------------------------
class TestTopLevelExport:
    """验证接口在顶层包导出。"""

    def test_protocols_exported(self):
        """协议在顶层包导出。"""
        import agent_whitepaper
        for name in ["PolicyProtocol", "MemoryProtocol", "AgentProtocol"]:
            assert hasattr(agent_whitepaper, name), f"{name} 未导出"
            assert name in agent_whitepaper.__all__, f"{name} 不在 __all__"


# ---------------------------------------------------------------------------
# 无循环导入验证
# ---------------------------------------------------------------------------
class TestNoCircularImport:
    """验证 _protocols.py 无循环导入。"""

    def test_protocols_no_runtime_import_of_stages(self):
        """_protocols.py 运行时不导入阶段模块（TYPE_CHECKING 块内允许）。"""
        import ast

        proto_file = SRC_ROOT / "_protocols.py"
        content = proto_file.read_text(encoding="utf-8")
        tree = ast.parse(content)

        # 收集 TYPE_CHECKING 块内的导入行号范围
        type_checking_ranges: list = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            # 检测 if TYPE_CHECKING: 块
            test = node.test
            is_tc = (
                (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
                or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")
            )
            if is_tc:
                type_checking_ranges.append((node.lineno, node.end_lineno or node.lineno))

        def in_type_checking(line: int) -> bool:
            return any(lo <= line <= hi for lo, hi in type_checking_ranges)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("agent_whitepaper.stage_"):
                    if not in_type_checking(node.lineno):
                        pytest.fail(
                            f"_protocols.py 运行时导入了阶段模块: {node.module}"
                        )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("agent_whitepaper.stage_"):
                        if not in_type_checking(node.lineno):
                            pytest.fail(
                                f"_protocols.py 运行时导入了阶段模块: {alias.name}"
                            )
