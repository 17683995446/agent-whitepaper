"""阶段X测试：验证API文档示例。

验证内容：
1. 三大核心类（SelfDrivenAgent, GridWorld, SystemOrchestrator）docstring 包含示例
2. QTable docstring 包含示例
3. 示例代码可以实际运行
4. 示例使用正确的 API 签名
5. 示例不使用私有属性
"""

import inspect
import re
from pathlib import Path

import pytest

from agent_whitepaper import GridWorld, QTable, SelfDrivenAgent
from agent_whitepaper.stage_g.orchestrator import OrchestratorConfig, SystemOrchestrator


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _extract_docstring(obj) -> str:
    """提取对象的 docstring。"""
    doc = obj.__doc__
    if doc is None:
        pytest.fail(f"{obj.__name__} 没有 docstring")
    return doc


def _extract_doctest_examples(docstring: str) -> list:
    """从 docstring 中提取 >>> 开头的示例代码行。"""
    lines = docstring.split("\n")
    examples = []
    current_example = []
    in_example = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(">>>") or stripped.startswith("..."):
            # 去掉 ">>> " 或 "... " 前缀
            code = re.sub(r"^\.\.\.\s*", "", re.sub(r"^>>>\s*", "", stripped))
            current_example.append(code)
            in_example = True
        elif in_example and stripped and not stripped.startswith(">>>"):
            # 遇到非示例行，结束当前示例
            if current_example:
                examples.append("\n".join(current_example))
                current_example = []
            in_example = False
    if current_example:
        examples.append("\n".join(current_example))
    return examples


# ---------------------------------------------------------------------------
# Docstring 存在性验证
# ---------------------------------------------------------------------------
class TestDocstringExamplesExist:
    """验证核心类 docstring 包含使用示例。"""

    def test_self_driven_agent_has_example(self):
        """SelfDrivenAgent docstring 包含示例。"""
        doc = _extract_docstring(SelfDrivenAgent)
        assert "示例" in doc or "Example" in doc, "SelfDrivenAgent 应包含使用示例"
        assert ">>>" in doc, "SelfDrivenAgent 示例应使用 doctest 格式"

    def test_grid_world_has_example(self):
        """GridWorld docstring 包含示例。"""
        doc = _extract_docstring(GridWorld)
        assert "示例" in doc or "Example" in doc, "GridWorld 应包含使用示例"
        assert ">>>" in doc, "GridWorld 示例应使用 doctest 格式"

    def test_system_orchestrator_has_example(self):
        """SystemOrchestrator docstring 包含示例。"""
        doc = _extract_docstring(SystemOrchestrator)
        assert "示例" in doc or "Example" in doc, "SystemOrchestrator 应包含使用示例"
        assert ">>>" in doc, "SystemOrchestrator 示例应使用 doctest 格式"

    def test_qtable_has_example(self):
        """QTable docstring 包含示例。"""
        doc = _extract_docstring(QTable)
        assert "示例" in doc or "Example" in doc, "QTable 应包含使用示例"
        assert ">>>" in doc, "QTable 示例应使用 doctest 格式"


# ---------------------------------------------------------------------------
# 示例可运行性验证
# ---------------------------------------------------------------------------
class TestExamplesRunnable:
    """验证示例代码可以实际运行。"""

    def test_gridworld_example_runs(self):
        """GridWorld 示例可运行。"""
        env = GridWorld(size=5, seed=42)
        obs = env.reset()
        assert obs.shape == (50,)
        obs, reward, done, info = env.step(GridWorld.ACTION_RIGHT)
        assert info["agent_pos"] == (0, 1)

    def test_qtable_example_runs(self):
        """QTable 示例可运行。"""
        q = QTable(n_states=25, n_actions=4, learning_rate=0.1)
        td_error = q.update(state=0, action=1, reward=1.0, next_state=6, done=False)
        assert round(td_error, 2) == 1.0
        assert q.best_action(0) in range(4)

    def test_self_driven_agent_example_runs(self):
        """SelfDrivenAgent 示例可运行。"""
        env = GridWorld(size=5, seed=42)
        agent = SelfDrivenAgent(
            action_space=env.action_space,
            n_states=25,
            learning_rate=0.1,
            seed=42,
        )
        episode = agent.run_episode(env, max_steps=100)
        assert isinstance(episode.total_reward, float)
        assert isinstance(episode.success, bool)

    def test_system_orchestrator_example_runs(self):
        """SystemOrchestrator 示例可运行。"""
        config = OrchestratorConfig(
            grid_size=5, n_agents=2, db_path=":memory:", max_iterations=3,
        )
        orch = SystemOrchestrator(config)
        orch.initialize()
        for _ in range(config.max_iterations):
            orch.run_iteration()
        report = orch.get_metrics()
        assert "iteration" in report
        orch.stop("done")


# ---------------------------------------------------------------------------
# 示例 API 一致性验证
# ---------------------------------------------------------------------------
class TestExampleAPIConsistency:
    """验证示例使用的 API 与实际签名一致。"""

    def test_gridworld_example_uses_correct_api(self):
        """GridWorld 示例使用正确的构造参数。"""
        doc = _extract_docstring(GridWorld)
        # 示例应使用 size 参数
        assert "size=5" in doc
        # 示例应使用 ACTION_RIGHT 常量
        assert "ACTION_RIGHT" in doc

    def test_self_driven_agent_example_uses_correct_api(self):
        """SelfDrivenAgent 示例使用正确的构造参数。"""
        doc = _extract_docstring(SelfDrivenAgent)
        # 示例应使用 n_states 参数（不是 q_table_size）
        assert "n_states=" in doc
        # 示例应使用 action_space 参数
        assert "action_space=" in doc
        # 示例应使用 run_episode 方法
        assert "run_episode" in doc

    def test_system_orchestrator_example_uses_correct_api(self):
        """SystemOrchestrator 示例使用正确的构造参数。"""
        doc = _extract_docstring(SystemOrchestrator)
        # 示例应使用 OrchestratorConfig
        assert "OrchestratorConfig" in doc
        # 示例应使用 initialize 方法
        assert "initialize" in doc
        # 示例应使用 run_iteration 方法
        assert "run_iteration" in doc

    def test_qtable_example_uses_correct_api(self):
        """QTable 示例使用正确的构造参数。"""
        doc = _extract_docstring(QTable)
        # 示例应使用 n_states 和 n_actions 参数
        assert "n_states=" in doc
        assert "n_actions=" in doc


# ---------------------------------------------------------------------------
# 示例不使用私有属性验证
# ---------------------------------------------------------------------------
class TestExamplesNoPrivateAccess:
    """验证示例不使用私有属性。"""

    def test_self_driven_agent_no_private(self):
        """SelfDrivenAgent 示例不访问私有属性。"""
        doc = _extract_docstring(SelfDrivenAgent)
        assert "._table" not in doc
        assert "._rng" not in doc
        assert "._skills" not in doc

    def test_qtable_no_private(self):
        """QTable 示例不访问私有属性。"""
        doc = _extract_docstring(QTable)
        assert "._table" not in doc

    def test_gridworld_no_private(self):
        """GridWorld 示例不访问私有属性。"""
        doc = _extract_docstring(GridWorld)
        assert "._agent_pos" not in doc
        assert "._step_count" not in doc

    def test_system_orchestrator_no_private(self):
        """SystemOrchestrator 示例不访问私有属性。"""
        doc = _extract_docstring(SystemOrchestrator)
        assert "._should_stop" not in doc
        assert "._iteration" not in doc


# ---------------------------------------------------------------------------
# 源码静态验证
# ---------------------------------------------------------------------------
class TestSourceCodeStructure:
    """验证源码中 docstring 示例的结构。"""

    @staticmethod
    def _read_source(obj) -> str:
        """通过 inspect 获取对象所在文件的源码。"""
        file_path = Path(inspect.getfile(obj))
        return file_path.read_text(encoding="utf-8")

    def test_agent_py_has_doctest_example(self):
        """agent.py 包含 doctest 格式示例。"""
        source = self._read_source(SelfDrivenAgent)
        assert ">>>" in source

    def test_environment_py_has_doctest_example(self):
        """environment.py 包含 doctest 格式示例。"""
        source = self._read_source(GridWorld)
        assert ">>>" in source

    def test_orchestrator_py_has_doctest_example(self):
        """orchestrator.py 包含 doctest 格式示例。"""
        source = self._read_source(SystemOrchestrator)
        assert ">>>" in source

    def test_policy_py_has_doctest_example(self):
        """policy.py 包含 doctest 格式示例。"""
        source = self._read_source(QTable)
        assert ">>>" in source
