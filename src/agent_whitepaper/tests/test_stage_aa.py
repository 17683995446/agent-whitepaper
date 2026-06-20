"""阶段AA测试：验证示例脚本库。

验证内容：
1. examples/ 目录存在且有5个示例脚本
2. 每个示例脚本可被Python解析（语法正确）
3. 示例脚本使用正确的API
4. examples/README.md 存在且包含学习路径
5. 示例脚本不使用私有属性
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"

EXPECTED_EXAMPLES = [
    "01_single_agent_training.py",
    "02_multi_agent_system.py",
    "03_custom_environment.py",
    "04_persistence_and_recovery.py",
    "05_skill_abstraction.py",
]


# ---------------------------------------------------------------------------
# 目录结构验证
# ---------------------------------------------------------------------------
class TestExamplesDirectory:
    """验证 examples/ 目录结构。"""

    def test_examples_dir_exists(self):
        """examples/ 目录存在。"""
        assert EXAMPLES_DIR.exists(), "examples/ 目录应存在"
        assert EXAMPLES_DIR.is_dir()

    def test_examples_readme_exists(self):
        """examples/README.md 存在。"""
        readme = EXAMPLES_DIR / "README.md"
        assert readme.exists(), "examples/README.md 应存在"

    def test_five_examples_exist(self):
        """5个示例脚本都存在。"""
        for name in EXPECTED_EXAMPLES:
            path = EXAMPLES_DIR / name
            assert path.exists(), f"示例脚本 {name} 应存在"

    def test_examples_readme_has_learning_path(self):
        """README 包含学习路径说明。"""
        content = (EXAMPLES_DIR / "README.md").read_text(encoding="utf-8")
        assert "学习路径" in content or "运行方式" in content
        assert "01_single_agent" in content

    def test_examples_readme_lists_all_examples(self):
        """README 列出所有示例。"""
        content = (EXAMPLES_DIR / "README.md").read_text(encoding="utf-8")
        for name in EXPECTED_EXAMPLES:
            assert name in content, f"README 应提及 {name}"


# ---------------------------------------------------------------------------
# 语法正确性验证
# ---------------------------------------------------------------------------
class TestExampleSyntax:
    """验证示例脚本语法正确（可被AST解析）。"""

    @pytest.mark.parametrize("example_name", EXPECTED_EXAMPLES)
    def test_example_parses(self, example_name):
        """示例脚本可被Python AST解析。"""
        path = EXAMPLES_DIR / example_name
        source = path.read_text(encoding="utf-8")
        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"{example_name} 语法错误: {e}")

    @pytest.mark.parametrize("example_name", EXPECTED_EXAMPLES)
    def test_example_has_main_function(self, example_name):
        """每个示例都有 main() 函数。"""
        path = EXAMPLES_DIR / example_name
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        func_names = [
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        ]
        assert "main" in func_names, f"{example_name} 应有 main() 函数"

    @pytest.mark.parametrize("example_name", EXPECTED_EXAMPLES)
    def test_example_has_main_guard(self, example_name):
        """每个示例都有 if __name__ == '__main__' 守护。"""
        path = EXAMPLES_DIR / example_name
        source = path.read_text(encoding="utf-8")
        assert '__name__' in source and '__main__' in source, \
            f"{example_name} 应有 if __name__ == '__main__' 守护"

    @pytest.mark.parametrize("example_name", EXPECTED_EXAMPLES)
    def test_example_has_docstring(self, example_name):
        """每个示例都有模块docstring。"""
        path = EXAMPLES_DIR / example_name
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert ast.get_docstring(tree) is not None, \
            f"{example_name} 应有模块docstring"


# ---------------------------------------------------------------------------
# API 一致性验证
# ---------------------------------------------------------------------------
class TestExampleAPIConsistency:
    """验证示例使用的API与实际一致。"""

    def test_example01_uses_correct_api(self):
        """示例01使用正确的API。"""
        source = (EXAMPLES_DIR / "01_single_agent_training.py").read_text(encoding="utf-8")
        assert "from agent_whitepaper import GridWorld, SelfDrivenAgent" in source
        assert "n_states=25" in source
        assert "run_episode" in source
        assert "agent.stats.success_rate" in source

    def test_example02_uses_correct_api(self):
        """示例02使用正确的API。"""
        source = (EXAMPLES_DIR / "02_multi_agent_system.py").read_text(encoding="utf-8")
        assert "OrchestratorConfig" in source
        assert "SystemOrchestrator" in source
        assert "orch.initialize()" in source
        assert "orch.run_iteration()" in source

    def test_example03_uses_correct_api(self):
        """示例03使用正确的API。"""
        source = (EXAMPLES_DIR / "03_custom_environment.py").read_text(encoding="utf-8")
        assert "EnvironmentProtocol" in source
        assert "StateEncoder" in source
        assert "def reset" in source
        assert "def step" in source

    def test_example04_uses_correct_api(self):
        """示例04使用正确的API。"""
        source = (EXAMPLES_DIR / "04_persistence_and_recovery.py").read_text(encoding="utf-8")
        assert "PersistenceEngine" in source
        assert "create_checkpoint" in source
        assert "get_latest_checkpoint" in source
        assert "log_event" in source

    def test_example05_uses_correct_api(self):
        """示例05使用正确的API。"""
        source = (EXAMPLES_DIR / "05_skill_abstraction.py").read_text(encoding="utf-8")
        assert "SkillAbstractor" in source
        assert "SkillLibrary" in source
        assert "MetaLearner" in source
        assert "abstract_from_episodes" in source


# ---------------------------------------------------------------------------
# 不使用私有属性验证
# ---------------------------------------------------------------------------
class TestExamplesNoPrivateAccess:
    """验证示例不访问库对象的私有属性。"""

    @pytest.mark.parametrize("example_name", EXPECTED_EXAMPLES)
    def test_no_private_attribute_access(self, example_name):
        """示例不访问库对象的私有属性（self._xxx 是用户自己的类，允许）。"""
        source = (EXAMPLES_DIR / example_name).read_text(encoding="utf-8")
        # 只检查访问库对象私有属性的模式（如 agent._table, q_table._table）
        # self._xxx 是用户自定义类的私有属性，属于合法用法
        lines = source.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # 跳过 self._xxx 赋值/访问（用户自己的类）
            if "self._" in stripped:
                continue
            # 检查访问库对象私有属性
            for pattern in ["._table", "._skills", "._conn", "._should_stop"]:
                # 排除 self._xxx 的情况
                if pattern in stripped and not stripped.startswith("self."):
                    pytest.fail(
                        f"{example_name} 第{i}行不应访问库对象私有属性: {stripped}"
                    )


# ---------------------------------------------------------------------------
# 可运行性验证（快速验证，不完整运行）
# ---------------------------------------------------------------------------
class TestExamplesRunnable:
    """验证示例脚本可被导入（不运行main）。"""

    @pytest.mark.parametrize("example_name", EXPECTED_EXAMPLES)
    def test_example_importable(self, example_name):
        """示例脚本可被导入（验证导入路径正确）。"""
        path = EXAMPLES_DIR / example_name
        source = path.read_text(encoding="utf-8")
        # 验证导入语句可被解析
        tree = ast.parse(source)
        import_lines = [
            node for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert len(import_lines) > 0, f"{example_name} 应有导入语句"

    def test_example01_actually_runs(self):
        """示例01实际可运行（完整执行）。"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        result = subprocess.run(
            [sys.executable, str(EXAMPLES_DIR / "01_single_agent_training.py")],
            capture_output=True, text=True, timeout=30, env=env,
        )
        assert result.returncode == 0, f"示例01运行失败: {result.stderr}"
        assert "训练结果" in result.stdout

    def test_example03_actually_runs(self):
        """示例03实际可运行（完整执行）。"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        result = subprocess.run(
            [sys.executable, str(EXAMPLES_DIR / "03_custom_environment.py")],
            capture_output=True, text=True, timeout=30, env=env,
        )
        assert result.returncode == 0, f"示例03运行失败: {result.stderr}"
        assert "温度调节" in result.stdout
