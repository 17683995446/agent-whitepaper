"""阶段V测试：验证快速入门修复。

验证内容：
1. README 中的快速入门示例代码可以实际运行
2. 示例使用的 API 签名与实际代码一致
3. CLI 命令示例与实际 CLI 定义一致
4. 示例不使用私有属性（如 _table）
"""

import io
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
README_PATH = PROJECT_ROOT / "README.md"


# ---------------------------------------------------------------------------
# README 示例可运行性验证
# ---------------------------------------------------------------------------
class TestReadmeQuickstart:
    """验证 README 快速入门示例。"""

    @staticmethod
    def _extract_python_block(readme: str, section: str) -> str:
        """从 README 中提取指定章节的 Python 代码块。"""
        # 找到章节标题
        idx = readme.find(section)
        if idx == -1:
            pytest.fail(f"README 中未找到章节: {section}")
        # 从章节标题开始找第一个 python 代码块
        rest = readme[idx:]
        match = re.search(r"```python\n(.*?)```", rest, re.DOTALL)
        if not match:
            pytest.fail(f"章节 {section} 中未找到 python 代码块")
        return match.group(1)

    def test_readme_exists(self):
        """README.md 存在。"""
        assert README_PATH.exists()

    def test_quickstart_python_example_runs(self):
        """README 最小示例的 Python 代码可以运行。"""
        readme = README_PATH.read_text(encoding="utf-8")
        code = self._extract_python_block(readme, "### 最小示例")

        # 验证代码中不包含错误 API
        assert "q_table_size" not in code, "示例不应使用不存在的 q_table_size 参数"
        assert "._table" not in code, "示例不应访问私有属性 _table"
        assert "reward=0" not in code, "示例不应使用不存在的 reward 关键字参数"

        # 实际运行代码
        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            exec(code, {"__name__": "__test__"})
        output = stdout_buf.getvalue()

        # 验证输出包含预期信息
        assert "成功率" in output, "示例输出应包含成功率"
        assert "平均奖励" in output, "示例输出应包含平均奖励"

    def test_quickstart_uses_correct_api(self):
        """示例使用正确的 API 签名。"""
        readme = README_PATH.read_text(encoding="utf-8")
        code = self._extract_python_block(readme, "### 最小示例")

        # 验证 SelfDrivenAgent 构造参数正确
        assert "n_states=" in code, "应使用 n_states 参数"
        assert "action_space=" in code, "应使用 action_space 参数"

        # 验证使用 run_episode 而非手动 perceive_and_act
        assert "run_episode" in code, "应使用 run_episode 方法"

    def test_quickstart_no_private_attrs(self):
        """示例不使用私有属性。"""
        readme = README_PATH.read_text(encoding="utf-8")
        code = self._extract_python_block(readme, "### 最小示例")
        # 主要检查 _table, _skills, _rng 等私有属性访问
        assert "._table" not in code
        assert "._skills" not in code
        assert "._rng" not in code


# ---------------------------------------------------------------------------
# CLI 示例一致性验证
# ---------------------------------------------------------------------------
class TestReadmeCLI:
    """验证 README CLI 示例与实际 CLI 一致。"""

    def test_cli_demo_no_quick_flag(self):
        """README 不应宣传不存在的 --quick 参数。"""
        readme = README_PATH.read_text(encoding="utf-8")
        assert "demo --quick" not in readme, "README 不应宣传不存在的 --quick 参数"

    def test_cli_commands_mentioned(self):
        """README 应提及所有 CLI 命令。"""
        readme = README_PATH.read_text(encoding="utf-8")
        for cmd in ["agentwp init", "agentwp demo", "agentwp run", "agentwp status"]:
            assert cmd in readme, f"README 应提及 CLI 命令: {cmd}"

    def test_cli_help_works(self):
        """agentwp --help 可以运行。"""
        result = subprocess.run(
            [sys.executable, "-m", "agent_whitepaper.stage_h.cli", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        # argparse 在 --help 时退出码为 0
        assert "init" in result.stdout
        assert "demo" in result.stdout
        assert "run" in result.stdout
        assert "status" in result.stdout


# ---------------------------------------------------------------------------
# 安装说明验证
# ---------------------------------------------------------------------------
class TestReadmeInstall:
    """验证 README 安装说明。"""

    def test_install_command_present(self):
        """README 包含安装命令。"""
        readme = README_PATH.read_text(encoding="utf-8")
        assert "pip install" in readme

    def test_python_version_mentioned(self):
        """README 提及 Python 版本要求。"""
        readme = README_PATH.read_text(encoding="utf-8")
        assert "3.8" in readme or "Python >= 3.8" in readme
