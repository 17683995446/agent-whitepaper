"""阶段Y测试：验证错误诊断增强。

验证内容：
1. TROUBLESHOOTING.md 存在且覆盖常见错误场景
2. resource_manager.py 缺失 psutil 时记录友好警告
3. 关键错误消息包含可操作的解决建议
4. CLI 异常消息包含诊断信息
"""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_whitepaper.stage_h.cli import build_parser
from agent_whitepaper.stage_j.resource_manager import ResourceManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TROUBLESHOOTING_PATH = PROJECT_ROOT / "TROUBLESHOOTING.md"


# ---------------------------------------------------------------------------
# TROUBLESHOOTING.md 文档验证
# ---------------------------------------------------------------------------
class TestTroubleshootingDoc:
    """验证故障排查文档。"""

    def test_troubleshooting_exists(self):
        """TROUBLESHOOTING.md 存在。"""
        assert TROUBLESHOOTING_PATH.exists(), "TROUBLESHOOTING.md 应存在"

    def test_troubleshooting_has_content(self):
        """TROUBLESHOOTING.md 有实质内容。"""
        content = TROUBLESHOOTING_PATH.read_text(encoding="utf-8")
        assert len(content) > 500, "故障排查文档应包含足够内容"

    def test_troubleshooting_covers_installation(self):
        """覆盖安装问题。"""
        content = TROUBLESHOOTING_PATH.read_text(encoding="utf-8")
        assert "安装" in content or "install" in content.lower()
        assert "numpy" in content.lower()
        assert "pyyaml" in content.lower()

    def test_troubleshooting_covers_configuration(self):
        """覆盖配置错误。"""
        content = TROUBLESHOOTING_PATH.read_text(encoding="utf-8")
        assert "配置" in content or "config" in content.lower()
        assert "ConfigError" in content or "grid_size" in content

    def test_troubleshooting_covers_runtime(self):
        """覆盖运行时错误。"""
        content = TROUBLESHOOTING_PATH.read_text(encoding="utf-8")
        assert "运行时" in content or "runtime" in content.lower()
        assert "数据库" in content or "database" in content.lower()

    def test_troubleshooting_covers_cli(self):
        """覆盖 CLI 错误。"""
        content = TROUBLESHOOTING_PATH.read_text(encoding="utf-8")
        assert "CLI" in content or "命令行" in content
        assert "agentwp" in content

    def test_troubleshooting_covers_api(self):
        """覆盖 API 使用错误。"""
        content = TROUBLESHOOTING_PATH.read_text(encoding="utf-8")
        assert "API" in content
        assert "n_states" in content

    def test_troubleshooting_covers_performance(self):
        """覆盖性能问题。"""
        content = TROUBLESHOOTING_PATH.read_text(encoding="utf-8")
        assert "性能" in content or "performance" in content.lower()

    def test_troubleshooting_has_solutions(self):
        """每个问题都有解决方案（pip install 等）。"""
        content = TROUBLESHOOTING_PATH.read_text(encoding="utf-8")
        assert "pip install" in content, "应包含安装解决命令"
        assert "解决" in content, "应包含解决步骤"

    def test_troubleshooting_mentions_verbose(self):
        """提及 --verbose 标志用于诊断。"""
        content = TROUBLESHOOTING_PATH.read_text(encoding="utf-8")
        assert "--verbose" in content, "应建议使用 --verbose 进行诊断"


# ---------------------------------------------------------------------------
# 可选依赖缺失提示验证
# ---------------------------------------------------------------------------
class TestOptionalDependencyHints:
    """验证可选依赖缺失时的友好提示。"""

    def test_psutil_missing_logs_warning(self, caplog):
        """psutil 缺失时记录 WARNING 级别日志。"""
        import logging as logging_module

        # _logging.py 设置 propagate=False，需临时启用以让 caplog 捕获
        aw_logger = logging_module.getLogger("agent_whitepaper")
        original_propagate = aw_logger.propagate
        aw_logger.propagate = True
        try:
            with patch("builtins.__import__", side_effect=_mock_import_without_psutil):
                with caplog.at_level(logging.WARNING, logger="agent_whitepaper"):
                    ResourceManager()
        finally:
            aw_logger.propagate = original_propagate
        # 验证警告日志包含 psutil 和 pip install
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("psutil" in msg for msg in warning_messages), \
            "应记录包含 psutil 的警告"
        assert any("pip install" in msg for msg in warning_messages), \
            "警告应包含安装命令"

    def test_config_yaml_error_has_pip_hint(self):
        """PyYAML 缺失时错误消息包含 pip install 提示。"""
        from agent_whitepaper.stage_h.config import ConfigLoader
        loader = ConfigLoader()
        with patch("builtins.__import__", side_effect=_mock_import_without_yaml):
            with pytest.raises(ImportError) as exc_info:
                loader.from_yaml("dummy.yaml")
        assert "pip install" in str(exc_info.value), \
            "ImportError 应包含 pip install 安装提示"
        assert "pyyaml" in str(exc_info.value).lower(), \
            "ImportError 应提及 pyyaml"


def _mock_import_without_psutil(name, *args, **kwargs):
    """模拟 psutil 不可用的 import 函数。"""
    if name == "psutil":
        raise ImportError("No module named 'psutil'")
    return __builtins__.__import__(name, *args, **kwargs) if hasattr(__builtins__, "__import__") \
        else __import__(name, *args, **kwargs)


def _mock_import_without_yaml(name, *args, **kwargs):
    """模拟 yaml 不可用的 import 函数。"""
    if name == "yaml":
        raise ImportError("No module named 'yaml'")
    return __builtins__.__import__(name, *args, **kwargs) if hasattr(__builtins__, "__import__") \
        else __import__(name, *args, **kwargs)


# ---------------------------------------------------------------------------
# 错误消息可操作性验证
# ---------------------------------------------------------------------------
class TestErrorMessageActionability:
    """验证错误消息包含可操作的解决建议。"""

    def test_gridworld_invalid_size_has_hint(self):
        """GridWorld 无效网格大小错误有明确提示。"""
        from agent_whitepaper import GridWorld
        with pytest.raises(ValueError) as exc_info:
            GridWorld(size=1)
        assert "2" in str(exc_info.value), "应提示最小值为 2"

    def test_qtable_invalid_learning_rate_has_hint(self):
        """QTable 无效学习率错误有明确提示。"""
        from agent_whitepaper import QTable
        with pytest.raises(ValueError) as exc_info:
            QTable(n_states=25, n_actions=4, learning_rate=0)
        assert "学习率" in str(exc_info.value), "应提及学习率参数"

    def test_qtable_invalid_states_has_hint(self):
        """QTable 无效状态数错误有明确提示。"""
        from agent_whitepaper import QTable
        with pytest.raises(ValueError) as exc_info:
            QTable(n_states=0, n_actions=4)
        assert "状态" in str(exc_info.value), "应提及状态参数"


# ---------------------------------------------------------------------------
# CLI 诊断能力验证
# ---------------------------------------------------------------------------
class TestCLIDiagnostics:
    """验证 CLI 的诊断能力。"""

    def test_cli_has_verbose_flag(self):
        """CLI 有 --verbose 标志用于诊断。"""
        parser = build_parser()
        args = parser.parse_args(["--verbose", "demo"])
        assert args.verbose is True

    def test_cli_has_quiet_flag(self):
        """CLI 有 --quiet 标志。"""
        parser = build_parser()
        args = parser.parse_args(["--quiet", "demo"])
        assert args.quiet is True

    def test_cli_has_version_flag(self):
        """CLI 有 --version 标志。"""
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_cli_help_mentions_troubleshooting_commands(self):
        """CLI --help 输出包含所有子命令。"""
        import io
        from contextlib import redirect_stdout

        parser = build_parser()
        buf = io.StringIO()
        with redirect_stdout(buf):
            with pytest.raises(SystemExit):
                parser.parse_args(["--help"])
        output = buf.getvalue()
        for cmd in ["init", "run", "demo", "status"]:
            assert cmd in output, f"help 输出应包含命令: {cmd}"


# ---------------------------------------------------------------------------
# 源码静态验证
# ---------------------------------------------------------------------------
class TestSourceCodeDiagnostics:
    """验证源码中的诊断增强。"""

    def test_resource_manager_imports_logger(self):
        """resource_manager.py 导入 get_logger。"""
        import inspect
        from pathlib import Path

        source = Path(inspect.getfile(ResourceManager)).read_text(encoding="utf-8")
        assert "from agent_whitepaper._logging import get_logger" in source
        assert "logger = get_logger" in source

    def test_resource_manager_logs_psutil_warning(self):
        """resource_manager.py 在 psutil 缺失时记录警告。"""
        import inspect
        from pathlib import Path

        source = Path(inspect.getfile(ResourceManager)).read_text(encoding="utf-8")
        assert "logger.warning" in source
        assert "psutil" in source
        assert "pip install psutil" in source

    def test_config_has_friendly_import_error(self):
        """config.py 的 ImportError 包含 pip install 提示。"""
        import inspect
        from pathlib import Path

        from agent_whitepaper.stage_h.config import ConfigLoader
        source = Path(inspect.getfile(ConfigLoader)).read_text(encoding="utf-8")
        assert "pip install pyyaml" in source
