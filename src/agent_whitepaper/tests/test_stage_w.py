"""阶段W测试：验证CLI增强。

验证内容：
1. --version 全局标志输出版本号
2. --verbose/--quiet 全局标志存在并连接到 configure_logging
3. demo 子命令支持 -n/-a/-g 参数化
4. 配置错误产生友好提示（非traceback）
5. CLI异常被友好捕获（ConfigError/AgentWhitepaperError）
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_whitepaper import __version__
from agent_whitepaper.stage_h import cli as cli_module
from agent_whitepaper.stage_h.cli import build_parser, main


# ---------------------------------------------------------------------------
# --version 标志验证
# ---------------------------------------------------------------------------
class TestVersionFlag:
    """验证 --version 全局标志。"""

    def test_version_flag_exists(self):
        """parser 包含 --version 标志。"""
        parser = build_parser()
        # 通过 --version 触发 SystemExit（argparse 的 version action 行为）
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        # argparse version action 退出码为 0
        assert exc_info.value.code == 0

    def test_version_flag_output(self, capsys):
        """--version 输出包含版本号。"""
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--version"])
        captured = capsys.readouterr()
        assert __version__ in captured.out

    def test_version_short_flag(self, capsys):
        """-V 短标志也工作。"""
        with pytest.raises(SystemExit):
            build_parser().parse_args(["-V"])
        captured = capsys.readouterr()
        assert __version__ in captured.out

    def test_version_via_subprocess(self):
        """通过子进程验证 agentwp --version。"""
        result = subprocess.run(
            [sys.executable, "-m", "agent_whitepaper.stage_h.cli", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        assert __version__ in result.stdout


# ---------------------------------------------------------------------------
# --verbose/--quiet 标志验证
# ---------------------------------------------------------------------------
class TestVerboseQuietFlags:
    """验证 --verbose/--quiet 全局标志。"""

    def test_verbose_flag_exists(self):
        """parser 包含 --verbose 标志。"""
        parser = build_parser()
        args = parser.parse_args(["--verbose", "demo"])
        assert args.verbose is True

    def test_verbose_default_false(self):
        """--verbose 默认为 False。"""
        parser = build_parser()
        args = parser.parse_args(["demo"])
        assert args.verbose is False

    def test_quiet_flag_exists(self):
        """parser 包含 --quiet 标志。"""
        parser = build_parser()
        args = parser.parse_args(["--quiet", "demo"])
        assert args.quiet is True

    def test_quiet_default_false(self):
        """--quiet 默认为 False。"""
        parser = build_parser()
        args = parser.parse_args(["demo"])
        assert args.quiet is False

    def test_verbose_connects_to_configure_logging(self):
        """--verbose 标志触发 configure_logging(DEBUG)。"""
        with patch.object(cli_module, "configure_logging") as mock_config:
            with patch.object(cli_module, "cmd_demo", return_value=0) as mock_cmd:
                main(["--verbose", "demo", "-n", "1"])
                mock_config.assert_called_once_with(level=logging.DEBUG)
                mock_cmd.assert_called_once()

    def test_quiet_connects_to_configure_logging(self):
        """--quiet 标志触发 configure_logging(ERROR)。"""
        with patch.object(cli_module, "configure_logging") as mock_config:
            with patch.object(cli_module, "cmd_demo", return_value=0) as mock_cmd:
                main(["--quiet", "demo", "-n", "1"])
                mock_config.assert_called_once_with(level=logging.ERROR)
                mock_cmd.assert_called_once()

    def test_no_flag_no_configure_logging(self):
        """无 --verbose/--quiet 时不调用 configure_logging。"""
        with patch.object(cli_module, "configure_logging") as mock_config:
            with patch.object(cli_module, "cmd_demo", return_value=0) as mock_cmd:
                main(["demo", "-n", "1"])
                mock_config.assert_not_called()
                mock_cmd.assert_called_once()


# ---------------------------------------------------------------------------
# demo 子命令参数化验证
# ---------------------------------------------------------------------------
class TestDemoParameterization:
    """验证 demo 子命令参数化。"""

    def test_demo_default_iterations(self):
        """demo 默认迭代次数为 30。"""
        parser = build_parser()
        args = parser.parse_args(["demo"])
        assert args.iterations == 30

    def test_demo_custom_iterations(self):
        """demo -n 指定迭代次数。"""
        parser = build_parser()
        args = parser.parse_args(["demo", "-n", "5"])
        assert args.iterations == 5

    def test_demo_long_iterations_flag(self):
        """demo --iterations 长标志。"""
        parser = build_parser()
        args = parser.parse_args(["demo", "--iterations", "10"])
        assert args.iterations == 10

    def test_demo_default_agents(self):
        """demo 默认 Agent 数为 3。"""
        parser = build_parser()
        args = parser.parse_args(["demo"])
        assert args.agents == 3

    def test_demo_custom_agents(self):
        """demo -a 指定 Agent 数。"""
        parser = build_parser()
        args = parser.parse_args(["demo", "-a", "2"])
        assert args.agents == 2

    def test_demo_default_grid(self):
        """demo 默认网格大小为 5。"""
        parser = build_parser()
        args = parser.parse_args(["demo"])
        assert args.grid == 5

    def test_demo_custom_grid(self):
        """demo -g 指定网格大小。"""
        parser = build_parser()
        args = parser.parse_args(["demo", "-g", "4"])
        assert args.grid == 4

    def test_demo_all_params(self):
        """demo 同时指定所有参数。"""
        parser = build_parser()
        args = parser.parse_args(["demo", "-n", "5", "-a", "2", "-g", "4"])
        assert args.iterations == 5
        assert args.agents == 2
        assert args.grid == 4


# ---------------------------------------------------------------------------
# 异常友好捕获验证
# ---------------------------------------------------------------------------
class TestExceptionHandling:
    """验证 CLI 异常友好捕获。"""

    def test_run_config_error_friendly(self, capsys):
        """cmd_run 配置错误产生友好提示，返回码1。"""
        from agent_whitepaper._exceptions import ConfigError
        from agent_whitepaper.stage_h.cli import cmd_run

        # 构造一个会触发 ConfigError 的 args
        args = argparse.Namespace(
            config="/nonexistent/config.yaml",
            iterations=None, agents=None, grid=None,
            memory=False, report_interval=10,
        )
        with patch.object(cli_module, "load_config",
                          side_effect=ConfigError("测试配置错误")):
            ret = cmd_run(args)
        assert ret == 1
        captured = capsys.readouterr()
        assert "配置错误" in captured.err
        assert "测试配置错误" in captured.err

    def test_run_file_not_found_friendly(self, capsys):
        """cmd_run 文件不存在产生友好提示，返回码1。"""
        from agent_whitepaper.stage_h.cli import cmd_run

        args = argparse.Namespace(
            config="/nonexistent/config.yaml",
            iterations=None, agents=None, grid=None,
            memory=False, report_interval=10,
        )
        with patch.object(cli_module, "load_config",
                          side_effect=FileNotFoundError("文件不存在")):
            ret = cmd_run(args)
        assert ret == 1
        captured = capsys.readouterr()
        assert "配置文件不存在" in captured.err

    def test_run_init_error_friendly(self, capsys):
        """cmd_run 系统初始化失败产生友好提示，返回码1。"""
        from agent_whitepaper._exceptions import AgentWhitepaperError
        from agent_whitepaper.stage_h.cli import cmd_run

        # 构造一个最小 config mock
        mock_config = argparse.Namespace(
            grid_size=5, n_agents=3, db_path=":memory:", max_iterations=1,
        )
        args = argparse.Namespace(
            config=None,
            iterations=None, agents=None, grid=None,
            memory=False, report_interval=10,
        )
        with patch.object(cli_module, "load_config", return_value=mock_config):
            with patch.object(cli_module, "SystemOrchestrator",
                              side_effect=AgentWhitepaperError("初始化失败")):
                ret = cmd_run(args)
        assert ret == 1
        captured = capsys.readouterr()
        assert "系统初始化失败" in captured.err

    def test_demo_init_error_friendly(self, capsys):
        """cmd_demo 演示启动失败产生友好提示，返回码1。"""
        from agent_whitepaper._exceptions import AgentWhitepaperError
        from agent_whitepaper.stage_h.cli import cmd_demo

        args = argparse.Namespace(
            iterations=5, agents=2, grid=4,
        )
        with patch.object(cli_module, "load_config",
                          side_effect=AgentWhitepaperError("演示配置错误")):
            ret = cmd_demo(args)
        assert ret == 1
        captured = capsys.readouterr()
        assert "演示启动失败" in captured.err


# ---------------------------------------------------------------------------
# main() 函数行为验证
# ---------------------------------------------------------------------------
class TestMainFunction:
    """验证 main() 函数行为。"""

    def test_main_no_args_prints_help(self, capsys):
        """无子命令时打印帮助，返回1。"""
        ret = main([])
        assert ret == 1
        captured = capsys.readouterr()
        assert "agentwp" in captured.out

    def test_main_returns_func_result(self):
        """main() 返回子命令函数的返回值。"""
        with patch.object(cli_module, "cmd_demo", return_value=42):
            ret = main(["demo", "-n", "1"])
            assert ret == 42


# ---------------------------------------------------------------------------
# 源码静态验证
# ---------------------------------------------------------------------------
class TestSourceCodeStructure:
    """验证 CLI 源码结构符合规范。"""

    @staticmethod
    def _read_cli_source() -> str:
        cli_path = Path(cli_module.__file__)
        return cli_path.read_text(encoding="utf-8")

    def test_imports_configure_logging(self):
        """cli.py 导入 configure_logging。"""
        source = self._read_cli_source()
        assert "from agent_whitepaper._logging import configure_logging" in source

    def test_imports_version(self):
        """cli.py 导入 __version__。"""
        source = self._read_cli_source()
        assert "from agent_whitepaper import __version__" in source

    def test_imports_exceptions(self):
        """cli.py 导入异常类。"""
        source = self._read_cli_source()
        assert "AgentWhitepaperError" in source
        assert "ConfigError" in source

    def test_main_configures_logging(self):
        """main() 函数包含日志配置逻辑。"""
        source = self._read_cli_source()
        assert "configure_logging(level=logging.DEBUG)" in source
        assert "configure_logging(level=logging.ERROR)" in source
        assert "getattr(args, \"verbose\"" in source
        assert "getattr(args, \"quiet\"" in source
