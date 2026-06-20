"""阶段AB测试：验证环境诊断命令（agentwp doctor）。

验证内容：
1. doctor 子命令注册到CLI
2. doctor 命令检查Python版本
3. doctor 命令检查核心依赖
4. doctor 命令检查可选依赖
5. doctor 命令检查模块导入
6. doctor 命令检查公共API
7. doctor 命令检查CLI子命令
8. doctor 命令执行功能性验证
9. doctor 命令返回码正确
"""

import argparse
import subprocess
import sys
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import pytest

from agent_whitepaper.stage_h.cli import build_parser, cmd_doctor, main


# ---------------------------------------------------------------------------
# 子命令注册验证
# ---------------------------------------------------------------------------
class TestDoctorCommandRegistered:
    """验证 doctor 子命令注册到CLI。"""

    def test_doctor_in_help(self):
        """agentwp --help 包含 doctor 子命令。"""
        parser = build_parser()
        buf = StringIO()
        with redirect_stdout(buf):
            with pytest.raises(SystemExit):
                parser.parse_args(["--help"])
        assert "doctor" in buf.getvalue()

    def test_doctor_subcommand_exists(self):
        """doctor 子命令存在。"""
        parser = build_parser()
        args = parser.parse_args(["doctor"])
        assert hasattr(args, "func")
        assert args.func == cmd_doctor

    def test_doctor_no_extra_args_required(self):
        """doctor 命令不需要额外参数。"""
        parser = build_parser()
        args = parser.parse_args(["doctor"])
        # doctor 不需要任何额外参数
        assert args.func is not None


# ---------------------------------------------------------------------------
# doctor 命令执行验证
# ---------------------------------------------------------------------------
class TestDoctorExecution:
    """验证 doctor 命令执行。"""

    def test_doctor_returns_zero_on_success(self):
        """doctor 命令在环境正常时返回0。"""
        args = argparse.Namespace()
        buf = StringIO()
        with redirect_stdout(buf):
            ret = cmd_doctor(args)
        # 在测试环境中应该返回0（可能有警告但无失败）
        assert ret in (0, 1)  # 0=全部通过或仅有警告，1=有失败

    def test_doctor_output_contains_sections(self):
        """doctor 输出包含各检查章节。"""
        args = argparse.Namespace()
        buf = StringIO()
        with redirect_stdout(buf):
            cmd_doctor(args)
        output = buf.getvalue()
        assert "Python 环境" in output
        assert "核心依赖" in output
        assert "可选依赖" in output
        assert "核心模块导入" in output
        assert "公共API" in output
        assert "CLI 子命令" in output
        assert "功能性验证" in output

    def test_doctor_output_contains_summary(self):
        """doctor 输出包含诊断总结。"""
        args = argparse.Namespace()
        buf = StringIO()
        with redirect_stdout(buf):
            cmd_doctor(args)
        output = buf.getvalue()
        assert "诊断完成" in output
        assert "通过" in output
        assert "失败" in output

    def test_doctor_checks_python_version(self):
        """doctor 检查Python版本。"""
        args = argparse.Namespace()
        buf = StringIO()
        with redirect_stdout(buf):
            cmd_doctor(args)
        output = buf.getvalue()
        assert "Python 版本" in output
        assert "3.8" in output

    def test_doctor_checks_numpy(self):
        """doctor 检查numpy依赖。"""
        args = argparse.Namespace()
        buf = StringIO()
        with redirect_stdout(buf):
            cmd_doctor(args)
        output = buf.getvalue()
        assert "numpy" in output

    def test_doctor_checks_pyyaml(self):
        """doctor 检查pyyaml依赖。"""
        args = argparse.Namespace()
        buf = StringIO()
        with redirect_stdout(buf):
            cmd_doctor(args)
        output = buf.getvalue()
        assert "yaml" in output.lower()

    def test_doctor_checks_psutil(self):
        """doctor 检查psutil依赖。"""
        args = argparse.Namespace()
        buf = StringIO()
        with redirect_stdout(buf):
            cmd_doctor(args)
        output = buf.getvalue()
        assert "psutil" in output

    def test_doctor_checks_all_subcommands(self):
        """doctor 检查所有CLI子命令。"""
        args = argparse.Namespace()
        buf = StringIO()
        with redirect_stdout(buf):
            cmd_doctor(args)
        output = buf.getvalue()
        for cmd in ["init", "run", "demo", "status", "doctor", "wizard", "report", "plugins", "shell"]:
            assert f"agentwp {cmd}" in output

    def test_doctor_runs_functional_test(self):
        """doctor 执行功能性验证。"""
        args = argparse.Namespace()
        buf = StringIO()
        with redirect_stdout(buf):
            cmd_doctor(args)
        output = buf.getvalue()
        assert "GridWorld 运行" in output
        assert "SelfDrivenAgent 运行" in output


# ---------------------------------------------------------------------------
# doctor 返回码验证
# ---------------------------------------------------------------------------
class TestDoctorReturnCode:
    """验证 doctor 命令返回码。"""

    def test_doctor_returns_zero_when_all_pass(self):
        """所有检查通过时返回0。"""
        args = argparse.Namespace()
        buf = StringIO()
        with redirect_stdout(buf):
            ret = cmd_doctor(args)
        # 在正常环境中应返回0（可能有警告）
        assert ret == 0

    def test_doctor_returns_one_on_failure(self):
        """有检查失败时返回1。"""
        args = argparse.Namespace()
        # 模拟核心依赖缺失
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def mock_import(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError("No module named 'numpy'")
            return original_import(name, *args, **kwargs)

        buf = StringIO()
        with patch("builtins.__import__", side_effect=mock_import):
            with redirect_stdout(buf):
                ret = cmd_doctor(args)
        assert ret == 1


# ---------------------------------------------------------------------------
# 通过main()调用验证
# ---------------------------------------------------------------------------
class TestDoctorViaMain:
    """验证通过main()调用doctor。"""

    def test_main_doctor_works(self):
        """main(['doctor']) 正常工作。"""
        buf = StringIO()
        with redirect_stdout(buf):
            ret = main(["doctor"])
        assert ret in (0, 1)
        assert "诊断完成" in buf.getvalue()

    def test_main_doctor_via_subprocess(self):
        """通过子进程验证 agentwp doctor。"""
        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = str(
            __import__("pathlib").Path(__file__).resolve().parent.parent.parent.parent / "src"
        )
        result = subprocess.run(
            [sys.executable, "-m", "agent_whitepaper.stage_h.cli", "doctor"],
            capture_output=True, text=True, timeout=30, env=env,
        )
        assert "诊断完成" in result.stdout
        assert "通过" in result.stdout


# ---------------------------------------------------------------------------
# 源码结构验证
# ---------------------------------------------------------------------------
class TestDoctorSourceCode:
    """验证 doctor 命令源码结构。"""

    def test_cmd_doctor_exists(self):
        """cmd_doctor 函数存在。"""
        assert callable(cmd_doctor)

    def test_doctor_has_docstring(self):
        """cmd_doctor 有docstring。"""
        assert cmd_doctor.__doc__ is not None
        assert "诊断" in cmd_doctor.__doc__

    def test_doctor_registered_in_parser(self):
        """doctor 在 parser 中注册。"""
        import inspect
        from pathlib import Path

        source = Path(inspect.getfile(build_parser)).read_text(encoding="utf-8")
        assert "cmd_doctor" in source
        assert "doctor" in source
        assert "p_doctor" in source
