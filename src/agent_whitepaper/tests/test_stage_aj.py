"""阶段AJ测试：验证交互式Shell。

验证内容：
1. InteractiveShell 类
2. 命令解析与执行
3. 各命令处理（help/init/run/status/metrics/agents/report/exit）
4. 非交互模式（run_with_commands）
5. 输入/输出函数注入
6. 错误处理（防火墙层）
7. 便捷函数
8. CLI集成（shell子命令、cmd_shell）
9. 源码结构符合规范
"""

import argparse
import inspect
from pathlib import Path

from agent_whitepaper.stage_h.shell import (
    InteractiveShell,
    run_shell,
    run_shell_with_commands,
)


# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------
class FakeInput:
    """模拟输入序列。"""

    def __init__(self, lines):
        self._lines = list(lines)
        self._idx = 0

    def __call__(self, prompt=""):
        if self._idx >= len(self._lines):
            raise EOFError()
        line = self._lines[self._idx]
        self._idx += 1
        return line


class OutputCollector:
    """收集输出。"""

    def __init__(self):
        self.lines = []

    def __call__(self, msg=""):
        self.lines.append(str(msg))

    @property
    def output(self):
        return "\n".join(self.lines)

    def contains(self, text):
        return any(text in line for line in self.lines)


# ---------------------------------------------------------------------------
# InteractiveShell 类验证
# ---------------------------------------------------------------------------
class TestShellClass:
    """验证 InteractiveShell 类。"""

    def test_shell_class_exists(self):
        """InteractiveShell 类存在且可调用。"""
        assert callable(InteractiveShell)

    def test_shell_has_docstring(self):
        """InteractiveShell 有docstring。"""
        assert InteractiveShell.__doc__ is not None
        assert "REPL" in InteractiveShell.__doc__ or "交互" in InteractiveShell.__doc__

    def test_shell_instantiation(self):
        """可以实例化 InteractiveShell。"""
        shell = InteractiveShell()
        assert shell is not None
        assert shell.orchestrator is None
        assert shell.is_running is False

    def test_shell_with_injected_io(self):
        """支持输入/输出函数注入。"""
        out = OutputCollector()
        shell = InteractiveShell(
            input_func=FakeInput(["exit"]),
            output_func=out,
        )
        shell.run()
        assert out.contains("再见")

    def test_prompt_attribute(self):
        """有提示符属性。"""
        shell = InteractiveShell()
        assert hasattr(shell, "PROMPT")
        assert "agentwp" in shell.PROMPT


# ---------------------------------------------------------------------------
# 命令解析验证
# ---------------------------------------------------------------------------
class TestCommandParsing:
    """验证命令解析。"""

    def test_execute_empty_line(self):
        """空行返回0。"""
        shell = InteractiveShell()
        assert shell.execute("") == 0
        assert shell.execute("   ") == 0

    def test_execute_unknown_command(self):
        """未知命令返回1。"""
        out = OutputCollector()
        shell = InteractiveShell(output_func=out)
        ret = shell.execute("nonexistent")
        assert ret == 1
        assert out.contains("未知命令")

    def test_execute_syntax_error(self):
        """语法错误返回1。"""
        out = OutputCollector()
        shell = InteractiveShell(output_func=out)
        # 引号不匹配
        ret = shell.execute('init "unclosed')
        assert ret == 1
        assert out.contains("语法错误")

    def test_execute_with_quotes(self):
        """支持引号参数。"""
        out = OutputCollector()
        shell = InteractiveShell(output_func=out)
        # report "my report.html" 应正确解析
        shell._orchestrator = None  # 未初始化
        ret = shell.execute('report "my report.html"')
        # 未初始化时 report 返回0（显示提示）
        assert ret == 0


# ---------------------------------------------------------------------------
# help 命令验证
# ---------------------------------------------------------------------------
class TestHelpCommand:
    """验证 help 命令。"""

    def test_help_returns_zero(self):
        """help 返回0。"""
        out = OutputCollector()
        shell = InteractiveShell(output_func=out)
        ret = shell.execute("help")
        assert ret == 0

    def test_help_lists_commands(self):
        """help 列出所有命令。"""
        out = OutputCollector()
        shell = InteractiveShell(output_func=out)
        shell.execute("help")
        for cmd in ["init", "run", "status", "metrics", "agents", "report", "exit"]:
            assert out.contains(cmd), f"help 未列出命令: {cmd}"


# ---------------------------------------------------------------------------
# init 命令验证
# ---------------------------------------------------------------------------
class TestInitCommand:
    """验证 init 命令。"""

    def test_init_default(self):
        """默认参数初始化。"""
        out = OutputCollector()
        shell = InteractiveShell(output_func=out)
        ret = shell.execute("init")
        assert ret == 0
        assert shell.orchestrator is not None
        assert out.contains("系统已初始化")

    def test_init_with_args(self):
        """带参数初始化。"""
        out = OutputCollector()
        shell = InteractiveShell(output_func=out)
        ret = shell.execute("init 2 3")
        assert ret == 0
        assert shell.orchestrator is not None
        assert out.contains("2 agents")

    def test_init_already_initialized(self):
        """重复初始化返回1。"""
        out = OutputCollector()
        shell = InteractiveShell(output_func=out)
        shell.execute("init")
        ret = shell.execute("init")
        assert ret == 1
        assert out.contains("已初始化")


# ---------------------------------------------------------------------------
# run 命令验证
# ---------------------------------------------------------------------------
class TestRunCommand:
    """验证 run 命令。"""

    def test_run_without_init(self):
        """未初始化时run返回1。"""
        out = OutputCollector()
        shell = InteractiveShell(output_func=out)
        ret = shell.execute("run 5")
        assert ret == 1
        assert out.contains("未初始化")

    def test_run_default(self):
        """默认运行1轮。"""
        out = OutputCollector()
        shell = InteractiveShell(output_func=out)
        shell.execute("init")
        ret = shell.execute("run")
        assert ret == 0
        assert out.contains("1 轮迭代")

    def test_run_multiple(self):
        """运行多轮。"""
        out = OutputCollector()
        shell = InteractiveShell(output_func=out)
        shell.execute("init 2 3")
        ret = shell.execute("run 3")
        assert ret == 0
        assert out.contains("3 轮迭代")

    def test_run_zero(self):
        """运行0轮返回1。"""
        out = OutputCollector()
        shell = InteractiveShell(output_func=out)
        shell.execute("init")
        ret = shell.execute("run 0")
        assert ret == 1


# ---------------------------------------------------------------------------
# status/metrics/agents 命令验证
# ---------------------------------------------------------------------------
class TestQueryCommands:
    """验证查询命令。"""

    def test_status_without_init(self):
        """未初始化时status返回0。"""
        out = OutputCollector()
        shell = InteractiveShell(output_func=out)
        ret = shell.execute("status")
        assert ret == 0
        assert out.contains("未初始化")

    def test_status_with_init(self):
        """初始化后status显示状态。"""
        out = OutputCollector()
        shell = InteractiveShell(output_func=out)
        shell.execute("init 2 3")
        ret = shell.execute("status")
        assert ret == 0
        assert out.contains("状态")

    def test_metrics_without_init(self):
        """未初始化时metrics返回0。"""
        out = OutputCollector()
        shell = InteractiveShell(output_func=out)
        ret = shell.execute("metrics")
        assert ret == 0
        assert out.contains("未初始化")

    def test_metrics_with_init(self):
        """初始化后metrics显示指标。"""
        out = OutputCollector()
        shell = InteractiveShell(output_func=out)
        shell.execute("init 2 3")
        shell.execute("run 2")
        ret = shell.execute("metrics")
        assert ret == 0
        assert out.contains("迭代次数")

    def test_agents_without_init(self):
        """未初始化时agents返回0。"""
        out = OutputCollector()
        shell = InteractiveShell(output_func=out)
        ret = shell.execute("agents")
        assert ret == 0
        assert out.contains("未初始化")

    def test_agents_with_init(self):
        """初始化后agents列出Agent。"""
        out = OutputCollector()
        shell = InteractiveShell(output_func=out)
        shell.execute("init 2 3")
        ret = shell.execute("agents")
        assert ret == 0
        assert out.contains("Agent")


# ---------------------------------------------------------------------------
# report 命令验证
# ---------------------------------------------------------------------------
class TestReportCommand:
    """验证 report 命令。"""

    def test_report_without_init(self):
        """未初始化时report返回0。"""
        out = OutputCollector()
        shell = InteractiveShell(output_func=out)
        ret = shell.execute("report")
        assert ret == 0
        assert out.contains("未初始化")

    def test_report_with_init(self, tmp_path):
        """初始化后report生成文件。"""
        out = OutputCollector()
        shell = InteractiveShell(output_func=out)
        shell.execute("init 2 3")
        shell.execute("run 2")
        path = str(tmp_path / "shell_report.html")
        ret = shell.execute(f"report {path}")
        assert ret == 0
        import os
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# exit 命令验证
# ---------------------------------------------------------------------------
class TestExitCommand:
    """验证 exit 命令。"""

    def test_exit_sets_running_false(self):
        """exit 设置 running 为 False。"""
        out = OutputCollector()
        shell = InteractiveShell(output_func=out)
        shell._running = True
        ret = shell.execute("exit")
        assert ret == 0
        assert shell.is_running is False
        assert out.contains("再见")

    def test_quit_same_as_exit(self):
        """quit 等同于 exit。"""
        out = OutputCollector()
        shell = InteractiveShell(output_func=out)
        shell._running = True
        ret = shell.execute("quit")
        assert ret == 0
        assert shell.is_running is False


# ---------------------------------------------------------------------------
# 非交互模式验证
# ---------------------------------------------------------------------------
class TestNonInteractiveMode:
    """验证 run_with_commands。"""

    def test_run_with_commands_basic(self):
        """基本命令序列。"""
        out = OutputCollector()
        shell = InteractiveShell(output_func=out)
        ret = shell.run_with_commands(["init 2 3", "run 2", "status", "exit"])
        assert ret == 0
        assert out.contains("系统已初始化")
        assert out.contains("2 轮迭代")

    def test_run_with_commands_stops_on_exit(self):
        """exit 后停止执行后续命令。"""
        out = OutputCollector()
        shell = InteractiveShell(output_func=out)
        ret = shell.run_with_commands(["exit", "init"])
        assert ret == 0
        # init 不应执行
        assert not out.contains("系统已初始化")

    def test_run_with_commands_error_propagates(self):
        """命令错误时返回非0。"""
        out = OutputCollector()
        shell = InteractiveShell(output_func=out)
        ret = shell.run_with_commands(["run 5"])
        # 未初始化时 run 返回1
        assert ret == 1

    def test_run_with_commands_cleans_up(self):
        """命令执行后清理资源。"""
        shell = InteractiveShell()
        shell.run_with_commands(["init 2 3", "exit"])
        assert shell.orchestrator is None


# ---------------------------------------------------------------------------
# 交互模式验证
# ---------------------------------------------------------------------------
class TestInteractiveMode:
    """验证 run（交互模式）。"""

    def test_run_exits_on_exit_command(self):
        """exit 命令退出REPL。"""
        out = OutputCollector()
        shell = InteractiveShell(
            input_func=FakeInput(["exit"]),
            output_func=out,
        )
        ret = shell.run()
        assert ret == 0
        assert out.contains("再见")

    def test_run_exits_on_eof(self):
        """EOF 退出REPL。"""
        out = OutputCollector()
        shell = InteractiveShell(
            input_func=FakeInput([]),
            output_func=out,
        )
        ret = shell.run()
        assert ret == 0

    def test_run_skips_empty_lines(self):
        """空行被跳过。"""
        out = OutputCollector()
        shell = InteractiveShell(
            input_func=FakeInput(["", "  ", "exit"]),
            output_func=out,
        )
        ret = shell.run()
        assert ret == 0

    def test_run_full_session(self):
        """完整会话。"""
        out = OutputCollector()
        shell = InteractiveShell(
            input_func=FakeInput(["init 2 3", "run 2", "status", "exit"]),
            output_func=out,
        )
        ret = shell.run()
        assert ret == 0
        assert out.contains("系统已初始化")
        assert out.contains("2 轮迭代")
        assert out.contains("状态")


# ---------------------------------------------------------------------------
# 错误处理验证
# ---------------------------------------------------------------------------
class TestErrorHandling:
    """验证错误处理。"""

    def test_command_error_doesnt_crash(self):
        """命令错误不崩溃Shell。"""
        out = OutputCollector()
        shell = InteractiveShell(
            input_func=FakeInput(["badcommand", "exit"]),
            output_func=out,
        )
        ret = shell.run()
        assert ret == 0
        assert out.contains("未知命令")

    def test_init_error_handled(self):
        """init 错误被处理。"""
        out = OutputCollector()
        shell = InteractiveShell(output_func=out)
        # 无效参数
        ret = shell.execute("init abc")
        assert ret == 1
        assert out.contains("失败") or out.contains("错误")


# ---------------------------------------------------------------------------
# 便捷函数验证
# ---------------------------------------------------------------------------
class TestConvenienceFunctions:
    """验证便捷函数。"""

    def test_run_shell_function_exists(self):
        """run_shell 函数存在。"""
        assert callable(run_shell)

    def test_run_shell_with_commands_exists(self):
        """run_shell_with_commands 函数存在。"""
        assert callable(run_shell_with_commands)

    def test_run_shell_with_commands_works(self):
        """run_shell_with_commands 可执行。"""
        ret = run_shell_with_commands(["init 2 3", "run 1", "exit"])
        assert ret == 0

    def test_run_shell_with_io_injection(self):
        """run_shell 支持IO注入。"""
        out = OutputCollector()
        ret = run_shell(
            input_func=FakeInput(["exit"]),
            output_func=out,
        )
        assert ret == 0
        assert out.contains("再见")


# ---------------------------------------------------------------------------
# CLI 集成验证
# ---------------------------------------------------------------------------
class TestCliIntegration:
    """验证CLI集成。"""

    def test_shell_subcommand_registered(self):
        """shell子命令已注册。"""
        from agent_whitepaper.stage_h.cli import build_parser

        parser = build_parser()
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                parser.parse_args(["shell", "--help"])
            except SystemExit:
                pass
        output = buf.getvalue()
        assert "shell" in output

    def test_cmd_shell_function_exists(self):
        """cmd_shell 函数存在。"""
        from agent_whitepaper.stage_h.cli import cmd_shell
        assert callable(cmd_shell)

    def test_cmd_shell_has_docstring(self):
        """cmd_shell 有docstring。"""
        from agent_whitepaper.stage_h.cli import cmd_shell
        assert cmd_shell.__doc__ is not None

    def test_shell_command_in_doctor_subcommands(self):
        """doctor命令检查的子命令列表包含shell。"""
        import contextlib
        import io

        from agent_whitepaper.stage_h.cli import cmd_doctor

        buf = io.StringIO()
        args = argparse.Namespace()
        with contextlib.redirect_stdout(buf):
            cmd_doctor(args)
        output = buf.getvalue()
        assert "agentwp shell" in output


# ---------------------------------------------------------------------------
# 源码结构验证
# ---------------------------------------------------------------------------
class TestShellSourceCode:
    """验证交互式Shell源码结构。"""

    def test_shell_module_exists(self):
        """shell.py 模块存在。"""
        from agent_whitepaper.stage_h import shell
        assert shell is not None

    def test_shell_class_exported_from_stage_h(self):
        """InteractiveShell 从 stage_h 包导出。"""
        import agent_whitepaper.stage_h as stage_h
        assert stage_h.InteractiveShell is InteractiveShell

    def test_run_shell_exported(self):
        """run_shell 从 stage_h 包导出。"""
        import agent_whitepaper.stage_h as stage_h
        assert stage_h.run_shell is run_shell

    def test_run_shell_with_commands_exported(self):
        """run_shell_with_commands 从 stage_h 包导出。"""
        import agent_whitepaper.stage_h as stage_h
        assert stage_h.run_shell_with_commands is run_shell_with_commands

    def test_module_has_first_principle_docstring(self):
        """模块docstring包含第一性原理。"""
        source = Path(
            inspect.getfile(InteractiveShell)
        ).read_text(encoding="utf-8")
        assert "第一性原理" in source
        assert "REPL" in source or "交互" in source

    def test_shell_class_has_first_principle(self):
        """InteractiveShell类docstring包含第一性原理。"""
        source = Path(
            inspect.getfile(InteractiveShell)
        ).read_text(encoding="utf-8")
        assert "第一性原理" in source

    def test_public_methods_have_type_hints(self):
        """公共方法有返回类型注解。"""
        for method_name in ["run", "run_with_commands", "execute"]:
            sig = inspect.signature(getattr(InteractiveShell, method_name))
            assert sig.return_annotation is not inspect.Signature.empty, (
                f"{method_name} 缺少返回类型注解"
            )

    def test_convenience_functions_have_type_hints(self):
        """便捷函数有返回类型注解。"""
        sig = inspect.signature(run_shell)
        assert sig.return_annotation is not inspect.Signature.empty

        sig = inspect.signature(run_shell_with_commands)
        assert sig.return_annotation is not inspect.Signature.empty

    def test_no_silent_except_exception_pass(self):
        """无静默 except Exception: pass。"""
        import ast
        source = Path(
            inspect.getfile(InteractiveShell)
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if not (isinstance(node.type, ast.Name) and node.type.id == "Exception"):
                continue
            body = node.body
            if len(body) == 1 and isinstance(body[0], ast.Pass):
                offenders.append(node.lineno)
        assert offenders == [], f"发现静默 except Exception: pass: {offenders}"

    def test_init_docstring_mentions_shell(self):
        """stage_h __init__.py docstring提及Shell。"""
        import agent_whitepaper.stage_h as stage_h
        init_source = Path(
            inspect.getfile(stage_h)
        ).read_text(encoding="utf-8")
        assert "InteractiveShell" in init_source
        assert "run_shell" in init_source
