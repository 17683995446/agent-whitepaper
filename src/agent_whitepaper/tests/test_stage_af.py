"""阶段AF测试：验证交互式配置向导。

验证内容：
1. WizardQuestion 问题定义和验证
2. ConfigWizard 交互式问答
3. ConfigWizard 非交互模式（run_with_answers）
4. ConfigWizard.save 保存YAML
5. run_wizard 便捷函数
6. CLI: agentwp wizard 子命令
7. 源码结构符合规范
"""

import argparse
import inspect
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pytest
import yaml

from agent_whitepaper.stage_h.cli import build_parser, cmd_wizard
from agent_whitepaper.stage_h.config import list_templates
from agent_whitepaper.stage_h.wizard import (
    ConfigWizard,
    WizardQuestion,
    run_wizard,
)


# ---------------------------------------------------------------------------
# WizardQuestion 验证
# ---------------------------------------------------------------------------
class TestWizardQuestion:
    """验证 WizardQuestion 类。"""

    def test_question_creation(self):
        """创建问题。"""
        q = WizardQuestion(
            key="grid_size",
            prompt="网格大小",
            default=5,
            help_text="网格边长",
        )
        assert q.key == "grid_size"
        assert q.prompt == "网格大小"
        assert q.default == 5

    def test_question_validate_no_validator(self):
        """无验证器时总是合法。"""
        q = WizardQuestion(key="x", prompt="X", default=1)
        ok, msg = q.validate("anything")
        assert ok is True
        assert msg == ""

    def test_question_validate_with_validator(self):
        """有验证器时调用验证器。"""
        def validator(value):
            try:
                v = int(value)
                if v < 0:
                    return False, "必须非负"
                return True, ""
            except ValueError:
                return False, "需要整数"

        q = WizardQuestion(
            key="count", prompt="数量", default=0, validator=validator
        )
        ok, _ = q.validate("5")
        assert ok is True
        ok, msg = q.validate("-1")
        assert ok is False
        assert "非负" in msg
        ok, msg = q.validate("abc")
        assert ok is False
        assert "整数" in msg


# ---------------------------------------------------------------------------
# ConfigWizard 非交互模式验证
# ---------------------------------------------------------------------------
class TestConfigWizardNonInteractive:
    """验证 ConfigWizard 非交互模式。"""

    def test_run_with_answers_returns_dict(self):
        """run_with_answers 返回字典。"""
        wizard = ConfigWizard()
        config = wizard.run_with_answers(
            ["default", "5", "3", "none", "agent_system.db", "42"]
        )
        assert isinstance(config, dict)

    def test_run_with_answers_uses_template(self):
        """基于模板构建配置。"""
        wizard = ConfigWizard()
        config = wizard.run_with_answers(
            ["quick-test", "3", "2", "10", ":memory:", "42"]
        )
        # quick-test 模板的值
        assert config["grid_size"] == 3
        assert config["n_agents"] == 2

    def test_run_with_answers_overrides_template(self):
        """用户答案覆盖模板值。"""
        wizard = ConfigWizard()
        config = wizard.run_with_answers(
            ["quick-test", "7", "4", "20", "test.db", "100"]
        )
        assert config["grid_size"] == 7
        assert config["n_agents"] == 4
        assert config["max_iterations"] == 20
        assert config["db_path"] == "test.db"
        assert config["seed"] == 100

    def test_run_with_answers_none_iterations(self):
        """none 表示无限迭代。"""
        wizard = ConfigWizard()
        config = wizard.run_with_answers(
            ["default", "5", "3", "none", "agent_system.db", "42"]
        )
        assert config["max_iterations"] is None

    def test_run_with_answers_wrong_count_raises(self):
        """答案数量不匹配抛出异常。"""
        wizard = ConfigWizard()
        with pytest.raises(ValueError):
            wizard.run_with_answers(["default", "5"])

    def test_run_with_answers_invalid_template_raises(self):
        """无效模板抛出异常。"""
        wizard = ConfigWizard()
        with pytest.raises(Exception):
            wizard.run_with_answers(
                ["nonexistent", "5", "3", "none", "db", "42"]
            )

    def test_run_with_answers_excludes_template_key(self):
        """结果不含 _template 元字段。"""
        wizard = ConfigWizard()
        config = wizard.run_with_answers(
            ["default", "5", "3", "none", "db", "42"]
        )
        assert "_template" not in config
        assert "_description" not in config


# ---------------------------------------------------------------------------
# ConfigWizard 交互模式验证
# ---------------------------------------------------------------------------
class TestConfigWizardInteractive:
    """验证 ConfigWizard 交互模式。"""

    def test_run_with_mock_input(self):
        """使用模拟input运行交互向导。"""
        answers = iter(["default", "5", "3", "none", "agent_system.db", "42"])
        outputs = []
        wizard = ConfigWizard(
            input_func=lambda _: next(answers),
            output_func=outputs.append,
        )
        config = wizard.run()
        assert config["grid_size"] == 5
        assert config["n_agents"] == 3
        assert config["max_iterations"] is None

    def test_run_uses_default_on_empty_input(self):
        """空输入使用默认值。"""
        answers = iter(["", "", "", "", "", ""])
        outputs = []
        wizard = ConfigWizard(
            input_func=lambda _: next(answers),
            output_func=outputs.append,
        )
        config = wizard.run()
        # 默认模板是 default，默认值来自问题定义
        assert config["grid_size"] == 5
        assert config["n_agents"] == 3

    def test_run_shows_help_on_question_mark(self):
        """输入 ? 显示帮助。"""
        answers = iter(["?", "default", "5", "3", "none", "db", "42"])
        outputs = []
        wizard = ConfigWizard(
            input_func=lambda _: next(answers),
            output_func=outputs.append,
        )
        wizard.run()
        help_output = " ".join(outputs)
        assert "帮助" in help_output

    def test_run_retries_on_invalid_input(self):
        """无效输入时重试。"""
        # 第一个问题先输入无效值，再输入有效值
        answers = iter([
            "invalid_template", "default",  # 模板：先错后对
            "5", "3", "none", "db", "42"
        ])
        outputs = []
        wizard = ConfigWizard(
            input_func=lambda _: next(answers),
            output_func=outputs.append,
        )
        config = wizard.run()
        assert config["grid_size"] == 5
        # 应该有"无效"提示
        assert any("无效" in o for o in outputs)

    def test_run_shows_summary(self):
        """运行结束显示配置摘要。"""
        answers = iter(["default", "5", "3", "none", "db", "42"])
        outputs = []
        wizard = ConfigWizard(
            input_func=lambda _: next(answers),
            output_func=outputs.append,
        )
        wizard.run()
        summary_output = " ".join(outputs)
        assert "配置摘要" in summary_output

    def test_run_shows_available_templates(self):
        """运行开始显示可用模板。"""
        answers = iter(["default", "5", "3", "none", "db", "42"])
        outputs = []
        wizard = ConfigWizard(
            input_func=lambda _: next(answers),
            output_func=outputs.append,
        )
        wizard.run()
        output = " ".join(outputs)
        assert "可用模板" in output
        for name in list_templates():
            assert name in output


# ---------------------------------------------------------------------------
# ConfigWizard.save 验证
# ---------------------------------------------------------------------------
class TestConfigWizardSave:
    """验证 ConfigWizard.save 方法。"""

    def test_save_creates_yaml_file(self):
        """save 创建YAML文件。"""
        wizard = ConfigWizard()
        config = wizard.run_with_answers(
            ["quick-test", "3", "2", "10", ":memory:", "42"]
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            path = f.name
        try:
            wizard.save(config, path)
            assert os.path.exists(path)
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            assert data["grid_size"] == 3
            assert data["n_agents"] == 2
        finally:
            os.unlink(path)

    def test_save_file_has_header(self):
        """保存的文件有头部注释。"""
        wizard = ConfigWizard()
        config = {"grid_size": 5, "n_agents": 3}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            path = f.name
        try:
            wizard.save(config, path)
            with open(path, encoding="utf-8") as f:
                content = f.read()
            assert "Agent Whitepaper" in content
            assert "向导" in content
        finally:
            os.unlink(path)

    def test_save_outputs_save_message(self):
        """save 输出保存消息。"""
        wizard = ConfigWizard()
        outputs = []
        wizard.output = outputs.append
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            path = f.name
        try:
            wizard.save({"grid_size": 5}, path)
            msg = " ".join(outputs)
            assert "已保存" in msg
            assert path in msg
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# run_wizard 便捷函数验证
# ---------------------------------------------------------------------------
class TestRunWizard:
    """验证 run_wizard 便捷函数。"""

    def test_run_wizard_exists(self):
        """run_wizard 函数存在。"""
        assert callable(run_wizard)

    def test_run_wizard_with_mock_input(self):
        """使用模拟input运行便捷函数。"""
        answers = iter(["default", "5", "3", "none", "db", "42"])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            path = f.name
        try:
            config = run_wizard(
                path,
                input_func=lambda _: next(answers),
                output_func=lambda _: None,
            )
            assert isinstance(config, dict)
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ---------------------------------------------------------------------------
# CLI: agentwp wizard 验证
# ---------------------------------------------------------------------------
class TestWizardCLI:
    """验证 agentwp wizard 子命令。"""

    def test_wizard_in_help(self):
        """agentwp --help 包含 wizard 子命令。"""
        parser = build_parser()
        buf = StringIO()
        with redirect_stdout(buf):
            with pytest.raises(SystemExit):
                parser.parse_args(["--help"])
        assert "wizard" in buf.getvalue()

    def test_wizard_subcommand_exists(self):
        """wizard 子命令存在。"""
        parser = build_parser()
        args = parser.parse_args(["wizard"])
        assert hasattr(args, "func")
        assert args.func == cmd_wizard

    def test_wizard_accepts_output_arg(self):
        """wizard 接受 -o 参数。"""
        parser = build_parser()
        args = parser.parse_args(["wizard", "-o", "my.yaml"])
        assert args.output == "my.yaml"

    def test_wizard_default_output(self):
        """wizard 默认输出 config.yaml。"""
        parser = build_parser()
        args = parser.parse_args(["wizard"])
        assert args.output == "config.yaml"

    def test_wizard_has_docstring(self):
        """cmd_wizard 有docstring。"""
        assert cmd_wizard.__doc__ is not None
        assert "向导" in cmd_wizard.__doc__ or "引导" in cmd_wizard.__doc__


# ---------------------------------------------------------------------------
# 通过 main() 调用验证
# ---------------------------------------------------------------------------
class TestWizardViaMain:
    """验证通过 main() 调用 wizard。"""

    def test_main_wizard_with_mock_input(self):
        """cmd_wizard 使用模拟input。"""
        answers = iter(["default", "5", "3", "none", "db", "42"])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            args = argparse.Namespace(
                output=path,
                _input_func=lambda _: next(answers),
                _output_func=lambda _: None,
            )
            ret = cmd_wizard(args)
            assert ret == 0
            assert os.path.exists(path)

    def test_main_wizard_via_subprocess(self):
        """通过子进程验证 agentwp wizard --help。"""
        env = os.environ.copy()
        src_path = str(
            Path(__file__).resolve().parent.parent.parent.parent / "src"
        )
        env["PYTHONPATH"] = src_path
        result = subprocess.run(
            [
                sys.executable, "-m", "agent_whitepaper.stage_h.cli",
                "wizard", "--help",
            ],
            capture_output=True, text=True, timeout=30, env=env,
        )
        assert result.returncode == 0
        assert "wizard" in result.stdout
        assert "--output" in result.stdout


# ---------------------------------------------------------------------------
# 源码结构验证
# ---------------------------------------------------------------------------
class TestWizardSourceCode:
    """验证向导源码结构。"""

    def test_wizard_module_exists(self):
        """wizard.py 模块存在。"""
        from agent_whitepaper.stage_h import wizard
        assert wizard is not None

    def test_config_wizard_class_exists(self):
        """ConfigWizard 类存在。"""
        assert callable(ConfigWizard)

    def test_wizard_question_class_exists(self):
        """WizardQuestion 类存在。"""
        assert callable(WizardQuestion)

    def test_run_wizard_function_exists(self):
        """run_wizard 函数存在。"""
        assert callable(run_wizard)

    def test_config_wizard_has_docstring(self):
        """ConfigWizard 有docstring。"""
        assert ConfigWizard.__doc__ is not None
        assert "向导" in ConfigWizard.__doc__ or "Guide" in ConfigWizard.__doc__

    def test_run_method_has_docstring(self):
        """run 方法有docstring。"""
        assert ConfigWizard.run.__doc__ is not None

    def test_save_method_has_docstring(self):
        """save 方法有docstring。"""
        assert ConfigWizard.save.__doc__ is not None

    def test_wizard_exported_from_stage_h(self):
        """向导组件从 stage_h 包导出。"""
        import agent_whitepaper.stage_h as stage_h
        assert stage_h.ConfigWizard is ConfigWizard
        assert stage_h.run_wizard is run_wizard

    def test_module_has_first_principle_docstring(self):
        """模块docstring包含第一性原理。"""
        source = Path(
            inspect.getfile(ConfigWizard)
        ).read_text(encoding="utf-8")
        assert "第一性原理" in source
        assert "Guide the user" in source

    def test_cli_imports_wizard(self):
        """CLI 导入 ConfigWizard。"""
        source = Path(
            inspect.getfile(build_parser)
        ).read_text(encoding="utf-8")
        assert "ConfigWizard" in source
        assert "cmd_wizard" in source

    def test_wizard_registered_in_parser(self):
        """wizard 在 parser 中注册。"""
        source = Path(
            inspect.getfile(build_parser)
        ).read_text(encoding="utf-8")
        assert "p_wizard" in source
        assert "wizard" in source
