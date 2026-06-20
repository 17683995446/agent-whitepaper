"""阶段AC测试：验证配置模板系统（agentwp init --template）。

验证内容：
1. CONFIG_TEMPLATES 注册表包含5个预置模板
2. list_templates() 返回所有模板名称
3. get_template() 返回正确配置 / 对未知模板抛异常
4. ConfigLoader.generate_default_yaml 支持 template 参数
5. agentwp init --template <名称> 生成正确配置
6. agentwp init --list-templates 列出所有模板
7. 生成的YAML文件可被 from_yaml 正确加载
8. 源码结构符合规范
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

from agent_whitepaper.stage_h.cli import build_parser, cmd_init, main
from agent_whitepaper.stage_h.config import (
    CONFIG_TEMPLATES,
    DEFAULT_CONFIG,
    ConfigLoader,
    ConfigValidationError,
    get_template,
    list_templates,
)


# ---------------------------------------------------------------------------
# CONFIG_TEMPLATES 注册表验证
# ---------------------------------------------------------------------------
class TestConfigTemplatesRegistry:
    """验证 CONFIG_TEMPLATES 注册表。"""

    def test_templates_registry_exists(self):
        """CONFIG_TEMPLATES 存在且为字典。"""
        assert isinstance(CONFIG_TEMPLATES, dict)

    def test_templates_count_is_five(self):
        """预置模板数量为5个。"""
        expected = {"default", "quick-test", "production", "multi-agent", "long-run"}
        assert set(CONFIG_TEMPLATES.keys()) == expected

    def test_each_template_has_description(self):
        """每个模板都包含 _description 元字段。"""
        for name, tpl in CONFIG_TEMPLATES.items():
            assert "_description" in tpl, f"模板 '{name}' 缺少 _description"
            assert isinstance(tpl["_description"], str)
            assert len(tpl["_description"]) > 0

    def test_each_template_has_required_keys(self):
        """每个模板包含所有必需配置键。"""
        required_keys = {
            "grid_size", "n_agents", "db_path", "checkpoint_interval",
            "skill_abstract_interval", "meta_learn_interval",
            "distill_interval", "evolve_interval", "coalition_interval",
            "max_iterations", "seed",
        }
        for name, tpl in CONFIG_TEMPLATES.items():
            actual_keys = set(tpl.keys()) - {"_description"}
            missing = required_keys - actual_keys
            assert not missing, f"模板 '{name}' 缺少键: {missing}"

    def test_default_template_matches_default_config(self):
        """default 模板与 DEFAULT_CONFIG 一致。"""
        tpl = get_template("default")
        for key, value in DEFAULT_CONFIG.items():
            assert tpl[key] == value, f"default 模板 {key} 不匹配"

    def test_quick_test_template_uses_memory_db(self):
        """quick-test 模板使用内存数据库。"""
        tpl = get_template("quick-test")
        assert tpl["db_path"] == ":memory:"

    def test_quick_test_template_has_max_iterations(self):
        """quick-test 模板有最大迭代限制。"""
        tpl = get_template("quick-test")
        assert tpl["max_iterations"] is not None
        assert tpl["max_iterations"] > 0

    def test_production_template_persists_to_disk(self):
        """production 模板使用磁盘数据库。"""
        tpl = get_template("production")
        assert tpl["db_path"] != ":memory:"

    def test_production_template_runs_indefinitely(self):
        """production 模板可无限运行。"""
        tpl = get_template("production")
        assert tpl["max_iterations"] is None

    def test_multi_agent_template_has_more_agents(self):
        """multi-agent 模板Agent数多于默认。"""
        tpl = get_template("multi-agent")
        assert tpl["n_agents"] >= DEFAULT_CONFIG["n_agents"]

    def test_long_run_template_has_large_intervals(self):
        """long-run 模板间隔大于默认。"""
        tpl = get_template("long-run")
        assert tpl["checkpoint_interval"] >= DEFAULT_CONFIG["checkpoint_interval"]


# ---------------------------------------------------------------------------
# list_templates() 验证
# ---------------------------------------------------------------------------
class TestListTemplates:
    """验证 list_templates() 函数。"""

    def test_list_templates_returns_list(self):
        """list_templates 返回列表。"""
        result = list_templates()
        assert isinstance(result, list)

    def test_list_templates_contains_all_names(self):
        """list_templates 包含所有模板名。"""
        result = list_templates()
        assert "default" in result
        assert "quick-test" in result
        assert "production" in result
        assert "multi-agent" in result
        assert "long-run" in result

    def test_list_templates_count(self):
        """list_templates 返回5个模板。"""
        assert len(list_templates()) == 5


# ---------------------------------------------------------------------------
# get_template() 验证
# ---------------------------------------------------------------------------
class TestGetTemplate:
    """验证 get_template() 函数。"""

    def test_get_template_returns_dict(self):
        """get_template 返回字典。"""
        tpl = get_template("default")
        assert isinstance(tpl, dict)

    def test_get_template_excludes_description(self):
        """get_template 返回的字典不含 _description。"""
        tpl = get_template("default")
        assert "_description" not in tpl

    def test_get_template_returns_copy(self):
        """get_template 返回副本，修改不影响原模板。"""
        tpl1 = get_template("default")
        tpl1["grid_size"] = 999
        tpl2 = get_template("default")
        assert tpl2["grid_size"] != 999

    def test_get_template_unknown_raises(self):
        """未知模板名抛出 ConfigValidationError。"""
        with pytest.raises(ConfigValidationError):
            get_template("nonexistent-template")

    def test_get_template_error_message_lists_available(self):
        """错误消息列出可用模板。"""
        with pytest.raises(ConfigValidationError) as exc_info:
            get_template("nonexistent")
        msg = str(exc_info.value)
        assert "default" in msg
        assert "quick-test" in msg

    def test_get_template_quick_test_values(self):
        """quick-test 模板值正确。"""
        tpl = get_template("quick-test")
        assert tpl["grid_size"] == 3
        assert tpl["n_agents"] == 2
        assert tpl["max_iterations"] == 10


# ---------------------------------------------------------------------------
# ConfigLoader.generate_default_yaml(template=) 验证
# ---------------------------------------------------------------------------
class TestGenerateYamlWithTemplate:
    """验证 ConfigLoader.generate_default_yaml 支持 template 参数。"""

    def test_generate_default_yaml_without_template(self):
        """无模板时生成默认配置。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            path = f.name
        try:
            loader = ConfigLoader()
            loader.generate_default_yaml(path)
            with open(path, encoding="utf-8") as f:
                content = f.read()
            assert "grid_size" in content
            data = yaml.safe_load(content)
            assert data["grid_size"] == DEFAULT_CONFIG["grid_size"]
        finally:
            os.unlink(path)

    def test_generate_yaml_with_quick_test_template(self):
        """使用 quick-test 模板生成配置。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            path = f.name
        try:
            loader = ConfigLoader()
            loader.generate_default_yaml(path, template="quick-test")
            with open(path, encoding="utf-8") as f:
                content = f.read()
            assert "quick-test" in content
            data = yaml.safe_load(content)
            assert data["grid_size"] == 3
            assert data["n_agents"] == 2
            assert data["db_path"] == ":memory:"
        finally:
            os.unlink(path)

    def test_generate_yaml_with_production_template(self):
        """使用 production 模板生成配置。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            path = f.name
        try:
            loader = ConfigLoader()
            loader.generate_default_yaml(path, template="production")
            with open(path, encoding="utf-8") as f:
                content = f.read()
            data = yaml.safe_load(content)
            assert data["grid_size"] == 7
            assert data["n_agents"] == 5
        finally:
            os.unlink(path)

    def test_generate_yaml_header_contains_template_name(self):
        """生成的YAML头部包含模板名。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            path = f.name
        try:
            loader = ConfigLoader()
            loader.generate_default_yaml(path, template="long-run")
            with open(path, encoding="utf-8") as f:
                content = f.read()
            assert "long-run" in content
            assert "模板" in content
        finally:
            os.unlink(path)

    def test_generate_yaml_with_unknown_template_raises(self):
        """未知模板抛出异常。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            path = f.name
        try:
            loader = ConfigLoader()
            with pytest.raises(ConfigValidationError):
                loader.generate_default_yaml(path, template="nonexistent")
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_generated_yaml_is_loadable_by_from_yaml(self):
        """生成的YAML可被 from_yaml 正确加载。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            path = f.name
        try:
            loader = ConfigLoader()
            loader.generate_default_yaml(path, template="quick-test")
            config = loader.from_yaml(path)
            assert config.grid_size == 3
            assert config.n_agents == 2
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# CLI: agentwp init --template 验证
# ---------------------------------------------------------------------------
class TestInitWithTemplate:
    """验证 agentwp init --template 命令。"""

    def test_init_parser_accepts_template_arg(self):
        """init 子命令接受 --template 参数。"""
        parser = build_parser()
        args = parser.parse_args(["init", "--template", "quick-test"])
        assert args.template == "quick-test"

    def test_init_parser_accepts_short_template_arg(self):
        """init 子命令接受 -t 短参数。"""
        parser = build_parser()
        args = parser.parse_args(["init", "-t", "production"])
        assert args.template == "production"

    def test_init_parser_template_default_none(self):
        """--template 默认为 None。"""
        parser = build_parser()
        args = parser.parse_args(["init"])
        assert args.template is None

    def test_init_parser_rejects_invalid_template(self):
        """无效模板名被 argparse choices 拒绝。"""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["init", "--template", "nonexistent"])

    def test_init_with_template_generates_correct_config(self):
        """init --template 生成正确的配置文件。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            path = f.name
        try:
            args = argparse.Namespace(
                output=path, template="quick-test", list_templates=False
            )
            buf = StringIO()
            with redirect_stdout(buf):
                ret = cmd_init(args)
            assert ret == 0
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            assert data["grid_size"] == 3
            assert data["db_path"] == ":memory:"
        finally:
            os.unlink(path)

    def test_init_without_template_uses_default(self):
        """无 --template 时使用默认配置。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            path = f.name
        try:
            args = argparse.Namespace(
                output=path, template=None, list_templates=False
            )
            buf = StringIO()
            with redirect_stdout(buf):
                ret = cmd_init(args)
            assert ret == 0
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            assert data["grid_size"] == DEFAULT_CONFIG["grid_size"]
        finally:
            os.unlink(path)

    def test_init_output_contains_template_label(self):
        """init --template 输出包含模板名。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            path = f.name
        try:
            args = argparse.Namespace(
                output=path, template="production", list_templates=False
            )
            buf = StringIO()
            with redirect_stdout(buf):
                cmd_init(args)
            output = buf.getvalue()
            assert "production" in output
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# CLI: agentwp init --list-templates 验证
# ---------------------------------------------------------------------------
class TestInitListTemplates:
    """验证 agentwp init --list-templates 命令。"""

    def test_init_parser_accepts_list_templates(self):
        """init 子命令接受 --list-templates 参数。"""
        parser = build_parser()
        args = parser.parse_args(["init", "--list-templates"])
        assert args.list_templates is True

    def test_list_templates_default_false(self):
        """--list-templates 默认为 False。"""
        parser = build_parser()
        args = parser.parse_args(["init"])
        assert args.list_templates is False

    def test_list_templates_outputs_all_templates(self):
        """--list-templates 输出所有模板名。"""
        args = argparse.Namespace(
            output="config.yaml", template=None, list_templates=True
        )
        buf = StringIO()
        with redirect_stdout(buf):
            ret = cmd_init(args)
        assert ret == 0
        output = buf.getvalue()
        for name in ["default", "quick-test", "production", "multi-agent", "long-run"]:
            assert name in output

    def test_list_templates_outputs_descriptions(self):
        """--list-templates 输出模板描述。"""
        args = argparse.Namespace(
            output="config.yaml", template=None, list_templates=True
        )
        buf = StringIO()
        with redirect_stdout(buf):
            cmd_init(args)
        output = buf.getvalue()
        assert "快速测试" in output or "默认配置" in output

    def test_list_templates_does_not_create_file(self):
        """--list-templates 不创建文件。"""
        test_path = "/tmp/_test_should_not_exist_ac.yaml"
        if os.path.exists(test_path):
            os.unlink(test_path)
        args = argparse.Namespace(
            output=test_path, template=None, list_templates=True
        )
        buf = StringIO()
        with redirect_stdout(buf):
            cmd_init(args)
        assert not os.path.exists(test_path)


# ---------------------------------------------------------------------------
# 通过 main() 调用验证
# ---------------------------------------------------------------------------
class TestInitViaMain:
    """验证通过 main() 调用 init。"""

    def test_main_init_with_template(self):
        """main(['init', '--template', ...]) 正常工作。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            buf = StringIO()
            with redirect_stdout(buf):
                ret = main(["init", "--template", "quick-test", "-o", path])
            assert ret == 0
            assert os.path.exists(path)

    def test_main_init_list_templates(self):
        """main(['init', '--list-templates']) 正常工作。"""
        buf = StringIO()
        with redirect_stdout(buf):
            ret = main(["init", "--list-templates"])
        assert ret == 0
        assert "default" in buf.getvalue()

    def test_main_init_via_subprocess(self):
        """通过子进程验证 agentwp init --template。"""
        env = os.environ.copy()
        src_path = str(
            Path(__file__).resolve().parent.parent.parent.parent / "src"
        )
        env["PYTHONPATH"] = src_path
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            result = subprocess.run(
                [
                    sys.executable, "-m", "agent_whitepaper.stage_h.cli",
                    "init", "--template", "quick-test", "-o", path,
                ],
                capture_output=True, text=True, timeout=30, env=env,
            )
            assert result.returncode == 0
            assert os.path.exists(path)
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            assert data["grid_size"] == 3


# ---------------------------------------------------------------------------
# 源码结构验证
# ---------------------------------------------------------------------------
class TestConfigTemplateSourceCode:
    """验证配置模板系统源码结构。"""

    def test_config_templates_defined(self):
        """CONFIG_TEMPLATES 在 config.py 中定义。"""
        source = Path(
            inspect.getfile(ConfigLoader)
        ).read_text(encoding="utf-8")
        assert "CONFIG_TEMPLATES" in source

    def test_list_templates_function_defined(self):
        """list_templates 函数在 config.py 中定义。"""
        source = Path(
            inspect.getfile(ConfigLoader)
        ).read_text(encoding="utf-8")
        assert "def list_templates" in source

    def test_get_template_function_defined(self):
        """get_template 函数在 config.py 中定义。"""
        source = Path(
            inspect.getfile(ConfigLoader)
        ).read_text(encoding="utf-8")
        assert "def get_template" in source

    def test_generate_default_yaml_has_template_param(self):
        """generate_default_yaml 方法接受 template 参数。"""
        source = Path(
            inspect.getfile(ConfigLoader)
        ).read_text(encoding="utf-8")
        assert "template" in source

    def test_cli_init_has_template_arg(self):
        """CLI 的 init 子命令注册了 --template 参数。"""
        source = Path(
            inspect.getfile(build_parser)
        ).read_text(encoding="utf-8")
        assert "--template" in source
        assert "--list-templates" in source

    def test_cmd_init_has_docstring(self):
        """cmd_init 有docstring。"""
        assert cmd_init.__doc__ is not None
        assert "配置" in cmd_init.__doc__ or "模板" in cmd_init.__doc__
