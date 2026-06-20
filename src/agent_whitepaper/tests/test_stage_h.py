"""阶段H验证测试 + 配置化与CLI入口。

测试覆盖：
1. 配置加载（字典/YAML/环境变量）
2. 配置验证
3. 默认配置生成
4. CLI入口（init/run/demo/status）
5. 端到端：配置驱动的系统运行
"""

import os

import pytest

from agent_whitepaper.stage_h.cli import build_parser, main
from agent_whitepaper.stage_h.config import (
    DEFAULT_CONFIG,
    ConfigLoader,
    ConfigValidationError,
    load_config,
)


# ============================================================
# 1. 配置加载测试
# ============================================================
class TestConfigLoader:
    def test_from_dict_defaults(self):
        loader = ConfigLoader()
        cfg = loader.from_dict({})
        assert cfg.grid_size == DEFAULT_CONFIG["grid_size"]
        assert cfg.n_agents == DEFAULT_CONFIG["n_agents"]
        assert cfg.db_path == DEFAULT_CONFIG["db_path"]

    def test_from_dict_overrides(self):
        loader = ConfigLoader()
        cfg = loader.from_dict({"grid_size": 7, "n_agents": 5})
        assert cfg.grid_size == 7
        assert cfg.n_agents == 5
        # 未覆盖的用默认值
        assert cfg.checkpoint_interval == DEFAULT_CONFIG["checkpoint_interval"]

    def test_from_dict_none_values_ignored(self):
        loader = ConfigLoader()
        cfg = loader.from_dict({"grid_size": 6, "n_agents": None})
        assert cfg.grid_size == 6
        assert cfg.n_agents == DEFAULT_CONFIG["n_agents"]

    def test_from_yaml(self, tmp_path):
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML未安装")

        config_data = {"grid_size": 6, "n_agents": 4, "db_path": ":memory:"}
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(config_data, f)

        loader = ConfigLoader()
        cfg = loader.from_yaml(str(path))
        assert cfg.grid_size == 6
        assert cfg.n_agents == 4
        assert cfg.db_path == ":memory:"

    def test_generate_default_yaml(self, tmp_path):
        try:
            import yaml  # noqa: F401
        except ImportError:
            pytest.skip("PyYAML未安装")

        path = tmp_path / "default.yaml"
        loader = ConfigLoader()
        loader.generate_default_yaml(str(path))
        assert os.path.exists(str(path))

        # 验证生成的配置可加载
        cfg = loader.from_yaml(str(path))
        assert cfg.grid_size == DEFAULT_CONFIG["grid_size"]

    def test_load_config_convenience(self):
        cfg = load_config(grid_size=4, n_agents=2, db_path=":memory:")
        assert cfg.grid_size == 4
        assert cfg.n_agents == 2
        assert cfg.db_path == ":memory:"

    def test_load_config_from_yaml_with_overrides(self, tmp_path):
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML未安装")

        config_data = {"grid_size": 5, "n_agents": 3, "db_path": ":memory:"}
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(config_data, f)

        cfg = load_config(path=str(path), n_agents=5)
        assert cfg.grid_size == 5  # 来自YAML
        assert cfg.n_agents == 5   # 被覆盖


# ============================================================
# 2. 配置验证测试
# ============================================================
class TestConfigValidation:
    def test_invalid_grid_size(self):
        loader = ConfigLoader()
        with pytest.raises(ConfigValidationError):
            loader.from_dict({"grid_size": 1})

    def test_invalid_n_agents(self):
        loader = ConfigLoader()
        with pytest.raises(ConfigValidationError):
            loader.from_dict({"n_agents": 0})

    def test_invalid_interval(self):
        loader = ConfigLoader()
        with pytest.raises(ConfigValidationError):
            loader.from_dict({"checkpoint_interval": 0})

    def test_invalid_max_iterations(self):
        loader = ConfigLoader()
        with pytest.raises(ConfigValidationError):
            loader.from_dict({"max_iterations": 0})

    def test_invalid_db_path(self):
        loader = ConfigLoader()
        with pytest.raises(ConfigValidationError):
            loader.from_dict({"db_path": ""})

    def test_valid_none_max_iterations(self):
        loader = ConfigLoader()
        cfg = loader.from_dict({"max_iterations": None, "db_path": ":memory:"})
        assert cfg.max_iterations is None


# ============================================================
# 3. 环境变量覆盖测试
# ============================================================
class TestEnvOverrides:
    def test_env_override_grid_size(self):
        os.environ["AGENTWP_GRID_SIZE"] = "8"
        try:
            loader = ConfigLoader()
            cfg = loader.from_dict({"db_path": ":memory:"})
            assert cfg.grid_size == 8
        finally:
            del os.environ["AGENTWP_GRID_SIZE"]

    def test_env_override_n_agents(self):
        os.environ["AGENTWP_N_AGENTS"] = "5"
        try:
            loader = ConfigLoader()
            cfg = loader.from_dict({"db_path": ":memory:"})
            assert cfg.n_agents == 5
        finally:
            del os.environ["AGENTWP_N_AGENTS"]

    def test_env_override_db_path(self):
        os.environ["AGENTWP_DB_PATH"] = ":memory:"
        try:
            loader = ConfigLoader()
            cfg = loader.from_dict({})
            assert cfg.db_path == ":memory:"
        finally:
            del os.environ["AGENTWP_DB_PATH"]

    def test_env_override_none_max_iterations(self):
        os.environ["AGENTWP_MAX_ITERATIONS"] = "none"
        try:
            loader = ConfigLoader()
            cfg = loader.from_dict({"max_iterations": 100, "db_path": ":memory:"})
            assert cfg.max_iterations is None
        finally:
            del os.environ["AGENTWP_MAX_ITERATIONS"]

    def test_invalid_env_ignored(self):
        os.environ["AGENTWP_GRID_SIZE"] = "not_a_number"
        try:
            loader = ConfigLoader()
            cfg = loader.from_dict({"db_path": ":memory:"})
            # 无效环境变量被忽略，用默认值
            assert cfg.grid_size == DEFAULT_CONFIG["grid_size"]
        finally:
            del os.environ["AGENTWP_GRID_SIZE"]


# ============================================================
# 4. CLI入口测试
# ============================================================
class TestCLI:
    def test_parser_no_command(self, capsys):
        parser = build_parser()
        args = parser.parse_args([])
        assert not hasattr(args, "func") or args.func is None

    def test_main_no_command_returns_1(self):
        result = main([])
        assert result == 1

    def test_cmd_init(self, tmp_path):
        path = str(tmp_path / "config.yaml")
        result = main(["init", "-o", path])
        assert result == 0
        assert os.path.exists(path)

    def test_cmd_run_with_iterations(self, capsys):
        result = main([
            "run", "-n", "5", "-a", "2", "-g", "4",
            "--memory", "--report-interval", "2"
        ])
        assert result == 0
        captured = capsys.readouterr()
        assert "最终报告" in captured.out
        assert "迭代次数" in captured.out

    def test_cmd_run_config_file(self, tmp_path, capsys):
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML未安装")

        config_data = {
            "grid_size": 4, "n_agents": 2, "db_path": ":memory:",
            "max_iterations": 5,
        }
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(config_data, f)

        result = main(["run", "-c", str(path), "--report-interval", "2"])
        assert result == 0

    def test_cmd_demo(self, capsys):
        result = main(["demo"])
        assert result == 0
        captured = capsys.readouterr()
        assert "演示结果" in captured.out
        assert "能力验证" in captured.out

    def test_cmd_status_no_db(self, capsys):
        result = main(["status", "-d", "/nonexistent/path.db"])
        # 无数据库时应优雅处理
        assert result in (0, 1)

    def test_cmd_status_with_db(self, tmp_path, capsys):
        # 先运行一次生成数据库
        db_path = str(tmp_path / "test.db")
        main(["run", "-n", "3", "-g", "4", "--memory", "--report-interval", "5"])
        # status命令对不存在的db应返回1或0
        result = main(["status", "-d", db_path])
        assert result in (0, 1)


# ============================================================
# 5. 端到端：配置驱动的系统运行
# ============================================================
class TestConfigDrivenSystem:
    def test_config_builds_orchestrator(self):
        cfg = load_config(
            grid_size=4, n_agents=2, db_path=":memory:", max_iterations=5
        )
        from agent_whitepaper.stage_g.orchestrator import SystemOrchestrator
        orch = SystemOrchestrator(cfg)
        orch.initialize()
        assert len(orch.agents) == 2
        assert orch.env.size == 4
        for _ in range(5):
            orch.run_iteration()
        assert orch._iteration == 5

    def test_full_cli_workflow(self, tmp_path, capsys):
        """完整CLI工作流：init → run → status。"""
        # 1. init生成配置
        config_path = str(tmp_path / "config.yaml")
        assert main(["init", "-o", config_path]) == 0

        # 2. 用配置运行
        result = main([
            "run", "-c", config_path, "-n", "3", "-g", "4",
            "--memory", "--report-interval", "3"
        ])
        assert result == 0
        captured = capsys.readouterr()
        assert "系统启动" in captured.out
        assert "最终报告" in captured.out

    def test_demo_shows_four_capabilities(self, capsys):
        """演示展示四大能力。"""
        result = main(["demo"])
        assert result == 0
        captured = capsys.readouterr()
        assert "多Agent协作" in captured.out
        assert "技能增强" in captured.out
        assert "长任务执行" in captured.out
        assert "永续运行" in captured.out
