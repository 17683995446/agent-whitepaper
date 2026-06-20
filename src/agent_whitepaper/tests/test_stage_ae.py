"""阶段AE测试：验证便捷API。

验证内容：
1. quick_start 一行启动系统
2. train_agent 一行训练Agent
3. run_demo 一行运行演示
4. create_system 从模板创建系统
5. get_status 查询运行状态
6. list_available_templates 列出模板
7. 便捷API从顶层包导出
8. 源码结构符合规范
"""

import inspect
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from agent_whitepaper import (
    SelfDrivenAgent,
    SystemOrchestrator,
    create_system,
    get_status,
    list_available_templates,
    quick_start,
    run_demo,
    train_agent,
)
from agent_whitepaper.stage_h.convenience import (
    create_system as _create_system,
)
from agent_whitepaper.stage_h.convenience import (
    get_status as _get_status,
)
from agent_whitepaper.stage_h.convenience import (
    list_available_templates as _list_available_templates,
)
from agent_whitepaper.stage_h.convenience import (
    quick_start as _quick_start,
)
from agent_whitepaper.stage_h.convenience import (
    run_demo as _run_demo,
)
from agent_whitepaper.stage_h.convenience import (
    train_agent as _train_agent,
)


# ---------------------------------------------------------------------------
# quick_start 验证
# ---------------------------------------------------------------------------
class TestQuickStart:
    """验证 quick_start 函数。"""

    def test_quick_start_returns_dict(self):
        """quick_start 返回字典。"""
        report = quick_start(iterations=3, n_agents=2, grid_size=3)
        assert isinstance(report, dict)

    def test_quick_start_report_has_iteration(self):
        """报告包含迭代次数。"""
        report = quick_start(iterations=5, n_agents=2, grid_size=3)
        assert report["iteration"] == 5

    def test_quick_start_report_has_status(self):
        """报告包含状态。"""
        report = quick_start(iterations=3, n_agents=2, grid_size=3)
        assert "status" in report

    def test_quick_start_report_has_latest_snapshot(self):
        """报告包含最新快照。"""
        report = quick_start(iterations=3, n_agents=2, grid_size=3)
        assert report["latest_snapshot"] is not None

    def test_quick_start_report_has_agent_count(self):
        """报告包含Agent数量。"""
        report = quick_start(iterations=3, n_agents=2, grid_size=3)
        assert report["agent_count"] == 2

    def test_quick_start_uses_memory_db_by_default(self):
        """默认使用内存数据库。"""
        # 不应产生文件
        report = quick_start(iterations=3, n_agents=2, grid_size=3)
        assert isinstance(report, dict)

    def test_quick_start_with_seed_reproducible(self):
        """相同seed产生相同迭代数。"""
        r1 = quick_start(iterations=3, n_agents=2, grid_size=3, seed=42)
        r2 = quick_start(iterations=3, n_agents=2, grid_size=3, seed=42)
        assert r1["iteration"] == r2["iteration"]

    def test_quick_start_accepts_extra_kwargs(self):
        """接受额外配置参数。"""
        report = quick_start(
            iterations=3, n_agents=2, grid_size=3,
            skill_abstract_interval=2,
            meta_learn_interval=3,
        )
        assert isinstance(report, dict)


# ---------------------------------------------------------------------------
# train_agent 验证
# ---------------------------------------------------------------------------
class TestTrainAgent:
    """验证 train_agent 函数。"""

    def test_train_agent_returns_agent(self):
        """train_agent 返回 SelfDrivenAgent。"""
        agent = train_agent(n_episodes=10, grid_size=3)
        assert isinstance(agent, SelfDrivenAgent)

    def test_train_agent_runs_episodes(self):
        """训练后Agent有情节数。"""
        agent = train_agent(n_episodes=20, grid_size=3)
        assert agent.stats.total_episodes == 20

    def test_train_agent_q_table_has_entries(self):
        """训练后Q表有条目。"""
        agent = train_agent(n_episodes=20, grid_size=3)
        assert agent.q_table.n_states == 9  # 3x3

    def test_train_agent_with_custom_params(self):
        """自定义参数训练。"""
        agent = train_agent(
            n_episodes=10,
            grid_size=4,
            max_steps=50,
            learning_rate=0.2,
            gamma=0.95,
            seed=123,
        )
        assert agent.q_table.n_states == 16  # 4x4
        assert agent.q_table.n_actions == 4

    def test_train_agent_stats_available(self):
        """训练后可查看统计。"""
        agent = train_agent(n_episodes=10, grid_size=3)
        assert hasattr(agent.stats, "success_rate")
        assert hasattr(agent.stats, "total_episodes")

    def test_train_agent_default_grid_size(self):
        """默认网格大小为5。"""
        agent = train_agent(n_episodes=5)
        assert agent.q_table.n_states == 25  # 5x5


# ---------------------------------------------------------------------------
# run_demo 验证
# ---------------------------------------------------------------------------
class TestRunDemo:
    """验证 run_demo 函数。"""

    def test_run_demo_returns_dict(self):
        """run_demo 返回字典。"""
        report = run_demo(n_iterations=5, n_agents=2, grid_size=3)
        assert isinstance(report, dict)

    def test_run_demo_report_has_iteration(self):
        """报告包含迭代次数。"""
        report = run_demo(n_iterations=5, n_agents=2, grid_size=3)
        assert report["iteration"] == 5

    def test_run_demo_report_has_tasks_completed(self):
        """报告包含完成任务数。"""
        report = run_demo(n_iterations=5, n_agents=2, grid_size=3)
        assert "tasks_completed" in report

    def test_run_demo_report_has_agent_count(self):
        """报告包含Agent数量。"""
        report = run_demo(n_iterations=5, n_agents=3, grid_size=3)
        assert report["agent_count"] == 3


# ---------------------------------------------------------------------------
# create_system 验证
# ---------------------------------------------------------------------------
class TestCreateSystem:
    """验证 create_system 函数。"""

    def test_create_system_returns_orchestrator(self):
        """create_system 返回 SystemOrchestrator。"""
        orch = create_system(template="quick-test")
        assert isinstance(orch, SystemOrchestrator)
        orch.stop("test")

    def test_create_system_default_template(self):
        """默认使用default模板。"""
        orch = create_system()
        assert isinstance(orch, SystemOrchestrator)
        orch.stop("test")

    def test_create_system_quick_test_template(self):
        """使用quick-test模板。"""
        orch = create_system(template="quick-test")
        assert orch.config.grid_size == 3
        assert orch.config.n_agents == 2
        orch.stop("test")

    def test_create_system_production_template(self):
        """使用production模板。"""
        orch = create_system(template="production")
        assert orch.config.grid_size == 7
        assert orch.config.n_agents == 5
        orch.stop("test")

    def test_create_system_with_overrides(self):
        """覆盖模板参数。"""
        orch = create_system(template="quick-test", n_agents=4)
        assert orch.config.n_agents == 4
        orch.stop("test")

    def test_create_system_can_run_iterations(self):
        """创建后可运行迭代。"""
        orch = create_system(template="quick-test")
        orch.run_iteration()
        assert orch._iteration == 1
        orch.stop("test")

    def test_create_system_unknown_template_raises(self):
        """未知模板抛出异常。"""
        from agent_whitepaper.stage_h.config import ConfigValidationError
        with pytest.raises(ConfigValidationError):
            create_system(template="nonexistent")


# ---------------------------------------------------------------------------
# get_status 验证
# ---------------------------------------------------------------------------
class TestGetStatus:
    """验证 get_status 函数。"""

    def test_get_status_nonexistent_db(self):
        """不存在的数据库返回空状态。"""
        status = get_status("/tmp/nonexistent_db_ae.db")
        assert isinstance(status, dict)
        assert status["exists"] is False
        assert status["checkpoints"] == []
        assert status["logs"] == []

    def test_get_status_returns_db_path(self):
        """返回包含db_path的字典。"""
        status = get_status("/tmp/nonexistent_db_ae.db")
        assert status["db_path"] == "/tmp/nonexistent_db_ae.db"

    def test_get_status_from_real_db(self):
        """从真实数据库获取状态。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            # 创建数据库并写入检查点
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "CREATE TABLE checkpoints ("
                "id INTEGER PRIMARY KEY, created_at TEXT, description TEXT)"
            )
            cursor.execute(
                "INSERT INTO checkpoints (created_at, description) VALUES (?, ?)",
                ("2024-01-01 12:00:00", "测试检查点"),
            )
            cursor.execute(
                "CREATE TABLE run_log ("
                "id INTEGER PRIMARY KEY, timestamp TEXT, "
                "event_type TEXT, message TEXT)"
            )
            cursor.execute(
                "INSERT INTO run_log (timestamp, event_type, message) "
                "VALUES (?, ?, ?)",
                ("2024-01-01 12:01:00", "start", "系统启动"),
            )
            conn.commit()
            conn.close()

            status = get_status(db_path)
            assert status["exists"] is True
            assert len(status["checkpoints"]) == 1
            assert status["checkpoints"][0]["description"] == "测试检查点"
            assert len(status["logs"]) == 1
            assert status["logs"][0]["event_type"] == "start"

    def test_get_status_empty_db(self):
        """空数据库（表不存在）返回空状态。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "empty.db")
            conn = sqlite3.connect(db_path)
            conn.close()
            status = get_status(db_path)
            # 数据库存在但表不存在
            assert status["checkpoints"] == []
            assert status["logs"] == []


# ---------------------------------------------------------------------------
# list_available_templates 验证
# ---------------------------------------------------------------------------
class TestListAvailableTemplates:
    """验证 list_available_templates 函数。"""

    def test_returns_dict(self):
        """返回字典。"""
        result = list_available_templates()
        assert isinstance(result, dict)

    def test_contains_all_templates(self):
        """包含所有5个模板。"""
        result = list_available_templates()
        assert "default" in result
        assert "quick-test" in result
        assert "production" in result
        assert "multi-agent" in result
        assert "long-run" in result

    def test_values_are_descriptions(self):
        """值是描述字符串。"""
        result = list_available_templates()
        for name, desc in result.items():
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_quick_test_description(self):
        """quick-test模板有描述。"""
        result = list_available_templates()
        assert "快速测试" in result["quick-test"]


# ---------------------------------------------------------------------------
# 顶层包导出验证
# ---------------------------------------------------------------------------
class TestTopLevelExport:
    """验证便捷API从顶层包导出。"""

    def test_quick_start_from_top_level(self):
        """quick_start 从顶层包可用。"""
        assert _quick_start is quick_start

    def test_train_agent_from_top_level(self):
        """train_agent 从顶层包可用。"""
        assert _train_agent is train_agent

    def test_run_demo_from_top_level(self):
        """run_demo 从顶层包可用。"""
        assert _run_demo is run_demo

    def test_create_system_from_top_level(self):
        """create_system 从顶层包可用。"""
        assert _create_system is create_system

    def test_get_status_from_top_level(self):
        """get_status 从顶层包可用。"""
        assert _get_status is get_status

    def test_list_available_templates_from_top_level(self):
        """list_available_templates 从顶层包可用。"""
        assert _list_available_templates is list_available_templates

    def test_all_in_all_list(self):
        """所有便捷函数在 __all__ 中。"""
        import agent_whitepaper
        for name in [
            "quick_start", "train_agent", "run_demo",
            "create_system", "get_status", "list_available_templates",
        ]:
            assert name in agent_whitepaper.__all__


# ---------------------------------------------------------------------------
# 源码结构验证
# ---------------------------------------------------------------------------
class TestConvenienceSourceCode:
    """验证便捷API源码结构。"""

    def test_convenience_module_exists(self):
        """convenience.py 模块存在。"""
        from agent_whitepaper.stage_h import convenience
        assert convenience is not None

    def test_quick_start_has_docstring(self):
        """quick_start 有docstring。"""
        assert quick_start.__doc__ is not None
        assert "启动" in quick_start.__doc__ or "start" in quick_start.__doc__.lower()

    def test_train_agent_has_docstring(self):
        """train_agent 有docstring。"""
        assert train_agent.__doc__ is not None
        assert "训练" in train_agent.__doc__ or "train" in train_agent.__doc__.lower()

    def test_run_demo_has_docstring(self):
        """run_demo 有docstring。"""
        assert run_demo.__doc__ is not None
        assert "演示" in run_demo.__doc__ or "demo" in run_demo.__doc__.lower()

    def test_create_system_has_docstring(self):
        """create_system 有docstring。"""
        assert create_system.__doc__ is not None
        assert "创建" in create_system.__doc__ or "create" in create_system.__doc__.lower()

    def test_get_status_has_docstring(self):
        """get_status 有docstring。"""
        assert get_status.__doc__ is not None
        assert "状态" in get_status.__doc__ or "status" in get_status.__doc__.lower()

    def test_module_has_first_principle_docstring(self):
        """模块docstring包含第一性原理。"""
        source = Path(
            inspect.getfile(quick_start)
        ).read_text(encoding="utf-8")
        assert "第一性原理" in source
        assert "Make the right thing easy" in source

    def test_quick_start_has_example(self):
        """quick_start docstring包含示例。"""
        assert "Example" in quick_start.__doc__ or ">>>" in quick_start.__doc__

    def test_train_agent_has_example(self):
        """train_agent docstring包含示例。"""
        assert "Example" in train_agent.__doc__ or ">>>" in train_agent.__doc__

    def test_all_functions_have_type_hints(self):
        """所有便捷函数有返回类型注解。"""
        sig = inspect.signature(quick_start)
        assert sig.return_annotation is not inspect.Signature.empty

        sig = inspect.signature(train_agent)
        assert sig.return_annotation is not inspect.Signature.empty

        sig = inspect.signature(run_demo)
        assert sig.return_annotation is not inspect.Signature.empty

        sig = inspect.signature(create_system)
        assert sig.return_annotation is not inspect.Signature.empty

        sig = inspect.signature(get_status)
        assert sig.return_annotation is not inspect.Signature.empty
