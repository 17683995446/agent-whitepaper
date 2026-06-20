"""阶段AH测试：验证可视化报告系统。

验证内容：
1. Visualizer 可视化报告生成器类
2. generate_html 生成HTML报告
3. save 保存HTML到文件
4. generate_and_save 生成并保存
5. _render_summary_cards 渲染摘要卡片
6. _render_trend_bars 渲染趋势柱状图
7. _render_agent_table 渲染Agent表格
8. _render_cumulative_stats 渲染累计统计
9. generate_report 便捷函数
10. generate_report_from_orchestrator 便捷函数
11. CLI集成（report子命令、cmd_report）
12. 源码结构符合规范
"""

import argparse
import inspect
import os
from pathlib import Path
from unittest.mock import MagicMock

from agent_whitepaper.stage_h.visualizer import (
    Visualizer,
    generate_report,
    generate_report_from_orchestrator,
)


# ---------------------------------------------------------------------------
# 测试夹具
# ---------------------------------------------------------------------------
def _make_report(
    iteration: int = 5,
    agent_count: int = 3,
    skill_count: int = 2,
    with_snapshot: bool = True,
    with_cumulative: bool = True,
) -> dict:
    """构造一个测试用指标报告。"""
    report = {
        "iteration": iteration,
        "status": "running",
        "uptime": 12.34,
        "agent_count": agent_count,
        "skill_count": skill_count,
        "tasks_completed": 10,
        "tasks_failed": 1,
        "latest_snapshot": None,
        "success_rate_summary": {"count": 3, "first": 0.1, "last": 0.5, "mean": 0.3},
        "skill_count_summary": {"count": 3, "first": 0, "last": 2, "mean": 1},
        "cumulative": {},
    }
    if with_snapshot:
        report["latest_snapshot"] = {
            "avg_success_rate": 0.5,
            "avg_reward": 0.25,
            "total_q_entries": 100,
            "total_episodes": 50,
            "health_score": 0.8,
            "distillation_count": 2,
            "recovery_count": 0,
        }
    if with_cumulative:
        report["cumulative"] = {
            "distillation_count": 2,
            "migration_count": 1,
            "injected_states": 5,
            "recovery_count": 0,
        }
    return report


# ---------------------------------------------------------------------------
# Visualizer 类验证
# ---------------------------------------------------------------------------
class TestVisualizerClass:
    """验证 Visualizer 类。"""

    def test_visualizer_class_exists(self):
        """Visualizer 类存在且可调用。"""
        assert callable(Visualizer)

    def test_visualizer_has_docstring(self):
        """Visualizer 有docstring。"""
        assert Visualizer.__doc__ is not None
        assert "可视化" in Visualizer.__doc__ or "Visualizer" in Visualizer.__doc__

    def test_visualizer_instantiation(self):
        """可以实例化 Visualizer。"""
        viz = Visualizer()
        assert viz is not None

    def test_generate_html_method_exists(self):
        """generate_html 方法存在。"""
        assert hasattr(Visualizer, "generate_html")
        assert callable(Visualizer.generate_html)

    def test_save_method_exists(self):
        """save 方法存在。"""
        assert hasattr(Visualizer, "save")
        assert callable(Visualizer.save)

    def test_generate_and_save_method_exists(self):
        """generate_and_save 方法存在。"""
        assert hasattr(Visualizer, "generate_and_save")
        assert callable(Visualizer.generate_and_save)


# ---------------------------------------------------------------------------
# generate_html 验证
# ---------------------------------------------------------------------------
class TestGenerateHtml:
    """验证 generate_html 方法。"""

    def test_generate_html_returns_string(self):
        """generate_html 返回字符串。"""
        viz = Visualizer()
        report = _make_report()
        result = viz.generate_html(report)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_html_is_valid_html(self):
        """生成的HTML包含基本结构。"""
        viz = Visualizer()
        report = _make_report()
        html_content = viz.generate_html(report)
        assert "<!DOCTYPE html>" in html_content
        assert "<html" in html_content
        assert "</html>" in html_content
        assert "<head>" in html_content
        assert "<body>" in html_content

    def test_generate_html_contains_title(self):
        """HTML包含标题。"""
        viz = Visualizer()
        report = _make_report()
        html_content = viz.generate_html(report)
        assert "Agent Whitepaper" in html_content or "运行报告" in html_content

    def test_generate_html_contains_iteration(self):
        """HTML包含迭代次数。"""
        viz = Visualizer()
        report = _make_report(iteration=42)
        html_content = viz.generate_html(report)
        assert "42" in html_content

    def test_generate_html_contains_agent_count(self):
        """HTML包含Agent数。"""
        viz = Visualizer()
        report = _make_report(agent_count=7)
        html_content = viz.generate_html(report)
        assert "7" in html_content

    def test_generate_html_contains_status(self):
        """HTML包含系统状态。"""
        viz = Visualizer()
        report = _make_report()
        report["status"] = "running"
        html_content = viz.generate_html(report)
        assert "running" in html_content

    def test_generate_html_with_empty_report(self):
        """空报告也能生成HTML（不崩溃）。"""
        viz = Visualizer()
        report = {}
        html_content = viz.generate_html(report)
        assert isinstance(html_content, str)
        assert "<!DOCTYPE html>" in html_content

    def test_generate_html_with_no_snapshot(self):
        """无快照的报告也能生成HTML。"""
        viz = Visualizer()
        report = _make_report(with_snapshot=False)
        html_content = viz.generate_html(report)
        assert "<!DOCTYPE html>" in html_content

    def test_generate_html_with_no_cumulative(self):
        """无累计统计的报告也能生成HTML。"""
        viz = Visualizer()
        report = _make_report(with_cumulative=False)
        html_content = viz.generate_html(report)
        assert "<!DOCTYPE html>" in html_content

    def test_generate_html_contains_css(self):
        """HTML包含CSS样式。"""
        viz = Visualizer()
        report = _make_report()
        html_content = viz.generate_html(report)
        assert "<style>" in html_content
        assert "</style>" in html_content

    def test_generate_html_escapes_content(self):
        """HTML对内容进行转义。"""
        viz = Visualizer()
        report = _make_report()
        report["status"] = "<script>alert('xss')</script>"
        html_content = viz.generate_html(report)
        # 危险内容应被转义
        assert "<script>alert" not in html_content or "&lt;script&gt;" in html_content


# ---------------------------------------------------------------------------
# save 方法验证
# ---------------------------------------------------------------------------
class TestSaveMethod:
    """验证 save 方法。"""

    def test_save_creates_file(self, tmp_path):
        """save 创建文件。"""
        viz = Visualizer()
        path = str(tmp_path / "report.html")
        viz.save("<html>test</html>", path)
        assert os.path.exists(path)

    def test_save_writes_content(self, tmp_path):
        """save 写入正确内容。"""
        viz = Visualizer()
        path = str(tmp_path / "report.html")
        content = "<html><body>hello</body></html>"
        viz.save(content, path)
        with open(path, encoding="utf-8") as f:
            assert f.read() == content

    def test_save_utf8_encoding(self, tmp_path):
        """save 使用UTF-8编码（支持中文）。"""
        viz = Visualizer()
        path = str(tmp_path / "report.html")
        content = "<html>中文测试</html>"
        viz.save(content, path)
        with open(path, encoding="utf-8") as f:
            assert "中文测试" in f.read()


# ---------------------------------------------------------------------------
# generate_and_save 方法验证
# ---------------------------------------------------------------------------
class TestGenerateAndSave:
    """验证 generate_and_save 方法。"""

    def test_generate_and_save_returns_html(self, tmp_path):
        """generate_and_save 返回HTML内容。"""
        viz = Visualizer()
        report = _make_report()
        path = str(tmp_path / "report.html")
        result = viz.generate_and_save(report, path)
        assert isinstance(result, str)
        assert "<!DOCTYPE html>" in result

    def test_generate_and_save_creates_file(self, tmp_path):
        """generate_and_save 创建文件。"""
        viz = Visualizer()
        report = _make_report()
        path = str(tmp_path / "report.html")
        viz.generate_and_save(report, path)
        assert os.path.exists(path)

    def test_generate_and_save_file_matches_return(self, tmp_path):
        """文件内容与返回值一致。"""
        viz = Visualizer()
        report = _make_report()
        path = str(tmp_path / "report.html")
        returned = viz.generate_and_save(report, path)
        with open(path, encoding="utf-8") as f:
            file_content = f.read()
        assert returned == file_content


# ---------------------------------------------------------------------------
# 渲染组件验证
# ---------------------------------------------------------------------------
class TestRenderComponents:
    """验证内部渲染组件。"""

    def test_render_summary_cards_returns_string(self):
        """_render_summary_cards 返回字符串。"""
        viz = Visualizer()
        report = _make_report()
        result = viz._render_summary_cards(report)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_render_summary_cards_contains_iteration(self):
        """摘要卡片包含迭代次数。"""
        viz = Visualizer()
        report = _make_report(iteration=99)
        result = viz._render_summary_cards(report)
        assert "99" in result

    def test_render_summary_cards_contains_card_class(self):
        """摘要卡片包含 card CSS类。"""
        viz = Visualizer()
        report = _make_report()
        result = viz._render_summary_cards(report)
        assert "card" in result

    def test_render_trend_bars_returns_string(self):
        """_render_trend_bars 返回字符串。"""
        viz = Visualizer()
        report = _make_report()
        result = viz._render_trend_bars(report, "success_rate_summary", "avg_success_rate")
        assert isinstance(result, str)

    def test_render_trend_bars_with_data(self):
        """有数据时渲染柱状图。"""
        viz = Visualizer()
        report = _make_report()
        result = viz._render_trend_bars(report, "success_rate_summary", "avg_success_rate")
        # 应包含 bar 类
        assert "bar" in result

    def test_render_trend_bars_empty_data(self):
        """无数据时显示提示。"""
        viz = Visualizer()
        report = {"success_rate_summary": {}}
        result = viz._render_trend_bars(report, "success_rate_summary", "avg_success_rate")
        assert "暂无数据" in result

    def test_render_trend_bars_missing_key(self):
        """缺少summary键时显示提示。"""
        viz = Visualizer()
        report = {}
        result = viz._render_trend_bars(report, "nonexistent", "value")
        assert "暂无数据" in result

    def test_render_trend_bars_with_values_list(self):
        """summary含values列表时正确渲染。"""
        viz = Visualizer()
        report = {"test_summary": {"values": [0.1, 0.5, 0.9]}}
        result = viz._render_trend_bars(report, "test_summary", "value")
        assert "bar" in result

    def test_render_trend_bars_with_history_list(self):
        """summary含history列表时正确渲染。"""
        viz = Visualizer()
        report = {"test_summary": {"history": [1, 2, 3]}}
        result = viz._render_trend_bars(report, "test_summary", "value")
        assert "bar" in result

    def test_render_agent_table_returns_string(self):
        """_render_agent_table 返回字符串。"""
        viz = Visualizer()
        report = _make_report(agent_count=3)
        result = viz._render_agent_table(report)
        assert isinstance(result, str)

    def test_render_agent_table_with_agents(self):
        """有Agent时渲染表格。"""
        viz = Visualizer()
        report = _make_report(agent_count=3)
        result = viz._render_agent_table(report)
        assert "<table" in result
        assert "agent_0" in result

    def test_render_agent_table_no_agents(self):
        """无Agent时显示提示。"""
        viz = Visualizer()
        report = _make_report(agent_count=0)
        result = viz._render_agent_table(report)
        assert "无Agent" in result

    def test_render_cumulative_stats_returns_string(self):
        """_render_cumulative_stats 返回字符串。"""
        viz = Visualizer()
        report = _make_report()
        result = viz._render_cumulative_stats(report)
        assert isinstance(result, str)

    def test_render_cumulative_stats_with_data(self):
        """有累计统计时渲染表格。"""
        viz = Visualizer()
        report = _make_report()
        result = viz._render_cumulative_stats(report)
        assert "<table" in result
        # 应包含中文翻译
        assert "蒸馏次数" in result or "distillation_count" in result

    def test_render_cumulative_stats_empty(self):
        """无累计统计时显示提示。"""
        viz = Visualizer()
        report = {"cumulative": {}}
        result = viz._render_cumulative_stats(report)
        assert "无累计统计" in result

    def test_translate_cumulative_key_known(self):
        """已知键翻译为中文。"""
        viz = Visualizer()
        assert viz._translate_cumulative_key("distillation_count") == "蒸馏次数"
        assert viz._translate_cumulative_key("migration_count") == "迁移次数"
        assert viz._translate_cumulative_key("injected_states") == "注入状态数"
        assert viz._translate_cumulative_key("recovery_count") == "恢复次数"

    def test_translate_cumulative_key_unknown(self):
        """未知键保持原样。"""
        viz = Visualizer()
        assert viz._translate_cumulative_key("unknown_key") == "unknown_key"


# ---------------------------------------------------------------------------
# 便捷函数验证
# ---------------------------------------------------------------------------
class TestConvenienceFunctions:
    """验证便捷函数。"""

    def test_generate_report_function_exists(self):
        """generate_report 函数存在。"""
        assert callable(generate_report)

    def test_generate_report_from_orchestrator_exists(self):
        """generate_report_from_orchestrator 函数存在。"""
        assert callable(generate_report_from_orchestrator)

    def test_generate_report_without_path(self):
        """不带path参数返回HTML但不保存。"""
        report = _make_report()
        result = generate_report(report)
        assert isinstance(result, str)
        assert "<!DOCTYPE html>" in result

    def test_generate_report_with_path(self, tmp_path):
        """带path参数保存文件并返回HTML。"""
        report = _make_report()
        path = str(tmp_path / "report.html")
        result = generate_report(report, path)
        assert isinstance(result, str)
        assert os.path.exists(path)

    def test_generate_report_from_orchestrator_with_mock(self, tmp_path):
        """用mock编排器生成报告。"""
        orch = MagicMock()
        orch.get_metrics.return_value = _make_report()
        orch.get_agent_summaries.return_value = [
            {"agent_id": "agent_0", "success_rate": 0.5, "total_episodes": 10, "q_entries": 5}
        ]
        path = str(tmp_path / "report.html")
        result = generate_report_from_orchestrator(orch, path)
        assert isinstance(result, str)
        assert os.path.exists(path)
        orch.get_metrics.assert_called_once()

    def test_generate_report_from_orchestrator_no_path(self):
        """不带path的编排器报告生成。"""
        orch = MagicMock()
        orch.get_metrics.return_value = _make_report()
        orch.get_agent_summaries.return_value = []
        result = generate_report_from_orchestrator(orch)
        assert isinstance(result, str)
        assert "<!DOCTYPE html>" in result

    def test_generate_report_from_orchestrator_handles_summary_failure(self, tmp_path):
        """get_agent_summaries失败时不影响报告生成。"""
        orch = MagicMock()
        orch.get_metrics.return_value = _make_report()
        orch.get_agent_summaries.side_effect = RuntimeError("boom")
        path = str(tmp_path / "report.html")
        # 不应抛出异常
        result = generate_report_from_orchestrator(orch, path)
        assert isinstance(result, str)
        assert "<!DOCTYPE html>" in result


# ---------------------------------------------------------------------------
# 端到端集成验证
# ---------------------------------------------------------------------------
class TestEndToEndIntegration:
    """端到端集成验证（使用真实编排器）。"""

    def test_generate_report_from_real_orchestrator(self, tmp_path):
        """从真实编排器生成报告。"""
        from agent_whitepaper.stage_g.orchestrator import SystemOrchestrator
        from agent_whitepaper.stage_h.config import load_config

        config = load_config(
            n_agents=2,
            grid_size=3,
            db_path=":memory:",
            max_iterations=3,
            seed=42,
        )
        orch = SystemOrchestrator(config)
        orch.initialize()
        for _ in range(3):
            orch.run_iteration()

        path = str(tmp_path / "e2e_report.html")
        result = generate_report_from_orchestrator(orch, path)
        orch.stop("test")

        assert isinstance(result, str)
        assert "<!DOCTYPE html>" in result
        assert os.path.exists(path)
        # 文件应有内容
        assert os.path.getsize(path) > 100


# ---------------------------------------------------------------------------
# CLI 集成验证
# ---------------------------------------------------------------------------
class TestCliIntegration:
    """验证CLI集成。"""

    def test_report_subcommand_registered(self):
        """report子命令已注册。"""
        from agent_whitepaper.stage_h.cli import build_parser

        parser = build_parser()
        # 解析 report --help 应不报错
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                parser.parse_args(["report", "--help"])
            except SystemExit:
                pass
        output = buf.getvalue()
        assert "report" in output

    def test_cmd_report_function_exists(self):
        """cmd_report 函数存在。"""
        from agent_whitepaper.stage_h.cli import cmd_report
        assert callable(cmd_report)

    def test_cmd_report_has_docstring(self):
        """cmd_report 有docstring。"""
        from agent_whitepaper.stage_h.cli import cmd_report
        assert cmd_report.__doc__ is not None

    def test_cmd_report_returns_int(self, tmp_path):
        """cmd_report 返回整数退出码。"""
        from agent_whitepaper.stage_h.cli import cmd_report

        args = argparse.Namespace(
            output=str(tmp_path / "cli_report.html"),
            iterations=2,
            agents=2,
            grid=3,
        )
        result = cmd_report(args)
        assert isinstance(result, int)
        assert result == 0
        assert os.path.exists(str(tmp_path / "cli_report.html"))

    def test_report_command_in_doctor_subcommands(self):
        """doctor命令检查的子命令列表包含report。"""
        import contextlib
        import io

        buf = io.StringIO()
        args = argparse.Namespace()
        from agent_whitepaper.stage_h.cli import cmd_doctor
        with contextlib.redirect_stdout(buf):
            cmd_doctor(args)
        output = buf.getvalue()
        assert "agentwp report" in output

    def test_report_subcommand_has_output_arg(self):
        """report子命令有 -o/--output 参数。"""
        from agent_whitepaper.stage_h.cli import build_parser

        parser = build_parser()
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                parser.parse_args(["report", "--help"])
            except SystemExit:
                pass
        output = buf.getvalue()
        assert "--output" in output or "-o" in output

    def test_report_subcommand_has_iterations_arg(self):
        """report子命令有 -n/--iterations 参数。"""
        from agent_whitepaper.stage_h.cli import build_parser

        parser = build_parser()
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                parser.parse_args(["report", "--help"])
            except SystemExit:
                pass
        output = buf.getvalue()
        assert "--iterations" in output or "-n" in output


# ---------------------------------------------------------------------------
# 源码结构验证
# ---------------------------------------------------------------------------
class TestVisualizerSourceCode:
    """验证可视化报告源码结构。"""

    def test_visualizer_module_exists(self):
        """visualizer.py 模块存在。"""
        from agent_whitepaper.stage_h import visualizer
        assert visualizer is not None

    def test_visualizer_class_exported_from_stage_h(self):
        """Visualizer 从 stage_h 包导出。"""
        import agent_whitepaper.stage_h as stage_h
        assert stage_h.Visualizer is Visualizer

    def test_generate_report_exported_from_stage_h(self):
        """generate_report 从 stage_h 包导出。"""
        import agent_whitepaper.stage_h as stage_h
        assert stage_h.generate_report is generate_report

    def test_generate_report_from_orchestrator_exported(self):
        """generate_report_from_orchestrator 从 stage_h 包导出。"""
        import agent_whitepaper.stage_h as stage_h
        assert stage_h.generate_report_from_orchestrator is generate_report_from_orchestrator

    def test_module_has_first_principle_docstring(self):
        """模块docstring包含第一性原理。"""
        source = Path(
            inspect.getfile(Visualizer)
        ).read_text(encoding="utf-8")
        assert "第一性原理" in source
        assert "picture" in source or "图表" in source or "可视化" in source

    def test_visualizer_class_has_first_principle(self):
        """Visualizer类docstring包含第一性原理。"""
        source = Path(
            inspect.getfile(Visualizer)
        ).read_text(encoding="utf-8")
        # 类定义附近应有第一性原理说明
        assert "第一性原理" in source

    def test_all_public_methods_have_type_hints(self):
        """所有公共方法有返回类型注解。"""
        sig = inspect.signature(Visualizer.generate_html)
        assert sig.return_annotation is not inspect.Signature.empty

        sig = inspect.signature(Visualizer.save)
        assert sig.return_annotation is not inspect.Signature.empty

        sig = inspect.signature(Visualizer.generate_and_save)
        assert sig.return_annotation is not inspect.Signature.empty

    def test_convenience_functions_have_type_hints(self):
        """便捷函数有返回类型注解。"""
        sig = inspect.signature(generate_report)
        assert sig.return_annotation is not inspect.Signature.empty

        sig = inspect.signature(generate_report_from_orchestrator)
        assert sig.return_annotation is not inspect.Signature.empty

    def test_no_silent_except_exception_pass(self):
        """无静默 except Exception: pass。"""
        import ast
        source = Path(
            inspect.getfile(Visualizer)
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

    def test_html_template_is_self_contained(self):
        """HTML模板是自包含的（无外部依赖）。"""
        source = Path(
            inspect.getfile(Visualizer)
        ).read_text(encoding="utf-8")
        # 不应引用外部CSS/JS资源
        assert "https://cdn" not in source
        assert "http://" not in source
        # 应内联样式
        assert "<style>" in source

    def test_init_docstring_mentions_visualizer(self):
        """stage_h __init__.py docstring提及Visualizer。"""
        import agent_whitepaper.stage_h as stage_h
        init_source = Path(
            inspect.getfile(stage_h)
        ).read_text(encoding="utf-8")
        assert "Visualizer" in init_source
        assert "generate_report" in init_source
