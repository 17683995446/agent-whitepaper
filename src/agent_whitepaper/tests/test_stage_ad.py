"""阶段AD测试：验证进度反馈系统。

验证内容：
1. format_duration 正确格式化时长
2. render_progress_bar 正确渲染进度条
3. predict_phases 正确预测迭代阶段
4. ProgressReporter 有限模式（进度条、ETA、速率）
5. ProgressReporter 无限模式（迭代数、时长、速率）
6. ProgressReporter 指标快照显示
7. SilentReporter 不产生输出
8. cmd_run 集成进度报告器
9. 源码结构符合规范
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

from agent_whitepaper.stage_h.cli import build_parser, cmd_run, main
from agent_whitepaper.stage_h.progress import (
    ProgressReporter,
    SilentReporter,
    format_duration,
    predict_phases,
    render_progress_bar,
)


# ---------------------------------------------------------------------------
# format_duration 验证
# ---------------------------------------------------------------------------
class TestFormatDuration:
    """验证 format_duration 函数。"""

    def test_format_seconds(self):
        """秒级时长。"""
        assert format_duration(5.0) == "5.0s"

    def test_format_minutes(self):
        """分钟级时长。"""
        assert format_duration(65.0) == "1.1m"

    def test_format_hours(self):
        """小时级时长。"""
        assert format_duration(3700.0) == "1.0h"

    def test_format_zero(self):
        """零时长。"""
        assert format_duration(0.0) == "0.0s"

    def test_format_negative(self):
        """负值返回'未知'。"""
        assert format_duration(-1.0) == "未知"

    def test_format_just_under_minute(self):
        """刚好低于60秒。"""
        assert format_duration(59.9) == "59.9s"

    def test_format_exactly_one_minute(self):
        """刚好60秒。"""
        assert format_duration(60.0) == "1.0m"


# ---------------------------------------------------------------------------
# render_progress_bar 验证
# ---------------------------------------------------------------------------
class TestRenderProgressBar:
    """验证 render_progress_bar 函数。"""

    def test_zero_progress(self):
        """零进度。"""
        bar = render_progress_bar(0, 100, width=10)
        assert "0/100" in bar
        assert "(0%)" in bar
        assert "░" * 10 in bar

    def test_half_progress(self):
        """半进度。"""
        bar = render_progress_bar(50, 100, width=10)
        assert "50/100" in bar
        assert "(50%)" in bar
        assert "█" * 5 in bar
        assert "░" * 5 in bar

    def test_full_progress(self):
        """满进度。"""
        bar = render_progress_bar(100, 100, width=10)
        assert "100/100" in bar
        assert "(100%)" in bar
        assert "█" * 10 in bar

    def test_zero_total_returns_empty(self):
        """total为0返回空字符串。"""
        assert render_progress_bar(5, 0) == ""

    def test_negative_total_returns_empty(self):
        """total为负返回空字符串。"""
        assert render_progress_bar(5, -1) == ""

    def test_current_exceeds_total_clamped(self):
        """current超过total被钳制。"""
        bar = render_progress_bar(150, 100, width=10)
        assert "100/100" in bar
        assert "(100%)" in bar

    def test_bar_has_brackets(self):
        """进度条有方括号。"""
        bar = render_progress_bar(50, 100, width=10)
        assert bar.startswith("[")
        assert "]" in bar


# ---------------------------------------------------------------------------
# predict_phases 验证
# ---------------------------------------------------------------------------
class TestPredictPhases:
    """验证 predict_phases 函数。"""

    def test_always_phases_present(self):
        """每轮必做的阶段始终存在。"""
        phases = predict_phases(iteration=1)
        assert "健康检查" in phases
        assert "任务执行" in phases
        assert "指标采集" in phases

    def test_skill_abstract_at_interval(self):
        """技能抽象在间隔触发。"""
        phases = predict_phases(iteration=5, skill_abstract_interval=5)
        assert "技能抽象" in phases

    def test_skill_abstract_not_at_interval(self):
        """技能抽象不在间隔时不触发。"""
        phases = predict_phases(iteration=4, skill_abstract_interval=5)
        assert "技能抽象" not in phases

    def test_meta_learn_at_interval(self):
        """元学习在间隔触发。"""
        phases = predict_phases(iteration=10, meta_learn_interval=10)
        assert "元学习" in phases

    def test_coalition_at_interval_multi_agent(self):
        """联盟协作在间隔且多Agent时触发。"""
        phases = predict_phases(
            iteration=15, coalition_interval=15, n_agents=3
        )
        assert "联盟协作" in phases

    def test_coalition_not_triggered_single_agent(self):
        """单Agent时不触发联盟协作。"""
        phases = predict_phases(
            iteration=15, coalition_interval=15, n_agents=1
        )
        assert "联盟协作" not in phases

    def test_distill_at_interval_multi_agent(self):
        """知识蒸馏在间隔且多Agent时触发。"""
        phases = predict_phases(
            iteration=30, distill_interval=30, n_agents=3
        )
        assert "知识蒸馏" in phases

    def test_distill_not_triggered_single_agent(self):
        """单Agent时不触发知识蒸馏。"""
        phases = predict_phases(
            iteration=30, distill_interval=30, n_agents=1
        )
        assert "知识蒸馏" not in phases

    def test_evolve_at_interval(self):
        """自主进化在间隔触发。"""
        phases = predict_phases(iteration=20, evolve_interval=20)
        assert "自主进化" in phases

    def test_checkpoint_at_interval(self):
        """检查点在间隔触发。"""
        phases = predict_phases(iteration=50, checkpoint_interval=50)
        assert "检查点" in phases

    def test_all_phases_at_common_multiple(self):
        """在公倍数处所有阶段都触发。"""
        phases = predict_phases(
            iteration=300,
            skill_abstract_interval=5,
            meta_learn_interval=10,
            coalition_interval=15,
            distill_interval=30,
            evolve_interval=20,
            checkpoint_interval=50,
            n_agents=3,
        )
        for phase in [
            "健康检查", "任务执行", "技能抽象", "元学习",
            "联盟协作", "知识蒸馏", "自主进化", "检查点", "指标采集",
        ]:
            assert phase in phases

    def test_phases_returns_list(self):
        """返回列表。"""
        phases = predict_phases(iteration=1)
        assert isinstance(phases, list)


# ---------------------------------------------------------------------------
# ProgressReporter 有限模式验证
# ---------------------------------------------------------------------------
class TestProgressReporterFinite:
    """验证 ProgressReporter 有限模式。"""

    def test_reporter_finite_mode(self):
        """有限模式创建。"""
        reporter = ProgressReporter(total=100, report_interval=10)
        assert reporter.total == 100
        assert reporter.show_bar is True

    def test_should_report_at_interval(self):
        """在间隔处应该报告。"""
        reporter = ProgressReporter(total=100, report_interval=10)
        assert reporter.should_report(10) is True
        assert reporter.should_report(20) is True

    def test_should_not_report_off_interval(self):
        """不在间隔处不应报告。"""
        reporter = ProgressReporter(total=100, report_interval=10)
        assert reporter.should_report(5) is False
        assert reporter.should_report(15) is False

    def test_should_report_at_total(self):
        """在total处应该报告。"""
        reporter = ProgressReporter(total=100, report_interval=10)
        assert reporter.should_report(100) is True

    def test_report_outputs_progress_bar(self):
        """报告输出包含进度条。"""
        buf = StringIO()
        reporter = ProgressReporter(
            total=100, report_interval=10, output=buf.write
        )
        reporter.report(50)
        output = buf.getvalue()
        assert "50/100" in output
        assert "(50%)" in output

    def test_report_outputs_rate(self):
        """报告输出包含速率。"""
        buf = StringIO()
        reporter = ProgressReporter(
            total=100, report_interval=1, output=buf.write
        )
        reporter.report(10)
        output = buf.getvalue()
        assert "it/s" in output

    def test_report_outputs_eta(self):
        """报告输出包含ETA。"""
        buf = StringIO()
        reporter = ProgressReporter(
            total=100, report_interval=1, output=buf.write
        )
        reporter.report(10)
        output = buf.getvalue()
        assert "ETA" in output

    def test_report_outputs_metrics_from_object(self):
        """报告从对象快照提取指标。"""

        class FakeSnapshot:
            avg_success_rate = 0.85
            skill_count = 12
            health_score = 0.92

        buf = StringIO()
        reporter = ProgressReporter(
            total=100, report_interval=1, output=buf.write
        )
        reporter.report(10, FakeSnapshot())
        output = buf.getvalue()
        assert "成功率=0.850" in output
        assert "技能=12" in output
        assert "健康=0.920" in output

    def test_report_outputs_metrics_from_dict(self):
        """报告从字典快照提取指标。"""
        snap = {
            "avg_success_rate": 0.85,
            "skill_count": 12,
            "health_score": 0.92,
        }
        buf = StringIO()
        reporter = ProgressReporter(
            total=100, report_interval=1, output=buf.write
        )
        reporter.report(10, snap)
        output = buf.getvalue()
        assert "成功率=0.850" in output

    def test_report_outputs_phases(self):
        """报告输出包含阶段。"""
        buf = StringIO()
        reporter = ProgressReporter(
            total=100, report_interval=1, output=buf.write
        )
        phases = ["健康检查", "任务执行", "技能抽象"]
        reporter.report(5, phases=phases)
        output = buf.getvalue()
        assert "技能抽象" in output

    def test_finish_outputs_completion(self):
        """finish输出完成信息。"""
        buf = StringIO()
        reporter = ProgressReporter(
            total=10, report_interval=5, output=buf.write
        )
        reporter.report(10)
        buf.truncate(0)
        buf.seek(0)
        reporter.finish()
        output = buf.getvalue()
        assert "完成" in output

    def test_no_output_when_off_interval(self):
        """不在间隔处不输出。"""
        buf = StringIO()
        reporter = ProgressReporter(
            total=100, report_interval=10, output=buf.write
        )
        reporter.report(5)
        assert buf.getvalue() == ""


# ---------------------------------------------------------------------------
# ProgressReporter 无限模式验证
# ---------------------------------------------------------------------------
class TestProgressReporterInfinite:
    """验证 ProgressReporter 无限模式。"""

    def test_reporter_infinite_mode(self):
        """无限模式创建。"""
        reporter = ProgressReporter(total=None, report_interval=10)
        assert reporter.total is None
        assert reporter.show_bar is False

    def test_infinite_no_progress_bar(self):
        """无限模式不显示进度条。"""
        buf = StringIO()
        reporter = ProgressReporter(
            total=None, report_interval=1, output=buf.write
        )
        reporter.report(10)
        output = buf.getvalue()
        assert "█" not in output
        assert "[10]" in output

    def test_infinite_no_eta(self):
        """无限模式不显示ETA。"""
        buf = StringIO()
        reporter = ProgressReporter(
            total=None, report_interval=1, output=buf.write
        )
        reporter.report(10)
        output = buf.getvalue()
        assert "ETA" not in output

    def test_infinite_outputs_iteration_count(self):
        """无限模式显示迭代数。"""
        buf = StringIO()
        reporter = ProgressReporter(
            total=None, report_interval=1, output=buf.write
        )
        reporter.report(150)
        output = buf.getvalue()
        assert "150" in output

    def test_infinite_outputs_rate(self):
        """无限模式显示速率。"""
        buf = StringIO()
        reporter = ProgressReporter(
            total=None, report_interval=1, output=buf.write
        )
        reporter.report(10)
        output = buf.getvalue()
        assert "it/s" in output


# ---------------------------------------------------------------------------
# SilentReporter 验证
# ---------------------------------------------------------------------------
class TestSilentReporter:
    """验证 SilentReporter。"""

    def test_silent_produces_no_output(self):
        """静默报告器不产生输出。"""
        reporter = SilentReporter(total=100, report_interval=1)
        buf = StringIO()
        with redirect_stdout(buf):
            reporter.report(50)
        assert buf.getvalue() == ""

    def test_silent_finish_no_output(self):
        """静默报告器finish不产生输出。"""
        reporter = SilentReporter(total=100, report_interval=1)
        buf = StringIO()
        with redirect_stdout(buf):
            reporter.finish()
        assert buf.getvalue() == ""

    def test_silent_default_output_is_noop(self):
        """SilentReporter 默认 output 是空操作。"""
        reporter = SilentReporter(total=100, report_interval=1)
        # 调用 output 不应抛异常且返回 None
        result = reporter.output("anything")
        assert result is None

    def test_silent_should_report_still_works(self):
        """静默报告器should_report仍正常工作。"""
        reporter = SilentReporter(total=100, report_interval=10)
        assert reporter.should_report(10) is True
        assert reporter.should_report(5) is False

    def test_silent_inherits_progress_reporter(self):
        """SilentReporter继承ProgressReporter。"""
        assert issubclass(SilentReporter, ProgressReporter)


# ---------------------------------------------------------------------------
# cmd_run 集成验证
# ---------------------------------------------------------------------------
class TestCmdRunIntegration:
    """验证 cmd_run 集成进度报告器。"""

    def test_run_with_progress_reporter(self):
        """cmd_run使用ProgressReporter输出进度。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            from agent_whitepaper.stage_h.config import ConfigLoader
            loader = ConfigLoader()
            loader.generate_default_yaml(config_path, template="quick-test")
            # 覆盖为更小的配置
            import yaml
            with open(config_path, "w") as f:
                yaml.dump({
                    "grid_size": 3,
                    "n_agents": 2,
                    "db_path": ":memory:",
                    "checkpoint_interval": 10,
                    "skill_abstract_interval": 3,
                    "meta_learn_interval": 5,
                    "distill_interval": 10,
                    "evolve_interval": 7,
                    "coalition_interval": 5,
                    "max_iterations": 5,
                    "seed": 42,
                }, f)

            args = argparse.Namespace(
                config=config_path,
                iterations=None,
                agents=None,
                grid=None,
                memory=False,
                report_interval=2,
                quiet=False,
                verbose=False,
            )
            buf = StringIO()
            with redirect_stdout(buf):
                ret = cmd_run(args)
            assert ret == 0
            output = buf.getvalue()
            # 应该有进度相关输出
            assert "系统启动" in output
            assert "最终报告" in output

    def test_run_quiet_mode_minimal_output(self):
        """cmd_run --quiet 模式输出最小化。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            import yaml
            with open(config_path, "w") as f:
                yaml.dump({
                    "grid_size": 3,
                    "n_agents": 2,
                    "db_path": ":memory:",
                    "checkpoint_interval": 10,
                    "skill_abstract_interval": 3,
                    "meta_learn_interval": 5,
                    "distill_interval": 10,
                    "evolve_interval": 7,
                    "coalition_interval": 5,
                    "max_iterations": 3,
                    "seed": 42,
                }, f)

            args = argparse.Namespace(
                config=config_path,
                iterations=None,
                agents=None,
                grid=None,
                memory=False,
                report_interval=1,
                quiet=True,
                verbose=False,
            )
            buf = StringIO()
            with redirect_stdout(buf):
                ret = cmd_run(args)
            assert ret == 0
            output = buf.getvalue()
            # quiet模式下不应有进度条
            assert "█" not in output
            assert "it/s" not in output


# ---------------------------------------------------------------------------
# 通过 main() 调用验证
# ---------------------------------------------------------------------------
class TestRunViaMain:
    """验证通过 main() 调用 run。"""

    def test_main_run_finite_mode(self):
        """main(['run', '-n', '3', '--memory']) 正常工作。"""
        buf = StringIO()
        with redirect_stdout(buf):
            ret = main([
                "run", "-n", "3", "--memory",
                "-a", "2", "-g", "3",
                "--report-interval", "1",
            ])
        assert ret == 0
        output = buf.getvalue()
        assert "最终报告" in output

    def test_main_run_via_subprocess(self):
        """通过子进程验证 agentwp run。"""
        env = os.environ.copy()
        src_path = str(
            Path(__file__).resolve().parent.parent.parent.parent / "src"
        )
        env["PYTHONPATH"] = src_path
        result = subprocess.run(
            [
                sys.executable, "-m", "agent_whitepaper.stage_h.cli",
                "run", "-n", "3", "--memory",
                "-a", "2", "-g", "3",
                "--report-interval", "1",
            ],
            capture_output=True, text=True, timeout=60, env=env,
        )
        assert result.returncode == 0
        assert "最终报告" in result.stdout


# ---------------------------------------------------------------------------
# 源码结构验证
# ---------------------------------------------------------------------------
class TestProgressSourceCode:
    """验证进度反馈系统源码结构。"""

    def test_progress_module_exists(self):
        """progress.py 模块存在。"""
        from agent_whitepaper.stage_h import progress
        assert progress is not None

    def test_progress_reporter_class_exists(self):
        """ProgressReporter 类存在。"""
        assert callable(ProgressReporter)

    def test_silent_reporter_class_exists(self):
        """SilentReporter 类存在。"""
        assert callable(SilentReporter)

    def test_format_duration_function_exists(self):
        """format_duration 函数存在。"""
        assert callable(format_duration)

    def test_render_progress_bar_function_exists(self):
        """render_progress_bar 函数存在。"""
        assert callable(render_progress_bar)

    def test_predict_phases_function_exists(self):
        """predict_phases 函数存在。"""
        assert callable(predict_phases)

    def test_progress_reporter_has_docstring(self):
        """ProgressReporter 有docstring。"""
        assert ProgressReporter.__doc__ is not None
        assert "进度" in ProgressReporter.__doc__ or "Feedback" in ProgressReporter.__doc__

    def test_report_method_has_docstring(self):
        """report 方法有docstring。"""
        assert ProgressReporter.report.__doc__ is not None

    def test_finish_method_has_docstring(self):
        """finish 方法有docstring。"""
        assert ProgressReporter.finish.__doc__ is not None

    def test_progress_exported_from_stage_h(self):
        """进度组件从 stage_h 包导出。"""
        import agent_whitepaper.stage_h as stage_h
        assert stage_h.ProgressReporter is ProgressReporter
        assert stage_h.SilentReporter is SilentReporter
        assert stage_h.format_duration is format_duration
        assert stage_h.predict_phases is predict_phases
        assert stage_h.render_progress_bar is render_progress_bar

    def test_cli_imports_progress_components(self):
        """CLI 导入进度组件。"""
        source = Path(
            inspect.getfile(build_parser)
        ).read_text(encoding="utf-8")
        assert "ProgressReporter" in source
        assert "SilentReporter" in source
        assert "predict_phases" in source

    def test_progress_module_has_first_principle_docstring(self):
        """progress.py 模块docstring包含第一性原理。"""
        source = Path(
            inspect.getfile(ProgressReporter)
        ).read_text(encoding="utf-8")
        assert "第一性原理" in source
        assert "Feedback" in source
