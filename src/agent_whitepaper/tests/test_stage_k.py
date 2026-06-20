"""阶段K验证测试 + 长时运行实证与报告。

测试覆盖：
1. 长时运行演示器
2. 报告生成器（文本/JSON/摘要）
3. 趋势分析
4. 能力评估
5. 端到端：完整实证运行
"""

import json

from agent_whitepaper.stage_k.long_run_demo import (
    LongRunConfig,
    LongRunDemo,
    LongRunResult,
    full_demo,
    quick_demo,
)
from agent_whitepaper.stage_k.report_generator import (
    ReportGenerator,
)


# ============================================================
# 1. 长时运行演示器测试
# ============================================================
class TestLongRunDemo:
    def test_run_returns_result(self):
        config = LongRunConfig(n_iterations=20, n_agents=2, grid_size=4)
        demo = LongRunDemo(config)
        result = demo.run(verbose=False)
        assert isinstance(result, LongRunResult)
        assert result.total_iterations == 20

    def test_result_has_curves(self):
        config = LongRunConfig(n_iterations=15, n_agents=2, grid_size=4)
        demo = LongRunDemo(config)
        result = demo.run(verbose=False)
        assert len(result.success_rate_curve) > 0
        assert len(result.skill_count_curve) > 0
        assert len(result.q_entries_curve) > 0
        assert len(result.health_score_curve) > 0

    def test_result_metrics(self):
        config = LongRunConfig(n_iterations=10, n_agents=2, grid_size=4)
        demo = LongRunDemo(config)
        result = demo.run(verbose=False)
        assert result.total_time > 0
        assert result.iterations_per_second > 0
        assert result.tasks_completed + result.tasks_failed > 0

    def test_q_table_grows_during_run(self):
        config = LongRunConfig(n_iterations=30, n_agents=2, grid_size=4)
        demo = LongRunDemo(config)
        result = demo.run(verbose=False)
        # Q表应增长
        assert result.final_q_entries > 0
        if len(result.q_entries_curve) > 1:
            assert result.q_entries_curve[-1] >= result.q_entries_curve[0]

    def test_format_report(self):
        config = LongRunConfig(n_iterations=10, n_agents=2, grid_size=4)
        demo = LongRunDemo(config)
        result = demo.run(verbose=False)
        report = demo.format_report(result)
        assert "长时运行实证报告" in report
        assert "运行概况" in report
        assert "四大能力验证" in report

    def test_quick_demo(self):
        result = quick_demo(n_iterations=10, verbose=False)
        assert result.total_iterations == 10

    def test_full_demo(self):
        report = full_demo(n_iterations=10, verbose=False)
        assert "长时运行实证报告" in report
        assert "四大能力验证" in report


# ============================================================
# 2. 报告生成器测试
# ============================================================
class TestReportGenerator:
    def _make_result(self) -> LongRunResult:
        """构造测试用结果。"""
        return LongRunResult(
            total_iterations=100,
            total_time=10.0,
            tasks_completed=50,
            tasks_failed=30,
            final_success_rate=0.6,
            final_skill_count=15,
            final_q_entries=200,
            total_distillations=3,
            total_recoveries=1,
            success_rate_curve=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            skill_count_curve=[0, 3, 7, 10, 13, 15],
            q_entries_curve=[0, 40, 80, 120, 160, 200],
            health_score_curve=[1.0, 1.0, 0.9, 1.0, 1.0, 1.0],
            capability_summary={"agent_count": 3, "skill_count": 15},
        )

    def test_text_report(self):
        gen = ReportGenerator()
        result = self._make_result()
        report = gen.generate_text_report(result)
        assert "运行报告" in report
        assert "运行概况" in report
        assert "趋势分析" in report
        assert "四大能力评估" in report
        assert "结论" in report

    def test_json_report(self):
        gen = ReportGenerator()
        result = self._make_result()
        report_json = gen.generate_json_report(result)
        data = json.loads(report_json)
        assert "metadata" in data
        assert "run_summary" in data
        assert "learning" in data
        assert "evolution" in data
        assert "trends" in data
        assert "capabilities" in data
        assert "curves" in data
        assert data["run_summary"]["total_iterations"] == 100

    def test_summary(self):
        gen = ReportGenerator()
        result = self._make_result()
        summary = gen.generate_summary(result)
        assert "100轮" in summary
        assert "成功率" in summary

    def test_summary_is_concise(self):
        gen = ReportGenerator()
        result = self._make_result()
        summary = gen.generate_summary(result)
        # 摘要应是一行
        assert summary.count("\n") == 0


# ============================================================
# 3. 趋势分析测试
# ============================================================
class TestTrendAnalysis:
    def test_rising_trend(self):
        gen = ReportGenerator()
        result = LongRunResult(
            success_rate_curve=[0.1, 0.2, 0.3, 0.4, 0.5],
            skill_count_curve=[1, 2, 3, 4, 5],
            q_entries_curve=[10, 20, 30, 40, 50],
            health_score_curve=[0.8, 0.85, 0.9, 0.95, 1.0],
        )
        trend = gen.analyze_trends(result)
        assert trend.success_rate_trend == "rising"
        assert trend.skill_count_trend == "rising"
        assert trend.q_entries_trend == "rising"
        assert trend.health_score_trend == "rising"
        assert trend.success_rate_slope > 0

    def test_falling_trend(self):
        gen = ReportGenerator()
        result = LongRunResult(
            success_rate_curve=[0.9, 0.8, 0.7, 0.6, 0.5],
            skill_count_curve=[10, 8, 6, 4, 2],
            q_entries_curve=[],
            health_score_curve=[],
        )
        trend = gen.analyze_trends(result)
        assert trend.success_rate_trend == "falling"
        assert trend.skill_count_trend == "falling"

    def test_stable_trend(self):
        gen = ReportGenerator()
        result = LongRunResult(
            success_rate_curve=[0.5, 0.5, 0.5, 0.5, 0.5],
            skill_count_curve=[],
            q_entries_curve=[],
            health_score_curve=[],
        )
        trend = gen.analyze_trends(result)
        assert trend.success_rate_trend == "stable"

    def test_insufficient_data(self):
        gen = ReportGenerator()
        result = LongRunResult(
            success_rate_curve=[0.5],
        )
        trend = gen.analyze_trends(result)
        assert trend.success_rate_trend == "insufficient"

    def test_trend_to_dict(self):
        gen = ReportGenerator()
        result = LongRunResult(
            success_rate_curve=[0.1, 0.5],
            skill_count_curve=[1, 5],
        )
        trend = gen.analyze_trends(result)
        d = trend.to_dict()
        assert "success_rate_trend" in d
        assert "skill_count_trend" in d
        assert "success_rate_slope" in d


# ============================================================
# 4. 能力评估测试
# ============================================================
class TestCapabilityAssessment:
    def _make_result(self) -> LongRunResult:
        return LongRunResult(
            total_iterations=100,
            tasks_completed=50,
            tasks_failed=30,
            final_success_rate=0.6,
            final_skill_count=15,
            final_q_entries=200,
            total_distillations=3,
            total_recoveries=1,
            success_rate_curve=[0.1, 0.6],
            skill_count_curve=[0, 15],
            q_entries_curve=[0, 200],
            health_score_curve=[1.0, 1.0],
            capability_summary={"agent_count": 3, "skill_count": 15},
        )

    def test_assess_multi_agent(self):
        gen = ReportGenerator()
        result = self._make_result()
        cap = gen.assess_capabilities(result)
        assert cap.multi_agent["score"] > 0
        assert cap.multi_agent["agent_count"] == 3

    def test_assess_skill_enhancement(self):
        gen = ReportGenerator()
        result = self._make_result()
        cap = gen.assess_capabilities(result)
        assert cap.skill_enhancement["score"] > 0
        assert cap.skill_enhancement["q_growth"] == 200

    def test_assess_long_task(self):
        gen = ReportGenerator()
        result = self._make_result()
        cap = gen.assess_capabilities(result)
        assert cap.long_task["score"] > 0
        assert cap.long_task["completed"] == 50

    def test_assess_perpetual(self):
        gen = ReportGenerator()
        result = self._make_result()
        cap = gen.assess_capabilities(result)
        assert cap.perpetual["score"] > 0
        assert cap.perpetual["iterations"] == 100

    def test_overall_score(self):
        gen = ReportGenerator()
        result = self._make_result()
        cap = gen.assess_capabilities(result)
        assert 0 < cap.overall_score <= 1.0

    def test_capabilities_to_dict(self):
        gen = ReportGenerator()
        result = self._make_result()
        cap = gen.assess_capabilities(result)
        d = cap.to_dict()
        assert "multi_agent" in d
        assert "skill_enhancement" in d
        assert "long_task" in d
        assert "perpetual" in d
        assert "overall_score" in d


# ============================================================
# 5. 端到端：完整实证运行
# ============================================================
class TestEndToEnd:
    def test_full_empirical_run(self):
        """端到端：完整实证运行并生成报告。"""
        config = LongRunConfig(
            n_iterations=50,
            n_agents=3,
            grid_size=5,
            skill_abstract_interval=3,
            meta_learn_interval=5,
            coalition_interval=8,
            distill_interval=10,
            evolve_interval=7,
            checkpoint_interval=25,
        )
        demo = LongRunDemo(config)
        result = demo.run(verbose=False)

        # 验证运行完成
        assert result.total_iterations == 50
        assert result.total_time > 0

        # 验证有学习发生
        assert result.final_q_entries > 0
        assert len(result.success_rate_curve) > 0

        # 生成报告
        gen = ReportGenerator()
        text_report = gen.generate_text_report(result)
        json_report = gen.generate_json_report(result)
        summary = gen.generate_summary(result)

        # 验证报告内容
        assert "运行报告" in text_report
        assert "四大能力评估" in text_report

        json_data = json.loads(json_report)
        assert json_data["run_summary"]["total_iterations"] == 50

        assert "50轮" in summary

    def test_system_actually_learns(self):
        """实证：系统确实在学习（Q表增长）。"""
        config = LongRunConfig(
            n_iterations=40,
            n_agents=2,
            grid_size=4,
        )
        demo = LongRunDemo(config)
        result = demo.run(verbose=False)

        # Q表必须增长（学习发生的证据）
        if len(result.q_entries_curve) > 1:
            assert result.q_entries_curve[-1] > result.q_entries_curve[0], \
                "Q表未增长，系统未学习"

    def test_system_runs_without_crash(self):
        """实证：系统永续运行不崩溃。"""
        config = LongRunConfig(
            n_iterations=30,
            n_agents=3,
            grid_size=5,
        )
        demo = LongRunDemo(config)
        result = demo.run(verbose=False)

        # 30轮全部完成，无崩溃
        assert result.total_iterations == 30
        assert len(result.success_rate_curve) == 30

    def test_report_assesses_all_four_capabilities(self):
        """报告评估四大能力。"""
        config = LongRunConfig(n_iterations=20, n_agents=3, grid_size=4)
        demo = LongRunDemo(config)
        result = demo.run(verbose=False)

        gen = ReportGenerator()
        cap = gen.assess_capabilities(result)

        # 四大能力都被评估
        assert "score" in cap.multi_agent
        assert "score" in cap.skill_enhancement
        assert "score" in cap.long_task
        assert "score" in cap.perpetual
        assert cap.overall_score > 0
