"""阶段K-2：报告生成器 - 生成可审查的运行报告。

第一性原理：
    实证需要可审查的证据。报告生成器将运行数据转化为：
    1. 文本报告：人类可读的运行总结
    2. JSON报告：机器可读的完整数据
    3. 趋势分析：识别成长/退化模式
    4. 能力评估：量化四大能力的达成度

    报告让系统的"自主进化"变得可验证、可审计、可比较。
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json
import time
import numpy as np

from .long_run_demo import LongRunResult, LongRunDemo


# ============================================================
# 能力评估结果
# ============================================================
@dataclass
class CapabilityAssessment:
    """四大能力评估结果。"""
    multi_agent: Dict[str, Any] = field(default_factory=dict)
    skill_enhancement: Dict[str, Any] = field(default_factory=dict)
    long_task: Dict[str, Any] = field(default_factory=dict)
    perpetual: Dict[str, Any] = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        """综合评分 [0,1]。"""
        scores = []
        for cap in (self.multi_agent, self.skill_enhancement,
                    self.long_task, self.perpetual):
            if "score" in cap:
                scores.append(cap["score"])
        return float(np.mean(scores)) if scores else 0.0

    def to_dict(self) -> Dict:
        return {
            "multi_agent": self.multi_agent,
            "skill_enhancement": self.skill_enhancement,
            "long_task": self.long_task,
            "perpetual": self.perpetual,
            "overall_score": round(self.overall_score, 4),
        }


# ============================================================
# 趋势分析结果
# ============================================================
@dataclass
class TrendAnalysis:
    """趋势分析结果。"""
    success_rate_trend: str = "unknown"
    skill_count_trend: str = "unknown"
    q_entries_trend: str = "unknown"
    health_score_trend: str = "unknown"
    success_rate_slope: float = 0.0
    skill_count_slope: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "success_rate_trend": self.success_rate_trend,
            "skill_count_trend": self.skill_count_trend,
            "q_entries_trend": self.q_entries_trend,
            "health_score_trend": self.health_score_trend,
            "success_rate_slope": round(self.success_rate_slope, 6),
            "skill_count_slope": round(self.skill_count_slope, 6),
        }


# ============================================================
# 报告生成器
# ============================================================
class ReportGenerator:
    """报告生成器：从运行结果生成可审查报告。

    支持格式：
    1. text: 人类可读文本报告
    2. json: 机器可读JSON报告
    3. summary: 简短摘要
    """

    def __init__(self):
        pass

    def generate_text_report(self, result: LongRunResult) -> str:
        """生成文本报告。"""
        lines = []
        lines.append("=" * 70)
        lines.append("  Agent Whitepaper 自主进化系统 - 运行报告")
        lines.append(f"  生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)

        # 运行概况
        lines.append("")
        lines.append("一、运行概况")
        lines.append(f"  总迭代: {result.total_iterations}")
        lines.append(f"  总耗时: {result.total_time:.2f}s")
        lines.append(f"  速度: {result.iterations_per_second:.1f} 轮/秒")

        # 任务执行
        lines.append("")
        lines.append("二、任务执行")
        lines.append(f"  完成: {result.tasks_completed}")
        lines.append(f"  失败: {result.tasks_failed}")
        total = result.tasks_completed + result.tasks_failed
        if total > 0:
            lines.append(f"  完成率: {result.tasks_completed/total:.1%}")

        # 学习成长
        lines.append("")
        lines.append("三、学习成长")
        lines.append(f"  最终成功率: {result.final_success_rate:.4f}")
        lines.append(f"  成功率变化: {result.success_rate_improvement:+.4f}")
        lines.append(f"  技能数: {result.final_skill_count} (增长{result.skill_growth:+d})")
        lines.append(f"  Q表条目: {result.final_q_entries}")

        # 自主进化
        lines.append("")
        lines.append("四、自主进化")
        lines.append(f"  知识蒸馏: {result.total_distillations}次")
        lines.append(f"  自动恢复: {result.total_recoveries}次")

        # 趋势分析
        trend = self.analyze_trends(result)
        lines.append("")
        lines.append("五、趋势分析")
        lines.append(f"  成功率趋势: {trend.success_rate_trend} (斜率={trend.success_rate_slope:.6f})")
        lines.append(f"  技能数趋势: {trend.skill_count_trend} (斜率={trend.skill_count_slope:.6f})")
        lines.append(f"  Q表趋势: {trend.q_entries_trend}")
        lines.append(f"  健康分趋势: {trend.health_score_trend}")

        # 能力评估
        cap = self.assess_capabilities(result)
        lines.append("")
        lines.append("六、四大能力评估")
        lines.append(f"  综合评分: {cap.overall_score:.1%}")
        lines.append(f"  1. 多Agent协作: {cap.multi_agent.get('score', 0):.1%} - {cap.multi_agent.get('detail', '')}")
        lines.append(f"  2. 技能增强: {cap.skill_enhancement.get('score', 0):.1%} - {cap.skill_enhancement.get('detail', '')}")
        lines.append(f"  3. 长任务执行: {cap.long_task.get('score', 0):.1%} - {cap.long_task.get('detail', '')}")
        lines.append(f"  4. 永续运行: {cap.perpetual.get('score', 0):.1%} - {cap.perpetual.get('detail', '')}")

        # 结论
        lines.append("")
        lines.append("七、结论")
        if cap.overall_score >= 0.8:
            lines.append("  系统四大能力均良好达成，自主进化闭环有效运行。")
        elif cap.overall_score >= 0.6:
            lines.append("  系统大部分能力达成，部分能力需进一步优化。")
        else:
            lines.append("  系统能力达成度不足，需检查各阶段配置。")

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)

    def generate_json_report(self, result: LongRunResult) -> str:
        """生成JSON报告。"""
        trend = self.analyze_trends(result)
        cap = self.assess_capabilities(result)

        report = {
            "metadata": {
                "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                "system": "Agent Whitepaper",
            },
            "run_summary": {
                "total_iterations": result.total_iterations,
                "total_time": round(result.total_time, 4),
                "iterations_per_second": round(result.iterations_per_second, 4),
                "tasks_completed": result.tasks_completed,
                "tasks_failed": result.tasks_failed,
            },
            "learning": {
                "final_success_rate": round(result.final_success_rate, 6),
                "success_rate_improvement": round(result.success_rate_improvement, 6),
                "final_skill_count": result.final_skill_count,
                "skill_growth": result.skill_growth,
                "final_q_entries": result.final_q_entries,
            },
            "evolution": {
                "total_distillations": result.total_distillations,
                "total_recoveries": result.total_recoveries,
            },
            "trends": trend.to_dict(),
            "capabilities": cap.to_dict(),
            "curves": {
                "success_rate": result.success_rate_curve,
                "skill_count": result.skill_count_curve,
                "q_entries": result.q_entries_curve,
                "health_score": result.health_score_curve,
            },
        }
        return json.dumps(report, indent=2, ensure_ascii=False)

    def generate_summary(self, result: LongRunResult) -> str:
        """生成简短摘要。"""
        cap = self.assess_capabilities(result)
        return (
            f"运行{result.total_iterations}轮/{result.total_time:.1f}s | "
            f"成功率{result.final_success_rate:.2f}({result.success_rate_improvement:+.2f}) | "
            f"技能{result.final_skill_count}(+{result.skill_growth}) | "
            f"Q表{result.final_q_entries} | "
            f"蒸馏{result.total_distillations}次 | "
            f"恢复{result.total_recoveries}次 | "
            f"综合评分{cap.overall_score:.0%}"
        )

    def analyze_trends(self, result: LongRunResult) -> TrendAnalysis:
        """分析成长趋势。"""
        analysis = TrendAnalysis()

        # 成功率趋势
        analysis.success_rate_trend = self._compute_trend(result.success_rate_curve)
        if len(result.success_rate_curve) >= 2:
            analysis.success_rate_slope = self._compute_slope(result.success_rate_curve)

        # 技能数趋势
        analysis.skill_count_trend = self._compute_trend(result.skill_count_curve)
        if len(result.skill_count_curve) >= 2:
            analysis.skill_count_slope = self._compute_slope(result.skill_count_curve)

        # Q表趋势
        analysis.q_entries_trend = self._compute_trend(result.q_entries_curve)

        # 健康分趋势
        analysis.health_score_trend = self._compute_trend(result.health_score_curve)

        return analysis

    def assess_capabilities(self, result: LongRunResult) -> CapabilityAssessment:
        """评估四大能力达成度。"""
        cap = CapabilityAssessment()

        # 能力1：多Agent协作
        cap.multi_agent = self._assess_multi_agent(result)

        # 能力2：技能增强
        cap.skill_enhancement = self._assess_skill_enhancement(result)

        # 能力3：长任务执行
        cap.long_task = self._assess_long_task(result)

        # 能力4：永续运行
        cap.perpetual = self._assess_perpetual(result)

        return cap

    def _assess_multi_agent(self, result: LongRunResult) -> Dict:
        """评估多Agent协作能力。"""
        agent_count = result.capability_summary.get("agent_count", 0)
        distillations = result.total_distillations
        # 评分：有多个Agent + 有知识蒸馏
        score = 0.0
        if agent_count >= 2:
            score += 0.5
        if agent_count >= 3:
            score += 0.2
        if distillations > 0:
            score += 0.3
        return {
            "score": min(score, 1.0),
            "detail": f"{agent_count}个Agent, {distillations}次蒸馏",
            "agent_count": agent_count,
            "distillations": distillations,
        }

    def _assess_skill_enhancement(self, result: LongRunResult) -> Dict:
        """评估技能增强能力。"""
        q_growth = (
            result.q_entries_curve[-1] - result.q_entries_curve[0]
            if len(result.q_entries_curve) > 1 else 0
        )
        skill_growth = result.skill_growth
        # 评分：Q表增长 + 技能增长
        score = 0.0
        if q_growth > 0:
            score += 0.5
        if skill_growth > 0:
            score += 0.3
        if result.final_q_entries > 10:
            score += 0.2
        return {
            "score": min(score, 1.0),
            "detail": f"Q表+{q_growth}, 技能+{skill_growth}",
            "q_growth": q_growth,
            "skill_growth": skill_growth,
            "final_q": result.final_q_entries,
        }

    def _assess_long_task(self, result: LongRunResult) -> Dict:
        """评估长任务执行能力。"""
        completed = result.tasks_completed
        total = completed + result.tasks_failed
        completion_rate = completed / total if total > 0 else 0.0
        # 评分：有任务完成
        score = 0.0
        if total > 0:
            score += 0.4
        if completed > 0:
            score += 0.4
        if completion_rate > 0.3:
            score += 0.2
        return {
            "score": min(score, 1.0),
            "detail": f"完成{completed}/{total} ({completion_rate:.0%})",
            "completed": completed,
            "total": total,
            "completion_rate": round(completion_rate, 4),
        }

    def _assess_perpetual(self, result: LongRunResult) -> Dict:
        """评估永续运行能力。"""
        iterations = result.total_iterations
        recoveries = result.total_recoveries
        # 评分：完成足够迭代 + 能恢复
        score = 0.0
        if iterations >= 50:
            score += 0.5
        elif iterations >= 20:
            score += 0.3
        elif iterations > 0:
            score += 0.1
        # 有恢复说明系统有韧性
        if recoveries > 0:
            score += 0.2
        # 健康分
        avg_health = (
            float(np.mean(result.health_score_curve))
            if result.health_score_curve else 0.0
        )
        if avg_health > 0.5:
            score += 0.3
        return {
            "score": min(score, 1.0),
            "detail": f"{iterations}轮无中断, 恢复{recoveries}次, 健康{avg_health:.2f}",
            "iterations": iterations,
            "recoveries": recoveries,
            "avg_health": round(avg_health, 4),
        }

    def _compute_trend(self, data: List[float]) -> str:
        """计算趋势。"""
        if len(data) < 2:
            return "insufficient"
        slope = self._compute_slope(data)
        mean_val = float(np.mean(data))
        if abs(mean_val) < 1e-9:
            return "stable" if abs(slope) < 1e-6 else ("rising" if slope > 0 else "falling")
        rel_slope = slope / abs(mean_val)
        if rel_slope > 0.01:
            return "rising"
        elif rel_slope < -0.01:
            return "falling"
        return "stable"

    def _compute_slope(self, data: List[float]) -> float:
        """计算线性回归斜率。"""
        n = len(data)
        if n < 2:
            return 0.0
        xs = list(range(n))
        mean_x = sum(xs) / n
        mean_y = sum(data) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, data))
        den = sum((x - mean_x) ** 2 for x in xs)
        if den == 0:
            return 0.0
        return num / den
