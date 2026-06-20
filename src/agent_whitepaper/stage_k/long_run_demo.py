"""阶段K-1：长时运行实证演示 - 证明系统持续进化。

第一性原理：
    实证 > 声明。一个声称"能永远运行并持续进化"的系统，
    必须通过实际运行来证明。长时运行演示：
    1. 运行足够长的迭代（100+轮）
    2. 记录完整成长曲线
    3. 验证四大能力在长时间运行中持续生效
    4. 生成可审查的运行报告

    这不是"测试"（测试验证正确性），而是"实证"（证明能力）。
    实证让系统从"理论上可行"变为"实际上可行"。
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import time
import numpy as np

from agent_whitepaper.stage_g.orchestrator import (
    SystemOrchestrator, OrchestratorConfig,
)
from agent_whitepaper.stage_g.metrics import MetricsCollector, MetricsSnapshot
from agent_whitepaper.stage_h.config import load_config


# ============================================================
# 长时运行配置
# ============================================================
@dataclass
class LongRunConfig:
    """长时运行配置。"""
    n_iterations: int = 100
    n_agents: int = 3
    grid_size: int = 5
    report_interval: int = 10
    db_path: str = ":memory:"
    # 各阶段间隔（紧凑以在100轮内触发所有能力）
    skill_abstract_interval: int = 3
    meta_learn_interval: int = 5
    coalition_interval: int = 8
    distill_interval: int = 10
    evolve_interval: int = 7
    checkpoint_interval: int = 20
    seed: int = 42


# ============================================================
# 长时运行结果
# ============================================================
@dataclass
class LongRunResult:
    """长时运行结果。"""
    total_iterations: int = 0
    total_time: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    final_success_rate: float = 0.0
    final_skill_count: int = 0
    final_q_entries: int = 0
    total_distillations: int = 0
    total_recoveries: int = 0
    success_rate_curve: List[float] = field(default_factory=list)
    skill_count_curve: List[int] = field(default_factory=list)
    q_entries_curve: List[int] = field(default_factory=list)
    health_score_curve: List[float] = field(default_factory=list)
    capability_summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def iterations_per_second(self) -> float:
        if self.total_time <= 0:
            return 0.0
        return self.total_iterations / self.total_time

    @property
    def success_rate_improvement(self) -> float:
        if len(self.success_rate_curve) < 2:
            return 0.0
        return self.success_rate_curve[-1] - self.success_rate_curve[0]

    @property
    def skill_growth(self) -> int:
        if len(self.skill_count_curve) < 2:
            return 0
        return self.skill_count_curve[-1] - self.skill_count_curve[0]


# ============================================================
# 长时运行演示器
# ============================================================
class LongRunDemo:
    """长时运行演示器：运行系统并收集完整实证数据。

    用法：
        demo = LongRunDemo()
        result = demo.run()
        print(demo.format_report(result))
    """

    def __init__(self, config: Optional[LongRunConfig] = None):
        self.config = config or LongRunConfig()
        self.orchestrator: Optional[SystemOrchestrator] = None

    def run(self, verbose: bool = False) -> LongRunResult:
        """执行长时运行。

        Args:
            verbose: 是否打印进度

        Returns:
            长时运行结果
        """
        # 构建编排器配置
        orch_cfg = OrchestratorConfig(
            grid_size=self.config.grid_size,
            n_agents=self.config.n_agents,
            db_path=self.config.db_path,
            checkpoint_interval=self.config.checkpoint_interval,
            skill_abstract_interval=self.config.skill_abstract_interval,
            meta_learn_interval=self.config.meta_learn_interval,
            distill_interval=self.config.distill_interval,
            evolve_interval=self.config.evolve_interval,
            coalition_interval=self.config.coalition_interval,
            max_iterations=self.config.n_iterations,
            seed=self.config.seed,
        )

        self.orchestrator = SystemOrchestrator(orch_cfg)
        self.orchestrator.initialize()

        start_time = time.time()

        if verbose:
            print(f"启动长时运行: {self.config.n_iterations}轮, "
                  f"{self.config.n_agents}个Agent...")

        # 运行所有迭代
        for i in range(self.config.n_iterations):
            self.orchestrator.run_iteration()

            if verbose and (i + 1) % self.config.report_interval == 0:
                snap = self.orchestrator.metrics.get_latest()
                if snap:
                    elapsed = time.time() - start_time
                    print(f"  [{i+1}/{self.config.n_iterations}] "
                          f"成功率={snap.avg_success_rate:.3f} "
                          f"技能={snap.skill_count} "
                          f"Q={snap.total_q_entries} "
                          f"健康={snap.health_score:.3f} "
                          f"({elapsed:.1f}s)")

        total_time = time.time() - start_time

        # 收集结果
        result = self._collect_result(total_time)

        if verbose:
            print(f"\n运行完成: {total_time:.1f}s, "
                  f"{result.iterations_per_second:.1f}轮/秒")

        return result

    def _collect_result(self, total_time: float) -> LongRunResult:
        """收集运行结果。"""
        orch = self.orchestrator
        metrics = orch.metrics

        # 提取曲线
        success_curve = metrics.get_history("avg_success_rate")
        skill_curve = metrics.get_history("skill_count")
        q_curve = metrics.get_history("total_q_entries")
        health_curve = metrics.get_history("health_score")

        latest = metrics.get_latest()

        result = LongRunResult(
            total_iterations=orch._iteration,
            total_time=total_time,
            tasks_completed=orch._tasks_completed,
            tasks_failed=orch._tasks_failed,
            final_success_rate=latest.avg_success_rate if latest else 0.0,
            final_skill_count=latest.skill_count if latest else 0,
            final_q_entries=latest.total_q_entries if latest else 0,
            total_distillations=metrics.get_cumulative("distillation_count"),
            total_recoveries=metrics.get_cumulative("recovery_count"),
            success_rate_curve=[round(s, 4) for s in success_curve],
            skill_count_curve=list(skill_curve),
            q_entries_curve=list(q_curve),
            health_score_curve=[round(h, 4) for h in health_curve],
            capability_summary=orch.get_metrics(),
        )

        return result

    def format_report(self, result: LongRunResult) -> str:
        """格式化运行报告为可读文本。"""
        lines = []
        lines.append("=" * 70)
        lines.append("  Agent Whitepaper 长时运行实证报告")
        lines.append("=" * 70)
        lines.append("")

        # 运行概况
        lines.append("【运行概况】")
        lines.append(f"  总迭代数: {result.total_iterations}")
        lines.append(f"  总耗时: {result.total_time:.2f}s")
        lines.append(f"  迭代速度: {result.iterations_per_second:.1f} 轮/秒")
        lines.append(f"  Agent数: {self.config.n_agents}")
        lines.append("")

        # 任务执行
        lines.append("【任务执行】")
        lines.append(f"  任务完成: {result.tasks_completed}")
        lines.append(f"  任务失败: {result.tasks_failed}")
        total_tasks = result.tasks_completed + result.tasks_failed
        if total_tasks > 0:
            lines.append(f"  完成率: {result.tasks_completed/total_tasks:.1%}")
        lines.append("")

        # 学习成长
        lines.append("【学习成长】")
        lines.append(f"  最终成功率: {result.final_success_rate:.4f}")
        lines.append(f"  成功率提升: {result.success_rate_improvement:+.4f}")
        lines.append(f"  最终技能数: {result.final_skill_count}")
        lines.append(f"  技能增长: {result.skill_growth:+d}")
        lines.append(f"  Q表条目: {result.final_q_entries}")
        lines.append("")

        # 自主进化
        lines.append("【自主进化】")
        lines.append(f"  知识蒸馏次数: {result.total_distillations}")
        lines.append(f"  自动恢复次数: {result.total_recoveries}")
        avg_health = (
            float(np.mean(result.health_score_curve))
            if result.health_score_curve else 0.0
        )
        lines.append(f"  平均健康分: {avg_health:.4f}")
        lines.append("")

        # 四大能力验证
        lines.append("【四大能力验证】")
        cap = result.capability_summary

        # 能力1：多Agent
        agent_count = cap.get("agent_count", 0)
        lines.append(f"  1. 自主操作多Agent: {agent_count}个Agent协同运行")

        # 能力2：技能增强
        skill_count = cap.get("skill_count", 0)
        q_growth = result.q_entries_curve[-1] - result.q_entries_curve[0] if len(result.q_entries_curve) > 1 else 0
        lines.append(f"  2. 自主增强技能: Q表增长{q_growth}条目, 技能库{skill_count}个")

        # 能力3：长任务
        lines.append(f"  3. 自驱完成长任务: 完成{result.tasks_completed}个任务")

        # 能力4：永续运行
        lines.append(f"  4. 永远运行不停止: {result.total_iterations}轮无中断, "
                      f"恢复{result.total_recoveries}次")
        lines.append("")

        # 成长曲线（简化ASCII图）
        lines.append("【成功率成长曲线】")
        if result.success_rate_curve:
            self._format_ascii_curve(
                lines, result.success_rate_curve, "成功率", 0.0, 1.0
            )
        lines.append("")

        lines.append("【技能数成长曲线】")
        if result.skill_count_curve:
            max_skill = max(result.skill_count_curve) if result.skill_count_curve else 1
            self._format_ascii_curve(
                lines, result.skill_count_curve, "技能数", 0, max(max_skill, 1)
            )
        lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    def _format_ascii_curve(self, lines, data, label, min_val, max_val):
        """格式化ASCII曲线图。"""
        if not data:
            lines.append(f"  {label}: 无数据")
            return

        width = 50
        height = 8
        val_range = max_val - min_val
        if val_range <= 0:
            val_range = 1.0

        # 采样到width个点
        if len(data) > width:
            indices = np.linspace(0, len(data) - 1, width).astype(int)
            sampled = [data[i] for i in indices]
        else:
            sampled = data

        # 生成ASCII图
        for row in range(height, -1, -1):
            threshold = min_val + (val_range * row / height)
            line_str = "  "
            for val in sampled:
                if val >= threshold:
                    line_str += "█"
                else:
                    line_str += " "
            label_str = f"{threshold:.2f}" if isinstance(threshold, float) else str(int(threshold))
            lines.append(f"{label_str:>6} |{line_str}")

        lines.append("       +" + "-" * width)
        lines.append(f"        迭代 0{'':>{width-10}}迭代 {len(data)}")


# ============================================================
# 快速演示函数
# ============================================================
def quick_demo(n_iterations: int = 50, verbose: bool = True) -> LongRunResult:
    """快速演示长时运行。

    Args:
        n_iterations: 迭代次数
        verbose: 是否打印进度

    Returns:
        运行结果
    """
    config = LongRunConfig(n_iterations=n_iterations)
    demo = LongRunDemo(config)
    return demo.run(verbose=verbose)


def full_demo(n_iterations: int = 100, verbose: bool = True) -> str:
    """完整演示并返回报告文本。

    Args:
        n_iterations: 迭代次数
        verbose: 是否打印进度

    Returns:
        报告文本
    """
    config = LongRunConfig(n_iterations=n_iterations)
    demo = LongRunDemo(config)
    result = demo.run(verbose=verbose)
    return demo.format_report(result)
