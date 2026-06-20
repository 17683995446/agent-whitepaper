"""阶段H-3：进度反馈 - 系统的"仪表盘"。

第一性原理：
    "Feedback is essential"——用户需要知道系统在工作。
    长时运行若无反馈，用户会怀疑系统卡死。进度反馈回答四个问题：
    1. 进度：完成了多少？还剩多少？
    2. 速度：多快？每秒多少迭代？
    3. 预估：还要多久？（ETA）
    4. 状态：当前在做什么？系统健康吗？

    进度反馈让长时运行从"黑盒"变为"透明"。
"""

import time
from typing import Any, Callable, List, Optional


# ============================================================
# 时间格式化
# ============================================================
def format_duration(seconds: float) -> str:
    """将秒数格式化为人类可读的时长。

    Examples:
        5.0 -> "5.0s"
        65.0 -> "1.1m"
        3700.0 -> "1.0h"
    """
    if seconds < 0:
        return "未知"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


# ============================================================
# 进度条渲染
# ============================================================
def render_progress_bar(
    current: int, total: int, width: int = 30
) -> str:
    """渲染进度条。

    Args:
        current: 当前进度
        total: 总量
        width: 进度条宽度（字符数）

    Returns:
        形如 "[████████░░░░░░░░░░] 40/100 (40%)" 的字符串
    """
    if total <= 0:
        return ""
    current = max(0, min(current, total))
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    percentage = int(100 * current / total)
    return f"[{bar}] {current}/{total} ({percentage}%)"


# ============================================================
# 阶段预测
# ============================================================
def predict_phases(
    iteration: int,
    skill_abstract_interval: int = 5,
    meta_learn_interval: int = 10,
    coalition_interval: int = 15,
    distill_interval: int = 30,
    evolve_interval: int = 20,
    checkpoint_interval: int = 50,
    n_agents: int = 3,
) -> List[str]:
    """预测当前迭代会执行哪些阶段。

    基于 OrchestratorConfig 的间隔参数，判断本轮迭代触发的阶段。
    每轮必做：健康检查、任务执行、指标采集。

    Args:
        iteration: 当前迭代轮数（从1开始）
        各 *_interval 参数：对应阶段的触发间隔
        n_agents: Agent数量（影响联盟和蒸馏是否触发）

    Returns:
        阶段名称列表，如 ["健康检查", "任务执行", "技能抽象", "指标采集"]
    """
    phases = ["健康检查", "任务执行"]
    if iteration % skill_abstract_interval == 0:
        phases.append("技能抽象")
    if iteration % meta_learn_interval == 0:
        phases.append("元学习")
    if iteration % coalition_interval == 0 and n_agents > 1:
        phases.append("联盟协作")
    if iteration % distill_interval == 0 and n_agents > 1:
        phases.append("知识蒸馏")
    if iteration % evolve_interval == 0:
        phases.append("自主进化")
    if iteration % checkpoint_interval == 0:
        phases.append("检查点")
    phases.append("指标采集")
    return phases


# ============================================================
# 进度报告器
# ============================================================
class ProgressReporter:
    """进度报告器：为长时运行提供实时进度反馈。

    第一性原理：
        "Feedback is essential"——用户需要知道系统在工作。
        ProgressReporter 将"黑盒运行"变为"透明运行"。

    用法：
        reporter = ProgressReporter(total=100, report_interval=10)
        for i in range(100):
            do_work()
            reporter.report(i + 1, snapshot=get_snapshot())
        reporter.finish()

    特性：
        - 有限模式：显示进度条、百分比、ETA
        - 无限模式：显示迭代数、运行时长、速率
        - 可定制输出函数（默认print，可替换为日志/GUI）
        - 支持指标快照显示
    """

    def __init__(
        self,
        total: Optional[int] = None,
        report_interval: int = 10,
        output: Callable[[str], None] = print,
        bar_width: int = 30,
        show_bar: bool = True,
    ) -> None:
        """初始化进度报告器。

        Args:
            total: 总迭代数（None=无限模式）
            report_interval: 报告间隔（每N次迭代报告一次）
            output: 输出函数（默认print）
            bar_width: 进度条宽度
            show_bar: 是否显示进度条
        """
        self.total = total
        self.report_interval = max(1, report_interval)
        self.output = output
        self.bar_width = bar_width
        self.show_bar = show_bar and total is not None
        self.start_time = time.time()
        self.iteration = 0
        self._last_report_time = self.start_time

    def should_report(self, iteration: int) -> bool:
        """判断当前迭代是否应该报告。"""
        return iteration % self.report_interval == 0 or iteration == self.total

    def report(
        self,
        iteration: int,
        snapshot: Optional[Any] = None,
        phases: Optional[List[str]] = None,
    ) -> None:
        """报告当前进度。

        Args:
            iteration: 当前迭代轮数（从1开始）
            snapshot: 指标快照（需有 avg_success_rate, skill_count,
                      health_score 等属性，或为None）
            phases: 当前迭代执行的阶段列表
        """
        self.iteration = iteration
        if not self.should_report(iteration):
            return

        elapsed = time.time() - self.start_time
        rate = iteration / elapsed if elapsed > 0 else 0.0
        line = self._format_line(iteration, elapsed, rate, snapshot, phases)
        self.output(line)
        self._last_report_time = time.time()

    def _format_line(
        self,
        iteration: int,
        elapsed: float,
        rate: float,
        snapshot: Optional[Any],
        phases: Optional[List[str]],
    ) -> str:
        """格式化进度行。"""
        parts: List[str] = []

        # 进度条或迭代计数
        if self.show_bar and self.total is not None:
            parts.append(
                render_progress_bar(iteration, self.total, self.bar_width)
            )
        else:
            parts.append(f"[{iteration}]")

        # 速率和时长
        parts.append(f"{format_duration(elapsed)} | {rate:.1f} it/s")

        # ETA（仅有限模式）
        if self.total is not None and rate > 0:
            remaining = (self.total - iteration) / rate
            parts.append(f"ETA: {format_duration(remaining)}")

        # 指标快照
        if snapshot is not None:
            metrics = self._extract_metrics(snapshot)
            if metrics:
                parts.append(metrics)

        # 阶段
        if phases:
            parts.append(f"[{'|'.join(phases)}]")

        return " | ".join(parts)

    def _extract_metrics(self, snapshot: Any) -> str:
        """从快照中提取关键指标。

        支持两种输入：
        - 有属性的对象（MetricsSnapshot）
        - 字典
        """
        def get(key: str, default: Any = 0) -> Any:
            if isinstance(snapshot, dict):
                return snapshot.get(key, default)
            return getattr(snapshot, key, default)

        success_rate = get("avg_success_rate", None)
        skill_count = get("skill_count", None)
        health = get("health_score", None)

        bits: List[str] = []
        if success_rate is not None:
            bits.append(f"成功率={success_rate:.3f}")
        if skill_count is not None:
            bits.append(f"技能={skill_count}")
        if health is not None:
            bits.append(f"健康={health:.3f}")
        return " ".join(bits)

    def finish(
        self,
        snapshot: Optional[Any] = None,
        phases: Optional[List[str]] = None,
    ) -> None:
        """报告完成。

        在所有迭代结束后调用，输出最终状态。
        """
        elapsed = time.time() - self.start_time
        iteration = self.iteration if self.iteration > 0 else 0
        rate = iteration / elapsed if elapsed > 0 and iteration > 0 else 0.0

        parts: List[str] = []
        parts.append("完成")
        parts.append(f"{format_duration(elapsed)} | {rate:.1f} it/s")
        if iteration > 0:
            parts.append(f"共{iteration}轮")
        if snapshot is not None:
            metrics = self._extract_metrics(snapshot)
            if metrics:
                parts.append(metrics)

        self.output(" | ".join(parts))


# ============================================================
# 无输出报告器（静默模式）
# ============================================================
class SilentReporter(ProgressReporter):
    """静默报告器：不产生任何输出。

    用于 --quiet 模式或程序化调用时不需要输出的场景。
    """

    def __init__(self, total: Optional[int] = None, **kwargs) -> None:
        super().__init__(
            total=total,
            output=lambda _: None,
            show_bar=False,
            **{k: v for k, v in kwargs.items() if k not in ("output", "show_bar")},
        )
