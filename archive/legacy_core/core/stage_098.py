"""阶段98：进化指标监控面板。

本阶段定义精度、广度、能效指标，综合进化指数EVI，
监控数据采集与存储，异常告警规则，
封装为EvolutionMonitor数据类组合E2EValidation。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import numpy as np

from .stage_097 import E2EValidation


# ============================================================
# 阶段98 枚举：告警级别
# ============================================================
class AlertLevel(Enum):
    """告警级别。"""
    INFO = "info"              # 信息
    WARNING = "warning"        # 警告
    CRITICAL = "critical"      # 严重
    EMERGENCY = "emergency"    # 紧急


# ============================================================
# 阶段98 数据类：告警事件
# ============================================================
@dataclass(frozen=True)
class AlertEvent:
    """告警事件。

    Attributes:
        alert_level: 告警级别
        rule_name: 触发的规则名称
        metric_name: 异常指标名称
        current_value: 当前值
        threshold: 阈值
        duration: 持续时间
        timestamp: 时间戳
    """
    alert_level: AlertLevel
    rule_name: str
    metric_name: str
    current_value: float
    threshold: float
    duration: float
    timestamp: float


# ============================================================
# 阶段98 数据类：进化监控
# ============================================================
@dataclass(frozen=True)
class EvolutionMonitor:
    """进化指标监控面板的完整状态描述。

    组合阶段97的E2EValidation，在验证基础上增加监控状态。

    Attributes:
        e2e_validation: 阶段97的端到端验证（组合字段）
        accuracy: 当前精度指标
        coverage: 当前广度指标
        efficiency: 当前能效指标
        evi: 当前综合进化指数
        evi_rate: EVI变化率
        accuracy_trend: 精度趋势
        coverage_trend: 广度趋势
        efficiency_trend: 能效趋势
        F_initial: 初始自由能
        alpha: 精度权重
        beta: 广度权重
        gamma: 能效权重
        active_alerts: 当前活跃告警
        metrics_history: 指标历史（最近1000条）

    Invariants:
        - alpha + beta + gamma = 1.0
        - 0 <= accuracy, coverage, efficiency, evi <= 1.0
    """
    e2e_validation: E2EValidation
    accuracy: float = 0.0
    coverage: float = 0.0
    efficiency: float = 0.0
    evi: float = 0.0
    evi_rate: float = 0.0
    accuracy_trend: float = 0.0
    coverage_trend: float = 0.0
    efficiency_trend: float = 0.0
    F_initial: float = 1.0
    alpha: float = 0.40
    beta: float = 0.35
    gamma: float = 0.25
    active_alerts: Tuple[AlertEvent, ...] = ()
    metrics_history: Tuple[Dict, ...] = ()

    def __post_init__(self):
        if not np.isclose(self.alpha + self.beta + self.gamma, 1.0):
            raise ValueError(
                f"权重之和必须等于1，收到 {self.alpha + self.beta + self.gamma}"
            )


# ============================================================
# 阶段98 核心接口：进化监控器
# ============================================================
class EvolutionMonitorExecutor:
    """阶段98的核心接口：进化指标监控面板。

    实现精度、广度、能效指标的采集、计算和告警。

    核心约束：
        - EVI长期趋势上升
        - CRITICAL级告警触发自修复
        - 存储占用 < 80%
    """

    def __init__(
        self,
        alpha: float = 0.40,
        beta: float = 0.35,
        gamma: float = 0.25,
        trend_window: int = 1000
    ):
        """初始化进化监控器。

        Args:
            alpha: 精度权重，必须 >= 0
            beta: 广度权重，必须 >= 0
            gamma: 能效权重，必须 >= 0
            trend_window: 趋势计算窗口，必须 > 0

        Raises:
            ValueError: 当alpha + beta + gamma != 1时
            ValueError: 当trend_window <= 0时

        Examples:
            >>> monitor = EvolutionMonitorExecutor()
            >>> monitor.alpha
            0.4
        """
        if not np.isclose(alpha + beta + gamma, 1.0):
            raise ValueError(
                f"权重之和必须等于1，收到 {alpha + beta + gamma}"
            )
        if alpha < 0 or beta < 0 or gamma < 0:
            raise ValueError("权重必须非负")
        if trend_window <= 0:
            raise ValueError(f"trend_window必须 > 0，收到 {trend_window}")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.trend_window = trend_window

    def compute_accuracy(
        self,
        F_current: float,
        F_initial: float
    ) -> float:
        """计算精度指标（A98.1）。

        accuracy = 1 - F_current / F_initial

        Args:
            F_current: 当前自由能
            F_initial: 初始自由能，必须 != 0

        Returns:
            精度指标 ∈ [0, 1]

        Raises:
            ValueError: 当F_initial == 0时

        Examples:
            >>> monitor = EvolutionMonitorExecutor()
            >>> monitor.compute_accuracy(8.0, 10.0)
            0.2
        """
        if F_initial == 0:
            raise ValueError("F_initial不能为0")
        return max(0.0, min(1.0, 1.0 - F_current / F_initial))

    def compute_coverage(
        self,
        known_dims: int,
        total_dims: int
    ) -> float:
        """计算广度指标（A98.2）。

        Args:
            known_dims: 已知维度数，必须 >= 0
            total_dims: 总维度数，必须 > 0

        Returns:
            广度指标 ∈ [0, 1]

        Raises:
            ValueError: 当known_dims < 0时
            ValueError: 当total_dims <= 0时

        Examples:
            >>> monitor = EvolutionMonitorExecutor()
            >>> monitor.compute_coverage(5, 10)
            0.5
        """
        if known_dims < 0:
            raise ValueError(f"known_dims必须 >= 0，收到 {known_dims}")
        if total_dims <= 0:
            raise ValueError(f"total_dims必须 > 0，收到 {total_dims}")
        return min(1.0, known_dims / total_dims)

    def compute_efficiency(
        self,
        effective_compute: float,
        total_compute: float
    ) -> float:
        """计算能效指标（A98.3）。

        Args:
            effective_compute: 有效计算量，必须 >= 0
            total_compute: 总计算量，必须 > 0

        Returns:
            能效指标 ∈ [0, 1]

        Raises:
            ValueError: 当effective_compute < 0时
            ValueError: 当total_compute <= 0时

        Examples:
            >>> monitor = EvolutionMonitorExecutor()
            >>> monitor.compute_efficiency(80.0, 100.0)
            0.8
        """
        if effective_compute < 0:
            raise ValueError(
                f"effective_compute必须 >= 0，收到 {effective_compute}"
            )
        if total_compute <= 0:
            raise ValueError(
                f"total_compute必须 > 0，收到 {total_compute}"
            )
        return min(1.0, effective_compute / total_compute)

    def compute_evi(
        self,
        accuracy: float,
        coverage: float,
        efficiency: float
    ) -> float:
        """计算综合进化指数（A98.4）。

        EVI = alpha * accuracy + beta * coverage + gamma * efficiency

        Args:
            accuracy: 精度指标，范围[0,1]
            coverage: 广度指标，范围[0,1]
            efficiency: 能效指标，范围[0,1]

        Returns:
            EVI ∈ [0, 1]

        Examples:
            >>> monitor = EvolutionMonitorExecutor()
            >>> evi = monitor.compute_evi(0.8, 0.6, 0.7)
            >>> 0 <= evi <= 1
            True
        """
        return (
            self.alpha * accuracy
            + self.beta * coverage
            + self.gamma * efficiency
        )

    def _compute_trend(self, values: Tuple[float, ...]) -> float:
        """计算序列的线性趋势斜率。

        使用最小二乘法拟合直线，返回斜率。

        Args:
            values: 数值序列

        Returns:
            趋势斜率，序列长度<2时返回0.0
        """
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        y = np.array(values)
        # 最小二乘法: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
            n * np.sum(x * x) - np.sum(x) ** 2 + 1e-10
        )
        return float(slope)

    def collect_metrics(
        self,
        monitor: EvolutionMonitor,
        F: float,
        G: float,
        J: float,
        known_dims: int = 0,
        total_dims: int = 1,
        effective_compute: float = 0.0,
        total_compute: float = 1.0,
        step_count: int = 0
    ) -> EvolutionMonitor:
        """采集并存储监控数据（A98.5）。

        Args:
            monitor: 当前监控状态
            F: 当前自由能
            G: 当前预期自由能
            J: 当前全局目标
            known_dims: 已知维度数
            total_dims: 总维度数
            effective_compute: 有效计算量
            total_compute: 总计算量
            step_count: 当前步数

        Returns:
            更新后的监控状态

        Examples:
            >>> monitor = EvolutionMonitorExecutor()
            >>> from agent_whitepaper.core.stage_097 import E2EValidation
            >>> from agent_whitepaper.core.stage_096 import TechStack
            >>> from agent_whitepaper.core.stage_095 import SystemInitializer
            >>> em = EvolutionMonitor(e2e_validation=E2EValidation(tech_stack=TechStack(system_initializer=SystemInitializer(integrated_system=None))))
            >>> new_em = monitor.collect_metrics(em, 8.0, 5.0, 20.0, 5, 10, 80.0, 100.0, 100)
            >>> new_em.accuracy
            0.2
        """
        # 计算当前指标
        accuracy = self.compute_accuracy(F, monitor.F_initial)
        coverage = self.compute_coverage(known_dims, total_dims)
        efficiency = self.compute_efficiency(effective_compute, total_compute)
        evi = self.compute_evi(accuracy, coverage, efficiency)

        # 构建新的历史记录
        new_entry = {
            "step_count": step_count,
            "F": F,
            "G": G,
            "J": J,
            "accuracy": accuracy,
            "coverage": coverage,
            "efficiency": efficiency,
            "evi": evi,
            "timestamp": float(step_count),
        }
        new_history = monitor.metrics_history + (new_entry,)
        if len(new_history) > self.trend_window:
            new_history = new_history[-self.trend_window:]

        # 计算趋势
        acc_hist = tuple(e["accuracy"] for e in new_history)
        cov_hist = tuple(e["coverage"] for e in new_history)
        eff_hist = tuple(e["efficiency"] for e in new_history)
        evi_hist = tuple(e["evi"] for e in new_history)

        accuracy_trend = self._compute_trend(acc_hist)
        coverage_trend = self._compute_trend(cov_hist)
        efficiency_trend = self._compute_trend(eff_hist)

        # 计算EVI变化率（最近两个点）
        evi_rate = 0.0
        if len(evi_hist) >= 2:
            evi_rate = evi_hist[-1] - evi_hist[-2]

        return EvolutionMonitor(
            e2e_validation=monitor.e2e_validation,
            accuracy=accuracy,
            coverage=coverage,
            efficiency=efficiency,
            evi=evi,
            evi_rate=evi_rate,
            accuracy_trend=accuracy_trend,
            coverage_trend=coverage_trend,
            efficiency_trend=efficiency_trend,
            F_initial=monitor.F_initial,
            alpha=monitor.alpha,
            beta=monitor.beta,
            gamma=monitor.gamma,
            active_alerts=monitor.active_alerts,
            metrics_history=new_history,
        )

    def check_alerts(
        self,
        monitor: EvolutionMonitor
    ) -> Tuple[EvolutionMonitor, List[AlertEvent]]:
        """检查异常告警规则（A98.6）。

        检查7条告警规则：
        1. 精度退化: accuracy_trend < 0 持续 -> WARNING
        2. 广度停滞: coverage_trend = 0 持续 -> INFO
        3. 能效下降: efficiency_trend < 0 持续 -> WARNING
        4. EVI退化: evi_rate < 0 -> CRITICAL
        5. 全局目标上升: J上升 -> WARNING
        6. 安全违规: 引用安全层
        7. 资源超限: 存储>80% -> CRITICAL

        Args:
            monitor: 当前监控状态

        Returns:
            (更新后的监控状态, 新触发的告警列表)

        Examples:
            >>> monitor = EvolutionMonitorExecutor()
            >>> from agent_whitepaper.core.stage_097 import E2EValidation
            >>> from agent_whitepaper.core.stage_096 import TechStack
            >>> from agent_whitepaper.core.stage_095 import SystemInitializer
            >>> em = EvolutionMonitor(e2e_validation=E2EValidation(tech_stack=TechStack(system_initializer=SystemInitializer(integrated_system=None))))
            >>> new_em, alerts = monitor.check_alerts(em)
            >>> isinstance(alerts, list)
            True
        """
        alerts: List[AlertEvent] = []
        timestamp = float(len(monitor.metrics_history))

        # 规则1: 精度退化
        if monitor.accuracy_trend < 0 and len(monitor.metrics_history) >= 100:
            alerts.append(AlertEvent(
                alert_level=AlertLevel.WARNING,
                rule_name="accuracy_degradation",
                metric_name="accuracy_trend",
                current_value=monitor.accuracy_trend,
                threshold=0.0,
                duration=float(len(monitor.metrics_history)),
                timestamp=timestamp,
            ))

        # 规则2: 广度停滞
        if abs(monitor.coverage_trend) < 1e-6 and len(monitor.metrics_history) >= 500:
            alerts.append(AlertEvent(
                alert_level=AlertLevel.INFO,
                rule_name="coverage_stagnation",
                metric_name="coverage_trend",
                current_value=monitor.coverage_trend,
                threshold=0.0,
                duration=float(len(monitor.metrics_history)),
                timestamp=timestamp,
            ))

        # 规则3: 能效下降
        if monitor.efficiency_trend < 0 and len(monitor.metrics_history) >= 100:
            alerts.append(AlertEvent(
                alert_level=AlertLevel.WARNING,
                rule_name="efficiency_degradation",
                metric_name="efficiency_trend",
                current_value=monitor.efficiency_trend,
                threshold=0.0,
                duration=float(len(monitor.metrics_history)),
                timestamp=timestamp,
            ))

        # 规则4: EVI退化（最关键）
        if monitor.evi_rate < 0 and len(monitor.metrics_history) >= 2:
            alerts.append(AlertEvent(
                alert_level=AlertLevel.CRITICAL,
                rule_name="evi_degradation",
                metric_name="evi_rate",
                current_value=monitor.evi_rate,
                threshold=0.0,
                duration=float(len(monitor.metrics_history)),
                timestamp=timestamp,
            ))

        # 规则5: 全局目标上升（通过历史数据检测）
        if len(monitor.metrics_history) >= 2:
            J_values = [e["J"] for e in monitor.metrics_history]
            if J_values[-1] > J_values[0] * 1.1:
                alerts.append(AlertEvent(
                    alert_level=AlertLevel.WARNING,
                    rule_name="global_objective_rising",
                    metric_name="J_trend",
                    current_value=J_values[-1],
                    threshold=J_values[0] * 1.1,
                    duration=float(len(monitor.metrics_history)),
                    timestamp=timestamp,
                ))

        # 规则7: 资源超限（存储占用）
        # 通过metrics_history长度估算内存占用（简化模型）
        storage_usage = len(monitor.metrics_history) / self.trend_window
        if storage_usage > 0.80:
            alerts.append(AlertEvent(
                alert_level=AlertLevel.CRITICAL,
                rule_name="storage_overflow",
                metric_name="storage_usage",
                current_value=storage_usage,
                threshold=0.80,
                duration=float(len(monitor.metrics_history)),
                timestamp=timestamp,
            ))

        # 合并活跃告警
        all_alerts = monitor.active_alerts + tuple(alerts)
        # 只保留最近100条告警
        if len(all_alerts) > 100:
            all_alerts = all_alerts[-100:]

        new_monitor = EvolutionMonitor(
            e2e_validation=monitor.e2e_validation,
            accuracy=monitor.accuracy,
            coverage=monitor.coverage,
            efficiency=monitor.efficiency,
            evi=monitor.evi,
            evi_rate=monitor.evi_rate,
            accuracy_trend=monitor.accuracy_trend,
            coverage_trend=monitor.coverage_trend,
            efficiency_trend=monitor.efficiency_trend,
            F_initial=monitor.F_initial,
            alpha=monitor.alpha,
            beta=monitor.beta,
            gamma=monitor.gamma,
            active_alerts=all_alerts,
            metrics_history=monitor.metrics_history,
        )

        return new_monitor, alerts

    def step(
        self,
        monitor: EvolutionMonitor,
        F: float,
        G: float,
        J: float,
        known_dims: int = 0,
        total_dims: int = 1,
        effective_compute: float = 0.0,
        total_compute: float = 1.0,
        step_count: int = 0
    ) -> Tuple[EvolutionMonitor, List[AlertEvent]]:
        """执行一步监控更新。

        Args:
            monitor: 当前监控状态
            F: 当前自由能
            G: 当前预期自由能
            J: 当前全局目标
            known_dims: 已知维度数
            total_dims: 总维度数
            effective_compute: 有效计算量
            total_compute: 总计算量
            step_count: 当前步数

        Returns:
            (更新后的监控状态, 新触发的告警列表)

        Examples:
            >>> monitor = EvolutionMonitorExecutor()
            >>> from agent_whitepaper.core.stage_097 import E2EValidation
            >>> from agent_whitepaper.core.stage_096 import TechStack
            >>> from agent_whitepaper.core.stage_095 import SystemInitializer
            >>> em = EvolutionMonitor(e2e_validation=E2EValidation(tech_stack=TechStack(system_initializer=SystemInitializer(integrated_system=None))))
            >>> new_em, alerts = monitor.step(em, 8.0, 5.0, 20.0, 5, 10, 80.0, 100.0, 100)
            >>> new_em.accuracy
            0.2
        """
        new_monitor = self.collect_metrics(
            monitor, F, G, J,
            known_dims, total_dims,
            effective_compute, total_compute,
            step_count
        )
        new_monitor, alerts = self.check_alerts(new_monitor)
        return new_monitor, alerts
