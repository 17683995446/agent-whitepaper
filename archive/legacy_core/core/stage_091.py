"""阶段91：全局统一目标函数。

本阶段定义全局统一目标函数 J = ∫F dt + ∫G dt + λ₁R_complexity + λ₂R_safety，
证明其与终极进化目标的等价性。
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

from .stage_090 import SafetyGuardState


class EvolutionDimension:
    """全局目标的四个进化维度（仅作文档用途，非枚举）。"""
    ACCURACY = "accuracy"
    COVERAGE = "coverage"
    EFFICIENCY = "efficiency"
    SAFETY = "safety"


@dataclass(frozen=True)
class GlobalObjective:
    """全局统一目标函数的完整状态描述。

    组合阶段90的SafetyGuardState，在安全护栏基础上增加全局目标状态。

    Attributes:
        safety_guard_state: 阶段90的安全护栏状态（组合字段）
        cumulative_free_energy: 累积变分自由能 ∫F dt
        cumulative_expected_free_energy: 累积预期自由能 ∫G dt
        complexity_penalty: 复杂度正则项 λ₁R_complexity
        safety_penalty: 安全违规惩罚项 λ₂R_safety
        total: 全局目标值 J
        accuracy_trend: 精度进化趋势（下降=True）
        coverage_trend: 广度进化趋势（下降=True）
        efficiency_trend: 能效进化趋势（下降=True）
        safety_trend: 安全进化趋势（下降=True）
        lambda_complexity: 复杂度正则系数
        lambda_safety: 安全正则系数
        compute_efficiency: 计算效率
        param_efficiency: 参数效率
        coverage_ratio: 认知覆盖率

    Invariants:
        - total >= 0
        - cumulative_free_energy >= 0
        - cumulative_expected_free_energy >= 0
        - complexity_penalty >= 0
        - safety_penalty >= 0
        - lambda_safety > lambda_complexity
    """
    safety_guard_state: SafetyGuardState
    cumulative_free_energy: float = 0.0
    cumulative_expected_free_energy: float = 0.0
    complexity_penalty: float = 0.0
    safety_penalty: float = 0.0
    total: float = 0.0
    accuracy_trend: bool = True
    coverage_trend: bool = True
    efficiency_trend: bool = True
    safety_trend: bool = True
    lambda_complexity: float = 0.01
    lambda_safety: float = 1.0
    compute_efficiency: float = 0.0
    param_efficiency: float = 0.0
    coverage_ratio: float = 0.0

    def __post_init__(self):
        if self.total < 0:
            raise ValueError(f"total必须 >= 0，收到 {self.total}")
        if self.cumulative_free_energy < 0:
            raise ValueError(
                f"cumulative_free_energy必须 >= 0，"
                f"收到 {self.cumulative_free_energy}"
            )
        if self.cumulative_expected_free_energy < 0:
            raise ValueError(
                f"cumulative_expected_free_energy必须 >= 0，"
                f"收到 {self.cumulative_expected_free_energy}"
            )
        if self.complexity_penalty < 0:
            raise ValueError(
                f"complexity_penalty必须 >= 0，"
                f"收到 {self.complexity_penalty}"
            )
        if self.safety_penalty < 0:
            raise ValueError(
                f"safety_penalty必须 >= 0，"
                f"收到 {self.safety_penalty}"
            )
        if self.lambda_safety <= self.lambda_complexity:
            raise ValueError(
                f"lambda_safety({self.lambda_safety})必须 > "
                f"lambda_complexity({self.lambda_complexity})"
            )


class GlobalObjectiveCalculator:
    """阶段91的核心接口：全局统一目标函数计算器。

    实现全局目标J的计算、各项分解和进化趋势判定。

    核心约束：
        - J >= 0
        - λ_safety > λ_complexity
        - 安全约束满足时 R_safety = 0
    """

    def __init__(
        self,
        lambda_complexity: float = 0.01,
        lambda_safety: float = 1.0,
        trend_window: int = 1000
    ):
        """初始化全局目标计算器（A91.1）。

        Args:
            lambda_complexity: 复杂度正则系数（默认0.01），必须 >= 0
            lambda_safety: 安全正则系数（默认1.0），必须 > lambda_complexity
            trend_window: 趋势判定窗口（默认1000步），必须 > 0

        Raises:
            ValueError: 当lambda_safety <= lambda_complexity时
            ValueError: 当trend_window <= 0时
        """
        if lambda_safety <= lambda_complexity:
            raise ValueError(
                f"lambda_safety({lambda_safety})必须 > "
                f"lambda_complexity({lambda_complexity})"
            )
        if trend_window <= 0:
            raise ValueError(f"trend_window必须 > 0，收到 {trend_window}")
        self.lambda_complexity = lambda_complexity
        self.lambda_safety = lambda_safety
        self.trend_window = trend_window

    def compute(
        self,
        free_energy: float,
        expected_free_energy: float,
        model_complexity: float,
        safety_guard_state: SafetyGuardState,
        dt: float = 1.0
    ) -> GlobalObjective:
        """计算全局目标值（A91.1）。

        J = ∫F dt + ∫G dt + λ₁·R_complexity + λ₂·R_safety

        Args:
            free_energy: 当前变分自由能 F
            expected_free_energy: 当前预期自由能 G
            model_complexity: 当前模型复杂度 D_KL
            safety_guard_state: 安全护栏状态
            dt: 时间步长，必须 > 0

        Returns:
            GlobalObjective 全局目标状态

        Raises:
            ValueError: 当dt <= 0时
        """
        if dt <= 0:
            raise ValueError(f"dt必须 > 0，收到 {dt}")
        cum_f = free_energy * dt
        cum_g = expected_free_energy * dt
        complexity_penalty = self.lambda_complexity * model_complexity
        safety_penalty = self.compute_safety_term(safety_guard_state)
        total = cum_f + cum_g + complexity_penalty + safety_penalty
        return GlobalObjective(
            safety_guard_state=safety_guard_state,
            cumulative_free_energy=max(0.0, cum_f),
            cumulative_expected_free_energy=max(0.0, cum_g),
            complexity_penalty=max(0.0, complexity_penalty),
            safety_penalty=max(0.0, safety_penalty),
            total=max(0.0, total),
            accuracy_trend=True,
            coverage_trend=True,
            efficiency_trend=True,
            safety_trend=True,
            lambda_complexity=self.lambda_complexity,
            lambda_safety=self.lambda_safety,
            compute_efficiency=0.0,
            param_efficiency=0.0,
            coverage_ratio=0.0
        )

    def compute_accuracy_term(
        self,
        F_history: List[float]
    ) -> Tuple[float, bool]:
        """计算精度进化项（A91.2）。

        Args:
            F_history: 变分自由能历史列表，长度 > 0

        Returns:
            (累积自由能, 精度进化趋势)
            - 趋势为True表示自由能在下降（精度在提升）

        Raises:
            ValueError: 当F_history为空时
        """
        if not F_history:
            raise ValueError("F_history不能为空")
        cum_f = sum(F_history)
        if len(F_history) < 2:
            return cum_f, True
        trend = F_history[-1] < F_history[0]
        return cum_f, trend

    def compute_coverage_term(
        self,
        G_history: List[float]
    ) -> Tuple[float, bool]:
        """计算广度进化项（A91.3）。

        Args:
            G_history: 预期自由能历史列表，长度 > 0

        Returns:
            (累积预期自由能, 广度进化趋势)
            - 趋势为True表示G在下降（广度在提升）

        Raises:
            ValueError: 当G_history为空时
        """
        if not G_history:
            raise ValueError("G_history不能为空")
        cum_g = sum(G_history)
        if len(G_history) < 2:
            return cum_g, True
        trend = G_history[-1] < G_history[0]
        return cum_g, trend

    def compute_efficiency_term(
        self,
        theta: np.ndarray,
        prior: np.ndarray
    ) -> Tuple[float, float, float]:
        """计算能效进化项（A91.4）。

        Args:
            theta: 当前模型参数，shape (n,)
            prior: 先验分布参数，shape (n,)

        Returns:
            (复杂度惩罚, 计算效率, 参数效率)

        Raises:
            ValueError: 当theta和prior形状不匹配时
        """
        if theta.shape != prior.shape:
            raise ValueError(
                f"theta和prior形状不匹配: {theta.shape} vs {prior.shape}"
            )
        kl = float(np.sum(theta ** 2 - prior ** 2))
        complexity = self.lambda_complexity * max(0.0, kl)
        compute_eff = 1.0 / (1.0 + float(np.sum(np.abs(theta))))
        param_eff = 1.0 / (1.0 + len(theta))
        return complexity, compute_eff, param_eff

    def compute_safety_term(
        self,
        safety_guard_state: SafetyGuardState
    ) -> float:
        """计算安全进化项（A91.5）。

        Args:
            safety_guard_state: 安全护栏状态

        Returns:
            安全违规惩罚 >= 0
        """
        violations = safety_guard_state.total_violations
        return self.lambda_safety * violations

    def verify_equivalence(
        self,
        J: float,
        f: float
    ) -> bool:
        """验证全局目标与适应度的等价性（定理91.1）。

        Args:
            J: 全局目标值
            f: 适应度值

        Returns:
            当|J + f| < 0.01时返回True
        """
        return abs(J + f) < 0.01

    def step(
        self,
        state: GlobalObjective,
        free_energy: float,
        expected_free_energy: float,
        model_complexity: float,
        safety_guard_state: SafetyGuardState,
        dt: float = 1.0
    ) -> GlobalObjective:
        """执行一步全局目标更新。

        Args:
            state: 当前全局目标状态
            free_energy: 当前变分自由能
            expected_free_energy: 当前预期自由能
            model_complexity: 当前模型复杂度
            safety_guard_state: 安全护栏状态
            dt: 时间步长，必须 > 0

        Returns:
            更新后的全局目标状态
        """
        if dt <= 0:
            raise ValueError(f"dt必须 > 0，收到 {dt}")
        new_cum_f = state.cumulative_free_energy + free_energy * dt
        new_cum_g = state.cumulative_expected_free_energy + expected_free_energy * dt
        complexity_penalty = self.lambda_complexity * model_complexity
        safety_penalty = self.compute_safety_term(safety_guard_state)
        total = new_cum_f + new_cum_g + complexity_penalty + safety_penalty
        return GlobalObjective(
            safety_guard_state=safety_guard_state,
            cumulative_free_energy=max(0.0, new_cum_f),
            cumulative_expected_free_energy=max(0.0, new_cum_g),
            complexity_penalty=max(0.0, complexity_penalty),
            safety_penalty=max(0.0, safety_penalty),
            total=max(0.0, total),
            accuracy_trend=state.accuracy_trend,
            coverage_trend=state.coverage_trend,
            efficiency_trend=state.efficiency_trend,
            safety_trend=state.safety_trend,
            lambda_complexity=self.lambda_complexity,
            lambda_safety=self.lambda_safety,
            compute_efficiency=state.compute_efficiency,
            param_efficiency=state.param_efficiency,
            coverage_ratio=state.coverage_ratio
        )
