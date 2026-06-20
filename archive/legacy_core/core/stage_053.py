"""阶段53：正向/负向反馈信号生成——从误差变化到驱动信号。

本阶段实现反馈信号的数学定义和计算接口，
将误差变化量映射为可路由的驱动信号。
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum
import numpy as np

from .stage_052 import ErrorAccumulation


class FeedbackType(Enum):
    """反馈信号类型。"""
    REWARD = "reward"
    PENALTY = "penalty"
    NEUTRAL = "neutral"


class RouteTarget(Enum):
    """反馈路由目标。"""
    DRIVE_ENGINE = "drive_engine"
    MODEL_UPDATER = "model_updater"
    TASK_GENERATOR = "task_generator"


@dataclass(frozen=True)
class FeedbackRoutes:
    """反馈路由结果。"""
    drive_signal: float
    model_signal: float
    task_signal: float
    drive_weight: float = 0.5
    model_weight: float = 0.3
    task_weight: float = 0.2

    def __post_init__(self):
        total = self.drive_weight + self.model_weight + self.task_weight
        if not np.allclose(total, 1.0):
            raise ValueError(f"路由权重之和必须为1，收到 {total}")
        for name, val in [("drive_signal", self.drive_signal),
                          ("model_signal", self.model_signal),
                          ("task_signal", self.task_signal)]:
            if val < 0:
                raise ValueError(f"{name} 必须 >= 0，收到 {val}")


@dataclass(frozen=True)
class FeedbackSignal:
    """正向/负向反馈信号的完整描述。

    组合阶段52的ErrorAccumulation，在误差累积的基础上增加
    反馈信号的生成和路由。

    Attributes:
        error_accumulation: 阶段52的误差累积状态（组合字段）
        delta_F: 自由能变化量 ΔF
        feedback_type: 反馈类型
        reward_signal: 正向反馈信号（A53.1）
        penalty_signal: 负向反馈信号（A53.2）
        feedback_intensity: 反馈强度（A53.3）
        kappa: 灵敏度参数 κ
        F_baseline: 自由能基线
        routes: 反馈路由结果（A53.4）
        per_layer_feedback: 逐层反馈强度

    Invariants:
        - reward_signal >= 0
        - penalty_signal >= 0
        - reward_signal 和 penalty_signal 不同时非零
        - feedback_intensity ∈ [0, 1)
        - kappa > 0
        - F_baseline > 0
    """
    error_accumulation: ErrorAccumulation
    delta_F: float
    feedback_type: FeedbackType
    reward_signal: float = 0.0
    penalty_signal: float = 0.0
    feedback_intensity: float = 0.0
    kappa: float = 2.0
    F_baseline: float = 1.0
    routes: Optional[FeedbackRoutes] = None
    per_layer_feedback: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.reward_signal < 0:
            raise ValueError(
                f"正向反馈信号必须 >= 0，收到 {self.reward_signal}"
            )
        if self.penalty_signal < 0:
            raise ValueError(
                f"负向反馈信号必须 >= 0，收到 {self.penalty_signal}"
            )
        if self.reward_signal > 0 and self.penalty_signal > 0:
            raise ValueError("正向和负向反馈信号不能同时非零")
        if not (0 <= self.feedback_intensity < 1):
            raise ValueError(
                f"反馈强度必须 ∈ [0,1)，收到 {self.feedback_intensity}"
            )
        if self.kappa <= 0:
            raise ValueError(
                f"灵敏度参数必须 > 0，收到 {self.kappa}"
            )
        if self.F_baseline <= 0:
            raise ValueError(
                f"自由能基线必须 > 0，收到 {self.F_baseline}"
            )

    @property
    def is_positive(self) -> bool:
        """是否为正向反馈。"""
        return self.feedback_type == FeedbackType.REWARD

    @property
    def is_negative(self) -> bool:
        """是否为负向反馈。"""
        return self.feedback_type == FeedbackType.PENALTY

    @property
    def raw_signal(self) -> float:
        """原始信号值（正或负）。"""
        return self.reward_signal - self.penalty_signal


class FeedbackGenerator:
    """阶段53的核心计算接口。

    实现从误差变化量到反馈信号的映射。
    核心约束：
        - 正向和负向反馈互斥
        - 反馈强度 ∈ [0, 1)
        - 路由权重归一化
    """

    def compute_delta_F(
        self,
        current_accumulation: ErrorAccumulation,
        prev_accumulation: ErrorAccumulation
    ) -> float:
        """计算自由能变化量 ΔF。

        ΔF ≈ Δε_global（稳态近似）

        Args:
            current_accumulation: 当前累积状态
            prev_accumulation: 上一累积状态

        Returns:
            自由能变化量 ΔF
        """
        current_F = current_accumulation.cumulative_error
        prev_F = prev_accumulation.cumulative_error
        return current_F - prev_F

    def generate_reward_signal(
        self,
        delta_F: float
    ) -> float:
        """生成正向反馈信号（A53.1）。

        reward_signal = |ΔF| 当 ΔF < 0

        Args:
            delta_F: 自由能变化量

        Returns:
            正向反馈信号 >= 0
        """
        if delta_F < 0:
            return -delta_F
        return 0.0

    def generate_penalty_signal(
        self,
        delta_F: float
    ) -> float:
        """生成负向反馈信号（A53.2）。

        penalty_signal = |ΔF| 当 ΔF > 0

        Args:
            delta_F: 自由能变化量

        Returns:
            负向反馈信号 >= 0
        """
        if delta_F > 0:
            return delta_F
        return 0.0

    def compute_feedback_intensity(
        self,
        delta_F: float,
        F_baseline: float,
        kappa: float
    ) -> float:
        """计算反馈强度（A53.3）。

        feedback = tanh(κ · |ΔF| / F_baseline)

        Args:
            delta_F: 自由能变化量
            F_baseline: 自由能基线
            kappa: 灵敏度参数

        Returns:
            反馈强度 ∈ [0, 1)
        """
        if kappa <= 0:
            raise ValueError(f"kappa 必须 > 0，收到 {kappa}")
        if F_baseline <= 0:
            raise ValueError(f"F_baseline 必须 > 0，收到 {F_baseline}")
        return float(np.tanh(kappa * abs(delta_F) / F_baseline))

    def route_feedback(
        self,
        feedback_intensity: float,
        feedback_type: FeedbackType,
        weights: tuple = (0.5, 0.3, 0.2)
    ) -> FeedbackRoutes:
        """路由反馈信号（A53.4）。

        Args:
            feedback_intensity: 反馈强度
            feedback_type: 反馈类型
            weights: 路由权重 (drive, model, task)

        Returns:
            FeedbackRoutes 实例
        """
        if not np.allclose(np.sum(weights), 1.0):
            raise ValueError(
                f"路由权重之和必须为1，收到 {np.sum(weights)}"
            )
        drive_w, model_w, task_w = weights
        return FeedbackRoutes(
            drive_signal=drive_w * feedback_intensity,
            model_signal=model_w * feedback_intensity,
            task_signal=task_w * feedback_intensity,
            drive_weight=drive_w,
            model_weight=model_w,
            task_weight=task_w,
        )

    def generate(
        self,
        accumulation: ErrorAccumulation,
        prev_accumulation: ErrorAccumulation,
        kappa: float = 2.0,
        F_baseline: float = 1.0
    ) -> FeedbackSignal:
        """完整生成反馈信号（A53.1-A53.4）。

        Args:
            accumulation: 当前累积状态
            prev_accumulation: 上一累积状态
            kappa: 灵敏度参数
            F_baseline: 自由能基线

        Returns:
            FeedbackSignal 实例
        """
        delta_F = self.compute_delta_F(accumulation, prev_accumulation)
        reward = self.generate_reward_signal(delta_F)
        penalty = self.generate_penalty_signal(delta_F)

        if delta_F < 0:
            feedback_type = FeedbackType.REWARD
        elif delta_F > 0:
            feedback_type = FeedbackType.PENALTY
        else:
            feedback_type = FeedbackType.NEUTRAL

        intensity = self.compute_feedback_intensity(
            delta_F, F_baseline, kappa
        )
        routes = self.route_feedback(intensity, feedback_type)

        # 逐层反馈强度
        current_eps = accumulation.standardized_error.epsilon_vector
        prev_eps = prev_accumulation.standardized_error.epsilon_vector
        delta_eps = np.abs(current_eps - prev_eps)
        baseline_eps = np.ones(3) * F_baseline / 3.0
        per_layer = np.tanh(kappa * delta_eps / baseline_eps)

        return FeedbackSignal(
            error_accumulation=accumulation,
            delta_F=delta_F,
            feedback_type=feedback_type,
            reward_signal=reward,
            penalty_signal=penalty,
            feedback_intensity=intensity,
            kappa=kappa,
            F_baseline=F_baseline,
            routes=routes,
            per_layer_feedback=per_layer,
        )
