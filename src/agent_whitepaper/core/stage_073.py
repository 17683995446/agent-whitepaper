"""阶段73：样本价值评估——误差惊奇度、信息增益与Top30%筛选。

本阶段实现样本价值评估机制。
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from .stage_072 import LongTermMemory


@dataclass(frozen=True)
class SampleValueEvaluator:
    """样本价值评估器的完整描述。

    组合阶段72的LongTermMemory，在长期记忆的基础上增加
    价值评估的状态。

    Attributes:
        ltm: 阶段72的长期记忆（组合字段）
        alpha: 误差惊奇度权重
        beta: 信息增益权重
        gamma: 奖励幅度权重
        error_mean: 滑动窗口误差均值
        error_var: 滑动窗口误差方差
        forget_factor: 遗忘因子λ
        top_k_ratio: Top筛选比例（默认0.3）
        last_update_step: 上次更新的步数
        update_interval: 更新间隔（步数）

    Invariants:
        - alpha + beta + gamma == 1.0
        - alpha, beta, gamma >= 0
        - error_var >= 0
        - 0 < forget_factor < 1
        - 0 < top_k_ratio <= 1
    """
    ltm: LongTermMemory
    alpha: float = 0.4
    beta: float = 0.4
    gamma: float = 0.2
    error_mean: float = 0.0
    error_var: float = 1.0
    forget_factor: float = 0.01
    top_k_ratio: float = 0.3
    last_update_step: int = 0
    update_interval: int = 100

    def __post_init__(self):
        if not np.isclose(self.alpha + self.beta + self.gamma, 1.0):
            raise ValueError(
                f"权重之和必须为1，收到 "
                f"α={self.alpha}+β={self.beta}+γ={self.gamma}="
                f"{self.alpha + self.beta + self.gamma}"
            )
        if self.alpha < 0 or self.beta < 0 or self.gamma < 0:
            raise ValueError("权重必须非负")
        if self.error_var < 0:
            raise ValueError(
                f"误差方差必须 >= 0，收到 {self.error_var}"
            )
        if not (0 < self.forget_factor < 1):
            raise ValueError(
                f"遗忘因子必须 ∈ (0,1)，收到 {self.forget_factor}"
            )
        if not (0 < self.top_k_ratio <= 1):
            raise ValueError(
                f"筛选比例必须 ∈ (0,1]，收到 {self.top_k_ratio}"
            )


class SampleValueEvaluatorInterface:
    """阶段73的核心接口：样本价值评估。

    实现价值评分计算、误差惊奇度、信息增益和Top-k筛选。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 价值评分非负
        - 权重之和为1
        - 信息增益非负
    """

    def compute_error_surprise(
        self,
        error: float,
        error_mean: float,
        error_var: float
    ) -> float:
        """计算误差惊奇度（A73.2）。

        error_surprise = -log p(error | N(μ, σ²))

        Args:
            error: 当前全局误差
            error_mean: 滑动窗口误差均值
            error_var: 滑动窗口误差方差

        Returns:
            误差惊奇度 >= 0

        Raises:
            ValueError: 当error_var <= 0时
        """
        if error_var <= 0:
            raise ValueError(
                f"误差方差必须 > 0，收到 {error_var}"
            )
        std = np.sqrt(error_var)
        prob = np.exp(-0.5 * ((error - error_mean) / std) ** 2) / (
            std * np.sqrt(2 * np.pi)
        )
        prob = max(prob, 1e-300)
        return float(-np.log(prob))

    def compute_info_gain(
        self,
        params_before: np.ndarray,
        params_after: np.ndarray,
        fisher_matrix: np.ndarray
    ) -> float:
        """计算信息增益（A73.3）。

        info_gain ≈ (1/2) Δθ^T F Δθ

        Args:
            params_before: 更新前参数，shape (n,)
            params_after: 更新后参数，shape (n,)
            fisher_matrix: Fisher信息矩阵，shape (n, n)

        Returns:
            信息增益 >= 0

        Raises:
            ValueError: 当参数维度不匹配时
        """
        if params_before.shape != params_after.shape:
            raise ValueError(
                f"参数维度不匹配: {params_before.shape} vs {params_after.shape}"
            )
        if fisher_matrix.shape != (
            params_before.shape[0], params_before.shape[0]
        ):
            raise ValueError(
                f"Fisher矩阵维度不匹配: {fisher_matrix.shape}"
            )
        delta = params_after - params_before
        info_gain = 0.5 * float(delta.T @ fisher_matrix @ delta)
        return max(info_gain, 0.0)

    def compute_value(
        self,
        error_surprise: float,
        info_gain: float,
        reward_magnitude: float,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2
    ) -> float:
        """计算价值评分（A73.1）。

        value = α·surprise + β·info_gain + γ·|reward|

        Args:
            error_surprise: 误差惊奇度
            info_gain: 信息增益
            reward_magnitude: 奖励幅度
            alpha: 惊奇度权重
            beta: 信息增益权重
            gamma: 奖励权重

        Returns:
            价值评分 >= 0

        Raises:
            ValueError: 当权重之和不等于1时
            ValueError: 当任何权重为负时
        """
        total = alpha + beta + gamma
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"权重之和必须等于1，收到 {total}"
            )
        if alpha < 0 or beta < 0 or gamma < 0:
            raise ValueError("权重必须非负")
        value = (
            alpha * error_surprise
            + beta * info_gain
            + gamma * abs(reward_magnitude)
        )
        return float(max(value, 0.0))

    def select_top_k(
        self,
        samples: List[Tuple],
        k_ratio: float = 0.3
    ) -> List[Tuple]:
        """Top-k高价值样本筛选（A73.4）。

        Args:
            samples: 样本列表（含value_score）
            k_ratio: 筛选比例，必须 ∈ (0, 1]

        Returns:
            价值最高的前k_ratio比例样本

        Raises:
            ValueError: 当k_ratio不在(0,1]范围内时
        """
        if not (0 < k_ratio <= 1):
            raise ValueError(
                f"k_ratio必须 ∈ (0,1]，收到 {k_ratio}"
            )
        if not samples:
            return []
        k = max(1, int(len(samples) * k_ratio))
        sorted_samples = sorted(
            samples, key=lambda x: x[1], reverse=True
        )
        return sorted_samples[:k]

    def update_online(
        self,
        evaluator: SampleValueEvaluator,
        new_error: float,
        step: int
    ) -> SampleValueEvaluator:
        """在线更新价值评估（A73.5）。

        更新误差分布统计量。

        Args:
            evaluator: 当前评估器状态
            new_error: 新的全局误差
            step: 当前步数

        Returns:
            更新后的评估器状态
        """
        ff = evaluator.forget_factor
        new_mean = ff * evaluator.error_mean + (1 - ff) * new_error
        new_var = ff * evaluator.error_var + (
            1 - ff
        ) * (new_error - new_mean) ** 2
        new_var = max(new_var, 1e-8)
        return SampleValueEvaluator(
            ltm=evaluator.ltm,
            error_mean=float(new_mean),
            error_var=float(new_var),
            alpha=evaluator.alpha,
            beta=evaluator.beta,
            gamma=evaluator.gamma,
            forget_factor=evaluator.forget_factor,
            top_k_ratio=evaluator.top_k_ratio,
            last_update_step=step,
            update_interval=evaluator.update_interval,
        )
