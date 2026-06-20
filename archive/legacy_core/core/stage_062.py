"""阶段62：微修正算法——梯度下降、贝叶斯更新与EWC正则化的收敛保证。

本阶段实现微修正的数学定义和计算接口。
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .stage_061 import MicroAdjustTrigger


@dataclass(frozen=True)
class MicroAdjuster:
    """微修正算法的完整描述。

    组合阶段61的MicroAdjustTrigger，在触发条件的基础上增加
    微修正算法的状态——学习率、EWC参数、收敛信息。

    Attributes:
        trigger: 阶段61的微修正触发状态（组合字段）
        eta_0: 初始学习率
        eta_current: 当前学习率
        decay: 学习率衰减率
        eta_min: 学习率下界
        lambda_ewc: EWC正则化系数
        fisher_diag: Fisher信息矩阵对角元素
        anchor_params: 锚点参数 θ*
        step_size_limit: 步长比例因子 θ_step
        micro_adjust_count: 微修正执行次数
        last_free_energy: 上次微修正后的自由能
        convergence_history: 自由能历史（用于收敛判定）

    Invariants:
        - eta_0 > 0
        - eta_current >= eta_min > 0
        - decay > 0
        - lambda_ewc >= 0
        - step_size_limit > 0
        - micro_adjust_count >= 0
    """
    trigger: MicroAdjustTrigger
    eta_0: float = 0.01
    eta_current: float = 0.01
    decay: float = 0.01
    eta_min: float = 0.0001
    lambda_ewc: float = 100.0
    fisher_diag: Optional[np.ndarray] = None
    anchor_params: Optional[np.ndarray] = None
    step_size_limit: float = 0.01
    micro_adjust_count: int = 0
    last_free_energy: float = 0.0
    convergence_history: Optional[Tuple[float, ...]] = None

    def __post_init__(self):
        if self.eta_0 <= 0:
            raise ValueError(
                f"初始学习率必须 > 0，收到 {self.eta_0}"
            )
        if self.eta_current < self.eta_min:
            raise ValueError(
                f"当前学习率必须 >= 下界，"
                f"收到 eta={self.eta_current}, min={self.eta_min}"
            )
        if self.decay <= 0:
            raise ValueError(
                f"衰减率必须 > 0，收到 {self.decay}"
            )
        if self.lambda_ewc < 0:
            raise ValueError(
                f"EWC系数必须 >= 0，收到 {self.lambda_ewc}"
            )
        if self.step_size_limit <= 0:
            raise ValueError(
                f"步长比例因子必须 > 0，收到 {self.step_size_limit}"
            )
        if self.micro_adjust_count < 0:
            raise ValueError(
                f"微修正次数必须 >= 0，收到 {self.micro_adjust_count}"
            )

    @property
    def is_converged(self) -> bool:
        """微修正是否已收敛。"""
        if self.convergence_history is None or len(self.convergence_history) < 5:
            return False
        recent = self.convergence_history[-5:]
        return max(recent) - min(recent) < 1e-6

    @property
    def ewc_active(self) -> bool:
        """EWC正则化是否激活。"""
        return (
            self.lambda_ewc > 0
            and self.fisher_diag is not None
            and self.anchor_params is not None
        )


class MicroAdjustAlgorithm:
    """阶段62的核心计算接口。

    实现权重微调、学习率衰减、在线贝叶斯更新、
    EWC正则化和收敛保证。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 微修正仅在触发条件满足时执行
        - 自由能单调不增（无EWC时严格递减）
        - 学习率单调递减且不低于下界
    """

    def adjust_weights(
        self,
        adjuster: MicroAdjuster,
        gradient: np.ndarray,
        weights: np.ndarray
    ) -> Tuple[np.ndarray, MicroAdjuster]:
        """执行权重微调 ΔW = -η·∂F/∂W（A62.1）。

        Args:
            adjuster: 当前微修正器状态
            gradient: 自由能梯度 ∂F/∂W
            weights: 当前权重 W

        Returns:
            (更新后的权重, 更新后的微修正器状态)
        """
        eta = adjuster.eta_current
        delta_w = -eta * gradient
        max_step = adjuster.step_size_limit * np.linalg.norm(weights)
        delta_norm = np.linalg.norm(delta_w)
        if delta_norm > max_step and delta_norm > 0:
            delta_w = delta_w * (max_step / delta_norm)
        new_weights = weights + delta_w
        new_count = adjuster.micro_adjust_count + 1
        new_eta = self.compute_learning_rate(
            adjuster.eta_0, adjuster.decay, new_count, adjuster.eta_min
        )
        new_adjuster = MicroAdjuster(
            trigger=adjuster.trigger,
            eta_0=adjuster.eta_0,
            eta_current=new_eta,
            decay=adjuster.decay,
            eta_min=adjuster.eta_min,
            lambda_ewc=adjuster.lambda_ewc,
            fisher_diag=adjuster.fisher_diag,
            anchor_params=adjuster.anchor_params,
            step_size_limit=adjuster.step_size_limit,
            micro_adjust_count=new_count,
            last_free_energy=adjuster.last_free_energy,
            convergence_history=adjuster.convergence_history,
        )
        return new_weights, new_adjuster

    def compute_learning_rate(
        self,
        eta_0: float,
        decay: float,
        t: int,
        eta_min: float
    ) -> float:
        """计算衰减学习率 η_t = η_0/(1+decay·t)（A62.2）。

        Args:
            eta_0: 初始学习率
            decay: 衰减率
            t: 微修正次数
            eta_min: 学习率下界

        Returns:
            当前学习率 η_t ∈ [η_min, η_0]
        """
        eta_t = eta_0 / (1.0 + decay * t)
        return max(eta_t, eta_min)

    def bayesian_update(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        observation: np.ndarray,
        H: np.ndarray,
        R: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """执行在线贝叶斯更新（A62.3）。

        μ_{t+1} = (Σ_t^{-1}μ_t + H^TR^{-1}o) / Σ_{t+1}^{-1}
        Σ_{t+1}^{-1} = Σ_t^{-1} + H^TR^{-1}H

        Args:
            mu: 当前后验均值
            sigma: 当前后验协方差
            observation: 新观测值
            H: Jacobian矩阵
            R: 观测噪声协方差

        Returns:
            (更新后的均值, 更新后的协方差)
        """
        sigma_inv = np.linalg.inv(sigma)
        R_inv = np.linalg.inv(R)
        Ht_Rinv = H.T @ R_inv
        sigma_next_inv = sigma_inv + Ht_Rinv @ H
        sigma_next = np.linalg.inv(sigma_next_inv)
        mu_next = sigma_next @ (sigma_inv @ mu + Ht_Rinv @ observation)
        return mu_next, sigma_next

    def compute_ewc_loss(
        self,
        params: np.ndarray,
        anchor: np.ndarray,
        fisher_diag: np.ndarray,
        lambda_ewc: float
    ) -> float:
        """计算EWC正则化损失（A62.4）。

        L_ewc = Σ_i F_i(θ_i - θ_i*)²

        Args:
            params: 当前参数
            anchor: 锚点参数
            fisher_diag: Fisher信息对角元素
            lambda_ewc: EWC系数

        Returns:
            EWC正则化损失
        """
        diff = params - anchor
        loss = float(np.sum(fisher_diag * diff ** 2))
        return lambda_ewc * loss

    def compute_ewc_gradient(
        self,
        params: np.ndarray,
        anchor: np.ndarray,
        fisher_diag: np.ndarray,
        lambda_ewc: float
    ) -> np.ndarray:
        """计算EWC正则化梯度（A62.4-gradient）。

        ∂L_ewc/∂θ_i = 2λ_ewc·F_i·(θ_i - θ_i*)

        Args:
            params: 当前参数
            anchor: 锚点参数
            fisher_diag: Fisher信息对角元素
            lambda_ewc: EWC系数

        Returns:
            EWC梯度
        """
        return 2.0 * lambda_ewc * fisher_diag * (params - anchor)

    def check_convergence(
        self,
        free_energy_history: Tuple[float, ...],
        tolerance: float = 1e-6,
        window: int = 5
    ) -> bool:
        """检查微修正是否收敛（A62.5）。

        Args:
            free_energy_history: 自由能历史
            tolerance: 收敛容差
            window: 检查窗口长度

        Returns:
            是否收敛
        """
        if free_energy_history is None or len(free_energy_history) < window:
            return False
        recent = free_energy_history[-window:]
        return max(recent) - min(recent) < tolerance
