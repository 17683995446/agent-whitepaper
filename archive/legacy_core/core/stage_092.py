"""阶段92：全局微分方程组。

本阶段定义感知、行动、学习、进化、注意力五类动力学，
建立全局耦合方程组，封装为GlobalDynamics数据类组合GlobalObjective。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np

from .stage_091 import GlobalObjective, GlobalObjectiveCalculator


# ============================================================
# 阶段92 数据类：系统动力学状态
# ============================================================
@dataclass(frozen=True)
class SystemDynamicsState:
    """系统动力学的瞬时状态。

    Attributes:
        mu: 感知状态向量
        action: 行动向量
        weights: 权重矩阵（展平）
        theta: 进化参数
        precision: 精度向量
        mu_dot: 感知状态变化率
        action_dot: 行动变化率
        weights_dot: 权重变化率
        theta_dot: 进化参数变化率
        precision_dot: 精度变化率
    """
    mu: np.ndarray
    action: np.ndarray
    weights: np.ndarray
    theta: np.ndarray
    precision: np.ndarray
    mu_dot: np.ndarray
    action_dot: np.ndarray
    weights_dot: np.ndarray
    theta_dot: np.ndarray
    precision_dot: np.ndarray


# ============================================================
# 阶段92 数据类：全局动力学
# ============================================================
@dataclass(frozen=True)
class GlobalDynamics:
    """全局微分方程组的完整状态描述。

    组合阶段91的GlobalObjective，在全局目标基础上增加动力学状态。

    Attributes:
        global_objective: 阶段91的全局目标（组合字段）
        dynamics_state: 系统动力学状态
        tau_mu: 感知时间常数（各层）
        tau_a: 行动时间常数
        tau_W: 学习时间常数（各层）
        tau_theta: 进化时间常数
        tau_precision: 注意力时间常数
        kappa_gamma: 注意力增益
        lambda_W: 权重衰减系数
        sigma_theta: 变异强度
        dt_fast: 快速积分步长
        dt_mid: 中速积分步长
        dt_slow: 慢速积分步长
        dt_ultimate: 终极积分步长

    Invariants:
        - 所有时间常数 > 0
        - dt_fast < dt_mid < dt_slow < dt_ultimate
        - sigma_theta >= 0
    """
    global_objective: GlobalObjective
    dynamics_state: SystemDynamicsState
    tau_mu: Tuple[float, ...] = (0.01, 0.1, 1.0)
    tau_a: float = 1.0
    tau_W: Tuple[float, ...] = (60.0, 300.0, 1800.0)
    tau_theta: float = 3600.0
    tau_precision: float = 10.0
    kappa_gamma: float = 1.0
    lambda_W: float = 1e-4
    sigma_theta: float = 0.01
    dt_fast: float = 0.01
    dt_mid: float = 1.0
    dt_slow: float = 60.0
    dt_ultimate: float = 3600.0

    def __post_init__(self):
        if any(t <= 0 for t in self.tau_mu):
            raise ValueError("tau_mu所有值必须 > 0")
        if self.tau_a <= 0:
            raise ValueError(f"tau_a必须 > 0，收到 {self.tau_a}")
        if any(t <= 0 for t in self.tau_W):
            raise ValueError("tau_W所有值必须 > 0")
        if self.tau_theta <= 0:
            raise ValueError(f"tau_theta必须 > 0，收到 {self.tau_theta}")
        if self.tau_precision <= 0:
            raise ValueError(f"tau_precision必须 > 0，收到 {self.tau_precision}")
        if not (self.dt_fast < self.dt_mid < self.dt_slow < self.dt_ultimate):
            raise ValueError(
                f"dt必须满足: fast({self.dt_fast}) < mid({self.dt_mid}) < "
                f"slow({self.dt_slow}) < ultimate({self.dt_ultimate})"
            )
        if self.sigma_theta < 0:
            raise ValueError(f"sigma_theta必须 >= 0，收到 {self.sigma_theta}")


# ============================================================
# 阶段92 核心接口：全局动力学计算器
# ============================================================
class GlobalDynamicsCalculator:
    """阶段92的核心接口：全局微分方程组计算器。

    实现五类动力学的计算和耦合积分。

    核心约束：
        - 时间尺度分离：感知>>行动>>注意力>>学习>>进化
        - 多速率积分：快速动力学细粒度，慢速动力学粗粒度
        - 安全约束：行动和进化受安全护栏约束
    """

    def __init__(
        self,
        tau_mu: Tuple[float, ...] = (0.01, 0.1, 1.0),
        tau_a: float = 1.0,
        tau_W: Tuple[float, ...] = (60.0, 300.0, 1800.0),
        tau_theta: float = 3600.0,
        tau_precision: float = 10.0,
        kappa_gamma: float = 1.0,
        lambda_W: float = 1e-4,
        sigma_theta: float = 0.01
    ):
        """初始化全局动力学计算器。

        Args:
            tau_mu: 感知时间常数（各层），所有值必须 > 0
            tau_a: 行动时间常数，必须 > 0
            tau_W: 学习时间常数（各层），所有值必须 > 0
            tau_theta: 进化时间常数，必须 > 0
            tau_precision: 注意力时间常数，必须 > 0
            kappa_gamma: 注意力增益，必须 >= 0
            lambda_W: 权重衰减系数，必须 >= 0
            sigma_theta: 变异强度，必须 >= 0

        Raises:
            ValueError: 当任何时间常数 <= 0时
            ValueError: 当kappa_gamma < 0时

        Examples:
            >>> calc = GlobalDynamicsCalculator()
            >>> calc.tau_a
            1.0
        """
        if any(t <= 0 for t in tau_mu):
            raise ValueError("tau_mu所有值必须 > 0")
        if tau_a <= 0:
            raise ValueError(f"tau_a必须 > 0，收到 {tau_a}")
        if any(t <= 0 for t in tau_W):
            raise ValueError("tau_W所有值必须 > 0")
        if tau_theta <= 0:
            raise ValueError(f"tau_theta必须 > 0，收到 {tau_theta}")
        if tau_precision <= 0:
            raise ValueError(f"tau_precision必须 > 0，收到 {tau_precision}")
        if kappa_gamma < 0:
            raise ValueError(f"kappa_gamma必须 >= 0，收到 {kappa_gamma}")
        if lambda_W < 0:
            raise ValueError(f"lambda_W必须 >= 0，收到 {lambda_W}")
        if sigma_theta < 0:
            raise ValueError(f"sigma_theta必须 >= 0，收到 {sigma_theta}")
        self.tau_mu = tau_mu
        self.tau_a = tau_a
        self.tau_W = tau_W
        self.tau_theta = tau_theta
        self.tau_precision = tau_precision
        self.kappa_gamma = kappa_gamma
        self.lambda_W = lambda_W
        self.sigma_theta = sigma_theta

    def perception_dynamics(
        self,
        mu: np.ndarray,
        F: float,
        tau_mu: np.ndarray
    ) -> np.ndarray:
        """感知动力学（A92.1）。

        τ_μ · μ̇ = -∂F/∂μ ≈ -F · μ / |μ|²

        Args:
            mu: 当前感知状态，shape (n,)
            F: 变分自由能
            tau_mu: 感知时间常数，shape (n,)

        Returns:
            感知状态变化率 μ̇，shape (n,)

        Raises:
            ValueError: 当mu和tau_mu形状不匹配时

        Examples:
            >>> calc = GlobalDynamicsCalculator()
            >>> mu_dot = calc.perception_dynamics(
            ...     np.array([1.0, 0.5]), 1.0, np.array([0.01, 0.1]))
            >>> mu_dot.shape
            (2,)
        """
        if mu.shape != tau_mu.shape:
            raise ValueError("mu和tau_mu形状不匹配")
        mu_norm_sq = float(np.sum(mu ** 2))
        if mu_norm_sq < 1e-12:
            return np.zeros_like(mu)
        return -F * mu / (tau_mu * mu_norm_sq)

    def action_dynamics(
        self,
        action: np.ndarray,
        F: float,
        obs_model_jacobian: np.ndarray,
        tau_a: float
    ) -> np.ndarray:
        """行动动力学（A92.2）。

        τ_a · ȧ = -∂F/∂a = -g'(a)^T · ε_o

        简化实现：ȧ = -(J^T · J · a) / τ_a

        Args:
            action: 当前行动，shape (m,)
            F: 变分自由能
            obs_model_jacobian: 观测模型雅可比矩阵 ∂g/∂a，shape (n, m)
            tau_a: 行动时间常数，必须 > 0

        Returns:
            行动变化率 ȧ，shape (m,)

        Raises:
            ValueError: 当tau_a <= 0时

        Examples:
            >>> calc = GlobalDynamicsCalculator()
            >>> a_dot = calc.action_dynamics(
            ...     np.array([0.5]), 1.0,
            ...     np.array([[1.0]]), 1.0)
            >>> a_dot.shape
            (1,)
        """
        if tau_a <= 0:
            raise ValueError(f"tau_a必须 > 0，收到 {tau_a}")
        grad = obs_model_jacobian.T @ (obs_model_jacobian @ action)
        return -grad / tau_a

    def learning_dynamics(
        self,
        weights: np.ndarray,
        F: float,
        tau_W: np.ndarray,
        lambda_W: float = 1e-4,
        clip_norm: float = 1.0
    ) -> np.ndarray:
        """学习动力学（A92.3）。

        τ_W · Ẇ = -∂F/∂W - λ_W · W

        简化实现：Ẇ = -F·W/(τ_W·max(1,‖W‖)) - λ_W·W

        Args:
            weights: 当前权重，shape (n,)
            F: 变分自由能
            tau_W: 学习时间常数，shape (n,)
            lambda_W: 权重衰减系数，必须 >= 0
            clip_norm: 梯度裁剪范数，必须 > 0

        Returns:
            权重变化率 Ẇ，shape (n,)

        Raises:
            ValueError: 当weights和tau_W形状不匹配时
            ValueError: 当clip_norm <= 0时

        Examples:
            >>> calc = GlobalDynamicsCalculator()
            >>> w_dot = calc.learning_dynamics(
            ...     np.array([1.0, 0.5]), 1.0,
            ...     np.array([60.0, 300.0]), 1e-4, 1.0)
            >>> w_dot.shape
            (2,)
        """
        if weights.shape != tau_W.shape:
            raise ValueError("weights和tau_W形状不匹配")
        if clip_norm <= 0:
            raise ValueError(f"clip_norm必须 > 0，收到 {clip_norm}")
        w_norm = float(np.linalg.norm(weights))
        scale = max(1.0, w_norm)
        grad = F * weights / (tau_W * scale)
        w_dot = -grad - lambda_W * weights
        norm = float(np.linalg.norm(w_dot))
        if norm > clip_norm:
            w_dot = w_dot * (clip_norm / norm)
        return w_dot

    def evolution_dynamics(
        self,
        theta: np.ndarray,
        J: float,
        tau_theta: float = 3600.0,
        sigma_theta: float = 0.01
    ) -> np.ndarray:
        """进化动力学（A92.4）。

        τ_θ · θ̇ = -∂J/∂θ + σ_θ · ξ

        简化实现：θ̇ = (-J·θ/τ_θ + σ·ξ) / max(1, ‖θ‖)

        Args:
            theta: 当前进化参数，shape (n,)
            J: 全局目标值
            tau_theta: 进化时间常数，必须 > 0
            sigma_theta: 变异强度，必须 >= 0

        Returns:
            参数变化率 θ̇，shape (n,)

        Raises:
            ValueError: 当tau_theta <= 0时

        Examples:
            >>> calc = GlobalDynamicsCalculator()
            >>> theta_dot = calc.evolution_dynamics(
            ...     np.array([0.1, 0.2]), 10.0, 3600.0, 0.01)
            >>> theta_dot.shape
            (2,)
        """
        if tau_theta <= 0:
            raise ValueError(f"tau_theta必须 > 0，收到 {tau_theta}")
        noise = sigma_theta * np.random.randn(*theta.shape)
        theta_norm = float(np.linalg.norm(theta))
        scale = max(1.0, theta_norm)
        return (-J * theta / tau_theta + noise) / scale

    def attention_dynamics(
        self,
        precision: np.ndarray,
        precision_optimal: np.ndarray,
        tau_precision: float = 10.0,
        kappa_gamma: float = 1.0
    ) -> np.ndarray:
        """注意力动力学（A92.5）。

        τ_Π · Π̇ = κ_γ · (Π_opt - Π)

        Args:
            precision: 当前精度，shape (n,)
            precision_optimal: 最优精度分配，shape (n,)
            tau_precision: 注意力时间常数，必须 > 0
            kappa_gamma: 注意力增益，必须 >= 0

        Returns:
            精度变化率 Π̇，shape (n,)

        Raises:
            ValueError: 当precision和precision_optimal形状不匹配时
            ValueError: 当tau_precision <= 0时

        Examples:
            >>> calc = GlobalDynamicsCalculator()
            >>> pi_dot = calc.attention_dynamics(
            ...     np.array([1.0, 2.0]), np.array([3.0, 3.0]), 10.0, 1.0)
            >>> pi_dot.shape
            (2,)
        """
        if precision.shape != precision_optimal.shape:
            raise ValueError("precision和precision_optimal形状不匹配")
        if tau_precision <= 0:
            raise ValueError(f"tau_precision必须 > 0，收到 {tau_precision}")
        return kappa_gamma * (precision_optimal - precision) / tau_precision

    def compute_optimal_precision(
        self,
        mu: np.ndarray,
        F: float,
        precision_total: float = 10.0
    ) -> np.ndarray:
        """计算最优精度分配（A92.5-optimal）。

        Π_opt,l = ‖∂F/∂μ_l‖ / Σ‖∂F/∂μ_l'‖ · Π_total

        简化实现：Π_opt ∝ μ²

        Args:
            mu: 当前感知状态，shape (n,)
            F: 变分自由能
            precision_total: 总精度预算，必须 > 0

        Returns:
            最优精度分配向量，shape (n,)，和 = precision_total

        Raises:
            ValueError: 当precision_total <= 0时

        Examples:
            >>> calc = GlobalDynamicsCalculator()
            >>> opt = calc.compute_optimal_precision(
            ...     np.array([1.0, 2.0]), 1.0, 10.0)
            >>> np.isclose(np.sum(opt), 10.0)
            True
        """
        if precision_total <= 0:
            raise ValueError(f"precision_total必须 > 0，收到 {precision_total}")
        mu_sq = mu ** 2
        total = np.sum(mu_sq)
        if total < 1e-12:
            return np.full_like(mu, precision_total / len(mu))
        return precision_total * mu_sq / total

    def coupled_step(
        self,
        state: GlobalDynamics,
        dt: float
    ) -> GlobalDynamics:
        """全局耦合方程组的一步积分（A92.6）。

        采用多速率积分策略：快速动力学细粒度积分，
        慢速动力学粗粒度积分。

        Args:
            state: 当前全局动力学状态
            dt: 外部时间步长，必须 > 0

        Returns:
            更新后的全局动力学状态

        Raises:
            ValueError: 当dt <= 0时

        Examples:
            >>> calc = GlobalDynamicsCalculator()
            >>> from stage_091 import GlobalObjectiveCalculator
            >>> goc = GlobalObjectiveCalculator()
            >>> sg = SafetyGuardExecutor().check_all()
            >>> obj = goc.compute(1.0, 2.0, 0.5, sg, 1.0)
            >>> dyn_state = SystemDynamicsState(
            ...     mu=np.array([1.0]), action=np.array([0.5]),
            ...     weights=np.array([0.1]), theta=np.array([0.01]),
            ...     precision=np.array([1.0]),
            ...     mu_dot=np.zeros(1), action_dot=np.zeros(1),
            ...     weights_dot=np.zeros(1), theta_dot=np.zeros(1),
            ...     precision_dot=np.zeros(1))
            >>> gd = GlobalDynamics(global_objective=obj, dynamics_state=dyn_state)
            >>> new_gd = calc.coupled_step(gd, 0.01)
            >>> isinstance(new_gd, GlobalDynamics)
            True
        """
        if dt <= 0:
            raise ValueError(f"dt必须 > 0，收到 {dt}")
        ds = state.dynamics_state
        n_substeps = max(1, int(dt / state.dt_fast))
        dt_sub = dt / n_substeps
        mu = ds.mu.copy()
        action = ds.action.copy()
        weights = ds.weights.copy()
        theta = ds.theta.copy()
        precision = ds.precision.copy()

        # 多速率积分：快速动力学每子步积分
        for _ in range(n_substeps):
            # 感知动力学
            tau_mu_arr = np.full_like(mu, self.tau_mu[0])
            mu_dot = self.perception_dynamics(mu, 1.0, tau_mu_arr)

            # 行动动力学
            action_dot = self.action_dynamics(
                action, 1.0, np.eye(len(action)), self.tau_a
            )

            # 学习动力学
            tau_W_arr = np.full_like(weights, self.tau_W[0])
            weights_dot = self.learning_dynamics(
                weights, 1.0, tau_W_arr, self.lambda_W
            )

            # 进化动力学
            theta_dot = self.evolution_dynamics(
                theta, state.global_objective.total, self.tau_theta
            )

            # 注意力动力学
            precision_opt = self.compute_optimal_precision(mu, 1.0)
            precision_dot = self.attention_dynamics(
                precision, precision_opt, self.tau_precision
            )

            # 欧拉积分
            mu += mu_dot * dt_sub
            action += action_dot * dt_sub
            weights += weights_dot * dt_sub
            theta += theta_dot * dt_sub
            precision += precision_dot * dt_sub

            # 精度边界约束
            precision = np.clip(precision, 0.1, 10.0)

        new_ds = SystemDynamicsState(
            mu=mu, action=action, weights=weights, theta=theta, precision=precision,
            mu_dot=mu_dot, action_dot=action_dot, weights_dot=weights_dot,
            theta_dot=theta_dot, precision_dot=precision_dot
        )
        return GlobalDynamics(
            global_objective=state.global_objective,
            dynamics_state=new_ds,
            tau_mu=state.tau_mu, tau_a=state.tau_a, tau_W=state.tau_W,
            tau_theta=state.tau_theta, tau_precision=state.tau_precision,
            kappa_gamma=state.kappa_gamma, lambda_W=state.lambda_W,
            sigma_theta=state.sigma_theta,
            dt_fast=state.dt_fast, dt_mid=state.dt_mid,
            dt_slow=state.dt_slow, dt_ultimate=state.dt_ultimate
        )
