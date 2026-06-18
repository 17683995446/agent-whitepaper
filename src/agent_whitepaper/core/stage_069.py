"""阶段69：稳态环境探索权重上浮——空闲时主动探索的增益机制。

本阶段实现稳态检测和探索权重上浮机制。
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .stage_068 import ExploreExploitEnforcer


@dataclass(frozen=True)
class StableExploreBooster:
    """稳态探索增强器的完整描述。

    组合阶段68的ExploreExploitEnforcer，在比例执行器的基础上增加
    稳态探索权重上浮的状态。

    Attributes:
        enforcer: 阶段68的探索-利用执行器（组合字段）
        is_stable: 系统是否处于稳态
        stable_duration: 稳态持续时间
        W_base: 基础探索权重（0.3）
        W_explore: 当前探索权重
        W_max: 探索权重上界（0.5）
        alpha: 上浮速率
        T_ref: 参考时间
        delta_W: 当前上浮量
        theta_stable_F: 自由能稳定性阈值
        theta_stable_W: 权重稳定性阈值
        beta_decay: 渐进回退衰减率
        resource_budget: 资源预算
        anomaly_detected: 是否检测到异常

    Invariants:
        - W_base > 0
        - W_max > W_base
        - W_explore ∈ [W_base, W_max]
        - alpha > 0
        - T_ref > 0
        - 0 < beta_decay < 1
    """
    enforcer: ExploreExploitEnforcer
    is_stable: bool = False
    stable_duration: int = 0
    W_base: float = 0.3
    W_explore: float = 0.3
    W_max: float = 0.5
    alpha: float = 0.05
    T_ref: int = 100
    delta_W: float = 0.0
    theta_stable_F: float = 0.01
    theta_stable_W: float = 0.001
    beta_decay: float = 0.9
    resource_budget: float = 1.0
    anomaly_detected: bool = False

    def __post_init__(self):
        if self.W_base <= 0:
            raise ValueError(
                f"基础探索权重必须 > 0，收到 {self.W_base}"
            )
        if self.W_max <= self.W_base:
            raise ValueError(
                f"探索权重上界必须 > 基础权重，"
                f"收到 max={self.W_max}, base={self.W_base}"
            )
        if not (self.W_base <= self.W_explore <= self.W_max):
            raise ValueError(
                f"当前探索权重必须 ∈ [W_base, W_max]，"
                f"收到 {self.W_explore}"
            )
        if self.alpha <= 0:
            raise ValueError(
                f"上浮速率必须 > 0，收到 {self.alpha}"
            )

    @property
    def exploit_weight(self) -> float:
        """利用权重。"""
        return 1.0 - self.W_explore

    @property
    def boost_active(self) -> bool:
        """上浮是否激活。"""
        return self.is_stable and self.delta_W > 0


class StableExploreBoosterInterface:
    """阶段69的核心接口：稳态探索增强器。

    实现稳态检测、探索权重上浮、上浮幅度计算和安全检查。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 探索权重不超过上界
        - 上浮后通过安全检查
        - 非稳态时渐进回退
    """

    def detect_stability(
        self,
        free_energy_history: np.ndarray,
        weight_norm_history: np.ndarray,
        window_F: int,
        window_W: int,
        theta_F: float,
        theta_W: float
    ) -> bool:
        """检测系统稳态（A69.1）。

        Args:
            free_energy_history: 自由能历史
            weight_norm_history: 权重范数历史
            window_F: 自由能窗口长度
            window_W: 权重窗口长度
            theta_F: 自由能稳定性阈值
            theta_W: 权重稳定性阈值

        Returns:
            是否处于稳态
        """
        if len(free_energy_history) < window_F or len(weight_norm_history) < window_W:
            return False
        var_F = float(np.var(free_energy_history[-window_F:]))
        var_W = float(np.var(weight_norm_history[-window_W:]))
        return var_F < theta_F and var_W < theta_W

    def compute_boost(
        self,
        stable_duration: int,
        alpha: float,
        T_ref: int,
        delta_W_max: float
    ) -> float:
        """计算探索权重上浮量（A69.3）。

        ΔW = min(α·ln(1 + T/T_ref), ΔW_max)

        Args:
            stable_duration: 稳态持续时间
            alpha: 上浮速率
            T_ref: 参考时间
            delta_W_max: 最大上浮量

        Returns:
            上浮量 ΔW ∈ [0, ΔW_max]
        """
        delta_W = alpha * np.log(1.0 + stable_duration / max(T_ref, 1))
        return float(min(delta_W, delta_W_max))

    def apply_boost(
        self,
        booster: StableExploreBooster
    ) -> StableExploreBooster:
        """应用探索权重上浮（A69.2）。

        Args:
            booster: 当前增强器状态

        Returns:
            更新后的增强器状态
        """
        if not booster.is_stable:
            delta_W = booster.delta_W * booster.beta_decay
        else:
            delta_W_max = booster.W_max - booster.W_base
            delta_W = self.compute_boost(
                booster.stable_duration,
                booster.alpha,
                booster.T_ref,
                delta_W_max,
            )
        W_explore = min(booster.W_base + delta_W, booster.W_max)
        return StableExploreBooster(
            enforcer=booster.enforcer,
            is_stable=booster.is_stable,
            stable_duration=booster.stable_duration,
            W_base=booster.W_base,
            W_explore=W_explore,
            W_max=booster.W_max,
            alpha=booster.alpha,
            T_ref=booster.T_ref,
            delta_W=delta_W,
            theta_stable_F=booster.theta_stable_F,
            theta_stable_W=booster.theta_stable_W,
            beta_decay=booster.beta_decay,
            resource_budget=booster.resource_budget,
            anomaly_detected=booster.anomaly_detected,
        )

    def safety_check(
        self,
        W_explore: float,
        W_max: float,
        resource_usage: float,
        resource_budget: float,
        anomaly_detected: bool
    ) -> Tuple[bool, float]:
        """执行安全检查（A69.4）。

        Args:
            W_explore: 当前探索权重
            W_max: 权重上界
            resource_usage: 资源使用量
            resource_budget: 资源预算
            anomaly_detected: 是否检测到异常

        Returns:
            (是否安全, 修正后的探索权重)
        """
        if anomaly_detected:
            return False, 0.0
        corrected_W = min(W_explore, W_max)
        if resource_usage > resource_budget:
            scale = resource_budget / max(resource_usage, 1e-12)
            corrected_W = corrected_W * scale
        safe = corrected_W <= W_max
        return safe, corrected_W

    def step(
        self,
        booster: StableExploreBooster,
        free_energy_history: np.ndarray,
        weight_norm_history: np.ndarray
    ) -> StableExploreBooster:
        """执行一步稳态探索增强（A69.1-A69.4）。

        Args:
            booster: 当前增强器状态
            free_energy_history: 自由能历史
            weight_norm_history: 权重范数历史

        Returns:
            更新后的增强器状态
        """
        is_stable = self.detect_stability(
            free_energy_history,
            weight_norm_history,
            window_F=min(10, len(free_energy_history)),
            window_W=min(10, len(weight_norm_history)),
            theta_F=booster.theta_stable_F,
            theta_W=booster.theta_stable_W,
        )
        stable_duration = booster.stable_duration + 1 if is_stable else 0

        new_booster = StableExploreBooster(
            enforcer=booster.enforcer,
            is_stable=is_stable,
            stable_duration=stable_duration,
            W_base=booster.W_base,
            W_explore=booster.W_explore,
            W_max=booster.W_max,
            alpha=booster.alpha,
            T_ref=booster.T_ref,
            delta_W=booster.delta_W,
            theta_stable_F=booster.theta_stable_F,
            theta_stable_W=booster.theta_stable_W,
            beta_decay=booster.beta_decay,
            resource_budget=booster.resource_budget,
            anomaly_detected=booster.anomaly_detected,
        )
        new_booster = self.apply_boost(new_booster)
        safe, corrected_W = self.safety_check(
            new_booster.W_explore,
            new_booster.W_max,
            resource_usage=new_booster.W_explore * 0.1,
            resource_budget=new_booster.resource_budget,
            anomaly_detected=new_booster.anomaly_detected,
        )
        if not safe:
            new_booster = StableExploreBooster(
                enforcer=new_booster.enforcer,
                is_stable=new_booster.is_stable,
                stable_duration=new_booster.stable_duration,
                W_base=new_booster.W_base,
                W_explore=corrected_W,
                W_max=new_booster.W_max,
                alpha=new_booster.alpha,
                T_ref=new_booster.T_ref,
                delta_W=corrected_W - new_booster.W_base,
                theta_stable_F=new_booster.theta_stable_F,
                theta_stable_W=new_booster.theta_stable_W,
                beta_decay=new_booster.beta_decay,
                resource_budget=new_booster.resource_budget,
                anomaly_detected=new_booster.anomaly_detected,
            )
        return new_booster
