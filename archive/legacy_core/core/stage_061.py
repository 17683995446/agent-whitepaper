"""阶段61：微修正触发条件——梯度范数与自由能稳定性的联合判定。

本阶段实现微修正触发条件的数学定义和计算接口。
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum
import numpy as np

from .stage_060 import ErrorValidationResult


class TriggerStatus(Enum):
    """微修正触发状态。"""
    NOT_TRIGGERED = "not_triggered"
    TRIGGERED = "triggered"
    COOLDOWN = "cooldown"
    UNSTABLE_ENV = "unstable_env"
    HIGH_GRADIENT = "high_gradient"


@dataclass(frozen=True)
class MicroAdjustTrigger:
    """微修正触发条件的完整描述。

    组合阶段60的ErrorValidationResult，在验证结果的基础上增加
    微修正触发的判定信息——梯度范数、自由能稳定性、环境稳定性、
    冷却期状态。

    Attributes:
        validation_result: 阶段60的验证结果（组合字段）
        gradient_norm: 自由能梯度范数 ‖∂F/∂μ‖
        free_energy_delta: 自由能变化量 ΔF
        env_variance: 环境方差 var(ε_t, W)
        theta_micro: 微修正梯度阈值
        theta_stable: 自由能稳定性阈值
        theta_var: 环境方差阈值
        cooldown_steps: 冷却期步数 T_cooldown
        steps_since_last: 距上次微修正的步数
        trigger_status: 当前触发状态
        gradient_condition_met: 梯度条件是否满足
        stability_condition_met: 稳定性条件是否满足
        env_stable: 环境是否稳定
        cooldown_elapsed: 冷却期是否已过

    Invariants:
        - theta_micro > 0
        - theta_stable > 0
        - theta_var > 0
        - cooldown_steps > 0
        - steps_since_last >= 0
    """
    validation_result: ErrorValidationResult
    gradient_norm: float = 0.0
    free_energy_delta: float = 0.0
    env_variance: float = 0.0
    theta_micro: float = 0.1
    theta_stable: float = 0.01
    theta_var: float = 0.5
    cooldown_steps: int = 10
    steps_since_last: int = 100
    trigger_status: TriggerStatus = TriggerStatus.NOT_TRIGGERED
    gradient_condition_met: bool = False
    stability_condition_met: bool = False
    env_stable: bool = True
    cooldown_elapsed: bool = True

    def __post_init__(self):
        if self.theta_micro <= 0:
            raise ValueError(
                f"微修正梯度阈值必须 > 0，收到 {self.theta_micro}"
            )
        if self.theta_stable <= 0:
            raise ValueError(
                f"自由能稳定性阈值必须 > 0，收到 {self.theta_stable}"
            )
        if self.theta_var <= 0:
            raise ValueError(
                f"环境方差阈值必须 > 0，收到 {self.theta_var}"
            )
        if self.cooldown_steps <= 0:
            raise ValueError(
                f"冷却期步数必须 > 0，收到 {self.cooldown_steps}"
            )
        if self.steps_since_last < 0:
            raise ValueError(
                f"距上次微修正步数必须 >= 0，收到 {self.steps_since_last}"
            )

    @property
    def should_trigger(self) -> bool:
        """是否应触发微修正（A61.3完整条件）。"""
        return (
            self.gradient_condition_met
            and self.stability_condition_met
            and self.env_stable
            and self.cooldown_elapsed
        )

    @property
    def trigger_reason(self) -> str:
        """触发或未触发的原因。"""
        if self.should_trigger:
            return "所有条件满足，触发微修正"
        if not self.env_stable:
            return "环境不稳定，暂停微修正"
        if not self.gradient_condition_met:
            return "梯度范数过大，应考虑增强更新或结构重构"
        if not self.stability_condition_met:
            return "自由能未稳定，继续正常更新"
        if not self.cooldown_elapsed:
            return "冷却期未结束，等待"
        return "未知原因"


class MicroAdjustTriggerChecker:
    """阶段61的核心计算接口。

    判定微修正触发条件——梯度范数、自由能稳定性、
    环境稳定性和冷却期四个条件的联合判定。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 微修正仅在梯度小、自由能稳定、环境稳定、冷却期过后触发
        - 触发条件与结构重构触发条件互斥
    """

    def check_gradient_condition(
        self,
        gradient: np.ndarray,
        theta_micro: float
    ) -> bool:
        """检查梯度条件 ‖∂F/∂μ‖ < θ_micro（A61.1部分1）。

        Args:
            gradient: 自由能梯度 ∂F/∂μ
            theta_micro: 微修正梯度阈值

        Returns:
            梯度条件是否满足
        """
        return float(np.linalg.norm(gradient)) < theta_micro

    def check_stability_condition(
        self,
        free_energy_delta: float,
        theta_stable: float
    ) -> bool:
        """检查稳定性条件 ΔF < θ_stable（A61.1部分2）。

        Args:
            free_energy_delta: 自由能变化量 ΔF
            theta_stable: 自由能稳定性阈值

        Returns:
            稳定性条件是否满足
        """
        return free_energy_delta < theta_stable

    def check_env_stability(
        self,
        error_history: np.ndarray,
        window: int,
        theta_var: float
    ) -> bool:
        """检查环境稳定性 var(ε_t, W) < θ_var（A61.2）。

        Args:
            error_history: 误差历史序列
            window: 滑动窗口长度
            theta_var: 环境方差阈值

        Returns:
            环境是否稳定

        Raises:
            ValueError: 当窗口长度 < 10 时
        """
        if window < 10:
            raise ValueError(f"窗口长度必须 >= 10，收到 {window}")
        if len(error_history) < window:
            window = len(error_history)
        if window < 2:
            return True
        window_errors = error_history[-window:]
        mean = float(np.mean(window_errors))
        variance = float(np.sum((window_errors - mean) ** 2) / (window - 1))
        return variance < theta_var

    def check_cooldown(
        self,
        steps_since_last: int,
        cooldown_steps: int
    ) -> bool:
        """检查冷却期 t - t_last ≥ T_cooldown（A61.3部分3）。

        Args:
            steps_since_last: 距上次微修正的步数
            cooldown_steps: 冷却期步数

        Returns:
            冷却期是否已过
        """
        return steps_since_last >= cooldown_steps

    def check_trigger(
        self,
        trigger: MicroAdjustTrigger
    ) -> MicroAdjustTrigger:
        """执行完整的微修正触发判定（A61.1-A61.3）。

        依次检查梯度条件、稳定性条件、环境稳定性和冷却期，
        返回更新后的MicroAdjustTrigger。

        Args:
            trigger: 当前触发状态

        Returns:
            更新后的MicroAdjustTrigger
        """
        gradient_met = self.check_gradient_condition(
            np.array([trigger.gradient_norm]), trigger.theta_micro
        )
        stability_met = self.check_stability_condition(
            trigger.free_energy_delta, trigger.theta_stable
        )
        env_stable = self.check_env_stability(
            np.array([trigger.env_variance]), 10, trigger.theta_var
        )
        cooldown_elapsed = self.check_cooldown(
            trigger.steps_since_last, trigger.cooldown_steps
        )

        if gradient_met and stability_met and env_stable and cooldown_elapsed:
            status = TriggerStatus.TRIGGERED
        elif not env_stable:
            status = TriggerStatus.UNSTABLE_ENV
        elif not gradient_met:
            status = TriggerStatus.HIGH_GRADIENT
        elif not cooldown_elapsed:
            status = TriggerStatus.COOLDOWN
        else:
            status = TriggerStatus.NOT_TRIGGERED

        return MicroAdjustTrigger(
            validation_result=trigger.validation_result,
            gradient_norm=trigger.gradient_norm,
            free_energy_delta=trigger.free_energy_delta,
            env_variance=trigger.env_variance,
            theta_micro=trigger.theta_micro,
            theta_stable=trigger.theta_stable,
            theta_var=trigger.theta_var,
            cooldown_steps=trigger.cooldown_steps,
            steps_since_last=trigger.steps_since_last,
            trigger_status=status,
            gradient_condition_met=gradient_met,
            stability_condition_met=stability_met,
            env_stable=env_stable,
            cooldown_elapsed=cooldown_elapsed,
        )
