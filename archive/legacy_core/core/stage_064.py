"""阶段64：结构性重构触发条件——持续高误差、长期停滞与分布偏移的联合判定。

本阶段实现结构性重构触发条件的数学定义和计算接口。
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum
import numpy as np

from .stage_063 import MicroAdjusterState


class ReformTriggerType(Enum):
    """结构性重构触发类型。"""
    HIGH_ERROR = "high_error"
    STAGNATION = "stagnation"
    NEW_ENVIRONMENT = "new_environment"
    NONE = "none"


class RiskLevel(Enum):
    """重构风险等级。"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class StructuralReformTrigger:
    """结构性重构触发条件的完整描述。

    组合阶段63的MicroAdjusterState，在微修正状态的基础上增加
    结构重构的触发判定信息。

    Attributes:
        micro_adjuster_state: 阶段63的微修正器状态（组合字段）
        trigger_type: 触发类型
        cumulative_error: 累积误差
        error_duration: 误差持续时间
        theta_high: 高误差阈值
        T_min: 最小持续时间
        fitness_current: 当前适应度
        fitness_best: 历史最优适应度
        stagnation_duration: 停滞持续时间
        epsilon_fitness: 适应度提升阈值
        T_stagnation: 停滞判定时间
        distribution_shift: 分布偏移度量
        theta_shift: 分布偏移阈值
        risk_score: 重构风险值
        risk_level: 风险等级
        high_error_triggered: 持续高误差是否触发
        stagnation_triggered: 长期停滞是否触发
        new_env_triggered: 新环境是否触发

    Invariants:
        - theta_high > 0
        - T_min > 0
        - epsilon_fitness > 0
        - T_stagnation > 0
        - theta_shift > 0
        - risk_score ∈ [0, 1]
    """
    micro_adjuster_state: MicroAdjusterState
    trigger_type: ReformTriggerType = ReformTriggerType.NONE
    cumulative_error: float = 0.0
    error_duration: int = 0
    theta_high: float = 50.0
    T_min: int = 50
    fitness_current: float = 0.0
    fitness_best: float = 0.0
    stagnation_duration: int = 0
    epsilon_fitness: float = 0.01
    T_stagnation: int = 200
    distribution_shift: float = 0.0
    theta_shift: float = 2.0
    risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    high_error_triggered: bool = False
    stagnation_triggered: bool = False
    new_env_triggered: bool = False

    def __post_init__(self):
        if self.theta_high <= 0:
            raise ValueError(
                f"高误差阈值必须 > 0，收到 {self.theta_high}"
            )
        if self.T_min <= 0:
            raise ValueError(
                f"最小持续时间必须 > 0，收到 {self.T_min}"
            )
        if self.epsilon_fitness <= 0:
            raise ValueError(
                f"适应度提升阈值必须 > 0，收到 {self.epsilon_fitness}"
            )
        if self.T_stagnation <= 0:
            raise ValueError(
                f"停滞判定时间必须 > 0，收到 {self.T_stagnation}"
            )
        if self.theta_shift <= 0:
            raise ValueError(
                f"分布偏移阈值必须 > 0，收到 {self.theta_shift}"
            )
        if not (0 <= self.risk_score <= 1):
            raise ValueError(
                f"风险值必须 ∈ [0,1]，收到 {self.risk_score}"
            )

    @property
    def any_triggered(self) -> bool:
        """是否有任何重构触发条件满足。"""
        return (
            self.high_error_triggered
            or self.stagnation_triggered
            or self.new_env_triggered
        )

    @property
    def should_reform(self) -> bool:
        """是否应执行结构重构（触发+低风险）。"""
        return self.any_triggered and self.risk_level != RiskLevel.HIGH


class StructuralReformTriggerChecker:
    """阶段64的核心计算接口。

    判定结构性重构的触发条件——持续高误差、长期停滞、
    新环境和重构风险评估。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 重构触发与微修正触发互斥
        - 高风险重构需要额外确认
    """

    def check_high_error(
        self,
        cumulative_error: float,
        duration: int,
        theta_high: float,
        T_min: int
    ) -> bool:
        """检查持续高误差触发（A64.1）。

        Args:
            cumulative_error: 累积误差
            duration: 误差持续时间
            theta_high: 高误差阈值
            T_min: 最小持续时间

        Returns:
            是否触发
        """
        return cumulative_error > theta_high and duration > T_min

    def check_stagnation(
        self,
        fitness_current: float,
        fitness_best: float,
        duration: int,
        epsilon: float,
        T_stagnation: int
    ) -> bool:
        """检查长期停滞触发（A64.2）。

        Args:
            fitness_current: 当前适应度
            fitness_best: 历史最优适应度
            duration: 停滞持续时间
            epsilon: 适应度提升阈值
            T_stagnation: 停滞判定时间

        Returns:
            是否触发
        """
        return (fitness_best - fitness_current) < epsilon and duration > T_stagnation

    def check_new_environment(
        self,
        distribution_shift: float,
        theta_shift: float
    ) -> bool:
        """检查新环境触发（A64.3）。

        Args:
            distribution_shift: 分布偏移度量
            theta_shift: 分布偏移阈值

        Returns:
            是否触发
        """
        return distribution_shift > theta_shift

    def assess_risk(
        self,
        complexity: float,
        uncertainty: float,
        resource_cost: float,
        w_c: float = 0.4,
        w_u: float = 0.4,
        w_r: float = 0.2
    ) -> Tuple[float, RiskLevel]:
        """评估重构风险（A64.4）。

        Args:
            complexity: 重构复杂度 ∈ [0, 1]
            uncertainty: 重构不确定性 ∈ [0, 1]
            resource_cost: 资源成本 ∈ [0, 1]
            w_c, w_u, w_r: 权重

        Returns:
            (风险值, 风险等级)
        """
        risk = w_c * complexity + w_u * uncertainty + w_r * resource_cost
        risk = max(0.0, min(1.0, risk))
        if risk < 0.3:
            level = RiskLevel.LOW
        elif risk < 0.7:
            level = RiskLevel.MEDIUM
        else:
            level = RiskLevel.HIGH
        return risk, level

    def check_trigger(
        self,
        trigger: StructuralReformTrigger
    ) -> StructuralReformTrigger:
        """执行完整的重构触发判定（A64.1-A64.4）。

        Args:
            trigger: 当前触发状态

        Returns:
            更新后的StructuralReformTrigger
        """
        high_error = self.check_high_error(
            trigger.cumulative_error, trigger.error_duration,
            trigger.theta_high, trigger.T_min
        )
        stagnation = self.check_stagnation(
            trigger.fitness_current, trigger.fitness_best,
            trigger.stagnation_duration, trigger.epsilon_fitness,
            trigger.T_stagnation
        )
        new_env = self.check_new_environment(
            trigger.distribution_shift, trigger.theta_shift
        )

        trigger_type = ReformTriggerType.NONE
        if high_error:
            trigger_type = ReformTriggerType.HIGH_ERROR
        elif stagnation:
            trigger_type = ReformTriggerType.STAGNATION
        elif new_env:
            trigger_type = ReformTriggerType.NEW_ENVIRONMENT

        return StructuralReformTrigger(
            micro_adjuster_state=trigger.micro_adjuster_state,
            trigger_type=trigger_type,
            cumulative_error=trigger.cumulative_error,
            error_duration=trigger.error_duration,
            theta_high=trigger.theta_high,
            T_min=trigger.T_min,
            fitness_current=trigger.fitness_current,
            fitness_best=trigger.fitness_best,
            stagnation_duration=trigger.stagnation_duration,
            epsilon_fitness=trigger.epsilon_fitness,
            T_stagnation=trigger.T_stagnation,
            distribution_shift=trigger.distribution_shift,
            theta_shift=trigger.theta_shift,
            risk_score=trigger.risk_score,
            risk_level=trigger.risk_level,
            high_error_triggered=high_error,
            stagnation_triggered=stagnation,
            new_env_triggered=new_env,
        )
