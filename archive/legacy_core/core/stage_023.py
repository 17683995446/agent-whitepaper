"""阶段23：探索驱力——从预测编码到探索需求。

本阶段实现探索驱力的数学定义和计算接口，
将预测编码验证结果映射为探索需求强度。
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .stage_020 import PCValidationResult


@dataclass(frozen=True)
class ExplorationDrive:
    """探索驱力的完整描述。

    组合阶段20的PCValidationResult，在验证结果的基础上增加
    探索需求强度、边界和紧急度。

    Attributes:
        validation_result: 阶段20的验证结果（组合字段）
        exploration_strength: 探索需求强度 s_explore ∈ [0, 1]
        exploration_boundary: 探索边界 b_explore ∈ [0, 1]
        exploration_urgency: 探索紧急度 u_explore ∈ [0, 1]

    Invariants:
        - 0 ≤ exploration_strength ≤ 1
        - 0 ≤ exploration_boundary ≤ 1
        - 0 ≤ exploration_urgency ≤ 1
        - exploration_urgency = max(0, (exploration_boundary - exploration_strength) / exploration_boundary)
          当 exploration_boundary > 0
    """
    validation_result: PCValidationResult
    exploration_strength: float
    exploration_boundary: float
    exploration_urgency: float

    def __post_init__(self):
        if not (0 <= self.exploration_strength <= 1):
            raise ValueError(
                f"探索需求强度必须 ∈ [0,1]，收到 {self.exploration_strength}"
            )
        if not (0 <= self.exploration_boundary <= 1):
            raise ValueError(
                f"探索边界必须 ∈ [0,1]，收到 {self.exploration_boundary}"
            )
        if not (0 <= self.exploration_urgency <= 1):
            raise ValueError(
                f"探索紧急度必须 ∈ [0,1]，收到 {self.exploration_urgency}"
            )
        if self.exploration_boundary > 1e-12:
            expected_urgency = max(
                0.0, (self.exploration_boundary - self.exploration_strength)
                / self.exploration_boundary
            )
            if abs(self.exploration_urgency - expected_urgency) > 1e-6:
                raise ValueError(
                    f"紧急度 {self.exploration_urgency} 与公式计算结果 "
                    f"{expected_urgency} 不一致"
                )

    @property
    def is_violated(self) -> bool:
        """探索边界是否被违反。"""
        return self.exploration_strength < self.exploration_boundary

    @property
    def violation_severity(self) -> float:
        """违反严重程度。"""
        return max(0.0, self.exploration_boundary - self.exploration_strength)


class ExplorationDriveCalculator:
    """阶段23的核心计算接口。

    实现从预测编码验证结果到探索驱力的映射。
    核心约束：
        - 探索需求强度 = 1 / (1 + mean_convergence_steps)（A23.1）
        - 探索边界 = 0.3（A23.2）
        - 紧急度 = max(0, (边界 - 强度) / 边界)（A23.3）
    """

    DEFAULT_EXPLORATION_BOUNDARY: float = 0.3

    def compute_exploration_strength(
        self,
        validation_result: PCValidationResult
    ) -> float:
        """计算探索需求强度（A23.1）。

        s_explore = 1 / (1 + mean_convergence_steps)。

        Args:
            validation_result: 阶段20的验证结果

        Returns:
            探索需求强度 ∈ [0, 1]

        Raises:
            ValueError: 若 validation_result 为 None
        """
        if validation_result is None:
            raise ValueError("validation_result 不能为空")
        mean_steps = validation_result.mean_convergence_steps
        strength = 1.0 / (1.0 + mean_steps)
        return float(np.clip(strength, 0.0, 1.0))

    def compute_exploration_boundary(
        self,
        validation_result: PCValidationResult,
        custom_boundary: float = DEFAULT_EXPLORATION_BOUNDARY
    ) -> float:
        """计算探索边界（A23.2）。

        Args:
            validation_result: 阶段20的验证结果
            custom_boundary: 自定义边界值

        Returns:
            探索边界 ∈ [0, 1]
        """
        if validation_result is None:
            raise ValueError("validation_result 不能为空")
        if not (0 <= custom_boundary <= 1):
            raise ValueError(
                f"自定义边界必须 ∈ [0,1]，收到 {custom_boundary}"
            )
        return float(custom_boundary)

    def compute_exploration_urgency(
        self,
        exploration_strength: float,
        exploration_boundary: float
    ) -> float:
        """计算探索紧急度（A23.3）。

        Args:
            exploration_strength: 探索需求强度
            exploration_boundary: 探索边界

        Returns:
            探索紧急度 ∈ [0, 1]

        Raises:
            ValueError: 若 exploration_boundary <= 0
        """
        if exploration_boundary <= 0:
            raise ValueError(
                f"探索边界必须 > 0，收到 {exploration_boundary}"
            )
        urgency = max(0.0, (exploration_boundary - exploration_strength)
                      / exploration_boundary)
        return float(np.clip(urgency, 0.0, 1.0))

    def from_validation_result(
        self,
        validation_result: PCValidationResult,
        custom_boundary: float = DEFAULT_EXPLORATION_BOUNDARY
    ) -> ExplorationDrive:
        """从验证结果构造完整的探索驱力描述。

        Args:
            validation_result: 阶段20的验证结果
            custom_boundary: 可选的自定义探索边界

        Returns:
            ExplorationDrive 实例
        """
        strength = self.compute_exploration_strength(validation_result)
        boundary = self.compute_exploration_boundary(
            validation_result, custom_boundary
        )
        urgency = self.compute_exploration_urgency(strength, boundary)
        return ExplorationDrive(
            validation_result=validation_result,
            exploration_strength=strength,
            exploration_boundary=boundary,
            exploration_urgency=urgency,
        )
