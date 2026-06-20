"""阶段22：社交驱力——从预测编码到社交需求。

本阶段实现社交驱力的数学定义和计算接口，
将预测编码验证结果映射为社交需求强度。
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .stage_020 import PCValidationResult


@dataclass(frozen=True)
class SocialDrive:
    """社交驱力的完整描述。

    组合阶段20的PCValidationResult，在验证结果的基础上增加
    社交需求强度、边界和紧急度。

    Attributes:
        validation_result: 阶段20的验证结果（组合字段）
        social_strength: 社交需求强度 s_social ∈ [0, 1]
        social_boundary: 社交边界 b_social ∈ [0, 1]
        social_urgency: 社交紧急度 u_social ∈ [0, 1]

    Invariants:
        - 0 ≤ social_strength ≤ 1
        - 0 ≤ social_boundary ≤ 1
        - 0 ≤ social_urgency ≤ 1
        - social_urgency = max(0, (social_boundary - social_strength) / social_boundary)
          当 social_boundary > 0
    """
    validation_result: PCValidationResult
    social_strength: float
    social_boundary: float
    social_urgency: float

    def __post_init__(self):
        if not (0 <= self.social_strength <= 1):
            raise ValueError(
                f"社交需求强度必须 ∈ [0,1]，收到 {self.social_strength}"
            )
        if not (0 <= self.social_boundary <= 1):
            raise ValueError(
                f"社交边界必须 ∈ [0,1]，收到 {self.social_boundary}"
            )
        if not (0 <= self.social_urgency <= 1):
            raise ValueError(
                f"社交紧急度必须 ∈ [0,1]，收到 {self.social_urgency}"
            )
        if self.social_boundary > 1e-12:
            expected_urgency = max(
                0.0, (self.social_boundary - self.social_strength)
                / self.social_boundary
            )
            if abs(self.social_urgency - expected_urgency) > 1e-6:
                raise ValueError(
                    f"紧急度 {self.social_urgency} 与公式计算结果 "
                    f"{expected_urgency} 不一致"
                )

    @property
    def is_violated(self) -> bool:
        """社交边界是否被违反。"""
        return self.social_strength < self.social_boundary

    @property
    def violation_severity(self) -> float:
        """违反严重程度。"""
        return max(0.0, self.social_boundary - self.social_strength)


class SocialDriveCalculator:
    """阶段22的核心计算接口。

    实现从预测编码验证结果到社交驱力的映射。
    核心约束：
        - 社交需求强度 = 稀疏度均值（A22.1）
        - 社交边界 = 0.5（A22.2）
        - 紧急度 = max(0, (边界 - 强度) / 边界)（A22.3）
    """

    DEFAULT_SOCIAL_BOUNDARY: float = 0.5

    def compute_social_strength(
        self,
        validation_result: PCValidationResult
    ) -> float:
        """计算社交需求强度（A22.1）。

        s_social = mean(sparsities)，归一化到 [0, 1]。

        Args:
            validation_result: 阶段20的验证结果

        Returns:
            社交需求强度 ∈ [0, 1]

        Raises:
            ValueError: 若 validation_result 为 None
        """
        if validation_result is None:
            raise ValueError("validation_result 不能为空")
        if len(validation_result.sparsities) == 0:
            return 0.0
        strength = float(np.mean(validation_result.sparsities))
        return float(np.clip(strength, 0.0, 1.0))

    def compute_social_boundary(
        self,
        validation_result: PCValidationResult,
        custom_boundary: float = DEFAULT_SOCIAL_BOUNDARY
    ) -> float:
        """计算社交边界（A22.2）。

        Args:
            validation_result: 阶段20的验证结果
            custom_boundary: 自定义边界值

        Returns:
            社交边界 ∈ [0, 1]
        """
        if validation_result is None:
            raise ValueError("validation_result 不能为空")
        if not (0 <= custom_boundary <= 1):
            raise ValueError(
                f"自定义边界必须 ∈ [0,1]，收到 {custom_boundary}"
            )
        return float(custom_boundary)

    def compute_social_urgency(
        self,
        social_strength: float,
        social_boundary: float
    ) -> float:
        """计算社交紧急度（A22.3）。

        Args:
            social_strength: 社交需求强度
            social_boundary: 社交边界

        Returns:
            社交紧急度 ∈ [0, 1]

        Raises:
            ValueError: 若 social_boundary <= 0
        """
        if social_boundary <= 0:
            raise ValueError(
                f"社交边界必须 > 0，收到 {social_boundary}"
            )
        urgency = max(0.0, (social_boundary - social_strength)
                      / social_boundary)
        return float(np.clip(urgency, 0.0, 1.0))

    def from_validation_result(
        self,
        validation_result: PCValidationResult,
        custom_boundary: float = DEFAULT_SOCIAL_BOUNDARY
    ) -> SocialDrive:
        """从验证结果构造完整的社交驱力描述。

        Args:
            validation_result: 阶段20的验证结果
            custom_boundary: 可选的自定义社交边界

        Returns:
            SocialDrive 实例
        """
        strength = self.compute_social_strength(validation_result)
        boundary = self.compute_social_boundary(
            validation_result, custom_boundary
        )
        urgency = self.compute_social_urgency(strength, boundary)
        return SocialDrive(
            validation_result=validation_result,
            social_strength=strength,
            social_boundary=boundary,
            social_urgency=urgency,
        )
