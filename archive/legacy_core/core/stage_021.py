"""阶段21：安全驱力——从预测编码到安全需求。

本阶段实现安全驱力的数学定义和计算接口，
将预测编码验证结果映射为安全需求强度。
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .stage_020 import PCValidationResult


@dataclass(frozen=True)
class SafetyDrive:
    """安全驱力的完整描述。

    组合阶段20的PCValidationResult，在验证结果的基础上增加
    安全需求强度、边界和紧急度。

    Attributes:
        validation_result: 阶段20的验证结果（组合字段）
        safety_strength: 安全需求强度 s_safety ∈ [0, 1]
        safety_boundary: 安全边界 b_safety ∈ [0, 1]
        safety_urgency: 安全紧急度 u_safety ∈ [0, 1]

    Invariants:
        - 0 ≤ safety_strength ≤ 1
        - 0 ≤ safety_boundary ≤ 1
        - 0 ≤ safety_urgency ≤ 1
        - safety_urgency = max(0, (safety_boundary - safety_strength) / safety_boundary)
          当 safety_boundary > 0
    """
    validation_result: PCValidationResult
    safety_strength: float
    safety_boundary: float
    safety_urgency: float

    def __post_init__(self):
        if not (0 <= self.safety_strength <= 1):
            raise ValueError(
                f"安全需求强度必须 ∈ [0,1]，收到 {self.safety_strength}"
            )
        if not (0 <= self.safety_boundary <= 1):
            raise ValueError(
                f"安全边界必须 ∈ [0,1]，收到 {self.safety_boundary}"
            )
        if not (0 <= self.safety_urgency <= 1):
            raise ValueError(
                f"安全紧急度必须 ∈ [0,1]，收到 {self.safety_urgency}"
            )
        # 验证紧急度与强度、边界的一致性
        if self.safety_boundary > 1e-12:
            expected_urgency = max(
                0.0, (self.safety_boundary - self.safety_strength)
                / self.safety_boundary
            )
            if abs(self.safety_urgency - expected_urgency) > 1e-6:
                raise ValueError(
                    f"紧急度 {self.safety_urgency} 与公式计算结果 "
                    f"{expected_urgency} 不一致"
                )

    @property
    def is_violated(self) -> bool:
        """安全边界是否被违反（s_safety < b_safety）。"""
        return self.safety_strength < self.safety_boundary

    @property
    def violation_severity(self) -> float:
        """违反严重程度 = max(0, b_safety - s_safety)。"""
        return max(0.0, self.safety_boundary - self.safety_strength)


class SafetyDriveCalculator:
    """阶段21的核心计算接口。

    实现从预测编码验证结果到安全驱力的映射，
    包括安全需求强度计算、边界设定和紧急度评估。
    所有方法均为纯函数（无副作用）。

    核心约束：
        - 安全需求强度 = 1 - 重构误差（A21.1）
        - 安全边界 = 0.5（A21.2）
        - 紧急度 = max(0, (边界 - 强度) / 边界)（A21.3）
    """

    DEFAULT_SAFETY_BOUNDARY: float = 0.5

    def compute_safety_strength(
        self,
        validation_result: PCValidationResult
    ) -> float:
        """计算安全需求强度（A21.1）。

        s_safety = 1 - MSE_recon，归一化到 [0, 1]。

        Args:
            validation_result: 阶段20的验证结果

        Returns:
            安全需求强度 ∈ [0, 1]

        Raises:
            ValueError: 若 validation_result 为 None
        """
        if validation_result is None:
            raise ValueError("validation_result 不能为空")
        mse = validation_result.reconstruction_error
        strength = 1.0 - min(mse, 1.0)
        return float(np.clip(strength, 0.0, 1.0))

    def compute_safety_boundary(
        self,
        validation_result: PCValidationResult,
        custom_boundary: float = DEFAULT_SAFETY_BOUNDARY
    ) -> float:
        """计算安全边界（A21.2）。

        默认 b_safety = 0.5，可自定义。

        Args:
            validation_result: 阶段20的验证结果
            custom_boundary: 自定义边界值

        Returns:
            安全边界 ∈ [0, 1]

        Raises:
            ValueError: 若 custom_boundary 不在 [0, 1] 内
        """
        if validation_result is None:
            raise ValueError("validation_result 不能为空")
        if not (0 <= custom_boundary <= 1):
            raise ValueError(
                f"自定义边界必须 ∈ [0,1]，收到 {custom_boundary}"
            )
        return float(custom_boundary)

    def compute_safety_urgency(
        self,
        safety_strength: float,
        safety_boundary: float
    ) -> float:
        """计算安全紧急度（A21.3）。

        u_safety = max(0, (b_safety - s_safety) / b_safety)。

        Args:
            safety_strength: 安全需求强度
            safety_boundary: 安全边界

        Returns:
            安全紧急度 ∈ [0, 1]

        Raises:
            ValueError: 若 safety_boundary <= 0
        """
        if safety_boundary <= 0:
            raise ValueError(
                f"安全边界必须 > 0，收到 {safety_boundary}"
            )
        urgency = max(0.0, (safety_boundary - safety_strength)
                      / safety_boundary)
        return float(np.clip(urgency, 0.0, 1.0))

    def from_validation_result(
        self,
        validation_result: PCValidationResult,
        custom_boundary: float = DEFAULT_SAFETY_BOUNDARY
    ) -> SafetyDrive:
        """从验证结果构造完整的安全驱力描述。

        Args:
            validation_result: 阶段20的验证结果
            custom_boundary: 可选的自定义安全边界

        Returns:
            SafetyDrive 实例
        """
        strength = self.compute_safety_strength(validation_result)
        boundary = self.compute_safety_boundary(
            validation_result, custom_boundary
        )
        urgency = self.compute_safety_urgency(strength, boundary)
        return SafetyDrive(
            validation_result=validation_result,
            safety_strength=strength,
            safety_boundary=boundary,
            safety_urgency=urgency,
        )
