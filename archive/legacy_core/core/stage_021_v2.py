"""阶段21强化版：安全驱力——从真实自由能计算安全需求。

本文件是 stage_021.py 的强化升级版，核心改进：
1. 从真实自由能计算安全驱力强度（而非启发式公式）
2. 支持 HierarchicalGenerativeModelV2
3. 增加自由能变化率作为安全指标
"""

from dataclasses import dataclass
from typing import Union, Optional
import numpy as np

from agent_whitepaper.core.stage_011 import HierarchicalGenerativeModel
from agent_whitepaper.core.stage_011_v2 import HierarchicalGenerativeModelV2


# ============================================================
# 阶段21-v2 数据类：安全驱力
# ============================================================
@dataclass(frozen=True)
class SafetyDriveV2:
    """安全驱力的完整描述（强化版）。

    Attributes:
        free_energy: 当前自由能值
        free_energy_rate: 自由能变化率（负值表示下降=安全）
        safety_strength: 安全需求强度 s_safety ∈ [0, 1]
        safety_boundary: 安全边界 b_safety ∈ [0, 1]
        safety_urgency: 安全紧急度 u_safety ∈ [0, 1]
        max_acceptable_fe: 最大可接受自由能

    Invariants:
        - 0 ≤ safety_strength ≤ 1
        - 0 ≤ safety_boundary ≤ 1
        - 0 ≤ safety_urgency ≤ 1
    """
    free_energy: float
    free_energy_rate: float
    safety_strength: float
    safety_boundary: float
    safety_urgency: float
    max_acceptable_fe: float

    def __post_init__(self):
        if not (0 <= self.safety_strength <= 1):
            raise ValueError(f"安全需求强度必须 ∈ [0,1]，收到 {self.safety_strength}")
        if not (0 <= self.safety_boundary <= 1):
            raise ValueError(f"安全边界必须 ∈ [0,1]，收到 {self.safety_boundary}")
        if not (0 <= self.safety_urgency <= 1):
            raise ValueError(f"安全紧急度必须 ∈ [0,1]，收到 {self.safety_urgency}")
        if self.max_acceptable_fe <= 0:
            raise ValueError(f"最大可接受自由能必须 > 0，收到 {self.max_acceptable_fe}")

    @property
    def is_violated(self) -> bool:
        """安全边界是否被违反。"""
        return self.safety_strength < self.safety_boundary

    @property
    def violation_severity(self) -> float:
        """违反严重程度。"""
        return max(0.0, self.safety_boundary - self.safety_strength)


# ============================================================
# 阶段21-v2 核心接口：安全驱力计算器
# ============================================================
class SafetyDriveCalculatorV2:
    """安全驱力计算器（强化版）。

    核心改进：
    1. 从真实自由能计算安全驱力
    2. 自由能变化率作为动态安全指标
    3. 支持 v1/v2 模型

    计算公式：
    - safety_strength = 1 - min(1, F / F_max)
    - safety_boundary = 0.7（固定）
    - safety_urgency = max(0, (boundary - strength) / boundary)
    """

    DEFAULT_BOUNDARY: float = 0.7
    DEFAULT_MAX_FE: float = 100.0

    def compute(
        self,
        free_energy: float,
        free_energy_history: Optional[tuple[float, ...]] = None,
        max_acceptable_fe: float = DEFAULT_MAX_FE
    ) -> SafetyDriveV2:
        """计算安全驱力。

        Args:
            free_energy: 当前自由能值，必须 >= 0
            free_energy_history: 自由能历史（用于计算变化率）
            max_acceptable_fe: 最大可接受自由能

        Returns:
            SafetyDriveV2 实例
        """
        if free_energy < 0:
            raise ValueError(f"free_energy 必须 >= 0，收到 {free_energy}")
        if max_acceptable_fe <= 0:
            raise ValueError(f"max_acceptable_fe 必须 > 0")

        # 计算自由能变化率
        fe_rate = 0.0
        if free_energy_history is not None and len(free_energy_history) >= 2:
            fe_rate = free_energy_history[-1] - free_energy_history[-2]

        # 安全强度 = 1 - 归一化自由能（自由能越低越安全）
        safety_strength = 1.0 - min(1.0, max(0.0, free_energy / max_acceptable_fe))

        # 如果自由能在上升，降低安全强度
        if fe_rate > 0:
            safety_strength = max(0.0, safety_strength - fe_rate / max_acceptable_fe)

        boundary = self.DEFAULT_BOUNDARY
        urgency = max(0.0, (boundary - safety_strength) / boundary) if boundary > 0 else 0.0

        return SafetyDriveV2(
            free_energy=free_energy,
            free_energy_rate=fe_rate,
            safety_strength=safety_strength,
            safety_boundary=boundary,
            safety_urgency=urgency,
            max_acceptable_fe=max_acceptable_fe,
        )

    def compute_from_model(
        self,
        model: Union[HierarchicalGenerativeModel, HierarchicalGenerativeModelV2],
        observation: np.ndarray,
        representations: tuple[np.ndarray, ...],
        free_energy_history: Optional[tuple[float, ...]] = None
    ) -> SafetyDriveV2:
        """从模型直接计算安全驱力。

        Args:
            model: 层级生成模型（v1 或 v2）
            observation: 当前观测
            representations: 当前表征
            free_energy_history: 自由能历史

        Returns:
            SafetyDriveV2 实例
        """
        if isinstance(model, HierarchicalGenerativeModelV2):
            fe = model.compute_free_energy(observation, representations)
        else:
            from agent_whitepaper.core.stage_011 import HierarchicalModelBuilder
            builder = HierarchicalModelBuilder()
            fe = builder.compute_free_energy_from_model(model, observation, representations)
        return self.compute(fe, free_energy_history)
