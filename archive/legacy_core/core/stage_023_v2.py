"""阶段23强化版：探索驱力——从收敛速度计算学习需求。

本文件是 stage_023.py 的强化升级版，核心改进：
1. 从真实收敛历史计算探索驱力（而非固定步数）
2. 增加收敛加速度指标
3. 支持 HierarchicalGenerativeModelV2
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


# ============================================================
# 阶段23-v2 数据类：探索驱力
# ============================================================
@dataclass(frozen=True)
class ExplorationDriveV2:
    """探索驱力的完整描述（强化版）。

    Attributes:
        convergence_rate: 收敛速度（梯度范数下降率）
        convergence_acceleration: 收敛加速度
        exploration_strength: 探索需求强度 ∈ [0, 1]
        exploration_boundary: 探索边界 ∈ [0, 1]
        exploration_urgency: 探索紧急度 ∈ [0, 1]
        max_gradient_norm: 最大梯度范数历史

    Invariants:
        - 0 ≤ exploration_strength ≤ 1
        - 0 ≤ exploration_boundary ≤ 1
        - 0 ≤ exploration_urgency ≤ 1
    """
    convergence_rate: float
    convergence_acceleration: float
    exploration_strength: float
    exploration_boundary: float
    exploration_urgency: float
    max_gradient_norm: float

    def __post_init__(self):
        if not (0 <= self.exploration_strength <= 1):
            raise ValueError(f"探索需求强度必须 ∈ [0,1]，收到 {self.exploration_strength}")
        if not (0 <= self.exploration_boundary <= 1):
            raise ValueError(f"探索边界必须 ∈ [0,1]，收到 {self.exploration_boundary}")
        if not (0 <= self.exploration_urgency <= 1):
            raise ValueError(f"探索紧急度必须 ∈ [0,1]，收到 {self.exploration_urgency}")


# ============================================================
# 阶段23-v2 核心接口：探索驱力计算器
# ============================================================
class ExplorationDriveCalculatorV2:
    """探索驱力计算器（强化版）。

    核心改进：
    1. 从真实梯度范数历史计算收敛速度
    2. 收敛加速度 = 收敛速度的变化率
    3. 探索强度 = 1 - 收敛速度（收敛越慢越需要探索）

    计算公式：
    - convergence_rate = (gn_prev - gn_curr) / gn_prev
    - exploration_strength = 1 - min(1, convergence_rate * scale)
    """

    DEFAULT_BOUNDARY: float = 0.5
    DEFAULT_SCALE: float = 10.0

    def compute(
        self,
        gradient_norm_history: tuple[float, ...],
        max_gradient_norm: Optional[float] = None
    ) -> ExplorationDriveV2:
        """计算探索驱力。

        Args:
            gradient_norm_history: 梯度范数历史
            max_gradient_norm: 最大梯度范数（用于归一化）

        Returns:
            ExplorationDriveV2 实例
        """
        if not gradient_norm_history:
            raise ValueError("梯度范数历史不能为空")

        gn = np.array(gradient_norm_history)
        max_gn = max_gradient_norm if max_gradient_norm is not None else float(np.max(gn))
        if max_gn <= 0:
            max_gn = 1.0

        # 计算收敛速度（最近两个点的下降率）
        if len(gn) >= 2:
            gn_prev = gn[-2]
            gn_curr = gn[-1]
            if gn_prev > 0:
                convergence_rate = (gn_prev - gn_curr) / gn_prev
            else:
                convergence_rate = 0.0
        else:
            convergence_rate = 0.0

        # 计算收敛加速度（速度的变化率）
        if len(gn) >= 3:
            rate_prev = (gn[-3] - gn[-2]) / gn[-3] if gn[-3] > 0 else 0.0
            convergence_acceleration = convergence_rate - rate_prev
        else:
            convergence_acceleration = 0.0

        # 探索强度 = 1 - 归一化收敛速度（收敛越慢越需要探索）
        scaled_rate = convergence_rate * self.DEFAULT_SCALE
        exploration_strength = 1.0 - min(1.0, max(0.0, scaled_rate))

        # 如果收敛在加速，降低探索需求
        if convergence_acceleration > 0:
            exploration_strength = max(0.0, exploration_strength - convergence_acceleration)

        boundary = self.DEFAULT_BOUNDARY
        urgency = max(0.0, (boundary - exploration_strength) / boundary) if boundary > 0 else 0.0

        return ExplorationDriveV2(
            convergence_rate=convergence_rate,
            convergence_acceleration=convergence_acceleration,
            exploration_strength=exploration_strength,
            exploration_boundary=boundary,
            exploration_urgency=urgency,
            max_gradient_norm=max_gn,
        )

    def compute_from_perception_update(
        self,
        gradient_norm_history: tuple[float, ...]
    ) -> ExplorationDriveV2:
        """从感知更新历史计算探索驱力。"""
        return self.compute(gradient_norm_history)
