"""阶段22强化版：社会驱力——从表征稀疏度计算信息效率需求。

本文件是 stage_022.py 的强化升级版，核心改进：
1. 从真实表征计算稀疏度（而非启发式公式）
2. 增加信息熵作为社会驱力指标
3. 支持 HierarchicalGenerativeModelV2
"""

from dataclasses import dataclass
from typing import Union, Optional
import numpy as np

from agent_whitepaper.core.stage_011 import HierarchicalGenerativeModel
from agent_whitepaper.core.stage_011_v2 import HierarchicalGenerativeModelV2


# ============================================================
# 阶段22-v2 数据类：社会驱力
# ============================================================
@dataclass(frozen=True)
class SocialDriveV2:
    """社会驱力的完整描述（强化版）。

    Attributes:
        sparsity: 表征稀疏度 ∈ [0, 1]
        entropy: 表征信息熵
        social_strength: 社会需求强度 ∈ [0, 1]
        social_boundary: 社会边界 ∈ [0, 1]
        social_urgency: 社会紧急度 ∈ [0, 1]
        target_sparsity: 目标稀疏度

    Invariants:
        - 0 ≤ social_strength ≤ 1
        - 0 ≤ social_boundary ≤ 1
        - 0 ≤ social_urgency ≤ 1
    """
    sparsity: float
    entropy: float
    social_strength: float
    social_boundary: float
    social_urgency: float
    target_sparsity: float

    def __post_init__(self):
        if not (0 <= self.social_strength <= 1):
            raise ValueError(f"社会需求强度必须 ∈ [0,1]，收到 {self.social_strength}")
        if not (0 <= self.social_boundary <= 1):
            raise ValueError(f"社会边界必须 ∈ [0,1]，收到 {self.social_boundary}")
        if not (0 <= self.social_urgency <= 1):
            raise ValueError(f"社会紧急度必须 ∈ [0,1]，收到 {self.social_urgency}")


# ============================================================
# 阶段22-v2 核心接口：社会驱力计算器
# ============================================================
class SocialDriveCalculatorV2:
    """社会驱力计算器（强化版）。

    核心改进：
    1. 从真实表征计算稀疏度和信息熵
    2. 稀疏度 = 接近零的元素比例
    3. 信息熵 = -sum(p * log(p))

    计算公式：
    - sparsity = count(|r| < threshold) / dim
    - social_strength = sparsity（越稀疏越强）
    """

    DEFAULT_BOUNDARY: float = 0.3
    DEFAULT_TARGET_SPARSITY: float = 0.5
    SPARSITY_THRESHOLD: float = 0.1

    def compute(
        self,
        representations: tuple[np.ndarray, ...],
        target_sparsity: float = DEFAULT_TARGET_SPARSITY
    ) -> SocialDriveV2:
        """计算社会驱力。

        Args:
            representations: 各层表征
            target_sparsity: 目标稀疏度

        Returns:
            SocialDriveV2 实例
        """
        if not representations:
            raise ValueError("表征不能为空")

        # 计算整体稀疏度
        all_elements = np.concatenate([r.flatten() for r in representations])
        sparsity = float(np.mean(np.abs(all_elements) < self.SPARSITY_THRESHOLD))

        # 计算信息熵（将表征值归一化为概率分布）
        abs_vals = np.abs(all_elements)
        if np.sum(abs_vals) > 0:
            probs = abs_vals / np.sum(abs_vals)
            entropy = -float(np.sum(probs * np.log(probs + 1e-10)))
        else:
            entropy = 0.0

        # 社会强度 = 当前稀疏度（越稀疏越强）
        social_strength = sparsity

        boundary = self.DEFAULT_BOUNDARY
        urgency = max(0.0, (boundary - social_strength) / boundary) if boundary > 0 else 0.0

        return SocialDriveV2(
            sparsity=sparsity,
            entropy=entropy,
            social_strength=social_strength,
            social_boundary=boundary,
            social_urgency=urgency,
            target_sparsity=target_sparsity,
        )

    def compute_from_model(
        self,
        model: Union[HierarchicalGenerativeModel, HierarchicalGenerativeModelV2],
        representations: tuple[np.ndarray, ...]
    ) -> SocialDriveV2:
        """从模型直接计算社会驱力。"""
        return self.compute(representations)
