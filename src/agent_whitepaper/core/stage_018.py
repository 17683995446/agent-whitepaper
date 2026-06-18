"""阶段18：最小占位实现——为阶段21-27提供依赖。

本文件仅包含 EquivalenceProof 数据类的最小实现，
满足阶段19-20的接口需求。
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class EquivalenceProof:
    """等价性证明的最小占位实现。

    Attributes:
        pc_gradient: 预测编码梯度（占位）
        fe_gradient: 自由能梯度（占位）
        max_difference: 最大差异（占位）
    """
    pc_gradient: Optional[np.ndarray] = None
    fe_gradient: Optional[np.ndarray] = None
    max_difference: float = 0.0
