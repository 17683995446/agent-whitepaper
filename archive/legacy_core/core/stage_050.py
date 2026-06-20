"""阶段50：最小占位实现——为阶段51-55提供依赖。

本文件仅包含 CrossLayerProtocol 数据类的最小实现，
满足阶段51的接口需求。
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class CrossLayerProtocol:
    """跨层交互协议的最小占位实现。

    Attributes:
        time_scale_ratio: 时间尺度比例 (low:mid:high)
        attention_weights: 注意力权重
    """
    time_scale_ratio: Tuple[int, int, int] = (1, 10, 100)
    attention_weights: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.attention_weights is not None:
            if not np.allclose(np.sum(self.attention_weights), 1.0):
                raise ValueError("注意力权重必须归一化")
