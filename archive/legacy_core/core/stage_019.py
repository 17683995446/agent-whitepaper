"""阶段19：最小占位实现——为阶段20-21提供依赖。

本文件仅包含 PCNetworkConfig 数据类的最小实现，
满足阶段20的接口需求。
"""

from dataclasses import dataclass
from typing import Literal, Tuple
import numpy as np

from .stage_018 import EquivalenceProof


@dataclass(frozen=True)
class PCNetworkConfig:
    """预测编码网络配置的最小占位实现。

    Attributes:
        equivalence_proof: 阶段18的等价性证明结果
        layer_dims: 各层维度元组
        activations: 各层激活函数元组
        weight_init: 权重初始化方法
        precision_init: 精度矩阵初始值
        num_epochs: 训练轮数
        batch_size: 批量大小
    """
    equivalence_proof: EquivalenceProof
    layer_dims: Tuple[int, ...]
    activations: Tuple[str, ...]
    weight_init: Literal['xavier', 'he', 'normal'] = 'xavier'
    precision_init: float = 1.0
    num_epochs: int = 100
    batch_size: int = 32

    def __post_init__(self):
        if len(self.layer_dims) < 3:
            raise ValueError(f"至少需要3层，收到 {len(self.layer_dims)} 层")
        if len(self.activations) != len(self.layer_dims) - 1:
            raise ValueError(
                f"激活函数数量 ({len(self.activations)}) 必须 = 层数-1 ({len(self.layer_dims) - 1})"
            )
        if self.precision_init <= 0:
            raise ValueError(f"精度初始值必须 > 0，收到 {self.precision_init}")
        if self.num_epochs < 1:
            raise ValueError(f"训练轮数必须 ≥ 1，收到 {self.num_epochs}")
        if self.batch_size < 1:
            raise ValueError(f"批量大小必须 ≥ 1，收到 {self.batch_size}")

    @property
    def num_layers(self) -> int:
        """网络层数（不含输入层）。"""
        return len(self.layer_dims) - 1

    @property
    def total_parameters(self) -> int:
        """总参数量。"""
        return sum(
            self.layer_dims[l] * self.layer_dims[l + 1]
            for l in range(self.num_layers)
        )
