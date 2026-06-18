"""阶段20：最小占位实现——为阶段21提供依赖。

本文件仅包含 PCValidationResult 数据类的最小实现，
满足阶段21的接口需求。
"""

from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np

from .stage_019 import PCNetworkConfig


@dataclass(frozen=True)
class PCValidationResult:
    """预测编码验证结果的最小占位实现。

    Attributes:
        network_config: 网络配置
        reconstruction_error: 重构误差
        sparsities: 各层稀疏度
        convergence_steps: 收敛步数
        ae_comparison: 自编码器对比结果
        error_maps: 误差热图
        feature_visualizations: 特征可视化
        free_energy_curve: 自由能曲线
        all_standards_met: 是否满足所有验证标准
    """
    network_config: PCNetworkConfig
    reconstruction_error: float
    sparsities: Tuple[float, ...]
    convergence_steps: Tuple[int, ...]
    ae_comparison: Dict[str, Tuple[float, float]]
    error_maps: Tuple[np.ndarray, ...]
    feature_visualizations: Tuple[np.ndarray, ...]
    free_energy_curve: Tuple[float, ...]
    all_standards_met: bool

    def __post_init__(self):
        if self.reconstruction_error < 0:
            raise ValueError(f"重构误差必须 ≥ 0，收到 {self.reconstruction_error}")
        for i, s in enumerate(self.sparsities):
            if not (0 <= s < 1):
                raise ValueError(f"第 {i} 层稀疏度必须 ∈ [0,1)，收到 {s}")
        for i, n in enumerate(self.convergence_steps):
            if n < 1:
                raise ValueError(f"第 {i} 个样本的收敛步数必须 ≥ 1，收到 {n}")
        standards = [
            self.reconstruction_error < 0.1,
            all(s > 0.5 for s in self.sparsities),
            np.mean(self.convergence_steps) < 20
        ]
        if self.all_standards_met and not all(standards):
            raise ValueError("all_standards_met 为 True，但验证标准未全部满足")

    @property
    def mean_convergence_steps(self) -> float:
        """平均收敛步数。"""
        return float(np.mean(self.convergence_steps))

    @property
    def min_sparsity(self) -> float:
        """最小稀疏度。"""
        return min(self.sparsities)

    @property
    def sparsity_by_layer(self) -> Dict[int, float]:
        """各层稀疏度。"""
        return {l: s for l, s in enumerate(self.sparsities)}
