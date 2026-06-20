"""阶段51：预测误差的标准化定义——三层预测体系的逐层误差度量与全局聚合。

本阶段实现标准化误差的数学定义和计算接口。
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum
import numpy as np

from .stage_050 import CrossLayerProtocol


class WeightScheme(Enum):
    """全局误差聚合权重方案。"""
    UNIFORM = "uniform"
    LOW_PRIORITY = "low_priority"
    HIGH_PRIORITY = "high_priority"
    DYNAMIC = "dynamic"


@dataclass(frozen=True)
class StandardizedError:
    """三层预测体系的标准化误差描述。

    组合阶段50的CrossLayerProtocol，在跨层协议的基础上增加
    标准化误差度量——逐层误差向量和全局误差。

    Attributes:
        cross_layer_protocol: 阶段50的跨层交互协议（组合字段）
        epsilon_low: 底层误差 ε_low
        epsilon_mid: 中层误差 ε_mid
        epsilon_high: 高层误差 ε_high
        weights: 聚合权重 [w_low, w_mid, w_high]
        weight_scheme: 权重方案
        epsilon_global: 全局误差 ε_global
        observation_low: 底层实际观测
        observation_mid: 中层实际趋势
        observation_high: 高层实际模式
        prediction_low: 底层预测
        prediction_mid: 中层趋势预测
        prediction_high: 高层模式预测
        timestamp: 误差计算时间戳

    Invariants:
        - epsilon_low, epsilon_mid, epsilon_high >= 0
        - weights 之和为 1
        - epsilon_global == Σ w_l · ε_l
    """
    cross_layer_protocol: CrossLayerProtocol
    epsilon_low: float
    epsilon_mid: float
    epsilon_high: float
    weights: np.ndarray
    weight_scheme: WeightScheme = WeightScheme.UNIFORM
    epsilon_global: float = 0.0
    observation_low: Optional[np.ndarray] = None
    observation_mid: Optional[np.ndarray] = None
    observation_high: Optional[np.ndarray] = None
    prediction_low: Optional[np.ndarray] = None
    prediction_mid: Optional[np.ndarray] = None
    prediction_high: Optional[np.ndarray] = None
    timestamp: float = 0.0

    def __post_init__(self):
        for name, val in [("epsilon_low", self.epsilon_low),
                          ("epsilon_mid", self.epsilon_mid),
                          ("epsilon_high", self.epsilon_high)]:
            if val < 0:
                raise ValueError(f"{name} 必须 >= 0，收到 {val}")
        if self.weights is not None:
            if len(self.weights) != 3:
                raise ValueError(
                    f"权重向量长度必须为3，收到 {len(self.weights)}"
                )
            if not np.allclose(np.sum(self.weights), 1.0):
                raise ValueError(
                    f"权重之和必须为1，收到 {np.sum(self.weights)}"
                )
            if np.any(self.weights < 0):
                raise ValueError("权重必须非负")
            expected_global = (
                self.weights[0] * self.epsilon_low
                + self.weights[1] * self.epsilon_mid
                + self.weights[2] * self.epsilon_high
            )
            if not np.allclose(self.epsilon_global, expected_global, atol=1e-6):
                raise ValueError(
                    f"全局误差与加权和不一致："
                    f"ε_global={self.epsilon_global}，加权和={expected_global}"
                )

    @property
    def epsilon_vector(self) -> np.ndarray:
        """逐层误差向量（A51.1）。"""
        return np.array([self.epsilon_low, self.epsilon_mid, self.epsilon_high])

    @property
    def dominant_error_layer(self) -> str:
        """误差最大的层级。"""
        layers = ["low", "mid", "high"]
        errors = [self.epsilon_low, self.epsilon_mid, self.epsilon_high]
        return layers[int(np.argmax(errors))]

    @property
    def is_zero(self) -> bool:
        """三层误差是否全部为零。"""
        return (self.epsilon_low == 0.0
                and self.epsilon_mid == 0.0
                and self.epsilon_high == 0.0)


class StandardizedErrorCalculator:
    """阶段51的核心计算接口。

    计算三层预测体系的逐层误差和全局误差。
    核心约束：
        - 逐层误差非负
        - 全局误差等于加权和
        - 权重归一化
    """

    def compute_layer_error(
        self,
        observation: np.ndarray,
        prediction: np.ndarray
    ) -> float:
        """计算单层误差（A51.2-A51.4）。

        ε_l = ||o_l - ô_l||²

        Args:
            observation: 实际观测
            prediction: 预测值

        Returns:
            误差 ε_l >= 0
        """
        if observation.shape != prediction.shape:
            raise ValueError(
                f"观测维度 {observation.shape} 与预测维度 {prediction.shape} 不匹配"
            )
        diff = observation - prediction
        return float(np.sum(diff ** 2))

    def compute_global_error(
        self,
        epsilon_low: float,
        epsilon_mid: float,
        epsilon_high: float,
        weights: np.ndarray
    ) -> float:
        """计算全局误差（A51.5）。

        ε_global = Σ_l w_l · ε_l

        Args:
            epsilon_low: 底层误差
            epsilon_mid: 中层误差
            epsilon_high: 高层误差
            weights: 聚合权重

        Returns:
            全局误差 >= 0
        """
        if not np.allclose(np.sum(weights), 1.0):
            raise ValueError("权重必须归一化")
        if np.any(weights < 0):
            raise ValueError("权重必须非负")
        return float(
            weights[0] * epsilon_low
            + weights[1] * epsilon_mid
            + weights[2] * epsilon_high
        )

    def get_weights_for_scheme(
        self,
        scheme: WeightScheme
    ) -> np.ndarray:
        """获取指定权重方案的权重向量。"""
        schemes = {
            WeightScheme.UNIFORM: np.array([1.0/3.0, 1.0/3.0, 1.0/3.0]),
            WeightScheme.LOW_PRIORITY: np.array([0.5, 0.3, 0.2]),
            WeightScheme.HIGH_PRIORITY: np.array([0.2, 0.3, 0.5]),
            WeightScheme.DYNAMIC: np.array([1.0/3.0, 1.0/3.0, 1.0/3.0]),
        }
        return schemes.get(scheme, schemes[WeightScheme.UNIFORM])

    def compute(
        self,
        cross_layer_protocol: CrossLayerProtocol,
        observation_low: np.ndarray,
        prediction_low: np.ndarray,
        observation_mid: np.ndarray,
        prediction_mid: np.ndarray,
        observation_high: np.ndarray,
        prediction_high: np.ndarray,
        weight_scheme: WeightScheme = WeightScheme.UNIFORM,
        timestamp: float = 0.0,
    ) -> StandardizedError:
        """计算完整的标准化误差。

        Args:
            cross_layer_protocol: 跨层协议
            observation_low: 底层观测
            prediction_low: 底层预测
            observation_mid: 中层观测
            prediction_mid: 中层预测
            observation_high: 高层观测
            prediction_high: 高层预测
            weight_scheme: 权重方案
            timestamp: 时间戳

        Returns:
            StandardizedError 实例
        """
        epsilon_low = self.compute_layer_error(observation_low, prediction_low)
        epsilon_mid = self.compute_layer_error(observation_mid, prediction_mid)
        epsilon_high = self.compute_layer_error(observation_high, prediction_high)
        weights = self.get_weights_for_scheme(weight_scheme)
        epsilon_global = self.compute_global_error(
            epsilon_low, epsilon_mid, epsilon_high, weights
        )
        return StandardizedError(
            cross_layer_protocol=cross_layer_protocol,
            epsilon_low=epsilon_low,
            epsilon_mid=epsilon_mid,
            epsilon_high=epsilon_high,
            weights=weights,
            weight_scheme=weight_scheme,
            epsilon_global=epsilon_global,
            observation_low=observation_low,
            observation_mid=observation_mid,
            observation_high=observation_high,
            prediction_low=prediction_low,
            prediction_mid=prediction_mid,
            prediction_high=prediction_high,
            timestamp=timestamp,
        )
