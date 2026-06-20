"""阶段54：误差信号的精度加权——逆协方差权重、自适应更新与注意力调制。

本阶段实现精度加权误差的数学定义和计算接口。
"""

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np

from .stage_053 import FeedbackSignal


@dataclass(frozen=True)
class PrecisionWeightedError:
    """精度加权误差的完整描述。

    组合阶段53的FeedbackSignal，在反馈信号的基础上增加
    精度加权机制——精度权重、加权误差、自适应更新、注意力调制。

    Attributes:
        feedback_signal: 阶段53的反馈信号（组合字段）
        precision_matrices: 各层精度矩阵
        weighted_error: 精度加权误差 ε_weighted（A54.2）
        layer_weights: 层间聚合权重
        layer_contributions: 各层贡献比
        update_beta: 自适应精度更新速率 β
        attention_weights: 注意力权重
        attention_gamma: 注意力温度 γ
        observed_precisions: 观测精度矩阵

    Invariants:
        - precision_matrices 各矩阵正定
        - weighted_error >= 0
        - layer_weights 归一化
        - attention_weights 归一化
        - update_beta ∈ (0, 1)
        - attention_gamma > 0
    """
    feedback_signal: FeedbackSignal
    precision_matrices: Dict[str, np.ndarray]
    weighted_error: float = 0.0
    layer_weights: Optional[np.ndarray] = None
    layer_contributions: Optional[np.ndarray] = None
    update_beta: float = 0.1
    attention_weights: Optional[np.ndarray] = None
    attention_gamma: float = 1.0
    observed_precisions: Optional[Dict[str, np.ndarray]] = None

    def __post_init__(self):
        if self.weighted_error < 0:
            raise ValueError(
                f"精度加权误差必须 >= 0，收到 {self.weighted_error}"
            )
        if self.layer_weights is not None:
            if not np.allclose(np.sum(self.layer_weights), 1.0):
                raise ValueError("层间权重必须归一化")
            if np.any(self.layer_weights < 0):
                raise ValueError("层间权重必须非负")
        if not (0 < self.update_beta < 1):
            raise ValueError(
                f"更新速率必须 ∈ (0,1)，收到 {self.update_beta}"
            )
        if self.attention_gamma <= 0:
            raise ValueError(
                f"注意力温度必须 > 0，收到 {self.attention_gamma}"
            )
        if self.attention_weights is not None:
            if not np.allclose(np.sum(self.attention_weights), 1.0):
                raise ValueError("注意力权重必须归一化")
        for name, mat in self.precision_matrices.items():
            try:
                np.linalg.cholesky(mat)
            except np.linalg.LinAlgError:
                raise ValueError(f"精度矩阵 {name} 必须正定")

    @property
    def dominant_layer(self) -> str:
        """精度加权误差贡献最大的层。"""
        if self.layer_contributions is None:
            return "unknown"
        layers = ["low", "mid", "high"]
        return layers[int(np.argmax(self.layer_contributions))]

    @property
    def attention_focus(self) -> str:
        """注意力最集中的层。"""
        if self.attention_weights is None:
            return "unknown"
        layers = ["low", "mid", "high"]
        return layers[int(np.argmax(self.attention_weights))]


class PrecisionWeighter:
    """阶段54的核心计算接口。

    实现精度权重的计算、精度加权误差的聚合、
    自适应精度更新和注意力调制。
    核心约束：
        - 精度矩阵正定
        - 精度加权误差非负
        - 注意力权重归一化
    """

    def compute_precision(
        self,
        covariance: np.ndarray,
        regularization: float = 1e-6
    ) -> np.ndarray:
        """计算精度权重（A54.1）。

        Π_l = Σ_l^{-1}

        Args:
            covariance: 协方差矩阵 Σ_l
            regularization: 正则化参数 λ

        Returns:
            精度矩阵 Π_l
        """
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            raise ValueError("协方差矩阵必须正定")
        reg_cov = covariance + regularization * np.eye(covariance.shape[0])
        precision = np.linalg.inv(reg_cov)
        return precision

    def compute_weighted_error(
        self,
        errors: Dict[str, np.ndarray],
        precisions: Dict[str, np.ndarray],
        weights: np.ndarray
    ) -> float:
        """计算精度加权误差（A54.2）。

        ε_weighted = Σ_l w_l · e_l^T Π_l e_l

        Args:
            errors: 各层误差信号
            precisions: 各层精度矩阵
            weights: 层间聚合权重

        Returns:
            精度加权误差 >= 0
        """
        if not np.allclose(np.sum(weights), 1.0):
            raise ValueError("层间权重必须归一化")
        if np.any(weights < 0):
            raise ValueError("层间权重必须非负")

        weighted = 0.0
        for i, (name, e) in enumerate(errors.items()):
            Pi = precisions[name]
            if e.shape[0] != Pi.shape[0]:
                raise ValueError(
                    f"误差 {name} 维度与精度矩阵不匹配"
                )
            energy = float(e.T @ Pi @ e)
            weighted += weights[i] * energy
        return weighted

    def update_precision(
        self,
        current_precision: np.ndarray,
        error_history: np.ndarray,
        beta: float,
        regularization: float = 1e-6
    ) -> np.ndarray:
        """自适应精度更新（A54.3）。

        Π(t+1) = (1-β)Π(t) + β·Π_observed(t)

        Args:
            current_precision: 当前精度矩阵 Π(t)
            error_history: 误差历史
            beta: 更新速率 ∈ (0,1)
            regularization: 正则化参数

        Returns:
            更新后的精度矩阵 Π(t+1)
        """
        if not (0 < beta < 1):
            raise ValueError(f"beta 必须 ∈ (0,1)，收到 {beta}")
        n = error_history.shape[0]
        mean_e = np.mean(error_history, axis=0)
        centered = error_history - mean_e
        sample_cov = (centered.T @ centered) / max(n, 1)
        reg_cov = sample_cov + regularization * np.eye(sample_cov.shape[0])
        observed_precision = np.linalg.inv(reg_cov)
        updated = (1.0 - beta) * current_precision + beta * observed_precision
        return updated

    def compute_attention(
        self,
        precision_weighted_errors: np.ndarray,
        gamma: float
    ) -> np.ndarray:
        """计算注意力权重（A54.4）。

        attention_l = softmax(γ · Π_l · ε_l)

        Args:
            precision_weighted_errors: 各层精度加权误差
            gamma: 注意力温度

        Returns:
            注意力权重
        """
        if gamma <= 0:
            raise ValueError(f"gamma 必须 > 0，收到 {gamma}")
        scaled = gamma * precision_weighted_errors
        exp_vals = np.exp(scaled - np.max(scaled))
        attention = exp_vals / np.sum(exp_vals)
        return attention

    def evaluate(
        self,
        feedback: FeedbackSignal,
        errors: Dict[str, np.ndarray],
        current_precisions: Dict[str, np.ndarray],
        error_history: Dict[str, np.ndarray],
        weights: Optional[np.ndarray] = None,
        beta: float = 0.1,
        gamma: float = 1.0
    ) -> PrecisionWeightedError:
        """完整评估精度加权误差（A54.1-A54.4）。

        Args:
            feedback: 反馈信号
            errors: 各层当前误差
            current_precisions: 当前精度矩阵
            error_history: 各层误差历史
            weights: 层间权重
            beta: 更新速率
            gamma: 注意力温度

        Returns:
            PrecisionWeightedError 实例
        """
        if weights is None:
            weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

        weighted_error = self.compute_weighted_error(
            errors, current_precisions, weights
        )

        updated_precisions = {}
        observed_precisions = {}
        for name, Pi in current_precisions.items():
            hist = error_history.get(name)
            if hist is not None and len(hist) > 0:
                updated = self.update_precision(Pi, hist, beta)
                updated_precisions[name] = updated
                n = hist.shape[0]
                mean_e = np.mean(hist, axis=0)
                centered = hist - mean_e
                sample_cov = (centered.T @ centered) / max(n, 1)
                reg_cov = sample_cov + 1e-6 * np.eye(sample_cov.shape[0])
                observed_precisions[name] = np.linalg.inv(reg_cov)
            else:
                updated_precisions[name] = Pi
                observed_precisions[name] = Pi

        pwe_vals = []
        for name in ["low", "mid", "high"]:
            e = errors.get(name)
            Pi = current_precisions.get(name)
            if e is not None and Pi is not None:
                val = float(e.T @ Pi @ e)
            else:
                val = 0.0
            pwe_vals.append(val)
        pwe_arr = np.array(pwe_vals)

        attention = self.compute_attention(pwe_arr, gamma)

        total = np.sum(pwe_arr)
        if total > 0:
            contributions = pwe_arr / total
        else:
            contributions = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

        return PrecisionWeightedError(
            feedback_signal=feedback,
            precision_matrices=updated_precisions,
            weighted_error=weighted_error,
            layer_weights=weights,
            layer_contributions=contributions,
            update_beta=beta,
            attention_weights=attention,
            attention_gamma=gamma,
            observed_precisions=observed_precisions,
        )
