"""Stage 13: Top-Down Prediction Propagation.

自顶向下预测传播，计算预测和更新表征。
"""

from dataclasses import dataclass
import numpy as np

from agent_whitepaper.core.stage_011 import HierarchicalGenerativeModel
from agent_whitepaper.core.stage_012 import BottomUpPropagation


# ============================================================
# Stage 13 Data Class: Top-Down Propagation
# ============================================================
@dataclass(frozen=True)
class TopDownPropagation:
    """自顶向下传播结果。

    Attributes:
        bottom_up: 自底向上传播结果
        predictions: 各层预测 (r̂^1, ..., r̂^{L-1})
        precision_weighted_predictions: 精度加权预测
        obs_prediction: 观测预测 ô
        updated_representations: 更新后的表征
    """
    bottom_up: BottomUpPropagation
    predictions: tuple[np.ndarray, ...]
    precision_weighted_predictions: tuple[np.ndarray, ...]
    obs_prediction: np.ndarray
    updated_representations: tuple[np.ndarray, ...]


# ============================================================
# Stage 13 Core Interface: Top-Down Inference Engine
# ============================================================
class TopDownInferenceEngine:
    """自顶向下推断引擎。"""

    def predict(
        self,
        model: HierarchicalGenerativeModel,
        bottom_up: BottomUpPropagation,
        top_representation: np.ndarray
    ) -> TopDownPropagation:
        """执行自顶向下预测传播。"""
        if top_representation.shape[0] != model.layer_dims[-1]:
            raise ValueError(
                f"顶层表征维度 {top_representation.shape[0]} 必须等于 {model.layer_dims[-1]}"
            )
        predictions, obs_prediction = self._cascade_predict(model, top_representation)
        precision_weighted = []
        updated = []
        deltas = bottom_up.propagated_signals
        for l in range(model.num_layers - 1):
            # predictions[l] = r^{l+1}，维度 = layer_dims[l]
            # precisions[l+1] 的维度也是 layer_dims[l]（child of layer l+1）
            pi = model.precisions[l + 1]
            pw = pi @ predictions[l]
            precision_weighted.append(pw)
            pi_inv = np.linalg.inv(pi)
            # 使用当前层的传播信号更新当前层预测
            upd = predictions[l] + pi_inv @ deltas[l]
            updated.append(upd)
        return TopDownPropagation(
            bottom_up=bottom_up,
            predictions=predictions,
            precision_weighted_predictions=tuple(precision_weighted),
            obs_prediction=obs_prediction,
            updated_representations=tuple(updated),
        )

    def _cascade_predict(
        self,
        model: HierarchicalGenerativeModel,
        top_representation: np.ndarray
    ) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
        """级联预测。"""
        L = model.num_layers
        predictions = []
        r_current = top_representation
        for l in range(L - 1, 0, -1):
            r_hat = model.generative_fns[l](r_current, model.weights[l])
            predictions.insert(0, r_hat)
            r_current = r_hat
        obs_prediction = model.obs_generative_fn(predictions[0]) if predictions else model.obs_generative_fn(top_representation)
        return tuple(predictions), obs_prediction

    def compute_updated_representation(
        self,
        prediction: np.ndarray,
        precision: np.ndarray,
        propagated_signal: np.ndarray
    ) -> np.ndarray:
        """计算更新后的表征。"""
        try:
            np.linalg.cholesky(precision)
        except np.linalg.LinAlgError:
            raise ValueError("精度矩阵必须正定")
        precision_inv = np.linalg.inv(precision)
        return prediction + precision_inv @ propagated_signal
