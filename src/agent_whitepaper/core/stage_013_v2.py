"""阶段13强化版：自顶向下预测传播（支持非线性激活与v2模型）。

本文件是 stage_013.py 的强化升级版，核心改进：
1. 支持 HierarchicalGenerativeModelV2（非线性激活）
2. 增加预测置信度计算
3. 保持与 stage_013.py 的完全向后兼容
"""

from dataclasses import dataclass
from typing import Union
import numpy as np

from agent_whitepaper.core.stage_011 import HierarchicalGenerativeModel
from agent_whitepaper.core.stage_011_v2 import HierarchicalGenerativeModelV2
from agent_whitepaper.core.stage_012_v2 import BottomUpPropagationV2


# ============================================================
# 阶段13-v2 数据类：自顶向下传播
# ============================================================
@dataclass(frozen=True)
class TopDownPropagationV2:
    """自顶向下传播结果（强化版）。

    Attributes:
        bottom_up: 自底向上传播结果
        predictions: 各层预测 (r̂^1, ..., r̂^{L-1})
        precision_weighted_predictions: 精度加权预测
        obs_prediction: 观测预测 ô
        updated_representations: 更新后的表征
        prediction_confidences: 各层预测置信度
    """
    bottom_up: BottomUpPropagationV2
    predictions: tuple[np.ndarray, ...]
    precision_weighted_predictions: tuple[np.ndarray, ...]
    obs_prediction: np.ndarray
    updated_representations: tuple[np.ndarray, ...]
    prediction_confidences: tuple[float, ...]


# ============================================================
# 阶段13-v2 核心接口：自顶向下推断引擎
# ============================================================
class TopDownInferenceEngineV2:
    """自顶向下推断引擎（强化版）。

    核心改进：
    1. 自动检测 v1/v2 模型并调用对应的前向传播
    2. 增加预测置信度计算（基于精度矩阵）
    """

    def predict(
        self,
        model: Union[HierarchicalGenerativeModel, HierarchicalGenerativeModelV2],
        bottom_up: BottomUpPropagationV2,
        top_representation: np.ndarray
    ) -> TopDownPropagationV2:
        """执行自顶向下预测传播。"""
        if top_representation.shape[0] != model.layer_dims[-1]:
            raise ValueError(
                f"顶层表征维度 {top_representation.shape[0]} 必须等于 {model.layer_dims[-1]}"
            )
        predictions, obs_prediction = self._cascade_predict(model, top_representation)
        precision_weighted = []
        updated = []
        confidences = []
        deltas = bottom_up.propagated_signals
        for l in range(model.num_layers - 1):
            if isinstance(model, HierarchicalGenerativeModelV2):
                pi = model.precisions[l + 1]  # v2: precision 对应 child dim
            else:
                pi = model.precisions[l]
            pw = pi @ predictions[l]
            precision_weighted.append(pw)
            try:
                pi_inv = np.linalg.inv(pi)
            except np.linalg.LinAlgError:
                pi_inv = np.linalg.pinv(pi)
            upd = predictions[l] + pi_inv @ deltas[l]
            updated.append(upd)
            # 预测置信度 = trace(precision) / dim
            confidences.append(float(np.trace(pi)) / pi.shape[0])
        return TopDownPropagationV2(
            bottom_up=bottom_up,
            predictions=predictions,
            precision_weighted_predictions=tuple(precision_weighted),
            obs_prediction=obs_prediction,
            updated_representations=tuple(updated),
            prediction_confidences=tuple(confidences),
        )

    def _cascade_predict(
        self,
        model: Union[HierarchicalGenerativeModel, HierarchicalGenerativeModelV2],
        top_representation: np.ndarray
    ) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
        """级联预测。"""
        L = model.num_layers
        predictions = []
        r_current = top_representation
        for l in range(L - 1, 0, -1):
            if isinstance(model, HierarchicalGenerativeModelV2):
                r_hat = model.generative_fns[l].forward(r_current)
            else:
                r_hat = model.generative_fns[l](r_current, model.weights[l])
            predictions.insert(0, r_hat)
            r_current = r_hat
        if isinstance(model, HierarchicalGenerativeModelV2):
            obs_prediction = model.obs_generative_fn.forward(predictions[0]) if predictions else model.obs_generative_fn.forward(top_representation)
        else:
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
