"""Stage 15: Weight Update (Hebbian Learning).

权重更新，通过Hebbian学习规则更新模型权重。
"""

from dataclasses import dataclass
import numpy as np

from agent_whitepaper.core.stage_014 import PerceptionUpdate


# ============================================================
# Stage 15 Data Class: Weight Update
# ============================================================
@dataclass(frozen=True)
class WeightUpdate:
    """权重更新结果。

    Attributes:
        perception: 感知更新结果
        weights_before: 更新前的权重
        weights_after: 更新后的权重
        weight_gradients: 权重梯度
        fisher_diagonals: Fisher信息对角
        anchor_weights: 锚定权重
        time_constant: 时间常数
        learning_rate: 学习率
        ewc_strength: EWC强度
        learning_rates_per_layer: 各层学习率
    """
    perception: PerceptionUpdate
    weights_before: tuple[np.ndarray, ...]
    weights_after: tuple[np.ndarray, ...]
    weight_gradients: tuple[np.ndarray, ...]
    fisher_diagonals: tuple[np.ndarray, ...]
    anchor_weights: tuple[np.ndarray, ...]
    time_constant: float
    learning_rate: float
    ewc_strength: float
    learning_rates_per_layer: tuple[float, ...]


# ============================================================
# Stage 15 Core Interface: Weight Update Engine
# ============================================================
class WeightUpdateEngine:
    """权重更新引擎。"""

    def update(
        self,
        perception: PerceptionUpdate,
        weights: tuple[np.ndarray, ...],
        fisher_diagonals: tuple[np.ndarray, ...],
        anchor_weights: tuple[np.ndarray, ...],
        time_constant: float = 100.0,
        learning_rate: float = 0.01,
        ewc_strength: float = 0.1
    ) -> WeightUpdate:
        """执行权重更新。"""
        if time_constant <= 0:
            raise ValueError(f"time_constant 必须 > 0，收到 {time_constant}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate 必须 > 0，收到 {learning_rate}")
        if ewc_strength < 0:
            raise ValueError(f"ewc_strength 必须 >= 0，收到 {ewc_strength}")

        model = perception.top_down.bottom_up.model
        L = model.num_layers
        weight_grads = []
        weights_after = []
        learning_rates = self._compute_adaptive_learning_rates(model, learning_rate)

        for l in range(L):
            # 使用子层误差信号计算权重梯度
            # W^l: r^l -> r^{l-1}, 梯度 ≈ e^{l-1} @ (r^l)^T
            child_error = perception.top_down.bottom_up.error_signals[l]
            parent_repr = perception.representations_after[l]
            wg = np.outer(child_error, parent_repr)
            weight_grads.append(wg)
            reg_grad = self._ewc_regularized_gradient(
                wg, fisher_diagonals[l], weights[l], anchor_weights[l], ewc_strength
            )
            delta_w = (learning_rates[l] / time_constant) * reg_grad
            weights_after.append(weights[l] - delta_w)

        return WeightUpdate(
            perception=perception,
            weights_before=weights,
            weights_after=tuple(weights_after),
            weight_gradients=tuple(weight_grads),
            fisher_diagonals=fisher_diagonals,
            anchor_weights=anchor_weights,
            time_constant=time_constant,
            learning_rate=learning_rate,
            ewc_strength=ewc_strength,
            learning_rates_per_layer=learning_rates,
        )

    def _compute_weight_gradient(
        self,
        model,
        layer_index: int,
        representation: np.ndarray,
        propagated_signal: np.ndarray,
        current_weight: np.ndarray
    ) -> np.ndarray:
        """计算权重梯度。

        对于线性生成函数 f(r, W) = W @ r，
        ∂F/∂W = -δ @ r^T，形状与 W 相同 (n_{l-1}, n_l)
        """
        return -np.outer(propagated_signal, representation)

    def _ewc_regularized_gradient(
        self,
        weight_gradient: np.ndarray,
        fisher_diagonal: np.ndarray,
        current_weight: np.ndarray,
        anchor_weight: np.ndarray,
        ewc_strength: float
    ) -> np.ndarray:
        """EWC正则化梯度。"""
        ewc_term = ewc_strength * fisher_diagonal * (current_weight - anchor_weight)
        return weight_gradient - ewc_term

    def _compute_adaptive_learning_rates(
        self,
        model,
        base_learning_rate: float
    ) -> tuple[float, ...]:
        """计算自适应学习率。"""
        rates = []
        for l in range(model.num_layers):
            pi_inv_trace = float(np.trace(np.linalg.inv(model.precisions[l])))
            # precisions[l] 的维度 = child dim（修复后）
            # l=0: child=obs_dim, l>0: child=layer_dims[l-1]
            dim = model.obs_dim if l == 0 else model.layer_dims[l - 1]
            rates.append(base_learning_rate * pi_inv_trace / dim)
        return tuple(rates)
