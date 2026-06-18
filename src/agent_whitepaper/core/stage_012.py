"""Stage 12: Bottom-Up Error Propagation.

自底向上误差传播，计算精度加权误差和Jacobian。
"""

from dataclasses import dataclass
import numpy as np

from agent_whitepaper.core.stage_011 import HierarchicalGenerativeModel


# ============================================================
# Stage 12 Data Class: Bottom-Up Propagation
# ============================================================
@dataclass(frozen=True)
class BottomUpPropagation:
    """自底向上传播结果。

    Attributes:
        model: 层级生成模型
        error_signals: 各层误差信号 (e^0, e^1, ..., e^{L-1})
        propagated_signals: 传播信号 (δ^1, δ^2, ..., δ^L)
        jacobians: Jacobian矩阵 (J^1, J^2, ..., J^{L-1})
        precision_weighted_errors: 精度加权误差 (Π_1 e^0, ..., Π_L e^{L-1})
        total_complexity: 总复杂度估计
    """
    model: HierarchicalGenerativeModel
    error_signals: tuple[np.ndarray, ...]
    propagated_signals: tuple[np.ndarray, ...]
    jacobians: tuple[np.ndarray, ...]
    precision_weighted_errors: tuple[np.ndarray, ...]
    total_complexity: int


# ============================================================
# Stage 12 Core Interface: Bottom-Up Inference Engine
# ============================================================
class BottomUpInferenceEngine:
    """自底向上推断引擎。"""

    def compute_errors(
        self,
        model: HierarchicalGenerativeModel,
        observation: np.ndarray,
        representations: tuple[np.ndarray, ...]
    ) -> tuple[np.ndarray, ...]:
        """计算各层误差信号 e^{l-1} = r^{l-1} - f^l(r^l, W^l)。

        返回 L+1 个误差信号: [e^0(obs), e^1, ..., e^{L-1}, e^L(prior)]
        """
        if len(representations) != model.num_layers:
            raise ValueError(
                f"表征数量 {len(representations)} 必须等于 {model.num_layers}"
            )
        errors = []
        # 观测层误差 e^0 = o - g(r^1)
        obs_pred = model.obs_generative_fn(representations[0])
        errors.append(observation - obs_pred)
        # 各层误差 e^l = r^l - f^{l+1}(r^{l+1}, W^{l+1})
        for l in range(model.num_layers - 1):
            pred = model.generative_fns[l + 1](representations[l + 1], model.weights[l + 1])
            errors.append(representations[l] - pred)
        # 顶层先验误差 e^L = r^L - μ_prior
        errors.append(representations[-1] - model.prior_mean)
        return tuple(errors)

    def compute_jacobian(
        self,
        model: HierarchicalGenerativeModel,
        layer_index: int,
        representation: np.ndarray
    ) -> np.ndarray:
        """数值计算Jacobian矩阵 J^l = ∂f^l/∂r^l。

        f^l: r^l -> r^{l-1}, 所以 J^l 的形状是 (n_{l-1}, n_l)
        """
        if not (1 <= layer_index <= model.num_layers - 1):
            raise ValueError(
                f"layer_index {layer_index} 超出范围 [1, {model.num_layers - 1}]"
            )
        n_lm1 = model.layer_dims[layer_index - 1]  # 输出维度 n_{l-1}
        n_l = model.layer_dims[layer_index]        # 输入维度 n_l
        f = model.generative_fns[layer_index]
        eps = 1e-5
        J = np.zeros((n_lm1, n_l))
        for i in range(n_l):
            r_plus = representation.copy()
            r_plus[i] += eps
            r_minus = representation.copy()
            r_minus[i] -= eps
            J[:, i] = (f(r_plus, model.weights[layer_index]) - f(r_minus, model.weights[layer_index])) / (2 * eps)
        return J

    def propagate(
        self,
        model: HierarchicalGenerativeModel,
        observation: np.ndarray,
        representations: tuple[np.ndarray, ...]
    ) -> BottomUpPropagation:
        """执行完整的自底向上传播。"""
        error_signals = self.compute_errors(model, observation, representations)
        L = model.num_layers
        # error_signals 有 L+1 个元素: [e^0(obs), e^1, ..., e^{L-1}]
        # 我们只取层内误差 e^1 到 e^{L-1}，以及观测误差 e^0 用于第一层
        # 从顶层到底层递归计算传播信号
        jacobians = []
        propagated = [None] * L
        precision_weighted = []
        # 顶层初始化 δ^L = Π_L e^{L-1} (error_signals 索引为 L)
        propagated[L - 1] = model.precisions[L - 1] @ error_signals[L]
        precision_weighted.append(model.precisions[L - 1] @ error_signals[L])
        # 从高层到低层递归
        for l in range(L - 2, -1, -1):
            # J^{l+1} = ∂f^{l+1}/∂r^{l+1}，需要 r^{l+1} 作为输入点
            J = self.compute_jacobian(model, l + 1, representations[l + 1])
            jacobians.insert(0, J)
            propagated[l] = model.precisions[l] @ error_signals[l + 1] + J @ propagated[l + 1]
            precision_weighted.insert(0, model.precisions[l] @ error_signals[l + 1])
        total_complexity = L * max(model.layer_dims) ** 2 if model.layer_dims else 1
        return BottomUpPropagation(
            model=model,
            error_signals=tuple(error_signals),
            propagated_signals=tuple(propagated),
            jacobians=tuple(jacobians) if jacobians else (np.zeros((1, 1)),),
            precision_weighted_errors=tuple(precision_weighted),
            total_complexity=total_complexity,
        )

    def check_stability(self, propagation: BottomUpPropagation) -> dict:
        """检查传播稳定性。"""
        spectral_norms = []
        layer_stability = []
        for J in propagation.jacobians:
            s_norm = float(np.linalg.norm(J, ord=2))
            spectral_norms.append(s_norm)
            layer_stability.append(s_norm <= 1.0 + 1e-8)
        return {
            "is_stable": all(layer_stability),
            "layer_stability": layer_stability,
            "spectral_norms": spectral_norms,
        }
