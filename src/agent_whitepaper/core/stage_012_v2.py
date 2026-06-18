"""阶段12强化版：自底向上误差传播（支持非线性激活与解析Jacobian）。

本文件是 stage_012.py 的强化升级版，核心改进：
1. 支持 HierarchicalGenerativeModelV2（非线性激活）
2. 利用解析Jacobian替代数值微分（O(1) vs O(n)）
3. 增加误差信号的能量分析
4. 保持与 stage_012.py 的完全向后兼容
"""

from dataclasses import dataclass
from typing import Union
import numpy as np

from agent_whitepaper.core.stage_011 import HierarchicalGenerativeModel
from agent_whitepaper.core.stage_011_v2 import HierarchicalGenerativeModelV2


# ============================================================
# 阶段12-v2 数据类：自底向上传播
# ============================================================
@dataclass(frozen=True)
class BottomUpPropagationV2:
    """自底向上传播结果（强化版）。

    Attributes:
        model: 层级生成模型（v1 或 v2）
        error_signals: 各层误差信号 (e^0, e^1, ..., e^L)
        propagated_signals: 传播信号 (δ^1, δ^2, ..., δ^L)
        jacobians: Jacobian矩阵 (J^1, J^2, ..., J^{L-1})
        precision_weighted_errors: 精度加权误差
        total_complexity: 总复杂度估计
        error_energies: 各层误差能量 ||e^l||^2
    """
    model: Union[HierarchicalGenerativeModel, HierarchicalGenerativeModelV2]
    error_signals: tuple[np.ndarray, ...]
    propagated_signals: tuple[np.ndarray, ...]
    jacobians: tuple[np.ndarray, ...]
    precision_weighted_errors: tuple[np.ndarray, ...]
    total_complexity: int
    error_energies: tuple[float, ...]


# ============================================================
# 阶段12-v2 核心接口：自底向上推断引擎
# ============================================================
class BottomUpInferenceEngineV2:
    """自底向上推断引擎（强化版）。

    核心改进：
    1. 自动检测 v1/v2 模型并调用对应的 Jacobian 计算
    2. v2 模型使用解析 Jacobian（O(1)），v1 模型回退到数值微分
    3. 增加误差能量分析
    """

    def compute_errors(
        self,
        model: Union[HierarchicalGenerativeModel, HierarchicalGenerativeModelV2],
        observation: np.ndarray,
        representations: tuple[np.ndarray, ...]
    ) -> tuple[np.ndarray, ...]:
        """计算各层误差信号 e^{l-1} = child - f^l(parent)。

        返回 L+1 个误差信号: [e^0(obs), e^1, ..., e^{L-1}, e^L(prior)]
        """
        if len(representations) != model.num_layers:
            raise ValueError(
                f"表征数量 {len(representations)} 必须等于 {model.num_layers}"
            )
        errors = []
        # 观测层误差 e^0 = o - g(r^1)
        if isinstance(model, HierarchicalGenerativeModelV2):
            obs_pred = model.obs_generative_fn.forward(representations[0])
        else:
            obs_pred = model.obs_generative_fn(representations[0])
        errors.append(observation - obs_pred)
        # 各层误差 e^l = r^l - f^{l+1}(r^{l+1}, W^{l+1})
        for l in range(model.num_layers - 1):
            if isinstance(model, HierarchicalGenerativeModelV2):
                pred = model.generative_fns[l + 1].forward(representations[l + 1])
            else:
                pred = model.generative_fns[l + 1](representations[l + 1], model.weights[l + 1])
            errors.append(representations[l] - pred)
        # 顶层先验误差 e^L = r^L - μ_prior
        errors.append(representations[-1] - model.prior_mean)
        return tuple(errors)

    def compute_jacobian(
        self,
        model: Union[HierarchicalGenerativeModel, HierarchicalGenerativeModelV2],
        layer_index: int,
        representation: np.ndarray
    ) -> np.ndarray:
        """计算Jacobian矩阵 J^l = ∂f^l/∂r^l。

        对于 v2 模型使用解析 Jacobian（O(1)），
        对于 v1 模型使用数值微分（O(n)）。
        """
        if not (1 <= layer_index <= model.num_layers - 1):
            raise ValueError(
                f"layer_index {layer_index} 超出范围 [1, {model.num_layers - 1}]"
            )
        # v2 模型：使用解析 Jacobian
        if isinstance(model, HierarchicalGenerativeModelV2):
            return model.generative_fns[layer_index].jacobian_analytical(representation)
        # v1 模型：回退到数值微分
        n_lm1 = model.layer_dims[layer_index - 1]
        n_l = model.layer_dims[layer_index]
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
        model: Union[HierarchicalGenerativeModel, HierarchicalGenerativeModelV2],
        observation: np.ndarray,
        representations: tuple[np.ndarray, ...]
    ) -> BottomUpPropagationV2:
        """执行完整的自底向上传播。"""
        error_signals = self.compute_errors(model, observation, representations)
        L = model.num_layers
        # 计算误差能量
        error_energies = tuple(float(np.dot(e, e)) for e in error_signals)
        # 从顶层到底层递归计算传播信号
        jacobians = []
        propagated = [None] * L
        precision_weighted = []
        # 顶层初始化 δ^L = Π_L e^{L-1} (error_signals 索引为 L)
        # 注意：对于 v2 模型，precisions 维度已修正为 child_dim
        if isinstance(model, HierarchicalGenerativeModelV2):
            # v2: precisions[l] 对应第 l 层 child 的精度
            # 顶层先验使用 prior_precision
            propagated[L - 1] = model.prior_precision @ error_signals[L]
        else:
            propagated[L - 1] = model.precisions[L - 1] @ error_signals[L]
        precision_weighted.append(propagated[L - 1].copy())
        # 从高层到低层递归
        for l in range(L - 2, -1, -1):
            J = self.compute_jacobian(model, l + 1, representations[l + 1])
            jacobians.insert(0, J)
            if isinstance(model, HierarchicalGenerativeModelV2):
                # v2: precisions[l] 对应 child=layer_dims[l-1]（l>0）或 obs_dim（l=0）
                # 但 error_signals[l+1] 是第 l+1 层的误差（child 是 layer_dims[l]）
                # 所以 precision 应该用 precisions[l+1]
                pw = model.precisions[l + 1] @ error_signals[l + 1]
            else:
                pw = model.precisions[l] @ error_signals[l + 1]
            propagated[l] = pw + J @ propagated[l + 1]
            precision_weighted.insert(0, pw)
        total_complexity = L * max(model.layer_dims) ** 2 if model.layer_dims else 1
        return BottomUpPropagationV2(
            model=model,
            error_signals=tuple(error_signals),
            propagated_signals=tuple(propagated),
            jacobians=tuple(jacobians) if jacobians else (np.zeros((1, 1)),),
            precision_weighted_errors=tuple(precision_weighted),
            total_complexity=total_complexity,
            error_energies=error_energies,
        )

    def check_stability(self, propagation: BottomUpPropagationV2) -> dict:
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
