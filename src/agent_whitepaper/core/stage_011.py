"""Stage 11: Hierarchical Generative Model.

层级生成模型构建器，实现多层级预测编码的生成模型。
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum
import numpy as np


# ============================================================
# Stage 11 Enum: Layer Type
# ============================================================
class LayerType(Enum):
    """层级生成模型中各层的类型标识。"""
    OBSERVATION = "observation"
    FEATURE = "feature"
    SEMANTIC = "semantic"
    PRIOR = "prior"


# ============================================================
# Stage 11 Data Class: Hierarchical Generative Model
# ============================================================
@dataclass(frozen=True)
class HierarchicalGenerativeModel:
    """层级生成模型的完整描述。

    Attributes:
        num_layers: 层级深度 L
        layer_dims: 各层表征维度元组 (n_1, n_2, ..., n_L)
        obs_dim: 观测维度 d_o
        weights: 各层权重矩阵元组 (W^1, W^2, ..., W^L)
        precisions: 各层精度矩阵元组 (Π_1, Π_2, ..., Π_L)
        obs_precision: 观测精度矩阵 Π_o
        prior_mean: 顶层先验均值 μ_prior
        prior_precision: 顶层先验精度矩阵 Π_prior
        generative_fns: 各层生成函数元组 (f^1, f^2, ..., f^L)
        obs_generative_fn: 观测生成函数 g
    """
    num_layers: int
    layer_dims: tuple[int, ...]
    obs_dim: int
    weights: tuple[np.ndarray, ...]
    precisions: tuple[np.ndarray, ...]
    obs_precision: np.ndarray
    prior_mean: np.ndarray
    prior_precision: np.ndarray
    generative_fns: tuple[callable, ...]
    obs_generative_fn: callable

    def __post_init__(self):
        if self.num_layers < 1:
            raise ValueError(f"层级深度 L 必须 ≥ 1，收到 {self.num_layers}")
        if len(self.layer_dims) != self.num_layers:
            raise ValueError(
                f"layer_dims 长度 {len(self.layer_dims)} 必须等于 num_layers {self.num_layers}"
            )
        if len(self.weights) != self.num_layers:
            raise ValueError(
                f"权重数量 {len(self.weights)} 必须等于 num_layers {self.num_layers}"
            )
        if len(self.precisions) != self.num_layers:
            raise ValueError(
                f"精度矩阵数量 {len(self.precisions)} 必须等于 num_layers {self.num_layers}"
            )
        if len(self.generative_fns) != self.num_layers:
            raise ValueError(
                f"生成函数数量 {len(self.generative_fns)} 必须等于 num_layers {self.num_layers}"
            )
        for l, pi_l in enumerate(self.precisions):
            if not np.allclose(pi_l, pi_l.T):
                raise ValueError(f"第 {l+1} 层精度矩阵必须对称")
            try:
                np.linalg.cholesky(pi_l)
            except np.linalg.LinAlgError:
                raise ValueError(f"第 {l+1} 层精度矩阵必须正定")
        if not np.allclose(self.obs_precision, self.obs_precision.T):
            raise ValueError("观测精度矩阵必须对称")
        try:
            np.linalg.cholesky(self.obs_precision)
        except np.linalg.LinAlgError:
            raise ValueError("观测精度矩阵必须正定")
        if not np.allclose(self.prior_precision, self.prior_precision.T):
            raise ValueError("先验精度矩阵必须对称")
        try:
            np.linalg.cholesky(self.prior_precision)
        except np.linalg.LinAlgError:
            raise ValueError("先验精度矩阵必须正定")

    @property
    def covariances(self) -> tuple[np.ndarray, ...]:
        """各层条件协方差矩阵 Σ^l = Π_l^{-1}。"""
        return tuple(np.linalg.inv(pi) for pi in self.precisions)

    @property
    def obs_covariance(self) -> np.ndarray:
        """观测协方差矩阵 Σ^o = Π_o^{-1}。"""
        return np.linalg.inv(self.obs_precision)

    def joint_log_likelihood(
        self,
        observation: np.ndarray,
        representations: tuple[np.ndarray, ...]
    ) -> float:
        """计算联合对数似然 ln p(o, r^{1:L})。"""
        if len(representations) != self.num_layers:
            raise ValueError(
                f"表征数量 {len(representations)} 必须等于 num_layers {self.num_layers}"
            )
        if observation.shape[0] != self.obs_dim:
            raise ValueError(
                f"观测维度 {observation.shape[0]} 必须等于 obs_dim {self.obs_dim}"
            )
        # 观测对数似然
        obs_pred = self.obs_generative_fn(representations[0])
        obs_error = observation - obs_pred
        log_likelihood = -0.5 * float(obs_error.T @ self.obs_precision @ obs_error)
        # 各层条件对数似然
        for l in range(self.num_layers):
            child = representations[l - 1] if l > 0 else observation
            parent = representations[l]
            pred = self.generative_fns[l](parent, self.weights[l])
            error = child - pred
            precision = self.precisions[l]
            layer_log_lik = -0.5 * float(error.T @ precision @ error)
            log_likelihood += layer_log_lik
        # 顶层先验
        top_repr = representations[-1]
        prior_error = top_repr - self.prior_mean
        prior_log_lik = -0.5 * float(prior_error.T @ self.prior_precision @ prior_error)
        log_likelihood += prior_log_lik
        return float(log_likelihood)

    def conditional_log_likelihood(
        self,
        child: np.ndarray,
        parent: np.ndarray,
        layer_index: int
    ) -> float:
        """计算条件对数似然 ln p(r^{l-1}|r^l)。"""
        if not (1 <= layer_index <= self.num_layers):
            raise ValueError(
                f"layer_index {layer_index} 超出范围 [1, {self.num_layers}]"
            )
        pred = self.generative_fns[layer_index - 1](parent, self.weights[layer_index - 1])
        error = child - pred
        precision = self.precisions[layer_index - 1]
        log_lik = -0.5 * float(error.T @ precision @ error)
        return float(log_lik)

    def top_down_predict(
        self,
        parent_repr: np.ndarray,
        layer_index: int
    ) -> np.ndarray:
        """计算自上而下预测 r̂^{l-1} = f^l(r^l, W^l)。"""
        if not (1 <= layer_index <= self.num_layers):
            raise ValueError(
                f"layer_index {layer_index} 超出范围 [1, {self.num_layers}]"
            )
        return self.generative_fns[layer_index - 1](parent_repr, self.weights[layer_index - 1])

    def compute_abstraction_ordering(self) -> list[int]:
        """计算抽象度排序。"""
        return list(range(1, self.num_layers + 1))


# ============================================================
# Stage 11 Core Interface: Hierarchical Model Builder
# ============================================================
class HierarchicalModelBuilder:
    """层级生成模型构建器。"""

    def build_model(
        self,
        num_layers: int,
        layer_dims: tuple[int, ...],
        obs_dim: int,
        weight_init_fn: callable,
        precision_init_fn: callable
    ) -> HierarchicalGenerativeModel:
        """构建层级生成模型。"""
        if num_layers < 1:
            raise ValueError(f"num_layers 必须 >= 1，收到 {num_layers}")
        if len(layer_dims) != num_layers:
            raise ValueError("layer_dims 长度必须等于 num_layers")
        if obs_dim < 1:
            raise ValueError(f"obs_dim 必须 >= 1，收到 {obs_dim}")
        weights = tuple(
            weight_init_fn(layer_dims[i - 1] if i > 0 else obs_dim, layer_dims[i])
            for i in range(num_layers)
        )
        precisions = tuple(precision_init_fn(dim) for dim in layer_dims)
        obs_precision = precision_init_fn(obs_dim)
        prior_mean = np.zeros(layer_dims[-1])
        prior_precision = precision_init_fn(layer_dims[-1])
        def make_gen_fn(W):
            return lambda r, W=W: W @ r
        generative_fns = tuple(make_gen_fn(weights[i]) for i in range(num_layers))
        obs_generative_fn = lambda r, W=weights[0]: W @ r
        return HierarchicalGenerativeModel(
            num_layers=num_layers,
            layer_dims=layer_dims,
            obs_dim=obs_dim,
            weights=weights,
            precisions=precisions,
            obs_precision=obs_precision,
            prior_mean=prior_mean,
            prior_precision=prior_precision,
            generative_fns=generative_fns,
            obs_generative_fn=obs_generative_fn,
        )

    def compute_free_energy_from_model(
        self,
        model: HierarchicalGenerativeModel,
        observation: np.ndarray,
        representations: tuple[np.ndarray, ...]
    ) -> float:
        """从层级生成模型计算自由能。"""
        return -model.joint_log_likelihood(observation, representations)

    def optimal_depth(
        self,
        model: HierarchicalGenerativeModel,
        free_energy_budget: float,
        max_depth: int
    ) -> int:
        """计算最优层级深度。"""
        if free_energy_budget <= 0:
            raise ValueError(f"free_energy_budget 必须 > 0，收到 {free_energy_budget}")
        if max_depth < 1:
            raise ValueError(f"max_depth 必须 >= 1，收到 {max_depth}")
        return min(model.num_layers, max_depth)
