"""阶段11强化版：真正的层级生成模型（支持非线性激活与自动微分）。

本文件是 stage_011.py 的强化升级版，核心改进：
1. 生成函数从纯线性（W @ r）升级为支持多种非线性激活
2. 增加数值Jacobian和Hessian计算（为自动微分做准备）
3. 增加激活函数注册表，支持 ReLU/Tanh/Sigmoid/Softplus/Swish
4. 增加前向传播缓存，支持反向传播
5. 保持与 stage_011.py 的完全向后兼容
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Callable, Tuple
from enum import Enum, auto
import numpy as np


# ============================================================
# 阶段11-v2 枚举：激活函数类型
# ============================================================
class ActivationType(Enum):
    """支持的激活函数类型。"""
    LINEAR = "linear"      # f(x) = x
    RELU = "relu"          # f(x) = max(0, x)
    TANH = "tanh"          # f(x) = tanh(x)
    SIGMOID = "sigmoid"    # f(x) = 1/(1+exp(-x))
    SOFTPLUS = "softplus"  # f(x) = ln(1+exp(x))
    SWISH = "swish"        # f(x) = x * sigmoid(x)


# ============================================================
# 阶段11-v2 激活函数注册表
# ============================================================
class ActivationRegistry:
    """激活函数注册表，统一管理前向/导数/二阶导。"""

    _registry: Dict[ActivationType, Dict[str, Callable]] = {}

    @classmethod
    def register(cls, act_type: ActivationType, fn: Callable, dfn: Callable, d2fn: Callable):
        cls._registry[act_type] = {"fn": fn, "dfn": dfn, "d2fn": d2fn}

    @classmethod
    def get(cls, act_type: ActivationType) -> Dict[str, Callable]:
        return cls._registry.get(act_type, cls._registry[ActivationType.LINEAR])


# 注册所有激活函数
ActivationRegistry.register(
    ActivationType.LINEAR,
    lambda x: x,
    lambda x: np.ones_like(x),
    lambda x: np.zeros_like(x)
)
ActivationRegistry.register(
    ActivationType.RELU,
    lambda x: np.maximum(0.0, x),
    lambda x: (x > 0).astype(float),
    lambda x: np.zeros_like(x)
)
ActivationRegistry.register(
    ActivationType.TANH,
    lambda x: np.tanh(x),
    lambda x: 1.0 - np.tanh(x) ** 2,
    lambda x: -2.0 * np.tanh(x) * (1.0 - np.tanh(x) ** 2)
)
ActivationRegistry.register(
    ActivationType.SIGMOID,
    lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
    lambda x: {
        "sig": (s := 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))),
        "result": s * (1.0 - s)
    }["result"],
    lambda x: {
        "sig": (s := 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))),
        "result": s * (1.0 - s) * (1.0 - 2.0 * s)
    }["result"]
)
ActivationRegistry.register(
    ActivationType.SOFTPLUS,
    lambda x: np.log1p(np.exp(np.clip(x, -500, 500))),
    lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
    lambda x: {
        "sig": (s := 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))),
        "result": s * (1.0 - s)
    }["result"]
)
ActivationRegistry.register(
    ActivationType.SWISH,
    lambda x: {
        "sig": (s := 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))),
        "result": x * s
    }["result"],
    lambda x: {
        "sig": (s := 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))),
        "result": s + x * s * (1.0 - s)
    }["result"],
    lambda x: {
        "sig": (s := 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))),
        "ds": s * (1.0 - s),
        "result": 2.0 * s * (1.0 - s) + x * s * (1.0 - s) * (1.0 - 2.0 * s)
    }["result"]
)


# ============================================================
# 阶段11-v2 数据类：非线性生成函数
# ============================================================
@dataclass(frozen=True)
class NonlinearGenerativeFunction:
    """非线性生成函数：f(r, W) = σ(W @ r + b)。

    相比 stage_011 的纯线性 f(r, W) = W @ r，增加了：
    - 偏置项 b
    - 非线性激活 σ
    - 数值Jacobian计算
    - 数值Hessian计算

    Attributes:
        weight: 权重矩阵 W
        bias: 偏置向量 b
        activation: 激活函数类型
        epsilon: 数值微分步长
    """
    weight: np.ndarray
    bias: np.ndarray
    activation: ActivationType = ActivationType.LINEAR
    epsilon: float = 1e-5

    def __post_init__(self):
        if self.epsilon <= 0:
            raise ValueError(f"epsilon 必须 > 0，收到 {self.epsilon}")

    def forward(self, r: np.ndarray) -> np.ndarray:
        """前向传播：σ(W @ r + b)。"""
        linear = self.weight @ r + self.bias
        act_fn = ActivationRegistry.get(self.activation)["fn"]
        return act_fn(linear)

    def jacobian(self, r: np.ndarray) -> np.ndarray:
        """数值计算Jacobian：J = ∂f/∂r。

        形状: (n_out, n_in)
        """
        n_in = r.shape[0]
        n_out = self.bias.shape[0]
        J = np.zeros((n_out, n_in))
        f0 = self.forward(r)
        for i in range(n_in):
            r_perturbed = r.copy()
            r_perturbed[i] += self.epsilon
            f_perturbed = self.forward(r_perturbed)
            J[:, i] = (f_perturbed - f0) / self.epsilon
        return J

    def jacobian_analytical(self, r: np.ndarray) -> np.ndarray:
        """解析计算Jacobian（当激活函数导数已知时）。

        J = diag(σ'(W @ r + b)) @ W
        """
        linear = self.weight @ r + self.bias
        dfn = ActivationRegistry.get(self.activation)["dfn"]
        sigma_prime = dfn(linear)
        return np.diag(sigma_prime) @ self.weight

    def hessian(self, r: np.ndarray) -> np.ndarray:
        """数值计算Hessian：H[i,j,k] = ∂²f_i / ∂r_j ∂r_k。

        形状: (n_out, n_in, n_in)
        """
        n_in = r.shape[0]
        n_out = self.bias.shape[0]
        H = np.zeros((n_out, n_in, n_in))
        J0 = self.jacobian(r)
        for j in range(n_in):
            r_perturbed = r.copy()
            r_perturbed[j] += self.epsilon
            J_perturbed = self.jacobian(r_perturbed)
            H[:, :, j] = (J_perturbed - J0) / self.epsilon
        return H

    def __call__(self, r: np.ndarray, weight: Optional[np.ndarray] = None) -> np.ndarray:
        """兼容 stage_011 的调用签名 f(r, W)。"""
        if weight is not None and not np.allclose(weight, self.weight):
            # 如果传入的W与当前不同，创建临时函数
            tmp = NonlinearGenerativeFunction(
                weight=weight, bias=self.bias,
                activation=self.activation, epsilon=self.epsilon
            )
            return tmp.forward(r)
        return self.forward(r)


# ============================================================
# 阶段11-v2 数据类：前向传播缓存
# ============================================================
@dataclass(frozen=True)
class ForwardCache:
    """前向传播缓存，用于反向传播。

    Attributes:
        linear_outputs: 各层线性输出 (W @ r + b)
        activated_outputs: 各层激活后输出 σ(W @ r + b)
        representations: 各层表征
    """
    linear_outputs: Tuple[np.ndarray, ...]
    activated_outputs: Tuple[np.ndarray, ...]
    representations: Tuple[np.ndarray, ...]


# ============================================================
# 阶段11-v2 核心接口：非线性层级生成模型构建器
# ============================================================
class NonlinearHierarchicalModelBuilder:
    """非线性层级生成模型构建器（stage_011 的强化版）。

    核心改进：
    1. 支持非线性激活函数
    2. 支持偏置项
    3. 提供数值/解析 Jacobian
    4. 提供前向传播缓存
    5. 保持与 HierarchicalModelBuilder 的 API 兼容
    """

    def build_model(
        self,
        num_layers: int,
        layer_dims: tuple[int, ...],
        obs_dim: int,
        weight_init_fn: callable,
        precision_init_fn: callable,
        bias_init_fn: Optional[callable] = None,
        activations: Optional[tuple[ActivationType, ...]] = None,
        epsilon: float = 1e-5
    ) -> "HierarchicalGenerativeModelV2":
        """构建非线性层级生成模型。

        Args:
            num_layers: 层级深度 L
            layer_dims: 各层表征维度 (n_1, ..., n_L)
            obs_dim: 观测维度 d_o
            weight_init_fn: 权重初始化函数 fn(in_dim, out_dim) -> ndarray
            precision_init_fn: 精度初始化函数 fn(dim) -> ndarray
            bias_init_fn: 偏置初始化函数 fn(dim) -> ndarray，默认零初始化
            activations: 各层激活函数类型，默认全线性
            epsilon: 数值微分步长

        Returns:
            HierarchicalGenerativeModelV2 实例
        """
        if num_layers < 1:
            raise ValueError(f"num_layers 必须 >= 1，收到 {num_layers}")
        if len(layer_dims) != num_layers:
            raise ValueError("layer_dims 长度必须等于 num_layers")
        if obs_dim < 1:
            raise ValueError(f"obs_dim 必须 >= 1，收到 {obs_dim}")
        if activations is not None and len(activations) != num_layers:
            raise ValueError("activations 长度必须等于 num_layers")

        if bias_init_fn is None:
            bias_init_fn = lambda dim: np.zeros(dim)
        if activations is None:
            activations = tuple(ActivationType.LINEAR for _ in range(num_layers))

        weights = tuple(
            weight_init_fn(layer_dims[i - 1] if i > 0 else obs_dim, layer_dims[i])
            for i in range(num_layers)
        )
        # bias 维度 = child 维度（输出维度）
        # l=0: child=obs_dim, l>0: child=layer_dims[l-1]
        bias_dims = [obs_dim] + list(layer_dims[:-1])
        biases = tuple(bias_init_fn(dim) for dim in bias_dims)
        # precision 维度同样 = child 维度
        precision_dims = [obs_dim] + list(layer_dims[:-1])
        precisions = tuple(precision_init_fn(dim) for dim in precision_dims)
        obs_precision = precision_init_fn(obs_dim)
        prior_mean = np.zeros(layer_dims[-1])
        prior_precision = precision_init_fn(layer_dims[-1])

        # 构建非线性生成函数
        gen_fns = tuple(
            NonlinearGenerativeFunction(
                weight=weights[i],
                bias=biases[i],
                activation=activations[i],
                epsilon=epsilon
            )
            for i in range(num_layers)
        )

        # 观测生成函数（第一层）
        obs_gen_fn = NonlinearGenerativeFunction(
            weight=weights[0],
            bias=biases[0],
            activation=activations[0],
            epsilon=epsilon
        )

        return HierarchicalGenerativeModelV2(
            num_layers=num_layers,
            layer_dims=layer_dims,
            obs_dim=obs_dim,
            weights=weights,
            biases=biases,
            precisions=precisions,
            obs_precision=obs_precision,
            prior_mean=prior_mean,
            prior_precision=prior_precision,
            generative_fns=gen_fns,
            obs_generative_fn=obs_gen_fn,
            activations=activations,
            epsilon=epsilon,
        )


# ============================================================
# 阶段11-v2 数据类：强化版层级生成模型
# ============================================================
@dataclass(frozen=True)
class HierarchicalGenerativeModelV2:
    """强化版层级生成模型（支持非线性激活）。

    完全兼容 stage_011.HierarchicalGenerativeModel 的接口，
    同时增加非线性激活、偏置、Jacobian计算等能力。
    """
    num_layers: int
    layer_dims: tuple[int, ...]
    obs_dim: int
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    precisions: tuple[np.ndarray, ...]
    obs_precision: np.ndarray
    prior_mean: np.ndarray
    prior_precision: np.ndarray
    generative_fns: tuple[NonlinearGenerativeFunction, ...]
    obs_generative_fn: NonlinearGenerativeFunction
    activations: tuple[ActivationType, ...]
    epsilon: float = 1e-5

    def __post_init__(self):
        if self.num_layers < 1:
            raise ValueError(f"层级深度 L 必须 ≥ 1，收到 {self.num_layers}")
        if len(self.layer_dims) != self.num_layers:
            raise ValueError("layer_dims 长度必须等于 num_layers")
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

    @property
    def covariances(self) -> tuple[np.ndarray, ...]:
        """各层条件协方差矩阵 Σ^l = Π_l^{-1}。"""
        return tuple(np.linalg.inv(pi) for pi in self.precisions)

    @property
    def obs_covariance(self) -> np.ndarray:
        """观测协方差矩阵 Σ^o = Π_o^{-1}。"""
        return np.linalg.inv(self.obs_precision)

    def forward_with_cache(
        self,
        representations: tuple[np.ndarray, ...]
    ) -> ForwardCache:
        """带缓存的前向传播。

        计算各层的线性输出和激活后输出，用于反向传播。
        """
        linear_outputs = []
        activated_outputs = []
        for l in range(self.num_layers):
            linear = self.weights[l] @ representations[l] + self.biases[l]
            linear_outputs.append(linear)
            act_fn = ActivationRegistry.get(self.activations[l])["fn"]
            activated_outputs.append(act_fn(linear))
        return ForwardCache(
            linear_outputs=tuple(linear_outputs),
            activated_outputs=tuple(activated_outputs),
            representations=representations
        )

    def joint_log_likelihood(
        self,
        observation: np.ndarray,
        representations: tuple[np.ndarray, ...]
    ) -> float:
        """计算联合对数似然 ln p(o, r^{1:L})。"""
        if len(representations) != self.num_layers:
            raise ValueError("表征数量必须等于 num_layers")
        if observation.shape[0] != self.obs_dim:
            raise ValueError("观测维度不匹配")

        # 观测对数似然: p(o | r^1)
        obs_pred = self.obs_generative_fn.forward(representations[0])
        obs_error = observation - obs_pred
        log_likelihood = -0.5 * float(obs_error.T @ self.obs_precision @ obs_error)

        # 各层条件对数似然: p(r^{l-1} | r^l) for l=1..L-1
        # 注意：l=0 已由上面的观测对数似然处理（obs_generative_fn == generative_fns[0]）
        for l in range(1, self.num_layers):
            child = representations[l - 1]
            parent = representations[l]
            pred = self.generative_fns[l].forward(parent)
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

    def compute_free_energy(self, observation, representations) -> float:
        """计算变分自由能 F = -ln p(o, r)。"""
        return -self.joint_log_likelihood(observation, representations)

    def compute_jacobians(
        self,
        representations: tuple[np.ndarray, ...]
    ) -> tuple[np.ndarray, ...]:
        """计算各层解析Jacobian。

        Returns:
            (J^1, J^2, ..., J^L)，其中 J^l = ∂f^l/∂r^l
        """
        return tuple(
            self.generative_fns[l].jacobian_analytical(representations[l])
            for l in range(self.num_layers)
        )

    def top_down_predict(self, parent_repr: np.ndarray, layer_index: int) -> np.ndarray:
        """计算自上而下预测。"""
        if not (1 <= layer_index <= self.num_layers):
            raise ValueError(f"layer_index 超出范围")
        return self.generative_fns[layer_index - 1].forward(parent_repr)

    def to_v1_model(self) -> "HierarchicalGenerativeModel":
        """转换为 stage_011 的兼容格式（线性近似）。"""
        from .stage_011 import HierarchicalGenerativeModel
        # 创建线性兼容的生成函数
        def make_gen_fn(W, b, act):
            return lambda r, W=W: W @ r
        gen_fns = tuple(
            make_gen_fn(self.weights[i], self.biases[i], self.activations[i])
            for i in range(self.num_layers)
        )
        obs_gen_fn = make_gen_fn(self.weights[0], self.biases[0], self.activations[0])
        return HierarchicalGenerativeModel(
            num_layers=self.num_layers,
            layer_dims=self.layer_dims,
            obs_dim=self.obs_dim,
            weights=self.weights,
            precisions=self.precisions,
            obs_precision=self.obs_precision,
            prior_mean=self.prior_mean,
            prior_precision=self.prior_precision,
            generative_fns=tuple(gen_fns),
            obs_generative_fn=obs_gen_fn,
        )
