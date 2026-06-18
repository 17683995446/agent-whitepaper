"""阶段50强化版：真正的多层预测模型。

本文件是 stage_050.py 的强化升级版，核心改进：
1. 实现真正的多层神经网络前向传播
2. 实现反向传播计算梯度
3. 实现基于梯度的权重更新（SGD + Momentum）
4. 集成 v2 非线性生成模型
5. 保持与 CrossLayerProtocol 的兼容
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np

from .stage_011_v2 import (
    ActivationType, ActivationRegistry,
    NonlinearGenerativeFunction, HierarchicalGenerativeModelV2
)


# ============================================================
# 阶段50-v2 数据类：层参数
# ============================================================
@dataclass(frozen=True)
class LayerParams:
    """单层神经网络参数。

    Attributes:
        weight: 权重矩阵 W (out_dim, in_dim)
        bias: 偏置向量 b (out_dim,)
        activation: 激活函数类型
        velocity_w: 权重动量（用于Momentum SGD）
        velocity_b: 偏置动量
    """
    weight: np.ndarray
    bias: np.ndarray
    activation: ActivationType
    velocity_w: np.ndarray
    velocity_b: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播：σ(W @ x + b)。"""
        act_fn = ActivationRegistry.get(self.activation)["fn"]
        return act_fn(self.weight @ x + self.bias)

    def forward_linear(self, x: np.ndarray) -> np.ndarray:
        """线性部分（激活前）：W @ x + b。"""
        return self.weight @ x + self.bias

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """解析Jacobian：diag(σ'(z)) @ W。"""
        z = self.forward_linear(x)
        dfn = ActivationRegistry.get(self.activation)["dfn"]
        return np.diag(dfn(z)) @ self.weight


# ============================================================
# 阶段50-v2 数据类：多层预测模型
# ============================================================
@dataclass(frozen=True)
class MultiLayerPredictorV2:
    """真正的多层预测模型。

    实现完整的前向传播、反向传播和权重更新。

    Attributes:
        layers: 各层参数
        input_dim: 输入维度
        output_dim: 输出维度
        learning_rate: 学习率
        momentum: 动量系数
        loss_history: 损失历史

    Invariants:
        - 相邻层维度匹配
        - learning_rate > 0
        - 0 <= momentum < 1
    """
    layers: Tuple[LayerParams, ...]
    input_dim: int
    output_dim: int
    learning_rate: float
    momentum: float
    loss_history: Tuple[float, ...] = ()

    def __post_init__(self):
        if not self.layers:
            raise ValueError("至少需要一层")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate 必须 > 0，收到 {self.learning_rate}")
        if not (0 <= self.momentum < 1):
            raise ValueError(f"momentum 必须 ∈ [0,1)，收到 {self.momentum}")
        # 检查维度匹配
        for i, layer in enumerate(self.layers):
            if i == 0:
                if layer.weight.shape[1] != self.input_dim:
                    raise ValueError(
                        f"第0层输入维度 {layer.weight.shape[1]} != input_dim {self.input_dim}"
                    )
            else:
                if layer.weight.shape[1] != self.layers[i-1].weight.shape[0]:
                    raise ValueError(
                        f"第{i}层输入维度 {layer.weight.shape[1]} != "
                        f"第{i-1}层输出维度 {self.layers[i-1].weight.shape[0]}"
                    )
        if self.layers[-1].weight.shape[0] != self.output_dim:
            raise ValueError(
                f"最后层输出维度 {self.layers[-1].weight.shape[0]} != "
                f"output_dim {self.output_dim}"
            )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播 through all layers。"""
        h = x
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def forward_with_cache(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """带缓存的前向传播（用于反向传播）。

        Returns:
            (output, linear_outputs, activated_outputs)
        """
        h = x
        linear_outs = []
        activated_outs = [x]  # 输入作为第0个激活输出
        for layer in self.layers:
            z = layer.forward_linear(h)
            linear_outs.append(z)
            h = layer.forward(h)
            activated_outs.append(h)
        return h, linear_outs, activated_outs

    def backward(
        self,
        x: np.ndarray,
        target: np.ndarray,
        loss_type: str = "mse"
    ) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], float]:
        """反向传播计算梯度。

        Args:
            x: 输入
            target: 目标
            loss_type: 损失函数类型 ("mse" 或 "cross_entropy")

        Returns:
            (weight_grads, bias_grads, loss_value)
        """
        output, linear_outs, activated_outs = self.forward_with_cache(x)

        # 计算损失和输出层梯度
        if loss_type == "mse":
            loss = float(np.mean((output - target) ** 2))
            delta = 2 * (output - target) / output.shape[0]
        elif loss_type == "cross_entropy":
            # 假设output已经过sigmoid
            eps = 1e-10
            loss = -float(np.sum(target * np.log(output + eps) + (1 - target) * np.log(1 - output + eps)))
            delta = (output - target) / (output * (1 - output) + eps)
        else:
            raise ValueError(f"未知损失类型: {loss_type}")

        # 反向传播
        weight_grads = [None] * len(self.layers)
        bias_grads = [None] * len(self.layers)

        for l in range(len(self.layers) - 1, -1, -1):
            # 乘以激活函数导数
            dfn = ActivationRegistry.get(self.layers[l].activation)["dfn"]
            sigma_prime = dfn(linear_outs[l])
            delta = delta * sigma_prime

            # 计算梯度
            prev_output = activated_outs[l]
            weight_grads[l] = np.outer(delta, prev_output)
            bias_grads[l] = delta.copy()

            # 传播到上一层
            if l > 0:
                delta = self.layers[l].weight.T @ delta

        return tuple(weight_grads), tuple(bias_grads), loss

    def train_step(
        self,
        x: np.ndarray,
        target: np.ndarray,
        loss_type: str = "mse"
    ) -> "MultiLayerPredictorV2":
        """执行一步训练（SGD + Momentum）。

        Args:
            x: 输入
            target: 目标
            loss_type: 损失函数类型

        Returns:
            更新后的模型
        """
        w_grads, b_grads, loss = self.backward(x, target, loss_type)

        new_layers = []
        for l, layer in enumerate(self.layers):
            # Momentum SGD: v = momentum * v - lr * grad; W += v
            new_vw = self.momentum * layer.velocity_w - self.learning_rate * w_grads[l]
            new_vb = self.momentum * layer.velocity_b - self.learning_rate * b_grads[l]
            new_w = layer.weight + new_vw
            new_b = layer.bias + new_vb
            new_layers.append(LayerParams(
                weight=new_w,
                bias=new_b,
                activation=layer.activation,
                velocity_w=new_vw,
                velocity_b=new_vb,
            ))

        new_loss_history = self.loss_history + (loss,)
        if len(new_loss_history) > 1000:
            new_loss_history = new_loss_history[-1000:]

        return MultiLayerPredictorV2(
            layers=tuple(new_layers),
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            loss_history=new_loss_history,
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """预测（等同forward）。"""
        return self.forward(x)


# ============================================================
# 阶段50-v2 构建器
# ============================================================
class MultiLayerPredictorBuilder:
    """多层预测模型构建器。"""

    @staticmethod
    def build(
        layer_dims: Tuple[int, ...],
        activations: Tuple[ActivationType, ...],
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_init: str = "xavier",
        seed: Optional[int] = None
    ) -> MultiLayerPredictorV2:
        """构建多层预测模型。

        Args:
            layer_dims: 各层维度 (input_dim, hidden1, ..., output_dim)
            activations: 各层激活函数
            learning_rate: 学习率
            momentum: 动量系数
            weight_init: 权重初始化方法 ("xavier" 或 "he" 或 "zeros")
            seed: 随机种子

        Returns:
            MultiLayerPredictorV2 实例
        """
        if len(layer_dims) < 2:
            raise ValueError("至少需要输入层和输出层")
        if len(activations) != len(layer_dims) - 1:
            raise ValueError("activations 长度必须等于层数")

        if seed is not None:
            np.random.seed(seed)

        num_layers = len(layer_dims) - 1
        layers = []
        for l in range(num_layers):
            in_dim = layer_dims[l]
            out_dim = layer_dims[l + 1]

            if weight_init == "xavier":
                scale = np.sqrt(2.0 / (in_dim + out_dim))
                W = np.random.randn(out_dim, in_dim) * scale
            elif weight_init == "he":
                scale = np.sqrt(2.0 / in_dim)
                W = np.random.randn(out_dim, in_dim) * scale
            else:
                W = np.zeros((out_dim, in_dim))

            b = np.zeros(out_dim)
            v_w = np.zeros((out_dim, in_dim))
            v_b = np.zeros(out_dim)

            layers.append(LayerParams(
                weight=W, bias=b,
                activation=activations[l],
                velocity_w=v_w, velocity_b=v_b,
            ))

        return MultiLayerPredictorV2(
            layers=tuple(layers),
            input_dim=layer_dims[0],
            output_dim=layer_dims[-1],
            learning_rate=learning_rate,
            momentum=momentum,
        )
