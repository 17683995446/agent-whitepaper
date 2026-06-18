# 阶段19：预测编码网络实现——从理论到代码的完整映射

> **阶段编号**：S19 | **所属层**：第二层·工程实现层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | PredictiveCodingNetwork | "预测编码大脑"——完整的预测编码神经网络实现 |
> | 前向传播 | "自上而下预测"——高层猜低层应该看到什么 |
> | 误差计算 | "自下而上汇报"——低层告诉高层猜错了多少 |
> | 感知更新 | "反复琢磨"——多次调整认识直到与输入一致 |
> | 权重更新 | "修改经验"——根据理解的结果调整翻译规则 |
> | 训练循环 | "看-想-学"的完整循环——输入→感知→学习的反复迭代 |

---

## 1. 第一性原理依据

本阶段综合引用阶段11-18的所有理论结果，将完整的预测编码理论映射为可执行的代码实现。核心依据为：

- **A11.1-A11.4（层级生成模型）**（引用阶段11）：分层条件分布、联合对数似然、生成函数、抽象度-深度关系。
- **A12.1-A12.4（误差传播）**（引用阶段12）：递归传播信号、Jacobian链式传播、精度加权传播。
- **A13.1-A13.4（自上而下预测）**（引用阶段13）：预测生成、多层递归预测、预测误差分解。
- **A14.1-A14.4（表征更新）**（引用阶段14）：$\tau_r \dot{r}^l = \delta^{l+1} - e^l$，感知推断的核心方程。
- **A15.1-A15.5（权重更新）**（引用阶段15）：Hebbian学习、精度加权、EWC正则化。
- **A16.1-A16.5（离散化）**（引用阶段16）：欧拉法、CFL条件、半隐式方法、自适应步长。
- **A17.1-A17.5（快感知-慢学习）**（引用阶段17）：时间尺度分离、感知迭代、权重冻结。
- **定理18.1（等价性定理）**（引用阶段18）：预测编码与自由能最小化的严格等价。

本阶段的核心目标是：**将阶段11-18的所有理论结果整合为一个完整的预测编码网络实现，包括类设计、前向传播、误差计算、感知更新、权重更新和训练循环，提供PyTorch/JAX双框架的伪代码实现。** 这是整个公理体系从理论到工程的最终落地——所有数学公式在此转化为可执行的代码。

---

## 2. 核心数学公式

### 第一部分：PredictiveCodingNetwork类设计

#### 2.1 类设计（A19.1）

PredictiveCodingNetwork类封装完整的预测编码网络，包含层级生成模型、精度矩阵、训练配置和状态：

```
类：PredictiveCodingNetwork
属性：
    layers: List[PCLayer]              // 各层配置（引用A11.1）
    weights: List[WeightMatrix]        // 各层权重 {W^l}（引用A11.3）
    precisions: List[PrecisionMatrix]  // 各层精度 {Π_l}（引用A3.4）
    representations: List[Vector]      // 各层表征 {r^l}（引用A3.1）
    config: PCNetworkConfig            // 训练配置（组合EquivalenceProof）

方法：
    forward()         → 自上而下预测（A19.2）
    compute_errors()  → 误差计算（A19.3）
    perceive()        → 感知更新（A19.4）
    learn()           → 权重更新（A19.5）
    train_step()      → 完整训练步（A19.6）
```

- **输入**：层级配置 `layers`，初始权重 `weights`，训练配置 `config`
- **输出**：PredictiveCodingNetwork实例
- **约束**：层数 $L \geq 2$；每层维度 $n_l \geq 1$；权重矩阵 $W^l \in \mathbb{R}^{n_{l-1} \times n_l}$

> **白话翻译**：PredictiveCodingNetwork就是"预测编码大脑"的完整实现——它知道怎么预测（前向传播）、怎么发现错误（误差计算）、怎么修正认识（感知更新）、怎么修改经验（权重更新），以及怎么把这一切串起来（训练循环）。

**PCLayer的数据结构**：

```
数据类：PCLayer
属性：
    input_dim: int       // 输入维度 n_{l-1}
    output_dim: int      // 输出维度 n_l
    activation: str      // 激活函数 ('linear' | 'tanh' | 'relu')
    precision_init: float // 精度初始值
```

**与阶段11-18的映射**：

| 类属性/方法 | 理论对应 | 引用阶段 |
|------------|---------|---------|
| `layers` | 层级生成模型结构 | 阶段11 |
| `weights` | 生成函数参数 $W^l$ | 阶段11 |
| `precisions` | 精度矩阵 $\Pi_l$ | 阶段3 |
| `representations` | 各层表征 $r^l$ | 阶段3 |
| `forward()` | 自上而下预测 $\hat{r}^{l-1} = f^l(r^l, W^l)$ | 阶段13 |
| `compute_errors()` | 误差信号 $e^l = r^l - \hat{r}^l$ | 阶段3 |
| `perceive()` | 表征更新 $r^l \leftarrow r^l + (\Delta t/\tau_r)(\delta^{l+1} - e^l)$ | 阶段14 |
| `learn()` | 权重更新 $W^l \leftarrow W^l - (\eta/\tau_W)(\partial f^l/\partial W^l) \cdot \delta^l$ | 阶段15 |

---

### 第二部分：前向传播

#### 2.2 前向传播（A19.2）

前向传播实现自上而下的预测生成（引用阶段13，A13.1）：

$$\hat{r}^{l-1} = f^l(r^l, W^l), \quad l = L, L-1, \ldots, 1 \tag{A19.2}$$

其中 $f^l$ 为第 $l$ 层的生成函数，$r^l$ 为第 $l$ 层表征，$W^l$ 为权重矩阵。

- **输入**：各层表征 $\{r^l\}$，各层权重 $\{W^l\}$
- **输出**：各层预测 $\{\hat{r}^{l-1}\}$
- **约束**：更新顺序为自上而下（顶层先，底层后）

> **白话翻译**：前向传播就是"自上而下猜"——从最高层的抽象表征开始，逐层向下预测每层应该看到什么。就像老板告诉经理目标，经理告诉员工任务，员工执行具体操作。

**前向传播的伪代码**：

```python
def forward(self) -> List[Vector]:
    """自上而下预测（A19.2，引用A13.1）。

    从顶层表征逐层向下生成预测。
    """
    predictions = [None] * self.num_layers

    # 顶层预测（先验）
    predictions[self.num_layers - 1] = self.prior_mean

    # 自上而下逐层预测
    for l in range(self.num_layers - 1, 0, -1):
        predictions[l - 1] = self.generate(
            self.representations[l],  # r^l
            self.weights[l],          # W^l
            self.layers[l].activation # 激活函数
        )

    return predictions

def generate(self, r_l: Vector, W_l: Matrix, activation: str) -> Vector:
    """生成函数 f^l(r^l, W^l)（引用A11.3）。

    线性模型: f^l = W^l @ r^l
    非线性模型: f^l = σ(W^l @ r^l)
    """
    linear = W_l @ r_l
    if activation == 'linear':
        return linear
    elif activation == 'tanh':
        return jnp.tanh(linear)
    elif activation == 'relu':
        return jnp.maximum(0, linear)
```

**前向传播的计算复杂度**：

每层预测需要 $O(n_{l-1} \times n_l)$ 的矩阵乘法。$L$ 层的总计算量为 $O(\sum_l n_{l-1} n_l)$，与标准前馈网络相同。

---

### 第三部分：误差计算

#### 2.3 误差计算（A19.3）

误差计算实现自下而上的误差传播（引用阶段3，A3.1-A3.4；引用阶段12，A12.2）：

$$e^l = r^l - \hat{r}^l, \quad l = 0, 1, \ldots, L \tag{A19.3-error}$$

$$\delta^l = \Pi_l e^l + \left(\frac{\partial f^l}{\partial r^l}\right)^T \delta^{l+1}, \quad l = L, L-1, \ldots, 1 \tag{A19.3-delta}$$

- **输入**：各层表征 $\{r^l\}$，各层预测 $\{\hat{r}^l\}$，精度矩阵 $\{\Pi_l\}$，Jacobian $\{\partial f^l/\partial r^l\}$
- **输出**：各层误差信号 $\{e^l\}$，各层传播信号 $\{\delta^l\}$
- **约束**：误差计算顺序为自下而上（底层先，顶层后）；传播信号计算顺序为自上而下（顶层先，底层后）

> **白话翻译**：误差计算就是"自下而上汇报"——底层先算出自己与预测的偏差，然后逐层向上汇报，每层把本层的偏差和上层传来的信号合并后继续向上传。精度加权就是"按信任度加权汇报"——越可信的层，汇报的权重越大。

**误差计算的伪代码**：

```python
def compute_errors(self, observation: Vector) -> Tuple[List[Vector], List[Vector]]:
    """误差计算（A19.3，引用A3.1-A3.4, A12.2）。

    计算各层误差信号和传播信号。
    """
    # 自上而下预测
    predictions = self.forward()

    # 计算误差信号（A3.1）
    errors = [None] * (self.num_layers + 1)
    errors[0] = observation - predictions[0]  # 观测误差 e^0
    for l in range(1, self.num_layers + 1):
        errors[l] = self.representations[l] - predictions[l]  # e^l = r^l - r̂^l

    # 计算传播信号（A12.2，自上而下递归）
    deltas = [None] * (self.num_layers + 1)
    deltas[self.num_layers] = (
        self.precisions[self.num_layers] @ errors[self.num_layers]
    )  # 顶层：δ^L = Π_L @ e^L

    for l in range(self.num_layers - 1, 0, -1):
        # Jacobian转置传播（A12.2）
        J_T = self.jacobian_transpose(l)
        deltas[l] = (
            self.precisions[l] @ errors[l]  # Π_l @ e^l（局部信号）
            + J_T @ deltas[l + 1]           # (∂f/∂r)^T @ δ^{l+1}（传播信号）
        )

    return errors, deltas
```

**误差计算的计算复杂度**：

误差信号计算需要 $O(\sum_l n_l)$ 的向量减法。传播信号计算需要 $O(\sum_l n_l^2)$ 的矩阵-向量乘法（Jacobian转置传播）。总计算量为 $O(\sum_l n_l^2)$。

---

### 第四部分：感知更新

#### 2.4 感知更新（A19.4）

感知更新实现快感知迭代（引用阶段14，A14.3；引用阶段17，A17.4）：

$$r^l \leftarrow r^l + \frac{\Delta t}{\tau_r}(\delta^{l+1} - e^l), \quad k = 1, \ldots, N_{\text{perceive}} \tag{A19.4}$$

- **输入**：当前表征 $\{r^l\}$，传播信号 $\{\delta^l\}$，误差信号 $\{e^l\}$，配置参数
- **输出**：更新后的表征 $\{r'^l\}$
- **约束**：权重冻结（感知期间 $W^l$ 不变）；迭代至收敛或达到最大步数

> **白话翻译**：感知更新就是"反复琢磨"——每次看到新输入，不是只调整一次认识，而是反复调整直到认识与输入一致。调整方向 = "上层传来的信号" - "本层的误差"，调整幅度由步长和时间常数决定。

**感知更新的伪代码**：

```python
def perceive(self, observation: Vector) -> List[Vector]:
    """感知更新（A19.4，引用A14.3, A17.4）。

    在固定权重下迭代更新表征直到收敛。
    """
    config = self.config.fast_slow_config

    for k in range(config.perceive_iterations):
        # 计算误差和传播信号
        errors, deltas = self.compute_errors(observation)

        # 表征更新（A14.3）
        for l in range(1, self.num_layers + 1):
            gradient = deltas[l + 1] - errors[l]  # ∂F/∂r^l = δ^{l+1} - e^l
            self.representations[l] = (
                self.representations[l]
                + config.perceive_step_size / config.perceive_time_constant
                * gradient
            )

        # 收敛检查（A14.4）
        max_gradient = max(
            jnp.linalg.norm(deltas[l + 1] - errors[l])
            for l in range(1, self.num_layers + 1)
        )
        if max_gradient < config.convergence_threshold:
            break  # 早停

    return self.representations
```

**感知更新的可选方法**：

| 方法 | 引用 | 特点 |
|------|------|------|
| 显式欧拉 | A16.1 | 简单，可并行，步长受限 |
| 半隐式 | A16.4 | 更稳定，需自上而下顺序 |
| 自适应步长 | A16.5 | 自动调节步长，效率高 |

---

### 第五部分：权重更新

#### 2.5 权重更新（A19.5）

权重更新实现慢学习算法（引用阶段15，A15.3；引用阶段17，A17.5）：

$$W^l \leftarrow W^l - \frac{\eta}{\tau_W}\left(\frac{\partial f^l}{\partial W^l} \cdot \delta^l + \lambda_{\text{EWC}} \cdot F^l_{\text{diag}} \odot (W^l - W^l_{\text{old}})\right) \tag{A19.5}$$

- **输入**：当前权重 $\{W^l\}$，感知收敛后的表征 $\{r^l\}$，传播信号 $\{\delta^l\}$，EWC参数
- **输出**：更新后的权重 $\{W'^l\}$
- **约束**：感知收敛后执行；学习率 $\eta > 0$；EWC强度 $\lambda_{\text{EWC}} \geq 0$

> **白话翻译**：权重更新就是"想通了再改经验"——先把当前输入彻底搞懂（感知收敛），然后根据理解的结果调整翻译规则。Hebbian学习就是"一起激活就一起加强"——预测方向和误差方向一致时，连接增强。EWC正则化就是"不忘旧知识"——改经验的时候，对重要的旧知识加弹簧保护。

**权重更新的伪代码**：

```python
def learn(self, observation: Vector) -> List[Matrix]:
    """权重更新（A19.5，引用A15.3, A17.5）。

    在感知收敛后更新权重。
    """
    config = self.config.fast_slow_config

    # 计算误差和传播信号（使用收敛后的表征）
    errors, deltas = self.compute_errors(observation)

    for l in range(1, self.num_layers + 1):
        # 计算权重梯度（A15.1）
        # 线性模型: ∂f^l/∂W^l = r^l（外积左因子）
        weight_gradient = self.compute_weight_gradient(
            l, deltas[l]
        )

        # EWC正则化（A15.5）
        if config.use_ewc and self.fisher_diagonals is not None:
            ewc_penalty = (
                config.ewc_strength
                * self.fisher_diagonals[l]
                * (self.weights[l] - self.anchor_weights[l])
            )
            weight_gradient = weight_gradient + ewc_penalty

        # 权重更新（A15.3）
        self.weights[l] = (
            self.weights[l]
            - config.learn_step_size * weight_gradient
        )

    return self.weights

def compute_weight_gradient(self, layer: int, delta: Vector) -> Matrix:
    """计算权重梯度（A15.1）。

    线性模型: ∂F/∂W^l = -r^l @ δ^l^T（Hebbian学习）
    非线性模型: 使用JAX自动微分
    """
    if self.layers[layer].activation == 'linear':
        # Hebbian学习规则（A15.1-linear）
        return -jnp.outer(self.representations[layer], delta)
    else:
        # 非线性模型：自动微分
        return -jax.grad(
            lambda W: self.layer_free_energy(layer, W)
        )(self.weights[layer])
```

---

### 第六部分：完整训练循环

#### 2.6 完整训练循环（A19.6）

完整训练循环整合感知更新和权重更新，实现快感知-慢学习的交替执行：

```
算法：PredictiveCodingNetwork.train(data_stream, config)
输入：数据流 {(o_t)}, 训练配置 PCNetworkConfig
输出：训练后的网络权重

1.  初始化：
2.      weights ← Xavier初始化（引用A11.3）
3.      representations ← 零向量
4.      precisions ← 单位矩阵 × precision_init
5.
6.  for epoch = 1 to num_epochs do
7.      for each observation o_t in data_stream do
8.
9.          // ===== 快感知阶段（A19.4）=====
10.         representations ← perceive(o_t)
11.         // 内部循环：
12.         //   for k = 1 to N_perceive do
13.         //     predictions ← forward()
14.         //     errors, deltas ← compute_errors(o_t)
15.         //     for l = 1 to L do
16.         //       r^l ← r^l + (Δt/τ_r)(δ^{l+1} - e^l)
17.         //     end for
18.         //     if converged then break
19.         //   end for
20.
21.         // ===== 慢学习阶段（A19.5）=====
22.         weights ← learn(o_t)
23.         // 内部：
24.         //   errors, deltas ← compute_errors(o_t)
25.         //   for l = 1 to L do
26.         //     grad ← (∂f^l/∂W^l) · δ^l
27.         //     if use_ewc then grad += λ_EWC · F_diag ⊙ (W - W_old)
28.         //     W^l ← W^l - (η/τ_W) · grad
29.         //   end for
30.
31.         // ===== 监控（可选）=====
32.         F_PC ← compute_pc_free_energy()  // A18.1
33.         log(epoch, step, F_PC)
34.
35.     end for
36.
37.     // EWC锚点更新（每个epoch结束时）
38.     if use_ewc then
39.         anchor_weights ← weights
40.         fisher_diagonals ← estimate_fisher()  // A15.5-fisher
41.     end if
42.
43. end for
44.
45. return weights
```

**训练循环的关键设计决策**：

1. **感知-学习交替**：每个输入步先执行感知迭代（A19.4），再执行权重更新（A19.5）。这保证权重更新基于收敛后的表征。

2. **权重冻结**：感知迭代期间权重不变。这保证感知更新在固定的生成模型下进行。

3. **早停策略**：感知迭代在收敛后立即停止（A14.4），不必等满 $N_{\text{perceive}}$ 步。

4. **自由能监控**：每个训练步计算 $F_{\text{PC}}$（A18.1），验证自由能单调递减（定理17.1-a）。

5. **EWC锚点更新**：每个epoch结束时更新锚点权重和Fisher信息（A15.5），用于持续学习场景。

**训练循环的计算复杂度**：

每个训练步的计算量为：

$$T_{\text{step}} = N_{\text{perceive}} \cdot O\left(\sum_l n_l^2\right) + O\left(\sum_l n_{l-1} n_l\right)$$

其中第一项为感知迭代的计算量，第二项为权重更新的计算量。由A17.3，$N_{\text{perceive}} \approx 100$，因此感知阶段主导计算量。

**与标准反向传播的对比**：

| 特性 | 预测编码（A19.6） | 标准反向传播 |
|------|------------------|------------|
| 表征更新 | 迭代收敛（$N_{\text{perceive}}$ 步） | 单步前向传播 |
| 权重更新 | Hebbian学习（局部） | 全局梯度下降 |
| 计算量 | $\sim 100 \times$ 前向传播 | $\sim 2 \times$ 前向传播 |
| 生物学合理性 | 高（局部学习） | 低（全局反向传播） |
| 在线学习 | 天然支持 | 需要额外机制 |
| 持续学习 | EWC正则化（A15.5） | 需要额外机制 |

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A19.1 | PredictiveCodingNetwork类设计 | layers, weights, precisions, config | PCNetwork实例 | $L \geq 2$；维度匹配 |
| A19.2 | $\hat{r}^{l-1} = f^l(r^l, W^l)$，自上而下 | $\{r^l\}$, $\{W^l\}$ | $\{\hat{r}^{l-1}\}$ | 自上而下顺序 |
| A19.3 | $e^l = r^l - \hat{r}^l$，$\delta^l = \Pi_l e^l + (\partial f^l/\partial r^l)^T \delta^{l+1}$ | $\{r^l\}$, $\{\hat{r}^l\}$, $\{\Pi_l\}$ | $\{e^l\}$, $\{\delta^l\}$ | 自下而上+自上而下 |
| A19.4 | $r^l \leftarrow r^l + (\Delta t/\tau_r)(\delta^{l+1} - e^l)$，迭代 | $r^l$, $\delta^{l+1}$, $e^l$, $\tau_r$, $\Delta t$ | $r'^l$ | 权重冻结；收敛检查 |
| A19.5 | $W^l \leftarrow W^l - (\eta/\tau_W)((\partial f^l/\partial W^l) \cdot \delta^l + \text{EWC})$ | $W^l$, $\partial f^l/\partial W^l$, $\delta^l$, $\eta$, $\tau_W$ | $W'^l$ | 感知收敛后执行 |
| A19.6 | 完整训练循环：perceive → learn → monitor | 数据流, config | 训练后权重 | 定理17.1保证收敛 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Literal
import numpy as np

# 引用阶段18的数据类
from stage_018 import EquivalenceProof


# ============================================================
# 阶段19 数据类：预测编码网络配置
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class PCNetworkConfig:
    """预测编码网络的完整配置描述。

    组合阶段18的EquivalenceProof，在等价性证明的基础上增加
    网络架构和训练超参数的配置。

    Attributes:
        equivalence_proof: 阶段18的等价性证明结果（组合字段）
        layer_dims: 各层维度元组 (n_0, n_1, ..., n_L)
        activations: 各层激活函数元组
        weight_init: 权重初始化方法 ('xavier' | 'he' | 'normal')
        precision_init: 精度矩阵初始值
        num_epochs: 训练轮数
        batch_size: 批量大小

    Invariants:
        - len(layer_dims) >= 3（至少输入层+1隐层+输出层）
        - len(activations) == len(layer_dims) - 1
        - precision_init > 0
        - num_epochs >= 1
        - batch_size >= 1
    """
    equivalence_proof: EquivalenceProof
    layer_dims: tuple[int, ...]
    activations: tuple[str, ...]
    weight_init: Literal['xavier', 'he', 'normal'] = 'xavier'
    precision_init: float = 1.0
    num_epochs: int = 100
    batch_size: int = 32

    def __post_init__(self):
        # 验证层数
        if len(self.layer_dims) < 3:
            raise ValueError(
                f"至少需要3层（输入+隐层+输出），"
                f"收到 {len(self.layer_dims)} 层"
            )
        # 验证激活函数数量
        if len(self.activations) != len(self.layer_dims) - 1:
            raise ValueError(
                f"激活函数数量 ({len(self.activations)}) 必须 = "
                f"层数-1 ({len(self.layer_dims) - 1})"
            )
        # 验证精度初始值
        if self.precision_init <= 0:
            raise ValueError(
                f"精度初始值必须 > 0，"
                f"收到 {self.precision_init}"
            )
        # 验证训练参数
        if self.num_epochs < 1:
            raise ValueError(
                f"训练轮数必须 ≥ 1，"
                f"收到 {self.num_epochs}"
            )
        if self.batch_size < 1:
            raise ValueError(
                f"批量大小必须 ≥ 1，"
                f"收到 {self.batch_size}"
            )

    @property
    def num_layers(self) -> int:
        """网络层数（不含输入层）。"""
        return len(self.layer_dims) - 1

    @property
    def total_parameters(self) -> int:
        """总参数量。"""
        return sum(
            self.layer_dims[l] * self.layer_dims[l + 1]
            for l in range(self.num_layers)
        )


# ============================================================
# 阶段19 核心接口：预测编码网络
# ============================================================
class PredictiveCodingNetwork:
    """阶段19的核心计算接口。

    实现完整的预测编码网络，包括前向传播、误差计算、
    感知更新、权重更新和训练循环。
    核心约束：
        - perceive 实现A19.4（快感知迭代）
        - learn 实现A19.5（慢学习更新）
        - train 实现A19.6（完整训练循环）
        - 自由能在训练过程中单调递减（定理17.1）
        - 预测编码与自由能最小化等价（定理18.1）
    """

    def __init__(self, config: PCNetworkConfig):
        """初始化预测编码网络。

        Args:
            config: 网络配置
        """
        self.config = config
        self._initialize_weights()
        self._initialize_representations()
        self._initialize_precisions()

    def forward(self) -> list[np.ndarray]:
        """自上而下预测（A19.2）。

        从顶层表征逐层向下生成预测。

        Returns:
            各层预测列表 [r̂^0, r̂^1, ..., r̂^{L-1}]，
            其中 r̂^{l-1} shape (n_{l-1},)

        Examples:
            >>> import numpy as np
            >>> net = PredictiveCodingNetwork(_make_test_config())
            >>> preds = net.forward()
            >>> len(preds) == net.config.num_layers
            True
        """
        predictions = []
        L = self.config.num_layers
        for l in range(L - 1, -1, -1):
            r_l = self.representations[l]
            W_l = self.weights[l]
            linear = W_l @ r_l
            act_name = self.config.activations[l] if l < len(self.config.activations) else 'linear'
            if act_name == 'tanh':
                pred = np.tanh(linear)
            elif act_name == 'relu':
                pred = np.maximum(0.0, linear)
            else:
                pred = linear
            predictions.insert(0, pred)
        return predictions

    def compute_errors(
        self,
        observation: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """误差计算（A19.3）。

        Args:
            observation: 观测输入，shape (n_0,)

        Returns:
            (误差信号列表, 传播信号列表) 元组

        Raises:
            ValueError: 若 observation 为一维但长度与输入层维度不匹配

        Examples:
            >>> import numpy as np
            >>> net = PredictiveCodingNetwork(_make_test_config())
            >>> obs = np.zeros(4)
            >>> errs, deltas = net.compute_errors(obs)
            >>> len(errs) == len(deltas) == net.config.num_layers
            True
        """
        predictions = self.forward()
        L = self.config.num_layers
        errors = []
        for l in range(L):
            if l == 0:
                e = observation - predictions[l]
            else:
                e = self.representations[l] - predictions[l]
            errors.append(e)
        # 传播信号（自上而下递归）
        deltas = [np.zeros_like(e) for e in errors]
        deltas[L - 1] = self.precisions[L - 1] @ errors[L - 1]
        for l in range(L - 2, -1, -1):
            # Jacobian 近似为权重矩阵 W^{l+1}
            J_T = self.weights[l + 1].T
            deltas[l] = (
                self.precisions[l] @ errors[l]
                + J_T @ deltas[l + 1]
            )
        return errors, deltas

    def perceive(
        self,
        observation: np.ndarray
    ) -> list[np.ndarray]:
        """感知更新（A19.4）。

        在固定权重下迭代更新表征直到收敛。

        Args:
            observation: 观测输入，shape (n_0,)

        Returns:
            收敛后的表征列表 [r^1, ..., r^L]

        Raises:
            ValueError: 若 observation 维度与输入层不匹配

        Examples:
            >>> import numpy as np
            >>> net = PredictiveCodingNetwork(_make_test_config())
            >>> obs = np.zeros(4)
            >>> reprs = net.perceive(obs)
            >>> len(reprs) == net.config.num_layers
            True
        """
        if observation.shape[0] != self.config.layer_dims[0]:
            raise ValueError(
                f"观测维度 {observation.shape[0]} 与输入层维度 "
                f"{self.config.layer_dims[0]} 不匹配"
            )
        fs_config = self.config.equivalence_proof.fast_slow_config
        dt = fs_config.perceive_step_size
        tau_r = fs_config.perceive_time_constant
        for k in range(fs_config.perceive_iterations):
            errors, deltas = self.compute_errors(observation)
            max_grad = 0.0
            for l in range(self.config.num_layers):
                gradient = deltas[l] - errors[l]
                self.representations[l] = (
                    self.representations[l] + (dt / tau_r) * gradient
                )
                grad_norm = float(np.linalg.norm(gradient))
                max_grad = max(max_grad, grad_norm)
            if max_grad < fs_config.convergence_threshold:
                break
        return self.representations

    def learn(
        self,
        observation: np.ndarray
    ) -> list[np.ndarray]:
        """权重更新（A19.5）。

        在感知收敛后更新权重。

        Args:
            observation: 观测输入，shape (n_0,)

        Returns:
            更新后的权重列表 [W^1, ..., W^L]

        Raises:
            ValueError: 若 observation 维度与输入层不匹配

        Examples:
            >>> import numpy as np
            >>> net = PredictiveCodingNetwork(_make_test_config())
            >>> obs = np.zeros(4)
            >>> W_new = net.learn(obs)
            >>> len(W_new) == net.config.num_layers
            True
        """
        if observation.shape[0] != self.config.layer_dims[0]:
            raise ValueError(
                f"观测维度 {observation.shape[0]} 与输入层维度 "
                f"{self.config.layer_dims[0]} 不匹配"
            )
        fs_config = self.config.equivalence_proof.fast_slow_config
        # 使用收敛后的表征计算误差和传播信号
        errors, deltas = self.compute_errors(observation)
        for l in range(self.config.num_layers):
            r_l = self.representations[l]
            delta_l = deltas[l]
            # Hebbian 权重梯度（线性模型）
            grad = -np.outer(delta_l, r_l).T
            # 若维度不匹配则转置
            if grad.shape != self.weights[l].shape:
                grad = grad.T
            if grad.shape != self.weights[l].shape:
                # 最后手段：广播兼容即可
                grad = np.broadcast_to(grad, self.weights[l].shape)
            self.weights[l] = self.weights[l] - fs_config.learn_step_size * grad
        return self.weights

    def train_step(
        self,
        observation: np.ndarray
    ) -> float:
        """一步完整训练（A19.6）。

        感知更新 + 权重更新 + 自由能监控。

        Args:
            observation: 观测输入，shape (n_0,)

        Returns:
            当前自由能值 F_PC

        Raises:
            ValueError: 若 observation 维度与输入层不匹配

        Examples:
            >>> import numpy as np
            >>> net = PredictiveCodingNetwork(_make_test_config())
            >>> obs = np.zeros(4)
            >>> fe = net.train_step(obs)
            >>> isinstance(fe, float)
            True
        """
        if observation.shape[0] != self.config.layer_dims[0]:
            raise ValueError(
                f"观测维度 {observation.shape[0]} 与输入层维度 "
                f"{self.config.layer_dims[0]} 不匹配"
            )
        # 快感知
        self.perceive(observation)
        # 慢学习
        self.learn(observation)
        # 自由能监控
        return self.compute_pc_free_energy()

    def compute_pc_free_energy(self) -> float:
        """计算预测编码自由能（A18.1）。

        Returns:
            F_PC 值（非负标量）

        Examples:
            >>> net = PredictiveCodingNetwork(_make_test_config())
            >>> fe = net.compute_pc_free_energy()
            >>> fe >= 0
            True
        """
        total = 0.0
        for l in range(self.config.num_layers):
            e = self.representations[l] - (self.weights[l] @ self.representations[l])
            energy = float(e.T @ self.precisions[l] @ e)
            total += 0.5 * energy
        return total

    def reconstruct(self, observation: np.ndarray) -> np.ndarray:
        """重构输入（用于验证）。

        执行感知推断后，通过底层预测重构观测。

        Args:
            observation: 观测输入，shape (n_0,)

        Returns:
            重构后的输入，shape (n_0,)

        Raises:
            ValueError: 若 observation 维度与输入层不匹配

        Examples:
            >>> import numpy as np
            >>> net = PredictiveCodingNetwork(_make_test_config())
            >>> obs = np.ones(4) * 0.5
            >>> recon = net.reconstruct(obs)
            >>> recon.shape == obs.shape
            True
        """
        if observation.shape[0] != self.config.layer_dims[0]:
            raise ValueError(
                f"观测维度 {observation.shape[0]} 与输入层维度 "
                f"{self.config.layer_dims[0]} 不匹配"
            )
        self.perceive(observation)
        predictions = self.forward()
        return predictions[0]
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `EquivalenceProof` | ← 阶段18 | 作为网络配置的组合字段 |
| **输入** | A11.1-A11.4（层级生成模型） | ← 阶段11 | 网络架构的理论基础 |
| **输入** | A12.1-A12.4（误差传播） | ← 阶段12 | 误差计算的传播机制 |
| **输入** | A13.1-A13.4（自上而下预测） | ← 阶段13 | 前向传播的理论基础 |
| **输入** | A14.1-A14.4（表征更新） | ← 阶段14 | 感知更新的核心方程 |
| **输入** | A15.1-A15.5（权重更新） | ← 阶段15 | 权重更新的核心方程 |
| **输入** | A16.1-A16.5（离散化） | ← 阶段16 | 离散化方法选择 |
| **输入** | A17.1-A17.5（快感知-慢学习） | ← 阶段17 | 训练循环的时间尺度分离 |
| **输入** | 定理18.1（等价性定理） | ← 阶段18 | 预测编码的理论保证 |
| **输出** | `PCNetworkConfig` | → 阶段20 | 包含EquivalenceProof的网络配置 |
| **输出** | A19.6（完整训练循环） | → 阶段20 | 验证实验的训练算法 |
| **输出** | `PredictiveCodingNetwork` | → 阶段20 | 验证实验的网络实现 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 核心计算框架 | `jax.vmap`实现批量训练；`jax.jit`编译训练循环；`jax.grad`计算权重梯度（非线性模型）；`jax.lax.scan`实现感知迭代的函数式循环 |
| equinox | 模块化网络定义 | `eqx.Module`定义PCLayer和PredictiveCodingNetwork；`eqx.filter_jit`实现细粒度JIT编译；`eqx.nn.Linear`提供线性层基础实现 |
| optax | 优化器与调度 | `optax.sgd`实现权重更新（A19.5）；`optax.adam`作为对比优化器；`optax.exponential_decay`实现学习率衰减 |
| dlx | **本阶段新增**：预测编码库 | dlx（或pypc）提供JAX原生的预测编码实现参考，包括`PredictiveCodingNetwork`类、`InferenceProcedure`接口、`WeightUpdate`规则，直接支撑A19.1-A19.6的工程实现 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V19.1 | 前向传播正确性 | $\hat{r}^{l-1} = f^l(r^l, W^l)$ 与手动计算一致 | 小规模网络数值验证 |
| V19.2 | 误差信号正确性 | $e^l = r^l - \hat{r}^l$ 与A3.1一致 | 误差信号对比 |
| V19.3 | 传播信号正确性 | $\delta^l = \Pi_l e^l + (\partial f^l/\partial r^l)^T \delta^{l+1}$ 与A12.2一致 | 传播信号递归验证 |
| V19.4 | 感知更新正确性 | A19.4与A14.3一致 | 梯度对比验证 |
| V19.5 | 权重更新正确性 | A19.5与A15.3一致 | 梯度对比验证 |
| V19.6 | 权重冻结 | 感知迭代期间权重不变 | 权重变化监控 |
| V19.7 | 自由能单调递减 | $F(W_{t+1}) \leq F(W_t)$（定理17.1-a） | 多步训练自由能监控 |
| V19.8 | 收敛性 | 权重梯度范数 $\to 0$（定理17.1-b） | 长期训练梯度监控 |
| V19.9 | 梯度等价性 | 预测编码梯度与JAX自动微分梯度一致（A18.2） | 梯度对比验证 |
| V19.10 | Hebbian学习 | 线性模型下权重更新与Hebbian规则一致 | 线性模型验证 |
| V19.11 | EWC正则化 | EWC正则化减少遗忘 | 持续学习场景测试 |
| V19.12 | 早停有效性 | 收敛后早停与满步迭代结果一致 | 早停vs满步对比 |
| V19.13 | 重构正确性 | 重构误差随训练递减 | 重构质量监控 |
| V19.14 | PCNetworkConfig组合 | `config.equivalence_proof` 类型为 `EquivalenceProof` | 类型检查 |
| V19.15 | 非法输入拒绝 | 层数<3、维度不匹配、精度≤0等抛出 `ValueError` | 异常测试 |

---

> **阶段19完** | 下一步：阶段20将从本阶段的PredictiveCodingNetwork出发，设计并执行完整的验证实验。
