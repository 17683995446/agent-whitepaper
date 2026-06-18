# 阶段15：权重更新规则（模型学习）——Hebbian学习与EWC正则化

> **阶段编号**：S15 | **所属层**：第二层·工程实现层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 权重更新 | "修改经验"——根据误差信号调整"翻译规则"，让未来的预测更准 |
> | Hebbian学习 | "一起激活就一起加强"——预测方向和误差方向一致时，连接增强 |
> | 精度加权权重更新 | "按信任度修改经验"——越可信的层级，经验修改越精细 |
> | 离散化权重更新 | "一步步修改经验"——每次只改一点点，避免改过头 |
> | 学习率与精度的关系 | "信任度越低，学得越快"——不确定的时候更需要学习 |
> | EWC正则化 | "不忘旧知识"——学新东西的时候，不能把旧知识全忘了 |

---

## 1. 第一性原理依据

本阶段从阶段4和阶段5的核心结论出发，无需引入新的物理公理。核心依据为：

- **A4.4（变分自由能全局唯一定义）**（引用阶段4）：$F[q] = -\mathbb{E}_q[\ln p(o|\psi)] + D_{\text{KL}}[q(\psi) \| p(\psi)]$，权重更新的优化目标。
- **A5.8（复杂性梯度）**（引用阶段5）：复杂性项关于变分参数的梯度，权重更新中复杂性惩罚的来源。
- **A5.14（L2正则化）**（引用阶段5）：L2正则化作为复杂性项的特例，EWC正则化的基础。
- **A14.1（表征更新方程）**（引用阶段14）：$\tau_r \dot{r}^l = \delta^{l+1} - e^l$，表征更新与权重更新的时间尺度分离。

本阶段的核心目标是：**从自由能最小化推导权重更新的连续时间动力学、精度调制形式、离散化算法、学习率与精度的关系，以及EWC正则化规则，建立完整的模型学习计算规则。** 权重更新与阶段14的表征更新构成预测编码的双学习过程——表征更新是"快速感知"（表层循环），权重更新是"慢速学习"（中层循环）。

---

## 2. 核心数学公式

### 第一部分：权重更新的连续时间动力学

#### 2.1 权重更新方程（A15.1）

权重更新方程从自由能关于权重的梯度推导。与表征更新（A14.1）不同，权重更新不改变表征 $r^l$，而是改变生成函数的参数 $W^l$，使未来的预测更准确：

$$\tau_W \dot{W}^l = -\frac{\partial F}{\partial W^l} = -\frac{\partial f^l}{\partial W^l} \cdot \delta^l \tag{A15.1}$$

其中：
- $\tau_W > 0$ 为权重更新的时间常数（$\tau_W \gg \tau_r$，权重更新远慢于表征更新）
- $\partial f^l / \partial W^l$ 为生成函数关于权重的偏导数（三维张量，但通常可以简化）
- $\delta^l$ 为第 $l$ 层的传播信号（引用阶段12，A12.2）

- **输入**：生成函数 $f^l$，权重 $W^l$，传播信号 $\delta^l$，时间常数 $\tau_W$
- **输出**：权重的时间导数 $\dot{W}^l \in \mathbb{R}^{n_{l-1} \times n_l}$
- **约束**：$\tau_W \gg \tau_r$（时间尺度分离）；$\delta^l$ 由A12.2递归计算

> **白话翻译**：权重更新就是"修改翻译规则"——如果发现翻译出来的预测和实际有偏差，就调整翻译规则让下次预测更准。更新方向 = 翻译规则的梯度 × 误差信号。这和Hebbian学习规则"一起激活就一起加强"完全一致——预测方向和误差方向一致时，连接增强。

**完整推导**：

**步骤1：自由能关于权重的依赖**

由A11.3（引用阶段11），联合对数似然为 $\ln p(o, r^{1:L}) = \sum_{l=0}^{L}\ln p(r^{l-1}|r^l) + \ln p(r^{L+1})$。权重 $W^l$ 仅出现在条件分布 $p(r^{l-1}|r^l) = \mathcal{N}(r^{l-1}; f^l(r^l, W^l), \Sigma^l)$ 中。

**步骤2：计算梯度**

由A11.2-log（引用阶段11），$\ln p(r^{l-1}|r^l) = -\frac{1}{2}(r^{l-1} - f^l(r^l, W^l))^T \Pi_l (r^{l-1} - f^l(r^l, W^l)) + \text{const}$。

对 $W^l$ 求梯度：

$$\frac{\partial \ln p(r^{l-1}|r^l)}{\partial W^l} = \left(\frac{\partial f^l}{\partial W^l}\right)^T \Pi_l (r^{l-1} - f^l(r^l, W^l)) = \left(\frac{\partial f^l}{\partial W^l}\right)^T \Pi_l e^{l-1}$$

由A3.4（引用阶段3），$\Pi_l e^{l-1} = \delta^l_{\text{local}}$（第 $l$ 层的局部传播信号）。因此：

$$\frac{\partial \ln p(r^{l-1}|r^l)}{\partial W^l} = \left(\frac{\partial f^l}{\partial W^l}\right)^T \delta^l_{\text{local}}$$

自由能关于 $W^l$ 的梯度为负对数似然的梯度：

$$\frac{\partial F}{\partial W^l} = -\frac{\partial \ln p(r^{l-1}|r^l)}{\partial W^l} = -\left(\frac{\partial f^l}{\partial W^l}\right)^T \delta^l_{\text{local}}$$

**步骤3：梯度下降**

$$\tau_W \dot{W}^l = -\frac{\partial F}{\partial W^l} = \frac{\partial f^l}{\partial W^l} \cdot \delta^l$$

此处使用简化的矩阵形式（省略转置的精确处理），得A15.1。

$\blacksquare$

**与Hebbian学习规则的对应**：

A15.1的形式 $\dot{W}^l \propto (\partial f^l/\partial W^l) \cdot \delta^l$ 与经典Hebbian学习规则 $\Delta W \propto x \cdot y^T$（前突触活动 × 后突触活动）在数学结构上同构：

| 预测编码 | Hebbian学习 |
|---------|------------|
| $\partial f^l/\partial W^l$（生成函数对权重的敏感度） | 前突触活动 $x$ |
| $\delta^l$（传播信号） | 后突触活动 $y$ |
| $\dot{W}^l \propto (\partial f^l/\partial W^l) \cdot \delta^l$ | $\Delta W \propto x \cdot y^T$ |

在线性模型 $f^l(r^l, W^l) = W^l r^l$ 下，$\partial f^l/\partial W^l = r^l$（作为外积的左因子），$\delta^l$ 为右因子，因此：

$$\dot{W}^l \propto r^l \cdot (\delta^l)^T = r^l \cdot (\Pi_l e^{l-1})^T$$

这正是Hebbian学习规则——"前突触表征 $r^l$ 和后突触误差 $\delta^l$ 一起激活时，连接 $W^l$ 增强"。

**与A5.8（复杂性梯度）的连接**：

由A5.8（引用阶段5），复杂性项关于变分参数的梯度惩罚"偏离先验"的参数变化。在权重更新中，复杂性梯度体现为权重正则化——防止权重过大导致过拟合。A15.1的权重更新仅包含准确性梯度（$\partial f^l/\partial W^l \cdot \delta^l$），复杂性惩罚将在A15.5的EWC正则化中引入。

---

### 第二部分：精度加权权重更新

#### 2.2 精度加权权重更新（A15.2）

与表征更新的精度调制类似（A14.2），权重更新也可以精度加权：

$$\tau_W \dot{W}^l = \Pi_l \cdot \left(-\frac{\partial f^l}{\partial W^l}\right) \cdot \delta^l \tag{A15.2}$$

- **输入**：精度矩阵 $\Pi_l$，生成函数偏导 $\partial f^l/\partial W^l$，传播信号 $\delta^l$，时间常数 $\tau_W$
- **输出**：精度调制的权重时间导数 $\dot{W}^l$
- **约束**：$\Pi_l$ 正定

> **白话翻译**：精度加权权重更新就是"按信任度修改经验"——精度高的层（预测很确定的层），权重更新更精细但更慢；精度低的层，权重更新更粗糙但更快。这和注意力机制一样——你更信任的感官，对它的"翻译规则"修改更谨慎。

**精度加权的物理含义**：

$\Pi_l$ 作为权重更新的"精度调制器"——高精度层的权重更新受到更强的约束（$\Pi_l$ 大意味着条件分布更集中，权重偏离最优值的代价更大），因此更新更精细。低精度层的权重更新更自由（$\Pi_l$ 小意味着条件分布更分散，权重偏离的代价更小），因此更新更激进。

**在线性模型下的简化**：

当 $f^l(r^l, W^l) = W^l r^l$ 时，$\partial f^l/\partial W^l = r^l$（外积左因子），A15.2简化为：

$$\tau_W \dot{W}^l = \Pi_l \cdot r^l \cdot (\delta^l)^T = \Pi_l \cdot r^l \cdot (\Pi_l e^{l-1})^T$$

这进一步简化为：

$$\tau_W \dot{W}^l = \Pi_l \cdot r^l \cdot e^{l-1,T} \cdot \Pi_l$$

即精度加权Hebbian学习——前突触和后突触信号都被精度加权。

---

### 第三部分：离散化权重更新

#### 2.3 离散化权重更新（A15.3）

将连续时间动力学A15.1用欧拉法离散化：

$$W^l_{t+1} = W^l_t - \frac{\eta}{\tau_W} \left(\frac{\partial f^l}{\partial W^l}\right) \cdot \delta^l \tag{A15.3}$$

其中 $\eta > 0$ 为学习率（与 $\Delta t$ 对应，但通常独立设置以区分表征更新和权重更新的时间尺度）。

- **输入**：当前权重 $W^l_t$，生成函数偏导 $\partial f^l/\partial W^l$，传播信号 $\delta^l$，时间常数 $\tau_W$，学习率 $\eta$
- **输出**：更新后的权重 $W^l_{t+1}$
- **约束**：$\eta > 0$；$\tau_W > 0$；稳定性要求 $\eta < 2\tau_W / \lambda_{\max}$

> **白话翻译**：离散化权重更新就是"一步步修改经验"——每次只改一点点。学习率 $\eta$ 控制每步改多少——太大可能改过头（过拟合），太小改得太慢（学不动）。时间常数 $\tau_W$ 控制整体速度——$\tau_W$ 越大，学习越慢但越稳。

**与表征更新的时间尺度分离**：

由A9.13（引用阶段9），$\tau_r \ll \tau_W$——表征更新远快于权重更新。这意味着：

1. **表征先收敛**：在每次权重更新前，表征已经近似收敛到当前权重下的最优值（$\delta^{l+1} \approx e^l$）
2. **权重慢更新**：权重在表征收敛的基础上缓慢调整，不会破坏表征的稳定性

这就是"感知-学习分离"——快速感知（表征更新）和慢速学习（权重更新）在不同时间尺度上运行，互不干扰。

**在线性模型下的简化**：

当 $f^l(r^l, W^l) = W^l r^l$ 时，$\partial f^l/\partial W^l = r^l$，A15.3简化为：

$$W^l_{t+1} = W^l_t - \frac{\eta}{\tau_W} r^l \cdot (\delta^l)^T \tag{A15.3-linear}$$

这与标准梯度下降 $W \leftarrow W - \eta \nabla_W L$ 在形式上完全一致。

---

### 第四部分：学习率与精度的关系

#### 2.4 学习率与精度的关系（A15.4）

学习率 $\eta_l$ 与精度 $\Pi_l$ 之间存在反比关系：

$$\eta_l \propto \Pi_l^{-1} \tag{A15.4}$$

即精度越高的层，学习率越低；精度越低的层，学习率越高。

**推导**：

由A15.2，精度加权权重更新为 $\tau_W \dot{W}^l = \Pi_l \cdot (-\partial f^l/\partial W^l) \cdot \delta^l$。离散化后的有效更新量为：

$$\Delta W^l = -\frac{\eta}{\tau_W} \Pi_l \cdot \frac{\partial f^l}{\partial W^l} \cdot \delta^l$$

为保证不同精度层的更新量在相同量级，需要 $\eta \cdot \Pi_l \approx \text{const}$，即 $\eta_l \propto \Pi_l^{-1}$。

**物理含义**：

- 高精度层（$\Pi_l$ 大）：预测已经很准确，不需要大幅修改权重，学习率小
- 低精度层（$\Pi_l$ 小）：预测不太准确，需要更多学习，学习率大

这对应"不确定性驱动学习"——越不确定的地方，越需要学习。在神经科学中，这对应**注意力调制可塑性**——注意力集中（高精度）时，突触可塑性降低（学习率小）；注意力分散（低精度）时，突触可塑性升高（学习率大）。

**自适应学习率算法**：

$$\eta_l(t) = \eta_0 \cdot \frac{\text{tr}(\Pi_l^{-1})}{n_l} \tag{A15.4-adaptive}$$

其中 $\eta_0$ 为基础学习率，$\text{tr}(\Pi_l^{-1})/n_l$ 为精度逆矩阵的平均对角元素（平均不确定性）。当精度更新（A3.8，引用阶段3）改变 $\Pi_l$ 时，学习率自适应调整。

---

### 第五部分：EWC正则化

#### 2.5 EWC正则化（A15.5）

Elastic Weight Consolidation（EWC）正则化防止权重更新破坏已学到的知识。其核心思想是：对每个权重参数，计算其对当前任务的重要性（Fisher信息矩阵的对角元素），并在学习新任务时惩罚重要权重的改变。

EWC正则化的权重更新为：

$$\tau_W \dot{W}^l = -\frac{\partial F}{\partial W^l} - \lambda_{\text{EWC}} \cdot F^l_{\text{diag}} \odot (W^l - W^l_{\text{old}}) \tag{A15.5}$$

其中：
- $\lambda_{\text{EWC}} > 0$ 为EWC正则化强度
- $F^l_{\text{diag}} = \text{diag}(\mathcal{F}^l)$ 为Fisher信息矩阵的对角元素
- $W^l_{\text{old}}$ 为上一任务学习结束时的权重（"锚点"）
- $\odot$ 为逐元素乘法

- **输入**：自由能梯度 $\partial F/\partial W^l$，Fisher对角 $F^l_{\text{diag}}$，当前权重 $W^l$，锚点权重 $W^l_{\text{old}}$，正则化强度 $\lambda_{\text{EWC}}$
- **输出**：正则化后的权重时间导数 $\dot{W}^l$
- **约束**：$\lambda_{\text{EWC}} \geq 0$；$F^l_{\text{diag}} \geq 0$（Fisher信息非负）

> **白话翻译**：EWC正则化就是"不忘旧知识"——学新东西的时候，对"旧知识中重要的部分"加一个弹簧，把它拉回原来的位置。弹簧的强度由Fisher信息决定——越重要的权重，弹簧越强，越不容易被新知识冲掉。

**与A5.14（L2正则化）的连接**：

由A5.14（引用阶段5），L2正则化惩罚权重偏离零的程度：$\lambda \|W\|^2$。EWC正则化是L2正则化的推广——将"偏离零"替换为"偏离旧权重"，将"统一惩罚强度 $\lambda$"替换为"按重要性加权的惩罚强度 $\lambda_{\text{EWC}} \cdot F^l_{\text{diag}}$"。

| 正则化类型 | 惩罚目标 | 惩罚强度 |
|-----------|---------|---------|
| L2正则化（A5.14） | $\|W\|^2$（偏离零） | 统一 $\lambda$ |
| EWC正则化（A15.5） | $\|W - W_{\text{old}}\|^2_{F_{\text{diag}}}$（偏离旧权重） | 按重要性 $\lambda_{\text{EWC}} \cdot F_{\text{diag}}$ |

当 $W_{\text{old}} = 0$ 且 $F_{\text{diag}} = I$ 时，EWC退化为L2正则化。

**Fisher信息矩阵的计算**：

Fisher信息矩阵的对角元素近似为：

$$F^l_{\text{diag},ij} \approx \mathbb{E}_{p(r^{l-1}|r^l)}\left[\left(\frac{\partial \ln p(r^{l-1}|r^l)}{\partial W^l_{ij}}\right)^2\right] \tag{A15.5-fisher}$$

在实际计算中，使用经验近似：

$$F^l_{\text{diag},ij} \approx \frac{1}{N}\sum_{n=1}^{N}\left(\frac{\partial \ln p(r^{l-1}_n|r^l_n)}{\partial W^l_{ij}}\right)^2$$

其中 $N$ 为样本数。

**EWC正则化的离散化**：

$$W^l_{t+1} = W^l_t - \frac{\eta}{\tau_W}\left[\frac{\partial f^l}{\partial W^l} \cdot \delta^l + \lambda_{\text{EWC}} \cdot F^l_{\text{diag}} \odot (W^l_t - W^l_{\text{old}})\right] \tag{A15.5-discrete}$$

**EWC正则化的自由能解释**：

EWC正则化项 $\lambda_{\text{EWC}} \cdot F^l_{\text{diag}} \odot (W^l - W^l_{\text{old}})$ 可以理解为对权重引入了一个高斯先验：

$$p(W^l | W^l_{\text{old}}, F^l_{\text{diag}}) = \mathcal{N}(W^l; W^l_{\text{old}}, (\lambda_{\text{EWC}} \cdot F^l_{\text{diag}})^{-1})$$

即"旧权重是先验均值，Fisher信息决定先验精度"。EWC正则化等价于在自由能中增加一个复杂性项——偏离旧权重的代价。这与A5.5（引用阶段5）的复杂性项 $C(q) = D_{\text{KL}}[q\|p]$ 在概念上一致——EWC将权重的先验从"偏离零"（L2）推广为"偏离旧值"（EWC）。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A15.1 | $\tau_W \dot{W}^l = -(\partial f^l/\partial W^l) \cdot \delta^l$ | $f^l$, $W^l$, $\delta^l$, $\tau_W$ | $\dot{W}^l \in \mathbb{R}^{n_{l-1} \times n_l}$ | $\tau_W \gg \tau_r$；Hebbian学习 |
| A15.2 | $\tau_W \dot{W}^l = \Pi_l \cdot (-\partial f^l/\partial W^l) \cdot \delta^l$ | $\Pi_l$, $f^l$, $W^l$, $\delta^l$, $\tau_W$ | $\dot{W}^l \in \mathbb{R}^{n_{l-1} \times n_l}$ | $\Pi_l$ 正定 |
| A15.3 | $W^l_{t+1} = W^l_t - (\eta/\tau_W)(\partial f^l/\partial W^l) \cdot \delta^l$ | $W^l_t$, $\partial f^l/\partial W^l$, $\delta^l$, $\eta$, $\tau_W$ | $W^l_{t+1} \in \mathbb{R}^{n_{l-1} \times n_l}$ | $\eta > 0$；稳定性约束 |
| A15.4 | $\eta_l \propto \Pi_l^{-1}$ | $\Pi_l$ | $\eta_l > 0$ | 精度越高，学习率越低 |
| A15.5 | $\tau_W \dot{W}^l = -\partial F/\partial W^l - \lambda_{\text{EWC}} F^l_{\text{diag}} \odot (W^l - W^l_{\text{old}})$ | $\partial F/\partial W^l$, $F^l_{\text{diag}}$, $W^l$, $W^l_{\text{old}}$, $\lambda_{\text{EWC}}$ | $\dot{W}^l \in \mathbb{R}^{n_{l-1} \times n_l}$ | $\lambda_{\text{EWC}} \geq 0$；引用A5.14 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# 引用阶段14的数据类
from stage_014 import PerceptionUpdate


# ============================================================
# 阶段15 数据类：权重更新
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class WeightUpdate:
    """权重更新（模型学习）的完整状态描述。

    组合阶段14的PerceptionUpdate，在感知更新的基础上增加
    权重更新的计算结果——更新后的权重、Fisher信息、EWC状态。

    Attributes:
        perception: 阶段14的感知更新结果（组合字段）
        weights_before: 更新前的权重元组 (W^1, ..., W^L)
        weights_after: 更新后的权重元组 (W'^1, ..., W'^L)
        weight_gradients: 各层权重梯度元组 (∂F/∂W^1, ..., ∂F/∂W^L)
        fisher_diagonals: 各层Fisher对角元组 (F^1_diag, ..., F^L_diag)
        anchor_weights: EWC锚点权重元组 (W^1_old, ..., W^L_old)
        time_constant: 权重更新时间常数 τ_W
        learning_rate: 学习率 η
        ewc_strength: EWC正则化强度 λ_EWC
        learning_rates_per_layer: 各层自适应学习率 (η_1, ..., η_L)

    Invariants:
        - len(weights_before) == len(weights_after) == model.num_layers
        - len(weight_gradients) == model.num_layers
        - len(fisher_diagonals) == model.num_layers
        - len(anchor_weights) == model.num_layers
        - time_constant > 0
        - learning_rate > 0
        - ewc_strength >= 0
        - τ_W >> τ_r（时间尺度分离）
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

    def __post_init__(self):
        L = self.perception.top_down.bottom_up.model.num_layers
        # 验证权重数量
        if len(self.weights_before) != L:
            raise ValueError(
                f"更新前权重数量 {len(self.weights_before)} "
                f"必须等于 {L}"
            )
        if len(self.weights_after) != L:
            raise ValueError(
                f"更新后权重数量 {len(self.weights_after)} "
                f"必须等于 {L}"
            )
        # 验证梯度数量
        if len(self.weight_gradients) != L:
            raise ValueError(
                f"权重梯度数量 {len(self.weight_gradients)} "
                f"必须等于 {L}"
            )
        # 验证Fisher对角数量
        if len(self.fisher_diagonals) != L:
            raise ValueError(
                f"Fisher对角数量 {len(self.fisher_diagonals)} "
                f"必须等于 {L}"
            )
        # 验证锚点权重数量
        if len(self.anchor_weights) != L:
            raise ValueError(
                f"锚点权重数量 {len(self.anchor_weights)} "
                f"必须等于 {L}"
            )
        # 验证时间常数
        if self.time_constant <= 0:
            raise ValueError(
                f"时间常数 τ_W 必须 > 0，"
                f"收到 {self.time_constant}"
            )
        # 验证学习率
        if self.learning_rate <= 0:
            raise ValueError(
                f"学习率 η 必须 > 0，"
                f"收到 {self.learning_rate}"
            )
        # 验证EWC强度
        if self.ewc_strength < 0:
            raise ValueError(
                f"EWC强度 λ_EWC 必须 ≥ 0，"
                f"收到 {self.ewc_strength}"
            )
        # 验证时间尺度分离
        repr_tau = self.perception.time_constant
        if self.time_constant <= repr_tau:
            raise ValueError(
                f"时间尺度分离违反：τ_W ({self.time_constant}) "
                f"必须 >> τ_r ({repr_tau})"
            )

    @property
    def ewc_penalty(self) -> float:
        """EWC正则化惩罚值。"""
        total = 0.0
        for l in range(len(self.weights_after)):
            diff = self.weights_after[l] - self.anchor_weights[l]
            total += float(np.sum(
                self.fisher_diagonals[l] * diff ** 2
            ))
        return self.ewc_strength * total

    @property
    def weight_change_norms(self) -> tuple[float, ...]:
        """各层权重变化的范数 ‖W'^l - W^l‖。"""
        return tuple(
            float(np.linalg.norm(wa - wb))
            for wa, wb in zip(self.weights_after, self.weights_before)
        )


# ============================================================
# 阶段15 核心接口：权重更新器
# ============================================================
class WeightUpdater:
    """阶段15的核心计算接口。

    实现权重更新的计算、Fisher信息估计和EWC正则化。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - update_step 返回的权重满足 A15.3 离散化更新
        - EWC正则化满足 A15.5
        - 学习率与精度满足 A15.4 的反比关系
        - τ_W >> τ_r（时间尺度分离）
    """

    def update_weights(
        self,
        perception: PerceptionUpdate,
        weights: tuple[np.ndarray, ...],
        fisher_diagonals: tuple[np.ndarray, ...],
        anchor_weights: tuple[np.ndarray, ...],
        time_constant: float,
        learning_rate: float,
        ewc_strength: float = 0.0
    ) -> WeightUpdate:
        """执行权重更新（A15.3 + A15.5）。

        Args:
            perception: 阶段14的感知更新结果
            weights: 当前权重 (W^1, ..., W^L)
            fisher_diagonals: Fisher对角元组
            anchor_weights: EWC锚点权重
            time_constant: 时间常数 τ_W
            learning_rate: 学习率 η
            ewc_strength: EWC强度 λ_EWC

        Returns:
            WeightUpdate 实例

        Raises:
            ValueError: 当参数不合法时
        """
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
        learning_rates = self.compute_adaptive_learning_rates(model, learning_rate)
        for l in range(L):
            wg = self.compute_weight_gradient(
                model, l + 1,
                perception.representations_after[l],
                perception.top_down.bottom_up.propagated_signals[l],
                weights[l],
            )
            weight_grads.append(wg)
            reg_grad = self.ewc_regularized_gradient(
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

    def compute_weight_gradient(
        self,
        model: 'HierarchicalGenerativeModel',
        layer_index: int,
        representation: np.ndarray,
        propagated_signal: np.ndarray,
        weights: np.ndarray
    ) -> np.ndarray:
        """计算权重梯度（A15.1）。

        ∂F/∂W^l = -(∂f^l/∂W^l) · δ^l

        Args:
            model: 层级生成模型
            layer_index: 层索引 l
            representation: 第 l 层表征 r^l
            propagated_signal: 传播信号 δ^l
            weights: 当前权重 W^l

        Returns:
            权重梯度 ∂F/∂W^l，形状与 W^l 相同
        """
        # 线性模型下 ∂f^l/∂W^l = r^l (外积左因子)
        # 返回 -r^l · (δ^l)^T
        return -np.outer(representation, propagated_signal)

    def compute_fisher_diagonal(
        self,
        model: 'HierarchicalGenerativeModel',
        layer_index: int,
        representations: tuple[np.ndarray, ...],
        num_samples: int = 100
    ) -> np.ndarray:
        """计算Fisher信息矩阵的对角元素（A15.5-fisher）。

        Args:
            model: 层级生成模型
            layer_index: 层索引 l
            representations: 各层表征样本
            num_samples: 采样数

        Returns:
            Fisher对角元素，形状与 W^l 相同
        """
        # 简化为单位矩阵（实际应通过蒙特卡洛估计）
        W_shape = model.weights[layer_index - 1].shape
        return np.ones(W_shape)

    def compute_adaptive_learning_rates(
        self,
        model: 'HierarchicalGenerativeModel',
        base_learning_rate: float
    ) -> tuple[float, ...]:
        """计算各层自适应学习率（A15.4）。

        η_l ∝ Π_l^{-1}

        Args:
            model: 层级生成模型
            base_learning_rate: 基础学习率 η_0

        Returns:
            各层学习率 (η_1, ..., η_L)
        """
        rates = []
        for l in range(model.num_layers):
            pi_inv_trace = float(np.trace(np.linalg.inv(model.precisions[l])))
            dim = model.layer_dims[l]
            rates.append(base_learning_rate * pi_inv_trace / dim)
        return tuple(rates)

    def ewc_regularized_gradient(
        self,
        weight_gradient: np.ndarray,
        fisher_diagonal: np.ndarray,
        current_weight: np.ndarray,
        anchor_weight: np.ndarray,
        ewc_strength: float
    ) -> np.ndarray:
        """计算EWC正则化后的梯度（A15.5）。

        -∂F/∂W^l - λ_EWC · F_diag ⊙ (W^l - W^l_old)

        Args:
            weight_gradient: 自由能梯度 ∂F/∂W^l
            fisher_diagonal: Fisher对角元素
            current_weight: 当前权重 W^l
            anchor_weight: 锚点权重 W^l_old
            ewc_strength: EWC强度 λ_EWC

        Returns:
            正则化后的梯度
        """
        ewc_term = ewc_strength * fisher_diagonal * (current_weight - anchor_weight)
        return weight_gradient - ewc_term

    def verify_timescale_separation(
        self,
        weight_time_constant: float,
        representation_time_constant: float,
        min_ratio: float = 10.0
    ) -> bool:
        """验证时间尺度分离条件。

        τ_W >> τ_r

        Args:
            weight_time_constant: 权重时间常数 τ_W
            representation_time_constant: 表征时间常数 τ_r
            min_ratio: 最小分离比

        Returns:
            True 如果 τ_W / τ_r ≥ min_ratio
        """
        if representation_time_constant <= 0:
            return False
        return weight_time_constant / representation_time_constant >= min_ratio
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `PerceptionUpdate` | ← 阶段14 | 作为权重更新的感知推断基础（组合字段） |
| **输入** | A4.4（变分自由能） | ← 阶段4 | 权重更新的优化目标 |
| **输入** | A5.8（复杂性梯度） | ← 阶段5 | 权重正则化的复杂性来源 |
| **输入** | A5.14（L2正则化） | ← 阶段5 | EWC正则化的基础形式 |
| **输出** | `WeightUpdate` | → 阶段9/29 | 包含PerceptionUpdate的权重更新结果 |
| **输出** | A15.1（权重更新方程） | → 阶段9 | 中层循环模型学习的计算规则 |
| **输出** | A15.3（离散化权重更新） | → 阶段9/29 | 工程实现的迭代算法 |
| **输出** | A15.5（EWC正则化） | → 阶段8/29 | 终极循环认知升级的持续学习保障 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 自动微分与JIT编译 | 权重梯度（A15.1）的自动微分计算；Fisher信息（A15.5-fisher）的蒙特卡洛估计；EWC正则化（A15.5）的JIT编译 |
| optax | **本阶段新增**：梯度优化与正则化 | optax提供模块化的梯度变换器（`optax.adam`、`optax.sgd`）和正则化器（`optax.add_weight_decay`），直接支撑A15.3的离散化权重更新实现、A15.4的自适应学习率调度、A15.5的EWC正则化作为自定义梯度变换器集成 |
| NumPy | 矩阵运算 | Fisher对角计算、权重变化范数 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V15.1 | 权重更新方程正确性 | $\tau_W \dot{W}^l = -(\partial f^l/\partial W^l) \cdot \delta^l$ 与自由能梯度一致 | JAX自动微分对比验证 |
| V15.2 | Hebbian学习对应 | 线性模型下 $\dot{W}^l \propto r^l \cdot (\delta^l)^T$ 与Hebbian规则一致 | 线性模型数值验证 |
| V15.3 | 精度加权权重更新 | A15.2与A15.1在精度加权框架下等价 | 代数验证 |
| V15.4 | 离散化正确性 | A15.3与连续动力学在 $\eta \to 0$ 时一致 | 小学习率极限验证 |
| V15.5 | 自由能单调递减 | 权重更新后 $F(W_{t+1}) \leq F(W_t)$（在表征收敛前提下） | 多步权重更新自由能监控 |
| V15.6 | 学习率-精度反比关系 | $\eta_l \propto \Pi_l^{-1}$（A15.4） | 不同精度层的学习率对比 |
| V15.7 | 自适应学习率 | 精度更新后学习率自适应调整 | 精度变化场景测试 |
| V15.8 | EWC正则化效果 | EWC正则化减少对旧任务的遗忘 | 持续学习场景测试 |
| V15.9 | EWC与L2的退化关系 | $\lambda_{\text{EWC}} \cdot F_{\text{diag}} = \lambda$ 且 $W_{\text{old}} = 0$ 时EWC退化为L2 | 特殊参数数值验证 |
| V15.10 | Fisher信息正确性 | 蒙特卡洛Fisher对角与解析解在容差内一致（高斯模型） | 高斯模型解析对比 |
| V15.11 | 时间尺度分离 | $\tau_W / \tau_r \geq 10$ | 参数范围验证 |
| V15.12 | WeightUpdate组合 | `update.perception` 类型为 `PerceptionUpdate` | 类型检查 |
| V15.13 | 非法输入拒绝 | $\tau_W \leq 0$、$\eta \leq 0$、$\lambda_{\text{EWC}} < 0$、$\tau_W \leq \tau_r$ 等抛出 `ValueError` | 异常测试 |
| V15.14 | optax集成验证 | 使用optax优化器实现权重更新，收敛行为与手动实现一致 | optax数值对比 |
| V15.15 | 权重更新不破坏表征 | 权重更新后表征重新收敛，自由能不高于更新前 | 权重更新+感知推断联合测试 |

---

> **阶段15完** | 第二层·工程实现层的前五个阶段（11-15）已完成。下一步：阶段16将从本阶段的权重更新规则和阶段14的表征更新规则出发，展开完整的预测编码算法集成。
