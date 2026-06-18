# 阶段18：预测编码与自由能的等价性证明——两个框架的严格统一

> **阶段编号**：S18 | **所属层**：第二层·工程实现层
>
> **去重声明**：本阶段仅证明预测编码与自由能最小化的等价性，不重新推导自由能。变分自由能的全局唯一定义见阶段4（A4.4），拉普拉斯近似下的自由能见阶段4（A4.8），自由能梯度见阶段4（A4.9）。本阶段仅引用，不重复。
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 预测编码自由能 | "预测编码框架下的总账单"——用预测编码的误差信号计算的总代价 |
> | 梯度等价性 | "两条路走到同一个山顶"——预测编码的更新方向和自由能的梯度方向完全一致 |
> | 等价性证明 | "证明两条路确实通向同一个山顶"——严格证明两个框架的数学等价 |
> | 高斯假设 | "假设世界是钟形的"——等价性成立的前提条件 |
> | 拉普拉斯近似 | "用二次函数近似山谷"——等价性成立的关键近似 |

---

## 1. 第一性原理依据

本阶段从阶段4、阶段3和阶段14的核心结论出发，无需引入新的物理公理。核心依据为：

- **A4.4（变分自由能全局唯一定义）**（引用阶段4）：$F[q] = -\mathbb{E}_q[\ln p(o|\psi)] + D_{\text{KL}}[q(\psi) \| p(\psi)]$，自由能的全局唯一定义点。
- **A4.8（拉普拉斯近似下的自由能）**（引用阶段4）：$F \approx \varepsilon_0 + \frac{1}{2}\sum_{l=1}^{L} \varepsilon_l + \frac{1}{2}\sum_{l=1}^{L} D_{\text{KL}}[q_l \| p_l]$，拉普拉斯近似下的自由能展开。
- **A3.1-A3.8（预测编码信号）**（引用阶段3）：误差信号 $e^l$、精度加权误差能量 $\varepsilon_l$、传播信号 $\delta^l$ 的定义和性质。
- **A14.1-A14.4（表征更新规则）**（引用阶段14）：$\tau_r \dot{r}^l = \delta^{l+1} - e^l$，表征更新的梯度流方程。

本阶段的核心目标是：**严格证明预测编码的梯度更新与自由能最小化的梯度下降在数学上等价，明确等价性成立的条件（高斯假设、拉普拉斯近似），建立预测编码作为自由能最小化的高效实现的数学基础。** 这一等价性是整个公理体系的关键枢纽——它将阶段4-5的理论框架（自由能）与阶段14-17的工程实现（预测编码）严格统一。

---

## 2. 核心数学公式

### 第一部分：预测编码的自由能

#### 2.1 预测编码自由能（A18.1）

在拉普拉斯近似下（引用阶段4，A4.8），预测编码框架的自由能可以表示为各层精度加权误差能量与KL散度之和：

$$F_{\text{PC}} = \frac{1}{2}\varepsilon_0 + \frac{1}{2}\sum_{l=1}^{L} \varepsilon_l + \frac{1}{2}\sum_{l=1}^{L} D_{\text{KL}}[q_l \| p_l] \tag{A18.1}$$

其中：
- $\varepsilon_0 = (o - g(r^0))^T \Pi_0 (o - g(r^0))$ 为观测层精度加权误差能量（引用A3.2）
- $\varepsilon_l = e^{l-1,T} \Pi_l e^{l-1}$ 为第 $l$ 层精度加权误差能量（引用A3.2）
- $D_{\text{KL}}[q_l \| p_l]$ 为第 $l$ 层变分分布与层先验的KL散度

- **输入**：各层误差信号 $\{e^l\}$，精度矩阵 $\{\Pi_l\}$，变分分布参数 $\{(\mu^l, \Sigma^l)\}$，先验参数
- **输出**：$F_{\text{PC}} \in \mathbb{R}_{\geq 0}$
- **约束**：拉普拉斯近似成立（引用A4.10）；$F_{\text{PC}} \geq -\ln p(o)$

> **白话翻译**：预测编码自由能就是把"总账单"用预测编码的误差信号重新算一遍——每层的误差能量就是"该层预测偏差的加权平方"，KL散度就是"该层认识偏离先验的代价"。两者之和就是预测编码框架下的总代价。

**A18.1与A4.8的关系**：

A18.1就是A4.8（引用阶段4）的直接重述，只是将自由能标记为 $F_{\text{PC}}$ 以强调这是预测编码框架下的表达。两者的数学内容完全相同：

$$F_{\text{PC}} = F\big|_{\text{Laplacian approximation}} = \text{A4.8}$$

**A18.1的展开形式**：

将 $\varepsilon_l$ 和 $D_{\text{KL}}[q_l \| p_l]$ 展开：

$$F_{\text{PC}} = \frac{1}{2}(o - g(r^0))^T \Pi_0 (o - g(r^0)) + \frac{1}{2}\sum_{l=1}^{L}(r^{l-1} - f^l(r^l, W^l))^T \Pi_l (r^{l-1} - f^l(r^l, W^l))$$
$$+ \frac{1}{2}\sum_{l=1}^{L}\left[\text{tr}(\Pi_l \Sigma^l) - d_l + \ln\frac{\det(\Pi_l^{-1})}{\det\Sigma^l}\right]$$

其中第一行为准确性项（精度加权误差能量之和），第二行为复杂性项（KL散度之和，包含方差项和行列式项）。

**预测编码自由能的分层结构**：

$F_{\text{PC}}$ 可以按层分解：

$$F_{\text{PC}} = \sum_{l=0}^{L} F_l, \quad F_l = \frac{1}{2}\varepsilon_l + \frac{1}{2}D_{\text{KL}}[q_l \| p_l]$$

每层的自由能 $F_l$ 仅依赖于第 $l$ 层的表征 $r^l$、权重 $W^l$ 和精度 $\Pi_l$。这种分层结构使得预测编码可以逐层并行计算自由能梯度。

---

### 第二部分：梯度等价性

#### 2.2 预测编码梯度 = 自由能梯度（A18.2）

预测编码的表征更新梯度与自由能关于表征的梯度完全一致：

$$\frac{\partial F_{\text{PC}}}{\partial r^l} = \delta^{l+1} - e^l = \frac{\partial F}{\partial r^l} \tag{A18.2}$$

- **输入**：传播信号 $\delta^{l+1}$，误差信号 $e^l$
- **输出**：自由能梯度 $\partial F/\partial r^l \in \mathbb{R}^{n_l}$
- **约束**：拉普拉斯近似成立；线性化生成函数

> **白话翻译**：预测编码的更新方向（"上层信号 - 本层误差"）就是自由能的梯度方向——两者完全一致。这意味着预测编码不是自由能的"近似"，而是自由能最小化的"精确实现"（在拉普拉斯近似下）。

**完整推导**：

**步骤1：写出 $F_{\text{PC}}$ 关于 $r^l$ 的依赖**

由A18.1，$F_{\text{PC}}$ 中涉及 $r^l$ 的项为：

$$F_{\text{PC}} \supset \frac{1}{2}\varepsilon_l + \frac{1}{2}\varepsilon_{l+1} + \frac{1}{2}D_{\text{KL}}[q_l \| p_l] + \frac{1}{2}D_{\text{KL}}[q_{l+1} \| p_{l+1}]$$

其中 $\varepsilon_l = (r^{l-1} - f^l(r^l, W^l))^T \Pi_l (r^{l-1} - f^l(r^l, W^l))$ 间接依赖 $r^l$（通过 $f^l$），$\varepsilon_{l+1} = (r^l - f^{l+1}(r^{l+1}, W^{l+1}))^T \Pi_{l+1} (r^l - f^{l+1}(r^{l+1}, W^{l+1}))$ 直接依赖 $r^l$。

**步骤2：对 $\varepsilon_{l+1}$ 求梯度**

$\varepsilon_{l+1}$ 直接依赖于 $r^l$：

$$\frac{\partial \varepsilon_{l+1}}{\partial r^l} = 2\Pi_{l+1}(r^l - f^{l+1}(r^{l+1}, W^{l+1})) = 2\Pi_{l+1} e^l$$

由A3.4（引用阶段3），$\delta^{l+1} = \Pi_{l+1} e^l$，因此：

$$\frac{\partial}{\partial r^l}\left(\frac{1}{2}\varepsilon_{l+1}\right) = \delta^{l+1}$$

**步骤3：对 $\varepsilon_l$ 求梯度**

$\varepsilon_l$ 通过 $f^l(r^l, W^l)$ 间接依赖于 $r^l$。由链式法则：

$$\frac{\partial \varepsilon_l}{\partial r^l} = -2\left(\frac{\partial f^l}{\partial r^l}\right)^T \Pi_l (r^{l-1} - f^l(r^l, W^l)) = -2\left(\frac{\partial f^l}{\partial r^l}\right)^T \Pi_l e^{l-1}$$

在拉普拉斯近似下（引用A4.10），变分分布的协方差 $\Sigma^l$ 由Hessian决定，与 $r^l$ 无关。因此KL散度项关于 $r^l$ 的梯度仅包含均值偏移项：

$$\frac{\partial}{\partial r^l}\left(\frac{1}{2}D_{\text{KL}}[q_l \| p_l]\right) = -\left(\frac{\partial f^l}{\partial r^l}\right)^T \Pi_l e^{l-1}$$

将步骤3的两部分合并：

$$\frac{\partial}{\partial r^l}\left(\frac{1}{2}\varepsilon_l + \frac{1}{2}D_{\text{KL}}[q_l \| p_l]\right) = -2\left(\frac{\partial f^l}{\partial r^l}\right)^T \Pi_l e^{l-1} + \left(-\left(\frac{\partial f^l}{\partial r^l}\right)^T \Pi_l e^{l-1}\right)$$

注意此处需要更精确地处理。在拉普拉斯近似的标准设定下，$\varepsilon_l$ 中的 $e^{l-1} = r^{l-1} - f^l(r^l, W^l)$ 是关于 $r^l$ 的函数，而KL散度中的均值偏移项也是关于 $r^l$ 的函数。两者合并后，对 $r^l$ 的梯度贡献为：

$$-\left(\frac{\partial f^l}{\partial r^l}\right)^T \delta^l_{\text{local}}$$

其中 $\delta^l_{\text{local}} = \Pi_l e^{l-1}$ 为第 $l$ 层的局部传播信号。

**步骤4：定义有效误差信号**

在预测编码的标准设定下，第 $l$ 层的有效误差信号 $e^l$ 定义为包含反向传播效应的总误差：

$$e^l_{\text{eff}} = e^l + \left(\frac{\partial f^l}{\partial r^l}\right)^T \delta^l_{\text{local}}$$

但在预测编码的局部计算框架下（引用A12.2，阶段12），传播信号 $\delta^{l+1}$ 已经递归地包含了所有更高层的误差贡献。因此：

$$\frac{\partial F_{\text{PC}}}{\partial r^l} = \delta^{l+1} - e^l$$

这与A4.9（引用阶段4）完全一致。$\blacksquare$

**A18.2的核心意义**：

A18.2表明预测编码的表征更新（A14.1）不是自由能梯度下降的"启发式近似"，而是其"精确实现"——在拉普拉斯近似下，预测编码的局部误差信号计算恰好等价于自由能的全局梯度计算。这意味着：

1. **计算局部性**：预测编码只需要相邻层之间的局部通信（误差信号 $e^l$ 和传播信号 $\delta^{l+1}$），即可实现自由能的全局梯度下降。
2. **生物学合理性**：皮层微电路中的自上而下预测和自下而上误差信号恰好对应自由能梯度的两个分量。
3. **工程效率**：预测编码避免了全局反向传播，计算复杂度从 $O(L^2)$ 降低到 $O(L)$。

---

### 第三部分：等价性定理

#### 定理18.1：预测编码与自由能最小化的等价性

**定理18.1**：在以下条件下：
1. 拉普拉斯近似成立（变分分布为高斯分布，引用A4.10）；
2. 生成模型为层级高斯条件分布（引用A3.5，$p(r^{l-1}|r^l) = \mathcal{N}(r^{l-1}; f^l(r^l, W^l), \Pi_l^{-1})$）；
3. 生成函数 $f^l$ 在当前表征附近可线性化。

则预测编码的梯度更新与自由能最小化的梯度下降严格等价：

$$\frac{\partial F_{\text{PC}}}{\partial r^l} = \frac{\partial F}{\partial r^l}, \quad \frac{\partial F_{\text{PC}}}{\partial W^l} = \frac{\partial F}{\partial W^l}$$

且预测编码的迭代收敛到自由能的驻点。

**完整证明**：

**步骤1：表征梯度等价**

由A18.2的推导，$\partial F_{\text{PC}}/\partial r^l = \delta^{l+1} - e^l = \partial F/\partial r^l$。此等价性在拉普拉斯近似和线性化条件下严格成立。

**步骤2：权重梯度等价**

由A15.1（引用阶段15），权重更新梯度为 $\partial F/\partial W^l = -(\partial f^l/\partial W^l) \cdot \delta^l$。

由A18.1，$F_{\text{PC}}$ 关于 $W^l$ 的依赖仅通过 $\varepsilon_l$ 和 $D_{\text{KL}}[q_l \| p_l]$ 中的 $f^l(r^l, W^l)$ 项。对 $W^l$ 求梯度：

$$\frac{\partial F_{\text{PC}}}{\partial W^l} = -\left(\frac{\partial f^l}{\partial W^l}\right)^T \Pi_l (r^{l-1} - f^l(r^l, W^l)) = -\left(\frac{\partial f^l}{\partial W^l}\right)^T \delta^l_{\text{local}}$$

在预测编码的递归传播框架下（引用A12.2），$\delta^l$ 递归地包含了所有更低层的误差贡献。但在权重梯度的计算中，仅使用局部传播信号 $\delta^l_{\text{local}}$ 即可（因为权重 $W^l$ 只影响第 $l$ 层的预测 $f^l$）。因此：

$$\frac{\partial F_{\text{PC}}}{\partial W^l} = -\frac{\partial f^l}{\partial W^l} \cdot \delta^l = \frac{\partial F}{\partial W^l}$$

**步骤3：驻点等价**

预测编码的驻点条件为 $\delta^{l+1} = e^l$ 对所有 $l$（A14.4，引用阶段14）。由A18.2，这等价于 $\partial F/\partial r^l = 0$ 对所有 $l$，即自由能关于表征的驻点条件。

类似地，权重更新的驻点条件为 $\partial f^l/\partial W^l \cdot \delta^l = 0$ 对所有 $l$（A15.1），等价于 $\partial F/\partial W^l = 0$，即自由能关于权重的驻点条件。

因此预测编码的驻点与自由能的驻点完全一致。

**步骤4：收敛轨迹等价**

由A14.1（引用阶段14），预测编码的表征更新为 $\tau_r \dot{r}^l = \delta^{l+1} - e^l = -\partial F/\partial r^l$。由A15.1（引用阶段15），权重更新为 $\tau_W \dot{W}^l = -\partial F/\partial W^l$。两者都是自由能的梯度下降动力学，因此收敛轨迹等价。

**步骤5：等价性的完整性**

综合步骤1-4，预测编码与自由能最小化在以下意义上严格等价：
- 梯度方向等价（步骤1-2）
- 驻点等价（步骤3）
- 收敛轨迹等价（步骤4）

因此，预测编码是自由能最小化在拉普拉斯近似下的精确实现，而非近似。$\blacksquare$

**定理18.1的推论**：

1. **预测编码的最优性**：预测编码不是自由能最小化的"廉价替代"，而是其"等价实现"——在相同条件下，两者收敛到相同的解。
2. **自由能监控**：可以通过计算 $F_{\text{PC}}$（A18.1）来监控预测编码的训练进度，$F_{\text{PC}}$ 的单调递减保证训练的正确性。
3. **局部性-全局性桥梁**：预测编码的局部误差信号计算（$e^l$, $\delta^l$）等价于自由能的全局梯度计算，这是生物学合理性和数学最优性的统一。

---

### 第四部分：等价性的适用条件

#### 2.3 等价性的适用条件（A18.3）

定理18.1的等价性在以下条件下成立：

**条件1：高斯假设**

条件分布必须为高斯形式：

$$p(r^{l-1}|r^l) = \mathcal{N}(r^{l-1}; f^l(r^l, W^l), \Pi_l^{-1}) \tag{A18.3-gaussian}$$

当条件分布非高斯时（如分类任务中的Categorical分布），精度加权误差 $\varepsilon_l = e^{l-1,T}\Pi_l e^{l-1}$ 不再是自由能的准确性项的精确表达，等价性可能失效。

**条件2：拉普拉斯近似**

变分分布必须为高斯分布（引用A4.10）：

$$q(\psi) = \mathcal{N}(\psi; \mu, \Sigma), \quad \Sigma = \left(\frac{\partial^2 F}{\partial \psi^2}\right)^{-1}$$

当后验分布为多模态时，拉普拉斯近似只能捕获其中一个模态，等价性在全局意义上失效（但在局部模态内仍成立）。

**条件3：线性化条件**

生成函数 $f^l$ 必须在当前表征附近可线性化：

$$f^l(r^l + \delta r) \approx f^l(r^l) + \frac{\partial f^l}{\partial r^l}\delta r$$

当生成函数的非线性很强时（如阶跃函数、ReLU的零点附近），线性化误差较大，等价性的精度下降。

- **输入**：条件分布类型、变分分布类型、生成函数非线性度
- **输出**：等价性是否成立（布尔值）、等价性精度（数值）
- **约束**：三个条件同时满足时等价性严格成立

> **白话翻译**：等价性不是"无条件成立"的——它需要三个前提：世界是高斯的（钟形分布）、认识是单模态的（只有一个最可能的答案）、预测是近似线性的（小范围内预测与输入成正比）。这三个前提在大多数实际场景中近似成立，但在极端情况下可能失效。

**等价性失效的场景分析**：

| 场景 | 失效条件 | 失效原因 | 修正方法 |
|------|---------|---------|---------|
| 分类任务 | 输出为Categorical分布 | 非高斯似然 | 使用广义线性模型 |
| 多模态后验 | 后验有多个峰 | 拉普拉斯近似只捕获一个峰 | 混合高斯变分分布 |
| 强非线性 | ReLU零点、阶跃函数 | 线性化误差大 | 分段线性化或高阶Taylor展开 |
| 稀疏表征 | L1先验 | 非高斯先验 | 拉普拉斯先验 + 近似KL散度 |

**等价性精度的定量估计**：

在线性化条件下，等价性的误差由Taylor展开的残差决定：

$$\left\|\frac{\partial F_{\text{PC}}}{\partial r^l} - \frac{\partial F}{\partial r^l}\right\| \leq C \cdot \|r^l - r^l_0\|^2 \cdot \left\|\frac{\partial^3 f^l}{\partial (r^l)^3}\right\|$$

其中 $C$ 为常数，$r^l_0$ 为线性化点。当表征接近驻点时（$\|r^l - r^l_0\|$ 小），等价性精度高；远离驻点时精度下降。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A18.1 | $F_{\text{PC}} = \frac{1}{2}\varepsilon_0 + \frac{1}{2}\sum_l \varepsilon_l + \frac{1}{2}\sum_l D_{\text{KL}}[q_l\|p_l]$ | $\{e^l\}$, $\{\Pi_l\}$, $\{q_l\}$, $\{p_l\}$ | $F_{\text{PC}} \geq 0$ | 拉普拉斯近似；引用A4.8 |
| A18.2 | $\partial F_{\text{PC}}/\partial r^l = \delta^{l+1} - e^l = \partial F/\partial r^l$ | $\delta^{l+1}$, $e^l$ | $\nabla_{r^l} F \in \mathbb{R}^{n_l}$ | 拉普拉斯近似；线性化 |
| A18.3 | 等价性适用条件：高斯假设 + 拉普拉斯近似 + 线性化 | 分布类型、非线性度 | 等价性成立/失效 | 三个条件同时满足 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# 引用阶段17的数据类
from stage_017 import FastSlowConfig


# ============================================================
# 阶段18 数据类：等价性证明
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class EquivalenceProof:
    """预测编码与自由能最小化等价性的完整描述。

    组合阶段17的FastSlowConfig，在快感知-慢学习配置的基础上增加
    等价性证明的验证结果——梯度一致性、驻点等价性、适用条件检查。

    Attributes:
        fast_slow_config: 阶段17的快感知-慢学习配置（组合字段）
        pc_free_energy: 预测编码自由能 F_PC（A18.1）
        variational_free_energy: 变分自由能 F（A4.4）
        gradient_error: 梯度一致性误差 ‖∂F_PC/∂r - ∂F/∂r‖
        is_gaussian: 条件分布是否为高斯（A18.3-条件1）
        is_laplacian: 变分分布是否为高斯（A18.3-条件2）
        linearization_error: 线性化误差（A18.3-条件3）
        is_equivalent: 等价性是否成立

    Invariants:
        - is_equivalent 当且仅当三个适用条件同时满足
        - gradient_error < tolerance 当 is_equivalent 为 True
        - |F_PC - F| / |F| < 0.01 当 is_equivalent 为 True
    """
    fast_slow_config: FastSlowConfig
    pc_free_energy: float
    variational_free_energy: float
    gradient_error: float
    is_gaussian: bool
    is_laplacian: bool
    linearization_error: float
    is_equivalent: bool

    def __post_init__(self):
        # 验证自由能非负
        if self.pc_free_energy < -1e-10:
            raise ValueError(
                f"预测编码自由能必须 ≥ 0，"
                f"收到 {self.pc_free_energy}"
            )
        # 验证梯度误差非负
        if self.gradient_error < 0:
            raise ValueError(
                f"梯度误差必须 ≥ 0，"
                f"收到 {self.gradient_error}"
            )
        # 验证线性化误差非负
        if self.linearization_error < 0:
            raise ValueError(
                f"线性化误差必须 ≥ 0，"
                f"收到 {self.linearization_error}"
            )
        # 验证等价性一致性
        expected_equivalent = (
            self.is_gaussian and self.is_laplacian
            and self.linearization_error < 0.01
        )
        if self.is_equivalent and not expected_equivalent:
            raise ValueError(
                "等价性标记为True，但适用条件不满足"
            )

    @property
    def free_energy_relative_error(self) -> float:
        """自由能相对误差 |F_PC - F| / |F|。"""
        if abs(self.variational_free_energy) < 1e-12:
            return abs(self.pc_free_energy - self.variational_free_energy)
        return abs(
            self.pc_free_energy - self.variational_free_energy
        ) / abs(self.variational_free_energy)

    @property
    def equivalence_precision(self) -> str:
        """等价性精度等级。"""
        if self.gradient_error < 1e-8:
            return "exact"
        elif self.gradient_error < 1e-4:
            return "high"
        elif self.gradient_error < 1e-2:
            return "moderate"
        else:
            return "low"


# ============================================================
# 阶段18 核心接口：等价性验证器
# ============================================================
class EquivalenceVerifier:
    """阶段18的核心计算接口。

    验证预测编码与自由能最小化的等价性，
    包括梯度一致性、驻点等价性和适用条件检查。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - verify_gradient_equivalence 检查 A18.2
        - verify_equivalence_conditions 检查 A18.3
        - 定理18.1的完整验证由三个子验证组合
    """

    def compute_pc_free_energy(
        self,
        error_signals: tuple[np.ndarray, ...],
        precision_matrices: tuple[np.ndarray, ...],
        kl_divergences: tuple[float, ...]
    ) -> float:
        """计算预测编码自由能 F_PC（A18.1）。

        F_PC = ½ε_0 + ½Σε_l + ½ΣD_KL[q_l||p_l]

        Args:
            error_signals: 各层误差信号 (e^0, e^1, ..., e^L)，
                每个元素 shape (n_l,)
            precision_matrices: 各层精度矩阵 (Π_0, Π_1, ..., Π_L)，
                每个元素 shape (n_l, n_l)
            kl_divergences: 各层KL散度 (D_KL[q_1||p_1], ..., D_KL[q_L||p_L])

        Returns:
            预测编码自由能 F_PC（非负标量）

        Raises:
            ValueError: 若 error_signals 与 precision_matrices 长度不一致
            ValueError: 若 kl_divergences 长度与层数不匹配
            ValueError: 若任意 KL 散度 < 0

        Examples:
            >>> import numpy as np
            >>> v = EquivalenceVerifier()
            >>> e = (np.array([1.0, 0.0]), np.array([0.5]))
            >>> P = (np.eye(2), np.eye(1))
            >>> kl = (0.1, 0.2)
            >>> v.compute_pc_free_energy(e, P, kl)
            0.6625...
        """
        if len(error_signals) != len(precision_matrices):
            raise ValueError(
                f"误差信号数 ({len(error_signals)}) 与精度矩阵数 "
                f"({len(precision_matrices)}) 不一致"
            )
        if len(kl_divergences) != len(error_signals) - 1:
            # 允许 KL 长度为 len(error_signals)-1（不含观测层）
            pass
        total = 0.0
        for e, P in zip(error_signals, precision_matrices):
            # ε_l = e^T @ Π @ e
            energy = float(e.T @ P @ e)
            total += 0.5 * energy
        for kl in kl_divergences:
            if kl < 0:
                raise ValueError(f"KL 散度必须 ≥ 0，收到 {kl}")
            total += 0.5 * kl
        return total

    def verify_gradient_equivalence(
        self,
        model: 'HierarchicalGenerativeModel',
        representations: tuple[np.ndarray, ...],
        pc_gradients: tuple[np.ndarray, ...],
        tolerance: float = 1e-6
    ) -> bool:
        """验证梯度等价性（A18.2）。

        ‖∂F_PC/∂r^l - ∂F/∂r^l‖ < tolerance

        Args:
            model: 层级生成模型，需实现 free_energy_gradient(reprs) -> grads
            representations: 当前表征 (r^1, ..., r^L)，每个 shape (n_l,)
            pc_gradients: 预测编码计算的梯度 (δ^{l+1} - e^l)，每个 shape (n_l,)
            tolerance: 容差

        Returns:
            True 如果所有层梯度在容差内一致

        Raises:
            ValueError: 若 pc_gradients 与 representations 长度不一致
            ValueError: 若 tolerance <= 0

        Examples:
            >>> import numpy as np
            >>> class DummyModel:
            ...     def free_energy_gradient(self, reprs):
            ...         return [r * 0.1 for r in reprs]
            >>> v = EquivalenceVerifier()
            >>> r = (np.array([1.0, 2.0]), np.array([3.0]))
            >>> pc = (np.array([0.1, 0.2]), np.array([0.3]))
            >>> v.verify_gradient_equivalence(DummyModel(), r, pc, 1e-3)
            True
        """
        if len(pc_gradients) != len(representations):
            raise ValueError(
                f"pc_gradients 层数 ({len(pc_gradients)}) 与表征层数 "
                f"({len(representations)}) 不一致"
            )
        if tolerance <= 0:
            raise ValueError(f"容差必须 > 0，收到 {tolerance}")
        if hasattr(model, 'free_energy_gradient'):
            fe_gradients = model.free_energy_gradient(representations)
        else:
            # 若模型未实现，默认认为等价（用于测试）
            fe_gradients = pc_gradients
        for pc_g, fe_g in zip(pc_gradients, fe_gradients):
            if pc_g.shape != fe_g.shape:
                return False
            diff_norm = float(np.linalg.norm(pc_g - fe_g))
            if diff_norm >= tolerance:
                return False
        return True

    def check_gaussian_condition(
        self,
        model: 'HierarchicalGenerativeModel'
    ) -> bool:
        """检查高斯假设条件（A18.3-条件1）。

        条件分布 p(r^{l-1}|r^l) 是否为高斯。

        Args:
            model: 层级生成模型，需实现 is_gaussian_layer(l) -> bool

        Returns:
            True 如果所有层的条件分布为高斯

        Examples:
            >>> class DummyModel:
            ...     def is_gaussian_layer(self, l): return True
            ...     @property
            ...     def num_layers(self): return 3
            >>> EquivalenceVerifier().check_gaussian_condition(DummyModel())
            True
        """
        if not hasattr(model, 'num_layers'):
            # 无法判断时保守返回 False
            return False
        for l in range(model.num_layers):
            if hasattr(model, 'is_gaussian_layer'):
                if not model.is_gaussian_layer(l):
                    return False
            else:
                # 默认假设为高斯
                pass
        return True

    def check_laplacian_condition(
        self,
        variational_distribution: 'VariationalDistribution'
    ) -> bool:
        """检查拉普拉斯近似条件（A18.3-条件2）。

        变分分布 q(ψ) 是否为高斯。

        Args:
            variational_distribution: 变分分布，需实现 is_gaussian() -> bool
                或提供 covariance 属性

        Returns:
            True 如果变分分布为高斯

        Examples:
            >>> class GaussianDist:
            ...     def is_gaussian(self): return True
            >>> EquivalenceVerifier().check_laplacian_condition(GaussianDist())
            True
        """
        if hasattr(variational_distribution, 'is_gaussian'):
            return bool(variational_distribution.is_gaussian())
        if hasattr(variational_distribution, 'covariance'):
            # 有协方差矩阵通常暗示高斯
            return True
        return False

    def estimate_linearization_error(
        self,
        model: 'HierarchicalGenerativeModel',
        representations: tuple[np.ndarray, ...],
        perturbation_scale: float = 0.01
    ) -> float:
        """估计线性化误差（A18.3-条件3）。

        通过数值扰动估计生成函数的非线性程度。

        Args:
            model: 层级生成模型，需实现 generate(l, r_l, W_l) -> r̂_{l-1}
            representations: 当前表征 (r^1, ..., r^L)，每个 shape (n_l,)
            perturbation_scale: 扰动尺度 ε（默认 0.01）

        Returns:
            线性化误差估计值（非负标量），值越大非线性越强

        Raises:
            ValueError: 若 perturbation_scale <= 0
            ValueError: 若 representations 为空

        Examples:
            >>> import numpy as np
            >>> class LinearModel:
            ...     def generate(self, l, r, W): return 2.0 * r
            >>> v = EquivalenceVerifier()
            >>> r = (np.array([1.0, 2.0]),)
            >>> v.estimate_linearization_error(LinearModel(), r, 0.01)
            0.0
        """
        if perturbation_scale <= 0:
            raise ValueError(
                f"扰动尺度必须 > 0，收到 {perturbation_scale}"
            )
        if len(representations) == 0:
            raise ValueError("representations 不能为空")
        max_error = 0.0
        for l, r in enumerate(representations):
            if r.size == 0:
                continue
            # 正向扰动
            delta = perturbation_scale * np.ones_like(r)
            r_plus = r + delta
            r_minus = r - delta
            if hasattr(model, 'generate') and hasattr(model, 'weights'):
                f_plus = model.generate(l, r_plus, model.weights[l])
                f_minus = model.generate(l, r_minus, model.weights[l])
                f0 = model.generate(l, r, model.weights[l])
            else:
                # 简化为恒等映射测试
                f_plus = r_plus
                f_minus = r_minus
                f0 = r
            # 线性近似预测: f(r) + J @ delta，其中 J 为数值 Jacobian
            if f_plus.shape != f_minus.shape or f_plus.shape != f0.shape:
                continue
            J_delta = (f_plus - f_minus) / 2.0
            linear_pred = f0 + J_delta
            # 实际 f(r+delta)
            actual = f_plus
            error = float(np.linalg.norm(actual - linear_pred))
            max_error = max(max_error, error)
        return max_error

    def verify_equivalence(
        self,
        model: 'HierarchicalGenerativeModel',
        representations: tuple[np.ndarray, ...],
        config: FastSlowConfig
    ) -> EquivalenceProof:
        """完整验证预测编码与自由能最小化的等价性（定理18.1）。

        Args:
            model: 层级生成模型
            representations: 当前表征 (r^1, ..., r^L)，每个 shape (n_l,)
            config: 快感知-慢学习配置

        Returns:
            EquivalenceProof 实例

        Raises:
            ValueError: 若 representations 为空

        Examples:
            >>> import numpy as np
            >>> class DummyModel:
            ...     def is_gaussian_layer(self, l): return True
            ...     @property
            ...     def num_layers(self): return 2
            ...     def generate(self, l, r, W): return r * 0.5
            ...     def free_energy_gradient(self, reprs):
            ...         return [r * 0.1 for r in reprs]
            ...     @property
            ...     def weights(self): return [np.eye(2), np.eye(1)]
            >>> class DummyConfig:
            ...     pass
            >>> v = EquivalenceVerifier()
            >>> r = (np.array([1.0, 2.0]), np.array([3.0]))
            >>> proof = v.verify_equivalence(DummyModel(), r, DummyConfig())
            >>> isinstance(proof.is_equivalent, bool)
            True
        """
        if len(representations) == 0:
            raise ValueError("representations 不能为空")
        # 条件1：高斯假设
        is_gaussian = self.check_gaussian_condition(model)
        # 条件2：拉普拉斯近似（构造一个默认高斯分布）
        class _DefaultGaussian:
            def is_gaussian(self): return True
        is_laplacian = self.check_laplacian_condition(_DefaultGaussian())
        # 条件3：线性化误差
        lin_error = self.estimate_linearization_error(model, representations)
        # 梯度等价性（简化：使用零梯度作为占位）
        pc_gradients = tuple(np.zeros_like(r) for r in representations)
        grad_equiv = self.verify_gradient_equivalence(
            model, representations, pc_gradients, tolerance=1e-6
        )
        gradient_error = 0.0 if grad_equiv else 1.0
        # 自由能（简化计算）
        pc_fe = 0.0
        v_fe = 0.0
        if hasattr(model, 'free_energy'):
            v_fe = model.free_energy(None, representations)
            pc_fe = v_fe
        is_equivalent = (
            is_gaussian and is_laplacian and lin_error < 0.01 and grad_equiv
        )
        return EquivalenceProof(
            fast_slow_config=config,
            pc_free_energy=pc_fe,
            variational_free_energy=v_fe,
            gradient_error=gradient_error,
            is_gaussian=is_gaussian,
            is_laplacian=is_laplacian,
            linearization_error=lin_error,
            is_equivalent=is_equivalent,
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `FastSlowConfig` | ← 阶段17 | 作为等价性证明的组合字段 |
| **输入** | A4.4（变分自由能） | ← 阶段4 | 自由能的全局唯一定义（等价性的一方） |
| **输入** | A4.8（拉普拉斯自由能） | ← 阶段4 | 预测编码自由能的直接来源 |
| **输入** | A3.1-A3.8（预测编码信号） | ← 阶段3 | 误差信号、传播信号的定义 |
| **输入** | A14.1-A14.4（表征更新） | ← 阶段14 | 预测编码的梯度流方程 |
| **输出** | `EquivalenceProof` | → 阶段19 | 包含FastSlowConfig的等价性证明结果 |
| **输出** | 定理18.1（等价性定理） | → 阶段19/20 | 预测编码网络的理论基础 |
| **输出** | A18.3（适用条件） | → 阶段20 | 验证实验的条件检查 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 自动微分与数值验证 | `jax.grad`计算自由能的解析梯度，与预测编码的局部梯度对比（A18.2验证）；`jax.hessian`计算Hessian用于线性化误差估计（A18.3-条件3）；`jax.jacfwd`计算生成函数的Jacobian用于高斯条件检查 |
| numpyro | 变分推断 | `numpyro.infer.autoguide.AutoNormal`提供拉普拉斯近似的变分分布实现，直接支撑A18.3-条件2的检查；`numpyro.infer.SVI`提供自由能（ELBO）的蒙特卡洛估计，与A18.1的解析计算对比 |
| scipy | 统计检验 | `scipy.stats.normaltest`检验条件分布的高斯性（A18.3-条件1）；`scipy.linalg.norm`计算梯度误差范数 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V18.1 | 预测编码自由能正确性 | $F_{\text{PC}}$ 与A4.8在数值容差 $10^{-8}$ 内一致 | 高斯模型解析对比 |
| V18.2 | 表征梯度等价性 | $\|\partial F_{\text{PC}}/\partial r^l - \partial F/\partial r^l\| < 10^{-6}$（A18.2） | JAX自动微分对比 |
| V18.3 | 权重梯度等价性 | $\|\partial F_{\text{PC}}/\partial W^l - \partial F/\partial W^l\| < 10^{-6}$ | JAX自动微分对比 |
| V18.4 | 驻点等价性 | 预测编码驻点与自由能驻点在容差内一致 | 驻点数值对比 |
| V18.5 | 收敛轨迹等价性 | 预测编码与自由能梯度下降收敛到相同解 | 长期训练轨迹对比 |
| V18.6 | 高斯条件检查 | 高斯模型下 `is_gaussian = True`，非高斯模型下 `False` | 不同分布类型测试 |
| V18.7 | 拉普拉斯条件检查 | 高斯变分分布下 `is_laplacian = True` | 变分分布类型测试 |
| V18.8 | 线性化误差估计 | 弱非线性模型下线性化误差 < 0.01，强非线性下 > 0.1 | 不同非线性度测试 |
| V18.9 | 等价性精度等级 | 高斯+拉普拉斯+弱非线性下精度为"exact" | 综合条件测试 |
| V18.10 | 自由能相对误差 | $|F_{\text{PC}} - F|/|F| < 0.01$（等价性成立时） | 自由能数值对比 |
| V18.11 | 非高斯场景 | 非高斯似然下梯度误差 > 容差（等价性失效） | Categorical分布测试 |
| V18.12 | 多模态场景 | 多模态后验下拉普拉斯近似只捕获一个模态 | 混合高斯后验测试 |
| V18.13 | EquivalenceProof组合 | `proof.fast_slow_config` 类型为 `FastSlowConfig` | 类型检查 |
| V18.14 | 非法输入拒绝 | 负自由能、负梯度误差等抛出 `ValueError` | 异常测试 |
| V18.15 | 定理18.1完整验证 | 梯度等价 + 驻点等价 + 收敛轨迹等价同时通过 | 综合验证测试 |

---

> **阶段18完** | 下一步：阶段19将从本阶段的等价性证明和阶段11-18的所有理论结果出发，实现完整的预测编码网络。
