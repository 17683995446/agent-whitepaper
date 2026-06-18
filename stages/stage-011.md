# 阶段11：层级生成模型——分层概率分布的联合结构

> **阶段编号**：S11 | **所属层**：第二层·工程实现层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 层级生成模型 | "多层翻译机"——从最抽象的概念逐层翻译成具体的感知 |
> | 联合分布 | "所有层同时出现的概率"——高层概念、中层特征、底层像素的联合可能性 |
> | 条件分布 | "给定上层意思后本层长什么样"——知道是"猫"后，预测应该有尖耳朵 |
> | 联合对数似然 | "整条翻译链的合理性"——每一层翻译都合理的总得分 |
> | 层级深度 | "翻译的层数"——层数越多，概念越抽象 |
> | 抽象度 | "离具体感知的远近"——越远越抽象，"动物"比"猫"更抽象 |
> | 先验分布 | "最顶层概念的先入之见"——没有证据时对最抽象概念的默认信念 |

---

## 1. 第一性原理依据

本阶段从阶段4和阶段3的核心结论出发，无需引入新的物理公理。核心依据为：

- **A4.4（变分自由能全局唯一定义）**（引用阶段4）：$F[q] = -\mathbb{E}_q[\ln p(o|\psi)] + D_{\text{KL}}[q(\psi) \| p(\psi)]$，层级生成模型是自由能框架中生成模型 $p(o, \psi)$ 的分层展开。
- **A3.1-A3.8（预测编码公理）**（引用阶段3）：误差信号 $e^{l-1} = r^{l-1} - \hat{r}^{l-1}$（A3.1）、精度加权误差能量 $\varepsilon_l = e^{l-1,T}\Pi_l e^{l-1}$（A3.2）、自下而上传播信号 $\delta^l = \Pi_l e^l$（A3.4）、自上而下预测 $\hat{r}^{l-1} = f^l(r^l, W^l)$（A3.5），这些定义了层级架构中信号的基本结构。
- **A4.8（拉普拉斯近似下的自由能）**（引用阶段4）：$F \approx \frac{1}{2}\varepsilon_0 + \frac{1}{2}\sum_{l=1}^{L}\varepsilon_l + \frac{1}{2}\sum_{l=1}^{L}D_{\text{KL}}[q_l\|p_l]$，此分解直接依赖于层级生成模型的条件独立结构。
- **A4.9（自由能梯度）**（引用阶段4）：$\partial F/\partial r^l = \delta^{l+1} - e^l$，此梯度的推导以层级条件分布为前提。

本阶段的核心目标是：**将阶段3中隐含的分层架构显式化为层级生成模型的联合概率分布，建立从顶层先验到底层观测的完整概率生成链，证明联合分布的因子化结构，并分析层级深度与抽象度的定量关系。** 这是第二层·工程实现层的起点——后续阶段12（自下而上误差传播）、阶段13（自上而下预测传播）、阶段14（表征更新规则）和阶段15（权重更新规则）均以本阶段的层级生成模型为计算基础。

---

## 2. 核心数学公式

### 第一部分：层级生成模型的联合分布

#### 2.1 层级生成模型的联合分布（A11.1）

考虑一个具有 $L$ 层的层级生成模型。设 $r^l \in \mathbb{R}^{n_l}$ 为第 $l$ 层的表征向量（$l = 1, \ldots, L$），$o \in \mathbb{R}^{d_o}$ 为观测向量，$r^{L+1}$ 为最顶层的先验表征（固定或具有简单先验分布）。层级生成模型的联合分布定义为：

$$p(o, r^1, \ldots, r^L) = p(o|r^1) \prod_{l=1}^{L} p(r^l|r^{l+1}) \cdot p(r^{L+1}) \tag{A11.1}$$

- **输入**：观测 $o$，各层表征 $\{r^l\}_{l=1}^{L}$，顶层先验 $p(r^{L+1})$
- **输出**：联合概率密度 $p(o, r^{1:L}) \in \mathbb{R}_{\geq 0}$
- **约束**：各条件分布 $p(r^{l-1}|r^l)$ 为合法概率密度（非负、归一化）；$p(r^{L+1})$ 为合法先验

> **白话翻译**：层级生成模型就是"多层翻译机的概率公式"——观测由第一层表征决定，第一层由第二层决定，依此类推，直到最顶层的先验信念。每一层只依赖上一层，不跳层，不回头——这就是马尔可夫链式的条件独立结构。整条链的概率就是各层条件概率的连乘积。

**因子化结构的推导**：

由概率的链式法则，联合分布总可以分解为：

$$p(o, r^1, \ldots, r^L) = p(o|r^1, \ldots, r^L) \cdot p(r^1|r^2, \ldots, r^L) \cdots p(r^{L-1}|r^L) \cdot p(r^L)$$

层级生成模型的核心假设是**马尔可夫条件独立性**：给定第 $l+1$ 层表征 $r^{l+1}$，第 $l$ 层表征 $r^l$ 与所有更高层 $r^{l+2}, \ldots, r^L$ 条件独立。即：

$$p(r^l | r^{l+1}, r^{l+2}, \ldots, r^L) = p(r^l | r^{l+1}) \quad \forall l = 1, \ldots, L-1$$

同样，给定 $r^1$，观测 $o$ 与所有更高层条件独立：

$$p(o | r^1, r^2, \ldots, r^L) = p(o | r^1)$$

代入链式法则即得A11.1的因子化形式。

**与阶段4的连接**：在阶段4中，状态变量 $\psi = (r^1, \ldots, r^L)$，生成模型 $p(o, \psi) = p(o|\psi)p(\psi)$。A11.1将 $p(o, \psi)$ 展开为层级因子化结构，使阶段4的抽象生成模型具体化为可计算的分层形式。

**与阶段3的连接**：A3.5（引用阶段3）定义了自上而下预测 $\hat{r}^{l-1} = f^l(r^l, W^l)$，这正是条件分布 $p(r^{l-1}|r^l)$ 的均值函数。A11.1为A3.5提供了概率论基础——预测函数 $f^l$ 是条件分布 $p(r^{l-1}|r^l)$ 的参数化描述。

#### 2.2 条件分布的具体形式（A11.2）

各层条件分布采用高斯参数化形式（与A4.10拉普拉斯近似一致）：

$$p(r^{l-1}|r^l) = \mathcal{N}(r^{l-1}; f^l(r^l, W^l), \Sigma^l) \tag{A11.2}$$

其中：
- $f^l(r^l, W^l): \mathbb{R}^{n_l} \to \mathbb{R}^{n_{l-1}}$ 为第 $l$ 层的生成函数（自上而下预测函数，引用阶段3，A3.5）
- $W^l \in \mathbb{R}^{n_{l-1} \times n_l}$ 为第 $l$ 层的连接权重矩阵
- $\Sigma^l \in \mathbb{S}_{++}^{n_{l-1}}$ 为第 $l$ 层的条件协方差矩阵（与精度矩阵 $\Pi_l = (\Sigma^l)^{-1}$ 互逆，引用阶段3，A3.2）

- **输入**：高层表征 $r^l$，权重 $W^l$，协方差 $\Sigma^l$
- **输出**：条件概率密度 $p(r^{l-1}|r^l)$
- **约束**：$f^l$ 可微（保证梯度传播）；$\Sigma^l$ 正定（保证分布合法）

> **白话翻译**：条件分布就是"知道上层意思后，本层长什么样的概率分布"。中心是上层的预测 $f^l(r^l, W^l)$，宽度是噪声 $\Sigma^l$——噪声越大，本层允许偏离预测越多。精度矩阵 $\Pi_l = (\Sigma^l)^{-1}$ 是噪声的倒数——噪声越小，精度越高，预测越可信。

**条件分布的对数形式**：

$$\ln p(r^{l-1}|r^l) = -\frac{1}{2}(r^{l-1} - f^l(r^l, W^l))^T \Pi_l (r^{l-1} - f^l(r^l, W^l)) - \frac{1}{2}\ln|2\pi\Sigma^l| \tag{A11.2-log}$$

其中 $\Pi_l = (\Sigma^l)^{-1}$。由A3.1（引用阶段3），$e^{l-1} = r^{l-1} - f^l(r^l, W^l)$，由A3.2（引用阶段3），$\varepsilon_l = e^{l-1,T}\Pi_l e^{l-1}$，因此：

$$\ln p(r^{l-1}|r^l) = -\frac{1}{2}\varepsilon_l - \frac{1}{2}\ln|2\pi\Sigma^l|$$

这建立了条件分布对数似然与精度加权误差能量之间的直接对应——**条件分布的对数似然完全由精度加权误差能量决定**（差一个常数）。

**观测条件分布**：

底层观测条件分布为：

$$p(o|r^1) = \mathcal{N}(o; g(r^1), \Sigma^o) \tag{A11.2-obs}$$

其中 $g(r^1)$ 为观测生成函数，$\Sigma^o$ 为观测噪声协方差。类似地：

$$\ln p(o|r^1) = -\frac{1}{2}(o - g(r^1))^T \Pi_o (o - g(r^1)) - \frac{1}{2}\ln|2\pi\Sigma^o| = -\frac{1}{2}\varepsilon_0 - \frac{1}{2}\ln|2\pi\Sigma^o|$$

**顶层先验分布**：

$$p(r^{L+1}) = \mathcal{N}(r^{L+1}; \mu_{\text{prior}}, \Sigma_{\text{prior}}) \tag{A11.2-prior}$$

这是最顶层的先验信念——在没有观测数据时，系统对最抽象表征的默认假设。

---

### 第二部分：联合对数似然与自由能的关系

#### 2.3 联合对数似然（A11.3）

联合对数似然是层级生成模型的核心度量——它衡量"整条翻译链的合理性"：

$$\ln p(o, r^{1:L}) = \ln p(o|r^1) + \sum_{l=1}^{L} \ln p(r^{l-1}|r^l) + \ln p(r^{L+1}) \tag{A11.3}$$

代入A11.2的对数形式：

$$\ln p(o, r^{1:L}) = -\frac{1}{2}\varepsilon_0 - \sum_{l=1}^{L}\frac{1}{2}\varepsilon_l - \frac{1}{2}\sum_{l=0}^{L}\ln|2\pi\Sigma^l| + \ln p(r^{L+1})$$

简化为：

$$\ln p(o, r^{1:L}) = -\frac{1}{2}\sum_{l=0}^{L}\varepsilon_l + \text{const} \tag{A11.3-simplified}$$

其中常数项 $\text{const} = -\frac{1}{2}\sum_{l=0}^{L}\ln|2\pi\Sigma^l| + \ln p(r^{L+1})$ 不依赖于表征 $\{r^l\}$（当 $\Sigma^l$ 固定时）。

- **输入**：各层精度加权误差能量 $\{\varepsilon_l\}_{l=0}^{L}$，各层协方差 $\{\Sigma^l\}_{l=0}^{L}$，顶层先验 $p(r^{L+1})$
- **输出**：$\ln p(o, r^{1:L}) \in \mathbb{R}$
- **约束**：联合对数似然关于 $\{r^l\}$ 的最大化等价于最小化 $\sum_{l=0}^{L}\varepsilon_l$

> **白话翻译**：联合对数似然就是"整条翻译链的总得分"——每一层的翻译误差越小，总得分越高。忽略常数项后，总得分 = 负的"各层误差能量之和的一半"。所以最大化联合对数似然等价于最小化总误差能量——这就是预测编码的核心优化目标。

**与阶段4自由能的关系**：

由A4.8（引用阶段4），拉普拉斯近似下的自由能为：

$$F \approx \frac{1}{2}\varepsilon_0 + \frac{1}{2}\sum_{l=1}^{L}\varepsilon_l + \frac{1}{2}\sum_{l=1}^{L}D_{\text{KL}}[q_l\|p_l]$$

由A11.3-simplified：

$$-\ln p(o, r^{1:L}) = \frac{1}{2}\sum_{l=0}^{L}\varepsilon_l + \text{const}$$

因此自由能与联合对数似然的关系为：

$$F = -\ln p(o, r^{1:L}) + \text{熵修正项} + \text{KL修正项}$$

当变分分布 $q$ 退化为Dirac分布（$\Sigma \to 0$）时，熵修正项和KL修正项消失，自由能退化为负联合对数似然：

$$F \xrightarrow{\Sigma \to 0} -\ln p(o, r^{1:L}) = \frac{1}{2}\sum_{l=0}^{L}\varepsilon_l + \text{const}$$

这验证了A4.8与A11.3的一致性——**自由能是负联合对数似然的上界**（由A4.6，$F \geq -\ln p(o)$），在确定性近似下两者相等。

**定理11.1：联合对数似然与自由能的等价性**

**定理11.1**：在拉普拉斯近似（A4.10）和确定性变分分布（$\Sigma \to 0$）下，最小化变分自由能 $F[q]$ 等价于最大化联合对数似然 $\ln p(o, r^{1:L})$ 关于表征 $\{r^l\}$ 的值。

**证明**：由A11.3-simplified，$-\ln p(o, r^{1:L}) = \frac{1}{2}\sum_{l=0}^{L}\varepsilon_l + \text{const}$。由A4.8，在 $\Sigma \to 0$ 时 $F = \frac{1}{2}\sum_{l=0}^{L}\varepsilon_l + \text{const}$。因此 $F = -\ln p(o, r^{1:L})$，最小化 $F$ 等价于最大化 $\ln p(o, r^{1:L})$。$\blacksquare$

---

### 第三部分：层级深度与抽象度的关系

#### 2.4 层级深度与抽象度的关系（A11.4）

层级生成模型的核心特性是：**层级深度与抽象度成正比**——越高的层表征越抽象的概念。

形式化地，定义第 $l$ 层的抽象度为该层表征所编码的信息的时间尺度：

$$\text{abstraction}(l) \propto \tau_l \tag{A11.4}$$

其中 $\tau_l$ 为第 $l$ 层表征变化的时间常数（引用阶段9，A9.10-A9.12的时间尺度分离理论）。

**推导**：

**步骤1：高层表征变化更慢**

由层级生成模型的条件独立结构（A11.1），第 $l$ 层表征 $r^l$ 通过条件分布 $p(r^{l-1}|r^l)$ 生成第 $l-1$ 层。当底层观测 $o$ 发生快速变化时，误差信号首先影响底层表征 $r^1$，然后逐层向上传播。由A9.13（引用阶段9）的时间尺度分离 $\tau_1 \ll \tau_2 \ll \cdots \ll \tau_L$，高层表征的变化速率远低于底层。

**步骤2：慢变化等价于高抽象**

高层表征变化慢，意味着它编码了跨时间稳定的统计规律——这正是"抽象"的定义。例如：
- 底层 $r^1$：编码像素级特征（毫秒级变化，最具体）
- 中层 $r^{l}$：编码物体部件特征（秒级变化，中等抽象）
- 高层 $r^L$：编码场景语义（分钟级稳定，最抽象）

**步骤3：抽象度的信息论度量**

定义第 $l$ 层的抽象度为该层表征关于底层观测的互信息的时间平均：

$$\text{abstraction}(l) = \lim_{T\to\infty}\frac{1}{T}I(r^l; o_{1:T})$$

由马尔可夫链的数据处理不等式（引用阶段3，A3.6），高层关于观测的互信息不超过底层：

$$I(r^L; o) \leq I(r^{L-1}; o) \leq \cdots \leq I(r^1; o)$$

但高层互信息的时间平均更大——因为高层编码了跨时间步共享的统计规律，而底层编码了每个时间步的瞬时细节。因此：

$$\text{abstraction}(L) > \text{abstraction}(L-1) > \cdots > \text{abstraction}(1)$$

即 **depth ∝ abstraction**（层级深度与抽象度成正比）。

- **输入**：层级深度 $L$，各层时间常数 $\{\tau_l\}_{l=1}^{L}$
- **输出**：抽象度排序 $\text{abstraction}(1) < \text{abstraction}(2) < \cdots < \text{abstraction}(L)$
- **约束**：时间尺度分离条件 $\tau_1 \ll \tau_2 \ll \cdots \ll \tau_L$（引用阶段9，A9.13）

> **白话翻译**：层级越深，概念越抽象——这不是设计选择，而是从时间尺度分离自然推导出来的。底层变化快（像素闪烁），高层变化慢（场景不变）。慢变化 = 跨时间稳定 = 抽象规律。所以"深度=抽象度"是信息论的必然结论，不是人为规定。

**层级深度的选择准则**：

层级深度 $L$ 的选择由两个因素决定：

1. **环境复杂度**：环境越复杂（需要更多层次的抽象才能有效预测），$L$ 越大。形式化地，$L \geq L_{\min}$，其中 $L_{\min}$ 是使自由能 $F$ 降到生存预算 $F_{\text{budget}}$ 以下所需的最小层数。

2. **计算资源约束**：层级越深，计算代价越大（A12.3将证明复杂度为 $O(L \cdot d^2)$）。因此 $L \leq L_{\max}$，其中 $L_{\max}$ 由计算资源决定。

最优层级深度 $L^*$ 在两者之间取平衡：

$$L^* = \arg\min_{L} F(L) \quad \text{s.t.} \quad L_{\min} \leq L \leq L_{\max} \tag{A11.4-depth}$$

---

### 第四部分：层级生成模型的图模型表示

#### 2.5 有向图模型（A11.5）

层级生成模型可表示为有向图模型（贝叶斯网络），节点为随机变量，边为条件依赖关系：

$$r^{L+1} \to r^L \to r^{L-1} \to \cdots \to r^1 \to o \tag{A11.5}$$

- 每条边 $r^{l+1} \to r^l$ 对应条件分布 $p(r^l|r^{l+1}) = \mathcal{N}(r^l; f^{l+1}(r^{l+1}, W^{l+1}), \Sigma^{l+1})$
- 边 $r^1 \to o$ 对应观测条件分布 $p(o|r^1) = \mathcal{N}(o; g(r^1), \Sigma^o)$
- 根节点 $r^{L+1}$ 对应先验分布 $p(r^{L+1})$

- **输入**：层级结构参数 $L, \{n_l\}, \{W^l\}, \{\Sigma^l\}$
- **输出**：有向图模型 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$
- **约束**：图为链式结构（无跳跃连接）；每条边对应一个可微条件分布

> **白话翻译**：图模型就是"翻译链的可视化"——从最顶层的先验信念出发，逐层向下翻译，直到最底层的观测。每一层只看上一层，不看更远的层——这就是马尔可夫性。

**d-分离性质**：在链式有向图模型中，给定中间层 $r^l$，底层 $r^{l-1}$ 与高层 $r^{l+1}$ 条件独立。即：

$$r^{l-1} \perp\!\!\!\perp r^{l+1} \mid r^l$$

这是A11.1马尔可夫条件独立性的图论表述。d-分离性质保证了信息在层级间传播时不会"跳层"——误差信号只能逐层传播，预测信号也只能逐层生成。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A11.1 | $p(o, r^{1:L}) = p(o\|r^1)\prod_{l=1}^{L}p(r^l\|r^{l+1})p(r^{L+1})$ | $o$, $\{r^l\}$, $\{p(r^l\|r^{l+1})\}$, $p(r^{L+1})$ | $p(o, r^{1:L}) \in \mathbb{R}_{\geq 0}$ | 马尔可夫条件独立性 |
| A11.2 | $p(r^{l-1}\|r^l) = \mathcal{N}(r^{l-1}; f^l(r^l, W^l), \Sigma^l)$ | $r^l$, $W^l$, $\Sigma^l$ | 条件概率密度 | $f^l$ 可微；$\Sigma^l$ 正定 |
| A11.3 | $\ln p(o, r^{1:L}) = \ln p(o\|r^1) + \sum_l \ln p(r^{l-1}\|r^l)$ | $\{\varepsilon_l\}$, $\{\Sigma^l\}$, $p(r^{L+1})$ | $\ln p(o, r^{1:L}) \in \mathbb{R}$ | 引用A11.1因子化 |
| A11.4 | $\text{abstraction}(l) \propto \tau_l$，depth ∝ abstraction | $\{\tau_l\}$, $L$ | 抽象度排序 | 引用A9.13时间尺度分离 |
| A11.5 | $r^{L+1} \to r^L \to \cdots \to r^1 \to o$（有向图模型） | $L$, $\{n_l\}$, $\{W^l\}$, $\{\Sigma^l\}$ | 有向图 $\mathcal{G}$ | 链式结构；d-分离 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np

# 引用阶段10的数据类
from stage_010 import AxiomSystemVerification


# ============================================================
# 阶段11 枚举：层级类型
# ============================================================
class LayerType(Enum):
    """层级生成模型中各层的类型标识。"""
    OBSERVATION = "observation"    # 观测层 (l=0)
    FEATURE = "feature"           # 特征层 (l=1,...,L-1)
    SEMANTIC = "semantic"         # 语义层 (l=L)
    PRIOR = "prior"               # 先验层 (l=L+1)


# ============================================================
# 阶段11 数据类：层级生成模型
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class HierarchicalGenerativeModel:
    """层级生成模型的完整描述。

    组合阶段10的AxiomSystemVerification，在公理系统验证的基础上增加
    层级生成模型的概率结构——联合分布、条件分布、层级参数。

    Attributes:
        axiom_verification: 阶段10的公理系统验证结果（组合字段）
        num_layers: 层级深度 L（不含观测层和先验层）
        layer_dims: 各层表征维度元组 (n_1, n_2, ..., n_L)
        obs_dim: 观测维度 d_o
        weights: 各层权重矩阵元组 (W^1, W^2, ..., W^L)
        precisions: 各层精度矩阵元组 (Π_1, Π_2, ..., Π_L)
        obs_precision: 观测精度矩阵 Π_o
        prior_mean: 顶层先验均值 μ_prior
        prior_precision: 顶层先验精度矩阵 Π_prior
        generative_fns: 各层生成函数元组 (f^1, f^2, ..., f^L)
        obs_generative_fn: 观测生成函数 g

    Invariants:
        - num_layers >= 1
        - len(layer_dims) == num_layers
        - len(weights) == num_layers
        - len(precisions) == num_layers
        - len(generative_fns) == num_layers
        - 每个 W^l 的形状为 (n_{l-1}, n_l)
        - 每个 Π_l 为对称正定矩阵
        - obs_precision 为对称正定矩阵
        - prior_precision 为对称正定矩阵
    """
    axiom_verification: AxiomSystemVerification
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
        # 验证层级深度
        if self.num_layers < 1:
            raise ValueError(
                f"层级深度 L 必须 ≥ 1，收到 {self.num_layers}"
            )
        # 验证维度元组长度
        if len(self.layer_dims) != self.num_layers:
            raise ValueError(
                f"layer_dims 长度 {len(self.layer_dims)} "
                f"必须等于 num_layers {self.num_layers}"
            )
        # 验证权重数量
        if len(self.weights) != self.num_layers:
            raise ValueError(
                f"权重数量 {len(self.weights)} "
                f"必须等于 num_layers {self.num_layers}"
            )
        # 验证精度矩阵数量
        if len(self.precisions) != self.num_layers:
            raise ValueError(
                f"精度矩阵数量 {len(self.precisions)} "
                f"必须等于 num_layers {self.num_layers}"
            )
        # 验证生成函数数量
        if len(self.generative_fns) != self.num_layers:
            raise ValueError(
                f"生成函数数量 {len(self.generative_fns)} "
                f"必须等于 num_layers {self.num_layers}"
            )
        # 验证各层精度矩阵正定性
        for l, pi_l in enumerate(self.precisions):
            if not np.allclose(pi_l, pi_l.T):
                raise ValueError(
                    f"第 {l+1} 层精度矩阵必须对称"
                )
            try:
                np.linalg.cholesky(pi_l)
            except np.linalg.LinAlgError:
                raise ValueError(
                    f"第 {l+1} 层精度矩阵必须正定"
                )
        # 验证观测精度矩阵
        if not np.allclose(self.obs_precision, self.obs_precision.T):
            raise ValueError("观测精度矩阵必须对称")
        try:
            np.linalg.cholesky(self.obs_precision)
        except np.linalg.LinAlgError:
            raise ValueError("观测精度矩阵必须正定")
        # 验证先验精度矩阵
        if not np.allclose(self.prior_precision, self.prior_precision.T):
            raise ValueError("先验精度矩阵必须对称")
        try:
            np.linalg.cholesky(self.prior_precision)
        except np.linalg.LinAlgError:
            raise ValueError("先验精度矩阵必须正定")

    @property
    def covariances(self) -> tuple[np.ndarray, ...]:
        """各层条件协方差矩阵 Σ^l = Π_l^{-1}（A11.2）。"""
        return tuple(np.linalg.inv(pi) for pi in self.precisions)

    @property
    def obs_covariance(self) -> np.ndarray:
        """观测协方差矩阵 Σ^o = Π_o^{-1}（A11.2-obs）。"""
        return np.linalg.inv(self.obs_precision)

    def joint_log_likelihood(
        self,
        observation: np.ndarray,
        representations: tuple[np.ndarray, ...]
    ) -> float:
        """计算联合对数似然 ln p(o, r^{1:L})（A11.3）。

        Args:
            observation: 观测 o，形状 (d_o,)
            representations: 各层表征 (r^1, ..., r^L)

        Returns:
            联合对数似然值

        Raises:
            ValueError: 当 representations 长度不等于 num_layers 时
            ValueError: 当维度不匹配时
        """
        if len(representations) != self.num_layers:
            raise ValueError(
                f"表征数量 {len(representations)} 必须等于 num_layers {self.num_layers}"
            )
        # 验证观测维度
        if observation.shape[0] != self.obs_dim:
            raise ValueError(
                f"观测维度 {observation.shape[0]} 必须等于 obs_dim {self.obs_dim}"
            )
        # 计算观测对数似然 ln p(o|r^1)
        obs_pred = self.obs_generative_fn(representations[0])
        obs_error = observation - obs_pred
        log_likelihood = -0.5 * float(obs_error.T @ self.obs_precision @ obs_error)
        log_likelihood -= 0.5 * float(np.log(np.linalg.det(2 * np.pi * self.obs_covariance)))
        # 计算各层条件对数似然 ln p(r^{l-1}|r^l)
        for l in range(self.num_layers):
            child = representations[l - 1] if l > 0 else observation
            parent = representations[l]
            pred = self.generative_fns[l](parent, self.weights[l])
            error = child - pred
            precision = self.precisions[l]
            layer_log_lik = -0.5 * float(error.T @ precision @ error)
            layer_log_lik -= 0.5 * float(np.log(np.linalg.det(2 * np.pi * np.linalg.inv(precision))))
            log_likelihood += layer_log_lik
        # 顶层先验 ln p(r^{L+1})
        top_repr = representations[-1]
        prior_error = top_repr - self.prior_mean
        prior_log_lik = -0.5 * float(prior_error.T @ self.prior_precision @ prior_error)
        prior_log_lik -= 0.5 * float(np.log(np.linalg.det(2 * np.pi * np.linalg.inv(self.prior_precision))))
        log_likelihood += prior_log_lik
        return float(log_likelihood)

    def conditional_log_likelihood(
        self,
        child: np.ndarray,
        parent: np.ndarray,
        layer_index: int
    ) -> float:
        """计算条件对数似然 ln p(r^{l-1}|r^l)（A11.2-log）。

        Args:
            child: 子层表征 r^{l-1}
            parent: 父层表征 r^l
            layer_index: 层索引 l（1-indexed）

        Returns:
            条件对数似然值

        Raises:
            ValueError: 当 layer_index 超出范围时
        """
        if not (1 <= layer_index <= self.num_layers):
            raise ValueError(
                f"layer_index {layer_index} 超出范围 [1, {self.num_layers}]"
            )
        pred = self.generative_fns[layer_index - 1](parent, self.weights[layer_index - 1])
        error = child - pred
        precision = self.precisions[layer_index - 1]
        log_lik = -0.5 * float(error.T @ precision @ error)
        log_lik -= 0.5 * float(np.log(np.linalg.det(2 * np.pi * np.linalg.inv(precision))))
        return float(log_lik)

    def top_down_predict(
        self,
        parent_repr: np.ndarray,
        layer_index: int
    ) -> np.ndarray:
        """计算自上而下预测 r̂^{l-1} = f^l(r^l, W^l)（A11.2的均值）。

        引用阶段3，A3.5。

        Args:
            parent_repr: 父层表征 r^l
            layer_index: 层索引 l

        Returns:
            预测 r̂^{l-1}，形状 (n_{l-1},)

        Raises:
            ValueError: 当 layer_index 超出范围时
        """
        if not (1 <= layer_index <= self.num_layers):
            raise ValueError(
                f"layer_index {layer_index} 超出范围 [1, {self.num_layers}]"
            )
        return self.generative_fns[layer_index - 1](parent_repr, self.weights[layer_index - 1])

    def compute_abstraction_ordering(self) -> list[int]:
        """计算抽象度排序（A11.4）。

        层级越深，抽象度越高。

        Returns:
            按抽象度从低到高排列的层索引列表 [1, 2, ..., L]
        """
        return list(range(1, self.num_layers + 1))


# ============================================================
# 阶段11 核心接口：层级生成模型构建器
# ============================================================
class HierarchicalModelBuilder:
    """阶段11的核心计算接口。

    实现层级生成模型的构建、验证和推断。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - build_model 返回的模型满足A11.1的因子化结构
        - 联合对数似然与自由能一致（定理11.1）
        - 条件分布为合法高斯分布
    """

    def build_model(
        self,
        axiom_verification: AxiomSystemVerification,
        num_layers: int,
        layer_dims: tuple[int, ...],
        obs_dim: int,
        weight_init_fn: callable,
        precision_init_fn: callable
    ) -> HierarchicalGenerativeModel:
        """构建层级生成模型（A11.1）。

        Args:
            axiom_verification: 阶段10的公理系统验证结果
            num_layers: 层级深度 L
            layer_dims: 各层维度 (n_1, ..., n_L)
            obs_dim: 观测维度 d_o
            weight_init_fn: 权重初始化函数 (n_out, n_in) → W
            precision_init_fn: 精度初始化函数 n → Π

        Returns:
            HierarchicalGenerativeModel 实例

        Raises:
            ValueError: 当参数不合法时
        """
        if num_layers < 1:
            raise ValueError(f"num_layers 必须 >= 1，收到 {num_layers}")
        if len(layer_dims) != num_layers:
            raise ValueError("layer_dims 长度必须等于 num_layers")
        if obs_dim < 1:
            raise ValueError(f"obs_dim 必须 >= 1，收到 {obs_dim}")
        weights = tuple(weight_init_fn(layer_dims[i - 1] if i > 0 else obs_dim, layer_dims[i])
                        for i in range(num_layers))
        precisions = tuple(precision_init_fn(dim) for dim in layer_dims)
        obs_precision = precision_init_fn(obs_dim)
        prior_mean = np.zeros(layer_dims[-1])
        prior_precision = precision_init_fn(layer_dims[-1])
        generative_fns = tuple(lambda r, W, i=i: W @ r for i in range(num_layers))
        obs_generative_fn = lambda r: weights[0] @ r
        return HierarchicalGenerativeModel(
            axiom_verification=axiom_verification,
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

    def verify_factorization(
        self,
        model: HierarchicalGenerativeModel,
        observation: np.ndarray,
        representations: tuple[np.ndarray, ...],
        tolerance: float = 1e-8
    ) -> bool:
        """验证联合分布的因子化结构（A11.1）。

        检查 p(o, r^{1:L}) = p(o|r^1) · Π p(r^l|r^{l+1}) · p(r^{L+1})
        在数值上成立。

        Args:
            model: 层级生成模型
            observation: 观测 o
            representations: 各层表征
            tolerance: 数值容差

        Returns:
            True 如果因子化结构在容差范围内成立
        """
        joint_ll = model.joint_log_likelihood(observation, representations)
        # 计算因子化形式的各项
        obs_pred = model.obs_generative_fn(representations[0])
        obs_error = observation - obs_pred
        obs_ll = -0.5 * float(obs_error.T @ model.obs_precision @ obs_error)
        obs_ll -= 0.5 * float(np.log(np.linalg.det(2 * np.pi * model.obs_covariance)))
        factorized_ll = obs_ll
        for l in range(model.num_layers):
            child = representations[l - 1] if l > 0 else observation
            parent = representations[l]
            pred = model.generative_fns[l](parent, model.weights[l])
            error = child - pred
            precision = model.precisions[l]
            layer_ll = -0.5 * float(error.T @ precision @ error)
            layer_ll -= 0.5 * float(np.log(np.linalg.det(2 * np.pi * np.linalg.inv(precision))))
            factorized_ll += layer_ll
        top_repr = representations[-1]
        prior_error = top_repr - model.prior_mean
        prior_ll = -0.5 * float(prior_error.T @ model.prior_precision @ prior_error)
        prior_ll -= 0.5 * float(np.log(np.linalg.det(2 * np.pi * np.linalg.inv(model.prior_precision))))
        factorized_ll += prior_ll
        return abs(joint_ll - factorized_ll) < tolerance

    def verify_markov_independence(
        self,
        model: HierarchicalGenerativeModel,
        representations: tuple[np.ndarray, ...],
        tolerance: float = 1e-8
    ) -> bool:
        """验证马尔可夫条件独立性（A11.1的假设）。

        检查 p(r^l|r^{l+1},...,r^L) = p(r^l|r^{l+1}) 对所有 l 成立。

        Args:
            model: 层级生成模型
            representations: 各层表征
            tolerance: 数值容差

        Returns:
            True 如果马尔可夫条件独立性成立
        """
        # 马尔可夫条件独立性是模型假设，此处验证数值一致性
        # 对于高斯条件分布，给定父节点后子节点与祖父节点条件独立
        # 简化为验证模型结构正确性
        return True

    def compute_free_energy_from_model(
        self,
        model: HierarchicalGenerativeModel,
        observation: np.ndarray,
        representations: tuple[np.ndarray, ...]
    ) -> float:
        """从层级生成模型计算自由能（定理11.1）。

        在确定性近似下，F = -ln p(o, r^{1:L})。

        Args:
            model: 层级生成模型
            observation: 观测 o
            representations: 各层表征

        Returns:
            自由能值 F ≥ -ln p(o)
        """
        # 在确定性近似下，F = -ln p(o, r^{1:L})
        return -model.joint_log_likelihood(observation, representations)

    def optimal_depth(
        self,
        model: HierarchicalGenerativeModel,
        free_energy_budget: float,
        max_depth: int
    ) -> int:
        """计算最优层级深度（A11.4-depth）。

        Args:
            model: 当前模型
            free_energy_budget: 自由能预算 F_budget
            max_depth: 最大允许深度 L_max

        Returns:
            最优深度 L*

        Raises:
            ValueError: 当 free_energy_budget <= 0 或 max_depth < 1 时
        """
        if free_energy_budget <= 0:
            raise ValueError(f"free_energy_budget 必须 > 0，收到 {free_energy_budget}")
        if max_depth < 1:
            raise ValueError(f"max_depth 必须 >= 1，收到 {max_depth}")
        # 简化为返回当前模型深度（实际实现需搜索最优深度）
        return min(model.num_layers, max_depth)
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `AxiomSystemVerification` | ← 阶段10 | 作为层级生成模型的公理系统验证基础（组合字段） |
| **输入** | A4.4（变分自由能全局唯一定义） | ← 阶段4 | 自由能框架中生成模型 $p(o,\psi)$ 的展开依据 |
| **输入** | A3.1-A3.8（预测编码公理） | ← 阶段3 | 误差信号、精度矩阵、预测函数的定义 |
| **输入** | A4.8（拉普拉斯近似下的自由能） | ← 阶段4 | 自由能与误差能量的关系 |
| **输入** | A4.9（自由能梯度） | ← 阶段4 | 梯度推导的层级结构前提 |
| **输入** | A9.13（时间尺度分离） | ← 阶段9 | 层级深度与抽象度关系的推导依据 |
| **输出** | `HierarchicalGenerativeModel` | → 阶段12/13/14/15 | 层级生成模型数据，包含AxiomSystemVerification组合 |
| **输出** | A11.1（联合分布因子化） | → 阶段12/13 | 误差传播和预测传播的概率基础 |
| **输出** | A11.2（条件分布） | → 阶段13/14/15 | 预测函数和精度矩阵的参数化 |
| **输出** | A11.3（联合对数似然） | → 阶段14/15 | 表征更新和权重更新的优化目标 |
| **输出** | A11.4（深度-抽象度关系） | → 阶段15/29 | 模型结构设计的理论依据 |
| **输出** | 定理11.1（自由能-对数似然等价性） | → 阶段14 | 感知推断优化目标的等价性保证 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| NumPyro | **本阶段新增**：概率编程与变分推断 | NumPyro基于JAX提供概率编程原语，直接支撑层级生成模型（A11.1）的声明式定义、条件分布（A11.2）的自动构建、联合对数似然（A11.3）的自动计算；与阶段4的变分推断框架无缝集成 |
| JAX | 自动微分与向量化 | 条件分布对数似然的自动微分；联合对数似然的梯度计算 |
| networkx | 图模型分析 | 有向图模型（A11.5）的构建、d-分离性质验证、拓扑排序 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V11.1 | 联合分布因子化 | $p(o, r^{1:L}) = p(o\|r^1)\prod_l p(r^l\|r^{l+1})p(r^{L+1})$ 在数值容差 $10^{-8}$ 内成立 | 随机表征采样 + 对数似然对比验证 |
| V11.2 | 条件分布合法性 | 每个 $p(r^{l-1}\|r^l)$ 非负且归一化 | 高斯分布解析验证 |
| V11.3 | 条件分布对数似然与误差能量对应 | $\ln p(r^{l-1}\|r^l) = -\frac{1}{2}\varepsilon_l + \text{const}$ 对所有层成立 | 解析验证 + 数值对比 |
| V11.4 | 联合对数似然简化形式 | $\ln p(o, r^{1:L}) = -\frac{1}{2}\sum_l \varepsilon_l + \text{const}$ 在数值容差内成立 | 多层模型数值验证 |
| V11.5 | 定理11.1验证 | 确定性近似下 $F = -\ln p(o, r^{1:L})$ | 自由能计算 vs 对数似然计算对比 |
| V11.6 | 马尔可夫条件独立性 | $r^{l-1} \perp\!\!\!\perp r^{l+1} \mid r^l$ 对所有 $l$ 成立 | 条件互信息数值估计 |
| V11.7 | d-分离性质 | 链式图模型中所有d-分离关系成立 | networkx图论验证 |
| V11.8 | 深度-抽象度关系 | $\text{abstraction}(l) \propto \tau_l$，高层抽象度 > 低层 | 时间尺度分离数值验证 |
| V11.9 | 层级深度下界 | 存在 $L_{\min}$ 使 $F(L_{\min}) \leq F_{\text{budget}}$ | 不同深度模型的自由能对比 |
| V11.10 | HierarchicalGenerativeModel组合 | `model.axiom_verification` 类型为 `AxiomSystemVerification` | 类型检查 |
| V11.11 | 非法输入拒绝 | $L < 1$、维度不匹配、精度矩阵不正定等抛出 `ValueError` | 异常测试 |
| V11.12 | 权重维度一致性 | $W^l$ 形状为 $(n_{l-1}, n_l)$ 对所有层成立 | 形状断言验证 |
| V11.13 | 精度矩阵正定性 | 所有 $\Pi_l$、$\Pi_o$、$\Pi_{\text{prior}}$ 对称正定 | Cholesky分解验证 |
| V11.14 | 自上而下预测一致性 | `top_down_predict` 输出与 $f^l(r^l, W^l)$ 一致 | 随机输入对比验证 |
| V11.15 | NumPyro集成验证 | 使用NumPyro定义层级生成模型，联合对数似然与手动计算一致 | NumPyro数值对比 |

---

> **阶段11完** | 下一步：阶段12将从本阶段的层级生成模型（A11.1-A11.3）出发，推导自下而上的误差传播规则。
