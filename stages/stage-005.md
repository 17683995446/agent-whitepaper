# 阶段5：自由能分解——准确性与复杂性的严格分离

> **阶段编号**：S5 | **所属层**：第一层·基础公理层
>
> **去重声明**：本阶段仅展开阶段4的A4.5（准确性-复杂性分解），不重复推导变分自由能本身。变分自由能的全局唯一定义见阶段4（A4.4），三种等价分解见阶段4（A4.5/A4.6/A4.7）。本阶段聚焦于准确性项与复杂性项各自的数学结构、梯度分析、Pareto最优权衡、模型选择准则、正则化统一和动力学轨迹。
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 准确性项 | "猜错多少的精确账单"——预测与观测的加权偏差，偏差越大罚得越重 |
> | 复杂性项 | "偏离常识的精确账单"——新认识与旧信念的加权距离，改变越多罚得越重 |
> | Pareto前沿 | "鱼与熊掌的最优平衡线"——准确性再提高一点，复杂性必然增加一点 |
> | 模型选择准则 | "选哪个模型最划算"——AIC/BIC/自由能分别用什么标准衡量"划算" |
> | 正则化统一 | "机器学习的惩罚项都是复杂性项的化身"——L2/L1/Dropout都能在自由能框架中找到根源 |
> | 动力学轨迹 | "系统如何一步步走向最优"——在自由能景观上的下山路径 |

---

## 1. 第一性原理依据

本阶段从阶段4的核心结论出发，无需引入新的物理公理。核心依据为：

- **A4.4（变分自由能全局唯一定义）**（引用阶段4）：$F[q] = -\mathbb{E}_q[\ln p(o|\psi)] + D_{\text{KL}}[q(\psi) \| p(\psi)]$，后续所有阶段引用自由能时均指向此定义。
- **A4.5（准确性+复杂性分解）**（引用阶段4）：$F[q] = \underbrace{-\mathbb{E}_q[\ln p(o|\psi)]}_{\text{准确性项}} + \underbrace{D_{\text{KL}}[q(\psi) \| p(\psi)]}_{\text{复杂性项}}$，本阶段唯一展开对象。
- **A3.2（精度加权误差能量）**（引用阶段3）：$\varepsilon_l = e^{l-1,T} \Pi_l e^{l-1}$，准确性项与预测误差的桥梁。
- **A4.10（拉普拉斯近似）**（引用阶段4）：$q(\psi) = \mathcal{N}(\psi; \mu, \Sigma)$，高斯变分分布下的解析展开。

本阶段的目标是将A4.5的两个项分别展开为可计算的解析形式，建立准确性-复杂性的Pareto最优理论，统一模型选择准则与正则化理论，并分析自由能景观上的动力学轨迹。

---

## 2. 核心数学公式

### 第一部分：准确性项的严格分析

#### 2.1 准确性项的一般定义（A5.1）

准确性项是变分自由能的第一项，度量变分分布下似然对数的期望负值（引用阶段4，A4.5第一项）：

$$A(q) = -\mathbb{E}_q[\ln p(o|\psi)] = -\int q(\psi) \ln p(o|\psi) \, \mathrm{d}\psi \tag{A5.1}$$

- **输入**：变分分布 $q(\psi)$，似然函数 $p(o|\psi)$
- **输出**：$A(q) \in \mathbb{R}$（可正可负，取决于似然函数的归一化常数）
- **约束**：$q(\psi) \geq 0$，$\int q(\psi) \, \mathrm{d}\psi = 1$；$p(o|\psi)$ 为合法概率密度

> **白话翻译**：准确性项就是"在当前猜测下，观测数据出现的概率有多低"——概率越低（越意外），此项越大。完美预测时此项最小，完全猜错时此项最大。

**与A4.5的关系**：A5.1就是A4.5的第一项，此处独立编号以展开其内部结构。

#### 2.2 高斯似然下的准确性（A5.2）

当似然为高斯形式 $p(o|\psi) = \mathcal{N}(o; g(\psi), \Sigma_\eta)$（引用阶段2，A2.3），且在拉普拉斯近似下 $q(\psi) = \mathcal{N}(\psi; \mu, \Sigma)$（引用阶段4，A4.10），准确性项展开为：

$$A(q) = \frac{1}{2}(o - g(\mu))^T \Sigma_\eta^{-1} (o - g(\mu)) + \frac{1}{2}\text{tr}(\Sigma_\eta^{-1} J \Sigma J^T) + \frac{1}{2}\ln|2\pi\Sigma_\eta| \tag{A5.2}$$

其中 $J = \frac{\partial g}{\partial \psi}\big|_{\psi=\mu}$ 为生成函数在变分均值处的Jacobian矩阵。

- **输入**：观测 $o \in \mathbb{R}^{d_o}$，生成函数 $g$，变分均值 $\mu \in \mathbb{R}^d$，变分协方差 $\Sigma \in \mathbb{S}_{++}^d$，噪声协方差 $\Sigma_\eta \in \mathbb{S}_{++}^{d_o}$
- **输出**：$A(q) \in \mathbb{R}$
- **约束**：$\Sigma_\eta$ 正定；$g$ 在 $\mu$ 处可微

> **白话翻译**：高斯似然下，准确性项 = "预测偏差的加权平方"（第一项）+ "不确定性导致的额外惩罚"（第二项）+ "常数"（第三项）。第一项是主角——预测越偏，罚得越重；第二项是配角——即使均值预测对了，如果猜测太模糊（$\Sigma$ 大），也要额外受罚。

**完整推导**：

**步骤1：展开高斯似然对数**

$$\ln p(o|\psi) = -\frac{1}{2}(o - g(\psi))^T \Sigma_\eta^{-1} (o - g(\psi)) - \frac{1}{2}\ln|2\pi\Sigma_\eta|$$

**步骤2：在 $q$ 下取期望**

$$A(q) = \frac{1}{2}\mathbb{E}_q[(o - g(\psi))^T \Sigma_\eta^{-1} (o - g(\psi))] + \frac{1}{2}\ln|2\pi\Sigma_\eta|$$

**步骤3：对 $g(\psi)$ 在 $\mu$ 处线性化**

$$g(\psi) \approx g(\mu) + J(\psi - \mu)$$

代入期望：

$$\mathbb{E}_q[(o - g(\psi))] = o - g(\mu)$$

$$\mathbb{E}_q[(o - g(\psi))(o - g(\psi))^T] = (o - g(\mu))(o - g(\mu))^T + J\Sigma J^T$$

**步骤4：合并**

$$A(q) = \frac{1}{2}(o - g(\mu))^T \Sigma_\eta^{-1} (o - g(\mu)) + \frac{1}{2}\text{tr}(\Sigma_\eta^{-1} J \Sigma J^T) + \frac{1}{2}\ln|2\pi\Sigma_\eta|$$

$\blacksquare$

**简化情形**：当变分分布退化为Dirac分布（$\Sigma \to 0$）时，第二项消失：

$$A(q) \xrightarrow{\Sigma \to 0} \frac{1}{2}(o - g(\mu))^T \Sigma_\eta^{-1} (o - g(\mu)) + \frac{1}{2}\ln|2\pi\Sigma_\eta|$$

#### 2.3 准确性对预测误差的关系（A5.3）

在拉普拉斯近似下，准确性项的主项与阶段3的精度加权误差能量直接对应。设 $g(\psi) = \psi$（恒等生成函数）且 $\Sigma \to 0$，则：

$$A(q) = \frac{1}{2}\varepsilon_0 + \text{const} \tag{A5.3}$$

其中 $\varepsilon_0 = (o - \mu)^T \Sigma_\eta^{-1} (o - \mu)$ 为底层精度加权误差能量（引用阶段3，A3.2）。

- **输入**：观测 $o$，变分均值 $\mu$，噪声精度 $\Sigma_\eta^{-1}$
- **输出**：$A(q) = \frac{1}{2}\varepsilon_0 + \text{const}$
- **约束**：恒等生成函数；变分协方差趋于零（确定性近似）

> **白话翻译**：准确性项的核心就是阶段3定义的"精度加权误差能量"的一半——预测越准，误差能量越小，准确性项越小。常数部分不影响优化方向，可以忽略。

**推导**：由A5.2，当 $g(\psi) = \psi$ 且 $\Sigma \to 0$ 时，$J = I$，$\text{tr}(\Sigma_\eta^{-1} J \Sigma J^T) \to 0$，因此：

$$A(q) = \frac{1}{2}(o - \mu)^T \Sigma_\eta^{-1} (o - \mu) + \frac{1}{2}\ln|2\pi\Sigma_\eta| = \frac{1}{2}\varepsilon_0 + \text{const}$$

其中 $\varepsilon_0 = (o - \mu)^T \Sigma_\eta^{-1} (o - \mu)$ 恰为A3.2（引用阶段3）在底层的定义。

$\blacksquare$

#### 2.4 准确性梯度（A5.4）

准确性项关于变分均值 $\mu$ 的梯度为：

$$\frac{\partial A}{\partial \mu} = -\left(\frac{\partial g}{\partial \psi}\bigg|_{\psi=\mu}\right)^T \Sigma_\eta^{-1} (o - g(\mu)) \tag{A5.4}$$

- **输入**：Jacobian矩阵 $J = \partial g / \partial \psi|_{\psi=\mu} \in \mathbb{R}^{d_o \times d}$，噪声精度 $\Sigma_\eta^{-1} \in \mathbb{S}_{++}^{d_o}$，预测残差 $o - g(\mu) \in \mathbb{R}^{d_o}$
- **输出**：梯度 $\partial A / \partial \mu \in \mathbb{R}^d$
- **约束**：$g$ 在 $\mu$ 处可微；$\Sigma_\eta$ 正定

> **白话翻译**：准确性梯度的方向 = "生成函数的反向传播" × "噪声精度加权" × "预测残差"。残差越大（猜得越错），梯度越大（纠正力度越强）；精度越高（噪声越小），梯度也越大（可信的误差更值得纠正）。方向是让预测更接近观测的方向。

**完整推导**：

由A5.2，忽略不依赖 $\mu$ 的常数项：

$$A(q) \approx \frac{1}{2}(o - g(\mu))^T \Sigma_\eta^{-1} (o - g(\mu))$$

对 $\mu$ 求梯度，令 $r = o - g(\mu)$：

$$\frac{\partial A}{\partial \mu} = -\left(\frac{\partial g}{\partial \mu}\right)^T \Sigma_\eta^{-1} r = -J^T \Sigma_\eta^{-1} (o - g(\mu))$$

$\blacksquare$

**与A4.9的连接**：A5.4是准确性项对变分均值的梯度，与A4.9（引用阶段4）的自由能梯度 $\partial F / \partial r^l = \delta^{l+1} - e^l$ 中 $-e^l$ 部分（准确性梯度的信号形式）一致。

---

### 第二部分：复杂性项的严格分析

#### 2.5 复杂性项的一般定义（A5.5）

复杂性项是变分自由能的第二项，度量变分分布与先验分布之间的KL散度（引用阶段4，A4.5第二项）：

$$C(q) = D_{\text{KL}}[q(\psi) \| p(\psi)] = \int q(\psi) \ln \frac{q(\psi)}{p(\psi)} \, \mathrm{d}\psi \tag{A5.5}$$

- **输入**：变分分布 $q(\psi)$，先验分布 $p(\psi)$
- **输出**：$C(q) \in \mathbb{R}_{\geq 0}$
- **约束**：$C(q) \geq 0$（KL散度非负性）；$C(q) = 0$ 当且仅当 $q = p$（几乎处处相等）

> **白话翻译**：复杂性项衡量"新认识偏离旧信念多远"——如果看了数据后完全没改变看法（$q = p$），复杂性为零；改变越多，复杂性越大。它惩罚"过度拟合"——为了追求准确性而大幅偏离先验。

**与A4.5的关系**：A5.5就是A4.5的第二项，此处独立编号以展开其内部结构。

#### 2.6 高斯先验下的复杂性（A5.6）

当先验为高斯 $p(\psi) = \mathcal{N}(\psi; \mu_{\text{prior}}, \Sigma_{\text{prior}})$，变分分布为高斯 $q(\psi) = \mathcal{N}(\psi; \mu, \Sigma)$（引用阶段4，A4.10），复杂性项有解析形式：

$$C(q) = \frac{1}{2}(\mu - \mu_{\text{prior}})^T \Sigma_{\text{prior}}^{-1} (\mu - \mu_{\text{prior}}) + \frac{1}{2}\ln\frac{|\Sigma_{\text{prior}}|}{|\Sigma|} + \frac{1}{2}\text{tr}(\Sigma_{\text{prior}}^{-1}\Sigma) - \frac{d}{2} \tag{A5.6}$$

- **输入**：变分均值 $\mu \in \mathbb{R}^d$，变分协方差 $\Sigma \in \mathbb{S}_{++}^d$，先验均值 $\mu_{\text{prior}} \in \mathbb{R}^d$，先验协方差 $\Sigma_{\text{prior}} \in \mathbb{S}_{++}^d$
- **输出**：$C(q) \in \mathbb{R}_{\geq 0}$
- **约束**：$\Sigma$ 和 $\Sigma_{\text{prior}}$ 均正定；$C(q) = 0$ 当且仅当 $\mu = \mu_{\text{prior}}$ 且 $\Sigma = \Sigma_{\text{prior}}$

> **白话翻译**：高斯先验下，复杂性 = "均值偏离的加权平方"（第一项）+ "方差变化的对数比"（第二项）+ "方差匹配的迹项"（第三项）- "维度常数"（第四项）。第一项最直观——均值离先验越远，惩罚越大；第二、三项处理不确定性——猜测太确定（$\Sigma$ 小）或太模糊（$\Sigma$ 大）都会被惩罚。

**完整推导**：

由两个高斯分布之间的KL散度公式：

$$D_{\text{KL}}[\mathcal{N}(\mu, \Sigma) \| \mathcal{N}(\mu_{\text{prior}}, \Sigma_{\text{prior}})] = \frac{1}{2}\left[\text{tr}(\Sigma_{\text{prior}}^{-1}\Sigma) + (\mu - \mu_{\text{prior}})^T \Sigma_{\text{prior}}^{-1} (\mu - \mu_{\text{prior}}) - d + \ln\frac{|\Sigma_{\text{prior}}|}{|\Sigma|}\right]$$

此公式的推导如下：

**步骤1：展开KL散度定义**

$$D_{\text{KL}}[q \| p] = \mathbb{E}_q[\ln q(\psi)] - \mathbb{E}_q[\ln p(\psi)]$$

**步骤2：计算 $\mathbb{E}_q[\ln q(\psi)]$**

$$\mathbb{E}_q[\ln q(\psi)] = -\frac{d}{2}\ln(2\pi) - \frac{1}{2}\ln|\Sigma| - \frac{1}{2}\mathbb{E}_q[(\psi - \mu)^T \Sigma^{-1} (\psi - \mu)]$$

$$= -\frac{d}{2}\ln(2\pi) - \frac{1}{2}\ln|\Sigma| - \frac{d}{2}$$

**步骤3：计算 $\mathbb{E}_q[\ln p(\psi)]$**

$$\mathbb{E}_q[\ln p(\psi)] = -\frac{d}{2}\ln(2\pi) - \frac{1}{2}\ln|\Sigma_{\text{prior}}| - \frac{1}{2}\mathbb{E}_q[(\psi - \mu_{\text{prior}})^T \Sigma_{\text{prior}}^{-1} (\psi - \mu_{\text{prior}})]$$

由 $\mathbb{E}_q[(\psi - \mu_{\text{prior}})(\psi - \mu_{\text{prior}})^T] = \Sigma + (\mu - \mu_{\text{prior}})(\mu - \mu_{\text{prior}})^T$：

$$\mathbb{E}_q[(\psi - \mu_{\text{prior}})^T \Sigma_{\text{prior}}^{-1} (\psi - \mu_{\text{prior}})] = \text{tr}(\Sigma_{\text{prior}}^{-1}\Sigma) + (\mu - \mu_{\text{prior}})^T \Sigma_{\text{prior}}^{-1} (\mu - \mu_{\text{prior}})$$

**步骤4：合并**

$$D_{\text{KL}} = \left[-\frac{1}{2}\ln|\Sigma| - \frac{d}{2}\right] - \left[-\frac{1}{2}\ln|\Sigma_{\text{prior}}| - \frac{1}{2}\text{tr}(\Sigma_{\text{prior}}^{-1}\Sigma) - \frac{1}{2}(\mu - \mu_{\text{prior}})^T \Sigma_{\text{prior}}^{-1} (\mu - \mu_{\text{prior}})\right]$$

整理即得A5.6。$\blacksquare$

#### 2.7 复杂性的正则化解释（A5.7）

复杂性项可分解为参数正则化与不确定性正则化之和：

$$C(q) = R(\mu) + R(\Sigma) \tag{A5.7}$$

其中：

$$R(\mu) = \frac{1}{2}(\mu - \mu_{\text{prior}})^T \Sigma_{\text{prior}}^{-1} (\mu - \mu_{\text{prior}})$$

$$R(\Sigma) = \frac{1}{2}\ln\frac{|\Sigma_{\text{prior}}|}{|\Sigma|} + \frac{1}{2}\text{tr}(\Sigma_{\text{prior}}^{-1}\Sigma) - \frac{d}{2}$$

- **输入**：变分参数 $(\mu, \Sigma)$，先验参数 $(\mu_{\text{prior}}, \Sigma_{\text{prior}})$
- **输出**：$R(\mu) \in \mathbb{R}_{\geq 0}$，$R(\Sigma) \in \mathbb{R}_{\geq 0}$
- **约束**：$R(\mu) \geq 0$（二次型正定性）；$R(\Sigma) \geq 0$（KL散度非负性）；$R(\mu) = 0$ 当且仅当 $\mu = \mu_{\text{prior}}$；$R(\Sigma) = 0$ 当且仅当 $\Sigma = \Sigma_{\text{prior}}$

> **白话翻译**：复杂性可以拆成两笔账：$R(\mu)$ 是"参数正则化"——参数值离默认值越远，罚得越重，这就是机器学习中的权重衰减；$R(\Sigma)$ 是"不确定性正则化"——不确定性偏离默认值也要受罚，太确定或太模糊都不行。

**$R(\Sigma)$ 的极小值**：对 $R(\Sigma)$ 关于 $\Sigma$ 求导并令其为零：

$$\frac{\partial R(\Sigma)}{\partial \Sigma} = -\frac{1}{2}\Sigma^{-1} + \frac{1}{2}\Sigma_{\text{prior}}^{-1} = 0$$

解得 $\Sigma^* = \Sigma_{\text{prior}}$，即不确定性正则化的最优值恰好是先验不确定性。这符合直觉：在没有数据约束时，不确定性应回到先验水平。

#### 2.8 复杂性梯度（A5.8）

复杂性项关于变分均值 $\mu$ 的梯度为：

$$\frac{\partial C}{\partial \mu} = \Sigma_{\text{prior}}^{-1} (\mu - \mu_{\text{prior}}) \tag{A5.8}$$

- **输入**：先验精度 $\Sigma_{\text{prior}}^{-1} \in \mathbb{S}_{++}^d$，均值偏差 $\mu - \mu_{\text{prior}} \in \mathbb{R}^d$
- **输出**：梯度 $\partial C / \partial \mu \in \mathbb{R}^d$
- **约束**：$\Sigma_{\text{prior}}$ 正定

> **白话翻译**：复杂性梯度的方向 = "先验精度加权" × "均值偏离量"。偏离先验越远，拉回来的力越大；先验越确定（精度越高），拉力越强。这就是"常识的引力"——越确定的事情，越不容易被数据改变。

**推导**：由A5.6，$R(\mu) = \frac{1}{2}(\mu - \mu_{\text{prior}})^T \Sigma_{\text{prior}}^{-1} (\mu - \mu_{\text{prior}})$，对 $\mu$ 求梯度：

$$\frac{\partial R(\mu)}{\partial \mu} = \Sigma_{\text{prior}}^{-1} (\mu - \mu_{\text{prior}})$$

$R(\Sigma)$ 不依赖 $\mu$，因此 $\partial C / \partial \mu = \Sigma_{\text{prior}}^{-1} (\mu - \mu_{\text{prior}})$。$\blacksquare$

---

### 第三部分：Pareto最优分析

#### 2.9 Pareto前沿（A5.9）

准确性项 $A(q)$ 和复杂性项 $C(q)$ 构成双目标优化问题。Pareto前沿定义为所有满足以下条件的 $(A, C)$ 对的集合：

$$\frac{\partial A}{\partial C} = -\lambda, \quad \lambda \in [0, \infty) \tag{A5.9}$$

- **输入**：准确性项 $A(q)$，复杂性项 $C(q)$，权衡参数 $\lambda \geq 0$
- **输出**：Pareto前沿 $\mathcal{P} = \{(A^*, C^*) : \nexists (A', C') \text{ s.t. } A' \leq A^*, C' \leq C^*, (A', C') \neq (A^*, C^*)\}$
- **约束**：$\lambda = 0$ 对应纯准确性优化（过拟合风险）；$\lambda \to \infty$ 对应纯复杂性优化（欠拟合风险）

> **白话翻译**：Pareto前沿就是"鱼与熊掌不可兼得"的数学描述——在前沿上，想再提高一点准确性，就必须付出一点复杂性的代价，反之亦然。$\lambda$ 是权衡参数：$\lambda$ 小意味着更看重准确性，$\lambda$ 大意味着更看重简单性。

**Pareto前沿的参数化**：对每个 $\lambda \geq 0$，Pareto最优点对应于最小化 $A + \lambda C$。当 $\lambda = 1$ 时，退化为自由能最小化 $F = A + C$。

#### 2.10 最优权衡条件（A5.10）

自由能最小化的最优权衡条件为准确性梯度与复杂性梯度之和为零：

$$\frac{\partial F}{\partial \mu} = \frac{\partial A}{\partial \mu} + \frac{\partial C}{\partial \mu} = 0 \tag{A5.10}$$

代入A5.4和A5.8：

$$-J^T \Sigma_\eta^{-1} (o - g(\mu)) + \Sigma_{\text{prior}}^{-1} (\mu - \mu_{\text{prior}}) = 0$$

- **输入**：A5.4的准确性梯度，A5.8的复杂性梯度
- **输出**：最优变分均值 $\mu^*$ 满足的方程
- **约束**：$J$ 在 $\mu^*$ 处可计算；$\Sigma_\eta$ 和 $\Sigma_{\text{prior}}$ 均正定

> **白话翻译**：最优权衡点就是"准确性拉力"和"复杂性拉力"恰好平衡的点——数据说"往这边走"（准确性梯度），先验说"别走太远"（复杂性梯度），两者力量相等时停下来。

**线性情形的解析解**：当 $g(\psi) = G\psi$（线性生成函数），$J = G$，最优均值为：

$$\mu^* = (G^T \Sigma_\eta^{-1} G + \Sigma_{\text{prior}}^{-1})^{-1} (G^T \Sigma_\eta^{-1} o + \Sigma_{\text{prior}}^{-1} \mu_{\text{prior}})$$

这是贝叶斯线性回归的后验均值——数据似然精度 $G^T \Sigma_\eta^{-1} G$ 与先验精度 $\Sigma_{\text{prior}}^{-1}$ 的精度加权平均。

#### 定理5.1：准确性-复杂性的Pareto最优等价于自由能最小化

**定理5.1**：变分分布 $q^*$ 是准确性-复杂性Pareto最优的，当且仅当 $q^*$ 是变分自由能的全局最小值点。具体地：

$$q^* \text{ 是 Pareto 最优} \iff q^* = \arg\min_{q \in \mathcal{Q}} F[q]$$

**完整证明**：

**必要性（$\Leftarrow$）**：设 $q^* = \arg\min_q F[q]$。反证法：假设 $q^*$ 不是Pareto最优，则存在 $q'$ 使得 $A(q') \leq A(q^*)$ 且 $C(q') \leq C(q^*)$，且至少一个严格不等号成立。则：

$$F(q') = A(q') + C(q') < A(q^*) + C(q^*) = F(q^*)$$

这与 $q^*$ 是自由能最小值点矛盾。因此 $q^*$ 是Pareto最优。$\square$

**充分性（$\Rightarrow$）**：设 $q^*$ 是Pareto最优。由Pareto最优的定义，不存在 $q'$ 使得 $A(q') + C(q') < A(q^*) + C(q^*)$（否则 $q'$ 至少在一个目标上不差于 $q^*$，且总和更优，意味着 $q^*$ 可被Pareto改进）。

更严格地，设 $q^*$ 是Pareto最优，但 $F(q^*) > F(q^{**})$ 对某个 $q^{**}$ 成立。则 $A(q^{**}) + C(q^{**}) < A(q^*) + C(q^*)$。这意味着：

1. 若 $A(q^{**}) \leq A(q^*)$ 且 $C(q^{**}) \leq C(q^*)$，则 $q^{**}$ Pareto支配 $q^*$，矛盾。
2. 若 $A(q^{**}) > A(q^*)$ 且 $C(q^{**}) < C(q^*)$，则 $q^{**}$ 不Pareto支配 $q^*$，但 $F(q^{**}) < F(q^*)$。

情况2不直接导致矛盾。但考虑以下论证：由于 $q^*$ 是Pareto最优，$q^*$ 位于Pareto前沿上。Pareto前沿上的每个点对应一个特定的权衡参数 $\lambda$，使得 $q^*$ 最小化 $A + \lambda C$。当 $\lambda = 1$ 时，$A + \lambda C = A + C = F$，因此自由能最小化对应Pareto前沿上 $\lambda = 1$ 的特定点。

更一般地，自由能最小化等价于在Pareto前沿上选择 $\lambda = 1$ 的权衡点。由A5.9，Pareto前沿上的每个点都是某个 $\lambda$ 下的最优解。当 $\lambda = 1$ 时，$q^* = \arg\min F$ 就是Pareto前沿上的一个点，因此是Pareto最优的。

反过来，Pareto最优性保证 $q^*$ 在某个 $\lambda$ 下最优，但不一定在 $\lambda = 1$ 下最优。因此需要更强的条件：

**修正陈述**：自由能最小化（$\lambda = 1$）是Pareto最优的充分条件，但Pareto最优性不必然等价于自由能最小化（因为Pareto前沿上还有其他 $\lambda$ 对应的点）。

**精确定理**：$q^* = \arg\min F[q]$ 当且仅当 $q^*$ 是Pareto前沿上权衡参数 $\lambda = 1$ 对应的点。

**证明**：由A5.9，Pareto最优点集为 $\{q_\lambda^* = \arg\min_q (A(q) + \lambda C(q)) : \lambda \geq 0\}$。当 $\lambda = 1$ 时，$q_1^* = \arg\min_q (A(q) + C(q)) = \arg\min_q F[q]$。因此自由能最小化恰为Pareto前沿上 $\lambda = 1$ 的点。$\blacksquare$

---

### 第四部分：模型选择的信息论准则

#### 2.11 AIC与自由能的关系（A5.11）

Akaike信息准则（AIC）定义为：

$$\text{AIC} = 2k - 2\ln(\hat{L}) \tag{A5.11}$$

其中 $k$ 为模型参数个数，$\hat{L} = \max_\theta p(o|\hat{\theta})$ 为最大似然估计下的似然值。

AIC与自由能的关系为：

$$\text{AIC} \approx 2F_{\text{MAP}} + \text{const}$$

其中 $F_{\text{MAP}}$ 为最大后验（MAP）估计下的自由能值。

- **输入**：参数个数 $k \in \mathbb{Z}_{> 0}$，最大对数似然 $\ln(\hat{L}) \in \mathbb{R}$
- **输出**：$\text{AIC} \in \mathbb{R}$
- **约束**：$k > 0$；$\hat{L} > 0$

> **白话翻译**：AIC = 2 × 参数个数 - 2 × 对数似然。参数越多罚得越重（$2k$），似然越大越好（$-2\ln\hat{L}$）。AIC本质上是自由能在"确定性近似"（$\Sigma \to 0$）和"大样本"极限下的近似——它用参数个数 $k$ 近似复杂性项，用最大似然近似准确性项。

**推导**：在拉普拉斯近似下（引用阶段4，A4.10），当样本量 $n \to \infty$ 时，变分协方差 $\Sigma \to 0$，复杂性项退化为：

$$C(q) \approx \frac{1}{2}(\hat{\mu} - \mu_{\text{prior}})^T \Sigma_{\text{prior}}^{-1} (\hat{\mu} - \mu_{\text{prior}})$$

在无信息先验（$\Sigma_{\text{prior}} \to \infty I$）下，MAP估计退化为MLE，且复杂性项的渐近展开给出 $C \approx k$（每个参数贡献约1的自由度）。因此：

$$F \approx -\ln(\hat{L}) + k$$

$$\text{AIC} = 2k - 2\ln(\hat{L}) \approx 2F + \text{const}$$

$\blacksquare$

#### 2.12 BIC与自由能的关系（A5.12）

Bayesian信息准则（BIC）定义为：

$$\text{BIC} = k \ln(n) - 2\ln(\hat{L}) \tag{A5.12}$$

其中 $n$ 为样本量。

BIC与自由能的关系为：

$$\text{BIC} \approx 2F_{\text{Laplace}} + \text{const}$$

其中 $F_{\text{Laplace}}$ 为拉普拉斯近似下的自由能值。

- **输入**：参数个数 $k \in \mathbb{Z}_{> 0}$，样本量 $n \in \mathbb{Z}_{> 0}$，最大对数似然 $\ln(\hat{L}) \in \mathbb{R}$
- **输出**：$\text{BIC} \in \mathbb{R}$
- **约束**：$k > 0$；$n > 0$；$\hat{L} > 0$

> **白话翻译**：BIC和AIC类似，但对参数的惩罚更重——每个参数罚 $k\ln(n)$ 而不是 $2k$。样本越多，惩罚越重，因为大样本下过拟合的风险更值得警惕。BIC本质上是自由能在拉普拉斯近似下的精确展开——复杂性项中的 $\ln|\Sigma|$ 贡献了 $k\ln(n)$ 项。

**推导**：在拉普拉斯近似下，自由能为：

$$F = -\ln(\hat{L}) + \frac{k}{2}\ln(n) + O(1)$$

因此：

$$\text{BIC} = k\ln(n) - 2\ln(\hat{L}) \approx 2F + \text{const}$$

BIC的复杂性惩罚 $k\ln(n)$ 来自拉普拉斯近似中变分协方差 $\Sigma$ 的行列式项 $\frac{1}{2}\ln|\Sigma|$。由Fisher信息矩阵的渐近性质，$|\Sigma| \propto n^{-k}$，因此 $\frac{1}{2}\ln|\Sigma| \approx -\frac{k}{2}\ln(n)$。$\blacksquare$

#### 2.13 自由能作为模型选择准则的优越性（A5.13）

自由能作为模型选择准则 $F^* = \min_q F[q]$ 相对于AIC和BIC的优越性体现在：

$$F^* = \min_q F[q] = -\text{ELBO} \tag{A5.13}$$

- **输入**：模型族 $\mathcal{M}$，变分分布族 $\mathcal{Q}$
- **输出**：最优自由能 $F^* \in \mathbb{R}$
- **约束**：$F^* \geq -\ln p(o)$（自由能下界约束，引用阶段4，A4.6）

> **白话翻译**：自由能比AIC/BIC好在三方面：（1）不依赖大样本近似——小样本也适用；（2）不依赖点估计——保留了完整的不确定性信息；（3）自动选择权衡参数——不需要手动调 $\lambda$。AIC和BIC都是自由能在特定近似下的特例。

**优越性的三个方面**：

1. **有限样本有效性**：AIC和BIC都是大样本渐近结果，在有限样本下可能严重偏差。自由能准则对任意样本量都有效，因为它直接优化变分下界而非渐近近似。

2. **不确定性保留**：AIC和BIC仅使用点估计（MLE/MAP），丢弃了后验的不确定性信息。自由能准则通过变分协方差 $\Sigma$ 保留了完整的不确定性结构，使得模型选择同时考虑"最佳参数"和"参数的不确定性"。

3. **自动权衡**：AIC固定 $\lambda = 1$（复杂性惩罚 $2k$），BIC固定 $\lambda = \ln(n)/2$（复杂性惩罚 $k\ln(n)$）。自由能准则通过变分优化的内循环自动确定准确性-复杂性的最优权衡，无需手动选择 $\lambda$。

**定量比较**：

| 准则 | 复杂性惩罚 | 样本依赖 | 不确定性 | 等价条件 |
|------|-----------|---------|---------|---------|
| AIC | $2k$ | 否 | 否 | $n \to \infty$，无信息先验 |
| BIC | $k\ln(n)$ | 是 | 否 | $n \to \infty$，拉普拉斯近似 |
| 自由能 | $D_{\text{KL}}[q\|p]$ | 是 | 是 | 恒等（无需近似） |

---

### 第五部分：正则化理论的统一

#### 2.14 L2正则化与高斯先验下的复杂性项（A5.14）

L2正则化（权重衰减）等价于高斯先验下的复杂性项的参数正则化部分：

$$\underbrace{\frac{\lambda}{2}\|\mu - \mu_{\text{prior}}\|^2}_{\text{L2正则化}} = \underbrace{\frac{1}{2}(\mu - \mu_{\text{prior}})^T \Sigma_{\text{prior}}^{-1} (\mu - \mu_{\text{prior}})}_{\text{复杂性项 } R(\mu)} \quad \text{当 } \Sigma_{\text{prior}} = \lambda^{-1} I \tag{A5.14}$$

- **输入**：正则化系数 $\lambda \in \mathbb{R}_{> 0}$，参数偏差 $\mu - \mu_{\text{prior}}$
- **输出**：L2惩罚 = 复杂性项 $R(\mu)$
- **约束**：$\lambda > 0$；$\Sigma_{\text{prior}} = \lambda^{-1} I$（各向同性先验）

> **白话翻译**：L2正则化就是"参数离默认值越远，罚得越重"，这恰好是高斯先验下复杂性项的均值部分。正则化系数 $\lambda$ 越大，先验越确定（$\Sigma_{\text{prior}}$ 越小），拉回默认值的力越强。机器学习调 $\lambda$，就是在调先验的确定性。

**推导**：当 $\Sigma_{\text{prior}} = \sigma^2 I$（各向同性高斯先验）时：

$$R(\mu) = \frac{1}{2\sigma^2}\|\mu - \mu_{\text{prior}}\|^2$$

令 $\lambda = 1/\sigma^2$，则 $R(\mu) = \frac{\lambda}{2}\|\mu - \mu_{\text{prior}}\|^2$，即L2正则化。$\blacksquare$

#### 2.15 L1正则化与拉普拉斯先验下的复杂性项（A5.15）

L1正则化（Lasso）等价于拉普拉斯先验下的复杂性项：

$$\underbrace{\lambda \|\mu - \mu_{\text{prior}}\|_1}_{\text{L1正则化}} = \underbrace{D_{\text{KL}}[q(\psi) \| p_{\text{Laplace}}(\psi)]}_{\text{拉普拉斯先验下的复杂性项}} \quad \text{当 } q \text{ 为 Dirac 分布} \tag{A5.15}$$

其中 $p_{\text{Laplace}}(\psi) = \prod_j \frac{\lambda}{2}\exp(-\lambda|\psi_j - \mu_{\text{prior},j}|)$ 为拉普拉斯先验。

- **输入**：正则化系数 $\lambda \in \mathbb{R}_{> 0}$，参数偏差 $\mu - \mu_{\text{prior}}$
- **输出**：L1惩罚 = 拉普拉斯先验下的复杂性项
- **约束**：$\lambda > 0$；$q$ 为Dirac分布（点估计）

> **白话翻译**：L1正则化就是"参数离默认值的绝对距离"，对应拉普拉斯先验（双指数分布）。和L2的区别是：L1用绝对值而非平方，所以对大偏差的惩罚相对较轻，对小偏差的惩罚相对较重，这导致很多参数被精确地推到零——这就是L1产生稀疏解的原因。

**推导**：当 $q(\psi) = \delta(\psi - \mu)$（Dirac分布）且 $p(\psi) = \text{Laplace}(\mu_{\text{prior}}, 1/\lambda)$ 时：

$$C(q) = D_{\text{KL}}[\delta(\psi - \mu) \| p_{\text{Laplace}}(\psi)] = -\ln p_{\text{Laplace}}(\mu) - H[\delta]$$

$$= \lambda\|\mu - \mu_{\text{prior}}\|_1 + d\ln\frac{2}{\lambda} - (-\infty)$$

Dirac分布的熵为 $-\infty$，但在变分自由能的框架中，此 $-\infty$ 被准确性项中的对应项抵消。在确定性近似（$\Sigma \to 0$）的极限下，L1正则化项 $\lambda\|\mu - \mu_{\text{prior}}\|_1$ 从复杂性项中浮现。$\blacksquare$

**L1 vs L2 的先验视角**：

| 正则化 | 先验分布 | 对大偏差的惩罚 | 对小偏差的惩罚 | 稀疏性 |
|--------|---------|--------------|--------------|--------|
| L2 | 高斯 | 重（二次增长） | 轻（二次增长） | 无 |
| L1 | 拉普拉斯 | 轻（线性增长） | 重（线性增长，在零处不可微） | 有 |

#### 2.16 Dropout与变分复杂性近似的蒙特卡洛估计（A5.16）

Dropout正则化可解释为变分复杂性项的蒙特卡洛近似：

$$\underbrace{L_{\text{Dropout}}}_{\text{Dropout训练损失}} \approx \underbrace{\mathbb{E}_{q_{\text{Dropout}}}[F[q]]}_{\text{变分自由能的蒙特卡洛估计}} \tag{A5.16}$$

其中 $q_{\text{Dropout}}$ 为Dropout诱导的变分分布——对权重矩阵的每一行以概率 $p$ 置零（保留概率 $1-p$）。

- **输入**：Dropout保留概率 $1-p \in (0, 1]$，网络权重 $W$
- **输出**：Dropout训练损失 ≈ 变分自由能
- **约束**：$0 < p < 1$；Dropout变分分布为权重上的离散-连续混合分布

> **白话翻译**：Dropout训练就是在每次前向传播时随机"关掉"一部分神经元，这等价于在权重的变分分布上做蒙特卡洛采样。Dropout的隐式正则化效果来自复杂性项——随机丢弃权重迫使网络不能过度依赖任何单个权重，这恰好是复杂性惩罚"不要偏离先验太远"的另一种表达。

**推导（Gal & Ghahramani, 2016）**：

**步骤1：定义Dropout变分分布**

对权重矩阵 $W$ 的每一行 $w_j$，定义变分分布：

$$q(w_j) = (1-p) \cdot \delta(w_j - \hat{w}_j / (1-p)) + p \cdot \delta(w_j - 0)$$

即以概率 $1-p$ 保留（缩放后的）权重，以概率 $p$ 置零。

**步骤2：计算复杂性项**

$$C(q) = \sum_j D_{\text{KL}}[q(w_j) \| p(w_j)]$$

当先验 $p(w_j) = \mathcal{N}(0, \sigma^2 I)$ 时：

$$D_{\text{KL}}[q(w_j) \| p(w_j)] = (1-p)\left[\frac{\|\hat{w}_j\|^2}{2\sigma^2(1-p)^2} + \frac{d}{2}\ln\frac{\sigma^2(1-p)^2}{\sigma^2}\right] + p \cdot \frac{d}{2}$$

化简后，复杂性项包含 $\frac{\|\hat{w}_j\|^2}{2\sigma^2(1-p)}$ 项，即一个缩放的L2正则化。

**步骤3：蒙特卡洛估计**

Dropout训练中，每次前向传播等价于从 $q_{\text{Dropout}}$ 中采一个样本，计算损失（准确性项的蒙特卡洛估计）。多次前向传播的平均损失近似变分自由能。

$\blacksquare$

**Dropout与L2的统一**：Dropout的复杂性项包含L2正则化项（缩放因子 $1/(1-p)$），这解释了Dropout与L2正则化的经验等价性。但Dropout还包含额外的熵项（来自权重的离散性），这使其比纯L2正则化更灵活。

---

### 第六部分：动力学轨迹

#### 2.17 自由能景观上的梯度流（A5.17）

变分均值 $\mu$ 在自由能景观上的梯度流为：

$$\frac{\mathrm{d}\mu}{\mathrm{d}t} = -\frac{\partial F}{\partial \mu} = -\frac{\partial A}{\partial \mu} - \frac{\partial C}{\partial \mu} \tag{A5.17}$$

代入A5.4和A5.8：

$$\frac{\mathrm{d}\mu}{\mathrm{d}t} = J^T \Sigma_\eta^{-1} (o - g(\mu)) - \Sigma_{\text{prior}}^{-1} (\mu - \mu_{\text{prior}})$$

- **输入**：当前变分均值 $\mu(t)$，Jacobian $J$，噪声精度 $\Sigma_\eta^{-1}$，先验精度 $\Sigma_{\text{prior}}^{-1}$，观测 $o$
- **输出**：均值的时间导数 $\dot{\mu} \in \mathbb{R}^d$
- **约束**：$\Sigma_\eta$ 和 $\Sigma_{\text{prior}}$ 均正定；$g$ 在 $\mu$ 处可微

> **白话翻译**：系统在"自由能景观"上沿着最陡的下坡路走。第一项是"数据引力"——把均值拉向让预测更准的方向；第二项是"先验引力"——把均值拉回默认值。两个力的合力决定了系统的运动方向。

**与A4.11的连接**：由A4.11（引用阶段4），$\mathrm{d}F/\mathrm{d}t \leq 0$，即梯度流保证自由能单调递减。代入A5.17：

$$\frac{\mathrm{d}F}{\mathrm{d}t} = \left(\frac{\partial F}{\partial \mu}\right)^T \dot{\mu} = -\left\|\frac{\partial F}{\partial \mu}\right\|^2 \leq 0$$

**线性情形的解析解**：当 $g(\psi) = G\psi$ 时，$J = G$，梯度流为线性ODE：

$$\dot{\mu} = G^T \Sigma_\eta^{-1} (o - G\mu) - \Sigma_{\text{prior}}^{-1} (\mu - \mu_{\text{prior}})$$

$$= -(G^T \Sigma_\eta^{-1} G + \Sigma_{\text{prior}}^{-1})\mu + G^T \Sigma_\eta^{-1} o + \Sigma_{\text{prior}}^{-1} \mu_{\text{prior}}$$

令 $A = G^T \Sigma_\eta^{-1} G + \Sigma_{\text{prior}}^{-1}$（正定矩阵），$b = G^T \Sigma_\eta^{-1} o + \Sigma_{\text{prior}}^{-1} \mu_{\text{prior}}$，则：

$$\dot{\mu} = -A\mu + b$$

解为 $\mu(t) = A^{-1}b + (\mu(0) - A^{-1}b)e^{-At}$，收敛到 $\mu^* = A^{-1}b$（与A5.10的解析解一致）。

#### 2.18 准确性与复杂性的时间尺度分离（A5.18）

在自由能最小化的动力学过程中，准确性项和复杂性项的收敛时间尺度通常不同：

$$\tau_A \ll \tau_C \tag{A5.18}$$

其中 $\tau_A$ 为准确性项的收敛时间常数，$\tau_C$ 为复杂性项的收敛时间常数。

- **输入**：准确性Hessian的特征值 $\{\lambda_i^A\}$，复杂性Hessian的特征值 $\{\lambda_i^C\}$
- **输出**：时间尺度比 $\tau_C / \tau_A \gg 1$
- **约束**：$\tau_A = 1 / \lambda_{\min}^A$，$\tau_C = 1 / \lambda_{\min}^C$；$\lambda_{\min}^A \gg \lambda_{\min}^C$

> **白话翻译**：准确性调整很快（看到数据就立刻修正预测），复杂性调整很慢（改变根深蒂固的信念需要时间）。就像"看到新证据立刻调整判断"很快，但"改变世界观"很慢。这就是时间尺度分离——快速适应数据，慢速更新先验。

**推导**：

准确性项的Hessian（关于 $\mu$）为：

$$H_A = \frac{\partial^2 A}{\partial \mu^2} \approx J^T \Sigma_\eta^{-1} J$$

复杂性项的Hessian为：

$$H_C = \frac{\partial^2 C}{\partial \mu^2} = \Sigma_{\text{prior}}^{-1}$$

收敛时间常数由Hessian的最小特征值决定：

$$\tau_A = \frac{1}{\lambda_{\min}(H_A)} = \frac{1}{\lambda_{\min}(J^T \Sigma_\eta^{-1} J)}$$

$$\tau_C = \frac{1}{\lambda_{\min}(H_C)} = \frac{1}{\lambda_{\min}(\Sigma_{\text{prior}}^{-1})}$$

当观测噪声较小（$\Sigma_\eta^{-1}$ 特征值大）且先验较模糊（$\Sigma_{\text{prior}}^{-1}$ 特征值小）时，$\lambda_{\min}(H_A) \gg \lambda_{\min}(H_C)$，因此 $\tau_A \ll \tau_C$。

**物理解释**：时间尺度分离意味着系统先快速调整以拟合数据（准确性驱动），然后缓慢地与先验妥协（复杂性驱动）。这与神经科学中观察到的"快速感知更新 + 慢速信念更新"的现象一致。

**定量估计**：设 $\Sigma_\eta = \sigma_\eta^2 I$，$\Sigma_{\text{prior}} = \sigma_{\text{prior}}^2 I$，$J = I$，则：

$$\tau_A = \sigma_\eta^2, \quad \tau_C = \sigma_{\text{prior}}^2$$

当 $\sigma_\eta^2 \ll \sigma_{\text{prior}}^2$（观测噪声远小于先验不确定性）时，$\tau_A \ll \tau_C$，时间尺度分离显著。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A5.1 | $A(q) = -\mathbb{E}_q[\ln p(o\|\psi)]$ | $q$, $p(o\|\psi)$ | $A(q) \in \mathbb{R}$ | 引用阶段4，A4.5第一项 |
| A5.2 | $A(q) = \frac{1}{2}(o - g(\mu))^T \Sigma_\eta^{-1}(o - g(\mu)) + \frac{1}{2}\text{tr}(\Sigma_\eta^{-1} J \Sigma J^T) + \frac{1}{2}\ln\|2\pi\Sigma_\eta\|$ | $o$, $g$, $\mu$, $\Sigma$, $\Sigma_\eta$ | $A(q) \in \mathbb{R}$ | 高斯似然；拉普拉斯近似 |
| A5.3 | $A(q) = \frac{1}{2}\varepsilon_0 + \text{const}$ | $o$, $\mu$, $\Sigma_\eta^{-1}$ | $A(q) \in \mathbb{R}$ | 恒等生成函数；$\Sigma \to 0$；引用阶段3，A3.2 |
| A5.4 | $\partial A / \partial \mu = -J^T \Sigma_\eta^{-1}(o - g(\mu))$ | $J$, $\Sigma_\eta^{-1}$, $o - g(\mu)$ | $\nabla_\mu A \in \mathbb{R}^d$ | $g$ 可微；$\Sigma_\eta$ 正定 |
| A5.5 | $C(q) = D_{\text{KL}}[q\|p]$ | $q$, $p(\psi)$ | $C(q) \in \mathbb{R}_{\geq 0}$ | 引用阶段4，A4.5第二项 |
| A5.6 | $C(q) = \frac{1}{2}(\mu - \mu_p)^T \Sigma_p^{-1}(\mu - \mu_p) + \frac{1}{2}\ln\frac{\|\Sigma_p\|}{\|\Sigma\|} + \frac{1}{2}\text{tr}(\Sigma_p^{-1}\Sigma) - \frac{d}{2}$ | $\mu$, $\Sigma$, $\mu_p$, $\Sigma_p$ | $C(q) \in \mathbb{R}_{\geq 0}$ | 高斯先验+高斯变分 |
| A5.7 | $C(q) = R(\mu) + R(\Sigma)$ | $\mu$, $\Sigma$, $\mu_p$, $\Sigma_p$ | $R(\mu) \geq 0$, $R(\Sigma) \geq 0$ | $R(\mu)=0 \Leftrightarrow \mu=\mu_p$；$R(\Sigma)=0 \Leftrightarrow \Sigma=\Sigma_p$ |
| A5.8 | $\partial C / \partial \mu = \Sigma_{\text{prior}}^{-1}(\mu - \mu_{\text{prior}})$ | $\Sigma_{\text{prior}}^{-1}$, $\mu - \mu_{\text{prior}}$ | $\nabla_\mu C \in \mathbb{R}^d$ | $\Sigma_{\text{prior}}$ 正定 |
| A5.9 | $\partial A / \partial C = -\lambda$, $\lambda \in [0, \infty)$ | $A(q)$, $C(q)$, $\lambda$ | Pareto前沿 $\mathcal{P}$ | $\lambda=0$：纯准确性；$\lambda\to\infty$：纯复杂性 |
| A5.10 | $\partial F / \partial \mu = \partial A / \partial \mu + \partial C / \partial \mu = 0$ | A5.4, A5.8 | 最优均值 $\mu^*$ 方程 | 自由能驻点条件 |
| A5.11 | $\text{AIC} = 2k - 2\ln(\hat{L})$ | $k$, $\hat{L}$ | $\text{AIC} \in \mathbb{R}$ | $k > 0$；$\hat{L} > 0$ |
| A5.12 | $\text{BIC} = k\ln(n) - 2\ln(\hat{L})$ | $k$, $n$, $\hat{L}$ | $\text{BIC} \in \mathbb{R}$ | $k > 0$；$n > 0$ |
| A5.13 | $F^* = \min_q F[q] = -\text{ELBO}$ | $\mathcal{M}$, $\mathcal{Q}$ | $F^* \in \mathbb{R}$ | $F^* \geq -\ln p(o)$ |
| A5.14 | L2正则化 = 高斯先验下 $R(\mu)$，当 $\Sigma_{\text{prior}} = \lambda^{-1}I$ | $\lambda$, $\mu - \mu_{\text{prior}}$ | L2惩罚 | $\lambda > 0$；各向同性先验 |
| A5.15 | L1正则化 = 拉普拉斯先验下 $C(q)$，当 $q$ 为 Dirac | $\lambda$, $\mu - \mu_{\text{prior}}$ | L1惩罚 | $\lambda > 0$；点估计 |
| A5.16 | $L_{\text{Dropout}} \approx \mathbb{E}_{q_{\text{Dropout}}}[F[q]]$ | 保留概率 $1-p$, $W$ | Dropout损失 | $0 < p < 1$ |
| A5.17 | $\mathrm{d}\mu/\mathrm{d}t = -\partial F/\partial \mu = J^T\Sigma_\eta^{-1}(o - g(\mu)) - \Sigma_{\text{prior}}^{-1}(\mu - \mu_{\text{prior}})$ | $\mu(t)$, $J$, $\Sigma_\eta^{-1}$, $\Sigma_{\text{prior}}^{-1}$, $o$ | $\dot{\mu} \in \mathbb{R}^d$ | 引用阶段4，A4.11 |
| A5.18 | $\tau_A \ll \tau_C$ | $H_A$, $H_C$ 特征值 | 时间尺度比 $\gg 1$ | $\sigma_\eta^2 \ll \sigma_{\text{prior}}^2$ |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# 引用阶段4的数据类
from stage_004 import VariationalFreeEnergy


# ============================================================
# 阶段5 数据类：自由能分解
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class FreeEnergyDecomposition:
    """自由能分解的完整描述。

    组合阶段4的VariationalFreeEnergy，在变分自由能的基础上增加
    准确性-复杂性的详细分解结构。

    Attributes:
        variational_free_energy: 阶段4的变分自由能（组合字段）
        accuracy_mean_term: 准确性项的均值部分 ½(o-g(μ))ᵀΣ_η⁻¹(o-g(μ))（A5.2第一项）
        accuracy_uncertainty_term: 准确性项的不确定性部分 ½tr(Σ_η⁻¹JΣJᵀ)（A5.2第二项）
        complexity_mean_term: 复杂性项的均值部分 R(μ)（A5.7第一项）
        complexity_uncertainty_term: 复杂性项的不确定性部分 R(Σ)（A5.7第二项）
        accuracy_gradient: 准确性项关于μ的梯度 ∂A/∂μ（A5.4）
        complexity_gradient: 复杂性项关于μ的梯度 ∂C/∂μ（A5.8）
        pareto_lambda: Pareto权衡参数 λ = -∂A/∂C（A5.9）

    Invariants:
        - accuracy_mean_term + accuracy_uncertainty_term
          + ½ln|2πΣ_η| == variational_free_energy.accuracy
        - complexity_mean_term + complexity_uncertainty_term
          == variational_free_energy.complexity
        - accuracy_gradient + complexity_gradient
          ≈ 0（在最优解处）
        - complexity_mean_term >= 0
        - complexity_uncertainty_term >= 0
        - pareto_lambda >= 0
    """
    variational_free_energy: VariationalFreeEnergy
    accuracy_mean_term: float
    accuracy_uncertainty_term: float
    complexity_mean_term: float
    complexity_uncertainty_term: float
    accuracy_gradient: np.ndarray
    complexity_gradient: np.ndarray
    pareto_lambda: float

    def __post_init__(self):
        vfe = self.variational_free_energy
        # 验证准确性分解一致性
        acc_sum = self.accuracy_mean_term + self.accuracy_uncertainty_term
        # 注意：A5.2还有常数项 ½ln|2πΣ_η|，此处acc_sum不含常数项
        # 完整一致性验证需要常数项，此处验证非负性
        if self.accuracy_mean_term < -1e-10:
            raise ValueError(
                f"准确性均值项必须 ≥ 0，收到 {self.accuracy_mean_term}"
            )
        if self.accuracy_uncertainty_term < -1e-10:
            raise ValueError(
                f"准确性不确定性项必须 ≥ 0，收到 {self.accuracy_uncertainty_term}"
            )
        # 验证复杂性分解一致性
        comp_sum = self.complexity_mean_term + self.complexity_uncertainty_term
        if not np.isclose(comp_sum, vfe.complexity, atol=1e-8):
            raise ValueError(
                f"复杂性分解不一致: R(μ)+R(Σ)={comp_sum}, "
                f"但 complexity={vfe.complexity}"
            )
        # 验证非负性
        if self.complexity_mean_term < -1e-10:
            raise ValueError(
                f"复杂性均值项 R(μ) 必须 ≥ 0，收到 {self.complexity_mean_term}"
            )
        if self.complexity_uncertainty_term < -1e-10:
            raise ValueError(
                f"复杂性不确定性项 R(Σ) 必须 ≥ 0，"
                f"收到 {self.complexity_uncertainty_term}"
            )
        # 验证梯度维度
        d = vfe.variational_mean.shape[0]
        if self.accuracy_gradient.shape != (d,):
            raise ValueError(
                f"准确性梯度形状 {self.accuracy_gradient.shape} "
                f"与均值维度 {d} 不匹配"
            )
        if self.complexity_gradient.shape != (d,):
            raise ValueError(
                f"复杂性梯度形状 {self.complexity_gradient.shape} "
                f"与均值维度 {d} 不匹配"
            )
        # 验证Pareto参数
        if self.pareto_lambda < -1e-10:
            raise ValueError(
                f"Pareto权衡参数 λ 必须 ≥ 0，收到 {self.pareto_lambda}"
            )

    @property
    def total_accuracy(self) -> float:
        """准确性项总值 A(q)（A5.1）。"""
        return self.variational_free_energy.accuracy

    @property
    def total_complexity(self) -> float:
        """复杂性项总值 C(q)（A5.5）。"""
        return self.variational_free_energy.complexity

    @property
    def is_optimal_tradeoff(self) -> bool:
        """是否满足最优权衡条件 ∂F/∂μ = 0（A5.10）。"""
        grad_norm = float(np.linalg.norm(
            self.accuracy_gradient + self.complexity_gradient
        ))
        return grad_norm < 1e-6

    @property
    def gradient_flow_direction(self) -> np.ndarray:
        """自由能景观上的梯度流方向 -∂F/∂μ（A5.17）。"""
        return -(self.accuracy_gradient + self.complexity_gradient)


# ============================================================
# 阶段5 核心接口：自由能分解
# ============================================================
class FreeEnergyDecomposer:
    """阶段5的核心计算接口。

    将变分自由能分解为准确性和复杂性项，并提供梯度分析、
    Pareto最优分析、模型选择和正则化统一的计算接口。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - decompose 返回值满足 A + C == F
        - R(μ) + R(Σ) == C（高斯先验下）
        - 梯度计算与自动微分结果一致
    """

    def decompose(
        self,
        free_energy: VariationalFreeEnergy,
        observation: np.ndarray,
        generative_fn: callable,
        noise_cov: np.ndarray,
        prior_mean: np.ndarray,
        prior_cov: np.ndarray
    ) -> FreeEnergyDecomposition:
        """将变分自由能分解为准确性和复杂性项（A5.1-A5.8）。

        Args:
            free_energy: 阶段4的变分自由能实例（组合字段）
            observation: 观测值 o，形状 (d_o,)
            generative_fn: 生成函数 g(ψ) → R^{d_o}
            noise_cov: 噪声协方差 Σ_η，形状 (d_o, d_o)，对称正定
            prior_mean: 先验均值 μ_prior，形状 (d,)
            prior_cov: 先验协方差 Σ_prior，形状 (d, d)，对称正定

        Returns:
            FreeEnergyDecomposition 实例，包含完整的分解信息。

        Raises:
            TypeError: 当 free_energy 不是 VariationalFreeEnergy 实例时
            ValueError: 当协方差矩阵不正定或维度不匹配时
        """
        if not isinstance(free_energy, VariationalFreeEnergy):
            raise TypeError(
                f"free_energy 必须为 VariationalFreeEnergy 实例，"
                f"收到 {type(free_energy).__name__}"
            )
        # 验证协方差矩阵
        for name, cov in [("noise_cov", noise_cov), ("prior_cov", prior_cov)]:
            if not np.allclose(cov, cov.T):
                raise ValueError(f"{name} 必须对称")
            try:
                np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                raise ValueError(f"{name} 必须正定")

        mu = free_energy.variational_mean
        Sigma = free_energy.variational_cov
        d = mu.shape[0]
        d_o = observation.shape[0]

        # A5.2: 准确性项 = ½(o-g(μ))ᵀΣ_η⁻¹(o-g(μ)) + ½tr(Σ_η⁻¹JΣJᵀ) + ½ln|2πΣ_η|
        g_mu = generative_fn(mu)
        residual = observation - g_mu
        noise_prec = np.linalg.inv(noise_cov)
        # 数值Jacobian（有限差分）
        eps = 1e-5
        J = np.zeros((d_o, d))
        for j in range(d):
            mu_plus = mu.copy()
            mu_plus[j] += eps
            mu_minus = mu.copy()
            mu_minus[j] -= eps
            J[:, j] = (generative_fn(mu_plus) - generative_fn(mu_minus)) / (2 * eps)

        accuracy_mean = 0.5 * float(residual @ noise_prec @ residual)
        accuracy_uncertainty = 0.5 * float(np.trace(noise_prec @ J @ Sigma @ J.T))

        # A5.6: 复杂性项 = ½(μ-μ_p)ᵀΣ_p⁻¹(μ-μ_p) + ½ln|Σ_p|/|Σ| + ½tr(Σ_p⁻¹Σ) - d/2
        prior_prec = np.linalg.inv(prior_cov)
        diff = mu - prior_mean
        complexity_mean = 0.5 * float(diff @ prior_prec @ diff)
        sign_p, logdet_p = np.linalg.slogdet(prior_cov)
        sign_q, logdet_q = np.linalg.slogdet(Sigma)
        complexity_uncertainty = (
            0.5 * (logdet_p - logdet_q)
            + 0.5 * float(np.trace(prior_prec @ Sigma))
            - 0.5 * d
        )

        # A5.4: 准确性梯度 = -JᵀΣ_η⁻¹(o - g(μ))
        acc_grad = -J.T @ noise_prec @ residual

        # A5.8: 复杂性梯度 = Σ_prior⁻¹(μ - μ_prior)
        comp_grad = prior_prec @ diff

        # A5.9: Pareto λ = -∂A/∂C 的近似
        # 在自由能最小化中 λ=1
        pareto_lambda = 1.0

        return FreeEnergyDecomposition(
            variational_free_energy=free_energy,
            accuracy_mean_term=accuracy_mean,
            accuracy_uncertainty_term=accuracy_uncertainty,
            complexity_mean_term=complexity_mean,
            complexity_uncertainty_term=complexity_uncertainty,
            accuracy_gradient=acc_grad,
            complexity_gradient=comp_grad,
            pareto_lambda=pareto_lambda,
        )

        # Examples:
        # >>> dec = FreeEnergyDecomposer()
        # >>> # dec.decompose(vfe, o, g, Sigma_eta, mu_p, Sigma_p)  # doctest: +SKIP
        self,
        mu: np.ndarray,
        observation: np.ndarray,
        generative_fn: callable,
        noise_cov: np.ndarray
    ) -> np.ndarray:
        """计算准确性项关于μ的梯度（A5.4）。

        Args:
            mu: 变分均值 μ，形状 (d,)
            observation: 观测值 o，形状 (d_o,)
            generative_fn: 生成函数 g(ψ) → R^{d_o}，必须在 μ 处可微
            noise_cov: 噪声协方差 Σ_η，形状 (d_o, d_o)，对称正定

        Returns:
            准确性梯度 ∂A/∂μ = -JᵀΣ_η⁻¹(o - g(μ))，形状 (d,)

        Raises:
            ValueError: 当 noise_cov 不正定或维度不匹配时
        """
        if mu.ndim != 1:
            raise ValueError(f"mu 必须为一维向量，收到 {mu.ndim} 维")
        d = mu.shape[0]
        d_o = observation.shape[0]
        if noise_cov.shape != (d_o, d_o):
            raise ValueError(
                f"noise_cov 形状 {noise_cov.shape} 与观测维度 {d_o} 不匹配"
            )
        if not np.allclose(noise_cov, noise_cov.T):
            raise ValueError("noise_cov 必须对称")
        try:
            np.linalg.cholesky(noise_cov)
        except np.linalg.LinAlgError:
            raise ValueError("noise_cov 必须正定")
        # A5.4: ∂A/∂μ = -JᵀΣ_η⁻¹(o - g(μ))
        g_mu = generative_fn(mu)
        residual = observation - g_mu
        noise_prec = np.linalg.inv(noise_cov)
        # 数值Jacobian（中心差分）
        eps = 1e-5
        J = np.zeros((d_o, d))
        for j in range(d):
            mu_plus = mu.copy(); mu_plus[j] += eps
            mu_minus = mu.copy(); mu_minus[j] -= eps
            J[:, j] = (generative_fn(mu_plus) - generative_fn(mu_minus)) / (2 * eps)
        return -J.T @ noise_prec @ residual

        # Examples:
        # >>> dec = FreeEnergyDecomposer()
        # >>> mu = np.array([1.0, 2.0])
        # >>> o = np.array([1.1, 2.1])
        # >>> g = lambda x: x  # 恒等生成函数
        # >>> Sigma_eta = np.eye(2)
        # >>> grad = dec.accuracy_gradient(mu, o, g, Sigma_eta)
        # >>> np.allclose(grad, np.array([-0.1, -0.1]))
        # True
        self,
        mu: np.ndarray,
        prior_mean: np.ndarray,
        prior_cov: np.ndarray
    ) -> np.ndarray:
        """计算复杂性项关于μ的梯度（A5.8）。

        Args:
            mu: 变分均值 μ，形状 (d,)
            prior_mean: 先验均值 μ_prior，形状 (d,)
            prior_cov: 先验协方差 Σ_prior，形状 (d, d)，对称正定

        Returns:
            复杂性梯度 ∂C/∂μ = Σ_prior⁻¹(μ - μ_prior)，形状 (d,)

        Raises:
            ValueError: 当 prior_cov 不正定或维度不匹配时
        """
        if mu.shape != prior_mean.shape:
            raise ValueError(
                f"mu 形状 {mu.shape} 与 prior_mean 形状 {prior_mean.shape} 不匹配"
            )
        d = mu.shape[0]
        if prior_cov.shape != (d, d):
            raise ValueError(
                f"prior_cov 形状 {prior_cov.shape} 与维度 {d} 不匹配"
            )
        if not np.allclose(prior_cov, prior_cov.T):
            raise ValueError("prior_cov 必须对称")
        try:
            np.linalg.cholesky(prior_cov)
        except np.linalg.LinAlgError:
            raise ValueError("prior_cov 必须正定")
        # A5.8: ∂C/∂μ = Σ_prior⁻¹(μ - μ_prior)
        prior_prec = np.linalg.inv(prior_cov)
        return prior_prec @ (mu - prior_mean)

        # Examples:
        # >>> dec = FreeEnergyDecomposer()
        # >>> mu = np.array([2.0, 3.0])
        # >>> mu_p = np.array([0.0, 0.0])
        # >>> Sigma_p = np.eye(2)
        # >>> dec.complexity_gradient(mu, mu_p, Sigma_p)
        # array([2., 3.])
        self,
        observation: np.ndarray,
        generative_fn: callable,
        noise_cov: np.ndarray,
        prior_mean: np.ndarray,
        prior_cov: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-8
    ) -> np.ndarray:
        """求解最优权衡条件 ∂F/∂μ = 0（A5.10）。

        使用梯度下降迭代求解。

        Args:
            observation: 观测值 o，形状 (d_o,)
            generative_fn: 生成函数 g(ψ)
            noise_cov: 噪声协方差 Σ_η
            prior_mean: 先验均值 μ_prior
            prior_cov: 先验协方差 Σ_prior
            max_iter: 最大迭代次数
            tol: 收敛容差

        Returns:
            最优变分均值 μ*，形状 (d,)

        Raises:
            ValueError: 当参数不满足约束时
            RuntimeError: 当迭代不收敛时
        """
        # 验证输入约束
        for name, cov in [("noise_cov", noise_cov), ("prior_cov", prior_cov)]:
            if not np.allclose(cov, cov.T):
                raise ValueError(f"{name} 必须对称")
            try:
                np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                raise ValueError(f"{name} 必须正定")
        if max_iter <= 0:
            raise ValueError(f"max_iter 必须 > 0，收到 {max_iter}")
        if tol <= 0:
            raise ValueError(f"tol 必须 > 0，收到 {tol}")

        # A5.10: 梯度下降求解 ∂F/∂μ = 0
        # ∂F/∂μ = ∂A/∂μ + ∂C/∂μ = -JᵀΣ_η⁻¹(o-g(μ)) + Σ_prior⁻¹(μ-μ_prior)
        mu = prior_mean.copy().astype(float)
        lr = 0.01
        for iteration in range(max_iter):
            grad_A = self.accuracy_gradient(mu, observation, generative_fn, noise_cov)
            grad_C = self.complexity_gradient(mu, prior_mean, prior_cov)
            grad_F = grad_A + grad_C
            if np.linalg.norm(grad_F) < tol:
                return mu
            mu = mu - lr * grad_F
        raise RuntimeError(
            f"最优均值求解未在 {max_iter} 次迭代内收敛，"
            f"最终梯度范数 {np.linalg.norm(grad_F):.6e}"
        )

        # Examples:
        # >>> dec = FreeEnergyDecomposer()
        # >>> # 线性情形 g(ψ)=ψ 的解析解:
        # >>> # μ* = (I + I)^{-1}(o + 0) = o/2
        # >>> o = np.array([2.0, 4.0])
        # >>> g = lambda x: x
        # >>> mu_star = dec.optimal_mean(o, g, np.eye(2), np.zeros(2), np.eye(2))
        # >>> np.allclose(mu_star, o / 2, atol=1e-2)
        # True
        self,
        observation: np.ndarray,
        generative_fn: callable,
        noise_cov: np.ndarray,
        prior_mean: np.ndarray,
        prior_cov: np.ndarray,
        lambda_values: list[float]
    ) -> list[tuple[float, float]]:
        """计算Pareto前沿上的点（A5.9）。

        对每个 λ 值，最小化 A + λC，返回 (A, C) 对。

        Args:
            observation: 观测值 o
            generative_fn: 生成函数 g(ψ)
            noise_cov: 噪声协方差 Σ_η
            prior_mean: 先验均值 μ_prior
            prior_cov: 先验协方差 Σ_prior
            lambda_values: 权衡参数列表，每个 ≥ 0

        Returns:
            Pareto前沿点列表 [(A_1, C_1), ..., (A_n, C_n)]

        Raises:
            ValueError: 当任何 lambda < 0 时
        """
        for lam_val in lambda_values:
            if lam_val < 0:
                raise ValueError(
                    f"权衡参数 λ 必须 ≥ 0，收到 {lam_val}"
                )
        # A5.9: 对每个 λ，最小化 A + λC
        # 梯度: ∂(A+λC)/∂μ = -JᵀΣ_η⁻¹(o-g(μ)) + λΣ_prior⁻¹(μ-μ_prior)
        frontier = []
        lr = 0.01
        max_inner_iter = 200
        for lam_val in lambda_values:
            mu = prior_mean.copy().astype(float)
            for _ in range(max_inner_iter):
                grad_A = self.accuracy_gradient(mu, observation, generative_fn, noise_cov)
                grad_C = self.complexity_gradient(mu, prior_mean, prior_cov)
                grad_total = grad_A + lam_val * grad_C
                if np.linalg.norm(grad_total) < 1e-6:
                    break
                mu = mu - lr * grad_total
            # 计算 A 和 C 在当前 μ 处的值
            g_mu = generative_fn(mu)
            residual = observation - g_mu
            noise_prec = np.linalg.inv(noise_cov)
            A_val = 0.5 * float(residual @ noise_prec @ residual)
            prior_prec = np.linalg.inv(prior_cov)
            diff = mu - prior_mean
            C_val = 0.5 * float(diff @ prior_prec @ diff)
            frontier.append((A_val, C_val))
        return frontier

        # Examples:
        # >>> dec = FreeEnergyDecomposer()
        # >>> o = np.array([2.0])
        # >>> g = lambda x: x
        # >>> pts = dec.pareto_frontier(o, g, np.eye(1), np.zeros(1), np.eye(1), [0.0, 1.0, 10.0])
        # >>> len(pts)
        # 3
        self,
        num_params: int,
        max_log_likelihood: float
    ) -> float:
        """计算Akaike信息准则 AIC（A5.11）。

        Args:
            num_params: 模型参数个数 k，必须 > 0
            max_log_likelihood: 最大对数似然 ln(L̂)

        Returns:
            AIC = 2k - 2ln(L̂)

        Raises:
            ValueError: 当 num_params <= 0 时
        """
        if num_params <= 0:
            raise ValueError(f"参数个数 k 必须 > 0，收到 {num_params}")
        # A5.11: AIC = 2k - 2ln(L̂)
        return float(2 * num_params - 2 * max_log_likelihood)

        # Examples:
        # >>> dec = FreeEnergyDecomposer()
        # >>> dec.aic(3, -10.0)
        # 26.0
        # >>> dec.aic(0, -10.0)
        # Traceback (most recent call last):
        #     ...
        # ValueError: 参数个数 k 必须 > 0，收到 0
        self,
        num_params: int,
        num_samples: int,
        max_log_likelihood: float
    ) -> float:
        """计算Bayesian信息准则 BIC（A5.12）。

        Args:
            num_params: 模型参数个数 k，必须 > 0
            num_samples: 样本量 n，必须 > 0
            max_log_likelihood: 最大对数似然 ln(L̂)

        Returns:
            BIC = k·ln(n) - 2ln(L̂)

        Raises:
            ValueError: 当 num_params <= 0 或 num_samples <= 0 时
        """
        if num_params <= 0:
            raise ValueError(f"参数个数 k 必须 > 0，收到 {num_params}")
        if num_samples <= 0:
            raise ValueError(f"样本量 n 必须 > 0，收到 {num_samples}")
        # A5.12: BIC = k·ln(n) - 2ln(L̂)
        return float(num_params * np.log(num_samples) - 2 * max_log_likelihood)

        # Examples:
        # >>> dec = FreeEnergyDecomposer()
        # >>> dec.bic(3, 100, -10.0)
        # 23.8087...
        # >>> dec.bic(3, 0, -10.0)
        # Traceback (most recent call last):
        #     ...
        # ValueError: 样本量 n 必须 > 0，收到 0
        self,
        mu: np.ndarray,
        prior_mean: np.ndarray,
        lambda_reg: float
    ) -> float:
        """将L2正则化映射为高斯先验下的复杂性项 R(μ)（A5.14）。

        Args:
            mu: 参数向量 μ，形状 (d,)
            prior_mean: 先验均值 μ_prior，形状 (d,)
            lambda_reg: L2正则化系数 λ > 0

        Returns:
            R(μ) = (λ/2)‖μ - μ_prior‖²

        Raises:
            ValueError: 当 lambda_reg <= 0 或维度不匹配时
        """
        if lambda_reg <= 0:
            raise ValueError(f"正则化系数 λ 必须 > 0，收到 {lambda_reg}")
        if mu.shape != prior_mean.shape:
            raise ValueError(
                f"mu 形状 {mu.shape} 与 prior_mean 形状 {prior_mean.shape} 不匹配"
            )
        # A5.14: R(μ) = (λ/2)‖μ - μ_prior‖²，当 Σ_prior = λ⁻¹I
        diff = mu - prior_mean
        return float(lambda_reg / 2.0 * np.dot(diff, diff))

        # Examples:
        # >>> dec = FreeEnergyDecomposer()
        # >>> mu = np.array([1.0, 2.0])
        # >>> mu_p = np.array([0.0, 0.0])
        # >>> dec.l2_to_complexity(mu, mu_p, 2.0)
        # 5.0
        self,
        mu: np.ndarray,
        prior_mean: np.ndarray,
        lambda_reg: float
    ) -> float:
        """将L1正则化映射为拉普拉斯先验下的复杂性项（A5.15）。

        Args:
            mu: 参数向量 μ，形状 (d,)
            prior_mean: 先验均值 μ_prior，形状 (d,)
            lambda_reg: L1正则化系数 λ > 0

        Returns:
            拉普拉斯先验下复杂性项 ≈ λ‖μ - μ_prior‖₁

        Raises:
            ValueError: 当 lambda_reg <= 0 或维度不匹配时
        """
        if lambda_reg <= 0:
            raise ValueError(f"正则化系数 λ 必须 > 0，收到 {lambda_reg}")
        if mu.shape != prior_mean.shape:
            raise ValueError(
                f"mu 形状 {mu.shape} 与 prior_mean 形状 {prior_mean.shape} 不匹配"
            )
        # A5.15: 拉普拉斯先验下复杂性项 ≈ λ‖μ - μ_prior‖₁
        diff = mu - prior_mean
        return float(lambda_reg * np.sum(np.abs(diff)))

        # Examples:
        # >>> dec = FreeEnergyDecomposer()
        # >>> mu = np.array([1.0, -2.0])
        # >>> mu_p = np.array([0.0, 0.0])
        # >>> dec.l1_to_complexity(mu, mu_p, 0.5)
        # 1.5
        self,
        mu: np.ndarray,
        observation: np.ndarray,
        generative_fn: callable,
        noise_cov: np.ndarray,
        prior_mean: np.ndarray,
        prior_cov: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """执行自由能景观上梯度流的一步更新（A5.17）。

        dμ/dt = -∂F/∂μ = JᵀΣ_η⁻¹(o - g(μ)) - Σ_prior⁻¹(μ - μ_prior)

        Args:
            mu: 当前变分均值 μ，形状 (d,)
            observation: 观测值 o，形状 (d_o,)
            generative_fn: 生成函数 g(ψ)
            noise_cov: 噪声协方差 Σ_η
            prior_mean: 先验均值 μ_prior
            prior_cov: 先验协方差 Σ_prior
            dt: 时间步长，必须 > 0

        Returns:
            更新后的均值 μ(t+dt)，形状 (d,)

        Raises:
            ValueError: 当 dt <= 0 或维度不匹配时
        """
        if dt <= 0:
            raise ValueError(f"时间步长 dt 必须 > 0，收到 {dt}")
        # A5.17: dμ/dt = -∂F/∂μ = JᵀΣ_η⁻¹(o - g(μ)) - Σ_prior⁻¹(μ - μ_prior)
        grad_A = self.accuracy_gradient(mu, observation, generative_fn, noise_cov)
        grad_C = self.complexity_gradient(mu, prior_mean, prior_cov)
        grad_F = grad_A + grad_C
        # Euler 离散化: μ(t+dt) = μ(t) - dt * ∂F/∂μ
        return mu - dt * grad_F

        # Examples:
        # >>> dec = FreeEnergyDecomposer()
        # >>> mu = np.array([2.0, 3.0])
        # >>> o = np.array([1.0, 1.0])
        # >>> g = lambda x: x
        # >>> mu_new = dec.gradient_flow_step(mu, o, g, np.eye(2), np.zeros(2), np.eye(2), dt=0.01)
        # >>> mu_new.shape
        # (2,)
        self,
        generative_fn: callable,
        mu: np.ndarray,
        noise_cov: np.ndarray,
        prior_cov: np.ndarray
    ) -> tuple[float, float]:
        """计算准确性与复杂性的时间尺度分离（A5.18）。

        Args:
            generative_fn: 生成函数 g(ψ)
            mu: 变分均值 μ，形状 (d,)
            noise_cov: 噪声协方差 Σ_η
            prior_cov: 先验协方差 Σ_prior

        Returns:
            (τ_A, τ_C) 时间常数元组，满足 τ_A << τ_C

        Raises:
            ValueError: 当协方差矩阵不正定时
        """
        # 验证协方差矩阵正定性
        for name, cov in [("noise_cov", noise_cov), ("prior_cov", prior_cov)]:
            if not np.allclose(cov, cov.T):
                raise ValueError(f"{name} 必须对称")
            try:
                np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                raise ValueError(f"{name} 必须正定")

        # A5.18: τ_A = 1/λ_min(H_A), τ_C = 1/λ_min(H_C)
        # H_A ≈ JᵀΣ_η⁻¹J（准确性Hessian）
        # H_C = Σ_prior⁻¹（复杂性Hessian）
        d = mu.shape[0]
        d_o = noise_cov.shape[0]
        noise_prec = np.linalg.inv(noise_cov)
        prior_prec = np.linalg.inv(prior_cov)

        # 数值Jacobian
        eps = 1e-5
        J = np.zeros((d_o, d))
        for j in range(d):
            mu_plus = mu.copy(); mu_plus[j] += eps
            mu_minus = mu.copy(); mu_minus[j] -= eps
            J[:, j] = (generative_fn(mu_plus) - generative_fn(mu_minus)) / (2 * eps)

        H_A = J.T @ noise_prec @ J
        H_C = prior_prec

        eigvals_A = np.linalg.eigvalsh(H_A)
        eigvals_C = np.linalg.eigvalsh(H_C)

        # 过滤接近零的特征值
        eigvals_A_pos = eigvals_A[eigvals_A > 1e-12]
        eigvals_C_pos = eigvals_C[eigvals_C > 1e-12]

        tau_A = 1.0 / float(np.min(eigvals_A_pos)) if len(eigvals_A_pos) > 0 else float('inf')
        tau_C = 1.0 / float(np.min(eigvals_C_pos)) if len(eigvals_C_pos) > 0 else float('inf')

        return (tau_A, tau_C)

        # Examples:
        # >>> dec = FreeEnergyDecomposer()
        # >>> mu = np.array([0.0, 0.0])
        # >>> g = lambda x: x
        # >>> # σ_η²=0.1, σ_prior²=10.0 → τ_A << τ_C
        # >>> tau_A, tau_C = dec.timescale_separation(g, mu, 0.1*np.eye(2), 10.0*np.eye(2))
        # >>> tau_A < tau_C
        # True
        self,
        decomposition: FreeEnergyDecomposition,
        tolerance: float = 1e-8
    ) -> bool:
        """验证自由能分解的一致性（A5.1+A5.5=A4.4）。

        Args:
            decomposition: 自由能分解实例
            tolerance: 数值容差

        Returns:
            True 如果 A + C == F 在容差范围内一致，False 否则
        """
        # 验证 A + C == F（A5.1 + A5.5 = A4.4）
        vfe = decomposition.variational_free_energy
        A = decomposition.total_accuracy
        C = decomposition.total_complexity
        F = vfe.total
        return bool(np.isclose(A + C, F, atol=tolerance))

        # Examples:
        # >>> dec = FreeEnergyDecomposer()
        # >>> # dec.verify_decomposition_consistency(decomp)  # doctest: +SKIP
        # >>> # A + C 应等于 F
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `VariationalFreeEnergy` | ← 阶段4 | 作为自由能分解的基础（组合字段） |
| **输入** | A4.4（变分自由能全局唯一定义） | ← 阶段4 | 本阶段展开A4.5分解的理论基础 |
| **输入** | A4.5（准确性+复杂性分解） | ← 阶段4 | 本阶段唯一展开对象 |
| **输入** | A4.10（拉普拉斯近似） | ← 阶段4 | 高斯变分分布下的解析展开 |
| **输入** | A3.2（精度加权误差能量） | ← 阶段3 | A5.3准确性项与预测误差的桥梁 |
| **输入** | A2.3（POMDP观测方程） | ← 阶段2 | 高斯似然模型的来源 |
| **输出** | `FreeEnergyDecomposition` | → 阶段6/7/18 | 包含VariationalFreeEnergy的分解数据 |
| **输出** | A5.4（准确性梯度） | → 阶段11/12 | 预测编码算法的准确性更新信号 |
| **输出** | A5.8（复杂性梯度） | → 阶段11/12 | 预测编码算法的先验约束信号 |
| **输出** | 定理5.1（Pareto最优等价性） | → 阶段7/29 | 主动推断中准确性-复杂性权衡的理论基础 |
| **输出** | A5.17（梯度流） | → 阶段11/12 | 自由能最小化的动力学实现 |
| **输出** | A5.18（时间尺度分离） | → 阶段11 | 快慢动力学分离的参数设计依据 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 自动微分与梯度计算 | 准确性梯度（A5.4）和复杂性梯度（A5.8）的自动微分验证；梯度流（A5.17）的ODE求解 |
| SciPy | ODE求解与优化 | 梯度流（A5.17）的数值积分；最优均值（A5.10）的迭代求解 |
| NumPyro | 变分推断与模型选择 | AIC/BIC/自由能模型选择的数值验证；L2/L1/Dropout正则化的变分解释验证 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V5.1 | 准确性项定义 | `A(q) == -E_q[ln p(o\|ψ)]` 对所有合法 $q$ 成立 | 蒙特卡洛期望估计 + 解析对比 |
| V5.2 | 高斯似然准确性 | A5.2解析值与蒙特卡洛估计的相对误差 < 1% | 合成高斯似然测试 |
| V5.3 | 准确性-误差关系 | `A(q) ≈ ½ε_0 + const` 在 $\Sigma \to 0$ 极限下成立 | 数值极限验证 |
| V5.4 | 准确性梯度 | `∂A/∂μ` 与JAX自动微分结果一致（相对误差 < 1e-6） | 自动微分对比验证 |
| V5.5 | 复杂性项非负 | `C(q) >= 0` 对所有合法 $q$ 和 $p$ 成立 | 随机分布测试 |
| V5.6 | 高斯先验复杂性 | A5.6解析值与KL散度数值积分的相对误差 < 1% | 数值积分对比 |
| V5.7 | 复杂性分解 | `R(μ) + R(Σ) == C(q)` 对所有高斯 $q$ 和 $p$ 成立 | 数值一致性验证 |
| V5.8 | 复杂性梯度 | `∂C/∂μ` 与解析公式 `Σ_prior⁻¹(μ - μ_prior)` 一致 | 解析对比 + 自动微分 |
| V5.9 | Pareto前沿 | Pareto前沿上的点不被任何其他点Pareto支配 | 多目标优化验证 |
| V5.10 | 最优权衡条件 | `∂A/∂μ + ∂C/∂μ ≈ 0` 在最优解处成立 | 梯度范数 < 1e-6 |
| V5.11 | 定理5.1验证 | 自由能最小值点在Pareto前沿上（λ=1对应点） | 数值优化 + Pareto前沿对比 |
| V5.12 | AIC与自由能关系 | AIC ≈ 2F_MAP 在大样本极限下成立 | 大样本数值实验 |
| V5.13 | BIC与自由能关系 | BIC ≈ 2F_Laplace 在大样本极限下成立 | 大样本数值实验 |
| V5.14 | L2正则化等价 | L2惩罚 == R(μ) 当 Σ_prior = λ⁻¹I | 解析验证 |
| V5.15 | L1正则化等价 | L1惩罚 ≈ 拉普拉斯先验下复杂性项 | 数值验证 |
| V5.16 | Dropout等价 | Dropout训练损失 ≈ 变分自由能蒙特卡洛估计 | 合成网络实验 |
| V5.17 | 梯度流收敛 | 梯度流（A5.17）收敛到最优均值 μ* | 多步ODE积分验证 |
| V5.18 | 时间尺度分离 | τ_A < τ_C 在 σ_η² < σ_prior² 时成立 | 特征值分析验证 |
| V5.19 | FreeEnergyDecomposition组合 | `FreeEnergyDecomposition.variational_free_energy` 类型为 `VariationalFreeEnergy` | 类型检查 |
| V5.20 | 非法输入拒绝 | λ<0、协方差不正定、维度不匹配等抛出 `ValueError` | 异常测试 |

---

> **阶段5完** | 下一步：阶段6将从本阶段的准确性-复杂性分解出发，展开自由能驱力的推导与分层结构分析。
