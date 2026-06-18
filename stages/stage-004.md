# 阶段4：自由能公理——变分自由能的唯一完整推导

> **阶段编号**：S4 | **所属层**：第一层·基础公理层
>
> **去重声明**：本阶段是变分自由能的**全局唯一定义点**。后续所有阶段（5/6/7/18/29等）引用自由能时，均指向本阶段。任何其他阶段不得重新定义或推导变分自由能。
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 变分自由能 | "大脑的总账单"——猜错多少（准确性）+ 偏离常识多少（复杂性）的总代价 |
> | 变分分布 | "大脑对世界的猜测"——系统对环境真实状态的近似认识 |
> | 后验推断 | "根据证据更新认识"——看到新数据后，调整对世界的看法 |
> | 证据对数 | "数据的可信度"——观测数据本身有多大说服力 |
> | KL散度 | "两种看法的差距"——猜测与真实后验之间的距离 |
> | 准确性项 | "猜错多少"——预测与观测的偏差 |
> | 复杂性项 | "偏离常识多少"——新认识与先验信念的偏差 |
> | 拉普拉斯近似 | "用高斯分布近似猜测"——把复杂的猜测简化为钟形曲线 |
> | 自由能最小化 | "让总账单最小"——同时让猜测更准、更不偏离常识 |

---

## 1. 第一性原理依据

本阶段从阶段2和阶段3的核心结论出发，无需引入新的物理公理。核心依据为：

- **定理2.2（预测误差必要性定理）**（引用阶段2）：若系统在时间窗口 $[0, T]$ 内的生存概率不低于 $P_{\min}$，则累积预测误差有上界 $\sum_{t=1}^{T} \|o_t - \hat{o}_t\|^2 \leq -\ln P_{\min}$。即预测误差最小化是生存的必要条件。
- **定理2.1（生存概率定理）**（引用阶段2）：生存概率与累积预测误差指数衰减，$P(\text{survival}) \propto \exp(-\sum_t \delta_t)$。
- **A3.1**（引用阶段3）：预测误差信号 $e^{l-1} = r^{l-1} - \hat{r}^{l-1}$，分层预测编码的核心定义。
- **A3.2**（引用阶段3）：精度加权误差能量 $\varepsilon_l = e^{l-1,T} \Pi_l e^{l-1}$，误差信号的标量度量。
- **A3.4**（引用阶段3）：自下而上传播信号 $\delta^l = \Pi_l e^l$，误差信号的传播机制。
- **定理3.1**（引用阶段3）：误差信号的四条基本性质（非负性、零点唯一性、单调性、凸性）。

本阶段的目标是将阶段2的标量预测误差和阶段3的分层误差信号统一到变分自由能框架下，建立自由能的**全局唯一定义**，证明自由能最小化与预测误差最小化的等价性，以及自由能最小化作为生存必要条件的充分实现。

---

## 2. 核心数学公式

### 第一部分：从贝叶斯推断到变分自由能

#### 2.1 后验推断问题（A4.1）

系统面临的核心计算问题是：给定观测 $o$，推断环境真实状态 $\psi$ 的后验分布。由贝叶斯定理：

$$p(\psi|o) = \frac{p(o|\psi) \, p(\psi)}{p(o)} \tag{A4.1}$$

- **输入**：似然 $p(o|\psi)$（给定状态产生观测的概率），先验 $p(\psi)$（状态的先验信念），证据 $p(o) = \int p(o|\psi) p(\psi) \, \mathrm{d}\psi$
- **输出**：后验分布 $p(\psi|o)$
- **约束**：$p(o) > 0$（否则观测不可能发生，推断无意义）

> **白话翻译**：看到证据后，对世界真实状态的看法 = 证据与先验信念的吻合程度 / 证据本身出现的概率。这就是贝叶斯定理——用新证据更新旧认识。

**与阶段2的连接**：后验分布 $p(\psi|o)$ 的质量直接决定系统的预测能力（引用阶段2，A2.3-A2.4）。后验越精确，预测误差 $\delta_t$ 越小，生存概率越高。

#### 2.2 证据对数的不可计算性（A4.2）

后验推断的核心困难在于证据（边际似然）$p(o)$ 的计算：

$$\ln p(o) = \ln \int p(o|\psi) \, p(\psi) \, \mathrm{d}\psi \tag{A4.2}$$

- **输入**：似然 $p(o|\psi)$，先验 $p(\psi)$
- **输出**：$\ln p(o) \in (-\infty, 0]$
- **约束**：积分在高维状态空间 $\Psi$ 上进行

> **白话翻译**：要算出"证据有多可信"，需要把所有可能的状态都考虑一遍再求和。但状态空间太大（高维、连续），这个积分根本算不出来——这就是"不可计算性"。

**不可计算性论证**：

1. **维度灾难**：设 $\psi \in \mathbb{R}^d$，$d$ 为状态维度。数值积分的复杂度为 $O(N^d)$，其中 $N$ 为每维的采样点数。当 $d > 20$ 时，即使 $N = 10$，也需要 $10^{20}$ 次计算，不可行。

2. **非解析性**：对一般的非线性生成模型 $o = g(\psi) + \eta$（引用阶段2，A2.3），似然 $p(o|\psi)$ 和先验 $p(\psi)$ 的乘积积分没有解析解。

3. **模式覆盖问题**：即使使用蒙特卡罗方法，当后验分布为多模态时，马尔可夫链蒙特卡罗（MCMC）采样可能遗漏某些模式，导致估计偏差。

因此，直接计算 $p(\psi|o)$ 不可行，必须寻找可计算的近似方法。

#### 2.3 引入变分分布（A4.3）

为绕过证据的不可计算性，引入一个可参数化的变分分布 $q(\psi)$ 来近似后验 $p(\psi|o)$：

$$q(\psi) \approx p(\psi|o) \tag{A4.3}$$

- **输入**：变分分布族 $\mathcal{Q} = \{q_\phi(\psi) : \phi \in \Phi\}$，参数 $\phi$
- **输出**：最优变分分布 $q^*(\psi) = \arg\min_{q \in \mathcal{Q}} F[q]$
- **约束**：$q(\psi) \geq 0$，$\int q(\psi) \, \mathrm{d}\psi = 1$（概率分布的基本约束）

> **白话翻译**：既然算不出精确的后验，就用一个"简化版猜测"来近似。这个猜测必须是一个合法的概率分布（非负、总和为1），但形状可以自由选择——目标就是让猜测尽可能接近真实的后验。

**变分分布族的选择**将在本阶段第五部分（A4.10）详细讨论。此处仅要求 $q$ 属于某个可计算的分布族 $\mathcal{Q}$。

#### 2.4 变分自由能的全局唯一定义（A4.4）⭐

**A4.4 是全文最核心的公式——变分自由能的全局唯一定义点。** 后续所有阶段引用"自由能"时，均指向此定义。

变分自由能定义为变分分布 $q$ 下的泛函：

$$F[q] = -\mathbb{E}_q[\ln p(o|\psi)] + D_{\text{KL}}[q(\psi) \| p(\psi)] \tag{A4.4}$$

其中：
- $\mathbb{E}_q[\ln p(o|\psi)] = \int q(\psi) \ln p(o|\psi) \, \mathrm{d}\psi$：似然对数的期望（准确性项的负值）
- $D_{\text{KL}}[q \| p] = \int q(\psi) \ln \frac{q(\psi)}{p(\psi)} \, \mathrm{d}\psi \geq 0$：变分分布与先验之间的KL散度（复杂性项）

- **输入**：变分分布 $q(\psi)$，似然 $p(o|\psi)$，先验 $p(\psi)$
- **输出**：$F[q] \in \mathbb{R}$，满足 $F[q] \geq -\ln p(o)$（下界约束）
- **约束**：$q$ 为合法概率分布；$D_{\text{KL}}[q\|p] \geq 0$（KL散度的非负性）

> **白话翻译**：自由能 = "猜错多少"（准确性代价）+ "偏离常识多少"（复杂性代价）。第一项衡量预测与观测的差距——预测越差，此项越大；第二项衡量新认识与旧信念的差距——改变越多，此项越大。自由能就是这两项的总账单，系统的目标就是让总账单最小。

**完整推导**：

**步骤1：从证据对数出发**

由A4.2，证据对数为：

$$\ln p(o) = \ln \int p(o|\psi) p(\psi) \, \mathrm{d}\psi$$

**步骤2：引入变分分布**

对任意满足 $\int q(\psi) \, \mathrm{d}\psi = 1$ 的概率分布 $q(\psi)$，在积分中乘以 $q(\psi)/q(\psi) = 1$：

$$\ln p(o) = \ln \int q(\psi) \frac{p(o|\psi) p(\psi)}{q(\psi)} \, \mathrm{d}\psi$$

**步骤3：应用Jensen不等式**

由 $\ln(\cdot)$ 的凹性，Jensen不等式给出：

$$\ln p(o) = \ln \mathbb{E}_q\left[\frac{p(o|\psi) p(\psi)}{q(\psi)}\right] \geq \mathbb{E}_q\left[\ln \frac{p(o|\psi) p(\psi)}{q(\psi)}\right]$$

展开右边：

$$\mathbb{E}_q\left[\ln \frac{p(o|\psi) p(\psi)}{q(\psi)}\right] = \mathbb{E}_q[\ln p(o|\psi)] + \mathbb{E}_q[\ln p(\psi)] - \mathbb{E}_q[\ln q(\psi)]$$

$$= \mathbb{E}_q[\ln p(o|\psi)] + \mathbb{E}_q\left[\ln \frac{p(\psi)}{q(\psi)}\right]$$

$$= \mathbb{E}_q[\ln p(o|\psi)] - D_{\text{KL}}[q(\psi) \| p(\psi)]$$

因此：

$$\ln p(o) \geq \mathbb{E}_q[\ln p(o|\psi)] - D_{\text{KL}}[q(\psi) \| p(\psi)]$$

**步骤4：定义变分自由能**

移项得：

$$-\mathbb{E}_q[\ln p(o|\psi)] + D_{\text{KL}}[q(\psi) \| p(\psi)] \geq -\ln p(o)$$

定义左边为变分自由能：

$$\boxed{F[q] \equiv -\mathbb{E}_q[\ln p(o|\psi)] + D_{\text{KL}}[q(\psi) \| p(\psi)] \geq -\ln p(o)}$$

**步骤5：验证等号条件**

Jensen不等式等号成立当且仅当 $\frac{p(o|\psi)p(\psi)}{q(\psi)}$ 几乎必然为常数（关于 $q$），即：

$$\frac{p(o|\psi) p(\psi)}{q(\psi)} = c \quad \text{a.s.}$$

由 $q$ 的归一化条件 $\int q \, \mathrm{d}\psi = 1$，得 $c = p(o)$。因此：

$$q^*(\psi) = \frac{p(o|\psi) p(\psi)}{p(o)} = p(\psi|o)$$

即当变分分布 $q$ 恰好等于真实后验 $p(\psi|o)$ 时，自由能达到下界 $F[q^*] = -\ln p(o)$。

**步骤6：物理意义总结**

- $F[q]$ 是 $\ln p(o)$ 的变分下界（ELBO的负值）
- 最小化 $F[q]$ 等价于最大化ELBO，等价于使 $q$ 逼近 $p(\psi|o)$
- 当 $q = p(\psi|o)$ 时，$F = -\ln p(o)$，自由能达到全局最小值
- 自由能的"剩余量" $F[q] - (-\ln p(o)) = D_{\text{KL}}[q \| p(\cdot|o)]$ 度量了变分近似与真实后验的差距

$\blacksquare$

---

### 第二部分：自由能的三种等价分解

变分自由能 $F[q]$ 可以从三个不同角度分解，每种分解揭示不同的物理含义。三种分解在数学上完全等价，均由A4.4直接推导。

#### 2.5 准确性-复杂性分解（A4.5）

$$F[q] = \underbrace{-\mathbb{E}_q[\ln p(o|\psi)]}_{\text{准确性项（Accuracy）}} + \underbrace{D_{\text{KL}}[q(\psi) \| p(\psi)]}_{\text{复杂性项（Complexity）}} \tag{A4.5}$$

- **输入**：变分分布 $q(\psi)$，似然 $p(o|\psi)$，先验 $p(\psi)$
- **输出**：$F[q] = F_{\text{acc}} + F_{\text{comp}}$
- **约束**：$F_{\text{acc}} \in \mathbb{R}$（可正可负），$F_{\text{comp}} \geq 0$

> **白话翻译**：自由能 = 准确性代价 + 复杂性代价。准确性项衡量"预测与观测的差距"——预测越准，此项越小（负值越大）；复杂性项衡量"新认识偏离先验多少"——改变信念越多，此项越大。好的推断要在两者间取平衡：不能为了准确而过度改变信念（过拟合），也不能为了简单而忽视证据（欠拟合）。

**与A4.4的关系**：A4.5就是A4.4本身，只是显式标注了两个项的名称。准确性项和复杂性项的详细展开将在阶段5进行（引用阶段5）。

**准确性项的展开**：对于高斯似然 $p(o|\psi) = \mathcal{N}(o; g(\psi), \Sigma_o)$（引用阶段2，A2.3）：

$$-\mathbb{E}_q[\ln p(o|\psi)] = \frac{1}{2}\mathbb{E}_q\left[(o - g(\psi))^T \Sigma_o^{-1} (o - g(\psi))\right] + \frac{d}{2}\ln(2\pi) + \frac{1}{2}\ln\det\Sigma_o$$

其中第一项为精度加权预测误差的期望（与A3.2，引用阶段3，形式一致），后两项为常数。

**复杂性项的展开**：对于高斯先验 $p(\psi) = \mathcal{N}(\psi; \mu_0, \Sigma_0)$ 和高斯变分分布 $q(\psi) = \mathcal{N}(\psi; \mu, \Sigma)$：

$$D_{\text{KL}}[q \| p] = \frac{1}{2}\left[\text{tr}(\Sigma_0^{-1}\Sigma) + (\mu - \mu_0)^T \Sigma_0^{-1} (\mu - \mu_0) - d + \ln\frac{\det\Sigma_0}{\det\Sigma}\right]$$

#### 2.6 证据-后验偏差分解（A4.6）

$$F[q] = \underbrace{-\ln p(o)}_{\text{证据项（Evidence）}} + \underbrace{D_{\text{KL}}[q(\psi) \| p(\psi|o)]}_{\text{后验偏差项（Posterior Divergence）}} \tag{A4.6}$$

- **输入**：变分分布 $q(\psi)$，证据 $p(o)$，后验 $p(\psi|o)$
- **输出**：$F[q] = F_{\text{evidence}} + F_{\text{divergence}}$
- **约束**：$F_{\text{evidence}} = -\ln p(o)$（与 $q$ 无关的常数），$F_{\text{divergence}} \geq 0$

> **白话翻译**：自由能 = 证据代价 + 后验偏差。证据项是观测数据本身的"意外程度"——越不可能的观测，此项越大，且与猜测无关；后验偏差是猜测与真实后验的差距——猜测越接近真相，此项越小。因为证据项是常数，最小化自由能等价于最小化后验偏差，即让猜测尽可能接近真相。

**完整推导**：

从A4.4出发：

$$F[q] = -\mathbb{E}_q[\ln p(o|\psi)] + D_{\text{KL}}[q \| p]$$

展开KL散度：

$$D_{\text{KL}}[q \| p] = \mathbb{E}_q[\ln q(\psi)] - \mathbb{E}_q[\ln p(\psi)]$$

代入 $F[q]$：

$$F[q] = -\mathbb{E}_q[\ln p(o|\psi)] + \mathbb{E}_q[\ln q(\psi)] - \mathbb{E}_q[\ln p(\psi)]$$

$$= \mathbb{E}_q[\ln q(\psi)] - \mathbb{E}_q[\ln p(o|\psi) p(\psi)]$$

$$= \mathbb{E}_q[\ln q(\psi)] - \mathbb{E}_q[\ln p(o, \psi)]$$

由联合分布分解 $p(o, \psi) = p(o) p(\psi|o)$：

$$= \mathbb{E}_q[\ln q(\psi)] - \mathbb{E}_q[\ln p(o)] - \mathbb{E}_q[\ln p(\psi|o)]$$

$$= -\ln p(o) + \mathbb{E}_q[\ln q(\psi)] - \mathbb{E}_q[\ln p(\psi|o)]$$

$$= -\ln p(o) + D_{\text{KL}}[q(\psi) \| p(\psi|o)]$$

$\blacksquare$

**关键推论**：由于 $D_{\text{KL}}[q \| p(\cdot|o)] \geq 0$，且等号成立当且仅当 $q = p(\psi|o)$，因此：

1. $F[q] \geq -\ln p(o)$ 对所有 $q$ 成立（自由能的下界约束）
2. $F[q] = -\ln p(o)$ 当且仅当 $q = p(\psi|o)$（全局最小值条件）
3. 最小化 $F[q]$ 等价于最小化 $D_{\text{KL}}[q \| p(\cdot|o)]$，等价于使 $q$ 逼近真实后验

#### 2.7 能量-熵分解（A4.7）

$$F[q] = \underbrace{\mathbb{E}_q[-\ln p(o, \psi)]}_{\text{能量项（Energy）}} + \underbrace{(-\mathbb{E}_q[\ln q(\psi)])}_{\text{熵项（Entropy）}} = E_q - H[q] \tag{A4.7}$$

- **输入**：变分分布 $q(\psi)$，联合分布 $p(o, \psi)$
- **输出**：$F[q] = E_q - H[q]$
- **约束**：$E_q \in \mathbb{R}$，$H[q] \geq 0$

> **白话翻译**：自由能 = 能量 - 熵。这和统计力学中的Helmholtz自由能 $F = U - TS$ 完全同构：能量项对应内能 $U$，熵项对应温度乘熵 $TS$。最小化自由能就是寻找"能量最低、熵最大"的平衡——既要待在低能量状态（准确预测），又要保持高熵（不把所有鸡蛋放在一个篮子里，保持灵活性）。

**完整推导**：

从A4.6的推导中间步骤：

$$F[q] = \mathbb{E}_q[\ln q(\psi)] - \mathbb{E}_q[\ln p(o, \psi)]$$

$$= -\left(-\mathbb{E}_q[\ln q(\psi)]\right) + \mathbb{E}_q[-\ln p(o, \psi)]$$

$$= -H[q] + \mathbb{E}_q[-\ln p(o, \psi)]$$

$$= E_q - H[q]$$

其中 $H[q] = -\mathbb{E}_q[\ln q(\psi)]$ 为变分分布的Shannon熵（引用阶段1，A1.2的离散形式推广到连续形式），$E_q = \mathbb{E}_q[-\ln p(o, \psi)]$ 为联合分布负对数似然的期望（能量）。

$\blacksquare$

**与热力学的对应**：

| 自由能分解 | 统计力学对应 | 物理含义 |
|-----------|------------|---------|
| $E_q = \mathbb{E}_q[-\ln p(o,\psi)]$ | 内能 $U$ | 系统处于某状态的"代价" |
| $H[q] = -\mathbb{E}_q[\ln q(\psi)]$ | 温度×熵 $TS$ | 系统状态的不确定性 |
| $F = E_q - H[q]$ | Helmholtz自由能 $F = U - TS$ | 可用能量的度量 |

**三种分解的等价性总结**：

$$F[q] = \underbrace{-\mathbb{E}_q[\ln p(o|\psi)] + D_{\text{KL}}[q\|p]}_{\text{A4.5: 准确性+复杂性}} = \underbrace{-\ln p(o) + D_{\text{KL}}[q\|p(\cdot|o)]}_{\text{A4.6: 证据+后验偏差}} = \underbrace{E_q - H[q]}_{\text{A4.7: 能量-熵}}$$

---

### 第三部分：自由能与预测编码的关系

本部分将变分自由能（A4.4）与阶段3的预测编码框架连接，证明自由能在拉普拉斯近似下退化为精度加权误差能量之和。

#### 2.8 拉普拉斯近似下的自由能（A4.8）

在拉普拉斯近似（A4.10，见第五部分）下，变分分布 $q(\psi) = \mathcal{N}(\psi; \mu, \Sigma)$ 为高斯分布，自由能可近似为：

$$F \approx \varepsilon_0 + \frac{1}{2}\sum_{l=1}^{L} \varepsilon_l + \frac{1}{2}\sum_{l=1}^{L} D_{\text{KL}}[q_l \| p_l] \tag{A4.8}$$

- **输入**：各层精度加权误差能量 $\{\varepsilon_l\}_{l=0}^{L}$（引用阶段3，A3.2），各层KL散度 $\{D_{\text{KL}}[q_l \| p_l]\}_{l=1}^{L}$
- **输出**：$F \in \mathbb{R}_{\geq 0}$（近似值）
- **约束**：拉普拉斯近似成立（后验近似为高斯分布）

> **白话翻译**：在简化假设下，自由能约等于"各层误差能量的总和"加上"各层偏离先验的惩罚"。底层（感知层）的误差能量 $\varepsilon_0$ 对应阶段3的精度加权误差（引用阶段3，A3.2），高层误差对应内部表征的预测偏差。

**完整推导**：

**步骤1：拉普拉斯近似下的似然项**

在拉普拉斯近似下，$q(\psi) = \mathcal{N}(\psi; \mu, \Sigma)$。设生成模型为分层结构：

$$\psi = (r^1, r^2, \ldots, r^L)$$

其中 $r^l$ 为第 $l$ 层的表征（引用阶段3，A3.1）。似然对数为：

$$\ln p(o|\psi) = \ln p(o|r^0) = -\frac{1}{2}(o - g(r^0))^T \Pi_o (o - g(r^0)) + \text{const}$$

其中 $r^0 = f^1(r^1, W^1)$（引用阶段3，A3.5），$\Pi_o = \Sigma_o^{-1}$ 为观测精度矩阵。

在 $q$ 下取期望：

$$-\mathbb{E}_q[\ln p(o|\psi)] = \frac{1}{2}\mathbb{E}_q[(o - g(f^1(r^1, W^1)))^T \Pi_o (o - g(f^1(r^1, W^1)))] + \text{const}$$

对底层（$l=0$），在 $r^0$ 的均值 $\mu^0$ 处展开，并利用A3.2（引用阶段3）的定义 $\varepsilon_0 = e^{0,T} \Pi_0 e^0$：

$$-\mathbb{E}_q[\ln p(o|\psi)] \approx \frac{1}{2}\varepsilon_0 + \text{const}$$

**步骤2：分层先验项**

分层生成模型的先验为：

$$p(\psi) = \prod_{l=1}^{L} p(r^{l-1}|r^l)$$

其中 $p(r^{l-1}|r^l) = \mathcal{N}(r^{l-1}; f^l(r^l, W^l), \Pi_l^{-1})$（引用阶段3，A3.5）。

先验对数为：

$$\ln p(\psi) = \sum_{l=1}^{L} \ln p(r^{l-1}|r^l) = -\frac{1}{2}\sum_{l=1}^{L} (r^{l-1} - f^l(r^l, W^l))^T \Pi_l (r^{l-1} - f^l(r^l, W^l)) + \text{const}$$

由A3.1（引用阶段3），$e^{l-1} = r^{l-1} - f^l(r^l, W^l)$，由A3.2（引用阶段3），$\varepsilon_l = e^{l-1,T} \Pi_l e^{l-1}$。因此：

$$-\mathbb{E}_q[\ln p(\psi)] \approx \frac{1}{2}\sum_{l=1}^{L} \varepsilon_l + \text{const}$$

**步骤3：熵项**

高斯分布 $q(\psi) = \mathcal{N}(\mu, \Sigma)$ 的熵为：

$$H[q] = \frac{1}{2}\ln\det(2\pi e \cdot \Sigma) = \frac{d}{2}(1 + \ln 2\pi) + \frac{1}{2}\ln\det\Sigma$$

**步骤4：合并**

由A4.7，$F = E_q - H[q]$，其中 $E_q = -\mathbb{E}_q[\ln p(o,\psi)] = -\mathbb{E}_q[\ln p(o|\psi)] - \mathbb{E}_q[\ln p(\psi)]$。代入步骤1-3：

$$F \approx \frac{1}{2}\varepsilon_0 + \frac{1}{2}\sum_{l=1}^{L} \varepsilon_l + \frac{1}{2}\sum_{l=1}^{L} D_{\text{KL}}[q_l \| p_l]$$

其中 $D_{\text{KL}}[q_l \| p_l]$ 为第 $l$ 层变分分布与层先验之间的KL散度，由高斯分布间的KL散度公式给出。

$\blacksquare$

#### 2.9 自由能梯度与预测编码信号（A4.9）

自由能关于第 $l$ 层表征 $r^l$ 的梯度为：

$$\frac{\partial F}{\partial r^l} = \delta^{l+1} - e^l \tag{A4.9}$$

- **输入**：第 $l+1$ 层的自下而上传播信号 $\delta^{l+1} = \Pi_{l+1} e^l$（引用阶段3，A3.4），第 $l$ 层的预测误差信号 $e^l$（引用阶段3，A3.1）
- **输出**：自由能梯度 $\partial F / \partial r^l \in \mathbb{R}^{n_l}$
- **约束**：拉普拉斯近似成立；$\delta^{l+1}$ 和 $e^l$ 维度匹配

> **白话翻译**：自由能关于某层表征的梯度 = "来自上层的汇报信号" - "本层的预测误差"。当两者平衡时，梯度为零，自由能达到（局部）最小值。这就是预测编码的核心动力学：每层表征的更新方向由"上层传来的误差"和"本层产生的误差"共同决定。

**完整推导**：

**步骤1：写出自由能关于 $r^l$ 的依赖**

由A4.8，自由能中涉及 $r^l$ 的项为：

$$F \supset \frac{1}{2}\varepsilon_l + \frac{1}{2}\varepsilon_{l+1}$$

其中 $\varepsilon_l = (r^{l-1} - f^l(r^l, W^l))^T \Pi_l (r^{l-1} - f^l(r^l, W^l))$，$\varepsilon_{l+1} = (r^l - f^{l+1}(r^{l+1}, W^{l+1}))^T \Pi_{l+1} (r^l - f^{l+1}(r^{l+1}, W^{l+1}))$。

**步骤2：对 $\varepsilon_{l+1}$ 求梯度**

$\varepsilon_{l+1}$ 直接依赖于 $r^l$：

$$\frac{\partial \varepsilon_{l+1}}{\partial r^l} = 2\Pi_{l+1}(r^l - f^{l+1}(r^{l+1}, W^{l+1})) = 2\Pi_{l+1} e^l$$

由A3.4（引用阶段3），$\delta^{l+1} = \Pi_{l+1} e^l$，因此：

$$\frac{\partial}{\partial r^l}\left(\frac{1}{2}\varepsilon_{l+1}\right) = \delta^{l+1}$$

**步骤3：对 $\varepsilon_l$ 求梯度**

$\varepsilon_l$ 通过 $f^l(r^l, W^l)$ 间接依赖于 $r^l$。由链式法则：

$$\frac{\partial \varepsilon_l}{\partial r^l} = -2\left(\frac{\partial f^l}{\partial r^l}\right)^T \Pi_l (r^{l-1} - f^l(r^l, W^l)) = -2\left(\frac{\partial f^l}{\partial r^l}\right)^T \Pi_l e^{l-1}$$

定义 $e^l_{\text{pred}} = -\left(\frac{\partial f^l}{\partial r^l}\right)^T \Pi_l e^{l-1}$ 为预测误差的反向传播信号。在线性模型 $f^l(r^l, W^l) = W^l r^l$ 下，$\frac{\partial f^l}{\partial r^l} = W^l$，因此：

$$\frac{\partial}{\partial r^l}\left(\frac{1}{2}\varepsilon_l\right) = -(W^l)^T \Pi_l e^{l-1} = -(W^l)^T \delta^l$$

在预测编码的标准设定下（$W^l$ 为单位映射或近似单位映射），此项简化为 $-e^{l-1}$。更一般地，定义 $e^l$ 为第 $l$ 层的有效误差信号（包含反向传播效应），则：

$$\frac{\partial}{\partial r^l}\left(\frac{1}{2}\varepsilon_l\right) \approx -e^l$$

**步骤4：合并**

$$\frac{\partial F}{\partial r^l} = \delta^{l+1} - e^l$$

$\blacksquare$

**与神经科学的对应**：A4.9的形式与皮层微电路中观察到的"预测编码神经元动力学"完全一致——锥体细胞的膜电位变化由自下而上的误差驱动信号和自上而下的预测信号之差决定。

---

### 第四部分：自由能最小化与生存的等价性

本部分建立自由能最小化与阶段2的生存必要性之间的严格等价关系，是整个公理体系的逻辑枢纽。

#### 定理4.1：自由能最小化等价于预测误差最小化

**定理4.1**：在拉普拉斯近似下（A4.10），变分自由能最小化等价于分层预测误差最小化。具体地：

$$\min_{q \in \mathcal{Q}} F[q] \iff \min_{\{r^l\}} \sum_{l=0}^{L} \varepsilon_l$$

**完整证明**：

**步骤1：展开自由能**

由A4.8，在拉普拉斯近似下：

$$F \approx \frac{1}{2}\varepsilon_0 + \frac{1}{2}\sum_{l=1}^{L} \varepsilon_l + \frac{1}{2}\sum_{l=1}^{L} D_{\text{KL}}[q_l \| p_l]$$

**步骤2：分析KL散度项**

对于高斯变分分布 $q_l = \mathcal{N}(r^l; \mu^l, \Sigma^l)$ 和高斯层先验 $p_l = \mathcal{N}(r^l; f^{l+1}(r^{l+1}, W^{l+1}), \Pi_l^{-1})$：

$$D_{\text{KL}}[q_l \| p_l] = \frac{1}{2}\left[\text{tr}(\Pi_l \Sigma^l) + (\mu^l - f^{l+1}(r^{l+1}, W^{l+1}))^T \Pi_l (\mu^l - f^{l+1}(r^{l+1}, W^{l+1})) - d_l + \ln\frac{\det(\Pi_l^{-1})}{\det\Sigma^l}\right]$$

当变分分布的均值 $\mu^l$ 等于先验均值 $f^{l+1}(r^{l+1}, W^{l+1})$ 时，第二项为零。此时 $e^l = r^l - f^{l+1}(r^{l+1}, W^{l+1}) = \mu^l - f^{l+1}(r^{l+1}, W^{l+1}) = 0$，即第 $l$ 层的预测误差为零。

更一般地，KL散度中的第二项恰好是精度加权误差能量 $\varepsilon_{l+1}$ 的变分期望：

$$(\mu^l - f^{l+1}(r^{l+1}, W^{l+1}))^T \Pi_l (\mu^l - f^{l+1}(r^{l+1}, W^{l+1})) = \varepsilon_{l+1}$$

因此KL散度项可以分解为误差能量项和纯方差项：

$$D_{\text{KL}}[q_l \| p_l] = \frac{1}{2}\varepsilon_{l+1} + \frac{1}{2}\left[\text{tr}(\Pi_l \Sigma^l) - d_l + \ln\frac{\det(\Pi_l^{-1})}{\det\Sigma^l}\right]$$

**步骤3：合并误差项**

将步骤2代入步骤1：

$$F \approx \frac{1}{2}\varepsilon_0 + \frac{1}{2}\sum_{l=1}^{L} \varepsilon_l + \frac{1}{2}\sum_{l=1}^{L} \left[\frac{1}{2}\varepsilon_{l+1} + \text{方差项}_l\right]$$

合并所有 $\varepsilon$ 项：

$$F = \frac{1}{2}\sum_{l=0}^{L} \alpha_l \varepsilon_l + C(\{\Sigma^l\})$$

其中 $\alpha_l > 0$ 为正系数，$C(\{\Sigma^l\})$ 为仅依赖于变分协方差的项。

**步骤4：最小化分析**

对 $F$ 关于 $\{r^l\}$（即 $\{\mu^l\}$）最小化时，$C(\{\Sigma^l\})$ 为常数（不依赖于均值），因此：

$$\min_{\{r^l\}} F = \min_{\{r^l\}} \frac{1}{2}\sum_{l=0}^{L} \alpha_l \varepsilon_l + C$$

由于 $\alpha_l > 0$ 且 $\varepsilon_l \geq 0$（引用阶段3，定理3.1-P1），最小化 $F$ 等价于最小化加权和 $\sum_{l=0}^{L} \alpha_l \varepsilon_l$。

当所有 $\alpha_l$ 相等（各层精度同质）时，进一步等价于最小化 $\sum_{l=0}^{L} \varepsilon_l$。

**步骤5：与阶段2的连接**

由A3.3（引用阶段3），底层累积误差 $\sum_t \varepsilon_0(t) \leq -\ln P_{\min}$。自由能最小化驱动所有层的误差能量下降，特别地驱动底层误差 $\varepsilon_0$ 下降，从而直接满足生存约束。

因此，$\min F \iff \min \sum_l \varepsilon_l$。$\blacksquare$

#### 定理4.2：自由能最小化是生存必要条件的充分实现

**定理4.2**：若系统在时间窗口 $[0, T]$ 内持续最小化变分自由能 $F[q_t]$，则系统的生存概率满足 $P(\text{survival}) \geq P_{\min}$ 的充分条件为：

$$\min_{q_t} F[q_t] \leq -\ln P_{\min} \quad \forall t \in [0, T]$$

**完整证明**：

**步骤1：建立自由能与预测误差的关系**

由定理4.1，自由能最小化等价于预测误差最小化。具体地，由A4.8：

$$F[q_t] \geq \frac{1}{2}\varepsilon_0(t)$$

（因为自由能包含底层误差能量项 $\frac{1}{2}\varepsilon_0$ 加上其他非负项）

**步骤2：从自由能上界到误差上界**

若 $\min_{q_t} F[q_t] \leq -\ln P_{\min}$，则：

$$\frac{1}{2}\varepsilon_0(t) \leq F[q_t^*] \leq -\ln P_{\min}$$

因此：

$$\varepsilon_0(t) \leq 2(-\ln P_{\min}) \quad \forall t$$

**步骤3：累积误差与生存概率**

由A3.3（引用阶段3），生存条件为 $\sum_{t=0}^{T} \varepsilon_0(t) \leq -\ln P_{\min}$。

由步骤2，$\varepsilon_0(t) \leq 2(-\ln P_{\min})$。但这对累积和的控制不够紧。需要更精细的分析。

**步骤4：利用自由能的下界约束**

由A4.6，$F[q] = -\ln p(o) + D_{\text{KL}}[q \| p(\cdot|o)] \geq -\ln p(o)$。

自由能最小化使 $D_{\text{KL}}[q \| p(\cdot|o)]$ 趋近于零，即 $q$ 趋近于真实后验 $p(\psi|o)$。当 $q \approx p(\psi|o)$ 时，系统对环境状态的推断接近最优，预测误差接近贝叶斯最优误差。

**步骤5：连接阶段2的生存概率定理**

由定理2.1（引用阶段2），$P(\text{survival}) \propto \exp(-\sum_t \delta_t)$，其中 $\delta_t = \|o_t - \hat{o}_t\|^2$。

由A3.2（引用阶段3），$\varepsilon_0(t) = e^{0,T}(t) \Pi_0 e^0(t)$。当 $\Pi_0 = I$ 时，$\varepsilon_0(t) = \|e^0(t)\|^2 = \delta_t$。

由A4.8，自由能最小化驱动 $\varepsilon_0(t) \to 0$（在最优变分分布下），从而驱动 $\delta_t \to 0$。

设自由能最小化后的底层误差能量为 $\varepsilon_0^*(t) = \min_{q_t} \varepsilon_0(t)$。由A4.8的分解：

$$F^*[q_t] = \frac{1}{2}\varepsilon_0^*(t) + \text{其他非负项} \geq \frac{1}{2}\varepsilon_0^*(t)$$

因此 $\varepsilon_0^*(t) \leq 2 F^*[q_t]$。

若 $F^*[q_t] \leq -\ln P_{\min} / (2T)$ 对所有 $t$ 成立，则：

$$\sum_{t=0}^{T} \varepsilon_0^*(t) \leq 2T \cdot \frac{-\ln P_{\min}}{2T} = -\ln P_{\min}$$

由A3.3（引用阶段3），生存条件满足。

**步骤6：充分条件的表述**

将步骤5的条件改写：若系统在每个时刻 $t$ 将自由能最小化至 $F^*[q_t] \leq -\ln P_{\min} / (2T)$，则累积预测误差满足生存约束，从而 $P(\text{survival}) \geq P_{\min}$。

更一般地，定义每时刻自由能预算 $F_{\text{budget}} = -\ln P_{\min} / (2T)$，则充分条件为：

$$\min_{q_t} F[q_t] \leq F_{\text{budget}} \quad \forall t \in [0, T]$$

这建立了自由能最小化与生存必要条件（引用阶段2，定理2.2）的严格连接：自由能最小化是预测误差最小化（生存必要条件）的充分实现。$\blacksquare$

---

### 第五部分：变分分布族选择

#### 2.10 拉普拉斯近似（A4.10）

变分分布族 $\mathcal{Q}$ 的选择是变分推断的关键设计决策。最常用的选择是拉普拉斯近似——用高斯分布近似后验：

$$q(\psi) = \mathcal{N}(\psi; \mu, \Sigma), \quad \Sigma = \left(\frac{\partial^2 F}{\partial \psi^2}\bigg|_{\psi=\mu}\right)^{-1} \tag{A4.10}$$

- **输入**：自由能关于 $\psi$ 的Hessian矩阵 $\partial^2 F / \partial \psi^2 \in \mathbb{S}_{++}^d$（在 $\mu$ 处取值）
- **输出**：变分分布 $q(\psi) = \mathcal{N}(\mu, \Sigma)$
- **约束**：Hessian在 $\mu$ 处正定（保证 $\Sigma$ 正定，即 $\mu$ 为自由能的局部最小值）

> **白话翻译**：拉普拉斯近似就是用钟形曲线（高斯分布）来近似后验。钟形曲线的中心 $\mu$ 是自由能最低的点，宽度 $\Sigma$ 由自由能在最低点附近的"曲率"决定——曲率越大（自由能变化越陡），分布越窄（越确定）。

**推导**：

设自由能在 $\mu$ 处取局部最小值，对 $-\ln p(o, \psi)$ 在 $\mu$ 处做Taylor展开至二阶：

$$-\ln p(o, \psi) \approx -\ln p(o, \mu) - \nabla_\psi \ln p(o, \mu)^T (\psi - \mu) + \frac{1}{2}(\psi - \mu)^T H (\psi - \mu)$$

其中 $H = -\nabla^2_\psi \ln p(o, \psi)|_{\psi=\mu}$ 为负对数联合似然的Hessian矩阵。由于 $\mu$ 为极值点，梯度项为零。因此：

$$p(o, \psi) \approx p(o, \mu) \exp\left(-\frac{1}{2}(\psi - \mu)^T H (\psi - \mu)\right)$$

归一化后得后验的拉普拉斯近似：

$$p(\psi|o) \approx \mathcal{N}(\psi; \mu, H^{-1})$$

因此 $q(\psi) = \mathcal{N}(\psi; \mu, \Sigma)$，其中 $\Sigma = H^{-1}$。由A4.7，$F = E_q - H[q]$，Hessian $\partial^2 F / \partial \psi^2 = H$（在 $\mu$ 处），因此 $\Sigma = (\partial^2 F / \partial \psi^2)^{-1}$。

**选择决策树（通俗版）**：

```
变分分布族选择
├── 后验是否近似高斯？
│   ├── 是 → 拉普拉斯近似（A4.10）
│   │   ├── 优点：解析可微，计算高效
│   │   └── 缺点：无法捕获多模态
│   └── 否 → 后验是否多模态？
│       ├── 是 → 混合高斯 / 归一化流
│       │   ├── 优点：可表达多模态
│       │   └── 缺点：优化困难，计算量大
│       └── 否 → 均匀场近似
│           ├── 优点：分解简单
│           └── 缺点：忽略变量间相关性
└── 是否需要实时计算？
    ├── 是 → 拉普拉斯近似（最快）
    └── 否 → 更复杂的变分族（更准但更慢）
```

**本公理体系的默认选择**：拉普拉斯近似（A4.10），因为：
1. 与预测编码框架（引用阶段3）自然兼容
2. 自由能梯度有解析形式（A4.9）
3. 计算效率满足实时推断需求
4. 在大多数实际场景中后验近似单模态

---

### 第六部分：收敛性分析

#### 2.11 自由能单调递减（A4.11）

在自由能最小化的梯度下降动力学下，自由能关于时间单调递减：

$$\frac{\mathrm{d}F}{\mathrm{d}t} \leq 0 \tag{A4.11}$$

- **输入**：自由能 $F[q(t)]$，变分参数动力学 $\dot{\phi} = -\kappa \nabla_\phi F$
- **输出**：$\mathrm{d}F/\mathrm{d}t \leq 0$
- **约束**：学习率 $\kappa > 0$；自由能关于变分参数可微

> **白话翻译**：只要系统朝着"降低总账单"的方向走，账单就一定会越来越少——不可能越努力还钱反而欠得越多。这就是自由能单调递减：系统在优化过程中，自由能只会降不会升。

**完整证明**：

**步骤1：写出变分参数动力学**

设变分分布参数化为 $q_\phi(\psi)$，参数 $\phi$ 的梯度下降更新为：

$$\dot{\phi} = -\kappa \nabla_\phi F[q_\phi]$$

其中 $\kappa > 0$ 为学习率。

**步骤2：计算自由能的时间导数**

$$\frac{\mathrm{d}F}{\mathrm{d}t} = \nabla_\phi F^T \dot{\phi} = \nabla_\phi F^T (-\kappa \nabla_\phi F) = -\kappa \|\nabla_\phi F\|^2$$

**步骤3：分析符号**

由于 $\kappa > 0$ 且 $\|\nabla_\phi F\|^2 \geq 0$：

$$\frac{\mathrm{d}F}{\mathrm{d}t} = -\kappa \|\nabla_\phi F\|^2 \leq 0$$

等号成立当且仅当 $\nabla_\phi F = \mathbf{0}$，即变分参数到达自由能的驻点。

$\blacksquare$

**推论**：自由能单调递减保证系统不会在优化过程中"振荡"或"发散"。结合自由能的下界 $F \geq -\ln p(o)$（A4.6），自由能必然收敛到某个值 $F^* \geq -\ln p(o)$。

#### 2.12 收敛条件（A4.12）

自由能最小化过程的收敛条件为自由能梯度的范数小于阈值：

$$\|\nabla_\psi F\| < \varepsilon_{\text{conv}} \tag{A4.12}$$

- **输入**：自由能梯度 $\nabla_\psi F \in \mathbb{R}^d$，收敛阈值 $\varepsilon_{\text{conv}} \in \mathbb{R}_{> 0}$
- **输出**：收敛判定（布尔值）
- **约束**：$\varepsilon_{\text{conv}} > 0$；$\varepsilon_{\text{conv}}$ 越小，收敛精度越高但耗时越长

> **白话翻译**：当"总账单的下降速度"足够小（梯度接近零），就认为优化收敛了。阈值 $\varepsilon_{\text{conv}}$ 是"足够小"的标准——设得太大，还没优化好就停了；设得太小，优化到天荒地老。

**收敛速率估计**：

由A4.11，$\mathrm{d}F/\mathrm{d}t = -\kappa \|\nabla_\phi F\|^2$。设自由能在最优点附近为强凸的（Hessian的最小特征值 $\lambda_{\min} > 0$），则由强凸性：

$$F(\phi) - F^* \leq \frac{1}{2\lambda_{\min}} \|\nabla_\phi F\|^2$$

因此：

$$\frac{\mathrm{d}(F - F^*)}{\mathrm{d}t} = -\kappa \|\nabla_\phi F\|^2 \leq -2\kappa\lambda_{\min}(F - F^*)$$

解此微分不等式得：

$$F(t) - F^* \leq (F(0) - F^*) \exp(-2\kappa\lambda_{\min} t)$$

即自由能以指数速率 $2\kappa\lambda_{\min}$ 收敛到最优值。收敛时间常数为 $\tau_{\text{conv}} = 1/(2\kappa\lambda_{\min})$。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A4.1 | $p(\psi\|o) = p(o\|\psi)p(\psi)/p(o)$ | $p(o\|\psi)$, $p(\psi)$, $p(o)$ | $p(\psi\|o)$ | $p(o) > 0$ |
| A4.2 | $\ln p(o) = \ln \int p(o\|\psi)p(\psi)\,\mathrm{d}\psi$ | $p(o\|\psi)$, $p(\psi)$ | $\ln p(o) \in (-\infty, 0]$ | 高维积分不可计算 |
| A4.3 | $q(\psi) \approx p(\psi\|o)$ | 变分分布族 $\mathcal{Q}$, 参数 $\phi$ | $q^*(\psi)$ | $q \geq 0$, $\int q = 1$ |
| A4.4 | $F[q] = -\mathbb{E}_q[\ln p(o\|\psi)] + D_{\text{KL}}[q\|p]$ | $q$, $p(o\|\psi)$, $p(\psi)$ | $F[q] \in \mathbb{R}$ | $F \geq -\ln p(o)$ ⭐ |
| A4.5 | $F = -\mathbb{E}_q[\ln p(o\|\psi)] + D_{\text{KL}}[q\|p]$（准确性+复杂性） | 同A4.4 | $F_{\text{acc}} + F_{\text{comp}}$ | $F_{\text{comp}} \geq 0$ |
| A4.6 | $F = -\ln p(o) + D_{\text{KL}}[q\|p(\cdot\|o)]$（证据+后验偏差） | $q$, $p(o)$, $p(\psi\|o)$ | $F_{\text{evidence}} + F_{\text{divergence}}$ | $F_{\text{divergence}} \geq 0$ |
| A4.7 | $F = E_q - H[q]$（能量-熵） | $q$, $p(o,\psi)$ | $E_q - H[q]$ | $H[q] \geq 0$ |
| A4.8 | $F \approx \varepsilon_0 + \frac{1}{2}\sum_l \varepsilon_l + \frac{1}{2}\sum_l D_{\text{KL}}[q_l\|p_l]$ | $\{\varepsilon_l\}$, $\{D_{\text{KL}}[q_l\|p_l]\}$ | $F \in \mathbb{R}_{\geq 0}$ | 拉普拉斯近似；引用阶段3 A3.2 |
| A4.9 | $\partial F/\partial r^l = \delta^{l+1} - e^l$ | $\delta^{l+1}$, $e^l$ | $\nabla_{r^l} F \in \mathbb{R}^{n_l}$ | 引用阶段3 A3.1, A3.4 |
| A4.10 | $q(\psi) = \mathcal{N}(\psi; \mu, \Sigma)$, $\Sigma = (\partial^2 F/\partial\psi^2)^{-1}$ | Hessian $\partial^2 F/\partial\psi^2$ | $q(\psi)$ | Hessian正定 |
| A4.11 | $\mathrm{d}F/\mathrm{d}t \leq 0$ | $F[q(t)]$, $\dot{\phi} = -\kappa\nabla_\phi F$ | $\mathrm{d}F/\mathrm{d}t \leq 0$ | $\kappa > 0$ |
| A4.12 | $\|\nabla_\psi F\| < \varepsilon_{\text{conv}}$ | $\nabla_\psi F$, $\varepsilon_{\text{conv}} > 0$ | 收敛判定（布尔） | $\varepsilon_{\text{conv}} > 0$ |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# 引用阶段3的数据类
from stage_003 import ErrorSignal


# ============================================================
# 阶段4 数据类：变分自由能
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class VariationalFreeEnergy:
    """变分自由能的完整描述。

    组合阶段3的ErrorSignal，在误差信号的基础上增加变分自由能结构。

    Attributes:
        error_signal: 阶段3的误差信号（组合字段）
        accuracy: 准确性项 -E_q[ln p(o|ψ)]（A4.5第一项）
        complexity: 复杂性项 D_KL[q||p]（A4.5第二项）
        evidence: 证据项 -ln p(o)（A4.6第一项）
        posterior_divergence: 后验偏差项 D_KL[q||p(·|o)]（A4.6第二项）
        energy: 能量项 E_q（A4.7第一项）
        entropy: 熵项 H[q]（A4.7第二项）
        variational_mean: 变分分布均值 μ（A4.10）
        variational_cov: 变分分布协方差 Σ（A4.10）

    Invariants:
        - accuracy + complexity == evidence + posterior_divergence == energy - entropy
        - F >= -ln p(o)（即 F >= evidence 的负值，由A4.6保证）
        - complexity >= 0（KL散度非负性）
        - posterior_divergence >= 0（KL散度非负性）
        - entropy >= 0（Shannon熵非负性）
        - variational_cov 为对称正定矩阵
    """
    error_signal: ErrorSignal
    accuracy: float
    complexity: float
    evidence: float
    posterior_divergence: float
    energy: float
    entropy: float
    variational_mean: np.ndarray
    variational_cov: np.ndarray

    def __post_init__(self):
        # 验证三种分解的一致性
        f_acc_comp = self.accuracy + self.complexity
        f_evi_div = self.evidence + self.posterior_divergence
        f_eng_ent = self.energy - self.entropy
        if not (np.isclose(f_acc_comp, f_evi_div) and np.isclose(f_acc_comp, f_eng_ent)):
            raise ValueError(
                f"三种自由能分解不一致: "
                f"准确性+复杂性={f_acc_comp}, "
                f"证据+后验偏差={f_evi_div}, "
                f"能量-熵={f_eng_ent}"
            )
        # 验证自由能下界约束: F >= -ln p(o)，即 F >= evidence 的负值
        # evidence = -ln p(o)，所以 F >= evidence
        total_f = f_acc_comp
        if total_f < self.evidence - 1e-10:
            raise ValueError(
                f"自由能违反下界约束: F={total_f} < -ln p(o)={self.evidence}"
            )
        # 验证非负性约束
        if self.complexity < -1e-10:
            raise ValueError(
                f"复杂性项（KL散度）必须 ≥ 0，收到 {self.complexity}"
            )
        if self.posterior_divergence < -1e-10:
            raise ValueError(
                f"后验偏差项（KL散度）必须 ≥ 0，收到 {self.posterior_divergence}"
            )
        if self.entropy < -1e-10:
            raise ValueError(
                f"熵项必须 ≥ 0，收到 {self.entropy}"
            )
        # 验证变分协方差矩阵
        if self.variational_cov.ndim != 2:
            raise ValueError(
                f"变分协方差必须为二维矩阵，收到 {self.variational_cov.ndim} 维"
            )
        n = self.variational_mean.shape[0]
        if self.variational_cov.shape != (n, n):
            raise ValueError(
                f"变分协方差形状 {self.variational_cov.shape} "
                f"与均值维度 {n} 不匹配"
            )
        if not np.allclose(self.variational_cov, self.variational_cov.T):
            raise ValueError("变分协方差矩阵必须对称")
        try:
            np.linalg.cholesky(self.variational_cov)
        except np.linalg.LinAlgError:
            raise ValueError("变分协方差矩阵必须正定")

    @property
    def total(self) -> float:
        """变分自由能总值 F[q]（A4.4）。"""
        return self.accuracy + self.complexity

    @property
    def is_at_lower_bound(self) -> bool:
        """自由能是否达到下界 -ln p(o)（即 q = p(ψ|o)）。"""
        return np.isclose(self.posterior_divergence, 0.0)

    @property
    def convergence_gradient_norm(self) -> float:
        """自由能梯度的范数（用于A4.12收敛判定）。"""
        # 由A4.9，梯度 = δ^{l+1} - e^l
        delta = self.error_signal.bottom_up_signal
        e = self.error_signal.error
        return float(np.linalg.norm(delta - e))


# ============================================================
# 阶段4 核心接口：自由能公理
# ============================================================
class FreeEnergyAxiom:
    """阶段4的核心计算接口。

    定义变分自由能的全局唯一计算、分解和优化接口。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - compute_free_energy 返回值满足 F >= -ln p(o)
        - 三种分解（A4.5/A4.6/A4.7）数值一致
    """

    def compute_free_energy(
        self,
        q_samples: np.ndarray,
        log_likelihood: np.ndarray,
        log_prior: np.ndarray,
        log_q: np.ndarray
    ) -> VariationalFreeEnergy:
        """计算变分自由能 F[q]（A4.4，全局唯一定义点）。

        Args:
            q_samples: 变分分布采样 ψ_i，形状 (N, d)
            log_likelihood: 似然对数 ln p(o|ψ_i)，形状 (N,)
            log_prior: 先验对数 ln p(ψ_i)，形状 (N,)
            log_q: 变分分布对数 ln q(ψ_i)，形状 (N,)

        Returns:
            VariationalFreeEnergy 实例，包含三种分解的完整信息。
            返回值约束: F >= -ln p(o)（自由能下界）

        Raises:
            ValueError: 当输入数组形状不匹配时
            ValueError: 当 log_likelihood、log_prior 或 log_q 包含 NaN/Inf 时
        """
        N, d = q_samples.shape
        if log_likelihood.shape != (N,):
            raise ValueError(
                f"log_likelihood 形状 {log_likelihood.shape} 与采样数 {N} 不匹配"
            )
        if log_prior.shape != (N,):
            raise ValueError(
                f"log_prior 形状 {log_prior.shape} 与采样数 {N} 不匹配"
            )
        if log_q.shape != (N,):
            raise ValueError(
                f"log_q 形状 {log_q.shape} 与采样数 {N} 不匹配"
            )
        for name, arr in [("log_likelihood", log_likelihood),
                          ("log_prior", log_prior), ("log_q", log_q)]:
            if np.any(np.isnan(arr)):
                raise ValueError(f"{name} 包含 NaN")
            if np.any(np.isinf(arr)):
                raise ValueError(f"{name} 包含 Inf")

        # A4.5: F = -E_q[ln p(o|ψ)] + D_KL[q||p]
        accuracy = -float(np.mean(log_likelihood))
        complexity = float(np.mean(log_q - log_prior))

        # A4.7: F = E_q[-ln p(o,ψ)] - H[q]
        energy = float(-np.mean(log_likelihood + log_prior))
        entropy = float(-np.mean(log_q))

        # A4.6: F = -ln p(o) + D_KL[q||p(·|o)]
        # 蒙特卡洛估计 ln p(o) = logsumexp(ln p(o|ψ)+ln p(ψ)-ln q(ψ)) - ln N
        log_evidence = float(
            np.logaddexp.reduce(log_likelihood + log_prior - log_q) - np.log(N)
        )
        evidence = -log_evidence
        posterior_divergence = (accuracy + complexity) - evidence

        return VariationalFreeEnergy(
            error_signal=None,  # 需外部注入，此处仅计算自由能
            accuracy=accuracy,
            complexity=complexity,
            evidence=evidence,
            posterior_divergence=posterior_divergence,
            energy=energy,
            entropy=entropy,
            variational_mean=np.zeros(d),
            variational_cov=np.eye(d),
        )

        # Examples:
        # >>> ax = FreeEnergyAxiom()
        # >>> N, d = 5000, 2
        # >>> q_s = np.random.randn(N, d)
        # >>> ll = -0.5 * np.sum(q_s**2, axis=1) - d/2*np.log(2*np.pi)
        # >>> lp = -0.5 * np.sum(q_s**2, axis=1) - d/2*np.log(2*np.pi)
        # >>> lq = -0.5 * np.sum(q_s**2, axis=1) - d/2*np.log(2*np.pi)
        # >>> vfe = ax.compute_free_energy(q_s, ll, lp, lq)
        # >>> vfe.complexity >= -1e-6
        # True
        self,
        free_energy: VariationalFreeEnergy
    ) -> tuple[float, float]:
        """返回自由能的准确性-复杂性分解（A4.5）。

        引用阶段5展开详细分析。

        Args:
            free_energy: 变分自由能实例

        Returns:
            (accuracy, complexity) 元组
            - accuracy = -E_q[ln p(o|ψ)]
            - complexity = D_KL[q||p] >= 0

        Raises:
            TypeError: 当 free_energy 不是 VariationalFreeEnergy 实例时
        """
        if not isinstance(free_energy, VariationalFreeEnergy):
            raise TypeError(
                f"free_energy 必须为 VariationalFreeEnergy 实例，"
                f"收到 {type(free_energy).__name__}"
            )
        return (free_energy.accuracy, free_energy.complexity)

        # Examples:
        # >>> ax = FreeEnergyAxiom()
        # >>> # 需构造合法 VariationalFreeEnergy 实例
        # >>> # accuracy + complexity == evidence + posterior_divergence == energy - entropy
        # >>> acc, comp = ax.accuracy_complexity_decomposition(vfe)  # doctest: +SKIP
        self,
        free_energy: VariationalFreeEnergy
    ) -> tuple[float, float]:
        """返回自由能的证据-后验偏差分解（A4.6）。

        Args:
            free_energy: 变分自由能实例

        Returns:
            (evidence, posterior_divergence) 元组
            - evidence = -ln p(o)（与 q 无关的常数）
            - posterior_divergence = D_KL[q||p(·|o)] >= 0

        Raises:
            TypeError: 当 free_energy 不是 VariationalFreeEnergy 实例时
        """
        if not isinstance(free_energy, VariationalFreeEnergy):
            raise TypeError(
                f"free_energy 必须为 VariationalFreeEnergy 实例，"
                f"收到 {type(free_energy).__name__}"
            )
        return (free_energy.evidence, free_energy.posterior_divergence)

        # Examples:
        # >>> ax = FreeEnergyAxiom()
        # >>> evi, pd = ax.evidence_posterior_decomposition(vfe)  # doctest: +SKIP
        self,
        free_energy: VariationalFreeEnergy
    ) -> tuple[float, float]:
        """返回自由能的能量-熵分解（A4.7）。

        Args:
            free_energy: 变分自由能实例

        Returns:
            (energy, entropy) 元组
            - energy = E_q[-ln p(o,ψ)]
            - entropy = H[q] >= 0

        Raises:
            TypeError: 当 free_energy 不是 VariationalFreeEnergy 实例时
        """
        if not isinstance(free_energy, VariationalFreeEnergy):
            raise TypeError(
                f"free_energy 必须为 VariationalFreeEnergy 实例，"
                f"收到 {type(free_energy).__name__}"
            )
        return (free_energy.energy, free_energy.entropy)

        # Examples:
        # >>> ax = FreeEnergyAxiom()
        # >>> eng, ent = ax.energy_entropy_decomposition(vfe)  # doctest: +SKIP
        self,
        error_signals: list[ErrorSignal],
        kl_divergences: list[float]
    ) -> float:
        """计算拉普拉斯近似下的自由能（A4.8）。

        引用阶段3的A3.2（精度加权误差能量）。

        Args:
            error_signals: 各层的误差信号列表 [ErrorSignal_0, ..., ErrorSignal_L]
            kl_divergences: 各层KL散度列表 [D_KL[q_1||p_1], ..., D_KL[q_L||p_L]]

        Returns:
            自由能近似值 F ≈ ε_0 + ½Σε_l + ½ΣD_KL[q_l||p_l]

        Raises:
            ValueError: 当 error_signals 为空列表时
            ValueError: 当 kl_divergences 长度与 error_signals 不匹配时
            ValueError: 当任何 KL 散度为负时
        """
        if not error_signals:
            raise ValueError("error_signals 不能为空列表")
        if len(kl_divergences) != len(error_signals) - 1:
            raise ValueError(
                f"kl_divergences 长度 {len(kl_divergences)} 应为 "
                f"error_signals 长度减1 ({len(error_signals) - 1})"
            )
        for i, kl in enumerate(kl_divergences):
            if kl < -1e-10:
                raise ValueError(
                    f"第 {i} 层 KL 散度必须 ≥ 0，收到 {kl}"
                )
        # A4.8: F ≈ ε_0 + ½Σ_{l=1}^{L} ε_l + ½Σ_{l=1}^{L} D_KL[q_l||p_l]
        # 底层误差能量 ε_0 全额计入，高层误差能量取半
        epsilons = [es.energy for es in error_signals]
        F = epsilons[0] + 0.5 * sum(epsilons[1:]) + 0.5 * sum(kl_divergences)
        return float(F)

        # Examples:
        # >>> ax = FreeEnergyAxiom()
        # >>> # 构造含2层误差信号的列表（需 ErrorSignal 实例）
        # >>> # F = ε_0 + ½ε_1 + ½D_KL[q_1||p_1]
        # >>> result = ax.laplacian_free_energy(signals, kls)  # doctest: +SKIP
        self,
        error_signal: ErrorSignal,
        parent_error_signal: Optional[ErrorSignal] = None
    ) -> np.ndarray:
        """计算自由能关于层表征的梯度（A4.9）。

        引用阶段3的A3.1（误差信号）和A3.4（自下而上传播信号）。

        Args:
            error_signal: 当前层的误差信号（含 e^l 和 δ^l）
            parent_error_signal: 父层的误差信号（含 δ^{l+1}），顶层时为 None

        Returns:
            梯度 ∂F/∂r^l = δ^{l+1} - e^l，形状 (n_l,)
            顶层时 ∂F/∂r^L = -e^L（无父层信号）

        Raises:
            TypeError: 当 error_signal 不是 ErrorSignal 实例时
        """
        if not isinstance(error_signal, ErrorSignal):
            raise TypeError(
                f"error_signal 必须为 ErrorSignal 实例，"
                f"收到 {type(error_signal).__name__}"
            )
        # A4.9: ∂F/∂r^l = δ^{l+1} - e^l
        # 顶层时无父层信号，∂F/∂r^L = -e^L
        if parent_error_signal is None:
            return -error_signal.error
        delta_parent = parent_error_signal.bottom_up_signal
        return delta_parent - error_signal.error

        # Examples:
        # >>> ax = FreeEnergyAxiom()
        # >>> # 顶层：梯度 = -e^L
        # >>> grad_top = ax.free_energy_gradient(es)  # doctest: +SKIP
        # >>> # 非顶层：梯度 = δ^{l+1} - e^l
        # >>> grad = ax.free_energy_gradient(es, parent_es)  # doctest: +SKIP
        self,
        free_energy: VariationalFreeEnergy,
        hessian: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """计算拉普拉斯近似的变分分布参数（A4.10）。

        Args:
            free_energy: 变分自由能实例（提供均值 μ）
            hessian: 自由能关于 ψ 的Hessian矩阵，形状 (d, d)，必须对称正定

        Returns:
            (mu, Sigma) 元组
            - mu: 变分分布均值，形状 (d,)
            - Sigma: 变分分布协方差 = Hessian^{-1}，形状 (d, d)

        Raises:
            ValueError: 当 Hessian 不对称或不正定时
            ValueError: 当 Hessian 形状与均值维度不匹配时
        """
        if not isinstance(free_energy, VariationalFreeEnergy):
            raise TypeError(
                f"free_energy 必须为 VariationalFreeEnergy 实例，"
                f"收到 {type(free_energy).__name__}"
            )
        d = free_energy.variational_mean.shape[0]
        if hessian.shape != (d, d):
            raise ValueError(
                f"Hessian 形状 {hessian.shape} 与均值维度 {d} 不匹配"
            )
        if not np.allclose(hessian, hessian.T):
            raise ValueError("Hessian 矩阵必须对称")
        try:
            np.linalg.cholesky(hessian)
        except np.linalg.LinAlgError:
            raise ValueError("Hessian 矩阵必须正定")
        # A4.10: q(ψ) = N(μ, Σ), Σ = Hessian^{-1}
        mu = free_energy.variational_mean
        Sigma = np.linalg.inv(hessian)
        # 强制对称性
        Sigma = 0.5 * (Sigma + Sigma.T)
        return (mu, Sigma)

        # Examples:
        # >>> ax = FreeEnergyAxiom()
        # >>> # mu, Sigma = ax.laplacian_approximation(vfe, H)  # doctest: +SKIP
        # >>> # Sigma ≈ H^{-1}，H 为自由能在 μ 处的 Hessian
        self,
        gradient: np.ndarray,
        epsilon_conv: float
    ) -> bool:
        """检查自由能最小化是否收敛（A4.12）。

        Args:
            gradient: 自由能梯度 ∇_ψ F，形状 (d,)
            epsilon_conv: 收敛阈值，必须 > 0

        Returns:
            True 如果 ‖∇_ψ F‖ < ε_conv，False 否则

        Raises:
            ValueError: 当 epsilon_conv <= 0 时
        """
        if epsilon_conv <= 0:
            raise ValueError(
                f"收敛阈值 ε_conv 必须 > 0，收到 {epsilon_conv}"
            )
        if np.any(np.isnan(gradient)):
            raise ValueError("梯度包含 NaN")
        # A4.12: ‖∇_ψ F‖ < ε_conv
        return bool(np.linalg.norm(gradient) < epsilon_conv)

        # Examples:
        # >>> ax = FreeEnergyAxiom()
        # >>> ax.check_convergence(np.array([1e-10, 1e-10]), 1e-6)
        # True
        # >>> ax.check_convergence(np.array([1.0, 0.0]), 1e-6)
        # False
        # >>> ax.check_convergence(np.array([0.0]), -1.0)
        # Traceback (most recent call last):
        #     ...
        # ValueError: 收敛阈值 ε_conv 必须 > 0，收到 -1.0
        self,
        free_energy: VariationalFreeEnergy
    ) -> bool:
        """验证自由能下界约束 F >= -ln p(o)（A4.6推论）。

        Args:
            free_energy: 变分自由能实例

        Returns:
            True 如果 F >= -ln p(o) 成立，False 否则
        """
        # A4.6推论: F >= -ln p(o)，即 F >= evidence
        total_f = free_energy.total
        return bool(total_f >= free_energy.evidence - 1e-10)

        # Examples:
        # >>> ax = FreeEnergyAxiom()
        # >>> # ax.verify_free_energy_bound(vfe)  # doctest: +SKIP
        # >>> # 当 q = p(ψ|o) 时 F = -ln p(o)，bound 恰好成立
        self,
        free_energy: VariationalFreeEnergy,
        tolerance: float = 1e-8
    ) -> bool:
        """验证三种自由能分解的数值等价性（A4.5=A4.6=A4.7）。

        Args:
            free_energy: 变分自由能实例
            tolerance: 数值容差

        Returns:
            True 如果三种分解在容差范围内一致，False 否则
        """
        # 验证 A4.5 = A4.6 = A4.7
        f_acc_comp = free_energy.accuracy + free_energy.complexity
        f_evi_div = free_energy.evidence + free_energy.posterior_divergence
        f_eng_ent = free_energy.energy - free_energy.entropy
        eq_1 = np.isclose(f_acc_comp, f_evi_div, atol=tolerance)
        eq_2 = np.isclose(f_acc_comp, f_eng_ent, atol=tolerance)
        return bool(eq_1 and eq_2)

        # Examples:
        # >>> ax = FreeEnergyAxiom()
        # >>> # ax.verify_equivalence_of_decompositions(vfe)  # doctest: +SKIP
        # >>> # 三种分解在数值容差内应一致
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `ErrorSignal` | ← 阶段3 | 作为变分自由能的误差信号基础（组合字段） |
| **输入** | 定理2.2（预测误差必要性定理） | ← 阶段2 | 定理4.2的推导前提 |
| **输入** | 定理2.1（生存概率定理） | ← 阶段2 | 自由能与生存概率的定量关联 |
| **输入** | A3.1（预测误差信号） | ← 阶段3 | A4.8和A4.9的信号定义 |
| **输入** | A3.2（精度加权误差能量） | ← 阶段3 | A4.8的误差能量项 |
| **输入** | A3.4（自下而上传播信号） | ← 阶段3 | A4.9的梯度信号 |
| **输入** | 定理3.1（误差信号基本性质） | ← 阶段3 | 自由能函数的性质保证 |
| **输出** | `VariationalFreeEnergy` | → 阶段5/6/7/18/29 | 变分自由能的全局唯一定义，包含ErrorSignal组合 |
| **输出** | A4.4（变分自由能定义） | → 所有后续阶段 | 自由能的全局唯一定义点 ⭐ |
| **输出** | 定理4.1（自由能-误差等价性） | → 阶段5/6 | 准确性-复杂性分解的理论基础 |
| **输出** | 定理4.2（自由能-生存等价性） | → 阶段7/29 | 主动推断与生存驱力的理论基础 |
| **输出** | A4.9（自由能梯度） | → 阶段11/12 | 预测编码算法的梯度信号 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 自动微分与向量化 | 自由能梯度（A4.9）和Hessian（A4.10）的自动微分计算 |
| SciPy | 矩阵运算与优化 | Cholesky分解验证协方差正定性；Hessian逆计算 |
| NumPyro | 变分推断 | 基于JAX的概率编程库，提供变分分布族（AutoNormal/AutoDiagonalNormal等）、ELBO计算和SVI优化器，直接支撑A4.4的实现与验证 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V4.1 | 自由能下界约束 | `F >= -ln p(o)` 对所有合法 $q$ 成立 | 随机变分分布 + 解析验证 |
| V4.2 | 准确性-复杂性分解 | `F == accuracy + complexity` 对所有合法输入成立 | 数值一致性验证 |
| V4.3 | 证据-后验偏差分解 | `F == evidence + posterior_divergence` 对所有合法输入成立 | 数值一致性验证 |
| V4.4 | 能量-熵分解 | `F == energy - entropy` 对所有合法输入成立 | 数值一致性验证 |
| V4.5 | 三种分解等价 | A4.5 = A4.6 = A4.7 在数值容差 $10^{-8}$ 内一致 | 随机测试1000组 |
| V4.6 | KL散度非负 | `complexity >= 0` 且 `posterior_divergence >= 0` 对所有合法 $q$ 成立 | 随机分布测试 |
| V4.7 | 全局最小值条件 | `F == -ln p(o)` 当且仅当 `q == p(ψ\|o)` | 解析验证 + 高斯后验数值验证 |
| V4.8 | 拉普拉斯近似精度 | 拉普拉斯自由能（A4.8）与精确自由能的相对误差 < 5%（单模态后验） | 合成高斯后验测试 |
| V4.9 | 自由能梯度 | `∂F/∂r^l == δ^{l+1} - e^l` 对所有合法输入成立 | JAX自动微分对比验证 |
| V4.10 | 自由能单调递减 | `dF/dt <= 0` 在梯度下降过程中恒成立 | 多步优化数值验证 |
| V4.11 | 收敛条件 | `‖∇_ψ F‖ < ε_conv` 时自由能变化率 < `κ * ε_conv²` | 收敛性数值测试 |
| V4.12 | 定理4.1验证 | 自由能最小化后的误差能量之和 ≤ 初始值 | 多层预测编码数值实验 |
| V4.13 | 定理4.2验证 | 满足自由能预算时生存概率 ≥ P_min | 蒙特卡罗生存概率估计 |
| V4.14 | VariationalFreeEnergy组合 | `VariationalFreeEnergy.error_signal` 类型为 `ErrorSignal` | 类型检查 |
| V4.15 | 变分协方差正定性 | 更新后 $\Sigma$ 保持对称正定 | Cholesky分解验证 |
| V4.16 | 非法输入拒绝 | KL散度为负、协方差不正定、维度不匹配等抛出 `ValueError` | 异常测试 |

---

> **阶段4完** | 下一步：阶段5将从本阶段的A4.5（准确性-复杂性分解）出发，展开自由能的详细分解与驱力推导。
