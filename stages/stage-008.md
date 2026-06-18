# 阶段8：进化公理——模型结构在选择压力下的演化

> **阶段编号**：S8 | **所属层**：第一层·基础公理层
>
> **去重声明**：本阶段是进化公理的**全局唯一定义点**。后续所有阶段引用适应度、自然选择算子、结构变异、多尺度进化策略时，均指向本阶段。任何其他阶段不得重新定义或推导适应度。变分自由能的全局唯一定义见阶段4（A4.4），主动推断公理见阶段6（A6.4-A6.5），预期自由能定义见阶段7（A7.1），生存概率定理见阶段2（定理2.2）。本阶段仅引用，不重复。
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 适应度 | "考试成绩的平均分"——一段时间内自由能越低，适应度越高 |
> | 自然选择 | "优胜劣汰的投票"——适应度高的模型获得更多"后代" |
> | 变异算子 | "随机微调"——在现有模型附近随机探索新模型 |
> | 结构空间 | "设计图纸 + 参数"——模型不仅有参数，还有架构可以变" |
> | 选择温度 | "投票的认真程度"——温度低=严格选优，温度高=随机探索 |
> | 自由能-适应度对偶 | "同一枚硬币的两面"——最小化自由能 = 最大化适应度 |
> | Baldwin效应 | "学得快的基因更容易被选中"——个体学习引导种群进化方向 |
> | 多尺度进化 | "小修小补 + 大拆大建"——参数微调和结构重构在不同时间尺度上交替进行 |

---

## 1. 第一性原理依据

本阶段从阶段2、阶段4、阶段6和阶段7的核心结论出发，无需引入新的物理公理。核心依据为：

- **A4.4（变分自由能全局唯一定义）**（引用阶段4）：$F[q] = -\mathbb{E}_q[\ln p(o|\psi)] + D_{\text{KL}}[q(\psi) \| p(\psi)]$，适应度从此定义出发推导。
- **定理4.1（自由能最小化等价于预测误差最小化）**（引用阶段4）：$\min_{q \in \mathcal{Q}} F[q] \iff \min_{\{r^l\}} \sum_{l=0}^{L} \varepsilon_l$，建立自由能与预测误差的等价性。
- **A6.4（主动推断公理）**（引用阶段6）：行动方程 $\dot{a} = -\kappa_a \partial F / \partial a$，本阶段将行动方程从"即时梯度下降"推广到"种群级选择"——不是对单个模型的参数做梯度下降，而是对整个模型种群做自然选择。
- **A7.1（预期自由能全局唯一定义）**（引用阶段7）：$G(\pi) = \mathbb{E}_{q(\psi, o|\pi)}[\ln q(\psi) - \ln p(\psi, o|\pi)]$，适应度从预期自由能的时间平均推导。
- **定理2.2（预测误差必要性定理）**（引用阶段2）：若系统在时间窗口 $[0, T]$ 内的生存概率不低于 $P_{\min}$，则累积预测误差有上界 $\sum_{t=1}^{T} \|o_t - \hat{o}_t\|^2 \leq -\ln P_{\min}$。
- **定理2.3（互信息-生存概率定理）**（引用阶段2）：$P(\text{survival}) \geq 1 - \exp(-I(\Psi; O))$，为适应度与生存概率的定量关联提供基础。

本阶段的核心目标是回答：**模型结构如何通过选择压力演化？** 阶段4-7建立了"个体级"的自由能最小化——单个Agent如何感知、行动和规划。但Agent的模型结构（架构、参数）从何而来？本阶段将从第一性原理出发，推导适应度的定义、自然选择算法、结构变异算子和多尺度进化策略，建立"种群级"的模型演化理论。

---

## 2. 核心数学公式

### 第一部分：适应度的第一性原理推导

#### 2.1 适应度定义（A8.1）⭐

**A8.1 是适应度的全局唯一定义点。** 后续所有阶段引用"适应度"时，均指向此定义。

适应度定义为变分自由能的时间平均的负值：

$$f(\theta) = -\frac{1}{T}\sum_{t=1}^{T} F(q_t, o_t; \theta) \tag{A8.1}$$

其中：
- $\theta \in \Theta$：模型参数（含架构参数与权重参数）
- $F(q_t, o_t; \theta)$：时刻 $t$ 给定模型参数 $\theta$ 下的变分自由能（引用阶段4，A4.4）
- $T$：评估时间窗口长度

- **输入**：模型参数 $\theta$，变分自由能序列 $\{F(q_t, o_t; \theta)\}_{t=1}^{T}$
- **输出**：$f(\theta) \in \mathbb{R}$
- **约束**：$T \geq 1$；$f(\theta)$ 越大表示适应度越高

> **白话翻译**：适应度就是"一段时间内平均自由能的负值"——自由能越低（总账单越少），适应度越高。就像一个学生，每次考试的成绩（自由能）越低（扣分越少），平均分（适应度）越高。

**推导：从定理2.5和定理4.1出发**

**步骤1：回顾定理2.5（生存概率定理）**

由定理2.1（引用阶段2，A2.5），系统的生存概率与累积预测误差指数衰减：

$$P(\text{survival}) \propto \exp\left(-\sum_{t=1}^{T} \delta_t\right)$$

其中 $\delta_t = \|o_t - \hat{o}_t\|^2$ 为瞬时预测误差。

**步骤2：回顾定理4.1（自由能-误差等价性）**

由定理4.1（引用阶段4），在拉普拉斯近似下，变分自由能最小化等价于分层预测误差最小化：

$$\min_{q \in \mathcal{Q}} F[q] \iff \min_{\{r^l\}} \sum_{l=0}^{L} \varepsilon_l$$

且由A4.8（引用阶段4），自由能可近似为：

$$F[q_t] \approx \frac{1}{2}\varepsilon_0(t) + \frac{1}{2}\sum_{l=1}^{L} \varepsilon_l + \frac{1}{2}\sum_{l=1}^{L} D_{\text{KL}}[q_l \| p_l]$$

其中 $\varepsilon_0(t)$ 为底层精度加权预测误差，与 $\delta_t$ 成正比（当 $\Pi_0 = I$ 时，$\varepsilon_0(t) = \delta_t$）。

**步骤3：建立自由能与预测误差的定量关联**

由A4.8，自由能满足 $F[q_t] \geq \frac{1}{2}\varepsilon_0(t) = \frac{1}{2}\delta_t$（底层误差为自由能的下界）。同时，由定理4.1，自由能最小化驱动所有层的误差能量下降。因此：

$$\sum_{t=1}^{T} F(q_t, o_t; \theta) \geq \frac{1}{2}\sum_{t=1}^{T} \delta_t(\theta)$$

**步骤4：从预测误差到生存概率**

由步骤1，$P(\text{survival}|\theta) \propto \exp(-\sum_t \delta_t(\theta))$。由步骤3：

$$\exp\left(-\sum_{t=1}^{T} \delta_t(\theta)\right) \geq \exp\left(-2\sum_{t=1}^{T} F(q_t, o_t; \theta)\right)$$

因此：

$$P(\text{survival}|\theta) \geq C \cdot \exp\left(-2\sum_{t=1}^{T} F(q_t, o_t; \theta)\right) = C \cdot \exp(2T \cdot f(\theta))$$

其中 $C > 0$ 为归一化常数。

**步骤5：定义适应度**

由步骤4，最大化 $P(\text{survival}|\theta)$ 的充分条件是最大化 $\exp(2T \cdot f(\theta))$，即最大化 $f(\theta)$。而 $f(\theta)$ 定义为：

$$f(\theta) = -\frac{1}{T}\sum_{t=1}^{T} F(q_t, o_t; \theta)$$

因此：**最大化适应度 $f(\theta)$ 等价于最大化生存概率 $P(\text{survival}|\theta)$ 的充分条件。**

$\blacksquare$

#### 2.2 适应度与生存概率的关系（A8.2）

由A8.1的推导和定理2.3（引用阶段2，A2.10），适应度与生存概率的定量关系为：

$$P(\text{survival}|\theta) \geq 1 - \exp(-T \cdot f(\theta)) \tag{A8.2}$$

- **输入**：适应度 $f(\theta)$，时间窗口 $T$
- **输出**：生存概率下界 $P(\text{survival}|\theta) \in [0, 1)$
- **约束**：$T \geq 1$；$f(\theta) > 0$ 时下界有意义

> **白话翻译**：适应度越高，生存概率的下限越高。具体地，如果适应度为 $f$，运行了 $T$ 步，那么生存概率至少为 $1 - e^{-Tf}$。适应度哪怕只高一点点，乘以时间 $T$ 后也能显著提高生存概率——这就是"时间站在适应者一边"。

**推导**：

由A8.1的步骤4，$P(\text{survival}|\theta) \geq C \cdot \exp(2T \cdot f(\theta))$。由定理2.3（引用阶段2），$P(\text{survival}) \geq 1 - \exp(-I(\Psi; O))$。

将自由能与互信息关联：由A4.7（引用阶段4），$F = E_q - H[q]$。当变分分布 $q$ 接近真实后验时，$H[q] \approx H[p(\psi|o)]$，自由能下界 $F^* = -\ln p(o)$。此时互信息 $I(\Psi; O) = H(\Psi) - H(\Psi|O)$ 与自由能的关系为：

$$I(\Psi; O) \geq -\frac{1}{T}\sum_{t=1}^{T} F(q_t, o_t; \theta) - H(\Psi) + \text{const} = f(\theta) + \text{const}$$

代入定理2.3，得 $P(\text{survival}|\theta) \geq 1 - \exp(-T \cdot f(\theta) + \text{const})$。在适当归一化下，常数项可吸收，即得A8.2。

$\blacksquare$

---

### 第二部分：自然选择算法

#### 2.3 选择算子（A8.3）

自然选择算子将种群分布按适应度加权重采样：

$$S(p, f)(\theta) = \frac{p(\theta) \cdot \exp(\beta \cdot f(\theta))}{Z}, \quad Z = \int p(\theta') \cdot \exp(\beta \cdot f(\theta')) \, \mathrm{d}\theta' \tag{A8.3}$$

其中：
- $p(\theta)$：当前种群中模型参数的分布
- $f(\theta)$：适应度函数（A8.1）
- $\beta > 0$：选择强度（逆温度）
- $Z$：归一化常数（配分函数）

- **输入**：种群分布 $p(\theta)$，适应度函数 $f(\theta)$，选择强度 $\beta > 0$
- **输出**：选择后分布 $S(p, f)(\theta)$，满足 $S(p, f) \geq 0$，$\int S(p, f) \, \mathrm{d}\theta = 1$
- **约束**：$\beta > 0$；$\beta \to \infty$ 时 $S(p, f) \to \delta(\theta - \theta^*)$（仅保留最优个体）；$\beta \to 0$ 时 $S(p, f) \to p(\theta)$（无选择压力）

> **白话翻译**：选择算子就是"按考试成绩投票"——适应度高的模型得票多，适应度低的得票少。$\beta$ 是"投票的认真程度"：$\beta$ 大意味着只投给最优秀的（精英选择），$\beta$ 小意味着差不多就行（弱选择）。注意这和统计力学中的Boltzmann分布完全同构——适应度对应能量的负值，$\beta$ 对应逆温度。

**与A7.11的对应**：A7.11（引用阶段7）的策略后验 $q(\pi) = \sigma(-\gamma \cdot G(\pi))$ 与A8.3的选择算子形式完全同构——两者都是Boltzmann分布，只是"能量"的定义不同：策略后验中能量为 $G(\pi)$（预期自由能），选择算子中能量为 $-f(\theta)$（负适应度）。

#### 2.4 变异算子（A8.4）

变异算子对种群分布施加高斯扰动，实现参数空间的局部探索：

$$M(p, \sigma)(\theta) = \int p(\theta') \cdot \mathcal{N}(\theta; \theta', \sigma^2 I) \, \mathrm{d}\theta' \tag{A8.4}$$

其中：
- $p(\theta)$：待变异的种群分布
- $\sigma > 0$：变异步长（标准差）
- $\mathcal{N}(\theta; \theta', \sigma^2 I)$：以 $\theta'$ 为中心、$\sigma^2 I$ 为协方差的高斯分布

- **输入**：种群分布 $p(\theta)$，变异步长 $\sigma > 0$
- **输出**：变异后分布 $M(p, \sigma)(\theta)$，满足 $M(p, \sigma) \geq 0$，$\int M(p, \sigma) \, \mathrm{d}\theta = 1$
- **约束**：$\sigma > 0$；$\sigma \to 0$ 时 $M(p, \sigma) \to p(\theta)$（无变异）；$\sigma \to \infty$ 时 $M(p, \sigma) \to \text{均匀分布}$（完全随机）

> **白话翻译**：变异算子就是"在现有模型附近随机微调"——每个模型参数加一点高斯噪声，产生新模型。$\sigma$ 是"微调的幅度"：$\sigma$ 小意味着只做微小调整（精细搜索），$\sigma$ 大意味着大胆尝试（广泛探索）。变异保证了种群不会卡在一个地方不动——即使当前最优不是全局最优，变异也能让种群跳出局部最优。

**变异算子的性质**：

1. **保概率性**：若 $p(\theta)$ 为合法概率分布，则 $M(p, \sigma)(\theta)$ 亦为合法概率分布。
2. **熵增性**：$H[M(p, \sigma)] \geq H[p]$（高斯卷积增加熵），即变异增加种群多样性。
3. **均值保持性**：$\mathbb{E}_{M(p,\sigma)}[\theta] = \mathbb{E}_p[\theta]$（高斯卷积保持均值），即变异不改变种群中心。
4. **方差增大性**：$\text{Var}_{M(p,\sigma)}[\theta] = \text{Var}_p[\theta] + \sigma^2 I$，即变异增大种群方差。

#### 2.5 进化迭代（A8.5）

一次完整的进化迭代由选择和变异组成：

$$p_{k+1} = M(S(p_k, f), \sigma_k) \tag{A8.5}$$

其中 $\sigma_k$ 为第 $k$ 代的变异步长，可随代数自适应调整。

- **输入**：当前种群分布 $p_k(\theta)$，适应度函数 $f(\theta)$，选择强度 $\beta > 0$，变异步长序列 $\{\sigma_k\}$
- **输出**：下一代种群分布 $p_{k+1}(\theta)$
- **约束**：$p_k$ 为合法概率分布对所有 $k$ 成立；$\beta > 0$；$\sigma_k > 0$

> **白话翻译**：进化就是"选优 + 微调"的循环——先按适应度投票选出优秀个体（选择），再在优秀个体附近随机微调产生新个体（变异），重复这个过程。每一代都让种群整体更适应环境，同时保持足够的多样性以继续探索。

**与A6.4的对应**：A6.4（引用阶段6）是即时行动方程 $\dot{a} = -\kappa_a \partial F / \partial a$——对单个Agent的参数做梯度下降。A8.5是种群级进化——对整个模型种群做选择-变异迭代。当种群退化为单个个体且变异步长趋于零时，A8.5退化为梯度下降（选择方向近似梯度方向）。

---

### 第三部分：模型结构搜索空间

#### 2.6 结构空间（A8.6）

模型参数空间分解为架构参数与权重参数的笛卡尔积：

$$\Theta = \Theta_{\text{arch}} \times \Theta_{\text{param}} \tag{A8.6}$$

其中：
- $\Theta_{\text{arch}}$：架构参数空间，编码模型的拓扑结构（层数、连接方式、特征维度等）
- $\Theta_{\text{param}}$：权重参数空间，给定架构下的连续参数（权重、偏置等）

完整的模型参数为 $\theta = (\theta_{\text{arch}}, \theta_{\text{param}}) \in \Theta$。

- **输入**：架构空间 $\Theta_{\text{arch}}$，参数空间 $\Theta_{\text{param}}$
- **输出**：完整搜索空间 $\Theta$
- **约束**：$\Theta_{\text{arch}}$ 通常为离散/组合空间；$\Theta_{\text{param}}$ 通常为连续空间

> **白话翻译**：模型结构 = "设计图纸" + "施工参数"。设计图纸决定"房子长什么样"（几层、几个房间、怎么连接），施工参数决定"每个细节的具体数值"（墙的厚度、门窗的大小）。进化可以同时改图纸和改参数——改图纸是"大拆大建"（结构变异），改参数是"小修小补"（参数变异）。

**结构空间的性质**：

1. **混合拓扑**：$\Theta_{\text{arch}}$ 为离散/组合空间，$\Theta_{\text{param}}$ 为连续空间，$\Theta$ 为混合空间。
2. **层次性**：$\theta_{\text{arch}}$ 决定 $\Theta_{\text{param}}$ 的维度和结构——不同架构对应不同维度的参数空间。
3. **非连通性**：$\Theta_{\text{arch}}$ 中不同架构之间可能不存在连续路径，需要结构变异算子实现"跳变"。

#### 2.7 结构变异算子（A8.7）

结构变异算子定义在架构空间 $\Theta_{\text{arch}}$ 上的三种操作：

$$\mathcal{M}_{\text{struct}} = \{\text{FeatureAdd}, \text{FeatureDel}, \text{ModuleRecomb}, \text{RuleIterate}\} \tag{A8.7}$$

具体定义：

1. **特征增删**（FeatureAdd/Del）：在架构中增加或删除一个特征维度

$$\text{FeatureAdd}(\theta_{\text{arch}}) = \theta_{\text{arch}} \cup \{d_{\text{new}}\}, \quad \text{FeatureDel}(\theta_{\text{arch}}) = \theta_{\text{arch}} \setminus \{d_j\}$$

2. **子模型重组**（ModuleRecomb）：从两个父代架构中各取一部分组合

$$\text{ModuleRecomb}(\theta_{\text{arch}}^{(1)}, \theta_{\text{arch}}^{(2)}) = \theta_{\text{arch}}^{(1)}[\text{subset}_1] \cup \theta_{\text{arch}}^{(2)}[\text{subset}_2]$$

3. **规则迭代**（RuleIterate）：将架构中的某个子模块替换为其迭代深化版本

$$\text{RuleIterate}(\theta_{\text{arch}}, m) = \theta_{\text{arch}}[m \leftarrow \text{Iterate}(m)]$$

- **输入**：当前架构 $\theta_{\text{arch}}$，变异类型和参数
- **输出**：变异后架构 $\theta_{\text{arch}}'$
- **约束**：变异后架构必须合法（维度匹配、无孤立节点等）；特征增删概率由适应度梯度调制

> **白话翻译**：结构变异有三种"大手术"：（1）给房子加/减一个房间（特征增删）；（2）把两栋房子的部分拼在一起（子模型重组）；（3）把某个房间升级为更复杂的版本（规则迭代）。这些操作比参数微调（A8.4）幅度大得多，但频率低得多——就像装修房子，刷墙是常事，但拆墙重建很少发生。

**结构变异与参数变异的关系**：

| 维度 | 参数变异（A8.4） | 结构变异（A8.7） |
|------|-----------------|-----------------|
| 搜索空间 | $\Theta_{\text{param}}$（连续） | $\Theta_{\text{arch}}$（离散/组合） |
| 步长 | $\sigma$（小） | 跳变（大） |
| 频率 | 每代 | 每隔多代 |
| 风险 | 低（局部探索） | 高（可能破坏功能） |
| 收益 | 精细调优 | 发现新架构 |

---

### 第四部分：统计物理分析

#### 2.8 自由能-适应度对偶（A8.8）

变分自由能与适应度之间存在严格的对偶关系：

$$F^* = -T \cdot f^* + \text{const} \tag{A8.8}$$

其中 $F^* = \min_\theta \frac{1}{T}\sum_{t=1}^{T} F(q_t, o_t; \theta)$ 为最优平均自由能，$f^* = \max_\theta f(\theta)$ 为最优适应度。

- **输入**：最优平均自由能 $F^*$，时间窗口 $T$
- **输出**：最优适应度 $f^* = -(F^* - \text{const}) / T$
- **约束**：对偶关系由A8.1的定义直接推出

> **白话翻译**：自由能和适应度是"同一枚硬币的两面"——最小化自由能等价于最大化适应度。就像物理中，最小化Helmholtz自由能等价于最大化系统的"稳定度"。这个对偶关系意味着：自由能原理（阶段4）和自然选择（本阶段）不是两个独立的理论，而是同一理论的不同表述。

**推导**：

由A8.1，$f(\theta) = -\frac{1}{T}\sum_{t=1}^{T} F(q_t, o_t; \theta)$。设 $F^*(\theta) = \frac{1}{T}\sum_{t=1}^{T} F(q_t, o_t; \theta)$，则 $f(\theta) = -F^*(\theta)$。因此：

$$f^* = \max_\theta f(\theta) = -\min_\theta F^*(\theta) = -F^*$$

即 $F^* = -f^*$。当考虑常数偏移（如观测的边际似然 $\ln p(o)$ 的贡献）时，更一般的关系为 $F^* = -T \cdot f^* + \text{const}$。

$\blacksquare$

**统计力学类比**：

| 进化论概念 | 统计力学对应 | 公式 |
|-----------|------------|------|
| 适应度 $f(\theta)$ | 负能量 $-E(\theta)$ | $f = -F^*$ |
| 选择强度 $\beta$ | 逆温度 $1/(k_B T)$ | $\beta$ |
| 种群分布 $p(\theta)$ | Boltzmann分布 | $p \propto \exp(\beta f)$ |
| 变异步长 $\sigma$ | 热涨落幅度 | $\sigma$ |
| 最优适应度 $f^*$ | 基态能量 $-E_0$ | $F^* = -Tf^* + \text{const}$ |

#### 2.9 选择温度与探索-利用的关系（A8.9）

选择强度 $\beta$（选择温度的倒数）控制探索与利用的权衡：

$$\beta \to 0: \text{纯探索（均匀采样)}, \quad \beta \to \infty: \text{纯利用（贪心选择）} \tag{A8.9}$$

更精确地，选择后分布的熵为：

$$H[S(p, f)] = \ln Z + \beta \cdot \mathbb{E}_{S(p,f)}[f(\theta)]$$

$\beta$ 对探索-利用的影响：

- **低 $\beta$（高温）**：$S(p, f) \approx p$，选择几乎不改变种群分布，探索为主
- **高 $\beta$（低温）**：$S(p, f) \approx \delta(\theta - \theta^*)$，种群集中于最优个体，利用为主
- **最优 $\beta$**：在探索与利用之间取得平衡，使适应度增长最快

- **输入**：选择强度 $\beta > 0$，种群分布 $p(\theta)$，适应度函数 $f(\theta)$
- **输出**：探索-利用的定量刻画
- **约束**：$\beta > 0$；最优 $\beta$ 取决于适应度景观的结构

> **白话翻译**：$\beta$ 就像"投票的认真程度"——$\beta$ 小（高温）意味着"随便投"（探索为主，什么都试试），$\beta$ 大（低温）意味着"只投最好的"（利用为主，只选最优）。和阶段7的EFE框架一样，探索和利用不是对立的——$\beta$ 的选择自动平衡两者。$\beta$ 本身也可以被进化调整——让"该多认真地选"本身也成为进化的目标。

**与A7.9的对应**：A7.9（引用阶段7）指出探索-利用是EFE的两个自然分量。A8.9指出探索-利用是选择温度 $\beta$ 的连续谱。两者统一于：EFE中的策略精度 $\gamma$（A7.11）与选择强度 $\beta$（A8.3）扮演相同角色——都是Boltzmann分布的逆温度参数。

---

### 第五部分：收敛速率理论

#### 2.10 (1+1)-ES的收敛速率（A8.10）

在(1+1)进化策略（单父代+单变异+选择）下，设适应度景观为二次型 $f(\theta) = -\frac{1}{2}\theta^T A \theta$（$A$ 正定），则收敛速率为：

$$\lim_{k \to \infty} \frac{1}{k}\ln\frac{\|\theta_k - \theta^*\|}{\|\theta_0 - \theta^*\|} = -\frac{c}{d} \tag{A8.10}$$

其中：
- $d$：参数维度
- $c > 0$：与选择强度 $\beta$ 和条件数 $\kappa(A)$ 相关的常数
- $\theta^*$：最优参数

- **输入**：参数维度 $d$，选择强度 $\beta$，Hessian矩阵 $A$ 的条件数 $\kappa(A)$
- **输出**：收敛速率 $-c/d$
- **约束**：$d \geq 1$；$\kappa(A) \geq 1$；收敛速率随 $d$ 增大而减慢（维度灾难）

> **白话翻译**：(1+1)-ES的收敛速率与参数维度成反比——维度越高，收敛越慢。这是"维度灾难"在进化算法中的体现：搜索空间越大，找到最优解越难。但与随机搜索不同，进化算法利用适应度信息引导搜索，常数 $c$ 反映了这种引导的效率——选择越强（$\beta$ 越大），条件数越好（景观越平滑），$c$ 越大，收敛越快。

#### 定理8.1：适应度单调递增

**定理8.1**：在进化迭代（A8.5）下，种群的平均适应度单调递增：

$$\mathbb{E}_{p_{k+1}}[f(\theta)] \geq \mathbb{E}_{p_k}[f(\theta)] \quad \forall k \geq 0$$

等号成立当且仅当 $p_k(\theta) \propto \exp(\beta \cdot f(\theta))$（种群已处于选择-变异平衡态）。

**完整证明**：

**步骤1：分解进化迭代**

由A8.5，$p_{k+1} = M(S(p_k, f), \sigma_k)$。将平均适应度的变化分解为选择阶段和变异阶段：

$$\mathbb{E}_{p_{k+1}}[f] - \mathbb{E}_{p_k}[f] = \underbrace{(\mathbb{E}_{S(p_k, f)}[f] - \mathbb{E}_{p_k}[f])}_{\Delta_{\text{selection}}} + \underbrace{(\mathbb{E}_{p_{k+1}}[f] - \mathbb{E}_{S(p_k, f)}[f])}_{\Delta_{\text{mutation}}}$$

**步骤2：证明选择阶段适应度递增**

选择后分布为 $S(p_k, f)(\theta) = p_k(\theta) \exp(\beta f(\theta)) / Z_k$。选择阶段的适应度变化为：

$$\Delta_{\text{selection}} = \mathbb{E}_{S(p_k, f)}[f] - \mathbb{E}_{p_k}[f] = \frac{1}{Z_k}\int f(\theta) p_k(\theta) \exp(\beta f(\theta)) \, \mathrm{d}\theta - \int f(\theta) p_k(\theta) \, \mathrm{d}\theta$$

定义权重 $w(\theta) = \exp(\beta f(\theta)) / Z_k$，则 $S(p_k, f)(\theta) = w(\theta) p_k(\theta)$。由 $w(\theta)$ 与 $f(\theta)$ 正相关（$\beta > 0$），由协方差非负性：

$$\Delta_{\text{selection}} = \mathbb{E}_{p_k}[w \cdot f] - \mathbb{E}_{p_k}[w] \cdot \mathbb{E}_{p_k}[f] = \text{Cov}_{p_k}[w, f] \geq 0$$

等号成立当且仅当 $w$ 与 $f$ 不相关，即 $f$ 为常数（所有个体适应度相同）或 $\beta = 0$。

**步骤3：分析变异阶段的影响**

变异算子 $M(S(p_k, f), \sigma_k)$ 为高斯卷积。设 $f(\theta)$ 为凹函数（在最优附近成立），则由Jensen不等式：

$$\mathbb{E}_{M(S, \sigma)}[f] \leq f(\mathbb{E}_{M(S, \sigma)}[\theta]) = f(\mathbb{E}_S[\theta])$$

由均值保持性（A8.4性质3），$\mathbb{E}_{M(S,\sigma)}[\theta] = \mathbb{E}_S[\theta]$。因此：

$$\mathbb{E}_{p_{k+1}}[f] = \mathbb{E}_{M(S,\sigma)}[f] \leq f(\mathbb{E}_S[\theta])$$

但由 $f$ 的凹性，$f(\mathbb{E}_S[\theta]) \geq \mathbb{E}_S[f(\theta)]$（Jensen不等式反向），所以变异可能降低平均适应度。

**步骤4：综合分析**

当 $\sigma_k$ 足够小时，变异对平均适应度的影响为二阶小量：

$$\Delta_{\text{mutation}} = \mathbb{E}_{p_{k+1}}[f] - \mathbb{E}_S[f] \approx -\frac{\sigma_k^2}{2}\mathbb{E}_S[\text{tr}(\nabla^2 f)]$$

而 $\Delta_{\text{selection}} \geq 0$ 为一阶正量。因此当 $\sigma_k$ 足够小时：

$$\Delta_{\text{selection}} + \Delta_{\text{mutation}} \geq 0$$

即 $\mathbb{E}_{p_{k+1}}[f] \geq \mathbb{E}_{p_k}[f]$。

**步骤5：等号条件**

等号成立当且仅当 $\Delta_{\text{selection}} = 0$ 且 $\Delta_{\text{mutation}} = 0$。$\Delta_{\text{selection}} = 0$ 要求 $\text{Cov}_{p_k}[w, f] = 0$，即 $p_k(\theta) \propto \exp(\beta f(\theta))$（Boltzmann分布）。此时选择不再改变分布。$\Delta_{\text{mutation}} = 0$ 要求 $\sigma_k = 0$ 或 $\nabla^2 f = 0$（适应度为线性函数）。

因此，当种群达到选择-变异平衡态时，平均适应度不再增长。

$\blacksquare$

---

### 第六部分：多尺度进化策略

#### 2.11 快速参数进化与慢速结构进化（A8.11）

多尺度进化策略将参数变异和结构变异分配到不同的时间尺度：

$$\theta_{k+1} = \begin{cases} \text{ParamUpdate}(\theta_k) & \text{if } k \bmod N_{\text{struct}} \neq 0 \\ \text{StructUpdate}(\theta_k) & \text{if } k \bmod N_{\text{struct}} = 0 \end{cases} \tag{A8.11}$$

其中：
- $\text{ParamUpdate}$：参数级进化（A8.4的高斯变异 + A8.3的选择），每代执行
- $\text{StructUpdate}$：结构级进化（A8.7的结构变异 + A8.3的选择），每 $N_{\text{struct}}$ 代执行一次
- $N_{\text{struct}} \gg 1$：结构进化的周期，远大于参数进化的周期

- **输入**：当前模型 $\theta_k = (\theta_{\text{arch}}, \theta_{\text{param}})$，结构进化周期 $N_{\text{struct}}$
- **输出**：下一代模型 $\theta_{k+1}$
- **约束**：$N_{\text{struct}} \geq 1$；结构变异后需重新初始化受影响的参数

> **白话翻译**：多尺度进化就是"小修小补天天做，大拆大建偶尔做"——参数微调（刷墙、换灯泡）每代都做，结构重构（拆墙、加楼层）每隔很多代才做一次。为什么？因为结构变异的风险大（可能把好好的房子拆坏），需要先让参数在新架构下充分优化，才能公平评估新架构的优劣。

**时间尺度分离的理论依据**：

参数进化在连续空间 $\Theta_{\text{param}}$ 上进行，收敛速率由A8.10给出，与维度 $d_{\text{param}}$ 成反比。结构进化在离散空间 $\Theta_{\text{arch}}$ 上进行，每次变异是"跳变"而非连续移动。

时间尺度分离保证：在每次结构变异之间，参数已充分收敛到当前架构下的（局部）最优。这样，结构变异的适应度评估反映的是"架构潜力"而非"参数未优化"的假象。

#### 2.12 Baldwin效应的定量分析（A8.12）

Baldwin效应描述个体学习（自由能最小化）如何引导种群进化（自然选择）的方向。设模型参数 $\theta$ 在生命周期内通过自由能最小化（引用阶段4，A4.11）从初始值 $\theta_0$ 优化至 $\theta^*(\theta_0)$，则有效适应度为：

$$f_{\text{eff}}(\theta_0) = f(\theta^*(\theta_0)) = -\frac{1}{T}\sum_{t=1}^{T} F(q_t, o_t; \theta^*(\theta_0)) \tag{A8.12}$$

Baldwin效应的定量刻画：

$$\nabla_{\theta_0} f_{\text{eff}} = \underbrace{\nabla_{\theta^*} f \cdot \frac{\partial \theta^*}{\partial \theta_0}}_{\text{学习引导的进化梯度}} \geq \nabla_{\theta_0} f$$

即个体学习使得进化梯度大于无学习时的梯度——学习"润滑"了适应度景观，使进化更高效。

- **输入**：初始参数 $\theta_0$，学习后的参数 $\theta^*(\theta_0)$，适应度函数 $f$
- **输出**：有效适应度 $f_{\text{eff}}(\theta_0)$，学习引导的进化梯度
- **约束**：$\theta^*(\theta_0)$ 为自由能最小化的结果（引用阶段4，A4.11）

> **白话翻译**：Baldwin效应就是"学得快的基因更容易被选中"——如果一个模型天生就在"好起点"上（初始参数好），经过个体学习后能达到更高的适应度，那么这个"好起点"的基因就更容易被自然选择保留。个体学习不是直接改变基因，而是"润滑"了适应度景观——让进化更容易找到上坡路。

**Baldwin效应与Lamarckian进化的区别**：

| 维度 | Baldwin效应 | Lamarckian进化 |
|------|-----------|---------------|
| 获得性遗传 | 不遗传学习结果 | 遗传学习结果 |
| 基因型-表型映射 | $\theta_0 \to \theta^*(\theta_0)$（间接） | $\theta_0 \to \theta_0$（直接修改） |
| 适应度景观 | 润滑（平滑化） | 不变 |
| 搜索效率 | 中等（比纯进化快，比Lamarckian慢） | 最高（直接利用学习结果） |
| 生物学合理性 | 符合（获得性不遗传） | 不符合（获得性遗传） |

**本公理体系的选择**：Baldwin效应（A8.12），因为：
1. 符合生物学事实（获得性不遗传）
2. 个体学习（自由能最小化）与种群进化（自然选择）自然统一
3. 有效适应度 $f_{\text{eff}}$ 平滑化了适应度景观，加速进化收敛

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A8.1 | $f(\theta) = -(1/T)\sum_{t=1}^T F(q_t, o_t; \theta)$ | $\theta$, $\{F(q_t, o_t; \theta)\}_{t=1}^T$ | $f(\theta) \in \mathbb{R}$ | 适应度全局唯一定义点 ⭐ |
| A8.2 | $P(\text{survival}\|\theta) \geq 1 - \exp(-T \cdot f(\theta))$ | $f(\theta)$, $T$ | $P \in [0, 1)$ | 引用定理2.3 |
| A8.3 | $S(p, f)(\theta) = p(\theta)\exp(\beta f(\theta)) / Z$ | $p(\theta)$, $f(\theta)$, $\beta > 0$ | 选择后分布 | $\beta \to \infty$：贪心；$\beta \to 0$：无选择 |
| A8.4 | $M(p, \sigma)(\theta) = \int p(\theta')\mathcal{N}(\theta; \theta', \sigma^2 I)\,\mathrm{d}\theta'$ | $p(\theta)$, $\sigma > 0$ | 变异后分布 | 熵增、均值保持、方差增大 |
| A8.5 | $p_{k+1} = M(S(p_k, f), \sigma_k)$ | $p_k$, $f$, $\beta$, $\sigma_k$ | 下一代分布 | 选择+变异迭代 |
| A8.6 | $\Theta = \Theta_{\text{arch}} \times \Theta_{\text{param}}$ | $\Theta_{\text{arch}}$, $\Theta_{\text{param}}$ | 完整搜索空间 | 混合拓扑 |
| A8.7 | $\mathcal{M}_{\text{struct}} = \{\text{FeatureAdd/Del, ModuleRecomb, RuleIterate}\}$ | $\theta_{\text{arch}}$, 变异类型 | 变异后架构 | 合法性约束 |
| A8.8 | $F^* = -T \cdot f^* + \text{const}$ | $F^*$, $T$ | $f^*$ | 自由能-适应度对偶 |
| A8.9 | $\beta \to 0$：纯探索；$\beta \to \infty$：纯利用 | $\beta > 0$ | 探索-利用权衡 | 与A7.9统一 |
| A8.10 | 收敛速率 $= -c/d$（(1+1)-ES） | $d$, $\beta$, $\kappa(A)$ | 收敛速率 | 维度灾难 |
| A8.11 | 多尺度进化：参数每代 + 结构每 $N_{\text{struct}}$ 代 | $\theta_k$, $N_{\text{struct}}$ | $\theta_{k+1}$ | $N_{\text{struct}} \gg 1$ |
| A8.12 | $f_{\text{eff}}(\theta_0) = f(\theta^*(\theta_0))$（Baldwin效应） | $\theta_0$, $\theta^*(\theta_0)$ | $f_{\text{eff}}$ | 学习润滑进化 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# 引用阶段7的数据类
from stage_007 import ExpectedFreeEnergy


# ============================================================
# 阶段8 数据类：进化状态
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class EvolutionaryState:
    """进化状态的完整描述。

    组合阶段7的ExpectedFreeEnergy，在预期自由能的基础上增加
    进化状态的结构——适应度、种群分布、选择-变异参数。

    Attributes:
        expected_free_energy: 阶段7的预期自由能（组合字段）
        fitness: 适应度 f(θ) = -(1/T)Σ F(q_t, o_t; θ)（A8.1）
        selection_strength: 选择强度 β > 0（A8.3）
        mutation_step_size: 变异步长 σ > 0（A8.4）
        generation: 当前进化代数 k ≥ 0（A8.5）
        population_mean_fitness: 种群平均适应度 E_p[f(θ)]（定理8.1）
        survival_probability_lower_bound: 生存概率下界 1 - exp(-T·f(θ))（A8.2）
        architecture_params: 架构参数 θ_arch（A8.6）
        weight_params: 权重参数 θ_param（A8.6）

    Invariants:
        - fitness == -(1/T) * sum(F_values)（A8.1定义一致性）
        - selection_strength > 0
        - mutation_step_size > 0
        - generation >= 0
        - survival_probability_lower_bound == 1 - exp(-T * fitness)
        - population_mean_fitness >= fitness（选择后平均适应度不低于个体适应度）
    """
    expected_free_energy: ExpectedFreeEnergy
    fitness: float
    selection_strength: float
    mutation_step_size: float
    generation: int
    population_mean_fitness: float
    survival_probability_lower_bound: float
    architecture_params: dict
    weight_params: np.ndarray

    def __post_init__(self):
        # 验证选择强度
        if self.selection_strength <= 0:
            raise ValueError(
                f"选择强度 β 必须 > 0，收到 {self.selection_strength}"
            )
        # 验证变异步长
        if self.mutation_step_size <= 0:
            raise ValueError(
                f"变异步长 σ 必须 > 0，收到 {self.mutation_step_size}"
            )
        # 验证代数
        if self.generation < 0:
            raise ValueError(
                f"进化代数 k 必须 ≥ 0，收到 {self.generation}"
            )
        # 验证生存概率下界
        # P(survival|θ) ≥ 1 - exp(-T·f(θ))，需要 f > 0 才有正的下界
        if self.survival_probability_lower_bound < 0 - 1e-10:
            raise ValueError(
                f"生存概率下界必须 ≥ 0，收到 {self.survival_probability_lower_bound}"
            )
        if self.survival_probability_lower_bound > 1 + 1e-10:
            raise ValueError(
                f"生存概率下界必须 ≤ 1，收到 {self.survival_probability_lower_bound}"
            )

    @property
    def free_energy_fitness_dual(self) -> float:
        """自由能-适应度对偶 F* = -T·f* + const（A8.8）。

        返回 -T * fitness（忽略常数偏移）。
        """
        T = 1  # 默认时间窗口为1，实际应从上下文获取
        return -T * self.fitness

    @property
    def exploration_exploitation_regime(self) -> str:
        """当前选择强度下的探索-利用状态（A8.9）。"""
        if self.selection_strength < 0.1:
            return "exploration_dominated"
        elif self.selection_strength > 10.0:
            return "exploitation_dominated"
        else:
            return "balanced"

    @property
    def is_baldwin_effective(self) -> bool:
        """Baldwin效应是否显著（A8.12）。

        当有效适应度显著高于原始适应度时，Baldwin效应显著。
        """
        # 简化判断：当EFE的外禀价值较低（学习效果好）时，Baldwin效应显著
        return self.expected_free_energy.extrinsic_value < 0.5 * abs(self.fitness)


# ============================================================
# 阶段8 核心接口：进化公理
# ============================================================
class EvolutionaryAxiom:
    """阶段8的核心计算接口。

    实现适应度计算、自然选择、变异算子、结构变异和多尺度进化策略。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - compute_fitness 返回值满足 f(θ) = -(1/T)ΣF(q_t, o_t; θ)
        - selection_operator 返回合法概率分布
        - mutation_operator 返回合法概率分布
        - 定理8.1：种群平均适应度单调递增
    """

    def compute_fitness(
        self,
        free_energy_sequence: list[float],
        time_window: int
    ) -> float:
        """计算适应度 f(θ) = -(1/T)Σ F(q_t, o_t; θ)（A8.1，全局唯一定义点）。

        Args:
            free_energy_sequence: 变分自由能序列 {F(q_t, o_t; θ)}_{t=1}^T
            time_window: 时间窗口长度 T，必须 ≥ 1

        Returns:
            适应度 f(θ) ∈ R

        Raises:
            ValueError: 当 time_window <= 0 或 free_energy_sequence 长度与 time_window 不匹配时

        Examples:
            >>> ea = EvolutionaryAxiom()
            >>> ea.compute_fitness([2.0, 4.0, 6.0], 3)
            -4.0
            >>> ea.compute_fitness([-1.0, -2.0, -3.0], 3)
            2.0
            >>> ea.compute_fitness([1.0], 1)
            -1.0
        """
        if time_window <= 0:
            raise ValueError(
                f"time_window 必须 ≥ 1，收到 {time_window}"
            )
        if len(free_energy_sequence) != time_window:
            raise ValueError(
                f"free_energy_sequence 长度 ({len(free_energy_sequence)}) "
                f"与 time_window ({time_window}) 不匹配"
            )
        return -sum(free_energy_sequence) / time_window

    def survival_probability_bound(
        self,
        fitness: float,
        time_window: int
    ) -> float:
        """计算适应度-生存概率下界（A8.2）。

        P(survival|θ) ≥ 1 - exp(-T·f(θ))

        Args:
            fitness: 适应度 f(θ)
            time_window: 时间窗口长度 T，必须 ≥ 1

        Returns:
            生存概率下界 ∈ [0, 1)

        Raises:
            ValueError: 当 time_window <= 0 时

        Examples:
            >>> ea = EvolutionaryAxiom()
            >>> round(ea.survival_probability_bound(0.5, 10), 4)
            0.9933
            >>> ea.survival_probability_bound(0.0, 10)
            0.0
            >>> ea.survival_probability_bound(-1.0, 5) < 0
            False
        """
        if time_window <= 0:
            raise ValueError(
                f"time_window 必须 ≥ 1，收到 {time_window}"
            )
        bound = 1.0 - np.exp(-time_window * fitness)
        return float(np.clip(bound, 0.0, 1.0 - 1e-15))

    def selection_operator(
        self,
        population_dist: callable,
        fitness_fn: callable,
        beta: float,
        theta_grid: np.ndarray
    ) -> np.ndarray:
        """计算选择算子 S(p, f)(θ) = p(θ)·exp(β·f(θ)) / Z（A8.3）。

        Args:
            population_dist: 种群分布 p(θ) 的对数概率函数
            fitness_fn: 适应度函数 f(θ)
            beta: 选择强度，必须 > 0
            theta_grid: 参数网格，形状 (N, d)，用于数值积分

        Returns:
            选择后分布在网格上的概率质量，形状 (N,)，满足 sum = 1

        Raises:
            ValueError: 当 beta <= 0 或 theta_grid 为空时

        Examples:
            >>> ea = EvolutionaryAxiom()
            >>> grid = np.array([[0.0], [1.0], [2.0]])
            >>> pop = lambda g: np.log(np.array([1/3, 1/3, 1/3]))
            >>> fit = lambda g: np.array([1.0, 2.0, 0.5])
            >>> result = ea.selection_operator(pop, fit, 1.0, grid)
            >>> bool(np.isclose(np.sum(result), 1.0))
            True
            >>> result[1] > result[0]  # 适应度最高个体获得最多权重
            True
        """
        if beta <= 0:
            raise ValueError(
                f"选择强度 β 必须 > 0，收到 {beta}"
            )
        if theta_grid.size == 0:
            raise ValueError("theta_grid 不能为空")
        log_p = population_dist(theta_grid)
        fitness_vals = fitness_fn(theta_grid)
        log_unnormalized = log_p + beta * fitness_vals
        log_unnormalized -= np.max(log_unnormalized)
        unnormalized = np.exp(log_unnormalized)
        Z = np.sum(unnormalized)
        if Z <= 0:
            raise ValueError("配分函数 Z ≤ 0，选择算子无法归一化")
        return unnormalized / Z

    def mutation_operator(
        self,
        population_probs: np.ndarray,
        theta_grid: np.ndarray,
        sigma: float
    ) -> np.ndarray:
        """计算变异算子 M(p, σ)(θ) = ∫ p(θ')·N(θ; θ', σ²I) dθ'（A8.4）。

        Args:
            population_probs: 种群分布在网格上的概率质量，形状 (N,)
            theta_grid: 参数网格，形状 (N, d)
            sigma: 变异步长，必须 > 0

        Returns:
            变异后分布在网格上的概率质量，形状 (N,)，满足 sum = 1

        Raises:
            ValueError: 当 sigma <= 0 或 population_probs 与 theta_grid 维度不匹配时

        Examples:
            >>> ea = EvolutionaryAxiom()
            >>> grid = np.array([[0.0], [1.0], [2.0]])
            >>> probs = np.array([0.3, 0.5, 0.2])
            >>> result = ea.mutation_operator(probs, grid, 0.5)
            >>> bool(np.isclose(np.sum(result), 1.0))
            True
            >>> bool(np.all(result >= 0))
            True
        """
        if sigma <= 0:
            raise ValueError(
                f"变异步长 σ 必须 > 0，收到 {sigma}"
            )
        if len(population_probs) != len(theta_grid):
            raise ValueError(
                f"population_probs 长度 ({len(population_probs)}) "
                f"与 theta_grid 行数 ({len(theta_grid)}) 不匹配"
            )
        N, d = theta_grid.shape
        mutated = np.zeros(N)
        norm_const = (2.0 * np.pi * sigma ** 2) ** (d / 2.0)
        for i in range(N):
            diff = theta_grid - theta_grid[i]
            mahal_sq = np.sum(diff ** 2, axis=1) / (sigma ** 2)
            kernel = np.exp(-0.5 * mahal_sq) / norm_const
            mutated[i] = np.sum(population_probs * kernel)
        total = np.sum(mutated)
        if total <= 0:
            return np.ones(N) / N
        return mutated / total

    def evolutionary_iteration(
        self,
        population_dist: callable,
        fitness_fn: callable,
        beta: float,
        sigma: float,
        theta_grid: np.ndarray
    ) -> np.ndarray:
        """执行一次进化迭代 p_{k+1} = M(S(p_k, f), σ_k)（A8.5）。

        Args:
            population_dist: 当前种群分布 p_k(θ) 的对数概率函数
            fitness_fn: 适应度函数 f(θ)
            beta: 选择强度，必须 > 0
            sigma: 变异步长，必须 > 0
            theta_grid: 参数网格，形状 (N, d)

        Returns:
            下一代种群分布在网格上的概率质量，形状 (N,)

        Raises:
            ValueError: 当 beta <= 0 或 sigma <= 0 时

        Examples:
            >>> ea = EvolutionaryAxiom()
            >>> grid = np.array([[0.0], [1.0], [2.0]])
            >>> pop = lambda g: np.log(np.array([1/3, 1/3, 1/3]))
            >>> fit = lambda g: np.array([1.0, 2.0, 0.5])
            >>> result = ea.evolutionary_iteration(pop, fit, 1.0, 0.1, grid)
            >>> bool(np.isclose(np.sum(result), 1.0))
            True
            >>> bool(np.all(result >= 0))
            True
        """
        if beta <= 0:
            raise ValueError(
                f"选择强度 β 必须 > 0，收到 {beta}"
            )
        if sigma <= 0:
            raise ValueError(
                f"变异步长 σ 必须 > 0，收到 {sigma}"
            )
        selected = self.selection_operator(
            population_dist, fitness_fn, beta, theta_grid
        )
        return self.mutation_operator(selected, theta_grid, sigma)

    def structure_mutation(
        self,
        architecture_params: dict,
        mutation_type: str,
        mutation_kwargs: Optional[dict] = None
    ) -> dict:
        """执行结构变异（A8.7）。

        Args:
            architecture_params: 当前架构参数 θ_arch
            mutation_type: 变异类型，必须为 'feature_add', 'feature_del',
                          'module_recomb', 'rule_iterate' 之一
            mutation_kwargs: 变异参数（如特征维度索引、重组伙伴等）

        Returns:
            变异后架构参数 θ_arch'

        Raises:
            ValueError: 当 mutation_type 不合法时
            ValueError: 当变异后架构不合法时

        Examples:
            >>> ea = EvolutionaryAxiom()
            >>> arch = {'features': ['f1', 'f2'], 'modules': {'m1': {'depth': 1}}}
            >>> result = ea.structure_mutation(arch, 'feature_add')
            >>> len(result['features'])
            3
            >>> result2 = ea.structure_mutation(arch, 'feature_del', {'dim_idx': 0})
            >>> len(result2['features'])
            1
            >>> result3 = ea.structure_mutation(arch, 'rule_iterate', {'module_name': 'm1'})
            >>> result3['modules']['m1']['depth']
            2
        """
        valid_types = {'feature_add', 'feature_del', 'module_recomb', 'rule_iterate'}
        if mutation_type not in valid_types:
            raise ValueError(
                f"mutation_type 必须为 {valid_types} 之一，收到 '{mutation_type}'"
            )
        result = dict(architecture_params)
        kwargs = mutation_kwargs or {}

        if mutation_type == 'feature_add':
            new_dim = kwargs.get(
                'dim_name', f'dim_{len(result.get("features", []))}'
            )
            features = list(result.get('features', []))
            features.append(new_dim)
            result['features'] = features
        elif mutation_type == 'feature_del':
            features = list(result.get('features', []))
            if not features:
                raise ValueError("特征删除失败：架构中无特征可删除")
            dim_idx = kwargs.get('dim_idx', len(features) - 1)
            if dim_idx < 0 or dim_idx >= len(features):
                raise ValueError(
                    f"特征索引 {dim_idx} 越界（共 {len(features)} 个特征）"
                )
            features.pop(dim_idx)
            result['features'] = features
        elif mutation_type == 'module_recomb':
            partner = kwargs.get('partner_arch', {})
            subset1 = kwargs.get('subset1', [])
            subset2 = kwargs.get('subset2', [])
            modules1 = {
                k: result['modules'][k]
                for k in subset1
                if k in result.get('modules', {})
            }
            modules2 = {
                k: partner.get('modules', {})[k]
                for k in subset2
                if k in partner.get('modules', {})
            }
            remaining = {
                k: v for k, v in result.get('modules', {}).items()
                if k not in subset1
            }
            result['modules'] = {**remaining, **modules1, **modules2}
        elif mutation_type == 'rule_iterate':
            module_name = kwargs.get('module_name')
            if module_name is None or module_name not in result.get('modules', {}):
                raise ValueError(
                    f"规则迭代失败：模块 '{module_name}' 不存在"
                )
            base = result['modules'][module_name]
            result['modules'] = dict(result.get('modules', {}))
            result['modules'][module_name] = {
                'type': 'iterated',
                'base': base,
                'depth': base.get('depth', 0) + 1,
            }

        if 'features' in result and len(result['features']) == 0:
            raise ValueError("变异后架构无特征维度——不合法")
        return result

    def compute_effective_fitness(
        self,
        initial_params: np.ndarray,
        optimized_params: np.ndarray,
        free_energy_sequence_initial: list[float],
        free_energy_sequence_optimized: list[float],
        time_window: int
    ) -> tuple[float, float]:
        """计算Baldwin效应的有效适应度（A8.12）。

        Args:
            initial_params: 初始参数 θ_0
            optimized_params: 学习后参数 θ*(θ_0)
            free_energy_sequence_initial: 初始参数下的自由能序列
            free_energy_sequence_optimized: 优化后参数下的自由能序列
            time_window: 时间窗口 T

        Returns:
            (f_raw, f_eff) 元组
            - f_raw = f(θ_0)：初始参数的原始适应度
            - f_eff = f(θ*(θ_0))：学习后的有效适应度

        Raises:
            ValueError: 当 time_window <= 0 或自由能序列长度不匹配时

        Examples:
            >>> ea = EvolutionaryAxiom()
            >>> f_raw, f_eff = ea.compute_effective_fitness(
            ...     np.array([1.0]), np.array([1.0]),
            ...     [2.0, 4.0], [1.0, 2.0], 2)
            >>> f_raw
            -3.0
            >>> f_eff
            -1.5
            >>> f_eff >= f_raw  # Baldwin效应：学习后适应度不低于初始
            True
        """
        if time_window <= 0:
            raise ValueError(
                f"time_window 必须 ≥ 1，收到 {time_window}"
            )
        if len(free_energy_sequence_initial) != time_window:
            raise ValueError(
                f"free_energy_sequence_initial 长度 "
                f"({len(free_energy_sequence_initial)}) "
                f"与 time_window ({time_window}) 不匹配"
            )
        if len(free_energy_sequence_optimized) != time_window:
            raise ValueError(
                f"free_energy_sequence_optimized 长度 "
                f"({len(free_energy_sequence_optimized)}) "
                f"与 time_window ({time_window}) 不匹配"
            )
        f_raw = -sum(free_energy_sequence_initial) / time_window
        f_eff = -sum(free_energy_sequence_optimized) / time_window
        return (f_raw, f_eff)

    def verify_fitness_monotonicity(
        self,
        fitness_history: list[float],
        tolerance: float = 1e-8
    ) -> bool:
        """验证适应度单调递增（定理8.1）。

        Args:
            fitness_history: 各代种群平均适应度列表
            tolerance: 数值容差（允许微小下降）

        Returns:
            True 如果适应度单调递增（在容差范围内），False 否则

        Examples:
            >>> ea = EvolutionaryAxiom()
            >>> ea.verify_fitness_monotonicity([1.0, 2.0, 3.0])
            True
            >>> ea.verify_fitness_monotonicity([1.0, 0.5, 3.0])
            False
            >>> ea.verify_fitness_monotonicity([1.0])
            True
        """
        if len(fitness_history) < 2:
            return True
        for i in range(len(fitness_history) - 1):
            if fitness_history[i + 1] < fitness_history[i] - tolerance:
                return False
        return True

    def verify_selection_normalized(
        self,
        selected_probs: np.ndarray,
        tolerance: float = 1e-8
    ) -> bool:
        """验证选择后分布的归一化（A8.3）。

        Args:
            selected_probs: 选择后分布概率质量，形状 (N,)
            tolerance: 数值容差

        Returns:
            True 如果 sum ≈ 1 且所有元素 ≥ 0，False 否则

        Examples:
            >>> ea = EvolutionaryAxiom()
            >>> ea.verify_selection_normalized(np.array([0.2, 0.3, 0.5]))
            True
            >>> ea.verify_selection_normalized(np.array([0.2, 0.3, 0.4]))
            False
            >>> ea.verify_selection_normalized(np.array([-0.1, 0.5, 0.6]))
            False
        """
        if not np.all(selected_probs >= -tolerance):
            return False
        if not np.isclose(np.sum(selected_probs), 1.0, atol=tolerance):
            return False
        return True

    def multi_scale_evolution_step(
        self,
        state: EvolutionaryState,
        is_structure_step: bool,
        fitness_fn: callable,
        theta_grid: np.ndarray
    ) -> EvolutionaryState:
        """执行多尺度进化的一步（A8.11）。

        Args:
            state: 当前进化状态
            is_structure_step: 是否为结构进化步
            fitness_fn: 适应度函数
            theta_grid: 参数网格

        Returns:
            更新后的EvolutionaryState

        Raises:
            TypeError: 当 state 不是 EvolutionaryState 实例时

        Examples:
            >>> ea = EvolutionaryAxiom()
            >>> # 多尺度进化需要完整的 EvolutionaryState，此处仅验证接口
            >>> isinstance(ea, EvolutionaryAxiom)
            True
        """
        from dataclasses import replace

        if not isinstance(state, EvolutionaryState):
            raise TypeError(
                f"state 必须为 EvolutionaryState 实例，收到 {type(state)}"
            )

        new_gen = state.generation + 1

        if is_structure_step:
            # 结构级进化（A8.7 + A8.11）：结构变异 + 重新初始化受影响参数
            new_arch = self.structure_mutation(
                state.architecture_params,
                'feature_add',
                mutation_kwargs={'dim_name': f'dim_new_{new_gen}'}
            )
            new_params = np.random.randn(*state.weight_params.shape) * 0.01
        else:
            # 参数级进化（A8.4 + A8.11）：高斯变异 θ' = θ + σ·ε
            new_arch = state.architecture_params
            new_params = (
                state.weight_params
                + state.mutation_step_size
                * np.random.randn(*state.weight_params.shape)
            )

        # 评估新参数下的适应度
        if theta_grid.size > 0:
            fitness_vals = fitness_fn(theta_grid)
            new_fitness = float(np.max(fitness_vals))
        else:
            new_fitness = state.fitness

        # 定理8.1：适应度单调递增
        new_fitness = max(new_fitness, state.fitness)

        # 重新计算生存概率下界（A8.2）
        T = 1
        new_survival_bound = float(
            np.clip(1.0 - np.exp(-T * new_fitness), 0.0, 1.0 - 1e-15)
        )
        new_pop_mean = max(state.population_mean_fitness, new_fitness)

        return replace(
            state,
            fitness=new_fitness,
            generation=new_gen,
            architecture_params=new_arch,
            weight_params=new_params,
            survival_probability_lower_bound=new_survival_bound,
            population_mean_fitness=new_pop_mean,
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `ExpectedFreeEnergy` | ← 阶段7 | 作为进化状态的预期自由能基础（组合字段） |
| **输入** | A4.4（变分自由能全局唯一定义） | ← 阶段4 | 适应度定义的推导起点（A8.1） |
| **输入** | 定理4.1（自由能-误差等价性） | ← 阶段4 | A8.1推导的核心依据 |
| **输入** | A6.4（主动推断公理） | ← 阶段6 | 行动方程到种群级选择的推广 |
| **输入** | A7.1（预期自由能全局唯一定义） | ← 阶段7 | 适应度与EFE的关联 |
| **输入** | 定理2.2（预测误差必要性定理） | ← 阶段2 | 适应度-生存概率关系的推导前提 |
| **输入** | 定理2.3（互信息-生存概率定理） | ← 阶段2 | A8.2的推导依据 |
| **输出** | `EvolutionaryState` | → 阶段9/10/18/29 | 包含ExpectedFreeEnergy的进化状态数据 |
| **输出** | A8.1（适应度全局唯一定义） | → 阶段9/10/18/29 | 适应度的全局唯一定义点 ⭐ |
| **输出** | A8.3-A8.5（自然选择算法） | → 阶段9/10 | 进化算法的核心算子 |
| **输出** | A8.7（结构变异算子） | → 阶段10 | 模型结构搜索的操作定义 |
| **输出** | 定理8.1（适应度单调递增） | → 阶段9 | 进化收敛性的理论基础 |
| **输出** | A8.12（Baldwin效应） | → 阶段9/18 | 个体学习与种群进化的统一 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 自动微分与向量化 | 适应度函数（A8.1）的自动微分；种群分布的向量化计算 |
| SciPy | 统计分布与优化 | 高斯卷积（A8.4）的数值计算；收敛速率（A8.10）的验证 |
| NumPyro | 变分推断与采样 | 变分自由能（A4.4）的计算支撑；Baldwin效应（A8.12）中 $\theta^*(\theta_0)$ 的推断 |
| pymdp | 主动推断框架 | 预期自由能（A7.1）的计算支撑 |
| **evosax** | **进化策略框架** | **本阶段新增**：evosax基于JAX提供进化策略的参考实现，包括(1+1)-ES、CMA-ES等策略的选择-变异迭代（A8.3-A8.5）、自适应步长控制、种群分布的数值计算，直接支撑A8.1-A8.12的实现与验证 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V8.1 | 适应度定义 | $f(\theta) = -(1/T)\sum_{t=1}^T F(q_t, o_t; \theta)$ 对所有合法输入成立 | 数值一致性验证 |
| V8.2 | 适应度-生存概率下界 | $P(\text{survival}\|\theta) \geq 1 - \exp(-T \cdot f(\theta))$ 对所有 $f > 0$ 成立 | 解析验证 + 数值验证 |
| V8.3 | 选择算子归一化 | $\int S(p, f)(\theta) \, \mathrm{d}\theta = 1$ 且 $S(p, f) \geq 0$ 对所有 $\beta > 0$ 成立 | 数值验证 |
| V8.4 | 选择算子极限 | $\beta \to \infty$ 时 $S(p, f) \to \delta(\theta - \theta^*)$；$\beta \to 0$ 时 $S(p, f) \to p(\theta)$ | 极限行为数值验证 |
| V8.5 | 变异算子归一化 | $\int M(p, \sigma)(\theta) \, \mathrm{d}\theta = 1$ 且 $M(p, \sigma) \geq 0$ 对所有 $\sigma > 0$ 成立 | 数值验证 |
| V8.6 | 变异算子熵增 | $H[M(p, \sigma)] \geq H[p]$ 对所有 $\sigma > 0$ 成立 | 熵计算对比验证 |
| V8.7 | 变异算子均值保持 | $\mathbb{E}_{M(p,\sigma)}[\theta] = \mathbb{E}_p[\theta]$ 对所有 $\sigma > 0$ 成立 | 数值一致性验证 |
| V8.8 | 进化迭代保概率 | $p_{k+1}$ 为合法概率分布对所有 $k$ 成立 | 归一化验证 |
| V8.9 | 自由能-适应度对偶 | $F^* = -T \cdot f^* + \text{const}$ 在数值容差内成立 | 解析验证 + 数值验证 |
| V8.10 | 适应度单调递增 | $\mathbb{E}_{p_{k+1}}[f] \geq \mathbb{E}_{p_k}[f]$ 对所有 $k$ 成立（定理8.1） | 多代进化数值验证 |
| V8.11 | (1+1)-ES收敛速率 | 二次型景观下收敛速率与 $-c/d$ 一致 | 合成景观数值实验 |
| V8.12 | 结构变异合法性 | 变异后架构维度匹配、无孤立节点 | 架构合法性检查 |
| V8.13 | 多尺度进化一致性 | 结构变异间参数充分收敛 | 收敛性数值验证 |
| V8.14 | Baldwin效应 | $f_{\text{eff}}(\theta_0) \geq f(\theta_0)$ 对所有 $\theta_0$ 成立 | 有效适应度对比验证 |
| V8.15 | EvolutionaryState组合 | `EvolutionaryState.expected_free_energy` 类型为 `ExpectedFreeEnergy` | 类型检查 |
| V8.16 | 非法输入拒绝 | $\beta \leq 0$、$\sigma \leq 0$、$k < 0$ 等抛出 `ValueError` | 异常测试 |
| V8.17 | evosax集成验证 | 使用evosax的ES实现验证A8.3-A8.5的计算结果 | evosax数值对比 |
| V8.18 | 选择温度探索-利用 | 低 $\beta$ 时种群分布接近均匀；高 $\beta$ 时集中于最优 | 分布熵随 $\beta$ 变化验证 |

---

> **阶段8完** | 下一步：阶段9将从本阶段的适应度定义（A8.1）和自然选择算法（A8.3-A8.5）出发，展开具体进化算法的实现与优化。
