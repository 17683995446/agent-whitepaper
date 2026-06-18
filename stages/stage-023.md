# 阶段23：能效驱力Reward_eff——从计算资源到信息效率

> **阶段编号**：S23 | **所属层**：第三层·驱力奖励层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 能效驱力 | "省着花钱"——系统对计算资源高效利用的内在追求 |
> | CPU占用率 | "大脑用了多少算力"——Agent占用总计算资源的比例 |
> | 内存占用率 | "大脑占了多少空间"——Agent占用总内存资源的比例 |
> | 推理耗时 | "想问题花了多少时间"——推理时间与预算的比值 |
> | 计算冗余度 | "浪费了多少算力"——信息产出与计算投入的比值补数 |
> | 信息/计算比 | "每块钱买了多少东西"——单位计算量产生的信息量 |

---

## 1. 第一性原理依据

本阶段从阶段1（生存条件）和阶段5（复杂性项）的核心结论出发，建立能效驱力的完整数学定义。核心依据为：

- **A1.8（生存条件）**（引用阶段1）：$\dot{S}_{\text{exchange}} \leq -\dot{S}_{\text{internal}}$，系统维持有序状态需要持续从环境获取负熵。获取负熵需要能量和计算资源——能效驱力确保系统以最少的资源消耗满足生存需求。
- **A5.5（复杂性项）**（引用阶段5）：$C(q) = D_{\text{KL}}[q(\psi) \| p(\psi)]$，复杂性项惩罚偏离先验的表征。计算冗余度与复杂性项直接相关——过多的计算资源用于维持偏离先验的表征，本质上是一种"计算浪费"。
- **A4.4（变分自由能全局唯一定义）**（引用阶段4）：$F[q] = -\mathbb{E}_q[\ln p(o|\psi)] + D_{\text{KL}}[q(\psi) \| p(\psi)]$，自由能框架中的复杂性项为能效驱力提供了信息论基础。

本阶段的核心目标是：**定义能效驱力 $R_{\text{eff}}$，将系统运行过程中的资源消耗分解为CPU占用率、内存占用率、推理耗时和计算冗余度四个分量，建立能效驱力与复杂性项的定量关系，证明能效驱力是复杂性项在资源层面的投影。** 能效驱力是三大驱力中确保系统"可持续运行"的关键——它对应阶段1的生存条件（A1.8）中的资源约束，是"省着活"这一效率需求的数学表达。

---

## 2. 核心数学公式

### 第一部分：能效驱力的定义

#### 2.1 能效驱力定义（A23.1）

能效驱力定义为资源消耗的加权和，取负值以符合"驱力越大越需要节约"的约定：

$$R_{\text{eff}} = -\alpha_e \cdot (\text{cpu\_ratio} + \text{mem\_ratio} + \text{time\_cost} + \text{redundancy}) \tag{A23.1}$$

其中：
- $\alpha_e > 0$：能效驱力的全局缩放系数
- $\text{cpu\_ratio} \in [0, 1]$：CPU占用率（A23.2）
- $\text{mem\_ratio} \in [0, 1]$：内存占用率（A23.3）
- $\text{time\_cost} \in [0, \infty)$：推理耗时比（A23.4）
- $\text{redundancy} \in [0, 1]$：计算冗余度（A23.5）

- **输入**：CPU占用率、内存占用率、推理耗时比、计算冗余度，缩放系数 $\alpha_e$
- **输出**：$R_{\text{eff}} \in \mathbb{R}_{\leq 0}$（能效驱力恒非正，资源消耗越多驱力越负）
- **约束**：$\alpha_e > 0$；$\text{cpu\_ratio} \in [0, 1]$；$\text{mem\_ratio} \in [0, 1]$；$\text{redundancy} \in [0, 1]$

> **白话翻译**：能效驱力 = "省着花钱的冲动"——CPU用多了、内存占多了、时间花多了、算力浪费了，能效驱力就越负，驱动系统节约资源。$\alpha_e$ 是"节俭程度"——调大则更注重效率，调小则更不在乎资源消耗。

**能效驱力与安全驱力的区别**：安全驱力关注"是否安全"（异常信号），能效驱力关注"是否高效"（资源消耗）。两者可以同时为负——系统可能既不安全（异常多）又低效（资源浪费），需要同时应对两种压力。

**能效驱力与收益驱力的张力**：收益驱力鼓励探索和学习（消耗资源），能效驱力鼓励节约（减少资源消耗）。两者构成"求新-求省"的基本张力——系统需要在"多学习"和"省资源"之间找到平衡。

---

### 第二部分：CPU占用率

#### 2.2 CPU占用率（A23.2）

CPU占用率度量Agent占用总计算资源的比例：

$$\text{cpu\_ratio} = \frac{\text{cpu}_{\text{agent}}}{\text{cpu}_{\text{total}}} \tag{A23.2}$$

其中：
- $\text{cpu}_{\text{agent}}$：Agent当前使用的CPU资源量（如CPU时间片、FLOPS占用）
- $\text{cpu}_{\text{total}}$：系统可用的总CPU资源量

- **输入**：Agent CPU使用量 $\text{cpu}_{\text{agent}} \in \mathbb{R}_{\geq 0}$，总CPU资源 $\text{cpu}_{\text{total}} \in \mathbb{R}_{> 0}$
- **输出**：$\text{cpu\_ratio} \in [0, 1]$（当Agent独占所有CPU时为1）
- **约束**：$\text{cpu}_{\text{agent}} \leq \text{cpu}_{\text{total}}$

> **白话翻译**：CPU占用率就是"大脑用了多少算力"——如果系统有100个计算单元，Agent用了30个，占用率就是0.3。占用率越高，留给其他任务的算力越少。

**CPU占用率的分层分解**：

在预测编码框架下，CPU占用可按层级分解：

$$\text{cpu}_{\text{agent}} = \sum_{l=0}^{L} \text{cpu}_l = \sum_{l=0}^{L} N_{\text{perceive}} \cdot O(n_{l-1} \cdot n_l) + O(n_{l-1} \cdot n_l) \tag{A23.2-decompose}$$

其中第一项为感知迭代的计算量（引用阶段19，A19.4），第二项为权重更新的计算量（引用阶段19，A19.5）。感知迭代主导计算量（$N_{\text{perceive}} \approx 100$，引用阶段17，A17.3）。

**CPU占用率与生存条件的关联**：

由A1.8（引用阶段1），系统生存需要持续获取负熵。获取负熵需要执行行动，执行行动需要计算资源。当CPU占用率过高时，系统可能无法及时响应环境变化，导致生存条件违反。具体地：

$$\text{cpu\_ratio} > \theta_{\text{cpu}} \implies \text{响应延迟} \implies \dot{S}_{\text{exchange}} > -\dot{S}_{\text{internal}} \tag{A23.2-survival}$$

其中 $\theta_{\text{cpu}}$ 为CPU占用率的安全阈值。超过阈值时，系统无法在时间约束内完成感知-行动循环，导致负熵获取不足。

---

### 第三部分：内存占用率

#### 2.3 内存占用率（A23.3）

内存占用率度量Agent占用总内存资源的比例：

$$\text{mem\_ratio} = \frac{\text{mem}_{\text{agent}}}{\text{mem}_{\text{total}}} \tag{A23.3}$$

其中：
- $\text{mem}_{\text{agent}}$：Agent当前使用的内存量
- $\text{mem}_{\text{total}}$：系统可用的总内存量

- **输入**：Agent内存使用量 $\text{mem}_{\text{agent}} \in \mathbb{R}_{\geq 0}$，总内存 $\text{mem}_{\text{total}} \in \mathbb{R}_{> 0}$
- **输出**：$\text{mem\_ratio} \in [0, 1]$
- **约束**：$\text{mem}_{\text{agent}} \leq \text{mem}_{\text{total}}$

> **白话翻译**：内存占用率就是"大脑占了多少空间"——如果系统有1GB内存，Agent用了300MB，占用率就是0.3。占用率越高，留给其他任务的空间越少。

**内存占用的分层分解**：

在预测编码框架下，内存占用可按层级分解：

$$\text{mem}_{\text{agent}} = \sum_{l=0}^{L} (n_l \cdot \text{size\_repr} + n_{l-1} \cdot n_l \cdot \text{size\_weight} + n_l^2 \cdot \text{size\_precision}) \tag{A23.3-decompose}$$

其中：
- $n_l \cdot \text{size\_repr}$：第 $l$ 层表征的内存（引用阶段3，A3.1）
- $n_{l-1} \cdot n_l \cdot \text{size\_weight}$：第 $l$ 层权重的内存（引用阶段11，A11.3）
- $n_l^2 \cdot \text{size\_precision}$：第 $l$ 层精度矩阵的内存（引用阶段3，A3.2）

精度矩阵是内存消耗的主要来源——$n_l^2$ 的空间复杂度使得高维层的精度矩阵占用大量内存。使用对角精度矩阵（$\Pi_l = \text{diag}(\pi_l)$）可将内存降至 $O(n_l)$，但牺牲了维度间的相关性信息。

**内存占用率与OOM风险**：

当 $\text{mem\_ratio}$ 接近1时，系统面临内存溢出（OOM）风险。OOM直接导致系统崩溃，违反生存条件（A1.8）。因此内存占用率是安全驱力和能效驱力的交叉指标——高内存占用既降低能效（$R_{\text{eff}}$ 变负），又增加风险（$R_{\text{safe}}$ 变负）。

---

### 第四部分：推理耗时

#### 2.4 推理耗时（A23.4）

推理耗时度量Agent完成一次感知-行动循环所需时间与预算的比值：

$$\text{time\_cost} = \frac{t_{\text{inference}}}{t_{\text{budget}}} \tag{A23.4}$$

其中：
- $t_{\text{inference}}$：实际推理时间（从接收观测到输出行动的时间）
- $t_{\text{budget}}$：推理时间预算（由环境的时间约束决定）

- **输入**：实际推理时间 $t_{\text{inference}} \in \mathbb{R}_{> 0}$，时间预算 $t_{\text{budget}} \in \mathbb{R}_{> 0}$
- **输出**：$\text{time\_cost} \in \mathbb{R}_{> 0}$（可超过1，表示超时）
- **约束**：$t_{\text{budget}} > 0$

> **白话翻译**：推理耗时就是"想问题花了多少时间"——如果环境要求100ms内做出反应，Agent花了150ms，耗时比就是1.5，意味着超时了。超时可能导致错过关键时机，直接影响生存。

**推理耗时的分层分解**：

$$t_{\text{inference}} = t_{\text{perceive}} + t_{\text{decide}} + t_{\text{act}} \tag{A23.4-decompose}$$

其中：
- $t_{\text{perceive}}$：感知推断时间（A19.4的感知迭代）
- $t_{\text{decide}}$：策略决策时间（A7.8的策略选择）
- $t_{\text{act}}$：行动执行时间

感知推断时间主导推理耗时：

$$t_{\text{perceive}} = N_{\text{perceive}} \cdot \sum_{l=0}^{L} O(n_{l-1} \cdot n_l) / \text{FLOPS}$$

**推理耗时与时间预算的关系**：

| $\text{time\_cost}$ 值 | 含义 | 后果 |
|----------------------|------|------|
| $< 0.5$ | 充裕 | 系统有充足时间响应 |
| $0.5 - 1.0$ | 紧张 | 系统勉强按时响应 |
| $> 1.0$ | 超时 | 系统无法按时响应，生存风险 |

**推理耗时与快感知-慢学习的关联**：

由A17.3（引用阶段17），分离比 $\alpha_{\text{separation}} = \tau_W / \tau_r \approx 100$。感知迭代步数 $N_{\text{perceive}}$ 与分离比成正比——分离比越大，感知越需要更多步才能收敛，推理耗时越长。因此，能效驱力与快感知-慢学习的时间尺度分离存在张力：更大的分离比提高学习质量（更准确的感知），但增加推理耗时（更低的能效）。

**自适应感知步数**：

为平衡感知质量和推理耗时，引入自适应感知步数：

$$N_{\text{perceive}}^{\text{adaptive}} = \min\left(N_{\text{perceive}}^{\text{max}}, \left\lceil\frac{t_{\text{budget}} - t_{\text{decide}} - t_{\text{act}}}{t_{\text{per\_step}}}\right\rceil\right) \tag{A23.4-adaptive}$$

其中 $t_{\text{per\_step}}$ 为每步感知迭代的平均时间。自适应步数确保推理不超时，但可能牺牲感知质量（收敛不充分）。

---

### 第五部分：计算冗余度

#### 2.5 计算冗余度（A23.5）

计算冗余度度量计算资源的信息效率——单位计算量产生的信息量越低，冗余度越高：

$$\text{redundancy} = 1 - \frac{I(\Psi; O)}{C_{\text{computational}}} \tag{A23.5}$$

其中：
- $I(\Psi; O)$ 为表征与观测之间的互信息（引用阶段7，A7.5）
- $C_{\text{computational}}$ 为计算成本（如FLOPS、参数量）

- **输入**：互信息 $I(\Psi; O) \in \mathbb{R}_{\geq 0}$，计算成本 $C_{\text{computational}} \in \mathbb{R}_{> 0}$
- **输出**：$\text{redundancy} \in (-\infty, 1]$（通常在 $[0, 1]$ 范围内）
- **约束**：当 $I(\Psi; O) = C_{\text{computational}}$ 时，$\text{redundancy} = 0$（完美效率）；当 $I(\Psi; O) = 0$ 时，$\text{redundancy} = 1$（完全冗余）

> **白话翻译**：计算冗余度就是"浪费了多少算力"——如果花了100个计算单位，只产出了相当于30个计算单位的信息量，冗余度就是0.7，意味着70%的算力白费了。完美效率（冗余度=0）意味着每一点算力都转化为了信息。

**信息/计算比的物理含义**：

$$\eta_{\text{info}} = \frac{I(\Psi; O)}{C_{\text{computational}}} \tag{A23.5-efficiency}$$

$\eta_{\text{info}}$ 为信息效率——单位计算成本产生的信息量。$\eta_{\text{info}} = 1$ 为理论最大效率（每个计算操作产生1 nat信息），$\eta_{\text{info}} = 0$ 为零效率（所有计算都是冗余的）。

**计算冗余度与预测编码架构的关系**：

在预测编码架构中，计算冗余度与感知迭代步数直接相关。由A19.4（引用阶段19），感知迭代需要 $N_{\text{perceive}}$ 步，每步的计算量为 $O(\sum_l n_l^2)$。但信息增益主要集中在最初的几步——后续步骤的自由能下降量越来越小（由A4.11的指数收敛保证）。

定义第 $k$ 步的信息增量为 $\Delta I_k = I(\Psi_k; O) - I(\Psi_{k-1}; O)$，则信息效率随迭代步数递减：

$$\eta_{\text{info}}(k) = \frac{\Delta I_k}{\text{cost\_per\_step}} \propto \rho^k \tag{A23.5-decay}$$

其中 $\rho < 1$ 为收敛速率（引用阶段14，A14.3-stability）。因此，感知迭代的后半段是计算冗余的主要来源——信息增益指数衰减，但计算成本线性增加。

**最优感知步数**：

最优感知步数在信息增益与计算成本之间取平衡：

$$N_{\text{perceive}}^* = \arg\max_{N} \left[\sum_{k=1}^{N} \Delta I_k - \lambda_{\text{cost}} \cdot N \cdot \text{cost\_per\_step}\right] \tag{A23.5-optimal}$$

其中 $\lambda_{\text{cost}}$ 为计算成本的权重。由 $\Delta I_k \propto \rho^k$，最优步数的近似解为：

$$N_{\text{perceive}}^* \approx \frac{\ln(\lambda_{\text{cost}} \cdot \text{cost\_per\_step} / \Delta I_1)}{\ln\rho}$$

---

### 第六部分：能效驱力与复杂性项的关系

#### 2.6 能效驱力与复杂性项的关系（A23.6）

能效驱力与变分自由能中的复杂性项（引用阶段5，A5.5）存在深刻的对应关系：

$$R_{\text{eff}} \propto -C(q) \tag{A23.6}$$

即能效驱力与复杂性项的负值成正比。

- **输入**：复杂性项 $C(q) = D_{\text{KL}}[q(\psi) \| p(\psi)]$
- **输出**：$R_{\text{eff}} \propto -C(q)$
- **约束**：$C(q) \geq 0$（KL散度非负性）

> **白话翻译**：能效驱力就是复杂性项在资源层面的投影——复杂性项惩罚"偏离先验的表征"，偏离先验意味着需要更多计算资源来维持这些"非标准"的表征。越偏离先验，计算越昂贵，能效驱力越负。

**完整论证**：

**步骤1：复杂性项与参数量的关系**

由A5.6（引用阶段5），高斯先验下的复杂性项为：

$$C(q) = \frac{1}{2}(\mu - \mu_{\text{prior}})^T \Sigma_{\text{prior}}^{-1} (\mu - \mu_{\text{prior}}) + \frac{1}{2}\ln\frac{|\Sigma_{\text{prior}}|}{|\Sigma|} + \frac{1}{2}\text{tr}(\Sigma_{\text{prior}}^{-1}\Sigma) - \frac{d}{2}$$

复杂性项的每个分量都对应计算资源消耗：
- 均值偏离项 $R(\mu)$：偏离先验的参数需要更多计算来维护（EWC正则化的计算成本，引用阶段15，A15.5）
- 方差偏离项 $R(\Sigma)$：非标准方差需要额外的存储和计算

**步骤2：复杂性项与计算冗余度的关系**

计算冗余度（A23.5）度量"单位计算量产生的信息量"。当复杂性项高时，系统的变分分布 $q$ 严重偏离先验 $p$，意味着系统需要维护大量"非先验"的信念。这些信念的维护需要额外的计算资源，但不一定带来等比例的信息增益——这就是计算冗余的来源。

具体地，设信息增益 $I(\Psi; O)$ 与准确性项的改善成正比（由A5.3，引用阶段5），计算成本 $C_{\text{computational}}$ 与复杂性项 $C(q)$ 成正比（维护偏离先验的表征需要更多计算），则：

$$\text{redundancy} = 1 - \frac{I(\Psi; O)}{C_{\text{computational}}} \approx 1 - \frac{\Delta A}{\Delta C} \propto 1 - \frac{1}{1 + C(q)/A(q)}$$

当 $C(q)$ 相对于 $A(q)$ 较大时（复杂性主导），冗余度接近1——大量计算用于维护偏离先验的表征，但信息增益有限。

**步骤3：从复杂性项到能效驱力**

由A23.1，$R_{\text{eff}} = -\alpha_e \cdot (\text{cpu\_ratio} + \text{mem\_ratio} + \text{time\_cost} + \text{redundancy})$。复杂性项通过以下路径影响各分量：

1. **CPU占用率**：高复杂性 → 更多参数偏离先验 → 更多计算用于维护 → CPU占用率升高
2. **内存占用率**：高复杂性 → 更大协方差矩阵 → 更多内存用于存储 → 内存占用率升高
3. **推理耗时**：高复杂性 → 更多感知迭代步数 → 更长推理时间 → 推理耗时升高
4. **计算冗余度**：高复杂性 → 信息/计算比降低 → 冗余度升高

因此，$R_{\text{eff}}$ 的四个分量都与 $C(q)$ 正相关，综合起来 $R_{\text{eff}} \propto -C(q)$。$\blacksquare$

**能效驱力与自由能最小化的协同**：

自由能最小化（A4.4）同时降低准确性项和复杂性项。降低复杂性项直接提高能效驱力（$R_{\text{eff}}$ 变大），降低准确性项直接提高收益驱力中的精度提升（$\delta_{\text{accuracy}}$ 变大）。因此，自由能最小化同时优化收益驱力和能效驱力——这是自由能原理的又一个优美性质。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A23.1 | $R_{\text{eff}} = -\alpha_e \cdot (\text{cpu\_ratio} + \text{mem\_ratio} + \text{time\_cost} + \text{redundancy})$ | 四个资源指标, $\alpha_e$ | $R_{\text{eff}} \in \mathbb{R}_{\leq 0}$ | $\alpha_e > 0$ |
| A23.2 | $\text{cpu\_ratio} = \text{cpu}_{\text{agent}} / \text{cpu}_{\text{total}}$ | $\text{cpu}_{\text{agent}}$, $\text{cpu}_{\text{total}}$ | $\text{cpu\_ratio} \in [0, 1]$ | $\text{cpu}_{\text{agent}} \leq \text{cpu}_{\text{total}}$ |
| A23.3 | $\text{mem\_ratio} = \text{mem}_{\text{agent}} / \text{mem}_{\text{total}}$ | $\text{mem}_{\text{agent}}$, $\text{mem}_{\text{total}}$ | $\text{mem\_ratio} \in [0, 1]$ | $\text{mem}_{\text{agent}} \leq \text{mem}_{\text{total}}$ |
| A23.4 | $\text{time\_cost} = t_{\text{inference}} / t_{\text{budget}}$ | $t_{\text{inference}}$, $t_{\text{budget}}$ | $\text{time\_cost} \in \mathbb{R}_{> 0}$ | $t_{\text{budget}} > 0$ |
| A23.5 | $\text{redundancy} = 1 - I(\Psi; O) / C_{\text{computational}}$ | $I(\Psi; O)$, $C_{\text{computational}}$ | $\text{redundancy} \in (-\infty, 1]$ | 引用阶段7，A7.5 |
| A23.6 | $R_{\text{eff}} \propto -C(q)$ | $C(q)$ | $R_{\text{eff}}$ | 引用阶段5，A5.5 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# 引用阶段22的数据类
from stage_022 import GainDrive


# ============================================================
# 阶段23 数据类：能效驱力
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class EfficiencyDrive:
    """能效驱力的完整描述。

    组合阶段22的GainDrive，在收益驱力的基础上增加
    能效驱力的计算结果。

    Attributes:
        gain_drive: 阶段22的收益驱力（组合字段）
        alpha_e: 能效驱力全局缩放系数（A23.1）
        cpu_ratio: CPU占用率（A23.2）
        mem_ratio: 内存占用率（A23.3）
        time_cost: 推理耗时比（A23.4）
        redundancy: 计算冗余度（A23.5）
        raw_eff_reward: 原始能效驱力值（A23.1）
        mutual_information: 表征与观测的互信息（A23.5）
        computational_cost: 计算成本（A23.5）
        info_efficiency: 信息效率 η_info（A23.5-efficiency）

    Invariants:
        - alpha_e > 0
        - cpu_ratio ∈ [0, 1]
        - mem_ratio ∈ [0, 1]
        - time_cost > 0
        - redundancy ≤ 1
        - raw_eff_reward ≤ 0
    """
    gain_drive: GainDrive
    alpha_e: float
    cpu_ratio: float
    mem_ratio: float
    time_cost: float
    redundancy: float
    raw_eff_reward: float
    mutual_information: float
    computational_cost: float
    info_efficiency: float

    def __post_init__(self):
        # 验证缩放系数
        if self.alpha_e <= 0:
            raise ValueError(
                f"能效驱力缩放系数必须 > 0，"
                f"收到 {self.alpha_e}"
            )
        # 验证CPU占用率
        if not (0 <= self.cpu_ratio <= 1):
            raise ValueError(
                f"CPU占用率必须 ∈ [0, 1]，"
                f"收到 {self.cpu_ratio}"
            )
        # 验证内存占用率
        if not (0 <= self.mem_ratio <= 1):
            raise ValueError(
                f"内存占用率必须 ∈ [0, 1]，"
                f"收到 {self.mem_ratio}"
            )
        # 验证推理耗时
        if self.time_cost <= 0:
            raise ValueError(
                f"推理耗时比必须 > 0，"
                f"收到 {self.time_cost}"
            )
        # 验证冗余度上界
        if self.redundancy > 1 + 1e-10:
            raise ValueError(
                f"计算冗余度不应超过1，"
                f"收到 {self.redundancy}"
            )
        # 验证能效驱力非正
        if self.raw_eff_reward > 1e-10:
            raise ValueError(
                f"原始能效驱力必须 ≤ 0，"
                f"收到 {self.raw_eff_reward}"
            )
        # 验证信息效率
        if self.info_efficiency < -1e-10:
            raise ValueError(
                f"信息效率不应为负，"
                f"收到 {self.info_efficiency}"
            )

    @property
    def is_overtime(self) -> bool:
        """推理是否超时。"""
        return self.time_cost > 1.0

    @property
    def is_efficient(self) -> bool:
        """系统是否高效运行（冗余度低于阈值）。"""
        return self.redundancy < 0.5

    @property
    def resource_pressure(self) -> float:
        """资源压力综合指标。"""
        return self.cpu_ratio + self.mem_ratio + min(self.time_cost, 2.0) + self.redundancy

    @property
    def complexity_proxy(self) -> float:
        """复杂性项的能效驱力代理（A23.6）。"""
        return -self.raw_eff_reward / self.alpha_e


# ============================================================
# 阶段23 核心接口：能效驱力计算器
# ============================================================
class EfficiencyDriveCalculator:
    """阶段23的核心计算接口。

    实现能效驱力的完整计算流程，包括CPU/内存占用率、
    推理耗时、计算冗余度和信息效率。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - R_eff ≤ 0（能效驱力恒非正）
        - cpu_ratio ∈ [0, 1]
        - mem_ratio ∈ [0, 1]
        - redundancy ≤ 1
    """

    def compute_cpu_ratio(
        self,
        cpu_agent: float,
        cpu_total: float
    ) -> float:
        """计算CPU占用率（A23.2）。

        Args:
            cpu_agent: Agent CPU使用量（单位：CPU时间片或FLOPS），>= 0
            cpu_total: 总CPU资源（单位：CPU时间片或FLOPS），> 0

        Returns:
            cpu_ratio ∈ [0, 1]

        Raises:
            ValueError: 当 cpu_total <= 0
            ValueError: 当 cpu_agent < 0

        Examples:
            >>> calc = EfficiencyDriveCalculator()
            >>> calc.compute_cpu_ratio(30.0, 100.0)
            0.3
        """
        if cpu_total <= 0:
            raise ValueError(f"cpu_total 必须 > 0，收到 {cpu_total}")
        if cpu_agent < 0:
            raise ValueError(f"cpu_agent 必须 >= 0，收到 {cpu_agent}")
        return min(1.0, cpu_agent / cpu_total)

    def compute_mem_ratio(
        self,
        mem_agent: float,
        mem_total: float
    ) -> float:
        """计算内存占用率（A23.3）。

        Args:
            mem_agent: Agent内存使用量（单位：bytes），>= 0
            mem_total: 总内存（单位：bytes），> 0

        Returns:
            mem_ratio ∈ [0, 1]

        Raises:
            ValueError: 当 mem_total <= 0
            ValueError: 当 mem_agent < 0

        Examples:
            >>> calc = EfficiencyDriveCalculator()
            >>> calc.compute_mem_ratio(300.0, 1000.0)
            0.3
        """
        if mem_total <= 0:
            raise ValueError(f"mem_total 必须 > 0，收到 {mem_total}")
        if mem_agent < 0:
            raise ValueError(f"mem_agent 必须 >= 0，收到 {mem_agent}")
        return min(1.0, mem_agent / mem_total)

    def compute_time_cost(
        self,
        t_inference: float,
        t_budget: float
    ) -> float:
        """计算推理耗时比（A23.4）。

        Args:
            t_inference: 实际推理时间（单位：秒），> 0
            t_budget: 时间预算（单位：秒），> 0

        Returns:
            time_cost > 0，可超过1表示超时

        Raises:
            ValueError: 当 t_inference <= 0 或 t_budget <= 0

        Examples:
            >>> calc = EfficiencyDriveCalculator()
            >>> calc.compute_time_cost(0.15, 0.1)
            1.5
        """
        if t_inference <= 0:
            raise ValueError(f"t_inference 必须 > 0，收到 {t_inference}")
        if t_budget <= 0:
            raise ValueError(f"t_budget 必须 > 0，收到 {t_budget}")
        return t_inference / t_budget

    def compute_redundancy(
        self,
        mutual_information: float,
        computational_cost: float
    ) -> float:
        """计算计算冗余度（A23.5）。

        Args:
            mutual_information: I(Ψ; O)（单位：nats），>= 0
            computational_cost: C_computational（单位：FLOPS或计算量），> 0

        Returns:
            redundancy = 1 - I/C，通常 ∈ [0, 1]，当 I > C 时可为负

        Raises:
            ValueError: 当 computational_cost <= 0
            ValueError: 当 mutual_information < 0

        Examples:
            >>> calc = EfficiencyDriveCalculator()
            >>> calc.compute_redundancy(30.0, 100.0)
            0.7
        """
        if computational_cost <= 0:
            raise ValueError(f"computational_cost 必须 > 0，收到 {computational_cost}")
        if mutual_information < 0:
            raise ValueError(f"mutual_information 必须 >= 0，收到 {mutual_information}")
        return 1.0 - mutual_information / computational_cost

    def compute_eff_reward(
        self,
        cpu_ratio: float,
        mem_ratio: float,
        time_cost: float,
        redundancy: float,
        alpha_e: float
    ) -> float:
        """计算原始能效驱力（A23.1）。

        Args:
            cpu_ratio: CPU占用率，∈ [0, 1]
            mem_ratio: 内存占用率，∈ [0, 1]
            time_cost: 推理耗时比，> 0
            redundancy: 计算冗余度
            alpha_e: 缩放系数，> 0

        Returns:
            R_eff = -α_e · (cpu + mem + time + redundancy) <= 0

        Raises:
            ValueError: 当 alpha_e <= 0
            ValueError: 当 cpu_ratio 或 mem_ratio 不在 [0, 1]
            ValueError: 当 time_cost <= 0

        Examples:
            >>> calc = EfficiencyDriveCalculator()
            >>> calc.compute_eff_reward(0.3, 0.3, 1.0, 0.4, alpha_e=1.0)
            -2.0
        """
        if alpha_e <= 0:
            raise ValueError(f"alpha_e 必须 > 0，收到 {alpha_e}")
        if not (0 <= cpu_ratio <= 1):
            raise ValueError(f"cpu_ratio 必须 ∈ [0, 1]，收到 {cpu_ratio}")
        if not (0 <= mem_ratio <= 1):
            raise ValueError(f"mem_ratio 必须 ∈ [0, 1]，收到 {mem_ratio}")
        if time_cost <= 0:
            raise ValueError(f"time_cost 必须 > 0，收到 {time_cost}")
        total_cost = cpu_ratio + mem_ratio + time_cost + redundancy
        return -alpha_e * total_cost
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `GainDrive` | ← 阶段22 | 作为能效驱力的收益驱力组合字段 |
| **输入** | A1.8（生存条件） | ← 阶段1 | 能效驱力的生存约束基础 |
| **输入** | A5.5（复杂性项） | ← 阶段5 | 能效驱力与复杂性项的对应关系 |
| **输入** | A7.5（认知价值 = 互信息） | ← 阶段7 | 计算冗余度的信息论基础 |
| **输出** | `EfficiencyDrive` | → 阶段24 | 能效驱力作为全局奖励的能效分量 |
| **输出** | A23.6（能效驱力-复杂性关系） | → 阶段24 | 全局奖励与EFE等价性的证明基础 |
| **输出** | A23.5（计算冗余度） | → 阶段25 | 缺口驱动中的能效缺口度量 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 计算成本度量 | `jax.xla_computation`获取计算图的FLOPS估计（A23.5）；`jax.device_count`获取设备数量用于CPU总量估计（A23.2） |
| psutil | **本阶段新增**：系统资源监控 | `psutil.cpu_percent`获取CPU占用率（A23.2）；`psutil.virtual_memory`获取内存占用率（A23.3）；`psutil.Process`获取Agent进程的资源消耗，直接支撑A23.1-A23.4的资源度量 |
| time | 推理耗时测量 | `time.perf_counter`高精度计时，支撑A23.4的推理耗时度量 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V23.1 | 能效驱力非正性 | $R_{\text{eff}} \leq 0$ 对所有合法输入成立 | 随机资源指标测试 |
| V23.2 | CPU占用率范围 | $\text{cpu\_ratio} \in [0, 1]$ | 边界值测试 |
| V23.3 | 内存占用率范围 | $\text{mem\_ratio} \in [0, 1]$ | 边界值测试 |
| V23.4 | 推理耗时正性 | $\text{time\_cost} > 0$ | 正值验证 |
| V23.5 | 冗余度上界 | $\text{redundancy} \leq 1$ 当 $I \leq C$ | 信息效率测试 |
| V23.6 | 冗余度零点 | $\text{redundancy} = 0$ 当 $I = C$ | 等式条件测试 |
| V23.7 | 信息效率范围 | $\eta_{\text{info}} \geq 0$ | 非负性验证 |
| V23.8 | 能效驱力-复杂性相关性 | $R_{\text{eff}}$ 与 $-C(q)$ 正相关 | 合成复杂性数据测试 |
| V23.9 | 自适应感知步数 | $N_{\text{perceive}}^{\text{adaptive}}$ 不超过预算 | 时间约束测试 |
| V23.10 | 信息效率递减 | 感知迭代后期的 $\eta_{\text{info}}(k)$ 递减 | 迭代步数扫描 |
| V23.11 | EfficiencyDrive组合 | `EfficiencyDrive.gain_drive` 类型为 `GainDrive` | 类型检查 |
| V23.12 | 非法输入拒绝 | $\alpha_e \leq 0$、$\text{cpu\_ratio} \notin [0,1]$、$\text{time\_cost} \leq 0$ 等抛出 `ValueError` | 异常测试 |

---

> **阶段23完** | 下一步：阶段24将从本阶段的三大驱力出发，定义全局总奖励公式 $R_{\text{total}}$，建立动态权重自适应机制，并证明全局奖励与EFE的等价性。
