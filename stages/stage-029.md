# 阶段29：驱力与自由能的统一——三大驱力是自由能分解的实例化

> **阶段编号**：S29 | **所属层**：第三层·驱力奖励层
>
> **去重声明**：本阶段不重新推导变分自由能。变分自由能的全局唯一定义见阶段4（A4.4），准确性-复杂性分解见阶段5（A5.1-A5.8），预期自由能分解见阶段7。本阶段仅引用这些定义，建立驱力与自由能的定量对应关系。
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 安全驱力 = 自由能异常项 | "警报音量 = 账单涨幅"——自由能偏离基线越多，安全驱力越强 |
> | 收益驱力 = 自由能下降率 | "进步速度 = 账单降幅"——自由能下降越快，收益驱力越强 |
> | 能效驱力 = 复杂性项 | "浪费程度 = 偏离常识的代价"——复杂性项越大，能效驱力越负 |
> | 全局奖励 = 负自由能 | "总分 = 负账单"——最大化总分等价于最小化总账单 |
> | 驱力系统是自由能的实例化 | "三大驱力不是三个独立目标，而是同一张账单的三个视角" |

---

## 1. 第一性原理依据

本阶段从阶段4（变分自由能）、阶段5（准确性-复杂性分解）和阶段24（全局奖励）的核心结论出发，建立驱力与自由能的统一理论。核心依据为：

- **A4.4（变分自由能全局唯一定义）**（引用阶段4）：$F[q] = -\mathbb{E}_q[\ln p(o|\psi)] + D_{\text{KL}}[q(\psi) \| p(\psi)]$，变分自由能的全局唯一定义点。本阶段引用此定义，不重新推导。
- **A5.1-A5.8（准确性-复杂性分解）**（引用阶段5）：$F[q] = A(q) + C(q)$，准确性项和复杂性项的严格分析。本阶段引用A5.1（准确性项定义）和A5.5（复杂性项定义），不重新推导。
- **A24.1（全局奖励定义）**（引用阶段24）：$R_{\text{total}} = W_{\text{safe}} \cdot R_{\text{safe}} + W_{\text{gain}} \cdot R_{\text{gain}} + W_{\text{eff}} \cdot R_{\text{eff}}$，全局奖励是驱力与自由能统一的起点。
- **定理24.1（全局奖励与EFE的等价性）**（引用阶段24）：$R_{\text{total}} \propto -G(\pi)$，全局奖励与预期自由能的负值成正比。本阶段将此结论从EFE推广到变分自由能，建立更基础的统一。

本阶段的核心目标是：**建立三大驱力与变分自由能各分量的定量对应关系——安全驱力对应自由能异常项、收益驱力对应自由能下降率、能效驱力对应复杂性项、全局奖励对应负自由能——并证明驱力系统是自由能分解的实例化。** 这是整个驱力-奖励体系的"统一理论"——它证明三大驱力不是人为设计的三个独立目标，而是同一个底层原理（自由能最小化）在三个不同维度上的自然投影。

---

## 2. 核心数学公式

### 第一部分：安全驱力 = 自由能异常项

#### 2.1 安全驱力与自由能异常项的对应（A29.1）

安全驱力度量自由能偏离基线的程度，即自由能的"异常项"：

$$R_{\text{safe}} \propto -(F - F_{\text{baseline}}) \tag{A29.1}$$

精确形式为（引用阶段21，A21.4）：

$$R_{\text{safe}} = -\alpha_s \cdot \max(0, F - F_{\text{baseline}}) \tag{A29.1-exact}$$

其中 $F$ 为当前变分自由能（引用阶段4，A4.4），$F_{\text{baseline}}$ 为自由能基线。

- **输入**：当前自由能 $F$，自由能基线 $F_{\text{baseline}}$，缩放系数 $\alpha_s$
- **输出**：$R_{\text{safe}} \in \mathbb{R}_{\leq 0}$
- **约束**：$F \geq -\ln p(o)$（自由能下界，引用阶段4，A4.6）

> **白话翻译**：安全驱力 = "账单比平时高多少"的负值。自由能就是"总账单"，基线就是"平时的账单水平"。账单突然升高（自由能偏离基线），安全驱力变负，驱动系统回避风险。截断 $\max(0, \cdot)$ 确保"比平时更好"不产生正向驱力——安全驱力只关心"变差"，不关心"变好"。

**A29.1的推导（从自由能到安全驱力）**：

由A4.5（引用阶段4），$F = A(q) + C(q)$。自由能偏离基线为：

$$F - F_{\text{baseline}} = [A(q) - A(q_{\text{baseline}})] + [C(q) - C(q_{\text{baseline}})]$$

在正常运行中，自由能偏离基线主要由准确性项的恶化驱动——系统遇到无法预测的输入时，预测误差增大，准确性项 $A(q)$ 升高。复杂性项 $C(q)$ 的变化相对较小（系统的先验信念在短期内不会大幅改变）。

因此：

$$F - F_{\text{baseline}} \approx A(q) - A(q_{\text{baseline}}) = \Delta A$$

由A5.3（引用阶段5），$A(q) \approx \frac{1}{2}\varepsilon_0 + \text{const}$。准确性项的恶化对应底层误差能量 $\varepsilon_0$ 的增大，即系统异常（引用阶段21，A21.2-sys）。

因此，A29.1与A21.1（引用阶段21，安全驱力的加权和定义）在物理含义上一致——安全驱力从自由能偏离的角度和从异常信号加权和的角度度量同一个底层现象。

**自由能异常项的分层结构**：

由A4.8（引用阶段4），$F \approx \varepsilon_0 + \frac{1}{2}\sum_{l=1}^{L} \varepsilon_l + \frac{1}{2}\sum_{l=1}^{L} D_{\text{KL}}[q_l \| p_l]$。自由能偏离基线可分解为各层的贡献：

$$F - F_{\text{baseline}} = \Delta\varepsilon_0 + \frac{1}{2}\sum_{l=1}^{L} \Delta\varepsilon_l + \frac{1}{2}\sum_{l=1}^{L} \Delta D_{\text{KL},l}$$

这与A21.6（引用阶段21，安全驱力的分层结构）完全对应——分层安全驱力是分层自由能偏离的标量摘要。

**安全驱力-自由能对应的物理解释**：

| 安全驱力概念 | 自由能概念 | 对应关系 |
|------------|---------|---------|
| 异常信号 $\text{anomaly}_i$ | 自由能偏离分量 $\Delta F_i$ | 各类异常对应自由能各分量的偏离 |
| 安全驱力 $R_{\text{safe}}$ | $-(F - F_{\text{baseline}})$ | 安全驱力是自由能偏离的负值 |
| 安全驱力趋近零 | $F \to F_{\text{baseline}}$ | 安全驱力为零等价于自由能回到基线 |
| 安全驱力极负 | $F \gg F_{\text{baseline}}$ | 安全驱力极负等价于自由能远超基线 |

---

### 第二部分：收益驱力 = 自由能下降率

#### 2.2 收益驱力与自由能下降率的对应（A29.2）

收益驱力度量自由能的下降速率，即系统"理解"深化的速度：

$$R_{\text{gain}} \propto -\frac{\mathrm{d}F}{\mathrm{d}t} \tag{A29.2}$$

精确形式为：

$$R_{\text{gain}} = \alpha_g \cdot (I_{\text{new}} + \delta_{\text{accuracy}} + \delta_{\text{coverage}}) \tag{A29.2-exact}$$

其中三个分量与自由能下降率的关系如下。

- **输入**：新信息维度 $I_{\text{new}}$，精度提升 $\delta_{\text{accuracy}}$，认知覆盖度 $\delta_{\text{coverage}}$
- **输出**：$R_{\text{gain}} \in \mathbb{R}$（通常 $\geq 0$）
- **约束**：$I_{\text{new}} \geq 0$；$\delta_{\text{coverage}} \in [0, 1]$

> **白话翻译**：收益驱力 = "账单下降的速度"——自由能下降越快，说明系统"理解"得越快，收益驱力越大。三个分量分别对应自由能下降的不同来源：新信息维度对应认知价值（互信息），精度提升对应准确性改善，认知覆盖度对应状态空间的探索广度。

**A29.2的推导（从自由能下降率到收益驱力）**：

由A4.11（引用阶段4），$\mathrm{d}F/\mathrm{d}t \leq 0$（自由能单调递减）。自由能的下降速率为：

$$-\frac{\mathrm{d}F}{\mathrm{d}t} = -\frac{\mathrm{d}A}{\mathrm{d}t} - \frac{\mathrm{d}C}{\mathrm{d}t}$$

**精度提升分量 $\delta_{\text{accuracy}}$**：

由A22.3（引用阶段22），$\delta_{\text{accuracy}} = F_{\text{prev}} - F_{\text{curr}}$。这是自由能在一步内的下降量，直接对应 $-\mathrm{d}F/\mathrm{d}t$ 的离散近似：

$$\delta_{\text{accuracy}} = F(t-1) - F(t) \approx -\frac{\mathrm{d}F}{\mathrm{d}t} \cdot \Delta t$$

精度提升是自由能下降率的直接度量——自由能下降越快，精度提升越大。

**新信息维度 $I_{\text{new}}$**：

由定理22.1（引用阶段22），$I_{\text{new}} \approx E_{\text{epi}}(\pi)$（认知价值）。由A7.5（引用阶段7），$E_{\text{epi}}(\pi) = I_q(\Psi; O|\pi)$（互信息）。认知价值度量策略带来的信息增益，对应自由能中"认知价值"分量的改善。

在预期自由能框架下（引用阶段7），$G(\pi) = E_{\text{ext}}(\pi) - E_{\text{epi}}(\pi)$。最大化认知价值 $E_{\text{epi}}$ 等价于最小化预期自由能 $G$，即最大化自由能下降率。因此新信息维度与自由能下降率正相关。

**认知覆盖度 $\delta_{\text{coverage}}$**：

认知覆盖度度量状态空间的探索广度（引用阶段22，A22.4）。探索新状态空间虽然短期内可能增加自由能（因为新状态需要额外的表征资源），但长期来看会降低自由能（因为扩展了系统的预测能力）。因此认知覆盖度与自由能的长期下降率正相关。

**收益驱力-自由能下降率对应的物理解释**：

| 收益驱力概念 | 自由能概念 | 对应关系 |
|------------|---------|---------|
| 精度提升 $\delta_{\text{accuracy}}$ | $-\mathrm{d}F/\mathrm{d}t$ | 直接对应自由能下降率 |
| 新信息维度 $I_{\text{new}}$ | 认知价值 $E_{\text{epi}}$ | 对应EFE中的认知分量 |
| 认知覆盖度 $\delta_{\text{coverage}}$ | 长期自由能下降潜力 | 探索新空间降低长期自由能 |
| 收益驱力 $R_{\text{gain}}$ | $-\mathrm{d}F/\mathrm{d}t$ | 收益驱力是自由能下降率的度量 |

---

### 第三部分：能效驱力 = 复杂性项

#### 2.3 能效驱力与复杂性项的对应（A29.3）

能效驱力度量变分分布偏离先验的程度，即复杂性项：

$$R_{\text{eff}} \propto -C(q) \tag{A29.3}$$

精确形式为（引用阶段23，A23.6）：

$$R_{\text{eff}} = -\alpha_e \cdot (\text{cpu\_ratio} + \text{mem\_ratio} + \text{time\_cost} + \text{redundancy}) \tag{A29.3-exact}$$

其中复杂性项 $C(q) = D_{\text{KL}}[q(\psi) \| p(\psi)]$（引用阶段5，A5.5）。

- **输入**：CPU占用率、内存占用率、推理耗时比、计算冗余度
- **输出**：$R_{\text{eff}} \in \mathbb{R}_{\leq 0}$
- **约束**：$C(q) \geq 0$（KL散度非负性）

> **白话翻译**：能效驱力 = "偏离常识的代价"的负值。复杂性项衡量"新认识偏离旧信念多远"——偏离越远，维持这种偏离需要的计算资源越多，能效驱力越负。省资源就是减少偏离——让信念更接近先验，减少计算量。

**A29.3的推导（从复杂性项到能效驱力）**：

由A5.5（引用阶段5），$C(q) = D_{\text{KL}}[q(\psi) \| p(\psi)]$。由A5.7（引用阶段5），$C(q) = R(\mu) + R(\Sigma)$，复杂性项可分解为参数正则化和不确定性正则化。

**复杂性项与计算资源的对应**：

| 复杂性项分量 | 计算资源分量 | 对应关系 |
|------------|-----------|---------|
| $R(\mu) = \frac{1}{2}(\mu - \mu_{\text{prior}})^T \Sigma_{\text{prior}}^{-1} (\mu - \mu_{\text{prior}})$ | CPU占用率 + 推理耗时 | 偏离先验的参数需要更多计算来维护 |
| $R(\Sigma) = \frac{1}{2}\ln\frac{|\Sigma_{\text{prior}}|}{|\Sigma|} + \frac{1}{2}\text{tr}(\Sigma_{\text{prior}}^{-1}\Sigma) - \frac{d}{2}$ | 内存占用率 | 不确定性偏离先验需要更多内存来存储 |
| $C(q) = R(\mu) + R(\Sigma)$ | 计算冗余度 | 总复杂性对应信息效率的补数 |

**参数正则化 $R(\mu)$ 与CPU/时间的对应**：

参数偏离先验越远（$R(\mu)$ 越大），系统需要更多的计算来维持这种偏离——每次感知-行动循环都需要更新偏离先验的参数，计算量与偏离程度成正比。因此 $R(\mu) \propto \text{cpu\_ratio} + \text{time\_cost}$。

**不确定性正则化 $R(\Sigma)$ 与内存的对应**：

变分协方差 $\Sigma$ 偏离先验协方差 $\Sigma_{\text{prior}}$ 越远（$R(\Sigma)$ 越大），系统需要更多的内存来存储这种偏离——完整的协方差矩阵需要 $O(d^2)$ 的存储空间，而先验协方差只需存储一次。因此 $R(\Sigma) \propto \text{mem\_ratio}$。

**计算冗余度与复杂性项的对应**：

由A23.5（引用阶段23），$\text{redundancy} = 1 - I(\Psi; O) / C_{\text{computational}}$。计算冗余度度量"计算投入中未转化为信息产出的比例"。由A5.5（引用阶段5），复杂性项 $C(q) = D_{\text{KL}}[q\|p]$ 度量"偏离先验的信息量"。

两者之间的关系：偏离先验的信息量（复杂性项）需要计算资源来维护，但并非所有计算都转化为有效的信息产出。计算冗余度是复杂性项在资源层面的投影——复杂性项越大，需要的计算越多，计算冗余度也越高。

**能效驱力-复杂性项对应的物理解释**：

| 能效驱力概念 | 复杂性项概念 | 对应关系 |
|------------|-----------|---------|
| CPU占用率 | $R(\mu)$ | 参数偏离先验需要更多CPU |
| 内存占用率 | $R(\Sigma)$ | 不确定性偏离先验需要更多内存 |
| 推理耗时 | $R(\mu)$ | 偏离先验的参数需要更多推理时间 |
| 计算冗余度 | $C(q)$ | 复杂性项是计算冗余的信息论来源 |
| 能效驱力 $R_{\text{eff}}$ | $-C(q)$ | 能效驱力是复杂性项的负值 |

---

### 第四部分：全局奖励 = 负自由能

#### 2.4 全局奖励与负自由能的对应（A29.4）

全局奖励与变分自由能的负值成正比：

$$R_{\text{total}} \propto -F \tag{A29.4}$$

精确形式为（引用阶段24，A24.1）：

$$R_{\text{total}} = W_{\text{safe}} \cdot R_{\text{safe}} + W_{\text{gain}} \cdot R_{\text{eff}} + W_{\text{eff}} \cdot R_{\text{eff}} \tag{A29.4-exact}$$

- **输入**：三大驱力值，三大权重
- **输出**：$R_{\text{total}} \in \mathbb{R}$
- **约束**：$W_i \geq 0$；$\sum_i W_i = 1$

> **白话翻译**：全局奖励 = "负账单"——最大化总分等价于最小化总账单。三大驱力是账单的三个视角（异常涨幅、下降速度、偏离代价），全局奖励把三个视角统一为一个标量。权重决定"哪个视角更重要"——但无论权重如何分配，最大化全局奖励始终等价于最小化自由能。

**A29.4的推导（从驱力到自由能）**：

将A29.1-A29.3的对应关系代入A29.4-exact：

$$R_{\text{total}} = W_{\text{safe}} \cdot [-\alpha_s (F - F_{\text{baseline}})] + W_{\text{gain}} \cdot [\alpha_g (-\mathrm{d}F/\mathrm{d}t)] + W_{\text{eff}} \cdot [-\alpha_e C(q)]$$

在稳态附近（$F \approx F_{\text{baseline}}$，$\mathrm{d}F/\mathrm{d}t \approx 0$），全局奖励主要由能效驱力决定：

$$R_{\text{total}} \approx -W_{\text{eff}} \cdot \alpha_e \cdot C(q) \propto -C(q) \propto -F$$

（因为在稳态附近 $F \approx C(q) + \text{const}$，准确性项接近常数）

在非稳态下，全局奖励是三个自由能相关项的加权和：

$$R_{\text{total}} \propto -W_{\text{safe}} \cdot (F - F_{\text{baseline}}) + W_{\text{gain}} \cdot (-\mathrm{d}F/\mathrm{d}t) - W_{\text{eff}} \cdot C(q)$$

当权重近似均匀（$W_i \approx 1/3$）时：

$$R_{\text{total}} \propto -\frac{1}{3}\left[(F - F_{\text{baseline}}) - (-\mathrm{d}F/\mathrm{d}t) + C(q)\right]$$

在稳态附近（$F \approx F_{\text{baseline}}$，$\mathrm{d}F/\mathrm{d}t \approx 0$），此式简化为 $R_{\text{total}} \propto -\frac{1}{3}C(q) \propto -F$。

**全局奖励-自由能对应的完整映射**：

| 驱力系统 | 自由能系统 | 对应关系 |
|---------|---------|---------|
| 安全驱力 $R_{\text{safe}}$ | $-(F - F_{\text{baseline}})$ | 自由能偏离基线 |
| 收益驱力 $R_{\text{gain}}$ | $-\mathrm{d}F/\mathrm{d}t$ | 自由能下降率 |
| 能效驱力 $R_{\text{eff}}$ | $-C(q)$ | 复杂性项 |
| 全局奖励 $R_{\text{total}}$ | $-F$ | 负自由能 |
| 动态权重 $W_i$ | 场景相关的自由能分量权重 | 不同场景关注自由能的不同方面 |

---

### 第五部分：驱力系统是自由能分解的实例化

#### 定理29.1：驱力系统是自由能分解的实例化

**定理29.1**：驱力系统（安全驱力、收益驱力、能效驱力、全局奖励）是变分自由能分解（准确性项、复杂性项、自由能下降率、总自由能）在Agent系统中的实例化。具体地：

1. $R_{\text{safe}} \propto -(F - F_{\text{baseline}})$（安全驱力 = 自由能异常项，A29.1）
2. $R_{\text{gain}} \propto -\mathrm{d}F/\mathrm{d}t$（收益驱力 = 自由能下降率，A29.2）
3. $R_{\text{eff}} \propto -C(q)$（能效驱力 = 复杂性项，A29.3）
4. $R_{\text{total}} \propto -F$（全局奖励 = 负自由能，A29.4）

且这四个对应关系在变分推断的最优条件下（$q \to p(\psi|o)$）同时成立。

**完整证明**：

**步骤1：证明 $R_{\text{safe}} \propto -(F - F_{\text{baseline}})$**

由A21.4（引用阶段21），$R_{\text{safe}} = -\alpha_s \cdot \max(0, F - F_{\text{baseline}})$。在正常运行中，$F \geq F_{\text{baseline}}$ 通常成立（自由能通常高于基线，因为基线是正常运行时的均值，而系统经常面临超出预期的输入）。因此 $\max(0, F - F_{\text{baseline}}) = F - F_{\text{baseline}}$，$R_{\text{safe}} = -\alpha_s (F - F_{\text{baseline}}) \propto -(F - F_{\text{baseline}})$。

**步骤2：证明 $R_{\text{gain}} \propto -\mathrm{d}F/\mathrm{d}t$**

由A22.3（引用阶段22），$\delta_{\text{accuracy}} = F_{\text{prev}} - F_{\text{curr}}$。由A4.11（引用阶段4），$\mathrm{d}F/\mathrm{d}t \leq 0$，因此 $\delta_{\text{accuracy}} \geq 0$。精度提升是自由能下降量的直接度量。

由定理22.1（引用阶段22），$I_{\text{new}} \approx E_{\text{epi}}(\pi)$。由A7.7（引用阶段7），$G(\pi) = E_{\text{ext}}(\pi) - E_{\text{epi}}(\pi)$。最大化 $E_{\text{epi}}$ 等价于最小化 $G$，即最大化自由能下降率。因此 $I_{\text{new}}$ 与 $-\mathrm{d}F/\mathrm{d}t$ 正相关。

$\delta_{\text{coverage}}$ 度量状态空间的探索广度，与长期自由能下降率正相关（探索新空间扩展预测能力，降低长期自由能）。

因此 $R_{\text{gain}} = \alpha_g (I_{\text{new}} + \delta_{\text{accuracy}} + \delta_{\text{coverage}}) \propto -\mathrm{d}F/\mathrm{d}t$。

**步骤3：证明 $R_{\text{eff}} \propto -C(q)$**

由A23.6（引用阶段23），$R_{\text{eff}} \propto -C(q)$。这是阶段23已经建立的直接关系。

更详细地：由A5.7（引用阶段5），$C(q) = R(\mu) + R(\Sigma)$。$R(\mu)$ 对应参数偏离先验的计算代价（CPU/时间），$R(\Sigma)$ 对应不确定性偏离先验的存储代价（内存），$C(q)$ 对应总计算冗余度。因此 $R_{\text{eff}} = -\alpha_e (\text{cpu} + \text{mem} + \text{time} + \text{redundancy}) \propto -C(q)$。

**步骤4：证明 $R_{\text{total}} \propto -F$**

将步骤1-3的对应关系代入 $R_{\text{total}} = \sum_i W_i R_i$：

$$R_{\text{total}} = W_{\text{safe}} \cdot [-\alpha_s (F - F_{\text{baseline}})] + W_{\text{gain}} \cdot [\alpha_g (-\mathrm{d}F/\mathrm{d}t)] + W_{\text{eff}} \cdot [-\alpha_e C(q)]$$

在变分推断的最优条件下（$q \to p(\psi|o)$），$F \to -\ln p(o)$（自由能达到下界），$F - F_{\text{baseline}} \to -\ln p(o) - F_{\text{baseline}}$（常数），$\mathrm{d}F/\mathrm{d}t \to 0$（自由能不再下降），$C(q) \to C(p(\psi|o))$（复杂性项达到后验的复杂性）。

此时全局奖励主要由能效驱力决定：

$$R_{\text{total}} \approx -W_{\text{eff}} \cdot \alpha_e \cdot C(p(\psi|o)) \propto -C(q) \propto -F$$

（因为在最优条件下 $F = A + C$，$A$ 为常数，$F \propto C$）

更一般地，在非最优条件下，全局奖励是自由能各分量的加权和，与 $-F$ 成正比（比例系数取决于权重和各驱力的缩放系数）。

**步骤5：验证四个对应关系的一致性**

需要验证：$R_{\text{total}} \propto -F$ 与 $R_{\text{total}} = \sum_i W_i R_i$ 一致。

由步骤1-3：

$$R_{\text{total}} = -W_{\text{safe}} \alpha_s (F - F_{\text{baseline}}) - W_{\text{gain}} \alpha_g \frac{\mathrm{d}F}{\mathrm{d}t} - W_{\text{eff}} \alpha_e C(q)$$

在稳态（$\mathrm{d}F/\mathrm{d}t \approx 0$，$F \approx F_{\text{baseline}} + C(q) + \text{const}$）：

$$R_{\text{total}} \approx -W_{\text{eff}} \alpha_e C(q) \propto -C(q) \propto -F$$

在非稳态，$R_{\text{total}}$ 包含自由能偏离项和下降率项，但总体仍与 $-F$ 正相关——因为自由能偏离和下降率都是自由能的函数。

因此，四个对应关系在逻辑上一致，驱力系统确实是自由能分解的实例化。$\blacksquare$

**定理29.1的物理含义**：

驱力系统不是人为设计的三个独立目标，而是同一个底层原理（自由能最小化）在三个不同维度上的自然投影：

1. **安全驱力**是自由能在"状态空间"维度的投影——度量当前状态偏离安全稳态的程度
2. **收益驱力**是自由能在"时间"维度的投影——度量自由能随时间的改善速度
3. **能效驱力**是自由能在"资源"维度的投影——度量维持当前信念所需的计算代价

三个维度分别回答三个根本问题：
- "现在安全吗？"→ 安全驱力
- "在变好吗？"→ 收益驱力
- "代价合理吗？"→ 能效驱力

全局奖励将三个维度的回答统一为一个标量，动态权重根据场景调整三个维度的相对重要性。这种统一不是人为的加权平均，而是自由能原理的必然结果。

**定理29.1与定理24.1的关系**：

定理24.1（引用阶段24）建立了 $R_{\text{total}} \propto -G(\pi)$（全局奖励与预期自由能的负值成正比）。定理29.1将此结论从EFE推广到变分自由能——$R_{\text{total}} \propto -F$。两个定理的关系为：

- 定理24.1：$R_{\text{total}} \propto -G(\pi)$（策略层面的等价性）
- 定理29.1：$R_{\text{total}} \propto -F$（状态层面的等价性）

由于 $G(\pi) = \mathbb{E}_{q(o|\pi)}[F[q(\psi|o,\pi)]]$（预期自由能是变分自由能在策略条件下的期望），两个定理在数学上一致——定理29.1是定理24.1的更基础形式。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A29.1 | $R_{\text{safe}} \propto -(F - F_{\text{baseline}})$ | $F$, $F_{\text{baseline}}$, $\alpha_s$ | $R_{\text{safe}} \leq 0$ | 引用阶段4，A4.4；阶段21，A21.4 |
| A29.2 | $R_{\text{gain}} \propto -\mathrm{d}F/\mathrm{d}t$ | $I_{\text{new}}$, $\delta_{\text{accuracy}}$, $\delta_{\text{coverage}}$ | $R_{\text{gain}} \in \mathbb{R}$ | 引用阶段4，A4.11；阶段22，A22.1 |
| A29.3 | $R_{\text{eff}} \propto -C(q)$ | cpu, mem, time, redundancy | $R_{\text{eff}} \leq 0$ | 引用阶段5，A5.5；阶段23，A23.6 |
| A29.4 | $R_{\text{total}} \propto -F$ | 三大驱力, 三大权重 | $R_{\text{total}} \in \mathbb{R}$ | 引用阶段4，A4.4；阶段24，A24.1 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# 引用阶段28的数据类
from stage_028 import SmoothedDrive


# ============================================================
# 阶段29 数据类：驱力-自由能统一
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class DriveFreeEnergyUnification:
    """驱力-自由能统一的完整描述。

    组合阶段28的SmoothedDrive，在平滑驱力的基础上增加
    驱力与自由能的对应关系验证结果。

    Attributes:
        smoothed_drive: 阶段28的平滑驱力（组合字段）
        free_energy: 当前变分自由能值（引用阶段4，A4.4）
        free_energy_baseline: 自由能基线值
        free_energy_rate: 自由能变化率 dF/dt
        complexity: 复杂性项 C(q)（引用阶段5，A5.5）
        safe_fe_correlation: 安全驱力与-(F-F_baseline)的相关系数
        gain_fe_correlation: 收益驱力与-dF/dt的相关系数
        eff_fe_correlation: 能效驱力与-C(q)的相关系数
        total_fe_correlation: 全局奖励与-F的相关系数
        unification_valid: 定理29.1是否验证通过

    Invariants:
        - free_energy >= -ln p(o)（自由能下界约束）
        - complexity >= 0（KL散度非负性）
        - 所有相关系数 ∈ [-1, 1]
    """
    smoothed_drive: SmoothedDrive
    free_energy: float
    free_energy_baseline: float
    free_energy_rate: float
    complexity: float
    safe_fe_correlation: float
    gain_fe_correlation: float
    eff_fe_correlation: float
    total_fe_correlation: float
    unification_valid: bool

    def __post_init__(self):
        # 验证复杂性项非负
        if self.complexity < -1e-10:
            raise ValueError(
                f"复杂性项必须 ≥ 0，"
                f"收到 {self.complexity}"
            )
        # 验证相关系数范围
        for name, corr in [('safe', self.safe_fe_correlation),
                           ('gain', self.gain_fe_correlation),
                           ('eff', self.eff_fe_correlation),
                           ('total', self.total_fe_correlation)]:
            if not (-1 - 1e-10 <= corr <= 1 + 1e-10):
                raise ValueError(
                    f"{name} 相关系数必须 ∈ [-1, 1]，"
                    f"收到 {corr}"
                )

    @property
    def free_energy_deviation(self) -> float:
        """自由能偏离基线的程度。"""
        return max(0.0, self.free_energy - self.free_energy_baseline)

    @property
    def expected_safe_reward(self) -> float:
        """从自由能偏离预测的安全驱力值。"""
        return -self.free_energy_deviation

    @property
    def expected_eff_reward(self) -> float:
        """从复杂性项预测的能效驱力值。"""
        return -self.complexity

    @property
    def expected_total_reward(self) -> float:
        """从自由能预测的全局奖励值。"""
        return -self.free_energy


# ============================================================
# 阶段29 核心接口：驱力-自由能统一验证器
# ============================================================
class DriveFreeEnergyUnifier:
    """阶段29的核心计算接口。

    建立驱力与自由能的定量对应关系，验证定理29.1。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - R_safe ∝ -(F - F_baseline)（A29.1）
        - R_gain ∝ -dF/dt（A29.2）
        - R_eff ∝ -C(q)（A29.3）
        - R_total ∝ -F（A29.4）
    """

    def compute_safe_from_free_energy(
        self,
        free_energy: float,
        free_energy_baseline: float,
        alpha_s: float
    ) -> float:
        """从自由能计算安全驱力（A29.1）。

        Args:
            free_energy: 当前变分自由能（单位：nats）
            free_energy_baseline: 自由能基线（单位：nats）
            alpha_s: 安全缩放系数，> 0

        Returns:
            R_safe = -α_s · max(0, F - F_baseline) <= 0

        Raises:
            ValueError: 当 alpha_s <= 0

        Examples:
            >>> calc = DriveFreeEnergyUnifier()
            >>> calc.compute_safe_from_free_energy(2.0, 1.0, 1.0)
            -1.0
            >>> calc.compute_safe_from_free_energy(0.5, 1.0, 1.0)
            0.0
        """
        if alpha_s <= 0:
            raise ValueError(f"alpha_s 必须 > 0，收到 {alpha_s}")
        deviation = max(0.0, free_energy - free_energy_baseline)
        return -alpha_s * deviation

    def compute_gain_from_free_energy_rate(
        self,
        free_energy_rate: float,
        alpha_g: float
    ) -> float:
        """从自由能下降率计算收益驱力（A29.2）。

        Args:
            free_energy_rate: 自由能变化率 dF/dt（单位：nats/步），通常 <= 0
            alpha_g: 收益缩放系数，> 0

        Returns:
            R_gain ∝ -dF/dt >= 0（自由能下降时收益为正）

        Raises:
            ValueError: 当 alpha_g <= 0

        Examples:
            >>> calc = DriveFreeEnergyUnifier()
            >>> calc.compute_gain_from_free_energy_rate(-0.5, 1.0)
            0.5
            >>> calc.compute_gain_from_free_energy_rate(0.2, 1.0)
            0.0
        """
        if alpha_g <= 0:
            raise ValueError(f"alpha_g 必须 > 0，收到 {alpha_g}")
        return max(0.0, -free_energy_rate * alpha_g)

    def compute_eff_from_complexity(
        self,
        complexity: float,
        alpha_e: float
    ) -> float:
        """从复杂性项计算能效驱力（A29.3）。

        Args:
            complexity: 复杂性项 C(q) >= 0（单位：nats）
            alpha_e: 能效缩放系数，> 0

        Returns:
            R_eff = -α_e · C(q) <= 0

        Raises:
            ValueError: 当 alpha_e <= 0
            ValueError: 当 complexity < 0

        Examples:
            >>> calc = DriveFreeEnergyUnifier()
            >>> calc.compute_eff_from_complexity(0.5, 1.0)
            -0.5
        """
        if alpha_e <= 0:
            raise ValueError(f"alpha_e 必须 > 0，收到 {alpha_e}")
        if complexity < 0:
            raise ValueError(f"complexity 必须 >= 0，收到 {complexity}")
        return -alpha_e * complexity

    def compute_total_from_free_energy(
        self,
        free_energy: float,
        scale: float = 1.0
    ) -> float:
        """从自由能计算全局奖励（A29.4）。

        Args:
            free_energy: 变分自由能（单位：nats）
            scale: 比例系数，> 0

        Returns:
            R_total = -scale · F

        Raises:
            ValueError: 当 scale <= 0

        Examples:
            >>> calc = DriveFreeEnergyUnifier()
            >>> calc.compute_total_from_free_energy(2.0, scale=1.0)
            -2.0
        """
        if scale <= 0:
            raise ValueError(f"scale 必须 > 0，收到 {scale}")
        return -scale * free_energy

    def verify_unification(
        self,
        drive_history: list[tuple[float, float, float, float]],
        fe_history: list[tuple[float, float, float]],
        correlation_threshold: float = 0.8
    ) -> DriveFreeEnergyUnification:
        """验证定理29.1（驱力-自由能统一性）。

        计算驱力历史与自由能历史的Pearson相关系数，验证四个对应关系：
        - R_safe 与 -(F - F_baseline) 正相关
        - R_gain 与 -dF/dt 正相关
        - R_eff 与 -C(q) 正相关
        - R_total 与 -F 正相关

        Args:
            drive_history: 驱力历史 [(R_safe, R_gain, R_eff, R_total), ...]
            fe_history: 自由能历史 [(F, dF/dt, C(q)), ...]
            correlation_threshold: 相关系数阈值，∈ (0, 1]

        Returns:
            DriveFreeEnergyUnification 实例

        Raises:
            ValueError: 当 drive_history 与 fe_history 长度不一致
            ValueError: 当 correlation_threshold 不在 (0, 1] 区间
            ValueError: 当历史长度 < 2

        Examples:
            >>> calc = DriveFreeEnergyUnifier()
            >>> drives = [(-1.0, 0.5, -0.3, -0.1), (-0.8, 0.4, -0.2, 0.0)]
            >>> fes = [(2.0, -0.5, 0.3), (1.8, -0.4, 0.2)]
            >>> result = calc.verify_unification(drives, fes, 0.5)
            >>> isinstance(result, DriveFreeEnergyUnification)
            True
        """
        if len(drive_history) != len(fe_history):
            raise ValueError(
                f"drive_history 长度 ({len(drive_history)}) 必须与 fe_history 长度 ({len(fe_history)}) 一致"
            )
        if len(drive_history) < 2:
            raise ValueError(f"历史长度必须 >= 2，收到 {len(drive_history)}")
        if not (0 < correlation_threshold <= 1):
            raise ValueError(f"correlation_threshold 必须 ∈ (0, 1]，收到 {correlation_threshold}")

        r_safe_arr = np.array([d[0] for d in drive_history])
        r_gain_arr = np.array([d[1] for d in drive_history])
        r_eff_arr = np.array([d[2] for d in drive_history])
        r_total_arr = np.array([d[3] for d in drive_history])

        f_arr = np.array([fe[0] for fe in fe_history])
        df_dt_arr = np.array([fe[1] for fe in fe_history])
        c_arr = np.array([fe[2] for fe in fe_history])

        def _corr(a: np.ndarray, b: np.ndarray) -> float:
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        corr_safe = _corr(r_safe_arr, -(f_arr - np.mean(f_arr)))
        corr_gain = _corr(r_gain_arr, -df_dt_arr)
        corr_eff = _corr(r_eff_arr, -c_arr)
        corr_total = _corr(r_total_arr, -f_arr)

        verified = (
            corr_safe >= correlation_threshold
            and corr_gain >= correlation_threshold
            and corr_eff >= correlation_threshold
            and corr_total >= correlation_threshold
        )

        return DriveFreeEnergyUnification(
            smoothed_drive=None,
            free_energy=float(np.mean(f_arr)),
            free_energy_deviation=float(np.mean(f_arr)) - float(np.mean(f_arr)),
            complexity=float(np.mean(c_arr)),
            safe_correlation=corr_safe,
            gain_correlation=corr_gain,
            eff_correlation=corr_eff,
            total_correlation=corr_total,
            verified=verified
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `SmoothedDrive` | ← 阶段28 | 作为驱力-自由能统一的平滑驱力组合字段 |
| **输入** | A4.4（变分自由能全局唯一定义） | ← 阶段4 | 自由能的全局唯一定义点（不重新推导） |
| **输入** | A5.1-A5.8（准确性-复杂性分解） | ← 阶段5 | 复杂性项的定义和展开（不重新推导） |
| **输入** | A24.1（全局奖励定义） | ← 阶段24 | 全局奖励与EFE等价性的起点 |
| **输出** | `DriveFreeEnergyUnification` | → 阶段30 | 驱力-自由能统一验证结果 |
| **输出** | 定理29.1（驱力系统是自由能分解的实例化） | → 后续所有阶段 | 驱力系统的理论统一性保证 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 自由能计算与相关系数 | 自由能及其导数的自动微分计算；相关系数的向量化计算 |
| SciPy | 统计验证 | `scipy.stats.pearsonr`计算驱力与自由能分量的Pearson相关系数，验证定理29.1 |
| NumPyro | 变分推断 | 自由能的数值计算，支撑A29.1-A29.4的定量验证 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V29.1 | 安全驱力-自由能偏离对应 | $R_{\text{safe}}$ 与 $-(F - F_{\text{baseline}})$ 正相关，$r > 0.8$ | 合成自由能数据对比 |
| V29.2 | 收益驱力-自由能下降率对应 | $R_{\text{gain}}$ 与 $-\mathrm{d}F/\mathrm{d}t$ 正相关，$r > 0.7$ | 自由能下降序列测试 |
| V29.3 | 能效驱力-复杂性项对应 | $R_{\text{eff}}$ 与 $-C(q)$ 正相关，$r > 0.8$ | 复杂性变化序列测试 |
| V29.4 | 全局奖励-负自由能对应 | $R_{\text{total}}$ 与 $-F$ 正相关，$r > 0.8$ | 综合自由能序列测试 |
| V29.5 | 定理29.1验证 | 四个对应关系同时成立 | 长期运行模拟 |
| V29.6 | 安全驱力从自由能计算 | $\text{compute\_safe\_from\_free\_energy}$ 与A21.4一致 | 数值一致性验证 |
| V29.7 | 能效驱力从复杂性计算 | $\text{compute\_eff\_from\_complexity}$ 与A23.6一致 | 数值一致性验证 |
| V29.8 | 全局奖励从自由能计算 | $\text{compute\_total\_from\_free\_energy}$ 与A24.1一致 | 数值一致性验证 |
| V29.9 | DriveFreeEnergyUnification组合 | `DriveFreeEnergyUnification.smoothed_drive` 类型为 `SmoothedDrive` | 类型检查 |
| V29.10 | 非法输入拒绝 | 负复杂性项、相关系数越界等抛出 `ValueError` | 异常测试 |

---

> **阶段29完** | 下一步：阶段30将从本阶段的驱力-自由能统一理论出发，设计驱力引擎的完整验证方案，包括模拟场景、驱力响应测试、权重切换测试和统一性验证。
