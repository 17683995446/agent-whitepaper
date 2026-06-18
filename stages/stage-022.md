# 阶段22：收益驱力Reward_gain——从认知价值到信息收益

> **阶段编号**：S22 | **所属层**：第三层·驱力奖励层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 收益驱力 | "好奇心+进步感"——系统对新知识和能力提升的内在追求 |
> | 新信息维度 | "学到了多少新东西"——新表征与观测之间的互信息 |
> | 精度提升 | "理解变深了"——自由能的下降量，代表预测精度的改善 |
> | 认知覆盖度 | "知识面有多广"——已知状态空间占总状态空间的比例 |
> | 收益驱力的时间平滑 | "进步感不能大起大落"——平滑处理让探索动力更稳定 |

---

## 1. 第一性原理依据

本阶段从阶段7（认知价值与互信息）和阶段5（准确性项）的核心结论出发，建立收益驱力的完整数学定义。核心依据为：

- **A7.5（认知价值 = 互信息）**（引用阶段7）：$E_{\text{epi}}(\pi) = I_q(\Psi; O | \pi) = \mathbb{E}_{q(o|\pi)}[D_{\text{KL}}[q(\psi|o, \pi) \| q(\psi)]]$，认知价值度量策略带来的信息增益，是收益驱力中新信息维度的直接来源。
- **A5.1（准确性项）**（引用阶段5）：$A(q) = -\mathbb{E}_q[\ln p(o|\psi)]$，准确性项的改善代表系统预测精度的提升，是收益驱力中精度提升分量的理论基础。
- **A4.4（变分自由能全局唯一定义）**（引用阶段4）：$F[q] = -\mathbb{E}_q[\ln p(o|\psi)] + D_{\text{KL}}[q(\psi) \| p(\psi)]$，自由能的下降量直接度量系统"理解"的深化程度。
- **A7.6（认知价值的信息论统一）**（引用阶段7）：$E_{\text{epi}} = I_q(\Psi; O|\pi) = H[q(\psi)] - \mathbb{E}_{q(o|\pi)}[H[q(\psi|o,\pi)]]$，认知价值 = 先验不确定性 - 后验不确定性的期望，为收益驱力提供信息论基础。

本阶段的核心目标是：**定义收益驱力 $R_{\text{gain}}$，将系统运行过程中的认知收益分解为新信息维度、精度提升和认知覆盖度三个分量，建立收益驱力与认知价值（互信息）和准确性改善的定量关系，并提供时间平滑机制。** 收益驱力是三大驱力中驱动系统"主动探索"的核心力量——它对应阶段7的认知价值（A7.5），是"学得更多"这一认知需求的数学表达。

---

## 2. 核心数学公式

### 第一部分：收益驱力的定义

#### 2.1 收益驱力定义（A22.1）

收益驱力定义为新信息、精度提升和认知覆盖度的加权和：

$$R_{\text{gain}} = \alpha_g \cdot (I_{\text{new}} + \delta_{\text{accuracy}} + \delta_{\text{coverage}}) \tag{A22.1}$$

其中：
- $\alpha_g > 0$：收益驱力的全局缩放系数
- $I_{\text{new}} \geq 0$：新信息维度，度量系统获取的新信息量（A22.2）
- $\delta_{\text{accuracy}} \in \mathbb{R}$：精度提升，度量预测精度的改善（A22.3）
- $\delta_{\text{coverage}} \in [0, 1]$：认知覆盖度，度量知识面的广度（A22.4）

- **输入**：新信息维度 $I_{\text{new}}$，精度提升 $\delta_{\text{accuracy}}$，认知覆盖度 $\delta_{\text{coverage}}$，缩放系数 $\alpha_g$
- **输出**：$R_{\text{gain}} \in \mathbb{R}$（收益驱力可正可负，但通常非负）
- **约束**：$\alpha_g > 0$；$I_{\text{new}} \geq 0$；$\delta_{\text{coverage}} \in [0, 1]$

> **白话翻译**：收益驱力 = "学到了新东西" + "理解变深了" + "知识面变广了"。三项都是正的——学得越多、理解越深、知识面越广，收益驱力越大。$\alpha_g$ 是"好奇心强度"——调大则更渴望探索，调小则更安于现状。

**收益驱力与安全驱力的互补性**：安全驱力 $R_{\text{safe}} \leq 0$（总是非正，驱动回避），收益驱力 $R_{\text{gain}} \geq 0$（通常非负，驱动探索）。两者构成"避险-求新"的基本张力——系统需要在"不冒险"和"多学习"之间找到平衡。

**三个分量的物理含义**：

| 分量 | 度量 | 信息论对应 | 行为表现 |
|------|------|-----------|---------|
| $I_{\text{new}}$ | 新信息量 | 互信息 $I(\Psi_{\text{new}}; O)$（引用阶段7，A7.5） | 好奇心驱动探索 |
| $\delta_{\text{accuracy}}$ | 精度改善 | 自由能差 $F_{\text{prev}} - F_{\text{curr}}$（引用阶段4，A4.4） | 理解深化驱动学习 |
| $\delta_{\text{coverage}}$ | 知识广度 | 表征空间覆盖比 | 全面性驱动扩展 |

---

### 第二部分：新信息维度

#### 2.2 新信息维度（A22.2）

新信息维度度量系统通过新表征 $\Psi_{\text{new}}$ 获取的关于观测 $O$ 的信息量，直接对应阶段7的认知价值（A7.5）：

$$I_{\text{new}} = I(\Psi_{\text{new}}; O) = H[O] - H[O | \Psi_{\text{new}}] \tag{A22.2}$$

其中：
- $\Psi_{\text{new}}$ 为系统新获得或新更新的表征
- $O$ 为观测随机变量
- $I(\Psi_{\text{new}}; O)$ 为互信息（引用阶段7，A7.5）
- $H[O]$ 为观测的边际熵
- $H[O | \Psi_{\text{new}}]$ 为给定新表征后观测的条件熵

- **输入**：新表征 $\Psi_{\text{new}}$，观测分布 $p(O)$，条件分布 $p(O | \Psi_{\text{new}})$
- **输出**：$I_{\text{new}} \in \mathbb{R}_{\geq 0}$
- **约束**：$I_{\text{new}} \geq 0$（互信息非负性）；$I_{\text{new}} = 0$ 当且仅当 $\Psi_{\text{new}}$ 与 $O$ 独立

> **白话翻译**：新信息维度就是"新学到的知识能解释多少观测"——如果新表征与观测高度相关（互信息大），说明新知识很有价值；如果新表征与观测无关（互信息为零），说明新知识没有带来任何信息增益。

**与A7.5的精确对应**：

由A7.5（引用阶段7），认知价值 $E_{\text{epi}}(\pi) = I_q(\Psi; O | \pi)$。新信息维度 $I_{\text{new}}$ 是认知价值在"当前步"的具体实现——不是评估策略 $\pi$ 的预期信息增益，而是度量已经获取的新表征 $\Psi_{\text{new}}$ 带来的实际信息增益。

**新信息维度的计算方法**：

在高斯变分分布下（引用阶段4，A4.10），互信息有解析形式。设 $q(\Psi_{\text{new}}, O) = \mathcal{N}\left(\begin{pmatrix} \mu_\psi \\ \mu_o \end{pmatrix}, \begin{pmatrix} \Sigma_\psi & \Sigma_{\psi o} \\ \Sigma_{o\psi} & \Sigma_o \end{pmatrix}\right)$，则：

$$I(\Psi_{\text{new}}; O) = \frac{1}{2}\ln\frac{\det\Sigma_o}{\det(\Sigma_o - \Sigma_{o\psi}\Sigma_\psi^{-1}\Sigma_{\psi o})} \tag{A22.2-gaussian}$$

当 $\Psi_{\text{new}}$ 与 $O$ 为标量时，简化为：

$$I(\Psi_{\text{new}}; O) = -\frac{1}{2}\ln(1 - \rho^2)$$

其中 $\rho$ 为相关系数。当 $|\rho| \to 1$ 时，$I \to \infty$（完全相关，信息量无限大）；当 $\rho = 0$ 时，$I = 0$（无关，无信息增益）。

**新信息维度的边际递减效应**：

随着系统积累越来越多的知识，新表征带来的信息增量递减。设系统已有表征集 $\Psi_{\text{known}}$，则新表征 $\Psi_{\text{new}}$ 的条件互信息为：

$$I(\Psi_{\text{new}}; O | \Psi_{\text{known}}) = I(\Psi_{\text{new}}, \Psi_{\text{known}}; O) - I(\Psi_{\text{known}}; O) \tag{A22.2-conditional}$$

条件互信息 $I(\Psi_{\text{new}}; O | \Psi_{\text{known}})$ 度量"在已有知识的基础上，新知识还能带来多少额外信息"。随着 $\Psi_{\text{known}}$ 的丰富，条件互信息递减——这就是"边际递减效应"的数学基础：知识越多，新知识的价值越低。

**新信息维度与探索行为的关系**：

当 $I_{\text{new}}$ 高时，系统处于"信息丰富"的环境中，收益驱力强，鼓励继续探索；当 $I_{\text{new}}$ 低时，系统处于"信息贫乏"的环境中（或已经充分探索），收益驱力弱，鼓励转向其他目标。这与A7.9（引用阶段7）的"探索-利用统一"框架一致——认知价值自动调节探索强度。

---

### 第三部分：精度提升

#### 2.3 精度提升（A22.3）

精度提升度量系统预测精度的改善，用自由能的下降量表示：

$$\delta_{\text{accuracy}} = F_{\text{prev}} - F_{\text{curr}} \tag{A22.3}$$

其中 $F_{\text{prev}}$ 和 $F_{\text{curr}}$ 分别为前一步和当前步的变分自由能（引用阶段4，A4.4）。

- **输入**：前一步自由能 $F_{\text{prev}}$，当前自由能 $F_{\text{curr}}$
- **输出**：$\delta_{\text{accuracy}} \in \mathbb{R}$（可正可负）
- **约束**：由A4.11（引用阶段4），自由能单调递减 $\mathrm{d}F/\mathrm{d}t \leq 0$，因此 $\delta_{\text{accuracy}} \geq 0$ 在正常学习过程中成立

> **白话翻译**：精度提升就是"理解变深了多少"——自由能就是"总账单"，账单下降了多少，理解就深化了多少。如果账单不降反升，说明理解退步了，精度提升为负。

**精度提升的分层分解**：

由A4.8（引用阶段4），自由能可分解为各层误差能量和KL散度之和。精度提升可相应分解：

$$\delta_{\text{accuracy}} = \sum_{l=0}^{L} \Delta\varepsilon_l + \sum_{l=1}^{L} \Delta D_{\text{KL},l} \tag{A22.3-decompose}$$

其中 $\Delta\varepsilon_l = \varepsilon_l^{\text{prev}} - \varepsilon_l^{\text{curr}}$ 为第 $l$ 层误差能量的下降量，$\Delta D_{\text{KL},l}$ 为第 $l$ 层KL散度的下降量。

分层分解揭示精度提升的来源：
- 底层 $\Delta\varepsilon_0$ 下降 → 感知精度提升（"看得更清楚了"）
- 中层 $\Delta\varepsilon_l$ 下降 → 特征提取精度提升（"特征更准确了"）
- 高层 $\Delta D_{\text{KL},l}$ 下降 → 信念更新精度提升（"理解更深刻了"）

**精度提升与A5.4的连接**：

由A5.4（引用阶段5），准确性梯度 $\partial A / \partial \mu = -J^T \Sigma_\eta^{-1} (o - g(\mu))$。精度提升的一阶近似为：

$$\delta_{\text{accuracy}} \approx -\eta \left\|\frac{\partial F}{\partial \mu}\right\|^2 \tag{A22.3-approx}$$

其中 $\eta$ 为学习率。这表明精度提升与自由能梯度的范数平方成正比——梯度越大（离最优解越远），一步学习带来的精度提升越大；梯度越小（接近最优解），精度提升越小。这就是"容易摘的果子先摘完"的数学描述。

**精度提升的非负性保证**：

由A4.11（引用阶段4），$\mathrm{d}F/\mathrm{d}t \leq 0$，即自由能在梯度下降过程中单调递减。因此 $F_{\text{curr}} \leq F_{\text{prev}}$，$\delta_{\text{accuracy}} \geq 0$。但需要注意：

1. **权重更新后**的自由能可能暂时升高（因为感知推断尚未收敛到新的最优表征）
2. **环境变化**可能导致自由能突然升高（新的观测与旧模型不匹配）

因此，$\delta_{\text{accuracy}}$ 在短期内可能为负，但长期趋势为正（由定理17.1保证，引用阶段17）。

**精度提升的归一化**：

为使精度提升与 $I_{\text{new}}$ 和 $\delta_{\text{coverage}}$ 的量纲可比，引入归一化：

$$\tilde{\delta}_{\text{accuracy}} = \frac{F_{\text{prev}} - F_{\text{curr}}}{F_{\text{prev}} + \epsilon} \tag{A22.3-normalized}$$

其中 $\epsilon > 0$ 为小常数，防止除零。归一化后的精度提升为相对改善率，与 $I_{\text{new}}$ 和 $\delta_{\text{coverage}}$ 在同一数量级。

---

### 第四部分：认知覆盖度

#### 2.4 认知覆盖度（A22.4）

认知覆盖度度量系统知识面的广度——已知状态空间占总状态空间的比例：

$$\delta_{\text{coverage}} = \frac{\dim(\Psi_{\text{known}})}{\dim(\Psi_{\text{total}})} \tag{A22.4}$$

其中：
- $\dim(\Psi_{\text{known}})$ 为系统已掌握的状态空间维数
- $\dim(\Psi_{\text{total}})$ 为环境总状态空间维数

- **输入**：已知状态空间维数 $\dim(\Psi_{\text{known}})$，总状态空间维数 $\dim(\Psi_{\text{total}})$
- **输出**：$\delta_{\text{coverage}} \in [0, 1]$
- **约束**：$\dim(\Psi_{\text{known}}) \leq \dim(\Psi_{\text{total}})$；$\dim(\Psi_{\text{total}}) > 0$

> **白话翻译**：认知覆盖度就是"知识面有多广"——如果总共有100个需要了解的状态维度，系统已经掌握了60个，覆盖度就是0.6。覆盖度越高，系统对环境的理解越全面。

**已知状态空间的度量**：

$\dim(\Psi_{\text{known}})$ 的计算需要定义"已知"的判据。在变分推断框架下，一个状态维度 $\psi_j$ 被认为"已知"的条件为：

$$\text{known}(\psi_j) \iff H[q(\psi_j | o)] < \theta_{\text{known}} \tag{A22.4-criterion}$$

其中 $H[q(\psi_j | o)]$ 为给定观测后第 $j$ 个状态维度的条件熵，$\theta_{\text{known}}$ 为"已知"阈值。条件熵低于阈值意味着观测已充分约束该维度——系统对该维度的理解足够确定。

**认知覆盖度的增量形式**：

在每一步学习中，认知覆盖度的增量为：

$$\Delta\delta_{\text{coverage}} = \frac{N_{\text{newly\_known}}}{\dim(\Psi_{\text{total}})} \tag{A22.4-increment}$$

其中 $N_{\text{newly\_known}}$ 为本步新达到"已知"标准的维度数。增量形式与A22.1中的 $\delta_{\text{coverage}}$ 的关系为：

$$\delta_{\text{coverage}}(t) = \delta_{\text{coverage}}(t-1) + \Delta\delta_{\text{coverage}}(t)$$

**认知覆盖度与探索策略的关系**：

当 $\delta_{\text{coverage}}$ 低时，系统对大部分状态空间尚不了解，收益驱力鼓励广泛探索；当 $\delta_{\text{coverage}}$ 高时，系统已覆盖大部分状态空间，收益驱力鼓励深度挖掘（精度提升）而非广度扩展。

**认知覆盖度的上界**：

由于 $\dim(\Psi_{\text{known}}) \leq \dim(\Psi_{\text{total}})$，$\delta_{\text{coverage}} \leq 1$。但 $\delta_{\text{coverage}} = 1$ 意味着系统对所有状态维度都有充分理解——这在实践中几乎不可能达到（因为环境可能持续变化，引入新的未知维度）。因此 $\delta_{\text{coverage}}$ 始终严格小于1，这与阶段25的"缺口永不为零"结论一致。

---

### 第五部分：收益驱力的时间平滑

#### 2.5 收益驱力的时间平滑（A22.5）

与安全驱力类似，收益驱力需要时间平滑以避免瞬时波动：

$$R_{\text{gain}}^{\text{smooth}}(t) = \gamma_g \cdot R_{\text{gain}}^{\text{smooth}}(t-1) + (1 - \gamma_g) \cdot R_{\text{gain}}(t) \tag{A22.5}$$

其中 $\gamma_g \in [0, 1)$ 为收益驱力的平滑系数。

- **输入**：当前收益驱力 $R_{\text{gain}}(t)$，上一时刻平滑驱力 $R_{\text{gain}}^{\text{smooth}}(t-1)$，平滑系数 $\gamma_g$
- **输出**：$R_{\text{gain}}^{\text{smooth}}(t) \in \mathbb{R}$
- **约束**：$\gamma_g \in [0, 1)$；初始值 $R_{\text{gain}}^{\text{smooth}}(0) = 0$

> **白话翻译**：收益驱力的时间平滑就是"进步感不能大起大落"——如果每一步的收益驱力都剧烈波动，系统的探索行为会很不稳定。平滑处理让进步感像"缓坡"一样稳定上升。

**收益驱力平滑与安全驱力平滑的差异**：

| 特性 | 安全驱力平滑 | 收益驱力平滑 |
|------|------------|------------|
| 信号范围 | $R_{\text{safe}} \leq 0$ | $R_{\text{gain}} \geq 0$（通常） |
| 平滑目的 | 防止警报忽高忽低 | 防止进步感大起大落 |
| 推荐平滑系数 | $\gamma_s \approx 0.9$（较快响应威胁） | $\gamma_g \approx 0.95$（较慢积累进步感） |
| 非正/非负性 | 平滑保持非正 | 平滑保持非负（若原始驱力非负） |

**收益驱力平滑的非负性**：

若 $R_{\text{gain}}(t) \geq 0$ 对所有 $t$ 成立，且 $R_{\text{gain}}^{\text{smooth}}(0) = 0$，则由数学归纳法：$R_{\text{gain}}^{\text{smooth}}(t) = \gamma_g \cdot R_{\text{gain}}^{\text{smooth}}(t-1) + (1-\gamma_g) \cdot R_{\text{gain}}(t) \geq 0$。因此时间平滑保持收益驱力的非负性。

**收益驱力的长期趋势**：

在正常学习过程中，自由能持续下降（A4.11），新信息持续获取，认知覆盖度持续增加。因此收益驱力的长期趋势为正——系统持续从学习中获益。但随着知识积累，边际收益递减（A22.2-conditional），收益驱力的增长速率逐渐放缓。

**收益驱力与安全驱力的动态平衡**：

当系统处于安全状态（$R_{\text{safe}} \approx 0$）时，收益驱力主导，系统倾向于探索新领域；当系统出现异常（$R_{\text{safe}} \ll 0$）时，安全驱力主导，系统倾向于回避风险。这种动态平衡将在阶段24的全局奖励公式中正式化。

---

### 第六部分：收益驱力与认知价值的等价性

收益驱力的三个分量与阶段7的认知价值框架存在深层对应：

**定理22.1**：在变分推断的最优条件下（$q \approx p(\psi|o)$），收益驱力的主项 $I_{\text{new}}$ 等价于认知价值 $E_{\text{epi}}$。

**证明**：

由A7.5（引用阶段7），$E_{\text{epi}}(\pi) = I_q(\Psi; O | \pi)$。当变分分布 $q$ 充分接近真实后验 $p(\psi|o)$ 时（变分推断的目标），$I_q(\Psi; O | \pi) \approx I(\Psi; O | \pi)$。

由A22.2，$I_{\text{new}} = I(\Psi_{\text{new}}; O)$。当 $\Psi_{\text{new}}$ 包含当前步的所有新表征更新时，$I(\Psi_{\text{new}}; O) = I(\Psi; O | \pi)$（在策略 $\pi$ 条件下）。

因此 $I_{\text{new}} \approx E_{\text{epi}}(\pi)$。$\blacksquare$

**推论**：收益驱力可视为认知价值（探索驱力）与准确性改善（利用驱力）的统一度量。在EFE框架（A7.7，引用阶段7）中，$G(\pi) = E_{\text{ext}}(\pi) - E_{\text{epi}}(\pi)$，最小化EFE等价于最大化 $E_{\text{epi}}(\pi) - E_{\text{ext}}(\pi)$。收益驱力 $R_{\text{gain}} = \alpha_g \cdot (I_{\text{new}} + \delta_{\text{accuracy}} + \delta_{\text{coverage}})$ 中的 $I_{\text{new}}$ 对应 $E_{\text{epi}}$，$\delta_{\text{accuracy}}$ 对应 $-E_{\text{ext}}$ 的改善，$\delta_{\text{coverage}}$ 是额外的广度度量。因此收益驱力是EFE中认知价值分量的自然推广。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A22.1 | $R_{\text{gain}} = \alpha_g \cdot (I_{\text{new}} + \delta_{\text{accuracy}} + \delta_{\text{coverage}})$ | $I_{\text{new}}$, $\delta_{\text{accuracy}}$, $\delta_{\text{coverage}}$, $\alpha_g$ | $R_{\text{gain}} \in \mathbb{R}$ | $\alpha_g > 0$；$I_{\text{new}} \geq 0$ |
| A22.2 | $I_{\text{new}} = I(\Psi_{\text{new}}; O)$ | $\Psi_{\text{new}}$, $O$ 分布 | $I_{\text{new}} \in \mathbb{R}_{\geq 0}$ | 引用阶段7，A7.5 |
| A22.3 | $\delta_{\text{accuracy}} = F_{\text{prev}} - F_{\text{curr}}$ | $F_{\text{prev}}$, $F_{\text{curr}}$ | $\delta_{\text{accuracy}} \in \mathbb{R}$ | 引用阶段4，A4.4；通常 $\geq 0$ |
| A22.4 | $\delta_{\text{coverage}} = \dim(\Psi_{\text{known}}) / \dim(\Psi_{\text{total}})$ | $\dim(\Psi_{\text{known}})$, $\dim(\Psi_{\text{total}})$ | $\delta_{\text{coverage}} \in [0, 1]$ | $\dim(\Psi_{\text{known}}) \leq \dim(\Psi_{\text{total}})$ |
| A22.5 | $R_{\text{gain}}^{\text{smooth}} = \gamma_g \cdot R_{\text{gain}}^{\text{prev}} + (1-\gamma_g) \cdot R_{\text{gain}}$ | $R_{\text{gain}}(t)$, $R_{\text{gain}}^{\text{smooth}}(t-1)$, $\gamma_g$ | $R_{\text{gain}}^{\text{smooth}} \in \mathbb{R}$ | $\gamma_g \in [0, 1)$ |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# 引用阶段21的数据类
from stage_021 import SafetyDrive


# ============================================================
# 阶段22 数据类：收益驱力
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class GainDrive:
    """收益驱力的完整描述。

    组合阶段21的SafetyDrive，在安全驱力的基础上增加
    收益驱力的计算结果。

    Attributes:
        safety_drive: 阶段21的安全驱力（组合字段）
        alpha_g: 收益驱力全局缩放系数（A22.1）
        new_information: 新信息维度 I_new（A22.2）
        accuracy_improvement: 精度提升 δ_accuracy（A22.3）
        coverage: 认知覆盖度 δ_coverage（A22.4）
        raw_gain_reward: 原始收益驱力值（A22.1）
        smooth_gain_reward: 时间平滑后的收益驱力值（A22.5）
        gamma_g: 时间平滑系数（A22.5）
        free_energy_prev: 前一步自由能（A22.3）
        free_energy_curr: 当前自由能（A22.3）
        known_dims: 已知状态维度数（A22.4）
        total_dims: 总状态维度数（A22.4）

    Invariants:
        - alpha_g > 0
        - new_information >= 0
        - coverage ∈ [0, 1]
        - known_dims <= total_dims
        - gamma_g ∈ [0, 1)
    """
    safety_drive: SafetyDrive
    alpha_g: float
    new_information: float
    accuracy_improvement: float
    coverage: float
    raw_gain_reward: float
    smooth_gain_reward: float
    gamma_g: float
    free_energy_prev: float
    free_energy_curr: float
    known_dims: int
    total_dims: int

    def __post_init__(self):
        # 验证缩放系数
        if self.alpha_g <= 0:
            raise ValueError(
                f"收益驱力缩放系数必须 > 0，"
                f"收到 {self.alpha_g}"
            )
        # 验证新信息非负
        if self.new_information < 0:
            raise ValueError(
                f"新信息维度必须 ≥ 0，"
                f"收到 {self.new_information}"
            )
        # 验证覆盖度范围
        if not (0 <= self.coverage <= 1):
            raise ValueError(
                f"认知覆盖度必须 ∈ [0, 1]，"
                f"收到 {self.coverage}"
            )
        # 验证维度约束
        if self.known_dims > self.total_dims:
            raise ValueError(
                f"已知维度 ({self.known_dims}) 不能超过总维度 ({self.total_dims})"
            )
        if self.total_dims <= 0:
            raise ValueError(
                f"总维度必须 > 0，"
                f"收到 {self.total_dims}"
            )
        # 验证平滑系数
        if not (0 <= self.gamma_g < 1):
            raise ValueError(
                f"平滑系数必须 ∈ [0, 1)，"
                f"收到 {self.gamma_g}"
            )

    @property
    def is_learning(self) -> bool:
        """系统是否正在有效学习（收益驱力为正）。"""
        return self.smooth_gain_reward > 0

    @property
    def coverage_deficit(self) -> float:
        """认知覆盖度缺口（1 - 当前覆盖度）。"""
        return 1.0 - self.coverage

    @property
    def dominant_gain_component(self) -> str:
        """主导收益分量。"""
        components = {
            'new_information': self.new_information,
            'accuracy_improvement': max(0, self.accuracy_improvement),
            'coverage': self.coverage
        }
        return max(components, key=components.get)


# ============================================================
# 阶段22 核心接口：收益驱力计算器
# ============================================================
class GainDriveCalculator:
    """阶段22的核心计算接口。

    实现收益驱力的完整计算流程，包括新信息维度、
    精度提升、认知覆盖度和时间平滑。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - I_new >= 0（互信息非负性）
        - δ_coverage ∈ [0, 1]
        - 时间平滑保持非负性（若原始驱力非负）
    """

    def compute_new_information(
        self,
        new_representation: np.ndarray,
        observations: np.ndarray,
        joint_cov: np.ndarray
    ) -> float:
        """计算新信息维度（A22.2）。

        在高斯变分分布下，使用解析形式计算互信息：
        I(Ψ; O) = 0.5 * ln(det(Σ_o) / det(Σ_o - Σ_oψ Σ_ψ^{-1} Σ_ψo))

        Args:
            new_representation: 新表征样本，shape (n_samples, d_ψ)
            observations: 观测样本，shape (n_samples, d_o)
            joint_cov: 联合协方差矩阵，shape (d_ψ + d_o, d_ψ + d_o)

        Returns:
            I_new >= 0（单位：nats）

        Raises:
            ValueError: 当 joint_cov 不是方阵
            ValueError: 当 joint_cov 奇异或不是正定

        Examples:
            >>> calc = GainDriveCalculator()
            >>> joint_cov = np.array([[1.0, 0.5], [0.5, 1.0]])
            >>> calc.compute_new_information(
            ...     np.array([[0.0], [1.0]]),
            ...     np.array([[0.0], [1.0]]),
            ...     joint_cov
            ... )
            0.1438...
        """
        if joint_cov.ndim != 2 or joint_cov.shape[0] != joint_cov.shape[1]:
            raise ValueError(
                f"joint_cov 必须是方阵，收到 shape {joint_cov.shape}"
            )
        d_psi = new_representation.shape[1] if new_representation.ndim > 1 else 1
        d_o = observations.shape[1] if observations.ndim > 1 else 1
        total_d = d_psi + d_o
        if joint_cov.shape[0] != total_d:
            raise ValueError(
                f"joint_cov 维度 ({joint_cov.shape[0]}) 必须与 "
                f"d_ψ + d_o ({total_d}) 一致"
            )
        sigma_psi = joint_cov[:d_psi, :d_psi]
        sigma_o = joint_cov[d_psi:, d_psi:]
        sigma_o_psi = joint_cov[d_psi:, :d_psi]
        sigma_psi_o = joint_cov[:d_psi, d_psi:]
        try:
            sigma_psi_inv = np.linalg.inv(sigma_psi)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Σ_ψ 矩阵奇异，无法求逆") from exc
        conditional_cov = sigma_o - sigma_o_psi @ sigma_psi_inv @ sigma_psi_o
        det_sigma_o = max(np.linalg.det(sigma_o), 1e-12)
        det_conditional = max(np.linalg.det(conditional_cov), 1e-12)
        mi = 0.5 * np.log(det_sigma_o / det_conditional)
        return max(0.0, float(mi))

    def compute_accuracy_improvement(
        self,
        free_energy_prev: float,
        free_energy_curr: float
    ) -> float:
        """计算精度提升（A22.3）。

        Args:
            free_energy_prev: 前一步自由能（单位：nats）
            free_energy_curr: 当前自由能（单位：nats）

        Returns:
            δ_accuracy = F_prev - F_curr

        Examples:
            >>> calc = GainDriveCalculator()
            >>> calc.compute_accuracy_improvement(2.0, 1.5)
            0.5
        """
        return free_energy_prev - free_energy_curr

    def compute_coverage(
        self,
        known_dims: int,
        total_dims: int
    ) -> float:
        """计算认知覆盖度（A22.4）。

        Args:
            known_dims: 已知维度数，>= 0
            total_dims: 总维度数，> 0

        Returns:
            δ_coverage ∈ [0, 1]

        Raises:
            ValueError: 当 known_dims < 0 或 total_dims <= 0
            ValueError: 当 known_dims > total_dims

        Examples:
            >>> calc = GainDriveCalculator()
            >>> calc.compute_coverage(60, 100)
            0.6
        """
        if known_dims < 0:
            raise ValueError(f"known_dims 必须 >= 0，收到 {known_dims}")
        if total_dims <= 0:
            raise ValueError(f"total_dims 必须 > 0，收到 {total_dims}")
        if known_dims > total_dims:
            raise ValueError(
                f"known_dims ({known_dims}) 不能大于 total_dims ({total_dims})"
            )
        return known_dims / total_dims

    def compute_gain_reward(
        self,
        new_information: float,
        accuracy_improvement: float,
        coverage: float,
        alpha_g: float
    ) -> float:
        """计算原始收益驱力（A22.1）。

        Args:
            new_information: I_new，>= 0（单位：nats）
            accuracy_improvement: δ_accuracy（单位：nats）
            coverage: δ_coverage，∈ [0, 1]
            alpha_g: 缩放系数，> 0

        Returns:
            R_gain = α_g · (I_new + δ_accuracy + δ_coverage)

        Raises:
            ValueError: 当 alpha_g <= 0
            ValueError: 当 new_information < 0 或 coverage 不在 [0, 1]

        Examples:
            >>> calc = GainDriveCalculator()
            >>> calc.compute_gain_reward(0.5, 0.3, 0.6, alpha_g=1.0)
            1.4
        """
        if alpha_g <= 0:
            raise ValueError(f"alpha_g 必须 > 0，收到 {alpha_g}")
        if new_information < 0:
            raise ValueError(f"new_information 必须 >= 0，收到 {new_information}")
        if not (0 <= coverage <= 1):
            raise ValueError(f"coverage 必须 ∈ [0, 1]，收到 {coverage}")
        return alpha_g * (new_information + accuracy_improvement + coverage)

    def smooth_gain_reward(
        self,
        current: float,
        previous_smooth: float,
        gamma_g: float
    ) -> float:
        """时间平滑（A22.5）。

        Args:
            current: 当前收益驱力（单位：无量纲）
            previous_smooth: 上一时刻平滑值（单位：无量纲）
            gamma_g: 平滑系数，∈ [0, 1)

        Returns:
            平滑后的收益驱力，若 current >= 0 且 previous_smooth >= 0 则返回值 >= 0

        Raises:
            ValueError: 当 gamma_g 不在 [0, 1) 区间

        Examples:
            >>> calc = GainDriveCalculator()
            >>> calc.smooth_gain_reward(1.0, 0.5, 0.95)
            0.55
        """
        if not (0 <= gamma_g < 1):
            raise ValueError(f"gamma_g 必须 ∈ [0, 1)，收到 {gamma_g}")
        return gamma_g * previous_smooth + (1.0 - gamma_g) * current
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `SafetyDrive` | ← 阶段21 | 作为收益驱力的安全驱力组合字段 |
| **输入** | A7.5（认知价值 = 互信息） | ← 阶段7 | 新信息维度的理论基础 |
| **输入** | A5.1（准确性项） | ← 阶段5 | 精度提升的准确性来源 |
| **输入** | A4.4（变分自由能定义） | ← 阶段4 | 精度提升的自由能度量 |
| **输出** | `GainDrive` | → 阶段23 | 收益驱力作为能效驱力的组合字段 |
| **输出** | A22.1（收益驱力定义） | → 阶段24 | 全局奖励公式的收益分量 |
| **输出** | 定理22.1（收益驱力-认知价值等价性） | → 阶段24 | 全局奖励与EFE等价性的证明基础 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 互信息计算 | `jax.numpy.linalg.slogdet`计算高斯互信息（A22.2-gaussian）；`jax.grad`计算自由能梯度用于精度提升近似（A22.3-approx）；`jax.lax.scan`实现时间平滑（A22.5） |
| scikit-learn | 覆盖度评估 | `sklearn.neighbors.KernelDensity`估计已知状态空间的密度，支撑A22.4中"已知"判据的条件熵估计 |
| dm-tree | **本阶段新增**：表征空间分析 | 提供嵌套数据结构的映射与度量工具，支撑A22.4中状态空间维度的结构化计算和A22.2中新表征的条件互信息计算 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V22.1 | 新信息非负性 | $I_{\text{new}} \geq 0$ 对所有合法输入成立 | 合成互信息计算 |
| V22.2 | 互信息计算正确性 | 高斯互信息解析值与数值估计的相对误差 < 5% | 合成高斯数据测试 |
| V22.3 | 精度提升与自由能一致 | $\delta_{\text{accuracy}} = F_{\text{prev}} - F_{\text{curr}}$ | 自由能差值验证 |
| V22.4 | 精度提升非负性（学习过程） | $\delta_{\text{accuracy}} \geq 0$ 在正常学习过程中成立 | 多步训练自由能监控 |
| V22.5 | 覆盖度范围 | $\delta_{\text{coverage}} \in [0, 1]$ 对所有合法输入成立 | 边界值测试 |
| V22.6 | 覆盖度单调递增 | 学习过程中 $\delta_{\text{coverage}}$ 不递减 | 序列学习测试 |
| V22.7 | 收益驱力组合正确性 | $R_{\text{gain}} = \alpha_g \cdot (I_{\text{new}} + \delta_{\text{accuracy}} + \delta_{\text{coverage}})$ | 数值一致性验证 |
| V22.8 | 时间平滑非负性 | $R_{\text{gain}}^{\text{smooth}} \geq 0$ 当 $R_{\text{gain}} \geq 0$ | 平滑迭代数值验证 |
| V22.9 | 时间平滑收敛 | 收益驱力稳定后平滑值趋近原始值 | 长期迭代测试 |
| V22.10 | 边际递减效应 | $I(\Psi_{\text{new}}; O | \Psi_{\text{known}})$ 随 $\Psi_{\text{known}}$ 增加而递减 | 条件互信息序列测试 |
| V22.11 | 定理22.1验证 | $I_{\text{new}} \approx E_{\text{epi}}$ 在变分推断收敛时成立 | 变分推断收敛后对比 |
| V22.12 | GainDrive组合 | `GainDrive.safety_drive` 类型为 `SafetyDrive` | 类型检查 |
| V22.13 | 非法输入拒绝 | $\alpha_g \leq 0$、$\text{coverage} \notin [0,1]$、$\text{known\_dims} > \text{total\_dims}$ 等抛出 `ValueError` | 异常测试 |

---

> **阶段22完** | 下一步：阶段23将从本阶段的收益驱力出发，定义能效驱力 $R_{\text{eff}}$，组合收益驱力形成更完整的驱力体系。
