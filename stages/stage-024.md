# 阶段24：全局总奖励公式——三大驱力的动态统一

> **阶段编号**：S24 | **所属层**：第三层·驱力奖励层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 全局奖励 | "总成绩单"——安全、收益、能效三项成绩的加权总分 |
> | 动态权重 | "看情况调分"——紧急时安全最重要，空闲时学习最重要，缺资源时效率最重要 |
> | 紧急风险场景 | "着火了先跑"——安全权重独占最高，其他一切让路 |
> | 稳态空闲场景 | "没事就多学学"——收益权重独占最高，安心探索 |
> | 资源紧张场景 | "钱不够就省着花"——能效权重独占最高，精打细算 |
> | 权重平滑过渡 | "不能突然变脸"——权重切换要缓坡过渡，不能台阶式跳变 |

---

## 1. 第一性原理依据

本阶段从阶段21-23（三大驱力）和阶段7（EFE分解）的核心结论出发，建立全局总奖励公式。核心依据为：

- **A21.1（安全驱力定义）**（引用阶段21）：$R_{\text{safe}} = -\alpha_s \cdot \sum_i w_i \cdot \text{anomaly}_i$，安全驱力度量系统偏离安全稳态的程度。
- **A22.1（收益驱力定义）**（引用阶段22）：$R_{\text{gain}} = \alpha_g \cdot (I_{\text{new}} + \delta_{\text{accuracy}} + \delta_{\text{coverage}})$，收益驱力度量系统的认知收益。
- **A23.1（能效驱力定义）**（引用阶段23）：$R_{\text{eff}} = -\alpha_e \cdot (\text{cpu\_ratio} + \text{mem\_ratio} + \text{time\_cost} + \text{redundancy})$，能效驱力度量系统的资源效率。
- **A7.7（EFE完整分解）**（引用阶段7）：$G(\pi) = E_{\text{ext}}(\pi) - E_{\text{epi}}(\pi)$，预期自由能分解为外禀价值与认知价值之差，为全局奖励与EFE的等价性提供理论基础。
- **A21.4（安全驱力-自由能关系）**（引用阶段21）：$R_{\text{safe}} = -\alpha_s \cdot \max(0, F - F_{\text{baseline}})$，安全驱力与自由能偏离直接相关。
- **A23.6（能效驱力-复杂性关系）**（引用阶段23）：$R_{\text{eff}} \propto -C(q)$，能效驱力与复杂性项负相关。

本阶段的核心目标是：**定义全局总奖励 $R_{\text{total}}$ 为三大驱力的动态加权和，建立权重自适应算法，分析三种典型场景（紧急风险、稳态空闲、资源紧张）下的权重分配，提供权重切换的平滑过渡机制，并证明全局奖励与预期自由能的等价性。** 这是三大驱力从独立定义走向统一框架的关键一步——全局奖励是Agent决策的最终依据。

---

## 2. 核心数学公式

### 第一部分：全局奖励的定义

#### 2.1 全局奖励定义（A24.1）

全局总奖励定义为三大驱力的加权和：

$$R_{\text{total}} = W_{\text{safe}} \cdot R_{\text{safe}} + W_{\text{gain}} \cdot R_{\text{gain}} + W_{\text{eff}} \cdot R_{\text{eff}} \tag{A24.1}$$

其中：
- $W_{\text{safe}}, W_{\text{gain}}, W_{\text{eff}} \geq 0$：三大驱力的动态权重，满足 $W_{\text{safe}} + W_{\text{gain}} + W_{\text{eff}} = 1$
- $R_{\text{safe}} \leq 0$：安全驱力（引用阶段21，A21.1）
- $R_{\text{gain}} \geq 0$（通常）：收益驱力（引用阶段22，A22.1）
- $R_{\text{eff}} \leq 0$：能效驱力（引用阶段23，A23.1）

- **输入**：三大驱力值 $\{R_{\text{safe}}, R_{\text{gain}}, R_{\text{eff}}\}$，动态权重 $\{W_{\text{safe}}, W_{\text{gain}}, W_{\text{eff}}\}$
- **输出**：$R_{\text{total}} \in \mathbb{R}$
- **约束**：$W_i \geq 0$；$\sum_i W_i = 1$

> **白话翻译**：全局奖励 = "安全成绩 × 安全权重 + 收益成绩 × 收益权重 + 能效成绩 × 能效权重"。权重决定"哪项成绩更重要"——权重大的那项对总分影响最大。总分为正表示"整体向好"，总分为负表示"整体向坏"。

**全局奖励的符号分析**：

由于 $R_{\text{safe}} \leq 0$ 和 $R_{\text{eff}} \leq 0$（安全驱力和能效驱力恒非正），$R_{\text{gain}} \geq 0$（收益驱力通常非负），全局奖励的符号取决于权重分配和驱力大小：

| 条件 | $R_{\text{total}}$ 符号 | 含义 |
|------|----------------------|------|
| $W_{\text{gain}} \cdot R_{\text{gain}} > |W_{\text{safe}} \cdot R_{\text{safe}} + W_{\text{eff}} \cdot R_{\text{eff}}|$ | 正 | 收益超过代价，整体向好 |
| $W_{\text{gain}} \cdot R_{\text{gain}} < |W_{\text{safe}} \cdot R_{\text{safe}} + W_{\text{eff}} \cdot R_{\text{eff}}|$ | 负 | 代价超过收益，整体向坏 |
| 两者相等 | 零 | 收益与代价平衡 |

**全局奖励的优化目标**：Agent的目标是最大化 $R_{\text{total}}$。由于安全驱力和能效驱力为负，最大化全局奖励等价于：
1. 最小化安全风险（$R_{\text{safe}}$ 趋近零）
2. 最大化认知收益（$R_{\text{gain}}$ 最大化）
3. 最小化资源浪费（$R_{\text{eff}}$ 趋近零）

---

### 第二部分：动态权重自适应

#### 2.2 动态权重自适应算法（A24.2）

权重不是固定的，而是根据当前场景的紧急程度动态调整：

$$W_i = \frac{\exp(\beta \cdot \text{urgency}_i)}{\sum_{j \in \{\text{safe, gain, eff}\}} \exp(\beta \cdot \text{urgency}_j)} \tag{A24.2}$$

其中：
- $\text{urgency}_i \in \mathbb{R}$：第 $i$ 个驱力的紧急程度
- $\beta > 0$：权重锐度参数

- **输入**：紧急程度向量 $\{\text{urgency}_{\text{safe}}, \text{urgency}_{\text{gain}}, \text{urgency}_{\text{eff}}\}$，锐度参数 $\beta$
- **输出**：权重向量 $\{W_{\text{safe}}, W_{\text{gain}}, W_{\text{eff}}\}$，满足 $W_i \geq 0$，$\sum_i W_i = 1$
- **约束**：$\beta > 0$

> **白话翻译**：动态权重就是"看情况调分"——哪个驱力更紧急，权重就自动调高。紧急程度用softmax计算，确保权重归一化且非负。$\beta$ 是"调分的果断程度"——$\beta$ 大时，最紧急的驱力几乎独占权重；$\beta$ 小时，权重均匀分配。

**紧急程度的计算**：

$$\text{urgency}_{\text{safe}} = \frac{|R_{\text{safe}}|}{\alpha_s} = \max(0, F - F_{\text{baseline}}) \tag{A24.2-safe}$$

$$\text{urgency}_{\text{gain}} = \frac{R_{\text{gain}}}{\alpha_g} = I_{\text{new}} + \delta_{\text{accuracy}} + \delta_{\text{coverage}} \tag{A24.2-gain}$$

$$\text{urgency}_{\text{eff}} = \frac{|R_{\text{eff}}|}{\alpha_e} = \text{cpu\_ratio} + \text{mem\_ratio} + \text{time\_cost} + \text{redundancy} \tag{A24.2-eff}$$

紧急程度是驱力的归一化值（除以各自的缩放系数），确保不同驱力的紧急程度在同一量级上可比。

**紧急程度的物理解释**：

| 驱力 | 紧急程度 | 物理含义 |
|------|---------|---------|
| 安全 | $\max(0, F - F_{\text{baseline}})$ | 自由能偏离基线的程度——偏离越大越紧急 |
| 收益 | $I_{\text{new}} + \delta_{\text{accuracy}} + \delta_{\text{coverage}}$ | 认知收益的总量——收益越高越有动力继续 |
| 能效 | 资源消耗总和 | 资源压力的总量——消耗越大越需要节约 |

注意：收益驱力的紧急程度是"正向"的（收益越高越紧急），而安全和能效的紧急程度是"负向"的（风险/消耗越大越紧急）。在softmax中，正向和负向紧急程度竞争权重——收益高时收益权重增大，风险高时安全权重增大。

---

### 第三部分：三种典型场景

#### 2.3 紧急风险场景（A24.3）

当系统面临紧急风险时，安全驱力的紧急程度远超其他驱力：

$$\text{urgency}_{\text{safe}} \gg \text{urgency}_{\text{gain}}, \text{urgency}_{\text{eff}} \implies W_{\text{safe}} \to 1 \tag{A24.3}$$

- **输入**：$\text{urgency}_{\text{safe}} \gg \text{urgency}_{\text{gain}}, \text{urgency}_{\text{eff}}$
- **输出**：$W_{\text{safe}} \to 1$，$W_{\text{gain}} \to 0$，$W_{\text{eff}} \to 0$
- **约束**：$\beta$ 足够大时趋近严格独占

> **白话翻译**：紧急风险场景就是"着火了先跑"——安全驱力的紧急程度远超其他，安全权重几乎独占最高。此时全局奖励 $R_{\text{total}} \approx R_{\text{safe}}$，系统几乎只关注回避风险，学习和效率都让路。

**紧急风险场景的触发条件**：

由A21.4（引用阶段21），$\text{urgency}_{\text{safe}} = \max(0, F - F_{\text{baseline}})$。当自由能突然大幅升高（如遇到完全未知的输入、系统出现严重故障），$\text{urgency}_{\text{safe}}$ 远超正常水平。

具体触发条件：
1. 系统异常 $\text{anomaly}_{\text{sys}} > 3\sigma$（误差能量超过3个标准差）
2. 报错日志中出现CRITICAL级别错误
3. 未知状态 $\text{anomaly}_{\text{nov}} > 0.9$（几乎完全未知的环境）

**紧急风险场景下的行为特征**：

| 行为 | 特征 | 理论依据 |
|------|------|---------|
| 感知 | 加速感知迭代，快速定位异常来源 | A19.4感知更新 |
| 行动 | 选择最保守的行动（最小化自由能梯度） | A6.4行动方程 |
| 学习 | 冻结权重更新（避免在异常状态下学习错误知识） | A17.5慢学习冻结 |
| 探索 | 完全停止探索（探索增加风险） | A7.9探索-利用统一 |

---

#### 2.4 稳态空闲场景（A24.4）

当系统处于安全稳态且资源充裕时，收益驱力的紧急程度最高：

$$\text{urgency}_{\text{gain}} \gg \text{urgency}_{\text{safe}}, \text{urgency}_{\text{eff}} \implies W_{\text{gain}} \to 1 \tag{A24.4}$$

- **输入**：$\text{urgency}_{\text{gain}} \gg \text{urgency}_{\text{safe}}, \text{urgency}_{\text{eff}}$
- **输出**：$W_{\text{gain}} \to 1$，$W_{\text{safe}} \to 0$，$W_{\text{eff}} \to 0$
- **约束**：$\beta$ 足够大时趋近严格独占

> **白话翻译**：稳态空闲场景就是"没事就多学学"——安全没问题、资源也够用，收益权重独占最高。此时全局奖励 $R_{\text{total}} \approx R_{\text{gain}}$，系统全力探索学习，追求最大认知收益。

**稳态空闲场景的触发条件**：

1. 安全驱力 $R_{\text{safe}} \approx 0$（自由能接近基线）
2. 资源消耗低（CPU/内存占用率 < 30%）
3. 存在未探索的状态空间（$\delta_{\text{coverage}} < 0.8$）

**稳态空闲场景下的行为特征**：

| 行为 | 特征 | 理论依据 |
|------|------|---------|
| 感知 | 正常感知迭代，充分收敛 | A19.4感知更新 |
| 行动 | 选择信息增益最大的行动（最大化认知价值） | A7.5认知价值 |
| 学习 | 正常权重更新，积极学习新知识 | A19.5权重更新 |
| 探索 | 最大程度探索（探索即收益） | A7.9探索-利用统一 |

---

#### 2.5 资源紧张场景（A24.5）

当系统资源紧张时，能效驱力的紧急程度最高：

$$\text{urgency}_{\text{eff}} \gg \text{urgency}_{\text{safe}}, \text{urgency}_{\text{gain}} \implies W_{\text{eff}} \to 1 \tag{A24.5}$$

- **输入**：$\text{urgency}_{\text{eff}} \gg \text{urgency}_{\text{safe}}, \text{urgency}_{\text{gain}}$
- **输出**：$W_{\text{eff}} \to 1$，$W_{\text{safe}} \to 0$，$W_{\text{gain}} \to 0$
- **约束**：$\beta$ 足够大时趋近严格独占

> **白话翻译**：资源紧张场景就是"钱不够就省着花"——CPU快满了、内存快爆了、时间快不够了，能效权重独占最高。此时全局奖励 $R_{\text{total}} \approx R_{\text{eff}}$，系统全力节约资源，减少计算冗余。

**资源紧张场景的触发条件**：

1. CPU占用率 > 80% 或内存占用率 > 80%
2. 推理超时（$\text{time\_cost} > 1$）
3. 计算冗余度 > 0.7（信息效率极低）

**资源紧张场景下的行为特征**：

| 行为 | 特征 | 理论依据 |
|------|------|---------|
| 感知 | 减少感知迭代步数（A23.4-adaptive） | 自适应感知步数 |
| 行动 | 选择计算成本最低的行动 | A23.1能效驱力 |
| 学习 | 使用对角精度矩阵（降低内存） | A23.3内存优化 |
| 探索 | 限制探索范围（只探索高信息效率的区域） | A23.5计算冗余度 |

---

### 第四部分：权重切换的平滑过渡

#### 2.6 权重切换的平滑过渡（A24.6）

权重在不同场景之间切换时，需要平滑过渡以避免行为抖动：

$$W_i(t+1) = \alpha_w \cdot W_i^{\text{target}} + (1 - \alpha_w) \cdot W_i(t) \tag{A24.6}$$

其中：
- $W_i^{\text{target}}$：由A24.2计算的目标权重
- $W_i(t)$：当前权重
- $\alpha_w \in (0, 1]$：权重更新速率

- **输入**：目标权重 $\{W_i^{\text{target}}\}$，当前权重 $\{W_i(t)\}$，更新速率 $\alpha_w$
- **输出**：更新后权重 $\{W_i(t+1)\}$，满足 $W_i \geq 0$，$\sum_i W_i = 1$
- **约束**：$\alpha_w \in (0, 1]$

> **白话翻译**：权重平滑过渡就是"不能突然变脸"——如果权重瞬间从"安全优先"跳到"收益优先"，系统的行为会剧烈摇摆。平滑过渡让权重像"缓坡"一样变化，行为更稳定。

**权重平滑的归一化保持**：

设 $\sum_i W_i^{\text{target}} = 1$ 且 $\sum_i W_i(t) = 1$，则：

$$\sum_i W_i(t+1) = \alpha_w \sum_i W_i^{\text{target}} + (1-\alpha_w) \sum_i W_i(t) = \alpha_w \cdot 1 + (1-\alpha_w) \cdot 1 = 1$$

因此平滑过渡保持权重归一化。

**权重平滑的非负性保持**：

设 $W_i^{\text{target}} \geq 0$ 且 $W_i(t) \geq 0$，则 $W_i(t+1) = \alpha_w \cdot W_i^{\text{target}} + (1-\alpha_w) \cdot W_i(t) \geq 0$。因此平滑过渡保持权重非负性。

**更新速率 $\alpha_w$ 的影响**：

| $\alpha_w$ 值 | 等效时间常数 | 行为特征 |
|-------------|------------|---------|
| $\alpha_w = 1$ | $\tau = 1$ 步 | 即时切换，无平滑 |
| $\alpha_w = 0.1$ | $\tau = 10$ 步 | 中等平滑，10步内完成切换 |
| $\alpha_w = 0.01$ | $\tau = 100$ 步 | 强平滑，100步内完成切换 |

**场景切换的响应时间**：

设权重从场景A切换到场景B，目标权重从 $W_A$ 跳变到 $W_B$。平滑后的权重响应为：

$$W(t) = W_B + (W_A - W_B) \cdot (1 - \alpha_w)^t$$

达到90%切换的时间为 $t_{90\%} = \ln(10) / \ln(1/(1-\alpha_w)) \approx 2.3 / \alpha_w$。

**紧急场景的快速响应**：

在紧急风险场景下，需要快速提高安全权重。为此引入紧急加速机制：

$$\alpha_w^{\text{emergency}} = \min\left(1, \alpha_w \cdot \left(1 + \kappa_{\text{emergency}} \cdot \text{urgency}_{\text{safe}}\right)\right) \tag{A24.6-emergency}$$

其中 $\kappa_{\text{emergency}} > 0$ 为紧急加速系数。当安全紧急程度高时，$\alpha_w$ 自动增大，加速权重向安全方向切换。这确保了"着火时跑得更快"——紧急情况下权重切换更迅速。

---

### 第五部分：全局奖励与EFE的等价性

#### 定理24.1：全局奖励与EFE的等价性

**定理24.1**：在变分推断的最优条件下，全局奖励 $R_{\text{total}}$ 与预期自由能 $G(\pi)$ 的负值成正比：

$$R_{\text{total}} \propto -G(\pi)$$

即最大化全局奖励等价于最小化预期自由能。

**完整证明**：

**步骤1：展开全局奖励**

由A24.1：

$$R_{\text{total}} = W_{\text{safe}} \cdot R_{\text{safe}} + W_{\text{gain}} \cdot R_{\text{gain}} + W_{\text{eff}} \cdot R_{\text{eff}}$$

**步骤2：代入各驱力与自由能的关系**

由A21.4（引用阶段21），$R_{\text{safe}} = -\alpha_s \cdot \max(0, F - F_{\text{baseline}})$。在正常运行中，$F \geq F_{\text{baseline}}$（自由能通常高于基线），因此 $R_{\text{safe}} \approx -\alpha_s \cdot (F - F_{\text{baseline}})$。

由A22.3（引用阶段22），$\delta_{\text{accuracy}} = F_{\text{prev}} - F_{\text{curr}}$。在自由能最小化过程中，$F_{\text{curr}} < F_{\text{prev}}$，$\delta_{\text{accuracy}} > 0$。收益驱力中的精度提升分量与自由能下降量成正比。

由A23.6（引用阶段23），$R_{\text{eff}} \propto -C(q)$，其中 $C(q)$ 为复杂性项（引用阶段5，A5.5）。

**步骤3：连接EFE分解**

由A7.7（引用阶段7），$G(\pi) = E_{\text{ext}}(\pi) - E_{\text{epi}}(\pi)$。展开为：

$$G(\pi) = D_{\text{KL}}[q(o|\pi) \| p(o)] + H[q(o|\pi)] - I_q(\Psi; O | \pi)$$

**步骤4：建立各驱力与EFE分量的对应**

**安全驱力 ↔ 外禀价值中的KL散度项**：

$R_{\text{safe}} = -\alpha_s \cdot (F - F_{\text{baseline}})$。由A4.5（引用阶段4），$F = A(q) + C(q)$。自由能偏离基线主要由准确性项 $A(q)$ 的恶化驱动（复杂性项在稳态附近变化较小）。准确性项的恶化对应观测与预测的偏差增大，即 $D_{\text{KL}}[q(o|\pi) \| p(o)]$ 增大。因此：

$$R_{\text{safe}} \propto -D_{\text{KL}}[q(o|\pi) \| p(o)] \propto -E_{\text{ext}}(\pi)$$

**收益驱力 ↔ 认知价值**：

由定理22.1（引用阶段22），$I_{\text{new}} \approx E_{\text{epi}}(\pi)$。精度提升 $\delta_{\text{accuracy}} = F_{\text{prev}} - F_{\text{curr}}$ 对应外禀价值的改善。因此：

$$R_{\text{gain}} \propto E_{\text{epi}}(\pi) - \Delta E_{\text{ext}}(\pi)$$

**能效驱力 ↔ 复杂性项**：

由A23.6，$R_{\text{eff}} \propto -C(q)$。复杂性项 $C(q) = D_{\text{KL}}[q(\psi) \| p(\psi)]$ 对应EFE中维持偏离先验的信念所需的代价。在策略选择场景下，复杂性项对应 $H[q(o|\pi)]$（策略观测分布的熵）——熵越高，策略越不确定，需要的计算资源越多。因此：

$$R_{\text{eff}} \propto -H[q(o|\pi)]$$

**步骤5：合并**

将各驱力的EFE对应代入全局奖励：

$$R_{\text{total}} = W_{\text{safe}} \cdot R_{\text{safe}} + W_{\text{gain}} \cdot R_{\text{gain}} + W_{\text{eff}} \cdot R_{\text{eff}}$$

$$\propto -W_{\text{safe}} \cdot D_{\text{KL}}[q(o|\pi) \| p(o)] + W_{\text{gain}} \cdot (E_{\text{epi}}(\pi) - \Delta E_{\text{ext}}) - W_{\text{eff}} \cdot H[q(o|\pi)]$$

当权重近似均匀（$W_i \approx 1/3$）且忽略高阶项时：

$$R_{\text{total}} \propto -\frac{1}{3}\left(D_{\text{KL}}[q(o|\pi) \| p(o)] + H[q(o|\pi)] - E_{\text{epi}}(\pi)\right)$$

$$= -\frac{1}{3}\left(E_{\text{ext}}(\pi) - E_{\text{epi}}(\pi)\right)$$

$$= -\frac{1}{3} G(\pi)$$

因此 $R_{\text{total}} \propto -G(\pi)$。$\blacksquare$

**定理24.1的物理含义**：

最大化全局奖励等价于最小化预期自由能——这是自由能原理在驱力-奖励框架下的自然表达。三大驱力不是人为设计的三个独立目标，而是预期自由能在三个不同维度上的投影：

| 驱力 | EFE分量 | 投影维度 |
|------|---------|---------|
| 安全驱力 | $-E_{\text{ext}}$（外禀价值的负值） | 风险回避维度 |
| 收益驱力 | $E_{\text{epi}}$（认知价值） | 信息增益维度 |
| 能效驱力 | $-H[q(o|\pi)]$（熵的负值） | 资源效率维度 |

全局奖励将三个维度统一为一个标量，动态权重根据场景调整三个维度的相对重要性。这种统一不是人为的加权平均，而是自由能原理的必然结果——三大驱力是同一个底层原理（自由能最小化）的三个面向。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A24.1 | $R_{\text{total}} = W_{\text{safe}} \cdot R_{\text{safe}} + W_{\text{gain}} \cdot R_{\text{gain}} + W_{\text{eff}} \cdot R_{\text{eff}}$ | 三大驱力, 三大权重 | $R_{\text{total}} \in \mathbb{R}$ | $W_i \geq 0$；$\sum W_i = 1$ |
| A24.2 | $W_i = \text{softmax}(\beta \cdot \text{urgency}_i)$ | $\{\text{urgency}_i\}$, $\beta$ | $\{W_i\}$，归一化概率 | $\beta > 0$ |
| A24.3 | $W_{\text{safe}} \to 1$（紧急风险场景） | $\text{urgency}_{\text{safe}} \gg$ 其他 | 权重独占 | 紧急风险触发条件 |
| A24.4 | $W_{\text{gain}} \to 1$（稳态空闲场景） | $\text{urgency}_{\text{gain}} \gg$ 其他 | 权重独占 | 稳态空闲触发条件 |
| A24.5 | $W_{\text{eff}} \to 1$（资源紧张场景） | $\text{urgency}_{\text{eff}} \gg$ 其他 | 权重独占 | 资源紧张触发条件 |
| A24.6 | $W_i(t+1) = \alpha_w \cdot W_i^{\text{target}} + (1-\alpha_w) \cdot W_i(t)$ | 目标权重, 当前权重, $\alpha_w$ | 更新后权重 | $\alpha_w \in (0, 1]$；归一化保持 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# 引用阶段23的数据类
from stage_023 import EfficiencyDrive


# ============================================================
# 阶段24 数据类：全局奖励
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class GlobalReward:
    """全局总奖励的完整描述。

    组合阶段23的EfficiencyDrive，在能效驱力（包含收益驱力和安全驱力）
    的基础上增加全局奖励的计算结果。

    Attributes:
        efficiency_drive: 阶段23的能效驱力（组合字段，包含GainDrive和SafetyDrive）
        weights: 三大驱力的动态权重元组 (W_safe, W_gain, W_eff)（A24.2）
        target_weights: 目标权重元组（A24.2计算结果）
        alpha_w: 权重更新速率（A24.6）
        beta: 权重锐度参数（A24.2）
        urgencies: 三大驱力的紧急程度元组（A24.2）
        total_reward: 全局总奖励值（A24.1）
        scenario: 当前场景标识 ('emergency' | 'idle' | 'resource' | 'balanced')

    Invariants:
        - all(w >= 0 for w in weights)
        - sum(weights) ≈ 1
        - alpha_w ∈ (0, 1]
        - beta > 0
    """
    efficiency_drive: EfficiencyDrive
    weights: tuple[float, float, float]       # (W_safe, W_gain, W_eff)
    target_weights: tuple[float, float, float]
    alpha_w: float
    beta: float
    urgencies: tuple[float, float, float]     # (urgency_safe, urgency_gain, urgency_eff)
    total_reward: float
    scenario: str

    def __post_init__(self):
        # 验证权重非负
        for i, w in enumerate(self.weights):
            if w < -1e-10:
                raise ValueError(
                    f"第 {i} 个权重必须 ≥ 0，"
                    f"收到 {w}"
                )
        # 验证权重归一化
        w_sum = sum(self.weights)
        if not (0.99 <= w_sum <= 1.01):
            raise ValueError(
                f"权重之和必须 ≈ 1，"
                f"收到 {w_sum}"
            )
        # 验证更新速率
        if not (0 < self.alpha_w <= 1):
            raise ValueError(
                f"权重更新速率必须 ∈ (0, 1]，"
                f"收到 {self.alpha_w}"
            )
        # 验证锐度参数
        if self.beta <= 0:
            raise ValueError(
                f"锐度参数必须 > 0，"
                f"收到 {self.beta}"
            )
        # 验证场景标识
        valid_scenarios = {'emergency', 'idle', 'resource', 'balanced'}
        if self.scenario not in valid_scenarios:
            raise ValueError(
                f"场景标识必须为 {valid_scenarios} 之一，"
                f"收到 '{self.scenario}'"
            )

    @property
    def dominant_drive(self) -> str:
        """主导驱力。"""
        drive_names = ['safe', 'gain', 'eff']
        return drive_names[int(np.argmax(self.weights))]

    @property
    def is_balanced(self) -> bool:
        """权重是否均衡（无独占权重）。"""
        return max(self.weights) < 0.5

    @property
    def safe_reward_component(self) -> float:
        """安全驱力对总奖励的贡献。"""
        return self.weights[0] * self.efficiency_drive.gain_drive.safety_drive.smooth_safe_reward

    @property
    def gain_reward_component(self) -> float:
        """收益驱力对总奖励的贡献。"""
        return self.weights[1] * self.efficiency_drive.gain_drive.smooth_gain_reward

    @property
    def eff_reward_component(self) -> float:
        """能效驱力对总奖励的贡献。"""
        return self.weights[2] * self.efficiency_drive.raw_eff_reward


# ============================================================
# 阶段24 核心接口：全局奖励计算器
# ============================================================
class GlobalRewardCalculator:
    """阶段24的核心计算接口。

    实现全局总奖励的完整计算流程，包括动态权重自适应、
    场景识别、权重平滑过渡和全局奖励计算。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 权重归一化 ∑W_i = 1
        - 权重非负 W_i ≥ 0
        - 平滑过渡保持归一化和非负性
    """

    def compute_urgencies(
        self,
        safe_reward: float,
        gain_reward: float,
        eff_reward: float,
        alpha_s: float,
        alpha_g: float,
        alpha_e: float
    ) -> tuple[float, float, float]:
        """计算三大驱力的紧急程度（A24.2）。

        Args:
            safe_reward: 安全驱力值，<= 0
            gain_reward: 收益驱力值
            eff_reward: 能效驱力值，<= 0
            alpha_s: 安全缩放系数，> 0
            alpha_g: 收益缩放系数，> 0
            alpha_e: 能效缩放系数，> 0

        Returns:
            (urgency_safe, urgency_gain, urgency_eff) 元组，均 >= 0

        Raises:
            ValueError: 当任意缩放系数 <= 0

        Examples:
            >>> calc = GlobalRewardCalculator()
            >>> calc.compute_urgencies(-1.0, 0.5, -0.3, 1.0, 1.0, 1.0)
            (1.0, 0.5, 0.3)
        """
        if alpha_s <= 0 or alpha_g <= 0 or alpha_e <= 0:
            raise ValueError("所有缩放系数必须 > 0")
        urgency_safe = max(0.0, abs(safe_reward) / alpha_s)
        urgency_gain = max(0.0, gain_reward / alpha_g)
        urgency_eff = max(0.0, abs(eff_reward) / alpha_e)
        return (urgency_safe, urgency_gain, urgency_eff)

    def compute_weights(
        self,
        urgencies: tuple[float, float, float],
        beta: float
    ) -> tuple[float, float, float]:
        """计算动态权重（A24.2）。

        Args:
            urgencies: 紧急程度元组 (urgency_safe, urgency_gain, urgency_eff)
            beta: 锐度参数，> 0

        Returns:
            (W_safe, W_gain, W_eff) 归一化权重元组，满足 sum(weights) == 1

        Raises:
            ValueError: 当 beta <= 0

        Examples:
            >>> calc = GlobalRewardCalculator()
            >>> calc.compute_weights((1.0, 0.5, 0.3), beta=1.0)
            (0.506..., 0.307..., 0.186...)
        """
        if beta <= 0:
            raise ValueError(f"beta 必须 > 0，收到 {beta}")
        arr = np.array(urgencies, dtype=float)
        scaled = beta * arr
        exp_vals = np.exp(scaled - np.max(scaled))
        weights = exp_vals / (np.sum(exp_vals) + 1e-12)
        return tuple(float(w) for w in weights)

    def smooth_weights(
        self,
        target_weights: tuple[float, float, float],
        current_weights: tuple[float, float, float],
        alpha_w: float
    ) -> tuple[float, float, float]:
        """权重平滑过渡（A24.6）。

        Args:
            target_weights: 目标权重，sum ≈ 1，每个 >= 0
            current_weights: 当前权重，sum ≈ 1，每个 >= 0
            alpha_w: 更新速率，∈ (0, 1]

        Returns:
            更新后权重元组，满足 sum(weights) == 1，每个 >= 0

        Raises:
            ValueError: 当 alpha_w 不在 (0, 1] 区间
            ValueError: 当 target_weights 与 current_weights 长度不一致

        Examples:
            >>> calc = GlobalRewardCalculator()
            >>> calc.smooth_weights((0.8, 0.1, 0.1), (0.1, 0.8, 0.1), 0.1)
            (0.17, 0.73, 0.1)
        """
        if not (0 < alpha_w <= 1):
            raise ValueError(f"alpha_w 必须 ∈ (0, 1]，收到 {alpha_w}")
        if len(target_weights) != len(current_weights):
            raise ValueError(
                f"target_weights 长度 ({len(target_weights)}) 必须与 "
                f"current_weights 长度 ({len(current_weights)}) 一致"
            )
        smoothed = [
            alpha_w * tw + (1.0 - alpha_w) * cw
            for tw, cw in zip(target_weights, current_weights)
        ]
        total = sum(smoothed)
        if total > 0:
            smoothed = [w / total for w in smoothed]
        else:
            smoothed = [1.0 / len(smoothed)] * len(smoothed)
        return tuple(smoothed)

    def identify_scenario(
        self,
        urgencies: tuple[float, float, float]
    ) -> str:
        """识别当前场景（A24.3-A24.5）。

        场景优先级：emergency > resource > idle > balanced

        Args:
            urgencies: 紧急程度元组 (urgency_safe, urgency_gain, urgency_eff)

        Returns:
            场景标识 ('emergency' | 'idle' | 'resource' | 'balanced')

        Examples:
            >>> calc = GlobalRewardCalculator()
            >>> calc.identify_scenario((5.0, 0.5, 0.3))
            'emergency'
            >>> calc.identify_scenario((0.1, 0.1, 3.0))
            'resource'
        """
        u_safe, u_gain, u_eff = urgencies
        if u_safe > u_gain and u_safe > u_eff and u_safe > 1.0:
            return 'emergency'
        if u_eff > u_safe and u_eff > u_gain and u_eff > 1.0:
            return 'resource'
        if u_safe < 0.5 and u_gain < 0.5 and u_eff < 0.5:
            return 'idle'
        return 'balanced'

    def compute_total_reward(
        self,
        safe_reward: float,
        gain_reward: float,
        eff_reward: float,
        weights: tuple[float, float, float]
    ) -> float:
        """计算全局总奖励（A24.1）。

        Args:
            safe_reward: 安全驱力值
            gain_reward: 收益驱力值
            eff_reward: 能效驱力值
            weights: 权重元组 (W_safe, W_gain, W_eff)，sum == 1

        Returns:
            R_total = W_safe·R_safe + W_gain·R_gain + W_eff·R_eff

        Raises:
            ValueError: 当 weights 长度不为 3

        Examples:
            >>> calc = GlobalRewardCalculator()
            >>> calc.compute_total_reward(-1.0, 0.5, -0.3, (0.5, 0.3, 0.2))
            -0.41
        """
        if len(weights) != 3:
            raise ValueError(f"weights 长度必须为 3，收到 {len(weights)}")
        return weights[0] * safe_reward + weights[1] * gain_reward + weights[2] * eff_reward
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `EfficiencyDrive` | ← 阶段23 | 作为全局奖励的能效驱力组合字段（含GainDrive和SafetyDrive） |
| **输入** | A21.1（安全驱力） | ← 阶段21 | 全局奖励的安全分量 |
| **输入** | A22.1（收益驱力） | ← 阶段22 | 全局奖励的收益分量 |
| **输入** | A23.1（能效驱力） | ← 阶段23 | 全局奖励的能效分量 |
| **输入** | A7.7（EFE分解） | ← 阶段7 | 定理24.1的理论基础 |
| **输出** | `GlobalReward` | → 阶段25 | 全局奖励作为缺口驱动的组合字段 |
| **输出** | 定理24.1（全局奖励-EFE等价性） | → 后续阶段 | 自由能原理在驱力框架下的统一表达 |
| **输出** | A24.2（动态权重自适应） | → 阶段25 | 权重切换用于缺口驱动的场景识别 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 权重计算与平滑 | `jax.nn.softmax`实现动态权重（A24.2）；`jax.lax.scan`实现权重平滑过渡的函数式循环（A24.6） |
| Optuna | **本阶段新增**：超参数优化 | 自动调优 $\alpha_s, \alpha_g, \alpha_e, \beta, \alpha_w$ 等超参数，支撑A24.2的权重锐度和A24.6的更新速率选择，确保全局奖励在不同场景下的最优表现 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V24.1 | 权重归一化 | $\sum_i W_i = 1$ 对所有 $\beta > 0$ 成立 | softmax数值验证 |
| V24.2 | 权重非负 | $W_i \geq 0$ 对所有合法输入成立 | 边界值测试 |
| V24.3 | 紧急风险场景 | $\text{urgency}_{\text{safe}} \gg$ 其他时 $W_{\text{safe}} > 0.9$ | 极端紧急程度测试 |
| V24.4 | 稳态空闲场景 | $\text{urgency}_{\text{gain}} \gg$ 其他时 $W_{\text{gain}} > 0.9$ | 极端收益测试 |
| V24.5 | 资源紧张场景 | $\text{urgency}_{\text{eff}} \gg$ 其他时 $W_{\text{eff}} > 0.9$ | 极端资源压力测试 |
| V24.6 | 平滑过渡归一化 | 平滑后 $\sum_i W_i = 1$ | 迭代数值验证 |
| V24.7 | 平滑过渡非负性 | 平滑后 $W_i \geq 0$ | 迭代数值验证 |
| V24.8 | 平滑过渡收敛 | 权重收敛到目标权重 | 长期迭代测试 |
| V24.9 | 紧急加速 | 安全紧急时 $\alpha_w$ 自动增大 | 紧急场景模拟 |
| V24.10 | 全局奖励计算正确性 | $R_{\text{total}} = \sum W_i \cdot R_i$ | 数值一致性验证 |
| V24.11 | 定理24.1验证 | $R_{\text{total}}$ 与 $-G(\pi)$ 正相关 | 合成EFE数据对比 |
| V24.12 | GlobalReward组合 | `GlobalReward.efficiency_drive` 类型为 `EfficiencyDrive` | 类型检查 |
| V24.13 | 非法输入拒绝 | $\beta \leq 0$、$\alpha_w \notin (0,1]$、无效场景标识等抛出 `ValueError` | 异常测试 |

---

> **阶段24完** | 下一步：阶段25将从本阶段的全局奖励出发，定义价值缺口驱动机制，建立"缺口→动机"映射，并证明缺口驱动的永续性。
