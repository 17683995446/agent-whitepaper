# 阶段37：任务综合得分公式——从多维评估到统一排序

> **阶段编号**：S37 | **所属层**：第四层·任务规划层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 综合得分 | "最终排名分"——把价值、紧迫、可行、成本四项综合算总分 |
> | 驱力加权 | "三科权重不同"——安全、收益、能效三科在总分中占比不同 |
> | 紧迫度 | "火烧眉毛程度"——缺口越大越紧迫，用S型曲线量化 |
> | 可行性 | "能不能做"——资源够不够，不够就是不可行 |
> | 成本 | "做一次花多少"——CPU、内存、时间三样都要花钱 |
> | 得分归一化 | "百分制换算"——把原始分换算到0-1范围，方便比较 |

---

## 1. 第一性原理依据

本阶段从阶段24（全局奖励）、阶段25（价值缺口）和阶段32（归一化缺口）的核心结论出发，建立任务综合得分公式。核心依据为：

- **A24.1（全局奖励定义）**（引用阶段24）：$R_{\text{total}} = W_{\text{safe}} \cdot R_{\text{safe}} + W_{\text{gain}} \cdot R_{\text{gain}} + W_{\text{eff}} \cdot R_{\text{eff}}$，全局奖励是三大驱力的动态加权和。任务综合得分需要将驱力加权推广到任务级别——不同类型的任务对应不同的驱力权重。
- **A25.1（价值缺口定义）**（引用阶段25）：$\Delta V = V_{\text{target}} - V_{\text{current}}$，价值缺口是任务得分的核心驱动力——缺口越大的任务越值得执行。综合得分需要将价值缺口作为主要评分维度。
- **A25.2-A25.4（三大缺口分解）**（引用阶段25）：$\Delta V_{\text{safe}}, \Delta V_{\text{gain}}, \Delta V_{\text{eff}}$ 分别度量安全、收益和能效的缺口。不同类型的任务对应不同的缺口分量——安全任务对应安全缺口，收益任务对应收益缺口，能效任务对应能效缺口。
- **A32.5（缺口归一化）**（引用阶段32）：$\Delta V_{i,\text{norm}} = \Delta V_i / \Delta V_{i,\max}$，归一化缺口将不同量纲的缺口统一到 $[0, 1]$ 范围，为跨驱力的得分比较提供基础。
- **A7.7（EFE完整分解）**（引用阶段7）：$G(\pi) = E_{\text{ext}}(\pi) - E_{\text{epi}}(\pi)$，预期自由能分解为外禀价值与认知价值。综合得分公式的设计遵循EFE的结构——价值项（对应外禀价值）减去成本项（对应复杂性项），加上紧迫度和可行性调节项。

本阶段的核心目标是：**定义任务综合得分 Score(task) = W·ΔV_gap + α·urgency(task) + β·feasibility(task) - γ·cost(task)，建立驱力加权 W·ΔV_gap 的三分量展开，定义紧迫度的S型曲线 urgency(task) = 1/(1 + exp(-κ·(ΔV_gap - θ)))，定义可行性 feasibility(task) = resource_available / resource_required，定义成本 cost(task) = cpu_cost + mem_cost + time_cost，实现得分归一化 Score_norm = (Score - Score_min) / (Score_max - Score_min)，并封装为TaskScorer数据类，组合DynamicTaskPool作为任务池的输入。** 综合得分公式是从"多维评估"到"统一排序"的关键一步——它将价值缺口、紧迫度、可行性和成本四个维度综合为单一得分，为任务的选择和调度提供精确的评分依据。

---

## 2. 核心数学公式

### 第一部分：综合得分定义

#### 2.1 综合得分定义（A37.1）

任务的综合得分定义为四个维度的加权和：

$$\text{Score}(\text{task}) = W \cdot \Delta V_{\text{gap}}(\text{task}) + \alpha \cdot \text{urgency}(\text{task}) + \beta \cdot \text{feasibility}(\text{task}) - \gamma \cdot \text{cost}(\text{task}) \tag{A37.1}$$

其中：
- $W \cdot \Delta V_{\text{gap}}(\text{task})$：驱力加权缺口项——任务对应驱力的加权缺口值（A37.2）
- $\alpha \cdot \text{urgency}(\text{task})$：紧迫度项——任务的紧急程度（A37.3）
- $\beta \cdot \text{feasibility}(\text{task})$：可行性项——任务的可执行程度（A37.4）
- $\gamma \cdot \text{cost}(\text{task})$：成本项——任务的资源消耗（A37.5）
- $\alpha, \beta, \gamma \geq 0$：各项的权重系数

**各项权重的默认值**：

| 系数 | 默认值 | 含义 | 设计理由 |
|------|--------|------|---------|
| $W$ | 1.0 | 缺口项权重 | 缺口是核心驱动力，权重最高 |
| $\alpha$ | 0.3 | 紧迫度权重 | 紧迫度是调节项，权重适中 |
| $\beta$ | 0.2 | 可行性权重 | 可行性是约束项，权重较低 |
| $\gamma$ | 0.5 | 成本权重 | 成本是惩罚项，权重较高（避免高成本低收益任务） |

**综合得分的符号分析**：

- $W \cdot \Delta V_{\text{gap}} \geq 0$（缺口非负，引用A25.6）
- $\alpha \cdot \text{urgency} \geq 0$（紧迫度非负）
- $\beta \cdot \text{feasibility} \in [0, \beta]$（可行性在0-1之间）
- $\gamma \cdot \text{cost} \geq 0$（成本非负）

因此 $\text{Score}(\text{task})$ 可正可负——当成本超过价值+紧迫度+可行性时，得分为负，表示该任务不值得执行。

**综合得分与EFE的对应关系**：

综合得分的结构与EFE（引用A7.7）存在深刻的对应：

| 综合得分项 | EFE对应项 | 含义 |
|-----------|----------|------|
| $W \cdot \Delta V_{\text{gap}}$ | $-E_{\text{ext}}(\pi)$（外禀价值的负值） | 价值越高，得分越高（等价于外禀价值越低，EFE越低） |
| $\alpha \cdot \text{urgency}$ | — | EFE中没有直接对应项，是工程调节项 |
| $\beta \cdot \text{feasibility}$ | — | EFE中没有直接对应项，是工程约束项 |
| $-\gamma \cdot \text{cost}$ | $C(q)$（复杂性项） | 成本越高，得分越低（等价于复杂性越高，自由能越高） |

紧迫度和可行性是EFE框架中没有直接对应的工程调节项——它们在理论框架中不是必需的（EFE自动平衡探索和利用），但在工程实践中对于避免不可行任务和高成本任务至关重要。

- **输入**：任务 task，驱力权重 $W$，调节系数 $\alpha, \beta, \gamma$
- **输出**：$\text{Score}(\text{task}) \in \mathbb{R}$
- **约束**：$\alpha, \beta, \gamma \geq 0$；$W > 0$

> **白话翻译**：综合得分就是"最终排名分"——把四个维度综合算总分：价值缺口（离目标差多远）是主分，紧迫度（火烧眉毛程度）是加分，可行性（能不能做）是加分，成本（花多少资源）是扣分。总分越高越值得做。如果成本太高，总分可能为负，说明不值得做。

---

### 第二部分：驱力加权

#### 2.2 驱力加权（A37.2）

驱力加权将价值缺口按三大驱力的动态权重展开：

$$W \cdot \Delta V_{\text{gap}}(\text{task}) = W_{\text{safe}} \cdot \Delta V_{\text{safe}}(\text{task}) + W_{\text{gain}} \cdot \Delta V_{\text{gain}}(\text{task}) + W_{\text{eff}} \cdot \Delta V_{\text{eff}}(\text{task}) \tag{A37.2}$$

其中 $W_{\text{safe}}, W_{\text{gain}}, W_{\text{eff}}$ 为动态驱力权重（引用A24.2），满足 $W_{\text{safe}} + W_{\text{gain}} + W_{\text{eff}} = 1$。

**任务类型与缺口的对应**：

不同类型的任务对应不同的缺口分量：

| 任务类型 | 主要缺口分量 | 次要缺口分量 | 说明 |
|---------|------------|------------|------|
| 安全任务（$\mathcal{T}_{\text{safe}}$） | $\Delta V_{\text{safe}}$ | $\Delta V_{\text{gain}} \cdot 0.1, \Delta V_{\text{eff}} \cdot 0.1$ | 安全任务主要消除安全缺口 |
| 收益任务（$\mathcal{T}_{\text{gain}}$） | $\Delta V_{\text{gain}}$ | $\Delta V_{\text{safe}} \cdot 0.1, \Delta V_{\text{eff}} \cdot 0.1$ | 收益任务主要消除收益缺口 |
| 能效任务（$\mathcal{T}_{\text{eff}}$） | $\Delta V_{\text{eff}}$ | $\Delta V_{\text{safe}} \cdot 0.1, \Delta V_{\text{gain}} \cdot 0.1$ | 能效任务主要消除能效缺口 |

**任务级别的缺口提取**：

$$\Delta V_{\text{safe}}(\text{task}) = \begin{cases} \Delta V_{\text{safe,norm}} & \text{if task} \in \mathcal{T}_{\text{safe}} \\ 0.1 \cdot \Delta V_{\text{safe,norm}} & \text{otherwise} \end{cases} \tag{A37.2-extract-safe}$$

$$\Delta V_{\text{gain}}(\text{task}) = \begin{cases} \Delta V_{\text{gain,norm}} & \text{if task} \in \mathcal{T}_{\text{gain}} \\ 0.1 \cdot \Delta V_{\text{gain,norm}} & \text{otherwise} \end{cases} \tag{A37.2-extract-gain}$$

$$\Delta V_{\text{eff}}(\text{task}) = \begin{cases} \Delta V_{\text{eff,norm}} & \text{if task} \in \mathcal{T}_{\text{eff}} \\ 0.1 \cdot \Delta V_{\text{eff,norm}} & \text{otherwise} \end{cases} \tag{A37.2-extract-eff}$$

非主要缺口分量乘以0.1的衰减系数——安全任务对收益缺口和能效缺口也有间接贡献（消除安全威胁释放资源用于探索和优化），但贡献较小。

**驱力加权的场景敏感性**：

驱力权重 $W_{\text{safe}}, W_{\text{gain}}, W_{\text{eff}}$ 是动态的（引用A24.2），随场景变化而调整：

- **紧急场景**：$W_{\text{safe}} = 0.8$，安全任务的驱力加权显著增大，安全任务得分远高于其他类型
- **稳定场景**：$W_{\text{gain}} = 0.8$，收益任务的驱力加权显著增大，探索任务得分最高
- **紧张场景**：$W_{\text{eff}} = 0.8$，能效任务的驱力加权显著增大，优化任务得分最高

这种场景敏感性确保综合得分与当前环境的优先级一致——紧急时安全任务得分最高，空闲时收益任务得分最高，缺资源时能效任务得分最高。

- **输入**：任务 task，三大缺口 $\Delta V_{\text{safe}}, \Delta V_{\text{gain}}, \Delta V_{\text{eff}}$，动态权重 $W_{\text{safe}}, W_{\text{gain}}, W_{\text{eff}}$
- **输出**：$W \cdot \Delta V_{\text{gap}}(\text{task}) \in \mathbb{R}_{\geq 0}$
- **约束**：权重非负且和为1；缺口归一化到 $[0, 1]$

> **白话翻译**：驱力加权就是"三科权重不同"——安全、收益、能效三科在总分中占比不同，而且这个占比会随场景变化。着火时安全科占80%，空闲时收益科占80%，缺钱时能效科占80%。每个任务主要对应自己那科的缺口，对其他两科也有少量间接贡献。

---

### 第三部分：紧迫度

#### 2.3 紧迫度（A37.3）

紧迫度用S型（sigmoid）曲线量化任务的紧急程度：

$$\text{urgency}(\text{task}) = \frac{1}{1 + \exp(-\kappa \cdot (\Delta V_{\text{gap}}(\text{task}) - \theta))} \tag{A37.3}$$

其中：
- $\kappa > 0$：S型曲线的陡度参数（默认 $\kappa = 10$）
- $\theta > 0$：紧迫度阈值（默认 $\theta = 0.5$）
- $\Delta V_{\text{gap}}(\text{task})$：任务对应的驱力加权缺口值（A37.2）

**S型曲线的性质**：

1. **单调递增**：$\text{urgency}'(\text{task}) > 0$——缺口越大，紧迫度越高
2. **有界性**：$\text{urgency}(\text{task}) \in (0, 1)$——紧迫度始终在0-1之间
3. **阈值效应**：当 $\Delta V_{\text{gap}} = \theta$ 时，$\text{urgency} = 0.5$——缺口恰好等于阈值时紧迫度为0.5
4. **饱和性**：当 $\Delta V_{\text{gap}} \gg \theta$ 时，$\text{urgency} \to 1$；当 $\Delta V_{\text{gap}} \ll \theta$ 时，$\text{urgency} \to 0$

**参数 $\kappa$ 的影响**：

$\kappa$ 控制S型曲线的陡度——$\kappa$ 越大，曲线越陡，阈值效应越明显：

| $\kappa$ | 曲线特征 | 适用场景 |
|----------|---------|---------|
| 5 | 缓坡过渡 | 缺口变化平缓的环境 |
| 10 | 中等陡度 | 通用场景（默认值） |
| 20 | 陡峭切换 | 缺口突变频繁的环境 |

**参数 $\theta$ 的影响**：

$\theta$ 控制紧迫度的触发阈值——$\theta$ 越小，越早触发紧迫感：

| $\theta$ | 触发时机 | 适用场景 |
|----------|---------|---------|
| 0.3 | 缺口较小时就紧迫 | 保守策略，提前响应 |
| 0.5 | 缺口中等时紧迫 | 通用场景（默认值） |
| 0.7 | 缺口较大时才紧迫 | 激进策略，延迟响应 |

**紧迫度与场景的交互**：

在不同场景下，紧迫度的参数可以自适应调整：

- **紧急场景**：$\theta = 0.3$（更早触发紧迫感），$\kappa = 20$（更陡峭的切换）
- **稳定场景**：$\theta = 0.7$（延迟触发紧迫感），$\kappa = 5$（更平缓的过渡）
- **紧张场景**：$\theta = 0.5$（默认阈值），$\kappa = 10$（默认陡度）

**紧迫度的边际分析**：

紧迫度对缺口的导数为：

$$\frac{\partial \text{urgency}}{\partial \Delta V_{\text{gap}}} = \kappa \cdot \text{urgency} \cdot (1 - \text{urgency}) \tag{A37.3-derivative}$$

当 $\text{urgency} = 0.5$（即 $\Delta V_{\text{gap}} = \theta$）时，导数最大——在阈值附近，紧迫度对缺口变化最敏感。这意味着系统在缺口接近阈值时反应最灵敏，而在缺口远低于或远高于阈值时反应较迟钝。

- **输入**：驱力加权缺口 $W \cdot \Delta V_{\text{gap}}(\text{task})$，参数 $\kappa, \theta$
- **输出**：$\text{urgency}(\text{task}) \in (0, 1)$
- **约束**：$\kappa > 0$；$\theta > 0$

> **白话翻译**：紧迫度就是"火烧眉毛程度"——用S型曲线量化，缺口小的时候不紧迫（曲线平缓），缺口大到阈值时突然紧迫（曲线陡升），缺口非常大时完全紧迫（曲线饱和）。阈值θ决定"什么时候开始急"，陡度κ决定"急得多快"。

---

### 第四部分：可行性

#### 2.4 可行性（A37.4）

可行性度量任务在当前资源条件下能否执行：

$$\text{feasibility}(\text{task}) = \frac{\text{resource\_available}}{\text{resource\_required}(\text{task})} \tag{A37.4}$$

其中：
- $\text{resource\_available}$：当前可用资源（CPU、内存、时间的可用量）
- $\text{resource\_required}(\text{task})$：任务所需的资源量

**可行性截断**：

当 $\text{resource\_available} \geq \text{resource\_required}$ 时，$\text{feasibility} \geq 1$，截断为1：

$$\text{feasibility}(\text{task}) = \min\left(1, \frac{\text{resource\_available}}{\text{resource\_required}(\text{task})}\right) \tag{A37.4-capped}$$

当 $\text{resource\_available} < \text{resource\_required}$ 时，$\text{feasibility} < 1$——资源不足，任务可行性降低。

**多维资源的可行性**：

任务可能同时需要CPU、内存和时间三种资源。多维可行性取各维度的最小值（短板效应）：

$$\text{feasibility}(\text{task}) = \min\left(\frac{\text{cpu\_available}}{\text{cpu\_required}}, \frac{\text{mem\_available}}{\text{mem\_required}}, \frac{\text{time\_available}}{\text{time\_required}}\right) \tag{A37.4-multi}$$

短板效应确保任务只有在所有资源维度都充足时才被认为高度可行——任何一个维度的资源不足都会降低可行性。

**可行性的特殊值**：

| feasibility值 | 含义 | 处理策略 |
|--------------|------|---------|
| 1.0 | 资源完全充足 | 正常执行 |
| 0.5-1.0 | 资源基本充足 | 正常执行，但需监控 |
| 0.2-0.5 | 资源紧张 | 降低优先级，等待资源释放 |
| 0.0-0.2 | 资源严重不足 | 冻结任务，等待资源 |
| 0.0 | 某维度资源完全不可用 | 冻结任务 |

**可行性与任务类型的交互**：

不同类型的任务对资源的需求不同：

| 任务类型 | CPU需求 | 内存需求 | 时间需求 | 短板维度 |
|---------|---------|---------|---------|---------|
| model_compress | 高 | 高 | 中 | CPU/内存 |
| explore_unknown | 中 | 低 | 高 | 时间 |
| stabilize | 高 | 中 | 低 | CPU |
| state_monitor | 低 | 低 | 低 | 无（通常可行） |

state_monitor的可行性通常接近1.0——它的资源需求很低，几乎在任何条件下都可执行。model_compress的可行性可能较低——它需要大量CPU和内存，在资源紧张时可能不可行。

- **输入**：任务 task，可用资源 (cpu_available, mem_available, time_available)
- **输出**：$\text{feasibility}(\text{task}) \in [0, 1]$
- **约束**：各维度资源量非负

> **白话翻译**：可行性就是"能不能做"——看当前资源够不够。CPU够不够、内存够不够、时间够不够，三个维度取最差的那个（短板效应）。资源完全够就是1.0，完全不够就是0.0。监控类任务几乎总是可行的（资源需求低），模型压缩类任务在资源紧张时可能不可行。

---

### 第五部分：成本

#### 2.5 成本（A37.5）

成本度量任务执行所需的资源消耗：

$$\text{cost}(\text{task}) = \text{cpu\_cost}(\text{task}) + \text{mem\_cost}(\text{task}) + \text{time\_cost}(\text{task}) \tag{A37.5}$$

**各维度成本的定义**：

$$\text{cpu\_cost}(\text{task}) = c_{\text{cpu}} \cdot \text{cpu\_units}(\text{task}) \tag{A37.5-cpu}$$

$$\text{mem\_cost}(\text{task}) = c_{\text{mem}} \cdot \text{mem\_units}(\text{task}) \tag{A37.5-mem}$$

$$\text{time\_cost}(\text{task}) = c_{\text{time}} \cdot \text{time\_units}(\text{task}) \tag{A37.5-time}$$

其中 $c_{\text{cpu}}, c_{\text{mem}}, c_{\text{time}}$ 为各维度的单位成本系数，$\text{cpu\_units}, \text{mem\_units}, \text{time\_units}$ 为任务在各维度的资源消耗量。

**单位成本系数的默认值**：

| 维度 | 单位成本系数 | 说明 |
|------|------------|------|
| CPU | $c_{\text{cpu}} = 0.4$ | CPU是主要瓶颈，成本系数最高 |
| 内存 | $c_{\text{mem}} = 0.3$ | 内存是次要瓶颈 |
| 时间 | $c_{\text{time}} = 0.3$ | 时间成本与场景相关 |

**成本与任务类型的关系**：

各类型任务的成本系数已在阶段33-35中定义（引用A33.3-cost, A34.3, A35.3）。综合得分中的成本项使用这些预定义的成本系数：

| 任务类型 | 基础成本 $c_{\text{type}}$ | 成本构成 |
|---------|--------------------------|---------|
| stabilize | 0.8 | CPU密集（快速响应） |
| model_compress | 0.8 | CPU+内存密集（搜索+压缩） |
| explore_unknown | 0.7 | 时间密集（探索需要时间） |
| improve_accuracy | 0.6 | CPU+时间中等 |
| unknown_eliminate | 0.6 | CPU中等 |
| optimize_compute | 0.5 | CPU中等 |
| anomaly_detect | 0.5 | CPU中等 |
| test_feature | 0.4 | 资源需求较低 |
| expand_coverage | 0.5 | 时间中等 |
| mine_pattern | 0.3 | 资源需求低 |
| cache_optimize | 0.2 | 资源需求很低 |
| risk_predict | 0.3 | 资源需求低 |
| state_monitor | 0.2 | 资源需求很低 |
| reduce_frequency | 0.1 | 资源需求极低 |

**成本的归一化**：

成本值需要归一化到 $[0, 1]$ 范围，以便与得分的其他项可比：

$$\text{cost}_{\text{norm}}(\text{task}) = \frac{\text{cost}(\text{task})}{\text{cost}_{\max}} \tag{A37.5-norm}$$

其中 $\text{cost}_{\max}$ 为所有任务中的最大成本值。归一化后，成本项的范围为 $[0, 1]$，与紧迫度和可行性在同一量纲。

- **输入**：任务 task，成本系数 $c_{\text{cpu}}, c_{\text{mem}}, c_{\text{time}}$
- **输出**：$\text{cost}(\text{task}) \in \mathbb{R}_{\geq 0}$
- **约束**：各维度成本非负；归一化后 $\in [0, 1]$

> **白话翻译**：成本就是"做一次花多少"——CPU花多少、内存花多少、时间花多少，三样加起来就是总成本。CPU最贵（系数0.4），内存和时间次之（各0.3）。模型压缩和稳定化最贵（0.8），状态监控和频率降低最便宜（0.1-0.2）。成本在总分里是扣分的——越贵扣越多。

---

### 第六部分：得分归一化

#### 2.6 得分归一化（A37.6）

综合得分归一化到 $[0, 1]$ 范围，便于跨任务比较和阈值设定：

$$\text{Score}_{\text{norm}}(\text{task}) = \frac{\text{Score}(\text{task}) - \text{Score}_{\min}}{\text{Score}_{\max} - \text{Score}_{\min}} \tag{A37.6}$$

其中：
- $\text{Score}_{\min} = \min_{\text{task} \in \text{pool}} \text{Score}(\text{task})$：池中最低得分
- $\text{Score}_{\max} = \max_{\text{task} \in \text{pool}} \text{Score}(\text{task})$：池中最高得分

**归一化的边界处理**：

当 $\text{Score}_{\max} = \text{Score}_{\min}$ 时（所有任务得分相同），归一化得分定义为0.5：

$$\text{Score}_{\text{norm}}(\text{task}) = \begin{cases} \frac{\text{Score} - \text{Score}_{\min}}{\text{Score}_{\max} - \text{Score}_{\min}} & \text{if } \text{Score}_{\max} > \text{Score}_{\min} \\ 0.5 & \text{if } \text{Score}_{\max} = \text{Score}_{\min} \end{cases} \tag{A37.6-edge}$$

**归一化得分的性质**：

1. **有界性**：$\text{Score}_{\text{norm}} \in [0, 1]$
2. **保序性**：如果 $\text{Score}(\text{task}_i) > \text{Score}(\text{task}_j)$，则 $\text{Score}_{\text{norm}}(\text{task}_i) > \text{Score}_{\text{norm}}(\text{task}_j)$——归一化不改变排序
3. **相对性**：归一化得分是相对于当前池中其他任务的相对得分——同一个任务在不同池中的归一化得分可能不同

**归一化得分的阈值设定**：

归一化得分为后续的决策机制提供了统一的阈值基础：

| 阈值 | 含义 | 用途 |
|------|------|------|
| 0.8 | 高分任务 | 利用择优的候选（引用A38.1） |
| 0.3 | 低分任务 | 探索随机的候选（引用A38.2） |
| 0.1 | 极低分任务 | 淘汰候选（引用A36.4） |

**得分分布的统计分析**：

归一化后，可以计算池中任务得分的统计量：

1. **均值**：$\bar{S} = \frac{1}{N}\sum_i \text{Score}_{\text{norm}}(\text{task}_i)$——反映池的整体质量
2. **方差**：$\sigma_S^2 = \frac{1}{N}\sum_i (\text{Score}_{\text{norm}}(\text{task}_i) - \bar{S})^2$——反映得分的区分度
3. **偏度**：$\gamma_S = \frac{1}{N\sigma_S^3}\sum_i (\text{Score}_{\text{norm}}(\text{task}_i) - \bar{S})^3$——反映得分分布的不对称性

方差过低意味着所有任务得分相近，难以区分优先级——此时应增大 $\kappa$ 或调整权重系数以增强区分度。

- **输入**：所有任务的原始得分 $\{\text{Score}(\text{task}_i)\}$
- **输出**：归一化得分 $\text{Score}_{\text{norm}}(\text{task}) \in [0, 1]$
- **约束**：保序性；边界处理

> **白话翻译**：得分归一化就是"百分制换算"——把原始分换算到0-1范围，方便比较和设阈值。最高分变成1.0，最低分变成0.0，中间的按比例换算。归一化不改变排名（保序性），但让分数更容易理解——0.8以上就是高分，0.3以下就是低分。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A37.1 | Score(task) = W·ΔV_gap + α·urgency + β·feasibility - γ·cost | 任务、权重、系数 | 实数得分 | α,β,γ ≥ 0; W > 0 |
| A37.2 | W·ΔV_gap = W_safe·ΔV_safe + W_gain·ΔV_gain + W_eff·ΔV_eff | 任务、缺口、动态权重 | 非负实数 | 权重非负且和为1 |
| A37.3 | urgency(task) = 1/(1 + exp(-κ·(ΔV_gap - θ))) | 缺口值、κ、θ | (0,1) | κ > 0; θ > 0 |
| A37.4 | feasibility(task) = min(resource_available / resource_required) | 任务、可用资源 | [0,1] | 各维度资源非负 |
| A37.5 | cost(task) = cpu_cost + mem_cost + time_cost | 任务、成本系数 | 非负实数 | 各维度成本非负 |
| A37.6 | Score_norm = (Score - Score_min) / (Score_max - Score_min) | 所有任务得分 | [0,1] | 保序性；边界处理 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np

# 引用阶段36的数据类
from stage_036 import DynamicTaskPool, PooledTask


# ============================================================
# 得分计算参数
# ============================================================
@dataclass(frozen=True)
class ScoringParams:
    """得分计算参数（A37.1）。

    Attributes:
        alpha: 紧迫度权重
        beta: 可行性权重
        gamma: 成本权重
        kappa: 紧迫度S型曲线陡度
        theta: 紧迫度阈值
        cpu_cost_coeff: CPU单位成本系数
        mem_cost_coeff: 内存单位成本系数
        time_cost_coeff: 时间单位成本系数
    """
    alpha: float = 0.3
    beta: float = 0.2
    gamma: float = 0.5
    kappa: float = 10.0
    theta: float = 0.5
    cpu_cost_coeff: float = 0.4
    mem_cost_coeff: float = 0.3
    time_cost_coeff: float = 0.3

    def __post_init__(self):
        if self.alpha < 0 or self.beta < 0 or self.gamma < 0:
            raise ValueError("权重系数必须非负")
        if self.kappa <= 0 or self.theta <= 0:
            raise ValueError("κ和θ必须为正")


# ============================================================
# 任务得分详情
# ============================================================
@dataclass(frozen=True)
class TaskScoreDetail:
    """任务得分的详细分解（A37.1）。

    Attributes:
        task: 被评分的任务
        drive_weighted_gap: 驱力加权缺口项（A37.2）
        urgency: 紧迫度（A37.3）
        feasibility: 可行性（A37.4）
        cost: 成本（A37.5）
        raw_score: 原始得分
        normalized_score: 归一化得分（A37.6）
    """
    task: PooledTask
    drive_weighted_gap: float
    urgency: float
    feasibility: float
    cost: float
    raw_score: float
    normalized_score: float

    @property
    def score_breakdown(self) -> dict[str, float]:
        """得分分解。"""
        return {
            "drive_weighted_gap": self.drive_weighted_gap,
            "urgency_contribution": 0.3 * self.urgency,
            "feasibility_contribution": 0.2 * self.feasibility,
            "cost_penalty": -0.5 * self.cost,
            "raw_score": self.raw_score,
            "normalized_score": self.normalized_score,
        }


# ============================================================
# 阶段37 数据类：任务评分器
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class TaskScorer:
    """任务评分器的完整描述。

    组合阶段36的DynamicTaskPool，在任务池的基础上增加
    综合得分计算能力。

    Attributes:
        pool: 阶段36的动态任务池（组合字段）
        params: 得分计算参数
        score_details: 所有任务的得分详情
        score_min: 池中最低原始得分
        score_max: 池中最高原始得分
        score_mean: 池中平均归一化得分
        score_variance: 池中归一化得分方差

    Invariants:
        - score_min <= score_max
        - 所有归一化得分 ∈ [0, 1]
    """
    pool: DynamicTaskPool
    params: ScoringParams
    score_details: tuple[TaskScoreDetail, ...]
    score_min: float
    score_max: float
    score_mean: float
    score_variance: float

    def __post_init__(self):
        if self.score_min > self.score_max:
            raise ValueError(
                f"最低得分 {self.score_min} 不能超过最高得分 {self.score_max}"
            )

    @property
    def ranked_tasks(self) -> list[TaskScoreDetail]:
        """按归一化得分降序排列的任务。"""
        return sorted(
            self.score_details,
            key=lambda d: d.normalized_score,
            reverse=True,
        )

    @property
    def high_score_tasks(self) -> list[TaskScoreDetail]:
        """高分任务（归一化得分 > 0.8）。"""
        return [d for d in self.score_details if d.normalized_score > 0.8]

    @property
    def low_score_tasks(self) -> list[TaskScoreDetail]:
        """低分任务（归一化得分 < 0.3）。"""
        return [d for d in self.score_details if d.normalized_score < 0.3]

    @property
    def score_discrimination(self) -> float:
        """得分区分度（方差越大区分度越高）。"""
        return self.score_variance


# ============================================================
# 阶段37 核心接口：任务评分器
# ============================================================
class TaskScorerEngine:
    """阶段37的核心计算接口。

    实现任务综合得分的完整计算，包括驱力加权、紧迫度、
    可行性、成本和归一化。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 权重系数非负
        - 紧迫度参数为正
        - 归一化保序
    """

    def compute_drive_weighted_gap(
        self,
        task: PooledTask,
        gaps: tuple[float, float, float],
        weights: tuple[float, float, float]
    ) -> float:
        """计算驱力加权缺口项（A37.2）。

        Args:
            task: 被评分的任务
            gaps: 三大缺口 (ΔV_safe, ΔV_gain, ΔV_eff)
            weights: 动态驱力权重 (W_safe, W_gain, W_eff)

        Returns:
            驱力加权缺口值 >= 0

        Raises:
            ValueError: 当 weights 和不等于1或 gaps 包含负值时

        Examples:
            >>> from stage_036 import PooledTask, TaskStatus
            >>> from stage_033 import SafetyTask, SafetyTaskType, SafetyTaskTarget
            >>> task = PooledTask(
            ...     task=SafetyTask(SafetyTaskType.STATE_MONITOR,
            ...         SafetyTaskTarget.SYSTEM, 0.5, 0.2, 0.3),
            ...     status=TaskStatus.ACTIVE, priority=0.5
            ... )
            >>> engine = TaskScorerEngine()
            >>> gap = engine.compute_drive_weighted_gap(task, (0.5, 0.3, 0.2),
            ...     (0.4, 0.3, 0.3))
            >>> gap >= 0
            True
        """
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"权重之和必须等于1，收到 {sum(weights)}")
        for i, v in enumerate(gaps):
            if v < 0:
                raise ValueError(f"缺口第 {i} 分量必须 ≥ 0，收到 {v}")
        # 根据任务类型选择对应缺口
        category = task.task_type_category
        if category == "safe":
            return gaps[0]
        elif category == "gain":
            return gaps[1]
        else:
            return gaps[2]

    def compute_urgency(
        self,
        drive_weighted_gap: float,
        kappa: float = 10.0,
        theta: float = 0.5
    ) -> float:
        """计算紧迫度（A37.3）。

        Args:
            drive_weighted_gap: 驱力加权缺口值
            kappa: S型曲线陡度 > 0
            theta: 紧迫度阈值 > 0

        Returns:
            紧迫度 ∈ (0, 1)

        Raises:
            ValueError: 当 kappa <= 0 或 theta <= 0 时

        Examples:
            >>> engine = TaskScorerEngine()
            >>> u = engine.compute_urgency(0.6, kappa=10.0, theta=0.5)
            >>> 0.0 < u < 1.0
            True
        """
        if kappa <= 0:
            raise ValueError(f"kappa 必须 > 0，收到 {kappa}")
        if theta <= 0:
            raise ValueError(f"theta 必须 > 0，收到 {theta}")
        return float(1.0 / (1.0 + np.exp(-kappa * (drive_weighted_gap - theta))))

    def compute_feasibility(
        self,
        task: PooledTask,
        cpu_available: float,
        mem_available: float,
        time_available: float
    ) -> float:
        """计算可行性（A37.4）。

        采用短板效应：可行性 = min(可用CPU/需求CPU, 可用内存/需求内存, 可用时间/需求时间)。

        Args:
            task: 被评分的任务
            cpu_available: 可用CPU资源 >= 0
            mem_available: 可用内存资源 >= 0
            time_available: 可用时间资源 >= 0

        Returns:
            可行性 ∈ [0, 1]

        Raises:
            ValueError: 当任一可用资源 < 0 时

        Examples:
            >>> from stage_036 import PooledTask, TaskStatus
            >>> from stage_033 import SafetyTask, SafetyTaskType, SafetyTaskTarget
            >>> task = PooledTask(
            ...     task=SafetyTask(SafetyTaskType.STATE_MONITOR,
            ...         SafetyTaskTarget.SYSTEM, 0.5, 0.2, 0.3),
            ...     status=TaskStatus.ACTIVE, priority=0.5
            ... )
            >>> engine = TaskScorerEngine()
            >>> f = engine.compute_feasibility(task, 0.8, 0.6, 0.9)
            >>> 0.0 <= f <= 1.0
            True
        """
        for name, val in [("cpu_available", cpu_available),
                          ("mem_available", mem_available),
                          ("time_available", time_available)]:
            if val < 0:
                raise ValueError(f"{name} 必须 ≥ 0，收到 {val}")
        # 从任务成本推导资源需求（简化：假设成本均匀分配）
        est_cost = getattr(task.task, "estimated_cost", 0.3)
        cpu_need = est_cost * 0.4
        mem_need = est_cost * 0.3
        time_need = est_cost * 0.3
        ratios = []
        if cpu_need > 0:
            ratios.append(cpu_available / cpu_need)
        if mem_need > 0:
            ratios.append(mem_available / mem_need)
        if time_need > 0:
            ratios.append(time_available / time_need)
        if not ratios:
            return 1.0
        return float(min(1.0, min(ratios)))

    def compute_cost(
        self,
        task: PooledTask,
        params: ScoringParams
    ) -> float:
        """计算成本（A37.5）。

        Args:
            task: 被评分的任务
            params: 得分计算参数

        Returns:
            成本值（归一化后 ∈ [0, 1]）

        Examples:
            >>> from stage_036 import PooledTask, TaskStatus
            >>> from stage_033 import SafetyTask, SafetyTaskType, SafetyTaskTarget
            >>> task = PooledTask(
            ...     task=SafetyTask(SafetyTaskType.STATE_MONITOR,
            ...         SafetyTaskTarget.SYSTEM, 0.5, 0.2, 0.3),
            ...     status=TaskStatus.ACTIVE, priority=0.5
            ... )
            >>> engine = TaskScorerEngine()
            >>> c = engine.compute_cost(task, ScoringParams())
            >>> c >= 0
            True
        """
        est_cost = getattr(task.task, "estimated_cost", 0.3)
        # 成本归一化：假设最大成本为1.0
        return float(min(1.0, est_cost))

    def compute_score(
        self,
        task: PooledTask,
        gaps: tuple[float, float, float],
        weights: tuple[float, float, float],
        resources: tuple[float, float, float],
        params: ScoringParams
    ) -> float:
        """计算任务综合得分（A37.1）。

        Score = W·ΔV_gap + α·urgency + β·feasibility - γ·cost

        Args:
            task: 被评分的任务
            gaps: 三大缺口 (ΔV_safe, ΔV_gain, ΔV_eff)
            weights: 动态驱力权重 (W_safe, W_gain, W_eff)
            resources: 可用资源 (cpu, mem, time)
            params: 得分计算参数

        Returns:
            原始综合得分

        Examples:
            >>> from stage_036 import PooledTask, TaskStatus
            >>> from stage_033 import SafetyTask, SafetyTaskType, SafetyTaskTarget
            >>> task = PooledTask(
            ...     task=SafetyTask(SafetyTaskType.STATE_MONITOR,
            ...         SafetyTaskTarget.SYSTEM, 0.5, 0.2, 0.3),
            ...     status=TaskStatus.ACTIVE, priority=0.5
            ... )
            >>> engine = TaskScorerEngine()
            >>> s = engine.compute_score(task, (0.5, 0.3, 0.2),
            ...     (0.4, 0.3, 0.3), (0.8, 0.6, 0.9), ScoringParams())
            >>> isinstance(s, float)
            True
        """
        drive_gap = self.compute_drive_weighted_gap(task, gaps, weights)
        urgency = self.compute_urgency(drive_gap, params.kappa, params.theta)
        feasibility = self.compute_feasibility(
            task, resources[0], resources[1], resources[2]
        )
        cost = self.compute_cost(task, params)
        score = (
            drive_gap
            + params.alpha * urgency
            + params.beta * feasibility
            - params.gamma * cost
        )
        return float(score)

    def normalize_scores(
        self,
        scores: list[float]
    ) -> list[float]:
        """得分归一化（A37.6）。

        使用 min-max 归一化到 [0, 1]，保序。

        Args:
            scores: 原始得分列表

        Returns:
            归一化得分列表 ∈ [0, 1]

        Examples:
            >>> engine = TaskScorerEngine()
            >>> normed = engine.normalize_scores([1.0, 2.0, 3.0])
            >>> normed
            [0.0, 0.5, 1.0]
            >>> engine.normalize_scores([5.0, 5.0, 5.0])
            [0.5, 0.5, 0.5]
        """
        if not scores:
            return []
        s_min = min(scores)
        s_max = max(scores)
        if abs(s_max - s_min) < 1e-12:
            return [0.5] * len(scores)
        return [(s - s_min) / (s_max - s_min) for s in scores]

    def score_pool(
        self,
        pool: DynamicTaskPool,
        params: ScoringParams = None
    ) -> TaskScorer:
        """对任务池中所有任务评分。

        Args:
            pool: 动态任务池
            params: 得分计算参数，默认 ScoringParams()

        Returns:
            TaskScorer 实例

        Examples:
            >>> from stage_036 import DynamicTaskPool, PoolSizeConfig
            >>> pool = DynamicTaskPool(
            ...     eff_generator=None, active_tasks=(), pending_tasks=(),
            ...     completed_tasks=(), frozen_tasks=(),
            ...     size_config=PoolSizeConfig(), last_refresh_step=0,
            ...     total_generated=0, total_evicted=0, total_frozen=0
            ... )
            >>> engine = TaskScorerEngine()
            >>> scorer = engine.score_pool(pool)
            >>> scorer.score_mean >= 0
            True
        """
        if params is None:
            params = ScoringParams()
        all_tasks = list(pool.active_tasks) + list(pool.pending_tasks)
        # 默认缺口和权重
        gaps = (0.5, 0.3, 0.2)
        weights = (0.4, 0.3, 0.3)
        resources = (0.8, 0.6, 0.9)
        raw_scores: list[float] = []
        details: list[TaskScoreDetail] = []
        for t in all_tasks:
            score = self.compute_score(t, gaps, weights, resources, params)
            raw_scores.append(score)
        normed_scores = self.normalize_scores(raw_scores)
        for t, raw, norm in zip(all_tasks, raw_scores, normed_scores):
            drive_gap = self.compute_drive_weighted_gap(t, gaps, weights)
            urgency = self.compute_urgency(drive_gap, params.kappa, params.theta)
            feasibility = self.compute_feasibility(
                t, resources[0], resources[1], resources[2]
            )
            cost = self.compute_cost(t, params)
            details.append(TaskScoreDetail(
                task=t,
                drive_weighted_gap=drive_gap,
                urgency=urgency,
                feasibility=feasibility,
                cost=cost,
                raw_score=raw,
                normalized_score=norm,
            ))
        if raw_scores:
            score_min = float(min(raw_scores))
            score_max = float(max(raw_scores))
            score_mean = float(np.mean(normed_scores))
            score_variance = float(np.var(normed_scores))
        else:
            score_min = 0.0
            score_max = 0.0
            score_mean = 0.0
            score_variance = 0.0
        return TaskScorer(
            pool=pool,
            params=params,
            score_details=tuple(details),
            score_min=score_min,
            score_max=score_max,
            score_mean=score_mean,
            score_variance=score_variance,
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `DynamicTaskPool` | ← 阶段36 | 作为任务评分器的任务池组合字段 |
| **输入** | A24.1（全局奖励定义） | ← 阶段24 | 驱力权重用于驱力加权计算 |
| **输入** | A25.1（价值缺口定义） | ← 阶段25 | 缺口值作为得分的核心驱动力 |
| **输入** | A32.5（缺口归一化） | ← 阶段32 | 归一化缺口提供跨驱力的可比尺度 |
| **输出** | `TaskScorer` | → 阶段38 | 评分结果作为双决策机制的输入 |
| **输出** | A37.6（得分归一化） | → 后续阶段 | 归一化得分为决策阈值提供统一基础 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 得分计算 | `jax.numpy.exp`实现S型紧迫度曲线（A37.3）；`jax.vmap`实现批量任务的并行评分（A37.1） |
| NumPy | 统计分析 | `numpy.mean/var/skew`计算得分分布的统计量（A37.6） |
| SciPy | 参数优化 | `scipy.optimize.minimize`优化得分参数（α,β,γ,κ,θ），使得分区分度最大化 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V37.1 | 综合得分计算 | Score = W·ΔV_gap + α·urgency + β·feasibility - γ·cost | 数值验证 |
| V37.2 | 驱力加权正确性 | 安全任务主要使用ΔV_safe，收益任务主要使用ΔV_gain | 任务类型测试 |
| V37.3 | 驱力加权场景敏感性 | 紧急场景安全任务得分最高 | 场景切换测试 |
| V37.4 | 紧迫度S型曲线 | urgency ∈ (0,1)，单调递增，θ处为0.5 | 曲线扫描测试 |
| V37.5 | 紧迫度参数影响 | κ增大曲线变陡，θ减小提前触发 | 参数扫描测试 |
| V37.6 | 紧迫度导数 | θ处导数最大 | 数值微分验证 |
| V37.7 | 可行性短板效应 | feasibility = min(cpu, mem, time) | 多维资源测试 |
| V37.8 | 可行性截断 | feasibility ≤ 1 | 资源充足场景测试 |
| V37.9 | 成本非负 | cost ≥ 0 对所有任务成立 | 全任务扫描 |
| V37.10 | 成本归一化 | cost_norm ∈ [0,1] | 归一化验证 |
| V37.11 | 得分归一化 | Score_norm ∈ [0,1] | 归一化验证 |
| V37.12 | 归一化保序性 | 原始得分排序与归一化得分排序一致 | 排序对比测试 |
| V37.13 | 归一化边界处理 | 所有得分相同时Score_norm=0.5 | 等分测试 |
| V37.14 | TaskScorer组合 | `TaskScorer.pool` 类型为 `DynamicTaskPool` | 类型检查 |
| V37.15 | 得分区分度 | 方差 > 0.01（任务间有区分） | 区分度测试 |
| V37.16 | 非法输入拒绝 | 负权重、非正κ/θ等抛出 `ValueError` | 异常测试 |

---

> **阶段37完** | 阶段37建立了任务综合得分的完整计算体系：四维评分公式（价值缺口+紧迫度+可行性-成本）将多维评估统一为单一得分，驱力加权将动态场景权重引入任务级评分，S型紧迫度曲线实现阈值敏感的紧急度量化，短板效应可行性确保资源约束的严格执行，成本归一化使各维度可比，得分归一化为决策阈值提供统一基础。下一步：阶段38将基于综合得分实现70/30双决策机制，平衡利用与探索。
