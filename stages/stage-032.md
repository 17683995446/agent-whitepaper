# 阶段32：缺口评估算法——从环境状态到缺口量化

> **阶段编号**：S32 | **所属层**：第四层·任务规划层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 缺口评估函数 | "三科成绩单"——从体检报告算出安全、收益、能效三个缺口 |
> | 安全缺口计算 | "危险指数"——异常越多、错误越多、未知越多，缺口越大 |
> | 收益缺口计算 | "知识差距"——覆盖不够、精度不高、新发现不多，缺口越大 |
> | 能效缺口计算 | "浪费指数"——CPU用多了、内存占多了、时间花多了，缺口越大 |
> | 缺口归一化 | "百分制打分"——把不同量纲的缺口统一到0-1范围 |
> | 缺口优先级排序 | "先救火还是先学习"——哪个缺口最大，先处理哪个 |

---

## 1. 第一性原理依据

本阶段从阶段25（缺口定义）和阶段31（环境状态向量）的核心结论出发，建立缺口评估算法。核心依据为：

- **A25.1-A25.5（缺口定义与分解）**（引用阶段25）：价值缺口 $\Delta V = V_{\text{target}} - V_{\text{current}}$，分解为安全缺口 $\Delta V_{\text{safe}}$、收益缺口 $\Delta V_{\text{gain}}$ 和能效缺口 $\Delta V_{\text{eff}}$。缺口评估算法是这些理论公式的工程化实现——从环境状态向量中提取具体的缺口量化值。
- **A25.2（安全缺口）**（引用阶段25）：$\Delta V_{\text{safe}} = -R_{\text{safe}} = \alpha_s \cdot \sum_i w_i \cdot \text{anomaly}_i$，安全缺口由异常信号的加权和决定。缺口评估算法需要从环境状态向量中提取异常信号并计算安全缺口。
- **A25.3（收益缺口）**（引用阶段25）：$\Delta V_{\text{gain}} = V_{\text{gain,target}} - R_{\text{gain}}$，收益缺口由收益目标与当前收益驱力的差值决定。缺口评估算法需要从环境状态向量中提取认知指标并计算收益缺口。
- **A25.4（能效缺口）**（引用阶段25）：$\Delta V_{\text{eff}} = -R_{\text{eff}} = \alpha_e \cdot (\text{cpu\_ratio} + \text{mem\_ratio} + \text{time\_cost} + \text{redundancy})$，能效缺口由资源消耗的加权和决定。缺口评估算法需要从环境状态向量中提取资源指标并计算能效缺口。
- **A31.1（环境状态向量）**（引用阶段31）：$\mathbf{s} = [\mathbf{s}_{\text{sys}}, \mathbf{s}_{\text{log}}, \mathbf{s}_{\text{file}}, \mathbf{s}_{\text{self}}]$，环境状态向量是缺口评估算法的输入——缺口评估从环境状态向量中提取三大缺口的具体数值。

本阶段的核心目标是：**定义缺口评估函数 $\text{gap}(\mathbf{s}) = (\Delta V_{\text{safe}}(\mathbf{s}), \Delta V_{\text{gain}}(\mathbf{s}), \Delta V_{\text{eff}}(\mathbf{s}))$，建立安全缺口、收益缺口和能效缺口的计算公式，实现缺口归一化 $\Delta V_{i,\text{norm}} = \Delta V_i / \Delta V_{i,\max}$，定义缺口优先级排序规则，并封装为GapAssessment数据类，组合EnvironmentStateVector作为环境感知的输入。** 缺口评估算法是从"感知"到"行动"的关键桥梁——它将环境状态向量中的原始指标转化为三大缺口的量化值，为后续的任务生成映射（阶段33-35）提供决策依据。

---

## 2. 核心数学公式

### 第一部分：缺口评估函数

#### 2.1 缺口评估函数（A32.1）

缺口评估函数将环境状态向量映射为三大缺口的元组：

$$\text{gap}(\mathbf{s}) = (\Delta V_{\text{safe}}(\mathbf{s}), \Delta V_{\text{gain}}(\mathbf{s}), \Delta V_{\text{eff}}(\mathbf{s})) \tag{A32.1}$$

其中：
- $\mathbf{s} = [\mathbf{s}_{\text{sys}}, \mathbf{s}_{\text{log}}, \mathbf{s}_{\text{file}}, \mathbf{s}_{\text{self}}]$：环境状态向量（引用阶段31，A31.1）
- $\Delta V_{\text{safe}}(\mathbf{s})$：安全缺口（A32.2）
- $\Delta V_{\text{gain}}(\mathbf{s})$：收益缺口（A32.3）
- $\Delta V_{\text{eff}}(\mathbf{s})$：能效缺口（A32.4）

- **输入**：环境状态向量 $\mathbf{s} \in \mathbb{R}^{16}$
- **输出**：缺口三元组 $(\Delta V_{\text{safe}}, \Delta V_{\text{gain}}, \Delta V_{\text{eff}}) \in \mathbb{R}_{\geq 0}^3$
- **约束**：三大缺口均非负（$\Delta V_i \geq 0$），由A25.6（引用阶段25）保证严格为正

> **白话翻译**：缺口评估函数就是"三科成绩单"——输入是体检报告（环境状态向量），输出是三个分数（安全缺口、收益缺口、能效缺口）。每个分数越高，说明那个方面越需要改善。

**缺口评估函数的性质**：

1. **连续性**：$\text{gap}(\mathbf{s})$ 是 $\mathbf{s}$ 的连续函数——环境状态的微小变化只导致缺口的微小变化
2. **单调性**：异常指标增大时安全缺口增大；认知指标减小时收益缺口增大；资源指标增大时能效缺口增大
3. **非负性**：$\Delta V_i \geq 0$ 对所有 $\mathbf{s}$ 成立（由A25.6保证严格为正）
4. **有界性**：归一化后 $\Delta V_{i,\text{norm}} \in [0, 1]$（A32.5）

**缺口评估与驱力计算的对称性**：

缺口评估函数 $\text{gap}(\mathbf{s})$ 与驱力计算函数（引用阶段27，A27.1）存在对称关系：

$$\Delta V_{\text{safe}} = -R_{\text{safe}}, \quad \Delta V_{\text{eff}} = -R_{\text{eff}}$$

$$\Delta V_{\text{gain}} = V_{\text{gain,target}} - R_{\text{gain}}$$

安全缺口和能效缺口是驱力的直接取负，收益缺口是驱力与目标的差值。这种对称性源于A25.2和A25.4（引用阶段25）的定义——安全驱力和能效驱力的目标值为零，因此缺口就是驱力的负值。

---

### 第二部分：安全缺口计算

#### 2.2 安全缺口计算（A32.2）

安全缺口从环境状态向量的系统层和日志层指标中计算：

$$\Delta V_{\text{safe}} = f_{\text{safe}}(\text{anomaly\_count}, \text{error\_rate}, \text{unknown\_ratio}) \tag{A32.2}$$

其中三个输入分量的定义如下：

**异常计数**：

$$\text{anomaly\_count} = \sum_{i \in \{\text{cpu, mem, disk, net, load, proc}\}} \mathbb{1}[z_i > \theta_{\text{anomaly}}] \tag{A32.2-count}$$

度量系统层标准化指标中超过异常阈值 $\theta_{\text{anomaly}}$ 的分量数。$\theta_{\text{anomaly}} = 2.0$（2倍标准差），对应正态分布下约2.3%的异常率。异常计数直接反映系统偏离正常状态的程度——计数越高，系统越不正常。

**错误率**：

$$\text{error\_rate} = s_{\text{log,errors}} \tag{A32.2-err}$$

直接取日志层的错误率分量（引用阶段31，A31.3-err）。错误率是安全驱力最直接的触发信号——由A21.2-err（引用阶段21），报错日志异常是安全驱力的核心输入。

**未知比率**：

$$\text{unknown\_ratio} = 1 - \frac{N_{\text{known\_states}}}{N_{\text{total\_states}}} \tag{A32.2-unknown}$$

度量当前状态空间中未知状态的比例。$N_{\text{known\_states}}$ 为已知状态数，$N_{\text{total\_states}}$ 为估计的总状态数。未知比率直接对应A21.2-nov（引用阶段21）的未知状态异常——未知越多，安全风险越高。

**安全缺口函数的具体形式**：

$$\Delta V_{\text{safe}} = \alpha_s \cdot \left( w_1 \cdot \text{anomaly\_count} + w_2 \cdot \text{error\_rate} + w_3 \cdot \text{unknown\_ratio} \right) \tag{A32.2-form}$$

其中 $w_1, w_2, w_3 \geq 0$ 为权重，$\sum w_i = 1$。默认权重为 $w_1 = 0.4, w_2 = 0.4, w_3 = 0.2$——异常计数和错误率同等重要且权重较高，未知比率权重较低（因为未知不一定是危险，只是风险因素）。

- **输入**：环境状态向量的系统层和日志层指标
- **输出**：$\Delta V_{\text{safe}} \in \mathbb{R}_{\geq 0}$
- **约束**：$\Delta V_{\text{safe}} > 0$（由A25.6保证）；权重归一化 $\sum w_i = 1$

> **白话翻译**：安全缺口就是"危险指数"——系统异常越多、错误越多、未知越多，缺口越大。三个因素中，异常和错误最直接反映危险，未知只是潜在风险，所以权重较低。

**安全缺口与A25.2的等价性论证**：

由A25.2（引用阶段25），$\Delta V_{\text{safe}} = \alpha_s \cdot \sum_i w_i \cdot \text{anomaly}_i$。A32.2的三个输入分量与A21.2（引用阶段21）的五类异常信号的对应关系为：

| A32.2分量 | 对应A21.2异常类型 | 映射关系 |
|-----------|------------------|---------|
| anomaly\_count | 系统异常 + 突变负载 | 标准化指标超阈值计数 |
| error\_rate | 报错日志 | 直接取日志层错误率 |
| unknown\_ratio | 未知状态 + 风险进程 | 未知状态比例 |

A32.2将A21.2的五类异常信号压缩为三个分量，简化了计算同时保留了核心信息。这种压缩是合理的——系统异常和突变负载都反映系统层面的异常（合并为anomaly\_count），未知状态和风险进程都反映环境的不确定性（合并为unknown\_ratio）。

**安全缺口的动态特性**：

安全缺口在时间上的演化由A25.7（引用阶段25）描述：

$$\frac{\mathrm{d}\Delta V_{\text{safe}}}{\mathrm{d}t} = -\lambda_{\text{safe}} \cdot \text{motivation}_{\text{safe}} + \sigma_{\text{safe}} \cdot \xi_{\text{safe}}(t)$$

安全缺口在动机驱动下收敛（系统采取安全行动减少异常），但环境扰动 $\xi_{\text{safe}}(t)$ 持续引入新的异常，使安全缺口在稳态下波动。

---

### 第三部分：收益缺口计算

#### 2.3 收益缺口计算（A32.3）

收益缺口从环境状态向量的文件层和自身层指标中计算：

$$\Delta V_{\text{gain}} = f_{\text{gain}}(\text{coverage}, \text{accuracy}, \text{novelty}) \tag{A32.3}$$

其中三个输入分量的定义如下：

**认知覆盖度**：

$$\text{coverage} = 1 - \frac{N_{\text{known\_states}}}{N_{\text{total\_states}}} \tag{A32.3-cov}$$

度量状态空间中尚未覆盖的比例。$\text{coverage} = 0$ 表示完全覆盖（全知），$\text{coverage} = 1$ 表示完全未知。直接对应A22.4（引用阶段22）的认知覆盖度 $\delta_{\text{coverage}}$ 的补数。文件层的目录状态熵（A31.4-dir）和增删比（A31.4-nd）是覆盖度的间接指标——文件结构变化可能暴露新的未知领域。

**模型精度**：

$$\text{accuracy} = 1 - s_{\text{self,model\_accuracy}} \tag{A32.3-acc}$$

度量模型精度的不足程度。$s_{\text{self,model\_accuracy}}$ 为自身层的模型精度指标（A31.5-acc），取补数后表示"还有多少精度提升空间"。精度不足意味着系统对环境的理解不够深入，对应A22.3（引用阶段22）的精度提升 $\delta_{\text{accuracy}}$ 的逆指标。

**新颖性**：

$$\text{novelty} = \max(0, s_{\text{log,new\_logs}} - 1.0) + \max(0, s_{\text{file,file\_changes}} - \theta_{\text{change}}) \tag{A32.3-nov}$$

度量环境中新信息出现的程度。新日志频率超过期望值（$> 1$）和文件变更率超过阈值 $\theta_{\text{change}}$ 都表示环境正在产生新信息。新颖性直接对应A22.2（引用阶段22）的新信息维度 $I_{\text{new}}$——新信息越多，收益驱力越大，但收益缺口也越大（因为新信息意味着还有更多未知）。

**收益缺口函数的具体形式**：

$$\Delta V_{\text{gain}} = \alpha_g \cdot \left( w_1 \cdot \text{coverage} + w_2 \cdot \text{accuracy} + w_3 \cdot \text{novelty} \right) \tag{A32.3-form}$$

其中 $w_1, w_2, w_3 \geq 0$ 为权重，$\sum w_i = 1$。默认权重为 $w_1 = 0.4, w_2 = 0.3, w_3 = 0.3$——认知覆盖度权重最高（知识面的广度是最基本的收益缺口），精度和新颖性权重相当。

- **输入**：环境状态向量的文件层和自身层指标
- **输出**：$\Delta V_{\text{gain}} \in \mathbb{R}_{\geq 0}$
- **约束**：$\Delta V_{\text{gain}} > 0$（由A25.6保证）；权重归一化 $\sum w_i = 1$

> **白话翻译**：收益缺口就是"知识差距"——知识面不够广（覆盖度低）、理解不够深（精度低）、新发现不多（新颖性低），缺口越大。三个因素中，知识面的广度最基本——不知道的越多，学得越不够。

**收益缺口与A25.3的等价性论证**：

由A25.3（引用阶段25），$\Delta V_{\text{gain}} = V_{\text{gain,target}} - R_{\text{gain}}$。由A22.1（引用阶段22），$R_{\text{gain}} = \alpha_g \cdot (I_{\text{new}} + \delta_{\text{accuracy}} + \delta_{\text{coverage}})$。收益缺口与收益驱力互补——收益驱力度量"已经获得的认知收益"，收益缺口度量"尚未获得的认知收益"。两者之和为常数（等于收益目标值），因此收益缺口增大等价于收益驱力减小。

**收益缺口的稳态特性**：

由定理25.1（引用阶段25），$\Delta V_{\text{gain}} \geq \alpha_g \cdot (1 - D_{\max}/D_{\text{env}}) > 0$。收益缺口有严格正下界——因为环境复杂度 $D_{\text{env}}$ 永远大于系统表征容量 $D_{\max}$。在稳态下，收益缺口在"学习收敛"和"环境变化"的动态平衡中波动。

---

### 第四部分：能效缺口计算

#### 2.4 能效缺口计算（A32.4）

能效缺口从环境状态向量的系统层和自身层指标中计算：

$$\Delta V_{\text{eff}} = f_{\text{eff}}(\text{cpu}, \text{mem}, \text{time}, \text{redundancy}) \tag{A32.4}$$

其中四个输入分量直接取自环境状态向量：

**CPU消耗**：

$$\text{cpu} = s_{\text{sys,cpu}} \tag{A32.4-cpu}$$

直接取系统层的CPU占用率（A31.2-cpu）。对应A23.2（引用阶段23）的CPU占用率。

**内存消耗**：

$$\text{mem} = s_{\text{sys,mem}} \tag{A32.4-mem}$$

直接取系统层的内存占用率（A31.2-mem）。对应A23.3（引用阶段23）的内存占用率。

**时间消耗**：

$$\text{time} = s_{\text{self,compute\_cost}} \tag{A32.4-time}$$

直接取自身层的计算成本比（A31.5-cost）。对应A23.4（引用阶段23）的推理耗时比。

**计算冗余**：

$$\text{redundancy} = 1 - \frac{I(\Psi; O)}{C_{\text{computational}}} \tag{A32.4-red}$$

度量计算资源的信息产出效率。$I(\Psi; O)$ 为互信息（引用阶段7，A7.5），$C_{\text{computational}}$ 为计算量。对应A23.5（引用阶段23）的计算冗余度。在实践中，redundancy可从自身层的误差分布和计算成本间接估计。

**能效缺口函数的具体形式**：

$$\Delta V_{\text{eff}} = \alpha_e \cdot (w_1 \cdot \text{cpu} + w_2 \cdot \text{mem} + w_3 \cdot \text{time} + w_4 \cdot \text{redundancy}) \tag{A32.4-form}$$

其中 $w_1, w_2, w_3, w_4 \geq 0$ 为权重，$\sum w_i = 1$。默认权重为 $w_1 = 0.3, w_2 = 0.2, w_3 = 0.2, w_4 = 0.3$——CPU和冗余度权重较高（CPU是最直接的成本，冗余度是最根本的效率问题），内存和时间权重相当。

- **输入**：环境状态向量的系统层和自身层指标
- **输出**：$\Delta V_{\text{eff}} \in \mathbb{R}_{\geq 0}$
- **约束**：$\Delta V_{\text{eff}} > 0$（由A25.6保证）；权重归一化 $\sum w_i = 1$

> **白话翻译**：能效缺口就是"浪费指数"——CPU用多了、内存占多了、时间花多了、算力浪费了，缺口越大。四个因素中，CPU和冗余度最关键——CPU是最显性的成本，冗余度是最根本的效率瓶颈。

**能效缺口与A25.4的等价性**：

A32.4与A25.4（引用阶段25）完全一致——$\Delta V_{\text{eff}} = -R_{\text{eff}} = \alpha_e \cdot (\text{cpu\_ratio} + \text{mem\_ratio} + \text{time\_cost} + \text{redundancy})$。能效缺口是能效驱力的直接取负，两者一一对应。

**能效缺口的资源约束**：

由定理25.1（引用阶段25），$\Delta V_{\text{eff}} \geq \alpha_e \cdot (1 - \eta_{\max}) > 0$。能效缺口有严格正下界——因为信息传输效率 $\eta_{\max} < 1$（物理信道容量限制）。即使系统运行在最优状态下，能效缺口也不会为零——因为任何计算都需要正的资源消耗。

---

### 第五部分：缺口归一化

#### 2.5 缺口归一化（A32.5）

三大缺口的量纲和数值范围不同，需要归一化到统一尺度以便比较和排序：

$$\Delta V_{i,\text{norm}} = \frac{\Delta V_i}{\Delta V_{i,\max}} \tag{A32.5}$$

其中 $\Delta V_{i,\max}$ 为第 $i$ 类缺口的历史最大值或理论最大值。

- **输入**：缺口值 $\Delta V_i$，最大值 $\Delta V_{i,\max}$
- **输出**：$\Delta V_{i,\text{norm}} \in [0, 1]$
- **约束**：$\Delta V_{i,\max} > 0$

> **白话翻译**：缺口归一化就是"百分制打分"——安全缺口可能是0.5，收益缺口可能是3.0，能效缺口可能是1.2，量纲不同没法直接比。归一化后都变成0到1之间的分数，哪个最接近1，哪个最需要优先处理。

**最大值的确定策略**：

| 策略 | 公式 | 特点 |
|------|------|------|
| 历史最大值 | $\Delta V_{i,\max} = \max_{t \in T_{\text{hist}}} \Delta V_i(t)$ | 自适应，但可能被极端值拉高 |
| 滑动窗口最大值 | $\Delta V_{i,\max} = \max_{t \in [t_0 - W, t_0]} \Delta V_i(t)$ | 适应近期变化，避免远古极端值 |
| 理论最大值 | $\Delta V_{i,\max} = \alpha_i \cdot N_{\text{anomaly}}$（安全）等 | 稳定，但可能过于保守 |
| 百分位最大值 | $\Delta V_{i,\max} = P_{99}(\Delta V_i)$ | 抗极端值，推荐默认策略 |

默认采用百分位最大值策略——使用历史缺口的99百分位数作为最大值，既避免极端值的干扰，又保证归一化值在正常情况下不超过1。

**归一化的边界处理**：

当 $\Delta V_i > \Delta V_{i,\max}$ 时（缺口超过历史最大值），归一化值超过1。此时截断为1：

$$\Delta V_{i,\text{norm}} = \min\left(1, \frac{\Delta V_i}{\Delta V_{i,\max}}\right) \tag{A32.5-capped}$$

截断保证归一化值始终在 $[0, 1]$ 范围内，同时标记了"前所未有"的极端缺口。

**归一化缺口的动态更新**：

$\Delta V_{i,\max}$ 需要定期更新以适应系统状态的变化。更新策略与A31.6-mu（引用阶段31）的EMA更新一致：

$$\Delta V_{i,\max}(t) = \max\left(\Delta V_{i,\max}(t-1), \beta_{\max} \cdot \Delta V_i(t) + (1 - \beta_{\max}) \cdot \Delta V_{i,\max}(t-1)\right) \tag{A32.5-update}$$

取max确保最大值只增不减（或缓慢减小），避免归一化值的突然跳变。

---

### 第六部分：缺口优先级排序

#### 2.6 缺口优先级排序（A32.6）

缺口优先级排序决定系统应优先处理哪个缺口：

$$\text{priority}(i) = \text{rank}(\Delta V_{i,\text{norm}}) \tag{A32.6}$$

按归一化缺口从大到小排序，缺口最大的驱力优先级最高。

**优先级排序规则**：

| 优先级 | 条件 | 行动策略 |
|--------|------|---------|
| 安全优先 | $\Delta V_{\text{safe,norm}} > \max(\Delta V_{\text{gain,norm}}, \Delta V_{\text{eff,norm}})$ | 优先执行安全任务 |
| 收益优先 | $\Delta V_{\text{gain,norm}} > \max(\Delta V_{\text{safe,norm}}, \Delta V_{\text{eff,norm}})$ | 优先执行收益任务 |
| 能效优先 | $\Delta V_{\text{eff,norm}} > \max(\Delta V_{\text{safe,norm}}, \Delta V_{\text{gain,norm}})$ | 优先执行能效任务 |

- **输入**：归一化缺口三元组 $(\Delta V_{\text{safe,norm}}, \Delta V_{\text{gain,norm}}, \Delta V_{\text{eff,norm}})$
- **输出**：优先级排序 $[\text{priority}_1, \text{priority}_2, \text{priority}_3]$
- **约束**：排序结果为 $\{\text{safe, gain, eff}\}$ 的排列

> **白话翻译**：缺口优先级排序就是"先救火还是先学习"——哪个缺口最大，先处理哪个。安全缺口最大就先救火，收益缺口最大就先学习，能效缺口最大就先优化。

**优先级排序与场景切换的对应**：

缺口优先级排序与A26.4（引用阶段26）的场景切换规则完全对应：

| 缺口优先级 | 对应场景 | 切换条件 |
|-----------|---------|---------|
| 安全优先 | urgent | $\Delta V_{\text{safe,norm}}$ 最大且 $> \theta_{\text{urgent}}$ |
| 收益优先 | stable | $\Delta V_{\text{gain,norm}}$ 最大且安全缺口不大 |
| 能效优先 | tight | $\Delta V_{\text{eff,norm}}$ 最大且安全缺口不大 |

这种对应不是巧合——场景切换的底层驱动力正是缺口的大小关系。缺口评估算法将环境状态量化为缺口值，场景切换规则根据缺口大小关系选择场景，两者构成"评估-决策"的完整链条。

**优先级排序的稳定性**：

为避免优先级频繁切换（类似于A28.5的防抖动机制），引入优先级驻留时间：

$$\text{priority}(i, t) = \text{priority}(i, t-1) \quad \text{if } t - t_{\text{last\_switch}} < T_{\text{priority\_debounce}} \tag{A32.6-debounce}$$

其中 $T_{\text{priority\_debounce}}$ 为优先级切换的最小驻留时间（默认10步）。只有当新优先级持续 $T_{\text{priority\_debounce}}$ 步更优时，才切换优先级。

**复合缺口的处理**：

当两个或三个缺口同时较大时（如 $\Delta V_{\text{safe,norm}} \approx \Delta V_{\text{eff,norm}} \gg \Delta V_{\text{gain,norm}}$），系统需要同时处理多个缺口。复合缺口的处理策略：

1. **安全优先原则**：安全缺口始终具有最高优先级（即使其他缺口略大），因为安全是生存前提
2. **资源分配**：按缺口比例分配行动资源——$\text{resource}_i \propto \Delta V_{i,\text{norm}}$
3. **任务混合**：生成的任务列表包含多个缺口的任务，按优先级排序执行

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A32.1 | $\text{gap}(\mathbf{s}) = (\Delta V_{\text{safe}}, \Delta V_{\text{gain}}, \Delta V_{\text{eff}})$ | $\mathbf{s} \in \mathbb{R}^{16}$ | 缺口三元组 $\in \mathbb{R}_{\geq 0}^3$ | 三大缺口均非负 |
| A32.2 | $\Delta V_{\text{safe}} = f_{\text{safe}}(\text{anomaly\_count}, \text{error\_rate}, \text{unknown\_ratio})$ | 系统层+日志层指标 | $\Delta V_{\text{safe}} \geq 0$ | 权重归一化 |
| A32.3 | $\Delta V_{\text{gain}} = f_{\text{gain}}(\text{coverage}, \text{accuracy}, \text{novelty})$ | 文件层+自身层指标 | $\Delta V_{\text{gain}} \geq 0$ | 权重归一化 |
| A32.4 | $\Delta V_{\text{eff}} = f_{\text{eff}}(\text{cpu}, \text{mem}, \text{time}, \text{redundancy})$ | 系统层+自身层指标 | $\Delta V_{\text{eff}} \geq 0$ | 权重归一化 |
| A32.5 | $\Delta V_{i,\text{norm}} = \Delta V_i / \Delta V_{i,\max}$ | 缺口值、最大值 | $\Delta V_{i,\text{norm}} \in [0, 1]$ | $\Delta V_{i,\max} > 0$ |
| A32.6 | $\text{priority}(i) = \text{rank}(\Delta V_{i,\text{norm}})$ | 归一化缺口三元组 | 优先级排序 | 安全优先原则 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# 引用阶段31的数据类
from stage_031 import EnvironmentStateVector


# ============================================================
# 阶段32 数据类：缺口评估
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class GapAssessment:
    """缺口评估的完整描述。

    组合阶段31的EnvironmentStateVector，在环境状态向量的基础上增加
    三大缺口的评估结果。

    Attributes:
        env_state: 阶段31的环境状态向量（组合字段）
        delta_v_safe: 安全缺口（A32.2）
        delta_v_gain: 收益缺口（A32.3）
        delta_v_eff: 能效缺口（A32.4）
        delta_v_safe_norm: 归一化安全缺口（A32.5）
        delta_v_gain_norm: 归一化收益缺口（A32.5）
        delta_v_eff_norm: 归一化能效缺口（A32.5）
        delta_v_safe_max: 安全缺口历史最大值
        delta_v_gain_max: 收益缺口历史最大值
        delta_v_eff_max: 能效缺口历史最大值
        priority_order: 缺口优先级排序（A32.6）
        safe_weights: 安全缺口权重元组
        gain_weights: 收益缺口权重元组
        eff_weights: 能效缺口权重元组

    Invariants:
        - delta_v_safe >= 0
        - delta_v_gain >= 0
        - delta_v_eff >= 0
        - delta_v_*_norm ∈ [0, 1]
        - priority_order 为 {'safe', 'gain', 'eff'} 的排列
    """
    env_state: EnvironmentStateVector
    delta_v_safe: float
    delta_v_gain: float
    delta_v_eff: float
    delta_v_safe_norm: float
    delta_v_gain_norm: float
    delta_v_eff_norm: float
    delta_v_safe_max: float
    delta_v_gain_max: float
    delta_v_eff_max: float
    priority_order: tuple[str, str, str]
    safe_weights: tuple[float, float, float]
    gain_weights: tuple[float, float, float]
    eff_weights: tuple[float, float, float, float]

    def __post_init__(self):
        # 验证缺口非负
        for name, val in [('safe', self.delta_v_safe),
                          ('gain', self.delta_v_gain),
                          ('eff', self.delta_v_eff)]:
            if val < -1e-10:
                raise ValueError(
                    f"{name} 缺口必须 ≥ 0，"
                    f"收到 {val}"
                )
        # 验证归一化缺口范围
        for name, val in [('safe_norm', self.delta_v_safe_norm),
                          ('gain_norm', self.delta_v_gain_norm),
                          ('eff_norm', self.delta_v_eff_norm)]:
            if not (-1e-10 <= val <= 1 + 1e-10):
                raise ValueError(
                    f"归一化{name}缺口必须 ∈ [0, 1]，"
                    f"收到 {val}"
                )
        # 验证最大值严格为正
        for name, val in [('safe_max', self.delta_v_safe_max),
                          ('gain_max', self.delta_v_gain_max),
                          ('eff_max', self.delta_v_eff_max)]:
            if val <= 0:
                raise ValueError(
                    f"{name} 必须严格 > 0，"
                    f"收到 {val}"
                )
        # 验证优先级排序
        if set(self.priority_order) != {'safe', 'gain', 'eff'}:
            raise ValueError(
                f"优先级排序必须为 {{'safe', 'gain', 'eff'}} 的排列，"
                f"收到 {self.priority_order}"
            )

    @property
    def dominant_gap(self) -> str:
        """主导缺口类型。"""
        return self.priority_order[0]

    @property
    def total_gap(self) -> float:
        """总缺口（归一化缺口的加权和）。"""
        return (self.delta_v_safe_norm
                + self.delta_v_gain_norm
                + self.delta_v_eff_norm)

    @property
    def gap_balance(self) -> dict[str, float]:
        """缺口平衡比例。"""
        total = self.delta_v_safe_norm + self.delta_v_gain_norm + self.delta_v_eff_norm
        if total < 1e-10:
            return {'safe': 1/3, 'gain': 1/3, 'eff': 1/3}
        return {
            'safe': self.delta_v_safe_norm / total,
            'gain': self.delta_v_gain_norm / total,
            'eff': self.delta_v_eff_norm / total
        }

    @property
    def is_balanced(self) -> bool:
        """三大缺口是否相对均衡。"""
        balance = self.gap_balance
        values = list(balance.values())
        return max(values) - min(values) < 0.3


# ============================================================
# 阶段32 核心接口：缺口评估器
# ============================================================
class GapAssessor:
    """阶段32的核心计算接口。

    实现缺口评估的完整流程，包括三大缺口的计算、
    归一化和优先级排序。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - ΔV_i ≥ 0（缺口非负）
        - ΔV_i_norm ∈ [0, 1]（归一化范围）
        - 安全优先原则
    """

    def compute_safe_gap(
        self,
        env_state: EnvironmentStateVector,
        weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
        alpha_s: float = 1.0,
        anomaly_threshold: float = 2.0
    ) -> float:
        """计算安全缺口（A32.2）。

        Args:
            env_state: 环境状态向量
            weights: (anomaly_count, error_rate, unknown_ratio) 权重
            alpha_s: 安全缩放系数
            anomaly_threshold: 异常阈值（标准差倍数）

        Returns:
            ΔV_safe ≥ 0

        Raises:
            ValueError: 当 weights 和不等于1或 alpha_s <= 0 时

        Examples:
            >>> import numpy as np
            >>> from stage_031 import EnvironmentStateVector
            >>> from stage_030 import DriveValidationResult
            >>> env = EnvironmentStateVector(
            ...     validation_result=DriveValidationResult(),
            ...     s_sys=(50.0, 60.0, 70.0, 30.0, 2.0, 100.0),
            ...     s_log=(10.0, 0.05, 0.0),
            ...     s_file=(0.1, 0.0, 0.0),
            ...     s_self=(0.8, 1.0, 1e6, 1000.0),
            ...     z_sys=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ...     z_log=(0.0, 0.0, 0.0),
            ...     z_file=(0.0, 0.0, 0.0),
            ...     z_self=(0.0, 0.0, 0.0, 0.0),
            ...     mu=tuple([0.0]*16), sigma=tuple([1.0]*16),
            ...     timestamps=(0.0, 0.0, 0.0, 0.0)
            ... )
            >>> assessor = GapAssessor()
            >>> gap = assessor.compute_safe_gap(env)
            >>> gap >= 0
            True
        """
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"安全缺口权重之和必须等于1，收到 {sum(weights)}")
        if alpha_s <= 0:
            raise ValueError(f"alpha_s 必须 > 0，收到 {alpha_s}")
        # 从标准化系统层提取异常计数（标准化值超过阈值的个数）
        z_sys = np.array(env.z_sys)
        anomaly_count = int(np.sum(np.abs(z_sys) > anomaly_threshold))
        # 错误率直接使用日志层原始值
        error_rate = float(env.s_log[1])
        # 未知比例：用系统层标准化值接近0的比例近似
        unknown_ratio = float(np.sum(np.abs(z_sys) < 0.5) / max(len(z_sys), 1))
        delta_v_safe = alpha_s * (
            weights[0] * float(anomaly_count) +
            weights[1] * error_rate +
            weights[2] * unknown_ratio
        )
        return max(delta_v_safe, 0.0)

    def compute_gain_gap(
        self,
        env_state: EnvironmentStateVector,
        weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
        alpha_g: float = 1.0
    ) -> float:
        """计算收益缺口（A32.3）。

        Args:
            env_state: 环境状态向量
            weights: (coverage, accuracy, novelty) 权重
            alpha_g: 收益缩放系数

        Returns:
            ΔV_gain ≥ 0

        Raises:
            ValueError: 当 weights 和不等于1或 alpha_g <= 0 时

        Examples:
            >>> import numpy as np
            >>> from stage_031 import EnvironmentStateVector
            >>> from stage_030 import DriveValidationResult
            >>> env = EnvironmentStateVector(
            ...     validation_result=DriveValidationResult(),
            ...     s_sys=(0.0,)*6, s_log=(0.0,)*3, s_file=(0.0,)*3,
            ...     s_self=(0.5, 2.0, 1e6, 100.0),
            ...     z_sys=(0.0,)*6, z_log=(0.0,)*3, z_file=(0.0,)*3,
            ...     z_self=(0.0, 0.0, 0.0, 0.0),
            ...     mu=tuple([0.0]*16), sigma=tuple([1.0]*16),
            ...     timestamps=(0.0,)*4
            ... )
            >>> assessor = GapAssessor()
            >>> gap = assessor.compute_gain_gap(env)
            >>> gap >= 0
            True
        """
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"收益缺口权重之和必须等于1，收到 {sum(weights)}")
        if alpha_g <= 0:
            raise ValueError(f"alpha_g 必须 > 0，收到 {alpha_g}")
        # coverage：用文件变更率反向表示（变更越少，覆盖度越低，缺口越大）
        coverage = 1.0 - float(env.s_file[0])
        # accuracy：用模型准确率反向表示（准确率越低，缺口越大）
        accuracy = 1.0 - float(env.s_self[0])
        # novelty：用误差分布熵表示（熵越大，不确定性越大，缺口越大）
        novelty = float(env.s_self[1])
        delta_v_gain = alpha_g * (
            weights[0] * coverage +
            weights[1] * accuracy +
            weights[2] * novelty
        )
        return max(delta_v_gain, 0.0)

    def compute_eff_gap(
        self,
        env_state: EnvironmentStateVector,
        weights: tuple[float, float, float, float] = (0.3, 0.2, 0.2, 0.3),
        alpha_e: float = 1.0
    ) -> float:
        """计算能效缺口（A32.4）。

        Args:
            env_state: 环境状态向量
            weights: (cpu, mem, time, redundancy) 权重
            alpha_e: 能效缩放系数

        Returns:
            ΔV_eff ≥ 0

        Raises:
            ValueError: 当 weights 和不等于1或 alpha_e <= 0 时

        Examples:
            >>> import numpy as np
            >>> from stage_031 import EnvironmentStateVector
            >>> from stage_030 import DriveValidationResult
            >>> env = EnvironmentStateVector(
            ...     validation_result=DriveValidationResult(),
            ...     s_sys=(90.0, 80.0, 70.0, 60.0, 5.0, 200.0),
            ...     s_log=(0.0,)*3, s_file=(0.0,)*3,
            ...     s_self=(0.0, 0.0, 5e6, 100.0),
            ...     z_sys=(0.0,)*6, z_log=(0.0,)*3, z_file=(0.0,)*3,
            ...     z_self=(0.0, 0.0, 0.0, 0.0),
            ...     mu=tuple([0.0]*16), sigma=tuple([1.0]*16),
            ...     timestamps=(0.0,)*4
            ... )
            >>> assessor = GapAssessor()
            >>> gap = assessor.compute_eff_gap(env)
            >>> gap >= 0
            True
        """
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"能效缺口权重之和必须等于1，收到 {sum(weights)}")
        if alpha_e <= 0:
            raise ValueError(f"alpha_e 必须 > 0，收到 {alpha_e}")
        # cpu, mem 使用系统层原始值（百分比）
        cpu = float(env.s_sys[0]) / 100.0
        mem = float(env.s_sys[1]) / 100.0
        # time：用计算成本的对数归一化近似
        compute_cost = float(env.s_self[2])
        time_cost = min(1.0, compute_cost / 1e7)
        # redundancy：用进程数反向表示（进程越多，冗余度越高，缺口越大）
        redundancy = min(1.0, float(env.s_sys[5]) / 500.0)
        delta_v_eff = alpha_e * (
            weights[0] * cpu +
            weights[1] * mem +
            weights[2] * time_cost +
            weights[3] * redundancy
        )
        return max(delta_v_eff, 0.0)

    def normalize_gap(
        self,
        gap_value: float,
        gap_max: float
    ) -> float:
        """归一化缺口（A32.5）。

        Args:
            gap_value: 缺口值
            gap_max: 历史最大值

        Returns:
            ΔV_norm ∈ [0, 1]

        Raises:
            ValueError: 当 gap_max <= 0 时

        Examples:
            >>> assessor = GapAssessor()
            >>> assessor.normalize_gap(5.0, 10.0)
            0.5
            >>> assessor.normalize_gap(15.0, 10.0)
            1.0
        """
        if gap_max <= 0:
            raise ValueError(f"gap_max 必须 > 0，收到 {gap_max}")
        return min(max(gap_value / gap_max, 0.0), 1.0)

    def rank_priorities(
        self,
        safe_norm: float,
        gain_norm: float,
        eff_norm: float,
        safe_first: bool = True
    ) -> tuple[str, str, str]:
        """缺口优先级排序（A32.6）。

        Args:
            safe_norm: 归一化安全缺口 ∈ [0,1]
            gain_norm: 归一化收益缺口 ∈ [0,1]
            eff_norm: 归一化能效缺口 ∈ [0,1]
            safe_first: 安全优先原则开关，默认True

        Returns:
            优先级排序元组，如 ('safe', 'gain', 'eff')

        Raises:
            ValueError: 当任一归一化缺口不在 [0,1] 时

        Examples:
            >>> assessor = GapAssessor()
            >>> assessor.rank_priorities(0.8, 0.3, 0.5)
            ('safe', 'eff', 'gain')
            >>> assessor.rank_priorities(0.3, 0.8, 0.5, safe_first=False)
            ('gain', 'eff', 'safe')
        """
        for name, val in [('safe_norm', safe_norm),
                          ('gain_norm', gain_norm),
                          ('eff_norm', eff_norm)]:
            if not (0.0 - 1e-10 <= val <= 1.0 + 1e-10):
                raise ValueError(f"{name} 必须在 [0,1] 内，收到 {val}")
        gaps = {'safe': safe_norm, 'gain': gain_norm, 'eff': eff_norm}
        # 安全优先：若安全缺口与其他最大缺口差距在 0.1 以内，安全优先
        if safe_first:
            max_other = max(gain_norm, eff_norm)
            if safe_norm >= max_other - 0.1:
                # 强制安全为最高优先级
                sorted_items = sorted(
                    gaps.items(), key=lambda x: (x[0] != 'safe', -x[1])
                )
                return tuple(item[0] for item in sorted_items)
        sorted_items = sorted(gaps.items(), key=lambda x: -x[1])
        return tuple(item[0] for item in sorted_items)

    def assess(
        self,
        env_state: EnvironmentStateVector
    ) -> GapAssessment:
        """完整缺口评估流程。

        依次计算三大缺口、归一化、排序。

        Args:
            env_state: 环境状态向量

        Returns:
            GapAssessment 实例

        Examples:
            >>> import numpy as np
            >>> from stage_031 import EnvironmentStateVector
            >>> from stage_030 import DriveValidationResult
            >>> env = EnvironmentStateVector(
            ...     validation_result=DriveValidationResult(),
            ...     s_sys=(50.0, 60.0, 70.0, 30.0, 2.0, 100.0),
            ...     s_log=(10.0, 0.05, 0.0),
            ...     s_file=(0.1, 0.0, 0.0),
            ...     s_self=(0.8, 1.0, 1e6, 1000.0),
            ...     z_sys=(0.0,)*6, z_log=(0.0,)*3, z_file=(0.0,)*3,
            ...     z_self=(0.0, 0.0, 0.0, 0.0),
            ...     mu=tuple([0.0]*16), sigma=tuple([1.0]*16),
            ...     timestamps=(0.0,)*4
            ... )
            >>> assessor = GapAssessor()
            >>> assessment = assessor.assess(env)
            >>> assessment.dominant_gap in {'safe', 'gain', 'eff'}
            True
        """
        # 计算原始缺口
        delta_v_safe = self.compute_safe_gap(env_state)
        delta_v_gain = self.compute_gain_gap(env_state)
        delta_v_eff = self.compute_eff_gap(env_state)
        # 历史最大值（简化：使用当前值与预设默认最大值的较大者）
        default_max_safe = 10.0
        default_max_gain = 10.0
        default_max_eff = 10.0
        delta_v_safe_max = max(delta_v_safe, default_max_safe)
        delta_v_gain_max = max(delta_v_gain, default_max_gain)
        delta_v_eff_max = max(delta_v_eff, default_max_eff)
        # 归一化
        delta_v_safe_norm = self.normalize_gap(delta_v_safe, delta_v_safe_max)
        delta_v_gain_norm = self.normalize_gap(delta_v_gain, delta_v_gain_max)
        delta_v_eff_norm = self.normalize_gap(delta_v_eff, delta_v_eff_max)
        # 优先级排序
        priority_order = self.rank_priorities(
            delta_v_safe_norm, delta_v_gain_norm, delta_v_eff_norm
        )
        return GapAssessment(
            env_state=env_state,
            delta_v_safe=delta_v_safe,
            delta_v_gain=delta_v_gain,
            delta_v_eff=delta_v_eff,
            delta_v_safe_norm=delta_v_safe_norm,
            delta_v_gain_norm=delta_v_gain_norm,
            delta_v_eff_norm=delta_v_eff_norm,
            delta_v_safe_max=delta_v_safe_max,
            delta_v_gain_max=delta_v_gain_max,
            delta_v_eff_max=delta_v_eff_max,
            priority_order=priority_order,
            safe_weights=(0.4, 0.4, 0.2),
            gain_weights=(0.4, 0.3, 0.3),
            eff_weights=(0.3, 0.2, 0.2, 0.3),
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `EnvironmentStateVector` | ← 阶段31 | 作为缺口评估的环境状态输入 |
| **输入** | A25.1-A25.5（缺口定义） | ← 阶段25 | 缺口计算的理论公式 |
| **输入** | A25.6（缺口永不为零） | ← 阶段25 | 缺口非负性的理论保证 |
| **输出** | `GapAssessment` | → 阶段33 | 安全缺口作为安全任务生成的输入 |
| **输出** | `GapAssessment` | → 阶段34 | 收益缺口作为收益任务生成的输入 |
| **输出** | `GapAssessment` | → 阶段35 | 能效缺口作为能效任务生成的输入 |
| **输出** | A32.6（优先级排序） | → 后续阶段 | 任务调度的优先级依据 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 缺口计算 | `jax.numpy`实现向量化的缺口计算（A32.2-A32.4）；`jax.lax.scan`实现缺口历史的最大值更新（A32.5） |
| NumPy | 统计计算 | 百分位最大值计算（`np.percentile`）；归一化计算 |
| SciPy | 缺口分布分析 | `scipy.stats`分析缺口的历史分布，确定归一化的最大值策略 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V32.1 | 安全缺口非负 | $\Delta V_{\text{safe}} \geq 0$ 对所有合法输入成立 | 随机环境状态测试 |
| V32.2 | 收益缺口非负 | $\Delta V_{\text{gain}} \geq 0$ 对所有合法输入成立 | 随机环境状态测试 |
| V32.3 | 能效缺口非负 | $\Delta V_{\text{eff}} \geq 0$ 对所有合法输入成立 | 随机环境状态测试 |
| V32.4 | 缺口永不为零 | $\Delta V_i > 0$ 在开放系统中恒成立 | 长期运行模拟 |
| V32.5 | 归一化范围 | $\Delta V_{i,\text{norm}} \in [0, 1]$ | 边界值测试 |
| V32.6 | 归一化截断 | $\Delta V_i > \Delta V_{i,\max}$ 时 $\Delta V_{i,\text{norm}} = 1$ | 极端缺口测试 |
| V32.7 | 安全缺口单调性 | 异常指标增大时 $\Delta V_{\text{safe}}$ 增大 | 异常扫描测试 |
| V32.8 | 收益缺口单调性 | 认知指标减小时 $\Delta V_{\text{gain}}$ 增大 | 覆盖度扫描测试 |
| V32.9 | 能效缺口单调性 | 资源指标增大时 $\Delta V_{\text{eff}}$ 增大 | 资源扫描测试 |
| V32.10 | 优先级排序正确性 | 归一化缺口最大的驱力优先级最高 | 合成缺口数据测试 |
| V32.11 | 安全优先原则 | 安全缺口与其他缺口接近时安全优先 | 等缺口测试 |
| V32.12 | 优先级防抖动 | 优先级切换频率 $\leq 1/T_{\text{debounce}}$ | 长期运行验证 |
| V32.13 | GapAssessment组合 | `GapAssessment.env_state` 类型为 `EnvironmentStateVector` | 类型检查 |
| V32.14 | 非法输入拒绝 | 负缺口、归一化值>1、非法优先级排序等抛出 `ValueError` | 异常测试 |
| V32.15 | 缺口与驱力对称性 | $\Delta V_{\text{safe}} \approx -R_{\text{safe}}$，$\Delta V_{\text{eff}} \approx -R_{\text{eff}}$ | 驱力-缺口对比测试 |

---

> **阶段32完** | 阶段32建立了缺口评估算法的完整体系：缺口评估函数将环境状态向量映射为三大缺口的量化值，归一化确保不同缺口的可比性，优先级排序决定行动方向。缺口评估是从"感知"到"行动"的关键桥梁——后续三个阶段（33-35）将分别从安全缺口、收益缺口和能效缺口出发，生成具体的任务列表。
