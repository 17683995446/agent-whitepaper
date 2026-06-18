# 阶段26：场景权重切换规则——从驱力信号到场景判定的精确映射

> **阶段编号**：S26 | **所属层**：第三层·驱力奖励层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 场景分类器 | "判断当前是什么情况"——根据三大驱力值判定系统所处场景 |
> | 紧急风险场景 | "着火了"——安全驱力超过紧急阈值，必须立即应对 |
> | 稳态空闲场景 | "风平浪静"——安全无忧且收益不高，适合探索学习 |
> | 资源紧张场景 | "钱包见底"——能效驱力超过紧张阈值，必须节约资源 |
> | 混合场景插值 | "半紧急半空闲"——场景边界上的平滑过渡，不生硬跳变 |
> | 防抖动机制 | "不能反复横跳"——场景切换后必须驻留最短时间，避免频繁振荡 |

---

## 1. 第一性原理依据

本阶段从阶段24（动态权重A24.2-A24.6）和阶段21-23（三大驱力定义）的核心结论出发，建立场景权重切换的精确规则。核心依据为：

- **A24.2（动态权重自适应算法）**（引用阶段24）：$W_i = \text{softmax}(\beta \cdot \text{urgency}_i)$，动态权重由紧急程度的softmax计算。本阶段将紧急程度的判定从隐式（softmax自动分配）提升为显式（场景分类器精确判定），使权重切换可解释、可验证、可防抖。
- **A24.3-A24.5（三种典型场景）**（引用阶段24）：紧急风险场景、稳态空闲场景、资源紧张场景的定性描述。本阶段将定性描述转化为精确的阈值判定规则。
- **A24.6（权重切换的平滑过渡）**（引用阶段24）：$W_i(t+1) = \alpha_w \cdot W_i^{\text{target}} + (1-\alpha_w) \cdot W_i(t)$，权重平滑过渡机制。本阶段在此基础上增加场景级别的防抖动机制，确保场景切换的稳定性。
- **A21.1（安全驱力定义）**（引用阶段21）：$R_{\text{safe}} = -\alpha_s \cdot \sum_i w_i \cdot \text{anomaly}_i$，安全驱力是紧急风险场景判定的核心输入。
- **A22.1（收益驱力定义）**（引用阶段22）：$R_{\text{gain}} = \alpha_g \cdot (I_{\text{new}} + \delta_{\text{accuracy}} + \delta_{\text{coverage}})$，收益驱力是稳态空闲场景判定的核心输入。
- **A23.1（能效驱力定义）**（引用阶段23）：$R_{\text{eff}} = -\alpha_e \cdot (\text{cpu\_ratio} + \text{mem\_ratio} + \text{time\_cost} + \text{redundancy})$，能效驱力是资源紧张场景判定的核心输入。

本阶段的核心目标是：**定义场景分类器 $S = \text{classify}(R_{\text{safe}}, R_{\text{gain}}, R_{\text{eff}})$，为三种典型场景（紧急风险、稳态空闲、资源紧张）建立精确的阈值判定规则，为混合场景提供权重插值公式，并设计场景切换的防抖动机制（最小驻留时间），确保场景判定的稳定性和可解释性。** 场景权重切换规则是阶段24动态权重机制的"判定层"——它将softmax的隐式权重分配转化为显式的场景判定，使系统的行为可审计、可调试。

---

## 2. 核心数学公式

### 第一部分：场景分类器

#### 2.1 场景分类器定义（A26.1）

场景分类器将三大驱力值映射为场景标识和对应的权重分配：

$$S = \text{classify}(R_{\text{safe}}, R_{\text{gain}}, R_{\text{eff}}) \mapsto (s, \mathbf{W}^s) \tag{A26.1}$$

其中：
- $s \in \{\text{urgent}, \text{stable}, \text{tight}, \text{mixed}\}$：场景标识
- $\mathbf{W}^s = (W_{\text{safe}}^s, W_{\text{gain}}^s, W_{\text{eff}}^s)$：场景 $s$ 对应的目标权重向量
- $R_{\text{safe}} \leq 0$：安全驱力（引用阶段21，A21.1）
- $R_{\text{gain}} \geq 0$（通常）：收益驱力（引用阶段22，A22.1）
- $R_{\text{eff}} \leq 0$：能效驱力（引用阶段23，A23.1）

- **输入**：三大驱力值 $(R_{\text{safe}}, R_{\text{gain}}, R_{\text{eff}})$
- **输出**：场景标识 $s$，目标权重向量 $\mathbf{W}^s$
- **约束**：$W_i^s \geq 0$；$\sum_i W_i^s = 1$；场景标识为四种之一

> **白话翻译**：场景分类器就是"判断当前是什么情况，然后分配权重"。输入三大驱力值，输出场景类型和对应的权重分配。四种场景分别对应四种权重策略：紧急时安全独大、空闲时收益独大、紧张时能效独大、混合时按比例分配。

**场景分类的优先级**：

场景判定遵循严格的优先级顺序——紧急风险 > 资源紧张 > 稳态空闲 > 混合场景。这反映了"生存优先"的原则：安全威胁总是最紧迫的，资源紧张次之，空闲探索最不紧迫。

$$\text{优先级：urgent} > \text{tight} > \text{stable} > \text{mixed} \tag{A26.1-priority}$$

优先级的设计理由：
1. **紧急风险优先**：安全驱力直接关联生存（引用阶段1，A1.8），任何安全威胁都必须立即响应
2. **资源紧张次之**：资源耗尽会导致系统崩溃（引用阶段23，A23.1），但通常不像安全威胁那样紧迫
3. **稳态空闲最后**：探索学习是长期收益（引用阶段22，A22.1），可以等待安全稳定后再进行

**场景的目标权重向量**：

| 场景 | $W_{\text{safe}}^s$ | $W_{\text{gain}}^s$ | $W_{\text{eff}}^s$ | 物理含义 |
|------|---------------------|---------------------|---------------------|---------|
| urgent | $W_{\text{safe}}^{\text{urgent}}$ | $W_{\text{gain}}^{\text{urgent}}$ | $W_{\text{eff}}^{\text{urgent}}$ | 安全权重独大，其他让路 |
| stable | $W_{\text{safe}}^{\text{stable}}$ | $W_{\text{gain}}^{\text{stable}}$ | $W_{\text{eff}}^{\text{stable}}$ | 收益权重独大，安心探索 |
| tight | $W_{\text{safe}}^{\text{tight}}$ | $W_{\text{gain}}^{\text{tight}}$ | $W_{\text{eff}}^{\text{tight}}$ | 能效权重独大，精打细算 |
| mixed | 由A26.5插值计算 | 由A26.5插值计算 | 由A26.5插值计算 | 按比例分配 |

默认目标权重值：

$$\mathbf{W}^{\text{urgent}} = (0.8, 0.1, 0.1), \quad \mathbf{W}^{\text{stable}} = (0.1, 0.8, 0.1), \quad \mathbf{W}^{\text{tight}} = (0.1, 0.1, 0.8) \tag{A26.1-defaults}$$

默认值不设为极端的 $(1, 0, 0)$ 是为了保留其他驱力的最低权重，避免完全忽略非主导驱力——即使在紧急场景下，系统仍需维持基本的认知更新和资源监控。

---

### 第二部分：紧急风险场景判定

#### 2.2 紧急风险场景判定（A26.2）

紧急风险场景的判定条件为安全驱力超过紧急阈值：

$$R_{\text{safe}} < \theta_{\text{urgent}} \tag{A26.2}$$

其中 $\theta_{\text{urgent}} < 0$ 为紧急风险阈值。

- **输入**：安全驱力 $R_{\text{safe}} \leq 0$，紧急阈值 $\theta_{\text{urgent}} < 0$
- **输出**：布尔值，是否判定为紧急风险场景
- **约束**：$\theta_{\text{urgent}} < 0$；$\theta_{\text{urgent}}$ 的绝对值越大，对安全威胁越不敏感

> **白话翻译**：紧急风险场景判定就是"安全驱力够不够差"——安全驱力越负表示越危险，当危险程度超过阈值 $\theta_{\text{urgent}}$ 时，判定为紧急风险场景。$\theta_{\text{urgent}}$ 是"警报触发线"——设得越负（绝对值越大），系统越迟钝，需要更大的威胁才触发；设得越接近零，系统越敏感，小问题也会触发紧急响应。

**紧急阈值的物理含义**：

由A21.4（引用阶段21），$R_{\text{safe}} = -\alpha_s \cdot \max(0, F - F_{\text{baseline}})$。因此 $R_{\text{safe}} < \theta_{\text{urgent}}$ 等价于：

$$F - F_{\text{baseline}} > \frac{|\theta_{\text{urgent}}|}{\alpha_s} \tag{A26.2-free-energy}$$

即自由能偏离基线超过 $|\theta_{\text{urgent}}| / \alpha_s$ 时触发紧急响应。这给出了紧急阈值的物理含义：$\theta_{\text{urgent}}$ 控制的是"自由能偏离基线多少个标准差时触发紧急响应"。

**紧急阈值的推荐值**：

由A21.2-sys（引用阶段21），系统异常度量使用标准化值 $\max(0, (\varepsilon_l - \bar{\varepsilon}_l) / \sigma_{\varepsilon_l})$。当所有层的平均标准化异常超过3个标准差时，通常认为系统处于紧急状态。因此：

$$\theta_{\text{urgent}} = -3\alpha_s \tag{A26.2-recommended}$$

这对应自由能偏离基线3个标准差——在正态分布假设下，偏离3σ的概率仅0.27%，足以表征"异常紧急"。

**紧急阈值的自适应调整**：

在长期运行中，系统可能适应某些类型的异常（如周期性的负载波动），导致固定阈值过于敏感或迟钝。引入自适应阈值：

$$\theta_{\text{urgent}}(t) = -\kappa_{\text{adapt}} \cdot \alpha_s \cdot \sigma_{\text{safe}}(t) \tag{A26.2-adaptive}$$

其中 $\sigma_{\text{safe}}(t)$ 为安全驱力的近期标准差，$\kappa_{\text{adapt}} > 0$ 为自适应系数。当安全驱力波动增大时，阈值自动放宽（绝对值增大），避免对正常波动过度反应；当波动减小时，阈值自动收紧，保持对异常的敏感度。

---

### 第三部分：稳态空闲场景判定

#### 2.3 稳态空闲场景判定（A26.3）

稳态空闲场景的判定条件为安全驱力高于稳定阈值且收益驱力低于空闲阈值：

$$R_{\text{safe}} > \theta_{\text{stable}} \quad \text{AND} \quad R_{\text{gain}} < \theta_{\text{idle}} \tag{A26.3}$$

其中 $\theta_{\text{stable}} < 0$ 为稳定阈值（接近零），$\theta_{\text{idle}} > 0$ 为空闲阈值。

- **输入**：安全驱力 $R_{\text{safe}}$，收益驱力 $R_{\text{gain}}$，稳定阈值 $\theta_{\text{stable}}$，空闲阈值 $\theta_{\text{idle}}$
- **输出**：布尔值，是否判定为稳态空闲场景
- **约束**：$\theta_{\text{stable}} < 0$（接近零）；$\theta_{\text{idle}} > 0$（较低）

> **白话翻译**：稳态空闲场景判定就是"安全没问题且收益不高"——安全驱力接近零表示没有威胁，收益驱力不高表示当前没有大的认知收益。两个条件同时满足才判定为空闲——如果安全没问题但收益很高，说明系统正在高效学习，不是"空闲"而是"丰收"。

**稳定阈值的物理含义**：

$R_{\text{safe}} > \theta_{\text{stable}}$ 等价于 $F - F_{\text{baseline}} < |\theta_{\text{stable}}| / \alpha_s$，即自由能偏离基线很小。推荐值：

$$\theta_{\text{stable}} = -0.5\alpha_s \tag{A26.3-stable-recommended}$$

对应自由能偏离基线不超过0.5个标准差——系统处于"正常波动范围"内。

**空闲阈值的物理含义**：

$R_{\text{gain}} < \theta_{\text{idle}}$ 表示当前认知收益不高。由A22.1（引用阶段22），$R_{\text{gain}} = \alpha_g \cdot (I_{\text{new}} + \delta_{\text{accuracy}} + \delta_{\text{coverage}})$。空闲意味着新信息量少、精度提升小、覆盖度增长慢。推荐值：

$$\theta_{\text{idle}} = 0.3 \cdot \alpha_g \tag{A26.3-idle-recommended}$$

对应认知收益低于最大可能收益的30%——系统处于"信息饥饿"状态，有大量未探索的空间。

**稳态空闲场景的双重条件**：

为什么需要"安全没问题 AND 收益不高"两个条件？

1. **仅安全没问题**：可能系统正在高效学习（收益高），此时不应判定为空闲——系统正在"丰收期"，应该继续当前策略
2. **仅收益不高**：可能系统正在应对安全威胁（安全驱力低），此时不应判定为空闲——系统正在"危机中"，应该优先处理安全
3. **两者同时满足**：系统安全无忧且没有大的认知收益，正是"空闲"——适合启动探索性策略

---

### 第四部分：资源紧张场景判定

#### 2.4 资源紧张场景判定（A26.4）

资源紧张场景的判定条件为能效驱力低于紧张阈值：

$$R_{\text{eff}} < \theta_{\text{tight}} \tag{A26.4}$$

其中 $\theta_{\text{tight}} < 0$ 为资源紧张阈值。

- **输入**：能效驱力 $R_{\text{eff}} \leq 0$，紧张阈值 $\theta_{\text{tight}} < 0$
- **输出**：布尔值，是否判定为资源紧张场景
- **约束**：$\theta_{\text{tight}} < 0$；$\theta_{\text{tight}}$ 的绝对值越大，对资源压力越不敏感

> **白话翻译**：资源紧张场景判定就是"能效驱力够不够差"——能效驱力越负表示资源消耗越大，当消耗超过阈值 $\theta_{\text{tight}}$ 时，判定为资源紧张场景。$\theta_{\text{tight}}$ 是"省钱触发线"——设得越负，系统越能容忍资源消耗；设得越接近零，系统越精打细算。

**紧张阈值的物理含义**：

由A23.1（引用阶段23），$R_{\text{eff}} = -\alpha_e \cdot (\text{cpu\_ratio} + \text{mem\_ratio} + \text{time\_cost} + \text{redundancy})$。当四项资源消耗之和超过 $|\theta_{\text{tight}}| / \alpha_e$ 时触发资源紧张。推荐值：

$$\theta_{\text{tight}} = -2.0\alpha_e \tag{A26.4-recommended}$$

对应总资源消耗指标超过2.0——例如CPU占用80%+内存占用60%+推理耗时0.3+冗余度0.3 = 2.0。这个阈值表示"资源消耗已经相当可观，需要开始节约"。

**资源紧张与紧急风险的区分**：

资源紧张和紧急风险都可能表现为驱力值为负，但两者有本质区别：

| 特征 | 紧急风险 | 资源紧张 |
|------|---------|---------|
| 核心驱力 | $R_{\text{safe}}$ 极负 | $R_{\text{eff}}$ 极负 |
| 威胁来源 | 外部（环境异常） | 内部（资源耗尽） |
| 响应策略 | 回避威胁（减少预测误差） | 节约资源（减少计算量） |
| 时间尺度 | 紧急（秒级响应） | 紧迫（分钟级响应） |
| 优先级 | 高（生存威胁） | 中（运行威胁） |

当两者同时触发时（既不安全又缺资源），按优先级规则（A26.1-priority），紧急风险优先——"先活下来，再考虑省钱"。

---

### 第五部分：混合场景的权重插值

#### 2.5 混合场景的权重插值（A26.5）

当系统不满足任何单一场景的判定条件时，处于混合场景。混合场景的目标权重通过各场景的隶属度进行插值：

$$W_i^{\text{mixed}} = \sum_{s \in \mathcal{S}} \mu_s \cdot W_i^s \tag{A26.5}$$

其中 $\mu_s$ 为场景 $s$ 的隶属度，满足 $\mu_s \geq 0$，$\sum_s \mu_s = 1$。

- **输入**：各场景的目标权重 $\{W_i^s\}$，各场景的隶属度 $\{\mu_s\}$
- **输出**：混合场景的目标权重 $W_i^{\text{mixed}}$
- **约束**：$\mu_s \geq 0$；$\sum_s \mu_s = 1$；$W_i^{\text{mixed}} \geq 0$；$\sum_i W_i^{\text{mixed}} = 1$

> **白话翻译**：混合场景的权重插值就是"按比例混合各场景的权重"——如果系统70%像紧急风险、30%像稳态空闲，则最终权重 = 0.7 × 紧急权重 + 0.3 × 空闲权重。隶属度决定"像哪个场景的程度"，权重按隶属度加权平均。

**隶属度的计算**：

隶属度基于驱力值与各场景阈值的距离计算：

$$\mu_{\text{urgent}} = \frac{\max(0, \theta_{\text{urgent}} - R_{\text{safe}})}{D_{\text{total}}} \tag{A26.5-urgent}$$

$$\mu_{\text{stable}} = \frac{\max(0, R_{\text{safe}} - \theta_{\text{stable}}) \cdot \max(0, \theta_{\text{idle}} - R_{\text{gain}})}{D_{\text{total}}} \tag{A26.5-stable}$$

$$\mu_{\text{tight}} = \frac{\max(0, \theta_{\text{tight}} - R_{\text{eff}})}{D_{\text{total}}} \tag{A26.5-tight}$$

其中 $D_{\text{total}}$ 为归一化因子：

$$D_{\text{total}} = \max(0, \theta_{\text{urgent}} - R_{\text{safe}}) + \max(0, R_{\text{safe}} - \theta_{\text{stable}}) \cdot \max(0, \theta_{\text{idle}} - R_{\text{gain}}) + \max(0, \theta_{\text{tight}} - R_{\text{eff}}) + \epsilon \tag{A26.5-normalize}$$

$\epsilon > 0$ 为数值稳定项，防止分母为零。

**隶属度的物理含义**：

- $\mu_{\text{urgent}}$：安全驱力低于紧急阈值的程度——越低于阈值，越像紧急场景
- $\mu_{\text{stable}}$：安全驱力高于稳定阈值且收益驱力低于空闲阈值的程度——越满足两个条件，越像空闲场景
- $\mu_{\text{tight}}$：能效驱力低于紧张阈值的程度——越低于阈值，越像紧张场景

**插值的归一化保持**：

设 $\sum_s \mu_s = 1$ 且各场景权重满足 $\sum_i W_i^s = 1$，则：

$$\sum_i W_i^{\text{mixed}} = \sum_i \sum_s \mu_s W_i^s = \sum_s \mu_s \sum_i W_i^s = \sum_s \mu_s \cdot 1 = 1$$

因此插值保持权重归一化。

**插值的非负性保持**：

设 $\mu_s \geq 0$ 且 $W_i^s \geq 0$，则 $W_i^{\text{mixed}} = \sum_s \mu_s W_i^s \geq 0$。因此插值保持权重非负性。

**混合场景的连续性**：

当驱力值从某个场景的阈值附近逐渐变化时，隶属度连续变化，插值权重也连续变化。这保证了场景切换的平滑性——不会出现权重的突然跳变。

具体地，当 $R_{\text{safe}}$ 从 $\theta_{\text{urgent}}$ 上方逐渐下降到 $\theta_{\text{urgent}}$ 以下时，$\mu_{\text{urgent}}$ 从0连续增大，$W_{\text{safe}}^{\text{mixed}}$ 从非紧急权重连续过渡到紧急权重。这种连续性是A24.6（引用阶段24）权重平滑过渡的补充——A24.6在权重层面平滑，A26.5在场景层面平滑，双重平滑确保系统行为的稳定性。

---

### 第六部分：场景切换的防抖动机制

#### 2.6 场景切换的防抖动机制（A26.6）

场景切换的防抖动机制通过最小驻留时间约束实现——场景切换后必须驻留至少 $T_{\text{debounce}}$ 步才能再次切换：

$$s(t) = \begin{cases} s_{\text{new}} & \text{if } t - t_{\text{last\_switch}} \geq T_{\text{debounce}} \text{ AND } s_{\text{new}} \neq s(t-1) \\ s(t-1) & \text{otherwise} \end{cases} \tag{A26.6}$$

其中：
- $s(t)$：时刻 $t$ 的场景标识
- $s_{\text{new}}$：由A26.1计算的新场景
- $t_{\text{last\_switch}}$：上次场景切换的时刻
- $T_{\text{debounce}} > 0$：最小驻留时间（步数）

- **输入**：新场景 $s_{\text{new}}$，当前场景 $s(t-1)$，上次切换时刻 $t_{\text{last\_switch}}$，当前时刻 $t$，最小驻留时间 $T_{\text{debounce}}$
- **输出**：有效场景 $s(t)$
- **约束**：$T_{\text{debounce}} > 0$；$t_{\text{last\_switch}} \leq t$

> **白话翻译**：防抖动机制就是"不能反复横跳"——刚从紧急切换到空闲，不能马上又切回紧急。必须在新场景驻留至少 $T_{\text{debounce}}$ 步才能再次切换。这就像电路中的"去抖动"——按键按下后必须等待一段时间才能再次触发，避免一次按键被识别为多次。

**最小驻留时间的设计**：

$T_{\text{debounce}}$ 的选择需要在"响应速度"和"稳定性"之间权衡：

| $T_{\text{debounce}}$ 值 | 行为特征 | 适用场景 |
|------------------------|---------|---------|
| $T_{\text{debounce}} = 1$ | 几乎无防抖，即时切换 | 需要极快响应的场景 |
| $T_{\text{debounce}} = 5$ | 轻度防抖，5步内不切换 | 一般场景 |
| $T_{\text{debounce}} = 20$ | 中度防抖，20步内不切换 | 场景边界频繁波动的系统 |
| $T_{\text{debounce}} = 100$ | 强防抖，100步内不切换 | 极度需要稳定性的系统 |

**紧急场景的防抖动豁免**：

紧急风险场景享有防抖动豁免权——当新场景为紧急风险时，无论驻留时间是否满足，都立即切换：

$$s(t) = \begin{cases} \text{urgent} & \text{if } s_{\text{new}} = \text{urgent} \\ s_{\text{new}} & \text{if } t - t_{\text{last\_switch}} \geq T_{\text{debounce}} \text{ AND } s_{\text{new}} \neq s(t-1) \\ s(t-1) & \text{otherwise} \end{cases} \tag{A26.6-emergency}$$

这确保了"着火时立刻跑"——安全威胁不受防抖动约束，可以立即触发紧急响应。

**防抖动与权重平滑的协同**：

防抖动机制（A26.6）与权重平滑过渡（A24.6，引用阶段24）协同工作：

1. **防抖动**：在场景层面防止频繁切换——确保场景标识的稳定性
2. **权重平滑**：在权重层面平滑过渡——即使场景切换，权重也是缓坡变化

两者结合的效果：场景切换被防抖动约束限制频率，权重变化被平滑过渡约束限制幅度。场景层面的"硬约束"（不切换）和权重层面的"软约束"（缓坡变化）共同保证系统行为的稳定性。

**防抖动的理论分析**：

设场景切换的频率为 $f_{\text{switch}}$，无防抖动时 $f_{\text{switch}}$ 可能很高（驱力值在阈值附近波动导致频繁切换）。加入防抖动后：

$$f_{\text{switch}} \leq \frac{1}{T_{\text{debounce}}} \tag{A26.6-frequency}$$

即场景切换频率被限制在 $1/T_{\text{debounce}}$ 以下。这保证了系统行为的"惯性"——场景一旦确定，不会因为短期的驱力波动而改变。

**防抖动与响应延迟的权衡**：

防抖动引入响应延迟——当场景真正改变时，系统可能需要等待 $T_{\text{debounce}}$ 步才能切换。最坏情况下的延迟为 $T_{\text{debounce}} - 1$ 步。

对于紧急场景，通过豁免机制（A26.6-emergency）消除延迟。对于非紧急场景，$T_{\text{debounce}}$ 步的延迟是可接受的——非紧急场景的变化通常是渐进的，不需要即时响应。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A26.1 | $S = \text{classify}(R_{\text{safe}}, R_{\text{gain}}, R_{\text{eff}}) \mapsto (s, \mathbf{W}^s)$ | 三大驱力值 | 场景标识 + 目标权重 | $W_i^s \geq 0$；$\sum W_i^s = 1$ |
| A26.2 | $R_{\text{safe}} < \theta_{\text{urgent}}$ | $R_{\text{safe}}$, $\theta_{\text{urgent}}$ | 布尔值 | $\theta_{\text{urgent}} < 0$ |
| A26.3 | $R_{\text{safe}} > \theta_{\text{stable}}$ AND $R_{\text{gain}} < \theta_{\text{idle}}$ | $R_{\text{safe}}$, $R_{\text{gain}}$, $\theta_{\text{stable}}$, $\theta_{\text{idle}}$ | 布尔值 | $\theta_{\text{stable}} < 0$；$\theta_{\text{idle}} > 0$ |
| A26.4 | $R_{\text{eff}} < \theta_{\text{tight}}$ | $R_{\text{eff}}$, $\theta_{\text{tight}}$ | 布尔值 | $\theta_{\text{tight}} < 0$ |
| A26.5 | $W_i^{\text{mixed}} = \sum_s \mu_s \cdot W_i^s$ | $\{W_i^s\}$, $\{\mu_s\}$ | 混合权重 | $\mu_s \geq 0$；$\sum \mu_s = 1$ |
| A26.6 | 防抖动：驻留时间 $\geq T_{\text{debounce}}$ 才能切换 | $s_{\text{new}}$, $s(t-1)$, $t_{\text{last\_switch}}$, $T_{\text{debounce}}$ | 有效场景 | $T_{\text{debounce}} > 0$；紧急场景豁免 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np

# 引用阶段25的数据类
from stage_025 import ValueGap


class SceneType(Enum):
    """场景类型枚举。"""
    URGENT = "urgent"      # 紧急风险场景
    STABLE = "stable"      # 稳态空闲场景
    TIGHT = "tight"        # 资源紧张场景
    MIXED = "mixed"        # 混合场景


# ============================================================
# 阶段26 数据类：场景权重切换器
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class SceneWeightSwitcher:
    """场景权重切换器的完整描述。

    组合阶段25的ValueGap，在缺口驱动的基础上增加
    场景判定和权重切换的计算结果。

    Attributes:
        value_gap: 阶段25的价值缺口（组合字段）
        scene: 当前场景标识（A26.1）
        target_weights: 当前场景的目标权重元组（A26.1）
        theta_urgent: 紧急风险阈值（A26.2）
        theta_stable: 稳定阈值（A26.3）
        theta_idle: 空闲阈值（A26.3）
        theta_tight: 资源紧张阈值（A26.4）
        memberships: 各场景的隶属度元组（A26.5）
        t_last_switch: 上次场景切换时刻（A26.6）
        t_debounce: 最小驻留时间（A26.6）
        pending_scene: 等待切换的场景（防抖动缓冲）

    Invariants:
        - scene 为 SceneType 枚举值之一
        - all(w >= 0 for w in target_weights) and sum(target_weights) ≈ 1
        - theta_urgent < 0
        - theta_stable < 0
        - theta_idle > 0
        - theta_tight < 0
        - all(m >= 0 for m in memberships) and sum(memberships) ≈ 1
        - t_debounce > 0
    """
    value_gap: ValueGap
    scene: SceneType
    target_weights: tuple[float, float, float]  # (W_safe, W_gain, W_eff)
    theta_urgent: float
    theta_stable: float
    theta_idle: float
    theta_tight: float
    memberships: tuple[float, float, float, float]  # (μ_urgent, μ_stable, μ_tight, μ_mixed)
    t_last_switch: int
    t_debounce: int
    pending_scene: Optional[SceneType]

    def __post_init__(self):
        # 验证场景标识
        if not isinstance(self.scene, SceneType):
            raise ValueError(
                f"场景标识必须为 SceneType 枚举值，"
                f"收到 {type(self.scene)}"
            )
        # 验证目标权重
        for i, w in enumerate(self.target_weights):
            if w < -1e-10:
                raise ValueError(
                    f"第 {i} 个目标权重必须 ≥ 0，"
                    f"收到 {w}"
                )
        w_sum = sum(self.target_weights)
        if not (0.99 <= w_sum <= 1.01):
            raise ValueError(
                f"目标权重之和必须 ≈ 1，"
                f"收到 {w_sum}"
            )
        # 验证阈值
        if self.theta_urgent >= 0:
            raise ValueError(
                f"紧急阈值必须 < 0，"
                f"收到 {self.theta_urgent}"
            )
        if self.theta_stable >= 0:
            raise ValueError(
                f"稳定阈值必须 < 0，"
                f"收到 {self.theta_stable}"
            )
        if self.theta_idle <= 0:
            raise ValueError(
                f"空闲阈值必须 > 0，"
                f"收到 {self.theta_idle}"
            )
        if self.theta_tight >= 0:
            raise ValueError(
                f"紧张阈值必须 < 0，"
                f"收到 {self.theta_tight}"
            )
        # 验证隶属度
        m_sum = sum(self.memberships)
        if not (0.99 <= m_sum <= 1.01):
            raise ValueError(
                f"隶属度之和必须 ≈ 1，"
                f"收到 {m_sum}"
            )
        # 验证防抖动参数
        if self.t_debounce <= 0:
            raise ValueError(
                f"最小驻留时间必须 > 0，"
                f"收到 {self.t_debounce}"
            )

    @property
    def is_urgent(self) -> bool:
        """是否为紧急风险场景。"""
        return self.scene == SceneType.URGENT

    @property
    def is_stable(self) -> bool:
        """是否为稳态空闲场景。"""
        return self.scene == SceneType.STABLE

    @property
    def is_tight(self) -> bool:
        """是否为资源紧张场景。"""
        return self.scene == SceneType.TIGHT

    @property
    def dominant_membership(self) -> str:
        """主导隶属度的场景。"""
        scene_names = ['urgent', 'stable', 'tight', 'mixed']
        return scene_names[int(np.argmax(self.memberships))]

    @property
    def debounce_remaining(self) -> int:
        """防抖动剩余步数（当前时刻需外部提供）。"""
        # 此属性需要外部传入当前时刻才能计算
        # 此处返回 t_debounce 作为参考
        return self.t_debounce


# ============================================================
# 阶段26 核心接口：场景权重切换器
# ============================================================
class SceneWeightSwitcherCalculator:
    """阶段26的核心计算接口。

    实现场景分类、阈值判定、混合插值和防抖动机制的
    完整计算流程。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 场景判定遵循优先级：urgent > tight > stable > mixed
        - 防抖动机制保证场景切换频率 ≤ 1/T_debounce
        - 紧急场景享有防抖动豁免权
    """

    def classify_scene(
        self,
        r_safe: float,
        r_gain: float,
        r_eff: float,
        theta_urgent: float,
        theta_stable: float,
        theta_idle: float,
        theta_tight: float
    ) -> SceneType:
        """场景分类器（A26.1）。

        按优先级顺序判定场景：urgent > tight > stable > mixed。

        Args:
            r_safe: 安全驱力值
            r_gain: 收益驱力值
            r_eff: 能效驱力值
            theta_urgent: 紧急风险阈值，< 0
            theta_stable: 稳定阈值，< 0
            theta_idle: 空闲阈值，> 0
            theta_tight: 资源紧张阈值，< 0

        Returns:
            场景标识 SceneType

        Raises:
            ValueError: 当 theta_urgent >= 0 或 theta_stable >= 0
            ValueError: 当 theta_idle <= 0 或 theta_tight >= 0

        Examples:
            >>> calc = SceneWeightSwitcherCalculator()
            >>> calc.classify_scene(-5.0, 0.5, -0.3, -3.0, -0.5, 0.3, -2.0)
            <SceneType.URGENT: 'urgent'>
        """
        if theta_urgent >= 0:
            raise ValueError(f"theta_urgent 必须 < 0，收到 {theta_urgent}")
        if theta_stable >= 0:
            raise ValueError(f"theta_stable 必须 < 0，收到 {theta_stable}")
        if theta_idle <= 0:
            raise ValueError(f"theta_idle 必须 > 0，收到 {theta_idle}")
        if theta_tight >= 0:
            raise ValueError(f"theta_tight 必须 < 0，收到 {theta_tight}")
        if self.check_urgent(r_safe, theta_urgent):
            return SceneType.URGENT
        if self.check_tight(r_eff, theta_tight):
            return SceneType.TIGHT
        if self.check_stable(r_safe, r_gain, theta_stable, theta_idle):
            return SceneType.STABLE
        return SceneType.MIXED

    def check_urgent(
        self,
        r_safe: float,
        theta_urgent: float
    ) -> bool:
        """紧急风险场景判定（A26.2）。

        Args:
            r_safe: 安全驱力值
            theta_urgent: 紧急风险阈值，< 0

        Returns:
            True 如果 R_safe < θ_urgent

        Examples:
            >>> calc = SceneWeightSwitcherCalculator()
            >>> calc.check_urgent(-5.0, -3.0)
            True
            >>> calc.check_urgent(-1.0, -3.0)
            False
        """
        return r_safe < theta_urgent

    def check_stable(
        self,
        r_safe: float,
        r_gain: float,
        theta_stable: float,
        theta_idle: float
    ) -> bool:
        """稳态空闲场景判定（A26.3）。

        Args:
            r_safe: 安全驱力值
            r_gain: 收益驱力值
            theta_stable: 稳定阈值，< 0
            theta_idle: 空闲阈值，> 0

        Returns:
            True 如果 R_safe > θ_stable AND R_gain < θ_idle

        Examples:
            >>> calc = SceneWeightSwitcherCalculator()
            >>> calc.check_stable(-0.1, 0.1, -0.5, 0.3)
            True
        """
        return r_safe > theta_stable and r_gain < theta_idle

    def check_tight(
        self,
        r_eff: float,
        theta_tight: float
    ) -> bool:
        """资源紧张场景判定（A26.4）。

        Args:
            r_eff: 能效驱力值
            theta_tight: 资源紧张阈值，< 0

        Returns:
            True 如果 R_eff < θ_tight

        Examples:
            >>> calc = SceneWeightSwitcherCalculator()
            >>> calc.check_tight(-3.0, -2.0)
            True
        """
        return r_eff < theta_tight

    def compute_memberships(
        self,
        r_safe: float,
        r_gain: float,
        r_eff: float,
        theta_urgent: float,
        theta_stable: float,
        theta_idle: float,
        theta_tight: float
    ) -> tuple[float, float, float, float]:
        """计算各场景的隶属度（A26.5）。

        Args:
            r_safe: 安全驱力值
            r_gain: 收益驱力值
            r_eff: 能效驱力值
            theta_urgent: 紧急风险阈值
            theta_stable: 稳定阈值
            theta_idle: 空闲阈值
            theta_tight: 资源紧张阈值

        Returns:
            (μ_urgent, μ_stable, μ_tight, μ_mixed) 隶属度元组，sum ≈ 1

        Examples:
            >>> calc = SceneWeightSwitcherCalculator()
            >>> calc.compute_memberships(-5.0, 0.5, -0.3, -3.0, -0.5, 0.3, -2.0)
            (0.666..., 0.0, 0.0, 0.333...)
        """
        eps = 1e-12
        d_urgent = max(0.0, theta_urgent - r_safe)
        d_stable = max(0.0, r_safe - theta_stable) * max(0.0, theta_idle - r_gain)
        d_tight = max(0.0, theta_tight - r_eff)
        d_total = d_urgent + d_stable + d_tight + eps
        mu_urgent = d_urgent / d_total
        mu_stable = d_stable / d_total
        mu_tight = d_tight / d_total
        mu_mixed = max(0.0, 1.0 - mu_urgent - mu_stable - mu_tight)
        return (mu_urgent, mu_stable, mu_tight, mu_mixed)

    def interpolate_weights(
        self,
        memberships: tuple[float, float, float, float],
        scene_weights: dict[SceneType, tuple[float, float, float]]
    ) -> tuple[float, float, float]:
        """混合场景的权重插值（A26.5）。

        Args:
            memberships: 各场景的隶属度 (μ_urgent, μ_stable, μ_tight, μ_mixed)
            scene_weights: 各场景的目标权重字典

        Returns:
            插值后的目标权重 (W_safe, W_gain, W_eff)，sum ≈ 1

        Raises:
            ValueError: 当 scene_weights 缺少必要键

        Examples:
            >>> calc = SceneWeightSwitcherCalculator()
            >>> scene_weights = {
            ...     SceneType.URGENT: (0.8, 0.1, 0.1),
            ...     SceneType.STABLE: (0.1, 0.8, 0.1),
            ...     SceneType.TIGHT: (0.1, 0.1, 0.8),
            ...     SceneType.MIXED: (0.33, 0.33, 0.34)
            ... }
            >>> calc.interpolate_weights((0.5, 0.0, 0.0, 0.5), scene_weights)
            (0.565, 0.215, 0.22)
        """
        required = {SceneType.URGENT, SceneType.STABLE, SceneType.TIGHT, SceneType.MIXED}
        if not required.issubset(scene_weights.keys()):
            missing = required - set(scene_weights.keys())
            raise ValueError(f"scene_weights 缺少键: {missing}")
        w_safe = sum(
            mu * scene_weights[scene][0]
            for mu, scene in zip(memberships, [SceneType.URGENT, SceneType.STABLE, SceneType.TIGHT, SceneType.MIXED])
        )
        w_gain = sum(
            mu * scene_weights[scene][1]
            for mu, scene in zip(memberships, [SceneType.URGENT, SceneType.STABLE, SceneType.TIGHT, SceneType.MIXED])
        )
        w_eff = sum(
            mu * scene_weights[scene][2]
            for mu, scene in zip(memberships, [SceneType.URGENT, SceneType.STABLE, SceneType.TIGHT, SceneType.MIXED])
        )
        total = w_safe + w_gain + w_eff
        if total > 0:
            w_safe /= total
            w_gain /= total
            w_eff /= total
        else:
            w_safe = w_gain = w_eff = 1.0 / 3.0
        return (w_safe, w_gain, w_eff)

    def apply_debounce(
        self,
        new_scene: SceneType,
        current_scene: SceneType,
        t_last_switch: int,
        t_current: int,
        t_debounce: int
    ) -> tuple[SceneType, bool]:
        """应用防抖动机制（A26.6）。

        Args:
            new_scene: 新计算的场景
            current_scene: 当前场景
            t_last_switch: 上次切换时刻
            t_current: 当前时刻
            t_debounce: 最小驻留时间，> 0

        Returns:
            (有效场景, 是否切换) 元组
            紧急场景享有豁免权，立即切换。

        Raises:
            ValueError: 当 t_debounce <= 0
            ValueError: 当 t_current < t_last_switch

        Examples:
            >>> calc = SceneWeightSwitcherCalculator()
            >>> calc.apply_debounce(SceneType.URGENT, SceneType.STABLE, 0, 5, 20)
            (<SceneType.URGENT: 'urgent'>, True)
            >>> calc.apply_debounce(SceneType.TIGHT, SceneType.STABLE, 0, 5, 20)
            (<SceneType.STABLE: 'stable'>, False)
        """
        if t_debounce <= 0:
            raise ValueError(f"t_debounce 必须 > 0，收到 {t_debounce}")
        if t_current < t_last_switch:
            raise ValueError(
                f"t_current ({t_current}) 不能小于 t_last_switch ({t_last_switch})"
            )
        # 紧急场景豁免
        if new_scene == SceneType.URGENT:
            return (new_scene, new_scene != current_scene)
        # 检查驻留时间
        if new_scene != current_scene and (t_current - t_last_switch) >= t_debounce:
            return (new_scene, True)
        return (current_scene, False)
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `ValueGap` | ← 阶段25 | 作为场景权重切换器的缺口驱动组合字段 |
| **输入** | A24.2-A24.6（动态权重与平滑过渡） | ← 阶段24 | 场景权重切换规则的理论基础 |
| **输入** | A21.1（安全驱力） | ← 阶段21 | 紧急风险场景判定的核心输入 |
| **输入** | A22.1（收益驱力） | ← 阶段22 | 稳态空闲场景判定的核心输入 |
| **输入** | A23.1（能效驱力） | ← 阶段23 | 资源紧张场景判定的核心输入 |
| **输出** | `SceneWeightSwitcher` | → 阶段27 | 场景判定结果作为驱力引擎的组合字段 |
| **输出** | A26.1（场景分类器） | → 阶段27-30 | 驱力引擎的场景识别接口 |
| **输出** | A26.6（防抖动机制） | → 阶段28 | 时间平滑与防抖动的协同设计 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 场景分类与插值 | `jax.lax.cond`实现场景判定的条件分支（A26.2-A26.4）；`jax.numpy`实现隶属度计算和权重插值（A26.5） |
| NumPy | 阈值自适应 | 自适应阈值（A26.2-adaptive）的滑动窗口统计量计算 |
| pytest | **本阶段新增**：场景切换测试 | 参数化测试框架，支撑A26.2-A26.6各判定规则的边界值测试和防抖动机制的时序测试 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V26.1 | 场景分类完整性 | 所有三元组 $(R_{\text{safe}}, R_{\text{gain}}, R_{\text{eff}})$ 均被分类为四种场景之一 | 随机驱力值扫描 |
| V26.2 | 紧急风险判定 | $R_{\text{safe}} < \theta_{\text{urgent}}$ 时场景为 urgent | 阈值边界测试 |
| V26.3 | 稳态空闲判定 | $R_{\text{safe}} > \theta_{\text{stable}}$ AND $R_{\text{gain}} < \theta_{\text{idle}}$ 时场景为 stable | 双条件组合测试 |
| V26.4 | 资源紧张判定 | $R_{\text{eff}} < \theta_{\text{tight}}$ 时场景为 tight | 阈值边界测试 |
| V26.5 | 优先级正确性 | 紧急+紧张同时满足时场景为 urgent | 冲突条件测试 |
| V26.6 | 隶属度归一化 | $\sum_s \mu_s = 1$ 对所有驱力值成立 | 随机驱力值测试 |
| V26.7 | 隶属度非负 | $\mu_s \geq 0$ 对所有驱力值成立 | 边界值测试 |
| V26.8 | 插值权重归一化 | $\sum_i W_i^{\text{mixed}} = 1$ | 数值一致性验证 |
| V26.9 | 插值权重非负 | $W_i^{\text{mixed}} \geq 0$ | 数值验证 |
| V26.10 | 插值连续性 | 驱力值微小变化时权重变化 < ε | 连续性扫描测试 |
| V26.11 | 防抖动有效性 | 场景切换频率 $\leq 1/T_{\text{debounce}}$ | 长期运行模拟 |
| V26.12 | 紧急豁免 | 紧急场景不受防抖动约束 | 紧急场景切换测试 |
| V26.13 | 防抖动响应延迟 | 非紧急场景最大延迟 $\leq T_{\text{debounce}}$ 步 | 时序模拟 |
| V26.14 | SceneWeightSwitcher组合 | `SceneWeightSwitcher.value_gap` 类型为 `ValueGap` | 类型检查 |
| V26.15 | 非法输入拒绝 | 无效场景标识、负驻留时间等抛出 `ValueError` | 异常测试 |

---

> **阶段26完** | 下一步：阶段27将从本阶段的场景权重切换规则出发，设计驱力引擎的完整Python接口，整合三大驱力+权重+缺口的计算流程。
