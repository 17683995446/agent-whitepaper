# 阶段21：安全驱力Reward_safe——从异常信号到生存保障

> **阶段编号**：S21 | **所属层**：第三层·驱力奖励层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 安全驱力 | "避险本能"——系统对一切异常信号的自动回避冲动 |
> | 异常加权和 | "各种警报的总音量"——系统异常、报错、未知状态等警报的加权汇总 |
> | 异常权重自适应 | "哪个警报更可怕就听哪个"——严重程度越高的异常，权重自动越大 |
> | 安全驱力与自由能 | "警报音量 = 账单涨幅"——自由能偏离基线越多，安全驱力越强 |
> | 时间平滑 | "警报不能忽高忽低"——平滑处理避免驱力剧烈抖动 |

---

## 1. 第一性原理依据

本阶段从阶段4（变分自由能）、阶段5（准确性项）和阶段3（精度加权误差）的核心结论出发，建立安全驱力的完整数学定义。核心依据为：

- **A4.4（变分自由能全局唯一定义）**（引用阶段4）：$F[q] = -\mathbb{E}_q[\ln p(o|\psi)] + D_{\text{KL}}[q(\psi) \| p(\psi)]$，安全驱力将自由能偏离基线的程度转化为驱力信号。
- **A5.1（准确性项）**（引用阶段5）：$A(q) = -\mathbb{E}_q[\ln p(o|\psi)]$，准确性项的恶化是异常信号的核心来源——预测与观测的偏差越大，异常越严重。
- **A3.2（精度加权误差能量）**（引用阶段3）：$\varepsilon_l = e^{l-1,T} \Pi_l e^{l-1}$，精度加权误差能量为异常信号的分层度量提供基础——各层误差能量的异常升高对应不同类型的系统异常。
- **A4.8（拉普拉斯近似下的自由能）**（引用阶段4）：$F \approx \varepsilon_0 + \frac{1}{2}\sum_{l=1}^{L} \varepsilon_l + \frac{1}{2}\sum_{l=1}^{L} D_{\text{KL}}[q_l \| p_l]$，自由能与精度加权误差能量的直接关系使得安全驱力可以从分层误差信号中直接计算。

本阶段的核心目标是：**定义安全驱力 $R_{\text{safe}}$，将系统运行过程中的各类异常信号（系统异常、报错日志、未知状态、风险进程、突变负载）统一为加权和形式，建立安全驱力与变分自由能偏离基线的定量关系，并提供异常权重自适应和时间平滑机制，确保安全驱力作为稳定的生存保障信号。** 安全驱力是三大驱力中最基本的——它直接对应阶段1的生存条件（A1.8），是"活下去"这一最低需求的数学表达。

---

## 2. 核心数学公式

### 第一部分：安全驱力的定义

#### 2.1 安全驱力定义（A21.1）

安全驱力定义为异常信号的加权和，取负值以符合"驱力越大越需要回避"的约定：

$$R_{\text{safe}} = -\alpha_s \cdot \sum_{i=1}^{N_{\text{anomaly}}} w_i \cdot \text{anomaly}_i \tag{A21.1}$$

其中：
- $\alpha_s > 0$：安全驱力的全局缩放系数，控制安全驱力在总奖励中的相对强度
- $w_i \geq 0$：第 $i$ 类异常的权重，由A21.3的自适应算法确定
- $\text{anomaly}_i \geq 0$：第 $i$ 类异常的标量度量，由A21.2定义
- $N_{\text{anomaly}}$：异常信号类型的总数

- **输入**：异常信号向量 $\{\text{anomaly}_i\}_{i=1}^{N_{\text{anomaly}}}$，权重向量 $\{w_i\}_{i=1}^{N_{\text{anomaly}}}$，缩放系数 $\alpha_s$
- **输出**：$R_{\text{safe}} \in \mathbb{R}_{\leq 0}$（安全驱力恒非正，异常越多驱力越负）
- **约束**：$\alpha_s > 0$；$w_i \geq 0$；$\text{anomaly}_i \geq 0$；$\sum_i w_i = 1$（权重归一化）

> **白话翻译**：安全驱力就是"各种警报的加权总音量"，取负号表示"警报越响，驱力越负，越需要回避"。$\alpha_s$ 是"警报的灵敏度"——调大则对异常更敏感，调小则更迟钝。$w_i$ 是"哪种警报更可怕"——系统异常可能比报错日志更致命，权重就更高。

**安全驱力的符号约定**：$R_{\text{safe}} \leq 0$，越接近零表示越安全，越负表示越危险。这与阶段4的自由能约定一致——自由能越低越好，安全驱力越高（越接近零）越好。

**安全驱力的物理含义**：安全驱力度量系统偏离"安全稳态"的程度。当系统运行在正常范围内时，所有异常信号接近零，$R_{\text{safe}} \approx 0$；当系统出现异常时，异常信号升高，$R_{\text{safe}}$ 变负，驱动系统采取回避行动。

---

### 第二部分：异常信号类型

#### 2.2 异常信号类型（A21.2）

系统运行过程中产生的异常信号分为五类，每类对应不同的异常来源和度量方式：

**类型1：系统异常（System Anomaly）**

系统异常度量系统内部状态的异常程度，直接对应变分自由能中准确性项的恶化：

$$\text{anomaly}_{\text{sys}} = \frac{1}{L}\sum_{l=0}^{L} \max\left(0, \frac{\varepsilon_l - \bar{\varepsilon}_l}{\sigma_{\varepsilon_l}}\right) \tag{A21.2-sys}$$

其中 $\varepsilon_l$ 为第 $l$ 层的精度加权误差能量（引用阶段3，A3.2），$\bar{\varepsilon}_l$ 和 $\sigma_{\varepsilon_l}$ 分别为历史均值和标准差。此度量检测各层误差能量是否显著高于历史基线——显著升高意味着系统对当前输入的预测能力下降，即系统内部状态异常。

- **输入**：各层精度加权误差能量 $\{\varepsilon_l\}_{l=0}^{L}$，历史统计量 $\{\bar{\varepsilon}_l, \sigma_{\varepsilon_l}\}_{l=0}^{L}$
- **输出**：$\text{anomaly}_{\text{sys}} \in \mathbb{R}_{\geq 0}$
- **约束**：$\sigma_{\varepsilon_l} > 0$；使用 $\max(0, \cdot)$ 确保非负性

> **白话翻译**：系统异常就是"猜得比平时差多少"——如果各层误差能量突然比历史平均值高出很多个标准差，说明系统对当前输入的理解出了问题。

**类型2：报错日志（Error Log Anomaly）**

报错日志度量系统运行时产生的错误和异常事件的频率与严重程度：

$$\text{anomaly}_{\text{err}} = \frac{1}{T_{\text{window}}}\sum_{t \in T_{\text{window}}} \text{severity}(\text{error}_t) \tag{A21.2-err}$$

其中 $T_{\text{window}}$ 为时间窗口，$\text{severity}(\cdot)$ 为错误严重程度的映射函数（如 CRITICAL=1.0, ERROR=0.8, WARNING=0.4, INFO=0.1）。

- **输入**：时间窗口内的错误事件序列 $\{\text{error}_t\}$，严重程度映射 $\text{severity}$
- **输出**：$\text{anomaly}_{\text{err}} \in \mathbb{R}_{\geq 0}$
- **约束**：$\text{severity} \in [0, 1]$；$T_{\text{window}} > 0$

> **白话翻译**：报错日志就是"系统自己报告了多少问题"——CRITICAL级别的错误比WARNING更严重，权重更高。这是最直接的异常信号来源。

**类型3：未知状态（Novelty Anomaly）**

未知状态度量系统遇到前所未见的状态的程度，对应变分自由能中复杂性项的升高：

$$\text{anomaly}_{\text{nov}} = 1 - \max_{k \in \mathcal{K}} \text{sim}(o_t, o_k) \tag{A21.2-nov}$$

其中 $\mathcal{K}$ 为已知状态库，$\text{sim}(\cdot, \cdot)$ 为相似度函数（如余弦相似度）。当当前观测与所有已知状态的相似度都很低时，$\text{anomaly}_{\text{nov}}$ 接近1，表示系统进入了未知领域。

- **输入**：当前观测 $o_t$，已知状态库 $\mathcal{K} = \{o_k\}$
- **输出**：$\text{anomaly}_{\text{nov}} \in [0, 1]$
- **约束**：$\text{sim} \in [0, 1]$；$\mathcal{K} \neq \emptyset$

> **白话翻译**：未知状态就是"从没见过这种情况"——如果当前观测和所有已知经验都不像，说明系统进入了陌生领域，风险自然更高。

**类型4：风险进程（Risk Process Anomaly）**

风险进程度量系统中正在运行的具有潜在风险的进程或操作：

$$\text{anomaly}_{\text{risk}} = \sum_{j \in \mathcal{P}_{\text{risk}}} r_j \cdot \text{impact}_j \tag{A21.2-risk}$$

其中 $\mathcal{P}_{\text{risk}}$ 为风险进程集合，$r_j \in [0, 1]$ 为进程 $j$ 的风险等级，$\text{impact}_j \in [0, 1]$ 为进程 $j$ 的潜在影响程度。

- **输入**：风险进程集合 $\mathcal{P}_{\text{risk}}$，风险等级 $\{r_j\}$，影响程度 $\{\text{impact}_j\}$
- **输出**：$\text{anomaly}_{\text{risk}} \in \mathbb{R}_{\geq 0}$
- **约束**：$r_j \in [0, 1]$；$\text{impact}_j \in [0, 1]$

> **白话翻译**：风险进程就是"正在做危险的事情"——比如修改关键配置、访问敏感数据、执行高权限操作。风险等级越高、影响越大，此项越高。

**类型5：突变负载（Load Shock Anomaly）**

突变负载度量系统负载的突然变化，对应环境动力学中的非平稳性：

$$\text{anomaly}_{\text{load}} = \max\left(0, \frac{|\dot{L}_t| - \bar{L}_{\text{rate}}}{\sigma_{L_{\text{rate}}}}\right) \tag{A21.2-load}$$

其中 $\dot{L}_t$ 为时刻 $t$ 的负载变化率，$\bar{L}_{\text{rate}}$ 和 $\sigma_{L_{\text{rate}}}$ 为负载变化率的历史均值和标准差。

- **输入**：负载变化率 $\dot{L}_t$，历史统计量 $\bar{L}_{\text{rate}}$，$\sigma_{L_{\text{rate}}}$
- **输出**：$\text{anomaly}_{\text{load}} \in \mathbb{R}_{\geq 0}$
- **约束**：$\sigma_{L_{\text{rate}}} > 0$

> **白话翻译**：突变负载就是"工作量突然暴增或暴跌"——如果负载变化率远超历史正常波动范围，说明环境发生了剧变，系统可能来不及适应。

**五类异常信号的汇总**：

| 编号 | 类型 | 符号 | 度量方式 | 与自由能的关联 |
|------|------|------|---------|--------------|
| 1 | 系统异常 | $\text{anomaly}_{\text{sys}}$ | 误差能量偏离基线的标准化 | 准确性项 $A(q)$（引用阶段5，A5.1） |
| 2 | 报错日志 | $\text{anomaly}_{\text{err}}$ | 错误频率与严重程度加权 | 系统运行状态的外部指标 |
| 3 | 未知状态 | $\text{anomaly}_{\text{nov}}$ | 与已知状态的最大相似度补 | 复杂性项 $C(q)$（引用阶段5，A5.5） |
| 4 | 风险进程 | $\text{anomaly}_{\text{risk}}$ | 风险等级与影响程度加权 | 行动空间的约束违反 |
| 5 | 突变负载 | $\text{anomaly}_{\text{load}}$ | 负载变化率偏离基线 | 环境非平稳性 |

---

### 第三部分：异常权重自适应

#### 2.3 异常权重自适应（A21.3）

不同异常信号的危险程度不同，且随上下文动态变化。异常权重通过softmax函数自适应调整：

$$w_i = \frac{\exp(\beta_s \cdot \text{severity}_i)}{\sum_{j=1}^{N_{\text{anomaly}}} \exp(\beta_s \cdot \text{severity}_j)} \tag{A21.3}$$

其中：
- $\text{severity}_i \in \mathbb{R}$：第 $i$ 类异常的严重程度标量，由异常信号值和历史上下文共同决定
- $\beta_s > 0$：温度参数，控制权重分布的锐度

- **输入**：严重程度向量 $\{\text{severity}_i\}_{i=1}^{N_{\text{anomaly}}}$，温度参数 $\beta_s$
- **输出**：权重向量 $\{w_i\}_{i=1}^{N_{\text{anomaly}}}$，满足 $w_i \geq 0$，$\sum_i w_i = 1$
- **约束**：$\beta_s > 0$；$\text{severity}_i \in \mathbb{R}$

> **白话翻译**：异常权重自适应就是"哪个警报更可怕就听哪个"——严重程度越高的异常，权重自动越大。$\beta_s$ 是"注意力集中度"——$\beta_s$ 大时，系统只关注最严重的异常（赢者通吃）；$\beta_s$ 小时，系统均匀关注所有异常（雨露均沾）。

**严重程度的计算**：

$$\text{severity}_i = \ln\left(1 + \frac{\text{anomaly}_i}{\text{threshold}_i}\right) \tag{A21.3-sev}$$

其中 $\text{threshold}_i$ 为第 $i$ 类异常的阈值。使用对数变换确保严重程度在异常值很大时不会爆炸式增长，同时保持单调递增。

**温度参数 $\beta_s$ 的影响**：

| $\beta_s$ 值 | 权重分布 | 行为特征 |
|-------------|---------|---------|
| $\beta_s \to 0$ | $w_i \approx 1/N_{\text{anomaly}}$（均匀分布） | 均匀关注所有异常 |
| $\beta_s = 1$ | 中等锐度 | 按严重程度适度加权 |
| $\beta_s \to \infty$ | $w_{i^*} \to 1$（赢者通吃） | 只关注最严重的异常 |

**自适应权重的动态性**：当某类异常的严重程度突然升高时，其权重自动增大，使安全驱力更敏感地响应最紧迫的威胁。当所有异常的严重程度相近时，权重趋于均匀分布，安全驱力综合考虑所有因素。

---

### 第四部分：安全驱力与自由能的关系

#### 2.4 安全驱力与自由能的关系（A21.4）

安全驱力与变分自由能偏离基线的程度直接相关。定义自由能基线 $F_{\text{baseline}}$ 为系统在正常运行状态下的自由能水平（如历史均值），则安全驱力可表示为：

$$R_{\text{safe}} = -\alpha_s \cdot (F - F_{\text{baseline}}) \tag{A21.4}$$

其中 $F$ 为当前变分自由能（引用阶段4，A4.4），$F_{\text{baseline}}$ 为自由能基线。

- **输入**：当前自由能 $F$，自由能基线 $F_{\text{baseline}}$，缩放系数 $\alpha_s$
- **输出**：$R_{\text{safe}} \in \mathbb{R}$
- **约束**：$F \geq -\ln p(o)$（自由能下界约束，引用阶段4，A4.6）；$F_{\text{baseline}}$ 为系统正常运行时的自由能均值

> **白话翻译**：安全驱力 = "账单比平时高多少"——自由能就是"总账单"，基线就是"平时的账单水平"。账单突然升高，说明系统出了问题，安全驱力变负，驱动系统回避风险。

**A21.4与A21.1的等价性论证**：

由A4.8（引用阶段4），自由能在拉普拉斯近似下为 $F \approx \varepsilon_0 + \frac{1}{2}\sum_{l=1}^{L} \varepsilon_l + \frac{1}{2}\sum_{l=1}^{L} D_{\text{KL}}[q_l \| p_l]$。自由能偏离基线的量为：

$$F - F_{\text{baseline}} = \Delta\varepsilon_0 + \frac{1}{2}\sum_{l=1}^{L} \Delta\varepsilon_l + \frac{1}{2}\sum_{l=1}^{L} \Delta D_{\text{KL},l}$$

其中 $\Delta\varepsilon_l = \varepsilon_l - \bar{\varepsilon}_l$ 为误差能量偏离基线的量，$\Delta D_{\text{KL},l}$ 为KL散度偏离基线的量。

系统异常（A21.2-sys）度量 $\Delta\varepsilon_l$ 的标准化值，未知状态（A21.2-nov）度量 $\Delta D_{\text{KL},l}$ 的间接指标，报错日志和风险进程是自由能恶化在系统层面的外在表现，突变负载是自由能恶化的环境原因。因此，A21.1的加权和形式与A21.4的自由能偏离形式在物理含义上一致——它们从不同角度度量同一个底层现象：系统偏离安全稳态。

**自由能基线的确定**：

$$F_{\text{baseline}} = \frac{1}{T_{\text{hist}}}\sum_{t=1}^{T_{\text{hist}}} F_t \cdot \mathbb{1}[\text{system\_normal}(t)] \tag{A21.4-baseline}$$

其中 $T_{\text{hist}}$ 为历史窗口长度，$\mathbb{1}[\text{system\_normal}(t)]$ 为系统正常运行指示函数。基线仅使用正常运行时段的自由能值，排除异常时段的干扰。

**安全驱力的边界性质**：

1. 当 $F = F_{\text{baseline}}$ 时，$R_{\text{safe}} = 0$（系统处于正常状态）
2. 当 $F > F_{\text{baseline}}$ 时，$R_{\text{safe}} < 0$（系统偏离正常，需要回避）
3. 当 $F < F_{\text{baseline}}$ 时，$R_{\text{safe}} > 0$（系统比正常更好，但这通常被截断为零，因为"比正常更安全"不应产生正向驱力）

为避免"比正常更安全"产生不合理的正向驱力，引入截断：

$$R_{\text{safe}} = -\alpha_s \cdot \max\left(0, F - F_{\text{baseline}}\right) \tag{A21.4-truncated}$$

截断后的安全驱力满足 $R_{\text{safe}} \leq 0$，与A21.1的符号约定一致。

---

### 第五部分：安全驱力的时间平滑

#### 2.5 安全驱力的时间平滑（A21.5）

安全驱力的原始值可能因瞬时噪声而剧烈波动，需要时间平滑以产生稳定的驱力信号：

$$R_{\text{safe}}^{\text{smooth}}(t) = \gamma_s \cdot R_{\text{safe}}^{\text{smooth}}(t-1) + (1 - \gamma_s) \cdot R_{\text{safe}}(t) \tag{A21.5}$$

其中 $\gamma_s \in [0, 1)$ 为平滑系数。

- **输入**：当前安全驱力 $R_{\text{safe}}(t)$，上一时刻平滑驱力 $R_{\text{safe}}^{\text{smooth}}(t-1)$，平滑系数 $\gamma_s$
- **输出**：$R_{\text{safe}}^{\text{smooth}}(t) \in \mathbb{R}_{\leq 0}$
- **约束**：$\gamma_s \in [0, 1)$；初始值 $R_{\text{safe}}^{\text{smooth}}(0) = 0$

> **白话翻译**：时间平滑就是"警报不能忽高忽低"——如果警报音量每秒都在剧烈变化，系统会无所适从。平滑处理让驱力信号像"缓坡"而不是"台阶"，使系统的回避行为更稳定。

**平滑系数 $\gamma_s$ 的影响**：

| $\gamma_s$ 值 | 等效时间常数 | 行为特征 |
|-------------|------------|---------|
| $\gamma_s = 0$ | $\tau = 1$ 步 | 无平滑，驱力即时响应 |
| $\gamma_s = 0.9$ | $\tau = 10$ 步 | 中等平滑，兼顾响应速度和稳定性 |
| $\gamma_s = 0.99$ | $\tau = 100$ 步 | 强平滑，驱力变化缓慢但极稳定 |

**时间平滑的等效时间常数**：

指数移动平均的等效时间常数为 $\tau_s = -1/\ln(\gamma_s)$。当 $\gamma_s$ 接近1时，$\tau_s \approx 1/(1-\gamma_s)$。

**时间平滑的稳定性保证**：

由于 $R_{\text{safe}}(t) \leq 0$ 且 $\gamma_s \in [0, 1)$，由数学归纳法：若 $R_{\text{safe}}^{\text{smooth}}(t-1) \leq 0$，则 $R_{\text{safe}}^{\text{smooth}}(t) = \gamma_s \cdot R_{\text{safe}}^{\text{smooth}}(t-1) + (1-\gamma_s) \cdot R_{\text{safe}}(t) \leq 0$。因此时间平滑不改变安全驱力的非正性。

**时间平滑与异常检测的延迟**：

时间平滑引入响应延迟。设异常在时刻 $t_0$ 突然出现，$R_{\text{safe}}(t)$ 从0跳变到 $R_0 < 0$，则平滑后的驱力响应为：

$$R_{\text{safe}}^{\text{smooth}}(t) = R_0 \cdot (1 - \gamma_s^{t - t_0}), \quad t \geq t_0$$

达到90%响应的时间为 $t_{90\%} = t_0 + \ln(10)/\ln(1/\gamma_s) \approx t_0 + 2.3\tau_s$。因此 $\gamma_s$ 的选择需要在"响应速度"和"信号稳定性"之间权衡。

**异常消失后的恢复**：

当异常在时刻 $t_1$ 消失，$R_{\text{safe}}(t)$ 回到0，平滑驱力的恢复过程为：

$$R_{\text{safe}}^{\text{smooth}}(t) = R_{\text{safe}}^{\text{smooth}}(t_1) \cdot \gamma_s^{t - t_1}, \quad t \geq t_1$$

恢复到10%残留的时间同样为 $t_{10\%} = t_1 + 2.3\tau_s$。这种"缓慢恢复"特性防止系统在异常刚消失时就立即放松警惕——类似于生物体的"应激后恢复期"。

---

### 第六部分：安全驱力的分层结构

安全驱力可以按层级分解，与预测编码的分层结构（引用阶段3，A3.1-A3.4）对应：

$$R_{\text{safe}} = \sum_{l=0}^{L} R_{\text{safe}}^l \tag{A21.6}$$

其中第 $l$ 层的安全驱力分量为：

$$R_{\text{safe}}^l = -\alpha_s \cdot w_l \cdot \max\left(0, \frac{\varepsilon_l - \bar{\varepsilon}_l}{\sigma_{\varepsilon_l}}\right) \tag{A21.6-layer}$$

底层（$l=0$）的安全驱力对应感知层的异常——感官输入与预测的偏差；高层（$l=L$）的安全驱力对应抽象层的异常——高层信念与先验的冲突。分层结构使得安全驱力可以精确定位异常来源，指导针对性的回避行动。

**分层安全驱力与A4.9的连接**：由A4.9（引用阶段4），$\partial F / \partial r^l = \delta^{l+1} - e^l$。当第 $l$ 层的安全驱力分量 $R_{\text{safe}}^l$ 显著为负时，意味着 $\varepsilon_l$ 异常升高，即 $e^l$ 或 $\delta^{l+1}$ 的幅度异常增大。安全驱力的分层分解与自由能梯度的分层结构完全对应——安全驱力是自由能梯度的"标量摘要"，而自由能梯度是安全驱力的"向量细节"。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A21.1 | $R_{\text{safe}} = -\alpha_s \cdot \sum_i w_i \cdot \text{anomaly}_i$ | $\{\text{anomaly}_i\}$, $\{w_i\}$, $\alpha_s$ | $R_{\text{safe}} \in \mathbb{R}_{\leq 0}$ | $\alpha_s > 0$；$w_i \geq 0$；$\sum w_i = 1$ |
| A21.2 | 五类异常信号定义 | 各类异常原始数据 | $\text{anomaly}_i \in \mathbb{R}_{\geq 0}$ | 各类约束见A21.2子公式 |
| A21.3 | $w_i = \text{softmax}(\beta_s \cdot \text{severity}_i)$ | $\{\text{severity}_i\}$, $\beta_s$ | $\{w_i\}$，归一化概率 | $\beta_s > 0$ |
| A21.4 | $R_{\text{safe}} = -\alpha_s \cdot \max(0, F - F_{\text{baseline}})$ | $F$, $F_{\text{baseline}}$, $\alpha_s$ | $R_{\text{safe}} \in \mathbb{R}_{\leq 0}$ | 引用阶段4，A4.4 |
| A21.5 | $R_{\text{safe}}^{\text{smooth}} = \gamma_s \cdot R_{\text{safe}}^{\text{prev}} + (1-\gamma_s) \cdot R_{\text{safe}}$ | $R_{\text{safe}}(t)$, $R_{\text{safe}}^{\text{smooth}}(t-1)$, $\gamma_s$ | $R_{\text{safe}}^{\text{smooth}} \in \mathbb{R}_{\leq 0}$ | $\gamma_s \in [0, 1)$ |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# 引用阶段20的数据类
from stage_020 import PCValidationResult


# ============================================================
# 阶段21 数据类：安全驱力
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class SafetyDrive:
    """安全驱力的完整描述。

    组合阶段20的PCValidationResult，在预测编码验证结果的基础上增加
    安全驱力的计算结果。

    Attributes:
        validation_result: 阶段20的预测编码验证结果（组合字段）
        alpha_s: 安全驱力全局缩放系数（A21.1）
        anomalies: 五类异常信号值元组（A21.2）
        severities: 各异常严重程度元组（A21.3-sev）
        weights: 自适应权重元组（A21.3）
        beta_s: softmax温度参数（A21.3）
        free_energy_current: 当前自由能值（A21.4）
        free_energy_baseline: 自由能基线值（A21.4）
        raw_safe_reward: 原始安全驱力值（A21.1/A21.4）
        smooth_safe_reward: 时间平滑后的安全驱力值（A21.5）
        gamma_s: 时间平滑系数（A21.5）

    Invariants:
        - alpha_s > 0
        - all(a >= 0 for a in anomalies)
        - all(w >= 0 for w in weights) and sum(weights) ≈ 1
        - beta_s > 0
        - gamma_s ∈ [0, 1)
        - raw_safe_reward <= 0
        - smooth_safe_reward <= 0
    """
    validation_result: PCValidationResult
    alpha_s: float
    anomalies: tuple[float, ...]       # 5类异常
    severities: tuple[float, ...]      # 5类严重程度
    weights: tuple[float, ...]         # 5类权重
    beta_s: float
    free_energy_current: float
    free_energy_baseline: float
    raw_safe_reward: float
    smooth_safe_reward: float
    gamma_s: float

    def __post_init__(self):
        # 验证缩放系数
        if self.alpha_s <= 0:
            raise ValueError(
                f"安全驱力缩放系数必须 > 0，"
                f"收到 {self.alpha_s}"
            )
        # 验证异常信号非负
        for i, a in enumerate(self.anomalies):
            if a < 0:
                raise ValueError(
                    f"第 {i} 类异常信号必须 ≥ 0，"
                    f"收到 {a}"
                )
        # 验证权重归一化
        w_sum = sum(self.weights)
        if not (0.99 <= w_sum <= 1.01):
            raise ValueError(
                f"权重之和必须 ≈ 1，"
                f"收到 {w_sum}"
            )
        # 验证温度参数
        if self.beta_s <= 0:
            raise ValueError(
                f"温度参数必须 > 0，"
                f"收到 {self.beta_s}"
            )
        # 验证平滑系数
        if not (0 <= self.gamma_s < 1):
            raise ValueError(
                f"平滑系数必须 ∈ [0, 1)，"
                f"收到 {self.gamma_s}"
            )
        # 验证安全驱力非正
        if self.raw_safe_reward > 1e-10:
            raise ValueError(
                f"原始安全驱力必须 ≤ 0，"
                f"收到 {self.raw_safe_reward}"
            )
        if self.smooth_safe_reward > 1e-10:
            raise ValueError(
                f"平滑安全驱力必须 ≤ 0，"
                f"收到 {self.smooth_safe_reward}"
            )

    @property
    def is_safe(self) -> bool:
        """系统是否处于安全状态（安全驱力接近零）。"""
        return self.smooth_safe_reward > -0.1 * self.alpha_s

    @property
    def dominant_anomaly(self) -> int:
        """主导异常类型索引（权重最大的异常）。"""
        return int(np.argmax(self.weights))

    @property
    def free_energy_deviation(self) -> float:
        """自由能偏离基线的程度。"""
        return max(0.0, self.free_energy_current - self.free_energy_baseline)


# ============================================================
# 阶段21 核心接口：安全驱力计算器
# ============================================================
class SafetyDriveCalculator:
    """阶段21的核心计算接口。

    实现安全驱力的完整计算流程，包括异常信号提取、
    权重自适应、自由能关联和时间平滑。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - R_safe ≤ 0（安全驱力恒非正）
        - 权重归一化 ∑w_i = 1
        - 时间平滑保持非正性
    """

    def compute_anomalies(
        self,
        error_energies: tuple[float, ...],
        error_energy_stats: tuple[tuple[float, float], ...],
        error_events: list[tuple[str, float]],
        observation: np.ndarray,
        known_states: np.ndarray,
        risk_processes: list[tuple[float, float]],
        load_rate: float,
        load_rate_stats: tuple[float, float]
    ) -> tuple[float, ...]:
        """计算五类异常信号（A21.2）。

        Args:
            error_energies: 各层精度加权误差能量 (ε_0, ..., ε_L)，shape (L+1,)
            error_energy_stats: 各层历史统计量 ((μ_0, σ_0), ..., (μ_L, σ_L))，shape (L+1, 2)
            error_events: 错误事件列表 [(level, severity), ...]，severity ∈ [0, 1]
            observation: 当前观测，shape (n_features,)
            known_states: 已知状态库，shape (n_states, n_features)
            risk_processes: 风险进程 [(risk_level, impact), ...]，risk_level ∈ [0, 1], impact ∈ [0, 1]
            load_rate: 负载变化率（单位：资源量/秒）
            load_rate_stats: 负载变化率统计量 (mean, std)（单位：资源量/秒）

        Returns:
            五类异常信号值元组 (anomaly_sys, anomaly_err, anomaly_nov, anomaly_risk, anomaly_load)

        Raises:
            ValueError: 当 error_energies 与 error_energy_stats 长度不一致
            ValueError: 当 load_rate_stats 中 std <= 0
            ValueError: 当 known_states 为空

        Examples:
            >>> calc = SafetyDriveCalculator()
            >>> calc.compute_anomalies(
            ...     error_energies=(0.5, 0.3, 0.2),
            ...     error_energy_stats=((0.4, 0.1), (0.3, 0.05), (0.2, 0.05)),
            ...     error_events=[("ERROR", 0.8)],
            ...     observation=np.array([1.0, 0.0]),
            ...     known_states=np.array([[1.0, 0.0], [0.0, 1.0]]),
            ...     risk_processes=[(0.5, 0.3)],
            ...     load_rate=0.1,
            ...     load_rate_stats=(0.05, 0.02)
            ... )
            (0.3333..., 0.8, 0.0, 0.15, 2.5)
        """
        if len(error_energies) != len(error_energy_stats):
            raise ValueError(
                f"error_energies 长度 ({len(error_energies)}) 必须与 "
                f"error_energy_stats 长度 ({len(error_energy_stats)}) 一致"
            )
        # A21.2-sys: 系统异常
        sys_vals = []
        for eps, (mu, sigma) in zip(error_energies, error_energy_stats):
            if sigma <= 0:
                raise ValueError(f"误差能量标准差必须 > 0，收到 {sigma}")
            sys_vals.append(max(0.0, (eps - mu) / sigma))
        anomaly_sys = float(np.mean(sys_vals)) if sys_vals else 0.0

        # A21.2-err: 报错日志异常
        severity_sum = sum(sev for _, sev in error_events)
        anomaly_err = severity_sum / max(1, len(error_events))

        # A21.2-nov: 未知状态异常
        if known_states.size == 0:
            raise ValueError("known_states 不能为空")
        obs_norm = observation / (np.linalg.norm(observation) + 1e-12)
        states_norm = known_states / (np.linalg.norm(known_states, axis=1, keepdims=True) + 1e-12)
        similarities = np.dot(states_norm, obs_norm)
        max_sim = float(np.max(similarities)) if similarities.size > 0 else 0.0
        anomaly_nov = max(0.0, 1.0 - max_sim)

        # A21.2-risk: 风险进程异常
        anomaly_risk = sum(r * imp for r, imp in risk_processes)

        # A21.2-load: 突变负载异常
        load_mean, load_std = load_rate_stats
        if load_std <= 0:
            raise ValueError(f"负载变化率标准差必须 > 0，收到 {load_std}")
        anomaly_load = max(0.0, (abs(load_rate) - load_mean) / load_std)

        return (anomaly_sys, anomaly_err, anomaly_nov, anomaly_risk, anomaly_load)

    def compute_weights(
        self,
        severities: tuple[float, ...],
        beta_s: float
    ) -> tuple[float, ...]:
        """计算自适应权重（A21.3）。

        Args:
            severities: 各异常严重程度，长度 N_anomaly
            beta_s: softmax温度参数，> 0（无量纲）

        Returns:
            归一化权重元组，满足 sum(weights) == 1，每个 weight >= 0

        Raises:
            ValueError: 当 beta_s <= 0
            ValueError: 当 severities 为空

        Examples:
            >>> calc = SafetyDriveCalculator()
            >>> calc.compute_weights((1.0, 0.5, 0.2), beta_s=1.0)
            (0.506..., 0.307..., 0.186...)
        """
        if beta_s <= 0:
            raise ValueError(f"beta_s 必须 > 0，收到 {beta_s}")
        if len(severities) == 0:
            raise ValueError("severities 不能为空")
        arr = np.array(severities, dtype=float)
        exp_vals = np.exp(beta_s * arr - np.max(beta_s * arr))
        weights = exp_vals / (np.sum(exp_vals) + 1e-12)
        return tuple(float(w) for w in weights)

    def compute_safe_reward(
        self,
        anomalies: tuple[float, ...],
        weights: tuple[float, ...],
        alpha_s: float
    ) -> float:
        """计算原始安全驱力（A21.1）。

        Args:
            anomalies: 异常信号值，每个 >= 0
            weights: 权重，满足 sum(weights) == 1，每个 >= 0
            alpha_s: 缩放系数，> 0

        Returns:
            R_safe = -α_s · Σ w_i · anomaly_i，返回值 <= 0

        Raises:
            ValueError: 当 alpha_s <= 0
            ValueError: 当 anomalies 与 weights 长度不一致

        Examples:
            >>> calc = SafetyDriveCalculator()
            >>> calc.compute_safe_reward(
            ...     anomalies=(1.0, 0.5, 0.0, 0.2, 0.1),
            ...     weights=(0.2, 0.2, 0.2, 0.2, 0.2),
            ...     alpha_s=1.0
            ... )
            -0.36
        """
        if alpha_s <= 0:
            raise ValueError(f"alpha_s 必须 > 0，收到 {alpha_s}")
        if len(anomalies) != len(weights):
            raise ValueError(
                f"anomalies 长度 ({len(anomalies)}) 必须与 weights 长度 ({len(weights)}) 一致"
            )
        weighted_sum = sum(float(w) * float(a) for w, a in zip(weights, anomalies))
        return -alpha_s * weighted_sum

    def compute_safe_reward_from_free_energy(
        self,
        free_energy: float,
        free_energy_baseline: float,
        alpha_s: float
    ) -> float:
        """从自由能计算安全驱力（A21.4）。

        Args:
            free_energy: 当前自由能（单位：nats）
            free_energy_baseline: 自由能基线（单位：nats）
            alpha_s: 缩放系数，> 0

        Returns:
            R_safe = -α_s · max(0, F - F_baseline) <= 0

        Raises:
            ValueError: 当 alpha_s <= 0

        Examples:
            >>> calc = SafetyDriveCalculator()
            >>> calc.compute_safe_reward_from_free_energy(2.0, 1.5, 1.0)
            -0.5
            >>> calc.compute_safe_reward_from_free_energy(1.0, 1.5, 1.0)
            0.0
        """
        if alpha_s <= 0:
            raise ValueError(f"alpha_s 必须 > 0，收到 {alpha_s}")
        deviation = max(0.0, free_energy - free_energy_baseline)
        return -alpha_s * deviation

    def smooth_safe_reward(
        self,
        current: float,
        previous_smooth: float,
        gamma_s: float
    ) -> float:
        """时间平滑（A21.5）。

        Args:
            current: 当前安全驱力（单位：无量纲）
            previous_smooth: 上一时刻平滑值（单位：无量纲）
            gamma_s: 平滑系数，∈ [0, 1)

        Returns:
            平滑后的安全驱力，满足返回值 <= 0（若 current <= 0 且 previous_smooth <= 0）

        Raises:
            ValueError: 当 gamma_s 不在 [0, 1) 区间

        Examples:
            >>> calc = SafetyDriveCalculator()
            >>> calc.smooth_safe_reward(-1.0, -0.5, 0.9)
            -0.6
        """
        if not (0 <= gamma_s < 1):
            raise ValueError(f"gamma_s 必须 ∈ [0, 1)，收到 {gamma_s}")
        return gamma_s * previous_smooth + (1.0 - gamma_s) * current
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `PCValidationResult` | ← 阶段20 | 作为安全驱力的验证结果组合字段 |
| **输入** | A4.4（变分自由能定义） | ← 阶段4 | 安全驱力与自由能偏离的定量关系 |
| **输入** | A5.1（准确性项） | ← 阶段5 | 系统异常信号的准确性来源 |
| **输入** | A3.2（精度加权误差能量） | ← 阶段3 | 分层异常信号的度量基础 |
| **输出** | `SafetyDrive` | → 阶段22 | 安全驱力作为收益驱力的组合字段 |
| **输出** | A21.4（安全驱力-自由能关系） | → 阶段24 | 全局奖励公式的安全分量 |
| **输出** | A21.5（时间平滑） | → 阶段24 | 平滑驱力用于权重切换 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 异常信号计算 | `jax.vmap`批量计算各层误差能量的标准化异常值（A21.2-sys）；`jax.nn.softmax`实现自适应权重（A21.3）；`jax.lax.scan`实现时间平滑的函数式循环（A21.5） |
| NumPy | 统计计算 | 历史均值和标准差的滑动窗口计算；相似度计算（A21.2-nov） |
| prometheus_client | **本阶段新增**：系统监控 | 提供系统级指标采集（CPU/内存/错误率/负载），直接支撑A21.2中报错日志、风险进程、突变负载三类异常信号的实时采集与度量 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V21.1 | 安全驱力非正性 | $R_{\text{safe}} \leq 0$ 对所有合法输入成立 | 随机异常信号测试 |
| V21.2 | 权重归一化 | $\sum_i w_i = 1$ 对所有 $\beta_s > 0$ 成立 | softmax数值验证 |
| V21.3 | 权重自适应 | 严重程度最高的异常权重最大 | 极端严重程度测试 |
| V21.4 | 自由能-安全驱力一致性 | A21.1与A21.4在正常范围内数值一致 | 合成自由能数据对比 |
| V21.5 | 截断正确性 | $F < F_{\text{baseline}}$ 时 $R_{\text{safe}} = 0$ | 低于基线自由能测试 |
| V21.6 | 时间平滑非正性 | $R_{\text{safe}}^{\text{smooth}} \leq 0$ 对所有 $\gamma_s \in [0,1)$ 成立 | 平滑迭代数值验证 |
| V21.7 | 时间平滑收敛 | 异常消失后平滑驱力指数衰减到零 | 长期迭代测试 |
| V21.8 | 系统异常度量 | 误差能量高于基线时 $\text{anomaly}_{\text{sys}} > 0$ | 合成误差能量测试 |
| V21.9 | 未知状态度量 | 未知观测的 $\text{anomaly}_{\text{nov}} > 0.5$ | 新颖性检测测试 |
| V21.10 | 突变负载度量 | 负载突变时 $\text{anomaly}_{\text{load}} > 0$ | 负载冲击模拟 |
| V21.11 | 温度参数影响 | $\beta_s \to \infty$ 时权重趋近赢者通吃 | 极端温度参数测试 |
| V21.12 | SafetyDrive组合 | `SafetyDrive.validation_result` 类型为 `PCValidationResult` | 类型检查 |
| V21.13 | 非法输入拒绝 | $\alpha_s \leq 0$、$\beta_s \leq 0$、$\gamma_s \notin [0,1)$ 等抛出 `ValueError` | 异常测试 |

---

> **阶段21完** | 下一步：阶段22将从本阶段的安全驱力出发，定义收益驱力 $R_{\text{gain}}$，组合安全驱力形成更完整的驱力体系。
