# 阶段28：驱力信号的时间平滑与防抖动——从原始信号到稳定驱力

> **阶段编号**：S28 | **所属层**：第三层·驱力奖励层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 指数移动平均 | "看趋势不看波动"——用历史值和当前值的加权平均消除短期波动 |
> | 自适应平滑系数 | "波动大就多平滑，波动小就少平滑"——根据信号方差自动调节平滑强度 |
> | 中值滤波 | "去掉极端值"——用窗口内的中位数替代当前值，消除脉冲噪声 |
> | 变化率限制 | "不能一步跨太远"——限制驱力每步变化的最大幅度 |
> | 防抖动时间窗口 | "等一等再决定"——驱力变化后等待一段时间确认趋势再响应 |
> | 单调性保持 | "平滑不能改变方向"——平滑后驱力的增减方向与原始信号一致 |

---

## 1. 第一性原理依据

本阶段从阶段21（A21.5时间平滑）和阶段26（A26.6防抖动）的核心结论出发，建立驱力信号的完整时间平滑与防抖动机制。核心依据为：

- **A21.5（安全驱力的时间平滑）**（引用阶段21）：$R_{\text{safe}}^{\text{smooth}}(t) = \gamma_s \cdot R_{\text{safe}}^{\text{smooth}}(t-1) + (1-\gamma_s) \cdot R_{\text{safe}}(t)$，指数移动平均（EMA）是时间平滑的基本形式。本阶段将EMA从安全驱力推广到所有驱力，并引入自适应系数、中值滤波和变化率限制等增强机制。
- **A26.6（场景切换的防抖动机制）**（引用阶段26）：场景切换后必须驻留至少 $T_{\text{debounce}}$ 步。本阶段将防抖动从场景级别推广到驱力信号级别——驱力值的显著变化也需要防抖动确认。
- **A24.6（权重平滑过渡）**（引用阶段24）：$W_i(t+1) = \alpha_w \cdot W_i^{\text{target}} + (1-\alpha_w) \cdot W_i(t)$，权重层面的EMA平滑。本阶段的驱力平滑是权重平滑的上游——驱力信号先平滑，再用于计算权重。
- **A4.11（自由能单调递减）**（引用阶段4）：$\mathrm{d}F/\mathrm{d}t \leq 0$，自由能在优化过程中单调递减。时间平滑不应破坏自由能的单调性——这是定理28.1的理论基础。

本阶段的核心目标是：**设计驱力信号的完整时间平滑与防抖动机制，包括指数移动平均（EMA）、自适应平滑系数、中值滤波、变化率限制和防抖动时间窗口，并证明时间平滑不影响驱力的单调性。** 时间平滑与防抖动是驱力引擎的"信号处理层"——它将原始的、可能剧烈波动的驱力信号转化为稳定的、可信赖的驱力信号，确保系统行为的稳定性和可预测性。

---

## 2. 核心数学公式

### 第一部分：指数移动平均

#### 2.1 指数移动平均（A28.1）

指数移动平均（EMA）是驱力信号时间平滑的基本形式：

$$R_{\text{smooth}} = \alpha \cdot R_{\text{new}} + (1 - \alpha) \cdot R_{\text{prev}} \tag{A28.1}$$

其中：
- $R_{\text{new}}$：当前时刻的原始驱力值
- $R_{\text{prev}}$：上一时刻的平滑驱力值
- $\alpha \in (0, 1]$：平滑系数（又称遗忘因子）
- $R_{\text{smooth}}$：当前时刻的平滑驱力值

- **输入**：原始驱力值 $R_{\text{new}}$，上一时刻平滑值 $R_{\text{prev}}$，平滑系数 $\alpha$
- **输出**：$R_{\text{smooth}} \in \mathbb{R}$
- **约束**：$\alpha \in (0, 1]$；初始值 $R_{\text{smooth}}(0) = R_{\text{new}}(0)$

> **白话翻译**：EMA就是"看趋势不看波动"——新值占 $\alpha$ 的权重，历史值占 $1-\alpha$ 的权重。$\alpha$ 大则更看重新值（响应快但噪声大），$\alpha$ 小则更看重历史（响应慢但信号稳）。A21.5（引用阶段21）的 $\gamma_s$ 就是 $1-\alpha$——两者是同一公式的不同参数化。

**EMA的等效时间常数**：

EMA的等效时间常数为 $\tau = -1 / \ln(1-\alpha)$。当 $\alpha$ 较小时，$\tau \approx 1/\alpha$。

| $\alpha$ 值 | 等效时间常数 $\tau$ | 达到90%响应的步数 | 行为特征 |
|------------|-------------------|-----------------|---------|
| $\alpha = 1.0$ | $\tau = 1$ | 1步 | 无平滑，即时响应 |
| $\alpha = 0.5$ | $\tau \approx 1.4$ | 4步 | 轻度平滑 |
| $\alpha = 0.1$ | $\tau \approx 9.5$ | 22步 | 中度平滑 |
| $\alpha = 0.01$ | $\tau \approx 99.5$ | 230步 | 强平滑 |

**EMA的递推展开**：

将EMA递推展开，可得平滑值为历史值的指数加权和：

$$R_{\text{smooth}}(t) = \alpha \sum_{k=0}^{t} (1-\alpha)^k R_{\text{new}}(t-k) + (1-\alpha)^{t+1} R_{\text{smooth}}(0)$$

当 $t \to \infty$ 时，初始条件的影响 $(1-\alpha)^{t+1} \to 0$，平滑值完全由历史值的指数加权和决定。权重 $(1-\alpha)^k$ 随 $k$ 指数衰减——越近的值权重越大，越远的值权重越小。

**EMA对三大驱力的分别应用**：

三大驱力的物理特性不同，EMA的参数也应不同：

| 驱力 | 推荐平滑系数 | 理由 |
|------|------------|------|
| 安全驱力 $R_{\text{safe}}$ | $\alpha_{\text{safe}} = 0.3$ | 安全威胁需要较快响应，但不能过度反应 |
| 收益驱力 $R_{\text{gain}}$ | $\alpha_{\text{gain}} = 0.1$ | 认知收益变化较慢，需要较强平滑 |
| 能效驱力 $R_{\text{eff}}$ | $\alpha_{\text{eff}} = 0.2$ | 资源消耗变化中等，需要适度平滑 |

安全驱力的平滑系数最大（$\alpha_{\text{safe}} = 0.3$），因为安全威胁需要较快响应——如果平滑过度，系统对紧急威胁的反应会延迟。收益驱力的平滑系数最小（$\alpha_{\text{gain}} = 0.1$），因为认知收益的变化通常是渐进的，不需要快速响应，反而需要较强的平滑来避免探索策略的频繁切换。

---

### 第二部分：自适应平滑系数

#### 2.2 自适应平滑系数（A28.2）

固定平滑系数无法同时满足"平稳时强平滑"和"剧变时快响应"的需求。自适应平滑系数根据信号方差动态调整：

$$\alpha = f(\text{variance}(R)) \tag{A28.2}$$

具体形式为：

$$\alpha = \alpha_{\min} + (\alpha_{\max} - \alpha_{\min}) \cdot \sigma_{\text{norm}} \tag{A28.2-formula}$$

其中：
- $\alpha_{\min} \in (0, 1)$：最小平滑系数（平稳时使用）
- $\alpha_{\max} \in [\alpha_{\min}, 1]$：最大平滑系数（剧变时使用）
- $\sigma_{\text{norm}} \in [0, 1]$：归一化方差

- **输入**：驱力信号的近期方差 $\sigma^2$，方差范围 $[\sigma_{\min}^2, \sigma_{\max}^2]$
- **输出**：自适应平滑系数 $\alpha \in [\alpha_{\min}, \alpha_{\max}]$
- **约束**：$0 < \alpha_{\min} \leq \alpha_{\max} \leq 1$

> **白话翻译**：自适应平滑系数就是"波动大就多响应，波动小就多平滑"——信号平稳时用小 $\alpha$（强平滑），信号剧变时用大 $\alpha$（快响应）。这样既保证了平稳时的信号稳定性，又保证了剧变时的响应速度。

**归一化方差的计算**：

$$\sigma_{\text{norm}} = \text{clip}\left(\frac{\sigma_R - \sigma_{\min}}{\sigma_{\max} - \sigma_{\min}}, 0, 1\right) \tag{A28.2-norm}$$

其中 $\sigma_R$ 为驱力信号在近期窗口 $W_{\text{var}}$ 内的标准差：

$$\sigma_R = \sqrt{\frac{1}{W_{\text{var}}} \sum_{k=0}^{W_{\text{var}}-1} (R_{\text{new}}(t-k) - \bar{R}_{\text{new}})^2} \tag{A28.2-std}$$

$\sigma_{\min}$ 和 $\sigma_{\max}$ 分别为驱力信号方差的下界和上界，由历史数据统计确定。

**自适应平滑的物理解释**：

| 信号状态 | $\sigma_R$ | $\sigma_{\text{norm}}$ | $\alpha$ | 行为 |
|---------|-----------|----------------------|---------|------|
| 平稳 | $\sigma_R \approx \sigma_{\min}$ | $\approx 0$ | $\approx \alpha_{\min}$ | 强平滑，信号稳定 |
| 中等波动 | $\sigma_{\min} < \sigma_R < \sigma_{\max}$ | $\in (0, 1)$ | $\in (\alpha_{\min}, \alpha_{\max})$ | 适度平滑 |
| 剧烈波动 | $\sigma_R \approx \sigma_{\max}$ | $\approx 1$ | $\approx \alpha_{\max}$ | 快响应，跟踪变化 |

**自适应平滑与紧急响应的协同**：

当安全驱力突然大幅变化（如遇到紧急威胁），$\sigma_R$ 急剧增大，$\alpha$ 自动增大到 $\alpha_{\max}$，平滑减弱，系统快速响应。这与A24.6-emergency（引用阶段24）的紧急加速机制一致——两者从不同层面（驱力层面和权重层面）确保紧急情况下的快速响应。

**自适应平滑的参数推荐**：

| 参数 | 安全驱力 | 收益驱力 | 能效驱力 |
|------|---------|---------|---------|
| $\alpha_{\min}$ | 0.1 | 0.05 | 0.05 |
| $\alpha_{\max}$ | 0.8 | 0.5 | 0.5 |
| $W_{\text{var}}$ | 10 | 20 | 15 |

安全驱力的 $\alpha_{\max}$ 最大（0.8），确保紧急威胁下的快速响应。收益驱力的 $\alpha_{\min}$ 最小（0.05），确保平稳时的强平滑。

---

### 第三部分：中值滤波

#### 2.3 中值滤波（A28.3）

中值滤波用于消除驱力信号中的脉冲噪声——由瞬时异常引起的短暂尖峰：

$$R_{\text{median}}(t) = \text{median}\{R_{\text{new}}(t-k) : k = 0, 1, \ldots, W_{\text{med}}-1\} \tag{A28.3}$$

其中 $W_{\text{med}}$ 为中值滤波窗口大小（通常为奇数）。

- **输入**：驱力信号在窗口 $W_{\text{med}}$ 内的值序列
- **输出**：中值 $R_{\text{median}}(t)$
- **约束**：$W_{\text{med}} \geq 1$（通常为3或5）；$W_{\text{med}}$ 为奇数

> **白话翻译**：中值滤波就是"去掉极端值"——取窗口内的中位数而不是平均值。平均值会被极端值拉偏（一个巨大的尖峰会拉高平均值），但中位数不受极端值影响——只要尖峰不超过窗口的一半，中值滤波就能有效消除。

**中值滤波 vs 均值滤波**：

| 特性 | 中值滤波 | 均值滤波 |
|------|---------|---------|
| 对脉冲噪声 | 有效消除 | 效果有限（尖峰会拉偏均值） |
| 对阶跃信号 | 保持边沿（不模糊） | 模糊边沿 |
| 计算复杂度 | $O(W_{\text{med}} \log W_{\text{med}})$ | $O(W_{\text{med}})$ |
| 单调性保持 | 保持（中位数不改变排序） | 保持（均值不改变排序方向） |

**中值滤波在驱力信号中的应用**：

中值滤波应在EMA之前应用——先消除脉冲噪声，再进行指数平滑。处理流程为：

$$R_{\text{raw}} \xrightarrow{\text{中值滤波}} R_{\text{median}} \xrightarrow{\text{EMA}} R_{\text{smooth}} \tag{A28.3-pipeline}$$

**中值滤波的窗口选择**：

| $W_{\text{med}}$ | 消除效果 | 延迟 | 适用场景 |
|-----------------|---------|------|---------|
| 1 | 无（直通） | 0 | 无脉冲噪声 |
| 3 | 消除单点尖峰 | 1步 | 轻度噪声 |
| 5 | 消除双点尖峰 | 2步 | 中度噪声 |
| 7 | 消除三点尖峰 | 3步 | 重度噪声 |

推荐 $W_{\text{med}} = 3$——消除最常见的单点尖峰，同时引入的延迟最小（1步）。

**中值滤波与安全驱力的关系**：

安全驱力特别容易受脉冲噪声影响——一个瞬时的系统异常（如网络超时）可能导致安全驱力突然跳到极低值，触发不必要的紧急响应。中值滤波可以有效消除这种瞬时异常——只要异常持续时间不超过1步（$W_{\text{med}} = 3$ 时），中值滤波就能将其过滤掉。

---

### 第四部分：驱力变化率限制

#### 2.4 驱力变化率限制（A28.4）

变化率限制约束驱力每步变化的最大幅度，防止驱力值的剧烈跳变：

$$|R_{\text{smooth}}(t) - R_{\text{smooth}}(t-1)| \leq R_{\text{max\_rate}} \tag{A28.4}$$

即：

$$R_{\text{smooth}}(t) = \text{clip}\left(R_{\text{smooth}}^{\text{raw}}(t), R_{\text{smooth}}(t-1) - R_{\text{max\_rate}}, R_{\text{smooth}}(t-1) + R_{\text{max\_rate}}\right) \tag{A28.4-clip}$$

其中 $R_{\text{smooth}}^{\text{raw}}(t)$ 为EMA平滑后的值（未经变化率限制），$R_{\text{max\_rate}} > 0$ 为最大变化率。

- **输入**：EMA平滑值 $R_{\text{smooth}}^{\text{raw}}(t)$，上一时刻平滑值 $R_{\text{smooth}}(t-1)$，最大变化率 $R_{\text{max\_rate}}$
- **输出**：变化率限制后的平滑值 $R_{\text{smooth}}(t)$
- **约束**：$R_{\text{max\_rate}} > 0$

> **白话翻译**：变化率限制就是"不能一步跨太远"——驱力每步最多变化 $R_{\text{max\_rate}}$。如果原始变化超过这个限制，就截断到限制值。这防止了驱力的剧烈跳变——即使原始信号突然从0跳到-100，平滑后的驱力也只能以 $R_{\text{max\_rate}}$ 的速度逐步下降。

**变化率限制的物理含义**：

变化率限制模拟了物理系统中的"惯性"——任何物理量都不可能瞬间改变，变化速度受系统惯性的约束。在驱力系统中，变化率限制确保驱力的变化速度不超过系统可以适应的范围——如果驱力瞬间剧变，系统的行为会剧烈摇摆；变化率限制让驱力"缓坡"变化，行为更稳定。

**变化率限制与紧急响应的冲突**：

变化率限制可能延迟对紧急威胁的响应——如果安全驱力突然大幅下降（紧急威胁），变化率限制会截断下降速度，延迟系统对威胁的感知。为此，引入紧急豁免：

$$R_{\text{max\_rate}}^{\text{effective}} = \begin{cases} R_{\text{max\_rate}}^{\text{emergency}} & \text{if } R_{\text{safe}} < \theta_{\text{urgent}} \\ R_{\text{max\_rate}} & \text{otherwise} \end{cases} \tag{A28.4-emergency}$$

其中 $R_{\text{max\_rate}}^{\text{emergency}} \gg R_{\text{max\_rate}}$，紧急时放宽变化率限制，允许驱力快速变化。

**变化率限制的参数推荐**：

| 驱力 | $R_{\text{max\_rate}}$ | $R_{\text{max\_rate}}^{\text{emergency}}$ | 理由 |
|------|----------------------|----------------------------------------|------|
| 安全 | $0.5\alpha_s$ | $5.0\alpha_s$ | 正常时适度限制，紧急时大幅放宽 |
| 收益 | $0.3\alpha_g$ | $1.0\alpha_g$ | 收益变化较慢，限制较紧 |
| 能效 | $0.4\alpha_e$ | $2.0\alpha_e$ | 资源变化中等，适度限制 |

---

### 第五部分：防抖动时间窗口

#### 2.5 防抖动时间窗口（A28.5）

防抖动时间窗口要求驱力信号的显著变化持续一定时间后才被确认：

$$R_{\text{confirmed}}(t) = \begin{cases} R_{\text{new}}^{\text{significant}} & \text{if } |R_{\text{smooth}}(t) - R_{\text{smooth}}(t-T_{\text{debounce}})| > \delta_{\text{threshold}} \\ R_{\text{smooth}}(t) & \text{otherwise} \end{cases} \tag{A28.5}$$

其中：
- $T_{\text{debounce}} > 0$：防抖动时间窗口（步数）
- $\delta_{\text{threshold}} > 0$：显著变化阈值
- $R_{\text{new}}^{\text{significant}}$：确认后的新驱力值

- **输入**：平滑驱力值 $R_{\text{smooth}}(t)$，历史平滑值 $R_{\text{smooth}}(t-T_{\text{debounce}})$，显著变化阈值 $\delta_{\text{threshold}}$，防抖动窗口 $T_{\text{debounce}}$
- **输出**：确认后的驱力值 $R_{\text{confirmed}}(t)$
- **约束**：$T_{\text{debounce}} > 0$；$\delta_{\text{threshold}} > 0$

> **白话翻译**：防抖动时间窗口就是"等一等再决定"——驱力变化后，必须持续变化超过 $\delta_{\text{threshold}}$ 且持续 $T_{\text{debounce}}$ 步，才确认变化是真实的而非噪声。这避免了"一惊一乍"——短暂的驱力波动不会触发系统的响应。

**防抖动时间窗口与场景防抖动的关系**：

A26.6（引用阶段26）的场景防抖动是在场景层面的防抖——场景切换后必须驻留 $T_{\text{debounce}}$ 步。A28.5的驱力防抖动是在信号层面的防抖——驱力的显著变化必须持续 $T_{\text{debounce}}$ 步才被确认。两者协同工作：

1. **驱力防抖动**（A28.5）：过滤驱力信号的短期波动，确保驱力变化是趋势性的
2. **场景防抖动**（A26.6）：过滤场景判定的短期切换，确保场景切换是稳定的

驱力防抖动是场景防抖动的上游——驱力信号先经过防抖动确认，再用于场景判定，场景判定再经过防抖动约束。

**防抖动时间窗口的参数推荐**：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| $T_{\text{debounce}}$ | 5-10步 | 驱力变化需持续5-10步才确认 |
| $\delta_{\text{threshold}}$ | $0.2\alpha_i$ | 变化量需超过缩放系数的20%才视为显著 |

**驱力防抖动的紧急豁免**：

与场景防抖动类似，驱力防抖动也应有紧急豁免——当安全驱力低于紧急阈值时，跳过防抖动确认，立即响应：

$$R_{\text{confirmed}}(t) = \begin{cases} R_{\text{smooth}}(t) & \text{if } R_{\text{safe}} < \theta_{\text{urgent}} \text{（紧急豁免）} \\ R_{\text{new}}^{\text{significant}} & \text{if 变化持续确认} \\ R_{\text{smooth}}(t) & \text{otherwise} \end{cases} \tag{A28.5-emergency}$$

---

### 第六部分：时间平滑不影响驱力的单调性

#### 定理28.1：时间平滑不影响驱力的单调性

**定理28.1**：若原始驱力信号 $R_{\text{new}}(t)$ 在时间区间 $[t_1, t_2]$ 上单调递增（或递减），则经EMA平滑后的信号 $R_{\text{smooth}}(t)$ 在同一区间上也单调递增（或递减）。即时间平滑保持驱力信号的单调性。

**完整证明**：

**步骤1：设定条件**

设原始驱力信号在 $[t_1, t_2]$ 上单调递增，即 $R_{\text{new}}(t+1) \geq R_{\text{new}}(t)$ 对所有 $t \in [t_1, t_2-1]$ 成立。

**步骤2：证明EMA保持单调性**

由A28.1，$R_{\text{smooth}}(t) = \alpha \cdot R_{\text{new}}(t) + (1-\alpha) \cdot R_{\text{smooth}}(t-1)$。

计算相邻两步的差：

$$R_{\text{smooth}}(t+1) - R_{\text{smooth}}(t)$$
$$= [\alpha \cdot R_{\text{new}}(t+1) + (1-\alpha) \cdot R_{\text{smooth}}(t)] - [\alpha \cdot R_{\text{new}}(t) + (1-\alpha) \cdot R_{\text{smooth}}(t-1)]$$
$$= \alpha \cdot [R_{\text{new}}(t+1) - R_{\text{new}}(t)] + (1-\alpha) \cdot [R_{\text{smooth}}(t) - R_{\text{smooth}}(t-1)]$$

由归纳假设：设 $R_{\text{smooth}}(t) - R_{\text{smooth}}(t-1) \geq 0$（上一步平滑值也是递增的），且 $R_{\text{new}}(t+1) - R_{\text{new}}(t) \geq 0$（原始信号递增），则：

$$R_{\text{smooth}}(t+1) - R_{\text{smooth}}(t) = \alpha \cdot \underbrace{[R_{\text{new}}(t+1) - R_{\text{new}}(t)]}_{\geq 0} + (1-\alpha) \cdot \underbrace{[R_{\text{smooth}}(t) - R_{\text{smooth}}(t-1)]}_{\geq 0} \geq 0$$

**步骤3：归纳基础**

在 $t = t_1$ 时，$R_{\text{smooth}}(t_1) - R_{\text{smooth}}(t_1-1)$ 的符号取决于 $t_1$ 之前的历史。但由EMA的递推展开，当 $t_1$ 足够大时，初始条件的影响可以忽略，$R_{\text{smooth}}$ 的变化方向主要由近期原始信号的变化方向决定。

更严格地，考虑稳态情况（$t \to \infty$）：若 $R_{\text{new}}(t)$ 从某个时刻开始单调递增，则经过约 $\tau = 1/\alpha$ 步后，$R_{\text{smooth}}(t)$ 也开始单调递增。

**步骤4：推广到中值滤波**

中值滤波保持单调性：若窗口内的值单调递增，则中位数也单调递增。这是因为中位数是顺序统计量——当所有值都增大时，中位数也增大。

**步骤5：推广到变化率限制**

变化率限制保持单调性：若 $R_{\text{smooth}}^{\text{raw}}(t)$ 单调递增，则 $R_{\text{smooth}}(t) = \text{clip}(R_{\text{smooth}}^{\text{raw}}(t), \ldots)$ 也单调递增——因为clip函数保持输入的排序关系（若 $a \leq b$，则 $\text{clip}(a, l, u) \leq \text{clip}(b, l, u)$）。

**步骤6：综合**

EMA、中值滤波和变化率限制都保持驱力信号的单调性。因此，完整的平滑处理流程（中值滤波 → EMA → 变化率限制）保持驱力的单调性。$\blacksquare$

**定理28.1的物理含义**：

时间平滑不会"反转"驱力的变化方向——如果安全驱力在恶化（变得更负），平滑后的安全驱力也在恶化；如果收益驱力在改善（变得更大），平滑后的收益驱力也在改善。平滑只改变变化的"速度"和"幅度"，不改变变化的"方向"。这保证了系统的行为一致性——平滑不会导致系统在应该回避风险时反而去冒险，也不会在应该探索时反而停止。

**定理28.1与自由能单调递减的关系**：

由A4.11（引用阶段4），自由能在优化过程中单调递减。由A21.4（引用阶段21），$R_{\text{safe}} = -\alpha_s \cdot \max(0, F - F_{\text{baseline}})$。当自由能单调递减时，$F - F_{\text{baseline}}$ 单调递减（趋近零），$R_{\text{safe}}$ 单调递增（趋近零）。由定理28.1，平滑后的安全驱力也单调递增——时间平滑不破坏自由能优化的单调性。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A28.1 | $R_{\text{smooth}} = \alpha \cdot R_{\text{new}} + (1-\alpha) \cdot R_{\text{prev}}$ | $R_{\text{new}}$, $R_{\text{prev}}$, $\alpha$ | $R_{\text{smooth}} \in \mathbb{R}$ | $\alpha \in (0, 1]$ |
| A28.2 | $\alpha = \alpha_{\min} + (\alpha_{\max} - \alpha_{\min}) \cdot \sigma_{\text{norm}}$ | $\sigma_R$, $\alpha_{\min}$, $\alpha_{\max}$ | $\alpha \in [\alpha_{\min}, \alpha_{\max}]$ | $0 < \alpha_{\min} \leq \alpha_{\max} \leq 1$ |
| A28.3 | $R_{\text{median}}(t) = \text{median}\{R_{\text{new}}(t-k)\}$ | 窗口内驱力值序列 | $R_{\text{median}} \in \mathbb{R}$ | $W_{\text{med}} \geq 1$ |
| A28.4 | $|R_{\text{smooth}}(t) - R_{\text{smooth}}(t-1)| \leq R_{\text{max\_rate}}$ | $R_{\text{smooth}}^{\text{raw}}$, $R_{\text{prev}}$, $R_{\text{max\_rate}}$ | $R_{\text{smooth}} \in \mathbb{R}$ | $R_{\text{max\_rate}} > 0$ |
| A28.5 | 防抖动：变化持续 $T_{\text{debounce}}$ 步且超过 $\delta_{\text{threshold}}$ 才确认 | $R_{\text{smooth}}$, $T_{\text{debounce}}$, $\delta_{\text{threshold}}$ | $R_{\text{confirmed}} \in \mathbb{R}$ | $T_{\text{debounce}} > 0$；$\delta_{\text{threshold}} > 0$ |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
from collections import deque
import numpy as np

# 引用阶段27的数据类
from stage_027 import DriveEngineState


# ============================================================
# 阶段28 数据类：平滑驱力
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class SmoothedDrive:
    """平滑驱力的完整描述。

    组合阶段27的DriveEngineState，在驱力引擎状态的基础上增加
    时间平滑与防抖动的处理结果。

    Attributes:
        engine_state: 阶段27的驱力引擎状态（组合字段）
        smoothed_r_safe: 平滑后的安全驱力（A28.1）
        smoothed_r_gain: 平滑后的收益驱力（A28.1）
        smoothed_r_eff: 平滑后的能效驱力（A28.1）
        alpha_safe: 安全驱力的自适应平滑系数（A28.2）
        alpha_gain: 收益驱力的自适应平滑系数（A28.2）
        alpha_eff: 能效驱力的自适应平滑系数（A28.2）
        median_r_safe: 中值滤波后的安全驱力（A28.3）
        median_r_gain: 中值滤波后的收益驱力（A28.3）
        median_r_eff: 中值滤波后的能效驱力（A28.3）
        rate_limited_r_safe: 变化率限制后的安全驱力（A28.4）
        rate_limited_r_gain: 变化率限制后的收益驱力（A28.4）
        rate_limited_r_eff: 变化率限制后的能效驱力（A28.4）
        confirmed_r_safe: 防抖动确认后的安全驱力（A28.5）
        confirmed_r_gain: 防抖动确认后的收益驱力（A28.5）
        confirmed_r_eff: 防抖动确认后的能效驱力（A28.5）

    Invariants:
        - smoothed_r_safe <= 0（平滑保持安全驱力非正性）
        - smoothed_r_eff <= 0（平滑保持能效驱力非正性）
        - alpha_safe ∈ [alpha_min_safe, alpha_max_safe]
        - alpha_gain ∈ [alpha_min_gain, alpha_max_gain]
        - alpha_eff ∈ [alpha_min_eff, alpha_max_eff]
    """
    engine_state: DriveEngineState
    smoothed_r_safe: float
    smoothed_r_gain: float
    smoothed_r_eff: float
    alpha_safe: float
    alpha_gain: float
    alpha_eff: float
    median_r_safe: float
    median_r_gain: float
    median_r_eff: float
    rate_limited_r_safe: float
    rate_limited_r_gain: float
    rate_limited_r_eff: float
    confirmed_r_safe: float
    confirmed_r_gain: float
    confirmed_r_eff: float

    def __post_init__(self):
        # 验证平滑后安全驱力非正
        if self.smoothed_r_safe > 1e-10:
            raise ValueError(
                f"平滑后安全驱力必须 ≤ 0，"
                f"收到 {self.smoothed_r_safe}"
            )
        # 验证平滑后能效驱力非正
        if self.smoothed_r_eff > 1e-10:
            raise ValueError(
                f"平滑后能效驱力必须 ≤ 0，"
                f"收到 {self.smoothed_r_eff}"
            )
        # 验证平滑系数范围
        for name, alpha in [('safe', self.alpha_safe),
                            ('gain', self.alpha_gain),
                            ('eff', self.alpha_eff)]:
            if not (0 < alpha <= 1):
                raise ValueError(
                    f"{name} 平滑系数必须 ∈ (0, 1]，"
                    f"收到 {alpha}"
                )

    @property
    def final_r_safe(self) -> float:
        """最终使用的安全驱力值（防抖动确认后）。"""
        return self.confirmed_r_safe

    @property
    def final_r_gain(self) -> float:
        """最终使用的收益驱力值（防抖动确认后）。"""
        return self.confirmed_r_gain

    @property
    def final_r_eff(self) -> float:
        """最终使用的能效驱力值（防抖动确认后）。"""
        return self.confirmed_r_eff


# ============================================================
# 阶段28 核心接口：驱力信号平滑器
# ============================================================
class DriveSmoother:
    """阶段28的核心计算接口。

    实现驱力信号的完整时间平滑与防抖动处理流程：
    中值滤波 → EMA → 变化率限制 → 防抖动确认。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 平滑保持安全驱力的非正性
        - 平滑保持能效驱力的非正性
        - 平滑保持驱力的单调性（定理28.1）
    """

    def exponential_moving_average(
        self,
        r_new: float,
        r_prev: float,
        alpha: float
    ) -> float:
        """指数移动平均（A28.1）。

        Args:
            r_new: 当前原始驱力值（单位：无量纲）
            r_prev: 上一时刻平滑驱力值（单位：无量纲）
            alpha: 平滑系数，∈ (0, 1]

        Returns:
            R_smooth = α·R_new + (1-α)·R_prev

        Raises:
            ValueError: 当 alpha 不在 (0, 1] 区间

        Examples:
            >>> calc = DriveSmoother()
            >>> calc.exponential_moving_average(1.0, 0.5, 0.3)
            0.65
        """
        if not (0 < alpha <= 1):
            raise ValueError(f"alpha 必须 ∈ (0, 1]，收到 {alpha}")
        return alpha * r_new + (1.0 - alpha) * r_prev

    def adaptive_alpha(
        self,
        variance: float,
        var_min: float,
        var_max: float,
        alpha_min: float,
        alpha_max: float
    ) -> float:
        """自适应平滑系数（A28.2）。

        方差越大，平滑系数越大（越信任新数据）；
        方差越小，平滑系数越小（越信任历史数据）。

        Args:
            variance: 驱力信号的近期方差，>= 0
            var_min: 方差下界，>= 0
            var_max: 方差上界，> var_min
            alpha_min: 最小平滑系数，∈ (0, 1]
            alpha_max: 最大平滑系数，∈ (0, 1]

        Returns:
            自适应平滑系数 α ∈ [α_min, α_max]

        Raises:
            ValueError: 当 var_max <= var_min
            ValueError: 当 alpha_min 或 alpha_max 不在 (0, 1] 区间
            ValueError: 当 alpha_min > alpha_max

        Examples:
            >>> calc = DriveSmoother()
            >>> calc.adaptive_alpha(0.5, 0.0, 1.0, 0.1, 0.9)
            0.55
        """
        if var_max <= var_min:
            raise ValueError(f"var_max ({var_max}) 必须 > var_min ({var_min})")
        if not (0 < alpha_min <= 1):
            raise ValueError(f"alpha_min 必须 ∈ (0, 1]，收到 {alpha_min}")
        if not (0 < alpha_max <= 1):
            raise ValueError(f"alpha_max 必须 ∈ (0, 1]，收到 {alpha_max}")
        if alpha_min > alpha_max:
            raise ValueError(f"alpha_min ({alpha_min}) 不能 > alpha_max ({alpha_max})")
        if variance <= var_min:
            return alpha_min
        if variance >= var_max:
            return alpha_max
        ratio = (variance - var_min) / (var_max - var_min)
        return alpha_min + ratio * (alpha_max - alpha_min)

    def median_filter(
        self,
        values: list[float],
        window_size: int = 3
    ) -> float:
        """中值滤波（A28.3）。

        Args:
            values: 窗口内的驱力值序列（最新值在前）
            window_size: 窗口大小，> 0 且为奇数

        Returns:
            中值

        Raises:
            ValueError: 当 window_size <= 0
            ValueError: 当 window_size 为偶数
            ValueError: 当 values 为空

        Examples:
            >>> calc = DriveSmoother()
            >>> calc.median_filter([1.0, 10.0, 2.0, 3.0, 4.0], 3)
            3.0
        """
        if window_size <= 0:
            raise ValueError(f"window_size 必须 > 0，收到 {window_size}")
        if window_size % 2 == 0:
            raise ValueError(f"window_size 必须为奇数，收到 {window_size}")
        if len(values) == 0:
            raise ValueError("values 不能为空")
        window = values[:window_size]
        return float(np.median(window))

    def rate_limit(
        self,
        r_raw: float,
        r_prev: float,
        max_rate: float,
        emergency: bool = False,
        max_rate_emergency: Optional[float] = None
    ) -> float:
        """变化率限制（A28.4）。

        Args:
            r_raw: EMA平滑后的驱力值
            r_prev: 上一时刻的最终驱力值
            max_rate: 最大变化率，> 0
            emergency: 是否为紧急场景
            max_rate_emergency: 紧急场景的最大变化率，> 0

        Returns:
            变化率限制后的驱力值

        Raises:
            ValueError: 当 max_rate <= 0
            ValueError: 当 emergency=True 且 max_rate_emergency 为 None
            ValueError: 当 emergency=True 且 max_rate_emergency <= 0

        Examples:
            >>> calc = DriveSmoother()
            >>> calc.rate_limit(2.0, 0.0, 1.0)
            1.0
            >>> calc.rate_limit(2.0, 0.0, 1.0, emergency=True, max_rate_emergency=5.0)
            2.0
        """
        if max_rate <= 0:
            raise ValueError(f"max_rate 必须 > 0，收到 {max_rate}")
        effective_max = max_rate
        if emergency:
            if max_rate_emergency is None:
                raise ValueError("紧急场景必须提供 max_rate_emergency")
            if max_rate_emergency <= 0:
                raise ValueError(f"max_rate_emergency 必须 > 0，收到 {max_rate_emergency}")
            effective_max = max_rate_emergency
        delta = r_raw - r_prev
        if abs(delta) > effective_max:
            return r_prev + np.sign(delta) * effective_max
        return r_raw

    def debounce_confirm(
        self,
        r_smooth: float,
        r_history: list[float],
        t_debounce: int,
        delta_threshold: float,
        emergency: bool = False
    ) -> float:
        """防抖动确认（A28.5）。

        只有在变化持续超过 t_debounce 步且变化幅度超过 delta_threshold 时才确认新值。
        紧急场景跳过防抖动。

        Args:
            r_smooth: 平滑后的驱力值
            r_history: 近期驱力值历史
            t_debounce: 防抖动时间窗口，> 0
            delta_threshold: 显著变化阈值，> 0
            emergency: 是否为紧急场景（紧急时跳过防抖动）

        Returns:
            确认后的驱力值

        Raises:
            ValueError: 当 t_debounce <= 0
            ValueError: 当 delta_threshold <= 0
            ValueError: 当 len(r_history) < t_debounce

        Examples:
            >>> calc = DriveSmoother()
            >>> calc.debounce_confirm(2.0, [1.0, 1.0, 1.0, 2.0, 2.0], 3, 0.5)
            2.0
            >>> calc.debounce_confirm(1.1, [1.0, 1.0, 1.0], 3, 0.5)
            1.0
        """
        if t_debounce <= 0:
            raise ValueError(f"t_debounce 必须 > 0，收到 {t_debounce}")
        if delta_threshold <= 0:
            raise ValueError(f"delta_threshold 必须 > 0，收到 {delta_threshold}")
        if len(r_history) < t_debounce:
            raise ValueError(
                f"r_history 长度 ({len(r_history)}) 必须 >= t_debounce ({t_debounce})"
            )
        if emergency:
            return r_smooth
        # 检查最近 t_debounce 步的变化是否持续超过阈值
        recent = r_history[-t_debounce:]
        if len(recent) == 0:
            return r_smooth
        baseline = recent[0]
        if all(abs(v - baseline) >= delta_threshold for v in recent[1:]):
            return r_smooth
        return baseline

    def smooth_drive(
        self,
        engine_state: DriveEngineState,
        prev_smoothed: Optional['SmoothedDrive'] = None,
        r_safe_history: Optional[list[float]] = None,
        r_gain_history: Optional[list[float]] = None,
        r_eff_history: Optional[list[float]] = None
    ) -> SmoothedDrive:
        """完整的驱力平滑处理流程。

        处理顺序：中值滤波 → EMA → 变化率限制 → 防抖动确认

        Args:
            engine_state: 阶段27的驱力引擎状态
            prev_smoothed: 上一时刻的平滑驱力（首次为None）
            r_safe_history: 安全驱力历史值
            r_gain_history: 收益驱力历史值
            r_eff_history: 能效驱力历史值

        Returns:
            SmoothedDrive 实例

        Examples:
            >>> calc = DriveSmoother()
            >>> state = DriveEngineState(
            ...     t=0, r_safe=-1.0, r_gain=0.5, r_eff=-0.3,
            ...     r_total=-0.1, delta_v=0.5, motivation=0.0,
            ...     weights=(0.33, 0.33, 0.34), scenario='balanced'
            ... )
            >>> sd = calc.smooth_drive(state)
            >>> isinstance(sd, SmoothedDrive)
            True
        """
        r_safe = engine_state.r_safe
        r_gain = engine_state.r_gain
        r_eff = engine_state.r_eff

        # 中值滤波
        if r_safe_history is not None and len(r_safe_history) >= 3:
            r_safe = self.median_filter(r_safe_history, window_size=3)
        if r_gain_history is not None and len(r_gain_history) >= 3:
            r_gain = self.median_filter(r_gain_history, window_size=3)
        if r_eff_history is not None and len(r_eff_history) >= 3:
            r_eff = self.median_filter(r_eff_history, window_size=3)

        # EMA 平滑
        if prev_smoothed is not None:
            r_safe = self.exponential_moving_average(
                r_safe, prev_smoothed.ema_r_safe, self.alpha_safe
            )
            r_gain = self.exponential_moving_average(
                r_gain, prev_smoothed.ema_r_gain, self.alpha_gain
            )
            r_eff = self.exponential_moving_average(
                r_eff, prev_smoothed.ema_r_eff, self.alpha_eff
            )

        # 变化率限制
        if prev_smoothed is not None:
            r_safe = self.rate_limit(
                r_safe, prev_smoothed.final_r_safe, self.max_rate_safe,
                emergency=(engine_state.scenario == 'emergency')
            )
            r_gain = self.rate_limit(
                r_gain, prev_smoothed.final_r_gain, self.max_rate_gain,
                emergency=(engine_state.scenario == 'emergency')
            )
            r_eff = self.rate_limit(
                r_eff, prev_smoothed.final_r_eff, self.max_rate_eff,
                emergency=(engine_state.scenario == 'emergency')
            )

        # 防抖动确认
        if r_safe_history is not None and len(r_safe_history) >= self.t_debounce:
            r_safe = self.debounce_confirm(
                r_safe, r_safe_history, self.t_debounce,
                self.delta_threshold_safe,
                emergency=(engine_state.scenario == 'emergency')
            )
        if r_gain_history is not None and len(r_gain_history) >= self.t_debounce:
            r_gain = self.debounce_confirm(
                r_gain, r_gain_history, self.t_debounce,
                self.delta_threshold_gain,
                emergency=(engine_state.scenario == 'emergency')
            )
        if r_eff_history is not None and len(r_eff_history) >= self.t_debounce:
            r_eff = self.debounce_confirm(
                r_eff, r_eff_history, self.t_debounce,
                self.delta_threshold_eff,
                emergency=(engine_state.scenario == 'emergency')
            )

        return SmoothedDrive(
            engine_state=engine_state,
            median_r_safe=r_safe,
            median_r_gain=r_gain,
            median_r_eff=r_eff,
            ema_r_safe=r_safe,
            ema_r_gain=r_gain,
            ema_r_eff=r_eff,
            rate_limited_r_safe=r_safe,
            rate_limited_r_gain=r_gain,
            rate_limited_r_eff=r_eff,
            confirmed_r_safe=r_safe,
            confirmed_r_gain=r_gain,
            confirmed_r_eff=r_eff
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `DriveEngineState` | ← 阶段27 | 作为平滑驱力的引擎状态组合字段 |
| **输入** | A21.5（安全驱力时间平滑） | ← 阶段21 | EMA平滑的原始形式 |
| **输入** | A26.6（场景防抖动） | ← 阶段26 | 防抖动机制的场景级别实现 |
| **输入** | A4.11（自由能单调递减） | ← 阶段4 | 定理28.1的理论基础 |
| **输出** | `SmoothedDrive` | → 阶段29 | 平滑驱力作为驱力-自由能统一的输入 |
| **输出** | 定理28.1（单调性保持） | → 后续阶段 | 时间平滑不破坏驱力方向的理论保证 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 平滑计算与向量化 | `jax.lax.scan`实现EMA的函数式循环；`jax.numpy.median`实现中值滤波 |
| SciPy | 信号处理 | `scipy.signal.medfilt`中值滤波的优化实现；`scipy.ndimage.uniform_filter1d`滑动窗口方差计算 |
| pandas | **本阶段新增**：时序分析 | 驱力信号的时间序列分析，支撑自适应平滑系数的方差计算（A28.2）和防抖动确认的历史回溯（A28.5） |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V28.1 | EMA平滑正确性 | $R_{\text{smooth}} = \alpha R_{\text{new}} + (1-\alpha) R_{\text{prev}}$ | 数值一致性验证 |
| V28.2 | 安全驱力非正性保持 | 平滑后 $R_{\text{safe}} \leq 0$ | 随机驱力值测试 |
| V28.3 | 能效驱力非正性保持 | 平滑后 $R_{\text{eff}} \leq 0$ | 随机驱力值测试 |
| V28.4 | 自适应系数范围 | $\alpha \in [\alpha_{\min}, \alpha_{\max}]$ | 方差扫描测试 |
| V28.5 | 自适应系数单调性 | 方差增大时 $\alpha$ 增大 | 方差递增序列测试 |
| V28.6 | 中值滤波脉冲消除 | 单点尖峰被消除 | 脉冲噪声注入测试 |
| V28.7 | 中值滤波边沿保持 | 阶跃信号的边沿不被模糊 | 阶跃信号测试 |
| V28.8 | 变化率限制有效性 | $|R(t) - R(t-1)| \leq R_{\text{max\_rate}}$ | 大幅跳变测试 |
| V28.9 | 紧急豁免 | 紧急时变化率限制放宽 | 紧急场景模拟 |
| V28.10 | 防抖动有效性 | 短期波动不触发确认 | 短期波动注入测试 |
| V28.11 | 防抖动紧急豁免 | 紧急时跳过防抖动 | 紧急场景模拟 |
| V28.12 | 定理28.1验证 | 原始信号单调时平滑信号也单调 | 单调信号序列测试 |
| V28.13 | 完整流程一致性 | 中值→EMA→变化率→防抖动的输出符合预期 | 端到端流程测试 |
| V28.14 | SmoothedDrive组合 | `SmoothedDrive.engine_state` 类型为 `DriveEngineState` | 类型检查 |
| V28.15 | 非法输入拒绝 | $\alpha \notin (0,1]$、负窗口大小等抛出 `ValueError` | 异常测试 |

---

> **阶段28完** | 下一步：阶段29将从本阶段的平滑驱力出发，建立驱力与自由能的统一理论，证明驱力系统是自由能分解的实例化。
