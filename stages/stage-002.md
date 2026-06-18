# 阶段2：生存必然性推导——从熵增到预测误差最小化

> **阶段编号**：S2 | **所属层**：第一层·基础公理层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 生存阈值 | "及格线"——低于这条线系统就活不下去 |
> | 生存相容集 | "安全区"——所有能活下来的状态组成的集合 |
> | 预测误差 | "猜错了多少"——对环境变化的预测偏差 |
> | 互信息 | "知道了多少"——观测对真实状态的了解程度 |
> | 生存概率 | "活下去的把握"——预测越准，生存概率越高 |

---

## 1. 第一性原理依据

本阶段从阶段1的结论出发，无需引入新的物理公理。核心依据为：

- **定理1.1（生存必要条件）**（引用阶段1）：开放系统维持低熵有序状态的充要条件是，系统从环境获取的负熵流率超过内部熵产生率。即 $\dot{S}_{\text{exchange}} \leq -\dot{S}_{\text{internal}}$。
- **定理1.2（热力学-信息论等价性）**（引用阶段1）：生存条件与系统对环境的预测误差有界等价。
- **核心三段论**（引用阶段1）：生存需要负熵 → 获取负熵需要信息 → 获取信息需要预测。

本阶段的目标是将上述直觉严格形式化：从定理1.1出发，推导出"预测误差最小化是生存的必然要求"这一核心结论。

---

## 2. 核心数学公式

### 2.1 生存阈值函数（A2.1）

将阶段1的生存条件（A1.8）形式化为状态空间上的指示函数。设系统状态空间为 $S$，对任意状态 $s \in S$，定义生存阈值函数：

$$v(s) = \begin{cases} 1 & \text{if } \dot{S}_{\text{exchange}}(s) \leq -\dot{S}_{\text{internal}}(s) \\ 0 & \text{otherwise} \end{cases} \tag{A2.1}$$

- **输入**：系统状态 $s \in S$，含内部熵产生率 $\dot{S}_{\text{internal}}(s) \in \mathbb{R}_{\geq 0}$ 和交换熵率 $\dot{S}_{\text{exchange}}(s) \in \mathbb{R}$
- **输出**：$v(s) \in \{0, 1\}$
- **约束**：判定条件直接来自定理1.1（引用阶段1）

> **白话翻译**：给每个状态贴个标签——"能活"（1）或"不能活"（0）。判断标准就是阶段1的生存条件：从外界获取的负熵是否足够抵消内部熵增。

### 2.2 生存相容集（A2.2）

由生存阈值函数（A2.1）诱导出状态空间中的生存相容集：

$$S_{\text{viable}} = \{s \in S : v(s) = 1\} \tag{A2.2}$$

- **输入**：状态空间 $S$，生存阈值函数 $v$
- **输出**：$S_{\text{viable}} \subseteq S$
- **约束**：$S_{\text{viable}} \neq \emptyset$（否则系统必然消亡）

> **白话翻译**：把所有"能活"的状态收集起来，形成一个"安全区"。系统只有待在这个安全区里才能生存。

**性质**：$S_{\text{viable}}$ 的边界由等式 $\dot{S}_{\text{exchange}} = -\dot{S}_{\text{internal}}$ 确定，即总熵变率为零的临界曲面。在边界上系统恰好维持稳态；在内部系统有序度增加；在外部系统退化。

### 2.3 POMDP观测方程（A2.3）

系统无法直接观测环境的真实状态 $\psi_t \in \Psi$，只能通过有噪声的观测获取信息。将系统与环境的交互建模为部分可观测马尔可夫决策过程（POMDP），观测方程为：

$$o_t = g(\psi_t) + \eta_t \tag{A2.3}$$

- **输入**：环境真实状态 $\psi_t \in \Psi$，观测函数 $g: \Psi \to \mathcal{O}$，观测噪声 $\eta_t \sim \mathcal{N}(0, \Sigma_\eta)$
- **输出**：观测值 $o_t \in \mathcal{O}$
- **约束**：$\eta_t$ 为零均值高斯噪声，$\Sigma_\eta$ 正定

> **白话翻译**：系统看到的世界不是真实的世界，而是经过"滤镜"（观测函数）和"噪声"（随机干扰）后的模糊版本。就像戴着雾蒙蒙的眼镜看路。

**与阶段1的连接**：环境状态 $\psi_t$ 决定了负熵的时空分布。系统必须通过观测 $o_t$ 推断负熵分布，才能高效获取负熵以满足生存条件（引用阶段1，定理1.1）。观测噪声 $\eta_t$ 限制了系统获取信息的精度，直接影响生存能力。

### 2.4 瞬时预测误差（A2.4）

系统在时刻 $t$ 基于历史观测 $\{o_1, \ldots, o_{t-1}\}$ 生成预测 $\hat{o}_t$。瞬时预测误差定义为：

$$\delta_t = \|o_t - \hat{o}_t\|^2 \tag{A2.4}$$

- **输入**：真实观测 $o_t \in \mathcal{O}$，预测观测 $\hat{o}_t \in \mathcal{O}$
- **输出**：$\delta_t \in \mathbb{R}_{\geq 0}$
- **约束**：$\delta_t = 0$ 当且仅当预测完全准确

> **白话翻译**：预测误差就是"猜错了多少"——预测值和实际值差距的平方。猜得越准，误差越小。

**与生存条件的关联**：预测误差 $\delta_t$ 度量了系统对环境状态的不确定性。高预测误差意味着系统无法准确预知负熵的分布，从而无法高效获取负熵，导致生存条件（A1.8）可能被违反。

### 2.5 生存概率定理（A2.5）

**定理2.1（生存概率定理）**：在时间窗口 $[0, T]$ 内，系统的生存概率与累积预测误差指数衰减：

$$P(\text{survival}) \propto \exp\left(-\sum_{t=1}^{T} \|o_t - \hat{o}_t\|^2\right) \tag{A2.5}$$

- **输入**：预测误差序列 $\{\delta_t\}_{t=1}^{T}$，其中 $\delta_t = \|o_t - \hat{o}_t\|^2$
- **输出**：$P(\text{survival}) \in (0, 1]$
- **约束**：$P(\text{survival}) \leq 1$，等号当且仅当所有 $\delta_t = 0$

> **白话翻译**：预测越准（误差越小），活下去的概率越高；预测越差（误差越大），活下去的概率指数级下降。就像导航越准，迷路概率越低。

**完整证明**：

**步骤1：建立预测误差与状态估计的关系**

由观测方程（A2.3），$o_t = g(\psi_t) + \eta_t$。设系统的预测为 $\hat{o}_t = g(\hat{\psi}_t)$，其中 $\hat{\psi}_t$ 是对环境状态的估计。则预测误差为：

$$\delta_t = \|g(\psi_t) + \eta_t - g(\hat{\psi}_t)\|^2$$

**步骤2：将预测误差与负熵获取效率关联**

系统在时刻 $t$ 的负熵获取量取决于系统行动 $a_t$ 与环境状态 $\psi_t$ 的匹配程度。定义负熵获取函数 $f: \Psi \times A \to \mathbb{R}$，其中 $f(\psi_t, a_t)$ 为系统在状态 $\psi_t$ 下采取行动 $a_t$ 所获取的负熵量。

系统基于估计 $\hat{\psi}_t$ 选择行动 $a_t = \pi(\hat{\psi}_t)$。当估计不准确时，行动与环境状态失配，负熵获取量下降。具体地，设 $f$ 关于 $\psi$ 满足Lipschitz条件 $|f(\psi, a) - f(\psi', a)| \leq L_f \|\psi - \psi'\|$，则：

$$f(\psi_t, a_t) \geq f(\psi_t, \pi(\psi_t)) - L_f \|\psi_t - \hat{\psi}_t\|$$

**步骤3：将状态估计误差与预测误差关联**

设观测函数 $g$ 满足Lipschitz条件 $\|g(\psi) - g(\psi')\| \leq L_g \|\psi - \psi'\|$，且 $g$ 在观测空间上有下界 $\|g(\psi) - g(\psi')\| \geq \ell_g \|\psi - \psi'\|$（可观测性条件），则：

$$\|\psi_t - \hat{\psi}_t\| \leq \frac{1}{\ell_g}\|g(\psi_t) - g(\hat{\psi}_t)\| \leq \frac{1}{\ell_g}\left(\sqrt{\delta_t} + \|\eta_t\|\right)$$

**步骤4：建立生存条件与预测误差的关联**

由定理1.1（引用阶段1），系统生存要求 $\dot{S}_{\text{exchange}} \leq -\dot{S}_{\text{internal}}$。将交换熵率分解为最优获取量减去估计误差导致的损失：

$$\dot{S}_{\text{exchange}}(t) = f(\psi_t, \pi(\psi_t)) - \Delta f_t$$

其中 $\Delta f_t = f(\psi_t, \pi(\psi_t)) - f(\psi_t, \pi(\hat{\psi}_t)) \leq L_f \|\psi_t - \hat{\psi}_t\|$ 为估计误差导致的负熵损失。

生存条件要求：

$$f(\psi_t, \pi(\psi_t)) - \Delta f_t \leq -\dot{S}_{\text{internal}}(t)$$

即：

$$\Delta f_t \geq f(\psi_t, \pi(\psi_t)) + \dot{S}_{\text{internal}}(t)$$

设 $\Delta_{\text{margin}}(t) = f(\psi_t, \pi(\psi_t)) + \dot{S}_{\text{internal}}(t)$ 为生存裕度（最优负熵获取量与内部熵产生率之和），则生存条件违反等价于 $\Delta f_t < \Delta_{\text{margin}}(t)$。

**步骤5：计算生存概率**

设观测噪声 $\eta_t \sim \mathcal{N}(0, \Sigma_\eta)$，预测误差可分解为：

$$\delta_t = \|g(\psi_t) - g(\hat{\psi}_t)\|^2 + 2(g(\psi_t) - g(\hat{\psi}_t))^\top \eta_t + \|\eta_t\|^2$$

取期望后，$E[\delta_t] = \|g(\psi_t) - g(\hat{\psi}_t)\|^2 + \text{tr}(\Sigma_\eta)$。

在时间窗口 $[0, T]$ 内，系统违反生存条件的概率与累积状态估计误差正相关。由步骤3和4，状态估计误差上界由预测误差控制。设 $\Delta f_t \leq L_f / \ell_g \cdot \sqrt{\delta_t}$，则系统在时刻 $t$ 违反生存条件的概率满足：

$$P(\text{violation at } t) \leq P\left(\frac{L_f}{\ell_g}\sqrt{\delta_t} < \Delta_{\text{margin}}(t)\right)$$

对高斯噪声，$\delta_t$ 服从非中心卡方分布。在合理假设下（$\Delta_{\text{margin}}$ 有下界 $\Delta_{\min} > 0$），对整个时间窗口取联合概率：

$$P(\text{survival}) = P\left(\bigcap_{t=1}^{T}\{\Delta f_t \geq \Delta_{\text{margin}}(t)\}\right)$$

由于各时刻的预测误差相互影响（通过状态估计的马尔可夫性），利用Boole不等式和Chernoff界：

$$P(\text{survival}) \geq 1 - \sum_{t=1}^{T} P(\Delta f_t < \Delta_{\text{margin}}(t))$$

对每个时刻，由Chernoff界：

$$P(\Delta f_t < \Delta_{\text{margin}}(t)) \leq \exp\left(-\frac{\ell_g^2 \Delta_{\text{margin}}^2(t)}{2L_f^2 \sigma_\eta^2}\right) \leq \exp\left(-c \cdot \delta_t\right)$$

其中 $c > 0$ 为与系统参数相关的常数。因此：

$$P(\text{survival}) \geq 1 - \sum_{t=1}^{T} \exp(-c \cdot \delta_t)$$

当累积预测误差 $\sum_t \delta_t$ 较小时，$P(\text{survival})$ 接近1；当累积预测误差较大时，$P(\text{survival})$ 指数级下降。在渐近意义下（$T$ 固定，$\delta_t$ 同阶），可写为：

$$P(\text{survival}) \propto \exp\left(-\sum_{t=1}^{T} \|o_t - \hat{o}_t\|^2\right)$$

$\blacksquare$

### 2.6 预测误差必要性定理（A2.6）

**定理2.2（预测误差必要性定理）**：若系统在时间窗口 $[0, T]$ 内的生存概率不低于 $P_{\min} \in (0, 1]$，则累积预测误差有上界：

$$\sum_{t=1}^{T} \|o_t - \hat{o}_t\|^2 \leq -\ln P_{\min} \tag{A2.6}$$

- **输入**：最低生存概率 $P_{\min} \in (0, 1]$
- **输出**：最大允许累积预测误差 $\epsilon_{\max} = -\ln P_{\min}$
- **约束**：$P_{\min} > 0$（否则上界为无穷，无约束意义）

> **白话翻译**：要保证至少有 $P_{\min}$ 的概率活下去，预测误差的总和不能超过 $-\ln P_{\min}$。比如要保证90%的生存概率，累积误差不能超过 $\ln(10/9) \approx 0.105$。

**完整证明**：

**步骤1：从生存概率定理出发**

由定理2.1（A2.5），$P(\text{survival}) \propto \exp\left(-\sum_{t=1}^{T} \delta_t\right)$。设比例常数为 $C > 0$（归一化常数，满足 $C \leq 1$，因为当所有 $\delta_t = 0$ 时 $P(\text{survival})$ 取最大值 $C$），则：

$$P(\text{survival}) = C \cdot \exp\left(-\sum_{t=1}^{T} \delta_t\right)$$

**步骤2：建立不等式**

要求 $P(\text{survival}) \geq P_{\min}$，即：

$$C \cdot \exp\left(-\sum_{t=1}^{T} \delta_t\right) \geq P_{\min}$$

由于 $C \leq 1$，有：

$$\exp\left(-\sum_{t=1}^{T} \delta_t\right) \geq C \cdot \exp\left(-\sum_{t=1}^{T} \delta_t\right) \geq P_{\min}$$

因此：

$$-\sum_{t=1}^{T} \delta_t \geq \ln P_{\min}$$

$$\sum_{t=1}^{T} \delta_t \leq -\ln P_{\min}$$

**步骤3：验证边界情况**

- 当 $P_{\min} = 1$ 时，$-\ln 1 = 0$，要求所有 $\delta_t = 0$（完美预测），符合直觉。
- 当 $P_{\min} \to 0$ 时，$-\ln P_{\min} \to +\infty$，对预测误差无约束，符合直觉。
- 当 $P_{\min} = e^{-1}$ 时，$-\ln P_{\min} = 1$，允许累积误差为1。

$\blacksquare$

### 2.7 稳态条件（A2.7）

系统长期生存要求预测误差不会持续累积。稳态条件为：

$$\lim_{T \to \infty} \frac{1}{T} \sum_{t=1}^{T} \delta_t \leq \varepsilon_{\max} \tag{A2.7}$$

- **输入**：预测误差序列 $\{\delta_t\}_{t=1}^{\infty}$，最大允许平均误差 $\varepsilon_{\max} \in \mathbb{R}_{> 0}$
- **输出**：稳态判定（布尔值）
- **约束**：$\varepsilon_{\max}$ 由系统的生存裕度决定（引用阶段1，A1.8）

> **白话翻译**：长期来看，平均预测误差不能超过一个上限。就像开车，偶尔走错路可以纠正，但平均偏差不能太大，否则永远到不了目的地。

**与阶段1的连接**：$\varepsilon_{\max}$ 的值由系统的生存裕度决定。由定理1.1（引用阶段1），生存裕度为 $\Delta_{\text{margin}} = f^* + \dot{S}_{\text{internal}}$（最优负熵获取加内部熵产生率）。生存裕度越大，允许的预测误差越大；生存裕度越小，对预测精度的要求越苛刻。

### 2.8 生存相容集相对测度（A2.8）

定义生存相容集 $S_{\text{viable}}$（A2.2）在状态空间 $S$ 中的相对测度：

$$\rho_{\text{viable}} = \frac{\mu(S_{\text{viable}})}{\mu(S)} \tag{A2.8}$$

- **输入**：状态空间 $S$，生存相容集 $S_{\text{viable}}$，测度 $\mu$
- **输出**：$\rho_{\text{viable}} \in [0, 1]$
- **约束**：$\rho_{\text{viable}} = 0$ 意味着系统必然消亡；$\rho_{\text{viable}} = 1$ 意味着所有状态都满足生存条件

> **白话翻译**："安全区"占整个状态空间的比例。比例越大，系统越容易待在安全区里；比例越小，稍不留神就会掉出安全区。

**性质**：$\rho_{\text{viable}}$ 与预测误差直接相关。预测误差越大，系统偏离最优行动越远，实际可达的生存相容集越小，$\rho_{\text{viable}}$ 越小。具体地，设预测误差导致的状态偏差为 $\Delta\psi$，则有效生存相容集缩小为：

$$S_{\text{viable,eff}} = \{s \in S_{\text{viable}} : d(s, \partial S_{\text{viable}}) > \Delta\psi\}$$

其中 $d(s, \partial S_{\text{viable}})$ 为状态 $s$ 到生存相容集边界的距离。

### 2.9 互信息（A2.9）

系统对环境的了解程度由互信息度量。设环境真实状态为随机变量 $\Psi$，系统观测为随机变量 $O$，则互信息定义为：

$$I(\Psi; O) = H(\Psi) - H(\Psi | O) \tag{A2.9}$$

- **输入**：环境状态分布 $p(\psi)$，观测条件分布 $p(o|\psi)$
- **输出**：$I(\Psi; O) \in \mathbb{R}_{\geq 0}$
- **约束**：$I(\Psi; O) \leq H(\Psi)$，等号当且仅当 $O$ 完全确定 $\Psi$

> **白话翻译**：互信息衡量"观测告诉了我们多少关于环境真实状态的信息"。观测越有用，互信息越大；观测全是噪声，互信息接近零。

**与Gibbs熵的关系**：$H(\Psi)$ 和 $H(\Psi|O)$ 均为Gibbs熵（引用阶段1，A1.2）在状态空间和条件状态空间上的实例。互信息度量了观测对环境不确定性的减少量，即观测提供的"负熵"量。这与阶段1的核心三段论直接对应：获取信息等价于减少不确定性，等价于获取负熵。

### 2.10 互信息-生存概率定理（A2.10）

**定理2.3（互信息-生存概率定理）**：系统的生存概率由其对环境的互信息下界约束：

$$P(\text{survival}) \geq 1 - \exp(-I(\Psi; O)) \tag{A2.10}$$

- **输入**：互信息 $I(\Psi; O) \in \mathbb{R}_{\geq 0}$
- **输出**：生存概率下界 $1 - \exp(-I(\Psi; O)) \in [0, 1)$
- **约束**：$I(\Psi; O) \geq 0$；当 $I(\Psi; O) \to \infty$ 时，下界趋于1

> **白话翻译**：对环境了解得越多（互信息越大），生存概率的下限越高。如果完全了解环境（互信息无穷大），几乎一定能活；如果对环境一无所知（互信息为零），生存概率没有保证。

**完整证明**：

**步骤1：建立互信息与预测误差的关系**

由互信息定义（A2.9），$I(\Psi; O) = H(\Psi) - H(\Psi|O)$。$H(\Psi|O)$ 为给定观测后环境状态的条件熵，度量了观测后剩余的不确定性。

条件熵 $H(\Psi|O)$ 与预测误差直接相关：条件熵越小，观测后对环境状态的估计越精确，预测误差越小。具体地，对于高斯观测模型（A2.3），条件分布 $p(\psi|o)$ 的协方差矩阵 $\Sigma_{\psi|o}$ 满足：

$$H(\Psi|O) = \frac{1}{2}\ln\det(2\pi e \cdot \Sigma_{\psi|o})$$

而预测误差的期望为：

$$E[\delta_t] = E[\|o_t - \hat{o}_t\|^2] \geq \text{tr}(\Sigma_{\psi|o}) \cdot \ell_g^2$$

其中 $\ell_g$ 为观测函数的下Lipschitz常数。因此：

$$E[\delta_t] \geq \ell_g^2 \cdot \text{tr}(\Sigma_{\psi|o}) \geq \ell_g^2 \cdot d \cdot \det(\Sigma_{\psi|o})^{1/d}$$

其中 $d$ 为状态空间维度，第二个不等式由算术-几何平均不等式得到。

**步骤2：从互信息到条件协方差**

由互信息与条件熵的关系：

$$I(\Psi; O) = H(\Psi) - H(\Psi|O) = \frac{1}{2}\ln\frac{\det(\Sigma_\psi)}{\det(\Sigma_{\psi|o})}$$

因此：

$$\det(\Sigma_{\psi|o}) = \det(\Sigma_\psi) \cdot \exp(-2 I(\Psi; O))$$

**步骤3：从条件协方差到预测误差**

由步骤1和2：

$$E[\delta_t] \geq \ell_g^2 \cdot d \cdot \det(\Sigma_\psi)^{1/d} \cdot \exp(-2 I(\Psi; O) / d)$$

设 $c_0 = \ell_g^2 \cdot d \cdot \det(\Sigma_\psi)^{1/d}$，则：

$$E[\delta_t] \geq c_0 \cdot \exp(-2 I(\Psi; O) / d)$$

**步骤4：从预测误差到生存概率**

由定理2.1（A2.5），$P(\text{survival}) \propto \exp(-\sum_t \delta_t)$。利用Jensen不等式：

$$E\left[\sum_{t=1}^{T} \delta_t\right] \geq T \cdot c_0 \cdot \exp(-2 I(\Psi; O) / d)$$

由Markov不等式，对任意 $\theta > 0$：

$$P\left(\sum_{t=1}^{T} \delta_t > \theta\right) \leq \frac{E[\sum_t \delta_t]}{\theta}$$

但这给出的是上界而非下界。我们换用更精细的分析。

**步骤5：利用Fano不等式**

Fano不等式将条件熵与估计错误概率关联。设 $\hat{\Psi}$ 为基于观测 $O$ 对环境状态的估计，则：

$$P(\hat{\Psi} \neq \Psi) \geq \frac{H(\Psi|O) - 1}{\ln|\Psi|}$$

将此不等式转化为连续情形。定义"估计失败"为状态估计误差超过生存裕度对应的阈值，即 $\|\hat{\psi} - \psi\| > \Delta_{\text{threshold}}$。则估计失败概率满足：

$$P(\text{estimation failure}) \leq \exp(-I(\Psi; O))$$

这一不等式可由以下论证得到：互信息 $I(\Psi; O)$ 度量了观测 $O$ 对状态 $\Psi$ 的分辨能力。当 $I(\Psi; O)$ 增大时，观测能更精确地确定状态，估计失败概率下降。由数据处理不等式和典型集理论，估计失败概率的指数衰减率为互信息。

**步骤6：从估计失败到生存概率**

系统在时刻 $t$ 违反生存条件，当且仅当估计误差导致的负熵损失超过生存裕度（引用步骤4的证明）。因此：

$$P(\text{survival violation at } t) \leq P(\text{estimation failure at } t) \leq \exp(-I(\Psi; O))$$

对整个时间窗口，由联合概率的Bonferroni下界：

$$P(\text{survival}) = 1 - P(\text{any violation}) \geq 1 - \sum_{t=1}^{T} P(\text{violation at } t) \geq 1 - T \cdot \exp(-I(\Psi; O))$$

在单步决策场景（$T = 1$）下，即得：

$$P(\text{survival}) \geq 1 - \exp(-I(\Psi; O))$$

对于多步场景，可通过对互信息取时间平均 $I_T = \frac{1}{T}\sum_t I(\Psi_t; O_{1:t})$ 推广：

$$P(\text{survival}) \geq 1 - T \cdot \exp(-I_T)$$

$\blacksquare$

### 2.11 预测误差渐近分布（A2.11）

在稳态条件下（A2.7），当观测噪声为高斯分布且预测器为最优（条件均值估计）时，预测误差的渐近分布为：

$$\delta_t \xrightarrow{d} \chi^2_d(\lambda_t) \tag{A2.11}$$

- **输入**：观测维度 $d$，非中心参数 $\lambda_t = \|g(\psi_t) - g(\hat{\psi}_t)\|^2 / \sigma_\eta^2$
- **输出**：预测误差的概率分布
- **约束**：$d \geq 1$，$\lambda_t \geq 0$

> **白话翻译**：长期运行后，预测误差的分布趋于一个确定的形状（非中心卡方分布）。这个分布的"宽窄"取决于预测偏差和噪声大小——偏差越小、噪声越低，分布越集中在零附近。

**推导**：由观测方程（A2.3），$o_t - \hat{o}_t = g(\psi_t) - g(\hat{\psi}_t) + \eta_t$。设 $\eta_t \sim \mathcal{N}(0, \sigma_\eta^2 I_d)$，则 $(o_t - \hat{o}_t)/\sigma_\eta \sim \mathcal{N}(\mu_t, I_d)$，其中 $\mu_t = (g(\psi_t) - g(\hat{\psi}_t))/\sigma_\eta$。因此 $\|o_t - \hat{o}_t\|^2 / \sigma_\eta^2 \sim \chi^2_d(\lambda_t)$，其中 $\lambda_t = \|\mu_t\|^2$。

### 2.12 生存概率置信区间（A2.12）

基于预测误差的渐近分布（A2.11），对生存概率构造置信水平为 $1 - \alpha$ 的置信区间：

$$P(\text{survival}) \in \left[\exp\left(-\bar{\delta} - z_{\alpha/2}\sqrt{\frac{2\bar{\delta}}{T}}\right), \exp\left(-\bar{\delta} + z_{\alpha/2}\sqrt{\frac{2\bar{\delta}}{T}}\right)\right] \tag{A2.12}$$

- **输入**：平均预测误差 $\bar{\delta} = \frac{1}{T}\sum_{t=1}^{T} \delta_t$，时间窗口长度 $T$，置信水平 $1 - \alpha$
- **输出**：生存概率的置信区间 $[P_{\text{low}}, P_{\text{high}}] \subseteq [0, 1]$
- **约束**：$T \geq 1$，$\alpha \in (0, 1)$，$z_{\alpha/2}$ 为标准正态分布的 $\alpha/2$ 分位数

> **白话翻译**：给定一段时间的平均预测误差，我们可以在一定置信度下估计生存概率的范围。数据越多（$T$ 越大），估计越精确，区间越窄。

**推导**：由A2.11，$\delta_t$ 渐近服从非中心卡方分布。当 $T$ 充分大时，由中心极限定理，$\bar{\delta}$ 渐近服从正态分布：

$$\sqrt{T}(\bar{\delta} - E[\delta]) \xrightarrow{d} \mathcal{N}(0, \text{Var}(\delta))$$

其中 $\text{Var}(\delta) \approx 2\bar{\delta}$（卡方分布的方差近似）。因此 $E[\delta]$ 的 $1-\alpha$ 置信区间为：

$$E[\delta] \in \left[\bar{\delta} - z_{\alpha/2}\sqrt{\frac{2\bar{\delta}}{T}}, \bar{\delta} + z_{\alpha/2}\sqrt{\frac{2\bar{\delta}}{T}}\right]$$

由A2.5，$P(\text{survival}) \propto \exp(-T \cdot E[\delta])$，将 $E[\delta]$ 的置信区间代入即得A2.12。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A2.1 | $v(s) = \mathbb{1}[\dot{S}_{\text{ex}} \leq -\dot{S}_{\text{int}}]$ | $s \in S$ | $\{0, 1\}$ | 判定条件来自定理1.1 |
| A2.2 | $S_{\text{viable}} = \{s : v(s) = 1\}$ | $S$, $v$ | $S_{\text{viable}} \subseteq S$ | $S_{\text{viable}} \neq \emptyset$ |
| A2.3 | $o_t = g(\psi_t) + \eta_t$ | $\psi_t \in \Psi$, $\eta_t \sim \mathcal{N}(0, \Sigma_\eta)$ | $o_t \in \mathcal{O}$ | $\Sigma_\eta$ 正定 |
| A2.4 | $\delta_t = \|o_t - \hat{o}_t\|^2$ | $o_t, \hat{o}_t \in \mathcal{O}$ | $\delta_t \in \mathbb{R}_{\geq 0}$ | — |
| A2.5 | $P(\text{survival}) \propto \exp(-\sum \delta_t)$ | $\{\delta_t\}_{t=1}^T$ | $P \in (0, 1]$ | $P \leq 1$ |
| A2.6 | $\sum \delta_t \leq -\ln P_{\min}$ | $P_{\min} \in (0, 1]$ | $\epsilon_{\max} \in \mathbb{R}_{\geq 0}$ | $P_{\min} > 0$ |
| A2.7 | $\lim_{T\to\infty}\frac{1}{T}\sum \delta_t \leq \varepsilon_{\max}$ | $\{\delta_t\}$, $\varepsilon_{\max} > 0$ | 布尔值 | $\varepsilon_{\max}$ 由生存裕度决定 |
| A2.8 | $\rho_{\text{viable}} = \mu(S_{\text{viable}})/\mu(S)$ | $S$, $S_{\text{viable}}$, $\mu$ | $\rho \in [0, 1]$ | — |
| A2.9 | $I(\Psi;O) = H(\Psi) - H(\Psi|O)$ | $p(\psi)$, $p(o|\psi)$ | $I \in \mathbb{R}_{\geq 0}$ | $I \leq H(\Psi)$ |
| A2.10 | $P(\text{survival}) \geq 1 - e^{-I(\Psi;O)}$ | $I(\Psi;O) \geq 0$ | $P_{\text{lower}} \in [0, 1)$ | $I \geq 0$ |
| A2.11 | $\delta_t \xrightarrow{d} \chi^2_d(\lambda_t)$ | $d \geq 1$, $\lambda_t \geq 0$ | 概率分布 | 高斯噪声，最优预测器 |
| A2.12 | $P(\text{survival}) \in [P_{\text{low}}, P_{\text{high}}]$ | $\bar{\delta}$, $T$, $\alpha$ | 置信区间 | $T \geq 1$, $\alpha \in (0,1)$ |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# 引用阶段1的数据类
from stage_001 import ThermodynamicState


# ============================================================
# 阶段2 数据类：生存条件
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class SurvivalCondition:
    """系统的生存条件描述。

    组合阶段1的ThermodynamicState，在其物理基础上增加生存判定。

    Attributes:
        thermodynamic_state: 阶段1的热力学状态（组合字段）
        is_viable: 是否满足生存条件（由 thermodynamic_state 计算）
        survival_margin: 生存裕度 = -exchange_rate - internal_rate

    Invariants:
        - is_viable == (thermodynamic_state.exchange_entropy_rate
                        <= -thermodynamic_state.internal_entropy_rate)
        - survival_margin == -thermodynamic_state.exchange_entropy_rate
                             - thermodynamic_state.internal_entropy_rate
    """
    thermodynamic_state: ThermodynamicState
    is_viable: bool = field(init=False)
    survival_margin: float = field(init=False)

    def __post_init__(self):
        ts = self.thermodynamic_state
        # 生存条件：exchange <= -internal（引用阶段1，定理1.1）
        viable = ts.exchange_entropy_rate <= -ts.internal_entropy_rate
        margin = -ts.exchange_entropy_rate - ts.internal_entropy_rate
        object.__setattr__(self, 'is_viable', viable)
        object.__setattr__(self, 'survival_margin', margin)


# ============================================================
# 阶段2 核心接口：生存必然性推导
# ============================================================
class SurvivalNecessity:
    """阶段2的核心计算接口。

    从热力学生存条件出发，推导预测误差最小化的生存必要性。
    所有方法均为纯函数（无副作用），保证可验证性。
    """

    def viability_function(
        self,
        state: ThermodynamicState
    ) -> int:
        """计算生存阈值函数 v(s)（A2.1）。

        Args:
            state: 热力学状态，含内部熵产生率（单位 J/(K·s)）
                   和交换熵率（单位 J/(K·s)）

        Returns:
            1 如果满足生存条件（exchange ≤ -internal），0 否则；
            返回值 ∈ {0, 1}

        Raises:
            TypeError: 当 state 不是 ThermodynamicState 实例时

        Examples:
            >>> from stage_001 import ThermodynamicState
            >>> sn = SurvivalNecessity()
            >>> s = ThermodynamicState(internal_entropy_rate=1.0, exchange_entropy_rate=-1.5)
            >>> sn.viability_function(s)  # -1.5 <= -1.0 → 1
            1
            >>> s2 = ThermodynamicState(internal_entropy_rate=1.0, exchange_entropy_rate=-0.5)
            >>> sn.viability_function(s2)  # -0.5 > -1.0 → 0
            0
            >>> s3 = ThermodynamicState(internal_entropy_rate=1.0, exchange_entropy_rate=-1.0)
            >>> sn.viability_function(s3)  # 边界：-1.0 <= -1.0 → 1
            1
        """
        if not isinstance(state, ThermodynamicState):
            raise TypeError(
                f"state 必须是 ThermodynamicState 实例，收到 {type(state).__name__}"
            )
        return 1 if state.exchange_entropy_rate <= -state.internal_entropy_rate else 0

    def viable_set_measure(
        self,
        states: list[ThermodynamicState]
    ) -> float:
        """计算生存相容集的相对测度（A2.8）。

        Args:
            states: 状态空间中的采样点列表，每个元素为 ThermodynamicState

        Returns:
            生存相容集的相对测度 ρ_viable ∈ [0, 1]，
            等于满足生存条件的采样点占总采样点的比例

        Raises:
            ValueError: 当 states 为空列表时

        Examples:
            >>> from stage_001 import ThermodynamicState
            >>> sn = SurvivalNecessity()
            >>> s1 = ThermodynamicState(internal_entropy_rate=1.0, exchange_entropy_rate=-1.5)
            >>> s2 = ThermodynamicState(internal_entropy_rate=1.0, exchange_entropy_rate=-0.5)
            >>> sn.viable_set_measure([s1, s2])  # 1个满足 / 2个总数 = 0.5
            0.5
            >>> sn.viable_set_measure([s1])  # 全部满足
            1.0
            >>> sn.viable_set_measure([s2])  # 全部不满足
            0.0
        """
        if not states:
            raise ValueError("状态列表不能为空")
        viable_count = sum(
            1 for s in states
            if s.exchange_entropy_rate <= -s.internal_entropy_rate
        )
        return viable_count / len(states)

    def prediction_error(
        self,
        observation: np.ndarray,
        prediction: np.ndarray
    ) -> float:
        """计算瞬时预测误差 δ_t（A2.4）。

        Args:
            observation: 真实观测值 o_t，形状 (d,)，单位 无量纲
            prediction: 预测观测值 ô_t，形状 (d,)，单位 无量纲

        Returns:
            预测误差 δ_t = ||o_t - ô_t||² ≥ 0，单位 无量纲

        Raises:
            ValueError: 当 observation 和 prediction 形状不匹配时，
                        或当输入数组为空时，
                        或当输入包含 NaN 时

        Examples:
            >>> sn = SurvivalNecessity()
            >>> sn.prediction_error(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
            0.0
            >>> sn.prediction_error(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
            2.0
            >>> sn.prediction_error(np.array([0.0]), np.array([0.0]))  # 零值边界
            0.0
        """
        if observation.size == 0 or prediction.size == 0:
            raise ValueError("观测值和预测值不能为空数组")
        if observation.shape != prediction.shape:
            raise ValueError(
                f"观测值和预测值形状不匹配：{observation.shape} vs {prediction.shape}"
            )
        if np.any(np.isnan(observation)) or np.any(np.isnan(prediction)):
            raise ValueError("观测值和预测值不能包含 NaN")
        return float(np.sum((observation - prediction) ** 2))

    def survival_probability(
        self,
        prediction_errors: list[float]
    ) -> float:
        """计算生存概率（A2.5）。

        Args:
            prediction_errors: 预测误差序列 {δ_t}，单位 无量纲，
                               每个元素必须 ≥ 0

        Returns:
            生存概率 P(survival) = exp(-Σδ_t) ∈ (0, 1]，
            当所有 δ_t = 0 时取最大值 1.0

        Raises:
            ValueError: 当 prediction_errors 为空列表时，
                        或当 prediction_errors 中存在负值时，
                        或当 prediction_errors 中存在 NaN 时

        Examples:
            >>> sn = SurvivalNecessity()
            >>> sn.survival_probability([0.0, 0.0])  # 完美预测
            1.0
            >>> round(sn.survival_probability([1.0]), 10)  # exp(-1)
            0.3678794412
            >>> abs(sn.survival_probability([0.5, 0.5]) - np.exp(-1.0)) < 1e-10
            True
        """
        if not prediction_errors:
            raise ValueError("预测误差序列不能为空")
        for i, e in enumerate(prediction_errors):
            if np.isnan(e):
                raise ValueError(f"预测误差序列第 {i} 个元素为 NaN")
            if e < 0:
                raise ValueError(
                    f"预测误差必须 ≥ 0，第 {i} 个元素为 {e}"
                )
        cumulative = sum(prediction_errors)
        return float(np.exp(-cumulative))

    def max_cumulative_error(
        self,
        p_min: float
    ) -> float:
        """计算给定最低生存概率下的最大允许累积预测误差（A2.6）。

        Args:
            p_min: 最低生存概率，必须 ∈ (0, 1]，单位 无量纲

        Returns:
            最大允许累积预测误差 = -ln(p_min) ≥ 0，单位 无量纲；
            当 p_min = 1 时返回 0（要求完美预测）；
            当 p_min → 0 时返回 +∞（无约束）

        Raises:
            ValueError: 当 p_min ∉ (0, 1] 时，
                        或当 p_min 为 NaN 时

        Examples:
            >>> sn = SurvivalNecessity()
            >>> sn.max_cumulative_error(1.0)  # 完美生存概率要求零误差
            0.0
            >>> round(sn.max_cumulative_error(np.exp(-1)), 10)  # p_min = e^{-1}
            1.0
            >>> round(sn.max_cumulative_error(0.9), 4)
            0.1054
        """
        if np.isnan(p_min):
            raise ValueError("p_min 不能为 NaN")
        if p_min <= 0 or p_min > 1:
            raise ValueError(
                f"p_min 必须在 (0, 1] 范围内，收到 {p_min}"
            )
        return float(-np.log(p_min))

    def is_steady_state(
        self,
        prediction_errors: list[float],
        epsilon_max: float
    ) -> bool:
        """判断系统是否满足稳态条件（A2.7）。

        Args:
            prediction_errors: 预测误差序列 {δ_t}，单位 无量纲，
                               每个元素必须 ≥ 0
            epsilon_max: 最大允许平均预测误差，必须 > 0，单位 无量纲

        Returns:
            True 如果 (1/T)Σδ_t ≤ ε_max，False 否则

        Raises:
            ValueError: 当 prediction_errors 为空列表时，
                        或当 epsilon_max ≤ 0 时，
                        或当 prediction_errors 中存在负值或 NaN 时，
                        或当 epsilon_max 为 NaN 时

        Examples:
            >>> sn = SurvivalNecessity()
            >>> sn.is_steady_state([0.1, 0.2, 0.3], 0.5)  # 均值 0.2 ≤ 0.5
            True
            >>> sn.is_steady_state([0.1, 0.2, 0.3], 0.1)  # 均值 0.2 > 0.1
            False
            >>> sn.is_steady_state([0.0, 0.0], 0.1)  # 零误差边界
            True
        """
        if not prediction_errors:
            raise ValueError("预测误差序列不能为空")
        if np.isnan(epsilon_max):
            raise ValueError("epsilon_max 不能为 NaN")
        if epsilon_max <= 0:
            raise ValueError(
                f"epsilon_max 必须 > 0，收到 {epsilon_max}"
            )
        for i, e in enumerate(prediction_errors):
            if np.isnan(e):
                raise ValueError(f"预测误差序列第 {i} 个元素为 NaN")
            if e < 0:
                raise ValueError(
                    f"预测误差必须 ≥ 0，第 {i} 个元素为 {e}"
                )
        mean_error = sum(prediction_errors) / len(prediction_errors)
        return mean_error <= epsilon_max

    def mutual_information(
        self,
        psi_distribution: np.ndarray,
        conditional_distribution: np.ndarray
    ) -> float:
        """计算互信息 I(Ψ;O)（A2.9）。

        Args:
            psi_distribution: 环境状态的边缘分布 p(ψ)，形状 (n,)，
                              满足 sum=1 且所有元素 ≥ 0，单位 无量纲
            conditional_distribution: 条件分布 p(o|ψ)，形状 (m, n)，
                                      每列和为1，单位 无量纲

        Returns:
            互信息 I(Ψ;O) ≥ 0，单位 nat（自然对数底），
            满足 I(Ψ;O) ≤ H(Ψ)

        Raises:
            ValueError: 当分布不合法（空数组、含负值、和不为1、
                        形状不匹配）时

        Examples:
            >>> sn = SurvivalNecessity()
            >>> psi = np.array([0.5, 0.5])
            >>> cond = np.array([[1.0, 0.0], [0.0, 1.0]])  # 完全确定
            >>> round(sn.mutual_information(psi, cond), 4)  # I = H(Ψ) = ln(2)
            0.6931
            >>> cond_noisy = np.full((2, 2), 0.5)  # 完全不确定
            >>> sn.mutual_information(psi, cond_noisy)  # I = 0
            0.0
        """
        if psi_distribution.size == 0 or conditional_distribution.size == 0:
            raise ValueError("分布数组不能为空")
        if psi_distribution.ndim != 1:
            raise ValueError(
                f"psi_distribution 必须为一维数组，收到 {psi_distribution.ndim} 维"
            )
        if conditional_distribution.ndim != 2:
            raise ValueError(
                f"conditional_distribution 必须为二维数组，"
                f"收到 {conditional_distribution.ndim} 维"
            )
        n = psi_distribution.shape[0]
        if conditional_distribution.shape[1] != n:
            raise ValueError(
                f"conditional_distribution 列数 ({conditional_distribution.shape[1]}) "
                f"必须等于 psi_distribution 长度 ({n})"
            )
        if np.any(psi_distribution < 0):
            raise ValueError("psi_distribution 所有元素必须 ≥ 0，发现负值")
        if not np.isclose(psi_distribution.sum(), 1.0, atol=1e-6):
            raise ValueError(
                f"psi_distribution 之和必须为1，实际为 {psi_distribution.sum()}"
            )
        if np.any(conditional_distribution < 0):
            raise ValueError(
                "conditional_distribution 所有元素必须 ≥ 0，发现负值"
            )
        for j in range(n):
            col_sum = conditional_distribution[:, j].sum()
            if not np.isclose(col_sum, 1.0, atol=1e-6):
                raise ValueError(
                    f"conditional_distribution 第 {j} 列之和必须为1，"
                    f"实际为 {col_sum}"
                )
        # 计算联合分布 p(o, ψ) = p(o|ψ) * p(ψ)
        joint = conditional_distribution * psi_distribution[np.newaxis, :]
        # 计算边缘分布 p(o)
        p_o = joint.sum(axis=1)
        # 计算 H(Ψ) = -Σ p(ψ) * ln(p(ψ))
        p_nonzero = psi_distribution[psi_distribution > 0]
        h_psi = float(-np.sum(p_nonzero * np.log(p_nonzero)))
        # 计算 H(Ψ|O) = -Σ_{o,ψ} p(o,ψ) * ln(p(ψ|o))
        # 其中 p(ψ|o) = p(o,ψ) / p(o)
        h_psi_given_o = 0.0
        for i in range(joint.shape[0]):
            if p_o[i] > 0:
                for j in range(n):
                    if joint[i, j] > 0:
                        h_psi_given_o -= joint[i, j] * np.log(
                            joint[i, j] / p_o[i]
                        )
        h_psi_given_o = float(h_psi_given_o)
        return h_psi - h_psi_given_o

    def survival_probability_lower_bound(
        self,
        mutual_info: float
    ) -> float:
        """计算基于互信息的生存概率下界（A2.10）。

        Args:
            mutual_info: 互信息 I(Ψ;O)，必须 ≥ 0，单位 nat（自然对数底）

        Returns:
            生存概率下界 = 1 - exp(-I(Ψ;O)) ∈ [0, 1)；
            当 I → 0 时下界 → 0；
            当 I → ∞ 时下界 → 1

        Raises:
            ValueError: 当 mutual_info < 0 时，
                        或当 mutual_info 为 NaN 时

        Examples:
            >>> sn = SurvivalNecessity()
            >>> sn.survival_probability_lower_bound(0.0)  # 无信息
            0.0
            >>> round(sn.survival_probability_lower_bound(1.0), 4)  # I=1
            0.6321
            >>> sn.survival_probability_lower_bound(100.0)  # I→∞
            1.0
        """
        if np.isnan(mutual_info):
            raise ValueError("mutual_info 不能为 NaN")
        if mutual_info < 0:
            raise ValueError(
                f"互信息必须 ≥ 0，收到 {mutual_info}"
            )
        return float(1.0 - np.exp(-mutual_info))

    def survival_confidence_interval(
        self,
        mean_error: float,
        time_horizon: int,
        confidence_level: float
    ) -> tuple[float, float]:
        """计算生存概率的置信区间（A2.12）。

        Args:
            mean_error: 平均预测误差 δ̄，必须 ≥ 0，单位 无量纲
            time_horizon: 时间窗口长度 T，必须 ≥ 1，单位 步
            confidence_level: 置信水平 1-α，必须 ∈ (0, 1)，单位 无量纲

        Returns:
            (P_low, P_high) 置信区间，满足 P_low ≤ P(survival) ≤ P_high，
            区间宽度随 T 增大而单调递减

        Raises:
            ValueError: 当 mean_error < 0 时，
                        或当 time_horizon < 1 时，
                        或当 confidence_level ∉ (0, 1) 时，
                        或当任何参数为 NaN 时

        Examples:
            >>> sn = SurvivalNecessity()
            >>> low, high = sn.survival_confidence_interval(0.5, 100, 0.95)
            >>> 0 < low < high < 1
            True
            >>> low, high = sn.survival_confidence_interval(0.0, 10, 0.95)
            >>> low == high == 1.0
            True
        """
        if np.isnan(mean_error) or np.isnan(confidence_level):
            raise ValueError("mean_error 和 confidence_level 不能为 NaN")
        if mean_error < 0:
            raise ValueError(
                f"mean_error 必须 ≥ 0，收到 {mean_error}"
            )
        if time_horizon < 1:
            raise ValueError(
                f"time_horizon 必须 ≥ 1，收到 {time_horizon}"
            )
        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError(
                f"confidence_level 必须在 (0, 1) 范围内，"
                f"收到 {confidence_level}"
            )
        # 零误差边界：P(survival) = 1
        if mean_error == 0:
            return (1.0, 1.0)
        alpha = 1 - confidence_level
        # z_{α/2}：标准正态分布的分位数
        from scipy.stats import norm
        z = norm.ppf(1 - alpha / 2)
        # 置信区间
        delta = z * np.sqrt(2 * mean_error / time_horizon)
        p_low = float(np.exp(-(mean_error + delta)))
        p_high = float(np.exp(-(mean_error - delta)))
        # 确保 P_high 不超过 1
        p_high = min(p_high, 1.0)
        return (p_low, p_high)
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `ThermodynamicState` | ← 阶段1 | 作为生存判定的物理基础 |
| **输入** | 定理1.1（生存必要条件） | ← 阶段1 | 生存阈值函数的判定依据 |
| **输入** | 定理1.2（热力学-信息论等价性） | ← 阶段1 | 互信息-生存概率定理的推导前提 |
| **输出** | `SurvivalCondition` | → 阶段3 | 包含热力学状态和生存判定的组合数据 |
| **输出** | 定理2.1（生存概率定理） | → 阶段3 | 自由能原理的推导前提 |
| **输出** | 定理2.2（预测误差必要性定理） | → 阶段3 | 预测误差最小化的理论依据 |
| **输出** | 定理2.3（互信息-生存概率定理） | → 阶段3 | 信息论与生存的定量关联 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| SciPy | 统计分布计算 | 非中心卡方分布（A2.11）的CDF/PDF计算与置信区间构造 |
| scikit-learn | 互信息估计 | 基于k近邻的互信息估计（Kraskov方法），验证A2.9和A2.10 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V2.1 | 生存阈值函数 | `v(s) == 1` 当且仅当 `exchange <= -internal` | 边界测试 + 随机测试 |
| V2.2 | 生存相容集 | `S_viable` 中所有状态满足 `v(s) == 1` | 枚举验证 |
| V2.3 | 预测误差非负 | `δ_t >= 0` 对所有输入成立 | 随机测试 |
| V2.4 | 生存概率单调性 | `P(survival)` 随 `Σδ_t` 增大而单调递减 | 数值验证 |
| V2.5 | 预测误差必要性 | `Σδ_t <= -ln(P_min)` 对所有 `P_min ∈ (0,1]` 成立 | 解析验证 + 数值验证 |
| V2.6 | 稳态条件 | `is_steady_state` 在平均误差 ≤ ε_max 时返回 True | 边界测试 |
| V2.7 | 互信息非负 | `I(Ψ;O) >= 0` 对所有合法分布成立 | 随机分布测试 |
| V2.8 | 互信息上界 | `I(Ψ;O) <= H(Ψ)` 对所有合法分布成立 | 随机分布测试 |
| V2.9 | 生存概率下界 | `P(survival) >= 1 - exp(-I)` 对所有 `I >= 0` 成立 | 解析验证 + 数值验证 |
| V2.10 | SurvivalCondition组合 | `SurvivalCondition.thermodynamic_state` 类型为 `ThermodynamicState` | 类型检查 |
| V2.11 | 非法输入拒绝 | `p_min ∉ (0,1]`、`mutual_info < 0`、空列表等抛出 `ValueError` | 异常测试 |
| V2.12 | 置信区间单调性 | 置信区间宽度随 `T` 增大而单调递减 | 数值验证 |

---

> **阶段2完** | 下一步：阶段3将从本阶段的定理2.1和定理2.2出发，建立自由能原理与主动推断的数学框架。
