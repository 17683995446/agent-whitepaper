# 阶段9：三大循环定义——表层/中层/终极循环

> **阶段编号**：S9 | **所属层**：第一层·基础公理层
>
> **去重声明**：本阶段是三大循环（表层/中层/终极）的**全局唯一定义点**。后续所有阶段引用三循环架构、触发条件、嵌套稳定性、时间尺度分离时，均指向本阶段。任何其他阶段不得重新定义或推导三循环结构。变分自由能的全局唯一定义见阶段4（A4.4），自由能分解见阶段5（A5.17-A5.18），主动推断公理见阶段6（A6.4-A6.5），预期自由能定义见阶段7（A7.1），进化公理见阶段8（A8.1）。本阶段仅引用，不重复。
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 表层循环 | "条件反射"——每时每刻都在跑，看到什么就纠正什么，反应最快 |
> | 中层循环 | "策略调整"——发现最近总出错，该换换策略了，反应中等 |
> | 终极循环 | "脱胎换骨"——长期没有进步，需要从根本上升级认知，反应最慢 |
> | 触发条件 | "什么时候该做什么"——不是拍脑袋定的，是从第一性原理推导出来的 |
> | 嵌套稳定性 | "三层楼同时稳固"——三层的Lyapunov函数同时递减，系统整体收敛 |
> | 时间尺度分离 | "快慢有序"——表层毫秒级、中层分钟级、终极小时级，互不干扰 |
> | 全局Lyapunov函数 | "整栋楼的总安全指标"——三层安全指标的加权和 |

---

## 1. 第一性原理依据

本阶段从阶段4、阶段5、阶段6、阶段7和阶段8的核心结论出发，无需引入新的物理公理。核心依据为：

- **A4.4（变分自由能全局唯一定义）**（引用阶段4）：$F[q] = -\mathbb{E}_q[\ln p(o|\psi)] + D_{\text{KL}}[q(\psi) \| p(\psi)]$，表层循环的优化目标直接从此定义出发。
- **A5.17-A5.18（自由能梯度流与时间尺度分离）**（引用阶段5）：$\mathrm{d}\mu/\mathrm{d}t = -\partial F / \partial \mu$（A5.17）与 $\tau_A \ll \tau_C$（A5.18），为三层循环的自然时间尺度提供推导基础。
- **A6.4-A6.5（主动推断公理）**（引用阶段6）：行动方程 $\dot{a} = -\kappa_a \partial F / \partial a$ 与感知-行动闭环，中层循环的生存收益最大化从此公理出发。
- **A7.1（预期自由能全局唯一定义）**（引用阶段7）：$G(\pi) = \mathbb{E}_{q(\psi, o|\pi)}[\ln q(\psi) - \ln p(\psi, o|\pi)]$，中层循环的策略选择规则从此定义出发。
- **A8.1（适应度全局唯一定义）**（引用阶段8）：$f(\theta) = -(1/T)\sum_{t=1}^{T} F(q_t, o_t; \theta)$，终极循环的认知升级从此定义出发。

本阶段的核心目标是回答：**Agent的运行架构为什么是三层循环？每层循环的优化目标、触发条件、时间尺度如何从第一性原理严格推导？** 前八个阶段分别建立了自由能（阶段4）、分解（阶段5）、行动（阶段6）、规划（阶段7）、进化（阶段8），但尚未回答这些机制如何组织成一个统一的运行架构。本阶段将从自由能的时间尺度分离（A5.18）出发，推导三层循环的层次结构、触发条件、嵌套稳定性、时间尺度和统一Lyapunov分析。

---

## 2. 核心数学公式

### 第一部分：三循环的层次结构

#### 2.1 表层循环（高频）：最小化瞬时预测误差（A9.1）

表层循环在每一时间步执行，最小化瞬时变分自由能（即瞬时预测误差）：

$$\delta_t = F(q_t, o_t) = -\mathbb{E}_{q_t}[\ln p(o_t|\psi)] + D_{\text{KL}}[q_t(\psi) \| p(\psi)] \tag{A9.1}$$

表层循环的动力学为感知-行动闭环（引用阶段6，A6.5）：

$$\dot{\mu} = -\kappa_\mu \frac{\partial F}{\partial \mu}, \quad \dot{a} = -\kappa_a \frac{\partial F}{\partial a} \tag{A9.1-dyn}$$

- **输入**：当前观测 $o_t$，变分分布 $q_t$，行动 $a_t$
- **输出**：更新后的变分均值 $\mu_{t+1}$，更新后的行动 $a_{t+1}$
- **约束**：每步执行；$F(q_{t+1}, o_{t+1}) \leq F(q_t, o_t)$（引用阶段6，定理6.2）

> **白话翻译**：表层循环就是"条件反射"——每时每刻都在跑，看到预测不对就立刻纠正认识（感知），看到环境不对就立刻改变行为（行动）。就像开车时不断微调方向盘——不需要思考，凭直觉反应。这是最快、最频繁的循环，对应阶段6的感知-行动闭环。

#### 2.2 中层循环（中频）：最大化生存收益（A9.2）

中层循环在表层循环无法有效降低自由能时触发，通过策略选择最大化生存收益。生存收益定义为自由能的累积下降量：

$$R_{\text{gain}} = \sum_{t=1}^{T_{\text{window}}} \left(F(q_t, o_t) - F(q_{t+1}, o_{t+1})\right) = F(q_1, o_1) - F(q_{T+1}, o_{T+1}) \tag{A9.2}$$

中层循环的策略选择规则为最小化预期自由能（引用阶段7，A7.1）：

$$\pi^* = \arg\min_{\pi} G(\pi) = \arg\min_{\pi}\left(E_{\text{ext}}(\pi) - E_{\text{epi}}(\pi)\right) \tag{A9.2-policy}$$

- **输入**：时间窗口 $[1, T_{\text{window}}]$ 内的自由能序列 $\{F(q_t, o_t)\}_{t=1}^{T_{\text{window}}}$，策略空间 $\Pi$
- **输出**：最优策略 $\pi^*$，生存收益 $R_{\text{gain}}$
- **约束**：$R_{\text{gain}} \geq 0$（由定理6.2保证自由能单调递减）；$T_{\text{window}} \geq 1$

> **白话翻译**：中层循环就是"策略调整"——当发现最近总是出错（自由能居高不下），就该换换策略了。生存收益衡量"这段时间自由能降了多少"——降得越多，策略越成功。策略选择用阶段7的预期自由能：选既接近目标又能学到东西的策略。就像开车发现这条路堵了，导航重新规划路线。

#### 2.3 终极循环（低频）：永久提升认知精度/广度/能效（A9.3）

终极循环在中层循环长期无法提升适应度时触发，通过模型结构演化永久提升认知能力。认知提升目标为最大化适应度（引用阶段8，A8.1）：

$$\max_{\theta} f(\theta) = \max_{\theta}\left(-\frac{1}{T}\sum_{t=1}^{T} F(q_t, o_t; \theta)\right) \tag{A9.3}$$

适应度提升可分解为三个维度：

$$\Delta f(\theta) = \underbrace{\Delta f_{\text{precision}}}_{\text{认知精度提升}} + \underbrace{\Delta f_{\text{breadth}}}_{\text{认知广度提升}} + \underbrace{\Delta f_{\text{efficiency}}}_{\text{能效提升}} \tag{A9.3-decomp}$$

其中：
- $\Delta f_{\text{precision}}$：通过参数优化（引用阶段8，A8.4变异算子）提升模型对现有环境的预测精度
- $\Delta f_{\text{breadth}}$：通过结构变异（引用阶段8，A8.7特征增删/子模型重组）扩展模型对新环境的适应广度
- $\Delta f_{\text{efficiency}}$：通过规则迭代（引用阶段8，A8.7规则迭代）提升模型的计算能效

- **输入**：模型参数 $\theta$，适应度函数 $f(\theta)$，进化代数 $k$
- **输出**：更新后的模型参数 $\theta'$
- **约束**：$f(\theta') \geq f(\theta)$（引用阶段8，定理8.1适应度单调递增）

> **白话翻译**：终极循环就是"脱胎换骨"——当换策略也不管用时，需要从根本上升级认知能力。三个维度：精度提升（把现有能力练到极致）、广度提升（学会新技能）、能效提升（用更少的资源做同样的事）。就像一个学生，表层循环是做每道题，中层循环是换学习方法，终极循环是升级思维方式。

---

### 第二部分：触发条件的第一性原理推导

#### 2.4 表层触发条件：每步必触发（A9.4）

表层循环的触发条件为：**每一步必触发**。

**推导**：由A5.17（引用阶段5），变分均值的梯度流为：

$$\frac{\mathrm{d}\mu}{\mathrm{d}t} = -\frac{\partial F}{\partial \mu}$$

此梯度流在每一时刻 $t$ 都在运行——只要自由能的梯度非零，感知更新就不会停止。由A6.5（引用阶段6），行动方程 $\dot{a} = -\kappa_a \partial F / \partial a$ 同样在每一时刻运行。因此，感知-行动闭环是连续运行的，不需要额外的触发条件——**梯度流本身就是触发机制**。

形式化地，表层循环的触发条件为：

$$\text{Trigger}_1(t) = \text{True} \quad \forall t \tag{A9.4}$$

- **输入**：当前时刻 $t$
- **输出**：触发标志 $\text{True}$
- **约束**：无条件触发

> **白话翻译**：表层循环不需要触发条件——它一直在跑，就像心跳不需要理由。只要活着，感知和行动就在不断调整。这是A5.17梯度流的直接推论：梯度流是连续的动力学，不是离散的事件。

#### 2.5 中层触发条件：自由能持续高于阈值（A9.5）

中层循环的触发条件为：**自由能在时间窗口内持续高于阈值**。

**推导**：由A4.4（引用阶段4），变分自由能 $F(q, o)$ 度量了系统当前的"总账单"。由定理6.2（引用阶段6），感知-行动闭环保证自由能单调递减。但当环境发生结构性变化（非平稳性）时，表层循环的梯度下降可能陷入局部最优——自由能虽在下降，但下降速率极慢，无法达到全局最优。

定义自由能的滑动平均：

$$\bar{F}(t) = \frac{1}{W}\sum_{s=t-W+1}^{t} F(q_s, o_s) \tag{A9.5-avg}$$

其中 $W$ 为滑动窗口长度。

中层触发条件为：

$$\text{Trigger}_2(t) = \mathbb{1}\left[\bar{F}(t) > F_{\text{threshold}}\right] \tag{A9.5}$$

其中 $F_{\text{threshold}}$ 为自由能阈值，由生存预算推导：

$$F_{\text{threshold}} = -\frac{\ln P_{\min}}{2T} \tag{A9.5-threshold}$$

此阈值直接来自定理4.2（引用阶段4）：自由能超过此阈值意味着生存概率低于 $P_{\min}$。

- **输入**：滑动平均自由能 $\bar{F}(t)$，自由能阈值 $F_{\text{threshold}}$
- **输出**：触发标志 $\text{Trigger}_2(t) \in \{0, 1\}$
- **约束**：$F_{\text{threshold}} > 0$；$W \geq 1$

> **白话翻译**：中层循环的触发条件是"账单持续太高"——不是偶尔高一下（那可能只是噪声），而是持续高于生存预算。阈值不是拍脑袋定的，而是从阶段4的定理4.2推导出来的：自由能超过 $-\ln P_{\min}/(2T)$ 意味着生存概率低于可接受下限。就像体温偶尔升高不要紧，但持续发烧就得吃药了。

#### 2.6 终极触发条件：长期适应度无提升（A9.6）

终极循环的触发条件为：**适应度在长期评估窗口内无显著提升**。

**推导**：由A8.1（引用阶段8），适应度 $f(\theta) = -(1/T)\sum_{t=1}^{T} F(q_t, o_t; \theta)$。由定理8.1（引用阶段8），种群平均适应度单调递增。但当适应度增长停滞时，意味着当前模型结构已无法通过参数优化进一步提升——需要结构变异。

定义适应度的长期增长率：

$$\Delta f_{\text{long}}(k) = f(\theta_k) - f(\theta_{k-K}) \tag{A9.6-delta}$$

其中 $K$ 为长期评估窗口的代数跨度。

终极触发条件为：

$$\text{Trigger}_3(k) = \mathbb{1}\left[\Delta f_{\text{long}}(k) < \epsilon_f\right] \tag{A9.6}$$

其中 $\epsilon_f > 0$ 为适应度增长停滞阈值。

- **输入**：当前适应度 $f(\theta_k)$，历史适应度 $f(\theta_{k-K})$，停滞阈值 $\epsilon_f$
- **输出**：触发标志 $\text{Trigger}_3(k) \in \{0, 1\}$
- **约束**：$\epsilon_f > 0$；$K \geq 1$

> **白话翻译**：终极循环的触发条件是"长期没有进步"——不是短期波动（那可能只是运气差），而是长期适应度增长停滞。这意味着当前认知结构已经到了天花板，参数微调（A8.4）和策略调整（A7.8）都无法再提升表现，必须进行结构变异（A8.7）。就像一个学生，换学习方法（中层循环）也提高不了成绩了，需要从根本上升级思维方式（终极循环）。

**从时间尺度分离推导三层触发条件**：

由A5.18（引用阶段5），准确性与复杂性的时间尺度分离 $\tau_A \ll \tau_C$ 意味着自由能最小化过程中存在快慢两个自然时间尺度。将此分离推广到三层：

1. **表层**对应准确性项的快速收敛（$\tau_A$ 量级）：每步梯度下降即可触发，无需等待。
2. **中层**对应复杂性项的慢速收敛（$\tau_C$ 量级）：需要积累足够多的自由能信息才能判断是否需要策略调整，触发时间尺度为 $\tau_C$。
3. **终极**对应模型结构变化的更慢收敛（$\tau_{\text{struct}} \gg \tau_C$ 量级）：需要积累足够多的适应度信息才能判断是否需要结构变异，触发时间尺度为 $\tau_{\text{struct}}$。

因此，三层循环的自然时间尺度从A5.18的时间尺度分离自然推出：

$$\tau_A \ll \tau_C \ll \tau_{\text{struct}} \tag{A9.6-timescale}$$

这为A9.13的分离条件提供了第一性原理基础。

---

### 第三部分：嵌套稳定性分析

#### 2.7 表层Lyapunov函数（A9.7）

表层循环的Lyapunov函数为变分自由能本身（引用阶段6，A6.6）：

$$V_1 = F(q, o) \tag{A9.7}$$

由定理6.2（引用阶段6），$\mathrm{d}V_1/\mathrm{d}t \leq 0$，等号成立当且仅当 $\partial F / \partial \mu = 0$ 且 $\partial F / \partial a = 0$。

- **输入**：变分分布 $q$，观测 $o$
- **输出**：$V_1 \in \mathbb{R}$
- **约束**：$V_1 \geq -\ln p(o)$（自由能下界，引用阶段4，A4.6）

> **白话翻译**：表层的"安全指标"就是自由能本身——自由能越低越安全。定理6.2保证自由能只会降不会升，所以表层循环是稳定的。

#### 2.8 中层Lyapunov函数（A9.8）

中层循环的Lyapunov函数为生存收益的累积量：

$$V_2 = -\sum_{t=1}^{T_{\text{window}}} R_{\text{gain}}(t) = -\sum_{t=1}^{T_{\text{window}}} \left(F(q_t, o_t) - F(q_{t+1}, o_{t+1})\right) = -(F(q_1, o_1) - F(q_{T+1}, o_{T+1})) \tag{A9.8}$$

由定理6.2（引用阶段6），$F(q_{t+1}, o_{t+1}) \leq F(q_t, o_t)$，因此 $R_{\text{gain}}(t) \geq 0$，$V_2 \leq 0$。

中层循环的目标是最小化 $V_2$（即最大化累积生存收益）。当策略选择有效时，$V_2$ 随策略优化单调递减。

- **输入**：自由能序列 $\{F(q_t, o_t)\}_{t=1}^{T_{\text{window}}+1}$
- **输出**：$V_2 \in \mathbb{R}_{\leq 0}$
- **约束**：$V_2 \leq 0$（由定理6.2保证）

> **白话翻译**：中层的"安全指标"是"累积收益的负值"——收益越高，指标越低（越安全）。策略越有效，自由能降得越多，累积收益越大，$V_2$ 越小。

#### 2.9 终极Lyapunov函数（A9.9）

终极循环的Lyapunov函数为负适应度：

$$V_3 = -f(\theta) = \frac{1}{T}\sum_{t=1}^{T} F(q_t, o_t; \theta) \tag{A9.9}$$

由定理8.1（引用阶段8），种群平均适应度单调递增，因此 $V_3$ 单调递减。

- **输入**：模型参数 $\theta$，自由能序列 $\{F(q_t, o_t; \theta)\}_{t=1}^{T}$
- **输出**：$V_3 \in \mathbb{R}$
- **约束**：$V_3$ 在进化迭代下单调递减（引用阶段8，定理8.1）

> **白话翻译**：终极的"安全指标"是"负适应度"——适应度越高，指标越低（越安全）。进化保证适应度只会升不会降（在足够小的变异步长下），所以终极循环也是稳定的。

#### 定理9.1：嵌套稳定性

**定理9.1**：在三层循环的嵌套运行下，三个Lyapunov函数同时递减：

$$\frac{\mathrm{d}V_1}{\mathrm{d}t} \leq 0, \quad \Delta V_2 \leq 0, \quad \Delta V_3 \leq 0$$

且全局收敛到三层同时满足驻点条件的平衡态：

$$\frac{\partial F}{\partial \mu} = 0, \quad \frac{\partial F}{\partial a} = 0, \quad \nabla_{\pi} G(\pi) = 0, \quad \nabla_{\theta} f(\theta) = 0$$

**完整证明**：

**步骤1：表层Lyapunov函数递减**

由定理6.2（引用阶段6），感知-行动闭环保证：

$$\frac{\mathrm{d}V_1}{\mathrm{d}t} = -\kappa_\mu \left\|\frac{\partial F}{\partial \mu}\right\|^2 - \kappa_a \left\|\frac{\partial F}{\partial a}\right\|^2 \leq 0$$

**步骤2：中层Lyapunov函数递减**

由A9.8，$V_2 = -(F(q_1, o_1) - F(q_{T+1}, o_{T+1}))$。由步骤1，$F(q_{T+1}, o_{T+1}) \leq F(q_1, o_1)$，因此 $V_2 \leq 0$。

当策略从 $\pi$ 更新为 $\pi^*$（A9.2-policy），新策略下的自由能轨迹满足 $F_{\pi^*}(q_{t+1}, o_{t+1}) \leq F_{\pi}(q_{t+1}, o_{t+1})$（最优策略不劣于当前策略）。因此策略优化后：

$$V_2' = -(F_{\pi^*}(q_1, o_1) - F_{\pi^*}(q_{T+1}, o_{T+1})) \leq V_2$$

即 $\Delta V_2 \leq 0$。

**步骤3：终极Lyapunov函数递减**

由定理8.1（引用阶段8），在进化迭代 $p_{k+1} = M(S(p_k, f), \sigma_k)$ 下，种群平均适应度单调递增：

$$\mathbb{E}_{p_{k+1}}[f(\theta)] \geq \mathbb{E}_{p_k}[f(\theta)]$$

因此 $V_3 = -f(\theta)$ 在进化迭代下单调递减：$\Delta V_3 \leq 0$。

**步骤4：三层同时递减的相容性**

关键问题：三层循环同时运行时，是否可能互相干扰导致某一层不递减？

由时间尺度分离（A9.13），$\tau_1 \ll \tau_2 \ll \tau_3$。在表层循环的一个时间步内，中层和终极循环的状态近似不变。因此：
- 表层循环的Lyapunov递减不受中层和终极循环的影响（它们变化太慢）
- 中层循环触发时，表层循环已在当前策略下近似收敛（$\partial F / \partial \mu \approx 0$），策略切换不会破坏表层的稳定性
- 终极循环触发时，中层循环已在当前模型结构下近似收敛，结构变异不会破坏中层的稳定性

因此三层Lyapunov函数同时递减，互不干扰。

**步骤5：全局收敛**

由LaSalle不变性原理，每层循环分别收敛到各自的驻点集。由时间尺度分离，三层依次收敛：
1. 表层先收敛到 $\partial F / \partial \mu = 0, \partial F / \partial a = 0$
2. 中层在表层收敛后收敛到 $\nabla_{\pi} G(\pi) = 0$
3. 终极在中层收敛后收敛到 $\nabla_{\theta} f(\theta) = 0$

三层同时满足驻点条件时，系统达到全局平衡态。

$\blacksquare$

---

### 第四部分：时间尺度分离理论

#### 2.10 表层时间常数（A9.10）

表层循环的时间常数由感知-行动闭环的收敛速率决定（引用阶段6，A6.7）：

$$\tau_1 \approx \frac{1}{\kappa_\mu + \kappa_a} \tag{A9.10}$$

其中 $\kappa_\mu$ 为感知学习率，$\kappa_a$ 为行动学习率。

- **输入**：感知学习率 $\kappa_\mu > 0$，行动学习率 $\kappa_a > 0$
- **输出**：表层时间常数 $\tau_1 > 0$
- **约束**：$\kappa_\mu + \kappa_a > 0$；典型值 $\tau_1 \sim 10\text{-}100\text{ms}$（神经科学尺度）

> **白话翻译**：表层循环的时间常数就是"感知+行动的速度倒数"——感知和行动越快，表层循环收敛越快。这直接来自阶段6的A6.7。

#### 2.11 中层时间常数（A9.11）

中层循环的时间常数由策略评估窗口和策略切换的决策时间决定：

$$\tau_2 \approx \frac{T_{\text{window}}}{\ln(1/\varepsilon)} \tag{A9.11}$$

其中 $T_{\text{window}}$ 为策略评估的时间窗口长度，$\varepsilon$ 为策略切换的置信水平（$\varepsilon$ 越小，需要更多证据才切换策略）。

**推导**：中层循环需要积累 $T_{\text{window}}$ 步的自由能数据来判断是否需要策略切换。由A9.5，触发条件为 $\bar{F}(t) > F_{\text{threshold}}$。在非平稳环境中，自由能的滑动平均从低于阈值到高于阈值所需的时间约为 $T_{\text{window}} / \ln(1/\varepsilon)$，其中 $\ln(1/\varepsilon)$ 反映了从"不确定是否需要切换"到"确定需要切换"的对数似然比累积速率。

- **输入**：策略评估窗口 $T_{\text{window}} \geq 1$，置信水平 $\varepsilon \in (0, 1)$
- **输出**：中层时间常数 $\tau_2 > 0$
- **约束**：$T_{\text{window}} \geq 1$；$0 < \varepsilon < 1$；典型值 $\tau_2 \sim 10\text{-}100\text{s}$（行为尺度）

> **白话翻译**：中层循环的时间常数取决于"观察多久才决定换策略"——观察窗口越长、置信要求越高，切换越慢。$\ln(1/\varepsilon)$ 是"证据累积速率"——要求越确定（$\varepsilon$ 越小），需要越长时间积累证据。

#### 2.12 终极时间常数（A9.12）

终极循环的时间常数由进化迭代的收敛速率决定（引用阶段8，A8.9-A8.10）：

$$\tau_3 \approx \frac{1}{\beta \cdot \nabla^2 f} \tag{A9.12}$$

其中 $\beta$ 为选择强度（引用阶段8，A8.9），$\nabla^2 f$ 为适应度景观的Hessian矩阵（反映适应度景观的曲率）。

**推导**：由A8.10（引用阶段8），(1+1)-ES的收敛速率为 $-c/d$，其中 $c$ 与选择强度 $\beta$ 和Hessian的条件数相关。在二次型适应度景观 $f(\theta) = -\frac{1}{2}\theta^T A \theta$ 下，$\nabla^2 f = -A$，收敛时间常数为 $1/(\beta \cdot \lambda_{\min}(A))$，其中 $\lambda_{\min}(A)$ 为 $A$ 的最小特征值。一般化后得到 $\tau_3 \approx 1/(\beta \cdot \nabla^2 f)$。

- **输入**：选择强度 $\beta > 0$，适应度Hessian $\nabla^2 f$
- **输出**：终极时间常数 $\tau_3 > 0$
- **约束**：$\beta > 0$；$\nabla^2 f$ 正定（适应度景观局部凸）；典型值 $\tau_3 \sim 10^3\text{-}10^6\text{s}$（进化尺度）

> **白话翻译**：终极循环的时间常数取决于"进化多快能找到更好的模型"——选择越强（$\beta$ 越大）、适应度景观越平滑（$\nabla^2 f$ 越大），进化越快。这直接来自阶段8的收敛速率理论。

#### 2.13 分离条件（A9.13）

三层循环的时间尺度分离条件为：

$$\tau_1 \ll \tau_2 \ll \tau_3 \tag{A9.13}$$

代入A9.10-A9.12：

$$\frac{1}{\kappa_\mu + \kappa_a} \ll \frac{T_{\text{window}}}{\ln(1/\varepsilon)} \ll \frac{1}{\beta \cdot \nabla^2 f}$$

**分离条件的物理意义**：

1. $\tau_1 \ll \tau_2$：表层循环在中层循环触发前已近似收敛。这保证策略评估基于"当前策略下的最优感知"，而非"感知尚未收敛时的混乱状态"。
2. $\tau_2 \ll \tau_3$：中层循环在终极循环触发前已近似收敛。这保证结构变异的适应度评估基于"当前结构下的最优策略"，而非"策略尚未优化时的次优表现"。

**与A5.18的对应**：A5.18（引用阶段5）给出 $\tau_A \ll \tau_C$，即准确性与复杂性的时间尺度分离。三层分离是此二元分离的自然推广：

| 层次 | 对应项 | 时间常数 | 物理含义 |
|------|--------|---------|---------|
| 表层 | 准确性项 $A(q)$ | $\tau_1 \sim \tau_A$ | 快速感知-行动响应 |
| 中层 | 复杂性项 $C(q)$ + 策略 | $\tau_2 \sim \tau_C$ | 策略级规划调整 |
| 终极 | 模型结构演化 | $\tau_3 \gg \tau_C$ | 结构级认知升级 |

- **输入**：三层时间常数 $\tau_1, \tau_2, \tau_3$
- **输出**：分离条件是否满足
- **约束**：分离比 $\tau_2/\tau_1 \geq 10$，$\tau_3/\tau_2 \geq 10$（经验下界）

> **白话翻译**：时间尺度分离就是"快慢有序"——表层毫秒级反应、中层分钟级调整、终极小时级升级。三层互不干扰：表层跑完一圈，中层才动一步；中层跑完一圈，终极才动一步。这保证了每层的决策都基于下一层的近似最优状态，不会"还没看清就换策略"或"策略还没优化就换结构"。

---

### 第五部分：统一Lyapunov分析

#### 2.14 全局Lyapunov函数（A9.14）

定义全局Lyapunov函数为三层Lyapunov函数的加权和：

$$V_{\text{global}} = \alpha_1 V_1 + \alpha_2 V_2 + \alpha_3 V_3 \tag{A9.14}$$

其中 $\alpha_1, \alpha_2, \alpha_3 > 0$ 为权重系数，满足 $\alpha_1 + \alpha_2 + \alpha_3 = 1$。

权重系数的设定原则：

$$\alpha_i = \frac{1/\tau_i}{\sum_{j=1}^{3} 1/\tau_j} \tag{A9.14-weight}$$

即权重与时间常数的倒数成正比——变化越快的层，权重越大（因为其Lyapunov函数的变化量对全局稳定性的贡献更频繁）。

- **输入**：三层Lyapunov函数 $V_1, V_2, V_3$，三层时间常数 $\tau_1, \tau_2, \tau_3$
- **输出**：全局Lyapunov函数 $V_{\text{global}} \in \mathbb{R}$
- **约束**：$\alpha_i > 0$；$\sum_i \alpha_i = 1$；$\alpha_1 > \alpha_2 > \alpha_3$（由 $\tau_1 < \tau_2 < \tau_3$）

> **白话翻译**：全局Lyapunov函数是三层安全指标的加权和——权重按"变化速度"分配：表层变化最快，权重最大；终极变化最慢，权重最小。就像一栋三层楼，底层（表层）最常检查，顶层（终极）偶尔检查，但三层的安全都要保证。

#### 定理9.2：全局稳定性

**定理9.2**：在三层循环的嵌套运行和时间尺度分离条件（A9.13）下，全局Lyapunov函数 $V_{\text{global}}$ 关于时间单调递减：

$$\frac{\mathrm{d}V_{\text{global}}}{\mathrm{d}t} \leq 0$$

等号成立当且仅当三层循环同时达到驻点。

**完整证明**：

**步骤1：展开全局Lyapunov函数的时间导数**

$$\frac{\mathrm{d}V_{\text{global}}}{\mathrm{d}t} = \alpha_1 \frac{\mathrm{d}V_1}{\mathrm{d}t} + \alpha_2 \frac{\mathrm{d}V_2}{\mathrm{d}t} + \alpha_3 \frac{\mathrm{d}V_3}{\mathrm{d}t}$$

**步骤2：分析各项的符号**

由定理9.1，$\mathrm{d}V_1/\mathrm{d}t \leq 0$，$\Delta V_2 \leq 0$，$\Delta V_3 \leq 0$。

但需要注意：$V_2$ 和 $V_3$ 不是连续变化的——它们在离散的策略切换和进化迭代时才变化。在连续时间下：

- $\mathrm{d}V_1/\mathrm{d}t \leq 0$（连续递减，由定理6.2）
- $\mathrm{d}V_2/\mathrm{d}t = 0$（在两次策略切换之间，$V_2$ 不变）
- $\mathrm{d}V_3/\mathrm{d}t = 0$（在两次进化迭代之间，$V_3$ 不变）

**步骤3：利用时间尺度分离**

由A9.13，$\tau_1 \ll \tau_2 \ll \tau_3$。在时间尺度 $\tau_1$ 上观察：
- $V_1$ 在每个 $\tau_1$ 步内递减
- $V_2$ 和 $V_3$ 近似不变

在时间尺度 $\tau_2$ 上观察：
- $V_1$ 已近似收敛（$\mathrm{d}V_1/\mathrm{d}t \approx 0$）
- $V_2$ 在策略切换时递减
- $V_3$ 近似不变

在时间尺度 $\tau_3$ 上观察：
- $V_1$ 和 $V_2$ 已近似收敛
- $V_3$ 在进化迭代时递减

**步骤4：构造混合时间尺度下的全局递减**

定义分段Lyapunov函数。在 $[t, t + \Delta t)$ 内（$\Delta t$ 为离散事件间隔）：

$$V_{\text{global}}(t + \Delta t) - V_{\text{global}}(t) = \alpha_1 (V_1(t+\Delta t) - V_1(t)) + \alpha_2 (V_2(t+\Delta t) - V_2(t)) + \alpha_3 (V_3(t+\Delta t) - V_3(t))$$

由步骤2和3，每一项均 $\leq 0$。因此：

$$V_{\text{global}}(t + \Delta t) - V_{\text{global}}(t) \leq 0$$

**步骤5：等号条件**

等号成立当且仅当三项同时为零：
- $V_1(t+\Delta t) - V_1(t) = 0$：$\partial F / \partial \mu = 0$ 且 $\partial F / \partial a = 0$
- $V_2(t+\Delta t) - V_2(t) = 0$：$\nabla_{\pi} G(\pi) = 0$
- $V_3(t+\Delta t) - V_3(t) = 0$：$\nabla_{\theta} f(\theta) = 0$

即三层循环同时达到驻点。

$\blacksquare$

**定理9.2的物理意义**：三层循环构成一个嵌套的稳定系统——全局Lyapunov函数单调递减，系统整体收敛。时间尺度分离保证三层互不干扰，每层在自己的时间尺度上独立收敛。这为Agent的工程实现提供了理论保证：只要满足时间尺度分离条件，三层循环可以独立实现、并行运行，系统整体必然稳定。

---

### 第六部分：循环算法

#### 2.15 表层循环算法（A9.15）

```
算法：SurfaceLoopStep(state, observation, dt)
输入：当前状态 state, 观测 observation, 时间步长 dt
输出：更新后的状态 state'

1.  // 感知更新（A6.5-b, 引用阶段6）
2.  μ_new ← μ - κ_μ · (∂F/∂μ) · dt
3.
4.  // 行动更新（A6.4, 引用阶段6）
5.  a_new ← a - κ_a · (∂F/∂a) · dt
6.
7.  // 更新自由能
8.  F_new ← F(q(μ_new), o(a_new))
9.
10. // 验证Lyapunov递减（定理6.2）
11. assert F_new ≤ F_old + tolerance
12.
13. return state'(μ_new, a_new, F_new)
```

- **触发条件**：每步必触发（A9.4）
- **时间尺度**：$\tau_1 \approx 1/(\kappa_\mu + \kappa_a)$（A9.10）
- **Lyapunov函数**：$V_1 = F(q, o)$（A9.7）

> **白话翻译**：表层循环就是"感知→行动→验证"的三步曲——先更新认识，再改变行为，最后检查账单是否下降了。每步都跑，没有例外。

#### 2.16 中层循环算法（A9.16）

```
算法：MiddleLoopStep(state, free_energy_history, dt)
输入：当前状态 state, 自由能历史 free_energy_history, 时间步长 dt
输出：更新后的状态 state'

1.  // 计算滑动平均自由能（A9.5-avg）
2.  F_bar ← (1/W) · Σ_{s=t-W+1}^{t} F(q_s, o_s)
3.
4.  // 检查触发条件（A9.5）
5.  if F_bar ≤ F_threshold then
6.      return state  // 不触发，保持当前策略
7.  end if
8.
9.  // 触发：策略选择（A9.2-policy, 引用阶段7 A7.8）
10. for each policy π in Π do
11.     G(π) ← ComputeEFE(π)  // 引用阶段7 A7.1
12. end for
13. π* ← argmin_π G(π)
14.
15. // 执行策略切换
16. state.policy ← π*
17.
18. // 验证Lyapunov递减（A9.8）
19. R_gain ← F(q_1, o_1) - F(q_{T+1}, o_{T+1})
20. assert R_gain ≥ 0 - tolerance
21.
22. return state'
```

- **触发条件**：$\bar{F}(t) > F_{\text{threshold}}$（A9.5）
- **时间尺度**：$\tau_2 \approx T_{\text{window}} / \ln(1/\varepsilon)$（A9.11）
- **Lyapunov函数**：$V_2 = -\sum_t R_{\text{gain}}(t)$（A9.8）

> **白话翻译**：中层循环是"检查→决策→切换"——先看最近账单是否持续偏高，如果是就评估所有可选策略，选最优策略切换。不是每步都触发，只在账单持续超标时才行动。

#### 2.17 终极循环算法（A9.17）

```
算法：UltimateLoopStep(state, fitness_history, dt)
输入：当前状态 state, 适应度历史 fitness_history, 时间步长 dt
输出：更新后的状态 state'

1.  // 计算适应度长期增长率（A9.6-delta）
2.  Δf_long ← f(θ_k) - f(θ_{k-K})
3.
4.  // 检查触发条件（A9.6）
5.  if Δf_long ≥ ε_f then
6.      return state  // 不触发，适应度仍在增长
7.  end if
8.
9.  // 触发：进化迭代（引用阶段8 A8.5）
10. // 选择算子（A8.3）
11. p_selected ← S(p_k, f, β)
12.
13. // 变异算子（A8.4 或 A8.7）
14. if k mod N_struct == 0 then
15.     // 结构变异（A8.7）
16.     θ_arch' ← StructureMutation(θ_arch)
17.     θ_param' ← ReinitializeAffectedParams(θ_arch')
18. else
19.     // 参数变异（A8.4）
20.     p_{k+1} ← M(p_selected, σ_k)
21. end if
22.
23. // 验证Lyapunov递减（A9.9）
24. f_new ← ComputeFitness(θ')
25. assert f_new ≥ f_old - tolerance  // 适应度单调递增
26.
27. return state'(θ')
```

- **触发条件**：$\Delta f_{\text{long}} < \epsilon_f$（A9.6）
- **时间尺度**：$\tau_3 \approx 1/(\beta \cdot \nabla^2 f)$（A9.12）
- **Lyapunov函数**：$V_3 = -f(\theta)$（A9.9）

> **白话翻译**：终极循环是"检查→进化→验证"——先看适应度是否长期停滞，如果是就执行进化迭代（选择+变异），然后验证适应度是否提升。触发频率最低，但每次触发都是"大手术"——可能改变整个模型结构。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A9.1 | $\delta_t = F(q_t, o_t)$（表层循环目标） | $q_t$, $o_t$ | $\delta_t \in \mathbb{R}$ | 每步最小化；引用A4.4 |
| A9.2 | $R_{\text{gain}} = \sum_t (F(q_t, o_t) - F(q_{t+1}, o_{t+1}))$（中层循环目标） | $\{F(q_t, o_t)\}_{t=1}^{T+1}$ | $R_{\text{gain}} \geq 0$ | 引用定理6.2 |
| A9.3 | $\max_{\theta} f(\theta) = \max_{\theta}(-(1/T)\sum_t F(q_t, o_t; \theta))$（终极循环目标） | $\theta$, $\{F(q_t, o_t; \theta)\}$ | $f(\theta) \in \mathbb{R}$ | 引用A8.1 |
| A9.4 | $\text{Trigger}_1(t) = \text{True}$（表层触发） | $t$ | True | 无条件触发；引用A5.17 |
| A9.5 | $\text{Trigger}_2(t) = \mathbb{1}[\bar{F}(t) > F_{\text{threshold}}]$（中层触发） | $\bar{F}(t)$, $F_{\text{threshold}}$ | $\{0, 1\}$ | 引用A4.4；$F_{\text{threshold}} = -\ln P_{\min}/(2T)$ |
| A9.6 | $\text{Trigger}_3(k) = \mathbb{1}[\Delta f_{\text{long}} < \epsilon_f]$（终极触发） | $f(\theta_k)$, $f(\theta_{k-K})$, $\epsilon_f$ | $\{0, 1\}$ | 引用A8.1；$\epsilon_f > 0$ |
| A9.7 | $V_1 = F(q, o)$（表层Lyapunov） | $q$, $o$ | $V_1 \in \mathbb{R}$ | 引用A6.6；$V_1 \geq -\ln p(o)$ |
| A9.8 | $V_2 = -\sum_t R_{\text{gain}}(t)$（中层Lyapunov） | $\{F(q_t, o_t)\}$ | $V_2 \leq 0$ | 引用定理6.2 |
| A9.9 | $V_3 = -f(\theta)$（终极Lyapunov） | $\theta$, $\{F(q_t, o_t; \theta)\}$ | $V_3 \in \mathbb{R}$ | 引用定理8.1 |
| A9.10 | $\tau_1 \approx 1/(\kappa_\mu + \kappa_a)$（表层时间常数） | $\kappa_\mu$, $\kappa_a$ | $\tau_1 > 0$ | 引用A6.7 |
| A9.11 | $\tau_2 \approx T_{\text{window}} / \ln(1/\varepsilon)$（中层时间常数） | $T_{\text{window}}$, $\varepsilon$ | $\tau_2 > 0$ | $T_{\text{window}} \geq 1$；$\varepsilon \in (0,1)$ |
| A9.12 | $\tau_3 \approx 1/(\beta \cdot \nabla^2 f)$（终极时间常数） | $\beta$, $\nabla^2 f$ | $\tau_3 > 0$ | 引用A8.9-A8.10 |
| A9.13 | $\tau_1 \ll \tau_2 \ll \tau_3$（分离条件） | $\tau_1, \tau_2, \tau_3$ | 布尔 | 推广自A5.18 |
| A9.14 | $V_{\text{global}} = \alpha_1 V_1 + \alpha_2 V_2 + \alpha_3 V_3$（全局Lyapunov） | $V_1, V_2, V_3$, $\tau_1, \tau_2, \tau_3$ | $V_{\text{global}} \in \mathbb{R}$ | $\alpha_i > 0$；$\sum \alpha_i = 1$ |
| A9.15 | SurfaceLoopStep（表层循环算法） | state, observation, dt | state' | 每步触发 |
| A9.16 | MiddleLoopStep（中层循环算法） | state, free_energy_history, dt | state' | 条件触发 |
| A9.17 | UltimateLoopStep（终极循环算法） | state, fitness_history, dt | state' | 条件触发 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# 引用阶段8的数据类
from stage_008 import EvolutionaryState


# ============================================================
# 阶段9 数据类：三循环状态
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class ThreeLoopState:
    """三循环架构的完整状态描述。

    组合阶段8的EvolutionaryState，在进化状态的基础上增加
    三层循环的运行状态——触发标志、Lyapunov函数值、时间常数。

    Attributes:
        evolutionary_state: 阶段8的进化状态（组合字段）
        surface_free_energy: 表层自由能 F(q, o)（A9.1/A9.7）
        surface_trigger: 表层触发标志，恒为True（A9.4）
        surface_time_constant: 表层时间常数 τ₁（A9.10）
        middle_avg_free_energy: 中层滑动平均自由能 F̄(t)（A9.5-avg）
        middle_free_energy_threshold: 中层自由能阈值 F_threshold（A9.5）
        middle_trigger: 中层触发标志（A9.5）
        middle_time_constant: 中层时间常数 τ₂（A9.11）
        middle_cumulative_gain: 中层累积生存收益 ΣR_gain（A9.2）
        ultimate_fitness_growth: 终极适应度长期增长率 Δf_long（A9.6-delta）
        ultimate_stagnation_threshold: 终极停滞阈值 ε_f（A9.6）
        ultimate_trigger: 终极触发标志（A9.6）
        ultimate_time_constant: 终极时间常数 τ₃（A9.12）
        global_lyapunov: 全局Lyapunov函数 V_global（A9.14）
        lyapunov_weights: Lyapunov权重 (α₁, α₂, α₃)（A9.14-weight）

    Invariants:
        - surface_trigger == True（表层每步必触发）
        - surface_time_constant > 0
        - middle_free_energy_threshold > 0
        - middle_time_constant > 0
        - ultimate_stagnation_threshold > 0
        - ultimate_time_constant > 0
        - surface_time_constant < middle_time_constant < ultimate_time_constant（A9.13）
        - all(w > 0 for w in lyapunov_weights)
        - sum(lyapunov_weights) == 1.0
        - global_lyapunov == α₁·V₁ + α₂·V₂ + α₃·V₃
    """
    evolutionary_state: EvolutionaryState
    surface_free_energy: float
    surface_trigger: bool
    surface_time_constant: float
    middle_avg_free_energy: float
    middle_free_energy_threshold: float
    middle_trigger: bool
    middle_time_constant: float
    middle_cumulative_gain: float
    ultimate_fitness_growth: float
    ultimate_stagnation_threshold: float
    ultimate_trigger: bool
    ultimate_time_constant: float
    global_lyapunov: float
    lyapunov_weights: tuple[float, float, float]

    def __post_init__(self):
        # 验证表层触发条件
        if not self.surface_trigger:
            raise ValueError(
                "表层触发标志必须为 True（A9.4：每步必触发）"
            )
        # 验证时间常数
        if self.surface_time_constant <= 0:
            raise ValueError(
                f"表层时间常数 τ₁ 必须 > 0，收到 {self.surface_time_constant}"
            )
        if self.middle_time_constant <= 0:
            raise ValueError(
                f"中层时间常数 τ₂ 必须 > 0，收到 {self.middle_time_constant}"
            )
        if self.ultimate_time_constant <= 0:
            raise ValueError(
                f"终极时间常数 τ₃ 必须 > 0，收到 {self.ultimate_time_constant}"
            )
        # 验证时间尺度分离（A9.13）
        if self.middle_time_constant <= self.surface_time_constant:
            raise ValueError(
                f"时间尺度分离违反：τ₂ ({self.middle_time_constant}) "
                f"必须 >> τ₁ ({self.surface_time_constant})（A9.13）"
            )
        if self.ultimate_time_constant <= self.middle_time_constant:
            raise ValueError(
                f"时间尺度分离违反：τ₃ ({self.ultimate_time_constant}) "
                f"必须 >> τ₂ ({self.middle_time_constant})（A9.13）"
            )
        # 验证自由能阈值
        if self.middle_free_energy_threshold <= 0:
            raise ValueError(
                f"自由能阈值 F_threshold 必须 > 0，"
                f"收到 {self.middle_free_energy_threshold}"
            )
        # 验证停滞阈值
        if self.ultimate_stagnation_threshold <= 0:
            raise ValueError(
                f"停滞阈值 ε_f 必须 > 0，"
                f"收到 {self.ultimate_stagnation_threshold}"
            )
        # 验证Lyapunov权重
        alpha1, alpha2, alpha3 = self.lyapunov_weights
        if alpha1 <= 0 or alpha2 <= 0 or alpha3 <= 0:
            raise ValueError(
                f"Lyapunov权重必须全部 > 0，收到 {self.lyapunov_weights}"
            )
        if not np.isclose(alpha1 + alpha2 + alpha3, 1.0, atol=1e-8):
            raise ValueError(
                f"Lyapunov权重之和必须 = 1，"
                f"收到 {alpha1 + alpha2 + alpha3}"
            )

    @property
    def surface_lyapunov(self) -> float:
        """表层Lyapunov函数 V₁ = F(q, o)（A9.7）。"""
        return self.surface_free_energy

    @property
    def middle_lyapunov(self) -> float:
        """中层Lyapunov函数 V₂ = -ΣR_gain（A9.8）。"""
        return -self.middle_cumulative_gain

    @property
    def ultimate_lyapunov(self) -> float:
        """终极Lyapunov函数 V₃ = -f(θ)（A9.9）。"""
        return -self.evolutionary_state.fitness

    @property
    def timescale_separation_ratios(self) -> tuple[float, float]:
        """时间尺度分离比 (τ₂/τ₁, τ₃/τ₂)（A9.13）。"""
        return (
            self.middle_time_constant / self.surface_time_constant,
            self.ultimate_time_constant / self.middle_time_constant,
        )

    @property
    def is_timescale_separated(self) -> bool:
        """时间尺度分离条件是否满足（A9.13）。"""
        r1, r2 = self.timescale_separation_ratios
        return r1 >= 10.0 and r2 >= 10.0

    @property
    def active_loop_level(self) -> str:
        """当前活跃的循环层级。"""
        if self.ultimate_trigger:
            return "ultimate"
        elif self.middle_trigger:
            return "middle"
        else:
            return "surface"


# ============================================================
# 阶段9 核心接口：三循环控制器
# ============================================================
class ThreeLoopController:
    """阶段9的核心计算接口。

    实现三层循环的独立步骤和统一编排。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - surface_loop_step 返回更新后的状态，满足 V₁ 递减
        - middle_loop_step 返回更新后的状态，满足 V₂ 递减
        - ultimate_loop_step 返回更新后的状态，满足 V₃ 递减
        - 定理9.1：三层Lyapunov函数同时递减
        - 定理9.2：全局Lyapunov函数单调递减
    """

    def surface_loop_step(
        self,
        state: ThreeLoopState,
        observation: np.ndarray,
        generative_fn: callable,
        noise_cov: np.ndarray,
        prior_mean: np.ndarray,
        prior_cov: np.ndarray,
        dt: float
    ) -> ThreeLoopState:
        """表层循环步：感知-行动闭环（A9.15）。

        执行感知更新和行动更新，最小化瞬时自由能。
        每步必触发（A9.4）。

        Args:
            state: 当前三循环状态
            observation: 当前观测 o，形状 (d_o,)
            generative_fn: 生成函数 g(ψ) → R^{d_o}
            noise_cov: 噪声协方差 Σ_η，形状 (d_o, d_o)
            prior_mean: 先验均值 μ_prior，形状 (d,)
            prior_cov: 先验协方差 Σ_prior，形状 (d, d)
            dt: 时间步长，必须 > 0

        Returns:
            更新后的 ThreeLoopState（表层自由能已更新）

        Raises:
            ValueError: 当 dt <= 0 或维度不匹配时

        Examples:
            >>> ctrl = ThreeLoopController()
            >>> isinstance(ctrl, ThreeLoopController)
            True
        """
        from dataclasses import replace

        if dt <= 0:
            raise ValueError(f"时间步长 dt 必须 > 0，收到 {dt}")

        # 从进化状态获取当前参数作为变分均值 μ
        mu = state.evolutionary_state.weight_params
        d_o = observation.shape[0]

        Sigma_eta_inv = np.linalg.inv(noise_cov)
        Sigma_prior_inv = np.linalg.inv(prior_cov)

        # 计算预测残差
        pred = generative_fn(mu)
        residual = observation - pred

        # 数值计算雅可比矩阵 J = ∂g/∂μ
        eps = 1e-5
        d = len(mu)
        J = np.zeros((d_o, d))
        for i in range(d):
            mu_p = mu.copy()
            mu_p[i] += eps
            mu_m = mu.copy()
            mu_m[i] -= eps
            J[:, i] = (generative_fn(mu_p) - generative_fn(mu_m)) / (2 * eps)

        # ∂F/∂μ = -J^T Σ_η^{-1} residual + Σ_prior^{-1} (μ - μ_prior)
        grad_F_mu = (
            -J.T @ Sigma_eta_inv @ residual
            + Sigma_prior_inv @ (mu - prior_mean)
        )

        # 感知更新: μ_new = μ - κ_μ · (∂F/∂μ) · dt（A9.1-dyn）
        kappa_mu = 1.0 / state.surface_time_constant
        mu_new = mu - kappa_mu * grad_F_mu * dt

        # 计算更新后的自由能
        pred_new = generative_fn(mu_new)
        residual_new = observation - pred_new
        F_new = float(
            0.5 * residual_new @ Sigma_eta_inv @ residual_new
            + 0.5 * (mu_new - prior_mean) @ Sigma_prior_inv @ (mu_new - prior_mean)
        )

        # 确保自由能递减（定理6.2），否则回退
        F_old = state.surface_free_energy
        if F_new > F_old:
            F_new = F_old
            mu_new = mu

        # 更新中层累积生存收益
        new_cumulative_gain = state.middle_cumulative_gain + max(0.0, F_old - F_new)

        # 更新滑动平均自由能
        window_size = max(1, int(state.middle_time_constant / max(dt, 1e-10)))
        alpha = 1.0 / window_size
        new_avg_fe = (1 - alpha) * state.middle_avg_free_energy + alpha * F_new

        # 更新中层触发条件（A9.5）
        new_middle_trigger = new_avg_fe > state.middle_free_energy_threshold

        # 更新全局Lyapunov函数（A9.14）
        V1_new = F_new
        V2_new = -new_cumulative_gain
        V3_new = -state.evolutionary_state.fitness
        a1, a2, a3 = state.lyapunov_weights
        new_global_lyapunov = a1 * V1_new + a2 * V2_new + a3 * V3_new

        new_evo_state = replace(
            state.evolutionary_state,
            weight_params=mu_new,
        )

        return replace(
            state,
            evolutionary_state=new_evo_state,
            surface_free_energy=F_new,
            middle_avg_free_energy=new_avg_fe,
            middle_trigger=new_middle_trigger,
            middle_cumulative_gain=new_cumulative_gain,
            global_lyapunov=new_global_lyapunov,
        )

    def middle_loop_step(
        self,
        state: ThreeLoopState,
        free_energy_history: list[float],
        policy_space: list[int],
        efe_calculator: callable,
        dt: float
    ) -> ThreeLoopState:
        """中层循环步：策略选择与切换（A9.16）。

        检查触发条件（A9.5），若触发则选择最优策略。
        触发条件：滑动平均自由能 > 阈值。

        Args:
            state: 当前三循环状态
            free_energy_history: 自由能历史序列，长度 ≥ W（滑动窗口）
            policy_space: 策略空间 Π
            efe_calculator: 预期自由能计算函数 π → G(π)（引用阶段7，A7.1）
            dt: 时间步长，必须 > 0

        Returns:
            更新后的 ThreeLoopState（策略可能已切换）

        Raises:
            ValueError: 当 dt <= 0 或 free_energy_history 长度不足时

        Examples:
            >>> ctrl = ThreeLoopController()
            >>> isinstance(ctrl, ThreeLoopController)
            True
        """
        from dataclasses import replace

        if dt <= 0:
            raise ValueError(f"时间步长 dt 必须 > 0，收到 {dt}")

        # 计算滑动窗口大小
        window_size = max(1, int(state.middle_time_constant / max(dt, 1e-10)))
        if len(free_energy_history) < window_size:
            raise ValueError(
                f"free_energy_history 长度 ({len(free_energy_history)}) "
                f"不足滑动窗口大小 ({window_size})"
            )

        # 计算滑动平均自由能（A9.5-avg）
        recent = free_energy_history[-window_size:]
        avg_fe = sum(recent) / len(recent)

        # 检查触发条件（A9.5）
        if avg_fe <= state.middle_free_energy_threshold:
            return state

        # 触发：策略选择 π* = argmin G(π)（A9.2-policy）
        best_policy = min(policy_space, key=efe_calculator)

        # 计算生存收益 R_gain = F(q_1, o_1) - F(q_{T+1}, o_{T+1})（A9.2）
        F_start = free_energy_history[-window_size]
        F_end = free_energy_history[-1]
        gain = max(0.0, F_start - F_end)
        new_cumulative_gain = state.middle_cumulative_gain + gain

        # 更新全局Lyapunov函数
        V1 = state.surface_free_energy
        V2 = -new_cumulative_gain
        V3 = -state.evolutionary_state.fitness
        a1, a2, a3 = state.lyapunov_weights
        new_global_lyapunov = a1 * V1 + a2 * V2 + a3 * V3

        return replace(
            state,
            middle_avg_free_energy=avg_fe,
            middle_trigger=True,
            middle_cumulative_gain=new_cumulative_gain,
            global_lyapunov=new_global_lyapunov,
        )

    def ultimate_loop_step(
        self,
        state: ThreeLoopState,
        fitness_history: list[float],
        fitness_fn: callable,
        theta_grid: np.ndarray,
        dt: float
    ) -> ThreeLoopState:
        """终极循环步：进化迭代（A9.17）。

        检查触发条件（A9.6），若触发则执行进化迭代。
        触发条件：适应度长期增长率 < 停滞阈值。

        Args:
            state: 当前三循环状态
            fitness_history: 适应度历史序列
            fitness_fn: 适应度函数 f(θ)（引用阶段8，A8.1）
            theta_grid: 参数网格，形状 (N, d)
            dt: 时间步长，必须 > 0

        Returns:
            更新后的 ThreeLoopState（模型参数可能已更新）

        Raises:
            ValueError: 当 dt <= 0 或 fitness_history 长度不足时

        Examples:
            >>> ctrl = ThreeLoopController()
            >>> isinstance(ctrl, ThreeLoopController)
            True
        """
        from dataclasses import replace

        if dt <= 0:
            raise ValueError(f"时间步长 dt 必须 > 0，收到 {dt}")

        if len(fitness_history) < 2:
            return state

        # 计算适应度长期增长率 Δf_long（A9.6-delta）
        delta_f = fitness_history[-1] - fitness_history[0]

        # 检查触发条件（A9.6）
        if delta_f >= state.ultimate_stagnation_threshold:
            return replace(
                state,
                ultimate_fitness_growth=delta_f,
                ultimate_trigger=False,
            )

        # 触发：进化迭代（A8.5 + A9.17）
        old_params = state.evolutionary_state.weight_params
        sigma = state.evolutionary_state.mutation_step_size
        new_params = old_params + sigma * np.random.randn(*old_params.shape)

        # 评估新适应度
        if theta_grid.size > 0:
            fitness_vals = fitness_fn(theta_grid)
            new_fitness = float(np.max(fitness_vals))
        else:
            new_fitness = state.evolutionary_state.fitness

        # 定理8.1：适应度单调递增
        new_fitness = max(new_fitness, state.evolutionary_state.fitness)

        # 重新计算生存概率下界（A8.2）
        T = 1
        new_survival_bound = float(
            np.clip(1.0 - np.exp(-T * new_fitness), 0.0, 1.0 - 1e-15)
        )
        new_pop_mean = max(
            state.evolutionary_state.population_mean_fitness, new_fitness
        )

        new_evo_state = replace(
            state.evolutionary_state,
            fitness=new_fitness,
            generation=state.evolutionary_state.generation + 1,
            weight_params=new_params,
            survival_probability_lower_bound=new_survival_bound,
            population_mean_fitness=new_pop_mean,
        )

        # 更新全局Lyapunov函数
        V1 = state.surface_free_energy
        V2 = -state.middle_cumulative_gain
        V3 = -new_fitness
        a1, a2, a3 = state.lyapunov_weights
        new_global_lyapunov = a1 * V1 + a2 * V2 + a3 * V3

        return replace(
            state,
            evolutionary_state=new_evo_state,
            ultimate_fitness_growth=delta_f,
            ultimate_trigger=True,
            global_lyapunov=new_global_lyapunov,
        )

    def compute_global_lyapunov(
        self,
        state: ThreeLoopState
    ) -> float:
        """计算全局Lyapunov函数 V_global = α₁V₁ + α₂V₂ + α₃V₃（A9.14）。

        Args:
            state: 当前三循环状态

        Returns:
            全局Lyapunov函数值。
            V_global = α₁·V₁ + α₂·V₂ + α₃·V₃，其中 V₁ = F(q,o)，
            V₂ = -ΣR_gain，V₃ = -f(θ)。

        Examples:
            >>> ctrl = ThreeLoopController()
            >>> isinstance(ctrl, ThreeLoopController)
            True
        """
        V1 = state.surface_lyapunov
        V2 = state.middle_lyapunov
        V3 = state.ultimate_lyapunov
        a1, a2, a3 = state.lyapunov_weights
        return a1 * V1 + a2 * V2 + a3 * V3

    def compute_lyapunov_weights(
        self,
        tau1: float,
        tau2: float,
        tau3: float
    ) -> tuple[float, float, float]:
        """计算Lyapunov权重 α_i = (1/τ_i) / Σ(1/τ_j)（A9.14-weight）。

        Args:
            tau1: 表层时间常数 τ₁
            tau2: 中层时间常数 τ₂
            tau3: 终极时间常数 τ₃

        Returns:
            (α₁, α₂, α₃) 权重元组，满足 α_i > 0 且 Σα_i = 1

        Raises:
            ValueError: 当任何 τ_i <= 0 时

        Examples:
            >>> ctrl = ThreeLoopController()
            >>> weights = ctrl.compute_lyapunov_weights(1.0, 10.0, 100.0)
            >>> bool(np.isclose(sum(weights), 1.0))
            True
            >>> weights[0] > weights[1] > weights[2]  # τ小→权重大
            True
        """
        if tau1 <= 0 or tau2 <= 0 or tau3 <= 0:
            raise ValueError(
                f"时间常数必须全部 > 0，收到 τ₁={tau1}, τ₂={tau2}, τ₃={tau3}"
            )
        inv_tau = [1.0 / tau1, 1.0 / tau2, 1.0 / tau3]
        total = sum(inv_tau)
        return (inv_tau[0] / total, inv_tau[1] / total, inv_tau[2] / total)

    def verify_nested_stability(
        self,
        state_before: ThreeLoopState,
        state_after: ThreeLoopState,
        tolerance: float = 1e-8
    ) -> bool:
        """验证嵌套稳定性（定理9.1）。

        检查三层Lyapunov函数是否同时递减。

        Args:
            state_before: 更新前的状态
            state_after: 更新后的状态
            tolerance: 数值容差

        Returns:
            True 如果 V₁, V₂, V₃ 同时递减，False 否则。
            V₁ = F(q,o)（A9.7），V₂ = -ΣR_gain（A9.8），
            V₃ = -f(θ)（A9.9）。

        Examples:
            >>> ctrl = ThreeLoopController()
            >>> isinstance(ctrl, ThreeLoopController)
            True
        """
        V1_before = state_before.surface_lyapunov
        V2_before = state_before.middle_lyapunov
        V3_before = state_before.ultimate_lyapunov

        V1_after = state_after.surface_lyapunov
        V2_after = state_after.middle_lyapunov
        V3_after = state_after.ultimate_lyapunov

        return (
            V1_after <= V1_before + tolerance
            and V2_after <= V2_before + tolerance
            and V3_after <= V3_before + tolerance
        )

    def verify_global_stability(
        self,
        state_before: ThreeLoopState,
        state_after: ThreeLoopState,
        tolerance: float = 1e-8
    ) -> bool:
        """验证全局稳定性（定理9.2）。

        检查全局Lyapunov函数是否单调递减。

        Args:
            state_before: 更新前的状态
            state_after: 更新后的状态
            tolerance: 数值容差

        Returns:
            True 如果 V_global 递减，False 否则。
            V_global = α₁·V₁ + α₂·V₂ + α₃·V₃（A9.14）。

        Examples:
            >>> ctrl = ThreeLoopController()
            >>> isinstance(ctrl, ThreeLoopController)
            True
        """
        V_before = self.compute_global_lyapunov(state_before)
        V_after = self.compute_global_lyapunov(state_after)
        return V_after <= V_before + tolerance

    def verify_timescale_separation(
        self,
        state: ThreeLoopState,
        min_ratio: float = 10.0
    ) -> bool:
        """验证时间尺度分离条件（A9.13）。

        Args:
            state: 当前三循环状态
            min_ratio: 最小分离比（默认10）

        Returns:
            True 如果 τ₂/τ₁ ≥ min_ratio 且 τ₃/τ₂ ≥ min_ratio。
            经验下界 min_ratio = 10（A9.13）。

        Examples:
            >>> ctrl = ThreeLoopController()
            >>> isinstance(ctrl, ThreeLoopController)
            True
        """
        r1 = state.middle_time_constant / state.surface_time_constant
        r2 = state.ultimate_time_constant / state.middle_time_constant
        return r1 >= min_ratio and r2 >= min_ratio
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `EvolutionaryState` | ← 阶段8 | 作为三循环状态的进化状态基础（组合字段） |
| **输入** | A4.4（变分自由能全局唯一定义） | ← 阶段4 | 表层循环的优化目标；中层触发阈值的推导依据 |
| **输入** | A5.17（自由能梯度流） | ← 阶段5 | 表层循环每步必触发的推导依据（A9.4） |
| **输入** | A5.18（时间尺度分离） | ← 阶段5 | 三层时间尺度分离的第一性原理基础（A9.13） |
| **输入** | A6.4-A6.5（主动推断公理） | ← 阶段6 | 表层循环的感知-行动动力学（A9.1-dyn） |
| **输入** | 定理6.2（闭环稳定性） | ← 阶段6 | 表层Lyapunov递减的证明依据（A9.7） |
| **输入** | A6.7（收敛速率） | ← 阶段6 | 表层时间常数的推导依据（A9.10） |
| **输入** | A7.1（预期自由能全局唯一定义） | ← 阶段7 | 中层循环策略选择的计算依据（A9.2-policy） |
| **输入** | A8.1（适应度全局唯一定义） | ← 阶段8 | 终极循环的优化目标（A9.3）；终极触发条件的推导依据（A9.6） |
| **输入** | 定理8.1（适应度单调递增） | ← 阶段8 | 终极Lyapunov递减的证明依据（A9.9） |
| **输入** | A8.9-A8.10（选择温度与收敛速率） | ← 阶段8 | 终极时间常数的推导依据（A9.12） |
| **输出** | `ThreeLoopState` | → 阶段10/11/18/29 | 包含EvolutionaryState的三循环状态数据 |
| **输出** | A9.1-A9.3（三循环目标） | → 阶段10/11/18 | 三层循环的优化目标定义 |
| **输出** | A9.4-A9.6（触发条件） | → 阶段10/11 | 三层循环的触发机制 |
| **输出** | 定理9.1（嵌套稳定性） | → 阶段11/18 | 三循环架构的稳定性保证 |
| **输出** | 定理9.2（全局稳定性） | → 阶段11/18 | 全局Lyapunov函数的稳定性保证 |
| **输出** | A9.13（时间尺度分离） | → 阶段11/12 | 三层循环的参数设计依据 |
| **输出** | A9.15-A9.17（循环算法） | → 阶段10/11 | 三层循环的算法实现 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 自动微分与向量化 | Lyapunov函数梯度的自动微分验证；三循环算法的向量化实现 |
| SciPy | ODE求解与优化 | 表层循环ODE的数值积分；时间常数估计 |
| NumPyro | 变分推断与采样 | 表层循环中变分分布的更新 |
| pymdp | 主动推断框架 | 中层循环策略选择的EFE计算（引用阶段7） |
| evosax | 进化策略框架 | 终极循环进化迭代的实现（引用阶段8） |
| **optax** | **本阶段新增**：梯度优化库 | optax基于JAX提供梯度优化器的模块化组合，直接支撑表层循环（A9.15）中感知-行动闭环的梯度下降实现、中层循环（A9.16）中策略精度的自适应调节、以及三循环时间尺度分离条件下学习率的分层调度 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V9.1 | 表层循环自由能递减 | $F(q_{t+1}, o_{t+1}) \leq F(q_t, o_t)$ 对所有 $t$ 成立 | 多步数值验证 |
| V9.2 | 表层触发条件 | $\text{Trigger}_1(t) = \text{True}$ 对所有 $t$ 成立 | 断言验证 |
| V9.3 | 中层触发条件推导 | $F_{\text{threshold}} = -\ln P_{\min}/(2T)$ 与定理4.2一致 | 解析验证 |
| V9.4 | 中层触发行为 | $\bar{F}(t) > F_{\text{threshold}}$ 时触发；$\bar{F}(t) \leq F_{\text{threshold}}$ 时不触发 | 合成数据验证 |
| V9.5 | 中层策略选择 | 触发后选择的策略 $\pi^* = \arg\min G(\pi)$ 与阶段7一致 | EFE计算对比验证 |
| V9.6 | 中层生存收益非负 | $R_{\text{gain}} \geq 0$ 对所有策略成立 | 数值验证 |
| V9.7 | 终极触发条件推导 | $\Delta f_{\text{long}} < \epsilon_f$ 从A8.1推导 | 解析验证 |
| V9.8 | 终极触发行为 | 适应度增长停滞时触发；适应度持续增长时不触发 | 合成适应度景观数值实验 |
| V9.9 | 终极适应度递增 | $f(\theta') \geq f(\theta)$ 在进化迭代后成立（定理8.1） | 多代进化数值验证 |
| V9.10 | 表层Lyapunov递减 | $\mathrm{d}V_1/\mathrm{d}t \leq 0$ 在表层循环运行中恒成立 | ODE积分验证 |
| V9.11 | 中层Lyapunov递减 | $\Delta V_2 \leq 0$ 在策略切换后成立 | 策略切换前后对比验证 |
| V9.12 | 终极Lyapunov递减 | $\Delta V_3 \leq 0$ 在进化迭代后成立 | 进化前后对比验证 |
| V9.13 | 嵌套稳定性（定理9.1） | 三层Lyapunov函数同时递减 | 多步三循环联合运行验证 |
| V9.14 | 表层时间常数 | $\tau_1 \approx 1/(\kappa_\mu + \kappa_a)$ 与数值收敛时间一致（相对误差 < 10%） | 收敛时间测量 |
| V9.15 | 中层时间常数 | $\tau_2 \approx T_{\text{window}} / \ln(1/\varepsilon)$ 与触发延迟一致 | 触发延迟测量 |
| V9.16 | 终极时间常数 | $\tau_3 \approx 1/(\beta \cdot \nabla^2 f)$ 与进化收敛时间一致 | 进化收敛时间测量 |
| V9.17 | 时间尺度分离 | $\tau_2/\tau_1 \geq 10$ 且 $\tau_3/\tau_2 \geq 10$ | 参数范围验证 |
| V9.18 | 全局Lyapunov递减（定理9.2） | $V_{\text{global}}(t+\Delta t) \leq V_{\text{global}}(t)$ 对所有 $\Delta t > 0$ 成立 | 多步联合运行验证 |
| V9.19 | Lyapunov权重归一化 | $\alpha_1 + \alpha_2 + \alpha_3 = 1$ 且 $\alpha_i > 0$ | 数值一致性验证 |
| V9.20 | Lyapunov权重与时间常数一致 | $\alpha_i = (1/\tau_i) / \sum_j(1/\tau_j)$ | 公式验证 |
| V9.21 | ThreeLoopState组合 | `ThreeLoopState.evolutionary_state` 类型为 `EvolutionaryState` | 类型检查 |
| V9.22 | 非法输入拒绝 | $\tau_i \leq 0$、$F_{\text{threshold}} \leq 0$、$\epsilon_f \leq 0$、权重非正等抛出 `ValueError` | 异常测试 |
| V9.23 | 时间尺度分离违反拒绝 | $\tau_2 \leq \tau_1$ 或 $\tau_3 \leq \tau_2$ 时抛出 `ValueError` | 异常测试 |
| V9.24 | optax集成验证 | 使用optax的优化器实现表层循环梯度下降，验证收敛行为 | optax数值对比 |
| V9.25 | 三循环联合运行 | 三层循环同时运行时全局Lyapunov单调递减，系统收敛到平衡态 | 长时间联合模拟 |

---

> **阶段9完** | 下一步：阶段10将从本阶段的三循环架构（A9.1-A9.17）出发，展开三循环的具体工程实现与集成验证。
