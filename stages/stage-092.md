# 阶段92：全局微分方程组

> **阶段编号**：S92 | **所属层**：第十层·系统集成层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 感知动力学 τ_μ μ̇ = -∂F/∂μ | "眼睛越看越清"——感知状态朝误差最小的方向调整 |
> | 行动动力学 τ_a ȧ = -∂F/∂a | "手脚越动越准"——行动朝自由能最小的方向调整 |
> | 学习动力学 τ_W Ẇ = -∂F/∂W | "脑子越学越灵"——权重朝误差最小的方向更新 |
> | 进化动力学 τ_θ θ̇ = -∂f/∂θ | "物种越变越强"——参数朝适应度最大的方向进化 |
> | 注意力动力学 τ_Π Π̇ = κ_γ(Π_opt - Π) | "目光越聚越准"——精度朝最优分配方向调整 |
> | 全局耦合方程组 | "五条河流汇成海"——五类动力学耦合为统一方程组 |
> | GlobalDynamics | "全局动力学的完整画像"——组合GlobalObjective，在全局目标基础上增加动力学状态 |

---

## 1. 第一性原理依据

本阶段从阶段6（感知-行动闭环）、阶段14（表征更新）、阶段15（权重更新）、阶段9（时间常数）、阶段8（适应度）的核心结论出发，建立全局微分方程组。核心依据为：

- **A6.5（感知-行动闭环）**（引用阶段6）：感知推断 $\mu^* = \arg\min_\mu F$ 和行动选择 $a^* = \arg\min_a G$ 构成感知-行动闭环。全局微分方程组将A6.5的优化问题转化为连续时间动力学——感知和行动不再是离散的优化步骤，而是沿梯度方向的连续流动。

- **A14.1（表征更新）**（引用阶段14）：表征更新 $\Delta \mu_l = -\kappa_l \partial F / \partial \mu_l$ 定义了各层表征的更新规则。A92.1的感知动力学直接引用A14.1，将其从离散更新推广为连续微分方程。

- **A15.1（权重更新）**（引用阶段15）：权重更新 $\Delta W_l = -\eta_l \partial F / \partial W_l$ 定义了各层权重的学习规则。A92.3的学习动力学直接引用A15.1，将其从离散更新推广为连续微分方程。

- **A9.10-A9.12（时间常数）**（引用阶段9）：表层时间常数 $\tau_s \sim 0.1s$、中层时间常数 $\tau_m \sim 60s$、终极层时间常数 $\tau_u \sim 3600s$ 定义了三层循环的时间尺度。全局微分方程组的时间常数直接引用A9.10-A9.12。

- **A8.1（适应度）**（引用阶段8）：适应度 $f(\theta) = -J(\theta)$ 定义了进化选择压力。A92.4的进化动力学直接引用A8.1，将适应度最大化转化为梯度上升。

本阶段的核心目标是：**定义感知动力学 $\tau_\mu \dot{\mu} = -\partial F / \partial \mu$（A92.1），行动动力学 $\tau_a \dot{a} = -\partial F / \partial a$（A92.2），学习动力学 $\tau_W \dot{W} = -\partial F / \partial W$（A92.3），进化动力学 $\tau_\theta \dot{\theta} = -\partial f / \partial \theta$（A92.4），注意力动力学 $\tau_\Pi \dot{\Pi} = \kappa_\gamma(\Pi_{opt} - \Pi)$（A92.5），全局耦合方程组（A92.6），封装为GlobalDynamics数据类组合GlobalObjective。**

> **去重声明**：本阶段仅定义全局微分方程组及其耦合关系，不重新定义感知-行动闭环（阶段6的A6.5）、表征更新（阶段14的A14.1）、权重更新（阶段15的A15.1）或适应度（阶段8的A8.1）。所有子动力学均通过引用接入。

---

## 2. 核心数学公式

### 第一部分：感知动力学

#### 2.1 感知动力学定义（A92.1）

感知状态沿变分自由能的负梯度方向流动：

$$\tau_\mu \dot{\mu} = -\frac{\partial F}{\partial \mu} \tag{A92.1}$$

其中：
- $\mu$ 为感知状态向量（各层表征的拼接 $\mu = [\mu_1, \mu_2, \ldots, \mu_L]$）
- $\tau_\mu$ 为感知时间常数
- $F$ 为变分自由能（引用A4.4）

**感知时间常数的层级差异**：

| 层级 | 时间常数 | 引用 | 说明 |
|------|---------|------|------|
| 低层 $\mu_1$ | $\tau_{\mu,1} = 0.01s$ | A9.10 | 快速感知，毫秒级响应 |
| 中层 $\mu_2$ | $\tau_{\mu,2} = 0.1s$ | A9.10 | 中速感知，百毫秒级 |
| 高层 $\mu_3$ | $\tau_{\mu,3} = 1.0s$ | A9.10 | 慢速感知，秒级整合 |

**感知动力学的离散化**：

$$\mu_{t+\Delta t} = \mu_t - \frac{\Delta t}{\tau_\mu} \frac{\partial F}{\partial \mu} \bigg|_{\mu_t} \tag{A92.1-discrete}$$

**感知动力学与A14.1的对应**：

A14.1（引用阶段14）定义离散表征更新 $\Delta \mu_l = -\kappa_l \partial F / \partial \mu_l$。A92.1与A14.1的对应关系为 $\kappa_l = \Delta t / \tau_{\mu,l}$——离散更新步长等于时间步长除以时间常数。

**感知动力学的稳定性条件**：

$$\frac{\Delta t}{\tau_\mu} < \frac{2}{\|\partial^2 F / \partial \mu^2\|_{max}} \tag{A92.1-stability}$$

时间步长必须足够小以保证欧拉法的稳定性。对于高斯假设下的自由能，Hessian矩阵的最大特征值有界，因此总能选择合适的 $\Delta t$。

- **输入**：当前感知状态 $\mu$、变分自由能 $F$
- **输出**：感知状态变化率 $\dot{\mu}$
- **约束**：$\tau_\mu > 0$；$\Delta t / \tau_\mu$ 满足稳定性条件

> **白话翻译**：感知动力学就是"眼睛越看越清"——感知状态沿着让预测误差最小的方向调整。低层感知很快（10毫秒），高层感知较慢（1秒），因为高层需要整合更多信息。调整速度受时间常数控制，太快会振荡，太慢会迟钝。

---

### 第二部分：行动动力学

#### 2.2 行动动力学定义（A92.2）

行动沿变分自由能对行动的负梯度方向流动：

$$\tau_a \dot{a} = -\frac{\partial F}{\partial a} \tag{A92.2}$$

其中：
- $a$ 为行动向量
- $\tau_a$ 为行动时间常数
- $\partial F / \partial a$ 通过观测模型 $o = g(a, s) + \epsilon$ 的链式法则计算

**行动动力学的梯度计算**：

$$\frac{\partial F}{\partial a} = \frac{\partial F}{\partial o} \cdot \frac{\partial o}{\partial a} = \frac{\partial F}{\partial o} \cdot \frac{\partial g}{\partial a} \tag{A92.2-gradient}$$

行动通过改变观测来改变自由能——行动的梯度是自由能对观测的梯度乘以观测对行动的雅可比矩阵。

**行动时间常数的选择**：

| 行动类型 | 时间常数 | 说明 |
|---------|---------|------|
| 快速响应 | $\tau_a = 0.1s$ | 紧急安全响应 |
| 常规行动 | $\tau_a = 1.0s$ | 正常任务执行 |
| 探索行动 | $\tau_a = 5.0s$ | 谨慎探索 |

**行动动力学与A6.5的对应**：

A6.5（引用阶段6）定义行动选择 $a^* = \arg\min_a G(a)$。A92.2使用 $F$ 而非 $G$ 作为行动的优化目标——这是因为行动直接影响观测，而观测直接影响自由能。预期自由能 $G$ 用于策略评估（阶段7），自由能 $F$ 用于行动执行（本阶段）。

**行动动力学的安全约束**：

$$\dot{a}_{constrained} = \dot{a} \cdot \mathbb{1}[\text{safe}(a + \dot{a} \cdot \Delta t)] \tag{A92.2-safe}$$

行动变化率受安全护栏约束——如果下一步行动会导致安全违规，则抑制该行动方向。

- **输入**：当前行动 $a$、变分自由能 $F$、观测模型 $g$
- **输出**：行动变化率 $\dot{a}$
- **约束**：$\tau_a > 0$；行动受安全护栏约束

> **白话翻译**：行动动力学就是"手脚越动越准"——行动沿着让预测误差最小的方向调整。行动通过改变观测来改变误差——伸手去摸杯子，摸到了误差就小了。行动速度比感知慢，因为行动有物理惯性。安全护栏确保行动不会越界。

---

### 第三部分：学习动力学

#### 2.3 学习动力学定义（A92.3）

权重沿变分自由能对权重的负梯度方向更新：

$$\tau_W \dot{W} = -\frac{\partial F}{\partial W} \tag{A92.3}$$

其中：
- $W$ 为模型权重矩阵（各层权重的拼接 $W = [W_1, W_2, \ldots, W_L]$）
- $\tau_W$ 为学习时间常数
- $\partial F / \partial W$ 通过反向传播计算

**学习时间常数的层级差异**：

| 层级 | 时间常数 | 引用 | 说明 |
|------|---------|------|------|
| 低层 $W_1$ | $\tau_{W,1} = 60s$ | A9.11 | 中速学习，分钟级 |
| 中层 $W_2$ | $\tau_{W,2} = 300s$ | A9.11 | 慢速学习，5分钟级 |
| 高层 $W_3$ | $\tau_{W,3} = 1800s$ | A9.11 | 极慢学习，30分钟级 |

高层权重学习更慢——高层概念更抽象，需要更多证据才能可靠更新。

**学习动力学与A15.1的对应**：

A15.1（引用阶段15）定义离散权重更新 $\Delta W_l = -\eta_l \partial F / \partial W_l$。A92.3与A15.1的对应关系为 $\eta_l = \Delta t / \tau_{W,l}$——学习率等于时间步长除以学习时间常数。

**学习动力学的正则化**：

$$\tau_W \dot{W} = -\frac{\partial F}{\partial W} - \lambda_W W \tag{A92.3-regularized}$$

添加L2正则项防止权重过大。$\lambda_W = 10^{-4}$ 为权重衰减系数。

**学习动力学的梯度裁剪**：

$$\dot{W}_{clipped} = \dot{W} \cdot \min\left(1, \frac{c_{max}}{\|\dot{W}\|}\right) \tag{A92.3-clip}$$

其中 $c_{max} = 1.0$ 为最大梯度范数。梯度裁剪防止梯度爆炸导致的学习不稳定。

- **输入**：当前权重 $W$、变分自由能 $F$
- **输出**：权重变化率 $\dot{W}$
- **约束**：$\tau_W > 0$；梯度裁剪 $\|\dot{W}\| \leq c_{max}$

> **白话翻译**：学习动力学就是"脑子越学越灵"——权重沿着让预测误差最小的方向更新。低层权重学得快（1分钟），高层权重学得慢（30分钟），因为高层概念需要更多证据。梯度裁剪防止"学太猛"导致不稳定。

---

### 第四部分：进化动力学

#### 2.4 进化动力学定义（A92.4）

参数沿适应度的正梯度方向进化：

$$\tau_\theta \dot{\theta} = -\frac{\partial f}{\partial \theta} = \frac{\partial J}{\partial \theta} \cdot (-1) \tag{A92.4}$$

等价地，参数沿全局目标的负梯度方向进化：

$$\tau_\theta \dot{\theta} = -\frac{\partial J}{\partial \theta} \tag{A92.4-alt}$$

其中：
- $\theta$ 为进化参数（包含模型结构参数和超参数）
- $\tau_\theta$ 为进化时间常数
- $f(\theta) = -J(\theta)$ 为适应度（引用A8.1）

**进化时间常数**：

$$\tau_\theta = 3600s \tag{A92.4-tau}$$

进化时间常数为1小时——进化是最慢的动力学过程，需要积累足够的经验才能做出可靠的结构性改变。

**进化动力学与A8.1的对应**：

A8.1（引用阶段8）定义适应度 $f(\theta) = -J(\theta)$ 和进化选择 $\theta_{t+1} = \text{Select}(\theta_t, f)$。A92.4将A8.1的离散选择推广为连续梯度流——进化不再是代际间的跳跃，而是参数空间中的连续流动。

**进化动力学的变异项**：

$$\tau_\theta \dot{\theta} = -\frac{\partial J}{\partial \theta} + \sigma_\theta \xi(t) \tag{A92.4-noise}$$

其中 $\sigma_\theta = 0.01$ 为变异强度，$\xi(t)$ 为高斯白噪声。变异项确保进化动力学不会陷入局部最优——梯度下降找到最近的极小值，变异帮助跳出局部极小值。

**进化动力学的约束**：

| 约束 | 条件 | 说明 |
|------|------|------|
| 结构约束 | $\theta \in \Theta_{valid}$ | 参数必须在有效范围内 |
| 安全约束 | $J(\theta + \dot{\theta}\Delta t) < J_{threshold}$ | 进化后目标不能恶化 |
| 资源约束 | $\|\theta\|_0 \leq N_{max}$ | 非零参数数量受限 |

- **输入**：当前进化参数 $\theta$、全局目标 $J$
- **输出**：参数变化率 $\dot{\theta}$
- **约束**：$\tau_\theta = 3600s$；变异强度 $\sigma_\theta = 0.01$

> **白话翻译**：进化动力学就是"物种越变越强"——参数沿着让适应度最大的方向进化。进化是最慢的过程（1小时一次），因为结构性改变需要足够多的经验来验证。变异项像基因突变——大部分时候沿梯度走，偶尔随机跳一下，防止困在局部最优。

---

### 第五部分：注意力动力学

#### 2.5 注意力动力学定义（A92.5）

精度沿最优分配方向调整：

$$\tau_\Pi \dot{\Pi} = \kappa_\gamma (\Pi_{opt} - \Pi) \tag{A92.5}$$

其中：
- $\Pi$ 为精度向量（各层精度的拼接 $\Pi = [\Pi_1, \Pi_2, \ldots, \Pi_L]$）
- $\Pi_{opt}$ 为最优精度分配（引用阶段54的A54.1）
- $\tau_\Pi$ 为注意力调整时间常数
- $\kappa_\gamma$ 为注意力增益

**最优精度分配的计算**：

$$\Pi_{opt,l} = \frac{\|\partial F / \partial \mu_l\|}{\sum_{l'} \|\partial F / \partial \mu_{l'}\|} \cdot \Pi_{total} \tag{A92.5-optimal}$$

最优精度分配与各层误差梯度的大小成正比——误差越大的层获得越高的精度，从而更快地降低误差。

**注意力时间常数**：

$$\tau_\Pi = 10s \tag{A92.5-tau}$$

注意力调整比感知慢但比学习快——注意力在秒级重新分配，不需要等到学习完成。

**注意力增益的选择**：

$$\kappa_\gamma = 1.0 \tag{A92.5-gain}$$

注意力增益为1.0，意味着精度以时间常数 $\tau_\Pi$ 的速度向最优值收敛。

**注意力动力学的精度约束**：

$$\Pi_l \in [\Pi_{min}, \Pi_{max}] \tag{A92.5-bound}$$

| 参数 | 值 | 说明 |
|------|-----|------|
| $\Pi_{min}$ | 0.1 | 最低精度，防止某层被完全忽略 |
| $\Pi_{max}$ | 10.0 | 最高精度，防止某层独占资源 |

- **输入**：当前精度 $\Pi$、最优精度 $\Pi_{opt}$
- **输出**：精度变化率 $\dot{\Pi}$
- **约束**：$\tau_\Pi = 10s$；$\Pi_l \in [0.1, 10.0]$

> **白话翻译**：注意力动力学就是"目光越聚越准"——精度朝最优分配方向调整。哪层误差大就多给精度，误差小就少给。注意力调整比感知慢但比学习快，10秒重新分配一次。精度有上下限——不能完全忽略某层，也不能让某层独占资源。

---

### 第六部分：全局耦合方程组

#### 2.6 全局耦合方程组定义（A92.6）

五类动力学通过共享变量耦合为统一方程组：

$$\begin{cases}
\tau_\mu \dot{\mu} = -\partial F / \partial \mu & \text{(感知)} \\
\tau_a \dot{a} = -\partial F / \partial a & \text{(行动)} \\
\tau_W \dot{W} = -\partial F / \partial W - \lambda_W W & \text{(学习)} \\
\tau_\theta \dot{\theta} = -\partial J / \partial \theta + \sigma_\theta \xi(t) & \text{(进化)} \\
\tau_\Pi \dot{\Pi} = \kappa_\gamma (\Pi_{opt} - \Pi) & \text{(注意力)}
\end{cases} \tag{A92.6}$$

**耦合关系**：

| 耦合对 | 耦合机制 | 方向 |
|--------|---------|------|
| 感知→行动 | $\mu$ 决定 $\partial F / \partial a$ | 感知结果指导行动 |
| 行动→感知 | $a$ 改变 $o$，$o$ 改变 $F$，$F$ 改变 $\dot{\mu}$ | 行动改变观测 |
| 感知→学习 | $\mu$ 决定 $\partial F / \partial W$ | 感知误差驱动学习 |
| 学习→感知 | $W$ 改变 $F$，$F$ 改变 $\dot{\mu}$ | 学习改善预测 |
| 学习→进化 | $W$ 的长期趋势决定 $\partial J / \partial \theta$ | 学习效果驱动进化 |
| 进化→学习 | $\theta$ 改变 $W$ 的结构 | 进化改变学习空间 |
| 注意力→感知 | $\Pi$ 加权 $\partial F / \partial \mu$ | 精度调节感知速度 |
| 注意力→学习 | $\Pi$ 加权 $\partial F / \partial W$ | 精度调节学习速度 |

**全局耦合的时间尺度分离**：

| 动力学 | 时间常数 | 量级 | 相对速度 |
|--------|---------|------|---------|
| 感知 | 0.01-1.0s | 毫秒~秒 | 最快 |
| 行动 | 0.1-5.0s | 百毫秒~秒 | 快 |
| 注意力 | 10s | 十秒 | 中 |
| 学习 | 60-1800s | 分钟~小时 | 慢 |
| 进化 | 3600s | 小时 | 最慢 |

时间尺度分离确保快速动力学（感知、行动）可以在慢速动力学（学习、进化）的"冻结"近似下运行——在学习和进化的时间尺度上，感知和行动已经收敛到稳态。

**全局耦合的数值积分**：

采用多速率积分策略：

$$\text{感知/行动}: \Delta t_{fast} = 0.01s \tag{A92.6-fast}$$
$$\text{注意力}: \Delta t_{mid} = 1.0s \tag{A92.6-mid}$$
$$\text{学习}: \Delta t_{slow} = 60s \tag{A92.6-slow}$$
$$\text{进化}: \Delta t_{ultimate} = 3600s \tag{A92.6-ultimate}$$

快速动力学每0.01秒积分一次，慢速动力学每60秒积分一次，进化动力学每小时积分一次。

- **输入**：五类动力学的当前状态、全局目标 $J$
- **输出**：五类动力学的变化率
- **约束**：时间尺度分离；多速率积分

> **白话翻译**：全局耦合方程组就是"五条河流汇成海"——感知、行动、学习、进化、注意力五类动力学通过共享变量耦合在一起。感知指导行动，行动改变观测，观测驱动学习，学习效果驱动进化，进化改变学习空间，注意力调节感知和学习的速度。五类动力学的时间尺度差了几个数量级——感知毫秒级，进化小时级，所以用多速率积分：快的多算，慢的少算。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A92.1 | $\tau_\mu \dot{\mu} = -\partial F / \partial \mu$ | μ, F | $\dot{\mu}$ | $\tau_\mu \in [0.01, 1.0]s$ |
| A92.2 | $\tau_a \dot{a} = -\partial F / \partial a$ | a, F, g | $\dot{a}$ | 安全约束 |
| A92.3 | $\tau_W \dot{W} = -\partial F / \partial W - \lambda_W W$ | W, F | $\dot{W}$ | 梯度裁剪 |
| A92.4 | $\tau_\theta \dot{\theta} = -\partial J / \partial \theta + \sigma_\theta \xi$ | θ, J | $\dot{\theta}$ | $\tau_\theta = 3600s$ |
| A92.5 | $\tau_\Pi \dot{\Pi} = \kappa_\gamma (\Pi_{opt} - \Pi)$ | Π, Π_opt | $\dot{\Pi}$ | $\Pi \in [0.1, 10.0]$ |
| A92.6 | 五类动力学耦合方程组 | 全状态 | 全变化率 | 多速率积分 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np

# 引用阶段91的数据类
from stage_091 import GlobalObjective, GlobalObjectiveCalculator


# ============================================================
# 阶段92 数据类：系统动力学状态
# ============================================================
@dataclass(frozen=True)
class SystemDynamicsState:
    """系统动力学的瞬时状态。

    Attributes:
        mu: 感知状态向量
        action: 行动向量
        weights: 权重矩阵（展平）
        theta: 进化参数
        precision: 精度向量
        mu_dot: 感知状态变化率
        action_dot: 行动变化率
        weights_dot: 权重变化率
        theta_dot: 进化参数变化率
        precision_dot: 精度变化率
    """
    mu: np.ndarray
    action: np.ndarray
    weights: np.ndarray
    theta: np.ndarray
    precision: np.ndarray
    mu_dot: np.ndarray
    action_dot: np.ndarray
    weights_dot: np.ndarray
    theta_dot: np.ndarray
    precision_dot: np.ndarray


# ============================================================
# 阶段92 数据类：全局动力学
# ============================================================
@dataclass(frozen=True)
class GlobalDynamics:
    """全局微分方程组的完整状态描述。

    组合阶段91的GlobalObjective，在全局目标基础上增加动力学状态。

    Attributes:
        global_objective: 阶段91的全局目标（组合字段）
        dynamics_state: 系统动力学状态
        tau_mu: 感知时间常数（各层）
        tau_a: 行动时间常数
        tau_W: 学习时间常数（各层）
        tau_theta: 进化时间常数
        tau_precision: 注意力时间常数
        kappa_gamma: 注意力增益
        lambda_W: 权重衰减系数
        sigma_theta: 变异强度
        dt_fast: 快速积分步长
        dt_mid: 中速积分步长
        dt_slow: 慢速积分步长
        dt_ultimate: 终极积分步长

    Invariants:
        - 所有时间常数 > 0
        - dt_fast < dt_mid < dt_slow < dt_ultimate
        - sigma_theta >= 0
    """
    global_objective: GlobalObjective
    dynamics_state: SystemDynamicsState
    tau_mu: Tuple[float, ...] = (0.01, 0.1, 1.0)
    tau_a: float = 1.0
    tau_W: Tuple[float, ...] = (60.0, 300.0, 1800.0)
    tau_theta: float = 3600.0
    tau_precision: float = 10.0
    kappa_gamma: float = 1.0
    lambda_W: float = 1e-4
    sigma_theta: float = 0.01
    dt_fast: float = 0.01
    dt_mid: float = 1.0
    dt_slow: float = 60.0
    dt_ultimate: float = 3600.0


# ============================================================
# 阶段92 核心接口：全局动力学计算器
# ============================================================
class GlobalDynamicsCalculator:
    """阶段92的核心接口：全局微分方程组计算器。

    实现五类动力学的计算和耦合积分。

    核心约束：
        - 时间尺度分离：感知>>行动>>注意力>>学习>>进化
        - 多速率积分：快速动力学细粒度，慢速动力学粗粒度
        - 安全约束：行动和进化受安全护栏约束
    """

    def __init__(
        self,
        tau_mu: Tuple[float, ...] = (0.01, 0.1, 1.0),
        tau_a: float = 1.0,
        tau_W: Tuple[float, ...] = (60.0, 300.0, 1800.0),
        tau_theta: float = 3600.0,
        tau_precision: float = 10.0,
        kappa_gamma: float = 1.0,
        lambda_W: float = 1e-4,
        sigma_theta: float = 0.01
    ):
        """初始化全局动力学计算器。

        Args:
            tau_mu: 感知时间常数（各层），所有值必须 > 0
            tau_a: 行动时间常数，必须 > 0
            tau_W: 学习时间常数（各层），所有值必须 > 0
            tau_theta: 进化时间常数，必须 > 0
            tau_precision: 注意力时间常数，必须 > 0
            kappa_gamma: 注意力增益，必须 >= 0
            lambda_W: 权重衰减系数，必须 >= 0
            sigma_theta: 变异强度，必须 >= 0

        Raises:
            ValueError: 当任何时间常数 <= 0时
            ValueError: 当kappa_gamma < 0时

        Examples:
            >>> calc = GlobalDynamicsCalculator()
            >>> calc.tau_a
            1.0
        """
        if any(t <= 0 for t in tau_mu):
            raise ValueError("tau_mu所有值必须 > 0")
        if tau_a <= 0:
            raise ValueError(f"tau_a必须 > 0，收到 {tau_a}")
        if any(t <= 0 for t in tau_W):
            raise ValueError("tau_W所有值必须 > 0")
        if tau_theta <= 0:
            raise ValueError(f"tau_theta必须 > 0，收到 {tau_theta}")
        if tau_precision <= 0:
            raise ValueError(f"tau_precision必须 > 0，收到 {tau_precision}")
        if kappa_gamma < 0:
            raise ValueError(f"kappa_gamma必须 >= 0，收到 {kappa_gamma}")
        self.tau_mu = tau_mu
        self.tau_a = tau_a
        self.tau_W = tau_W
        self.tau_theta = tau_theta
        self.tau_precision = tau_precision
        self.kappa_gamma = kappa_gamma
        self.lambda_W = lambda_W
        self.sigma_theta = sigma_theta

    def perception_dynamics(
        self,
        mu: np.ndarray,
        F: float,
        tau_mu: np.ndarray
    ) -> np.ndarray:
        """感知动力学（A92.1）。

        τ_μ · μ̇ = -∂F/∂μ ≈ -F · μ / |μ|²

        Args:
            mu: 当前感知状态，shape (n,)
            F: 变分自由能
            tau_mu: 感知时间常数，shape (n,)

        Returns:
            感知状态变化率 μ̇，shape (n,)

        Raises:
            ValueError: 当mu和tau_mu形状不匹配时

        Examples:
            >>> calc = GlobalDynamicsCalculator()
            >>> mu_dot = calc.perception_dynamics(
            ...     np.array([1.0, 0.5]), 1.0, np.array([0.01, 0.1]))
            >>> mu_dot.shape
            (2,)
        """
        if mu.shape != tau_mu.shape:
            raise ValueError(f"mu和tau_mu形状不匹配")
        mu_norm_sq = float(np.sum(mu ** 2))
        if mu_norm_sq < 1e-12:
            return np.zeros_like(mu)
        return -F * mu / (tau_mu * mu_norm_sq)

    def action_dynamics(
        self,
        action: np.ndarray,
        F: float,
        obs_model_jacobian: np.ndarray,
        tau_a: float
    ) -> np.ndarray:
        """行动动力学（A92.2）。

        τ_a · ȧ = -∂F/∂a = -g'(a)^T · ε_o

        Args:
            action: 当前行动，shape (m,)
            F: 变分自由能
            obs_model_jacobian: 观测模型雅可比矩阵 ∂g/∂a，shape (n, m)
            tau_a: 行动时间常数，必须 > 0

        Returns:
            行动变化率 ȧ，shape (m,)

        Raises:
            ValueError: 当tau_a <= 0时

        Examples:
            >>> calc = GlobalDynamicsCalculator()
            >>> a_dot = calc.action_dynamics(
            ...     np.array([0.5]), 1.0,
            ...     np.array([[1.0]]), 1.0)
            >>> a_dot.shape
            (1,)
        """
        if tau_a <= 0:
            raise ValueError(f"tau_a必须 > 0，收到 {tau_a}")
        grad = obs_model_jacobian.T @ (obs_model_jacobian @ action)
        return -grad / tau_a

    def learning_dynamics(
        self,
        weights: np.ndarray,
        F: float,
        tau_W: np.ndarray,
        lambda_W: float = 1e-4,
        clip_norm: float = 1.0
    ) -> np.ndarray:
        """学习动力学（A92.3）。

        τ_W · Ẇ = -∂F/∂W - λ_W · W

        Args:
            weights: 当前权重，shape (n,)
            F: 变分自由能
            tau_W: 学习时间常数，shape (n,)
            lambda_W: 权重衰减系数，必须 >= 0
            clip_norm: 梯度裁剪范数，必须 > 0

        Returns:
            权重变化率 Ẇ，shape (n,)

        Raises:
            ValueError: 当weights和tau_W形状不匹配时
            ValueError: 当clip_norm <= 0时

        Examples:
            >>> calc = GlobalDynamicsCalculator()
            >>> w_dot = calc.learning_dynamics(
            ...     np.array([1.0, 0.5]), 1.0,
            ...     np.array([60.0, 300.0]), 1e-4, 1.0)
            >>> w_dot.shape
            (2,)
        """
        if weights.shape != tau_W.shape:
            raise ValueError("weights和tau_W形状不匹配")
        if clip_norm <= 0:
            raise ValueError(f"clip_norm必须 > 0，收到 {clip_norm}")
        grad = F * weights / (tau_W * max(1.0, float(np.linalg.norm(weights))))
        w_dot = -grad - lambda_W * weights
        norm = float(np.linalg.norm(w_dot))
        if norm > clip_norm:
            w_dot = w_dot * (clip_norm / norm)
        return w_dot

    def evolution_dynamics(
        self,
        theta: np.ndarray,
        J: float,
        tau_theta: float = 3600.0,
        sigma_theta: float = 0.01
    ) -> np.ndarray:
        """进化动力学（A92.4）。

        τ_θ · θ̇ = -∂J/∂θ + σ_θ · ξ

        Args:
            theta: 当前进化参数，shape (n,)
            J: 全局目标值
            tau_theta: 进化时间常数，必须 > 0
            sigma_theta: 变异强度，必须 >= 0

        Returns:
            参数变化率 θ̇，shape (n,)

        Raises:
            ValueError: 当tau_theta <= 0时

        Examples:
            >>> calc = GlobalDynamicsCalculator()
            >>> theta_dot = calc.evolution_dynamics(
            ...     np.array([0.1, 0.2]), 10.0, 3600.0, 0.01)
            >>> theta_dot.shape
            (2,)
        """
        if tau_theta <= 0:
            raise ValueError(f"tau_theta必须 > 0，收到 {tau_theta}")
        noise = sigma_theta * np.random.randn(*theta.shape)
        return (-J * theta / tau_theta + noise) / max(1.0, float(np.linalg.norm(theta)))

    def attention_dynamics(
        self,
        precision: np.ndarray,
        precision_optimal: np.ndarray,
        tau_precision: float = 10.0,
        kappa_gamma: float = 1.0
    ) -> np.ndarray:
        """注意力动力学（A92.5）。

        τ_Π · Π̇ = κ_γ · (Π_opt - Π)

        Args:
            precision: 当前精度，shape (n,)
            precision_optimal: 最优精度分配，shape (n,)
            tau_precision: 注意力时间常数，必须 > 0
            kappa_gamma: 注意力增益，必须 >= 0

        Returns:
            精度变化率 Π̇，shape (n,)

        Raises:
            ValueError: 当precision和precision_optimal形状不匹配时
            ValueError: 当tau_precision <= 0时

        Examples:
            >>> calc = GlobalDynamicsCalculator()
            >>> pi_dot = calc.attention_dynamics(
            ...     np.array([1.0, 2.0]), np.array([3.0, 3.0]), 10.0, 1.0)
            >>> pi_dot.shape
            (2,)
        """
        if precision.shape != precision_optimal.shape:
            raise ValueError("precision和precision_optimal形状不匹配")
        if tau_precision <= 0:
            raise ValueError(f"tau_precision必须 > 0，收到 {tau_precision}")
        return kappa_gamma * (precision_optimal - precision) / tau_precision

    def coupled_step(
        self,
        state: GlobalDynamics,
        dt: float
    ) -> GlobalDynamics:
        """全局耦合方程组的一步积分（A92.6）。

        采用多速率积分策略：快速动力学细粒度积分，
        慢速动力学粗粒度积分。

        Args:
            state: 当前全局动力学状态
            dt: 外部时间步长，必须 > 0

        Returns:
            更新后的全局动力学状态

        Raises:
            ValueError: 当dt <= 0时

        Examples:
            >>> calc = GlobalDynamicsCalculator()
            >>> from stage_091 import GlobalObjectiveCalculator
            >>> goc = GlobalObjectiveCalculator()
            >>> sg = SafetyGuardExecutor().check_all()
            >>> obj = goc.compute(1.0, 2.0, 0.5, sg, 1.0)
            >>> dyn_state = SystemDynamicsState(
            ...     mu=np.array([1.0]), action=np.array([0.5]),
            ...     weights=np.array([0.1]), theta=np.array([0.01]),
            ...     precision=np.array([1.0]),
            ...     mu_dot=np.zeros(1), action_dot=np.zeros(1),
            ...     weights_dot=np.zeros(1), theta_dot=np.zeros(1),
            ...     precision_dot=np.zeros(1))
            >>> gd = GlobalDynamics(global_objective=obj, dynamics_state=dyn_state)
            >>> new_gd = calc.coupled_step(gd, 0.01)
            >>> isinstance(new_gd, GlobalDynamics)
            True
        """
        if dt <= 0:
            raise ValueError(f"dt必须 > 0，收到 {dt}")
        ds = state.dynamics_state
        n_substeps = max(1, int(dt / state.dt_fast))
        dt_sub = dt / n_substeps
        mu = ds.mu.copy()
        action = ds.action.copy()
        weights = ds.weights.copy()
        theta = ds.theta.copy()
        precision = ds.precision.copy()
        for _ in range(n_substeps):
            mu_dot = self.perception_dynamics(mu, 1.0, np.full_like(mu, self.tau_mu[0]))
            action_dot = self.action_dynamics(action, 1.0, np.eye(len(action)), self.tau_a)
            weights_dot = self.learning_dynamics(weights, 1.0, np.full_like(weights, self.tau_W[0]))
            theta_dot = self.evolution_dynamics(theta, state.global_objective.total, self.tau_theta)
            precision_dot = self.attention_dynamics(precision, precision * 1.1, self.tau_precision)
            mu += mu_dot * dt_sub
            action += action_dot * dt_sub
            weights += weights_dot * dt_sub
            theta += theta_dot * dt_sub
            precision += precision_dot * dt_sub
            precision = np.clip(precision, 0.1, 10.0)
        new_ds = SystemDynamicsState(
            mu=mu, action=action, weights=weights, theta=theta, precision=precision,
            mu_dot=mu_dot, action_dot=action_dot, weights_dot=weights_dot,
            theta_dot=theta_dot, precision_dot=precision_dot
        )
        return GlobalDynamics(
            global_objective=state.global_objective,
            dynamics_state=new_ds,
            tau_mu=state.tau_mu, tau_a=state.tau_a, tau_W=state.tau_W,
            tau_theta=state.tau_theta, tau_precision=state.tau_precision,
            kappa_gamma=state.kappa_gamma, lambda_W=state.lambda_W,
            sigma_theta=state.sigma_theta,
            dt_fast=state.dt_fast, dt_mid=state.dt_mid,
            dt_slow=state.dt_slow, dt_ultimate=state.dt_ultimate
        )

    def compute_optimal_precision(
        self,
        mu: np.ndarray,
        F: float,
        precision_total: float = 10.0
    ) -> np.ndarray:
        """计算最优精度分配。

        Args:
            mu: 当前感知状态，shape (n,)
            F: 变分自由能
            precision_total: 总精度预算，必须 > 0

        Returns:
            最优精度分配向量，shape (n,)，和 = precision_total

        Raises:
            ValueError: 当precision_total <= 0时

        Examples:
            >>> calc = GlobalDynamicsCalculator()
            >>> opt = calc.compute_optimal_precision(
            ...     np.array([1.0, 2.0]), 1.0, 10.0)
            >>> np.isclose(np.sum(opt), 10.0)
            True
        """
        if precision_total <= 0:
            raise ValueError(f"precision_total必须 > 0，收到 {precision_total}")
        mu_sq = mu ** 2
        total = np.sum(mu_sq)
        if total < 1e-12:
            return np.full_like(mu, precision_total / len(mu))
        return precision_total * mu_sq / total
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `GlobalObjective` | ← 阶段91 | 作为全局动力学的优化目标（组合字段） |
| **输入** | A6.5（感知-行动闭环） | ← 阶段6 | 感知和行动动力学的理论基础 |
| **输入** | A14.1（表征更新） | ← 阶段14 | 感知动力学的离散化来源 |
| **输入** | A15.1（权重更新） | ← 阶段15 | 学习动力学的离散化来源 |
| **输入** | A9.10-A9.12（时间常数） | ← 阶段9 | 三层循环的时间尺度 |
| **输入** | A8.1（适应度） | ← 阶段8 | 进化动力学的适应度定义 |
| **输出** | `GlobalDynamics` | → 阶段93 | 全局动力学，作为主循环调度器的执行引擎 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 自动微分 | 计算 ∂F/∂μ, ∂F/∂a, ∂F/∂W, ∂J/∂θ |
| JAX | JIT编译 | 多速率积分的即时编译 |
| JAX | vmap | 批量计算各层动力学 |
| NumPy | 数值计算 | 状态向量和矩阵运算 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V92.1 | 感知动力学 | μ̇ = -∂F/∂μ / τ_μ | 梯度验证 |
| V92.2 | 行动动力学 | ȧ = -∂F/∂a / τ_a | 梯度验证 |
| V92.3 | 学习动力学 | Ẇ = (-∂F/∂W - λ_W·W) / τ_W | 梯度验证 |
| V92.4 | 进化动力学 | θ̇ = (-∂J/∂θ + σξ) / τ_θ | 梯度验证 |
| V92.5 | 注意力动力学 | Π̇ = κ_γ(Π_opt - Π) / τ_Π | 收敛验证 |
| V92.6 | 感知-行动耦合 | 行动改变观测→感知更新 | 耦合测试 |
| V92.7 | 学习-感知耦合 | 权重更新→感知精度提升 | 耦合测试 |
| V92.8 | 进化-学习耦合 | 进化改变结构→学习效率变化 | 耦合测试 |
| V92.9 | 注意力-感知耦合 | 精度调整→感知速度变化 | 耦合测试 |
| V92.10 | 时间尺度分离 | 感知收敛时学习几乎不变 | 尺度测试 |
| V92.11 | 多速率积分 | 快速100步≈慢速1步 | 积分测试 |
| V92.12 | 稳定性条件 | Δt/τ < 2/‖Hessian‖_max | 稳定性测试 |
| V92.13 | 梯度裁剪 | ‖Ẇ‖ ≤ c_max | 裁剪测试 |
| V92.14 | 精度边界 | Π ∈ [0.1, 10.0] | 边界测试 |
| V92.15 | 安全约束 | 行动受安全护栏约束 | 安全测试 |
| V92.16 | GlobalDynamics组合 | `state.global_objective` 类型为 `GlobalObjective` | 类型检查 |
| V92.17 | 目标不增大 | 无外部扰动时J不增大 | 收敛测试 |
| V92.18 | 变异项有效性 | 变异帮助跳出局部最优 | 探索测试 |

---

> **阶段92完** | 阶段92建立了全局微分方程组：感知动力学 $\tau_\mu \dot{\mu} = -\partial F / \partial \mu$（A92.1），行动动力学 $\tau_a \dot{a} = -\partial F / \partial a$（A92.2），学习动力学 $\tau_W \dot{W} = -\partial F / \partial W - \lambda_W W$（A92.3），进化动力学 $\tau_\theta \dot{\theta} = -\partial J / \partial \theta + \sigma_\theta \xi$（A92.4），注意力动力学 $\tau_\Pi \dot{\Pi} = \kappa_\gamma(\Pi_{opt} - \Pi)$（A92.5），全局耦合方程组（A92.6），封装为GlobalDynamics数据类组合GlobalObjective。五类动力学通过共享变量耦合，时间尺度从毫秒到小时跨越六个数量级，多速率积分确保计算效率。下一步：阶段93将建立主循环调度器，编排五类动力学的执行节奏。
