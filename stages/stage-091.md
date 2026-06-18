# 阶段91：全局统一目标函数

> **阶段编号**：S91 | **所属层**：第十层·系统集成层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 全局目标函数 J | "终极KPI"——衡量系统整体表现的唯一指标 |
> | 精度进化项 ∫F dt | "越猜越准"——长期预测误差持续下降 |
> | 广度进化项 ∫G dt | "越懂越多"——认知覆盖范围持续扩张 |
> | 能效进化项 λ₁R_complexity | "越算越省"——算力消耗持续降低 |
> | 安全进化项 λ₂R_safety | "越走越稳"——风险持续消除 |
> | 定理91.1 | "KPI与使命等价"——优化J等价于实现终极进化目标 |
> | GlobalObjective | "全局目标的完整画像"——组合SafetyGuardState，在安全护栏基础上增加全局目标状态 |

---

## 1. 第一性原理依据

本阶段从阶段4（变分自由能）、阶段7（预期自由能）、阶段24（全局奖励）、阶段8（适应度）的核心结论出发，建立全局统一目标函数。核心依据为：

- **A4.4（变分自由能）**（引用阶段4）：变分自由能 $F(\theta, o) = D_{KL}[q(s) \| p(s|o)] + E_q[-\ln p(o|s)]$ 是感知精度的度量。$F$ 越小，模型对观测的解释越精确。全局目标函数的精度进化项直接引用A4.4的变分自由能定义。

- **A7.1（预期自由能）**（引用阶段7）：预期自由能 $G(\pi) = E_{q(o|π)}[D_{KL}[q(s|o,π) \| q(s|π)] - \ln p(o)]$ 是决策质量的度量。$G$ 越小，策略的信息增益越大、偏好满足度越高。全局目标函数的广度进化项直接引用A7.1的预期自由能定义。

- **A24.1（全局奖励）**（引用阶段24）：全局奖励 $R_{total} = R_{accuracy} + R_{coverage} + R_{efficiency}$ 将三大驱力的收益统一为标量。全局目标函数与A24.1的对应关系为 $J \leftrightarrow -\int R_{total} \, dt$——最小化J等价于最大化累积奖励。

- **A8.1（适应度）**（引用阶段8）：适应度 $f(\theta) = -J(\theta)$ 将全局目标与进化选择压力关联。适应度越高（J越低），参数集在进化中的存活概率越大。全局目标函数的进化动力学直接引用A8.1的适应度定义。

本阶段的核心目标是：**定义全局统一目标函数 $J = \int_0^T F(\theta_t, o_t) \, dt + \int_0^T G(\pi_t) \, dt + \lambda_1 R_{complexity}(\theta) + \lambda_2 R_{safety}(o)$（A91.1），精度进化项 $\int F \, dt$ → 长期预测误差下降（A91.2），广度进化项 $\int G \, dt$ → 认知覆盖扩张（A91.3），能效进化项 $\lambda_1 R_{complexity}$ → 算力消耗降低（A91.4），安全进化项 $\lambda_2 R_{safety}$ → 风险消除（A91.5），证明全局目标函数与终极进化目标的等价性（定理91.1），封装为GlobalObjective数据类组合SafetyGuardState。**

> **去重声明**：本阶段仅定义全局统一目标函数及其与终极进化目标的等价性证明，不重新定义变分自由能（阶段4的A4.4）、预期自由能（阶段7的A7.1）、全局奖励（阶段24的A24.1）或适应度（阶段8的A8.1）。所有子项均通过引用接入。

---

## 2. 核心数学公式

### 第一部分：全局统一目标函数

#### 2.1 全局统一目标函数定义（A91.1）

全局统一目标函数 $J$ 是系统整体表现的唯一度量，整合精度、广度、能效、安全四个进化维度：

$$J = \int_0^T F(\theta_t, o_t) \, dt + \int_0^T G(\pi_t) \, dt + \lambda_1 R_{complexity}(\theta) + \lambda_2 R_{safety}(o) \tag{A91.1}$$

其中：
- $F(\theta_t, o_t)$ 为变分自由能（引用A4.4），度量感知精度
- $G(\pi_t)$ 为预期自由能（引用A7.1），度量决策质量
- $R_{complexity}(\theta) = D_{KL}[q_\theta(s) \| p(s)]$ 为模型复杂度正则项（引用阶段5的A5.2复杂性项）
- $R_{safety}(o) = \sum_i \max(0, o_i - o_i^{limit})$ 为安全违规惩罚项
- $\lambda_1 = 0.01$ 为复杂度正则系数
- $\lambda_2 = 1.0$ 为安全正则系数

**全局目标函数的四个进化维度**：

| 维度 | 数学项 | 物理含义 | 进化方向 |
|------|--------|---------|---------|
| 精度 | $\int_0^T F \, dt$ | 累积感知误差 | ↓ 下降 |
| 广度 | $\int_0^T G \, dt$ | 累积决策不确定性 | ↓ 下降 |
| 能效 | $\lambda_1 R_{complexity}$ | 模型复杂度 | ↓ 下降 |
| 安全 | $\lambda_2 R_{safety}$ | 安全违规程度 | ↓ 下降 |

**全局目标函数的离散化形式**：

$$J = \sum_{t=0}^{T-1} \left[ F(\theta_t, o_t) + G(\pi_t) \right] \Delta t + \lambda_1 R_{complexity}(\theta_T) + \lambda_2 R_{safety}(o_T) \tag{A91.1-discrete}$$

其中 $\Delta t$ 为时间步长。积分项离散化为求和，正则项取终态值。

**全局目标函数的单步增量**：

$$\Delta J_t = F(\theta_t, o_t) \Delta t + G(\pi_t) \Delta t \tag{A91.1-increment}$$

单步增量仅包含积分项的增量，正则项在终态计算。系统运行时追踪 $\Delta J_t$ 的累积值。

**正则系数的选择依据**：

| 系数 | 默认值 | 选择依据 |
|------|--------|---------|
| $\lambda_1$ | 0.01 | 复杂度惩罚应远小于感知误差，避免过度压缩模型 |
| $\lambda_2$ | 1.0 | 安全违规应与感知误差同量级，确保安全约束有效 |

- **输入**：变分自由能 $F$、预期自由能 $G$、模型参数 $\theta$、观测 $o$
- **输出**：全局目标值 $J$
- **约束**：$J \geq 0$（各项均非负）；$\lambda_2 > \lambda_1$（安全优先于能效）

> **白话翻译**：全局目标函数J就是"终极KPI"——把系统的四个进化维度打包成一个数：精度（预测误差）、广度（决策不确定性）、能效（模型复杂度）、安全（违规程度）。J越小系统越好，J持续下降说明系统在进化。安全系数比能效系数大100倍，因为安全比省算力重要得多。

---

### 第二部分：精度进化项

#### 2.2 精度进化项定义（A91.2）

精度进化项度量系统在时间窗口内的累积感知误差：

$$J_{accuracy} = \int_0^T F(\theta_t, o_t) \, dt \tag{A91.2}$$

**精度进化的含义**：

$J_{accuracy}$ 随时间下降意味着系统的预测越来越精确——模型对环境的理解持续深化。这是"精度进化"的核心：不是单步误差的偶然下降，而是长期趋势的持续改善。

**精度进化与A4.4的对应**：

A4.4（引用阶段4）定义变分自由能 $F = D_{KL}[q(s) \| p(s|o)] + E_q[-\ln p(o|s)]$。A91.2将A4.4的瞬时量扩展为时间积分——单步自由能低是"猜对了"，积分自由能持续下降才是"越猜越准"。

**精度进化的长期趋势判定**：

$$\text{accuracy\_evolving} = \mathbb{1}\left[\frac{d}{dT} \int_0^T F \, dt < 0\right] = \mathbb{1}[F(T) < \bar{F}_{baseline}] \tag{A91.2-trend}$$

精度进化等价于当前自由能低于基线平均值。基线取初始100步的滑动平均。

**精度进化的收敛速度**：

$$\tau_{accuracy} = -\frac{F_0}{\dot{F}} \tag{A91.2-tau}$$

$\tau_{accuracy}$ 为精度进化的时间常数——自由能从初始值下降到0所需的时间估计。$\tau_{accuracy}$ 越小，精度进化越快。

- **输入**：变分自由能时间序列 $\{F_t\}_{t=0}^T$
- **输出**：精度进化项 $J_{accuracy}$、精度进化标志、收敛速度
- **约束**：$J_{accuracy} \geq 0$；长期趋势 $\dot{J}_{accuracy} < 0$

> **白话翻译**：精度进化项就是"越猜越准"——把所有时刻的预测误差加起来，看总和是不是在下降。单次猜对不算进化，长期趋势下降才是。收敛速度告诉我们"还要多久能猜准"。

---

### 第三部分：广度进化项

#### 2.3 广度进化项定义（A91.3）

广度进化项度量系统在时间窗口内的累积决策不确定性：

$$J_{coverage} = \int_0^T G(\pi_t) \, dt \tag{A91.3}$$

**广度进化的含义**：

$J_{coverage}$ 随时间下降意味着系统的决策越来越有效——策略覆盖的认知空间持续扩张。预期自由能 $G$ 包含认知价值项（信息增益）和实用价值项（偏好满足），$G$ 下降意味着系统在探索新知和满足偏好之间找到了更优平衡。

**广度进化与A7.1的对应**：

A7.1（引用阶段7）定义预期自由能 $G(\pi) = -E_{q(o|π)}[\ln p(o) - D_{KL}[q(s|o,π) \| q(s|π)]]$。A91.3将A7.1的瞬时量扩展为时间积分——单步预期自由能低是"选对了"，积分预期自由能持续下降才是"越懂越多"。

**广度进化的认知覆盖度量**：

$$\text{coverage}(T) = \frac{|\Psi_{known}(T)|}{|\Psi_{total}|} \tag{A91.3-coverage}$$

其中 $\Psi_{known}(T)$ 为截至时刻 $T$ 系统已探索的认知空间子集，$\Psi_{total}$ 为全部认知空间。$J_{coverage}$ 下降等价于 $\text{coverage}(T)$ 上升——累积决策不确定性降低意味着认知覆盖扩张。

**广度进化与精度的协同**：

| 场景 | $J_{accuracy}$ | $J_{coverage}$ | 系统状态 |
|------|----------------|----------------|---------|
| 新环境 | 高（预测不准） | 高（决策迷茫） | 适应期 |
| 适应中 | 下降 | 下降 | 进化期 |
| 稳态 | 低 | 低 | 成熟期 |
| 环境变化 | 上升 | 上升 | 重适应期 |

精度和广度同向变化——预测越准，决策越有效；反之亦然。

- **输入**：预期自由能时间序列 $\{G_t\}_{t=0}^T$
- **输出**：广度进化项 $J_{coverage}$、认知覆盖率、广度进化标志
- **约束**：$J_{coverage} \geq 0$；长期趋势 $\dot{J}_{coverage} < 0$

> **白话翻译**：广度进化项就是"越懂越多"——把所有时刻的决策不确定性加起来，看总和是不是在下降。预期自由能下降意味着系统在探索和利用之间找到了更好的平衡，认知覆盖范围在扩张。

---

### 第四部分：能效进化项

#### 2.4 能效进化项定义（A91.4）

能效进化项度量模型复杂度，驱动系统用更少的参数实现同等精度：

$$J_{efficiency} = \lambda_1 R_{complexity}(\theta) = \lambda_1 D_{KL}[q_\theta(s) \| p(s)] \tag{A91.4}$$

**能效进化的含义**：

$J_{efficiency}$ 随时间下降意味着模型的复杂度在降低——系统用更简洁的内部表征实现同等甚至更优的预测精度。这是奥卡姆剃刀的数学实现：在同等预测能力下，优先选择更简单的模型。

**能效进化与A5.2的对应**：

A5.2（引用阶段5）定义自由能分解 $F = Accuracy + Complexity$，其中 $Complexity = D_{KL}[q(s) \| p(s)]$。A91.4直接引用A5.2的复杂性项，将其从自由能的组成部分提升为独立的进化目标。

**能效进化的计算度量**：

$$\text{compute\_efficiency}(T) = \frac{\text{effective\_compute}(T)}{\text{total\_compute}(T)} \tag{A91.4-compute}$$

其中 $\text{effective\_compute}$ 为产生有效预测（误差低于阈值）的计算量，$\text{total\_compute}$ 为总计算量。$J_{efficiency}$ 下降等价于 $\text{compute\_efficiency}$ 上升。

**能效进化的参数效率**：

$$\text{param\_efficiency} = \frac{\text{prediction\_accuracy}}{\|\theta\|_0} \tag{A91.4-param}$$

其中 $\|\theta\|_0$ 为非零参数数量。参数效率越高，模型越精简。

- **输入**：模型参数 $\theta$、先验 $p(s)$
- **输出**：能效进化项 $J_{efficiency}$、计算效率、参数效率
- **约束**：$J_{efficiency} \geq 0$；$\lambda_1 = 0.01$（远小于安全系数）

> **白话翻译**：能效进化项就是"越算越省"——用更少的参数实现同样的预测精度。这是奥卡姆剃刀：同等能力下选更简单的模型。复杂度惩罚系数只有0.01，因为省算力不如保安全重要，但也不能完全不管——模型太复杂会浪费资源。

---

### 第五部分：安全进化项

#### 2.5 安全进化项定义（A91.5）

安全进化项度量安全违规程度，驱动系统消除风险：

$$J_{safety} = \lambda_2 R_{safety}(o) = \lambda_2 \sum_i \max(0, o_i - o_i^{limit}) \tag{A91.5}$$

**安全进化的含义**：

$J_{safety}$ 随时间下降意味着系统的安全违规在减少——资源使用更规范、行为更安全、风险更低。安全进化项是全局目标的硬约束——$\lambda_2 = 1.0$ 确保安全违规的惩罚与感知误差同量级。

**安全违规项的具体定义**：

$$R_{safety}(o) = \max(0, u_{cpu} - 0.20) + \max(0, u_{mem} - 0.20) + \max(0, u_{disk} - 0.80) + \mathbb{1}[\text{file\_violation}] + \mathbb{1}[\text{regression}] \tag{A91.5-detail}$$

| 违规项 | 阈值 | 引用 |
|--------|------|------|
| CPU超限 | $u_{cpu} > 0.20$ | A81.2 |
| 内存超限 | $u_{mem} > 0.20$ | A82.2 |
| 磁盘超限 | $u_{disk} > 0.80$ | A83.2 |
| 文件违规 | 修改受保护文件 | A84.2 |
| 性能倒退 | 重构后性能下降>5% | A86.2 |

**安全进化与SafetyGuardState的对应**：

A90（引用阶段90）定义SafetyGuardState，包含九个安全子系统的状态。A91.5的安全违规项直接从SafetyGuardState提取——$R_{safety}$ 是SafetyGuardState的标量摘要。

**安全进化的硬约束特性**：

安全项使用 $\max(0, \cdot)$ 形式，意味着阈值内的行为不产生惩罚，超出阈值则线性惩罚。这种"软约束+硬阈值"的设计既允许系统在安全边界内自由运行，又确保越界行为受到足够惩罚。

- **输入**：SafetyGuardState（引用阶段90）
- **输出**：安全进化项 $J_{safety}$、安全违规详情
- **约束**：$J_{safety} \geq 0$；$\lambda_2 = 1.0$（与感知误差同量级）

> **白话翻译**：安全进化项就是"越走越稳"——CPU超20%要罚、内存超20%要罚、磁盘超80%要罚、改关键文件要罚、性能倒退要罚。安全系数是1.0，跟感知误差一样重——安全不是可选项，是硬指标。

---

### 第六部分：等价性定理

#### 2.6 定理91.1：全局目标函数与终极进化目标的等价性

**定理陈述**：

最小化全局目标函数 $J$ 等价于实现系统的终极进化目标——在资源约束下持续提升适应度。

$$\min_\theta J(\theta) \iff \max_\theta f(\theta) \tag{T91.1}$$

其中 $f(\theta) = -J(\theta)$ 为适应度（引用A8.1）。

**证明**：

**步骤1**：建立 $J$ 与 $f$ 的对偶关系。

由A8.1（引用阶段8），适应度定义为 $f(\theta) = -J(\theta)$。因此：

$$\min_\theta J(\theta) \iff \max_\theta f(\theta) \tag{T91.1-step1}$$

这是定义层面的直接对偶，无需额外证明。

**步骤2**：证明 $J$ 的各项与进化维度的一一对应。

| $J$ 的项 | 进化维度 | 对应关系 |
|----------|---------|---------|
| $\int F \, dt$ | 精度进化 | $F \downarrow \implies$ 预测更准 $\implies$ 适应度↑ |
| $\int G \, dt$ | 广度进化 | $G \downarrow \implies$ 决策更优 $\implies$ 适应度↑ |
| $\lambda_1 R_{complexity}$ | 能效进化 | $R_{complexity} \downarrow \implies$ 模型更简 $\implies$ 适应度↑ |
| $\lambda_2 R_{safety}$ | 安全进化 | $R_{safety} \downarrow \implies$ 风险更低 $\implies$ 适应度↑ |

$J$ 的每一项下降都对应适应度的一个维度提升，因此 $J$ 的整体下降等价于适应度的整体提升。

**步骤3**：证明 $J$ 与A24.1全局奖励的等价。

由A24.1（引用阶段24），全局奖励 $R_{total} = R_{accuracy} + R_{coverage} + R_{efficiency}$。$J$ 与 $R_{total}$ 的对应关系为：

$$J = -\int_0^T R_{total} \, dt + \lambda_1 R_{complexity} + \lambda_2 R_{safety} \tag{T91.1-reward}$$

其中 $R_{accuracy} = -F$，$R_{coverage} = -G$，$R_{efficiency} = -R_{complexity}$。因此：

$$\min J \iff \max \int R_{total} \, dt - \lambda_1 R_{complexity} - \lambda_2 R_{safety} \tag{T91.1-equiv}$$

在安全约束满足时（$R_{safety} = 0$），最小化 $J$ 等价于最大化累积奖励减去复杂度惩罚——这正是强化学习中带正则化的奖励最大化目标。

**步骤4**：证明资源约束下的等价性。

在资源约束下（CPU ≤ 20%，内存 ≤ 20%），安全项 $R_{safety} = 0$，此时：

$$\min J = \min \left[\int F \, dt + \int G \, dt + \lambda_1 R_{complexity}\right] \tag{T91.1-constrained}$$

这等价于在资源约束下最大化适应度——系统用最少的资源实现最优的预测和决策。

**结论**：$\min_\theta J(\theta) \iff \max_\theta f(\theta)$，全局目标函数与终极进化目标等价。$\blacksquare$

**定理的工程意义**：

定理91.1确保了全局目标函数的优化方向与系统的终极进化目标完全一致——工程师只需最小化 $J$，系统就会自动朝着"更准、更广、更省、更稳"的方向进化，无需手动平衡各维度。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A91.1 | $J = \int F \, dt + \int G \, dt + \lambda_1 R_{complexity} + \lambda_2 R_{safety}$ | F, G, θ, o | 标量 J | J ≥ 0, λ₂ > λ₁ |
| A91.2 | $J_{accuracy} = \int_0^T F \, dt$ | F时间序列 | 精度进化项 | 长期↓ |
| A91.3 | $J_{coverage} = \int_0^T G \, dt$ | G时间序列 | 广度进化项 | 长期↓ |
| A91.4 | $J_{efficiency} = \lambda_1 D_{KL}[q_\theta \| p]$ | θ, p(s) | 能效进化项 | λ₁ = 0.01 |
| A91.5 | $J_{safety} = \lambda_2 \sum \max(0, o_i - o_i^{limit})$ | SafetyGuardState | 安全进化项 | λ₂ = 1.0 |
| T91.1 | $\min J \iff \max f$ | J, f | 等价性 | 证明见2.6节 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import numpy as np

# 引用阶段90的数据类
from stage_090 import SafetyGuardState, SafetyCheckResult


# ============================================================
# 阶段91 枚举：进化维度
# ============================================================
class EvolutionDimension(Enum):
    """全局目标的四个进化维度。"""
    ACCURACY = "accuracy"      # 精度进化
    COVERAGE = "coverage"      # 广度进化
    EFFICIENCY = "efficiency"  # 能效进化
    SAFETY = "safety"          # 安全进化


# ============================================================
# 阶段91 数据类：全局目标
# ============================================================
@dataclass(frozen=True)
class GlobalObjective:
    """全局统一目标函数的完整状态描述。

    组合阶段90的SafetyGuardState，在安全护栏基础上增加全局目标状态。

    Attributes:
        safety_guard_state: 阶段90的安全护栏状态（组合字段）
        cumulative_free_energy: 累积变分自由能 ∫F dt
        cumulative_expected_free_energy: 累积预期自由能 ∫G dt
        complexity_penalty: 复杂度正则项 λ₁R_complexity
        safety_penalty: 安全违规惩罚项 λ₂R_safety
        total: 全局目标值 J
        accuracy_trend: 精度进化趋势（下降=True）
        coverage_trend: 广度进化趋势（下降=True）
        efficiency_trend: 能效进化趋势（下降=True）
        safety_trend: 安全进化趋势（下降=True）
        lambda_complexity: 复杂度正则系数
        lambda_safety: 安全正则系数
        compute_efficiency: 计算效率
        param_efficiency: 参数效率
        coverage_ratio: 认知覆盖率

    Invariants:
        - total >= 0
        - cumulative_free_energy >= 0
        - cumulative_expected_free_energy >= 0
        - complexity_penalty >= 0
        - safety_penalty >= 0
        - lambda_safety > lambda_complexity
    """
    safety_guard_state: SafetyGuardState
    cumulative_free_energy: float = 0.0
    cumulative_expected_free_energy: float = 0.0
    complexity_penalty: float = 0.0
    safety_penalty: float = 0.0
    total: float = 0.0
    accuracy_trend: bool = True
    coverage_trend: bool = True
    efficiency_trend: bool = True
    safety_trend: bool = True
    lambda_complexity: float = 0.01
    lambda_safety: float = 1.0
    compute_efficiency: float = 0.0
    param_efficiency: float = 0.0
    coverage_ratio: float = 0.0


# ============================================================
# 阶段91 核心接口：全局目标计算器
# ============================================================
class GlobalObjectiveCalculator:
    """阶段91的核心接口：全局统一目标函数计算器。

    实现全局目标J的计算、各项分解和进化趋势判定。

    核心约束：
        - J >= 0
        - λ_safety > λ_complexity
        - 安全约束满足时 R_safety = 0
    """

    def __init__(
        self,
        lambda_complexity: float = 0.01,
        lambda_safety: float = 1.0,
        trend_window: int = 1000
    ):
        """初始化全局目标计算器（A91.1）。

        Args:
            lambda_complexity: 复杂度正则系数（默认0.01），必须 >= 0
            lambda_safety: 安全正则系数（默认1.0），必须 > lambda_complexity
            trend_window: 趋势判定窗口（默认1000步），必须 > 0

        Raises:
            ValueError: 当lambda_safety <= lambda_complexity时
            ValueError: 当trend_window <= 0时

        Examples:
            >>> calc = GlobalObjectiveCalculator()
            >>> calc.lambda_safety
            1.0
        """
        if lambda_safety <= lambda_complexity:
            raise ValueError(
                f"lambda_safety({lambda_safety})必须 > "
                f"lambda_complexity({lambda_complexity})"
            )
        if trend_window <= 0:
            raise ValueError(f"trend_window必须 > 0，收到 {trend_window}")
        self.lambda_complexity = lambda_complexity
        self.lambda_safety = lambda_safety
        self.trend_window = trend_window

    def compute(
        self,
        free_energy: float,
        expected_free_energy: float,
        model_complexity: float,
        safety_guard_state: SafetyGuardState,
        dt: float = 1.0
    ) -> GlobalObjective:
        """计算全局目标值（A91.1）。

        J = ∫F dt + ∫G dt + λ₁·R_complexity + λ₂·R_safety

        Args:
            free_energy: 当前变分自由能 F
            expected_free_energy: 当前预期自由能 G
            model_complexity: 当前模型复杂度 D_KL
            safety_guard_state: 安全护栏状态
            dt: 时间步长，必须 > 0

        Returns:
            GlobalObjective 全局目标状态

        Raises:
            ValueError: 当dt <= 0时

        Examples:
            >>> calc = GlobalObjectiveCalculator()
            >>> sg = SafetyGuardExecutor().check_all()
            >>> obj = calc.compute(1.0, 2.0, 0.5, sg, 1.0)
            >>> obj.total >= 0
            True
        """
        if dt <= 0:
            raise ValueError(f"dt必须 > 0，收到 {dt}")
        cum_f = free_energy * dt
        cum_g = expected_free_energy * dt
        complexity_penalty = self.lambda_complexity * model_complexity
        safety_penalty = self.compute_safety_term(safety_guard_state)
        total = cum_f + cum_g + complexity_penalty + safety_penalty
        return GlobalObjective(
            safety_guard_state=safety_guard_state,
            cumulative_free_energy=max(0.0, cum_f),
            cumulative_expected_free_energy=max(0.0, cum_g),
            complexity_penalty=max(0.0, complexity_penalty),
            safety_penalty=max(0.0, safety_penalty),
            total=max(0.0, total),
            accuracy_trend=True,
            coverage_trend=True,
            efficiency_trend=True,
            safety_trend=True,
            lambda_complexity=self.lambda_complexity,
            lambda_safety=self.lambda_safety,
            compute_efficiency=0.0,
            param_efficiency=0.0,
            coverage_ratio=0.0
        )

    def compute_accuracy_term(
        self,
        F_history: List[float]
    ) -> Tuple[float, bool]:
        """计算精度进化项（A91.2）。

        Args:
            F_history: 变分自由能历史列表，长度 > 0

        Returns:
            (累积自由能, 精度进化趋势)
            - 趋势为True表示自由能在下降（精度在提升）

        Raises:
            ValueError: 当F_history为空时

        Examples:
            >>> calc = GlobalObjectiveCalculator()
            >>> cum, trend = calc.compute_accuracy_term([10.0, 9.5, 9.0])
            >>> trend
            True
        """
        if not F_history:
            raise ValueError("F_history不能为空")
        cum_f = sum(F_history)
        if len(F_history) < 2:
            return cum_f, True
        trend = F_history[-1] < F_history[0]
        return cum_f, trend

    def compute_coverage_term(
        self,
        G_history: List[float]
    ) -> Tuple[float, bool]:
        """计算广度进化项（A91.3）。

        Args:
            G_history: 预期自由能历史列表，长度 > 0

        Returns:
            (累积预期自由能, 广度进化趋势)
            - 趋势为True表示G在下降（广度在提升）

        Raises:
            ValueError: 当G_history为空时

        Examples:
            >>> calc = GlobalObjectiveCalculator()
            >>> cum, trend = calc.compute_coverage_term([5.0, 4.5, 4.0])
            >>> trend
            True
        """
        if not G_history:
            raise ValueError("G_history不能为空")
        cum_g = sum(G_history)
        if len(G_history) < 2:
            return cum_g, True
        trend = G_history[-1] < G_history[0]
        return cum_g, trend

    def compute_efficiency_term(
        self,
        theta: np.ndarray,
        prior: np.ndarray
    ) -> Tuple[float, float, float]:
        """计算能效进化项（A91.4）。

        Args:
            theta: 当前模型参数，shape (n,)
            prior: 先验分布参数，shape (n,)

        Returns:
            (复杂度惩罚, 计算效率, 参数效率)

        Raises:
            ValueError: 当theta和prior形状不匹配时

        Examples:
            >>> calc = GlobalObjectiveCalculator()
            >>> theta = np.array([1.0, 0.5])
            >>> prior = np.array([0.0, 0.0])
            >>> penalty, ce, pe = calc.compute_efficiency_term(theta, prior)
            >>> penalty >= 0
            True
        """
        if theta.shape != prior.shape:
            raise ValueError(
                f"theta和prior形状不匹配: {theta.shape} vs {prior.shape}"
            )
        kl = float(np.sum(theta ** 2 - prior ** 2))
        complexity = self.lambda_complexity * max(0.0, kl)
        compute_eff = 1.0 / (1.0 + float(np.sum(np.abs(theta))))
        param_eff = 1.0 / (1.0 + len(theta))
        return complexity, compute_eff, param_eff

    def compute_safety_term(
        self,
        safety_guard_state: SafetyGuardState
    ) -> float:
        """计算安全进化项（A91.5）。

        Args:
            safety_guard_state: 安全护栏状态

        Returns:
            安全违规惩罚 >= 0

        Examples:
            >>> calc = GlobalObjectiveCalculator()
            >>> sg = SafetyGuardExecutor().check_all()
            >>> penalty = calc.compute_safety_term(sg)
            >>> penalty >= 0
            True
        """
        violations = safety_guard_state.total_violations
        return self.lambda_safety * violations

    def verify_equivalence(
        self,
        J: float,
        f: float
    ) -> bool:
        """验证全局目标与适应度的等价性（定理91.1）。

        Args:
            J: 全局目标值
            f: 适应度值

        Returns:
            当|J + f| < 0.01时返回True

        Examples:
            >>> calc = GlobalObjectiveCalculator()
            >>> calc.verify_equivalence(1.0, -1.0)
            True
            >>> calc.verify_equivalence(1.0, -1.1)
            False
        """
        return abs(J + f) < 0.01

    def step(
        self,
        state: GlobalObjective,
        free_energy: float,
        expected_free_energy: float,
        model_complexity: float,
        safety_guard_state: SafetyGuardState,
        dt: float = 1.0
    ) -> GlobalObjective:
        """执行一步全局目标更新。

        Args:
            state: 当前全局目标状态
            free_energy: 当前变分自由能
            expected_free_energy: 当前预期自由能
            model_complexity: 当前模型复杂度
            safety_guard_state: 安全护栏状态
            dt: 时间步长，必须 > 0

        Returns:
            更新后的全局目标状态

        Examples:
            >>> calc = GlobalObjectiveCalculator()
            >>> sg = SafetyGuardExecutor().check_all()
            >>> obj = calc.compute(1.0, 2.0, 0.5, sg, 1.0)
            >>> new_obj = calc.step(obj, 0.9, 1.9, 0.5, sg, 1.0)
            >>> new_obj.cumulative_free_energy >= obj.cumulative_free_energy
            True
        """
        if dt <= 0:
            raise ValueError(f"dt必须 > 0，收到 {dt}")
        new_cum_f = state.cumulative_free_energy + free_energy * dt
        new_cum_g = state.cumulative_expected_free_energy + expected_free_energy * dt
        complexity_penalty = self.lambda_complexity * model_complexity
        safety_penalty = self.compute_safety_term(safety_guard_state)
        total = new_cum_f + new_cum_g + complexity_penalty + safety_penalty
        return GlobalObjective(
            safety_guard_state=safety_guard_state,
            cumulative_free_energy=max(0.0, new_cum_f),
            cumulative_expected_free_energy=max(0.0, new_cum_g),
            complexity_penalty=max(0.0, complexity_penalty),
            safety_penalty=max(0.0, safety_penalty),
            total=max(0.0, total),
            accuracy_trend=state.accuracy_trend,
            coverage_trend=state.coverage_trend,
            efficiency_trend=state.efficiency_trend,
            safety_trend=state.safety_trend,
            lambda_complexity=self.lambda_complexity,
            lambda_safety=self.lambda_safety,
            compute_efficiency=state.compute_efficiency,
            param_efficiency=state.param_efficiency,
            coverage_ratio=state.coverage_ratio
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `SafetyGuardState` | ← 阶段90 | 作为全局目标的安全护栏输入（组合字段） |
| **输入** | A4.4（变分自由能） | ← 阶段4 | 精度进化项的定义来源 |
| **输入** | A7.1（预期自由能） | ← 阶段7 | 广度进化项的定义来源 |
| **输入** | A24.1（全局奖励） | ← 阶段24 | 全局目标与奖励的等价关系 |
| **输入** | A8.1（适应度） | ← 阶段8 | 全局目标与适应度的对偶关系 |
| **输入** | A5.2（复杂性项） | ← 阶段5 | 能效进化项的定义来源 |
| **输出** | `GlobalObjective` | → 阶段92 | 全局目标函数，作为全局动力学的优化目标 |
| **输出** | `GlobalObjective` | → 阶段93 | 全局目标函数，作为调度器的评估依据 |
| **输出** | `GlobalObjective` | → 阶段98 | 全局目标函数，作为监控面板的数据源 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 自动微分 | 计算 ∂J/∂θ 用于全局动力学 |
| NumPy | 数值计算 | 累积求和和趋势判定 |
| SciPy | 统计检验 | 趋势显著性检验 |
| psutil | 资源监控 | 安全违规项的数据来源 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V91.1 | 全局目标计算 | J = ∫F dt + ∫G dt + λ₁R + λ₂S | 公式验证 |
| V91.2 | J非负性 | J ≥ 0 恒成立 | 边界测试 |
| V91.3 | 精度进化项 | ∫F dt 正确累积 | 累积测试 |
| V91.4 | 广度进化项 | ∫G dt 正确累积 | 累积测试 |
| V91.5 | 能效进化项 | λ₁R_complexity 正确计算 | 复杂度测试 |
| V91.6 | 安全进化项 | λ₂R_safety 从SafetyGuardState提取 | 安全测试 |
| V91.7 | 精度趋势判定 | F持续下降时accuracy_trend=True | 趋势测试 |
| V91.8 | 广度趋势判定 | G持续下降时coverage_trend=True | 趋势测试 |
| V91.9 | 安全约束满足时 | R_safety = 0 | 约束测试 |
| V91.10 | 安全约束违反时 | R_safety > 0 | 违规测试 |
| V91.11 | 等价性验证 | J + f ≈ 0 | 对偶测试 |
| V91.12 | GlobalObjective组合 | `state.safety_guard_state` 类型为 `SafetyGuardState` | 类型检查 |
| V91.13 | 系数约束 | λ_safety > λ_complexity | 参数测试 |
| V91.14 | 离散化一致性 | 离散J与连续J误差<1% | 精度测试 |
| V91.15 | 单步增量 | ΔJ_t = (F + G)Δt | 增量测试 |
| V91.16 | 长期运行 | J单调递减（无外部扰动时） | 集成测试 |
| V91.17 | 四维协同 | 精度和广度同向变化 | 协同测试 |
| V91.18 | 资源约束下 | R_safety=0时J简化正确 | 约束测试 |

---

> **阶段91完** | 阶段91建立了全局统一目标函数：$J = \int_0^T F \, dt + \int_0^T G \, dt + \lambda_1 R_{complexity} + \lambda_2 R_{safety}$（A91.1），精度进化项 ∫F dt → 长期预测误差下降（A91.2），广度进化项 ∫G dt → 认知覆盖扩张（A91.3），能效进化项 λ₁R_complexity → 算力消耗降低（A91.4），安全进化项 λ₂R_safety → 风险消除（A91.5），证明了全局目标函数与终极进化目标的等价性（定理91.1），封装为GlobalObjective数据类组合SafetyGuardState。至此，系统有了唯一的优化目标——最小化J等价于最大化适应度，等价于实现终极进化。下一步：阶段92将建立全局微分方程组，定义感知、行动、学习、进化、注意力五类动力学。
