# 阶段64：结构性重构触发条件——持续高误差、长期停滞与分布偏移的联合判定

> **阶段编号**：S64 | **所属层**：第七层·进化适应层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 持续高误差触发 | "修不好就重建"——误差持续居高不下，微修正不够用，需要重构 |
> | 长期停滞触发 | "原地踏步太久"——适应度长期不提升，模型已到瓶颈 |
> | 新环境探索触发 | "世界变了"——环境分布发生根本性变化，旧模型不再适用 |
> | 重构风险评估 | "重建值不值"——评估重构的成本、不确定性和潜在收益 |
> | StructuralReformTrigger | "重构触发的完整画像"——组合MicroAdjusterState，在微修正状态基础上增加重构判定 |

---

## 1. 第一性原理依据

本阶段从阶段58（大误差→结构重构触发）和阶段8（适应度与自然选择）的核心结论出发，建立结构性重构触发条件的精确判定规则。核心依据为：

- **A58.2（大误差→结构重构触发）**（引用阶段58）：$\varepsilon_{\text{weighted}} \geq \varepsilon_{\text{restructure\_threshold}} \implies \text{trigger\_restructure}$，当误差较大时触发结构重构。本阶段将"误差较大"从定性描述提升为精确的定量触发条件——不仅要求误差大，还要求误差持续足够长时间，且微修正已无法改善。

- **A8.1（适应度定义）**（引用阶段8）：$f(\theta) = -(1/T)\sum_{t=1}^{T} F(q_t, o_t; \theta)$，适应度是自由能时间平均的负值。长期停滞触发基于适应度不再提升的判定——当适应度长期不变时，说明当前模型结构已达到瓶颈。

- **A8.7（结构变异算子）**（引用阶段8）：$\mathcal{M}_{\text{struct}} = \{\text{FeatureAdd/Del, ModuleRecomb, RuleIterate}\}$，结构变异的具体操作。本阶段定义何时触发这些结构变异操作。

- **A56.1-A56.4（分布偏移检测）**（引用阶段56）：KS检验、MMD、在线检测和偏移量化。新环境探索触发基于分布偏移检测——当环境分布发生根本性变化时，旧模型不再适用，需要结构重构。

- **A63.1-A63.4（微修正Python接口）**（引用阶段63）：MicroAdjusterState记录了微修正的完整状态。结构性重构触发条件需要检查微修正是否已收敛但效果不佳——如果微修正已收敛但自由能仍高，说明需要结构重构。

本阶段的核心目标是：**定义持续高误差触发 $E_{\text{cumulative}} > \theta_{\text{high}}$ AND duration $> T_{\min}$（A64.1），长期停滞触发 $f(\theta) - f(\theta^*) < \epsilon$ AND duration $> T_{\text{stagnation}}$（A64.2），新环境探索触发 distribution_shift $> \theta_{\text{shift}}$（A64.3），重构风险评估 risk = f(complexity, uncertainty, resource_cost)（A64.4），封装为StructuralReformTrigger数据类，组合MicroAdjusterState作为微修正状态的输入。**

> **去重声明**：本阶段仅定义结构性重构的触发条件，不重新定义结构变异算子（阶段8，A8.7）或结构重构算法（阶段65）。触发条件是A58.2中"误差较大"的精确化，不改变A58.2的定义。

---

## 2. 核心数学公式

### 第一部分：持续高误差触发

#### 2.1 持续高误差触发定义（A64.1）

当误差持续居高不下且持续时间超过最小阈值时，触发结构性重构：

$$\text{trigger\_high\_error} = \mathbb{1}\left[E_{\text{cumulative}} > \theta_{\text{high}} \;\wedge\; \text{duration} > T_{\min}\right] \tag{A64.1}$$

其中：
- $E_{\text{cumulative}} = \sum_{t=t_0}^{t_0 + \text{duration}} \varepsilon_t$ 为累积误差
- $\theta_{\text{high}} > 0$ 为高误差阈值（通常设为 $\theta_{\text{high}} = T_{\min} \cdot \varepsilon_{\text{restructure\_threshold}}$，即最小持续时间内的重构阈值累积误差）
- $\text{duration}$ 为误差持续超过微修正阈值的时间长度
- $T_{\min} > 0$ 为最小持续时间（通常设为50-100步）

**持续时间的必要性**：

仅检查累积误差不够——短暂的误差尖峰（如噪声）不应触发重构。持续时间要求确保误差是"持续"的而非"瞬时的"：

| 误差模式 | 累积误差 | 持续时间 | 触发动作 |
|---------|---------|---------|---------|
| 短暂尖峰 | 可能超过阈值 | < $T_{\min}$ | 不触发（可能是噪声） |
| 持续高误差 | 超过阈值 | > $T_{\min}$ | 触发结构重构 |
| 间歇高误差 | 可能超过阈值 | < $T_{\min}$（每次） | 不触发（微修正可能足够） |

**与A61.1（微修正触发条件）的互补性**：

A61.1要求梯度小且自由能稳定才触发微修正。A64.1是A61.1的"反面"——当梯度大（误差高）且持续时间长时，微修正无法解决问题，需要结构重构。两者构成互补的触发机制：

| 条件 | 触发动作 |
|------|---------|
| 梯度小 + 自由能稳定 + 环境稳定 | 微修正（A61.1） |
| 误差高 + 持续时间长 | 结构重构（A64.1） |
| 误差高 + 持续时间短 | 增强更新（增大步长） |

- **输入**：累积误差 $E_{\text{cumulative}}$，持续时间，阈值 $\theta_{\text{high}}$，$T_{\min}$
- **输出**：触发标志
- **约束**：$\theta_{\text{high}} > 0$；$T_{\min} > 0$

> **白话翻译**：持续高误差触发就是"修不好就重建"——误差持续居高不下超过一定时间，说明微修正已经不够用了，需要更激进的结构重构。关键是"持续"——短暂的误差尖峰不算，必须持续足够长时间才能触发重构。

---

### 第二部分：长期停滞触发

#### 2.2 长期停滞触发定义（A64.2）

当适应度长期不提升时，说明当前模型结构已达到瓶颈，需要结构重构来突破：

$$\text{trigger\_stagnation} = \mathbb{1}\left[f(\theta_t) - f(\theta^*) < \epsilon \;\wedge\; \text{duration} > T_{\text{stagnation}}\right] \tag{A64.2}$$

其中：
- $f(\theta_t)$ 为当前适应度（引用阶段8，A8.1）
- $f(\theta^*)$ 为历史最优适应度
- $\epsilon > 0$ 为适应度提升阈值（通常设为 $0.01 \cdot |f(\theta^*)|$，即历史最优适应度的1%）
- $T_{\text{stagnation}} > 0$ 为停滞判定时间（通常设为200-500步）

**长期停滞的物理含义**：

适应度长期不提升意味着当前模型结构已达到其表达能力上限——无论怎么微调参数，都无法进一步提升适应度。此时需要结构重构来增加模型的表达能力。

**停滞与微修正收敛的区别**：

| 维度 | 微修正收敛（A63.4） | 长期停滞（A64.2） |
|------|-------------------|-------------------|
| 自由能水平 | 低（接近最优） | 高（远离最优） |
| 适应度水平 | 高 | 低或停滞 |
| 含义 | 当前结构下已达到最优 | 当前结构无法达到更优 |
| 动作 | 停止微修正 | 触发结构重构 |

关键区别：微修正收敛时自由能低（已接近最优），长期停滞时自由能可能仍高（当前结构无法进一步降低自由能）。

**停滞检测的滑动窗口**：

为避免偶发的适应度波动误触发停滞检测，使用滑动窗口内的适应度趋势：

$$\text{trend} = \frac{f_{\max}(W_{\text{stag}}) - f_{\min}(W_{\text{stag}})}{T_{\text{stagnation}}} \tag{A64.2-trend}$$

当 trend $< \epsilon / T_{\text{stagnation}}$ 时判定为停滞。

- **输入**：适应度历史，历史最优适应度，阈值 $\epsilon$，$T_{\text{stagnation}}$
- **输出**：触发标志
- **约束**：$\epsilon > 0$；$T_{\text{stagnation}} > 0$

> **白话翻译**：长期停滞触发就是"原地踏步太久"——适应度长期不提升，说明当前模型结构已经到瓶颈了，再怎么微调也没用。和微修正收敛的区别是：收敛是"已经到最优了"，停滞是"到不了最优"——当前结构的表达能力不够，需要重构来突破瓶颈。

---

### 第三部分：新环境探索触发

#### 2.3 新环境探索触发定义（A64.3）

当环境分布发生根本性变化时，旧模型不再适用，需要结构重构来适应新环境：

$$\text{trigger\_new\_env} = \mathbb{1}\left[\text{distribution\_shift} > \theta_{\text{shift}}\right] \tag{A64.3}$$

其中：
- $\text{distribution\_shift}$ 为分布偏移度量（引用阶段56，A56.4）
- $\theta_{\text{shift}} > 0$ 为分布偏移阈值

**分布偏移的量化**：

分布偏移度量使用阶段56定义的KS统计量和MMD的组合：

$$\text{distribution\_shift} = \alpha_{\text{KS}} \cdot \frac{D}{D_{\text{threshold}}} + (1 - \alpha_{\text{KS}}) \cdot \frac{\text{MMD}^2}{\text{MMD}^2_{\text{threshold}}} \tag{A64.3-metric}$$

其中 $\alpha_{\text{KS}} \in [0, 1]$ 为KS权重（通常设为0.5）。

**新环境触发的特殊性**：

新环境触发与持续高误差触发和长期停滞触发不同——它不是基于模型性能的下降，而是基于环境本身的变化。即使当前模型性能尚未显著下降，如果检测到环境分布发生了根本性变化，也应提前触发结构重构，避免模型性能在未来急剧恶化。

**新环境触发的保守策略**：

由于分布偏移检测可能存在误报（如噪声导致的短暂偏移），新环境触发应采用保守策略：

1. **持续性检查**：分布偏移必须持续 $T_{\text{shift}}$ 步（通常10-20步）才触发
2. **幅度检查**：偏移幅度必须超过 $\theta_{\text{shift}}$（通常设为2.0，即偏移度量为阈值的2倍）
3. **性能验证**：偏移后模型性能确实下降才触发重构

- **输入**：分布偏移度量，偏移阈值，持续性参数
- **输出**：触发标志
- **约束**：$\theta_{\text{shift}} > 0$

> **白话翻译**：新环境探索触发就是"世界变了"——环境分布发生了根本性变化，旧模型不再适用。和持续高误差触发的区别是：高误差触发是"模型修不好"，新环境触发是"世界变了"。即使模型当前还没出问题，如果检测到环境已经变了，也要提前重构，避免将来出大问题。

---

### 第四部分：重构风险评估

#### 2.4 重构风险评估定义（A64.4）

在触发结构重构之前，需要评估重构的风险，避免盲目重构导致性能倒退：

$$\text{risk} = f(\text{complexity}, \text{uncertainty}, \text{resource\_cost}) \tag{A64.4}$$

具体形式为：

$$\text{risk} = w_c \cdot \text{complexity} + w_u \cdot \text{uncertainty} + w_r \cdot \text{resource\_cost} \tag{A64.4-formula}$$

其中：
- $\text{complexity} \in [0, 1]$ 为重构复杂度（结构变化越大，复杂度越高）
- $\text{uncertainty} \in [0, 1]$ 为重构不确定性（新结构的性能越不确定，不确定性越高）
- $\text{resource\_cost} \in [0, 1]$ 为资源成本（重构消耗的计算资源越多，成本越高）
- $w_c, w_u, w_r \geq 0$ 为权重（$w_c + w_u + w_r = 1$）

**重构复杂度的量化**：

$$\text{complexity} = \frac{|\theta_{\text{new}}| - |\theta_{\text{old}}|}{|\theta_{\text{old}}|} + \frac{\text{structural\_changes}}{\text{total\_components}} \tag{A64.4-complexity}$$

其中 $|\theta|$ 为参数数量，structural_changes 为结构变化次数。

**重构不确定性的量化**：

$$\text{uncertainty} = 1 - \exp(-\lambda \cdot \text{distance\_to\_known}) \tag{A64.4-uncertainty}$$

其中 distance_to_known 为新结构与已知良好结构的距离，$\lambda > 0$ 为距离敏感度。

**资源成本的量化**：

$$\text{resource\_cost} = \frac{\text{estimated\_time}}{T_{\max}} + \frac{\text{estimated\_memory}}{M_{\max}} \tag{A64.4-resource}$$

**风险评估的决策规则**：

| 风险等级 | risk范围 | 决策 |
|---------|---------|------|
| 低风险 | $[0, 0.3)$ | 允许重构 |
| 中风险 | $[0.3, 0.7)$ | 允许重构但需渐进式（A66.4） |
| 高风险 | $[0.7, 1.0]$ | 仅允许最小化重构或暂停 |

- **输入**：重构方案，当前模型，资源限制
- **输出**：风险值 $\text{risk} \in [0, 1]$
- **约束**：$w_c + w_u + w_r = 1$；$\text{risk} \in [0, 1]$

> **白话翻译**：重构风险评估就是"重建值不值"——评估重构的复杂度、不确定性和资源成本。复杂度越高（改动越大）、不确定性越高（新结构没试过）、资源成本越高（耗时耗力），风险越大。低风险就大胆重构，高风险就谨慎行事，只做最小化的改动。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A64.1 | $\text{trigger} = \mathbb{1}[E_{\text{cumulative}} > \theta_{\text{high}} \wedge \text{duration} > T_{\min}]$ | 累积误差、持续时间、阈值 | 触发标志 | $\theta_{\text{high}}, T_{\min} > 0$ |
| A64.2 | $\text{trigger} = \mathbb{1}[f(\theta) - f(\theta^*) < \epsilon \wedge \text{duration} > T_{\text{stagnation}}]$ | 适应度历史、阈值 | 触发标志 | $\epsilon, T_{\text{stagnation}} > 0$ |
| A64.3 | $\text{trigger} = \mathbb{1}[\text{distribution\_shift} > \theta_{\text{shift}}]$ | 分布偏移度量、阈值 | 触发标志 | $\theta_{\text{shift}} > 0$ |
| A64.4 | $\text{risk} = w_c \cdot \text{complexity} + w_u \cdot \text{uncertainty} + w_r \cdot \text{resource\_cost}$ | 重构方案、模型、资源 | 风险值 | $\text{risk} \in [0, 1]$ |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from enum import Enum
import numpy as np

# 引用阶段63的数据类
from stage_063 import MicroAdjusterState


# ============================================================
# 阶段64 枚举：重构触发类型
# ============================================================
class ReformTriggerType(Enum):
    """结构性重构触发类型。"""
    HIGH_ERROR = "high_error"            # 持续高误差
    STAGNATION = "stagnation"            # 长期停滞
    NEW_ENVIRONMENT = "new_environment"  # 新环境
    NONE = "none"                        # 未触发


# ============================================================
# 阶段64 枚举：风险等级
# ============================================================
class RiskLevel(Enum):
    """重构风险等级。"""
    LOW = "low"          # [0, 0.3)
    MEDIUM = "medium"    # [0.3, 0.7)
    HIGH = "high"        # [0.7, 1.0]


# ============================================================
# 阶段64 数据类：结构性重构触发
# ============================================================
@dataclass(frozen=True)
class StructuralReformTrigger:
    """结构性重构触发条件的完整描述。

    组合阶段63的MicroAdjusterState，在微修正状态的基础上增加
    结构重构的触发判定信息。

    Attributes:
        micro_adjuster_state: 阶段63的微修正器状态（组合字段）
        trigger_type: 触发类型
        cumulative_error: 累积误差
        error_duration: 误差持续时间
        theta_high: 高误差阈值
        T_min: 最小持续时间
        fitness_current: 当前适应度
        fitness_best: 历史最优适应度
        stagnation_duration: 停滞持续时间
        epsilon_fitness: 适应度提升阈值
        T_stagnation: 停滞判定时间
        distribution_shift: 分布偏移度量
        theta_shift: 分布偏移阈值
        risk_score: 重构风险值
        risk_level: 风险等级
        high_error_triggered: 持续高误差是否触发
        stagnation_triggered: 长期停滞是否触发
        new_env_triggered: 新环境是否触发

    Invariants:
        - theta_high > 0
        - T_min > 0
        - epsilon_fitness > 0
        - T_stagnation > 0
        - theta_shift > 0
        - risk_score ∈ [0, 1]
    """
    micro_adjuster_state: MicroAdjusterState
    trigger_type: ReformTriggerType = ReformTriggerType.NONE
    cumulative_error: float = 0.0
    error_duration: int = 0
    theta_high: float = 50.0
    T_min: int = 50
    fitness_current: float = 0.0
    fitness_best: float = 0.0
    stagnation_duration: int = 0
    epsilon_fitness: float = 0.01
    T_stagnation: int = 200
    distribution_shift: float = 0.0
    theta_shift: float = 2.0
    risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    high_error_triggered: bool = False
    stagnation_triggered: bool = False
    new_env_triggered: bool = False

    def __post_init__(self):
        if self.theta_high <= 0:
            raise ValueError(
                f"高误差阈值必须 > 0，收到 {self.theta_high}"
            )
        if self.T_min <= 0:
            raise ValueError(
                f"最小持续时间必须 > 0，收到 {self.T_min}"
            )
        if self.epsilon_fitness <= 0:
            raise ValueError(
                f"适应度提升阈值必须 > 0，收到 {self.epsilon_fitness}"
            )
        if self.T_stagnation <= 0:
            raise ValueError(
                f"停滞判定时间必须 > 0，收到 {self.T_stagnation}"
            )
        if self.theta_shift <= 0:
            raise ValueError(
                f"分布偏移阈值必须 > 0，收到 {self.theta_shift}"
            )
        if not (0 <= self.risk_score <= 1):
            raise ValueError(
                f"风险值必须 ∈ [0,1]，收到 {self.risk_score}"
            )

    @property
    def any_triggered(self) -> bool:
        """是否有任何重构触发条件满足。"""
        return (
            self.high_error_triggered
            or self.stagnation_triggered
            or self.new_env_triggered
        )

    @property
    def should_reform(self) -> bool:
        """是否应执行结构重构（触发+低风险）。"""
        return self.any_triggered and self.risk_level != RiskLevel.HIGH


# ============================================================
# 阶段64 核心接口：结构性重构触发判定器
# ============================================================
class StructuralReformTriggerChecker:
    """阶段64的核心接口：结构性重构触发判定器。

    判定结构性重构的触发条件——持续高误差、长期停滞、
    新环境和重构风险评估。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 重构触发与微修正触发互斥
        - 高风险重构需要额外确认
    """

    def check_high_error(
        self,
        cumulative_error: float,
        duration: int,
        theta_high: float,
        T_min: int
    ) -> bool:
        """检查持续高误差触发（A64.1）。

        Args:
            cumulative_error: 累积误差
            duration: 误差持续时间
            theta_high: 高误差阈值
            T_min: 最小持续时间

        Returns:
            是否触发
        """
        return cumulative_error > theta_high and duration > T_min

    def check_stagnation(
        self,
        fitness_current: float,
        fitness_best: float,
        duration: int,
        epsilon: float,
        T_stagnation: int
    ) -> bool:
        """检查长期停滞触发（A64.2）。

        Args:
            fitness_current: 当前适应度
            fitness_best: 历史最优适应度
            duration: 停滞持续时间
            epsilon: 适应度提升阈值
            T_stagnation: 停滞判定时间

        Returns:
            是否触发
        """
        return (fitness_best - fitness_current) < epsilon and duration > T_stagnation

    def check_new_environment(
        self,
        distribution_shift: float,
        theta_shift: float
    ) -> bool:
        """检查新环境触发（A64.3）。

        Args:
            distribution_shift: 分布偏移度量
            theta_shift: 分布偏移阈值

        Returns:
            是否触发
        """
        return distribution_shift > theta_shift

    def assess_risk(
        self,
        complexity: float,
        uncertainty: float,
        resource_cost: float,
        w_c: float = 0.4,
        w_u: float = 0.4,
        w_r: float = 0.2
    ) -> Tuple[float, RiskLevel]:
        """评估重构风险（A64.4）。

        Args:
            complexity: 重构复杂度 ∈ [0, 1]
            uncertainty: 重构不确定性 ∈ [0, 1]
            resource_cost: 资源成本 ∈ [0, 1]
            w_c, w_u, w_r: 权重

        Returns:
            (风险值, 风险等级)
        """
        risk = w_c * complexity + w_u * uncertainty + w_r * resource_cost
        risk = max(0.0, min(1.0, risk))
        if risk < 0.3:
            level = RiskLevel.LOW
        elif risk < 0.7:
            level = RiskLevel.MEDIUM
        else:
            level = RiskLevel.HIGH
        return risk, level

    def check_trigger(
        self,
        trigger: StructuralReformTrigger
    ) -> StructuralReformTrigger:
        """执行完整的重构触发判定（A64.1-A64.4）。

        Args:
            trigger: 当前触发状态

        Returns:
            更新后的StructuralReformTrigger
        """
        high_error = self.check_high_error(
            trigger.cumulative_error, trigger.error_duration,
            trigger.theta_high, trigger.T_min
        )
        stagnation = self.check_stagnation(
            trigger.fitness_current, trigger.fitness_best,
            trigger.stagnation_duration, trigger.epsilon_fitness,
            trigger.T_stagnation
        )
        new_env = self.check_new_environment(
            trigger.distribution_shift, trigger.theta_shift
        )
        trigger_type = ReformTriggerType.NONE
        if high_error:
            trigger_type = ReformTriggerType.HIGH_ERROR
        elif stagnation:
            trigger_type = ReformTriggerType.STAGNATION
        elif new_env:
            trigger_type = ReformTriggerType.NEW_ENVIRONMENT
        return StructuralReformTrigger(
            micro_adjuster_state=trigger.micro_adjuster_state,
            trigger_type=trigger_type,
            cumulative_error=trigger.cumulative_error,
            error_duration=trigger.error_duration,
            theta_high=trigger.theta_high,
            T_min=trigger.T_min,
            fitness_current=trigger.fitness_current,
            fitness_best=trigger.fitness_best,
            stagnation_duration=trigger.stagnation_duration,
            epsilon_fitness=trigger.epsilon_fitness,
            T_stagnation=trigger.T_stagnation,
            distribution_shift=trigger.distribution_shift,
            theta_shift=trigger.theta_shift,
            risk_score=trigger.risk_score,
            risk_level=trigger.risk_level,
            high_error_triggered=high_error,
            stagnation_triggered=stagnation,
            new_env_triggered=new_env,
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `MicroAdjusterState` | ← 阶段63 | 作为重构触发的微修正状态输入（组合字段） |
| **输入** | A58.2（大误差→结构重构触发） | ← 阶段58 | 结构重构触发的理论基础 |
| **输入** | A8.1（适应度定义） | ← 阶段8 | 长期停滞判定的理论依据 |
| **输入** | A56.1-A56.4（分布偏移检测） | ← 阶段56 | 新环境触发的理论依据 |
| **输出** | `StructuralReformTrigger` | → 阶段65 | 重构触发状态，作为重构算法的输入 |
| **输出** | A64.1-A64.4（重构触发条件定义） | → 阶段65-67 | 结构重构触发的精确条件 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 适应度计算 | 适应度函数的自动微分和JIT编译 |
| SciPy | 统计检验 | KS检验和MMD计算支撑分布偏移检测 |
| NumPy | 累积误差计算 | 滑动窗口内的误差累积和方差计算 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V64.1 | 持续高误差触发正确性 | $E_{\text{cumulative}} > \theta_{\text{high}}$ 且 duration $> T_{\min}$ 时触发 | 条件测试 |
| V64.2 | 短暂尖峰不触发 | 持续时间 < $T_{\min}$ 时不触发 | 条件测试 |
| V64.3 | 长期停滞触发正确性 | 适应度不提升且 duration $> T_{\text{stagnation}}$ 时触发 | 条件测试 |
| V64.4 | 适应度仍在提升时不触发 | 适应度提升 > $\epsilon$ 时不触发 | 条件测试 |
| V64.5 | 新环境触发正确性 | distribution_shift $> \theta_{\text{shift}}$ 时触发 | 条件测试 |
| V64.6 | 小偏移不触发 | distribution_shift $< \theta_{\text{shift}}$ 时不触发 | 条件测试 |
| V64.7 | 风险评估正确性 | risk ∈ [0, 1] 且风险等级与范围对应 | 范围检查 |
| V64.8 | 高风险不执行重构 | risk ≥ 0.7 时不执行重构 | 条件测试 |
| V64.9 | 触发类型正确 | 优先级：高误差 > 停滞 > 新环境 | 优先级测试 |
| V64.10 | StructuralReformTrigger组合 | `trigger.micro_adjuster_state` 类型为 `MicroAdjusterState` | 类型检查 |
| V64.11 | 非法阈值拒绝 | $\theta_{\text{high}} \leq 0$ 等抛出 `ValueError` | 异常测试 |
| V64.12 | 微修正与重构互斥 | 同一时间步不同时触发微修正和重构 | 互斥测试 |
| V64.13 | 长期运行触发合理性 | 1000步循环中重构触发次数合理 | 长期运行测试 |

---

> **阶段64完** | 阶段64建立了结构性重构触发条件的精确判定规则：持续高误差触发 $E_{\text{cumulative}} > \theta_{\text{high}}$ AND duration $> T_{\min}$（A64.1），长期停滞触发 $f(\theta) - f(\theta^*) < \epsilon$ AND duration $> T_{\text{stagnation}}$（A64.2），新环境触发 distribution_shift $> \theta_{\text{shift}}$（A64.3），重构风险评估 risk = f(complexity, uncertainty, resource_cost)（A64.4），封装为StructuralReformTrigger数据类组合MicroAdjusterState。下一步：阶段65将定义结构性重构的具体算法。
