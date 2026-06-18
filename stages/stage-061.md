# 阶段61：微修正触发条件——梯度范数与自由能稳定性的联合判定

> **阶段编号**：S61 | **所属层**：第七层·进化适应层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 微修正触发条件 | "什么时候该微调"——梯度小且自由能稳定时才做微修正 |
> | 梯度范数阈值 | "还有多少可调"——梯度范数低于阈值说明已接近最优 |
> | 自由能稳定性 | "最近稳不稳"——自由能变化量低于阈值说明系统已稳定 |
> | 稳定环境判定 | "环境变没变"——环境方差低于阈值说明环境未发生变化 |
> | 触发频率限制 | "别修太勤"——两次微修正之间必须间隔足够步数 |
> | MicroAdjustTrigger | "微修正触发的完整画像"——组合ErrorValidationResult，在验证结果基础上增加触发判定 |

---

## 1. 第一性原理依据

本阶段从阶段58（误差反馈到模型的映射）和阶段5（自由能分解）的核心结论出发，建立微修正触发条件的精确判定规则。核心依据为：

- **A58.1（小误差→微修正触发）**（引用阶段58）：$\mu_{\text{update}} = -\kappa \cdot \partial F / \partial \mu$，当误差较小时触发基于自由能梯度的微修正。本阶段将"误差较小"从定性描述提升为精确的定量触发条件——不仅要求误差小，还要求自由能梯度小且自由能变化量稳定。

- **A5.17（自由能梯度流）**（引用阶段5）：$\mathrm{d}\mu / \mathrm{d}t = -\partial F / \partial \mu$，系统在自由能景观上沿梯度流运动。本阶段利用梯度流的性质——当梯度范数趋近于零时，系统接近驻点（局部最优或全局最优），此时微修正才是安全且有效的。

- **A60.1-A60.5（误差评估验证）**（引用阶段60）：合成数据生成、检测延迟、误报率、灵敏度测试和路由正确性测试。ErrorValidationResult为微修正触发提供了经过验证的误差评估基础。

- **A55.1-A55.4（异常检测）**（引用阶段55）：阈值检测、CUSUM、EWMA和组合策略。微修正触发条件需要排除异常状态——当系统处于异常状态时，不应触发微修正，而应触发更激进的响应。

本阶段的核心目标是：**定义微修正触发条件 $\|\partial F / \partial \mu\| < \theta_{\text{micro}}$ AND $\Delta F < \theta_{\text{stable}}$，定义稳定环境判定 $\text{var}(\varepsilon_t, W) < \theta_{\text{var}}$，定义触发频率限制以避免过度修正，封装为MicroAdjustTrigger数据类，组合ErrorValidationResult作为验证结果的输入。** 微修正触发条件是"微修正"与"结构重构"的分水岭——它精确地界定何时应该做微调、何时应该做重构。

> **去重声明**：本阶段仅定义微修正的触发条件，不重新定义微修正算法（阶段62）、表征更新（阶段14）或权重更新（阶段15）。微修正的触发条件是A58.1中"误差较小"的精确化，不改变A58.1的定义。

---

## 2. 核心数学公式

### 第一部分：微修正触发条件

#### 2.1 微修正触发条件定义（A61.1）

微修正触发条件要求自由能梯度范数低于阈值且自由能变化量低于稳定性阈值：

$$\text{trigger\_micro} = \mathbb{1}\left[\left\|\frac{\partial F}{\partial \mu}\right\| < \theta_{\text{micro}} \;\wedge\; \Delta F < \theta_{\text{stable}}\right] \tag{A61.1}$$

其中：
- $\|\partial F / \partial \mu\|$ 为自由能关于模型参数的梯度范数
- $\theta_{\text{micro}} > 0$ 为微修正梯度阈值（通常设为 $0.1 \cdot \|\partial F / \partial \mu\|_{\text{initial}}$，即初始梯度范数的10%）
- $\Delta F = |F_t - F_{t-1}|$ 为相邻步的自由能变化量
- $\theta_{\text{stable}} > 0$ 为自由能稳定性阈值（通常设为 $0.01 \cdot F_{\text{baseline}}$，即自由能基线的1%）

**两个条件的物理含义**：

| 条件 | 含义 | 不满足时的含义 |
|------|------|--------------|
| $\|\partial F / \partial \mu\| < \theta_{\text{micro}}$ | 系统接近驻点，梯度方向不明确 | 系统远离驻点，梯度方向明确，应做更大幅度的更新 |
| $\Delta F < \theta_{\text{stable}}$ | 自由能已稳定，不再大幅波动 | 自由能仍在波动，系统未进入稳态 |

**两个条件缺一不可**：
- 仅满足梯度条件：系统可能处于鞍点（梯度小但自由能仍高），此时微修正无效
- 仅满足稳定性条件：系统可能处于高原（自由能稳定但梯度大），此时微修正方向错误
- 两者同时满足：系统接近局部最优且已稳定，微修正可以精细调优

**与A58.1的对应**：A61.1是A58.1中 $\varepsilon_{\text{weighted}} < \varepsilon_{\text{micro\_threshold}}$ 的精确化。A58.1仅检查误差大小，A61.1同时检查梯度范数和自由能稳定性，避免在鞍点或高原上触发无效的微修正。

- **输入**：自由能梯度 $\partial F / \partial \mu$，自由能变化量 $\Delta F$，阈值 $\theta_{\text{micro}}$，$\theta_{\text{stable}}$
- **输出**：触发标志 $\text{trigger\_micro} \in \{0, 1\}$
- **约束**：$\theta_{\text{micro}} > 0$；$\theta_{\text{stable}} > 0$

> **白话翻译**：微修正触发条件就是"什么时候该微调"——两个条件缺一不可：梯度小（已经接近最优了）且自由能稳定（最近没有大波动）。如果梯度还很大，说明还没走到最优附近，微修正力度不够；如果自由能还在波动，说明系统还不稳定，微修正可能被噪声淹没。

---

### 第二部分：稳定环境判定

#### 2.2 稳定环境判定定义（A61.2）

微修正仅在环境稳定时触发。环境稳定性通过误差序列在滑动窗口内的方差衡量：

$$\text{is\_stable\_env} = \mathbb{1}\left[\text{var}(\varepsilon_t, W) < \theta_{\text{var}}\right] \tag{A61.2}$$

其中：
- $\text{var}(\varepsilon_t, W) = \frac{1}{W-1}\sum_{i=t-W+1}^{t}(\varepsilon_i - \bar{\varepsilon}_W)^2$ 为误差序列在窗口 $W$ 内的样本方差
- $\bar{\varepsilon}_W = \frac{1}{W}\sum_{i=t-W+1}^{t}\varepsilon_i$ 为窗口内的样本均值
- $W$ 为滑动窗口长度（通常设为50-100步）
- $\theta_{\text{var}} > 0$ 为环境方差阈值（通常设为 $2 \cdot \text{var}(\varepsilon, W_{\text{ref}})$，即参考窗口方差的2倍）

**稳定环境判定的物理含义**：

环境稳定性意味着外部世界在近期没有发生显著变化。如果环境正在变化（方差增大），微修正基于旧数据的梯度方向可能已经失效，此时应暂停微修正，等待环境稳定后再调整。

**与A56.3（在线分布偏移检测）的关系**：A61.2的稳定环境判定与A56.3（引用阶段56）的在线分布偏移检测互补：
- A56.3检测分布偏移（环境是否发生了根本性变化）
- A61.2检测方差增大（环境是否正在波动，无论是否发生根本性变化）

两者可以组合使用：当A56.3检测到分布偏移时，A61.2必然判定为不稳定环境；但A61.2判定为不稳定环境时，不一定发生了分布偏移（可能只是噪声增大）。

**环境方差阈值的自适应**：

$\theta_{\text{var}}$ 可以根据历史方差自适应调整：

$$\theta_{\text{var}}(t) = k_\text{var} \cdot \text{median}\left(\{\text{var}(\varepsilon, W)\}_{i=t-T_{\text{ref}}}^{t}\right) \tag{A61.2-adaptive}$$

其中 $k_\text{var} > 1$ 为方差倍数因子（通常设为2），$T_{\text{ref}}$ 为参考窗口长度。自适应阈值使系统能够适应不同噪声水平的环境。

- **输入**：误差序列 $\{\varepsilon_t\}$，窗口长度 $W$，方差阈值 $\theta_{\text{var}}$
- **输出**：环境稳定标志 $\text{is\_stable\_env} \in \{0, 1\}$
- **约束**：$W \geq 10$；$\theta_{\text{var}} > 0$

> **白话翻译**：稳定环境判定就是"环境变没变"——用误差序列的方差衡量环境是否在波动。方差小说明环境稳定，微修正基于的梯度方向是可靠的；方差大说明环境在变化，微修正可能基于过时的信息，应该暂停。

---

### 第三部分：触发频率限制

#### 2.3 触发频率限制定义（A61.3）

为避免过度修正（over-correction），两次微修正之间必须间隔足够步数：

$$\text{trigger\_micro\_final} = \text{trigger\_micro} \;\wedge\; \text{is\_stable\_env} \;\wedge\; (t - t_{\text{last\_micro}} \geq T_{\text{cooldown}}) \tag{A61.3}$$

其中：
- $t$ 为当前时间步
- $t_{\text{last\_micro}}$ 为上一次微修正的时间步
- $T_{\text{cooldown}} > 0$ 为冷却期（通常设为10-20步）

**过度修正的风险**：

过度修正是指微修正过于频繁，导致系统在最优解附近来回振荡而无法收敛。具体风险包括：

1. **振荡风险**：每次微修正都基于当前梯度方向，但梯度方向在最优解附近可能频繁变化，过于频繁的修正导致参数在最优解附近振荡
2. **过拟合风险**：微修正基于最近的观测数据，过于频繁的修正使模型过度拟合最近的噪声
3. **资源浪费**：每次微修正都需要计算梯度和更新参数，过于频繁的修正浪费计算资源

**冷却期的设计原则**：

冷却期 $T_{\text{cooldown}}$ 的选择需要平衡两个因素：
- 太短（<5步）：无法有效防止振荡
- 太长（>50步）：错过精细调优的机会

经验上，$T_{\text{cooldown}}$ 应与表征更新的时间常数 $\tau_r$（引用阶段14，A14.1）同量级，确保每次微修正之间表征已充分更新。

**冷却期的自适应调整**：

冷却期可以根据微修正的效果自适应调整：

$$T_{\text{cooldown}}(t) = T_{\text{base}} \cdot \left(1 + \frac{\Delta F_{\text{last\_micro}}}{\theta_{\text{stable}}}\right) \tag{A61.3-adaptive}$$

其中 $\Delta F_{\text{last\_micro}}$ 为上一次微修正后的自由能变化量。如果上次微修正效果显著（$\Delta F_{\text{last\_micro}}$ 大），则缩短冷却期（微修正有效，可以更频繁）；如果上次微修正效果不显著，则延长冷却期（微修正无效，需要更长时间观察）。

- **输入**：微修正触发标志，环境稳定标志，当前时间步，上次微修正时间步，冷却期
- **输出**：最终触发标志 $\text{trigger\_micro\_final} \in \{0, 1\}$
- **约束**：$T_{\text{cooldown}} > 0$；$t \geq t_{\text{last\_micro}}$

> **白话翻译**：触发频率限制就是"别修太勤"——两次微修正之间必须间隔足够步数。修太勤会导致系统在最优解附近来回振荡，反而越修越差。冷却期就像吃药的间隔——两次服药之间要等足够时间，让药效充分发挥后再决定是否需要再吃。

---

### 第四部分：完整触发流程

#### 2.4 微修正触发完整流程

微修正触发的完整判定流程如下：

```
输入：自由能梯度 ∂F/∂μ，自由能 F_t, F_{t-1}，误差序列 {ε_t}，当前时间步 t

步骤1：计算梯度范数 ‖∂F/∂μ‖
步骤2：计算自由能变化量 ΔF = |F_t - F_{t-1}|
步骤3：判定梯度条件：‖∂F/∂μ‖ < θ_micro ?
步骤4：判定稳定性条件：ΔF < θ_stable ?
步骤5：判定环境稳定性：var(ε_t, W) < θ_var ?
步骤6：判定冷却期：t - t_last_micro ≥ T_cooldown ?
步骤7：综合判定：trigger_micro_final = 步骤3 ∧ 步骤4 ∧ 步骤5 ∧ 步骤6
步骤8：若触发，记录 t_last_micro = t
```

**触发条件的优先级**：

| 优先级 | 条件 | 不满足时的动作 |
|--------|------|--------------|
| 1 | 环境稳定性 | 等待环境稳定，不触发任何更新 |
| 2 | 梯度条件 | 触发增强更新（增大步长的梯度下降） |
| 3 | 稳定性条件 | 继续正常更新（标准梯度下降） |
| 4 | 冷却期 | 等待冷却期结束 |

**与A58.2（结构重构触发）的关系**：

当A61.1的梯度条件不满足且误差持续较大时，系统应进入结构重构触发判定（阶段64）。微修正触发和结构重构触发构成互补的触发机制：

| 条件 | 触发动作 |
|------|---------|
| 梯度小 + 自由能稳定 + 环境稳定 + 冷却期结束 | 微修正（A61.1） |
| 梯度大 + 误差持续大 | 增强更新或结构重构（A58.2 → 阶段64） |
| 环境不稳定 | 暂停更新，等待环境稳定 |

- **输入**：所有触发条件的中间结果
- **输出**：最终触发决策
- **约束**：微修正与结构重构互斥（同一时间步只能触发一种）

> **白话翻译**：完整触发流程就是四道关卡——环境稳不稳、梯度小不小、自由能稳不稳、冷却期过了没。四道关卡全部通过才触发微修正。任何一道关卡不通过，都有对应的处理策略——环境不稳定就等，梯度大就做更大幅度的更新，自由能波动就继续正常更新，冷却期没过就等。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A61.1 | $\text{trigger\_micro} = \mathbb{1}[\|\partial F / \partial \mu\| < \theta_{\text{micro}} \wedge \Delta F < \theta_{\text{stable}}]$ | 梯度、自由能变化、阈值 | 触发标志 | $\theta_{\text{micro}}, \theta_{\text{stable}} > 0$ |
| A61.2 | $\text{is\_stable\_env} = \mathbb{1}[\text{var}(\varepsilon_t, W) < \theta_{\text{var}}]$ | 误差序列、窗口、阈值 | 稳定标志 | $W \geq 10$；$\theta_{\text{var}} > 0$ |
| A61.3 | $\text{trigger\_final} = \text{trigger\_micro} \wedge \text{is\_stable\_env} \wedge (t - t_{\text{last}} \geq T_{\text{cooldown}})$ | 所有中间标志、时间步 | 最终触发标志 | $T_{\text{cooldown}} > 0$ |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np

# 引用阶段60的数据类
from stage_060 import ErrorValidationResult


# ============================================================
# 阶段61 枚举：触发状态
# ============================================================
class TriggerStatus(Enum):
    """微修正触发状态。"""
    NOT_TRIGGERED = "not_triggered"          # 未触发
    TRIGGERED = "triggered"                  # 已触发
    COOLDOWN = "cooldown"                    # 冷却期中
    UNSTABLE_ENV = "unstable_env"            # 环境不稳定
    HIGH_GRADIENT = "high_gradient"          # 梯度过大


# ============================================================
# 阶段61 数据类：微修正触发
# ============================================================
@dataclass(frozen=True)
class MicroAdjustTrigger:
    """微修正触发条件的完整描述。

    组合阶段60的ErrorValidationResult，在验证结果的基础上增加
    微修正触发的判定信息——梯度范数、自由能稳定性、环境稳定性、
    冷却期状态。

    Attributes:
        validation_result: 阶段60的验证结果（组合字段）
        gradient_norm: 自由能梯度范数 ‖∂F/∂μ‖
        free_energy_delta: 自由能变化量 ΔF
        env_variance: 环境方差 var(ε_t, W)
        theta_micro: 微修正梯度阈值
        theta_stable: 自由能稳定性阈值
        theta_var: 环境方差阈值
        cooldown_steps: 冷却期步数 T_cooldown
        steps_since_last: 距上次微修正的步数
        trigger_status: 当前触发状态
        gradient_condition_met: 梯度条件是否满足
        stability_condition_met: 稳定性条件是否满足
        env_stable: 环境是否稳定
        cooldown_elapsed: 冷却期是否已过

    Invariants:
        - theta_micro > 0
        - theta_stable > 0
        - theta_var > 0
        - cooldown_steps > 0
        - steps_since_last >= 0
    """
    validation_result: ErrorValidationResult
    gradient_norm: float = 0.0
    free_energy_delta: float = 0.0
    env_variance: float = 0.0
    theta_micro: float = 0.1
    theta_stable: float = 0.01
    theta_var: float = 0.5
    cooldown_steps: int = 10
    steps_since_last: int = 100
    trigger_status: TriggerStatus = TriggerStatus.NOT_TRIGGERED
    gradient_condition_met: bool = False
    stability_condition_met: bool = False
    env_stable: bool = True
    cooldown_elapsed: bool = True

    def __post_init__(self):
        if self.theta_micro <= 0:
            raise ValueError(
                f"微修正梯度阈值必须 > 0，收到 {self.theta_micro}"
            )
        if self.theta_stable <= 0:
            raise ValueError(
                f"自由能稳定性阈值必须 > 0，收到 {self.theta_stable}"
            )
        if self.theta_var <= 0:
            raise ValueError(
                f"环境方差阈值必须 > 0，收到 {self.theta_var}"
            )
        if self.cooldown_steps <= 0:
            raise ValueError(
                f"冷却期步数必须 > 0，收到 {self.cooldown_steps}"
            )
        if self.steps_since_last < 0:
            raise ValueError(
                f"距上次微修正步数必须 >= 0，收到 {self.steps_since_last}"
            )

    @property
    def should_trigger(self) -> bool:
        """是否应触发微修正（A61.3完整条件）。"""
        return (
            self.gradient_condition_met
            and self.stability_condition_met
            and self.env_stable
            and self.cooldown_elapsed
        )

    @property
    def trigger_reason(self) -> str:
        """触发或未触发的原因。"""
        if self.should_trigger:
            return "所有条件满足，触发微修正"
        if not self.env_stable:
            return "环境不稳定，暂停微修正"
        if not self.gradient_condition_met:
            return "梯度范数过大，应考虑增强更新或结构重构"
        if not self.stability_condition_met:
            return "自由能未稳定，继续正常更新"
        if not self.cooldown_elapsed:
            return "冷却期未结束，等待"
        return "未知原因"


# ============================================================
# 阶段61 核心接口：微修正触发判定器
# ============================================================
class MicroAdjustTriggerChecker:
    """阶段61的核心接口：微修正触发判定器。

    判定微修正触发条件——梯度范数、自由能稳定性、
    环境稳定性和冷却期四个条件的联合判定。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 微修正仅在梯度小、自由能稳定、环境稳定、冷却期过后触发
        - 触发条件与结构重构触发条件互斥
    """

    def check_gradient_condition(
        self,
        gradient: np.ndarray,
        theta_micro: float
    ) -> bool:
        """检查梯度条件 ‖∂F/∂μ‖ < θ_micro（A61.1部分1）。

        Args:
            gradient: 自由能梯度 ∂F/∂μ
            theta_micro: 微修正梯度阈值

        Returns:
            梯度条件是否满足
        """
        return float(np.linalg.norm(gradient)) < theta_micro

    def check_stability_condition(
        self,
        free_energy_delta: float,
        theta_stable: float
    ) -> bool:
        """检查稳定性条件 ΔF < θ_stable（A61.1部分2）。

        Args:
            free_energy_delta: 自由能变化量 ΔF
            theta_stable: 自由能稳定性阈值

        Returns:
            稳定性条件是否满足
        """
        return free_energy_delta < theta_stable

    def check_env_stability(
        self,
        error_history: np.ndarray,
        window: int,
        theta_var: float
    ) -> bool:
        """检查环境稳定性 var(ε_t, W) < θ_var（A61.2）。

        Args:
            error_history: 误差历史序列
            window: 滑动窗口长度
            theta_var: 环境方差阈值

        Returns:
            环境是否稳定

        Raises:
            ValueError: 当窗口长度 < 10 时
        """
        if window < 10:
            raise ValueError(f"窗口长度必须 >= 10，收到 {window}")
        if len(error_history) < window:
            window = len(error_history)
        window_errors = error_history[-window:]
        mean = float(np.mean(window_errors))
        variance = float(np.sum((window_errors - mean) ** 2) / (window - 1))
        return variance < theta_var

    def check_cooldown(
        self,
        steps_since_last: int,
        cooldown_steps: int
    ) -> bool:
        """检查冷却期 t - t_last ≥ T_cooldown（A61.3部分3）。

        Args:
            steps_since_last: 距上次微修正的步数
            cooldown_steps: 冷却期步数

        Returns:
            冷却期是否已过
        """
        return steps_since_last >= cooldown_steps

    def check_trigger(
        self,
        trigger: MicroAdjustTrigger
    ) -> MicroAdjustTrigger:
        """执行完整的微修正触发判定（A61.1-A61.3）。

        依次检查梯度条件、稳定性条件、环境稳定性和冷却期，
        返回更新后的MicroAdjustTrigger。

        Args:
            trigger: 当前触发状态

        Returns:
            更新后的MicroAdjustTrigger
        """
        gradient_met = self.check_gradient_condition(
            np.array([trigger.gradient_norm]), trigger.theta_micro
        )
        stability_met = self.check_stability_condition(
            trigger.free_energy_delta, trigger.theta_stable
        )
        env_stable = self.check_env_stability(
            np.array([trigger.env_variance]), 10, trigger.theta_var
        )
        cooldown_elapsed = self.check_cooldown(
            trigger.steps_since_last, trigger.cooldown_steps
        )
        return MicroAdjustTrigger(
            validation_result=trigger.validation_result,
            gradient_norm=trigger.gradient_norm,
            free_energy_delta=trigger.free_energy_delta,
            env_variance=trigger.env_variance,
            theta_micro=trigger.theta_micro,
            theta_stable=trigger.theta_stable,
            theta_var=trigger.theta_var,
            cooldown_steps=trigger.cooldown_steps,
            steps_since_last=trigger.steps_since_last,
            trigger_status=TriggerStatus.TRIGGERED
            if (gradient_met and stability_met and env_stable and cooldown_elapsed)
            else TriggerStatus.NOT_TRIGGERED,
            gradient_condition_met=gradient_met,
            stability_condition_met=stability_met,
            env_stable=env_stable,
            cooldown_elapsed=cooldown_elapsed,
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `ErrorValidationResult` | ← 阶段60 | 作为触发判定的验证结果基础（组合字段） |
| **输入** | A58.1（小误差→微修正触发） | ← 阶段58 | 微修正触发的理论基础 |
| **输入** | A5.17（自由能梯度流） | ← 阶段5 | 梯度范数条件的理论依据 |
| **输入** | A56.3（在线分布偏移检测） | ← 阶段56 | 环境稳定性判定的参考 |
| **输出** | `MicroAdjustTrigger` | → 阶段62 | 微修正触发状态，作为微修正算法的输入 |
| **输出** | A61.1-A61.3（触发条件定义） | → 阶段62-63 | 微修正触发的精确条件 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 梯度范数计算 | `jax.grad`自动微分计算自由能梯度，`jax.numpy.linalg.norm`计算梯度范数 |
| NumPy | 方差计算 | 滑动窗口内的样本方差计算 |
| Pydantic | 数据验证 | 支撑MicroAdjustTrigger的运行时类型检查 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V61.1 | 梯度条件正确性 | $\|\partial F / \partial \mu\| < \theta_{\text{micro}}$ 时梯度条件满足 | 条件测试 |
| V61.2 | 稳定性条件正确性 | $\Delta F < \theta_{\text{stable}}$ 时稳定性条件满足 | 条件测试 |
| V61.3 | 环境稳定性正确性 | $\text{var}(\varepsilon_t, W) < \theta_{\text{var}}$ 时环境稳定 | 方差计算验证 |
| V61.4 | 冷却期正确性 | $t - t_{\text{last}} \geq T_{\text{cooldown}}$ 时冷却期通过 | 计数验证 |
| V61.5 | 完整触发判定 | 四个条件同时满足时触发 | 组合条件测试 |
| V61.6 | 梯度大时不触发 | $\|\partial F / \partial \mu\| \geq \theta_{\text{micro}}$ 时不触发微修正 | 条件测试 |
| V61.7 | 自由能波动时不触发 | $\Delta F \geq \theta_{\text{stable}}$ 时不触发微修正 | 条件测试 |
| V61.8 | 环境不稳定时不触发 | $\text{var}(\varepsilon_t, W) \geq \theta_{\text{var}}$ 时不触发微修正 | 条件测试 |
| V61.9 | 冷却期未过时不触发 | $t - t_{\text{last}} < T_{\text{cooldown}}$ 时不触发微修正 | 条件测试 |
| V61.10 | MicroAdjustTrigger组合 | `trigger.validation_result` 类型为 `ErrorValidationResult` | 类型检查 |
| V61.11 | 非法阈值拒绝 | $\theta_{\text{micro}} \leq 0$ 等抛出 `ValueError` | 异常测试 |
| V61.12 | 自适应方差阈值 | $\theta_{\text{var}}(t)$ 随历史方差自适应调整 | 自适应验证 |
| V61.13 | 触发原因可解释 | `trigger_reason` 正确反映未触发原因 | 字符串验证 |
| V61.14 | 长期运行稳定性 | 1000步循环中触发频率合理（不过频不过疏） | 长期运行测试 |

---

> **阶段61完** | 阶段61建立了微修正触发条件的精确判定规则：梯度范数条件 $\|\partial F / \partial \mu\| < \theta_{\text{micro}}$（A61.1），自由能稳定性条件 $\Delta F < \theta_{\text{stable}}$（A61.1），环境稳定性判定 $\text{var}(\varepsilon_t, W) < \theta_{\text{var}}$（A61.2），触发频率限制 $t - t_{\text{last}} \geq T_{\text{cooldown}}$（A61.3），封装为MicroAdjustTrigger数据类组合ErrorValidationResult。下一步：阶段62将定义微修正的具体算法。
