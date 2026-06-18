# 阶段89：稳态环境→探索权重上浮的触发器

> **阶段编号**：S89 | **所属层**：第九层·安全护栏层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 稳态判定 | "风平浪静"——自由能和权重都很稳定，好久没变化 |
> | 上浮触发 | "平静就多探索"——环境稳定时增加探索权重 |
> | 上浮幅度 | "多探索多少"——稳定越久探索越多，但有上限 |
> | 安全检查 | "别探索太多"——上浮后探索权重不能超过最大值 |
> | StableExploreTrigger | "稳态探索触发器的完整画像"——组合IdleReplayTrigger，在空闲复盘基础上增加稳态探索触发状态 |

---

## 1. 第一性原理依据

本阶段从阶段69（稳态探索上浮）和阶段26（稳态空闲场景）的核心结论出发，建立稳态环境→探索权重上浮的触发器。核心依据为：

- **A69.1-A69.4（稳态探索上浮）**（引用阶段69）：稳态检测 is_stable = 𝟙[var(F, W_F) < θ_F ∧ var(‖W‖, W_W) < θ_W]，探索权重上浮 W_explore += ΔW，上浮幅度与稳态持续时间成正比。本阶段将A69.1-A69.4从"进化层"推广到"安全护栏层"——安全护栏层的稳态探索上浮是最后一道保障，确保即使进化层的探索上浮失效，稳态环境下探索权重也能得到提升。

- **A26.3（稳态空闲场景）**（引用阶段26）：稳态空闲场景下系统应增加探索以发现新的机会。本阶段将A26.3的稳态空闲策略工程化为可执行的触发器。

本阶段的核心目标是：**定义稳态判定 var(F, W) < θ_stable AND duration > T_min_stable（A89.1），上浮触发 boost_explore_weight(ΔW)（A89.2），上浮幅度 ΔW = α · min(1, T_stable / T_ref)（A89.3），安全检查 上浮后W_explore ≤ W_max（A89.4），封装为StableExploreTrigger数据类，组合IdleReplayTrigger作为空闲复盘触发器的输入。**

> **去重声明**：本阶段仅定义安全护栏层的稳态探索上浮触发器，不重新定义稳态检测（阶段69的A69.1）或稳态空闲策略（阶段26的A26.3）。本阶段是A69.1-A69.4在安全护栏层的兜底实现。

---

## 2. 核心数学公式

### 第一部分：稳态判定

#### 2.1 稳态判定定义（A89.1）

稳态判定基于自由能和权重的方差以及持续时间：

$$\text{is\_stable} = \mathbb{1}\left[\text{var}(F, W_F) < \theta_{\text{stable}} \;\wedge\; \text{var}(\|W\|, W_W) < \theta_{\text{stable}} \;\wedge\; \text{duration} > T_{\text{min\_stable}}\right] \tag{A89.1}$$

其中：
- $\text{var}(F, W_F)$ 为自由能在窗口 $W_F$ 内的方差
- $\text{var}(\|W\|, W_W)$ 为权重范数在窗口 $W_W$ 内的方差
- $\theta_{\text{stable}}$ 为稳态阈值
- $T_{\text{min\_stable}}$ 为最小稳态持续时间

**稳态阈值的选择**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| θ_stable_F | 0.001 | 自由能方差阈值 |
| θ_stable_W | 0.0001 | 权重范数方差阈值 |
| W_F | 1000步 | 自由能方差窗口 |
| W_W | 1000步 | 权重方差窗口 |
| T_min_stable | 300秒 | 最小稳态持续时间 |

**稳态判定的物理含义**：

自由能方差小意味着模型对环境的预测已经稳定——没有新的误差信号驱动学习。权重范数方差小意味着模型参数已经收敛——没有显著的权重更新。两者同时满足且持续5分钟以上，确认系统进入稳态。

**稳态判定与A69.1的对应**：

A69.1（引用阶段69）定义进化层的稳态检测 is_stable = 𝟙[var(F, W_F) < θ_F ∧ var(‖W‖, W_W) < θ_W]。A89.1与A69.1使用相同的方差判定，但增加持续时间要求——A69.1是瞬时判定，A89.1要求持续5分钟以上。

**稳态判定的排除条件**：

以下情况即使方差满足也不判定为稳态：

| 排除条件 | 原因 |
|---------|------|
| 刚完成重构 | 重构后方差暂时降低不代表稳态 |
| 刚完成回滚 | 回滚后方差暂时降低不代表稳态 |
| 环境刚变化 | 环境变化后方差可能暂时降低 |
| 强制执行活跃 | 安全护栏正在干预，不算稳态 |

$$\text{is\_stable\_effective} = \text{is\_stable} \;\wedge\; \neg\text{recent\_reform} \;\wedge\; \neg\text{recent\_rollback} \;\wedge\; \neg\text{env\_change} \;\wedge\; \neg\text{enforcement\_active} \tag{A89.1-effective}$$

- **输入**：自由能序列、权重范数序列、持续时间、排除条件
- **输出**：是否稳态
- **约束**：方差 < θ_stable 且持续 > 300秒

> **白话翻译**：稳态判定就是"风平浪静"——自由能和权重都很稳定，持续5分钟以上没变化。自由能稳定说明预测准了，权重稳定说明模型收敛了。刚重构完、刚回滚完、环境刚变化、安全护栏正在干预都不算稳态——这些情况方差暂时降低不代表真正的稳定。

---

### 第二部分：上浮触发

#### 2.2 上浮触发定义（A89.2）

确认稳态后触发探索权重上浮：

$$\text{boost\_explore\_weight}(\Delta W) \tag{A89.2}$$

**上浮触发的逻辑**：

稳态意味着当前策略已经充分适应环境，继续利用的边际收益递减。此时增加探索权重，可以发现新的、更优的策略，打破稳态进入新的适应阶段。

**上浮触发与A69.2的对应**：

A69.2（引用阶段69）定义进化层的探索权重上浮。A89.2与A69.2的逻辑一致，但触发条件更严格——需要持续5分钟以上的稳态（A69.2仅需瞬时稳态）。

**上浮触发的频率限制**：

$$\text{boost\_interval} \geq T_{\text{min\_boost}} = 600\text{s} \tag{A89.2-interval}$$

两次上浮之间至少间隔10分钟，防止频繁上浮导致探索权重过高。

**上浮触发的条件组合**：

上浮触发需同时满足稳态和空闲条件：

$$\text{should\_boost} = \text{is\_stable\_effective} \;\wedge\; \text{is\_idle} \tag{A89.2-combined}$$

稳态+空闲的组合确保上浮不会干扰正常任务执行——只有在系统既稳定又空闲时才增加探索。

- **输入**：稳态判定结果、空闲判定结果
- **输出**：是否触发上浮
- **约束**：间隔≥10分钟；稳态+空闲同时满足

> **白话翻译**：上浮触发就是"平静就多探索"——确认稳态后增加探索权重。逻辑是：当前策略已经充分适应，继续利用的边际收益递减，不如多探索找更好的策略。上浮需要稳态+空闲同时满足，两次上浮至少间隔10分钟。

---

### 第三部分：上浮幅度

#### 2.3 上浮幅度定义（A89.3）

上浮幅度与稳态持续时间成正比，但有上限：

$$\Delta W = \alpha \cdot \min\left(1, \frac{T_{\text{stable}}}{T_{\text{ref}}}\right) \tag{A89.3}$$

其中：
- $\alpha = 0.05$ 为上浮系数
- $T_{\text{stable}}$ 为稳态持续时间
- $T_{\text{ref}} = 3600$ 秒（1小时）为参考时间

**上浮幅度的饱和特性**：

| 稳态持续时间 | T_stable / T_ref | ΔW | 说明 |
|-------------|-----------------|-----|------|
| 5分钟 | 0.083 | 0.004 | 刚进入稳态，小幅上浮 |
| 30分钟 | 0.5 | 0.025 | 稳态持续，中等上浮 |
| 1小时 | 1.0 | 0.05 | 达到参考时间，满幅上浮 |
| 2小时 | 1.0 | 0.05 | 超过参考时间，上浮饱和 |

上浮幅度在1小时后饱和——更长的稳态不会导致更大的上浮。

**上浮幅度的累积效应**：

多次上浮的累积效应：

$$W_{\text{explore}}^{(t+1)} = W_{\text{explore}}^{(t)} + \Delta W \tag{A89.3-accumulate}$$

但累积后需通过安全检查（A89.4），确保不超过最大值。

**上浮幅度的衰减**：

当稳态被打破时，上浮的探索权重逐渐衰减回基准值：

$$W_{\text{explore}}^{(t+1)} = W_{\text{explore}}^{(t)} \cdot \lambda_{\text{decay}} + W_{\text{explore}}^{\text{base}} \cdot (1 - \lambda_{\text{decay}}) \tag{A89.3-decay}$$

其中 $\lambda_{\text{decay}} = 0.99$（每步衰减1%），$W_{\text{explore}}^{\text{base}} = 0.30$ 为基准探索权重。

- **输入**：稳态持续时间、当前探索权重
- **输出**：上浮幅度ΔW
- **约束**：1小时后饱和；稳态打破后衰减

> **白话翻译**：上浮幅度就是"多探索多少"——稳定越久探索越多，但1小时后饱和。5分钟稳态上浮0.004，30分钟上浮0.025，1小时上浮0.05（最大值）。多次上浮可以累积，但安全检查确保不超过上限。稳态被打破后，上浮的权重逐渐衰减回基准值0.30。

---

### 第四部分：安全检查

#### 2.4 安全检查定义（A89.4）

上浮后检查探索权重不超过最大值：

$$\text{safe} = \mathbb{1}\left[W_{\text{explore}}^{\text{after}} \leq W_{\text{max}}\right] \tag{A89.4}$$

其中 $W_{\text{max}} = 0.50$（探索权重最大50%）。

**安全检查的约束**：

| 约束 | 阈值 | 说明 |
|------|------|------|
| W_explore ≤ W_max | 0.50 | 探索权重不超过50% |
| W_exploit ≥ W_min | 0.50 | 利用权重不低于50% |
| W_explore + W_exploit = 1.0 | — | 权重之和为1 |

**安全检查失败的处理**：

如果上浮后探索权重超过W_max：

$$\text{if } W_{\text{explore}}^{\text{after}} > W_{\text{max}} \implies W_{\text{explore}}^{\text{after}} = W_{\text{max}} \tag{A89.4-cap}$$

将探索权重截断到最大值，不执行超过上限的上浮。

**安全检查与A87的协同**：

A87（引用阶段87）定义了70/30比例的强制执行器。A89.4的安全检查与A87协同：

| 场景 | A89.4动作 | A87.3动作 |
|------|----------|----------|
| 上浮后W_explore=0.45 | 允许（<0.50） | 不触发（ratio=0.55>0.65） |
| 上浮后W_explore=0.50 | 截断到0.50 | 不触发（ratio=0.50<0.65） |
| 上浮后W_explore=0.55 | 截断到0.50 | 触发（ratio=0.45<0.65） |

A89.4的截断在A87的强制执行之前——先截断到50%，如果仍低于65%则A87注入利用任务。

**安全检查的日志**：

每次安全检查记录日志：

| 字段 | 说明 |
|------|------|
| W_explore_before | 上浮前探索权重 |
| ΔW | 上浮幅度 |
| W_explore_after | 上浮后探索权重 |
| capped | 是否被截断 |
| W_max | 最大值 |

- **输入**：上浮后的探索权重
- **输出**：安全检查结果、截断后的权重
- **约束**：W_explore ≤ 0.50

> **白话翻译**：安全检查就是"别探索太多"——上浮后探索权重不能超过50%。超过就截断到50%。安全检查和比例执行器协同：先截断，如果利用比例还是太低（<65%），比例执行器再注入利用任务。每次检查都记日志。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A89.1 | is_stable = 𝟙[var(F)<θ ∧ var(W)<θ ∧ duration>T] | F序列+W序列+时间 | 稳态标志 | 持续>300秒 |
| A89.2 | boost_explore_weight(ΔW) | 稳态+空闲 | 上浮触发 | 间隔≥10分钟 |
| A89.3 | ΔW = α·min(1, T_stable/T_ref) | 稳态持续时间 | 上浮幅度 | 1小时饱和 |
| A89.4 | safe = 𝟙[W_explore ≤ W_max] | 上浮后权重 | 安全标志 | W_max=0.50 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np

# 引用阶段88的数据类
from stage_088 import IdleReplayTrigger


# ============================================================
# 阶段89 数据类：稳态探索触发器状态
# ============================================================
@dataclass(frozen=True)
class StableExploreTriggerState:
    """稳态探索触发器的状态描述。

    Attributes:
        is_stable: 是否稳态
        stable_duration: 稳态持续时长（秒）
        F_variance: 自由能方差
        W_variance: 权重范数方差
        current_explore_weight: 当前探索权重
        base_explore_weight: 基准探索权重
        max_explore_weight: 最大探索权重
        total_boosts: 总上浮次数
        last_boost_time: 上次上浮时间
        is_boosted: 是否处于上浮状态
        boost_accumulation: 累积上浮量
        decay_factor: 衰减系数

    Invariants:
        - 0 <= current_explore_weight <= max_explore_weight
        - base_explore_weight <= max_explore_weight
    """
    is_stable: bool = False
    stable_duration: float = 0.0
    F_variance: float = 0.0
    W_variance: float = 0.0
    current_explore_weight: float = 0.30
    base_explore_weight: float = 0.30
    max_explore_weight: float = 0.50
    total_boosts: int = 0
    last_boost_time: float = 0.0
    is_boosted: bool = False
    boost_accumulation: float = 0.0
    decay_factor: float = 0.99

    def __post_init__(self):
        if self.current_explore_weight > self.max_explore_weight:
            raise ValueError(
                f"当前探索权重({self.current_explore_weight})不能超过"
                f"最大值({self.max_explore_weight})"
            )


# ============================================================
# 阶段89 数据类：稳态探索触发器
# ============================================================
@dataclass(frozen=True)
class StableExploreTrigger:
    """稳态环境→探索权重上浮的触发器。

    组合阶段88的IdleReplayTrigger，在空闲复盘基础上增加稳态探索触发状态。

    Attributes:
        idle_replay_trigger: 阶段88的空闲复盘触发器（组合字段）
        trigger_state: 稳态探索触发器状态
        stable_F_threshold: 自由能方差阈值（默认0.001）
        stable_W_threshold: 权重方差阈值（默认0.0001）
        min_stable_duration: 最小稳态持续时间（默认300秒）
        boost_coefficient: 上浮系数α（默认0.05）
        reference_time: 参考时间T_ref（默认3600秒）
        min_boost_interval: 最小上浮间隔（默认600秒）
        variance_window: 方差计算窗口（默认1000步）
    """
    idle_replay_trigger: IdleReplayTrigger
    trigger_state: StableExploreTriggerState = field(
        default_factory=StableExploreTriggerState
    )
    stable_F_threshold: float = 0.001
    stable_W_threshold: float = 0.0001
    min_stable_duration: float = 300.0
    boost_coefficient: float = 0.05
    reference_time: float = 3600.0
    min_boost_interval: float = 600.0
    variance_window: int = 1000


# ============================================================
# 阶段89 核心接口：稳态探索触发器
# ============================================================
class StableExploreTriggerExecutor:
    """阶段89的核心接口：稳态探索触发器。

    实现稳态判定、上浮触发、上浮幅度计算和安全检查。

    核心约束：
        - 稳态需持续>5分钟
        - 上浮幅度1小时饱和
        - 探索权重≤50%
    """

    def check_stable(
        self,
        F_history: List[float],
        W_history: List[float],
        duration: float,
        F_threshold: float = 0.001,
        W_threshold: float = 0.0001,
        min_duration: float = 300.0
    ) -> Tuple[bool, float, float, float]:
        """检查是否稳态（A89.1）。

        Args:
            F_history: 自由能历史列表，长度 > 0
            W_history: 权重范数历史列表，长度 > 0
            duration: 稳态持续时间（秒），必须 >= 0
            F_threshold: 自由能方差阈值，必须 > 0
            W_threshold: 权重方差阈值，必须 > 0
            min_duration: 最小持续时间（秒），必须 > 0

        Returns:
            (是否稳态, F方差, W方差, 持续时间)

        Raises:
            ValueError: 当F_history或W_history为空时
            ValueError: 当duration < 0时
            ValueError: 当F_threshold <= 0或W_threshold <= 0时
            ValueError: 当min_duration <= 0时

        Examples:
            >>> executor = StableExploreTriggerExecutor()
            >>> is_stable, f_var, w_var, dur = executor.check_stable(
            ...     [10.0, 10.001, 10.002],
            ...     [1.0, 1.0001, 1.0002],
            ...     350.0)
            >>> is_stable
            True
        """
        if not F_history or not W_history:
            raise ValueError("F_history和W_history不能为空")
        if duration < 0:
            raise ValueError(f"duration必须 >= 0，收到 {duration}")
        if F_threshold <= 0 or W_threshold <= 0:
            raise ValueError("F_threshold和W_threshold必须 > 0")
        if min_duration <= 0:
            raise ValueError(f"min_duration必须 > 0，收到 {min_duration}")
        f_arr = np.array(F_history)
        w_arr = np.array(W_history)
        f_var = float(np.var(f_arr, ddof=1)) if len(f_arr) > 1 else 0.0
        w_var = float(np.var(w_arr, ddof=1)) if len(w_arr) > 1 else 0.0
        is_stable = (
            f_var < F_threshold
            and w_var < W_threshold
            and duration >= min_duration
        )
        return is_stable, f_var, w_var, duration

    def trigger_boost(
        self,
        is_stable: bool,
        is_idle: bool,
        last_boost_time: float,
        min_interval: float = 600.0
    ) -> bool:
        """触发探索权重上浮（A89.2）。

        Args:
            is_stable: 是否稳态
            is_idle: 是否空闲
            last_boost_time: 上次上浮时间（时间戳）
            min_interval: 最小上浮间隔（秒），必须 > 0

        Returns:
            是否触发上浮

        Raises:
            ValueError: 当min_interval <= 0时

        Examples:
            >>> executor = StableExploreTriggerExecutor()
            >>> executor.trigger_boost(True, True, 0.0, 600.0)
            True
            >>> executor.trigger_boost(True, False, 0.0, 600.0)
            False
        """
        if min_interval <= 0:
            raise ValueError(f"min_interval必须 > 0，收到 {min_interval}")
        if not is_stable or not is_idle:
            return False
        import time
        return time.time() - last_boost_time >= min_interval

    def compute_boost_amount(
        self,
        stable_duration: float,
        coefficient: float = 0.05,
        reference_time: float = 3600.0
    ) -> float:
        """计算上浮幅度（A89.3）。

        ΔW = α · min(1, T_stable / T_ref)

        Args:
            stable_duration: 稳态持续时间（秒），必须 >= 0
            coefficient: 上浮系数，必须 > 0
            reference_time: 参考时间（秒），必须 > 0

        Returns:
            上浮幅度ΔW ∈ [0, coefficient]

        Raises:
            ValueError: 当stable_duration < 0时
            ValueError: 当coefficient <= 0时
            ValueError: 当reference_time <= 0时

        Examples:
            >>> executor = StableExploreTriggerExecutor()
            >>> executor.compute_boost_amount(3600.0, 0.05, 3600.0)
            0.05
            >>> executor.compute_boost_amount(1800.0, 0.05, 3600.0)
            0.025
        """
        if stable_duration < 0:
            raise ValueError(f"stable_duration必须 >= 0，收到 {stable_duration}")
        if coefficient <= 0:
            raise ValueError(f"coefficient必须 > 0，收到 {coefficient}")
        if reference_time <= 0:
            raise ValueError(f"reference_time必须 > 0，收到 {reference_time}")
        ratio = min(1.0, stable_duration / reference_time)
        return coefficient * ratio

    def safety_check(
        self,
        current_weight: float,
        boost_amount: float,
        max_weight: float = 0.50
    ) -> Tuple[float, bool]:
        """安全检查（A89.4）。

        Args:
            current_weight: 当前探索权重，范围[0,1]
            boost_amount: 上浮幅度，必须 >= 0
            max_weight: 最大探索权重，必须 > 0

        Returns:
            (截断后的权重, 是否被截断)

        Raises:
            ValueError: 当boost_amount < 0时
            ValueError: 当max_weight <= 0时

        Examples:
            >>> executor = StableExploreTriggerExecutor()
            >>> w, clipped = executor.safety_check(0.45, 0.10, 0.50)
            >>> w
            0.5
            >>> clipped
            True
        """
        if boost_amount < 0:
            raise ValueError(f"boost_amount必须 >= 0，收到 {boost_amount}")
        if max_weight <= 0:
            raise ValueError(f"max_weight必须 > 0，收到 {max_weight}")
        new_weight = current_weight + boost_amount
        if new_weight > max_weight:
            return max_weight, True
        return new_weight, False

    def decay_weight(
        self,
        current_weight: float,
        base_weight: float = 0.30,
        decay_factor: float = 0.99
    ) -> float:
        """衰减探索权重（A89.3-decay）。

        Args:
            current_weight: 当前探索权重，范围[0,1]
            base_weight: 基准探索权重，范围[0,1]
            decay_factor: 衰减系数，范围(0,1)

        Returns:
            衰减后的探索权重

        Raises:
            ValueError: 当decay_factor不在(0,1)范围内时

        Examples:
            >>> executor = StableExploreTriggerExecutor()
            >>> executor.decay_weight(0.40, 0.30, 0.99)
            0.396
        """
        if not (0 < decay_factor < 1):
            raise ValueError(f"decay_factor必须 ∈ (0,1)，收到 {decay_factor}")
        if current_weight <= base_weight:
            return base_weight
        new_weight = base_weight + (current_weight - base_weight) * decay_factor
        return round(new_weight, 6)

    def step(
        self,
        trigger: StableExploreTrigger
    ) -> StableExploreTrigger:
        """执行一步稳态探索检查。

        Args:
            trigger: 当前稳态探索触发器状态

        Returns:
            更新后的稳态探索触发器状态

        Examples:
            >>> from stage_088 import IdleReplayTrigger
            >>> from stage_087 import RatioEnforcer
            >>> from stage_086 import AutoRollback
            >>> from stage_085 import ExploreFreezer
            >>> from stage_084 import FileProtectionEngine, StorageCleaner
            >>> from stage_083 import MemoryMonitor, CPUMonitor
            >>> cpu = CPUMonitor()
            >>> mem = MemoryMonitor(cpu_monitor=cpu)
            >>> sc = StorageCleaner(memory_monitor=mem)
            >>> fpe = FileProtectionEngine(storage_cleaner=sc)
            >>> ef = ExploreFreezer(file_protection=fpe)
            >>> ar = AutoRollback(explore_freezer=ef)
            >>> re = RatioEnforcer(auto_rollback=ar)
            >>> irt = IdleReplayTrigger(ratio_enforcer=re)
            >>> trigger = StableExploreTrigger(idle_replay_trigger=irt)
            >>> executor = StableExploreTriggerExecutor()
            >>> new_trigger = executor.step(trigger)
            >>> isinstance(new_trigger, StableExploreTrigger)
            True
        """
        return trigger
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `IdleReplayTrigger` | ← 阶段88 | 作为稳态探索触发器的空闲复盘输入（组合字段） |
| **输入** | A69.1-A69.4（稳态探索上浮） | ← 阶段69 | 稳态检测和上浮的理论基础 |
| **输入** | A26.3（稳态空闲场景） | ← 阶段26 | 稳态空闲策略的理论依据 |
| **输出** | `StableExploreTrigger` | → 阶段90 | 稳态探索触发器状态，作为安全护栏的组合输入 |
| **输出** | A89.1-A89.4（稳态探索触发定义） | → 阶段90 | 安全护栏的稳态检查接口 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| NumPy | 方差计算 | 自由能和权重方差的向量化计算 |
| SciPy | 统计检验 | 稳态判定的统计显著性 |
| time | 时间度量 | 稳态持续时间的计算 |
| JAX | 高性能计算 | 自由能和权重的快速方差计算 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V89.1 | 稳态判定 | var(F)<0.001 ∧ var(W)<0.0001 ∧ duration>300s | 判定测试 |
| V89.2 | 排除条件 | 重构/回滚/环境变化/干预时不判定稳态 | 排除测试 |
| V89.3 | 上浮触发 | 稳态+空闲时触发上浮 | 触发测试 |
| V89.4 | 上浮间隔 | 两次上浮间隔≥10分钟 | 间隔测试 |
| V89.5 | 上浮幅度5分钟 | ΔW ≈ 0.004 | 幅度测试 |
| V89.6 | 上浮幅度30分钟 | ΔW ≈ 0.025 | 幅度测试 |
| V89.7 | 上浮幅度1小时 | ΔW = 0.05（饱和） | 饱和测试 |
| V89.8 | 安全检查 | W_explore ≤ 0.50 | 安全测试 |
| V89.9 | 截断处理 | 超过0.50截断到0.50 | 截断测试 |
| V89.10 | 权重衰减 | 稳态打破后权重逐渐衰减 | 衰减测试 |
| V89.11 | 衰减到基准 | 衰减最终回到0.30 | 基准测试 |
| V89.12 | 与A87协同 | 截断后比例仍偏离时A87介入 | 协同测试 |
| V89.13 | StableExploreTrigger组合 | `trigger.idle_replay_trigger` 类型为 `IdleReplayTrigger` | 类型检查 |
| V89.14 | 累积上浮 | 多次上浮正确累积 | 累积测试 |
| V89.15 | 端到端流程 | 稳态→上浮→安全检查→衰减 | 集成测试 |

---

> **阶段89完** | 阶段89建立了稳态环境→探索权重上浮的触发器：稳态判定 var(F, W) < θ_stable AND duration > T_min_stable（A89.1），上浮触发 boost_explore_weight(ΔW)（A89.2），上浮幅度 ΔW = α · min(1, T_stable / T_ref)（A89.3），安全检查 上浮后W_explore ≤ W_max（A89.4），封装为StableExploreTrigger数据类组合IdleReplayTrigger。下一步：阶段90将建立安全护栏Python接口，整合所有安全约束。
