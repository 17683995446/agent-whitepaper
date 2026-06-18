# 阶段69：稳态环境探索权重上浮——空闲时主动探索的增益机制

> **阶段编号**：S69 | **所属层**：第七层·进化适应层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 稳态检测 | "风平浪静"——自由能和权重都稳定，系统处于空闲状态 |
> | 探索权重上浮 | "闲着就多探索"——稳态时增加探索的权重，主动寻找改进机会 |
> | 上浮幅度与稳态时间 | "越闲探索越多"——稳态持续越久，探索权重上浮越大 |
> | 上浮后安全检查 | "探索不能过头"——上浮不超过资源上限，不破坏系统稳定性 |
> | StableExploreBooster | "稳态探索增强器的完整画像"——组合ExploreExploitEnforcer，在比例执行器基础上增加上浮控制 |

---

## 1. 第一性原理依据

本阶段从阶段26（稳态空闲场景）和阶段5（自由能梯度流）的核心结论出发，建立稳态环境探索权重上浮机制。核心依据为：

- **A26.3（稳态空闲场景）**（引用阶段26）：当安全驱力低、收益驱力低时，系统处于稳态空闲场景，适合探索学习。本阶段将A26.3的定性描述转化为定量的探索权重上浮机制——稳态时主动增加探索权重，利用空闲资源寻找改进机会。

- **A5.17（自由能梯度流）**（引用阶段5）：$\mathrm{d}\mu / \mathrm{d}t = -\partial F / \partial \mu$，系统在自由能景观上沿梯度流运动。当梯度趋近于零时（稳态），系统接近驻点，此时探索可以帮助系统跳出局部最优，寻找更好的驻点。

- **A61.2（稳定环境判定）**（引用阶段61）：$\text{var}(\varepsilon_t, W) < \theta_{\text{var}}$，环境方差低于阈值时判定为稳定。本阶段将环境稳定性与探索权重上浮关联——环境越稳定，探索越安全，上浮幅度越大。

- **A68.1-A68.4（探索-利用强制比例）**（引用阶段68）：70/30的强制比例执行。本阶段在70/30的基础上增加动态调整——稳态时探索比例可以从30%上浮到40-50%，利用空闲资源进行更多探索。

本阶段的核心目标是：**定义稳态检测 var(F, W) < θ_stable（A69.1），探索权重上浮 W_explore = W_base + ΔW_explore（A69.2），上浮幅度与稳态持续时间的关系 ΔW = f(T_stable)（A69.3），上浮后的安全检查（A69.4），封装为StableExploreBooster数据类，组合ExploreExploitEnforcer作为比例执行器的输入。**

> **去重声明**：本阶段仅定义稳态环境探索权重上浮机制，不重新定义场景分类（阶段26）或探索-利用比例（阶段68）。上浮是在70/30基础上的动态调整，不改变70/30的长期目标比例。

---

## 2. 核心数学公式

### 第一部分：稳态检测

#### 2.1 稳态检测定义（A69.1）

稳态检测基于自由能和权重的联合稳定性：

$$\text{is\_stable} = \mathbb{1}\left[\text{var}(F, W_F) < \theta_{\text{stable}}^F \;\wedge\; \text{var}(\|W\|, W_W) < \theta_{\text{stable}}^W\right] \tag{A69.1}$$

其中：
- $\text{var}(F, W_F)$ 为自由能在窗口 $W_F$ 内的方差
- $\text{var}(\|W\|, W_W)$ 为权重范数在窗口 $W_W$ 内的方差
- $\theta_{\text{stable}}^F > 0$ 为自由能稳定性阈值
- $\theta_{\text{stable}}^W > 0$ 为权重稳定性阈值

**双条件检测的必要性**：

仅检测自由能稳定性不够——自由能稳定可能是因为系统处于局部最优（梯度小但自由能仍高）。同时检测权重稳定性可以确认系统确实已收敛（权重不再变化），而非卡在鞍点。

**与A61.2（环境稳定性）的区别**：

| 维度 | A61.2（环境稳定性） | A69.1（系统稳态） |
|------|-------------------|-------------------|
| 检测对象 | 误差序列方差 | 自由能方差 + 权重方差 |
| 含义 | 外部环境是否稳定 | 系统内部是否已收敛 |
| 用途 | 微修正触发条件 | 探索权重上浮条件 |
| 方向 | 环境→系统 | 系统→探索策略 |

A61.2检测外部环境的稳定性，A69.1检测系统内部的稳定性。两者可以同时满足（环境稳定且系统已收敛），此时是探索的最佳时机。

- **输入**：自由能序列、权重范数序列、窗口长度、阈值
- **输出**：稳态标志
- **约束**：$\theta_{\text{stable}}^F, \theta_{\text{stable}}^W > 0$

> **白话翻译**：稳态检测就是"风平浪静"——自由能和权重都稳定，系统已经收敛到当前最优了。仅自由能稳定不够——可能卡在鞍点。同时权重也稳定才说明真的收敛了。和环境稳定性（A61.2）的区别是：环境稳定性看外面，系统稳态看里面。

---

### 第二部分：探索权重上浮

#### 2.2 探索权重上浮定义（A69.2）

稳态时增加探索的权重，使系统主动寻找改进机会：

$$W_{\text{explore}} = W_{\text{base}} + \Delta W_{\text{explore}} \tag{A69.2}$$

其中：
- $W_{\text{base}} = 0.3$ 为基础探索权重（70/30比例）
- $\Delta W_{\text{explore}} \geq 0$ 为探索权重上浮量
- $W_{\text{explore}} \leq W_{\max} = 0.5$ 为探索权重上界

**探索权重上浮的物理含义**：

在稳态时，系统已收敛到当前结构下的最优，微修正的收益递减。此时增加探索权重，让系统有更多机会尝试新的结构方向——即使当前结构已经不错，也可能存在更好的结构。

**上浮后的探索-利用比例**：

| 状态 | 探索权重 | 利用权重 | 说明 |
|------|---------|---------|------|
| 非稳态 | 0.30 | 0.70 | 标准70/30比例 |
| 稳态初期 | 0.35 | 0.65 | 轻微上浮 |
| 稳态中期 | 0.40 | 0.60 | 中度上浮 |
| 稳态长期 | 0.45 | 0.55 | 较大上浮 |
| 上浮上限 | 0.50 | 0.50 | 最大上浮（50/50） |

**上浮的渐进性**：

探索权重上浮是渐进的——不是从30%直接跳到50%，而是随稳态持续时间逐步增加。这避免了突然的大幅上浮导致系统不稳定。

- **输入**：基础探索权重、上浮量、上界
- **输出**：上浮后的探索权重
- **约束**：$W_{\text{base}} \leq W_{\text{explore}} \leq W_{\max}$

> **白话翻译**：探索权重上浮就是"闲着就多探索"——稳态时增加探索的权重，从30%逐步上浮到最多50%。不是一下子跳上去，而是慢慢增加——稳态持续越久，上浮越多。上浮有上限（50%），不能让探索超过利用。

---

### 第三部分：上浮幅度与稳态持续时间

#### 2.3 上浮幅度与稳态持续时间的关系（A69.3）

上浮幅度与稳态持续时间成正比，但有上限：

$$\Delta W_{\text{explore}} = \min\left(\alpha \cdot \ln(1 + T_{\text{stable}} / T_{\text{ref}}), \Delta W_{\max}\right) \tag{A69.3}$$

其中：
- $\alpha > 0$ 为上浮速率（通常设为0.05）
- $T_{\text{stable}}$ 为稳态持续时间
- $T_{\text{ref}} > 0$ 为参考时间（通常设为100步）
- $\Delta W_{\max} = W_{\max} - W_{\text{base}} = 0.2$ 为最大上浮量

**对数增长的原因**：

上浮幅度采用对数增长而非线性增长，原因：
1. **边际递减**：稳态初期的上浮最有价值（从30%到35%的增益大于从45%到50%）
2. **安全性**：对数增长比线性增长慢，避免上浮过快导致系统不稳定
3. **饱和性**：对数增长自然趋于饱和，不需要硬性截断

**上浮幅度的数值示例**：

| $T_{\text{stable}}$ | $\Delta W$ | $W_{\text{explore}}$ |
|---------------------|-----------|---------------------|
| 0 | 0.000 | 0.30 |
| 50 | 0.020 | 0.32 |
| 100 | 0.035 | 0.335 |
| 200 | 0.055 | 0.355 |
| 500 | 0.089 | 0.389 |
| 1000 | 0.124 | 0.424 |
| 2000 | 0.159 | 0.459 |
| 5000+ | 0.200 | 0.500 |

**稳态中断时的上浮重置**：

当稳态被打破（自由能或权重方差超过阈值）时，上浮量重置为零：

$$\text{if not is\_stable} \implies \Delta W_{\text{explore}} = 0 \tag{A69.3-reset}$$

这确保了非稳态时系统回到标准的70/30比例，避免在不确定状态下过度探索。

- **输入**：稳态持续时间、上浮速率、参考时间、最大上浮量
- **输出**：上浮量 $\Delta W_{\text{explore}}$
- **约束**：$0 \leq \Delta W_{\text{explore}} \leq \Delta W_{\max}$

> **白话翻译**：上浮幅度与稳态时间的关系就是"越闲探索越多"——稳态持续越久，探索权重上浮越大。但增长是对数的——刚开始上浮快，后面越来越慢，最终趋于饱和。稳态一打破，上浮立刻归零——回到标准的70/30。

---

### 第四部分：上浮后的安全检查

#### 2.4 上浮后安全检查定义（A69.4）

探索权重上浮后需要进行安全检查，确保上浮不会导致系统不稳定或资源超限：

$$\text{safety\_check} = \mathbb{1}\left[W_{\text{explore}} \leq W_{\max} \;\wedge\; \text{resource\_usage} \leq \text{budget} \;\wedge\; \text{no\_anomaly}\right] \tag{A69.4}$$

**安全检查的三项内容**：

1. **权重上界检查**：$W_{\text{explore}} \leq W_{\max} = 0.5$，探索权重不超过50%
2. **资源预算检查**：上浮后的探索操作不超过资源预算
   $$\text{resource\_usage} = W_{\text{explore}} \cdot \text{cost\_per\_explore} \leq \text{budget} \tag{A69.4-resource}$$
3. **异常检查**：上浮后未检测到异常（引用阶段55）

**安全检查失败的处理**：

| 失败项 | 处理 |
|--------|------|
| 权重超限 | 截断到 $W_{\max}$ |
| 资源超限 | 降低上浮量直到资源满足 |
| 检测到异常 | 立即重置上浮量为零 |

**上浮的渐进回退**：

当系统从稳态过渡到非稳态时，上浮量不是立即归零，而是渐进回退：

$$\Delta W_{\text{explore}}(t+1) = \beta \cdot \Delta W_{\text{explore}}(t), \quad \beta \in (0, 1) \tag{A69.4-decay}$$

通常 $\beta = 0.9$，即每步衰减10%。渐进回退避免了突然的比例变化导致系统震荡。

- **输入**：上浮后的探索权重、资源使用、异常状态
- **输出**：安全检查结果
- **约束**：权重不超限；资源不超预算

> **白话翻译**：上浮后安全检查就是"探索不能过头"——三项检查：权重不超过50%、资源不超预算、没有异常。任何一项不满足都要处理——超限就截断，超预算就降低上浮量，有异常就归零。从稳态回到非稳态时，上浮量不是突然归零，而是慢慢衰减——避免比例突变导致震荡。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A69.1 | is_stable = 𝟙[var(F,W_F) < θ_F ∧ var(‖W‖,W_W) < θ_W] | 自由能序列、权重范数序列 | 稳态标志 | θ_F, θ_W > 0 |
| A69.2 | W_explore = W_base + ΔW_explore | 基础权重、上浮量 | 上浮后权重 | W_base ≤ W ≤ W_max |
| A69.3 | ΔW = min(α·ln(1+T/T_ref), ΔW_max) | 稳态时间、速率、参考时间 | 上浮量 | 0 ≤ ΔW ≤ ΔW_max |
| A69.4 | safety = 𝟙[W ≤ W_max ∧ resource ≤ budget ∧ no_anomaly] | 权重、资源、异常 | 安全标志 | 三项全部满足 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import numpy as np

# 引用阶段68的数据类
from stage_068 import ExploreExploitEnforcer


# ============================================================
# 阶段69 数据类：稳态探索增强器
# ============================================================
@dataclass(frozen=True)
class StableExploreBooster:
    """稳态探索增强器的完整描述。

    组合阶段68的ExploreExploitEnforcer，在比例执行器的基础上增加
    稳态探索权重上浮的状态。

    Attributes:
        enforcer: 阶段68的探索-利用执行器（组合字段）
        is_stable: 系统是否处于稳态
        stable_duration: 稳态持续时间
        W_base: 基础探索权重（0.3）
        W_explore: 当前探索权重
        W_max: 探索权重上界（0.5）
        alpha: 上浮速率
        T_ref: 参考时间
        delta_W: 当前上浮量
        theta_stable_F: 自由能稳定性阈值
        theta_stable_W: 权重稳定性阈值
        beta_decay: 渐进回退衰减率
        resource_budget: 资源预算
        anomaly_detected: 是否检测到异常

    Invariants:
        - W_base > 0
        - W_max > W_base
        - W_explore ∈ [W_base, W_max]
        - alpha > 0
        - T_ref > 0
        - 0 < beta_decay < 1
    """
    enforcer: ExploreExploitEnforcer
    is_stable: bool = False
    stable_duration: int = 0
    W_base: float = 0.3
    W_explore: float = 0.3
    W_max: float = 0.5
    alpha: float = 0.05
    T_ref: int = 100
    delta_W: float = 0.0
    theta_stable_F: float = 0.01
    theta_stable_W: float = 0.001
    beta_decay: float = 0.9
    resource_budget: float = 1.0
    anomaly_detected: bool = False

    def __post_init__(self):
        if self.W_base <= 0:
            raise ValueError(
                f"基础探索权重必须 > 0，收到 {self.W_base}"
            )
        if self.W_max <= self.W_base:
            raise ValueError(
                f"探索权重上界必须 > 基础权重，"
                f"收到 max={self.W_max}, base={self.W_base}"
            )
        if not (self.W_base <= self.W_explore <= self.W_max):
            raise ValueError(
                f"当前探索权重必须 ∈ [W_base, W_max]，"
                f"收到 {self.W_explore}"
            )
        if self.alpha <= 0:
            raise ValueError(
                f"上浮速率必须 > 0，收到 {self.alpha}"
            )

    @property
    def exploit_weight(self) -> float:
        """利用权重。"""
        return 1.0 - self.W_explore

    @property
    def boost_active(self) -> bool:
        """上浮是否激活。"""
        return self.is_stable and self.delta_W > 0


# ============================================================
# 阶段69 核心接口：稳态探索增强器
# ============================================================
class StableExploreBoosterInterface:
    """阶段69的核心接口：稳态探索增强器。

    实现稳态检测、探索权重上浮、上浮幅度计算和安全检查。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 探索权重不超过上界
        - 上浮后通过安全检查
        - 非稳态时渐进回退
    """

    def detect_stability(
        self,
        free_energy_history: np.ndarray,
        weight_norm_history: np.ndarray,
        window_F: int,
        window_W: int,
        theta_F: float,
        theta_W: float
    ) -> bool:
        """检测系统稳态（A69.1）。

        Args:
            free_energy_history: 自由能历史
            weight_norm_history: 权重范数历史
            window_F: 自由能窗口长度
            window_W: 权重窗口长度
            theta_F: 自由能稳定性阈值
            theta_W: 权重稳定性阈值

        Returns:
            是否处于稳态
        """
        if len(free_energy_history) < window_F or len(weight_norm_history) < window_W:
            return False
        var_F = float(np.var(free_energy_history[-window_F:]))
        var_W = float(np.var(weight_norm_history[-window_W:]))
        return var_F < theta_F and var_W < theta_W

    def compute_boost(
        self,
        stable_duration: int,
        alpha: float,
        T_ref: int,
        delta_W_max: float
    ) -> float:
        """计算探索权重上浮量（A69.3）。

        ΔW = min(α·ln(1 + T/T_ref), ΔW_max)

        Args:
            stable_duration: 稳态持续时间
            alpha: 上浮速率
            T_ref: 参考时间
            delta_W_max: 最大上浮量

        Returns:
            上浮量 ΔW ∈ [0, ΔW_max]
        """
        delta_W = alpha * np.log(1.0 + stable_duration / max(T_ref, 1))
        return float(min(delta_W, delta_W_max))

    def apply_boost(
        self,
        booster: StableExploreBooster
    ) -> StableExploreBooster:
        """应用探索权重上浮（A69.2）。

        Args:
            booster: 当前增强器状态

        Returns:
            更新后的增强器状态
        """
        from dataclasses import replace
        if not booster.is_stable:
            delta_W = booster.delta_W * booster.beta_decay
        else:
            delta_W_max = booster.W_max - booster.W_base
            delta_W = self.compute_boost(
                booster.stable_duration,
                booster.alpha,
                booster.T_ref,
                delta_W_max,
            )
        W_explore = min(booster.W_base + delta_W, booster.W_max)
        return replace(
            booster,
            delta_W=delta_W,
            W_explore=W_explore,
        )

    def safety_check(
        self,
        W_explore: float,
        W_max: float,
        resource_usage: float,
        resource_budget: float,
        anomaly_detected: bool
    ) -> Tuple[bool, float]:
        """执行安全检查（A69.4）。

        Args:
            W_explore: 当前探索权重
            W_max: 权重上界
            resource_usage: 资源使用量
            resource_budget: 资源预算
            anomaly_detected: 是否检测到异常

        Returns:
            (是否安全, 修正后的探索权重)
        """
        if anomaly_detected:
            return False, 0.0
        corrected_W = min(W_explore, W_max)
        if resource_usage > resource_budget:
            scale = resource_budget / max(resource_usage, 1e-12)
            corrected_W = corrected_W * scale
        safe = corrected_W <= W_max and resource_usage * (corrected_W / max(W_explore, 1e-12)) <= resource_budget
        return safe, corrected_W

    def step(
        self,
        booster: StableExploreBooster,
        free_energy_history: np.ndarray,
        weight_norm_history: np.ndarray
    ) -> StableExploreBooster:
        """执行一步稳态探索增强（A69.1-A69.4）。

        Args:
            booster: 当前增强器状态
            free_energy_history: 自由能历史
            weight_norm_history: 权重范数历史

        Returns:
            更新后的增强器状态
        """
        from dataclasses import replace
        is_stable = self.detect_stability(
            free_energy_history,
            weight_norm_history,
            window_F=max(1, len(free_energy_history)),
            window_W=max(1, len(weight_norm_history)),
            theta_F=booster.theta_stable_F,
            theta_W=booster.theta_stable_W,
        )
        stable_duration = booster.stable_duration + 1 if is_stable else 0
        booster = replace(booster, is_stable=is_stable, stable_duration=stable_duration)
        booster = self.apply_boost(booster)
        safe, corrected_W = self.safety_check(
            booster.W_explore,
            booster.W_max,
            resource_usage=booster.W_explore * booster.enforcer.reformer_state.safety.reformer.resource_cost if hasattr(booster.enforcer.reformer_state.safety.reformer, "resource_cost") else 0.0,
            resource_budget=booster.resource_budget,
            anomaly_detected=booster.anomaly_detected,
        )
        if not safe:
            booster = replace(booster, W_explore=corrected_W, delta_W=corrected_W - booster.W_base)
        return booster
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `ExploreExploitEnforcer` | ← 阶段68 | 作为增强器的比例执行器输入（组合字段） |
| **输入** | A26.3（稳态空闲场景） | ← 阶段26 | 稳态探索的理论依据 |
| **输入** | A5.17（自由能梯度流） | ← 阶段5 | 稳态时跳出局部最优的理论基础 |
| **输出** | `StableExploreBooster` | → 阶段70 | 增强器状态，作为进化验证的输入 |
| **输出** | A69.1-A69.4（稳态探索定义） | → 阶段70 | 验证测试的依据 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 稳态检测计算 | 自由能和权重方差的快速计算 |
| NumPy | 历史数据分析 | 滑动窗口内的方差计算 |
| Pydantic | 数据验证 | 支撑StableExploreBooster的运行时类型检查 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V69.1 | 稳态检测正确性 | 自由能和权重都稳定时判定为稳态 | 条件测试 |
| V69.2 | 仅自由能稳定不判定 | 权重不稳定时不判定为稳态 | 条件测试 |
| V69.3 | 探索权重上浮 | 稳态时 W_explore > W_base | 条件测试 |
| V69.4 | 上浮幅度对数增长 | ΔW = α·ln(1+T/T_ref) | 数值验证 |
| V69.5 | 上浮不超过上界 | W_explore ≤ W_max | 范围检查 |
| V69.6 | 稳态中断重置 | 非稳态时 ΔW → 0 | 状态验证 |
| V69.7 | 渐进回退 | 非稳态时 ΔW 衰减而非突然归零 | 衰减验证 |
| V69.8 | 安全检查权重 | W > W_max 时截断 | 安全测试 |
| V69.9 | 安全检查资源 | 资源超限时降低上浮量 | 安全测试 |
| V69.10 | 安全检查异常 | 异常时归零 | 安全测试 |
| V69.11 | StableExploreBooster组合 | `booster.enforcer` 类型为 `ExploreExploitEnforcer` | 类型检查 |
| V69.12 | 非法参数拒绝 | W_base ≤ 0 抛出 `ValueError` | 异常测试 |
| V69.13 | 长期稳态上浮 | 1000步稳态后 W_explore 趋近 W_max | 长期运行测试 |
| V69.14 | 稳态-非稳态切换 | 切换时上浮量正确调整 | 切换测试 |

---

> **阶段69完** | 阶段69建立了稳态环境探索权重上浮机制：稳态检测 var(F, W) < θ_stable（A69.1），探索权重上浮 W_explore = W_base + ΔW_explore（A69.2），上浮幅度与稳态时间的关系 ΔW = min(α·ln(1+T/T_ref), ΔW_max)（A69.3），上浮后安全检查（A69.4），封装为StableExploreBooster数据类组合ExploreExploitEnforcer。下一步：阶段70将定义双模式进化的完整验证方案。
