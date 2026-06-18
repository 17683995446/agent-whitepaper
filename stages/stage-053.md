# 阶段53：正向/负向反馈信号生成——从误差变化到驱动信号

> **阶段编号**：S53 | **所属层**：第六层·误差评估层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 正向反馈 | "好消息"——误差下降了，系统在变好 |
> | 负向反馈 | "坏消息"——误差上升了，系统在变差 |
> | 反馈强度 | "消息有多重要"——变化越大，信号越强 |
> | 反馈路由 | "消息发给谁"——驱力引擎、模型更新器、任务生成器各得一份 |
> | FeedbackSignal | "反馈信号的完整画像"——组合ErrorAccumulation，在误差累积基础上增加反馈信号 |

---

## 1. 第一性原理依据

本阶段从阶段52（误差累积策略）的核心结论出发，建立正向/负向反馈信号的生成机制。核心依据为：

- **A52.2-A52.3（累积误差与EWMA）**（引用阶段52）：$E_W(t) = \sum_{s \in W} \varepsilon_s$ 和 $E_{\text{ewma}}(t) = \alpha \varepsilon_t + (1-\alpha) E_{\text{ewma}}(t-1)$。反馈信号基于累积误差的变化量 $\Delta F$ 生成——误差下降产生正向反馈，误差上升产生负向反馈。

- **A52.4（多尺度趋势）**（引用阶段52）：$\text{trend} = (E_{\text{short}} / T_{\text{short}}) / (E_{\text{long}} / T_{\text{long}})$。多尺度趋势为反馈信号提供时间上下文——短期改善但长期恶化时，正向反馈应被抑制。

- **A4.4（变分自由能全局唯一定义）**（引用阶段4）：$F[q] = -\mathbb{E}_q[\ln p(o|\psi)] + D_{\text{KL}}[q(\psi) \| p(\psi)]$。反馈信号中的 $\Delta F$ 对应自由能的变化——自由能下降是正向反馈的物理基础，自由能上升是负向反馈的物理基础。

- **A29.2（收益驱力与自由能下降率）**（引用阶段29）：$R_{\text{gain}} \propto -\dot{F}$，收益驱力与自由能下降率成正比。本阶段的正向反馈与A29.2的收益驱力激活条件一致——误差下降→自由能下降→收益驱力激活。

本阶段的核心目标是：**定义正向反馈 $\text{reward\_signal} = |\Delta F|$（$\Delta F < 0$时）、负向反馈 $\text{penalty\_signal} = |\Delta F|$（$\Delta F > 0$时）、反馈强度 $\text{feedback} = \tanh(\kappa \cdot |\Delta F| / F_{\text{baseline}})$、反馈路由 $\text{feedback} \to [\text{drive\_engine}, \text{model\_updater}, \text{task\_generator}]$，封装为FeedbackSignal数据类，组合ErrorAccumulation作为误差累积的输入。** 反馈信号是误差评估到行动的"桥梁"——它将误差变化转化为可路由的驱动信号。

> **去重声明**：本阶段仅定义反馈信号的生成机制，不涉及驱力激活的具体计算（阶段57）或模型更新的具体规则（阶段58）。反馈路由定义了信号的去向，但不定义接收方的处理逻辑。

---

## 2. 核心数学公式

### 第一部分：正向反馈

#### 2.1 正向反馈定义（A53.1）

当自由能变化量 $\Delta F < 0$（误差下降）时，生成正向反馈信号：

$$\Delta F < 0 \implies \text{reward\_signal} = |\Delta F| = -\Delta F \tag{A53.1}$$

其中 $\Delta F = F(t) - F(t-1)$ 为自由能的变化量。

**自由能变化量与误差变化量的关系**：

由A4.8（引用阶段4）的拉普拉斯近似，自由能变化量近似为：

$$\Delta F \approx \Delta \varepsilon_{\text{global}} + \Delta C \tag{A53.1-approx}$$

其中 $\Delta \varepsilon_{\text{global}}$ 为全局误差变化量（引用A51.5），$\Delta C$ 为复杂性项变化量。在稳态附近，$\Delta C \approx 0$，因此 $\Delta F \approx \Delta \varepsilon_{\text{global}}$。

**正向反馈的物理含义**：

正向反馈意味着"系统在变好"——预测误差下降，自由能降低，系统对环境的预测更准确。正向反馈对应：
1. 收益驱力激活（引用A29.2）
2. 当前策略有效，应继续执行
3. 模型参数在正确方向上更新

**正向反馈的边界条件**：

当 $\Delta F = 0$ 时，不生成正向反馈——系统既没变好也没变差。当 $|\Delta F|$ 极小时，正向反馈趋近于零，避免对噪声过度响应。

- **输入**：自由能变化量 $\Delta F$
- **输出**：$\text{reward\_signal} \in \mathbb{R}_{\geq 0}$（仅当 $\Delta F < 0$ 时非零）
- **约束**：$\text{reward\_signal} = 0$ 当 $\Delta F \geq 0$

> **白话翻译**：正向反馈就是"好消息"——误差下降了，系统在变好。好消息的"音量"等于误差下降的幅度——降得越多，好消息越响。如果误差没变或上升，就没有好消息。这和"收益驱力"一致——误差下降意味着"赚到了"。

---

### 第二部分：负向反馈

#### 2.2 负向反馈定义（A53.2）

当自由能变化量 $\Delta F > 0$（误差上升）时，生成负向反馈信号：

$$\Delta F > 0 \implies \text{penalty\_signal} = |\Delta F| = \Delta F \tag{A53.2}$$

**负向反馈的物理含义**：

负向反馈意味着"系统在变差"——预测误差上升，自由能升高，系统对环境的预测更不准确。负向反馈对应：
1. 安全驱力激活（引用A29.1，自由能偏离基线）
2. 当前策略可能失效，需要调整
3. 模型参数可能过期，需要更新

**负向反馈与正向反馈的对称性**：

| 特性 | 正向反馈 | 负向反馈 |
|------|---------|---------|
| 触发条件 | $\Delta F < 0$ | $\Delta F > 0$ |
| 信号值 | $|\Delta F|$ | $|\Delta F|$ |
| 物理含义 | 误差下降 | 误差上升 |
| 驱力关联 | 收益驱力 | 安全驱力 |
| 行动倾向 | 继续当前策略 | 调整/回避 |

**负向反馈的边界条件**：

当 $\Delta F = 0$ 时，不生成负向反馈。当 $|\Delta F|$ 极小时，负向反馈趋近于零。

- **输入**：自由能变化量 $\Delta F$
- **输出**：$\text{penalty\_signal} \in \mathbb{R}_{\geq 0}$（仅当 $\Delta F > 0$ 时非零）
- **约束**：$\text{penalty\_signal} = 0$ 当 $\Delta F \leq 0$

> **白话翻译**：负向反馈就是"坏消息"——误差上升了，系统在变差。坏消息的"音量"等于误差上升的幅度——升得越多，坏消息越响。这和"安全驱力"一致——误差上升意味着"出问题了"。

---

### 第三部分：反馈强度

#### 2.3 反馈强度定义（A53.3）

反馈强度通过tanh函数将原始信号归一化到 $[0, 1)$：

$$\text{feedback} = \tanh\left(\kappa \cdot \frac{|\Delta F|}{F_{\text{baseline}}}\right) \tag{A53.3}$$

其中：
- $\kappa > 0$ 为灵敏度参数（$\kappa$ 越大，对误差变化越敏感）
- $F_{\text{baseline}} > 0$ 为自由能基线（引用A29.1），用于归一化
- $\tanh$ 将输出限制在 $[0, 1)$，避免信号爆炸

**tanh归一化的优势**：

1. **有界性**：$\text{feedback} \in [0, 1)$，无论 $|\Delta F|$ 多大，反馈强度不会超过1
2. **单调性**：$|\Delta F|$ 越大，feedback 越大
3. **饱和性**：当 $|\Delta F| \gg F_{\text{baseline}}$ 时，feedback → 1（避免过度反应）
4. **灵敏度**：当 $|\Delta F| \ll F_{\text{baseline}}$ 时，feedback ≈ $\kappa \cdot |\Delta F| / F_{\text{baseline}}$（线性响应）

**灵敏度参数 $\kappa$ 的选择**：

| $\kappa$ | $|\Delta F|/F_{\text{baseline}}$ 达到feedback=0.5的倍数 | 含义 |
|----------|---------------------------------------------|------|
| 1 | 0.55 | 基线变化55%时反馈强度达0.5 |
| 2 | 0.275 | 基线变化27.5%时反馈强度达0.5 |
| 5 | 0.11 | 基线变化11%时反馈强度达0.5 |

**逐层反馈强度**：

对每一层分别计算反馈强度：

$$\text{feedback}_l = \tanh\left(\kappa_l \cdot \frac{|\Delta \varepsilon_l|}{\varepsilon_{l,\text{baseline}}}\right) \tag{A53.3-per-layer}$$

不同层可以使用不同的 $\kappa_l$——底层用较大的 $\kappa$（对即时变化敏感），高层用较小的 $\kappa$（对长期变化敏感）。

- **输入**：自由能变化量 $\Delta F$，基线 $F_{\text{baseline}}$，灵敏度 $\kappa$
- **输出**：$\text{feedback} \in [0, 1)$
- **约束**：$\kappa > 0$；$F_{\text{baseline}} > 0$

> **白话翻译**：反馈强度就是"消息有多重要"——用tanh函数把原始变化量归一化到0到1之间。变化量相对于基线越大，反馈越强；但再大的变化也不会让反馈超过1（饱和了）。$\kappa$ 是"灵敏度旋钮"——调大则对微小变化也敏感，调小则只对大变化有反应。

---

### 第四部分：反馈路由

#### 2.4 反馈路由定义（A53.4）

反馈信号路由到三个目标模块：

$$\text{feedback} \xrightarrow{\text{route}} [\text{drive\_engine}, \text{model\_updater}, \text{task\_generator}] \tag{A53.4}$$

**路由规则**：

| 反馈类型 | drive_engine | model_updater | task_generator |
|---------|-------------|--------------|---------------|
| 正向反馈（reward） | 激活收益驱力 $R_{\text{gain}}$ | 降低更新增益 | 降低探索强度 |
| 负向反馈（penalty） | 激活安全驱力 $R_{\text{safe}}$ | 提高更新增益 | 提高探索强度 |

**路由权重**：

不同目标模块对反馈的敏感度不同，通过路由权重调节：

$$\text{signal}_i = w_i \cdot \text{feedback}, \quad i \in \{\text{drive, model, task}\} \tag{A53.4-weights}$$

其中 $w_{\text{drive}} + w_{\text{model}} + w_{\text{task}} = 1$。

**默认路由权重**：

| 模块 | 权重 | 理由 |
|------|------|------|
| drive_engine | 0.5 | 驱力是最直接的响应通道 |
| model_updater | 0.3 | 模型更新需要更审慎 |
| task_generator | 0.2 | 任务调整是间接响应 |

**路由的自适应调整**：

路由权重可以根据系统状态动态调整：
- 异常检测触发时（引用阶段55）：$w_{\text{drive}}$ 增大，优先响应安全需求
- 误差持续下降时：$w_{\text{model}}$ 增大，趁机优化模型
- 误差长期稳定时：$w_{\text{task}}$ 增大，探索新任务

**反馈路由与后续阶段的对应**：

| 路由目标 | 后续阶段 | 处理逻辑 |
|---------|---------|---------|
| drive_engine | 阶段57 | 误差反馈到驱力的映射 |
| model_updater | 阶段58 | 误差反馈到模型的映射 |
| task_generator | 阶段61+ | 任务生成与调度 |

- **输入**：反馈强度 feedback，反馈类型（正向/负向），路由权重
- **输出**：三个目标模块的信号 $[\text{signal}_{\text{drive}}, \text{signal}_{\text{model}}, \text{signal}_{\text{task}}]$
- **约束**：路由权重归一化；信号值 ∈ $[0, 1)$

> **白话翻译**：反馈路由就是"消息发给谁"——好消息和坏消息分别发给驱力引擎、模型更新器和任务生成器。好消息让驱力引擎"开心"（收益驱力激活），让模型更新器"别改了"（当前挺好），让任务生成器"别探索了"（当前策略有效）。坏消息则相反——驱力引擎"警惕"（安全驱力激活），模型更新器"快改"（出问题了），任务生成器"多探索"（当前策略可能失效）。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A53.1 | $\text{reward\_signal} = |\Delta F|$（$\Delta F < 0$） | 自由能变化量 $\Delta F$ | 正向反馈 $\in \mathbb{R}_{\geq 0}$ | 仅 $\Delta F < 0$ 时非零 |
| A53.2 | $\text{penalty\_signal} = |\Delta F|$（$\Delta F > 0$） | 自由能变化量 $\Delta F$ | 负向反馈 $\in \mathbb{R}_{\geq 0}$ | 仅 $\Delta F > 0$ 时非零 |
| A53.3 | $\text{feedback} = \tanh(\kappa \cdot |\Delta F| / F_{\text{baseline}})$ | $\Delta F$、基线、灵敏度 $\kappa$ | 反馈强度 $\in [0, 1)$ | $\kappa > 0$，$F_{\text{baseline}} > 0$ |
| A53.4 | $\text{feedback} \to [\text{drive}, \text{model}, \text{task}]$ | 反馈强度、路由权重 | 三路信号 | 权重归一化 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict
from enum import Enum
import numpy as np

# 引用阶段52的数据类
from stage_052 import ErrorAccumulation


# ============================================================
# 阶段53 枚举：反馈类型
# ============================================================
class FeedbackType(Enum):
    """反馈信号类型。"""
    REWARD = "reward"      # 正向反馈（误差下降）
    PENALTY = "penalty"    # 负向反馈（误差上升）
    NEUTRAL = "neutral"    # 中性（误差不变）


# ============================================================
# 阶段53 枚举：路由目标
# ============================================================
class RouteTarget(Enum):
    """反馈路由目标。"""
    DRIVE_ENGINE = "drive_engine"    # 驱力引擎
    MODEL_UPDATER = "model_updater"  # 模型更新器
    TASK_GENERATOR = "task_generator"  # 任务生成器


# ============================================================
# 阶段53 数据类：反馈路由结果
# ============================================================
@dataclass(frozen=True)
class FeedbackRoutes:
    """反馈路由结果。"""
    drive_signal: float       # 驱力引擎信号
    model_signal: float       # 模型更新器信号
    task_signal: float        # 任务生成器信号
    drive_weight: float = 0.5
    model_weight: float = 0.3
    task_weight: float = 0.2


# ============================================================
# 阶段53 数据类：反馈信号
# ============================================================
@dataclass(frozen=True)
class FeedbackSignal:
    """正向/负向反馈信号的完整描述。

    组合阶段52的ErrorAccumulation，在误差累积的基础上增加
    反馈信号的生成和路由。

    Attributes:
        error_accumulation: 阶段52的误差累积状态（组合字段）
        delta_F: 自由能变化量 ΔF
        feedback_type: 反馈类型（正向/负向/中性）
        reward_signal: 正向反馈信号（A53.1）
        penalty_signal: 负向反馈信号（A53.2）
        feedback_intensity: 反馈强度（A53.3）
        kappa: 灵敏度参数 κ
        F_baseline: 自由能基线
        routes: 反馈路由结果（A53.4）
        per_layer_feedback: 逐层反馈强度

    Invariants:
        - reward_signal >= 0
        - penalty_signal >= 0
        - reward_signal 和 penalty_signal 不同时非零
        - feedback_intensity ∈ [0, 1)
        - kappa > 0
        - F_baseline > 0
    """
    error_accumulation: ErrorAccumulation
    delta_F: float
    feedback_type: FeedbackType
    reward_signal: float = 0.0
    penalty_signal: float = 0.0
    feedback_intensity: float = 0.0
    kappa: float = 2.0
    F_baseline: float = 1.0
    routes: Optional[FeedbackRoutes] = None
    per_layer_feedback: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.reward_signal < 0:
            raise ValueError(
                f"正向反馈信号必须 >= 0，收到 {self.reward_signal}"
            )
        if self.penalty_signal < 0:
            raise ValueError(
                f"负向反馈信号必须 >= 0，收到 {self.penalty_signal}"
            )
        if self.reward_signal > 0 and self.penalty_signal > 0:
            raise ValueError("正向和负向反馈信号不能同时非零")
        if not (0 <= self.feedback_intensity < 1):
            raise ValueError(
                f"反馈强度必须 ∈ [0,1)，收到 {self.feedback_intensity}"
            )
        if self.kappa <= 0:
            raise ValueError(
                f"灵敏度参数必须 > 0，收到 {self.kappa}"
            )
        if self.F_baseline <= 0:
            raise ValueError(
                f"自由能基线必须 > 0，收到 {self.F_baseline}"
            )

    @property
    def is_positive(self) -> bool:
        """是否为正向反馈。"""
        return self.feedback_type == FeedbackType.REWARD

    @property
    def is_negative(self) -> bool:
        """是否为负向反馈。"""
        return self.feedback_type == FeedbackType.PENALTY

    @property
    def raw_signal(self) -> float:
        """原始信号值（正或负）。"""
        return self.reward_signal - self.penalty_signal


# ============================================================
# 阶段53 核心接口：反馈信号生成器
# ============================================================
class FeedbackGenerator:
    """阶段53的核心接口：反馈信号生成器。

    从误差变化量生成正向/负向反馈信号，计算反馈强度，
    并路由到目标模块。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 正向和负向反馈互斥
        - 反馈强度 ∈ [0, 1)
        - 路由权重归一化
    """

    def compute_delta_F(
        self,
        current_accumulation: ErrorAccumulation,
        prev_accumulation: ErrorAccumulation
    ) -> float:
        """计算自由能变化量 ΔF。

        ΔF ≈ Δε_global（稳态近似）

        Args:
            current_accumulation: 当前累积状态
            prev_accumulation: 上一累积状态

        Returns:
            自由能变化量 ΔF
        """
        # 使用全局误差的累积作为自由能的稳态近似
        current_F = current_accumulation.cumulative_error
        prev_F = prev_accumulation.cumulative_error
        return current_F - prev_F

    def generate_reward_signal(
        self,
        delta_F: float
    ) -> float:
        """生成正向反馈信号（A53.1）。

        reward_signal = |ΔF| 当 ΔF < 0

        Args:
            delta_F: 自由能变化量

        Returns:
            正向反馈信号 >= 0
        """
        if delta_F < 0:
            return -delta_F
        return 0.0

    def generate_penalty_signal(
        self,
        delta_F: float
    ) -> float:
        """生成负向反馈信号（A53.2）。

        penalty_signal = |ΔF| 当 ΔF > 0

        Args:
            delta_F: 自由能变化量

        Returns:
            负向反馈信号 >= 0
        """
        if delta_F > 0:
            return delta_F
        return 0.0

    def compute_feedback_intensity(
        self,
        delta_F: float,
        F_baseline: float,
        kappa: float
    ) -> float:
        """计算反馈强度（A53.3）。

        feedback = tanh(κ · |ΔF| / F_baseline)

        Args:
            delta_F: 自由能变化量
            F_baseline: 自由能基线
            kappa: 灵敏度参数

        Returns:
            反馈强度 ∈ [0, 1)

        Raises:
            ValueError: 当 kappa <= 0 或 F_baseline <= 0 时
        """
        if kappa <= 0:
            raise ValueError(f"kappa 必须 > 0，收到 {kappa}")
        if F_baseline <= 0:
            raise ValueError(f"F_baseline 必须 > 0，收到 {F_baseline}")
        return np.tanh(kappa * abs(delta_F) / F_baseline)

    def route_feedback(
        self,
        feedback_intensity: float,
        feedback_type: FeedbackType,
        weights: tuple = (0.5, 0.3, 0.2)
    ) -> FeedbackRoutes:
        """路由反馈信号（A53.4）。

        Args:
            feedback_intensity: 反馈强度
            feedback_type: 反馈类型
            weights: 路由权重 (drive, model, task)

        Returns:
            FeedbackRoutes 实例

        Raises:
            ValueError: 当权重之和不近似1时
        """
        if not np.allclose(np.sum(weights), 1.0):
            raise ValueError(
                f"路由权重之和必须为1，收到 {np.sum(weights)}"
            )
        drive_w, model_w, task_w = weights
        drive_signal = drive_w * feedback_intensity
        model_signal = model_w * feedback_intensity
        task_signal = task_w * feedback_intensity
        return FeedbackRoutes(
            drive_signal=drive_signal,
            model_signal=model_signal,
            task_signal=task_signal,
            drive_weight=drive_w,
            model_weight=model_w,
            task_weight=task_w,
        )

    def generate(
        self,
        accumulation: ErrorAccumulation,
        prev_accumulation: ErrorAccumulation,
        kappa: float = 2.0,
        F_baseline: float = 1.0
    ) -> FeedbackSignal:
        """完整生成反馈信号（A53.1-A53.4）。

        Args:
            accumulation: 当前累积状态
            prev_accumulation: 上一累积状态
            kappa: 灵敏度参数
            F_baseline: 自由能基线

        Returns:
            FeedbackSignal 实例
        """
        delta_F = self.compute_delta_F(accumulation, prev_accumulation)

        reward = self.generate_reward_signal(delta_F)
        penalty = self.generate_penalty_signal(delta_F)

        if delta_F < 0:
            feedback_type = FeedbackType.REWARD
        elif delta_F > 0:
            feedback_type = FeedbackType.PENALTY
        else:
            feedback_type = FeedbackType.NEUTRAL

        intensity = self.compute_feedback_intensity(
            delta_F, F_baseline, kappa
        )

        routes = self.route_feedback(intensity, feedback_type)

        # 逐层反馈强度（基于逐层误差变化）
        current_eps = accumulation.standardized_error.epsilon_vector
        prev_eps = prev_accumulation.standardized_error.epsilon_vector
        delta_eps = np.abs(current_eps - prev_eps)
        baseline_eps = np.ones(3) * F_baseline / 3.0
        per_layer = np.tanh(kappa * delta_eps / baseline_eps)

        return FeedbackSignal(
            error_accumulation=accumulation,
            delta_F=delta_F,
            feedback_type=feedback_type,
            reward_signal=reward,
            penalty_signal=penalty,
            feedback_intensity=intensity,
            kappa=kappa,
            F_baseline=F_baseline,
            routes=routes,
            per_layer_feedback=per_layer,
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `ErrorAccumulation` | ← 阶段52 | 作为反馈生成的误差累积输入（组合字段） |
| **输入** | A52.2-A52.3（累积误差与EWMA） | ← 阶段52 | 自由能变化量的计算基础 |
| **输入** | A4.4（变分自由能） | ← 阶段4 | $\Delta F$ 的物理含义 |
| **输入** | A29.1-A29.2（驱力与自由能） | ← 阶段29 | 正向/负向反馈与驱力的对应关系 |
| **输出** | `FeedbackSignal` | → 阶段54 | 反馈信号，作为精度加权的输入 |
| **输出** | `FeedbackRoutes` | → 阶段57/58 | 路由信号，驱力映射和模型映射的输入 |
| **输出** | A53.1-A53.4（反馈信号定义） | → 阶段55-60 | 反馈生成的基础定义 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 反馈计算JIT编译 | `jax.nn.tanh`实现反馈强度的向量化计算 |
| NumPy | 信号路由计算 | 支撑多路信号的权重分配和路由 |
| Pydantic | 数据验证 | 支撑FeedbackSignal的运行时类型检查 |
| structlog | 反馈审计 | 结构化日志库，支撑反馈信号的完整审计追踪 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V53.1 | 正向反馈正确性 | $\Delta F < 0$ 时 reward_signal = $|\Delta F|$ | 数值验证 |
| V53.2 | 负向反馈正确性 | $\Delta F > 0$ 时 penalty_signal = $|\Delta F|$ | 数值验证 |
| V53.3 | 正负互斥 | reward_signal 和 penalty_signal 不同时非零 | 逻辑验证 |
| V53.4 | 中性条件 | $\Delta F = 0$ 时两信号均为零 | 边界测试 |
| V53.5 | 反馈强度范围 | $\text{feedback} \in [0, 1)$ | 范围检查 |
| V53.6 | 反馈强度单调性 | $|\Delta F|$ 越大 feedback 越大 | 单调性测试 |
| V53.7 | 反馈强度饱和 | $|\Delta F| \to \infty$ 时 feedback → 1 | 极限测试 |
| V53.8 | tanh正确性 | feedback = $\tanh(\kappa \cdot |\Delta F| / F_{\text{baseline}})$ | 数值对比 |
| V53.9 | 路由权重归一化 | $w_{\text{drive}} + w_{\text{model}} + w_{\text{task}} = 1$ | 求和验证 |
| V53.10 | 路由信号正确性 | $\text{signal}_i = w_i \cdot \text{feedback}$ | 数值验证 |
| V53.11 | 正向路由方向 | 正向反馈→驱力激活收益、模型降低增益 | 逻辑验证 |
| V53.12 | 负向路由方向 | 负向反馈→驱力激活安全、模型提高增益 | 逻辑验证 |
| V53.13 | FeedbackSignal组合 | `signal.error_accumulation` 类型为 `ErrorAccumulation` | 类型检查 |
| V53.14 | 非法输入拒绝 | $\kappa \leq 0$、$F_{\text{baseline}} \leq 0$ 抛出 `ValueError` | 异常测试 |
| V53.15 | 端到端反馈生成 | 1000步误差变化循环无异常 | 长期运行测试 |

---

> **阶段53完** | 阶段53建立了正向/负向反馈信号的生成机制：正向反馈 $\text{reward\_signal} = |\Delta F|$ 在误差下降时触发（A53.1），负向反馈 $\text{penalty\_signal} = |\Delta F|$ 在误差上升时触发（A53.2），反馈强度 $\text{feedback} = \tanh(\kappa \cdot |\Delta F| / F_{\text{baseline}})$ 将原始信号归一化到 $[0,1)$（A53.3），反馈路由将信号分发到驱力引擎、模型更新器和任务生成器（A53.4），封装为FeedbackSignal数据类组合ErrorAccumulation。下一步：阶段54将定义误差信号的精度加权机制。
