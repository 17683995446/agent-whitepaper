# 阶段52：误差评估的时间窗口与累积策略——滑动窗口、指数加权与多尺度分析

> **阶段编号**：S52 | **所属层**：第六层·误差评估层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 滑动窗口 | "只看最近N步"——不关心远古历史，只关注近期表现 |
> | 累积误差 | "最近N步的总偏差"——窗口内误差的简单求和 |
> | 指数加权累积 | "越近越重要"——近期误差权重大，远期误差权重小 |
> | 多尺度窗口 | "短中长三个望远镜"——同时看10步、100步、1000步的误差趋势 |
> | ErrorAccumulation | "误差累积的完整画像"——组合StandardizedError，在标准化误差基础上增加时间累积 |

---

## 1. 第一性原理依据

本阶段从阶段51（标准化误差定义）的核心结论出发，建立误差评估的时间窗口与累积策略。核心依据为：

- **A51.1-A51.5（标准化误差定义）**（引用阶段51）：逐层误差向量 $\boldsymbol{\varepsilon} = [\varepsilon_{\text{low}}, \varepsilon_{\text{mid}}, \varepsilon_{\text{high}}]$ 和全局误差 $\varepsilon_{\text{global}}$。本阶段在A51.1-A51.5的瞬时误差基础上，引入时间维度——将单步误差扩展为时间窗口内的累积误差。

- **A3.3（累积误差与生存条件）**（引用阶段3）：$\sum_{t=0}^{T} \varepsilon_0(t) \leq -\ln P_{\min}$，累积误差有上界。本阶段的滑动窗口累积策略为A3.3提供可操作的计算框架——不是无限累积，而是在有限窗口内累积。

- **A17.1-A17.2（快感知-慢学习）**（引用阶段17）：$\tau_{\text{perceive}} \ll \tau_{\text{learn}}$。本阶段的多尺度窗口与快感知-慢学习原则一致——短窗口对应快感知（捕捉即时变化），长窗口对应慢学习（捕捉长期趋势）。

- **A50.4（跨层时间尺度对齐）**（引用阶段50）：$\Delta t_{\text{low}} : \Delta t_{\text{mid}} : \Delta t_{\text{high}} = 1 : 10 : 100$。本阶段的窗口大小与跨层时间尺度对齐——短窗口（10步）对应底层时间尺度，中窗口（100步）对应中层时间尺度，长窗口（1000步）对应高层时间尺度。

本阶段的核心目标是：**定义滑动窗口 $W = [t-T, t]$、累积误差 $E_W = \sum_{t \in W} \varepsilon_t$、指数加权累积 $E_{\text{ewma}} = \alpha \varepsilon_t + (1-\alpha) E_{\text{ewma}}^{\text{prev}}$、多尺度窗口 $[W_{\text{short}}, W_{\text{mid}}, W_{\text{long}}] = [10, 100, 1000]$，封装为ErrorAccumulation数据类，组合StandardizedError作为标准化误差的输入。** 时间窗口与累积策略是误差评估的"时间维度"——它将瞬时误差转化为具有时间上下文的累积信号，为异常检测（阶段55）和反馈生成（阶段53）提供时间敏感的输入。

> **去重声明**：本阶段仅定义误差的时间累积策略，不涉及异常检测阈值（阶段55）、反馈信号生成（阶段53）或精度加权（阶段54）。

---

## 2. 核心数学公式

### 第一部分：滑动窗口

#### 2.1 滑动窗口定义（A52.1）

滑动窗口定义了误差累积的时间范围：

$$W(t) = [t - T, t] \tag{A52.1}$$

其中：
- $t$ 为当前时刻
- $T$ 为窗口长度（回看步数）
- $W(t)$ 包含从 $t-T$ 到 $t$ 的所有时刻

**滑动窗口的更新**：

每一步，窗口向前滑动一步——丢弃最旧的误差 $\varepsilon_{t-T}$，加入最新的误差 $\varepsilon_t$：

$$W(t+1) = [t - T + 1, t + 1] = W(t) \cup \{t+1\} \setminus \{t-T\} \tag{A52.1-update}$$

**窗口长度的选择原则**：

窗口长度 $T$ 的选择需要在"灵敏度"和"稳定性"之间权衡：
- $T$ 太小：对突变敏感，但容易受噪声影响（误报率高）
- $T$ 太大：平滑噪声，但对真实变化反应迟钝（检测延迟大）

**与A50.4时间尺度的对应**：

| 窗口类型 | 窗口长度 | 对应层级 | 时间跨度 |
|---------|---------|---------|---------|
| 短窗口 | $T_{\text{short}} = 10$ | 底层 | 10秒 |
| 中窗口 | $T_{\text{mid}} = 100$ | 中层 | 100秒 |
| 长窗口 | $T_{\text{long}} = 1000$ | 高层 | 1000秒 |

- **输入**：当前时刻 $t$，窗口长度 $T$
- **输出**：窗口 $W(t) = [t-T, t]$
- **约束**：$T > 0$；窗口内时刻数 = $T + 1$

> **白话翻译**：滑动窗口就是"只看最近N步"——不关心远古历史，只关注近期表现。窗口每步向前滑动一步，丢弃最旧的数据，加入最新的数据。窗口太短容易受噪声干扰，太长则反应迟钝。

---

### 第二部分：累积误差

#### 2.2 累积误差定义（A52.2）

累积误差是窗口内所有误差的简单求和：

$$E_W(t) = \sum_{s \in W(t)} \varepsilon_s = \sum_{s=t-T}^{t} \varepsilon_s \tag{A52.2}$$

其中 $\varepsilon_s$ 为时刻 $s$ 的全局误差（引用A51.5）。

**累积误差的递推计算**：

利用滑动窗口的更新规则，累积误差可以高效递推：

$$E_W(t+1) = E_W(t) + \varepsilon_{t+1} - \varepsilon_{t-T} \tag{A52.2-recursive}$$

递推计算的时间复杂度为 $O(1)$，而非重新求和的 $O(T)$。

**逐层累积误差**：

对每一层分别计算累积误差：

$$E_W^l(t) = \sum_{s=t-T}^{t} \varepsilon_s^l, \quad l \in \{\text{low, mid, high}\} \tag{A52.2-per-layer}$$

逐层累积误差保留了层级信息，为后续的层级敏感异常检测提供输入。

**累积误差与A3.3生存约束的关系**：

A3.3（引用阶段3）要求 $\sum_{t=0}^{T} \varepsilon_0(t) \leq -\ln P_{\min}$。当窗口长度 $T$ 足够大时，$E_W(t)$ 近似于A3.3中的累积误差。因此 $E_W(t) \leq -\ln P_{\min}$ 是系统在窗口 $W$ 内满足生存条件的必要条件。

**累积误差的性质**：

**(P1) 非负性**：$E_W(t) \geq 0$（由各 $\varepsilon_s \geq 0$ 保证）。

**(P2) 单调性**：若 $\varepsilon_{t+1} \geq \varepsilon_{t-T}$，则 $E_W(t+1) \geq E_W(t)$——新误差不小于旧误差时，累积误差不降。

**(P3) 有界性**：$E_W(t) \leq (T+1) \cdot \varepsilon_{\max}$，其中 $\varepsilon_{\max}$ 为窗口内最大单步误差。

- **输入**：窗口 $W(t)$，误差序列 $\{\varepsilon_s\}$
- **输出**：$E_W(t) \in \mathbb{R}_{\geq 0}$
- **约束**：窗口内数据完整

> **白话翻译**：累积误差就是"最近N步的总偏差"——窗口内所有误差简单求和。可以用递推公式高效计算：新累积 = 旧累积 + 新误差 - 旧误差。累积误差越大，说明系统在近期表现越差。累积误差有上界——不可能无限大，因为窗口长度有限。

---

### 第三部分：指数加权累积

#### 2.3 指数加权累积定义（A52.3）

指数加权移动平均（EWMA）对近期误差赋予更大权重：

$$E_{\text{ewma}}(t) = \alpha \cdot \varepsilon_t + (1 - \alpha) \cdot E_{\text{ewma}}(t-1) \tag{A52.3}$$

其中：
- $\alpha \in (0, 1)$ 为平滑因子（$\alpha$ 越大，对近期误差越敏感）
- $E_{\text{ewma}}(0) = \varepsilon_0$（初始值等于第一步误差）

**EWMA的展开形式**：

将A52.3递归展开：

$$E_{\text{ewma}}(t) = \alpha \sum_{s=0}^{t} (1-\alpha)^{t-s} \varepsilon_s \tag{A52.3-expand}$$

展开形式揭示了EWMA的本质——误差的指数衰减加权和，权重 $(1-\alpha)^{t-s}$ 随时间距离指数衰减。

**EWMA的有效窗口长度**：

EWMA的有效窗口长度定义为权重衰减到 $1/e$ 的时间步数：

$$T_{\text{eff}} = \frac{1}{\alpha} \tag{A52.3-effective}$$

| $\alpha$ | $T_{\text{eff}}$ | 含义 |
|----------|---------|------|
| 0.1 | 10 | 等效10步窗口 |
| 0.01 | 100 | 等效100步窗口 |
| 0.001 | 1000 | 等效1000步窗口 |

**EWMA与简单累积的对比**：

| 特性 | 简单累积 $E_W$ | EWMA $E_{\text{ewma}}$ |
|------|---------|---------|
| 权重 | 均匀 | 指数衰减 |
| 突变响应 | 立即反映（窗口内） | 逐步反映（按 $\alpha$ 速率） |
| 历史遗忘 | 硬截断（窗口外=0） | 软遗忘（指数衰减） |
| 计算复杂度 | $O(1)$（递推） | $O(1)$ |
| 存储需求 | 需存储窗口内数据 | 仅需存储上一值 |

**逐层EWMA**：

对每一层分别计算EWMA：

$$E_{\text{ewma}}^l(t) = \alpha_l \cdot \varepsilon_t^l + (1 - \alpha_l) \cdot E_{\text{ewma}}^l(t-1) \tag{A52.3-per-layer}$$

不同层可以使用不同的 $\alpha_l$——底层用较大的 $\alpha$（快感知），高层用较小的 $\alpha$（慢学习），与A17.1-A17.2的时间尺度分离一致。

- **输入**：当前误差 $\varepsilon_t$，上一EWMA值 $E_{\text{ewma}}(t-1)$，平滑因子 $\alpha$
- **输出**：$E_{\text{ewma}}(t) \in \mathbb{R}_{\geq 0}$
- **约束**：$\alpha \in (0, 1)$

> **白话翻译**：指数加权累积就是"越近越重要"——近期误差权重大，远期误差权重小，权重按指数衰减。$\alpha$ 控制"遗忘速度"——$\alpha$ 大则快速遗忘（只看最近几步），$\alpha$ 小则慢速遗忘（长期记忆）。和简单累积相比，EWMA不需要存储窗口内所有数据，只需记住上一个值。

---

### 第四部分：多尺度窗口

#### 2.4 多尺度窗口定义（A52.4）

多尺度窗口同时维护短、中、长三个窗口，捕捉不同时间尺度的误差特征：

$$[W_{\text{short}}, W_{\text{mid}}, W_{\text{long}}] = [10, 100, 1000] \tag{A52.4}$$

**多尺度累积误差**：

对每个尺度分别计算累积误差和EWMA：

$$\mathbf{E}(t) = [E_{\text{short}}(t), E_{\text{mid}}(t), E_{\text{long}}(t)] \tag{A52.4-vector}$$

$$\mathbf{E}_{\text{ewma}}(t) = [E_{\text{ewma}}^{\text{short}}(t), E_{\text{ewma}}^{\text{mid}}(t), E_{\text{ewma}}^{\text{long}}(t)] \tag{A52.4-ewma-vector}$$

其中：
- $E_{\text{short}}(t) = \sum_{s=t-9}^{t} \varepsilon_s$（10步窗口）
- $E_{\text{mid}}(t) = \sum_{s=t-99}^{t} \varepsilon_s$（100步窗口）
- $E_{\text{long}}(t) = \sum_{s=t-999}^{t} \varepsilon_s$（1000步窗口）

**多尺度误差趋势**：

通过比较不同尺度的累积误差，可以判断误差的变化趋势：

$$\text{trend}(t) = \frac{E_{\text{short}}(t) / T_{\text{short}}}{E_{\text{long}}(t) / T_{\text{long}}} \tag{A52.4-trend}$$

| trend值 | 含义 |
|---------|------|
| $\text{trend} \approx 1$ | 短期平均误差 ≈ 长期平均误差，系统稳定 |
| $\text{trend} > 1$ | 短期平均误差 > 长期平均误差，误差在恶化 |
| $\text{trend} < 1$ | 短期平均误差 < 长期平均误差，误差在改善 |

**多尺度与A50.4时间尺度的对齐**：

多尺度窗口与跨层时间尺度对齐（引用A50.4）：
- 短窗口（10步）对应底层时间尺度（1秒×10 = 10秒）
- 中窗口（100步）对应中层时间尺度（1秒×100 = 100秒 = 1.67分钟）
- 长窗口（1000步）对应高层时间尺度（1秒×1000 = 1000秒 ≈ 16.7分钟）

**多尺度EWMA的平滑因子**：

三个尺度的EWMA使用不同的 $\alpha$：

| 尺度 | 窗口长度 | $\alpha$ | $T_{\text{eff}}$ |
|------|---------|---------|---------|
| 短 | 10 | 0.1 | 10 |
| 中 | 100 | 0.01 | 100 |
| 长 | 1000 | 0.001 | 1000 |

- **输入**：误差序列 $\{\varepsilon_t\}$，三个窗口长度
- **输出**：多尺度累积误差向量 $\mathbf{E}(t)$，多尺度EWMA向量 $\mathbf{E}_{\text{ewma}}(t)$
- **约束**：$T_{\text{short}} < T_{\text{mid}} < T_{\text{long}}$

> **白话翻译**：多尺度窗口就是"短中长三个望远镜"——同时看10步、100步、1000步的误差趋势。短望远镜捕捉即时变化（"刚刚发生了什么"），中望远镜捕捉中期趋势（"最近几分钟怎么样"），长望远镜捕捉长期模式（"整体表现如何"）。比较短望远镜和长望远镜的平均误差，就能判断系统是在变好还是变差。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A52.1 | $W(t) = [t-T, t]$ | 当前时刻 $t$，窗口长度 $T$ | 时间窗口 | $T > 0$ |
| A52.2 | $E_W(t) = \sum_{s=t-T}^{t} \varepsilon_s$ | 窗口、误差序列 | 累积误差 $\in \mathbb{R}_{\geq 0}$ | 窗口内数据完整 |
| A52.3 | $E_{\text{ewma}}(t) = \alpha \varepsilon_t + (1-\alpha) E_{\text{ewma}}(t-1)$ | 当前误差、上一EWMA、$\alpha$ | EWMA值 $\in \mathbb{R}_{\geq 0}$ | $\alpha \in (0,1)$ |
| A52.4 | $[W_{\text{short}}, W_{\text{mid}}, W_{\text{long}}] = [10, 100, 1000]$ | 误差序列 | 多尺度累积/趋势向量 | $T_s < T_m < T_l$ |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum
import numpy as np

# 引用阶段51的数据类
from stage_051 import StandardizedError, WeightScheme


# ============================================================
# 阶段52 枚举：累积策略
# ============================================================
class AccumulationStrategy(Enum):
    """误差累积策略。"""
    SIMPLE_SUM = "simple_sum"    # 简单求和（A52.2）
    EWMA = "ewma"               # 指数加权移动平均（A52.3）
    MULTI_SCALE = "multi_scale"  # 多尺度（A52.4）


# ============================================================
# 阶段52 数据类：误差累积
# ============================================================
@dataclass(frozen=True)
class ErrorAccumulation:
    """误差的时间窗口累积描述。

    组合阶段51的StandardizedError，在标准化误差的基础上增加
    时间维度的累积策略——滑动窗口、累积误差、EWMA、多尺度。

    Attributes:
        standardized_error: 阶段51的标准化误差（组合字段）
        window_length: 当前窗口长度 T
        cumulative_error: 累积误差 E_W（A52.2）
        ewma_value: 指数加权累积 E_ewma（A52.3）
        ewma_alpha: EWMA平滑因子 α
        multi_scale_cumulative: 多尺度累积误差 [E_short, E_mid, E_long]
        multi_scale_ewma: 多尺度EWMA值
        multi_scale_alphas: 多尺度EWMA平滑因子 [α_short, α_mid, α_long]
        multi_scale_trend: 多尺度趋势指标 trend
        error_history: 窗口内的误差历史（用于递推计算）
        step_count: 累积步数

    Invariants:
        - cumulative_error >= 0
        - ewma_value >= 0
        - ewma_alpha ∈ (0, 1)
        - multi_scale_trend > 0
        - step_count >= 0
    """
    standardized_error: StandardizedError
    window_length: int = 100
    cumulative_error: float = 0.0
    ewma_value: float = 0.0
    ewma_alpha: float = 0.01
    multi_scale_cumulative: Optional[np.ndarray] = None
    multi_scale_ewma: Optional[np.ndarray] = None
    multi_scale_alphas: Optional[np.ndarray] = None
    multi_scale_trend: float = 1.0
    error_history: Optional[np.ndarray] = None
    step_count: int = 0

    def __post_init__(self):
        if self.window_length <= 0:
            raise ValueError(
                f"窗口长度必须 > 0，收到 {self.window_length}"
            )
        if self.cumulative_error < 0:
            raise ValueError(
                f"累积误差必须 >= 0，收到 {self.cumulative_error}"
            )
        if self.ewma_value < 0:
            raise ValueError(
                f"EWMA值必须 >= 0，收到 {self.ewma_value}"
            )
        if not (0 < self.ewma_alpha < 1):
            raise ValueError(
                f"EWMA平滑因子必须 ∈ (0,1)，收到 {self.ewma_alpha}"
            )
        if self.step_count < 0:
            raise ValueError(
                f"累积步数必须 >= 0，收到 {self.step_count}"
            )

    @property
    def effective_window(self) -> int:
        """EWMA有效窗口长度（A52.3-effective）。"""
        return int(1.0 / self.ewma_alpha)

    @property
    def is_improving(self) -> bool:
        """误差是否在改善（trend < 1）。"""
        return self.multi_scale_trend < 1.0

    @property
    def is_deteriorating(self) -> bool:
        """误差是否在恶化（trend > 1）。"""
        return self.multi_scale_trend > 1.0

    @property
    def is_stable(self) -> bool:
        """误差是否稳定（trend ≈ 1）。"""
        return 0.9 < self.multi_scale_trend < 1.1


# ============================================================
# 阶段52 核心接口：误差累积器
# ============================================================
class ErrorAccumulator:
    """阶段52的核心接口：误差累积器。

    实现误差的滑动窗口累积、EWMA和多尺度分析。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 累积误差非负
        - EWMA平滑因子 ∈ (0,1)
        - 多尺度窗口严格递增
    """

    def compute_cumulative_error(
        self,
        error_history: np.ndarray,
        window_length: int
    ) -> float:
        """计算滑动窗口累积误差（A52.2）。

        E_W = Σ_{s∈W} ε_s

        Args:
            error_history: 误差历史序列
            window_length: 窗口长度 T

        Returns:
            累积误差 E_W >= 0

        Raises:
            ValueError: 当 window_length <= 0 时
            ValueError: 当 error_history 长度不足时
        """
        if window_length <= 0:
            raise ValueError(
                f"窗口长度必须 > 0，收到 {window_length}"
            )
        if len(error_history) < window_length:
            raise ValueError(
                f"误差历史长度 {len(error_history)} 不足窗口长度 {window_length}"
            )
        window = error_history[-window_length:]
        return float(np.sum(window))

    def update_cumulative_recursive(
        self,
        prev_cumulative: float,
        new_error: float,
        oldest_error: float
    ) -> float:
        """递推更新累积误差（A52.2-recursive）。

        E_W(t+1) = E_W(t) + ε_{t+1} - ε_{t-T}

        Args:
            prev_cumulative: 上一累积误差
            new_error: 新误差 ε_{t+1}
            oldest_error: 最旧误差 ε_{t-T}

        Returns:
            更新后的累积误差
        """
        return prev_cumulative + new_error - oldest_error

    def update_ewma(
        self,
        prev_ewma: float,
        current_error: float,
        alpha: float
    ) -> float:
        """更新EWMA值（A52.3）。

        E_ewma(t) = α·ε_t + (1-α)·E_ewma(t-1)

        Args:
            prev_ewma: 上一EWMA值
            current_error: 当前误差 ε_t
            alpha: 平滑因子 α ∈ (0,1)

        Returns:
            更新后的EWMA值

        Raises:
            ValueError: 当 alpha ∉ (0,1) 时
        """
        if not (0 < alpha < 1):
            raise ValueError(
                f"alpha 必须 ∈ (0,1)，收到 {alpha}"
            )
        return alpha * current_error + (1.0 - alpha) * prev_ewma

    def compute_multi_scale(
        self,
        error_history: np.ndarray,
        window_lengths: tuple = (10, 100, 1000),
        alphas: tuple = (0.1, 0.01, 0.001)
    ) -> dict:
        """计算多尺度累积误差和EWMA（A52.4）。

        Args:
            error_history: 误差历史序列
            window_lengths: 三个窗口长度
            alphas: 三个EWMA平滑因子

        Returns:
            字典包含：
            - "cumulative": 多尺度累积误差 [E_short, E_mid, E_long]
            - "ewma": 多尺度EWMA值
            - "trend": 趋势指标

        Raises:
            ValueError: 当窗口长度不严格递增时
            ValueError: 当alpha数量与窗口数量不匹配时
        """
        if len(window_lengths) != 3 or len(alphas) != 3:
            raise ValueError(
                "窗口长度和 alphas 必须各有3个元素"
            )
        if not (window_lengths[0] < window_lengths[1] < window_lengths[2]):
            raise ValueError(
                "窗口长度必须严格递增"
            )
        cumulative = []
        ewma = []
        for i, (T, alpha) in enumerate(zip(window_lengths, alphas)):
            if len(error_history) < T:
                raise ValueError(
                    f"误差历史长度不足窗口 {T}"
                )
            # 累积误差
            cum = float(np.sum(error_history[-T:]))
            cumulative.append(cum)
            # EWMA（从窗口起始递推）
            e = float(error_history[-T])
            for val in error_history[-T+1:]:
                e = alpha * float(val) + (1.0 - alpha) * e
            ewma.append(e)
        cumulative = np.array(cumulative)
        ewma = np.array(ewma)
        # 趋势指标
        short_avg = cumulative[0] / window_lengths[0]
        long_avg = cumulative[2] / window_lengths[2]
        trend = self.compute_trend(short_avg, long_avg)
        return {
            "cumulative": cumulative,
            "ewma": ewma,
            "trend": trend,
        }

    def compute_trend(
        self,
        short_avg: float,
        long_avg: float
    ) -> float:
        """计算多尺度趋势指标（A52.4-trend）。

        trend = (E_short / T_short) / (E_long / T_long)

        Args:
            short_avg: 短窗口平均误差
            long_avg: 长窗口平均误差

        Returns:
            趋势指标 trend > 0

        Raises:
            ValueError: 当 long_avg <= 0 时
        """
        if long_avg <= 0:
            raise ValueError(
                f"长期平均误差必须 > 0，收到 {long_avg}"
            )
        return short_avg / long_avg

    def step(
        self,
        accumulation: ErrorAccumulation,
        new_error: StandardizedError
    ) -> ErrorAccumulation:
        """执行一步完整的误差累积更新。

        更新累积误差、EWMA和多尺度指标。

        Args:
            accumulation: 当前累积状态
            new_error: 新的标准化误差

        Returns:
            更新后的ErrorAccumulation

        Raises:
            ValueError: 当输入数据不合法时
        """
        epsilon_global = new_error.epsilon_global

        # 更新误差历史
        if accumulation.error_history is None:
            history = np.array([epsilon_global])
        else:
            history = np.append(accumulation.error_history, epsilon_global)

        # 更新累积误差（滑动窗口）
        T = accumulation.window_length
        if len(history) <= T:
            cumulative = float(np.sum(history))
        else:
            cumulative = float(np.sum(history[-T:]))

        # 递推更新累积误差（等价）
        # if len(history) > T:
        #     cumulative = self.update_cumulative_recursive(
        #         accumulation.cumulative_error, epsilon_global, history[-T-1]
        #     )

        # 更新 EWMA
        ewma = self.update_ewma(
            accumulation.ewma_value,
            epsilon_global,
            accumulation.ewma_alpha,
        )

        # 更新多尺度
        multi = self.compute_multi_scale(
            history,
            window_lengths=(10, 100, 1000),
            alphas=(0.1, 0.01, 0.001),
        )

        step_count = accumulation.step_count + 1

        return ErrorAccumulation(
            standardized_error=new_error,
            window_length=T,
            cumulative_error=cumulative,
            ewma_value=ewma,
            ewma_alpha=accumulation.ewma_alpha,
            multi_scale_cumulative=multi["cumulative"],
            multi_scale_ewma=multi["ewma"],
            multi_scale_alphas=np.array([0.1, 0.01, 0.001]),
            multi_scale_trend=multi["trend"],
            error_history=history,
            step_count=step_count,
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `StandardizedError` | ← 阶段51 | 作为误差累积的标准化误差输入（组合字段） |
| **输入** | A51.5（全局误差） | ← 阶段51 | 累积误差的计算基础 |
| **输入** | A3.3（累积误差生存约束） | ← 阶段3 | 累积误差上界的理论依据 |
| **输入** | A17.1-A17.2（快感知-慢学习） | ← 阶段17 | 多尺度窗口的时间尺度分离依据 |
| **输入** | A50.4（跨层时间尺度对齐） | ← 阶段50 | 多尺度窗口与层级时间尺度的对齐 |
| **输出** | `ErrorAccumulation` | → 阶段53 | 误差累积状态，作为反馈生成的输入 |
| **输出** | A52.1-A52.4（累积策略定义） | → 阶段55 | 异常检测的时间窗口基础 |
| **输出** | `ErrorAccumulator` | → 阶段59 | 误差累积器的接口 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 累积计算JIT编译 | `jax.lax.scan`实现高效递推累积和EWMA更新 |
| collections.deque | 滑动窗口实现 | 双端队列实现O(1)的窗口滑动操作 |
| NumPy | 多尺度向量化 | 支撑多尺度累积误差的向量化计算 |
| Pydantic | 数据验证 | 支撑ErrorAccumulation的运行时类型检查 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V52.1 | 累积误差正确性 | $E_W = \sum_{s \in W} \varepsilon_s$ | 手动求和对比 |
| V52.2 | 递推累积正确性 | 递推结果与重新求和一致 | 数值对比 |
| V52.3 | 累积误差非负 | $E_W \geq 0$ 对所有合法输入 | 随机测试 |
| V52.4 | EWMA正确性 | $E_{\text{ewma}}(t) = \alpha \varepsilon_t + (1-\alpha) E_{\text{ewma}}(t-1)$ | 逐步验证 |
| V52.5 | EWMA展开一致性 | 递推结果与展开形式一致 | 数值对比 |
| V52.6 | EWMA有效窗口 | $T_{\text{eff}} = 1/\alpha$ | 参数验证 |
| V52.7 | 多尺度累积正确性 | 三个尺度分别正确 | 逐一验证 |
| V52.8 | 多尺度趋势正确性 | trend = 短期平均/长期平均 | 数值验证 |
| V52.9 | 趋势判断正确 | 改善时trend<1，恶化时trend>1，稳定时trend≈1 | 合成数据测试 |
| V52.10 | 窗口长度非法拒绝 | $T \leq 0$ 时抛出 `ValueError` | 异常测试 |
| V52.11 | alpha非法拒绝 | $\alpha \notin (0,1)$ 时抛出 `ValueError` | 异常测试 |
| V52.12 | ErrorAccumulation组合 | `acc.standardized_error` 类型为 `StandardizedError` | 类型检查 |
| V52.13 | 递推效率 | 递推计算时间 < 重新求和时间的1/10 | 性能测试 |
| V52.14 | 长期累积稳定性 | 10000步EWMA累积无溢出或NaN | 长期运行测试 |
| V52.15 | 多尺度与时间尺度对齐 | 短/中/长窗口与底层/中层/高层时间尺度对应 | 对齐验证 |

---

> **阶段52完** | 阶段52建立了误差评估的时间窗口与累积策略：滑动窗口 $W = [t-T, t]$ 定义了误差累积的时间范围（A52.1），累积误差 $E_W = \sum_{t \in W} \varepsilon_t$ 提供了窗口内的总偏差度量（A52.2），指数加权累积 $E_{\text{ewma}} = \alpha \varepsilon_t + (1-\alpha) E_{\text{ewma}}^{\text{prev}}$ 实现了近期加权的平滑累积（A52.3），多尺度窗口 $[10, 100, 1000]$ 同时捕捉短中长三个时间尺度的误差特征（A52.4），封装为ErrorAccumulation数据类组合StandardizedError。下一步：阶段53将定义正向/负向反馈信号的生成机制。
