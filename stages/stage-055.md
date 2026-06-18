# 阶段55：异常误差检测——阈值检测、CUSUM与EWMA的统计过程控制

> **阶段编号**：S55 | **所属层**：第六层·误差评估层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 阈值检测 | "超标报警"——误差超过正常范围就报警 |
> | CUSUM | "持续偏高报警"——误差持续偏高就报警，即使每次都没超标 |
> | EWMA检测 | "趋势偏移报警"——误差的加权平均偏移就报警 |
> | 检测延迟与误报率 | "早报vs乱报"——报得太早容易误报，报得太晚又来不及 |
> | AnomalyDetector | "异常检测器的完整画像"——组合PrecisionWeightedError，在精度加权误差基础上增加异常检测 |

---

## 1. 第一性原理依据

本阶段从阶段54（精度加权误差）的核心结论出发，建立异常误差检测机制。核心依据为：

- **A54.2（精度加权误差）**（引用阶段54）：$\varepsilon_{\text{weighted}} = \sum_l w_l \cdot \mathbf{e}_l^T \Pi_l \mathbf{e}_l$，精度加权误差为异常检测提供了标准化的检测信号——精度加权后的误差更可靠，减少了噪声导致的误报。

- **A3.3（累积误差与生存条件）**（引用阶段3）：$\sum_{t=0}^{T} \varepsilon_0(t) \leq -\ln P_{\min}$，累积误差有上界。异常检测是A3.3的操作化——当累积误差接近或超过生存约束时，系统必须及时检测并响应。

- **A21.4（安全驱力与自由能）**（引用阶段21）：$R_{\text{safe}} = -\alpha_s \cdot \max(0, F - F_{\text{baseline}})$，安全驱力在自由能偏离基线时激活。异常检测为安全驱力提供触发信号——检测到异常时，安全驱力应被激活。

- **A52.3（EWMA累积）**（引用阶段52）：$E_{\text{ewma}}(t) = \alpha \varepsilon_t + (1-\alpha) E_{\text{ewma}}(t-1)$，EWMA为异常检测提供了平滑的检测信号。

本阶段的核心目标是：**定义阈值检测 $\varepsilon_t > \mu + k\sigma$、CUSUM $S_t = \max(0, S_{t-1} + \varepsilon_t - \mu_0 - k_\sigma)$、EWMA检测 $z_t = \lambda \varepsilon_t + (1-\lambda) z_{t-1}$、检测延迟与误报率的权衡分析，封装为AnomalyDetector数据类，组合PrecisionWeightedError作为精度加权误差的输入。** 异常检测是误差评估的"预警系统"——它在误差恶化到不可逆之前发出警报。

> **去重声明**：本阶段仅定义异常误差的检测算法，不涉及分布偏移检测（阶段56）或驱力映射（阶段57）。EWMA累积策略在阶段52定义，本阶段仅引用并扩展为检测用途。

---

## 2. 核心数学公式

### 第一部分：阈值检测

#### 2.1 阈值检测定义（A55.1）

阈值检测是最简单的异常检测方法——当误差超过正常范围的上下界时触发警报：

$$\text{anomaly}_{\text{threshold}}(t) = \mathbb{1}[\varepsilon_t > \mu + k \cdot \sigma] \tag{A55.1}$$

其中：
- $\varepsilon_t$ 为当前精度加权误差（引用A54.2）
- $\mu$ 为误差的稳态均值
- $\sigma$ 为误差的稳态标准差
- $k > 0$ 为阈值倍数（通常 $k = 3$，对应99.7%置信区间）
- $\mathbb{1}[\cdot]$ 为指示函数

**阈值倍数 $k$ 的选择**：

| $k$ | 置信区间（高斯假设） | 误报率 | 适用场景 |
|-----|-------------------|--------|---------|
| 2 | 95.4% | 4.6% | 宽松检测，宁可误报 |
| 3 | 99.7% | 0.3% | 标准检测 |
| 4 | 99.99% | 0.01% | 严格检测，避免误报 |
| 6 | 99.9999998% | ~0 | 极严格检测 |

**稳态参数的估计**：

$\mu$ 和 $\sigma$ 从历史误差数据中估计（引用A52.1的滑动窗口）：

$$\mu = \frac{1}{|W|} \sum_{s \in W} \varepsilon_s \tag{A55.1-mu}$$

$$\sigma = \sqrt{\frac{1}{|W|-1} \sum_{s \in W} (\varepsilon_s - \mu)^2} \tag{A55.1-sigma}$$

**阈值检测的局限性**：

阈值检测只能检测"瞬时超标"——如果误差持续偏高但每次都未超过阈值，阈值检测不会触发。这正是CUSUM和EWMA检测的优势所在。

- **输入**：当前误差 $\varepsilon_t$，稳态均值 $\mu$，稳态标准差 $\sigma$，阈值倍数 $k$
- **输出**：$\text{anomaly}_{\text{threshold}} \in \{0, 1\}$
- **约束**：$k > 0$；$\sigma > 0$

> **白话翻译**：阈值检测就是"超标报警"——误差超过正常范围的上下界就报警。$k$ 是"报警灵敏度"——$k$ 小则灵敏（容易误报），$k$ 大则迟钝（可能漏报）。标准选择 $k=3$，对应99.7%的置信区间——正常情况下只有0.3%的概率会误报。但阈值检测只能抓"瞬间超标"，持续偏高但每次都不超标的情况就抓不到。

---

### 第二部分：CUSUM检测

#### 2.2 CUSUM检测定义（A55.2）

CUSUM（累积和）检测通过跟踪误差的累积偏移来检测持续性的异常变化：

$$S_t = \max(0, S_{t-1} + \varepsilon_t - \mu_0 - k_\sigma) \tag{A55.2}$$

$$\text{anomaly}_{\text{CUSUM}}(t) = \mathbb{1}[S_t > h] \tag{A55.2-alarm}$$

其中：
- $S_t$ 为CUSUM统计量（非负，累积正偏移）
- $\mu_0$ 为目标均值（允许的稳态均值）
- $k_\sigma$ 为参考值（允许的偏移量，通常 $k_\sigma = \sigma / 2$）
- $h$ 为决策阈值（通常 $h = 5\sigma$）

**CUSUM的物理含义**：

CUSUM统计量 $S_t$ 追踪误差超出允许偏移 $k_\sigma$ 的累积量。当误差持续偏高时，$S_t$ 不断累积；当误差回归正常时，$S_t$ 被重置为零（通过 $\max(0, \cdot)$）。当 $S_t$ 超过决策阈值 $h$ 时，触发异常警报。

**CUSUM的参数选择**：

| 参数 | 推荐值 | 含义 |
|------|--------|------|
| $k_\sigma$ | $\sigma / 2$ | 允许偏移量——小于此偏移不累积 |
| $h$ | $5\sigma$ | 决策阈值——累积偏移超过此值报警 |

**CUSUM与阈值检测的对比**：

| 特性 | 阈值检测 | CUSUM |
|------|---------|-------|
| 检测瞬时超标 | ✓ | ✗（不擅长） |
| 检测持续偏移 | ✗ | ✓ |
| 检测延迟 | 低（1步） | 中（需要累积） |
| 误报率 | 取决于 $k$ | 取决于 $h$ |
| 对小偏移的灵敏度 | 低 | 高 |

**CUSUM的重置机制**：

当异常被确认并处理后，CUSUM统计量应重置：

$$S_t \leftarrow 0 \quad \text{（异常确认后）} \tag{A55.2-reset}$$

重置避免已处理的异常持续影响后续检测。

- **输入**：误差序列 $\{\varepsilon_t\}$，目标均值 $\mu_0$，参考值 $k_\sigma$，决策阈值 $h$
- **输出**：CUSUM统计量 $S_t$，异常标志 $\text{anomaly}_{\text{CUSUM}}$
- **约束**：$S_t \geq 0$；$k_\sigma > 0$；$h > 0$

> **白话翻译**：CUSUM就是"持续偏高报警"——每次误差超出允许偏移量，就把超出部分累积起来。累积量超过阈值就报警。即使每次误差都没超标，但持续偏高也会被CUSUM抓住——"温水煮青蛙"式的恶化逃不过CUSUM。$k_\sigma$ 是"允许偏移量"——小于此偏移不累积；$h$ 是"报警阈值"——累积量超过此值报警。

---

### 第三部分：EWMA检测

#### 2.3 EWMA检测定义（A55.3）

EWMA检测通过跟踪误差的指数加权移动平均来检测均值偏移：

$$z_t = \lambda \varepsilon_t + (1 - \lambda) z_{t-1} \tag{A55.3}$$

$$\text{anomaly}_{\text{EWMA}}(t) = \mathbb{1}[|z_t - \mu_0| > L \cdot \sigma_z] \tag{A55.3-alarm}$$

其中：
- $z_t$ 为EWMA统计量
- $\lambda \in (0, 1)$ 为平滑因子（通常 $\lambda = 0.1$）
- $\mu_0$ 为目标均值
- $\sigma_z$ 为EWMA统计量的稳态标准差
- $L > 0$ 为控制限倍数（通常 $L = 3$）

**EWMA统计量的稳态标准差**：

EWMA统计量的方差为：

$$\sigma_z^2 = \sigma^2 \cdot \frac{\lambda}{2 - \lambda} \tag{A55.3-var}$$

其中 $\sigma$ 为原始误差的标准差。当 $\lambda \to 1$ 时，$\sigma_z \to \sigma$（退化为阈值检测）；当 $\lambda \to 0$ 时，$\sigma_z \to 0$（极端平滑）。

**EWMA检测与A52.3 EWMA累积的关系**：

A55.3的EWMA检测与A52.3（引用阶段52）的EWMA累积在计算形式上完全一致——都是指数加权移动平均。区别在于用途：
- A52.3：EWMA用于误差的时间累积（平滑信号）
- A55.3：EWMA用于异常检测（检测均值偏移）

**EWMA检测与CUSUM的对比**：

| 特性 | CUSUM | EWMA |
|------|-------|------|
| 检测持续偏移 | ✓ | ✓ |
| 对历史数据的权重 | 均匀累积 | 指数衰减 |
| 参数调节 | $k_\sigma$, $h$ | $\lambda$, $L$ |
| 对小偏移的灵敏度 | 高 | 中 |
| 计算复杂度 | $O(1)$ | $O(1)$ |
| 稳态分析 | 复杂 | 简单（高斯近似） |

- **输入**：误差序列 $\{\varepsilon_t\}$，平滑因子 $\lambda$，控制限倍数 $L$
- **输出**：EWMA统计量 $z_t$，异常标志 $\text{anomaly}_{\text{EWMA}}$
- **约束**：$\lambda \in (0, 1)$；$L > 0$

> **白话翻译**：EWMA检测就是"趋势偏移报警"——跟踪误差的加权平均，当加权平均偏离目标值超过控制限时报警。和CUSUM的区别是：CUSUM均匀累积历史偏移，EWMA对近期偏移赋予更大权重。$\lambda$ 控制"遗忘速度"——$\lambda$ 大则快速遗忘（只看最近），$\lambda$ 小则慢速遗忘（长期记忆）。

---

### 第四部分：检测延迟与误报率的权衡

#### 2.4 检测延迟与误报率权衡分析（A55.4）

异常检测的核心权衡是检测延迟与误报率之间的矛盾：

$$\text{检测延迟} \downarrow \iff \text{误报率} \uparrow \tag{A55.4}$$

**权衡的数学表述**：

对于给定的异常幅度 $\delta$（均值偏移量），检测延迟 $D$ 和平均运行长度 $\text{ARL}_0$（无异常时的平均报警间隔）满足：

$$D(\delta) \propto \frac{\ln(\text{ARL}_0)}{\delta} \tag{A55.4-tradeoff}$$

即：要降低误报率（增大 $\text{ARL}_0$），必须接受更长的检测延迟；异常幅度越小，检测延迟越长。

**三种检测方法的权衡特性**：

| 方法 | 小偏移检测 | 大偏移检测 | 误报率控制 | 推荐场景 |
|------|----------|----------|----------|---------|
| 阈值检测 | 差 | 优 | 取决于 $k$ | 突变检测 |
| CUSUM | 优 | 良 | 取决于 $h$ | 持续偏移检测 |
| EWMA | 良 | 良 | 取决于 $L$ | 一般偏移检测 |

**组合检测策略**：

为同时覆盖不同类型的异常，采用三种方法的组合：

$$\text{anomaly}_{\text{combined}}(t) = \text{anomaly}_{\text{threshold}}(t) \lor \text{anomaly}_{\text{CUSUM}}(t) \lor \text{anomaly}_{\text{EWMA}}(t) \tag{A55.4-combined}$$

组合检测的优势：
1. 阈值检测捕捉瞬时突变
2. CUSUM捕捉持续小偏移
3. EWMA捕捉趋势偏移
4. 三者互补，覆盖所有异常类型

**异常分数**：

将三种检测方法的结果融合为连续的异常分数：

$$\text{anomaly\_score}(t) = w_1 \cdot \frac{\max(0, \varepsilon_t - \mu - k\sigma)}{k\sigma} + w_2 \cdot \frac{S_t}{h} + w_3 \cdot \frac{|z_t - \mu_0|}{L \cdot \sigma_z} \tag{A55.4-score}$$

其中 $w_1 + w_2 + w_3 = 1$。异常分数 $\in [0, \infty)$，越大表示异常越严重。

- **输入**：三种检测方法的结果，权重
- **输出**：组合异常标志，异常分数
- **约束**：权重归一化

> **白话翻译**：检测延迟和误报率是"早报vs乱报"的矛盾——报得太早容易误报（把噪声当异常），报得太晚又来不及（异常已经恶化）。三种方法各有擅长：阈值检测擅长抓突变，CUSUM擅长抓持续偏移，EWMA擅长抓趋势偏移。组合使用三者，可以覆盖所有类型的异常。异常分数把三种方法的结果融合为一个0到无穷的连续值——越大越异常。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A55.1 | $\text{anomaly}_{\text{threshold}} = \mathbb{1}[\varepsilon_t > \mu + k\sigma]$ | 误差、均值、标准差、$k$ | 二值标志 | $k > 0$ |
| A55.2 | $S_t = \max(0, S_{t-1} + \varepsilon_t - \mu_0 - k_\sigma)$ | 误差序列、$\mu_0$、$k_\sigma$、$h$ | CUSUM统计量+标志 | $S_t \geq 0$ |
| A55.3 | $z_t = \lambda \varepsilon_t + (1-\lambda) z_{t-1}$ | 误差序列、$\lambda$、$L$ | EWMA统计量+标志 | $\lambda \in (0,1)$ |
| A55.4 | $\text{anomaly\_score} = w_1 \cdot \text{threshold} + w_2 \cdot \text{CUSUM} + w_3 \cdot \text{EWMA}$ | 三种检测结果 | 异常分数 | 权重归一化 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from enum import Enum
import numpy as np

# 引用阶段54的数据类
from stage_054 import PrecisionWeightedError


# ============================================================
# 阶段55 枚举：异常类型
# ============================================================
class AnomalyType(Enum):
    """异常类型。"""
    THRESHOLD = "threshold"    # 阈值超标
    CUSUM = "cusum"           # CUSUM持续偏移
    EWMA = "ewma"             # EWMA趋势偏移
    COMBINED = "combined"     # 组合检测


# ============================================================
# 阶段55 数据类：异常报告
# ============================================================
@dataclass(frozen=True)
class AnomalyReport:
    """异常检测报告。"""
    is_anomaly: bool
    anomaly_type: AnomalyType
    anomaly_score: float
    threshold_flag: bool
    cusum_flag: bool
    ewma_flag: bool
    cusum_statistic: float
    ewma_statistic: float
    threshold_value: float
    detection_delay: int = 0


# ============================================================
# 阶段55 数据类：异常检测器
# ============================================================
@dataclass(frozen=True)
class AnomalyDetector:
    """异常误差检测器的完整描述。

    组合阶段54的PrecisionWeightedError，在精度加权误差的基础上增加
    异常检测能力——阈值检测、CUSUM、EWMA、组合策略。

    Attributes:
        precision_weighted_error: 阶段54的精度加权误差（组合字段）
        threshold_k: 阈值倍数 k
        cusum_statistic: 当前CUSUM统计量 S_t
        cusum_k_sigma: CUSUM参考值 k_σ
        cusum_h: CUSUM决策阈值 h
        ewma_statistic: 当前EWMA统计量 z_t
        ewma_lambda: EWMA平滑因子 λ
        ewma_L: EWMA控制限倍数 L
        error_mean: 误差稳态均值 μ
        error_std: 误差稳态标准差 σ
        anomaly_score: 组合异常分数
        score_weights: 异常分数权重 [w1, w2, w3]
        last_anomaly_report: 最近一次异常报告

    Invariants:
        - threshold_k > 0
        - cusum_statistic >= 0
        - cusum_h > 0
        - ewma_lambda ∈ (0, 1)
        - ewma_L > 0
        - error_std > 0
        - anomaly_score >= 0
        - score_weights 归一化
    """
    precision_weighted_error: PrecisionWeightedError
    threshold_k: float = 3.0
    cusum_statistic: float = 0.0
    cusum_k_sigma: float = 0.0
    cusum_h: float = 0.0
    ewma_statistic: float = 0.0
    ewma_lambda: float = 0.1
    ewma_L: float = 3.0
    error_mean: float = 0.0
    error_std: float = 1.0
    anomaly_score: float = 0.0
    score_weights: np.ndarray = None
    last_anomaly_report: Optional[AnomalyReport] = None

    def __post_init__(self):
        if self.threshold_k <= 0:
            raise ValueError(
                f"阈值倍数必须 > 0，收到 {self.threshold_k}"
            )
        if self.cusum_statistic < 0:
            raise ValueError(
                f"CUSUM统计量必须 >= 0，收到 {self.cusum_statistic}"
            )
        if self.cusum_h <= 0:
            raise ValueError(
                f"CUSUM决策阈值必须 > 0，收到 {self.cusum_h}"
            )
        if not (0 < self.ewma_lambda < 1):
            raise ValueError(
                f"EWMA平滑因子必须 ∈ (0,1)，收到 {self.ewma_lambda}"
            )
        if self.ewma_L <= 0:
            raise ValueError(
                f"EWMA控制限倍数必须 > 0，收到 {self.ewma_L}"
            )
        if self.error_std <= 0:
            raise ValueError(
                f"误差标准差必须 > 0，收到 {self.error_std}"
            )
        if self.anomaly_score < 0:
            raise ValueError(
                f"异常分数必须 >= 0，收到 {self.anomaly_score}"
            )
        if self.score_weights is not None:
            if not np.allclose(np.sum(self.score_weights), 1.0):
                raise ValueError("异常分数权重必须归一化")

    @property
    def is_anomaly(self) -> bool:
        """是否检测到异常。"""
        if self.last_anomaly_report is None:
            return False
        return self.last_anomaly_report.is_anomaly

    @property
    def anomaly_severity(self) -> str:
        """异常严重程度。"""
        if self.anomaly_score < 1.0:
            return "low"
        elif self.anomaly_score < 3.0:
            return "medium"
        else:
            return "high"


# ============================================================
# 阶段55 核心接口：异常检测引擎
# ============================================================
class AnomalyDetectionEngine:
    """阶段55的核心接口：异常检测引擎。

    实现阈值检测、CUSUM检测、EWMA检测和组合策略。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - CUSUM统计量非负
        - EWMA平滑因子 ∈ (0,1)
        - 异常分数非负
    """

    def detect_threshold(
        self,
        error: float,
        mean: float,
        std: float,
        k: float
    ) -> bool:
        """阈值检测（A55.1）。

        anomaly = 1[ε_t > μ + k·σ]

        Args:
            error: 当前误差 ε_t
            mean: 稳态均值 μ
            std: 稳态标准差 σ
            k: 阈值倍数

        Returns:
            是否异常

        Raises:
            ValueError: 当 k <= 0 或 std <= 0 时
        """
        if k <= 0:
            raise ValueError(f"k 必须 > 0，收到 {k}")
        if std <= 0:
            raise ValueError(f"std 必须 > 0，收到 {std}")
        return error > mean + k * std

    def update_cusum(
        self,
        prev_statistic: float,
        error: float,
        mu_0: float,
        k_sigma: float
    ) -> float:
        """更新CUSUM统计量（A55.2）。

        S_t = max(0, S_{t-1} + ε_t - μ_0 - k_σ)

        Args:
            prev_statistic: 上一CUSUM统计量 S_{t-1}
            error: 当前误差 ε_t
            mu_0: 目标均值 μ_0
            k_sigma: 参考值 k_σ

        Returns:
            更新后的CUSUM统计量 S_t >= 0
        """
        return max(0.0, prev_statistic + error - mu_0 - k_sigma)

    def detect_cusum(
        self,
        statistic: float,
        h: float
    ) -> bool:
        """CUSUM异常判定（A55.2-alarm）。

        anomaly = 1[S_t > h]

        Args:
            statistic: CUSUM统计量 S_t
            h: 决策阈值

        Returns:
            是否异常
        """
        return statistic > h

    def update_ewma(
        self,
        prev_statistic: float,
        error: float,
        lam: float
    ) -> float:
        """更新EWMA统计量（A55.3）。

        z_t = λ·ε_t + (1-λ)·z_{t-1}

        Args:
            prev_statistic: 上一EWMA统计量 z_{t-1}
            error: 当前误差 ε_t
            lam: 平滑因子 λ ∈ (0,1)

        Returns:
            更新后的EWMA统计量 z_t

        Raises:
            ValueError: 当 lam ∉ (0,1) 时
        """
        if not (0 < lam < 1):
            raise ValueError(f"lam 必须 ∈ (0,1)，收到 {lam}")
        return lam * error + (1.0 - lam) * prev_statistic

    def detect_ewma(
        self,
        statistic: float,
        mean: float,
        sigma_z: float,
        L: float
    ) -> bool:
        """EWMA异常判定（A55.3-alarm）。

        anomaly = 1[|z_t - μ_0| > L·σ_z]

        Args:
            statistic: EWMA统计量 z_t
            mean: 目标均值 μ_0
            sigma_z: EWMA稳态标准差
            L: 控制限倍数

        Returns:
            是否异常
        """
        return abs(statistic - mean) > L * sigma_z

    def compute_anomaly_score(
        self,
        error: float,
        mean: float,
        std: float,
        k: float,
        cusum_stat: float,
        h: float,
        ewma_stat: float,
        sigma_z: float,
        L: float,
        weights: np.ndarray
    ) -> float:
        """计算组合异常分数（A55.4）。

        Args:
            error: 当前误差
            mean: 稳态均值
            std: 稳态标准差
            k: 阈值倍数
            cusum_stat: CUSUM统计量
            h: CUSUM决策阈值
            ewma_stat: EWMA统计量
            sigma_z: EWMA稳态标准差
            L: EWMA控制限倍数
            weights: 权重 [w1, w2, w3]

        Returns:
            异常分数 >= 0
        """
        if not np.allclose(np.sum(weights), 1.0):
            raise ValueError("异常分数权重必须归一化")

        # 阈值检测分量
        threshold_flag = self.detect_threshold(error, mean, std, k)
        threshold_component = max(0.0, error - mean - k * std) / (k * std) if (k * std) > 0 else 0.0

        # CUSUM 分量
        cusum_component = cusum_stat / h if h > 0 else 0.0

        # EWMA 分量
        ewma_component = abs(ewma_stat - mean) / (L * sigma_z) if (L * sigma_z) > 0 else 0.0

        score = (
            weights[0] * threshold_component
            + weights[1] * cusum_component
            + weights[2] * ewma_component
        )
        return max(0.0, score)

    def step(
        self,
        detector: AnomalyDetector,
        current_error: float
    ) -> AnomalyDetector:
        """执行一步完整的异常检测更新。

        Args:
            detector: 当前检测器状态
            current_error: 当前精度加权误差

        Returns:
            更新后的AnomalyDetector
        """
        error = current_error
        mean = detector.error_mean
        std = detector.error_std
        k = detector.threshold_k

        # 更新 CUSUM
        cusum_stat = self.update_cusum(
            detector.cusum_statistic, error, mean, detector.cusum_k_sigma
        )
        cusum_flag = self.detect_cusum(cusum_stat, detector.cusum_h)

        # 更新 EWMA
        ewma_stat = self.update_ewma(
            detector.ewma_statistic, error, detector.ewma_lambda
        )
        # EWMA 稳态标准差
        lam = detector.ewma_lambda
        sigma_z = std * np.sqrt(lam / (2.0 - lam)) if (2.0 - lam) > 0 else std
        ewma_flag = self.detect_ewma(
            ewma_stat, mean, sigma_z, detector.ewma_L
        )

        # 阈值检测
        threshold_flag = self.detect_threshold(error, mean, std, k)

        # 组合异常分数
        weights = (detector.score_weights
                   if detector.score_weights is not None
                   else np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]))
        score = self.compute_anomaly_score(
            error, mean, std, k,
            cusum_stat, detector.cusum_h,
            ewma_stat, sigma_z, detector.ewma_L,
            weights,
        )

        is_anomaly = threshold_flag or cusum_flag or ewma_flag
        anomaly_type = AnomalyType.COMBINED
        if is_anomaly:
            if threshold_flag and not cusum_flag and not ewma_flag:
                anomaly_type = AnomalyType.THRESHOLD
            elif cusum_flag and not threshold_flag and not ewma_flag:
                anomaly_type = AnomalyType.CUSUM
            elif ewma_flag and not threshold_flag and not cusum_flag:
                anomaly_type = AnomalyType.EWMA

        report = AnomalyReport(
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type,
            anomaly_score=score,
            threshold_flag=threshold_flag,
            cusum_flag=cusum_flag,
            ewma_flag=ewma_flag,
            cusum_statistic=cusum_stat,
            ewma_statistic=ewma_stat,
            threshold_value=mean + k * std,
        )

        # 返回更新后的检测器（注意 dataclass 为 frozen，需要构造新实例）
        return AnomalyDetector(
            precision_weighted_error=detector.precision_weighted_error,
            threshold_k=k,
            cusum_statistic=cusum_stat,
            cusum_k_sigma=detector.cusum_k_sigma,
            cusum_h=detector.cusum_h,
            ewma_statistic=ewma_stat,
            ewma_lambda=lam,
            ewma_L=detector.ewma_L,
            error_mean=mean,
            error_std=std,
            anomaly_score=score,
            score_weights=weights,
            last_anomaly_report=report,
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `PrecisionWeightedError` | ← 阶段54 | 作为异常检测的精度加权误差输入（组合字段） |
| **输入** | A54.2（精度加权误差） | ← 阶段54 | 异常检测的信号来源 |
| **输入** | A52.3（EWMA累积） | ← 阶段52 | EWMA检测的理论基础 |
| **输入** | A3.3（累积误差生存约束） | ← 阶段3 | 异常检测的物理动机 |
| **输出** | `AnomalyDetector` | → 阶段56 | 异常检测器状态，作为分布偏移检测的输入 |
| **输出** | `AnomalyReport` | → 阶段57 | 异常报告，驱力映射的输入 |
| **输出** | A55.1-A55.4（异常检测定义） | → 阶段59-60 | 异常检测的基础定义 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 检测算法JIT编译 | `jax.lax.cond`实现条件检测的JIT编译 |
| NumPy | 统计量计算 | 支撑CUSUM和EWMA的向量化计算 |
| SciPy | 统计分布 | `scipy.stats.norm`计算高斯分布下的ARL |
| Pydantic | 数据验证 | 支撑AnomalyDetector的运行时类型检查 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V55.1 | 阈值检测正确性 | $\varepsilon > \mu + k\sigma$ 时触发 | 合成数据测试 |
| V55.2 | 阈值检测不误报 | 稳态数据误报率 < 预期值 | 高斯噪声测试 |
| V55.3 | CUSUM统计量非负 | $S_t \geq 0$ 对所有输入 | 随机测试 |
| V55.4 | CUSUM检测持续偏移 | 持续小偏移被检测到 | 合成偏移测试 |
| V55.5 | CUSUM重置有效 | 重置后 $S_t = 0$ | 重置测试 |
| V55.6 | EWMA统计量正确 | $z_t = \lambda \varepsilon_t + (1-\lambda) z_{t-1}$ | 逐步验证 |
| V55.7 | EWMA检测均值偏移 | 均值偏移被检测到 | 合成偏移测试 |
| V55.8 | EWMA方差正确 | $\sigma_z^2 = \sigma^2 \lambda / (2-\lambda)$ | 方差验证 |
| V55.9 | 组合检测覆盖全面 | 突变+持续偏移+趋势偏移均被检测 | 综合测试 |
| V55.10 | 异常分数非负 | anomaly_score >= 0 | 随机测试 |
| V55.11 | 检测延迟合理 | 突变检测延迟 < 5步 | 延迟测试 |
| V55.12 | 误报率可控 | 稳态误报率 < 5% | 长期稳态测试 |
| V55.13 | AnomalyDetector组合 | `detector.precision_weighted_error` 类型为 `PrecisionWeightedError` | 类型检查 |
| V55.14 | 非法输入拒绝 | $k \leq 0$、$\lambda \notin (0,1)$ 抛出 `ValueError` | 异常测试 |
| V55.15 | 长期检测稳定性 | 10000步检测无退化 | 长期运行测试 |

---

> **阶段55完** | 阶段55建立了异常误差检测机制：阈值检测 $\varepsilon_t > \mu + k\sigma$ 捕捉瞬时超标（A55.1），CUSUM $S_t = \max(0, S_{t-1} + \varepsilon_t - \mu_0 - k_\sigma)$ 捕捉持续偏移（A55.2），EWMA $z_t = \lambda \varepsilon_t + (1-\lambda) z_{t-1}$ 捕捉趋势偏移（A55.3），组合策略融合三种方法覆盖所有异常类型（A55.4），封装为AnomalyDetector数据类组合PrecisionWeightedError。下一步：阶段56将定义分布偏移检测机制。
