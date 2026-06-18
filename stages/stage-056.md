# 阶段56：分布偏移检测——KS检验、MMD与在线分布监控

> **阶段编号**：S56 | **所属层**：第六层·误差评估层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | KS检验 | "分布形状对比"——比较当前误差分布和历史分布的最大差距 |
> | MMD | "分布距离"——用核函数度量两个分布的"远近" |
> | 在线分布偏移检测 | "实时监控分布变化"——滑动参考窗口，持续对比 |
> | 偏移程度量化 | "偏移有多严重"——偏移量除以阈值，得到归一化的严重度 |
> | DistributionShiftDetector | "分布偏移检测器的完整画像"——组合AnomalyDetector，在异常检测基础上增加分布偏移检测 |

---

## 1. 第一性原理依据

本阶段从阶段55（异常误差检测）的核心结论出发，建立分布偏移检测机制。核心依据为：

- **A55.1-A55.4（异常误差检测）**（引用阶段55）：阈值检测、CUSUM、EWMA三种方法检测误差的异常值。但异常检测只能检测"误差值异常"，不能检测"误差分布变化"——当误差的均值不变但方差增大时，异常检测可能不触发，但分布已经发生了偏移。分布偏移检测补充了这一盲区。

- **A3.7（误差动力学）**（引用阶段3）：$\tau_l \dot{e}^{l-1} = -e^{l-1} + r^{l-1} - f^l(r^l, W^l)$，误差动力学方程。当环境分布发生变化时，误差动力学方程的稳态分布也会变化——分布偏移检测正是通过监控误差分布的变化来间接检测环境变化。

- **A54.3（自适应精度更新）**（引用阶段54）：$\Pi_l(t+1) = (1-\beta)\Pi_l(t) + \beta \Pi_l^{\text{observed}}(t)$，精度矩阵的自适应更新。分布偏移检测为精度更新提供触发信号——当检测到分布偏移时，应增大 $\beta$ 以加速精度适应。

- **A21.4（安全驱力与自由能）**（引用阶段21）：$R_{\text{safe}} = -\alpha_s \cdot \max(0, F - F_{\text{baseline}})$。分布偏移是安全驱力的重要触发源——环境分布变化意味着系统对环境的理解可能已经过时。

本阶段的核心目标是：**定义KS检验 $D = \sup|F_{\text{obs}}(x) - F_{\text{ref}}(x)|$、MMD $\text{MMD}^2 = \mathbb{E}[k(x,x')] - 2\mathbb{E}[k(x,y')] + \mathbb{E}[k(y,y')]$、在线分布偏移检测（滑动参考窗口）、偏移程度量化 $\text{shift\_magnitude} = D / D_{\text{threshold}}$，封装为DistributionShiftDetector数据类，组合AnomalyDetector作为异常检测的输入。** 分布偏移检测是异常检测的"升级版"——它不只看误差值是否异常，还看误差的分布是否发生了根本性变化。

> **去重声明**：本阶段仅定义分布偏移的检测算法，不涉及驱力映射（阶段57）或模型更新（阶段58）。异常检测在阶段55定义，本阶段不重复。

---

## 2. 核心数学公式

### 第一部分：KS检验

#### 2.1 KS检验定义（A56.1）

Kolmogorov-Smirnov检验通过比较经验累积分布函数（ECDF）的最大差距来检测分布偏移：

$$D = \sup_x |F_{\text{obs}}(x) - F_{\text{ref}}(x)| \tag{A56.1}$$

其中：
- $F_{\text{obs}}(x)$ 为观测样本的经验累积分布函数
- $F_{\text{ref}}(x)$ 为参考样本的经验累积分布函数
- $D$ 为KS统计量，取值范围 $[0, 1]$

**经验累积分布函数**：

$$F_n(x) = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}[X_i \leq x] \tag{A56.1-ecdf}$$

其中 $n$ 为样本量，$X_i$ 为第 $i$ 个样本。

**KS检验的判定**：

分布偏移的判定基于KS统计量是否超过临界值：

$$\text{shift}_{\text{KS}} = \mathbb{1}\left[D > c(\alpha, n_{\text{obs}}, n_{\text{ref}})\right] \tag{A56.1-test}$$

其中 $c(\alpha, n_{\text{obs}}, n_{\text{ref}})$ 为显著性水平 $\alpha$ 下的临界值。对于大样本：

$$c(\alpha, n) \approx \sqrt{-\frac{1}{2}\ln(\alpha/2)} \cdot \sqrt{\frac{n_{\text{obs}} + n_{\text{ref}}}{n_{\text{obs}} \cdot n_{\text{ref}}}} \tag{A56.1-critical}$$

**KS检验的优势与局限**：

| 优势 | 局限 |
|------|------|
| 非参数检验，无需假设分布形状 | 对尾部偏移不敏感（只看最大差距） |
| 计算简单高效 | 对高维数据需要逐维检测 |
| 统计理论完善 | 只能检测一维分布偏移 |

**逐层KS检验**：

对每一层的误差分别进行KS检验：

$$D_l = \sup_x |F_{\text{obs}}^l(x) - F_{\text{ref}}^l(x)|, \quad l \in \{\text{low, mid, high}\} \tag{A56.1-per-layer}$$

- **输入**：观测样本、参考样本
- **输出**：KS统计量 $D \in [0, 1]$，偏移标志
- **约束**：样本量 $\geq 10$

> **白话翻译**：KS检验就是"分布形状对比"——把当前误差的累积分布曲线和历史分布曲线叠在一起，看最大的差距有多大。差距越大，说明分布偏移越严重。KS检验不需要假设分布的形状（非参数），但只看最大差距，对尾部变化不敏感。

---

### 第二部分：MMD

#### 2.2 MMD定义（A56.2）

最大均值差异（Maximum Mean Discrepancy）通过核函数度量两个分布的距离：

$$\text{MMD}^2 = \mathbb{E}[k(x, x')] - 2\mathbb{E}[k(x, y')] + \mathbb{E}[k(y, y')] \tag{A56.2}$$

其中：
- $x, x'$ 独立同分布于观测分布 $P$
- $y, y'$ 独立同分布于参考分布 $Q$
- $k(\cdot, \cdot)$ 为正定核函数

**核函数的选择**：

常用核函数为高斯核（RBF核）：

$$k(x, y) = \exp\left(-\frac{\|x - y\|^2}{2\sigma_k^2}\right) \tag{A56.2-rbf}$$

其中 $\sigma_k$ 为核带宽参数。核带宽的选择对MMD的灵敏度至关重要：
- $\sigma_k$ 太小：MMD对局部差异敏感，但方差大
- $\sigma_k$ 太大：MMD平滑了差异，灵敏度低

**多尺度MMD**：

使用多个核带宽的MMD组合，提高检测的鲁棒性：

$$\text{MMD}^2_{\text{multi}} = \sum_{i=1}^{M} w_i \cdot \text{MMD}^2(\sigma_k^{(i)}) \tag{A56.2-multi}$$

其中 $\sigma_k^{(1)} < \sigma_k^{(2)} < \cdots < \sigma_k^{(M)}$，$w_i$ 为各尺度的权重。

**MMD的无偏估计**：

给定观测样本 $\{x_i\}_{i=1}^n$ 和参考样本 $\{y_j\}_{j=1}^m$，MMD的无偏估计为：

$$\widehat{\text{MMD}}^2 = \frac{1}{n(n-1)} \sum_{i \neq j} k(x_i, x_j) - \frac{2}{nm} \sum_{i,j} k(x_i, y_j) + \frac{1}{m(m-1)} \sum_{i \neq j} k(y_i, y_j) \tag{A56.2-estimate}$$

**MMD与KS的对比**：

| 特性 | KS | MMD |
|------|-----|-----|
| 维度 | 一维 | 多维 |
| 分布假设 | 无 | 无 |
| 计算复杂度 | $O(n \log n)$ | $O(n^2)$ |
| 尾部灵敏度 | 低 | 中 |
| 多维支持 | 否 | 是 |

- **输入**：观测样本、参考样本、核函数
- **输出**：$\text{MMD}^2 \in \mathbb{R}_{\geq 0}$
- **约束**：核函数正定；样本量 $\geq 10$

> **白话翻译**：MMD就是"分布距离"——用核函数度量两个分布的"远近"。核函数就像一个"相似度尺子"——同分布的样本互相相似（核值大），不同分布的样本互相不相似（核值小）。MMD把"自己人和自己人的相似度"减去"自己人和外人的相似度"，差值越大说明两个分布越远。MMD支持多维数据，比KS更强大，但计算更慢。

---

### 第三部分：在线分布偏移检测

#### 2.3 在线分布偏移检测定义（A56.3）

在线分布偏移检测使用滑动参考窗口持续监控分布变化：

$$\text{reference\_window}(t) = [t - T_{\text{ref}} - T_{\text{obs}}, t - T_{\text{obs}}] \tag{A56.3-ref}$$

$$\text{observation\_window}(t) = [t - T_{\text{obs}}, t] \tag{A56.3-obs}$$

其中：
- $T_{\text{ref}}$ 为参考窗口长度
- $T_{\text{obs}}$ 为观测窗口长度
- 参考窗口和观测窗口不重叠

**在线检测流程**：

```
每一步：
1. 更新观测窗口：加入新数据，丢弃最旧数据
2. 更新参考窗口：如果需要，滑动参考窗口
3. 计算KS统计量 D 或 MMD²
4. 与阈值比较，判定是否发生分布偏移
5. 如果偏移确认，更新参考窗口为当前观测窗口
```

**参考窗口的更新策略**：

| 策略 | 描述 | 优势 | 劣势 |
|------|------|------|------|
| 固定参考 | 参考窗口始终为初始数据 | 检测任何偏离初始分布的变化 | 对渐进漂移不敏感 |
| 滑动参考 | 参考窗口随时间滑动 | 适应渐进漂移 | 可能漏检慢速漂移 |
| 自适应参考 | 偏移确认后更新参考 | 平衡灵敏度和适应性 | 更新逻辑复杂 |

**在线检测的延迟**：

在线检测的延迟取决于观测窗口长度 $T_{\text{obs}}$：
- $T_{\text{obs}}$ 太短：样本量不足，检测不可靠
- $T_{\text{obs}}$ 太长：检测延迟大

推荐 $T_{\text{obs}} \geq 50$，$T_{\text{ref}} \geq 100$。

- **输入**：误差数据流，窗口长度
- **输出**：在线偏移标志，KS/MMD统计量
- **约束**：$T_{\text{ref}} \geq T_{\text{obs}}$；窗口不重叠

> **白话翻译**：在线分布偏移检测就是"实时监控分布变化"——维护两个窗口：参考窗口（"以前长什么样"）和观测窗口（"现在长什么样"），持续对比。参考窗口可以固定（和初始分布比）、滑动（和最近的过去比）或自适应（偏移确认后更新）。关键是要有足够的样本量——窗口太短检测不可靠，太长又反应慢。

---

### 第四部分：偏移程度量化

#### 2.4 偏移程度量化定义（A56.4）

偏移程度通过归一化的偏移量量化：

$$\text{shift\_magnitude} = \frac{D}{D_{\text{threshold}}} \tag{A56.4-ks}$$

或使用MMD：

$$\text{shift\_magnitude} = \frac{\text{MMD}^2}{\text{MMD}^2_{\text{threshold}}} \tag{A56.4-mmd}$$

其中 $D_{\text{threshold}}$ 和 $\text{MMD}^2_{\text{threshold}}$ 为对应的检测阈值。

**偏移程度的物理含义**：

| shift_magnitude | 含义 |
|----------------|------|
| $< 1$ | 未检测到分布偏移 |
| $= 1$ | 刚好达到偏移阈值 |
| $1 \sim 3$ | 轻度偏移 |
| $3 \sim 10$ | 中度偏移 |
| $> 10$ | 严重偏移 |

**逐层偏移程度**：

对每一层分别计算偏移程度：

$$\text{shift\_magnitude}_l = \frac{D_l}{D_{l,\text{threshold}}}, \quad l \in \{\text{low, mid, high}\} \tag{A56.4-per-layer}$$

**偏移方向**：

偏移方向由观测分布相对于参考分布的均值偏移确定：

$$\text{shift\_direction}_l = \text{sign}(\mu_{\text{obs}}^l - \mu_{\text{ref}}^l) \tag{A56.4-direction}$$

- **输入**：KS/MMD统计量，检测阈值
- **输出**：$\text{shift\_magnitude} \in \mathbb{R}_{\geq 0}$，偏移方向
- **约束**：阈值 > 0

> **白话翻译**：偏移程度量化就是"偏移有多严重"——偏移量除以阈值，得到归一化的严重度。严重度小于1说明没偏移，1到3是轻度偏移，3到10是中度偏移，超过10是严重偏移。偏移方向告诉我们误差是变大了还是变小了。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A56.1 | $D = \sup_x \|F_{\text{obs}}(x) - F_{\text{ref}}(x)\|$ | 观测样本、参考样本 | KS统计量 $\in [0,1]$ | 样本量 $\geq 10$ |
| A56.2 | $\text{MMD}^2 = \mathbb{E}[k(x,x')] - 2\mathbb{E}[k(x,y')] + \mathbb{E}[k(y,y')]$ | 观测样本、参考样本、核函数 | MMD² $\in \mathbb{R}_{\geq 0}$ | 核正定 |
| A56.3 | 滑动参考窗口在线检测 | 误差数据流、窗口长度 | 在线偏移标志 | $T_{\text{ref}} \geq T_{\text{obs}}$ |
| A56.4 | $\text{shift\_magnitude} = D / D_{\text{threshold}}$ | KS/MMD统计量、阈值 | 偏移程度 $\in \mathbb{R}_{\geq 0}$ | 阈值 > 0 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum
import numpy as np

# 引用阶段55的数据类
from stage_055 import AnomalyDetector, AnomalyReport


# ============================================================
# 阶段56 枚举：偏移检测方法
# ============================================================
class ShiftDetectionMethod(Enum):
    """分布偏移检测方法。"""
    KS = "ks"           # Kolmogorov-Smirnov检验
    MMD = "mmd"         # 最大均值差异
    COMBINED = "combined"  # 组合检测


# ============================================================
# 阶段56 枚举：参考窗口策略
# ============================================================
class ReferenceWindowStrategy(Enum):
    """参考窗口更新策略。"""
    FIXED = "fixed"          # 固定参考
    SLIDING = "sliding"      # 滑动参考
    ADAPTIVE = "adaptive"    # 自适应参考


# ============================================================
# 阶段56 数据类：偏移报告
# ============================================================
@dataclass(frozen=True)
class ShiftReport:
    """分布偏移检测报告。"""
    is_shift: bool
    method: ShiftDetectionMethod
    ks_statistic: float
    mmd_squared: float
    shift_magnitude: float
    shift_direction: float
    per_layer_magnitude: Optional[np.ndarray] = None
    per_layer_direction: Optional[np.ndarray] = None


# ============================================================
# 阶段56 数据类：分布偏移检测器
# ============================================================
@dataclass(frozen=True)
class DistributionShiftDetector:
    """分布偏移检测器的完整描述。

    组合阶段55的AnomalyDetector，在异常检测的基础上增加
    分布偏移检测能力——KS检验、MMD、在线检测、偏移量化。

    Attributes:
        anomaly_detector: 阶段55的异常检测器（组合字段）
        reference_window: 参考窗口数据
        observation_window: 观测窗口数据
        ref_window_length: 参考窗口长度 T_ref
        obs_window_length: 观测窗口长度 T_obs
        ref_strategy: 参考窗口更新策略
        ks_statistic: 当前KS统计量 D
        mmd_squared: 当前MMD²值
        ks_threshold: KS检测阈值
        mmd_threshold: MMD检测阈值
        kernel_bandwidth: MMD核带宽 σ_k
        shift_magnitude: 偏移程度
        shift_direction: 偏移方向
        last_shift_report: 最近一次偏移报告

    Invariants:
        - ks_statistic ∈ [0, 1]
        - mmd_squared >= 0
        - shift_magnitude >= 0
        - ref_window_length >= obs_window_length
    """
    anomaly_detector: AnomalyDetector
    reference_window: Optional[np.ndarray] = None
    observation_window: Optional[np.ndarray] = None
    ref_window_length: int = 200
    obs_window_length: int = 100
    ref_strategy: ReferenceWindowStrategy = ReferenceWindowStrategy.SLIDING
    ks_statistic: float = 0.0
    mmd_squared: float = 0.0
    ks_threshold: float = 0.1
    mmd_threshold: float = 0.01
    kernel_bandwidth: float = 1.0
    shift_magnitude: float = 0.0
    shift_direction: float = 0.0
    last_shift_report: Optional[ShiftReport] = None

    def __post_init__(self):
        if self.ref_window_length < self.obs_window_length:
            raise ValueError(
                f"参考窗口长度必须 >= 观测窗口长度，"
                f"收到 ref={self.ref_window_length}, obs={self.obs_window_length}"
            )
        if not (0 <= self.ks_statistic <= 1):
            raise ValueError(
                f"KS统计量必须 ∈ [0,1]，收到 {self.ks_statistic}"
            )
        if self.mmd_squared < 0:
            raise ValueError(
                f"MMD²必须 >= 0，收到 {self.mmd_squared}"
            )
        if self.shift_magnitude < 0:
            raise ValueError(
                f"偏移程度必须 >= 0，收到 {self.shift_magnitude}"
            )
        if self.kernel_bandwidth <= 0:
            raise ValueError(
                f"核带宽必须 > 0，收到 {self.kernel_bandwidth}"
            )

    @property
    def is_shift_detected(self) -> bool:
        """是否检测到分布偏移。"""
        if self.last_shift_report is None:
            return False
        return self.last_shift_report.is_shift

    @property
    def shift_severity(self) -> str:
        """偏移严重程度。"""
        if self.shift_magnitude < 1.0:
            return "none"
        elif self.shift_magnitude < 3.0:
            return "mild"
        elif self.shift_magnitude < 10.0:
            return "moderate"
        else:
            return "severe"


# ============================================================
# 阶段56 核心接口：分布偏移检测引擎
# ============================================================
class ShiftDetectionEngine:
    """阶段56的核心接口：分布偏移检测引擎。

    实现KS检验、MMD、在线分布偏移检测和偏移程度量化。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - KS统计量 ∈ [0, 1]
        - MMD² >= 0
        - 偏移程度 >= 0
    """

    def compute_ks(
        self,
        obs_samples: np.ndarray,
        ref_samples: np.ndarray
    ) -> float:
        """计算KS统计量（A56.1）。

        D = sup|F_obs(x) - F_ref(x)|

        Args:
            obs_samples: 观测样本
            ref_samples: 参考样本

        Returns:
            KS统计量 D ∈ [0, 1]

        Raises:
            ValueError: 当样本量 < 10 时
        """
        if len(obs_samples) < 10 or len(ref_samples) < 10:
            raise ValueError(
                f"样本量必须 >= 10，"
                f"收到 obs={len(obs_samples)}, ref={len(ref_samples)}"
            )
        obs = np.asarray(obs_samples).flatten()
        ref = np.asarray(ref_samples).flatten()
        all_samples = np.concatenate([obs, ref])
        all_samples = np.sort(all_samples)
        obs_sorted = np.sort(obs)
        ref_sorted = np.sort(ref)
        n_obs = len(obs_sorted)
        n_ref = len(ref_sorted)
        obs_cdf = np.searchsorted(obs_sorted, all_samples, side='right') / n_obs
        ref_cdf = np.searchsorted(ref_sorted, all_samples, side='right') / n_ref
        d = float(np.max(np.abs(obs_cdf - ref_cdf)))
        return d

    def compute_mmd(
        self,
        obs_samples: np.ndarray,
        ref_samples: np.ndarray,
        bandwidth: float
    ) -> float:
        """计算MMD²（A56.2）。

        MMD² = E[k(x,x')] - 2E[k(x,y')] + E[k(y,y')]

        Args:
            obs_samples: 观测样本
            ref_samples: 参考样本
            bandwidth: 核带宽 σ_k

        Returns:
            MMD² >= 0

        Raises:
            ValueError: 当 bandwidth <= 0 时
            ValueError: 当样本量 < 10 时
        """
        if bandwidth <= 0:
            raise ValueError(f"核带宽必须 > 0，收到 {bandwidth}")
        if len(obs_samples) < 10 or len(ref_samples) < 10:
            raise ValueError(
                f"样本量必须 >= 10，"
                f"收到 obs={len(obs_samples)}, ref={len(ref_samples)}"
            )
        x = np.asarray(obs_samples)
        y = np.asarray(ref_samples)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        def rbf_kernel(a, b, sigma):
            sq_dists = (
                np.sum(a ** 2, axis=1).reshape(-1, 1)
                + np.sum(b ** 2, axis=1).reshape(1, -1)
                - 2 * np.dot(a, b.T)
            )
            return np.exp(-sq_dists / (2 * sigma ** 2))

        n = x.shape[0]
        m = y.shape[0]
        K_xx = rbf_kernel(x, x, bandwidth)
        K_yy = rbf_kernel(y, y, bandwidth)
        K_xy = rbf_kernel(x, y, bandwidth)
        # 无偏估计
        term_xx = np.sum(K_xx - np.eye(n) * 1.0) / (n * (n - 1))
        term_yy = np.sum(K_yy - np.eye(m) * 1.0) / (m * (m - 1))
        term_xy = np.sum(K_xy) / (n * m)
        mmd2 = float(term_xx + term_yy - 2 * term_xy)
        return max(mmd2, 0.0)

    def detect_shift_online(
        self,
        detector: DistributionShiftDetector,
        new_data: np.ndarray
    ) -> DistributionShiftDetector:
        """在线分布偏移检测（A56.3）。

        更新观测窗口，计算KS/MMD，判定偏移。

        Args:
            detector: 当前检测器状态
            new_data: 新的误差数据

        Returns:
            更新后的DistributionShiftDetector
        """
        new_data = np.asarray(new_data).flatten()
        obs = detector.observation_window
        if obs is None:
            obs = new_data
        else:
            obs = np.concatenate([obs, new_data])
        if len(obs) > detector.obs_window_length:
            obs = obs[-detector.obs_window_length:]

        ref = detector.reference_window
        if ref is None or len(ref) < detector.ref_window_length:
            # 尚未积累足够参考窗口，仅更新观测窗口
            return self._replace_detector(detector, observation_window=obs)

        ks_d = self.compute_ks(obs, ref)
        mmd2 = self.compute_mmd(obs, ref, detector.kernel_bandwidth)
        shift_mag = self.compute_shift_magnitude(ks_d, detector.ks_threshold)
        shift_dir = float(np.sign(np.mean(obs) - np.mean(ref)))

        is_shift = ks_d > detector.ks_threshold or mmd2 > detector.mmd_threshold

        report = ShiftReport(
            is_shift=is_shift,
            method=ShiftDetectionMethod.COMBINED,
            ks_statistic=ks_d,
            mmd_squared=mmd2,
            shift_magnitude=shift_mag,
            shift_direction=shift_dir,
        )

        new_ref = ref
        if is_shift and detector.ref_strategy == ReferenceWindowStrategy.ADAPTIVE:
            new_ref = obs.copy()
        elif detector.ref_strategy == ReferenceWindowStrategy.SLIDING:
            combined = np.concatenate([ref, new_data])
            new_ref = combined[-detector.ref_window_length:]

        return self._replace_detector(
            detector,
            observation_window=obs,
            reference_window=new_ref,
            ks_statistic=ks_d,
            mmd_squared=mmd2,
            shift_magnitude=shift_mag,
            shift_direction=shift_dir,
            last_shift_report=report,
        )

    def _replace_detector(self, detector: DistributionShiftDetector, **kwargs):
        """辅助函数：基于dataclasses.replace更新不可变对象。"""
        from dataclasses import replace
        return replace(detector, **kwargs)

    def compute_shift_magnitude(
        self,
        statistic: float,
        threshold: float
    ) -> float:
        """计算偏移程度（A56.4）。

        shift_magnitude = D / D_threshold

        Args:
            statistic: KS或MMD统计量
            threshold: 检测阈值

        Returns:
            偏移程度 >= 0

        Raises:
            ValueError: 当 threshold <= 0 时
        """
        if threshold <= 0:
            raise ValueError(f"阈值必须 > 0，收到 {threshold}")
        return float(statistic) / float(threshold)

    def step(
        self,
        detector: DistributionShiftDetector,
        new_error: float
    ) -> DistributionShiftDetector:
        """执行一步完整的分布偏移检测更新。

        Args:
            detector: 当前检测器状态
            new_error: 新的精度加权误差

        Returns:
            更新后的DistributionShiftDetector
        """
        return self.detect_shift_online(detector, np.array([new_error]))
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `AnomalyDetector` | ← 阶段55 | 作为分布偏移检测的异常检测基础（组合字段） |
| **输入** | A55.1-A55.4（异常检测） | ← 阶段55 | 分布偏移检测的补充基础 |
| **输入** | A54.3（自适应精度更新） | ← 阶段54 | 分布偏移触发精度更新的依据 |
| **输出** | `DistributionShiftDetector` | → 阶段57 | 分布偏移检测器状态，作为驱力映射的输入 |
| **输出** | `ShiftReport` | → 阶段57 | 偏移报告，驱力映射的输入 |
| **输出** | A56.1-A56.4（分布偏移检测定义） | → 阶段59-60 | 分布偏移检测的基础定义 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| SciPy | KS检验 | `scipy.stats.ks_2samp`实现双样本KS检验 |
| JAX | MMD计算JIT编译 | `jax.vmap`实现核函数的向量化计算 |
| NumPy | ECDF计算 | 支撑经验累积分布函数的高效计算 |
| Pydantic | 数据验证 | 支撑DistributionShiftDetector的运行时类型检查 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V56.1 | KS统计量正确性 | $D = \sup|F_{\text{obs}} - F_{\text{ref}}|$ | 与SciPy结果对比 |
| V56.2 | KS统计量范围 | $D \in [0, 1]$ | 范围检查 |
| V56.3 | KS检测同分布 | 同分布时 $D < D_{\text{threshold}}$ | 合成同分布测试 |
| V56.4 | KS检测异分布 | 均值偏移时 $D > D_{\text{threshold}}$ | 合成偏移测试 |
| V56.5 | MMD²正确性 | MMD²与手动计算一致 | 数值对比 |
| V56.6 | MMD²非负 | MMD² >= 0 | 随机测试 |
| V56.7 | MMD检测同分布 | 同分布时 MMD² ≈ 0 | 合成同分布测试 |
| V56.8 | MMD检测异分布 | 分布偏移时 MMD² > 阈值 | 合成偏移测试 |
| V56.9 | 多尺度MMD | 多带宽组合检测更鲁棒 | 对比实验 |
| V56.10 | 在线检测正确性 | 滑动窗口检测偏移 | 渐进偏移测试 |
| V56.11 | 偏移程度量化 | shift_magnitude = D / D_threshold | 数值验证 |
| V56.12 | 偏移方向正确 | sign(μ_obs - μ_ref) | 方向验证 |
| V56.13 | DistributionShiftDetector组合 | `detector.anomaly_detector` 类型为 `AnomalyDetector` | 类型检查 |
| V56.14 | 非法输入拒绝 | 样本量<10、带宽≤0 抛出 `ValueError` | 异常测试 |
| V56.15 | 长期在线检测 | 10000步在线检测无退化 | 长期运行测试 |

---

> **阶段56完** | 阶段56建立了分布偏移检测机制：KS检验 $D = \sup|F_{\text{obs}}(x) - F_{\text{ref}}(x)|$ 通过比较累积分布函数的最大差距检测一维偏移（A56.1），MMD $\text{MMD}^2 = \mathbb{E}[k(x,x')] - 2\mathbb{E}[k(x,y')] + \mathbb{E}[k(y,y')]$ 通过核函数度量多维分布距离（A56.2），在线分布偏移检测使用滑动参考窗口持续监控分布变化（A56.3），偏移程度量化 $\text{shift\_magnitude} = D / D_{\text{threshold}}$ 归一化偏移严重度（A56.4），封装为DistributionShiftDetector数据类组合AnomalyDetector。下一步：阶段57将定义误差反馈到驱力的映射。
