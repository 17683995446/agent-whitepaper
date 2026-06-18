# 阶段54：误差信号的精度加权——逆协方差权重、自适应更新与注意力调制

> **阶段编号**：S54 | **所属层**：第六层·误差评估层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 精度权重 | "信任度"——逆协方差矩阵，越可信的信号权重越大 |
> | 精度加权误差 | "加权后的偏差"——按信任度加权汇总各层误差 |
> | 自适应精度更新 | "动态调整信任度"——根据观测不断修正信任度 |
> | 注意力调制 | "按重要性聚焦"——softmax将精度加权误差转化为注意力分布 |
> | PrecisionWeightedError | "精度加权误差的完整画像"——组合FeedbackSignal，在反馈信号基础上增加精度加权 |

---

## 1. 第一性原理依据

本阶段从阶段3（精度加权误差能量）和阶段53（反馈信号生成）的核心结论出发，建立误差信号的精度加权机制。核心依据为：

- **A3.2（精度加权误差能量）**（引用阶段3）：$\varepsilon_l = e^{l-1,T} \Pi_l e^{l-1}$，精度加权误差能量的原始定义。本阶段将A3.2从单层推广到多层的加权聚合——不仅单层内有精度加权，层间也有精度加权。

- **A3.8（精度更新方程）**（引用阶段3）：$\dot{\Pi}_l = \kappa_\gamma(\Pi_l^{\text{opt}} - \Pi_l)$，精度矩阵的动态更新方程。本阶段将A3.8从连续时间推广到离散时间的自适应更新——$\Pi_l(t+1) = (1-\beta)\Pi_l(t) + \beta \Pi_l^{\text{observed}}$。

- **A50.3（注意力调制协议）**（引用阶段50）：$\text{precision\_weight}_l = \text{softmax}(\beta \cdot \text{error\_magnitude}_l)$，跨层注意力权重。本阶段的注意力调制与A50.3概念一致，但引入了精度加权的调制信号——$\text{attention}_l = \text{softmax}(\gamma \cdot \Pi_l \cdot \varepsilon_l)$。

- **A53.3（反馈强度）**（引用阶段53）：$\text{feedback} = \tanh(\kappa \cdot |\Delta F| / F_{\text{baseline}})$。本阶段在反馈强度的基础上增加精度加权——不同层的反馈信号按精度加权后再路由。

本阶段的核心目标是：**定义精度权重 $\Pi_l = \Sigma_l^{-1}$（逆协方差）、精度加权误差 $\varepsilon_{\text{weighted}} = \sum_l w_l \varepsilon_l$（层间加权）、自适应精度更新 $\Pi_l(t+1) = (1-\beta)\Pi_l(t) + \beta \Pi_l^{\text{observed}}$、注意力调制 $\text{attention}_l = \text{softmax}(\gamma \cdot \Pi_l \cdot \varepsilon_l)$，封装为PrecisionWeightedError数据类，组合FeedbackSignal作为反馈信号的输入。** 精度加权是误差评估的"信任度维度"——它确保更可信的信号在误差评估中占据更大权重。

> **去重声明**：本阶段仅定义精度加权的计算规则，不涉及异常检测（阶段55）或驱力映射（阶段57）。A3.2的精度加权误差能量定义在阶段3，本阶段不重复推导，仅引用并推广到多层聚合。

---

## 2. 核心数学公式

### 第一部分：精度权重

#### 2.1 精度权重定义（A54.1）

精度权重定义为各层误差协方差矩阵的逆：

$$\Pi_l = \Sigma_l^{-1} \tag{A54.1}$$

其中：
- $\Sigma_l \in \mathbb{R}^{d_l \times d_l}$ 为第 $l$ 层误差的协方差矩阵
- $\Pi_l \in \mathbb{R}^{d_l \times d_l}$ 为第 $l$ 层的精度矩阵（对称正定）
- $d_l$ 为第 $l$ 层的维度

**精度权重的物理含义**：

精度矩阵 $\Pi_l$ 度量第 $l$ 层误差的"可信度"——协方差越小（误差越稳定），精度越高（越可信）。具体地：

| $\Sigma_l$ | $\Pi_l$ | 含义 |
|-----------|---------|------|
| 小（误差稳定） | 大（高精度） | 该层误差可信，应赋予大权重 |
| 大（误差波动） | 小（低精度） | 该层误差不可靠，应赋予小权重 |

**精度权重的标量形式**：

在误差各向同性的简化假设下，$\Sigma_l = \sigma_l^2 I$，精度矩阵退化为标量：

$$\Pi_l = \frac{1}{\sigma_l^2} I \tag{A54.1-scalar}$$

此时精度加权误差能量（引用A3.2）退化为：

$$\varepsilon_l = \frac{\|\mathbf{e}_l\|^2}{\sigma_l^2} \tag{A54.1-isotropic}$$

即误差的方差归一化形式——方差越大的层，误差在总能量中占的比重越小。

**与A3.2的对应**：

A54.1的精度权重与A3.2（引用阶段3）的精度矩阵 $\Pi_l$ 完全一致——A54.1只是明确了 $\Pi_l$ 的来源（逆协方差），而A3.2定义了 $\Pi_l$ 的使用方式（精度加权误差能量）。

- **输入**：各层误差协方差矩阵 $\{\Sigma_l\}$
- **输出**：精度矩阵 $\{\Pi_l\}$
- **约束**：$\Sigma_l$ 正定；$\Pi_l = \Sigma_l^{-1}$ 正定

> **白话翻译**：精度权重就是"信任度"——协方差矩阵的逆矩阵。误差越稳定的层，信任度越高；误差越波动的层，信任度越低。就像一个从不犯错的人说"有问题"，比一个经常犯错的人说同样的话，更值得重视。

---

### 第二部分：精度加权误差

#### 2.2 精度加权误差定义（A54.2）

精度加权误差是各层精度加权误差能量的加权和：

$$\varepsilon_{\text{weighted}} = \sum_{l} w_l \cdot \varepsilon_l = \sum_{l} w_l \cdot \mathbf{e}_l^T \Pi_l \mathbf{e}_l \tag{A54.2}$$

其中：
- $\varepsilon_l = \mathbf{e}_l^T \Pi_l \mathbf{e}_l$ 为第 $l$ 层的精度加权误差能量（引用A3.2）
- $w_l \geq 0$ 为第 $l$ 层的聚合权重
- $\sum_l w_l = 1$

**与A51.5全局误差的关系**：

A51.5（引用阶段51）的全局误差 $\varepsilon_{\text{global}} = \sum_l w_l \|\mathbf{o}_l - \hat{\mathbf{o}}_l\|^2$ 是未加权（$\Pi_l = I$）的特例。A54.2的精度加权误差是A51.5的推广——当 $\Pi_l = I$ 时，A54.2退化为A51.5。

**精度加权误差的分层贡献**：

各层对精度加权误差的贡献比为：

$$\text{contribution}_l = \frac{w_l \cdot \varepsilon_l}{\varepsilon_{\text{weighted}}} \tag{A54.2-contribution}$$

贡献比揭示了哪一层的精度加权误差最显著——贡献比最高的层是系统最需要关注的层。

**精度加权误差与自由能的关系**：

由A4.8（引用阶段4），自由能的准确性项为 $\sum_l \varepsilon_l / 2$（在均匀权重下）。A54.2的精度加权误差是自由能准确性项的加权近似——权重 $w_l$ 反映了不同层在自由能中的相对重要性。

- **输入**：各层误差信号 $\{\mathbf{e}_l\}$，精度矩阵 $\{\Pi_l\}$，聚合权重 $\{w_l\}$
- **输出**：$\varepsilon_{\text{weighted}} \in \mathbb{R}_{\geq 0}$
- **约束**：权重归一化；各 $\varepsilon_l \geq 0$

> **白话翻译**：精度加权误差就是"加权后的偏差"——各层的精度加权误差能量再按权重汇总。和全局误差的区别是：全局误差用欧氏距离（信任度相同），精度加权误差用精度加权距离（信任度不同）。精度高的层，其误差在总加权误差中占的比重更大——"可信的人说有问题，比不可信的人说同样的话，更值得重视"。

---

### 第三部分：自适应精度更新

#### 2.3 自适应精度更新定义（A54.3）

精度矩阵根据观测数据自适应更新：

$$\Pi_l(t+1) = (1-\beta) \Pi_l(t) + \beta \cdot \Pi_l^{\text{observed}}(t) \tag{A54.3}$$

其中：
- $\beta \in (0, 1)$ 为更新速率（$\beta$ 越大，更新越快）
- $\Pi_l^{\text{observed}}(t) = \hat{\Sigma}_l^{-1}(t)$ 为观测精度（基于近期误差的样本协方差逆）
- $\hat{\Sigma}_l(t) = \frac{1}{|W|} \sum_{s \in W} \mathbf{e}_l(s) \mathbf{e}_l(s)^T$ 为窗口 $W$ 内的样本协方差

**与A3.8的对应**：

A54.3是A3.8（引用阶段3）$\dot{\Pi}_l = \kappa_\gamma(\Pi_l^{\text{opt}} - \Pi_l)$ 的离散时间版本。对应关系：
- $\beta$ 对应 $\kappa_\gamma \cdot \Delta t$（更新速率 × 时间步长）
- $\Pi_l^{\text{observed}}$ 对应 $\Pi_l^{\text{opt}}$（最优精度）

**更新速率 $\beta$ 的选择**：

| $\beta$ | 更新速度 | 适用场景 |
|---------|---------|---------|
| 0.01 | 慢更新 | 环境稳定，精度变化缓慢 |
| 0.1 | 中速更新 | 一般场景 |
| 0.5 | 快更新 | 环境变化快，精度需要快速适应 |

**正定性保持**：

A54.3的更新保持精度矩阵的正定性——两个正定矩阵的凸组合仍为正定矩阵：

$$\Pi_l(t+1) = (1-\beta) \Pi_l(t) + \beta \Pi_l^{\text{observed}}(t) \in \mathbb{S}_{++}^{d_l} \tag{A54.3-positive}$$

**观测精度的计算**：

观测精度 $\Pi_l^{\text{observed}}(t)$ 基于滑动窗口内的样本协方差（引用A52.1的滑动窗口）：

$$\hat{\Sigma}_l(t) = \frac{1}{T+1} \sum_{s=t-T}^{t} \mathbf{e}_l(s) \mathbf{e}_l(s)^T \tag{A54.3-sample-cov}$$

$$\Pi_l^{\text{observed}}(t) = \hat{\Sigma}_l^{-1}(t) \tag{A54.3-observed}$$

当样本协方差矩阵奇异（窗口内误差维度 > 样本数）时，使用正则化逆：

$$\Pi_l^{\text{observed}}(t) = (\hat{\Sigma}_l(t) + \lambda I)^{-1} \tag{A54.3-regularized}$$

其中 $\lambda > 0$ 为正则化参数。

- **输入**：当前精度 $\Pi_l(t)$，误差历史 $\{\mathbf{e}_l(s)\}$，更新速率 $\beta$
- **输出**：更新后精度 $\Pi_l(t+1)$
- **约束**：$\beta \in (0, 1)$；$\Pi_l(t+1)$ 正定

> **白话翻译**：自适应精度更新就是"动态调整信任度"——根据最近的观测数据不断修正各层的信任度。$\beta$ 是"修正速度"——$\beta$ 大则快速修正（适合环境变化快的场景），$\beta$ 小则慢速修正（适合环境稳定的场景）。关键是信任度更新后仍然保持正定——"信任度不会变成负数"。

---

### 第四部分：注意力调制

#### 2.4 注意力调制定义（A54.4）

注意力调制将精度加权误差转化为注意力分布：

$$\text{attention}_l = \text{softmax}(\gamma \cdot \Pi_l \cdot \varepsilon_l) = \frac{\exp(\gamma \cdot \Pi_l \cdot \varepsilon_l)}{\sum_{l'} \exp(\gamma \cdot \Pi_{l'} \cdot \varepsilon_{l'})} \tag{A54.4}$$

其中：
- $\gamma > 0$ 为注意力温度（$\gamma$ 越大，注意力越集中在精度加权误差最大的层）
- $\Pi_l \cdot \varepsilon_l$ 为第 $l$ 层的精度加权误差能量（标量，即 $\mathbf{e}_l^T \Pi_l \mathbf{e}_l$）

**与A50.3的关系**：

A54.4的注意力调制与A50.3（引用阶段50）$\text{precision\_weight}_l = \text{softmax}(\beta \cdot \text{error\_magnitude}_l)$ 在形式上类似，但调制信号不同：
- A50.3：调制信号 = 误差能量（未加权）
- A54.4：调制信号 = 精度 × 误差能量（精度加权）

A54.4是A50.3的精度加权推广——精度高的层，其误差在注意力分配中权重更大。

**注意力调制的物理含义**：

| $\gamma$ | 注意力分布 | 含义 |
|----------|----------|------|
| $\gamma \to 0$ | 均匀分布 $[1/3, 1/3, 1/3]$ | 不区分各层 |
| $\gamma = 1$ | 按精度加权误差比例分配 | 精度加权误差大的层获得更多注意力 |
| $\gamma \to \infty$ | 全部集中在最大精度加权误差层 | 只关注最需要处理的层 |

**注意力调制的应用**：

注意力权重可以用于：
1. **计算资源分配**：注意力高的层获得更多计算资源（更频繁的更新、更精细的模型）
2. **反馈信号加权**：注意力高的层的反馈信号在路由时权重更大
3. **异常检测聚焦**：注意力高的层是异常检测的重点监控对象

- **输入**：各层精度加权误差 $\{\Pi_l \cdot \varepsilon_l\}$，注意力温度 $\gamma$
- **输出**：注意力权重 $\{\text{attention}_l\}$
- **约束**：$\sum_l \text{attention}_l = 1$；$\gamma > 0$

> **白话翻译**：注意力调制就是"按重要性聚焦"——用softmax把精度加权误差转化为注意力分布。精度高且误差大的层，获得最多的注意力。温度 $\gamma$ 控制"偏心程度"——$\gamma$ 小就一视同仁，$\gamma$ 大就只关注最出问题的层。这和A50.3的注意力调制类似，但多了精度加权——"可信且出问题的层，最值得重视"。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A54.1 | $\Pi_l = \Sigma_l^{-1}$ | 协方差矩阵 $\Sigma_l$ | 精度矩阵 $\Pi_l$ | $\Sigma_l$ 正定 |
| A54.2 | $\varepsilon_{\text{weighted}} = \sum_l w_l \cdot \mathbf{e}_l^T \Pi_l \mathbf{e}_l$ | 误差信号、精度矩阵、权重 | 加权误差 $\in \mathbb{R}_{\geq 0}$ | 权重归一化 |
| A54.3 | $\Pi_l(t+1) = (1-\beta)\Pi_l(t) + \beta \Pi_l^{\text{observed}}(t)$ | 当前精度、观测精度、$\beta$ | 更新后精度 | $\beta \in (0,1)$，正定保持 |
| A54.4 | $\text{attention}_l = \text{softmax}(\gamma \cdot \Pi_l \cdot \varepsilon_l)$ | 精度加权误差、温度 $\gamma$ | 注意力权重 | 归一化，$\gamma > 0$ |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict
from enum import Enum
import numpy as np

# 引用阶段53的数据类
from stage_053 import FeedbackSignal, FeedbackType


# ============================================================
# 阶段54 数据类：精度加权误差
# ============================================================
@dataclass(frozen=True)
class PrecisionWeightedError:
    """精度加权误差的完整描述。

    组合阶段53的FeedbackSignal，在反馈信号的基础上增加
    精度加权机制——精度权重、加权误差、自适应更新、注意力调制。

    Attributes:
        feedback_signal: 阶段53的反馈信号（组合字段）
        precision_matrices: 各层精度矩阵 {low: Π_low, mid: Π_mid, high: Π_high}
        weighted_error: 精度加权误差 ε_weighted（A54.2）
        layer_weights: 层间聚合权重 [w_low, w_mid, w_high]
        layer_contributions: 各层贡献比（A54.2-contribution）
        update_beta: 自适应精度更新速率 β
        attention_weights: 注意力权重 [attention_low, attention_mid, attention_high]
        attention_gamma: 注意力温度 γ
        observed_precisions: 观测精度矩阵

    Invariants:
        - precision_matrices 各矩阵正定
        - weighted_error >= 0
        - layer_weights 归一化
        - attention_weights 归一化
        - update_beta ∈ (0, 1)
        - attention_gamma > 0
    """
    feedback_signal: FeedbackSignal
    precision_matrices: Dict[str, np.ndarray]
    weighted_error: float = 0.0
    layer_weights: np.ndarray = None
    layer_contributions: Optional[np.ndarray] = None
    update_beta: float = 0.1
    attention_weights: Optional[np.ndarray] = None
    attention_gamma: float = 1.0
    observed_precisions: Optional[Dict[str, np.ndarray]] = None

    def __post_init__(self):
        if self.weighted_error < 0:
            raise ValueError(
                f"精度加权误差必须 >= 0，收到 {self.weighted_error}"
            )
        if self.layer_weights is not None:
            if not np.allclose(np.sum(self.layer_weights), 1.0):
                raise ValueError("层间权重必须归一化")
            if np.any(self.layer_weights < 0):
                raise ValueError("层间权重必须非负")
        if not (0 < self.update_beta < 1):
            raise ValueError(
                f"更新速率必须 ∈ (0,1)，收到 {self.update_beta}"
            )
        if self.attention_gamma <= 0:
            raise ValueError(
                f"注意力温度必须 > 0，收到 {self.attention_gamma}"
            )
        if self.attention_weights is not None:
            if not np.allclose(np.sum(self.attention_weights), 1.0):
                raise ValueError("注意力权重必须归一化")
        # 验证精度矩阵正定性
        for name, mat in self.precision_matrices.items():
            try:
                np.linalg.cholesky(mat)
            except np.linalg.LinAlgError:
                raise ValueError(f"精度矩阵 {name} 必须正定")

    @property
    def dominant_layer(self) -> str:
        """精度加权误差贡献最大的层。"""
        if self.layer_contributions is None:
            return "unknown"
        layers = ["low", "mid", "high"]
        return layers[int(np.argmax(self.layer_contributions))]

    @property
    def attention_focus(self) -> str:
        """注意力最集中的层。"""
        if self.attention_weights is None:
            return "unknown"
        layers = ["low", "mid", "high"]
        return layers[int(np.argmax(self.attention_weights))]


# ============================================================
# 阶段54 核心接口：精度加权器
# ============================================================
class PrecisionWeighter:
    """阶段54的核心接口：精度加权器。

    实现精度权重的计算、精度加权误差的聚合、
    自适应精度更新和注意力调制。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 精度矩阵正定
        - 精度加权误差非负
        - 注意力权重归一化
    """

    def compute_precision(
        self,
        covariance: np.ndarray,
        regularization: float = 1e-6
    ) -> np.ndarray:
        """计算精度权重（A54.1）。

        Π_l = Σ_l^{-1}

        Args:
            covariance: 协方差矩阵 Σ_l
            regularization: 正则化参数 λ

        Returns:
            精度矩阵 Π_l

        Raises:
            ValueError: 当协方差矩阵不正定时
        """
        # 验证协方差矩阵正定性
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            raise ValueError("协方差矩阵必须正定")
        # 正则化后求逆
        reg_cov = covariance + regularization * np.eye(covariance.shape[0])
        precision = np.linalg.inv(reg_cov)
        return precision

    def compute_weighted_error(
        self,
        errors: Dict[str, np.ndarray],
        precisions: Dict[str, np.ndarray],
        weights: np.ndarray
    ) -> float:
        """计算精度加权误差（A54.2）。

        ε_weighted = Σ_l w_l · e_l^T Π_l e_l

        Args:
            errors: 各层误差信号 {low: e_low, mid: e_mid, high: e_high}
            precisions: 各层精度矩阵
            weights: 层间聚合权重

        Returns:
            精度加权误差 >= 0

        Raises:
            ValueError: 当权重不归一化时
            ValueError: 当误差和精度矩阵维度不匹配时
        """
        if not np.allclose(np.sum(weights), 1.0):
            raise ValueError("层间权重必须归一化")
        if np.any(weights < 0):
            raise ValueError("层间权重必须非负")

        weighted = 0.0
        for i, (name, e) in enumerate(errors.items()):
            Pi = precisions[name]
            if e.shape[0] != Pi.shape[0]:
                raise ValueError(
                    f"误差 {name} 维度 {e.shape} 与精度矩阵 {Pi.shape} 不匹配"
                )
            energy = float(e.T @ Pi @ e)
            weighted += weights[i] * energy
        return weighted

    def update_precision(
        self,
        current_precision: np.ndarray,
        error_history: np.ndarray,
        beta: float,
        regularization: float = 1e-6
    ) -> np.ndarray:
        """自适应精度更新（A54.3）。

        Π(t+1) = (1-β)Π(t) + β·Π_observed(t)

        Args:
            current_precision: 当前精度矩阵 Π(t)
            error_history: 误差历史 [e(t-T), ..., e(t)]
            beta: 更新速率 ∈ (0,1)
            regularization: 正则化参数

        Returns:
            更新后的精度矩阵 Π(t+1)

        Raises:
            ValueError: 当 beta ∉ (0,1) 时
        """
        if not (0 < beta < 1):
            raise ValueError(f"beta 必须 ∈ (0,1)，收到 {beta}")
        # 样本协方差
        n = error_history.shape[0]
        mean_e = np.mean(error_history, axis=0)
        centered = error_history - mean_e
        sample_cov = (centered.T @ centered) / n
        # 正则化逆
        reg_cov = sample_cov + regularization * np.eye(sample_cov.shape[0])
        observed_precision = np.linalg.inv(reg_cov)
        # 凸组合更新
        updated = (1.0 - beta) * current_precision + beta * observed_precision
        return updated

    def compute_attention(
        self,
        precision_weighted_errors: np.ndarray,
        gamma: float
    ) -> np.ndarray:
        """计算注意力权重（A54.4）。

        attention_l = softmax(γ · Π_l · ε_l)

        Args:
            precision_weighted_errors: 各层精度加权误差 [Π_low·ε_low, ...]
            gamma: 注意力温度

        Returns:
            注意力权重 [attention_low, attention_mid, attention_high]

        Raises:
            ValueError: 当 gamma <= 0 时
        """
        if gamma <= 0:
            raise ValueError(f"gamma 必须 > 0，收到 {gamma}")
        scaled = gamma * precision_weighted_errors
        exp_vals = np.exp(scaled - np.max(scaled))
        attention = exp_vals / np.sum(exp_vals)
        return attention

    def evaluate(
        self,
        feedback: FeedbackSignal,
        errors: Dict[str, np.ndarray],
        current_precisions: Dict[str, np.ndarray],
        error_history: Dict[str, np.ndarray],
        weights: np.ndarray = None,
        beta: float = 0.1,
        gamma: float = 1.0
    ) -> PrecisionWeightedError:
        """完整评估精度加权误差（A54.1-A54.4）。

        Args:
            feedback: 反馈信号
            errors: 各层当前误差
            current_precisions: 当前精度矩阵
            error_history: 各层误差历史
            weights: 层间权重
            beta: 更新速率
            gamma: 注意力温度

        Returns:
            PrecisionWeightedError 实例
        """
        if weights is None:
            weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

        # 计算精度加权误差
        weighted_error = self.compute_weighted_error(
            errors, current_precisions, weights
        )

        # 更新精度矩阵
        updated_precisions = {}
        observed_precisions = {}
        for name, Pi in current_precisions.items():
            hist = error_history.get(name)
            if hist is not None and len(hist) > 0:
                updated = self.update_precision(Pi, hist, beta)
                updated_precisions[name] = updated
                # 观测精度（用于记录）
                n = hist.shape[0]
                mean_e = np.mean(hist, axis=0)
                centered = hist - mean_e
                sample_cov = (centered.T @ centered) / n
                reg_cov = sample_cov + 1e-6 * np.eye(sample_cov.shape[0])
                observed_precisions[name] = np.linalg.inv(reg_cov)
            else:
                updated_precisions[name] = Pi
                observed_precisions[name] = Pi

        # 计算各层精度加权误差能量（用于注意力调制）
        pwe_vals = []
        for name in ["low", "mid", "high"]:
            e = errors.get(name)
            Pi = current_precisions.get(name)
            if e is not None and Pi is not None:
                val = float(e.T @ Pi @ e)
            else:
                val = 0.0
            pwe_vals.append(val)
        pwe_arr = np.array(pwe_vals)

        # 注意力调制
        attention = self.compute_attention(pwe_arr, gamma)

        # 各层贡献比
        total = np.sum(pwe_arr)
        if total > 0:
            contributions = pwe_arr / total
        else:
            contributions = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

        return PrecisionWeightedError(
            feedback_signal=feedback,
            precision_matrices=updated_precisions,
            weighted_error=weighted_error,
            layer_weights=weights,
            layer_contributions=contributions,
            update_beta=beta,
            attention_weights=attention,
            attention_gamma=gamma,
            observed_precisions=observed_precisions,
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `FeedbackSignal` | ← 阶段53 | 作为精度加权的反馈信号输入（组合字段） |
| **输入** | A3.2（精度加权误差能量） | ← 阶段3 | 精度加权的理论基础 |
| **输入** | A3.8（精度更新方程） | ← 阶段3 | 自适应精度更新的理论基础 |
| **输入** | A50.3（注意力调制协议） | ← 阶段50 | 注意力调制的跨层基础 |
| **输出** | `PrecisionWeightedError` | → 阶段55 | 精度加权误差，作为异常检测的输入 |
| **输出** | A54.1-A54.4（精度加权定义） | → 阶段55-60 | 精度加权的基础定义 |
| **输出** | `PrecisionWeighter` | → 阶段59 | 精度加权器的接口 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 精度矩阵计算JIT编译 | `jax.numpy.linalg.inv`和`cholesky`的JIT编译 |
| SciPy | 协方差估计与正则化逆 | `scipy.linalg.pinvh`实现正则化伪逆 |
| NumPy | softmax向量化 | 支撑注意力调制的向量化计算 |
| Pydantic | 数据验证 | 支撑PrecisionWeightedError的运行时类型检查 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V54.1 | 精度权重正确性 | $\Pi_l = \Sigma_l^{-1}$ | 矩阵乘法验证 |
| V54.2 | 精度矩阵正定性 | Cholesky分解成功 | 正定性检查 |
| V54.3 | 精度加权误差正确性 | $\varepsilon_{\text{weighted}} = \sum_l w_l \mathbf{e}_l^T \Pi_l \mathbf{e}_l$ | 数值对比 |
| V54.4 | 精度加权误差非负 | $\varepsilon_{\text{weighted}} \geq 0$ | 随机测试 |
| V54.5 | 退化为全局误差 | $\Pi_l = I$ 时 $\varepsilon_{\text{weighted}} = \varepsilon_{\text{global}}$ | 特例验证 |
| V54.6 | 自适应更新正确性 | $\Pi(t+1) = (1-\beta)\Pi(t) + \beta \Pi^{\text{obs}}$ | 数值对比 |
| V54.7 | 更新后正定性 | 更新后 $\Pi_l$ 仍正定 | Cholesky验证 |
| V54.8 | 观测精度正确性 | $\Pi^{\text{obs}} = \hat{\Sigma}^{-1}$ | 样本协方差验证 |
| V54.9 | 正则化逆有效 | 奇异矩阵时正则化逆仍正定 | 边界测试 |
| V54.10 | 注意力权重归一化 | $\sum_l \text{attention}_l = 1$ | 求和验证 |
| V54.11 | 注意力与精度加权误差正相关 | 精度加权误差大的层注意力更大 | 合成数据测试 |
| V54.12 | 注意力温度效果 | $\gamma \to \infty$ 时注意力集中在最大层 | 极限测试 |
| V54.13 | PrecisionWeightedError组合 | `pwe.feedback_signal` 类型为 `FeedbackSignal` | 类型检查 |
| V54.14 | 非法输入拒绝 | $\beta \notin (0,1)$、$\gamma \leq 0$ 抛出 `ValueError` | 异常测试 |
| V54.15 | 长期精度更新稳定性 | 10000步自适应更新无退化 | 长期运行测试 |

---

> **阶段54完** | 阶段54建立了误差信号的精度加权机制：精度权重 $\Pi_l = \Sigma_l^{-1}$ 定义了各层误差的可信度（A54.1），精度加权误差 $\varepsilon_{\text{weighted}} = \sum_l w_l \mathbf{e}_l^T \Pi_l \mathbf{e}_l$ 按信任度加权汇总各层误差（A54.2），自适应精度更新 $\Pi_l(t+1) = (1-\beta)\Pi_l(t) + \beta \Pi_l^{\text{observed}}$ 根据观测动态修正信任度（A54.3），注意力调制 $\text{attention}_l = \text{softmax}(\gamma \cdot \Pi_l \cdot \varepsilon_l)$ 将精度加权误差转化为注意力分布（A54.4），封装为PrecisionWeightedError数据类组合FeedbackSignal。下一步：阶段55将定义异常误差检测机制。
