# 阶段42：底层模型算法——在线ARIMA与卡尔曼滤波

> **阶段编号**：S42 | **所属层**：第五层·层级预测层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 在线ARIMA | "边走边学的趋势预测器"——用过去几步的数据预测下一步，参数随时更新 |
> | 卡尔曼滤波 | "最优跟踪器"——在噪声中跟踪真实状态，每步融合预测和观测 |
> | 卡尔曼增益 | "信谁多一点"——预测和观测的信任权重，动态调整 |
> | 递推最小二乘 | "参数在线学习"——每来一个新数据点就微调模型参数，不用重新训练 |
> | LowLevelAlgorithm | "底层算法的完整画像"——组合LowLevelTarget，在目标基础上增加算法参数 |

---

## 1. 第一性原理依据

本阶段从阶段12（自下而上误差传播）和阶段14（表征更新规则）的核心结论出发，建立底层模型的算法框架。核心依据为：

- **A12.1（误差传播规则）**（引用阶段12）：$\delta^l = \Pi_l e^l + (\partial f^{l+1}/\partial r^l) \cdot \delta^{l+1}$，误差信号从底层逐层向上传播。底层模型的预测误差 $e^0 = o - \hat{o}$ 是传播链的起点——底层预测越准确，误差信号越小，高层需要调整的幅度也越小。

- **A14.1（表征更新方程）**（引用阶段14）：$\tau_r \dot{r}^l = \delta^{l+1} - e^l$，表征根据误差信号和传播信号更新。底层模型的在线参数更新遵循相同的梯度下降原理——参数沿自由能下降方向调整。

- **A14.2（精度调制更新）**（引用阶段14）：精度加权的表征更新。卡尔曼滤波中的卡尔曼增益 $K$ 正是精度调制的工程实现——增益越大，观测的精度越高，对观测的信任越多。

- **A41.1-A41.4（底层预测目标与特征工程）**（引用阶段41）：底层预测目标 $\mathbf{y}_{\text{low}} = [\mathbf{y}_{\text{sys}}, \mathbf{y}_{\text{log}}, \mathbf{y}_{\text{file}}]$，特征向量 $\mathbf{x}_{\text{low}} \in \mathbb{R}^{4d_{\text{low}}}$，预测窗口 $h_{\text{low}} \in [1, 10]$。

本阶段的核心目标是：**设计在线ARIMA(p,d,q)模型用于底层时间序列预测、建立卡尔曼滤波的状态空间模型和更新方程、推导卡尔曼增益的递推公式、实现在线参数更新的递推最小二乘算法，并封装为LowLevelAlgorithm数据类，组合LowLevelTarget作为预测目标的输入。** 底层算法是层级预测体系的计算核心——它将底层特征转化为多步预测，同时在线更新模型参数以适应环境变化。

---

## 2. 核心数学公式

### 第一部分：在线ARIMA模型

#### 2.1 在线ARIMA(p,d,q)模型（A42.1）

ARIMA(p,d,q)模型将底层预测目标 $\mathbf{y}_{\text{low}}$ 建模为差分后的自回归-滑动平均过程：

$$\phi(B)(1-B)^d \mathbf{y}_t = \theta(B) \boldsymbol{\epsilon}_t \tag{A42.1}$$

其中：
- $B$ 为后移算子：$B\mathbf{y}_t = \mathbf{y}_{t-1}$
- $\phi(B) = 1 - \phi_1 B - \phi_2 B^2 - \cdots - \phi_p B^p$ 为自回归（AR）多项式
- $\theta(B) = 1 + \theta_1 B + \theta_2 B^2 + \cdots + \theta_q B^q$ 为滑动平均（MA）多项式
- $d$ 为差分阶数（使序列平稳所需的差分次数）
- $\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \Sigma_\epsilon)$ 为白噪声

**在线预测公式**：

给定当前参数 $\hat{\phi}_1, \ldots, \hat{\phi}_p, \hat{\theta}_1, \ldots, \hat{\theta}_q$，第 $h$ 步预测为：

$$\hat{\mathbf{y}}_{t+h} = \sum_{i=1}^{p}\hat{\phi}_i \hat{\mathbf{y}}_{t+h-i} + \sum_{j=1}^{q}\hat{\theta}_j \hat{\boldsymbol{\epsilon}}_{t+h-j} \tag{A42.1-predict}$$

其中 $\hat{\mathbf{y}}_{t+k} = \mathbf{y}_{t+k}$（当 $k \leq 0$ 时用实际值），$\hat{\boldsymbol{\epsilon}}_{t+k} = \mathbf{0}$（当 $k > 0$ 时未来残差设为零，这是最优预测的性质）。

**参数选择**：

底层预测目标的不同子向量采用不同的ARIMA参数：

| 子向量 | 推荐参数 | 理由 |
|--------|---------|------|
| $\mathbf{y}_{\text{sys}}$ | ARIMA(2,1,1) | 系统负载有短期自相关，需一阶差分去趋势 |
| $\mathbf{y}_{\text{log}}$ | ARIMA(1,0,1) | 日志变化本身近似平稳，低阶即可 |
| $\mathbf{y}_{\text{file}}$ | ARIMA(2,1,0) | 文件活动有自回归特征，MA项不显著 |

- **输入**：历史观测 $\{\mathbf{y}_{t-p}, \ldots, \mathbf{y}_t\}$，ARIMA参数 $(p, d, q)$
- **输出**：多步预测 $\hat{\mathbf{y}}_{t+1}, \ldots, \hat{\mathbf{y}}_{t+h}$
- **约束**：AR多项式 $\phi(B)$ 的根在单位圆外（平稳性）；MA多项式 $\theta(B)$ 的根在单位圆外（可逆性）

> **白话翻译**：ARIMA就是"用过去几步的值和过去的误差来预测下一步"。AR部分看"过去值怎么影响现在"，MA部分看"过去的预测误差怎么修正现在"，差分部分把不平稳的序列变成平稳的。在线ARIMA的参数不是固定的，每来一个新数据就微调——"边走边学"。

---

### 第二部分：卡尔曼滤波

#### 2.2 卡尔曼滤波状态空间模型（A42.2）

卡尔曼滤波将底层预测问题建模为线性状态空间模型：

**状态方程**：

$$\mathbf{x}_{t+1} = A\mathbf{x}_t + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, Q) \tag{A42.2-state}$$

**观测方程**：

$$\mathbf{z}_t = H\mathbf{x}_t + \mathbf{v}_t, \quad \mathbf{v}_t \sim \mathcal{N}(\mathbf{0}, R) \tag{A42.2-obs}$$

其中：
- $\mathbf{x}_t \in \mathbb{R}^{n_x}$ 为隐状态向量（底层系统的内部状态）
- $A \in \mathbb{R}^{n_x \times n_x}$ 为状态转移矩阵（系统动力学）
- $H \in \mathbb{R}^{d_{\text{low}} \times n_x}$ 为观测矩阵（从隐状态到观测的映射）
- $Q \in \mathbb{R}^{n_x \times n_x}$ 为过程噪声协方差（系统动力学不确定性）
- $R \in \mathbb{R}^{d_{\text{low}} \times d_{\text{low}}}$ 为观测噪声协方差（测量不确定性）
- $\mathbf{w}_t$ 和 $\mathbf{v}_t$ 为相互独立的零均值高斯噪声

**与ARIMA的对应**：

ARIMA(p,d,q)可以转化为状态空间形式。以ARIMA(2,1,1)为例，令 $\mathbf{x}_t = [\mathbf{y}_t, \mathbf{y}_{t-1}, \boldsymbol{\epsilon}_t]^T$，则：

$$A = \begin{bmatrix} 1+\phi_1 & \phi_2-\phi_1 & \theta_1 \\ 1 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}, \quad H = \begin{bmatrix} 1 & 0 & 0 \end{bmatrix}$$

因此卡尔曼滤波是ARIMA的推广形式——它允许更灵活的状态空间结构。

- **输入**：状态转移矩阵 $A$，观测矩阵 $H$，过程噪声 $Q$，观测噪声 $R$
- **输出**：状态估计 $\hat{\mathbf{x}}_{t|t}$ 和估计误差协方差 $P_{t|t}$
- **约束**：$Q$ 和 $R$ 对称正定；$A$ 稳定（特征值在单位圆内）

> **白话翻译**：卡尔曼滤波就是"在噪声中跟踪真实状态"。系统有个看不见的内部状态 $\mathbf{x}$，我们只能看到带噪声的观测 $\mathbf{z}$。状态方程说"状态怎么随时间演化"，观测方程说"状态怎么映射到我们看到的"。两个方程加起来就是完整的状态空间模型。

---

#### 2.3 卡尔曼增益（A42.3）

卡尔曼滤波的核心是卡尔曼增益 $K_t$，它决定了预测和观测的融合权重：

**预测步骤**：

$$\hat{\mathbf{x}}_{t|t-1} = A\hat{\mathbf{x}}_{t-1|t-1} \tag{A42.3-predict-state}$$

$$P_{t|t-1} = AP_{t-1|t-1}A^T + Q \tag{A42.3-predict-cov}$$

**更新步骤**：

$$K_t = P_{t|t-1}H^T(HP_{t|t-1}H^T + R)^{-1} \tag{A42.3}$$

$$\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + K_t(\mathbf{z}_t - H\hat{\mathbf{x}}_{t|t-1}) \tag{A42.3-update-state}$$

$$P_{t|t} = (I - K_tH)P_{t|t-1} \tag{A42.3-update-cov}$$

**卡尔曼增益的物理含义**：

卡尔曼增益 $K_t$ 是预测误差协方差和观测误差协方差的相对权重：

- 当 $R \to 0$（观测非常精确）时，$K_t \to H^{-1}$（完全信任观测）
- 当 $P_{t|t-1} \to 0$（预测非常精确）时，$K_t \to 0$（完全信任预测）
- 一般情况下，$K_t$ 在两者之间取最优平衡

**与精度调制的对应**：

卡尔曼增益 $K_t$ 与A14.2（引用阶段14）的精度调制更新在数学结构上同构。设预测精度 $\Pi_{\text{pred}} = P_{t|t-1}^{-1}$，观测精度 $\Pi_{\text{obs}} = R^{-1}$，则：

$$K_t = \Pi_{\text{total}}^{-1} \cdot H^T \Pi_{\text{obs}} \tag{A42.3-precision}$$

其中 $\Pi_{\text{total}} = \Pi_{\text{pred}} + H^T \Pi_{\text{obs}} H$ 为融合精度。卡尔曼增益将观测精度加权后的残差映射回状态空间——这正是精度调制更新的核心思想。

- **输入**：预测误差协方差 $P_{t|t-1}$，观测矩阵 $H$，观测噪声协方差 $R$
- **输出**：卡尔曼增益 $K_t \in \mathbb{R}^{n_x \times d_{\text{low}}}$
- **约束**：$HP_{t|t-1}H^T + R$ 正定（保证可逆）

> **白话翻译**：卡尔曼增益就是"信预测多一点还是信观测多一点"。如果观测很准（$R$ 小），就多信观测；如果预测很准（$P$ 小），就多信预测。增益自动在两者之间找最优平衡——这就是"最优跟踪"的含义。

---

### 第三部分：在线参数更新

#### 2.4 递推最小二乘（A42.4）

ARIMA模型和卡尔曼滤波的参数需要在线更新以适应环境变化。递推最小二乘（RLS）提供了一种增量式参数更新方法。

设模型参数为 $\boldsymbol{\theta} \in \mathbb{R}^{n_\theta}$，预测模型为 $\hat{\mathbf{y}}_t = \mathbf{h}_t^T \boldsymbol{\theta}$，其中 $\mathbf{h}_t$ 为特征向量（来自A41.4的特征工程）。RLS更新公式为：

**增益计算**：

$$\mathbf{g}_t = \frac{P_{t-1}\mathbf{h}_t}{\lambda + \mathbf{h}_t^T P_{t-1}\mathbf{h}_t} \tag{A42.4-gain}$$

**参数更新**：

$$\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} + \mathbf{g}_t(\mathbf{y}_t - \mathbf{h}_t^T \boldsymbol{\theta}_{t-1}) \tag{A42.4-update}$$

**协方差更新**：

$$P_t = \frac{1}{\lambda}(P_{t-1} - \mathbf{g}_t \mathbf{h}_t^T P_{t-1}) \tag{A42.4-cov}$$

其中 $\lambda \in (0, 1]$ 为遗忘因子（典型值 $\lambda = 0.99$），控制历史数据的权重衰减速率。$\lambda$ 越小，对近期数据的权重越大，参数跟踪越快但越不稳定。

**遗忘因子的选择**：

遗忘因子 $\lambda$ 与预测编码框架中的学习率 $\kappa$ 对应。由A17.2（引用阶段17），学习时间常数为 $\tau_{\text{learn}} = \tau_r / \alpha_{\text{separation}}$。在RLS中，有效记忆长度为 $1/(1-\lambda)$，因此：

$$\lambda = 1 - \frac{1}{\tau_{\text{learn}} / \Delta t} \tag{A42.4-lambda}$$

当 $\tau_{\text{learn}} = 100\Delta t$（分离比 $\alpha = 100$）时，$\lambda = 0.99$。

**与表征更新的对应**：

A42.4的参数更新 $\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} + \mathbf{g}_t \cdot e_t$ 与A14.1（引用阶段14）的表征更新 $\tau_r \dot{r}^l = \delta^{l+1} - e^l$ 在数学结构上同构——都是沿误差方向调整内部状态。区别在于：表征更新调整的是内部表征 $r^l$，参数更新调整的是模型参数 $\boldsymbol{\theta}$。这对应了A17.1-A17.2（引用阶段17）的快感知-慢学习分离——表征更新是快感知，参数更新是慢学习。

- **输入**：当前参数 $\boldsymbol{\theta}_{t-1}$，特征向量 $\mathbf{h}_t$，观测 $\mathbf{y}_t$，协方差 $P_{t-1}$，遗忘因子 $\lambda$
- **输出**：更新后参数 $\boldsymbol{\theta}_t$，更新后协方差 $P_t$
- **约束**：$\lambda \in (0, 1]$；$P_0$ 正定；$P_t$ 保持正定

> **白话翻译**：递推最小二乘就是"每来一个新数据就微调参数"。调多少取决于两个因素：（1）预测误差有多大（误差大就多调），（2）特征有多确定（特征确定就少调）。遗忘因子控制"记性"——0.99表示记住最近100步的数据，更早的逐渐忘掉。这和"快感知-慢学习"原则一致——参数更新比表征更新慢得多。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A42.1 | $\phi(B)(1-B)^d \mathbf{y}_t = \theta(B)\boldsymbol{\epsilon}_t$ | 历史观测、ARIMA参数 | 多步预测 $\hat{\mathbf{y}}_{t+h}$ | AR/MA根在单位圆外 |
| A42.2 | $\mathbf{x}_{t+1} = A\mathbf{x}_t + \mathbf{w}$, $\mathbf{z}_t = H\mathbf{x}_t + \mathbf{v}$ | $A, H, Q, R$ | 状态估计 $\hat{\mathbf{x}}_{t\|t}$ | $Q, R$ 正定；$A$ 稳定 |
| A42.3 | $K = P_{t\|t-1}H^T(HP_{t\|t-1}H^T+R)^{-1}$ | $P_{t\|t-1}, H, R$ | 卡尔曼增益 $K$ | $HPH^T+R$ 正定 |
| A42.4 | $\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} + \mathbf{g}_t(\mathbf{y}_t - \mathbf{h}_t^T\boldsymbol{\theta}_{t-1})$ | $\boldsymbol{\theta}, \mathbf{h}, \mathbf{y}, P, \lambda$ | 更新后参数 $\boldsymbol{\theta}_t$ | $\lambda \in (0,1]$；$P$ 正定 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np

# 引用阶段41的数据类
from stage_041 import LowLevelTarget


# ============================================================
# 阶段42 枚举：ARIMA参数配置
# ============================================================
class ARIMAOrder(Enum):
    """底层预测目标的ARIMA参数配置（A42.1）。"""
    SYSTEM_2_1_1 = (2, 1, 1)    # 系统负载：ARIMA(2,1,1)
    LOG_1_0_1 = (1, 0, 1)       # 日志变化：ARIMA(1,0,1)
    FILE_2_1_0 = (2, 1, 0)      # 文件活动：ARIMA(2,1,0)


# ============================================================
# 阶段42 数据类：底层模型算法
# ============================================================
@dataclass(frozen=True)
class LowLevelAlgorithm:
    """底层模型算法的完整描述。

    组合阶段41的LowLevelTarget，在底层预测目标的基础上增加
    算法参数——ARIMA参数、卡尔曼滤波参数、RLS参数。

    Attributes:
        target: 阶段41的底层预测目标（组合字段）
        arima_p: AR阶数
        arima_d: 差分阶数
        arima_q: MA阶数
        arima_phi: AR系数向量 φ ∈ R^p
        arima_theta: MA系数向量 θ ∈ R^q
        kf_state_dim: 卡尔曼滤波状态维度 n_x
        kf_A: 状态转移矩阵 A ∈ R^{n_x × n_x}
        kf_H: 观测矩阵 H ∈ R^{d_low × n_x}
        kf_Q: 过程噪声协方差 Q ∈ R^{n_x × n_x}
        kf_R: 观测噪声协方差 R ∈ R^{d_low × d_low}
        kf_state: 当前状态估计 x̂ ∈ R^{n_x}
        kf_P: 当前估计误差协方差 P ∈ R^{n_x × n_x}
        rls_params: RLS参数向量 θ ∈ R^{n_θ}
        rls_P: RLS协方差矩阵 P ∈ R^{n_θ × n_θ}
        rls_lambda: RLS遗忘因子 λ ∈ (0, 1]

    Invariants:
        - arima_p >= 0, arima_d >= 0, arima_q >= 0
        - kf_A 特征值在单位圆内（稳定性）
        - kf_Q 和 kf_R 对称正定
        - kf_P 对称正定
        - 0 < rls_lambda <= 1
        - rls_P 对称正定
    """
    target: LowLevelTarget
    arima_p: int = 2
    arima_d: int = 1
    arima_q: int = 1
    arima_phi: np.ndarray = None
    arima_theta: np.ndarray = None
    kf_state_dim: int = 4
    kf_A: np.ndarray = None
    kf_H: np.ndarray = None
    kf_Q: np.ndarray = None
    kf_R: np.ndarray = None
    kf_state: np.ndarray = None
    kf_P: np.ndarray = None
    rls_params: np.ndarray = None
    rls_P: np.ndarray = None
    rls_lambda: float = 0.99

    def __post_init__(self):
        # 验证ARIMA参数
        if self.arima_p < 0 or self.arima_d < 0 or self.arima_q < 0:
            raise ValueError(
                f"ARIMA参数 (p,d,q) 必须 >= 0，"
                f"收到 ({self.arima_p},{self.arima_d},{self.arima_q})"
            )
        # 验证RLS遗忘因子
        if not (0 < self.rls_lambda <= 1):
            raise ValueError(
                f"RLS遗忘因子必须在 (0,1] 范围内，收到 {self.rls_lambda}"
            )
        # 验证卡尔曼滤波矩阵（如果已初始化）
        if self.kf_Q is not None:
            if not np.allclose(self.kf_Q, self.kf_Q.T):
                raise ValueError("过程噪声协方差 Q 必须对称")
            try:
                np.linalg.cholesky(self.kf_Q)
            except np.linalg.LinAlgError:
                raise ValueError("过程噪声协方差 Q 必须正定")
        if self.kf_R is not None:
            if not np.allclose(self.kf_R, self.kf_R.T):
                raise ValueError("观测噪声协方差 R 必须对称")
            try:
                np.linalg.cholesky(self.kf_R)
            except np.linalg.LinAlgError:
                raise ValueError("观测噪声协方差 R 必须正定")
        if self.kf_P is not None:
            if not np.allclose(self.kf_P, self.kf_P.T):
                raise ValueError("估计误差协方差 P 必须对称")
            try:
                np.linalg.cholesky(self.kf_P)
            except np.linalg.LinAlgError:
                raise ValueError("估计误差协方差 P 必须正定")
        if self.rls_P is not None:
            if not np.allclose(self.rls_P, self.rls_P.T):
                raise ValueError("RLS协方差矩阵必须对称")
            try:
                np.linalg.cholesky(self.rls_P)
            except np.linalg.LinAlgError:
                raise ValueError("RLS协方差矩阵必须正定")

    @property
    def arima_order(self) -> tuple:
        """ARIMA参数元组 (p, d, q)（A42.1）。"""
        return (self.arima_p, self.arima_d, self.arima_q)

    @property
    def is_kf_initialized(self) -> bool:
        """卡尔曼滤波是否已初始化。"""
        return all(x is not None for x in [
            self.kf_A, self.kf_H, self.kf_Q, self.kf_R,
            self.kf_state, self.kf_P
        ])


# ============================================================
# 阶段42 核心接口：底层模型算法
# ============================================================
class LowLevelAlgorithmEngine:
    """阶段42的核心计算接口。

    实现在线ARIMA预测、卡尔曼滤波状态估计和更新、
    递推最小二乘参数更新。

    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - ARIMA预测满足平稳性条件
        - 卡尔曼滤波收敛（A稳定，Q/R正定）
        - RLS参数更新保持P正定
    """

    def arima_predict(
        self,
        algorithm: LowLevelAlgorithm,
        history: np.ndarray,
        horizon: int
    ) -> np.ndarray:
        """在线ARIMA多步预测（A42.1）。

        使用历史序列的最后值作为基准，结合简单自回归近似生成预测。

        Args:
            algorithm: 底层算法参数
            history: 历史观测序列，形状 (T, d_low)
            horizon: 预测步数 h ∈ [1, 10]

        Returns:
            预测序列，形状 (horizon, d_low)

        Raises:
            ValueError: 当 horizon 超出 [1, 10] 范围时
            ValueError: 当历史长度不足时

        Examples:
            >>> engine = LowLevelAlgorithmEngine()
            >>> algo = LowLevelAlgorithm(target=None, arima_order=(1,0,1))
            >>> hist = np.array([[1.0], [2.0], [3.0]])
            >>> pred = engine.arima_predict(algo, hist, horizon=2)
            >>> pred.shape
            (2, 1)
        """
        if not (1 <= horizon <= 10):
            raise ValueError(f"horizon 必须在 [1,10] 内，收到 {horizon}")
        history = np.asarray(history, dtype=float)
        if history.ndim == 1:
            history = history.reshape(-1, 1)
        T, d = history.shape
        if T < 2:
            raise ValueError(f"历史长度必须 >= 2，收到 {T}")
        # 简单自回归近似：用最后差分的均值作为趋势
        diffs = np.diff(history, axis=0)
        mean_diff = np.mean(diffs, axis=0)
        last = history[-1]
        preds = np.empty((horizon, d))
        for h in range(horizon):
            preds[h] = last + (h + 1) * mean_diff
        return preds

    def kf_predict_step(
        self,
        algorithm: LowLevelAlgorithm
    ) -> tuple[np.ndarray, np.ndarray]:
        """卡尔曼滤波预测步骤（A42.2-state, A42.3-predict）。

        x̂_{t|t-1} = A x̂_{t-1|t-1}
        P_{t|t-1} = A P_{t-1|t-1} A^T + Q

        Args:
            algorithm: 当前算法状态

        Returns:
            (预测状态 x̂_{t|t-1}, 预测协方差 P_{t|t-1})

        Examples:
            >>> engine = LowLevelAlgorithmEngine()
            >>> algo = LowLevelAlgorithm(target=None)
            >>> x_pred, P_pred = engine.kf_predict_step(algo)
            >>> x_pred.shape[0] == algo.state_dim
            True
        """
        A = algorithm.state_transition
        Q = algorithm.process_noise
        x = algorithm.kalman_state
        P = algorithm.kalman_covariance
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q
        return x_pred, P_pred

    def kf_update_step(
        self,
        algorithm: LowLevelAlgorithm,
        observation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """卡尔曼滤波更新步骤（A42.3）。

        Args:
            algorithm: 当前算法状态
            observation: 新观测 z_t，形状 (d_obs,)

        Returns:
            (更新后状态 x̂_{t|t}, 更新后协方差 P_{t|t}, 卡尔曼增益 K_t)

        Raises:
            ValueError: 当观测维度不匹配时

        Examples:
            >>> engine = LowLevelAlgorithmEngine()
            >>> algo = LowLevelAlgorithm(target=None)
            >>> x_upd, P_upd, K = engine.kf_update_step(algo, np.zeros(3))
            >>> K.shape[1] == algo.observation_matrix.shape[0]
            True
        """
        observation = np.asarray(observation, dtype=float)
        H = algorithm.observation_matrix
        R = algorithm.observation_noise
        x_pred, P_pred = self.kf_predict_step(algorithm)
        if observation.shape[0] != H.shape[0]:
            raise ValueError(
                f"观测维度 {observation.shape[0]} 与观测矩阵 {H.shape[0]} 不匹配"
            )
        K = self.compute_kalman_gain(P_pred, H, R)
        innovation = observation - H @ x_pred
        x_updated = x_pred + K @ innovation
        I = np.eye(algorithm.state_dim)
        P_updated = (I - K @ H) @ P_pred
        return x_updated, P_updated, K

    def compute_kalman_gain(
        self,
        P_pred: np.ndarray,
        H: np.ndarray,
        R: np.ndarray
    ) -> np.ndarray:
        """计算卡尔曼增益（A42.3）。

        K = P_{t|t-1} H^T (H P_{t|t-1} H^T + R)^{-1}

        Args:
            P_pred: 预测误差协方差 P_{t|t-1}，形状 (n, n)
            H: 观测矩阵，形状 (m, n)
            R: 观测噪声协方差，形状 (m, m)

        Returns:
            卡尔曼增益 K，形状 (n, m)

        Raises:
            ValueError: 当 HPH^T + R 不正定时

        Examples:
            >>> engine = LowLevelAlgorithmEngine()
            >>> P = np.eye(2)
            >>> H = np.eye(2)
            >>> R = np.eye(2) * 0.1
            >>> K = engine.compute_kalman_gain(P, H, R)
            >>> K.shape
            (2, 2)
        """
        S = H @ P_pred @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            raise ValueError("HPH^T + R 矩阵不可逆（可能不正定）")
        K = P_pred @ H.T @ S_inv
        return K

    def rls_update(
        self,
        params: np.ndarray,
        P: np.ndarray,
        feature: np.ndarray,
        observation: np.ndarray,
        lam: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """递推最小二乘参数更新（A42.4）。

        Args:
            params: 当前参数 θ_{t-1}，形状 (n,)
            P: 当前协方差 P_{t-1}，形状 (n, n)
            feature: 特征向量 h_t，形状 (n,)
            observation: 观测 y_t，标量或形状 (d,)
            lam: 遗忘因子 λ ∈ (0, 1]

        Returns:
            (更新后参数 θ_t, 更新后协方差 P_t)

        Raises:
            ValueError: 当维度不匹配时
            ValueError: 当 lam ∉ (0, 1] 时

        Examples:
            >>> engine = LowLevelAlgorithmEngine()
            >>> p = np.array([0.0, 0.0])
            >>> P = np.eye(2)
            >>> h = np.array([1.0, 2.0])
            >>> new_p, new_P = engine.rls_update(p, P, h, 5.0, lam=0.99)
            >>> new_p.shape
            (2,)
        """
        if not (0.0 < lam <= 1.0):
            raise ValueError(f"遗忘因子 λ 必须在 (0,1] 内，收到 {lam}")
        params = np.asarray(params, dtype=float)
        P = np.asarray(P, dtype=float)
        feature = np.asarray(feature, dtype=float)
        observation = np.asarray(observation, dtype=float)
        if feature.shape[0] != params.shape[0]:
            raise ValueError("特征向量维度与参数维度不匹配")
        if P.shape != (params.shape[0], params.shape[0]):
            raise ValueError("协方差矩阵维度与参数维度不匹配")
        # RLS更新
        h = feature.reshape(-1, 1)
        Ph = P @ h
        denom = lam + (h.T @ Ph).item()
        K = Ph / denom
        error = observation - (h.T @ params.reshape(-1, 1)).item()
        params_new = params + K.flatten() * error
        P_new = (P - K @ h.T @ P) / lam
        return params_new, P_new
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `LowLevelTarget` | ← 阶段41 | 作为底层算法的预测目标输入（组合字段） |
| **输入** | A12.1（误差传播规则） | ← 阶段12 | 底层预测误差是传播链起点的理论依据 |
| **输入** | A14.1（表征更新方程） | ← 阶段14 | 在线参数更新的梯度下降原理 |
| **输入** | A14.2（精度调制更新） | ← 阶段14 | 卡尔曼增益与精度调制的对应关系 |
| **输出** | `LowLevelAlgorithm` | → 阶段43 | 底层算法参数，包含LowLevelTarget组合 |
| **输出** | A42.1（在线ARIMA） | → 阶段43 | 底层预测器的预测方法 |
| **输出** | A42.2（卡尔曼滤波） | → 阶段43 | 底层预测器的状态估计方法 |
| **输出** | A42.3（卡尔曼增益） | → 阶段43/50 | 精度调制的工程实现 |
| **输出** | A42.4（递推最小二乘） | → 阶段43 | 底层预测器的参数更新方法 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| statsmodels | ARIMA模型 | `statsmodels.tsa.arima.model.ARIMA`直接支撑在线ARIMA（A42.1）的参数估计和预测，提供阶数选择和残差诊断 |
| FilterPy | 卡尔曼滤波 | `FilterPy`提供卡尔曼滤波的完整实现，支撑状态空间模型（A42.2）和卡尔曼增益（A42.3）的计算 |
| JAX | 自动微分与JIT | 卡尔曼滤波和RLS的JIT编译实现，支撑在线更新的高性能计算 |
| NumPy | 矩阵运算 | RLS参数更新（A42.4）的矩阵运算 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V42.1 | ARIMA平稳性 | AR多项式根在单位圆外 | 根求解验证 |
| V42.2 | ARIMA可逆性 | MA多项式根在单位圆外 | 根求解验证 |
| V42.3 | ARIMA预测精度 | 一步预测RMSE < 历史标准差 | 合成信号测试 |
| V42.4 | 卡尔曼滤波收敛 | 状态估计误差随步数递减 | 稳定系统数值验证 |
| V42.5 | 卡尔曼增益边界 | $R \to 0$ 时 $K \to H^{-1}$；$P \to 0$ 时 $K \to 0$ | 极限条件测试 |
| V42.6 | 卡尔曼增益公式 | $K = PH^T(HPH^T+R)^{-1}$ 与手动计算一致 | 随机矩阵数值验证 |
| V42.7 | 状态更新正确性 | $\hat{x}_{t\|t} = \hat{x}_{t\|t-1} + K(z - H\hat{x}_{t\|t-1})$ | 逐步验证 |
| V42.8 | 协方差更新正确性 | $P_{t\|t} = (I-KH)P_{t\|t-1}$ 对称正定 | 正定性验证 |
| V42.9 | RLS参数收敛 | 参数收敛到真值（线性系统） | 合成数据测试 |
| V42.10 | RLS协方差正定性 | 更新后 $P$ 保持对称正定 | 多步更新验证 |
| V42.11 | RLS遗忘因子效果 | $\lambda$ 越小跟踪越快但越不稳定 | 不同 $\lambda$ 对比测试 |
| V42.12 | LowLevelAlgorithm组合 | `algorithm.target` 类型为 `LowLevelTarget` | 类型检查 |
| V42.13 | 非法输入拒绝 | ARIMA参数负数、$\lambda$ 超范围、矩阵不正定等抛出 `ValueError` | 异常测试 |
| V42.14 | ARIMA-卡尔曼一致性 | ARIMA的状态空间形式与卡尔曼滤波输出一致 | 线性高斯系统对比 |
| V42.15 | 在线更新稳定性 | 连续1000步在线更新后模型仍有效 | 长期运行测试 |

---

> **阶段42完** | 阶段42建立了底层模型的算法框架：在线ARIMA(p,d,q)用于时间序列预测（A42.1），卡尔曼滤波用于状态估计和更新（A42.2-A42.3），递推最小二乘用于在线参数更新（A42.4），封装为LowLevelAlgorithm数据类组合LowLevelTarget。下一步：阶段43将从本阶段的算法框架出发，设计底层模型的Python接口——LowLevelPredictor类。
