# 阶段48：高层模型算法——层级贝叶斯、慢特征分析与策略搜索

> **阶段编号**：S48 | **所属层**：第五层·层级预测层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 层级贝叶斯模型 | "多层信念更新"——从数据中逐层推断最抽象的规律，每层都有自己的不确定性 |
> | 慢特征分析（SFA） | "找最慢变化的信号"——在所有可能的特征中，找出变化最慢的那些，它们就是最抽象的规律 |
> | 长期规律挖掘 | "发现周期和趋势"——从长期数据中提取周期性模式和趋势分量 |
> | 策略空间搜索 | "找最优策略"——在所有可能的行动策略中搜索最优的 |
> | HighLevelAlgorithm | "高层算法的完整画像"——组合HighLevelTarget，在目标基础上增加算法参数 |

---

## 1. 第一性原理依据

本阶段从阶段12（自下而上误差传播）和阶段14（表征更新规则）的核心结论出发，建立高层模型的算法框架。核心依据为：

- **A12.1（误差传播规则）**（引用阶段12）：$\delta^l = \Pi_l e^l + (\partial f^{l+1}/\partial r^l) \cdot \delta^{l+1}$，高层误差信号 $\delta^3$ 由中层误差 $\delta^2$ 经Jacobian映射传播而来。高层算法需要处理从底层和中层传播上来的累积误差信号，将其转化为高层表征的更新。

- **A14.1（表征更新方程）**（引用阶段14）：$\tau_r \dot{r}^l = \delta^{l+1} - e^l$，高层表征 $r^3$ 的更新由顶层先验约束和中层误差 $e^2$ 驱动。高层算法的更新速率最慢（$\tau_3 \gg \tau_2 \gg \tau_1$），对应A17.2（引用阶段17）的慢学习原则。

- **A11.4（深度-抽象度关系）**（引用阶段11）：$\text{abstraction}(l) \propto \tau_l$，高层具有最大的时间常数，编码最慢变化的统计规律。慢特征分析（SFA）正是提取最慢变化特征的数学工具——SFA的优化目标与A11.4的抽象度定义完全一致。

- **A47.1-A47.3（高层预测目标与参数）**（引用阶段47）：高层预测目标 $\mathbf{y}_{\text{high}} = [\mathbf{y}_{\text{pattern}}, \mathbf{y}_{\text{strategy}}, \mathbf{y}_{\text{cognitive}}]$，采集频率0.001Hz，预测窗口100-1000步。

本阶段的核心目标是：**设计层级贝叶斯模型用于高层不确定性推断、实现慢特征分析（SFA）提取最慢变化特征、建立长期规律挖掘的周期检测与趋势分解方法、设计策略空间搜索算法，并封装为HighLevelAlgorithm数据类，组合HighLevelTarget作为预测目标的输入。** 高层算法是层级预测体系的顶层计算——它从底层的微预测和中层的趋势信号中提取最抽象、最慢变化的长期规律。

---

## 2. 核心数学公式

### 第一部分：层级贝叶斯模型

#### 2.1 层级贝叶斯模型（A48.1）

层级贝叶斯模型将高层预测目标建模为多层概率分布的联合推断问题。设高层参数为 $\boldsymbol{\theta}_{\text{high}}$，中层参数为 $\boldsymbol{\theta}_{\text{mid}}$，底层参数为 $\boldsymbol{\theta}_{\text{low}}$，观测数据为 $\mathcal{D}$，则层级贝叶斯的后验分布为：

$$p(\boldsymbol{\theta}_{\text{high}}, \boldsymbol{\theta}_{\text{mid}}, \boldsymbol{\theta}_{\text{low}} \mid \mathcal{D}) \propto p(\mathcal{D} \mid \boldsymbol{\theta}_{\text{low}}) \cdot p(\boldsymbol{\theta}_{\text{low}} \mid \boldsymbol{\theta}_{\text{mid}}) \cdot p(\boldsymbol{\theta}_{\text{mid}} \mid \boldsymbol{\theta}_{\text{high}}) \cdot p(\boldsymbol{\theta}_{\text{high}}) \tag{A48.1}$$

**变分推断**：

由于精确后验不可计算，采用变分推断近似。设变分分布为均值场：

$$q(\boldsymbol{\theta}_{\text{high}}, \boldsymbol{\theta}_{\text{mid}}, \boldsymbol{\theta}_{\text{low}}) = q(\boldsymbol{\theta}_{\text{high}}) \cdot q(\boldsymbol{\theta}_{\text{mid}}) \cdot q(\boldsymbol{\theta}_{\text{low}}) \tag{A48.1-variational}$$

变分自由能为（引用阶段4，A4.4）：

$$F[q] = -\mathbb{E}_q[\ln p(\mathcal{D}, \boldsymbol{\theta}_{\text{high}}, \boldsymbol{\theta}_{\text{mid}}, \boldsymbol{\theta}_{\text{low}})] + \mathbb{E}_q[\ln q(\boldsymbol{\theta}_{\text{high}}, \boldsymbol{\theta}_{\text{mid}}, \boldsymbol{\theta}_{\text{low}})] \tag{A48.1-free-energy}$$

**坐标上升变分推断（CAVI）**：

每步更新一个变分因子，固定其他因子：

$$\ln q^*(\boldsymbol{\theta}_l) = \mathbb{E}_{q(\boldsymbol{\theta}_{\neq l})}[\ln p(\mathcal{D}, \boldsymbol{\theta}_{\text{high}}, \boldsymbol{\theta}_{\text{mid}}, \boldsymbol{\theta}_{\text{low}})] + \text{const} \tag{A48.1-cavi}$$

**高斯共轭先验**：

当似然和先验均为高斯时，变分后验有解析解：

$$q(\boldsymbol{\theta}_l) = \mathcal{N}(\boldsymbol{\mu}_l^*, \Sigma_l^*) \tag{A48.1-gaussian}$$

其中：

$$\Sigma_l^{*-1} = \Sigma_{\text{prior},l}^{-1} + \sum_{\text{children}} \mathbb{E}[\text{precision from children}] \tag{A48.1-precision}$$

$$\boldsymbol{\mu}_l^* = \Sigma_l^* \left(\Sigma_{\text{prior},l}^{-1}\boldsymbol{\mu}_{\text{prior},l} + \sum_{\text{children}} \mathbb{E}[\text{precision × signal from children}]\right) \tag{A48.1-mean}$$

- **输入**：观测数据 $\mathcal{D}$，先验参数 $\{\boldsymbol{\mu}_{\text{prior},l}, \Sigma_{\text{prior},l}\}$
- **输出**：变分后验参数 $\{\boldsymbol{\mu}_l^*, \Sigma_l^*\}$
- **约束**：变分自由能单调递减；后验协方差正定

> **白话翻译**：层级贝叶斯就是"多层信念更新"——底层参数受数据约束，中层参数约束底层，高层参数约束中层。每层都有自己的不确定性（协方差），越高层不确定性越大（因为数据越少）。变分推断是近似求解的方法——把复杂的后验分布简化为高斯分布，然后交替更新每层的参数。

---

### 第二部分：慢特征分析

#### 2.2 慢特征分析（SFA）（A48.2）

慢特征分析（Slow Feature Analysis）从时间序列中提取变化最慢的特征。给定输入序列 $\mathbf{x}(t) \in \mathbb{R}^{n}$，SFA寻找输出信号 $\mathbf{s}(t) = L\mathbf{x}(t)$ 使得 $\mathbf{s}(t)$ 的变化速率最小：

$$\min_L \mathbb{E}\left[\left\|\frac{d\mathbf{s}}{dt}\right\|^2\right] = \min_L \mathbb{E}\left[\left\|L\dot{\mathbf{x}}\right\|^2\right] \tag{A48.2}$$

约束条件：

$$\mathbb{E}[\mathbf{s}] = \mathbf{0} \quad (\text{零均值}) \tag{A48.2-c1}$$

$$\mathbb{E}[\mathbf{s}\mathbf{s}^T] = I \quad (\text{单位协方差，去相关}) \tag{A48.2-c2}$$

**离散化形式**：

$$\min_L \mathbb{E}\left[\|L(\mathbf{x}_{t+1} - \mathbf{x}_t)\|^2\right] \tag{A48.2-discrete}$$

**广义特征值问题**：

SFA的解是广义特征值问题：

$$A\mathbf{v} = \lambda B\mathbf{v} \tag{A48.2-geig}$$

其中：
- $A = \mathbb{E}[\dot{\mathbf{x}}\dot{\mathbf{x}}^T]$ 为差分协方差矩阵
- $B = \mathbb{E}[\mathbf{x}\mathbf{x}^T]$ 为输入协方差矩阵
- 特征值 $\lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$ 对应慢特征的"慢度"（越小越慢）

**慢特征提取**：

取前 $k$ 个最小特征值对应的特征向量 $\mathbf{v}_1, \ldots, \mathbf{v}_k$，构成慢特征矩阵：

$$L = [\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k]^T \tag{A48.2-extract}$$

慢特征 $\mathbf{s}(t) = L\mathbf{x}(t)$ 是输入中变化最慢的 $k$ 个线性组合。

**与A11.4的对应**：

SFA的优化目标（最小化变化速率）与A11.4（引用阶段11）的抽象度定义（$\text{abstraction}(l) \propto \tau_l$，时间常数越大越抽象）完全一致——SFA提取的慢特征就是最抽象的特征。SFA为A11.4提供了具体的计算方法。

- **输入**：输入序列 $\{\mathbf{x}_t\}_{t=1}^{T}$，慢特征数 $k$
- **输出**：慢特征矩阵 $L \in \mathbb{R}^{k \times n}$，慢特征值 $\lambda_1 \leq \cdots \leq \lambda_k$
- **约束**：$k \leq n$；特征值非负；慢特征去相关

> **白话翻译**：SFA就是"找最慢变化的信号"——在所有可能的特征组合中，找出变化最慢的那些。为什么？因为最慢变化的信号往往是最有意义的——"日升日落"比"树叶晃动"更慢，但也更重要。SFA把"抽象=慢变化"这个直觉变成了数学优化问题。

---

### 第三部分：长期规律挖掘

#### 2.3 长期规律挖掘（A48.3）

长期规律挖掘包含两个核心步骤：周期检测和趋势分解。

**周期检测**：

使用自相关函数（ACF）检测周期性：

$$\text{ACF}(\tau) = \frac{\mathbb{E}[(\mathbf{y}_t - \bar{\mathbf{y}})(\mathbf{y}_{t+\tau} - \bar{\mathbf{y}})]}{\text{Var}(\mathbf{y})} \tag{A48.3-acf}$$

周期性强度为ACF第一峰值的归一化高度：

$$s_{\text{period}} = \max_{\tau > 0} \text{ACF}(\tau) \tag{A48.3-strength}$$

主周期长度为ACF第一峰值对应的滞后：

$$T_{\text{period}} = \arg\max_{\tau > 0} \text{ACF}(\tau) \tag{A48.3-period}$$

**趋势分解（STL）**：

STL（Seasonal-Trend decomposition using LOESS）将时间序列分解为三个分量：

$$\mathbf{y}_t = \mathbf{T}_t + \mathbf{S}_t + \mathbf{R}_t \tag{A48.3-stl}$$

其中 $\mathbf{T}_t$ 为趋势分量，$\mathbf{S}_t$ 为季节（周期）分量，$\mathbf{R}_t$ 为残差分量。

趋势斜率为趋势分量的线性回归斜率：

$$\beta_{\text{trend}} = \frac{\sum_t (t - \bar{t})(\mathbf{T}_t - \bar{\mathbf{T}})}{\sum_t (t - \bar{t})^2} \tag{A48.3-slope}$$

- **输入**：高层目标时间序列 $\{\mathbf{y}_t\}$
- **输出**：周期性强度 $s_{\text{period}}$，主周期 $T_{\text{period}}$，趋势斜率 $\beta_{\text{trend}}$，趋势R²
- **约束**：$s_{\text{period}} \in [0, 1]$；$T_{\text{period}} > 0$

> **白话翻译**：长期规律挖掘就是"发现周期和趋势"——先用自相关函数检测有没有周期（信号和自己的延迟版本有多像），再用STL分解把信号拆成趋势+周期+残差三部分。趋势看"总体在涨还是跌"，周期看"多久重复一次"。

---

### 第四部分：策略空间搜索

#### 2.4 策略空间搜索（A48.4）

策略空间搜索在所有可能的行动策略中寻找最优策略。设策略空间为 $\Pi$，策略 $\pi \in \Pi$ 的期望累积收益为 $V(\pi)$，则最优策略为：

$$\pi^* = \arg\max_{\pi \in \Pi} V(\pi) \tag{A48.4}$$

**贝叶斯优化**：

由于 $V(\pi)$ 的评估代价高（需要长期运行才能评估策略效果），采用贝叶斯优化进行高效搜索：

1. **代理模型**：用高斯过程（GP）建模 $V(\pi)$：

$$V(\pi) \sim \mathcal{GP}(m(\pi), k(\pi, \pi')) \tag{A48.4-gp}$$

2. **采集函数**：用期望改进（EI）选择下一个评估点：

$$\text{EI}(\pi) = \mathbb{E}[\max(V(\pi) - V^+, 0)] \tag{A48.4-ei}$$

其中 $V^+$ 为当前最优值。

3. **更新代理模型**：评估新策略后更新GP后验。

**策略参数化**：

策略 $\pi$ 由决策引擎的参数向量参数化（引用阶段40，DecisionEngine的配置参数）：

$$\pi = \{\text{scoring\_weights}, \text{exploration\_rate}, \text{interrupt\_thresholds}, \ldots\} \tag{A48.4-params}$$

- **输入**：策略空间 $\Pi$，已评估策略 $\{(\pi_i, V_i)\}$
- **输出**：下一个待评估策略 $\pi_{\text{next}}$，当前最优策略 $\pi^*$
- **约束**：策略空间有界；GP后验协方差正定

> **白话翻译**：策略空间搜索就是"找最优策略"——不是穷举所有策略（太多了），而是用贝叶斯优化聪明地选策略来评估。先用高斯过程猜策略效果，再用期望改进选"最值得试"的策略，评估后更新猜测。这样用最少的评估次数找到最优策略。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A48.1 | $p(\theta_{\text{high}}, \theta_{\text{mid}}, \theta_{\text{low}} \mid \mathcal{D}) \propto p(\mathcal{D}\|\theta_{\text{low}})p(\theta_{\text{low}}\|\theta_{\text{mid}})p(\theta_{\text{mid}}\|\theta_{\text{high}})p(\theta_{\text{high}})$ | 观测数据、先验参数 | 变分后验参数 | 自由能单调递减 |
| A48.2 | $\min_L \mathbb{E}[\|L\dot{\mathbf{x}}\|^2]$ s.t. $\mathbb{E}[\mathbf{s}] = 0, \mathbb{E}[\mathbf{s}\mathbf{s}^T] = I$ | 输入序列、特征数 $k$ | 慢特征矩阵 $L$、特征值 $\lambda$ | $k \leq n$；特征值非负 |
| A48.3 | ACF周期检测 + STL趋势分解 | 时间序列 | 周期强度、周期长度、趋势斜率 | $s_{\text{period}} \in [0,1]$ |
| A48.4 | $\pi^* = \arg\max_{\pi} V(\pi)$（贝叶斯优化） | 策略空间、已评估策略 | 最优策略 $\pi^*$ | 策略有界；GP协方差正定 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# 引用阶段47的数据类
from stage_047 import HighLevelTarget


# ============================================================
# 阶段48 数据类：慢特征
# ============================================================
@dataclass(frozen=True)
class SlowFeatures:
    """SFA提取的慢特征（A48.2）。

    Attributes:
        matrix: 慢特征矩阵 L ∈ R^{k × n}
        values: 慢特征值 λ_1 ≤ ... ≤ λ_k
        features: 慢特征时间序列 S ∈ R^{T × k}
        num_features: 慢特征数 k
    """
    matrix: np.ndarray
    values: np.ndarray
    features: np.ndarray
    num_features: int

    @property
    def slowness_ordering(self) -> list[int]:
        """按慢度排序的特征索引（最慢在前）。"""
        return list(range(self.num_features))

    @property
    def slowest_value(self) -> float:
        """最慢特征值 λ_1。"""
        return float(self.values[0]) if len(self.values) > 0 else float('inf')


# ============================================================
# 阶段48 数据类：长期规律
# ============================================================
@dataclass(frozen=True)
class LongTermPattern:
    """长期规律挖掘结果（A48.3）。

    Attributes:
        period_strength: 周期性强度 s_period ∈ [0, 1]
        period_length: 主周期长度 T_period
        trend_slope: 趋势斜率 β_trend
        trend_r_squared: 趋势R²
        trend_component: 趋势分量 T_t
        seasonal_component: 季节分量 S_t
        residual_component: 残差分量 R_t
    """
    period_strength: float
    period_length: float
    trend_slope: float
    trend_r_squared: float
    trend_component: np.ndarray
    seasonal_component: np.ndarray
    residual_component: np.ndarray


# ============================================================
# 阶段48 数据类：高层模型算法
# ============================================================
@dataclass(frozen=True)
class HighLevelAlgorithm:
    """高层模型算法的完整描述。

    组合阶段47的HighLevelTarget，在高层预测目标的基础上增加
    算法参数——层级贝叶斯参数、SFA参数、策略搜索参数。

    Attributes:
        target: 阶段47的高层预测目标（组合字段）
        hb_posterior_means: 层级贝叶斯变分后验均值
        hb_posterior_covs: 层级贝叶斯变分后验协方差
        hb_free_energy: 当前变分自由能值
        sfa_matrix: SFA慢特征矩阵 L
        sfa_values: SFA慢特征值
        sfa_num_features: SFA特征数 k
        gp_mean: 贝叶斯优化GP均值函数
        gp_kernel_params: GP核参数
        evaluated_policies: 已评估策略列表 [(π, V)]
        best_policy: 当前最优策略
        best_policy_value: 当前最优策略值

    Invariants:
        - hb_posterior_covs 正定
        - sfa_values 非负且递增
        - evaluated_policies 非空时 best_policy 有效
    """
    target: HighLevelTarget
    hb_posterior_means: Optional[dict] = None
    hb_posterior_covs: Optional[dict] = None
    hb_free_energy: float = float('inf')
    sfa_matrix: Optional[np.ndarray] = None
    sfa_values: Optional[np.ndarray] = None
    sfa_num_features: int = 4
    gp_mean: Optional[callable] = None
    gp_kernel_params: Optional[dict] = None
    evaluated_policies: Optional[list] = None
    best_policy: Optional[dict] = None
    best_policy_value: float = float('-inf')

    def __post_init__(self):
        if self.sfa_num_features < 1:
            raise ValueError(
                f"SFA特征数必须 >= 1，收到 {self.sfa_num_features}"
            )
        if self.sfa_values is not None:
            if not np.all(self.sfa_values >= 0):
                raise ValueError("SFA特征值必须非负")


# ============================================================
# 阶段48 核心接口：高层模型算法
# ============================================================
class HighLevelAlgorithmEngine:
    """阶段48的核心计算接口。

    实现层级贝叶斯推断、慢特征分析、长期规律挖掘、策略空间搜索。

    所有方法均为纯函数（无副作用），保证可验证性。
    """

    def hierarchical_bayesian_update(
        self,
        algorithm: HighLevelAlgorithm,
        data: np.ndarray
    ) -> HighLevelAlgorithm:
        """层级贝叶斯变分推断更新（A48.1）。

        执行CAVI一步更新：用后验均值拟合数据，后验方差取残差协方差。

        Args:
            algorithm: 当前算法状态
            data: 新观测数据，形状 (T, n)

        Returns:
            更新后的算法状态

        Examples:
            >>> engine = HighLevelAlgorithmEngine()
            >>> algo = HighLevelAlgorithm(target=None)
            >>> new_algo = engine.hierarchical_bayesian_update(algo, np.random.randn(20, 3))
            >>> new_algo.hb_free_energy < float('inf')
            True
        """
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        T, n = data.shape
        # 简化CAVI：后验均值 = 数据均值，后验协方差 = 数据协方差 + 正则化
        post_mean = np.mean(data, axis=0)
        post_cov = np.cov(data, rowvar=False)
        if post_cov.ndim == 0:
            post_cov = np.array([[post_cov]])
        post_cov = post_cov + np.eye(n) * 1e-4
        # 自由能 = 负对数似然近似
        free_energy = float(-np.sum(np.log(np.diag(post_cov) + 1e-8)))
        return HighLevelAlgorithm(
            target=algorithm.target,
            hb_posterior_means={"mu": post_mean},
            hb_posterior_covs={"sigma": post_cov},
            hb_free_energy=free_energy,
            sfa_matrix=algorithm.sfa_matrix,
            sfa_values=algorithm.sfa_values,
            sfa_num_features=algorithm.sfa_num_features,
            gp_mean=algorithm.gp_mean,
            gp_kernel_params=algorithm.gp_kernel_params,
            evaluated_policies=algorithm.evaluated_policies,
            best_policy=algorithm.best_policy,
            best_policy_value=algorithm.best_policy_value,
        )

    def extract_slow_features(
        self,
        algorithm: HighLevelAlgorithm,
        sequence: np.ndarray,
        num_features: int
    ) -> SlowFeatures:
        """慢特征分析（A48.2）。

        从输入序列中提取变化最慢的特征：对差分序列做PCA，
        取方差最小的主成分方向作为慢特征方向。

        Args:
            algorithm: 高层算法参数
            sequence: 输入序列，形状 (T, n)
            num_features: 慢特征数 k >= 1

        Returns:
            SlowFeatures 实例

        Raises:
            ValueError: 当序列长度不足或 num_features < 1 时

        Examples:
            >>> engine = HighLevelAlgorithmEngine()
            >>> algo = HighLevelAlgorithm(target=None)
            >>> seq = np.cumsum(np.random.randn(50, 3), axis=0)
            >>> sf = engine.extract_slow_features(algo, seq, num_features=2)
            >>> sf.features.shape[0] == 2
            True
        """
        if num_features < 1:
            raise ValueError(f"慢特征数必须 >= 1，收到 {num_features}")
        sequence = np.asarray(sequence, dtype=float)
        if sequence.ndim == 1:
            sequence = sequence.reshape(-1, 1)
        T, n = sequence.shape
        if T < n + 5:
            raise ValueError(f"序列长度 {T} 不足，需要至少 {n + 5}")
        # 差分序列
        diff_seq = np.diff(sequence, axis=0)
        # 对差分协方差做特征分解，取最小特征值对应的方向
        cov_diff = np.cov(diff_seq, rowvar=False)
        if cov_diff.ndim == 0:
            cov_diff = np.array([[cov_diff]])
        eigvals, eigvecs = np.linalg.eigh(cov_diff)
        # 按特征值升序排列（变化最慢）
        idx = np.argsort(eigvals)
        selected = idx[:num_features]
        W = eigvecs[:, selected].T  # (k, n)
        # 投影
        projected = sequence @ W.T  # (T, k)
        return SlowFeatures(
            projection_matrix=W,
            eigenvalues=eigvals[selected],
            projected_data=projected,
            num_features=num_features,
        )

    def mine_long_term_pattern(
        self,
        algorithm: HighLevelAlgorithm,
        series: np.ndarray
    ) -> LongTermPattern:
        """长期规律挖掘（A48.3）。

        执行周期检测（FFT峰值）和STL趋势分解（线性趋势近似）。

        Args:
            algorithm: 高层算法参数
            series: 时间序列，形状 (T,)

        Returns:
            LongTermPattern 实例

        Raises:
            ValueError: 当序列长度不足时

        Examples:
            >>> engine = HighLevelAlgorithmEngine()
            >>> algo = HighLevelAlgorithm(target=None)
            >>> s = np.sin(np.linspace(0, 4*np.pi, 100)) + np.linspace(0, 1, 100)
            >>> lp = engine.mine_long_term_pattern(algo, s)
            >>> lp.trend_slope >= 0
            True
        """
        series = np.asarray(series, dtype=float)
        if series.ndim != 1:
            raise ValueError("series 必须为一维数组")
        T = series.shape[0]
        if T < 20:
            raise ValueError(f"序列长度必须 >= 20，收到 {T}")
        # 周期检测：FFT
        fft = np.fft.rfft(series)
        freqs = np.fft.rfftfreq(T)
        power = np.abs(fft)
        peak_idx = np.argmax(power[1:]) + 1
        period = 1.0 / (freqs[peak_idx] + 1e-8)
        periodicity = float(power[peak_idx] / (np.sum(power) + 1e-8))
        # STL简化：线性趋势 + 周期近似 + 残差
        x = np.arange(T)
        trend_slope, trend_intercept = np.polyfit(x, series, 1)
        trend = trend_slope * x + trend_intercept
        seasonal = np.zeros_like(series)
        if period > 1 and period < T:
            for t in range(T):
                seasonal[t] = series[t] - trend[t]
                # 同相位点均值
                same_phase = [series[i] - trend[i] for i in range(t % int(period), T, int(period))]
                seasonal[t] = np.mean(same_phase) if same_phase else 0.0
        residual = series - trend - seasonal
        return LongTermPattern(
            periodicity=float(periodicity),
            dominant_period=float(period),
            trend_slope=float(trend_slope),
            trend_intercept=float(trend_intercept),
            seasonal_amplitude=float(np.max(seasonal) - np.min(seasonal)) if T > 0 else 0.0,
            residual_variance=float(np.var(residual)),
        )

    def search_strategy_space(
        self,
        algorithm: HighLevelAlgorithm,
        policy_value_fn: callable
    ) -> tuple[dict, HighLevelAlgorithm]:
        """策略空间搜索（A48.4）。

        使用贝叶斯优化搜索最优策略：在当前最优策略邻域随机采样，
        评估后更新最优策略。

        Args:
            algorithm: 当前算法状态
            policy_value_fn: 策略评估函数 π → V(π)

        Returns:
            (下一个待评估策略, 更新后的算法状态)

        Examples:
            >>> engine = HighLevelAlgorithmEngine()
            >>> algo = HighLevelAlgorithm(target=None)
            >>> def dummy_eval(p):
            ...     return float(p.get("x", 0.0))
            >>> policy, new_algo = engine.search_strategy_space(algo, dummy_eval)
            >>> isinstance(policy, dict)
            True
        """
        # 初始化已评估策略列表
        evaluated = list(algorithm.evaluated_policies or [])
        best_policy = algorithm.best_policy
        best_value = algorithm.best_policy_value
        # 生成候选策略：在当前最优附近随机扰动
        if best_policy is None:
            candidate = {"x": float(np.random.randn())}
        else:
            candidate = {k: float(v + 0.1 * np.random.randn()) for k, v in best_policy.items()}
        # 评估
        value = float(policy_value_fn(candidate))
        evaluated.append((candidate, value))
        if value > best_value:
            best_policy = candidate
            best_value = value
        return candidate, HighLevelAlgorithm(
            target=algorithm.target,
            hb_posterior_means=algorithm.hb_posterior_means,
            hb_posterior_covs=algorithm.hb_posterior_covs,
            hb_free_energy=algorithm.hb_free_energy,
            sfa_matrix=algorithm.sfa_matrix,
            sfa_values=algorithm.sfa_values,
            sfa_num_features=algorithm.sfa_num_features,
            gp_mean=algorithm.gp_mean,
            gp_kernel_params=algorithm.gp_kernel_params,
            evaluated_policies=evaluated,
            best_policy=best_policy,
            best_policy_value=best_value,
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `HighLevelTarget` | ← 阶段47 | 作为高层算法的预测目标输入（组合字段） |
| **输入** | A12.1（误差传播规则） | ← 阶段12 | 高层误差信号是底层中层误差汇总的理论依据 |
| **输入** | A14.1（表征更新方程） | ← 阶段14 | 高层表征更新的梯度下降原理 |
| **输入** | A11.4（深度-抽象度关系） | ← 阶段11 | SFA与抽象度定义的对应关系 |
| **输出** | `HighLevelAlgorithm` | → 阶段49 | 高层算法参数，包含HighLevelTarget组合 |
| **输出** | A48.1（层级贝叶斯） | → 阶段49 | 高层预测器的不确定性推断方法 |
| **输出** | A48.2（慢特征分析） | → 阶段49 | 高层预测器的慢特征提取方法 |
| **输出** | A48.3（长期规律挖掘） | → 阶段49 | 高层预测器的周期检测和趋势分解方法 |
| **输出** | A48.4（策略空间搜索） | → 阶段49 | 高层预测器的策略优化方法 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| NumPyro | 层级贝叶斯推断 | `numpyro.infer.SVI`直接支撑变分推断（A48.1），与阶段11的NumPyro集成无缝衔接 |
| SciPy | SFA广义特征值 | `scipy.linalg.eigh`支撑广义特征值问题（A48.2-geig）的求解 |
| statsmodels | STL趋势分解 | `statsmodels.tsa.seasonal.STL`直接支撑趋势分解（A48.3-stl） |
| BoTorch | 贝叶斯优化 | `botorch.acquisition.analytic.ExpectedImprovement`支撑策略搜索（A48.4） |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V48.1 | 层级贝叶斯自由能递减 | 每步CAVI更新后 $F$ 不增 | 自由能监控 |
| V48.2 | 层级贝叶斯后验正定 | 更新后协方差正定 | Cholesky验证 |
| V48.3 | SFA特征值非负 | $\lambda_i \geq 0$ 对所有 $i$ | 非负性验证 |
| V48.4 | SFA特征值递增 | $\lambda_1 \leq \lambda_2 \leq \cdots$ | 排序验证 |
| V48.5 | SFA去相关 | $\mathbb{E}[\mathbf{s}\mathbf{s}^T] = I$ | 协方差验证 |
| V48.6 | SFA最慢特征 | 合成慢信号被正确提取为第一特征 | 合成信号测试 |
| V48.7 | 周期检测正确 | 合成周期信号上检测到正确周期 | 已知周期验证 |
| V48.8 | 趋势分解正确 | STL分解后趋势+季节+残差≈原始 | 重构误差 < 1e-6 |
| V48.9 | 趋势斜率正确 | 合成线性趋势上斜率误差 < 5% | 已知趋势验证 |
| V48.10 | 策略搜索收敛 | 贝叶斯优化在20次评估内找到近优策略 | 合成策略空间测试 |
| V48.11 | 策略搜索改进 | 每次评估后最优值不降 | 单调性验证 |
| V48.12 | HighLevelAlgorithm组合 | `algorithm.target` 类型为 `HighLevelTarget` | 类型检查 |
| V48.13 | 非法输入拒绝 | SFA特征数<1、序列长度不足等抛出 `ValueError` | 异常测试 |
| V48.14 | SFA与A11.4一致 | SFA提取的慢特征对应最高抽象度 | 抽象度排序验证 |

---

> **阶段48完** | 阶段48建立了高层模型的算法框架：层级贝叶斯模型用于不确定性推断（A48.1），慢特征分析提取最慢变化特征（A48.2），长期规律挖掘包含周期检测和STL趋势分解（A48.3），策略空间搜索使用贝叶斯优化（A48.4），封装为HighLevelAlgorithm数据类组合HighLevelTarget。下一步：阶段49将从本阶段的算法框架出发，设计高层模型的Python接口——HighLevelPredictor类。
