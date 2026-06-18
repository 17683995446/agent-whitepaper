# 阶段47：高层模型（低频抽象预测）——长期规律与策略优化的高层目标

> **阶段编号**：S47 | **所属层**：第五层·层级预测层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 高层预测目标 | "卫星视角的观察对象"——长期模式、策略优化、认知迭代，最抽象但最深远 |
> | 采集频率0.001Hz | "每1000秒看一次"——高层看的是小时级的变化，关注长期规律 |
> | 预测窗口100-1000步 | "看100到1000步远"——高层预测覆盖小时到天的范围 |
> | HighLevelTarget | "高层预测的完整画像"——组合MidLevelPredictorState，在中层预测基础上增加高层目标 |

---

## 1. 第一性原理依据

本阶段从阶段11（层级生成模型）的核心结论出发，建立层级预测体系的最高层——低频抽象预测模型。核心依据为：

- **A11.1（层级生成模型联合分布）**（引用阶段11）：$p(o, r^{1:L}) = p(o|r^1)\prod_{l=1}^{L}p(r^l|r^{l+1})p(r^{L+1})$，高层模型对应第三层表征 $r^3$ 所编码的信息——长期模式、策略优化、认知迭代。高层表征 $r^3$ 通过条件分布 $p(r^2|r^3)$ 生成中层表征 $r^2$ 的预测，是层级生成模型的最顶层。

- **A11.4（深度-抽象度关系）**（引用阶段11）：$\text{abstraction}(l) \propto \tau_l$，高层的时间常数 $\tau_3$ 满足 $\tau_1 \ll \tau_2 \ll \tau_3$。高层表征变化最慢——编码的是跨小时到天的稳定统计规律，这正是"抽象"的定义。

- **A14.1（表征更新方程）**（引用阶段14）：$\tau_r \dot{r}^l = \delta^{l+1} - e^l$，高层表征 $r^3$ 的更新由顶层的先验约束和中层误差 $e^2$ 驱动。高层预测误差 $e^3$ 度量了高层抽象预测与中层趋势汇总的偏差。

- **A46.1-A46.4（中层预测器接口）**（引用阶段46）：中层预测器提供predict/update/discover_causal三个方法，高层模型使用中层预测器的输出作为输入——中层趋势的汇总和因果结构是高层长期预测的数据来源。

本阶段的核心目标是：**定义高层预测目标（long_term_pattern, strategy_optimize, cognitive_iteration）、确定采集频率（0.001Hz）、设定预测窗口（100-1000步）、建立高层与中层的信号聚合关系，并封装为HighLevelTarget数据类，组合MidLevelPredictorState作为中层预测器的输入。** 高层模型是层级预测体系的顶层——它将中层的趋势和因果信号聚合为长期抽象预测，为整个预测体系提供最宏观的视野。

---

## 2. 核心数学公式

### 第一部分：高层预测目标

#### 2.1 高层预测目标定义（A47.1）

高层预测目标 $\mathbf{y}_{\text{high}} \in \mathbb{R}^{d_{\text{high}}}$ 由三个子向量组成：

$$\mathbf{y}_{\text{high}} = [\mathbf{y}_{\text{pattern}}, \mathbf{y}_{\text{strategy}}, \mathbf{y}_{\text{cognitive}}] \tag{A47.1}$$

其中：

- $\mathbf{y}_{\text{pattern}} \in \mathbb{R}^{d_{\text{pattern}}}$：长期模式指标向量，包含周期性强度、周期长度、趋势斜率、趋势拐点频率、模式复杂度等。$d_{\text{pattern}}$ 为模式指标维度（典型值 $d_{\text{pattern}} = 8$）。

  长期模式指标的具体定义：

  | 指标 | 符号 | 含义 |
  |------|------|------|
  | 周期性强度 | $s_{\text{period}} \in [0, 1]$ | 自相关函数第一峰值的归一化高度 |
  | 主周期长度 | $T_{\text{period}} \in \mathbb{R}_{> 0}$ | 自相关函数第一峰值对应的滞后步数 |
  | 趋势斜率 | $\beta_{\text{trend}} \in \mathbb{R}$ | 线性回归斜率（正=上升，负=下降） |
  | 趋势R² | $R^2_{\text{trend}} \in [0, 1]$ | 线性回归的拟合优度 |
  | 拐点频率 | $f_{\text{inflection}} \in \mathbb{R}_{\geq 0}$ | 单位时间内趋势拐点的次数 |
  | 模式复杂度 | $C_{\text{pattern}} \in \mathbb{R}_{\geq 0}$ | 样本熵或近似熵（度量序列的复杂程度） |
  | 长期均值 | $\mu_{\text{long}} \in \mathbb{R}$ | 长期滑动均值 |
  | 长期方差 | $\sigma^2_{\text{long}} \in \mathbb{R}_{\geq 0}$ | 长期滑动方差 |

- $\mathbf{y}_{\text{strategy}} \in \mathbb{R}^{d_{\text{strategy}}}$：策略优化指标向量，包含当前策略得分、策略空间探索率、策略改进速率、最优策略距离等。$d_{\text{strategy}}$ 为策略指标维度（典型值 $d_{\text{strategy}} = 4$）。

  策略优化指标的具体定义：

  | 指标 | 符号 | 含义 |
  |------|------|------|
  | 当前策略得分 | $V_{\text{strategy}} \in \mathbb{R}$ | 当前策略的期望累积收益 |
  | 策略探索率 | $\rho_{\text{explore}} \in [0, 1]$ | 策略空间中已探索区域的比例 |
  | 策略改进速率 | $\dot{V}_{\text{strategy}} \in \mathbb{R}$ | 策略得分的导数（每步改进量） |
  | 最优策略距离 | $d_{\text{optimal}} \in \mathbb{R}_{\geq 0}$ | 当前策略与估计最优策略的KL散度 |

- $\mathbf{y}_{\text{cognitive}} \in \mathbb{R}^{d_{\text{cognitive}}}$：认知迭代指标向量，包含模型复杂度、自由能水平、预测准确率、认知稳定度等。$d_{\text{cognitive}}$ 为认知指标维度（典型值 $d_{\text{cognitive}} = 4$）。

  认知迭代指标的具体定义：

  | 指标 | 符号 | 含义 |
  |------|------|------|
  | 模型复杂度 | $C_{\text{model}} \in \mathbb{R}_{\geq 0}$ | 变分分布与先验的KL散度 $D_{\text{KL}}[q\|p]$ |
  | 自由能水平 | $F \in \mathbb{R}$ | 当前变分自由能值 |
  | 预测准确率 | $\text{acc} \in [0, 1]$ | 长期预测误差低于阈值的比例 |
  | 认知稳定度 | $S_{\text{cog}} \in [0, 1]$ | 表征变化的逆方差（越稳定越高） |

高层预测目标的总维度为：

$$d_{\text{high}} = d_{\text{pattern}} + d_{\text{strategy}} + d_{\text{cognitive}} \tag{A47.1-dim}$$

典型值 $d_{\text{high}} = 16$。

- **输入**：中层预测器输出（TrendPrediction, CausalGraph）、长期历史数据
- **输出**：高层预测目标向量 $\mathbf{y}_{\text{high}} \in \mathbb{R}^{d_{\text{high}}}$
- **约束**：周期性强度 ∈ [0,1]；策略探索率 ∈ [0,1]；预测准确率 ∈ [0,1]

> **白话翻译**：高层预测目标就是"卫星视角看到的东西"——长期模式（有没有周期、趋势怎么走、多复杂）、策略优化（当前策略好不好、还有多少改进空间）、认知迭代（模型多复杂、预测多准、认知稳不稳定）。这三个维度拼起来就是高层预测的完整目标，最抽象但最深远。

**高层目标与中层目标的信号聚合**：

高层预测目标从中层预测器的长期输出聚合而来：

$$\mathbf{y}_{\text{high}} = \text{Aggregate}_{\text{long}}(\{\hat{\mathbf{y}}_{\text{mid}}(t-k), \mathcal{G}_{\text{mid}}(t-k)\}_{k=0}^{W_{\text{long}}-1}) \tag{A47.1-aggregate}$$

其中 $W_{\text{long}}$ 为长期聚合窗口（典型值 $W_{\text{long}} = 1000$，对应1000步中层预测 = 约28小时的实际时间）。

---

### 第二部分：采集频率

#### 2.2 采集频率定义（A47.2）

高层预测目标的采集频率为0.001Hz：

$$f_{\text{high}} = 0.001\text{Hz}, \quad \Delta t_{\text{high}} = 1000\text{s} \approx 17\text{min} \tag{A47.2}$$

**频率选择的理论依据**：

由A11.4（引用阶段11），高层时间常数 $\tau_3$ 满足 $\tau_2 \ll \tau_3$。由A50.4（跨层时间尺度对齐），$\Delta t_{\text{mid}} : \Delta t_{\text{high}} = 10 : 1$，因此 $\Delta t_{\text{high}} = 10 \times \Delta t_{\text{mid}} = 1000\text{s}$。

**三层频率对齐**：

$$\Delta t_{\text{low}} : \Delta t_{\text{mid}} : \Delta t_{\text{high}} = 1 : 10 : 100 \tag{A47.2-align}$$

对应 $1\text{s} : 10\text{s} : 100\text{s}$（以底层1Hz为基准，中层0.1Hz，高层0.01Hz的等效采样率）。但考虑到中层需要100步底层聚合（A44.2），高层需要100步中层聚合，实际有效采样间隔为：

$$\Delta t_{\text{high}}^{\text{eff}} = 100 \times \Delta t_{\text{mid}} = 100 \times 100\text{s} = 10000\text{s} \approx 2.8\text{h} \tag{A47.2-effective}$$

- **输入**：中层预测器的100步输出
- **输出**：高层预测目标的一次采样
- **约束**：三层频率比 1:10:100

> **白话翻译**：高层每1000秒看一次——不是直接看原始数据，而是看中层100次采样的汇总。三层频率严格对齐：底层1秒、中层100秒、高层1000秒（实际有效间隔约2.8小时，因为需要聚合）。越高层看得越粗但越远。

---

### 第三部分：预测窗口

#### 2.3 预测窗口定义（A47.3）

高层预测的预测窗口 $h_{\text{high}}$ 定义为预测步数范围：

$$h_{\text{high}} \in [100, 1000] \tag{A47.3}$$

以高层的有效采样间隔为基准，预测窗口对应的时间范围为：

$$T_{\text{predict}}^{\text{high}} = h_{\text{high}} \cdot \Delta t_{\text{high}}^{\text{eff}} \in [10^6\text{s}, 10^7\text{s}] \approx [12\text{days}, 116\text{days}] \tag{A47.3-time}$$

**预测窗口的选择依据**：

1. **周期检测的最小窗口**：周期检测（阶段48，A48.3）需要至少覆盖2个完整周期。对于日周期（$T = 86400\text{s}$），至少需要 $2T / \Delta t_{\text{high}}^{\text{eff}} \approx 18$ 步。因此 $h_{\text{high}} \geq 100$ 提供了充足的余量。

2. **策略优化的收敛需求**：策略空间搜索（阶段48，A48.4）需要足够的步数才能收敛到最优策略附近。经验上，$h_{\text{high}} \geq 200$ 步才能可靠评估策略效果。

3. **预测精度约束**：高层预测误差方差增长更慢（$\alpha_{\text{high}} < \alpha_{\text{mid}}$），但步数范围更大。当 $h > 1000$ 时，预测不再可靠。

- **输入**：当前时刻 $t$，预测步数 $h_{\text{high}} \in [100, 1000]$
- **输出**：预测序列 $\hat{\mathbf{Y}}_{\text{high}} \in \mathbb{R}^{h_{\text{high}} \times d_{\text{high}}}$
- **约束**：$100 \leq h_{\text{high}} \leq 1000$；预测误差方差随步数递增

> **白话翻译**：高层预测看100到1000步远（约12天到116天）。比中层的10-100步远得多，因为高层关注的是长期规律和策略，变化最慢，可以看得最远。但超过1000步（约4个月）就不靠谱了。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A47.1 | $\mathbf{y}_{\text{high}} = [\mathbf{y}_{\text{pattern}}, \mathbf{y}_{\text{strategy}}, \mathbf{y}_{\text{cognitive}}]$ | 中层预测器输出、长期历史 | $\mathbf{y}_{\text{high}} \in \mathbb{R}^{d_{\text{high}}}$ | $d_{\text{high}} = d_{\text{pattern}} + d_{\text{strategy}} + d_{\text{cognitive}}$ |
| A47.2 | $f_{\text{high}} = 0.001\text{Hz}$, $\Delta t_{\text{high}} = 1000\text{s}$ | 中层100步聚合 | 高层一次采样 | 三层频率比 1:10:100 |
| A47.3 | $h_{\text{high}} \in [100, 1000]$ | 预测步数 | 预测序列 $\hat{\mathbf{Y}}_{\text{high}}$ | 误差方差随步数递增 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np

# 引用阶段46的数据类
from stage_046 import MidLevelPredictorState, TrendPrediction, CausalGraph


# ============================================================
# 阶段47 数据类：高层预测目标
# ============================================================
@dataclass(frozen=True)
class HighLevelTarget:
    """高层预测目标的完整描述。

    组合阶段46的MidLevelPredictorState，在中层预测器状态的基础上增加
    高层预测目标——长期模式、策略优化、认知迭代三个子向量，
    以及采集频率、预测窗口、聚合参数。

    Attributes:
        mid_predictor_state: 阶段46的中层预测器状态（组合字段）
        long_term_pattern: 长期模式指标向量 y_pattern ∈ R^{d_pattern}
        strategy_optimize: 策略优化指标向量 y_strategy ∈ R^{d_strategy}
        cognitive_iteration: 认知迭代指标向量 y_cognitive ∈ R^{d_cognitive}
        sampling_hz: 高层采集频率（默认0.001Hz）
        prediction_horizon: 预测窗口 h_high ∈ [100, 1000]
        aggregation_window: 中层聚合窗口（默认100步）
        long_aggregation_window: 长期聚合窗口 W_long（默认1000步）

    Invariants:
        - long_term_pattern 维度 d_pattern >= 1
        - strategy_optimize 维度 d_strategy >= 1
        - cognitive_iteration 维度 d_cognitive >= 1
        - sampling_hz > 0
        - 100 <= prediction_horizon <= 1000
        - aggregation_window >= 10
    """
    mid_predictor_state: MidLevelPredictorState
    long_term_pattern: np.ndarray
    strategy_optimize: np.ndarray
    cognitive_iteration: np.ndarray
    sampling_hz: float = 0.001
    prediction_horizon: int = 500
    aggregation_window: int = 100
    long_aggregation_window: int = 1000

    def __post_init__(self):
        # 验证长期模式维度
        if self.long_term_pattern.ndim != 1 or len(self.long_term_pattern) < 1:
            raise ValueError("长期模式必须为一维非空向量")
        # 验证策略优化维度
        if self.strategy_optimize.ndim != 1 or len(self.strategy_optimize) < 1:
            raise ValueError("策略优化必须为一维非空向量")
        # 验证认知迭代维度
        if self.cognitive_iteration.ndim != 1 or len(self.cognitive_iteration) < 1:
            raise ValueError("认知迭代必须为一维非空向量")
        # 验证采集频率
        if self.sampling_hz <= 0:
            raise ValueError(
                f"采集频率必须 > 0，收到 {self.sampling_hz}"
            )
        # 验证预测窗口
        if not (100 <= self.prediction_horizon <= 1000):
            raise ValueError(
                f"预测窗口必须在 [100, 1000] 范围内，"
                f"收到 {self.prediction_horizon}"
            )
        # 验证聚合窗口
        if self.aggregation_window < 10:
            raise ValueError(
                f"聚合窗口必须 >= 10，收到 {self.aggregation_window}"
            )

    @property
    def target_vector(self) -> np.ndarray:
        """完整高层预测目标向量 y_high（A47.1）。"""
        return np.concatenate([
            self.long_term_pattern, self.strategy_optimize,
            self.cognitive_iteration
        ])

    @property
    def target_dim(self) -> int:
        """高层预测目标维度 d_high（A47.1-dim）。"""
        return (len(self.long_term_pattern) + len(self.strategy_optimize)
                + len(self.cognitive_iteration))

    @property
    def effective_sampling_interval(self) -> float:
        """有效采样间隔 Δt_high^eff（A47.2-effective）。"""
        return self.aggregation_window * (1.0 / self.sampling_hz)


# ============================================================
# 阶段47 核心接口：高层信号聚合
# ============================================================
class HighLevelAggregator:
    """阶段47的核心计算接口。

    实现从中层预测器输出到高层预测目标的信号聚合：
    中层趋势汇总 → 长期模式，中层因果汇总 → 策略优化，
    中层误差汇总 → 认知迭代。

    所有方法均为纯函数（无副作用），保证可验证性。
    """

    def aggregate_from_mid_level(
        self,
        mid_predictions: list[TrendPrediction],
        mid_causal_graphs: list[CausalGraph],
        window: int
    ) -> HighLevelTarget:
        """从中层预测器输出聚合高层目标（A47.1-aggregate）。

        Args:
            mid_predictions: 中层趋势预测结果列表
            mid_causal_graphs: 中层因果图列表
            window: 长期聚合窗口 >= 100

        Returns:
            HighLevelTarget 实例

        Raises:
            ValueError: 当输入列表长度不足或 window < 100 时

        Examples:
            >>> engine = HighLevelFeatureEngine()
            >>> preds = [TrendPrediction(values=np.zeros((10,3)), directions=np.zeros(10), strengths=np.zeros(10), causal_context={}, confidences=np.ones(10), horizon=10, timestamp=0.0)]
            >>> cgs = [CausalGraph(adjacency_matrix=np.zeros((3,3)), strength_matrix=np.zeros((3,3)), num_nodes=3, num_edges=0, density=0.0)]
            >>> target = engine.aggregate_from_mid_level(preds, cgs, window=100)
            >>> target.target_dim > 0
            True
        """
        if window < 100:
            raise ValueError(f"长期聚合窗口必须 >= 100，收到 {window}")
        if len(mid_predictions) < 10 or len(mid_causal_graphs) < 1:
            raise ValueError("输入列表长度不足")
        y_pattern = self.extract_long_term_pattern(mid_predictions, window)
        y_strategy = self.extract_strategy_optimize(mid_causal_graphs, window)
        y_cognitive = self.extract_cognitive_iteration(mid_predictions, window)
        return HighLevelTarget(
            mid_predictor_state=None,
            periodicity_strength=float(y_pattern[0]),
            dominant_period=float(y_pattern[1]),
            trend_slope=float(y_pattern[2]),
            pattern_complexity=float(y_pattern[3]),
            strategy_score=float(y_strategy[0]),
            exploration_rate=float(y_strategy[1]),
            improvement_rate=float(y_strategy[2]),
            model_complexity=float(y_cognitive[0]),
            free_energy=float(y_cognitive[1]),
            prediction_accuracy=float(y_cognitive[2]),
            sampling_hz=0.001,
            prediction_horizon=500,
            aggregation_window=window,
        )

    def extract_long_term_pattern(
        self,
        mid_predictions: list[TrendPrediction],
        window: int
    ) -> np.ndarray:
        """从中层趋势提取长期模式指标（A47.1）。

        计算周期性、趋势斜率、模式复杂度等。

        Args:
            mid_predictions: 中层趋势预测结果列表
            window: 长期聚合窗口

        Returns:
            长期模式指标向量 y_pattern，形状 (4,)

        Examples:
            >>> engine = HighLevelFeatureEngine()
            >>> preds = [TrendPrediction(values=np.array([[float(i)]]*10), directions=np.zeros(10), strengths=np.zeros(10), causal_context={}, confidences=np.ones(10), horizon=10, timestamp=float(i)) for i in range(12)]
            >>> y = engine.extract_long_term_pattern(preds, window=100)
            >>> y.shape[0] == 4
            True
        """
        recent = mid_predictions[-min(len(mid_predictions), 20):]
        values = np.concatenate([p.values.flatten() for p in recent])
        # 周期性强度：自相关峰值
        if len(values) > 1:
            corr = np.correlate(values - np.mean(values), values - np.mean(values), mode='full')
            corr = corr[len(corr)//2:]
            periodicity = float(corr[1:].max() / (corr[0] + 1e-8)) if len(corr) > 1 else 0.0
        else:
            periodicity = 0.0
        # 主导周期：FFT峰值频率
        if len(values) > 2:
            fft = np.fft.rfft(values)
            freqs = np.fft.rfftfreq(len(values))
            dominant_period = 1.0 / (freqs[np.argmax(np.abs(fft[1:])) + 1] + 1e-8)
        else:
            dominant_period = 0.0
        # 趋势斜率
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0] if len(values) > 1 else 0.0
        # 模式复杂度：样本熵近似（标准差比率）
        complexity = float(np.std(np.diff(values)) / (np.std(values) + 1e-8))
        return np.array([periodicity, dominant_period, slope, complexity], dtype=float)

    def extract_strategy_optimize(
        self,
        mid_causal_graphs: list[CausalGraph],
        window: int
    ) -> np.ndarray:
        """从中层因果图提取策略优化指标（A47.1）。

        计算策略得分、探索率、改进速率等。

        Args:
            mid_causal_graphs: 中层因果图列表
            window: 长期聚合窗口

        Returns:
            策略优化指标向量 y_strategy，形状 (3,)

        Examples:
            >>> engine = HighLevelFeatureEngine()
            >>> cgs = [CausalGraph(adjacency_matrix=np.eye(3), strength_matrix=np.eye(3), num_nodes=3, num_edges=3, density=0.5)]
            >>> y = engine.extract_strategy_optimize(cgs, window=100)
            >>> y.shape[0] == 3
            True
        """
        recent = mid_causal_graphs[-min(len(mid_causal_graphs), 10):]
        densities = np.array([cg.density for cg in recent])
        # 策略得分：因果图密度均值
        score = float(np.mean(densities))
        # 探索率：密度变化率
        exploration = float(np.std(densities))
        # 改进速率：密度增长趋势
        if len(densities) > 1:
            improvement = float(np.polyfit(np.arange(len(densities)), densities, 1)[0])
        else:
            improvement = 0.0
        return np.array([score, exploration, improvement], dtype=float)

    def extract_cognitive_iteration(
        self,
        mid_predictions: list[TrendPrediction],
        window: int
    ) -> np.ndarray:
        """从中层预测提取认知迭代指标（A47.1）。

        计算模型复杂度、自由能、预测准确率等。

        Args:
            mid_predictions: 中层趋势预测结果列表
            window: 长期聚合窗口

        Returns:
            认知迭代指标向量 y_cognitive，形状 (3,)

        Examples:
            >>> engine = HighLevelFeatureEngine()
            >>> preds = [TrendPrediction(values=np.random.randn(10,3), directions=np.zeros(10), strengths=np.zeros(10), causal_context={}, confidences=np.ones(10), horizon=10, timestamp=float(i)) for i in range(12)]
            >>> y = engine.extract_cognitive_iteration(preds, window=100)
            >>> y.shape[0] == 3
            True
        """
        recent = mid_predictions[-min(len(mid_predictions), 20):]
        confidences = np.concatenate([p.confidences for p in recent])
        # 模型复杂度：置信度方差
        complexity = float(np.var(confidences))
        # 自由能：负对数置信度均值
        free_energy = float(-np.mean(np.log(confidences + 1e-8)))
        # 预测准确率：置信度均值
        accuracy = float(np.mean(confidences))
        return np.array([complexity, free_energy, accuracy], dtype=float)
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `MidLevelPredictorState` | ← 阶段46 | 作为高层目标的中层预测器状态基础（组合字段） |
| **输入** | A11.1（层级生成模型） | ← 阶段11 | 高层目标对应第三层表征的理论依据 |
| **输入** | A11.4（深度-抽象度关系） | ← 阶段11 | 高层时间常数和采集频率的理论依据 |
| **输入** | A14.1（表征更新方程） | ← 阶段14 | 高层表征更新的驱动信号 |
| **输出** | `HighLevelTarget` | → 阶段48 | 高层预测目标数据，包含MidLevelPredictorState组合 |
| **输出** | A47.1（高层预测目标） | → 阶段48/49 | 高层模型算法的输入目标 |
| **输出** | A47.2（采集频率） | → 阶段48/50 | 三层频率对齐和时间尺度协议的依据 |
| **输出** | A47.3（预测窗口） | → 阶段48/49 | 高层预测步数和预测序列的格式定义 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| SciPy | 周期检测与谱分析 | `scipy.signal.periodogram`和`scipy.signal.find_peaks`直接支撑周期性强度和主周期长度的计算（A47.1） |
| NumPy | 统计计算 | 长期均值、方差、线性回归、样本熵的计算 |
| statsmodels | 时间序列分解 | STL分解支撑趋势-周期-残差的分离（A47.1） |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V47.1 | 目标向量组装 | `target_vector == [pattern, strategy, cognitive]` | 拼接验证 |
| V47.2 | 目标维度 | `target_dim == d_pattern + d_strategy + d_cognitive` | 维度断言 |
| V47.3 | 采集频率正值 | `sampling_hz > 0` | 边界测试 |
| V47.4 | 预测窗口范围 | `100 <= prediction_horizon <= 1000` | 边界测试 |
| V47.5 | 周期性强度范围 | `s_period ∈ [0, 1]` | 合成周期信号测试 |
| V47.6 | 主周期长度正确 | 合成周期信号上检测到正确周期 | 已知周期验证 |
| V47.7 | 趋势斜率正确 | 合成线性趋势上斜率误差 < 5% | 已知趋势验证 |
| V47.8 | 策略探索率范围 | `ρ_explore ∈ [0, 1]` | 范围检查 |
| V47.9 | 预测准确率范围 | `acc ∈ [0, 1]` | 范围检查 |
| V47.10 | 认知稳定度范围 | `S_cog ∈ [0, 1]` | 范围检查 |
| V47.11 | 信号聚合正确性 | 聚合结果与手动计算一致 | 合成数据对比 |
| V47.12 | HighLevelTarget组合 | `target.mid_predictor_state` 类型为 `MidLevelPredictorState` | 类型检查 |
| V47.13 | 非法输入拒绝 | 维度为0、频率≤0、窗口超范围等抛出 `ValueError` | 异常测试 |
| V47.14 | 三层频率对齐 | $\Delta t_{\text{low}} : \Delta t_{\text{mid}} : \Delta t_{\text{high}} = 1 : 10 : 100$ | 对齐验证 |

---

> **阶段47完** | 阶段47建立了高层低频抽象预测的目标体系：预测目标包含长期模式、策略优化、认知迭代三个子向量（A47.1），采集频率0.001Hz（A47.2），预测窗口100-1000步（A47.3），封装为HighLevelTarget数据类组合MidLevelPredictorState。下一步：阶段48将从本阶段的高层预测目标出发，设计高层模型的算法——层级贝叶斯、慢特征分析、策略空间搜索。
