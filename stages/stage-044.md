# 阶段44：中层模型（中频趋势预测）——事件序列与状态趋势的中层预测目标

> **阶段编号**：S44 | **所属层**：第五层·层级预测层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 中层预测目标 | "望远镜下的观察对象"——事件序列、状态趋势、短期因果，比底层更粗但更有意义 |
> | 采集频率0.01Hz | "每100秒看一次"——中层看的是分钟级的变化，不需要秒级采样 |
> | 预测窗口10-100步 | "看10到100步远"——中层预测覆盖10分钟到几小时的范围 |
> | MidLevelTarget | "中层预测的完整画像"——组合LowLevelPredictorState，在底层预测基础上增加中层目标 |

---

## 1. 第一性原理依据

本阶段从阶段11（层级生成模型）和阶段14（表征更新规则）的核心结论出发，建立层级预测体系的中层——中频趋势预测模型。核心依据为：

- **A11.1（层级生成模型联合分布）**（引用阶段11）：$p(o, r^{1:L}) = p(o|r^1)\prod_{l=1}^{L}p(r^l|r^{l+1})p(r^{L+1})$，中层模型对应第二层表征 $r^2$ 所编码的信息——事件序列模式、状态趋势方向、短期因果关系。中层表征 $r^2$ 通过条件分布 $p(r^1|r^2)$ 生成底层表征 $r^1$ 的预测。

- **A11.4（深度-抽象度关系）**（引用阶段11）：$\text{abstraction}(l) \propto \tau_l$，中层的时间常数 $\tau_2$ 满足 $\tau_1 \ll \tau_2 \ll \tau_3$。中层表征变化速率介于底层和高层之间——比底层的秒级波动慢，比高层的长期规律快。中层编码的是分钟级到小时级的趋势和模式。

- **A14.1（表征更新方程）**（引用阶段14）：$\tau_r \dot{r}^l = \delta^{l+1} - e^l$，中层表征 $r^2$ 的更新由来自高层的传播信号 $\delta^3$ 和中层的预测误差 $e^2$ 驱动。中层预测误差 $e^2$ 度量了中层趋势预测与底层信号汇总的偏差。

- **A43.1-A43.4（底层预测器接口）**（引用阶段43）：底层预测器提供predict/update/get_error三个方法，中层模型使用底层预测器的输出作为输入——底层预测的汇总信号是中层趋势检测的数据来源。

本阶段的核心目标是：**定义中层预测目标（event_sequence, state_trend, short_term_cause）、确定采集频率（0.01Hz）、设定预测窗口（10-100步）、建立中层与底层的信号聚合关系，并封装为MidLevelTarget数据类，组合LowLevelPredictorState作为底层预测器的输入。** 中层模型是层级预测体系的桥梁——它将底层的高频微预测聚合为中频趋势信号，为高层的长期抽象预测提供结构化的输入。

---

## 2. 核心数学公式

### 第一部分：中层预测目标

#### 2.1 中层预测目标定义（A44.1）

中层预测目标 $\mathbf{y}_{\text{mid}} \in \mathbb{R}^{d_{\text{mid}}}$ 由三个子向量组成：

$$\mathbf{y}_{\text{mid}} = [\mathbf{y}_{\text{event}}, \mathbf{y}_{\text{trend}}, \mathbf{y}_{\text{cause}}] \tag{A44.1}$$

其中：

- $\mathbf{y}_{\text{event}} \in \mathbb{R}^{d_{\text{event}}}$：事件序列指标向量，包含事件类型分布、事件频率、事件间隔统计、事件关联强度等。$d_{\text{event}}$ 为事件指标维度（典型值 $d_{\text{event}} = 8$）。

  事件序列指标的具体定义：

  | 指标 | 符号 | 含义 |
  |------|------|------|
  | 事件类型分布 | $p_{\text{type}} \in \Delta^{K-1}$ | 最近窗口内各类事件的比例（$K$ 为事件类型数） |
  | 事件频率 | $f_{\text{event}} \in \mathbb{R}_{\geq 0}$ | 单位时间内事件发生的次数 |
  | 事件间隔均值 | $\bar{\tau}_{\text{interval}} \in \mathbb{R}_{> 0}$ | 相邻事件间隔时间的均值 |
  | 事件间隔方差 | $\sigma^2_{\text{interval}} \in \mathbb{R}_{\geq 0}$ | 事件间隔时间的方差 |
  | 事件关联强度 | $\rho_{\text{assoc}} \in [-1, 1]$ | 不同类型事件之间的关联度 |

- $\mathbf{y}_{\text{trend}} \in \mathbb{R}^{d_{\text{trend}}}$：状态趋势指标向量，包含趋势方向、趋势强度、趋势持续时间、转折点计数等。$d_{\text{trend}}$ 为趋势指标维度（典型值 $d_{\text{trend}} = 6$）。

  状态趋势指标的具体定义：

  | 指标 | 符号 | 含义 |
  |------|------|------|
  | 趋势方向 | $\mathbf{d}_{\text{trend}} \in \{-1, 0, +1\}^{d_{\text{low}}}$ | 各底层指标的趋势方向（下降/平稳/上升） |
  | 趋势强度 | $s_{\text{trend}} \in \mathbb{R}_{\geq 0}$ | 趋势的统计显著性（CUSUM统计量） |
  | 趋势持续时间 | $T_{\text{trend}} \in \mathbb{R}_{> 0}$ | 当前趋势已持续的步数 |
  | 转折点计数 | $n_{\text{change}} \in \mathbb{Z}_{\geq 0}$ | 最近窗口内趋势反转的次数 |

- $\mathbf{y}_{\text{cause}} \in \mathbb{R}^{d_{\text{cause}}}$：短期因果指标向量，包含因果图密度、因果强度矩阵的谱特征、因果预测误差等。$d_{\text{cause}}$ 为因果指标维度（典型值 $d_{\text{cause}} = 4$）。

  短期因果指标的具体定义：

  | 指标 | 符号 | 含义 |
  |------|------|------|
  | 因果图密度 | $\rho_{\text{graph}} \in [0, 1]$ | 因果图中边的比例 |
  | 最大因果强度 | $s_{\text{max}} \in \mathbb{R}_{\geq 0}$ | 因果强度矩阵的最大特征值 |
  | 因果预测误差 | $e_{\text{cause}} \in \mathbb{R}_{\geq 0}$ | 基于因果模型的预测误差 |
  | Granger因果数 | $n_{\text{granger}} \in \mathbb{Z}_{\geq 0}$ | 通过Granger因果检验的变量对数 |

中层预测目标的总维度为：

$$d_{\text{mid}} = d_{\text{event}} + d_{\text{trend}} + d_{\text{cause}} \tag{A44.1-dim}$$

典型值 $d_{\text{mid}} = 18$。

- **输入**：底层预测器输出（Prediction, ErrorVector）、事件流数据
- **输出**：中层预测目标向量 $\mathbf{y}_{\text{mid}} \in \mathbb{R}^{d_{\text{mid}}}$
- **约束**：事件类型分布为概率单纯形上的点；趋势方向为离散值；因果强度非负

> **白话翻译**：中层预测目标就是"望远镜下看到的东西"——事件序列（什么事在发生、多频繁、怎么关联）、状态趋势（在涨还是在跌、趋势多强、持续多久）、短期因果（谁导致了谁、因果关系多强）。这三个维度拼起来就是中层预测的完整目标，比底层更粗但更有意义。

**中层目标与底层目标的信号聚合**：

中层预测目标不是直接从原始数据提取的，而是从底层预测器的输出聚合而来。聚合公式为：

$$\mathbf{y}_{\text{mid}} = \text{Aggregate}(\{\hat{\mathbf{y}}_{\text{low}}(t-k), \mathbf{e}_{\text{low}}(t-k)\}_{k=0}^{W-1}) \tag{A44.1-aggregate}$$

其中 $W$ 为聚合窗口大小（典型值 $W = 100$，对应100步底层预测）。聚合操作包括：
- 底层预测值的滑动统计量 → 状态趋势
- 底层误差的累积模式 → 事件序列
- 底层指标间的相关性 → 短期因果

---

### 第二部分：采集频率

#### 2.2 采集频率定义（A44.2）

中层预测目标的采集频率为0.01Hz：

$$f_{\text{mid}} = 0.01\text{Hz}, \quad \Delta t_{\text{mid}} = 100\text{s} \tag{A44.2}$$

**频率选择的理论依据**：

由A11.4（引用阶段11），中层时间常数 $\tau_2$ 满足 $\tau_1 \ll \tau_2$。由A50.4（跨层时间尺度对齐），$\Delta t_{\text{low}} : \Delta t_{\text{mid}} = 1 : 10$，因此 $\Delta t_{\text{mid}} = 10 \times \Delta t_{\text{low}} = 10\text{s}$。

但中层预测目标需要从底层预测器的输出聚合而来，聚合窗口 $W = 100$ 步（对应100秒的底层数据）。因此中层有效采样间隔为 $W \times \Delta t_{\text{base}} = 100\text{s}$，即 $f_{\text{mid}} = 0.01\text{Hz}$。

**与底层频率的对齐**：

中层每采样一次，底层已经采样了100次。中层目标是从这100次底层采样中提取的统计量：

$$\mathbf{y}_{\text{mid}}(t) = \text{Aggregate}\left(\{\mathbf{y}_{\text{low}}(t - 99), \ldots, \mathbf{y}_{\text{low}}(t)\}\right) \tag{A44.2-align}$$

这种多尺度对齐确保了中层信号是底层信号的充分统计量——不丢失关键信息，但大幅降低了数据维度和计算量。

- **输入**：底层预测器的100步输出
- **输出**：中层预测目标的一次采样
- **约束**：聚合窗口 $W = 100$；中层采样间隔 = $W \times \Delta t_{\text{base}}$

> **白话翻译**：中层每100秒看一次——不是直接看原始数据，而是看底层100次采样的汇总。就像日报——不是每小时的数据，而是一天的统计。底层看秒级细节，中层看分钟级趋势。

---

### 第三部分：预测窗口

#### 2.3 预测窗口定义（A44.3）

中层预测的预测窗口 $h_{\text{mid}}$ 定义为预测步数范围：

$$h_{\text{mid}} \in [10, 100] \tag{A44.3}$$

以中层的100秒采样间隔为基准，预测窗口对应的时间范围为：

$$T_{\text{predict}}^{\text{mid}} = h_{\text{mid}} \cdot \Delta t_{\text{mid}} \in [1000\text{s}, 10000\text{s}] \approx [17\text{min}, 167\text{min}] \tag{A44.3-time}$$

**预测窗口与底层窗口的对应**：

中层预测窗口 $h_{\text{mid}}$ 步对应底层预测窗口 $h_{\text{mid}} \times W$ 步。例如，中层预测10步 = 底层预测1000步（约17分钟），中层预测100步 = 底层预测10000步（约167分钟）。

**预测窗口的选择依据**：

1. **趋势检测的最小窗口**：趋势检测（CUSUM变体，阶段45，A45.4）需要至少10步数据才能可靠检测趋势变化。因此 $h_{\text{mid}} \geq 10$。

2. **因果发现的数据需求**：因果图发现（PC算法，阶段45，A45.2）需要至少50步数据才能建立可靠的因果边。因此 $h_{\text{mid}} \geq 50$ 用于因果预测。

3. **预测精度约束**：与底层类似，中层预测误差方差随步数增长：

$$\sigma^2_{\text{mid}}(h) = \sigma^2_{\text{mid},0} \cdot (1 + \alpha_{\text{mid}} h) \tag{A44.3-error}$$

其中 $\alpha_{\text{mid}} < \alpha_{\text{low}}$（中层变化更慢，误差增长更慢），但步数范围更大。当 $h > 100$ 时，预测不再可靠。

- **输入**：当前时刻 $t$，预测步数 $h_{\text{mid}} \in [10, 100]$
- **输出**：预测序列 $\hat{\mathbf{Y}}_{\text{mid}} \in \mathbb{R}^{h_{\text{mid}} \times d_{\text{mid}}}$
- **约束**：$10 \leq h_{\text{mid}} \leq 100$；预测误差方差随步数递增

> **白话翻译**：中层预测看10到100步远（17分钟到3小时）。比底层的1-10步远得多，因为中层关注的是趋势和模式，变化更慢，可以看得更远。但也不能太远——超过100步（3小时以上）就不靠谱了。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A44.1 | $\mathbf{y}_{\text{mid}} = [\mathbf{y}_{\text{event}}, \mathbf{y}_{\text{trend}}, \mathbf{y}_{\text{cause}}]$ | 底层预测器输出、事件流 | $\mathbf{y}_{\text{mid}} \in \mathbb{R}^{d_{\text{mid}}}$ | $d_{\text{mid}} = d_{\text{event}} + d_{\text{trend}} + d_{\text{cause}}$ |
| A44.2 | $f_{\text{mid}} = 0.01\text{Hz}$, $\Delta t_{\text{mid}} = 100\text{s}$ | 底层100步聚合 | 中层一次采样 | 聚合窗口 $W = 100$ |
| A44.3 | $h_{\text{mid}} \in [10, 100]$ | 预测步数 | 预测序列 $\hat{\mathbf{Y}}_{\text{mid}} \in \mathbb{R}^{h \times d_{\text{mid}}}$ | 误差方差随步数递增 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np

# 引用阶段43的数据类
from stage_043 import LowLevelPredictorState, Prediction, ErrorVector


# ============================================================
# 阶段44 枚举：趋势方向
# ============================================================
class TrendDirection(Enum):
    """状态趋势方向（A44.1）。"""
    DECREASING = -1    # 下降趋势
    STABLE = 0         # 平稳
    INCREASING = 1     # 上升趋势


# ============================================================
# 阶段44 数据类：中层预测目标
# ============================================================
@dataclass(frozen=True)
class MidLevelTarget:
    """中层预测目标的完整描述。

    组合阶段43的LowLevelPredictorState，在底层预测器状态的基础上增加
    中层预测目标——事件序列、状态趋势、短期因果三个子向量，
    以及采集频率、预测窗口、聚合参数。

    Attributes:
        low_predictor_state: 阶段43的底层预测器状态（组合字段）
        event_sequence: 事件序列指标向量 y_event ∈ R^{d_event}
        state_trend: 状态趋势指标向量 y_trend ∈ R^{d_trend}
        short_term_cause: 短期因果指标向量 y_cause ∈ R^{d_cause}
        sampling_hz: 中层采集频率（默认0.01Hz）
        prediction_horizon: 预测窗口 h_mid ∈ [10, 100]
        aggregation_window: 底层聚合窗口 W（默认100步）
        trend_directions: 各底层指标的趋势方向

    Invariants:
        - event_sequence 维度 d_event >= 1
        - state_trend 维度 d_trend >= 1
        - short_term_cause 维度 d_cause >= 1
        - sampling_hz > 0
        - 10 <= prediction_horizon <= 100
        - aggregation_window >= 10
    """
    low_predictor_state: LowLevelPredictorState
    event_sequence: np.ndarray
    state_trend: np.ndarray
    short_term_cause: np.ndarray
    sampling_hz: float = 0.01
    prediction_horizon: int = 50
    aggregation_window: int = 100
    trend_directions: Optional[np.ndarray] = None

    def __post_init__(self):
        # 验证事件序列维度
        if self.event_sequence.ndim != 1 or len(self.event_sequence) < 1:
            raise ValueError(
                f"事件序列必须为一维非空向量"
            )
        # 验证状态趋势维度
        if self.state_trend.ndim != 1 or len(self.state_trend) < 1:
            raise ValueError(
                f"状态趋势必须为一维非空向量"
            )
        # 验证短期因果维度
        if self.short_term_cause.ndim != 1 or len(self.short_term_cause) < 1:
            raise ValueError(
                f"短期因果必须为一维非空向量"
            )
        # 验证采集频率
        if self.sampling_hz <= 0:
            raise ValueError(
                f"采集频率必须 > 0，收到 {self.sampling_hz}"
            )
        # 验证预测窗口
        if not (10 <= self.prediction_horizon <= 100):
            raise ValueError(
                f"预测窗口必须在 [10, 100] 范围内，"
                f"收到 {self.prediction_horizon}"
            )
        # 验证聚合窗口
        if self.aggregation_window < 10:
            raise ValueError(
                f"聚合窗口必须 >= 10，收到 {self.aggregation_window}"
            )

    @property
    def target_vector(self) -> np.ndarray:
        """完整中层预测目标向量 y_mid（A44.1）。"""
        return np.concatenate([
            self.event_sequence, self.state_trend, self.short_term_cause
        ])

    @property
    def target_dim(self) -> int:
        """中层预测目标维度 d_mid（A44.1-dim）。"""
        return (len(self.event_sequence) + len(self.state_trend)
                + len(self.short_term_cause))

    @property
    def sampling_interval(self) -> float:
        """中层采样间隔 Δt_mid = 1/f_mid（A44.2）。"""
        return 1.0 / self.sampling_hz


# ============================================================
# 阶段44 核心接口：中层信号聚合
# ============================================================
class MidLevelAggregator:
    """阶段44的核心计算接口。

    实现从底层预测器输出到中层预测目标的信号聚合：
    底层预测值 → 状态趋势，底层误差 → 事件序列，
    底层相关性 → 短期因果。

    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 聚合窗口 W >= 10
        - 事件类型分布为概率单纯形上的点
        - 因果强度非负
    """

    def aggregate_from_low_level(
        self,
        low_predictions: list[Prediction],
        low_errors: list[ErrorVector],
        window: int
    ) -> MidLevelTarget:
        """从底层预测器输出聚合中层目标（A44.1-aggregate）。

        Args:
            low_predictions: 底层预测结果列表
            low_errors: 底层误差向量列表
            window: 聚合窗口大小 >= 10

        Returns:
            MidLevelTarget 实例

        Raises:
            ValueError: 当输入列表长度不足或 window < 10 时

        Examples:
            >>> engine = MidLevelFeatureEngine()
            >>> preds = [Prediction(values=np.zeros((1,3)), covariances=np.zeros((1,3,3)), confidences=np.array([1.0]), horizon=1, timestamp=0.0)]
            >>> errs = [ErrorVector(raw_error=np.zeros(3), precision_weighted=np.zeros(3), energy=0.0, system_error=np.zeros(3), log_error=np.zeros(0), file_error=np.zeros(0), timestamp=0.0)]
            >>> target = engine.aggregate_from_low_level(preds, errs, window=10)
            >>> target.target_dim > 0
            True
        """
        if window < 10:
            raise ValueError(f"聚合窗口必须 >= 10，收到 {window}")
        if len(low_predictions) < window or len(low_errors) < window:
            raise ValueError("输入列表长度不足聚合窗口")
        y_event = self.extract_event_sequence(low_errors, window)
        y_trend = self.extract_state_trend(low_predictions, window)
        y_cause = self.extract_short_term_cause(low_predictions, window)
        return MidLevelTarget(
            low_predictor_state=None,
            event_type_distribution=tuple(y_event[:3].tolist()),
            event_frequency=float(y_event[3]),
            event_interval_mean=float(y_event[4]),
            event_correlation=float(y_event[5]),
            trend_direction=int(np.sign(y_trend[0])),
            trend_strength=float(abs(y_trend[0])),
            trend_duration=float(y_trend[1]),
            turning_point_count=int(y_trend[2]),
            cause_graph_density=float(y_cause[0]),
            max_cause_strength=float(y_cause[1]),
            cause_spectral_entropy=float(y_cause[2]),
            sampling_hz=0.01,
            prediction_horizon=10,
            aggregation_window=window,
        )

    def extract_event_sequence(
        self,
        low_errors: list[ErrorVector],
        window: int
    ) -> np.ndarray:
        """从底层误差提取事件序列指标（A44.1）。

        误差突增视为事件，统计事件频率、间隔、关联。

        Args:
            low_errors: 底层误差向量列表
            window: 聚合窗口

        Returns:
            事件序列指标向量 y_event，形状 (6,)

        Examples:
            >>> engine = MidLevelFeatureEngine()
            >>> errs = [ErrorVector(raw_error=np.ones(3)*0.1, precision_weighted=np.ones(3)*0.1, energy=0.03, system_error=np.ones(3)*0.1, log_error=np.zeros(0), file_error=np.zeros(0), timestamp=float(i)) for i in range(12)]
            >>> y = engine.extract_event_sequence(errs, window=10)
            >>> y.shape[0] == 6
            True
        """
        recent = low_errors[-window:]
        energies = np.array([e.energy for e in recent])
        threshold = np.mean(energies) + np.std(energies)
        events = energies > threshold
        event_count = int(np.sum(events))
        freq = event_count / window
        intervals = np.diff(np.where(events)[0])
        interval_mean = float(np.mean(intervals)) if len(intervals) > 0 else 0.0
        # 事件类型分布（简化：按能量三等分）
        p_types = np.array([np.sum(energies < threshold / 2),
                            np.sum((energies >= threshold / 2) & (energies < threshold)),
                            np.sum(energies >= threshold)]) / window
        p_types = p_types / np.sum(p_types) if np.sum(p_types) > 0 else np.ones(3) / 3
        corr = float(np.corrcoef(energies[:-1], energies[1:])[0, 1]) if window > 1 else 0.0
        return np.array([
            p_types[0], p_types[1], p_types[2],
            freq, interval_mean, corr,
        ], dtype=float)

    def extract_state_trend(
        self,
        low_predictions: list[Prediction],
        window: int
    ) -> np.ndarray:
        """从底层预测提取状态趋势指标（A44.1）。

        计算趋势方向、强度、持续时间、转折点。

        Args:
            low_predictions: 底层预测结果列表
            window: 聚合窗口

        Returns:
            状态趋势指标向量 y_trend，形状 (3,)

        Examples:
            >>> engine = MidLevelFeatureEngine()
            >>> preds = [Prediction(values=np.array([[float(i)]]), covariances=np.zeros((1,1,1)), confidences=np.array([1.0]), horizon=1, timestamp=float(i)) for i in range(12)]
            >>> y = engine.extract_state_trend(preds, window=10)
            >>> y.shape[0] == 3
            True
        """
        recent = low_predictions[-window:]
        values = np.array([p.values[0, 0] for p in recent])
        # 线性回归斜率作为趋势方向和强度
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0] if len(values) > 1 else 0.0
        # 转折点：符号变化
        diffs = np.diff(values)
        sign_changes = np.sum((diffs[:-1] * diffs[1:]) < 0)
        return np.array([slope, float(window), float(sign_changes)], dtype=float)

    def extract_short_term_cause(
        self,
        low_predictions: list[Prediction],
        window: int
    ) -> np.ndarray:
        """从底层预测提取短期因果指标（A44.1）。

        计算因果图密度、最大因果强度等。

        Args:
            low_predictions: 底层预测结果列表
            window: 聚合窗口

        Returns:
            短期因果指标向量 y_cause，形状 (3,)

        Examples:
            >>> engine = MidLevelFeatureEngine()
            >>> preds = [Prediction(values=np.random.randn(1,3), covariances=np.zeros((1,3,3)), confidences=np.array([1.0]), horizon=1, timestamp=float(i)) for i in range(12)]
            >>> y = engine.extract_short_term_cause(preds, window=10)
            >>> y.shape[0] == 3
            True
        """
        recent = low_predictions[-window:]
        values = np.stack([p.values[0] for p in recent])
        # 简化因果：计算变量间最大互相关
        d = values.shape[1]
        max_corr = 0.0
        for i in range(d):
            for j in range(i + 1, d):
                c = np.corrcoef(values[:, i], values[:, j])[0, 1]
                max_corr = max(max_corr, abs(c))
        density = max_corr
        spectral = float(-np.sum(values ** 2) / (np.sum(values ** 2) + 1e-8))
        return np.array([density, max_corr, spectral], dtype=float)
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `LowLevelPredictorState` | ← 阶段43 | 作为中层目标的底层预测器状态基础（组合字段） |
| **输入** | A11.1（层级生成模型） | ← 阶段11 | 中层目标对应第二层表征的理论依据 |
| **输入** | A11.4（深度-抽象度关系） | ← 阶段11 | 中层时间常数和采集频率的理论依据 |
| **输入** | A14.1（表征更新方程） | ← 阶段14 | 中层表征更新的驱动信号 |
| **输出** | `MidLevelTarget` | → 阶段45 | 中层预测目标数据，包含LowLevelPredictorState组合 |
| **输出** | A44.1（中层预测目标） | → 阶段45/46 | 中层模型算法的输入目标 |
| **输出** | A44.2（采集频率） | → 阶段45/50 | 多频率对齐和时间尺度协议的依据 |
| **输出** | A44.3（预测窗口） | → 阶段45/46 | 中层预测步数和预测序列的格式定义 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| pandas | 时间序列聚合 | `resample()`和`rolling()`直接支撑底层到中层的信号聚合（A44.1-aggregate），事件序列提取和趋势计算 |
| NumPy | 统计计算 | 事件频率、间隔统计、因果强度矩阵的谱特征计算 |
| SciPy | 统计检验 | Granger因果检验支撑短期因果指标（A44.1）的计算 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V44.1 | 目标向量组装 | `target_vector == [event, trend, cause]` | 拼接验证 |
| V44.2 | 目标维度 | `target_dim == d_event + d_trend + d_cause` | 维度断言 |
| V44.3 | 采集频率正值 | `sampling_hz > 0` | 边界测试 |
| V44.4 | 预测窗口范围 | `10 <= prediction_horizon <= 100` | 边界测试 |
| V44.5 | 聚合窗口最小值 | `aggregation_window >= 10` | 边界测试 |
| V44.6 | 事件类型分布 | `sum(p_type) == 1` 且 `p_type >= 0` | 概率单纯形验证 |
| V44.7 | 趋势方向值域 | `trend_direction ∈ {-1, 0, +1}` | 值域检查 |
| V44.8 | 因果强度非负 | `cause_strength >= 0` | 非负性验证 |
| V44.9 | 信号聚合正确性 | 聚合结果与手动计算一致 | 合成数据对比 |
| V44.10 | 事件提取正确性 | 误差突增被正确识别为事件 | 合成事件测试 |
| V44.11 | 趋势提取正确性 | 上升/下降/平稳趋势被正确识别 | 合成趋势测试 |
| V44.12 | MidLevelTarget组合 | `target.low_predictor_state` 类型为 `LowLevelPredictorState` | 类型检查 |
| V44.13 | 非法输入拒绝 | 维度为0、频率≤0、窗口<10等抛出 `ValueError` | 异常测试 |
| V44.14 | 底层-中层频率对齐 | 中层采样间隔 = 聚合窗口 × 底层采样间隔 | 对齐验证 |

---

> **阶段44完** | 阶段44建立了中层中频趋势预测的目标体系：预测目标包含事件序列、状态趋势、短期因果三个子向量（A44.1），采集频率0.01Hz（A44.2），预测窗口10-100步（A44.3），封装为MidLevelTarget数据类组合LowLevelPredictorState。下一步：阶段45将从本阶段的中层预测目标出发，设计中层模型的算法——RNN/GRU状态空间模型、因果图发现、趋势检测。
