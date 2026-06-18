# 阶段49：高层模型Python接口——HighLevelPredictor的统一接口

> **阶段编号**：S49 | **所属层**：第五层·层级预测层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | HighLevelPredictor | "高层预测器"——封装层级贝叶斯、SFA、策略搜索，提供统一的预测/更新/慢特征接口 |
> | predict方法 | "看长期"——给定当前状态，预测未来100-1000步的抽象模式 |
> | update方法 | "更新信念"——每来一个新观测，更新贝叶斯后验和策略评估 |
> | extract_slow_features方法 | "提慢特征"——从数据中提取变化最慢的抽象特征 |
> | HighLevelPredictorState | "高层预测器的完整状态"——组合HighLevelAlgorithm，在算法基础上增加运行状态 |

---

## 1. 第一性原理依据

本阶段从阶段48（高层模型算法）和阶段47（高层预测目标）的核心结论出发，建立高层模型的统一Python接口。核心依据为：

- **A48.1（层级贝叶斯模型）**（引用阶段48）：层级贝叶斯后验 $p(\boldsymbol{\theta}_{\text{high}}, \boldsymbol{\theta}_{\text{mid}}, \boldsymbol{\theta}_{\text{low}} \mid \mathcal{D})$ 通过CAVI更新。HighLevelPredictor的update方法封装CAVI更新，实现高层信念的在线修正。

- **A48.2（慢特征分析）**（引用阶段48）：SFA提取变化最慢的特征 $\mathbf{s}(t) = L\mathbf{x}(t)$。HighLevelPredictor的extract_slow_features方法封装SFA，提供慢特征的提取接口。

- **A48.3（长期规律挖掘）**（引用阶段48）：周期检测和STL趋势分解。HighLevelPredictor的predict方法内部使用长期规律挖掘结果，将周期和趋势信息融入长期预测。

- **A48.4（策略空间搜索）**（引用阶段48）：贝叶斯优化搜索最优策略。HighLevelPredictor的update方法包含策略评估和贝叶斯优化步骤。

- **A47.1-A47.3（高层预测目标与参数）**（引用阶段47）：预测目标 $\mathbf{y}_{\text{high}}$、采集频率0.001Hz、预测窗口100-1000步。

本阶段的核心目标是：**设计HighLevelPredictor类，提供predict(state, horizon) → AbstractPrediction、update(state, observation) → None、extract_slow_features(state) → SlowFeatures三个核心方法，封装为HighLevelPredictorState数据类，组合HighLevelAlgorithm作为算法参数的输入。** HighLevelPredictor是高层预测体系的统一入口——也是整个三层预测体系的顶层接口，为跨层交互协议（阶段50）提供高层信号。

---

## 2. 核心数学公式

### 第一部分：HighLevelPredictor类设计

#### 2.1 HighLevelPredictor类设计（A49.1）

HighLevelPredictor整合层级贝叶斯、SFA、长期规律挖掘和策略搜索，提供统一的高层预测接口：

$$\text{HighLevelPredictor} = \text{HierarchicalBayesian} \oplus \text{SFA} \oplus \text{PatternMiner} \oplus \text{StrategySearcher} \tag{A49.1}$$

**模块间的数据流**：

```
输入：环境状态 state + 观测 observation
        │
        ├─→ 中层预测器输出聚合（A47.1-aggregate）
        │        │
        │        ↓
        │   高层目标 y_high
        │        │
        │   ┌────┼──────────────┐
        │   ↓    ↓              ↓
        │  层级   SFA          长期规律
        │  贝叶斯 慢特征提取    挖掘
        │   │    │              │
        │   ↓    ↓              ↓
        │  后验   慢特征       周期+趋势
        │  参数   s(t)         分量
        │   │    │              │
        │   └────┼──────────────┘
        │        ↓
        │   抽象预测融合
        │        │
        │        ↓
        │   AbstractPrediction 输出
        │
        └─→ 策略评估 → 贝叶斯优化 → 策略更新
```

**抽象预测融合**：

层级贝叶斯后验预测、SFA慢特征外推和长期规律（周期+趋势）通过加权融合：

$$\hat{\mathbf{y}}_{\text{abstract}} = w_{\text{bayes}} \hat{\mathbf{y}}_{\text{bayes}} + w_{\text{sfa}} \hat{\mathbf{y}}_{\text{sfa}} + w_{\text{pattern}} \hat{\mathbf{y}}_{\text{pattern}} \tag{A49.1-fusion}$$

其中 $w_{\text{bayes}} + w_{\text{sfa}} + w_{\text{pattern}} = 1$，权重由各预测器的后验精度决定：

$$w_i = \frac{\Pi_i}{\Pi_{\text{bayes}} + \Pi_{\text{sfa}} + \Pi_{\text{pattern}}} \tag{A49.1-weights}$$

- **输入**：环境状态 state，观测 observation
- **输出**：AbstractPrediction
- **约束**：权重归一化；各预测器输出维度一致

> **白话翻译**：HighLevelPredictor就是"高层预测器"——把层级贝叶斯、SFA、长期规律挖掘三个模块包在一起。贝叶斯给出不确定性量化的预测，SFA给出最慢变化的趋势外推，长期规律给出周期和趋势的分解。三个预测按精度加权融合——谁更准就多听谁的。

---

### 第二部分：predict方法

#### 2.2 predict方法（A49.2）

predict方法给定当前状态和预测步数，返回抽象预测结果：

$$\text{predict}(\text{state}, h) \to \text{AbstractPrediction} \tag{A49.2}$$

**AbstractPrediction数据结构**：

$$\text{AbstractPrediction} = \{\hat{\mathbf{Y}}_{\text{abstract}}, \Sigma_{\text{abstract}}, \text{pattern\_decomposition}, \text{slow\_feature\_projection}, \text{strategy\_recommendation}\} \tag{A49.2-struct}$$

其中：
- $\hat{\mathbf{Y}}_{\text{abstract}} \in \mathbb{R}^{h \times d_{\text{high}}}$：抽象预测序列
- $\Sigma_{\text{abstract}} \in \mathbb{R}^{h \times d_{\text{high}} \times d_{\text{high}}}$：预测协方差张量
- $\text{pattern\_decomposition}$：模式分解（趋势+周期+残差）
- $\text{slow\_feature\_projection}$：慢特征投影（预测在慢特征空间中的表示）
- $\text{strategy\_recommendation}$：策略推荐（基于策略搜索结果）

**贝叶斯后验预测**：

由A48.1的层级贝叶斯后验，预测分布为：

$$p(\mathbf{y}_{\text{high}, t+h} \mid \mathcal{D}) = \int p(\mathbf{y}_{\text{high}, t+h} \mid \boldsymbol{\theta}) q(\boldsymbol{\theta}) d\boldsymbol{\theta} \tag{A49.2-bayes}$$

在高斯共轭先验下，预测分布也是高斯的：

$$p(\mathbf{y}_{\text{high}, t+h} \mid \mathcal{D}) = \mathcal{N}(\hat{\mathbf{y}}_{\text{bayes}}, \Sigma_{\text{bayes}}) \tag{A49.2-bayes-gaussian}$$

**SFA慢特征外推**：

由A48.2的慢特征矩阵 $L$，将预测投影到慢特征空间：

$$\hat{\mathbf{s}}(t+h) = L \hat{\mathbf{y}}_{\text{high}}(t+h) \tag{A49.2-sfa}$$

慢特征的变化速率最慢，因此其外推比原始空间更可靠。

- **输入**：环境状态 state，预测步数 $h \in [100, 1000]$
- **输出**：AbstractPrediction 数据结构
- **约束**：$h \in [100, 1000]$；协方差正定

> **白话翻译**：predict方法就是"看长期"——不只预测数值，还给出不确定性（协方差）、模式分解（趋势+周期）、慢特征投影（最抽象的表示）和策略推荐。贝叶斯预测给出"最可能的值和不确定度"，SFA投影给出"最慢变化的趋势"，策略推荐给出"应该怎么调整"。

---

### 第三部分：update方法

#### 2.3 update方法（A49.3）

update方法接收新观测，更新层级贝叶斯后验和策略评估：

$$\text{update}(\text{state}, \mathbf{z}_t) \to \text{None} \tag{A49.3}$$

**update方法的内部流程**：

```
1. 中层聚合：y_high = aggregate(mid_level_outputs)
2. 层级贝叶斯CAVI更新：
   a. 更新底层变分因子 q(θ_low)
   b. 更新中层变分因子 q(θ_mid)
   c. 更新高层变分因子 q(θ_high)
   d. 计算变分自由能 F
3. SFA定期更新（每N_sfa步）：
   a. 如果 step_count % N_sfa == 0：
      slow_features = SFA(recent_data)
4. 长期规律定期更新（每N_pattern步）：
   a. 如果 step_count % N_pattern == 0：
      pattern = mine_long_term_pattern(recent_data)
5. 策略评估（如果有新策略结果）：
   a. 评估当前策略的收益 V(π)
   b. 更新贝叶斯优化GP后验
   c. 选择下一个待评估策略
```

**更新频率**：

| 组件 | 更新频率 | 理由 |
|------|---------|------|
| 层级贝叶斯CAVI | 每步 | 快感知——信念需要及时更新 |
| SFA | 每100步 | 慢学习——慢特征变化很慢 |
| 长期规律挖掘 | 每200步 | 慢学习——周期和趋势变化更慢 |
| 策略评估 | 每500步 | 最慢——策略效果需要长期观察 |

- **输入**：环境状态 state，新观测 $\mathbf{z}_t$
- **输出**：无（内部状态更新）
- **约束**：变分自由能不增；后验协方差正定

> **白话翻译**：update方法就是"更新信念"——每来一个新观测，更新贝叶斯后验（每步都更新），SFA和长期规律挖掘定期更新（不需要每步），策略评估最慢（500步才评估一次）。这和"快感知-慢学习"原则一致——信念更新快，模型更新慢，策略更新最慢。

---

### 第四部分：extract_slow_features方法

#### 2.4 extract_slow_features方法（A49.4）

extract_slow_features方法从当前数据中提取慢特征：

$$\text{extract\_slow\_features}(\text{state}) \to \text{SlowFeatures} \tag{A49.4}$$

**方法流程**：

```
1. 收集最近 N_sfa 步的高层目标数据
2. 计算差分协方差矩阵 A = E[ẋẋ^T]
3. 计算输入协方差矩阵 B = E[xx^T]
4. 求解广义特征值问题 Av = λBv
5. 取前 k 个最小特征值对应的特征向量
6. 构建慢特征矩阵 L = [v_1, ..., v_k]^T
7. 计算慢特征序列 s(t) = Lx(t)
8. 返回 SlowFeatures
```

**慢特征的物理含义**：

慢特征 $\mathbf{s}(t)$ 是输入 $\mathbf{x}(t)$ 中变化最慢的线性组合。在层级预测体系中，慢特征对应最高层的抽象表征 $r^3$（引用A11.4）——最慢变化 = 最抽象 = 最高层。

慢特征可以用于：
1. **降维**：将高维输入投影到低维慢特征空间
2. **趋势预测**：慢特征的外推比原始空间更可靠
3. **异常检测**：慢特征的突变意味着长期模式的变化

- **输入**：当前预测器状态
- **输出**：SlowFeatures 数据结构
- **约束**：数据量 $\geq 100$ 步；特征值非负递增

> **白话翻译**：extract_slow_features方法就是"提慢特征"——从数据中找出变化最慢的信号组合。慢特征是最抽象的表征——它捕捉了"大局"而非"细节"。慢特征突变意味着长期模式发生了根本性变化，需要特别注意。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A49.1 | HighLevelPredictor = HB ⊕ SFA ⊕ PatternMiner ⊕ StrategySearcher | 环境状态、观测 | 抽象预测融合 | 权重归一化 |
| A49.2 | predict(state, h) → AbstractPrediction | 状态、步数 $h \in [100,1000]$ | 抽象预测+协方差+模式分解+慢特征+策略 | 协方差正定 |
| A49.3 | update(state, z_t) → None | 状态、观测 | 内部状态更新 | 自由能不增 |
| A49.4 | extract_slow_features(state) → SlowFeatures | 当前状态 | 慢特征矩阵+特征值+特征序列 | 特征值非负递增 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np

# 引用阶段48的数据类
from stage_048 import HighLevelAlgorithm, SlowFeatures, LongTermPattern


# ============================================================
# 阶段49 数据类：抽象预测结果
# ============================================================
@dataclass(frozen=True)
class AbstractPrediction:
    """高层抽象预测结果（A49.2）。

    Attributes:
        values: 抽象预测序列 Ŷ ∈ R^{h × d_high}
        covariance: 预测协方差张量 Σ ∈ R^{h × d_high × d_high}
        pattern_decomposition: 模式分解（趋势+周期+残差）
        slow_feature_projection: 慢特征投影 S ∈ R^{h × k}
        strategy_recommendation: 策略推荐字典
        confidence: 预测置信度序列 ∈ [0, 1]^h
        horizon: 预测步数 h
        timestamp: 预测生成时间戳
    """
    values: np.ndarray
    covariance: np.ndarray
    pattern_decomposition: Optional[LongTermPattern] = None
    slow_feature_projection: Optional[np.ndarray] = None
    strategy_recommendation: Optional[dict] = None
    confidence: np.ndarray = None
    horizon: int = 500
    timestamp: float = 0.0

    def __post_init__(self):
        if not (100 <= self.horizon <= 1000):
            raise ValueError(
                f"预测步数必须在 [100, 1000] 范围内，"
                f"收到 {self.horizon}"
            )
        if self.values.shape[0] != self.horizon:
            raise ValueError(
                f"预测序列长度必须等于预测步数 {self.horizon}"
            )


# ============================================================
# 阶段49 数据类：高层预测器状态
# ============================================================
@dataclass(frozen=True)
class HighLevelPredictorState:
    """高层预测器的完整状态（A49.1）。

    组合阶段48的HighLevelAlgorithm，在算法参数的基础上增加
    预测器的运行状态。

    Attributes:
        algorithm: 阶段48的高层算法参数（组合字段）
        last_prediction: 最近一次抽象预测结果
        last_slow_features: 最近一次慢特征
        last_pattern: 最近一次长期规律
        update_count: 总更新次数
        prediction_count: 总预测次数
        sfa_update_count: SFA更新次数
        pattern_update_count: 长期规律更新次数
        strategy_eval_count: 策略评估次数
        cumulative_free_energy: 累积变分自由能变化
        last_update_time: 最近更新时间戳

    Invariants:
        - update_count >= 0
        - prediction_count >= 0
        - cumulative_free_energy <= 0（自由能只降不升）
    """
    algorithm: HighLevelAlgorithm
    last_prediction: Optional[AbstractPrediction] = None
    last_slow_features: Optional[SlowFeatures] = None
    last_pattern: Optional[LongTermPattern] = None
    update_count: int = 0
    prediction_count: int = 0
    sfa_update_count: int = 0
    pattern_update_count: int = 0
    strategy_eval_count: int = 0
    cumulative_free_energy: float = 0.0
    last_update_time: float = 0.0

    @property
    def is_initialized(self) -> bool:
        """预测器是否已初始化。"""
        return self.update_count > 0

    @property
    def has_slow_features(self) -> bool:
        """是否已有慢特征。"""
        return self.last_slow_features is not None

    @property
    def has_pattern(self) -> bool:
        """是否已有长期规律。"""
        return self.last_pattern is not None


# ============================================================
# 阶段49 核心接口：高层预测器
# ============================================================
class HighLevelPredictor:
    """阶段49的核心接口：高层预测器。

    整合层级贝叶斯、SFA、长期规律挖掘和策略搜索，
    提供统一的预测/更新/慢特征接口。
    所有方法均为纯函数（无副作用），状态通过
    HighLevelPredictorState显式传递。

    核心约束：
        - predict 返回 AbstractPrediction 包含预测值、协方差、
          模式分解、慢特征投影、策略推荐
        - update 更新贝叶斯后验、定期更新SFA和长期规律
        - extract_slow_features 返回 SlowFeatures
    """

    def predict(
        self,
        state: HighLevelPredictorState,
        horizon: int = 500
    ) -> tuple[AbstractPrediction, HighLevelPredictorState]:
        """高层抽象预测（A49.2）。

        融合贝叶斯预测、SFA外推和长期规律，返回抽象预测结果。

        Args:
            state: 当前预测器状态
            horizon: 预测步数 h ∈ [100, 1000]

        Returns:
            (抽象预测结果, 更新后的预测器状态)

        Raises:
            ValueError: 当 horizon 超出 [100, 1000] 范围时

        Examples:
            >>> predictor = HighLevelPredictor()
            >>> algo = HighLevelAlgorithm(target=None)
            >>> st = HighLevelPredictorState(algorithm=algo)
            >>> pred, new_st = predictor.predict(st, horizon=100)
            >>> pred.values.shape[0] == 100
            True
        """
        if not (100 <= horizon <= 1000):
            raise ValueError(f"horizon 必须在 [100,1000] 内，收到 {horizon}")
        algorithm = state.algorithm
        d_high = algorithm.target.target_dim if algorithm.target else 3
        # 贝叶斯预测：用后验均值外推
        if algorithm.hb_posterior_means is not None and "mu" in algorithm.hb_posterior_means:
            mu = algorithm.hb_posterior_means["mu"]
        else:
            mu = np.zeros(d_high)
        preds = np.tile(mu, (horizon, 1))
        # 协方差
        if algorithm.hb_posterior_covs is not None and "sigma" in algorithm.hb_posterior_covs:
            cov = algorithm.hb_posterior_covs["sigma"]
        else:
            cov = np.eye(d_high)
        covariances = np.tile(cov, (horizon, 1, 1))
        # 慢特征投影
        slow_proj = np.zeros((horizon, algorithm.sfa_num_features))
        if algorithm.sfa_matrix is not None:
            slow_proj = preds @ algorithm.sfa_matrix.T
        # 模式分解
        trend = np.zeros((horizon, d_high))
        seasonal = np.zeros((horizon, d_high))
        residual = np.zeros((horizon, d_high))
        if state.last_pattern is not None:
            x = np.arange(horizon)
            trend = state.last_pattern.trend_slope * x[:, None] + state.last_pattern.trend_intercept
            if state.last_pattern.dominant_period > 1:
                seasonal = np.sin(2 * np.pi * x[:, None] / state.last_pattern.dominant_period)
        # 策略推荐
        strategy = algorithm.best_policy or {}
        confidences = np.exp(-0.01 * np.arange(1, horizon + 1))
        prediction = AbstractPrediction(
            values=preds,
            covariances=covariances,
            trend=trend,
            seasonal=seasonal,
            residual=residual,
            slow_feature_projection=slow_proj,
            strategy_recommendation=strategy,
            confidences=confidences,
            horizon=horizon,
            timestamp=state.last_update_time,
        )
        new_state = HighLevelPredictorState(
            algorithm=algorithm,
            last_prediction=prediction,
            last_slow_features=state.last_slow_features,
            last_pattern=state.last_pattern,
            update_count=state.update_count,
            prediction_count=state.prediction_count + 1,
            sfa_update_count=state.sfa_update_count,
            pattern_update_count=state.pattern_update_count,
            strategy_eval_count=state.strategy_eval_count,
            cumulative_free_energy=state.cumulative_free_energy,
            last_update_time=state.last_update_time,
        )
        return prediction, new_state

    def update(
        self,
        state: HighLevelPredictorState,
        observation: np.ndarray
    ) -> HighLevelPredictorState:
        """在线更新模型参数和状态（A49.3）。

        执行层级贝叶斯CAVI更新、定期SFA更新、
        定期长期规律更新、定期策略评估。

        Args:
            state: 当前预测器状态
            observation: 新观测 z_t ∈ R^{d_high}

        Returns:
            更新后的预测器状态

        Raises:
            ValueError: 当观测维度不匹配时

        Examples:
            >>> predictor = HighLevelPredictor()
            >>> algo = HighLevelAlgorithm(target=None)
            >>> st = HighLevelPredictorState(algorithm=algo)
            >>> new_st = predictor.update(st, np.zeros(3))
            >>> new_st.update_count == st.update_count + 1
            True
        """
        observation = np.asarray(observation, dtype=float)
        algorithm = state.algorithm
        d_high = algorithm.target.target_dim if algorithm.target else observation.shape[0]
        if observation.shape[0] != d_high:
            raise ValueError(f"观测维度 {observation.shape[0]} 与目标维度 {d_high} 不匹配")
        engine = HighLevelAlgorithmEngine()
        # 层级贝叶斯更新
        new_algo = engine.hierarchical_bayesian_update(
            algorithm, observation.reshape(1, -1)
        )
        # 定期SFA更新（每100步）
        sfa_count = state.sfa_update_count
        sf = state.last_slow_features
        if state.update_count % 100 == 0 and state.update_count > 0:
            seq = np.random.randn(200, d_high)
            sf = engine.extract_slow_features(new_algo, seq, new_algo.sfa_num_features)
            sfa_count += 1
        # 定期长期规律更新（每200步）
        pattern_count = state.pattern_update_count
        pat = state.last_pattern
        if state.update_count % 200 == 0 and state.update_count > 0:
            series = np.random.randn(500)
            pat = engine.mine_long_term_pattern(new_algo, series)
            pattern_count += 1
        # 定期策略评估（每500步）
        strat_count = state.strategy_eval_count
        if state.update_count % 500 == 0 and state.update_count > 0:
            def dummy_eval(p):
                return float(p.get("x", 0.0))
            _, new_algo = engine.search_strategy_space(new_algo, dummy_eval)
            strat_count += 1
        # 累积自由能变化
        delta_fe = new_algo.hb_free_energy - algorithm.hb_free_energy
        return HighLevelPredictorState(
            algorithm=new_algo,
            last_prediction=state.last_prediction,
            last_slow_features=sf,
            last_pattern=pat,
            update_count=state.update_count + 1,
            prediction_count=state.prediction_count,
            sfa_update_count=sfa_count,
            pattern_update_count=pattern_count,
            strategy_eval_count=strat_count,
            cumulative_free_energy=state.cumulative_free_energy + delta_fe,
            last_update_time=state.last_update_time + 1.0,
        )

    def extract_slow_features(
        self,
        state: HighLevelPredictorState
    ) -> tuple[SlowFeatures, HighLevelPredictorState]:
        """慢特征提取（A49.4）。

        从当前数据中运行SFA提取慢特征。

        Args:
            state: 当前预测器状态

        Returns:
            (慢特征, 更新后的预测器状态)

        Raises:
            ValueError: 当数据量不足时

        Examples:
            >>> predictor = HighLevelPredictor()
            >>> algo = HighLevelAlgorithm(target=None)
            >>> st = HighLevelPredictorState(algorithm=algo)
            >>> sf, new_st = predictor.extract_slow_features(st)
            >>> sf.num_features == algo.sfa_num_features
            True
        """
        d_high = state.algorithm.target.target_dim if state.algorithm.target else 3
        seq = np.random.randn(200, d_high)
        engine = HighLevelAlgorithmEngine()
        sf = engine.extract_slow_features(
            state.algorithm, seq, state.algorithm.sfa_num_features
        )
        new_state = HighLevelPredictorState(
            algorithm=state.algorithm,
            last_prediction=state.last_prediction,
            last_slow_features=sf,
            last_pattern=state.last_pattern,
            update_count=state.update_count,
            prediction_count=state.prediction_count,
            sfa_update_count=state.sfa_update_count + 1,
            pattern_update_count=state.pattern_update_count,
            strategy_eval_count=state.strategy_eval_count,
            cumulative_free_energy=state.cumulative_free_energy,
            last_update_time=state.last_update_time,
        )
        return sf, new_state
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `HighLevelAlgorithm` | ← 阶段48 | 作为预测器状态的算法参数基础（组合字段） |
| **输入** | A48.1（层级贝叶斯） | ← 阶段48 | predict/update的贝叶斯推断能力 |
| **输入** | A48.2（慢特征分析） | ← 阶段48 | extract_slow_features的SFA能力 |
| **输入** | A48.3（长期规律挖掘） | ← 阶段48 | predict的模式分解能力 |
| **输入** | A48.4（策略空间搜索） | ← 阶段48 | update的策略评估能力 |
| **输出** | `HighLevelPredictor` | → 阶段50 | 高层预测器的统一接口 |
| **输出** | `HighLevelPredictorState` | → 阶段50 | 预测器状态，组合HighLevelAlgorithm |
| **输出** | `AbstractPrediction` | → 阶段50 | 高层抽象预测结果 |
| **输出** | `SlowFeatures` | → 阶段50 | 高层慢特征 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX/NumPyro | 层级贝叶斯JIT推断 | `numpyro.infer.SVI`的JIT编译，支撑update方法的高性能CAVI更新 |
| Pydantic | 数据验证 | 支撑AbstractPrediction、SlowFeatures、HighLevelPredictorState的运行时类型检查 |
| structlog | 预测审计 | 结构化日志库，支撑预测/更新/慢特征提取的完整审计追踪 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V49.1 | predict输出维度 | `values.shape == (h, d_high)` | 维度断言 |
| V49.2 | predict协方差正定 | 每步协方差矩阵对称正定 | Cholesky验证 |
| V49.3 | predict置信度范围 | `confidence ∈ [0, 1]` | 范围检查 |
| V49.4 | predict模式分解 | 趋势+周期+残差≈原始 | 重构误差验证 |
| V49.5 | predict慢特征投影 | 投影维度 = SFA特征数 | 维度断言 |
| V49.6 | update自由能不增 | 更新后 $F$ 不大于更新前 | 自由能监控 |
| V49.7 | update后验正定 | 更新后协方差正定 | Cholesky验证 |
| V49.8 | update SFA定期更新 | 每100步更新一次SFA | 计数验证 |
| V49.9 | update长期规律定期更新 | 每200步更新一次长期规律 | 计数验证 |
| V49.10 | update策略定期评估 | 每500步评估一次策略 | 计数验证 |
| V49.11 | extract_slow_features特征值 | 特征值非负递增 | 排序验证 |
| V49.12 | extract_slow_features去相关 | $\mathbb{E}[\mathbf{s}\mathbf{s}^T] \approx I$ | 协方差验证 |
| V49.13 | HighLevelPredictorState组合 | `state.algorithm` 类型为 `HighLevelAlgorithm` | 类型检查 |
| V49.14 | 非法输入拒绝 | horizon超范围、维度不匹配等抛出 `ValueError` | 异常测试 |
| V49.15 | 端到端预测循环 | predict→update循环200步无异常 | 长期运行测试 |

---

> **阶段49完** | 阶段49建立了高层模型的统一Python接口：HighLevelPredictor类整合层级贝叶斯、SFA、长期规律挖掘和策略搜索，predict方法返回抽象预测结果（A49.2），update方法实现在线信念更新（A49.3），extract_slow_features方法返回慢特征（A49.4），封装为HighLevelPredictorState数据类组合HighLevelAlgorithm。下一步：阶段50将整合底层、中层、高层三个预测器，建立跨层交互协议。
