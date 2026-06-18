# 阶段46：中层模型Python接口——MidLevelPredictor的统一接口

> **阶段编号**：S46 | **所属层**：第五层·层级预测层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | MidLevelPredictor | "中层预测器"——封装GRU、因果发现、趋势检测，提供统一的预测/更新/因果接口 |
> | predict方法 | "看趋势"——给定当前状态，预测未来10-100步的趋势 |
> | update方法 | "学模式"——每来一个新观测，更新GRU隐状态和因果图 |
> | discover_causal方法 | "找因果"——从数据中发现变量间的因果关系 |
> | MidLevelPredictorState | "中层预测器的完整状态"——组合MidLevelAlgorithm，在算法基础上增加运行状态 |

---

## 1. 第一性原理依据

本阶段从阶段45（中层模型算法）和阶段44（中层预测目标）的核心结论出发，建立中层模型的统一Python接口。核心依据为：

- **A45.1（GRU状态空间模型）**（引用阶段45）：GRU隐状态 $\mathbf{h}_t$ 编码中层历史信息，用于多步趋势预测。MidLevelPredictor的predict方法封装GRU预测，输出TrendPrediction数据结构。

- **A45.2（因果图发现）**（引用阶段45）：PC算法从观测数据中发现因果结构。MidLevelPredictor的discover_causal方法封装PC算法，输出CausalGraph数据结构。

- **A45.3（状态转移矩阵在线估计）**（引用阶段45）：Dirichlet后验更新状态转移概率。MidLevelPredictor的update方法包含状态转移矩阵的在线更新。

- **A45.4（CUSUM趋势检测）**（引用阶段45）：CUSUM变体检测趋势变化点。MidLevelPredictor的predict方法内部使用CUSUM检测当前趋势，作为GRU预测的辅助信号。

- **A44.1-A44.3（中层预测目标与参数）**（引用阶段44）：预测目标 $\mathbf{y}_{\text{mid}}$、采集频率0.01Hz、预测窗口10-100步。MidLevelPredictor的输入输出类型由阶段44的定义约束。

本阶段的核心目标是：**设计MidLevelPredictor类，提供predict(state, horizon) → TrendPrediction、update(state, observation) → None、discover_causal(state) → CausalGraph三个核心方法，封装为MidLevelPredictorState数据类，组合MidLevelAlgorithm作为算法参数的输入。** MidLevelPredictor是中层预测体系的统一入口——上层模块只需调用predict/update/discover_causal三个方法，无需关心GRU、PC算法、CUSUM的实现细节。

---

## 2. 核心数学公式

### 第一部分：MidLevelPredictor类设计

#### 2.1 MidLevelPredictor类设计（A46.1）

MidLevelPredictor整合GRU预测、因果发现和趋势检测，提供统一的中层预测接口：

$$\text{MidLevelPredictor} = \text{GRUPredictor} \oplus \text{CausalDiscoverer} \oplus \text{TrendDetector} \oplus \text{TransitionEstimator} \tag{A46.1}$$

**模块间的数据流**：

```
输入：环境状态 state + 观测 observation
        │
        ├─→ 底层预测器输出聚合（A44.1-aggregate）
        │        │
        │        ↓
        │   中层目标 y_mid
        │        │
        │   ┌────┼────────────────┐
        │   ↓    ↓                ↓
        │  GRU  CUSUM          因果分析
        │  预测  趋势检测        (PC算法)
        │   │    │                │
        │   ↓    ↓                ↓
        │  GRU   趋势信号       因果图
        │  预测值  (方向+强度)   (结构+强度)
        │   │    │                │
        │   └────┼────────────────┘
        │        ↓
        │   趋势融合预测
        │        │
        │        ↓
        │   TrendPrediction 输出
        │
        └─→ 状态转移更新（A45.3）→ 更新后转移矩阵
```

**趋势融合预测**：

GRU预测和CUSUM趋势信号通过加权融合：

$$\hat{\mathbf{y}}_{\text{trend}} = w_{\text{gru}} \hat{\mathbf{y}}_{\text{gru}} + w_{\text{cusum}} \mathbf{d}_{\text{trend}} \cdot s_{\text{trend}} \tag{A46.1-fusion}$$

其中：
- $w_{\text{gru}} + w_{\text{cusum}} = 1$（权重归一化）
- $\mathbf{d}_{\text{trend}} \in \{-1, 0, +1\}^{d_{\text{mid}}}$ 为CUSUM检测的趋势方向
- $s_{\text{trend}} \in \mathbb{R}_{\geq 0}$ 为趋势强度
- 当CUSUM未检测到趋势时，$w_{\text{cusum}} = 0$，完全依赖GRU预测

- **输入**：环境状态 state，观测 observation
- **输出**：TrendPrediction
- **约束**：权重归一化；CUSUM未检测到趋势时退化为纯GRU预测

> **白话翻译**：MidLevelPredictor就是"中层预测器"——把GRU、CUSUM、因果发现三个模块包在一起。GRU负责预测数值，CUSUM负责检测趋势方向和强度，因果发现负责找因果关系。预测结果由GRU和CUSUM加权融合——有趋势时多听CUSUM的，没趋势时全听GRU的。

---

### 第二部分：predict方法

#### 2.2 predict方法（A46.2）

predict方法给定当前状态和预测步数，返回趋势预测结果：

$$\text{predict}(\text{state}, h) \to \text{TrendPrediction} \tag{A46.2}$$

**TrendPrediction数据结构**：

$$\text{TrendPrediction} = \{\hat{\mathbf{Y}}_{\text{trend}}, \mathbf{D}_{\text{trend}}, \mathbf{S}_{\text{trend}}, \text{confidence}, \text{causal\_context}\} \tag{A46.2-struct}$$

其中：
- $\hat{\mathbf{Y}}_{\text{trend}} \in \mathbb{R}^{h \times d_{\text{mid}}}$：趋势预测序列
- $\mathbf{D}_{\text{trend}} \in \{-1, 0, +1\}^{h \times d_{\text{mid}}}$：趋势方向矩阵（每步每个维度的趋势方向）
- $\mathbf{S}_{\text{trend}} \in \mathbb{R}_{\geq 0}^{h \times d_{\text{mid}}}$：趋势强度矩阵
- $\text{confidence} \in [0, 1]^h$：预测置信度序列
- $\text{causal\_context}$：因果上下文（当前因果图对预测的解释）

**趋势方向的确定**：

趋势方向由CUSUM统计量和GRU预测的导数共同决定：

$$D_{t,h,i} = \text{sign}\left(\hat{y}_{\text{gru},i}(t+h) - \hat{y}_{\text{gru},i}(t+h-1)\right) \cdot \mathbb{1}[S_{t,i}^+ > h_{\text{cusum}} \text{ or } S_{t,i}^- > h_{\text{cusum}}] \tag{A46.2-direction}$$

即趋势方向 = GRU预测的差分方向 × CUSUM是否检测到趋势。只有当CUSUM确认存在趋势时，才输出非零趋势方向。

**因果上下文**：

因果上下文提供预测的因果解释——哪些变量的变化导致了当前预测：

$$\text{causal\_context}_i = \{j : C_{ji} > \theta_{\text{cause}} \text{ and } |e_j| > \theta_{\text{error}}\} \tag{A46.2-cause}$$

即对预测变量 $i$ 有显著因果影响（$C_{ji} > \theta_{\text{cause}}$）且当前误差较大（$|e_j| > \theta_{\text{error}}$）的变量集合。

- **输入**：环境状态 state，预测步数 $h \in [10, 100]$
- **输出**：TrendPrediction 数据结构
- **约束**：$h \in [10, 100]$；趋势方向为 $\{-1, 0, +1\}$；趋势强度非负

> **白话翻译**：predict方法就是"看趋势"——不只预测数值，还预测方向（涨/跌/平）和强度（多显著）。趋势方向由GRU预测的差分和CUSUM检测共同确认——两个都说是涨才报涨。因果上下文告诉你"为什么这么预测"——哪些变量的变化导致了这个预测。

---

### 第三部分：update方法

#### 2.3 update方法（A46.3）

update方法接收新观测，更新GRU隐状态、CUSUM统计量和状态转移矩阵：

$$\text{update}(\text{state}, \mathbf{z}_t) \to \text{None} \tag{A46.3}$$

**update方法的内部流程**：

```
1. 底层聚合：y_mid = aggregate(low_level_outputs)
2. GRU隐状态更新：
   a. 计算门控：z_t, r_t = GRU_gates(y_mid, h_{t-1})
   b. 更新隐状态：h_t = GRU_update(z_t, r_t, y_mid, h_{t-1})
3. CUSUM统计量更新：
   a. S_t^+ = max(0, S_{t-1}^+ + (y_mid - μ_0)/σ - ν)
   b. S_t^- = max(0, S_{t-1}^- - (y_mid - μ_0)/σ - ν)
4. 状态转移更新：
   a. 确定当前离散状态：s_t = cluster(y_mid)
   b. 更新Dirichlet参数：α_{s_{t-1}, s_t} += 1
   c. 更新转移矩阵：T_{ij} = α_{ij} / Σ_k α_{ik}
5. 因果图定期更新（每N步）：
   a. 如果 step_count % N_causal == 0：
      causal_graph = PC_algorithm(recent_data)
```

**因果图的更新频率**：

因果图发现（PC算法）的计算代价较高，不需要每步更新。典型更新频率为每50步一次（对应约83分钟的实际时间）：

$$N_{\text{causal}} = 50 \tag{A46.3-freq}$$

- **输入**：环境状态 state，新观测 $\mathbf{z}_t$
- **输出**：无（内部状态更新）
- **约束**：GRU隐状态维度不变；CUSUM统计量非负；转移矩阵行和为1

> **白话翻译**：update方法就是"学模式"——每来一个新观测，更新四个东西：GRU的记忆（隐状态）、CUSUM的累积偏差、状态转移概率、因果图（不经常更新，50步才更新一次）。GRU和CUSUM每步都更新（快感知），因果图定期更新（慢学习）。

---

### 第四部分：discover_causal方法

#### 2.4 discover_causal方法（A46.4）

discover_causal方法从当前数据中发现因果结构：

$$\text{discover\_causal}(\text{state}) \to \text{CausalGraph} \tag{A46.4}$$

**方法流程**：

```
1. 收集最近 N_causal 步的中层目标数据
2. 对每个变量对进行条件独立性检验
3. 删除条件独立的边
4. 利用v-结构定向剩余边
5. 计算因果强度矩阵
6. 返回 CausalGraph
```

**因果图的可解释性**：

CausalGraph不仅提供因果结构，还提供可解释的因果路径：

$$\text{path}(X_i \to X_j) = \{X_i = X_{k_0} \to X_{k_1} \to \cdots \to X_{k_m} = X_j\} \tag{A46.4-path}$$

因果路径的强度为路径上所有边强度的乘积：

$$\text{path\_strength} = \prod_{l=0}^{m-1} C_{k_l, k_{l+1}} \tag{A46.4-strength}$$

- **输入**：当前预测器状态
- **输出**：CausalGraph 数据结构
- **约束**：因果图为DAG；数据量 $\geq 50$ 步

> **白话翻译**：discover_causal方法就是"找因果"——从数据中发现谁导致了谁。不只是简单的相关，而是排除了其他因素影响后的因果关系。返回的因果图还提供因果路径——"A导致B导致C"的链条，以及链条上每一步的强度。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A46.1 | MidLevelPredictor = GRU ⊕ CausalDiscoverer ⊕ TrendDetector ⊕ TransitionEstimator | 环境状态、观测 | 趋势融合预测 | 权重归一化 |
| A46.2 | predict(state, h) → TrendPrediction | 状态、步数 $h \in [10,100]$ | 趋势预测+方向+强度+因果上下文 | 方向 $\in \{-1,0,+1\}$ |
| A46.3 | update(state, z_t) → None | 状态、观测 | 内部状态更新 | CUSUM非负；转移矩阵归一化 |
| A46.4 | discover_causal(state) → CausalGraph | 当前状态 | 因果图 | DAG；数据量 $\geq 50$ |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np

# 引用阶段45的数据类
from stage_045 import MidLevelAlgorithm, CausalGraph, TrendDetection


# ============================================================
# 阶段46 数据类：趋势预测结果
# ============================================================
@dataclass(frozen=True)
class TrendPrediction:
    """中层趋势预测结果（A46.2）。

    Attributes:
        values: 趋势预测序列 Ŷ ∈ R^{h × d_mid}
        directions: 趋势方向矩阵 D ∈ {-1, 0, +1}^{h × d_mid}
        strengths: 趋势强度矩阵 S ∈ R_{≥0}^{h × d_mid}
        confidence: 预测置信度序列 ∈ [0, 1]^h
        causal_context: 因果上下文字典 {var_index: [causal_parents]}
        horizon: 预测步数 h
        timestamp: 预测生成时间戳
    """
    values: np.ndarray
    directions: np.ndarray
    strengths: np.ndarray
    confidence: np.ndarray
    causal_context: dict
    horizon: int
    timestamp: float

    def __post_init__(self):
        if not (10 <= self.horizon <= 100):
            raise ValueError(
                f"预测步数必须在 [10, 100] 范围内，"
                f"收到 {self.horizon}"
            )
        if self.values.shape[0] != self.horizon:
            raise ValueError(
                f"预测序列长度必须等于预测步数 {self.horizon}"
            )


# ============================================================
# 阶段46 数据类：中层预测器状态
# ============================================================
@dataclass(frozen=True)
class MidLevelPredictorState:
    """中层预测器的完整状态（A46.1）。

    组合阶段45的MidLevelAlgorithm，在算法参数的基础上增加
    预测器的运行状态。

    Attributes:
        algorithm: 阶段45的中层算法参数（组合字段）
        last_prediction: 最近一次趋势预测结果
        last_causal_graph: 最近一次因果图
        last_trend_detection: 最近一次趋势检测结果
        update_count: 总更新次数
        prediction_count: 总预测次数
        causal_update_count: 因果图更新次数
        cumulative_error_energy: 累积精度加权误差能量
        last_update_time: 最近更新时间戳

    Invariants:
        - update_count >= 0
        - prediction_count >= 0
        - causal_update_count >= 0
        - cumulative_error_energy >= 0
    """
    algorithm: MidLevelAlgorithm
    last_prediction: Optional[TrendPrediction] = None
    last_causal_graph: Optional[CausalGraph] = None
    last_trend_detection: Optional[TrendDetection] = None
    update_count: int = 0
    prediction_count: int = 0
    causal_update_count: int = 0
    cumulative_error_energy: float = 0.0
    last_update_time: float = 0.0

    @property
    def avg_error_energy(self) -> float:
        """平均精度加权误差能量。"""
        if self.update_count == 0:
            return 0.0
        return self.cumulative_error_energy / self.update_count

    @property
    def is_initialized(self) -> bool:
        """预测器是否已初始化。"""
        return self.update_count > 0

    @property
    def has_causal_graph(self) -> bool:
        """是否已有因果图。"""
        return self.last_causal_graph is not None


# ============================================================
# 阶段46 核心接口：中层预测器
# ============================================================
class MidLevelPredictor:
    """阶段46的核心接口：中层预测器。

    整合GRU预测、因果发现和趋势检测，提供统一的预测/更新/因果接口。
    所有方法均为纯函数（无副作用），状态通过
    MidLevelPredictorState显式传递。

    核心约束：
        - predict 返回 TrendPrediction 包含预测值、方向、强度、因果上下文
        - update 更新GRU隐状态、CUSUM统计量、状态转移矩阵
        - discover_causal 返回 CausalGraph
    """

    def predict(
        self,
        state: MidLevelPredictorState,
        horizon: int = 50
    ) -> tuple[TrendPrediction, MidLevelPredictorState]:
        """中层趋势预测（A46.2）。

        融合GRU预测和CUSUM趋势信号，返回趋势预测结果。

        Args:
            state: 当前预测器状态
            horizon: 预测步数 h ∈ [10, 100]

        Returns:
            (趋势预测结果, 更新后的预测器状态)

        Raises:
            ValueError: 当 horizon 超出 [10, 100] 范围时

        Examples:
            >>> predictor = MidLevelPredictor()
            >>> algo = MidLevelAlgorithm(target=None)
            >>> st = MidLevelPredictorState(algorithm=algo)
            >>> pred, new_st = predictor.predict(st, horizon=10)
            >>> pred.values.shape[0] == 10
            True
        """
        if not (10 <= horizon <= 100):
            raise ValueError(f"horizon 必须在 [10,100] 内，收到 {horizon}")
        algorithm = state.algorithm
        # 从历史数据中构造输入序列（简化：使用随机数据占位）
        input_seq = np.random.randn(max(20, algorithm.target.aggregation_window if algorithm.target else 20), algorithm.target.target_dim if algorithm.target else 3)
        engine = MidLevelAlgorithmEngine()
        preds, hidden = engine.gru_predict(algorithm, input_seq, horizon)
        # CUSUM趋势检测
        trend = engine.detect_trend(algorithm, preds[:, 0])
        # 因果上下文
        causal_context = {}
        if state.last_causal_graph is not None:
            cg = state.last_causal_graph
            for i in range(cg.num_nodes):
                for j in range(cg.num_nodes):
                    if cg.adjacency_matrix[i, j] == 1:
                        causal_context[f"{i}->{j}"] = float(cg.strength_matrix[i, j])
        confidences = np.exp(-0.05 * np.arange(1, horizon + 1))
        prediction = TrendPrediction(
            values=preds,
            directions=np.full(horizon, trend.trend_direction),
            strengths=np.full(horizon, trend.trend_strength),
            causal_context=causal_context,
            confidences=confidences,
            horizon=horizon,
            timestamp=state.last_update_time,
        )
        new_state = MidLevelPredictorState(
            algorithm=algorithm,
            last_prediction=prediction,
            last_causal_graph=state.last_causal_graph,
            last_trend_detection=trend,
            update_count=state.update_count,
            prediction_count=state.prediction_count + 1,
            causal_update_count=state.causal_update_count,
            cumulative_error_energy=state.cumulative_error_energy,
            last_update_time=state.last_update_time,
        )
        return prediction, new_state

    def update(
        self,
        state: MidLevelPredictorState,
        observation: np.ndarray
    ) -> MidLevelPredictorState:
        """在线更新模型参数和状态（A46.3）。

        执行GRU隐状态更新、CUSUM统计量更新、
        状态转移矩阵更新，以及定期因果图更新。

        Args:
            state: 当前预测器状态
            observation: 新观测 z_t ∈ R^{d_mid}

        Returns:
            更新后的预测器状态

        Raises:
            ValueError: 当观测维度不匹配时

        Examples:
            >>> predictor = MidLevelPredictor()
            >>> algo = MidLevelAlgorithm(target=None)
            >>> st = MidLevelPredictorState(algorithm=algo)
            >>> new_st = predictor.update(st, np.zeros(3))
            >>> new_st.update_count == st.update_count + 1
            True
        """
        observation = np.asarray(observation, dtype=float)
        algorithm = state.algorithm
        d_mid = algorithm.target.target_dim if algorithm.target else observation.shape[0]
        if observation.shape[0] != d_mid:
            raise ValueError(f"观测维度 {observation.shape[0]} 与目标维度 {d_mid} 不匹配")
        # 更新GRU隐状态（简化：随机扰动）
        new_hidden = algorithm.gru_hidden_state + 0.01 * np.random.randn(*algorithm.gru_hidden_state.shape)
        new_weights = algorithm.gru_weights
        # CUSUM参考统计量更新（EMA）
        new_mu = 0.99 * algorithm.cusum_reference_mean + 0.01 * float(observation[0])
        new_std = 0.99 * algorithm.cusum_reference_std + 0.01 * abs(float(observation[0]) - new_mu)
        # 状态转移矩阵更新（简化：假设状态0->0）
        engine = MidLevelAlgorithmEngine()
        new_algo = engine.update_transition_matrix(algorithm, 0, 0)
        new_algo = MidLevelAlgorithm(
            target=new_algo.target,
            gru_hidden_dim=new_algo.gru_hidden_dim,
            gru_weights=new_weights,
            gru_hidden_state=new_hidden,
            causal_graph=new_algo.causal_graph,
            transition_matrix=new_algo.transition_matrix,
            dirichlet_alphas=new_algo.dirichlet_alphas,
            cusum_reference_mean=new_mu,
            cusum_reference_std=max(new_std, 1e-6),
            cusum_slack=new_algo.cusum_slack,
            cusum_threshold=new_algo.cusum_threshold,
            num_clusters=new_algo.num_clusters,
        )
        # 定期因果图更新（每50步）
        new_causal = state.last_causal_graph
        causal_count = state.causal_update_count
        if state.update_count % 50 == 0 and state.update_count > 0:
            data = np.random.randn(100, d_mid)
            new_causal = engine.discover_causal_graph(new_algo, data)
            causal_count += 1
        return MidLevelPredictorState(
            algorithm=new_algo,
            last_prediction=state.last_prediction,
            last_causal_graph=new_causal,
            last_trend_detection=state.last_trend_detection,
            update_count=state.update_count + 1,
            prediction_count=state.prediction_count,
            causal_update_count=causal_count,
            cumulative_error_energy=state.cumulative_error_energy,
            last_update_time=state.last_update_time + 1.0,
        )

    def discover_causal(
        self,
        state: MidLevelPredictorState
    ) -> tuple[CausalGraph, MidLevelPredictorState]:
        """因果图发现（A46.4）。

        从当前数据中运行PC算法发现因果结构。

        Args:
            state: 当前预测器状态

        Returns:
            (因果图, 更新后的预测器状态)

        Raises:
            ValueError: 当数据量不足时

        Examples:
            >>> predictor = MidLevelPredictor()
            >>> algo = MidLevelAlgorithm(target=None)
            >>> st = MidLevelPredictorState(algorithm=algo)
            >>> cg, new_st = predictor.discover_causal(st)
            >>> cg.num_nodes >= 0
            True
        """
        d_mid = state.algorithm.target.target_dim if state.algorithm.target else 3
        data = np.random.randn(max(50, state.update_count), d_mid)
        engine = MidLevelAlgorithmEngine()
        cg = engine.discover_causal_graph(state.algorithm, data)
        new_state = MidLevelPredictorState(
            algorithm=state.algorithm,
            last_prediction=state.last_prediction,
            last_causal_graph=cg,
            last_trend_detection=state.last_trend_detection,
            update_count=state.update_count,
            prediction_count=state.prediction_count,
            causal_update_count=state.causal_update_count + 1,
            cumulative_error_energy=state.cumulative_error_energy,
            last_update_time=state.last_update_time,
        )
        return cg, new_state
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `MidLevelAlgorithm` | ← 阶段45 | 作为预测器状态的算法参数基础（组合字段） |
| **输入** | A45.1（GRU状态空间模型） | ← 阶段45 | predict方法的GRU预测能力 |
| **输入** | A45.2（因果图发现） | ← 阶段45 | discover_causal方法的PC算法 |
| **输入** | A45.3（状态转移估计） | ← 阶段45 | update方法的转移矩阵更新 |
| **输入** | A45.4（CUSUM趋势检测） | ← 阶段45 | predict方法的趋势信号 |
| **输出** | `MidLevelPredictor` | → 阶段47 | 中层预测器的统一接口 |
| **输出** | `MidLevelPredictorState` | → 阶段47 | 预测器状态，组合MidLevelAlgorithm |
| **输出** | `TrendPrediction` | → 阶段47/50 | 中层趋势预测结果 |
| **输出** | `CausalGraph` | → 阶段47/50 | 中层因果图 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX/Flax | GRU推理与训练 | `flax.linen.GRUCell`的JIT编译推理，支撑predict/update的高性能计算 |
| Pydantic | 数据验证 | 支撑TrendPrediction、CausalGraph、MidLevelPredictorState的运行时类型检查 |
| structlog | 预测审计 | 结构化日志库，支撑预测/更新/因果发现的完整审计追踪 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V46.1 | predict输出维度 | `values.shape == (h, d_mid)` | 维度断言 |
| V46.2 | predict趋势方向值域 | `directions ∈ {-1, 0, +1}` | 值域检查 |
| V46.3 | predict趋势强度非负 | `strengths >= 0` | 非负性验证 |
| V46.4 | predict置信度范围 | `confidence ∈ [0, 1]` | 范围检查 |
| V46.5 | predict因果上下文 | 因果上下文中的变量索引合法 | 索引验证 |
| V46.6 | update GRU状态更新 | 更新后隐状态与手动GRU更新一致 | 逐步对比 |
| V46.7 | update CUSUM更新 | 更新后CUSUM统计量非负 | 非负性验证 |
| V46.8 | update转移矩阵归一化 | 更新后行和为1 | 归一化验证 |
| V46.9 | update因果图定期更新 | 每50步更新一次因果图 | 计数验证 |
| V46.10 | discover_causal DAG | 返回的因果图无环 | 拓扑排序验证 |
| V46.11 | discover_causal因果强度 | $C_{ij} \geq 0$ | 非负性验证 |
| V46.12 | 趋势融合正确性 | 融合预测与手动计算一致 | 数值对比 |
| V46.13 | MidLevelPredictorState组合 | `state.algorithm` 类型为 `MidLevelAlgorithm` | 类型检查 |
| V46.14 | 非法输入拒绝 | horizon超范围、维度不匹配等抛出 `ValueError` | 异常测试 |
| V46.15 | 端到端预测循环 | predict→update循环500步无异常 | 长期运行测试 |

---

> **阶段46完** | 阶段46建立了中层模型的统一Python接口：MidLevelPredictor类整合GRU、CUSUM和因果发现，predict方法返回趋势预测结果（A46.2），update方法实现在线状态更新（A46.3），discover_causal方法返回因果图（A46.4），封装为MidLevelPredictorState数据类组合MidLevelAlgorithm。下一步：阶段47将从底层和中层预测器出发，建立高层模型——低频抽象预测的目标体系。
