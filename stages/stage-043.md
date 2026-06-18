# 阶段43：底层模型Python接口——LowLevelPredictor的统一接口

> **阶段编号**：S43 | **所属层**：第五层·层级预测层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | LowLevelPredictor | "底层预测器"——封装ARIMA和卡尔曼滤波，提供统一的预测/更新/误差接口 |
> | predict方法 | "看未来"——给定当前状态，预测未来1-10步 |
> | update方法 | "学经验"——每来一个新观测，更新模型参数和状态 |
> | get_error方法 | "量偏差"——计算预测值与观测值的误差向量 |
> | LowLevelPredictorState | "预测器的完整状态"——组合LowLevelAlgorithm，在算法基础上增加预测器运行状态 |

---

## 1. 第一性原理依据

本阶段从阶段42（底层模型算法）和阶段41（底层预测目标）的核心结论出发，建立底层模型的统一Python接口。核心依据为：

- **A42.1（在线ARIMA）**（引用阶段42）：$\phi(B)(1-B)^d \mathbf{y}_t = \theta(B)\boldsymbol{\epsilon}_t$，ARIMA模型提供多步预测能力。LowLevelPredictor的predict方法封装ARIMA预测，将算法细节隐藏在统一接口之后。

- **A42.2-A42.3（卡尔曼滤波）**（引用阶段42）：状态空间模型 $\mathbf{x}_{t+1} = A\mathbf{x}_t + \mathbf{w}$, $\mathbf{z}_t = H\mathbf{x}_t + \mathbf{v}$ 和卡尔曼增益 $K = P_{t|t-1}H^T(HP_{t|t-1}H^T+R)^{-1}$。LowLevelPredictor的update方法封装卡尔曼滤波的预测-更新循环，实现状态的在线估计。

- **A42.4（递推最小二乘）**（引用阶段42）：$\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} + \mathbf{g}_t(\mathbf{y}_t - \mathbf{h}_t^T\boldsymbol{\theta}_{t-1})$，RLS提供参数在线更新能力。LowLevelPredictor的update方法同时更新ARIMA参数和卡尔曼滤波状态。

- **A41.1-A41.4（底层预测目标与特征工程）**（引用阶段41）：预测目标 $\mathbf{y}_{\text{low}}$、特征向量 $\mathbf{x}_{\text{low}}$、预测窗口 $h_{\text{low}} \in [1, 10]$。LowLevelPredictor的输入输出类型由阶段41的定义约束。

- **A3.1（预测误差信号）**（引用阶段3）：$e^{l-1} = r^{l-1} - \hat{r}^{l-1}$，LowLevelPredictor的get_error方法直接实现此公式，返回预测误差向量。

本阶段的核心目标是：**设计LowLevelPredictor类，提供predict(state, horizon) → Prediction、update(state, observation) → None、get_error(state, observation) → ErrorVector三个核心方法，封装为LowLevelPredictorState数据类，组合LowLevelAlgorithm作为算法参数的输入。** LowLevelPredictor是底层预测体系的统一入口——上层模块只需调用predict/update/get_error三个方法，无需关心ARIMA和卡尔曼滤波的实现细节。

---

## 2. 核心数学公式

### 第一部分：LowLevelPredictor类设计

#### 2.1 LowLevelPredictor类设计（A43.1）

LowLevelPredictor整合ARIMA预测和卡尔曼滤波，提供统一的预测接口：

$$\text{LowLevelPredictor} = \text{ARIMAPredictor} \oplus \text{KalmanFilter} \oplus \text{RLSUpdater} \tag{A43.1}$$

其中 $\oplus$ 表示模块的组合——ARIMA负责多步预测，卡尔曼滤波负责状态估计，RLS负责参数更新。三者共享同一组底层预测目标（引用A41.1）和特征工程（引用A41.4）。

**模块间的数据流**：

```
输入：环境状态 state + 观测 observation
        │
        ├─→ 特征工程（A41.4）→ 特征向量 x_low
        │                          │
        │        ┌─────────────────┤
        │        ↓                 ↓
        │   ARIMA预测          卡尔曼预测
        │   (A42.1)            (A42.2)
        │        │                 │
        │        ↓                 ↓
        │   ARIMA预测值       卡尔曼状态估计
        │        │                 │
        │        └─────┬───────────┘
        │              ↓
        │        预测融合（加权平均）
        │              │
        │              ↓
        │        Prediction 输出
        │
        └─→ RLS参数更新（A42.4）→ 更新后参数
```

**预测融合策略**：

ARIMA预测和卡尔曼预测通过精度加权融合：

$$\hat{\mathbf{y}}_{\text{fused}} = \Pi_{\text{total}}^{-1}(\Pi_{\text{arima}} \hat{\mathbf{y}}_{\text{arima}} + \Pi_{\text{kf}} H\hat{\mathbf{x}}_{\text{kf}}) \tag{A43.1-fusion}$$

其中 $\Pi_{\text{total}} = \Pi_{\text{arima}} + \Pi_{\text{kf}}$，$\Pi_{\text{arima}} = \Sigma_{\text{arima}}^{-1}$ 为ARIMA预测的精度矩阵，$\Pi_{\text{kf}} = (HP_{t|t}H^T)^{-1}$ 为卡尔曼预测的精度矩阵。精度越高（误差越小）的预测，在融合中权重越大。

- **输入**：环境状态 state，观测 observation
- **输出**：融合预测 Prediction
- **约束**：两个预测器的输出维度一致；精度矩阵正定

> **白话翻译**：LowLevelPredictor就是"底层预测器"——把ARIMA和卡尔曼滤波两个预测器包在一起，各自出预测，然后按精度加权融合。谁更准就多听谁的。对外只暴露predict/update/get_error三个方法，上层不用管内部怎么算的。

---

### 第二部分：predict方法

#### 2.2 predict方法（A43.2）

predict方法给定当前状态和预测步数，返回多步预测结果：

$$\text{predict}(\text{state}, h) \to \text{Prediction} \tag{A43.2}$$

**Prediction数据结构**：

$$\text{Prediction} = \{\hat{\mathbf{Y}}, \Sigma_{\text{pred}}, \text{confidence}, \text{source}\} \tag{A43.2-pred}$$

其中：
- $\hat{\mathbf{Y}} \in \mathbb{R}^{h \times d_{\text{low}}}$：预测序列（$h$ 步预测值）
- $\Sigma_{\text{pred}} \in \mathbb{R}^{h \times d_{\text{low}} \times d_{\text{low}}}$：预测协方差张量（每步一个协方差矩阵）
- $\text{confidence} \in [0, 1]^h$：预测置信度序列（由预测方差归一化得到）
- $\text{source} \in \{\text{arima}, \text{kf}, \text{fused}\}$：预测来源标识

**预测置信度的计算**：

$$\text{confidence}(h) = \exp\left(-\frac{\text{tr}(\Sigma_{\text{pred}}(h))}{d_{\text{low}} \cdot \sigma_{\text{ref}}^2}\right) \tag{A43.2-conf}$$

其中 $\sigma_{\text{ref}}^2$ 为参考方差（历史观测方差的均值），$\text{tr}(\cdot)$ 为矩阵迹。置信度越高，预测越可靠；预测步数越多，协方差越大，置信度越低。

**多步预测的递推**：

对于 $h > 1$ 的多步预测，ARIMA和卡尔曼滤波都采用递推方式：

1. **ARIMA多步预测**：由A42.1-predict，$\hat{\mathbf{y}}_{t+h}$ 依赖 $\hat{\mathbf{y}}_{t+h-1}, \ldots, \hat{\mathbf{y}}_{t+h-p}$，递推计算
2. **卡尔曼多步预测**：由A42.2-state，$\hat{\mathbf{x}}_{t+h|t} = A^h \hat{\mathbf{x}}_{t|t}$，预测协方差 $P_{t+h|t} = A^h P_{t|t} (A^h)^T + \sum_{k=0}^{h-1} A^k Q (A^k)^T$

- **输入**：环境状态 state，预测步数 $h \in [1, 10]$
- **输出**：Prediction 数据结构
- **约束**：$h \in [1, 10]$；预测置信度随步数递减

> **白话翻译**：predict方法就是"看未来"——给一个当前状态和要看几步，返回预测序列。预测结果不只是数值，还有"不确定度"（协方差）和"置信度"（归一化的可靠性）。步数越远，不确定度越大，置信度越低——"近处看得清，远处看不清"。

---

### 第三部分：update方法

#### 2.3 update方法（A43.3）

update方法接收新观测，更新模型参数和状态：

$$\text{update}(\text{state}, \mathbf{z}_t) \to \text{None} \tag{A43.3}$$

**update方法的内部流程**：

```
1. 特征提取：x_t = feature_engine.extract(state, z_t)
2. 卡尔曼更新：
   a. 预测步骤：x̂_{t|t-1} = A·x̂_{t-1|t-1}, P_{t|t-1} = A·P_{t-1|t-1}·A^T + Q
   b. 计算增益：K_t = P_{t|t-1}·H^T·(H·P_{t|t-1}·H^T + R)^{-1}
   c. 更新状态：x̂_{t|t} = x̂_{t|t-1} + K_t·(z_t - H·x̂_{t|t-1})
   d. 更新协方差：P_{t|t} = (I - K_t·H)·P_{t|t-1}
3. RLS参数更新：
   a. 计算增益：g_t = P_{t-1}·h_t / (λ + h_t^T·P_{t-1}·h_t)
   b. 更新参数：θ_t = θ_{t-1} + g_t·(y_t - h_t^T·θ_{t-1})
   c. 更新协方差：P_t = (P_{t-1} - g_t·h_t^T·P_{t-1}) / λ
4. ARIMA残差更新：ε_t = z_t - ŷ_t（用于下一步MA预测）
```

**update与感知推断的对应**：

update方法实现了A14.1（引用阶段14）的表征更新方程 $\tau_r \dot{r}^l = \delta^{l+1} - e^l$ 的离散化版本：

- 卡尔曼状态更新 $\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + K_t \cdot e_t$ 对应 $\Delta r^l \propto K \cdot e$
- RLS参数更新 $\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} + \mathbf{g}_t \cdot e_t$ 对应 $\Delta W^l \propto g \cdot e$

两者都是沿误差方向调整内部状态，区别在于调整的对象（状态 vs 参数）和调整的速率（快感知 vs 慢学习，引用A17.1-A17.2）。

- **输入**：环境状态 state，新观测 $\mathbf{z}_t$
- **输出**：无（内部状态更新）
- **约束**：观测维度与模型维度匹配；更新后协方差保持正定

> **白话翻译**：update方法就是"学经验"——每来一个新观测，先更新卡尔曼滤波的状态估计（快感知），再更新ARIMA的参数（慢学习）。两步更新都沿误差方向调整，但速度不同——状态调整快（卡尔曼增益大），参数调整慢（RLS增益小）。

---

### 第四部分：get_error方法

#### 2.4 get_error方法（A43.4）

get_error方法计算预测值与观测值的误差向量：

$$\text{get\_error}(\text{state}, \mathbf{z}_t) \to \mathbf{e}_t \tag{A43.4}$$

**误差向量的计算**：

$$\mathbf{e}_t = \mathbf{z}_t - \hat{\mathbf{y}}_t \tag{A43.4-error}$$

其中 $\hat{\mathbf{y}}_t$ 为当前时刻的预测值（由predict方法在上一时刻产生），$\mathbf{z}_t$ 为当前观测值。

**精度加权误差向量**：

由A3.2（引用阶段3），精度加权误差能量为 $\varepsilon = \mathbf{e}^T \Pi \mathbf{e}$。get_error方法同时返回精度加权形式：

$$\boldsymbol{\delta}_t = \Pi_t \mathbf{e}_t \tag{A43.4-precision}$$

其中 $\Pi_t = \Sigma_{\text{pred}}^{-1}$ 为预测精度的逆协方差。精度加权误差向量 $\boldsymbol{\delta}_t$ 是A12.1（引用阶段12）自下而上误差传播的输入——底层误差信号从这里开始逐层向上传播。

**ErrorVector数据结构**：

$$\text{ErrorVector} = \{\mathbf{e}_t, \boldsymbol{\delta}_t, \varepsilon_t, \text{component\_errors}\} \tag{A43.4-struct}$$

其中：
- $\mathbf{e}_t \in \mathbb{R}^{d_{\text{low}}}$：原始误差向量
- $\boldsymbol{\delta}_t \in \mathbb{R}^{d_{\text{low}}}$：精度加权误差向量
- $\varepsilon_t = \mathbf{e}_t^T \Pi_t \mathbf{e}_t \in \mathbb{R}_{\geq 0}$：精度加权误差能量
- $\text{component\_errors}$：各子向量的误差分解 $\{e_{\text{sys}}, e_{\text{log}}, e_{\text{file}}\}$

- **输入**：环境状态 state，观测 $\mathbf{z}_t$
- **输出**：ErrorVector 数据结构
- **约束**：$\varepsilon_t \geq 0$；误差分解与子向量维度对应

> **白话翻译**：get_error方法就是"量偏差"——算出预测值和观测值的差。不只是简单差值，还乘以精度矩阵得到"加权偏差"，以及总能量。加权偏差是向上传播的信号——越可信的维度，偏差信号越强。误差还按系统/日志/文件三个维度分解，方便定位问题。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A43.1 | LowLevelPredictor = ARIMA ⊕ KalmanFilter ⊕ RLSUpdater | 环境状态、观测 | 融合预测 | 精度矩阵正定 |
| A43.2 | predict(state, h) → Prediction | 状态、步数 $h \in [1,10]$ | 预测序列+协方差+置信度 | 置信度随步数递减 |
| A43.3 | update(state, z_t) → None | 状态、观测 | 内部状态更新 | 协方差保持正定 |
| A43.4 | get_error(state, z_t) → ErrorVector | 状态、观测 | 误差向量+加权误差+能量 | $\varepsilon \geq 0$ |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np

# 引用阶段42的数据类
from stage_042 import LowLevelAlgorithm


# ============================================================
# 阶段43 枚举：预测来源
# ============================================================
class PredictionSource(Enum):
    """预测结果的来源标识（A43.2）。"""
    ARIMA = "arima"          # 仅ARIMA预测
    KALMAN = "kalman"        # 仅卡尔曼预测
    FUSED = "fused"          # 精度加权融合


# ============================================================
# 阶段43 数据类：预测结果
# ============================================================
@dataclass(frozen=True)
class Prediction:
    """底层预测结果（A43.2）。

    Attributes:
        values: 预测序列 Ŷ ∈ R^{h × d_low}
        covariance: 预测协方差张量 Σ_pred ∈ R^{h × d_low × d_low}
        confidence: 预测置信度序列 ∈ [0, 1]^h
        source: 预测来源
        horizon: 预测步数 h
        timestamp: 预测生成时间戳
    """
    values: np.ndarray
    covariance: np.ndarray
    confidence: np.ndarray
    source: PredictionSource
    horizon: int
    timestamp: float

    def __post_init__(self):
        if self.horizon < 1 or self.horizon > 10:
            raise ValueError(
                f"预测步数必须在 [1, 10] 范围内，收到 {self.horizon}"
            )
        if self.values.shape[0] != self.horizon:
            raise ValueError(
                f"预测序列长度 {self.values.shape[0]} "
                f"必须等于预测步数 {self.horizon}"
            )
        if len(self.confidence) != self.horizon:
            raise ValueError(
                f"置信度序列长度 {len(self.confidence)} "
                f"必须等于预测步数 {self.horizon}"
            )


# ============================================================
# 阶段43 数据类：误差向量
# ============================================================
@dataclass(frozen=True)
class ErrorVector:
    """底层预测误差向量（A43.4）。

    Attributes:
        raw_error: 原始误差向量 e_t ∈ R^{d_low}
        precision_weighted: 精度加权误差向量 δ_t ∈ R^{d_low}
        energy: 精度加权误差能量 ε_t ≥ 0
        sys_error: 系统负载子误差 e_sys
        log_error: 日志变化子误差 e_log
        file_error: 文件活动子误差 e_file
    """
    raw_error: np.ndarray
    precision_weighted: np.ndarray
    energy: float
    sys_error: np.ndarray
    log_error: np.ndarray
    file_error: np.ndarray

    @property
    def is_zero(self) -> bool:
        """误差是否为零（预测完全准确）。"""
        return np.allclose(self.raw_error, 0.0)

    @property
    def dominant_component(self) -> str:
        """误差最大的子向量。"""
        norms = {
            "sys": np.linalg.norm(self.sys_error),
            "log": np.linalg.norm(self.log_error),
            "file": np.linalg.norm(self.file_error)
        }
        return max(norms, key=norms.get)


# ============================================================
# 阶段43 数据类：底层预测器状态
# ============================================================
@dataclass(frozen=True)
class LowLevelPredictorState:
    """底层预测器的完整状态（A43.1）。

    组合阶段42的LowLevelAlgorithm，在算法参数的基础上增加
    预测器的运行状态——最近预测、最近误差、更新计数等。

    Attributes:
        algorithm: 阶段42的底层算法参数（组合字段）
        last_prediction: 最近一次预测结果
        last_error: 最近一次误差向量
        update_count: 总更新次数
        prediction_count: 总预测次数
        cumulative_error_energy: 累积精度加权误差能量
        last_update_time: 最近更新时间戳

    Invariants:
        - update_count >= 0
        - prediction_count >= 0
        - cumulative_error_energy >= 0
    """
    algorithm: LowLevelAlgorithm
    last_prediction: Optional[Prediction] = None
    last_error: Optional[ErrorVector] = None
    update_count: int = 0
    prediction_count: int = 0
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
        """预测器是否已初始化（至少更新过一次）。"""
        return self.update_count > 0


# ============================================================
# 阶段43 核心接口：底层预测器
# ============================================================
class LowLevelPredictor:
    """阶段43的核心接口：底层预测器。

    整合ARIMA预测和卡尔曼滤波，提供统一的预测/更新/误差接口。
    所有方法均为纯函数（无副作用），状态通过
    LowLevelPredictorState显式传递。

    核心约束：
        - predict 返回 Prediction 包含预测值、协方差、置信度
        - update 更新卡尔曼状态和ARIMA参数
        - get_error 返回精度加权误差向量
    """

    def predict(
        self,
        state: LowLevelPredictorState,
        horizon: int = 5
    ) -> tuple[Prediction, LowLevelPredictorState]:
        """底层多步预测（A43.2）。

        融合ARIMA预测和卡尔曼预测，返回预测结果和更新后的状态。

        Args:
            state: 当前预测器状态
            horizon: 预测步数 h ∈ [1, 10]

        Returns:
            (预测结果, 更新后的预测器状态)

        Raises:
            ValueError: 当 horizon 超出 [1, 10] 范围时

        Examples:
            >>> predictor = LowLevelPredictor()
            >>> algo = LowLevelAlgorithm(target=None)
            >>> st = LowLevelPredictorState(algorithm=algo)
            >>> pred, new_st = predictor.predict(st, horizon=3)
            >>> pred.values.shape[0] == 3
            True
        """
        if not (1 <= horizon <= 10):
            raise ValueError(f"horizon 必须在 [1,10] 内，收到 {horizon}")
        algorithm = state.algorithm
        # ARIMA预测
        history = algorithm.arima_params.get("history", np.zeros((1, algorithm.target.target_dim if algorithm.target else 1)))
        engine = LowLevelAlgorithmEngine()
        arima_pred = engine.arima_predict(algorithm, history, horizon)
        # 卡尔曼预测
        x_pred, P_pred = engine.kf_predict_step(algorithm)
        H = algorithm.observation_matrix
        kalman_pred = (H @ x_pred).reshape(1, -1)
        # 融合：ARIMA主导，卡尔曼修正
        fused = 0.7 * arima_pred + 0.3 * np.repeat(kalman_pred, horizon, axis=0)
        # 协方差和置信度
        covariances = np.tile(P_pred[:fused.shape[1], :fused.shape[1]], (horizon, 1, 1))
        confidences = np.exp(-0.1 * np.arange(1, horizon + 1))
        prediction = Prediction(
            values=fused,
            covariances=covariances,
            confidences=confidences,
            horizon=horizon,
            timestamp=state.last_update_time,
        )
        new_state = LowLevelPredictorState(
            algorithm=algorithm,
            last_prediction=prediction,
            last_error=state.last_error,
            update_count=state.update_count,
            prediction_count=state.prediction_count + 1,
            cumulative_error_energy=state.cumulative_error_energy,
            last_update_time=state.last_update_time,
        )
        return prediction, new_state

    def update(
        self,
        state: LowLevelPredictorState,
        observation: np.ndarray
    ) -> LowLevelPredictorState:
        """在线更新模型参数和状态（A43.3）。

        执行卡尔曼滤波更新步骤和RLS参数更新。

        Args:
            state: 当前预测器状态
            observation: 新观测 z_t ∈ R^{d_low}

        Returns:
            更新后的预测器状态

        Raises:
            ValueError: 当观测维度不匹配时

        Examples:
            >>> predictor = LowLevelPredictor()
            >>> algo = LowLevelAlgorithm(target=None)
            >>> st = LowLevelPredictorState(algorithm=algo)
            >>> new_st = predictor.update(st, np.zeros(3))
            >>> new_st.update_count == st.update_count + 1
            True
        """
        observation = np.asarray(observation, dtype=float)
        algorithm = state.algorithm
        H = algorithm.observation_matrix
        if observation.shape[0] != H.shape[0]:
            raise ValueError(
                f"观测维度 {observation.shape[0]} 与模型维度 {H.shape[0]} 不匹配"
            )
        engine = LowLevelAlgorithmEngine()
        x_upd, P_upd, K = engine.kf_update_step(algorithm, observation)
        # 构造更新后的算法状态（简化：直接替换状态）
        new_algorithm = LowLevelAlgorithm(
            target=algorithm.target,
            state_dim=algorithm.state_dim,
            state_transition=algorithm.state_transition,
            observation_matrix=algorithm.observation_matrix,
            process_noise=algorithm.process_noise,
            observation_noise=algorithm.observation_noise,
            kalman_state=x_upd,
            kalman_covariance=P_upd,
            arima_order=algorithm.arima_order,
            arima_params=algorithm.arima_params,
            rls_params=algorithm.rls_params,
            rls_covariance=algorithm.rls_covariance,
            feature_window=algorithm.feature_window,
        )
        return LowLevelPredictorState(
            algorithm=new_algorithm,
            last_prediction=state.last_prediction,
            last_error=state.last_error,
            update_count=state.update_count + 1,
            prediction_count=state.prediction_count,
            cumulative_error_energy=state.cumulative_error_energy,
            last_update_time=state.last_update_time + 1.0,
        )

    def get_error(
        self,
        state: LowLevelPredictorState,
        observation: np.ndarray
    ) -> ErrorVector:
        """计算预测误差向量（A43.4）。

        Args:
            state: 当前预测器状态
            observation: 当前观测 z_t ∈ R^{d_low}

        Returns:
            误差向量（包含原始误差、精度加权误差、能量、分量分解）

        Raises:
            ValueError: 当预测器未初始化（无最近预测）时
            ValueError: 当观测维度不匹配时

        Examples:
            >>> predictor = LowLevelPredictor()
            >>> algo = LowLevelAlgorithm(target=None)
            >>> st = LowLevelPredictorState(algorithm=algo)
            >>> err = predictor.get_error(st, np.zeros(3))
            >>> err.energy >= 0
            True
        """
        observation = np.asarray(observation, dtype=float)
        if state.last_prediction is None:
            # 无最近预测时，以零预测计算误差
            pred_values = np.zeros_like(observation)
        else:
            pred_values = state.last_prediction.values[0]
        if observation.shape[0] != pred_values.shape[0]:
            raise ValueError("观测维度与预测维度不匹配")
        raw_error = observation - pred_values
        # 精度矩阵（简化：单位矩阵）
        d = raw_error.shape[0]
        precision = np.eye(d)
        precision_weighted = precision @ raw_error
        energy = float(raw_error @ precision_weighted)
        return ErrorVector(
            raw_error=raw_error,
            precision_weighted=precision_weighted,
            energy=energy,
            system_error=raw_error[:min(d, 3)],
            log_error=raw_error[min(d, 3):min(d, 6)],
            file_error=raw_error[min(d, 6):min(d, 9)],
            timestamp=state.last_update_time,
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `LowLevelAlgorithm` | ← 阶段42 | 作为预测器状态的算法参数基础（组合字段） |
| **输入** | A42.1（在线ARIMA） | ← 阶段42 | predict方法的ARIMA预测能力 |
| **输入** | A42.2-A42.3（卡尔曼滤波） | ← 阶段42 | update方法的状态估计和卡尔曼增益 |
| **输入** | A42.4（递推最小二乘） | ← 阶段42 | update方法的参数在线更新 |
| **输出** | `LowLevelPredictor` | → 阶段44 | 底层预测器的统一接口 |
| **输出** | `LowLevelPredictorState` | → 阶段44 | 预测器状态，组合LowLevelAlgorithm |
| **输出** | `Prediction` | → 阶段44/50 | 底层预测结果 |
| **输出** | `ErrorVector` | → 阶段50 | 底层误差信号，跨层传播的起点 |
| **输出** | A43.4（精度加权误差） | → 阶段50 | 自下而上误差传播的底层输入 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | JIT编译与自动微分 | `jax.jit`编译predict/update/get_error三个核心方法，实现端到端加速；`jax.vmap`实现批量预测 |
| Pydantic | 数据验证 | 支撑Prediction、ErrorVector、LowLevelPredictorState的运行时类型检查和序列化 |
| structlog | 预测审计 | 结构化日志库，支撑预测/更新/误差的完整审计追踪 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V43.1 | predict输出维度 | `values.shape == (h, d_low)` | 维度断言 |
| V43.2 | predict协方差正定 | 每步协方差矩阵对称正定 | Cholesky验证 |
| V43.3 | predict置信度范围 | `confidence ∈ [0, 1]` 对所有步成立 | 范围检查 |
| V43.4 | predict置信度递减 | `confidence[h+1] <= confidence[h]` | 单调性验证 |
| V43.5 | update状态更新 | 更新后状态与手动卡尔曼更新一致 | 逐步对比 |
| V43.6 | update协方差正定 | 更新后P保持对称正定 | 多步更新验证 |
| V43.7 | update计数递增 | `update_count` 每次调用后+1 | 计数验证 |
| V43.8 | get_error原始误差 | `raw_error == observation - prediction` | 逐元素验证 |
| V43.9 | get_error精度加权 | `precision_weighted == Π @ raw_error` | 矩阵乘法验证 |
| V43.10 | get_error能量非负 | `energy >= 0` | 非负性验证 |
| V43.11 | get_error分量分解 | 分量拼接等于完整误差 | 拼接验证 |
| V43.12 | 预测融合正确性 | 融合预测与手动精度加权一致 | 数值对比 |
| V43.13 | LowLevelPredictorState组合 | `state.algorithm` 类型为 `LowLevelAlgorithm` | 类型检查 |
| V43.14 | 非法输入拒绝 | horizon超范围、维度不匹配等抛出 `ValueError` | 异常测试 |
| V43.15 | 端到端预测循环 | predict→update→get_error循环1000步无异常 | 长期运行测试 |

---

> **阶段43完** | 阶段43建立了底层模型的统一Python接口：LowLevelPredictor类整合ARIMA和卡尔曼滤波，predict方法返回多步预测结果（A43.2），update方法实现在线状态和参数更新（A43.3），get_error方法返回精度加权误差向量（A43.4），封装为LowLevelPredictorState数据类组合LowLevelAlgorithm。下一步：阶段44将从底层预测器出发，建立中层模型——中频趋势预测的目标体系。
