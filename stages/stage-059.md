# 阶段59：误差评估模块Python接口——ErrorEvaluator的统一接口

> **阶段编号**：S59 | **所属层**：第六层·误差评估层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | ErrorEvaluator | "误差评估总管"——统一封装标准化误差、异常检测、偏移检测、反馈路由 |
> | evaluate方法 | "打分"——给定预测和观测，计算标准化误差 |
> | detect_anomaly方法 | "查异常"——给定误差历史，检测异常 |
> | detect_shift方法 | "查偏移"——给定误差历史，检测分布偏移 |
> | route_feedback方法 | "发通知"——给定误差、异常、偏移，路由反馈信号 |
> | ErrorEvaluatorState | "误差评估器的完整状态"——组合ErrorModelMapping，在模型映射基础上增加评估器状态 |

---

## 1. 第一性原理依据

本阶段从阶段51-58的核心结论出发，建立误差评估模块的统一Python接口。核心依据为：

- **A51.1-A51.5（标准化误差定义）**（引用阶段51）：逐层误差向量和全局误差。ErrorEvaluator的evaluate方法封装A51.1-A51.5的标准化误差计算。

- **A55.1-A55.4（异常误差检测）**（引用阶段55）：阈值检测、CUSUM、EWMA和组合策略。ErrorEvaluator的detect_anomaly方法封装A55.1-A55.4的异常检测。

- **A56.1-A56.4（分布偏移检测）**（引用阶段56）：KS检验、MMD、在线检测和偏移量化。ErrorEvaluator的detect_shift方法封装A56.1-A56.4的分布偏移检测。

- **A53.1-A53.4（反馈信号生成）**（引用阶段53）：正向/负向反馈、反馈强度和反馈路由。ErrorEvaluator的route_feedback方法封装A53.1-A53.4的反馈路由。

- **A57.1-A57.4（误差反馈到驱力的映射）**（引用阶段57）和**A58.1-A58.4（误差反馈到模型的映射）**（引用阶段58）：驱力映射和模型映射。ErrorEvaluator内部调用驱力映射和模型映射。

本阶段的核心目标是：**设计ErrorEvaluator类，提供evaluate(prediction, observation) → StandardizedError、detect_anomaly(error_history) → AnomalyReport、detect_shift(error_history) → ShiftReport、route_feedback(error, anomaly, shift) → FeedbackRoutes四个核心方法，封装为ErrorEvaluatorState数据类，组合ErrorModelMapping作为模型映射的输入。** ErrorEvaluator是误差评估体系的统一入口——它将阶段51-58的所有功能封装为一个简洁的接口。

> **去重声明**：本阶段仅定义ErrorEvaluator的统一接口，不重新实现阶段51-58的算法。各算法的实现由对应阶段的类提供，ErrorEvaluator仅负责组合调用。

---

## 2. 核心数学公式

### 第一部分：ErrorEvaluator类设计

#### 2.1 ErrorEvaluator类设计（A59.1）

ErrorEvaluator整合标准化误差计算、异常检测、分布偏移检测和反馈路由，提供统一的评估接口：

$$\text{ErrorEvaluator} = \text{StandardizedErrorCalculator} \oplus \text{AnomalyDetectionEngine} \oplus \text{ShiftDetectionEngine} \oplus \text{FeedbackGenerator} \oplus \text{ErrorDriveMapper} \oplus \text{ErrorModelMapper} \tag{A59.1}$$

**模块间的数据流**：

```
输入：prediction + observation
        │
        ↓
   evaluate() → StandardizedError
        │
        ├─→ detect_anomaly() → AnomalyReport
        │        │
        │        ↓
        │   anomaly_score
        │
        ├─→ detect_shift() → ShiftReport
        │        │
        │        ↓
        │   shift_magnitude
        │
        └─→ route_feedback() → FeedbackRoutes
                 │
                 ├─→ drive_engine (A57.1-A57.4)
                 └─→ model_updater (A58.1-A58.4)
```

**ErrorEvaluator的设计原则**：

1. **单一入口**：所有误差评估功能通过ErrorEvaluator访问，用户无需直接操作底层模块
2. **纯函数**：所有方法均为纯函数，状态通过ErrorEvaluatorState显式传递
3. **可组合**：底层模块可独立使用，也可通过ErrorEvaluator组合使用
4. **可审计**：每步评估的结果都有完整的审计追踪

- **输入**：预测值、观测值、评估器状态
- **输出**：标准化误差、异常报告、偏移报告、反馈路由
- **约束**：纯函数；状态显式传递

> **白话翻译**：ErrorEvaluator就是"误差评估总管"——把标准化误差计算、异常检测、偏移检测、反馈路由全部包在一起，提供统一的接口。用户只需要调用四个方法：打分（evaluate）、查异常（detect_anomaly）、查偏移（detect_shift）、发通知（route_feedback），不需要关心底层细节。

---

### 第二部分：evaluate方法

#### 2.2 evaluate方法定义（A59.2）

evaluate方法计算三层预测体系的标准化误差：

$$\text{evaluate}(\text{prediction}, \text{observation}) \to \text{StandardizedError} \tag{A59.2}$$

**方法签名**：

```python
def evaluate(
    self,
    state: ErrorEvaluatorState,
    prediction_low: np.ndarray,
    prediction_mid: np.ndarray,
    prediction_high: np.ndarray,
    observation_low: np.ndarray,
    observation_mid: np.ndarray,
    observation_high: np.ndarray
) -> tuple[StandardizedError, ErrorEvaluatorState]
```

**evaluate方法的内部流程**：

```
1. 计算底层误差：ε_low = ||o_low - ô_low||²（A51.2）
2. 计算中层误差：ε_mid = ||o_mid - ô_mid||²（A51.3）
3. 计算高层误差：ε_high = ||o_high - ô_high||²（A51.4）
4. 获取聚合权重：w = get_weights(scheme, protocol)（A51.5）
5. 计算全局误差：ε_global = Σ w_l · ε_l（A51.5）
6. 构建StandardizedError
7. 更新ErrorEvaluatorState
```

**evaluate方法的异常规范**：

| 异常条件 | 异常类型 | 消息 |
|---------|---------|------|
| 预测-观测维度不匹配 | ValueError | "底层/中层/高层预测与观测维度不匹配" |
| 预测值为NaN/Inf | ValueError | "预测值包含NaN或Inf" |
| 观测值为NaN/Inf | ValueError | "观测值包含NaN或Inf" |

- **输入**：三层预测值、三层观测值、评估器状态
- **输出**：StandardizedError，更新后的ErrorEvaluatorState
- **约束**：预测和观测维度一致；值合法

> **白话翻译**：evaluate方法就是"打分"——给定预测和观测，计算三层的标准化误差和全局误差。内部流程是：先算各层误差，再按权重聚合为全局误差。如果预测和观测维度不匹配或包含非法值，抛出异常。

---

### 第三部分：detect_anomaly方法

#### 2.3 detect_anomaly方法定义（A59.3）

detect_anomaly方法检测误差历史中的异常：

$$\text{detect\_anomaly}(\text{error\_history}) \to \text{AnomalyReport} \tag{A59.3}$$

**方法签名**：

```python
def detect_anomaly(
    self,
    state: ErrorEvaluatorState,
    error_history: np.ndarray
) -> tuple[AnomalyReport, ErrorEvaluatorState]
```

**detect_anomaly方法的内部流程**：

```
1. 估计稳态参数：μ, σ（A55.1-mu, A55.1-sigma）
2. 阈值检测：anomaly_threshold = 1[ε > μ + kσ]（A55.1）
3. 更新CUSUM：S_t = max(0, S_{t-1} + ε - μ_0 - k_σ)（A55.2）
4. CUSUM判定：anomaly_cusum = 1[S_t > h]（A55.2-alarm）
5. 更新EWMA：z_t = λ·ε + (1-λ)·z_{t-1}（A55.3）
6. EWMA判定：anomaly_ewma = 1[|z_t - μ_0| > L·σ_z]（A55.3-alarm）
7. 计算组合异常分数（A55.4）
8. 构建AnomalyReport
9. 更新ErrorEvaluatorState
```

- **输入**：误差历史，评估器状态
- **输出**：AnomalyReport，更新后的ErrorEvaluatorState
- **约束**：误差历史长度 ≥ 10

> **白话翻译**：detect_anomaly方法就是"查异常"——给定误差历史，用阈值检测、CUSUM、EWMA三种方法检测异常，并融合为组合异常分数。如果误差历史太短（<10步），抛出异常。

---

### 第四部分：detect_shift方法

#### 2.4 detect_shift方法定义（A59.4）

detect_shift方法检测误差分布的偏移：

$$\text{detect\_shift}(\text{error\_history}) \to \text{ShiftReport} \tag{A59.4}$$

**方法签名**：

```python
def detect_shift(
    self,
    state: ErrorEvaluatorState,
    error_history: np.ndarray
) -> tuple[ShiftReport, ErrorEvaluatorState]
```

**detect_shift方法的内部流程**：

```
1. 划分参考窗口和观测窗口（A56.3）
2. 计算KS统计量：D = sup|F_obs - F_ref|（A56.1）
3. 计算MMD²（A56.2）
4. 计算偏移程度：shift_magnitude = D / D_threshold（A56.4）
5. 计算偏移方向：shift_direction = sign(μ_obs - μ_ref)（A56.4-direction）
6. 构建ShiftReport
7. 更新ErrorEvaluatorState
```

- **输入**：误差历史，评估器状态
- **输出**：ShiftReport，更新后的ErrorEvaluatorState
- **约束**：误差历史长度 ≥ 参考窗口长度 + 观测窗口长度

> **白话翻译**：detect_shift方法就是"查偏移"——给定误差历史，用KS检验和MMD检测分布偏移，并量化偏移程度和方向。需要足够的误差历史（至少参考窗口+观测窗口的长度）。

---

### 第五部分：route_feedback方法

#### 2.5 route_feedback方法定义（A59.5）

route_feedback方法将误差、异常和偏移信息路由到驱力引擎和模型更新器：

$$\text{route\_feedback}(\text{error}, \text{anomaly}, \text{shift}) \to \text{FeedbackRoutes} \tag{A59.5}$$

**方法签名**：

```python
def route_feedback(
    self,
    state: ErrorEvaluatorState,
    error: StandardizedError,
    anomaly: AnomalyReport,
    shift: ShiftReport
) -> tuple[FeedbackRoutes, ErrorEvaluatorState]
```

**route_feedback方法的内部流程**：

```
1. 计算自由能变化量 ΔF ≈ Δε_global
2. 生成正向/负向反馈（A53.1-A53.2）
3. 计算反馈强度 feedback = tanh(κ·|ΔF|/F_baseline)（A53.3）
4. 路由到驱力引擎：
   a. 异常→安全驱力（A57.1）
   b. 偏移→安全驱力（A57.2）
   c. 误差下降→收益驱力（A57.3）
   d. 误差上升→收益缺口（A57.4）
5. 路由到模型更新器：
   a. 确定更新类型（A58.1/A58.2）
   b. 计算更新方向（A58.3）
   c. 计算自适应增益（A58.4）
6. 构建FeedbackRoutes
7. 更新ErrorEvaluatorState
```

- **输入**：标准化误差、异常报告、偏移报告、评估器状态
- **输出**：FeedbackRoutes，更新后的ErrorEvaluatorState
- **约束**：反馈强度 ∈ [0, 1)

> **白话翻译**：route_feedback方法就是"发通知"——给定误差、异常和偏移信息，生成反馈信号并路由到驱力引擎和模型更新器。驱力引擎收到异常/偏移信号后激活安全驱力或收益驱力，模型更新器收到信号后决定微修正还是结构重构。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A59.1 | ErrorEvaluator = Calculator ⊕ AnomalyEngine ⊕ ShiftEngine ⊕ FeedbackGen ⊕ DriveMapper ⊕ ModelMapper | 预测、观测、状态 | 误差+异常+偏移+反馈 | 纯函数 |
| A59.2 | evaluate(prediction, observation) → StandardizedError | 三层预测/观测 | 标准化误差 | 维度一致 |
| A59.3 | detect_anomaly(error_history) → AnomalyReport | 误差历史 | 异常报告 | 历史 ≥ 10步 |
| A59.4 | detect_shift(error_history) → ShiftReport | 误差历史 | 偏移报告 | 历史 ≥ 窗口长度 |
| A59.5 | route_feedback(error, anomaly, shift) → FeedbackRoutes | 误差+异常+偏移 | 反馈路由 | 反馈强度 ∈ [0,1) |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from enum import Enum
import numpy as np

# 引用阶段51-58的数据类
from stage_051 import StandardizedError, StandardizedErrorCalculator, WeightScheme
from stage_052 import ErrorAccumulation, ErrorAccumulator
from stage_053 import FeedbackSignal, FeedbackRoutes, FeedbackGenerator, FeedbackType
from stage_054 import PrecisionWeightedError, PrecisionWeighter
from stage_055 import AnomalyDetector, AnomalyReport, AnomalyDetectionEngine
from stage_056 import DistributionShiftDetector, ShiftReport, ShiftDetectionEngine
from stage_057 import ErrorDriveMapping, ErrorDriveMapper
from stage_058 import ErrorModelMapping, ErrorModelMapper, UpdateType


# ============================================================
# 阶段59 数据类：误差评估器状态
# ============================================================
@dataclass(frozen=True)
class ErrorEvaluatorState:
    """误差评估器的完整状态。

    组合阶段58的ErrorModelMapping，在模型映射的基础上增加
    评估器的运行状态。

    Attributes:
        model_mapping: 阶段58的误差-模型映射（组合字段）
        error_history: 误差历史序列
        step_count: 评估步数
        last_error: 最近一次标准化误差
        last_anomaly_report: 最近一次异常报告
        last_shift_report: 最近一次偏移报告
        last_feedback_routes: 最近一次反馈路由
        cumulative_anomaly_count: 累积异常次数
        cumulative_shift_count: 累积偏移次数
        cumulative_free_energy: 累积自由能变化

    Invariants:
        - step_count >= 0
        - cumulative_anomaly_count >= 0
        - cumulative_shift_count >= 0
    """
    model_mapping: ErrorModelMapping
    error_history: Optional[np.ndarray] = None
    step_count: int = 0
    last_error: Optional[StandardizedError] = None
    last_anomaly_report: Optional[AnomalyReport] = None
    last_shift_report: Optional[ShiftReport] = None
    last_feedback_routes: Optional[FeedbackRoutes] = None
    cumulative_anomaly_count: int = 0
    cumulative_shift_count: int = 0
    cumulative_free_energy: float = 0.0

    def __post_init__(self):
        if self.step_count < 0:
            raise ValueError(
                f"评估步数必须 >= 0，收到 {self.step_count}"
            )
        if self.cumulative_anomaly_count < 0:
            raise ValueError(
                f"累积异常次数必须 >= 0，收到 {self.cumulative_anomaly_count}"
            )
        if self.cumulative_shift_count < 0:
            raise ValueError(
                f"累积偏移次数必须 >= 0，收到 {self.cumulative_shift_count}"
            )

    @property
    def is_initialized(self) -> bool:
        """评估器是否已初始化。"""
        return self.step_count > 0

    @property
    def anomaly_rate(self) -> float:
        """异常率（异常次数/总步数）。"""
        if self.step_count == 0:
            return 0.0
        return self.cumulative_anomaly_count / self.step_count

    @property
    def shift_rate(self) -> float:
        """偏移率（偏移次数/总步数）。"""
        if self.step_count == 0:
            return 0.0
        return self.cumulative_shift_count / self.step_count


# ============================================================
# 阶段59 核心接口：误差评估器
# ============================================================
class ErrorEvaluator:
    """阶段59的核心接口：误差评估器。

    整合标准化误差计算、异常检测、分布偏移检测和反馈路由，
    提供统一的评估接口。
    所有方法均为纯函数（无副作用），状态通过
    ErrorEvaluatorState显式传递。

    核心约束：
        - evaluate 返回 StandardizedError
        - detect_anomaly 返回 AnomalyReport
        - detect_shift 返回 ShiftReport
        - route_feedback 返回 FeedbackRoutes
    """

    def __init__(self):
        """初始化误差评估器。"""
        self._error_calculator = StandardizedErrorCalculator()
        self._error_accumulator = ErrorAccumulator()
        self._feedback_generator = FeedbackGenerator()
        self._precision_weighter = PrecisionWeighter()
        self._anomaly_engine = AnomalyDetectionEngine()
        self._shift_engine = ShiftDetectionEngine()
        self._drive_mapper = ErrorDriveMapper()
        self._model_mapper = ErrorModelMapper()

    def evaluate(
        self,
        state: ErrorEvaluatorState,
        prediction_low: np.ndarray,
        prediction_mid: np.ndarray,
        prediction_high: np.ndarray,
        observation_low: np.ndarray,
        observation_mid: np.ndarray,
        observation_high: np.ndarray,
        weight_scheme: WeightScheme = WeightScheme.UNIFORM
    ) -> Tuple[StandardizedError, ErrorEvaluatorState]:
        """计算标准化误差（A59.2）。

        Args:
            state: 当前评估器状态
            prediction_low/mid/high: 三层预测值
            observation_low/mid/high: 三层观测值
            weight_scheme: 权重方案

        Returns:
            (标准化误差, 更新后的评估器状态)

        Raises:
            ValueError: 当预测-观测维度不匹配时
            ValueError: 当预测或观测包含NaN/Inf时
        """
        for name, pred, obs in [
            ("low", prediction_low, observation_low),
            ("mid", prediction_mid, observation_mid),
            ("high", prediction_high, observation_high),
        ]:
            if np.shape(pred) != np.shape(obs):
                raise ValueError(
                    f"{name}层预测与观测维度不匹配: "
                    f"pred={np.shape(pred)}, obs={np.shape(obs)}"
                )
            if not np.all(np.isfinite(pred)):
                raise ValueError(f"{name}层预测值包含NaN或Inf")
            if not np.all(np.isfinite(obs)):
                raise ValueError(f"{name}层观测值包含NaN或Inf")

        error_low = float(np.mean((observation_low - prediction_low) ** 2))
        error_mid = float(np.mean((observation_mid - prediction_mid) ** 2))
        error_high = float(np.mean((observation_high - prediction_high) ** 2))

        if weight_scheme == WeightScheme.UNIFORM:
            weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        elif weight_scheme == WeightScheme.LOW_WEIGHTED:
            weights = np.array([0.5, 0.3, 0.2])
        elif weight_scheme == WeightScheme.HIGH_WEIGHTED:
            weights = np.array([0.2, 0.3, 0.5])
        else:
            weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

        global_error = float(
            weights[0] * error_low
            + weights[1] * error_mid
            + weights[2] * error_high
        )

        standardized_error = StandardizedError(
            error_low=error_low,
            error_mid=error_mid,
            error_high=error_high,
            global_error=global_error,
            weights=weights,
        )

        new_history = state.error_history
        if new_history is None:
            new_history = np.array([global_error])
        else:
            new_history = np.concatenate([new_history, [global_error]])

        new_state = self._replace_state(
            state,
            error_history=new_history,
            step_count=state.step_count + 1,
            last_error=standardized_error,
            cumulative_free_energy=state.cumulative_free_energy + global_error,
        )
        return standardized_error, new_state

    def _replace_state(self, state: ErrorEvaluatorState, **kwargs):
        """辅助函数：基于dataclasses.replace更新不可变状态。"""
        from dataclasses import replace
        return replace(state, **kwargs)

    def detect_anomaly(
        self,
        state: ErrorEvaluatorState,
        error_history: Optional[np.ndarray] = None
    ) -> Tuple[AnomalyReport, ErrorEvaluatorState]:
        """检测异常误差（A59.3）。

        Args:
            state: 当前评估器状态
            error_history: 误差历史（None则使用内部历史）

        Returns:
            (异常报告, 更新后的评估器状态)

        Raises:
            ValueError: 当误差历史长度 < 10 时
        """
        history = error_history if error_history is not None else state.error_history
        if history is None or len(history) < 10:
            raise ValueError(f"误差历史长度必须 >= 10，收到 {len(history) if history is not None else 0}")
        report = self._anomaly_engine.detect(history)
        is_anomaly = report.is_anomaly if hasattr(report, 'is_anomaly') else False
        new_state = self._replace_state(
            state,
            last_anomaly_report=report,
            cumulative_anomaly_count=state.cumulative_anomaly_count + (1 if is_anomaly else 0),
        )
        return report, new_state

    def detect_shift(
        self,
        state: ErrorEvaluatorState,
        error_history: Optional[np.ndarray] = None
    ) -> Tuple[ShiftReport, ErrorEvaluatorState]:
        """检测分布偏移（A59.4）。

        Args:
            state: 当前评估器状态
            error_history: 误差历史（None则使用内部历史）

        Returns:
            (偏移报告, 更新后的评估器状态)

        Raises:
            ValueError: 当误差历史长度不足时
        """
        history = error_history if error_history is not None else state.error_history
        if history is None:
            raise ValueError("误差历史不能为空")
        detector = state.model_mapping.shift_detector
        min_len = detector.ref_window_length + detector.obs_window_length
        if len(history) < min_len:
            raise ValueError(
                f"误差历史长度必须 >= 参考窗口+观测窗口长度({min_len})，"
                f"收到 {len(history)}"
            )
        for val in history:
            detector = self._shift_engine.step(detector, float(val))
        report = detector.last_shift_report
        is_shift = report.is_shift if report is not None else False
        new_state = self._replace_state(
            state,
            last_shift_report=report,
            cumulative_shift_count=state.cumulative_shift_count + (1 if is_shift else 0),
        )
        return report, new_state

    def route_feedback(
        self,
        state: ErrorEvaluatorState,
        error: StandardizedError,
        anomaly: AnomalyReport,
        shift: ShiftReport
    ) -> Tuple[FeedbackRoutes, ErrorEvaluatorState]:
        """路由反馈信号（A59.5）。

        Args:
            state: 当前评估器状态
            error: 标准化误差
            anomaly: 异常报告
            shift: 偏移报告

        Returns:
            (反馈路由, 更新后的评估器状态)
        """
        delta_F = error.global_error - (
            state.last_error.global_error if state.last_error is not None else 0.0
        )
        feedback_strength = float(np.tanh(abs(delta_F) / max(error.global_error, 1e-8)))
        routes = self._feedback_generator.generate(
            error=error,
            anomaly=anomaly,
            shift=shift,
            delta_F=delta_F,
            strength=feedback_strength,
        )
        new_state = self._replace_state(
            state,
            last_feedback_routes=routes,
        )
        return routes, new_state

    def full_step(
        self,
        state: ErrorEvaluatorState,
        prediction_low: np.ndarray,
        prediction_mid: np.ndarray,
        prediction_high: np.ndarray,
        observation_low: np.ndarray,
        observation_mid: np.ndarray,
        observation_high: np.ndarray
    ) -> Tuple[StandardizedError, AnomalyReport, ShiftReport, FeedbackRoutes, ErrorEvaluatorState]:
        """执行一步完整的误差评估（A59.1-A59.5）。

        依次调用 evaluate → detect_anomaly → detect_shift → route_feedback。

        Args:
            state: 当前评估器状态
            prediction_low/mid/high: 三层预测值
            observation_low/mid/high: 三层观测值

        Returns:
            (标准化误差, 异常报告, 偏移报告, 反馈路由, 更新后的状态)
        """
        error, state = self.evaluate(
            state,
            prediction_low,
            prediction_mid,
            prediction_high,
            observation_low,
            observation_mid,
            observation_high,
        )
        anomaly, state = self.detect_anomaly(state)
        shift, state = self.detect_shift(state)
        routes, state = self.route_feedback(state, error, anomaly, shift)
        return error, anomaly, shift, routes, state
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `ErrorModelMapping` | ← 阶段58 | 作为评估器状态的模型映射基础（组合字段） |
| **输入** | A51.1-A51.5（标准化误差） | ← 阶段51 | evaluate方法的计算基础 |
| **输入** | A55.1-A55.4（异常检测） | ← 阶段55 | detect_anomaly方法的计算基础 |
| **输入** | A56.1-A56.4（分布偏移检测） | ← 阶段56 | detect_shift方法的计算基础 |
| **输入** | A53.1-A53.4（反馈信号） | ← 阶段53 | route_feedback方法的计算基础 |
| **输入** | A57.1-A57.4（驱力映射） | ← 阶段57 | route_feedback方法的驱力映射 |
| **输入** | A58.1-A58.4（模型映射） | ← 阶段58 | route_feedback方法的模型映射 |
| **输出** | `ErrorEvaluator` | → 阶段60 | 误差评估器的统一接口 |
| **输出** | `ErrorEvaluatorState` | → 阶段60 | 评估器状态 |
| **输出** | A59.1-A59.5（评估器接口定义） | → 工程实现 | 误差评估模块的统一接口规范 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 评估计算JIT编译 | `jax.jit`编译evaluate/detect_anomaly/detect_shift/route_feedback |
| SciPy | KS检验 | `scipy.stats.ks_2samp`支撑detect_shift方法 |
| Pydantic | 数据验证 | 支撑ErrorEvaluatorState的运行时类型检查 |
| structlog | 评估审计 | 结构化日志库，支撑评估器每步操作的完整审计追踪 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V59.1 | evaluate输出正确性 | StandardizedError各字段与手动计算一致 | 数值对比 |
| V59.2 | evaluate维度检查 | 维度不匹配时抛出 `ValueError` | 异常测试 |
| V59.3 | evaluate NaN/Inf检查 | 输入含NaN/Inf时抛出 `ValueError` | 异常测试 |
| V59.4 | detect_anomaly输出正确性 | AnomalyReport各字段与手动计算一致 | 数值对比 |
| V59.5 | detect_anomaly历史长度检查 | 历史<10步时抛出 `ValueError` | 异常测试 |
| V59.6 | detect_shift输出正确性 | ShiftReport各字段与手动计算一致 | 数值对比 |
| V59.7 | detect_shift历史长度检查 | 历史不足时抛出 `ValueError` | 异常测试 |
| V59.8 | route_feedback输出正确性 | FeedbackRoutes各信号与手动计算一致 | 数值对比 |
| V59.9 | full_step一致性 | full_step结果与依次调用各方法一致 | 对比验证 |
| V59.10 | ErrorEvaluatorState组合 | `state.model_mapping` 类型为 `ErrorModelMapping` | 类型检查 |
| V59.11 | 状态更新正确性 | 每步更新后step_count+1 | 计数验证 |
| V59.12 | 异常率计算正确性 | anomaly_rate = cumulative_anomaly_count / step_count | 比率验证 |
| V59.13 | 偏移率计算正确性 | shift_rate = cumulative_shift_count / step_count | 比率验证 |
| V59.14 | 非法状态拒绝 | step_count < 0 等抛出 `ValueError` | 异常测试 |
| V59.15 | 端到端评估循环 | 1000步full_step循环无异常 | 长期运行测试 |

---

> **阶段59完** | 阶段59建立了误差评估模块的统一Python接口：ErrorEvaluator类整合标准化误差计算、异常检测、分布偏移检测和反馈路由，evaluate方法计算标准化误差（A59.2），detect_anomaly方法检测异常误差（A59.3），detect_shift方法检测分布偏移（A59.4），route_feedback方法路由反馈信号（A59.5），封装为ErrorEvaluatorState数据类组合ErrorModelMapping。下一步：阶段60将定义误差评估的验证方案。
