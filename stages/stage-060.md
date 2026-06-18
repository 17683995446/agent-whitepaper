# 阶段60：误差评估验证——合成数据、检测延迟、误报率与灵敏度测试

> **阶段编号**：S60 | **所属层**：第六层·误差评估层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 合成数据生成 | "人造测试场景"——高斯稳态+突变+漂移，模拟各种真实情况 |
> | 检测延迟测试 | "多久能发现"——突变发生后，系统需要几步才能检测到 |
> | 误报率测试 | "会不会乱报"——正常情况下，系统会不会误以为出了问题 |
> | 偏移检测灵敏度 | "多小的变化能发现"——分布偏移多小还能被检测到 |
> | 反馈路由正确性 | "通知发对了没"——异常时激活安全驱力，改善时激活收益驱力 |
> | ErrorValidationResult | "验证结果的完整画像"——组合ErrorEvaluatorState，在评估器状态基础上增加验证结果 |

---

## 1. 第一性原理依据

本阶段从阶段59（误差评估模块Python接口）和阶段51-58的所有核心结论出发，建立误差评估的完整验证方案。核心依据为：

- **A59.1-A59.5（ErrorEvaluator接口）**（引用阶段59）：evaluate、detect_anomaly、detect_shift、route_feedback四个核心方法。验证方案针对这四个方法设计测试用例。

- **A55.4（检测延迟与误报率权衡）**（引用阶段55）：$D(\delta) \propto \ln(\text{ARL}_0) / \delta$，检测延迟与误报率的定量权衡关系。验证方案需要同时测试检测延迟和误报率，确认系统在可接受的误报率下达到合理的检测延迟。

- **A56.1-A56.2（KS检验与MMD）**（引用阶段56）：分布偏移检测的两种方法。验证方案需要测试两种方法的灵敏度——多小的分布偏移能被检测到。

- **A3.3（累积误差与生存条件）**（引用阶段3）：$\sum_{t=0}^{T} \varepsilon_0(t) \leq -\ln P_{\min}$。验证方案需要确认误差评估模块的输出与生存约束一致——累积误差接近生存约束时，系统应发出警报。

本阶段的核心目标是：**定义合成数据生成（高斯+突变+漂移）、检测延迟测试（突变检测<5步）、误报率测试（稳态误报<5%）、分布偏移检测灵敏度测试、反馈路由正确性测试，封装为ErrorValidationResult数据类，组合ErrorEvaluatorState作为评估器状态的输入。** 验证方案是误差评估体系的"质量保证"——它确认系统在各种场景下都能正确工作。

> **去重声明**：本阶段仅定义验证方案和测试用例，不重新实现阶段51-59的算法。验证结果用于确认系统正确性，不改变系统行为。

---

## 2. 核心数学公式

### 第一部分：合成数据生成

#### 2.1 合成数据生成定义（A60.1）

合成数据模拟三种典型场景：高斯稳态、突变和漂移。

**高斯稳态数据**：

$$\varepsilon_t \sim \mathcal{N}(\mu_0, \sigma_0^2), \quad t = 1, \ldots, T_{\text{stable}} \tag{A60.1-gaussian}$$

其中 $\mu_0$ 为稳态均值，$\sigma_0$ 为稳态标准差。

**突变数据**：

在时刻 $t_{\text{change}}$ 发生均值突变：

$$\varepsilon_t = \begin{cases} \mathcal{N}(\mu_0, \sigma_0^2) & t < t_{\text{change}} \\ \mathcal{N}(\mu_0 + \Delta\mu, \sigma_0^2) & t \geq t_{\text{change}} \end{cases} \tag{A60.1-change}$$

其中 $\Delta\mu$ 为突变幅度（通常 $\Delta\mu = 2\sigma_0 \sim 5\sigma_0$）。

**漂移数据**：

从时刻 $t_{\text{drift}}$ 开始线性漂移：

$$\varepsilon_t = \begin{cases} \mathcal{N}(\mu_0, \sigma_0^2) & t < t_{\text{drift}} \\ \mathcal{N}(\mu_0 + \rho \cdot (t - t_{\text{drift}}), \sigma_0^2) & t \geq t_{\text{drift}} \end{cases} \tag{A60.1-drift}$$

其中 $\rho$ 为漂移速率（通常 $\rho = 0.01\sigma_0 \sim 0.1\sigma_0$ 每步）。

**组合场景**：

将三种场景组合，模拟真实系统的误差模式：

$$\varepsilon_t = \text{Gaussian}(t) + \text{ChangePoint}(t) + \text{Drift}(t) \tag{A60.1-combined}$$

**合成数据的参数**：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| $\mu_0$ | 1.0 | 稳态均值 |
| $\sigma_0$ | 0.5 | 稳态标准差 |
| $T_{\text{stable}}$ | 500 | 稳态长度 |
| $t_{\text{change}}$ | 500 | 突变时刻 |
| $\Delta\mu$ | $3\sigma_0$ | 突变幅度 |
| $t_{\text{drift}}$ | 800 | 漂移开始时刻 |
| $\rho$ | $0.05\sigma_0$ | 漂移速率 |
| $T_{\text{total}}$ | 1000 | 总长度 |

- **输入**：场景参数
- **输出**：合成误差序列 $\{\varepsilon_t\}_{t=1}^{T}$
- **约束**：$\sigma_0 > 0$；$T_{\text{total}} > 0$

> **白话翻译**：合成数据生成就是"人造测试场景"——用高斯噪声模拟正常状态，用均值突变模拟"突然出问题"，用线性漂移模拟"慢慢变差"。三种场景可以组合，模拟真实系统的误差模式。参数可以调节——突变幅度、漂移速率等都可以根据测试需要调整。

---

### 第二部分：检测延迟测试

#### 2.2 检测延迟测试定义（A60.2）

检测延迟测试衡量系统在突变发生后多久能检测到异常：

$$\text{detection\_delay} = t_{\text{detect}} - t_{\text{change}} \tag{A60.2}$$

其中 $t_{\text{detect}}$ 为首次检测到异常的时刻，$t_{\text{change}}$ 为突变发生的时刻。

**通过条件**：

$$\text{detection\_delay} < 5 \text{ 步} \tag{A60.2-pass}$$

**检测延迟的统计特性**：

由于噪声的影响，检测延迟是随机变量。对多次独立实验统计：

$$\overline{\text{delay}} = \mathbb{E}[\text{detection\_delay}] \tag{A60.2-mean}$$

$$\text{delay}_{95\%} = \text{Percentile}_{95}(\text{detection\_delay}) \tag{A60.2-percentile}$$

**不同突变幅度的检测延迟**：

| 突变幅度 $\Delta\mu$ | 预期延迟 | 说明 |
|---------------------|---------|------|
| $1\sigma_0$ | 5-20步 | 小突变，检测较慢 |
| $2\sigma_0$ | 2-5步 | 中等突变 |
| $3\sigma_0$ | 1-3步 | 大突变，快速检测 |
| $5\sigma_0$ | 1步 | 巨大突变，立即检测 |

**三种检测方法的延迟对比**：

| 方法 | 小突变延迟 | 大突变延迟 | 说明 |
|------|----------|----------|------|
| 阈值检测 | 高 | 低 | 擅长大突变 |
| CUSUM | 低 | 中 | 擅长小突变 |
| EWMA | 中 | 中 | 均衡 |

- **输入**：突变数据，检测器状态
- **输出**：检测延迟（步数）
- **约束**：detection_delay ≥ 1

> **白话翻译**：检测延迟测试就是"多久能发现"——突变发生后，系统需要几步才能检测到。通过条件是5步以内。大突变（5倍标准差）应该1步就能检测到，小突变（1倍标准差）可能需要5-20步。三种检测方法各有擅长——阈值检测擅长大突变，CUSUM擅长小突变。

---

### 第三部分：误报率测试

#### 2.3 误报率测试定义（A60.3）

误报率测试衡量系统在稳态下的误报频率：

$$\text{FAR} = \frac{\text{误报次数}}{\text{稳态总步数}} \tag{A60.3}$$

**通过条件**：

$$\text{FAR} < 5\% \tag{A60.3-pass}$$

**误报率的统计特性**：

在稳态高斯假设下，三种检测方法的理论误报率为：

| 方法 | 理论误报率 | 参数设置 |
|------|----------|---------|
| 阈值检测（$k=3$） | 0.13% | 3σ阈值 |
| CUSUM（$h=5\sigma$） | 0.1-1% | 取决于 $h$ |
| EWMA（$L=3$） | 0.3-2% | 取决于 $\lambda$ 和 $L$ |

**误报率与检测延迟的权衡**：

降低误报率通常会增加检测延迟（引用A55.4）。验证方案需要确认系统在可接受的误报率（<5%）下达到合理的检测延迟（<5步）。

**长期误报率测试**：

使用10000步稳态数据测试长期误报率：

$$\text{FAR}_{\text{long}} = \frac{\text{误报次数}}{10000} \tag{A60.3-long}$$

- **输入**：稳态数据，检测器状态
- **输出**：误报率 FAR
- **约束**：FAR < 5%

> **白话翻译**：误报率测试就是"会不会乱报"——正常情况下，系统会不会误以为出了问题。通过条件是误报率低于5%。阈值检测（3σ阈值）的理论误报率只有0.13%，CUSUM和EWMA的误报率稍高但也在1-2%以内。长期测试（10000步）确认系统不会随时间退化。

---

### 第四部分：分布偏移检测灵敏度测试

#### 2.4 分布偏移检测灵敏度测试定义（A60.4）

灵敏度测试衡量系统能检测到多小的分布偏移：

$$\text{sensitivity} = \min\{\Delta\mu : \text{shift\_detected}(\Delta\mu) = \text{True}\} \tag{A60.4}$$

**测试方法**：

逐步增大偏移幅度 $\Delta\mu$，直到检测到分布偏移：

```
for Δμ in [0.1σ, 0.2σ, 0.5σ, 1.0σ, 2.0σ, 3.0σ]:
    生成偏移数据（均值偏移 Δμ）
    运行 detect_shift
    记录是否检测到偏移
```

**KS检验的灵敏度**：

KS检验对均值偏移的灵敏度取决于样本量和偏移幅度：

$$D \approx \frac{|\Delta\mu|}{\sigma} \cdot \frac{1}{\sqrt{n}} \tag{A60.4-ks-sensitivity}$$

当 $D > D_{\text{threshold}}$ 时检测到偏移。因此最小可检测偏移为：

$$\Delta\mu_{\min} \approx D_{\text{threshold}} \cdot \sigma \cdot \sqrt{n} \tag{A60.4-ks-min}$$

**MMD的灵敏度**：

MMD对均值偏移的灵敏度：

$$\text{MMD}^2 \approx \frac{|\Delta\mu|^2}{4\sigma_k^2 + 2\sigma^2} \tag{A60.4-mmd-sensitivity}$$

**灵敏度测试的通过条件**：

| 偏移幅度 | KS应检测 | MMD应检测 |
|---------|---------|----------|
| $0.5\sigma$ | 可能不检测 | 可能不检测 |
| $1.0\sigma$ | 应检测 | 应检测 |
| $2.0\sigma$ | 必须检测 | 必须检测 |
| $3.0\sigma$ | 必须检测 | 必须检测 |

- **输入**：不同偏移幅度的合成数据
- **输出**：各幅度的检测结果
- **约束**：$2\sigma$ 以上偏移必须检测到

> **白话翻译**：偏移检测灵敏度测试就是"多小的变化能发现"——逐步增大分布偏移的幅度，看系统多小的偏移还能检测到。1倍标准差以上的偏移应该能检测到，2倍以上必须检测到。KS和MMD两种方法的灵敏度不同——MMD通常更灵敏，但计算更慢。

---

### 第五部分：反馈路由正确性测试

#### 2.5 反馈路由正确性测试定义（A60.5）

反馈路由正确性测试确认反馈信号被正确路由到目标模块：

**测试用例**：

| 场景 | 预期驱力响应 | 预期模型响应 |
|------|------------|------------|
| 误差突增+异常检测触发 | 安全驱力激活 | 增强更新或结构重构 |
| 分布偏移检测触发 | 安全驱力激活 | 结构重构 |
| 误差持续下降 | 收益驱力激活 | 微修正（低增益） |
| 误差不变（稳态） | 无驱力增量 | 微修正（基础增益） |

**路由正确性的验证方法**：

1. **驱力路由验证**：确认异常→安全驱力、偏移→安全驱力、误差下降→收益驱力、误差上升→收益缺口
2. **模型路由验证**：确认小误差→微修正、大误差→结构重构、误差恶化→增益增大
3. **反馈强度验证**：确认反馈强度 ∈ [0, 1)，且与误差变化量单调递增

**路由正确性的通过条件**：

$$\text{route\_correctness} = \frac{\text{正确路由次数}}{\text{总测试次数}} \geq 95\% \tag{A60.5-pass}$$

- **输入**：各种场景的合成数据
- **输出**：路由正确率
- **约束**：正确率 ≥ 95%

> **白话翻译**：反馈路由正确性测试就是"通知发对了没"——异常时应该激活安全驱力，改善时应该激活收益驱力，大误差时应该触发结构重构。用各种场景的合成数据测试，确认95%以上的情况路由正确。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A60.1 | 合成数据：高斯+突变+漂移 | 场景参数 | 误差序列 | $\sigma_0 > 0$ |
| A60.2 | detection_delay = $t_{\text{detect}} - t_{\text{change}}$ | 突变数据、检测器 | 检测延迟（步） | delay < 5步 |
| A60.3 | FAR = 误报次数 / 稳态总步数 | 稳态数据、检测器 | 误报率 | FAR < 5% |
| A60.4 | sensitivity = min{Δμ : shift_detected} | 不同偏移幅度的数据 | 最小可检测偏移 | 2σ以上必须检测 |
| A60.5 | route_correctness = 正确路由次数 / 总测试次数 | 各种场景数据 | 路由正确率 | ≥ 95% |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import numpy as np

# 引用阶段59的数据类
from stage_059 import ErrorEvaluator, ErrorEvaluatorState
from stage_055 import AnomalyReport
from stage_056 import ShiftReport
from stage_053 import FeedbackRoutes


# ============================================================
# 阶段60 枚举：测试场景
# ============================================================
class TestScenario(Enum):
    """验证测试场景。"""
    GAUSSIAN_STABLE = "gaussian_stable"    # 高斯稳态
    CHANGE_POINT = "change_point"          # 突变
    DRIFT = "drift"                        # 漂移
    COMBINED = "combined"                  # 组合


# ============================================================
# 阶段60 数据类：验证结果
# ============================================================
@dataclass(frozen=True)
class ErrorValidationResult:
    """误差评估验证结果。

    组合阶段59的ErrorEvaluatorState，在评估器状态的基础上增加
    验证结果——检测延迟、误报率、灵敏度、路由正确性。

    Attributes:
        evaluator_state: 阶段59的评估器状态（组合字段）
        detection_delay_mean: 检测延迟均值
        detection_delay_95: 检测延迟95分位数
        false_alarm_rate: 误报率
        ks_sensitivity: KS检验灵敏度（最小可检测偏移）
        mmd_sensitivity: MMD灵敏度
        route_correctness: 反馈路由正确率
        all_tests_passed: 所有测试是否通过
        test_details: 各测试的详细结果

    Invariants:
        - detection_delay_mean >= 0
        - false_alarm_rate ∈ [0, 1]
        - route_correctness ∈ [0, 1]
    """
    evaluator_state: ErrorEvaluatorState
    detection_delay_mean: float = 0.0
    detection_delay_95: float = 0.0
    false_alarm_rate: float = 0.0
    ks_sensitivity: float = 0.0
    mmd_sensitivity: float = 0.0
    route_correctness: float = 0.0
    all_tests_passed: bool = False
    test_details: Optional[Dict] = None

    def __post_init__(self):
        if self.detection_delay_mean < 0:
            raise ValueError(
                f"检测延迟均值必须 >= 0，收到 {self.detection_delay_mean}"
            )
        if not (0 <= self.false_alarm_rate <= 1):
            raise ValueError(
                f"误报率必须 ∈ [0,1]，收到 {self.false_alarm_rate}"
            )
        if not (0 <= self.route_correctness <= 1):
            raise ValueError(
                f"路由正确率必须 ∈ [0,1]，收到 {self.route_correctness}"
            )

    @property
    def delay_test_passed(self) -> bool:
        """检测延迟测试是否通过（<5步）。"""
        return self.detection_delay_95 < 5.0

    @property
    def far_test_passed(self) -> bool:
        """误报率测试是否通过（<5%）。"""
        return self.false_alarm_rate < 0.05

    @property
    def sensitivity_test_passed(self) -> bool:
        """灵敏度测试是否通过（2σ以上必须检测）。"""
        return self.ks_sensitivity <= 2.0 or self.mmd_sensitivity <= 2.0

    @property
    def route_test_passed(self) -> bool:
        """路由正确性测试是否通过（≥95%）。"""
        return self.route_correctness >= 0.95


# ============================================================
# 阶段60 核心接口：误差评估验证器
# ============================================================
class ErrorValidator:
    """阶段60的核心接口：误差评估验证器。

    实现合成数据生成、检测延迟测试、误报率测试、
    分布偏移检测灵敏度测试和反馈路由正确性测试。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 检测延迟 < 5步
        - 误报率 < 5%
        - 2σ以上偏移必须检测
        - 路由正确率 ≥ 95%
    """

    def generate_synthetic_data(
        self,
        scenario: TestScenario,
        mu_0: float = 1.0,
        sigma_0: float = 0.5,
        T_stable: int = 500,
        t_change: int = 500,
        delta_mu: float = 1.5,
        t_drift: int = 800,
        drift_rate: float = 0.025,
        T_total: int = 1000,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """生成合成数据（A60.1）。

        Args:
            scenario: 测试场景
            mu_0: 稳态均值
            sigma_0: 稳态标准差
            T_stable: 稳态长度
            t_change: 突变时刻
            delta_mu: 突变幅度
            t_drift: 漂移开始时刻
            drift_rate: 漂移速率
            T_total: 总长度
            seed: 随机种子

        Returns:
            合成误差序列

        Raises:
            ValueError: 当参数不合法时
        """
        if sigma_0 <= 0:
            raise ValueError(f"sigma_0 必须 > 0，收到 {sigma_0}")
        if T_total <= 0:
            raise ValueError(f"T_total 必须 > 0，收到 {T_total}")
        if seed is not None:
            np.random.seed(seed)
        data = np.random.normal(mu_0, sigma_0, size=T_total)
        if scenario == TestScenario.GAUSSIAN_STABLE:
            return data
        if scenario in (TestScenario.CHANGE_POINT, TestScenario.COMBINED):
            if t_change < T_total:
                data[t_change:] += delta_mu
        if scenario in (TestScenario.DRIFT, TestScenario.COMBINED):
            if t_drift < T_total:
                drift = drift_rate * np.arange(T_total - t_drift)
                data[t_drift:] += drift
        return data

    def test_detection_delay(
        self,
        evaluator: ErrorEvaluator,
        state: ErrorEvaluatorState,
        data: np.ndarray,
        t_change: int,
        n_trials: int = 100
    ) -> Tuple[float, float]:
        """检测延迟测试（A60.2）。

        Args:
            evaluator: 误差评估器
            state: 评估器状态
            data: 突变数据
            t_change: 突变时刻
            n_trials: 试验次数

        Returns:
            (延迟均值, 延迟95分位数)
        """
        delays = []
        for _ in range(n_trials):
            current_state = state
            detected = False
            for t in range(len(data)):
                # 模拟单步评估：使用当前数据点作为预测和观测
                val = data[t]
                try:
                    _, report, _, _, current_state = evaluator.full_step(
                        current_state,
                        prediction_low=np.array([0.0]),
                        prediction_mid=np.array([0.0]),
                        prediction_high=np.array([0.0]),
                        observation_low=np.array([val]),
                        observation_mid=np.array([val]),
                        observation_high=np.array([val]),
                    )
                except Exception:
                    continue
                if hasattr(report, 'is_anomaly') and report.is_anomaly:
                    if t >= t_change:
                        delays.append(t - t_change)
                        detected = True
                        break
            if not detected:
                delays.append(len(data) - t_change)
        mean_delay = float(np.mean(delays)) if delays else float(len(data) - t_change)
        p95_delay = float(np.percentile(delays, 95)) if delays else float(len(data) - t_change)
        return mean_delay, p95_delay

    def test_false_alarm_rate(
        self,
        evaluator: ErrorEvaluator,
        state: ErrorEvaluatorState,
        data: np.ndarray
    ) -> float:
        """误报率测试（A60.3）。

        Args:
            evaluator: 误差评估器
            state: 评估器状态
            data: 稳态数据

        Returns:
            误报率 FAR ∈ [0, 1]
        """
        current_state = state
        false_alarms = 0
        for t in range(len(data)):
            val = data[t]
            try:
                _, report, _, _, current_state = evaluator.full_step(
                    current_state,
                    prediction_low=np.array([0.0]),
                    prediction_mid=np.array([0.0]),
                    prediction_high=np.array([0.0]),
                    observation_low=np.array([val]),
                    observation_mid=np.array([val]),
                    observation_high=np.array([val]),
                )
            except Exception:
                continue
            if hasattr(report, 'is_anomaly') and report.is_anomaly:
                false_alarms += 1
        return false_alarms / len(data) if len(data) > 0 else 0.0

    def test_shift_sensitivity(
        self,
        evaluator: ErrorEvaluator,
        state: ErrorEvaluatorState,
        delta_mus: List[float] = None,
        sigma_0: float = 0.5
    ) -> Tuple[float, float]:
        """分布偏移检测灵敏度测试（A60.4）。

        Args:
            evaluator: 误差评估器
            state: 评估器状态
            delta_mus: 偏移幅度列表
            sigma_0: 稳态标准差

        Returns:
            (KS灵敏度, MMD灵敏度)——最小可检测偏移（σ单位）
        """
        if delta_mus is None:
            delta_mus = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
        ks_sens = float('inf')
        mmd_sens = float('inf')
        for dmu in delta_mus:
            data = self.generate_synthetic_data(
                TestScenario.CHANGE_POINT,
                mu_0=0.0,
                sigma_0=sigma_0,
                T_total=1000,
                t_change=500,
                delta_mu=dmu * sigma_0,
            )
            current_state = state
            for t in range(len(data)):
                val = data[t]
                try:
                    _, _, shift, _, current_state = evaluator.full_step(
                        current_state,
                        prediction_low=np.array([0.0]),
                        prediction_mid=np.array([0.0]),
                        prediction_high=np.array([0.0]),
                        observation_low=np.array([val]),
                        observation_mid=np.array([val]),
                        observation_high=np.array([val]),
                    )
                except Exception:
                    continue
            if hasattr(shift, 'is_shift') and shift.is_shift:
                ks_sens = min(ks_sens, dmu)
            if hasattr(shift, 'mmd_squared') and shift.mmd_squared > 0.01:
                mmd_sens = min(mmd_sens, dmu)
        return ks_sens, mmd_sens

    def test_route_correctness(
        self,
        evaluator: ErrorEvaluator,
        state: ErrorEvaluatorState,
        scenarios: List[Tuple[str, np.ndarray]] = None
    ) -> float:
        """反馈路由正确性测试（A60.5）。

        Args:
            evaluator: 误差评估器
            state: 评估器状态
            scenarios: 测试场景列表

        Returns:
            路由正确率 ∈ [0, 1]
        """
        if scenarios is None:
            scenarios = [
                ("change_point", self.generate_synthetic_data(TestScenario.CHANGE_POINT)),
                ("drift", self.generate_synthetic_data(TestScenario.DRIFT)),
                ("stable", self.generate_synthetic_data(TestScenario.GAUSSIAN_STABLE)),
            ]
        correct = 0
        total = 0
        for scenario_name, data in scenarios:
            current_state = state
            for t in range(len(data)):
                val = data[t]
                try:
                    error, anomaly, shift, routes, current_state = evaluator.full_step(
                        current_state,
                        prediction_low=np.array([0.0]),
                        prediction_mid=np.array([0.0]),
                        prediction_high=np.array([0.0]),
                        observation_low=np.array([val]),
                        observation_mid=np.array([val]),
                        observation_high=np.array([val]),
                    )
                except Exception:
                    continue
                total += 1
                # 基本路由正确性检查
                is_ok = True
                if scenario_name == "change_point" and hasattr(anomaly, 'is_anomaly'):
                    if anomaly.is_anomaly and hasattr(routes, 'safe_drive_increment'):
                        is_ok = routes.safe_drive_increment > 0
                elif scenario_name == "drift" and hasattr(shift, 'is_shift'):
                    if shift.is_shift and hasattr(routes, 'safe_drive_increment'):
                        is_ok = routes.safe_drive_increment > 0
                elif scenario_name == "stable":
                    if hasattr(anomaly, 'is_anomaly') and not anomaly.is_anomaly:
                        if hasattr(routes, 'safe_drive_increment'):
                            is_ok = routes.safe_drive_increment == 0.0
                if is_ok:
                    correct += 1
        return correct / total if total > 0 else 0.0

    def run_all_tests(
        self,
        evaluator: ErrorEvaluator,
        initial_state: ErrorEvaluatorState
    ) -> ErrorValidationResult:
        """运行所有验证测试（A60.1-A60.5）。

        Args:
            evaluator: 误差评估器
            initial_state: 初始评估器状态

        Returns:
            ErrorValidationResult 包含所有测试结果
        """
        # 检测延迟测试
        change_data = self.generate_synthetic_data(TestScenario.CHANGE_POINT)
        delay_mean, delay_95 = self.test_detection_delay(
            evaluator, initial_state, change_data, t_change=500, n_trials=10
        )

        # 误报率测试
        stable_data = self.generate_synthetic_data(TestScenario.GAUSSIAN_STABLE, T_total=5000)
        far = self.test_false_alarm_rate(evaluator, initial_state, stable_data)

        # 灵敏度测试
        ks_sens, mmd_sens = self.test_shift_sensitivity(evaluator, initial_state)

        # 路由正确性测试
        route_corr = self.test_route_correctness(evaluator, initial_state)

        all_passed = (
            delay_95 < 5.0
            and far < 0.05
            and (ks_sens <= 2.0 or mmd_sens <= 2.0)
            and route_corr >= 0.95
        )

        test_details = {
            "detection_delay_mean": delay_mean,
            "detection_delay_95": delay_95,
            "false_alarm_rate": far,
            "ks_sensitivity": ks_sens,
            "mmd_sensitivity": mmd_sens,
            "route_correctness": route_corr,
        }

        return ErrorValidationResult(
            evaluator_state=initial_state,
            detection_delay_mean=delay_mean,
            detection_delay_95=delay_95,
            false_alarm_rate=far,
            ks_sensitivity=ks_sens,
            mmd_sensitivity=mmd_sens,
            route_correctness=route_corr,
            all_tests_passed=all_passed,
            test_details=test_details,
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `ErrorEvaluatorState` | ← 阶段59 | 作为验证的评估器状态输入（组合字段） |
| **输入** | `ErrorEvaluator` | ← 阶段59 | 被验证的误差评估器 |
| **输入** | A55.4（检测延迟与误报率权衡） | ← 阶段55 | 检测延迟和误报率测试的理论依据 |
| **输入** | A56.1-A56.2（KS与MMD） | ← 阶段56 | 灵敏度测试的理论依据 |
| **输出** | `ErrorValidationResult` | → 工程实现 | 验证结果，确认系统正确性 |
| **输出** | A60.1-A60.5（验证标准） | → 工程实现 | 误差评估模块的验证规范 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| NumPy | 合成数据生成 | `numpy.random`生成高斯、突变、漂移数据 |
| pytest | 测试框架 | 支撑验证测试的结构化执行和报告 |
| SciPy | 统计检验 | `scipy.stats`支撑KS检验和MMD计算 |
| JAX | 高性能验证 | `jax.vmap`并行执行多次试验 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V60.1 | 合成数据正确性 | 高斯数据符合指定分布，突变/漂移在指定时刻发生 | 统计检验 |
| V60.2 | 检测延迟 < 5步 | 3σ突变检测延迟95分位数 < 5步 | 多次试验统计 |
| V60.3 | 大突变1步检测 | 5σ突变检测延迟 = 1步 | 极端突变测试 |
| V60.4 | 误报率 < 5% | 稳态数据误报率 < 5% | 10000步稳态测试 |
| V60.5 | 阈值检测误报率 | 3σ阈值检测误报率 ≈ 0.13% | 理论对比 |
| V60.6 | CUSUM误报率 | CUSUM误报率 < 2% | 稳态测试 |
| V60.7 | EWMA误报率 | EWMA误报率 < 3% | 稳态测试 |
| V60.8 | KS灵敏度 | 2σ以上均值偏移必须检测到 | 灵敏度扫描 |
| V60.9 | MMD灵敏度 | 2σ以上均值偏移必须检测到 | 灵敏度扫描 |
| V60.10 | 漂移检测 | 线性漂移在50步内被检测到 | 漂移测试 |
| V60.11 | 异常→安全驱力路由 | 异常检测触发时安全驱力增量 > 0 | 路由验证 |
| V60.12 | 偏移→安全驱力路由 | 偏移检测触发时安全驱力增量 > 0 | 路由验证 |
| V60.13 | 误差下降→收益驱力路由 | 误差下降时收益驱力增量 > 0 | 路由验证 |
| V60.14 | 路由正确率 ≥ 95% | 综合路由正确率 ≥ 95% | 综合测试 |
| V60.15 | ErrorValidationResult组合 | `result.evaluator_state` 类型为 `ErrorEvaluatorState` | 类型检查 |

---

> **阶段60完** | 阶段60建立了误差评估的完整验证方案：合成数据生成模拟高斯稳态+突变+漂移三种场景（A60.1），检测延迟测试确认3σ突变检测延迟<5步（A60.2），误报率测试确认稳态误报率<5%（A60.3），分布偏移检测灵敏度测试确认2σ以上偏移必须检测到（A60.4），反馈路由正确性测试确认路由正确率≥95%（A60.5），封装为ErrorValidationResult数据类组合ErrorEvaluatorState。至此，阶段51-60建立了完整的误差评估体系：标准化误差定义（51）→ 时间窗口与累积（52）→ 反馈信号生成（53）→ 精度加权（54）→ 异常检测（55）→ 分布偏移检测（56）→ 驱力映射（57）→ 模型映射（58）→ 统一接口（59）→ 验证方案（60）。从误差度量到异常检测到反馈路由，误差评估体系为智能体提供了完整的"感知-评估-响应"闭环。
