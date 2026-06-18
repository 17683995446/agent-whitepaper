# 阶段98：进化指标监控面板

> **阶段编号**：S98 | **所属层**：第十层·系统集成层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 精度指标 | "猜得多准"——预测误差的长期趋势下降 |
> | 广度指标 | "懂了多少"——认知覆盖范围的长期趋势上升 |
> | 能效指标 | "算得多省"——计算效率的长期趋势上升 |
> | 综合进化指数EVI | "进化成绩单"——精度+广度+能效的加权综合 |
> | 监控数据采集与存储 | "体检数据存档"——定期采集指标，持久化存储 |
> | 异常告警规则 | "健康警报"——指标异常时自动告警 |
> | EvolutionMonitor | "进化监控的完整画像"——组合E2EValidation，在验证基础上增加监控状态 |

---

## 1. 第一性原理依据

本阶段从阶段91（全局目标函数）的核心结论出发，建立进化指标监控面板。核心依据为：

- **A91.1（全局目标函数）**（引用阶段91）：$J = \int F \, dt + \int G \, dt + \lambda_1 R_{complexity} + \lambda_2 R_{safety}$ 定义了系统的四维进化目标。监控面板的四个指标直接对应 $J$ 的四个维度：精度指标对应 $\int F \, dt$，广度指标对应 $\int G \, dt$，能效指标对应 $\lambda_1 R_{complexity}$，安全指标对应 $\lambda_2 R_{safety}$。

本阶段的核心目标是：**定义精度指标 accuracy = 1 - F/F_initial（A98.1），广度指标 coverage = dim(Ψ_known)/dim(Ψ_total)（A98.2），能效指标 efficiency = effective_compute/total_compute（A98.3），综合进化指数 EVI = α·accuracy + β·coverage + γ·efficiency（A98.4），监控数据采集与存储（A98.5），异常告警规则（A98.6），封装为EvolutionMonitor数据类组合E2EValidation。**

> **去重声明**：本阶段仅定义进化指标的监控面板和告警规则，不重新定义全局目标函数（阶段91的A91.1）或端到端验证（阶段97）。监控面板是对系统运行状态的实时观测工具。

---

## 2. 核心数学公式

### 第一部分：精度指标

#### 2.1 精度指标定义（A98.1）

精度指标度量系统预测的准确程度，长期趋势应下降：

$$\text{accuracy}_t = 1 - \frac{F_t}{F_{initial}} \tag{A98.1}$$

其中：
- $F_t$ 为当前变分自由能（引用A4.4）
- $F_{initial}$ 为初始变分自由能（bootstrap后的值）

**精度指标的范围**：

| accuracy_t | 含义 | 说明 |
|-----------|------|------|
| < 0 | 性能劣化 | 比初始状态还差 |
| = 0 | 初始水平 | 没有改善 |
| 0 ~ 0.5 | 适应中 | 预测误差在下降 |
| 0.5 ~ 0.9 | 成熟期 | 预测相当准确 |
| > 0.9 | 高精度 | 预测非常准确 |

**精度指标的长期趋势判定**：

$$\text{accuracy\_trend} = \text{slope}(\text{EMA}(\text{accuracy}, W)) \tag{A98.1-trend}$$

| 趋势 | slope | 含义 |
|------|-------|------|
| 上升 | > 0 | 精度在改善（好） |
| 平稳 | = 0 | 精度稳定 |
| 下降 | < 0 | 精度在退化（坏） |

**精度指标的滑动平均窗口**：

$$W_{accuracy} = 1000 \text{ 步} \tag{A98.1-window}$$

使用1000步的滑动平均消除短期波动，揭示长期趋势。

- **输入**：变分自由能时间序列 $\{F_t\}$
- **输出**：精度指标、精度趋势
- **约束**：长期趋势应上升（accuracy↑ = F↓）

> **白话翻译**：精度指标就是"猜得多准"——用1减去当前误差除以初始误差。0表示没进步，0.5表示误差减半，0.9表示误差只有最初的10%。长期趋势应该上升——越猜越准。

---

### 第二部分：广度指标

#### 2.2 广度指标定义（A98.2）

广度指标度量系统认知覆盖的范围，长期趋势应上升：

$$\text{coverage}_t = \frac{\dim(\Psi_{known}(t))}{\dim(\Psi_{total})} \tag{A98.2}$$

其中：
- $\Psi_{known}(t)$ 为截至时刻 $t$ 系统已探索的认知空间子集
- $\Psi_{total}$ 为全部认知空间
- $\dim(\cdot)$ 为有效维度

**广度指标的计算方法**：

认知空间的有效维度通过预期自由能的信息增益项估计：

$$\dim(\Psi_{known}) = \sum_{t'=0}^{t} \mathbb{1}[I(s; o | \pi_{t'}) > \theta_{IG}] \tag{A98.2-dim}$$

其中 $I(s; o | \pi)$ 为策略 $\pi$ 下的互信息（信息增益），$\theta_{IG}$ 为信息增益阈值。

**广度指标的范围**：

| coverage_t | 含义 | 说明 |
|-----------|------|------|
| 0 ~ 0.2 | 初期 | 认知覆盖很小 |
| 0.2 ~ 0.5 | 扩展中 | 认知覆盖在增长 |
| 0.5 ~ 0.8 | 成熟期 | 认知覆盖较广 |
| > 0.8 | 全面 | 认知覆盖接近完全 |

**广度指标与A91.3的对应**：

A91.3（引用阶段91）定义广度进化项 $J_{coverage} = \int G \, dt$。A98.2与A91.3的对应关系为：$J_{coverage}$ 下降等价于 $\text{coverage}_t$ 上升——累积决策不确定性降低意味着认知覆盖扩张。

- **输入**：信息增益时间序列
- **输出**：广度指标、广度趋势
- **约束**：长期趋势应上升

> **白话翻译**：广度指标就是"懂了多少"——已探索的认知空间除以总认知空间。0表示啥也不懂，0.5表示懂了一半，1表示全懂了。长期趋势应该上升——越懂越多。

---

### 第三部分：能效指标

#### 2.3 能效指标定义（A98.3）

能效指标度量系统的计算效率，长期趋势应上升：

$$\text{efficiency}_t = \frac{\text{effective\_compute}(t)}{\text{total\_compute}(t)} \tag{A98.3}$$

其中：
- $\text{effective\_compute}(t)$ 为产生有效预测的计算量
- $\text{total\_compute}(t)$ 为总计算量

**有效计算量的定义**：

$$\text{effective\_compute} = \sum_{t'=0}^{t} \mathbb{1}[F_{t'} < F_{threshold}] \cdot c_{t'} \tag{A98.3-effective}$$

其中 $F_{threshold}$ 为有效预测的误差阈值，$c_{t'}$ 为第 $t'$ 步的计算量。

**能效指标与A91.4的对应**：

A91.4（引用阶段91）定义能效进化项 $J_{efficiency} = \lambda_1 R_{complexity}$。A98.3与A91.4的对应关系为：$J_{efficiency}$ 下降等价于 $\text{efficiency}_t$ 上升——模型复杂度降低意味着计算效率提升。

**能效指标的资源维度**：

| 资源 | 效率计算 | 说明 |
|------|---------|------|
| CPU | effective_cpu / total_cpu | CPU利用率 |
| 内存 | active_params / total_params | 参数利用率 |
| 存储 | useful_data / total_data | 数据利用率 |

- **输入**：计算量统计
- **输出**：能效指标、能效趋势
- **约束**：长期趋势应上升

> **白话翻译**：能效指标就是"算得多省"——有效计算量除以总计算量。有效计算是指产生准确预测的计算，浪费的计算不算。长期趋势应该上升——越算越省。

---

### 第四部分：综合进化指数

#### 2.4 综合进化指数定义（A98.4）

综合进化指数EVI将三个指标加权综合：

$$\text{EVI}_t = \alpha \cdot \text{accuracy}_t + \beta \cdot \text{coverage}_t + \gamma \cdot \text{efficiency}_t \tag{A98.4}$$

其中：
- $\alpha = 0.40$ 为精度权重
- $\beta = 0.35$ 为广度权重
- $\gamma = 0.25$ 为能效权重

**EVI的权重选择依据**：

| 维度 | 权重 | 依据 |
|------|------|------|
| 精度 | 0.40 | 最核心——预测不准其他都白搭 |
| 广度 | 0.35 | 次核心——认知覆盖决定适应范围 |
| 能效 | 0.25 | 重要但非核心——效率可以慢慢优化 |

**EVI的范围和含义**：

| EVI | 含义 | 说明 |
|-----|------|------|
| < 0.2 | 初期 | 系统刚启动，各方面都在适应 |
| 0.2 ~ 0.5 | 成长期 | 系统在快速进化 |
| 0.5 ~ 0.8 | 成熟期 | 系统进化趋于稳定 |
| > 0.8 | 高进化 | 系统进化水平很高 |

**EVI的进化率**：

$$\text{EVI\_rate} = \frac{d(\text{EVI})}{dt} \tag{A98.4-rate}$$

EVI_rate > 0 表示系统在进化，EVI_rate < 0 表示系统在退化。

**EVI与全局目标的对应**：

$$\text{EVI}_t \uparrow \iff J_t \downarrow \tag{A98.4-equiv}$$

EVI上升等价于全局目标J下降——两者从不同角度度量同一个进化过程。

- **输入**：精度、广度、能效三个指标
- **输出**：EVI、EVI_rate
- **约束**：α+β+γ = 1.0；EVI_rate > 0（正常进化时）

> **白话翻译**：综合进化指数EVI就是"进化成绩单"——精度40%+广度35%+能效25%的加权平均。EVI上升说明系统在进化，下降说明在退化。EVI和全局目标J是同一个硬币的两面——EVI上升等于J下降。

---

### 第五部分：监控数据采集与存储

#### 2.5 监控数据采集与存储定义（A98.5）

定期采集进化指标，持久化存储供分析和告警：

$$\text{metrics}_t = \{\text{accuracy}_t, \text{coverage}_t, \text{efficiency}_t, \text{EVI}_t, F_t, G_t, J_t\} \tag{A98.5}$$

**采集频率**：

| 指标 | 采集频率 | 存储方式 |
|------|---------|---------|
| F, G, J | 每步 | 内存环形缓冲区 |
| accuracy, coverage, efficiency | 每100步 | 内存+SQLite |
| EVI | 每100步 | 内存+SQLite |
| 趋势判定 | 每1000步 | SQLite |

**存储结构**：

```sql
CREATE TABLE evolution_metrics (
    timestamp REAL PRIMARY KEY,
    step_count INTEGER,
    free_energy REAL,
    expected_free_energy REAL,
    global_objective REAL,
    accuracy REAL,
    coverage REAL,
    efficiency REAL,
    evi REAL,
    evi_rate REAL
);
```

**数据保留策略**：

| 数据粒度 | 保留时长 | 说明 |
|---------|---------|------|
| 原始数据（每步） | 24小时 | 内存中，滚动覆盖 |
| 100步聚合 | 7天 | SQLite |
| 1000步聚合 | 30天 | SQLite |
| 1小时聚合 | 永久 | SQLite |

- **输入**：系统运行数据
- **输出**：持久化的指标数据
- **约束**：存储占用 < 磁盘80%

> **白话翻译**：监控数据采集与存储就是"体检数据存档"——定期采集精度、广度、能效、EVI等指标，存到SQLite。原始数据保留一天，聚合数据保留更久。存储占用不能超过磁盘80%。

---

### 第六部分：异常告警规则

#### 2.6 异常告警规则定义（A98.6）

当进化指标出现异常时自动告警：

**告警规则**：

| 规则 | 条件 | 级别 | 动作 |
|------|------|------|------|
| 精度退化 | accuracy_trend < 0 持续 1000步 | WARNING | 记录日志 |
| 广度停滞 | coverage_trend = 0 持续 5000步 | INFO | 记录日志 |
| 能效下降 | efficiency_trend < 0 持续 1000步 | WARNING | 记录日志 |
| EVI退化 | EVI_rate < 0 持续 T_degradation | CRITICAL | 触发自修复（阶段99） |
| 全局目标上升 | dJ/dt > 0 持续 1000步 | WARNING | 记录日志 |
| 安全违规 | R_safety > 0 | CRITICAL | 安全护栏处理 |
| 资源超限 | CPU>20% 或 内存>20% | CRITICAL | 安全护栏处理 |

**告警的升级机制**：

$$\text{WARNING} \xrightarrow{\text{持续} 2T} \text{CRITICAL} \xrightarrow{\text{持续} 3T} \text{EMERGENCY} \tag{A98.6-escalation}$$

| 级别 | 持续时间 | 动作 |
|------|---------|------|
| WARNING | 首次检测 | 记录日志 |
| CRITICAL | 持续2T | 触发自修复 |
| EMERGENCY | 持续3T | 人工告警 |

**告警与阶段99的对接**：

CRITICAL级告警直接触发阶段99的退化检测与自修复机制。EVI_rate < 0 持续 $T_{degradation}$ 秒是自修复的触发条件。

- **输入**：进化指标实时数据
- **输出**：告警事件
- **约束**：CRITICAL级100%触发自修复

> **白话翻译**：异常告警规则就是"健康警报"——精度退化、广度停滞、能效下降、EVI退化、全局目标上升、安全违规、资源超限，各有各的告警规则。WARNING记日志，CRITICAL触发自修复，EMERGENCY通知人工。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A98.1 | accuracy = 1 - F/F_initial | F序列 | 精度指标 | 长期↑ |
| A98.2 | coverage = dim(Ψ_known)/dim(Ψ_total) | 信息增益 | 广度指标 | 长期↑ |
| A98.3 | efficiency = effective_compute/total_compute | 计算统计 | 能效指标 | 长期↑ |
| A98.4 | EVI = α·accuracy + β·coverage + γ·efficiency | 三项指标 | 综合指数 | α+β+γ=1 |
| A98.5 | metrics_t = {accuracy, coverage, efficiency, EVI, F, G, J} | 运行数据 | 持久化指标 | 存储<80% |
| A98.6 | 告警规则7条 | 指标实时数据 | 告警事件 | CRITICAL→自修复 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import numpy as np

# 引用阶段97的数据类
from stage_097 import E2EValidation


# ============================================================
# 阶段98 枚举：告警级别
# ============================================================
class AlertLevel(Enum):
    """告警级别。"""
    INFO = "info"              # 信息
    WARNING = "warning"        # 警告
    CRITICAL = "critical"      # 严重
    EMERGENCY = "emergency"    # 紧急


# ============================================================
# 阶段98 数据类：告警事件
# ============================================================
@dataclass(frozen=True)
class AlertEvent:
    """告警事件。

    Attributes:
        alert_level: 告警级别
        rule_name: 触发的规则名称
        metric_name: 异常指标名称
        current_value: 当前值
        threshold: 阈值
        duration: 持续时间
        timestamp: 时间戳
    """
    alert_level: AlertLevel
    rule_name: str
    metric_name: str
    current_value: float
    threshold: float
    duration: float
    timestamp: float


# ============================================================
# 阶段98 数据类：进化监控
# ============================================================
@dataclass(frozen=True)
class EvolutionMonitor:
    """进化指标监控面板的完整状态描述。

    组合阶段97的E2EValidation，在验证基础上增加监控状态。

    Attributes:
        e2e_validation: 阶段97的端到端验证（组合字段）
        accuracy: 当前精度指标
        coverage: 当前广度指标
        efficiency: 当前能效指标
        evi: 当前综合进化指数
        evi_rate: EVI变化率
        accuracy_trend: 精度趋势
        coverage_trend: 广度趋势
        efficiency_trend: 能效趋势
        F_initial: 初始自由能
        alpha: 精度权重
        beta: 广度权重
        gamma: 能效权重
        active_alerts: 当前活跃告警
        metrics_history: 指标历史（最近1000条）

    Invariants:
        - alpha + beta + gamma = 1.0
        - 0 <= accuracy, coverage, efficiency, evi <= 1.0
    """
    e2e_validation: E2EValidation
    accuracy: float = 0.0
    coverage: float = 0.0
    efficiency: float = 0.0
    evi: float = 0.0
    evi_rate: float = 0.0
    accuracy_trend: float = 0.0
    coverage_trend: float = 0.0
    efficiency_trend: float = 0.0
    F_initial: float = 1.0
    alpha: float = 0.40
    beta: float = 0.35
    gamma: float = 0.25
    active_alerts: Tuple[AlertEvent, ...] = ()
    metrics_history: Tuple[Dict, ...] = ()


# ============================================================
# 阶段98 核心接口：进化监控器
# ============================================================
class EvolutionMonitorExecutor:
    """阶段98的核心接口：进化指标监控面板。

    实现精度、广度、能效指标的采集、计算和告警。

    核心约束：
        - EVI长期趋势上升
        - CRITICAL级告警触发自修复
        - 存储占用 < 80%
    """

    def __init__(
        self,
        alpha: float = 0.40,
        beta: float = 0.35,
        gamma: float = 0.25,
        trend_window: int = 1000
    ):
        """初始化进化监控器。

        Args:
            alpha: 精度权重，必须 >= 0
            beta: 广度权重，必须 >= 0
            gamma: 能效权重，必须 >= 0
            trend_window: 趋势计算窗口，必须 > 0

        Raises:
            ValueError: 当alpha + beta + gamma != 1时
            ValueError: 当trend_window <= 0时

        Examples:
            >>> monitor = EvolutionMonitorExecutor()
            >>> monitor.alpha
            0.4
        """
        if not np.isclose(alpha + beta + gamma, 1.0):
            raise ValueError(
                f"权重之和必须等于1，收到 {alpha + beta + gamma}"
            )
        if alpha < 0 or beta < 0 or gamma < 0:
            raise ValueError("权重必须非负")
        if trend_window <= 0:
            raise ValueError(f"trend_window必须 > 0，收到 {trend_window}")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.trend_window = trend_window

    def compute_accuracy(
        self,
        F_current: float,
        F_initial: float
    ) -> float:
        """计算精度指标（A98.1）。

        accuracy = 1 - F_current / F_initial

        Args:
            F_current: 当前自由能
            F_initial: 初始自由能，必须 != 0

        Returns:
            精度指标 ∈ [0, 1]

        Raises:
            ValueError: 当F_initial == 0时

        Examples:
            >>> monitor = EvolutionMonitorExecutor()
            >>> monitor.compute_accuracy(8.0, 10.0)
            0.2
        """
        if F_initial == 0:
            raise ValueError("F_initial不能为0")
        return max(0.0, min(1.0, 1.0 - F_current / F_initial))

    def compute_coverage(
        self,
        known_dims: int,
        total_dims: int
    ) -> float:
        """计算广度指标（A98.2）。

        Args:
            known_dims: 已知维度数，必须 >= 0
            total_dims: 总维度数，必须 > 0

        Returns:
            广度指标 ∈ [0, 1]

        Raises:
            ValueError: 当known_dims < 0时
            ValueError: 当total_dims <= 0时

        Examples:
            >>> monitor = EvolutionMonitorExecutor()
            >>> monitor.compute_coverage(5, 10)
            0.5
        """
        if known_dims < 0:
            raise ValueError(f"known_dims必须 >= 0，收到 {known_dims}")
        if total_dims <= 0:
            raise ValueError(f"total_dims必须 > 0，收到 {total_dims}")
        return min(1.0, known_dims / total_dims)

    def compute_efficiency(
        self,
        effective_compute: float,
        total_compute: float
    ) -> float:
        """计算能效指标（A98.3）。

        Args:
            effective_compute: 有效计算量，必须 >= 0
            total_compute: 总计算量，必须 > 0

        Returns:
            能效指标 ∈ [0, 1]

        Raises:
            ValueError: 当effective_compute < 0时
            ValueError: 当total_compute <= 0时

        Examples:
            >>> monitor = EvolutionMonitorExecutor()
            >>> monitor.compute_efficiency(80.0, 100.0)
            0.8
        """
        if effective_compute < 0:
            raise ValueError(
                f"effective_compute必须 >= 0，收到 {effective_compute}"
            )
        if total_compute <= 0:
            raise ValueError(
                f"total_compute必须 > 0，收到 {total_compute}"
            )
        return min(1.0, effective_compute / total_compute)

    def compute_evi(
        self,
        accuracy: float,
        coverage: float,
        efficiency: float
    ) -> float:
        """计算综合进化指数（A98.4）。

        EVI = α·accuracy + β·coverage + γ·efficiency

        Args:
            accuracy: 精度指标，范围[0,1]
            coverage: 广度指标，范围[0,1]
            efficiency: 能效指标，范围[0,1]

        Returns:
            EVI ∈ [0, 1]

        Examples:
            >>> monitor = EvolutionMonitorExecutor()
            >>> evi = monitor.compute_evi(0.8, 0.6, 0.7)
            >>> 0 <= evi <= 1
            True
        """
        return (
            self.alpha * accuracy
            + self.beta * coverage
            + self.gamma * efficiency
        )

    def collect_metrics(
        self,
        monitor: EvolutionMonitor,
        F: float,
        G: float,
        J: float
    ) -> EvolutionMonitor:
        """采集并存储监控数据（A98.5）。

        Args:
            monitor: 当前监控状态
            F: 当前自由能
            G: 当前预期自由能
            J: 当前全局目标

        Returns:
            更新后的监控状态

        Examples:
            >>> monitor = EvolutionMonitorExecutor()
            >>> em = EvolutionMonitor()
            >>> new_em = monitor.collect_metrics(em, 8.0, 5.0, 20.0)
            >>> new_em.current_F
            8.0
        """
        new_history = monitor.F_history + (F,)
        if len(new_history) > self.trend_window:
            new_history = new_history[-self.trend_window:]
        return EvolutionMonitor(
            F_history=new_history,
            G_history=monitor.G_history + (G,),
            J_history=monitor.J_history + (J,),
            accuracy_history=monitor.accuracy_history,
            coverage_history=monitor.coverage_history,
            efficiency_history=monitor.efficiency_history,
            evi_history=monitor.evi_history,
            current_F=F,
            current_G=G,
            current_J=J,
            current_accuracy=monitor.current_accuracy,
            current_coverage=monitor.current_coverage,
            current_efficiency=monitor.current_efficiency,
            current_evi=monitor.current_evi,
            trend_accuracy=monitor.trend_accuracy,
            trend_coverage=monitor.trend_coverage,
            trend_efficiency=monitor.trend_efficiency,
            trend_evi=monitor.trend_evi,
            alert_history=monitor.alert_history,
            storage_usage=monitor.storage_usage
        )

    def check_alerts(
        self,
        monitor: EvolutionMonitor
    ) -> Tuple[EvolutionMonitor, List[AlertEvent]]:
        """检查异常告警规则（A98.6）。

        Args:
            monitor: 当前监控状态

        Returns:
            (更新后的监控状态, 新触发的告警列表)

        Examples:
            >>> monitor = EvolutionMonitorExecutor()
            >>> em = EvolutionMonitor()
            >>> new_em, alerts = monitor.check_alerts(em)
            >>> isinstance(alerts, list)
            True
        """
        alerts: List[AlertEvent] = []
        if monitor.storage_usage > 0.80:
            alerts.append(AlertEvent(
                level=AlertLevel.CRITICAL,
                metric="storage_usage",
                value=monitor.storage_usage,
                threshold=0.80,
                timestamp=0.0,
                message="存储占用超过80%"
            ))
        if len(monitor.F_history) >= 2:
            if monitor.F_history[-1] > monitor.F_history[0] * 1.1:
                alerts.append(AlertEvent(
                    level=AlertLevel.WARNING,
                    metric="F_trend",
                    value=monitor.F_history[-1],
                    threshold=monitor.F_history[0] * 1.1,
                    timestamp=0.0,
                    message="自由能上升趋势"
                ))
        return monitor, alerts

    def step(
        self,
        monitor: EvolutionMonitor,
        F: float,
        G: float,
        J: float
    ) -> Tuple[EvolutionMonitor, List[AlertEvent]]:
        """执行一步监控更新。

        Args:
            monitor: 当前监控状态
            F: 当前自由能
            G: 当前预期自由能
            J: 当前全局目标

        Returns:
            (更新后的监控状态, 新触发的告警列表)

        Examples:
            >>> monitor = EvolutionMonitorExecutor()
            >>> em = EvolutionMonitor()
            >>> new_em, alerts = monitor.step(em, 8.0, 5.0, 20.0)
            >>> new_em.current_F
            8.0
        """
        new_monitor = self.collect_metrics(monitor, F, G, J)
        new_monitor, alerts = self.check_alerts(new_monitor)
        return new_monitor, alerts
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `E2EValidation` | ← 阶段97 | 作为监控的验证基础（组合字段） |
| **输入** | `GlobalObjective` | ← 阶段91 | 指标计算的数据源 |
| **输出** | `EvolutionMonitor` | → 阶段99 | 进化监控状态，作为退化检测的输入 |
| **输出** | AlertEvent | → 阶段99 | CRITICAL级告警触发自修复 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| NumPy | 指标计算 | 滑动平均和趋势计算 |
| SQLite | 指标存储 | 持久化监控数据 |
| Prometheus | 指标暴露 | 可选，供Grafana可视化 |
| Grafana | 可视化 | 可选，监控面板展示 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V98.1 | 精度指标计算 | accuracy = 1 - F/F_initial | 计算测试 |
| V98.2 | 精度趋势 | F下降时accuracy上升 | 趋势测试 |
| V98.3 | 广度指标计算 | coverage = known/total | 计算测试 |
| V98.4 | 广度趋势 | 探索增加时coverage上升 | 趋势测试 |
| V98.5 | 能效指标计算 | efficiency = effective/total | 计算测试 |
| V98.6 | 能效趋势 | 优化后efficiency上升 | 趋势测试 |
| V98.7 | EVI计算 | α·acc + β·cov + γ·eff | 计算测试 |
| V98.8 | EVI权重 | α+β+γ = 1.0 | 权重测试 |
| V98.9 | EVI与J对应 | EVI↑ ↔ J↓ | 对应测试 |
| V98.10 | 数据采集 | 指标正确采集和存储 | 采集测试 |
| V98.11 | 数据保留 | 各粒度保留时长正确 | 保留测试 |
| V98.12 | 精度退化告警 | accuracy↓持续1000步触发WARNING | 告警测试 |
| V98.13 | EVI退化告警 | EVI_rate<0持续T触发CRITICAL | 告警测试 |
| V98.14 | 告警升级 | WARNING→CRITICAL→EMERGENCY | 升级测试 |
| V98.15 | CRITICAL触发自修复 | CRITICAL告警触发阶段99 | 对接测试 |
| V98.16 | EvolutionMonitor组合 | `state.e2e_validation` 类型为 `E2EValidation` | 类型检查 |
| V98.17 | 存储约束 | 监控数据存储<80%磁盘 | 存储测试 |
| V98.18 | 长期运行 | 72小时监控无异常 | 稳定性测试 |

---

> **阶段98完** | 阶段98建立了进化指标监控面板：精度指标 accuracy = 1 - F/F_initial（A98.1），广度指标 coverage = dim(Ψ_known)/dim(Ψ_total)（A98.2），能效指标 efficiency = effective_compute/total_compute（A98.3），综合进化指数 EVI = α·accuracy + β·coverage + γ·efficiency（A98.4），监控数据采集与存储（A98.5），异常告警规则（A98.6），封装为EvolutionMonitor数据类组合E2EValidation。EVI是系统进化的"成绩单"——精度40%+广度35%+能效25%，三项加权综合反映进化水平。七条告警规则守护系统健康，CRITICAL级告警直接触发自修复。下一步：阶段99将建立系统退化检测与自修复机制。
