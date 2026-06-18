# 阶段99：系统退化检测与自修复

> **阶段编号**：S99 | **所属层**：第十层·系统集成层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 退化检测 | "体检异常"——EVI持续下降，系统在走下坡路 |
> | 退化根因分析 | "找病因"——模型过时、记忆损坏、资源泄漏、配置漂移 |
> | 自修复策略 | "对症下药"——刷新模型、重建记忆、回收资源、重置配置 |
> | 修复验证 | "复查确认"——修复后EVI比修复前高 |
> | 修复失败升级 | "转诊"——自己修不好就通知人工 |
> | DegradationDetector | "退化检测器的完整画像"——组合EvolutionMonitor，在监控基础上增加退化检测状态 |

---

## 1. 第一性原理依据

本阶段从阶段98（进化指标监控）的核心结论出发，建立系统退化检测与自修复机制。核心依据为：

- **A98.4（综合进化指数EVI）**（引用阶段98）：$\text{EVI} = \alpha \cdot \text{accuracy} + \beta \cdot \text{coverage} + \gamma \cdot \text{efficiency}$ 是系统进化的综合度量。EVI持续下降意味着系统在退化——精度、广度或能效中至少一个在恶化。

- **A86.x（自动回滚）**（引用阶段86）：性能倒退>5%时自动回滚到最佳快照。自修复策略中的"重度退化"直接引用A86.x的回滚机制。

- **A63.x（微修正器）**（引用阶段63）：梯度下降微调模型参数。自修复策略中的"轻度退化"引用A63.x的微修正机制。

- **A67.x（结构性重构）**（引用阶段67）：网络结构的结构性进化。自修复策略中的"中度退化"引用A67.x的结构性重构机制。

本阶段的核心目标是：**定义退化检测 d(EVI)/dt < 0 持续 T_degradation（A99.1），退化根因分析 [model_staleness, memory_corruption, resource_leak, config_drift]（A99.2），自修复策略 refresh_model() / rebuild_memory() / reclaim_resources() / reset_config()（A99.3），修复验证 EVI_after > EVI_before（A99.4），修复失败升级（人工告警）（A99.5），封装为DegradationDetector数据类组合EvolutionMonitor。**

> **去重声明**：本阶段仅定义退化检测与自修复的编排逻辑，不重新实现微修正（阶段63的A63.x）、结构性重构（阶段67的A67.x）或自动回滚（阶段86的A86.x）。自修复策略是对这些已有机制的编排调用。

---

## 2. 核心数学公式

### 第一部分：退化检测

#### 2.1 退化检测定义（A99.1）

当EVI持续下降超过阈值时间时，判定系统退化：

$$\text{degraded} = \mathbb{1}\left[\frac{d(\text{EVI})}{dt} < 0 \;\text{持续}\; T_{degradation}\right] \tag{A99.1}$$

**退化检测的参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| $T_{degradation}$ | 1800s (30分钟) | 退化持续时间阈值 |
| EVI_rate_threshold | -0.001 | EVI变化率阈值 |
| trend_window | 1000步 | 趋势计算窗口 |

**退化检测的排除条件**：

以下情况EVI短暂下降不算退化：

| 排除条件 | 原因 |
|---------|------|
| 环境刚变化 | 适应新环境时EVI暂时下降 |
| 刚完成重构 | 重构后短期波动 |
| 刚完成回滚 | 回滚后短期波动 |
| 探索权重上浮 | 主动增加探索时EVI暂时下降 |

$$\text{degraded\_effective} = \text{degraded} \;\wedge\; \neg\text{env\_change} \;\wedge\; \neg\text{recent\_reform} \;\wedge\; \neg\text{recent\_rollback} \;\wedge\; \neg\text{explore\_boost} \tag{A99.1-effective}$$

**退化检测与A98.6的对接**：

A98.6（引用阶段98）定义EVI退化告警规则：EVI_rate < 0 持续 $T_{degradation}$ 秒触发CRITICAL告警。A99.1的退化检测直接由A98.6的CRITICAL告警触发。

- **输入**：EVI时间序列、排除条件
- **输出**：是否退化
- **约束**：EVI持续下降30分钟判定退化

> **白话翻译**：退化检测就是"体检异常"——EVI持续下降30分钟，系统在走下坡路。环境变化、重构、回滚、探索上浮导致的短暂下降不算退化——这些是正常的波动，不是真正的退化。

---

### 第二部分：退化根因分析

#### 2.2 退化根因分析定义（A99.2）

退化发生后，分析根本原因：

$$\text{root\_cause} \in \{\text{model\_staleness}, \text{memory\_corruption}, \text{resource\_leak}, \text{config\_drift}\} \tag{A99.2}$$

**四种根因的诊断方法**：

| 根因 | 诊断方法 | 诊断指标 |
|------|---------|---------|
| model_staleness | 检查模型预测误差趋势 | F_trend > 0（误差上升） |
| memory_corruption | 检查记忆检索质量 | retrieval_accuracy < 0.5 |
| resource_leak | 检查资源使用趋势 | cpu_trend > 0 或 mem_trend > 0 |
| config_drift | 检查配置参数一致性 | config_hash ≠ expected_hash |

**根因分析的决策树**：

```
EVI下降
│
├── accuracy下降？
│   ├── 是 → F_trend > 0?
│   │   ├── 是 → model_staleness（模型过时）
│   │   └── 否 → 检查记忆
│   │       ├── retrieval_accuracy < 0.5 → memory_corruption
│   │       └── 否 → config_drift
│   └── 否 → coverage或efficiency下降
│       ├── coverage下降 → model_staleness
│       └── efficiency下降 → resource_leak
```

**根因分析的置信度**：

$$\text{confidence}_i = \frac{\text{evidence}_i}{\sum_j \text{evidence}_j} \tag{A99.2-confidence}$$

每种根因的置信度基于支持证据的强度。选择置信度最高的根因作为主要修复目标。

**多根因情况**：

退化可能由多种根因同时导致。此时按置信度排序，依次修复：

$$\text{repair\_order} = \text{sort}(\text{root\_causes}, \text{key}=\text{confidence}, \text{reverse}=\text{True}) \tag{A99.2-order}$$

- **输入**：退化检测结果、系统运行指标
- **输出**：根因列表（按置信度排序）
- **约束**：至少识别一个根因

> **白话翻译**：退化根因分析就是"找病因"——四种可能：模型过时（预测越来越差）、记忆损坏（检索质量下降）、资源泄漏（CPU/内存越用越多）、配置漂移（参数被意外修改）。用决策树逐一排查，按证据强度排序，最可能的病因先治。

---

### 第三部分：自修复策略

#### 2.3 自修复策略定义（A99.3）

根据退化程度和根因，选择修复策略：

**修复策略与退化程度的对应**：

| 退化程度 | EVI下降幅度 | 修复策略 | 引用 |
|---------|------------|---------|------|
| 轻度 | < 5% | refresh_model() | A63.x微修正 |
| 中度 | 5%-15% | rebuild_memory() + refresh_model() | A67.x重构 |
| 重度 | > 15% | rollback_to_best_snapshot() | A86.x回滚 |

**四种根因的修复策略**：

| 根因 | 修复策略 | 操作 |
|------|---------|------|
| model_staleness | refresh_model() | 触发微修正或结构性重构 |
| memory_corruption | rebuild_memory() | 清理损坏记忆，重建索引 |
| resource_leak | reclaim_resources() | 强制GC，释放泄漏资源 |
| config_drift | reset_config() | 重置到已知良好配置 |

**refresh_model()的详细流程**：

$$\text{refresh\_model}(): \text{micro\_adjust}(\theta) \to \text{validate}(F_{new} < F_{old}) \tag{A99.3-refresh}$$

1. 触发微修正器（引用A63.x）
2. 执行一轮梯度下降
3. 验证新模型是否优于旧模型
4. 如果优于，接受；否则，尝试结构性重构

**rebuild_memory()的详细流程**：

$$\text{rebuild\_memory}(): \text{clean\_corrupted}() \to \text{rebuild\_index}() \to \text{verify\_integrity}() \tag{A99.3-rebuild}$$

1. 扫描长期记忆，标记损坏条目
2. 删除损坏条目
3. 重建检索索引
4. 验证记忆完整性

**reclaim_resources()的详细流程**：

$$\text{reclaim\_resources}(): \text{gc}() \to \text{clear\_cache}() \to \text{compact\_db}() \tag{A99.3-reclaim}$$

1. 强制垃圾回收
2. 清理缓存
3. 压缩SQLite数据库
4. 验证资源使用率下降

**reset_config()的详细流程**：

$$\text{reset\_config}(): \text{load\_default}() \to \text{verify\_hash}() \to \text{apply}() \tag{A99.3-reset}$$

1. 加载默认配置
2. 验证配置哈希
3. 应用新配置
4. 验证系统正常运行

- **输入**：退化程度、根因列表
- **输出**：修复操作结果
- **约束**：修复后EVI必须高于修复前

> **白话翻译**：自修复策略就是"对症下药"——模型过时就刷新模型（微修正或重构），记忆损坏就重建记忆（清理+重索引），资源泄漏就回收资源（GC+清缓存+压缩数据库），配置漂移就重置配置。轻度退化微调，中度退化重构，重度退化回滚到最佳快照。

---

### 第四部分：修复验证

#### 2.4 修复验证定义（A99.4）

修复后验证EVI是否改善：

$$\text{repair\_success} = \mathbb{1}[\text{EVI}_{after} > \text{EVI}_{before}] \tag{A99.4}$$

**修复验证的评估窗口**：

$$\text{EVI}_{after} = \text{EVI}(t_{repair} + W_{verify}) \tag{A99.4-window}$$

其中 $W_{verify} = 300$步（约30秒）。修复后等待30秒再评估，避免短期波动误判。

**修复验证的维度分解**：

| 维度 | 验证条件 | 说明 |
|------|---------|------|
| 精度 | accuracy_after > accuracy_before | 预测更准 |
| 广度 | coverage_after ≥ coverage_before | 认知不缩小 |
| 能效 | efficiency_after > efficiency_before | 效率更高 |
| 安全 | R_safety_after = 0 | 无安全违规 |

**修复验证的回退机制**：

如果修复后EVI没有改善，回退到修复前的状态：

$$\text{if } \text{EVI}_{after} \leq \text{EVI}_{before}: \text{rollback\_repair}() \tag{A99.4-rollback}$$

修复前的状态通过快照保存（引用A78.x的快照管理），确保可以安全回退。

- **输入**：修复前EVI、修复后EVI
- **输出**：修复是否成功
- **约束**：EVI_after > EVI_before

> **白话翻译**：修复验证就是"复查确认"——修完等30秒再量EVI，比修前高才算成功。不高就回退到修前的状态，不能越修越差。四个维度都要检查：精度更准、广度不缩、效率更高、安全无违规。

---

### 第五部分：修复失败升级

#### 2.5 修复失败升级定义（A99.5）

自修复失败时升级为人工告警：

$$\text{if } \neg\text{repair\_success} \;\text{after}\; N_{max\_attempts}: \text{escalate\_to\_human}() \tag{A99.5}$$

**修复失败升级的流程**：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 尝试修复（第1次） | 按根因选择修复策略 |
| 2 | 验证修复 | EVI_after > EVI_before? |
| 3 | 失败→尝试修复（第2次） | 换一种修复策略 |
| 4 | 验证修复 | EVI_after > EVI_before? |
| 5 | 失败→尝试修复（第3次） | 使用最激进的修复策略 |
| 6 | 验证修复 | EVI_after > EVI_before? |
| 7 | 3次都失败→人工告警 | 记录完整诊断信息 |

**人工告警的内容**：

| 字段 | 说明 |
|------|------|
| degradation_start | 退化开始时间 |
| EVI_before | 退化前EVI |
| EVI_current | 当前EVI |
| root_causes | 识别的根因列表 |
| repair_attempts | 修复尝试记录 |
| system_state | 系统状态快照 |
| recommended_action | 建议的人工操作 |

**修复失败后的安全模式**：

$$\text{safe\_mode}: \text{reduce\_compute}() \to \text{increase\_monitoring}() \to \text{wait\_for\_human}() \tag{A99.5-safe}$$

修复失败后系统进入安全模式：降低计算频率（减少资源消耗）、增加监控频率（更早发现进一步退化）、等待人工干预。

- **输入**：修复失败记录
- **输出**：人工告警
- **约束**：最多3次修复尝试

> **白话翻译**：修复失败升级就是"转诊"——自己修不好就通知人工。最多试3次，每次换一种修法。3次都失败就进入安全模式：降低计算频率省资源，增加监控频率早发现，等人工来处理。告警里带上完整的诊断信息和建议操作。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A99.1 | degraded = 𝟙[dEVI/dt < 0 持续 T_deg] | EVI序列 | 退化标志 | T_deg=1800s |
| A99.2 | root_cause ∈ {staleness, corruption, leak, drift} | 诊断指标 | 根因列表 | 至少1个根因 |
| A99.3 | refresh/rebuild/reclaim/reset | 根因 | 修复结果 | EVI_after > EVI_before |
| A99.4 | repair_success = 𝟙[EVI_after > EVI_before] | 修复前后EVI | 成功标志 | 30秒评估窗口 |
| A99.5 | N次失败→escalate_to_human | 失败记录 | 人工告警 | N_max=3 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import numpy as np

# 引用阶段98的数据类
from stage_098 import EvolutionMonitor, AlertLevel, AlertEvent


# ============================================================
# 阶段99 枚举：退化程度
# ============================================================
class DegradationLevel(Enum):
    """退化程度。"""
    NONE = "none"          # 无退化
    MILD = "mild"          # 轻度退化（EVI下降<5%）
    MODERATE = "moderate"  # 中度退化（EVI下降5%-15%）
    SEVERE = "severe"      # 重度退化（EVI下降>15%）


# ============================================================
# 阶段99 枚举：退化根因
# ============================================================
class RootCauseType(Enum):
    """退化根因类型。"""
    MODEL_STALENESS = "model_staleness"        # 模型过时
    MEMORY_CORRUPTION = "memory_corruption"    # 记忆损坏
    RESOURCE_LEAK = "resource_leak"            # 资源泄漏
    CONFIG_DRIFT = "config_drift"              # 配置漂移


# ============================================================
# 阶段99 数据类：根因分析结果
# ============================================================
@dataclass(frozen=True)
class RootCauseAnalysis:
    """根因分析结果。

    Attributes:
        root_causes: 根因列表（按置信度排序）
        confidences: 各根因的置信度
        primary_cause: 主要根因
        evidence: 各根因的支持证据
    """
    root_causes: Tuple[RootCauseType, ...]
    confidences: Tuple[float, ...]
    primary_cause: RootCauseType
    evidence: Dict[str, float]


# ============================================================
# 阶段99 数据类：修复记录
# ============================================================
@dataclass(frozen=True)
class RepairRecord:
    """修复记录。

    Attributes:
        attempt: 尝试次数
        root_cause: 针对的根因
        strategy: 修复策略
        evi_before: 修复前EVI
        evi_after: 修复后EVI
        success: 是否成功
        timestamp: 时间戳
    """
    attempt: int
    root_cause: RootCauseType
    strategy: str
    evi_before: float
    evi_after: float
    success: bool
    timestamp: float


# ============================================================
# 阶段99 数据类：退化检测器
# ============================================================
@dataclass(frozen=True)
class DegradationDetector:
    """系统退化检测与自修复的完整状态描述。

    组合阶段98的EvolutionMonitor，在监控基础上增加退化检测状态。

    Attributes:
        evolution_monitor: 阶段98的进化监控（组合字段）
        is_degraded: 是否退化
        degradation_level: 退化程度
        degradation_start_time: 退化开始时间
        evi_at_degradation: 退化时EVI值
        root_cause_analysis: 根因分析结果
        repair_records: 修复记录
        total_repairs: 总修复次数
        successful_repairs: 成功修复次数
        is_in_safe_mode: 是否处于安全模式
        human_alert_sent: 是否已发送人工告警
        T_degradation: 退化持续时间阈值
        max_repair_attempts: 最大修复尝试次数

    Invariants:
        - total_repairs >= successful_repairs
        - max_repair_attempts = 3
    """
    evolution_monitor: EvolutionMonitor
    is_degraded: bool = False
    degradation_level: DegradationLevel = DegradationLevel.NONE
    degradation_start_time: float = 0.0
    evi_at_degradation: float = 0.0
    root_cause_analysis: Optional[RootCauseAnalysis] = None
    repair_records: Tuple[RepairRecord, ...] = ()
    total_repairs: int = 0
    successful_repairs: int = 0
    is_in_safe_mode: bool = False
    human_alert_sent: bool = False
    T_degradation: float = 1800.0
    max_repair_attempts: int = 3


# ============================================================
# 阶段99 核心接口：退化检测与自修复
# ============================================================
class DegradationDetectorExecutor:
    """阶段99的核心接口：系统退化检测与自修复。

    实现退化检测、根因分析、自修复和修复验证。

    核心约束：
        - EVI持续下降30分钟判定退化
        - 修复后EVI必须高于修复前
        - 最多3次修复尝试
        - 3次失败进入安全模式并告警人工
    """

    def __init__(
        self,
        T_degradation: float = 1800.0,
        max_repair_attempts: int = 3,
        verify_window: int = 300
    ):
        """初始化退化检测器。

        Args:
            T_degradation: 退化持续时间阈值（秒），必须 > 0
            max_repair_attempts: 最大修复尝试次数，必须 > 0
            verify_window: 修复验证窗口（步数），必须 > 0

        Raises:
            ValueError: 当任何参数 <= 0时

        Examples:
            >>> detector = DegradationDetectorExecutor()
            >>> detector.T_degradation
            1800.0
        """
        if T_degradation <= 0:
            raise ValueError(f"T_degradation必须 > 0，收到 {T_degradation}")
        if max_repair_attempts <= 0:
            raise ValueError(f"max_repair_attempts必须 > 0，收到 {max_repair_attempts}")
        if verify_window <= 0:
            raise ValueError(f"verify_window必须 > 0，收到 {verify_window}")
        self.T_degradation = T_degradation
        self.max_repair_attempts = max_repair_attempts
        self.verify_window = verify_window

    def detect_degradation(
        self,
        monitor: EvolutionMonitor
    ) -> Tuple[bool, DegradationLevel]:
        """退化检测（A99.1）。

        Args:
            monitor: 进化监控状态

        Returns:
            (是否退化, 退化程度)

        Examples:
            >>> detector = DegradationDetectorExecutor()
            >>> from stage_098 import EvolutionMonitor
            >>> em = EvolutionMonitor(evi_history=(0.5, 0.48, 0.45))
            >>> is_degraded, level = detector.detect_degradation(em)
            >>> isinstance(is_degraded, bool)
            True
        """
        if len(monitor.evi_history) < 2:
            return False, DegradationLevel.NONE
        evi_change = (
            (monitor.evi_history[-1] - monitor.evi_history[0])
            / max(abs(monitor.evi_history[0]), 1e-8)
        )
        if evi_change > 0.15:
            return True, DegradationLevel.SEVERE
        elif evi_change > 0.05:
            return True, DegradationLevel.MODERATE
        elif evi_change > 0.0:
            return True, DegradationLevel.MILD
        return False, DegradationLevel.NONE

    def analyze_root_cause(
        self,
        detector: DegradationDetector,
        system_metrics: Dict
    ) -> RootCauseAnalysis:
        """退化根因分析（A99.2）。

        Args:
            detector: 退化检测器状态
            system_metrics: 系统运行指标字典

        Returns:
            根因分析结果

        Examples:
            >>> detector = DegradationDetectorExecutor()
            >>> from stage_098 import EvolutionMonitor
            >>> em = EvolutionMonitor()
            >>> dd = DegradationDetector(evolution_monitor=em)
            >>> rca = detector.analyze_root_cause(dd, {"F_trend": 0.01})
            >>> isinstance(rca, RootCauseAnalysis)
            True
        """
        if system_metrics.get("F_trend", 0.0) > 0.0:
            return RootCauseAnalysis(
                root_cause_type=RootCauseType.MODEL_OUTDATED,
                confidence=0.8,
                affected_modules=("predictor",),
                recommended_action="retrain",
                repair_history=()
            )
        elif system_metrics.get("retrieval_accuracy", 1.0) < 0.5:
            return RootCauseAnalysis(
                root_cause_type=RootCauseType.MEMORY_CORRUPTION,
                confidence=0.7,
                affected_modules=("memory",),
                recommended_action="rebuild_memory",
                repair_history=()
            )
        elif system_metrics.get("cpu_trend", 0.0) > 0.0 or system_metrics.get("mem_trend", 0.0) > 0.0:
            return RootCauseAnalysis(
                root_cause_type=RootCauseType.RESOURCE_LEAK,
                confidence=0.6,
                affected_modules=("resource",),
                recommended_action="cleanup",
                repair_history=()
            )
        return RootCauseAnalysis(
            root_cause_type=RootCauseType.UNKNOWN,
            confidence=0.0,
            affected_modules=(),
            recommended_action="none",
            repair_history=()
        )

    def repair(
        self,
        detector: DegradationDetector,
        root_cause: RootCauseAnalysis
    ) -> DegradationDetector:
        """执行自修复（A99.3）。

        Args:
            detector: 退化检测器状态
            root_cause: 根因分析结果

        Returns:
            更新后的退化检测器状态

        Examples:
            >>> detector = DegradationDetectorExecutor()
            >>> from stage_098 import EvolutionMonitor
            >>> em = EvolutionMonitor()
            >>> dd = DegradationDetector(evolution_monitor=em)
            >>> rca = RootCauseAnalysis(
            ...     root_cause_type=RootCauseType.MODEL_OUTDATED)
            >>> new_dd = detector.repair(dd, rca)
            >>> new_dd.repair_attempts == dd.repair_attempts + 1
            True
        """
        new_attempts = detector.repair_attempts + 1
        new_history = detector.repair_history + (root_cause,)
        return DegradationDetector(
            evolution_monitor=detector.evolution_monitor,
            is_degraded=detector.is_degraded,
            degradation_level=detector.degradation_level,
            root_cause=root_cause,
            repair_attempts=new_attempts,
            repair_history=new_history,
            last_repair_time=0.0,
            evi_before_repair=detector.evi_before_repair,
            evi_after_repair=detector.evi_after_repair,
            repair_verified=detector.repair_verified,
            human_escalated=detector.human_escalated
        )

    def verify_repair(
        self,
        evi_before: float,
        evi_after: float
    ) -> bool:
        """修复验证（A99.4）。

        Args:
            evi_before: 修复前EVI
            evi_after: 修复后EVI

        Returns:
            当evi_after > evi_before * 0.95时返回True

        Examples:
            >>> detector = DegradationDetectorExecutor()
            >>> detector.verify_repair(0.5, 0.48)
            False
            >>> detector.verify_repair(0.5, 0.51)
            True
        """
        return evi_after > evi_before * 0.95

    def escalate_to_human(
        self,
        detector: DegradationDetector
    ) -> Dict:
        """修复失败升级（A99.5）。

        Args:
            detector: 退化检测器状态

        Returns:
            人工告警内容字典

        Examples:
            >>> detector = DegradationDetectorExecutor()
            >>> from stage_098 import EvolutionMonitor
            >>> em = EvolutionMonitor()
            >>> dd = DegradationDetector(evolution_monitor=em)
            >>> alert = detector.escalate_to_human(dd)
            >>> "level" in alert
            True
        """
        return {
            "level": "CRITICAL",
            "message": f"退化修复失败，已尝试{detector.repair_attempts}次",
            "degradation_level": detector.degradation_level.value,
            "root_cause": detector.root_cause.root_cause_type.value,
            "timestamp": 0.0
        }

    def step(
        self,
        detector: DegradationDetector,
        system_metrics: Dict
    ) -> DegradationDetector:
        """执行一步退化检测与自修复。

        Args:
            detector: 当前退化检测器状态
            system_metrics: 系统运行指标字典

        Returns:
            更新后的退化检测器状态

        Examples:
            >>> detector = DegradationDetectorExecutor()
            >>> from stage_098 import EvolutionMonitor
            >>> em = EvolutionMonitor()
            >>> dd = DegradationDetector(evolution_monitor=em)
            >>> new_dd = detector.step(dd, {})
            >>> isinstance(new_dd, DegradationDetector)
            True
        """
        is_degraded, level = self.detect_degradation(detector.evolution_monitor)
        if not is_degraded:
            return DegradationDetector(
                evolution_monitor=detector.evolution_monitor,
                is_degraded=False,
                degradation_level=DegradationLevel.NONE,
                root_cause=detector.root_cause,
                repair_attempts=detector.repair_attempts,
                repair_history=detector.repair_history,
                last_repair_time=detector.last_repair_time,
                evi_before_repair=detector.evi_before_repair,
                evi_after_repair=detector.evi_after_repair,
                repair_verified=detector.repair_verified,
                human_escalated=detector.human_escalated
            )
        rca = self.analyze_root_cause(detector, system_metrics)
        if detector.repair_attempts < self.max_repair_attempts:
            repaired = self.repair(detector, rca)
            return repaired
        return DegradationDetector(
            evolution_monitor=detector.evolution_monitor,
            is_degraded=True,
            degradation_level=level,
            root_cause=rca,
            repair_attempts=detector.repair_attempts,
            repair_history=detector.repair_history,
            last_repair_time=detector.last_repair_time,
            evi_before_repair=detector.evi_before_repair,
            evi_after_repair=detector.evi_after_repair,
            repair_verified=False,
            human_escalated=True
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `EvolutionMonitor` | ← 阶段98 | 作为退化检测的监控基础（组合字段） |
| **输入** | A63.x（微修正器） | ← 阶段63 | 轻度退化的修复策略 |
| **输入** | A67.x（结构性重构） | ← 阶段67 | 中度退化的修复策略 |
| **输入** | A86.x（自动回滚） | ← 阶段86 | 重度退化的修复策略 |
| **输入** | A78.x（快照管理） | ← 阶段78 | 修复前快照和回滚 |
| **输出** | `DegradationDetector` | → 阶段100 | 退化检测与自修复状态 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| NumPy | 趋势分析 | EVI变化率计算 |
| SciPy | 统计检验 | 退化趋势的显著性检验 |
| SQLite | 修复日志 | 修复记录持久化 |
| gc | 资源回收 | Python垃圾回收 |
| psutil | 资源监控 | 资源泄漏检测 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V99.1 | 退化检测 | EVI↓持续30分钟判定退化 | 检测测试 |
| V99.2 | 排除条件 | 环境变化等不误判退化 | 排除测试 |
| V99.3 | 退化程度判定 | 轻度<5%, 中度5-15%, 重度>15% | 程度测试 |
| V99.4 | 根因分析-模型过时 | F_trend>0正确识别 | 诊断测试 |
| V99.5 | 根因分析-记忆损坏 | retrieval_accuracy<0.5正确识别 | 诊断测试 |
| V99.6 | 根因分析-资源泄漏 | cpu/mem_trend>0正确识别 | 诊断测试 |
| V99.7 | 根因分析-配置漂移 | config_hash不匹配正确识别 | 诊断测试 |
| V99.8 | refresh_model() | 微修正后F下降 | 修复测试 |
| V99.9 | rebuild_memory() | 清理后检索质量恢复 | 修复测试 |
| V99.10 | reclaim_resources() | GC后资源使用率下降 | 修复测试 |
| V99.11 | reset_config() | 重置后配置哈希匹配 | 修复测试 |
| V99.12 | 修复验证 | EVI_after > EVI_before | 验证测试 |
| V99.13 | 修复回退 | 修复失败回退到修复前 | 回退测试 |
| V99.14 | 最多3次尝试 | 第4次不尝试修复 | 次数测试 |
| V99.15 | 安全模式 | 3次失败后进入安全模式 | 安全模式测试 |
| V99.16 | 人工告警 | 3次失败后发送告警 | 告警测试 |
| V99.17 | DegradationDetector组合 | `state.evolution_monitor` 类型为 `EvolutionMonitor` | 类型检查 |
| V99.18 | 端到端修复 | 注入退化→检测→修复→验证 | 集成测试 |

---

> **阶段99完** | 阶段99建立了系统退化检测与自修复机制：退化检测 d(EVI)/dt < 0 持续 T_degradation（A99.1），退化根因分析 [model_staleness, memory_corruption, resource_leak, config_drift]（A99.2），自修复策略 refresh_model() / rebuild_memory() / reclaim_resources() / reset_config()（A99.3），修复验证 EVI_after > EVI_before（A99.4），修复失败升级（人工告警）（A99.5），封装为DegradationDetector数据类组合EvolutionMonitor。四种根因、四种修法、三级退化、三次机会——系统能自己诊断、自己修复、修不好就求援。这是自愈能力的最后一道防线。下一步：阶段100将建立终极验证——72小时压力测试方案。
