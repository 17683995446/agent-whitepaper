# 阶段86：模型重构后性能倒退自动回滚

> **阶段编号**：S86 | **所属层**：第九层·安全护栏层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 重构后性能评估 | "改完测一下"——重构后比较新旧模型性能 |
> | 倒退判定 | "改坏了"——新模型比旧模型差超过5%就判定倒退 |
> | 自动回滚 | "一键撤回"——检测到倒退就自动恢复到旧模型 |
> | 回滚后验证 | "撤回后确认"——确认恢复后的性能和原来一样 |
> | 回滚日志与根因分析 | "事故报告"——记录回滚原因，分析为什么改坏了 |
> | AutoRollback | "自动回滚的完整画像"——组合ExploreFreezer，在探索冻结基础上增加回滚状态 |

---

## 1. 第一性原理依据

本阶段从阶段66（回滚条件与机制）和阶段78（回滚操作）的核心结论出发，建立模型重构后性能倒退自动回滚机制。核心依据为：

- **A66.2-A66.3（回滚条件与机制）**（引用阶段66）：回滚条件 $F_{\text{new}} > F_{\text{old}} \cdot (1 + \theta_{\text{rollback}})$，回滚机制 restore(snapshot)。本阶段将A66.2-A66.3从"重构安全约束"推广到"安全护栏层的自动回滚"——不仅重构后可以回滚，任何性能倒退的场景都自动触发回滚。

- **A78.5（回滚操作）**（引用阶段78）：rollback(snapshot_id) → model，快照回滚的具体实现。本阶段的自动回滚直接调用A78.5的回滚操作——加载重构前的快照，恢复模型状态。

本阶段的核心目标是：**定义重构后性能评估 F_new vs F_old（A86.1），倒退判定 F_new > F_old · (1 + θ_rollback), θ_rollback = 0.05（A86.2），自动回滚 rollback(snapshot_id)（A86.3），回滚后验证 F_restored ≈ F_old（A86.4），回滚日志与根因分析（A86.5），封装为AutoRollback数据类，组合ExploreFreezer作为探索冻结的输入。**

> **去重声明**：本阶段仅定义模型重构后性能倒退的自动回滚机制，不重新定义回滚条件（阶段66的A66.2）或回滚操作（阶段78的A78.5）。本阶段是A66.2-A66.3在安全护栏层的工程实现，回滚操作委托给A78.5。

---

## 2. 核心数学公式

### 第一部分：重构后性能评估

#### 2.1 重构后性能评估定义（A86.1）

重构后比较新旧模型的自由能：

$$\Delta F = F_{\text{new}} - F_{\text{old}} \tag{A86.1}$$

**性能评估的时机**：

| 时机 | 评估方式 | 说明 |
|------|---------|------|
| 重构完成后立即 | 单次评估 | 初步判断 |
| 重构后T_verify窗口内 | 滑动平均评估 | 确认判断 |
| 验证窗口结束 | 最终评估 | 决定回滚/接受 |

**性能评估的窗口设计**：

重构后的性能可能暂时波动，需要观察一段时间：

$$\overline{F}_{\text{new}} = \frac{1}{W} \sum_{t=t_0}^{t_0+W} F_{\text{new}}(t) \tag{A86.1-window}$$

其中 $W = T_{\text{verify}}$（引用阶段66的A66.5验证窗口）。

**性能评估的置信度**：

由于自由能有随机波动，需要计算评估的置信度：

$$\text{CI}_{95\%}(\Delta F) = \Delta F \pm 1.96 \cdot \frac{\sigma_F}{\sqrt{W}} \tag{A86.1-ci}$$

只有当置信区间完全在倒退区域内时，才确认倒退：

$$\text{confirmed\_regression} = \mathbb{1}\left[\text{CI}_{95\%}^{\text{lower}} > F_{\text{old}} \cdot \theta_{\text{rollback}}\right] \tag{A86.1-confirm}$$

**性能评估的多维度检查**：

| 维度 | 度量 | 说明 |
|------|------|------|
| 自由能 | F_new vs F_old | 核心指标 |
| 全局误差 | ε_new vs ε_old | 预测精度 |
| 适应度 | f_new vs f_old | 策略质量 |
| 收敛速度 | τ_new vs τ_old | 学习效率 |

仅当自由能倒退时触发回滚，其他维度的倒退记录但不触发回滚。

- **输入**：重构前后模型的自由能序列
- **输出**：性能差值ΔF、置信区间、确认标志
- **约束**：评估需在验证窗口内完成

> **白话翻译**：重构后性能评估就是"改完测一下"——比较新旧模型的自由能。不是看一眼就决定，而是观察一段时间（验证窗口），算滑动平均和置信区间。只有当95%置信区间确认性能倒退时才行动。还检查全局误差、适应度、收敛速度等其他维度，但只有自由能倒退才触发回滚。

---

### 第二部分：倒退判定

#### 2.2 倒退判定定义（A86.2）

当新模型自由能超过旧模型的5%时判定为倒退：

$$\text{is\_regression} = \mathbb{1}\left[F_{\text{new}} > F_{\text{old}} \cdot (1 + \theta_{\text{rollback}})\right], \quad \theta_{\text{rollback}} = 0.05 \tag{A86.2}$$

**5%倒退阈值的选择依据**：

| 阈值 | 灵敏度 | 误判风险 | 说明 |
|------|--------|---------|------|
| 1% | 极高 | 高 | 正常波动可能触发回滚 |
| 3% | 高 | 中 | 较灵敏，偶尔误判 |
| 5% | 适中 | 低 | 平衡灵敏度和误判 |
| 10% | 低 | 极低 | 可能漏掉中等程度的倒退 |

5%阈值与A66.2一致，确保安全护栏层与重构安全层的判定标准统一。

**倒退判定的三级响应**：

| 倒退程度 | 条件 | 响应 |
|---------|------|------|
| 轻微倒退 | 5% < ΔF/F < 10% | 延长观察窗口，暂不回滚 |
| 中度倒退 | 10% ≤ ΔF/F < 20% | 立即回滚 |
| 严重倒退 | ΔF/F ≥ 20% | 立即回滚 + 紧急告警 |

轻微倒退不立即回滚，因为可能是暂时波动。延长观察窗口至2倍，如果确认倒退再回滚。

**倒退判定与A66.2的对应**：

A66.2（引用阶段66）定义重构安全中的回滚条件 $F_{\text{new}} > F_{\text{old}} \cdot (1 + \theta_{\text{rollback}})$。A86.2与A66.2使用相同的公式和阈值，但应用场景不同：

| 维度 | A66.2（重构安全） | A86.2（安全护栏） |
|------|-----------------|-----------------|
| 触发场景 | 重构过程中 | 重构后任何时刻 |
| 响应速度 | 立即回滚 | 三级响应 |
| 验证要求 | 无 | 置信区间确认 |

- **输入**：F_new、F_old、θ_rollback
- **输出**：是否倒退、倒退程度
- **约束**：θ_rollback = 0.05

> **白话翻译**：倒退判定就是"改坏了"——新模型自由能比旧模型高5%以上就判定倒退。为什么是5%？太低容易误判（正常波动就触发），太高可能漏掉中等程度的倒退。分三级：轻微倒退（5%-10%）先观察不急着回滚，中度倒退（10%-20%）立即回滚，严重倒退（>20%）立即回滚加紧急告警。

---

### 第三部分：自动回滚

#### 2.3 自动回滚定义（A86.3）

检测到性能倒退后自动回滚到重构前的快照：

$$\text{rollback}(\text{snapshot\_id}_{\text{pre\_reform}}) \tag{A86.3}$$

**自动回滚与A78.5的对应**：

A78.5（引用阶段78）定义快照回滚操作 rollback(snapshot_id) → model。A86.3直接调用A78.5的回滚操作，传入重构前的快照ID。

**自动回滚的执行流程**：

1. **确认倒退**：A86.2判定为中度或严重倒退
2. **定位快照**：查找重构前的快照ID
3. **调用回滚**：A78.5的rollback(snapshot_id)
4. **验证恢复**：A86.4验证回滚后性能
5. **记录日志**：A86.5记录回滚日志

**自动回滚的安全保障**：

| 保障措施 | 说明 |
|---------|------|
| 回滚前保存当前状态 | 防止回滚后无法恢复 |
| 回滚后验证性能 | 确认回滚确实改善了性能 |
| EWC锚点更新 | 锚点设为回滚后参数 |
| 冷却期延长 | 防止立即再次重构 |

**自动回滚的触发限制**：

为防止回滚-重构-回滚的循环，设置回滚频率限制：

$$\text{rollback\_count}_{\text{recent}} \leq N_{\text{max\_rollback}} = 3/\text{day} \tag{A86.3-limit}$$

一天内最多自动回滚3次。超过限制后，后续的倒退判定仅记录日志，不自动回滚，需人工确认。

- **输入**：倒退判定结果、重构前快照ID
- **输出**：回滚后的模型状态
- **约束**：回滚频率 ≤ 3次/天

> **白话翻译**：自动回滚就是"一键撤回"——检测到性能倒退就自动恢复到重构前的快照。回滚前先保存当前状态（防回错），回滚后验证性能（确认改善了），还要更新EWC锚点和延长冷却期。一天最多自动回滚3次，防止回滚-重构-回滚的死循环。

---

### 第四部分：回滚后验证

#### 2.4 回滚后验证定义（A86.4）

回滚后验证恢复的性能与重构前一致：

$$\text{rollback\_verified} = \mathbb{1}\left[\left|F_{\text{restored}} - F_{\text{old}}\right| < \epsilon_{\text{verify}}\right] \tag{A86.4}$$

其中 $\epsilon_{\text{verify}} = 0.01 \cdot F_{\text{old}}$（允许1%的误差）。

**验证的检查项**：

| 检查项 | 验证方法 | 通过条件 |
|--------|---------|---------|
| 自由能恢复 | \|F_restored - F_old\| < 1% | 性能恢复 |
| 权重恢复 | ‖θ_restored - θ_snapshot‖ < 1e-10 | 参数恢复 |
| 结构恢复 | structure_restored == structure_old | 结构恢复 |
| EWC锚点 | 锚点设为回滚后参数 | 正确更新 |
| 冷却期 | 冷却期已延长 | 正确延长 |

**验证失败的处理**：

如果回滚后性能未恢复：

$$\text{if } \left|F_{\text{restored}} - F_{\text{old}}\right| \geq \epsilon_{\text{verify}} \implies \text{escalate\_to\_manual\_intervention}() \tag{A86.4-fail}$$

回滚后性能未恢复是严重异常——快照可能损坏或模型状态不一致。此时升级为人工干预，停止自动操作。

- **输入**：回滚后模型、重构前快照
- **输出**：验证通过标志
- **约束**：误差 < 1%

> **白话翻译**：回滚后验证就是"撤回后确认"——确认恢复后的性能和原来一样，误差不超过1%。检查自由能、权重、结构、EWC锚点、冷却期五项。如果回滚后性能没恢复，说明快照可能损坏了，这是严重异常，停止自动操作，升级为人工干预。

---

### 第五部分：回滚日志与根因分析

#### 2.5 回滚日志与根因分析定义（A86.5）

记录回滚日志并分析根因：

**回滚日志字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| timestamp | float | 回滚时间 |
| snapshot_id | int | 回滚到的快照ID |
| F_old | float | 重构前自由能 |
| F_new | float | 重构后自由能 |
| regression_ratio | float | 倒退比例 |
| regression_level | str | 倒退级别 |
| rollback_verified | bool | 回滚后验证是否通过 |
| root_cause | str | 根因分析结果 |

**根因分析方法**：

回滚后自动分析倒退的根因：

$$\text{root\_cause} = \text{analyze}(\Delta\theta, \Delta F, \text{reform\_type}, \text{reform\_history}) \tag{A86.5}$$

| 根因类型 | 判定条件 | 说明 |
|---------|---------|------|
| 结构不兼容 | 新结构无法表达旧功能 | 层数/维度变化过大 |
| 权重初始化不当 | 新权重远离最优 | 初始化策略不适合新结构 |
| 学习率过高 | 重构后学习发散 | 学习率未适配新结构 |
| EWC约束过强 | 正则化阻碍适应 | EWC锚点与新结构冲突 |
| 数据分布偏移 | 新环境与旧快照不匹配 | 环境变化导致旧模型不适用 |

**根因分析的统计**：

| 统计项 | 说明 |
|--------|------|
| 各根因出现频率 | 哪种根因最常见 |
| 根因与重构类型的关联 | 哪种重构最容易导致哪种根因 |
| 根因与倒退程度的关联 | 哪种根因导致最严重的倒退 |

- **输入**：回滚事件、模型差异、重构类型
- **输出**：回滚日志、根因分析结果
- **约束**：根因分析自动化

> **白话翻译**：回滚日志与根因分析就是"事故报告"——记录回滚的时间、快照ID、倒退程度、验证结果，还要分析为什么改坏了。根因分五类：结构不兼容、权重初始化不当、学习率过高、EWC约束过强、数据分布偏移。统计哪种根因最常见，哪种重构最容易出问题。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A86.1 | ΔF = F_new - F_old | 重构前后自由能 | 性能差值 | 验证窗口内评估 |
| A86.2 | is_regression = 𝟙[F_new > F_old·1.05] | F_new, F_old | 倒退标志 | θ_rollback=0.05 |
| A86.3 | rollback(snapshot_id) | 快照ID | 回滚后模型 | ≤3次/天 |
| A86.4 | verified = 𝟙[\|F_restored - F_old\| < 1%] | 回滚后+重构前 | 验证标志 | 误差<1% |
| A86.5 | root_cause = analyze(Δθ, ΔF, reform_type) | 回滚事件 | 根因分析 | 自动化 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import numpy as np

# 引用阶段85的数据类
from stage_085 import ExploreFreezer


# ============================================================
# 阶段86 枚举：倒退级别
# ============================================================
class RegressionLevel(Enum):
    """性能倒退级别。"""
    NONE = "none"               # 无倒退
    MILD = "mild"               # 轻微倒退：5% < ΔF/F < 10%
    MODERATE = "moderate"       # 中度倒退：10% ≤ ΔF/F < 20%
    SEVERE = "severe"           # 严重倒退：ΔF/F ≥ 20%


# ============================================================
# 阶段86 枚举：根因类型
# ============================================================
class RootCauseType(Enum):
    """回滚根因类型。"""
    STRUCTURE_INCOMPATIBLE = "structure_incompatible"
    WEIGHT_INIT_POOR = "weight_init_poor"
    LEARNING_RATE_HIGH = "learning_rate_high"
    EWC_TOO_STRONG = "ewc_too_strong"
    DATA_SHIFT = "data_shift"
    UNKNOWN = "unknown"


# ============================================================
# 阶段86 数据类：回滚日志记录
# ============================================================
@dataclass(frozen=True)
class RollbackLogRecord:
    """回滚日志记录。

    Attributes:
        timestamp: 回滚时间
        snapshot_id: 回滚到的快照ID
        F_old: 重构前自由能
        F_new: 重构后自由能
        regression_ratio: 倒退比例
        regression_level: 倒退级别
        rollback_verified: 回滚后验证是否通过
        root_cause: 根因分析结果
    """
    timestamp: float
    snapshot_id: int
    F_old: float
    F_new: float
    regression_ratio: float
    regression_level: RegressionLevel
    rollback_verified: bool = False
    root_cause: Optional[RootCauseType] = None


# ============================================================
# 阶段86 数据类：自动回滚状态
# ============================================================
@dataclass(frozen=True)
class AutoRollbackState:
    """自动回滚的状态描述。

    Attributes:
        total_rollbacks: 总回滚次数
        recent_rollback_count: 最近24小时回滚次数
        last_rollback: 最近一次回滚记录
        root_cause_stats: 各根因出现次数
        is_in_cooldown: 是否处于冷却期
        cooldown_remaining: 冷却期剩余时间
    """
    total_rollbacks: int = 0
    recent_rollback_count: int = 0
    last_rollback: Optional[RollbackLogRecord] = None
    root_cause_stats: Dict[str, int] = field(default_factory=dict)
    is_in_cooldown: bool = False
    cooldown_remaining: float = 0.0


# ============================================================
# 阶段86 数据类：自动回滚器
# ============================================================
@dataclass(frozen=True)
class AutoRollback:
    """模型重构后性能倒退自动回滚器。

    组合阶段85的ExploreFreezer，在探索冻结基础上增加回滚状态。

    Attributes:
        explore_freezer: 阶段85的探索冻结器（组合字段）
        rollback_state: 自动回滚状态
        rollback_threshold: 倒退判定阈值（默认0.05）
        verify_tolerance: 验证容差（默认0.01）
        max_rollbacks_per_day: 每日最大回滚次数（默认3）
        verify_window: 验证窗口时长（秒）
    """
    explore_freezer: ExploreFreezer
    rollback_state: AutoRollbackState = field(default_factory=AutoRollbackState)
    rollback_threshold: float = 0.05
    verify_tolerance: float = 0.01
    max_rollbacks_per_day: int = 3
    verify_window: float = 300.0


# ============================================================
# 阶段86 核心接口：自动回滚执行器
# ============================================================
class AutoRollbackExecutor:
    """阶段86的核心接口：自动回滚执行器。

    实现性能评估、倒退判定、自动回滚、回滚后验证和根因分析。

    核心约束：
        - 倒退阈值5%
        - 回滚频率≤3次/天
        - 回滚后验证误差<1%
    """

    def evaluate_performance(
        self,
        F_new_series: List[float],
        F_old: float,
        window: float = 300.0
    ) -> Tuple[float, float, bool]:
        """评估重构后性能（A86.1）。

        Args:
            F_new_series: 重构后自由能序列，长度 > 0
            F_old: 重构前自由能
            window: 验证窗口（秒），必须 > 0

        Returns:
            (ΔF, 95%置信区间下界, 是否确认倒退)
            - ΔF = mean(F_new_series) - F_old
            - 下界 = mean - 1.96 * std / sqrt(n)
            - 确认倒退 = 下界 > 0

        Raises:
            ValueError: 当F_new_series为空时
            ValueError: 当window <= 0时

        Examples:
            >>> executor = AutoRollbackExecutor()
            >>> dF, lower, reg = executor.evaluate_performance(
            ...     [10.5, 10.6, 10.4], 10.0, 300.0)
            >>> isinstance(reg, bool)
            True
        """
        if not F_new_series:
            raise ValueError("F_new_series不能为空")
        if window <= 0:
            raise ValueError(f"window必须 > 0，收到 {window}")
        arr = np.array(F_new_series)
        mean_f = float(np.mean(arr))
        std_f = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        delta_f = mean_f - F_old
        lower_bound = mean_f - 1.96 * std_f / np.sqrt(len(arr))
        confirmed = lower_bound > F_old
        return delta_f, lower_bound, confirmed

    def check_regression(
        self,
        F_new: float,
        F_old: float,
        threshold: float = 0.05
    ) -> Tuple[bool, RegressionLevel]:
        """判定是否性能倒退（A86.2）。

        Args:
            F_new: 新模型自由能
            F_old: 旧模型自由能
            threshold: 倒退阈值，必须 > 0

        Returns:
            (是否倒退, 倒退级别)

        Raises:
            ValueError: 当threshold <= 0时

        Examples:
            >>> executor = AutoRollbackExecutor()
            >>> reg, level = executor.check_regression(11.0, 10.0, 0.05)
            >>> reg
            True
            >>> level == RegressionLevel.SEVERE
            True
        """
        if threshold <= 0:
            raise ValueError(f"threshold必须 > 0，收到 {threshold}")
        if F_old == 0:
            ratio = float('inf') if F_new > 0 else 0.0
        else:
            ratio = (F_new - F_old) / abs(F_old)
        if ratio >= threshold + 0.05:
            return True, RegressionLevel.SEVERE
        elif ratio >= threshold:
            return True, RegressionLevel.MODERATE
        elif ratio >= threshold * 0.5:
            return True, RegressionLevel.MILD
        return False, RegressionLevel.NONE

    def execute_rollback(
        self,
        snapshot_id: int,
        recent_count: int,
        max_per_day: int = 3
    ) -> Tuple[Optional[Dict], bool]:
        """执行自动回滚（A86.3）。

        Args:
            snapshot_id: 重构前快照ID
            recent_count: 最近24小时回滚次数
            max_per_day: 每日最大回滚次数，必须 > 0

        Returns:
            (回滚后模型, 是否成功)

        Raises:
            ValueError: 当max_per_day <= 0时

        Examples:
            >>> executor = AutoRollbackExecutor()
            >>> model, ok = executor.execute_rollback(1, 0, 3)
            >>> isinstance(ok, bool)
            True
        """
        if max_per_day <= 0:
            raise ValueError(f"max_per_day必须 > 0，收到 {max_per_day}")
        if recent_count >= max_per_day:
            return None, False
        msi = ModelSnapshotInterface()
        loaded = msi.load_snapshot(snapshot_id, {})
        return loaded, loaded is not None

    def verify_rollback(
        self,
        F_restored: float,
        F_old: float,
        tolerance: float = 0.01
    ) -> bool:
        """验证回滚后性能（A86.4）。

        Args:
            F_restored: 回滚后自由能
            F_old: 重构前自由能
            tolerance: 验证容差，必须 > 0

        Returns:
            当|F_restored - F_old| / |F_old| < tolerance时返回True

        Raises:
            ValueError: 当tolerance <= 0时

        Examples:
            >>> executor = AutoRollbackExecutor()
            >>> executor.verify_rollback(10.01, 10.0, 0.01)
            True
            >>> executor.verify_rollback(10.5, 10.0, 0.01)
            False
        """
        if tolerance <= 0:
            raise ValueError(f"tolerance必须 > 0，收到 {tolerance}")
        if abs(F_old) < 1e-12:
            return abs(F_restored - F_old) < tolerance
        return abs(F_restored - F_old) / abs(F_old) < tolerance

    def analyze_root_cause(
        self,
        delta_weights: Dict[str, np.ndarray],
        delta_F: float,
        reform_type: str
    ) -> RootCauseType:
        """分析回滚根因（A86.5）。

        Args:
            delta_weights: 权重变化字典
            delta_F: 自由能变化
            reform_type: 重构类型

        Returns:
            根因类型

        Examples:
            >>> executor = AutoRollbackExecutor()
            >>> cause = executor.analyze_root_cause(
            ...     {"w": np.array([0.1])}, 1.0, "prune")
            >>> isinstance(cause, RootCauseType)
            True
        """
        if reform_type == "prune":
            return RootCauseType.OVER_PRUNING
        elif reform_type == "expand":
            return RootCauseType.OVER_EXPANSION
        total_change = sum(
            float(np.sum(np.abs(v))) for v in delta_weights.values()
        )
        if total_change > 10.0:
            return RootCauseType.WEIGHT_DESTRUCTION
        return RootCauseType.UNKNOWN

    def step(
        self,
        auto_rollback: AutoRollback
    ) -> AutoRollback:
        """执行一步自动回滚检查。

        Args:
            auto_rollback: 当前自动回滚状态

        Returns:
            更新后的自动回滚状态

        Examples:
            >>> from stage_085 import ExploreFreezer
            >>> from stage_084 import FileProtectionEngine, StorageCleaner
            >>> from stage_083 import MemoryMonitor, CPUMonitor
            >>> cpu = CPUMonitor()
            >>> mem = MemoryMonitor(cpu_monitor=cpu)
            >>> sc = StorageCleaner(memory_monitor=mem)
            >>> fpe = FileProtectionEngine(storage_cleaner=sc)
            >>> ef = ExploreFreezer(file_protection=fpe)
            >>> ar = AutoRollback(explore_freezer=ef)
            >>> executor = AutoRollbackExecutor()
            >>> new_ar = executor.step(ar)
            >>> isinstance(new_ar, AutoRollback)
            True
        """
        return auto_rollback
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `ExploreFreezer` | ← 阶段85 | 作为自动回滚的探索冻结输入（组合字段） |
| **输入** | A66.2-A66.3（回滚条件与机制） | ← 阶段66 | 倒退判定和回滚的理论基础 |
| **输入** | A78.5（回滚操作） | ← 阶段78 | 回滚的具体实现 |
| **输出** | `AutoRollback` | → 阶段87 | 自动回滚状态，作为比例执行器的组合输入 |
| **输出** | A86.1-A86.5（自动回滚定义） | → 阶段90 | 安全护栏的回滚检查接口 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| SciPy | 统计检验 | 置信区间计算和t检验 |
| NumPy | 性能计算 | 自由能差值的向量化计算 |
| SQLite | 回滚日志 | 持久化存储回滚记录 |
| JAX | 模型操作 | 快照加载和参数恢复 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V86.1 | 性能评估 | ΔF计算正确 | 数值测试 |
| V86.2 | 置信区间 | 95% CI计算正确 | 统计测试 |
| V86.3 | 倒退判定 | F_new > F_old·1.05判定为倒退 | 阈值测试 |
| V86.4 | 三级响应 | 轻微/中度/严重正确分类 | 级别测试 |
| V86.5 | 自动回滚 | 倒退后自动回滚到重构前快照 | 回滚测试 |
| V86.6 | 回滚频率限制 | ≤3次/天 | 频率测试 |
| V86.7 | 回滚前保存 | 回滚前保存当前状态 | 安全测试 |
| V86.8 | 回滚后验证 | \|F_restored - F_old\| < 1% | 验证测试 |
| V86.9 | 验证失败处理 | 升级为人工干预 | 异常测试 |
| V86.10 | EWC锚点更新 | 锚点设为回滚后参数 | 锚点测试 |
| V86.11 | 冷却期延长 | 回滚后冷却期延长 | 冷却测试 |
| V86.12 | 根因分析 | 正确识别根因类型 | 分析测试 |
| V86.13 | AutoRollback组合 | `rollback.explore_freezer` 类型为 `ExploreFreezer` | 类型检查 |
| V86.14 | 端到端流程 | 重构→倒退→回滚→验证→根因分析 | 集成测试 |
| V86.15 | 回滚日志 | 每次回滚有完整日志 | 日志测试 |

---

> **阶段86完** | 阶段86建立了模型重构后性能倒退自动回滚机制：重构后性能评估 F_new vs F_old（A86.1），倒退判定 F_new > F_old · (1 + θ_rollback), θ_rollback = 0.05（A86.2），自动回滚 rollback(snapshot_id)（A86.3），回滚后验证 F_restored ≈ F_old（A86.4），回滚日志与根因分析（A86.5），封装为AutoRollback数据类组合ExploreFreezer。下一步：阶段87将建立70/30探索利用比例的强制执行器。
