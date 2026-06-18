# 阶段87：70/30探索利用比例的强制执行器

> **阶段编号**：S87 | **所属层**：第九层·安全护栏层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 全局比例监控 | "看天平"——实时查看探索和利用各占多少 |
> | 利用过多强制探索 | "别光吃老本"——利用超过75%就注入探索任务 |
> | 探索过多强制利用 | "别光瞎试"——利用低于65%就注入利用任务 |
> | 执行日志与告警 | "纠偏记录"——记录每次强制调整 |
> | RatioEnforcer | "比例执行器的完整画像"——组合AutoRollback，在自动回滚基础上增加比例执行状态 |

---

## 1. 第一性原理依据

本阶段从阶段38（比例强制执行器）和阶段68（进化层比例）的核心结论出发，建立70/30探索利用比例的强制执行器。核心依据为：

- **A38.3（比例强制执行器）**（引用阶段38）：当探索利用比例偏离目标时强制注入任务调整比例。本阶段将A38.3从"决策层"推广到"安全护栏层"——安全护栏层的比例强制执行是最后一道防线，确保即使决策层和进化层的比例调节失效，全局比例也不会严重偏离。

- **A68.1-A68.4（进化层比例）**（引用阶段68）：进化层的探索利用比例定义、调整和约束。安全护栏层的比例强制执行与进化层的比例调节协同工作——进化层负责细粒度的比例调整，安全护栏层负责粗粒度的比例兜底。

本阶段的核心目标是：**定义全局比例监控 ratio = count_exploit / count_total（A87.1），强制执行 if ratio > 0.75 then inject_explore_task()（A87.2），强制执行 if ratio < 0.65 then inject_exploit_task()（A87.3），执行日志与告警（A87.4），封装为RatioEnforcer数据类，组合AutoRollback作为自动回滚的输入。**

> **去重声明**：本阶段是安全护栏层的比例强制执行器，理论基础在阶段7，工程实现在阶段38/68，本阶段是安全护栏层的强制执行兜底。不重新定义探索利用比例的理论基础（阶段7）或决策层/进化层的比例调节机制（阶段38/68）。

---

## 2. 核心数学公式

### 第一部分：全局比例监控

#### 2.1 全局比例监控定义（A87.1）

全局比例监控度量利用任务在总任务中的占比：

$$\text{ratio} = \frac{\text{count\_exploit}}{\text{count\_total}} = \frac{\text{count\_exploit}}{\text{count\_explore} + \text{count\_exploit}} \tag{A87.1}$$

**比例监控的滑动窗口**：

与CPU/内存监控类似，采用滑动窗口避免历史数据影响：

$$\text{ratio}_W = \frac{\sum_{t=t_0}^{t_0+W} \mathbb{1}[\text{type}(t) = \text{exploit}]}{W} \tag{A87.1-window}$$

其中 $W = 1000$ 步（约100秒的运行时间）。

**比例监控的目标范围**：

| 指标 | 目标值 | 安全范围 | 说明 |
|------|--------|---------|------|
| 利用比例 | 0.70 | [0.65, 0.75] | 70%利用，30%探索 |
| 探索比例 | 0.30 | [0.25, 0.35] | 30%探索，70%利用 |

**比例监控的采样频率**：

| 状态 | 采样间隔 | 说明 |
|------|---------|------|
| 正常（0.65 ≤ ratio ≤ 0.75） | 100步 | 低频采样 |
| 偏离（ratio < 0.60 或 ratio > 0.80） | 10步 | 高频采样 |

**比例监控的多层级协同**：

| 层级 | 调节粒度 | 响应速度 | 说明 |
|------|---------|---------|------|
| 决策层（A38.3） | 细粒度（±5%） | 快 | 主动调节 |
| 进化层（A68.1-A68.4） | 中粒度（±10%） | 中 | 进化调节 |
| 安全护栏层（A87.1-A87.4） | 粗粒度（±15%） | 慢 | 兜底保障 |

安全护栏层仅在比例严重偏离时介入，不干扰决策层和进化层的正常调节。

- **输入**：任务执行历史（类型+时间）
- **输出**：当前利用比例、是否偏离
- **约束**：滑动窗口1000步

> **白话翻译**：全局比例监控就是"看天平"——实时查看探索和利用各占多少。目标70%利用、30%探索，安全范围65%-75%。只看最近1000步，更早的不算。安全护栏层是最后一道防线，只在比例严重偏离时介入，不干扰决策层和进化层的正常调节。

---

### 第二部分：利用过多强制探索

#### 2.2 利用过多强制探索定义（A87.2）

当利用比例超过75%时，强制注入探索任务：

$$\text{if } \text{ratio} > 0.75 \implies \text{inject\_explore\_task}() \tag{A87.2}$$

**注入探索任务的策略**：

| 策略 | 说明 | 优先级 |
|------|------|--------|
| 解冻已冻结的探索任务 | A85.4的解冻条件可能已满足 | 最高 |
| 创建新的随机探索任务 | 在未探索的状态空间随机采样 | 中 |
| 降低探索任务的资源门槛 | 允许更多探索任务同时运行 | 低 |

**注入探索任务的数量**：

$$n_{\text{inject}} = \left\lceil \frac{\text{ratio} - 0.70}{0.05} \right\rceil \tag{A87.2-count}$$

| 当前比例 | 注入探索任务数 | 说明 |
|---------|-------------|------|
| 0.76 | 2 | 轻微偏离 |
| 0.80 | 2 | 中度偏离 |
| 0.85 | 3 | 严重偏离 |
| 0.90 | 4 | 极端偏离 |

**注入探索任务的约束**：

注入的探索任务不能违反其他安全约束：

$$\text{inject\_safe} = \mathbb{1}\left[\text{cpu\_ratio} < 0.20 \;\wedge\; \text{mem\_ratio} < 0.20\right] \tag{A87.2-safe}$$

如果CPU或内存已超限，不注入新的探索任务——资源约束优先于比例约束。

**注入探索任务与A85的协同**：

A85（引用阶段85）定义了探索任务的冻结和解冻机制。A87.2的注入策略优先解冻已冻结的探索任务——这些任务之前有过收益历史，比全新随机探索更有可能产生价值。

- **输入**：当前利用比例、冻结任务列表、资源状态
- **输出**：注入的探索任务列表
- **约束**：资源约束优先于比例约束

> **白话翻译**：利用过多强制探索就是"别光吃老本"——利用超过75%就注入探索任务。优先解冻之前冻住的探索任务（它们有收益历史），不够再创建新的随机探索。注入多少取决于偏离多少：76%注入2个，85%注入3个。但如果CPU或内存已超限，就不注入——资源约束比比例约束更重要。

---

### 第三部分：探索过多强制利用

#### 2.3 探索过多强制利用定义（A87.3）

当利用比例低于65%时，强制注入利用任务：

$$\text{if } \text{ratio} < 0.65 \implies \text{inject\_exploit\_task}() \tag{A87.3}$$

**注入利用任务的策略**：

| 策略 | 说明 | 优先级 |
|------|------|--------|
| 执行当前最优策略 | 选择适应度最高的策略执行 | 最高 |
| 冻结低收益探索任务 | A85.2的低收益判定 | 中 |
| 提高利用任务优先级 | 调度器优先分配资源给利用 | 低 |

**注入利用任务的数量**：

$$n_{\text{inject}} = \left\lceil \frac{0.70 - \text{ratio}}{0.05} \right\rceil \tag{A87.3-count}$$

| 当前比例 | 注入利用任务数 | 说明 |
|---------|-------------|------|
| 0.64 | 2 | 轻微偏离 |
| 0.60 | 2 | 中度偏离 |
| 0.55 | 3 | 严重偏离 |
| 0.50 | 4 | 极端偏离 |

**探索过多与A85的协同**：

A85（引用阶段85）定义了探索任务的无收益自动冻结。当利用比例过低时，A87.3与A85协同工作：

1. A87.3触发强制利用
2. 检查是否有低收益探索任务（A85.2的软判定）
3. 冻结低收益探索任务，释放资源
4. 注入利用任务

**探索过多的风险评估**：

探索过多比利用过多更危险——探索消耗大量资源但收益不确定：

| 风险 | 说明 |
|------|------|
| 资源浪费 | 探索任务消耗CPU和内存但可能无收益 |
| 性能下降 | 过多探索导致利用不足，当前最优策略执行不够 |
| 比例失衡 | 长期探索过多导致认知偏差 |

因此，探索过多的强制执行比利用过多更积极——65%的阈值比75%更紧。

- **输入**：当前利用比例、探索任务收益、资源状态
- **输出**：注入的利用任务列表、冻结的探索任务列表
- **约束**：探索过多更积极干预

> **白话翻译**：探索过多强制利用就是"别光瞎试"——利用低于65%就注入利用任务。优先执行当前最优策略，不够就冻结低收益的探索任务释放资源。探索过多比利用过多更危险（浪费资源、性能下降），所以65%的阈值比75%更紧——更早干预。

---

### 第四部分：执行日志与告警

#### 2.4 执行日志与告警定义（A87.4）

记录所有比例强制执行操作和告警：

**执行日志字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| timestamp | float | 执行时间 |
| ratio_before | float | 执行前利用比例 |
| ratio_target | float | 目标比例 |
| action | str | INJECT_EXPLORE / INJECT_EXPLOIT |
| inject_count | int | 注入任务数 |
| freeze_count | int | 冻结任务数（如有） |
| ratio_after | float | 执行后利用比例 |

**告警机制**：

| 告警级别 | 条件 | 说明 |
|---------|------|------|
| INFO | 65% ≤ ratio ≤ 75% | 正常范围 |
| WARNING | ratio < 60% 或 ratio > 80% | 偏离安全范围 |
| CRITICAL | ratio < 50% 或 ratio > 90% | 严重偏离 |

CRITICAL级别的告警意味着决策层和进化层的比例调节全部失效，安全护栏层正在兜底。

**比例偏离的统计**：

| 统计项 | 计算方式 | 说明 |
|--------|---------|------|
| 强制执行次数 | count(enforcement_events) | 安全护栏层介入频率 |
| 平均偏离程度 | mean(\|ratio - 0.70\|) | 偏离70%的平均幅度 |
| 最大偏离程度 | max(\|ratio - 0.70\|) | 历史最大偏离 |
| 偏离持续时间 | duration(ratio outside [0.65, 0.75]) | 偏离持续多久 |

**统计报告**：

每小时生成一次比例统计报告，包含上述所有统计项。如果强制执行频率异常升高（>10次/小时），触发系统级告警——可能意味着决策层或进化层存在缺陷。

- **输入**：强制执行事件、比例历史
- **输出**：执行日志、告警、统计报告
- **约束**：CRITICAL级别告警需立即响应

> **白话翻译**：执行日志与告警就是"纠偏记录"——记录每次强制调整前后的比例、注入了多少任务、冻结了多少任务。偏离60%以下或80%以上发WARNING，50%以下或90%以上发CRITICAL（说明决策层和进化层都失效了）。每小时出统计报告，强制执行太频繁（>10次/小时）说明系统有缺陷。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A87.1 | ratio = count_exploit / count_total | 任务执行历史 | 利用比例∈[0,1] | 滑动窗口1000步 |
| A87.2 | if ratio > 0.75 → inject_explore_task() | 比例+冻结任务+资源 | 探索任务列表 | 资源约束优先 |
| A87.3 | if ratio < 0.65 → inject_exploit_task() | 比例+探索收益+资源 | 利用任务列表 | 探索过多更积极 |
| A87.4 | 日志+告警+统计 | 强制执行事件 | 日志记录 | CRITICAL立即响应 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import numpy as np

# 引用阶段86的数据类
from stage_086 import AutoRollback


# ============================================================
# 阶段87 枚举：比例偏离方向
# ============================================================
class RatioDeviation(Enum):
    """比例偏离方向。"""
    NONE = "none"               # 正常范围
    TOO_MUCH_EXPLOIT = "too_much_exploit"  # 利用过多
    TOO_MUCH_EXPLORE = "too_much_explore"  # 探索过多


# ============================================================
# 阶段87 枚举：强制执行动作
# ============================================================
class EnforcementAction(Enum):
    """强制执行动作。"""
    INJECT_EXPLORE = "inject_explore"
    INJECT_EXPLOIT = "inject_exploit"
    NONE = "none"


# ============================================================
# 阶段87 枚举：告警级别
# ============================================================
class RatioAlertLevel(Enum):
    """比例告警级别。"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ============================================================
# 阶段87 数据类：比例执行日志记录
# ============================================================
@dataclass(frozen=True)
class RatioEnforcementRecord:
    """比例强制执行日志记录。

    Attributes:
        timestamp: 执行时间
        ratio_before: 执行前利用比例
        ratio_target: 目标比例
        action: 执行动作
        inject_count: 注入任务数
        freeze_count: 冻结任务数
        ratio_after: 执行后利用比例
    """
    timestamp: float
    ratio_before: float
    ratio_target: float = 0.70
    action: EnforcementAction = EnforcementAction.NONE
    inject_count: int = 0
    freeze_count: int = 0
    ratio_after: float = 0.0


# ============================================================
# 阶段87 数据类：比例执行器状态
# ============================================================
@dataclass(frozen=True)
class RatioEnforcerState:
    """比例执行器的状态描述。

    Attributes:
        current_ratio: 当前利用比例
        target_ratio: 目标利用比例
        exploit_upper_bound: 利用上限（默认0.75）
        exploit_lower_bound: 利用下限（默认0.65）
        total_enforcements: 总强制执行次数
        recent_enforcement_rate: 最近强制执行频率（次/小时）
        alert_level: 当前告警级别
        enforcement_history: 最近强制执行记录
    """
    current_ratio: float = 0.70
    target_ratio: float = 0.70
    exploit_upper_bound: float = 0.75
    exploit_lower_bound: float = 0.65
    total_enforcements: int = 0
    recent_enforcement_rate: float = 0.0
    alert_level: RatioAlertLevel = RatioAlertLevel.INFO
    enforcement_history: Tuple[RatioEnforcementRecord, ...] = ()


# ============================================================
# 阶段87 数据类：比例执行器
# ============================================================
@dataclass(frozen=True)
class RatioEnforcer:
    """70/30探索利用比例的强制执行器。

    组合阶段86的AutoRollback，在自动回滚基础上增加比例执行状态。

    Attributes:
        auto_rollback: 阶段86的自动回滚器（组合字段）
        enforcer_state: 比例执行器状态
        window_size: 滑动窗口大小（默认1000步）
        max_enforcement_rate: 最大强制执行频率（默认10次/小时）
    """
    auto_rollback: AutoRollback
    enforcer_state: RatioEnforcerState = field(default_factory=RatioEnforcerState)
    window_size: int = 1000
    max_enforcement_rate: float = 10.0


# ============================================================
# 阶段87 核心接口：比例执行器
# ============================================================
class RatioEnforcerExecutor:
    """阶段87的核心接口：比例强制执行器。

    实现全局比例监控、强制探索注入、强制利用注入和执行日志。

    核心约束：
        - 目标比例70%利用/30%探索
        - 安全范围[65%, 75%]
        - 资源约束优先于比例约束
    """

    def monitor_ratio(
        self,
        task_history: List[Dict],
        window: int = 1000
    ) -> float:
        """监控全局比例（A87.1）。

        Args:
            task_history: 任务执行历史列表，每个元素为{"type": "explore"|"exploit"}
            window: 滑动窗口大小，必须 > 0

        Returns:
            当前利用比例 ∈ [0, 1]

        Raises:
            ValueError: 当window <= 0时

        Examples:
            >>> executor = RatioEnforcerExecutor()
            >>> history = [{"type": "exploit"}] * 7 + [{"type": "explore"}] * 3
            >>> ratio = executor.monitor_ratio(history, 10)
            >>> abs(ratio - 0.7) < 1e-6
            True
        """
        if window <= 0:
            raise ValueError(f"window必须 > 0，收到 {window}")
        recent = task_history[-window:] if len(task_history) > window else task_history
        if not recent:
            return 0.5
        exploit_count = sum(1 for t in recent if t.get("type") == "exploit")
        return exploit_count / len(recent)

    def check_deviation(
        self,
        ratio: float,
        upper_bound: float = 0.75,
        lower_bound: float = 0.65
    ) -> RatioDeviation:
        """检查比例偏离（A87.1 + A87.2 + A87.3）。

        Args:
            ratio: 当前利用比例
            upper_bound: 利用上限，必须 > lower_bound
            lower_bound: 利用下限，必须 >= 0

        Returns:
            偏离方向

        Raises:
            ValueError: 当upper_bound <= lower_bound时

        Examples:
            >>> executor = RatioEnforcerExecutor()
            >>> executor.check_deviation(0.8)
            <RatioDeviation.EXPLOIT_TOO_HIGH: 'exploit_too_high'>
            >>> executor.check_deviation(0.6)
            <RatioDeviation.EXPLOIT_TOO_LOW: 'exploit_too_low'>
        """
        if upper_bound <= lower_bound:
            raise ValueError("upper_bound必须 > lower_bound")
        if ratio > upper_bound:
            return RatioDeviation.EXPLOIT_TOO_HIGH
        elif ratio < lower_bound:
            return RatioDeviation.EXPLOIT_TOO_LOW
        return RatioDeviation.NORMAL

    def inject_explore_task(
        self,
        ratio: float,
        frozen_tasks: List[str],
        cpu_ratio: float,
        mem_ratio: float
    ) -> List[str]:
        """注入探索任务（A87.2）。

        Args:
            ratio: 当前利用比例
            frozen_tasks: 可解冻的冻结任务列表
            cpu_ratio: CPU占用率，范围[0,1]
            mem_ratio: 内存占用率，范围[0,1]

        Returns:
            注入的探索任务ID列表

        Examples:
            >>> executor = RatioEnforcerExecutor()
            >>> tasks = executor.inject_explore_task(0.8, ["t1"], 0.1, 0.1)
            >>> len(tasks) > 0
            True
        """
        if cpu_ratio > 0.20 or mem_ratio > 0.20:
            return []
        deviation = ratio - 0.70
        n_inject = max(1, int(deviation * 10))
        return frozen_tasks[:n_inject] if frozen_tasks else ["new_explore_task"]

    def inject_exploit_task(
        self,
        ratio: float,
        low_benefit_explore_tasks: List[str]
    ) -> Tuple[List[str], List[str]]:
        """注入利用任务（A87.3）。

        Args:
            ratio: 当前利用比例
            low_benefit_explore_tasks: 低收益探索任务列表

        Returns:
            (注入的利用任务列表, 冻结的探索任务列表)

        Examples:
            >>> executor = RatioEnforcerExecutor()
            >>> exploit, freeze = executor.inject_exploit_task(
            ...     0.6, ["e1", "e2"])
            >>> len(exploit) > 0
            True
        """
        deviation = 0.70 - ratio
        n_inject = max(1, int(deviation * 10))
        return ["exploit_task"] * n_inject, low_benefit_explore_tasks[:n_inject]

    def compute_alert_level(
        self,
        ratio: float
    ) -> RatioAlertLevel:
        """计算告警级别（A87.4）。

        Args:
            ratio: 当前利用比例

        Returns:
            告警级别

        Examples:
            >>> executor = RatioEnforcerExecutor()
            >>> executor.compute_alert_level(0.8)
            <RatioAlertLevel.CRITICAL: 'critical'>
            >>> executor.compute_alert_level(0.68)
            <RatioAlertLevel.INFO: 'info'>
        """
        if ratio > 0.80 or ratio < 0.60:
            return RatioAlertLevel.CRITICAL
        elif ratio > 0.75 or ratio < 0.65:
            return RatioAlertLevel.WARNING
        return RatioAlertLevel.INFO

    def step(
        self,
        enforcer: RatioEnforcer
    ) -> RatioEnforcer:
        """执行一步比例监控与强制执行。

        Args:
            enforcer: 当前比例执行器状态

        Returns:
            更新后的比例执行器状态

        Examples:
            >>> from stage_086 import AutoRollback
            >>> from stage_085 import ExploreFreezer
            >>> from stage_084 import FileProtectionEngine, StorageCleaner
            >>> from stage_083 import MemoryMonitor, CPUMonitor
            >>> cpu = CPUMonitor()
            >>> mem = MemoryMonitor(cpu_monitor=cpu)
            >>> sc = StorageCleaner(memory_monitor=mem)
            >>> fpe = FileProtectionEngine(storage_cleaner=sc)
            >>> ef = ExploreFreezer(file_protection=fpe)
            >>> ar = AutoRollback(explore_freezer=ef)
            >>> enforcer = RatioEnforcer(auto_rollback=ar)
            >>> executor = RatioEnforcerExecutor()
            >>> new_enforcer = executor.step(enforcer)
            >>> isinstance(new_enforcer, RatioEnforcer)
            True
        """
        return enforcer
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `AutoRollback` | ← 阶段86 | 作为比例执行器的自动回滚输入（组合字段） |
| **输入** | A38.3（比例强制执行器） | ← 阶段38 | 决策层比例调节 |
| **输入** | A68.1-A68.4（进化层比例） | ← 阶段68 | 进化层比例调节 |
| **输出** | `RatioEnforcer` | → 阶段88 | 比例执行器状态，作为空闲复盘触发器的组合输入 |
| **输出** | A87.1-A87.4（比例执行定义） | → 阶段90 | 安全护栏的比例检查接口 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| collections.deque | 滑动窗口 | 任务历史的固定长度队列 |
| heapq | 优先级调度 | 利用任务的最优策略选择 |
| SQLite | 执行日志 | 持久化存储强制执行记录 |
| psutil | 资源监控 | 注入任务前的资源检查 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V87.1 | 比例监控 | ratio ∈ [0,1] | 度量测试 |
| V87.2 | 滑动窗口 | 只统计最近1000步 | 窗口测试 |
| V87.3 | 利用过多判定 | ratio > 0.75判定偏离 | 阈值测试 |
| V87.4 | 探索过多判定 | ratio < 0.65判定偏离 | 阈值测试 |
| V87.5 | 注入探索任务 | 优先解冻已有探索任务 | 优先级测试 |
| V87.6 | 注入利用任务 | 优先执行最优策略 | 策略测试 |
| V87.7 | 资源约束优先 | CPU/内存超限时不注入 | 安全测试 |
| V87.8 | 注入数量 | 按偏离程度计算 | 数量测试 |
| V87.9 | 告警级别 | INFO/WARNING/CRITICAL正确分类 | 告警测试 |
| V87.10 | 执行日志 | 每次强制执行有记录 | 日志测试 |
| V87.11 | 统计报告 | 每小时生成 | 报告测试 |
| V87.12 | 频率告警 | >10次/小时触发系统告警 | 频率测试 |
| V87.13 | RatioEnforcer组合 | `enforcer.auto_rollback` 类型为 `AutoRollback` | 类型检查 |
| V87.14 | 多层级协同 | 安全护栏不干扰决策层/进化层 | 协同测试 |
| V87.15 | 端到端流程 | 比例偏离→强制执行→比例恢复 | 集成测试 |

---

> **阶段87完** | 阶段87建立了70/30探索利用比例的强制执行器：全局比例监控 ratio = count_exploit / count_total（A87.1），强制执行 if ratio > 0.75 then inject_explore_task()（A87.2），强制执行 if ratio < 0.65 then inject_exploit_task()（A87.3），执行日志与告警（A87.4），封装为RatioEnforcer数据类组合AutoRollback。下一步：阶段88将建立系统空闲→离线复盘的触发器。
