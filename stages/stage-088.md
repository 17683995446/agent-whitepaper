# 阶段88：系统空闲→离线复盘的触发器

> **阶段编号**：S88 | **所属层**：第九层·安全护栏层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 空闲判定 | "闲着呢"——CPU低、没任务、好久没干活 |
> | 触发离线复盘 | "闲着就复习"——空闲时自动开始离线复盘 |
> | 复盘资源分配 | "复习预算"——复盘最多用10%的CPU |
> | 复盘终止条件 | "有活干就停"——新任务来了或CPU超15%就停 |
> | IdleReplayTrigger | "空闲复盘触发器的完整画像"——组合RatioEnforcer，在比例执行基础上增加空闲复盘触发状态 |

---

## 1. 第一性原理依据

本阶段从阶段76（空闲判定）和阶段9（终极循环触发）的核心结论出发，建立系统空闲→离线复盘的触发器。核心依据为：

- **A76.1（空闲判定）**（引用阶段76）：is_idle = 𝟙[cpu_usage < θ_idle ∧ no_active_task]，空闲判定基于CPU使用率和活跃任务状态。本阶段在A76.1的基础上增加时间维度——不仅CPU低且无任务，还需持续一段时间才判定为空闲。

- **A9.6（终极循环触发条件）**（引用阶段9）：终极循环在长期无进步时触发认知升级。空闲离线复盘是终极循环在安全护栏层的实现——当系统空闲时，利用空闲资源进行离线复盘，推动系统持续进步。

本阶段的核心目标是：**定义空闲判定 cpu_usage < 0.10 AND no_active_task AND time_since_last_task > T_idle（A88.1），触发离线复盘 trigger_offline_replay()（A88.2），复盘资源分配 cpu_budget = 0.10（A88.3），复盘终止条件 new_task_available OR cpu_usage > 0.15（A88.4），封装为IdleReplayTrigger数据类，组合RatioEnforcer作为比例执行器的输入。**

> **去重声明**：本阶段仅定义安全护栏层的空闲→复盘触发器，不重新定义空闲判定（阶段76的A76.1）或离线复盘的具体实现（阶段76的A76.2-A76.5）。本阶段是A76.1在安全护栏层的扩展，复盘操作委托给A76.2-A76.5。

---

## 2. 核心数学公式

### 第一部分：空闲判定

#### 2.1 空闲判定定义（A88.1）

空闲判定基于三个条件的联合：

$$\text{is\_idle} = \mathbb{1}\left[\text{cpu\_usage} < \theta_{\text{idle}} \;\wedge\; \text{no\_active\_task} \;\wedge\; \text{time\_since\_last\_task} > T_{\text{idle}}\right] \tag{A88.1}$$

其中：
- $\theta_{\text{idle}} = 0.10$（CPU使用率低于10%）
- $\text{no\_active\_task}$ 为无活跃任务标志
- $T_{\text{idle}} = 60$ 秒（距离上一个任务完成超过60秒）

**空闲判定与A76.1的区别**：

| 维度 | A76.1（记忆系统层） | A88.1（安全护栏层） |
|------|-------------------|-------------------|
| CPU阈值 | 0.30 | 0.10 |
| 时间条件 | 无 | > 60秒 |
| 触发动作 | 记忆系统内部复盘 | 安全护栏层强制复盘 |

安全护栏层的空闲判定更严格——CPU需低于10%（而非30%），且需持续60秒无任务。这确保安全护栏层仅在系统真正空闲时触发复盘，不干扰正常运行的复盘调度。

**空闲判定的防误触机制**：

为避免短暂空闲触发不必要的复盘，增加持续时间要求：

$$\text{idle\_duration} = t - t_{\text{last\_task\_end}} \tag{A88.1-duration}$$

| idle_duration | 判定 | 说明 |
|--------------|------|------|
| < 30秒 | 不空闲 | 可能只是任务间隙 |
| 30-60秒 | 可能空闲 | 继续观察 |
| > 60秒 | 确认空闲 | 触发复盘 |

**空闲判定的排除条件**：

以下情况即使满足三条件也不判定为空闲：

| 排除条件 | 原因 |
|---------|------|
| 正在执行模型重构 | 重构期间复盘可能干扰 |
| 正在执行回滚 | 回滚期间复盘可能不一致 |
| 正在执行存储清理 | 清理期间复盘可能读取不完整数据 |
| 安全告警活跃 | 告警期间应优先处理安全问题 |

$$\text{is\_idle\_effective} = \text{is\_idle} \;\wedge\; \neg\text{reforming} \;\wedge\; \neg\text{rolling\_back} \;\wedge\; \neg\text{cleaning} \;\wedge\; \neg\text{alerting} \tag{A88.1-effective}$$

- **输入**：CPU使用率、活跃任务状态、上次任务结束时间、排除条件
- **输出**：是否空闲
- **约束**：CPU < 10%、无任务、持续60秒、无排除条件

> **白话翻译**：空闲判定就是"闲着呢"——CPU低于10%、没有活跃任务、上一个任务结束超过60秒，三个条件同时满足才算空闲。比记忆系统层的空闲判定更严格（CPU阈值10% vs 30%），因为安全护栏层只在真正空闲时才介入。还有排除条件：正在重构、回滚、清理或有安全告警时不算空闲。

---

### 第二部分：触发离线复盘

#### 2.2 触发离线复盘定义（A88.2）

确认空闲后触发离线复盘：

$$\text{trigger\_offline\_replay}() \tag{A88.2}$$

**触发离线复盘与A76.2-A76.5的对应**：

A76.2-A76.5（引用阶段76）定义了离线复盘的完整流程：批量重放、批量优化、收益评估和终止条件。A88.2触发后直接调用A76.2-A76.5的流程。

**触发离线复盘的优先级**：

| 优先级 | 复盘类型 | 说明 |
|--------|---------|------|
| 1 | 高误差样本复盘 | 优先复习误差最大的样本 |
| 2 | 巩固期复盘 | 复习需要巩固的记忆 |
| 3 | 生成式回放 | 用VAE生成新样本增强学习 |
| 4 | 随机回放 | 随机采样复习 |

**触发离线复盘的频率限制**：

为防止频繁触发复盘消耗资源：

$$\text{replay\_interval} \geq T_{\text{min\_replay}} = 300\text{s} \tag{A88.2-interval}$$

两次复盘之间至少间隔5分钟。

**触发离线复盘的收益预期**：

根据A80.6的验证结果，离线复盘的收益应>1%：

$$\mathbb{E}[\text{replay\_benefit}] > 0.01 \tag{A88.2-benefit}$$

如果连续3次复盘收益<0.5%，暂停复盘触发——可能当前模型已接近最优，复盘边际收益递减。

- **输入**：空闲判定结果
- **输出**：离线复盘触发信号
- **约束**：复盘间隔≥5分钟；收益>1%

> **白话翻译**：触发离线复盘就是"闲着就复习"——确认空闲后自动开始离线复盘。优先复习误差最大的样本，然后是巩固期记忆，再是生成式回放，最后随机回放。两次复盘至少间隔5分钟。如果连续3次复习收益不到0.5%，暂停复习——可能已经学得够好了，再复习也没用。

---

### 第三部分：复盘资源分配

#### 2.3 复盘资源分配定义（A88.3）

离线复盘的CPU预算不超过10%：

$$\text{cpu\_budget}_{\text{replay}} = 0.10 \tag{A88.3}$$

**复盘资源分配的约束**：

复盘资源与正常运行资源隔离，复盘不能影响正常任务的执行：

$$\text{cpu\_replay} + \text{cpu\_normal} \leq \theta_{\text{cpu\_max}} = 0.20 \tag{A88.3-isolation}$$

由于空闲时cpu_normal接近0，复盘的10%预算不会导致总CPU超限。

**复盘资源的动态调整**：

复盘资源根据系统负载动态调整：

$$\text{cpu\_budget}_{\text{replay}} = \min(0.10, \theta_{\text{cpu\_max}} - \text{cpu\_normal} - 0.05) \tag{A88.3-dynamic}$$

预留5%的安全余量，确保复盘不会导致CPU超限。

**复盘资源的内存预算**：

| 资源 | 预算 | 说明 |
|------|------|------|
| CPU | 10% | 复盘计算 |
| 内存 | 5% | 批量样本缓存 |
| 磁盘I/O | 低优先级 | 不影响正常I/O |

**复盘资源的监控**：

复盘过程中持续监控资源使用：

$$\text{if } \text{cpu\_replay} > \text{cpu\_budget}_{\text{replay}} \implies \text{reduce\_replay\_batch\_size}() \tag{A88.3-monitor}$$

如果复盘实际CPU超过预算，自动减小批量大小降低CPU占用。

- **输入**：系统CPU使用率、复盘预算
- **输出**：复盘资源分配方案
- **约束**：CPU预算≤10%；预留5%安全余量

> **白话翻译**：复盘资源分配就是"复习预算"——复盘最多用10%的CPU、5%的内存。复盘资源与正常资源隔离，不能影响正常任务。还要预留5%安全余量。如果复盘实际CPU超了预算，自动减小批量大小。

---

### 第四部分：复盘终止条件

#### 2.4 复盘终止条件定义（A88.4）

离线复盘在以下任一条件满足时终止：

$$\text{stop\_replay} = \mathbb{1}\left[\text{new\_task\_available} \;\vee\; \text{cpu\_usage} > 0.15\right] \tag{A88.4}$$

**终止条件1：新任务可用 new_task_available**

当有新的外部任务到达时，复盘立即终止：

$$\text{new\_task\_available} = \mathbb{1}\left[\text{task\_queue\_length} > 0\right] \tag{A88.4-task}$$

新任务优先级高于复盘——复盘是利用空闲资源的活动，有正事做就停。

**终止条件2：CPU超限 cpu_usage > 0.15**

当系统CPU使用率超过15%时终止复盘：

$$\text{cpu\_over} = \mathbb{1}\left[\text{cpu\_usage} > 0.15\right] \tag{A88.4-cpu}$$

15%的终止阈值高于10%的空闲阈值，留出5%的缓冲区。CPU从10%增长到15%的过程中，复盘仍在继续，但已准备终止。

**终止条件的优先级**：

| 终止条件 | 优先级 | 响应速度 |
|---------|--------|---------|
| 新任务可用 | 最高 | 立即终止 |
| CPU超15% | 高 | 当前批次完成后终止 |
| 复盘收益<0.5% | 中 | 当前轮次完成后终止 |
| 复盘时间超限 | 低 | 强制终止 |

**终止后的状态保存**：

复盘终止时保存当前进度：

$$\text{save\_replay\_progress}(\text{current\_batch}, \text{current\_epoch}) \tag{A88.4-save}$$

下次空闲触发复盘时，可从保存的进度继续，而非从头开始。

- **输入**：任务队列、CPU使用率、复盘收益
- **输出**：是否终止复盘
- **约束**：新任务立即终止；CPU超15%批次后终止

> **白话翻译**：复盘终止条件就是"有活干就停"——两种情况停：新任务来了立即停，CPU超15%当前批次做完停。还有两个软条件：收益太低（<0.5%）当前轮次做完停，复盘时间太长强制停。停的时候保存进度，下次空闲可以接着来。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A88.1 | is_idle = 𝟙[cpu<0.10 ∧ no_task ∧ idle>60s] | CPU+任务+时间 | 空闲标志 | 排除条件检查 |
| A88.2 | trigger_offline_replay() | 空闲判定 | 复盘触发 | 间隔≥5分钟 |
| A88.3 | cpu_budget_replay = 0.10 | 系统CPU | 资源分配 | 预留5%余量 |
| A88.4 | stop = 𝟙[new_task ∨ cpu>0.15] | 任务+CPU | 终止标志 | 新任务立即停 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import numpy as np

# 引用阶段87的数据类
from stage_087 import RatioEnforcer


# ============================================================
# 阶段88 枚举：复盘优先级
# ============================================================
class ReplayPriority(Enum):
    """离线复盘优先级。"""
    HIGH_ERROR = 1       # 高误差样本
    CONSOLIDATION = 2    # 巩固期复盘
    GENERATIVE = 3       # 生成式回放
    RANDOM = 4           # 随机回放


# ============================================================
# 阶段88 数据类：空闲复盘触发器状态
# ============================================================
@dataclass(frozen=True)
class IdleReplayTriggerState:
    """空闲复盘触发器的状态描述。

    Attributes:
        is_idle: 是否空闲
        idle_duration: 空闲持续时长（秒）
        is_replaying: 是否正在复盘
        replay_priority: 当前复盘优先级
        replay_start_time: 复盘开始时间
        replay_cpu_budget: 复盘CPU预算
        replay_batches_completed: 已完成批次数
        replay_benefit: 复盘收益
        last_replay_time: 上次复盘时间
        consecutive_low_benefit: 连续低收益次数
        saved_progress: 保存的复盘进度

    Invariants:
        - idle_duration >= 0
        - replay_cpu_budget <= 0.10
    """
    is_idle: bool = False
    idle_duration: float = 0.0
    is_replaying: bool = False
    replay_priority: Optional[ReplayPriority] = None
    replay_start_time: float = 0.0
    replay_cpu_budget: float = 0.10
    replay_batches_completed: int = 0
    replay_benefit: float = 0.0
    last_replay_time: float = 0.0
    consecutive_low_benefit: int = 0
    saved_progress: Optional[Dict] = None


# ============================================================
# 阶段88 数据类：空闲复盘触发器
# ============================================================
@dataclass(frozen=True)
class IdleReplayTrigger:
    """系统空闲→离线复盘的触发器。

    组合阶段87的RatioEnforcer，在比例执行基础上增加空闲复盘触发状态。

    Attributes:
        ratio_enforcer: 阶段87的比例执行器（组合字段）
        trigger_state: 空闲复盘触发器状态
        idle_cpu_threshold: 空闲CPU阈值（默认0.10）
        idle_time_threshold: 空闲时间阈值（默认60秒）
        replay_cpu_budget: 复盘CPU预算（默认0.10）
        replay_stop_cpu: 复盘终止CPU阈值（默认0.15）
        min_replay_interval: 最小复盘间隔（默认300秒）
        low_benefit_threshold: 低收益阈值（默认0.005）
        max_low_benefit_count: 最大连续低收益次数（默认3）
    """
    ratio_enforcer: RatioEnforcer
    trigger_state: IdleReplayTriggerState = field(
        default_factory=IdleReplayTriggerState
    )
    idle_cpu_threshold: float = 0.10
    idle_time_threshold: float = 60.0
    replay_cpu_budget: float = 0.10
    replay_stop_cpu: float = 0.15
    min_replay_interval: float = 300.0
    low_benefit_threshold: float = 0.005
    max_low_benefit_count: int = 3


# ============================================================
# 阶段88 核心接口：空闲复盘触发器
# ============================================================
class IdleReplayTriggerExecutor:
    """阶段88的核心接口：空闲复盘触发器。

    实现空闲判定、复盘触发、资源分配和终止条件。

    核心约束：
        - CPU < 10% + 无任务 + 60秒
        - 复盘CPU预算 ≤ 10%
        - 新任务立即终止复盘
    """

    def check_idle(
        self,
        cpu_usage: float,
        has_active_task: bool,
        time_since_last_task: float,
        is_reforming: bool = False,
        is_rolling_back: bool = False,
        is_cleaning: bool = False,
        is_alerting: bool = False
    ) -> Tuple[bool, float]:
        """检查是否空闲（A88.1）。

        Args:
            cpu_usage: CPU使用率，范围[0,1]
            has_active_task: 是否有活跃任务
            time_since_last_task: 距上次任务时间（秒），必须 >= 0
            is_reforming: 是否正在重构
            is_rolling_back: 是否正在回滚
            is_cleaning: 是否正在清理
            is_alerting: 是否有安全告警

        Returns:
            (是否空闲, 空闲持续时长)

        Raises:
            ValueError: 当time_since_last_task < 0时

        Examples:
            >>> executor = IdleReplayTriggerExecutor()
            >>> is_idle, duration = executor.check_idle(
            ...     0.05, False, 70.0)
            >>> is_idle
            True
            >>> executor.check_idle(0.05, False, 70.0,
            ...     is_reforming=True)[0]
            False
        """
        if time_since_last_task < 0:
            raise ValueError(
                f"time_since_last_task必须 >= 0，"
                f"收到 {time_since_last_task}"
            )
        if is_reforming or is_rolling_back or is_cleaning or is_alerting:
            return False, 0.0
        if cpu_usage < 0.10 and not has_active_task and time_since_last_task >= 60.0:
            return True, time_since_last_task
        return False, 0.0

    def trigger_replay(
        self,
        is_idle: bool,
        last_replay_time: float,
        min_interval: float = 300.0,
        consecutive_low_benefit: int = 0,
        max_low_benefit: int = 3
    ) -> Tuple[bool, ReplayPriority]:
        """触发离线复盘（A88.2）。

        Args:
            is_idle: 是否空闲
            last_replay_time: 上次复盘时间（时间戳）
            min_interval: 最小复盘间隔（秒），必须 > 0
            consecutive_low_benefit: 连续低收益次数，必须 >= 0
            max_low_benefit: 最大连续低收益次数，必须 > 0

        Returns:
            (是否触发, 复盘优先级)

        Raises:
            ValueError: 当min_interval <= 0时
            ValueError: 当max_low_benefit <= 0时
            ValueError: 当consecutive_low_benefit < 0时

        Examples:
            >>> executor = IdleReplayTriggerExecutor()
            >>> trigger, prio = executor.trigger_replay(
            ...     True, 0.0, 300.0, 0, 3)
            >>> trigger
            True
        """
        if min_interval <= 0:
            raise ValueError(
                f"min_interval必须 > 0，收到 {min_interval}"
            )
        if max_low_benefit <= 0:
            raise ValueError(
                f"max_low_benefit必须 > 0，收到 {max_low_benefit}"
            )
        if consecutive_low_benefit < 0:
            raise ValueError(
                f"consecutive_low_benefit必须 >= 0，"
                f"收到 {consecutive_low_benefit}"
            )
        if not is_idle:
            return False, ReplayPriority.NONE
        if consecutive_low_benefit >= max_low_benefit:
            return False, ReplayPriority.NONE
        import time
        if time.time() - last_replay_time < min_interval:
            return False, ReplayPriority.LOW
        return True, ReplayPriority.HIGH

    def allocate_resources(
        self,
        cpu_normal: float,
        cpu_max: float = 0.20,
        safety_margin: float = 0.05
    ) -> float:
        """分配复盘资源（A88.3）。

        Args:
            cpu_normal: 正常任务CPU使用率，范围[0,1]
            cpu_max: CPU上限，必须 > 0
            safety_margin: 安全余量，必须 >= 0

        Returns:
            复盘CPU预算 ∈ [0, cpu_max]

        Raises:
            ValueError: 当cpu_max <= 0时
            ValueError: 当safety_margin < 0时

        Examples:
            >>> executor = IdleReplayTriggerExecutor()
            >>> budget = executor.allocate_resources(0.05, 0.20, 0.05)
            >>> budget <= 0.20
            True
        """
        if cpu_max <= 0:
            raise ValueError(f"cpu_max必须 > 0，收到 {cpu_max}")
        if safety_margin < 0:
            raise ValueError(
                f"safety_margin必须 >= 0，收到 {safety_margin}"
            )
        budget = cpu_max - cpu_normal - safety_margin
        return max(0.0, min(budget, 0.10))

    def check_stop(
        self,
        has_new_task: bool,
        cpu_usage: float,
        replay_benefit: float,
        stop_cpu: float = 0.15,
        low_benefit_threshold: float = 0.005
    ) -> Tuple[bool, str]:
        """检查复盘终止条件（A88.4）。

        Args:
            has_new_task: 是否有新任务
            cpu_usage: CPU使用率
            replay_benefit: 当前复盘收益
            stop_cpu: 终止CPU阈值
            low_benefit_threshold: 低收益阈值

        Returns:
            (是否终止, 终止原因)

        Examples:
            >>> executor = IdleReplayTriggerExecutor()
            >>> stop, reason = executor.check_stop(
            ...     True, 0.10, 0.01)
            >>> stop
            True
            >>> "new_task" in reason
            True
        """
        if has_new_task:
            return True, "new_task"
        if cpu_usage > stop_cpu:
            return True, "cpu_over_limit"
        if replay_benefit < low_benefit_threshold:
            return True, "low_benefit"
        return False, ""

    def step(
        self,
        trigger: IdleReplayTrigger
    ) -> IdleReplayTrigger:
        """执行一步空闲复盘检查。

        Args:
            trigger: 当前空闲复盘触发器状态

        Returns:
            更新后的空闲复盘触发器状态

        Examples:
            >>> from stage_087 import RatioEnforcer
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
            >>> re = RatioEnforcer(auto_rollback=ar)
            >>> trigger = IdleReplayTrigger(ratio_enforcer=re)
            >>> executor = IdleReplayTriggerExecutor()
            >>> new_trigger = executor.step(trigger)
            >>> isinstance(new_trigger, IdleReplayTrigger)
            True
        """
        return trigger
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `RatioEnforcer` | ← 阶段87 | 作为空闲复盘触发器的比例执行输入（组合字段） |
| **输入** | A76.1（空闲判定） | ← 阶段76 | 空闲判定的理论基础 |
| **输入** | A76.2-A76.5（离线复盘） | ← 阶段76 | 复盘操作的具体实现 |
| **输入** | A9.6（终极循环触发） | ← 阶段9 | 空闲复盘的理论依据 |
| **输出** | `IdleReplayTrigger` | → 阶段89 | 空闲复盘触发器状态，作为稳态探索触发器的组合输入 |
| **输出** | A88.1-A88.4（空闲复盘触发定义） | → 阶段90 | 安全护栏的空闲检查接口 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| psutil | CPU监控 | 空闲判定的CPU度量 |
| asyncio | 异步复盘 | 非阻塞的复盘执行 |
| SQLite | 进度保存 | 复盘进度的持久化 |
| time | 时间度量 | 空闲持续时间的计算 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V88.1 | 空闲判定 | CPU<10% ∧ 无任务 ∧ >60秒 | 判定测试 |
| V88.2 | 排除条件 | 重构/回滚/清理/告警时不判定空闲 | 排除测试 |
| V88.3 | 触发复盘 | 空闲后触发离线复盘 | 触发测试 |
| V88.4 | 复盘间隔 | 两次复盘间隔≥5分钟 | 间隔测试 |
| V88.5 | 低收益暂停 | 连续3次收益<0.5%暂停复盘 | 暂停测试 |
| V88.6 | CPU预算 | 复盘CPU≤10% | 预算测试 |
| V88.7 | 安全余量 | 预留5% CPU余量 | 余量测试 |
| V88.8 | 动态调整 | CPU超预算时减小批量 | 调整测试 |
| V88.9 | 新任务终止 | 新任务到达立即终止复盘 | 终止测试 |
| V88.10 | CPU超限终止 | CPU>15%终止复盘 | 终止测试 |
| V88.11 | 进度保存 | 终止时保存复盘进度 | 保存测试 |
| V88.12 | 进度恢复 | 下次复盘从保存进度继续 | 恢复测试 |
| V88.13 | IdleReplayTrigger组合 | `trigger.ratio_enforcer` 类型为 `RatioEnforcer` | 类型检查 |
| V88.14 | 复盘收益 | 复盘后F下降>1% | 收益测试 |
| V88.15 | 端到端流程 | 空闲→触发→复盘→终止→恢复 | 集成测试 |

---

> **阶段88完** | 阶段88建立了系统空闲→离线复盘的触发器：空闲判定 cpu_usage < 0.10 AND no_active_task AND time_since_last_task > T_idle（A88.1），触发离线复盘 trigger_offline_replay()（A88.2），复盘资源分配 cpu_budget = 0.10（A88.3），复盘终止条件 new_task_available OR cpu_usage > 0.15（A88.4），封装为IdleReplayTrigger数据类组合RatioEnforcer。下一步：阶段89将建立稳态环境→探索权重上浮的触发器。
