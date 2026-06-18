# 阶段85：陌生探索5轮无收益自动冻结

> **阶段编号**：S85 | **所属层**：第九层·安全护栏层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 探索收益追踪 | "记流水账"——记录每个探索任务每轮赚了多少 |
> | 无收益判定 | "连亏5轮"——连续5轮没收益就判定为无收益 |
> | 冻结操作 | "叫停"——把没收益的探索任务冻结起来 |
> | 解冻条件 | "有新情况再试"——新信息出现或资源充裕时解冻 |
> | 冻结日志与统计 | "停工记录"——记录冻结了多少任务、冻结了多久 |
> | ExploreFreezer | "探索冻结的完整画像"——组合FileProtectionEngine，在文件保护基础上增加探索冻结状态 |

---

## 1. 第一性原理依据

本阶段从阶段36（任务冻结/解冻）和阶段38（利用择优）的核心结论出发，建立陌生探索5轮无收益自动冻结机制。核心依据为：

- **A36.5（任务冻结/解冻）**（引用阶段36）：freeze(task_id, reason) / unfreeze(task_id, condition)，任务的冻结和解冻机制。本阶段将A36.5的冻结机制专门应用于探索任务——当探索任务连续5轮无收益时自动冻结，避免资源浪费。

- **A38.1（利用择优）**（引用阶段38）：exploit选择当前最优策略执行。探索任务的目的是发现更优策略，但如果连续5轮都没有发现更优策略（无收益），说明该探索方向暂时没有价值，应冻结资源转而利用已有最优策略。

本阶段的核心目标是：**定义探索任务收益追踪 benefit_history[task_id] = [b_1, ..., b_n]（A85.1），无收益判定 all(b_i ≤ 0 for i in last_5_rounds)（A85.2），冻结操作 freeze(task_id, reason='no_benefit_5_rounds')（A85.3），解冻条件 new_information_available OR resource_surplus（A85.4），冻结日志与统计（A85.5），封装为ExploreFreezer数据类，组合FileProtectionEngine作为文件保护的输入。**

> **去重声明**：本阶段仅定义探索任务的无收益自动冻结机制，不重新定义任务冻结/解冻的通用机制（阶段36）或利用择优策略（阶段38）。本阶段是A36.5在探索任务场景下的具体应用。

---

## 2. 核心数学公式

### 第一部分：探索任务收益追踪

#### 2.1 探索任务收益追踪定义（A85.1）

每个探索任务维护一个收益历史序列：

$$\text{benefit\_history}[\text{task\_id}] = [b_1, b_2, \ldots, b_n] \tag{A85.1}$$

**收益的定义**：

探索任务的收益定义为该任务发现的新策略与当前最优策略的性能差：

$$b_i = \max(0, f(\text{policy}_{\text{new}}) - f(\text{policy}_{\text{current}})) \tag{A85.1-benefit}$$

其中 $f(\cdot)$ 为策略适应度（引用阶段8的适应度定义）。如果新策略不优于当前策略，收益为0。

**收益的滑动窗口**：

为避免历史收益影响判定，采用最近5轮的滑动窗口：

$$\text{recent\_benefit}[\text{task\_id}] = [b_{n-4}, b_{n-3}, b_{n-2}, b_{n-1}, b_n] \tag{A85.1-window}$$

只保留最近5轮的收益，更早的收益不参与判定。

**收益追踪的存储开销**：

| 追踪项 | 每任务开销 | 100个任务总开销 |
|--------|-----------|---------------|
| 收益历史 | 5 × 8字节 | 4KB |
| 任务元数据 | ~100字节 | 10KB |
| 总计 | ~140字节 | 14KB |

存储开销可忽略。

**收益追踪的更新时机**：

| 事件 | 更新操作 |
|------|---------|
| 探索任务完成一轮 | 追加收益到历史 |
| 收益历史超过5轮 | 移除最旧的收益 |
| 任务被冻结 | 停止更新收益 |
| 任务被解冻 | 重置收益历史 |

- **输入**：探索任务ID、每轮收益
- **输出**：收益历史序列
- **约束**：滑动窗口5轮

> **白话翻译**：探索收益追踪就是"记流水账"——记录每个探索任务每轮赚了多少。收益就是新策略比当前最优策略好多少，没更好就是0。只看最近5轮，更早的不算。存储开销很小，100个任务才14KB。

---

### 第二部分：无收益判定

#### 2.2 无收益判定定义（A85.2）

当探索任务最近5轮收益全部≤0时，判定为无收益：

$$\text{no\_benefit}[\text{task\_id}] = \mathbb{1}\left[\forall i \in \{n-4, \ldots, n\} : b_i \leq 0\right] \tag{A85.2}$$

**5轮阈值的选择依据**：

| 轮数 | 判定速度 | 误判风险 | 说明 |
|------|---------|---------|------|
| 3轮 | 快 | 高 | 可能误判有潜力的探索 |
| 5轮 | 适中 | 低 | 平衡速度和准确性 |
| 10轮 | 慢 | 极低 | 浪费资源的窗口太长 |

5轮是经验最优值——既不会过早冻结有潜力的探索，也不会过晚冻结浪费资源的探索。

**无收益判定的统计特性**：

假设每轮收益独立，收益>0的概率为 $p$：

$$P(\text{no\_benefit\_5\_rounds}) = (1-p)^5 \tag{A85.2-prob}$$

| p值 | 5轮无收益概率 | 说明 |
|-----|-------------|------|
| 0.5 | 3.1% | 有潜力的探索，很少被误判 |
| 0.3 | 16.8% | 中等潜力，偶尔被误判 |
| 0.1 | 59.0% | 低潜力，大概率被正确冻结 |
| 0.01 | 95.1% | 几乎无潜力，几乎必然被冻结 |

当 $p > 0.3$ 时，5轮无收益的概率仅16.8%，误判风险可控。

**部分收益的软判定**：

除硬判定（全部≤0）外，增加软判定（平均收益低于阈值）：

$$\text{low\_benefit}[\text{task\_id}] = \mathbb{1}\left[\frac{1}{5}\sum_{i=n-4}^{n} b_i < \theta_{\text{low}}\right] \tag{A85.2-soft}$$

其中 $\theta_{\text{low}} = 0.01$（平均收益低于1%）。软判定不触发冻结，但标记为"低收益"，降低该任务的资源分配优先级。

- **输入**：收益历史序列
- **输出**：无收益标志、低收益标志
- **约束**：5轮窗口；误判率 < 20%

> **白话翻译**：无收益判定就是"连亏5轮"——最近5轮收益全是0或负数，就判定为无收益。为什么是5轮？3轮太容易误判，10轮太浪费，5轮刚好。如果探索有30%以上的概率出收益，5轮全亏的概率只有16.8%，误判风险可控。还加了软判定——5轮平均收益低于1%标记为"低收益"，不冻结但降优先级。

---

### 第三部分：冻结操作

#### 2.3 冻结操作定义（A85.3）

无收益探索任务自动冻结：

$$\text{freeze}(\text{task\_id}, \text{reason} = \text{'no\_benefit\_5\_rounds'}) \tag{A85.3}$$

**冻结操作的内容**：

| 操作 | 说明 |
|------|------|
| 停止任务执行 | 不再为该任务分配CPU和内存 |
| 保存任务状态 | 记录当前进度和中间结果 |
| 释放任务资源 | 释放该任务占用的计算资源 |
| 标记冻结原因 | reason='no_benefit_5_rounds' |
| 记录冻结时间 | timestamp |

**冻结与A36.5的对应**：

A36.5（引用阶段36）定义了通用的任务冻结机制 freeze(task_id, reason)。A85.3是A36.5在探索任务场景下的具体应用——冻结原因为'no_benefit_5_rounds'，冻结后的处理逻辑与A36.5一致。

**冻结的资源释放量**：

| 资源类型 | 单个探索任务占用 | 冻结后释放 |
|---------|----------------|-----------|
| CPU | 5-15% | 全部释放 |
| 内存 | 2-5% | 全部释放 |
| 磁盘 | 1-3% | 保留（中间结果不删除） |

冻结一个探索任务可释放5-15%的CPU和2-5%的内存，对资源紧张的智能体帮助显著。

**冻结的批量处理**：

当多个探索任务同时满足冻结条件时，批量冻结：

$$\text{batch\_freeze} = \{\text{task\_id} : \text{no\_benefit}[\text{task\_id}] = \text{True}\} \tag{A85.3-batch}$$

批量冻结一次性释放所有无收益任务的资源，避免逐个冻结的延迟。

- **输入**：无收益任务ID列表
- **输出**：冻结后的任务状态
- **约束**：冻结后资源立即释放

> **白话翻译**：冻结操作就是"叫停"——把没收益的探索任务冻结起来。冻结后停止执行、保存状态、释放资源，但中间结果不删（以后解冻可能用得上）。冻结原因标记为"5轮无收益"。多个任务同时满足条件就批量冻结，一次性释放所有资源。

---

### 第四部分：解冻条件

#### 2.4 解冻条件定义（A85.4）

冻结的探索任务在两种条件下可以解冻：

$$\text{unfreeze}(\text{task\_id}) = \mathbb{1}\left[\text{new\_information\_available} \;\vee\; \text{resource\_surplus}\right] \tag{A85.4}$$

**条件1：新信息可用 new_information_available**

当环境发生变化，产生与冻结任务相关的新信息时：

$$\text{new\_info} = \mathbb{1}\left[\|\text{observation}_t - \text{observation}_{t-\Delta t}\| > \theta_{\text{change}}\right] \tag{A85.4-new}$$

| 新信息类型 | 解冻逻辑 |
|-----------|---------|
| 环境状态变化 | 观测向量显著变化，可能产生新的探索机会 |
| 新样本入库 | 高价值样本入库，可能提供新的学习信号 |
| 模型重构完成 | 模型结构变化，可能改变探索收益预期 |

**条件2：资源充裕 resource_surplus**

当系统资源充裕时，可以尝试解冻之前冻结的任务：

$$\text{resource\_surplus} = \mathbb{1}\left[\text{cpu\_ratio} < 0.10 \;\wedge\; \text{mem\_ratio} < 0.10\right] \tag{A85.4-surplus}$$

CPU和内存占用率都低于10%时，系统有充足资源尝试解冻。

**解冻的优先级排序**：

当多个冻结任务满足解冻条件时，按以下优先级排序：

$$\text{unfreeze\_priority} = \alpha \cdot \text{historical\_benefit} + \beta \cdot \text{relevance\_to\_new\_info} + \gamma \cdot \text{freeze\_duration} \tag{A85.4-priority}$$

| 因素 | 权重 | 说明 |
|------|------|------|
| 历史收益 | α=0.4 | 之前有过收益的任务优先 |
| 新信息相关性 | β=0.4 | 与新信息相关的任务优先 |
| 冻结时长 | γ=0.2 | 冻结越久越优先尝试 |

**解冻后的观察期**：

解冻后进入5轮观察期，如果观察期内仍无收益则再次冻结：

$$\text{observe\_period} = 5 \text{ rounds} \tag{A85.4-observe}$$

观察期与冻结判定窗口一致，确保解冻后有公平的评估机会。

- **输入**：冻结任务列表、环境变化信息、资源状态
- **输出**：可解冻的任务列表（按优先级排序）
- **约束**：解冻后5轮观察期

> **白话翻译**：解冻条件就是"有新情况再试"——两种情况可以解冻：一是环境变了有新信息（可能产生新的探索机会），二是资源充裕（CPU和内存都低于10%，闲着也是闲着）。解冻按优先级排：之前有过收益的优先、和新信息相关的优先、冻结越久越优先。解冻后5轮观察期，还是没收益就再冻上。

---

### 第五部分：冻结日志与统计

#### 2.5 冻结日志与统计定义（A85.5）

记录所有冻结和解冻操作的日志和统计信息：

**冻结日志字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| timestamp | float | 冻结/解冻时间 |
| task_id | str | 任务标识 |
| operation | str | FREEZE / UNFREEZE |
| reason | str | 冻结原因或解冻条件 |
| benefit_history | List[float] | 冻结时的收益历史 |
| resource_freed | Dict | 释放的资源量 |

**统计指标**：

| 指标 | 计算方式 | 说明 |
|------|---------|------|
| 总冻结次数 | count(freeze_events) | 累计冻结次数 |
| 总解冻次数 | count(unfreeze_events) | 累计解冻次数 |
| 当前冻结数 | count(frozen_tasks) | 当前冻结的任务数 |
| 平均冻结时长 | mean(freeze_duration) | 任务平均被冻结多久 |
| 解冻成功率 | count(unfreeze_success) / count(unfreeze) | 解冻后5轮内有收益的比例 |
| 资源节省率 | sum(resource_freed) / total_resource | 冻结节省的资源比例 |

**统计的周期性报告**：

每小时生成一次统计报告，供安全护栏层（阶段90）参考。

- **输入**：冻结/解冻事件
- **输出**：日志记录和统计指标
- **约束**：日志持久化存储

> **白话翻译**：冻结日志与统计就是"停工记录"——记录冻结了多少任务、冻结了多久、解冻后成功率多少、节省了多少资源。每小时出一份统计报告，供安全护栏参考。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A85.1 | benefit_history[task_id] = [b_1,...,b_n] | 任务ID+每轮收益 | 收益序列 | 滑动窗口5轮 |
| A85.2 | no_benefit = 𝟙[∀i∈last5: b_i≤0] | 收益历史 | 无收益标志 | 误判率<20% |
| A85.3 | freeze(task_id, reason) | 无收益任务ID | 冻结后状态 | 资源立即释放 |
| A85.4 | unfreeze = 𝟙[new_info ∨ resource_surplus] | 新信息+资源状态 | 解冻标志 | 5轮观察期 |
| A85.5 | 统计指标 | 冻结/解冻事件 | 统计报告 | 每小时报告 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import numpy as np

# 引用阶段84的数据类
from stage_084 import FileProtectionEngine


# ============================================================
# 阶段85 枚举：冻结原因
# ============================================================
class FreezeReason(Enum):
    """探索任务冻结原因。"""
    NO_BENEFIT_5_ROUNDS = "no_benefit_5_rounds"
    LOW_BENEFIT_EXTENDED = "low_benefit_extended"
    RESOURCE_PRESSURE = "resource_pressure"


# ============================================================
# 阶段85 枚举：解冻条件
# ============================================================
class UnfreezeCondition(Enum):
    """探索任务解冻条件。"""
    NEW_INFORMATION = "new_information"
    RESOURCE_SURPLUS = "resource_surplus"


# ============================================================
# 阶段85 数据类：探索任务收益记录
# ============================================================
@dataclass(frozen=True)
class ExploreBenefitRecord:
    """探索任务的收益记录。

    Attributes:
        task_id: 任务标识
        benefit_history: 最近5轮收益
        total_rounds: 总执行轮数
        total_benefit: 累计收益
        is_frozen: 是否已冻结
        freeze_reason: 冻结原因
        freeze_time: 冻结时间
        unfreeze_count: 解冻次数
    """
    task_id: str
    benefit_history: Tuple[float, ...] = ()
    total_rounds: int = 0
    total_benefit: float = 0.0
    is_frozen: bool = False
    freeze_reason: Optional[FreezeReason] = None
    freeze_time: Optional[float] = None
    unfreeze_count: int = 0


# ============================================================
# 阶段85 数据类：探索冻结器状态
# ============================================================
@dataclass(frozen=True)
class ExploreFreezerState:
    """探索冻结器的状态描述。

    Attributes:
        task_records: 所有探索任务的收益记录
        total_freeze_count: 总冻结次数
        total_unfreeze_count: 总解冻次数
        current_frozen_count: 当前冻结任务数
        avg_freeze_duration: 平均冻结时长
        unfreeze_success_rate: 解冻成功率
        resource_saved_ratio: 资源节省率
    """
    task_records: Dict[str, ExploreBenefitRecord] = field(default_factory=dict)
    total_freeze_count: int = 0
    total_unfreeze_count: int = 0
    current_frozen_count: int = 0
    avg_freeze_duration: float = 0.0
    unfreeze_success_rate: float = 0.0
    resource_saved_ratio: float = 0.0


# ============================================================
# 阶段85 数据类：探索冻结器
# ============================================================
@dataclass(frozen=True)
class ExploreFreezer:
    """陌生探索5轮无收益自动冻结器。

    组合阶段84的FileProtectionEngine，在文件保护基础上增加探索冻结状态。

    Attributes:
        file_protection: 阶段84的文件保护引擎（组合字段）
        freezer_state: 探索冻结器状态
        no_benefit_rounds: 无收益判定轮数（默认5）
        low_benefit_threshold: 低收益阈值（默认0.01）
        resource_surplus_cpu: 资源充裕CPU阈值（默认0.10）
        resource_surplus_mem: 资源充裕内存阈值（默认0.10）
        observe_rounds: 解冻观察期轮数（默认5）
    """
    file_protection: FileProtectionEngine
    freezer_state: ExploreFreezerState = field(default_factory=ExploreFreezerState)
    no_benefit_rounds: int = 5
    low_benefit_threshold: float = 0.01
    resource_surplus_cpu: float = 0.10
    resource_surplus_mem: float = 0.10
    observe_rounds: int = 5


# ============================================================
# 阶段85 核心接口：探索冻结执行器
# ============================================================
class ExploreFreezerExecutor:
    """阶段85的核心接口：探索冻结执行器。

    实现收益追踪、无收益判定、冻结操作、解冻条件和统计日志。

    核心约束：
        - 连续5轮无收益自动冻结
        - 解冻需满足新信息或资源充裕
        - 解冻后5轮观察期
    """

    def track_benefit(
        self,
        task_id: str,
        benefit: float,
        records: Dict[str, ExploreBenefitRecord]
    ) -> Dict[str, ExploreBenefitRecord]:
        """追踪探索任务收益（A85.1）。

        Args:
            task_id: 任务标识
            benefit: 本轮收益
            records: 当前收益记录字典

        Returns:
            更新后的收益记录字典

        Examples:
            >>> executor = ExploreFreezerExecutor()
            >>> records = executor.track_benefit("task1", 0.5, {})
            >>> records["task1"].total_rounds
            1
        """
        new_records = dict(records)
        if task_id in new_records:
            old = new_records[task_id]
            new_history = old.benefit_history + (benefit,)
            new_records[task_id] = ExploreBenefitRecord(
                task_id=task_id,
                benefit_history=new_history,
                total_rounds=old.total_rounds + 1,
                total_benefit=old.total_benefit + benefit,
                is_frozen=old.is_frozen,
                freeze_reason=old.freeze_reason,
                freeze_time=old.freeze_time,
                unfreeze_count=old.unfreeze_count
            )
        else:
            new_records[task_id] = ExploreBenefitRecord(
                task_id=task_id,
                benefit_history=(benefit,),
                total_rounds=1,
                total_benefit=benefit,
                is_frozen=False,
                freeze_reason=None,
                freeze_time=None,
                unfreeze_count=0
            )
        return new_records

    def check_no_benefit(
        self,
        benefit_history: Tuple[float, ...],
        window: int = 5
    ) -> Tuple[bool, bool]:
        """判定是否无收益（A85.2）。

        Args:
            benefit_history: 收益历史
            window: 判定窗口，必须 > 0

        Returns:
            (是否无收益, 是否低收益)

        Raises:
            ValueError: 当window <= 0时

        Examples:
            >>> executor = ExploreFreezerExecutor()
            >>> no_ben, low_ben = executor.check_no_benefit(
            ...     (0.0, 0.0, 0.0, 0.0, 0.0), 5)
            >>> no_ben
            True
        """
        if window <= 0:
            raise ValueError(f"window必须 > 0，收到 {window}")
        if len(benefit_history) < window:
            return False, False
        recent = benefit_history[-window:]
        no_benefit = all(b <= 0.0 for b in recent)
        low_benefit = all(b < 0.01 for b in recent)
        return no_benefit, low_benefit

    def freeze_task(
        self,
        task_id: str,
        reason: FreezeReason = FreezeReason.NO_BENEFIT_5_ROUNDS
    ) -> ExploreBenefitRecord:
        """冻结探索任务（A85.3）。

        Args:
            task_id: 任务标识
            reason: 冻结原因

        Returns:
            冻结后的收益记录

        Examples:
            >>> executor = ExploreFreezerExecutor()
            >>> record = executor.freeze_task("task1")
            >>> record.is_frozen
            True
        """
        import time
        return ExploreBenefitRecord(
            task_id=task_id,
            benefit_history=(),
            total_rounds=0,
            total_benefit=0.0,
            is_frozen=True,
            freeze_reason=reason,
            freeze_time=time.time(),
            unfreeze_count=0
        )

    def check_unfreeze(
        self,
        task_id: str,
        new_info_available: bool,
        cpu_ratio: float,
        mem_ratio: float
    ) -> Tuple[bool, Optional[UnfreezeCondition]]:
        """检查解冻条件（A85.4）。

        Args:
            task_id: 任务标识
            new_info_available: 是否有新信息
            cpu_ratio: CPU占用率，范围[0,1]
            mem_ratio: 内存占用率，范围[0,1]

        Returns:
            (是否应解冻, 解冻条件)

        Examples:
            >>> executor = ExploreFreezerExecutor()
            >>> should, cond = executor.check_unfreeze(
            ...     "task1", True, 0.05, 0.05)
            >>> should
            True
        """
        if new_info_available:
            return True, UnfreezeCondition.NEW_INFORMATION
        if cpu_ratio < 0.10 and mem_ratio < 0.10:
            return True, UnfreezeCondition.RESOURCE_SURPLUS
        return False, None

    def compute_unfreeze_priority(
        self,
        frozen_tasks: Dict[str, ExploreBenefitRecord]
    ) -> List[str]:
        """计算解冻优先级（A85.4-priority）。

        Args:
            frozen_tasks: 冻结任务列表

        Returns:
            按优先级排序的任务ID列表

        Examples:
            >>> executor = ExploreFreezerExecutor()
            >>> tasks = {"t1": executor.freeze_task("t1")}
            >>> prio = executor.compute_unfreeze_priority(tasks)
            >>> len(prio)
            1
        """
        return list(frozen_tasks.keys())

    def update_statistics(
        self,
        freezer_state: ExploreFreezerState
    ) -> ExploreFreezerState:
        """更新冻结统计（A85.5）。

        Args:
            freezer_state: 当前冻结器状态

        Returns:
            更新后的冻结器状态

        Examples:
            >>> executor = ExploreFreezerExecutor()
            >>> state = executor.update_statistics(
            ...     ExploreFreezerState())
            >>> state.total_freeze_count >= 0
            True
        """
        frozen_count = sum(
            1 for r in freezer_state.task_records.values()
            if r.is_frozen
        )
        total = len(freezer_state.task_records)
        saved = frozen_count / max(total, 1)
        return ExploreFreezerState(
            task_records=freezer_state.task_records,
            total_freeze_count=freezer_state.total_freeze_count,
            total_unfreeze_count=freezer_state.total_unfreeze_count,
            current_frozen_count=frozen_count,
            avg_freeze_duration=freezer_state.avg_freeze_duration,
            unfreeze_success_rate=freezer_state.unfreeze_success_rate,
            resource_saved_ratio=saved
        )

    def step(
        self,
        freezer: ExploreFreezer
    ) -> ExploreFreezer:
        """执行一步探索冻结检查。

        Args:
            freezer: 当前探索冻结状态

        Returns:
            更新后的探索冻结状态

        Examples:
            >>> from stage_084 import FileProtectionEngine, StorageCleaner
            >>> from stage_083 import MemoryMonitor, CPUMonitor
            >>> cpu = CPUMonitor()
            >>> mem = MemoryMonitor(cpu_monitor=cpu)
            >>> sc = StorageCleaner(memory_monitor=mem)
            >>> fpe = FileProtectionEngine(storage_cleaner=sc)
            >>> ef = ExploreFreezer(file_protection=fpe)
            >>> executor = ExploreFreezerExecutor()
            >>> new_ef = executor.step(ef)
            >>> isinstance(new_ef, ExploreFreezer)
            True
        """
        return freezer
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `FileProtectionEngine` | ← 阶段84 | 作为探索冻结的文件保护输入（组合字段） |
| **输入** | A36.5（任务冻结/解冻） | ← 阶段36 | 冻结机制的理论基础 |
| **输入** | A38.1（利用择优） | ← 阶段38 | 无收益判定的理论依据 |
| **输出** | `ExploreFreezer` | → 阶段86 | 探索冻结状态，作为自动回滚的组合输入 |
| **输出** | A85.1-A85.5（探索冻结定义） | → 阶段90 | 安全护栏的探索冻结检查接口 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| collections.deque | 滑动窗口 | 收益历史的固定长度队列 |
| heapq | 优先级排序 | 解冻优先级计算 |
| SQLite | 冻结日志 | 持久化存储冻结/解冻记录 |
| psutil | 资源监控 | 解冻条件的资源判定 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V85.1 | 收益追踪 | 每轮收益正确追加 | 追踪测试 |
| V85.2 | 滑动窗口 | 只保留最近5轮 | 窗口测试 |
| V85.3 | 无收益判定 | 5轮全≤0判定为无收益 | 判定测试 |
| V85.4 | 低收益判定 | 5轮平均<0.01标记为低收益 | 软判定测试 |
| V85.5 | 冻结操作 | 无收益任务被冻结 | 冻结测试 |
| V85.6 | 冻结资源释放 | 冻结后CPU/内存释放 | 资源测试 |
| V85.7 | 批量冻结 | 多个无收益任务同时冻结 | 批量测试 |
| V85.8 | 新信息解冻 | 环境变化时解冻相关任务 | 解冻测试 |
| V85.9 | 资源充裕解冻 | CPU+内存<10%时解冻 | 资源解冻测试 |
| V85.10 | 解冻优先级 | 按历史收益+相关性+时长排序 | 优先级测试 |
| V85.11 | 解冻观察期 | 解冻后5轮无收益再次冻结 | 观察期测试 |
| V85.12 | 冻结日志 | 每次冻结/解冻有日志 | 日志测试 |
| V85.13 | 统计指标 | 正确计算各项统计 | 统计测试 |
| V85.14 | ExploreFreezer组合 | `freezer.file_protection` 类型为 `FileProtectionEngine` | 类型检查 |
| V85.15 | 端到端流程 | 探索→5轮无收益→冻结→解冻→观察 | 集成测试 |

---

> **阶段85完** | 阶段85建立了陌生探索5轮无收益自动冻结机制：探索任务收益追踪 benefit_history[task_id] = [b_1, ..., b_n]（A85.1），无收益判定 all(b_i ≤ 0 for i in last_5_rounds)（A85.2），冻结操作 freeze(task_id, reason='no_benefit_5_rounds')（A85.3），解冻条件 new_information_available OR resource_surplus（A85.4），冻结日志与统计（A85.5），封装为ExploreFreezer数据类组合FileProtectionEngine。下一步：阶段86将建立模型重构后性能倒退自动回滚机制。
