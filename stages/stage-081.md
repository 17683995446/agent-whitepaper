# 阶段81：CPU占用监控与≤20%上限执行器

> **阶段编号**：S81 | **所属层**：第九层·安全护栏层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | CPU监控 | "看表"——实时查看自己占了多少CPU |
> | 上限判定 | "超速检测"——CPU占比超过20%就报警 |
> | 节流策略 | "踩刹车"——超了就降频、跳过低优先级、暂停探索 |
> | 恢复条件 | "松油门"——CPU降到15%以下就恢复正常运行 |
> | CPUMonitor | "CPU监控的完整画像"——组合MemoryValidationResult，在记忆验证基础上增加CPU监控状态 |

---

## 1. 第一性原理依据

本阶段从阶段1（生存条件）和阶段23（CPU占用率）的核心结论出发，建立CPU占用监控与上限执行器。核心依据为：

- **A1.8（生存条件）**（引用阶段1）：智能体的生存条件包括资源约束——CPU、内存、存储等计算资源不得超过宿主系统的合理份额。A1.8将CPU占用约束定义为智能体存在的边界条件，本阶段将这一边界条件工程化为可执行的监控与节流机制。

- **A23.2（CPU占用率）**（引用阶段23）：cpu_ratio = cpu_agent / cpu_total，智能体CPU占用与系统总CPU的比值。A23.2定义了CPU占用率的度量方式，本阶段在此基础上增加上限判定（>20%触发节流）和恢复判定（<15%恢复运行）。

本阶段的核心目标是：**定义CPU监控 cpu_ratio = cpu_agent / cpu_total（A81.1），上限判定 cpu_ratio > 0.20 → throttle()（A81.2），节流策略 reduce_frequency() / skip_low_priority() / pause_exploration()（A81.3），恢复条件 cpu_ratio < 0.15 → resume()（A81.4），封装为CPUMonitor数据类，组合MemoryValidationResult作为记忆系统验证的输入。**

> **去重声明**：本阶段仅定义CPU占用的监控与节流执行器，不重新定义生存条件（阶段1）或CPU占用率的度量方式（阶段23）。A1.8和A23.2是本阶段的理论基础，本阶段是其在安全护栏层的工程实现。

---

## 2. 核心数学公式

### 第一部分：CPU监控

#### 2.1 CPU监控定义（A81.1）

CPU监控持续度量智能体的CPU占用率：

$$\text{cpu\_ratio} = \frac{\text{cpu\_agent}}{\text{cpu\_total}} \tag{A81.1}$$

其中：
- $\text{cpu\_agent} \in [0, \text{cpu\_total}]$ 为智能体进程的CPU使用量（以CPU时间片度量）
- $\text{cpu\_total}$ 为系统可用CPU总量（核心数 × 100%）
- $\text{cpu\_ratio} \in [0, 1]$ 为智能体CPU占用率

**CPU使用量的度量方法**：

| 度量方式 | 精度 | 开销 | 说明 |
|---------|------|------|------|
| /proc/stat 读取 | 高 | 低 | Linux内核级精度 |
| psutil.cpu_percent() | 中 | 低 | Python库封装 |
| cgroup cpuacct | 高 | 中 | 容器级精确计量 |

默认采用psutil度量，容器环境切换到cgroup。

**CPU监控的采样频率**：

$$\text{sample\_interval} = \min(T_{\text{base}}, T_{\text{adaptive}}) \tag{A81.1-interval}$$

| 状态 | 采样间隔 | 说明 |
|------|---------|------|
| 正常（cpu_ratio < 0.15） | 10秒 | 低频采样，减少开销 |
| 警戒（0.15 ≤ cpu_ratio < 0.20） | 2秒 | 高频采样，提前预警 |
| 超限（cpu_ratio ≥ 0.20） | 1秒 | 最高频采样，精确控制 |

自适应采样确保在正常状态下不浪费监控开销，在超限状态下提供精确控制。

**CPU占用的滑动平均**：

为避免瞬时波动导致误判，采用指数滑动平均：

$$\overline{\text{cpu\_ratio}}_t = \alpha \cdot \text{cpu\_ratio}_t + (1 - \alpha) \cdot \overline{\text{cpu\_ratio}}_{t-1} \tag{A81.1-ema}$$

其中 $\alpha = 0.3$，平衡响应速度和平滑性。所有判定基于滑动平均值而非瞬时值。

- **输入**：系统CPU总量、智能体CPU使用量
- **输出**：cpu_ratio（瞬时值和滑动平均值）
- **约束**：采样开销 < 0.1% CPU

> **白话翻译**：CPU监控就是"看表"——实时查看自己占了多少CPU。正常时10秒看一次，快超了2秒看一次，已经超了1秒看一次。用滑动平均避免瞬时波动导致误判——不是看一眼就决定，而是看一段时间的趋势。

---

### 第二部分：上限判定

#### 2.2 上限判定定义（A81.2）

当CPU占用率超过20%时触发节流：

$$\text{should\_throttle} = \mathbb{1}\left[\overline{\text{cpu\_ratio}} > \theta_{\text{cpu\_max}}\right], \quad \theta_{\text{cpu\_max}} = 0.20 \tag{A81.2}$$

**上限阈值的选择依据**：

20%上限的设定基于A1.8的生存条件——智能体是宿主系统的一个租户，不应独占资源。具体分析：

| 场景 | 宿主CPU | 20%上限 | 智能体可用 |
|------|---------|---------|-----------|
| 个人电脑4核 | 400% | 80% | 0.8核 |
| 服务器16核 | 1600% | 320% | 3.2核 |
| 边缘设备2核 | 200% | 40% | 0.4核 |

20%上限在个人电脑上留出0.8核给智能体，足以运行推理和轻量学习；在服务器上留出3.2核，足以运行复杂模型。

**上限判定的三级预警**：

| 级别 | 条件 | 动作 |
|------|------|------|
| 绿色 | cpu_ratio < 0.15 | 正常运行 |
| 黄色 | 0.15 ≤ cpu_ratio < 0.20 | 预警，准备节流 |
| 红色 | cpu_ratio ≥ 0.20 | 立即节流 |

黄色预警阶段不执行节流，但提前准备——将低优先级任务标记为可跳过，以便红色阶段快速响应。

**上限判定的防抖机制**：

避免CPU占用在阈值附近频繁切换节流/恢复状态：

$$\text{throttle\_state}_{t} = \begin{cases} \text{THROTTLED} & \text{if } \overline{\text{cpu\_ratio}} > 0.20 \\ \text{RESUMED} & \text{if } \overline{\text{cpu\_ratio}} < 0.15 \text{ for } T_{\text{stable}} \\ \text{throttle\_state}_{t-1} & \text{otherwise} \end{cases} \tag{A81.2-debounce}$$

其中 $T_{\text{stable}} = 30$ 秒，CPU占用率需持续低于15%达30秒才恢复。滞后区间（0.15-0.20）防止状态抖动。

- **输入**：cpu_ratio滑动平均值
- **输出**：是否应节流
- **约束**：防抖滞后区间5%

> **白话翻译**：上限判定就是"超速检测"——CPU占比超过20%就报警并踩刹车。分三级：绿色（<15%）正常跑，黄色（15%-20%）准备刹车，红色（>20%）立即刹车。恢复有防抖——不是刚降到20%以下就松油门，要持续降到15%以下30秒才松。

---

### 第三部分：节流策略

#### 2.3 节流策略定义（A81.3）

节流策略按优先级递进执行，直到CPU占用率降至安全范围：

$$\text{throttle}() = \text{reduce\_frequency}() \;\oplus\; \text{skip\_low\_priority}() \;\oplus\; \text{pause\_exploration}() \tag{A81.3}$$

其中 $\oplus$ 表示按优先级递进——先执行第一级，如果仍超限则追加第二级，以此类推。

**第一级：降低执行频率 reduce_frequency()**

降低非关键任务的执行频率：

$$f_{\text{new}} = f_{\text{old}} \cdot \text{scale\_factor}, \quad \text{scale\_factor} = \frac{\theta_{\text{cpu\_max}}}{\overline{\text{cpu\_ratio}}} \tag{A81.3-freq}$$

| 任务类型 | 原始频率 | 缩放后频率 | 说明 |
|---------|---------|-----------|------|
| 感知循环 | 10Hz | 10·scale Hz | 核心循环，优先保障 |
| 模型推理 | 5Hz | 5·scale Hz | 核心功能 |
| 价值评估 | 1Hz | max(0.1, 1·scale) Hz | 可降频 |
| 记忆巩固 | 0.1Hz | max(0.01, 0.1·scale) Hz | 可大幅降频 |
| 快照保存 | 0.01Hz | max(0.001, 0.01·scale) Hz | 最低优先级 |

频率下限确保关键功能不被完全停止。

**第二级：跳过低优先级任务 skip_low_priority()**

当降频不足以控制CPU时，直接跳过低优先级任务：

$$\text{skip}(task) = \mathbb{1}\left[\text{priority}(task) < \theta_{\text{skip}}\right] \tag{A81.3-skip}$$

| 优先级 | 任务示例 | 跳过阈值 |
|--------|---------|---------|
| P0（关键） | 感知、推理、安全检查 | 永不跳过 |
| P1（重要） | 价值评估、记忆巩固 | 仅红色+第二级 |
| P2（一般） | 快照、日志、统计 | 黄色以上 |
| P3（可选） | 探索性实验、格式化 | 任何节流状态 |

**第三级：暂停探索 pause_exploration()**

当第一级和第二级仍不足以控制CPU时，暂停所有探索活动：

$$\text{pause\_exploration}() \implies \forall t \in \text{explore\_tasks}: \text{freeze}(t) \tag{A81.3-pause}$$

探索任务（引用阶段36的A36.5任务冻结机制）是CPU消耗最大的活动——模型搜索、环境仿真、策略试验等。暂停探索后，系统仅保留利用（exploit）循环，CPU占用可降至10%以下。

**节流效果预估**：

| 节流级别 | CPU降幅 | 预期cpu_ratio |
|---------|---------|--------------|
| 第一级（降频） | 30-50% | 0.10-0.14 |
| 第一+二级 | 50-70% | 0.06-0.10 |
| 第一+二+三级 | 70-90% | 0.02-0.06 |

- **输入**：当前cpu_ratio、任务列表及优先级
- **输出**：节流后的任务调度
- **约束**：P0任务永不跳过

> **白话翻译**：节流策略就是"踩刹车"——分三级踩：第一级降频（把所有任务频率按比例降低），第二级跳过低优先级（P3先跳，P2次之），第三级暂停探索（探索最耗CPU，暂停后CPU可降到10%以下）。P0关键任务（感知、推理、安全检查）永远不能跳过。

---

### 第四部分：恢复条件

#### 2.4 恢复条件定义（A81.4）

当CPU占用率持续低于15%时恢复正常运行：

$$\text{should\_resume} = \mathbb{1}\left[\overline{\text{cpu\_ratio}} < \theta_{\text{cpu\_resume}} \;\wedge\; \text{duration}(\overline{\text{cpu\_ratio}} < \theta_{\text{cpu\_resume}}) > T_{\text{stable}}\right] \tag{A81.4}$$

其中 $\theta_{\text{cpu\_resume}} = 0.15$，$T_{\text{stable}} = 30$ 秒。

**恢复的渐进式策略**：

恢复不是一次性全部放开，而是逐级恢复：

| 恢复阶段 | 条件 | 动作 |
|---------|------|------|
| 阶段1 | cpu_ratio < 0.15 持续30秒 | 恢复探索任务（解冻） |
| 阶段2 | cpu_ratio < 0.12 持续60秒 | 恢复P2任务 |
| 阶段3 | cpu_ratio < 0.10 持续120秒 | 恢复所有频率到原始值 |

渐进式恢复防止"恢复→超限→节流→恢复"的振荡循环。每个阶段都有更长的稳定时间要求和更低的CPU阈值。

**恢复后的监控增强**：

恢复后的前5分钟，采样频率保持1秒（与超限状态相同），以便快速检测CPU是否再次攀升：

$$\text{sample\_interval}_{\text{post\_resume}} = \begin{cases} 1\text{s} & t_{\text{since\_resume}} < 300\text{s} \\ 10\text{s} & t_{\text{since\_resume}} \geq 300\text{s} \end{cases} \tag{A81.4-enhanced}$$

- **输入**：cpu_ratio滑动平均值、持续时间
- **输出**：是否应恢复
- **约束**：渐进式恢复；恢复后增强监控

> **白话翻译**：恢复条件就是"松油门"——CPU持续降到15%以下30秒才开始恢复，而且不是一下子全放开：先恢复探索（阶段1），再恢复低优先级任务（阶段2），最后恢复所有频率（阶段3）。每级恢复要求CPU更低、稳定更久。恢复后5分钟内保持高频监控，防止CPU再次飙升。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A81.1 | cpu_ratio = cpu_agent / cpu_total | CPU使用量/总量 | 占用率∈[0,1] | 采样开销<0.1% |
| A81.2 | should_throttle = 𝟙[cu_ratio > 0.20] | 滑动平均cpu_ratio | 节流标志 | 防抖滞后5% |
| A81.3 | throttle() = reduce_freq ⊕ skip_low ⊕ pause_explore | cpu_ratio+任务列表 | 节流调度 | P0永不跳过 |
| A81.4 | should_resume = 𝟙[cu_ratio < 0.15 ∧ duration > 30s] | 滑动平均+持续时间 | 恢复标志 | 渐进式恢复 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import numpy as np

# 引用阶段80的数据类
from stage_080 import MemoryValidationResult


# ============================================================
# 阶段81 枚举：CPU节流级别
# ============================================================
class CPUThrottleLevel(Enum):
    """CPU节流级别。"""
    NONE = 0           # 无节流
    REDUCE_FREQUENCY = 1  # 第一级：降频
    SKIP_LOW_PRIORITY = 2  # 第二级：跳过低优先级
    PAUSE_EXPLORATION = 3  # 第三级：暂停探索


# ============================================================
# 阶段81 枚举：CPU预警级别
# ============================================================
class CPUAlertLevel(Enum):
    """CPU预警级别。"""
    GREEN = "green"    # cpu_ratio < 0.15
    YELLOW = "yellow"  # 0.15 <= cpu_ratio < 0.20
    RED = "red"        # cpu_ratio >= 0.20


# ============================================================
# 阶段81 数据类：CPU监控状态
# ============================================================
@dataclass(frozen=True)
class CPUMonitorState:
    """CPU监控的完整状态描述。

    Attributes:
        cpu_ratio: 当前CPU占用率（瞬时值）
        cpu_ratio_ema: CPU占用率指数滑动平均
        alert_level: 预警级别
        throttle_level: 当前节流级别
        throttle_duration: 当前节流持续时长（秒）
        low_cpu_duration: CPU低于恢复阈值的持续时长（秒）
        sample_interval: 当前采样间隔（秒）
        is_throttled: 是否处于节流状态
        is_resuming: 是否处于渐进恢复中
        resume_stage: 渐进恢复阶段（0=未恢复, 1/2/3=阶段）

    Invariants:
        - 0 <= cpu_ratio <= 1
        - 0 <= cpu_ratio_ema <= 1
    """
    cpu_ratio: float = 0.0
    cpu_ratio_ema: float = 0.0
    alert_level: CPUAlertLevel = CPUAlertLevel.GREEN
    throttle_level: CPUThrottleLevel = CPUThrottleLevel.NONE
    throttle_duration: float = 0.0
    low_cpu_duration: float = 0.0
    sample_interval: float = 10.0
    is_throttled: bool = False
    is_resuming: bool = False
    resume_stage: int = 0

    def __post_init__(self):
        if not (0 <= self.cpu_ratio <= 1):
            raise ValueError(
                f"cpu_ratio必须在[0,1]，收到 {self.cpu_ratio}"
            )
        if not (0 <= self.cpu_ratio_ema <= 1):
            raise ValueError(
                f"cpu_ratio_ema必须在[0,1]，收到 {self.cpu_ratio_ema}"
            )


# ============================================================
# 阶段81 数据类：CPU监控器
# ============================================================
@dataclass(frozen=True)
class CPUMonitor:
    """CPU占用监控与≤20%上限执行器的完整状态。

    组合阶段80的MemoryValidationResult，在记忆验证基础上增加
    CPU监控状态。

    Attributes:
        validation_result: 阶段80的记忆验证结果（组合字段）
        monitor_state: CPU监控状态
        cpu_max_threshold: CPU上限阈值（默认0.20）
        cpu_resume_threshold: CPU恢复阈值（默认0.15）
        ema_alpha: 指数滑动平均系数（默认0.3）
        stable_duration: 恢复所需稳定时长（默认30秒）

    Invariants:
        - cpu_max_threshold > cpu_resume_threshold
        - 0 < ema_alpha <= 1
    """
    validation_result: MemoryValidationResult
    monitor_state: CPUMonitorState = field(default_factory=CPUMonitorState)
    cpu_max_threshold: float = 0.20
    cpu_resume_threshold: float = 0.15
    ema_alpha: float = 0.3
    stable_duration: float = 30.0

    def __post_init__(self):
        if self.cpu_max_threshold <= self.cpu_resume_threshold:
            raise ValueError(
                f"上限阈值({self.cpu_max_threshold})必须大于恢复阈值({self.cpu_resume_threshold})"
            )


# ============================================================
# 阶段81 核心接口：CPU监控执行器
# ============================================================
class CPUMonitorExecutor:
    """阶段81的核心接口：CPU监控与节流执行器。

    实现CPU监控、上限判定、节流策略和恢复条件。

    核心约束：
        - CPU占用率 ≤ 20%
        - P0任务永不跳过
        - 恢复需持续30秒低于15%
    """

    def measure_cpu_ratio(self) -> float:
        """度量当前CPU占用率（A81.1）。

        Returns:
            cpu_ratio ∈ [0, 1]

        Examples:
            >>> executor = CPUMonitorExecutor()
            >>> ratio = executor.measure_cpu_ratio()
            >>> 0 <= ratio <= 1
            True
        """
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1) / 100.0
        except Exception:
            return 0.0

    def update_ema(
        self,
        current_ratio: float,
        previous_ema: float,
        alpha: float = 0.3
    ) -> float:
        """更新CPU占用率滑动平均（A81.1-ema）。

        Args:
            current_ratio: 当前CPU占用率，范围[0,1]
            previous_ema: 上一次滑动平均
            alpha: 平滑系数，范围(0,1)

        Returns:
            更新后的滑动平均

        Raises:
            ValueError: 当alpha不在(0,1)范围内时

        Examples:
            >>> executor = CPUMonitorExecutor()
            >>> executor.update_ema(0.5, 0.3, 0.3)
            0.36
        """
        if not (0 < alpha < 1):
            raise ValueError(
                f"alpha必须 ∈ (0,1)，收到 {alpha}"
            )
        return alpha * current_ratio + (1 - alpha) * previous_ema

    def check_throttle(
        self,
        cpu_ratio_ema: float,
        threshold: float = 0.20
    ) -> Tuple[bool, CPUAlertLevel]:
        """判定是否应节流（A81.2）。

        Args:
            cpu_ratio_ema: CPU占用率滑动平均
            threshold: 上限阈值

        Returns:
            (是否应节流, 预警级别)

        Examples:
            >>> executor = CPUMonitorExecutor()
            >>> should, level = executor.check_throttle(0.25)
            >>> should
            True
            >>> level == CPUAlertLevel.RED
            True
        """
        if cpu_ratio_ema >= threshold + 0.05:
            return True, CPUAlertLevel.RED
        elif cpu_ratio_ema >= threshold:
            return True, CPUAlertLevel.YELLOW
        elif cpu_ratio_ema >= threshold - 0.05:
            return False, CPUAlertLevel.YELLOW
        else:
            return False, CPUAlertLevel.GREEN

    def throttle(
        self,
        cpu_ratio_ema: float,
        current_level: CPUThrottleLevel
    ) -> CPUThrottleLevel:
        """执行节流策略（A81.3）。

        按优先级递进：降频 → 跳过低优先级 → 暂停探索

        Args:
            cpu_ratio_ema: CPU占用率滑动平均
            current_level: 当前节流级别

        Returns:
            新的节流级别

        Examples:
            >>> executor = CPUMonitorExecutor()
            >>> executor.throttle(0.25, CPUThrottleLevel.NONE)
            <CPUThrottleLevel.LEVEL1: 'level1'>
        """
        if cpu_ratio_ema < 0.15:
            return CPUThrottleLevel.NONE
        if current_level == CPUThrottleLevel.NONE:
            return CPUThrottleLevel.LEVEL1
        elif current_level == CPUThrottleLevel.LEVEL1:
            return CPUThrottleLevel.LEVEL2
        elif current_level == CPUThrottleLevel.LEVEL2:
            return CPUThrottleLevel.LEVEL3
        else:
            return CPUThrottleLevel.LEVEL3

    def reduce_frequency(
        self,
        cpu_ratio_ema: float,
        max_threshold: float = 0.20
    ) -> Dict[str, float]:
        """第一级节流：降低执行频率（A81.3-freq）。

        Args:
            cpu_ratio_ema: CPU占用率滑动平均
            max_threshold: CPU上限阈值

        Returns:
            各任务的缩放后频率字典

        Examples:
            >>> executor = CPUMonitorExecutor()
            >>> freqs = executor.reduce_frequency(0.25)
            >>> all(v <= 1.0 for v in freqs.values())
            True
        """
        scale = max(0.0, 1.0 - (cpu_ratio_ema - max_threshold))
        return {
            "exploration": scale,
            "learning": scale,
            "monitoring": 1.0
        }

    def skip_low_priority(
        self,
        min_priority_to_keep: int = 1
    ) -> List[str]:
        """第二级节流：跳过低优先级任务（A81.3-skip）。

        Args:
            min_priority_to_keep: 保留的最低优先级，必须 >= 0

        Returns:
            被跳过的任务ID列表

        Raises:
            ValueError: 当min_priority_to_keep < 0时

        Examples:
            >>> executor = CPUMonitorExecutor()
            >>> skipped = executor.skip_low_priority(1)
            >>> isinstance(skipped, list)
            True
        """
        if min_priority_to_keep < 0:
            raise ValueError(
                f"min_priority_to_keep必须 >= 0，"
                f"收到 {min_priority_to_keep}"
            )
        return ["explore_task_p2", "explore_task_p3"]

    def pause_exploration(self) -> List[str]:
        """第三级节流：暂停探索（A81.3-pause）。

        Returns:
            被冻结的探索任务ID列表

        Examples:
            >>> executor = CPUMonitorExecutor()
            >>> paused = executor.pause_exploration()
            >>> isinstance(paused, list)
            True
        """
        return ["exploration_all"]

    def check_resume(
        self,
        cpu_ratio_ema: float,
        low_cpu_duration: float,
        resume_threshold: float = 0.15,
        stable_duration: float = 30.0
    ) -> Tuple[bool, int]:
        """判定是否应恢复（A81.4）。

        Args:
            cpu_ratio_ema: CPU占用率滑动平均
            low_cpu_duration: 低于阈值的持续时长（秒）
            resume_threshold: 恢复阈值
            stable_duration: 所需稳定时长（秒），必须 > 0

        Returns:
            (是否应恢复, 恢复阶段)

        Raises:
            ValueError: 当stable_duration <= 0时

        Examples:
            >>> executor = CPUMonitorExecutor()
            >>> resume, phase = executor.check_resume(
            ...     0.10, 35.0, 0.15, 30.0)
            >>> resume
            True
        """
        if stable_duration <= 0:
            raise ValueError(
                f"stable_duration必须 > 0，收到 {stable_duration}"
            )
        if cpu_ratio_ema >= resume_threshold:
            return False, 0
        if low_cpu_duration >= stable_duration:
            return True, 3
        elif low_cpu_duration >= stable_duration * 0.66:
            return False, 2
        elif low_cpu_duration >= stable_duration * 0.33:
            return False, 1
        return False, 0

    def step(
        self,
        monitor: CPUMonitor
    ) -> CPUMonitor:
        """执行一步CPU监控与节流。

        Args:
            monitor: 当前CPU监控状态

        Returns:
            更新后的CPU监控状态

        Examples:
            >>> monitor = CPUMonitor()
            >>> executor = CPUMonitorExecutor()
            >>> new_monitor = executor.step(monitor)
            >>> isinstance(new_monitor, CPUMonitor)
            True
        """
        current_ratio = self.measure_cpu_ratio()
        new_ema = self.update_ema(
            current_ratio,
            monitor.monitor_state.cpu_ratio_ema,
            monitor.ema_alpha
        )
        should_throttle, alert = self.check_throttle(
            new_ema, monitor.cpu_max_threshold
        )
        new_level = monitor.monitor_state.throttle_level
        if should_throttle:
            new_level = self.throttle(new_ema, new_level)
        else:
            resume, _ = self.check_resume(
                new_ema,
                monitor.monitor_state.low_cpu_duration,
                monitor.cpu_resume_threshold,
                monitor.stable_duration
            )
            if resume:
                new_level = CPUThrottleLevel.NONE
        new_state = CPUMonitorState(
            cpu_ratio_ema=new_ema,
            throttle_level=new_level,
            alert_level=alert,
            low_cpu_duration=(
                monitor.monitor_state.low_cpu_duration + 1.0
                if current_ratio < monitor.cpu_resume_threshold
                else 0.0
            )
        )
        return CPUMonitor(
            monitor_state=new_state,
            cpu_max_threshold=monitor.cpu_max_threshold,
            cpu_resume_threshold=monitor.cpu_resume_threshold,
            ema_alpha=monitor.ema_alpha,
            stable_duration=monitor.stable_duration
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `MemoryValidationResult` | ← 阶段80 | 作为CPU监控的记忆验证输入（组合字段） |
| **输入** | A1.8（生存条件） | ← 阶段1 | CPU约束的理论依据 |
| **输入** | A23.2（CPU占用率） | ← 阶段23 | CPU度量的定义 |
| **输出** | `CPUMonitor` | → 阶段82 | CPU监控状态，作为内存监控的组合输入 |
| **输出** | A81.1-A81.4（CPU监控定义） | → 阶段90 | 安全护栏的CPU检查接口 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| psutil | CPU度量 | 跨平台CPU使用率监控 |
| cgroup v2 | 容器级CPU计量 | 容器环境下的精确CPU计量 |
| asyncio | 异步监控 | 非阻塞的周期性CPU采样 |
| NumPy | 滑动平均计算 | EMA的向量化实现 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V81.1 | CPU占用率度量 | cpu_ratio ∈ [0,1] | 度量测试 |
| V81.2 | EMA更新正确 | α·x + (1-α)·prev | 数值验证 |
| V81.3 | 上限判定 | cpu_ratio_ema > 0.20 → throttle | 阈值测试 |
| V81.4 | 三级预警 | 绿/黄/红正确切换 | 状态测试 |
| V81.5 | 防抖机制 | 0.15-0.20区间保持原状态 | 边界测试 |
| V81.6 | 第一级节流 | 频率按scale_factor缩放 | 频率测试 |
| V81.7 | 第二级节流 | P3任务被跳过 | 优先级测试 |
| V81.8 | 第三级节流 | 探索任务被冻结 | 冻结测试 |
| V81.9 | P0永不跳过 | 感知/推理/安全检查始终运行 | 安全测试 |
| V81.10 | 恢复条件 | cpu_ratio < 0.15 持续30秒 | 时间测试 |
| V81.11 | 渐进恢复 | 三阶段逐步恢复 | 阶段测试 |
| V81.12 | 恢复后增强监控 | 5分钟内1秒采样 | 监控测试 |
| V81.13 | CPUMonitor组合 | `monitor.validation_result` 类型为 `MemoryValidationResult` | 类型检查 |
| V81.14 | 采样开销 | 监控本身CPU开销 < 0.1% | 性能测试 |
| V81.15 | 节流效果 | 节流后cpu_ratio < 0.20 | 端到端测试 |

---

> **阶段81完** | 阶段81建立了CPU占用监控与≤20%上限执行器：CPU监控 cpu_ratio = cpu_agent / cpu_total（A81.1），上限判定 cpu_ratio > 0.20 → throttle()（A81.2），节流策略 reduce_frequency() / skip_low_priority() / pause_exploration()（A81.3），恢复条件 cpu_ratio < 0.15 → resume()（A81.4），封装为CPUMonitor数据类组合MemoryValidationResult。下一步：阶段82将建立内存占用监控与≤20%上限执行器。
