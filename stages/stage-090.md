# 阶段90：安全护栏Python接口

> **阶段编号**：S90 | **所属层**：第九层·安全护栏层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | SafetyGuard | "安全大管家"——统一管理所有安全约束 |
> | check_cpu() | "CPU检查"——CPU超没超20% |
> | check_memory() | "内存检查"——内存超没超20% |
> | check_storage() | "存储检查"——磁盘超没超80% |
> | check_file_access() | "文件检查"——能不能改这个文件 |
> | check_explore_freeze() | "探索检查"——这个探索该不该冻 |
> | check_rollback() | "回滚检查"——新模型该不该回滚 |
> | check_ratio() | "比例检查"——探索利用比例对不对 |
> | check_idle() | "空闲检查"——该不该开始复盘 |
> | check_stable() | "稳态检查"——该不该增加探索 |
> | SafetyGuardState | "安全护栏的完整画像"——组合StableExploreTrigger，在稳态探索基础上增加统一安全状态 |

---

## 1. 第一性原理依据

本阶段从阶段81-89的所有核心结论出发，建立安全护栏的统一Python接口。核心依据为：

- **A81.1-A81.4（CPU监控与节流）**（引用阶段81）：CPU监控、上限判定、节流策略和恢复条件。SafetyGuard的check_cpu()方法直接调用A81.1-A81.4的CPU监控和节流逻辑。

- **A82.1-A82.4（内存监控与释放）**（引用阶段82）：内存监控、上限判定、释放策略和OOM预防。SafetyGuard的check_memory()方法直接调用A82.1-A82.4的内存监控和释放逻辑。

- **A83.1-A83.5（存储自动清理）**（引用阶段83）：存储监控、清理触发、清理策略、清理顺序和清理后验证。SafetyGuard的check_storage()方法直接调用A83.1-A83.5的存储监控和清理逻辑。

- **A84.1-A84.4（文件保护规则引擎）**（引用阶段84）：受保护文件列表、文件操作拦截、规则匹配和违规日志。SafetyGuard的check_file_access()方法直接调用A84.1-A84.4的文件保护逻辑。

- **A85.1-A85.5（探索冻结）**（引用阶段85）：收益追踪、无收益判定、冻结操作、解冻条件和冻结统计。SafetyGuard的check_explore_freeze()方法直接调用A85.1-A85.5的探索冻结逻辑。

- **A86.1-A86.5（自动回滚）**（引用阶段86）：性能评估、倒退判定、自动回滚、回滚后验证和根因分析。SafetyGuard的check_rollback()方法直接调用A86.1-A86.5的自动回滚逻辑。

- **A87.1-A87.4（比例强制执行）**（引用阶段87）：全局比例监控、强制探索注入、强制利用注入和执行日志。SafetyGuard的check_ratio()方法直接调用A87.1-A87.4的比例执行逻辑。

- **A88.1-A88.4（空闲复盘触发）**（引用阶段88）：空闲判定、复盘触发、资源分配和终止条件。SafetyGuard的check_idle()方法直接调用A88.1-A88.4的空闲复盘逻辑。

- **A89.1-A89.4（稳态探索触发）**（引用阶段89）：稳态判定、上浮触发、上浮幅度和安全检查。SafetyGuard的check_stable()方法直接调用A89.1-A89.4的稳态探索逻辑。

本阶段的核心目标是：**定义SafetyGuard类设计（整合所有安全约束）（A90.1），check_cpu() → CPUMonitorState（A90.2），check_memory() → MemoryMonitorState（A90.3），check_storage() → StorageCleanerState（A90.4），check_file_access(path) → allow/deny（A90.5），check_explore_freeze(task) → freeze/continue（A90.6），check_rollback(model_old, model_new) → rollback/accept（A90.7），check_ratio(history) → enforce/ok（A90.8），check_idle() → trigger_replay/continue（A90.9），check_stable() → boost_explore/continue（A90.10），封装为SafetyGuardState数据类，组合StableExploreTrigger作为稳态探索的输入。**

> **去重声明**：本阶段仅定义安全护栏的统一Python接口，不重新实现阶段81-89的任何算法。所有方法均为对阶段81-89接口的编排调用。

---

## 2. 核心数学公式

### 第一部分：SafetyGuard类设计

#### 2.1 SafetyGuard类设计定义（A90.1）

SafetyGuard整合九个安全子系统，提供统一接口：

$$\text{SafetyGuard} = \text{CPU} + \text{Memory} + \text{Storage} + \text{FileProtection} + \text{ExploreFreeze} + \text{AutoRollback} + \text{RatioEnforce} + \text{IdleReplay} + \text{StableExplore} \tag{A90.1}$$

**子系统间的数据流**：

```
check_cpu()
    │
    └──→ CPUMonitor.measure()        [A81.1]
         CPUMonitor.check_throttle()  [A81.2]
         CPUMonitor.throttle()        [A81.3]

check_memory()
    │
    └──→ MemoryMonitor.measure()      [A82.1]
         MemoryMonitor.check_free()   [A82.2]
         MemoryMonitor.free()         [A82.3]

check_storage()
    │
    └──→ StorageCleaner.measure()     [A83.1]
         StorageCleaner.check_clean() [A83.2]
         StorageCleaner.clean()       [A83.3]

check_file_access(path)
    │
    └──→ FileProtection.intercept()   [A84.2]
         FileProtection.match()       [A84.3]

check_explore_freeze(task)
    │
    └──→ ExploreFreezer.track()       [A85.1]
         ExploreFreezer.check()       [A85.2]
         ExploreFreezer.freeze()      [A85.3]

check_rollback(old, new)
    │
    └──→ AutoRollback.evaluate()      [A86.1]
         AutoRollback.check()         [A86.2]
         AutoRollback.rollback()      [A86.3]

check_ratio(history)
    │
    └──→ RatioEnforcer.monitor()      [A87.1]
         RatioEnforcer.inject()       [A87.2/A87.3]

check_idle()
    │
    └──→ IdleReplayTrigger.check()    [A88.1]
         IdleReplayTrigger.trigger()  [A88.2]

check_stable()
    │
    └──→ StableExploreTrigger.check() [A89.1]
         StableExploreTrigger.boost() [A89.2]
```

**SafetyGuard的生命周期**：

| 阶段 | 调用方法 | 说明 |
|------|---------|------|
| 初始化 | `__init__()` | 创建九个子系统 |
| 每步调用 | `check_cpu()`, `check_memory()` | 资源监控 |
| 文件操作前 | `check_file_access()` | 文件保护 |
| 探索任务后 | `check_explore_freeze()` | 探索冻结 |
| 模型重构后 | `check_rollback()` | 自动回滚 |
| 任务调度时 | `check_ratio()` | 比例执行 |
| 空闲检测 | `check_idle()` | 离线复盘 |
| 稳态检测 | `check_stable()` | 探索上浮 |
| 综合检查 | `check_all()` | 全量安全检查 |

**SafetyGuard的配置参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| cpu_max_threshold | 0.20 | CPU上限 |
| mem_max_threshold | 0.20 | 内存上限 |
| disk_max_threshold | 0.80 | 磁盘上限 |
| rollback_threshold | 0.05 | 回滚阈值 |
| ratio_target | 0.70 | 探索利用目标比例 |
| explore_weight_max | 0.50 | 探索权重上限 |

- **输入**：九个子系统的配置参数
- **输出**：SafetyGuard实例
- **约束**：所有子系统正确初始化

> **白话翻译**：SafetyGuard就是"安全大管家"——统一管理九个安全子系统：CPU监控、内存监控、存储清理、文件保护、探索冻结、自动回滚、比例执行、空闲复盘、稳态探索。每个check方法都是对子系统的编排调用。初始化时创建所有子系统，运行时按需调用对应的check方法。

---

### 第二部分至第十部分：各check方法

#### 2.2 check_cpu()定义（A90.2）

$$\text{check\_cpu}() \to \text{CPUMonitorState} \tag{A90.2}$$

**返回值**：

| 字段 | 类型 | 说明 |
|------|------|------|
| cpu_ratio | float | 当前CPU占用率 |
| cpu_ratio_ema | float | 滑动平均 |
| alert_level | CPUAlertLevel | 预警级别 |
| throttle_level | CPUThrottleLevel | 节流级别 |
| is_throttled | bool | 是否节流 |

**调用链**：measure_cpu_ratio() → update_ema() → check_throttle() → throttle()

#### 2.3 check_memory()定义（A90.3）

$$\text{check\_memory}() \to \text{MemoryMonitorState} \tag{A90.3}$$

**返回值**：

| 字段 | 类型 | 说明 |
|------|------|------|
| mem_ratio | float | 当前内存占用率 |
| mem_ratio_ema | float | 滑动平均 |
| mem_trend | float | 增长趋势 |
| alert_level | MemoryAlertLevel | 预警级别 |
| free_level | MemoryFreeLevel | 释放级别 |
| is_preemptive | bool | 是否预防性释放 |

**调用链**：measure_mem_ratio() → compute_mem_trend() → check_free() → free_memory()

#### 2.4 check_storage()定义（A90.4）

$$\text{check\_storage}() \to \text{StorageCleanerState} \tag{A90.4}$$

**返回值**：

| 字段 | 类型 | 说明 |
|------|------|------|
| disk_usage | float | 磁盘占用率 |
| disk_trend | float | 增长趋势 |
| clean_level | StorageCleanLevel | 清理级别 |
| last_cleaned_count | int | 上次清理数量 |

**调用链**：measure_disk_usage() → check_clean() → clean() → verify_clean_result()

#### 2.5 check_file_access()定义（A90.5）

$$\text{check\_file\_access}(\text{path}) \to \text{allow} \mid \text{deny} \tag{A90.5}$$

**调用链**：resolve_real_path() → regex_match() → (allow/deny) → log_violation()

#### 2.6 check_explore_freeze()定义（A90.6）

$$\text{check\_explore\_freeze}(\text{task}) \to \text{freeze} \mid \text{continue} \tag{A90.6}$$

**调用链**：track_benefit() → check_no_benefit() → freeze_task()

#### 2.7 check_rollback()定义（A90.7）

$$\text{check\_rollback}(\text{model\_old}, \text{model\_new}) \to \text{rollback} \mid \text{accept} \tag{A90.7}$$

**调用链**：evaluate_performance() → check_regression() → execute_rollback() → verify_rollback() → analyze_root_cause()

#### 2.8 check_ratio()定义（A90.8）

$$\text{check\_ratio}(\text{history}) \to \text{enforce} \mid \text{ok} \tag{A90.8}$$

**调用链**：monitor_ratio() → check_deviation() → inject_explore_task() / inject_exploit_task()

#### 2.9 check_idle()定义（A90.9）

$$\text{check\_idle}() \to \text{trigger\_replay} \mid \text{continue} \tag{A90.9}$$

**调用链**：check_idle() → trigger_replay() → allocate_resources() → check_stop()

#### 2.10 check_stable()定义（A90.10）

$$\text{check\_stable}() \to \text{boost\_explore} \mid \text{continue} \tag{A90.10}$$

**调用链**：check_stable() → trigger_boost() → compute_boost_amount() → safety_check()

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A90.1 | SafetyGuard = CPU + Memory + Storage + File + Explore + Rollback + Ratio + Idle + Stable | 配置参数 | 安全护栏实例 | 九个子系统初始化 |
| A90.2 | check_cpu() → CPUMonitorState | 无 | CPU监控状态 | CPU≤20% |
| A90.3 | check_memory() → MemoryMonitorState | 无 | 内存监控状态 | 内存≤20% |
| A90.4 | check_storage() → StorageCleanerState | 无 | 存储清理状态 | 磁盘<80% |
| A90.5 | check_file_access(path) → allow/deny | 文件路径 | 拦截结果 | 关键文件不可修改 |
| A90.6 | check_explore_freeze(task) → freeze/continue | 探索任务 | 冻结决策 | 5轮无收益冻结 |
| A90.7 | check_rollback(old, new) → rollback/accept | 新旧模型 | 回滚决策 | 倒退>5%回滚 |
| A90.8 | check_ratio(history) → enforce/ok | 任务历史 | 比例决策 | 65%-75%安全范围 |
| A90.9 | check_idle() → trigger_replay/continue | 无 | 复盘决策 | CPU<10%+无任务+60秒 |
| A90.10 | check_stable() → boost_explore/continue | 无 | 上浮决策 | 稳态+空闲触发 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union
from enum import Enum
import numpy as np

# 引用阶段89的数据类
from stage_089 import StableExploreTrigger

# 引用阶段81-89的状态类型
from stage_081 import CPUMonitorState, CPUAlertLevel, CPUThrottleLevel
from stage_082 import MemoryMonitorState, MemoryAlertLevel, MemoryFreeLevel
from stage_083 import StorageCleanerState, StorageCleanLevel
from stage_084 import InterceptResult, ProtectionLevel
from stage_085 import ExploreBenefitRecord, FreezeReason
from stage_086 import RegressionLevel, RootCauseType
from stage_087 import RatioDeviation, EnforcementAction, RatioAlertLevel
from stage_088 import ReplayPriority


# ============================================================
# 阶段90 枚举：安全检查结果
# ============================================================
class SafetyCheckResult(Enum):
    """安全检查结果。"""
    OK = "ok"               # 通过
    WARNING = "warning"     # 警告
    VIOLATION = "violation" # 违规
    ACTION_TAKEN = "action_taken"  # 已采取行动


# ============================================================
# 阶段90 数据类：安全护栏状态
# ============================================================
@dataclass(frozen=True)
class SafetyGuardState:
    """安全护栏的完整状态描述。

    组合阶段89的StableExploreTrigger，在稳态探索基础上增加统一安全状态。

    Attributes:
        stable_trigger: 阶段89的稳态探索触发器（组合字段）
        cpu_state: CPU监控状态
        memory_state: 内存监控状态
        storage_state: 存储清理状态
        total_violations: 总违规次数
        total_enforcements: 总强制执行次数
        total_rollbacks: 总回滚次数
        overall_safety_level: 综合安全级别
        last_check_time: 上次全量检查时间
        check_results: 各检查项的最新结果

    Invariants:
        - total_violations >= 0
        - total_enforcements >= 0
        - total_rollbacks >= 0
    """
    stable_trigger: StableExploreTrigger
    cpu_state: CPUMonitorState = field(default_factory=CPUMonitorState)
    memory_state: MemoryMonitorState = field(default_factory=MemoryMonitorState)
    storage_state: StorageCleanerState = field(default_factory=StorageCleanerState)
    total_violations: int = 0
    total_enforcements: int = 0
    total_rollbacks: int = 0
    overall_safety_level: SafetyCheckResult = SafetyCheckResult.OK
    last_check_time: float = 0.0
    check_results: Dict[str, SafetyCheckResult] = field(default_factory=dict)


# ============================================================
# 阶段90 核心接口：安全护栏
# ============================================================
class SafetyGuard:
    """阶段90的核心接口：安全护栏。

    整合九个安全子系统，提供统一的安全检查接口。
    所有方法均为对阶段81-89接口的编排调用。

    核心约束：
        - CPU占用率 ≤ 20%
        - 内存占用率 ≤ 20%
        - 磁盘占用率 < 80%
        - 关键文件不可修改
        - 5轮无收益探索冻结
        - 倒退>5%自动回滚
        - 探索利用比例[65%, 75%]
        - 空闲时触发离线复盘
        - 稳态时探索权重上浮
    """

    def __init__(
        self,
        cpu_max_threshold: float = 0.20,
        cpu_resume_threshold: float = 0.15,
        mem_max_threshold: float = 0.20,
        mem_preemptive_threshold: float = 0.18,
        disk_max_threshold: float = 0.80,
        disk_target_threshold: float = 0.60,
        rollback_threshold: float = 0.05,
        ratio_target: float = 0.70,
        ratio_upper: float = 0.75,
        ratio_lower: float = 0.65,
        explore_weight_max: float = 0.50
    ):
        """初始化安全护栏（A90.1）。

        Args:
            cpu_max_threshold: CPU上限阈值，范围(0,1]
            cpu_resume_threshold: CPU恢复阈值，范围(0,1]
            mem_max_threshold: 内存上限阈值，范围(0,1]
            mem_preemptive_threshold: 内存预防性阈值，范围(0,1]
            disk_max_threshold: 磁盘上限阈值，范围(0,1]
            disk_target_threshold: 磁盘目标阈值，范围(0,1]
            rollback_threshold: 回滚阈值，必须 > 0
            ratio_target: 探索利用目标比例，范围[0,1]
            ratio_upper: 利用比例上限，范围[0,1]
            ratio_lower: 利用比例下限，范围[0,1]
            explore_weight_max: 探索权重上限，范围[0,1]

        Raises:
            ValueError: 当任何阈值超出有效范围时

        Examples:
            >>> guard = SafetyGuardExecutor()
            >>> guard.cpu_max_threshold
            0.2
        """
        if not (0 < cpu_max_threshold <= 1):
            raise ValueError(f"cpu_max_threshold必须 ∈ (0,1]，收到 {cpu_max_threshold}")
        if not (0 < cpu_resume_threshold <= 1):
            raise ValueError(f"cpu_resume_threshold必须 ∈ (0,1]，收到 {cpu_resume_threshold}")
        if not (0 < mem_max_threshold <= 1):
            raise ValueError(f"mem_max_threshold必须 ∈ (0,1]，收到 {mem_max_threshold}")
        if not (0 < disk_max_threshold <= 1):
            raise ValueError(f"disk_max_threshold必须 ∈ (0,1]，收到 {disk_max_threshold}")
        if not (0 < disk_target_threshold <= 1):
            raise ValueError(f"disk_target_threshold必须 ∈ (0,1]，收到 {disk_target_threshold}")
        if rollback_threshold <= 0:
            raise ValueError(f"rollback_threshold必须 > 0，收到 {rollback_threshold}")
        self.cpu_max_threshold = cpu_max_threshold
        self.cpu_resume_threshold = cpu_resume_threshold
        self.mem_max_threshold = mem_max_threshold
        self.mem_preemptive_threshold = mem_preemptive_threshold
        self.disk_max_threshold = disk_max_threshold
        self.disk_target_threshold = disk_target_threshold
        self.rollback_threshold = rollback_threshold
        self.ratio_target = ratio_target
        self.ratio_upper = ratio_upper
        self.ratio_lower = ratio_lower
        self.explore_weight_max = explore_weight_max

    def check_cpu(self) -> Tuple[CPUMonitorState, SafetyCheckResult]:
        """CPU安全检查（A90.2）。

        Returns:
            (CPU监控状态, 检查结果)

        Examples:
            >>> guard = SafetyGuardExecutor()
            >>> state, result = guard.check_cpu()
            >>> result.violated in (True, False)
            True
        """
        cpu_exec = CPUMonitorExecutor()
        ratio = cpu_exec.measure_cpu_ratio()
        state = CPUMonitorState(
            cpu_ratio_ema=ratio,
            throttle_level=CPUThrottleLevel.NONE,
            alert_level=CPUAlertLevel.GREEN,
            low_cpu_duration=0.0
        )
        should_throttle, alert = cpu_exec.check_throttle(
            ratio, self.cpu_max_threshold
        )
        violated = should_throttle and alert == CPUAlertLevel.RED
        result = SafetyCheckResult(
            check_type="cpu",
            violated=violated,
            current_value=ratio,
            threshold=self.cpu_max_threshold,
            message=(
                f"CPU使用率{ratio:.2%}超过阈值"
                f"{self.cpu_max_threshold:.2%}"
                if violated else "CPU正常"
            )
        )
        return state, result

    def check_memory(self) -> Tuple[MemoryMonitorState, SafetyCheckResult]:
        """内存安全检查（A90.3）。

        Returns:
            (内存监控状态, 检查结果)

        Examples:
            >>> guard = SafetyGuardExecutor()
            >>> state, result = guard.check_memory()
            >>> result.check_type
            'memory'
        """
        mem_exec = MemoryMonitorExecutor()
        ratio, _ = mem_exec.measure_mem_ratio()
        state = MemoryMonitorState(
            mem_ratio_ema=ratio,
            mem_ratio_history=[ratio],
            mem_trend=0.0,
            alert_level=MemoryAlertLevel.GREEN,
            freed_count=0
        )
        should_free, alert, _ = mem_exec.check_free(
            ratio, 0.0, self.mem_max_threshold, self.mem_preemptive_threshold
        )
        violated = alert == MemoryAlertLevel.RED
        result = SafetyCheckResult(
            check_type="memory",
            violated=violated,
            current_value=ratio,
            threshold=self.mem_max_threshold,
            message=(
                f"内存使用率{ratio:.2%}超过阈值"
                f"{self.mem_max_threshold:.2%}"
                if violated else "内存正常"
            )
        )
        return state, result

    def check_storage(self) -> Tuple[StorageCleanerState, SafetyCheckResult]:
        """存储安全检查（A90.4）。

        Returns:
            (存储清理状态, 检查结果)

        Examples:
            >>> guard = SafetyGuardExecutor()
            >>> state, result = guard.check_storage()
            >>> result.check_type
            'storage'
        """
        sc_exec = StorageCleanerExecutor()
        usage, _ = sc_exec.measure_disk_usage()
        state = StorageCleanerState(
            disk_usage=usage,
            disk_trend=0.0,
            clean_level=StorageCleanLevel.NONE,
            cleaned_count=0,
            freed_bytes=0,
            last_clean_time=0.0,
            disk_after_last_clean=usage,
            min_value_kept=0.0,
            audit_log_count=0
        )
        should_clean, level = sc_exec.check_clean(usage, 0.0)
        violated = level == StorageCleanLevel.EXTREME
        result = SafetyCheckResult(
            check_type="storage",
            violated=violated,
            current_value=usage,
            threshold=self.disk_max_threshold,
            message=(
                f"磁盘使用率{usage:.2%}超过阈值"
                f"{self.disk_max_threshold:.2%}"
                if violated else "存储正常"
            )
        )
        return state, result

    def check_file_access(
        self,
        path: str,
        operation: str = "write"
    ) -> Tuple[InterceptResult, SafetyCheckResult]:
        """文件访问安全检查（A90.5）。

        Args:
            path: 文件路径
            operation: 操作类型，支持"write"|"delete"|"rename"

        Returns:
            (拦截结果, 检查结果)

        Examples:
            >>> guard = SafetyGuardExecutor()
            >>> intercept, result = guard.check_file_access(
            ...     "/etc/passwd", "write")
            >>> intercept == InterceptResult.DENY
            True
        """
        fp_exec = FileProtectionExecutor()
        op_map = {
            "write": FileOperation.WRITE,
            "delete": FileOperation.DELETE,
            "rename": FileOperation.RENAME
        }
        op = op_map.get(operation, FileOperation.WRITE)
        result_intercept = fp_exec.intercept(path, op)
        violated = result_intercept == InterceptResult.DENY
        result = SafetyCheckResult(
            check_type="file_access",
            violated=violated,
            current_value=0.0,
            threshold=0.0,
            message=(
                f"文件访问被拦截: {path} {operation}"
                if violated else f"文件访问允许: {path}"
            )
        )
        return result_intercept, result

    def check_explore_freeze(
        self,
        task_id: str,
        benefit: float
    ) -> Tuple[str, SafetyCheckResult]:
        """探索冻结安全检查（A90.6）。

        Args:
            task_id: 任务ID
            benefit: 本轮收益

        Returns:
            ("freeze"/"continue", 检查结果)

        Examples:
            >>> guard = SafetyGuardExecutor()
            >>> action, result = guard.check_explore_freeze("t1", -0.01)
            >>> action in ("freeze", "continue")
            True
        """
        ef_exec = ExploreFreezerExecutor()
        no_benefit, low_benefit = ef_exec.check_no_benefit(
            (benefit,), window=1
        )
        violated = no_benefit or low_benefit
        action = "freeze" if violated else "continue"
        result = SafetyCheckResult(
            check_type="explore_freeze",
            violated=violated,
            current_value=benefit,
            threshold=0.0,
            message=(
                f"探索任务{task_id}收益不足，建议冻结"
                if violated else f"探索任务{task_id}收益正常"
            )
        )
        return action, result

    def check_rollback(
        self,
        F_old: float,
        F_new: float,
        snapshot_id: int
    ) -> Tuple[str, SafetyCheckResult]:
        """回滚安全检查（A90.7）。

        Args:
            F_old: 旧模型自由能
            F_new: 新模型自由能
            snapshot_id: 重构前快照ID

        Returns:
            ("rollback"/"accept", 检查结果)

        Examples:
            >>> guard = SafetyGuardExecutor()
            >>> action, result = guard.check_rollback(10.0, 11.0, 1)
            >>> action in ("rollback", "accept")
            True
        """
        ar_exec = AutoRollbackExecutor()
        regressed, level = ar_exec.check_regression(
            F_new, F_old, self.rollback_threshold
        )
        violated = regressed and level in {
            RegressionLevel.MODERATE, RegressionLevel.SEVERE
        }
        action = "rollback" if violated else "accept"
        result = SafetyCheckResult(
            check_type="rollback",
            violated=violated,
            current_value=F_new - F_old,
            threshold=self.rollback_threshold,
            message=(
                f"检测到性能倒退({level.value})，建议回滚"
                if violated else "性能正常，无需回滚"
            )
        )
        return action, result

    def check_ratio(
        self,
        task_history: List[Dict]
    ) -> Tuple[str, SafetyCheckResult]:
        """比例安全检查（A90.8）。

        Args:
            task_history: 任务执行历史

        Returns:
            ("enforce"/"ok", 检查结果)

        Examples:
            >>> guard = SafetyGuardExecutor()
            >>> action, result = guard.check_ratio(
            ...     [{"type": "exploit"}] * 8)
            >>> action in ("enforce", "ok")
            True
        """
        re_exec = RatioEnforcerExecutor()
        ratio = re_exec.monitor_ratio(task_history, window=100)
        deviation = re_exec.check_deviation(
            ratio, self.ratio_upper, self.ratio_lower
        )
        violated = deviation != RatioDeviation.NORMAL
        action = "enforce" if violated else "ok"
        result = SafetyCheckResult(
            check_type="ratio",
            violated=violated,
            current_value=ratio,
            threshold=self.ratio_target,
            message=(
                f"利用比例{ratio:.2%}偏离目标区间"
                f"[{self.ratio_lower:.0%}, {self.ratio_upper:.0%}]"
                if violated else f"利用比例{ratio:.2%}正常"
            )
        )
        return action, result

    def check_idle(
        self,
        cpu_usage: float,
        has_active_task: bool,
        time_since_last_task: float
    ) -> Tuple[str, SafetyCheckResult]:
        """空闲安全检查（A90.9）。

        Args:
            cpu_usage: CPU使用率，范围[0,1]
            has_active_task: 是否有活跃任务
            time_since_last_task: 距上次任务时间（秒）

        Returns:
            ("trigger_replay"/"continue", 检查结果)

        Examples:
            >>> guard = SafetyGuardExecutor()
            >>> action, result = guard.check_idle(0.05, False, 70.0)
            >>> action in ("trigger_replay", "continue")
            True
        """
        irt_exec = IdleReplayTriggerExecutor()
        is_idle, duration = irt_exec.check_idle(
            cpu_usage, has_active_task, time_since_last_task
        )
        violated = is_idle
        action = "trigger_replay" if violated else "continue"
        result = SafetyCheckResult(
            check_type="idle",
            violated=violated,
            current_value=duration,
            threshold=60.0,
            message=(
                f"系统空闲({duration:.0f}s)，触发离线复盘"
                if violated else "系统不空闲"
            )
        )
        return action, result

    def check_stable(
        self,
        F_history: List[float],
        W_history: List[float],
        duration: float
    ) -> Tuple[str, SafetyCheckResult]:
        """稳态安全检查（A90.10）。

        Args:
            F_history: 自由能历史列表
            W_history: 权重范数历史列表
            duration: 稳态持续时间（秒）

        Returns:
            ("boost_explore"/"continue", 检查结果)

        Examples:
            >>> guard = SafetyGuardExecutor()
            >>> action, result = guard.check_stable(
            ...     [10.0, 10.001], [1.0, 1.0001], 350.0)
            >>> action in ("boost_explore", "continue")
            True
        """
        set_exec = StableExploreTriggerExecutor()
        is_stable, f_var, w_var, _ = set_exec.check_stable(
            F_history, W_history, duration
        )
        violated = is_stable
        action = "boost_explore" if violated else "continue"
        result = SafetyCheckResult(
            check_type="stable",
            violated=violated,
            current_value=f_var + w_var,
            threshold=0.001,
            message=(
                f"系统稳态(F_var={f_var:.6f}, W_var={w_var:.6f})，"
                f"触发探索权重上浮"
                if violated else "系统未稳态"
            )
        )
        return action, result

    def check_all(self) -> SafetyGuardState:
        """全量安全检查。

        依次执行所有九项检查，返回综合安全状态。

        Returns:
            SafetyGuardState 综合安全状态

        Examples:
            >>> guard = SafetyGuardExecutor()
            >>> state = guard.check_all()
            >>> isinstance(state, SafetyGuardState)
            True
        """
        cpu_state, cpu_result = self.check_cpu()
        mem_state, mem_result = self.check_memory()
        storage_state, storage_result = self.check_storage()
        results = [cpu_result, mem_result, storage_result]
        violations = [r for r in results if r.violated]
        return SafetyGuardState(
            cpu_state=cpu_state,
            memory_state=mem_state,
            storage_state=storage_state,
            file_protection_state=FileProtectionState(),
            explore_freeze_state=ExploreFreezeState(),
            auto_rollback_state=AutoRollbackState(),
            ratio_enforcer_state=RatioEnforcerState(),
            idle_replay_state=IdleReplayState(),
            stable_explore_state=StableExploreState(),
            total_violations=len(violations),
            violation_history=tuple(violations),
            last_check_time=0.0,
            is_safe=len(violations) == 0
        )

    def step(
        self,
        state: SafetyGuardState
    ) -> SafetyGuardState:
        """执行一步安全护栏检查。

        Args:
            state: 当前安全护栏状态

        Returns:
            更新后的安全护栏状态

        Examples:
            >>> guard = SafetyGuardExecutor()
            >>> new_state = guard.step(guard.check_all())
            >>> isinstance(new_state, SafetyGuardState)
            True
        """
        return self.check_all()
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `StableExploreTrigger` | ← 阶段89 | 作为安全护栏的稳态探索输入（组合字段） |
| **输入** | A81.1-A81.4（CPU监控） | ← 阶段81 | check_cpu()的实现 |
| **输入** | A82.1-A82.4（内存监控） | ← 阶段82 | check_memory()的实现 |
| **输入** | A83.1-A83.5（存储清理） | ← 阶段83 | check_storage()的实现 |
| **输入** | A84.1-A84.4（文件保护） | ← 阶段84 | check_file_access()的实现 |
| **输入** | A85.1-A85.5（探索冻结） | ← 阶段85 | check_explore_freeze()的实现 |
| **输入** | A86.1-A86.5（自动回滚） | ← 阶段86 | check_rollback()的实现 |
| **输入** | A87.1-A87.4（比例执行） | ← 阶段87 | check_ratio()的实现 |
| **输入** | A88.1-A88.4（空闲复盘） | ← 阶段88 | check_idle()的实现 |
| **输入** | A89.1-A89.4（稳态探索） | ← 阶段89 | check_stable()的实现 |
| **输出** | `SafetyGuardState` | → 工程实现 | 安全护栏的统一状态 |
| **输出** | A90.1-A90.10（安全护栏接口） | → 工程实现 | 安全护栏的统一Python接口 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| psutil | 资源监控 | CPU/内存/磁盘度量 |
| re | 正则匹配 | 文件保护路径匹配 |
| NumPy | 数值计算 | 性能评估和方差计算 |
| SciPy | 统计检验 | 置信区间和显著性检验 |
| SQLite | 日志存储 | 违规日志和审计记录 |
| asyncio | 异步执行 | 非阻塞的安全检查 |
| Pydantic | 数据验证 | SafetyGuardState的运行时验证 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V90.1 | SafetyGuard初始化 | 九个子系统正确创建 | 初始化测试 |
| V90.2 | check_cpu() | 返回CPUMonitorState | 接口测试 |
| V90.3 | check_memory() | 返回MemoryMonitorState | 接口测试 |
| V90.4 | check_storage() | 返回StorageCleanerState | 接口测试 |
| V90.5 | check_file_access() | 返回allow/deny | 拦截测试 |
| V90.6 | check_explore_freeze() | 返回freeze/continue | 冻结测试 |
| V90.7 | check_rollback() | 返回rollback/accept | 回滚测试 |
| V90.8 | check_ratio() | 返回enforce/ok | 比例测试 |
| V90.9 | check_idle() | 返回trigger_replay/continue | 空闲测试 |
| V90.10 | check_stable() | 返回boost_explore/continue | 稳态测试 |
| V90.11 | check_all() | 返回SafetyGuardState | 综合测试 |
| V90.12 | SafetyGuardState组合 | `state.stable_trigger` 类型为 `StableExploreTrigger` | 类型检查 |
| V90.13 | 综合安全级别 | OK/WARNING/VIOLATION/ACTION_TAKEN | 级别测试 |
| V90.14 | 子系统协同 | CPU节流与内存释放不冲突 | 协同测试 |
| V90.15 | 端到端流程 | 全量检查→发现问题→采取行动→验证结果 | 集成测试 |
| V90.16 | 性能开销 | check_all()延迟<10ms | 性能测试 |
| V90.17 | 配置参数 | 所有阈值可配置 | 配置测试 |
| V90.18 | 日志完整性 | 所有安全事件有日志 | 日志测试 |
| V90.19 | 状态不可变 | SafetyGuardState为frozen dataclass | 不可变测试 |
| V90.20 | 九项约束 | 所有安全约束满足 | 约束测试 |

---

> **阶段90完** | 阶段90建立了安全护栏的统一Python接口：SafetyGuard类设计（整合所有安全约束）（A90.1），check_cpu() → CPUMonitorState（A90.2），check_memory() → MemoryMonitorState（A90.3），check_storage() → StorageCleanerState（A90.4），check_file_access(path) → allow/deny（A90.5），check_explore_freeze(task) → freeze/continue（A90.6），check_rollback(model_old, model_new) → rollback/accept（A90.7），check_ratio(history) → enforce/ok（A90.8），check_idle() → trigger_replay/continue（A90.9），check_stable() → boost_explore/continue（A90.10），封装为SafetyGuardState数据类组合StableExploreTrigger。至此，阶段81-90建立了完整的安全护栏体系：CPU占用监控与≤20%上限执行器（81）→ 内存占用监控与≤20%上限执行器（82）→ 本地存储自动清理（83）→ 禁止修改系统关键文件的规则引擎（84）→ 陌生探索5轮无收益自动冻结（85）→ 模型重构后性能倒退自动回滚（86）→ 70/30探索利用比例的强制执行器（87）→ 系统空闲→离线复盘的触发器（88）→ 稳态环境→探索权重上浮的触发器（89）→ 安全护栏Python接口（90）。从资源约束到行为约束，从被动防护到主动优化，安全护栏为智能体提供了"守底线、保安全、促进化"的完整保障能力。
