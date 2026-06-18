"""阶段90：安全护栏Python接口。

本阶段整合阶段81-89的所有安全子系统，提供统一的安全检查接口。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import time

from .stage_089 import StableExploreTrigger
from .stage_081 import CPUMonitorState, CPUAlertLevel, CPUThrottleLevel
from .stage_082 import MemoryMonitorState, MemoryAlertLevel, MemoryFreeLevel
from .stage_083 import StorageCleanerState, StorageCleanLevel
from .stage_084 import InterceptResult, FileOperation
from .stage_085 import ExploreBenefitRecord, FreezeReason
from .stage_086 import RegressionLevel, RootCauseType
from .stage_087 import RatioDeviation, EnforcementAction, RatioAlertLevel
from .stage_088 import ReplayPriority

from .stage_081 import CPUMonitorExecutor
from .stage_082 import MemoryMonitorExecutor
from .stage_083 import StorageCleanerExecutor
from .stage_084 import FileProtectionExecutor
from .stage_085 import ExploreFreezerExecutor
from .stage_086 import AutoRollbackExecutor
from .stage_087 import RatioEnforcerExecutor
from .stage_088 import IdleReplayTriggerExecutor
from .stage_089 import StableExploreTriggerExecutor


class SafetyCheckResult(Enum):
    """安全检查结果。"""
    OK = "ok"
    WARNING = "warning"
    VIOLATION = "violation"
    ACTION_TAKEN = "action_taken"


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

    def __post_init__(self):
        if self.total_violations < 0:
            raise ValueError(
                f"total_violations必须 >= 0，收到 {self.total_violations}"
            )
        if self.total_enforcements < 0:
            raise ValueError(
                f"total_enforcements必须 >= 0，收到 {self.total_enforcements}"
            )
        if self.total_rollbacks < 0:
            raise ValueError(
                f"total_rollbacks必须 >= 0，收到 {self.total_rollbacks}"
            )


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
            cpu_max_threshold: CPU上限阈值
            cpu_resume_threshold: CPU恢复阈值
            mem_max_threshold: 内存上限阈值
            mem_preemptive_threshold: 内存预防性阈值
            disk_max_threshold: 磁盘上限阈值
            disk_target_threshold: 磁盘目标阈值
            rollback_threshold: 回滚阈值
            ratio_target: 探索利用目标比例
            ratio_upper: 利用比例上限
            ratio_lower: 利用比例下限
            explore_weight_max: 探索权重上限
        """
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

    def check_cpu(
        self,
        current_ratio: float = 0.0
    ) -> Tuple[CPUMonitorState, SafetyCheckResult]:
        """CPU安全检查（A90.2）。

        Args:
            current_ratio: 当前CPU占用率（外部传入）

        Returns:
            (CPU监控状态, 检查结果)
        """
        cpu_exec = CPUMonitorExecutor()
        if current_ratio == 0.0:
            current_ratio = cpu_exec.measure_cpu_ratio()
        ema = cpu_exec.update_ema(current_ratio, 0.0, 0.3)
        alert = cpu_exec.check_alert_level(
            ema, self.cpu_max_threshold, self.cpu_resume_threshold
        )
        should_throttle = cpu_exec.check_throttle(ema, self.cpu_max_threshold)
        throttle_level = CPUThrottleLevel.NONE
        if should_throttle:
            throttle_level = cpu_exec.throttle(throttle_level)
        state = CPUMonitorState(
            cpu_ratio=current_ratio,
            cpu_ratio_ema=ema,
            alert_level=alert,
            throttle_level=throttle_level,
            is_throttled=should_throttle
        )
        violated = alert == CPUAlertLevel.RED
        result = (
            SafetyCheckResult.VIOLATION if violated
            else SafetyCheckResult.OK
        )
        return state, result

    def check_memory(
        self,
        current_ratio: float = 0.0
    ) -> Tuple[MemoryMonitorState, SafetyCheckResult]:
        """内存安全检查（A90.3）。

        Args:
            current_ratio: 当前内存占用率（外部传入）

        Returns:
            (内存监控状态, 检查结果)
        """
        mem_exec = MemoryMonitorExecutor()
        if current_ratio == 0.0:
            current_ratio, _ = mem_exec.measure_mem_ratio()
        ema = mem_exec.update_ema(current_ratio, 0.0, 0.2)
        trend = mem_exec.compute_mem_trend((current_ratio,), 300.0)
        should_free, alert, is_preemptive = mem_exec.check_free(
            ema, trend, self.mem_max_threshold, self.mem_preemptive_threshold
        )
        free_level = MemoryFreeLevel.NONE
        if should_free:
            free_level = (
                MemoryFreeLevel.EVICT_LOW_VALUE if is_preemptive
                else MemoryFreeLevel.FLUSH_CACHE
            )
        state = MemoryMonitorState(
            mem_ratio=current_ratio,
            mem_ratio_ema=ema,
            mem_trend=trend,
            alert_level=alert,
            free_level=free_level,
            is_preemptive=is_preemptive
        )
        violated = alert == MemoryAlertLevel.RED
        result = (
            SafetyCheckResult.VIOLATION if violated
            else SafetyCheckResult.OK
        )
        return state, result

    def check_storage(
        self,
        current_usage: float = 0.0
    ) -> Tuple[StorageCleanerState, SafetyCheckResult]:
        """存储安全检查（A90.4）。

        Args:
            current_usage: 当前磁盘占用率（外部传入）

        Returns:
            (存储清理状态, 检查结果)
        """
        sc_exec = StorageCleanerExecutor()
        if current_usage == 0.0:
            current_usage, _ = sc_exec.measure_disk_usage()
        trend = current_usage  # 简化
        should_clean, level = sc_exec.check_clean(
            current_usage, trend, self.disk_max_threshold
        )
        state = StorageCleanerState(
            disk_usage=current_usage,
            disk_trend=trend,
            clean_level=level
        )
        violated = level == StorageCleanLevel.HIGH_VALUE
        result = (
            SafetyCheckResult.VIOLATION if violated
            else SafetyCheckResult.OK
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
            operation: 操作类型

        Returns:
            (拦截结果, 检查结果)
        """
        fp_exec = FileProtectionExecutor()
        patterns = fp_exec.build_protected_patterns("linux")
        op_map = {
            "write": FileOperation.WRITE,
            "delete": FileOperation.DELETE,
            "rename": FileOperation.RENAME,
            "read": FileOperation.READ
        }
        op = op_map.get(operation, FileOperation.WRITE)
        result_intercept, matched = fp_exec.intercept(path, op, patterns)
        violated = result_intercept == InterceptResult.DENY
        result = (
            SafetyCheckResult.VIOLATION if violated
            else SafetyCheckResult.OK
        )
        return result_intercept, result

    def check_explore_freeze(
        self,
        benefit_history: Tuple[float, ...]
    ) -> Tuple[str, SafetyCheckResult]:
        """探索冻结安全检查（A90.6）。

        Args:
            benefit_history: 收益历史

        Returns:
            ("freeze"/"continue", 检查结果)
        """
        ef_exec = ExploreFreezerExecutor()
        no_benefit, low_benefit = ef_exec.check_no_benefit(
            benefit_history, window=5
        )
        violated = no_benefit
        action = "freeze" if violated else "continue"
        result = (
            SafetyCheckResult.ACTION_TAKEN if violated
            else SafetyCheckResult.OK
        )
        return action, result

    def check_rollback(
        self,
        F_old: float,
        F_new: float
    ) -> Tuple[str, SafetyCheckResult]:
        """回滚安全检查（A90.7）。

        Args:
            F_old: 旧模型自由能
            F_new: 新模型自由能

        Returns:
            ("rollback"/"accept", 检查结果)
        """
        ar_exec = AutoRollbackExecutor()
        regressed, level = ar_exec.check_regression(
            F_new, F_old, self.rollback_threshold
        )
        violated = regressed and level in {
            RegressionLevel.MODERATE, RegressionLevel.SEVERE
        }
        action = "rollback" if violated else "accept"
        result = (
            SafetyCheckResult.ACTION_TAKEN if violated
            else SafetyCheckResult.OK
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
        """
        re_exec = RatioEnforcerExecutor()
        ratio = re_exec.monitor_ratio(task_history, window=100)
        deviation = re_exec.check_deviation(
            ratio, self.ratio_upper, self.ratio_lower
        )
        violated = deviation != RatioDeviation.NONE
        action = "enforce" if violated else "ok"
        result = (
            SafetyCheckResult.ACTION_TAKEN if violated
            else SafetyCheckResult.OK
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
            cpu_usage: CPU使用率
            has_active_task: 是否有活跃任务
            time_since_last_task: 距上次任务时间

        Returns:
            ("trigger_replay"/"continue", 检查结果)
        """
        irt_exec = IdleReplayTriggerExecutor()
        is_idle, _ = irt_exec.check_idle(
            cpu_usage, has_active_task, time_since_last_task
        )
        action = "trigger_replay" if is_idle else "continue"
        result = (
            SafetyCheckResult.ACTION_TAKEN if is_idle
            else SafetyCheckResult.OK
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
            F_history: 自由能历史
            W_history: 权重范数历史
            duration: 稳态持续时间

        Returns:
            ("boost_explore"/"continue", 检查结果)
        """
        set_exec = StableExploreTriggerExecutor()
        is_stable, _, _, _ = set_exec.check_stable(
            F_history, W_history, duration
        )
        action = "boost_explore" if is_stable else "continue"
        result = (
            SafetyCheckResult.ACTION_TAKEN if is_stable
            else SafetyCheckResult.OK
        )
        return action, result

    def check_all(
        self,
        state: SafetyGuardState
    ) -> SafetyGuardState:
        """全量安全检查。

        依次执行所有九项检查，返回综合安全状态。

        Args:
            state: 当前安全护栏状态

        Returns:
            更新后的综合安全状态
        """
        cpu_state, cpu_result = self.check_cpu()
        mem_state, mem_result = self.check_memory()
        storage_state, storage_result = self.check_storage()

        results = {
            "cpu": cpu_result,
            "memory": mem_result,
            "storage": storage_result,
        }

        violations = sum(
            1 for r in results.values()
            if r in {SafetyCheckResult.VIOLATION, SafetyCheckResult.ACTION_TAKEN}
        )

        overall = SafetyCheckResult.OK
        if violations > 0:
            overall = SafetyCheckResult.WARNING

        return SafetyGuardState(
            stable_trigger=state.stable_trigger,
            cpu_state=cpu_state,
            memory_state=mem_state,
            storage_state=storage_state,
            total_violations=violations,
            total_enforcements=state.total_enforcements,
            total_rollbacks=state.total_rollbacks,
            overall_safety_level=overall,
            last_check_time=time.time(),
            check_results=results
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
        """
        return self.check_all(state)
