"""阶段81：CPU占用监控与≤20%上限执行器。

本阶段实现CPU监控与节流机制。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import numpy as np

from .stage_080 import MemoryValidationResult


class CPUThrottleLevel(Enum):
    """CPU节流级别。"""
    NONE = 0
    REDUCE_FREQUENCY = 1
    SKIP_LOW_PRIORITY = 2
    PAUSE_EXPLORATION = 3


class CPUAlertLevel(Enum):
    """CPU预警级别。"""
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


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
                f"上限阈值({self.cpu_max_threshold})必须大于"
                f"恢复阈值({self.cpu_resume_threshold})"
            )


class CPUMonitorExecutor:
    """阶段81的核心接口：CPU监控与节流执行器。

    实现CPU监控、上限判定、节流策略和恢复条件。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - CPU占用率 ≤ 20%
        - P0任务永不跳过
        - 恢复需持续30秒低于15%
    """

    def measure_cpu_ratio(self) -> float:
        """度量当前CPU占用率（A81.1）。"""
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
        """更新CPU占用率滑动平均（A81.1-ema）。"""
        if not (0 < alpha < 1):
            raise ValueError(f"alpha必须 ∈ (0,1)，收到 {alpha}")
        return alpha * current_ratio + (1 - alpha) * previous_ema

    def check_alert_level(
        self,
        cpu_ratio_ema: float,
        max_threshold: float = 0.20,
        resume_threshold: float = 0.15
    ) -> CPUAlertLevel:
        """判定预警级别（A81.2）。"""
        if cpu_ratio_ema >= max_threshold:
            return CPUAlertLevel.RED
        elif cpu_ratio_ema >= resume_threshold:
            return CPUAlertLevel.YELLOW
        else:
            return CPUAlertLevel.GREEN

    def check_throttle(
        self,
        cpu_ratio_ema: float,
        threshold: float = 0.20
    ) -> bool:
        """判定是否应节流（A81.2）。"""
        return cpu_ratio_ema >= threshold

    def throttle(
        self,
        current_level: CPUThrottleLevel
    ) -> CPUThrottleLevel:
        """执行节流策略（A81.3）。

        按优先级递进：降频 → 跳过低优先级 → 暂停探索
        """
        if current_level == CPUThrottleLevel.NONE:
            return CPUThrottleLevel.REDUCE_FREQUENCY
        elif current_level == CPUThrottleLevel.REDUCE_FREQUENCY:
            return CPUThrottleLevel.SKIP_LOW_PRIORITY
        elif current_level == CPUThrottleLevel.SKIP_LOW_PRIORITY:
            return CPUThrottleLevel.PAUSE_EXPLORATION
        else:
            return CPUThrottleLevel.PAUSE_EXPLORATION

    def reduce_frequency(
        self,
        cpu_ratio_ema: float,
        max_threshold: float = 0.20
    ) -> Dict[str, float]:
        """第一级节流：降低执行频率（A81.3-freq）。"""
        scale = max(0.1, max_threshold / max(cpu_ratio_ema, 0.001))
        return {
            "perception": min(1.0, scale),
            "inference": min(1.0, scale),
            "value_eval": max(0.1, scale),
            "consolidation": max(0.01, scale),
            "snapshot": max(0.001, scale),
        }

    def skip_low_priority(
        self,
        min_priority_to_keep: int = 1
    ) -> List[str]:
        """第二级节流：跳过低优先级任务（A81.3-skip）。"""
        if min_priority_to_keep < 0:
            raise ValueError(
                f"min_priority_to_keep必须 >= 0，"
                f"收到 {min_priority_to_keep}"
            )
        return ["explore_task_p2", "explore_task_p3"]

    def pause_exploration(self) -> List[str]:
        """第三级节流：暂停探索（A81.3-pause）。"""
        return ["exploration_all"]

    def check_resume(
        self,
        cpu_ratio_ema: float,
        low_cpu_duration: float,
        resume_threshold: float = 0.15,
        stable_duration: float = 30.0
    ) -> Tuple[bool, int]:
        """判定是否应恢复（A81.4）。"""
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

    def compute_sample_interval(
        self,
        alert_level: CPUAlertLevel
    ) -> float:
        """计算采样间隔（A81.1-interval）。"""
        if alert_level == CPUAlertLevel.GREEN:
            return 10.0
        elif alert_level == CPUAlertLevel.YELLOW:
            return 2.0
        else:
            return 1.0

    def step(
        self,
        monitor: CPUMonitor,
        current_ratio: float
    ) -> CPUMonitor:
        """执行一步CPU监控与节流。

        Args:
            monitor: 当前CPU监控状态
            current_ratio: 当前CPU占用率（外部传入，避免副作用）

        Returns:
            更新后的CPU监控状态
        """
        new_ema = self.update_ema(
            current_ratio,
            monitor.monitor_state.cpu_ratio_ema,
            monitor.ema_alpha
        )
        alert = self.check_alert_level(
            new_ema, monitor.cpu_max_threshold, monitor.cpu_resume_threshold
        )
        should_throttle = self.check_throttle(
            new_ema, monitor.cpu_max_threshold
        )
        new_level = monitor.monitor_state.throttle_level
        new_throttle_duration = monitor.monitor_state.throttle_duration
        new_low_cpu_duration = monitor.monitor_state.low_cpu_duration
        new_is_throttled = monitor.monitor_state.is_throttled
        new_is_resuming = monitor.monitor_state.is_resuming
        new_resume_stage = monitor.monitor_state.resume_stage

        if should_throttle:
            new_level = self.throttle(new_level)
            new_throttle_duration += 1.0
            new_low_cpu_duration = 0.0
            new_is_throttled = True
            new_is_resuming = False
            new_resume_stage = 0
        else:
            new_throttle_duration = 0.0
            if current_ratio < monitor.cpu_resume_threshold:
                new_low_cpu_duration += 1.0
            else:
                new_low_cpu_duration = 0.0
            resume, stage = self.check_resume(
                new_ema,
                new_low_cpu_duration,
                monitor.cpu_resume_threshold,
                monitor.stable_duration
            )
            if resume:
                new_level = CPUThrottleLevel.NONE
                new_is_throttled = False
                new_is_resuming = True
                new_resume_stage = stage
            elif new_level != CPUThrottleLevel.NONE:
                new_is_throttled = True

        new_interval = self.compute_sample_interval(alert)
        new_state = CPUMonitorState(
            cpu_ratio=current_ratio,
            cpu_ratio_ema=new_ema,
            alert_level=alert,
            throttle_level=new_level,
            throttle_duration=new_throttle_duration,
            low_cpu_duration=new_low_cpu_duration,
            sample_interval=new_interval,
            is_throttled=new_is_throttled,
            is_resuming=new_is_resuming,
            resume_stage=new_resume_stage,
        )
        return CPUMonitor(
            validation_result=monitor.validation_result,
            monitor_state=new_state,
            cpu_max_threshold=monitor.cpu_max_threshold,
            cpu_resume_threshold=monitor.cpu_resume_threshold,
            ema_alpha=monitor.ema_alpha,
            stable_duration=monitor.stable_duration,
        )
