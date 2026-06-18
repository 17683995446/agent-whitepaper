"""阶段88：系统空闲→离线复盘触发器。

本阶段实现系统空闲时自动触发离线复盘的机制。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import time

from .stage_087 import RatioEnforcer


class ReplayPriority(Enum):
    """离线复盘优先级。"""
    NONE = 0             # 不复盘
    HIGH_ERROR = 1       # 高误差样本
    CONSOLIDATION = 2    # 巩固期复盘
    GENERATIVE = 3       # 生成式回放
    RANDOM = 4           # 随机回放


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

    def __post_init__(self):
        if self.idle_duration < 0:
            raise ValueError(
                f"idle_duration必须 >= 0，收到 {self.idle_duration}"
            )
        if self.replay_cpu_budget > 0.10:
            raise ValueError(
                f"replay_cpu_budget必须 <= 0.10，"
                f"收到 {self.replay_cpu_budget}"
            )


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


class IdleReplayTriggerExecutor:
    """阶段88的核心接口：空闲复盘触发器。

    实现空闲判定、复盘触发、资源分配和终止条件。
    所有方法均为纯函数（无副作用），保证可验证性。

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
        """
        if time_since_last_task < 0:
            raise ValueError(
                f"time_since_last_task必须 >= 0，"
                f"收到 {time_since_last_task}"
            )
        if is_reforming or is_rolling_back or is_cleaning or is_alerting:
            return False, 0.0
        if (cpu_usage < 0.10 and not has_active_task
                and time_since_last_task >= 60.0):
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
        if time.time() - last_replay_time < min_interval:
            return False, ReplayPriority.NONE
        return True, ReplayPriority.HIGH_ERROR

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
            复盘CPU预算 ∈ [0, 0.10]

        Raises:
            ValueError: 当cpu_max <= 0时
            ValueError: 当safety_margin < 0时
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
        trigger: IdleReplayTrigger,
        cpu_usage: float = 0.05,
        has_active_task: bool = False,
        time_since_last_task: float = 70.0,
        has_new_task: bool = False,
        is_reforming: bool = False,
        is_rolling_back: bool = False,
        is_cleaning: bool = False,
        is_alerting: bool = False
    ) -> IdleReplayTrigger:
        """执行一步空闲复盘检查。

        1. 检查空闲状态
        2. 检查是否触发复盘
        3. 分配资源
        4. 检查终止条件
        5. 更新状态

        Args:
            trigger: 当前空闲复盘触发器状态
            cpu_usage: CPU使用率
            has_active_task: 是否有活跃任务
            time_since_last_task: 距上次任务时间
            has_new_task: 是否有新任务
            is_reforming: 是否正在重构
            is_rolling_back: 是否正在回滚
            is_cleaning: 是否正在清理
            is_alerting: 是否有安全告警

        Returns:
            更新后的空闲复盘触发器状态
        """
        # 1. 检查空闲
        is_idle, idle_duration = self.check_idle(
            cpu_usage, has_active_task, time_since_last_task,
            is_reforming, is_rolling_back, is_cleaning, is_alerting
        )

        # 2. 检查终止条件（如果正在复盘）
        is_replaying = trigger.trigger_state.is_replaying
        stop_reason = ""
        if is_replaying:
            should_stop, stop_reason = self.check_stop(
                has_new_task, cpu_usage,
                trigger.trigger_state.replay_benefit,
                trigger.replay_stop_cpu,
                trigger.low_benefit_threshold
            )
            if should_stop:
                is_replaying = False

        # 3. 检查是否触发复盘（如果刚因新任务停止，则不触发）
        should_trigger = False
        priority = ReplayPriority.NONE
        if is_idle and not is_replaying and stop_reason != "new_task":
            should_trigger, priority = self.trigger_replay(
                is_idle,
                trigger.trigger_state.last_replay_time,
                trigger.min_replay_interval,
                trigger.trigger_state.consecutive_low_benefit,
                trigger.max_low_benefit_count
            )
            if should_trigger:
                is_replaying = True

        # 4. 分配资源
        cpu_budget = self.allocate_resources(
            cpu_usage if not is_idle else 0.0,
            0.20, 0.05
        )

        # 5. 更新连续低收益计数
        consecutive_low = trigger.trigger_state.consecutive_low_benefit
        if stop_reason == "low_benefit":
            consecutive_low += 1
        elif is_replaying and trigger.trigger_state.replay_benefit >= trigger.low_benefit_threshold:
            consecutive_low = 0

        # 6. 更新复盘批次数
        batches = trigger.trigger_state.replay_batches_completed
        if is_replaying:
            batches += 1

        new_state = IdleReplayTriggerState(
            is_idle=is_idle,
            idle_duration=idle_duration,
            is_replaying=is_replaying,
            replay_priority=priority if should_trigger else trigger.trigger_state.replay_priority,
            replay_start_time=time.time() if should_trigger else trigger.trigger_state.replay_start_time,
            replay_cpu_budget=cpu_budget,
            replay_batches_completed=batches,
            replay_benefit=trigger.trigger_state.replay_benefit,
            last_replay_time=time.time() if not is_replaying and trigger.trigger_state.is_replaying else trigger.trigger_state.last_replay_time,
            consecutive_low_benefit=consecutive_low,
            saved_progress=trigger.trigger_state.saved_progress
        )
        return IdleReplayTrigger(
            ratio_enforcer=trigger.ratio_enforcer,
            trigger_state=new_state,
            idle_cpu_threshold=trigger.idle_cpu_threshold,
            idle_time_threshold=trigger.idle_time_threshold,
            replay_cpu_budget=trigger.replay_cpu_budget,
            replay_stop_cpu=trigger.replay_stop_cpu,
            min_replay_interval=trigger.min_replay_interval,
            low_benefit_threshold=trigger.low_benefit_threshold,
            max_low_benefit_count=trigger.max_low_benefit_count
        )
