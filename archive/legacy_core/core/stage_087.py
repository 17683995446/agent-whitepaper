"""阶段87：70/30探索利用比例的强制执行器。

本阶段实现探索利用比例的监控与强制执行机制。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import time

from .stage_086 import AutoRollback


class RatioDeviation(Enum):
    """比例偏离方向。"""
    NONE = "none"
    TOO_MUCH_EXPLOIT = "too_much_exploit"  # 利用过多
    TOO_MUCH_EXPLORE = "too_much_explore"  # 探索过多


class EnforcementAction(Enum):
    """强制执行动作。"""
    INJECT_EXPLORE = "inject_explore"
    INJECT_EXPLOIT = "inject_exploit"
    NONE = "none"


class RatioAlertLevel(Enum):
    """比例告警级别。"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


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


class RatioEnforcerExecutor:
    """阶段87的核心接口：比例强制执行器。

    实现全局比例监控、强制探索注入、强制利用注入和执行日志。
    所有方法均为纯函数（无副作用），保证可验证性。

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
        """
        if upper_bound <= lower_bound:
            raise ValueError("upper_bound必须 > lower_bound")
        if ratio > upper_bound:
            return RatioDeviation.TOO_MUCH_EXPLOIT
        elif ratio < lower_bound:
            return RatioDeviation.TOO_MUCH_EXPLORE
        return RatioDeviation.NONE

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
        """
        # 资源约束优先
        if cpu_ratio > 0.20 or mem_ratio > 0.20:
            return []
        deviation = ratio - 0.70
        n_inject = max(1, int(deviation / 0.05) + 1)
        if frozen_tasks:
            return frozen_tasks[:n_inject]
        return [f"new_explore_task_{i}" for i in range(n_inject)]

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
        """
        deviation = 0.70 - ratio
        n_inject = max(1, int(deviation / 0.05) + 1)
        exploit_tasks = [f"exploit_task_{i}" for i in range(n_inject)]
        freeze_tasks = low_benefit_explore_tasks[:n_inject]
        return exploit_tasks, freeze_tasks

    def compute_alert_level(
        self,
        ratio: float
    ) -> RatioAlertLevel:
        """计算告警级别（A87.4）。

        Args:
            ratio: 当前利用比例

        Returns:
            告警级别
        """
        if ratio >= 0.90 or ratio <= 0.50:
            return RatioAlertLevel.CRITICAL
        elif ratio >= 0.80 or ratio <= 0.60:
            return RatioAlertLevel.WARNING
        return RatioAlertLevel.INFO

    def step(
        self,
        enforcer: RatioEnforcer,
        task_history: List[Dict],
        frozen_tasks: List[str] = None,
        low_benefit_tasks: List[str] = None,
        cpu_ratio: float = 0.1,
        mem_ratio: float = 0.1
    ) -> RatioEnforcer:
        """执行一步比例监控与强制执行。

        Args:
            enforcer: 当前比例执行器状态
            task_history: 任务执行历史
            frozen_tasks: 可解冻的冻结任务列表
            low_benefit_tasks: 低收益探索任务列表
            cpu_ratio: CPU占用率
            mem_ratio: 内存占用率

        Returns:
            更新后的比例执行器状态
        """
        frozen_tasks = frozen_tasks or []
        low_benefit_tasks = low_benefit_tasks or []

        # 1. 监控比例
        ratio = self.monitor_ratio(task_history, enforcer.window_size)

        # 2. 检查偏离
        deviation = self.check_deviation(
            ratio,
            enforcer.enforcer_state.exploit_upper_bound,
            enforcer.enforcer_state.exploit_lower_bound
        )

        # 3. 计算告警级别
        alert = self.compute_alert_level(ratio)

        # 4. 强制执行
        action = EnforcementAction.NONE
        inject_count = 0
        freeze_count = 0
        total_enforcements = enforcer.enforcer_state.total_enforcements
        history = list(enforcer.enforcer_state.enforcement_history)

        if deviation == RatioDeviation.TOO_MUCH_EXPLOIT:
            injected = self.inject_explore_task(
                ratio, frozen_tasks, cpu_ratio, mem_ratio
            )
            action = EnforcementAction.INJECT_EXPLORE
            inject_count = len(injected)
            total_enforcements += 1
        elif deviation == RatioDeviation.TOO_MUCH_EXPLORE:
            exploit_tasks, freeze_tasks = self.inject_exploit_task(
                ratio, low_benefit_tasks
            )
            action = EnforcementAction.INJECT_EXPLOIT
            inject_count = len(exploit_tasks)
            freeze_count = len(freeze_tasks)
            total_enforcements += 1

        # 5. 记录执行日志
        if action != EnforcementAction.NONE:
            record = RatioEnforcementRecord(
                timestamp=time.time(),
                ratio_before=ratio,
                ratio_target=enforcer.enforcer_state.target_ratio,
                action=action,
                inject_count=inject_count,
                freeze_count=freeze_count,
                ratio_after=ratio  # 简化：假设比例不变
            )
            history.append(record)
            # 限制历史长度
            if len(history) > 100:
                history = history[-100:]

        new_state = RatioEnforcerState(
            current_ratio=ratio,
            target_ratio=enforcer.enforcer_state.target_ratio,
            exploit_upper_bound=enforcer.enforcer_state.exploit_upper_bound,
            exploit_lower_bound=enforcer.enforcer_state.exploit_lower_bound,
            total_enforcements=total_enforcements,
            recent_enforcement_rate=float(total_enforcements),  # 简化
            alert_level=alert,
            enforcement_history=tuple(history)
        )
        return RatioEnforcer(
            auto_rollback=enforcer.auto_rollback,
            enforcer_state=new_state,
            window_size=enforcer.window_size,
            max_enforcement_rate=enforcer.max_enforcement_rate
        )
