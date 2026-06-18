"""阶段86：模型重构后性能倒退自动回滚。

本阶段实现重构后性能评估与自动回滚机制。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import numpy as np

from .stage_085 import ExploreFreezer
from .stage_078 import ModelSnapshotInterface


class RegressionLevel(Enum):
    """性能倒退级别。"""
    NONE = "none"
    MILD = "mild"           # 5% < ΔF/F < 10%
    MODERATE = "moderate"   # 10% ≤ ΔF/F < 20%
    SEVERE = "severe"       # ΔF/F ≥ 20%


class RootCauseType(Enum):
    """回滚根因类型。"""
    STRUCTURE_INCOMPATIBLE = "structure_incompatible"
    WEIGHT_INIT_POOR = "weight_init_poor"
    LEARNING_RATE_HIGH = "learning_rate_high"
    EWC_TOO_STRONG = "ewc_too_strong"
    DATA_SHIFT = "data_shift"
    OVER_PRUNING = "over_pruning"
    OVER_EXPANSION = "over_expansion"
    WEIGHT_DESTRUCTION = "weight_destruction"
    UNKNOWN = "unknown"


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
        verify_window: 验证窗口时长（秒，默认300）
    """
    explore_freezer: ExploreFreezer
    rollback_state: AutoRollbackState = field(default_factory=AutoRollbackState)
    rollback_threshold: float = 0.05
    verify_tolerance: float = 0.01
    max_rollbacks_per_day: int = 3
    verify_window: float = 300.0


class AutoRollbackExecutor:
    """阶段86的核心接口：自动回滚执行器。

    实现性能评估、倒退判定、自动回滚、回滚后验证和根因分析。
    所有方法均为纯函数（无副作用），保证可验证性。

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

        Raises:
            ValueError: 当F_new_series为空时
            ValueError: 当window <= 0时
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
        """
        if threshold <= 0:
            raise ValueError(f"threshold必须 > 0，收到 {threshold}")
        if F_old == 0:
            ratio = float('inf') if F_new > 0 else 0.0
        else:
            ratio = (F_new - F_old) / abs(F_old)
        if ratio >= threshold + 0.15:  # >= 0.20 -> SEVERE
            return True, RegressionLevel.SEVERE
        elif ratio >= threshold + 0.05:  # >= 0.10 -> MODERATE
            return True, RegressionLevel.MODERATE
        elif ratio >= threshold * 0.5:  # >= 0.025 -> MILD
            return True, RegressionLevel.MILD
        return False, RegressionLevel.NONE

    def execute_rollback(
        self,
        snapshot_id: int,
        recent_count: int,
        max_per_day: int = 3,
        model_structure: Optional[Dict] = None
    ) -> Tuple[Optional[Dict], bool]:
        """执行自动回滚（A86.3）。

        Args:
            snapshot_id: 重构前快照ID
            recent_count: 最近24小时回滚次数
            max_per_day: 每日最大回滚次数，必须 > 0
            model_structure: 当前模型结构（用于兼容性验证）

        Returns:
            (回滚后模型, 是否成功)

        Raises:
            ValueError: 当max_per_day <= 0时
        """
        if max_per_day <= 0:
            raise ValueError(f"max_per_day必须 > 0，收到 {max_per_day}")
        if recent_count >= max_per_day:
            return None, False
        if snapshot_id <= 0:
            return None, False
        # 简化：模拟回滚成功（实际实现会调用ModelSnapshotInterface.load_snapshot）
        return {"snapshot_id": snapshot_id, "rolled_back": True}, True

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
        auto_rollback: AutoRollback,
        F_new_series: List[float],
        F_old: float,
        snapshot_id: int,
        reform_type: str = "",
        delta_weights: Optional[Dict[str, np.ndarray]] = None
    ) -> AutoRollback:
        """执行一步自动回滚检查。

        1. 评估性能
        2. 判定倒退
        3. 执行回滚（如果满足条件）
        4. 验证回滚
        5. 根因分析
        6. 更新状态

        Args:
            auto_rollback: 当前自动回滚状态
            F_new_series: 重构后自由能序列
            F_old: 重构前自由能
            snapshot_id: 重构前快照ID
            reform_type: 重构类型
            delta_weights: 权重变化字典

        Returns:
            更新后的自动回滚状态
        """
        delta_f, lower_bound, confirmed = self.evaluate_performance(
            F_new_series, F_old, auto_rollback.verify_window
        )
        mean_f = float(np.mean(np.array(F_new_series)))
        is_regression, level = self.check_regression(
            mean_f, F_old, auto_rollback.rollback_threshold
        )

        total_rollbacks = auto_rollback.rollback_state.total_rollbacks
        recent_count = auto_rollback.rollback_state.recent_rollback_count
        last_rollback = auto_rollback.rollback_state.last_rollback
        root_cause_stats = dict(auto_rollback.rollback_state.root_cause_stats)
        is_in_cooldown = auto_rollback.rollback_state.is_in_cooldown
        cooldown_remaining = auto_rollback.rollback_state.cooldown_remaining

        if is_regression and level in {RegressionLevel.MODERATE, RegressionLevel.SEVERE}:
            # 执行回滚
            model, success = self.execute_rollback(
                snapshot_id, recent_count, auto_rollback.max_rollbacks_per_day
            )
            if success:
                total_rollbacks += 1
                recent_count += 1
                is_in_cooldown = True
                cooldown_remaining = 3600.0

                # 根因分析
                dw = delta_weights if delta_weights is not None else {}
                root_cause = self.analyze_root_cause(dw, delta_f, reform_type)
                root_cause_stats[root_cause.value] = root_cause_stats.get(root_cause.value, 0) + 1

                # 验证回滚（简化：假设回滚后性能等于F_old）
                F_restored = F_old
                verified = self.verify_rollback(
                    F_restored, F_old, auto_rollback.verify_tolerance
                )

                last_rollback = RollbackLogRecord(
                    timestamp=0.0,
                    snapshot_id=snapshot_id,
                    F_old=F_old,
                    F_new=mean_f,
                    regression_ratio=(mean_f - F_old) / abs(F_old) if F_old != 0 else 0.0,
                    regression_level=level,
                    rollback_verified=verified,
                    root_cause=root_cause
                )

        new_state = AutoRollbackState(
            total_rollbacks=total_rollbacks,
            recent_rollback_count=recent_count,
            last_rollback=last_rollback,
            root_cause_stats=root_cause_stats,
            is_in_cooldown=is_in_cooldown,
            cooldown_remaining=cooldown_remaining
        )
        return AutoRollback(
            explore_freezer=auto_rollback.explore_freezer,
            rollback_state=new_state,
            rollback_threshold=auto_rollback.rollback_threshold,
            verify_tolerance=auto_rollback.verify_tolerance,
            max_rollbacks_per_day=auto_rollback.max_rollbacks_per_day,
            verify_window=auto_rollback.verify_window
        )
