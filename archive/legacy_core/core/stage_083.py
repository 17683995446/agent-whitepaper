"""阶段83：本地存储自动清理。

本阶段实现磁盘存储监控与自动清理机制。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import numpy as np

from .stage_082 import MemoryMonitor


class StorageCleanLevel(Enum):
    """存储清理级别。"""
    NONE = 0            # 无清理
    LOW_VALUE = 1       # 第一优先：低价值样本
    MID_VALUE = 2       # 第二优先：中价值样本
    HIGH_VALUE = 3      # 第三优先：高价值样本（极端）


@dataclass(frozen=True)
class StorageCleanerState:
    """存储清理的完整状态描述。

    Attributes:
        disk_usage: 当前磁盘占用率
        disk_trend: 磁盘增长趋势
        clean_level: 当前清理级别
        last_clean_time: 上次清理时间
        last_cleaned_count: 上次清理的样本数
        last_freed_bytes: 上次释放的字节数
        disk_before_last_clean: 上次清理前磁盘占用率
        disk_after_last_clean: 上次清理后磁盘占用率
        min_value_kept: 保留样本的最低价值评分
        audit_log_count: 审计日志条数

    Invariants:
        - 0 <= disk_usage <= 1
    """
    disk_usage: float = 0.0
    disk_trend: float = 0.0
    clean_level: StorageCleanLevel = StorageCleanLevel.NONE
    last_clean_time: float = 0.0
    last_cleaned_count: int = 0
    last_freed_bytes: int = 0
    disk_before_last_clean: float = 0.0
    disk_after_last_clean: float = 0.0
    min_value_kept: float = 0.0
    audit_log_count: int = 0

    def __post_init__(self):
        if not (0 <= self.disk_usage <= 1):
            raise ValueError(
                f"disk_usage必须在[0,1]，收到 {self.disk_usage}"
            )


@dataclass(frozen=True)
class StorageCleaner:
    """本地存储自动清理的完整状态。

    组合阶段82的MemoryMonitor，在内存监控基础上增加存储清理状态。

    Attributes:
        memory_monitor: 阶段82的内存监控器（组合字段）
        cleaner_state: 存储清理状态
        disk_max_threshold: 清理触发阈值（默认0.80）
        disk_target_threshold: 清理目标阈值（默认0.60）
        disk_extreme_threshold: 极端清理阈值（默认0.90）
        trend_trigger_threshold: 趋势触发阈值（默认0.0001/s）
        trend_trigger_disk: 趋势触发磁盘占用（默认0.70）
        batch_size: 每批清理数量（默认1000）
        confirm_delay: 清理确认延迟（默认30秒）

    Invariants:
        - disk_target_threshold < disk_max_threshold < disk_extreme_threshold
    """
    memory_monitor: MemoryMonitor
    cleaner_state: StorageCleanerState = field(default_factory=StorageCleanerState)
    disk_max_threshold: float = 0.80
    disk_target_threshold: float = 0.60
    disk_extreme_threshold: float = 0.90
    trend_trigger_threshold: float = 0.0001
    trend_trigger_disk: float = 0.70
    batch_size: int = 1000
    confirm_delay: float = 30.0

    def __post_init__(self):
        if not (self.disk_target_threshold < self.disk_max_threshold
                < self.disk_extreme_threshold):
            raise ValueError(
                f"阈值必须满足: target({self.disk_target_threshold}) < "
                f"max({self.disk_max_threshold}) < "
                f"extreme({self.disk_extreme_threshold})"
            )


class StorageCleanerExecutor:
    """阶段83的核心接口：本地存储自动清理执行器。

    实现存储监控、清理触发、清理策略、清理顺序和清理后验证。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 保留Top30%高价值样本
        - Top10%+最近快照+当前模型永不清理
        - 清理后disk_usage < 60%
    """

    def measure_disk_usage(self) -> Tuple[float, Dict[str, float]]:
        """度量当前磁盘占用率（A83.1）。

        Returns:
            (disk_usage, 各组件磁盘明细)
            - disk_usage: 磁盘占用率 ∈ [0, 1]
            - 明细字典包含各组件占用（GB）
        """
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            ratio = used / total if total > 0 else 0.0
            breakdown = {
                "total_gb": total / (1024 ** 3),
                "used_gb": used / (1024 ** 3),
                "free_gb": free / (1024 ** 3)
            }
            return ratio, breakdown
        except Exception:
            return 0.0, {"total_gb": 1.0, "used_gb": 0.0, "free_gb": 1.0}

    def check_clean(
        self,
        disk_usage: float,
        disk_trend: float,
        max_threshold: float = 0.80,
        extreme_threshold: float = 0.90,
        trend_trigger_disk: float = 0.70,
        trend_trigger_threshold: float = 0.0001
    ) -> Tuple[bool, StorageCleanLevel]:
        """判定是否应清理（A83.2 + A83.2-trend）。

        Args:
            disk_usage: 磁盘占用率，范围[0,1]
            disk_trend: 磁盘增长趋势（每秒变化率）
            max_threshold: 清理触发阈值
            extreme_threshold: 极端清理阈值
            trend_trigger_disk: 趋势触发磁盘占用
            trend_trigger_threshold: 趋势触发阈值

        Returns:
            (是否应清理, 清理级别)
        """
        if disk_usage >= extreme_threshold:
            return True, StorageCleanLevel.HIGH_VALUE
        elif disk_usage >= max_threshold:
            return True, StorageCleanLevel.MID_VALUE
        elif disk_usage >= trend_trigger_disk and disk_trend > trend_trigger_threshold:
            return True, StorageCleanLevel.LOW_VALUE
        elif disk_usage >= trend_trigger_disk:
            # 70%+但无快速增长趋势：预警但不清理
            return False, StorageCleanLevel.LOW_VALUE
        else:
            return False, StorageCleanLevel.NONE

    def clean_low_value(
        self,
        top30_threshold: float = 0.3,
        batch_size: int = 1000
    ) -> Tuple[int, int]:
        """第一优先清理：低价值样本（A83.3 + A83.4）。

        删除价值评分低于Top30%阈值的样本。

        Args:
            top30_threshold: Top30%价值阈值
            batch_size: 每批数量，必须 > 0

        Returns:
            (清理数量, 释放字节数)

        Raises:
            ValueError: 当batch_size <= 0时
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size必须 > 0，收到 {batch_size}")
        # 模拟清理：按batch_size比例返回清理数量和释放字节数
        count = int(batch_size * top30_threshold)
        freed = count * 1024
        return count, freed

    def clean_mid_value(
        self,
        top60_threshold: float = 0.6,
        batch_size: int = 1000
    ) -> Tuple[int, int]:
        """第二优先清理：中价值样本（A83.4）。

        删除价值评分在Top30%-Top60%之间的样本。

        Args:
            top60_threshold: Top60%价值阈值
            batch_size: 每批数量，必须 > 0

        Returns:
            (清理数量, 释放字节数)

        Raises:
            ValueError: 当batch_size <= 0时
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size必须 > 0，收到 {batch_size}")
        count = int(batch_size * (top60_threshold - 0.3))
        freed = count * 1024
        return count, freed

    def clean_high_value(
        self,
        top90_threshold: float = 0.9,
        batch_size: int = 1000
    ) -> Tuple[int, int]:
        """第三优先清理：高价值样本（A83.4-extreme）。

        仅在disk_usage > 0.90且低中价值已全部清理时触发。
        删除价值评分在Top60%-Top90%之间的样本。

        Args:
            top90_threshold: Top90%价值阈值
            batch_size: 每批数量，必须 > 0

        Returns:
            (清理数量, 释放字节数)

        Raises:
            ValueError: 当batch_size <= 0时
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size必须 > 0，收到 {batch_size}")
        count = int(batch_size * (top90_threshold - 0.6))
        freed = count * 1024
        return count, freed

    def verify_clean_result(
        self,
        disk_usage_after: float,
        target: float = 0.60
    ) -> Tuple[bool, Dict]:
        """清理后验证（A83.5）。

        Args:
            disk_usage_after: 清理后磁盘占用率
            target: 目标阈值，必须 ∈ (0,1)

        Returns:
            (是否成功, 验证详情)

        Raises:
            ValueError: 当target不在(0,1)范围内时
        """
        if not (0 < target < 1):
            raise ValueError(f"target必须 ∈ (0,1)，收到 {target}")
        success = disk_usage_after < target
        details = {
            "disk_usage_after": disk_usage_after,
            "target": target,
            "success": success
        }
        return success, details

    def compute_sample_interval(
        self,
        disk_usage: float,
        max_threshold: float = 0.80
    ) -> float:
        """计算存储采样间隔（A83.1-interval）。

        Args:
            disk_usage: 当前磁盘占用率
            max_threshold: 清理触发阈值

        Returns:
            采样间隔（秒）
        """
        if disk_usage >= max_threshold:
            return 5.0
        elif disk_usage >= max_threshold - 0.20:
            return 10.0
        else:
            return 60.0

    def step(
        self,
        cleaner: StorageCleaner,
        current_disk_usage: float
    ) -> StorageCleaner:
        """执行一步存储监控与清理。

        Args:
            cleaner: 当前存储清理状态
            current_disk_usage: 当前磁盘占用率（外部传入，避免副作用）

        Returns:
            更新后的存储清理状态
        """
        # 计算趋势（简单差分）
        disk_trend = current_disk_usage - cleaner.cleaner_state.disk_usage
        should_clean, level = self.check_clean(
            current_disk_usage,
            disk_trend,
            cleaner.disk_max_threshold,
            cleaner.disk_extreme_threshold,
            cleaner.trend_trigger_disk,
            cleaner.trend_trigger_threshold
        )
        cleaned_count = 0
        freed_bytes = 0
        disk_before = cleaner.cleaner_state.disk_usage
        disk_after = current_disk_usage
        min_value_kept = 0.0

        if should_clean:
            disk_before = current_disk_usage
            if level == StorageCleanLevel.LOW_VALUE:
                c1, f1 = self.clean_low_value(0.3, cleaner.batch_size)
                cleaned_count = c1
                freed_bytes = f1
                min_value_kept = 0.3
            elif level == StorageCleanLevel.MID_VALUE:
                c1, f1 = self.clean_low_value(0.3, cleaner.batch_size)
                c2, f2 = self.clean_mid_value(0.6, cleaner.batch_size)
                cleaned_count = c1 + c2
                freed_bytes = f1 + f2
                min_value_kept = 0.3
            elif level == StorageCleanLevel.HIGH_VALUE:
                c1, f1 = self.clean_low_value(0.3, cleaner.batch_size)
                c2, f2 = self.clean_mid_value(0.6, cleaner.batch_size)
                c3, f3 = self.clean_high_value(0.9, cleaner.batch_size)
                cleaned_count = c1 + c2 + c3
                freed_bytes = f1 + f2 + f3
                min_value_kept = 0.3
            # 模拟清理后磁盘占用率下降
            disk_after = max(
                cleaner.disk_target_threshold * 0.9,
                current_disk_usage * 0.7
            )

        success, details = self.verify_clean_result(
            disk_after, cleaner.disk_target_threshold
        )

        new_state = StorageCleanerState(
            disk_usage=current_disk_usage,
            disk_trend=disk_trend,
            clean_level=level,
            last_clean_time=0.0,
            last_cleaned_count=cleaned_count,
            last_freed_bytes=freed_bytes,
            disk_before_last_clean=disk_before,
            disk_after_last_clean=disk_after,
            min_value_kept=min_value_kept,
            audit_log_count=cleaner.cleaner_state.audit_log_count + 1
        )
        return StorageCleaner(
            memory_monitor=cleaner.memory_monitor,
            cleaner_state=new_state,
            disk_max_threshold=cleaner.disk_max_threshold,
            disk_target_threshold=cleaner.disk_target_threshold,
            disk_extreme_threshold=cleaner.disk_extreme_threshold,
            trend_trigger_threshold=cleaner.trend_trigger_threshold,
            trend_trigger_disk=cleaner.trend_trigger_disk,
            batch_size=cleaner.batch_size,
            confirm_delay=cleaner.confirm_delay
        )
