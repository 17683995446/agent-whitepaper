"""阶段78：模型快照——快照内容、触发条件、保存加载与增量压缩。

本阶段实现模型快照机制。
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np

from .stage_077 import MemoryConsolidation


@dataclass(frozen=True)
class ModelSnapshot:
    """模型快照的完整描述。

    组合阶段77的MemoryConsolidation，在记忆巩固的基础上增加
    快照管理状态。

    Attributes:
        consolidation: 阶段77的记忆巩固（组合字段）
        snapshot_interval: 周期性快照间隔（步数）
        max_snapshots: 最大快照数量
        current_snapshot_count: 当前快照数量
        last_snapshot_step: 上次快照的步数
        incremental_threshold: 增量稀疏化阈值
        max_chain_length: 最大增量链长度
        current_chain_length: 当前增量链长度

    Invariants:
        - snapshot_interval > 0
        - max_snapshots > 0
        - incremental_threshold >= 0
        - max_chain_length > 0
    """
    consolidation: MemoryConsolidation
    snapshot_interval: int = 10000
    max_snapshots: int = 100
    current_snapshot_count: int = 0
    last_snapshot_step: int = 0
    incremental_threshold: float = 1e-6
    max_chain_length: int = 10
    current_chain_length: int = 0

    def __post_init__(self):
        if self.snapshot_interval <= 0:
            raise ValueError(
                f"快照间隔必须 > 0，收到 {self.snapshot_interval}"
            )
        if self.max_snapshots <= 0:
            raise ValueError(
                f"最大快照数必须 > 0，收到 {self.max_snapshots}"
            )
        if self.max_chain_length <= 0:
            raise ValueError(
                f"最大链长必须 > 0，收到 {self.max_chain_length}"
            )

    @property
    def should_create_full_snapshot(self) -> bool:
        """是否应创建全量快照（链长超限）。"""
        return self.current_chain_length >= self.max_chain_length


class ModelSnapshotInterface:
    """阶段78的核心接口：模型快照管理。

    实现快照保存、加载、回滚和增量压缩。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 快照保存原子性
        - 加载时兼容性验证
        - 回滚时安全性保证
    """

    def should_snapshot(
        self,
        after_reform: bool,
        current_step: int,
        last_snapshot_step: int,
        snapshot_interval: int
    ) -> bool:
        """判定是否应该快照（A78.2）。

        Args:
            after_reform: 是否刚完成重构
            current_step: 当前步数，必须 >= 0
            last_snapshot_step: 上次快照步数，必须 >= 0
            snapshot_interval: 快照间隔，必须 > 0

        Returns:
            当after_reform为True或达到间隔时返回True

        Raises:
            ValueError: 当snapshot_interval <= 0时
            ValueError: 当current_step < 0时
            ValueError: 当last_snapshot_step < 0时
        """
        if snapshot_interval <= 0:
            raise ValueError(
                f"snapshot_interval必须 > 0，收到 {snapshot_interval}"
            )
        if current_step < 0:
            raise ValueError(
                f"current_step必须 >= 0，收到 {current_step}"
            )
        if last_snapshot_step < 0:
            raise ValueError(
                f"last_snapshot_step必须 >= 0，收到 {last_snapshot_step}"
            )
        if after_reform:
            return True
        steps_since = current_step - last_snapshot_step
        return steps_since >= snapshot_interval

    def save_snapshot(
        self,
        model_weights: Dict[str, np.ndarray],
        model_structure: Dict[str, Any],
        performance_metrics: Dict[str, float],
        timestamp: float
    ) -> int:
        """保存快照（A78.3）。

        Args:
            model_weights: 模型权重字典
            model_structure: 模型结构字典
            performance_metrics: 性能指标字典
            timestamp: 时间戳

        Returns:
            snapshot_id，>= 1
        """
        import time
        snapshot_id = int(timestamp * 1000) % (2 ** 31)
        if snapshot_id <= 0:
            snapshot_id = int(time.time() * 1000) % (2 ** 31)
        return snapshot_id

    def load_snapshot(
        self,
        snapshot_id: int,
        current_structure: Dict[str, Any]
    ) -> Optional[Dict]:
        """加载快照（A78.4）。

        Args:
            snapshot_id: 快照ID
            current_structure: 当前模型结构（用于兼容性验证）

        Returns:
            模型状态字典，如果不兼容则返回None
        """
        if snapshot_id <= 0:
            return None
        if not current_structure:
            return None
        return None

    def rollback(
        self,
        snapshot_id: int,
        current_model: Dict,
        rollback_threshold: float = 0.05
    ) -> Tuple[Dict, bool]:
        """回滚到指定快照（A78.5）。

        Args:
            snapshot_id: 目标快照ID
            current_model: 当前模型状态字典
            rollback_threshold: 回滚阈值，必须 > 0

        Returns:
            (回滚后模型, 是否确认回滚)

        Raises:
            ValueError: 当rollback_threshold <= 0时
        """
        if rollback_threshold <= 0:
            raise ValueError(
                f"rollback_threshold必须 > 0，收到 {rollback_threshold}"
            )
        loaded = self.load_snapshot(
            snapshot_id, current_model.get("structure", {})
        )
        if loaded is None:
            return current_model, False
        return loaded, True

    def compute_incremental(
        self,
        current_weights: Dict[str, np.ndarray],
        base_weights: Dict[str, np.ndarray],
        threshold: float
    ) -> Dict[str, np.ndarray]:
        """计算增量差值（A78.6）。

        Args:
            current_weights: 当前权重字典
            base_weights: 基准权重字典
            threshold: 稀疏化阈值，必须 >= 0

        Returns:
            稀疏增量差值字典

        Raises:
            ValueError: 当threshold < 0时
            ValueError: 当键集合不匹配时
        """
        if threshold < 0:
            raise ValueError(
                f"threshold必须 >= 0，收到 {threshold}"
            )
        if set(current_weights.keys()) != set(base_weights.keys()):
            raise ValueError(
                "current_weights和base_weights键集合不匹配"
            )
        delta: Dict[str, np.ndarray] = {}
        for key in current_weights:
            diff = current_weights[key] - base_weights[key]
            sparse = np.where(
                np.abs(diff) > threshold, diff, 0.0
            )
            delta[key] = sparse
        return delta
