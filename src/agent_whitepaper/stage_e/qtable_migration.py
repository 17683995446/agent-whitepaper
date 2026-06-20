"""阶段E-1：Q表序列化与迁移 - Agent间知识传递的基础。

第一性原理：
    之前演示中"技能共享+0%"的根本原因：Q表是独立的，共享的技能只是参考，
    没有真正注入到Q值中。要实现真正的技能迁移，必须能：
    1. 序列化Agent的Q表（提取知识）
    2. 将Q表迁移给另一个Agent（注入知识）
    3. 合并多个Q表（集体知识聚合）

    Q表迁移是"自主增强技能"的真正实现：
    Agent A学会的"如何做"直接变成Agent B的初始能力。
"""

import base64
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from agent_whitepaper.stage_a.policy import QTable


# ============================================================
# Q表快照
# ============================================================
@dataclass(frozen=True)
class QTableSnapshot:
    """Q表快照：可序列化、可迁移的知识载体。

    Attributes:
        n_states: 状态数
        n_actions: 动作数
        table_data: Q表数据（base64编码的pickle）
        learning_rate: 学习率
        contributor_id: 贡献者Agent ID
        performance_score: 性能评分（贡献时的成功率）
        created_at: 创建时间戳
    """
    n_states: int
    n_actions: int
    table_data: str  # base64编码
    learning_rate: float
    contributor_id: str
    performance_score: float
    created_at: float

    @classmethod
    def from_q_table(
        cls,
        q_table: QTable,
        contributor_id: str,
        performance_score: float = 0.0
    ) -> "QTableSnapshot":
        """从Q表创建快照。"""
        import time
        table_bytes = pickle.dumps(q_table._table)
        table_b64 = base64.b64encode(table_bytes).decode('ascii')
        return cls(
            n_states=q_table.n_states,
            n_actions=q_table.n_actions,
            table_data=table_b64,
            learning_rate=q_table.lr,
            contributor_id=contributor_id,
            performance_score=performance_score,
            created_at=time.time(),
        )

    def to_q_table(self) -> QTable:
        """从快照恢复Q表。"""
        table_bytes = base64.b64decode(self.table_data)
        table_array = pickle.loads(table_bytes)
        q_table = QTable(
            n_states=self.n_states,
            n_actions=self.n_actions,
            learning_rate=self.learning_rate,
        )
        q_table._table = table_array.copy()
        return q_table

    def get_table_array(self) -> np.ndarray:
        """直接获取Q表数组。"""
        table_bytes = base64.b64decode(self.table_data)
        return pickle.loads(table_bytes)


# ============================================================
# Q表迁移器
# ============================================================
class QTableMigrator:
    """Q表迁移器：在Agent间迁移Q表知识。

    迁移策略：
    1. 直接迁移：用源Q表完全替换目标Q表
    2. 软迁移：源Q表与目标Q表加权平均
    3. 选择性迁移：只迁移高置信度的状态-动作对
    4. 增量迁移：在目标Q表基础上叠加源Q表的增量
    """

    @staticmethod
    def direct_migrate(
        source: QTableSnapshot,
        target_q_table: QTable
    ) -> QTable:
        """直接迁移：完全替换。"""
        if source.n_states != target_q_table.n_states or \
           source.n_actions != target_q_table.n_actions:
            raise ValueError(
                f"维度不匹配: 源({source.n_states},{source.n_actions}) "
                f"目标({target_q_table.n_states},{target_q_table.n_actions})"
            )
        target_q_table._table = source.get_table_array().copy()
        return target_q_table

    @staticmethod
    def soft_migrate(
        source: QTableSnapshot,
        target_q_table: QTable,
        alpha: float = 0.5
    ) -> QTable:
        """软迁移：加权平均。

        target = (1-α) * target + α * source
        """
        if not (0 <= alpha <= 1):
            raise ValueError("alpha 必须 ∈ [0,1]")
        if source.n_states != target_q_table.n_states or \
           source.n_actions != target_q_table.n_actions:
            raise ValueError("维度不匹配")

        source_table = source.get_table_array()
        target_q_table._table = (
            (1 - alpha) * target_q_table._table + alpha * source_table
        )
        return target_q_table

    @staticmethod
    def selective_migrate(
        source: QTableSnapshot,
        target_q_table: QTable,
        confidence_threshold: float = 0.1
    ) -> QTable:
        """选择性迁移：只迁移高置信度的状态-动作对。

        置信度 = 源Q表中该状态-动作的绝对值（值越大说明越确定）
        """
        if source.n_states != target_q_table.n_states or \
           source.n_actions != target_q_table.n_actions:
            raise ValueError("维度不匹配")

        source_table = source.get_table_array()
        # 计算每个状态的置信度（max Q值的绝对值）
        confidence = np.max(np.abs(source_table), axis=1)
        # 只迁移高置信度状态
        mask = confidence >= confidence_threshold
        for state_idx in range(source.n_states):
            if mask[state_idx]:
                target_q_table._table[state_idx] = source_table[state_idx].copy()
        return target_q_table

    @staticmethod
    def incremental_migrate(
        source: QTableSnapshot,
        target_q_table: QTable,
        scale: float = 0.3
    ) -> QTable:
        """增量迁移：在目标基础上叠加源的增量。

        target = target + scale * source
        """
        if source.n_states != target_q_table.n_states or \
           source.n_actions != target_q_table.n_actions:
            raise ValueError("维度不匹配")

        source_table = source.get_table_array()
        target_q_table._table += scale * source_table
        return target_q_table

    @staticmethod
    def merge_q_tables(
        snapshots: List[QTableSnapshot],
        weights: Optional[List[float]] = None
    ) -> QTableSnapshot:
        """合并多个Q表快照。

        Args:
            snapshots: 多个Q表快照
            weights: 各快照的权重（None=等权）

        Returns:
            合并后的新快照
        """
        if not snapshots:
            raise ValueError("快照列表不能为空")
        if weights is None:
            weights = [1.0 / len(snapshots)] * len(snapshots)
        if len(weights) != len(snapshots):
            raise ValueError("权重数必须等于快照数")
        if abs(sum(weights) - 1.0) > 1e-6:
            # 归一化
            weights = [w / sum(weights) for w in weights]

        # 加权平均
        merged_table = np.zeros_like(snapshots[0].get_table_array())
        for snapshot, weight in zip(snapshots, weights):
            merged_table += weight * snapshot.get_table_array()

        # 创建新快照
        import time
        table_bytes = pickle.dumps(merged_table)
        table_b64 = base64.b64encode(table_bytes).decode('ascii')
        avg_performance = sum(
            s.performance_score * w for s, w in zip(snapshots, weights)
        )
        return QTableSnapshot(
            n_states=snapshots[0].n_states,
            n_actions=snapshots[0].n_actions,
            table_data=table_b64,
            learning_rate=snapshots[0].learning_rate,
            contributor_id="merged",
            performance_score=avg_performance,
            created_at=time.time(),
        )


# ============================================================
# Q表知识库
# ============================================================
class QTableKnowledgeBase:
    """Q表知识库：存储和管理所有Agent的Q表快照。

    功能：
    1. 存储Agent的Q表快照
    2. 按性能评分检索最佳Q表
    3. 合并多个Q表形成"集体知识"
    4. 为新Agent提供初始Q表
    """

    def __init__(self) -> None:
        self._snapshots: Dict[str, QTableSnapshot] = {}
        self._index_by_contributor: Dict[str, List[str]] = {}

    def contribute(
        self,
        snapshot: QTableSnapshot,
        name: Optional[str] = None
    ) -> str:
        """贡献Q表快照。"""
        snapshot_name = name or f"qtable_{len(self._snapshots)}"
        self._snapshots[snapshot_name] = snapshot
        if snapshot.contributor_id not in self._index_by_contributor:
            self._index_by_contributor[snapshot.contributor_id] = []
        self._index_by_contributor[snapshot.contributor_id].append(snapshot_name)
        return snapshot_name

    def get_snapshot(self, name: str) -> Optional[QTableSnapshot]:
        """获取快照。"""
        return self._snapshots.get(name)

    def get_best_snapshot(self) -> Optional[QTableSnapshot]:
        """获取性能最好的快照。"""
        if not self._snapshots:
            return None
        return max(self._snapshots.values(), key=lambda s: s.performance_score)

    def get_merged_knowledge(
        self,
        min_performance: float = 0.0,
        weights: Optional[List[float]] = None
    ) -> Optional[QTableSnapshot]:
        """获取合并的集体知识。

        只合并性能超过阈值的快照。
        """
        qualified = [
            s for s in self._snapshots.values()
            if s.performance_score >= min_performance
        ]
        if not qualified:
            return None
        if len(qualified) == 1:
            return qualified[0]
        return QTableMigrator.merge_q_tables(qualified, weights)

    def initialize_new_agent(
        self,
        q_table: QTable,
        migration_strategy: str = "soft",
        alpha: float = 0.7
    ) -> QTable:
        """用集体知识初始化新Agent的Q表。

        Args:
            q_table: 新Agent的Q表（初始为零）
            migration_strategy: 迁移策略 (direct/soft/selective/incremental)
            alpha: 软迁移的权重

        Returns:
            初始化后的Q表
        """
        best = self.get_best_snapshot()
        if best is None:
            return q_table  # 没有知识，保持零初始化

        if migration_strategy == "direct":
            return QTableMigrator.direct_migrate(best, q_table)
        elif migration_strategy == "soft":
            return QTableMigrator.soft_migrate(best, q_table, alpha)
        elif migration_strategy == "selective":
            return QTableMigrator.selective_migrate(best, q_table)
        elif migration_strategy == "incremental":
            return QTableMigrator.incremental_migrate(best, q_table)
        else:
            raise ValueError(f"未知迁移策略: {migration_strategy}")

    def get_stats(self) -> Dict:
        """获取知识库统计。"""
        return {
            "total_snapshots": len(self._snapshots),
            "contributors": len(self._index_by_contributor),
            "avg_performance": (
                sum(s.performance_score for s in self._snapshots.values())
                / len(self._snapshots)
            ) if self._snapshots else 0.0,
            "best_performance": max(
                (s.performance_score for s in self._snapshots.values()),
                default=0.0
            ),
        }
