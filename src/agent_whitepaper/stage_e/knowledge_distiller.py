"""阶段E-3：跨Agent知识蒸馏 - 集体智慧的提取与传播。

第一性原理：
    知识蒸馏（Knowledge Distillation）是从多个"教师"Agent
    提取知识，传授给"学生"Agent的过程。

    与简单Q表迁移的区别：
    - 迁移：直接复制Q表
    - 蒸馏：从多个Agent的Q表中提取"共识"和"精华"

    蒸馏策略：
    1. 投票蒸馏：多个Agent对同一状态-动作的价值取平均/中位数
    2. 置信度加权：性能好的Agent权重更大
    3. 冲突消解：当Agent意见不一时，取多数派或加权平均
    4. 精华提取：只提取所有Agent都认为高价值的状态-动作对

    这实现了真正的"集体智慧"：1+1>2。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from agent_whitepaper.stage_a.policy import QTable
from agent_whitepaper.stage_b.skill_abstractor import SkillLibrary

from .qtable_migration import QTableKnowledgeBase, QTableSnapshot
from .skill_injector import SkillQValueInjector


# ============================================================
# 蒸馏结果
# ============================================================
@dataclass(frozen=True)
class DistillationResult:
    """知识蒸馏结果。

    Attributes:
        student_id: 学生Agent ID
        teacher_ids: 教师Agent ID列表
        strategy: 蒸馏策略
        states_distilled: 蒸馏的状态数
        consensus_ratio: 共识比例（教师意见一致的比例）
        performance_gain: 性能提升估计
    """
    student_id: str
    teacher_ids: Tuple[str, ...]
    strategy: str
    states_distilled: int
    consensus_ratio: float
    performance_gain: float


# ============================================================
# 知识蒸馏器
# ============================================================
class KnowledgeDistiller:
    """知识蒸馏器：从多个教师Agent提取知识传授给学生。

    支持的蒸馏策略：
    1. average: 教师Q表的平均值
    2. weighted: 按教师性能加权平均
    3. median: 教师Q表的中位数（抗异常值）
    4. consensus: 只保留所有教师都认为高价值的状态-动作
    """

    def __init__(self, grid_size: int = 5):
        self.grid_size = grid_size

    def distill(
        self,
        teacher_snapshots: List[QTableSnapshot],
        student_q_table: QTable,
        strategy: str = "weighted",
        injection_alpha: float = 0.8
    ) -> DistillationResult:
        """执行知识蒸馏。

        Args:
            teacher_snapshots: 教师Agent的Q表快照列表
            student_q_table: 学生Agent的Q表（将被修改）
            strategy: 蒸馏策略
            injection_alpha: 注入权重（0=不注入, 1=完全替换）

        Returns:
            蒸馏结果
        """
        if not teacher_snapshots:
            return DistillationResult(
                student_id="",
                teacher_ids=(),
                strategy=strategy,
                states_distilled=0,
                consensus_ratio=0.0,
                performance_gain=0.0,
            )

        # 验证维度一致
        n_states = student_q_table.n_states
        n_actions = student_q_table.n_actions
        for snap in teacher_snapshots:
            if snap.n_states != n_states or snap.n_actions != n_actions:
                raise ValueError("教师与学生Q表维度不匹配")

        # 提取所有教师Q表
        teacher_tables = [s.get_table_array() for s in teacher_snapshots]
        teacher_ids = tuple(s.contributor_id for s in teacher_snapshots)

        # 根据策略计算蒸馏后的Q表
        if strategy == "average":
            distilled = np.mean(teacher_tables, axis=0)
        elif strategy == "weighted":
            # 按性能评分加权
            weights = np.array([s.performance_score for s in teacher_snapshots])
            weights = weights / (weights.sum() + 1e-10)
            distilled = np.zeros_like(teacher_tables[0])
            for w, table in zip(weights, teacher_tables):
                distilled += w * table
        elif strategy == "median":
            distilled = np.median(teacher_tables, axis=0)
        elif strategy == "consensus":
            # 只保留所有教师都认为高价值的状态-动作
            threshold = 0.1
            all_high = np.all(
                [np.max(t, axis=1, keepdims=True) > threshold for t in teacher_tables],
                axis=0
            )
            avg_table = np.mean(teacher_tables, axis=0)
            distilled = np.where(all_high, avg_table, student_q_table._table)
        else:
            raise ValueError(f"未知蒸馏策略: {strategy}")

        # 计算共识比例
        consensus_ratio = self._compute_consensus(teacher_tables)

        # 注入到学生Q表
        original_table = student_q_table._table.copy()
        student_q_table._table = (
            (1 - injection_alpha) * student_q_table._table
            + injection_alpha * distilled
        )

        # 估计性能提升
        performance_gain = self._estimate_performance_gain(
            original_table, student_q_table._table
        )

        return DistillationResult(
            student_id="student",
            teacher_ids=teacher_ids,
            strategy=strategy,
            states_distilled=n_states,
            consensus_ratio=consensus_ratio,
            performance_gain=performance_gain,
        )

    def _compute_consensus(self, teacher_tables: List[np.ndarray]) -> float:
        """计算教师间的共识比例。

        对每个状态，检查所有教师是否选择了相同的最优动作。
        """
        if len(teacher_tables) <= 1:
            return 1.0
        # 每个教师的最优动作
        best_actions = [np.argmax(t, axis=1) for t in teacher_tables]
        # 检查一致性
        n_states = teacher_tables[0].shape[0]
        consensus_count = 0
        for state in range(n_states):
            actions = [ba[state] for ba in best_actions]
            if len(set(actions)) == 1:  # 所有教师一致
                consensus_count += 1
        return consensus_count / n_states

    def _estimate_performance_gain(
        self,
        old_table: np.ndarray,
        new_table: np.ndarray
    ) -> float:
        """估计性能提升。

        简化估计：新Q表的max值总和 - 旧Q表的max值总和
        """
        old_max_sum = float(np.sum(np.max(old_table, axis=1)))
        new_max_sum = float(np.sum(np.max(new_table, axis=1)))
        return new_max_sum - old_max_sum

    def distill_from_knowledge_base(
        self,
        knowledge_base: QTableKnowledgeBase,
        student_q_table: QTable,
        strategy: str = "weighted",
        min_teachers: int = 1,
        min_performance: float = 0.1
    ) -> DistillationResult:
        """从知识库蒸馏知识到学生。

        只使用性能超过阈值的教师。
        """
        # 筛选合格教师
        qualified = [
            s for s in knowledge_base._snapshots.values()
            if s.performance_score >= min_performance
        ]
        if len(qualified) < min_teachers:
            return DistillationResult(
                student_id="student",
                teacher_ids=(),
                strategy=strategy,
                states_distilled=0,
                consensus_ratio=0.0,
                performance_gain=0.0,
            )

        return self.distill(qualified, student_q_table, strategy)


# ============================================================
# 集体学习协调器
# ============================================================
class CollectiveLearningCoordinator:
    """集体学习协调器：协调多Agent的集体学习过程。

    工作流程：
    1. 收集：从所有Agent收集Q表快照
    2. 评估：评估各Agent性能
    3. 蒸馏：从优秀教师蒸馏知识
    4. 注入：将蒸馏知识注入新Agent或弱Agent
    5. 技能迁移：同时迁移技能库
    """

    def __init__(self, grid_size: int = 5):
        self.grid_size = grid_size
        self.knowledge_base = QTableKnowledgeBase()
        self.distiller = KnowledgeDistiller(grid_size=grid_size)
        self.skill_injector = SkillQValueInjector(grid_size=grid_size)
        self._distillation_history: List[DistillationResult] = []

    def collect_from_agent(
        self,
        agent_id: str,
        q_table: QTable,
        performance_score: float
    ) -> str:
        """从Agent收集知识。"""
        snapshot = QTableSnapshot.from_q_table(
            q_table, agent_id, performance_score
        )
        return self.knowledge_base.contribute(snapshot, name=f"qtable_{agent_id}")

    def teach_new_agent(
        self,
        new_q_table: QTable,
        strategy: str = "weighted",
        inject_skills: bool = True,
        skill_library: Optional[SkillLibrary] = None
    ) -> Tuple[DistillationResult, List]:
        """教导新Agent。

        Returns:
            (蒸馏结果, 技能注入结果列表)
        """
        # 蒸馏知识
        distill_result = self.distiller.distill_from_knowledge_base(
            self.knowledge_base, new_q_table, strategy=strategy
        )
        self._distillation_history.append(distill_result)

        # 注入技能
        injection_results = []
        if inject_skills and skill_library is not None:
            injection_results = self.skill_injector.inject_all_skills(
                skill_library, new_q_table, min_success_rate=0.3
            )

        return distill_result, injection_results

    def get_collective_stats(self) -> Dict:
        """获取集体学习统计。"""
        kb_stats = self.knowledge_base.get_stats()
        return {
            **kb_stats,
            "total_distillations": len(self._distillation_history),
            "avg_consensus": (
                sum(r.consensus_ratio for r in self._distillation_history)
                / len(self._distillation_history)
            ) if self._distillation_history else 0.0,
            "avg_performance_gain": (
                sum(r.performance_gain for r in self._distillation_history)
                / len(self._distillation_history)
            ) if self._distillation_history else 0.0,
        }
