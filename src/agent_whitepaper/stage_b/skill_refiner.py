"""阶段B-2：技能微调器 - 基于自由能反馈优化技能。

第一性原理：
    技能抽象后并非最优，需要通过实践反馈微调。
    自由能原理：低自由能 = 好的预测 = 技能匹配环境。

    微调策略：
    1. 执行技能时记录自由能变化
    2. 自由能下降 → 技能有效，强化（增加权重）
    3. 自由能上升 → 技能不匹配，弱化或变异
    4. 变异：随机修改动作序列中的1-2个动作，探索更好的变体

    这实现了"自主增强技能"：技能不是固定的，
    而是随经验不断优化的活体知识。
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .skill_abstractor import Skill, SkillLibrary


# ============================================================
# 技能执行记录
# ============================================================
@dataclass(frozen=True)
class SkillExecutionRecord:
    """技能执行记录。

    Attributes:
        skill_name: 技能名称
        precondition_matched: 前置条件是否匹配
        free_energy_before: 执行前自由能
        free_energy_after: 执行后自由能
        actual_reward: 实际奖励
        success: 是否成功
        actions_executed: 实际执行的动作序列
    """
    skill_name: str
    precondition_matched: bool
    free_energy_before: float
    free_energy_after: float
    actual_reward: float
    success: bool
    actions_executed: Tuple[int, ...]

    @property
    def free_energy_delta(self) -> float:
        """自由能变化（负值=下降=好）。"""
        return self.free_energy_after - self.free_energy_before

    @property
    def effectiveness(self) -> float:
        """技能有效性评分 ∈ [0,1]。"""
        # 综合自由能变化和奖励
        fe_score = 1.0 / (1.0 + max(self.free_energy_delta, 0))
        reward_score = 1.0 / (1.0 + max(-self.actual_reward, 0))
        return 0.5 * fe_score + 0.5 * reward_score


# ============================================================
# 技能微调器
# ============================================================
class SkillRefiner:
    """技能微调器：基于执行反馈优化技能。

    核心机制：
    1. 强化：有效技能增加使用权重
    2. 弱化：无效技能降低使用权重
    3. 变异：对低效技能尝试动作变体
    4. 剪枝：删除长期无效的技能
    """

    def __init__(
        self,
        n_actions: int,
        mutation_rate: float = 0.1,
        min_effectiveness: float = 0.3,
        max_variants: int = 5
    ) -> None:
        if not (0 <= mutation_rate <= 1):
            raise ValueError("mutation_rate 必须 ∈ [0,1]")
        if not (0 <= min_effectiveness <= 1):
            raise ValueError("min_effectiveness 必须 ∈ [0,1]")
        self.n_actions = n_actions
        self.mutation_rate = mutation_rate
        self.min_effectiveness = min_effectiveness
        self.max_variants = max_variants
        self._execution_history: List[SkillExecutionRecord] = []
        self._rng = np.random.RandomState(42)

    def record_execution(self, record: SkillExecutionRecord) -> None:
        """记录技能执行结果。"""
        self._execution_history.append(record)
        # 只保留最近1000条
        if len(self._execution_history) > 1000:
            self._execution_history = self._execution_history[-1000:]

    def refine_skill(
        self,
        skill: Skill,
        library: SkillLibrary
    ) -> Optional[Skill]:
        """微调单个技能。

        Returns:
            微调后的新技能（如果产生了变体），否则None
        """
        # 获取该技能的执行记录
        records = [r for r in self._execution_history if r.skill_name == skill.name]
        if len(records) < 3:
            # 执行次数太少，不微调
            return None

        # 计算平均有效性
        avg_effectiveness = np.mean([r.effectiveness for r in records])
        avg_fe_delta = np.mean([r.free_energy_delta for r in records])

        if avg_effectiveness < self.min_effectiveness:
            # 技能无效，尝试变异
            return self._mutate_skill(skill, library)
        elif avg_fe_delta < 0:
            # 自由能下降，技能有效，强化（更新期望奖励）
            avg_reward = np.mean([r.actual_reward for r in records])
            success_count = sum(1 for r in records if r.success)
            refined = Skill(
                name=skill.name,
                precondition=skill.precondition,
                action_sequence=skill.action_sequence,
                postcondition=skill.postcondition,
                expected_reward=0.7 * skill.expected_reward + 0.3 * avg_reward,
                success_count=skill.success_count + success_count,
                total_count=skill.total_count + len(records),
                abstraction_level=skill.abstraction_level,
            )
            library.add_skill(refined)
            return refined
        return None

    def _mutate_skill(
        self,
        skill: Skill,
        library: SkillLibrary
    ) -> Optional[Skill]:
        """变异技能：随机修改动作序列。"""
        actions = list(skill.action_sequence)
        if len(actions) < 2:
            return None

        # 随机选择变异点
        n_mutations = max(1, int(len(actions) * self.mutation_rate))
        mutated = actions.copy()
        for _ in range(n_mutations):
            idx = self._rng.randint(len(mutated))
            mutated[idx] = self._rng.randint(self.n_actions)

        # 创建变体技能
        variant_name = f"{skill.name}_variant_{self._rng.randint(10000)}"
        variant = Skill(
            name=variant_name,
            precondition=skill.precondition,
            action_sequence=tuple(mutated),
            postcondition=skill.postcondition,
            expected_reward=skill.expected_reward * 0.8,  # 初始略低
            success_count=0,
            total_count=0,
            abstraction_level=skill.abstraction_level,
        )
        library.add_skill(variant)
        return variant

    def prune_ineffective_skills(self, library: SkillLibrary) -> List[str]:
        """剪枝：删除长期无效的技能。

        Returns:
            被删除的技能名称列表
        """
        pruned = []
        for skill in library.get_all_skills():
            records = [r for r in self._execution_history if r.skill_name == skill.name]
            if len(records) < 5:
                continue
            avg_effectiveness = np.mean([r.effectiveness for r in records])
            if avg_effectiveness < self.min_effectiveness * 0.5:
                # 非常无效，删除
                pruned.append(skill.name)

        # 注意：实际删除需要SkillLibrary支持删除操作
        # 这里只返回建议删除的列表
        return pruned

    def refine_all(self, library: SkillLibrary) -> Tuple[List[Skill], List[str]]:
        """微调技能库中的所有技能。

        Returns:
            (refined_skills, pruned_names)
        """
        refined = []
        for skill in library.get_all_skills():
            result = self.refine_skill(skill, library)
            if result is not None:
                refined.append(result)
        pruned = self.prune_ineffective_skills(library)
        return refined, pruned

    def get_skill_statistics(self, skill_name: str) -> dict:
        """获取技能执行统计。"""
        records = [r for r in self._execution_history if r.skill_name == skill_name]
        if not records:
            return {"count": 0}
        return {
            "count": len(records),
            "avg_effectiveness": float(np.mean([r.effectiveness for r in records])),
            "avg_free_energy_delta": float(np.mean([r.free_energy_delta for r in records])),
            "avg_reward": float(np.mean([r.actual_reward for r in records])),
            "success_rate": sum(1 for r in records if r.success) / len(records),
        }
