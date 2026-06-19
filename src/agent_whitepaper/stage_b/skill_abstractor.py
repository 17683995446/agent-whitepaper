"""阶段B-1：技能抽象器 - 从成功情节中抽象出可复用技能。

第一性原理：
    技能是从具体情节中抽象出的"动作模式"。
    一个技能 = (前置条件, 动作序列, 后置条件, 期望收益)

    抽象过程：
    1. 从成功情节中提取频繁出现的动作子序列
    2. 用n-gram统计找出高频模式
    3. 将高频模式封装为技能，记录其前置/后置状态

    技能让Agent"站在巨人的肩膀上"：
    - 不需要每次从头探索
    - 可以组合已有技能解决新问题
    - 实现了"自主增强技能"的目标
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from collections import Counter
import numpy as np

from agent_whitepaper.stage_a.memory import Episode
from agent_whitepaper.stage_a.environment import Transition


# ============================================================
# 技能数据类
# ============================================================
@dataclass(frozen=True)
class Skill:
    """一个可复用技能。

    Attributes:
        name: 技能名称
        precondition: 前置状态特征（技能可执行的状态描述）
        action_sequence: 动作序列
        postcondition: 后置状态特征（技能执行后的预期状态）
        expected_reward: 期望收益
        success_count: 成功执行次数
        total_count: 总执行次数
        abstraction_level: 抽象层级（0=原始动作，1=组合技能，2=元技能）
    """
    name: str
    precondition: Tuple[int, ...]  # 状态特征哈希
    action_sequence: Tuple[int, ...]
    postcondition: Tuple[int, ...]
    expected_reward: float
    success_count: int = 0
    total_count: int = 0
    abstraction_level: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count

    @property
    def length(self) -> int:
        return len(self.action_sequence)

    def matches_precondition(self, state_features: Tuple[int, ...]) -> bool:
        """检查当前状态是否满足技能前置条件。"""
        return state_features == self.precondition


# ============================================================
# 技能库
# ============================================================
class SkillLibrary:
    """技能库：存储和管理所有学到的技能。

    支持按前置条件检索、按成功率排序、技能组合。
    """

    def __init__(self):
        self._skills: Dict[str, Skill] = {}
        self._index_by_precondition: Dict[Tuple[int, ...], List[str]] = {}

    def __len__(self) -> int:
        return len(self._skills)

    def add_skill(self, skill: Skill) -> None:
        """添加或更新技能。"""
        if skill.name in self._skills:
            # 更新已有技能的统计
            existing = self._skills[skill.name]
            updated = Skill(
                name=existing.name,
                precondition=existing.precondition,
                action_sequence=existing.action_sequence,
                postcondition=existing.postcondition,
                expected_reward=(
                    existing.expected_reward * existing.total_count
                    + skill.expected_reward
                ) / max(existing.total_count + 1, 1),
                success_count=existing.success_count + skill.success_count,
                total_count=existing.total_count + skill.total_count,
                abstraction_level=max(existing.abstraction_level, skill.abstraction_level),
            )
            self._skills[skill.name] = updated
        else:
            self._skills[skill.name] = skill
            # 更新索引
            if skill.precondition not in self._index_by_precondition:
                self._index_by_precondition[skill.precondition] = []
            self._index_by_precondition[skill.precondition].append(skill.name)

    def find_by_precondition(self, state_features: Tuple[int, ...]) -> List[Skill]:
        """根据前置条件查找可用技能。"""
        names = self._index_by_precondition.get(state_features, [])
        return [self._skills[n] for n in names]

    def find_best_skill(self, state_features: Tuple[int, ...]) -> Optional[Skill]:
        """找到最适合当前状态的技能（成功率最高）。"""
        candidates = self.find_by_precondition(state_features)
        if not candidates:
            return None
        # 按成功率*期望收益排序
        return max(candidates, key=lambda s: s.success_rate * s.expected_reward)

    def get_all_skills(self) -> List[Skill]:
        """获取所有技能。"""
        return list(self._skills.values())

    def get_top_skills(self, n: int) -> List[Skill]:
        """获取表现最好的n个技能。"""
        sorted_skills = sorted(
            self._skills.values(),
            key=lambda s: s.success_rate * s.expected_reward,
            reverse=True
        )
        return sorted_skills[:n]


# ============================================================
# 技能抽象器
# ============================================================
class SkillAbstractor:
    """从成功情节中抽象技能。

    算法：
    1. 收集所有成功情节
    2. 提取动作序列的n-gram（n=2~5）
    3. 统计高频n-gram
    4. 将高频n-gram封装为技能
    5. 记录前置/后置状态
    """

    def __init__(
        self,
        min_n: int = 2,
        max_n: int = 5,
        min_frequency: int = 3,
        min_success_rate: float = 0.5
    ):
        if min_n < 1 or max_n < min_n:
            raise ValueError("必须 1 <= min_n <= max_n")
        if min_frequency < 1:
            raise ValueError("min_frequency 必须 >= 1")
        if not (0 < min_success_rate <= 1):
            raise ValueError("min_success_rate 必须 ∈ (0,1]")
        self.min_n = min_n
        self.max_n = max_n
        self.min_frequency = min_frequency
        self.min_success_rate = min_success_rate

    def _extract_state_features(self, observation: np.ndarray) -> Tuple[int, ...]:
        """从观测中提取状态特征（简化为离散哈希）。"""
        # 将连续观测离散化：每个维度分箱
        discretized = tuple(int(v * 10) for v in observation[:5])  # 取前5维
        return discretized

    def _extract_ngrams(
        self,
        action_sequence: List[int]
    ) -> List[Tuple[int, Tuple[int, ...]]]:
        """提取所有n-gram及其位置。

        Returns:
            [(n, gram), ...]
        """
        ngrams = []
        for n in range(self.min_n, min(self.max_n + 1, len(action_sequence) + 1)):
            for i in range(len(action_sequence) - n + 1):
                gram = tuple(action_sequence[i:i + n])
                ngrams.append((n, gram))
        return ngrams

    def abstract_from_episodes(
        self,
        episodes: List[Episode],
        skill_library: SkillLibrary
    ) -> List[Skill]:
        """从情节列表中抽象技能。

        Args:
            episodes: 情节列表（优先使用成功情节）
            skill_library: 技能库（新技能将添加到此）

        Returns:
            新抽象出的技能列表
        """
        # 只从成功情节中学习
        successful_episodes = [ep for ep in episodes if ep.success]
        if not successful_episodes:
            return []

        # 统计所有n-gram
        ngram_stats: Dict[Tuple[int, ...], List[Tuple[Episode, int, int]]] = {}
        # gram -> [(episode, start_idx, end_idx)]

        for ep in successful_episodes:
            actions = [t.action for t in ep.transitions]
            for n, gram in self._extract_ngrams(actions):
                if gram not in ngram_stats:
                    ngram_stats[gram] = []
                # 找到这个gram在序列中的位置
                for i in range(len(actions) - len(gram) + 1):
                    if tuple(actions[i:i + len(gram)]) == gram:
                        ngram_stats[gram].append((ep, i, i + len(gram)))
                        break

        # 筛选高频n-gram
        new_skills = []
        for gram, occurrences in ngram_stats.items():
            if len(occurrences) < self.min_frequency:
                continue

            # 计算这个gram的平均奖励
            rewards = []
            preconditions = set()
            postconditions = set()
            for ep, start, end in occurrences:
                # gram对应的奖励
                segment_reward = sum(
                    t.reward for t in ep.transitions[start:end]
                )
                rewards.append(segment_reward)
                # 前置状态
                pre = self._extract_state_features(ep.transitions[start].observation)
                preconditions.add(pre)
                # 后置状态
                post = self._extract_state_features(
                    ep.transitions[end - 1].next_observation
                )
                postconditions.add(post)

            avg_reward = float(np.mean(rewards))
            success_rate = len(occurrences) / len(successful_episodes)

            if success_rate < self.min_success_rate:
                continue

            # 为每个不同的前置条件创建技能
            for pre in preconditions:
                skill_name = f"skill_{len(gram)}gram_{'-'.join(map(str, gram))}_{hash(pre) % 10000}"
                # 取对应的后置条件（简化：取第一个匹配的）
                matching_post = pre  # 默认回退
                for ep, s, e in occurrences:
                    if s < len(ep.transitions) and e - 1 < len(ep.transitions):
                        if self._extract_state_features(ep.transitions[s].observation) == pre:
                            matching_post = self._extract_state_features(
                                ep.transitions[e - 1].next_observation
                            )
                            break
                skill = Skill(
                    name=skill_name,
                    precondition=pre,
                    action_sequence=gram,
                    postcondition=matching_post,
                    expected_reward=avg_reward,
                    success_count=len(occurrences),
                    total_count=len(occurrences),
                    abstraction_level=1,
                )
                skill_library.add_skill(skill)
                new_skills.append(skill)

        return new_skills

    def abstract_composite_skills(
        self,
        skill_library: SkillLibrary,
        existing_skills: List[Skill]
    ) -> List[Skill]:
        """从已有技能中组合出更高层级的技能。

        如果技能A的后置条件 == 技能B的前置条件，
        且A+B频繁共现，则创建组合技能 A→B。
        """
        if len(existing_skills) < 2:
            return []

        composite_skills = []
        # 寻找可链接的技能对
        for skill_a in existing_skills:
            for skill_b in existing_skills:
                if skill_a.name == skill_b.name:
                    continue
                if skill_a.postcondition == skill_b.precondition:
                    # 可以组合
                    composite_name = f"composite_{skill_a.name}_+_{skill_b.name}"
                    if composite_name in [s.name for s in skill_library.get_all_skills()]:
                        continue
                    composite = Skill(
                        name=composite_name,
                        precondition=skill_a.precondition,
                        action_sequence=skill_a.action_sequence + skill_b.action_sequence,
                        postcondition=skill_b.postcondition,
                        expected_reward=skill_a.expected_reward + skill_b.expected_reward,
                        success_count=min(skill_a.success_count, skill_b.success_count),
                        total_count=min(skill_a.total_count, skill_b.total_count),
                        abstraction_level=2,
                    )
                    skill_library.add_skill(composite)
                    composite_skills.append(composite)

        return composite_skills
