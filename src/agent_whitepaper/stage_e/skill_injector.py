"""阶段E-2：技能→Q值注入器 - 将抽象技能转化为Q值。

第一性原理：
    技能是"动作序列"，Q表是"状态-动作价值"。
    要让技能真正影响Agent行为，必须将技能转化为Q值注入Q表。

    转化逻辑：
    技能 = (前置状态, 动作序列, 期望奖励)
    →  对序列中每个 (状态_i, 动作_i)，注入 Q值 = 期望奖励 * 折扣因子^i

    这样Agent在遇到匹配的前置状态时，会倾向于执行技能中的动作序列。
    这是"自主增强技能"的真正闭环：抽象→注入→行为改变→验证。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from agent_whitepaper.stage_a.policy import QTable
from agent_whitepaper.stage_b.skill_abstractor import Skill, SkillLibrary


# ============================================================
# 注入结果
# ============================================================
@dataclass(frozen=True)
class InjectionResult:
    """技能注入结果。

    Attributes:
        skill_name: 技能名称
        states_injected: 注入的状态数
        q_values_before: 注入前Q值总和
        q_values_after: 注入后Q值总和
        injection_strength: 注入强度
    """
    skill_name: str
    states_injected: int
    q_values_before: float
    q_values_after: float
    injection_strength: float


# ============================================================
# 技能→Q值注入器
# ============================================================
class SkillQValueInjector:
    """技能→Q值注入器：将技能转化为Q值注入Q表。

    注入策略：
    1. 状态映射：将技能的前置条件映射到Q表状态索引
    2. 动作序列展开：沿技能的动作序列，为每个状态-动作对注入Q值
    3. 价值衰减：序列中越靠后的动作，Q值越小（折扣因子）
    4. 置信度加权：技能成功率越高，注入强度越大
    """

    def __init__(
        self,
        grid_size: int = 5,
        gamma: float = 0.9,
        injection_scale: float = 1.0
    ):
        self.grid_size = grid_size
        self.gamma = gamma
        self.injection_scale = injection_scale

    def _state_features_to_index(self, features: Tuple[int, ...]) -> int:
        """将状态特征转换为Q表状态索引。

        对于GridWorld，特征 = (row, col)，索引 = row * grid_size + col
        """
        if len(features) >= 2:
            row, col = features[0], features[1]
            return row * self.grid_size + col
        return features[0] if features else 0

    def _simulate_action_sequence(
        self,
        start_state: int,
        action_sequence: Tuple[int, ...]
    ) -> List[Tuple[int, int]]:
        """模拟动作序列，生成 (状态, 动作) 列表。

        对于GridWorld：
        - 动作0=上, 1=下, 2=左, 3=右
        - 状态 = row * grid_size + col
        """
        states_actions = []
        current_state = start_state
        row, col = current_state // self.grid_size, current_state % self.grid_size

        for action in action_sequence:
            states_actions.append((current_state, action))
            # 模拟动作执行
            if action == 0:  # 上
                row = max(0, row - 1)
            elif action == 1:  # 下
                row = min(self.grid_size - 1, row + 1)
            elif action == 2:  # 左
                col = max(0, col - 1)
            elif action == 3:  # 右
                col = min(self.grid_size - 1, col + 1)
            current_state = row * self.grid_size + col

        return states_actions

    def inject_skill(
        self,
        skill: Skill,
        q_table: QTable,
        strength: Optional[float] = None
    ) -> InjectionResult:
        """将单个技能注入Q表。

        Args:
            skill: 要注入的技能
            q_table: 目标Q表
            strength: 注入强度（None=用技能成功率）

        Returns:
            注入结果
        """
        # 计算注入强度
        if strength is None:
            strength = skill.success_rate * self.injection_scale

        # 起始状态索引
        start_state = self._state_features_to_index(skill.precondition)
        if start_state >= q_table.n_states:
            return InjectionResult(
                skill_name=skill.name,
                states_injected=0,
                q_values_before=0.0,
                q_values_after=0.0,
                injection_strength=0.0,
            )

        # 模拟动作序列
        states_actions = self._simulate_action_sequence(
            start_state, skill.action_sequence
        )

        # 记录注入前Q值
        q_before = float(sum(
            q_table._table[s, a] for s, a in states_actions
            if s < q_table.n_states and a < q_table.n_actions
        ))

        # 注入Q值（沿动作序列衰减）
        injected_count = 0
        for i, (state, action) in enumerate(states_actions):
            if state >= q_table.n_states or action >= q_table.n_actions:
                continue
            # Q值 = 期望奖励 * γ^i * 强度
            discounted_reward = skill.expected_reward * (self.gamma ** i) * strength
            # 取max(现有值, 注入值) 避免覆盖更好的已有知识
            current_q = q_table._table[state, action]
            q_table._table[state, action] = max(current_q, discounted_reward)
            injected_count += 1

        # 记录注入后Q值
        q_after = float(sum(
            q_table._table[s, a] for s, a in states_actions
            if s < q_table.n_states and a < q_table.n_actions
        ))

        return InjectionResult(
            skill_name=skill.name,
            states_injected=injected_count,
            q_values_before=q_before,
            q_values_after=q_after,
            injection_strength=strength,
        )

    def inject_all_skills(
        self,
        skill_library: SkillLibrary,
        q_table: QTable,
        min_success_rate: float = 0.3,
        max_skills: int = 50
    ) -> List[InjectionResult]:
        """将技能库中的所有有效技能注入Q表。

        Args:
            skill_library: 技能库
            q_table: 目标Q表
            min_success_rate: 最低成功率阈值
            max_skills: 最大注入技能数

        Returns:
            所有注入结果
        """
        results = []
        # 按成功率*期望奖励排序，优先注入最好的技能
        skills = sorted(
            skill_library.get_all_skills(),
            key=lambda s: s.success_rate * s.expected_reward,
            reverse=True
        )

        for skill in skills:
            if len(results) >= max_skills:
                break
            if skill.success_rate < min_success_rate:
                continue
            if skill.total_count < 2:  # 经验不足的技能不注入
                continue
            result = self.inject_skill(skill, q_table)
            results.append(result)

        return results

    def inject_from_shared_library(
        self,
        shared_library,  # SharedSkillLibrary
        q_table: QTable,
        max_skills: int = 20
    ) -> List[InjectionResult]:
        """从共享技能库注入技能。

        优先注入已验证的高质量技能。
        """
        results = []
        # 获取已验证技能
        verified = shared_library.get_verified_skills()
        if not verified:
            # 没有已验证的，用top技能
            verified = shared_library.get_top_skills(max_skills)

        for shared_skill in verified[:max_skills]:
            result = self.inject_skill(shared_skill.skill, q_table)
            results.append(result)

        return results

    def evaluate_injection_impact(
        self,
        q_table: QTable,
        injection_results: List[InjectionResult]
    ) -> Dict:
        """评估注入影响。"""
        total_before = sum(r.q_values_before for r in injection_results)
        total_after = sum(r.q_values_after for r in injection_results)
        total_states = sum(r.states_injected for r in injection_results)
        return {
            "total_skills_injected": len(injection_results),
            "total_states_affected": total_states,
            "q_value_change": total_after - total_before,
            "avg_injection_strength": (
                sum(r.injection_strength for r in injection_results)
                / len(injection_results)
            ) if injection_results else 0.0,
        }
