"""阶段C-3：角色分配 - 基于驱力互补的多Agent角色协调。

第一性原理：
    多Agent协作中，不同Agent应承担不同角色：
    - 探索者（高探索驱力）：负责侦察未知区域
    - 守护者（高安全驱力）：负责保护已获取资源
    - 协调者（高社会驱力）：负责通信和协调

    角色分配基于驱力互补：
    - 如果团队缺乏探索者，分配探索驱力高的Agent
    - 如果团队缺乏守护者，分配安全驱力高的Agent
    - 目标是团队整体驱力均衡，覆盖所有必要角色

    这实现了"自主学会操作多个agent"：系统根据各Agent的驱力特征，
    自动分配最适合的角色，无需人工指定。
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum
import numpy as np

from .registry import AgentRegistry, AgentRegistration, AgentCapability


# ============================================================
# 角色定义
# ============================================================
class Role(Enum):
    """Agent角色。"""
    EXPLORER = "explorer"    # 探索者：高探索驱力
    GUARDIAN = "guardian"    # 守护者：高安全驱力
    COORDINATOR = "coordinator"  # 协调者：高社会驱力
    WORKER = "worker"        # 工作者：均衡驱力
    LEARNER = "learner"      # 学习者：高探索+高安全


# 角色与驱力的理想匹配
ROLE_DRIVE_PROFILES: Dict[Role, Tuple[float, float, float]] = {
    # (safety, social, exploration)
    Role.EXPLORER: (0.2, 0.4, 0.9),   # 低安全，高探索
    Role.GUARDIAN: (0.9, 0.5, 0.2),   # 高安全，低探索
    Role.COORDINATOR: (0.5, 0.9, 0.4), # 高社会
    Role.WORKER: (0.6, 0.5, 0.5),     # 均衡
    Role.LEARNER: (0.7, 0.4, 0.8),    # 高探索+高安全
}


# ============================================================
# 角色分配结果
# ============================================================
@dataclass(frozen=True)
class RoleAssignment:
    """角色分配结果。

    Attributes:
        agent_id: Agent ID
        role: 分配的角色
        match_score: 匹配度 [0,1]
        reason: 分配原因
    """
    agent_id: str
    role: Role
    match_score: float
    reason: str


# ============================================================
# 团队配置
# ============================================================
@dataclass
class TeamConfiguration:
    """团队配置：定义团队需要哪些角色及数量。

    Attributes:
        required_roles: 角色需求 {role: count}
        team_name: 团队名称
    """
    required_roles: Dict[Role, int] = field(default_factory=dict)
    team_name: str = "default_team"

    @property
    def total_required(self) -> int:
        return sum(self.required_roles.values())

    @classmethod
    def balanced_team(cls, n_agents: int) -> "TeamConfiguration":
        """创建均衡团队配置。"""
        if n_agents < 1:
            raise ValueError("n_agents 必须 >= 1")
        roles = {}
        if n_agents >= 1:
            roles[Role.EXPLORER] = max(1, n_agents // 3)
        if n_agents >= 2:
            roles[Role.GUARDIAN] = max(1, n_agents // 3)
        if n_agents >= 3:
            roles[Role.COORDINATOR] = max(1, n_agents - sum(roles.values()))
        return cls(required_roles=roles, team_name="balanced_team")

    @classmethod
    def exploration_team(cls, n_agents: int) -> "TeamConfiguration":
        """创建探索型团队（多探索者）。"""
        roles = {
            Role.EXPLORER: max(1, n_agents - 1),
            Role.COORDINATOR: 1 if n_agents > 1 else 0,
        }
        return cls(required_roles=roles, team_name="exploration_team")


# ============================================================
# 角色分配器
# ============================================================
class RoleAssigner:
    """角色分配器：基于驱力互补分配角色。

    算法：
    1. 获取所有活跃Agent的驱力特征
    2. 对每个需要的角色，计算每个Agent的匹配度
    3. 用贪心算法分配：每次选择最佳匹配的Agent
    4. 已分配的Agent不再参与后续分配
    """

    def __init__(self, registry: AgentRegistry):
        self.registry = registry

    def compute_match_score(
        self,
        agent_drives: Tuple[float, float, float],
        role: Role
    ) -> float:
        """计算Agent驱力与角色的匹配度。

        匹配度 = 1 - normalized_distance(drives, role_profile)
        """
        role_profile = ROLE_DRIVE_PROFILES[role]
        # 欧氏距离
        distance = np.sqrt(sum(
            (a - b) ** 2 for a, b in zip(agent_drives, role_profile)
        ))
        # 归一化到 [0,1]（最大距离 = sqrt(3) ≈ 1.732）
        max_distance = np.sqrt(3)
        normalized = distance / max_distance
        return float(1.0 - normalized)

    def assign_roles(
        self,
        config: TeamConfiguration,
        candidate_agent_ids: Optional[List[str]] = None
    ) -> List[RoleAssignment]:
        """为团队分配角色。

        Args:
            config: 团队配置
            candidate_agent_ids: 候选Agent列表（None则用所有活跃Agent）

        Returns:
            角色分配列表
        """
        # 获取候选Agent
        if candidate_agent_ids is None:
            candidates = self.registry.get_active_agents()
        else:
            candidates = [
                self.registry.get_agent(aid) for aid in candidate_agent_ids
                if self.registry.get_agent(aid) is not None
            ]

        if not candidates:
            return []

        # 计算所有 (agent, role) 的匹配度
        assignments = []
        assigned_agents = set()

        # 按角色需求分配
        for role, count in config.required_roles.items():
            for _ in range(count):
                if not candidates:
                    break

                # 计算每个未分配Agent的匹配度
                best_agent = None
                best_score = -1
                for reg in candidates:
                    if reg.agent_id in assigned_agents:
                        continue
                    score = self.compute_match_score(
                        reg.capability.drive_profile, role
                    )
                    if score > best_score:
                        best_score = score
                        best_agent = reg

                if best_agent is None:
                    break

                # 确定分配原因
                drives = best_agent.capability.drive_profile
                if role == Role.EXPLORER and drives[2] > 0.6:
                    reason = "高探索驱力，适合侦察"
                elif role == Role.GUARDIAN and drives[0] > 0.6:
                    reason = "高安全驱力，适合守护"
                elif role == Role.COORDINATOR and drives[1] > 0.6:
                    reason = "高社会驱力，适合协调"
                else:
                    reason = f"匹配度 {best_score:.2f}"

                assignments.append(RoleAssignment(
                    agent_id=best_agent.agent_id,
                    role=role,
                    match_score=best_score,
                    reason=reason,
                ))
                assigned_agents.add(best_agent.agent_id)

        return assignments

    def suggest_role_for_agent(self, agent_id: str) -> Optional[Role]:
        """为单个Agent建议最佳角色。"""
        reg = self.registry.get_agent(agent_id)
        if reg is None:
            return None

        drives = reg.capability.drive_profile
        best_role = None
        best_score = -1
        for role in Role:
            score = self.compute_match_score(drives, role)
            if score > best_score:
                best_score = score
                best_role = role
        return best_role

    def get_team_composition(
        self,
        assignments: List[RoleAssignment]
    ) -> Dict[str, int]:
        """获取团队组成统计。"""
        composition = {}
        for assignment in assignments:
            role_name = assignment.role.value
            composition[role_name] = composition.get(role_name, 0) + 1
        return composition

    def evaluate_team_balance(
        self,
        assignments: List[RoleAssignment]
    ) -> float:
        """评估团队平衡度 [0,1]。

        平衡度 = 1 - 角色分布的方差
        """
        if not assignments:
            return 0.0
        composition = self.get_team_composition(assignments)
        counts = list(composition.values())
        if len(counts) <= 1:
            return 1.0
        # 归一化方差
        mean_count = np.mean(counts)
        variance = np.var(counts)
        max_variance = mean_count ** 2  # 最大可能方差
        if max_variance == 0:
            return 1.0
        balance = 1.0 - (variance / max_variance)
        return float(np.clip(balance, 0, 1))
