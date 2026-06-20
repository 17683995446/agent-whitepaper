"""阶段C-4：联盟形成与任务分配 - 多Agent协作执行任务。

第一性原理：
    复杂任务需要多个Agent协作。联盟形成是：
    1. 任务到达 → 分析需要哪些能力
    2. 从注册中心找到具备能力的Agent
    3. 形成临时联盟（ coalition ）
    4. 将任务子目标分配给联盟成员
    5. 协调执行，汇总结果

    这实现了"操作多个agent完成任务"的核心：
    系统自动组建团队、分配任务、协调执行。
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from agent_whitepaper.stage_b.task_decomposer import Task

from .message_bus import MessageBus, MessageType
from .registry import AgentRegistry
from .role_assigner import Role, RoleAssigner, RoleAssignment, TeamConfiguration


# ============================================================
# 联盟状态
# ============================================================
class CoalitionStatus(Enum):
    """联盟状态。"""
    FORMING = "forming"      # 组建中
    ACTIVE = "active"        # 活跃中
    COMPLETED = "completed"  # 已完成
    DISBANDED = "disbanded"  # 已解散
    FAILED = "failed"        # 失败


# ============================================================
# 联盟
# ============================================================
@dataclass
class Coalition:
    """一个Agent联盟。

    Attributes:
        coalition_id: 联盟ID
        task: 关联的任务
        members: 成员Agent ID列表
        role_assignments: 角色分配
        status: 联盟状态
        created_time: 创建时间
        subgoal_assignments: 子目标到Agent的分配
    """
    coalition_id: str
    task: Task
    members: List[str] = field(default_factory=list)
    role_assignments: List[RoleAssignment] = field(default_factory=list)
    status: CoalitionStatus = CoalitionStatus.FORMING
    created_time: float = field(default_factory=time.time)
    subgoal_assignments: Dict[int, str] = field(default_factory=dict)  # goal_id -> agent_id
    progress: float = 0.0  # 完成进度 [0,1]

    @property
    def size(self) -> int:
        return len(self.members)


# ============================================================
# 联盟形成器
# ============================================================
class CoalitionFormer:
    """联盟形成器：根据任务需求组建Agent联盟。

    工作流程：
    1. 分析任务需求（需要哪些角色/能力）
    2. 从注册中心筛选候选Agent
    3. 用角色分配器分配角色
    4. 将子目标分配给各成员
    5. 通过消息总线通知各成员
    """

    def __init__(
        self,
        registry: AgentRegistry,
        message_bus: MessageBus,
        role_assigner: Optional[RoleAssigner] = None
    ) -> None:
        self.registry = registry
        self.message_bus = message_bus
        self.role_assigner = role_assigner or RoleAssigner(registry)
        self._coalitions: Dict[str, Coalition] = {}
        self._coalition_counter = 0

    def analyze_task_requirements(self, task: Task) -> TeamConfiguration:
        """分析任务需求，确定需要哪些角色。

        简化规则：
        - 子目标数 > 3 → 需要协调者
        - 总是有探索者
        - 子目标数 >= 2 → 需要守护者
        """
        n_subgoals = task.num_sub_goals
        required_roles = {}

        # 总是需要探索者
        required_roles[Role.EXPLORER] = min(n_subgoals, 2)

        # 子目标多时需要协调者
        if n_subgoals > 2:
            required_roles[Role.COORDINATOR] = 1

        # 需要守护者
        if n_subgoals >= 2:
            required_roles[Role.GUARDIAN] = 1

        return TeamConfiguration(
            required_roles=required_roles,
            team_name=task.name
        )

    def form_coalition(
        self,
        task: Task,
        candidate_agent_ids: Optional[List[str]] = None
    ) -> Optional[Coalition]:
        """为任务组建联盟。

        Args:
            task: 需要执行的任务
            candidate_agent_ids: 候选Agent列表

        Returns:
            组建的联盟，如果Agent不足则返回None
        """
        # 分析任务需求
        config = self.analyze_task_requirements(task)

        # 分配角色
        assignments = self.role_assigner.assign_roles(config, candidate_agent_ids)

        if not assignments:
            return None

        # 检查是否满足最小需求
        if len(assignments) < min(config.total_required, 1):
            return None

        # 创建联盟
        self._coalition_counter += 1
        coalition_id = f"coalition_{self._coalition_counter:04d}"
        members = [a.agent_id for a in assignments]

        coalition = Coalition(
            coalition_id=coalition_id,
            task=task,
            members=members,
            role_assignments=assignments,
            status=CoalitionStatus.ACTIVE,
        )

        # 分配子目标给成员
        self._assign_subgoals(coalition)

        # 通知成员
        self._notify_members(coalition)

        self._coalitions[coalition_id] = coalition
        return coalition

    def _assign_subgoals(self, coalition: Coalition) -> None:
        """将子目标分配给联盟成员。

        分配策略：
        - 探索者负责前面的子目标
        - 守护者负责中间的子目标
        - 协调者负责最后的子目标（汇总）
        - 按角色能力分配
        """
        task = coalition.task
        members_by_role: Dict[Role, List[str]] = {}
        for assignment in coalition.role_assignments:
            if assignment.role not in members_by_role:
                members_by_role[assignment.role] = []
            members_by_role[assignment.role].append(assignment.agent_id)

        # 简化分配：轮流分配给所有成员
        member_cycle = iter(coalition.members * (len(task.sub_goals) // len(coalition.members) + 1))

        for subgoal in task.sub_goals:
            try:
                agent_id = next(member_cycle)
                coalition.subgoal_assignments[subgoal.goal_id] = agent_id
            except StopIteration:
                # 如果成员不够，分配给第一个成员
                if coalition.members:
                    coalition.subgoal_assignments[subgoal.goal_id] = coalition.members[0]

    def _notify_members(self, coalition: Coalition) -> None:
        """通知联盟成员。"""
        for assignment in coalition.role_assignments:
            msg = self.message_bus.create_message(
                sender_id="system",
                receiver_id=assignment.agent_id,
                message_type=MessageType.COORDINATION,
                content={
                    "coalition_id": coalition.coalition_id,
                    "role": assignment.role.value,
                    "task_name": coalition.task.name,
                    "assigned_subgoals": [
                        gid for gid, aid in coalition.subgoal_assignments.items()
                        if aid == assignment.agent_id
                    ],
                },
                priority=1,
            )
            self.message_bus.send(msg)

    def update_progress(
        self,
        coalition_id: str,
        completed_subgoals: int,
        total_subgoals: int
    ) -> bool:
        """更新联盟进度。"""
        if coalition_id not in self._coalitions:
            return False
        coalition = self._coalitions[coalition_id]
        coalition.progress = completed_subgoals / max(total_subgoals, 1)
        if coalition.progress >= 1.0:
            coalition.status = CoalitionStatus.COMPLETED
        return True

    def disband_coalition(self, coalition_id: str) -> bool:
        """解散联盟。"""
        if coalition_id not in self._coalitions:
            return False
        coalition = self._coalitions[coalition_id]
        coalition.status = CoalitionStatus.DISBANDED
        # 通知成员解散
        for member_id in coalition.members:
            msg = self.message_bus.create_message(
                sender_id="system",
                receiver_id=member_id,
                message_type=MessageType.NOTIFICATION,
                content={"coalition_id": coalition_id, "action": "disband"},
            )
            self.message_bus.send(msg)
        return True

    def get_coalition(self, coalition_id: str) -> Optional[Coalition]:
        """获取联盟信息。"""
        return self._coalitions.get(coalition_id)

    def get_active_coalitions(self) -> List[Coalition]:
        """获取所有活跃联盟。"""
        return [
            c for c in self._coalitions.values()
            if c.status == CoalitionStatus.ACTIVE
        ]

    def get_coalition_summary(self) -> dict:
        """获取联盟摘要。"""
        status_counts = {}
        for c in self._coalitions.values():
            status_counts[c.status.value] = status_counts.get(c.status.value, 0) + 1
        return {
            "total_coalitions": len(self._coalitions),
            "status_counts": status_counts,
            "active_coalitions": len(self.get_active_coalitions()),
        }
