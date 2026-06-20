"""阶段B-4：任务分解器 - 将长任务分解为子目标序列。

第一性原理：
    长任务（如"从A到B再到C"）对Agent来说搜索空间巨大。
    任务分解将长任务拆解为一系列子目标，每个子目标可独立学习。

    分解策略：
    1. 空间分解：将路径拆分为多个航点（waypoints）
    2. 技能分解：检查技能库能否覆盖部分子目标
    3. 层次分解：高层规划子目标序列，低层执行单个子目标

    这实现了"自驱完成长任务"：Agent不需要一次性规划整条路径，
    而是逐个攻克子目标，每个子目标都是可学习的短任务。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


# ============================================================
# 子目标状态
# ============================================================
class SubGoalStatus(Enum):
    """子目标状态。"""
    PENDING = "pending"      # 待执行
    ACTIVE = "active"        # 正在执行
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"        # 失败
    SKIPPED = "skipped"      # 已跳过


# ============================================================
# 子目标
# ============================================================
@dataclass(frozen=True)
class SubGoal:
    """一个子目标。

    Attributes:
        goal_id: 子目标ID
        description: 描述
        target_state: 目标状态特征
        reward: 完成奖励
        max_steps: 最大步数
        prerequisites: 前置子目标ID列表
    """
    goal_id: int
    description: str
    target_state: Tuple[int, ...]
    reward: float = 1.0
    max_steps: int = 50
    prerequisites: Tuple[int, ...] = ()


# ============================================================
# 任务
# ============================================================
@dataclass
class Task:
    """一个完整任务。

    Attributes:
        name: 任务名称
        description: 任务描述
        sub_goals: 子目标列表（按执行顺序）
        final_goal: 最终目标状态
        total_reward: 总奖励
    """
    name: str
    description: str
    sub_goals: List[SubGoal] = field(default_factory=list)
    final_goal: Optional[Tuple[int, ...]] = None
    total_reward: float = 0.0

    @property
    def num_sub_goals(self) -> int:
        return len(self.sub_goals)


# ============================================================
# 任务分解器
# ============================================================
class TaskDecomposer:
    """任务分解器：将长任务拆分为子目标序列。

    支持的分解方式：
    1. 路径分解：将起点到终点的路径拆分为航点
    2. 技能引导分解：利用已有技能定义子目标
    3. 自适应分解：根据Agent能力动态调整子目标粒度
    """

    def __init__(self, grid_size: int = 5):
        self.grid_size = grid_size

    def decompose_path_task(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        n_waypoints: int = 2,
        task_name: str = "path_task"
    ) -> Task:
        """将路径任务分解为带航点的子目标序列。

        Args:
            start: 起点坐标
            end: 终点坐标
            n_waypoints: 中间航点数
            task_name: 任务名称

        Returns:
            分解后的任务
        """
        if n_waypoints < 0:
            raise ValueError("航点数必须 >= 0")

        # 在起点和终点之间均匀插入航点
        waypoints = []
        for i in range(1, n_waypoints + 1):
            t = i / (n_waypoints + 1)
            wp = (
                int(start[0] + t * (end[0] - start[0])),
                int(start[1] + t * (end[1] - start[1]))
            )
            # 确保在网格内
            wp = (
                max(0, min(self.grid_size - 1, wp[0])),
                max(0, min(self.grid_size - 1, wp[1]))
            )
            waypoints.append(wp)

        # 构建子目标序列
        sub_goals = []
        all_points = [start] + waypoints + [end]
        for i, point in enumerate(all_points):
            prereq = (i - 1,) if i > 0 else ()
            sub_goal = SubGoal(
                goal_id=i,
                description=f"到达航点{point}",
                target_state=self._encode_position(point),
                reward=0.5 if i < len(all_points) - 1 else 1.0,  # 终点奖励更高
                max_steps=30,
                prerequisites=prereq,
            )
            sub_goals.append(sub_goal)

        return Task(
            name=task_name,
            description=f"从{start}到{end}，经过{len(waypoints)}个航点",
            sub_goals=sub_goals,
            final_goal=self._encode_position(end),
            total_reward=sum(sg.reward for sg in sub_goals),
        )

    def decompose_multi_target_task(
        self,
        targets: List[Tuple[int, int]],
        task_name: str = "multi_target_task"
    ) -> Task:
        """将多目标访问任务分解为子目标序列。

        如：访问A→B→C三个位置
        """
        if len(targets) < 2:
            raise ValueError("至少需要2个目标点")

        sub_goals = []
        for i, target in enumerate(targets):
            prereq = (i - 1,) if i > 0 else ()
            sub_goal = SubGoal(
                goal_id=i,
                description=f"访问目标{target}",
                target_state=self._encode_position(target),
                reward=1.0,
                max_steps=50,
                prerequisites=prereq,
            )
            sub_goals.append(sub_goal)

        return Task(
            name=task_name,
            description=f"依次访问{len(targets)}个目标",
            sub_goals=sub_goals,
            final_goal=self._encode_position(targets[-1]),
            total_reward=sum(sg.reward for sg in sub_goals),
        )

    def decompose_with_skills(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        available_skills: List,  # List[Skill]
        task_name: str = "skill_guided_task"
    ) -> Task:
        """利用已有技能引导任务分解。

        如果技能库中有从某中间点到终点的技能，
        则将该中间点作为子目标。
        """
        # 简化版：检查技能的前置/后置条件是否匹配路径
        # 这里先用简单的路径分解
        task = self.decompose_path_task(start, end, n_waypoints=1, task_name=task_name)

        # 尝试用技能优化子目标
        # 如果有技能的postcondition匹配某个子目标，可以减少步数预算
        for skill in available_skills:
            for sg in task.sub_goals:
                # 简化匹配：如果技能后置条件包含目标位置
                if any(c in skill.postcondition for c in sg.target_state):
                    # 减少该子目标的步数预算（技能可以加速）
                    object.__setattr__(sg, 'max_steps', max(10, sg.max_steps // 2))
                    break

        return task

    def _encode_position(self, pos: Tuple[int, int]) -> Tuple[int, ...]:
        """将位置编码为状态特征。"""
        return (pos[0], pos[1])

    def adaptive_decompose(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        agent_skill_count: int,
        task_name: str = "adaptive_task"
    ) -> Task:
        """自适应分解：根据Agent能力调整粒度。

        技能多 → 粗粒度（少航点，信任Agent能力）
        技能少 → 细粒度（多航点，降低难度）
        """
        if agent_skill_count > 10:
            n_waypoints = 0  # 高能力，直接到终点
        elif agent_skill_count > 3:
            n_waypoints = 1
        else:
            n_waypoints = 2  # 低能力，多航点

        return self.decompose_path_task(
            start, end, n_waypoints=n_waypoints, task_name=task_name
        )
