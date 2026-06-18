"""阶段25：驱力到目标的映射——从需求到行动的桥梁。

本阶段实现驱力到目标的映射机制，将获胜驱力转化为
可执行的目标描述。
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np

from .stage_024 import DriveCompetitionResult


@dataclass(frozen=True)
class Goal:
    """目标的完整描述。

    组合阶段24的DriveCompetitionResult，在竞争结果的基础上增加
    目标描述、目标类型和优先级。

    Attributes:
        competition_result: 阶段24的竞争结果（组合字段）
        goal_description: 目标描述字符串
        goal_type: 目标类型 ('safety' | 'social' | 'exploration')
        priority: 目标优先级 ∈ [0, 1]

    Invariants:
        - goal_type 与 competition_result.winner 一致
        - 0 ≤ priority ≤ 1
        - goal_description 非空
    """
    competition_result: DriveCompetitionResult
    goal_description: str
    goal_type: str
    priority: float

    def __post_init__(self):
        valid_types = {'safety', 'social', 'exploration'}
        if self.goal_type not in valid_types:
            raise ValueError(
                f"目标类型必须是 {valid_types} 之一，"
                f"收到 '{self.goal_type}'"
            )
        if self.goal_type != self.competition_result.winner:
            raise ValueError(
                f"目标类型 '{self.goal_type}' 与获胜驱力 "
                f"'{self.competition_result.winner}' 不一致"
            )
        if not (0 <= self.priority <= 1):
            raise ValueError(
                f"优先级必须 ∈ [0,1]，收到 {self.priority}"
            )
        if not self.goal_description or not self.goal_description.strip():
            raise ValueError("目标描述不能为空")


class GoalMapper:
    """阶段25的核心计算接口。

    实现从驱力竞争结果到目标的映射。
    核心约束：
        - 目标类型与获胜驱力一致（A25.1）
        - 目标优先级 = 获胜驱力紧急度（A25.2）
    """

    GOAL_TEMPLATES: Dict[str, str] = {
        'safety': '确保系统安全运行，最小化预测误差',
        'social': '优化表征稀疏度，提升信息编码效率',
        'exploration': '加速感知收敛，探索新的表征空间',
    }

    def map_goal(
        self,
        competition_result: DriveCompetitionResult
    ) -> Goal:
        """将驱力竞争结果映射为目标（A25.1-A25.2）。

        Args:
            competition_result: 阶段24的竞争结果

        Returns:
            Goal 实例

        Raises:
            ValueError: 若 competition_result 为 None
        """
        if competition_result is None:
            raise ValueError("competition_result 不能为空")

        winner = competition_result.winner
        # 获取获胜驱力的紧急度作为优先级
        winner_priority = competition_result.priorities[0].urgency

        description = self.GOAL_TEMPLATES.get(
            winner,
            f'执行 {winner} 相关目标'
        )

        return Goal(
            competition_result=competition_result,
            goal_description=description,
            goal_type=winner,
            priority=winner_priority,
        )

    def get_goal_for_drive(
        self,
        drive_name: str,
        competition_result: DriveCompetitionResult
    ) -> Goal:
        """为指定驱力构造目标（不依赖获胜者）。

        Args:
            drive_name: 驱力名称
            competition_result: 竞争结果

        Returns:
            Goal 实例
        """
        if competition_result is None:
            raise ValueError("competition_result 不能为空")
        valid_names = {'safety', 'social', 'exploration'}
        if drive_name not in valid_names:
            raise ValueError(
                f"驱力名称必须是 {valid_names} 之一，"
                f"收到 '{drive_name}'"
            )

        # 找到对应驱力的优先级
        priority = 0.0
        for p in competition_result.priorities:
            if p.drive_name == drive_name:
                priority = p.urgency
                break

        description = self.GOAL_TEMPLATES.get(
            drive_name,
            f'执行 {drive_name} 相关目标'
        )

        return Goal(
            competition_result=competition_result,
            goal_description=description,
            goal_type=drive_name,
            priority=priority,
        )
