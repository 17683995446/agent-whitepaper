"""阶段26：目标分解——将高层目标拆解为可执行子目标。

本阶段实现目标分解机制，将高层目标分解为多个可执行的子目标。
"""

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np

from .stage_025 import Goal


@dataclass(frozen=True)
class SubGoal:
    """子目标的完整描述。

    Attributes:
        parent_goal: 父目标（组合字段）
        subgoal_description: 子目标描述
        subgoal_index: 子目标索引
        weight: 子目标权重 ∈ [0, 1]

    Invariants:
        - weight ∈ [0, 1]
        - subgoal_description 非空
        - subgoal_index ≥ 0
    """
    parent_goal: Goal
    subgoal_description: str
    subgoal_index: int
    weight: float

    def __post_init__(self):
        if not (0 <= self.weight <= 1):
            raise ValueError(
                f"子目标权重必须 ∈ [0,1]，收到 {self.weight}"
            )
        if not self.subgoal_description or not self.subgoal_description.strip():
            raise ValueError("子目标描述不能为空")
        if self.subgoal_index < 0:
            raise ValueError(
                f"子目标索引必须 ≥ 0，收到 {self.subgoal_index}"
            )


@dataclass(frozen=True)
class GoalDecomposition:
    """目标分解的完整结果。

    组合阶段25的Goal，在目标的基础上增加子目标列表。

    Attributes:
        parent_goal: 父目标（组合字段）
        subgoals: 子目标列表
        total_weight: 总权重

    Invariants:
        - 所有子目标的 parent_goal 一致
        - total_weight = sum(subgoal.weight)
        - 0 ≤ total_weight ≤ len(subgoals)
    """
    parent_goal: Goal
    subgoals: Tuple[SubGoal, ...]
    total_weight: float

    def __post_init__(self):
        if len(self.subgoals) == 0:
            raise ValueError("子目标列表不能为空")
        expected_weight = sum(sg.weight for sg in self.subgoals)
        if abs(self.total_weight - expected_weight) > 1e-6:
            raise ValueError(
                f"总权重 {self.total_weight} 与计算结果 {expected_weight} 不一致"
            )


class GoalDecomposer:
    """阶段26的核心计算接口。

    实现目标分解机制，将高层目标分解为可执行子目标。
    核心约束：
        - 子目标列表非空（A26.1）
        - 子目标权重和 = 1.0（A26.2，可选归一化）
    """

    SUBGOAL_TEMPLATES: dict[str, List[str]] = {
        'safety': [
            '监控输入数据质量',
            '检测异常预测误差',
            '触发安全保护机制',
        ],
        'social': [
            '评估当前表征稀疏度',
            '调整网络结构优化稀疏性',
            '验证稀疏度提升效果',
        ],
        'exploration': [
            '分析当前收敛速度',
            '尝试新的感知初始化策略',
            '评估收敛性能改善',
        ],
    }

    def decompose(
        self,
        goal: Goal
    ) -> GoalDecomposition:
        """将目标分解为子目标（A26.1-A26.2）。

        Args:
            goal: 阶段25的目标

        Returns:
            GoalDecomposition 实例

        Raises:
            ValueError: 若 goal 为 None
        """
        if goal is None:
            raise ValueError("goal 不能为空")

        templates = self.SUBGOAL_TEMPLATES.get(
            goal.goal_type,
            ['执行默认子目标1', '执行默认子目标2']
        )

        n = len(templates)
        # 均等权重
        weight = 1.0 / n

        subgoals: List[SubGoal] = []
        for i, desc in enumerate(templates):
            subgoals.append(SubGoal(
                parent_goal=goal,
                subgoal_description=desc,
                subgoal_index=i,
                weight=weight,
            ))

        total_weight = sum(sg.weight for sg in subgoals)

        return GoalDecomposition(
            parent_goal=goal,
            subgoals=tuple(subgoals),
            total_weight=total_weight,
        )

    def decompose_with_weights(
        self,
        goal: Goal,
        weights: Tuple[float, ...]
    ) -> GoalDecomposition:
        """使用自定义权重分解目标。

        Args:
            goal: 目标
            weights: 自定义权重元组

        Returns:
            GoalDecomposition 实例
        """
        if goal is None:
            raise ValueError("goal 不能为空")
        if len(weights) == 0:
            raise ValueError("权重不能为空")
        for i, w in enumerate(weights):
            if not (0 <= w <= 1):
                raise ValueError(
                    f"权重[{i}] 必须 ∈ [0,1]，收到 {w}"
                )

        templates = self.SUBGOAL_TEMPLATES.get(
            goal.goal_type,
            [f'执行默认子目标{i+1}' for i in range(len(weights))]
        )

        n = min(len(templates), len(weights))
        subgoals: List[SubGoal] = []
        for i in range(n):
            subgoals.append(SubGoal(
                parent_goal=goal,
                subgoal_description=templates[i],
                subgoal_index=i,
                weight=weights[i],
            ))

        total_weight = sum(sg.weight for sg in subgoals)

        return GoalDecomposition(
            parent_goal=goal,
            subgoals=tuple(subgoals),
            total_weight=total_weight,
        )
