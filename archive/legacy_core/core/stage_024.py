"""阶段24：驱力竞争与优先级排序——多需求冲突的数学解决。

本阶段实现驱力竞争机制，将多个驱力按紧急度排序，
决定执行优先级。
"""

from dataclasses import dataclass
from typing import Sequence, Tuple, List, NamedTuple
import numpy as np

from .stage_021 import SafetyDrive
from .stage_022 import SocialDrive
from .stage_023 import ExplorationDrive


class _DrivePriorityTemp(NamedTuple):
    """内部临时结构，用于排序前存储驱力信息。"""
    drive_name: str
    urgency: float
    strength: float
    boundary: float


@dataclass(frozen=True)
class DrivePriority:
    """单个驱力的优先级描述。

    Attributes:
        drive_name: 驱力名称 ('safety' | 'social' | 'exploration')
        urgency: 紧急度 ∈ [0, 1]
        strength: 需求强度 ∈ [0, 1]
        boundary: 边界 ∈ [0, 1]
        priority_rank: 优先级排序（1为最高）

    Invariants:
        - drive_name 为预定义名称之一
        - 0 ≤ urgency ≤ 1
        - priority_rank ≥ 1
    """
    drive_name: str
    urgency: float
    strength: float
    boundary: float
    priority_rank: int

    def __post_init__(self):
        valid_names = {'safety', 'social', 'exploration'}
        if self.drive_name not in valid_names:
            raise ValueError(
                f"驱力名称必须是 {valid_names} 之一，"
                f"收到 '{self.drive_name}'"
            )
        if not (0 <= self.urgency <= 1):
            raise ValueError(
                f"紧急度必须 ∈ [0,1]，收到 {self.urgency}"
            )
        if not (0 <= self.strength <= 1):
            raise ValueError(
                f"需求强度必须 ∈ [0,1]，收到 {self.strength}"
            )
        if not (0 <= self.boundary <= 1):
            raise ValueError(
                f"边界必须 ∈ [0,1]，收到 {self.boundary}"
            )
        if self.priority_rank < 1:
            raise ValueError(
                f"优先级排序必须 ≥ 1，收到 {self.priority_rank}"
            )


@dataclass(frozen=True)
class DriveCompetitionResult:
    """驱力竞争的完整结果。

    组合三个驱力，按紧急度排序后的优先级列表。

    Attributes:
        safety_drive: 安全驱力（组合字段）
        social_drive: 社交驱力（组合字段）
        exploration_drive: 探索驱力（组合字段）
        priorities: 按优先级排序的驱力列表
        winner: 获胜驱力名称

    Invariants:
        - priorities 按 urgency 降序排列
        - winner 为 priorities[0].drive_name
    """
    safety_drive: SafetyDrive
    social_drive: SocialDrive
    exploration_drive: ExplorationDrive
    priorities: Tuple[DrivePriority, ...]
    winner: str

    def __post_init__(self):
        if len(self.priorities) == 0:
            raise ValueError("优先级列表不能为空")
        # 验证按紧急度降序排列
        for i in range(len(self.priorities) - 1):
            if self.priorities[i].urgency < self.priorities[i + 1].urgency:
                raise ValueError(
                    f"优先级列表必须按紧急度降序排列，"
                    f"但位置 {i} 的紧急度 ({self.priorities[i].urgency}) "
                    f"< 位置 {i+1} 的紧急度 ({self.priorities[i+1].urgency})"
                )
        # 验证winner与最高优先级一致
        if self.winner != self.priorities[0].drive_name:
            raise ValueError(
                f"获胜驱力 '{self.winner}' 与最高优先级 "
                f"'{self.priorities[0].drive_name}' 不一致"
            )


class DriveCompetition:
    """阶段24的核心计算接口。

    实现驱力竞争机制，按紧急度对多个驱力排序。
    核心约束：
        - 优先级按紧急度降序排列（A24.1）
        - 获胜驱力为紧急度最高者（A24.2）
    """

    def compute_priorities(
        self,
        safety_drive: SafetyDrive,
        social_drive: SocialDrive,
        exploration_drive: ExplorationDrive
    ) -> Tuple[DrivePriority, ...]:
        """计算驱力优先级排序（A24.1）。

        按紧急度降序排列。

        Args:
            safety_drive: 安全驱力
            social_drive: 社交驱力
            exploration_drive: 探索驱力

        Returns:
            按优先级排序的 DrivePriority 元组

        Raises:
            ValueError: 若任何驱力为 None
        """
        if safety_drive is None:
            raise ValueError("safety_drive 不能为空")
        if social_drive is None:
            raise ValueError("social_drive 不能为空")
        if exploration_drive is None:
            raise ValueError("exploration_drive 不能为空")

        drives = [
            _DrivePriorityTemp(
                drive_name='safety',
                urgency=safety_drive.safety_urgency,
                strength=safety_drive.safety_strength,
                boundary=safety_drive.safety_boundary,
            ),
            _DrivePriorityTemp(
                drive_name='social',
                urgency=social_drive.social_urgency,
                strength=social_drive.social_strength,
                boundary=social_drive.social_boundary,
            ),
            _DrivePriorityTemp(
                drive_name='exploration',
                urgency=exploration_drive.exploration_urgency,
                strength=exploration_drive.exploration_strength,
                boundary=exploration_drive.exploration_boundary,
            ),
        ]

        # 按紧急度降序排列
        sorted_drives = sorted(
            drives, key=lambda d: d.urgency, reverse=True
        )

        # 构造最终的 DrivePriority
        result: List[DrivePriority] = []
        for rank, drive in enumerate(sorted_drives, start=1):
            result.append(DrivePriority(
                drive_name=drive.drive_name,
                urgency=drive.urgency,
                strength=drive.strength,
                boundary=drive.boundary,
                priority_rank=rank,
            ))

        return tuple(result)

    def determine_winner(
        self,
        priorities: Sequence[DrivePriority]
    ) -> str:
        """确定获胜驱力（A24.2）。

        获胜驱力为紧急度最高者（即 priorities[0]）。

        Args:
            priorities: 按优先级排序的驱力列表

        Returns:
            获胜驱力名称

        Raises:
            ValueError: 若 priorities 为空
        """
        if len(priorities) == 0:
            raise ValueError("优先级列表不能为空")
        return priorities[0].drive_name

    def compete(
        self,
        safety_drive: SafetyDrive,
        social_drive: SocialDrive,
        exploration_drive: ExplorationDrive
    ) -> DriveCompetitionResult:
        """执行完整的驱力竞争。

        Args:
            safety_drive: 安全驱力
            social_drive: 社交驱力
            exploration_drive: 探索驱力

        Returns:
            DriveCompetitionResult 实例
        """
        priorities = self.compute_priorities(
            safety_drive, social_drive, exploration_drive
        )
        winner = self.determine_winner(priorities)
        return DriveCompetitionResult(
            safety_drive=safety_drive,
            social_drive=social_drive,
            exploration_drive=exploration_drive,
            priorities=priorities,
            winner=winner,
        )
