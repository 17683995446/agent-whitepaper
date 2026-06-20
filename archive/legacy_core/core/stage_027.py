"""阶段27：驱力引擎集成——从感知到行动的完整链条。

本阶段实现驱力引擎的集成，将阶段21-26的所有组件
串联为完整的从感知到行动的链条。
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .stage_020 import PCValidationResult
from .stage_021 import SafetyDrive, SafetyDriveCalculator
from .stage_022 import SocialDrive, SocialDriveCalculator
from .stage_023 import ExplorationDrive, ExplorationDriveCalculator
from .stage_024 import DriveCompetitionResult, DriveCompetition
from .stage_025 import Goal, GoalMapper
from .stage_026 import GoalDecomposition, GoalDecomposer


@dataclass(frozen=True)
class DriveEngineResult:
    """驱力引擎的完整输出。

    组合阶段21-26的所有结果，形成从感知到行动的完整链条。

    Attributes:
        validation_result: 阶段20的验证结果（组合字段）
        safety_drive: 安全驱力（组合字段）
        social_drive: 社交驱力（组合字段）
        exploration_drive: 探索驱力（组合字段）
        competition_result: 驱力竞争结果（组合字段）
        goal: 目标（组合字段）
        decomposition: 目标分解（组合字段）

    Invariants:
        - 所有组合字段非空
        - goal.goal_type == competition_result.winner
    """
    validation_result: PCValidationResult
    safety_drive: SafetyDrive
    social_drive: SocialDrive
    exploration_drive: ExplorationDrive
    competition_result: DriveCompetitionResult
    goal: Goal
    decomposition: GoalDecomposition

    def __post_init__(self):
        if self.goal.goal_type != self.competition_result.winner:
            raise ValueError(
                f"目标类型 '{self.goal.goal_type}' 与获胜驱力 "
                f"'{self.competition_result.winner}' 不一致"
            )

    @property
    def winner_drive_name(self) -> str:
        """获胜驱力名称。"""
        return self.competition_result.winner

    @property
    def num_subgoals(self) -> int:
        """子目标数量。"""
        return len(self.decomposition.subgoals)


class DriveEngine:
    """阶段27的核心计算接口。

    实现驱力引擎的完整集成，将阶段21-26串联为完整链条。
    核心约束：
        - 输入为阶段20的验证结果
        - 输出为完整的目标分解
        - 所有中间步骤可追踪
    """

    def __init__(self):
        """初始化驱力引擎。"""
        self._safety_calc = SafetyDriveCalculator()
        self._social_calc = SocialDriveCalculator()
        self._explore_calc = ExplorationDriveCalculator()
        self._competition = DriveCompetition()
        self._goal_mapper = GoalMapper()
        self._decomposer = GoalDecomposer()

    def run(
        self,
        validation_result: PCValidationResult
    ) -> DriveEngineResult:
        """执行完整的驱力引擎链条。

        流程：验证结果 → 三种驱力 → 竞争 → 目标 → 分解。

        Args:
            validation_result: 阶段20的验证结果

        Returns:
            DriveEngineResult 实例

        Raises:
            ValueError: 若 validation_result 为 None
        """
        if validation_result is None:
            raise ValueError("validation_result 不能为空")

        # 步骤1: 计算三种驱力（阶段21-23）
        safety_drive = self._safety_calc.from_validation_result(
            validation_result
        )
        social_drive = self._social_calc.from_validation_result(
            validation_result
        )
        exploration_drive = self._explore_calc.from_validation_result(
            validation_result
        )

        # 步骤2: 驱力竞争（阶段24）
        competition_result = self._competition.compete(
            safety_drive, social_drive, exploration_drive
        )

        # 步骤3: 目标映射（阶段25）
        goal = self._goal_mapper.map_goal(competition_result)

        # 步骤4: 目标分解（阶段26）
        decomposition = self._decomposer.decompose(goal)

        return DriveEngineResult(
            validation_result=validation_result,
            safety_drive=safety_drive,
            social_drive=social_drive,
            exploration_drive=exploration_drive,
            competition_result=competition_result,
            goal=goal,
            decomposition=decomposition,
        )

    def run_with_custom_boundaries(
        self,
        validation_result: PCValidationResult,
        safety_boundary: float = 0.5,
        social_boundary: float = 0.5,
        exploration_boundary: float = 0.3,
    ) -> DriveEngineResult:
        """使用自定义边界执行驱力引擎。

        Args:
            validation_result: 阶段20的验证结果
            safety_boundary: 安全边界
            social_boundary: 社交边界
            exploration_boundary: 探索边界

        Returns:
            DriveEngineResult 实例
        """
        if validation_result is None:
            raise ValueError("validation_result 不能为空")

        safety_drive = self._safety_calc.from_validation_result(
            validation_result, custom_boundary=safety_boundary
        )
        social_drive = self._social_calc.from_validation_result(
            validation_result, custom_boundary=social_boundary
        )
        exploration_drive = self._explore_calc.from_validation_result(
            validation_result, custom_boundary=exploration_boundary
        )

        competition_result = self._competition.compete(
            safety_drive, social_drive, exploration_drive
        )
        goal = self._goal_mapper.map_goal(competition_result)
        decomposition = self._decomposer.decompose(goal)

        return DriveEngineResult(
            validation_result=validation_result,
            safety_drive=safety_drive,
            social_drive=social_drive,
            exploration_drive=exploration_drive,
            competition_result=competition_result,
            goal=goal,
            decomposition=decomposition,
        )
