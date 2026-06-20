"""阶段B：技能增强与长任务包。

公共API：
    Skill: 抽象技能
    SkillLibrary: 技能库
    SkillAbstractor: 技能抽象器（n-gram提取）
    SkillExecutionRecord: 技能执行记录
    SkillRefiner: 技能精炼器
    MetaParameters: 元学习参数
    LearningProgress: 学习进度
    MetaLearner: 元学习器（学习如何学习）
    SubGoalStatus: 子目标状态枚举
    SubGoal: 子目标
    Task: 任务
    TaskDecomposer: 任务分解器
    SubGoalExecution: 子目标执行
    TaskProgress: 任务进度
    SubGoalStateMachine: 子目标状态机
    LongTaskExecutor: 长任务执行器
"""

from .skill_abstractor import Skill, SkillLibrary, SkillAbstractor
from .skill_refiner import SkillExecutionRecord, SkillRefiner
from .meta_learner import MetaParameters, LearningProgress, MetaLearner
from .task_decomposer import SubGoalStatus, SubGoal, Task, TaskDecomposer
from .subgoal_manager import (
    SubGoalExecution,
    TaskProgress,
    SubGoalStateMachine,
    LongTaskExecutor,
)

__all__ = [
    "Skill",
    "SkillLibrary",
    "SkillAbstractor",
    "SkillExecutionRecord",
    "SkillRefiner",
    "MetaParameters",
    "LearningProgress",
    "MetaLearner",
    "SubGoalStatus",
    "SubGoal",
    "Task",
    "TaskDecomposer",
    "SubGoalExecution",
    "TaskProgress",
    "SubGoalStateMachine",
    "LongTaskExecutor",
]
