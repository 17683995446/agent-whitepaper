"""阶段A：单Agent闭环包。

公共API：
    SelfDrivenAgent: 自驱Agent，感知-决策-行动-学习闭环
    GridWorld: 网格世界环境
    QTable: Q值表
    DriveBasedPolicy: 基于驱力的策略
    EpisodicBuffer: 情节记忆缓冲
    Episode: 情节记录
    Transition: 单步转移记录
    DiscreteActionSpace: 离散动作空间
    AgentStats: Agent运行统计
    DriveState: 驱力状态
"""

from .action_space import DiscreteActionSpace, ActionSpace
from .environment import Environment, GridWorld, Transition
from .memory import EpisodicBuffer, Episode
from .policy import QTable, DriveBasedPolicy, DriveState
from .agent import SelfDrivenAgent, AgentStats

__all__ = [
    "DiscreteActionSpace",
    "ActionSpace",
    "Environment",
    "GridWorld",
    "Transition",
    "EpisodicBuffer",
    "Episode",
    "QTable",
    "DriveBasedPolicy",
    "DriveState",
    "SelfDrivenAgent",
    "AgentStats",
]
