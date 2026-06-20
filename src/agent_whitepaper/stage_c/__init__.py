"""阶段C：多Agent协作包。

公共API：
    AgentStatus: Agent状态枚举
    AgentCapability: Agent能力
    AgentRegistration: Agent注册信息
    AgentRegistry: Agent注册中心
    MessageType: 消息类型枚举
    Message: 消息
    MessageQueue: 消息队列
    MessageBus: 消息总线
    Role: 角色枚举
    RoleAssignment: 角色分配
    TeamConfiguration: 团队配置
    RoleAssigner: 角色分配器
    CoalitionStatus: 联盟状态枚举
    Coalition: 联盟
    CoalitionFormer: 联盟形成器
    SharedSkill: 共享技能
    SharedSkillLibrary: 共享技能库
"""

from .coalition import Coalition, CoalitionFormer, CoalitionStatus
from .message_bus import (
    Message,
    MessageBus,
    MessageQueue,
    MessageType,
)
from .registry import (
    AgentCapability,
    AgentRegistration,
    AgentRegistry,
    AgentStatus,
)
from .role_assigner import (
    Role,
    RoleAssigner,
    RoleAssignment,
    TeamConfiguration,
)
from .shared_skills import SharedSkill, SharedSkillLibrary

__all__ = [
    "AgentStatus",
    "AgentCapability",
    "AgentRegistration",
    "AgentRegistry",
    "MessageType",
    "Message",
    "MessageQueue",
    "MessageBus",
    "Role",
    "RoleAssignment",
    "TeamConfiguration",
    "RoleAssigner",
    "CoalitionStatus",
    "Coalition",
    "CoalitionFormer",
    "SharedSkill",
    "SharedSkillLibrary",
]
