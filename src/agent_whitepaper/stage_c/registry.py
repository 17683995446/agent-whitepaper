"""阶段C-1：Agent注册中心 - 管理多Agent的注册、发现与状态。

第一性原理：
    多Agent协作的前提是"知道有谁"。
    注册中心是Agent的"通讯录"：
    - 注册：Agent加入系统时登记
    - 发现：查询当前有哪些Agent、各自能力如何
    - 心跳：监控Agent是否存活
    - 注销：Agent退出时清理

    这是多Agent系统的基础设施，类似于微服务中的服务注册发现。
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ============================================================
# Agent状态
# ============================================================
class AgentStatus(Enum):
    """Agent运行状态。"""
    REGISTERED = "registered"    # 已注册，未启动
    ACTIVE = "active"            # 活跃中
    BUSY = "busy"                # 忙碌（执行任务中）
    IDLE = "idle"                # 空闲
    OFFLINE = "offline"          # 离线


# ============================================================
# Agent能力描述
# ============================================================
@dataclass(frozen=True)
class AgentCapability:
    """Agent能力描述。

    Attributes:
        skills: 已掌握技能名称列表
        specialization: 专长领域（如"导航""搬运""感知"）
        performance_score: 性能评分 [0,1]
        drive_profile: 驱力特征 (safety, social, exploration)
    """
    skills: Tuple[str, ...] = ()
    specialization: str = "general"
    performance_score: float = 0.5
    drive_profile: Tuple[float, float, float] = (0.5, 0.5, 0.5)

    def __post_init__(self):
        if not (0 <= self.performance_score <= 1):
            raise ValueError("performance_score 必须 ∈ [0,1]")
        for d in self.drive_profile:
            if not (0 <= d <= 1):
                raise ValueError("驱力值必须 ∈ [0,1]")


# ============================================================
# Agent注册信息
# ============================================================
@dataclass
class AgentRegistration:
    """Agent注册信息。

    Attributes:
        agent_id: 唯一标识
        name: 可读名称
        capability: 能力描述
        status: 当前状态
        register_time: 注册时间戳
        last_heartbeat: 最后心跳时间
        address: 通信地址（如消息队列名）
    """
    agent_id: str
    name: str
    capability: AgentCapability
    status: AgentStatus = AgentStatus.REGISTERED
    register_time: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    address: Optional[str] = None


# ============================================================
# Agent注册中心
# ============================================================
class AgentRegistry:
    """Agent注册中心。

    功能：
    1. 注册/注销Agent
    2. 查询Agent信息
    3. 按能力发现Agent
    4. 心跳监控
    5. 状态管理
    """

    def __init__(self, heartbeat_timeout: float = 30.0):
        if heartbeat_timeout <= 0:
            raise ValueError("heartbeat_timeout 必须 > 0")
        self.heartbeat_timeout = heartbeat_timeout
        self._agents: Dict[str, AgentRegistration] = {}
        self._id_counter = 0

    def register(
        self,
        name: str,
        capability: Optional[AgentCapability] = None,
        address: Optional[str] = None
    ) -> str:
        """注册一个新Agent。

        Returns:
            agent_id
        """
        self._id_counter += 1
        agent_id = f"agent_{self._id_counter:04d}"
        capability = capability or AgentCapability()

        registration = AgentRegistration(
            agent_id=agent_id,
            name=name,
            capability=capability,
            status=AgentStatus.ACTIVE,
            address=address,
        )
        self._agents[agent_id] = registration
        return agent_id

    def unregister(self, agent_id: str) -> bool:
        """注销Agent。"""
        if agent_id not in self._agents:
            return False
        del self._agents[agent_id]
        return True

    def get_agent(self, agent_id: str) -> Optional[AgentRegistration]:
        """获取Agent信息。"""
        return self._agents.get(agent_id)

    def get_all_agents(self) -> List[AgentRegistration]:
        """获取所有Agent。"""
        return list(self._agents.values())

    def get_active_agents(self) -> List[AgentRegistration]:
        """获取所有活跃Agent。"""
        return [
            reg for reg in self._agents.values()
            if reg.status in (AgentStatus.ACTIVE, AgentStatus.IDLE)
        ]

    def find_by_specialization(self, specialization: str) -> List[AgentRegistration]:
        """按专长查找Agent。"""
        return [
            reg for reg in self._agents.values()
            if reg.capability.specialization == specialization
            and reg.status in (AgentStatus.ACTIVE, AgentStatus.IDLE)
        ]

    def find_by_skill(self, skill_name: str) -> List[AgentRegistration]:
        """按技能查找Agent。"""
        return [
            reg for reg in self._agents.values()
            if skill_name in reg.capability.skills
            and reg.status in (AgentStatus.ACTIVE, AgentStatus.IDLE)
        ]

    def find_best_agent(
        self,
        specialization: Optional[str] = None,
        required_skill: Optional[str] = None
    ) -> Optional[AgentRegistration]:
        """找到最适合的Agent（性能评分最高）。"""
        candidates = []
        if specialization:
            candidates = self.find_by_specialization(specialization)
        elif required_skill:
            candidates = self.find_by_skill(required_skill)
        else:
            candidates = self.get_active_agents()

        if not candidates:
            return None
        return max(candidates, key=lambda r: r.capability.performance_score)

    def update_status(self, agent_id: str, status: AgentStatus) -> bool:
        """更新Agent状态。"""
        if agent_id not in self._agents:
            return False
        self._agents[agent_id].status = status
        return True

    def heartbeat(self, agent_id: str) -> bool:
        """Agent心跳更新。"""
        if agent_id not in self._agents:
            return False
        self._agents[agent_id].last_heartbeat = time.time()
        return True

    def check_timeouts(self) -> List[str]:
        """检查超时的Agent，标记为离线。

        Returns:
            被标记为离线的agent_id列表
        """
        current_time = time.time()
        timed_out = []
        for agent_id, reg in self._agents.items():
            if reg.status == AgentStatus.OFFLINE:
                continue
            if current_time - reg.last_heartbeat > self.heartbeat_timeout:
                reg.status = AgentStatus.OFFLINE
                timed_out.append(agent_id)
        return timed_out

    def update_capability(self, agent_id: str, capability: AgentCapability) -> bool:
        """更新Agent能力描述。"""
        if agent_id not in self._agents:
            return False
        self._agents[agent_id].capability = capability
        return True

    def get_registry_summary(self) -> dict:
        """获取注册中心摘要。"""
        status_counts = {}
        for reg in self._agents.values():
            status_counts[reg.status.value] = status_counts.get(reg.status.value, 0) + 1
        specializations = set(
            reg.capability.specialization for reg in self._agents.values()
        )
        return {
            "total_agents": len(self._agents),
            "status_counts": status_counts,
            "specializations": list(specializations),
        }
