"""核心抽象接口 - 接口分离=可替换性。

第一性原理：
    接口分离原则（ISP）：不应强迫客户依赖其不使用的方法。
    依赖倒置原则（DIP）：高层模块不应依赖低层模块，二者都应依赖抽象。

    通过定义抽象接口，实现：
    1. 可替换性：任何实现接口的类都可替换（如 QTable 可替换为神经网络策略）
    2. 可测试性：测试可 mock 接口而非依赖具体实现
    3. 可扩展性：新实现只需满足接口契约

    已有接口（stage_i）：
    - EnvironmentProtocol: 环境接口（reset/step/action_space/observation_dim/n_states）
    - StateEncoder: 状态编码器接口（encode/n_states）

    本模块补充核心抽象（stage_a 组件）：
    - PolicyProtocol: 策略/值函数接口（get/best_action/update）
    - MemoryProtocol: 记忆缓冲接口（add/sample/get_recent_episodes/__len__）
    - AgentProtocol: 自驱Agent接口（perceive_and_act/run_episode）

    依赖方向：
    - _protocols.py 是基础模块，运行时不依赖任何阶段
    - stage_a 的具体类导入并实现这些协议
    - TYPE_CHECKING 块仅用于类型提示，不产生运行时导入
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from agent_whitepaper.stage_a.environment import Environment, Transition
    from agent_whitepaper.stage_a.memory import Episode


# ============================================================
# 策略协议
# ============================================================
class PolicyProtocol(ABC):
    """策略协议：值函数与动作选择的抽象接口。

    任何策略实现（表格法、神经网络等）都应满足此接口，
    使 Agent 可在不修改代码的情况下切换策略后端。

    契约：
    - get(state): 返回状态对应的值向量（如Q值向量）
    - best_action(state): 返回当前最优动作
    - update(...): 根据经验更新策略，返回TD误差
    """

    @abstractmethod
    def get(self, state: int) -> np.ndarray:
        """获取状态对应的值向量。"""

    @abstractmethod
    def best_action(self, state: int) -> int:
        """获取当前最优动作。"""

    @abstractmethod
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        gamma: float = 0.99,
    ) -> float:
        """根据经验更新策略。

        Returns:
            TD误差（用于优先级回放）
        """


# ============================================================
# 记忆协议
# ============================================================
class MemoryProtocol(ABC):
    """记忆协议：经验存储与采样的抽象接口。

    任何记忆实现（环形缓冲、优先级回放、情景记忆等）
    都应满足此接口，使 Agent 可切换记忆后端。

    契约：
    - add(transition): 存入一个转移
    - sample(batch_size, rng): 采样一批转移（返回索引+转移）
    - get_recent_episodes(n): 获取最近n个完整情节
    - __len__(): 当前存储的转移数
    """

    @abstractmethod
    def add(self, transition: "Transition") -> None:
        """添加一个转移到记忆。"""

    @abstractmethod
    def sample(
        self, batch_size: int, rng: np.random.RandomState
    ) -> Tuple[List[int], List["Transition"]]:
        """随机采样，返回(索引列表, 转移列表)。"""

    @abstractmethod
    def get_recent_episodes(self, n: int) -> List["Episode"]:
        """获取最近 n 个完整情节。"""

    @abstractmethod
    def __len__(self) -> int:
        """当前存储的转移数。"""


# ============================================================
# Agent协议
# ============================================================
class AgentProtocol(ABC):
    """Agent协议：自驱Agent的抽象接口。

    任何Agent实现都应满足此接口，
    使编排器可在不修改代码的情况下切换Agent实现。

    契约：
    - perceive_and_act(...): 感知-决策-行动一步闭环
    - run_episode(env): 运行一个完整情节
    """

    @abstractmethod
    def perceive_and_act(
        self,
        observation: np.ndarray,
        state_index: int,
        env: "Environment",
    ) -> Tuple[int, np.ndarray, float, bool, dict]:
        """感知观测并选择动作，执行一步闭环。

        Returns:
            (action, next_observation, reward, done, info)
        """

    @abstractmethod
    def run_episode(self, env: "Environment", max_steps: int = 100) -> "Episode":
        """运行一个完整情节。"""


__all__ = [
    "PolicyProtocol",
    "MemoryProtocol",
    "AgentProtocol",
]
