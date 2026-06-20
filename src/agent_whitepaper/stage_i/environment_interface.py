"""阶段I-1：环境接口与状态编码器 - 泛化的基础。

第一性原理：
    泛化能力是实用性的核心。当前系统与GridWorld耦合：
    - QTable.state_to_index 假设观测是GridWorld的one-hot格式
    - LongTaskExecutor 的 state_extractor 假设GridWorld位置编码

    要让Agent能学习**任何**任务，需要抽象两个东西：
    1. 环境接口：reset/step/action_space（已有，但需正式化）
    2. 状态编码器：observation → state_index（关键缺失抽象）

    状态编码器是"感知"的抽象——它定义了Agent如何"看"世界。
    不同的环境需要不同的编码器，但Q表和策略不依赖具体编码。

    这实现了"一个大脑，多个世界"：同一个Agent框架，
    配上不同的环境+编码器，就能学习不同任务。
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import numpy as np

from agent_whitepaper.stage_a.action_space import ActionSpace, DiscreteActionSpace


# ============================================================
# 正式环境协议
# ============================================================
class EnvironmentProtocol(ABC):
    """环境协议：所有环境必须实现的接口。

    遵循MDP（马尔可夫决策过程）：
    - reset() → 初始观测
    - step(action) → (观测, 奖励, 是否终止, 信息)
    - action_space → 动作空间
    - observation_dim → 观测维度
    - n_states → 离散状态数（用于Q表）
    """

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """重置环境，返回初始观测。"""

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """执行动作，返回(观测, 奖励, 是否终止, 信息)。"""

    @property
    @abstractmethod
    def action_space(self) -> ActionSpace:
        """动作空间。"""

    @property
    @abstractmethod
    def observation_dim(self) -> int:
        """观测向量维度。"""

    @property
    @abstractmethod
    def n_states(self) -> int:
        """离散状态数（用于Q表索引）。"""


# ============================================================
# 状态编码器：observation → state_index
# ============================================================
class StateEncoder(ABC):
    """状态编码器：将观测向量编码为离散状态索引。

    这是Agent"感知"世界的抽象。不同环境有不同的编码方式：
    - GridWorld: agent位置 one-hot → 位置索引
    - 函数调用: 已完成步骤的位图 → 步骤组合索引
    - 自定义: 任意观测 → 离散索引

    编码器必须满足：
    1. 确定性：相同观测 → 相同索引
    2. 完备性：所有合法观测都能编码
    3. 紧凑性：索引空间尽可能小
    """

    @abstractmethod
    def encode(self, observation: np.ndarray) -> int:
        """将观测编码为状态索引。"""

    @property
    @abstractmethod
    def n_states(self) -> int:
        """状态空间大小。"""


# ============================================================
# GridWorld状态编码器
# ============================================================
class GridWorldStateEncoder(StateEncoder):
    """GridWorld状态编码器：从one-hot观测提取agent位置索引。

    观测格式：[agent_one_hot (size*size), goal_one_hot (size*size)]
    状态 = agent位置索引
    """

    def __init__(self, grid_size: int):
        if grid_size < 2:
            raise ValueError("grid_size 必须 >= 2")
        self.grid_size = grid_size

    def encode(self, observation: np.ndarray) -> int:
        agent_part = observation[:self.grid_size * self.grid_size]
        return int(np.argmax(agent_part))

    @property
    def n_states(self) -> int:
        return self.grid_size * self.grid_size


# ============================================================
# 位图状态编码器（通用）
# ============================================================
class BitmapStateEncoder(StateEncoder):
    """位图状态编码器：将二进制位图观测编码为状态索引。

    适用于：函数调用环境、多步骤任务等。
    观测的每一位代表一个二元状态（如：某步骤是否完成）。
    状态索引 = 位图的二进制值。

    例如：3个步骤，观测=[1,0,1] → 状态索引=5 (二进制101)
    """

    def __init__(self, n_bits: int):
        if n_bits < 1:
            raise ValueError("n_bits 必须 >= 1")
        if n_bits > 20:
            raise ValueError("n_bits 过大（>20），状态空间爆炸")
        self.n_bits = n_bits

    def encode(self, observation: np.ndarray) -> int:
        # 将观测二值化后按位编码
        bits = (np.asarray(observation).flatten() > 0.5).astype(int)[:self.n_bits]
        index = 0
        for i, b in enumerate(bits):
            if b:
                index |= (1 << i)
        return index

    @property
    def n_states(self) -> int:
        return 2 ** self.n_bits


# ============================================================
# 环境适配器：将环境+编码器组合为Agent可用的形式
# ============================================================
class EnvironmentAdapter:
    """环境适配器：组合环境与状态编码器。

    提供统一的接口给Agent使用：
    - reset() → (观测, 状态索引)
    - step(action) → (观测, 状态索引, 奖励, 是否终止, 信息)

    Agent只需关心状态索引，不需要知道观测的具体格式。
    """

    def __init__(self, env: EnvironmentProtocol, encoder: StateEncoder):
        self.env = env
        self.encoder = encoder

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """重置环境，返回(观测, 状态索引)。"""
        obs = self.env.reset(seed)
        state_idx = self.encoder.encode(obs)
        return obs, state_idx

    def step(self, action: int) -> Tuple[np.ndarray, int, float, bool, dict]:
        """执行动作，返回(观测, 状态索引, 奖励, 是否终止, 信息)。"""
        obs, reward, done, info = self.env.step(action)
        state_idx = self.encoder.encode(obs)
        return obs, state_idx, reward, done, info

    @property
    def action_space(self) -> ActionSpace:
        return self.env.action_space

    @property
    def n_states(self) -> int:
        return self.encoder.n_states

    @property
    def observation_dim(self) -> int:
        return self.env.observation_dim
