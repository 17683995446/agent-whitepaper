"""阶段A：单Agent闭环 - 环境接口与GridWorld实现。

第一性原理：
    环境是Agent的"世界"。Agent通过动作改变环境状态，
    环境返回新观测和奖励。这个闭环是所有学习的基础。

    环境接口遵循MDP（马尔可夫决策过程）：
    - 状态转移 P(s'|s,a)
    - 奖励函数 R(s,a,s')
    - 观测函数 O(s)

    GridWorld是最简单的有目标环境：
    - 网格世界，Agent在格子中移动
    - 到达目标得正奖励，碰墙不得分
    - 用于验证Agent能否"自驱学会"到达目标
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .action_space import ActionSpace, DiscreteActionSpace


# ============================================================
# 环境接口（抽象基类）
# ============================================================
@dataclass
class Environment:
    """环境抽象基类。

    环境是有状态的（状态随step改变），因此使用可变dataclass。
    但每次step返回不可变的Transition，保证历史可追溯。
    """

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """重置环境到初始状态，返回初始观测。"""
        raise NotImplementedError

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        """执行动作，返回 (observation, reward, done, info)。"""
        raise NotImplementedError

    @property
    def action_space(self) -> ActionSpace:
        """返回动作空间。"""
        raise NotImplementedError

    @property
    def observation_dim(self) -> int:
        """观测维度。"""
        raise NotImplementedError


# ============================================================
# 转移记录（不可变）
# ============================================================
@dataclass(frozen=True)
class Transition:
    """单步转移记录。

    Attributes:
        observation: 执行动作前的观测
        action: 执行的动作
        reward: 获得的奖励
        next_observation: 执行动作后的观测
        done: 是否终止
        info: 额外信息
    """
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    done: bool
    info: dict


# ============================================================
# GridWorld环境
# ============================================================
class GridWorld(Environment):
    """网格世界环境。

    Agent在 size×size 的网格中移动，目标是到达goal位置。
    动作：0=上, 1=下, 2=左, 3=右
    奖励：到达目标 +1，每步 -0.01（鼓励快速到达）

    观测编码：Agent位置 + 目标位置的one-hot拼接
    """

    # 动作定义
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3

    def __init__(
        self,
        size: int = 5,
        goal: Optional[Tuple[int, int]] = None,
        start: Optional[Tuple[int, int]] = None,
        max_steps: int = 100,
        seed: Optional[int] = None
    ) -> None:
        if size < 2:
            raise ValueError(f"网格大小必须 >= 2，收到 {size}")
        self.size = size
        self.goal = goal if goal is not None else (size - 1, size - 1)
        self.start = start if start is not None else (0, 0)
        self.max_steps = max_steps
        self._rng = np.random.RandomState(seed)

        # 验证目标与起点在网格内
        for pos, name in [(self.goal, "goal"), (self.start, "start")]:
            if not (0 <= pos[0] < size and 0 <= pos[1] < size):
                raise ValueError(f"{name} 位置 {pos} 超出网格范围")

        self._action_space = DiscreteActionSpace(
            n=4,
            action_names=("up", "down", "left", "right")
        )
        self._agent_pos = self.start
        self._step_count = 0

    @property
    def action_space(self) -> DiscreteActionSpace:
        return self._action_space

    @property
    def observation_dim(self) -> int:
        # Agent位置one-hot (size*size) + 目标位置one-hot (size*size)
        return 2 * self.size * self.size

    def _encode_observation(self) -> np.ndarray:
        """将状态编码为观测向量。"""
        obs = np.zeros(2 * self.size * self.size)
        agent_idx = self._agent_pos[0] * self.size + self._agent_pos[1]
        goal_idx = self.goal[0] * self.size + self.goal[1]
        obs[agent_idx] = 1.0
        obs[self.size * self.size + goal_idx] = 1.0
        return obs

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """重置环境。"""
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._agent_pos = self.start
        self._step_count = 0
        return self._encode_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """执行一步。"""
        if not self._action_space.contains(action):
            raise ValueError(f"非法动作 {action}")

        row, col = self._agent_pos
        # 动作映射
        if action == self.ACTION_UP:
            new_pos = (max(0, row - 1), col)
        elif action == self.ACTION_DOWN:
            new_pos = (min(self.size - 1, row + 1), col)
        elif action == self.ACTION_LEFT:
            new_pos = (row, max(0, col - 1))
        elif action == self.ACTION_RIGHT:
            new_pos = (row, min(self.size - 1, col + 1))
        else:
            new_pos = (row, col)

        self._agent_pos = new_pos
        self._step_count += 1

        # 奖励计算
        reached_goal = (new_pos == self.goal)
        reward = 1.0 if reached_goal else -0.01

        # 终止条件
        done = reached_goal or (self._step_count >= self.max_steps)

        info = {
            "agent_pos": new_pos,
            "goal_pos": self.goal,
            "step_count": self._step_count,
            "reached_goal": reached_goal,
        }
        return self._encode_observation(), reward, done, info
