"""阶段A：单Agent闭环 - 情节记忆缓冲。

第一性原理：
    记忆是学习的载体。Agent必须记住"在什么状态下做了什么动作得到了什么结果"，
    才能从中学习。情节记忆（Episodic Memory）存储完整的转移序列。

    本模块实现：
    1. EpisodicBuffer：环形缓冲，存储最近N个转移
    2. Episode：一个完整情节（从reset到done）
    3. 优先级回放：基于TD误差的优先级采样（加速学习）

    设计原则：
    - 缓冲区是有状态的（可变），但单个Transition是不可变的
    - 采样通过显式传入rng，保证可复现
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .environment import Transition


# ============================================================
# 情节（完整的一次任务经历）
# ============================================================
@dataclass(frozen=True)
class Episode:
    """一个完整情节。

    Attributes:
        transitions: 转移序列
        total_reward: 总奖励
        length: 情节长度
        success: 是否成功（到达目标）
    """
    transitions: Tuple[Transition, ...]
    total_reward: float
    length: int
    success: bool

    @classmethod
    def from_transitions(cls, transitions: List[Transition]) -> "Episode":
        """从转移列表构建情节。"""
        if not transitions:
            raise ValueError("情节必须包含至少一个转移")
        total_reward = sum(t.reward for t in transitions)
        success = transitions[-1].info.get("reached_goal", False)
        return cls(
            transitions=tuple(transitions),
            total_reward=total_reward,
            length=len(transitions),
            success=success,
        )


# ============================================================
# 情节记忆缓冲（环形缓冲）
# ============================================================
class EpisodicBuffer:
    """情节记忆缓冲区。

    存储最近 capacity 个转移，支持：
    - 添加转移
    - 均匀随机采样
    - 优先级采样（基于TD误差）
    - 提取完整情节
    """

    def __init__(self, capacity: int = 10000):
        if capacity < 1:
            raise ValueError(f"容量必须 >= 1，收到 {capacity}")
        self.capacity = capacity
        self._transitions: List[Transition] = []
        self._priorities: List[float] = []
        self._position = 0
        self._current_episode: List[Transition] = []
        self._completed_episodes: List[Episode] = []

    def __len__(self) -> int:
        return len(self._transitions)

    @property
    def num_episodes(self) -> int:
        return len(self._completed_episodes)

    def add(self, transition: Transition) -> None:
        """添加一个转移。"""
        # 默认优先级为最大优先级（新样本优先探索）
        max_priority = max(self._priorities) if self._priorities else 1.0
        if len(self._transitions) < self.capacity:
            self._transitions.append(transition)
            self._priorities.append(max_priority)
        else:
            self._transitions[self._position] = transition
            self._priorities[self._position] = max_priority
        self._position = (self._position + 1) % self.capacity

        # 累积当前情节
        self._current_episode.append(transition)
        if transition.done:
            episode = Episode.from_transitions(self._current_episode)
            self._completed_episodes.append(episode)
            # 只保留最近100个完整情节
            if len(self._completed_episodes) > 100:
                self._completed_episodes = self._completed_episodes[-100:]
            self._current_episode = []

    def update_priority(self, indices: List[int], priorities: List[float]) -> None:
        """更新转移的优先级。"""
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self._transitions):
                self._priorities[idx] = max(priority, 1e-6)

    def sample(self, batch_size: int, rng: np.random.RandomState) -> Tuple[List[int], List[Transition]]:
        """均匀随机采样。"""
        if len(self._transitions) < batch_size:
            raise ValueError(
                f"缓冲区大小 {len(self._transitions)} < 批大小 {batch_size}"
            )
        indices = rng.choice(len(self._transitions), size=batch_size, replace=False).tolist()
        transitions = [self._transitions[i] for i in indices]
        return indices, transitions

    def sample_prioritized(
        self,
        batch_size: int,
        rng: np.random.RandomState,
        alpha: float = 0.6
    ) -> Tuple[List[int], List[Transition], np.ndarray]:
        """优先级采样。

        Args:
            batch_size: 批大小
            rng: 随机数生成器
            alpha: 优先级强度（0=均匀，1=完全按优先级）

        Returns:
            (indices, transitions, weights) - weights用于重要性采样校正
        """
        if len(self._transitions) < batch_size:
            raise ValueError(
                f"缓冲区大小 {len(self._transitions)} < 批大小 {batch_size}"
            )
        priorities = np.array(self._priorities, dtype=float)
        probs = priorities ** alpha
        probs = probs / probs.sum()

        indices = rng.choice(len(self._transitions), size=batch_size, replace=False, p=probs)
        transitions = [self._transitions[i] for i in indices]

        # 重要性采样权重
        weights = (len(self._transitions) * probs[indices]) ** (-0.4)
        weights = weights / weights.max()

        return indices.tolist(), transitions, weights

    def get_recent_episodes(self, n: int) -> List[Episode]:
        """获取最近n个完整情节。"""
        return self._completed_episodes[-n:]

    def get_best_episodes(self, n: int) -> List[Episode]:
        """获取奖励最高的n个情节。"""
        sorted_episodes = sorted(
            self._completed_episodes,
            key=lambda e: e.total_reward,
            reverse=True
        )
        return sorted_episodes[:n]

    def clear(self) -> None:
        """清空缓冲区。"""
        self._transitions = []
        self._priorities = []
        self._position = 0
        self._current_episode = []
        self._completed_episodes = []
