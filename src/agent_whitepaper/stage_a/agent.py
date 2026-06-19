"""阶段A：单Agent闭环集成。

第一性原理：
    闭环是Agent存在的本质。Agent通过以下循环实现"自驱"：
    1. 感知环境 → 获得观测
    2. 推断表征 → 用预测编码模型计算自由能和驱力
    3. 选择动作 → 基于驱力的策略
    4. 执行动作 → 作用于环境
    5. 获得反馈 → 奖励和新观测
    6. 更新记忆 → 存入情节缓冲
    7. 学习更新 → Q-learning更新Q表
    8. 回到步骤1

    这个循环不需要外部指令，Agent"自己驱动自己"完成任务。
    驱力系统决定Agent何时探索、何时利用，实现自主性。
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np

from .action_space import DiscreteActionSpace
from .environment import Environment, Transition
from .memory import EpisodicBuffer, Episode
from .policy import QTable, DriveBasedPolicy, DriveState


# ============================================================
# Agent运行统计
# ============================================================
@dataclass
class AgentStats:
    """Agent运行统计。"""
    total_steps: int = 0
    total_episodes: int = 0
    total_reward: float = 0.0
    successes: int = 0
    recent_rewards: List[float] = field(default_factory=list)
    recent_successes: List[bool] = field(default_factory=list)
    free_energy_history: List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.recent_successes:
            return 0.0
        return sum(self.recent_successes) / len(self.recent_successes)

    @property
    def avg_reward(self) -> float:
        if not self.recent_rewards:
            return 0.0
        return sum(self.recent_rewards) / len(self.recent_rewards)

    def record_episode(self, reward: float, success: bool, free_energy: float = 0.0) -> None:
        self.total_episodes += 1
        self.total_reward += reward
        if success:
            self.successes += 1
        self.recent_rewards.append(reward)
        self.recent_successes.append(success)
        self.free_energy_history.append(free_energy)
        # 只保留最近50个情节的统计
        if len(self.recent_rewards) > 50:
            self.recent_rewards = self.recent_rewards[-50:]
            self.recent_successes = self.recent_successes[-50:]
            self.free_energy_history = self.free_energy_history[-50:]


# ============================================================
# 自驱Agent
# ============================================================
class SelfDrivenAgent:
    """自驱Agent：感知-推断-决策-行动-学习的完整闭环。

    核心特性：
    1. 自驱：不需要外部指令，基于驱力自主选择探索或利用
    2. 学习：通过Q-learning从记忆中学习最优策略
    3. 适应：驱力根据自由能和奖励动态调整
    4. 记忆：情节记忆缓冲存储经历，支持优先级回放
    """

    def __init__(
        self,
        action_space: DiscreteActionSpace,
        n_states: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        min_epsilon: float = 0.05,
        seed: Optional[int] = None
    ):
        self.action_space = action_space
        self.q_table = QTable(n_states, action_space.n, learning_rate)
        self.policy = DriveBasedPolicy(action_space, min_epsilon=min_epsilon)
        self.memory = EpisodicBuffer(capacity=10000)
        self.gamma = gamma
        self._rng = np.random.RandomState(seed)
        self.drives = DriveState(
            safety=0.5, social=0.5, exploration=0.8  # 初始高探索
        )
        self.stats = AgentStats()

    def perceive_and_act(
        self,
        observation: np.ndarray,
        state_index: int,
        env: Environment
    ) -> Tuple[int, np.ndarray, float, bool, dict]:
        """感知-决策-行动一步。"""
        # 基于驱力选择动作
        action = self.policy.select_action(
            self.q_table, state_index, self.drives, self._rng
        )
        # 执行动作
        next_obs, reward, done, info = env.step(action)
        next_state_index = self.q_table.state_to_index(
            next_obs, info.get("grid_size", 5)
        ) if "grid_size" in info else state_index

        # 创建转移记录
        transition = Transition(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_obs,
            done=done,
            info=info
        )
        # 存入记忆
        self.memory.add(transition)

        # Q-learning更新
        td_error = self.q_table.update(
            state_index, action, reward, next_state_index, done, self.gamma
        )
        # 更新优先级
        idx = len(self.memory) - 1
        self.memory.update_priority([idx], [abs(td_error) + 1e-6])

        # 更新统计
        self.stats.total_steps += 1

        # 更新驱力（用TD误差的绝对值作为"自由能"代理）
        proxy_free_energy = abs(td_error)
        self.drives = self.policy.update_drives_from_free_energy(
            self.drives, proxy_free_energy,
            self.stats.free_energy_history, reward
        )

        return action, next_obs, reward, done, info

    def run_episode(self, env: Environment, max_steps: int = 100) -> Episode:
        """运行一个完整情节。"""
        obs = env.reset()
        grid_size = int(np.sqrt(len(obs) // 2))
        state_index = self.q_table.state_to_index(obs, grid_size)

        episode_reward = 0.0
        for step in range(max_steps):
            action, next_obs, reward, done, info = self.perceive_and_act(
                obs, state_index, env
            )
            episode_reward += reward
            obs = next_obs
            # 更新state_index
            state_index = self.q_table.state_to_index(obs, grid_size)
            if done:
                break

        # 记录情节统计
        success = info.get("reached_goal", False)
        proxy_fe = self.stats.free_energy_history[-1] if self.stats.free_energy_history else 0.0
        self.stats.record_episode(episode_reward, success, proxy_fe)

        # 从记忆获取刚完成的情节
        recent_episodes = self.memory.get_recent_episodes(1)
        if recent_episodes:
            return recent_episodes[0]
        # 如果没有完整情节（理论上不应该），构造一个
        return Episode(
            transitions=(),
            total_reward=episode_reward,
            length=step + 1,
            success=success,
        )

    def replay_learning(self, batch_size: int = 32, n_updates: int = 10) -> float:
        """从记忆中回放学习（优先级回放）。"""
        if len(self.memory) < batch_size:
            return 0.0

        total_td_error = 0.0
        grid_size = 5  # 默认，实际应从环境获取
        for _ in range(n_updates):
            try:
                indices, transitions, weights = self.memory.sample_prioritized(
                    batch_size, self._rng
                )
            except ValueError:
                break

            td_errors = []
            for idx, transition, weight in zip(indices, transitions, weights):
                s = self.q_table.state_to_index(transition.observation, grid_size)
                ns = self.q_table.state_to_index(transition.next_observation, grid_size)
                td_error = self.q_table.update(
                    s, transition.action, transition.reward,
                    ns, transition.done, self.gamma
                )
                td_errors.append(abs(td_error))
            # 更新优先级
            self.memory.update_priority(indices, td_errors)
            total_td_error += sum(td_errors)

        return total_td_error / max(n_updates, 1)
