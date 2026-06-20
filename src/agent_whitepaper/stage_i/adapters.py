"""阶段I-3：通用Agent适配器 - 让一个Agent框架学习多种任务。

第一性原理：
    现有SelfDrivenAgent与GridWorld耦合在state_to_index上。
    要实现泛化，需要让Agent通过StateEncoder感知任意环境。

    适配器模式：不修改现有Agent代码，而是创建一个通用运行函数，
    用EnvironmentAdapter替代Agent内部的GridWorld状态编码。

    这样同一个SelfDrivenAgent + Q-learning算法，
    配上不同的(EnvironmentAdapter)就能学习不同任务。
    这就是"一个大脑，多个世界"。
"""

from typing import List, Optional, Tuple

from agent_whitepaper.stage_a.action_space import DiscreteActionSpace
from agent_whitepaper.stage_a.agent import SelfDrivenAgent
from agent_whitepaper.stage_a.memory import Episode, Transition

from .environment_interface import EnvironmentAdapter, StateEncoder


# ============================================================
# 通用情节运行函数
# ============================================================
def run_episode_universal(
    agent: SelfDrivenAgent,
    adapter: EnvironmentAdapter,
    max_steps: int = 100
) -> Episode:
    """用任意环境运行一个完整情节。

    与SelfDrivenAgent.run_episode的区别：
    - 使用EnvironmentAdapter计算状态索引（不依赖GridWorld编码）
    - 适用于任何符合EnvironmentProtocol的环境

    Args:
        agent: SelfDrivenAgent实例
        adapter: 环境适配器（环境+编码器）
        max_steps: 最大步数

    Returns:
        完成的情节
    """
    obs, state_idx = adapter.reset()
    episode_reward = 0.0
    transitions = []
    info = {}
    step = 0

    for step in range(max_steps):
        # 基于驱力选择动作
        action = agent.policy.select_action(
            agent.q_table, state_idx, agent.drives, agent._rng
        )

        # 通过适配器执行（用编码器计算下一状态索引）
        next_obs, next_state_idx, reward, done, info = adapter.step(action)

        # 创建转移记录
        transition = Transition(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=next_obs,
            done=done,
            info=info,
        )
        transitions.append(transition)
        agent.memory.add(transition)

        # Q-learning更新（用适配器提供的状态索引）
        td_error = agent.q_table.update(
            state_idx, action, reward, next_state_idx, done, agent.gamma
        )

        # 更新优先级
        idx = len(agent.memory) - 1
        agent.memory.update_priority([idx], [abs(td_error) + 1e-6])

        # 更新统计
        agent.stats.total_steps += 1

        # 更新驱力
        proxy_free_energy = abs(td_error)
        agent.drives = agent.policy.update_drives_from_free_energy(
            agent.drives, proxy_free_energy,
            agent.stats.free_energy_history, reward
        )

        episode_reward += reward
        obs = next_obs
        state_idx = next_state_idx

        if done:
            break

    # 记录情节统计
    success = info.get("reached_goal", False)
    proxy_fe = (
        agent.stats.free_energy_history[-1]
        if agent.stats.free_energy_history else 0.0
    )
    agent.stats.record_episode(episode_reward, success, proxy_fe)

    return Episode(
        transitions=tuple(transitions),
        total_reward=episode_reward,
        length=step + 1,
        success=success,
    )


# ============================================================
# 通用回放学习
# ============================================================
def replay_learning_universal(
    agent: SelfDrivenAgent,
    encoder: StateEncoder,
    batch_size: int = 32,
    n_updates: int = 10
) -> float:
    """通用回放学习（用编码器计算状态索引）。

    与SelfDrivenAgent.replay_learning的区别：
    - 使用StateEncoder而非grid_size参数
    """
    if len(agent.memory) < batch_size:
        return 0.0

    total_td_error = 0.0
    for _ in range(n_updates):
        try:
            indices, transitions, weights = agent.memory.sample_prioritized(
                batch_size, agent._rng
            )
        except ValueError:
            break

        td_errors = []
        for idx, transition, weight in zip(indices, transitions, weights):
            s = encoder.encode(transition.observation)
            ns = encoder.encode(transition.next_observation)
            td_error = agent.q_table.update(
                s, transition.action, transition.reward,
                ns, transition.done, agent.gamma
            )
            td_errors.append(abs(td_error))
        agent.memory.update_priority(indices, td_errors)
        total_td_error += sum(td_errors)

    return total_td_error / max(n_updates, 1)


# ============================================================
# 通用Agent工厂
# ============================================================
def create_universal_agent(
    n_states: int,
    n_actions: int,
    learning_rate: float = 0.3,
    min_epsilon: float = 0.05,
    seed: Optional[int] = None
) -> SelfDrivenAgent:
    """创建一个通用Agent（不绑定特定环境）。

    Args:
        n_states: 状态空间大小
        n_actions: 动作空间大小
        learning_rate: 学习率
        min_epsilon: 最小探索率
        seed: 随机种子

    Returns:
        配置好的SelfDrivenAgent
    """
    action_space = DiscreteActionSpace(
        n=n_actions,
        action_names=tuple(f"action_{i}" for i in range(n_actions))
    )
    agent = SelfDrivenAgent(
        action_space=action_space,
        n_states=n_states,
        learning_rate=learning_rate,
        min_epsilon=min_epsilon,
        seed=seed,
    )
    return agent


# ============================================================
# 多环境训练器
# ============================================================
class MultiEnvironmentTrainer:
    """多环境训练器：让Agent在多个环境中学习。

    第一性原理：
        一个"通用智能体"应该能学习多种任务。
        多环境训练器让同一个Agent在不同环境间切换学习，
        积累跨领域的技能。

    工作流：
    1. 注册多个(EnvironmentAdapter)
    2. 每轮在某个环境运行一个情节
    3. Agent的Q表按环境维度独立学习
    4. 跨环境的元学习（学习如何学习）共享
    """

    def __init__(self, agent: SelfDrivenAgent) -> None:
        self.agent = agent
        self._adapters: List[Tuple[str, EnvironmentAdapter]] = []
        self._env_index = 0

    def add_environment(self, name: str, adapter: EnvironmentAdapter) -> None:
        """添加一个训练环境。"""
        self._adapters.append((name, adapter))

    def train_one_episode(self) -> Tuple[str, Episode]:
        """在一个环境上训练一个情节（轮换环境）。"""
        if not self._adapters:
            raise RuntimeError("未注册任何环境")

        name, adapter = self._adapters[self._env_index]
        self._env_index = (self._env_index + 1) % len(self._adapters)

        episode = run_episode_universal(self.agent, adapter)
        return name, episode

    def train(self, n_episodes: int) -> List[Tuple[str, Episode]]:
        """训练多个情节。"""
        results = []
        for _ in range(n_episodes):
            results.append(self.train_one_episode())
        return results

    @property
    def n_environments(self) -> int:
        return len(self._adapters)
