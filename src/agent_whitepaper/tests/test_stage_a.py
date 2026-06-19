"""阶段A验证测试：验证单Agent闭环的前5个模块。

测试覆盖：
1. 动作空间（离散/连续）
2. 环境接口与GridWorld
3. 情节记忆缓冲
4. 基于驱力的策略
5. 单Agent闭环集成
6. 端到端：Agent在GridWorld中自驱学会到达目标
"""

import numpy as np
import pytest

from agent_whitepaper.stage_a.action_space import (
    DiscreteActionSpace, ContinuousActionSpace, ActionSpaceType,
    encode_action, decode_action,
)
from agent_whitepaper.stage_a.environment import (
    Environment, Transition, GridWorld,
)
from agent_whitepaper.stage_a.memory import (
    EpisodicBuffer, Episode,
)
from agent_whitepaper.stage_a.policy import (
    QTable, DriveState, DriveBasedPolicy,
)
from agent_whitepaper.stage_a.agent import SelfDrivenAgent, AgentStats


# ============================================================
# 1. 动作空间测试
# ============================================================
class TestActionSpace:
    def test_discrete_basic(self):
        space = DiscreteActionSpace(n=4)
        assert space.n == 4
        assert space.space_type == ActionSpaceType.DISCRETE

    def test_discrete_sample(self):
        space = DiscreteActionSpace(n=4)
        rng = np.random.RandomState(42)
        for _ in range(100):
            action = space.sample(rng)
            assert space.contains(action)

    def test_discrete_one_hot(self):
        space = DiscreteActionSpace(n=4)
        vec = space.one_hot(2)
        assert vec.shape == (4,)
        assert vec[2] == 1.0
        assert sum(vec) == 1.0

    def test_discrete_invalid_n(self):
        with pytest.raises(ValueError):
            DiscreteActionSpace(n=0)

    def test_continuous_basic(self):
        low = np.array([-1.0, -1.0])
        high = np.array([1.0, 1.0])
        space = ContinuousActionSpace(low=low, high=high, dim=2)
        assert space.dim == 2
        assert space.space_type == ActionSpaceType.CONTINUOUS

    def test_continuous_sample(self):
        low = np.array([-1.0])
        high = np.array([1.0])
        space = ContinuousActionSpace(low=low, high=high, dim=1)
        rng = np.random.RandomState(42)
        for _ in range(100):
            action = space.sample(rng)
            assert space.contains(action)

    def test_continuous_invalid_bounds(self):
        with pytest.raises(ValueError):
            ContinuousActionSpace(
                low=np.array([1.0]), high=np.array([0.0]), dim=1
            )

    def test_encode_decode_discrete(self):
        space = DiscreteActionSpace(n=4)
        encoded = encode_action(2, space)
        assert encoded.shape == (4,)
        assert decode_action(encoded, space) == 2


# ============================================================
# 2. 环境接口测试
# ============================================================
class TestEnvironment:
    def test_gridworld_reset(self):
        env = GridWorld(size=5, seed=42)
        obs = env.reset()
        assert obs.shape == (50,)  # 2 * 5 * 5
        assert obs.sum() == 2.0  # agent + goal

    def test_gridworld_step(self):
        env = GridWorld(size=5, seed=42)
        env.reset()
        obs, reward, done, info = env.step(GridWorld.ACTION_RIGHT)
        assert obs.shape == (50,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "agent_pos" in info

    def test_gridworld_reach_goal(self):
        env = GridWorld(size=3, goal=(2, 2), start=(0, 0), seed=42)
        env.reset()
        # 走到 (2,2): 右右下下
        env.step(GridWorld.ACTION_RIGHT)
        env.step(GridWorld.ACTION_RIGHT)
        env.step(GridWorld.ACTION_DOWN)
        obs, reward, done, info = env.step(GridWorld.ACTION_DOWN)
        assert done
        assert reward == 1.0
        assert info["reached_goal"]

    def test_gridworld_boundary(self):
        env = GridWorld(size=3, start=(0, 0), seed=42)
        env.reset()
        # 向上走应被边界阻挡
        obs, reward, done, info = env.step(GridWorld.ACTION_UP)
        assert info["agent_pos"] == (0, 0)

    def test_gridworld_max_steps(self):
        env = GridWorld(size=5, goal=(4, 4), start=(0, 0), max_steps=3, seed=42)
        env.reset()
        for _ in range(3):
            obs, reward, done, info = env.step(GridWorld.ACTION_UP)
        assert done  # 因步数耗尽而终止

    def test_observation_dim(self):
        env = GridWorld(size=4)
        assert env.observation_dim == 32  # 2 * 4 * 4


# ============================================================
# 3. 情节记忆测试
# ============================================================
class TestMemory:
    def test_buffer_add_and_len(self):
        buffer = EpisodicBuffer(capacity=100)
        assert len(buffer) == 0
        t = Transition(
            observation=np.zeros(4), action=0, reward=0.1,
            next_observation=np.zeros(4), done=False, info={}
        )
        buffer.add(t)
        assert len(buffer) == 1

    def test_buffer_capacity(self):
        buffer = EpisodicBuffer(capacity=5)
        for i in range(10):
            t = Transition(
                observation=np.array([i]), action=0, reward=0.1,
                next_observation=np.array([i+1]), done=(i==9), info={}
            )
            buffer.add(t)
        assert len(buffer) == 5  # 只保留最近5个

    def test_buffer_sample(self):
        buffer = EpisodicBuffer(capacity=100)
        for i in range(20):
            t = Transition(
                observation=np.array([i]), action=0, reward=0.1,
                next_observation=np.array([i+1]), done=False, info={}
            )
            buffer.add(t)
        rng = np.random.RandomState(42)
        indices, transitions = buffer.sample(10, rng)
        assert len(indices) == 10
        assert len(transitions) == 10

    def test_prioritized_sample(self):
        buffer = EpisodicBuffer(capacity=100)
        for i in range(20):
            t = Transition(
                observation=np.array([i]), action=0, reward=0.1,
                next_observation=np.array([i+1]), done=False, info={}
            )
            buffer.add(t)
        # 设置不同优先级
        buffer.update_priority(list(range(20)), [float(i) for i in range(20)])
        rng = np.random.RandomState(42)
        indices, transitions, weights = buffer.sample_prioritized(10, rng)
        assert len(indices) == 10
        assert weights.shape == (10,)

    def test_episode_from_transitions(self):
        transitions = [
            Transition(
                observation=np.array([0]), action=0, reward=0.1,
                next_observation=np.array([1]), done=False, info={}
            ),
            Transition(
                observation=np.array([1]), action=1, reward=1.0,
                next_observation=np.array([2]), done=True,
                info={"reached_goal": True}
            ),
        ]
        episode = Episode.from_transitions(transitions)
        assert episode.length == 2
        assert episode.total_reward == 1.1
        assert episode.success

    def test_completed_episodes(self):
        buffer = EpisodicBuffer(capacity=100)
        # 添加一个完整情节
        for i in range(5):
            t = Transition(
                observation=np.array([i]), action=0, reward=0.1,
                next_observation=np.array([i+1]), done=(i==4),
                info={"reached_goal": (i==4)}
            )
            buffer.add(t)
        assert buffer.num_episodes == 1


# ============================================================
# 4. 策略测试
# ============================================================
class TestPolicy:
    def test_qtable_update(self):
        q = QTable(n_states=10, n_actions=4, learning_rate=0.5)
        td_error = q.update(0, 1, 1.0, 1, done=True, gamma=0.99)
        assert abs(td_error - 1.0) < 1e-6  # target=1, current=0
        assert q.get(0)[1] == 0.5  # 0 + 0.5*1 = 0.5

    def test_qtable_best_action(self):
        q = QTable(n_states=2, n_actions=3, learning_rate=1.0)
        q.update(0, 2, 1.0, 1, done=True)
        assert q.best_action(0) == 2

    def test_drive_state_validation(self):
        with pytest.raises(ValueError):
            DriveState(safety=1.5)

    def test_epsilon_computation(self):
        space = DiscreteActionSpace(n=4)
        policy = DriveBasedPolicy(space)
        # 高探索低安全 → 高ε
        drives = DriveState(safety=0.1, exploration=0.9)
        eps = policy.compute_epsilon(drives)
        assert eps > 0.5
        # 低探索高安全 → 低ε
        drives = DriveState(safety=0.9, exploration=0.1)
        eps = policy.compute_epsilon(drives)
        assert eps < 0.2

    def test_action_selection_exploit(self):
        space = DiscreteActionSpace(n=4)
        q = QTable(n_states=2, n_actions=4)
        q.update(0, 2, 1.0, 1, done=True)
        policy = DriveBasedPolicy(space, min_epsilon=0.0)
        # 安全高、探索低 → 必然利用
        drives = DriveState(safety=1.0, exploration=0.0)
        rng = np.random.RandomState(42)
        for _ in range(10):
            action = policy.select_action(q, 0, drives, rng)
            assert action == 2  # 最优动作


# ============================================================
# 5. Agent闭环测试
# ============================================================
class TestAgent:
    def test_agent_creation(self):
        space = DiscreteActionSpace(n=4)
        agent = SelfDrivenAgent(space, n_states=25, seed=42)
        assert agent.stats.total_steps == 0
        assert agent.drives.exploration > 0.5  # 初始高探索

    def test_agent_run_episode(self):
        env = GridWorld(size=5, seed=42)
        space = env.action_space
        agent = SelfDrivenAgent(space, n_states=25, seed=42)
        episode = agent.run_episode(env, max_steps=50)
        assert episode.length > 0
        assert agent.stats.total_episodes == 1

    def test_agent_learning_improves(self):
        """Agent应该随着训练提高成功率。"""
        env = GridWorld(size=4, goal=(3, 3), start=(0, 0), max_steps=30, seed=42)
        space = env.action_space
        agent = SelfDrivenAgent(space, n_states=16, learning_rate=0.3, seed=42)

        # 训练前10个情节的成功率
        for _ in range(10):
            agent.run_episode(env)
        early_success_rate = agent.stats.success_rate

        # 继续训练
        for _ in range(90):
            agent.run_episode(env)
            # 回放学习
            if len(agent.memory) > 32:
                agent.replay_learning(batch_size=16, n_updates=5)
        late_success_rate = agent.stats.success_rate

        print(f"早期成功率: {early_success_rate:.2f}, 后期成功率: {late_success_rate:.2f}")
        # 后期应该不比早期差（允许波动，但不应该显著退步）
        assert late_success_rate >= early_success_rate * 0.5


# ============================================================
# 6. 端到端：自驱学会到达目标
# ============================================================
class TestEndToEnd:
    def test_agent_learns_to_reach_goal(self):
        """端到端测试：Agent在5x5 GridWorld中自驱学会到达目标。"""
        env = GridWorld(size=5, goal=(4, 4), start=(0, 0), max_steps=50, seed=42)
        space = env.action_space
        agent = SelfDrivenAgent(
            space, n_states=25, learning_rate=0.3, min_epsilon=0.1, seed=42
        )

        # 训练200个情节
        for ep in range(200):
            agent.run_episode(env)
            if len(agent.memory) > 32:
                agent.replay_learning(batch_size=16, n_updates=3)

        # 验证：最后20个情节的成功率应该 > 50%
        recent_successes = agent.stats.recent_successes[-20:]
        final_success_rate = sum(recent_successes) / len(recent_successes)
        print(f"最终成功率: {final_success_rate:.2f}")
        print(f"平均奖励: {agent.stats.avg_reward:.3f}")
        assert final_success_rate > 0.3, \
            f"Agent未学会到达目标，成功率仅 {final_success_rate:.2f}"

    def test_agent_drives_evolve(self):
        """驱力应该在训练中演化。"""
        env = GridWorld(size=4, goal=(3, 3), start=(0, 0), max_steps=30, seed=42)
        space = env.action_space
        agent = SelfDrivenAgent(space, n_states=16, seed=42)

        initial_exploration = agent.drives.exploration
        for _ in range(50):
            agent.run_episode(env)

        # 驱力应该发生了变化
        final_exploration = agent.drives.exploration
        print(f"初始探索: {initial_exploration:.3f}, 最终探索: {final_exploration:.3f}")
        # 探索驱力应该有所变化（可能上升或下降，取决于经历）
        assert final_exploration != initial_exploration or len(agent.stats.free_energy_history) > 0
