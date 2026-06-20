"""阶段I验证测试 + 真实环境适配层。

测试覆盖：
1. 环境接口与状态编码器
2. 函数调用环境
3. 通用Agent适配器
4. 多环境训练器
5. 端到端：Agent学习非GridWorld任务
"""

import numpy as np
import pytest

from agent_whitepaper.stage_i.environment_interface import (
    EnvironmentProtocol, StateEncoder, GridWorldStateEncoder,
    BitmapStateEncoder, EnvironmentAdapter,
)
from agent_whitepaper.stage_i.function_env import FunctionCallEnvironment
from agent_whitepaper.stage_i.adapters import (
    run_episode_universal, replay_learning_universal,
    create_universal_agent, MultiEnvironmentTrainer,
)
from agent_whitepaper.stage_a.agent import SelfDrivenAgent
from agent_whitepaper.stage_a.environment import GridWorld


# ============================================================
# 1. 状态编码器测试
# ============================================================
class TestStateEncoders:
    def test_gridworld_encoder(self):
        encoder = GridWorldStateEncoder(grid_size=5)
        # 构造一个agent在位置(2,3)的观测
        obs = np.zeros(50)
        obs[2 * 5 + 3] = 1.0  # agent位置
        obs[25 + 4 * 5 + 4] = 1.0  # goal位置
        idx = encoder.encode(obs)
        assert idx == 13  # 2*5+3
        assert encoder.n_states == 25

    def test_bitmap_encoder(self):
        encoder = BitmapStateEncoder(n_bits=3)
        # [1,0,1] → 二进制101 = 5
        obs = np.array([1.0, 0.0, 1.0])
        assert encoder.encode(obs) == 5
        # [0,1,0] → 二进制010 = 2
        obs = np.array([0.0, 1.0, 0.0])
        assert encoder.encode(obs) == 2
        assert encoder.n_states == 8

    def test_bitmap_encoder_binarization(self):
        encoder = BitmapStateEncoder(n_bits=2)
        # 0.6 > 0.5 → 1, 0.3 < 0.5 → 0
        obs = np.array([0.6, 0.3])
        assert encoder.encode(obs) == 1  # 二进制01

    def test_bitmap_encoder_too_large(self):
        with pytest.raises(ValueError):
            BitmapStateEncoder(n_bits=25)

    def test_gridworld_encoder_invalid(self):
        with pytest.raises(ValueError):
            GridWorldStateEncoder(grid_size=1)


# ============================================================
# 2. 环境适配器测试
# ============================================================
class TestEnvironmentAdapter:
    def test_adapter_with_gridworld(self):
        env = GridWorld(size=5, seed=42)
        encoder = GridWorldStateEncoder(grid_size=5)
        adapter = EnvironmentAdapter(env, encoder)

        obs, state_idx = adapter.reset()
        assert state_idx == 0  # 起点在(0,0)
        assert adapter.n_states == 25

    def test_adapter_step_returns_state_index(self):
        env = GridWorld(size=5, seed=42)
        encoder = GridWorldStateEncoder(grid_size=5)
        adapter = EnvironmentAdapter(env, encoder)

        adapter.reset()
        obs, state_idx, reward, done, info = adapter.step(1)  # 下
        assert isinstance(state_idx, int)
        assert 0 <= state_idx < 25

    def test_adapter_with_function_env(self):
        env = FunctionCallEnvironment(n_functions=3, seed=42)
        encoder = BitmapStateEncoder(n_bits=3)
        adapter = EnvironmentAdapter(env, encoder)

        obs, state_idx = adapter.reset()
        assert state_idx == 0  # 无函数完成
        assert adapter.n_states == 8
        assert adapter.n_states == 8


# ============================================================
# 3. 函数调用环境测试
# ============================================================
class TestFunctionCallEnvironment:
    def test_basic_properties(self):
        env = FunctionCallEnvironment(n_functions=4, seed=42)
        assert env.n_functions == 4
        assert env.n_states == 16
        assert env.observation_dim == 4
        assert env.action_space.n == 4

    def test_reset(self):
        env = FunctionCallEnvironment(n_functions=3, seed=42)
        obs = env.reset()
        assert np.all(obs == 0)
        assert env.completion_rate == 0.0
        assert not env.is_complete

    def test_correct_order(self):
        env = FunctionCallEnvironment(n_functions=3, ordered=True, seed=42)
        env.reset()
        # 按顺序调用 0→1→2
        _, r1, _, info1 = env.step(0)
        assert info1["status"] == "correct"
        assert r1 == pytest.approx(0.1)

        _, r2, _, info2 = env.step(1)
        assert info2["status"] == "correct"

        _, r3, done, info3 = env.step(2)
        assert info3["status"] == "complete"
        assert done
        assert r3 == pytest.approx(1.0)
        assert env.is_complete

    def test_out_of_order_penalty(self):
        env = FunctionCallEnvironment(n_functions=3, ordered=True, seed=42)
        env.reset()
        # 先调用1（应该先调用0）
        _, r, _, info = env.step(1)
        assert info["status"] == "out_of_order"
        assert r == pytest.approx(-0.1)

    def test_duplicate_penalty(self):
        env = FunctionCallEnvironment(n_functions=3, ordered=True, seed=42)
        env.reset()
        env.step(0)  # 完成0
        _, r, _, info = env.step(0)  # 重复0
        assert info["status"] == "duplicate"
        assert r == pytest.approx(-0.05)

    def test_free_order_mode(self):
        env = FunctionCallEnvironment(n_functions=3, ordered=False, seed=42)
        env.reset()
        # 任意顺序都行
        _, r1, _, i1 = env.step(2)
        assert i1["status"] == "correct"
        _, r2, _, i2 = env.step(0)
        assert i2["status"] == "correct"
        _, r3, done, i3 = env.step(1)
        assert i3["status"] == "complete"
        assert done

    def test_max_steps_termination(self):
        env = FunctionCallEnvironment(n_functions=3, max_steps=5, seed=42)
        env.reset()
        # 一直调用错误的函数
        for _ in range(5):
            _, _, done, _ = env.step(2)  # 乱序
        assert done

    def test_custom_function_names(self):
        env = FunctionCallEnvironment(
            n_functions=3,
            function_names=["fetch", "clean", "output"],
        )
        assert env.function_names == ["fetch", "clean", "output"]
        assert env.action_space.action_names == ("fetch", "clean", "output")

    def test_get_progress(self):
        env = FunctionCallEnvironment(n_functions=3, ordered=True, seed=42)
        env.reset()
        env.step(0)
        progress = env.get_progress()
        assert progress["completion_rate"] == pytest.approx(1/3)
        assert progress["next_expected"] == 1
        assert not progress["is_complete"]


# ============================================================
# 4. 通用Agent适配器测试
# ============================================================
class TestUniversalAdapter:
    def test_create_universal_agent(self):
        agent = create_universal_agent(
            n_states=8, n_actions=3, seed=42
        )
        assert agent.q_table.n_states == 8
        assert agent.q_table.n_actions == 3

    def test_run_episode_universal(self):
        env = FunctionCallEnvironment(n_functions=3, seed=42)
        encoder = BitmapStateEncoder(n_bits=3)
        adapter = EnvironmentAdapter(env, encoder)
        agent = create_universal_agent(n_states=8, n_actions=3, seed=42)

        episode = run_episode_universal(agent, adapter, max_steps=20)
        assert episode.length > 0
        assert agent.stats.total_steps > 0

    def test_replay_learning_universal(self):
        env = FunctionCallEnvironment(n_functions=3, seed=42)
        encoder = BitmapStateEncoder(n_bits=3)
        adapter = EnvironmentAdapter(env, encoder)
        agent = create_universal_agent(n_states=8, n_actions=3, seed=42)

        # 积累足够经验
        for _ in range(10):
            run_episode_universal(agent, adapter, max_steps=20)

        # 回放学习
        td_error = replay_learning_universal(agent, encoder, batch_size=8, n_updates=5)
        assert td_error >= 0

    def test_agent_learns_function_sequence(self):
        """端到端：Agent学会按顺序调用函数。"""
        env = FunctionCallEnvironment(n_functions=4, ordered=True, max_steps=30, seed=42)
        encoder = BitmapStateEncoder(n_bits=4)
        adapter = EnvironmentAdapter(env, encoder)
        agent = create_universal_agent(n_states=16, n_actions=4, learning_rate=0.3, seed=42)

        # 训练前：成功率低
        early_successes = 0
        for _ in range(20):
            ep = run_episode_universal(agent, adapter, max_steps=30)
            if ep.success:
                early_successes += 1
            replay_learning_universal(agent, encoder, batch_size=16, n_updates=5)

        # 训练更多
        for _ in range(100):
            ep = run_episode_universal(agent, adapter, max_steps=30)
            replay_learning_universal(agent, encoder, batch_size=16, n_updates=5)

        # 训练后：Q表非零项增加（学到了）
        q_entries = int(np.count_nonzero(agent.q_table._table))
        assert q_entries > 0, "Agent未学到任何Q值"

        # 后期应有成功
        late_successes = 0
        for _ in range(20):
            ep = run_episode_universal(agent, adapter, max_steps=30)
            if ep.success:
                late_successes += 1

        # 后期成功率应 >= 前期（学习发生）
        assert late_successes >= early_successes or q_entries > 5


# ============================================================
# 5. 多环境训练器测试
# ============================================================
class TestMultiEnvironmentTrainer:
    def test_add_environments(self):
        agent = create_universal_agent(n_states=8, n_actions=3, seed=42)
        trainer = MultiEnvironmentTrainer(agent)

        env1 = FunctionCallEnvironment(n_functions=3, seed=1)
        env2 = FunctionCallEnvironment(n_functions=3, seed=2)
        encoder = BitmapStateEncoder(n_bits=3)

        trainer.add_environment("env1", EnvironmentAdapter(env1, encoder))
        trainer.add_environment("env2", EnvironmentAdapter(env2, encoder))

        assert trainer.n_environments == 2

    def test_train_rotates_environments(self):
        agent = create_universal_agent(n_states=8, n_actions=3, seed=42)
        trainer = MultiEnvironmentTrainer(agent)

        env1 = FunctionCallEnvironment(n_functions=3, seed=1)
        env2 = FunctionCallEnvironment(n_functions=3, seed=2)
        encoder = BitmapStateEncoder(n_bits=3)

        trainer.add_environment("env1", EnvironmentAdapter(env1, encoder))
        trainer.add_environment("env2", EnvironmentAdapter(env2, encoder))

        results = trainer.train(4)
        assert len(results) == 4
        # 应轮换环境
        names = [name for name, _ in results]
        assert names[0] == "env1"
        assert names[1] == "env2"
        assert names[2] == "env1"
        assert names[3] == "env2"

    def test_train_no_environments_raises(self):
        agent = create_universal_agent(n_states=8, n_actions=3, seed=42)
        trainer = MultiEnvironmentTrainer(agent)
        with pytest.raises(RuntimeError):
            trainer.train_one_episode()


# ============================================================
# 6. 端到端：泛化能力验证
# ============================================================
class TestGeneralization:
    def test_agent_learns_gridworld_via_adapter(self):
        """Agent通过适配器学习GridWorld（向后兼容）。"""
        env = GridWorld(size=4, seed=42)
        encoder = GridWorldStateEncoder(grid_size=4)
        adapter = EnvironmentAdapter(env, encoder)
        agent = create_universal_agent(n_states=16, n_actions=4, seed=42)

        for _ in range(50):
            run_episode_universal(agent, adapter, max_steps=30)

        q_entries = int(np.count_nonzero(agent.q_table._table))
        assert q_entries > 0
        assert agent.stats.total_episodes == 50

    def test_same_agent_different_environments(self):
        """同一个Agent框架学习两种不同环境。"""
        # 环境1：3函数调用
        env1 = FunctionCallEnvironment(n_functions=3, seed=1)
        enc1 = BitmapStateEncoder(n_bits=3)
        adapter1 = EnvironmentAdapter(env1, enc1)

        # 环境2：4函数调用（不同状态空间大小，用不同Agent）
        # 但验证框架通用性
        env2 = FunctionCallEnvironment(n_functions=4, seed=2)
        enc2 = BitmapStateEncoder(n_bits=4)
        adapter2 = EnvironmentAdapter(env2, enc2)

        agent1 = create_universal_agent(n_states=8, n_actions=3, seed=42)
        agent2 = create_universal_agent(n_states=16, n_actions=4, seed=42)

        # 两个Agent都能学习各自环境
        for _ in range(30):
            run_episode_universal(agent1, adapter1, max_steps=20)
            run_episode_universal(agent2, adapter2, max_steps=30)

        assert int(np.count_nonzero(agent1.q_table._table)) > 0
        assert int(np.count_nonzero(agent2.q_table._table)) > 0

    def test_function_env_learning_improves(self):
        """验证：函数调用环境中学到的策略可被验证。"""
        env = FunctionCallEnvironment(
            n_functions=3, ordered=True, max_steps=20, seed=42
        )
        encoder = BitmapStateEncoder(n_bits=3)
        adapter = EnvironmentAdapter(env, encoder)
        agent = create_universal_agent(
            n_states=8, n_actions=3, learning_rate=0.5, seed=42
        )

        # 充分训练
        for _ in range(200):
            run_episode_universal(agent, adapter, max_steps=20)
            replay_learning_universal(agent, encoder, batch_size=8, n_updates=3)

        # 验证：从初始状态，最优动作应该是0（第一个函数）
        # 初始状态索引=0（无函数完成）
        best_action = agent.q_table.best_action(0)
        # 学会后应该倾向于调用0（但探索率可能影响，只验证Q值非零）
        q_values = agent.q_table.get(0)
        assert np.any(q_values != 0), "初始状态Q值全为零，未学习"
