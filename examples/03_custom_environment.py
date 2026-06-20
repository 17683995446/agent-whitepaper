"""示例03：自定义环境接入。

本示例演示：
1. 实现 EnvironmentProtocol 创建自定义环境
2. 实现 StateEncoder 编码状态
3. 用自驱Agent学习自定义任务

场景：简单温度调节——Agent学习调节温度到目标值。
运行：PYTHONPATH=src python examples/03_custom_environment.py
"""

from typing import Optional, Tuple

import numpy as np

from agent_whitepaper import SelfDrivenAgent
from agent_whitepaper.stage_a.action_space import DiscreteActionSpace
from agent_whitepaper.stage_i.environment_interface import (
    EnvironmentProtocol,
    StateEncoder,
)


# ============================================================
# 自定义环境：温度调节
# ============================================================
class TemperatureEnv(EnvironmentProtocol):
    """温度调节环境。

    Agent通过"加热"/"冷却"/"保持"动作将温度调节到目标值。
    - 状态：当前温度（离散化为10档）
    - 动作：0=冷却, 1=保持, 2=加热
    - 奖励：接近目标 +1，到达目标 +10
    """

    def __init__(self, target: int = 7, max_steps: int = 20, seed: Optional[int] = None) -> None:
        self.target = target
        self.max_steps = max_steps
        self._rng = np.random.RandomState(seed)
        self._temp = 0
        self._steps = 0
        self._action_space = DiscreteActionSpace(
            n=3, action_names=("cool", "hold", "heat")
        )

    @property
    def action_space(self) -> DiscreteActionSpace:
        return self._action_space

    @property
    def observation_dim(self) -> int:
        return 10  # 温度one-hot编码

    @property
    def n_states(self) -> int:
        return 10  # 10个离散温度档

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._temp = int(self._rng.randint(0, 10))
        self._steps = 0
        return self._encode()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if action == 0:  # cool
            self._temp = max(0, self._temp - 1)
        elif action == 2:  # heat
            self._temp = min(9, self._temp + 1)
        # action == 1: hold

        self._steps += 1
        distance = abs(self._temp - self.target)
        reward = 10.0 if distance == 0 else -float(distance)
        done = (distance == 0) or (self._steps >= self.max_steps)
        info = {"temp": self._temp, "target": self.target, "distance": distance}
        return self._encode(), reward, done, info

    def _encode(self) -> np.ndarray:
        obs = np.zeros(10)
        obs[self._temp] = 1.0
        return obs


# ============================================================
# 状态编码器
# ============================================================
class TemperatureEncoder(StateEncoder):
    """温度状态编码器：one-hot → 索引。"""

    def __init__(self, n_states: int = 10) -> None:
        self._n_states = n_states

    def encode(self, observation: np.ndarray) -> int:
        return int(np.argmax(observation))

    @property
    def n_states(self) -> int:
        return self._n_states


def main() -> None:
    print("=" * 60)
    print("  示例03：自定义环境接入")
    print("=" * 60)

    # 1. 创建自定义环境
    env = TemperatureEnv(target=7, seed=42)
    print("\n环境: 温度调节")
    print(f"  目标温度: {env.target}")
    print("  动作: 冷却(0)/保持(1)/加热(2)")
    print(f"  状态数: {env.n_states}")

    # 2. 创建Agent
    agent = SelfDrivenAgent(
        action_space=env.action_space,
        n_states=env.n_states,
        learning_rate=0.2,
        seed=42,
    )
    encoder = TemperatureEncoder()

    # 3. 训练
    print("\n训练中（50个情节）...")
    for i in range(50):
        obs = env.reset()
        state = encoder.encode(obs)
        episode_reward = 0.0
        done = False
        while not done:
            action, next_obs, reward, done, info = agent.perceive_and_act(
                obs, state, env
            )
            episode_reward += reward
            obs = next_obs
            state = encoder.encode(obs)
        if (i + 1) % 10 == 0:
            print(
                f"  情节 {i + 1:2d}: "
                f"奖励={episode_reward:+.1f} "
                f"成功率={agent.stats.success_rate:.1%}"
            )

    # 4. 测试最终策略
    print("\n" + "=" * 60)
    print("  最终策略测试")
    print("=" * 60)
    obs = env.reset(seed=0)
    state = encoder.encode(obs)
    print(f"  初始温度: {state}")
    actions = ["冷却", "保持", "加热"]
    done = False
    steps = 0
    while not done:
        action, obs, reward, done, info = agent.perceive_and_act(obs, state, env)
        state = encoder.encode(obs)
        steps += 1
        print(f"  步骤{steps}: {actions[action]} → 温度={state}, 奖励={reward:+.1f}")
    print(f"\n  结果: {'成功' if info['distance'] == 0 else '失败'} ({steps}步)")


if __name__ == "__main__":
    main()
