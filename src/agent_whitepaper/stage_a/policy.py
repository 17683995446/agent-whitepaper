"""阶段A：单Agent闭环 - 基于驱力的策略映射。

第一性原理：
    策略 π(a|s) 是从状态到动作分布的映射。
    本系统的核心创新：用三大驱力（安全/社会/探索）调制策略。

    驱力如何影响策略：
    - 安全驱力高 → 倾向选择"降低自由能"的动作（利用已知好动作）
    - 探索驱力高 → 倾向选择"新颖"的动作（尝试未知）
    - 社会驱力高 → 倾向选择"稀疏表征"对应的动作（在多agent时影响协作）

    实现方式：
    1. Q值估计：从记忆中学习每个 (s,a) 的价值
    2. 驱力调制：用驱力强度调整ε-greedy的ε
       ε = exploration_strength（探索驱力直接决定探索率）
    3. 动作选择：以 (1-ε) 选最优动作，以 ε 随机探索
"""

from dataclasses import dataclass

import numpy as np

from .action_space import DiscreteActionSpace


# ============================================================
# Q表（简单表格法Q-learning）
# ============================================================
class QTable:
    """Q值表。

    对于离散状态空间较小的情况，表格法比神经网络更可靠。
    状态用离散索引表示（如GridWorld的格子编号）。
    """

    def __init__(self, n_states: int, n_actions: int, learning_rate: float = 0.1) -> None:
        if n_states < 1 or n_actions < 1:
            raise ValueError("状态数和动作数必须 >= 1")
        if not (0 < learning_rate <= 1):
            raise ValueError(f"学习率必须 ∈ (0,1]，收到 {learning_rate}")
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self._table = np.zeros((n_states, n_actions))

    def get(self, state: int) -> np.ndarray:
        """获取状态对应的Q值向量。"""
        if not (0 <= state < self.n_states):
            raise ValueError(f"状态 {state} 超出范围 [0,{self.n_states})")
        return self._table[state].copy()

    def best_action(self, state: int) -> int:
        """获取最优动作。"""
        return int(np.argmax(self._table[state]))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        gamma: float = 0.99
    ) -> float:
        """Q-learning更新。

        Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a')·(1-done) - Q(s,a)]

        Returns:
            TD误差（用于优先级回放）
        """
        if not (0 <= state < self.n_states):
            raise ValueError(f"状态 {state} 超出范围")
        if not (0 <= action < self.n_actions):
            raise ValueError(f"动作 {action} 超出范围")
        if not (0 <= next_state < self.n_states):
            raise ValueError(f"下一状态 {next_state} 超出范围")

        current_q = self._table[state, action]
        if done:
            target = reward
        else:
            target = reward + gamma * np.max(self._table[next_state])
        td_error = target - current_q
        self._table[state, action] += self.lr * td_error
        return float(td_error)

    def state_to_index(self, observation: np.ndarray, grid_size: int) -> int:
        """将GridWorld观测转换为状态索引。

        观测格式：[agent_one_hot (size*size), goal_one_hot (size*size)]
        状态 = agent_pos * size*size + goal_pos（简化为agent位置）
        """
        # 只用agent位置作为状态（goal固定时）
        agent_part = observation[:grid_size * grid_size]
        return int(np.argmax(agent_part))


# ============================================================
# 驱力状态
# ============================================================
@dataclass(frozen=True)
class DriveState:
    """三大驱力状态。

    Attributes:
        safety: 安全驱力强度 [0,1]
        social: 社会驱力强度 [0,1]
        exploration: 探索驱力强度 [0,1]
    """
    safety: float = 0.5
    social: float = 0.5
    exploration: float = 0.5

    def __post_init__(self) -> None:
        for name, val in [("safety", self.safety), ("social", self.social),
                          ("exploration", self.exploration)]:
            if not (0 <= val <= 1):
                raise ValueError(f"{name} 必须 ∈ [0,1]，收到 {val}")


# ============================================================
# 基于驱力的策略
# ============================================================
class DriveBasedPolicy:
    """基于驱力的ε-greedy策略。

    核心思想：
    - 探索驱力决定ε（探索率）
    - 安全驱力决定是否坚持已知好动作（safety高→更greedy）
    - 社会驱力在单agent中暂不直接影响，为多agent预留

    有效探索率：
        ε_effective = exploration * (1 - safety * 0.5)
    即：安全驱力越高，越倾向于利用（降低探索）
    """

    def __init__(
        self,
        action_space: DiscreteActionSpace,
        min_epsilon: float = 0.05,
        max_epsilon: float = 1.0
    ) -> None:
        if not (0 <= min_epsilon <= max_epsilon <= 1):
            raise ValueError("必须 0 <= min_epsilon <= max_epsilon <= 1")
        self.action_space = action_space
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon

    def compute_epsilon(self, drives: DriveState) -> float:
        """根据驱力计算有效探索率。"""
        # 探索驱力提升ε，安全驱力降低ε
        base_epsilon = drives.exploration * (1.0 - drives.safety * 0.5)
        return float(np.clip(base_epsilon, self.min_epsilon, self.max_epsilon))

    def select_action(
        self,
        q_table: QTable,
        state: int,
        drives: DriveState,
        rng: np.random.RandomState
    ) -> int:
        """选择动作。"""
        epsilon = self.compute_epsilon(drives)
        if rng.rand() < epsilon:
            # 探索
            return self.action_space.sample(rng)
        else:
            # 利用
            return q_table.best_action(state)

    def update_drives_from_free_energy(
        self,
        current_drives: DriveState,
        free_energy: float,
        free_energy_history: list,
        reward: float
    ) -> DriveState:
        """根据自由能和奖励更新驱力。

        - 自由能高 → 安全驱力上升（感到不安全）
        - 自由能下降 → 探索驱力上升（探索有成效）
        - 奖励高 → 安全驱力上升（感到安全）
        """
        safety = current_drives.safety
        exploration = current_drives.exploration
        social = current_drives.social

        # 安全驱力：自由能高则降低（不安全），奖励高则升高（安全）
        safety = 0.9 * safety + 0.1 * (1.0 / (1.0 + free_energy))
        safety = float(np.clip(safety + 0.01 * reward, 0, 1))

        # 探索驱力：自由能下降则上升（探索有效）
        if len(free_energy_history) >= 2:
            fe_delta = free_energy_history[-1] - free_energy_history[-2]
            if fe_delta < 0:  # 自由能下降
                exploration = float(np.clip(exploration + 0.05, 0, 1))
            else:  # 自由能上升
                exploration = float(np.clip(exploration - 0.02, 0, 1))

        return DriveState(safety=safety, social=social, exploration=exploration)
