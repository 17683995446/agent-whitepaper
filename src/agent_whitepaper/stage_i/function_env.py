"""阶段I-2：函数调用环境 - Agent学习操作"工具"完成任务。

第一性原理：
    真实世界的任务不是"在网格里移动"，而是"调用工具达成目标"。
    例如数据处理流水线：fetch → clean → transform → validate → output

    函数调用环境模拟这种场景：
    - 状态：已完成的步骤（位图编码）
    - 动作：调用某个函数（每个函数是一个动作）
    - 转移：调用正确的函数 → 标记该步骤完成
    - 奖励：完成所有步骤 +1，调用无效函数 -0.1
    - 目标：学会正确的函数调用顺序

    这证明了Agent能学习**非空间导航**的任务——
    从"网格世界"泛化到"工具使用"，这是实用性的关键跃迁。

    环境支持两种模式：
    1. 严格顺序：必须按固定顺序调用（如流水线）
    2. 自由顺序：任意顺序都行（如独立任务集合）
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from agent_whitepaper.stage_a.action_space import DiscreteActionSpace

from .environment_interface import EnvironmentProtocol


# ============================================================
# 函数调用环境
# ============================================================
class FunctionCallEnvironment(EnvironmentProtocol):
    """函数调用环境：Agent学习按正确顺序调用函数。

    场景：有N个函数（步骤），Agent需要学会调用它们完成任务。
    - 严格模式：必须按顺序 0→1→2→...→N-1 调用
    - 自由模式：任意顺序都行，只要全部调用过

    状态编码：位图，第i位=1表示第i个函数已调用
    观测：位图的one-hot编码（长度=N，每位置0/1）

    奖励设计：
    - 调用正确的下一个函数：+0.1（正向引导）
    - 完成所有函数：+1.0（终极目标）
    - 调用已完成的函数：-0.05（浪费）
    - 调用乱序函数（严格模式）：-0.1（惩罚）
    """

    def __init__(
        self,
        n_functions: int = 5,
        ordered: bool = True,
        max_steps: int = 50,
        seed: Optional[int] = None,
        function_names: Optional[List[str]] = None,
    ):
        if n_functions < 2:
            raise ValueError("n_functions 必须 >= 2")
        if n_functions > 15:
            raise ValueError("n_functions 过大（>15），状态空间爆炸")

        self.n_functions = n_functions
        self.ordered = ordered
        self.max_steps = max_steps
        self._rng = np.random.RandomState(seed)

        # 函数名称
        if function_names is None:
            function_names = [f"func_{i}" for i in range(n_functions)]
        self.function_names = function_names

        # 动作空间
        self._action_space = DiscreteActionSpace(
            n=n_functions,
            action_names=tuple(function_names),
        )

        # 状态：哪些函数已调用
        self._completed = np.zeros(n_functions, dtype=np.int8)
        self._step_count = 0
        self._next_expected = 0  # 严格模式下下一个期望的函数

    @property
    def action_space(self) -> DiscreteActionSpace:
        return self._action_space

    @property
    def observation_dim(self) -> int:
        return self.n_functions

    @property
    def n_states(self) -> int:
        return 2 ** self.n_functions

    def _encode_observation(self) -> np.ndarray:
        """编码状态为观测向量（位图）。"""
        return self._completed.astype(np.float64)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """重置环境。"""
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._completed = np.zeros(self.n_functions, dtype=np.int8)
        self._step_count = 0
        self._next_expected = 0
        return self._encode_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """执行函数调用。"""
        if not self._action_space.contains(action):
            raise ValueError(f"非法动作 {action}")

        self._step_count += 1
        reward = 0.0
        info = {"function": self.function_names[action]}

        # 检查是否已完成所有函数
        all_done = bool(np.all(self._completed))

        if all_done:
            # 已经完成，不应再调用
            reward = -0.05
            info["status"] = "already_complete"
        elif self._completed[action]:
            # 重复调用已完成的函数
            reward = -0.05
            info["status"] = "duplicate"
        elif self.ordered and action != self._next_expected:
            # 严格模式下乱序调用
            reward = -0.1
            info["status"] = "out_of_order"
            info["expected"] = self._next_expected
        else:
            # 正确调用
            self._completed[action] = 1
            reward = 0.1
            info["status"] = "correct"

            if self.ordered:
                # 更新下一个期望
                self._next_expected = action + 1

            # 检查是否全部完成
            if np.all(self._completed):
                reward = 1.0
                info["status"] = "complete"
                info["reached_goal"] = True

        # 终止条件
        done = bool(np.all(self._completed)) or self._step_count >= self.max_steps
        info["grid_size"] = self.n_functions  # 兼容现有Agent代码
        info["steps_taken"] = self._step_count

        return self._encode_observation(), reward, done, info

    @property
    def is_complete(self) -> bool:
        """是否所有函数都已调用。"""
        return bool(np.all(self._completed))

    @property
    def completion_rate(self) -> float:
        """完成率。"""
        return float(np.mean(self._completed))

    def get_progress(self) -> Dict:
        """获取进度信息。"""
        return {
            "completed": self._completed.tolist(),
            "completion_rate": self.completion_rate,
            "steps_taken": self._step_count,
            "is_complete": self.is_complete,
            "next_expected": self._next_expected if self.ordered else None,
        }
