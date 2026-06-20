"""阶段A：单Agent闭环 - 动作空间定义。

第一性原理：
    Agent通过动作作用于环境，环境返回新观测与奖励。
    动作空间定义了Agent"能做什么"，是感知-行动闭环的输出端。

    动作空间分为：
    - 离散动作空间：有限可枚举的动作集合（如上下左右）
    - 连续动作空间：实数向量空间中的动作（如关节力矩）

    本模块采用纯函数式设计：ActionSpace是不可变描述，
    动作采样通过显式传入随机数生成器实现，保证可复现性。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np


# ============================================================
# 动作空间类型枚举
# ============================================================
class ActionSpaceType(Enum):
    """动作空间类型。"""
    DISCRETE = "discrete"    # 离散动作空间
    CONTINUOUS = "continuous"  # 连续动作空间


# ============================================================
# 动作空间基类（联合类型）
# ============================================================
@dataclass(frozen=True)
class DiscreteActionSpace:
    """离散动作空间。

    Attributes:
        n: 动作数量
        action_names: 各动作的可读名称（可选）
    """
    n: int
    action_names: Optional[tuple[str, ...]] = None

    def __post_init__(self):
        if self.n < 1:
            raise ValueError(f"动作数量 n 必须 >= 1，收到 {self.n}")
        if self.action_names is not None and len(self.action_names) != self.n:
            raise ValueError(
                f"action_names 长度 {len(self.action_names)} 必须等于 n={self.n}"
            )

    @property
    def space_type(self) -> ActionSpaceType:
        return ActionSpaceType.DISCRETE

    def sample(self, rng: np.random.RandomState) -> int:
        """均匀随机采样一个动作。"""
        return int(rng.randint(self.n))

    def contains(self, action: int) -> bool:
        """检查动作是否在空间内。"""
        return isinstance(action, (int, np.integer)) and 0 <= action < self.n

    def one_hot(self, action: int) -> np.ndarray:
        """将离散动作转为one-hot向量。"""
        if not self.contains(action):
            raise ValueError(f"动作 {action} 不在空间内")
        vec = np.zeros(self.n)
        vec[action] = 1.0
        return vec


@dataclass(frozen=True)
class ContinuousActionSpace:
    """连续动作空间。

    Attributes:
        low: 各维度下界（形状 (dim,)）
        high: 各维度上界（形状 (dim,)）
        dim: 动作维度
    """
    low: np.ndarray
    high: np.ndarray
    dim: int

    def __post_init__(self):
        if self.low.shape != (self.dim,) or self.high.shape != (self.dim,):
            raise ValueError(
                f"low/high 形状必须为 ({self.dim},)，"
                f"收到 low={self.low.shape}, high={self.high.shape}"
            )
        if np.any(self.low >= self.high):
            raise ValueError("low 必须严格小于 high")

    @property
    def space_type(self) -> ActionSpaceType:
        return ActionSpaceType.CONTINUOUS

    def sample(self, rng: np.random.RandomState) -> np.ndarray:
        """在 [low, high] 范围内均匀采样。"""
        return self.low + rng.rand(self.dim) * (self.high - self.low)

    def contains(self, action: np.ndarray) -> bool:
        """检查动作是否在空间内。"""
        if action.shape != (self.dim,):
            return False
        return bool(np.all(action >= self.low) and np.all(action <= self.high))

    def clip(self, action: np.ndarray) -> np.ndarray:
        """将动作裁剪到合法范围。"""
        return np.clip(action, self.low, self.high)


# 联合类型
ActionSpace = Union[DiscreteActionSpace, ContinuousActionSpace]


# ============================================================
# 动作编码工具
# ============================================================
def encode_action(action: Union[int, np.ndarray], space: ActionSpace) -> np.ndarray:
    """将动作编码为向量（用于输入到预测编码模型）。"""
    if isinstance(space, DiscreteActionSpace):
        if not space.contains(action):
            raise ValueError(f"动作 {action} 不在离散空间内")
        return space.one_hot(action)
    else:
        if not space.contains(action):
            raise ValueError(f"动作 {action} 不在连续空间内")
        return np.asarray(action, dtype=float).copy()


def decode_action(encoded: np.ndarray, space: ActionSpace) -> Union[int, np.ndarray]:
    """将向量解码为动作。"""
    if isinstance(space, DiscreteActionSpace):
        return int(np.argmax(encoded))
    else:
        return space.clip(encoded)
