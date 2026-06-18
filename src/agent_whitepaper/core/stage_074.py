"""阶段74：优先级经验回放——价值驱动采样、偏差校正与优先级更新。

本阶段实现优先级经验回放机制。
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .stage_073 import SampleValueEvaluator


@dataclass(frozen=True)
class PriorityReplay:
    """优先级经验回放的完整描述。

    组合阶段73的SampleValueEvaluator，在价值评估的基础上增加
    优先级采样和偏差校正的状态。

    Attributes:
        evaluator: 阶段73的样本价值评估器（组合字段）
        alpha: 优先级指数
        beta: 重要性采样指数
        beta_start: β初始值
        beta_frames: β退火总步数
        batch_size: 批量采样大小
        epsilon: 零价值保护常数
        current_step: 当前步数
        total_updates: 优先级更新次数

    Invariants:
        - alpha >= 0
        - 0 <= beta <= 1
        - batch_size > 0
        - epsilon > 0
    """
    evaluator: SampleValueEvaluator
    alpha: float = 0.6
    beta: float = 0.4
    beta_start: float = 0.4
    beta_frames: int = 100000
    batch_size: int = 32
    epsilon: float = 1e-6
    current_step: int = 0
    total_updates: int = 0

    def __post_init__(self):
        if self.alpha < 0:
            raise ValueError(
                f"优先级指数必须 >= 0，收到 {self.alpha}"
            )
        if not (0 <= self.beta <= 1):
            raise ValueError(
                f"重要性采样指数必须 ∈ [0,1]，收到 {self.beta}"
            )
        if self.batch_size <= 0:
            raise ValueError(
                f"批量大小必须 > 0，收到 {self.batch_size}"
            )
        if self.epsilon <= 0:
            raise ValueError(
                f"保护常数必须 > 0，收到 {self.epsilon}"
            )

    @property
    def beta_annealed(self) -> float:
        """退火后的β值（A74.2-anneal）。"""
        progress = min(
            1.0, self.current_step / max(1, self.beta_frames)
        )
        return self.beta_start + (1.0 - self.beta_start) * progress


class PriorityReplayInterface:
    """阶段74的核心接口：优先级经验回放。

    实现优先级采样、重要性权重和优先级更新。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 采样概率归一化
        - 重要性权重归一化
        - 优先级更新后概率归一化
    """

    def compute_sampling_probabilities(
        self,
        values: np.ndarray,
        alpha: float,
        epsilon: float = 1e-6
    ) -> np.ndarray:
        """计算采样概率（A74.1）。

        P(i) = (value_i + ε)^α / Σ(value_j + ε)^α

        Args:
            values: 价值评分数组，shape (n,)
            alpha: 优先级指数，必须 >= 0
            epsilon: 零价值保护常数，必须 > 0

        Returns:
            采样概率数组，shape (n,)，和为1

        Raises:
            ValueError: 当alpha < 0时
            ValueError: 当epsilon <= 0时
            ValueError: 当values为空时
        """
        if alpha < 0:
            raise ValueError(f"alpha必须 >= 0，收到 {alpha}")
        if epsilon <= 0:
            raise ValueError(f"epsilon必须 > 0，收到 {epsilon}")
        if values.size == 0:
            raise ValueError("values不能为空数组")
        scaled = (values + epsilon) ** alpha
        total = np.sum(scaled)
        if total <= 0:
            return np.ones_like(values) / values.size
        return scaled / total

    def compute_importance_weights(
        self,
        probabilities: np.ndarray,
        n_samples: int,
        beta: float
    ) -> np.ndarray:
        """计算重要性权重（A74.2）。

        w_i = (1/(N·P_i))^β, 归一化为 w_i/max(w)

        Args:
            probabilities: 采样概率数组，shape (n,)
            n_samples: 样本总数N，必须 > 0
            beta: 重要性采样指数，必须 ∈ [0, 1]

        Returns:
            归一化重要性权重数组，shape (n,)，max=1.0

        Raises:
            ValueError: 当beta不在[0,1]范围内时
            ValueError: 当n_samples <= 0时
            ValueError: 当probabilities包含0时
        """
        if not (0 <= beta <= 1):
            raise ValueError(f"beta必须 ∈ [0,1]，收到 {beta}")
        if n_samples <= 0:
            raise ValueError(f"n_samples必须 > 0，收到 {n_samples}")
        if np.any(probabilities <= 0):
            raise ValueError("probabilities必须全部 > 0")
        weights = (1.0 / (n_samples * probabilities)) ** beta
        max_w = np.max(weights)
        if max_w > 0:
            weights = weights / max_w
        return weights

    def sample_batch(
        self,
        replay: PriorityReplay,
        buffer_values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """批量采样（A74.3）。

        Args:
            replay: 当前回放状态
            buffer_values: 缓冲区中所有样本的价值评分，shape (n,)

        Returns:
            (采样索引, 采样概率, 重要性权重)
            - 采样索引: shape (batch_size,)
            - 采样概率: shape (batch_size,)
            - 重要性权重: shape (batch_size,)

        Raises:
            ValueError: 当buffer_values为空时
            ValueError: 当batch_size > len(buffer_values)时
        """
        if buffer_values.size == 0:
            raise ValueError("buffer_values不能为空")
        if replay.batch_size > buffer_values.size:
            raise ValueError(
                f"batch_size({replay.batch_size})不能超过"
                f"buffer大小({buffer_values.size})"
            )
        probs = self.compute_sampling_probabilities(
            buffer_values, replay.alpha, replay.epsilon
        )
        indices = np.random.choice(
            len(buffer_values),
            size=replay.batch_size,
            replace=False,
            p=probs
        )
        sampled_probs = probs[indices]
        current_beta = replay.beta_annealed
        weights = self.compute_importance_weights(
            sampled_probs, len(buffer_values), current_beta
        )
        return indices, sampled_probs, weights

    def update_priority(
        self,
        replay: PriorityReplay,
        indices: np.ndarray,
        new_values: np.ndarray
    ) -> PriorityReplay:
        """更新优先级（A74.4）。

        Args:
            replay: 当前回放状态
            indices: 待更新的样本索引，shape (k,)
            new_values: 新的价值评分，shape (k,)

        Returns:
            更新后的回放状态

        Raises:
            ValueError: 当indices和new_values长度不匹配时
        """
        if len(indices) != len(new_values):
            raise ValueError(
                f"indices长度({len(indices)})与new_values长度"
                f"({len(new_values)})不匹配"
            )
        return PriorityReplay(
            evaluator=replay.evaluator,
            alpha=replay.alpha,
            beta=replay.beta,
            beta_start=replay.beta_start,
            beta_frames=replay.beta_frames,
            batch_size=replay.batch_size,
            epsilon=replay.epsilon,
            current_step=replay.current_step,
            total_updates=replay.total_updates + len(indices)
        )
