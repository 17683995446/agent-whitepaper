"""阶段B-3：元学习器 - 学习如何学习。

第一性原理：
    元学习是"学会学习"——不是学习具体技能，
    而是学习"什么学习策略最有效"。

    元参数 vs 底层参数：
    - 底层参数：Q表的值、技能的动作序列
    - 元参数：学习率、探索率、技能抽象的阈值

    元学习器监控学习过程，根据效果调整元参数：
    - 如果学习进展快 → 保持当前学习率
    - 如果学习停滞 → 增加探索或降低学习率
    - 如果性能波动大 → 降低学习率（稳定化）

    这实现了真正的"自主增强"：Agent不仅学会技能，
    还学会如何更有效地学习新技能。
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ============================================================
# 元参数
# ============================================================
@dataclass
class MetaParameters:
    """元参数：控制学习过程的参数。

    Attributes:
        learning_rate: Q学习的学习率
        exploration_rate: 基础探索率
        skill_abstraction_threshold: 技能抽象的最小频率阈值
        replay_batch_size: 回放批量大小
        drive_sensitivity: 驱力对策略的影响强度
    """
    learning_rate: float = 0.1
    exploration_rate: float = 0.3
    skill_abstraction_threshold: int = 3
    replay_batch_size: int = 32
    drive_sensitivity: float = 0.5

    def clamp(self) -> "MetaParameters":
        """将参数限制在合理范围。"""
        return MetaParameters(
            learning_rate=float(np.clip(self.learning_rate, 0.01, 1.0)),
            exploration_rate=float(np.clip(self.exploration_rate, 0.01, 0.99)),
            skill_abstraction_threshold=int(np.clip(self.skill_abstraction_threshold, 1, 20)),
            replay_batch_size=int(np.clip(self.replay_batch_size, 4, 128)),
            drive_sensitivity=float(np.clip(self.drive_sensitivity, 0.0, 1.0)),
        )


# ============================================================
# 学习进度记录
# ============================================================
@dataclass(frozen=True)
class LearningProgress:
    """一个学习周期的进度记录。

    Attributes:
        episode_range: (起始情节, 结束情节)
        avg_reward: 平均奖励
        success_rate: 成功率
        learning_speed: 学习速度（奖励提升率）
        stability: 稳定性（奖励方差的倒数）
        free_energy_trend: 自由能趋势（斜率）
    """
    episode_range: Tuple[int, int]
    avg_reward: float
    success_rate: float
    learning_speed: float
    stability: float
    free_energy_trend: float


# ============================================================
# 元学习器
# ============================================================
class MetaLearner:
    """元学习器：根据学习进度调整元参数。

    策略：
    1. 监控窗口：每N个情节计算一次学习进度
    2. 趋势分析：奖励是否在提升？自由能是否在下降？
    3. 参数调整：
       - 学习停滞 → 增加探索率
       - 学习波动 → 降低学习率
       - 学习顺利 → 微调（保持稳定）
    4. 历史记忆：记录哪些参数组合效果好
    """

    def __init__(
        self,
        initial_params: Optional[MetaParameters] = None,
        window_size: int = 20,
        adaptation_rate: float = 0.1
    ):
        if window_size < 5:
            raise ValueError("window_size 必须 >= 5")
        if not (0 < adaptation_rate <= 0.5):
            raise ValueError("adaptation_rate 必须 ∈ (0, 0.5]")
        self.params = initial_params or MetaParameters()
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        self._reward_history: List[float] = []
        self._success_history: List[bool] = []
        self._free_energy_history: List[float] = []
        self._progress_history: List[LearningProgress] = []
        self._param_history: List[Tuple[int, MetaParameters]] = []

    def record_episode(self, reward: float, success: bool, free_energy: float = 0.0) -> None:
        """记录一个情节的结果。"""
        self._reward_history.append(reward)
        self._success_history.append(success)
        self._free_energy_history.append(free_energy)

    def should_adapt(self) -> bool:
        """是否到了调整参数的时机。"""
        return len(self._reward_history) >= self.window_size and \
               len(self._reward_history) % self.window_size == 0

    def compute_progress(self) -> Optional[LearningProgress]:
        """计算当前窗口的学习进度。"""
        if len(self._reward_history) < self.window_size:
            return None

        start = len(self._reward_history) - self.window_size
        end = len(self._reward_history)
        window_rewards = self._reward_history[start:end]
        window_successes = self._success_history[start:end]
        window_fe = self._free_energy_history[start:end] if len(self._free_energy_history) >= end else []

        # 学习速度：前后半段奖励差
        half = self.window_size // 2
        first_half_avg = np.mean(window_rewards[:half]) if half > 0 else 0
        second_half_avg = np.mean(window_rewards[half:]) if half > 0 else 0
        learning_speed = float(second_half_avg - first_half_avg)

        # 稳定性：奖励方差的倒数
        reward_std = float(np.std(window_rewards))
        stability = 1.0 / (1.0 + reward_std)

        # 自由能趋势：线性回归斜率
        if len(window_fe) >= self.window_size:
            x = np.arange(len(window_fe))
            try:
                fe_trend = float(np.polyfit(x, window_fe, 1)[0])
            except (np.linalg.LinAlgError, ValueError):
                fe_trend = 0.0
        else:
            fe_trend = 0.0

        progress = LearningProgress(
            episode_range=(start, end),
            avg_reward=float(np.mean(window_rewards)),
            success_rate=float(np.mean(window_successes)),
            learning_speed=learning_speed,
            stability=stability,
            free_energy_trend=fe_trend,
        )
        self._progress_history.append(progress)
        return progress

    def adapt_parameters(self) -> MetaParameters:
        """根据学习进度调整元参数。"""
        progress = self.compute_progress()
        if progress is None:
            return self.params

        # 记录调整前的参数
        self._param_history.append((len(self._reward_history), self.params))

        new_lr = self.params.learning_rate
        new_explore = self.params.exploration_rate
        new_threshold = self.params.skill_abstraction_threshold
        new_batch = self.params.replay_batch_size
        new_sensitivity = self.params.drive_sensitivity

        # 规则1：学习停滞（速度接近0或负）→ 增加探索
        if progress.learning_speed < 0.01:
            new_explore = min(0.9, new_explore + self.adaptation_rate)
            new_lr = max(0.05, new_lr - self.adaptation_rate * 0.5)
        # 规则2：学习波动大（稳定性低）→ 降低学习率
        elif progress.stability < 0.3:
            new_lr = max(0.02, new_lr - self.adaptation_rate * 0.3)
        # 规则3：学习顺利 → 略降探索，增加利用
        elif progress.learning_speed > 0.1 and progress.stability > 0.5:
            new_explore = max(0.05, new_explore - self.adaptation_rate * 0.5)
            new_threshold = max(2, new_threshold - 1)  # 更容易抽象技能

        # 规则4：自由能持续上升 → 增加驱力敏感度
        if progress.free_energy_trend > 0.1:
            new_sensitivity = min(1.0, new_sensitivity + self.adaptation_rate)
        elif progress.free_energy_trend < -0.1:
            new_sensitivity = max(0.1, new_sensitivity - self.adaptation_rate * 0.5)

        self.params = MetaParameters(
            learning_rate=new_lr,
            exploration_rate=new_explore,
            skill_abstraction_threshold=new_threshold,
            replay_batch_size=new_batch,
            drive_sensitivity=new_sensitivity,
        ).clamp()

        return self.params

    def get_best_params(self) -> Optional[MetaParameters]:
        """获取历史上效果最好的参数。"""
        if not self._progress_history or not self._param_history:
            return None
        # 找成功率最高的时期
        best_progress = max(self._progress_history, key=lambda p: p.success_rate)
        # 找对应时期的参数
        for episode_count, params in reversed(self._param_history):
            if episode_count <= best_progress.episode_range[1]:
                return params
        return self.params

    def get_progress_summary(self) -> dict:
        """获取学习进度摘要。"""
        if not self._progress_history:
            return {"status": "warming_up", "episodes": len(self._reward_history)}
        latest = self._progress_history[-1]
        return {
            "status": "learning",
            "episodes": len(self._reward_history),
            "latest_avg_reward": latest.avg_reward,
            "latest_success_rate": latest.success_rate,
            "learning_speed": latest.learning_speed,
            "stability": latest.stability,
            "free_energy_trend": latest.free_energy_trend,
            "current_lr": self.params.learning_rate,
            "current_explore": self.params.exploration_rate,
        }
