"""阶段F-2：性能退化检测器 - 检测Agent性能下降并触发恢复。

第一性原理：
    "永远运行不停止"的系统中，性能可能因多种原因退化：
    - 灾难性遗忘：新学习覆盖了旧知识
    - 概念漂移：环境变化导致旧策略失效
    - 过拟合：对近期任务过拟合，泛化能力下降
    - 数值不稳定：Q值发散

    退化检测器是系统的"体检医生"：
    1. 持续监控关键指标（成功率、奖励、自由能）
    2. 检测 statistically significant 的下降趋势
    3. 区分"正常波动"与"真正退化"
    4. 触发恢复机制（重训练、回滚、知识刷新）
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import numpy as np


# ============================================================
# 退化类型
# ============================================================
class DegradationType(Enum):
    """性能退化类型。"""
    NONE = "none"                    # 无退化
    SUCCESS_RATE_DROP = "success_rate_drop"      # 成功率下降
    REWARD_DECLINE = "reward_decline"            # 奖励下降
    FREE_ENERGY_INCREASE = "free_energy_increase" # 自由能上升
    CATASTROPHIC_FORGETTING = "catastrophic_forgetting"  # 灾难性遗忘
    INSTABILITY = "instability"                  # 不稳定（高方差）


# ============================================================
# 退化检测结果
# ============================================================
@dataclass(frozen=True)
class DegradationReport:
    """退化检测报告。

    Attributes:
        degradation_type: 退化类型
        severity: 严重程度 [0,1]
        detected_at: 检测时间
        baseline_value: 基线值
        current_value: 当前值
        trend: 趋势（负=下降）
        recommendation: 建议的恢复操作
    """
    degradation_type: DegradationType
    severity: float
    detected_at: float
    baseline_value: float
    current_value: float
    trend: float
    recommendation: str


# ============================================================
# 性能退化检测器
# ============================================================
class DegradationDetector:
    """性能退化检测器。

    检测算法：
    1. 维护滑动窗口的性能指标
    2. 计算基线（历史最佳）与当前值的差异
    3. 用线性回归检测趋势
    4. 用统计检验区分波动与退化
    """

    def __init__(
        self,
        window_size: int = 50,
        baseline_window: int = 100,
        degradation_threshold: float = 0.15,
        trend_threshold: float = -0.01
    ):
        if window_size < 10:
            raise ValueError("window_size 必须 >= 10")
        self.window_size = window_size
        self.baseline_window = baseline_window
        self.degradation_threshold = degradation_threshold
        self.trend_threshold = trend_threshold

        # 性能指标历史
        self._success_rates: deque = deque(maxlen=baseline_window)
        self._rewards: deque = deque(maxlen=baseline_window)
        self._free_energies: deque = deque(maxlen=baseline_window)
        self._q_value_norms: deque = deque(maxlen=baseline_window)

        # 基线（历史最佳）
        self._best_success_rate: float = 0.0
        self._best_reward: float = float('-inf')

    def record(
        self,
        success_rate: float,
        avg_reward: float,
        free_energy: float = 0.0,
        q_value_norm: float = 0.0
    ) -> None:
        """记录性能指标。"""
        self._success_rates.append(success_rate)
        self._rewards.append(avg_reward)
        self._free_energies.append(free_energy)
        self._q_value_norms.append(q_value_norm)

        # 更新基线
        if success_rate > self._best_success_rate:
            self._best_success_rate = success_rate
        if avg_reward > self._best_reward:
            self._best_reward = avg_reward

    def _compute_trend(self, values: List[float]) -> float:
        """计算线性趋势斜率。"""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        try:
            slope = float(np.polyfit(x, values, 1)[0])
        except (np.linalg.LinAlgError, ValueError):
            slope = 0.0
        return slope

    def _compute_variance(self, values: List[float]) -> float:
        """计算方差。"""
        if len(values) < 2:
            return 0.0
        return float(np.var(values))

    def detect(self) -> DegradationReport:
        """检测性能退化。"""
        import time
        now = time.time()

        if len(self._success_rates) < self.window_size:
            return DegradationReport(
                degradation_type=DegradationType.NONE,
                severity=0.0,
                detected_at=now,
                baseline_value=0.0,
                current_value=0.0,
                trend=0.0,
                recommendation="数据不足，继续收集",
            )

        # 取最近窗口的数据
        recent_success = list(self._success_rates)[-self.window_size:]
        recent_rewards = list(self._rewards)[-self.window_size:]
        recent_fe = list(self._free_energies)[-self.window_size:]
        recent_qnorms = list(self._q_value_norms)[-self.window_size:]

        current_success = float(np.mean(recent_success))
        current_reward = float(np.mean(recent_rewards))
        current_fe = float(np.mean(recent_fe))

        # 检测1: 成功率下降
        success_trend = self._compute_trend(recent_success)
        success_drop = self._best_success_rate - current_success
        if success_drop > self.degradation_threshold and success_trend < self.trend_threshold:
            severity = min(1.0, success_drop / max(self._best_success_rate, 0.01))
            return DegradationReport(
                degradation_type=DegradationType.SUCCESS_RATE_DROP,
                severity=severity,
                detected_at=now,
                baseline_value=self._best_success_rate,
                current_value=current_success,
                trend=success_trend,
                recommendation="触发重训练，恢复成功率",
            )

        # 检测2: 奖励下降
        reward_trend = self._compute_trend(recent_rewards)
        if self._best_reward != float('-inf'):
            reward_drop = self._best_reward - current_reward
            if reward_drop > abs(self._best_reward) * self.degradation_threshold:
                severity = min(1.0, reward_drop / max(abs(self._best_reward), 0.01))
                return DegradationReport(
                    degradation_type=DegradationType.REWARD_DECLINE,
                    severity=severity,
                    detected_at=now,
                    baseline_value=self._best_reward,
                    current_value=current_reward,
                    trend=reward_trend,
                    recommendation="检查奖励函数，可能需要回滚Q表",
                )

        # 检测3: 自由能上升
        fe_trend = self._compute_trend(recent_fe)
        if fe_trend > 0.01 and len(recent_fe) >= self.window_size:
            fe_baseline = float(np.mean(list(self._free_energies)[:-self.window_size])) \
                if len(self._free_energies) > self.window_size else current_fe
            fe_increase = current_fe - fe_baseline
            if fe_increase > 0.1:
                return DegradationReport(
                    degradation_type=DegradationType.FREE_ENERGY_INCREASE,
                    severity=min(1.0, fe_increase),
                    detected_at=now,
                    baseline_value=fe_baseline,
                    current_value=current_fe,
                    trend=fe_trend,
                    recommendation="自由能上升，预测能力下降，需重训练",
                )

        # 检测4: 灾难性遗忘（Q值范数突变）
        if len(recent_qnorms) >= self.window_size and all(q > 0 for q in recent_qnorms[-5:]):
            q_trend = self._compute_trend(recent_qnorms)
            q_variance = self._compute_variance(recent_qnorms)
            if q_variance > 1.0:  # Q值剧烈变化
                return DegradationReport(
                    degradation_type=DegradationType.CATASTROPHIC_FORGETTING,
                    severity=min(1.0, q_variance / 5.0),
                    detected_at=now,
                    baseline_value=float(np.mean(recent_qnorms[:10])),
                    current_value=float(np.mean(recent_qnorms[-10:])),
                    trend=q_trend,
                    recommendation="Q值不稳定，可能灾难性遗忘，回滚到检查点",
                )

        # 检测5: 不稳定（高方差）
        success_variance = self._compute_variance(recent_success)
        if success_variance > 0.15:
            return DegradationReport(
                degradation_type=DegradationType.INSTABILITY,
                severity=min(1.0, success_variance),
                detected_at=now,
                baseline_value=current_success,
                current_value=current_success,
                trend=success_trend,
                recommendation="性能不稳定，降低学习率",
            )

        # 无退化
        return DegradationReport(
            degradation_type=DegradationType.NONE,
            severity=0.0,
            detected_at=now,
            baseline_value=self._best_success_rate,
            current_value=current_success,
            trend=success_trend,
            recommendation="性能正常",
        )

    def get_health_score(self) -> float:
        """获取综合健康评分 [0,1]。"""
        if len(self._success_rates) < self.window_size:
            return 1.0
        report = self.detect()
        return 1.0 - report.severity

    def get_stats(self) -> Dict:
        """获取检测器统计。"""
        return {
            "samples_collected": len(self._success_rates),
            "best_success_rate": self._best_success_rate,
            "best_reward": self._best_reward,
            "current_health": self.get_health_score(),
            "window_size": self.window_size,
        }
