"""阶段J-1：安全边界 - 系统的"免疫系统"。

第一性原理：
    永续运行的系统必须能自我保护。一个不受约束的自主系统可能：
    1. 陷入无限循环（状态循环检测）
    2. 持续做出有害行为（奖励持续为负）
    3. 探索危险状态（状态空间越界）
    4. 资源耗尽（内存/时间爆炸）

    安全边界是系统的"免疫系统"：
    - 动作过滤：屏蔽不安全动作
    - 状态监控：检测异常状态模式
    - 紧急停止：触发保护性停止
    - 行为审计：记录所有决策供事后审查

    约束不是限制智能，而是保障智能可持续发挥。
    正如生物的痛觉——它不阻止行动，但防止毁灭性行动。
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ============================================================
# 安全等级
# ============================================================
class SafetyLevel(Enum):
    """安全等级。"""
    SAFE = "safe"           # 安全
    WARNING = "warning"     # 警告（接近边界）
    DANGER = "danger"       # 危险（需干预）
    CRITICAL = "critical"   # 危急（必须停止）


# ============================================================
# 安全配置
# ============================================================
@dataclass
class SafetyConfig:
    """安全边界配置。"""
    max_consecutive_failures: int = 20      # 最大连续失败次数
    max_negative_reward_streak: int = 50    # 最大连续负奖励次数
    min_avg_reward_threshold: float = -1.0  # 平均奖励下限
    state_loop_window: int = 20             # 状态循环检测窗口
    state_loop_threshold: int = 5           # 状态重复次数阈值
    max_q_value: float = 1e6                # Q值上限（防数值爆炸）
    min_q_value: float = -1e6               # Q值下限
    max_episode_length: int = 1000          # 单情节最大步数
    forbidden_actions: Set[int] = field(default_factory=set)  # 禁用动作


# ============================================================
# 安全违规记录
# ============================================================
@dataclass
class SafetyViolation:
    """安全违规记录。"""
    level: SafetyLevel
    violation_type: str
    message: str
    timestamp: float
    context: Dict = field(default_factory=dict)


# ============================================================
# 安全监控器
# ============================================================
class SafetyMonitor:
    """安全监控器：实时监控Agent行为，检测异常。

    检测类型：
    1. 连续失败：Agent连续失败次数过多
    2. 负奖励持续：长期获得负奖励
    3. 状态循环：Agent陷入状态循环
    4. 数值爆炸：Q值超出合理范围
    5. 情节过长：单情节步数过多
    6. 禁用动作：尝试执行被禁用的动作
    """

    def __init__(self, config: Optional[SafetyConfig] = None) -> None:
        self.config = config or SafetyConfig()
        self._consecutive_failures = 0
        self._negative_reward_streak = 0
        self._recent_states = deque(maxlen=self.config.state_loop_window)
        self._state_counts: Dict[int, int] = {}
        self._violations: List[SafetyViolation] = []
        self._emergency_stop = False
        self._stop_reason = ""

    def check_action(self, state: int, action: int) -> Tuple[bool, Optional[SafetyViolation]]:
        """检查动作是否安全。

        Returns:
            (是否允许, 违规记录)
        """
        import time

        # 检查禁用动作
        if action in self.config.forbidden_actions:
            v = SafetyViolation(
                level=SafetyLevel.DANGER,
                violation_type="forbidden_action",
                message=f"禁用动作 {action} 被拦截",
                timestamp=time.time(),
                context={"state": state, "action": action},
            )
            self._record_violation(v)
            return False, v

        return True, None

    def record_step(self, state: int, reward: float, done: bool, success: bool) -> Optional[SafetyViolation]:
        """记录一步执行结果，返回违规（如有）。"""
        import time

        # 1. 连续失败检测
        if done:
            if success:
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.config.max_consecutive_failures:
                    v = SafetyViolation(
                        level=SafetyLevel.DANGER,
                        violation_type="consecutive_failures",
                        message=f"连续失败 {self._consecutive_failures} 次",
                        timestamp=time.time(),
                        context={"count": self._consecutive_failures},
                    )
                    self._record_violation(v)
                    return v

        # 2. 负奖励持续检测
        if reward < 0:
            self._negative_reward_streak += 1
            if self._negative_reward_streak >= self.config.max_negative_reward_streak:
                v = SafetyViolation(
                    level=SafetyLevel.WARNING,
                    violation_type="negative_reward_streak",
                    message=f"连续 {self._negative_reward_streak} 步负奖励",
                    timestamp=time.time(),
                    context={"streak": self._negative_reward_streak},
                )
                self._record_violation(v)
                return v
        else:
            self._negative_reward_streak = 0

        # 3. 状态循环检测
        self._recent_states.append(state)
        self._state_counts[state] = self._state_counts.get(state, 0) + 1
        if self._state_counts[state] >= self.config.state_loop_threshold:
            v = SafetyViolation(
                level=SafetyLevel.WARNING,
                violation_type="state_loop",
                message=f"状态 {state} 在窗口内重复 {self._state_counts[state]} 次",
                timestamp=time.time(),
                context={"state": state, "count": self._state_counts[state]},
            )
            self._record_violation(v)
            return v

        return None

    def check_q_table(self, q_table: np.ndarray) -> Optional[SafetyViolation]:
        """检查Q表数值是否异常。"""
        import time

        max_q = float(np.max(q_table)) if q_table.size > 0 else 0.0
        min_q = float(np.min(q_table)) if q_table.size > 0 else 0.0

        if max_q > self.config.max_q_value or min_q < self.config.min_q_value:
            v = SafetyViolation(
                level=SafetyLevel.CRITICAL,
                violation_type="q_value_explosion",
                message=f"Q值越界: [{min_q:.2f}, {max_q:.2f}]",
                timestamp=time.time(),
                context={"max_q": max_q, "min_q": min_q},
            )
            self._record_violation(v)
            self._emergency_stop = True
            self._stop_reason = "Q值数值爆炸"
            return v

        # 检查NaN/Inf
        if np.any(np.isnan(q_table)) or np.any(np.isinf(q_table)):
            v = SafetyViolation(
                level=SafetyLevel.CRITICAL,
                violation_type="q_value_nan",
                message="Q表包含NaN或Inf",
                timestamp=time.time(),
            )
            self._record_violation(v)
            self._emergency_stop = True
            self._stop_reason = "Q表数值异常"
            return v

        return None

    def check_episode_length(self, length: int) -> Optional[SafetyViolation]:
        """检查情节长度是否异常。"""
        import time

        if length > self.config.max_episode_length:
            v = SafetyViolation(
                level=SafetyLevel.WARNING,
                violation_type="episode_too_long",
                message=f"情节长度 {length} 超过上限 {self.config.max_episode_length}",
                timestamp=time.time(),
                context={"length": length},
            )
            self._record_violation(v)
            return v
        return None

    def trigger_emergency_stop(self, reason: str) -> None:
        """触发紧急停止。"""
        import time
        self._emergency_stop = True
        self._stop_reason = reason
        v = SafetyViolation(
            level=SafetyLevel.CRITICAL,
            violation_type="emergency_stop",
            message=f"紧急停止: {reason}",
            timestamp=time.time(),
        )
        self._record_violation(v)

    def reset_state_tracking(self) -> None:
        """重置状态跟踪（每个新情节开始时调用）。"""
        self._recent_states.clear()
        self._state_counts.clear()

    def _record_violation(self, violation: SafetyViolation) -> None:
        self._violations.append(violation)
        # 保留最近1000条
        if len(self._violations) > 1000:
            self._violations = self._violations[-1000:]

    @property
    def should_stop(self) -> bool:
        """是否应紧急停止。"""
        return self._emergency_stop

    @property
    def stop_reason(self) -> str:
        return self._stop_reason

    @property
    def is_safe(self) -> bool:
        """当前是否安全（无危急违规）。"""
        return not self._emergency_stop

    def get_safety_report(self) -> Dict:
        """获取安全报告。"""
        level_counts = {level.value: 0 for level in SafetyLevel}
        for v in self._violations:
            level_counts[v.level.value] += 1
        return {
            "is_safe": self.is_safe,
            "should_stop": self.should_stop,
            "stop_reason": self._stop_reason,
            "total_violations": len(self._violations),
            "violations_by_level": level_counts,
            "consecutive_failures": self._consecutive_failures,
            "negative_reward_streak": self._negative_reward_streak,
            "recent_violation_types": [
                v.violation_type for v in self._violations[-10:]
            ],
        }


# ============================================================
# 安全动作过滤器
# ============================================================
class SafeActionFilter:
    """安全动作过滤器：在策略选择动作后进行安全过滤。

    工作流：
    1. 策略选择动作
    2. 过滤器检查动作是否安全
    3. 不安全则选择次优安全动作
    """

    def __init__(self, monitor: SafetyMonitor) -> None:
        self.monitor = monitor

    def filter(
        self,
        state: int,
        proposed_action: int,
        q_values: np.ndarray,
        n_actions: int
    ) -> int:
        """过滤动作，返回安全的动作。

        如果提议动作不安全，选择Q值次高的安全动作。
        """
        allowed, _ = self.monitor.check_action(state, proposed_action)
        if allowed:
            return proposed_action

        # 找到所有安全动作，按Q值排序
        safe_actions = [
            a for a in range(n_actions)
            if a not in self.monitor.config.forbidden_actions
        ]
        if not safe_actions:
            return proposed_action  # 全部禁用，返回原动作

        # 按Q值排序，选最优安全动作
        safe_q = [(q_values[a], a) for a in safe_actions]
        safe_q.sort(reverse=True)
        return safe_q[0][1]
