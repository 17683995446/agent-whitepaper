"""阶段89：稳态环境→探索权重上浮触发器。

本阶段实现稳态检测与探索权重上浮机制。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np
import time

from .stage_088 import IdleReplayTrigger


@dataclass(frozen=True)
class StableExploreTriggerState:
    """稳态探索触发器的状态描述。

    Attributes:
        is_stable: 是否稳态
        stable_duration: 稳态持续时长（秒）
        F_variance: 自由能方差
        W_variance: 权重范数方差
        current_explore_weight: 当前探索权重
        base_explore_weight: 基准探索权重
        max_explore_weight: 最大探索权重
        total_boosts: 总上浮次数
        last_boost_time: 上次上浮时间
        is_boosted: 是否处于上浮状态
        boost_accumulation: 累积上浮量
        decay_factor: 衰减系数

    Invariants:
        - 0 <= current_explore_weight <= max_explore_weight
        - base_explore_weight <= max_explore_weight
    """
    is_stable: bool = False
    stable_duration: float = 0.0
    F_variance: float = 0.0
    W_variance: float = 0.0
    current_explore_weight: float = 0.30
    base_explore_weight: float = 0.30
    max_explore_weight: float = 0.50
    total_boosts: int = 0
    last_boost_time: float = 0.0
    is_boosted: bool = False
    boost_accumulation: float = 0.0
    decay_factor: float = 0.99

    def __post_init__(self):
        if self.current_explore_weight > self.max_explore_weight:
            raise ValueError(
                f"当前探索权重({self.current_explore_weight})不能超过"
                f"最大值({self.max_explore_weight})"
            )


@dataclass(frozen=True)
class StableExploreTrigger:
    """稳态环境→探索权重上浮的触发器。

    组合阶段88的IdleReplayTrigger，在空闲复盘基础上增加稳态探索触发状态。

    Attributes:
        idle_replay_trigger: 阶段88的空闲复盘触发器（组合字段）
        trigger_state: 稳态探索触发器状态
        stable_F_threshold: 自由能方差阈值（默认0.001）
        stable_W_threshold: 权重方差阈值（默认0.0001）
        min_stable_duration: 最小稳态持续时间（默认300秒）
        boost_coefficient: 上浮系数α（默认0.05）
        reference_time: 参考时间T_ref（默认3600秒）
        min_boost_interval: 最小上浮间隔（默认600秒）
        variance_window: 方差计算窗口（默认1000步）
    """
    idle_replay_trigger: IdleReplayTrigger
    trigger_state: StableExploreTriggerState = field(
        default_factory=StableExploreTriggerState
    )
    stable_F_threshold: float = 0.001
    stable_W_threshold: float = 0.0001
    min_stable_duration: float = 300.0
    boost_coefficient: float = 0.05
    reference_time: float = 3600.0
    min_boost_interval: float = 600.0
    variance_window: int = 1000


class StableExploreTriggerExecutor:
    """阶段89的核心接口：稳态探索触发器。

    实现稳态判定、上浮触发、上浮幅度计算和安全检查。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 稳态需持续>5分钟
        - 上浮幅度1小时饱和
        - 探索权重≤50%
    """

    def check_stable(
        self,
        F_history: List[float],
        W_history: List[float],
        duration: float,
        F_threshold: float = 0.001,
        W_threshold: float = 0.0001,
        min_duration: float = 300.0
    ) -> Tuple[bool, float, float, float]:
        """检查是否稳态（A89.1）。

        Args:
            F_history: 自由能历史列表，长度 > 0
            W_history: 权重范数历史列表，长度 > 0
            duration: 稳态持续时间（秒），必须 >= 0
            F_threshold: 自由能方差阈值，必须 > 0
            W_threshold: 权重方差阈值，必须 > 0
            min_duration: 最小持续时间（秒），必须 > 0

        Returns:
            (是否稳态, F方差, W方差, 持续时间)

        Raises:
            ValueError: 当F_history或W_history为空时
            ValueError: 当duration < 0时
            ValueError: 当F_threshold <= 0或W_threshold <= 0时
            ValueError: 当min_duration <= 0时
        """
        if not F_history or not W_history:
            raise ValueError("F_history和W_history不能为空")
        if duration < 0:
            raise ValueError(f"duration必须 >= 0，收到 {duration}")
        if F_threshold <= 0 or W_threshold <= 0:
            raise ValueError("F_threshold和W_threshold必须 > 0")
        if min_duration <= 0:
            raise ValueError(f"min_duration必须 > 0，收到 {min_duration}")
        f_arr = np.array(F_history)
        w_arr = np.array(W_history)
        f_var = float(np.var(f_arr, ddof=1)) if len(f_arr) > 1 else 0.0
        w_var = float(np.var(w_arr, ddof=1)) if len(w_arr) > 1 else 0.0
        is_stable = (
            f_var < F_threshold
            and w_var < W_threshold
            and duration >= min_duration
        )
        return is_stable, f_var, w_var, duration

    def trigger_boost(
        self,
        is_stable: bool,
        is_idle: bool,
        last_boost_time: float,
        min_interval: float = 600.0
    ) -> bool:
        """触发探索权重上浮（A89.2）。

        Args:
            is_stable: 是否稳态
            is_idle: 是否空闲
            last_boost_time: 上次上浮时间（时间戳）
            min_interval: 最小上浮间隔（秒），必须 > 0

        Returns:
            是否触发上浮

        Raises:
            ValueError: 当min_interval <= 0时
        """
        if min_interval <= 0:
            raise ValueError(f"min_interval必须 > 0，收到 {min_interval}")
        if not is_stable or not is_idle:
            return False
        return time.time() - last_boost_time >= min_interval

    def compute_boost_amount(
        self,
        stable_duration: float,
        coefficient: float = 0.05,
        reference_time: float = 3600.0
    ) -> float:
        """计算上浮幅度（A89.3）。

        ΔW = α · min(1, T_stable / T_ref)

        Args:
            stable_duration: 稳态持续时间（秒），必须 >= 0
            coefficient: 上浮系数，必须 > 0
            reference_time: 参考时间（秒），必须 > 0

        Returns:
            上浮幅度ΔW ∈ [0, coefficient]

        Raises:
            ValueError: 当stable_duration < 0时
            ValueError: 当coefficient <= 0时
            ValueError: 当reference_time <= 0时
        """
        if stable_duration < 0:
            raise ValueError(f"stable_duration必须 >= 0，收到 {stable_duration}")
        if coefficient <= 0:
            raise ValueError(f"coefficient必须 > 0，收到 {coefficient}")
        if reference_time <= 0:
            raise ValueError(f"reference_time必须 > 0，收到 {reference_time}")
        ratio = min(1.0, stable_duration / reference_time)
        return coefficient * ratio

    def safety_check(
        self,
        current_weight: float,
        boost_amount: float,
        max_weight: float = 0.50
    ) -> Tuple[float, bool]:
        """安全检查（A89.4）。

        Args:
            current_weight: 当前探索权重，范围[0,1]
            boost_amount: 上浮幅度，必须 >= 0
            max_weight: 最大探索权重，必须 > 0

        Returns:
            (截断后的权重, 是否被截断)

        Raises:
            ValueError: 当boost_amount < 0时
            ValueError: 当max_weight <= 0时
        """
        if boost_amount < 0:
            raise ValueError(f"boost_amount必须 >= 0，收到 {boost_amount}")
        if max_weight <= 0:
            raise ValueError(f"max_weight必须 > 0，收到 {max_weight}")
        new_weight = current_weight + boost_amount
        if new_weight > max_weight:
            return max_weight, True
        return new_weight, False

    def decay_weight(
        self,
        current_weight: float,
        base_weight: float = 0.30,
        decay_factor: float = 0.99
    ) -> float:
        """衰减探索权重（A89.3-decay）。

        Args:
            current_weight: 当前探索权重，范围[0,1]
            base_weight: 基准探索权重，范围[0,1]
            decay_factor: 衰减系数，范围(0,1)

        Returns:
            衰减后的探索权重

        Raises:
            ValueError: 当decay_factor不在(0,1)范围内时
        """
        if not (0 < decay_factor < 1):
            raise ValueError(f"decay_factor必须 ∈ (0,1)，收到 {decay_factor}")
        if current_weight <= base_weight:
            return base_weight
        new_weight = base_weight + (current_weight - base_weight) * decay_factor
        return round(new_weight, 6)

    def step(
        self,
        trigger: StableExploreTrigger,
        F_history: List[float] = None,
        W_history: List[float] = None,
        duration: float = 0.0,
        is_idle: bool = False,
        is_reforming: bool = False,
        is_rolling_back: bool = False,
        is_alerting: bool = False
    ) -> StableExploreTrigger:
        """执行一步稳态探索检查。

        1. 检查稳态
        2. 检查排除条件
        3. 检查上浮触发
        4. 计算上浮幅度
        5. 安全检查
        6. 衰减处理

        Args:
            trigger: 当前稳态探索触发器状态
            F_history: 自由能历史
            W_history: 权重范数历史
            duration: 稳态持续时间
            is_idle: 是否空闲
            is_reforming: 是否正在重构
            is_rolling_back: 是否正在回滚
            is_alerting: 是否有安全告警

        Returns:
            更新后的稳态探索触发器状态
        """
        state = trigger.trigger_state
        current_weight = state.current_explore_weight
        is_stable = False
        f_var = state.F_variance
        w_var = state.W_variance
        stable_dur = state.stable_duration
        total_boosts = state.total_boosts
        last_boost = state.last_boost_time
        is_boosted = state.is_boosted
        boost_acc = state.boost_accumulation

        # 1. 检查稳态
        if F_history and W_history:
            is_stable, f_var, w_var, stable_dur = self.check_stable(
                F_history, W_history, duration,
                trigger.stable_F_threshold, trigger.stable_W_threshold,
                trigger.min_stable_duration
            )

        # 2. 排除条件
        if is_reforming or is_rolling_back or is_alerting:
            is_stable = False

        # 3. 检查上浮触发
        should_boost = False
        if is_stable and is_idle:
            should_boost = self.trigger_boost(
                is_stable, is_idle, last_boost, trigger.min_boost_interval
            )

        # 4. 计算上浮幅度并执行
        if should_boost:
            boost_amount = self.compute_boost_amount(
                stable_dur, trigger.boost_coefficient, trigger.reference_time
            )
            new_weight, clipped = self.safety_check(
                current_weight, boost_amount, state.max_explore_weight
            )
            current_weight = new_weight
            total_boosts += 1
            last_boost = time.time()
            is_boosted = True
            boost_acc += boost_amount
        elif not is_stable and current_weight > state.base_explore_weight:
            # 5. 稳态打破后衰减
            current_weight = self.decay_weight(
                current_weight, state.base_explore_weight, state.decay_factor
            )
            if abs(current_weight - state.base_explore_weight) < 1e-6:
                is_boosted = False
                boost_acc = 0.0

        new_state = StableExploreTriggerState(
            is_stable=is_stable,
            stable_duration=stable_dur,
            F_variance=f_var,
            W_variance=w_var,
            current_explore_weight=current_weight,
            base_explore_weight=state.base_explore_weight,
            max_explore_weight=state.max_explore_weight,
            total_boosts=total_boosts,
            last_boost_time=last_boost,
            is_boosted=is_boosted,
            boost_accumulation=boost_acc,
            decay_factor=state.decay_factor
        )
        return StableExploreTrigger(
            idle_replay_trigger=trigger.idle_replay_trigger,
            trigger_state=new_state,
            stable_F_threshold=trigger.stable_F_threshold,
            stable_W_threshold=trigger.stable_W_threshold,
            min_stable_duration=trigger.min_stable_duration,
            boost_coefficient=trigger.boost_coefficient,
            reference_time=trigger.reference_time,
            min_boost_interval=trigger.min_boost_interval,
            variance_window=trigger.variance_window
        )
