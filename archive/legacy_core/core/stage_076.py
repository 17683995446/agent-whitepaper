"""阶段76：空闲离线复盘——空闲判定、批量重放、批量优化、收益评估与终止条件。

本阶段实现空闲离线复盘机制。
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from .stage_075 import GenerativeReplay


@dataclass(frozen=True)
class OfflineReplay:
    """空闲离线复盘的完整描述。

    组合阶段75的GenerativeReplay，在生成式回放的基础上增加
    离线复盘的状态。

    Attributes:
        generative_replay: 阶段75的生成式回放（组合字段）
        idle_threshold: 空闲判定CPU阈值（%）
        batch_size: 离线复盘批量大小
        min_improvement: 最小收益阈值
        max_rounds: 最大复盘轮次
        debounce_steps: 防抖动步数
        current_round: 当前复盘轮次
        total_benefit: 累计复盘收益
        last_replay_time: 上次复盘时间

    Invariants:
        - idle_threshold >= 0
        - batch_size > 0
        - max_rounds > 0
        - debounce_steps >= 0
    """
    generative_replay: GenerativeReplay
    idle_threshold: float = 10.0
    batch_size: int = 128
    min_improvement: float = 0.001
    max_rounds: int = 10
    debounce_steps: int = 5
    current_round: int = 0
    total_benefit: float = 0.0
    last_replay_time: float = 0.0

    def __post_init__(self):
        if self.idle_threshold < 0:
            raise ValueError(
                f"空闲阈值必须 >= 0，收到 {self.idle_threshold}"
            )
        if self.batch_size <= 0:
            raise ValueError(
                f"批量大小必须 > 0，收到 {self.batch_size}"
            )
        if self.max_rounds <= 0:
            raise ValueError(
                f"最大轮次必须 > 0，收到 {self.max_rounds}"
            )
        if self.debounce_steps < 0:
            raise ValueError(
                f"防抖动步数必须 >= 0，收到 {self.debounce_steps}"
            )


class OfflineReplayInterface:
    """阶段76的核心接口：空闲离线复盘。

    实现空闲判定、批量重放、收益评估和终止条件。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 空闲时触发复盘
        - 收益下降时终止
        - 防抖动避免频繁触发
    """

    def is_idle(
        self,
        cpu_usage: float,
        has_active_task: bool,
        idle_threshold: float
    ) -> bool:
        """空闲判定（A76.1）。

        Args:
            cpu_usage: CPU使用率，单位%，范围[0,100]
            has_active_task: 是否有活跃任务
            idle_threshold: 空闲阈值

        Returns:
            当CPU使用率低于阈值且无活跃任务时返回True
        """
        return cpu_usage < idle_threshold and not has_active_task

    def evaluate_benefit(
        self,
        free_energy_before: float,
        free_energy_after: float
    ) -> Tuple[float, float]:
        """评估复盘收益（A76.4）。

        Args:
            free_energy_before: 复盘前自由能
            free_energy_after: 复盘后自由能

        Returns:
            (absolute_benefit, relative_benefit)
            - absolute_benefit: F_before - F_after
            - relative_benefit: (F_before - F_after) / F_before * 100%

        Raises:
            ValueError: 当F_before <= 0时
        """
        if free_energy_before <= 0:
            raise ValueError(
                f"复盘前自由能必须 > 0，收到 {free_energy_before}"
            )
        abs_benefit = free_energy_before - free_energy_after
        rel_benefit = (abs_benefit / free_energy_before) * 100.0
        return float(abs_benefit), float(rel_benefit)

    def should_continue(
        self,
        current_round: int,
        max_rounds: int,
        current_benefit: float,
        min_improvement: float,
        prev_benefit: float
    ) -> bool:
        """判定是否继续复盘（A76.5）。

        Args:
            current_round: 当前轮次
            max_rounds: 最大轮次
            current_benefit: 当前轮收益
            min_improvement: 最小收益阈值
            prev_benefit: 上一轮收益

        Returns:
            当未达最大轮次且收益未递减时返回True
        """
        if current_round >= max_rounds:
            return False
        if current_round > 0 and current_benefit < prev_benefit:
            return False
        if current_benefit < min_improvement:
            return False
        return True

    def step(
        self,
        offline_replay: OfflineReplay,
        cpu_usage: float,
        has_active_task: bool,
        free_energy_before: float,
        free_energy_after: float
    ) -> Tuple[OfflineReplay, bool]:
        """执行一步离线复盘（A76-step）。

        Args:
            offline_replay: 当前离线复盘状态
            cpu_usage: CPU使用率
            has_active_task: 是否有活跃任务
            free_energy_before: 复盘前自由能
            free_energy_after: 复盘后自由能

        Returns:
            (更新后的状态, 是否继续复盘)
        """
        idle = self.is_idle(
            cpu_usage, has_active_task, offline_replay.idle_threshold
        )
        if not idle:
            return offline_replay, False
        abs_ben, rel_ben = self.evaluate_benefit(
            free_energy_before, free_energy_after
        )
        new_total = offline_replay.total_benefit + rel_ben
        new_round = offline_replay.current_round + 1
        new_state = OfflineReplay(
            generative_replay=offline_replay.generative_replay,
            idle_threshold=offline_replay.idle_threshold,
            batch_size=offline_replay.batch_size,
            min_improvement=offline_replay.min_improvement,
            max_rounds=offline_replay.max_rounds,
            debounce_steps=offline_replay.debounce_steps,
            current_round=new_round,
            total_benefit=new_total,
            last_replay_time=free_energy_after,
        )
        should_cont = self.should_continue(
            new_round, offline_replay.max_rounds,
            rel_ben, offline_replay.min_improvement,
            offline_replay.total_benefit
        )
        return new_state, should_cont
