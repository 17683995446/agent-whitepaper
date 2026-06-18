"""阶段77：记忆巩固——转移条件、转移协议、巩固期重放、遗忘曲线与间隔重复。

本阶段实现记忆巩固机制。
"""

from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np

from .stage_076 import OfflineReplay


@dataclass(frozen=True)
class MemoryConsolidation:
    """记忆巩固的完整描述。

    组合阶段76的OfflineReplay，在离线复盘的基础上增加
    记忆巩固的状态。

    Attributes:
        offline_replay: 阶段76的离线复盘（组合字段）
        consolidate_threshold: 巩固年龄阈值（秒）
        forget_tau_base: 基础遗忘时间常数（秒）
        spacing_factor: 间隔扩展因子
        retention_threshold: 保持率阈值
        total_consolidated: 累计巩固样本数
        total_reviewed: 累计复习样本数
        current_step: 当前步数

    Invariants:
        - consolidate_threshold >= 0
        - forget_tau_base > 0
        - spacing_factor > 1
        - 0 < retention_threshold < 1
    """
    offline_replay: OfflineReplay
    consolidate_threshold: float = 1000.0
    forget_tau_base: float = 86400.0
    spacing_factor: float = 2.0
    retention_threshold: float = 0.85
    total_consolidated: int = 0
    total_reviewed: int = 0
    current_step: int = 0

    def __post_init__(self):
        if self.consolidate_threshold < 0:
            raise ValueError(
                f"巩固阈值必须 >= 0，收到 {self.consolidate_threshold}"
            )
        if self.forget_tau_base <= 0:
            raise ValueError(
                f"遗忘时间常数必须 > 0，收到 {self.forget_tau_base}"
            )
        if self.spacing_factor <= 1:
            raise ValueError(
                f"间隔因子必须 > 1，收到 {self.spacing_factor}"
            )
        if not (0 < self.retention_threshold < 1):
            raise ValueError(
                f"保持率阈值必须 ∈ (0,1)，收到 {self.retention_threshold}"
            )


class MemoryConsolidationInterface:
    """阶段77的核心接口：记忆巩固。

    实现转移条件、转移协议、遗忘曲线和间隔重复。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 转移后样本保留
        - 遗忘曲线单调递减
        - 间隔重复τ递增
    """

    def should_consolidate(
        self,
        age: float,
        error: float,
        consolidate_threshold: float,
        persist_threshold: float
    ) -> bool:
        """判定是否应巩固（A77.1）。

        Args:
            age: 样本年龄（秒）
            error: 样本误差
            consolidate_threshold: 年龄阈值
            persist_threshold: 持久化阈值

        Returns:
            当年龄超过阈值且误差超过阈值时返回True
        """
        return age >= consolidate_threshold and error >= persist_threshold

    def transfer(
        self,
        sample: Dict,
        value_score: float
    ) -> Dict:
        """执行转移协议（A77.2）。

        Args:
            sample: 样本字典
            value_score: 样本价值评分

        Returns:
            转移后的样本字典（增加value_score字段）
        """
        result = dict(sample)
        result["value_score"] = value_score
        return result

    def retention(
        self,
        time_elapsed: float,
        tau_forget: float
    ) -> float:
        """计算记忆保持率（A77.4）。

        R(t) = exp(-t / τ_forget)

        Args:
            time_elapsed: 经过时间（秒）
            tau_forget: 遗忘时间常数（秒）

        Returns:
            保持率 ∈ (0, 1]

        Raises:
            ValueError: 当tau_forget <= 0时
        """
        if tau_forget <= 0:
            raise ValueError(
                f"遗忘时间常数必须 > 0，收到 {tau_forget}"
            )
        return float(np.exp(-time_elapsed / tau_forget))

    def next_review_time(
        self,
        current_time: float,
        review_count: int,
        tau_base: float,
        spacing_factor: float
    ) -> float:
        """计算下次复习时间（A77.5）。

        Args:
            current_time: 当前时间
            review_count: 已复习次数
            tau_base: 基础遗忘时间常数
            spacing_factor: 间隔扩展因子

        Returns:
            下次复习时间
        """
        interval = tau_base * (spacing_factor ** review_count)
        return current_time + interval

    def update_tau(
        self,
        tau_current: float,
        spacing_factor: float
    ) -> float:
        """更新遗忘时间常数（A77.4-update）。

        Args:
            tau_current: 当前遗忘时间常数
            spacing_factor: 间隔扩展因子

        Returns:
            更新后的遗忘时间常数
        """
        return tau_current * spacing_factor
