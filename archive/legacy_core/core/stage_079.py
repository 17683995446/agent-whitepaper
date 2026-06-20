"""阶段79：记忆管理器Python接口——统一整合STM+LTM+价值评估+回放+巩固+快照。

本阶段实现记忆管理器的统一接口。
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from enum import Enum
import numpy as np

from .stage_078 import ModelSnapshot


class ReplayMode(Enum):
    """回放模式。"""
    PRIORITY = "priority"
    GENERATIVE = "generative"


@dataclass(frozen=True)
class MemoryManagerState:
    """记忆管理器的完整状态描述。

    组合阶段78的ModelSnapshot，在快照基础上增加统一管理状态。

    Attributes:
        model_snapshot: 阶段78的模型快照（组合字段）
        total_samples_stored: 累计存储样本数
        total_samples_consolidated: 累计巩固样本数
        total_replays: 累计回放次数
        total_snapshots: 累计快照次数
        total_rollbacks: 累计回滚次数
        current_step: 当前步数
        is_idle: 是否空闲
        last_consolidate_step: 上次巩固步数
        last_snapshot_step: 上次快照步数

    Invariants:
        - total_samples_stored >= 0
        - total_samples_consolidated >= 0
        - total_replays >= 0
        - current_step >= 0
    """
    model_snapshot: ModelSnapshot
    total_samples_stored: int = 0
    total_samples_consolidated: int = 0
    total_replays: int = 0
    total_snapshots: int = 0
    total_rollbacks: int = 0
    current_step: int = 0
    is_idle: bool = False
    last_consolidate_step: int = 0
    last_snapshot_step: int = 0

    def __post_init__(self):
        if self.total_samples_stored < 0:
            raise ValueError(
                f"累计存储数必须 >= 0，收到 {self.total_samples_stored}"
            )
        if self.current_step < 0:
            raise ValueError(
                f"当前步数必须 >= 0，收到 {self.current_step}"
            )
