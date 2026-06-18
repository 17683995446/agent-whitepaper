"""阶段71：在线短期记忆——环形缓冲区设计、FIFO写入与排序读取。

本阶段实现在线短期记忆的环形缓冲区机制。
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum
import numpy as np

from .stage_070 import EvolutionValidationResult


class QueryType(Enum):
    """短期记忆读取查询类型。"""
    TOP_ERROR = "top_error"
    TOP_REWARD = "top_reward"
    RECENT = "recent"
    RANGE = "range"


@dataclass(frozen=True)
class MemorySample:
    """短期记忆中的单个样本。

    Attributes:
        observation: 观测向量
        prediction: 预测向量
        error: 全局误差（A51.5标准化误差）
        reward: 奖励信号
        timestamp: 时间戳

    Invariants:
        - error >= 0
        - timestamp >= 0
    """
    observation: np.ndarray
    prediction: np.ndarray
    error: float
    reward: float
    timestamp: float

    def __post_init__(self):
        if self.error < 0:
            raise ValueError(
                f"误差必须 >= 0，收到 {self.error}"
            )
        if self.timestamp < 0:
            raise ValueError(
                f"时间戳必须 >= 0，收到 {self.timestamp}"
            )


@dataclass(frozen=True)
class ShortTermMemory:
    """在线短期记忆的完整描述。

    组合阶段70的EvolutionValidationResult，在进化验证状态的基础上
    增加环形缓冲区的短期记忆。

    Attributes:
        validation_result: 阶段70的进化验证结果（组合字段）
        buffer_size: 缓冲区容量N
        write_ptr: 写入指针
        count: 当前样本数
        is_full: 缓冲区是否已满
        samples: 缓冲区中的样本列表

    Invariants:
        - buffer_size > 0
        - 0 <= write_ptr < buffer_size
        - 0 <= count <= buffer_size
        - is_full == (count == buffer_size)
    """
    validation_result: EvolutionValidationResult
    buffer_size: int = 10000
    write_ptr: int = 0
    count: int = 0
    is_full: bool = False
    samples: Tuple[Optional[MemorySample], ...] = ()

    def __post_init__(self):
        if self.buffer_size <= 0:
            raise ValueError(
                f"缓冲区大小必须 > 0，收到 {self.buffer_size}"
            )
        if not (0 <= self.write_ptr < self.buffer_size):
            raise ValueError(
                f"写入指针必须在 [0, {self.buffer_size}) 范围内，"
                f"收到 {self.write_ptr}"
            )
        if not (0 <= self.count <= self.buffer_size):
            raise ValueError(
                f"样本数必须在 [0, {self.buffer_size}] 范围内，"
                f"收到 {self.count}"
            )

    @property
    def memory_depth_seconds(self) -> float:
        """记忆深度（秒），假设10Hz表层循环。"""
        return self.count / 10.0

    @property
    def utilization(self) -> float:
        """缓冲区利用率。"""
        if self.buffer_size == 0:
            return 0.0
        return self.count / self.buffer_size


class ShortTermMemoryManager:
    """阶段71的核心接口：在线短期记忆管理器。

    实现环形缓冲区的写入和读取策略。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 写入时间O(1)
        - FIFO覆盖最旧样本
        - 读取不阻塞写入
    """

    def write(
        self,
        stm: ShortTermMemory,
        item: MemorySample
    ) -> ShortTermMemory:
        """写入样本到环形缓冲区（A71.3）。

        FIFO覆盖策略：缓冲区满时覆盖最旧样本。

        Args:
            stm: 当前短期记忆状态
            item: 待写入的样本

        Returns:
            更新后的短期记忆状态
        """
        samples_list: List[Optional[MemorySample]] = list(stm.samples)
        if len(samples_list) < stm.buffer_size:
            samples_list.append(item)
        else:
            samples_list[stm.write_ptr] = item
        new_count = min(stm.count + 1, stm.buffer_size)
        new_ptr = (stm.write_ptr + 1) % stm.buffer_size
        new_is_full = new_count == stm.buffer_size
        return ShortTermMemory(
            validation_result=stm.validation_result,
            buffer_size=stm.buffer_size,
            write_ptr=new_ptr,
            count=new_count,
            is_full=new_is_full,
            samples=tuple(samples_list)
        )

    def read(
        self,
        stm: ShortTermMemory,
        query_type: QueryType,
        k: int,
        time_range: Optional[Tuple[float, float]] = None
    ) -> List[MemorySample]:
        """从缓冲区读取样本（A71.4）。

        按查询类型排序返回前k个样本。

        Args:
            stm: 当前短期记忆状态
            query_type: 查询类型
            k: 返回数量
            time_range: 时间范围（仅RANGE查询需要）

        Returns:
            排序后的样本列表

        Raises:
            ValueError: 当k > count时
            ValueError: 当RANGE查询但time_range为None时
        """
        if k > stm.count:
            raise ValueError(
                f"请求数量k({k})不能超过当前样本数({stm.count})"
            )
        if query_type == QueryType.RANGE and time_range is None:
            raise ValueError(
                "RANGE查询必须提供time_range参数"
            )
        valid_samples = [s for s in stm.samples if s is not None]
        if query_type == QueryType.TOP_ERROR:
            sorted_samples = sorted(
                valid_samples, key=lambda x: x.error, reverse=True
            )
        elif query_type == QueryType.TOP_REWARD:
            sorted_samples = sorted(
                valid_samples, key=lambda x: x.reward, reverse=True
            )
        elif query_type == QueryType.RECENT:
            sorted_samples = sorted(
                valid_samples, key=lambda x: x.timestamp, reverse=True
            )
        elif query_type == QueryType.RANGE:
            t_start, t_end = time_range
            sorted_samples = [
                s for s in valid_samples
                if t_start <= s.timestamp <= t_end
            ]
            sorted_samples.sort(key=lambda x: x.timestamp)
        else:
            raise ValueError(f"未知的查询类型: {query_type}")
        return sorted_samples[:k]

    def read_by_error(
        self,
        stm: ShortTermMemory,
        k: int
    ) -> List[MemorySample]:
        """按误差降序读取前k个样本（A71.4-error）。"""
        return self.read(stm, QueryType.TOP_ERROR, k)

    def read_by_reward(
        self,
        stm: ShortTermMemory,
        k: int
    ) -> List[MemorySample]:
        """按奖励降序读取前k个样本（A71.4-reward）。"""
        return self.read(stm, QueryType.TOP_REWARD, k)

    def create_buffer(
        self,
        validation_result: EvolutionValidationResult,
        buffer_size: int = 10000
    ) -> ShortTermMemory:
        """创建短期记忆缓冲区（A71.1, A71.5）。

        Args:
            validation_result: 进化验证结果
            buffer_size: 缓冲区大小，必须 > 0

        Returns:
            初始化的短期记忆状态

        Raises:
            ValueError: 当buffer_size <= 0时
        """
        if buffer_size <= 0:
            raise ValueError(
                f"缓冲区大小必须 > 0，收到 {buffer_size}"
            )
        return ShortTermMemory(
            validation_result=validation_result,
            buffer_size=buffer_size,
            write_ptr=0,
            count=0,
            is_full=False,
            samples=tuple([None] * buffer_size)
        )
