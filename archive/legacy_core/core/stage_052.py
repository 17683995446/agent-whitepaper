"""阶段52：误差评估的时间窗口与累积策略——滑动窗口、指数加权与多尺度分析。

本阶段实现误差累积的数学定义和计算接口。
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .stage_051 import StandardizedError


@dataclass(frozen=True)
class ErrorAccumulation:
    """误差累积的完整描述。

    组合阶段51的StandardizedError，在标准化误差的基础上增加
    时间累积策略——滑动窗口、指数加权、多尺度分析。

    Attributes:
        standardized_error: 阶段51的标准化误差（组合字段）
        window_size: 滑动窗口大小
        cumulative_error: 窗口内累积误差 E_W（A52.2）
        ewma_error: 指数加权移动平均 E_ewma（A52.3）
        ewma_alpha: EWMA平滑因子 α
        multi_scale_errors: 多尺度窗口误差 [E_short, E_mid, E_long]
        multi_scale_windows: 多尺度窗口大小 [T_short, T_mid, T_long]
        prev_ewma: 上一时刻EWMA值

    Invariants:
        - window_size > 0
        - cumulative_error >= 0
        - ewma_alpha ∈ (0, 1)
        - multi_scale_errors 各分量 >= 0
    """
    standardized_error: StandardizedError
    window_size: int = 100
    cumulative_error: float = 0.0
    ewma_error: float = 0.0
    ewma_alpha: float = 0.1
    multi_scale_errors: Optional[Tuple[float, float, float]] = None
    multi_scale_windows: Tuple[int, int, int] = (10, 100, 1000)
    prev_ewma: float = 0.0

    def __post_init__(self):
        if self.window_size < 1:
            raise ValueError(f"窗口大小必须 >= 1，收到 {self.window_size}")
        if self.cumulative_error < 0:
            raise ValueError(
                f"累积误差必须 >= 0，收到 {self.cumulative_error}"
            )
        if not (0 < self.ewma_alpha < 1):
            raise ValueError(
                f"EWMA平滑因子必须 ∈ (0,1)，收到 {self.ewma_alpha}"
            )
        if self.multi_scale_errors is not None:
            for i, e in enumerate(self.multi_scale_errors):
                if e < 0:
                    raise ValueError(
                        f"多尺度误差[{i}] 必须 >= 0，收到 {e}"
                    )

    @property
    def current_error(self) -> float:
        """当前标准化误差（全局误差）。"""
        return self.standardized_error.epsilon_global


class ErrorAccumulator:
    """阶段52的核心计算接口。

    实现误差累积策略——滑动窗口、指数加权、多尺度分析。
    核心约束：
        - 累积误差非负
        - EWMA平滑因子 ∈ (0,1)
        - 多尺度误差非负
    """

    DEFAULT_WINDOWS: Tuple[int, int, int] = (10, 100, 1000)

    def compute_cumulative_error(
        self,
        error_history: np.ndarray,
        window_size: int
    ) -> float:
        """计算滑动窗口累积误差（A52.2）。

        E_W = Σ_{s ∈ W} ε_s

        Args:
            error_history: 误差历史序列
            window_size: 窗口大小

        Returns:
            累积误差 >= 0
        """
        if window_size < 1:
            raise ValueError(f"窗口大小必须 >= 1，收到 {window_size}")
        if len(error_history) == 0:
            return 0.0
        window = error_history[-window_size:]
        return float(np.sum(window))

    def compute_ewma(
        self,
        current_error: float,
        prev_ewma: float,
        alpha: float
    ) -> float:
        """计算指数加权移动平均（A52.3）。

        E_ewma = α · ε_t + (1-α) · E_ewma^{prev}

        Args:
            current_error: 当前误差
            prev_ewma: 上一时刻EWMA
            alpha: 平滑因子

        Returns:
            EWMA值
        """
        if not (0 < alpha < 1):
            raise ValueError(f"alpha 必须 ∈ (0,1)，收到 {alpha}")
        return alpha * current_error + (1.0 - alpha) * prev_ewma

    def compute_multi_scale_errors(
        self,
        error_history: np.ndarray,
        windows: Tuple[int, int, int] = DEFAULT_WINDOWS
    ) -> Tuple[float, float, float]:
        """计算多尺度窗口误差。

        Args:
            error_history: 误差历史序列
            windows: 多尺度窗口大小

        Returns:
            (E_short, E_mid, E_long)
        """
        results = []
        for w in windows:
            if len(error_history) == 0:
                results.append(0.0)
            else:
                window = error_history[-w:]
                results.append(float(np.sum(window)))
        return tuple(results)

    def accumulate(
        self,
        standardized_error: StandardizedError,
        error_history: np.ndarray,
        prev_ewma: float = 0.0,
        alpha: float = 0.1,
        window_size: int = 100,
        windows: Tuple[int, int, int] = DEFAULT_WINDOWS,
    ) -> ErrorAccumulation:
        """执行完整的误差累积。

        Args:
            standardized_error: 标准化误差
            error_history: 误差历史序列
            prev_ewma: 上一时刻EWMA
            alpha: EWMA平滑因子
            window_size: 滑动窗口大小
            windows: 多尺度窗口大小

        Returns:
            ErrorAccumulation 实例
        """
        cumulative = self.compute_cumulative_error(error_history, window_size)
        ewma = self.compute_ewma(
            standardized_error.epsilon_global, prev_ewma, alpha
        )
        multi_scale = self.compute_multi_scale_errors(error_history, windows)
        return ErrorAccumulation(
            standardized_error=standardized_error,
            window_size=window_size,
            cumulative_error=cumulative,
            ewma_error=ewma,
            ewma_alpha=alpha,
            multi_scale_errors=multi_scale,
            multi_scale_windows=windows,
            prev_ewma=prev_ewma,
        )
