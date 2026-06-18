"""Stage 14: Perception Update (Gradient Descent).

感知更新，通过梯度下降优化表征。
"""

from dataclasses import dataclass
import numpy as np

from agent_whitepaper.core.stage_013 import TopDownPropagation


# ============================================================
# Stage 14 Data Class: Perception Update
# ============================================================
@dataclass(frozen=True)
class PerceptionUpdate:
    """感知更新结果。

    Attributes:
        top_down: 自顶向下传播结果
        representations_before: 更新前的表征
        representations_after: 更新后的表征
        gradients: 各层梯度
        time_constant: 时间常数
        step_size: 步长
        is_converged: 是否收敛
        convergence_threshold: 收敛阈值
        num_steps: 实际迭代步数
    """
    top_down: TopDownPropagation
    representations_before: tuple[np.ndarray, ...]
    representations_after: tuple[np.ndarray, ...]
    gradients: tuple[np.ndarray, ...]
    time_constant: float
    step_size: float
    is_converged: bool
    convergence_threshold: float
    num_steps: int


# ============================================================
# Stage 14 Core Interface: Perception Update Engine
# ============================================================
class PerceptionUpdateEngine:
    """感知更新引擎。"""

    def update(
        self,
        top_down: TopDownPropagation,
        initial_representations: tuple[np.ndarray, ...],
        time_constant: float = 1.0,
        step_size: float = 0.1,
        max_steps: int = 100,
        convergence_threshold: float = 1e-6
    ) -> PerceptionUpdate:
        """执行感知更新。"""
        if time_constant <= 0:
            raise ValueError(f"time_constant 必须 > 0，收到 {time_constant}")
        if step_size <= 0:
            raise ValueError(f"step_size 必须 > 0，收到 {step_size}")
        if max_steps < 1:
            raise ValueError(f"max_steps 必须 >= 1，收到 {max_steps}")
        if convergence_threshold <= 0:
            raise ValueError(f"convergence_threshold 必须 > 0，收到 {convergence_threshold}")

        model = top_down.bottom_up.model
        reprs = list(initial_representations)
        gradients = []
        num_steps = 0

        for step in range(max_steps):
            num_steps = step + 1
            # 计算梯度
            grads = self._compute_gradients(top_down.bottom_up.propagated_signals, top_down.bottom_up.error_signals)
            gradients = grads
            # 更新表征
            alpha = step_size / time_constant
            for l in range(model.num_layers):
                reprs[l] = reprs[l] + alpha * grads[l]
            # 检查收敛
            if self._check_convergence(top_down.bottom_up.propagated_signals, top_down.bottom_up.error_signals, convergence_threshold):
                break

        return PerceptionUpdate(
            top_down=top_down,
            representations_before=initial_representations,
            representations_after=tuple(reprs),
            gradients=gradients,
            time_constant=time_constant,
            step_size=step_size,
            is_converged=num_steps < max_steps,
            convergence_threshold=convergence_threshold,
            num_steps=num_steps,
        )

    def _compute_gradients(
        self,
        propagated_signals: tuple[np.ndarray, ...],
        error_signals: tuple[np.ndarray, ...]
    ) -> tuple[np.ndarray, ...]:
        """计算各层梯度。

        propagated_signals 有 L 个元素 (δ^1, ..., δ^L)
        error_signals 有 L+1 个元素 (e^0, e^1, ..., e^L)
        我们使用 e^1 到 e^L 与 propagated_signals 对应
        """
        L = len(propagated_signals)
        gradients = []
        for l in range(L):
            gradients.append(propagated_signals[l] - error_signals[l + 1])
        return tuple(gradients)

    def _check_convergence(
        self,
        propagated_signals: tuple[np.ndarray, ...],
        error_signals: tuple[np.ndarray, ...],
        threshold: float
    ) -> bool:
        """检查收敛。"""
        for l in range(len(propagated_signals)):
            diff = propagated_signals[l] - error_signals[l + 1]
            if float(np.linalg.norm(diff)) >= threshold:
                return False
        return True
