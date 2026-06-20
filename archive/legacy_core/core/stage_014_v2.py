"""阶段14强化版：感知更新（支持非线性激活与v2模型）。

本文件是 stage_014.py 的强化升级版，核心改进：
1. 支持 HierarchicalGenerativeModelV2（非线性激活）
2. 使用真正的自由能梯度下降（解析梯度，而非简化梯度）
3. 增加学习率自适应（基于梯度范数）
4. 保持与 stage_014.py 的完全向后兼容

核心数学：
    自由能 F = 1/2 * Σ_l ||e^l||²_{Π^l}
    其中 e^l = r^{l-1} - f^l(r^l)（l=0时 e^0 = o - g(r^1)）

    梯度：∂F/∂r^l = -J^{l-1,T}·Π^{l-1}·e^{l-1} + Π^l·e^l
    其中 J^{l-1} = ∂f^{l-1}/∂r^l（生成函数的Jacobian）

    梯度下降更新：r^l ← r^l - α·∂F/∂r^l
"""

from dataclasses import dataclass
from typing import Union
import numpy as np

from agent_whitepaper.core.stage_011 import HierarchicalGenerativeModel
from agent_whitepaper.core.stage_011_v2 import HierarchicalGenerativeModelV2
from agent_whitepaper.core.stage_012_v2 import BottomUpInferenceEngineV2, BottomUpPropagationV2
from agent_whitepaper.core.stage_013_v2 import TopDownPropagationV2


# ============================================================
# 阶段14-v2 数据类：感知更新
# ============================================================
@dataclass(frozen=True)
class PerceptionUpdateV2:
    """感知更新结果（强化版）。

    Attributes:
        top_down: 自顶向下传播结果
        representations_before: 更新前的表征
        representations_after: 更新后的表征
        gradients: 各层梯度（∂F/∂r^l）
        time_constant: 时间常数
        step_size: 步长
        is_converged: 是否收敛
        convergence_threshold: 收敛阈值
        num_steps: 实际迭代步数
        free_energy_history: 自由能历史
        gradient_norms: 最终梯度范数
    """
    top_down: TopDownPropagationV2
    representations_before: tuple[np.ndarray, ...]
    representations_after: tuple[np.ndarray, ...]
    gradients: tuple[np.ndarray, ...]
    time_constant: float
    step_size: float
    is_converged: bool
    convergence_threshold: float
    num_steps: int
    free_energy_history: tuple[float, ...]
    gradient_norms: tuple[float, ...]


# ============================================================
# 阶段14-v2 核心接口：感知更新引擎
# ============================================================
class PerceptionUpdateEngineV2:
    """感知更新引擎（强化版）。

    核心改进：
    1. 使用真正的自由能解析梯度 ∂F/∂r^l
    2. 自适应学习率（基于梯度范数归一化）
    3. 记录自由能下降历史
    4. 每步重新计算误差信号和梯度（真正的梯度下降）
    """

    def update(
        self,
        top_down: TopDownPropagationV2,
        initial_representations: tuple[np.ndarray, ...],
        time_constant: float = 1.0,
        step_size: float = 0.1,
        max_steps: int = 100,
        convergence_threshold: float = 1e-6,
        adaptive_lr: bool = True,
        recompute_gradients: bool = True
    ) -> PerceptionUpdateV2:
        """执行感知更新（真正的自由能梯度下降）。

        Args:
            top_down: 自顶向下传播结果
            initial_representations: 初始表征
            time_constant: 时间常数
            step_size: 步长
            max_steps: 最大迭代步数
            convergence_threshold: 收敛阈值
            adaptive_lr: 是否使用自适应学习率
            recompute_gradients: 是否每步重新计算误差信号和梯度

        Returns:
            PerceptionUpdateV2 实例
        """
        if time_constant <= 0:
            raise ValueError(f"time_constant 必须 > 0，收到 {time_constant}")
        if step_size <= 0:
            raise ValueError(f"step_size 必须 > 0，收到 {step_size}")
        if max_steps < 1:
            raise ValueError(f"max_steps 必须 >= 1，收到 {max_steps}")
        if convergence_threshold <= 0:
            raise ValueError(f"convergence_threshold 必须 > 0，收到 {convergence_threshold}")

        model = top_down.bottom_up.model
        observation = top_down.bottom_up.observation
        reprs = list(initial_representations)
        fe_history = []
        num_steps = 0
        bu_engine = BottomUpInferenceEngineV2()

        # 初始误差信号
        current_errors = top_down.bottom_up.error_signals

        for step in range(max_steps):
            num_steps = step + 1
            # 如果启用梯度重算，每步重新计算误差信号
            if recompute_gradients and observation is not None:
                try:
                    bu_new = bu_engine.propagate(model, observation, tuple(reprs))
                    current_errors = bu_new.error_signals
                except Exception:
                    # 如果重算失败，回退到上一次的误差信号
                    pass
            # 计算自由能解析梯度 ∂F/∂r^l
            grads = self._compute_free_energy_gradient(
                model, observation, tuple(reprs), current_errors
            )
            # 计算当前自由能（用于监控）
            fe = self._compute_free_energy(model, observation, tuple(reprs))
            fe_history.append(fe)
            # 自适应学习率
            alpha = step_size / time_constant
            if adaptive_lr:
                grad_norms = [float(np.linalg.norm(g)) for g in grads]
                max_norm = max(grad_norms) if grad_norms else 1.0
                if max_norm > 1.0:
                    alpha = alpha / max_norm
            # 梯度下降更新：r ← r - α·∂F/∂r
            for l in range(model.num_layers):
                reprs[l] = reprs[l] - alpha * grads[l]
            # 检查收敛
            if self._check_convergence(grads, convergence_threshold):
                break

        # 最终梯度
        if recompute_gradients and observation is not None:
            try:
                bu_final = bu_engine.propagate(model, observation, tuple(reprs))
                final_errors = bu_final.error_signals
            except Exception:
                final_errors = current_errors
        else:
            final_errors = current_errors
        final_grads = self._compute_free_energy_gradient(
            model, observation, tuple(reprs), final_errors
        )
        gradient_norms = tuple(float(np.linalg.norm(g)) for g in final_grads)

        return PerceptionUpdateV2(
            top_down=top_down,
            representations_before=initial_representations,
            representations_after=tuple(reprs),
            gradients=tuple(final_grads),
            time_constant=time_constant,
            step_size=step_size,
            is_converged=num_steps < max_steps,
            convergence_threshold=convergence_threshold,
            num_steps=num_steps,
            free_energy_history=tuple(fe_history),
            gradient_norms=gradient_norms,
        )

    def _compute_free_energy_gradient(
        self,
        model: Union[HierarchicalGenerativeModel, HierarchicalGenerativeModelV2],
        observation: np.ndarray,
        representations: tuple[np.ndarray, ...],
        error_signals: tuple[np.ndarray, ...]
    ) -> tuple[np.ndarray, ...]:
        """计算自由能对各层表征的解析梯度。

        数学推导：
            F = 1/2·||e^0||²_{Π_obs} + 1/2·Σ_{l=1}^{L-1}||e^l||²_{Π^l} + 1/2·||e^L||²_{Π_prior}

            其中：
                e^0 = o - g(r^1)           （观测误差）
                e^l = r^l - f^l(r^{l+1})    （层间误差，l=1..L-1）
                e^L = r^L - μ               （先验误差）

            梯度（r^l 出现在 e^{l-1} 和 e^l 中）：
                ∂F/∂r^l = -J^{l-1,T}·Π^{l-1}·e^{l-1} + Π^l·e^l

            其中 J^{l-1} = ∂f^{l-1}/∂r^l（生成函数Jacobian，形状 child_dim×parent_dim）

        0索引映射：representations[l] = r^{l+1}, error_signals[l] = e^l

        Args:
            model: 层级生成模型
            observation: 观测向量
            representations: 各层表征
            error_signals: 误差信号 (e^0, e^1, ..., e^L)

        Returns:
            各层梯度 tuple (∂F/∂r^1, ∂F/∂r^2, ..., ∂F/∂r^L)
        """
        L = model.num_layers
        is_v2 = isinstance(model, HierarchicalGenerativeModelV2)
        gradients = []

        for l in range(L):
            # representations[l] = r^{l+1}
            # 下方误差: e^l = error_signals[l]（来自子层的误差）
            # 上方误差: e^{l+1} = error_signals[l+1]（来自本层的误差）

            # ---- 计算 J^{l}（下方生成函数的Jacobian）----
            # l=0: J = ∂g/∂r^1（观测生成函数）
            # l>0: J = ∂f^l/∂r^{l+1}（层间生成函数）
            if l == 0:
                if is_v2:
                    J_below = model.obs_generative_fn.jacobian_analytical(
                        representations[0]
                    )
                else:
                    # v1: f(r, W) = W @ r, J = W
                    J_below = model.weights[0]
                pi_below = model.obs_precision
            else:
                if is_v2:
                    J_below = model.generative_fns[l].jacobian_analytical(
                        representations[l]
                    )
                else:
                    J_below = model.weights[l]
                pi_below = model.precisions[l]

            e_below = error_signals[l]  # e^l

            # 下方贡献: -J^{l,T}·Π^l·e^l
            pw_below = pi_below @ e_below  # Π^l·e^l
            grad_below = J_below.T @ pw_below  # J^{l,T}·Π^l·e^l

            # ---- 计算上方精度加权误差 ----
            if l < L - 1:
                # 非顶层: Π^{l+1}·e^{l+1}
                pi_above = model.precisions[l + 1]
                e_above = error_signals[l + 1]
            else:
                # 顶层: Π_prior·e^L
                pi_above = model.prior_precision
                e_above = error_signals[L]
            pw_above = pi_above @ e_above  # Π^{l+1}·e^{l+1}

            # 梯度: ∂F/∂r^{l+1} = -J^{l,T}·Π^l·e^l + Π^{l+1}·e^{l+1}
            gradient = -grad_below + pw_above
            gradients.append(gradient)

        return tuple(gradients)

    def _compute_gradients(
        self,
        propagated_signals: tuple[np.ndarray, ...]
    ) -> tuple[np.ndarray, ...]:
        """[已弃用] 旧的梯度计算方法，保留用于向后兼容。

        注意：此方法返回 -delta，并非真正的自由能梯度。
        新代码应使用 _compute_free_energy_gradient。
        """
        return tuple(-delta for delta in propagated_signals)

    def _compute_free_energy(
        self,
        model: Union[HierarchicalGenerativeModel, HierarchicalGenerativeModelV2],
        observation: np.ndarray,
        representations: tuple[np.ndarray, ...]
    ) -> float:
        """计算当前表征下的自由能 F = -ln p(o, r)。

        Args:
            model: 层级生成模型
            observation: 实际观测向量
            representations: 各层表征
        """
        if observation is None:
            # 如果没有观测，使用零向量作为回退
            observation = np.zeros(model.obs_dim)
        if isinstance(model, HierarchicalGenerativeModelV2):
            return model.compute_free_energy(observation, representations)
        else:
            return -model.joint_log_likelihood(observation, representations)

    def _check_convergence(
        self,
        gradients: tuple[np.ndarray, ...],
        threshold: float
    ) -> bool:
        """检查收敛（基于梯度范数）。"""
        for g in gradients:
            if float(np.linalg.norm(g)) >= threshold:
                return False
        return True
