"""阶段63：微修正Python接口——MicroAdjuster类的统一接口设计。

本阶段实现微修正模块的统一Python接口。
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import numpy as np

from .stage_062 import MicroAdjuster, MicroAdjustAlgorithm


@dataclass(frozen=True)
class MicroAdjusterState:
    """微修正器的完整运行状态。

    组合阶段62的MicroAdjuster，在算法状态的基础上增加
    微修正器的运行状态——模型快照、自由能历史、收敛状态。

    Attributes:
        adjuster: 阶段62的微修正器（组合字段）
        model_snapshot: 当前模型快照（用于回滚）
        free_energy_history: 自由能历史序列
        is_converged: 微修正是否已收敛
        total_adjustments: 总微修正次数
        total_rollbacks: 总回滚次数
        last_adjustment_step: 上次微修正的全局时间步

    Invariants:
        - total_adjustments >= 0
        - total_rollbacks >= 0
        - total_rollbacks <= total_adjustments
        - last_adjustment_step >= 0
    """
    adjuster: MicroAdjuster
    model_snapshot: Optional[Dict] = None
    free_energy_history: Tuple[float, ...] = ()
    is_converged: bool = False
    total_adjustments: int = 0
    total_rollbacks: int = 0
    last_adjustment_step: int = 0

    def __post_init__(self):
        if self.total_adjustments < 0:
            raise ValueError(
                f"总微修正次数必须 >= 0，收到 {self.total_adjustments}"
            )
        if self.total_rollbacks < 0:
            raise ValueError(
                f"总回滚次数必须 >= 0，收到 {self.total_rollbacks}"
            )
        if self.total_rollbacks > self.total_adjustments:
            raise ValueError(
                f"总回滚次数不能超过总微修正次数，"
                f"收到 rollbacks={self.total_rollbacks}, "
                f"adjustments={self.total_adjustments}"
            )

    @property
    def rollback_rate(self) -> float:
        """回滚率（回滚次数/总微修正次数）。"""
        if self.total_adjustments == 0:
            return 0.0
        return self.total_rollbacks / self.total_adjustments

    @property
    def is_healthy(self) -> bool:
        """微修正器是否健康（回滚率 < 20%）。"""
        return self.rollback_rate < 0.2


class MicroAdjusterInterface:
    """阶段63的核心接口：微修正器。

    整合触发判定、权重微调、参数更新和收敛检查，
    提供统一的微修正接口。
    所有方法均为纯函数（无副作用），状态通过
    MicroAdjusterState显式传递。

    核心约束：
        - adjust_weights 返回自由能不增的模型
        - adjust_params 返回后验正定的模型
        - check_convergence 返回收敛标志
    """

    def __init__(self):
        """初始化微修正器接口。"""
        self._algo = MicroAdjustAlgorithm()

    def adjust_weights(
        self,
        state: MicroAdjusterState,
        model: Dict,
        gradient: np.ndarray
    ) -> Tuple[Dict, MicroAdjusterState]:
        """执行权重微调（A63.2）。

        内部流程：触发检查 → EWC梯度 → 总梯度 →
        学习率 → 权重更新 → 安全检查 → 状态更新。

        Args:
            state: 当前微修正器状态
            model: 当前模型（字典形式）
            gradient: 自由能梯度

        Returns:
            (更新后的模型, 更新后的状态)
        """
        if not state.adjuster.trigger.should_trigger:
            return model, state

        new_model = dict(model)
        weights = new_model.get("weights", np.zeros_like(gradient))

        # 计算EWC梯度（如果激活）
        total_gradient = gradient.copy()
        if state.adjuster.ewc_active:
            ewc_grad = self._algo.compute_ewc_gradient(
                weights,
                state.adjuster.anchor_params,
                state.adjuster.fisher_diag,
                state.adjuster.lambda_ewc,
            )
            total_gradient = total_gradient + ewc_grad

        new_weights, new_adjuster = self._algo.adjust_weights(
            state.adjuster, total_gradient, weights
        )
        new_model["weights"] = new_weights

        new_state = MicroAdjusterState(
            adjuster=new_adjuster,
            model_snapshot=new_model,
            free_energy_history=state.free_energy_history,
            is_converged=state.is_converged,
            total_adjustments=state.total_adjustments + 1,
            total_rollbacks=state.total_rollbacks,
            last_adjustment_step=state.last_adjustment_step + 1,
        )
        return new_model, new_state

    def adjust_params(
        self,
        state: MicroAdjusterState,
        model: Dict,
        observation: np.ndarray
    ) -> Tuple[Dict, MicroAdjusterState]:
        """执行贝叶斯参数更新（A63.3）。

        内部流程：触发检查 → Jacobian → 贝叶斯更新 →
        安全检查 → 状态更新。

        Args:
            state: 当前微修正器状态
            model: 当前模型（字典形式）
            observation: 新观测值

        Returns:
            (更新后的模型, 更新后的状态)
        """
        if not state.adjuster.trigger.should_trigger:
            return model, state

        new_model = dict(model)
        mu = new_model.get("mu", observation)
        sigma = new_model.get("sigma", np.eye(len(observation)))
        H = new_model.get("H", np.eye(len(observation)))
        R = new_model.get("R", np.eye(len(observation)))

        new_mu, new_sigma = self._algo.bayesian_update(
            mu, sigma, observation, H, R
        )
        new_model["mu"] = new_mu
        new_model["sigma"] = new_sigma

        new_state = MicroAdjusterState(
            adjuster=state.adjuster,
            model_snapshot=new_model,
            free_energy_history=state.free_energy_history,
            is_converged=state.is_converged,
            total_adjustments=state.total_adjustments + 1,
            total_rollbacks=state.total_rollbacks,
            last_adjustment_step=state.last_adjustment_step + 1,
        )
        return new_model, new_state

    def check_convergence(
        self,
        state: MicroAdjusterState,
        model: Dict,
        history: List[float]
    ) -> Tuple[bool, MicroAdjusterState]:
        """检查微修正是否收敛（A63.4）。

        三重条件：自由能稳定 + 梯度小 + 参数变化小。

        Args:
            state: 当前微修正器状态
            model: 当前模型
            history: 自由能历史

        Returns:
            (是否收敛, 更新后的状态)
        """
        if len(history) < 5:
            return False, state

        fe_stable = self._algo.check_convergence(
            tuple(history), tolerance=1e-6, window=5
        )
        gradient_small = state.adjuster.trigger.gradient_condition_met

        params = model.get("weights", np.array(history[-1:]))
        prev_params = (
            state.model_snapshot.get("weights", params)
            if state.model_snapshot else params
        )
        param_change_small = float(np.linalg.norm(params - prev_params)) < 1e-6

        converged = fe_stable and gradient_small and param_change_small

        new_state = MicroAdjusterState(
            adjuster=state.adjuster,
            model_snapshot=model,
            free_energy_history=tuple(history),
            is_converged=converged,
            total_adjustments=state.total_adjustments,
            total_rollbacks=state.total_rollbacks,
            last_adjustment_step=state.last_adjustment_step,
        )
        return converged, new_state

    def step(
        self,
        state: MicroAdjusterState,
        model: Dict,
        gradient: Optional[np.ndarray] = None,
        observation: Optional[np.ndarray] = None
    ) -> Tuple[Dict, MicroAdjusterState]:
        """执行一步完整的微修正（A63.1-A63.4）。

        根据输入类型自动选择adjust_weights或adjust_params，
        然后检查收敛。

        Args:
            state: 当前微修正器状态
            model: 当前模型
            gradient: 自由能梯度（可选）
            observation: 新观测值（可选）

        Returns:
            (更新后的模型, 更新后的状态)

        Raises:
            ValueError: 当gradient和observation都为None时
        """
        if gradient is None and observation is None:
            raise ValueError("gradient 和 observation 不能同时为 None")

        if gradient is not None:
            new_model, new_state = self.adjust_weights(state, model, gradient)
        else:
            new_model, new_state = self.adjust_params(state, model, observation)

        history = list(new_state.free_energy_history)
        converged, final_state = self.check_convergence(
            new_state, new_model, history
        )
        return new_model, final_state
