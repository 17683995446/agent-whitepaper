"""阶段67：结构性重构Python接口——StructuralReformer类的统一接口设计。

本阶段实现结构性重构模块的统一Python接口。
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import numpy as np

from .stage_065 import StructuralReformer, ReformOperation, StructuralReformAlgorithm
from .stage_066 import ReformSafetyConstraint, ReformSafetyStatus, ReformSafetyChecker


@dataclass(frozen=True)
class StructuralReformerState:
    """结构性重构器的完整运行状态。

    组合阶段66的ReformSafetyConstraint，在安全约束的基础上增加
    重构器的运行状态——模型快照、验证窗口、收敛状态。

    Attributes:
        safety: 阶段66的安全约束（组合字段）
        model_snapshot: 当前模型快照（用于回滚）
        reform_count: 总重构次数
        rollback_count: 总回滚次数
        last_reform_step: 上次重构的全局时间步
        validation_window: 验证窗口数据
        is_converged: 重构是否已收敛
        pending_reform: 待执行的重构操作

    Invariants:
        - reform_count >= 0
        - rollback_count >= 0
        - rollback_count <= reform_count
        - last_reform_step >= 0
    """
    safety: ReformSafetyConstraint
    model_snapshot: Optional[Dict] = None
    reform_count: int = 0
    rollback_count: int = 0
    last_reform_step: int = 0
    validation_window: Optional[Tuple[float, ...]] = None
    is_converged: bool = False
    pending_reform: Optional[ReformOperation] = None

    def __post_init__(self):
        if self.reform_count < 0:
            raise ValueError(
                f"总重构次数必须 >= 0，收到 {self.reform_count}"
            )
        if self.rollback_count < 0:
            raise ValueError(
                f"总回滚次数必须 >= 0，收到 {self.rollback_count}"
            )
        if self.rollback_count > self.reform_count:
            raise ValueError(
                f"总回滚次数不能超过总重构次数，"
                f"收到 rollbacks={self.rollback_count}, "
                f"reforms={self.reform_count}"
            )

    @property
    def rollback_rate(self) -> float:
        """回滚率（回滚次数/总重构次数）。"""
        if self.reform_count == 0:
            return 0.0
        return self.rollback_count / self.reform_count

    @property
    def is_healthy(self) -> bool:
        """重构器是否健康（回滚率 < 20%）。"""
        return self.rollback_rate < 0.2


class StructuralReformerInterface:
    """阶段67的核心接口：结构性重构器。

    整合触发判定、重构执行、安全检查和回滚机制，
    提供统一的结构重构接口。
    所有方法均为纯函数（无副作用），状态通过
    StructuralReformerState显式传递。

    核心约束：
        - reform_structure 返回通过安全检查的模型
        - rollback 返回与快照一致的模型
        - check_convergence 返回收敛标志
    """

    def __init__(self):
        """初始化结构性重构器接口。"""
        self._algo = StructuralReformAlgorithm()
        self._safety = ReformSafetyChecker()

    def reform_structure(
        self,
        state: StructuralReformerState,
        model: Dict,
        operation: ReformOperation,
        **kwargs
    ) -> Tuple[Dict, StructuralReformerState]:
        """执行结构重构（A67.2）。

        内部流程：触发检查 → 执行重构 → 安全检查 →
        回滚判定 → 状态更新。

        Args:
            state: 当前重构器状态
            model: 当前模型（字典形式）
            operation: 重构操作类型
            **kwargs: 重构操作的额外参数

        Returns:
            (更新后的模型, 更新后的状态)
        """
        if not state.safety.reformer.trigger.any_triggered:
            return model, state

        new_model = dict(model)
        params_old = new_model.get("weights", np.array([]))
        free_energy_old = new_model.get("free_energy", 0.0)

        # 执行重构操作
        if operation == ReformOperation.ADD_FEATURE:
            num = kwargs.get("num_features", 1)
            new_model = self._algo.add_feature(new_model, num_features=num)
        elif operation == ReformOperation.REMOVE_FEATURE:
            importance = kwargs.get("feature_importance", np.array([]))
            num = kwargs.get("num_features", 1)
            new_model = self._algo.remove_feature(new_model, importance, num)
        elif operation == ReformOperation.MERGE_MODELS:
            m1 = kwargs.get("model_1", {})
            m2 = kwargs.get("model_2", {})
            new_model = self._algo.merge_models(m1, m2)
        elif operation == ReformOperation.SPLIT_MODEL:
            strengths = kwargs.get("connection_strengths", np.array([]))
            m1, m2 = self._algo.split_model(new_model, strengths)
            new_model = {"modules": [m1, m2]}
        elif operation == ReformOperation.UPDATE_RULES:
            pattern = kwargs.get("error_pattern", "")
            sequence = kwargs.get("error_sequence", np.array([]))
            rules = new_model.get("rules", {})
            new_model["rules"] = self._algo.update_rules(rules, pattern, sequence)
        elif operation == ReformOperation.RESTRUCTURE_HIERARCHY:
            flows = kwargs.get("info_flows", np.array([]))
            hierarchy = new_model.get("hierarchy", {})
            new_model["hierarchy"] = self._algo.restructure_hierarchy(hierarchy, flows)

        params_new = new_model.get("weights", params_old)
        free_energy_new = new_model.get("free_energy", free_energy_old)

        # 安全检查
        new_safety = ReformSafetyConstraint(
            reformer=state.safety.reformer,
            max_change_ratio=state.safety.max_change_ratio,
            theta_rollback=state.safety.theta_rollback,
            rollback_window=state.safety.rollback_window,
            lambda_ewc_max=state.safety.lambda_ewc_max,
            free_energy_old=free_energy_old,
            free_energy_new=free_energy_new,
            params_old=params_old if params_old.size > 0 else None,
            params_new=params_new if params_new.size > 0 else None,
        )
        checked_safety = self._safety.check_safety(new_safety)

        # 回滚判定
        if checked_safety.should_rollback:
            new_model = state.model_snapshot if state.model_snapshot else model
            rollback_count = state.rollback_count + 1
        else:
            rollback_count = state.rollback_count

        new_state = StructuralReformerState(
            safety=checked_safety,
            model_snapshot=new_model if not checked_safety.should_rollback else state.model_snapshot,
            reform_count=state.reform_count + 1,
            rollback_count=rollback_count,
            last_reform_step=state.last_reform_step + 1,
            validation_window=state.validation_window,
            is_converged=state.is_converged,
            pending_reform=operation,
        )
        return new_model, new_state

    def rollback(
        self,
        state: StructuralReformerState,
        model: Dict
    ) -> Tuple[Dict, StructuralReformerState]:
        """执行回滚（A67.3）。

        内部流程：快照恢复 → 冷却期延长 → EWC锚点更新 → 状态更新。

        Args:
            state: 当前重构器状态
            model: 当前模型

        Returns:
            (回滚后的模型, 更新后的状态)
        """
        rolled_back_model = state.model_snapshot if state.model_snapshot else model
        new_state = StructuralReformerState(
            safety=ReformSafetyConstraint(
                reformer=state.safety.reformer,
                safety_status=ReformSafetyStatus.ROLLBACK,
                should_rollback=True,
                rollback_reason="手动触发回滚",
            ),
            model_snapshot=state.model_snapshot,
            reform_count=state.reform_count,
            rollback_count=state.rollback_count + 1,
            last_reform_step=state.last_reform_step,
            validation_window=state.validation_window,
            is_converged=False,
            pending_reform=None,
        )
        return rolled_back_model, new_state

    def check_convergence(
        self,
        state: StructuralReformerState,
        model: Dict,
        history: List[float]
    ) -> Tuple[bool, StructuralReformerState]:
        """检查重构是否收敛（A67.4）。

        三重条件：自由能稳定 + 适应度提升 + 结构变化小。

        Args:
            state: 当前重构器状态
            model: 当前模型
            history: 自由能历史

        Returns:
            (是否收敛, 更新后的状态)
        """
        if len(history) < 5:
            return False, state

        recent = history[-5:]
        fe_stable = max(recent) - min(recent) < 1e-6

        fitness = model.get("fitness", 0.0)
        old_fitness = state.model_snapshot.get("fitness", 0.0) if state.model_snapshot else 0.0
        fitness_improved = fitness > old_fitness

        params = model.get("weights", np.array(recent[-1:]))
        old_params = state.model_snapshot.get("weights", params) if state.model_snapshot else params
        if hasattr(params, 'size') and params.size > 0 and hasattr(old_params, 'size') and old_params.size > 0:
            min_len = min(params.size, old_params.size)
            if min_len > 0:
                diff = params[:min_len] - old_params[:min_len]
                param_change_small = float(np.linalg.norm(diff)) < 1e-6
            else:
                param_change_small = True
        else:
            param_change_small = True

        converged = fe_stable and fitness_improved and param_change_small

        new_state = StructuralReformerState(
            safety=state.safety,
            model_snapshot=model,
            reform_count=state.reform_count,
            rollback_count=state.rollback_count,
            last_reform_step=state.last_reform_step,
            validation_window=tuple(history),
            is_converged=converged,
            pending_reform=state.pending_reform,
        )
        return converged, new_state

    def step(
        self,
        state: StructuralReformerState,
        model: Dict,
        operation: Optional[ReformOperation] = None,
        **kwargs
    ) -> Tuple[Dict, StructuralReformerState]:
        """执行一步完整的结构重构（A67.1-A67.4）。

        根据操作类型自动选择重构方法，然后检查收敛。

        Args:
            state: 当前重构器状态
            model: 当前模型
            operation: 重构操作类型（可选）
            **kwargs: 重构操作的额外参数

        Returns:
            (更新后的模型, 更新后的状态)

        Raises:
            ValueError: 当operation为None时
        """
        if operation is None:
            raise ValueError("operation 不能为 None")

        new_model, new_state = self.reform_structure(state, model, operation, **kwargs)

        history = list(new_state.validation_window) if new_state.validation_window else []
        converged, final_state = self.check_convergence(new_state, new_model, history)
        return new_model, final_state
