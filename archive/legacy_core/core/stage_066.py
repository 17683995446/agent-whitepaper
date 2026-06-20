"""阶段66：重构安全约束——渐进式重构、性能回滚与知识保留的数学保证。

本阶段实现结构性重构的安全约束——渐进式重构、性能回滚与知识保留。
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from enum import Enum
import numpy as np

from .stage_065 import StructuralReformer, ReformOperation


class ReformSafetyStatus(Enum):
    """重构安全状态。"""
    SAFE = "safe"
    ROLLBACK = "rollback"
    PENDING = "pending"
    VIOLATED = "violated"


@dataclass(frozen=True)
class ReformSafetyConstraint:
    """重构安全约束的完整描述。

    组合阶段65的StructuralReformer，在重构器的基础上增加
    安全约束的状态——渐进式重构、回滚条件和知识保留。

    Attributes:
        reformer: 阶段65的重构器（组合字段）
        max_change_ratio: 最大变化比例
        theta_rollback: 回滚阈值
        rollback_window: 回滚验证窗口
        lambda_ewc_max: EWC正则化损失上限
        free_energy_old: 重构前自由能
        free_energy_new: 重构后自由能
        params_old: 重构前参数
        params_new: 重构后参数
        safety_status: 当前安全状态
        should_rollback: 是否应回滚
        rollback_reason: 回滚原因
        knowledge_retained: 知识是否保留
        ewc_loss: EWC正则化损失

    Invariants:
        - max_change_ratio > 0
        - theta_rollback > 0
        - rollback_window > 0
        - lambda_ewc_max > 0
    """
    reformer: StructuralReformer
    max_change_ratio: float = 0.1
    theta_rollback: float = 0.05
    rollback_window: int = 50
    lambda_ewc_max: float = 1.0
    free_energy_old: float = 0.0
    free_energy_new: float = 0.0
    params_old: Optional[np.ndarray] = None
    params_new: Optional[np.ndarray] = None
    safety_status: ReformSafetyStatus = ReformSafetyStatus.PENDING
    should_rollback: bool = False
    rollback_reason: str = ""
    knowledge_retained: bool = True
    ewc_loss: float = 0.0

    def __post_init__(self):
        if self.max_change_ratio <= 0:
            raise ValueError(
                f"最大变化比例必须 > 0，收到 {self.max_change_ratio}"
            )
        if self.theta_rollback <= 0:
            raise ValueError(
                f"回滚阈值必须 > 0，收到 {self.theta_rollback}"
            )
        if self.rollback_window <= 0:
            raise ValueError(
                f"回滚验证窗口必须 > 0，收到 {self.rollback_window}"
            )
        if self.lambda_ewc_max <= 0:
            raise ValueError(
                f"EWC损失上限必须 > 0，收到 {self.lambda_ewc_max}"
            )

    @property
    def free_energy_increased(self) -> bool:
        """自由能是否增加（需要回滚）。"""
        return self.free_energy_new > self.free_energy_old * (1 + self.theta_rollback)

    @property
    def change_ratio(self) -> float:
        """参数变化比例。"""
        if self.params_old is None or self.params_new is None:
            return 0.0
        old_norm = float(np.linalg.norm(self.params_old))
        if old_norm < 1e-12:
            return 0.0
        diff_norm = float(np.linalg.norm(self.params_new - self.params_old))
        return diff_norm / old_norm

    @property
    def change_acceptable(self) -> bool:
        """参数变化是否在可接受范围内。"""
        return self.change_ratio <= self.max_change_ratio


class ReformSafetyChecker:
    """阶段66的核心计算接口。

    实现渐进式重构、性能回滚和知识保留的安全约束检查。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 单次重构变化不超过10%
        - 性能倒退时自动回滚
        - 旧知识通过EWC保留
    """

    def check_change_ratio(
        self,
        params_old: np.ndarray,
        params_new: np.ndarray,
        max_ratio: float
    ) -> bool:
        """检查参数变化比例（A66.1）。

        当参数长度不同时，计算重叠部分的差异比例。

        Args:
            params_old: 重构前参数
            params_new: 重构后参数
            max_ratio: 最大允许变化比例

        Returns:
            变化是否在允许范围内
        """
        old_norm = float(np.linalg.norm(params_old))
        if old_norm < 1e-12:
            return True
        min_len = min(len(params_old), len(params_new))
        if min_len == 0:
            return True
        diff = params_new[:min_len] - params_old[:min_len]
        diff_norm = float(np.linalg.norm(diff))
        return (diff_norm / old_norm) <= max_ratio

    def check_rollback_condition(
        self,
        free_energy_old: float,
        free_energy_new: float,
        theta_rollback: float
    ) -> bool:
        """检查是否需要回滚（A66.2）。

        Args:
            free_energy_old: 重构前自由能
            free_energy_new: 重构后自由能
            theta_rollback: 回滚阈值

        Returns:
            是否需要回滚
        """
        return free_energy_new > free_energy_old * (1 + theta_rollback)

    def compute_rollback_params(
        self,
        params_old: np.ndarray,
        params_new: np.ndarray,
        free_energy_old: float,
        free_energy_new: float,
        theta_rollback: float
    ) -> Tuple[np.ndarray, str]:
        """计算回滚参数（A66.3）。

        Args:
            params_old: 重构前参数
            params_new: 重构后参数
            free_energy_old: 重构前自由能
            free_energy_new: 重构后自由能
            theta_rollback: 回滚阈值

        Returns:
            (回滚后的参数, 回滚原因)
        """
        if free_energy_new > free_energy_old * (1 + theta_rollback):
            return params_old, "自由能上升超过阈值"
        return params_new, ""

    def check_knowledge_retention(
        self,
        ewc_loss: float,
        lambda_ewc_max: float
    ) -> bool:
        """检查知识保留（A66.4）。

        Args:
            ewc_loss: EWC正则化损失
            lambda_ewc_max: EWC损失上限

        Returns:
            知识是否保留
        """
        return ewc_loss <= lambda_ewc_max

    def check_safety(
        self,
        constraint: ReformSafetyConstraint
    ) -> ReformSafetyConstraint:
        """执行完整的安全检查（A66.1-A66.4）。

        Args:
            constraint: 当前安全约束状态

        Returns:
            更新后的安全约束状态
        """
        if constraint.params_old is not None and constraint.params_new is not None:
            change_ok = self.check_change_ratio(
                constraint.params_old,
                constraint.params_new,
                constraint.max_change_ratio,
            )
        else:
            change_ok = True

        rollback_needed = self.check_rollback_condition(
            constraint.free_energy_old,
            constraint.free_energy_new,
            constraint.theta_rollback,
        )

        knowledge_ok = self.check_knowledge_retention(
            constraint.ewc_loss,
            constraint.lambda_ewc_max,
        )

        if rollback_needed:
            status = ReformSafetyStatus.ROLLBACK
            reason = "自由能上升超过阈值，触发回滚"
        elif not change_ok:
            status = ReformSafetyStatus.VIOLATED
            reason = f"参数变化比例 {constraint.change_ratio:.4f} 超过上限 {constraint.max_change_ratio}"
        elif not knowledge_ok:
            status = ReformSafetyStatus.VIOLATED
            reason = f"EWC损失 {constraint.ewc_loss:.4f} 超过上限 {constraint.lambda_ewc_max}"
        else:
            status = ReformSafetyStatus.SAFE
            reason = "所有安全约束满足"

        return ReformSafetyConstraint(
            reformer=constraint.reformer,
            max_change_ratio=constraint.max_change_ratio,
            theta_rollback=constraint.theta_rollback,
            rollback_window=constraint.rollback_window,
            lambda_ewc_max=constraint.lambda_ewc_max,
            free_energy_old=constraint.free_energy_old,
            free_energy_new=constraint.free_energy_new,
            params_old=constraint.params_old,
            params_new=constraint.params_new,
            safety_status=status,
            should_rollback=rollback_needed,
            rollback_reason=reason,
            knowledge_retained=knowledge_ok,
            ewc_loss=constraint.ewc_loss,
        )
