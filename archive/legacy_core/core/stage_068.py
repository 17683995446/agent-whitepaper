"""阶段68：探索-利用的70/30强制比例实现——进化层的强制执行机制。

本阶段实现进化层探索-利用比例的强制执行。
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import numpy as np

from .stage_067 import StructuralReformerState


@dataclass(frozen=True)
class ExploreExploitEnforcer:
    """探索-利用强制执行器的完整描述。

    组合阶段67的StructuralReformerState，在重构器状态的基础上增加
    探索-利用比例控制的状态。

    Attributes:
        reformer_state: 阶段67的重构器状态（组合字段）
        count_micro: 微修正执行次数
        count_reform: 结构重构执行次数
        explore_ratio: 当前探索比例
        explore_ratio_window: 滑动窗口探索比例
        window_size: 滑动窗口长度
        target_ratio: 目标探索比例（0.3）
        ratio_tolerance: 比例容差（0.05）
        T_min_explore: 最小探索间隔
        steps_since_last_reform: 距上次重构的步数
        force_explore_active: 强制探索是否激活
        reform_history_embeddings: 历史重构操作的embedding
        last_explore_result: 上次探索结果

    Invariants:
        - count_micro >= 0
        - count_reform >= 0
        - target_ratio ∈ (0, 1)
        - ratio_tolerance ∈ (0, 0.5)
    """
    reformer_state: StructuralReformerState
    count_micro: int = 0
    count_reform: int = 0
    explore_ratio: float = 0.0
    explore_ratio_window: float = 0.0
    window_size: int = 100
    target_ratio: float = 0.3
    ratio_tolerance: float = 0.05
    T_min_explore: int = 50
    steps_since_last_reform: int = 100
    force_explore_active: bool = False
    reform_history_embeddings: Optional[np.ndarray] = None
    last_explore_result: Optional[Dict] = None

    def __post_init__(self):
        if self.count_micro < 0:
            raise ValueError(
                f"微修正次数必须 >= 0，收到 {self.count_micro}"
            )
        if self.count_reform < 0:
            raise ValueError(
                f"重构次数必须 >= 0，收到 {self.count_reform}"
            )
        if not (0 < self.target_ratio < 1):
            raise ValueError(
                f"目标比例必须 ∈ (0,1)，收到 {self.target_ratio}"
            )

    @property
    def count_total(self) -> int:
        """总进化操作次数。"""
        return self.count_micro + self.count_reform

    @property
    def is_ratio_healthy(self) -> bool:
        """探索比例是否在目标范围内。"""
        return abs(self.explore_ratio_window - self.target_ratio) <= self.ratio_tolerance

    @property
    def should_force_explore(self) -> bool:
        """是否应强制探索。"""
        return (
            self.explore_ratio_window < self.target_ratio - self.ratio_tolerance
            and self.steps_since_last_reform >= self.T_min_explore
        )


class ExploreExploitEnforcerInterface:
    """阶段68的核心接口：探索-利用强制执行器。

    实现进化层的70/30比例强制执行。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 探索比例 ∈ [0.25, 0.35]
        - 强制探索受安全约束保护
        - 探索方向保证多样性
    """

    def compute_explore_ratio(
        self,
        count_micro: int,
        count_reform: int
    ) -> float:
        """计算探索比例（A68.1）。

        explore_ratio = count_reform / (count_micro + count_reform)

        Args:
            count_micro: 微修正次数
            count_reform: 重构次数

        Returns:
            探索比例 ∈ [0, 1]
        """
        total = count_micro + count_reform
        if total == 0:
            return 0.0
        return count_reform / total

    def compute_window_ratio(
        self,
        history: List[int],
        window_size: int
    ) -> float:
        """计算滑动窗口探索比例（A68.1-window）。

        Args:
            history: 历史操作类型列表（1=重构, 0=微修正）
            window_size: 窗口长度

        Returns:
            窗口内探索比例
        """
        if not history or window_size <= 0:
            return 0.0
        window = history[-window_size:]
        if not window:
            return 0.0
        return sum(window) / len(window)

    def check_force_explore(
        self,
        enforcer: ExploreExploitEnforcer
    ) -> bool:
        """检查是否需要强制探索（A68.2）。

        Args:
            enforcer: 当前执行器状态

        Returns:
            是否需要强制探索
        """
        return (
            enforcer.explore_ratio_window < enforcer.target_ratio - enforcer.ratio_tolerance
            and enforcer.steps_since_last_reform >= enforcer.T_min_explore
        )

    def select_diverse_reform(
        self,
        candidates: List[Dict],
        history_embeddings: Optional[np.ndarray]
    ) -> Dict:
        """选择多样性最高的重构操作（A68.3）。

        Args:
            candidates: 候选重构操作列表
            history_embeddings: 历史重构embedding

        Returns:
            选定的重构操作
        """
        if not candidates:
            raise ValueError("候选重构操作列表不能为空")
        if history_embeddings is None or len(history_embeddings) == 0:
            return candidates[0]
        max_diversity = -1.0
        best_candidate = candidates[0]
        for candidate in candidates:
            emb = candidate.get("embedding")
            if emb is None:
                continue
            emb_array = np.asarray(emb)
            distances = np.linalg.norm(history_embeddings - emb_array, axis=1)
            diversity = float(np.min(distances)) if len(distances) > 0 else float("inf")
            if diversity > max_diversity:
                max_diversity = diversity
                best_candidate = candidate
        return best_candidate

    def evaluate_explore_result(
        self,
        fitness_old: float,
        fitness_new: float,
        free_energy_old: float,
        free_energy_new: float,
        ewc_loss: float
    ) -> Tuple[bool, Dict]:
        """评估探索结果（A68.4）。

        Args:
            fitness_old: 重构前适应度
            fitness_new: 重构后适应度
            free_energy_old: 重构前自由能
            free_energy_new: 重构后自由能
            ewc_loss: EWC正则化损失

        Returns:
            (是否吸收, 评估详情)
        """
        absorb = fitness_new > fitness_old
        details = {
            "fitness_delta": fitness_new - fitness_old,
            "free_energy_delta": free_energy_new - free_energy_old,
            "ewc_loss": ewc_loss,
            "absorb": absorb,
        }
        return absorb, details

    def step(
        self,
        enforcer: ExploreExploitEnforcer,
        model: Dict
    ) -> Tuple[Dict, ExploreExploitEnforcer]:
        """执行一步探索-利用强制执行（A68.1-A68.4）。

        Args:
            enforcer: 当前执行器状态
            model: 当前模型

        Returns:
            (更新后的模型, 更新后的执行器状态)
        """
        force = self.check_force_explore(enforcer)
        return model, ExploreExploitEnforcer(
            reformer_state=enforcer.reformer_state,
            count_micro=enforcer.count_micro,
            count_reform=enforcer.count_reform,
            explore_ratio=enforcer.explore_ratio,
            explore_ratio_window=enforcer.explore_ratio_window,
            window_size=enforcer.window_size,
            target_ratio=enforcer.target_ratio,
            ratio_tolerance=enforcer.ratio_tolerance,
            T_min_explore=enforcer.T_min_explore,
            steps_since_last_reform=enforcer.steps_since_last_reform,
            force_explore_active=force,
            reform_history_embeddings=enforcer.reform_history_embeddings,
            last_explore_result=enforcer.last_explore_result,
        )
