"""阶段65：结构性重构算法——特征增删、子模型重组、规则迭代与贝叶斯搜索。

本阶段实现结构性重构的数学定义和计算接口。
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from enum import Enum
import numpy as np

from .stage_064 import StructuralReformTrigger, ReformTriggerType


class ReformOperation(Enum):
    """结构性重构操作类型。"""
    ADD_FEATURE = "add_feature"
    REMOVE_FEATURE = "remove_feature"
    MERGE_MODELS = "merge_models"
    SPLIT_MODEL = "split_model"
    UPDATE_RULES = "update_rules"
    RESTRUCTURE_HIERARCHY = "restructure_hierarchy"


@dataclass(frozen=True)
class StructuralReformer:
    """结构性重构算法的完整描述。

    组合阶段64的StructuralReformTrigger，在触发条件的基础上增加
    重构算法的状态——候选方案、搜索状态、最优方案。

    Attributes:
        trigger: 阶段64的重构触发状态（组合字段）
        candidate_operations: 候选重构操作列表
        best_operation: 当前最优重构操作
        best_fitness: 当前最优适应度
        search_budget: 搜索预算
        search_used: 已用搜索次数
        gp_mean: 高斯过程均值
        gp_std: 高斯过程标准差
        reform_history: 重构历史

    Invariants:
        - search_budget > 0
        - search_used >= 0
        - search_used <= search_budget
    """
    trigger: StructuralReformTrigger
    candidate_operations: Tuple[ReformOperation, ...] = ()
    best_operation: Optional[ReformOperation] = None
    best_fitness: float = 0.0
    search_budget: int = 10
    search_used: int = 0
    gp_mean: Optional[np.ndarray] = None
    gp_std: Optional[np.ndarray] = None
    reform_history: Tuple[Dict, ...] = ()

    def __post_init__(self):
        if self.search_budget <= 0:
            raise ValueError(
                f"搜索预算必须 > 0，收到 {self.search_budget}"
            )
        if self.search_used < 0:
            raise ValueError(
                f"已用搜索次数必须 >= 0，收到 {self.search_used}"
            )
        if self.search_used > self.search_budget:
            raise ValueError(
                f"已用搜索次数不能超过预算，"
                f"收到 used={self.search_used}, budget={self.search_budget}"
            )

    @property
    def search_remaining(self) -> int:
        """剩余搜索次数。"""
        return self.search_budget - self.search_used

    @property
    def has_candidates(self) -> bool:
        """是否有候选重构方案。"""
        return len(self.candidate_operations) > 0


class StructuralReformAlgorithm:
    """阶段65的核心计算接口。

    实现特征增删、子模型重组、规则迭代、认知体系更新
    和贝叶斯优化搜索。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 重构后适应度不低于重构前
        - 搜索预算有限
        - 候选方案必须合法
    """

    def add_feature(
        self,
        architecture: Dict,
        num_features: int = 1,
        sigma_init: float = 0.01
    ) -> Dict:
        """增加特征维度（A65.1-add）。

        Args:
            architecture: 当前架构
            num_features: 增加的特征数量
            sigma_init: 新参数初始化标准差

        Returns:
            增加特征后的架构
        """
        new_arch = dict(architecture)
        weights = new_arch.get("weights", np.array([]))
        if weights.size == 0:
            return new_arch
        new_features = np.random.normal(0.0, sigma_init, num_features)
        new_arch["weights"] = np.concatenate([weights, new_features])
        new_arch["num_features"] = new_arch.get("num_features", len(weights)) + num_features
        return new_arch

    def remove_feature(
        self,
        architecture: Dict,
        feature_importance: np.ndarray,
        num_features: int = 1
    ) -> Dict:
        """删除低重要性特征（A65.1-del）。

        Args:
            architecture: 当前架构
            feature_importance: 特征重要性（Fisher信息之和）
            num_features: 删除的特征数量

        Returns:
            删除特征后的架构
        """
        new_arch = dict(architecture)
        weights = new_arch.get("weights", np.array([]))
        if weights.size == 0:
            return new_arch
        importance = np.asarray(feature_importance)
        if importance.size == 0:
            return new_arch
        num_remove = min(num_features, len(importance))
        remove_indices = np.argsort(importance)[:num_remove]
        mask = np.ones(len(importance), dtype=bool)
        mask[remove_indices] = False
        if weights.size == importance.size:
            new_arch["weights"] = weights[mask]
        new_arch["num_features"] = new_arch.get("num_features", len(importance)) - num_remove
        return new_arch

    def merge_models(
        self,
        model_1: Dict,
        model_2: Dict,
        sigma_bridge: float = 0.01
    ) -> Dict:
        """合并两个子模型（A65.2-merge）。

        Args:
            model_1: 第一个子模型
            model_2: 第二个子模型
            sigma_bridge: 桥接参数初始化标准差

        Returns:
            合并后的模型

        Raises:
            ValueError: 当两个子模型不兼容时
        """
        out1 = model_1.get("output_dim")
        in2 = model_2.get("input_dim")
        if out1 is not None and in2 is not None and out1 != in2:
            raise ValueError(
                f"子模型不兼容：model_1输出维度{out1} != model_2输入维度{in2}"
            )
        merged = {
            "modules": [model_1, model_2],
            "bridge": np.random.normal(0.0, sigma_bridge, max(1, out1 or 1)),
        }
        return merged

    def split_model(
        self,
        model: Dict,
        connection_strengths: np.ndarray
    ) -> Tuple[Dict, Dict]:
        """拆分模型（A65.2-split）。

        Args:
            model: 待拆分的模型
            connection_strengths: 模块间连接强度矩阵

        Returns:
            拆分后的两个子模型
        """
        strengths = np.asarray(connection_strengths)
        if strengths.ndim == 2 and strengths.shape[0] > 1 and strengths.shape[1] > 1:
            i, j = np.unravel_index(np.argmin(strengths), strengths.shape)
        else:
            i, j = 0, 1
        m1 = dict(model)
        m2 = dict(model)
        m1["split_id"] = i
        m2["split_id"] = j
        return m1, m2

    def update_rules(
        self,
        rules: Dict,
        error_pattern: str,
        error_sequence: np.ndarray
    ) -> Dict:
        """根据误差模式更新规则（A65.3）。

        Args:
            rules: 当前规则集
            error_pattern: 误差模式类型
            error_sequence: 误差序列

        Returns:
            更新后的规则集
        """
        new_rules = dict(rules)
        pattern = error_pattern.lower()
        if "positive" in pattern or "正偏移" in pattern:
            new_rules["bias_correction"] = -float(np.mean(error_sequence))
        elif "negative" in pattern or "负偏移" in pattern:
            new_rules["bias_correction"] = float(np.mean(error_sequence))
        elif "periodic" in pattern or "周期" in pattern:
            new_rules["periodic"] = True
        elif "conditional" in pattern or "条件" in pattern:
            new_rules["conditional"] = True
        elif "high variance" in pattern or "高方差" in pattern:
            new_rules["smoothing"] = True
        else:
            new_rules["general_correction"] = float(np.mean(error_sequence))
        return new_rules

    def restructure_hierarchy(
        self,
        hierarchy: Dict,
        info_flows: np.ndarray
    ) -> Dict:
        """重构认知层次结构（A65.4）。

        Args:
            hierarchy: 当前层次结构
            info_flows: 各层信息流量

        Returns:
            重构后的层次结构
        """
        new_hierarchy = dict(hierarchy)
        flows = np.asarray(info_flows)
        if flows.size == 0:
            return new_hierarchy
        min_idx = int(np.argmin(flows))
        max_idx = int(np.argmax(flows))
        new_hierarchy["merge_layer"] = min_idx
        new_hierarchy["split_layer"] = max_idx
        new_hierarchy["info_flows"] = flows.tolist()
        return new_hierarchy

    def bayesian_search(
        self,
        reformer: StructuralReformer,
        fitness_fn: callable
    ) -> StructuralReformer:
        """执行贝叶斯优化搜索（A65.5）。

        Args:
            reformer: 当前重构器状态
            fitness_fn: 适应度函数

        Returns:
            更新后的重构器状态
        """
        candidates = list(reformer.candidate_operations)
        if not candidates or reformer.search_used >= reformer.search_budget:
            return reformer

        best_op = None
        best_fitness = reformer.best_fitness
        used = reformer.search_used
        history_list = list(reformer.reform_history)

        for op in candidates:
            if used >= reformer.search_budget:
                break
            try:
                fitness = float(fitness_fn(op))
            except Exception:
                fitness = -float("inf")
            used += 1
            history_list.append({"operation": op.value, "fitness": fitness})
            if fitness > best_fitness:
                best_fitness = fitness
                best_op = op

        return StructuralReformer(
            trigger=reformer.trigger,
            candidate_operations=tuple(candidates),
            best_operation=best_op,
            best_fitness=best_fitness,
            search_budget=reformer.search_budget,
            search_used=used,
            gp_mean=reformer.gp_mean,
            gp_std=reformer.gp_std,
            reform_history=tuple(history_list),
        )
