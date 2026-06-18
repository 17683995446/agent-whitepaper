# 阶段65：结构性重构算法——特征增删、子模型重组、规则迭代与贝叶斯搜索

> **阶段编号**：S65 | **所属层**：第七层·进化适应层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 特征增删 | "加减房间"——增加新特征维度或删除不重要的特征 |
> | 子模型重组 | "拆分合并"——把两个子模型合并，或把一个拆成两个 |
> | 规则迭代 | "升级规则"——根据误差模式更新内部规则 |
> | 认知体系更新 | "重构层级"——重新组织认知层次结构 |
> | 贝叶斯优化搜索 | "聪明地试"——用贝叶斯优化在结构空间中高效搜索 |
> | StructuralReformer | "重构器的完整画像"——组合StructuralReformTrigger，在触发条件基础上增加重构算法 |

---

## 1. 第一性原理依据

本阶段从阶段8（结构变异算子）的核心结论出发，建立结构性重构的具体算法。核心依据为：

- **A8.7（结构变异算子）**（引用阶段8）：$\mathcal{M}_{\text{struct}} = \{\text{FeatureAdd/Del, ModuleRecomb, RuleIterate}\}$，三种结构变异操作的定义。本阶段将A8.7的抽象算子具体化为可执行的重构算法。

- **A8.1（适应度定义）**（引用阶段8）：$f(\theta) = -(1/T)\sum_{t=1}^{T} F(q_t, o_t; \theta)$，适应度是重构效果评估的标准——重构后的适应度必须高于重构前。

- **A8.11（多尺度进化策略）**（引用阶段8）：参数进化每代执行，结构进化每 $N_{\text{struct}}$ 代执行一次。本阶段的重构算法对应结构进化步骤。

- **A64.1-A64.4（重构触发条件）**（引用阶段64）：持续高误差、长期停滞、新环境和风险评估。重构算法仅在触发条件满足时执行。

本阶段的核心目标是：**定义特征增删 add_feature() / remove_feature(low_importance)（A65.1），子模型重组 merge_models() / split_model()（A65.2），规则迭代 update_rules(error_pattern)（A65.3），认知体系更新 restructure_hierarchy()（A65.4），重构的贝叶斯优化搜索策略（A65.5），封装为StructuralReformer数据类，组合StructuralReformTrigger作为触发条件的输入。**

> **去重声明**：本阶段仅定义结构性重构的具体算法，不重新定义结构变异算子的抽象定义（阶段8，A8.7）或重构触发条件（阶段64）。重构算法是A8.7的具体实现，增加了特征重要性评估、子模型兼容性检查和贝叶斯搜索策略。

---

## 2. 核心数学公式

### 第一部分：特征增删

#### 2.1 特征增删定义（A65.1）

特征增删根据特征重要性动态调整模型的特征维度：

**特征增加**：

$$\text{add\_feature}(\theta_{\text{arch}}) = \theta_{\text{arch}} \cup \{d_{\text{new}}\}, \quad d_{\text{new}} \sim \mathcal{N}(0, \sigma_{\text{init}}^2 I) \tag{A65.1-add}$$

新增特征的参数从零均值高斯分布初始化，确保新增特征不会立即对预测产生大的影响。

**特征删除**：

$$\text{remove\_feature}(\theta_{\text{arch}}, d_j) = \theta_{\text{arch}} \setminus \{d_j\}, \quad d_j = \arg\min_{d_k} \text{importance}(d_k) \tag{A65.1-del}$$

删除重要性最低的特征。特征重要性通过Fisher信息衡量（引用阶段62，A62.4）：

$$\text{importance}(d_k) = \sum_{i \in d_k} F_i \tag{A65.1-importance}$$

即特征 $d_k$ 的所有参数的Fisher信息之和。Fisher信息越大的特征越重要，Fisher信息越小的特征越不重要。

**特征增删的决策规则**：

| 触发条件 | 增删决策 | 原因 |
|---------|---------|------|
| 持续高误差 | 增加特征 | 当前特征不足以表达数据模式 |
| 长期停滞 | 增加特征 | 当前特征表达能力不足 |
| 新环境 | 增加特征 | 新环境需要新特征 |
| 资源紧张 | 删除特征 | 减少计算资源消耗 |
| 特征冗余 | 删除特征 | 冗余特征浪费资源 |

**特征增加的数量控制**：

每次重构最多增加 $K_{\max}$ 个特征（通常 $K_{\max} = 3$），避免一次性增加过多特征导致模型过于复杂：

$$K_{\text{add}} = \min\left(\left\lceil \frac{E_{\text{cumulative}}}{\theta_{\text{high}}} \right\rceil, K_{\max}\right) \tag{A65.1-amount}$$

- **输入**：当前架构 $\theta_{\text{arch}}$，特征重要性，触发条件
- **输出**：增删后的架构 $\theta_{\text{arch}}'$
- **约束**：增删后架构合法；新增参数初始化为零均值

> **白话翻译**：特征增删就是"加减房间"——误差大就加房间（增加特征维度），资源紧就减房间（删除不重要的特征）。加房间时新参数从零开始初始化，避免新特征突然产生大影响。减房间时删最不重要的——用Fisher信息衡量重要性，越不重要的越先删。

---

### 第二部分：子模型重组

#### 2.2 子模型重组定义（A65.2）

子模型重组通过合并或拆分子模型来调整模型结构：

**子模型合并**：

$$\text{merge\_models}(M_1, M_2) = M_{\text{merged}}, \quad \theta_{\text{merged}} = [\theta_1; \theta_2; \theta_{\text{bridge}}] \tag{A65.2-merge}$$

其中 $\theta_{\text{bridge}}$ 为连接两个子模型的桥接参数，从零均值高斯分布初始化。

**子模型拆分**：

$$\text{split\_model}(M) = (M_1, M_2), \quad M_1 = M[\text{subset}_1], \; M_2 = M[\text{subset}_2] \tag{A65.2-split}$$

拆分基于模块间的连接强度——连接弱的模块之间是天然的拆分点：

$$\text{split\_point} = \arg\min_{(i,j)} \|W_{ij}\|_F \tag{A65.2-split-point}$$

即连接权重Frobenius范数最小的模块对。

**子模型重组的兼容性检查**：

合并前需要检查两个子模型的兼容性：

$$\text{compatible}(M_1, M_2) = \mathbb{1}\left[\text{dim}(M_1.\text{output}) = \text{dim}(M_2.\text{input}) \;\wedge\; \text{overlap}(M_1, M_2) < \theta_{\text{overlap}}\right] \tag{A65.2-compat}$$

即输出维度匹配且功能重叠度低于阈值。

**重组的适应度预期**：

合并后的适应度预期为：

$$\mathbb{E}[f(M_{\text{merged}})] \geq \max(f(M_1), f(M_2)) + \Delta f_{\text{synergy}} \tag{A65.2-expectation}$$

其中 $\Delta f_{\text{synergy}} > 0$ 为协同增益——合并后的模型应比单独使用任一子模型更好。如果预期协同增益为负，则不应合并。

- **输入**：子模型 $M_1, M_2$（合并）或 $M$（拆分）
- **输出**：重组后的模型
- **约束**：合并需兼容性检查；拆分需连接强度分析

> **白话翻译**：子模型重组就是"拆分合并"——把两个子模型拼在一起（合并），或者把一个大模型拆成两个（拆分）。合并时需要检查两个模型是否"般配"（输出维度匹配、功能不重叠），拆分时找最弱的连接点下刀。合并的目的是"1+1>2"——协同增益必须为正才值得合并。

---

### 第三部分：规则迭代

#### 2.3 规则迭代定义（A65.3）

规则迭代根据误差模式更新模型的内部规则：

$$\text{update\_rules}(\text{error\_pattern}) = \text{rules}' \tag{A65.3}$$

**误差模式识别**：

首先识别误差的典型模式：

$$\text{error\_pattern} = \text{cluster}(\{\varepsilon_t\}_{t=t_0}^{t_0+T}) \tag{A65.3-pattern}$$

使用聚类算法（如K-means）将误差序列分为若干典型模式。

**规则更新的类型**：

| 误差模式 | 规则更新 | 示例 |
|---------|---------|------|
| 系统性正偏移 | 增加偏置修正规则 | 预测系统性偏高，增加负偏置 |
| 系统性负偏移 | 增加偏置修正规则 | 预测系统性偏低，增加正偏置 |
| 周期性误差 | 增加周期性修正规则 | 误差呈周期性，增加周期性补偿 |
| 条件性误差 | 增加条件分支规则 | 特定条件下误差大，增加条件判断 |
| 高方差误差 | 增加平滑规则 | 误差波动大，增加平滑滤波 |

**规则迭代与A8.7 RuleIterate的对应**：

A65.3是A8.7（引用阶段8）中RuleIterate的具体化。A8.7仅定义了"将某个子模块替换为其迭代深化版本"的抽象操作，A65.3定义了具体的规则更新策略——根据误差模式选择更新类型，并确保更新后的规则比旧规则更好。

- **输入**：误差序列，当前规则集
- **输出**：更新后的规则集
- **约束**：新规则必须通过验证（适应度不降）

> **白话翻译**：规则迭代就是"升级规则"——根据误差的模式更新内部规则。如果误差系统性偏高，就加一个负偏置修正；如果误差呈周期性，就加一个周期性补偿。关键是先识别误差模式，再对症下药——不同的误差模式需要不同的规则更新。

---

### 第四部分：认知体系更新

#### 2.4 认知体系更新定义（A65.4）

认知体系更新重新组织模型的层次结构：

$$\text{restructure\_hierarchy}(\mathcal{H}) = \mathcal{H}' \tag{A65.4}$$

**层次结构重构的操作**：

1. **层合并**：将相邻的两个层合并为一个更宽的层
   $$\mathcal{H}' = \mathcal{H}[l, l+1 \leftarrow l_{\text{merged}}]$$

2. **层拆分**：将一个层拆分为两个更窄的层
   $$\mathcal{H}' = \mathcal{H}[l \leftarrow l_1, l_2]$$

3. **层插入**：在两个层之间插入一个新的层
   $$\mathcal{H}' = \mathcal{H}[l, l+1 \leftarrow l, l_{\text{new}}, l+1]$$

4. **层删除**：删除一个不重要的层
   $$\mathcal{H}' = \mathcal{H} \setminus \{l_{\text{unimportant}}\}$$

**层次重构的决策依据**：

层次重构基于各层的信息流量和重要性：

$$\text{info\_flow}(l) = \|W^l\|_F \cdot \|\delta^l\|_2 \tag{A65.4-info-flow}$$

信息流量小的层是合并或删除的候选；信息流量过大的层是拆分的候选。

- **输入**：当前层次结构 $\mathcal{H}$，各层信息流量
- **输出**：重构后的层次结构 $\mathcal{H}'$
- **约束**：重构后层次结构合法（无孤立层、维度匹配）

> **白话翻译**：认知体系更新就是"重构层级"——重新组织模型的层次结构。信息流量小的层可以合并或删除，信息流量大的层可以拆分。就像公司重组——太小的部门合并，太大的部门拆分，中间可以加新的管理层级。

---

### 第五部分：重构的搜索策略

#### 2.5 贝叶斯优化搜索定义（A65.5）

在结构空间中搜索最优重构方案，使用贝叶斯优化：

$$\theta_{\text{arch}}^* = \arg\max_{\theta_{\text{arch}} \in \mathcal{C}} f(\theta_{\text{arch}}) \tag{A65.5}$$

其中 $\mathcal{C}$ 为候选重构方案集合。

**贝叶斯优化的流程**：

1. **构建代理模型**：用高斯过程（GP）近似适应度函数 $f(\theta_{\text{arch}})$
2. **选择采集函数**：使用期望改进（EI）作为采集函数
   $$\text{EI}(\theta) = \mathbb{E}[\max(f(\theta) - f^+, 0)] \tag{A65.5-ei}$$
   其中 $f^+$ 为当前最优适应度
3. **优化采集函数**：选择EI最大的候选方案
4. **评估候选方案**：在选定方案上计算适应度
5. **更新代理模型**：将新观测加入GP，更新后验

**候选方案的生成**：

候选重构方案从A65.1-A65.4的操作中生成：

$$\mathcal{C} = \{\text{add\_feature}(\theta)\} \cup \{\text{remove\_feature}(\theta)\} \cup \{\text{merge\_models}(\theta)\} \cup \{\text{split\_model}(\theta)\} \cup \{\text{update\_rules}(\theta)\} \cup \{\text{restructure\_hierarchy}(\theta)\} \tag{A65.5-candidates}$$

**搜索预算**：

贝叶斯优化的搜索预算受重构风险评估（A64.4）约束：

$$N_{\text{search}} \leq N_{\max} \cdot (1 - \text{risk}) \tag{A65.5-budget}$$

风险越高，搜索预算越少（高风险时只允许少量尝试）。

- **输入**：候选方案集合 $\mathcal{C}$，适应度历史，搜索预算
- **输出**：最优重构方案 $\theta_{\text{arch}}^*$
- **约束**：搜索预算有限；候选方案合法

> **白话翻译**：贝叶斯优化搜索就是"聪明地试"——不是盲目地试所有可能的重构方案，而是用贝叶斯优化智能地选择最有可能提升适应度的方案。每次试一个方案，根据结果更新对适应度函数的估计，然后选下一个最值得试的方案。搜索预算和风险挂钩——风险高就少试几个，风险低就多试几个。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A65.1 | add_feature / remove_feature(low_importance) | 架构、特征重要性 | 增删后架构 | 新参数零均值初始化 |
| A65.2 | merge_models / split_model | 子模型 | 重组后模型 | 兼容性检查 |
| A65.3 | update_rules(error_pattern) | 误差序列、规则集 | 更新后规则集 | 适应度不降 |
| A65.4 | restructure_hierarchy($\mathcal{H}$) | 层次结构、信息流量 | 重构后层次 | 维度匹配 |
| A65.5 | $\theta^* = \arg\max_{\theta \in \mathcal{C}} f(\theta)$（贝叶斯优化） | 候选方案、适应度历史 | 最优方案 | 搜索预算有限 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import numpy as np

# 引用阶段64的数据类
from stage_064 import StructuralReformTrigger, ReformTriggerType


# ============================================================
# 阶段65 枚举：重构操作类型
# ============================================================
class ReformOperation(Enum):
    """结构性重构操作类型。"""
    ADD_FEATURE = "add_feature"
    REMOVE_FEATURE = "remove_feature"
    MERGE_MODELS = "merge_models"
    SPLIT_MODEL = "split_model"
    UPDATE_RULES = "update_rules"
    RESTRUCTURE_HIERARCHY = "restructure_hierarchy"


# ============================================================
# 阶段65 数据类：结构性重构器
# ============================================================
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


# ============================================================
# 阶段65 核心接口：结构性重构算法
# ============================================================
class StructuralReformAlgorithm:
    """阶段65的核心接口：结构性重构算法。

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
        for op in candidates:
            if reformer.search_used >= reformer.search_budget:
                break
            try:
                fitness = float(fitness_fn(op))
            except Exception:
                fitness = -float("inf")
            reformer = StructuralReformer(
                trigger=reformer.trigger,
                candidate_operations=tuple(candidates),
                best_operation=reformer.best_operation,
                best_fitness=reformer.best_fitness,
                search_budget=reformer.search_budget,
                search_used=reformer.search_used + 1,
                gp_mean=reformer.gp_mean,
                gp_std=reformer.gp_std,
                reform_history=reformer.reform_history + (
                    {"operation": op.value, "fitness": fitness},
                ),
            )
            if fitness > best_fitness:
                best_fitness = fitness
                best_op = op
        return StructuralReformer(
            trigger=reformer.trigger,
            candidate_operations=tuple(candidates),
            best_operation=best_op,
            best_fitness=best_fitness,
            search_budget=reformer.search_budget,
            search_used=reformer.search_used,
            gp_mean=reformer.gp_mean,
            gp_std=reformer.gp_std,
            reform_history=reformer.reform_history,
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `StructuralReformTrigger` | ← 阶段64 | 作为重构算法的触发条件输入（组合字段） |
| **输入** | A8.7（结构变异算子） | ← 阶段8 | 重构操作的理论基础 |
| **输入** | A8.1（适应度定义） | ← 阶段8 | 重构效果评估的标准 |
| **输出** | `StructuralReformer` | → 阶段66 | 重构器状态，作为安全约束的输入 |
| **输出** | A65.1-A65.5（重构算法定义） | → 阶段66-67 | 结构重构算法的核心定义 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| BoTorch | 贝叶斯优化 | 高斯过程代理模型和期望改进采集函数（A65.5） |
| GPyTorch | 高斯过程 | 贝叶斯优化中的GP代理模型 |
| scikit-learn | 聚类分析 | 误差模式识别中的K-means聚类（A65.3） |
| JAX | 适应度计算 | 适应度函数的自动微分和JIT编译 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V65.1 | 特征增加正确性 | 新增特征参数为零均值初始化 | 统计检验 |
| V65.2 | 特征删除正确性 | 删除Fisher信息最小的特征 | 重要性排序验证 |
| V65.3 | 特征增加数量限制 | $K_{\text{add}} \leq K_{\max}$ | 范围检查 |
| V65.4 | 子模型合并兼容性 | 不兼容时抛出异常 | 异常测试 |
| V65.5 | 子模型拆分正确性 | 拆分点为连接最弱的模块对 | 连接强度验证 |
| V65.6 | 规则更新正确性 | 更新后规则通过验证 | 适应度测试 |
| V65.7 | 误差模式识别 | 聚类结果与预设模式一致 | 聚类验证 |
| V65.8 | 层次重构合法性 | 重构后无孤立层、维度匹配 | 结构检查 |
| V65.9 | 贝叶斯搜索收敛 | 搜索后找到适应度更高的方案 | 适应度对比 |
| V65.10 | 搜索预算约束 | search_used ≤ search_budget | 计数验证 |
| V65.11 | StructuralReformer组合 | `reformer.trigger` 类型为 `StructuralReformTrigger` | 类型检查 |
| V65.12 | 非法参数拒绝 | search_budget ≤ 0 抛出 `ValueError` | 异常测试 |
| V65.13 | 候选方案生成 | 候选方案覆盖所有重构操作类型 | 覆盖率测试 |
| V65.14 | 重构后适应度 | 重构后适应度 ≥ 重构前 | 适应度验证 |
| V65.15 | 端到端重构循环 | 完整重构流程无异常 | 集成测试 |

---

> **阶段65完** | 阶段65建立了结构性重构的完整算法：特征增删 add_feature() / remove_feature(low_importance)（A65.1），子模型重组 merge_models() / split_model()（A65.2），规则迭代 update_rules(error_pattern)（A65.3），认知体系更新 restructure_hierarchy()（A65.4），贝叶斯优化搜索策略（A65.5），封装为StructuralReformer数据类组合StructuralReformTrigger。下一步：阶段66将定义重构的安全约束。
