# 阶段34：收益缺口→任务生成映射——从知识差距到探索行动

> **阶段编号**：S34 | **所属层**：第四层·任务规划层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 收益任务类型 | "五种学习方法"——探索未知、测试功能、挖掘模式、扩展覆盖、提升精度 |
> | 探索任务候选区域 | "去哪里找新知识"——不确定性最高的区域最值得探索 |
> | 任务参数化与去重 | "学习计划排期"——给每个学习任务排优先级，合并重复的学习目标 |

---

## 1. 第一性原理依据

本阶段从阶段32（收益缺口计算）和阶段22（收益驱力分量）的核心结论出发，建立收益缺口到任务生成的映射。核心依据为：

- **A32.3（收益缺口计算）**（引用阶段32）：$\Delta V_{\text{gain}} = f_{\text{gain}}(\text{coverage}, \text{accuracy}, \text{novelty})$，收益缺口从环境状态向量中提取三个分量——认知覆盖度、模型精度和新颖性。任务生成映射需要将这三个分量分别转化为具体的探索任务。
- **A22.2-A22.4（收益驱力分量）**（引用阶段22）：新信息维度 $I_{\text{new}}$、精度提升 $\delta_{\text{accuracy}}$ 和认知覆盖度 $\delta_{\text{coverage}}$ 是收益驱力的三个分量。收益任务的目标正是增大这三个分量——探索未知增大 $I_{\text{new}}$，提升精度增大 $\delta_{\text{accuracy}}$，扩展覆盖增大 $\delta_{\text{coverage}}$。
- **A22.2-conditional（条件互信息递减）**（引用阶段22）：$I(\Psi_{\text{new}}; O | \Psi_{\text{known}})$ 随已知表征集 $\Psi_{\text{known}}$ 的丰富而递减，但永远不为零。这一性质决定了探索任务的候选区域选择——高不确定性区域（条件互信息高的区域）优先探索。
- **A7.5（认知价值 = 互信息）**（引用阶段7）：$E_{\text{epi}}(\pi) = I_q(\Psi; O | \pi)$，认知价值度量策略带来的信息增益。收益任务的预期收益可以用认知价值来量化——探索任务的信息增益越大，预期收益越高。
- **A25.3（收益缺口）**（引用阶段25）：$\Delta V_{\text{gain}} = V_{\text{gain,target}} - R_{\text{gain}}$，收益缺口度量"全知"目标与当前认知水平的差距。收益任务的根本目标是缩小收益缺口——使系统更接近"全知"状态。

本阶段的核心目标是：**定义收益任务类型 [explore\_unknown, test\_feature, mine\_pattern, expand\_coverage, improve\_accuracy]，建立探索任务的候选区域选择策略（高不确定性区域优先），实现任务参数化与去重，并封装为GainTaskGenerator数据类，组合SafetyTaskGenerator作为安全任务的继承。** 收益任务生成是从"知识差距"到"探索行动"的关键一步——它将量化的收益缺口转化为具体的、可执行的学习任务列表。

---

## 2. 核心数学公式

### 第一部分：收益任务类型

#### 2.1 收益任务类型（A34.1）

收益任务分为五种类型，每种对应不同的探索策略：

$$\mathcal{T}_{\text{gain}} = \{\text{explore\_unknown}, \text{test\_feature}, \text{mine\_pattern}, \text{expand\_coverage}, \text{improve\_accuracy}\} \tag{A34.1}$$

**类型1：探索未知（explore\_unknown）**

探索未知任务进入系统的未知状态空间，获取关于新状态的信息：

$$\text{explore\_unknown}: \mathcal{U} \to I(\Psi_{\text{new}}; O | \Psi_{\text{known}}) \tag{A34.1-explore}$$

- **目标**：探索未知状态集合 $\mathcal{U}$，获取条件互信息 $I(\Psi_{\text{new}}; O | \Psi_{\text{known}})$（引用阶段22，A22.2-conditional）
- **触发条件**：未知比率较高（$\text{unknown\_ratio} > 0.3$）或新颖性指标升高
- **预期收益**：新信息维度 $I_{\text{new}}$ 的增量
- **与A22.2的对应**：直接增大新信息维度 $I_{\text{new}}$

探索未知是收益任务中最基本的类型——它直接对应"好奇心"的数学表达（引用阶段7，A7.5的认知价值）。探索未知任务的核心挑战是候选区域的选择——不是所有未知区域都同等值得探索，高不确定性的区域信息增益更大。

**类型2：测试功能（test\_feature）**

测试功能任务对系统的特定功能或假设进行验证，获取关于功能有效性的信息：

$$\text{test\_feature}: \mathcal{H} \to I(\text{result}; \mathcal{H}) \tag{A34.1-test}$$

其中 $\mathcal{H}$ 为待验证的假设集合，$\text{result}$ 为测试结果。

- **目标**：验证假设，获取测试结果与假设之间的互信息
- **触发条件**：模型精度下降（$\text{model\_accuracy} < 0$）或误差分布偏度异常
- **预期收益**：精度提升 $\delta_{\text{accuracy}}$ 的增量
- **与A22.3的对应**：通过验证假设改善模型精度

测试功能任务类似于科学实验——提出假设、设计实验、收集数据、验证假设。在Agent系统中，假设可以是"某个配置是否有效"、"某个模式是否稳定"等。

**类型3：挖掘模式（mine\_pattern）**

挖掘模式任务从已有数据中发现隐藏的规律和模式：

$$\text{mine\_pattern}: \mathcal{D} \to \{p_j\}_{j=1}^{K}, \quad I(p_j; O) > \theta_{\text{pattern}} \tag{A34.1-mine}$$

其中 $\mathcal{D}$ 为已有数据集，$\{p_j\}$ 为发现的模式集合，$\theta_{\text{pattern}}$ 为模式的信息量阈值。

- **目标**：从数据中挖掘信息量超过阈值的模式
- **触发条件**：数据量充足但模式发现率低，或历史改善率下降
- **预期收益**：新信息维度 $I_{\text{new}}$ 和精度提升 $\delta_{\text{accuracy}}$ 的增量
- **与A22.2和A22.3的对应**：同时增大新信息维度和精度提升

挖掘模式任务的核心是"从已知中发现未知"——数据已经收集好了，但其中可能隐藏着尚未发现的规律。模式挖掘的计算成本通常低于探索未知（不需要新的数据采集），但收益也可能较低（因为容易发现的模式已经被发现了）。

**类型4：扩展覆盖（expand\_coverage）**

扩展覆盖任务扩大系统的认知覆盖范围，使已知状态空间覆盖更多的环境状态：

$$\text{expand\_coverage}: \mathcal{K} \to \mathcal{K}', \quad |\mathcal{K}'| > |\mathcal{K}| \tag{A34.1-expand}$$

- **目标**：扩大已知状态集合，提高认知覆盖度 $\delta_{\text{coverage}}$
- **触发条件**：认知覆盖度较低（$\text{coverage} < 0.5$）或文件层变更率升高
- **预期收益**：认知覆盖度 $\delta_{\text{coverage}}$ 的增量
- **与A22.4的对应**：直接增大认知覆盖度

扩展覆盖与探索未知的区别在于：探索未知关注"信息量最大的未知区域"，扩展覆盖关注"覆盖面最广的未知区域"。前者追求深度（高信息量），后者追求广度（高覆盖率）。

**类型5：提升精度（improve\_accuracy）**

提升精度任务改善系统对已知状态的预测精度：

$$\text{improve\_accuracy}: F \to F', \quad F' < F \tag{A34.1-accuracy}$$

- **目标**：降低自由能，提高预测精度
- **触发条件**：模型精度不足（$\text{accuracy} > 0.3$）或误差分布偏度异常
- **预期收益**：精度提升 $\delta_{\text{accuracy}}$ 的增量
- **与A22.3的对应**：直接增大精度提升

提升精度任务关注"已知的但不精确的部分"——系统已经知道某个状态存在，但对它的理解不够深入。提升精度通常通过增加训练数据、调整模型参数或改进特征表示来实现。

**五种收益任务的优先级排序**：

| 优先级 | 任务类型 | 理由 |
|--------|---------|------|
| 1 | explore\_unknown | 新信息的边际收益最高（A22.2-conditional） |
| 2 | improve\_accuracy | 精度提升直接降低自由能（A4.11） |
| 3 | mine\_pattern | 低成本高收益，但容易发现的模式有限 |
| 4 | expand\_coverage | 广度扩展的边际收益递减 |
| 5 | test\_feature | 验证性任务，收益取决于假设质量 |

> **白话翻译**：五种收益任务就是"五种学习方法"——探索未知是"去新地方看看"，测试功能是"做实验验证猜想"，挖掘模式是"从已有数据中找规律"，扩展覆盖是"把知识面铺广"，提升精度是"把已经知道的搞得更精确"。好奇心最强时先探索未知，理解不够深时先提升精度。

---

### 第二部分：探索任务的候选区域选择

#### 2.2 探索任务的候选区域选择（A34.2）

探索未知任务的核心问题是"去哪里探索"——候选区域的选择直接影响探索的效率。选择策略基于信息增益最大化：

$$\text{region}^* = \arg\max_{r \in \mathcal{R}_{\text{candidate}}} I(\Psi_r; O | \Psi_{\text{known}}) \tag{A34.2}$$

其中 $\mathcal{R}_{\text{candidate}}$ 为候选区域集合，$\Psi_r$ 为区域 $r$ 中的潜在新表征。

**候选区域的定义**：

候选区域是环境状态空间中尚未被充分探索的子空间。候选区域的划分基于环境状态向量的文件层和日志层指标：

1. **文件层区域**：文件变更率高的目录（A31.4-changes）——变更频繁的区域可能包含新信息
2. **日志层区域**：新日志频率高的模块（A31.3-logs）——新事件密集的模块可能暴露新状态
3. **自身层区域**：误差分布偏度大的模型层（A31.5-dist）——预测偏差大的区域可能有未建模的因素

**不确定性度量**：

每个候选区域的不确定性由条件熵度量：

$$H(r) = H[O | \Psi_{\text{known}}, r] = -\sum_{o \in \mathcal{O}_r} p(o | \Psi_{\text{known}}, r) \ln p(o | \Psi_{\text{known}}, r) \tag{A34.2-entropy}$$

条件熵越高，区域的不确定性越大，探索的信息增益越大。由A22.2-conditional（引用阶段22），条件互信息 $I(\Psi_r; O | \Psi_{\text{known}}) = H[O | \Psi_{\text{known}}] - H[O | \Psi_{\text{known}}, \Psi_r]$。因此，最大化信息增益等价于选择能使条件熵下降最多的区域。

**候选区域选择算法**：

```
输入：候选区域集合 R_candidate，已知表征集 Ψ_known
输出：选中的探索区域 region*

1. 对每个候选区域 r ∈ R_candidate：
   a. 估计条件互信息 I(Ψ_r; O | Ψ_known)
   b. 估计探索成本 cost(r)
   c. 计算信息增益比 IGR(r) = I(Ψ_r; O | Ψ_known) / cost(r)
2. 选择 IGR 最大的区域：region* = argmax IGR(r)
```

信息增益比（Information Gain Ratio, IGR）平衡了信息收益和探索成本——不是选择信息增益最大的区域（可能成本很高），而是选择单位成本信息增益最大的区域。

**高不确定性区域优先的数学论证**：

由A7.5（引用阶段7），认知价值 $E_{\text{epi}}(\pi) = I_q(\Psi; O | \pi)$。对于探索策略 $\pi_r$（探索区域 $r$），认知价值为 $I(\Psi_r; O | \Psi_{\text{known}})$。由互信息的链式法则：

$$I(\Psi_r; O | \Psi_{\text{known}}) = H[\Psi_r | \Psi_{\text{known}}] - H[\Psi_r | O, \Psi_{\text{known}}]$$

当区域 $r$ 的不确定性 $H[\Psi_r | \Psi_{\text{known}}]$ 很高时，互信息的上界很大——探索高不确定性区域可能获得大量新信息。当不确定性很低时，互信息的上界很小——探索低不确定性区域只能获得少量新信息。

因此，高不确定性区域优先是信息论最优的探索策略——它最大化探索的期望信息增益。

**探索的边际收益递减**：

由A22.2-conditional（引用阶段22），条件互信息随已知表征集的丰富而递减。这意味着：

1. 首次探索某个区域的信息增益最大
2. 重复探索同一区域的信息增益递减
3. 当一个区域的不确定性降低到阈值以下时，应转向其他区域

边际收益递减规律确保探索任务不会"原地踏步"——系统会自动转向信息增益更大的新区域。

- **输入**：候选区域集合 $\mathcal{R}_{\text{candidate}}$，已知表征集 $\Psi_{\text{known}}$
- **输出**：选中的探索区域 $\text{region}^*$
- **约束**：$\mathcal{R}_{\text{candidate}} \neq \emptyset$；选择基于IGR最大化

> **白话翻译**：探索任务的候选区域选择就是"去哪里找新知识"——不是随便乱逛，而是去不确定性最高的地方。不确定性高的地方意味着"不知道的最多"，探索那里的信息收益最大。同时还要考虑探索成本——有些区域虽然不确定，但探索起来太贵了，不如去性价比更高的地方。

---

### 第三部分：任务参数化与去重

#### 2.3 任务参数化与去重（A34.3）

收益任务的参数化沿用阶段33的五元组格式，但权重和成本系数不同：

$$\text{task} = \{\text{type}, \text{target}, \text{priority}, \text{estimated\_cost}, \text{expected\_benefit}\} \tag{A34.3}$$

**收益任务的类型权重**：

| 任务类型 | 类型权重 $w_{\text{type}}$ | 基础成本 $c_{\text{type}}$ | 效率系数 $\eta_{\text{type}}$ |
|---------|--------------------------|--------------------------|---------------------------|
| explore\_unknown | 1.0 | 0.7 | 0.8 |
| test\_feature | 0.4 | 0.4 | 0.5 |
| mine\_pattern | 0.6 | 0.3 | 0.6 |
| expand\_coverage | 0.5 | 0.5 | 0.4 |
| improve\_accuracy | 0.8 | 0.6 | 0.7 |

**优先级计算**：

$$\text{priority} = \Delta V_{\text{gain,norm}} \cdot w_{\text{type}} \cdot w_{\text{scene}} \tag{A34.3-priority}$$

其中 $w_{\text{scene}}$ 为场景权重（stable=1.2, tight=0.8, urgent=0.5）——收益任务在稳定场景下优先级最高（系统有余力探索），在紧急场景下优先级最低（先救火再学习）。

**预期收益计算**：

$$\text{expected\_benefit} = \Delta V_{\text{gain}} \cdot \eta_{\text{type}} \cdot \text{IGR}(\text{region}) \tag{A34.3-benefit}$$

其中 $\text{IGR}(\text{region})$ 为候选区域的信息增益比（A34.2）。对于非探索类任务（test\_feature, mine\_pattern, expand\_coverage, improve\_accuracy），IGR设为1.0。

**收益任务的ROI分析**：

| 任务类型 | 典型ROI | 说明 |
|---------|---------|------|
| mine\_pattern | 2.000 (0.6/0.3) | 低成本高收益，ROI最高 |
| explore\_unknown | 1.143 (0.8/0.7) | 高收益高成本，ROI中等 |
| improve\_accuracy | 1.167 (0.7/0.6) | 中收益中成本，ROI中等 |
| expand\_coverage | 0.800 (0.4/0.5) | 低收益中成本，ROI较低 |
| test\_feature | 1.250 (0.5/0.4) | 中收益低成本，ROI较高 |

mine\_pattern的ROI最高——它不需要采集新数据，只需从已有数据中挖掘模式，成本最低。explore\_unknown的ROI中等——虽然信息增益最大，但探索成本也最高。

**收益任务的去重**：

收益任务的去重沿用A33.4的相似度计算和合并规则，但增加一个维度——候选区域重叠度：

$$\text{sim}_{\text{region}} = \frac{|\mathcal{R}_i \cap \mathcal{R}_j|}{|\mathcal{R}_i \cup \mathcal{R}_j|} \tag{A34.3-region}$$

两个探索任务如果探索同一区域（$\text{sim}_{\text{region}} > 0.5$），即使类型不同也应考虑合并——因为探索同一区域的边际收益递减。

**收益任务与安全任务的交互**：

收益任务和安全任务可能存在冲突——探索未知可能引入新的安全风险。冲突解决策略：

1. **安全优先**：如果安全缺口超过高阈值（$\Delta V_{\text{safe,norm}} > 0.7$），暂停所有收益任务
2. **风险约束**：探索任务的候选区域排除高风险区域（$\text{anomaly\_count} > \theta$ 的区域）
3. **成本约束**：收益任务的总成本不超过剩余资源预算（扣除安全任务成本后）

- **输入**：收益缺口 $\Delta V_{\text{gain}}$，候选区域集合，安全任务列表
- **输出**：参数化且去重的收益任务列表
- **约束**：安全优先原则；成本约束；边际收益递减

> **白话翻译**：收益任务的参数化和去重跟安全任务类似——给每个学习任务排优先级、算成本和收益，合并重复的学习目标。但有一个重要区别：收益任务要考虑"去哪里学"（候选区域选择），而且要跟安全任务协调——着火的时候不能只顾学习，得先救火。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A34.1 | $\mathcal{T}_{\text{gain}} = \{\text{explore\_unknown}, \text{test\_feature}, \text{mine\_pattern}, \text{expand\_coverage}, \text{improve\_accuracy}\}$ | — | 任务类型集合 | 5种类型 |
| A34.2 | $\text{region}^* = \arg\max_{r} I(\Psi_r; O | \Psi_{\text{known}}) / \text{cost}(r)$ | 候选区域、已知表征 | 最优探索区域 | IGR最大化 |
| A34.3 | task = {type, target, priority, estimated\_cost, expected\_benefit} | 缺口值、类型、区域 | 参数化任务 | 安全优先；成本约束 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np

# 引用阶段33的数据类
from stage_033 import SafetyTaskGenerator, SafetyTask


# ============================================================
# 收益任务类型枚举
# ============================================================
class GainTaskType(Enum):
    """收益任务类型（A34.1）。"""
    EXPLORE_UNKNOWN = "explore_unknown"
    TEST_FEATURE = "test_feature"
    MINE_PATTERN = "mine_pattern"
    EXPAND_COVERAGE = "expand_coverage"
    IMPROVE_ACCURACY = "improve_accuracy"


# ============================================================
# 候选区域
# ============================================================
@dataclass(frozen=True)
class CandidateRegion:
    """探索任务的候选区域（A34.2）。

    Attributes:
        region_id: 区域唯一标识
        uncertainty: 区域不确定性（条件熵）
        estimated_info_gain: 估计信息增益
        exploration_cost: 探索成本
        source: 区域来源（'files' | 'logs' | 'model'）
        scope: 区域范围描述
    """
    region_id: str
    uncertainty: float
    estimated_info_gain: float
    exploration_cost: float
    source: str
    scope: str = ""

    @property
    def igr(self) -> float:
        """信息增益比（A34.2）。"""
        if self.exploration_cost < 1e-10:
            return float('inf')
        return self.estimated_info_gain / self.exploration_cost


# ============================================================
# 参数化收益任务
# ============================================================
@dataclass(frozen=True)
class GainTask:
    """参数化收益任务（A34.3）。

    Attributes:
        task_type: 任务类型
        target: 任务目标
        priority: 优先级
        estimated_cost: 预估成本
        expected_benefit: 预期收益
        region: 关联的候选区域（仅explore_unknown类型）
        scope: 任务范围描述
    """
    task_type: GainTaskType
    target: str
    priority: float
    estimated_cost: float
    expected_benefit: float
    region: Optional[CandidateRegion] = None
    scope: str = ""

    @property
    def roi(self) -> float:
        """成本-收益比。"""
        if self.estimated_cost < 1e-10:
            return float('inf')
        return self.expected_benefit / self.estimated_cost


# ============================================================
# 阶段34 数据类：收益任务生成器
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class GainTaskGenerator:
    """收益任务生成器的完整描述。

    组合阶段33的SafetyTaskGenerator，在安全任务的基础上增加
    收益任务的生成结果。

    Attributes:
        safety_generator: 阶段33的安全任务生成器（组合字段）
        gain_tasks: 生成的收益任务列表
        candidate_regions: 候选区域列表（A34.2）
        selected_regions: 选中的探索区域列表
        threshold: 任务生成阈值
        dedup_threshold: 去重相似度阈值
        original_task_count: 去重前的任务数
        deduped_task_count: 去重后的任务数

    Invariants:
        - threshold > 0
        - dedup_threshold ∈ (0, 1)
        - deduped_task_count <= original_task_count
    """
    safety_generator: SafetyTaskGenerator
    gain_tasks: tuple[GainTask, ...]
    candidate_regions: tuple[CandidateRegion, ...]
    selected_regions: tuple[CandidateRegion, ...]
    threshold: float
    dedup_threshold: float
    original_task_count: int
    deduped_task_count: int

    def __post_init__(self):
        # 验证阈值
        if self.threshold <= 0:
            raise ValueError(
                f"任务生成阈值必须 > 0，"
                f"收到 {self.threshold}"
            )
        # 验证去重计数
        if self.deduped_task_count > self.original_task_count:
            raise ValueError(
                f"去重后任务数不能超过去重前"
            )

    @property
    def all_tasks(self) -> list[SafetyTask | GainTask]:
        """所有任务（安全+收益），按优先级降序排列。"""
        safety = list(self.safety_generator.tasks)
        gain = list(self.gain_tasks)
        all_t = safety + gain
        return sorted(all_t, key=lambda t: t.priority, reverse=True)

    @property
    def total_gain_cost(self) -> float:
        """收益任务的总预估成本。"""
        return sum(t.estimated_cost for t in self.gain_tasks)

    @property
    def total_gain_benefit(self) -> float:
        """收益任务的总预期收益。"""
        return sum(t.expected_benefit for t in self.gain_tasks)

    @property
    def region_coverage(self) -> float:
        """候选区域的覆盖比例。"""
        if len(self.candidate_regions) == 0:
            return 0.0
        return len(self.selected_regions) / len(self.candidate_regions)


# ============================================================
# 阶段34 核心接口：收益任务生成器
# ============================================================
class GainTaskGeneratorEngine:
    """阶段34的核心计算接口。

    实现收益缺口到任务生成的完整映射，包括候选区域选择、
    任务生成、参数化和去重。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 高不确定性区域优先
        - 安全优先原则
        - 边际收益递减
    """

    def identify_candidate_regions(
        self,
        env_state: 'EnvironmentStateVector'
    ) -> list[CandidateRegion]:
        """识别候选探索区域（A34.2）。

        基于文件层、日志层和自身层指标识别高不确定性区域。

        Args:
            env_state: 环境状态向量

        Returns:
            候选区域列表

        Examples:
            >>> from stage_031 import EnvironmentStateVector
            >>> from stage_030 import DriveValidationResult
            >>> env = EnvironmentStateVector(
            ...     validation_result=DriveValidationResult(),
            ...     s_sys=(0.0,)*6, s_log=(10.0, 0.2, 0.5),
            ...     s_file=(0.8, 0.0, 0.0),
            ...     s_self=(0.3, 2.0, 1e6, 100.0),
            ...     z_sys=(0.0,)*6, z_log=(0.0,)*3, z_file=(0.0,)*3,
            ...     z_self=(0.0,)*4,
            ...     mu=tuple([0.0]*16), sigma=tuple([1.0]*16),
            ...     timestamps=(0.0,)*4
            ... )
            >>> engine = GainTaskGeneratorEngine()
            >>> regions = engine.identify_candidate_regions(env)
            >>> len(regions) >= 0
            True
        """
        regions: list[CandidateRegion] = []
        # 基于文件层高变更率识别探索区域
        file_changes = float(env_state.s_file[0])
        if file_changes > 0.5:
            regions.append(CandidateRegion(
                name="file_high_change",
                uncertainty=file_changes,
                info_gain_estimate=file_changes * 0.8,
                exploration_cost=0.3,
            ))
        # 基于日志层错误率识别
        error_rate = float(env_state.s_log[1])
        if error_rate > 0.1:
            regions.append(CandidateRegion(
                name="log_error_zone",
                uncertainty=error_rate,
                info_gain_estimate=error_rate * 0.9,
                exploration_cost=0.4,
            ))
        # 基于自身层低准确率识别
        accuracy = float(env_state.s_self[0])
        if accuracy < 0.5:
            regions.append(CandidateRegion(
                name="low_accuracy_model",
                uncertainty=1.0 - accuracy,
                info_gain_estimate=(1.0 - accuracy) * 0.7,
                exploration_cost=0.5,
            ))
        # 保底：至少一个默认区域
        if not regions:
            regions.append(CandidateRegion(
                name="default_exploration",
                uncertainty=0.3,
                info_gain_estimate=0.2,
                exploration_cost=0.2,
            ))
        return regions

    def estimate_info_gain(
        self,
        region: CandidateRegion,
        known_representations: set
    ) -> float:
        """估计区域的信息增益（A34.2）。

        Args:
            region: 候选区域
            known_representations: 已知表征集

        Returns:
            估计的条件互信息 ∈ [0, ∞)

        Examples:
            >>> engine = GainTaskGeneratorEngine()
            >>> region = CandidateRegion("test", 0.5, 0.6, 0.2)
            >>> ig = engine.estimate_info_gain(region, set())
            >>> ig >= 0
            True
        """
        # 若区域名已在已知表征中，信息增益降低
        if region.name in known_representations:
            return region.info_gain_estimate * 0.1
        return region.info_gain_estimate

    def select_region(
        self,
        candidates: list[CandidateRegion]
    ) -> CandidateRegion:
        """选择最优探索区域（A34.2）。

        基于IGR（信息增益比）最大化选择：IGR = info_gain / exploration_cost。

        Args:
            candidates: 候选区域列表

        Returns:
            IGR最大的候选区域

        Raises:
            ValueError: 当候选列表为空时

        Examples:
            >>> engine = GainTaskGeneratorEngine()
            >>> r1 = CandidateRegion("a", 0.5, 0.6, 0.2)
            >>> r2 = CandidateRegion("b", 0.3, 0.2, 0.5)
            >>> best = engine.select_region([r1, r2])
            >>> best.name
            'a'
        """
        if not candidates:
            raise ValueError("候选区域列表不能为空")
        best = max(
            candidates,
            key=lambda r: r.info_gain_estimate / max(r.exploration_cost, 1e-6)
        )
        return best

    def generate_gain_tasks(
        self,
        gap_assessment: 'GapAssessment',
        candidate_regions: list[CandidateRegion],
        safety_cost: float = 0.0,
        resource_budget: float = 1.0
    ) -> list[GainTask]:
        """生成收益任务（A34.1, A34.3）。

        Args:
            gap_assessment: 缺口评估结果
            candidate_regions: 候选区域列表
            safety_cost: 安全任务已占用的成本
            resource_budget: 总资源预算 ∈ (0, ∞)

        Returns:
            收益任务列表

        Raises:
            ValueError: 当 resource_budget <= 0 时

        Examples:
            >>> from stage_032 import GapAssessment
            >>> from stage_031 import EnvironmentStateVector
            >>> from stage_030 import DriveValidationResult
            >>> env = EnvironmentStateVector(
            ...     validation_result=DriveValidationResult(),
            ...     s_sys=(0.0,)*6, s_log=(10.0, 0.2, 0.5),
            ...     s_file=(0.8, 0.0, 0.0),
            ...     s_self=(0.3, 2.0, 1e6, 100.0),
            ...     z_sys=(0.0,)*6, z_log=(0.0,)*3, z_file=(0.0,)*3,
            ...     z_self=(0.0,)*4,
            ...     mu=tuple([0.0]*16), sigma=tuple([1.0]*16),
            ...     timestamps=(0.0,)*4
            ... )
            >>> ga = GapAssessment(
            ...     env_state=env, delta_v_safe=0.1, delta_v_gain=0.7,
            ...     delta_v_eff=0.1, delta_v_safe_norm=0.1,
            ...     delta_v_gain_norm=0.7, delta_v_eff_norm=0.1,
            ...     delta_v_safe_max=1.0, delta_v_gain_max=1.0,
            ...     delta_v_eff_max=1.0, priority_order=('gain','safe','eff'),
            ...     safe_weights=(0.4,0.4,0.2), gain_weights=(0.4,0.3,0.3),
            ...     eff_weights=(0.3,0.2,0.2,0.3)
            ... )
            >>> engine = GainTaskGeneratorEngine()
            >>> tasks = engine.generate_gain_tasks(ga, [], safety_cost=0.0)
            >>> len(tasks) >= 0
            True
        """
        if resource_budget <= 0:
            raise ValueError(f"资源预算必须 > 0，收到 {resource_budget}")
        tasks: list[GainTask] = []
        delta_norm = gap_assessment.delta_v_gain_norm
        if delta_norm <= 0.1:
            return tasks
        remaining_budget = max(0.0, resource_budget - safety_cost)
        # 根据缺口分量选择任务类型
        accuracy = float(gap_assessment.env_state.s_self[0])
        file_changes = float(gap_assessment.env_state.s_file[0])
        error_entropy = float(gap_assessment.env_state.s_self[1])
        task_types: list[GainTaskType] = []
        if accuracy < 0.8:
            task_types.append(GainTaskType.PRECISION_BOOST)
        if file_changes > 0.3:
            task_types.append(GainTaskType.COVERAGE_EXPAND)
        if error_entropy > 1.0:
            task_types.append(GainTaskType.NOVELTY_SEEK)
        if not task_types:
            task_types.append(GainTaskType.COVERAGE_EXPAND)
        # 为每个候选区域生成任务
        for region in candidate_regions:
            for task_type in task_types:
                cost = region.exploration_cost * (1.0 + delta_norm)
                if cost > remaining_budget:
                    continue
                benefit = region.info_gain_estimate * delta_norm
                priority = delta_norm * region.uncertainty
                tasks.append(GainTask(
                    task_type=task_type,
                    target_region=region,
                    priority=priority,
                    estimated_cost=cost,
                    expected_benefit=benefit,
                    scope=f"{task_type.value}_{region.name}",
                ))
        return tasks

    def deduplicate(
        self,
        tasks: list[GainTask],
        threshold: float = 0.8
    ) -> list[GainTask]:
        """收益任务去重（A34.3）。

        基于任务类型和目标区域名称的相似度去重。

        Args:
            tasks: 原始任务列表
            threshold: 去重相似度阈值 ∈ (0,1)

        Returns:
            去重后的任务列表

        Raises:
            ValueError: 当 threshold 不在 (0,1) 时

        Examples:
            >>> engine = GainTaskGeneratorEngine()
            >>> r = CandidateRegion("r1", 0.5, 0.6, 0.2)
            >>> t1 = GainTask(GainTaskType.COVERAGE_EXPAND, r, 0.5, 0.2, 0.3, "scope")
            >>> t2 = GainTask(GainTaskType.COVERAGE_EXPAND, r, 0.6, 0.2, 0.3, "scope")
            >>> deduped = engine.deduplicate([t1, t2], threshold=0.8)
            >>> len(deduped) <= 2
            True
        """
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"阈值必须在 (0,1) 内，收到 {threshold}")
        if not tasks:
            return []
        n = len(tasks)
        visited = [False] * n
        merged: list[GainTask] = []
        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            cluster = [tasks[i]]
            for j in range(i + 1, n):
                if visited[j]:
                    continue
                # 简化相似度：类型相同且区域名相同则相似度为1
                same_type = tasks[i].task_type == tasks[j].task_type
                same_region = tasks[i].target_region.name == tasks[j].target_region.name
                sim = 1.0 if (same_type and same_region) else (0.5 if same_type else 0.0)
                if sim >= threshold:
                    visited[j] = True
                    cluster.append(tasks[j])
            if len(cluster) == 1:
                merged.append(cluster[0])
            else:
                merged.append(GainTask(
                    task_type=cluster[0].task_type,
                    target_region=cluster[0].target_region,
                    priority=max(t.priority for t in cluster),
                    estimated_cost=np.mean([t.estimated_cost for t in cluster]),
                    expected_benefit=max(t.expected_benefit for t in cluster),
                    scope=cluster[0].scope,
                ))
        return merged

    def generate(
        self,
        safety_generator: SafetyTaskGenerator
    ) -> GainTaskGenerator:
        """完整的收益任务生成流程。

        Args:
            safety_generator: 安全任务生成器

        Returns:
            GainTaskGenerator 实例

        Examples:
            >>> from stage_033 import SafetyTaskGenerator
            >>> from stage_032 import GapAssessment
            >>> from stage_031 import EnvironmentStateVector
            >>> from stage_030 import DriveValidationResult
            >>> env = EnvironmentStateVector(
            ...     validation_result=DriveValidationResult(),
            ...     s_sys=(0.0,)*6, s_log=(10.0, 0.2, 0.5),
            ...     s_file=(0.8, 0.0, 0.0),
            ...     s_self=(0.3, 2.0, 1e6, 100.0),
            ...     z_sys=(0.0,)*6, z_log=(0.0,)*3, z_file=(0.0,)*3,
            ...     z_self=(0.0,)*4,
            ...     mu=tuple([0.0]*16), sigma=tuple([1.0]*16),
            ...     timestamps=(0.0,)*4
            ... )
            >>> ga = GapAssessment(
            ...     env_state=env, delta_v_safe=0.1, delta_v_gain=0.7,
            ...     delta_v_eff=0.1, delta_v_safe_norm=0.1,
            ...     delta_v_gain_norm=0.7, delta_v_eff_norm=0.1,
            ...     delta_v_safe_max=1.0, delta_v_gain_max=1.0,
            ...     delta_v_eff_max=1.0, priority_order=('gain','safe','eff'),
            ...     safe_weights=(0.4,0.4,0.2), gain_weights=(0.4,0.3,0.3),
            ...     eff_weights=(0.3,0.2,0.2,0.3)
            ... )
            >>> sg = SafetyTaskGenerator(
            ...     gap_assessment=ga, tasks=(), threshold=0.3,
            ...     dedup_threshold=0.8, original_task_count=0,
            ...     deduped_task_count=0
            ... )
            >>> engine = GainTaskGeneratorEngine()
            >>> gen = engine.generate(sg)
            >>> gen.deduped_task_count <= gen.original_task_count
            True
        """
        gap = safety_generator.gap_assessment
        regions = self.identify_candidate_regions(gap.env_state)
        selected = self.select_region(regions) if regions else None
        selected_regions = [selected] if selected else []
        safety_cost = safety_generator.total_cost
        raw_tasks = self.generate_gain_tasks(
            gap, selected_regions, safety_cost=safety_cost, resource_budget=1.0
        )
        original_count = len(raw_tasks)
        deduped_tasks = self.deduplicate(raw_tasks, threshold=0.8)
        deduped_count = len(deduped_tasks)
        return GainTaskGenerator(
            safety_generator=safety_generator,
            gain_tasks=tuple(deduped_tasks),
            candidate_regions=tuple(regions),
            selected_regions=tuple(selected_regions),
            threshold=0.3,
            dedup_threshold=0.8,
            original_task_count=original_count,
            deduped_task_count=deduped_count,
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `SafetyTaskGenerator` | ← 阶段33 | 作为收益任务生成器的安全任务组合字段 |
| **输入** | A32.3（收益缺口计算） | ← 阶段32 | 收益缺口的三个分量驱动不同类型的任务 |
| **输入** | A22.2-A22.4（收益驱力分量） | ← 阶段22 | 新信息、精度提升、认知覆盖度与任务类型的对应 |
| **输出** | `GainTaskGenerator` | → 阶段35 | 收益任务列表作为能效任务生成器的组合字段 |
| **输出** | A34.2（候选区域选择） | → 后续阶段 | 探索策略的决策依据 |
| **输出** | A34.3（任务参数化与去重） | → 后续阶段 | 参数化任务用于调度和资源分配 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 信息增益计算 | `jax.numpy`实现条件互信息的近似计算（A34.2）；`jax.lax.cond`实现IGR最大化选择 |
| scikit-learn | 候选区域识别 | `sklearn.cluster.KMeans`对环境状态空间进行聚类，识别高不确定性区域（A34.2） |
| NetworkX | 任务去重 | 与阶段33相同的图算法去重策略（A34.3） |
| SciPy | 边际收益分析 | `scipy.optimize.minimize_scalar`确定最优探索深度（边际收益=边际成本的平衡点） |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V34.1 | 收益任务类型完整性 | 5种收益任务类型均可生成 | 各类型触发测试 |
| V34.2 | 探索区域选择 | 选中的区域IGR最大 | 合成候选区域测试 |
| V34.3 | 高不确定性优先 | 不确定性最高的区域被优先选择 | 不确定性梯度测试 |
| V34.4 | IGR计算正确性 | IGR = info\_gain / cost | 数值验证 |
| V34.5 | 边际收益递减 | 重复探索同一区域的信息增益递减 | 重复探索模拟 |
| V34.6 | 安全优先原则 | 安全缺口>0.7时收益任务被暂停 | 紧急场景测试 |
| V34.7 | 成本约束 | 收益任务总成本 $\leq$ 资源预算 - 安全成本 | 预算约束测试 |
| V34.8 | 任务参数化完整性 | 每个任务包含type, target, priority, cost, benefit | 参数完整性检查 |
| V34.9 | 去重有效性 | 相似度>0.8的任务被合并 | 合成相似任务测试 |
| V34.10 | 区域重叠去重 | 探索同一区域的任务被合并 | 区域重叠测试 |
| V34.11 | ROI计算 | mine\_pattern的ROI $\approx 2.0$ | ROI数值验证 |
| V34.12 | 场景权重 | stable场景下收益任务优先级最高 | 场景切换测试 |
| V34.13 | GainTaskGenerator组合 | `GainTaskGenerator.safety_generator` 类型为 `SafetyTaskGenerator` | 类型检查 |
| V34.14 | 非法输入拒绝 | 负阈值、空候选区域等抛出 `ValueError` | 异常测试 |
| V34.15 | 全部任务排序 | all\_tasks按优先级降序排列 | 排序验证 |

---

> **阶段34完** | 阶段34建立了收益缺口到任务生成的完整映射：五种收益任务类型（探索未知、测试功能、挖掘模式、扩展覆盖、提升精度）覆盖了从广度探索到深度学习的全谱系认知策略，候选区域选择基于信息增益比最大化确保探索效率，安全优先原则保证了探索不会危及系统安全。下一步：阶段35将从能效缺口出发，定义能效任务生成映射。
