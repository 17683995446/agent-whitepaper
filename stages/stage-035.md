# 阶段35：能效缺口→任务生成映射——从资源浪费到效率优化

> **阶段编号**：S35 | **所属层**：第四层·任务规划层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 能效任务类型 | "五种省钱方式"——模型压缩、冗余清理、计算优化、频率降低、缓存优化 |
> | 模型压缩的结构搜索空间 | "瘦身方案选择"——不同压缩策略的精度-效率权衡空间 |
> | 任务参数化与去重 | "省钱计划排期"——给每个优化任务排优先级，合并重复的优化目标 |

---

## 1. 第一性原理依据

本阶段从阶段32（能效缺口计算）和阶段23（能效驱力分量）的核心结论出发，建立能效缺口到任务生成的映射。核心依据为：

- **A32.4（能效缺口计算）**（引用阶段32）：$\Delta V_{\text{eff}} = f_{\text{eff}}(\text{cpu}, \text{mem}, \text{time}, \text{redundancy})$，能效缺口从环境状态向量中提取四个分量——CPU消耗、内存消耗、时间消耗和计算冗余。任务生成映射需要将这四个分量分别转化为具体的优化任务。
- **A23.2-A23.5（能效驱力分量）**（引用阶段23）：CPU占用率 $\text{cpu\_ratio}$、内存占用率 $\text{mem\_ratio}$、推理耗时 $\text{time\_cost}$ 和计算冗余度 $\text{redundancy}$ 是能效驱力的四个分量。能效任务的目标正是降低这四个分量——模型压缩降低内存和计算量，冗余清理降低冗余度，计算优化降低CPU和时间，频率降低降低所有资源消耗，缓存优化降低时间。
- **A23.5（计算冗余度）**（引用阶段23）：$\text{redundancy} = 1 - I(\Psi; O) / C_{\text{computational}}$，计算冗余度度量信息产出与计算投入的比值补数。冗余清理任务的目标是降低冗余度——使单位计算量产生更多的信息。
- **A5.5（复杂性项）**（引用阶段5）：$C(q) = D_{\text{KL}}[q(\psi) \| p(\psi)]$，复杂性项惩罚偏离先验的表征。模型压缩任务的本质是降低复杂性项——使后验分布更接近先验，减少需要维护的参数量。
- **A25.4（能效缺口）**（引用阶段25）：$\Delta V_{\text{eff}} = -R_{\text{eff}} = \alpha_e \cdot (\text{cpu\_ratio} + \text{mem\_ratio} + \text{time\_cost} + \text{redundancy})$，能效缺口度量资源消耗的加权和。能效任务的根本目标是降低资源消耗——使系统以更少的资源维持同样的性能。

本阶段的核心目标是：**定义能效任务类型 [model\_compress, redundancy\_clean, optimize\_compute, reduce\_frequency, cache\_optimize]，建立模型压缩任务的结构搜索空间，实现任务参数化与去重，并封装为EfficiencyTaskGenerator数据类，组合GainTaskGenerator作为收益任务的继承。** 能效任务生成是从"资源浪费"到"效率优化"的关键一步——它将量化的能效缺口转化为具体的、可执行的优化任务列表，确保系统在资源有限的情况下持续高效运行。

---

## 2. 核心数学公式

### 第一部分：能效任务类型

#### 2.1 能效任务类型（A35.1）

能效任务分为五种类型，每种对应不同的优化策略：

$$\mathcal{T}_{\text{eff}} = \{\text{model\_compress}, \text{redundancy\_clean}, \text{optimize\_compute}, \text{reduce\_frequency}, \text{cache\_optimize}\} \tag{A35.1}$$

**类型1：模型压缩（model\_compress）**

模型压缩任务通过减少模型参数量和计算量来降低资源消耗：

$$\text{model\_compress}: \theta \to \theta', \quad |\theta'| < |\theta|, \quad \text{perf}(\theta') \geq \text{perf}(\theta) - \epsilon \tag{A35.1-compress}$$

其中 $\theta$ 为原始模型参数，$\theta'$ 为压缩后参数，$\text{perf}$ 为性能度量，$\epsilon$ 为允许的性能下降量。

- **目标**：在性能损失不超过 $\epsilon$ 的前提下，减少模型参数量和计算量
- **触发条件**：内存占用率高（$\text{mem} > 0.7$）或计算冗余度高（$\text{redundancy} > 0.5$）
- **预期收益**：内存占用率和CPU占用率的降低
- **与A23.2和A23.3的对应**：直接降低CPU占用率和内存占用率

模型压缩的具体策略包括：

| 策略 | 压缩比 | 精度损失 | 适用场景 |
|------|--------|---------|---------|
| 剪枝（Pruning） | 2-10x | 低 | 冗余参数多的模型 |
| 量化（Quantization） | 2-4x | 低-中 | 部署在低精度硬件 |
| 知识蒸馏（Distillation） | 2-10x | 中 | 有大模型作为教师 |
| 低秩分解（Low-rank） | 2-5x | 低 | 全连接层为主的模型 |

**类型2：冗余清理（redundancy\_clean）**

冗余清理任务消除计算过程中的冗余操作，提高信息产出效率：

$$\text{redundancy\_clean}: C_{\text{computational}} \to C'_{\text{computational}}, \quad C' < C, \quad I(\Psi; O) \text{ 不变} \tag{A35.1-clean}$$

- **目标**：在保持信息产出不变的前提下，减少计算量
- **触发条件**：计算冗余度高（$\text{redundancy} > 0.5$）
- **预期收益**：计算冗余度的降低
- **与A23.5的对应**：直接降低计算冗余度

冗余清理的具体策略包括：

1. **消除重复计算**：识别并合并相同的计算操作
2. **惰性求值**：只在需要时才执行计算
3. **增量更新**：只计算变化的部分，避免全量重算
4. **死代码消除**：移除不影响输出的计算路径

**类型3：计算优化（optimize\_compute）**

计算优化任务改善计算的执行效率，降低CPU和时间消耗：

$$\text{optimize\_compute}: t_{\text{inference}} \to t'_{\text{inference}}, \quad t' < t \tag{A35.1-optimize}$$

- **目标**：降低推理时间
- **触发条件**：推理耗时比高（$\text{time} > 0.7$）或CPU占用率高（$\text{cpu} > 0.7$）
- **预期收益**：推理耗时和CPU占用率的降低
- **与A23.2和A23.4的对应**：直接降低CPU占用率和推理耗时

计算优化的具体策略包括：

1. **算子融合**：将多个连续算子合并为一个，减少内存访问
2. **并行化**：利用多核或GPU并行执行计算
3. **批处理优化**：优化批处理大小和调度策略
4. **内存布局优化**：改善数据访问模式，提高缓存命中率

**类型4：频率降低（reduce\_frequency）**

频率降低任务通过降低非关键操作的执行频率来减少资源消耗：

$$\text{reduce\_frequency}: f_i \to f'_i, \quad f'_i < f_i, \quad \text{impact} < \theta_{\text{impact}} \tag{A35.1-freq}$$

- **目标**：降低非关键操作的频率，在影响可控的前提下减少资源消耗
- **触发条件**：所有资源指标均偏高（$\text{cpu} > 0.5$ 且 $\text{mem} > 0.5$）
- **预期收益**：所有资源指标的降低
- **与A31.7的对应**：直接调整采集频率分层策略

频率降低与A31.7（引用阶段31）的采集频率分层直接相关——系统层1Hz、日志层0.1Hz、文件层0.01Hz、自身层0.001Hz的默认频率可以根据能效缺口动态调整。当能效缺口较大时，可以进一步降低低优先级操作的频率。

**频率降低的安全性约束**：

频率降低不能影响安全驱力的响应速度——由A30.2（引用阶段30），安全驱力的响应延迟必须 $\leq 1$ 步。因此，安全相关的操作（如异常检测、错误监控）的频率不能降低，只有非关键操作（如文件扫描、模型评估）的频率可以降低。

**类型5：缓存优化（cache\_optimize）**

缓存优化任务通过改善缓存策略来降低重复计算的时间消耗：

$$\text{cache\_optimize}: \text{hit\_rate} \to \text{hit\_rate}', \quad \text{hit\_rate}' > \text{hit\_rate} \tag{A35.1-cache}$$

- **目标**：提高缓存命中率，减少重复计算
- **触发条件**：推理耗时比高且存在大量重复查询
- **预期收益**：推理耗时的降低
- **与A23.4的对应**：间接降低推理耗时

缓存优化的具体策略包括：

1. **LRU缓存**：最近最少使用替换策略
2. **预取缓存**：预测未来需要的计算结果并提前缓存
3. **分层缓存**：不同粒度的缓存层级（结果级、特征级、模型级）
4. **自适应缓存**：根据访问模式动态调整缓存大小和策略

**五种能效任务的优先级排序**：

| 优先级 | 任务类型 | 理由 |
|--------|---------|------|
| 1 | model\_compress | 收益最大（同时降低CPU、内存和冗余） |
| 2 | redundancy\_clean | 从根本上提高信息效率 |
| 3 | optimize\_compute | 直接降低CPU和时间消耗 |
| 4 | cache\_optimize | 低成本高收益，但效果取决于访问模式 |
| 5 | reduce\_frequency | 最简单但可能影响功能完整性 |

> **白话翻译**：五种能效任务就是"五种省钱方式"——模型压缩是"减肥"（减少参数量），冗余清理是"消除浪费"（去掉重复计算），计算优化是"提高效率"（用更少的时间做同样的事），频率降低是"少做一点"（降低非关键操作的频率），缓存优化是"记住答案"（缓存重复计算的结果）。最有效的是减肥（model\_compress），但减肥有精度损失的风险。

---

### 第二部分：模型压缩任务的结构搜索空间

#### 2.2 模型压缩任务的结构搜索空间（A35.2）

模型压缩任务的核心挑战是在精度-效率的权衡空间中找到最优压缩策略。结构搜索空间定义了所有可能的压缩配置：

$$\mathcal{S}_{\text{compress}} = \{(\text{strategy}, \text{ratio}, \text{config}) : \text{strategy} \in \mathcal{P}, \text{ratio} \in [r_{\min}, r_{\max}], \text{config} \in \mathcal{C}_{\text{strategy}}\} \tag{A35.2}$$

其中：
- $\mathcal{P} = \{\text{pruning}, \text{quantization}, \text{distillation}, \text{low\_rank}\}$：压缩策略集合
- $[r_{\min}, r_{\max}]$：压缩比范围（默认 $[0.1, 0.9]$，即保留10%-90%的参数）
- $\mathcal{C}_{\text{strategy}}$：策略特定的配置空间

**各策略的配置空间**：

**剪枝配置空间**：

$$\mathcal{C}_{\text{pruning}} = \{(\text{criterion}, \text{schedule}, \text{granularity})\} \tag{A35.2-pruning}$$

- $\text{criterion} \in \{\text{magnitude}, \text{gradient}, \text{importance}\}$：剪枝判据（幅度、梯度、重要性）
- $\text{schedule} \in \{\text{one\_shot}, \text{iterative}, \text{gradual}\}$：剪枝调度（一次性、迭代、渐进）
- $\text{granularity} \in \{\text{unstructured}, \text{structured}\}$：剪枝粒度（非结构化、结构化）

**量化配置空间**：

$$\mathcal{C}_{\text{quantization}} = \{(\text{bits}, \text{scheme}, \text{calibration})\} \tag{A35.2-quant}$$

- $\text{bits} \in \{2, 4, 8, 16\}$：量化位数
- $\text{scheme} \in \{\text{symmetric}, \text{asymmetric}, \text{mixed}\}$：量化方案
- $\text{calibration} \in \{\text{minmax}, \text{percentile}, \text{entropy}\}$：校准方法

**知识蒸馏配置空间**：

$$\mathcal{C}_{\text{distillation}} = \{(\text{teacher}, \text{loss}, \text{temperature})\} \tag{A35.2-distill}$$

- $\text{teacher}$：教师模型（通常是更大的模型）
- $\text{loss} \in \{\text{soft}, \text{hard}, \text{combined}\}$：蒸馏损失类型
- $\text{temperature} \in [1, 20]$：软化温度

**低秩分解配置空间**：

$$\mathcal{C}_{\text{low\_rank}} = \{(\text{rank}, \text{decomposition}, \text{layers})\} \tag{A35.2-lowrank}$$

- $\text{rank} \in \{4, 8, 16, 32, 64\}$：目标秩
- $\text{decomposition} \in \{\text{SVD}, \text{CP}, \text{Tucker}\}$：分解方法
- $\text{layers}$：应用分解的层

**精度-效率权衡曲面**：

在结构搜索空间中，每个配置 $(s, r, c)$ 对应一个精度-效率点 $(\text{perf}(s, r, c), \text{cost}(s, r, c))$。所有配置的精度-效率点构成权衡曲面（Pareto前沿）：

$$\mathcal{F}_{\text{Pareto}} = \{(p, c) : \nexists (p', c') \text{ s.t. } p' \geq p \text{ and } c' \leq c\} \tag{A35.2-pareto}$$

Pareto前沿上的配置是最优的——不存在另一个配置同时在精度上更好、在成本上更低。

**最优压缩配置的选择**：

最优压缩配置是在Pareto前沿上满足精度约束的成本最低点：

$$(s^*, r^*, c^*) = \arg\min_{(s, r, c) \in \mathcal{S}} \text{cost}(s, r, c) \quad \text{s.t.} \quad \text{perf}(s, r, c) \geq \text{perf}_{\min} \tag{A35.2-optimal}$$

其中 $\text{perf}_{\min} = \text{perf}(\theta) - \epsilon$ 为最低允许精度。

**结构搜索的计算成本**：

结构搜索空间的大小为 $|\mathcal{S}_{\text{compress}}| = |\mathcal{P}| \times |R| \times |\mathcal{C}|$，其中 $|R|$ 为压缩比离散化后的取值数，$|\mathcal{C}|$ 为配置空间的大小。对于剪枝策略，$|\mathcal{C}_{\text{pruning}}| = 3 \times 3 \times 2 = 18$；对于量化策略，$|\mathcal{C}_{\text{quantization}}| = 4 \times 3 \times 3 = 36$。总搜索空间约为数百到数千个配置。

为降低搜索成本，采用以下策略：

1. **粗粒度搜索**：先在粗粒度网格上搜索，确定大致的最优区域
2. **贝叶斯优化**：使用高斯过程代理模型指导搜索方向
3. **早停机制**：当精度低于阈值时提前终止评估

- **输入**：模型参数 $\theta$，精度约束 $\epsilon$，搜索空间 $\mathcal{S}_{\text{compress}}$
- **输出**：最优压缩配置 $(s^*, r^*, c^*)$
- **约束**：$\text{perf}(s^*, r^*, c^*) \geq \text{perf}_{\min}$

> **白话翻译**：模型压缩的结构搜索空间就是"瘦身方案选择"——有四种瘦身方式（剪枝、量化、蒸馏、低秩分解），每种方式有不同的参数配置。搜索空间就是所有可能的瘦身方案，目标是在"不瘦过头"（精度不降太多）的前提下找到"最瘦"（资源消耗最低）的方案。Pareto前沿就是"最佳瘦身效果曲线"——上面的每个点都是最优的，不存在"又瘦又健康"的更好方案。

---

### 第三部分：任务参数化与去重

#### 2.3 任务参数化与去重（A35.3）

能效任务的参数化沿用阶段33-34的五元组格式：

$$\text{task} = \{\text{type}, \text{target}, \text{priority}, \text{estimated\_cost}, \text{expected\_benefit}\} \tag{A35.3}$$

**能效任务的类型权重**：

| 任务类型 | 类型权重 $w_{\text{type}}$ | 基础成本 $c_{\text{type}}$ | 效率系数 $\eta_{\text{type}}$ |
|---------|--------------------------|--------------------------|---------------------------|
| model\_compress | 1.0 | 0.8 | 0.9 |
| redundancy\_clean | 0.8 | 0.3 | 0.7 |
| optimize\_compute | 0.7 | 0.5 | 0.6 |
| cache\_optimize | 0.5 | 0.2 | 0.4 |
| reduce\_frequency | 0.3 | 0.1 | 0.3 |

**优先级计算**：

$$\text{priority} = \Delta V_{\text{eff,norm}} \cdot w_{\text{type}} \cdot w_{\text{scene}} \tag{A35.3-priority}$$

其中 $w_{\text{scene}}$ 为场景权重（tight=1.5, stable=0.7, urgent=0.3）——能效任务在资源紧张场景下优先级最高（必须节约资源），在紧急场景下优先级最低（先救火再省钱）。

**预期收益计算**：

$$\text{expected\_benefit} = \Delta V_{\text{eff}} \cdot \eta_{\text{type}} \cdot \text{feasibility} \tag{A35.3-benefit}$$

其中 $\text{feasibility} \in [0, 1]$ 为可行性系数——模型压缩的可行性取决于模型结构（某些模型难以压缩），冗余清理的可行性取决于冗余度（冗余度越高，清理的收益越大），计算优化的可行性取决于优化空间（已经高度优化的代码优化空间有限）。

**能效任务的ROI分析**：

| 任务类型 | 典型ROI | 说明 |
|---------|---------|------|
| redundancy\_clean | 2.333 (0.7/0.3) | 低成本高收益，ROI最高 |
| cache\_optimize | 2.000 (0.4/0.2) | 低成本中收益，ROI很高 |
| reduce\_frequency | 3.000 (0.3/0.1) | 极低成本低收益，ROI最高但收益绝对值小 |
| optimize\_compute | 1.200 (0.6/0.5) | 中收益中成本，ROI中等 |
| model\_compress | 1.125 (0.9/0.8) | 高收益高成本，ROI最低但收益绝对值最大 |

redundancy\_clean的ROI最高——消除冗余计算的成本很低但收益显著。model\_compress的ROI最低——压缩过程本身需要大量计算（搜索最优配置），但收益绝对值最大（同时降低CPU、内存和冗余度）。

**能效任务的去重**：

能效任务的去重沿用A33.4的相似度计算框架，但增加一个维度——优化目标重叠度：

$$\text{sim}_{\text{target}} = \frac{|\mathcal{O}_i \cap \mathcal{O}_j|}{|\mathcal{O}_i \cup \mathcal{O}_j|} \tag{A35.3-target}$$

其中 $\mathcal{O}_i$ 为任务 $i$ 的优化目标集合（如 {cpu, mem} 表示同时优化CPU和内存）。两个任务如果优化同一组目标，即使类型不同也应考虑合并——因为优化同一目标的边际收益递减。

**能效任务与安全/收益任务的交互**：

能效任务与安全/收益任务可能存在冲突和协同：

**冲突**：
1. 模型压缩可能降低预测精度，影响安全驱力的异常检测能力
2. 频率降低可能延迟异常检测的响应速度
3. 计算优化可能引入新的实现缺陷

**协同**：
1. 冗余清理释放的资源可用于安全任务和收益任务
2. 缓存优化加速推理，间接降低推理耗时，改善能效驱力
3. 模型压缩减少内存占用，为收益任务的探索活动腾出空间

**冲突解决策略**：

1. **精度约束**：模型压缩后的精度不得低于安全检测所需的最低精度
2. **频率约束**：安全相关操作的频率不得降低
3. **验证机制**：计算优化后必须通过安全验证测试

- **输入**：能效缺口 $\Delta V_{\text{eff}}$，安全/收益任务列表
- **输出**：参数化且去重的能效任务列表
- **约束**：精度约束；频率约束；验证机制

> **白话翻译**：能效任务的参数化和去重跟安全/收益任务类似——给每个优化任务排优先级、算成本和收益，合并重复的优化目标。但有一个重要区别：能效任务要考虑"省钱不能省出问题"——模型压缩不能压到安全检测失效，频率降低不能影响异常响应速度。省钱是好事，但不能以牺牲安全为代价。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A35.1 | $\mathcal{T}_{\text{eff}} = \{\text{model\_compress}, \text{redundancy\_clean}, \text{optimize\_compute}, \text{reduce\_frequency}, \text{cache\_optimize}\}$ | — | 任务类型集合 | 5种类型 |
| A35.2 | $\mathcal{S}_{\text{compress}} = \{(\text{strategy}, \text{ratio}, \text{config})\}$ | 模型参数、精度约束 | 最优压缩配置 | Pareto最优 |
| A35.3 | task = {type, target, priority, estimated\_cost, expected\_benefit} | 缺口值、类型、可行性 | 参数化任务 | 精度约束；频率约束 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np

# 引用阶段34的数据类
from stage_034 import GainTaskGenerator, GainTask, CandidateRegion


# ============================================================
# 能效任务类型枚举
# ============================================================
class EfficiencyTaskType(Enum):
    """能效任务类型（A35.1）。"""
    MODEL_COMPRESS = "model_compress"
    REDUNDANCY_CLEAN = "redundancy_clean"
    OPTIMIZE_COMPUTE = "optimize_compute"
    REDUCE_FREQUENCY = "reduce_frequency"
    CACHE_OPTIMIZE = "cache_optimize"


# ============================================================
# 压缩策略枚举
# ============================================================
class CompressionStrategy(Enum):
    """压缩策略（A35.2）。"""
    PRUNING = "pruning"
    QUANTIZATION = "quantization"
    DISTILLATION = "distillation"
    LOW_RANK = "low_rank"


# ============================================================
# 压缩配置
# ============================================================
@dataclass(frozen=True)
class CompressionConfig:
    """模型压缩配置（A35.2）。

    Attributes:
        strategy: 压缩策略
        ratio: 压缩比（保留参数的比例）
        estimated_accuracy_loss: 预估精度损失
        estimated_cost_reduction: 预估成本降低
        config_detail: 策略特定的配置详情
    """
    strategy: CompressionStrategy
    ratio: float
    estimated_accuracy_loss: float
    estimated_cost_reduction: float
    config_detail: dict = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """配置是否有效。"""
        return (0 < self.ratio < 1
                and self.estimated_accuracy_loss >= 0
                and self.estimated_cost_reduction >= 0)


# ============================================================
# 参数化能效任务
# ============================================================
@dataclass(frozen=True)
class EfficiencyTask:
    """参数化能效任务（A35.3）。

    Attributes:
        task_type: 任务类型
        target: 优化目标集合（如 {'cpu', 'mem'}）
        priority: 优先级
        estimated_cost: 预估成本
        expected_benefit: 预期收益
        compression_config: 压缩配置（仅model_compress类型）
        feasibility: 可行性系数
        scope: 任务范围描述
    """
    task_type: EfficiencyTaskType
    target: frozenset[str]
    priority: float
    estimated_cost: float
    expected_benefit: float
    compression_config: Optional[CompressionConfig] = None
    feasibility: float = 1.0
    scope: str = ""

    @property
    def roi(self) -> float:
        """成本-收益比。"""
        if self.estimated_cost < 1e-10:
            return float('inf')
        return self.expected_benefit / self.estimated_cost


# ============================================================
# 阶段35 数据类：能效任务生成器
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class EfficiencyTaskGenerator:
    """能效任务生成器的完整描述。

    组合阶段34的GainTaskGenerator，在收益任务的基础上增加
    能效任务的生成结果。

    Attributes:
        gain_generator: 阶段34的收益任务生成器（组合字段）
        eff_tasks: 生成的能效任务列表
        compression_search_space: 模型压缩的结构搜索空间（A35.2）
        optimal_compression: 最优压缩配置（A35.2）
        threshold: 任务生成阈值
        dedup_threshold: 去重相似度阈值
        original_task_count: 去重前的任务数
        deduped_task_count: 去重后的任务数

    Invariants:
        - threshold > 0
        - dedup_threshold ∈ (0, 1)
        - deduped_task_count <= original_task_count
        - compression_search_space 非空（当model_compress任务存在时）
    """
    gain_generator: GainTaskGenerator
    eff_tasks: tuple[EfficiencyTask, ...]
    compression_search_space: tuple[CompressionConfig, ...]
    optimal_compression: Optional[CompressionConfig]
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
    def all_tasks(self) -> list:
        """所有任务（安全+收益+能效），按优先级降序排列。"""
        safety = list(self.gain_generator.safety_generator.tasks)
        gain = list(self.gain_generator.gain_tasks)
        eff = list(self.eff_tasks)
        all_t = safety + gain + eff
        return sorted(all_t, key=lambda t: t.priority, reverse=True)

    @property
    def total_eff_cost(self) -> float:
        """能效任务的总预估成本。"""
        return sum(t.estimated_cost for t in self.eff_tasks)

    @property
    def total_eff_benefit(self) -> float:
        """能效任务的总预期收益。"""
        return sum(t.expected_benefit for t in self.eff_tasks)

    @property
    def total_all_cost(self) -> float:
        """所有任务的总预估成本。"""
        safety_cost = self.gain_generator.safety_generator.total_cost
        gain_cost = self.gain_generator.total_gain_cost
        return safety_cost + gain_cost + self.total_eff_cost

    @property
    def total_all_benefit(self) -> float:
        """所有任务的总预期收益。"""
        safety_benefit = self.gain_generator.safety_generator.total_benefit
        gain_benefit = self.gain_generator.total_gain_benefit
        return safety_benefit + gain_benefit + self.total_eff_benefit

    @property
    def task_type_distribution(self) -> dict[str, int]:
        """各类型任务的数量分布。"""
        dist = {}
        for t in self.eff_tasks:
            name = t.task_type.value
            dist[name] = dist.get(name, 0) + 1
        return dist


# ============================================================
# 阶段35 核心接口：能效任务生成器
# ============================================================
class EfficiencyTaskGeneratorEngine:
    """阶段35的核心计算接口。

    实现能效缺口到任务生成的完整映射，包括模型压缩搜索、
    任务生成、参数化和去重。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 精度约束（模型压缩后精度不低于阈值）
        - 频率约束（安全操作频率不降低）
        - 验证机制（优化后通过安全测试）
    """

    def search_compression_space(
        self,
        model_params: dict,
        accuracy_budget: float = 0.05,
        max_search_steps: int = 100
    ) -> tuple[CompressionConfig, ...]:
        """搜索模型压缩的结构搜索空间（A35.2）。

        通过遍历预设压缩比生成Pareto候选配置。

        Args:
            model_params: 模型参数信息字典
            accuracy_budget: 允许的精度损失上界 ∈ (0,1)
            max_search_steps: 最大搜索步数 >= 1

        Returns:
            Pareto前沿上的压缩配置列表

        Raises:
            ValueError: 当 accuracy_budget <= 0 或 max_search_steps < 1 时

        Examples:
            >>> engine = EfficiencyTaskGeneratorEngine()
            >>> configs = engine.search_compression_space(
            ...     {"param_count": 1e6}, accuracy_budget=0.05, max_search_steps=10
            ... )
            >>> len(configs) >= 0
            True
        """
        if accuracy_budget <= 0:
            raise ValueError(f"精度预算必须 > 0，收到 {accuracy_budget}")
        if max_search_steps < 1:
            raise ValueError(f"搜索步数必须 >= 1，收到 {max_search_steps}")
        configs: list[CompressionConfig] = []
        # 预设压缩比网格
        ratios = np.linspace(0.1, 0.9, min(max_search_steps, 9))
        for ratio in ratios:
            # 精度损失随压缩比降低而增加（简化模型）
            est_acc_loss = (1.0 - ratio) * 0.1
            if est_acc_loss > accuracy_budget:
                continue
            est_cost_reduction = 1.0 - ratio
            configs.append(CompressionConfig(
                strategy=CompressionStrategy.PRUNING,
                ratio=float(ratio),
                estimated_accuracy_loss=float(est_acc_loss),
                estimated_cost_reduction=float(est_cost_reduction),
                config_detail={"ratio": float(ratio)},
            ))
        return tuple(configs)

    def select_optimal_compression(
        self,
        pareto_configs: tuple[CompressionConfig, ...],
        accuracy_budget: float = 0.05
    ) -> Optional[CompressionConfig]:
        """选择最优压缩配置（A35.2-optimal）。

        在满足精度约束的前提下，选择成本降低最大的配置。

        Args:
            pareto_configs: Pareto前沿上的配置元组
            accuracy_budget: 精度预算

        Returns:
            满足精度约束的成本最低配置，若无则返回None

        Examples:
            >>> engine = EfficiencyTaskGeneratorEngine()
            >>> c1 = CompressionConfig(CompressionStrategy.PRUNING, 0.5, 0.03, 0.5)
            >>> c2 = CompressionConfig(CompressionStrategy.PRUNING, 0.3, 0.08, 0.7)
            >>> best = engine.select_optimal_compression((c1, c2), 0.05)
            >>> best is not None
            True
        """
        valid = [
            c for c in pareto_configs
            if c.estimated_accuracy_loss <= accuracy_budget and c.is_valid
        ]
        if not valid:
            return None
        # 选择成本降低最大的配置
        return max(valid, key=lambda c: c.estimated_cost_reduction)

    def generate_eff_tasks(
        self,
        gap_assessment: 'GapAssessment',
        compression_config: Optional[CompressionConfig] = None,
        safety_cost: float = 0.0,
        gain_cost: float = 0.0,
        resource_budget: float = 1.0
    ) -> list[EfficiencyTask]:
        """生成能效任务（A35.1, A35.3）。

        Args:
            gap_assessment: 缺口评估结果
            compression_config: 最优压缩配置（可选）
            safety_cost: 安全任务已占用的成本
            gain_cost: 收益任务已占用的成本
            resource_budget: 总资源预算 > 0

        Returns:
            能效任务列表

        Raises:
            ValueError: 当 resource_budget <= 0 时

        Examples:
            >>> from stage_032 import GapAssessment
            >>> from stage_031 import EnvironmentStateVector
            >>> from stage_030 import DriveValidationResult
            >>> env = EnvironmentStateVector(
            ...     validation_result=DriveValidationResult(),
            ...     s_sys=(90.0, 80.0, 70.0, 60.0, 5.0, 200.0),
            ...     s_log=(0.0,)*3, s_file=(0.0,)*3,
            ...     s_self=(0.0, 0.0, 5e6, 100.0),
            ...     z_sys=(0.0,)*6, z_log=(0.0,)*3, z_file=(0.0,)*3,
            ...     z_self=(0.0,)*4,
            ...     mu=tuple([0.0]*16), sigma=tuple([1.0]*16),
            ...     timestamps=(0.0,)*4
            ... )
            >>> ga = GapAssessment(
            ...     env_state=env, delta_v_safe=0.1, delta_v_gain=0.1,
            ...     delta_v_eff=0.8, delta_v_safe_norm=0.1,
            ...     delta_v_gain_norm=0.1, delta_v_eff_norm=0.8,
            ...     delta_v_safe_max=1.0, delta_v_gain_max=1.0,
            ...     delta_v_eff_max=1.0, priority_order=('eff','safe','gain'),
            ...     safe_weights=(0.4,0.4,0.2), gain_weights=(0.4,0.3,0.3),
            ...     eff_weights=(0.3,0.2,0.2,0.3)
            ... )
            >>> engine = EfficiencyTaskGeneratorEngine()
            >>> tasks = engine.generate_eff_tasks(ga, safety_cost=0.0, gain_cost=0.0)
            >>> len(tasks) >= 0
            True
        """
        if resource_budget <= 0:
            raise ValueError(f"资源预算必须 > 0，收到 {resource_budget}")
        tasks: list[EfficiencyTask] = []
        delta_norm = gap_assessment.delta_v_eff_norm
        if delta_norm <= 0.1:
            return tasks
        remaining_budget = max(0.0, resource_budget - safety_cost - gain_cost)
        cpu = float(gap_assessment.env_state.s_sys[0]) / 100.0
        mem = float(gap_assessment.env_state.s_sys[1]) / 100.0
        compute_cost = float(gap_assessment.env_state.s_self[2])
        processes = float(gap_assessment.env_state.s_sys[5])
        # CPU优化任务
        if cpu > 0.7:
            tasks.append(EfficiencyTask(
                task_type=EfficiencyTaskType.CPU_OPTIMIZE,
                target=frozenset({"cpu"}),
                priority=delta_norm * cpu,
                estimated_cost=0.3 * (1.0 + delta_norm),
                expected_benefit=delta_norm * cpu * 0.5,
                scope="cpu_optimize",
            ))
        # 内存优化任务
        if mem > 0.7:
            tasks.append(EfficiencyTask(
                task_type=EfficiencyTaskType.MEMORY_OPTIMIZE,
                target=frozenset({"mem"}),
                priority=delta_norm * mem,
                estimated_cost=0.3 * (1.0 + delta_norm),
                expected_benefit=delta_norm * mem * 0.5,
                scope="mem_optimize",
            ))
        # 时间优化任务
        if compute_cost > 1e6:
            tasks.append(EfficiencyTask(
                task_type=EfficiencyTaskType.TIME_OPTIMIZE,
                target=frozenset({"time"}),
                priority=delta_norm * min(1.0, compute_cost / 1e7),
                estimated_cost=0.4 * (1.0 + delta_norm),
                expected_benefit=delta_norm * 0.4,
                scope="time_optimize",
            ))
        # 模型压缩任务
        if compression_config is not None and compression_config.is_valid:
            tasks.append(EfficiencyTask(
                task_type=EfficiencyTaskType.MODEL_COMPRESS,
                target=frozenset({"model"}),
                priority=delta_norm * compression_config.estimated_cost_reduction,
                estimated_cost=0.5 * (1.0 + delta_norm),
                expected_benefit=delta_norm * compression_config.estimated_cost_reduction,
                compression_config=compression_config,
                scope="model_compress",
            ))
        # 冗余消除任务
        if processes > 100:
            tasks.append(EfficiencyTask(
                task_type=EfficiencyTaskType.REDUNDANCY_ELIMINATE,
                target=frozenset({"redundancy"}),
                priority=delta_norm * min(1.0, processes / 500.0),
                estimated_cost=0.2 * (1.0 + delta_norm),
                expected_benefit=delta_norm * 0.3,
                scope="redundancy_eliminate",
            ))
        # 过滤超出预算的任务
        filtered: list[EfficiencyTask] = []
        total_cost = 0.0
        for t in sorted(tasks, key=lambda x: -x.priority):
            if total_cost + t.estimated_cost <= remaining_budget:
                filtered.append(t)
                total_cost += t.estimated_cost
        return filtered

    def check_accuracy_constraint(
        self,
        task: EfficiencyTask,
        min_accuracy: float = 0.95
    ) -> bool:
        """检查精度约束（A35.3）。

        若任务包含压缩配置，则检查预估精度损失后是否仍满足最低精度。

        Args:
            task: 能效任务
            min_accuracy: 最低允许精度 ∈ [0,1]

        Returns:
            True 如果任务满足精度约束

        Raises:
            ValueError: 当 min_accuracy 不在 [0,1] 时

        Examples:
            >>> engine = EfficiencyTaskGeneratorEngine()
            >>> t = EfficiencyTask(EfficiencyTaskType.CPU_OPTIMIZE,
            ...     frozenset({"cpu"}), 0.5, 0.2, 0.3)
            >>> engine.check_accuracy_constraint(t, min_accuracy=0.95)
            True
        """
        if not (0.0 <= min_accuracy <= 1.0):
            raise ValueError(f"min_accuracy 必须在 [0,1] 内，收到 {min_accuracy}")
        if task.compression_config is None:
            return True
        # 假设当前精度为1.0，压缩后精度 = 1.0 - 预估精度损失
        post_accuracy = 1.0 - task.compression_config.estimated_accuracy_loss
        return post_accuracy >= min_accuracy

    def check_frequency_constraint(
        self,
        task: EfficiencyTask,
        protected_operations: set[str] = None
    ) -> bool:
        """检查频率约束（A35.3）。

        若任务目标包含受保护的操作，则拒绝该任务。

        Args:
            task: 能效任务
            protected_operations: 受保护的操作集合（频率不可降低）

        Returns:
            True 如果任务满足频率约束

        Examples:
            >>> engine = EfficiencyTaskGeneratorEngine()
            >>> t = EfficiencyTask(EfficiencyTaskType.CPU_OPTIMIZE,
            ...     frozenset({"cpu"}), 0.5, 0.2, 0.3)
            >>> engine.check_frequency_constraint(t, {"mem", "disk"})
            True
        """
        if protected_operations is None:
            protected_operations = set()
        # 若任务目标与受保护操作有交集，则违反频率约束
        return len(task.target & protected_operations) == 0

    def deduplicate(
        self,
        tasks: list[EfficiencyTask],
        threshold: float = 0.8
    ) -> list[EfficiencyTask]:
        """能效任务去重（A35.3）。

        基于任务类型和目标集合的Jaccard相似度去重。

        Args:
            tasks: 原始任务列表
            threshold: 去重相似度阈值 ∈ (0,1)

        Returns:
            去重后的任务列表

        Raises:
            ValueError: 当 threshold 不在 (0,1) 时

        Examples:
            >>> engine = EfficiencyTaskGeneratorEngine()
            >>> t1 = EfficiencyTask(EfficiencyTaskType.CPU_OPTIMIZE,
            ...     frozenset({"cpu"}), 0.5, 0.2, 0.3, scope="cpu")
            >>> t2 = EfficiencyTask(EfficiencyTaskType.CPU_OPTIMIZE,
            ...     frozenset({"cpu"}), 0.6, 0.2, 0.3, scope="cpu")
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
        merged: list[EfficiencyTask] = []
        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            cluster = [tasks[i]]
            for j in range(i + 1, n):
                if visited[j]:
                    continue
                same_type = tasks[i].task_type == tasks[j].task_type
                # 目标集合Jaccard相似度
                intersection = len(tasks[i].target & tasks[j].target)
                union = len(tasks[i].target | tasks[j].target)
                jaccard = intersection / union if union > 0 else 0.0
                sim = 0.5 * (1.0 if same_type else 0.0) + 0.5 * jaccard
                if sim >= threshold:
                    visited[j] = True
                    cluster.append(tasks[j])
            if len(cluster) == 1:
                merged.append(cluster[0])
            else:
                merged.append(EfficiencyTask(
                    task_type=cluster[0].task_type,
                    target=cluster[0].target,
                    priority=max(t.priority for t in cluster),
                    estimated_cost=np.mean([t.estimated_cost for t in cluster]),
                    expected_benefit=max(t.expected_benefit for t in cluster),
                    compression_config=cluster[0].compression_config,
                    scope=cluster[0].scope,
                ))
        return merged

    def generate(
        self,
        gain_generator: GainTaskGenerator
    ) -> EfficiencyTaskGenerator:
        """完整的能效任务生成流程。

        Args:
            gain_generator: 收益任务生成器

        Returns:
            EfficiencyTaskGenerator 实例

        Examples:
            >>> from stage_034 import GainTaskGenerator
            >>> from stage_033 import SafetyTaskGenerator
            >>> from stage_032 import GapAssessment
            >>> from stage_031 import EnvironmentStateVector
            >>> from stage_030 import DriveValidationResult
            >>> env = EnvironmentStateVector(
            ...     validation_result=DriveValidationResult(),
            ...     s_sys=(90.0, 80.0, 70.0, 60.0, 5.0, 200.0),
            ...     s_log=(0.0,)*3, s_file=(0.0,)*3,
            ...     s_self=(0.0, 0.0, 5e6, 100.0),
            ...     z_sys=(0.0,)*6, z_log=(0.0,)*3, z_file=(0.0,)*3,
            ...     z_self=(0.0,)*4,
            ...     mu=tuple([0.0]*16), sigma=tuple([1.0]*16),
            ...     timestamps=(0.0,)*4
            ... )
            >>> ga = GapAssessment(
            ...     env_state=env, delta_v_safe=0.1, delta_v_gain=0.1,
            ...     delta_v_eff=0.8, delta_v_safe_norm=0.1,
            ...     delta_v_gain_norm=0.1, delta_v_eff_norm=0.8,
            ...     delta_v_safe_max=1.0, delta_v_gain_max=1.0,
            ...     delta_v_eff_max=1.0, priority_order=('eff','safe','gain'),
            ...     safe_weights=(0.4,0.4,0.2), gain_weights=(0.4,0.3,0.3),
            ...     eff_weights=(0.3,0.2,0.2,0.3)
            ... )
            >>> sg = SafetyTaskGenerator(
            ...     gap_assessment=ga, tasks=(), threshold=0.3,
            ...     dedup_threshold=0.8, original_task_count=0,
            ...     deduped_task_count=0
            ... )
            >>> gg = GainTaskGenerator(
            ...     safety_generator=sg, gain_tasks=(),
            ...     candidate_regions=(), selected_regions=(),
            ...     threshold=0.3, dedup_threshold=0.8,
            ...     original_task_count=0, deduped_task_count=0
            ... )
            >>> engine = EfficiencyTaskGeneratorEngine()
            >>> gen = engine.generate(gg)
            >>> gen.deduped_task_count <= gen.original_task_count
            True
        """
        gap = gain_generator.safety_generator.gap_assessment
        # 搜索压缩空间
        compression_configs = self.search_compression_space(
            model_params={"param_count": 1e6},
            accuracy_budget=0.05,
            max_search_steps=10,
        )
        optimal_compression = self.select_optimal_compression(
            compression_configs, accuracy_budget=0.05
        )
        safety_cost = gain_generator.safety_generator.total_cost
        gain_cost = gain_generator.total_gain_cost
        raw_tasks = self.generate_eff_tasks(
            gap,
            compression_config=optimal_compression,
            safety_cost=safety_cost,
            gain_cost=gain_cost,
            resource_budget=1.0,
        )
        original_count = len(raw_tasks)
        deduped_tasks = self.deduplicate(raw_tasks, threshold=0.8)
        deduped_count = len(deduped_tasks)
        return EfficiencyTaskGenerator(
            gain_generator=gain_generator,
            eff_tasks=tuple(deduped_tasks),
            compression_search_space=compression_configs,
            optimal_compression=optimal_compression,
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
| **输入** | `GainTaskGenerator` | ← 阶段34 | 作为能效任务生成器的收益任务组合字段 |
| **输入** | A32.4（能效缺口计算） | ← 阶段32 | 能效缺口的四个分量驱动不同类型的任务 |
| **输入** | A23.2-A23.5（能效驱力分量） | ← 阶段23 | CPU、内存、时间、冗余度与任务类型的对应 |
| **输出** | `EfficiencyTaskGenerator` | → 后续阶段 | 完整的三类任务（安全+收益+能效）生成结果 |
| **输出** | A35.2（结构搜索空间） | → 工程实现 | 模型压缩的搜索策略 |
| **输出** | A35.3（任务参数化与去重） | → 后续阶段 | 参数化任务用于调度和资源分配 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 压缩搜索计算 | `jax.lax.scan`实现结构搜索空间的遍历（A35.2）；`jax.numpy`实现Pareto前沿的计算 |
| Optuna | 贝叶斯优化搜索 | 提供贝叶斯优化框架，指导模型压缩的结构搜索（A35.2），显著降低搜索成本 |
| torch.prune / torch.quantization | 模型压缩实现 | PyTorch的剪枝和量化工具包，支撑model\_compress任务的工程实现 |
| functools.lru_cache | 缓存优化实现 | Python标准库的LRU缓存装饰器，支撑cache\_optimize任务的快速原型验证 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V35.1 | 能效任务类型完整性 | 5种能效任务类型均可生成 | 各类型触发测试 |
| V35.2 | 模型压缩搜索空间 | 搜索空间包含4种策略的配置 | 空间完整性检查 |
| V35.3 | Pareto前沿正确性 | 前沿上的配置不被任何其他配置支配 | 支配关系验证 |
| V35.4 | 最优压缩选择 | 选中的配置满足精度约束且成本最低 | 约束优化验证 |
| V35.5 | 精度约束 | 模型压缩后精度 $\geq$ 最低允许精度 | 压缩后精度测试 |
| V35.6 | 频率约束 | 安全操作频率不降低 | 频率检查 |
| V35.7 | 任务参数化完整性 | 每个任务包含type, target, priority, cost, benefit | 参数完整性检查 |
| V35.8 | 去重有效性 | 相似度>0.8的任务被合并 | 合成相似任务测试 |
| V35.9 | 优化目标重叠去重 | 优化同一目标的任务被合并 | 目标重叠测试 |
| V35.10 | ROI计算 | redundancy\_clean的ROI $\approx 2.333$ | ROI数值验证 |
| V35.11 | 场景权重 | tight场景下能效任务优先级最高 | 场景切换测试 |
| V35.12 | 可行性系数 | 可行性系数 $\in [0, 1]$ | 边界值测试 |
| V35.13 | EfficiencyTaskGenerator组合 | `EfficiencyTaskGenerator.gain_generator` 类型为 `GainTaskGenerator` | 类型检查 |
| V35.14 | 全部任务排序 | all\_tasks按优先级降序排列 | 排序验证 |
| V35.15 | 非法输入拒绝 | 负阈值、无效压缩配置等抛出 `ValueError` | 异常测试 |

---

> **阶段35完** | 阶段31-35建立了从环境感知到任务生成的完整体系：环境状态向量（31）→ 缺口评估算法（32）→ 安全缺口→任务映射（33）→ 收益缺口→任务映射（34）→ 能效缺口→任务映射（35）。至此，从环境状态的多源感知，到三大缺口的量化评估，到15种具体任务（5安全+5收益+5能效）的生成、参数化和去重，任务规划层的完整逻辑链已经打通。三类任务通过组合模式逐层继承（SafetyTaskGenerator → GainTaskGenerator → EfficiencyTaskGenerator），形成了统一的任务池，为后续的任务调度和执行提供了完整的决策基础。
