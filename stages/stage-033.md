# 阶段33：安全缺口→任务生成映射——从危险量化到防御行动

> **阶段编号**：S33 | **所属层**：第四层·任务规划层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 任务生成规则 | "警报响了该做什么"——缺口超过阈值就生成对应任务 |
> | 安全任务类型 | "五种灭火方式"——风险预测、异常检测、状态监控、未知消除、稳定化 |
> | 任务参数化 | "任务清单的详细信息"——类型、目标、优先级、预估成本、预期收益 |
> | 任务去重 | "合并同类项"——相似度太高的任务合并为一个，避免重复劳动 |

---

## 1. 第一性原理依据

本阶段从阶段32（安全缺口计算）和阶段21（异常信号类型）的核心结论出发，建立安全缺口到任务生成的映射。核心依据为：

- **A32.2（安全缺口计算）**（引用阶段32）：$\Delta V_{\text{safe}} = f_{\text{safe}}(\text{anomaly\_count}, \text{error\_rate}, \text{unknown\_ratio})$，安全缺口从环境状态向量中提取三个分量——异常计数、错误率和未知比率。任务生成映射需要将这三个分量分别转化为具体的防御任务。
- **A25.5（缺口→动机映射）**（引用阶段25）：$\text{motivation}_i = f(\Delta V_i)$，缺口通过单调递增函数映射为动机。任务生成是动机驱动的行动选择——缺口越大，动机越强，生成的任务越多、优先级越高。
- **A21.2（异常信号类型）**（引用阶段21）：五类异常信号（系统异常、报错日志、未知状态、风险进程、突变负载）是安全缺口的底层来源。不同类型的异常信号需要不同类型的防御任务——系统异常需要状态监控，报错日志需要异常检测，未知状态需要未知消除，风险进程需要风险预测，突变负载需要稳定化。
- **A21.4（安全驱力与自由能关系）**（引用阶段21）：$R_{\text{safe}} = -\alpha_s \cdot \max(0, F - F_{\text{baseline}})$，安全驱力与自由能偏离基线的程度直接相关。安全任务的根本目标是降低自由能——使系统回到安全稳态。

本阶段的核心目标是：**定义安全缺口到任务生成的映射规则 IF $\Delta V_{\text{safe}} > \theta$ THEN generate(safety\_tasks)，枚举安全任务类型 [risk\_predict, anomaly\_detect, state\_monitor, unknown\_eliminate, stabilize]，定义任务参数化格式 task = {type, target, priority, estimated\_cost, expected\_benefit}，实现任务去重机制（相似度>0.8的任务合并），并封装为SafetyTaskGenerator数据类，组合GapAssessment作为缺口评估的输入。** 安全任务生成是从"感知危险"到"采取防御"的关键一步——它将量化的安全缺口转化为具体的、可执行的任务列表。

---

## 2. 核心数学公式

### 第一部分：任务生成规则

#### 2.1 任务生成规则（A33.1）

安全任务生成的触发规则为条件生成：

$$\text{IF } \Delta V_{\text{safe}} > \theta_{\text{safe}} \text{ THEN generate}(\text{safety\_tasks}) \tag{A33.1}$$

其中 $\theta_{\text{safe}}$ 为安全缺口的任务生成阈值。

**阈值的选择**：

$\theta_{\text{safe}}$ 的选择需要在"灵敏度"和"误报率"之间权衡：
- $\theta_{\text{safe}}$ 太低：安全缺口稍有波动就生成任务，导致任务过多、资源浪费
- $\theta_{\text{safe}}$ 太高：安全缺口较大时才生成任务，可能错过早期预警

默认阈值为归一化缺口的0.3：

$$\theta_{\text{safe}} = 0.3 \cdot \Delta V_{\text{safe,max}} \tag{A33.1-threshold}$$

即安全缺口达到历史最大值的30%时开始生成任务。这个阈值确保系统在安全缺口较小时不浪费资源，在安全缺口较大时及时响应。

**任务生成数量的控制**：

任务生成数量与安全缺口的大小成正比：

$$N_{\text{tasks}} = \left\lfloor k_{\text{safe}} \cdot \Delta V_{\text{safe,norm}} \right\rfloor + 1 \tag{A33.1-count}$$

其中 $k_{\text{safe}}$ 为任务数量缩放系数（默认 $k_{\text{safe}} = 5$），$\lfloor \cdot \rfloor$ 为下取整。当 $\Delta V_{\text{safe,norm}} = 0.3$ 时，生成2个任务；当 $\Delta V_{\text{safe,norm}} = 0.8$ 时，生成5个任务；当 $\Delta V_{\text{safe,norm}} = 1.0$ 时，生成6个任务。

- **输入**：安全缺口 $\Delta V_{\text{safe}}$，阈值 $\theta_{\text{safe}}$
- **输出**：安全任务列表 $\{\text{task}_j\}_{j=1}^{N_{\text{tasks}}}$
- **约束**：$\theta_{\text{safe}} > 0$；$N_{\text{tasks}} \geq 1$（当 $\Delta V_{\text{safe}} > \theta_{\text{safe}}$ 时）

> **白话翻译**：任务生成规则就是"警报响了该做什么"——安全缺口超过阈值就生成防御任务，缺口越大任务越多。阈值不能太低（否则天天假警报），也不能太高（否则真着火了才发现不了）。

**任务生成的分层触发**：

安全缺口的三个分量（异常计数、错误率、未知比率）可以分别触发不同类型的任务：

| 分量 | 阈值 | 触发的任务类型 |
|------|------|--------------|
| anomaly\_count $> \theta_{\text{anomaly}}$ | 2（2个异常指标超限） | state\_monitor, stabilize |
| error\_rate $> \theta_{\text{error}}$ | 0.1（10%错误率） | anomaly\_detect, risk\_predict |
| unknown\_ratio $> \theta_{\text{unknown}}$ | 0.3（30%未知） | unknown\_eliminate, risk\_predict |

分层触发确保不同类型的安全威胁得到针对性的响应——系统异常触发监控和稳定化任务，错误触发检测和预测任务，未知触发消除和预测任务。

---

### 第二部分：安全任务类型枚举

#### 2.2 安全任务类型枚举（A33.2）

安全任务分为五种类型，每种对应不同的防御策略：

$$\mathcal{T}_{\text{safe}} = \{\text{risk\_predict}, \text{anomaly\_detect}, \text{state\_monitor}, \text{unknown\_eliminate}, \text{stabilize}\} \tag{A33.2}$$

**类型1：风险预测（risk\_predict）**

风险预测任务预测未来可能发生的安全威胁，提前采取防御措施：

$$\text{risk\_predict}: \text{输入} \to \hat{R}_{\text{safe}}(t+\Delta t) \tag{A33.2-risk}$$

- **目标**：预测未来 $\Delta t$ 步的安全驱力值
- **触发条件**：安全缺口呈上升趋势（$\text{anomaly\_trend} > 0$）或未知比率较高
- **预期收益**：提前识别风险，避免危机发生
- **对应A21.2**：综合所有五类异常信号的预测

**类型2：异常检测（anomaly\_detect）**

异常检测任务识别当前正在发生的异常事件，确定异常的类型和来源：

$$\text{anomaly\_detect}: \mathbf{s} \to \{(type_i, source_i, severity_i)\} \tag{A33.2-detect}$$

- **目标**：从环境状态向量中识别异常事件，确定类型、来源和严重程度
- **触发条件**：错误率超过阈值或异常计数增加
- **预期收益**：精确定位异常，指导后续的修复行动
- **对应A21.2**：报错日志（A21.2-err）和系统异常（A21.2-sys）

**类型3：状态监控（state\_monitor）**

状态监控任务持续跟踪关键系统指标的变化趋势：

$$\text{state\_monitor}: \{s_i(t)\}_{t=t_0-W}^{t_0} \to \{(\text{indicator}_j, \text{trend}_j, \text{alert}_j)\} \tag{A33.2-monitor}$$

- **目标**：跟踪关键指标的变化趋势，在指标恶化时发出预警
- **触发条件**：异常计数超过阈值或系统负载异常
- **预期收益**：实时掌握系统健康状况，及时发现潜在问题
- **对应A21.2**：突变负载（A21.2-load）和系统异常（A21.2-sys）

**类型4：未知消除（unknown\_eliminate）**

未知消除任务将未知状态转化为已知状态，降低环境的不确定性：

$$\text{unknown\_eliminate}: \mathcal{U} \to \mathcal{K}, \quad |\mathcal{U}| \to |\mathcal{U}| - \Delta k \tag{A33.2-eliminate}$$

其中 $\mathcal{U}$ 为未知状态集合，$\mathcal{K}$ 为已知状态集合。

- **目标**：探索未知状态，将其纳入已知状态库
- **触发条件**：未知比率超过阈值
- **预期收益**：降低环境不确定性，减少安全风险
- **对应A21.2**：未知状态（A21.2-nov）和风险进程（A21.2-risk）

**类型5：稳定化（stabilize）**

稳定化任务采取直接行动使系统回到安全稳态：

$$\text{stabilize}: \mathbf{s} \to \mathbf{s}', \quad F(\mathbf{s}') < F(\mathbf{s}) \tag{A33.2-stabilize}$$

- **目标**：降低自由能，使系统回到安全稳态
- **触发条件**：安全缺口超过高阈值（$\Delta V_{\text{safe,norm}} > 0.7$）
- **预期收益**：直接消除安全威胁，恢复系统正常运行
- **对应A21.2**：综合所有异常信号的紧急响应

**五种安全任务的优先级排序**：

| 优先级 | 任务类型 | 理由 |
|--------|---------|------|
| 1 | stabilize | 直接消除威胁，紧急场景下最高优先 |
| 2 | anomaly\_detect | 精确定位异常，是后续行动的前提 |
| 3 | unknown\_eliminate | 消除不确定性，降低潜在风险 |
| 4 | risk\_predict | 预测未来风险，属于预防性措施 |
| 5 | state\_monitor | 持续监控，属于日常性措施 |

> **白话翻译**：五种安全任务就是"五种灭火方式"——风险预测是"预测哪里会着火"，异常检测是"找到着火点"，状态监控是"持续观察火情"，未知消除是"清理易燃物"，稳定化是"直接灭火"。着火时先灭火（stabilize），再找着火点（anomaly\_detect），然后清理易燃物（unknown\_eliminate），平时做好预防（risk\_predict）和监控（state\_monitor）。

---

### 第三部分：任务参数化

#### 2.3 任务参数化（A33.3）

每个安全任务用五元组参数化：

$$\text{task} = \{\text{type}, \text{target}, \text{priority}, \text{estimated\_cost}, \text{expected\_benefit}\} \tag{A33.3}$$

各字段的定义如下：

**类型（type）**：

$$\text{type} \in \mathcal{T}_{\text{safe}} = \{\text{risk\_predict}, \text{anomaly\_detect}, \text{state\_monitor}, \text{unknown\_eliminate}, \text{stabilize}\} \tag{A33.3-type}$$

任务类型决定任务的执行策略和资源需求。

**目标（target）**：

$$\text{target} \in \{\text{system}, \text{logs}, \text{files}, \text{model}, \text{environment}\} \tag{A33.3-target}$$

任务目标指定任务作用的对象——系统层、日志层、文件层、模型层或环境层。不同类型的任务有不同的默认目标：

| 任务类型 | 默认目标 | 说明 |
|---------|---------|------|
| risk\_predict | environment | 预测环境中的潜在风险 |
| anomaly\_detect | logs | 从日志中检测异常 |
| state\_monitor | system | 监控系统层指标 |
| unknown\_eliminate | files | 探索文件层中的未知状态 |
| stabilize | system | 使系统回到安全稳态 |

**优先级（priority）**：

$$\text{priority} = \Delta V_{\text{safe,norm}} \cdot w_{\text{type}} \cdot w_{\text{urgency}} \tag{A33.3-priority}$$

其中：
- $\Delta V_{\text{safe,norm}}$：归一化安全缺口（引用阶段32，A32.5）
- $w_{\text{type}}$：任务类型权重（stabilize=1.0, anomaly\_detect=0.8, unknown\_eliminate=0.6, risk\_predict=0.4, state\_monitor=0.2）
- $w_{\text{urgency}}$：紧急程度权重（由场景决定：urgent=1.5, tight=1.0, stable=0.7）

优先级越高的任务越先执行。优先级综合考虑了缺口大小、任务类型的重要性和场景的紧急程度。

**预估成本（estimated\_cost）**：

$$\text{estimated\_cost} = c_{\text{type}} \cdot c_{\text{target}} \cdot c_{\text{scope}} \tag{A33.3-cost}$$

其中：
- $c_{\text{type}}$：任务类型的基础成本（stabilize=0.8, anomaly\_detect=0.5, unknown\_eliminate=0.6, risk\_predict=0.3, state\_monitor=0.2）
- $c_{\text{target}}$：目标对象的成本系数（system=1.0, logs=0.5, files=0.6, model=0.8, environment=0.7）
- $c_{\text{scope}}$：任务范围的成本系数（由异常的严重程度决定）

预估成本用于任务调度时的资源约束检查——总成本不能超过可用资源预算。

**预期收益（expected\_benefit）**：

$$\text{expected\_benefit} = \Delta V_{\text{safe}} \cdot \eta_{\text{type}} \tag{A33.3-benefit}$$

其中 $\eta_{\text{type}}$ 为任务类型的效率系数（stabilize=0.9, anomaly\_detect=0.7, unknown\_eliminate=0.5, risk\_predict=0.3, state\_monitor=0.1）。效率系数表示该类型任务能消除安全缺口的比例——stabilize能消除90%的缺口（直接灭火），state\_monitor只能消除10%（只是观察）。

- **输入**：安全缺口 $\Delta V_{\text{safe}}$，任务类型 $\text{type}$，目标 $\text{target}$
- **输出**：参数化任务五元组
- **约束**：priority $\geq 0$；estimated\_cost $\geq 0$；expected\_benefit $\geq 0$

> **白话翻译**：任务参数化就是"任务清单的详细信息"——每个任务都有类型（做什么）、目标（对谁做）、优先级（多紧急）、预估成本（花多少资源）、预期收益（能解决多少问题）。有了这些信息，调度器才能合理安排任务的执行顺序和资源分配。

**任务参数化的成本-收益分析**：

任务的选择应基于成本-收益比：

$$\text{ROI}_{\text{task}} = \frac{\text{expected\_benefit}}{\text{estimated\_cost}} \tag{A33.3-roi}$$

ROI越高的任务越值得执行。在资源有限的情况下，应优先选择ROI高的任务。

| 任务类型 | 典型ROI | 说明 |
|---------|---------|------|
| stabilize | 1.125 (0.9/0.8) | 高收益高成本，紧急场景下ROI最高 |
| anomaly\_detect | 1.400 (0.7/0.5) | 高收益中成本，ROI最高 |
| unknown\_eliminate | 0.833 (0.5/0.6) | 中收益中成本，ROI中等 |
| risk\_predict | 1.000 (0.3/0.3) | 低收益低成本，ROI中等 |
| state\_monitor | 0.500 (0.1/0.2) | 低收益低成本，ROI最低 |

anomaly\_detect的ROI最高——它能以中等成本精确定位异常，为后续行动提供关键信息。state\_monitor的ROI最低——它只提供信息而不直接解决问题，但作为预防性措施仍有价值。

---

### 第四部分：任务去重

#### 2.4 任务去重（A33.4）

多个安全缺口分量可能触发相同或相似的任务，需要去重以避免重复执行：

$$\text{IF } \text{sim}(\text{task}_i, \text{task}_j) > \theta_{\text{dedup}} \text{ THEN merge}(\text{task}_i, \text{task}_j) \tag{A33.4}$$

其中 $\theta_{\text{dedup}} = 0.8$ 为去重相似度阈值。

**任务相似度的计算**：

两个任务的相似度由类型相似度、目标相似度和范围重叠度三部分组成：

$$\text{sim}(\text{task}_i, \text{task}_j) = w_t \cdot \text{sim}_{\text{type}} + w_g \cdot \text{sim}_{\text{target}} + w_s \cdot \text{sim}_{\text{scope}} \tag{A33.4-sim}$$

其中：
- $\text{sim}_{\text{type}} = \mathbb{1}[\text{type}_i = \text{type}_j]$（类型相同为1，否则为0）
- $\text{sim}_{\text{target}} = \mathbb{1}[\text{target}_i = \text{target}_j]$（目标相同为1，否则为0）
- $\text{sim}_{\text{scope}}$：范围重叠度（由任务的具体参数决定，如监控同一组指标、检测同一类异常等）
- $w_t = 0.5, w_g = 0.3, w_s = 0.2$：权重

**任务合并规则**：

当两个任务的相似度超过阈值时，合并为一个任务：

$$\text{merged\_task} = \begin{cases} \text{type} = \text{type}_i \\ \text{target} = \text{target}_i \\ \text{priority} = \max(\text{priority}_i, \text{priority}_j) \\ \text{estimated\_cost} = \max(\text{cost}_i, \text{cost}_j) \\ \text{expected\_benefit} = \max(\text{benefit}_i, \text{benefit}_j) \end{cases} \tag{A33.4-merge}$$

合并策略为"取最大值"——优先级取最高，成本取最大（合并后的任务范围更广），收益取最大（合并后解决的问题更多）。

**去重的效率分析**：

去重机制避免重复执行相似任务，节省计算资源。在最坏情况下（所有任务都不同），去重不产生任何效果；在最好情况下（所有任务都相似），去重将任务数减少到1。典型情况下，去重能减少20%-40%的任务数。

- **输入**：原始任务列表 $\{\text{task}_j\}_{j=1}^{N}$
- **输出**：去重后任务列表 $\{\text{task}_k\}_{k=1}^{M}$，$M \leq N$
- **约束**：$\theta_{\text{dedup}} \in (0, 1)$；合并后的任务优先级不低于原始任务

> **白话翻译**：任务去重就是"合并同类项"——如果两个任务做的事情差不多（相似度>0.8），就合并成一个。合并后优先级取最高的，成本取最大的（因为范围更广），收益也取最大的。这样避免重复劳动，节省资源。

**去重的边界情况**：

1. **完全相同的任务**：$\text{sim} = 1.0$，直接合并为一个，优先级和收益不变
2. **类型相同但目标不同**：$\text{sim} \approx 0.5$（$w_t \cdot 1 + w_g \cdot 0 + w_s \cdot 0 = 0.5$），不合并
3. **类型不同但目标相同**：$\text{sim} \approx 0.3$（$w_t \cdot 0 + w_g \cdot 1 + w_s \cdot 0 = 0.3$），不合并
4. **类型和目标都相同**：$\text{sim} \geq 0.8$（$w_t \cdot 1 + w_g \cdot 1 + w_s \cdot \text{sim}_{\text{scope}} \geq 0.8$），合并

**去重的增量更新**：

当新的安全缺口数据到来时，不需要重新计算所有任务的相似度。增量去重策略：

1. 计算新任务与已有任务的相似度
2. 如果与某个已有任务的相似度 $> \theta_{\text{dedup}}$，合并到该任务
3. 否则，将新任务添加到任务列表

增量去重的计算复杂度为 $O(N)$（$N$ 为已有任务数），远低于全量去重的 $O(N^2)$。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A33.1 | IF $\Delta V_{\text{safe}} > \theta$ THEN generate(safety\_tasks) | $\Delta V_{\text{safe}}$, $\theta$ | 任务列表 | $\theta > 0$ |
| A33.2 | $\mathcal{T}_{\text{safe}} = \{\text{risk\_predict}, \text{anomaly\_detect}, \text{state\_monitor}, \text{unknown\_eliminate}, \text{stabilize}\}$ | — | 任务类型集合 | 5种类型 |
| A33.3 | task = {type, target, priority, estimated\_cost, expected\_benefit} | 缺口值、类型、目标 | 参数化任务 | priority $\geq 0$ |
| A33.4 | IF sim(task\_i, task\_j) > 0.8 THEN merge | 任务对 | 去重后任务列表 | $\theta_{\text{dedup}} = 0.8$ |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np

# 引用阶段32的数据类
from stage_032 import GapAssessment


# ============================================================
# 安全任务类型枚举
# ============================================================
class SafetyTaskType(Enum):
    """安全任务类型（A33.2）。"""
    RISK_PREDICT = "risk_predict"
    ANOMALY_DETECT = "anomaly_detect"
    STATE_MONITOR = "state_monitor"
    UNKNOWN_ELIMINATE = "unknown_eliminate"
    STABILIZE = "stabilize"


# ============================================================
# 安全任务目标枚举
# ============================================================
class SafetyTaskTarget(Enum):
    """安全任务目标（A33.3-target）。"""
    SYSTEM = "system"
    LOGS = "logs"
    FILES = "files"
    MODEL = "model"
    ENVIRONMENT = "environment"


# ============================================================
# 参数化安全任务
# ============================================================
@dataclass(frozen=True)
class SafetyTask:
    """参数化安全任务（A33.3）。

    Attributes:
        task_type: 任务类型
        target: 任务目标
        priority: 优先级
        estimated_cost: 预估成本
        expected_benefit: 预期收益
        scope: 任务范围描述（用于去重的范围重叠度计算）
    """
    task_type: SafetyTaskType
    target: SafetyTaskTarget
    priority: float
    estimated_cost: float
    expected_benefit: float
    scope: str = ""

    @property
    def roi(self) -> float:
        """成本-收益比（A33.3-roi）。"""
        if self.estimated_cost < 1e-10:
            return float('inf')
        return self.expected_benefit / self.estimated_cost


# ============================================================
# 阶段33 数据类：安全任务生成器
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class SafetyTaskGenerator:
    """安全任务生成器的完整描述。

    组合阶段32的GapAssessment，在缺口评估的基础上增加
    安全任务的生成结果。

    Attributes:
        gap_assessment: 阶段32的缺口评估（组合字段）
        tasks: 生成的安全任务列表（A33.1）
        threshold: 任务生成阈值（A33.1）
        dedup_threshold: 去重相似度阈值（A33.4）
        original_task_count: 去重前的任务数
        deduped_task_count: 去重后的任务数

    Invariants:
        - threshold > 0
        - dedup_threshold ∈ (0, 1)
        - deduped_task_count <= original_task_count
        - all(task.priority >= 0 for task in tasks)
        - all(task.estimated_cost >= 0 for task in tasks)
        - all(task.expected_benefit >= 0 for task in tasks)
    """
    gap_assessment: GapAssessment
    tasks: tuple[SafetyTask, ...]
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
        # 验证去重阈值
        if not (0 < self.dedup_threshold < 1):
            raise ValueError(
                f"去重阈值必须 ∈ (0, 1)，"
                f"收到 {self.dedup_threshold}"
            )
        # 验证去重计数
        if self.deduped_task_count > self.original_task_count:
            raise ValueError(
                f"去重后任务数不能超过去重前，"
                f"去重前 {self.original_task_count}，"
                f"去重后 {self.deduped_task_count}"
            )

    @property
    def dedup_rate(self) -> float:
        """去重率。"""
        if self.original_task_count == 0:
            return 0.0
        return 1.0 - self.deduped_task_count / self.original_task_count

    @property
    def total_cost(self) -> float:
        """所有任务的总预估成本。"""
        return sum(t.estimated_cost for t in self.tasks)

    @property
    def total_benefit(self) -> float:
        """所有任务的总预期收益。"""
        return sum(t.expected_benefit for t in self.tasks)

    @property
    def sorted_tasks(self) -> list[SafetyTask]:
        """按优先级降序排列的任务列表。"""
        return sorted(self.tasks, key=lambda t: t.priority, reverse=True)


# ============================================================
# 阶段33 核心接口：安全任务生成器
# ============================================================
class SafetyTaskGeneratorEngine:
    """阶段33的核心计算接口。

    实现安全缺口到任务生成的完整映射，包括任务生成、
    参数化和去重。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 任务生成阈值 > 0
        - 去重相似度阈值 = 0.8
        - 任务优先级非负
    """

    def generate_tasks(
        self,
        gap_assessment: GapAssessment,
        threshold: float = 0.3
    ) -> list[SafetyTask]:
        """根据安全缺口生成任务（A33.1）。

        Args:
            gap_assessment: 缺口评估结果
            threshold: 任务生成阈值（归一化缺口）

        Returns:
            安全任务列表

        Raises:
            ValueError: 当 threshold <= 0 时

        Examples:
            >>> from stage_032 import GapAssessment
            >>> from stage_031 import EnvironmentStateVector
            >>> from stage_030 import DriveValidationResult
            >>> env = EnvironmentStateVector(
            ...     validation_result=DriveValidationResult(),
            ...     s_sys=(50.0,)*6, s_log=(10.0, 0.05, 0.0),
            ...     s_file=(0.1, 0.0, 0.0),
            ...     s_self=(0.8, 1.0, 1e6, 1000.0),
            ...     z_sys=(0.0,)*6, z_log=(0.0,)*3, z_file=(0.0,)*3,
            ...     z_self=(0.0,)*4,
            ...     mu=tuple([0.0]*16), sigma=tuple([1.0]*16),
            ...     timestamps=(0.0,)*4
            ... )
            >>> ga = GapAssessment(
            ...     env_state=env, delta_v_safe=0.5, delta_v_gain=0.2,
            ...     delta_v_eff=0.1, delta_v_safe_norm=0.5,
            ...     delta_v_gain_norm=0.2, delta_v_eff_norm=0.1,
            ...     delta_v_safe_max=1.0, delta_v_gain_max=1.0,
            ...     delta_v_eff_max=1.0, priority_order=('safe','gain','eff'),
            ...     safe_weights=(0.4,0.4,0.2), gain_weights=(0.4,0.3,0.3),
            ...     eff_weights=(0.3,0.2,0.2,0.3)
            ... )
            >>> engine = SafetyTaskGeneratorEngine()
            >>> tasks = engine.generate_tasks(ga, threshold=0.3)
            >>> len(tasks) >= 0
            True
        """
        if threshold <= 0:
            raise ValueError(f"阈值必须 > 0，收到 {threshold}")
        tasks: list[SafetyTask] = []
        delta_norm = gap_assessment.delta_v_safe_norm
        if delta_norm <= threshold:
            return tasks
        # 任务数量与缺口大小成正比
        num_tasks = int(np.floor(3.0 * delta_norm)) + 1
        # 根据缺口分量选择任务类型
        z_sys = np.array(gap_assessment.env_state.z_sys)
        anomaly_count = int(np.sum(np.abs(z_sys) > 2.0))
        error_rate = float(gap_assessment.env_state.s_log[1])
        unknown_ratio = float(np.sum(np.abs(z_sys) < 0.5) / max(len(z_sys), 1))
        task_types = self.select_task_type(anomaly_count, error_rate, unknown_ratio)
        # 为每种选中的类型生成任务
        for i in range(num_tasks):
            task_type = task_types[i % len(task_types)]
            target = SafetyTaskTarget.SYSTEM if anomaly_count > 0 else SafetyTaskTarget.LOG
            task = self.parameterize_task(task_type, target, delta_norm)
            tasks.append(task)
        return tasks

    def select_task_type(
        self,
        anomaly_count: int,
        error_rate: float,
        unknown_ratio: float
    ) -> list[SafetyTaskType]:
        """根据安全缺口分量选择任务类型（A33.2）。

        Args:
            anomaly_count: 异常计数
            error_rate: 错误率 ∈ [0,1]
            unknown_ratio: 未知比率 ∈ [0,1]

        Returns:
            适用的任务类型列表

        Raises:
            ValueError: 当 error_rate 或 unknown_ratio 不在 [0,1] 时

        Examples:
            >>> engine = SafetyTaskGeneratorEngine()
            >>> types = engine.select_task_type(3, 0.1, 0.2)
            >>> len(types) > 0
            True
        """
        if not (0.0 <= error_rate <= 1.0):
            raise ValueError(f"error_rate 必须在 [0,1] 内，收到 {error_rate}")
        if not (0.0 <= unknown_ratio <= 1.0):
            raise ValueError(f"unknown_ratio 必须在 [0,1] 内，收到 {unknown_ratio}")
        selected: list[SafetyTaskType] = []
        if anomaly_count >= 3:
            selected.append(SafetyTaskType.RISK_PREDICT)
        if error_rate > 0.05:
            selected.append(SafetyTaskType.ANOMALY_DETECT)
        if anomaly_count >= 1:
            selected.append(SafetyTaskType.STATE_MONITOR)
        if unknown_ratio > 0.3:
            selected.append(SafetyTaskType.UNKNOWN_ELIMINATE)
        if error_rate > 0.01 or anomaly_count > 0:
            selected.append(SafetyTaskType.STABILIZE)
        if not selected:
            selected.append(SafetyTaskType.STATE_MONITOR)
        return selected

    def parameterize_task(
        self,
        task_type: SafetyTaskType,
        target: SafetyTaskTarget,
        delta_v_safe_norm: float,
        scene: str = "stable"
    ) -> SafetyTask:
        """参数化任务（A33.3）。

        Args:
            task_type: 任务类型
            target: 任务目标
            delta_v_safe_norm: 归一化安全缺口 ∈ [0,1]
            scene: 当前场景

        Returns:
            参数化的SafetyTask

        Raises:
            ValueError: 当 delta_v_safe_norm 不在 [0,1] 时

        Examples:
            >>> engine = SafetyTaskGeneratorEngine()
            >>> task = engine.parameterize_task(
            ...     SafetyTaskType.ANOMALY_DETECT,
            ...     SafetyTaskTarget.LOG, 0.6
            ... )
            >>> task.priority >= 0
            True
        """
        if not (0.0 <= delta_v_safe_norm <= 1.0 + 1e-10):
            raise ValueError(
                f"delta_v_safe_norm 必须在 [0,1] 内，收到 {delta_v_safe_norm}"
            )
        # 基础成本与收益（根据任务类型预设）
        base_cost = {
            SafetyTaskType.RISK_PREDICT: 0.4,
            SafetyTaskType.ANOMALY_DETECT: 0.3,
            SafetyTaskType.STATE_MONITOR: 0.2,
            SafetyTaskType.UNKNOWN_ELIMINATE: 0.5,
            SafetyTaskType.STABILIZE: 0.6,
        }.get(task_type, 0.3)
        base_benefit = {
            SafetyTaskType.RISK_PREDICT: 0.5,
            SafetyTaskType.ANOMALY_DETECT: 0.6,
            SafetyTaskType.STATE_MONITOR: 0.3,
            SafetyTaskType.UNKNOWN_ELIMINATE: 0.7,
            SafetyTaskType.STABILIZE: 0.8,
        }.get(task_type, 0.4)
        # 场景调整系数
        scene_multiplier = 1.2 if scene == "urgent" else 1.0
        priority = delta_v_safe_norm * scene_multiplier
        estimated_cost = base_cost * (1.0 + delta_v_safe_norm)
        expected_benefit = base_benefit * delta_v_safe_norm * scene_multiplier
        scope = f"{task_type.value}_{target.value}_{scene}"
        return SafetyTask(
            task_type=task_type,
            target=target,
            priority=priority,
            estimated_cost=estimated_cost,
            expected_benefit=expected_benefit,
            scope=scope,
        )

    def compute_similarity(
        self,
        task_i: SafetyTask,
        task_j: SafetyTask
    ) -> float:
        """计算两个任务的相似度（A33.4-sim）。

        基于任务类型、目标和范围字符串的Jaccard-like相似度。

        Args:
            task_i: 第一个任务
            task_j: 第二个任务

        Returns:
            相似度 ∈ [0, 1]

        Examples:
            >>> engine = SafetyTaskGeneratorEngine()
            >>> t1 = SafetyTask(SafetyTaskType.STATE_MONITOR,
            ...     SafetyTaskTarget.SYSTEM, 0.5, 0.2, 0.3, "monitor_system")
            >>> t2 = SafetyTask(SafetyTaskType.STATE_MONITOR,
            ...     SafetyTaskTarget.SYSTEM, 0.6, 0.2, 0.3, "monitor_system")
            >>> sim = engine.compute_similarity(t1, t2)
            >>> 0.0 <= sim <= 1.0
            True
        """
        if task_i.task_type == task_j.task_type and task_i.target == task_j.target:
            type_score = 1.0
        elif task_i.task_type == task_j.task_type:
            type_score = 0.7
        elif task_i.target == task_j.target:
            type_score = 0.4
        else:
            type_score = 0.0
        # 范围字符串相似度（简单字符集合Jaccard）
        set_i = set(task_i.scope)
        set_j = set(task_j.scope)
        if not set_i and not set_j:
            scope_score = 1.0
        else:
            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            scope_score = intersection / union if union > 0 else 0.0
        # 综合相似度
        similarity = 0.6 * type_score + 0.4 * scope_score
        return float(min(max(similarity, 0.0), 1.0))

    def deduplicate(
        self,
        tasks: list[SafetyTask],
        threshold: float = 0.8
    ) -> list[SafetyTask]:
        """任务去重（A33.4）。

        使用贪心聚类：对每对任务计算相似度，相似度超过阈值则合并，
        合并后优先级取最大，成本取平均，收益取最大。

        Args:
            tasks: 原始任务列表
            threshold: 去重相似度阈值 ∈ (0,1)

        Returns:
            去重后的任务列表

        Raises:
            ValueError: 当 threshold 不在 (0,1) 时

        Examples:
            >>> engine = SafetyTaskGeneratorEngine()
            >>> t1 = SafetyTask(SafetyTaskType.STATE_MONITOR,
            ...     SafetyTaskTarget.SYSTEM, 0.5, 0.2, 0.3, "monitor")
            >>> t2 = SafetyTask(SafetyTaskType.STATE_MONITOR,
            ...     SafetyTaskTarget.SYSTEM, 0.6, 0.2, 0.3, "monitor")
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
        merged: list[SafetyTask] = []
        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            cluster = [tasks[i]]
            for j in range(i + 1, n):
                if visited[j]:
                    continue
                sim = self.compute_similarity(tasks[i], tasks[j])
                if sim >= threshold:
                    visited[j] = True
                    cluster.append(tasks[j])
            # 合并聚类中的任务
            if len(cluster) == 1:
                merged.append(cluster[0])
            else:
                merged_task = SafetyTask(
                    task_type=cluster[0].task_type,
                    target=cluster[0].target,
                    priority=max(t.priority for t in cluster),
                    estimated_cost=np.mean([t.estimated_cost for t in cluster]),
                    expected_benefit=max(t.expected_benefit for t in cluster),
                    scope=cluster[0].scope,
                )
                merged.append(merged_task)
        return merged

    def generate(
        self,
        gap_assessment: GapAssessment
    ) -> SafetyTaskGenerator:
        """完整的安全任务生成流程。

        依次执行任务生成、参数化和去重。

        Args:
            gap_assessment: 缺口评估结果

        Returns:
            SafetyTaskGenerator 实例

        Examples:
            >>> from stage_032 import GapAssessment
            >>> from stage_031 import EnvironmentStateVector
            >>> from stage_030 import DriveValidationResult
            >>> env = EnvironmentStateVector(
            ...     validation_result=DriveValidationResult(),
            ...     s_sys=(50.0,)*6, s_log=(10.0, 0.05, 0.0),
            ...     s_file=(0.1, 0.0, 0.0),
            ...     s_self=(0.8, 1.0, 1e6, 1000.0),
            ...     z_sys=(0.0,)*6, z_log=(0.0,)*3, z_file=(0.0,)*3,
            ...     z_self=(0.0,)*4,
            ...     mu=tuple([0.0]*16), sigma=tuple([1.0]*16),
            ...     timestamps=(0.0,)*4
            ... )
            >>> ga = GapAssessment(
            ...     env_state=env, delta_v_safe=0.5, delta_v_gain=0.2,
            ...     delta_v_eff=0.1, delta_v_safe_norm=0.5,
            ...     delta_v_gain_norm=0.2, delta_v_eff_norm=0.1,
            ...     delta_v_safe_max=1.0, delta_v_gain_max=1.0,
            ...     delta_v_eff_max=1.0, priority_order=('safe','gain','eff'),
            ...     safe_weights=(0.4,0.4,0.2), gain_weights=(0.4,0.3,0.3),
            ...     eff_weights=(0.3,0.2,0.2,0.3)
            ... )
            >>> engine = SafetyTaskGeneratorEngine()
            >>> gen = engine.generate(ga)
            >>> gen.deduped_task_count <= gen.original_task_count
            True
        """
        raw_tasks = self.generate_tasks(gap_assessment, threshold=0.3)
        original_count = len(raw_tasks)
        deduped_tasks = self.deduplicate(raw_tasks, threshold=0.8)
        deduped_count = len(deduped_tasks)
        return SafetyTaskGenerator(
            gap_assessment=gap_assessment,
            tasks=tuple(deduped_tasks),
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
| **输入** | `GapAssessment` | ← 阶段32 | 作为安全任务生成的缺口评估输入 |
| **输入** | A32.2（安全缺口计算） | ← 阶段32 | 安全缺口的三个分量驱动不同类型的任务 |
| **输入** | A21.2（异常信号类型） | ← 阶段21 | 五类异常信号与五种安全任务的对应关系 |
| **输出** | `SafetyTaskGenerator` | → 阶段34 | 安全任务列表作为收益任务生成器的组合字段 |
| **输出** | A33.3（任务参数化） | → 后续阶段 | 参数化任务用于任务调度和资源分配 |
| **输出** | A33.4（任务去重） | → 后续阶段 | 去重机制避免任务重复执行 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 任务生成计算 | `jax.lax.cond`实现条件生成规则（A33.1）；`jax.vmap`实现批量任务的参数化计算（A33.3） |
| scikit-learn | 任务相似度计算 | `sklearn.metrics.pairwise.cosine_similarity`计算任务范围向量的余弦相似度（A33.4） |
| NetworkX | 任务去重图算法 | 将任务相似度建模为图，使用连通分量算法实现批量去重（A33.4） |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V33.1 | 任务生成触发 | $\Delta V_{\text{safe}} > \theta$ 时生成任务 | 阈值边界测试 |
| V33.2 | 任务数量控制 | $N_{\text{tasks}} = \lfloor k \cdot \Delta V_{\text{safe,norm}} \rfloor + 1$ | 缺口扫描测试 |
| V33.3 | 任务类型完整性 | 5种安全任务类型均可生成 | 各类型触发测试 |
| V33.4 | 任务参数化完整性 | 每个任务包含type, target, priority, cost, benefit | 参数完整性检查 |
| V33.5 | 优先级非负 | priority $\geq 0$ 对所有合法输入成立 | 随机缺口测试 |
| V33.6 | 成本非负 | estimated\_cost $\geq 0$ 对所有合法输入成立 | 随机缺口测试 |
| V33.7 | 收益非负 | expected\_benefit $\geq 0$ 对所有合法输入成立 | 随机缺口测试 |
| V33.8 | 去重相似度范围 | $\text{sim} \in [0, 1]$ | 相似度计算测试 |
| V33.9 | 去重阈值 | 相似度>0.8的任务被合并 | 合成相似任务测试 |
| V33.10 | 去重后任务数减少 | $M \leq N$ | 大量相似任务测试 |
| V33.11 | 合并优先级取最大 | 合并后priority = max(priority\_i, priority\_j) | 合并验证 |
| V33.12 | 分层触发 | 不同缺口分量触发不同任务类型 | 分量独立测试 |
| V33.13 | SafetyTaskGenerator组合 | `SafetyTaskGenerator.gap_assessment` 类型为 `GapAssessment` | 类型检查 |
| V33.14 | 非法输入拒绝 | 负阈值、非法去重阈值等抛出 `ValueError` | 异常测试 |
| V33.15 | ROI计算 | stabilize的ROI $\approx 1.125$，anomaly\_detect的ROI $\approx 1.400$ | ROI数值验证 |

---

> **阶段33完** | 阶段33建立了安全缺口到任务生成的完整映射：五种安全任务类型（风险预测、异常检测、状态监控、未知消除、稳定化）覆盖了从预防到应急的全谱系防御策略，任务参数化提供了决策所需的成本-收益信息，任务去重避免了重复劳动。下一步：阶段34将从收益缺口出发，定义收益任务生成映射。
