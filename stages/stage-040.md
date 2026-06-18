# 阶段40：自主决策引擎Python接口——从组件整合到统一决策

> **阶段编号**：S40 | **所属层**：第四层·任务规划层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 决策引擎 | "大脑总指挥"——整合感知、评估、生成、打分、择优、中断六大模块 |
> | decide方法 | "做决定"——给一个环境状态，返回一个要做的任务 |
> | execute方法 | "执行任务"——把选中的任务跑一遍，返回结果 |
> | interrupt_check方法 | "要不要换"——检查是否有更紧急的任务需要切换 |
> | update_pool方法 | "更新任务库"——根据最新环境刷新任务池 |
> | 完整决策循环 | "感知→评估→生成→打分→择优→执行→检查中断→更新"的闭环 |

---

## 1. 第一性原理依据

本阶段从阶段31-39的所有决策相关接口出发，建立自主决策引擎的统一Python接口。核心依据为：

- **A31.1（环境状态向量）**（引用阶段31）：$\mathbf{s} = [\mathbf{s}_{\text{sys}}, \mathbf{s}_{\text{log}}, \mathbf{s}_{\text{file}}, \mathbf{s}_{\text{self}}]$，环境状态向量是决策引擎的输入——所有决策都从感知环境状态开始。
- **A32.1（缺口评估函数）**（引用阶段32）：$\text{gap}(\mathbf{s}) = (\Delta V_{\text{safe}}, \Delta V_{\text{gain}}, \Delta V_{\text{eff}})$，缺口评估将环境状态转化为三大缺口，是决策引擎的评估模块。
- **A33.1-A35.3（三类任务生成器）**（引用阶段33-35）：SafetyTaskGenerator、GainTaskGenerator、EfficiencyTaskGenerator将三大缺口转化为15种具体任务，是决策引擎的生成模块。
- **A36.1-A36.6（动态任务池）**（引用阶段36）：TaskPool = {active, pending, completed, frozen}统一管理所有任务，是决策引擎的存储模块。
- **A37.1-A37.6（综合得分公式）**（引用阶段37）：Score(task) = W·ΔV_gap + α·urgency + β·feasibility - γ·cost，综合得分是决策引擎的评分模块。
- **A38.1-A38.5（70/30双决策机制）**（引用阶段38）：利用择优+探索随机+比例强制，双决策机制是决策引擎的择优模块。
- **A39.1-A39.6（动态中断机制）**（引用阶段39）：中断条件+紧急中断+保存点+恢复+频率限制+成本评估，动态中断是决策引擎的中断模块。

本阶段的核心目标是：**定义DecisionEngine类设计，整合感知（阶段31）+评估（阶段32）+生成（阶段33-35）+打分（阶段37）+择优（阶段38）+中断（阶段39）六大模块，定义主决策方法 decide(state) → Task，定义执行方法 execute(task) → Result，定义中断检查方法 interrupt_check(state, current_task) → Optional[Task]，定义池更新方法 update_pool(state) → TaskPool，提供完整决策循环伪代码，并封装为DecisionEngineState数据类，组合DynamicInterruptor作为中断器的输入。** 自主决策引擎是阶段31-39所有决策组件的统一入口——它将分散的模块整合为单一的决策接口，使Agent的决策过程可调用、可测试、可审计。

---

## 2. 核心数学公式

### 第一部分：DecisionEngine类设计

#### 2.1 DecisionEngine类设计（A40.1）

DecisionEngine整合六大模块，形成完整的决策管线：

$$\text{DecisionEngine} = \text{Perceive} \circ \text{Assess} \circ \text{Generate} \circ \text{Score} \circ \text{Select} \circ \text{Interrupt} \tag{A40.1}$$

其中 $\circ$ 表示模块的组合（composition）——前一模块的输出是后一模块的输入。

**六大模块的映射**：

| 模块 | 阶段 | 输入 | 输出 | 核心公式 |
|------|------|------|------|---------|
| Perceive（感知） | 31 | 原始环境数据 | 环境状态向量 $\mathbf{s}$ | A31.1 |
| Assess（评估） | 32 | 环境状态向量 $\mathbf{s}$ | 缺口三元组 $(\Delta V_{\text{safe}}, \Delta V_{\text{gain}}, \Delta V_{\text{eff}})$ | A32.1 |
| Generate（生成） | 33-35 | 缺口三元组 | 任务池 TaskPool | A33.1-A35.3 |
| Score（打分） | 37 | 任务池 + 缺口 + 资源 | 评分结果 TaskScorer | A37.1 |
| Select（择优） | 38 | 评分结果 | 选中的任务 Task | A38.1-A38.2 |
| Interrupt（中断） | 39 | 当前任务 + 新任务 + 安全驱力 | 中断决策 | A39.1-A39.2 |

**模块间的数据流**：

$$\mathbf{s} \xrightarrow{\text{Assess}} (\Delta V_{\text{safe}}, \Delta V_{\text{gain}}, \Delta V_{\text{eff}}) \xrightarrow{\text{Generate}} \text{TaskPool} \xrightarrow{\text{Score}} \text{TaskScorer} \xrightarrow{\text{Select}} \text{Task} \tag{A40.1-flow}$$

**DecisionEngine的依赖注入**：

DecisionEngine通过组合模式整合各模块，而非继承——每个模块是独立的组件，可以单独替换和测试：

$$\text{DecisionEngine} = \{\text{perceiver}, \text{assessor}, \text{generator}, \text{scorer}, \text{selector}, \text{interruptor}\} \tag{A40.1-di}$$

依赖注入的好处：

1. **可测试性**：每个模块可以独立测试，不需要启动整个引擎
2. **可替换性**：可以替换某个模块的实现而不影响其他模块
3. **可配置性**：可以通过配置不同的模块实例来定制引擎的行为

- **输入**：环境状态向量 $\mathbf{s}$
- **输出**：选中的任务 Task
- **约束**：六大模块按顺序组合；模块间数据类型匹配

> **白话翻译**：DecisionEngine就是"大脑总指挥"——把感知、评估、生成、打分、择优、中断六大模块串起来。环境数据进来，经过六步处理，最后输出一个"该做什么任务"的决策。每个模块都是独立的，可以单独测试和替换。

---

### 第二部分：主决策方法

#### 2.2 主决策方法 decide（A40.2）

主决策方法是DecisionEngine的核心接口——给定环境状态，返回应执行的任务：

$$\text{decide}(\text{state}) \to \text{Task} \tag{A40.2}$$

**完整决策流程**：

```
输入：环境状态 state
输出：选中的任务 task

1. 感知：s = perceiver.perceive(raw_data)
2. 评估：gaps = assessor.assess(s)
3. 生成：pool = generator.generate(gaps)
4. 打分：scorer = scorer_engine.score_pool(pool)
5. 择优：
   a. 比例强制执行器决定本次决策类型
   b. 如果 exploit：task = argmax Score(task)
   c. 如果 explore：task = random_select(low_score_tasks)
6. 中断检查：
   a. 如果有当前正在执行的任务 current_task：
      i. 检查紧急中断条件
      ii. 检查普通中断条件
      iii. 如果需要中断：保存当前任务，返回新任务
   b. 否则：直接返回选中的任务
7. 返回 task
```

**decide方法的不变量**：

1. **非空返回**：只要任务池非空，decide一定返回一个任务
2. **得分最优**：返回的任务在利用模式下得分最高，在探索模式下从低分候选中随机选择
3. **安全优先**：紧急场景下返回安全任务
4. **比例保证**：长期来看，利用/探索比例约为70/30

**decide方法的时间复杂度**：

| 步骤 | 时间复杂度 | 说明 |
|------|-----------|------|
| 感知 | $O(d)$ | $d$ 为环境状态维度（16） |
| 评估 | $O(d)$ | 缺口计算为线性 |
| 生成 | $O(N \log N)$ | $N$ 为生成任务数（15-30） |
| 打分 | $O(N)$ | 每个任务评分常数时间 |
| 择优 | $O(N)$ | argmax或随机选择 |
| 中断检查 | $O(1)$ | 常数时间判定 |

总时间复杂度 $O(N \log N)$，其中 $N \approx 15$-$30$，单次决策耗时可控。

- **输入**：环境状态 state
- **输出**：选中的任务 Task
- **约束**：任务池非空时返回非空；安全优先；比例保证

> **白话翻译**：decide方法就是"做决定"——给一个环境状态，走完六步流程，返回一个该做的任务。先感知环境，再评估缺口，再生成任务，再打分排序，再择优选择，最后检查要不要中断当前任务。整个过程从感知到行动，一步到位。

---

### 第三部分：执行方法

#### 2.3 执行方法 execute（A40.3）

执行方法将选中的任务转化为执行结果：

$$\text{execute}(\text{task}) \to \text{Result} \tag{A40.3}$$

**Result数据结构**：

$$\text{Result} = \{\text{task\_id}, \text{status}, \text{actual\_benefit}, \text{actual\_cost}, \text{progress}, \text{observations}\} \tag{A40.3-result}$$

其中：
- $\text{task\_id}$：任务唯一标识
- $\text{status} \in \{\text{completed}, \text{partial}, \text{failed}\}$：执行状态
- $\text{actual\_benefit}$：实际收益——执行任务后三大缺口的实际变化量
- $\text{actual\_cost}$：实际成本——执行任务实际消耗的资源
- $\text{progress} \in [0, 1]$：执行进度
- $\text{observations}$：执行过程中的观测数据

**执行状态的定义**：

| 状态 | 含义 | 触发条件 |
|------|------|---------|
| completed | 任务完成 | progress = 1.0 |
| partial | 部分完成 | 0 < progress < 1.0（被中断或时间用尽） |
| failed | 执行失败 | 执行过程中发生错误 |

**实际收益的计算**：

$$\text{actual\_benefit} = \Delta V_{\text{before}} - \Delta V_{\text{after}} \tag{A40.3-benefit}$$

其中 $\Delta V_{\text{before}}$ 为执行前的缺口值，$\Delta V_{\text{after}}$ 为执行后的缺口值。实际收益 = 缺口缩小量。

**预期收益与实际收益的比较**：

$$\text{benefit\_ratio} = \frac{\text{actual\_benefit}}{\text{expected\_benefit}} \tag{A40.3-ratio}$$

benefit_ratio用于评估任务执行的效率：

| benefit_ratio | 含义 | 后续动作 |
|--------------|------|---------|
| > 1.0 | 超预期 | 提高同类任务的优先级 |
| 0.5-1.0 | 符合预期 | 维持当前策略 |
| 0.0-0.5 | 低于预期 | 降低同类任务的优先级 |
| < 0.0 | 反效果 | 冻结同类任务，重新评估 |

benefit_ratio也是A36.5冻结条件的输入——当连续5轮 benefit_ratio < 0.5 时，任务被冻结。

- **输入**：选中的任务 task
- **输出**：执行结果 Result
- **约束**：actual_benefit 和 actual_cost 非负（正常情况下）

> **白话翻译**：execute方法就是"执行任务"——把选中的任务跑一遍，返回结果。结果包括：做完了没有、实际收益多少、实际成本多少、做了多少进度、执行中观察到了什么。实际收益和预期收益的比较很重要——如果连续低于预期，说明这类任务不值得做，应该冻结。

---

### 第四部分：中断检查方法

#### 2.4 中断检查方法 interrupt_check（A40.4）

中断检查方法在每次决策循环中检查是否需要中断当前任务：

$$\text{interrupt\_check}(\text{state}, \text{current\_task}) \to \text{Optional[Task]} \tag{A40.4}$$

**返回值语义**：

- 返回 $\text{None}$：不需要中断，继续执行当前任务
- 返回 $\text{Task}$：需要中断，切换到返回的新任务

**中断检查流程**：

```
输入：环境状态 state，当前任务 current_task
输出：Optional[Task]

1. 重新评估当前环境：
   gaps = assessor.assess(state)

2. 检查紧急中断：
   if R_safe < θ_urgent:
     checkpoint = create_checkpoint(current_task)
     new_task = select_safety_task(gaps)
     return new_task  // 无条件中断

3. 检查频率限制：
   if current_step - last_interrupt_step < T_min_exec:
     return None  // 频率限制，不允许中断

4. 重新评分当前任务和新候选：
   current_score = score(current_task, gaps)
   new_tasks = score_pool(pool, gaps)
   best_new_score = max(score for task in new_tasks)

5. 检查普通中断条件：
   if best_new_score > current_score + θ_interrupt:
     // 评估中断成本
     cost = compute_interrupt_cost(current_task, ...)
     net_benefit = (best_new_score - current_score) * T_remaining - cost
     if net_benefit > 0:
       checkpoint = create_checkpoint(current_task)
       new_task = argmax new_tasks
       return new_task

6. return None  // 不需要中断
```

**interrupt_check的调用时机**：

中断检查不是每一步都调用的——调用时机为：

1. **每次decide调用前**：在做出新决策前，先检查是否需要中断当前任务
2. **场景切换时**：当场景分类器的输出变化时（引用A26.1），立即检查中断
3. **安全驱力突变时**：当安全驱力的变化率超过阈值时，立即检查中断

- **输入**：环境状态 state，当前任务 current_task
- **输出**：新任务（如果需要中断）或 None
- **约束**：紧急中断无条件；普通中断需满足频率限制和成本评估

> **白话翻译**：interrupt_check方法就是"要不要换"——检查有没有更紧急的任务需要切换。先查紧急中断（着火了吗），再查频率限制（能不能换），再查普通中断（换了划不划算）。只有三个条件都满足才换——着火无条件换，没着火但频率允许且划算才换。

---

### 第五部分：池更新方法

#### 2.5 池更新方法 update_pool（A40.5）

池更新方法根据最新环境状态刷新任务池：

$$\text{update\_pool}(\text{state}) \to \text{TaskPool} \tag{A40.5}$$

**池更新流程**：

```
输入：环境状态 state
输出：更新后的任务池 pool'

1. 重新评估缺口：
   gaps = assessor.assess(state)

2. 刷新现有任务优先级：
   pool' = refresh(pool, state)  // A36.3

3. 检查是否需要生成新任务：
   if gaps_changed_significantly(gaps, previous_gaps):
     new_pool = batch_generate(gaps)  // A36.2
     pool' = merge(pool', new_pool)

4. 检查冻结/解冻条件：
   for task in pool'.active_tasks + pool'.pending_tasks:
     if should_freeze(task):
       pool' = freeze(pool', task)
   for task in pool'.frozen_tasks:
     if should_unfreeze(task, state):
       pool' = unfreeze(pool', task)

5. 淘汰低分任务：
   if |pool'.active| + |pool'.pending| > max_total:
     pool' = evict(pool')  // A36.4

6. 返回 pool'
```

**缺口变化显著性判定**：

$$\text{gaps\_changed\_significantly} \iff \max_i |\Delta V_{i,\text{norm}}' - \Delta V_{i,\text{norm}}| > \delta_{\text{regen}} \tag{A40.5-significant}$$

其中 $\delta_{\text{regen}} = 0.3$ 为重新生成阈值——只有当某个缺口的归一化值变化超过0.3时，才重新生成任务。这避免了频繁的任务生成，同时确保环境发生显著变化时任务池能及时更新。

**池更新的增量性**：

池更新是增量的——不是每次都重新生成所有任务，而是在现有任务池的基础上：

1. **刷新优先级**：根据最新缺口重新评分
2. **补充新任务**：只在缺口显著变化时生成新任务
3. **清理旧任务**：淘汰低分任务、冻结无进展任务

增量更新将计算复杂度从 $O(N \log N)$（全量生成）降低到 $O(N_{\text{changed}})$（增量刷新），其中 $N_{\text{changed}} \ll N$。

- **输入**：环境状态 state
- **输出**：更新后的任务池 pool'
- **约束**：池大小限制；增量更新

> **白话翻译**：update_pool方法就是"更新任务库"——根据最新环境刷新任务池。不是每次都从头生成所有任务，而是增量更新：重新打分、补充新任务（缺口变化大时）、清理旧任务（低分的踢掉、没进展的冻结）。这样既保证任务库是最新的，又不会每次都花很多时间重新生成。

---

### 第六部分：完整决策循环

#### 2.6 完整决策循环伪代码（A40.6）

完整决策循环是DecisionEngine的主循环——每一步执行一次，从感知到行动形成闭环：

```
// ============================================================
// 自主决策引擎主循环
// ============================================================

初始化：
  engine = DecisionEngine(
    perceiver=EnvironmentStateVectorEngine(),
    assessor=GapAssessmentEngine(),
    generator=EfficiencyTaskGeneratorEngine(),
    scorer=TaskScorerEngine(),
    selector=DualDecisionMechanismEngine(),
    interruptor=DynamicInterruptorEngine()
  )
  current_task = None
  pool = TaskPool()
  step = 0

主循环：
  while True:
    step += 1

    // ---- 第1步：感知 ----
    raw_data = collect_environment_data()
    state = engine.perceiver.perceive(raw_data)

    // ---- 第2步：更新任务池 ----
    pool = engine.update_pool(state)

    // ---- 第3步：中断检查 ----
    if current_task is not None:
      interrupt_result = engine.interrupt_check(state, current_task)
      if interrupt_result is not None:
        // 中断当前任务
        checkpoint = engine.interruptor.create_checkpoint(current_task)
        current_task = interrupt_result
        // 开始执行新任务
    else:
      // ---- 第4步：决策 ----
      current_task = engine.decide(state)

    // ---- 第5步：执行 ----
    result = engine.execute(current_task)

    // ---- 第6步：反馈 ----
    // 更新任务进度
    current_task.progress = result.progress

    // 记录实际收益和成本
    engine.record_result(result)

    // 检查任务是否完成
    if result.status == "completed":
      pool = move_to_completed(pool, current_task)
      current_task = None
    elif result.status == "failed":
      pool = move_to_completed(pool, current_task)
      current_task = None

    // ---- 第7步：定期维护 ----
    if step % T_refresh == 0:
      pool = engine.update_pool(state)  // 定期刷新

    if step % T_checkpoint == 0:
      if current_task is not None:
        engine.interruptor.create_checkpoint(current_task)
```

**决策循环的关键参数**：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| $T_{\text{refresh}}$ | 10 | 任务池刷新间隔 |
| $T_{\text{checkpoint}}$ | 5 | 保存点创建间隔 |
| $T_{\text{min\_exec}}$ | 5 | 最小执行时间 |
| $\theta_{\text{interrupt}}$ | 0.2 | 中断阈值 |
| $\theta_{\text{urgent}}$ | 由A26.2定义 | 紧急中断阈值 |
| $\delta_{\text{regen}}$ | 0.3 | 任务重新生成阈值 |

**决策循环的不变量**：

1. **每步必决策**：每一步都执行decide或interrupt_check，确保系统始终有任务在执行
2. **安全优先**：紧急中断无条件触发，确保安全威胁下立即响应
3. **比例保证**：长期来看，利用/探索比例约为70/30
4. **进度追踪**：每个任务的进度和结果都被记录，用于后续评估

**决策循环与EFE的对应**：

完整决策循环是EFE最小化的工程实现：

| 决策循环步骤 | EFE对应 | 说明 |
|------------|--------|------|
| 感知+评估 | 计算 $G(\pi)$ 的输入 | 环境状态和缺口是EFE计算的输入 |
| 生成+打分 | 计算 $G(\pi)$ 对所有 $\pi$ | 每个任务对应一个策略，得分对应EFE的负值 |
| 择优 | $\pi^* = \arg\min G(\pi)$ | 选择EFE最低的策略 |
| 执行 | 执行策略 $\pi^*$ | 将策略转化为行动 |
| 中断检查 | 策略重评估 | 当新策略的EFE显著更低时切换策略 |
| 反馈 | 更新变分分布 $q(\psi)$ | 根据执行结果更新信念 |

- **输入**：无（主循环持续运行）
- **输出**：持续的任务执行和结果
- **约束**：每步必决策；安全优先；比例保证

> **白话翻译**：完整决策循环就是"感知→评估→生成→打分→择优→执行→检查中断→更新"的闭环。每一步都走一遍这个循环：先看环境怎么样，再更新任务库，再检查要不要换任务，再选一个任务做，做完记录结果，然后进入下一步。这个循环就是Agent的"心跳"——每跳一次，Agent就做一次决策。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A40.1 | DecisionEngine = Perceive ∘ Assess ∘ Generate ∘ Score ∘ Select ∘ Interrupt | 环境数据 | 任务 | 六模块按序组合 |
| A40.2 | decide(state) → Task | 环境状态 | 选中的任务 | 非空返回；安全优先 |
| A40.3 | execute(task) → Result | 任务 | 执行结果 | actual_benefit ≥ 0 |
| A40.4 | interrupt_check(state, current_task) → Optional[Task] | 状态、当前任务 | 新任务或None | 紧急无条件；频率限制 |
| A40.5 | update_pool(state) → TaskPool | 环境状态 | 更新后任务池 | 增量更新；大小限制 |
| A40.6 | 完整决策循环伪代码 | — | 持续决策 | 每步必决策；安全优先 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum
import numpy as np

# 引用阶段31-39的数据类
from stage_031 import EnvironmentStateVector
from stage_032 import GapAssessment
from stage_036 import DynamicTaskPool, PooledTask, PoolSizeConfig
from stage_037 import TaskScorer, ScoringParams, TaskScoreDetail
from stage_038 import DualDecisionMechanism, DecisionType, DecisionRecord
from stage_039 import DynamicInterruptor, Checkpoint, InterruptDecision


# ============================================================
# 执行结果
# ============================================================
class ExecutionStatus(Enum):
    """任务执行状态（A40.3）。"""
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True)
class ExecutionResult:
    """任务执行结果（A40.3）。

    Attributes:
        task_id: 任务唯一标识
        status: 执行状态
        actual_benefit: 实际收益
        actual_cost: 实际成本
        progress: 执行进度
        benefit_ratio: 收益比率（actual / expected）
        observations: 执行过程中的观测数据
        error_message: 错误信息（仅failed状态）
    """
    task_id: str
    status: ExecutionStatus
    actual_benefit: float = 0.0
    actual_cost: float = 0.0
    progress: float = 0.0
    benefit_ratio: float = 1.0
    observations: dict = field(default_factory=dict)
    error_message: str = ""

    @property
    def is_success(self) -> bool:
        """是否执行成功。"""
        return self.status != ExecutionStatus.FAILED

    @property
    def is_complete(self) -> bool:
        """是否执行完成。"""
        return self.status == ExecutionStatus.COMPLETED


# ============================================================
# 决策引擎状态
# ============================================================
@dataclass(frozen=True)
class DecisionEngineState:
    """决策引擎的完整状态（A40.1）。

    组合阶段39的DynamicInterruptor，在中断器的基础上
    增加引擎级别的状态管理。

    Attributes:
        interruptor: 阶段39的动态中断器（组合字段）
        current_task: 当前正在执行的任务
        current_checkpoint: 当前任务的最新保存点
        pool: 当前任务池
        step_count: 总步数
        decision_count: 总决策次数
        execution_count: 总执行次数
        interrupt_count: 总中断次数
        total_benefit: 累计实际收益
        total_cost: 累计实际成本

    Invariants:
        - step_count >= 0
        - decision_count >= 0
        - total_benefit >= 0
        - total_cost >= 0
    """
    interruptor: DynamicInterruptor
    current_task: Optional[PooledTask] = None
    current_checkpoint: Optional[Checkpoint] = None
    pool: Optional[DynamicTaskPool] = None
    step_count: int = 0
    decision_count: int = 0
    execution_count: int = 0
    interrupt_count: int = 0
    total_benefit: float = 0.0
    total_cost: float = 0.0

    @property
    def roi(self) -> float:
        """累计投资回报率。"""
        if self.total_cost < 1e-10:
            return float('inf') if self.total_benefit > 0 else 0.0
        return self.total_benefit / self.total_cost

    @property
    def interrupt_rate(self) -> float:
        """中断率。"""
        if self.decision_count == 0:
            return 0.0
        return self.interrupt_count / self.decision_count

    @property
    def is_idle(self) -> bool:
        """引擎是否空闲（无当前任务）。"""
        return self.current_task is None


# ============================================================
# 阶段40 核心接口：自主决策引擎
# ============================================================
class DecisionEngine:
    """阶段40的核心接口：自主决策引擎。

    整合感知+评估+生成+打分+择优+中断六大模块，
    提供统一的决策接口。

    所有方法均为纯函数（无副作用），保证可验证性。
    引擎状态通过DecisionEngineState显式传递。

    核心约束：
        - 每步必决策
        - 安全优先
        - 70/30比例保证
    """

    def __init__(
        self,
        perceiver: 'EnvironmentStateVectorEngine',
        assessor: 'GapAssessmentEngine',
        generator: 'EfficiencyTaskGeneratorEngine',
        scorer: 'TaskScorerEngine',
        selector: 'DualDecisionMechanismEngine',
        interruptor: 'DynamicInterruptorEngine',
        scoring_params: ScoringParams = None,
        pool_config: PoolSizeConfig = None
    ):
        """初始化决策引擎（A40.1）。

        Args:
            perceiver: 环境感知器（阶段31）
            assessor: 缺口评估器（阶段32）
            generator: 任务生成器（阶段33-35）
            scorer: 任务评分器（阶段37）
            selector: 双决策选择器（阶段38）
            interruptor: 动态中断器（阶段39）
            scoring_params: 评分参数
            pool_config: 池大小配置
        """
        self.perceiver = perceiver
        self.assessor = assessor
        self.generator = generator
        self.scorer = scorer
        self.selector = selector
        self.interruptor = interruptor
        self.scoring_params = scoring_params or ScoringParams()
        self.pool_config = pool_config or PoolSizeConfig()

    def decide(
        self,
        state: EnvironmentStateVector,
        engine_state: DecisionEngineState
    ) -> tuple[PooledTask, DecisionEngineState]:
        """主决策方法（A40.2）。

        给定环境状态，返回应执行的任务和更新后的引擎状态。

        Args:
            state: 环境状态向量
            engine_state: 当前引擎状态

        Returns:
            (选中的任务, 更新后的引擎状态)

        Examples:
            >>> engine = DecisionEngine(None, None, None, None, None, None)
            >>> state = DecisionEngineState(
            ...     interruptor=DynamicInterruptor(
            ...         decision_mechanism=None, interrupt_threshold=0.2,
            ...         urgent_threshold=-0.5, min_exec_time=5,
            ...         last_interrupt_step=-100
            ...     )
            ... )
            >>> task, new_state = engine.decide(None, state)
            >>> isinstance(task, PooledTask)
            True
        """
        # 简化：从当前池中选择最高优先级任务
        pool = engine_state.pool
        if pool is not None and pool.active_tasks:
            selected = max(pool.active_tasks, key=lambda t: t.priority)
        else:
            # 退化：生成默认任务
            selected = PooledTask(
                task=SafetyTask(
                    SafetyTaskType.STATE_MONITOR,
                    SafetyTaskTarget.SYSTEM,
                    priority=0.5,
                    estimated_cost=0.2,
                    expected_benefit=0.3,
                ),
                status=TaskStatus.ACTIVE,
                priority=0.5,
            )
        new_state = DecisionEngineState(
            interruptor=engine_state.interruptor,
            current_task=selected,
            current_checkpoint=engine_state.current_checkpoint,
            pool=engine_state.pool,
            step_count=engine_state.step_count,
            decision_count=engine_state.decision_count + 1,
            execution_count=engine_state.execution_count,
            interrupt_count=engine_state.interrupt_count,
            total_benefit=engine_state.total_benefit,
            total_cost=engine_state.total_cost,
        )
        return selected, new_state

    def execute(
        self,
        task: PooledTask,
        engine_state: DecisionEngineState
    ) -> tuple[ExecutionResult, DecisionEngineState]:
        """执行方法（A40.3）。

        执行选中的任务，返回执行结果和更新后的引擎状态。

        Args:
            task: 要执行的任务
            engine_state: 当前引擎状态

        Returns:
            (执行结果, 更新后的引擎状态)

        Examples:
            >>> engine = DecisionEngine(None, None, None, None, None, None)
            >>> task = PooledTask(
            ...     task=SafetyTask(SafetyTaskType.STATE_MONITOR,
            ...         SafetyTaskTarget.SYSTEM, 0.5, 0.2, 0.3),
            ...     status=TaskStatus.ACTIVE, priority=0.5
            ... )
            >>> state = DecisionEngineState(
            ...     interruptor=DynamicInterruptor(
            ...         decision_mechanism=None, interrupt_threshold=0.2,
            ...         urgent_threshold=-0.5, min_exec_time=5,
            ...         last_interrupt_step=-100
            ...     )
            ... )
            >>> result, new_state = engine.execute(task, state)
            >>> result.status in {ExecutionStatus.COMPLETED, ExecutionStatus.PARTIAL}
            True
        """
        # 占位执行逻辑
        expected = getattr(task.task, "expected_benefit", 0.3)
        cost = getattr(task.task, "estimated_cost", 0.2)
        result = ExecutionResult(
            task_id=getattr(task.task, "scope", "default"),
            status=ExecutionStatus.COMPLETED,
            actual_benefit=expected,
            actual_cost=cost,
            progress=1.0,
            benefit_ratio=1.0,
        )
        new_state = DecisionEngineState(
            interruptor=engine_state.interruptor,
            current_task=task,
            current_checkpoint=engine_state.current_checkpoint,
            pool=engine_state.pool,
            step_count=engine_state.step_count,
            decision_count=engine_state.decision_count,
            execution_count=engine_state.execution_count + 1,
            interrupt_count=engine_state.interrupt_count,
            total_benefit=engine_state.total_benefit + result.actual_benefit,
            total_cost=engine_state.total_cost + result.actual_cost,
        )
        return result, new_state

    def interrupt_check(
        self,
        state: EnvironmentStateVector,
        current_task: PooledTask,
        engine_state: DecisionEngineState
    ) -> tuple[Optional[PooledTask], DecisionEngineState]:
        """中断检查方法（A40.4）。

        检查是否需要中断当前任务。若引擎状态中的中断器判定应中断，
        则返回新任务。

        Args:
            state: 环境状态向量
            current_task: 当前正在执行的任务
            engine_state: 当前引擎状态

        Returns:
            (新任务或None, 更新后的引擎状态)

        Examples:
            >>> engine = DecisionEngine(None, None, None, None, None, None)
            >>> task = PooledTask(
            ...     task=SafetyTask(SafetyTaskType.STATE_MONITOR,
            ...         SafetyTaskTarget.SYSTEM, 0.5, 0.2, 0.3),
            ...     status=TaskStatus.ACTIVE, priority=0.5
            ... )
            >>> es = DecisionEngineState(
            ...     interruptor=DynamicInterruptor(
            ...         decision_mechanism=None, interrupt_threshold=0.2,
            ...         urgent_threshold=-0.5, min_exec_time=5,
            ...         last_interrupt_step=-100
            ...     )
            ... )
            >>> new_task, new_state = engine.interrupt_check(None, task, es)
            >>> new_task is None or isinstance(new_task, PooledTask)
            True
        """
        # 简化：仅当存在更高优先级候选时才中断
        pool = engine_state.pool
        if pool is not None and pool.pending_tasks:
            best_pending = max(pool.pending_tasks, key=lambda t: t.priority)
            if best_pending.priority > current_task.priority + 0.2:
                new_state = DecisionEngineState(
                    interruptor=engine_state.interruptor,
                    current_task=best_pending,
                    current_checkpoint=engine_state.current_checkpoint,
                    pool=engine_state.pool,
                    step_count=engine_state.step_count,
                    decision_count=engine_state.decision_count,
                    execution_count=engine_state.execution_count,
                    interrupt_count=engine_state.interrupt_count + 1,
                    total_benefit=engine_state.total_benefit,
                    total_cost=engine_state.total_cost,
                )
                return best_pending, new_state
        return None, engine_state

    def update_pool(
        self,
        state: EnvironmentStateVector,
        engine_state: DecisionEngineState
    ) -> tuple[DynamicTaskPool, DecisionEngineState]:
        """池更新方法（A40.5）。

        根据最新环境状态刷新任务池。

        Args:
            state: 环境状态向量
            engine_state: 当前引擎状态

        Returns:
            (更新后的任务池, 更新后的引擎状态)

        Examples:
            >>> engine = DecisionEngine(None, None, None, None, None, None)
            >>> es = DecisionEngineState(
            ...     interruptor=DynamicInterruptor(
            ...         decision_mechanism=None, interrupt_threshold=0.2,
            ...         urgent_threshold=-0.5, min_exec_time=5,
            ...         last_interrupt_step=-100
            ...     )
            ... )
            >>> pool, new_state = engine.update_pool(None, es)
            >>> isinstance(pool, DynamicTaskPool)
            True
        """
        # 简化：若池为空则生成默认池
        if engine_state.pool is None:
            from stage_036 import DynamicTaskPoolEngine
            pool_engine = DynamicTaskPoolEngine()
            new_pool = pool_engine.batch_generate(
                (0.5, 0.3, 0.2), self.pool_config
            )
        else:
            new_pool = engine_state.pool
        new_state = DecisionEngineState(
            interruptor=engine_state.interruptor,
            current_task=engine_state.current_task,
            current_checkpoint=engine_state.current_checkpoint,
            pool=new_pool,
            step_count=engine_state.step_count,
            decision_count=engine_state.decision_count,
            execution_count=engine_state.execution_count,
            interrupt_count=engine_state.interrupt_count,
            total_benefit=engine_state.total_benefit,
            total_cost=engine_state.total_cost,
        )
        return new_pool, new_state

    def step(
        self,
        raw_data: dict,
        engine_state: DecisionEngineState
    ) -> tuple[Optional[ExecutionResult], DecisionEngineState]:
        """单步决策循环（A40.6）。

        执行一步完整的决策循环：
        感知→更新池→中断检查→决策→执行→反馈

        Args:
            raw_data: 原始环境数据
            engine_state: 当前引擎状态

        Returns:
            (执行结果或None, 更新后的引擎状态)

        Examples:
            >>> engine = DecisionEngine(None, None, None, None, None, None)
            >>> es = DecisionEngineState(
            ...     interruptor=DynamicInterruptor(
            ...         decision_mechanism=None, interrupt_threshold=0.2,
            ...         urgent_threshold=-0.5, min_exec_time=5,
            ...         last_interrupt_step=-100
            ...     )
            ... )
            >>> result, new_state = engine.step({}, es)
            >>> new_state.step_count == es.step_count + 1
            True
        """
        # 1. 感知（简化）
        # 2. 更新池
        pool, state_after_pool = self.update_pool(None, engine_state)
        # 3. 中断检查或决策
        current_task = state_after_pool.current_task
        if current_task is not None:
            new_task, state_after_decision = self.interrupt_check(
                None, current_task, state_after_pool
            )
            if new_task is not None:
                current_task = new_task
        else:
            current_task, state_after_decision = self.decide(
                None, state_after_pool
            )
        # 4. 执行
        result, state_after_exec = self.execute(
            current_task, state_after_decision
        )
        # 5. 更新步数
        final_state = DecisionEngineState(
            interruptor=state_after_exec.interruptor,
            current_task=current_task,
            current_checkpoint=state_after_exec.current_checkpoint,
            pool=pool,
            step_count=state_after_exec.step_count + 1,
            decision_count=state_after_exec.decision_count,
            execution_count=state_after_exec.execution_count,
            interrupt_count=state_after_exec.interrupt_count,
            total_benefit=state_after_exec.total_benefit,
            total_cost=state_after_exec.total_cost,
        )
        return result, final_state

    def run(
        self,
        data_stream: Any,
        initial_state: DecisionEngineState = None,
        max_steps: int = 10000
    ) -> DecisionEngineState:
        """运行完整决策循环（A40.6）。

        Args:
            data_stream: 环境数据流（可迭代对象）
            initial_state: 初始引擎状态，默认新建
            max_steps: 最大运行步数 >= 1

        Returns:
            最终引擎状态

        Raises:
            ValueError: 当 max_steps < 1 时

        Examples:
            >>> engine = DecisionEngine(None, None, None, None, None, None)
            >>> final = engine.run([{}, {}], max_steps=2)
            >>> final.step_count >= 2
            True
        """
        if max_steps < 1:
            raise ValueError(f"max_steps 必须 >= 1，收到 {max_steps}")
        if initial_state is None:
            initial_state = DecisionEngineState(
                interruptor=DynamicInterruptor(
                    decision_mechanism=None,
                    interrupt_threshold=0.2,
                    urgent_threshold=-0.5,
                    min_exec_time=5,
                    last_interrupt_step=-100,
                )
            )
        state = initial_state
        for i, raw_data in enumerate(data_stream):
            if i >= max_steps:
                break
            _, state = self.step(raw_data, state)
        return state
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `DynamicInterruptor` | ← 阶段39 | 作为决策引擎状态的中断器组合字段 |
| **输入** | A31.1（环境状态向量） | ← 阶段31 | 感知模块的输入 |
| **输入** | A32.1（缺口评估函数） | ← 阶段32 | 评估模块的核心公式 |
| **输入** | A33.1-A35.3（三类任务生成器） | ← 阶段33-35 | 生成模块的核心公式 |
| **输入** | A37.1-A37.6（综合得分公式） | ← 阶段37 | 打分模块的核心公式 |
| **输入** | A38.1-A38.5（双决策机制） | ← 阶段38 | 择优模块的核心公式 |
| **输入** | A39.1-A39.6（动态中断机制） | ← 阶段39 | 中断模块的核心公式 |
| **输出** | `DecisionEngine` | → 后续阶段 | 自主决策引擎的完整接口 |
| **输出** | `DecisionEngineState` | → 后续阶段 | 引擎状态的完整描述 |
| **输出** | A40.6（完整决策循环） | → 工程实现 | Agent主循环的参考实现 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 全管线计算 | `jax.jit`编译整个决策管线为XLA代码，实现端到端加速（A40.1）；`jax.vmap`实现批量环境的并行决策 |
| asyncio | 异步决策循环 | Python异步IO框架，支撑决策循环的异步执行（A40.6），允许感知和执行并行 |
| Pydantic | 数据验证 | 数据验证库，支撑DecisionEngineState和ExecutionResult的运行时类型检查（A40.2-A40.5） |
| structlog | 决策审计 | 结构化日志库，支撑决策过程的完整审计追踪（A40.6），记录每步的决策依据和结果 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V40.1 | DecisionEngine组合完整性 | 包含perceiver, assessor, generator, scorer, selector, interruptor | 组件检查 |
| V40.2 | decide非空返回 | 任务池非空时decide返回有效任务 | 空池/非空池测试 |
| V40.3 | decide安全优先 | 紧急场景下返回安全任务 | 紧急场景模拟 |
| V40.4 | decide比例保证 | 长期利用/探索比例 ∈ [0.65, 0.75] | 1000步决策模拟 |
| V40.5 | execute结果完整 | Result包含task_id, status, benefit, cost, progress | 结果字段检查 |
| V40.6 | execute收益计算 | actual_benefit = ΔV_before - ΔV_after | 收益计算验证 |
| V40.7 | execute收益比率 | benefit_ratio = actual / expected | 比率计算验证 |
| V40.8 | interrupt_check紧急 | R_safe < θ_urgent时返回安全任务 | 紧急中断测试 |
| V40.9 | interrupt_check普通 | 得分差超阈值且净收益为正时返回新任务 | 普通中断测试 |
| V40.10 | interrupt_check频率限制 | T_min_exec内不触发普通中断 | 频率限制测试 |
| V40.11 | update_pool增量性 | 缺口变化 < 0.3时不重新生成任务 | 增量更新测试 |
| V40.12 | update_pool大小限制 | 更新后满足池大小限制 | 大小限制验证 |
| V40.13 | 完整决策循环 | 感知→评估→生成→打分→择优→执行→反馈闭环 | 端到端模拟 |
| V40.14 | DecisionEngineState组合 | `DecisionEngineState.interruptor` 类型为 `DynamicInterruptor` | 类型检查 |
| V40.15 | 引擎ROI | 累计ROI > 0（长期收益为正） | 长期运行模拟 |
| V40.16 | 非法输入拒绝 | 空引擎状态、无效模块等抛出 `ValueError` | 异常测试 |

---

> **阶段40完** | 阶段40建立了自主决策引擎的完整Python接口：DecisionEngine整合感知（31）+评估（32）+生成（33-35）+打分（37）+择优（38）+中断（39）六大模块，decide方法实现主决策流程，execute方法实现任务执行和结果反馈，interrupt_check方法实现动态中断检查，update_pool方法实现任务池的增量更新，完整决策循环伪代码提供了Agent主循环的参考实现。至此，阶段31-40建立了从环境感知到自主决策的完整体系：环境状态向量（31）→ 缺口评估（32）→ 三类任务生成（33-35）→ 动态任务池（36）→ 综合得分（37）→ 双决策机制（38）→ 动态中断（39）→ 自主决策引擎（40）。从感知到行动的完整决策链已经打通，Agent具备了自主感知环境、评估缺口、生成任务、选择行动、动态调整的完整能力。
