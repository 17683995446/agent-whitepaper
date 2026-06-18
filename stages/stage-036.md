# 阶段36：动态任务池——从任务生成到动态调度

> **阶段编号**：S36 | **所属层**：第四层·任务规划层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 任务池数据结构 | "四格收纳箱"——正在做、等着做、做完了、暂停做，四格分类管理 |
> | 批量任务生成 | "一键下单"——根据所有缺口一次性生成一批任务 |
> | 任务刷新 | "重新打分"——环境变了，任务的优先级也要跟着变 |
> | 任务淘汰 | "清仓处理"——池子满了，把低分任务踢出去 |
> | 任务冻结/解冻 | "暂停/恢复"——长期没进展的任务先放一放，不占位置 |
> | 池大小限制 | "仓库容量"——活跃任务最多10个，候补最多40个 |

---

## 1. 第一性原理依据

本阶段从阶段33-35（三类任务生成器）的核心结论出发，建立动态任务池的完整数据结构与操作机制。核心依据为：

- **A33.1-A33.4（安全任务生成器）**（引用阶段33）：SafetyTaskGenerator将安全缺口映射为参数化的安全任务列表，包含任务生成规则、类型枚举、参数化和去重。任务池需要将这些安全任务纳入统一管理。
- **A34.1-A34.3（收益任务生成器）**（引用阶段34）：GainTaskGenerator在安全任务基础上增加收益任务，组合SafetyTaskGenerator作为安全任务的继承。任务池需要同时管理安全任务和收益任务。
- **A35.1-A35.3（能效任务生成器）**（引用阶段35）：EfficiencyTaskGenerator在收益任务基础上增加能效任务，组合GainTaskGenerator。至此，三类15种任务（5安全+5收益+5能效）全部生成，任务池需要统一管理所有类型的任务。
- **A25.1（价值缺口定义）**（引用阶段25）：$\Delta V = V_{\text{target}} - V_{\text{current}}$，缺口是任务生成的驱动力。任务池的刷新操作需要根据最新的缺口状态重新评估任务的优先级。
- **A32.5（缺口归一化）**（引用阶段32）：$\Delta V_{i,\text{norm}} = \Delta V_i / \Delta V_{i,\max}$，归一化缺口为任务评分提供了跨驱力的可比尺度。

本阶段的核心目标是：**定义任务池数据结构 TaskPool = {active_tasks, pending_tasks, completed_tasks, frozen_tasks}，实现批量任务生成 batch_generate(gaps) → TaskPool，任务刷新 refresh(pool, state) → TaskPool，任务淘汰 evict(pool, max_size=50) → TaskPool，任务冻结/解冻 freeze/unfreeze，设定池大小限制 max_active=10, max_pending=40，并封装为DynamicTaskPool数据类，组合EfficiencyTaskGenerator作为任务生成的来源。** 动态任务池是从"任务生成"到"任务调度"的关键枢纽——它将三类任务生成器的输出统一管理，根据环境变化动态调整任务的状态和优先级，确保系统始终聚焦于最有价值的任务。

---

## 2. 核心数学公式

### 第一部分：任务池数据结构

#### 2.1 任务池数据结构（A36.1）

任务池定义为四个子池的联合结构：

$$\text{TaskPool} = \{\text{active\_tasks}, \text{pending\_tasks}, \text{completed\_tasks}, \text{frozen\_tasks}\} \tag{A36.1}$$

其中：

- $\text{active\_tasks} \subseteq \mathcal{T}$：活跃任务集——当前正在执行的任务，$|\text{active\_tasks}| \leq \text{max\_active} = 10$
- $\text{pending\_tasks} \subseteq \mathcal{T}$：待执行任务集——已生成但尚未开始执行的任务，$|\text{pending\_tasks}| \leq \text{max\_pending} = 40$
- $\text{completed\_tasks} \subseteq \mathcal{T}$：已完成任务集——已经执行完毕的任务（历史记录）
- $\text{frozen\_tasks} \subseteq \mathcal{T}$：冻结任务集——暂时挂起的任务，不参与调度

**四池的互斥性与完备性**：

$$\mathcal{T}_{\text{all}} = \text{active} \cup \text{pending} \cup \text{completed} \cup \text{frozen}, \quad \text{active} \cap \text{pending} \cap \text{completed} \cap \text{frozen} = \emptyset \tag{A36.1-disjoint}$$

任意时刻，每个任务恰好属于四个子池之一——不存在同时属于两个子池的任务，也不存在不属于任何子池的任务。

**任务状态转移图**：

任务在四个子池之间的状态转移遵循以下规则：

$$\text{pending} \xrightarrow{\text{schedule}} \text{active} \xrightarrow{\text{complete}} \text{completed} \tag{A36.1-transition-1}$$

$$\text{active} \xrightarrow{\text{freeze}} \text{frozen} \xrightarrow{\text{unfreeze}} \text{pending} \tag{A36.1-transition-2}$$

$$\text{pending} \xrightarrow{\text{evict}} \text{removed} \tag{A36.1-transition-3}$$

状态转移的约束：

1. **pending → active**：仅当 $|\text{active}| < \text{max\_active}$ 时允许
2. **active → completed**：任务执行完毕后自动转移
3. **active → frozen**：满足冻结条件时转移（A36.5）
4. **frozen → pending**：满足解冻条件时转移（A36.5）
5. **pending → removed**：淘汰低分任务时移除（A36.4）

**池大小限制的物理依据**：

max_active=10 的限制基于认知负荷理论——人类工作记忆的容量约为7±2个信息块，Agent同时执行的活跃任务数也应有限。过多的活跃任务会导致：

1. **上下文切换成本**：每次切换任务需要保存和恢复上下文，频繁切换浪费资源
2. **注意力分散**：同时执行过多任务，每个任务获得的资源不足
3. **调度复杂性**：活跃任务越多，调度决策越复杂

max_pending=40 的限制基于内存和计算约束——待执行任务需要存储其参数和优先级信息，过多的待执行任务消耗内存且增加排序成本。

- **输入**：三类任务生成器的输出（SafetyTask, GainTask, EfficiencyTask）
- **输出**：TaskPool实例，包含四个子池
- **约束**：$|\text{active}| \leq 10$；$|\text{pending}| \leq 40$；四池互斥且完备

> **白话翻译**：任务池就像"四格收纳箱"——正在做的放active格，等着做的放pending格，做完的放completed格，暂停的放frozen格。active格最多放10个（同时做太多事做不好），pending格最多放40个（候补太多也记不住）。任务可以在四格之间移动：从候补到正在做，从正在做到做完，从正在做到暂停，从暂停回到候补。

---

### 第二部分：批量任务生成

#### 2.2 批量任务生成（A36.2）

批量任务生成将三类缺口一次性转化为完整的任务池：

$$\text{batch\_generate}(\text{gaps}) \to \text{TaskPool} \tag{A36.2}$$

其中 $\text{gaps} = (\Delta V_{\text{safe}}, \Delta V_{\text{gain}}, \Delta V_{\text{eff}})$ 为三大缺口的当前值。

**生成流程**：

```
输入：缺口三元组 gaps = (ΔV_safe, ΔV_gain, ΔV_eff)
输出：TaskPool 实例

1. 安全任务生成：
   safety_gen = SafetyTaskGenerator.generate(gaps.ΔV_safe)
   safety_tasks = safety_gen.tasks

2. 收益任务生成：
   gain_gen = GainTaskGenerator.generate(safety_gen)
   gain_tasks = gain_gen.gain_tasks

3. 能效任务生成：
   eff_gen = EfficiencyTaskGenerator.generate(gain_gen)
   eff_tasks = eff_gen.eff_tasks

4. 合并所有任务：
   all_tasks = safety_tasks ∪ gain_tasks ∪ eff_tasks

5. 初始分配：
   active_tasks = top_k(all_tasks, max_active=10)
   pending_tasks = rest(all_tasks) ∩ top_k(all_tasks, max_active + max_pending)
   completed_tasks = ∅
   frozen_tasks = ∅

6. 返回 TaskPool(active, pending, completed, frozen)
```

**初始分配策略**：

初始分配时，按优先级从高到低选择活跃任务：

$$\text{active\_tasks} = \text{top\_k}(\text{all\_tasks}, \text{key}=\text{priority}, k=\text{max\_active}) \tag{A36.2-active}$$

$$\text{pending\_tasks} = \text{sorted}(\text{all\_tasks} \setminus \text{active\_tasks}, \text{key}=\text{priority})[:\text{max\_pending}] \tag{A36.2-pending}$$

超出 max_active + max_pending 容量的任务直接丢弃——这些是优先级最低的任务，在当前环境下不值得执行。

**批量生成的去重**：

三类任务生成器各自内部已经做了去重（A33.4, A34.3, A35.3），但跨类型的任务仍可能存在重叠。跨类型去重使用扩展的相似度计算：

$$\text{sim}_{\text{cross}}(\text{task}_i, \text{task}_j) = w_t \cdot \text{sim}_{\text{type}} + w_g \cdot \text{sim}_{\text{target}} + w_s \cdot \text{sim}_{\text{scope}} + w_o \cdot \text{sim}_{\text{objective}} \tag{A36.2-dedup}$$

其中 $\text{sim}_{\text{objective}}$ 为优化目标重叠度（引用A35.3-target），权重 $w_t = 0.3, w_g = 0.2, w_s = 0.2, w_o = 0.3$。跨类型去重时，类型相同的权重降低（因为不同类型可能有不同的执行策略），优化目标重叠度的权重提高（因为优化同一目标的任务应合并）。

**批量生成的时间复杂度**：

设三类任务生成器分别生成 $n_s, n_g, n_e$ 个任务，总任务数 $N = n_s + n_g + n_e$。初始分配的时间复杂度为 $O(N \log N)$（排序），跨类型去重的时间复杂度为 $O(N^2)$（两两比较）。在典型场景下，$N \approx 15$-$30$（每类5-10个任务），计算成本可控。

- **输入**：缺口三元组 $\text{gaps} = (\Delta V_{\text{safe}}, \Delta V_{\text{gain}}, \Delta V_{\text{eff}})$
- **输出**：TaskPool实例
- **约束**：$|\text{active}| \leq 10$；$|\text{pending}| \leq 40$；跨类型去重阈值0.8

> **白话翻译**：批量任务生成就是"一键下单"——根据当前三个缺口（安全、收益、能效），一次性生成所有类型的任务，然后按优先级排好队。最紧急的10个直接开始做（active），接下来40个排队等候（pending），剩下的先不要了。不同类型的任务如果做的事情差不多（比如安全任务和收益任务都在探索同一个区域），就合并成一个。

---

### 第三部分：任务刷新

#### 2.3 任务刷新（A36.3）

任务刷新根据最新的环境状态重新评估所有任务的优先级和状态：

$$\text{refresh}(\text{pool}, \text{state}) \to \text{TaskPool} \tag{A36.3}$$

其中 $\text{state}$ 为当前环境状态向量（引用阶段31，A31.1）。

**刷新流程**：

```
输入：当前任务池 pool，环境状态 state
输出：刷新后的任务池 pool'

1. 重新计算缺口：
   gaps' = gap(state)  // A32.1

2. 对每个活跃任务和待执行任务：
   a. 重新计算优先级 priority' = f(ΔV_i', type, scene)
   b. 重新计算预期收益 expected_benefit' = f(ΔV_i', η_type)
   c. 更新任务参数

3. 重新排序：
   active_tasks' = top_k(all_tasks, key=priority', k=max_active)
   pending_tasks' = sorted(rest, key=priority')[:max_pending]

4. 检查冻结条件：
   for task in active_tasks' ∪ pending_tasks':
     if should_freeze(task, state):
       move to frozen_tasks

5. 检查解冻条件：
   for task in frozen_tasks:
     if should_unfreeze(task, state):
       move to pending_tasks

6. 返回 pool'
```

**优先级重算公式**：

刷新时，任务的优先级根据最新的缺口值重新计算：

$$\text{priority}' = \Delta V_{i,\text{norm}}' \cdot w_{\text{type}} \cdot w_{\text{scene}}' \tag{A36.3-priority}$$

其中 $\Delta V_{i,\text{norm}}'$ 为最新的归一化缺口（引用A32.5），$w_{\text{scene}}'$ 为最新的场景权重（引用A26.1）。

**刷新的触发条件**：

任务刷新不是每一步都执行的——频繁刷新的计算成本较高。刷新的触发条件为：

1. **定期刷新**：每隔 $T_{\text{refresh}}$ 步刷新一次（默认 $T_{\text{refresh}} = 10$ 步）
2. **场景切换触发**：当场景分类器的输出发生变化时（引用A26.1），立即刷新
3. **缺口突变触发**：当任一缺口的归一化值变化超过 $\delta_{\text{refresh}} = 0.2$ 时，立即刷新

**刷新的增量性**：

为降低刷新的计算成本，采用增量刷新策略：

1. **已完成任务不刷新**：completed_tasks中的任务不需要重新评分
2. **冻结任务延迟刷新**：frozen_tasks中的任务在解冻时才重新评分
3. **优先级变化小的任务不重排**：如果任务的新优先级与旧优先级的差值 $< \epsilon_{\text{refresh}} = 0.05$，保持原排序

增量刷新将计算复杂度从 $O(N \log N)$ 降低到 $O(N_{\text{changed}} \log N_{\text{changed}})$，其中 $N_{\text{changed}}$ 为优先级显著变化的任务数。

- **输入**：当前任务池 pool，环境状态 state
- **输出**：刷新后的任务池 pool'
- **约束**：刷新后仍满足池大小限制；优先级变化连续（不突变）

> **白话翻译**：任务刷新就是"重新打分"——环境变了，原来紧急的任务可能不紧急了，原来不紧急的可能变紧急了。刷新就是根据最新的环境状态重新算一遍所有任务的优先级，然后重新排队。但不需要每一步都刷新——定期刷新就行，除非发生大事（场景切换或缺口突变）才立即刷新。

---

### 第四部分：任务淘汰

#### 2.4 任务淘汰（A36.4）

任务淘汰在池容量超限时移除最低优先级的任务：

$$\text{evict}(\text{pool}, \text{max\_size}=50) \to \text{TaskPool} \tag{A36.4}$$

其中 $\text{max\_size} = \text{max\_active} + \text{max\_pending} = 10 + 40 = 50$ 为活跃任务和待执行任务的总容量上限。

**淘汰策略**：

当 $|\text{active}| + |\text{pending}| > \text{max\_size}$ 时，需要淘汰低分任务。淘汰策略为优先级最低的任务优先淘汰：

$$\text{evicted} = \text{bottom\_k}(\text{pending\_tasks}, \text{key}=\text{priority}, k=|\text{active}| + |\text{pending}| - \text{max\_size}) \tag{A36.4-evict}$$

**淘汰的保护机制**：

并非所有低优先级任务都可以淘汰。以下类型的任务受保护：

1. **安全任务不淘汰**：$\text{type} \in \mathcal{T}_{\text{safe}}$ 的任务不参与淘汰——安全任务直接关联系统生存，即使优先级低也不能丢弃
2. **正在执行的任务不淘汰**：$\text{active\_tasks}$ 中的任务不参与淘汰——正在执行的任务不能中途取消（除非通过中断机制，引用阶段39）
3. **刚生成的任务保护期**：新生成的任务在 $T_{\text{protect}} = 5$ 步内不参与淘汰——给新任务一个"试用期"，避免刚生成就被淘汰

**淘汰后的重新填充**：

淘汰低分任务后，如果待执行队列有空位，可以从冻结任务中解冻高分任务填充：

$$\text{refill\_from\_frozen} = \text{top\_k}(\text{frozen\_tasks}, \text{key}=\text{priority}, k=\text{max\_pending} - |\text{pending}'|) \tag{A36.4-refill}$$

**淘汰的成本-收益分析**：

淘汰低分任务的收益是释放池容量（为更高分的任务腾出空间），成本是丢失了被淘汰任务的信息（如果环境恢复，需要重新生成）。在典型场景下，被淘汰的任务优先级 $< 0.1$（归一化后），重新生成的成本很低，因此淘汰的净收益为正。

- **输入**：当前任务池 pool，最大容量 max_size=50
- **输出**：淘汰后的任务池 pool'
- **约束**：安全任务不淘汰；活跃任务不淘汰；新任务保护期内不淘汰

> **白话翻译**：任务淘汰就是"清仓处理"——池子满了（最多50个任务），就把优先级最低的踢出去。但有三条保护规则：安全任务不能踢（人命关天），正在做的不能踢（半途而废浪费更大），刚来的给5步试用期（别刚进门就赶走）。踢完之后，如果候补队列有空位，可以从暂停区调高分任务回来。

---

### 第五部分：任务冻结/解冻

#### 2.5 任务冻结/解冻（A36.5）

任务冻结将长期无进展的任务暂时挂起，解冻将条件满足的冻结任务恢复到待执行队列。

**冻结条件**：

任务满足以下条件之一时被冻结：

$$\text{freeze}(\text{task}) \iff \text{rounds\_no\_progress}(\text{task}) \geq 5 \tag{A36.5-freeze}$$

其中 $\text{rounds\_no\_progress}(\text{task})$ 为任务连续无收益的轮次数。引用阶段85的5轮无收益冻结规则——如果一个任务连续5轮执行都没有产生任何收益（预期收益与实际收益的差值 $> 0.5 \cdot \text{expected\_benefit}$），则冻结该任务。

**无收益的判定标准**：

$$\text{no\_progress}(\text{task}, t) \iff \text{actual\_benefit}(\text{task}, t) < 0.5 \cdot \text{expected\_benefit}(\text{task}) \tag{A36.5-noprogress}$$

即实际收益低于预期收益的50%时，视为该轮无进展。

**冻结的附加条件**：

除了5轮无收益外，以下条件也触发冻结：

1. **资源不可用**：任务所需的资源（CPU、内存、时间）当前不可用，且短期内不会释放
2. **依赖未满足**：任务依赖的前置任务尚未完成
3. **场景不匹配**：任务类型与当前场景严重不匹配（如收益任务在紧急场景下）

**解冻条件**：

冻结任务满足以下条件之一时被解冻：

$$\text{unfreeze}(\text{task}) \iff \text{condition\_changed}(\text{task}) \tag{A36.5-unfreeze}$$

具体解冻条件：

1. **环境变化**：缺口值变化超过 $\delta_{\text{unfreeze}} = 0.3$——原来无进展的任务可能因为环境变化而变得有价值
2. **资源释放**：任务所需的资源变为可用
3. **依赖完成**：前置任务已完成
4. **场景切换**：场景切换到与任务类型匹配的场景
5. **定期重评**：每 $T_{\text{reassess}} = 50$ 步对冻结任务重新评估一次

**冻结任务的信息保留**：

冻结任务保留其完整的参数信息（type, target, priority, estimated_cost, expected_benefit），以及冻结时的环境状态快照和冻结原因。这些信息在解冻时用于快速恢复任务的执行上下文。

**冻结/解冻的状态转移**：

$$\text{active} \xrightarrow{\text{5轮无收益}} \text{frozen} \xrightarrow{\text{条件变化}} \text{pending} \tag{A36.5-state}$$

注意：冻结任务解冻后进入 pending 而非 active——解冻只是恢复候选资格，是否立即执行取决于优先级排序。

- **输入**：任务 task，环境状态 state
- **输出**：冻结/解冻决策
- **约束**：冻结阈值5轮；解冻需满足至少一个条件

> **白话翻译**：任务冻结/解冻就像"暂停/恢复"——一个任务连续5轮都没啥进展，就先暂停它，把资源让给更有希望的任务。暂停不是放弃——任务的所有信息都保留着，等条件变了（环境变了、资源有了、场景换了）再恢复。恢复后重新排队，不一定马上执行。

---

### 第六部分：任务池大小限制

#### 2.6 任务池大小限制（A36.6）

任务池的大小限制确保系统在有限的计算资源下高效运行：

$$\text{max\_active} = 10, \quad \text{max\_pending} = 40, \quad \text{max\_total} = 50 \tag{A36.6}$$

**大小限制的自适应调整**：

在极端场景下，池大小限制可以动态调整：

| 场景 | max_active | max_pending | 理由 |
|------|-----------|------------|------|
| urgent | 5 | 20 | 紧急场景聚焦少量关键任务 |
| stable | 15 | 50 | 空闲场景允许更多并行探索 |
| tight | 8 | 30 | 资源紧张时适度缩减 |

自适应调整公式：

$$\text{max\_active}' = \text{max\_active} \cdot \text{scale}(s) \tag{A36.6-scale}$$

其中 $\text{scale}(s)$ 为场景缩放因子：$\text{scale}(\text{urgent}) = 0.5$，$\text{scale}(\text{stable}) = 1.5$，$\text{scale}(\text{tight}) = 0.8$。

**completed_tasks的容量管理**：

completed_tasks没有硬性大小限制（作为历史记录），但需要定期清理：

$$\text{clean\_completed}(\text{pool}) = \text{pool}.\text{completed\_tasks}[-K:] \tag{A36.6-clean}$$

默认保留最近 $K = 100$ 条完成记录，更早的记录归档到长期存储。

**frozen_tasks的容量管理**：

frozen_tasks的大小限制为 max_frozen = 20。超过限制时，冻结时间最长的任务被永久移除：

$$\text{evict\_frozen}(\text{pool}) = \text{oldest\_k}(\text{frozen\_tasks}, k=|\text{frozen}| - \text{max\_frozen}) \tag{A36.6-evict-frozen}$$

- **输入**：场景标识 s
- **输出**：调整后的池大小限制
- **约束**：max_active ≥ 1；max_pending ≥ max_active；缩放因子 > 0

> **白话翻译**：池大小限制就是"仓库容量"——正常情况下活跃任务最多10个，候补最多40个。但可以根据情况调整：着火时少做几件事但每件做精（5个活跃），空闲时多线并行探索（15个活跃），缺钱时适度缩减（8个活跃）。做完的任务保留最近100条记录，暂停的任务最多保留20个。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A36.1 | TaskPool = {active_tasks, pending_tasks, completed_tasks, frozen_tasks} | 任务集合 | 任务池 | 四池互斥完备；max_active=10, max_pending=40 |
| A36.2 | batch_generate(gaps) → TaskPool | 缺口三元组 | 任务池 | 跨类型去重阈值0.8 |
| A36.3 | refresh(pool, state) → TaskPool | 任务池、环境状态 | 刷新后任务池 | 刷新触发条件；增量刷新 |
| A36.4 | evict(pool, max_size=50) → TaskPool | 任务池、最大容量 | 淘汰后任务池 | 安全任务不淘汰；新任务保护期 |
| A36.5 | freeze(task) ⟺ rounds_no_progress ≥ 5 | 任务、环境状态 | 冻结/解冻决策 | 5轮无收益冻结；解冻条件 |
| A36.6 | max_active=10, max_pending=40, max_total=50 | 场景标识 | 池大小限制 | 自适应缩放 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np

# 引用阶段35的数据类
from stage_035 import EfficiencyTaskGenerator, EfficiencyTask
from stage_034 import GainTask
from stage_033 import SafetyTask


# ============================================================
# 任务状态枚举
# ============================================================
class TaskStatus(Enum):
    """任务在池中的状态（A36.1）。"""
    ACTIVE = "active"
    PENDING = "pending"
    COMPLETED = "completed"
    FROZEN = "frozen"


# ============================================================
# 任务池中的任务包装器
# ============================================================
@dataclass(frozen=True)
class PooledTask:
    """任务池中的任务包装器（A36.1）。

    为任务增加池管理所需的元数据。

    Attributes:
        task: 原始任务（SafetyTask | GainTask | EfficiencyTask）
        status: 任务状态
        priority: 当前优先级（可随刷新更新）
        rounds_no_progress: 连续无收益轮次
        frozen_reason: 冻结原因（仅frozen状态）
        created_at: 任务创建时间步
        last_refreshed: 上次刷新时间步
    """
    task: SafetyTask | GainTask | EfficiencyTask
    status: TaskStatus
    priority: float
    rounds_no_progress: int = 0
    frozen_reason: str = ""
    created_at: int = 0
    last_refreshed: int = 0

    @property
    def task_type_category(self) -> str:
        """任务类型分类（safe/gain/eff）。"""
        if isinstance(self.task, SafetyTask):
            return "safe"
        elif isinstance(self.task, GainTask):
            return "gain"
        else:
            return "eff"

    @property
    def should_freeze(self) -> bool:
        """是否满足冻结条件（A36.5）。"""
        return self.rounds_no_progress >= 5


# ============================================================
# 任务池大小配置
# ============================================================
@dataclass(frozen=True)
class PoolSizeConfig:
    """任务池大小配置（A36.6）。

    Attributes:
        max_active: 活跃任务上限
        max_pending: 待执行任务上限
        max_frozen: 冻结任务上限
        max_completed: 完成任务保留数
    """
    max_active: int = 10
    max_pending: int = 40
    max_frozen: int = 20
    max_completed: int = 100

    @property
    def max_total(self) -> int:
        """活跃+待执行总上限。"""
        return self.max_active + self.max_pending

    def scale_for_scene(self, scene: str) -> 'PoolSizeConfig':
        """根据场景缩放池大小（A36.6-scale）。"""
        scale_map = {"urgent": 0.5, "stable": 1.5, "tight": 0.8}
        scale = scale_map.get(scene, 1.0)
        return PoolSizeConfig(
            max_active=max(1, int(self.max_active * scale)),
            max_pending=max(self.max_active, int(self.max_pending * scale)),
            max_frozen=self.max_frozen,
            max_completed=self.max_completed,
        )


# ============================================================
# 阶段36 数据类：动态任务池
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class DynamicTaskPool:
    """动态任务池的完整描述。

    组合阶段35的EfficiencyTaskGenerator，在三类任务生成器
    的基础上增加动态池管理能力。

    Attributes:
        eff_generator: 阶段35的能效任务生成器（组合字段）
        active_tasks: 活跃任务列表
        pending_tasks: 待执行任务列表
        completed_tasks: 已完成任务列表
        frozen_tasks: 冻结任务列表
        size_config: 池大小配置
        last_refresh_step: 上次刷新的时间步
        total_generated: 累计生成的任务数
        total_evicted: 累计淘汰的任务数
        total_frozen: 累计冻结的任务数

    Invariants:
        - len(active_tasks) <= size_config.max_active
        - len(pending_tasks) <= size_config.max_pending
        - len(frozen_tasks) <= size_config.max_frozen
        - 四池中的任务不重复
    """
    eff_generator: EfficiencyTaskGenerator
    active_tasks: tuple[PooledTask, ...]
    pending_tasks: tuple[PooledTask, ...]
    completed_tasks: tuple[PooledTask, ...]
    frozen_tasks: tuple[PooledTask, ...]
    size_config: PoolSizeConfig
    last_refresh_step: int
    total_generated: int
    total_evicted: int
    total_frozen: int

    def __post_init__(self):
        # 验证池大小限制
        if len(self.active_tasks) > self.size_config.max_active:
            raise ValueError(
                f"活跃任务数 {len(self.active_tasks)} "
                f"超过上限 {self.size_config.max_active}"
            )
        if len(self.pending_tasks) > self.size_config.max_pending:
            raise ValueError(
                f"待执行任务数 {len(self.pending_tasks)} "
                f"超过上限 {self.size_config.max_pending}"
            )

    @property
    def all_active_pending(self) -> list[PooledTask]:
        """所有活跃和待执行任务，按优先级降序排列。"""
        tasks = list(self.active_tasks) + list(self.pending_tasks)
        return sorted(tasks, key=lambda t: t.priority, reverse=True)

    @property
    def active_task_types(self) -> dict[str, int]:
        """活跃任务的类型分布。"""
        dist = {}
        for t in self.active_tasks:
            cat = t.task_type_category
            dist[cat] = dist.get(cat, 0) + 1
        return dist

    @property
    def utilization(self) -> float:
        """池利用率。"""
        total = self.size_config.max_total
        used = len(self.active_tasks) + len(self.pending_tasks)
        if total == 0:
            return 0.0
        return used / total

    @property
    def freeze_rate(self) -> float:
        """冻结率。"""
        total = self.total_generated
        if total == 0:
            return 0.0
        return self.total_frozen / total


# ============================================================
# 阶段36 核心接口：动态任务池管理器
# ============================================================
class DynamicTaskPoolEngine:
    """阶段36的核心计算接口。

    实现动态任务池的完整管理，包括批量生成、刷新、
    淘汰、冻结/解冻。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 池大小限制
        - 安全任务不淘汰
        - 5轮无收益冻结
    """

    def batch_generate(
        self,
        gaps: tuple[float, float, float],
        size_config: PoolSizeConfig = None
    ) -> DynamicTaskPool:
        """批量任务生成（A36.2）。

        根据缺口三元组生成初始任务池。

        Args:
            gaps: 缺口三元组 (ΔV_safe, ΔV_gain, ΔV_eff)
            size_config: 池大小配置，默认使用 PoolSizeConfig()

        Returns:
            DynamicTaskPool 实例

        Raises:
            ValueError: 当 gaps 包含负值时

        Examples:
            >>> engine = DynamicTaskPoolEngine()
            >>> pool = engine.batch_generate((0.5, 0.3, 0.2))
            >>> len(pool.active_tasks) <= pool.size_config.max_active
            True
        """
        if size_config is None:
            size_config = PoolSizeConfig()
        for i, v in enumerate(gaps):
            if v < 0:
                raise ValueError(f"缺口第 {i} 分量必须 ≥ 0，收到 {v}")
        # 根据缺口生成占位任务
        tasks: list[PooledTask] = []
        delta_safe, delta_gain, delta_eff = gaps
        # 安全任务
        if delta_safe > 0.1:
            for _ in range(min(3, int(delta_safe * 5) + 1)):
                tasks.append(PooledTask(
                    task=SafetyTask(
                        task_type=SafetyTaskType.STATE_MONITOR,
                        target=SafetyTaskTarget.SYSTEM,
                        priority=delta_safe,
                        estimated_cost=0.2,
                        expected_benefit=delta_safe * 0.5,
                        scope="monitor_system",
                    ),
                    status=TaskStatus.ACTIVE,
                    priority=delta_safe,
                ))
        # 收益任务
        if delta_gain > 0.1:
            for _ in range(min(3, int(delta_gain * 5) + 1)):
                tasks.append(PooledTask(
                    task=GainTask(
                        task_type=GainTaskType.COVERAGE_EXPAND,
                        target_region=CandidateRegion(
                            name="default", uncertainty=0.5,
                            info_gain_estimate=0.3, exploration_cost=0.2,
                        ),
                        priority=delta_gain,
                        estimated_cost=0.3,
                        expected_benefit=delta_gain * 0.4,
                        scope="expand_coverage",
                    ),
                    status=TaskStatus.PENDING,
                    priority=delta_gain,
                ))
        # 能效任务
        if delta_eff > 0.1:
            for _ in range(min(3, int(delta_eff * 5) + 1)):
                tasks.append(PooledTask(
                    task=EfficiencyTask(
                        task_type=EfficiencyTaskType.CPU_OPTIMIZE,
                        target=frozenset({"cpu"}),
                        priority=delta_eff,
                        estimated_cost=0.3,
                        expected_benefit=delta_eff * 0.3,
                        scope="cpu_optimize",
                    ),
                    status=TaskStatus.PENDING,
                    priority=delta_eff,
                ))
        # 分配到活跃和待执行池
        active = tuple(tasks[:size_config.max_active])
        pending = tuple(tasks[size_config.max_active:size_config.max_total])
        return DynamicTaskPool(
            eff_generator=EfficiencyTaskGenerator(
                gain_generator=GainTaskGenerator(
                    safety_generator=SafetyTaskGenerator(
                        gap_assessment=GapAssessment(...),
                        tasks=(), threshold=0.3, dedup_threshold=0.8,
                        original_task_count=0, deduped_task_count=0,
                    ),
                    gain_tasks=(), candidate_regions=(), selected_regions=(),
                    threshold=0.3, dedup_threshold=0.8,
                    original_task_count=0, deduped_task_count=0,
                ),
                eff_tasks=(), compression_search_space=(),
                optimal_compression=None, threshold=0.3, dedup_threshold=0.8,
                original_task_count=0, deduped_task_count=0,
            ),
            active_tasks=active,
            pending_tasks=pending,
            completed_tasks=(),
            frozen_tasks=(),
            size_config=size_config,
            last_refresh_step=0,
            total_generated=len(tasks),
            total_evicted=0,
            total_frozen=0,
        )

    def refresh(
        self,
        pool: DynamicTaskPool,
        state: 'EnvironmentStateVector',
        current_step: int
    ) -> DynamicTaskPool:
        """任务刷新（A36.3）。

        根据最新环境状态重新评估任务优先级。

        Args:
            pool: 当前任务池
            state: 环境状态向量
            current_step: 当前时间步 >= 0

        Returns:
            刷新后的任务池

        Raises:
            ValueError: 当 current_step < 0 时

        Examples:
            >>> engine = DynamicTaskPoolEngine()
            >>> pool = engine.batch_generate((0.5, 0.3, 0.2))
            >>> refreshed = engine.refresh(pool, None, 1)
            >>> refreshed.last_refresh_step
            1
        """
        if current_step < 0:
            raise ValueError(f"当前时间步必须 ≥ 0，收到 {current_step}")
        # 简单刷新：更新所有任务的 last_refreshed 和优先级微调
        def _update(t: PooledTask) -> PooledTask:
            # 基于环境状态微调优先级（占位逻辑）
            new_priority = t.priority
            return PooledTask(
                task=t.task,
                status=t.status,
                priority=min(1.0, max(0.0, new_priority)),
                rounds_no_progress=t.rounds_no_progress,
                frozen_reason=t.frozen_reason,
                created_at=t.created_at,
                last_refreshed=current_step,
            )
        active = tuple(_update(t) for t in pool.active_tasks)
        pending = tuple(_update(t) for t in pool.pending_tasks)
        frozen = tuple(_update(t) for t in pool.frozen_tasks)
        return DynamicTaskPool(
            eff_generator=pool.eff_generator,
            active_tasks=active,
            pending_tasks=pending,
            completed_tasks=pool.completed_tasks,
            frozen_tasks=frozen,
            size_config=pool.size_config,
            last_refresh_step=current_step,
            total_generated=pool.total_generated,
            total_evicted=pool.total_evicted,
            total_frozen=pool.total_frozen,
        )

    def evict(
        self,
        pool: DynamicTaskPool,
        max_size: int = 50
    ) -> DynamicTaskPool:
        """任务淘汰（A36.4）。

        淘汰低优先级任务以释放池容量，安全任务不淘汰。

        Args:
            pool: 当前任务池
            max_size: 最大容量 >= 1

        Returns:
            淘汰后的任务池

        Raises:
            ValueError: 当 max_size < 1 时

        Examples:
            >>> engine = DynamicTaskPoolEngine()
            >>> pool = engine.batch_generate((0.5, 0.3, 0.2))
            >>> evicted = engine.evict(pool, max_size=5)
            >>> len(evicted.active_tasks) + len(evicted.pending_tasks) <= 5
            True
        """
        if max_size < 1:
            raise ValueError(f"最大容量必须 ≥ 1，收到 {max_size}")
        all_tasks = list(pool.active_tasks) + list(pool.pending_tasks)
        if len(all_tasks) <= max_size:
            return pool
        # 安全任务不淘汰
        safe_tasks = [t for t in all_tasks if t.task_type_category == "safe"]
        other_tasks = [t for t in all_tasks if t.task_type_category != "safe"]
        # 按优先级降序排列非安全任务，保留高分
        other_tasks.sort(key=lambda t: t.priority, reverse=True)
        keep_count = max_size - len(safe_tasks)
        kept_other = other_tasks[:max(0, keep_count)]
        kept = safe_tasks + kept_other
        # 重新分配活跃/待执行
        active = tuple(
            t for t in kept[:pool.size_config.max_active]
        )
        pending = tuple(
            t for t in kept[pool.size_config.max_active:pool.size_config.max_total]
        )
        evicted_count = len(all_tasks) - len(kept)
        return DynamicTaskPool(
            eff_generator=pool.eff_generator,
            active_tasks=active,
            pending_tasks=pending,
            completed_tasks=pool.completed_tasks,
            frozen_tasks=pool.frozen_tasks,
            size_config=pool.size_config,
            last_refresh_step=pool.last_refresh_step,
            total_generated=pool.total_generated,
            total_evicted=pool.total_evicted + evicted_count,
            total_frozen=pool.total_frozen,
        )

    def freeze(
        self,
        pool: DynamicTaskPool,
        task: PooledTask,
        reason: str = ""
    ) -> DynamicTaskPool:
        """冻结任务（A36.5）。

        Args:
            pool: 当前任务池
            task: 要冻结的任务
            reason: 冻结原因

        Returns:
            冻结后的任务池

        Examples:
            >>> engine = DynamicTaskPoolEngine()
            >>> pool = engine.batch_generate((0.5, 0.3, 0.2))
            >>> task = pool.active_tasks[0]
            >>> frozen_pool = engine.freeze(pool, task, reason="no_progress")
            >>> any(t.status == TaskStatus.FROZEN for t in frozen_pool.frozen_tasks)
            True
        """
        # 从活跃/待执行池中移除
        active = tuple(t for t in pool.active_tasks if t is not task)
        pending = tuple(t for t in pool.pending_tasks if t is not task)
        frozen_task = PooledTask(
            task=task.task,
            status=TaskStatus.FROZEN,
            priority=task.priority,
            rounds_no_progress=task.rounds_no_progress,
            frozen_reason=reason,
            created_at=task.created_at,
            last_refreshed=task.last_refreshed,
        )
        # 若frozen_tasks超过上限，淘汰最早的
        frozen = list(pool.frozen_tasks) + [frozen_task]
        if len(frozen) > pool.size_config.max_frozen:
            frozen.sort(key=lambda t: t.last_refreshed)
            frozen = frozen[-pool.size_config.max_frozen:]
        return DynamicTaskPool(
            eff_generator=pool.eff_generator,
            active_tasks=active,
            pending_tasks=pending,
            completed_tasks=pool.completed_tasks,
            frozen_tasks=tuple(frozen),
            size_config=pool.size_config,
            last_refresh_step=pool.last_refresh_step,
            total_generated=pool.total_generated,
            total_evicted=pool.total_evicted,
            total_frozen=pool.total_frozen + 1,
        )

    def unfreeze(
        self,
        pool: DynamicTaskPool,
        task: PooledTask
    ) -> DynamicTaskPool:
        """解冻任务（A36.5）。

        将冻结任务移回待执行池。

        Args:
            pool: 当前任务池
            task: 要解冻的任务

        Returns:
            解冻后的任务池

        Examples:
            >>> engine = DynamicTaskPoolEngine()
            >>> pool = engine.batch_generate((0.5, 0.3, 0.2))
            >>> task = pool.active_tasks[0]
            >>> p2 = engine.freeze(pool, task)
            >>> p3 = engine.unfreeze(p2, p2.frozen_tasks[0])
            >>> len(p3.pending_tasks) > len(p2.pending_tasks)
            True
        """
        frozen = tuple(t for t in pool.frozen_tasks if t is not task)
        unfrozen_task = PooledTask(
            task=task.task,
            status=TaskStatus.PENDING,
            priority=task.priority,
            rounds_no_progress=0,
            frozen_reason="",
            created_at=task.created_at,
            last_refreshed=task.last_refreshed,
        )
        pending = list(pool.pending_tasks) + [unfrozen_task]
        # 若pending超过上限，淘汰最低优先级
        if len(pending) > pool.size_config.max_pending:
            pending.sort(key=lambda t: t.priority, reverse=True)
            pending = pending[:pool.size_config.max_pending]
        return DynamicTaskPool(
            eff_generator=pool.eff_generator,
            active_tasks=pool.active_tasks,
            pending_tasks=tuple(pending),
            completed_tasks=pool.completed_tasks,
            frozen_tasks=frozen,
            size_config=pool.size_config,
            last_refresh_step=pool.last_refresh_step,
            total_generated=pool.total_generated,
            total_evicted=pool.total_evicted,
            total_frozen=max(0, pool.total_frozen - 1),
        )

    def check_freeze_conditions(
        self,
        pool: DynamicTaskPool
    ) -> list[PooledTask]:
        """检查所有任务的冻结条件（A36.5）。

        连续5轮无收益的任务进入冻结候选。

        Args:
            pool: 当前任务池

        Returns:
            需要冻结的任务列表

        Examples:
            >>> engine = DynamicTaskPoolEngine()
            >>> pool = engine.batch_generate((0.5, 0.3, 0.2))
            >>> to_freeze = engine.check_freeze_conditions(pool)
            >>> len(to_freeze) >= 0
            True
        """
        to_freeze: list[PooledTask] = []
        for t in pool.active_tasks:
            if t.should_freeze:
                to_freeze.append(t)
        return to_freeze

    def check_unfreeze_conditions(
        self,
        pool: DynamicTaskPool,
        state: 'EnvironmentStateVector'
    ) -> list[PooledTask]:
        """检查冻结任务的解冻条件（A36.5）。

        当环境状态显著变化时，允许解冻相关任务。

        Args:
            pool: 当前任务池
            state: 当前环境状态

        Returns:
            可以解冻的任务列表

        Examples:
            >>> engine = DynamicTaskPoolEngine()
            >>> pool = engine.batch_generate((0.5, 0.3, 0.2))
            >>> to_unfreeze = engine.check_unfreeze_conditions(pool, None)
            >>> len(to_unfreeze) >= 0
            True
        """
        # 简化：所有冻结任务都允许解冻（实际应根据环境状态匹配）
        return list(pool.frozen_tasks)
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `EfficiencyTaskGenerator` | ← 阶段35 | 作为动态任务池的任务生成组合字段 |
| **输入** | A32.1（缺口评估函数） | ← 阶段32 | 缺口三元组驱动批量任务生成 |
| **输入** | A31.1（环境状态向量） | ← 阶段31 | 任务刷新的环境状态输入 |
| **输出** | `DynamicTaskPool` | → 阶段37 | 任务池作为综合得分计算的输入 |
| **输出** | A36.1（任务池数据结构） | → 后续阶段 | 四池结构为任务调度提供统一管理 |
| **输出** | A36.5（冻结/解冻机制） | → 阶段85 | 5轮无收益冻结规则的实现 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 池操作计算 | `jax.lax.cond`实现冻结/解冻条件判断（A36.5）；`jax.numpy.argsort`实现优先级排序（A36.3） |
| heapq | 优先级队列 | Python标准库的堆队列，支撑pending_tasks的高效优先级排序和top_k选择（A36.2, A36.4） |
| collections.deque | 完成任务队列 | 双端队列支撑completed_tasks的FIFO管理和容量限制（A36.6-clean） |
| asyncio | 异步任务调度 | Python异步IO框架，支撑active_tasks的并发执行和上下文切换（A36.1） |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V36.1 | 四池互斥性 | 每个任务恰好属于一个子池 | 任务ID唯一性检查 |
| V36.2 | 四池完备性 | 所有任务都属于某个子池 | 任务计数验证 |
| V36.3 | 活跃任务上限 | $|\text{active}| \leq 10$ | 边界值测试 |
| V36.4 | 待执行任务上限 | $|\text{pending}| \leq 40$ | 边界值测试 |
| V36.5 | 批量生成完整性 | 三类任务均被纳入池中 | 生成后计数验证 |
| V36.6 | 跨类型去重 | 相似度>0.8的跨类型任务被合并 | 合成跨类型相似任务测试 |
| V36.7 | 刷新优先级更新 | 刷新后优先级反映最新缺口 | 缺口变化前后对比测试 |
| V36.8 | 刷新触发条件 | 定期/场景切换/缺口突变均触发刷新 | 触发条件模拟测试 |
| V36.9 | 安全任务不淘汰 | 淘汰后安全任务仍在池中 | 淘汰压力测试 |
| V36.10 | 新任务保护期 | 5步内新任务不被淘汰 | 保护期测试 |
| V36.11 | 5轮无收益冻结 | 连续5轮无收益的任务被冻结 | 冻结条件模拟 |
| V36.12 | 解冻条件 | 环境变化/资源释放/依赖完成触发解冻 | 解冻条件模拟 |
| V36.13 | 场景缩放 | urgent场景max_active=5，stable场景=15 | 场景切换测试 |
| V36.14 | DynamicTaskPool组合 | `DynamicTaskPool.eff_generator` 类型为 `EfficiencyTaskGenerator` | 类型检查 |
| V36.15 | 池利用率计算 | utilization = (active + pending) / max_total | 数值验证 |
| V36.16 | 非法输入拒绝 | 超限任务数、负优先级等抛出 `ValueError` | 异常测试 |

---

> **阶段36完** | 阶段36建立了动态任务池的完整管理体系：四池数据结构（active/pending/completed/frozen）统一管理三类15种任务，批量生成将缺口一次性转化为任务池，刷新机制根据环境变化动态调整优先级，淘汰机制在容量超限时移除低分任务，冻结/解冻机制处理长期无进展的任务。下一步：阶段37将定义任务综合得分公式，为任务的选择和调度提供精确的评分依据。
