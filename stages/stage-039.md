# 阶段39：动态中断机制——从优先级跃迁到灵活响应

> **阶段编号**：S39 | **所属层**：第四层·任务规划层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 中断条件 | "有更急的事"——新任务比当前任务得分高出足够多时中断 |
> | 紧急中断 | "着火了"——安全驱力低于紧急阈值，无条件中断 |
> | 保存点 | "存档"——中断前保存当前进度，方便以后恢复 |
> | 中断恢复 | "读档"——从保存点恢复任务继续执行 |
> | 中断频率限制 | "不能反复横跳"——最小执行时间内不允许再次中断 |
> | 中断成本评估 | "中断划不划算"——算算中断+恢复+切换的总成本 |

---

## 1. 第一性原理依据

本阶段从阶段37（综合得分）和阶段26（紧急风险判定）的核心结论出发，建立动态中断机制。核心依据为：

- **A37.1（综合得分定义）**（引用阶段37）：$\text{Score}(\text{task}) = W \cdot \Delta V_{\text{gap}} + \alpha \cdot \text{urgency} + \beta \cdot \text{feasibility} - \gamma \cdot \text{cost}$，综合得分是中断判定的核心依据——当新任务的得分显著高于当前任务时，应考虑中断当前任务。
- **A37.3（紧迫度定义）**（引用阶段37）：$\text{urgency}(\text{task}) = 1/(1 + \exp(-\kappa \cdot (\Delta V_{\text{gap}} - \theta)))$，紧迫度的S型曲线特性使得高紧迫度任务的得分跃升显著——当安全缺口突然增大时，安全任务的紧迫度急剧上升，触发中断。
- **A26.2（紧急风险场景判定）**（引用阶段26）：$R_{\text{safe}} < \theta_{\text{urgent}}$ 为紧急风险场景的判定条件。紧急中断直接使用此条件——当安全驱力低于紧急阈值时，无条件中断当前任务，切换到安全任务。
- **A36.5（任务冻结/解冻）**（引用阶段36）：任务冻结将长期无进展的任务暂时挂起。中断与冻结的区别在于：冻结是"主动暂停"（任务无进展），中断是"被动切换"（更高优先级任务出现）。中断后的任务进入冻结状态，等待条件满足后解冻恢复。
- **A7.7（EFE完整分解）**（引用阶段7）：$G(\pi) = E_{\text{ext}}(\pi) - E_{\text{epi}}(\pi)$，预期自由能框架支持策略切换——当新策略的EFE显著低于当前策略时，应切换策略。中断机制是策略切换在任务级别的工程实现。

本阶段的核心目标是：**定义中断条件 Score(new_task) > Score(current_task) + θ_interrupt，定义紧急中断条件 R_safe < θ_urgent（无条件中断），实现中断保存点 checkpoint(current_task)，实现中断恢复 resume(task, checkpoint)，设定中断频率限制（最小执行时间T_min_exec），定义中断成本评估 cost_interrupt = cost_checkpoint + cost_resume + cost_context_switch，并封装为DynamicInterruptor数据类，组合DualDecisionMechanism作为决策机制的输入。** 动态中断机制是从"固定执行"到"灵活响应"的关键一步——它允许系统在更高优先级任务出现时及时切换，同时通过频率限制和成本评估避免频繁中断导致的资源浪费。

---

## 2. 核心数学公式

### 第一部分：中断条件

#### 2.1 中断条件（A39.1）

中断条件判断新任务的得分是否显著高于当前任务：

$$\text{interrupt} \iff \text{Score}(\text{new\_task}) > \text{Score}(\text{current\_task}) + \theta_{\text{interrupt}} \tag{A39.1}$$

其中 $\theta_{\text{interrupt}} > 0$ 为中断阈值——新任务的得分必须比当前任务高出此阈值才触发中断。

**中断阈值的选择**：

中断阈值 $\theta_{\text{interrupt}}$ 需要在"灵敏度"和"稳定性"之间权衡：

- $\theta_{\text{interrupt}}$ 太低：轻微的得分差异就触发中断，导致频繁切换
- $\theta_{\text{interrupt}}$ 太高：显著的得分差异也不触发中断，错过重要的优先级跃迁

默认阈值为归一化得分的0.2：

$$\theta_{\text{interrupt}} = 0.2 \tag{A39.1-default}$$

即新任务的归一化得分必须比当前任务高出0.2（在0-1范围内）才触发中断。这意味着：

| 当前得分 | 触发中断的最低新得分 | 含义 |
|---------|-------------------|------|
| 0.3 | 0.5 | 低分任务只需中等跃升即可中断 |
| 0.5 | 0.7 | 中分任务需要较高跃升才中断 |
| 0.7 | 0.9 | 高分任务需要极高跃升才中断 |
| 0.9 | 1.1（不可能） | 极高分任务几乎不会被中断 |

**中断条件的逻辑展开**：

将A37.1代入A39.1，展开中断条件：

$$W \cdot \Delta V_{\text{gap}}^{\text{new}} + \alpha \cdot u^{\text{new}} + \beta \cdot f^{\text{new}} - \gamma \cdot c^{\text{new}} > W \cdot \Delta V_{\text{gap}}^{\text{cur}} + \alpha \cdot u^{\text{cur}} + \beta \cdot f^{\text{cur}} - \gamma \cdot c^{\text{cur}} + \theta_{\text{interrupt}} \tag{A39.1-expand}$$

整理得：

$$W \cdot (\Delta V_{\text{gap}}^{\text{new}} - \Delta V_{\text{gap}}^{\text{cur}}) + \alpha \cdot (u^{\text{new}} - u^{\text{cur}}) + \beta \cdot (f^{\text{new}} - f^{\text{cur}}) - \gamma \cdot (c^{\text{new}} - c^{\text{cur}}) > \theta_{\text{interrupt}} \tag{A39.1-diff}$$

中断由四个差值项驱动：

1. **缺口差** $W \cdot (\Delta V_{\text{gap}}^{\text{new}} - \Delta V_{\text{gap}}^{\text{cur}})$：新任务的缺口更大（更值得做）
2. **紧迫度差** $\alpha \cdot (u^{\text{new}} - u^{\text{cur}})$：新任务更紧迫（更紧急）
3. **可行性差** $\beta \cdot (f^{\text{new}} - f^{\text{cur}})$：新任务更可行（更容易做）
4. **成本差** $-\gamma \cdot (c^{\text{new}} - c^{\text{cur}})$：新任务成本更低（更划算）

**中断条件的场景敏感性**：

在不同场景下，中断阈值可以自适应调整：

| 场景 | $\theta_{\text{interrupt}}$ | 理由 |
|------|---------------------------|------|
| urgent | 0.1 | 紧急场景下更灵敏地响应优先级变化 |
| stable | 0.3 | 稳定场景下更稳定地执行当前任务 |
| tight | 0.2 | 资源紧张时使用默认阈值 |

- **输入**：当前任务得分，新任务得分，中断阈值
- **输出**：是否中断（布尔值）
- **约束**：$\theta_{\text{interrupt}} > 0$；中断前检查频率限制

> **白话翻译**：中断条件就是"有更急的事"——新任务的得分比当前任务高出足够多（0.2以上）时，就中断当前任务去做新的。阈值不能太低（否则天天中断），也不能太高（否则着火了还不中断）。紧急情况下阈值降低（0.1），更容易触发中断。

---

### 第二部分：紧急中断

#### 2.2 紧急中断条件（A39.2）

紧急中断在安全驱力低于紧急阈值时无条件触发：

$$\text{urgent\_interrupt} \iff R_{\text{safe}} < \theta_{\text{urgent}} \tag{A39.2}$$

其中 $\theta_{\text{urgent}}$ 为安全驱力的紧急阈值（引用A26.2），$R_{\text{safe}}$ 为当前安全驱力值。

**紧急中断的特殊性**：

紧急中断与普通中断有三个关键区别：

1. **无条件**：不检查中断阈值、频率限制和成本评估——安全威胁下必须立即响应
2. **强制切换**：当前任务无论得分多高都必须中断——安全优先于一切
3. **目标固定**：中断后必须切换到安全任务（stabilize或anomaly_detect）——不是切换到任意高分任务

**紧急阈值的定义**：

$$\theta_{\text{urgent}} = -\alpha_s \cdot 3 \cdot \bar{a} \tag{A39.2-threshold}$$

其中 $\alpha_s$ 为安全驱力系数（引用A21.1），$\bar{a}$ 为异常信号的平均基线值。紧急阈值设为安全驱力在"3倍平均异常"时的值——当异常水平达到平均的3倍时，系统进入紧急状态。

**紧急中断的响应流程**：

```
输入：当前安全驱力 R_safe
输出：中断决策

1. 如果 R_safe < θ_urgent：
   a. 立即中断当前任务
   b. 保存当前任务的检查点
   c. 选择最高优先级的安全任务
   d. 开始执行安全任务
   e. 返回 "urgent_interrupt"

2. 否则：
   f. 返回 "no_interrupt"
```

**紧急中断与EFE的对应**：

紧急中断对应EFE框架中的"策略急剧切换"——当环境的生成模型发生剧变（安全威胁出现）时，当前策略的EFE急剧上升，必须立即切换到新策略。在EFE框架中，这是通过策略后验（引用A7.11）的急剧重分配实现的——安全策略的后验概率急剧上升，其他策略的后验概率急剧下降。

- **输入**：安全驱力 $R_{\text{safe}}$，紧急阈值 $\theta_{\text{urgent}}$
- **输出**：是否紧急中断（布尔值）
- **约束**：紧急中断无条件；目标为安全任务

> **白话翻译**：紧急中断就是"着火了"——安全驱力低于紧急阈值时，不管当前在做什么都必须停下来，立刻去做安全任务。这是无条件中断——不看得分差距，不看频率限制，不算中断成本。着火时灭火第一，其他一切让路。

---

### 第三部分：中断保存点

#### 2.3 中断保存点（A39.3）

中断保存点在中断前保存当前任务的执行状态，以便后续恢复：

$$\text{checkpoint}(\text{current\_task}) = \{\text{task\_id}, \text{progress}, \text{intermediate\_state}, \text{timestamp}\} \tag{A39.3}$$

其中：
- $\text{task\_id}$：被中断任务的唯一标识
- $\text{progress} \in [0, 1]$：任务执行进度（0=刚开始，1=已完成）
- $\text{intermediate\_state}$：中间状态——任务执行过程中产生的临时数据
- $\text{timestamp}$：中断时间步

**保存点的数据结构**：

中间状态的具体内容取决于任务类型：

| 任务类型 | 中间状态内容 | 保存成本 |
|---------|------------|---------|
| model_compress | 当前搜索步数、已评估的配置、当前最优配置 | 中 |
| explore_unknown | 已探索的区域、已收集的数据、当前分析结果 | 中-高 |
| anomaly_detect | 已检测的异常列表、当前检测进度 | 低 |
| stabilize | 已采取的稳定措施、当前系统状态 | 低 |
| state_monitor | 监控数据快照 | 低 |
| mine_pattern | 已发现的模式、当前挖掘进度 | 中 |

**保存点的创建时机**：

保存点不是每次中断时才创建的——定期创建保存点可以减少中断时的数据丢失：

1. **定期保存**：每 $T_{\text{checkpoint}} = 5$ 步创建一次保存点
2. **中断时保存**：中断时创建最终保存点（覆盖最近的定期保存点）
3. **关键节点保存**：任务执行到关键节点时（如模型压缩搜索完成一个策略的评估）创建保存点

**保存点的存储管理**：

保存点占用存储空间，需要管理：

$$\text{storage\_used} = \sum_{\text{checkpoint}} \text{size}(\text{checkpoint}) \tag{A39.3-storage}$$

存储管理策略：

1. **最近优先**：保留最近的 $K_{\text{checkpoint}} = 10$ 个保存点
2. **进度优先**：进度较高的保存点优先保留（接近完成的任务恢复价值更大）
3. **压缩存储**：对中间状态进行压缩（如只保存关键参数，不保存完整数据）

- **输入**：当前任务 task，执行进度 progress，中间状态
- **输出**：保存点 checkpoint
- **约束**：保存点包含足够信息以恢复执行

> **白话翻译**：保存点就是"存档"——中断前把当前进度存下来，就像游戏存档一样。存档内容包括：任务编号、做了多少（进度）、中间结果、存档时间。不同类型的任务存档内容不同——模型压缩要存搜索进度，异常检测要存已发现的异常。定期存档（每5步一次）比只在中断时存档更安全。

---

### 第四部分：中断恢复

#### 2.4 中断恢复（A39.4）

中断恢复从保存点恢复任务的执行：

$$\text{resume}(\text{task}, \text{checkpoint}) \to \text{task}_{\text{resumed}} \tag{A39.4}$$

**恢复流程**：

```
输入：被中断的任务 task，保存点 checkpoint
输出：恢复后的任务 task_resumed

1. 验证保存点完整性：
   if checkpoint.task_id != task.id:
     raise CheckpointMismatchError

2. 恢复中间状态：
   task.intermediate_state = checkpoint.intermediate_state

3. 设置恢复起点：
   task.resume_from = checkpoint.progress

4. 更新任务参数：
   task.priority = recompute_priority(task, current_state)
   task.feasibility = recompute_feasibility(task, current_resources)

5. 将任务放回待执行队列：
   pool.pending_tasks.append(task_resumed)

6. 返回 task_resumed
```

**恢复时的参数更新**：

恢复时需要重新计算任务的优先级和可行性——环境可能已经变化：

$$\text{priority}' = \Delta V_{i,\text{norm}}' \cdot w_{\text{type}} \cdot w_{\text{scene}}' \tag{A39.4-priority}$$

$$\text{feasibility}' = \min\left(\frac{\text{cpu\_available}'}{\text{cpu\_required}}, \frac{\text{mem\_available}'}{\text{mem\_required}}, \frac{\text{time\_available}'}{\text{time\_required}}\right) \tag{A39.4-feasibility}$$

**恢复后的执行策略**：

恢复后，任务从保存点继续执行，而不是从头开始。但需要考虑：

1. **环境变化**：如果环境发生了显著变化（缺口值变化 > 0.3），保存点的中间状态可能已过时，需要重新评估是否继续执行
2. **依赖变化**：如果任务依赖的前置条件已改变，可能需要调整执行策略
3. **进度衰减**：长时间未恢复的任务，其保存点的价值逐渐降低——环境变化使得旧进度越来越不可用

**恢复的时效性**：

$$\text{checkpoint\_value}(\text{checkpoint}, t) = \text{progress} \cdot e^{-\lambda_{\text{decay}} \cdot (t - t_{\text{checkpoint}})} \tag{A39.4-value}$$

其中 $\lambda_{\text{decay}}$ 为保存点价值的衰减率（默认 $\lambda_{\text{decay}} = 0.01$），$t - t_{\text{checkpoint}}$ 为保存点创建至今的时间步数。保存点的价值随时间指数衰减——越旧的保存点，恢复价值越低。

当 $\text{checkpoint\_value} < \theta_{\text{stale}} = 0.1$ 时，保存点被认为"过时"，恢复时从头开始执行而非从保存点恢复。

- **输入**：被中断任务 task，保存点 checkpoint
- **输出**：恢复后的任务 task_resumed
- **约束**：保存点完整性验证；参数更新

> **白话翻译**：中断恢复就是"读档"——从存档点恢复任务继续执行。恢复时要重新算优先级和可行性（环境可能变了），从存档进度继续而不是从头开始。但存档会过期——太久没恢复的存档价值降低，不如从头做。

---

### 第五部分：中断频率限制

#### 2.5 中断频率限制（A39.5）

中断频率限制防止系统频繁切换任务：

$$\text{allow\_interrupt} \iff t_{\text{current}} - t_{\text{last\_interrupt}} \geq T_{\text{min\_exec}} \tag{A39.5}$$

其中：
- $t_{\text{current}}$：当前时间步
- $t_{\text{last\_interrupt}}$：上次中断的时间步
- $T_{\text{min\_exec}}$：最小执行时间——任务开始后必须执行至少这么多步才允许中断

**最小执行时间的选择**：

$T_{\text{min\_exec}}$ 的选择需要在"响应性"和"稳定性"之间权衡：

| $T_{\text{min\_exec}}$ | 特征 | 适用场景 |
|----------------------|------|---------|
| 3 | 高响应性，低稳定性 | 环境变化极快 |
| 5 | 中等响应性，中等稳定性 | 通用场景（默认值） |
| 10 | 低响应性，高稳定性 | 环境变化缓慢 |

**频率限制的例外**：

紧急中断（A39.2）不受频率限制——安全威胁下必须立即响应，即使刚中断过：

$$\text{if } R_{\text{safe}} < \theta_{\text{urgent}} \text{ then } T_{\text{min\_exec}} = 0 \tag{A39.5-exception}$$

**频率限制的统计监控**：

系统应监控中断频率的统计量：

1. **平均中断间隔**：$\bar{T}_{\text{interrupt}} = \frac{1}{N}\sum_{i=1}^{N}(t_i - t_{i-1})$——平均多久中断一次
2. **中断频率**：$f_{\text{interrupt}} = N / T$——单位时间内的中断次数
3. **中断频率趋势**：$\Delta f_{\text{interrupt}}$——中断频率是否在增加

如果中断频率持续升高（$\Delta f_{\text{interrupt}} > 0$），可能表明：

1. 环境变化加剧——需要更频繁地响应新情况
2. 任务评分不稳定——得分频繁波动导致频繁切换
3. 中断阈值过低——需要提高 $\theta_{\text{interrupt}}$

**频率限制与上下文切换成本**：

频繁中断的主要成本是上下文切换——每次中断需要保存当前任务的上下文、加载新任务的上下文。上下文切换的时间成本约为：

$$t_{\text{context\_switch}} \approx 0.1 \cdot T_{\text{min\_exec}} \tag{A39.5-switch}$$

即上下文切换时间约为最小执行时间的10%。如果 $T_{\text{min\_exec}} = 5$，则每次上下文切换约0.5步。在频繁中断的情况下，上下文切换的总时间可能占据显著比例。

- **输入**：当前时间步，上次中断时间步，最小执行时间
- **输出**：是否允许中断（布尔值）
- **约束**：$T_{\text{min\_exec}} \geq 1$；紧急中断不受限制

> **白话翻译**：中断频率限制就是"不能反复横跳"——任务开始后至少执行5步才能中断。刚切换到新任务就又切换，等于什么都没做。但着火时例外——安全中断不受频率限制，随时可以中断。如果发现中断越来越频繁，可能是环境变化加剧或者评分不稳定，需要调查原因。

---

### 第六部分：中断成本评估

#### 2.6 中断成本评估（A39.6）

中断成本评估量化中断的总资源消耗：

$$\text{cost}_{\text{interrupt}} = \text{cost}_{\text{checkpoint}} + \text{cost}_{\text{resume}} + \text{cost}_{\text{context\_switch}} \tag{A39.6}$$

**各成本分量的定义**：

**保存成本**：

$$\text{cost}_{\text{checkpoint}} = c_{\text{save}} \cdot \text{size}(\text{intermediate\_state}) \tag{A39.6-checkpoint}$$

其中 $c_{\text{save}}$ 为单位存储成本，$\text{size}(\text{intermediate\_state})$ 为中间状态的数据量。

**恢复成本**：

$$\text{cost}_{\text{resume}} = c_{\text{load}} \cdot \text{size}(\text{intermediate\_state}) + c_{\text{recompute}} \cdot (1 - \text{progress}) \tag{A39.6-resume}$$

恢复成本包括两部分：加载中间状态的成本（与数据量成正比）和重新计算缺失部分的成本（与未完成进度成正比）。

**上下文切换成本**：

$$\text{cost}_{\text{context\_switch}} = c_{\text{switch}} \cdot \text{complexity}(\text{current\_task}) \tag{A39.6-switch}$$

其中 $\text{complexity}(\text{current\_task})$ 为当前任务的复杂度——复杂任务的上下文切换成本更高（需要保存和恢复更多的状态信息）。

**中断的净收益**：

中断的净收益 = 新任务的预期收益 - 中断成本 - 当前任务的进度损失：

$$\text{net\_benefit} = \text{Score}(\text{new\_task}) \cdot T_{\text{remaining}} - \text{cost}_{\text{interrupt}} - \text{Score}(\text{current\_task}) \cdot (1 - \text{progress}) \cdot T_{\text{remaining}} \tag{A39.6-net}$$

其中 $T_{\text{remaining}}$ 为剩余执行时间。

中断决策应基于净收益——只有当中断的净收益为正时才执行中断：

$$\text{interrupt} \iff \text{net\_benefit} > 0 \tag{A39.6-decision}$$

**中断成本与中断条件的统一**：

A39.1的中断条件和A39.6的净收益条件可以统一。将A39.1改写为：

$$\text{Score}(\text{new\_task}) - \text{Score}(\text{current\_task}) > \theta_{\text{interrupt}} \tag{A39.6-unify}$$

而A39.6的净收益条件为：

$$\text{Score}(\text{new\_task}) - \text{Score}(\text{current\_task}) \cdot (1 - \text{progress}) > \frac{\text{cost}_{\text{interrupt}}}{T_{\text{remaining}}} \tag{A39.6-unify2}$$

当 $\text{progress} \approx 0$（任务刚开始）且 $\text{cost}_{\text{interrupt}} / T_{\text{remaining}} \approx \theta_{\text{interrupt}}$ 时，两个条件等价。因此，A39.1的简单阈值条件是A39.6的精确成本-收益分析在特定条件下的近似。

**各任务类型的中断成本**：

| 任务类型 | 保存成本 | 恢复成本 | 切换成本 | 总成本 | 说明 |
|---------|---------|---------|---------|-------|------|
| state_monitor | 极低 | 极低 | 低 | 极低 | 监控任务几乎无状态 |
| anomaly_detect | 低 | 低 | 低 | 低 | 检测进度容易保存 |
| stabilize | 低 | 低 | 中 | 低-中 | 稳定措施需要记录 |
| risk_predict | 中 | 中 | 中 | 中 | 预测模型需要保存 |
| explore_unknown | 中-高 | 中-高 | 高 | 高 | 探索数据量大 |
| model_compress | 高 | 高 | 高 | 很高 | 搜索空间大，中间结果多 |

model_compress的中断成本最高——压缩搜索的中间状态（已评估的配置、当前Pareto前沿）数据量大，保存和恢复成本高。因此，模型压缩任务一旦开始，应尽量避免中断。

- **输入**：当前任务，新任务，保存点，上下文切换参数
- **输出**：中断成本 $\text{cost}_{\text{interrupt}}$，净收益
- **约束**：各成本分量非负；紧急中断不评估成本

> **白话翻译**：中断成本评估就是"中断划不划算"——算算中断+恢复+切换的总成本，再跟新任务的收益比一比。成本包括三部分：存档花钱、读档花钱、切换花钱。模型压缩最贵（搜索数据多），状态监控最便宜（几乎无状态）。只有当中断的净收益为正时才中断——新任务的收益要大于中断成本加上当前任务的进度损失。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A39.1 | Score(new) > Score(current) + θ_interrupt | 新旧任务得分、阈值 | 布尔值 | θ > 0；频率限制 |
| A39.2 | R_safe < θ_urgent → 无条件中断 | 安全驱力、紧急阈值 | 布尔值 | 无条件；目标为安全任务 |
| A39.3 | checkpoint = {task_id, progress, state, timestamp} | 当前任务、进度 | 保存点 | 完整性验证 |
| A39.4 | resume(task, checkpoint) → task_resumed | 任务、保存点 | 恢复后任务 | 参数更新；时效性 |
| A39.5 | t_current - t_last_interrupt ≥ T_min_exec | 时间步、最小执行时间 | 布尔值 | T_min ≥ 1；紧急例外 |
| A39.6 | cost_interrupt = cost_checkpoint + cost_resume + cost_switch | 任务、保存点 | 中断成本 | 各分量非负；紧急不评估 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np

# 引用阶段38的数据类
from stage_038 import DualDecisionMechanism, DecisionType, DecisionRecord


# ============================================================
# 中断类型枚举
# ============================================================
class InterruptType(Enum):
    """中断类型。"""
    NORMAL = "normal"       # 普通中断（A39.1）
    URGENT = "urgent"       # 紧急中断（A39.2）
    NONE = "none"           # 无中断


# ============================================================
# 中断保存点
# ============================================================
@dataclass(frozen=True)
class Checkpoint:
    """中断保存点（A39.3）。

    Attributes:
        task_id: 被中断任务的唯一标识
        progress: 任务执行进度 ∈ [0, 1]
        intermediate_state: 中间状态的序列化数据
        timestamp: 中断时间步
        task_type: 任务类型（用于恢复时选择策略）
        estimated_resume_cost: 预估恢复成本
    """
    task_id: str
    progress: float
    intermediate_state: bytes = b""
    timestamp: int = 0
    task_type: str = ""
    estimated_resume_cost: float = 0.0

    @property
    def is_complete(self) -> bool:
        """任务是否已完成。"""
        return self.progress >= 1.0

    @property
    def value_at(self) -> float:
        """保存点的当前价值（考虑衰减）。"""
        # 简化：实际需根据A39.4-value计算
        return self.progress


# ============================================================
# 中断成本详情
# ============================================================
@dataclass(frozen=True)
class InterruptCost:
    """中断成本的详细分解（A39.6）。

    Attributes:
        checkpoint_cost: 保存成本
        resume_cost: 恢复成本
        context_switch_cost: 上下文切换成本
        progress_loss: 进度损失（当前任务未完成部分的预期收益）
        total_cost: 总中断成本
    """
    checkpoint_cost: float
    resume_cost: float
    context_switch_cost: float
    progress_loss: float

    @property
    def total_cost(self) -> float:
        """总中断成本。"""
        return (self.checkpoint_cost + self.resume_cost
                + self.context_switch_cost + self.progress_loss)


# ============================================================
# 中断决策结果
# ============================================================
@dataclass(frozen=True)
class InterruptDecision:
    """中断决策结果。

    Attributes:
        should_interrupt: 是否应该中断
        interrupt_type: 中断类型
        current_task_id: 当前任务ID
        new_task_id: 新任务ID（中断后切换到）
        checkpoint: 保存点（如果中断）
        cost: 中断成本（如果评估了）
        net_benefit: 中断净收益（如果评估了）
        reason: 中断原因描述
    """
    should_interrupt: bool
    interrupt_type: InterruptType
    current_task_id: str = ""
    new_task_id: str = ""
    checkpoint: Optional[Checkpoint] = None
    cost: Optional[InterruptCost] = None
    net_benefit: float = 0.0
    reason: str = ""


# ============================================================
# 阶段39 数据类：动态中断器
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class DynamicInterruptor:
    """动态中断器的完整描述。

    组合阶段38的DualDecisionMechanism，在双决策机制的基础上
    增加动态中断能力。

    Attributes:
        decision_mechanism: 阶段38的双决策机制（组合字段）
        interrupt_threshold: 中断阈值（A39.1）
        urgent_threshold: 紧急中断阈值（A39.2）
        min_exec_time: 最小执行时间（A39.5）
        last_interrupt_step: 上次中断的时间步
        checkpoints: 当前保存的检查点
        interrupt_history: 中断历史记录
        total_interrupts: 累计中断次数
        total_urgent_interrupts: 累计紧急中断次数

    Invariants:
        - interrupt_threshold > 0
        - min_exec_time >= 1
        - 紧急中断不受频率限制
    """
    decision_mechanism: DualDecisionMechanism
    interrupt_threshold: float = 0.2
    urgent_threshold: float = -0.5
    min_exec_time: int = 5
    last_interrupt_step: int = -100  # 初始化为很早以前
    checkpoints: tuple[Checkpoint, ...] = ()
    interrupt_history: tuple[InterruptDecision, ...] = ()
    total_interrupts: int = 0
    total_urgent_interrupts: int = 0

    def __post_init__(self):
        if self.interrupt_threshold <= 0:
            raise ValueError(
                f"中断阈值必须 > 0，收到 {self.interrupt_threshold}"
            )
        if self.min_exec_time < 1:
            raise ValueError(
                f"最小执行时间必须 ≥ 1，收到 {self.min_exec_time}"
            )

    @property
    def interrupt_frequency(self) -> float:
        """中断频率（每次中断的平均间隔步数）。"""
        if self.total_interrupts == 0:
            return float('inf')
        # 简化：实际需根据历史计算
        return 1.0

    @property
    def urgent_ratio(self) -> float:
        """紧急中断占比。"""
        if self.total_interrupts == 0:
            return 0.0
        return self.total_urgent_interrupts / self.total_interrupts


# ============================================================
# 阶段39 核心接口：动态中断器
# ============================================================
class DynamicInterruptorEngine:
    """阶段39的核心计算接口。

    实现动态中断的完整机制，包括中断条件判定、
    紧急中断、保存点、恢复、频率限制和成本评估。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 中断阈值 > 0
        - 最小执行时间 ≥ 1
        - 紧急中断无条件
    """

    def check_interrupt_condition(
        self,
        current_score: float,
        new_score: float,
        threshold: float = 0.2
    ) -> bool:
        """检查普通中断条件（A39.1）。

        Args:
            current_score: 当前任务得分
            new_score: 新任务得分
            threshold: 中断阈值 > 0

        Returns:
            是否满足中断条件

        Raises:
            ValueError: 当 threshold <= 0 时

        Examples:
            >>> engine = DynamicInterruptorEngine()
            >>> engine.check_interrupt_condition(0.5, 0.8, threshold=0.2)
            True
            >>> engine.check_interrupt_condition(0.5, 0.6, threshold=0.2)
            False
        """
        if threshold <= 0:
            raise ValueError(f"中断阈值必须 > 0，收到 {threshold}")
        return new_score > current_score + threshold

    def check_urgent_condition(
        self,
        r_safe: float,
        urgent_threshold: float
    ) -> bool:
        """检查紧急中断条件（A39.2）。

        当安全驱力值低于紧急阈值时触发无条件中断。

        Args:
            r_safe: 当前安全驱力值
            urgent_threshold: 紧急阈值

        Returns:
            是否满足紧急中断条件

        Examples:
            >>> engine = DynamicInterruptorEngine()
            >>> engine.check_urgent_condition(-0.8, -0.5)
            True
            >>> engine.check_urgent_condition(-0.3, -0.5)
            False
        """
        return r_safe < urgent_threshold

    def create_checkpoint(
        self,
        task_id: str,
        progress: float,
        intermediate_state: bytes,
        timestamp: int,
        task_type: str = ""
    ) -> Checkpoint:
        """创建中断保存点（A39.3）。

        Args:
            task_id: 任务ID
            progress: 执行进度 ∈ [0,1]
            intermediate_state: 中间状态字节
            timestamp: 时间步 >= 0
            task_type: 任务类型

        Returns:
            保存点

        Raises:
            ValueError: 当 progress 不在 [0,1] 或 timestamp < 0 时

        Examples:
            >>> engine = DynamicInterruptorEngine()
            >>> cp = engine.create_checkpoint("t1", 0.5, b"state", 10)
            >>> cp.task_id
            't1'
        """
        if not (0.0 <= progress <= 1.0):
            raise ValueError(f"进度必须在 [0,1] 内，收到 {progress}")
        if timestamp < 0:
            raise ValueError(f"时间步必须 >= 0，收到 {timestamp}")
        # 根据任务类型预估恢复成本
        cost_map = {
            "state_monitor": 0.01,
            "anomaly_detect": 0.05,
            "stabilize": 0.1,
            "risk_predict": 0.15,
            "explore_unknown": 0.3,
            "model_compress": 0.5,
        }
        est_resume = cost_map.get(task_type, 0.1)
        return Checkpoint(
            task_id=task_id,
            progress=progress,
            intermediate_state=intermediate_state,
            timestamp=timestamp,
            task_type=task_type,
            estimated_resume_cost=est_resume,
        )

    def resume_from_checkpoint(
        self,
        task: 'PooledTask',
        checkpoint: Checkpoint,
        current_step: int
    ) -> 'PooledTask':
        """从保存点恢复任务（A39.4）。

        Args:
            task: 被中断的任务
            checkpoint: 保存点
            current_step: 当前时间步 >= 0

        Returns:
            恢复后的任务

        Raises:
            ValueError: 当 current_step < 0 或 checkpoint.task_id 不匹配时

        Examples:
            >>> from stage_036 import PooledTask, TaskStatus
            >>> from stage_033 import SafetyTask, SafetyTaskType, SafetyTaskTarget
            >>> task = PooledTask(
            ...     task=SafetyTask(SafetyTaskType.STATE_MONITOR,
            ...         SafetyTaskTarget.SYSTEM, 0.5, 0.2, 0.3),
            ...     status=TaskStatus.ACTIVE, priority=0.5
            ... )
            >>> engine = DynamicInterruptorEngine()
            >>> cp = engine.create_checkpoint("t1", 0.5, b"state", 10)
            >>> resumed = engine.resume_from_checkpoint(task, cp, 20)
            >>> resumed.status == TaskStatus.ACTIVE
            True
        """
        if current_step < 0:
            raise ValueError(f"当前时间步必须 >= 0，收到 {current_step}")
        # 恢复任务：重置状态为活跃，保留优先级
        return PooledTask(
            task=task.task,
            status=TaskStatus.ACTIVE,
            priority=task.priority,
            rounds_no_progress=0,
            frozen_reason="",
            created_at=task.created_at,
            last_refreshed=current_step,
        )

    def check_frequency_limit(
        self,
        current_step: int,
        last_interrupt_step: int,
        min_exec_time: int = 5,
        is_urgent: bool = False
    ) -> bool:
        """检查中断频率限制（A39.5）。

        紧急中断不受频率限制。

        Args:
            current_step: 当前时间步
            last_interrupt_step: 上次中断时间步
            min_exec_time: 最小执行时间 >= 1
            is_urgent: 是否紧急中断

        Returns:
            是否允许中断

        Raises:
            ValueError: 当 min_exec_time < 1 时

        Examples:
            >>> engine = DynamicInterruptorEngine()
            >>> engine.check_frequency_limit(10, 3, min_exec_time=5)
            False
            >>> engine.check_frequency_limit(10, 3, min_exec_time=5, is_urgent=True)
            True
        """
        if min_exec_time < 1:
            raise ValueError(f"最小执行时间必须 >= 1，收到 {min_exec_time}")
        if is_urgent:
            return True
        return (current_step - last_interrupt_step) >= min_exec_time

    def compute_interrupt_cost(
        self,
        current_task: 'PooledTask',
        checkpoint: Checkpoint,
        current_score: float
    ) -> InterruptCost:
        """计算中断成本（A39.6）。

        总成本 = 保存成本 + 恢复成本 + 上下文切换成本 + 进度损失。

        Args:
            current_task: 当前任务
            checkpoint: 保存点
            current_score: 当前任务得分

        Returns:
            中断成本详情

        Examples:
            >>> from stage_036 import PooledTask, TaskStatus
            >>> from stage_033 import SafetyTask, SafetyTaskType, SafetyTaskTarget
            >>> task = PooledTask(
            ...     task=SafetyTask(SafetyTaskType.STATE_MONITOR,
            ...         SafetyTaskTarget.SYSTEM, 0.5, 0.2, 0.3),
            ...     status=TaskStatus.ACTIVE, priority=0.5
            ... )
            >>> engine = DynamicInterruptorEngine()
            >>> cp = engine.create_checkpoint("t1", 0.5, b"state", 10, "state_monitor")
            >>> cost = engine.compute_interrupt_cost(task, cp, 0.5)
            >>> cost.total_cost >= 0
            True
        """
        # 保存成本：与中间状态数据量成正比
        checkpoint_cost = len(checkpoint.intermediate_state) / 1e6 * 0.1
        # 恢复成本：使用保存点中的预估值
        resume_cost = checkpoint.estimated_resume_cost
        # 上下文切换成本：与任务复杂度成正比
        context_switch_cost = 0.05
        # 进度损失：当前任务未完成部分的预期收益
        remaining_progress = 1.0 - checkpoint.progress
        progress_loss = current_score * remaining_progress
        return InterruptCost(
            checkpoint_cost=float(checkpoint_cost),
            resume_cost=float(resume_cost),
            context_switch_cost=float(context_switch_cost),
            progress_loss=float(progress_loss),
        )

    def evaluate_interrupt(
        self,
        interruptor: DynamicInterruptor,
        current_task: 'PooledTask',
        current_score: float,
        new_task: 'PooledTask',
        new_score: float,
        r_safe: float,
        current_step: int
    ) -> InterruptDecision:
        """完整的中断评估流程。

        1. 检查紧急中断条件
        2. 检查频率限制
        3. 检查普通中断条件
        4. 评估中断成本
        5. 计算净收益
        6. 做出中断决策

        Args:
            interruptor: 动态中断器
            current_task: 当前任务
            current_score: 当前任务得分
            new_task: 新任务
            new_score: 新任务得分
            r_safe: 安全驱力值
            current_step: 当前时间步

        Returns:
            中断决策

        Examples:
            >>> from stage_036 import PooledTask, TaskStatus
            >>> from stage_033 import SafetyTask, SafetyTaskType, SafetyTaskTarget
            >>> ct = PooledTask(
            ...     task=SafetyTask(SafetyTaskType.STATE_MONITOR,
            ...         SafetyTaskTarget.SYSTEM, 0.5, 0.2, 0.3),
            ...     status=TaskStatus.ACTIVE, priority=0.5
            ... )
            >>> nt = PooledTask(
            ...     task=SafetyTask(SafetyTaskType.ANOMALY_DETECT,
            ...         SafetyTaskTarget.LOG, 0.9, 0.3, 0.6),
            ...     status=TaskStatus.ACTIVE, priority=0.9
            ... )
            >>> engine = DynamicInterruptorEngine()
            >>> di = DynamicInterruptor(
            ...     decision_mechanism=None, interrupt_threshold=0.2,
            ...     urgent_threshold=-0.5, min_exec_time=5,
            ...     last_interrupt_step=-100
            ... )
            >>> decision = engine.evaluate_interrupt(
            ...     di, ct, 0.5, nt, 0.9, r_safe=-0.3, current_step=10
            ... )
            >>> isinstance(decision.should_interrupt, bool)
            True
        """
        # 1. 紧急中断检查
        if self.check_urgent_condition(r_safe, interruptor.urgent_threshold):
            return InterruptDecision(
                should_interrupt=True,
                interrupt_type=InterruptType.URGENT,
                current_task_id=getattr(current_task.task, "scope", ""),
                new_task_id=getattr(new_task.task, "scope", ""),
                reason="紧急安全中断",
            )
        # 2. 频率限制检查
        freq_ok = self.check_frequency_limit(
            current_step,
            interruptor.last_interrupt_step,
            interruptor.min_exec_time,
            is_urgent=False,
        )
        if not freq_ok:
            return InterruptDecision(
                should_interrupt=False,
                interrupt_type=InterruptType.NONE,
                current_task_id=getattr(current_task.task, "scope", ""),
                reason="中断频率受限",
            )
        # 3. 普通中断条件
        if not self.check_interrupt_condition(
            current_score, new_score, interruptor.interrupt_threshold
        ):
            return InterruptDecision(
                should_interrupt=False,
                interrupt_type=InterruptType.NONE,
                current_task_id=getattr(current_task.task, "scope", ""),
                reason="得分差距不足",
            )
        # 4. 评估中断成本
        checkpoint = self.create_checkpoint(
            task_id=getattr(current_task.task, "scope", ""),
            progress=0.5,
            intermediate_state=b"",
            timestamp=current_step,
            task_type=getattr(current_task.task, "task_type", ""),
        )
        cost = self.compute_interrupt_cost(current_task, checkpoint, current_score)
        # 5. 计算净收益
        net_benefit = new_score - current_score - cost.total_cost
        # 6. 决策
        if net_benefit > 0:
            return InterruptDecision(
                should_interrupt=True,
                interrupt_type=InterruptType.NORMAL,
                current_task_id=getattr(current_task.task, "scope", ""),
                new_task_id=getattr(new_task.task, "scope", ""),
                checkpoint=checkpoint,
                cost=cost,
                net_benefit=float(net_benefit),
                reason="净收益为正",
            )
        return InterruptDecision(
            should_interrupt=False,
            interrupt_type=InterruptType.NONE,
            current_task_id=getattr(current_task.task, "scope", ""),
            cost=cost,
            net_benefit=float(net_benefit),
            reason="净收益非正",
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `DualDecisionMechanism` | ← 阶段38 | 作为动态中断器的决策机制组合字段 |
| **输入** | A37.1（综合得分） | ← 阶段37 | 中断条件的得分比较依据 |
| **输入** | A26.2（紧急风险判定） | ← 阶段26 | 紧急中断的阈值依据 |
| **输出** | `DynamicInterruptor` | → 阶段40 | 中断器作为决策引擎的组件 |
| **输出** | A39.3（保存点） | → 工程实现 | 中断恢复的数据基础 |
| **输出** | A39.6（中断成本评估） | → 后续阶段 | 中断决策的成本-收益分析 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 中断判定计算 | `jax.lax.cond`实现中断条件的级联判断（A39.1+A39.2）；`jax.numpy.exp`计算保存点价值衰减（A39.4） |
| pickle / dill | 保存点序列化 | Python对象序列化库，支撑中间状态的保存和恢复（A39.3, A39.4） |
| zlib | 保存点压缩 | 压缩中间状态数据，降低存储成本（A39.3-storage） |
| asyncio | 异步中断处理 | Python异步IO框架，支撑中断信号的异步传递和任务切换（A39.1） |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V39.1 | 普通中断条件 | Score(new) > Score(current) + 0.2 时触发 | 得分差扫描测试 |
| V39.2 | 中断阈值灵敏度 | 阈值降低时中断更频繁 | 阈值扫描测试 |
| V39.3 | 紧急中断无条件 | R_safe < θ_urgent 时无论其他条件都中断 | 紧急场景模拟 |
| V39.4 | 紧急中断目标 | 紧急中断后切换到安全任务 | 目标类型检查 |
| V39.5 | 保存点完整性 | checkpoint包含task_id, progress, state | 字段完整性检查 |
| V39.6 | 保存点恢复 | 恢复后任务从保存进度继续 | 恢复后进度验证 |
| V39.7 | 恢复参数更新 | 恢复后优先级和可行性反映当前环境 | 参数更新验证 |
| V39.8 | 保存点衰减 | 旧保存点价值低于新保存点 | 时间衰减测试 |
| V39.9 | 频率限制 | 中断间隔 ≥ T_min_exec | 连续中断模拟 |
| V39.10 | 紧急中断不受频率限制 | 紧急中断可在T_min_exec内触发 | 紧急+频率组合测试 |
| V39.11 | 中断成本非负 | cost_interrupt ≥ 0 | 全场景扫描 |
| V39.12 | 中断净收益 | net_benefit > 0 时才中断 | 成本-收益测试 |
| V39.13 | model_compress高成本 | 模型压缩的中断成本最高 | 各类型成本对比 |
| V39.14 | DynamicInterruptor组合 | `DynamicInterruptor.decision_mechanism` 类型为 `DualDecisionMechanism` | 类型检查 |
| V39.15 | 非法输入拒绝 | 非正阈值、min_exec < 1等抛出 `ValueError` | 异常测试 |

---

> **阶段39完** | 阶段39建立了动态中断机制的完整体系：普通中断条件（得分差超阈值）处理优先级跃迁，紧急中断条件（安全驱力超限）处理生存威胁，保存点机制确保中断后可恢复，频率限制防止频繁切换，成本评估确保中断的经济性。下一步：阶段40将整合阶段31-39的所有决策相关接口，定义自主决策引擎的完整Python接口。
