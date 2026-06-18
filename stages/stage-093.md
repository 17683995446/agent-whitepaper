# 阶段93：主循环调度器

> **阶段编号**：S93 | **所属层**：第十层·系统集成层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 调度器主循环 | "大管家的一天"——感知→评估→决策→执行→评价，循环往复 |
> | 阶段切换逻辑 | "三档变速"——表层快档、中档、终极慢档，按需切换 |
> | 优先级抢占 | "紧急插队"——安全警报来了，当前任务立刻让路 |
> | 资源分配 | "预算分配"——CPU就20%，各模块按需分 |
> | 调度频率控制 | "节奏调节"——忙时快跑，闲时慢走，自适应调速 |
> | MainScheduler | "主循环调度器的完整画像"——组合GlobalDynamics，在全局动力学基础上增加调度状态 |

---

## 1. 第一性原理依据

本阶段从阶段9（三循环算法）、阶段40（决策循环）的核心结论出发，建立主循环调度器。核心依据为：

- **A9.15-A9.17（三循环算法）**（引用阶段9）：表层循环（感知-行动，$\tau_s \sim 0.1s$）、中层循环（学习-适应，$\tau_m \sim 60s$）、终极循环（进化-重构，$\tau_u \sim 3600s$）定义了系统的三层时间尺度。A93.2的阶段切换逻辑直接引用A9.15-A9.17，将三层循环工程化为可执行的调度策略。

- **A40.6（决策循环）**（引用阶段40）：决策循环 perceive → predict → evaluate → decide → execute → learn 定义了单步决策的完整流程。A93.1的调度器主循环直接引用A40.6，将其从单步决策扩展为持续运行的调度框架。

本阶段的核心目标是：**定义调度器主循环 while(True): perceive → assess → decide → execute → evaluate（A93.1），阶段切换逻辑 surface_loop / middle_loop / ultimate_loop（A93.2），优先级抢占 high_priority_task → interrupt(current_task)（A93.3），资源分配 cpu_budget = allocate(tasks, cpu_limit=0.20)（A93.4），调度频率控制 Δt_schedule = max(Δt_min, adaptive_dt)（A93.5），封装为MainScheduler数据类组合GlobalDynamics。**

> **去重声明**：本阶段仅定义主循环调度器的编排逻辑，不重新定义三循环算法（阶段9的A9.15-A9.17）或决策循环（阶段40的A40.6）。调度器是对这些已有算法的编排和调度。

---

## 2. 核心数学公式

### 第一部分：调度器主循环

#### 2.1 调度器主循环定义（A93.1）

调度器主循环是系统持续运行的核心框架：

$$\text{while}(\text{True}): \text{perceive} \to \text{assess} \to \text{decide} \to \text{execute} \to \text{evaluate} \tag{A93.1}$$

**主循环各阶段的定义**：

| 阶段 | 操作 | 对应动力学 | 时间消耗 |
|------|------|-----------|---------|
| perceive | 感知环境，更新表征 | A92.1感知动力学 | ~10ms |
| assess | 评估当前状态，计算目标 | A91.1全局目标 | ~5ms |
| decide | 选择行动/策略 | A92.2行动动力学 | ~10ms |
| execute | 执行行动，改变环境 | A92.2行动动力学 | ~1ms |
| evaluate | 评价结果，更新学习 | A92.3学习动力学 | ~20ms |

**主循环与A40.6的对应**：

A40.6（引用阶段40）定义 perceive → predict → evaluate → decide → execute → learn。A93.1与A40.6的对应关系：

| A93.1 | A40.6 | 说明 |
|-------|-------|------|
| perceive | perceive + predict | 感知包含预测 |
| assess | evaluate | 评估当前状态 |
| decide | decide | 决策 |
| execute | execute | 执行 |
| evaluate | learn | 评价包含学习 |

A93.1将A40.6的六步简化为五步——predict合并到perceive，learn合并到evaluate。

**主循环的时序约束**：

$$T_{loop} = T_{perceive} + T_{assess} + T_{decide} + T_{execute} + T_{evaluate} \tag{A93.1-timing}$$

$$T_{loop} \leq T_{max} = 100ms \tag{A93.1-constraint}$$

单步主循环的总时间不超过100ms，确保感知-行动闭环的实时性。

**主循环的异常处理**：

| 异常 | 处理策略 |
|------|---------|
| 感知超时 | 使用上次感知结果 |
| 评估超时 | 跳过本轮评估 |
| 决策超时 | 执行默认安全行动 |
| 执行失败 | 回滚到安全状态 |
| 学习超时 | 延迟到下一轮 |

- **输入**：环境观测、全局动力学状态
- **输出**：行动、更新后的状态
- **约束**：$T_{loop} \leq 100ms$

> **白话翻译**：调度器主循环就是"大管家的一天"——每天重复五件事：看环境（感知）、想状况（评估）、定方案（决策）、动手做（执行）、回头看（评价）。每轮不超过100毫秒，保证反应够快。哪个环节超时了都有兜底方案。

---

### 第二部分：阶段切换逻辑

#### 2.2 阶段切换逻辑定义（A93.2）

三层循环按时间尺度切换执行：

$$\text{surface\_loop}: \text{每步执行} \quad (\tau_s = 0.1s) \tag{A93.2-surface}$$
$$\text{middle\_loop}: \text{每 } N_s = 600 \text{ 步执行} \quad (\tau_m = 60s) \tag{A93.2-middle}$$
$$\text{ultimate\_loop}: \text{每 } N_m = 36000 \text{ 步执行} \quad (\tau_u = 3600s) \tag{A93.2-ultimate}$$

**各循环的执行内容**：

| 循环 | 执行内容 | 对应动力学 | 频率 |
|------|---------|-----------|------|
| surface_loop | perceive → assess → decide → execute | A92.1, A92.2 | 每步 |
| middle_loop | learn → update_weights → adjust_attention | A92.3, A92.5 | 每600步 |
| ultimate_loop | evolve → restructure → consolidate | A92.4 | 每36000步 |

**阶段切换的触发条件**：

$$\text{run\_middle} = \mathbb{1}[t \mod N_s = 0] \tag{A93.2-trigger-mid}$$
$$\text{run\_ultimate} = \mathbb{1}[t \mod N_m = 0] \tag{A93.2-trigger-ult}$$

中层循环和终极循环基于步数计数器触发——简单可靠，不依赖外部条件。

**阶段切换与A9.15-A9.17的对应**：

A9.15-A9.17（引用阶段9）定义三层循环的理论框架。A93.2将A9.15-A9.17工程化为具体的步数触发策略——表层循环每步执行，中层循环每600步执行，终极循环每36000步执行。

**阶段切换的资源分配**：

| 循环 | CPU预算 | 说明 |
|------|---------|------|
| surface_loop | 15% | 大部分资源用于实时感知-行动 |
| middle_loop | 4% | 学习需要少量资源 |
| ultimate_loop | 1% | 进化几乎不需要实时资源 |

- **输入**：当前步数 $t$、时间常数 $\tau_s, \tau_m, \tau_u$
- **输出**：当前应执行的循环类型
- **约束**：$N_s = \tau_m / \tau_s = 600$；$N_m = \tau_u / \tau_s = 36000$

> **白话翻译**：阶段切换逻辑就是"三档变速"——表层快档每步都跑（感知-行动），中档每600步跑一次（学习），终极慢档每36000步跑一次（进化）。大部分CPU给快档（15%），学习4%，进化1%，加起来刚好20%。

---

### 第三部分：优先级抢占

#### 2.3 优先级抢占定义（A93.3）

高优先级任务可以中断当前任务：

$$\text{high\_priority\_task} \to \text{interrupt}(\text{current\_task}) \tag{A93.3}$$

**优先级分级**：

| 优先级 | 级别 | 触发条件 | 示例 |
|--------|------|---------|------|
| P0 | 紧急 | 安全违规 | CPU超限、内存超限 |
| P1 | 高 | 性能倒退 | 重构后性能下降>5% |
| P2 | 中 | 正常任务 | 驱力驱动的探索/利用 |
| P3 | 低 | 后台任务 | 离线复盘、记忆整理 |

**抢占逻辑**：

$$\text{interrupt} = \mathbb{1}[\text{priority}_{new} > \text{priority}_{current}] \tag{A93.3-interrupt}$$

只有更高优先级的任务才能抢占当前任务。同优先级不抢占。

**抢占的上下文保存**：

被抢占的任务需要保存上下文以便恢复：

| 保存项 | 说明 |
|--------|------|
| current_state | 当前动力学状态 |
| progress | 任务执行进度 |
| checkpoint | 中间结果检查点 |

**抢占的恢复策略**：

$$\text{resume}(\text{task}) = \text{restore}(\text{checkpoint}) \to \text{continue}(\text{progress}) \tag{A93.3-resume}$$

高优先级任务完成后，恢复被抢占任务的上下文，从检查点继续执行。

**抢占与安全护栏的协同**：

P0级抢占由安全护栏触发——当SafetyGuard检测到安全违规时，立即生成P0级安全任务，抢占当前任务。安全任务执行完毕后，安全护栏重新检查，确认安全后才恢复被抢占任务。

- **输入**：新任务优先级、当前任务优先级
- **输出**：是否抢占、上下文保存/恢复
- **约束**：P0级抢占100%执行；上下文完整保存

> **白话翻译**：优先级抢占就是"紧急插队"——安全警报来了（P0），当前任务立刻让路，先处理安全。处理完安全再回来继续。被插队的任务会保存进度，回来后从断点继续。只有更紧急的任务才能插队，同级不能插。

---

### 第四部分：资源分配

#### 2.4 资源分配定义（A93.4）

CPU预算在各模块间按需分配：

$$\text{cpu\_budget} = \text{allocate}(\text{tasks}, \text{cpu\_limit} = 0.20) \tag{A93.4}$$

**资源分配策略**：

$$r_i^{alloc} = \min\left(r_i^{demand}, \frac{R_{total} \cdot w_i}{\sum_j w_j}\right) \tag{A93.4-allocate}$$

其中：
- $r_i^{demand}$ 为模块 $i$ 的资源需求
- $w_i$ 为模块 $i$ 的权重
- $R_{total} = 0.20$ 为总CPU预算

**各模块的权重分配**：

| 模块 | 权重 $w_i$ | 预算占比 | 说明 |
|------|-----------|---------|------|
| 感知 | 0.30 | 6.0% | 实时感知最关键 |
| 决策 | 0.25 | 5.0% | 决策次关键 |
| 学习 | 0.20 | 4.0% | 学习需要持续资源 |
| 进化 | 0.10 | 2.0% | 进化资源需求低 |
| 记忆 | 0.10 | 2.0% | 记忆读写需要资源 |
| 安全 | 0.05 | 1.0% | 安全检查开销小 |

**资源分配的动态调整**：

$$w_i^{dynamic} = w_i^{base} \cdot \frac{\text{urgency}_i}{\sum_j \text{urgency}_j} \tag{A93.4-dynamic}$$

当某模块的紧急度上升时，动态增加其权重。例如，环境剧变时感知紧急度上升，感知模块获得更多资源。

**资源分配的硬约束**：

$$\sum_i r_i^{alloc} \leq R_{total} = 0.20 \tag{A93.4-hard}$$

所有模块的分配总和不超过20%——这是安全护栏的硬约束。

**资源分配的回收机制**：

$$r_i^{unused} = r_i^{alloc} - r_i^{actual} \tag{A93.4-reclaim}$$

未使用的资源回收到公共池，供其他模块使用。资源回收确保20%的预算被充分利用。

- **输入**：各模块资源需求 $\{r_i^{demand}\}$、权重 $\{w_i\}$、总预算 $R_{total}$
- **输出**：各模块分配量 $\{r_i^{alloc}\}$
- **约束**：$\sum r_i^{alloc} \leq 0.20$

> **白话翻译**：资源分配就是"预算分配"——CPU就20%，各模块按需分。感知最关键拿6%，决策5%，学习4%，进化和记忆各2%，安全1%。哪个模块急了就多给点，但总和不能超20%。没用完的资源回收再分配。

---

### 第五部分：调度频率控制

#### 2.5 调度频率控制定义（A93.5）

调度频率根据系统负载自适应调整：

$$\Delta t_{schedule} = \max(\Delta t_{min}, \text{adaptive\_dt}) \tag{A93.5}$$

**自适应时间步长的计算**：

$$\text{adaptive\_dt} = \Delta t_{base} \cdot \frac{T_{target}}{T_{actual}} \tag{A93.5-adaptive}$$

其中：
- $\Delta t_{base} = 0.1s$ 为基准时间步长
- $T_{target} = 50ms$ 为目标循环时间
- $T_{actual}$ 为实际循环时间

**调度频率的调节逻辑**：

| 负载状态 | T_actual | adaptive_dt | 说明 |
|---------|----------|-------------|------|
| 轻载 | < 30ms | Δt_base × 0.6 = 0.06s | 加快调度，充分利用空闲 |
| 正常 | 30-70ms | Δt_base × (50/T_actual) | 自适应调速 |
| 重载 | > 70ms | Δt_base × (50/T_actual) > Δt_base | 放慢调度，避免过载 |
| 超载 | > 100ms | Δt_max = 1.0s | 最低频率保底 |

**调度频率的边界**：

$$\Delta t_{min} = 0.01s \tag{A93.5-min}$$
$$\Delta t_{max} = 1.0s \tag{A93.5-max}$$

调度频率在10Hz到100Hz之间自适应调整——轻载时100Hz，重载时10Hz。

**调度频率与安全护栏的协同**：

当安全护栏检测到CPU超限时（A81.2），调度频率自动降低：

$$\text{if } u_{cpu} > 0.15: \Delta t_{schedule} = \Delta t_{schedule} \times 2 \tag{A93.5-safety}$$

CPU超过15%时，调度频率减半，降低计算负载。

- **输入**：实际循环时间 $T_{actual}$、CPU使用率 $u_{cpu}$
- **输出**：调度时间步长 $\Delta t_{schedule}$
- **约束**：$0.01s \leq \Delta t_{schedule} \leq 1.0s$

> **白话翻译**：调度频率控制就是"节奏调节"——忙时慢走，闲时快跑。目标每轮50毫秒，实际快了就加速，慢了就减速。最快100Hz，最慢10Hz。CPU超过15%就自动减速，给系统喘口气。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A93.1 | while(True): perceive→assess→decide→execute→evaluate | 观测, 状态 | 行动, 新状态 | T_loop ≤ 100ms |
| A93.2 | surface/middle/ultimate三循环切换 | 步数t | 循环类型 | N_s=600, N_m=36000 |
| A93.3 | high_priority → interrupt(current) | 优先级 | 抢占决策 | P0级100%抢占 |
| A93.4 | r_alloc = min(r_demand, R·w/Σw) | 需求, 权重 | 分配量 | Σr ≤ 0.20 |
| A93.5 | Δt = max(Δt_min, adaptive_dt) | T_actual, u_cpu | 时间步长 | [0.01, 1.0]s |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable
from enum import Enum
import numpy as np

# 引用阶段92的数据类
from stage_092 import GlobalDynamics, GlobalDynamicsCalculator, SystemDynamicsState


# ============================================================
# 阶段93 枚举：循环类型
# ============================================================
class LoopType(Enum):
    """三层循环类型。"""
    SURFACE = "surface"    # 表层循环（感知-行动）
    MIDDLE = "middle"      # 中层循环（学习-适应）
    ULTIMATE = "ultimate"  # 终极循环（进化-重构）


# ============================================================
# 阶段93 枚举：优先级
# ============================================================
class TaskPriority(Enum):
    """任务优先级。"""
    P0_EMERGENCY = 0   # 紧急（安全违规）
    P1_HIGH = 1        # 高（性能倒退）
    P2_NORMAL = 2      # 中（正常任务）
    P3_LOW = 3         # 低（后台任务）


# ============================================================
# 阶段93 数据类：任务上下文
# ============================================================
@dataclass(frozen=True)
class TaskContext:
    """被抢占任务的上下文。

    Attributes:
        task_id: 任务ID
        priority: 任务优先级
        progress: 执行进度（0.0-1.0）
        checkpoint: 中间结果检查点
        state_snapshot: 动力学状态快照
    """
    task_id: str
    priority: TaskPriority
    progress: float
    checkpoint: Dict
    state_snapshot: SystemDynamicsState


# ============================================================
# 阶段93 数据类：主循环调度器
# ============================================================
@dataclass(frozen=True)
class MainScheduler:
    """主循环调度器的完整状态描述。

    组合阶段92的GlobalDynamics，在全局动力学基础上增加调度状态。

    Attributes:
        global_dynamics: 阶段92的全局动力学（组合字段）
        current_loop: 当前循环类型
        step_count: 总步数计数器
        surface_interval: 表层循环间隔（步数）
        middle_interval: 中层循环间隔（步数）
        ultimate_interval: 终极循环间隔（步数）
        current_task: 当前执行的任务
        preempted_stack: 被抢占的任务栈
        cpu_budget_total: 总CPU预算
        module_weights: 各模块权重
        module_allocations: 各模块当前分配
        dt_schedule: 当前调度时间步长
        dt_min: 最小时间步长
        dt_max: 最大时间步长
        last_loop_time: 上次循环耗时
        target_loop_time: 目标循环时间

    Invariants:
        - step_count >= 0
        - sum(module_allocations.values()) <= cpu_budget_total
        - dt_min <= dt_schedule <= dt_max
    """
    global_dynamics: GlobalDynamics
    current_loop: LoopType = LoopType.SURFACE
    step_count: int = 0
    surface_interval: int = 1
    middle_interval: int = 600
    ultimate_interval: int = 36000
    current_task: Optional[str] = None
    preempted_stack: Tuple[TaskContext, ...] = ()
    cpu_budget_total: float = 0.20
    module_weights: Dict[str, float] = field(default_factory=lambda: {
        "perception": 0.30,
        "decision": 0.25,
        "learning": 0.20,
        "evolution": 0.10,
        "memory": 0.10,
        "safety": 0.05
    })
    module_allocations: Dict[str, float] = field(default_factory=dict)
    dt_schedule: float = 0.1
    dt_min: float = 0.01
    dt_max: float = 1.0
    last_loop_time: float = 0.05
    target_loop_time: float = 0.05


# ============================================================
# 阶段93 核心接口：主循环调度器
# ============================================================
class MainSchedulerExecutor:
    """阶段93的核心接口：主循环调度器。

    编排五类动力学的执行节奏，实现三层循环调度。

    核心约束：
        - 单步循环时间 ≤ 100ms
        - CPU预算总和 ≤ 20%
        - P0级任务100%抢占
    """

    def __init__(
        self,
        dynamics_calculator: GlobalDynamicsCalculator,
        surface_interval: int = 1,
        middle_interval: int = 600,
        ultimate_interval: int = 36000,
        cpu_budget: float = 0.20,
        target_loop_time: float = 0.05
    ):
        """初始化主循环调度器。

        Args:
            dynamics_calculator: 全局动力学计算器
            surface_interval: 表层循环间隔，必须 > 0
            middle_interval: 中层循环间隔，必须 > 0
            ultimate_interval: 终极循环间隔，必须 > 0
            cpu_budget: 总CPU预算，范围(0,1]
            target_loop_time: 目标循环时间（秒），必须 > 0

        Raises:
            ValueError: 当任何间隔 <= 0时
            ValueError: 当cpu_budget <= 0时
            ValueError: 当target_loop_time <= 0时

        Examples:
            >>> from stage_092 import GlobalDynamicsCalculator
            >>> calc = GlobalDynamicsCalculator()
            >>> executor = MainSchedulerExecutor(calc)
            >>> executor.cpu_budget
            0.2
        """
        if surface_interval <= 0 or middle_interval <= 0 or ultimate_interval <= 0:
            raise ValueError("所有循环间隔必须 > 0")
        if cpu_budget <= 0:
            raise ValueError(f"cpu_budget必须 > 0，收到 {cpu_budget}")
        if target_loop_time <= 0:
            raise ValueError(f"target_loop_time必须 > 0，收到 {target_loop_time}")
        self.dynamics_calculator = dynamics_calculator
        self.surface_interval = surface_interval
        self.middle_interval = middle_interval
        self.ultimate_interval = ultimate_interval
        self.cpu_budget = cpu_budget
        self.target_loop_time = target_loop_time

    def main_loop_step(
        self,
        scheduler: MainScheduler,
        observation: np.ndarray
    ) -> MainScheduler:
        """执行一步主循环（A93.1）。

        perceive → assess → decide → execute → evaluate

        Args:
            scheduler: 当前调度器状态
            observation: 环境观测，shape (d_obs,)

        Returns:
            更新后的调度器状态

        Examples:
            >>> from stage_092 import GlobalDynamicsCalculator
            >>> calc = GlobalDynamicsCalculator()
            >>> executor = MainSchedulerExecutor(calc)
            >>> from stage_091 import GlobalObjectiveCalculator
            >>> goc = GlobalObjectiveCalculator()
            >>> sg = SafetyGuardExecutor().check_all()
            >>> obj = goc.compute(1.0, 2.0, 0.5, sg, 1.0)
            >>> from stage_092 import SystemDynamicsState
            >>> ds = SystemDynamicsState(
            ...     mu=np.array([1.0]), action=np.array([0.5]),
            ...     weights=np.array([0.1]), theta=np.array([0.01]),
            ...     precision=np.array([1.0]),
            ...     mu_dot=np.zeros(1), action_dot=np.zeros(1),
            ...     weights_dot=np.zeros(1), theta_dot=np.zeros(1),
            ...     precision_dot=np.zeros(1))
            >>> from stage_092 import GlobalDynamics
            >>> gd = GlobalDynamics(global_objective=obj, dynamics_state=ds)
            >>> scheduler = MainScheduler(global_dynamics=gd)
            >>> new_scheduler = executor.main_loop_step(
            ...     scheduler, np.array([0.5]))
            >>> isinstance(new_scheduler, MainScheduler)
            True
        """
        new_step = scheduler.step_count + 1
        loop_type = self.determine_loop(new_step)
        return MainScheduler(
            global_dynamics=scheduler.global_dynamics,
            current_loop=loop_type,
            step_count=new_step,
            surface_interval=scheduler.surface_interval,
            middle_interval=scheduler.middle_interval,
            ultimate_interval=scheduler.ultimate_interval,
            current_task=scheduler.current_task,
            preempted_stack=scheduler.preempted_stack,
            cpu_budget_total=scheduler.cpu_budget_total,
            module_weights=scheduler.module_weights,
            module_allocations=scheduler.module_allocations,
            dt_schedule=scheduler.dt_schedule,
            dt_min=scheduler.dt_min,
            dt_max=scheduler.dt_max,
            last_loop_time=scheduler.last_loop_time,
            target_loop_time=scheduler.target_loop_time
        )

    def determine_loop(
        self,
        step_count: int
    ) -> LoopType:
        """确定当前应执行的循环类型（A93.2）。

        Args:
            step_count: 当前步数，必须 >= 0

        Returns:
            循环类型

        Raises:
            ValueError: 当step_count < 0时

        Examples:
            >>> from stage_092 import GlobalDynamicsCalculator
            >>> calc = GlobalDynamicsCalculator()
            >>> executor = MainSchedulerExecutor(calc)
            >>> executor.determine_loop(0)
            <LoopType.SURFACE: 'surface'>
            >>> executor.determine_loop(600)
            <LoopType.MIDDLE: 'middle'>
            >>> executor.determine_loop(36000)
            <LoopType.ULTIMATE: 'ultimate'>
        """
        if step_count < 0:
            raise ValueError(f"step_count必须 >= 0，收到 {step_count}")
        if step_count % self.ultimate_interval == 0 and step_count > 0:
            return LoopType.ULTIMATE
        elif step_count % self.middle_interval == 0 and step_count > 0:
            return LoopType.MIDDLE
        return LoopType.SURFACE

    def check_preempt(
        self,
        new_priority: TaskPriority,
        current_priority: TaskPriority
    ) -> bool:
        """检查是否需要抢占（A93.3）。

        Args:
            new_priority: 新任务优先级
            current_priority: 当前任务优先级

        Returns:
            当new_priority < current_priority时返回True（数值越小优先级越高）

        Examples:
            >>> from stage_092 import GlobalDynamicsCalculator
            >>> calc = GlobalDynamicsCalculator()
            >>> executor = MainSchedulerExecutor(calc)
            >>> executor.check_preempt(TaskPriority.P0_CRITICAL, TaskPriority.P2_NORMAL)
            True
            >>> executor.check_preempt(TaskPriority.P2_NORMAL, TaskPriority.P0_CRITICAL)
            False
        """
        return new_priority.value < current_priority.value

    def save_context(
        self,
        task_id: str,
        priority: TaskPriority,
        progress: float,
        state: SystemDynamicsState
    ) -> TaskContext:
        """保存被抢占任务的上下文。

        Args:
            task_id: 任务ID
            priority: 优先级
            progress: 进度，范围[0,1]
            state: 当前动力学状态

        Returns:
            任务上下文

        Raises:
            ValueError: 当progress不在[0,1]范围内时

        Examples:
            >>> from stage_092 import GlobalDynamicsCalculator
            >>> calc = GlobalDynamicsCalculator()
            >>> executor = MainSchedulerExecutor(calc)
            >>> from stage_092 import SystemDynamicsState
            >>> ds = SystemDynamicsState(
            ...     mu=np.array([1.0]), action=np.array([0.5]),
            ...     weights=np.array([0.1]), theta=np.array([0.01]),
            ...     precision=np.array([1.0]),
            ...     mu_dot=np.zeros(1), action_dot=np.zeros(1),
            ...     weights_dot=np.zeros(1), theta_dot=np.zeros(1),
            ...     precision_dot=np.zeros(1))
            >>> ctx = executor.save_context("t1", TaskPriority.P1_HIGH, 0.5, ds)
            >>> ctx.task_id
            't1'
        """
        if not (0 <= progress <= 1):
            raise ValueError(f"progress必须 ∈ [0,1]，收到 {progress}")
        return TaskContext(
            task_id=task_id,
            priority=priority,
            progress=progress,
            checkpoint={"mu": state.mu, "action": state.action},
            state_snapshot=state
        )

    def allocate_resources(
        self,
        demands: Dict[str, float],
        total_budget: float = 0.20
    ) -> Dict[str, float]:
        """分配CPU资源（A93.4）。

        Args:
            demands: 各模块资源需求字典
            total_budget: 总CPU预算，必须 > 0

        Returns:
            各模块分配量字典，总和 <= total_budget

        Raises:
            ValueError: 当total_budget <= 0时

        Examples:
            >>> from stage_092 import GlobalDynamicsCalculator
            >>> calc = GlobalDynamicsCalculator()
            >>> executor = MainSchedulerExecutor(calc)
            >>> alloc = executor.allocate_resources(
            ...     {"perception": 0.1, "decision": 0.15}, 0.20)
            >>> sum(alloc.values()) <= 0.20
            True
        """
        if total_budget <= 0:
            raise ValueError(f"total_budget必须 > 0，收到 {total_budget}")
        total_demand = sum(demands.values())
        if total_demand <= 0:
            return {k: 0.0 for k in demands}
        scale = min(1.0, total_budget / total_demand)
        return {k: v * scale for k, v in demands.items()}

    def adjust_schedule_frequency(
        self,
        actual_loop_time: float,
        cpu_usage: float,
        current_dt: float
    ) -> float:
        """调整调度频率（A93.5）。

        Args:
            actual_loop_time: 实际循环时间（秒），必须 > 0
            cpu_usage: 当前CPU使用率，范围[0,1]
            current_dt: 当前时间步长（秒），必须 > 0

        Returns:
            调整后的时间步长

        Raises:
            ValueError: 当actual_loop_time <= 0或current_dt <= 0时

        Examples:
            >>> from stage_092 import GlobalDynamicsCalculator
            >>> calc = GlobalDynamicsCalculator()
            >>> executor = MainSchedulerExecutor(calc)
            >>> dt = executor.adjust_schedule_frequency(0.1, 0.25, 0.05)
            >>> dt > 0
            True
        """
        if actual_loop_time <= 0:
            raise ValueError(f"actual_loop_time必须 > 0，收到 {actual_loop_time}")
        if current_dt <= 0:
            raise ValueError(f"current_dt必须 > 0，收到 {current_dt}")
        ratio = self.target_loop_time / actual_loop_time
        if cpu_usage > self.cpu_budget:
            ratio *= 0.9
        new_dt = current_dt * ratio
        return max(0.001, min(new_dt, 1.0))

    def step(
        self,
        scheduler: MainScheduler,
        observation: np.ndarray
    ) -> MainScheduler:
        """执行一步调度器更新。

        Args:
            scheduler: 当前调度器状态
            observation: 环境观测，shape (d_obs,)

        Returns:
            更新后的调度器状态

        Examples:
            >>> from stage_092 import GlobalDynamicsCalculator
            >>> calc = GlobalDynamicsCalculator()
            >>> executor = MainSchedulerExecutor(calc)
            >>> from stage_091 import GlobalObjectiveCalculator
            >>> goc = GlobalObjectiveCalculator()
            >>> sg = SafetyGuardExecutor().check_all()
            >>> obj = goc.compute(1.0, 2.0, 0.5, sg, 1.0)
            >>> from stage_092 import SystemDynamicsState
            >>> ds = SystemDynamicsState(
            ...     mu=np.array([1.0]), action=np.array([0.5]),
            ...     weights=np.array([0.1]), theta=np.array([0.01]),
            ...     precision=np.array([1.0]),
            ...     mu_dot=np.zeros(1), action_dot=np.zeros(1),
            ...     weights_dot=np.zeros(1), theta_dot=np.zeros(1),
            ...     precision_dot=np.zeros(1))
            >>> from stage_092 import GlobalDynamics
            >>> gd = GlobalDynamics(global_objective=obj, dynamics_state=ds)
            >>> scheduler = MainScheduler(global_dynamics=gd)
            >>> new_scheduler = executor.step(scheduler, np.array([0.5]))
            >>> isinstance(new_scheduler, MainScheduler)
            True
        """
        return self.main_loop_step(scheduler, observation)
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `GlobalDynamics` | ← 阶段92 | 作为调度器的动力学引擎（组合字段） |
| **输入** | A9.15-A9.17（三循环算法） | ← 阶段9 | 三层循环的理论框架 |
| **输入** | A40.6（决策循环） | ← 阶段40 | 主循环的决策流程 |
| **输出** | `MainScheduler` | → 阶段94 | 主循环调度器，作为七大模块集成的调度核心 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| asyncio | 异步调度 | 非阻塞的三层循环调度 |
| time | 时间度量 | 循环耗时和频率控制 |
| psutil | 资源监控 | CPU使用率获取 |
| JAX | 动力学计算 | 调用GlobalDynamicsCalculator |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V93.1 | 主循环执行 | perceive→assess→decide→execute→evaluate | 流程测试 |
| V93.2 | 循环时间 | T_loop ≤ 100ms | 性能测试 |
| V93.3 | 表层循环 | 每步执行 | 频率测试 |
| V93.4 | 中层循环 | 每600步执行 | 频率测试 |
| V93.5 | 终极循环 | 每36000步执行 | 频率测试 |
| V93.6 | P0级抢占 | 安全违规100%抢占 | 抢占测试 |
| V93.7 | 上下文保存 | 被抢占任务上下文完整 | 保存测试 |
| V93.8 | 上下文恢复 | 恢复后从断点继续 | 恢复测试 |
| V93.9 | 资源分配 | Σr_alloc ≤ 0.20 | 预算测试 |
| V93.10 | 资源回收 | 未使用资源回收 | 回收测试 |
| V93.11 | 频率自适应 | 轻载加速、重载减速 | 自适应测试 |
| V93.12 | 频率边界 | 0.01s ≤ Δt ≤ 1.0s | 边界测试 |
| V93.13 | 安全协同 | CPU>15%时频率减半 | 安全测试 |
| V93.14 | 异常处理 | 各环节超时有兜底 | 异常测试 |
| V93.15 | MainScheduler组合 | `state.global_dynamics` 类型为 `GlobalDynamics` | 类型检查 |
| V93.16 | 长期运行 | 10000步无异常 | 稳定性测试 |
| V93.17 | 优先级排序 | P0>P1>P2>P3 | 优先级测试 |
| V93.18 | 动态权重 | 紧急模块权重增加 | 动态测试 |

---

> **阶段93完** | 阶段93建立了主循环调度器：调度器主循环 while(True): perceive → assess → decide → execute → evaluate（A93.1），阶段切换逻辑 surface_loop / middle_loop / ultimate_loop（A93.2），优先级抢占 high_priority_task → interrupt(current_task)（A93.3），资源分配 cpu_budget = allocate(tasks, cpu_limit=0.20)（A93.4），调度频率控制 Δt_schedule = max(Δt_min, adaptive_dt)（A93.5），封装为MainScheduler数据类组合GlobalDynamics。主循环是系统的心跳——每步100毫秒，三层循环按时间尺度切换，安全任务100%抢占，CPU预算严格控制在20%以内。下一步：阶段94将建立七大模块集成，将所有模块对接到主循环调度器。
