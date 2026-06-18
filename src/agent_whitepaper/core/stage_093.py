"""阶段93：主循环调度器。

本阶段定义调度器主循环、阶段切换逻辑、优先级抢占、
资源分配和调度频率控制，封装为MainScheduler数据类组合GlobalDynamics。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable
from enum import Enum
import numpy as np

from .stage_092 import GlobalDynamics, GlobalDynamicsCalculator, SystemDynamicsState


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

    def __post_init__(self):
        if self.step_count < 0:
            raise ValueError(f"step_count必须 >= 0，收到 {self.step_count}")
        total_alloc = sum(self.module_allocations.values())
        if total_alloc > self.cpu_budget_total + 1e-9:
            raise ValueError(
                f"资源分配总和({total_alloc})不能超过预算({self.cpu_budget_total})"
            )
        if not (self.dt_min <= self.dt_schedule <= self.dt_max):
            raise ValueError(
                f"dt_schedule必须在[{self.dt_min}, {self.dt_max}]，"
                f"收到 {self.dt_schedule}"
            )


# ============================================================
# 阶段93 核心接口：主循环调度器
# ============================================================
class MainSchedulerExecutor:
    """阶段93的核心接口：主循环调度器。

    编排五类动力学的执行节奏，实现三层循环调度。

    核心约束：
        - 单步循环时间 <= 100ms
        - CPU预算总和 <= 20%
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

        perceive -> assess -> decide -> execute -> evaluate

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
            >>> executor.check_preempt(TaskPriority.P0_EMERGENCY, TaskPriority.P2_NORMAL)
            True
            >>> executor.check_preempt(TaskPriority.P2_NORMAL, TaskPriority.P0_EMERGENCY)
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
