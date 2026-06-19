"""阶段D-2：任务调度器 - 持续任务生成与分配。

第一性原理：
    "永远运行不停止"需要持续有任务可做。
    调度器是系统的"心跳"：
    1. 持续生成新任务（自主任务生成）
    2. 将任务分配给空闲Agent
    3. 管理任务队列（优先级、超时、重试）
    4. 汇总任务结果

    调度器让系统从"运行一次就停"变为"持续运转"。
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Any
from enum import Enum
from collections import deque
import time
import uuid
import numpy as np

from .persistence import PersistenceEngine
from agent_whitepaper.stage_b.task_decomposer import Task, SubGoal, TaskDecomposer


# ============================================================
# 调度任务状态
# ============================================================
class ScheduledTaskStatus(Enum):
    """调度任务状态。"""
    QUEUED = "queued"        # 在队列中等待
    ASSIGNED = "assigned"    # 已分配给Agent
    RUNNING = "running"      # 正在执行
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"        # 失败
    TIMEOUT = "timeout"      # 超时


# ============================================================
# 调度任务
# ============================================================
@dataclass
class ScheduledTask:
    """一个调度任务。

    Attributes:
        task_id: 任务ID
        task: 实际任务对象
        priority: 优先级（0最高）
        status: 当前状态
        assigned_agent_id: 分配的Agent ID
        created_at: 创建时间
        started_at: 开始执行时间
        completed_at: 完成时间
        result: 执行结果
        retry_count: 重试次数
        max_retries: 最大重试次数
        timeout: 超时时间（秒）
    """
    task_id: str
    task: Task
    priority: int = 5
    status: ScheduledTaskStatus = ScheduledTaskStatus.QUEUED
    assigned_agent_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict] = None
    retry_count: int = 0
    max_retries: int = 2
    timeout: float = 60.0


# ============================================================
# 任务生成器
# ============================================================
class TaskGenerator:
    """任务生成器：持续生成新任务。

    生成策略：
    1. 随机路径任务：随机起终点
    2. 难度递增：根据Agent能力调整难度
    3. 探索任务：访问未探索区域
    4. 复合任务：多目标访问
    """

    def __init__(self, grid_size: int = 5, seed: Optional[int] = None):
        self.grid_size = grid_size
        self._rng = np.random.RandomState(seed)
        self.decomposer = TaskDecomposer(grid_size=grid_size)
        self._task_counter = 0
        self._difficulty_level = 1

    def generate_random_path_task(self) -> Task:
        """生成随机路径任务。"""
        self._task_counter += 1
        # 随机起终点（确保不同）
        start = (
            int(self._rng.randint(0, self.grid_size)),
            int(self._rng.randint(0, self.grid_size))
        )
        end = start
        while end == start:
            end = (
                int(self._rng.randint(0, self.grid_size)),
                int(self._rng.randint(0, self.grid_size))
            )
        # 根据难度决定航点数
        n_waypoints = min(self._difficulty_level - 1, 3)
        return self.decomposer.decompose_path_task(
            start=start, end=end, n_waypoints=max(0, n_waypoints),
            task_name=f"auto_path_{self._task_counter}"
        )

    def generate_multi_target_task(self, n_targets: int = 3) -> Task:
        """生成多目标访问任务。"""
        self._task_counter += 1
        targets = []
        while len(targets) < n_targets:
            t = (
                int(self._rng.randint(0, self.grid_size)),
                int(self._rng.randint(0, self.grid_size))
            )
            if t not in targets:
                targets.append(t)
        return self.decomposer.decompose_multi_target_task(
            targets=targets,
            task_name=f"auto_multi_{self._task_counter}"
        )

    def generate_exploration_task(self) -> Task:
        """生成探索任务（访问随机点）。"""
        return self.generate_random_path_task()

    def generate_task(self) -> Task:
        """根据当前难度生成任务。"""
        task_type = self._rng.choice(["path", "multi", "explore"], p=[0.5, 0.3, 0.2])
        if task_type == "path":
            return self.generate_random_path_task()
        elif task_type == "multi":
            # 确保至少2个目标
            n_targets = max(2, min(3, self._difficulty_level))
            return self.generate_multi_target_task(n_targets=n_targets)
        else:
            return self.generate_exploration_task()

    def increase_difficulty(self) -> None:
        """提升难度。"""
        self._difficulty_level = min(self._difficulty_level + 1, 5)

    def decrease_difficulty(self) -> None:
        """降低难度。"""
        self._difficulty_level = max(self._difficulty_level - 1, 1)

    @property
    def current_difficulty(self) -> int:
        return self._difficulty_level


# ============================================================
# 任务调度器
# ============================================================
class TaskScheduler:
    """任务调度器：管理任务队列和分配。

    核心功能：
    1. 维护任务优先级队列
    2. 将任务分配给空闲Agent
    3. 监控任务超时
    4. 处理失败重试
    5. 持久化任务状态
    """

    def __init__(
        self,
        persistence: Optional[PersistenceEngine] = None,
        max_queue_size: int = 100
    ):
        self.persistence = persistence
        self.max_queue_size = max_queue_size
        self._task_queue: deque = deque()
        self._running_tasks: Dict[str, ScheduledTask] = {}
        self._completed_tasks: List[ScheduledTask] = []
        self._failed_tasks: List[ScheduledTask] = []
        self._task_generator = TaskGenerator()
        self._stats = {
            "total_generated": 0,
            "total_completed": 0,
            "total_failed": 0,
            "total_retries": 0,
        }

    def submit_task(self, task: Task, priority: int = 5) -> str:
        """提交任务到队列。"""
        if len(self._task_queue) >= self.max_queue_size:
            # 队列满，丢弃最低优先级任务
            return ""
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        scheduled = ScheduledTask(
            task_id=task_id,
            task=task,
            priority=priority,
        )
        # 按优先级插入
        inserted = False
        for i, existing in enumerate(self._task_queue):
            if priority < existing.priority:
                self._task_queue.insert(i, scheduled)
                inserted = True
                break
        if not inserted:
            self._task_queue.append(scheduled)
        self._stats["total_generated"] += 1

        # 持久化
        if self.persistence:
            self.persistence.save_task_progress(
                task_id=task_id,
                task_name=task.name,
                task_data={"description": task.description},
                current_goal_index=0,
                goal_statuses={},
                total_steps=0,
                total_reward=0.0,
                status="queued"
            )
        return task_id

    def auto_generate_task(self) -> str:
        """自动生成并提交任务。"""
        task = self._task_generator.generate_task()
        return self.submit_task(task, priority=5)

    def get_next_task(self, agent_id: str) -> Optional[ScheduledTask]:
        """获取下一个待执行任务并分配给Agent。"""
        if not self._task_queue:
            return None
        scheduled = self._task_queue.popleft()
        scheduled.status = ScheduledTaskStatus.ASSIGNED
        scheduled.assigned_agent_id = agent_id
        scheduled.started_at = time.time()
        self._running_tasks[scheduled.task_id] = scheduled
        return scheduled

    def complete_task(self, task_id: str, result: Dict) -> bool:
        """标记任务完成。"""
        if task_id not in self._running_tasks:
            return False
        scheduled = self._running_tasks.pop(task_id)
        scheduled.status = ScheduledTaskStatus.COMPLETED
        scheduled.completed_at = time.time()
        scheduled.result = result
        self._completed_tasks.append(scheduled)
        self._stats["total_completed"] += 1

        # 持久化
        if self.persistence:
            self.persistence.save_task_progress(
                task_id=task_id,
                task_name=scheduled.task.name,
                task_data={"result": result},
                current_goal_index=result.get("current_goal_index", 0),
                goal_statuses=result.get("goal_statuses", {}),
                total_steps=result.get("total_steps", 0),
                total_reward=result.get("total_reward", 0.0),
                status="completed"
            )
        return True

    def fail_task(self, task_id: str, reason: str = "") -> bool:
        """标记任务失败，处理重试。"""
        if task_id not in self._running_tasks:
            return False
        scheduled = self._running_tasks.pop(task_id)
        scheduled.retry_count += 1
        self._stats["total_retries"] += 1

        if scheduled.retry_count <= scheduled.max_retries:
            # 重试：重新入队
            scheduled.status = ScheduledTaskStatus.QUEUED
            scheduled.assigned_agent_id = None
            scheduled.started_at = None
            self._task_queue.append(scheduled)
            return True
        else:
            # 彻底失败
            scheduled.status = ScheduledTaskStatus.FAILED
            scheduled.completed_at = time.time()
            scheduled.result = {"error": reason}
            self._failed_tasks.append(scheduled)
            self._stats["total_failed"] += 1
            return False

    def check_timeouts(self) -> List[str]:
        """检查超时任务，返回超时的task_id列表。"""
        current_time = time.time()
        timed_out = []
        for task_id, scheduled in list(self._running_tasks.items()):
            if scheduled.started_at is None:
                continue
            elapsed = current_time - scheduled.started_at
            if elapsed > scheduled.timeout:
                scheduled.status = ScheduledTaskStatus.TIMEOUT
                timed_out.append(task_id)
                # 超时视为失败
                self.fail_task(task_id, reason="timeout")
        return timed_out

    def get_queue_size(self) -> int:
        """获取队列大小。"""
        return len(self._task_queue)

    def get_running_count(self) -> int:
        """获取正在运行的任务数。"""
        return len(self._running_tasks)

    def ensure_queue_filled(self, min_size: int = 3) -> int:
        """确保队列至少有min_size个任务。

        Returns:
            新生成的任务数
        """
        generated = 0
        while len(self._task_queue) < min_size:
            self.auto_generate_task()
            generated += 1
        return generated

    def get_stats(self) -> Dict[str, Any]:
        """获取调度器统计。"""
        return {
            **self._stats,
            "queue_size": len(self._task_queue),
            "running_count": len(self._running_tasks),
            "completed_count": len(self._completed_tasks),
            "failed_count": len(self._failed_tasks),
            "current_difficulty": self._task_generator.current_difficulty,
        }

    def get_recent_results(self, n: int = 10) -> List[Dict]:
        """获取最近n个完成的任务结果。"""
        recent = self._completed_tasks[-n:]
        return [
            {
                "task_id": s.task_id,
                "task_name": s.task.name,
                "result": s.result,
                "duration": (s.completed_at - s.started_at) if s.completed_at and s.started_at else 0,
            }
            for s in recent
        ]
