"""阶段B-5：子目标状态机与进度监控。

第一性原理：
    任务分解后，需要一个状态机来管理子目标的执行流程：
    1. 检查前置条件是否满足
    2. 激活子目标
    3. 监控执行进度
    4. 判断完成/失败
    5. 转移到下一个子目标

    进度监控确保Agent"知道自己走到哪了"，
    这是长任务执行的关键——没有进度感知，Agent会在子目标间迷失。
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum
import numpy as np

from .task_decomposer import Task, SubGoal, SubGoalStatus


# ============================================================
# 子目标执行记录
# ============================================================
@dataclass(frozen=True)
class SubGoalExecution:
    """子目标执行记录。

    Attributes:
        goal_id: 子目标ID
        status: 最终状态
        steps_taken: 耗时步数
        reward_earned: 获得奖励
        attempts: 尝试次数
    """
    goal_id: int
    status: SubGoalStatus
    steps_taken: int
    reward_earned: float
    attempts: int


# ============================================================
# 任务进度
# ============================================================
@dataclass
class TaskProgress:
    """任务整体进度。

    Attributes:
        task: 关联的任务
        current_goal_index: 当前子目标索引
        goal_statuses: 各子目标状态
        executions: 子目标执行记录
        total_steps: 总步数
        total_reward: 总奖励
        start_time: 开始时间戳
    """
    task: Task
    current_goal_index: int = 0
    goal_statuses: Dict[int, SubGoalStatus] = field(default_factory=dict)
    executions: List[SubGoalExecution] = field(default_factory=list)
    total_steps: int = 0
    total_reward: float = 0.0
    start_time: float = 0.0

    @property
    def completion_rate(self) -> float:
        """完成率。"""
        if not self.task.sub_goals:
            return 0.0
        completed = sum(1 for s in self.goal_statuses.values() if s == SubGoalStatus.COMPLETED)
        return completed / len(self.task.sub_goals)

    @property
    def is_complete(self) -> bool:
        """任务是否全部完成。"""
        return all(s == SubGoalStatus.COMPLETED for s in self.goal_statuses.values())

    @property
    def current_goal(self) -> Optional[SubGoal]:
        """当前子目标。"""
        if self.current_goal_index >= len(self.task.sub_goals):
            return None
        return self.task.sub_goals[self.current_goal_index]


# ============================================================
# 子目标状态机
# ============================================================
class SubGoalStateMachine:
    """子目标状态机：管理子目标的执行流程。

    状态转移：
    PENDING → (前置满足) → ACTIVE → (目标达成) → COMPLETED → (下一个) → PENDING
                            → (超时) → FAILED → (重试) → ACTIVE
                            → (放弃) → SKIPPED → (下一个) → PENDING
    """

    def __init__(self, max_retries: int = 3):
        if max_retries < 1:
            raise ValueError("max_retries 必须 >= 1")
        self.max_retries = max_retries
        self._retry_counts: Dict[int, int] = {}

    def initialize(self, task: Task) -> TaskProgress:
        """初始化任务进度。"""
        progress = TaskProgress(task=task)
        for sg in task.sub_goals:
            progress.goal_statuses[sg.goal_id] = SubGoalStatus.PENDING
        # 激活第一个子目标
        if task.sub_goals:
            progress.goal_statuses[task.sub_goals[0].goal_id] = SubGoalStatus.ACTIVE
        return progress

    def check_prerequisites(self, goal: SubGoal, progress: TaskProgress) -> bool:
        """检查子目标的前置条件是否满足。"""
        for prereq_id in goal.prerequisites:
            if progress.goal_statuses.get(prereq_id) != SubGoalStatus.COMPLETED:
                return False
        return True

    def check_goal_reached(
        self,
        goal: SubGoal,
        current_state: Tuple[int, ...]
    ) -> bool:
        """检查是否到达子目标。"""
        return current_state == goal.target_state

    def update(
        self,
        progress: TaskProgress,
        current_state: Tuple[int, ...],
        step_reward: float,
        steps_in_current_goal: int
    ) -> TaskProgress:
        """更新任务进度。

        Args:
            progress: 当前进度
            current_state: 当前状态特征
            step_reward: 本步奖励
            steps_in_current_goal: 在当前子目标上花费的步数

        Returns:
            更新后的进度
        """
        progress.total_steps += 1
        progress.total_reward += step_reward

        current_goal = progress.current_goal
        if current_goal is None:
            return progress

        # 检查是否到达当前子目标
        if self.check_goal_reached(current_goal, current_state):
            # 子目标完成
            progress.goal_statuses[current_goal.goal_id] = SubGoalStatus.COMPLETED
            progress.executions.append(SubGoalExecution(
                goal_id=current_goal.goal_id,
                status=SubGoalStatus.COMPLETED,
                steps_taken=steps_in_current_goal,
                reward_earned=current_goal.reward,
                attempts=self._retry_counts.get(current_goal.goal_id, 0) + 1,
            ))
            progress.total_reward += current_goal.reward

            # 转移到下一个子目标
            progress.current_goal_index += 1
            if progress.current_goal_index < len(progress.task.sub_goals):
                next_goal = progress.task.sub_goals[progress.current_goal_index]
                if self.check_prerequisites(next_goal, progress):
                    progress.goal_statuses[next_goal.goal_id] = SubGoalStatus.ACTIVE
                else:
                    # 前置不满足，标记为PENDING等待
                    progress.goal_statuses[next_goal.goal_id] = SubGoalStatus.PENDING

        # 检查是否超时
        elif steps_in_current_goal >= current_goal.max_steps:
            retry_count = self._retry_counts.get(current_goal.goal_id, 0)
            if retry_count < self.max_retries:
                # 重试
                self._retry_counts[current_goal.goal_id] = retry_count + 1
                progress.goal_statuses[current_goal.goal_id] = SubGoalStatus.ACTIVE
            else:
                # 放弃，跳过
                progress.goal_statuses[current_goal.goal_id] = SubGoalStatus.SKIPPED
                progress.executions.append(SubGoalExecution(
                    goal_id=current_goal.goal_id,
                    status=SubGoalStatus.SKIPPED,
                    steps_taken=steps_in_current_goal,
                    reward_earned=0.0,
                    attempts=retry_count + 1,
                ))
                progress.current_goal_index += 1
                if progress.current_goal_index < len(progress.task.sub_goals):
                    next_goal = progress.task.sub_goals[progress.current_goal_index]
                    progress.goal_statuses[next_goal.goal_id] = SubGoalStatus.ACTIVE

        return progress

    def get_progress_report(self, progress: TaskProgress) -> dict:
        """获取进度报告。"""
        completed = sum(1 for s in progress.goal_statuses.values() if s == SubGoalStatus.COMPLETED)
        failed = sum(1 for s in progress.goal_statuses.values() if s == SubGoalStatus.FAILED)
        skipped = sum(1 for s in progress.goal_statuses.values() if s == SubGoalStatus.SKIPPED)
        active = sum(1 for s in progress.goal_statuses.values() if s == SubGoalStatus.ACTIVE)
        pending = sum(1 for s in progress.goal_statuses.values() if s == SubGoalStatus.PENDING)

        return {
            "task_name": progress.task.name,
            "total_sub_goals": len(progress.task.sub_goals),
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "active": active,
            "pending": pending,
            "completion_rate": progress.completion_rate,
            "is_complete": progress.is_complete,
            "total_steps": progress.total_steps,
            "total_reward": progress.total_reward,
            "current_goal": progress.current_goal.description if progress.current_goal else "None",
        }


# ============================================================
# 长任务执行器
# ============================================================
class LongTaskExecutor:
    """长任务执行器：协调Agent执行分解后的长任务。

    工作流程：
    1. 接收分解后的任务
    2. 用状态机管理子目标执行
    3. 对每个子目标，调用Agent执行
    4. 监控进度，必要时重试或跳过
    5. 汇总结果
    """

    def __init__(self, max_retries: int = 3):
        self.state_machine = SubGoalStateMachine(max_retries=max_retries)

    def execute_task(
        self,
        task: Task,
        agent,
        env,
        state_extractor=None
    ) -> TaskProgress:
        """执行完整长任务。

        Args:
            task: 分解后的任务
            agent: SelfDrivenAgent实例
            env: 环境实例
            state_extractor: 从观测提取状态特征的函数

        Returns:
            任务进度
        """
        progress = self.state_machine.initialize(task)

        if state_extractor is None:
            def state_extractor(obs, grid_size=5):
                # 默认：从GridWorld观测提取agent位置
                agent_part = obs[:grid_size * grid_size]
                pos_idx = int(np.argmax(agent_part))
                return (pos_idx // grid_size, pos_idx % grid_size)

        max_total_steps = sum(sg.max_steps for sg in task.sub_goals) * 2

        for global_step in range(max_total_steps):
            if progress.is_complete:
                break

            current_goal = progress.current_goal
            if current_goal is None:
                break

            # 运行一个情节来尝试完成当前子目标
            obs = env.reset()
            steps_in_goal = 0
            goal_start_reward = progress.total_reward

            for step in range(current_goal.max_steps):
                # Agent感知并行动
                grid_size = int(np.sqrt(len(obs) // 2))
                state_idx = agent.q_table.state_to_index(obs, grid_size)
                action, next_obs, reward, done, info = agent.perceive_and_act(
                    obs, state_idx, env
                )
                steps_in_goal += 1
                obs = next_obs

                # 检查子目标是否达成
                current_state = state_extractor(obs, grid_size)
                progress = self.state_machine.update(
                    progress, current_state, reward, steps_in_goal
                )

                if progress.current_goal != current_goal:
                    break  # 子目标已完成或跳过，进入下一个

                if done:
                    break

        return progress
