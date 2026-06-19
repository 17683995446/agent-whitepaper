"""阶段F-1：自主任务生成器 - 根据Agent能力自适应生成任务。

第一性原理：
    "永远运行不停止"需要持续有合适的任务。
    但任务不能太简单（学不到东西）也不能太难（总是失败）。
    自主任务生成器根据Agent当前能力，生成"最近发展区"的任务：
    - 比当前能力略难一点
    - 成功率约50-70%（最有学习价值）
    - 随能力提升自动增加难度

    这是"自主驱动自己"的真正实现：系统自己决定下一步做什么。
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum
import numpy as np

from agent_whitepaper.stage_b.task_decomposer import Task, TaskDecomposer


# ============================================================
# 任务难度等级
# ============================================================
class TaskDifficulty(Enum):
    """任务难度等级。"""
    TRIVIAL = 1     # 平凡（距离1）
    EASY = 2        # 简单（距离2-3）
    MEDIUM = 3      # 中等（距离4-6）
    HARD = 4        # 困难（距离7-10，带航点）
    EXTREME = 5     # 极端（多目标访问）


# ============================================================
# Agent能力评估
# ============================================================
@dataclass(frozen=True)
class AgentCapabilityAssessment:
    """Agent能力评估结果。

    Attributes:
        success_rate: 近期成功率
        avg_reward: 平均奖励
        avg_steps: 平均步数
        skill_count: 技能数
        exploration_coverage: 探索覆盖率
        capability_level: 能力等级 [1,5]
    """
    success_rate: float
    avg_reward: float
    avg_steps: float
    skill_count: int
    exploration_coverage: float
    capability_level: int

    @property
    def zone_of_proximal_development(self) -> TaskDifficulty:
        """最近发展区：比当前能力难一级。"""
        target_level = min(self.capability_level + 1, 5)
        return TaskDifficulty(target_level)


# ============================================================
# 自主任务生成器
# ============================================================
class AutonomousTaskGenerator:
    """自主任务生成器：根据Agent能力生成合适难度的任务。

    核心算法：
    1. 评估Agent当前能力
    2. 确定最近发展区（ZPD）
    3. 生成ZPD难度的任务
    4. 根据反馈调整难度
    """

    def __init__(self, grid_size: int = 5, seed: Optional[int] = None):
        self.grid_size = grid_size
        self._rng = np.random.RandomState(seed)
        self.decomposer = TaskDecomposer(grid_size=grid_size)
        self._task_counter = 0
        self._difficulty_history: List[Tuple[TaskDifficulty, float]] = []
        self._adaptive_level = 2  # 自适应难度等级

    def assess_capability(
        self,
        success_rate: float,
        avg_reward: float,
        avg_steps: float,
        skill_count: int,
        exploration_coverage: float = 0.0
    ) -> AgentCapabilityAssessment:
        """评估Agent能力等级。

        能力等级规则：
        - 1级: 成功率<30%
        - 2级: 成功率30-50%
        - 3级: 成功率50-70%
        - 4级: 成功率70-90%
        - 5级: 成功率>90%
        """
        if success_rate < 0.3:
            level = 1
        elif success_rate < 0.5:
            level = 2
        elif success_rate < 0.7:
            level = 3
        elif success_rate < 0.9:
            level = 4
        else:
            level = 5

        # 技能数加成
        if skill_count > 20 and level < 5:
            level = min(level + 1, 5)

        return AgentCapabilityAssessment(
            success_rate=success_rate,
            avg_reward=avg_reward,
            avg_steps=avg_steps,
            skill_count=skill_count,
            exploration_coverage=exploration_coverage,
            capability_level=level,
        )

    def generate_for_capability(
        self,
        assessment: AgentCapabilityAssessment
    ) -> Task:
        """根据能力评估生成任务。"""
        zpd = assessment.zone_of_proximal_development
        return self._generate_task_by_difficulty(zpd)

    def _generate_task_by_difficulty(self, difficulty: TaskDifficulty) -> Task:
        """按难度生成任务。"""
        self._task_counter += 1

        if difficulty == TaskDifficulty.TRIVIAL:
            # 距离1的简单任务
            start = (0, 0)
            end = (0, 1) if self.grid_size > 1 else (1, 0)
            return self.decomposer.decompose_path_task(
                start, end, n_waypoints=0, task_name=f"auto_trivial_{self._task_counter}"
            )
        elif difficulty == TaskDifficulty.EASY:
            # 距离2-3
            start = (0, 0)
            end = (min(2, self.grid_size - 1), min(2, self.grid_size - 1))
            return self.decomposer.decompose_path_task(
                start, end, n_waypoints=0, task_name=f"auto_easy_{self._task_counter}"
            )
        elif difficulty == TaskDifficulty.MEDIUM:
            # 距离4-6
            start = (0, 0)
            end = (min(3, self.grid_size - 1), min(3, self.grid_size - 1))
            return self.decomposer.decompose_path_task(
                start, end, n_waypoints=0, task_name=f"auto_medium_{self._task_counter}"
            )
        elif difficulty == TaskDifficulty.HARD:
            # 距离7-10，带航点
            start = (0, 0)
            end = (self.grid_size - 1, self.grid_size - 1)
            return self.decomposer.decompose_path_task(
                start, end, n_waypoints=1, task_name=f"auto_hard_{self._task_counter}"
            )
        else:  # EXTREME
            # 多目标访问
            n_targets = min(3, self.grid_size)
            targets = []
            while len(targets) < n_targets:
                t = (
                    int(self._rng.randint(0, self.grid_size)),
                    int(self._rng.randint(0, self.grid_size))
                )
                if t not in targets:
                    targets.append(t)
            return self.decomposer.decompose_multi_target_task(
                targets, task_name=f"auto_extreme_{self._task_counter}"
            )

    def record_task_result(
        self,
        difficulty: TaskDifficulty,
        success: bool
    ) -> None:
        """记录任务结果，用于自适应难度调整。"""
        self._difficulty_history.append((difficulty, 1.0 if success else 0.0))

        # 每10个任务调整一次难度
        if len(self._difficulty_history) >= 10:
            recent = self._difficulty_history[-10:]
            avg_success = np.mean([s for _, s in recent])
            if avg_success > 0.8:
                # 成功率太高，提升难度
                self._adaptive_level = min(self._adaptive_level + 1, 5)
            elif avg_success < 0.3:
                # 成功率太低，降低难度
                self._adaptive_level = max(self._adaptive_level - 1, 1)

    def generate_adaptive(self) -> Task:
        """生成自适应难度任务。"""
        difficulty = TaskDifficulty(self._adaptive_level)
        return self._generate_task_by_difficulty(difficulty)

    def generate_curriculum(self, n_tasks: int = 10) -> List[Task]:
        """生成课程学习序列：从简单到困难。"""
        curriculum = []
        for i in range(n_tasks):
            # 难度从1到5渐进
            level = min(5, max(1, (i * 5 // n_tasks) + 1))
            difficulty = TaskDifficulty(level)
            curriculum.append(self._generate_task_by_difficulty(difficulty))
        return curriculum

    def generate_exploration_task(
        self,
        visited_states: Optional[set] = None
    ) -> Task:
        """生成探索任务：访问未探索的状态。

        Args:
            visited_states: 已访问的状态集合
        """
        self._task_counter += 1
        if visited_states is None:
            visited_states = set()

        # 找一个未访问的状态作为目标
        all_states = set()
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                all_states.add((r, c))

        unvisited = all_states - visited_states
        if unvisited:
            target = list(unvisited)[self._rng.randint(len(unvisited))]
        else:
            # 全部访问过，随机选一个
            target = (
                int(self._rng.randint(0, self.grid_size)),
                int(self._rng.randint(0, self.grid_size))
            )

        start = (0, 0)
        return self.decomposer.decompose_path_task(
            start, target, n_waypoints=0,
            task_name=f"auto_explore_{self._task_counter}"
        )

    @property
    def current_difficulty_level(self) -> int:
        return self._adaptive_level

    def get_stats(self) -> Dict:
        """获取生成器统计。"""
        if not self._difficulty_history:
            return {"total_generated": self._task_counter, "current_level": self._adaptive_level}
        recent = self._difficulty_history[-10:]
        return {
            "total_generated": self._task_counter,
            "current_level": self._adaptive_level,
            "recent_avg_success": float(np.mean([s for _, s in recent])),
            "recent_difficulties": [d.value for d, _ in recent],
        }
