"""阶段D-4：永续主循环 - daemon模式运行，永远不停止。

第一性原理：
    "永远运行不停止"的核心是一个永续主循环：
    1. 循环执行：感知→调度→执行→学习→检查点→重复
    2. 异常隔离：单个任务失败不影响主循环
    3. 定期检查点：保存状态以便恢复
    4. 健康自检：每轮检查系统健康，必要时自恢复
    5. 优雅停止：收到停止信号时保存状态再退出

    主循环是系统的"心脏"，只要它在跳，系统就活着。
"""

import signal
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from agent_whitepaper.stage_a.agent import SelfDrivenAgent
from agent_whitepaper.stage_a.environment import GridWorld
from agent_whitepaper.stage_b.meta_learner import MetaLearner
from agent_whitepaper.stage_b.skill_abstractor import SkillAbstractor, SkillLibrary
from agent_whitepaper.stage_b.subgoal_manager import LongTaskExecutor

from .health_monitor import HealthMonitor, HealthStatus
from .persistence import PersistenceEngine
from .scheduler import ScheduledTaskStatus, TaskScheduler


# ============================================================
# 运行状态
# ============================================================
class RuntimeStatus(Enum):
    """运行时状态。"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    RECOVERING = "recovering"
    STOPPING = "stopping"
    STOPPED = "stopped"


# ============================================================
# 运行配置
# ============================================================
@dataclass
class RuntimeConfig:
    """永续运行配置。

    Attributes:
        db_path: 持久化数据库路径
        grid_size: 环境网格大小
        n_agents: Agent数量
        checkpoint_interval: 检查点间隔（秒）
        heartbeat_interval: 心跳间隔（秒）
        task_fill_threshold: 任务队列填充阈值
        max_iterations: 最大迭代次数（None=无限）
        iteration_delay: 每轮迭代间隔（秒）
        auto_recover: 是否自动恢复
    """
    db_path: str = "agent_runtime.db"
    grid_size: int = 5
    n_agents: int = 1
    checkpoint_interval: float = 30.0
    heartbeat_interval: float = 5.0
    task_fill_threshold: int = 3
    max_iterations: Optional[int] = None
    iteration_delay: float = 0.0
    auto_recover: bool = True


# ============================================================
# 运行统计
# ============================================================
@dataclass
class RuntimeStats:
    """运行时统计。"""
    start_time: float = field(default_factory=time.time)
    total_iterations: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    total_episodes: int = 0
    total_steps: int = 0
    total_recoveries: int = 0
    last_checkpoint_time: float = field(default_factory=time.time)
    last_heartbeat_time: float = field(default_factory=time.time)

    @property
    def uptime(self) -> float:
        return time.time() - self.start_time


# ============================================================
# 永续运行时
# ============================================================
class PerpetualRuntime:
    """永续运行时：让系统永远运行不停止。

    核心循环：
    while not should_stop:
        1. 健康检查
        2. 填充任务队列
        3. 分配任务给Agent
        4. Agent执行任务
        5. 技能抽象与学习
        6. 元学习参数调整
        7. 定期检查点
        8. 异常处理与恢复
    """

    def __init__(self, config: Optional[RuntimeConfig] = None) -> None:
        self.config = config or RuntimeConfig()
        self.status = RuntimeStatus.INITIALIZING
        self.stats = RuntimeStats()
        self._should_stop = False
        self._stop_reason = ""

        # 核心组件
        self.persistence = PersistenceEngine(self.config.db_path)
        self.scheduler = TaskScheduler(self.persistence)
        self.health_monitor = HealthMonitor(heartbeat_timeout=30.0)
        self.meta_learner = MetaLearner(window_size=20)

        # Agent和环境
        self.agents: Dict[str, SelfDrivenAgent] = {}
        self.env: Optional[GridWorld] = None
        self.skill_library = SkillLibrary()
        self.skill_abstractor = SkillAbstractor(min_n=2, max_n=4, min_frequency=2)
        self.task_executor = LongTaskExecutor(max_retries=2)

        # 注册健康组件
        self.health_monitor.register_component("main_loop", self._recover_main_loop)
        self.health_monitor.register_component("scheduler")
        self.health_monitor.register_component("agent_runner")
        self.health_monitor.register_component("persistence")

        # 记录启动
        self.persistence.log_event("start", message="系统启动")

    def initialize(self) -> None:
        """初始化系统。"""
        self.status = RuntimeStatus.INITIALIZING

        # 创建环境
        self.env = GridWorld(
            size=self.config.grid_size,
            goal=(self.config.grid_size - 1, self.config.grid_size - 1),
            start=(0, 0),
            max_steps=50,
            seed=42
        )

        # 创建Agent
        for i in range(self.config.n_agents):
            agent_id = f"agent_{i:04d}"
            agent = SelfDrivenAgent(
                self.env.action_space,
                n_states=self.config.grid_size ** 2,
                learning_rate=0.3,
                seed=42 + i
            )
            self.agents[agent_id] = agent
            self.health_monitor.register_component(f"agent_{agent_id}")

        # 尝试从持久化恢复
        self._try_recover_state()

        self.status = RuntimeStatus.RUNNING
        self.persistence.log_event("initialized", message="系统初始化完成")

    def _try_recover_state(self) -> None:
        """尝试从持久化存储恢复状态。"""
        try:
            # 恢复Agent状态
            for agent_id in self.agents:
                saved = self.persistence.load_agent_state(agent_id)
                if saved:
                    agent = self.agents[agent_id]
                    agent.q_table._table = saved["q_table"]
                    drives = saved["drives"]
                    from agent_whitepaper.stage_a.policy import DriveState
                    agent.drives = DriveState(
                        safety=drives.get("safety", 0.5),
                        social=drives.get("social", 0.5),
                        exploration=drives.get("exploration", 0.5),
                    )
                    self.persistence.log_event(
                        "recover", agent_id=agent_id,
                        message="Agent状态已恢复"
                    )

            # 恢复技能库
            saved_skills = self.persistence.load_all_skills()
            from agent_whitepaper.stage_b.skill_abstractor import Skill
            for s in saved_skills:
                skill = Skill(
                    name=s["skill_name"],
                    precondition=s["precondition"],
                    action_sequence=s["action_sequence"],
                    postcondition=s["postcondition"],
                    expected_reward=s["expected_reward"],
                    success_count=s["success_count"],
                    total_count=s["total_count"],
                    abstraction_level=s["abstraction_level"],
                )
                self.skill_library.add_skill(skill)
        except Exception as e:
            self.persistence.log_event(
                "recover_error", message=f"状态恢复失败: {e}"
            )

    def _save_state(self) -> None:
        """保存当前状态到持久化存储。"""
        try:
            # 保存Agent状态
            for agent_id, agent in self.agents.items():
                self.persistence.save_agent_state(
                    agent_id=agent_id,
                    name=agent_id,
                    q_table=agent.q_table._table,
                    drives={
                        "safety": agent.drives.safety,
                        "social": agent.drives.social,
                        "exploration": agent.drives.exploration,
                    },
                    stats={
                        "total_steps": agent.stats.total_steps,
                        "total_episodes": agent.stats.total_episodes,
                        "total_reward": agent.stats.total_reward,
                        "successes": agent.stats.successes,
                    },
                    n_states=agent.q_table.n_states,
                    n_actions=agent.q_table.n_actions,
                    learning_rate=agent.q_table.lr,
                    gamma=agent.gamma,
                )
            self.health_monitor.heartbeat("persistence")
        except Exception as e:
            self.health_monitor.report_failure("persistence", str(e))

    def _create_checkpoint(self, description: str = "") -> None:
        """创建检查点。"""
        snapshot = {
            "total_iterations": self.stats.total_iterations,
            "total_tasks_completed": self.stats.total_tasks_completed,
            "total_episodes": self.stats.total_episodes,
            "uptime": self.stats.uptime,
            "scheduler_stats": self.scheduler.get_stats(),
            "health_report": self.health_monitor.get_health_report(),
        }
        self.persistence.create_checkpoint(description, snapshot)
        self.stats.last_checkpoint_time = time.time()
        self.persistence.log_event(
            "checkpoint", message=f"检查点: {description}"
        )

    def _recover_main_loop(self) -> None:
        """主循环恢复处理。"""
        self.status = RuntimeStatus.RECOVERING
        self.persistence.log_event("recover", message="主循环恢复中")
        # 重新初始化关键组件
        if self.env is None:
            self.env = GridWorld(size=self.config.grid_size, seed=42)
        self.status = RuntimeStatus.RUNNING
        self.stats.total_recoveries += 1

    def _execute_one_iteration(self) -> bool:
        """执行一次主循环迭代。

        Returns:
            是否成功执行
        """
        success = False
        try:
            # 1. 健康检查
            if self.config.auto_recover:
                health_checks = self.health_monitor.check_all()
                for check in health_checks:
                    if check.status in (HealthStatus.UNHEALTHY, HealthStatus.DEAD):
                        self.health_monitor.attempt_recovery(check.component)

            # 2. 填充任务队列
            self.scheduler.ensure_queue_filled(self.config.task_fill_threshold)
            self.health_monitor.heartbeat("scheduler")

            # 3. 分配并执行任务
            for agent_id, agent in self.agents.items():
                scheduled = self.scheduler.get_next_task(agent_id)
                if scheduled is None:
                    continue

                scheduled.status = ScheduledTaskStatus.RUNNING
                try:
                    # 执行任务
                    progress = self.task_executor.execute_task(
                        scheduled.task, agent, self.env
                    )
                    report = self.task_executor.state_machine.get_progress_report(progress)

                    if progress.is_complete:
                        self.scheduler.complete_task(scheduled.task_id, {
                            "completion_rate": report["completion_rate"],
                            "total_steps": report["total_steps"],
                            "total_reward": report["total_reward"],
                            "current_goal_index": progress.current_goal_index,
                            "goal_statuses": {
                                k: v.value for k, v in progress.goal_statuses.items()
                            },
                        })
                        self.stats.total_tasks_completed += 1
                    else:
                        self.scheduler.fail_task(
                            scheduled.task_id, "任务未完成"
                        )
                        self.stats.total_tasks_failed += 1

                    self.stats.total_episodes += 1
                    self.stats.total_steps += report["total_steps"]
                    self.health_monitor.heartbeat(f"agent_{agent_id}")

                except Exception as e:
                    self.scheduler.fail_task(scheduled.task_id, str(e))
                    self.health_monitor.report_failure(
                        "agent_runner", f"Agent {agent_id}: {str(e)}"
                    )

            # 4. 技能抽象（每10轮一次）
            if self.stats.total_iterations % 10 == 0:
                self._abstract_skills()

            # 5. 元学习（每20轮一次）
            if self.stats.total_iterations % 20 == 0:
                self._meta_learn()

            # 6. 定期检查点
            if time.time() - self.stats.last_checkpoint_time > self.config.checkpoint_interval:
                self._create_checkpoint(f"迭代 {self.stats.total_iterations}")
                self._save_state()

            success = True
            self.health_monitor.heartbeat("main_loop")

        except Exception as e:
            self.health_monitor.report_failure(
                "main_loop", f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            )
        finally:
            # 无论成功失败，迭代计数都增加（保证永续运行）
            self.stats.total_iterations += 1

        return success

    def _abstract_skills(self) -> None:
        """从Agent经验抽象技能。"""
        try:
            for agent in self.agents.values():
                successful = [
                    ep for ep in agent.memory.get_recent_episodes(20) if ep.success
                ]
                if successful:
                    self.skill_abstractor.abstract_from_episodes(
                        successful, self.skill_library
                    )
        except Exception:
            pass  # 技能抽象失败不影响主循环

    def _meta_learn(self) -> None:
        """元学习参数调整。"""
        try:
            for agent in self.agents.values():
                for i, reward in enumerate(agent.stats.recent_rewards[-20:]):
                    success = agent.stats.recent_successes[i] if i < len(agent.stats.recent_successes) else False
                    fe = agent.stats.free_energy_history[i] if i < len(agent.stats.free_energy_history) else 0.5
                    self.meta_learner.record_episode(reward, success, fe)
                if self.meta_learner.should_adapt():
                    self.meta_learner.adapt_parameters()
        except Exception:
            pass

    def run(self) -> None:
        """永续运行主循环。"""
        self.initialize()

        # 注册信号处理（优雅停止）
        def signal_handler(signum: int, frame: Any) -> None:
            self.stop("signal")
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except (ValueError, AttributeError):
            pass  # 非主线程无法注册信号

        self.persistence.log_event("run_start", message="主循环启动")

        while not self._should_stop:
            # 检查最大迭代限制
            if (self.config.max_iterations is not None and
                    self.stats.total_iterations >= self.config.max_iterations):
                self.stop("max_iterations_reached")
                break

            # 执行一次迭代
            self._execute_one_iteration()

            # 迭代间延迟
            if self.config.iteration_delay > 0:
                time.sleep(self.config.iteration_delay)

        # 优雅停止：保存状态
        self._save_state()
        self._create_checkpoint("优雅停止")
        self.status = RuntimeStatus.STOPPED
        self.persistence.log_event("stop", message=f"停止原因: {self._stop_reason}")
        self.persistence.close()

    def stop(self, reason: str = "manual") -> None:
        """请求系统停止。"""
        self._should_stop = True
        self._stop_reason = reason
        self.status = RuntimeStatus.STOPPING

    def get_status_report(self) -> Dict[str, Any]:
        """获取状态报告。"""
        return {
            "status": self.status.value,
            "uptime": self.stats.uptime,
            "total_iterations": self.stats.total_iterations,
            "total_tasks_completed": self.stats.total_tasks_completed,
            "total_tasks_failed": self.stats.total_tasks_failed,
            "total_episodes": self.stats.total_episodes,
            "total_steps": self.stats.total_steps,
            "total_recoveries": self.stats.total_recoveries,
            "scheduler_stats": self.scheduler.get_stats(),
            "health_report": self.health_monitor.get_health_report(),
            "skill_count": len(self.skill_library),
            "meta_learner_summary": self.meta_learner.get_progress_summary(),
        }
