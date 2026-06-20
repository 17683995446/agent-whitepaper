"""阶段G-2：全系统编排器 - 将A-F所有能力编排为统一永续自主进化系统。

第一性原理：
    整体大于部分之和。阶段A-F各自实现了原子能力，但它们是分散的。
    编排器的职责是让这些能力形成一个**闭环**：

    感知(任务生成) → 驱动(能力评估) → 协作(联盟形成) → 行动(任务执行)
    → 学习(Q学习+技能抽象) → 增强(知识蒸馏+技能注入) → 进化(退化检测+恢复)
    → 持久化(检查点) → 回到感知

    这个闭环一旦启动，系统就能：
    1. 自主学会操作多个agent完成任务（阶段C联盟 + 阶段F任务生成）
    2. 自主学会增强自己的技能（阶段B技能抽象 + 阶段E知识蒸馏）
    3. 自主学会自己驱动自己完成长任务（阶段B长任务 + 阶段F ZPD）
    4. 永远运行不停止（阶段D永续运行时 + 阶段F自主恢复）

    编排器是系统的"大脑皮层"，协调各脑区协同工作。
"""

import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from agent_whitepaper._logging import get_logger

# 阶段A：单Agent闭环
from agent_whitepaper.stage_a.agent import SelfDrivenAgent
from agent_whitepaper.stage_a.environment import GridWorld
from agent_whitepaper.stage_b.meta_learner import MetaLearner

# 阶段B：技能增强与长任务
from agent_whitepaper.stage_b.skill_abstractor import SkillAbstractor, SkillLibrary
from agent_whitepaper.stage_b.subgoal_manager import LongTaskExecutor
from agent_whitepaper.stage_c.coalition import CoalitionFormer, CoalitionStatus
from agent_whitepaper.stage_c.message_bus import MessageBus

# 阶段C：多Agent协作
from agent_whitepaper.stage_c.registry import AgentCapability, AgentRegistry
from agent_whitepaper.stage_d.health_monitor import HealthMonitor, HealthStatus

# 阶段D：永续运行时
from agent_whitepaper.stage_d.persistence import PersistenceEngine
from agent_whitepaper.stage_d.scheduler import TaskScheduler

# 阶段E：技能迁移与真增强
from agent_whitepaper.stage_e.knowledge_distiller import CollectiveLearningCoordinator

# 阶段F：自主进化
from agent_whitepaper.stage_f.autonomous_generator import AutonomousTaskGenerator
from agent_whitepaper.stage_f.autonomous_retrainer import (
    AutonomousEvolutionController,
)

# 阶段G：可观测性
from .metrics import MetricsCollector, MetricsSnapshot


# ============================================================
# 编排器配置
# ============================================================
@dataclass
class OrchestratorConfig:
    """全系统编排配置。"""
    grid_size: int = 5
    n_agents: int = 3
    db_path: str = ":memory:"          # 默认内存数据库（测试用）
    checkpoint_interval: int = 50      # 每N轮迭代检查点
    skill_abstract_interval: int = 5   # 每N轮抽象技能
    meta_learn_interval: int = 10      # 每N轮元学习
    distill_interval: int = 30         # 每N轮知识蒸馏
    evolve_interval: int = 20          # 每N轮进化检测
    coalition_interval: int = 15       # 每N轮尝试联盟协作
    max_iterations: Optional[int] = None
    seed: Optional[int] = 42


# ============================================================
# 编排器状态
# ============================================================
class OrchestratorStatus(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    RECOVERING = "recovering"
    STOPPED = "stopped"


# ============================================================
# 全系统编排器
# ============================================================
logger = get_logger(__name__)


class SystemOrchestrator:
    """全系统编排器：将A-F所有能力编排为统一永续自主进化系统。

    核心方法：
    - initialize(): 初始化所有组件
    - run_iteration(): 执行一次完整闭环迭代
    - run(): 永续运行主循环
    - get_metrics(): 获取可观测性指标
    - stop(): 优雅停止
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None) -> None:
        self.config = config or OrchestratorConfig()
        self.status = OrchestratorStatus.INITIALIZING
        self._should_stop = False
        self._iteration = 0
        self._start_time = time.time()

        # 阶段A：Agent与环境
        self.agents: Dict[str, SelfDrivenAgent] = {}
        self.env: Optional[GridWorld] = None

        # 阶段B：技能与学习
        self.skill_library = SkillLibrary()
        self.skill_abstractor = SkillAbstractor(min_n=2, max_n=4, min_frequency=2)
        self.meta_learner = MetaLearner(window_size=20)
        self.task_executor = LongTaskExecutor(max_retries=2)

        # 阶段C：多Agent协作
        self.registry = AgentRegistry(heartbeat_timeout=60.0)
        self.message_bus = MessageBus()
        self.coalition_former: Optional[CoalitionFormer] = None

        # 阶段D：永续运行时
        self.persistence = PersistenceEngine(self.config.db_path)
        self.health_monitor = HealthMonitor(heartbeat_timeout=30.0)
        self.scheduler = TaskScheduler(self.persistence)

        # 阶段E：知识迁移
        self.collective: Optional[CollectiveLearningCoordinator] = None

        # 阶段F：自主进化
        self.task_generator: Optional[AutonomousTaskGenerator] = None
        self.evolution_controller: Optional[AutonomousEvolutionController] = None

        # 阶段G：可观测性
        self.metrics = MetricsCollector(max_history=10000)

        # 累计统计
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._recoveries = 0

    # --------------------------------------------------------
    # 初始化
    # --------------------------------------------------------
    def initialize(self) -> None:
        """初始化所有组件，构建完整系统。"""
        self.status = OrchestratorStatus.INITIALIZING
        cfg = self.config

        # 1. 创建环境
        self.env = GridWorld(
            size=cfg.grid_size,
            goal=(cfg.grid_size - 1, cfg.grid_size - 1),
            start=(0, 0),
            max_steps=50,
            seed=cfg.seed,
        )

        # 2. 创建Agent并注册（阶段A + 阶段C）
        for i in range(cfg.n_agents):
            agent_id = f"agent_{i:04d}"
            agent = SelfDrivenAgent(
                self.env.action_space,
                n_states=cfg.grid_size ** 2,
                learning_rate=0.3,
                seed=cfg.seed + i if cfg.seed else None,
            )
            self.agents[agent_id] = agent

            # 注册到注册中心（阶段C）
            capability = AgentCapability(
                specialization="navigation",
                performance_score=0.0,
                drive_profile=(
                    agent.drives.safety,
                    agent.drives.social,
                    agent.drives.exploration,
                ),
            )
            self.registry.register(
                name=f"Navigator-{i}",
                capability=capability,
                address=agent_id,
            )
            self.message_bus.register_agent(agent_id)
            self.health_monitor.register_component(agent_id)

        # 3. 初始化协作组件（阶段C）
        self.coalition_former = CoalitionFormer(
            self.registry, self.message_bus
        )

        # 4. 初始化知识迁移组件（阶段E）
        self.collective = CollectiveLearningCoordinator(grid_size=cfg.grid_size)

        # 5. 初始化自主进化组件（阶段F）
        self.task_generator = AutonomousTaskGenerator(
            grid_size=cfg.grid_size, seed=cfg.seed
        )
        self.evolution_controller = AutonomousEvolutionController(
            grid_size=cfg.grid_size,
            collective_coordinator=self.collective,
        )

        # 6. 注册健康组件（阶段D）
        self.health_monitor.register_component("orchestrator")
        self.health_monitor.register_component("scheduler")
        self.health_monitor.register_component("persistence")

        self.status = OrchestratorStatus.RUNNING
        self.persistence.log_event("orchestrator_init", message="全系统初始化完成")

    # --------------------------------------------------------
    # 单次迭代：完整闭环
    # --------------------------------------------------------
    def run_iteration(self) -> bool:
        """执行一次完整闭环迭代。

        闭环：感知→驱动→协作→行动→学习→增强→进化→持久化

        Returns:
            是否成功完成
        """
        if self.status != OrchestratorStatus.RUNNING:
            return False

        success = False
        try:
            self._iteration += 1
            it = self._iteration

            # 1. 健康检查（阶段D）
            self._health_check()

            # 2. 自主任务生成与执行（阶段F + 阶段B + 阶段A）
            self._generate_and_execute_tasks()

            # 3. 技能抽象（阶段B）
            if it % self.config.skill_abstract_interval == 0:
                self._abstract_skills()

            # 4. 元学习（阶段B）
            if it % self.config.meta_learn_interval == 0:
                self._meta_learn()

            # 5. 多Agent联盟协作（阶段C）
            if it % self.config.coalition_interval == 0 and len(self.agents) > 1:
                self._try_coalition()

            # 6. 知识蒸馏与迁移（阶段E）
            if it % self.config.distill_interval == 0 and len(self.agents) > 1:
                self._distill_knowledge()

            # 7. 自主进化检测与恢复（阶段F）
            if it % self.config.evolve_interval == 0:
                self._evolve()

            # 8. 检查点（阶段D）
            if it % self.config.checkpoint_interval == 0:
                self._checkpoint()

            # 9. 采集指标（阶段G）
            self._collect_metrics()

            success = True
            self.health_monitor.heartbeat("orchestrator")

        except Exception as e:  # 防火墙：永续运行保障，单步失败不终止主循环
            self.health_monitor.report_failure(
                "orchestrator", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            )
            self._recoveries += 1
            self.status = OrchestratorStatus.RUNNING  # 保持运行

        return success

    # --------------------------------------------------------
    # 闭环各阶段实现
    # --------------------------------------------------------
    def _health_check(self) -> None:
        """阶段D：健康检查与自动恢复。"""
        checks = self.health_monitor.check_all()
        for check in checks:
            if check.status in (HealthStatus.UNHEALTHY, HealthStatus.DEAD):
                self.health_monitor.attempt_recovery(check.component)
        # Agent心跳
        for agent_id in self.agents:
            self.health_monitor.heartbeat(agent_id)
        self.health_monitor.heartbeat("scheduler")
        self.health_monitor.heartbeat("persistence")

    def _generate_and_execute_tasks(self) -> None:
        """阶段F+阶段B+阶段A：自主生成任务并执行。"""
        for agent_id, agent in self.agents.items():
            try:
                # 评估能力（阶段F）
                assessment = self.task_generator.assess_capability(
                    success_rate=agent.stats.success_rate,
                    avg_reward=agent.stats.avg_reward,
                    avg_steps=float(np.mean(agent.stats.recent_rewards)) if agent.stats.recent_rewards else 0.0,
                    skill_count=len(self.skill_library),
                    exploration_coverage=0.0,
                )

                # 生成ZPD任务（阶段F）
                task = self.task_generator.generate_for_capability(assessment)

                # 执行任务（阶段B + 阶段A）
                progress = self.task_executor.execute_task(
                    task, agent, self.env
                )

                # 记录episode统计（task_executor直接调用perceive_and_act，
                # 不经过run_episode，需手动记录episode计数）
                proxy_fe = (
                    agent.stats.free_energy_history[-1]
                    if agent.stats.free_energy_history else 0.0
                )
                agent.stats.record_episode(
                    progress.total_reward, progress.is_complete, proxy_fe
                )

                # 记录结果（阶段F自适应）
                self.task_generator.record_task_result(
                    assessment.zone_of_proximal_development,
                    progress.is_complete,
                )

                if progress.is_complete:
                    self._tasks_completed += 1
                else:
                    self._tasks_failed += 1

                # 更新注册中心性能（阶段C）
                reg = self.registry.get_agent(agent_id)
                if reg:
                    new_perf = 0.3 * reg.capability.performance_score + 0.7 * agent.stats.success_rate
                    reg.capability = AgentCapability(
                        skills=reg.capability.skills,
                        specialization=reg.capability.specialization,
                        performance_score=float(np.clip(new_perf, 0.0, 1.0)),
                        drive_profile=reg.capability.drive_profile,
                    )

            except Exception:  # 防火墙：单Agent失败隔离，不影响其他Agent
                # 单个Agent失败不影响其他
                continue

    def _abstract_skills(self) -> None:
        """阶段B：从成功情节抽象技能。"""
        try:
            for agent in self.agents.values():
                successful = [
                    ep for ep in agent.memory.get_recent_episodes(20) if ep.success
                ]
                if successful:
                    self.skill_abstractor.abstract_from_episodes(
                        successful, self.skill_library
                    )
        except Exception as e:  # 组件降级：技能抽象失败不影响主循环
            logger.debug("技能抽象失败: %s", e, exc_info=True)

    def _meta_learn(self) -> None:
        """阶段B：元学习参数调整。"""
        try:
            for agent in self.agents.values():
                n = min(
                    len(agent.stats.recent_rewards),
                    len(agent.stats.recent_successes),
                    len(agent.stats.free_energy_history),
                )
                for i in range(n):
                    self.meta_learner.record_episode(
                        agent.stats.recent_rewards[i],
                        agent.stats.recent_successes[i],
                        agent.stats.free_energy_history[i],
                    )
                if self.meta_learner.should_adapt():
                    self.meta_learner.adapt_parameters()
                    # 应用元参数到Agent
                    params = self.meta_learner.params
                    agent.q_table.lr = params.learning_rate
        except Exception as e:  # 组件降级：元学习失败不影响主循环
            logger.debug("元学习失败: %s", e, exc_info=True)

    def _try_coalition(self) -> None:
        """阶段C：尝试为复杂任务组建联盟。"""
        try:
            # 生成一个困难任务
            from agent_whitepaper.stage_f.autonomous_generator import TaskDifficulty
            task = self.task_generator._generate_task_by_difficulty(TaskDifficulty.HARD)
            candidate_ids = list(self.agents.keys())
            coalition = self.coalition_former.form_coalition(task, candidate_ids)
            if coalition is not None:
                # 联盟成员共同执行（简化：每个成员各执行一次以加速学习）
                for member_id in coalition.members:
                    agent = self.agents.get(member_id)
                    if agent:
                        self.task_executor.execute_task(task, agent, self.env)
                coalition.status = CoalitionStatus.COMPLETED
        except Exception as e:  # 组件降级：联盟组建失败不影响主循环
            logger.debug("联盟组建失败: %s", e, exc_info=True)

    def _distill_knowledge(self) -> None:
        """阶段E：跨Agent知识蒸馏。"""
        try:
            # 收集所有Agent的知识
            for agent_id, agent in self.agents.items():
                perf = agent.stats.success_rate
                self.collective.collect_from_agent(agent_id, agent.q_table, perf)

            # 找到最弱的Agent，用集体知识教导它
            weakest_id = min(
                self.agents.keys(),
                key=lambda aid: self.agents[aid].stats.success_rate,
            )
            weakest = self.agents[weakest_id]
            distill_result, injection_results = self.collective.teach_new_agent(
                weakest.q_table,
                strategy="weighted",
                inject_skills=True,
                skill_library=self.skill_library,
            )
            self.metrics.increment("distillation_count")
            self.metrics.increment(
                "injected_states",
                getattr(distill_result, "states_distilled", 0),
            )
        except Exception as e:  # 组件降级：知识蒸馏失败不影响主循环
            logger.debug("知识蒸馏失败: %s", e, exc_info=True)

    def _evolve(self) -> None:
        """阶段F：自主进化检测与恢复。"""
        try:
            # 对每个Agent执行进化步骤
            for agent_id, agent in self.agents.items():
                log = self.evolution_controller.evolution_step(agent, self.env)
                if log.get("recovery_strategy", "none") != "none":
                    self.metrics.increment("recovery_count")
        except Exception as e:  # 组件降级：自主进化失败不影响主循环
            logger.debug("自主进化失败: %s", e, exc_info=True)

    def _checkpoint(self) -> None:
        """阶段D：创建检查点。"""
        try:
            snapshot = {
                "iteration": self._iteration,
                "tasks_completed": self._tasks_completed,
                "tasks_failed": self._tasks_failed,
                "skill_count": len(self.skill_library),
                "agent_count": len(self.agents),
                "health_report": self.health_monitor.get_health_report(),
            }
            self.persistence.create_checkpoint(f"迭代 {self._iteration}", snapshot)
        except Exception as e:  # 组件降级：检查点失败不影响主循环
            logger.debug("检查点创建失败: %s", e, exc_info=True)

    def _collect_metrics(self) -> None:
        """阶段G：采集指标快照。"""
        try:
            agents = list(self.agents.values())
            avg_success = (
                float(np.mean([a.stats.success_rate for a in agents]))
                if agents else 0.0
            )
            avg_reward = (
                float(np.mean([a.stats.avg_reward for a in agents]))
                if agents else 0.0
            )
            total_q = int(sum(
                int(np.count_nonzero(a.q_table._table)) for a in agents
            ))
            total_episodes = sum(a.stats.total_episodes for a in agents)
            total_steps = sum(a.stats.total_steps for a in agents)

            # 技能统计
            skill_count = len(self.skill_library)
            composite_count = sum(
                1 for s in self.skill_library._skills.values()
                if s.abstraction_level >= 1
            )

            # 元参数
            params = self.meta_learner.params

            # 协作统计
            mb_stats = self.message_bus.get_stats()
            active_coalitions = sum(
                1 for c in self.coalition_former._coalitions.values()
                if c.status == CoalitionStatus.ACTIVE
            ) if self.coalition_former else 0

            # 健康统计
            health_report = self.health_monitor.get_health_report()
            health_score = health_report.get("overall_health_score", 1.0)

            snapshot = MetricsSnapshot(
                iteration=self._iteration,
                avg_success_rate=avg_success,
                avg_reward=avg_reward,
                total_q_entries=total_q,
                total_episodes=total_episodes,
                total_steps=total_steps,
                skill_count=skill_count,
                composite_skill_count=composite_count,
                exploration_rate=params.exploration_rate,
                learning_rate=params.learning_rate,
                active_coalitions=active_coalitions,
                total_messages=mb_stats.get("total_sent", 0),
                avg_role_match=0.0,
                distillation_count=self.metrics.get_cumulative("distillation_count"),
                migration_count=self.metrics.get_cumulative("migration_count"),
                injected_states=self.metrics.get_cumulative("injected_states"),
                health_score=health_score,
                degradation_detected=False,
                recovery_count=self.metrics.get_cumulative("recovery_count"),
                uptime=time.time() - self._start_time,
                tasks_completed=self._tasks_completed,
                tasks_failed=self._tasks_failed,
                status=self.status.value,
            )
            self.metrics.record(snapshot)
        except Exception as e:  # 组件降级：指标采集失败不影响主循环
            logger.debug("指标采集失败: %s", e, exc_info=True)

    # --------------------------------------------------------
    # 永续运行主循环
    # --------------------------------------------------------
    def run(self) -> None:
        """永续运行主循环。"""
        self.initialize()
        self.persistence.log_event("orchestrator_run", message="主循环启动")

        while not self._should_stop:
            if (self.config.max_iterations is not None
                    and self._iteration >= self.config.max_iterations):
                self.stop("max_iterations")
                break
            self.run_iteration()

        self._checkpoint()
        self.status = OrchestratorStatus.STOPPED
        self.persistence.log_event("orchestrator_stop", message="主循环停止")

    def stop(self, reason: str = "manual") -> None:
        """请求停止。"""
        self._should_stop = True
        self.persistence.log_event("stop_request", message=f"停止原因: {reason}")

    # --------------------------------------------------------
    # 查询接口
    # --------------------------------------------------------
    def get_metrics(self) -> Dict[str, Any]:
        """获取完整指标报告。"""
        latest = self.metrics.get_latest()
        return {
            "iteration": self._iteration,
            "status": self.status.value,
            "uptime": time.time() - self._start_time,
            "latest_snapshot": latest.to_dict() if latest else None,
            "success_rate_trend": self.metrics.get_trend("avg_success_rate"),
            "skill_count_trend": self.metrics.get_trend("skill_count"),
            "success_rate_summary": self.metrics.get_summary("avg_success_rate"),
            "skill_count_summary": self.metrics.get_summary("skill_count"),
            "cumulative": {
                "distillation_count": self.metrics.get_cumulative("distillation_count"),
                "migration_count": self.metrics.get_cumulative("migration_count"),
                "injected_states": self.metrics.get_cumulative("injected_states"),
                "recovery_count": self.metrics.get_cumulative("recovery_count"),
            },
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
            "agent_count": len(self.agents),
            "skill_count": len(self.skill_library),
        }

    def get_agent_summaries(self) -> List[Dict[str, Any]]:
        """获取所有Agent的摘要。"""
        result = []
        for agent_id, agent in self.agents.items():
            result.append({
                "agent_id": agent_id,
                "success_rate": round(agent.stats.success_rate, 4),
                "avg_reward": round(agent.stats.avg_reward, 4),
                "total_episodes": agent.stats.total_episodes,
                "total_steps": agent.stats.total_steps,
                "q_entries": int(np.count_nonzero(agent.q_table._table)),
                "learning_rate": round(agent.q_table.lr, 4),
                "drives": {
                    "safety": round(agent.drives.safety, 3),
                    "social": round(agent.drives.social, 3),
                    "exploration": round(agent.drives.exploration, 3),
                },
            })
        return result
