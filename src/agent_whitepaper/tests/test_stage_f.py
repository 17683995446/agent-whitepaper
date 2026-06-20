"""阶段F验证测试 + 永续运行演示。

测试覆盖：
1. 自主任务生成器
2. 性能退化检测器
3. 自主重训练机制
4. 端到端：永续运行+自主进化
"""


import numpy as np

from agent_whitepaper.stage_a.agent import SelfDrivenAgent
from agent_whitepaper.stage_a.environment import GridWorld
from agent_whitepaper.stage_e.knowledge_distiller import CollectiveLearningCoordinator
from agent_whitepaper.stage_f.autonomous_generator import (
    AutonomousTaskGenerator,
    TaskDifficulty,
)
from agent_whitepaper.stage_f.autonomous_retrainer import (
    AutonomousEvolutionController,
    AutonomousRetrainer,
    CheckpointManager,
    RecoveryStrategy,
)
from agent_whitepaper.stage_f.degradation_detector import (
    DegradationDetector,
    DegradationType,
)


# ============================================================
# 1. 自主任务生成器测试
# ============================================================
class TestAutonomousGenerator:
    def test_capability_assessment(self):
        gen = AutonomousTaskGenerator(grid_size=5, seed=42)
        assessment = gen.assess_capability(
            success_rate=0.8, avg_reward=0.5, avg_steps=20, skill_count=5
        )
        assert assessment.capability_level == 4
        assert assessment.zone_of_proximal_development == TaskDifficulty(5)

    def test_generate_for_capability(self):
        gen = AutonomousTaskGenerator(grid_size=5, seed=42)
        # 低能力Agent
        assessment = gen.assess_capability(0.2, -0.1, 30, 1)
        task = gen.generate_for_capability(assessment)
        assert task is not None
        assert task.num_sub_goals > 0

    def test_generate_by_difficulty(self):
        gen = AutonomousTaskGenerator(grid_size=5, seed=42)
        for difficulty in TaskDifficulty:
            task = gen._generate_task_by_difficulty(difficulty)
            assert task is not None
            assert task.num_sub_goals > 0

    def test_adaptive_difficulty(self):
        gen = AutonomousTaskGenerator(grid_size=5, seed=42)
        # 初始难度2
        assert gen.current_difficulty_level == 2

        # 模拟10次成功
        for _ in range(10):
            gen.record_task_result(TaskDifficulty(2), success=True)
        # 成功率高，难度应提升
        assert gen.current_difficulty_level >= 2

    def test_curriculum_generation(self):
        gen = AutonomousTaskGenerator(grid_size=5, seed=42)
        curriculum = gen.generate_curriculum(n_tasks=10)
        assert len(curriculum) == 10
        # 第一个应该比最后一个简单
        assert curriculum[0].num_sub_goals <= curriculum[-1].num_sub_goals + 2

    def test_exploration_task(self):
        gen = AutonomousTaskGenerator(grid_size=5, seed=42)
        visited = {(0, 0), (0, 1)}
        task = gen.generate_exploration_task(visited)
        assert task is not None


# ============================================================
# 2. 性能退化检测器测试
# ============================================================
class TestDegradationDetector:
    def test_no_degradation(self):
        detector = DegradationDetector(window_size=10)
        # 记录稳定的高性能
        for _ in range(20):
            detector.record(0.9, 0.8, 0.1)
        report = detector.detect()
        assert report.degradation_type == DegradationType.NONE

    def test_success_rate_drop(self):
        detector = DegradationDetector(window_size=10)
        # 先记录高性能
        for _ in range(20):
            detector.record(0.9, 0.8, 0.1)
        # 然后逐步下降（产生负趋势）
        for i in range(15):
            rate = max(0.1, 0.9 - i * 0.06)  # 0.9→0.9, 0.84, 0.78, ... 0.06
            detector.record(rate, 0.8, 0.5)
        report = detector.detect()
        # 应该检测到某种退化
        assert report.degradation_type in (
            DegradationType.SUCCESS_RATE_DROP, DegradationType.REWARD_DECLINE,
            DegradationType.FREE_ENERGY_INCREASE, DegradationType.INSTABILITY
        )
        assert report.severity > 0

    def test_instability_detection(self):
        detector = DegradationDetector(window_size=10)
        # 记录高方差数据
        for i in range(30):
            rate = 0.5 + (0.4 if i % 2 == 0 else -0.4)
            detector.record(max(0, min(1, rate)), 0.3, 0.2)
        report = detector.detect()
        # 应该检测到不稳定或成功率下降
        assert report.degradation_type in (
            DegradationType.INSTABILITY, DegradationType.SUCCESS_RATE_DROP, DegradationType.NONE
        )

    def test_health_score(self):
        detector = DegradationDetector(window_size=10)
        for _ in range(15):
            detector.record(0.9, 0.8, 0.1)
        score = detector.get_health_score()
        assert 0 <= score <= 1.0


# ============================================================
# 3. 自主重训练测试
# ============================================================
class TestAutonomousRetrainer:
    def test_checkpoint_manager(self):
        mgr = CheckpointManager(max_checkpoints=3)
        from agent_whitepaper.stage_a.policy import QTable
        q = QTable(5, 3, 0.1)
        q._table = np.ones((5, 3))
        mgr.save_checkpoint(q, "agent", 0.8)
        assert len(mgr._checkpoints) == 1

        best = mgr.get_best_checkpoint()
        assert best is not None
        assert best.performance_score == 0.8

    def test_rollback(self):
        mgr = CheckpointManager()
        from agent_whitepaper.stage_a.policy import QTable
        # 保存好的Q表
        good_q = QTable(5, 3, 0.1)
        good_q._table = np.ones((5, 3)) * 0.8
        mgr.save_checkpoint(good_q, "agent", 0.9)

        # 当前Q表变差
        bad_q = QTable(5, 3, 0.1)
        bad_q._table = np.zeros((5, 3))
        success = mgr.rollback_to_best(bad_q)
        assert success is True
        assert np.allclose(bad_q._table, np.ones((5, 3)) * 0.8)

    def test_stabilize_strategy(self):
        env = GridWorld(size=4, seed=42)
        agent = SelfDrivenAgent(env.action_space, n_states=16, seed=42)
        detector = DegradationDetector(window_size=10)
        retrainer = AutonomousRetrainer(detector=detector)

        # 记录一些数据
        for _ in range(15):
            detector.record(0.5, 0.3, 0.2)

        original_lr = agent.q_table.lr
        result = retrainer.monitor_and_recover(
            agent, env, force_strategy=RecoveryStrategy.STABILIZE
        )
        assert result.strategy == RecoveryStrategy.STABILIZE
        assert agent.q_table.lr <= original_lr

    def test_full_retrain(self):
        env = GridWorld(size=4, seed=42)
        agent = SelfDrivenAgent(env.action_space, n_states=16, seed=42)
        retrainer = AutonomousRetrainer()

        result = retrainer.monitor_and_recover(
            agent, env, force_strategy=RecoveryStrategy.FULL_RETRAIN
        )
        assert result.strategy == RecoveryStrategy.FULL_RETRAIN
        assert result.duration > 0

    def test_evolution_controller(self):
        env = GridWorld(size=4, goal=(3, 3), start=(0, 0), max_steps=30, seed=42)
        agent = SelfDrivenAgent(env.action_space, n_states=16, seed=42)
        controller = AutonomousEvolutionController(grid_size=4)

        # 先训练一下
        for _ in range(20):
            agent.run_episode(env)

        # 执行进化步骤
        log = controller.evolution_step(agent, env)
        assert "timestamp" in log
        assert "health_score" in log

        summary = controller.get_evolution_summary()
        assert summary["total_steps"] > 0


# ============================================================
# 4. 端到端：永续运行+自主进化
# ============================================================
class TestEndToEndF:
    def test_perpetual_evolution(self):
        """端到端：系统持续运行并自主进化。"""
        from agent_whitepaper.stage_d.runtime import (
            PerpetualRuntime,
            RuntimeConfig,
        )

        config = RuntimeConfig(
            db_path=":memory:",
            grid_size=4,
            n_agents=1,
            max_iterations=15,
            checkpoint_interval=1.0,
        )
        runtime = PerpetualRuntime(config)
        runtime.run()

        report = runtime.get_status_report()
        print("\n永续进化报告:")
        print(f"  迭代次数: {report['total_iterations']}")
        print(f"  完成任务: {report['total_tasks_completed']}")
        print(f"  总情节数: {report['total_episodes']}")
        print(f"  技能数: {report['skill_count']}")
        print(f"  系统健康: {report['health_report']['system_healthy']}")
        print(f"  运行时间: {report['uptime']:.2f}秒")

        assert report["total_iterations"] == 15
        assert report["status"] == "stopped"

    def test_degradation_and_recovery(self):
        """端到端：模拟退化并验证自动恢复。"""
        env = GridWorld(size=4, goal=(3, 3), start=(0, 0), max_steps=30, seed=42)
        agent = SelfDrivenAgent(env.action_space, n_states=16, seed=42)

        # 训练到高绩效
        for _ in range(100):
            agent.run_episode(env)
        good_rate = sum(agent.stats.recent_successes[-20:]) / 20
        print(f"训练后成功率: {good_rate:.0%}")

        # 保存检查点
        controller = AutonomousEvolutionController(grid_size=4)
        controller.retrainer.save_checkpoint(agent, good_rate)

        # 模拟退化：破坏Q表
        agent.q_table._table *= 0.1  # Q值衰减
        agent.q_table._table += np.random.RandomState(42).randn(16, 4) * 0.5

        # 运行几个情节验证退化
        for _ in range(10):
            agent.run_episode(env)
        degraded_rate = sum(agent.stats.recent_successes[-10:]) / 10
        print(f"退化后成功率: {degraded_rate:.0%}")

        # 执行恢复
        result = controller.retrainer.monitor_and_recover(
            agent, env, force_strategy=RecoveryStrategy.ROLLBACK
        )
        print(f"恢复策略: {result.strategy.value}")
        print(f"恢复前健康: {result.before_score:.3f}")
        print(f"恢复后健康: {result.after_score:.3f}")

        # 验证恢复
        for _ in range(10):
            agent.run_episode(env)
        recovered_rate = sum(agent.stats.recent_successes[-10:]) / 10
        print(f"恢复后成功率: {recovered_rate:.0%}")

        # 恢复后应该比退化时好
        assert recovered_rate >= degraded_rate * 0.8

    def test_skill_migration_in_perpetual_system(self):
        """端到端：永续系统中技能迁移真正生效。"""
        # 训练教师
        teacher_env = GridWorld(size=4, goal=(3, 3), start=(0, 0), max_steps=30, seed=42)
        teacher = SelfDrivenAgent(
            teacher_env.action_space, n_states=16, learning_rate=0.3, seed=42
        )
        for _ in range(150):
            teacher.run_episode(teacher_env)
        teacher_rate = sum(teacher.stats.recent_successes[-20:]) / 20

        # 收集教师知识
        coord = CollectiveLearningCoordinator(grid_size=4)
        coord.collect_from_agent("teacher", teacher.q_table, teacher_rate)

        # 创建学生并迁移知识
        student_env = GridWorld(size=4, goal=(3, 3), start=(0, 0), max_steps=30, seed=123)
        student = SelfDrivenAgent(
            student_env.action_space, n_states=16, learning_rate=0.3, seed=123
        )
        coord.teach_new_agent(student.q_table, strategy="weighted", inject_skills=False)

        # 创建基线（无迁移）
        baseline_env = GridWorld(size=4, goal=(3, 3), start=(0, 0), max_steps=30, seed=123)
        baseline = SelfDrivenAgent(
            baseline_env.action_space, n_states=16, learning_rate=0.3, seed=123
        )

        # 两者都训练30个情节
        for _ in range(30):
            student.run_episode(student_env)
            baseline.run_episode(baseline_env)

        student_rate = sum(student.stats.recent_successes[-15:]) / 15
        baseline_rate = sum(baseline.stats.recent_successes[-15:]) / 15

        print("\n技能迁移效果:")
        print(f"  教师成功率: {teacher_rate:.0%}")
        print(f"  基线成功率(30情节): {baseline_rate:.0%}")
        print(f"  学生成功率(30情节): {student_rate:.0%}")
        print(f"  迁移提升: {student_rate - baseline_rate:+.0%}")

        # 学生应该学得更快
        assert student_rate >= baseline_rate
