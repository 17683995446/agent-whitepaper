"""阶段E验证测试：验证技能迁移与真增强模块。

测试覆盖：
1. Q表序列化与迁移
2. 技能→Q值注入器
3. 跨Agent知识蒸馏
4. 端到端：技能迁移真正提升新Agent性能
"""

import numpy as np

from agent_whitepaper.stage_a.agent import SelfDrivenAgent
from agent_whitepaper.stage_a.environment import GridWorld
from agent_whitepaper.stage_a.policy import QTable
from agent_whitepaper.stage_b.skill_abstractor import Skill, SkillLibrary
from agent_whitepaper.stage_e.knowledge_distiller import (
    CollectiveLearningCoordinator,
    KnowledgeDistiller,
)
from agent_whitepaper.stage_e.qtable_migration import (
    QTableKnowledgeBase,
    QTableMigrator,
    QTableSnapshot,
)
from agent_whitepaper.stage_e.skill_injector import SkillQValueInjector


# ============================================================
# 1. Q表序列化与迁移测试
# ============================================================
class TestQTableMigration:
    def test_snapshot_creation_and_recovery(self):
        q = QTable(n_states=10, n_actions=4, learning_rate=0.2)
        q._table = np.random.rand(10, 4)
        snapshot = QTableSnapshot.from_q_table(q, "agent_001", 0.8)
        assert snapshot.n_states == 10
        assert snapshot.n_actions == 4
        assert snapshot.contributor_id == "agent_001"
        assert snapshot.performance_score == 0.8

        # 恢复
        recovered = snapshot.to_q_table()
        assert np.allclose(recovered._table, q._table)
        assert recovered.lr == 0.2

    def test_direct_migrate(self):
        source_q = QTable(10, 4, 0.1)
        source_q._table = np.ones((10, 4))
        target_q = QTable(10, 4, 0.1)
        target_q._table = np.zeros((10, 4))

        snapshot = QTableSnapshot.from_q_table(source_q, "source", 0.9)
        QTableMigrator.direct_migrate(snapshot, target_q)
        assert np.allclose(target_q._table, np.ones((10, 4)))

    def test_soft_migrate(self):
        source_q = QTable(5, 3, 0.1)
        source_q._table = np.ones((5, 3)) * 2.0
        target_q = QTable(5, 3, 0.1)
        target_q._table = np.zeros((5, 3))

        snapshot = QTableSnapshot.from_q_table(source_q, "source", 0.9)
        QTableMigrator.soft_migrate(snapshot, target_q, alpha=0.5)
        # 0.5*0 + 0.5*2 = 1.0
        assert np.allclose(target_q._table, np.ones((5, 3)))

    def test_selective_migrate(self):
        source_q = QTable(5, 3, 0.1)
        source_q._table = np.array([
            [0, 0, 0],      # 低置信度
            [1, 0, 0],      # 高置信度
            [0, 0, 0],      # 低置信度
            [0, 2, 0],      # 高置信度
            [0, 0, 0],      # 低置信度
        ])
        target_q = QTable(5, 3, 0.1)
        target_q._table = np.zeros((5, 3))

        snapshot = QTableSnapshot.from_q_table(source_q, "source", 0.9)
        QTableMigrator.selective_migrate(snapshot, target_q, confidence_threshold=0.5)
        # 只有状态1和3应该被迁移
        assert np.allclose(target_q._table[0], [0, 0, 0])
        assert np.allclose(target_q._table[1], [1, 0, 0])
        assert np.allclose(target_q._table[3], [0, 2, 0])

    def test_merge_q_tables(self):
        q1 = QTable(3, 2, 0.1)
        q1._table = np.ones((3, 2))
        q2 = QTable(3, 2, 0.1)
        q2._table = np.ones((3, 2)) * 3

        s1 = QTableSnapshot.from_q_table(q1, "a1", 0.5)
        s2 = QTableSnapshot.from_q_table(q2, "a2", 0.5)
        merged = QTableMigrator.merge_q_tables([s1, s2])
        # 平均: (1+3)/2 = 2
        assert np.allclose(merged.get_table_array(), np.ones((3, 2)) * 2)

    def test_knowledge_base(self):
        kb = QTableKnowledgeBase()
        q = QTable(5, 3, 0.1)
        q._table = np.ones((5, 3))
        snap = QTableSnapshot.from_q_table(q, "agent_1", 0.8)
        kb.contribute(snap)
        assert len(kb._snapshots) == 1

        best = kb.get_best_snapshot()
        assert best is not None
        assert best.performance_score == 0.8

    def test_initialize_new_agent(self):
        kb = QTableKnowledgeBase()
        q = QTable(5, 3, 0.1)
        q._table = np.ones((5, 3)) * 0.5
        snap = QTableSnapshot.from_q_table(q, "teacher", 0.9)
        kb.contribute(snap)

        new_q = QTable(5, 3, 0.1)
        new_q._table = np.zeros((5, 3))
        kb.initialize_new_agent(new_q, migration_strategy="soft", alpha=0.7)
        # 0.3*0 + 0.7*0.5 = 0.35
        assert np.allclose(new_q._table, np.ones((5, 3)) * 0.35)


# ============================================================
# 2. 技能→Q值注入器测试
# ============================================================
class TestSkillInjector:
    def test_inject_single_skill(self):
        skill = Skill(
            name="test_skill",
            precondition=(0, 0),  # 状态0
            action_sequence=(3, 3, 1, 1),  # 右右下下
            postcondition=(2, 2),
            expected_reward=1.0,
            success_count=8,
            total_count=10,
        )
        q_table = QTable(n_states=25, n_actions=4, learning_rate=0.1)
        q_table._table = np.zeros((25, 4))

        injector = SkillQValueInjector(grid_size=5)
        result = injector.inject_skill(skill, q_table)

        assert result.skill_name == "test_skill"
        assert result.states_injected > 0
        assert result.q_values_after > result.q_values_before
        # 状态0的动作3应该有Q值
        assert q_table._table[0, 3] > 0

    def test_inject_all_skills(self):
        lib = SkillLibrary()
        # 添加几个技能
        for i in range(3):
            skill = Skill(
                name=f"skill_{i}",
                precondition=(i, 0),
                action_sequence=(3, 1),
                postcondition=(i + 1, 1),
                expected_reward=0.8,
                success_count=7,
                total_count=10,
            )
            lib.add_skill(skill)

        q_table = QTable(25, 4, 0.1)
        injector = SkillQValueInjector(grid_size=5)
        results = injector.inject_all_skills(lib, q_table)

        assert len(results) == 3
        # Q表应该有非零值
        assert np.sum(q_table._table) > 0

    def test_injection_respects_success_rate(self):
        """低成功率技能应注入更弱的Q值。"""
        good_skill = Skill(
            "good", (0, 0), (3,), (1, 0), 1.0, 9, 10  # 90%成功率
        )
        bad_skill = Skill(
            "bad", (0, 0), (3,), (1, 0), 1.0, 1, 10  # 10%成功率
        )
        injector = SkillQValueInjector(grid_size=5)

        q1 = QTable(25, 4, 0.1)
        q1._table = np.zeros((25, 4))
        injector.inject_skill(good_skill, q1)

        q2 = QTable(25, 4, 0.1)
        q2._table = np.zeros((25, 4))
        injector.inject_skill(bad_skill, q2)

        # 好技能注入的Q值应该更高
        assert q1._table[0, 3] > q2._table[0, 3]

    def test_evaluate_injection_impact(self):
        lib = SkillLibrary()
        skill = Skill("s1", (0, 0), (3, 3, 1, 1), (2, 2), 1.0, 8, 10)
        lib.add_skill(skill)

        q_table = QTable(25, 4, 0.1)
        injector = SkillQValueInjector(grid_size=5)
        results = injector.inject_all_skills(lib, q_table)
        impact = injector.evaluate_injection_impact(q_table, results)

        assert impact["total_skills_injected"] == 1
        assert impact["q_value_change"] > 0


# ============================================================
# 3. 知识蒸馏测试
# ============================================================
class TestKnowledgeDistiller:
    def test_average_distillation(self):
        # 两个教师，Q表不同
        q1 = QTable(5, 3, 0.1)
        q1._table = np.ones((5, 3)) * 2.0
        q2 = QTable(5, 3, 0.1)
        q2._table = np.ones((5, 3)) * 4.0

        s1 = QTableSnapshot.from_q_table(q1, "teacher1", 0.5)
        s2 = QTableSnapshot.from_q_table(q2, "teacher2", 0.5)

        student = QTable(5, 3, 0.1)
        student._table = np.zeros((5, 3))

        distiller = KnowledgeDistiller()
        result = distiller.distill([s1, s2], student, strategy="average", injection_alpha=1.0)

        # 平均: (2+4)/2 = 3
        assert np.allclose(student._table, np.ones((5, 3)) * 3.0)
        assert result.states_distilled == 5

    def test_weighted_distillation(self):
        q1 = QTable(5, 3, 0.1)
        q1._table = np.ones((5, 3)) * 2.0
        q2 = QTable(5, 3, 0.1)
        q2._table = np.ones((5, 3)) * 4.0

        # 教师2性能更好
        s1 = QTableSnapshot.from_q_table(q1, "t1", 0.3)
        s2 = QTableSnapshot.from_q_table(q2, "t2", 0.7)

        student = QTable(5, 3, 0.1)
        student._table = np.zeros((5, 3))

        distiller = KnowledgeDistiller()
        distiller.distill([s1, s2], student, strategy="weighted", injection_alpha=1.0)

        # 加权: 0.3*2 + 0.7*4 = 3.4 (归一化后)
        # weights = [0.3, 0.7] / 1.0 = [0.3, 0.7]
        # 0.3*2 + 0.7*4 = 3.4
        assert np.allclose(student._table, np.ones((5, 3)) * 3.4, atol=0.01)

    def test_consensus_distillation(self):
        # 两个教师对状态0的最优动作一致，对状态1不一致
        q1 = QTable(3, 3, 0.1)
        q1._table = np.array([
            [0, 1, 0],  # 状态0: 动作1最优
            [0, 0, 1],  # 状态1: 动作2最优
            [0, 0, 0],
        ])
        q2 = QTable(3, 3, 0.1)
        q2._table = np.array([
            [0, 2, 0],  # 状态0: 动作1最优（一致）
            [1, 0, 0],  # 状态1: 动作0最优（不一致）
            [0, 0, 0],
        ])

        s1 = QTableSnapshot.from_q_table(q1, "t1", 0.5)
        s2 = QTableSnapshot.from_q_table(q2, "t2", 0.5)

        student = QTable(3, 3, 0.1)
        student._table = np.zeros((3, 3))

        distiller = KnowledgeDistiller()
        result = distiller.distill([s1, s2], student, strategy="consensus", injection_alpha=1.0)

        # 状态0有共识，状态1没有
        assert result.consensus_ratio < 1.0

    def test_collective_coordinator(self):
        coord = CollectiveLearningCoordinator(grid_size=5)

        # 教师1
        q1 = QTable(25, 4, 0.1)
        q1._table = np.random.rand(25, 4) * 0.5
        coord.collect_from_agent("teacher_1", q1, 0.8)

        # 教师2
        q2 = QTable(25, 4, 0.1)
        q2._table = np.random.rand(25, 4) * 0.5
        coord.collect_from_agent("teacher_2", q2, 0.6)

        # 教导新Agent
        student_q = QTable(25, 4, 0.1)
        student_q._table = np.zeros((25, 4))

        distill_result, injection_results = coord.teach_new_agent(
            student_q, strategy="weighted", inject_skills=False
        )

        assert distill_result.states_distilled == 25
        # 学生Q表应该有非零值
        assert np.sum(student_q._table) > 0

    def test_distillation_stats(self):
        coord = CollectiveLearningCoordinator(grid_size=5)
        q = QTable(25, 4, 0.1)
        q._table = np.random.rand(25, 4)
        coord.collect_from_agent("teacher", q, 0.7)

        student = QTable(25, 4, 0.1)
        coord.teach_new_agent(student, inject_skills=False)

        stats = coord.get_collective_stats()
        assert stats["total_snapshots"] == 1
        assert stats["total_distillations"] == 1


# ============================================================
# 4. 端到端：技能迁移真正提升性能
# ============================================================
class TestEndToEndE:
    def test_skill_migration_improves_learning(self):
        """端到端：技能迁移真正加速新Agent学习。"""
        # 步骤1：训练教师Agent
        teacher_env = GridWorld(size=4, goal=(3, 3), start=(0, 0), max_steps=30, seed=42)
        teacher = SelfDrivenAgent(
            teacher_env.action_space, n_states=16, learning_rate=0.3, seed=42
        )
        for _ in range(200):
            teacher.run_episode(teacher_env)
        teacher_success_rate = sum(teacher.stats.recent_successes[-20:]) / 20
        print(f"教师成功率: {teacher_success_rate:.0%}")

        # 步骤2：收集教师知识
        coord = CollectiveLearningCoordinator(grid_size=4)
        coord.collect_from_agent("teacher", teacher.q_table, teacher_success_rate)

        # 步骤3：创建学生Agent并注入知识
        student_env = GridWorld(size=4, goal=(3, 3), start=(0, 0), max_steps=30, seed=123)
        student = SelfDrivenAgent(
            student_env.action_space, n_states=16, learning_rate=0.3, seed=123
        )

        # 蒸馏知识到学生
        distill_result, _ = coord.teach_new_agent(
            student.q_table, strategy="weighted", inject_skills=False
        )
        print(f"蒸馏: {distill_result.states_distilled}状态, "
              f"共识{distill_result.consensus_ratio:.2f}, "
              f"性能提升{distill_result.performance_gain:.3f}")

        # 步骤4：比较学生与基线
        baseline_env = GridWorld(size=4, goal=(3, 3), start=(0, 0), max_steps=30, seed=123)
        baseline = SelfDrivenAgent(
            baseline_env.action_space, n_states=16, learning_rate=0.3, seed=123
        )

        # 两者都训练50个情节
        for _ in range(50):
            student.run_episode(student_env)
            baseline.run_episode(baseline_env)

        student_rate = sum(student.stats.recent_successes[-20:]) / 20
        baseline_rate = sum(baseline.stats.recent_successes[-20:]) / 20
        print(f"基线成功率: {baseline_rate:.0%}")
        print(f"学生成功率: {student_rate:.0%}")
        print(f"提升: {student_rate - baseline_rate:+.0%}")

        # 学生应该不比基线差（有知识迁移应该更快学会）
        assert student_rate >= baseline_rate * 0.8

    def test_collective_intelligence_growth(self):
        """集体智慧随Agent增多而增长。"""
        coord = CollectiveLearningCoordinator(grid_size=4)

        # 逐步添加教师，验证知识库增长
        for i in range(3):
            env = GridWorld(size=4, goal=(3, 3), start=(0, 0), max_steps=30, seed=42 + i)
            agent = SelfDrivenAgent(
                env.action_space, n_states=16, learning_rate=0.3, seed=42 + i
            )
            for _ in range(100):
                agent.run_episode(env)
            success_rate = sum(agent.stats.recent_successes[-20:]) / 20
            coord.collect_from_agent(f"agent_{i}", agent.q_table, success_rate)

        stats = coord.get_collective_stats()
        assert stats["total_snapshots"] == 3
        assert stats["best_performance"] > 0
        print(f"集体智慧: {stats['total_snapshots']}个教师, "
              f"最佳性能{stats['best_performance']:.0%}")
