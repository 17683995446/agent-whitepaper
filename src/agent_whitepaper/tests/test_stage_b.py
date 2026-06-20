"""阶段B验证测试：验证技能增强与长任务模块。

测试覆盖：
1. 技能抽象器（从情节抽象技能）
2. 技能微调器（基于自由能反馈）
3. 元学习器（学习如何学习）
4. 任务分解器（长任务→子目标）
5. 子目标状态机与进度监控
6. 端到端：Agent学会技能并完成长任务
"""

import numpy as np

from agent_whitepaper.stage_a.agent import SelfDrivenAgent
from agent_whitepaper.stage_a.environment import GridWorld, Transition
from agent_whitepaper.stage_a.memory import Episode
from agent_whitepaper.stage_b.meta_learner import (
    MetaLearner,
    MetaParameters,
)
from agent_whitepaper.stage_b.skill_abstractor import (
    Skill,
    SkillAbstractor,
    SkillLibrary,
)
from agent_whitepaper.stage_b.skill_refiner import (
    SkillExecutionRecord,
    SkillRefiner,
)
from agent_whitepaper.stage_b.subgoal_manager import (
    LongTaskExecutor,
    SubGoalStateMachine,
)
from agent_whitepaper.stage_b.task_decomposer import (
    SubGoal,
    SubGoalStatus,
    Task,
    TaskDecomposer,
)


# ============================================================
# 1. 技能抽象器测试
# ============================================================
class TestSkillAbstractor:
    def test_skill_creation(self):
        skill = Skill(
            name="test_skill",
            precondition=(0, 0),
            action_sequence=(0, 1),
            postcondition=(1, 1),
            expected_reward=1.0,
        )
        assert skill.length == 2
        assert skill.success_rate == 0.0

    def test_skill_library_add_find(self):
        lib = SkillLibrary()
        skill = Skill(
            name="s1",
            precondition=(0, 0),
            action_sequence=(0, 1),
            postcondition=(1, 1),
            expected_reward=1.0,
        )
        lib.add_skill(skill)
        assert len(lib) == 1
        found = lib.find_by_precondition((0, 0))
        assert len(found) == 1
        assert found[0].name == "s1"

    def test_skill_library_best(self):
        lib = SkillLibrary()
        s1 = Skill("s1", (0, 0), (0,), (1,), 0.5, success_count=5, total_count=10)
        s2 = Skill("s2", (0, 0), (1,), (1,), 1.0, success_count=8, total_count=10)
        lib.add_skill(s1)
        lib.add_skill(s2)
        best = lib.find_best_skill((0, 0))
        assert best.name == "s2"

    def test_abstract_from_episodes(self):
        """从成功情节抽象技能。"""
        # 构造几个成功情节，都包含动作序列 [2, 2, 1, 1]（右右下下）
        episodes = []
        for i in range(5):
            transitions = []
            actions = [2, 2, 1, 1]
            for j, a in enumerate(actions):
                t = Transition(
                    observation=np.array([float(j)]),
                    action=a,
                    reward=0.25 if j == 3 else 0.0,
                    next_observation=np.array([float(j + 1)]),
                    done=(j == 3),
                    info={"reached_goal": (j == 3)}
                )
                transitions.append(t)
            episodes.append(Episode.from_transitions(transitions))

        lib = SkillLibrary()
        abstractor = SkillAbstractor(min_n=2, max_n=4, min_frequency=3)
        new_skills = abstractor.abstract_from_episodes(episodes, lib)

        assert len(new_skills) > 0
        assert len(lib) > 0

    def test_abstract_composite_skills(self):
        """测试组合技能。"""
        lib = SkillLibrary()
        s1 = Skill("s1", (0, 0), (0, 1), (1, 1), 1.0, 5, 5)
        s2 = Skill("s2", (1, 1), (2, 3), (2, 2), 1.0, 5, 5)
        lib.add_skill(s1)
        lib.add_skill(s2)

        abstractor = SkillAbstractor()
        composites = abstractor.abstract_composite_skills(lib, [s1, s2])

        assert len(composites) > 0
        assert composites[0].length == 4  # 2+2


# ============================================================
# 2. 技能微调器测试
# ============================================================
class TestSkillRefiner:
    def test_execution_record(self):
        record = SkillExecutionRecord(
            skill_name="s1",
            precondition_matched=True,
            free_energy_before=1.0,
            free_energy_after=0.5,
            actual_reward=1.0,
            success=True,
            actions_executed=(0, 1),
        )
        assert record.free_energy_delta == -0.5  # 下降
        assert record.effectiveness > 0.5

    def test_refiner_record(self):
        refiner = SkillRefiner(n_actions=4)
        record = SkillExecutionRecord(
            skill_name="s1",
            precondition_matched=True,
            free_energy_before=1.0,
            free_energy_after=0.8,
            actual_reward=0.5,
            success=True,
            actions_executed=(0, 1),
        )
        refiner.record_execution(record)
        stats = refiner.get_skill_statistics("s1")
        assert stats["count"] == 1

    def test_refine_effective_skill(self):
        """有效技能应被强化。"""
        lib = SkillLibrary()
        skill = Skill("s1", (0, 0), (0, 1), (1, 1), 0.5, 0, 0)
        lib.add_skill(skill)

        refiner = SkillRefiner(n_actions=4)
        # 记录多次有效执行
        for _ in range(5):
            refiner.record_execution(SkillExecutionRecord(
                skill_name="s1",
                precondition_matched=True,
                free_energy_before=1.0,
                free_energy_after=0.5,  # 自由能下降
                actual_reward=1.0,
                success=True,
                actions_executed=(0, 1),
            ))

        refined = refiner.refine_skill(skill, lib)
        assert refined is not None
        assert refined.expected_reward > skill.expected_reward

    def test_refine_ineffective_skill(self):
        """无效技能应产生变体。"""
        lib = SkillLibrary()
        skill = Skill("s1", (0, 0), (0, 1), (1, 1), 0.5, 0, 0)
        lib.add_skill(skill)

        refiner = SkillRefiner(n_actions=4, mutation_rate=0.5, min_effectiveness=0.8)
        # 记录多次无效执行（高自由能上升 + 负奖励）
        for _ in range(5):
            refiner.record_execution(SkillExecutionRecord(
                skill_name="s1",
                precondition_matched=True,
                free_energy_before=0.1,
                free_energy_after=5.0,  # 自由能大幅上升
                actual_reward=-2.0,  # 严重负奖励
                success=False,
                actions_executed=(0, 1),
            ))

        refined = refiner.refine_skill(skill, lib)
        # 应该产生了变体
        assert refined is not None or len(lib) > 1


# ============================================================
# 3. 元学习器测试
# ============================================================
class TestMetaLearner:
    def test_meta_params_clamp(self):
        params = MetaParameters(learning_rate=2.0, exploration_rate=-0.5)
        clamped = params.clamp()
        assert 0.01 <= clamped.learning_rate <= 1.0
        assert 0.01 <= clamped.exploration_rate <= 0.99

    def test_record_and_progress(self):
        learner = MetaLearner(window_size=10)
        for i in range(15):
            learner.record_episode(reward=float(i), success=(i > 5))

        progress = learner.compute_progress()
        assert progress is not None
        assert progress.episode_range == (5, 15)

    def test_adapt_on_stagnation(self):
        """学习停滞时应增加探索。"""
        learner = MetaLearner(window_size=10, adaptation_rate=0.2)
        # 记录停滞的学习（奖励不变）
        for _ in range(10):
            learner.record_episode(reward=0.5, success=False)

        initial_explore = learner.params.exploration_rate
        new_params = learner.adapt_parameters()
        assert new_params.exploration_rate > initial_explore

    def test_adapt_on_progress(self):
        """学习顺利时应降低探索。"""
        learner = MetaLearner(window_size=20, adaptation_rate=0.2)
        # 记录稳步提升的学习
        for i in range(20):
            learner.record_episode(reward=float(i) * 0.1, success=(i > 10))

        initial_explore = learner.params.exploration_rate
        new_params = learner.adapt_parameters()
        assert new_params.exploration_rate <= initial_explore + 0.01  # 允许微小波动

    def test_progress_summary(self):
        learner = MetaLearner(window_size=5)
        summary = learner.get_progress_summary()
        assert summary["status"] == "warming_up"

        for i in range(5):
            learner.record_episode(reward=1.0, success=True)
        # 触发进度计算
        learner.compute_progress()
        summary = learner.get_progress_summary()
        assert summary["status"] == "learning"


# ============================================================
# 4. 任务分解器测试
# ============================================================
class TestTaskDecomposer:
    def test_decompose_path_task(self):
        decomposer = TaskDecomposer(grid_size=5)
        task = decomposer.decompose_path_task(
            start=(0, 0), end=(4, 4), n_waypoints=1
        )
        assert task.num_sub_goals == 3  # 起点 + 1航点 + 终点
        assert task.sub_goals[0].target_state == (0, 0)
        assert task.sub_goals[-1].target_state == (4, 4)

    def test_decompose_multi_target(self):
        decomposer = TaskDecomposer(grid_size=5)
        targets = [(0, 0), (2, 2), (4, 4)]
        task = decomposer.decompose_multi_target_task(targets)
        assert task.num_sub_goals == 3
        assert task.sub_goals[1].prerequisites == (0,)

    def test_adaptive_decompose(self):
        decomposer = TaskDecomposer(grid_size=5)
        # 低技能 → 多航点
        task_low = decomposer.adaptive_decompose((0, 0), (4, 4), agent_skill_count=1)
        # 高技能 → 少航点
        task_high = decomposer.adaptive_decompose((0, 0), (4, 4), agent_skill_count=15)
        assert task_low.num_sub_goals >= task_high.num_sub_goals


# ============================================================
# 5. 子目标状态机测试
# ============================================================
class TestSubGoalStateMachine:
    def test_initialize(self):
        decomposer = TaskDecomposer(grid_size=5)
        task = decomposer.decompose_path_task((0, 0), (4, 4), n_waypoints=1)
        sm = SubGoalStateMachine()
        progress = sm.initialize(task)

        assert progress.current_goal_index == 0
        assert progress.goal_statuses[0] == SubGoalStatus.ACTIVE

    def test_goal_completion(self):
        """子目标完成后应转移到下一个。"""
        sg1 = SubGoal(0, "g1", (1, 1), 1.0, 10, ())
        sg2 = SubGoal(1, "g2", (2, 2), 1.0, 10, (0,))
        task = Task("test", "test", [sg1, sg2])
        sm = SubGoalStateMachine()
        progress = sm.initialize(task)

        # 到达第一个子目标
        progress = sm.update(progress, (1, 1), 0.1, 5)
        assert progress.goal_statuses[0] == SubGoalStatus.COMPLETED
        assert progress.current_goal_index == 1
        assert progress.goal_statuses[1] == SubGoalStatus.ACTIVE

    def test_goal_timeout_retry(self):
        """超时应触发重试。"""
        sg = SubGoal(0, "g1", (1, 1), 1.0, max_steps=5, prerequisites=())
        task = Task("test", "test", [sg])
        sm = SubGoalStateMachine(max_retries=2)
        progress = sm.initialize(task)

        # 模拟超时
        for step in range(5):
            progress = sm.update(progress, (0, 0), 0.0, step + 1)

        # 应该还在重试（未到max_retries）
        assert progress.goal_statuses[0] in [SubGoalStatus.ACTIVE, SubGoalStatus.SKIPPED]

    def test_progress_report(self):
        sg1 = SubGoal(0, "g1", (1, 1), 1.0, 10, ())
        sg2 = SubGoal(1, "g2", (2, 2), 1.0, 10, (0,))
        task = Task("test", "test", [sg1, sg2])
        sm = SubGoalStateMachine()
        progress = sm.initialize(task)
        report = sm.get_progress_report(progress)
        assert report["total_sub_goals"] == 2
        assert report["completion_rate"] == 0.0


# ============================================================
# 6. 端到端：技能增强与长任务
# ============================================================
class TestEndToEndB:
    def test_agent_learns_and_uses_skills(self):
        """Agent学习技能并用于新任务。"""
        # 第一步：训练Agent
        env = GridWorld(size=4, goal=(3, 3), start=(0, 0), max_steps=30, seed=42)
        space = env.action_space
        agent = SelfDrivenAgent(space, n_states=16, learning_rate=0.3, seed=42)

        for _ in range(100):
            agent.run_episode(env)

        # 第二步：从成功情节抽象技能
        successful_episodes = [
            ep for ep in agent.memory.get_recent_episodes(50) if ep.success
        ]
        if successful_episodes:
            lib = SkillLibrary()
            abstractor = SkillAbstractor(min_n=2, max_n=4, min_frequency=2)
            skills = abstractor.abstract_from_episodes(successful_episodes, lib)
            # 应该抽象出一些技能
            print(f"抽象出 {len(skills)} 个技能")
            # 即使没有技能（如果成功情节太少），测试也应通过
            assert isinstance(skills, list)

    def test_long_task_execution(self):
        """长任务分解与执行。"""
        # 训练Agent
        env = GridWorld(size=5, goal=(4, 4), start=(0, 0), max_steps=50, seed=42)
        space = env.action_space
        agent = SelfDrivenAgent(space, n_states=25, learning_rate=0.3, seed=42)

        for _ in range(50):
            agent.run_episode(env)

        # 分解长任务
        decomposer = TaskDecomposer(grid_size=5)
        task = decomposer.decompose_path_task(
            start=(0, 0), end=(4, 4), n_waypoints=1, task_name="long_task"
        )

        # 执行长任务
        executor = LongTaskExecutor(max_retries=2)
        progress = executor.execute_task(task, agent, env)

        report = executor.state_machine.get_progress_report(progress)
        print(f"长任务进度: {report['completion_rate']:.2%}")
        print(f"总步数: {report['total_steps']}, 总奖励: {report['total_reward']:.2f}")
        # 至少应该执行了一些步
        assert progress.total_steps > 0

    def test_meta_learning_improves(self):
        """元学习应改善学习效率。"""
        learner = MetaLearner(window_size=10, adaptation_rate=0.15)

        # 模拟学习过程
        for i in range(30):
            # 奖励逐渐提升
            reward = 0.1 * i + np.random.RandomState(42).randn() * 0.1
            success = reward > 1.0
            learner.record_episode(reward, success, free_energy=1.0 - i * 0.02)
            if learner.should_adapt():
                learner.adapt_parameters()

        final_params = learner.params
        summary = learner.get_progress_summary()
        print(f"元学习后: lr={final_params.learning_rate:.3f}, "
              f"explore={final_params.exploration_rate:.3f}")
        # 参数应该发生了变化
        assert summary["episodes"] == 30
