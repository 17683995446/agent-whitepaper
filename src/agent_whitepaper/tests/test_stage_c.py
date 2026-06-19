"""阶段C验证测试：验证多Agent协作模块。

测试覆盖：
1. Agent注册中心
2. 异步消息总线
3. 角色分配（基于驱力互补）
4. 联盟形成与任务分配
5. 共享技能库
6. 端到端：多Agent协作完成任务
"""

import numpy as np
import pytest
import time

from agent_whitepaper.stage_c.registry import (
    AgentRegistry, AgentRegistration, AgentCapability, AgentStatus,
)
from agent_whitepaper.stage_c.message_bus import (
    MessageBus, Message, MessageType, MessageQueue,
)
from agent_whitepaper.stage_c.role_assigner import (
    RoleAssigner, RoleAssignment, Role, TeamConfiguration, ROLE_DRIVE_PROFILES,
)
from agent_whitepaper.stage_c.coalition import (
    Coalition, CoalitionStatus, CoalitionFormer,
)
from agent_whitepaper.stage_c.shared_skills import (
    SharedSkill, SharedSkillLibrary,
)
from agent_whitepaper.stage_b.skill_abstractor import Skill, SkillLibrary
from agent_whitepaper.stage_b.task_decomposer import TaskDecomposer, Task


# ============================================================
# 1. Agent注册中心测试
# ============================================================
class TestAgentRegistry:
    def test_register_agent(self):
        registry = AgentRegistry()
        agent_id = registry.register("Agent1", AgentCapability(specialization="navigation"))
        assert agent_id.startswith("agent_")
        assert len(registry.get_all_agents()) == 1

    def test_unregister_agent(self):
        registry = AgentRegistry()
        agent_id = registry.register("Agent1")
        assert registry.unregister(agent_id) is True
        assert len(registry.get_all_agents()) == 0

    def test_find_by_specialization(self):
        registry = AgentRegistry()
        registry.register("Nav1", AgentCapability(specialization="navigation"))
        registry.register("Nav2", AgentCapability(specialization="navigation"))
        registry.register("Manip1", AgentCapability(specialization="manipulation"))
        nav_agents = registry.find_by_specialization("navigation")
        assert len(nav_agents) == 2

    def test_find_best_agent(self):
        registry = AgentRegistry()
        registry.register("A1", AgentCapability(performance_score=0.5))
        registry.register("A2", AgentCapability(performance_score=0.9))
        best = registry.find_best_agent()
        assert best is not None
        assert best.capability.performance_score == 0.9

    def test_heartbeat_and_timeout(self):
        registry = AgentRegistry(heartbeat_timeout=0.1)
        agent_id = registry.register("A1")
        # 模拟超时
        registry._agents[agent_id].last_heartbeat = time.time() - 1.0
        timed_out = registry.check_timeouts()
        assert agent_id in timed_out
        assert registry.get_agent(agent_id).status == AgentStatus.OFFLINE

    def test_update_status(self):
        registry = AgentRegistry()
        agent_id = registry.register("A1")
        assert registry.update_status(agent_id, AgentStatus.BUSY) is True
        assert registry.get_agent(agent_id).status == AgentStatus.BUSY


# ============================================================
# 2. 消息总线测试
# ============================================================
class TestMessageBus:
    def test_register_and_send(self):
        bus = MessageBus()
        bus.register_agent("agent_001")
        bus.register_agent("agent_002")
        msg = bus.create_message("agent_001", "agent_002", MessageType.REQUEST, "hello")
        assert bus.send(msg) is True
        received = bus.receive("agent_002")
        assert received is not None
        assert received.content == "hello"

    def test_broadcast(self):
        bus = MessageBus()
        bus.register_agent("a1")
        bus.register_agent("a2")
        bus.register_agent("a3")
        msg = bus.create_message("a1", None, MessageType.BROADCAST, "announcement")
        assert bus.broadcast(msg) is True
        assert bus.has_messages("a2")
        assert bus.has_messages("a3")
        assert not bus.has_messages("a1")  # 不发给自己

    def test_subscribe_publish(self):
        bus = MessageBus()
        bus.register_agent("a1")
        bus.register_agent("a2")
        bus.subscribe("a2", "alerts")
        msg = bus.create_message("a1", None, MessageType.NOTIFICATION, "alert!")
        bus.publish("alerts", msg)
        assert bus.has_messages("a2")
        assert not bus.has_messages("a1")  # a1没订阅

    def test_message_queue_priority(self):
        queue = MessageQueue()
        m1 = Message("1", "a", "b", MessageType.REQUEST, "low", priority=5)
        m2 = Message("2", "a", "b", MessageType.REQUEST, "high", priority=1)
        queue.push(m1)
        queue.push(m2)
        first = queue.pop()
        assert first.priority == 1  # 高优先级先出

    def test_receive_all(self):
        bus = MessageBus()
        bus.register_agent("a1")
        bus.register_agent("a2")
        for i in range(3):
            msg = bus.create_message("a1", "a2", MessageType.REQUEST, f"msg{i}")
            bus.send(msg)
        messages = bus.receive_all("a2")
        assert len(messages) == 3

    def test_stats(self):
        bus = MessageBus()
        bus.register_agent("a1")
        bus.register_agent("a2")
        msg = bus.create_message("a1", "a2", MessageType.REQUEST, "test")
        bus.send(msg)
        stats = bus.get_stats()
        assert stats["total_sent"] == 1
        assert stats["total_delivered"] == 1


# ============================================================
# 3. 角色分配测试
# ============================================================
class TestRoleAssigner:
    def test_match_score(self):
        registry = AgentRegistry()
        assigner = RoleAssigner(registry)
        # 完美匹配
        score = assigner.compute_match_score((0.2, 0.4, 0.9), Role.EXPLORER)
        assert score > 0.95
        # 完全不匹配
        score = assigner.compute_match_score((0.9, 0.5, 0.2), Role.EXPLORER)
        assert score < 0.5

    def test_assign_roles(self):
        registry = AgentRegistry()
        # 注册不同驱力特征的Agent
        registry.register("Explorer1", AgentCapability(
            drive_profile=(0.2, 0.4, 0.9)  # 探索者
        ))
        registry.register("Guardian1", AgentCapability(
            drive_profile=(0.9, 0.5, 0.2)  # 守护者
        ))
        registry.register("Coordinator1", AgentCapability(
            drive_profile=(0.5, 0.9, 0.4)  # 协调者
        ))

        assigner = RoleAssigner(registry)
        config = TeamConfiguration(required_roles={
            Role.EXPLORER: 1,
            Role.GUARDIAN: 1,
            Role.COORDINATOR: 1,
        })
        assignments = assigner.assign_roles(config)
        assert len(assignments) == 3

        # 验证角色分配正确
        roles_assigned = {a.role for a in assignments}
        assert Role.EXPLORER in roles_assigned
        assert Role.GUARDIAN in roles_assigned
        assert Role.COORDINATOR in roles_assigned

    def test_suggest_role(self):
        registry = AgentRegistry()
        agent_id = registry.register("A1", AgentCapability(
            drive_profile=(0.9, 0.5, 0.2)  # 守护者特征
        ))
        assigner = RoleAssigner(registry)
        suggested = assigner.suggest_role_for_agent(agent_id)
        assert suggested == Role.GUARDIAN

    def test_team_balance(self):
        registry = AgentRegistry()
        assigner = RoleAssigner(registry)
        # 均衡分配
        assignments = [
            RoleAssignment("a1", Role.EXPLORER, 0.9, ""),
            RoleAssignment("a2", Role.GUARDIAN, 0.9, ""),
            RoleAssignment("a3", Role.COORDINATOR, 0.9, ""),
        ]
        balance = assigner.evaluate_team_balance(assignments)
        assert balance > 0.8  # 均衡


# ============================================================
# 4. 联盟形成测试
# ============================================================
class TestCoalition:
    def test_form_coalition(self):
        registry = AgentRegistry()
        for i in range(5):
            registry.register(f"Agent{i}", AgentCapability(
                drive_profile=tuple(np.random.RandomState(i).rand(3))
            ))

        bus = MessageBus()
        for reg in registry.get_all_agents():
            bus.register_agent(reg.agent_id)

        former = CoalitionFormer(registry, bus)
        decomposer = TaskDecomposer(grid_size=5)
        task = decomposer.decompose_path_task((0, 0), (4, 4), n_waypoints=2)

        coalition = former.form_coalition(task)
        assert coalition is not None
        assert coalition.size > 0
        assert coalition.status == CoalitionStatus.ACTIVE

    def test_subgoal_assignment(self):
        registry = AgentRegistry()
        for i in range(3):
            registry.register(f"Agent{i}")

        bus = MessageBus()
        for reg in registry.get_all_agents():
            bus.register_agent(reg.agent_id)

        former = CoalitionFormer(registry, bus)
        decomposer = TaskDecomposer(grid_size=5)
        task = decomposer.decompose_path_task((0, 0), (4, 4), n_waypoints=1)

        coalition = former.form_coalition(task)
        assert coalition is not None
        # 每个子目标都应该分配了Agent
        assert len(coalition.subgoal_assignments) == task.num_sub_goals

    def test_update_progress(self):
        registry = AgentRegistry()
        registry.register("A1")
        bus = MessageBus()
        bus.register_agent("A1")

        former = CoalitionFormer(registry, bus)
        decomposer = TaskDecomposer(grid_size=5)
        task = decomposer.decompose_path_task((0, 0), (4, 4), n_waypoints=0)

        coalition = former.form_coalition(task)
        former.update_progress(coalition.coalition_id, 1, 1)
        updated = former.get_coalition(coalition.coalition_id)
        assert updated.progress == 1.0
        assert updated.status == CoalitionStatus.COMPLETED

    def test_disband(self):
        registry = AgentRegistry()
        registry.register("A1")
        bus = MessageBus()
        bus.register_agent("A1")

        former = CoalitionFormer(registry, bus)
        decomposer = TaskDecomposer(grid_size=5)
        task = decomposer.decompose_path_task((0, 0), (4, 4), n_waypoints=0)

        coalition = former.form_coalition(task)
        assert former.disband_coalition(coalition.coalition_id) is True
        assert former.get_coalition(coalition.coalition_id).status == CoalitionStatus.DISBANDED


# ============================================================
# 5. 共享技能库测试
# ============================================================
class TestSharedSkillLibrary:
    def test_contribute_and_download(self):
        shared_lib = SharedSkillLibrary()
        skill = Skill(
            name="test_skill",
            precondition=(0, 0),
            action_sequence=(0, 1),
            postcondition=(1, 1),
            expected_reward=1.0,
            success_count=5,
            total_count=10,
        )
        shared_name = shared_lib.contribute(skill, "agent_001")
        assert shared_name.startswith("shared_")
        assert len(shared_lib) == 1

        downloaded = shared_lib.download(shared_name, "agent_002")
        assert downloaded is not None
        assert downloaded.name == "test_skill"

    def test_endorse_and_verify(self):
        shared_lib = SharedSkillLibrary()
        skill = Skill("s1", (0,), (0,), (1,), 1.0, 5, 5)
        shared_name = shared_lib.contribute(skill, "agent_001")

        # 3次好评后应验证
        for i in range(3):
            shared_lib.endorse(shared_name, f"agent_{i}", 4.5)

        shared = shared_lib._shared_skills[shared_name]
        assert shared.verified is True
        assert shared.endorsements == 3

    def test_find_best(self):
        shared_lib = SharedSkillLibrary()
        s1 = Skill("s1", (0,), (0,), (1,), 0.5, 3, 10)  # 30%成功率
        s2 = Skill("s2", (0,), (1,), (1,), 1.0, 8, 10)  # 80%成功率
        shared_lib.contribute(s1, "a1")
        shared_lib.contribute(s2, "a2")

        best = shared_lib.find_best_shared_skill((0,))
        assert best is not None
        assert best.skill.success_rate > 0.5

    def test_sync_to_local(self):
        shared_lib = SharedSkillLibrary()
        skill = Skill("s1", (0,), (0,), (1,), 1.0, 5, 5)
        shared_lib.contribute(skill, "agent_001")

        local_lib = SkillLibrary()
        synced = shared_lib.sync_to_local(local_lib, "agent_002")
        assert len(synced) == 1
        assert len(local_lib) == 1

    def test_collect_from_local(self):
        shared_lib = SharedSkillLibrary()
        local_lib = SkillLibrary()
        # 添加高质量技能
        good_skill = Skill("good", (0,), (0,), (1,), 1.0, 8, 10)
        bad_skill = Skill("bad", (0,), (0,), (1,), 0.1, 1, 10)
        local_lib.add_skill(good_skill)
        local_lib.add_skill(bad_skill)

        contributed = shared_lib.collect_from_local(local_lib, "agent_001", min_success_rate=0.5)
        assert len(contributed) == 1  # 只有good_skill被收集

    def test_stats(self):
        shared_lib = SharedSkillLibrary()
        skill = Skill("s1", (0,), (0,), (1,), 1.0, 5, 5)
        shared_lib.contribute(skill, "a1")
        shared_lib.download("shared_s1", "a2")
        stats = shared_lib.get_stats()
        assert stats["total_skills"] == 1
        assert stats["total_downloads"] == 1


# ============================================================
# 6. 端到端：多Agent协作
# ============================================================
class TestEndToEndC:
    def test_multi_agent_skill_sharing(self):
        """多Agent技能共享：Agent A学会技能，Agent B下载使用。"""
        # Agent A学习并贡献技能
        local_lib_a = SkillLibrary()
        skill = Skill(
            name="navigation_skill",
            precondition=(0, 0),
            action_sequence=(2, 2, 1, 1),  # 右右下下
            postcondition=(2, 2),
            expected_reward=1.0,
            success_count=8,
            total_count=10,
        )
        local_lib_a.add_skill(skill)

        # 共享到共享库
        shared_lib = SharedSkillLibrary()
        contributed = shared_lib.collect_from_local(local_lib_a, "agent_A")
        assert len(contributed) > 0

        # Agent B从共享库下载
        local_lib_b = SkillLibrary()
        synced = shared_lib.sync_to_local(local_lib_b, "agent_B")
        assert len(synced) > 0
        assert local_lib_b.find_by_precondition((0, 0)) is not None

    def test_full_coalition_workflow(self):
        """完整联盟工作流：注册→分配角色→形成联盟→执行。"""
        # 1. 注册多个Agent
        registry = AgentRegistry()
        drive_profiles = [
            (0.2, 0.4, 0.9),  # 探索者
            (0.9, 0.5, 0.2),  # 守护者
            (0.5, 0.9, 0.4),  # 协调者
        ]
        for i, profile in enumerate(drive_profiles):
            registry.register(
                f"Agent{i}",
                AgentCapability(drive_profile=profile, performance_score=0.8)
            )

        # 2. 设置消息总线
        bus = MessageBus()
        for reg in registry.get_all_agents():
            bus.register_agent(reg.agent_id)

        # 3. 形成联盟
        former = CoalitionFormer(registry, bus)
        decomposer = TaskDecomposer(grid_size=5)
        task = decomposer.decompose_path_task(
            start=(0, 0), end=(4, 4), n_waypoints=2, task_name="collab_task"
        )

        coalition = former.form_coalition(task)
        assert coalition is not None
        assert coalition.size == 3

        # 4. 验证角色分配
        roles = {a.role for a in coalition.role_assignments}
        assert Role.EXPLORER in roles or Role.GUARDIAN in roles

        # 5. 验证子目标分配
        assert len(coalition.subgoal_assignments) == task.num_sub_goals

        # 6. 验证消息通知
        # 至少有一个Agent收到了协调消息
        any_messages = any(
            bus.has_messages(reg.agent_id)
            for reg in registry.get_all_agents()
        )
        assert any_messages

        # 7. 更新进度并完成
        former.update_progress(coalition.coalition_id, task.num_sub_goals, task.num_sub_goals)
        assert former.get_coalition(coalition.coalition_id).status == CoalitionStatus.COMPLETED

    def test_collective_intelligence(self):
        """集体智慧：多个Agent贡献技能，形成更丰富的共享库。"""
        shared_lib = SharedSkillLibrary()

        # 3个Agent各自学会不同技能
        skills = [
            Skill("open_door", (0,), (0, 1), (1,), 1.0, 7, 10),
            Skill("pick_key", (1,), (2, 3), (2,), 1.0, 8, 10),
            Skill("unlock", (2,), (4, 5), (3,), 1.0, 9, 10),
        ]
        for i, skill in enumerate(skills):
            local_lib = SkillLibrary()
            local_lib.add_skill(skill)
            shared_lib.collect_from_local(local_lib, f"agent_{i}")

        assert len(shared_lib) == 3

        # 新Agent可以下载所有技能
        new_agent_lib = SkillLibrary()
        synced = shared_lib.sync_to_local(new_agent_lib, "agent_new", max_skills=10)
        assert len(synced) == 3

        # 验证技能质量
        top_skills = shared_lib.get_top_skills(3)
        assert len(top_skills) == 3
        # 所有技能都应该有较高的成功率
        for s in top_skills:
            assert s.skill.success_rate > 0.5
