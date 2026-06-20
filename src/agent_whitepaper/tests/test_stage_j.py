"""阶段J验证测试 + 安全约束与资源管理。

测试覆盖：
1. 安全监控器（6类异常检测）
2. 安全动作过滤器
3. 资源管理器（监控/限制/背压）
4. 资源感知控制器
5. 端到端：安全+资源保护的运行
"""


import numpy as np

from agent_whitepaper.stage_j.resource_manager import (
    BackpressureLevel,
    ResourceAwareController,
    ResourceLimits,
    ResourceManager,
    ResourceUsage,
)
from agent_whitepaper.stage_j.safety_bounds import (
    SafeActionFilter,
    SafetyConfig,
    SafetyMonitor,
)


# ============================================================
# 1. 安全监控器测试
# ============================================================
class TestSafetyMonitor:
    def test_safe_initial_state(self):
        monitor = SafetyMonitor()
        assert monitor.is_safe
        assert not monitor.should_stop
        report = monitor.get_safety_report()
        assert report["is_safe"]
        assert report["total_violations"] == 0

    def test_forbidden_action_blocked(self):
        config = SafetyConfig(forbidden_actions={2})
        monitor = SafetyMonitor(config)
        allowed, violation = monitor.check_action(state=0, action=2)
        assert not allowed
        assert violation is not None
        assert violation.violation_type == "forbidden_action"

    def test_allowed_action_passes(self):
        config = SafetyConfig(forbidden_actions={2})
        monitor = SafetyMonitor(config)
        allowed, violation = monitor.check_action(state=0, action=1)
        assert allowed
        assert violation is None

    def test_consecutive_failures_detected(self):
        config = SafetyConfig(max_consecutive_failures=3)
        monitor = SafetyMonitor(config)
        # 3次失败
        for _ in range(3):
            v = monitor.record_step(state=0, reward=-0.1, done=True, success=False)
        assert v is not None
        assert v.violation_type == "consecutive_failures"

    def test_success_resets_failure_count(self):
        config = SafetyConfig(max_consecutive_failures=5)
        monitor = SafetyMonitor(config)
        monitor.record_step(state=0, reward=1.0, done=True, success=False)
        monitor.record_step(state=0, reward=1.0, done=True, success=False)
        monitor.record_step(state=0, reward=1.0, done=True, success=True)  # 成功重置
        assert monitor._consecutive_failures == 0

    def test_negative_reward_streak_detected(self):
        config = SafetyConfig(max_negative_reward_streak=5)
        monitor = SafetyMonitor(config)
        for i in range(5):
            v = monitor.record_step(state=i, reward=-0.1, done=False, success=False)
        assert v is not None
        assert v.violation_type == "negative_reward_streak"

    def test_positive_reward_resets_streak(self):
        config = SafetyConfig(max_negative_reward_streak=5)
        monitor = SafetyMonitor(config)
        for _ in range(3):
            monitor.record_step(state=0, reward=-0.1, done=False, success=False)
        monitor.record_step(state=0, reward=0.1, done=False, success=False)
        assert monitor._negative_reward_streak == 0

    def test_state_loop_detected(self):
        config = SafetyConfig(state_loop_window=10, state_loop_threshold=3)
        monitor = SafetyMonitor(config)
        v = None
        for _ in range(3):
            v = monitor.record_step(state=5, reward=0.0, done=False, success=False)
        assert v is not None
        assert v.violation_type == "state_loop"

    def test_q_value_explosion_detected(self):
        config = SafetyConfig(max_q_value=100.0)
        monitor = SafetyMonitor(config)
        q_table = np.array([[200.0, 0.0], [0.0, 0.0]])
        v = monitor.check_q_table(q_table)
        assert v is not None
        assert v.violation_type == "q_value_explosion"
        assert monitor.should_stop

    def test_q_value_nan_detected(self):
        monitor = SafetyMonitor()
        q_table = np.array([[np.nan, 0.0], [0.0, 0.0]])
        v = monitor.check_q_table(q_table)
        assert v is not None
        assert v.violation_type == "q_value_nan"
        assert monitor.should_stop

    def test_q_table_normal_passes(self):
        monitor = SafetyMonitor()
        q_table = np.array([[1.0, 0.5], [0.3, 0.8]])
        v = monitor.check_q_table(q_table)
        assert v is None

    def test_episode_too_long_detected(self):
        config = SafetyConfig(max_episode_length=50)
        monitor = SafetyMonitor(config)
        v = monitor.check_episode_length(100)
        assert v is not None
        assert v.violation_type == "episode_too_long"

    def test_emergency_stop(self):
        monitor = SafetyMonitor()
        monitor.trigger_emergency_stop("测试停止")
        assert monitor.should_stop
        assert monitor.stop_reason == "测试停止"
        assert not monitor.is_safe

    def test_reset_state_tracking(self):
        monitor = SafetyMonitor()
        monitor.record_step(state=1, reward=0, done=False, success=False)
        monitor.record_step(state=2, reward=0, done=False, success=False)
        monitor.reset_state_tracking()
        assert len(monitor._recent_states) == 0
        assert len(monitor._state_counts) == 0

    def test_safety_report(self):
        monitor = SafetyMonitor()
        monitor.record_step(state=0, reward=-0.1, done=True, success=False)
        report = monitor.get_safety_report()
        assert "is_safe" in report
        assert "total_violations" in report
        assert "violations_by_level" in report
        assert "consecutive_failures" in report


# ============================================================
# 2. 安全动作过滤器测试
# ============================================================
class TestSafeActionFilter:
    def test_safe_action_passes(self):
        config = SafetyConfig(forbidden_actions={2})
        monitor = SafetyMonitor(config)
        f = SafeActionFilter(monitor)
        q = np.array([0.5, 0.3, 0.8])
        action = f.filter(state=0, proposed_action=1, q_values=q, n_actions=3)
        assert action == 1

    def test_forbidden_action_replaced(self):
        config = SafetyConfig(forbidden_actions={2})
        monitor = SafetyMonitor(config)
        f = SafeActionFilter(monitor)
        q = np.array([0.5, 0.3, 0.8])
        # 提议动作2被禁用，应选次优的安全动作0
        action = f.filter(state=0, proposed_action=2, q_values=q, n_actions=3)
        assert action == 0  # Q值0.5 > 0.3

    def test_all_forbidden_returns_proposed(self):
        config = SafetyConfig(forbidden_actions={0, 1, 2})
        monitor = SafetyMonitor(config)
        f = SafeActionFilter(monitor)
        q = np.array([0.5, 0.3, 0.8])
        action = f.filter(state=0, proposed_action=1, q_values=q, n_actions=3)
        # 全部禁用时返回原提议
        assert action == 1


# ============================================================
# 3. 资源管理器测试
# ============================================================
class TestResourceManager:
    def test_monitor_returns_usage(self):
        mgr = ResourceManager()
        usage = mgr.monitor(iteration_count=5)
        assert usage.iteration_count == 5
        assert usage.uptime_seconds >= 0

    def test_check_limits_no_violation(self):
        limits = ResourceLimits(max_memory_mb=10000, max_total_iterations=1000)
        mgr = ResourceManager(limits)
        usage = ResourceUsage(memory_mb=100, iteration_count=50)
        violations = mgr.check_limits(usage)
        assert len(violations) == 0

    def test_check_limits_memory_violation(self):
        limits = ResourceLimits(max_memory_mb=100)
        mgr = ResourceManager(limits)
        usage = ResourceUsage(memory_mb=200)
        violations = mgr.check_limits(usage)
        assert any("内存超限" in v for v in violations)

    def test_check_limits_iteration_violation(self):
        limits = ResourceLimits(max_total_iterations=100)
        mgr = ResourceManager(limits)
        usage = ResourceUsage(iteration_count=150)
        violations = mgr.check_limits(usage)
        assert any("迭代超限" in v for v in violations)

    def test_check_limits_buffer_violation(self):
        limits = ResourceLimits(max_memory_buffer_size=1000)
        mgr = ResourceManager(limits)
        usage = ResourceUsage(memory_buffer_size=2000)
        violations = mgr.check_limits(usage)
        assert any("记忆缓冲超限" in v for v in violations)

    def test_check_limits_skill_violation(self):
        limits = ResourceLimits(max_skill_library_size=500)
        mgr = ResourceManager(limits)
        usage = ResourceUsage(skill_library_size=800)
        violations = mgr.check_limits(usage)
        assert any("技能库超限" in v for v in violations)

    def test_check_limits_qtable_violation(self):
        limits = ResourceLimits(max_q_table_entries=1000)
        mgr = ResourceManager(limits)
        usage = ResourceUsage(q_table_entries=2000)
        violations = mgr.check_limits(usage)
        assert any("Q表超限" in v for v in violations)

    def test_backpressure_none(self):
        limits = ResourceLimits(max_memory_mb=1000, max_total_iterations=1000)
        mgr = ResourceManager(limits)
        usage = ResourceUsage(memory_mb=100, iteration_count=50)
        assert mgr.get_backpressure(usage) == BackpressureLevel.NONE

    def test_backpressure_light(self):
        limits = ResourceLimits(max_memory_mb=1000)
        mgr = ResourceManager(limits)
        usage = ResourceUsage(memory_mb=650)  # 65%
        assert mgr.get_backpressure(usage) == BackpressureLevel.LIGHT

    def test_backpressure_moderate(self):
        limits = ResourceLimits(max_memory_mb=1000)
        mgr = ResourceManager(limits)
        usage = ResourceUsage(memory_mb=800)  # 80%
        assert mgr.get_backpressure(usage) == BackpressureLevel.MODERATE

    def test_backpressure_heavy(self):
        limits = ResourceLimits(max_memory_mb=1000)
        mgr = ResourceManager(limits)
        usage = ResourceUsage(memory_mb=900)  # 90%
        assert mgr.get_backpressure(usage) == BackpressureLevel.HEAVY

    def test_backpressure_critical(self):
        limits = ResourceLimits(max_memory_mb=1000)
        mgr = ResourceManager(limits)
        usage = ResourceUsage(memory_mb=980)  # 98%
        assert mgr.get_backpressure(usage) == BackpressureLevel.CRITICAL

    def test_should_accept_task_none(self):
        limits = ResourceLimits(max_memory_mb=1000)
        mgr = ResourceManager(limits)
        mgr.monitor()
        mgr._usage_history[-1].memory_mb = 100
        assert mgr.should_accept_task("low")

    def test_should_accept_task_critical(self):
        limits = ResourceLimits(max_memory_mb=1000)
        mgr = ResourceManager(limits)
        mgr.monitor()
        mgr._usage_history[-1].memory_mb = 980
        assert not mgr.should_accept_task("high")

    def test_should_accept_task_heavy_only_high(self):
        limits = ResourceLimits(max_memory_mb=1000)
        mgr = ResourceManager(limits)
        mgr.monitor()
        mgr._usage_history[-1].memory_mb = 900
        assert mgr.should_accept_task("high")
        assert not mgr.should_accept_task("low")

    def test_recommend_delay(self):
        limits = ResourceLimits(max_memory_mb=1000)
        mgr = ResourceManager(limits)
        mgr.monitor()
        mgr._usage_history[-1].memory_mb = 100
        assert mgr.recommend_delay() == 0.0

        mgr._usage_history[-1].memory_mb = 980
        assert mgr.recommend_delay() == 2.0

    def test_usage_report(self):
        mgr = ResourceManager()
        mgr.monitor(iteration_count=10)
        report = mgr.get_usage_report()
        assert "current_usage" in report
        assert "limits" in report
        assert "backpressure" in report
        assert "violations" in report


# ============================================================
# 4. 资源感知控制器测试
# ============================================================
class TestResourceAwareController:
    def test_before_iteration_returns_decision(self):
        mgr = ResourceManager()
        ctrl = ResourceAwareController(mgr)
        decision = ctrl.before_iteration()
        assert "should_run" in decision
        assert "delay_seconds" in decision
        assert "backpressure" in decision

    def test_critical_resource_pauses(self):
        # 用迭代限制触发临界（内存需psutil，迭代计数可靠）
        limits = ResourceLimits(max_total_iterations=2)
        mgr = ResourceManager(limits)
        ctrl = ResourceAwareController(mgr)
        # 前2次正常（iteration 1, 2），第3次超限
        ctrl.before_iteration()  # iteration=1
        ctrl.before_iteration()  # iteration=2
        decision = ctrl.before_iteration()  # iteration=3 > 2 → 100%超限
        assert not decision["should_run"]
        assert ctrl.is_paused

    def test_resume(self):
        mgr = ResourceManager()
        ctrl = ResourceAwareController(mgr)
        ctrl._paused = True
        ctrl._pause_reason = "测试"
        ctrl.resume()
        assert not ctrl.is_paused

    def test_safety_stop_propagates(self):
        from agent_whitepaper.stage_j.safety_bounds import SafetyMonitor
        mgr = ResourceManager()
        monitor = SafetyMonitor()
        monitor.trigger_emergency_stop("安全停止")
        ctrl = ResourceAwareController(mgr, safety_monitor=monitor)
        decision = ctrl.before_iteration()
        assert not decision["should_run"]
        assert ctrl.is_paused


# ============================================================
# 5. 端到端：安全+资源保护
# ============================================================
class TestEndToEnd:
    def test_safe_operation_under_normal_load(self):
        """正常负载下系统正常运行。"""
        limits = ResourceLimits(max_memory_mb=10000, max_total_iterations=10000)
        mgr = ResourceManager(limits)
        monitor = SafetyMonitor()
        ctrl = ResourceAwareController(mgr, monitor)

        for i in range(10):
            decision = ctrl.before_iteration()
            assert decision["should_run"]
            ctrl.after_iteration()

        assert not ctrl.is_paused

    def test_safety_monitor_integrates_with_agent(self):
        """安全监控器与Agent运行集成。"""
        from agent_whitepaper.stage_a.agent import SelfDrivenAgent
        from agent_whitepaper.stage_a.environment import GridWorld

        env = GridWorld(size=4, seed=42)
        agent = SelfDrivenAgent(env.action_space, n_states=16, seed=42)
        monitor = SafetyMonitor(SafetyConfig(max_consecutive_failures=100))

        # 运行情节，安全监控器跟踪
        for _ in range(5):
            monitor.reset_state_tracking()
            obs = env.reset()
            state_idx = agent.q_table.state_to_index(obs, 4)
            for step in range(20):
                # 安全过滤
                agent.q_table.get(state_idx)
                action = agent.policy.select_action(
                    agent.q_table, state_idx, agent.drives, agent._rng
                )
                allowed, _ = monitor.check_action(state_idx, action)
                if not allowed:
                    continue
                _, next_obs, reward, done, info = agent.perceive_and_act(
                    obs, state_idx, env
                )
                monitor.record_step(state_idx, reward, done, info.get("reached_goal", False))
                obs = next_obs
                state_idx = agent.q_table.state_to_index(obs, 4)
                if done:
                    break

        # 系统应仍安全
        assert monitor.is_safe

    def test_resource_protection_prevents_overrun(self):
        """资源保护防止超限运行。"""
        limits = ResourceLimits(max_total_iterations=5)
        mgr = ResourceManager(limits)
        ctrl = ResourceAwareController(mgr)

        # 前5次正常
        for i in range(5):
            decision = ctrl.before_iteration()
            # 第6次应触发（iteration_count从1开始，第6次=6>5）
            if i < 5:
                pass  # 仍在限制内

        # 第6次迭代超限
        decision = ctrl.before_iteration()
        violations = decision["violations"]
        # 迭代数超限应被检测
        assert any("迭代超限" in v for v in violations) or not decision["should_run"]
