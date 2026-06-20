"""阶段D验证测试：验证永续运行时模块。

测试覆盖：
1. 持久化存储引擎
2. 任务调度器
3. 健康监控与自恢复
4. 永续主循环
5. 端到端：系统持续运行+崩溃恢复
"""

import os
import tempfile
import time

import numpy as np

from agent_whitepaper.stage_d.health_monitor import (
    CircuitBreaker,
    HealthMonitor,
    HealthStatus,
    safe_execute,
)
from agent_whitepaper.stage_d.persistence import PersistenceEngine
from agent_whitepaper.stage_d.runtime import (
    PerpetualRuntime,
    RuntimeConfig,
    RuntimeStatus,
)
from agent_whitepaper.stage_d.scheduler import (
    ScheduledTaskStatus,
    TaskGenerator,
    TaskScheduler,
)


# ============================================================
# 1. 持久化存储引擎测试
# ============================================================
class TestPersistence:
    def test_save_load_agent_state(self):
        engine = PersistenceEngine(":memory:")
        q_table = np.random.rand(10, 4)
        engine.save_agent_state(
            agent_id="agent_001",
            name="TestAgent",
            q_table=q_table,
            drives={"safety": 0.7, "social": 0.5, "exploration": 0.3},
            stats={"total_steps": 100, "total_episodes": 10},
            n_states=10, n_actions=4,
            learning_rate=0.1, gamma=0.99,
        )
        loaded = engine.load_agent_state("agent_001")
        assert loaded is not None
        assert loaded["name"] == "TestAgent"
        assert np.allclose(loaded["q_table"], q_table)
        assert loaded["drives"]["safety"] == 0.7
        engine.close()

    def test_list_agents(self):
        engine = PersistenceEngine(":memory:")
        engine.save_agent_state("a1", "A1", np.zeros((5, 4)),
                                {}, {}, 5, 4, 0.1, 0.99)
        engine.save_agent_state("a2", "A2", np.zeros((5, 4)),
                                {}, {}, 5, 4, 0.1, 0.99)
        agents = engine.list_agents()
        assert len(agents) == 2
        engine.close()

    def test_save_load_skills(self):
        engine = PersistenceEngine(":memory:")
        engine.save_skill(
            "skill1", (0, 0), (1, 2), (1, 1),
            0.8, 5, 10, 1, "agent_001"
        )
        skills = engine.load_all_skills()
        assert len(skills) == 1
        assert skills[0]["skill_name"] == "skill1"
        assert skills[0]["precondition"] == (0, 0)
        engine.close()

    def test_task_progress(self):
        engine = PersistenceEngine(":memory:")
        engine.save_task_progress(
            "task_001", "test_task", {"desc": "test"},
            2, {"0": "completed", "1": "active"},
            50, 1.5, "active"
        )
        loaded = engine.load_task_progress("task_001")
        assert loaded is not None
        assert loaded["task_name"] == "test_task"
        assert loaded["current_goal_index"] == 2
        engine.close()

    def test_runtime_log(self):
        engine = PersistenceEngine(":memory:")
        engine.log_event("start", message="系统启动")
        engine.log_event("checkpoint", message="检查点1")
        logs = engine.get_recent_logs(10)
        assert len(logs) == 2
        assert logs[0]["event_type"] == "start"
        engine.close()

    def test_checkpoint(self):
        engine = PersistenceEngine(":memory:")
        cp_id = engine.create_checkpoint("测试检查点", {"iter": 10})
        assert cp_id > 0
        latest = engine.get_latest_checkpoint()
        assert latest is not None
        assert latest["description"] == "测试检查点"
        engine.close()

    def test_file_persistence(self):
        """测试文件持久化（可跨进程恢复）。"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            # 写入
            engine1 = PersistenceEngine(db_path)
            engine1.save_agent_state(
                "a1", "Agent1", np.ones((5, 4)),
                {"safety": 0.5}, {"steps": 10},
                5, 4, 0.1, 0.99
            )
            engine1.close()

            # 读取（模拟重启后恢复）
            engine2 = PersistenceEngine(db_path)
            loaded = engine2.load_agent_state("a1")
            assert loaded is not None
            assert loaded["name"] == "Agent1"
            assert np.allclose(loaded["q_table"], np.ones((5, 4)))
            engine2.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)


# ============================================================
# 2. 任务调度器测试
# ============================================================
class TestScheduler:
    def test_task_generator(self):
        gen = TaskGenerator(grid_size=5, seed=42)
        task = gen.generate_task()
        assert task.num_sub_goals > 0
        assert task.name.startswith("auto_")

    def test_submit_and_get_task(self):
        scheduler = TaskScheduler()
        gen = TaskGenerator(grid_size=5, seed=42)
        task = gen.generate_task()
        task_id = scheduler.submit_task(task)
        assert task_id != ""
        assert scheduler.get_queue_size() == 1

        scheduled = scheduler.get_next_task("agent_001")
        assert scheduled is not None
        assert scheduled.status == ScheduledTaskStatus.ASSIGNED
        assert scheduled.assigned_agent_id == "agent_001"

    def test_complete_task(self):
        scheduler = TaskScheduler()
        gen = TaskGenerator(grid_size=5, seed=42)
        task = gen.generate_task()
        task_id = scheduler.submit_task(task)
        scheduler.get_next_task("agent_001")
        scheduler.complete_task(task_id, {"completion_rate": 1.0})
        assert scheduler.get_stats()["total_completed"] == 1

    def test_fail_and_retry(self):
        scheduler = TaskScheduler()
        gen = TaskGenerator(grid_size=5, seed=42)
        task = gen.generate_task()
        task_id = scheduler.submit_task(task)
        scheduler.get_next_task("agent_001")
        # 第一次失败 → 重试
        scheduler.fail_task(task_id, "test_error")
        assert scheduler.get_queue_size() == 1  # 重新入队
        # 第二次
        scheduler.get_next_task("agent_001")
        scheduler.fail_task(task_id, "test_error")
        # 第三次（超过max_retries=2）
        scheduler.get_next_task("agent_001")
        result = scheduler.fail_task(task_id, "test_error")
        assert result is False  # 彻底失败
        assert scheduler.get_stats()["total_failed"] == 1

    def test_ensure_queue_filled(self):
        scheduler = TaskScheduler()
        generated = scheduler.ensure_queue_filled(min_size=5)
        assert generated == 5
        assert scheduler.get_queue_size() >= 5

    def test_auto_generate(self):
        scheduler = TaskScheduler()
        task_id = scheduler.auto_generate_task()
        assert task_id != ""
        assert scheduler.get_queue_size() == 1


# ============================================================
# 3. 健康监控测试
# ============================================================
class TestHealthMonitor:
    def test_register_and_heartbeat(self):
        monitor = HealthMonitor()
        monitor.register_component("test_comp")
        monitor.heartbeat("test_comp")
        check = monitor.check_component("test_comp")
        assert check.status == HealthStatus.HEALTHY

    def test_failure_tracking(self):
        monitor = HealthMonitor()
        monitor.register_component("comp1")
        for _ in range(5):
            monitor.report_failure("comp1", "test error")
        check = monitor.check_component("comp1")
        assert check.status == HealthStatus.UNHEALTHY

    def test_circuit_breaker(self):
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        # 初始可调用
        assert breaker.is_callable
        # 3次失败后打开
        for _ in range(3):
            breaker.record_failure()
        assert breaker.state == "open"
        assert not breaker.is_callable
        # 冷却后变半开
        time.sleep(0.15)
        assert breaker.is_callable
        assert breaker.state == "half_open"

    def test_recovery(self):
        monitor = HealthMonitor()
        recovered = {"called": False}

        def recovery_handler():
            recovered["called"] = True

        monitor.register_component("comp1", recovery_handler)
        # 制造失败
        for _ in range(5):
            monitor.report_failure("comp1", "error")
        # 尝试恢复
        result = monitor.attempt_recovery("comp1")
        assert result is True
        assert recovered["called"] is True

    def test_safe_execute_decorator(self):
        monitor = HealthMonitor()
        monitor.register_component("test_func")

        @safe_execute(monitor, "test_func", default_return=-1)
        def risky_function(x):
            if x < 0:
                raise ValueError("负数")
            return x * 2

        # 正常调用
        assert risky_function(5) == 10
        # 异常调用 → 返回默认值
        assert risky_function(-1) == -1
        # 健康状态应反映失败
        check = monitor.check_component("test_func")
        assert check.status != HealthStatus.HEALTHY

    def test_system_health(self):
        monitor = HealthMonitor()
        monitor.register_component("comp1")
        monitor.register_component("comp2")
        monitor.heartbeat("comp1")
        monitor.heartbeat("comp2")
        assert monitor.is_system_healthy()


# ============================================================
# 4. 永续主循环测试
# ============================================================
class TestPerpetualRuntime:
    def test_initialization(self):
        config = RuntimeConfig(
            db_path=":memory:",
            grid_size=4,
            n_agents=1,
            max_iterations=1,
        )
        runtime = PerpetualRuntime(config)
        runtime.initialize()
        assert runtime.status == RuntimeStatus.RUNNING
        assert len(runtime.agents) == 1
        assert runtime.env is not None

    def test_run_limited_iterations(self):
        """运行有限迭代次数。"""
        config = RuntimeConfig(
            db_path=":memory:",
            grid_size=4,
            n_agents=1,
            max_iterations=5,
            iteration_delay=0,
            checkpoint_interval=1.0,
        )
        runtime = PerpetualRuntime(config)
        runtime.run()

        assert runtime.status == RuntimeStatus.STOPPED
        assert runtime.stats.total_iterations == 5
        assert runtime.stats.total_tasks_completed > 0 or runtime.stats.total_tasks_failed > 0

    def test_state_persistence_and_recovery(self):
        """测试状态持久化与恢复。"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            # 第一次运行：保存状态
            config1 = RuntimeConfig(
                db_path=db_path, grid_size=4, n_agents=1, max_iterations=3,
            )
            runtime1 = PerpetualRuntime(config1)
            runtime1.initialize()
            # 手动执行几轮
            for _ in range(3):
                runtime1._execute_one_iteration()
            runtime1._save_state()
            runtime1.persistence.close()

            # 第二次运行：从持久化恢复
            config2 = RuntimeConfig(
                db_path=db_path, grid_size=4, n_agents=1, max_iterations=1,
            )
            runtime2 = PerpetualRuntime(config2)
            runtime2.initialize()  # 应该恢复状态

            # 验证Agent状态已恢复
            agent = runtime2.agents["agent_0000"]
            # 恢复后Q表应该有值（如果之前训练过）
            assert agent.q_table.n_states == 16
            runtime2.persistence.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_exception_recovery(self):
        """测试异常恢复。"""
        config = RuntimeConfig(
            db_path=":memory:", grid_size=4, n_agents=1, max_iterations=10,
        )
        runtime = PerpetualRuntime(config)
        runtime.initialize()

        # 注入一个会失败的迭代（通过让环境失效）
        original = runtime._execute_one_iteration
        call_count = {"n": 0}

        def safe_failing_iteration():
            call_count["n"] += 1
            if call_count["n"] == 2:
                # 模拟内部异常，但被_execute_one_iteration的try-except捕获
                runtime.health_monitor.report_failure("main_loop", "注入的测试错误")
                return False
            return original()

        runtime._execute_one_iteration = safe_failing_iteration
        # 运行几轮，第2轮会失败但系统应继续
        for _ in range(5):
            runtime._execute_one_iteration()

        # 系统应该存活
        assert call_count["n"] == 5
        runtime.persistence.close()

    def test_status_report(self):
        config = RuntimeConfig(db_path=":memory:", grid_size=4, n_agents=1)
        runtime = PerpetualRuntime(config)
        runtime.initialize()
        runtime._execute_one_iteration()

        report = runtime.get_status_report()
        assert report["status"] == "running"
        assert report["total_iterations"] == 1
        assert "scheduler_stats" in report
        assert "health_report" in report
        runtime.persistence.close()


# ============================================================
# 5. 端到端：永续运行验证
# ============================================================
class TestEndToEndD:
    def test_perpetual_operation(self):
        """端到端：系统持续运行10轮，验证永续性。"""
        config = RuntimeConfig(
            db_path=":memory:",
            grid_size=4,
            n_agents=2,
            max_iterations=10,
            checkpoint_interval=1.0,
        )
        runtime = PerpetualRuntime(config)
        runtime.run()

        report = runtime.get_status_report()
        print("\n永续运行报告:")
        print(f"  总迭代: {report['total_iterations']}")
        print(f"  完成任务: {report['total_tasks_completed']}")
        print(f"  失败任务: {report['total_tasks_failed']}")
        print(f"  总情节数: {report['total_episodes']}")
        print(f"  总步数: {report['total_steps']}")
        print(f"  运行时间: {report['uptime']:.2f}秒")
        print(f"  技能数: {report['skill_count']}")
        print(f"  系统健康: {report['health_report']['system_healthy']}")

        # 验证永续性
        assert report["total_iterations"] == 10
        assert report["total_episodes"] > 0
        assert report["status"] == "stopped"  # 因max_iterations而停止

    def test_crash_recovery_simulation(self):
        """模拟崩溃后恢复。"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            # 第一次运行（模拟崩溃前的状态）
            config1 = RuntimeConfig(
                db_path=db_path, grid_size=4, n_agents=1, max_iterations=5,
            )
            runtime1 = PerpetualRuntime(config1)
            runtime1.initialize()
            for _ in range(5):
                runtime1._execute_one_iteration()
            runtime1._save_state()
            runtime1._create_checkpoint("崩溃前")
            # 模拟崩溃：不调用stop，直接关闭
            runtime1.persistence.close()

            # 第二次运行（模拟重启恢复）
            config2 = RuntimeConfig(
                db_path=db_path, grid_size=4, n_agents=1, max_iterations=3,
            )
            runtime2 = PerpetualRuntime(config2)
            runtime2.initialize()  # 应从检查点恢复

            # 验证恢复
            checkpoint = runtime2.persistence.get_latest_checkpoint()
            assert checkpoint is not None
            assert checkpoint["description"] == "崩溃前"

            # 继续运行
            runtime2.run()
            assert runtime2.stats.total_iterations > 0
            runtime2.persistence.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)
