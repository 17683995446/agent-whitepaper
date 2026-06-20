"""示例02：多Agent协作系统。

本示例演示：
1. 配置多Agent系统
2. 启动系统编排器
3. 运行迭代观察协作
4. 查看系统指标

运行：PYTHONPATH=src python examples/02_multi_agent_system.py
"""

from agent_whitepaper.stage_g.orchestrator import (
    OrchestratorConfig,
    SystemOrchestrator,
)


def main() -> None:
    print("=" * 60)
    print("  示例02：多Agent协作系统")
    print("=" * 60)

    # 1. 配置系统：3个Agent，5x5网格，内存数据库
    config = OrchestratorConfig(
        grid_size=5,
        n_agents=3,
        db_path=":memory:",
        max_iterations=20,
        # 各能力触发间隔（紧凑以在20轮内触发）
        skill_abstract_interval=3,
        meta_learn_interval=5,
        coalition_interval=8,
        distill_interval=10,
        evolve_interval=7,
        seed=42,
    )
    print(f"\n配置: {config.n_agents}个Agent, {config.grid_size}x{config.grid_size}网格")
    print(f"迭代: {config.max_iterations}轮")

    # 2. 初始化系统
    print("\n初始化系统...")
    orch = SystemOrchestrator(config)
    orch.initialize()
    print(f"已注册Agent: {len(orch.agents)}")
    print(f"系统状态: {orch.status.value}")

    # 3. 运行迭代
    print("\n运行中...")
    for i in range(config.max_iterations):
        orch.run_iteration()
        snap = orch.metrics.get_latest()
        if snap and (i + 1) % 5 == 0:
            print(
                f"  [{i + 1:2d}/{config.max_iterations}] "
                f"成功率={snap.avg_success_rate:.3f} "
                f"技能={snap.skill_count} "
                f"Q条目={snap.total_q_entries} "
                f"健康={snap.health_score:.3f}"
            )

    # 4. 最终报告
    report = orch.get_metrics()
    orch.stop("done")

    print("\n" + "=" * 60)
    print("  系统报告")
    print("=" * 60)
    print(f"  总迭代: {report.get('iteration', 0)}")
    print(f"  运行时间: {report.get('uptime', 0):.1f}s")
    print(f"  Agent数: {report.get('agent_count', 0)}")
    print(f"  技能数: {report.get('skill_count', 0)}")
    print(f"  任务完成: {report.get('tasks_completed', 0)}")
    print(f"  任务失败: {report.get('tasks_failed', 0)}")

    snap = report.get("latest_snapshot")
    if snap:
        print("\n  最新快照:")
        print(f"    平均成功率: {snap.get('avg_success_rate', 0):.4f}")
        print(f"    平均奖励: {snap.get('avg_reward', 0):.4f}")
        print(f"    Q表条目: {snap.get('total_q_entries', 0)}")
        print(f"    健康分: {snap.get('health_score', 0):.4f}")

    # 5. Agent详情
    print("\n  Agent详情:")
    for s in orch.get_agent_summaries():
        print(
            f"    {s['agent_id']}: "
            f"成功率={s['success_rate']:.3f} "
            f"情节={s['total_episodes']} "
            f"Q条目={s['q_entries']}"
        )


if __name__ == "__main__":
    main()
