"""示例04：持久化与状态恢复。

本示例演示：
1. 使用持久化数据库保存系统状态
2. 创建检查点
3. 从检查点恢复状态
4. 查询运行日志

运行：PYTHONPATH=src python examples/04_persistence_and_recovery.py
"""

import os
import tempfile

from agent_whitepaper import PersistenceEngine


def main() -> None:
    print("=" * 60)
    print("  示例04：持久化与状态恢复")
    print("=" * 60)

    # 1. 创建临时数据库
    db_path = os.path.join(tempfile.gettempdir(), "agent_example.db")
    print(f"\n数据库: {db_path}")

    engine = PersistenceEngine(db_path)
    print(f"存储统计: {engine.get_storage_stats()}")

    # 2. 记录运行日志
    print("\n记录运行日志...")
    engine.log_event("start", message="系统启动")
    engine.log_event("info", message="Agent-0 注册成功")
    engine.log_event("info", message="Agent-1 注册成功")
    engine.log_event("info", message="Agent-2 注册成功")
    engine.log_event("iteration", message="迭代 10 完成，成功率=0.65")
    engine.log_event("iteration", message="迭代 20 完成，成功率=0.78")

    # 3. 创建检查点
    print("\n创建检查点...")
    cp1 = engine.create_checkpoint(
        "迭代20检查点",
        {
            "iteration": 20,
            "agents": {"agent_0": {"success_rate": 0.78}},
            "skills": ["skill_1", "skill_2"],
        },
    )
    print(f"  检查点ID: {cp1}")

    cp2 = engine.create_checkpoint(
        "迭代40检查点",
        {
            "iteration": 40,
            "agents": {"agent_0": {"success_rate": 0.92}},
            "skills": ["skill_1", "skill_2", "skill_3"],
        },
    )
    print(f"  检查点ID: {cp2}")

    # 4. 列出检查点
    print("\n检查点列表:")
    for cp in engine.list_checkpoints(5):
        print(f"  [{cp['checkpoint_id']}] {cp['description']}")

    # 5. 从最新检查点恢复
    print("\n从最新检查点恢复...")
    latest = engine.get_latest_checkpoint()
    if latest:
        snapshot = latest["state_snapshot"]
        print(f"  检查点: {latest['description']}")
        print(f"  迭代: {snapshot.get('iteration')}")
        print(f"  技能: {snapshot.get('skills')}")
        agents = snapshot.get("agents", {})
        for aid, stats in agents.items():
            print(f"  {aid}: {stats}")

    # 6. 查询运行日志
    print("\n运行日志:")
    for log in engine.get_recent_logs(5):
        print(f"  [{log['event_type']}] {log['message']}")

    # 7. 清理
    final_stats = engine.get_storage_stats()
    engine.close()
    print(f"\n存储统计: {final_stats}")
    print(f"\n提示: 数据库已保存到 {db_path}")


if __name__ == "__main__":
    main()
