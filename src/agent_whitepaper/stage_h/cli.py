"""阶段H-2：命令行入口 - 系统的"启动按钮"。

第一性原理：
    CLI是人与系统交互的最简接口。一个实用系统必须能被命令行启动：
    1. run: 启动永续自主进化系统
    2. demo: 快速演示系统能力
    3. init: 生成默认配置文件
    4. status: 查询运行状态（从持久化数据库）

    CLI让系统从"代码库"变为"可执行程序"。
"""

import argparse
import sqlite3
import sys
import time
from typing import List, Optional

from agent_whitepaper.stage_g.orchestrator import SystemOrchestrator

from .config import ConfigLoader, load_config


# ============================================================
# 格式化输出工具
# ============================================================
def _print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _print_kv(key: str, value, indent: int = 2) -> None:
    prefix = " " * indent
    if isinstance(value, float):
        print(f"{prefix}{key}: {value:.4f}")
    elif isinstance(value, dict):
        print(f"{prefix}{key}:")
        for k, v in value.items():
            _print_kv(k, v, indent + 2)
    else:
        print(f"{prefix}{key}: {value}")


def _format_metrics_report(report: dict) -> str:
    """格式化指标报告为可读文本。"""
    lines = []
    lines.append(f"  迭代次数: {report.get('iteration', 0)}")
    lines.append(f"  状态: {report.get('status', 'unknown')}")
    lines.append(f"  运行时间: {report.get('uptime', 0):.1f}s")
    lines.append(f"  Agent数: {report.get('agent_count', 0)}")
    lines.append(f"  技能数: {report.get('skill_count', 0)}")
    lines.append(f"  任务完成: {report.get('tasks_completed', 0)}")
    lines.append(f"  任务失败: {report.get('tasks_failed', 0)}")

    snap = report.get("latest_snapshot")
    if snap:
        lines.append("")
        lines.append("  最新快照:")
        lines.append(f"    平均成功率: {snap.get('avg_success_rate', 0):.4f}")
        lines.append(f"    平均奖励: {snap.get('avg_reward', 0):.4f}")
        lines.append(f"    Q表条目: {snap.get('total_q_entries', 0)}")
        lines.append(f"    总情节数: {snap.get('total_episodes', 0)}")
        lines.append(f"    健康分: {snap.get('health_score', 0):.4f}")
        lines.append(f"    蒸馏次数: {snap.get('distillation_count', 0)}")
        lines.append(f"    恢复次数: {snap.get('recovery_count', 0)}")

    cum = report.get("cumulative", {})
    if cum:
        lines.append("")
        lines.append("  累计统计:")
        for k, v in cum.items():
            lines.append(f"    {k}: {v}")

    trends = {
        "成功率趋势": report.get("success_rate_trend", "n/a"),
        "技能数趋势": report.get("skill_count_trend", "n/a"),
    }
    lines.append("")
    lines.append("  趋势分析:")
    for k, v in trends.items():
        lines.append(f"    {k}: {v}")

    return "\n".join(lines)


# ============================================================
# 命令处理函数
# ============================================================
def cmd_init(args: argparse.Namespace) -> int:
    """生成默认配置文件。"""
    path = args.output
    loader = ConfigLoader()
    loader.generate_default_yaml(path)
    print(f"默认配置已生成: {path}")
    print("修改后运行: agentwp run --config {path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """启动永续自主进化系统。"""
    # 加载配置
    overrides = {}
    if args.iterations is not None:
        overrides["max_iterations"] = args.iterations
    if args.agents is not None:
        overrides["n_agents"] = args.agents
    if args.grid is not None:
        overrides["grid_size"] = args.grid
    if args.memory:
        overrides["db_path"] = ":memory:"

    config = load_config(path=args.config, **overrides)

    _print_section("Agent Whitepaper 永续自主进化系统")
    _print_kv("网格大小", config.grid_size)
    _print_kv("Agent数", config.n_agents)
    _print_kv("数据库", config.db_path)
    _print_kv("最大迭代", config.max_iterations or "无限")
    print()

    # 创建并运行
    orch = SystemOrchestrator(config)
    orch.initialize()

    _print_section("系统启动")
    print(f"  状态: {orch.status.value}")
    print(f"  已注册Agent: {len(orch.agents)}")

    # 运行
    report_interval = args.report_interval
    start_time = time.time()

    try:
        if config.max_iterations is not None:
            # 有限迭代模式
            for i in range(config.max_iterations):
                orch.run_iteration()
                if (i + 1) % report_interval == 0:
                    elapsed = time.time() - start_time
                    snap = orch.metrics.get_latest()
                    if snap:
                        print(
                            f"  [{i + 1}/{config.max_iterations}] "
                            f"成功率={snap.avg_success_rate:.3f} "
                            f"技能={snap.skill_count} "
                            f"Q条目={snap.total_q_entries} "
                            f"健康={snap.health_score:.3f} "
                            f"({elapsed:.1f}s)"
                        )
        else:
            # 无限模式
            print("  进入永续运行模式（Ctrl+C 停止）...")
            count = 0
            while True:
                orch.run_iteration()
                count += 1
                if count % report_interval == 0:
                    elapsed = time.time() - start_time
                    snap = orch.metrics.get_latest()
                    if snap:
                        print(
                            f"  [{count}] "
                            f"成功率={snap.avg_success_rate:.3f} "
                            f"技能={snap.skill_count} "
                            f"Q条目={snap.total_q_entries} "
                            f"({elapsed:.1f}s)"
                        )
    except KeyboardInterrupt:
        print("\n  收到中断信号，正在停止...")

    orch.stop("cli")
    # 最终报告
    _print_section("最终报告")
    report = orch.get_metrics()
    print(_format_metrics_report(report))

    _print_section("Agent详情")
    for s in orch.get_agent_summaries():
        print(f"  {s['agent_id']}: "
              f"成功率={s['success_rate']:.3f} "
              f"情节={s['total_episodes']} "
              f"Q条目={s['q_entries']}")

    print()
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """快速演示系统能力。"""
    _print_section("Agent Whitepaper 快速演示")

    config = load_config(
        n_agents=3,
        grid_size=5,
        db_path=":memory:",
        max_iterations=30,
        skill_abstract_interval=3,
        meta_learn_interval=5,
        coalition_interval=8,
        distill_interval=10,
        evolve_interval=7,
    )

    orch = SystemOrchestrator(config)
    orch.initialize()

    print(f"  启动 {len(orch.agents)} 个Agent，运行30轮迭代...\n")

    start = time.time()
    for _ in range(30):
        orch.run_iteration()
    elapsed = time.time() - start

    _print_section("演示结果")
    report = orch.get_metrics()
    print(_format_metrics_report(report))

    _print_section("能力验证")
    # 能力1：多Agent
    print(f"  1. 多Agent协作: {len(orch.agents)}个Agent运行，"
          f"消息总线统计={orch.message_bus.get_stats()['total_sent']}条消息")
    # 能力2：技能增强
    print(f"  2. 技能增强: Q表共{report['latest_snapshot']['total_q_entries']}条目，"
          f"技能库{report['skill_count']}个技能")
    # 能力3：长任务
    print(f"  3. 长任务执行: 完成{report['tasks_completed']}个任务，"
          f"总情节{report['latest_snapshot']['total_episodes']}")
    # 能力4：永续运行
    print(f"  4. 永续运行: 30轮迭代全部完成，耗时{elapsed:.1f}s，"
          f"恢复{report['cumulative']['recovery_count']}次")

    print()
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """查询持久化数据库中的运行状态。"""
    from agent_whitepaper.stage_d.persistence import PersistenceEngine

    db_path = args.db
    try:
        engine = PersistenceEngine(db_path)
    except (OSError, sqlite3.Error) as e:  # 边界层：DB打开失败（文件不存在/权限/格式）
        print(f"无法打开数据库 {db_path}: {e}")
        return 1

    _print_section(f"运行状态 ({db_path})")

    # 获取最近的检查点
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 检查点
        cursor.execute(
            "SELECT created_at, description FROM checkpoints ORDER BY created_at DESC LIMIT 5"
        )
        checkpoints = cursor.fetchall()
        if checkpoints:
            print("\n  最近检查点:")
            for ts, desc in checkpoints:
                print(f"    [{ts}] {desc}")
        else:
            print("\n  无检查点记录")

        # 运行日志
        cursor.execute(
            "SELECT timestamp, event_type, message FROM run_log ORDER BY timestamp DESC LIMIT 10"
        )
        logs = cursor.fetchall()
        if logs:
            print("\n  最近日志:")
            for ts, etype, msg in logs:
                print(f"    [{ts}] {etype}: {msg}")

        conn.close()
    except sqlite3.Error as e:  # 边界层：DB查询失败（表不存在/SQL错误）
        print(f"  读取数据库失败: {e}")

    engine.close()
    return 0


# ============================================================
# 主入口
# ============================================================
def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="agentwp",
        description="Agent Whitepaper - 永续自主进化Agent系统",
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # init
    p_init = subparsers.add_parser("init", help="生成默认配置文件")
    p_init.add_argument(
        "-o", "--output", default="config.yaml",
        help="输出配置文件路径（默认: config.yaml）"
    )
    p_init.set_defaults(func=cmd_init)

    # run
    p_run = subparsers.add_parser("run", help="启动永续自主进化系统")
    p_run.add_argument(
        "-c", "--config", default=None,
        help="YAML配置文件路径"
    )
    p_run.add_argument(
        "-n", "--iterations", type=int, default=None,
        help="运行迭代次数（默认: 无限）"
    )
    p_run.add_argument(
        "-a", "--agents", type=int, default=None,
        help="Agent数量"
    )
    p_run.add_argument(
        "-g", "--grid", type=int, default=None,
        help="网格大小"
    )
    p_run.add_argument(
        "--memory", action="store_true",
        help="使用内存数据库（不持久化）"
    )
    p_run.add_argument(
        "--report-interval", type=int, default=10,
        help="报告间隔（迭代次数，默认: 10）"
    )
    p_run.set_defaults(func=cmd_run)

    # demo
    p_demo = subparsers.add_parser("demo", help="快速演示系统能力")
    p_demo.set_defaults(func=cmd_demo)

    # status
    p_status = subparsers.add_parser("status", help="查询运行状态")
    p_status.add_argument(
        "-d", "--db", default="agent_system.db",
        help="数据库路径（默认: agent_system.db）"
    )
    p_status.set_defaults(func=cmd_status)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI主入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
