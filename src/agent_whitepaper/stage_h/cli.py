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
import logging
import os
import sqlite3
import sys
import time
from typing import List, Optional

from agent_whitepaper import __version__
from agent_whitepaper._exceptions import AgentWhitepaperError, ConfigError
from agent_whitepaper._logging import configure_logging
from agent_whitepaper.stage_g.orchestrator import SystemOrchestrator

from .config import (
    CONFIG_TEMPLATES,
    ConfigLoader,
    list_templates,
    load_config,
)
from .progress import (
    ProgressReporter,
    SilentReporter,
    predict_phases,
)
from .recovery import format_error_message
from .visualizer import generate_report_from_orchestrator
from .wizard import ConfigWizard


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
    """生成配置文件。

    第一性原理：
        "Sensible defaults"——好默认值降低认知负荷。
        预置模板覆盖常见场景，用户无需理解所有参数即可启动。
    """
    # 列出模板模式
    if getattr(args, "list_templates", False):
        _print_section("可用配置模板")
        for name in list_templates():
            desc = CONFIG_TEMPLATES[name].get("_description", "")
            print(f"  {name:<15} {desc}")
        print("\n  使用: agentwp init --template <名称> -o config.yaml")
        return 0

    path = args.output
    template = getattr(args, "template", None)
    loader = ConfigLoader()

    try:
        loader.generate_default_yaml(path, template=template)
    except ConfigError as e:
        print(f"配置错误: {e}", file=sys.stderr)
        return 1

    label = f"模板 '{template}'" if template else "默认配置"
    print(f"{label}已生成: {path}")
    print("修改后运行: agentwp run --config " + path)
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

    try:
        config = load_config(path=args.config, **overrides)
    except ConfigError as e:
        print(format_error_message(e), file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(format_error_message(e), file=sys.stderr)
        return 1

    _print_section("Agent Whitepaper 永续自主进化系统")
    _print_kv("网格大小", config.grid_size)
    _print_kv("Agent数", config.n_agents)
    _print_kv("数据库", config.db_path)
    _print_kv("最大迭代", config.max_iterations or "无限")
    print()

    # 创建并运行
    try:
        orch = SystemOrchestrator(config)
        orch.initialize()
    except AgentWhitepaperError as e:
        print(f"系统初始化失败: {e}", file=sys.stderr)
        return 1

    _print_section("系统启动")
    print(f"  状态: {orch.status.value}")
    print(f"  已注册Agent: {len(orch.agents)}")

    # 运行
    report_interval = args.report_interval
    is_quiet = getattr(args, "quiet", False)

    # 创建进度报告器
    if is_quiet:
        reporter = SilentReporter(
            total=config.max_iterations,
            report_interval=report_interval,
        )
    else:
        reporter = ProgressReporter(
            total=config.max_iterations,
            report_interval=report_interval,
            output=lambda msg: print(f"  {msg}"),
        )

    try:
        if config.max_iterations is not None:
            # 有限迭代模式
            for i in range(config.max_iterations):
                orch.run_iteration()
                if reporter.should_report(i + 1):
                    snap = orch.metrics.get_latest()
                    phases = predict_phases(
                        iteration=i + 1,
                        skill_abstract_interval=config.skill_abstract_interval,
                        meta_learn_interval=config.meta_learn_interval,
                        coalition_interval=config.coalition_interval,
                        distill_interval=config.distill_interval,
                        evolve_interval=config.evolve_interval,
                        checkpoint_interval=config.checkpoint_interval,
                        n_agents=config.n_agents,
                    )
                    reporter.report(i + 1, snap, phases)
        else:
            # 无限模式
            if not is_quiet:
                print("  进入永续运行模式（Ctrl+C 停止）...")
            count = 0
            while True:
                orch.run_iteration()
                count += 1
                if reporter.should_report(count):
                    snap = orch.metrics.get_latest()
                    phases = predict_phases(
                        iteration=count,
                        skill_abstract_interval=config.skill_abstract_interval,
                        meta_learn_interval=config.meta_learn_interval,
                        coalition_interval=config.coalition_interval,
                        distill_interval=config.distill_interval,
                        evolve_interval=config.evolve_interval,
                        checkpoint_interval=config.checkpoint_interval,
                        n_agents=config.n_agents,
                    )
                    reporter.report(count, snap, phases)
    except KeyboardInterrupt:
        if not is_quiet:
            print("\n  收到中断信号，正在停止...")

    orch.stop("cli")

    # 进度完成
    if not is_quiet:
        final_snap = orch.metrics.get_latest()
        reporter.finish(final_snap)

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

    n_agents = args.agents
    grid_size = args.grid
    n_iterations = args.iterations

    try:
        config = load_config(
            n_agents=n_agents,
            grid_size=grid_size,
            db_path=":memory:",
            max_iterations=n_iterations,
            skill_abstract_interval=3,
            meta_learn_interval=5,
            coalition_interval=8,
            distill_interval=10,
            evolve_interval=7,
        )

        orch = SystemOrchestrator(config)
        orch.initialize()
    except AgentWhitepaperError as e:
        print(f"演示启动失败: {e}", file=sys.stderr)
        return 1

    print(f"  启动 {len(orch.agents)} 个Agent，运行{n_iterations}轮迭代...\n")

    start = time.time()
    for _ in range(n_iterations):
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
    print(f"  4. 永续运行: {n_iterations}轮迭代全部完成，耗时{elapsed:.1f}s，"
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


def cmd_doctor(args: argparse.Namespace) -> int:
    """诊断环境：检查依赖、导入、配置等是否正常。

    第一性原理：
        "Fail fast, fail helpfully"——主动诊断优于被动排错。
        用户遇到问题时，doctor命令提供一站式环境检查，
        避免在多个地方分别排查。
    """
    _print_section("Agent Whitepaper 环境诊断")

    checks_passed = 0
    checks_failed = 0
    checks_warned = 0

    def check(name: str, status: str, detail: str = "") -> None:
        nonlocal checks_passed, checks_failed, checks_warned
        symbol = {"pass": "[OK]", "fail": "[FAIL]", "warn": "[WARN]"}[status]
        if status == "pass":
            checks_passed += 1
        elif status == "fail":
            checks_failed += 1
        else:
            checks_warned += 1
        line = f"  {symbol} {name}"
        if detail:
            line += f": {detail}"
        print(line)

    # 1. Python版本检查
    print("\nPython 环境:")
    py_version = sys.version.split()[0]
    version_tuple = sys.version_info
    if version_tuple >= (3, 8):
        check("Python 版本", "pass", f"{py_version} (要求 >= 3.8)")
    else:
        check("Python 版本", "fail", f"{py_version} (要求 >= 3.8)")

    # 2. 核心依赖检查
    print("\n核心依赖:")
    core_deps = [("numpy", "numpy"), ("yaml", "pyyaml"), ("psutil", "psutil")]
    for import_name, pip_name in core_deps:
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "unknown")
            check(import_name, "pass", f"v{version}")
        except ImportError:
            check(import_name, "fail", f"未安装 (pip install {pip_name})")

    # 3. 可选依赖检查
    print("\n可选依赖:")
    optional_deps = [("msgpack", "msgpack"), ("sympy", "sympy")]
    for import_name, pip_name in optional_deps:
        try:
            __import__(import_name)
            check(import_name, "pass")
        except ImportError:
            check(import_name, "warn", f"未安装 (pip install {pip_name})，部分功能降级")

    # 4. 核心模块导入检查
    print("\n核心模块导入:")
    core_modules = [
        ("agent_whitepaper", "顶层包"),
        ("agent_whitepaper.stage_a", "单Agent闭环"),
        ("agent_whitepaper.stage_g", "全系统集成"),
        ("agent_whitepaper.stage_h", "配置化与CLI"),
    ]
    for module_name, desc in core_modules:
        try:
            __import__(module_name)
            check(desc, "pass", module_name)
        except ImportError as e:
            check(desc, "fail", f"{module_name}: {e}")

    # 5. 公共API可用性检查
    print("\n公共API:")
    try:
        from agent_whitepaper import GridWorld, SelfDrivenAgent, __version__
        check("版本号", "pass", __version__)
        check("GridWorld", "pass")
        check("SelfDrivenAgent", "pass")
    except ImportError as e:
        check("公共API", "fail", str(e))

    # 6. CLI子命令检查
    print("\nCLI 子命令:")
    parser = build_parser()
    subcommands = ["init", "run", "demo", "status", "doctor", "wizard", "report"]
    for cmd in subcommands:
        # 检查子命令是否注册
        try:
            # 通过解析 --help 验证子命令存在
            sub_parser = parser._subparsers._group_actions[0].choices.get(cmd)
            if sub_parser is not None:
                check(f"agentwp {cmd}", "pass")
            else:
                check(f"agentwp {cmd}", "fail", "未注册")
        except Exception:  # 防火墙层：parser内部结构访问可能失败
            check(f"agentwp {cmd}", "fail", "检查失败")

    # 7. 功能性检查
    print("\n功能性验证:")
    try:
        env = GridWorld(size=3, seed=42)
        obs = env.reset()
        assert obs.shape == (18,), f"观测维度错误: {obs.shape}"
        check("GridWorld 运行", "pass")
    except Exception as e:  # 防火墙层：功能性验证可能因环境问题失败
        check("GridWorld 运行", "fail", str(e))

    try:
        agent = SelfDrivenAgent(
            action_space=env.action_space, n_states=9, seed=42
        )
        agent.run_episode(env, max_steps=10)
        check("SelfDrivenAgent 运行", "pass")
    except Exception as e:  # 防火墙层：功能性验证可能因环境问题失败
        check("SelfDrivenAgent 运行", "fail", str(e))

    # 8. 总结
    print("\n" + "=" * 60)
    total = checks_passed + checks_failed + checks_warned
    print(f"  诊断完成: {total} 项检查")
    print(f"  通过: {checks_passed}  失败: {checks_failed}  警告: {checks_warned}")

    if checks_failed > 0:
        print("\n  请修复失败的检查项后重试。")
        print("  参考: TROUBLESHOOTING.md")
        return 1
    elif checks_warned > 0:
        print("\n  警告项不影响核心功能，可选择性修复。")
        return 0
    else:
        print("\n  所有检查通过，环境正常！")
        return 0


def cmd_wizard(args: argparse.Namespace) -> int:
    """交互式配置向导。

    第一性原理：
        "Guide the user"——对话式引导优于记忆参数。
        向导通过问答引导用户完成配置，降低认知负荷。
    """
    output_path = args.output
    # 支持测试注入 input/output 函数
    input_func = getattr(args, "_input_func", None) or input
    output_func = getattr(args, "_output_func", None) or print
    wizard = ConfigWizard(input_func=input_func, output_func=output_func)
    try:
        config = wizard.run()
        wizard.save(config, output_path)
    except (KeyboardInterrupt, EOFError):
        output_func("\n  向导已取消")
        return 1
    except ConfigError as e:
        output_func(f"配置错误: {e}")
        return 1
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """生成可视化HTML报告。

    第一性原理：
        "A picture is worth a thousand words"——图表胜过数字。
        将运行结果转换为自包含的HTML可视化报告。
    """
    output_path = args.output
    n_iterations = args.iterations

    # 创建并运行系统以获取指标
    try:
        config = load_config(
            n_agents=args.agents,
            grid_size=args.grid,
            db_path=":memory:",
            max_iterations=n_iterations,
            seed=42,
        )
        orch = SystemOrchestrator(config)
        orch.initialize()
    except AgentWhitepaperError as e:
        print(format_error_message(e), file=sys.stderr)
        return 1

    print(f"  运行 {n_iterations} 轮迭代以采集数据...")
    for _ in range(n_iterations):
        orch.run_iteration()

    # 生成报告
    print("  生成可视化报告...")
    try:
        generate_report_from_orchestrator(orch, output_path)
    except Exception as e:  # 防火墙层：报告生成失败不影响系统
        print(f"报告生成失败: {e}", file=sys.stderr)
        orch.stop("report")
        return 1

    orch.stop("report")
    print(f"  报告已生成: {output_path}")
    print(f"  用浏览器打开查看: file://{os.path.abspath(output_path)}")
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
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"agentwp {__version__}",
        help="显示版本号并退出",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="详细输出（DEBUG级别日志）",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        default=False,
        help="静默模式（仅输出错误）",
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # init
    p_init = subparsers.add_parser("init", help="生成配置文件（支持预置模板）")
    p_init.add_argument(
        "-o", "--output", default="config.yaml",
        help="输出配置文件路径（默认: config.yaml）"
    )
    p_init.add_argument(
        "-t", "--template", default=None,
        choices=list_templates(),
        help="使用预置模板（可选: " + ", ".join(list_templates()) + "）"
    )
    p_init.add_argument(
        "--list-templates", action="store_true",
        help="列出所有可用模板并退出"
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
    p_demo.add_argument(
        "-n", "--iterations", type=int, default=30,
        help="演示迭代次数（默认: 30）",
    )
    p_demo.add_argument(
        "-a", "--agents", type=int, default=3,
        help="Agent数量（默认: 3）",
    )
    p_demo.add_argument(
        "-g", "--grid", type=int, default=5,
        help="网格大小（默认: 5）",
    )
    p_demo.set_defaults(func=cmd_demo)

    # status
    p_status = subparsers.add_parser("status", help="查询运行状态")
    p_status.add_argument(
        "-d", "--db", default="agent_system.db",
        help="数据库路径（默认: agent_system.db）"
    )
    p_status.set_defaults(func=cmd_status)

    # doctor
    p_doctor = subparsers.add_parser(
        "doctor", help="诊断环境（检查依赖、导入、配置等）"
    )
    p_doctor.set_defaults(func=cmd_doctor)

    # wizard
    p_wizard = subparsers.add_parser(
        "wizard", help="交互式配置向导（问答式生成配置）"
    )
    p_wizard.add_argument(
        "-o", "--output", default="config.yaml",
        help="输出配置文件路径（默认: config.yaml）"
    )
    p_wizard.set_defaults(func=cmd_wizard)

    # report
    p_report = subparsers.add_parser(
        "report", help="生成可视化HTML报告"
    )
    p_report.add_argument(
        "-o", "--output", default="report.html",
        help="输出HTML文件路径（默认: report.html）"
    )
    p_report.add_argument(
        "-n", "--iterations", type=int, default=20,
        help="运行迭代次数以采集数据（默认: 20）"
    )
    p_report.add_argument(
        "-a", "--agents", type=int, default=3,
        help="Agent数量（默认: 3）"
    )
    p_report.add_argument(
        "-g", "--grid", type=int, default=5,
        help="网格大小（默认: 5）"
    )
    p_report.set_defaults(func=cmd_report)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI主入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    # 根据全局标志配置日志级别
    if getattr(args, "verbose", False):
        configure_logging(level=logging.DEBUG)
    elif getattr(args, "quiet", False):
        configure_logging(level=logging.ERROR)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
