"""阶段H-9：交互式Shell - 系统的"探索界面"。

第一性原理：
    "REPL is power"——交互式探索优于脚本编写。
    脚本是"一次性"的，而交互式Shell允许用户：
    1. 探索：逐步运行并观察结果
    2. 调试：出错后可检查状态再继续
    3. 学习：通过试错理解系统行为
    4. 即兴：无需预先编写完整脚本

    REPL将"使用系统"变为"对话系统"。

用法：
    from agent_whitepaper.stage_h.shell import InteractiveShell

    shell = InteractiveShell()
    shell.run()  # 进入交互式REPL

    # 非交互模式（用于测试/自动化）
    shell.run_with_commands(["init", "run 5", "status", "exit"])
"""

import shlex
from typing import Callable, Dict, List, Optional

from agent_whitepaper._exceptions import AgentWhitepaperError
from agent_whitepaper._logging import get_logger
from agent_whitepaper.stage_g.orchestrator import SystemOrchestrator

from .config import load_config
from .visualizer import generate_report_from_orchestrator

logger = get_logger(__name__)


# ============================================================
# 交互式Shell
# ============================================================
class InteractiveShell:
    """交互式Shell。

    第一性原理：
        "REPL is power"——交互式探索优于脚本编写。
        提供REPL界面，让用户交互式地初始化、运行、查询系统。
        支持输入/输出函数注入，便于测试。

    用法：
        shell = InteractiveShell()
        shell.run()

        # 非交互模式
        shell.run_with_commands(["init", "run 5", "exit"])
    """

    PROMPT = "agentwp> "

    def __init__(
        self,
        input_func: Optional[Callable[[], str]] = None,
        output_func: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._input = input_func or input
        self._output = output_func or print
        self._orchestrator: Optional[SystemOrchestrator] = None
        self._running = False
        self._commands: Dict[str, Callable[[List[str]], int]] = {
            "help": self._cmd_help,
            "init": self._cmd_init,
            "run": self._cmd_run,
            "status": self._cmd_status,
            "metrics": self._cmd_metrics,
            "agents": self._cmd_agents,
            "report": self._cmd_report,
            "exit": self._cmd_exit,
            "quit": self._cmd_exit,
        }

    @property
    def orchestrator(self) -> Optional[SystemOrchestrator]:
        """当前编排器实例。"""
        return self._orchestrator

    @property
    def is_running(self) -> bool:
        """Shell是否正在运行。"""
        return self._running

    def run(self) -> int:
        """启动交互式REPL。

        Returns:
            退出码（0=正常）
        """
        self._running = True
        self._output("Agent Whitepaper 交互式Shell")
        self._output("输入 'help' 查看可用命令，'exit' 退出")

        while self._running:
            try:
                line = self._input(self.PROMPT)
            except (KeyboardInterrupt, EOFError):
                self._output("\n再见！")
                break

            if not line.strip():
                continue

            ret = self.execute(line)
            if ret == 0 and not self._running:
                break

        self._cleanup()
        return 0

    def run_with_commands(self, commands: List[str]) -> int:
        """非交互模式：执行命令列表后退出。

        Args:
            commands: 命令字符串列表

        Returns:
            退出码
        """
        self._running = True
        for cmd in commands:
            self._output(f"{self.PROMPT}{cmd}")
            ret = self.execute(cmd)
            if ret != 0:
                self._cleanup()
                return ret
            if not self._running:
                break
        self._cleanup()
        return 0

    def execute(self, line: str) -> int:
        """执行单条命令。

        Args:
            line: 命令行字符串

        Returns:
            退出码（0=成功，1=错误）
        """
        try:
            parts = shlex.split(line)
        except ValueError as e:  # 边界层：引号不匹配等语法错误
            self._output(f"  语法错误: {e}")
            return 1

        if not parts:
            return 0

        cmd = parts[0]
        args = parts[1:]

        handler = self._commands.get(cmd)
        if handler is None:
            self._output(f"  未知命令: {cmd}（输入 'help' 查看可用命令）")
            return 1

        try:
            return handler(args)
        except Exception as e:  # 防火墙层：命令执行失败不退出Shell
            self._output(f"  命令执行失败: {e}")
            logger.debug("命令 %s 执行失败: %s", cmd, e)
            return 1

    def _cleanup(self) -> None:
        """清理资源。"""
        if self._orchestrator is not None:
            try:
                self._orchestrator.stop("shell-exit")
            except Exception:  # 防火墙层：清理失败不影响退出
                logger.debug("清理编排器失败")
            self._orchestrator = None
        self._running = False

    # --------------------------------------------------------
    # 命令处理
    # --------------------------------------------------------
    def _cmd_help(self, args: List[str]) -> int:
        """显示帮助。"""
        self._output("可用命令:")
        self._output("  init [agents] [grid]  初始化系统（默认: 3 agents, 5x5 grid）")
        self._output("  run [n]               运行 n 轮迭代（默认: 1）")
        self._output("  status                显示系统状态")
        self._output("  metrics               显示指标报告")
        self._output("  agents                列出所有Agent")
        self._output("  report [path]         生成HTML报告（默认: report.html）")
        self._output("  help                  显示此帮助")
        self._output("  exit / quit           退出Shell")
        return 0

    def _cmd_init(self, args: List[str]) -> int:
        """初始化系统。"""
        if self._orchestrator is not None:
            self._output("  系统已初始化，先退出再重新初始化")
            return 1

        n_agents = int(args[0]) if len(args) >= 1 else 3
        grid_size = int(args[1]) if len(args) >= 2 else 5

        try:
            config = load_config(
                n_agents=n_agents,
                grid_size=grid_size,
                db_path=":memory:",
                max_iterations=None,
                seed=42,
            )
            self._orchestrator = SystemOrchestrator(config)
            self._orchestrator.initialize()
        except (AgentWhitepaperError, ValueError) as e:
            self._output(f"  初始化失败: {e}")
            self._orchestrator = None
            return 1

        self._output(
            f"  系统已初始化: {n_agents} agents, {grid_size}x{grid_size} grid"
        )
        return 0

    def _cmd_run(self, args: List[str]) -> int:
        """运行迭代。"""
        if self._orchestrator is None:
            self._output("  系统未初始化，请先运行 'init'")
            return 1

        n = int(args[0]) if len(args) >= 1 else 1
        if n <= 0:
            self._output("  迭代次数必须 > 0")
            return 1

        for i in range(n):
            self._orchestrator.run_iteration()
        self._output(f"  已运行 {n} 轮迭代")
        return 0

    def _cmd_status(self, args: List[str]) -> int:
        """显示状态。"""
        if self._orchestrator is None:
            self._output("  系统未初始化")
            return 0

        report = self._orchestrator.get_metrics()
        self._output(f"  状态: {report.get('status', 'unknown')}")
        self._output(f"  迭代: {report.get('iteration', 0)}")
        self._output(f"  Agent数: {report.get('agent_count', 0)}")
        self._output(f"  技能数: {report.get('skill_count', 0)}")
        self._output(f"  任务完成: {report.get('tasks_completed', 0)}")
        return 0

    def _cmd_metrics(self, args: List[str]) -> int:
        """显示指标。"""
        if self._orchestrator is None:
            self._output("  系统未初始化")
            return 0

        report = self._orchestrator.get_metrics()
        self._output(f"  迭代次数: {report.get('iteration', 0)}")
        self._output(f"  运行时间: {report.get('uptime', 0):.1f}s")
        snap = report.get("latest_snapshot")
        if snap:
            self._output(f"  平均成功率: {snap.get('avg_success_rate', 0):.4f}")
            self._output(f"  平均奖励: {snap.get('avg_reward', 0):.4f}")
            self._output(f"  Q表条目: {snap.get('total_q_entries', 0)}")
            self._output(f"  健康分: {snap.get('health_score', 0):.4f}")

        cum = report.get("cumulative", {})
        if cum:
            self._output("  累计统计:")
            for k, v in cum.items():
                self._output(f"    {k}: {v}")
        return 0

    def _cmd_agents(self, args: List[str]) -> int:
        """列出Agent。"""
        if self._orchestrator is None:
            self._output("  系统未初始化")
            return 0

        summaries = self._orchestrator.get_agent_summaries()
        if not summaries:
            self._output("  无Agent")
            return 0

        self._output(f"  共 {len(summaries)} 个Agent:")
        for s in summaries:
            self._output(
                f"    {s['agent_id']}: "
                f"成功率={s['success_rate']:.3f} "
                f"情节={s['total_episodes']} "
                f"Q条目={s['q_entries']}"
            )
        return 0

    def _cmd_report(self, args: List[str]) -> int:
        """生成HTML报告。"""
        if self._orchestrator is None:
            self._output("  系统未初始化")
            return 0

        path = args[0] if len(args) >= 1 else "report.html"
        try:
            generate_report_from_orchestrator(self._orchestrator, path)
        except Exception as e:  # 防火墙层：报告生成失败不影响Shell
            self._output(f"  报告生成失败: {e}")
            return 1

        self._output(f"  报告已生成: {path}")
        return 0

    def _cmd_exit(self, args: List[str]) -> int:
        """退出Shell。"""
        self._output("  再见！")
        self._running = False
        return 0


# ============================================================
# 便捷函数
# ============================================================
def run_shell(
    input_func: Optional[Callable[[], str]] = None,
    output_func: Optional[Callable[[str], None]] = None,
) -> int:
    """启动交互式Shell。

    Args:
        input_func: 输入函数（默认: input）
        output_func: 输出函数（默认: print）

    Returns:
        退出码
    """
    shell = InteractiveShell(input_func=input_func, output_func=output_func)
    return shell.run()


def run_shell_with_commands(commands: List[str]) -> int:
    """非交互模式运行Shell命令列表。

    Args:
        commands: 命令字符串列表

    Returns:
        退出码

    Example:
        >>> from agent_whitepaper.stage_h.shell import run_shell_with_commands
        >>> run_shell_with_commands(["init", "run 5", "status", "exit"])
    """
    shell = InteractiveShell()
    return shell.run_with_commands(commands)
