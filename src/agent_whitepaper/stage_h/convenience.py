"""阶段H-4：便捷API - 系统的"快捷方式"。

第一性原理：
    "Make the right thing easy"——常见任务应是一行代码。
    用户最常见的任务：
    1. 快速启动系统并获取结果
    2. 训练单个Agent
    3. 运行演示
    4. 从模板创建系统
    5. 查询运行状态

    便捷API将多步操作封装为一行调用，降低入门门槛。
    每个函数都是"正确做法"的快捷方式——使用合理默认值，
    处理资源清理，返回有用结果。

用法示例：
    from agent_whitepaper import quick_start, train_agent, run_demo

    # 一行启动系统
    report = quick_start(iterations=10)

    # 一行训练Agent
    agent = train_agent(n_episodes=100)

    # 一行运行演示
    report = run_demo(n_iterations=30)
"""

import sqlite3
from typing import Any, Dict

from agent_whitepaper._logging import get_logger
from agent_whitepaper.stage_a import GridWorld, SelfDrivenAgent
from agent_whitepaper.stage_g.orchestrator import SystemOrchestrator

from .config import (
    CONFIG_TEMPLATES,
    get_template,
    load_config,
)

logger = get_logger(__name__)


# ============================================================
# 快速启动
# ============================================================
def quick_start(
    iterations: int = 10,
    n_agents: int = 3,
    grid_size: int = 5,
    db_path: str = ":memory:",
    seed: int = 42,
    **kwargs: Any,
) -> Dict[str, Any]:
    """一行启动永续自主进化系统并获取结果。

    创建系统、初始化、运行指定迭代次数、返回指标报告。
    使用内存数据库（默认），运行结束后自动清理资源。

    Args:
        iterations: 运行迭代次数
        n_agents: Agent数量
        grid_size: 网格大小
        db_path: 数据库路径（默认:memory:不持久化）
        seed: 随机种子
        **kwargs: 其他配置参数（如 skill_abstract_interval 等）

    Returns:
        指标报告字典，包含 iteration, status, latest_snapshot 等

    Example:
        >>> from agent_whitepaper import quick_start
        >>> report = quick_start(iterations=10)
        >>> print(report["latest_snapshot"]["avg_success_rate"])
    """
    config = load_config(
        n_agents=n_agents,
        grid_size=grid_size,
        db_path=db_path,
        max_iterations=iterations,
        seed=seed,
        **kwargs,
    )
    orch = SystemOrchestrator(config)
    try:
        orch.initialize()
        for _ in range(iterations):
            orch.run_iteration()
        report = orch.get_metrics()
        return report
    finally:
        orch.stop("quick_start")


# ============================================================
# 训练单个Agent
# ============================================================
def train_agent(
    n_episodes: int = 100,
    grid_size: int = 5,
    max_steps: int = 100,
    learning_rate: float = 0.1,
    gamma: float = 0.99,
    seed: int = 42,
) -> SelfDrivenAgent:
    """一行训练单个自驱Agent。

    创建GridWorld环境和SelfDrivenAgent，训练指定情节数，
    返回训练好的Agent。Agent可通过 agent.stats 查看训练统计。

    Args:
        n_episodes: 训练情节数
        grid_size: 网格大小（默认5x5）
        max_steps: 每个情节最大步数
        learning_rate: Q学习率
        gamma: 折扣因子
        seed: 随机种子

    Returns:
        训练好的 SelfDrivenAgent 实例

    Example:
        >>> from agent_whitepaper import train_agent
        >>> agent = train_agent(n_episodes=100)
        >>> print(f"成功率: {agent.stats.success_rate:.2%}")
        >>> print(f"Q表: {agent.q_table.n_states}x{agent.q_table.n_actions}")
    """
    env = GridWorld(size=grid_size, seed=seed)
    agent = SelfDrivenAgent(
        action_space=env.action_space,
        n_states=grid_size * grid_size,
        learning_rate=learning_rate,
        gamma=gamma,
        seed=seed,
    )

    for _ in range(n_episodes):
        agent.run_episode(env, max_steps=max_steps)

    logger.debug(
        "训练完成: %d情节, 成功率=%.2f%%",
        n_episodes, agent.stats.success_rate * 100,
    )
    return agent


# ============================================================
# 运行演示
# ============================================================
def run_demo(
    n_iterations: int = 30,
    n_agents: int = 3,
    grid_size: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """一行运行系统演示。

    快速演示多Agent协作系统的能力，使用内存数据库，
    运行结束后自动清理资源。

    Args:
        n_iterations: 演示迭代次数
        n_agents: Agent数量
        grid_size: 网格大小
        seed: 随机种子

    Returns:
        指标报告字典

    Example:
        >>> from agent_whitepaper import run_demo
        >>> report = run_demo(n_iterations=30)
        >>> print(f"完成任务数: {report['tasks_completed']}")
    """
    return quick_start(
        iterations=n_iterations,
        n_agents=n_agents,
        grid_size=grid_size,
        db_path=":memory:",
        seed=seed,
        skill_abstract_interval=3,
        meta_learn_interval=5,
        coalition_interval=8,
        distill_interval=10,
        evolve_interval=7,
    )


# ============================================================
# 从模板创建系统
# ============================================================
def create_system(
    template: str = "default",
    **overrides: Any,
) -> SystemOrchestrator:
    """从预置模板创建系统。

    使用配置模板创建并初始化 SystemOrchestrator。
    调用者负责运行迭代和停止系统。

    Args:
        template: 模板名称（default/quick-test/production/multi-agent/long-run）
        **overrides: 覆盖模板中的配置参数

    Returns:
        已初始化的 SystemOrchestrator 实例

    Raises:
        ConfigValidationError: 未知模板

    Example:
        >>> from agent_whitepaper import create_system
        >>> orch = create_system(template="quick-test")
        >>> for _ in range(10):
        ...     orch.run_iteration()
        >>> report = orch.get_metrics()
        >>> orch.stop("done")
    """
    template_config = get_template(template)
    template_config.update(
        {k: v for k, v in overrides.items() if v is not None}
    )
    config = load_config(**template_config)
    orch = SystemOrchestrator(config)
    orch.initialize()
    return orch


# ============================================================
# 查询运行状态
# ============================================================
def get_status(db_path: str = "agent_system.db") -> Dict[str, Any]:
    """查询持久化数据库中的运行状态。

    从SQLite数据库读取最近的检查点和日志，返回状态摘要。
    数据库不存在时返回空状态。

    Args:
        db_path: 数据库路径

    Returns:
        状态字典，包含 checkpoints, logs, db_path, exists 字段

    Example:
        >>> from agent_whitepaper import get_status
        >>> status = get_status("agent_system.db")
        >>> print(f"检查点数: {len(status['checkpoints'])}")
    """
    result: Dict[str, Any] = {
        "db_path": db_path,
        "exists": False,
        "checkpoints": [],
        "logs": [],
    }

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 检查点
        try:
            cursor.execute(
                "SELECT created_at, description FROM checkpoints "
                "ORDER BY created_at DESC LIMIT 10"
            )
            result["checkpoints"] = [
                {"timestamp": ts, "description": desc}
                for ts, desc in cursor.fetchall()
            ]
            result["exists"] = True
        except sqlite3.Error:
            pass  # 表不存在，忽略

        # 运行日志
        try:
            cursor.execute(
                "SELECT timestamp, event_type, message FROM run_log "
                "ORDER BY timestamp DESC LIMIT 10"
            )
            result["logs"] = [
                {"timestamp": ts, "event_type": etype, "message": msg}
                for ts, etype, msg in cursor.fetchall()
            ]
        except sqlite3.Error:
            pass  # 表不存在，忽略

        conn.close()
    except sqlite3.Error as e:
        logger.debug("读取数据库失败: %s", e)

    return result


# ============================================================
# 列出可用模板（便捷重导出）
# ============================================================
def list_available_templates() -> Dict[str, str]:
    """列出所有可用配置模板及其描述。

    Returns:
        模板名到描述的映射字典

    Example:
        >>> from agent_whitepaper import list_available_templates
        >>> templates = list_available_templates()
        >>> for name, desc in templates.items():
        ...     print(f"  {name}: {desc}")
    """
    return {
        name: tpl.get("_description", "")
        for name, tpl in CONFIG_TEMPLATES.items()
    }
