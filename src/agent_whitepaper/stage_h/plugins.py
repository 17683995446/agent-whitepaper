"""阶段H-8：插件系统 - 系统的"扩展点"。

第一性原理：
    "Extensibility"——允许用户扩展而非修改核心。
    一个成熟系统应提供扩展点，让用户在不修改核心代码的情况下
    注入自定义逻辑。插件系统通过：
    1. 协议：定义清晰的插件接口（生命周期钩子）
    2. 隔离：插件失败不影响核心系统（防火墙层）
    3. 组合：多个插件可协同工作（事件广播）
    4. 内置：提供常用插件开箱即用

    插件让系统从"封闭"变为"开放"。

用法：
    from agent_whitepaper.stage_h.plugins import (
        Plugin, PluginManager, LoggingPlugin,
    )

    # 创建管理器并注册插件
    mgr = PluginManager()
    mgr.register(LoggingPlugin())

    # 在生命周期点触发事件
    mgr.emit("on_init", orchestrator=orch)
    mgr.emit("on_iteration", iteration=1, snapshot=snap)
    mgr.emit("on_stop", reason="completed")
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_whitepaper._logging import get_logger

logger = get_logger(__name__)


# ============================================================
# 插件上下文
# ============================================================
@dataclass
class PluginContext:
    """插件事件上下文。

    传递给插件钩子的上下文对象，包含事件相关数据。
    不同事件填充不同字段，未使用字段为 None。

    Attributes:
        event: 事件名（on_init/on_iteration/on_metrics/on_error/on_stop）
        orchestrator: 系统编排器实例（on_init/on_metrics 时可用）
        iteration: 当前迭代序号（on_iteration 时可用）
        snapshot: 当前指标快照（on_iteration/on_metrics 时可用）
        error: 异常对象（on_error 时可用）
        reason: 停止原因（on_stop 时可用）
        report: 指标报告（on_metrics 时可用）
        extra: 额外数据（插件自定义）
    """

    event: str = ""
    orchestrator: Any = None
    iteration: Optional[int] = None
    snapshot: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None
    reason: Optional[str] = None
    report: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 插件基类
# ============================================================
class Plugin:
    """插件基类。

    第一性原理：
        "Extensibility"——允许用户扩展而非修改核心。
        子类通过覆盖钩子方法注入自定义逻辑。
        所有钩子默认空实现，子类按需覆盖。

    Attributes:
        name: 插件名（唯一标识）
        version: 插件版本
        description: 插件描述
    """

    name: str = "base"
    version: str = "0.1.0"
    description: str = "插件基类"

    def on_init(self, ctx: PluginContext) -> None:
        """系统初始化时调用。"""

    def on_iteration(self, ctx: PluginContext) -> None:
        """每轮迭代后调用。"""

    def on_metrics(self, ctx: PluginContext) -> None:
        """指标报告生成时调用。"""

    def on_error(self, ctx: PluginContext) -> None:
        """系统出错时调用。"""

    def on_stop(self, ctx: PluginContext) -> None:
        """系统停止时调用。"""

    def __repr__(self) -> str:
        return f"<Plugin {self.name} v{self.version}>"


# ============================================================
# 插件管理器
# ============================================================
class PluginManager:
    """插件管理器。

    第一性原理：
        "Extensibility"——允许用户扩展而非修改核心。
        管理插件的注册、启停和事件分发。
        插件失败被防火墙层捕获，不影响核心系统。

    用法：
        mgr = PluginManager()
        mgr.register(LoggingPlugin())
        mgr.emit("on_init", orchestrator=orch)
    """

    def __init__(self) -> None:
        self._plugins: Dict[str, Plugin] = {}
        self._enabled: Dict[str, bool] = {}
        self._event_counts: Dict[str, int] = {}

    def register(self, plugin: Plugin) -> None:
        """注册插件。

        Args:
            plugin: 插件实例
        """
        if not plugin.name:
            raise ValueError("插件名不能为空")
        if plugin.name in self._plugins:
            raise ValueError(f"插件已存在: {plugin.name}")
        self._plugins[plugin.name] = plugin
        self._enabled[plugin.name] = True
        logger.debug("已注册插件: %s", plugin.name)

    def unregister(self, name: str) -> bool:
        """注销插件。

        Args:
            name: 插件名

        Returns:
            是否成功注销
        """
        if name not in self._plugins:
            return False
        del self._plugins[name]
        del self._enabled[name]
        logger.debug("已注销插件: %s", name)
        return True

    def get(self, name: str) -> Optional[Plugin]:
        """获取插件实例。

        Args:
            name: 插件名

        Returns:
            插件实例或 None
        """
        return self._plugins.get(name)

    def list_plugins(self) -> List[Plugin]:
        """列出所有已注册插件。

        Returns:
            插件实例列表
        """
        return list(self._plugins.values())

    def list_enabled(self) -> List[Plugin]:
        """列出所有已启用插件。

        Returns:
            已启用插件实例列表
        """
        return [
            p for name, p in self._plugins.items()
            if self._enabled.get(name, False)
        ]

    def is_enabled(self, name: str) -> bool:
        """检查插件是否启用。

        Args:
            name: 插件名

        Returns:
            是否启用
        """
        return self._enabled.get(name, False)

    def enable(self, name: str) -> bool:
        """启用插件。

        Args:
            name: 插件名

        Returns:
            是否成功启用（插件存在时）
        """
        if name not in self._plugins:
            return False
        self._enabled[name] = True
        return True

    def disable(self, name: str) -> bool:
        """禁用插件。

        Args:
            name: 插件名

        Returns:
            是否成功禁用（插件存在时）
        """
        if name not in self._plugins:
            return False
        self._enabled[name] = False
        return True

    def emit(self, event: str, **kwargs: Any) -> None:
        """触发事件，调用所有已启用插件对应钩子。

        插件钩子失败被防火墙层捕获，不影响其他插件和核心系统。

        Args:
            event: 事件名（on_init/on_iteration/on_metrics/on_error/on_stop）
            **kwargs: 上下文数据
        """
        self._event_counts[event] = self._event_counts.get(event, 0) + 1
        ctx = PluginContext(event=event, **kwargs)
        for plugin in self.list_enabled():
            hook = getattr(plugin, event, None)
            if hook is None or not callable(hook):
                continue
            try:
                hook(ctx)
            except Exception:  # 防火墙层：插件失败不影响核心
                logger.debug("插件 %s 的 %s 钩子失败", plugin.name, event)

    @property
    def event_counts(self) -> Dict[str, int]:
        """各事件触发次数。"""
        return dict(self._event_counts)

    def clear(self) -> None:
        """清空所有插件和统计。"""
        self._plugins.clear()
        self._enabled.clear()
        self._event_counts.clear()

    def __len__(self) -> int:
        return len(self._plugins)

    def __contains__(self, name: str) -> bool:
        return name in self._plugins


# ============================================================
# 内置插件
# ============================================================
class LoggingPlugin(Plugin):
    """日志插件。

    第一性原理：
        "Extensibility"——允许用户扩展而非修改核心。
        将生命周期事件记录到日志，便于调试和审计。

    用法：
        mgr.register(LoggingPlugin(level="info"))
    """

    name = "logging"
    version = "1.0.0"
    description = "记录生命周期事件到日志"

    def __init__(self, level: str = "debug") -> None:
        self._level = level.lower()
        self._log_func = {
            "debug": logger.debug,
            "info": logger.info,
            "warning": logger.warning,
            "error": logger.error,
        }.get(self._level, logger.debug)

    def on_init(self, ctx: PluginContext) -> None:
        self._log_func("[插件] 系统初始化")

    def on_iteration(self, ctx: PluginContext) -> None:
        self._log_func("[插件] 迭代 #%d", ctx.iteration or 0)

    def on_metrics(self, ctx: PluginContext) -> None:
        self._log_func("[插件] 指标报告生成")

    def on_error(self, ctx: PluginContext) -> None:
        self._log_func("[插件] 系统错误: %s", ctx.error)

    def on_stop(self, ctx: PluginContext) -> None:
        self._log_func("[插件] 系统停止: %s", ctx.reason)


class MetricsExportPlugin(Plugin):
    """指标导出插件。

    第一性原理：
        "Extensibility"——允许用户扩展而非修改核心。
        将每轮迭代的指标快照导出为JSON文件，便于外部分析。

    用法：
        mgr.register(MetricsExportPlugin(path="metrics.jsonl"))
    """

    name = "metrics-export"
    version = "1.0.0"
    description = "将指标快照导出为JSONL文件"

    def __init__(self, path: str = "metrics.jsonl") -> None:
        self._path = path
        self._count = 0

    @property
    def export_path(self) -> str:
        return self._path

    @property
    def export_count(self) -> int:
        return self._count

    def on_iteration(self, ctx: PluginContext) -> None:
        """每轮迭代导出快照。"""
        import json
        if ctx.snapshot is None:
            return
        record = {
            "iteration": ctx.iteration,
            "snapshot": ctx.snapshot,
        }
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._count += 1
        except Exception:  # 防火墙层：文件写入失败不中断系统
            logger.debug("指标导出失败")

    def on_stop(self, ctx: PluginContext) -> None:
        """停止时记录总数。"""
        logger.debug("指标导出完成，共 %d 条", self._count)


# ============================================================
# 内置插件注册表
# ============================================================
BUILTIN_PLUGINS: Dict[str, type] = {
    "logging": LoggingPlugin,
    "metrics-export": MetricsExportPlugin,
}


def list_builtin_plugins() -> Dict[str, str]:
    """列出所有内置插件及描述。

    Returns:
        插件名到描述的映射
    """
    return {
        name: cls.description
        for name, cls in BUILTIN_PLUGINS.items()
    }


def create_plugin_manager(
    enabled: Optional[List[str]] = None,
) -> PluginManager:
    """创建插件管理器并注册指定内置插件。

    Args:
        enabled: 要启用的内置插件名列表（None=全部）

    Returns:
        配置好的 PluginManager 实例

    Raises:
        ValueError: 指定的插件名不存在
    """
    mgr = PluginManager()
    names = enabled if enabled is not None else list(BUILTIN_PLUGINS.keys())
    for name in names:
        if name not in BUILTIN_PLUGINS:
            raise ValueError(f"未知插件: {name}")
        mgr.register(BUILTIN_PLUGINS[name]())
    return mgr
