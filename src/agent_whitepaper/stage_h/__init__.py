"""阶段H：配置化与CLI入口包。

公共API：
    ConfigValidationError: 配置校验错误
    ConfigLoader: 配置加载器
    load_config: 加载配置（便捷函数）
    CONFIG_TEMPLATES: 预置配置模板
    list_templates: 列出模板名
    get_template: 获取模板配置
    ProgressReporter: 进度报告器
    SilentReporter: 静默报告器
    predict_phases: 预测迭代阶段
    format_duration: 格式化时长
    render_progress_bar: 渲染进度条
    quick_start: 一行启动系统
    train_agent: 一行训练Agent
    run_demo: 一行运行演示
    create_system: 从模板创建系统
    get_status: 查询运行状态
    list_available_templates: 列出模板及描述
    ConfigWizard: 交互式配置向导
    run_wizard: 运行向导便捷函数
    ErrorRecovery: 智能错误恢复
    Diagnosis: 错误诊断结果
    AutoFix: 自动修复方案
    diagnose_error: 诊断异常
    suggest_fix: 建议修复
    format_error_message: 格式化错误消息
    Visualizer: 可视化报告生成器
    generate_report: 生成可视化报告
    generate_report_from_orchestrator: 从编排器生成可视化报告
    Plugin: 插件基类
    PluginContext: 插件事件上下文
    PluginManager: 插件管理器
    LoggingPlugin: 日志插件
    MetricsExportPlugin: 指标导出插件
    list_builtin_plugins: 列出内置插件
    create_plugin_manager: 创建插件管理器
    InteractiveShell: 交互式Shell
    run_shell: 启动交互式Shell
    run_shell_with_commands: 非交互模式运行Shell命令
    build_parser: 构建CLI参数解析器
    main: CLI主入口
    cmd_init: init子命令
    cmd_run: run子命令
    cmd_demo: demo子命令
    cmd_status: status子命令
"""

from .cli import (
    build_parser,
    cmd_demo,
    cmd_init,
    cmd_run,
    cmd_status,
    main,
)
from .config import (
    CONFIG_TEMPLATES,
    ConfigLoader,
    ConfigValidationError,
    get_template,
    list_templates,
    load_config,
)
from .convenience import (
    create_system,
    get_status,
    list_available_templates,
    quick_start,
    run_demo,
    train_agent,
)
from .plugins import (
    BUILTIN_PLUGINS,
    LoggingPlugin,
    MetricsExportPlugin,
    Plugin,
    PluginContext,
    PluginManager,
    create_plugin_manager,
    list_builtin_plugins,
)
from .progress import (
    ProgressReporter,
    SilentReporter,
    format_duration,
    predict_phases,
    render_progress_bar,
)
from .recovery import (
    AutoFix,
    Diagnosis,
    ErrorRecovery,
    diagnose_error,
    format_error_message,
    suggest_fix,
)
from .shell import InteractiveShell, run_shell, run_shell_with_commands
from .visualizer import Visualizer, generate_report, generate_report_from_orchestrator
from .wizard import ConfigWizard, run_wizard

__all__ = [
    "ConfigValidationError",
    "ConfigLoader",
    "load_config",
    "CONFIG_TEMPLATES",
    "list_templates",
    "get_template",
    "ProgressReporter",
    "SilentReporter",
    "predict_phases",
    "format_duration",
    "render_progress_bar",
    "quick_start",
    "train_agent",
    "run_demo",
    "create_system",
    "get_status",
    "list_available_templates",
    "ConfigWizard",
    "run_wizard",
    "ErrorRecovery",
    "Diagnosis",
    "AutoFix",
    "diagnose_error",
    "suggest_fix",
    "format_error_message",
    "Visualizer",
    "generate_report",
    "generate_report_from_orchestrator",
    "Plugin",
    "PluginContext",
    "PluginManager",
    "LoggingPlugin",
    "MetricsExportPlugin",
    "BUILTIN_PLUGINS",
    "list_builtin_plugins",
    "create_plugin_manager",
    "InteractiveShell",
    "run_shell",
    "run_shell_with_commands",
    "build_parser",
    "main",
    "cmd_init",
    "cmd_run",
    "cmd_demo",
    "cmd_status",
]
