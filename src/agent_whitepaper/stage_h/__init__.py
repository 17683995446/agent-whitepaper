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
from .progress import (
    ProgressReporter,
    SilentReporter,
    format_duration,
    predict_phases,
    render_progress_bar,
)

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
    "build_parser",
    "main",
    "cmd_init",
    "cmd_run",
    "cmd_demo",
    "cmd_status",
]
