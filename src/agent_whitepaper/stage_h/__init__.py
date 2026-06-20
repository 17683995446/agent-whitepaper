"""阶段H：配置化与CLI入口包。

公共API：
    ConfigValidationError: 配置校验错误
    ConfigLoader: 配置加载器
    load_config: 加载配置（便捷函数）
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
    ConfigLoader,
    ConfigValidationError,
    load_config,
)

__all__ = [
    "ConfigValidationError",
    "ConfigLoader",
    "load_config",
    "build_parser",
    "main",
    "cmd_init",
    "cmd_run",
    "cmd_demo",
    "cmd_status",
]
