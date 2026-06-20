"""统一日志配置 - 可观测性=可调试性。

第一性原理：
    可观测性是可调试性的基础。一个不可观测的系统无法被诊断：
    1. 静默失败：错误被吞掉，无法定位问题
    2. 无结构：print 散落各处，无法过滤级别
    3. 无上下文：不知道错误来自哪个组件

    统一日志框架让系统：
    1. 可诊断：所有降级路径都有日志记录
    2. 可过滤：按级别（DEBUG/INFO/WARNING/ERROR）过滤
    3. 可结构化：统一格式，便于工具处理

    设计原则：
    - 用户面输出（CLI/demos）保留 print（面向用户的输出）
    - 系统内部事件使用 logging（面向开发者的诊断）
    - 降级路径（except Exception: pass）记录 DEBUG（预期失败，仅调试时可见）
    - 关键状态转换记录 INFO（启动/停止/恢复）
    - 不可恢复错误记录 ERROR

    使用方式：
        from agent_whitepaper._logging import get_logger
        logger = get_logger(__name__)
        logger.debug("技能抽象失败: %s", e, exc_info=True)
"""

import logging
import sys
from typing import Optional

# 系统日志根名
LOGGER_NAME = "agent_whitepaper"

# 默认日志格式
DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """获取统一命名的 logger。

    所有模块应使用此函数获取 logger，确保日志名空间统一。
    传入 __name__ 可获得模块级 logger（如 agent_whitepaper.stage_d.runtime）。
    """
    if not name.startswith(LOGGER_NAME):
        name = f"{LOGGER_NAME}.{name}"
    return logging.getLogger(name)


def configure_logging(
    level: int = logging.WARNING,
    fmt: str = DEFAULT_FORMAT,
    stream: Optional[object] = None,
) -> None:
    """配置统一日志。

    Args:
        level: 日志级别（默认 WARNING，生产环境不输出 DEBUG/INFO）
        fmt: 日志格式字符串
        stream: 输出流（默认 stderr，避免干扰 CLI 的 stdout 输出）
    """
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(fmt))

    root_logger = logging.getLogger(LOGGER_NAME)
    root_logger.setLevel(level)
    # 避免重复添加 handler
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    # 不向上传播（避免 root logger 重复输出）
    root_logger.propagate = False


# 默认配置：WARNING 级别，不干扰正常输出
configure_logging()


__all__ = [
    "LOGGER_NAME",
    "DEFAULT_FORMAT",
    "get_logger",
    "configure_logging",
]
