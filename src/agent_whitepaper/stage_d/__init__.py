"""阶段D：永续运行时包 - 让系统永远运行不停止。

公共API：
    PersistenceEngine: 持久化引擎（SQLite）
    ScheduledTaskStatus: 调度任务状态枚举
    ScheduledTask: 调度任务
    TaskGenerator: 任务生成器
    TaskScheduler: 任务调度器
    HealthStatus: 健康状态枚举
    HealthCheck: 健康检查
    ComponentHealth: 组件健康
    CircuitBreaker: 断路器
    HealthMonitor: 健康监控
    safe_execute: 安全执行（带熔断）
    RuntimeStatus: 运行时状态枚举
    RuntimeConfig: 运行时配置
    RuntimeStats: 运行时统计
    PerpetualRuntime: 永续运行时
"""

from .health_monitor import (
    CircuitBreaker,
    ComponentHealth,
    HealthCheck,
    HealthMonitor,
    HealthStatus,
    safe_execute,
)
from .persistence import PersistenceEngine
from .runtime import (
    PerpetualRuntime,
    RuntimeConfig,
    RuntimeStats,
    RuntimeStatus,
)
from .scheduler import (
    ScheduledTask,
    ScheduledTaskStatus,
    TaskGenerator,
    TaskScheduler,
)

__all__ = [
    "PersistenceEngine",
    "ScheduledTaskStatus",
    "ScheduledTask",
    "TaskGenerator",
    "TaskScheduler",
    "HealthStatus",
    "HealthCheck",
    "ComponentHealth",
    "CircuitBreaker",
    "HealthMonitor",
    "safe_execute",
    "RuntimeStatus",
    "RuntimeConfig",
    "RuntimeStats",
    "PerpetualRuntime",
]
