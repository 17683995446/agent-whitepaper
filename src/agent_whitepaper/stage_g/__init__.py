"""阶段G：全系统集成与可观测性包。

公共API：
    MetricsSnapshot: 指标快照
    MetricsCollector: 指标采集器
    OrchestratorConfig: 编排器配置
    OrchestratorStatus: 编排器状态枚举
    SystemOrchestrator: 全系统编排器（A-F闭环编排）
"""

from .metrics import MetricsCollector, MetricsSnapshot
from .orchestrator import (
    OrchestratorConfig,
    OrchestratorStatus,
    SystemOrchestrator,
)

__all__ = [
    "MetricsSnapshot",
    "MetricsCollector",
    "OrchestratorConfig",
    "OrchestratorStatus",
    "SystemOrchestrator",
]
