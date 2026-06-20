"""阶段F：自主进化闭环包。

公共API：
    TaskDifficulty: 任务难度枚举
    AgentCapabilityAssessment: Agent能力评估
    AutonomousTaskGenerator: 自主任务生成器（最近发展区）
    DegradationType: 性能退化类型枚举
    DegradationReport: 退化报告
    DegradationDetector: 退化检测器
    RecoveryStrategy: 恢复策略枚举
    RecoveryResult: 恢复结果
    CheckpointManager: 检查点管理器
    AutonomousRetrainer: 自主重训练器
    AutonomousEvolutionController: 自主进化控制器
"""

from .autonomous_generator import (
    AgentCapabilityAssessment,
    AutonomousTaskGenerator,
    TaskDifficulty,
)
from .autonomous_retrainer import (
    AutonomousEvolutionController,
    AutonomousRetrainer,
    CheckpointManager,
    RecoveryResult,
    RecoveryStrategy,
)
from .degradation_detector import (
    DegradationDetector,
    DegradationReport,
    DegradationType,
)

__all__ = [
    "TaskDifficulty",
    "AgentCapabilityAssessment",
    "AutonomousTaskGenerator",
    "DegradationType",
    "DegradationReport",
    "DegradationDetector",
    "RecoveryStrategy",
    "RecoveryResult",
    "CheckpointManager",
    "AutonomousRetrainer",
    "AutonomousEvolutionController",
]
