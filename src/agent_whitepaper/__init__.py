"""Agent Whitepaper - 自驱Agent自主进化系统。

11个阶段（A-K）构成完整的自驱Agent自主进化闭环：
    A: 单Agent闭环（感知-驱力-策略-行动-学习）
    B: 技能增强与长任务（n-gram抽象、元学习、任务分解）
    C: 多Agent协作（注册中心、消息总线、联盟、共享技能）
    D: 永续运行时（SQLite持久化、调度器、健康监控、daemon）
    E: 技能迁移（Q表序列化、技能Q值注入、知识蒸馏）
    F: 自主进化（任务生成、退化检测、自主重训练）
    G: 全系统集成（指标采集、A-F闭环编排）
    H: 配置化与CLI（YAML配置、命令行入口）
    I: 真实环境适配（EnvironmentProtocol、状态编码器）
    J: 安全约束（SafetyMonitor、资源管理、背压）
    K: 长时运行实证（演示、报告生成）

顶层公共API（各阶段核心入口）：
    SelfDrivenAgent, GridWorld, QTable  -- 阶段A
    Skill, SkillLibrary, MetaLearner, TaskDecomposer  -- 阶段B
    AgentRegistry, MessageBus, CoalitionFormer  -- 阶段C
    PerpetualRuntime, PersistenceEngine, TaskScheduler, HealthMonitor  -- 阶段D
    QTableMigrator, SkillQValueInjector, KnowledgeDistiller  -- 阶段E
    AutonomousTaskGenerator, DegradationDetector, AutonomousRetrainer  -- 阶段F
    SystemOrchestrator, MetricsCollector  -- 阶段G
    load_config, main  -- 阶段H
    EnvironmentProtocol, FunctionCallEnvironment  -- 阶段I
    SafetyMonitor, ResourceManager  -- 阶段J
    LongRunDemo, ReportGenerator  -- 阶段K
"""

__version__ = "0.1.0"

from ._exceptions import (
    AgentWhitepaperError,
    CheckpointError,
    CLIError,
    CoalitionError,
    ComponentError,
    ConfigError,
    DistillationError,
    EvolutionError,
    MetaLearningError,
    MetricsCollectionError,
    PersistenceError,
    RecoveryError,
    RuntimeExecutionError,
    SkillAbstractionError,
    StateRecoveryError,
    StateSerializationError,
    TransactionError,
)
from ._logging import configure_logging, get_logger
from ._protocols import AgentProtocol, MemoryProtocol, PolicyProtocol
from .stage_a import GridWorld, QTable, SelfDrivenAgent
from .stage_b import MetaLearner, Skill, SkillLibrary, TaskDecomposer
from .stage_c import AgentRegistry, CoalitionFormer, MessageBus
from .stage_d import (
    HealthMonitor,
    PerpetualRuntime,
    PersistenceEngine,
    TaskScheduler,
)
from .stage_e import (
    KnowledgeDistiller,
    QTableMigrator,
    SkillQValueInjector,
)
from .stage_f import (
    AutonomousRetrainer,
    AutonomousTaskGenerator,
    DegradationDetector,
)
from .stage_g import MetricsCollector, SystemOrchestrator
from .stage_h import (
    create_system,
    get_status,
    list_available_templates,
    load_config,
    main,
    quick_start,
    run_demo,
    train_agent,
)
from .stage_i import EnvironmentProtocol, FunctionCallEnvironment
from .stage_j import ResourceManager, SafetyMonitor
from .stage_k import LongRunDemo, ReportGenerator

__all__ = [
    # 阶段A
    "SelfDrivenAgent",
    "GridWorld",
    "QTable",
    # 阶段B
    "Skill",
    "SkillLibrary",
    "MetaLearner",
    "TaskDecomposer",
    # 阶段C
    "AgentRegistry",
    "MessageBus",
    "CoalitionFormer",
    # 阶段D
    "PerpetualRuntime",
    "PersistenceEngine",
    "TaskScheduler",
    "HealthMonitor",
    # 阶段E
    "QTableMigrator",
    "SkillQValueInjector",
    "KnowledgeDistiller",
    # 阶段F
    "AutonomousTaskGenerator",
    "DegradationDetector",
    "AutonomousRetrainer",
    # 阶段G
    "SystemOrchestrator",
    "MetricsCollector",
    # 阶段H
    "load_config",
    "main",
    "quick_start",
    "train_agent",
    "run_demo",
    "create_system",
    "get_status",
    "list_available_templates",
    # 阶段I
    "EnvironmentProtocol",
    "FunctionCallEnvironment",
    # 阶段J
    "SafetyMonitor",
    "ResourceManager",
    # 阶段K
    "LongRunDemo",
    "ReportGenerator",
    # 异常层次体系（阶段R）
    "AgentWhitepaperError",
    "ConfigError",
    "PersistenceError",
    "TransactionError",
    "StateSerializationError",
    "RuntimeExecutionError",
    "StateRecoveryError",
    "ComponentError",
    "SkillAbstractionError",
    "MetaLearningError",
    "CoalitionError",
    "DistillationError",
    "EvolutionError",
    "MetricsCollectionError",
    "CheckpointError",
    "RecoveryError",
    "CLIError",
    # 抽象接口（阶段S）
    "PolicyProtocol",
    "MemoryProtocol",
    "AgentProtocol",
    # 日志配置（阶段T）
    "get_logger",
    "configure_logging",
]
