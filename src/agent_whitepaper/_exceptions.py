"""异常层次体系 - 系统可恢复性的基础。

第一性原理：
    错误是系统可恢复性的基础。一个良好的异常层次体系让系统：
    1. 可分类：不同错误类别需要不同恢复策略（瞬时 vs 致命）
    2. 可传播：错误在合适的层级被捕获和处理（组件级 vs 防火墙级）
    3. 可诊断：错误携带足够的上下文信息（来源、原因、可恢复性）
    4. 可恢复：区分可降级的组件错误与不可恢复的系统错误

    层次设计原则：
    - 顶层 AgentWhitepaperError：所有自定义异常的根，便于统一捕获
    - 中层按领域分类：Config / Persistence / Runtime / Component / Recovery / CLI
    - 底层按具体场景：TransactionError / SkillAbstractionError 等

    与标准库的关系：
    - 参数校验错误继续使用 ValueError（Python 惯例，不重复造轮子）
    - 自定义异常覆盖标准库未表达的领域语义
    - ConfigValidationError 兼容 ValueError（多重继承，向后兼容）

    异常捕获规范：
    - 防火墙层（主循环、Agent隔离）：except Exception（永续运行保障）
    - 组件层（技能抽象、元学习等）：except Exception + 降级注释
    - 边界层（CLI、DB）：except 具体标准库异常
"""


# ============================================================
# 根异常
# ============================================================
class AgentWhitepaperError(Exception):
    """所有 agent-whitepaper 自定义异常的基类。

    捕获此异常等价于捕获所有本系统自定义异常，
    用于防火墙层需要区分"系统已知错误"与"未知错误"的场景。
    """


# ============================================================
# 配置域
# ============================================================
class ConfigError(AgentWhitepaperError):
    """配置相关错误（加载、解析、校验）。"""


# ============================================================
# 持久化域
# ============================================================
class PersistenceError(AgentWhitepaperError):
    """持久化存储相关错误（数据库、序列化）。"""


class TransactionError(PersistenceError):
    """数据库事务失败（提交/回滚阶段）。"""


class StateSerializationError(PersistenceError):
    """状态序列化/反序列化失败（保存/恢复阶段）。"""


# ============================================================
# 运行时域
# ============================================================
class RuntimeExecutionError(AgentWhitepaperError):
    """运行时主循环执行错误。"""


class IterationError(RuntimeExecutionError):
    """主循环单次迭代失败。"""


class StateRecoveryError(RuntimeExecutionError):
    """运行时状态恢复失败。"""


# ============================================================
# 组件域（可降级，非致命）
# ============================================================
class ComponentError(AgentWhitepaperError):
    """子系统组件错误。

    此类错误通常可降级处理：失败不影响主循环永续运行，
    仅影响该组件的功能（如技能抽象失败不影响Agent继续学习）。
    """


class SkillAbstractionError(ComponentError):
    """技能抽象失败（n-gram 抽取、技能合成）。"""


class MetaLearningError(ComponentError):
    """元学习参数调整失败。"""


class CoalitionError(ComponentError):
    """多Agent联盟组建失败。"""


class DistillationError(ComponentError):
    """跨Agent知识蒸馏失败。"""


class EvolutionError(ComponentError):
    """自主进化检测或恢复失败。"""


class MetricsCollectionError(ComponentError):
    """系统指标采集失败。"""


class CheckpointError(ComponentError):
    """检查点创建失败。"""


class AgentTaskError(ComponentError):
    """单个 Agent 任务执行失败（隔离级，不影响其他 Agent）。"""


# ============================================================
# 恢复域
# ============================================================
class RecoveryError(AgentWhitepaperError):
    """健康恢复操作失败（恢复处理器自身抛错）。"""


# ============================================================
# CLI 域
# ============================================================
class CLIError(AgentWhitepaperError):
    """命令行界面错误（用户交互层）。"""


# ============================================================
# 异常注册表（便于测试与反射式查询）
# ============================================================
#: 所有自定义异常类的列表（不含根类），按域分组
EXCEPTION_HIERARCHY: dict = {
    "config": [ConfigError],
    "persistence": [PersistenceError, TransactionError, StateSerializationError],
    "runtime": [RuntimeExecutionError, IterationError, StateRecoveryError],
    "component": [
        ComponentError,
        SkillAbstractionError,
        MetaLearningError,
        CoalitionError,
        DistillationError,
        EvolutionError,
        MetricsCollectionError,
        CheckpointError,
        AgentTaskError,
    ],
    "recovery": [RecoveryError],
    "cli": [CLIError],
}


def all_exception_types() -> list:
    """返回所有自定义异常类型（含根类），用于测试验证。"""
    types: list = [AgentWhitepaperError]
    for excs in EXCEPTION_HIERARCHY.values():
        types.extend(excs)
    return types


__all__ = [
    # 根
    "AgentWhitepaperError",
    # 配置域
    "ConfigError",
    # 持久化域
    "PersistenceError",
    "TransactionError",
    "StateSerializationError",
    # 运行时域
    "RuntimeExecutionError",
    "IterationError",
    "StateRecoveryError",
    # 组件域
    "ComponentError",
    "SkillAbstractionError",
    "MetaLearningError",
    "CoalitionError",
    "DistillationError",
    "EvolutionError",
    "MetricsCollectionError",
    "CheckpointError",
    "AgentTaskError",
    # 恢复域
    "RecoveryError",
    # CLI 域
    "CLIError",
    # 注册表
    "EXCEPTION_HIERARCHY",
    "all_exception_types",
]
