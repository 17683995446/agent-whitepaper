"""阶段E：技能迁移与真增强包。

公共API：
    QTableSnapshot: Q表快照
    QTableMigrator: Q表迁移器
    QTableKnowledgeBase: Q表知识库
    InjectionResult: 注入结果
    SkillQValueInjector: 技能Q值注入器
    DistillationResult: 蒸馏结果
    KnowledgeDistiller: 知识蒸馏器
    CollectiveLearningCoordinator: 集体学习协调器
"""

from .qtable_migration import (
    QTableSnapshot,
    QTableMigrator,
    QTableKnowledgeBase,
)
from .skill_injector import InjectionResult, SkillQValueInjector
from .knowledge_distiller import (
    DistillationResult,
    KnowledgeDistiller,
    CollectiveLearningCoordinator,
)

__all__ = [
    "QTableSnapshot",
    "QTableMigrator",
    "QTableKnowledgeBase",
    "InjectionResult",
    "SkillQValueInjector",
    "DistillationResult",
    "KnowledgeDistiller",
    "CollectiveLearningCoordinator",
]
