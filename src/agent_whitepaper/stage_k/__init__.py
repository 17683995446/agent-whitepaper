"""阶段K：长时运行实证与报告包。

公共API：
    LongRunConfig: 长时运行配置
    LongRunResult: 长时运行结果
    LongRunDemo: 长时运行演示
    quick_demo: 快速演示（便捷函数）
    full_demo: 完整演示（便捷函数）
    CapabilityAssessment: 能力评估
    TrendAnalysis: 趋势分析
    ReportGenerator: 报告生成器
"""

from .long_run_demo import (
    LongRunConfig,
    LongRunDemo,
    LongRunResult,
    full_demo,
    quick_demo,
)
from .report_generator import (
    CapabilityAssessment,
    ReportGenerator,
    TrendAnalysis,
)

__all__ = [
    "LongRunConfig",
    "LongRunResult",
    "LongRunDemo",
    "quick_demo",
    "full_demo",
    "CapabilityAssessment",
    "TrendAnalysis",
    "ReportGenerator",
]
