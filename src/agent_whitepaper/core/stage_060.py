"""阶段60：最小占位实现——为阶段61提供ErrorValidationResult依赖。

本文件仅包含 ErrorValidationResult 数据类的最小实现，
满足阶段61的接口需求。完整实现见 stages/stage-060.md。
"""

from dataclasses import dataclass
from typing import Optional, Dict


@dataclass(frozen=True)
class ErrorValidationResult:
    """误差评估验证结果的最小占位实现。

    Attributes:
        detection_delay_mean: 检测延迟均值
        detection_delay_95: 检测延迟95分位数
        false_alarm_rate: 误报率
        ks_sensitivity: KS检验灵敏度
        mmd_sensitivity: MMD灵敏度
        route_correctness: 反馈路由正确率
        all_tests_passed: 所有测试是否通过
        test_details: 各测试的详细结果
    """
    detection_delay_mean: float = 0.0
    detection_delay_95: float = 0.0
    false_alarm_rate: float = 0.0
    ks_sensitivity: float = 0.0
    mmd_sensitivity: float = 0.0
    route_correctness: float = 0.0
    all_tests_passed: bool = False
    test_details: Optional[Dict] = None

    def __post_init__(self):
        if self.detection_delay_mean < 0:
            raise ValueError(
                f"检测延迟均值必须 >= 0，收到 {self.detection_delay_mean}"
            )
        if not (0 <= self.false_alarm_rate <= 1):
            raise ValueError(
                f"误报率必须 ∈ [0,1]，收到 {self.false_alarm_rate}"
            )
        if not (0 <= self.route_correctness <= 1):
            raise ValueError(
                f"路由正确率必须 ∈ [0,1]，收到 {self.route_correctness}"
            )

    @property
    def delay_test_passed(self) -> bool:
        """检测延迟测试是否通过（<5步）。"""
        return self.detection_delay_95 < 5.0

    @property
    def far_test_passed(self) -> bool:
        """误报率测试是否通过（<5%）。"""
        return self.false_alarm_rate < 0.05

    @property
    def sensitivity_test_passed(self) -> bool:
        """灵敏度测试是否通过。"""
        return self.ks_sensitivity <= 2.0 or self.mmd_sensitivity <= 2.0

    @property
    def route_test_passed(self) -> bool:
        """路由正确性测试是否通过（≥95%）。"""
        return self.route_correctness >= 0.95
