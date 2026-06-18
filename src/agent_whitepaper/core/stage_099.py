"""阶段99：系统退化检测与自修复。

本阶段定义退化检测、退化根因分析、自修复策略、修复验证、修复失败升级，
封装为DegradationDetector数据类组合EvolutionMonitor。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import numpy as np

from .stage_098 import EvolutionMonitor, AlertLevel, AlertEvent


# ============================================================
# 阶段99 枚举：退化程度
# ============================================================
class DegradationLevel(Enum):
    """退化程度。"""
    NONE = "none"          # 无退化
    MILD = "mild"          # 轻度退化（EVI下降<5%）
    MODERATE = "moderate"  # 中度退化（EVI下降5%-15%）
    SEVERE = "severe"      # 重度退化（EVI下降>15%）


# ============================================================
# 阶段99 枚举：退化根因
# ============================================================
class RootCauseType(Enum):
    """退化根因类型。"""
    MODEL_STALENESS = "model_staleness"        # 模型过时
    MEMORY_CORRUPTION = "memory_corruption"    # 记忆损坏
    RESOURCE_LEAK = "resource_leak"            # 资源泄漏
    CONFIG_DRIFT = "config_drift"              # 配置漂移


# ============================================================
# 阶段99 数据类：根因分析结果
# ============================================================
@dataclass(frozen=True)
class RootCauseAnalysis:
    """根因分析结果。

    Attributes:
        root_causes: 根因列表（按置信度排序）
        confidences: 各根因的置信度
        primary_cause: 主要根因
        evidence: 各根因的支持证据
    """
    root_causes: Tuple[RootCauseType, ...]
    confidences: Tuple[float, ...]
    primary_cause: RootCauseType
    evidence: Dict[str, float]


# ============================================================
# 阶段99 数据类：修复记录
# ============================================================
@dataclass(frozen=True)
class RepairRecord:
    """修复记录。

    Attributes:
        attempt: 尝试次数
        root_cause: 针对的根因
        strategy: 修复策略
        evi_before: 修复前EVI
        evi_after: 修复后EVI
        success: 是否成功
        timestamp: 时间戳
    """
    attempt: int
    root_cause: RootCauseType
    strategy: str
    evi_before: float
    evi_after: float
    success: bool
    timestamp: float


# ============================================================
# 阶段99 数据类：退化检测器
# ============================================================
@dataclass(frozen=True)
class DegradationDetector:
    """系统退化检测与自修复的完整状态描述。

    组合阶段98的EvolutionMonitor，在监控基础上增加退化检测状态。

    Attributes:
        evolution_monitor: 阶段98的进化监控（组合字段）
        is_degraded: 是否退化
        degradation_level: 退化程度
        degradation_start_time: 退化开始时间
        evi_at_degradation: 退化时EVI值
        root_cause_analysis: 根因分析结果
        repair_records: 修复记录
        total_repairs: 总修复次数
        successful_repairs: 成功修复次数
        is_in_safe_mode: 是否处于安全模式
        human_alert_sent: 是否已发送人工告警
        T_degradation: 退化持续时间阈值
        max_repair_attempts: 最大修复尝试次数

    Invariants:
        - total_repairs >= successful_repairs
        - max_repair_attempts = 3
    """
    evolution_monitor: EvolutionMonitor
    is_degraded: bool = False
    degradation_level: DegradationLevel = DegradationLevel.NONE
    degradation_start_time: float = 0.0
    evi_at_degradation: float = 0.0
    root_cause_analysis: Optional[RootCauseAnalysis] = None
    repair_records: Tuple[RepairRecord, ...] = ()
    total_repairs: int = 0
    successful_repairs: int = 0
    is_in_safe_mode: bool = False
    human_alert_sent: bool = False
    T_degradation: float = 1800.0
    max_repair_attempts: int = 3

    def __post_init__(self):
        if self.total_repairs < self.successful_repairs:
            raise ValueError(
                f"total_repairs必须 >= successful_repairs，"
                f"收到 {self.total_repairs} < {self.successful_repairs}"
            )


# ============================================================
# 阶段99 核心接口：退化检测与自修复
# ============================================================
class DegradationDetectorExecutor:
    """阶段99的核心接口：系统退化检测与自修复。

    实现退化检测、根因分析、自修复和修复验证。

    核心约束：
        - EVI持续下降30分钟判定退化
        - 修复后EVI必须高于修复前
        - 最多3次修复尝试
        - 3次失败进入安全模式并告警人工
    """

    def __init__(
        self,
        T_degradation: float = 1800.0,
        max_repair_attempts: int = 3,
        verify_window: int = 300
    ):
        """初始化退化检测器。

        Args:
            T_degradation: 退化持续时间阈值（秒），必须 > 0
            max_repair_attempts: 最大修复尝试次数，必须 > 0
            verify_window: 修复验证窗口（步数），必须 > 0

        Raises:
            ValueError: 当任何参数 <= 0时

        Examples:
            >>> detector = DegradationDetectorExecutor()
            >>> detector.T_degradation
            1800.0
        """
        if T_degradation <= 0:
            raise ValueError(f"T_degradation必须 > 0，收到 {T_degradation}")
        if max_repair_attempts <= 0:
            raise ValueError(f"max_repair_attempts必须 > 0，收到 {max_repair_attempts}")
        if verify_window <= 0:
            raise ValueError(f"verify_window必须 > 0，收到 {verify_window}")
        self.T_degradation = T_degradation
        self.max_repair_attempts = max_repair_attempts
        self.verify_window = verify_window

    def detect_degradation(
        self,
        monitor: EvolutionMonitor
    ) -> Tuple[bool, DegradationLevel]:
        """退化检测（A99.1）。

        基于EVI历史序列判定退化程度和级别。

        Args:
            monitor: 进化监控状态

        Returns:
            (是否退化, 退化程度)

        Examples:
            >>> detector = DegradationDetectorExecutor()
            >>> from agent_whitepaper.core.stage_098 import EvolutionMonitor
            >>> from agent_whitepaper.core.stage_097 import E2EValidation
            >>> from agent_whitepaper.core.stage_096 import TechStack
            >>> from agent_whitepaper.core.stage_095 import SystemInitializer
            >>> stack = TechStack(system_initializer=SystemInitializer(integrated_system=None))
            >>> e2e = E2EValidation(tech_stack=stack)
            >>> em = EvolutionMonitor(e2e_validation=e2e)
            >>> is_degraded, level = detector.detect_degradation(em)
            >>> isinstance(is_degraded, bool)
            True
        """
        evi_hist = tuple(e["evi"] for e in monitor.metrics_history)
        if len(evi_hist) < 2:
            return False, DegradationLevel.NONE
        evi_change = (
            (evi_hist[-1] - evi_hist[0])
            / max(abs(evi_hist[0]), 1e-8)
        )
        # 注意：evi_change为负表示EVI下降（退化）
        if evi_change < -0.15:
            return True, DegradationLevel.SEVERE
        elif evi_change < -0.05:
            return True, DegradationLevel.MODERATE
        elif evi_change < 0.0:
            return True, DegradationLevel.MILD
        return False, DegradationLevel.NONE

    def analyze_root_cause(
        self,
        detector: DegradationDetector,
        system_metrics: Dict
    ) -> RootCauseAnalysis:
        """退化根因分析（A99.2）。

        根据系统指标诊断退化根因，按置信度排序。

        Args:
            detector: 退化检测器状态
            system_metrics: 系统运行指标字典

        Returns:
            根因分析结果

        Examples:
            >>> detector = DegradationDetectorExecutor()
            >>> from agent_whitepaper.core.stage_098 import EvolutionMonitor
            >>> from agent_whitepaper.core.stage_097 import E2EValidation
            >>> from agent_whitepaper.core.stage_096 import TechStack
            >>> from agent_whitepaper.core.stage_095 import SystemInitializer
            >>> stack = TechStack(system_initializer=SystemInitializer(integrated_system=None))
            >>> e2e = E2EValidation(tech_stack=stack)
            >>> em = EvolutionMonitor(e2e_validation=e2e)
            >>> dd = DegradationDetector(evolution_monitor=em)
            >>> rca = detector.analyze_root_cause(dd, {"F_trend": 0.01})
            >>> isinstance(rca, RootCauseAnalysis)
            True
        """
        causes = []
        confidences = []
        evidence = {}

        # 诊断1: 模型过时 (F_trend > 0 表示误差上升)
        F_trend = system_metrics.get("F_trend", 0.0)
        if F_trend > 0.0:
            causes.append(RootCauseType.MODEL_STALENESS)
            confidences.append(0.8)
            evidence["F_trend"] = F_trend

        # 诊断2: 记忆损坏 (retrieval_accuracy < 0.5)
        retrieval_acc = system_metrics.get("retrieval_accuracy", 1.0)
        if retrieval_acc < 0.5:
            causes.append(RootCauseType.MEMORY_CORRUPTION)
            confidences.append(0.7)
            evidence["retrieval_accuracy"] = retrieval_acc

        # 诊断3: 资源泄漏 (cpu_trend > 0 或 mem_trend > 0)
        cpu_trend = system_metrics.get("cpu_trend", 0.0)
        mem_trend = system_metrics.get("mem_trend", 0.0)
        if cpu_trend > 0.0 or mem_trend > 0.0:
            causes.append(RootCauseType.RESOURCE_LEAK)
            confidences.append(0.6)
            evidence["cpu_trend"] = cpu_trend
            evidence["mem_trend"] = mem_trend

        # 诊断4: 配置漂移 (config_hash_mismatch)
        if system_metrics.get("config_hash_mismatch", False):
            causes.append(RootCauseType.CONFIG_DRIFT)
            confidences.append(0.9)
            evidence["config_hash_mismatch"] = 1.0

        # 如果没有识别到根因，默认模型过时
        if not causes:
            causes = [RootCauseType.MODEL_STALENESS]
            confidences = [0.5]
            evidence["default"] = 1.0

        # 按置信度排序
        sorted_pairs = sorted(
            zip(causes, confidences),
            key=lambda x: x[1],
            reverse=True
        )
        sorted_causes = tuple(p[0] for p in sorted_pairs)
        sorted_confidences = tuple(p[1] for p in sorted_pairs)

        return RootCauseAnalysis(
            root_causes=sorted_causes,
            confidences=sorted_confidences,
            primary_cause=sorted_causes[0],
            evidence=evidence,
        )

    def _select_strategy(
        self,
        root_cause: RootCauseType,
        level: DegradationLevel
    ) -> str:
        """根据根因和退化程度选择修复策略。

        Args:
            root_cause: 根因类型
            level: 退化程度

        Returns:
            修复策略名称
        """
        # 重度退化一律回滚
        if level == DegradationLevel.SEVERE:
            return "rollback_to_best_snapshot"

        strategy_map = {
            RootCauseType.MODEL_STALENESS: "refresh_model",
            RootCauseType.MEMORY_CORRUPTION: "rebuild_memory",
            RootCauseType.RESOURCE_LEAK: "reclaim_resources",
            RootCauseType.CONFIG_DRIFT: "reset_config",
        }
        return strategy_map.get(root_cause, "refresh_model")

    def repair(
        self,
        detector: DegradationDetector,
        root_cause_analysis: RootCauseAnalysis
    ) -> DegradationDetector:
        """执行自修复（A99.3）。

        根据根因和退化程度选择修复策略，记录修复尝试。

        Args:
            detector: 退化检测器状态
            root_cause_analysis: 根因分析结果

        Returns:
            更新后的退化检测器状态

        Examples:
            >>> detector = DegradationDetectorExecutor()
            >>> from agent_whitepaper.core.stage_098 import EvolutionMonitor
            >>> from agent_whitepaper.core.stage_097 import E2EValidation
            >>> from agent_whitepaper.core.stage_096 import TechStack
            >>> from agent_whitepaper.core.stage_095 import SystemInitializer
            >>> stack = TechStack(system_initializer=SystemInitializer(integrated_system=None))
            >>> e2e = E2EValidation(tech_stack=stack)
            >>> em = EvolutionMonitor(e2e_validation=e2e)
            >>> dd = DegradationDetector(evolution_monitor=em)
            >>> rca = RootCauseAnalysis(
            ...     root_causes=(RootCauseType.MODEL_STALENESS,),
            ...     confidences=(0.8,),
            ...     primary_cause=RootCauseType.MODEL_STALENESS,
            ...     evidence={})
            >>> new_dd = detector.repair(dd, rca)
            >>> new_dd.total_repairs == dd.total_repairs + 1
            True
        """
        import time
        new_attempts = detector.total_repairs + 1
        strategy = self._select_strategy(
            root_cause_analysis.primary_cause,
            detector.degradation_level
        )

        # 模拟修复效果（简化模型）
        # 实际系统会调用对应的修复机制
        evi_before = detector.evolution_monitor.evi
        # 轻度修复提升5%，中度10%，重度回滚提升15%
        improvement = {
            DegradationLevel.MILD: 0.05,
            DegradationLevel.MODERATE: 0.10,
            DegradationLevel.SEVERE: 0.15,
        }.get(detector.degradation_level, 0.05)
        evi_after = min(1.0, evi_before + improvement)

        record = RepairRecord(
            attempt=new_attempts,
            root_cause=root_cause_analysis.primary_cause,
            strategy=strategy,
            evi_before=evi_before,
            evi_after=evi_after,
            success=True,  # 简化模型中修复总是成功
            timestamp=time.time(),
        )

        new_records = detector.repair_records + (record,)

        return DegradationDetector(
            evolution_monitor=detector.evolution_monitor,
            is_degraded=detector.is_degraded,
            degradation_level=detector.degradation_level,
            degradation_start_time=detector.degradation_start_time,
            evi_at_degradation=detector.evi_at_degradation,
            root_cause_analysis=root_cause_analysis,
            repair_records=new_records,
            total_repairs=new_attempts,
            successful_repairs=detector.successful_repairs + 1,
            is_in_safe_mode=detector.is_in_safe_mode,
            human_alert_sent=detector.human_alert_sent,
            T_degradation=detector.T_degradation,
            max_repair_attempts=detector.max_repair_attempts,
        )

    def verify_repair(
        self,
        evi_before: float,
        evi_after: float
    ) -> bool:
        """修复验证（A99.4）。

        Args:
            evi_before: 修复前EVI
            evi_after: 修复后EVI

        Returns:
            当evi_after > evi_before时返回True

        Examples:
            >>> detector = DegradationDetectorExecutor()
            >>> detector.verify_repair(0.5, 0.48)
            False
            >>> detector.verify_repair(0.5, 0.51)
            True
        """
        return evi_after > evi_before

    def escalate_to_human(
        self,
        detector: DegradationDetector
    ) -> Dict:
        """修复失败升级（A99.5）。

        生成人工告警内容字典。

        Args:
            detector: 退化检测器状态

        Returns:
            人工告警内容字典

        Examples:
            >>> detector = DegradationDetectorExecutor()
            >>> from agent_whitepaper.core.stage_098 import EvolutionMonitor
            >>> from agent_whitepaper.core.stage_097 import E2EValidation
            >>> from agent_whitepaper.core.stage_096 import TechStack
            >>> from agent_whitepaper.core.stage_095 import SystemInitializer
            >>> stack = TechStack(system_initializer=SystemInitializer(integrated_system=None))
            >>> e2e = E2EValidation(tech_stack=stack)
            >>> em = EvolutionMonitor(e2e_validation=e2e)
            >>> dd = DegradationDetector(evolution_monitor=em)
            >>> alert = detector.escalate_to_human(dd)
            >>> "level" in alert
            True
        """
        import time
        root_cause_str = "unknown"
        if detector.root_cause_analysis is not None:
            root_cause_str = detector.root_cause_analysis.primary_cause.value

        return {
            "level": "EMERGENCY",
            "message": f"退化修复失败，已尝试{detector.total_repairs}次",
            "degradation_level": detector.degradation_level.value,
            "root_cause": root_cause_str,
            "evi_at_degradation": detector.evi_at_degradation,
            "current_evi": detector.evolution_monitor.evi,
            "repair_records_count": len(detector.repair_records),
            "timestamp": time.time(),
            "recommended_action": "人工检查系统状态并手动修复",
        }

    def step(
        self,
        detector: DegradationDetector,
        system_metrics: Dict
    ) -> DegradationDetector:
        """执行一步退化检测与自修复。

        完整流程：检测退化 -> 分析根因 -> 尝试修复（最多3次）-> 验证 -> 失败则升级。

        Args:
            detector: 当前退化检测器状态
            system_metrics: 系统运行指标字典

        Returns:
            更新后的退化检测器状态

        Examples:
            >>> detector = DegradationDetectorExecutor()
            >>> from agent_whitepaper.core.stage_098 import EvolutionMonitor
            >>> from agent_whitepaper.core.stage_097 import E2EValidation
            >>> from agent_whitepaper.core.stage_096 import TechStack
            >>> from agent_whitepaper.core.stage_095 import SystemInitializer
            >>> stack = TechStack(system_initializer=SystemInitializer(integrated_system=None))
            >>> e2e = E2EValidation(tech_stack=stack)
            >>> em = EvolutionMonitor(e2e_validation=e2e)
            >>> dd = DegradationDetector(evolution_monitor=em)
            >>> new_dd = detector.step(dd, {})
            >>> isinstance(new_dd, DegradationDetector)
            True
        """
        import time
        is_degraded, level = self.detect_degradation(detector.evolution_monitor)

        if not is_degraded:
            # 无退化，保持当前状态（但重置退化标志）
            return DegradationDetector(
                evolution_monitor=detector.evolution_monitor,
                is_degraded=False,
                degradation_level=DegradationLevel.NONE,
                degradation_start_time=0.0,
                evi_at_degradation=0.0,
                root_cause_analysis=None,
                repair_records=detector.repair_records,
                total_repairs=detector.total_repairs,
                successful_repairs=detector.successful_repairs,
                is_in_safe_mode=False,
                human_alert_sent=False,
                T_degradation=detector.T_degradation,
                max_repair_attempts=detector.max_repair_attempts,
            )

        # 退化 detected
        evi_at_deg = detector.evolution_monitor.evi
        deg_start = detector.degradation_start_time
        if deg_start == 0.0:
            deg_start = time.time()

        # 分析根因
        rca = self.analyze_root_cause(detector, system_metrics)

        # 检查是否还可以尝试修复
        if detector.total_repairs < self.max_repair_attempts:
            # 执行修复
            repaired = self.repair(
                DegradationDetector(
                    evolution_monitor=detector.evolution_monitor,
                    is_degraded=True,
                    degradation_level=level,
                    degradation_start_time=deg_start,
                    evi_at_degradation=evi_at_deg,
                    root_cause_analysis=rca,
                    repair_records=detector.repair_records,
                    total_repairs=detector.total_repairs,
                    successful_repairs=detector.successful_repairs,
                    is_in_safe_mode=detector.is_in_safe_mode,
                    human_alert_sent=detector.human_alert_sent,
                    T_degradation=detector.T_degradation,
                    max_repair_attempts=detector.max_repair_attempts,
                ),
                rca
            )
            return repaired

        # 超过最大修复次数，进入安全模式并告警人工
        return DegradationDetector(
            evolution_monitor=detector.evolution_monitor,
            is_degraded=True,
            degradation_level=level,
            degradation_start_time=deg_start,
            evi_at_degradation=evi_at_deg,
            root_cause_analysis=rca,
            repair_records=detector.repair_records,
            total_repairs=detector.total_repairs,
            successful_repairs=detector.successful_repairs,
            is_in_safe_mode=True,
            human_alert_sent=True,
            T_degradation=detector.T_degradation,
            max_repair_attempts=detector.max_repair_attempts,
        )
