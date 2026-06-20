"""阶段55：异常误差检测——阈值检测、CUSUM与EWMA的统计过程控制。

本阶段实现异常检测的数学定义和计算接口。
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum
import numpy as np

from .stage_054 import PrecisionWeightedError


class AnomalyType(Enum):
    """异常类型。"""
    THRESHOLD = "threshold"
    CUSUM = "cusum"
    EWMA = "ewma"
    COMBINED = "combined"


@dataclass(frozen=True)
class AnomalyReport:
    """异常检测报告。"""
    is_anomaly: bool
    anomaly_type: AnomalyType
    anomaly_score: float
    threshold_flag: bool
    cusum_flag: bool
    ewma_flag: bool
    cusum_statistic: float
    ewma_statistic: float
    threshold_value: float
    detection_delay: int = 0

    def __post_init__(self):
        if self.anomaly_score < 0:
            raise ValueError(
                f"异常分数必须 >= 0，收到 {self.anomaly_score}"
            )


@dataclass(frozen=True)
class AnomalyDetector:
    """异常误差检测器的完整描述。

    组合阶段54的PrecisionWeightedError，在精度加权误差的基础上增加
    异常检测能力——阈值检测、CUSUM、EWMA、组合策略。

    Attributes:
        precision_weighted_error: 阶段54的精度加权误差（组合字段）
        threshold_k: 阈值倍数 k
        cusum_statistic: 当前CUSUM统计量 S_t
        cusum_k_sigma: CUSUM参考值 k_σ
        cusum_h: CUSUM决策阈值 h
        ewma_statistic: 当前EWMA统计量 z_t
        ewma_lambda: EWMA平滑因子 λ
        ewma_L: EWMA控制限倍数 L
        error_mean: 误差稳态均值 μ
        error_std: 误差稳态标准差 σ
        anomaly_score: 组合异常分数
        score_weights: 异常分数权重 [w1, w2, w3]
        last_anomaly_report: 最近一次异常报告

    Invariants:
        - threshold_k > 0
        - cusum_statistic >= 0
        - cusum_h > 0
        - ewma_lambda ∈ (0, 1)
        - ewma_L > 0
        - error_std > 0
        - anomaly_score >= 0
        - score_weights 归一化
    """
    precision_weighted_error: PrecisionWeightedError
    threshold_k: float = 3.0
    cusum_statistic: float = 0.0
    cusum_k_sigma: float = 0.0
    cusum_h: float = 0.0
    ewma_statistic: float = 0.0
    ewma_lambda: float = 0.1
    ewma_L: float = 3.0
    error_mean: float = 0.0
    error_std: float = 1.0
    anomaly_score: float = 0.0
    score_weights: Optional[np.ndarray] = None
    last_anomaly_report: Optional[AnomalyReport] = None

    def __post_init__(self):
        if self.threshold_k <= 0:
            raise ValueError(
                f"阈值倍数必须 > 0，收到 {self.threshold_k}"
            )
        if self.cusum_statistic < 0:
            raise ValueError(
                f"CUSUM统计量必须 >= 0，收到 {self.cusum_statistic}"
            )
        if self.cusum_h <= 0:
            raise ValueError(
                f"CUSUM决策阈值必须 > 0，收到 {self.cusum_h}"
            )
        if not (0 < self.ewma_lambda < 1):
            raise ValueError(
                f"EWMA平滑因子必须 ∈ (0,1)，收到 {self.ewma_lambda}"
            )
        if self.ewma_L <= 0:
            raise ValueError(
                f"EWMA控制限倍数必须 > 0，收到 {self.ewma_L}"
            )
        if self.error_std <= 0:
            raise ValueError(
                f"误差标准差必须 > 0，收到 {self.error_std}"
            )
        if self.anomaly_score < 0:
            raise ValueError(
                f"异常分数必须 >= 0，收到 {self.anomaly_score}"
            )
        if self.score_weights is not None:
            if not np.allclose(np.sum(self.score_weights), 1.0):
                raise ValueError("异常分数权重必须归一化")

    @property
    def is_anomaly(self) -> bool:
        """是否检测到异常。"""
        if self.last_anomaly_report is None:
            return False
        return self.last_anomaly_report.is_anomaly

    @property
    def anomaly_severity(self) -> str:
        """异常严重程度。"""
        if self.anomaly_score < 1.0:
            return "low"
        elif self.anomaly_score < 3.0:
            return "medium"
        else:
            return "high"


class AnomalyDetectionEngine:
    """阶段55的核心计算接口。

    实现阈值检测、CUSUM检测、EWMA检测和组合策略。
    核心约束：
        - CUSUM统计量非负
        - EWMA平滑因子 ∈ (0,1)
        - 异常分数非负
    """

    def detect_threshold(
        self,
        error: float,
        mean: float,
        std: float,
        k: float
    ) -> bool:
        """阈值检测（A55.1）。

        anomaly = 1[ε_t > μ + k·σ]

        Args:
            error: 当前误差 ε_t
            mean: 稳态均值 μ
            std: 稳态标准差 σ
            k: 阈值倍数

        Returns:
            是否异常
        """
        if k <= 0:
            raise ValueError(f"k 必须 > 0，收到 {k}")
        if std <= 0:
            raise ValueError(f"std 必须 > 0，收到 {std}")
        return error > mean + k * std

    def update_cusum(
        self,
        prev_statistic: float,
        error: float,
        mu_0: float,
        k_sigma: float
    ) -> float:
        """更新CUSUM统计量（A55.2）。

        S_t = max(0, S_{t-1} + ε_t - μ_0 - k_σ)

        Args:
            prev_statistic: 上一CUSUM统计量 S_{t-1}
            error: 当前误差 ε_t
            mu_0: 目标均值 μ_0
            k_sigma: 参考值 k_σ

        Returns:
            更新后的CUSUM统计量 S_t >= 0
        """
        return max(0.0, prev_statistic + error - mu_0 - k_sigma)

    def detect_cusum(
        self,
        statistic: float,
        h: float
    ) -> bool:
        """CUSUM异常判定（A55.2-alarm）。

        anomaly = 1[S_t > h]

        Args:
            statistic: CUSUM统计量 S_t
            h: 决策阈值

        Returns:
            是否异常
        """
        return statistic > h

    def update_ewma(
        self,
        prev_statistic: float,
        error: float,
        lam: float
    ) -> float:
        """更新EWMA统计量（A55.3）。

        z_t = λ·ε_t + (1-λ)·z_{t-1}

        Args:
            prev_statistic: 上一EWMA统计量 z_{t-1}
            error: 当前误差 ε_t
            lam: 平滑因子 λ ∈ (0,1)

        Returns:
            更新后的EWMA统计量 z_t
        """
        if not (0 < lam < 1):
            raise ValueError(f"lam 必须 ∈ (0,1)，收到 {lam}")
        return lam * error + (1.0 - lam) * prev_statistic

    def detect_ewma(
        self,
        statistic: float,
        mean: float,
        sigma_z: float,
        L: float
    ) -> bool:
        """EWMA异常判定（A55.3-alarm）。

        anomaly = 1[|z_t - μ_0| > L·σ_z]

        Args:
            statistic: EWMA统计量 z_t
            mean: 目标均值 μ_0
            sigma_z: EWMA稳态标准差
            L: 控制限倍数

        Returns:
            是否异常
        """
        return abs(statistic - mean) > L * sigma_z

    def compute_anomaly_score(
        self,
        error: float,
        mean: float,
        std: float,
        k: float,
        cusum_stat: float,
        h: float,
        ewma_stat: float,
        sigma_z: float,
        L: float,
        weights: np.ndarray
    ) -> float:
        """计算组合异常分数（A55.4）。

        Args:
            error: 当前误差
            mean: 稳态均值
            std: 稳态标准差
            k: 阈值倍数
            cusum_stat: CUSUM统计量
            h: CUSUM决策阈值
            ewma_stat: EWMA统计量
            sigma_z: EWMA稳态标准差
            L: EWMA控制限倍数
            weights: 权重 [w1, w2, w3]

        Returns:
            异常分数 >= 0
        """
        if not np.allclose(np.sum(weights), 1.0):
            raise ValueError("异常分数权重必须归一化")

        k_std = k * std
        threshold_component = (
            max(0.0, error - mean - k_std) / k_std
            if k_std > 0 else 0.0
        )
        cusum_component = cusum_stat / h if h > 0 else 0.0
        l_sigma_z = L * sigma_z
        ewma_component = (
            abs(ewma_stat - mean) / l_sigma_z
            if l_sigma_z > 0 else 0.0
        )

        score = (
            weights[0] * threshold_component
            + weights[1] * cusum_component
            + weights[2] * ewma_component
        )
        return max(0.0, score)

    def step(
        self,
        detector: AnomalyDetector,
        current_error: float
    ) -> AnomalyDetector:
        """执行一步完整的异常检测更新。

        Args:
            detector: 当前检测器状态
            current_error: 当前精度加权误差

        Returns:
            更新后的AnomalyDetector
        """
        error = current_error
        mean = detector.error_mean
        std = detector.error_std
        k = detector.threshold_k

        # 更新 CUSUM
        cusum_stat = self.update_cusum(
            detector.cusum_statistic, error, mean, detector.cusum_k_sigma
        )
        cusum_flag = self.detect_cusum(cusum_stat, detector.cusum_h)

        # 更新 EWMA
        ewma_stat = self.update_ewma(
            detector.ewma_statistic, error, detector.ewma_lambda
        )
        lam = detector.ewma_lambda
        denom = 2.0 - lam
        sigma_z = std * np.sqrt(lam / denom) if denom > 0 else std
        ewma_flag = self.detect_ewma(
            ewma_stat, mean, sigma_z, detector.ewma_L
        )

        # 阈值检测
        threshold_flag = self.detect_threshold(error, mean, std, k)

        # 组合异常分数
        weights = (
            detector.score_weights
            if detector.score_weights is not None
            else np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        )
        score = self.compute_anomaly_score(
            error, mean, std, k,
            cusum_stat, detector.cusum_h,
            ewma_stat, sigma_z, detector.ewma_L,
            weights,
        )

        is_anomaly = threshold_flag or cusum_flag or ewma_flag
        anomaly_type = AnomalyType.COMBINED
        if is_anomaly:
            if threshold_flag and not cusum_flag and not ewma_flag:
                anomaly_type = AnomalyType.THRESHOLD
            elif cusum_flag and not threshold_flag and not ewma_flag:
                anomaly_type = AnomalyType.CUSUM
            elif ewma_flag and not threshold_flag and not cusum_flag:
                anomaly_type = AnomalyType.EWMA

        report = AnomalyReport(
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type,
            anomaly_score=score,
            threshold_flag=threshold_flag,
            cusum_flag=cusum_flag,
            ewma_flag=ewma_flag,
            cusum_statistic=cusum_stat,
            ewma_statistic=ewma_stat,
            threshold_value=mean + k * std,
        )

        return AnomalyDetector(
            precision_weighted_error=detector.precision_weighted_error,
            threshold_k=k,
            cusum_statistic=cusum_stat,
            cusum_k_sigma=detector.cusum_k_sigma,
            cusum_h=detector.cusum_h,
            ewma_statistic=ewma_stat,
            ewma_lambda=lam,
            ewma_L=detector.ewma_L,
            error_mean=mean,
            error_std=std,
            anomaly_score=score,
            score_weights=weights,
            last_anomaly_report=report,
        )
