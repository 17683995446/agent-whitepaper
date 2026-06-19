"""阶段F-3：自主重训练机制 - 检测到退化后自动恢复。

第一性原理：
    当退化检测器发现性能下降时，系统需要自主决定如何恢复：
    1. 轻微退化 → 降低学习率，稳定化
    2. 中度退化 → 从检查点回滚Q表
    3. 严重退化 → 从集体知识库重新蒸馏
    4. 灾难性退化 → 完全重训练

    这是"自主进化"的闭环：退化→检测→恢复→验证→继续运行。
    系统不需要人工干预，自己发现问题、自己修复。
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum
import time
import numpy as np

from .degradation_detector import DegradationDetector, DegradationReport, DegradationType
from agent_whitepaper.stage_a.agent import SelfDrivenAgent
from agent_whitepaper.stage_a.environment import GridWorld
from agent_whitepaper.stage_e.knowledge_distiller import CollectiveLearningCoordinator
from agent_whitepaper.stage_e.qtable_migration import QTableSnapshot, QTableKnowledgeBase


# ============================================================
# 恢复策略
# ============================================================
class RecoveryStrategy(Enum):
    """恢复策略。"""
    NONE = "none"                       # 无需恢复
    STABILIZE = "stabilize"             # 稳定化（降学习率）
    ROLLBACK = "rollback"               # 回滚到检查点
    DISTILL = "distill"                 # 从集体知识重新蒸馏
    FULL_RETRAIN = "full_retrain"       # 完全重训练


# ============================================================
# 恢复结果
# ============================================================
@dataclass(frozen=True)
class RecoveryResult:
    """恢复操作结果。

    Attributes:
        strategy: 使用的恢复策略
        success: 是否成功
        before_score: 恢复前健康评分
        after_score: 恢复后健康评分
        duration: 恢复耗时（秒）
        details: 详细信息
    """
    strategy: RecoveryStrategy
    success: bool
    before_score: float
    after_score: float
    duration: float
    details: str


# ============================================================
# 检查点管理器
# ============================================================
class CheckpointManager:
    """检查点管理器：保存和恢复Q表快照。

    当Agent表现好时保存检查点，退化时可以回滚。
    """

    def __init__(self, max_checkpoints: int = 10):
        self.max_checkpoints = max_checkpoints
        self._checkpoints: List[Tuple[float, QTableSnapshot, float]] = []
        # (timestamp, snapshot, performance_score)

    def save_checkpoint(
        self,
        q_table,
        agent_id: str,
        performance_score: float
    ) -> None:
        """保存检查点。"""
        snapshot = QTableSnapshot.from_q_table(q_table, agent_id, performance_score)
        self._checkpoints.append((time.time(), snapshot, performance_score))
        # 只保留最好的max_checkpoints个
        if len(self._checkpoints) > self.max_checkpoints:
            # 按性能排序，保留最好的
            self._checkpoints.sort(key=lambda x: x[2], reverse=True)
            self._checkpoints = self._checkpoints[:self.max_checkpoints]

    def get_best_checkpoint(self) -> Optional[QTableSnapshot]:
        """获取性能最好的检查点。"""
        if not self._checkpoints:
            return None
        best = max(self._checkpoints, key=lambda x: x[2])
        return best[1]

    def get_latest_checkpoint(self) -> Optional[QTableSnapshot]:
        """获取最新的检查点。"""
        if not self._checkpoints:
            return None
        return self._checkpoints[-1][1]

    def rollback_to_best(self, q_table) -> bool:
        """回滚到最好的检查点。"""
        best = self.get_best_checkpoint()
        if best is None:
            return False
        if best.n_states != q_table.n_states or best.n_actions != q_table.n_actions:
            return False
        q_table._table = best.get_table_array().copy()
        return True


# ============================================================
# 自主重训练器
# ============================================================
class AutonomousRetrainer:
    """自主重训练器：检测退化并自动恢复。

    工作流程：
    1. 持续监控Agent性能
    2. 检测到退化时选择恢复策略
    3. 执行恢复操作
    4. 验证恢复效果
    5. 如果恢复失败，升级到更强策略
    """

    def __init__(
        self,
        detector: Optional[DegradationDetector] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        collective_coordinator: Optional[CollectiveLearningCoordinator] = None
    ):
        self.detector = detector or DegradationDetector()
        self.checkpoint_mgr = checkpoint_manager or CheckpointManager()
        self.collective = collective_coordinator
        self._recovery_history: List[RecoveryResult] = []
        self._recovery_count = 0

    def monitor_and_recover(
        self,
        agent: SelfDrivenAgent,
        env: GridWorld,
        force_strategy: Optional[RecoveryStrategy] = None
    ) -> RecoveryResult:
        """监控并恢复Agent性能。

        Args:
            agent: 要监控的Agent
            env: 训练环境
            force_strategy: 强制使用的恢复策略（None=自动选择）

        Returns:
            恢复结果
        """
        start_time = time.time()
        before_score = self.detector.get_health_score()

        # 检测退化
        report = self.detector.detect()

        # 选择策略
        if force_strategy:
            strategy = force_strategy
        else:
            strategy = self._select_strategy(report)

        if strategy == RecoveryStrategy.NONE:
            return RecoveryResult(
                strategy=strategy,
                success=True,
                before_score=before_score,
                after_score=before_score,
                duration=0.0,
                details="无需恢复",
            )

        # 执行恢复
        details = self._execute_recovery(strategy, agent, env, report)

        # 验证恢复效果
        after_score = self._verify_recovery(agent, env)

        duration = time.time() - start_time
        success = after_score >= before_score * 0.8  # 恢复到至少80%

        result = RecoveryResult(
            strategy=strategy,
            success=success,
            before_score=before_score,
            after_score=after_score,
            duration=duration,
            details=details,
        )
        self._recovery_history.append(result)
        self._recovery_count += 1
        return result

    def _select_strategy(self, report: DegradationReport) -> RecoveryStrategy:
        """根据退化报告选择恢复策略。"""
        if report.degradation_type == DegradationType.NONE:
            return RecoveryStrategy.NONE
        elif report.severity < 0.2:
            return RecoveryStrategy.STABILIZE
        elif report.severity < 0.4:
            return RecoveryStrategy.ROLLBACK
        elif report.severity < 0.7:
            return RecoveryStrategy.DISTILL
        else:
            return RecoveryStrategy.FULL_RETRAIN

    def _execute_recovery(
        self,
        strategy: RecoveryStrategy,
        agent: SelfDrivenAgent,
        env: GridWorld,
        report: DegradationReport
    ) -> str:
        """执行恢复操作。"""
        if strategy == RecoveryStrategy.STABILIZE:
            # 降低学习率
            old_lr = agent.q_table.lr
            agent.q_table.lr = max(0.01, old_lr * 0.5)
            return f"学习率 {old_lr:.3f} → {agent.q_table.lr:.3f}"

        elif strategy == RecoveryStrategy.ROLLBACK:
            # 回滚到检查点
            if self.checkpoint_mgr.rollback_to_best(agent.q_table):
                return "Q表已回滚到最佳检查点"
            return "无可用检查点，回滚失败"

        elif strategy == RecoveryStrategy.DISTILL:
            # 从集体知识重新蒸馏
            if self.collective is not None:
                distill_result, _ = self.collective.teach_new_agent(
                    agent.q_table, strategy="weighted", inject_skills=False
                )
                return f"蒸馏 {distill_result.states_distilled} 状态"
            return "无集体知识库，蒸馏失败"

        elif strategy == RecoveryStrategy.FULL_RETRAIN:
            # 完全重训练
            agent.q_table._table = np.zeros_like(agent.q_table._table)
            # 快速训练
            for _ in range(50):
                agent.run_episode(env)
            return "完全重训练（50情节）"

        return "未知策略"

    def _verify_recovery(
        self,
        agent: SelfDrivenAgent,
        env: GridWorld
    ) -> float:
        """验证恢复效果。"""
        # 运行几个测试情节
        test_successes = 0
        n_tests = 10
        for _ in range(n_tests):
            episode = agent.run_episode(env)
            if episode.success:
                test_successes += 1
        test_rate = test_successes / n_tests

        # 记录新指标
        self.detector.record(
            success_rate=test_rate,
            avg_reward=agent.stats.avg_reward,
            free_energy=agent.stats.free_energy_history[-1] if agent.stats.free_energy_history else 0.0,
        )
        return self.detector.get_health_score()

    def save_checkpoint(
        self,
        agent: SelfDrivenAgent,
        performance_score: float
    ) -> None:
        """保存Agent当前状态为检查点。"""
        self.checkpoint_mgr.save_checkpoint(
            agent.q_table, "agent", performance_score
        )

    def get_recovery_stats(self) -> Dict:
        """获取恢复统计。"""
        if not self._recovery_history:
            return {"total_recoveries": 0}
        successful = sum(1 for r in self._recovery_history if r.success)
        return {
            "total_recoveries": self._recovery_count,
            "successful_recoveries": successful,
            "success_rate": successful / len(self._recovery_history),
            "avg_recovery_duration": float(np.mean([r.duration for r in self._recovery_history])),
            "strategies_used": [r.strategy.value for r in self._recovery_history],
        }


# ============================================================
# 自主进化控制器
# ============================================================
class AutonomousEvolutionController:
    """自主进化控制器：协调退化检测与重训练。

    这是"自主进化"的顶层控制器：
    1. 定期检查Agent性能
    2. 发现退化时触发恢复
    3. 恢复后验证效果
    4. 记录进化历史
    5. 持续优化恢复策略
    """

    def __init__(
        self,
        grid_size: int = 5,
        collective_coordinator: Optional[CollectiveLearningCoordinator] = None
    ):
        self.grid_size = grid_size
        self.detector = DegradationDetector()
        self.checkpoint_mgr = CheckpointManager()
        self.retrainer = AutonomousRetrainer(
            detector=self.detector,
            checkpoint_manager=self.checkpoint_mgr,
            collective_coordinator=collective_coordinator,
        )
        self._evolution_log: List[Dict] = []

    def evolution_step(
        self,
        agent: SelfDrivenAgent,
        env: GridWorld
    ) -> Dict:
        """执行一次进化步骤。

        Returns:
            进化步骤报告
        """
        import time
        step_time = time.time()

        # 1. 记录当前性能
        recent_success = sum(agent.stats.recent_successes[-20:]) / max(len(agent.stats.recent_successes[-20:]), 1)
        self.detector.record(
            success_rate=recent_success,
            avg_reward=agent.stats.avg_reward,
            free_energy=agent.stats.free_energy_history[-1] if agent.stats.free_energy_history else 0.0,
        )

        # 2. 如果性能好，保存检查点
        if recent_success > 0.7:
            self.retrainer.save_checkpoint(agent, recent_success)

        # 3. 检测并恢复
        recovery = self.retrainer.monitor_and_recover(agent, env)

        # 4. 记录进化日志
        log_entry = {
            "timestamp": step_time,
            "success_rate": recent_success,
            "health_score": self.detector.get_health_score(),
            "recovery_strategy": recovery.strategy.value,
            "recovery_success": recovery.success,
            "details": recovery.details,
        }
        self._evolution_log.append(log_entry)

        return log_entry

    def get_evolution_summary(self) -> Dict:
        """获取进化摘要。"""
        if not self._evolution_log:
            return {"total_steps": 0, "status": "未开始"}
        recent = self._evolution_log[-20:]
        recoveries = sum(1 for e in recent if e["recovery_strategy"] != "none")
        return {
            "total_steps": len(self._evolution_log),
            "recent_recoveries": recoveries,
            "avg_health": float(np.mean([e["health_score"] for e in recent])),
            "current_success_rate": recent[-1]["success_rate"],
            "recovery_stats": self.retrainer.get_recovery_stats(),
        }
