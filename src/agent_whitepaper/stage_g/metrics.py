"""阶段G-1：指标采集与可观测性 - 系统的"仪表盘"。

第一性原理：
    可观测性是实用系统的前提。一个"黑盒"系统无法被信任、无法被调试、
    无法被改进。指标采集让系统的内部状态变得可见：

    1. 感知层：Agent学到了什么？（成功率、奖励、Q表规模）
    2. 增强层：技能增长如何？（技能数、组合技能数、元参数）
    3. 协作层：多Agent协作如何？（联盟数、消息数、角色匹配度）
    4. 迁移层：知识传播如何？（蒸馏次数、迁移状态数）
    5. 进化层：系统健康如何？（健康分、退化检测、恢复次数）
    6. 运行层：永续性如何？（迭代数、运行时间、任务完成数）

    指标是时间序列，记录系统"成长曲线"，让"自主进化"可被验证。
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================
# 单次快照：某一时刻系统的完整指标
# ============================================================
@dataclass
class MetricsSnapshot:
    """系统指标快照 - 某一迭代时刻的完整状态度量。"""
    iteration: int = 0
    timestamp: float = field(default_factory=time.time)

    # 感知层：Agent学习指标
    avg_success_rate: float = 0.0
    avg_reward: float = 0.0
    total_q_entries: int = 0          # Q表非零项数
    total_episodes: int = 0
    total_steps: int = 0

    # 增强层：技能指标
    skill_count: int = 0
    composite_skill_count: int = 0
    exploration_rate: float = 0.0     # 元学习探索率
    learning_rate: float = 0.0        # 元学习学习率

    # 协作层：多Agent指标
    active_coalitions: int = 0
    total_messages: int = 0
    avg_role_match: float = 0.0

    # 迁移层：知识传播指标
    distillation_count: int = 0
    migration_count: int = 0
    injected_states: int = 0

    # 进化层：健康指标
    health_score: float = 1.0
    degradation_detected: bool = False
    recovery_count: int = 0

    # 运行层：永续性指标
    uptime: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    status: str = "running"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "avg_success_rate": round(self.avg_success_rate, 4),
            "avg_reward": round(self.avg_reward, 4),
            "total_q_entries": self.total_q_entries,
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "skill_count": self.skill_count,
            "composite_skill_count": self.composite_skill_count,
            "exploration_rate": round(self.exploration_rate, 4),
            "learning_rate": round(self.learning_rate, 4),
            "active_coalitions": self.active_coalitions,
            "total_messages": self.total_messages,
            "avg_role_match": round(self.avg_role_match, 4),
            "distillation_count": self.distillation_count,
            "migration_count": self.migration_count,
            "injected_states": self.injected_states,
            "health_score": round(self.health_score, 4),
            "degradation_detected": self.degradation_detected,
            "recovery_count": self.recovery_count,
            "uptime": round(self.uptime, 2),
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "status": self.status,
        }


# ============================================================
# 指标采集器：收集、存储、查询时间序列指标
# ============================================================
class MetricsCollector:
    """指标采集器：记录系统成长曲线。

    核心功能：
    1. record(): 在每次迭代末尾记录一个快照
    2. get_history(): 获取指标时间序列
    3. get_trend(): 计算某指标的趋势（上升/下降/平稳）
    4. get_summary(): 获取指标摘要（首末值、均值、峰值）
    5. export(): 导出为JSON用于外部可视化
    """

    def __init__(self, max_history: int = 10000) -> None:
        self.max_history = max_history
        self._snapshots: List[MetricsSnapshot] = []
        # 累计计数器（不随快照重置）
        self._cumulative = {
            "distillation_count": 0,
            "migration_count": 0,
            "injected_states": 0,
            "recovery_count": 0,
        }

    def record(self, snapshot: MetricsSnapshot) -> None:
        """记录一个指标快照。"""
        # 累计计数器递增
        for key in self._cumulative:
            current = getattr(snapshot, key, 0)
            self._cumulative[key] = max(self._cumulative[key], current)
        self._snapshots.append(snapshot)
        # 限制历史长度（环形缓冲语义）
        if len(self._snapshots) > self.max_history:
            self._snapshots = self._snapshots[-self.max_history:]

    def increment(self, key: str, amount: int = 1) -> None:
        """递增累计计数器。"""
        if key in self._cumulative:
            self._cumulative[key] += amount

    def get_cumulative(self, key: str) -> int:
        """获取累计计数。"""
        return self._cumulative.get(key, 0)

    def get_history(self, metric: Optional[str] = None) -> List[Any]:
        """获取指标历史序列。

        Args:
            metric: 指标名（None=返回快照列表）
        """
        if metric is None:
            return list(self._snapshots)
        return [getattr(s, metric, None) for s in self._snapshots]

    def get_latest(self) -> Optional[MetricsSnapshot]:
        """获取最新快照。"""
        return self._snapshots[-1] if self._snapshots else None

    def get_trend(self, metric: str, window: int = 10) -> str:
        """计算指标趋势。

        Returns:
            "rising" | "falling" | "stable" | "insufficient"
        """
        history = self.get_history(metric)
        if len(history) < 2:
            return "insufficient"
        recent = history[-window:] if len(history) >= window else history
        if len(recent) < 2:
            return "insufficient"
        # 线性回归斜率
        n = len(recent)
        xs = list(range(n))
        mean_x = sum(xs) / n
        mean_y = sum(recent) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, recent))
        den = sum((x - mean_x) ** 2 for x in xs)
        if den == 0:
            return "stable"
        slope = num / den
        # 归一化斜率（相对于均值）
        if abs(mean_y) < 1e-9:
            return "stable" if abs(slope) < 1e-6 else ("rising" if slope > 0 else "falling")
        rel_slope = slope / abs(mean_y)
        if rel_slope > 0.01:
            return "rising"
        elif rel_slope < -0.01:
            return "falling"
        return "stable"

    def get_summary(self, metric: str) -> Dict[str, float]:
        """获取指标摘要。"""
        history = self.get_history(metric)
        if not history:
            return {"count": 0}
        return {
            "count": len(history),
            "first": history[0],
            "last": history[-1],
            "min": min(history),
            "max": max(history),
            "mean": sum(history) / len(history),
            "trend": self.get_trend(metric),
        }

    def export_json(self, path: Optional[str] = None) -> str:
        """导出指标为JSON。"""
        data = {
            "snapshots": [s.to_dict() for s in self._snapshots],
            "cumulative": dict(self._cumulative),
        }
        text = json.dumps(data, indent=2, ensure_ascii=False)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def __len__(self) -> int:
        return len(self._snapshots)
