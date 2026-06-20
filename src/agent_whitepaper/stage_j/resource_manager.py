"""阶段J-2：资源管理器 - 系统的"代谢调节"。

第一性原理：
    永续运行需要资源管理。一个不受资源约束的系统会：
    1. 内存耗尽 → 崩溃
    2. CPU过载 → 响应变慢
    3. 时间失控 → 任务堆积
    4. 存储爆炸 → 持久化失败

    资源管理器是系统的"代谢调节"：
    - 监控：实时跟踪资源使用
    - 限制：设定资源上限
    - 背压：资源紧张时减速或拒绝新任务
    - 回收：主动释放不用的资源

    背压（backpressure）是关键概念：
    当系统负载超过处理能力时，不是崩溃，而是优雅降级——
    减慢速度、拒绝新任务、清理旧数据。
    这让系统在压力下"活着"，而不是"死去"。
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from agent_whitepaper._logging import get_logger

logger = get_logger(__name__)


# ============================================================
# 资源限制配置
# ============================================================
@dataclass
class ResourceLimits:
    """资源限制配置。"""
    max_memory_mb: float = 512.0         # 最大内存（MB）
    max_cpu_percent: float = 90.0        # 最大CPU使用率
    max_uptime_seconds: float = 0.0      # 最大运行时间（0=无限）
    max_total_iterations: int = 0        # 最大迭代数（0=无限）
    max_memory_buffer_size: int = 10000  # 最大记忆缓冲大小
    max_skill_library_size: int = 10000  # 最大技能库大小
    max_q_table_entries: int = 1000000   # 最大Q表条目数
    max_persistence_size_mb: float = 100.0  # 最大持久化大小


# ============================================================
# 资源使用快照
# ============================================================
@dataclass
class ResourceUsage:
    """资源使用快照。"""
    timestamp: float = field(default_factory=time.time)
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    uptime_seconds: float = 0.0
    iteration_count: int = 0
    memory_buffer_size: int = 0
    skill_library_size: int = 0
    q_table_entries: int = 0

    @property
    def is_over_memory(self) -> bool:
        return self.memory_mb > 0  # 由manager判断阈值


# ============================================================
# 背压状态
# ============================================================
class BackpressureLevel(Enum):
    """背压等级。"""
    NONE = "none"           # 无背压
    LIGHT = "light"         # 轻度（减速）
    MODERATE = "moderate"   # 中度（拒绝低优先级任务）
    HEAVY = "heavy"         # 重度（只接受高优先级任务）
    CRITICAL = "critical"   # 危急（停止接受新任务）


# ============================================================
# 资源管理器
# ============================================================
class ResourceManager:
    """资源管理器：监控、限制、背压。

    核心功能：
    1. monitor(): 采集当前资源使用
    2. check_limits(): 检查是否超限
    3. get_backpressure(): 根据负载返回背压等级
    4. should_accept_task(): 根据背压决定是否接受新任务
    5. recommend_delay(): 根据背压推荐延迟时间
    """

    def __init__(
        self,
        limits: Optional[ResourceLimits] = None,
        start_time: Optional[float] = None
    ) -> None:
        self.limits = limits or ResourceLimits()
        self._start_time = start_time or time.time()
        self._usage_history: List[ResourceUsage] = []
        self._max_history = 1000
        self._psutil_available = self._check_psutil()

    def _check_psutil(self) -> bool:
        """检查psutil是否可用，缺失时记录友好提示。"""
        try:
            import psutil  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "psutil 未安装，CPU/内存监控将降级（仅跟踪迭代/缓冲区/技能库大小）。"
                "安装方式: pip install psutil"
            )
            return False

    def monitor(
        self,
        iteration_count: int = 0,
        memory_buffer_size: int = 0,
        skill_library_size: int = 0,
        q_table_entries: int = 0
    ) -> ResourceUsage:
        """采集当前资源使用。"""
        usage = ResourceUsage(
            uptime_seconds=time.time() - self._start_time,
            iteration_count=iteration_count,
            memory_buffer_size=memory_buffer_size,
            skill_library_size=skill_library_size,
            q_table_entries=q_table_entries,
        )

        if self._psutil_available:
            import os

            import psutil
            try:
                process = psutil.Process(os.getpid())
                usage.memory_mb = process.memory_info().rss / (1024 * 1024)
                usage.cpu_percent = process.cpu_percent(interval=0.0)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        else:
            # 无psutil时用sys.getsizeof粗略估计
            usage.memory_mb = 0.0
            usage.cpu_percent = 0.0

        self._usage_history.append(usage)
        if len(self._usage_history) > self._max_history:
            self._usage_history = self._usage_history[-self._max_history:]

        return usage

    def check_limits(self, usage: Optional[ResourceUsage] = None) -> List[str]:
        """检查是否超限，返回违规列表。"""
        usage = usage or (self._usage_history[-1] if self._usage_history else ResourceUsage())
        violations = []

        if self.limits.max_memory_mb > 0 and usage.memory_mb > self.limits.max_memory_mb:
            violations.append(f"内存超限: {usage.memory_mb:.1f}MB > {self.limits.max_memory_mb}MB")

        if self.limits.max_cpu_percent > 0 and usage.cpu_percent > self.limits.max_cpu_percent:
            violations.append(f"CPU超限: {usage.cpu_percent:.1f}% > {self.limits.max_cpu_percent}%")

        if self.limits.max_uptime_seconds > 0 and usage.uptime_seconds > self.limits.max_uptime_seconds:
            violations.append(f"运行时间超限: {usage.uptime_seconds:.0f}s > {self.limits.max_uptime_seconds}s")

        if self.limits.max_total_iterations > 0 and usage.iteration_count > self.limits.max_total_iterations:
            violations.append(f"迭代超限: {usage.iteration_count} > {self.limits.max_total_iterations}")

        if self.limits.max_memory_buffer_size > 0 and usage.memory_buffer_size > self.limits.max_memory_buffer_size:
            violations.append(f"记忆缓冲超限: {usage.memory_buffer_size} > {self.limits.max_memory_buffer_size}")

        if self.limits.max_skill_library_size > 0 and usage.skill_library_size > self.limits.max_skill_library_size:
            violations.append(f"技能库超限: {usage.skill_library_size} > {self.limits.max_skill_library_size}")

        if self.limits.max_q_table_entries > 0 and usage.q_table_entries > self.limits.max_q_table_entries:
            violations.append(f"Q表超限: {usage.q_table_entries} > {self.limits.max_q_table_entries}")

        return violations

    def get_backpressure(self, usage: Optional[ResourceUsage] = None) -> BackpressureLevel:
        """根据当前负载返回背压等级。

        背压分级逻辑（基于资源使用比例）：
        - <60%: NONE
        - 60-75%: LIGHT
        - 75-85%: MODERATE
        - 85-95%: HEAVY
        - >95%: CRITICAL
        """
        usage = usage or (self._usage_history[-1] if self._usage_history else ResourceUsage())

        # 计算各资源的压力比例
        pressures = []

        if self.limits.max_memory_mb > 0 and usage.memory_mb > 0:
            pressures.append(usage.memory_mb / self.limits.max_memory_mb)

        if self.limits.max_cpu_percent > 0 and usage.cpu_percent > 0:
            pressures.append(usage.cpu_percent / self.limits.max_cpu_percent)

        if self.limits.max_uptime_seconds > 0:
            pressures.append(usage.uptime_seconds / self.limits.max_uptime_seconds)

        if self.limits.max_total_iterations > 0:
            pressures.append(usage.iteration_count / self.limits.max_total_iterations)

        if self.limits.max_memory_buffer_size > 0:
            pressures.append(usage.memory_buffer_size / self.limits.max_memory_buffer_size)

        if self.limits.max_skill_library_size > 0:
            pressures.append(usage.skill_library_size / self.limits.max_skill_library_size)

        if not pressures:
            return BackpressureLevel.NONE

        max_pressure = max(pressures)

        if max_pressure > 0.95:
            return BackpressureLevel.CRITICAL
        elif max_pressure > 0.85:
            return BackpressureLevel.HEAVY
        elif max_pressure > 0.75:
            return BackpressureLevel.MODERATE
        elif max_pressure > 0.60:
            return BackpressureLevel.LIGHT
        else:
            return BackpressureLevel.NONE

    def should_accept_task(self, priority: str = "normal") -> bool:
        """根据背压决定是否接受新任务。

        Args:
            priority: 任务优先级 ("high"/"normal"/"low")
        """
        level = self.get_backpressure()
        if level == BackpressureLevel.NONE:
            return True
        elif level == BackpressureLevel.LIGHT:
            return True  # 轻度背压仍接受
        elif level == BackpressureLevel.MODERATE:
            return priority in ("high", "normal")  # 拒绝低优先级
        elif level == BackpressureLevel.HEAVY:
            return priority == "high"  # 只接受高优先级
        else:  # CRITICAL
            return False  # 停止接受

    def recommend_delay(self) -> float:
        """根据背压推荐迭代间延迟（秒）。"""
        level = self.get_backpressure()
        delays = {
            BackpressureLevel.NONE: 0.0,
            BackpressureLevel.LIGHT: 0.01,
            BackpressureLevel.MODERATE: 0.1,
            BackpressureLevel.HEAVY: 0.5,
            BackpressureLevel.CRITICAL: 2.0,
        }
        return delays.get(level, 0.0)

    def get_usage_report(self) -> Dict:
        """获取资源使用报告。"""
        latest = self._usage_history[-1] if self._usage_history else ResourceUsage()
        violations = self.check_limits(latest)
        backpressure = self.get_backpressure(latest)
        return {
            "current_usage": {
                "memory_mb": round(latest.memory_mb, 2),
                "cpu_percent": round(latest.cpu_percent, 2),
                "uptime_seconds": round(latest.uptime_seconds, 1),
                "iteration_count": latest.iteration_count,
                "memory_buffer_size": latest.memory_buffer_size,
                "skill_library_size": latest.skill_library_size,
                "q_table_entries": latest.q_table_entries,
            },
            "limits": {
                "max_memory_mb": self.limits.max_memory_mb,
                "max_cpu_percent": self.limits.max_cpu_percent,
                "max_uptime_seconds": self.limits.max_uptime_seconds,
                "max_total_iterations": self.limits.max_total_iterations,
            },
            "backpressure": backpressure.value,
            "violations": violations,
            "is_over_limit": len(violations) > 0,
            "psutil_available": self._psutil_available,
        }

    @property
    def latest_usage(self) -> Optional[ResourceUsage]:
        return self._usage_history[-1] if self._usage_history else None


# ============================================================
# 资源感知的迭代控制器
# ============================================================
class ResourceAwareController:
    """资源感知的迭代控制器：将资源管理集成到运行循环。

    工作流：
    1. 每次迭代前检查资源
    2. 根据背压调整行为（延迟/拒绝任务）
    3. 资源超限时触发保护
    """

    def __init__(
        self,
        resource_manager: ResourceManager,
        safety_monitor=None,
    ) -> None:
        self.resource_mgr = resource_manager
        self.safety_monitor = safety_monitor
        self._iteration = 0
        self._paused = False
        self._pause_reason = ""

    def before_iteration(self) -> Dict:
        """迭代前检查，返回控制决策。"""
        self._iteration += 1
        usage = self.resource_mgr.monitor(iteration_count=self._iteration)
        violations = self.resource_mgr.check_limits(usage)
        backpressure = self.resource_mgr.get_backpressure(usage)
        delay = self.resource_mgr.recommend_delay()

        decision = {
            "should_run": True,
            "delay_seconds": delay,
            "backpressure": backpressure.value,
            "violations": violations,
            "accept_new_tasks": self.resource_mgr.should_accept_task(),
        }

        # 严重违规时暂停
        if violations and backpressure == BackpressureLevel.CRITICAL:
            self._paused = True
            self._pause_reason = "资源临界: " + "; ".join(violations)
            decision["should_run"] = False

        # 安全监控器触发停止
        if self.safety_monitor and self.safety_monitor.should_stop:
            decision["should_run"] = False
            self._paused = True
            self._pause_reason = self.safety_monitor.stop_reason

        return decision

    def after_iteration(self) -> None:
        """迭代后清理。"""
        pass

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def pause_reason(self) -> str:
        return self._pause_reason

    def resume(self) -> None:
        """恢复运行。"""
        self._paused = False
        self._pause_reason = ""
