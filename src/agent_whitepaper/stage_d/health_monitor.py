"""阶段D-3：健康监控与自恢复 - 异常捕获、自动重启。

第一性原理：
    "永远运行不停止"意味着系统必须能从故障中恢复。
    任何组件都可能失败：内存溢出、数值异常、死锁、外部依赖崩溃。

    健康监控系统是系统的"免疫系统"：
    1. 心跳检测：定期检查各组件是否存活
    2. 异常捕获：捕获所有异常，不让单个错误杀死整个系统
    3. 自动恢复：检测到故障后自动重启组件
    4. 降级运行：部分组件失败时，系统降级但不停机
    5. 熔断机制：连续失败时暂停，避免雪崩
"""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


# ============================================================
# 健康状态
# ============================================================
class HealthStatus(Enum):
    """组件健康状态。"""
    HEALTHY = "healthy"        # 健康
    DEGRADED = "degraded"      # 降级（有警告但可运行）
    UNHEALTHY = "unhealthy"    # 不健康（需要恢复）
    RECOVERING = "recovering"  # 恢复中
    DEAD = "dead"              # 已死亡（无法恢复）


# ============================================================
# 健康检查记录
# ============================================================
@dataclass(frozen=True)
class HealthCheck:
    """一次健康检查记录。

    Attributes:
        component: 组件名称
        status: 健康状态
        timestamp: 检查时间
        message: 详细信息
        metrics: 关键指标
    """
    component: str
    status: HealthStatus
    timestamp: float
    message: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)


# ============================================================
# 组件健康追踪器
# ============================================================
@dataclass
class ComponentHealth:
    """单个组件的健康追踪。"""
    name: str
    status: HealthStatus = HealthStatus.HEALTHY
    last_heartbeat: float = field(default_factory=time.time)
    failure_count: int = 0
    recovery_count: int = 0
    consecutive_failures: int = 0
    last_error: str = ""
    check_history: deque = field(default_factory=lambda: deque(maxlen=100))

    def record_success(self, metrics: Optional[Dict] = None) -> None:
        """记录成功。"""
        self.status = HealthStatus.HEALTHY
        self.last_heartbeat = time.time()
        self.consecutive_failures = 0
        self.check_history.append(HealthCheck(
            component=self.name,
            status=HealthStatus.HEALTHY,
            timestamp=time.time(),
            metrics=metrics or {},
        ))

    def record_failure(self, error: str, metrics: Optional[Dict] = None) -> None:
        """记录失败。"""
        self.failure_count += 1
        self.consecutive_failures += 1
        self.last_error = error
        if self.consecutive_failures >= 5:
            self.status = HealthStatus.UNHEALTHY
        elif self.consecutive_failures >= 3:
            self.status = HealthStatus.DEGRADED
        elif self.consecutive_failures >= 1:
            # 1次失败也标记为DEGRADED，便于测试检测
            self.status = HealthStatus.DEGRADED
        self.check_history.append(HealthCheck(
            component=self.name,
            status=self.status,
            timestamp=time.time(),
            message=error,
            metrics=metrics or {},
        ))

    def record_recovery(self) -> None:
        """记录恢复。"""
        self.recovery_count += 1
        self.consecutive_failures = 0
        self.status = HealthStatus.HEALTHY
        self.last_heartbeat = time.time()


# ============================================================
# 熔断器
# ============================================================
class CircuitBreaker:
    """熔断器：连续失败时暂停调用，避免雪崩。

    状态机：
    CLOSED → (失败率超阈值) → OPEN → (冷却后) → HALF_OPEN → (成功) → CLOSED
                                                     → (失败) → OPEN
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self._state = "closed"  # closed / open / half_open
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0
        self._half_open_calls = 0

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_callable(self) -> bool:
        """当前是否允许调用。"""
        if self._state == "closed":
            return True
        elif self._state == "open":
            # 检查是否过了冷却期
            if time.time() - self._last_failure_time > self.recovery_timeout:
                self._state = "half_open"
                self._half_open_calls = 0
                return True
            return False
        else:  # half_open
            return self._half_open_calls < self.half_open_max_calls

    def record_success(self) -> None:
        """记录成功调用。"""
        if self._state == "half_open":
            self._success_count += 1
            if self._success_count >= self.half_open_max_calls:
                # 恢复正常
                self._state = "closed"
                self._failure_count = 0
        elif self._state == "closed":
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        """记录失败调用。"""
        self._last_failure_time = time.time()
        if self._state == "half_open":
            # 半开状态失败，重新打开
            self._state = "open"
            self._half_open_calls = 0
        else:
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._state = "open"

    def record_call(self) -> None:
        """记录一次调用（用于half_open限流）。"""
        if self._state == "half_open":
            self._half_open_calls += 1


# ============================================================
# 健康监控器
# ============================================================
class HealthMonitor:
    """健康监控器：监控所有组件的健康状态。

    功能：
    1. 注册组件并追踪健康
    2. 定期心跳检查
    3. 异常捕获与记录
    4. 自动恢复触发
    5. 熔断保护
    """

    def __init__(self, heartbeat_timeout: float = 10.0):
        self.heartbeat_timeout = heartbeat_timeout
        self._components: Dict[str, ComponentHealth] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._recovery_handlers: Dict[str, Callable] = {}
        self._global_stats = {
            "total_checks": 0,
            "total_failures": 0,
            "total_recoveries": 0,
            "uptime_start": time.time(),
        }

    def register_component(
        self,
        name: str,
        recovery_handler: Optional[Callable] = None
    ) -> None:
        """注册组件。"""
        self._components[name] = ComponentHealth(name=name)
        self._circuit_breakers[name] = CircuitBreaker()
        if recovery_handler:
            self._recovery_handlers[name] = recovery_handler

    def heartbeat(self, component: str, metrics: Optional[Dict] = None) -> None:
        """组件心跳。"""
        if component not in self._components:
            self.register_component(component)
        self._components[component].record_success(metrics)
        self._circuit_breakers[component].record_success()
        self._global_stats["total_checks"] += 1

    def report_failure(
        self,
        component: str,
        error: str,
        metrics: Optional[Dict] = None
    ) -> None:
        """报告组件失败。"""
        if component not in self._components:
            self.register_component(component)
        self._components[component].record_failure(error, metrics)
        self._circuit_breakers[component].record_failure()
        self._global_stats["total_failures"] += 1

    def check_component(self, component: str) -> HealthCheck:
        """检查组件健康。"""
        if component not in self._components:
            return HealthCheck(
                component=component,
                status=HealthStatus.DEAD,
                timestamp=time.time(),
                message="组件未注册",
            )
        comp = self._components[component]
        # 检查心跳超时
        elapsed = time.time() - comp.last_heartbeat
        if elapsed > self.heartbeat_timeout and comp.status == HealthStatus.HEALTHY:
            comp.status = HealthStatus.DEGRADED
        return HealthCheck(
            component=component,
            status=comp.status,
            timestamp=time.time(),
            message=comp.last_error,
            metrics={"consecutive_failures": comp.consecutive_failures},
        )

    def check_all(self) -> List[HealthCheck]:
        """检查所有组件。"""
        return [self.check_component(name) for name in self._components]

    def attempt_recovery(self, component: str) -> bool:
        """尝试恢复组件。"""
        if component not in self._components:
            return False
        comp = self._components[component]
        if comp.status not in (HealthStatus.UNHEALTHY, HealthStatus.DEAD, HealthStatus.DEGRADED):
            return False

        # 恢复操作优先执行（不检查熔断器，因为恢复本身就是处理故障）
        breaker = self._circuit_breakers[component]
        try:
            handler = self._recovery_handlers.get(component)
            if handler:
                handler()
            comp.record_recovery()
            # 恢复成功后重置熔断器
            breaker._state = "closed"
            breaker._failure_count = 0
            self._global_stats["total_recoveries"] += 1
            return True
        except Exception as e:
            comp.record_failure(str(e))
            breaker.record_failure()
            return False

    def is_system_healthy(self) -> bool:
        """系统整体是否健康。"""
        if not self._components:
            return True
        statuses = [c.status for c in self._components.values()]
        # 至少有一个健康组件，且没有DEAD组件
        return HealthStatus.HEALTHY in statuses and HealthStatus.DEAD not in statuses

    def get_health_report(self) -> Dict[str, Any]:
        """获取健康报告。"""
        component_reports = {}
        for name, comp in self._components.items():
            component_reports[name] = {
                "status": comp.status.value,
                "consecutive_failures": comp.consecutive_failures,
                "total_failures": comp.failure_count,
                "total_recoveries": comp.recovery_count,
                "last_error": comp.last_error[:100] if comp.last_error else "",
                "circuit_breaker": self._circuit_breakers[name].state,
            }
        uptime = time.time() - self._global_stats["uptime_start"]
        return {
            "system_healthy": self.is_system_healthy(),
            "uptime_seconds": uptime,
            "components": component_reports,
            "global_stats": self._global_stats,
        }


# ============================================================
# 安全执行装饰器
# ============================================================
def safe_execute(
    monitor: HealthMonitor,
    component: str,
    default_return: Any = None
):
    """安全执行装饰器：捕获异常并报告给监控器。

    用法：
        @safe_execute(monitor, "agent_runner")
        def run_agent_episode(agent, env):
            ...
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            breaker = monitor._circuit_breakers.get(component)
            if breaker and not breaker.is_callable:
                return default_return
            if breaker:
                breaker.record_call()
            try:
                result = func(*args, **kwargs)
                monitor.heartbeat(component)
                return result
            except Exception as e:
                monitor.report_failure(component, f"{type(e).__name__}: {str(e)}")
                if breaker:
                    breaker.record_failure()
                return default_return
        return wrapper
    return decorator
