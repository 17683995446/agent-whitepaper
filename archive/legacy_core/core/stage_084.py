"""阶段84：禁止修改系统关键文件的规则引擎。

本阶段实现文件操作拦截与保护机制。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import re
import time

from .stage_083 import StorageCleaner


class FileOperation(Enum):
    """文件操作类型。"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    RENAME = "rename"
    CHMOD = "chmod"
    CREATE = "create"
    EXECUTE = "execute"


class ProtectionLevel(Enum):
    """文件保护级别。"""
    ABSOLUTE = "absolute"      # 绝对保护：任何修改操作都不允许
    CONDITIONAL = "conditional"  # 条件保护：仅允许读取


class InterceptResult(Enum):
    """文件操作拦截结果。"""
    ALLOW = "allow"
    DENY = "deny"


@dataclass(frozen=True)
class ProtectedPattern:
    """受保护路径规则。

    Attributes:
        pattern: 路径模式（支持通配符）
        regex: 编译后的正则表达式
        level: 保护级别
        description: 规则描述
    """
    pattern: str
    regex: re.Pattern
    level: ProtectionLevel
    description: str = ""


@dataclass(frozen=True)
class ViolationRecord:
    """违规操作记录。

    Attributes:
        timestamp: 违规时间戳
        operation: 操作类型
        file_path: 目标文件路径
        real_path: 解析后的真实路径
        matched_pattern: 匹配的受保护模式
        protection_level: 保护级别
        caller_info: 调用方信息
    """
    timestamp: float
    operation: FileOperation
    file_path: str
    real_path: str
    matched_pattern: str
    protection_level: ProtectionLevel
    caller_info: str = ""


@dataclass(frozen=True)
class FileProtectionEngineState:
    """文件保护引擎的状态描述。

    Attributes:
        protected_patterns: 受保护路径规则列表
        total_violations: 总违规次数
        violation_rate: 违规频率（次/小时）
        is_alert: 是否处于告警状态
        cache_size: 匹配缓存大小
        last_violation: 最近一次违规记录

    Invariants:
        - total_violations >= 0
        - violation_rate >= 0
    """
    protected_patterns: Tuple[ProtectedPattern, ...] = ()
    total_violations: int = 0
    violation_rate: float = 0.0
    is_alert: bool = False
    cache_size: int = 0
    last_violation: Optional[ViolationRecord] = None

    def __post_init__(self):
        if self.total_violations < 0:
            raise ValueError(
                f"total_violations必须 >= 0，收到 {self.total_violations}"
            )
        if self.violation_rate < 0:
            raise ValueError(
                f"violation_rate必须 >= 0，收到 {self.violation_rate}"
            )


@dataclass(frozen=True)
class FileProtectionEngine:
    """禁止修改系统关键文件的规则引擎。

    组合阶段83的StorageCleaner，在存储清理基础上增加文件保护状态。

    Attributes:
        storage_cleaner: 阶段83的存储清理器（组合字段）
        engine_state: 文件保护引擎状态
        alert_threshold: 违规告警阈值（默认10次/小时）
        max_cache_size: 最大缓存大小（默认10000）
        log_path: 违规日志文件路径
    """
    storage_cleaner: StorageCleaner
    engine_state: FileProtectionEngineState = field(
        default_factory=FileProtectionEngineState
    )
    alert_threshold: float = 10.0
    max_cache_size: int = 10000
    log_path: str = "violation.log"


class FileProtectionExecutor:
    """阶段84的核心接口：文件保护规则引擎执行器。

    实现受保护文件列表、文件操作拦截、规则匹配和违规日志。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 受保护路径列表运行时不可修改
        - 拦截延迟 < 1ms
        - 违规日志只追加不可修改
    """

    # 默认受保护路径模式（Linux）
    DEFAULT_LINUX_PATTERNS: Tuple[Tuple[str, ProtectionLevel, str], ...] = (
        (r"^/boot(/.*)?$", ProtectionLevel.ABSOLUTE, "引导加载器"),
        (r"^/etc/passwd$", ProtectionLevel.ABSOLUTE, "用户账户信息"),
        (r"^/etc/shadow$", ProtectionLevel.ABSOLUTE, "用户密码哈希"),
        (r"^/etc/sudoers$", ProtectionLevel.ABSOLUTE, "sudo权限配置"),
        (r"^/etc/ssh(/.*)?$", ProtectionLevel.ABSOLUTE, "SSH配置和密钥"),
        (r"^/etc/systemd(/.*)?$", ProtectionLevel.ABSOLUTE, "系统服务配置"),
        (r"^/proc(/.*)?$", ProtectionLevel.ABSOLUTE, "内核虚拟文件系统"),
        (r"^/sys(/.*)?$", ProtectionLevel.ABSOLUTE, "内核设备树"),
        (r"^/dev(/.*)?$", ProtectionLevel.ABSOLUTE, "设备文件"),
        (r"^/usr/bin(/.*)?$", ProtectionLevel.CONDITIONAL, "系统二进制文件"),
        (r"^/usr/lib(/.*)?$", ProtectionLevel.CONDITIONAL, "系统库文件"),
        (r"^/bin(/.*)?$", ProtectionLevel.CONDITIONAL, "核心可执行文件"),
        (r"^/sbin(/.*)?$", ProtectionLevel.CONDITIONAL, "系统管理命令"),
    )

    # 默认受保护路径模式（Windows）
    DEFAULT_WINDOWS_PATTERNS: Tuple[Tuple[str, ProtectionLevel, str], ...] = (
        (r"^C:\\Windows\\System32(\\.*)?$", ProtectionLevel.ABSOLUTE, "系统核心文件"),
        (r"^C:\\Windows\\SysWOW64(\\.*)?$", ProtectionLevel.ABSOLUTE, "64位系统文件"),
        (r"^C:\\Windows\\Win\.ini$", ProtectionLevel.ABSOLUTE, "Windows初始化文件"),
        (r"^HKEY_LOCAL_MACHINE(\\.*)?$", ProtectionLevel.ABSOLUTE, "注册表核心键"),
    )

    def _compile_pattern(
        self,
        pattern_str: str,
        level: ProtectionLevel,
        description: str
    ) -> ProtectedPattern:
        """编译路径模式为正则表达式。"""
        return ProtectedPattern(
            pattern=pattern_str,
            regex=re.compile(pattern_str),
            level=level,
            description=description
        )

    def build_protected_patterns(
        self,
        os_type: str = "linux",
        agent_install_path: str = ""
    ) -> Tuple[ProtectedPattern, ...]:
        """构建受保护路径列表（A84.1）。

        Args:
            os_type: 操作系统类型，支持"linux"|"darwin"|"windows"
            agent_install_path: 智能体安装路径

        Returns:
            受保护路径规则列表

        Raises:
            ValueError: 当os_type不支持时
        """
        if os_type not in {"linux", "darwin", "windows"}:
            raise ValueError(f"不支持的os_type: {os_type}")

        patterns: List[ProtectedPattern] = []

        if os_type in {"linux", "darwin"}:
            for pat, level, desc in self.DEFAULT_LINUX_PATTERNS:
                patterns.append(self._compile_pattern(pat, level, desc))
        elif os_type == "windows":
            for pat, level, desc in self.DEFAULT_WINDOWS_PATTERNS:
                patterns.append(self._compile_pattern(pat, level, desc))

        # 自身启动代码保护
        if agent_install_path:
            # 保护agent核心代码目录
            escaped_path = re.escape(agent_install_path)
            patterns.append(self._compile_pattern(
                rf"^{escaped_path}(/.*)?$",
                ProtectionLevel.ABSOLUTE,
                "智能体核心文件"
            ))

        return tuple(patterns)

    def normalize_path(self, file_path: str) -> str:
        """规范化文件路径（A84.2-normpath）。

        处理符号链接、..、.、//等。

        Args:
            file_path: 原始文件路径

        Returns:
            规范化后的绝对路径
        """
        import os
        try:
            # 解析符号链接并规范化路径
            real_path = os.path.realpath(file_path)
            return os.path.normpath(real_path)
        except Exception:
            # 如果路径不存在，仅做规范化
            return os.path.normpath(os.path.abspath(file_path))

    def regex_match(
        self,
        file_path: str,
        patterns: Tuple[ProtectedPattern, ...]
    ) -> Optional[ProtectedPattern]:
        """规则匹配（A84.3）。

        使用正则表达式检查文件路径是否匹配受保护模式。

        Args:
            file_path: 目标文件路径（已规范化）
            patterns: 受保护路径规则

        Returns:
            匹配的规则，无匹配返回None
        """
        for pattern in patterns:
            if pattern.regex.match(file_path):
                return pattern
        return None

    def intercept(
        self,
        file_path: str,
        operation: FileOperation,
        patterns: Tuple[ProtectedPattern, ...]
    ) -> Tuple[InterceptResult, Optional[ProtectedPattern]]:
        """拦截文件操作（A84.2）。

        Args:
            file_path: 目标文件路径
            operation: 操作类型
            patterns: 受保护路径规则列表

        Returns:
            (拦截结果, 匹配的规则)
        """
        # 路径规范化
        normalized = self.normalize_path(file_path)

        # 规则匹配
        matched = self.regex_match(normalized, patterns)
        if matched is None:
            return InterceptResult.ALLOW, None

        # 绝对保护：任何修改操作都拒绝
        if matched.level == ProtectionLevel.ABSOLUTE:
            if operation in {
                FileOperation.WRITE, FileOperation.DELETE,
                FileOperation.RENAME, FileOperation.CHMOD, FileOperation.CREATE
            }:
                return InterceptResult.DENY, matched
            # READ和EXECUTE在绝对保护路径上也拒绝（更安全）
            if operation in {FileOperation.READ, FileOperation.EXECUTE}:
                return InterceptResult.DENY, matched
            return InterceptResult.ALLOW, matched

        # 条件保护：仅允许读取，拒绝写入类操作
        if matched.level == ProtectionLevel.CONDITIONAL:
            if operation in {
                FileOperation.WRITE, FileOperation.DELETE,
                FileOperation.RENAME, FileOperation.CHMOD, FileOperation.CREATE
            }:
                return InterceptResult.DENY, matched
            return InterceptResult.ALLOW, matched

        return InterceptResult.ALLOW, matched

    def log_violation(
        self,
        operation: FileOperation,
        file_path: str,
        real_path: str,
        matched_pattern: ProtectedPattern,
        caller_info: str = ""
    ) -> ViolationRecord:
        """记录违规日志（A84.4）。

        Args:
            operation: 操作类型
            file_path: 目标文件路径
            real_path: 解析后的真实路径
            matched_pattern: 匹配的规则
            caller_info: 调用方信息

        Returns:
            违规记录
        """
        return ViolationRecord(
            timestamp=time.time(),
            operation=operation,
            file_path=file_path,
            real_path=real_path,
            matched_pattern=matched_pattern.pattern,
            protection_level=matched_pattern.level,
            caller_info=caller_info
        )

    def check_alert(
        self,
        violation_rate: float,
        threshold: float = 10.0
    ) -> bool:
        """检查是否应触发告警（A84.4-alert）。

        Args:
            violation_rate: 违规频率（次/小时），必须 >= 0
            threshold: 告警阈值，必须 > 0

        Returns:
            当violation_rate >= threshold时返回True

        Raises:
            ValueError: 当violation_rate < 0时
            ValueError: 当threshold <= 0时
        """
        if violation_rate < 0:
            raise ValueError(
                f"violation_rate必须 >= 0，收到 {violation_rate}"
            )
        if threshold <= 0:
            raise ValueError(
                f"threshold必须 > 0，收到 {threshold}"
            )
        return violation_rate >= threshold

    def step(
        self,
        engine: FileProtectionEngine,
        file_path: str,
        operation: FileOperation
    ) -> Tuple[FileProtectionEngine, InterceptResult, Optional[ViolationRecord]]:
        """执行一步文件保护检查。

        Args:
            engine: 当前文件保护引擎状态
            file_path: 目标文件路径
            operation: 操作类型

        Returns:
            (更新后的引擎状态, 拦截结果, 违规记录或None)
        """
        patterns = engine.engine_state.protected_patterns
        if not patterns:
            # 如果没有配置保护规则，使用默认规则
            patterns = self.build_protected_patterns("linux")

        result, matched = self.intercept(file_path, operation, patterns)

        if result == InterceptResult.DENY and matched is not None:
            # 记录违规
            normalized = self.normalize_path(file_path)
            violation = self.log_violation(
                operation, file_path, normalized, matched
            )
            new_total = engine.engine_state.total_violations + 1
            # 简化：假设每次step代表一小时内的操作，violation_rate直接计算
            new_rate = float(new_total)  # 简化模型
            is_alert = self.check_alert(new_rate, engine.alert_threshold)
            new_state = FileProtectionEngineState(
                protected_patterns=patterns,
                total_violations=new_total,
                violation_rate=new_rate,
                is_alert=is_alert,
                cache_size=engine.engine_state.cache_size,
                last_violation=violation
            )
            new_engine = FileProtectionEngine(
                storage_cleaner=engine.storage_cleaner,
                engine_state=new_state,
                alert_threshold=engine.alert_threshold,
                max_cache_size=engine.max_cache_size,
                log_path=engine.log_path
            )
            return new_engine, result, violation

        # 允许操作，不记录违规
        return engine, result, None
