"""阶段H-6：智能错误恢复 - 系统的"自愈助手"。

第一性原理：
    "Self-healing"——错误时提供修复建议而非仅报错。
    用户遇到错误时，最需要的是"怎么修"，而非"什么错了"。
    智能错误恢复通过：
    1. 诊断：分析异常类型和上下文，识别根因
    2. 建议：针对根因提供具体修复步骤
    3. 自动修复：可安全自动执行的修复（如安装缺失依赖）
    4. 文档链接：指向相关文档章节

    将"排错专家知识"内置到系统中，降低支持成本。

用法：
    from agent_whitepaper.stage_h.recovery import diagnose_error, suggest_fix

    try:
        risky_operation()
    except Exception as e:
        diagnosis = diagnose_error(e)
        print(diagnosis.format())
        fix = suggest_fix(e)
        if fix:
            fix.apply()
"""

import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from agent_whitepaper._exceptions import (
    AgentWhitepaperError,
    CheckpointError,
    CoalitionError,
    ComponentError,
    ConfigError,
    DistillationError,
    EvolutionError,
    MetaLearningError,
    MetricsCollectionError,
    PersistenceError,
    RuntimeExecutionError,
    SkillAbstractionError,
    StateRecoveryError,
    StateSerializationError,
    TransactionError,
)


# ============================================================
# 诊断结果
# ============================================================
@dataclass
class Diagnosis:
    """错误诊断结果。

    Attributes:
        error_type: 异常类型名
        category: 错误类别（config/persistence/runtime/component/recovery/cli/unknown）
        root_cause: 根因分析
        suggestions: 修复建议列表
        is_recoverable: 是否可恢复
        auto_fix_available: 是否有自动修复可用
        docs_link: 相关文档链接
    """

    error_type: str
    category: str
    root_cause: str
    suggestions: List[str] = field(default_factory=list)
    is_recoverable: bool = True
    auto_fix_available: bool = False
    docs_link: str = ""

    def format(self) -> str:
        """格式化诊断结果为可读文本。"""
        lines = []
        lines.append(f"错误类型: {self.error_type}")
        lines.append(f"类别: {self.category}")
        lines.append(f"根因: {self.root_cause}")
        lines.append(f"可恢复: {'是' if self.is_recoverable else '否'}")
        if self.suggestions:
            lines.append("修复建议:")
            for i, s in enumerate(self.suggestions, 1):
                lines.append(f"  {i}. {s}")
        if self.auto_fix_available:
            lines.append("提示: 有可自动执行的修复")
        if self.docs_link:
            lines.append(f"文档: {self.docs_link}")
        return "\n".join(lines)


# ============================================================
# 自动修复
# ============================================================
@dataclass
class AutoFix:
    """自动修复方案。

    Attributes:
        name: 修复名称
        description: 修复描述
        apply_func: 执行修复的函数
        is_safe: 是否安全自动执行
    """

    name: str
    description: str
    apply_func: Callable[[], bool]
    is_safe: bool = True

    def apply(self) -> bool:
        """执行修复，返回是否成功。"""
        try:
            return self.apply_func()
        except Exception:  # 防火墙层：自动修复失败不应抛异常
            return False


# ============================================================
# 错误诊断器
# ============================================================
class ErrorRecovery:
    """智能错误恢复系统。

    第一性原理：
        "Self-healing"——错误时提供修复建议而非仅报错。
        将排错专家知识内置到系统，降低用户支持成本。

    用法：
        recovery = ErrorRecovery()
        diagnosis = recovery.diagnose(error)
        print(diagnosis.format())
        fix = recovery.suggest_fix(error)
        if fix and fix.is_safe:
            fix.apply()
    """

    def __init__(self) -> None:
        self._diagnosers: List[Callable[[Exception], Optional[Diagnosis]]] = [
            self._diagnose_import_error,
            self._diagnose_config_error,
            self._diagnose_persistence_error,
            self._diagnose_component_error,
            self._diagnose_runtime_error,
            self._diagnose_file_not_found,
            self._diagnose_permission_error,
            self._diagnose_generic,
        ]
        self._fixes: Dict[str, AutoFix] = self._build_fixes()

    def diagnose(self, error: Exception) -> Diagnosis:
        """诊断异常，返回诊断结果。"""
        for diagnoser in self._diagnosers:
            result = diagnoser(error)
            if result is not None:
                return result
        return self._diagnose_generic(error)

    def suggest_fix(self, error: Exception) -> Optional[AutoFix]:
        """建议自动修复方案。"""
        diagnosis = self.diagnose(error)
        if not diagnosis.auto_fix_available:
            return None
        # 根据错误类型查找修复
        for key, fix in self._fixes.items():
            if key in diagnosis.error_type.lower() or key in diagnosis.category:
                return fix
        return None

    def format_error(self, error: Exception) -> str:
        """格式化错误为用户友好的消息。"""
        diagnosis = self.diagnose(error)
        lines = [
            f"错误: {error}",
            "",
            diagnosis.format(),
        ]
        fix = self.suggest_fix(error)
        if fix:
            lines.append("")
            lines.append(f"可用自动修复: {fix.name}")
            lines.append(f"  {fix.description}")
        return "\n".join(lines)

    # --------------------------------------------------------
    # 诊断器
    # --------------------------------------------------------
    def _diagnose_import_error(
        self, error: Exception
    ) -> Optional[Diagnosis]:
        """诊断 ImportError。"""
        if not isinstance(error, ImportError):
            return None
        msg = str(error)
        # 提取模块名
        if "No module named" in msg:
            module_name = msg.split("'")[1] if "'" in msg else msg
            pip_name = self._module_to_pip(module_name)
            return Diagnosis(
                error_type="ImportError",
                category="dependency",
                root_cause=f"缺少依赖: {module_name}",
                suggestions=[
                    f"安装依赖: pip install {pip_name}",
                    "或安装全部依赖: pip install -e .",
                    "检查虚拟环境是否激活",
                ],
                is_recoverable=True,
                auto_fix_available=True,
                docs_link="TROUBLESHOOTING.md#依赖问题",
            )
        return Diagnosis(
            error_type="ImportError",
            category="dependency",
            root_cause=f"导入失败: {msg}",
            suggestions=["检查模块路径", "检查Python版本兼容性"],
            is_recoverable=True,
            docs_link="TROUBLESHOOTING.md#依赖问题",
        )

    def _diagnose_config_error(
        self, error: Exception
    ) -> Optional[Diagnosis]:
        """诊断配置错误。"""
        if isinstance(error, ConfigError):
            return Diagnosis(
                error_type=type(error).__name__,
                category="config",
                root_cause=f"配置问题: {error}",
                suggestions=[
                    "检查配置文件语法（YAML缩进）",
                    "使用 agentwp init 重新生成配置",
                    "使用 agentwp wizard 交互式配置",
                    "查看可用模板: agentwp init --list-templates",
                ],
                is_recoverable=True,
                auto_fix_available=False,
                docs_link="TROUBLESHOOTING.md#配置问题",
            )
        return None

    def _diagnose_persistence_error(
        self, error: Exception
    ) -> Optional[Diagnosis]:
        """诊断持久化错误。"""
        if isinstance(error, (PersistenceError, TransactionError,
                              StateSerializationError)):
            return Diagnosis(
                error_type=type(error).__name__,
                category="persistence",
                root_cause=f"持久化失败: {error}",
                suggestions=[
                    "检查数据库文件权限",
                    "检查磁盘空间",
                    "使用 :memory: 内存数据库测试",
                    "清理旧数据库文件后重试",
                ],
                is_recoverable=True,
                docs_link="TROUBLESHOOTING.md#持久化问题",
            )
        return None

    def _diagnose_component_error(
        self, error: Exception
    ) -> Optional[Diagnosis]:
        """诊断组件错误（可降级）。"""
        if isinstance(error, ComponentError):
            component_map = {
                SkillAbstractionError: "技能抽象",
                MetaLearningError: "元学习",
                CoalitionError: "联盟协作",
                DistillationError: "知识蒸馏",
                EvolutionError: "自主进化",
                MetricsCollectionError: "指标采集",
                CheckpointError: "检查点",
            }
            component = component_map.get(type(error), "组件")
            return Diagnosis(
                error_type=type(error).__name__,
                category="component",
                root_cause=f"{component}失败: {error}",
                suggestions=[
                    f"{component}失败不影响主循环，系统继续运行",
                    "检查日志获取详细信息: agentwp --verbose run",
                    "如果持续失败，尝试重启系统",
                ],
                is_recoverable=True,
                docs_link="TROUBLESHOOTING.md#组件问题",
            )
        return None

    def _diagnose_runtime_error(
        self, error: Exception
    ) -> Optional[Diagnosis]:
        """诊断运行时错误。"""
        if isinstance(error, (RuntimeExecutionError, StateRecoveryError)):
            return Diagnosis(
                error_type=type(error).__name__,
                category="runtime",
                root_cause=f"运行时错误: {error}",
                suggestions=[
                    "检查系统资源（内存/CPU）",
                    "查看健康监控日志",
                    "尝试从检查点恢复: agentwp status --db agent_system.db",
                    "重启系统: agentwp run --config config.yaml",
                ],
                is_recoverable=True,
                docs_link="TROUBLESHOOTING.md#运行时问题",
            )
        return None

    def _diagnose_file_not_found(
        self, error: Exception
    ) -> Optional[Diagnosis]:
        """诊断文件未找到错误。"""
        if isinstance(error, FileNotFoundError):
            filename = error.filename or str(error)
            return Diagnosis(
                error_type="FileNotFoundError",
                category="filesystem",
                root_cause=f"文件不存在: {filename}",
                suggestions=[
                    f"检查文件路径: {filename}",
                    "使用绝对路径",
                    "生成配置文件: agentwp init -o config.yaml",
                    "检查当前工作目录",
                ],
                is_recoverable=True,
                docs_link="TROUBLESHOOTING.md#文件问题",
            )
        return None

    def _diagnose_permission_error(
        self, error: Exception
    ) -> Optional[Diagnosis]:
        """诊断权限错误。"""
        if isinstance(error, PermissionError):
            return Diagnosis(
                error_type="PermissionError",
                category="filesystem",
                root_cause=f"权限不足: {error}",
                suggestions=[
                    "检查文件/目录权限",
                    "确认当前用户有读写权限",
                    "避免在系统目录运行",
                ],
                is_recoverable=True,
                docs_link="TROUBLESHOOTING.md#权限问题",
            )
        return None

    def _diagnose_generic(self, error: Exception) -> Diagnosis:
        """通用诊断（兜底）。"""
        if isinstance(error, AgentWhitepaperError):
            return Diagnosis(
                error_type=type(error).__name__,
                category="system",
                root_cause=f"系统错误: {error}",
                suggestions=[
                    "查看详细日志: agentwp --verbose run",
                    "运行环境诊断: agentwp doctor",
                    "查看文档: TROUBLESHOOTING.md",
                ],
                is_recoverable=True,
                docs_link="TROUBLESHOOTING.md",
            )
        return Diagnosis(
            error_type=type(error).__name__,
            category="unknown",
            root_cause=f"未知错误: {error}",
            suggestions=[
                "运行环境诊断: agentwp doctor",
                "查看详细日志: agentwp --verbose run",
                "提交问题报告（附完整错误信息）",
            ],
            is_recoverable=False,
            docs_link="TROUBLESHOOTING.md",
        )

    # --------------------------------------------------------
    # 自动修复方案
    # --------------------------------------------------------
    def _build_fixes(self) -> Dict[str, AutoFix]:
        """构建自动修复方案注册表。"""
        return {
            "dependency": AutoFix(
                name="安装缺失依赖",
                description="尝试 pip install 安装缺失的依赖",
                apply_func=self._fix_install_dependency,
                is_safe=True,
            ),
            "import": AutoFix(
                name="安装缺失模块",
                description="尝试安装缺失的Python模块",
                apply_func=self._fix_install_dependency,
                is_safe=True,
            ),
        }

    def _fix_install_dependency(self) -> bool:
        """尝试安装依赖（需要知道具体模块名）。"""
        # 注意：实际安装需要模块名，这里只是框架
        # 真正的安装会在 suggest_fix 后由调用者执行
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", "."],
                capture_output=True, timeout=60,
            )
            return True
        except (subprocess.SubprocessError, OSError):
            return False

    def _module_to_pip(self, module_name: str) -> str:
        """将模块名映射到pip包名。"""
        mapping = {
            "numpy": "numpy",
            "yaml": "pyyaml",
            "psutil": "psutil",
            "msgpack": "msgpack",
            "sympy": "sympy",
            "pytest": "pytest",
            "ruff": "ruff",
        }
        return mapping.get(module_name, module_name)


# ============================================================
# 模块级便捷函数
# ============================================================
_recovery: Optional[ErrorRecovery] = None


def _get_recovery() -> ErrorRecovery:
    """获取全局 ErrorRecovery 实例（惰性初始化）。"""
    global _recovery
    if _recovery is None:
        _recovery = ErrorRecovery()
    return _recovery


def diagnose_error(error: Exception) -> Diagnosis:
    """诊断异常，返回诊断结果。

    Args:
        error: 异常实例

    Returns:
        Diagnosis 诊断结果

    Example:
        >>> try:
        ...     risky()
        ... except Exception as e:
        ...     d = diagnose_error(e)
        ...     print(d.format())
    """
    return _get_recovery().diagnose(error)


def suggest_fix(error: Exception) -> Optional[AutoFix]:
    """建议自动修复方案。

    Args:
        error: 异常实例

    Returns:
        AutoFix 修复方案，或 None
    """
    return _get_recovery().suggest_fix(error)


def format_error_message(error: Exception) -> str:
    """格式化错误为用户友好的消息。

    Args:
        error: 异常实例

    Returns:
        格式化的错误消息字符串
    """
    return _get_recovery().format_error(error)
