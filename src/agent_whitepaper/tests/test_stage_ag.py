"""阶段AG测试：验证智能错误恢复系统。

验证内容：
1. Diagnosis 诊断结果数据类
2. AutoFix 自动修复方案
3. ErrorRecovery 错误诊断器
4. diagnose_error 便捷函数
5. suggest_fix 便捷函数
6. format_error_message 便捷函数
7. CLI集成（cmd_run 使用 format_error_message）
8. 源码结构符合规范
"""

import inspect
from pathlib import Path

from agent_whitepaper._exceptions import (
    AgentWhitepaperError,
    CheckpointError,
    CoalitionError,
    ConfigError,
    DistillationError,
    EvolutionError,
    MetaLearningError,
    PersistenceError,
    RuntimeExecutionError,
    SkillAbstractionError,
    StateRecoveryError,
    StateSerializationError,
    TransactionError,
)
from agent_whitepaper.stage_h.recovery import (
    AutoFix,
    Diagnosis,
    ErrorRecovery,
    diagnose_error,
    format_error_message,
    suggest_fix,
)


# ---------------------------------------------------------------------------
# Diagnosis 数据类验证
# ---------------------------------------------------------------------------
class TestDiagnosis:
    """验证 Diagnosis 数据类。"""

    def test_diagnosis_creation(self):
        """创建诊断结果。"""
        d = Diagnosis(
            error_type="ConfigError",
            category="config",
            root_cause="配置错误",
        )
        assert d.error_type == "ConfigError"
        assert d.category == "config"
        assert d.root_cause == "配置错误"
        assert d.is_recoverable is True
        assert d.auto_fix_available is False
        assert d.suggestions == []

    def test_diagnosis_with_suggestions(self):
        """带建议的诊断结果。"""
        d = Diagnosis(
            error_type="ImportError",
            category="dependency",
            root_cause="缺少numpy",
            suggestions=["pip install numpy", "检查虚拟环境"],
        )
        assert len(d.suggestions) == 2

    def test_diagnosis_format(self):
        """format 方法生成可读文本。"""
        d = Diagnosis(
            error_type="ConfigError",
            category="config",
            root_cause="配置错误",
            suggestions=["检查YAML语法", "重新生成配置"],
            is_recoverable=True,
            docs_link="TROUBLESHOOTING.md",
        )
        text = d.format()
        assert "ConfigError" in text
        assert "config" in text
        assert "配置错误" in text
        assert "可恢复: 是" in text
        assert "检查YAML语法" in text
        assert "TROUBLESHOOTING.md" in text

    def test_diagnosis_format_no_suggestions(self):
        """无建议时format不包含建议章节。"""
        d = Diagnosis(
            error_type="Error",
            category="unknown",
            root_cause="未知",
        )
        text = d.format()
        assert "修复建议" not in text


# ---------------------------------------------------------------------------
# AutoFix 自动修复验证
# ---------------------------------------------------------------------------
class TestAutoFix:
    """验证 AutoFix 自动修复方案。"""

    def test_autofix_creation(self):
        """创建自动修复方案。"""
        fix = AutoFix(
            name="安装依赖",
            description="pip install",
            apply_func=lambda: True,
        )
        assert fix.name == "安装依赖"
        assert fix.is_safe is True

    def test_autofix_apply_success(self):
        """apply 成功返回 True。"""
        fix = AutoFix(
            name="test",
            description="test",
            apply_func=lambda: True,
        )
        assert fix.apply() is True

    def test_autofix_apply_failure(self):
        """apply 失败返回 False。"""
        fix = AutoFix(
            name="test",
            description="test",
            apply_func=lambda: False,
        )
        assert fix.apply() is False

    def test_autofix_apply_exception_returns_false(self):
        """apply 抛异常时返回 False（防火墙）。"""
        def failing():
            raise RuntimeError("boom")

        fix = AutoFix(
            name="test",
            description="test",
            apply_func=failing,
        )
        assert fix.apply() is False


# ---------------------------------------------------------------------------
# ErrorRecovery 诊断验证
# ---------------------------------------------------------------------------
class TestErrorRecoveryDiagnosis:
    """验证 ErrorRecovery 诊断功能。"""

    def setup_method(self):
        self.recovery = ErrorRecovery()

    def test_diagnose_import_error(self):
        """诊断 ImportError。"""
        error = ImportError("No module named 'numpy'")
        d = self.recovery.diagnose(error)
        assert d.error_type == "ImportError"
        assert d.category == "dependency"
        assert "numpy" in d.root_cause
        assert d.is_recoverable is True
        assert d.auto_fix_available is True
        assert any("pip install" in s for s in d.suggestions)

    def test_diagnose_config_error(self):
        """诊断 ConfigError。"""
        error = ConfigError("grid_size 必须 >= 2")
        d = self.recovery.diagnose(error)
        assert d.category == "config"
        assert "配置" in d.root_cause
        assert d.is_recoverable is True
        assert any("agentwp" in s for s in d.suggestions)

    def test_diagnose_persistence_error(self):
        """诊断 PersistenceError。"""
        error = PersistenceError("数据库锁定")
        d = self.recovery.diagnose(error)
        assert d.category == "persistence"
        assert "持久化" in d.root_cause

    def test_diagnose_transaction_error(self):
        """诊断 TransactionError（PersistenceError子类）。"""
        error = TransactionError("提交失败")
        d = self.recovery.diagnose(error)
        assert d.category == "persistence"

    def test_diagnose_state_serialization_error(self):
        """诊断 StateSerializationError。"""
        error = StateSerializationError("序列化失败")
        d = self.recovery.diagnose(error)
        assert d.category == "persistence"

    def test_diagnose_skill_abstraction_error(self):
        """诊断 SkillAbstractionError（组件错误）。"""
        error = SkillAbstractionError("n-gram抽取失败")
        d = self.recovery.diagnose(error)
        assert d.category == "component"
        assert "技能抽象" in d.root_cause
        assert d.is_recoverable is True

    def test_diagnose_meta_learning_error(self):
        """诊断 MetaLearningError。"""
        error = MetaLearningError("参数调整失败")
        d = self.recovery.diagnose(error)
        assert d.category == "component"
        assert "元学习" in d.root_cause

    def test_diagnose_coalition_error(self):
        """诊断 CoalitionError。"""
        error = CoalitionError("联盟组建失败")
        d = self.recovery.diagnose(error)
        assert d.category == "component"
        assert "联盟协作" in d.root_cause

    def test_diagnose_distillation_error(self):
        """诊断 DistillationError。"""
        error = DistillationError("蒸馏失败")
        d = self.recovery.diagnose(error)
        assert d.category == "component"
        assert "知识蒸馏" in d.root_cause

    def test_diagnose_evolution_error(self):
        """诊断 EvolutionError。"""
        error = EvolutionError("进化检测失败")
        d = self.recovery.diagnose(error)
        assert d.category == "component"
        assert "自主进化" in d.root_cause

    def test_diagnose_checkpoint_error(self):
        """诊断 CheckpointError。"""
        error = CheckpointError("检查点创建失败")
        d = self.recovery.diagnose(error)
        assert d.category == "component"
        assert "检查点" in d.root_cause

    def test_diagnose_runtime_error(self):
        """诊断 RuntimeExecutionError。"""
        error = RuntimeExecutionError("主循环错误")
        d = self.recovery.diagnose(error)
        assert d.category == "runtime"
        assert "运行时" in d.root_cause

    def test_diagnose_state_recovery_error(self):
        """诊断 StateRecoveryError。"""
        error = StateRecoveryError("恢复失败")
        d = self.recovery.diagnose(error)
        assert d.category == "runtime"

    def test_diagnose_file_not_found(self):
        """诊断 FileNotFoundError。"""
        error = FileNotFoundError(2, "No such file", "config.yaml")
        d = self.recovery.diagnose(error)
        assert d.error_type == "FileNotFoundError"
        assert d.category == "filesystem"
        assert "config.yaml" in d.root_cause

    def test_diagnose_permission_error(self):
        """诊断 PermissionError。"""
        error = PermissionError("权限不足")
        d = self.recovery.diagnose(error)
        assert d.error_type == "PermissionError"
        assert d.category == "filesystem"

    def test_diagnose_generic_agent_error(self):
        """诊断通用 AgentWhitepaperError。"""
        error = AgentWhitepaperError("系统错误")
        d = self.recovery.diagnose(error)
        assert d.category == "system"
        assert d.is_recoverable is True

    def test_diagnose_unknown_error(self):
        """诊断未知错误。"""
        error = RuntimeError("未知问题")
        d = self.recovery.diagnose(error)
        assert d.category == "unknown"
        assert d.is_recoverable is False

    def test_diagnose_returns_diagnosis(self):
        """diagnose 返回 Diagnosis 实例。"""
        d = self.recovery.diagnose(ValueError("test"))
        assert isinstance(d, Diagnosis)


# ---------------------------------------------------------------------------
# ErrorRecovery 修复建议验证
# ---------------------------------------------------------------------------
class TestErrorRecoveryFixSuggestion:
    """验证 ErrorRecovery 修复建议。"""

    def setup_method(self):
        self.recovery = ErrorRecovery()

    def test_suggest_fix_for_import_error(self):
        """ImportError 有自动修复。"""
        error = ImportError("No module named 'numpy'")
        fix = self.recovery.suggest_fix(error)
        assert fix is not None
        assert isinstance(fix, AutoFix)
        assert fix.is_safe is True

    def test_suggest_fix_for_config_error(self):
        """ConfigError 无自动修复。"""
        error = ConfigError("配置错误")
        fix = self.recovery.suggest_fix(error)
        assert fix is None

    def test_suggest_fix_returns_none_for_unknown(self):
        """未知错误无自动修复。"""
        error = RuntimeError("unknown")
        fix = self.recovery.suggest_fix(error)
        assert fix is None


# ---------------------------------------------------------------------------
# ErrorRecovery format_error 验证
# ---------------------------------------------------------------------------
class TestErrorRecoveryFormatError:
    """验证 ErrorRecovery.format_error。"""

    def setup_method(self):
        self.recovery = ErrorRecovery()

    def test_format_error_contains_error_message(self):
        """格式化错误包含原始错误消息。"""
        error = ConfigError("grid_size 必须 >= 2")
        text = self.recovery.format_error(error)
        assert "grid_size 必须 >= 2" in text

    def test_format_error_contains_diagnosis(self):
        """格式化错误包含诊断信息。"""
        error = ImportError("No module named 'numpy'")
        text = self.recovery.format_error(error)
        assert "ImportError" in text
        assert "dependency" in text
        assert "numpy" in text

    def test_format_error_contains_suggestions(self):
        """格式化错误包含修复建议。"""
        error = ConfigError("配置错误")
        text = self.recovery.format_error(error)
        assert "修复建议" in text

    def test_format_error_contains_autofix_for_import(self):
        """ImportError 的格式化错误包含自动修复提示。"""
        error = ImportError("No module named 'numpy'")
        text = self.recovery.format_error(error)
        assert "自动修复" in text


# ---------------------------------------------------------------------------
# 便捷函数验证
# ---------------------------------------------------------------------------
class TestConvenienceFunctions:
    """验证模块级便捷函数。"""

    def test_diagnose_error_returns_diagnosis(self):
        """diagnose_error 返回 Diagnosis。"""
        d = diagnose_error(ConfigError("test"))
        assert isinstance(d, Diagnosis)

    def test_diagnose_error_import_error(self):
        """diagnose_error 诊断 ImportError。"""
        d = diagnose_error(ImportError("No module named 'numpy'"))
        assert d.category == "dependency"

    def test_suggest_fix_returns_autofix_or_none(self):
        """suggest_fix 返回 AutoFix 或 None。"""
        fix = suggest_fix(ImportError("No module named 'numpy'"))
        assert fix is None or isinstance(fix, AutoFix)

    def test_format_error_message_returns_string(self):
        """format_error_message 返回字符串。"""
        text = format_error_message(ConfigError("test"))
        assert isinstance(text, str)
        assert "test" in text


# ---------------------------------------------------------------------------
# CLI集成验证
# ---------------------------------------------------------------------------
class TestCLIIntegration:
    """验证 CLI 集成智能错误恢复。"""

    def test_cmd_run_config_error_shows_diagnosis(self):
        """cmd_run 配置错误显示诊断信息。"""
        import argparse

        from agent_whitepaper.stage_h.cli import cmd_run

        args = argparse.Namespace(
            config="/nonexistent/config.yaml",
            iterations=None,
            agents=None,
            grid=None,
            memory=False,
            report_interval=10,
            quiet=False,
            verbose=False,
        )
        # 捕获 stderr
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            ret = cmd_run(args)
        assert ret == 1
        output = buf.getvalue()
        # 应该包含诊断信息（FileNotFoundError）
        assert "FileNotFoundError" in output or "文件不存在" in output

    def test_format_error_message_used_in_cli(self):
        """CLI 使用 format_error_message。"""
        source = Path(
            inspect.getfile(__import__("agent_whitepaper.stage_h.cli", fromlist=["x"]))
        ).read_text(encoding="utf-8")
        assert "format_error_message" in source


# ---------------------------------------------------------------------------
# 源码结构验证
# ---------------------------------------------------------------------------
class TestRecoverySourceCode:
    """验证智能错误恢复源码结构。"""

    def test_recovery_module_exists(self):
        """recovery.py 模块存在。"""
        from agent_whitepaper.stage_h import recovery
        assert recovery is not None

    def test_error_recovery_class_exists(self):
        """ErrorRecovery 类存在。"""
        assert callable(ErrorRecovery)

    def test_diagnosis_class_exists(self):
        """Diagnosis 类存在。"""
        assert callable(Diagnosis)

    def test_autofix_class_exists(self):
        """AutoFix 类存在。"""
        assert callable(AutoFix)

    def test_diagnose_error_function_exists(self):
        """diagnose_error 函数存在。"""
        assert callable(diagnose_error)

    def test_suggest_fix_function_exists(self):
        """suggest_fix 函数存在。"""
        assert callable(suggest_fix)

    def test_format_error_message_function_exists(self):
        """format_error_message 函数存在。"""
        assert callable(format_error_message)

    def test_error_recovery_has_docstring(self):
        """ErrorRecovery 有docstring。"""
        assert ErrorRecovery.__doc__ is not None
        assert "自愈" in ErrorRecovery.__doc__ or "恢复" in ErrorRecovery.__doc__

    def test_diagnose_method_has_docstring(self):
        """diagnose 方法有docstring。"""
        assert ErrorRecovery.diagnose.__doc__ is not None

    def test_recovery_exported_from_stage_h(self):
        """恢复组件从 stage_h 包导出。"""
        import agent_whitepaper.stage_h as stage_h
        assert stage_h.ErrorRecovery is ErrorRecovery
        assert stage_h.Diagnosis is Diagnosis
        assert stage_h.AutoFix is AutoFix
        assert stage_h.diagnose_error is diagnose_error
        assert stage_h.suggest_fix is suggest_fix
        assert stage_h.format_error_message is format_error_message

    def test_module_has_first_principle_docstring(self):
        """模块docstring包含第一性原理。"""
        source = Path(
            inspect.getfile(ErrorRecovery)
        ).read_text(encoding="utf-8")
        assert "第一性原理" in source
        assert "Self-healing" in source

    def test_all_functions_have_type_hints(self):
        """所有公共函数有返回类型注解。"""
        sig = inspect.signature(diagnose_error)
        assert sig.return_annotation is not inspect.Signature.empty

        sig = inspect.signature(suggest_fix)
        assert sig.return_annotation is not inspect.Signature.empty

        sig = inspect.signature(format_error_message)
        assert sig.return_annotation is not inspect.Signature.empty
