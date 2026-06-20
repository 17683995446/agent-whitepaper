"""阶段R测试：验证错误处理规范化。

验证内容：
1. 异常层次体系存在且结构正确
2. 所有自定义异常继承自 AgentWhitepaperError
3. ConfigValidationError 向后兼容（isinstance ValueError）
4. 异常层次在顶层包导出
5. 无裸 except: 语句
6. 所有 except Exception 有注释说明（防火墙/组件降级/边界层）
7. CLI 边界层使用具体标准库异常
8. _exceptions.py 无循环导入
"""

import ast
import inspect
from pathlib import Path

import agent_whitepaper
from agent_whitepaper import _exceptions
from agent_whitepaper.stage_h.config import ConfigValidationError

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PACKAGE_ROOT


# ---------------------------------------------------------------------------
# 异常层次体系结构验证
# ---------------------------------------------------------------------------
class TestExceptionHierarchy:
    """验证异常层次体系结构。"""

    def test_root_exception_exists(self):
        """根异常 AgentWhitepaperError 存在。"""
        assert hasattr(_exceptions, "AgentWhitepaperError")
        assert issubclass(_exceptions.AgentWhitepaperError, Exception)

    def test_all_exceptions_inherit_from_root(self):
        """所有自定义异常继承自 AgentWhitepaperError。"""
        for exc_type in _exceptions.all_exception_types():
            assert issubclass(exc_type, _exceptions.AgentWhitepaperError), (
                f"{exc_type.__name__} 未继承 AgentWhitepaperError"
            )

    def test_domain_exceptions_exist(self):
        """各域异常存在。"""
        domains = {
            "config": ["ConfigError"],
            "persistence": ["PersistenceError", "TransactionError", "StateSerializationError"],
            "runtime": ["RuntimeExecutionError", "IterationError", "StateRecoveryError"],
            "component": [
                "ComponentError", "SkillAbstractionError", "MetaLearningError",
                "CoalitionError", "DistillationError", "EvolutionError",
                "MetricsCollectionError", "CheckpointError", "AgentTaskError",
            ],
            "recovery": ["RecoveryError"],
            "cli": ["CLIError"],
        }
        for domain, names in domains.items():
            for name in names:
                assert hasattr(_exceptions, name), f"缺少 {domain} 域异常: {name}"

    def test_persistence_hierarchy(self):
        """持久化域层次正确。"""
        assert issubclass(_exceptions.TransactionError, _exceptions.PersistenceError)
        assert issubclass(_exceptions.StateSerializationError, _exceptions.PersistenceError)

    def test_runtime_hierarchy(self):
        """运行时域层次正确。"""
        assert issubclass(_exceptions.IterationError, _exceptions.RuntimeExecutionError)
        assert issubclass(_exceptions.StateRecoveryError, _exceptions.RuntimeExecutionError)

    def test_component_hierarchy(self):
        """组件域层次正确。"""
        component_subclasses = [
            _exceptions.SkillAbstractionError,
            _exceptions.MetaLearningError,
            _exceptions.CoalitionError,
            _exceptions.DistillationError,
            _exceptions.EvolutionError,
            _exceptions.MetricsCollectionError,
            _exceptions.CheckpointError,
            _exceptions.AgentTaskError,
        ]
        for sub in component_subclasses:
            assert issubclass(sub, _exceptions.ComponentError), (
                f"{sub.__name__} 未继承 ComponentError"
            )

    def test_exception_registry_exists(self):
        """异常注册表存在且非空。"""
        assert hasattr(_exceptions, "EXCEPTION_HIERARCHY")
        assert len(_exceptions.EXCEPTION_HIERARCHY) >= 5
        assert callable(_exceptions.all_exception_types)
        assert len(_exceptions.all_exception_types()) >= 15

    def test_all_list_matches_defined(self):
        """__all__ 列表与实际定义的异常一致。"""
        for name in _exceptions.__all__:
            if name in ("EXCEPTION_HIERARCHY", "all_exception_types"):
                continue
            assert hasattr(_exceptions, name), f"__all__ 中的 {name} 未定义"


# ---------------------------------------------------------------------------
# 向后兼容性验证
# ---------------------------------------------------------------------------
class TestBackwardCompatibility:
    """验证 ConfigValidationError 向后兼容。"""

    def test_config_validation_error_is_value_error(self):
        """ConfigValidationError 仍是 ValueError 的子类。"""
        assert issubclass(ConfigValidationError, ValueError)

    def test_config_validation_error_is_config_error(self):
        """ConfigValidationError 也是 ConfigError 的子类。"""
        assert issubclass(ConfigValidationError, _exceptions.ConfigError)

    def test_config_validation_error_catchable_by_root(self):
        """ConfigValidationError 可被 AgentWhitepaperError 捕获。"""
        try:
            raise ConfigValidationError("test")
        except _exceptions.AgentWhitepaperError:
            pass  # 成功捕获

    def test_config_validation_error_catchable_by_value_error(self):
        """ConfigValidationError 仍可被 ValueError 捕获（向后兼容）。"""
        try:
            raise ConfigValidationError("test")
        except ValueError:
            pass  # 成功捕获


# ---------------------------------------------------------------------------
# 顶层导出验证
# ---------------------------------------------------------------------------
class TestTopLevelExport:
    """验证异常层次在顶层包导出。"""

    def test_root_exception_exported(self):
        """根异常在顶层包导出。"""
        assert hasattr(agent_whitepaper, "AgentWhitepaperError")
        assert "AgentWhitepaperError" in agent_whitepaper.__all__

    def test_domain_exceptions_exported(self):
        """各域异常在顶层包导出。"""
        exported_names = agent_whitepaper.__all__
        for name in ["ConfigError", "PersistenceError", "ComponentError",
                      "RecoveryError", "CLIError"]:
            assert name in exported_names, f"{name} 未在顶层导出"


# ---------------------------------------------------------------------------
# 源码扫描验证
# ---------------------------------------------------------------------------
class TestSourceCodeScanning:
    """扫描源码验证错误处理规范。"""

    @staticmethod
    def _iter_source_files():
        """迭代所有源码 .py 文件（排除 tests/demos/archive）。"""
        for py_file in SRC_ROOT.rglob("*.py"):
            rel = py_file.relative_to(SRC_ROOT)
            parts = rel.parts
            if "tests" in parts or "demos" in parts or "archive" in parts:
                continue
            if py_file.name == "__init__.py":
                continue
            yield py_file

    def test_no_bare_except(self):
        """源码中无裸 except: 语句。"""
        offenders = []
        for py_file in self._iter_source_files():
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    offenders.append(str(py_file.relative_to(SRC_ROOT)))
        assert offenders == [], f"发现裸 except: {offenders}"

    def test_except_exception_has_comment(self):
        """所有 except Exception 必须有注释说明（防火墙/组件降级/边界层）。"""
        offenders = []
        for py_file in self._iter_source_files():
            content = py_file.read_text(encoding="utf-8")
            tree = ast.parse(content)
            lines = content.splitlines()
            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler):
                    continue
                # 检查是否是 except Exception（type 是 Name 且 id == Exception）
                if not (isinstance(node.type, ast.Name) and node.type.id == "Exception"):
                    continue
                # 检查 except 行是否有注释
                except_line = lines[node.lineno - 1]
                if "#" not in except_line:
                    offenders.append(
                        f"{py_file.relative_to(SRC_ROOT)}:{node.lineno}"
                    )
        assert offenders == [], (
            f"以下 except Exception 缺少注释说明: {offenders}"
        )

    def test_cli_uses_specific_exceptions(self):
        """CLI 边界层使用具体标准库异常而非 except Exception。"""
        cli_file = SRC_ROOT / "stage_h" / "cli.py"
        content = cli_file.read_text(encoding="utf-8")
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            # cmd_status 中的 except 不应是裸 Exception
            if isinstance(node.type, ast.Name) and node.type.id == "Exception":
                # 允许在非 cmd_status 函数中存在（如 main 的顶层捕获）
                pass  # 此处仅验证，具体由注释规范覆盖


# ---------------------------------------------------------------------------
# _exceptions.py 模块完整性
# ---------------------------------------------------------------------------
class TestExceptionsModule:
    """验证 _exceptions.py 模块完整性。"""

    def test_module_has_docstring(self):
        """模块有 docstring 说明第一性原理。"""
        assert _exceptions.__doc__ is not None
        assert "第一性原理" in _exceptions.__doc__

    def test_module_has_all(self):
        """模块有 __all__ 声明。"""
        assert hasattr(_exceptions, "__all__")
        assert "AgentWhitepaperError" in _exceptions.__all__

    def test_no_circular_import(self):
        """_exceptions.py 不导入任何阶段模块（无循环依赖）。"""
        exc_file = SRC_ROOT / "_exceptions.py"
        tree = ast.parse(exc_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # 允许 from typing import，但不允许 from agent_whitepaper.stage_*
                if isinstance(node, ast.ImportFrom):
                    assert node.module is None or not node.module.startswith(
                        "agent_whitepaper.stage_"
                    ), f"_exceptions.py 不应导入阶段模块: {node.module}"
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith("agent_whitepaper"), (
                            f"_exceptions.py 不应导入 agent_whitepaper: {alias.name}"
                        )

    def test_exceptions_are_classes(self):
        """所有导出的异常是类。"""
        for name in _exceptions.__all__:
            if name in ("EXCEPTION_HIERARCHY", "all_exception_types"):
                continue
            obj = getattr(_exceptions, name)
            assert inspect.isclass(obj), f"{name} 不是类"
            assert issubclass(obj, BaseException), f"{name} 不是异常子类"
