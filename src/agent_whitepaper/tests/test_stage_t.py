"""阶段T测试：验证日志与可观测性规范化。

验证内容：
1. _logging.py 模块存在且提供 get_logger/configure_logging
2. 所有组件降级路径（except Exception）使用 logger.debug 而非 pass
3. 顶层包导出日志配置函数
4. _logging.py 无循环导入
5. 日志名空间统一（agent_whitepaper.*）
"""

import ast
import logging
from pathlib import Path

import pytest

import agent_whitepaper
from agent_whitepaper import _logging

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PACKAGE_ROOT


# ---------------------------------------------------------------------------
# 模块定义验证
# ---------------------------------------------------------------------------
class TestLoggingModule:
    """验证 _logging.py 模块定义。"""

    def test_module_exists(self):
        """_logging 模块存在。"""
        assert _logging is not None

    def test_get_logger_callable(self):
        """get_logger 是可调用的。"""
        assert callable(_logging.get_logger)

    def test_configure_logging_callable(self):
        """configure_logging 是可调用的。"""
        assert callable(_logging.configure_logging)

    def test_module_has_all(self):
        """模块有 __all__ 声明。"""
        assert hasattr(_logging, "__all__")
        assert "get_logger" in _logging.__all__
        assert "configure_logging" in _logging.__all__

    def test_module_has_docstring(self):
        """模块有 docstring 说明第一性原理。"""
        assert _logging.__doc__ is not None
        assert "第一性原理" in _logging.__doc__

    def test_logger_name_constant(self):
        """LOGGER_NAME 常量存在。"""
        assert hasattr(_logging, "LOGGER_NAME")
        assert _logging.LOGGER_NAME == "agent_whitepaper"


# ---------------------------------------------------------------------------
# 日志功能验证
# ---------------------------------------------------------------------------
class TestLoggingFunctionality:
    """验证日志功能。"""

    def test_get_logger_returns_logger(self):
        """get_logger 返回 logging.Logger 实例。"""
        logger = _logging.get_logger("test")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_namespaced(self):
        """get_logger 返回的 logger 名称在 agent_whitepaper 命名空间下。"""
        logger = _logging.get_logger("my_module")
        assert logger.name.startswith("agent_whitepaper")

    def test_get_logger_with_full_name(self):
        """传入完整模块名时保留命名空间。"""
        logger = _logging.get_logger("agent_whitepaper.stage_d.runtime")
        assert logger.name == "agent_whitepaper.stage_d.runtime"

    def test_configure_logging_sets_level(self):
        """configure_logging 设置日志级别。"""
        _logging.configure_logging(level=logging.DEBUG)
        root = logging.getLogger("agent_whitepaper")
        assert root.level == logging.DEBUG
        # 恢复默认
        _logging.configure_logging(level=logging.WARNING)

    def test_configure_logging_no_duplicate_handlers(self):
        """多次调用 configure_logging 不产生重复 handler。"""
        _logging.configure_logging()
        _logging.configure_logging()
        root = logging.getLogger("agent_whitepaper")
        assert len(root.handlers) == 1


# ---------------------------------------------------------------------------
# 降级路径日志验证
# ---------------------------------------------------------------------------
class TestDegradationLogging:
    """验证组件降级路径使用日志而非静默 pass。"""

    @staticmethod
    def _iter_source_files():
        """迭代源码 .py 文件（排除 tests/demos/archive）。"""
        for py_file in SRC_ROOT.rglob("*.py"):
            rel = py_file.relative_to(SRC_ROOT)
            parts = rel.parts
            if "tests" in parts or "demos" in parts or "archive" in parts:
                continue
            if py_file.name == "__init__.py":
                continue
            yield py_file

    def test_no_silent_pass_after_except_exception(self):
        """组件降级路径不应有 'except Exception: pass'（静默吞错）。

        允许的例外：
        - except 具体异常: pass（如 ValueError, sqlite3.Error）是精确处理
        - 事务回滚（raise 重新抛出）
        """
        offenders = []
        for py_file in self._iter_source_files():
            content = py_file.read_text(encoding="utf-8")
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler):
                    continue
                # 只检查 except Exception（type 是 Name 且 id == Exception）
                if not (isinstance(node.type, ast.Name) and node.type.id == "Exception"):
                    continue
                # 检查 except 块体是否只有 pass
                body = node.body
                if len(body) != 1:
                    continue
                stmt = body[0]
                if not isinstance(stmt, ast.Pass):
                    continue
                # 这是 except Exception: pass，记录
                offenders.append(
                    f"{py_file.relative_to(SRC_ROOT)}:{node.lineno}"
                )
        assert offenders == [], (
            f"发现静默 except Exception: pass（应改为 logger.debug）: {offenders}"
        )

    def test_orchestrator_uses_logger(self):
        """orchestrator.py 使用 logger.debug 记录降级。"""
        orch_file = SRC_ROOT / "stage_g" / "orchestrator.py"
        content = orch_file.read_text(encoding="utf-8")
        assert "from agent_whitepaper._logging import get_logger" in content
        assert "logger = get_logger(__name__)" in content
        assert "logger.debug(" in content

    def test_runtime_uses_logger(self):
        """runtime.py 使用 logger.debug 记录降级。"""
        runtime_file = SRC_ROOT / "stage_d" / "runtime.py"
        content = runtime_file.read_text(encoding="utf-8")
        assert "from agent_whitepaper._logging import get_logger" in content
        assert "logger = get_logger(__name__)" in content
        assert "logger.debug(" in content


# ---------------------------------------------------------------------------
# 顶层导出验证
# ---------------------------------------------------------------------------
class TestTopLevelExport:
    """验证日志函数在顶层包导出。"""

    def test_get_logger_exported(self):
        """get_logger 在顶层包导出。"""
        assert hasattr(agent_whitepaper, "get_logger")
        assert "get_logger" in agent_whitepaper.__all__

    def test_configure_logging_exported(self):
        """configure_logging 在顶层包导出。"""
        assert hasattr(agent_whitepaper, "configure_logging")
        assert "configure_logging" in agent_whitepaper.__all__


# ---------------------------------------------------------------------------
# 无循环导入验证
# ---------------------------------------------------------------------------
class TestNoCircularImport:
    """验证 _logging.py 无循环导入。"""

    def test_logging_no_import_of_stages(self):
        """_logging.py 不导入任何阶段模块。"""
        log_file = SRC_ROOT / "_logging.py"
        content = log_file.read_text(encoding="utf-8")
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("agent_whitepaper.stage_"):
                    pytest.fail(
                        f"_logging.py 导入了阶段模块: {node.module}"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("agent_whitepaper.stage_"):
                        pytest.fail(
                            f"_logging.py 导入了阶段模块: {alias.name}"
                        )
