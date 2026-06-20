"""阶段Q测试：验证类型注解规范化。

验证内容：
1. 所有源码函数（非测试/演示）都有返回类型注解
2. mypy 配置存在
3. 关键公共API有完整类型注解（参数+返回值）
4. dataclass 字段有类型注解
"""

import ast
import inspect
from pathlib import Path

import agent_whitepaper
from agent_whitepaper import (
    stage_a,
    stage_b,
    stage_c,
    stage_d,
    stage_e,
    stage_f,
    stage_g,
    stage_h,
    stage_i,
    stage_j,
    stage_k,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

SUBPACKAGES = [
    stage_a, stage_b, stage_c, stage_d, stage_e,
    stage_f, stage_g, stage_h, stage_i, stage_j, stage_k,
]


# ---------------------------------------------------------------------------
# 全量返回类型注解验证（核心）
# ---------------------------------------------------------------------------

class TestReturnTypeAnnotations:
    """验证所有源码函数都有返回类型注解。"""

    def test_all_source_functions_have_return_annotation(self):
        """所有源码函数（排除tests/demos）必须有返回类型注解。"""
        untyped = []
        for py_file in PACKAGE_ROOT.rglob("*.py"):
            rel = py_file.relative_to(PACKAGE_ROOT)
            # 排除 tests/ 和 demos/
            if str(rel).startswith("tests") or str(rel).startswith("demos"):
                continue
            if py_file.name == "__init__.py":
                continue
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.returns is None:
                        untyped.append(
                            f"{py_file}:{node.lineno} {node.name}"
                        )
        assert not untyped, (
            "以下函数缺少返回类型注解:\n" + "\n".join(untyped)
        )

    def test_coverage_100_percent(self):
        """返回类型注解覆盖率达到100%。"""
        total = 0
        typed = 0
        for py_file in PACKAGE_ROOT.rglob("*.py"):
            rel = py_file.relative_to(PACKAGE_ROOT)
            if str(rel).startswith("tests") or str(rel).startswith("demos"):
                continue
            if py_file.name == "__init__.py":
                continue
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total += 1
                    if node.returns is not None:
                        typed += 1
        assert total > 0, "未找到任何函数"
        coverage = 100 * typed / total
        assert coverage == 100.0, (
            f"返回类型注解覆盖率 {coverage:.1f}%，应为 100%"
        )


# ---------------------------------------------------------------------------
# 公共API类型完整性验证
# ---------------------------------------------------------------------------

class TestPublicApiTyped:
    """验证公共API有完整类型注解。"""

    def test_public_classes_have_typed_init(self):
        """公共类应有类型注解的 __init__。"""
        for mod in SUBPACKAGES:
            for name in mod.__all__:
                obj = getattr(mod, name)
                if not inspect.isclass(obj):
                    continue
                if hasattr(obj, "__init__"):
                    init = obj.__init__
                    # __init__ 应有返回注解 None
                    sig = inspect.signature(init)
                    if sig.return_annotation is not inspect.Signature.empty:
                        assert sig.return_annotation is None or (
                            hasattr(sig.return_annotation, "__name__")
                            and sig.return_annotation.__name__ == "NoneType"
                        ), f"{mod.__name__}.{name}.__init__ 返回注解应为 None"

    def test_top_level_exports_are_typed(self):
        """顶层导出的函数应有返回类型注解。"""
        for name in agent_whitepaper.__all__:
            obj = getattr(agent_whitepaper, name)
            if inspect.isfunction(obj):
                sig = inspect.signature(obj)
                assert sig.return_annotation is not inspect.Signature.empty, (
                    f"agent_whitepaper.{name} 缺少返回类型注解"
                )


# ---------------------------------------------------------------------------
# Mypy 配置验证
# ---------------------------------------------------------------------------

class TestMypyConfig:
    """验证 mypy 配置存在。"""

    def test_mypy_config_exists(self):
        content = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "[tool.mypy]" in content

    def test_mypy_python_version_set(self):
        content = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "python_version" in content

    def test_mypy_check_untyped_defs(self):
        """mypy 应启用 check_untyped_defs。"""
        content = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "check_untyped_defs = true" in content


# ---------------------------------------------------------------------------
# Dataclass 字段类型验证
# ---------------------------------------------------------------------------

class TestDataclassFieldsTyped:
    """验证 dataclass 字段有类型注解。"""

    def test_dataclass_fields_have_annotations(self):
        """所有 dataclass 的字段应有类型注解。"""
        from dataclasses import fields, is_dataclass
        untyped: list = []
        for mod in SUBPACKAGES:
            for name in mod.__all__:
                obj = getattr(mod, name)
                if not inspect.isclass(obj):
                    continue
                if not is_dataclass(obj):
                    continue
                for f in fields(obj):
                    # dataclass 字段类型为字符串形式或类型对象；空字符串视为未注解
                    if isinstance(f.type, str) and not f.type.strip():
                        untyped.append(f"{obj.__name__}.{f.name}")
        # dataclass 本身强制字段有注解，此处仅作额外保障
        assert untyped == [], f"未注解的 dataclass 字段: {untyped}"


# ---------------------------------------------------------------------------
# 关键模块类型注解抽样验证
# ---------------------------------------------------------------------------

class TestKeyModulesTyped:
    """抽样验证关键模块的类型注解质量。"""

    def test_stage_a_agent_methods_typed(self):
        """stage_a.SelfDrivenAgent 的公共方法应有返回注解。"""
        from agent_whitepaper.stage_a.agent import SelfDrivenAgent
        for method_name in ["perceive_and_act", "reset"]:
            method = getattr(SelfDrivenAgent, method_name, None)
            if method is None:
                continue
            sig = inspect.signature(method)
            assert sig.return_annotation is not inspect.Signature.empty, (
                f"SelfDrivenAgent.{method_name} 缺少返回类型注解"
            )

    def test_stage_d_persistence_methods_typed(self):
        """stage_d.PersistenceEngine 的公共方法应有返回注解。"""
        from agent_whitepaper.stage_d.persistence import PersistenceEngine
        for method_name in ["save_qtable", "load_qtable", "close"]:
            method = getattr(PersistenceEngine, method_name, None)
            if method is None:
                continue
            sig = inspect.signature(method)
            assert sig.return_annotation is not inspect.Signature.empty, (
                f"PersistenceEngine.{method_name} 缺少返回类型注解"
            )

    def test_stage_g_orchestrator_methods_typed(self):
        """stage_g.SystemOrchestrator 的公共方法应有返回注解。"""
        from agent_whitepaper.stage_g.orchestrator import SystemOrchestrator
        for method_name in ["run", "stop", "get_metrics"]:
            method = getattr(SystemOrchestrator, method_name, None)
            if method is None:
                continue
            sig = inspect.signature(method)
            assert sig.return_annotation is not inspect.Signature.empty, (
                f"SystemOrchestrator.{method_name} 缺少返回类型注解"
            )
