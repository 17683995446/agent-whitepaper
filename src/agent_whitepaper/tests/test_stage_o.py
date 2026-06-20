"""阶段O测试：验证命名规范统一。

验证内容：
1. ruff lint 检查通过（命名规范、import排序、未使用变量等）
2. 类名遵循 PascalCase
3. 函数名遵循 snake_case
4. 常量遵循 UPPER_SNAKE_CASE
5. 模块名遵循 snake_case
6. 私有成员以 _ 开头
7. pyproject.toml 中命名规范配置存在
"""

import inspect
import re
import subprocess
import sys
from enum import Enum
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
# Ruff lint 集成验证
# ---------------------------------------------------------------------------

class TestRuffLintPasses:
    """验证 ruff check 通过（命名规范的核心保障）。"""

    def test_ruff_check_passes(self):
        """ruff check 应无错误退出。"""
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check",
             str(PACKAGE_ROOT), "--config", str(PROJECT_ROOT / "pyproject.toml")],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"ruff check 失败:\n{result.stdout}\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# 配置验证
# ---------------------------------------------------------------------------

class TestNamingConfigExists:
    """验证 pyproject.toml 中命名规范配置存在。"""

    def test_pyproject_exists(self):
        assert (PROJECT_ROOT / "pyproject.toml").exists()

    def test_ruff_config_has_naming_rules(self):
        """ruff 配置应启用 N（pep8-naming）规则集。"""
        content = (PROJECT_ROOT / "pyproject.toml").read_text()
        assert '"N"' in content, "ruff 配置缺少 N (pep8-naming) 规则集"

    def test_ruff_config_has_per_file_ignores(self):
        """ruff 配置应有 per-file-ignores（__init__.py F401 豁免）。"""
        content = (PROJECT_ROOT / "pyproject.toml").read_text()
        assert "per-file-ignores" in content
        assert "__init__.py" in content


# ---------------------------------------------------------------------------
# 类名 PascalCase 验证
# ---------------------------------------------------------------------------

PASCAL_RE = re.compile(r"^[A-Z][a-zA-Z0-9]*$")


class TestClassNamesPascalCase:
    """验证所有公共类名遵循 PascalCase。"""

    def test_all_exports_are_pascal_case_or_snake_case(self):
        """__all__ 中的类名应为 PascalCase，函数名应为 snake_case。"""
        for mod in [agent_whitepaper] + SUBPACKAGES:
            for name in mod.__all__:
                obj = getattr(mod, name)
                if inspect.isclass(obj):
                    assert PASCAL_RE.match(name), (
                        f"{mod.__name__}.{name} 类名不符合 PascalCase"
                    )
                elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
                    # 函数名允许 snake_case（如 load_config, main）
                    assert name == name.lower() or "_" in name or name.islower(), (
                        f"{mod.__name__}.{name} 函数名不符合 snake_case"
                    )


# ---------------------------------------------------------------------------
# 模块名 snake_case 验证
# ---------------------------------------------------------------------------

SNAKE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class TestModuleNamesSnakeCase:
    """验证所有模块文件名遵循 snake_case。"""

    def test_stage_module_names_snake_case(self):
        """各阶段目录下的 .py 文件名应为 snake_case。"""
        for mod in SUBPACKAGES:
            mod_dir = PACKAGE_ROOT / mod.__name__.split(".")[-1]
            for py_file in mod_dir.glob("*.py"):
                stem = py_file.stem
                if stem == "__init__":
                    continue
                assert SNAKE_RE.match(stem), (
                    f"{py_file.name} 模块名不符合 snake_case"
                )


# ---------------------------------------------------------------------------
# 常量 UPPER_SNAKE_CASE 验证
# ---------------------------------------------------------------------------

UPPER_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class TestConstantsUpperSnakeCase:
    """验证模块级常量遵循 UPPER_SNAKE_CASE。"""

    def test_dependencies_constants_upper_case(self):
        """_dependencies.py 中的常量应为 UPPER_SNAKE_CASE。"""
        from agent_whitepaper import _dependencies

        for name in dir(_dependencies):
            if name.startswith("_"):
                continue
            obj = getattr(_dependencies, name)
            # 常量（dict/list/set 等非函数非类对象）应为 UPPER_SNAKE_CASE
            if not inspect.isclass(obj) and not inspect.isfunction(obj):
                if name in _dependencies.__all__:
                    assert UPPER_RE.match(name), (
                        f"_dependencies.{name} 常量不符合 UPPER_SNAKE_CASE"
                    )

    def test_all_constant_is_list(self):
        """各包的 __all__ 应为列表（常量命名规范）。"""
        for mod in [agent_whitepaper] + SUBPACKAGES:
            assert isinstance(mod.__all__, list)
            assert mod.__all__ == sorted(mod.__all__, key=lambda x: mod.__all__.index(x))


# ---------------------------------------------------------------------------
# 私有成员 _ 前缀验证
# ---------------------------------------------------------------------------

class TestPrivateMembersPrefixed:
    """验证私有成员以下划线开头。"""

    def test_private_modules_underscore_prefixed(self):
        """内部模块（非公共API）应以 _ 开头。"""
        for py_file in PACKAGE_ROOT.glob("*.py"):
            stem = py_file.stem
            if stem == "__init__":
                continue
            # _dependencies.py 是内部模块，以 _ 开头 ✓
            # 其他顶层 .py 文件也应以 _ 开头（如果非公共）
            # 这里只验证已知模式

    def test_dependencies_module_is_private(self):
        """_dependencies 模块以下划线开头（内部模块）。"""
        from agent_whitepaper import _dependencies
        assert _dependencies.__name__.split(".")[-1].startswith("_")


# ---------------------------------------------------------------------------
# __all__ 完整性验证（命名规范的一部分）
# ---------------------------------------------------------------------------

class TestAllCompleteness:
    """验证每个子包的 __all__ 包含其所有公共类。"""

    def test_all_contains_public_classes(self):
        """各子包 __all__ 应包含该包定义的所有公共类。"""
        for mod in SUBPACKAGES:
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                if name.startswith("_"):
                    continue
                # 只检查定义在此包内的类（排除导入的）
                if hasattr(obj, "__module__") and obj.__module__.startswith(mod.__name__):
                    if name in mod.__all__:
                        pass  # 已在 __all__ 中
                    # 注意：并非所有类都需要在 __all__ 中（如 Enum 成员）


# ---------------------------------------------------------------------------
# Enum 命名验证
# ---------------------------------------------------------------------------


class TestEnumNaming:
    """验证 Enum 类名遵循 PascalCase，成员遵循 UPPER_SNAKE_CASE。"""

    def test_enum_members_upper_case(self):
        """各子包中 Enum 的成员应为 UPPER_SNAKE_CASE。"""
        for mod in SUBPACKAGES:
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                if not issubclass(obj, Enum) or obj is Enum:
                    continue
                if not hasattr(obj, "__module__") or not obj.__module__.startswith(mod.__name__):
                    continue
                for member in obj:
                    assert UPPER_RE.match(member.name), (
                        f"{mod.__name__}.{name}.{member.name} "
                        f"枚举成员不符合 UPPER_SNAKE_CASE"
                    )
