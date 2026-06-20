"""阶段P测试：验证文档秩序化。

验证内容：
1. README.md 存在且描述当前A-K系统（非旧100阶段系统）
2. README.md 包含所有11个阶段的描述
3. README.md 的公共API示例与实际 __all__ 一致
4. README.md 的依赖图与 _dependencies.py 声明一致
5. archive/README.md 存在且说明归档原因
6. pyproject.toml 存在且包含 ruff 配置
7. 各子包 __init__.py 的 docstring 作为模块文档
"""

from pathlib import Path

import agent_whitepaper
from agent_whitepaper._dependencies import topological_order

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PACKAGE_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# README.md 验证
# ---------------------------------------------------------------------------

class TestReadmeExists:
    """验证 README.md 存在且描述当前系统。"""

    def test_readme_exists(self):
        assert (PROJECT_ROOT / "README.md").exists()

    def test_readme_describes_current_system(self):
        """README 应描述A-K系统，而非旧的100阶段系统。"""
        content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        # 应提及A-K阶段
        for stage in ["stage_a", "stage_k", "阶段A", "阶段K"]:
            assert stage in content, f"README 缺少对 {stage} 的描述"
        # 不应以旧的 core/ 为主标题
        assert "core/stage_050" not in content
        assert "core/stage_011" not in content

    def test_readme_has_quick_start(self):
        content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        assert "快速开始" in content or "Quick" in content
        assert "pip install" in content

    def test_readme_has_architecture(self):
        content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        assert "架构" in content or "Architecture" in content

    def test_readme_has_testing_section(self):
        content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        assert "测试" in content or "Test" in content
        assert "pytest" in content


class TestReadmeMentionsAllStages:
    """验证 README 提及所有11个阶段。"""

    def test_all_stages_mentioned(self):
        content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        for stage in [
            "stage_a", "stage_b", "stage_c", "stage_d", "stage_e",
            "stage_f", "stage_g", "stage_h", "stage_i", "stage_j",
            "stage_k",
        ]:
            assert stage in content, f"README 未提及 {stage}"


class TestReadmeApiConsistency:
    """验证 README 中的公共API示例与实际 __all__ 一致。"""

    def test_readme_api_examples_match_all(self):
        """README 中导出的类名应存在于顶层 __all__。"""
        content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        for name in agent_whitepaper.__all__:
            assert name in content, (
                f"README 未提及公共API: {name}"
            )


class TestReadmeDependencyConsistency:
    """验证 README 中的依赖图与 _dependencies.py 一致。"""

    def test_readme_mentions_dependencies(self):
        content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        assert "_dependencies" in content or "依赖" in content

    def test_readme_topological_order_matches(self):
        """README 中的拓扑排序应与实际一致。"""
        content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        actual_order = topological_order()
        # README 应包含所有阶段名（拓扑排序的元素）
        for stage in actual_order:
            assert stage in content


# ---------------------------------------------------------------------------
# archive/README.md 验证
# ---------------------------------------------------------------------------

class TestArchiveReadme:
    """验证归档目录文档。"""

    def test_archive_readme_exists(self):
        assert (PROJECT_ROOT / "archive" / "README.md").exists()

    def test_archive_readme_explains_reason(self):
        content = (PROJECT_ROOT / "archive" / "README.md").read_text(encoding="utf-8")
        assert "归档" in content
        assert "历史" in content or "legacy" in content.lower()

    def test_archive_readme_mentions_main_system(self):
        """归档README应说明与主系统(stage_a~k)的关系。"""
        content = (PROJECT_ROOT / "archive" / "README.md").read_text(encoding="utf-8")
        assert "stage_a" in content


# ---------------------------------------------------------------------------
# pyproject.toml 验证
# ---------------------------------------------------------------------------

class TestPyprojectExists:
    """验证 pyproject.toml 存在且配置完整。"""

    def test_pyproject_exists(self):
        assert (PROJECT_ROOT / "pyproject.toml").exists()

    def test_pyproject_has_ruff_config(self):
        content = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "[tool.ruff]" in content
        assert "pep8-naming" in content or '"N"' in content

    def test_pyproject_has_pytest_config(self):
        content = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "[tool.pytest" in content

    def test_pyproject_has_project_metadata(self):
        content = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "[project]" in content
        assert "agent-whitepaper" in content


# ---------------------------------------------------------------------------
# 模块文档验证
# ---------------------------------------------------------------------------

class TestModuleDocstrings:
    """验证各子包有描述性 docstring 作为模块文档。"""

    def test_top_level_docstring_describes_system(self):
        """顶层包 docstring 应描述A-K系统。"""
        doc = agent_whitepaper.__doc__
        assert doc is not None
        assert "A" in doc and "K" in doc
        assert "阶段" in doc or "Stage" in doc or "stage" in doc

    def test_each_stage_has_descriptive_docstring(self):
        """各子包 docstring 应包含公共API说明。"""
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
        for mod in [
            stage_a, stage_b, stage_c, stage_d, stage_e,
            stage_f, stage_g, stage_h, stage_i, stage_j, stage_k,
        ]:
            doc = mod.__doc__
            assert doc is not None, f"{mod.__name__} 缺少 docstring"
            assert "公共API" in doc or "API" in doc, (
                f"{mod.__name__} docstring 缺少公共API说明"
            )

    def test_dependencies_module_has_docstring(self):
        """_dependencies.py 应有描述依赖关系的 docstring。"""
        from agent_whitepaper import _dependencies
        doc = _dependencies.__doc__
        assert doc is not None
        assert "依赖" in doc or "dependency" in doc.lower()


# ---------------------------------------------------------------------------
# 文档完整性验证
# ---------------------------------------------------------------------------

class TestDocumentationCompleteness:
    """验证文档覆盖系统的关键方面。"""

    def test_readme_covers_installation(self):
        content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        assert "pip install" in content

    def test_readme_covers_cli_usage(self):
        content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        assert "agentwp" in content or "CLI" in content or "cli" in content

    def test_readme_covers_project_structure(self):
        content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        assert "项目结构" in content or "Structure" in content or "structure" in content

    def test_readme_covers_license(self):
        content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        assert "MIT" in content or "LICENSE" in content or "许可证" in content

    def test_readme_covers_archive(self):
        """README 应提及归档目录。"""
        content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        assert "archive" in content
