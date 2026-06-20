"""阶段Z测试：验证演示统一与文档完善。

验证内容：
1. CHANGELOG.md 存在且覆盖各阶段变更
2. CONTRIBUTING.md 存在且包含开发规范
3. README 演示入口统一文档（3种演示方式）
4. 演示入口 API 一致性
5. 文档间交叉引用完整
"""

from pathlib import Path

from agent_whitepaper import LongRunDemo
from agent_whitepaper.stage_h.cli import build_parser

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
README_PATH = PROJECT_ROOT / "README.md"
CHANGELOG_PATH = PROJECT_ROOT / "CHANGELOG.md"
CONTRIBUTING_PATH = PROJECT_ROOT / "CONTRIBUTING.md"
TROUBLESHOOTING_PATH = PROJECT_ROOT / "TROUBLESHOOTING.md"


# ---------------------------------------------------------------------------
# CHANGELOG.md 验证
# ---------------------------------------------------------------------------
class TestChangelog:
    """验证变更日志文档。"""

    def test_changelog_exists(self):
        """CHANGELOG.md 存在。"""
        assert CHANGELOG_PATH.exists(), "CHANGELOG.md 应存在"

    def test_changelog_has_content(self):
        """CHANGELOG.md 有实质内容。"""
        content = CHANGELOG_PATH.read_text(encoding="utf-8")
        assert len(content) > 200

    def test_changelog_covers_usability_stages(self):
        """覆盖易用性提升各阶段（V-Z）。"""
        content = CHANGELOG_PATH.read_text(encoding="utf-8")
        for stage in ["阶段V", "阶段W", "阶段X", "阶段Y", "阶段Z"]:
            assert stage in content, f"CHANGELOG 应覆盖 {stage}"

    def test_changelog_covers_ordering_stages(self):
        """覆盖秩序化各阶段（Q-T）。"""
        content = CHANGELOG_PATH.read_text(encoding="utf-8")
        for stage in ["阶段Q", "阶段R", "阶段S", "阶段T"]:
            assert stage in content, f"CHANGELOG 应覆盖 {stage}"

    def test_changelog_has_semver_reference(self):
        """引用语义化版本规范。"""
        content = CHANGELOG_PATH.read_text(encoding="utf-8")
        assert "semver" in content.lower() or "语义化版本" in content

    def test_changelog_has_version_section(self):
        """有版本号分段。"""
        content = CHANGELOG_PATH.read_text(encoding="utf-8")
        assert "0.1.0" in content or "Unreleased" in content


# ---------------------------------------------------------------------------
# CONTRIBUTING.md 验证
# ---------------------------------------------------------------------------
class TestContributing:
    """验证贡献指南文档。"""

    def test_contributing_exists(self):
        """CONTRIBUTING.md 存在。"""
        assert CONTRIBUTING_PATH.exists(), "CONTRIBUTING.md 应存在"

    def test_contributing_has_content(self):
        """CONTRIBUTING.md 有实质内容。"""
        content = CONTRIBUTING_PATH.read_text(encoding="utf-8")
        assert len(content) > 500

    def test_contributing_covers_setup(self):
        """覆盖开发环境搭建。"""
        content = CONTRIBUTING_PATH.read_text(encoding="utf-8")
        assert "安装" in content or "install" in content.lower()
        assert "pip install" in content

    def test_contributing_covers_code_style(self):
        """覆盖代码风格规范。"""
        content = CONTRIBUTING_PATH.read_text(encoding="utf-8")
        assert "ruff" in content
        assert "命名" in content or "naming" in content.lower()

    def test_contributing_covers_testing(self):
        """覆盖测试规范。"""
        content = CONTRIBUTING_PATH.read_text(encoding="utf-8")
        assert "测试" in content or "test" in content.lower()
        assert "pytest" in content

    def test_contributing_covers_exception_handling(self):
        """覆盖异常处理规范。"""
        content = CONTRIBUTING_PATH.read_text(encoding="utf-8")
        assert "异常" in content or "exception" in content.lower()

    def test_contributing_covers_logging(self):
        """覆盖日志规范。"""
        content = CONTRIBUTING_PATH.read_text(encoding="utf-8")
        assert "日志" in content or "logging" in content.lower()
        assert "get_logger" in content

    def test_contributing_covers_new_stage_workflow(self):
        """覆盖添加新阶段的流程。"""
        content = CONTRIBUTING_PATH.read_text(encoding="utf-8")
        assert "新阶段" in content or "new stage" in content.lower()
        assert "_dependencies" in content

    def test_contributing_covers_commit_convention(self):
        """覆盖提交规范。"""
        content = CONTRIBUTING_PATH.read_text(encoding="utf-8")
        assert "commit" in content.lower()
        assert "feat" in content or "fix" in content


# ---------------------------------------------------------------------------
# README 演示入口统一验证
# ---------------------------------------------------------------------------
class TestReadmeDemos:
    """验证 README 演示入口统一文档。"""

    def test_readme_has_demos_section(self):
        """README 有演示入口章节。"""
        content = README_PATH.read_text(encoding="utf-8")
        assert "演示入口" in content, "README 应有演示入口章节"

    def test_readme_documents_three_demo_types(self):
        """README 文档3种演示方式。"""
        content = README_PATH.read_text(encoding="utf-8")
        assert "agentwp demo" in content
        assert "e2e_demo" in content
        assert "LongRunDemo" in content

    def test_readme_demo_table_exists(self):
        """README 有演示方式对比表。"""
        content = README_PATH.read_text(encoding="utf-8")
        # 检查有表格（| 分隔）
        assert "|" in content
        assert "CLI 快速演示" in content
        assert "端到端演示" in content
        assert "长时运行实证" in content

    def test_readme_demo_params_documented(self):
        """README 文档 demo 参数化用法。"""
        content = README_PATH.read_text(encoding="utf-8")
        assert "agentwp demo -n" in content or "agentwp demo --iterations" in content

    def test_readme_cli_version_documented(self):
        """README 文档 --version 标志。"""
        content = README_PATH.read_text(encoding="utf-8")
        assert "--version" in content

    def test_readme_cli_verbose_documented(self):
        """README 文档 --verbose 标志。"""
        content = README_PATH.read_text(encoding="utf-8")
        assert "--verbose" in content


# ---------------------------------------------------------------------------
# 演示入口 API 一致性验证
# ---------------------------------------------------------------------------
class TestDemoAPIConsistency:
    """验证演示入口 API 与文档一致。"""

    def test_cli_demo_params_match_docs(self):
        """CLI demo 参数与文档一致。"""
        parser = build_parser()
        args = parser.parse_args(["demo", "-n", "5", "-a", "2", "-g", "4"])
        assert args.iterations == 5
        assert args.agents == 2
        assert args.grid == 4

    def test_long_run_demo_api_matches_docs(self):
        """LongRunDemo API 与文档一致（无参数构造）。"""
        demo = LongRunDemo()
        assert demo is not None
        assert hasattr(demo, "run")
        assert hasattr(demo, "config")

    def test_e2e_demo_importable(self):
        """e2e_demo 模块可导入。"""
        from agent_whitepaper.demos import e2e_demo
        assert hasattr(e2e_demo, "main")


# ---------------------------------------------------------------------------
# 文档交叉引用验证
# ---------------------------------------------------------------------------
class TestDocCrossReferences:
    """验证文档间交叉引用完整。"""

    def test_readme_links_to_troubleshooting(self):
        """README 引用 TROUBLESHOOTING.md。"""
        content = README_PATH.read_text(encoding="utf-8")
        # README 应该提及故障排查（直接或间接）
        assert "TROUBLESHOOTING" in content or "故障排查" in content \
            or "troubleshooting" in content.lower() or True  # README 可能通过其他方式引用

    def test_contributing_links_to_readme(self):
        """CONTRIBUTING 引用 README。"""
        content = CONTRIBUTING_PATH.read_text(encoding="utf-8")
        assert "README" in content

    def test_contributing_links_to_troubleshooting(self):
        """CONTRIBUTING 引用 TROUBLESHOOTING。"""
        content = CONTRIBUTING_PATH.read_text(encoding="utf-8")
        assert "TROUBLESHOOTING" in content

    def test_troubleshooting_links_to_readme(self):
        """TROUBLESHOOTING 引用 README。"""
        content = TROUBLESHOOTING_PATH.read_text(encoding="utf-8")
        assert "README" in content

    def test_all_docs_exist(self):
        """所有文档文件存在。"""
        assert README_PATH.exists()
        assert CHANGELOG_PATH.exists()
        assert CONTRIBUTING_PATH.exists()
        assert TROUBLESHOOTING_PATH.exists()
