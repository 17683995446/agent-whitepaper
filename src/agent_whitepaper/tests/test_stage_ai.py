"""阶段AI测试：验证插件系统。

验证内容：
1. Plugin 插件基类
2. PluginContext 插件事件上下文
3. PluginManager 插件管理器
4. LoggingPlugin 日志插件
5. MetricsExportPlugin 指标导出插件
6. list_builtin_plugins 便捷函数
7. create_plugin_manager 便捷函数
8. CLI集成（plugins子命令、cmd_plugins）
9. 源码结构符合规范
"""

import argparse
import inspect
import os
from pathlib import Path

from agent_whitepaper.stage_h.plugins import (
    BUILTIN_PLUGINS,
    LoggingPlugin,
    MetricsExportPlugin,
    Plugin,
    PluginContext,
    PluginManager,
    create_plugin_manager,
    list_builtin_plugins,
)


# ---------------------------------------------------------------------------
# Plugin 基类验证
# ---------------------------------------------------------------------------
class TestPluginBase:
    """验证 Plugin 基类。"""

    def test_plugin_class_exists(self):
        """Plugin 类存在且可调用。"""
        assert callable(Plugin)

    def test_plugin_has_docstring(self):
        """Plugin 有docstring。"""
        assert Plugin.__doc__ is not None
        assert "扩展" in Plugin.__doc__ or "插件" in Plugin.__doc__

    def test_plugin_can_be_subclassed(self):
        """Plugin 可被继承。"""

        class MyPlugin(Plugin):
            name = "my"
            version = "1.0.0"
            description = "测试插件"

        p = MyPlugin()
        assert p.name == "my"
        assert p.version == "1.0.0"

    def test_plugin_hooks_exist(self):
        """Plugin 有所有生命周期钩子。"""
        for hook in ["on_init", "on_iteration", "on_metrics", "on_error", "on_stop"]:
            assert hasattr(Plugin, hook)
            assert callable(getattr(Plugin, hook))

    def test_plugin_hooks_default_noop(self):
        """默认钩子是空操作（不抛异常）。"""
        p = Plugin()
        ctx = PluginContext()
        # 所有钩子默认应不抛异常
        p.on_init(ctx)
        p.on_iteration(ctx)
        p.on_metrics(ctx)
        p.on_error(ctx)
        p.on_stop(ctx)

    def test_plugin_repr(self):
        """Plugin 有 __repr__。"""
        p = Plugin()
        repr_str = repr(p)
        assert "Plugin" in repr_str


# ---------------------------------------------------------------------------
# PluginContext 验证
# ---------------------------------------------------------------------------
class TestPluginContext:
    """验证 PluginContext 数据类。"""

    def test_context_creation_default(self):
        """默认创建上下文。"""
        ctx = PluginContext()
        assert ctx.event == ""
        assert ctx.orchestrator is None
        assert ctx.iteration is None
        assert ctx.snapshot is None
        assert ctx.error is None
        assert ctx.reason is None
        assert ctx.report is None
        assert ctx.extra == {}

    def test_context_creation_with_data(self):
        """带数据创建上下文。"""
        ctx = PluginContext(
            event="on_iteration",
            iteration=5,
            snapshot={"avg_success_rate": 0.5},
        )
        assert ctx.event == "on_iteration"
        assert ctx.iteration == 5
        assert ctx.snapshot == {"avg_success_rate": 0.5}

    def test_context_extra_is_independent(self):
        """extra 字段各实例独立（可变默认值安全）。"""
        ctx1 = PluginContext()
        ctx2 = PluginContext()
        ctx1.extra["key"] = "value"
        assert "key" not in ctx2.extra


# ---------------------------------------------------------------------------
# PluginManager 验证
# ---------------------------------------------------------------------------
class TestPluginManager:
    """验证 PluginManager。"""

    def test_manager_creation(self):
        """创建管理器。"""
        mgr = PluginManager()
        assert mgr is not None
        assert len(mgr) == 0

    def test_register_plugin(self):
        """注册插件。"""
        mgr = PluginManager()
        plugin = LoggingPlugin()
        mgr.register(plugin)
        assert len(mgr) == 1
        assert "logging" in mgr

    def test_register_duplicate_raises(self):
        """重复注册同名插件抛出异常。"""
        mgr = PluginManager()
        mgr.register(LoggingPlugin())
        try:
            mgr.register(LoggingPlugin())
            assert False, "应抛出 ValueError"
        except ValueError:
            pass

    def test_register_empty_name_raises(self):
        """注册空名插件抛出异常。"""

        class EmptyPlugin(Plugin):
            name = ""

        mgr = PluginManager()
        try:
            mgr.register(EmptyPlugin())
            assert False, "应抛出 ValueError"
        except ValueError:
            pass

    def test_unregister_plugin(self):
        """注销插件。"""
        mgr = PluginManager()
        mgr.register(LoggingPlugin())
        assert mgr.unregister("logging") is True
        assert len(mgr) == 0
        assert "logging" not in mgr

    def test_unregister_nonexistent(self):
        """注销不存在的插件返回False。"""
        mgr = PluginManager()
        assert mgr.unregister("nonexistent") is False

    def test_get_plugin(self):
        """获取插件实例。"""
        mgr = PluginManager()
        plugin = LoggingPlugin()
        mgr.register(plugin)
        assert mgr.get("logging") is plugin
        assert mgr.get("nonexistent") is None

    def test_list_plugins(self):
        """列出所有插件。"""
        mgr = PluginManager()
        mgr.register(LoggingPlugin())
        mgr.register(MetricsExportPlugin())
        plugins = mgr.list_plugins()
        assert len(plugins) == 2

    def test_enable_disable(self):
        """启用/禁用插件。"""
        mgr = PluginManager()
        mgr.register(LoggingPlugin())
        assert mgr.is_enabled("logging") is True
        assert mgr.disable("logging") is True
        assert mgr.is_enabled("logging") is False
        assert mgr.enable("logging") is True
        assert mgr.is_enabled("logging") is True

    def test_enable_nonexistent(self):
        """启用不存在的插件返回False。"""
        mgr = PluginManager()
        assert mgr.enable("nonexistent") is False
        assert mgr.disable("nonexistent") is False

    def test_list_enabled(self):
        """列出已启用插件。"""
        mgr = PluginManager()
        mgr.register(LoggingPlugin())
        mgr.register(MetricsExportPlugin())
        mgr.disable("metrics-export")
        enabled = mgr.list_enabled()
        assert len(enabled) == 1
        assert enabled[0].name == "logging"

    def test_emit_calls_enabled_hooks(self):
        """emit 调用已启用插件的钩子。"""
        mgr = PluginManager()

        call_count = [0]

        class CountingPlugin(Plugin):
            name = "counting"
            version = "1.0.0"
            description = "计数插件"

            def on_init(self, ctx):
                call_count[0] += 1

        mgr.register(CountingPlugin())
        mgr.emit("on_init", orchestrator=None)
        assert call_count[0] == 1

    def test_emit_skips_disabled(self):
        """emit 跳过已禁用插件。"""
        mgr = PluginManager()

        call_count = [0]

        class CountingPlugin(Plugin):
            name = "counting"
            version = "1.0.0"
            description = "计数插件"

            def on_iteration(self, ctx):
                call_count[0] += 1

        mgr.register(CountingPlugin())
        mgr.disable("counting")
        mgr.emit("on_iteration", iteration=1)
        assert call_count[0] == 0

    def test_emit_swallows_plugin_errors(self):
        """emit 吞掉插件异常（防火墙层）。"""
        mgr = PluginManager()

        class BadPlugin(Plugin):
            name = "bad"
            version = "1.0.0"
            description = "坏插件"

            def on_init(self, ctx):
                raise RuntimeError("插件崩溃")

        mgr.register(BadPlugin())
        # 不应抛出异常
        mgr.emit("on_init", orchestrator=None)

    def test_emit_counts_events(self):
        """emit 统计事件次数。"""
        mgr = PluginManager()
        mgr.register(LoggingPlugin())
        mgr.emit("on_init", orchestrator=None)
        mgr.emit("on_iteration", iteration=1)
        mgr.emit("on_iteration", iteration=2)
        counts = mgr.event_counts
        assert counts.get("on_init") == 1
        assert counts.get("on_iteration") == 2

    def test_emit_passes_context(self):
        """emit 传递上下文给钩子。"""
        mgr = PluginManager()
        received = []

        class CapturePlugin(Plugin):
            name = "capture"
            version = "1.0.0"
            description = "捕获插件"

            def on_iteration(self, ctx):
                received.append((ctx.iteration, ctx.event))

        mgr.register(CapturePlugin())
        mgr.emit("on_iteration", iteration=42)
        assert received == [(42, "on_iteration")]

    def test_clear(self):
        """clear 清空所有插件。"""
        mgr = PluginManager()
        mgr.register(LoggingPlugin())
        mgr.emit("on_init", orchestrator=None)
        mgr.clear()
        assert len(mgr) == 0
        assert mgr.event_counts == {}

    def test_contains(self):
        """__contains__ 工作。"""
        mgr = PluginManager()
        mgr.register(LoggingPlugin())
        assert "logging" in mgr
        assert "nonexistent" not in mgr


# ---------------------------------------------------------------------------
# LoggingPlugin 验证
# ---------------------------------------------------------------------------
class TestLoggingPlugin:
    """验证日志插件。"""

    def test_logging_plugin_attributes(self):
        """日志插件属性正确。"""
        p = LoggingPlugin()
        assert p.name == "logging"
        assert p.version == "1.0.0"
        assert "日志" in p.description or "记录" in p.description

    def test_logging_plugin_hooks_no_error(self):
        """日志插件钩子不抛异常。"""
        p = LoggingPlugin()
        ctx = PluginContext(
            event="on_iteration",
            iteration=1,
            error=RuntimeError("test"),
            reason="done",
        )
        p.on_init(ctx)
        p.on_iteration(ctx)
        p.on_metrics(ctx)
        p.on_error(ctx)
        p.on_stop(ctx)

    def test_logging_plugin_custom_level(self):
        """日志插件支持自定义级别。"""
        p = LoggingPlugin(level="info")
        assert p._level == "info"

    def test_logging_plugin_in_builtin(self):
        """日志插件在内置注册表中。"""
        assert "logging" in BUILTIN_PLUGINS
        assert BUILTIN_PLUGINS["logging"] is LoggingPlugin


# ---------------------------------------------------------------------------
# MetricsExportPlugin 验证
# ---------------------------------------------------------------------------
class TestMetricsExportPlugin:
    """验证指标导出插件。"""

    def test_metrics_plugin_attributes(self):
        """指标插件属性正确。"""
        p = MetricsExportPlugin()
        assert p.name == "metrics-export"
        assert p.version == "1.0.0"
        assert "导出" in p.description or "指标" in p.description

    def test_metrics_plugin_export_path(self):
        """指标插件有导出路径。"""
        p = MetricsExportPlugin(path="test.jsonl")
        assert p.export_path == "test.jsonl"

    def test_metrics_plugin_writes_file(self, tmp_path):
        """指标插件写入JSONL文件。"""
        path = str(tmp_path / "metrics.jsonl")
        p = MetricsExportPlugin(path=path)
        ctx = PluginContext(
            event="on_iteration",
            iteration=1,
            snapshot={"avg_success_rate": 0.5},
        )
        p.on_iteration(ctx)
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "avg_success_rate" in content
        assert p.export_count == 1

    def test_metrics_plugin_no_snapshot_skips(self, tmp_path):
        """无快照时跳过导出。"""
        path = str(tmp_path / "metrics.jsonl")
        p = MetricsExportPlugin(path=path)
        ctx = PluginContext(event="on_iteration", iteration=1, snapshot=None)
        p.on_iteration(ctx)
        assert p.export_count == 0

    def test_metrics_plugin_multiple_iterations(self, tmp_path):
        """多轮迭代追加写入。"""
        path = str(tmp_path / "metrics.jsonl")
        p = MetricsExportPlugin(path=path)
        for i in range(3):
            ctx = PluginContext(
                event="on_iteration",
                iteration=i,
                snapshot={"rate": 0.1 * i},
            )
            p.on_iteration(ctx)
        assert p.export_count == 3
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_metrics_plugin_in_builtin(self):
        """指标插件在内置注册表中。"""
        assert "metrics-export" in BUILTIN_PLUGINS
        assert BUILTIN_PLUGINS["metrics-export"] is MetricsExportPlugin


# ---------------------------------------------------------------------------
# 便捷函数验证
# ---------------------------------------------------------------------------
class TestConvenienceFunctions:
    """验证便捷函数。"""

    def test_list_builtin_plugins(self):
        """list_builtin_plugins 返回字典。"""
        plugins = list_builtin_plugins()
        assert isinstance(plugins, dict)
        assert "logging" in plugins
        assert "metrics-export" in plugins

    def test_create_plugin_manager_default(self):
        """默认创建包含所有内置插件的管理器。"""
        mgr = create_plugin_manager()
        assert len(mgr) == 2
        assert "logging" in mgr
        assert "metrics-export" in mgr

    def test_create_plugin_manager_specific(self):
        """指定插件创建管理器。"""
        mgr = create_plugin_manager(enabled=["logging"])
        assert len(mgr) == 1
        assert "logging" in mgr
        assert "metrics-export" not in mgr

    def test_create_plugin_manager_unknown_raises(self):
        """未知插件名抛出异常。"""
        try:
            create_plugin_manager(enabled=["unknown"])
            assert False, "应抛出 ValueError"
        except ValueError:
            pass

    def test_create_plugin_manager_empty(self):
        """空列表创建空管理器。"""
        mgr = create_plugin_manager(enabled=[])
        assert len(mgr) == 0


# ---------------------------------------------------------------------------
# 端到端集成验证
# ---------------------------------------------------------------------------
class TestEndToEndIntegration:
    """端到端集成验证。"""

    def test_plugin_manager_with_real_orchestrator(self, tmp_path):
        """插件管理器与真实编排器协同。"""
        from agent_whitepaper.stage_g.orchestrator import SystemOrchestrator
        from agent_whitepaper.stage_h.config import load_config

        config = load_config(
            n_agents=2,
            grid_size=3,
            db_path=":memory:",
            max_iterations=3,
            seed=42,
        )
        orch = SystemOrchestrator(config)
        orch.initialize()

        # 创建管理器并注册指标导出插件
        path = str(tmp_path / "e2e_metrics.jsonl")
        mgr = PluginManager()
        mgr.register(MetricsExportPlugin(path=path))

        # 模拟生命周期
        mgr.emit("on_init", orchestrator=orch)
        for i in range(3):
            orch.run_iteration()
            snap = orch.metrics.get_latest()
            mgr.emit("on_iteration", iteration=i + 1, snapshot=snap.to_dict() if snap else None)

        report = orch.get_metrics()
        mgr.emit("on_metrics", report=report)
        mgr.emit("on_stop", reason="completed")
        orch.stop("test")

        # 验证指标导出
        assert os.path.exists(path)
        assert mgr.event_counts.get("on_iteration") == 3
        assert mgr.event_counts.get("on_stop") == 1


# ---------------------------------------------------------------------------
# CLI 集成验证
# ---------------------------------------------------------------------------
class TestCliIntegration:
    """验证CLI集成。"""

    def test_plugins_subcommand_registered(self):
        """plugins子命令已注册。"""
        from agent_whitepaper.stage_h.cli import build_parser

        parser = build_parser()
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                parser.parse_args(["plugins", "--help"])
            except SystemExit:
                pass
        output = buf.getvalue()
        assert "plugins" in output

    def test_cmd_plugins_function_exists(self):
        """cmd_plugins 函数存在。"""
        from agent_whitepaper.stage_h.cli import cmd_plugins
        assert callable(cmd_plugins)

    def test_cmd_plugins_has_docstring(self):
        """cmd_plugins 有docstring。"""
        from agent_whitepaper.stage_h.cli import cmd_plugins
        assert cmd_plugins.__doc__ is not None

    def test_cmd_plugins_list(self):
        """cmd_plugins list 列出插件。"""
        import contextlib
        import io

        from agent_whitepaper.stage_h.cli import cmd_plugins

        args = argparse.Namespace(action="list", name=None)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = cmd_plugins(args)
        assert result == 0
        output = buf.getvalue()
        assert "logging" in output
        assert "metrics-export" in output

    def test_cmd_plugins_info(self):
        """cmd_plugins info 显示插件详情。"""
        import contextlib
        import io

        from agent_whitepaper.stage_h.cli import cmd_plugins

        args = argparse.Namespace(action="info", name="logging")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = cmd_plugins(args)
        assert result == 0
        output = buf.getvalue()
        assert "logging" in output
        assert "钩子" in output

    def test_cmd_plugins_info_unknown(self):
        """cmd_plugins info 未知插件返回1。"""
        import contextlib
        import io

        from agent_whitepaper.stage_h.cli import cmd_plugins

        args = argparse.Namespace(action="info", name="nonexistent")
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            result = cmd_plugins(args)
        assert result == 1

    def test_plugins_command_in_doctor_subcommands(self):
        """doctor命令检查的子命令列表包含plugins。"""
        import contextlib
        import io

        from agent_whitepaper.stage_h.cli import cmd_doctor

        buf = io.StringIO()
        args = argparse.Namespace()
        with contextlib.redirect_stdout(buf):
            cmd_doctor(args)
        output = buf.getvalue()
        assert "agentwp plugins" in output


# ---------------------------------------------------------------------------
# 源码结构验证
# ---------------------------------------------------------------------------
class TestPluginsSourceCode:
    """验证插件系统源码结构。"""

    def test_plugins_module_exists(self):
        """plugins.py 模块存在。"""
        from agent_whitepaper.stage_h import plugins
        assert plugins is not None

    def test_plugin_class_exported_from_stage_h(self):
        """Plugin 从 stage_h 包导出。"""
        import agent_whitepaper.stage_h as stage_h
        assert stage_h.Plugin is Plugin

    def test_plugin_manager_exported(self):
        """PluginManager 从 stage_h 包导出。"""
        import agent_whitepaper.stage_h as stage_h
        assert stage_h.PluginManager is PluginManager

    def test_plugin_context_exported(self):
        """PluginContext 从 stage_h 包导出。"""
        import agent_whitepaper.stage_h as stage_h
        assert stage_h.PluginContext is PluginContext

    def test_logging_plugin_exported(self):
        """LoggingPlugin 从 stage_h 包导出。"""
        import agent_whitepaper.stage_h as stage_h
        assert stage_h.LoggingPlugin is LoggingPlugin

    def test_metrics_export_plugin_exported(self):
        """MetricsExportPlugin 从 stage_h 包导出。"""
        import agent_whitepaper.stage_h as stage_h
        assert stage_h.MetricsExportPlugin is MetricsExportPlugin

    def test_list_builtin_plugins_exported(self):
        """list_builtin_plugins 从 stage_h 包导出。"""
        import agent_whitepaper.stage_h as stage_h
        assert stage_h.list_builtin_plugins is list_builtin_plugins

    def test_create_plugin_manager_exported(self):
        """create_plugin_manager 从 stage_h 包导出。"""
        import agent_whitepaper.stage_h as stage_h
        assert stage_h.create_plugin_manager is create_plugin_manager

    def test_module_has_first_principle_docstring(self):
        """模块docstring包含第一性原理。"""
        source = Path(
            inspect.getfile(Plugin)
        ).read_text(encoding="utf-8")
        assert "第一性原理" in source
        assert "Extensibility" in source or "扩展" in source

    def test_plugin_class_has_first_principle(self):
        """Plugin类docstring包含第一性原理。"""
        source = Path(
            inspect.getfile(Plugin)
        ).read_text(encoding="utf-8")
        assert "第一性原理" in source

    def test_all_public_methods_have_type_hints(self):
        """所有公共方法有返回类型注解。"""
        for method_name in ["on_init", "on_iteration", "on_metrics", "on_error", "on_stop"]:
            sig = inspect.signature(getattr(Plugin, method_name))
            assert sig.return_annotation is not inspect.Signature.empty, (
                f"{method_name} 缺少返回类型注解"
            )

    def test_manager_methods_have_type_hints(self):
        """PluginManager 公共方法有返回类型注解。"""
        for method_name in ["register", "unregister", "get", "list_plugins", "list_enabled",
                            "is_enabled", "enable", "disable", "emit", "clear"]:
            sig = inspect.signature(getattr(PluginManager, method_name))
            assert sig.return_annotation is not inspect.Signature.empty, (
                f"{method_name} 缺少返回类型注解"
            )

    def test_convenience_functions_have_type_hints(self):
        """便捷函数有返回类型注解。"""
        sig = inspect.signature(list_builtin_plugins)
        assert sig.return_annotation is not inspect.Signature.empty

        sig = inspect.signature(create_plugin_manager)
        assert sig.return_annotation is not inspect.Signature.empty

    def test_no_silent_except_exception_pass(self):
        """无静默 except Exception: pass。"""
        import ast
        source = Path(
            inspect.getfile(Plugin)
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if not (isinstance(node.type, ast.Name) and node.type.id == "Exception"):
                continue
            body = node.body
            if len(body) == 1 and isinstance(body[0], ast.Pass):
                offenders.append(node.lineno)
        assert offenders == [], f"发现静默 except Exception: pass: {offenders}"

    def test_init_docstring_mentions_plugins(self):
        """stage_h __init__.py docstring提及插件。"""
        import agent_whitepaper.stage_h as stage_h
        init_source = Path(
            inspect.getfile(stage_h)
        ).read_text(encoding="utf-8")
        assert "Plugin" in init_source
        assert "PluginManager" in init_source
