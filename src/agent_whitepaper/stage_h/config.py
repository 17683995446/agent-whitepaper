"""阶段H-1：配置加载与验证 - 系统的"启动参数"。

第一性原理：
    可部署性是实用性的前提。一个只能通过修改代码来配置的系统
    无法被真正使用。配置化让系统：
    1. 可定制：不同场景用不同配置（网格大小、Agent数、间隔参数）
    2. 可复现：配置文件记录运行参数，结果可复现
    3. 可运维：无需改代码即可调整系统行为
    4. 可环境化：支持环境变量覆盖，适配部署环境

    配置是"声明式"的——描述"想要什么"而非"怎么做"。
"""

import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from agent_whitepaper._exceptions import ConfigError
from agent_whitepaper.stage_g.orchestrator import OrchestratorConfig

# ============================================================
# 默认配置
# ============================================================
DEFAULT_CONFIG: Dict[str, Any] = {
    "grid_size": 5,
    "n_agents": 3,
    "db_path": "agent_system.db",
    "checkpoint_interval": 50,
    "skill_abstract_interval": 5,
    "meta_learn_interval": 10,
    "distill_interval": 30,
    "evolve_interval": 20,
    "coalition_interval": 15,
    "max_iterations": None,
    "seed": 42,
}


# ============================================================
# 配置验证错误
# ============================================================
class ConfigValidationError(ConfigError, ValueError):
    """配置验证失败。

    多重继承保证向后兼容：
    - ConfigError：纳入统一异常层次体系
    - ValueError：保持与既有代码/测试的 isinstance 兼容
    """


# ============================================================
# 配置模板（预置场景配置）
# ============================================================
#: 配置模板注册表：每个模板是一个完整配置字典
CONFIG_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "default": {
        "_description": "默认配置：平衡的功能与性能",
        **DEFAULT_CONFIG,
    },
    "quick-test": {
        "_description": "快速测试：小规模、短迭代，用于验证安装",
        "grid_size": 3,
        "n_agents": 2,
        "db_path": ":memory:",
        "checkpoint_interval": 10,
        "skill_abstract_interval": 3,
        "meta_learn_interval": 5,
        "distill_interval": 10,
        "evolve_interval": 7,
        "coalition_interval": 5,
        "max_iterations": 10,
        "seed": 42,
    },
    "production": {
        "_description": "生产环境：持久化、较大规模、稳定间隔",
        "grid_size": 7,
        "n_agents": 5,
        "db_path": "agent_system.db",
        "checkpoint_interval": 100,
        "skill_abstract_interval": 10,
        "meta_learn_interval": 20,
        "distill_interval": 50,
        "evolve_interval": 30,
        "coalition_interval": 25,
        "max_iterations": None,
        "seed": 42,
    },
    "multi-agent": {
        "_description": "多Agent协作：更多Agent、更紧密的协作间隔",
        "grid_size": 5,
        "n_agents": 6,
        "db_path": "agent_system.db",
        "checkpoint_interval": 50,
        "skill_abstract_interval": 5,
        "meta_learn_interval": 10,
        "distill_interval": 20,
        "evolve_interval": 15,
        "coalition_interval": 5,
        "max_iterations": None,
        "seed": 42,
    },
    "long-run": {
        "_description": "长时运行：大间隔、持久化，适合长时间自主进化",
        "grid_size": 5,
        "n_agents": 3,
        "db_path": "agent_system.db",
        "checkpoint_interval": 200,
        "skill_abstract_interval": 20,
        "meta_learn_interval": 40,
        "distill_interval": 100,
        "evolve_interval": 60,
        "coalition_interval": 50,
        "max_iterations": None,
        "seed": 42,
    },
}


def list_templates() -> List[str]:
    """返回所有可用模板名称。"""
    return [name for name in CONFIG_TEMPLATES.keys()]


def get_template(name: str) -> Dict[str, Any]:
    """获取指定模板的配置（不含 _description 元字段）。

    Args:
        name: 模板名称

    Returns:
        配置字典（不含 _description）

    Raises:
        ConfigValidationError: 模板不存在
    """
    if name not in CONFIG_TEMPLATES:
        available = ", ".join(CONFIG_TEMPLATES.keys())
        raise ConfigValidationError(
            f"未知模板 '{name}'，可用模板: {available}"
        )
    template = dict(CONFIG_TEMPLATES[name])
    template.pop("_description", None)
    return template


# ============================================================
# 配置加载器
# ============================================================
class ConfigLoader:
    """配置加载器：从多种来源加载并验证配置。

    支持的来源（优先级从高到低）：
    1. 环境变量（AGENTWP_前缀）
    2. 显式参数覆盖
    3. YAML文件
    4. 字典
    5. 默认值
    """

    ENV_PREFIX = "AGENTWP_"

    def __init__(self, defaults: Optional[Dict[str, Any]] = None) -> None:
        self.defaults = defaults or dict(DEFAULT_CONFIG)

    # --------------------------------------------------------
    # 从字典加载
    # --------------------------------------------------------
    def from_dict(self, data: Dict[str, Any]) -> OrchestratorConfig:
        """从字典加载配置。"""
        merged = dict(self.defaults)
        merged.update({k: v for k, v in data.items() if v is not None})
        merged = self._apply_env_overrides(merged)
        self._validate(merged)
        return self._build(merged)

    # --------------------------------------------------------
    # 从YAML文件加载
    # --------------------------------------------------------
    def from_yaml(self, path: str) -> OrchestratorConfig:
        """从YAML文件加载配置。"""
        try:
            import yaml
        except ImportError as e:
            raise ImportError("需要PyYAML来加载YAML配置: pip install pyyaml") from e

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return self.from_dict(data)

    # --------------------------------------------------------
    # 生成默认配置文件
    # --------------------------------------------------------
    def generate_default_yaml(self, path: str, template: Optional[str] = None) -> None:
        """生成配置YAML文件。

        Args:
            path: 输出文件路径
            template: 模板名称（None=使用默认配置）
        """
        try:
            import yaml
        except ImportError as e:
            raise ImportError("需要PyYAML: pip install pyyaml") from e

        if template is not None:
            config_with_comments = get_template(template)
            description = CONFIG_TEMPLATES[template].get("_description", "")
        else:
            # 带注释的默认配置
            config_with_comments = {
                "grid_size": self.defaults["grid_size"],
                "n_agents": self.defaults["n_agents"],
                "db_path": self.defaults["db_path"],
                "checkpoint_interval": self.defaults["checkpoint_interval"],
                "skill_abstract_interval": self.defaults["skill_abstract_interval"],
                "meta_learn_interval": self.defaults["meta_learn_interval"],
                "distill_interval": self.defaults["distill_interval"],
                "evolve_interval": self.defaults["evolve_interval"],
                "coalition_interval": self.defaults["coalition_interval"],
                "max_iterations": self.defaults["max_iterations"],
                "seed": self.defaults["seed"],
            }
            description = "默认配置：平衡的功能与性能"

        header = (
            f"# Agent Whitepaper 永续自主进化系统配置\n"
            f"# 模板: {template or 'default'} - {description}\n"
            f"# 修改此文件后运行: agentwp run --config config.yaml\n\n"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)
            yaml.dump(config_with_comments, f, default_flow_style=False, sort_keys=False)

    # --------------------------------------------------------
    # 环境变量覆盖
    # --------------------------------------------------------
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """应用环境变量覆盖。"""
        result = dict(config)
        type_map = {
            "grid_size": int,
            "n_agents": int,
            "checkpoint_interval": int,
            "skill_abstract_interval": int,
            "meta_learn_interval": int,
            "distill_interval": int,
            "evolve_interval": int,
            "coalition_interval": int,
            "max_iterations": int,
            "seed": int,
        }
        for key, cast in type_map.items():
            env_key = self.ENV_PREFIX + key.upper()
            env_val = os.environ.get(env_key)
            if env_val is not None:
                try:
                    if cast is int and env_val.lower() in ("none", "null", ""):
                        result[key] = None
                    else:
                        result[key] = cast(env_val)
                except (ValueError, TypeError):
                    pass  # 忽略无效环境变量
        # db_path 是字符串
        env_db = os.environ.get(self.ENV_PREFIX + "DB_PATH")
        if env_db is not None:
            result["db_path"] = env_db
        return result

    # --------------------------------------------------------
    # 配置验证
    # --------------------------------------------------------
    def _validate(self, config: Dict[str, Any]) -> None:
        """验证配置合法性。"""
        if config.get("grid_size", 5) < 2:
            raise ConfigValidationError("grid_size 必须 >= 2")
        if config.get("n_agents", 1) < 1:
            raise ConfigValidationError("n_agents 必须 >= 1")
        for key in ("checkpoint_interval", "skill_abstract_interval",
                    "meta_learn_interval", "distill_interval",
                    "evolve_interval", "coalition_interval"):
            if config.get(key, 1) < 1:
                raise ConfigValidationError(f"{key} 必须 >= 1")
        mi = config.get("max_iterations")
        if mi is not None and mi < 1:
            raise ConfigValidationError("max_iterations 必须为 None 或 >= 1")
        if not config.get("db_path"):
            raise ConfigValidationError("db_path 不能为空")

    # --------------------------------------------------------
    # 构建OrchestratorConfig
    # --------------------------------------------------------
    def _build(self, config: Dict[str, Any]) -> OrchestratorConfig:
        """构建OrchestratorConfig。"""
        return OrchestratorConfig(
            grid_size=config["grid_size"],
            n_agents=config["n_agents"],
            db_path=config["db_path"],
            checkpoint_interval=config["checkpoint_interval"],
            skill_abstract_interval=config["skill_abstract_interval"],
            meta_learn_interval=config["meta_learn_interval"],
            distill_interval=config["distill_interval"],
            evolve_interval=config["evolve_interval"],
            coalition_interval=config["coalition_interval"],
            max_iterations=config["max_iterations"],
            seed=config["seed"],
        )


# ============================================================
# 便捷函数
# ============================================================
def load_config(path: Optional[str] = None, **overrides) -> OrchestratorConfig:
    """加载配置的便捷函数。

    Args:
        path: YAML配置文件路径（None=用默认值）
        **overrides: 覆盖参数

    Returns:
        OrchestratorConfig实例
    """
    loader = ConfigLoader()
    if path:
        cfg = loader.from_yaml(path)
        if overrides:
            # 在YAML基础上再覆盖
            data = asdict(cfg)
            data.update({k: v for k, v in overrides.items() if v is not None})
            return loader.from_dict(data)
        return cfg
    return loader.from_dict(overrides)
