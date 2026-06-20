"""阶段H-5：交互式配置向导 - 系统的"对话式引导"。

第一性原理：
    "Guide the user"——对话式引导优于记忆参数。
    新用户面对11个配置参数会感到困惑。交互式向导通过：
    1. 提问：用自然语言询问用户意图
    2. 推荐：基于用户回答推荐合适配置
    3. 解释：每个参数都有简短说明
    4. 验证：即时验证输入合法性
    5. 生成：最终生成可用的配置文件

    向导将"配置专家知识"内置到系统中，降低入门门槛。

用法：
    from agent_whitepaper.stage_h.wizard import ConfigWizard

    wizard = ConfigWizard()
    config = wizard.run()  # 交互式问答
    wizard.save(config, "config.yaml")

    # 或非交互模式（用于测试/自动化）
    config = wizard.run_with_answers(["quick-test", "3", "2", "10"])
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import (
    CONFIG_TEMPLATES,
    get_template,
    list_templates,
)


# ============================================================
# 向导问题定义
# ============================================================
class WizardQuestion:
    """向导问题定义。

    Attributes:
        key: 配置键名
        prompt: 提示文本
        default: 默认值
        validator: 验证函数（返回(是否合法, 错误消息)）
        help_text: 帮助文本
    """

    def __init__(
        self,
        key: str,
        prompt: str,
        default: Any,
        validator: Optional[Any] = None,
        help_text: str = "",
    ) -> None:
        self.key = key
        self.prompt = prompt
        self.default = default
        self.validator = validator
        self.help_text = help_text

    def validate(self, value: Any) -> Tuple[bool, str]:
        """验证输入值。"""
        if self.validator is None:
            return True, ""
        return self.validator(value)


# ============================================================
# 验证函数
# ============================================================
def _validate_int_min(min_val: int) -> Callable[[str], Tuple[bool, str]]:
    """创建最小值验证器。"""
    def validator(value: str) -> Tuple[bool, str]:
        try:
            v = int(value)
            if v < min_val:
                return False, f"值必须 >= {min_val}"
            return True, ""
        except ValueError:
            return False, "请输入整数"
    return validator


def _validate_int_or_none() -> Callable[[str], Tuple[bool, str]]:
    """创建整数或None验证器。"""
    def validator(value: str) -> Tuple[bool, str]:
        if value.lower() in ("none", "null", ""):
            return True, ""
        try:
            int(value)
            return True, ""
        except ValueError:
            return False, "请输入整数或 'none'"
    return validator


def _validate_template() -> Callable[[str], Tuple[bool, str]]:
    """创建模板名验证器。"""
    def validator(value: str) -> Tuple[bool, str]:
        if value in list_templates():
            return True, ""
        return False, f"未知模板，可选: {', '.join(list_templates())}"
    return validator


def _validate_db_path() -> Callable[[str], Tuple[bool, str]]:
    """创建数据库路径验证器。"""
    def validator(value: str) -> Tuple[bool, str]:
        if not value or not value.strip():
            return False, "路径不能为空"
        return True, ""
    return validator


# ============================================================
# 配置向导
# ============================================================
class ConfigWizard:
    """交互式配置向导。

    第一性原理：
        "Guide the user"——对话式引导优于记忆参数。
        向导通过问答引导用户完成配置，降低认知负荷。

    用法：
        wizard = ConfigWizard()
        config = wizard.run()
        wizard.save(config, "config.yaml")
    """

    def __init__(
        self,
        input_func: Any = input,
        output_func: Any = print,
    ) -> None:
        """初始化向导。

        Args:
            input_func: 输入函数（默认input，可替换用于测试）
            output_func: 输出函数（默认print）
        """
        self.input = input_func
        self.output = output_func
        self._questions = self._build_questions()

    def _build_questions(self) -> List[WizardQuestion]:
        """构建问题序列。"""
        return [
            WizardQuestion(
                key="_template",
                prompt="选择配置模板",
                default="default",
                validator=_validate_template(),
                help_text="模板提供预置参数，后续可覆盖",
            ),
            WizardQuestion(
                key="grid_size",
                prompt="网格大小",
                default=5,
                validator=_validate_int_min(2),
                help_text="GridWorld的边长（>=2），决定状态空间大小",
            ),
            WizardQuestion(
                key="n_agents",
                prompt="Agent数量",
                default=3,
                validator=_validate_int_min(1),
                help_text="协作Agent的数量（>=1）",
            ),
            WizardQuestion(
                key="max_iterations",
                prompt="最大迭代次数（none=无限）",
                default=None,
                validator=_validate_int_or_none(),
                help_text="运行多少轮迭代后停止，none表示无限运行",
            ),
            WizardQuestion(
                key="db_path",
                prompt="数据库路径（:memory:=内存）",
                default="agent_system.db",
                validator=_validate_db_path(),
                help_text="持久化数据库路径，:memory:表示不持久化",
            ),
            WizardQuestion(
                key="seed",
                prompt="随机种子",
                default=42,
                validator=_validate_int_min(0),
                help_text="随机种子，相同种子保证结果可复现",
            ),
        ]

    def run(self) -> Dict[str, Any]:
        """运行交互式向导。

        通过问答收集用户偏好，返回配置字典。

        Returns:
            配置字典（不含 _template 元字段）
        """
        self.output("\n=== Agent Whitepaper 配置向导 ===")
        self.output("按 Enter 接受默认值，输入 ? 查看帮助\n")

        # 第一步：选择模板
        self.output("可用模板:")
        for name in list_templates():
            desc = CONFIG_TEMPLATES[name].get("_description", "")
            self.output(f"  {name:<15} {desc}")

        answers: Dict[str, Any] = {}
        for question in self._questions:
            value = self._ask_question(question)
            answers[question.key] = value

        # 基于模板构建最终配置
        template_name = answers.pop("_template")
        config = get_template(template_name)

        # 用用户回答覆盖模板值
        for key, value in answers.items():
            if value is not None:
                config[key] = value

        self.output("\n=== 配置摘要 ===")
        for key, value in config.items():
            self.output(f"  {key}: {value}")

        return config

    def run_with_answers(self, answers: List[str]) -> Dict[str, Any]:
        """非交互模式运行向导（用于测试/自动化）。

        Args:
            answers: 按问题顺序的答案列表

        Returns:
            配置字典
        """
        if len(answers) != len(self._questions):
            raise ValueError(
                f"需要 {len(self._questions)} 个答案，得到 {len(answers)}"
            )

        result: Dict[str, Any] = {}
        for question, raw_answer in zip(self._questions, answers):
            value = self._parse_answer(question, raw_answer)
            result[question.key] = value

        template_name = result.pop("_template")
        config = get_template(template_name)
        for key, value in result.items():
            if value is not None:
                config[key] = value
        return config

    def _ask_question(self, question: WizardQuestion) -> Any:
        """询问单个问题。"""
        while True:
            default_str = str(question.default) if question.default is not None else "none"
            prompt = f"  {question.prompt} [{default_str}]: "
            raw = self.input(prompt)

            # 空输入使用默认值
            if not raw.strip():
                return question.default

            # 帮助
            if raw.strip() == "?":
                self.output(f"    帮助: {question.help_text}")
                continue

            # 验证
            ok, msg = question.validate(raw)
            if not ok:
                self.output(f"    无效: {msg}")
                continue

            return self._parse_answer(question, raw)

    def _parse_answer(self, question: WizardQuestion, raw: str) -> Any:
        """将字符串答案解析为合适的类型。"""
        raw = raw.strip()
        if not raw:
            return question.default

        # none 处理
        if raw.lower() in ("none", "null"):
            return None

        key = question.key
        if key in ("grid_size", "n_agents", "seed",
                    "checkpoint_interval", "skill_abstract_interval",
                    "meta_learn_interval", "distill_interval",
                    "evolve_interval", "coalition_interval"):
            return int(raw)
        if key == "max_iterations":
            if raw.lower() in ("none", "null", ""):
                return None
            return int(raw)
        return raw

    def save(self, config: Dict[str, Any], path: str) -> None:
        """保存配置到YAML文件。

        Args:
            config: 配置字典
            path: 输出文件路径
        """
        # 生成带注释的YAML
        try:
            import yaml
        except ImportError as e:
            raise ImportError("需要PyYAML: pip install pyyaml") from e

        header = (
            "# Agent Whitepaper 配置文件\n"
            "# 由配置向导生成\n\n"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        self.output(f"\n配置已保存到: {path}")
        self.output(f"运行: agentwp run --config {path}")


# ============================================================
# 便捷函数
# ============================================================
def run_wizard(
    output_path: str = "config.yaml",
    input_func: Any = None,
    output_func: Any = None,
) -> Dict[str, Any]:
    """运行配置向导并保存结果。

    Args:
        output_path: 输出配置文件路径
        input_func: 输入函数（默认input，可替换用于测试）
        output_func: 输出函数（默认print，可替换用于测试）

    Returns:
        配置字典
    """
    kwargs: Dict[str, Any] = {}
    if input_func is not None:
        kwargs["input_func"] = input_func
    if output_func is not None:
        kwargs["output_func"] = output_func
    wizard = ConfigWizard(**kwargs)
    config = wizard.run()
    wizard.save(config, output_path)
    return config
