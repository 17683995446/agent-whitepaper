# 示例脚本库 (Examples)

本目录包含 Agent Whitepaper 的可运行示例脚本，每个脚本覆盖一个常见用例。

## 示例列表

| 编号 | 脚本 | 说明 | 复杂度 |
|------|------|------|--------|
| 01 | `01_single_agent_training.py` | 单Agent训练：最简入门 | 入门 |
| 02 | `02_multi_agent_system.py` | 多Agent协作系统 | 中级 |
| 03 | `03_custom_environment.py` | 自定义环境接入 | 中级 |
| 04 | `04_persistence_and_recovery.py` | 持久化与状态恢复 | 高级 |
| 05 | `05_skill_abstraction.py` | 技能抽象与元学习 | 高级 |

## 运行方式

```bash
# 从项目根目录运行
PYTHONPATH=src python examples/01_single_agent_training.py

# 或安装包后
pip install -e .
python examples/01_single_agent_training.py
```

## 学习路径

1. **入门**: 先运行 `01_single_agent_training.py`，理解单Agent如何学习
2. **进阶**: 运行 `02_multi_agent_system.py`，理解多Agent协作
3. **定制**: 运行 `03_custom_environment.py`，学习如何接入自己的任务
4. **生产**: 运行 `04_persistence_and_recovery.py`，理解持久化与恢复
5. **优化**: 运行 `05_skill_abstraction.py`，理解技能抽象如何加速学习

每个脚本都是独立可运行的，也可以作为自己项目的起点。
