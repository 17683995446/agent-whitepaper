# 贡献指南 (CONTRIBUTING)

感谢您对 Agent Whitepaper 项目的关注！本文档指导您如何参与贡献。

## 开发环境搭建

### 1. 克隆仓库

```bash
git clone https://github.com/17683995446/agent-whitepaper.git
cd agent-whitepaper
```

### 2. 安装依赖

```bash
# 安装核心依赖 + 开发依赖
pip install -e ".[dev]"

# 安装可选依赖（完整功能）
pip install -e ".[full]"
```

### 3. 验证安装

```bash
# 运行测试
python -m pytest src/agent_whitepaper/tests/ -q

# 代码检查
python -m ruff check src/agent_whitepaper/

# 运行演示
agentwp demo -n 5
```

## 开发规范

### 代码风格

本项目使用 [ruff](https://github.com/astral-sh/ruff) 进行代码检查，配置见 `pyproject.toml`。

**启用的规则集**:
- `N` - pep8-naming（命名规范）
- `E` - pycodestyle errors
- `W` - pycodestyle warnings
- `I` - isort（import 排序）
- `F` - pyflakes（未使用变量等）
- `UP` - pyupgrade（语法现代化）

**运行检查**:
```bash
python -m ruff check src/agent_whitepaper/
```

### 命名约定

- **类名**: PascalCase（如 `SelfDrivenAgent`）
- **函数/变量**: snake_case（如 `run_episode`）
- **常量**: UPPER_SNAKE_CASE（如 `ACTION_UP`）
- **私有成员**: 前缀下划线（如 `_table`）
- **数学符号豁免**: `Q`, `T`, `N`, `M`, `K` 等大写字母允许（领域约定）

### 异常处理规范

遵循三层异常捕获规范（见 `_exceptions.py` docstring）:

1. **防火墙层**（主循环、Agent隔离）: `except Exception`（永续运行保障）
2. **组件层**（技能抽象、元学习等）: `except Exception` + 降级 + `logger.debug()`
3. **边界层**（CLI、DB）: `except 具体标准库异常`

**自定义异常**: 使用 `_exceptions.py` 中的异常类，不要创建新的根异常。

### 日志规范

- **用户面输出**（CLI/demos）: 使用 `print`
- **系统内部事件**: 使用 `logging`（通过 `get_logger(__name__)`）
- **降级路径**: `logger.debug("...: %s", e, exc_info=True)`
- **关键状态转换**: `logger.info(...)`
- **不可恢复错误**: `logger.error(...)`

```python
from agent_whitepaper._logging import get_logger

logger = get_logger(__name__)
```

### 测试规范

- 每个新阶段创建对应的 `test_stage_<X>.py`
- 测试类使用 `Test*` 前缀
- 测试方法使用 `test_*` 前缀
- 验证内容: 功能正确性 + API 一致性 + 源码结构

**运行测试**:
```bash
# 全量测试
python -m pytest src/agent_whitepaper/tests/ -q

# 单个阶段测试
python -m pytest src/agent_whitepaper/tests/test_stage_a.py -v
```

### Docstring 规范

- 模块 docstring: 说明第一性原理（为什么这样设计）
- 类 docstring: 说明职责 + 使用示例（doctest 格式）
- 方法 docstring: 说明功能 + 参数 + 返回值

**示例**:
```python
class MyClass:
    """一句话说明职责。

    更详细的说明...

    示例：
        >>> from agent_whitepaper import MyClass
        >>> obj = MyClass(param=1)
        >>> obj.method()
        result
    """
```

## 添加新阶段

### 1. 创建阶段包

```
src/agent_whitepaper/stage_<x>/
├── __init__.py    # 导出该阶段公共 API
├── module1.py     # 核心模块
└── module2.py
```

### 2. 注册依赖关系

在 `_dependencies.py` 中声明新阶段的依赖:

```python
STAGE_DEPENDENCIES: Dict[str, Set[str]] = {
    ...
    "stage_x": {"stage_a", "stage_b"},  # 依赖的阶段
}
```

### 3. 导出公共 API

在 `__init__.py` 中导出新阶段的核心类:

```python
from .stage_x import NewClass
```

### 4. 编写测试

创建 `tests/test_stage_x.py`，验证:
- 模块可导入
- 核心功能正确
- API 签名一致
- 异常处理正确

### 5. 更新文档

- 更新 `README.md` 项目结构和阶段详解
- 更新 `CHANGELOG.md` 记录变更

## 提交规范

### Commit 消息格式

```
<类型>: <简述>

<详细说明>
```

**类型**:
- `feat`: 新功能
- `fix`: 修复 bug
- `refactor`: 重构
- `docs`: 文档
- `test`: 测试
- `chore`: 构建/工具

**示例**:
```
feat: 实现阶段X - 新功能描述

- 模块1: 功能说明
- 模块2: 功能说明
- 新增 test_stage_x.py: N个测试
```

### 提交前检查

```bash
# 1. 代码检查
python -m ruff check src/agent_whitepaper/

# 2. 全量测试
python -m pytest src/agent_whitepaper/tests/ -q

# 3. 确认无回归
```

## 项目结构

```
agent-whitepaper/
├── src/agent_whitepaper/
│   ├── __init__.py            # 顶层包（聚合公共API）
│   ├── _exceptions.py         # 异常层次体系
│   ├── _protocols.py          # 抽象接口
│   ├── _logging.py            # 统一日志
│   ├── _dependencies.py       # 阶段依赖关系
│   ├── stage_a/ ~ stage_k/    # 11个核心阶段
│   ├── demos/                 # 端到端演示
│   └── tests/                 # 测试套件
├── README.md                  # 项目说明
├── CHANGELOG.md               # 变更日志
├── CONTRIBUTING.md            # 本文件
├── TROUBLESHOOTING.md         # 故障排查
└── pyproject.toml             # 项目配置
```

## 获取帮助

- 查看 [README.md](README.md) 了解项目概览
- 查看 [TROUBLESHOOTING.md](TROUBLESHOOTING.md) 解决常见问题
- 使用 `help()` 查看 API 文档
