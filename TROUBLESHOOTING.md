# 故障排查指南 (Troubleshooting)

本指南列出使用 Agent Whitepaper 时常见的错误及其解决方案。

## 目录

1. [安装问题](#安装问题)
2. [配置错误](#配置错误)
3. [运行时错误](#运行时错误)
4. [CLI 错误](#cli-错误)
5. [API 使用错误](#api-使用错误)
6. [性能问题](#性能问题)

---

## 安装问题

### ImportError: No module named 'numpy'

**原因**: 未安装 numpy 核心依赖。

**解决**:
```bash
pip install numpy
# 或安装完整依赖
pip install -e ".[dev]"
```

### ImportError: No module named 'yaml'

**原因**: 未安装 PyYAML，仅在加载 YAML 配置文件时需要。

**解决**:
```bash
pip install pyyaml
```

**说明**: 如果仅通过 Python API 使用（不加载 YAML 配置），则无需安装 PyYAML。

### psutil 未安装警告

**现象**: 日志中出现 `psutil 未安装，CPU/内存监控将降级` 警告。

**原因**: psutil 是可选依赖，用于监控 CPU 和内存使用。

**解决**:
```bash
pip install psutil
```

**说明**: 不安装 psutil 不影响系统核心功能，仅 CPU/内存监控降级为只跟踪迭代次数和缓冲区大小。

### Python 版本不兼容

**现象**: 安装或运行时报语法错误。

**原因**: 本项目要求 Python >= 3.8。

**解决**:
```bash
python --version  # 检查版本
# 升级 Python 到 3.8+
```

---

## 配置错误

### ConfigError: 配置验证失败

**现象**: `agentwp run --config config.yaml` 报配置错误。

**常见原因与解决**:

1. **网格大小无效**: `grid_size` 必须 >= 2
   ```yaml
   grid_size: 5  # 而非 1 或 0
   ```

2. **Agent 数量无效**: `n_agents` 必须 >= 1
   ```yaml
   n_agents: 3  # 而非 0 或负数
   ```

3. **迭代次数无效**: `max_iterations` 为 None（无限）或正整数

### FileNotFoundError: 配置文件不存在

**现象**: `agentwp run --config /path/to/config.yaml` 报文件不存在。

**解决**:
```bash
# 先生成默认配置
agentwp init -o config.yaml
# 修改后运行
agentwp run --config config.yaml
```

### YAML 语法错误

**现象**: `yaml.YAMLError` 或配置解析失败。

**解决**:
- 检查缩进（YAML 使用空格，不用 Tab）
- 检查冒号后有空格: `key: value` 而非 `key:value`
- 使用 `agentwp init` 重新生成合法配置模板

---

## 运行时错误

### PersistenceError: 数据库错误

**现象**: SQLite 数据库操作失败。

**常见原因与解决**:

1. **权限不足**: 数据库文件所在目录无写权限
   ```bash
   chmod 755 /path/to/db/dir
   ```

2. **磁盘空间不足**: 检查磁盘空间
   ```bash
   df -h
   ```

3. **数据库锁定**: 其他进程占用数据库
   - 使用 `:memory:` 内存数据库进行测试
   - 等待其他进程释放

### RecoveryError: 状态恢复失败

**现象**: 从检查点恢复状态时失败。

**解决**:
- 检查检查点文件是否完整
- 使用 `agentwp status -d <db_path>` 查看检查点状态
- 必要时删除数据库重新开始

### ComponentError: 组件降级

**现象**: 技能抽象、元学习等组件失败。

**说明**: 这类错误通常是非致命的（可降级），系统会继续运行但功能受限。使用 `--verbose` 标志查看详细日志:
```bash
agentwp --verbose run -n 100
```

---

## CLI 错误

### 未知命令

**现象**: `agentwp unknown_command` 报错。

**解决**: 查看可用命令
```bash
agentwp --help
```

可用命令: `init`, `run`, `demo`, `status`

### 无效参数

**现象**: `agentwp demo -n abc` 报类型错误。

**解决**: 参数类型说明
- `-n/--iterations`: 整数（如 `-n 30`）
- `-a/--agents`: 整数（如 `-a 3`）
- `-g/--grid`: 整数（如 `-g 5`）

### 查看版本

```bash
agentwp --version
# 或
agentwp -V
```

### 日志级别控制

```bash
# 详细日志（DEBUG级别，显示降级路径等诊断信息）
agentwp --verbose demo

# 静默模式（仅ERROR级别）
agentwp --quiet demo
```

---

## API 使用错误

### ValueError: 网格大小必须 >= 2

**原因**: `GridWorld(size=1)` 网格太小。

**解决**:
```python
env = GridWorld(size=5)  # 使用 >= 2 的网格大小
```

### ValueError: 状态数和动作数必须 >= 1

**原因**: `QTable(n_states=0, n_actions=4)` 参数无效。

**解决**:
```python
q = QTable(n_states=25, n_actions=4)  # 状态数 = 网格大小²
```

### ValueError: 学习率必须 ∈ (0,1]

**原因**: `QTable(learning_rate=0)` 或 `learning_rate=1.5`。

**解决**:
```python
q = QTable(n_states=25, n_actions=4, learning_rate=0.1)  # 0 < lr <= 1
```

### AttributeError: 'SelfDrivenAgent' object has no attribute 'q_table_size'

**原因**: 使用了不存在的构造参数。

**正确用法**:
```python
agent = SelfDrivenAgent(
    action_space=env.action_space,
    n_states=25,           # 不是 q_table_size
    learning_rate=0.1,
    seed=42,
)
```

---

## 性能问题

### 运行速度慢

**可能原因与解决**:

1. **psutil 未安装**: 资源监控降级不影响速度，但建议安装
   ```bash
   pip install psutil
   ```

2. **网格过大**: 减小网格大小
   ```python
   env = GridWorld(size=5)  # 而非 size=50
   ```

3. **Agent 过多**: 减少 Agent 数量
   ```bash
   agentwp demo -a 2  # 而非 -a 100
   ```

4. **使用内存数据库**: 测试时使用 `:memory:` 避免磁盘 I/O
   ```bash
   agentwp run --memory
   ```

### 内存占用高

**解决**:
- 减小 `max_iterations`
- 使用持久化数据库（非 `:memory:`）让系统定期清理
- 减小记忆缓冲区容量

---

## 获取帮助

如果以上指南未能解决您的问题：

1. 使用 `--verbose` 标志获取详细日志
2. 查看 [README.md](README.md) 了解基本用法
3. 使用 `help()` 查看 API 文档:
   ```python
   from agent_whitepaper import SelfDrivenAgent
   help(SelfDrivenAgent)
   ```
