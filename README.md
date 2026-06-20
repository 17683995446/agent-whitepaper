# Agent Whitepaper — 自驱Agent自主进化系统

基于物理第一性原理（自由能原理、最近发展区理论）构建的自驱、永续、进化的智能Agent框架。

系统由11个阶段（A-K）构成完整的自主进化闭环：从单Agent感知-行动闭环，到技能抽象与元学习，到多Agent协作，到永续运行时，到技能迁移与知识蒸馏，到自主退化检测与重训练，最终由全系统编排器统一编排。

## 核心特性

- **自驱Agent闭环**：感知→驱力→策略→行动→学习（Q-learning + ε-greedy + 情节记忆）
- **技能增强**：n-gram技能抽象、技能精炼、元学习（学习如何学习）
- **长任务分解**：任务→子目标序列 + 状态机管理
- **多Agent协作**：注册中心、消息总线、联盟形成、共享技能库
- **永续运行时**：SQLite持久化、任务调度器、健康监控（断路器）、daemon主循环
- **技能迁移**：Q表序列化与迁移、技能→Q值注入、跨Agent知识蒸馏
- **自主进化**：最近发展区任务生成、性能退化检测、自主重训练
- **全系统编排**：A-F闭环编排、指标采集与可观测性
- **配置化与CLI**：YAML配置、命令行入口（agentwp）
- **真实环境适配**：EnvironmentProtocol + StateEncoder 抽象层
- **安全约束**：SafetyMonitor + SafeActionFilter + 资源管理（背压机制）

## 快速开始

### 安装

```bash
pip install -e .
```

> 要求 Python >= 3.8。核心依赖：numpy, pyyaml, psutil。

### 最小示例：单Agent学习

```python
from agent_whitepaper import SelfDrivenAgent, GridWorld

# 创建 5x5 网格世界（25个状态，4个动作）
env = GridWorld(size=5, seed=42)
agent = SelfDrivenAgent(
    action_space=env.action_space,
    n_states=25,
    learning_rate=0.1,
    seed=42,
)

# 训练 100 个情节
for episode in range(100):
    episode_obj = agent.run_episode(env, max_steps=100)

print(f"成功率: {agent.stats.success_rate:.1%}")
print(f"平均奖励: {agent.stats.avg_reward:.2f}")
print(f"Q表条目数: {agent.q_table.n_states * agent.q_table.n_actions}")
```

### 便捷API（一行代码）

常见任务封装为单行调用，降低入门门槛：

```python
from agent_whitepaper import quick_start, train_agent, run_demo, create_system, get_status

# 一行启动多Agent系统并获取结果
report = quick_start(iterations=10, n_agents=3, grid_size=5)
print(f"完成任务: {report['tasks_completed']}")

# 一行训练单个Agent
agent = train_agent(n_episodes=100)
print(f"成功率: {agent.stats.success_rate:.1%}")

# 一行运行演示
report = run_demo(n_iterations=30)

# 从模板创建系统（default/quick-test/production/multi-agent/long-run）
orch = create_system(template="quick-test")
for _ in range(10):
    orch.run_iteration()
orch.stop("done")

# 查询持久化数据库状态
status = get_status("agent_system.db")

# 列出可用配置模板及描述
from agent_whitepaper import list_available_templates
for name, desc in list_available_templates().items():
    print(f"  {name}: {desc}")
```

### CLI 入口

```bash
# 生成默认配置文件
agentwp init -o config.yaml

# 使用预置模板生成配置（quick-test/production/multi-agent/long-run）
agentwp init --template quick-test -o config.yaml

# 列出所有可用模板
agentwp init --list-templates

# 快速演示（30轮迭代）
agentwp demo

# 启动永续运行
agentwp run --config config.yaml

# 查询运行状态
agentwp status --db agent_system.db

# 环境诊断（检查依赖、导入、配置等）
agentwp doctor

# 查看版本
agentwp --version

# 详细日志模式
agentwp --verbose demo -n 50
```

### 演示入口

系统提供三种演示方式，适用于不同场景：

| 演示方式 | 命令 | 适用场景 | 迭代数 |
|---------|------|---------|--------|
| CLI 快速演示 | `agentwp demo` | 快速验证系统能力 | 30（可配置） |
| 端到端演示 | `python -m agent_whitepaper.demos.e2e_demo` | 验证三大核心能力 | 按能力分步 |
| 长时运行实证 | 见下方代码 | 验证持续进化能力 | 100+ |

**CLI 快速演示**（可参数化）:
```bash
# 自定义参数：5轮迭代，2个Agent，4x4网格
agentwp demo -n 5 -a 2 -g 4
```

**端到端演示**（三大能力闭环）:
```bash
python -m agent_whitepaper.demos.e2e_demo
```
验证：1) 自驱长任务 2) 技能增强 3) 多Agent协作

**长时运行实证**:
```python
from agent_whitepaper import LongRunDemo

demo = LongRunDemo()  # 默认100轮迭代
result = demo.run()
print(f"最终成功率: {result.final_success_rate:.1%}")
```

## 架构总览

```
依赖层次（从底到顶，箭头表示依赖方向）:

  stage_a (基础层: 单Agent闭环)
    ├── stage_b → {a}        (技能增强与长任务)
    ├── stage_i → {a}        (真实环境适配)
    └── stage_j → {}         (安全约束，独立)
  stage_c → {b}              (多Agent协作)
    ├── stage_d → {a, b}     (永续运行时)
    └── stage_e → {a, b}     (技能迁移)
  stage_f → {a, b, e}        (自主进化)
  stage_g → {a,b,c,d,e,f}    (全系统集成)
  stage_h → {d, g}           (配置化与CLI)
  stage_k → {g}              (长时运行实证)
  demos  → {a, b, c}         (端到端演示)
```

## 项目结构

```
agent-whitepaper/
├── src/agent_whitepaper/
│   ├── __init__.py            # 顶层包（聚合公共API，30个headline导出）
│   ├── _dependencies.py       # 阶段间依赖关系声明（单一事实来源）
│   ├── stage_a/               # 单Agent闭环（感知-驱力-策略-行动-学习）
│   ├── stage_b/               # 技能增强与长任务（n-gram、元学习、任务分解）
│   ├── stage_c/               # 多Agent协作（注册中心、消息总线、联盟）
│   ├── stage_d/               # 永续运行时（持久化、调度器、健康监控）
│   ├── stage_e/               # 技能迁移（Q表迁移、Q值注入、知识蒸馏）
│   ├── stage_f/               # 自主进化（任务生成、退化检测、重训练）
│   ├── stage_g/               # 全系统集成（指标采集、A-F闭环编排）
│   ├── stage_h/               # 配置化与CLI（YAML配置、命令行入口）
│   ├── stage_i/               # 真实环境适配（EnvironmentProtocol、状态编码器）
│   ├── stage_j/               # 安全约束（SafetyMonitor、资源管理、背压）
│   ├── stage_k/               # 长时运行实证（演示、报告生成）
│   ├── demos/                 # 端到端演示
│   └── tests/                 # 测试套件（stage_a~k + stage_m~p 秩序化测试）
├── archive/                   # 历史归档代码（预测编码框架 Stages 11-100）
├── stages/                    # 100阶段规范文档（设计参考）
├── pyproject.toml             # 项目配置 + ruff lint 规则
├── setup.py                   # 包安装
└── requirements.txt           # 依赖
```

## 11阶段详解

| 阶段 | 包名 | 核心模块 | 说明 |
|------|------|----------|------|
| A | stage_a | agent, environment, policy, memory, action_space | 单Agent闭环：Q-learning + ε-greedy + 情节记忆 + 驱力系统 |
| B | stage_b | skill_abstractor, skill_refiner, meta_learner, task_decomposer, subgoal_manager | 技能n-gram抽象、元学习、长任务分解与状态机 |
| C | stage_c | registry, message_bus, role_assigner, coalition, shared_skills | 多Agent注册中心、消息总线、联盟形成、共享技能库 |
| D | stage_d | persistence, scheduler, health_monitor, runtime | SQLite持久化、任务调度、断路器、daemon永续主循环 |
| E | stage_e | qtable_migration, skill_injector, knowledge_distiller | Q表序列化迁移、技能→Q值注入、跨Agent知识蒸馏 |
| F | stage_f | autonomous_generator, degradation_detector, autonomous_retrainer | 最近发展区任务生成、退化检测、自主重训练 |
| G | stage_g | metrics, orchestrator | 指标采集、A-F全闭环编排 |
| H | stage_h | config, cli | YAML配置加载、agentwp CLI入口 |
| I | stage_i | environment_interface, function_env, adapters | EnvironmentProtocol抽象、函数调用环境、多环境训练器 |
| J | stage_j | safety_bounds, resource_manager | SafetyMonitor、SafeActionFilter、资源管理（背压） |
| K | stage_k | long_run_demo, report_generator | 长时运行演示、能力评估与报告生成 |

## 公共API

顶层包直接导出各阶段核心入口：

```python
from agent_whitepaper import (
    # 阶段A
    SelfDrivenAgent, GridWorld, QTable,
    # 阶段B
    Skill, SkillLibrary, MetaLearner, TaskDecomposer,
    # 阶段C
    AgentRegistry, MessageBus, CoalitionFormer,
    # 阶段D
    PerpetualRuntime, PersistenceEngine, TaskScheduler, HealthMonitor,
    # 阶段E
    QTableMigrator, SkillQValueInjector, KnowledgeDistiller,
    # 阶段F
    AutonomousTaskGenerator, DegradationDetector, AutonomousRetrainer,
    # 阶段G
    SystemOrchestrator, MetricsCollector,
    # 阶段H
    load_config, main,
    # 阶段I
    EnvironmentProtocol, FunctionCallEnvironment,
    # 阶段J
    SafetyMonitor, ResourceManager,
    # 阶段K
    LongRunDemo, ReportGenerator,
    # 异常层次体系
    AgentWhitepaperError, ConfigError, PersistenceError,
    ComponentError, RecoveryError, CLIError,
    # 抽象接口
    PolicyProtocol, MemoryProtocol, AgentProtocol,
    # 日志配置
    get_logger, configure_logging,
)
```

各子包也提供完整的公共API（通过 `__all__` 声明），可按需导入：

```python
from agent_whitepaper.stage_b import SkillAbstractor, LongTaskExecutor
from agent_whitepaper.stage_d import CircuitBreaker, safe_execute
```

## 异常层次体系

系统在 `_exceptions.py` 中定义了统一的异常层次（阶段R），所有自定义异常继承自 `AgentWhitepaperError`：

```
AgentWhitepaperError
├── ConfigError              # 配置域
├── PersistenceError         # 持久化域
│   ├── TransactionError
│   └── StateSerializationError
├── RuntimeExecutionError    # 运行时域
│   ├── IterationError
│   └── StateRecoveryError
├── ComponentError           # 组件域（可降级，非致命）
│   ├── SkillAbstractionError
│   ├── MetaLearningError
│   ├── CoalitionError
│   ├── DistillationError
│   ├── EvolutionError
│   ├── MetricsCollectionError
│   ├── CheckpointError
│   └── AgentTaskError
├── RecoveryError            # 恢复域
└── CLIError                 # CLI域
```

异常捕获规范：
- **防火墙层**（主循环、Agent隔离）：`except Exception`（永续运行保障）
- **组件层**（技能抽象、元学习等）：`except Exception` + 降级注释
- **边界层**（CLI、DB）：`except` 具体标准库异常

## 抽象接口

系统在 `_protocols.py` 中定义核心抽象接口（阶段S），实现接口分离=可替换性：

| 接口 | 定义位置 | 具体实现 | 契约方法 |
|------|----------|----------|----------|
| `EnvironmentProtocol` | stage_i | GridWorld, FunctionCallEnvironment | reset/step/action_space/observation_dim/n_states |
| `StateEncoder` | stage_i | GridWorldStateEncoder, BitmapStateEncoder | encode/n_states |
| `PolicyProtocol` | _protocols | QTable | get/best_action/update |
| `MemoryProtocol` | _protocols | EpisodicBuffer | add/sample/get_recent_episodes/\_\_len\_\_ |
| `AgentProtocol` | _protocols | SelfDrivenAgent | perceive_and_act/run_episode |

设计原则：
- **接口分离**：高层模块依赖抽象，不依赖具体实现
- **可替换性**：任何满足接口的实现都可替换（如 QTable → 神经网络策略）
- **可测试性**：测试可 mock 接口而非依赖具体实现

## 日志与可观测性

系统在 `_logging.py` 中提供统一日志配置（阶段T），替代散落的 `print`：

```python
from agent_whitepaper import get_logger, configure_logging
import logging

# 配置日志级别（默认 WARNING，调试时设为 DEBUG）
configure_logging(level=logging.DEBUG)

# 在任意模块中获取 logger
logger = get_logger(__name__)
logger.debug("技能抽象失败: %s", e, exc_info=True)
```

日志规范：
- **用户面输出**（CLI/demos）：保留 `print`（面向用户的输出）
- **系统内部事件**：使用 `logging`（面向开发者的诊断）
- **降级路径**（`except Exception`）：记录 `DEBUG`（预期失败，仅调试时可见）
- **关键状态转换**：记录 `INFO`（启动/停止/恢复）

## 依赖关系

阶段间依赖关系在 `_dependencies.py` 中显式声明（单一事实来源）：

```python
from agent_whitepaper._dependencies import STAGE_DEPENDENCIES, topological_order

print(topological_order())
# ['stage_a', 'stage_b', 'stage_c', 'stage_d', 'stage_e',
#  'stage_f', 'stage_g', 'stage_h', 'stage_i', 'stage_j',
#  'stage_k', 'demos']
```

## 测试

```bash
# 运行全部测试
python -m pytest src/agent_whitepaper/tests/ -q

# 运行特定阶段测试
python -m pytest src/agent_whitepaper/tests/test_stage_a.py -v

# 代码规范检查
python -m ruff check src/agent_whitepaper/
```

测试套件包含：
- `test_stage_a.py` ~ `test_stage_k.py`：各阶段功能测试
- `test_stage_m.py`：公共API导出验证（26个测试）
- `test_stage_n.py`：依赖关系显式化验证（23个测试）
- `test_stage_o.py`：命名规范统一验证（12个测试）

## 历史归档

`archive/` 目录存放早期预测编码框架代码（Stages 11-100），与当前A-K系统无关。
详见 [archive/README.md](archive/README.md)。

## 许可证

MIT
