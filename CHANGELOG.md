# 变更日志 (CHANGELOG)

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/) 规范。

## [Unreleased]

### 阶段Z: 演示统一与文档完善
- 统一 demo 入口文档（README 演示章节）
- 新增 CHANGELOG.md 变更日志
- 新增 CONTRIBUTING.md 贡献指南

### 阶段Y: 错误诊断增强
- 新增 TROUBLESHOOTING.md 故障排查指南（6类常见问题）
- resource_manager.py: psutil 缺失时记录 WARNING 级别友好提示
- 验证关键错误消息包含可操作的解决建议

### 阶段X: API文档示例
- SelfDrivenAgent docstring 添加 doctest 格式使用示例
- GridWorld docstring 添加 doctest 格式使用示例
- SystemOrchestrator docstring 添加 doctest 格式使用示例
- QTable docstring 添加 doctest 格式使用示例

### 阶段W: CLI增强
- 添加 `--version/-V` 全局标志显示版本号
- 添加 `--verbose/-v` 和 `--quiet/-q` 全局标志控制日志级别
- demo 子命令参数化: `-n/--iterations`, `-a/--agents`, `-g/--grid`
- cmd_run/cmd_demo 异常友好捕获（ConfigError/FileNotFoundError/AgentWhitepaperError）
- 修复循环导入: `__version__` 移至 `__init__.py` 顶部

### 阶段V: 快速入门修复
- 修复 README 快速入门示例代码错误（API签名不匹配）
- `SelfDrivenAgent(q_table_size=16)` → `SelfDrivenAgent(n_states=25)`
- 移除不存在的 `perceive_and_act(obs, reward=0, done=False, env=env)` 调用
- 改用 `agent.run_episode(env, max_steps=100)` 简洁 API
- 移除私有属性 `agent.q_table._table` 访问
- 添加安装说明和完整 CLI 命令示例

## [0.1.0] - 基础版本

### 阶段T: 日志与可观测性规范化
- 新增 `_logging.py` 统一日志模块（`get_logger`/`configure_logging`）
- 9处 `except Exception: pass` 替换为 `logger.debug()` 降级日志
- 默认 WARNING 级别，不干扰正常输出

### 阶段S: 模块化接口抽象
- 新增 `_protocols.py` 抽象接口（`PolicyProtocol`/`MemoryProtocol`/`AgentProtocol`）
- `Environment` 升级为正式 ABC（抽象基类）
- `QTable`/`EpisodicBuffer`/`SelfDrivenAgent` 实现对应协议

### 阶段R: 错误处理规范化
- 新增 `_exceptions.py` 异常层次体系（17个异常类，6个域）
- 根异常 `AgentWhitepaperError`，按域分类：Config/Persistence/Runtime/Component/Recovery/CLI
- `ConfigValidationError` 多重继承保证向后兼容

### 阶段Q: 类型注解规范化
- 修复 F841 未使用变量警告
- 类型注解规范化检查

### 阶段P: 文档秩序化
- 模块 docstring 统一格式（第一性原理说明）
- 公共 API 文档完善

### 阶段O: 命名规范统一
- ruff 配置启用 N/E/W/I/F/UP 规则集
- 命名约定文档化（数学符号、领域约定豁免）

### 阶段N: 依赖关系显式化
- `_dependencies.py` 依赖关系邻接表 + 拓扑排序
- 阶段间依赖关系显式声明

### 阶段M: 公共API导出秩序化
- `__init__.py` 公共 API 导出规范化
- `__all__` 列表完善

### 阶段L: 历史代码归档
- 历史代码归档至 `archive/`
- 秩序化清理

### 阶段A-K: 核心功能实现
- 阶段A: 单Agent闭环（感知-驱力-策略-行动-学习）
- 阶段B: 技能增强与长任务（n-gram抽象、元学习、任务分解）
- 阶段C: 多Agent协作（注册中心、消息总线、联盟、共享技能）
- 阶段D: 永续运行时（SQLite持久化、调度器、健康监控、daemon）
- 阶段E: 技能迁移（Q表序列化、技能Q值注入、知识蒸馏）
- 阶段F: 自主进化（任务生成、退化检测、自主重训练）
- 阶段G: 全系统集成（指标采集、A-F闭环编排）
- 阶段H: 配置化与CLI（YAML配置、命令行入口）
- 阶段I: 真实环境适配（EnvironmentProtocol、状态编码器）
- 阶段J: 安全约束（SafetyMonitor、资源管理、背压）
- 阶段K: 长时运行实证（演示、报告生成）
