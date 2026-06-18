# Agent Whitepaper — 自驱永续进化预测修正Agent系统

基于物理第一性原理（热力学第二定律、自由能原理）构建的自驱、永续、进化的智能Agent框架。

## 核心特性

- **预测编码**：基于变分自由能的层级生成模型，支持6种非线性激活函数
- **三大驱力**：安全驱力（自由能驱动）、社会驱力（稀疏度驱动）、探索驱力（收敛速度驱动）
- **双模式进化**：70%微修正 + 30%结构性重构
- **记忆系统**：短期记忆（环形缓冲区）+ 长期记忆（SQLite）+ 优先级回放 + 生成式回放
- **安全护栏**：CPU/内存/磁盘监控、文件保护、探索冻结、自动回滚
- **退化自修复**：EVI监控 → 根因诊断 → 自动修复 → 3次失败升级人工
- **72小时压力测试**：五场景验证（正常/异常/压力/新环境/长期）

## 快速开始

```bash
pip install -e .

# 基本使用
python -c "
from agent_whitepaper.core.stage_050_v2 import MultiLayerPredictorBuilder
from agent_whitepaper.core.stage_011_v2 import ActivationType
import numpy as np

model = MultiLayerPredictorBuilder.build(
    layer_dims=(4, 8, 2),
    activations=(ActivationType.RELU, ActivationType.LINEAR),
    seed=42)
x = np.random.randn(4)
print('Prediction:', model.predict(x))
"
```

## 项目结构

```
agent-whitepaper/
├── src/agent_whitepaper/core/    # 核心实现（69个stage文件 + 8个v2强化文件）
├── stages/                       # 100个阶段规范文档
├── 100-stage-plan.md             # 总体计划
├── setup.py                      # 包安装
└── requirements.txt              # 依赖
```

## 100阶段架构

| 层 | 阶段 | 名称 |
|----|------|------|
| 二 | 11-20 | 预测编码层 |
| 三 | 21-30 | 驱力引擎层 |
| 五 | 41-55 | 多层预测模型层 |
| 六 | 51-60 | 误差评估反馈层 |
| 七 | 61-70 | 双模式进化层 |
| 八 | 71-80 | 记忆与复盘层 |
| 九 | 81-90 | 安全护栏层 |
| 十 | 91-100 | 系统集成层 |

## 许可证

MIT
