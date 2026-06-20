# 归档代码说明

本目录存放历史遗留代码，已从主代码库中归档。

## 归档原因

这些代码是项目早期实验阶段的产物（预测编码框架 Stages 11-100），
与当前的自主进化Agent系统（stage_a ~ stage_k）架构无关。
保留它们供历史参考，但不参与系统运行。

## 目录结构

```
archive/
├── README.md           # 本说明文件
├── legacy_core/        # 历史预测编码框架代码
│   └── core/           # Stages 11-100 的实现
│       ├── stage_011.py ~ stage_100.py
│       └── *_v2.py     # 各阶段的v2改进版
└── legacy_tests/       # 历史测试文件
    ├── test_core_pipeline.py      # 核心管道测试
    ├── test_stage_011.py          # Stage 11 单元测试
    ├── test_stress_scenarios.py   # 压力测试
    └── test_v2_comprehensive.py   # v2综合测试
```

## 与主系统的关系

主系统（`src/agent_whitepaper/stage_a/` ~ `stage_k/`）完全不依赖这些归档代码。
归档代码仅依赖自身的 `core/` 包内部模块。

## 如需恢复

这些代码通过 `git mv` 移动，完整历史保留在 git 中。
如需恢复，使用：
```bash
git mv archive/legacy_core/core src/agent_whitepaper/core
git mv archive/legacy_tests/test_*.py src/agent_whitepaper/tests/
```
