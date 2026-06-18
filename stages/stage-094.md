# 阶段94：七大模块集成

> **阶段编号**：S94 | **所属层**：第十层·系统集成层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 模块1-环境感知 | "眼睛耳朵"——感知环境，构建内部表征 |
> | 模块2-驱力引擎 | "饥渴驱动"——产生内在动机，驱动行为 |
> | 模块3-自主目标生成 | "自己定目标"——从驱力缺口生成任务 |
> | 模块4-多层预测模型 | "三层预言机"——低中高三层预测 |
> | 模块5-误差评估反馈 | "裁判吹哨"——评估预测误差，路由反馈 |
> | 模块6-双模式模型优化 | "两种修法"——微调和重构两种进化方式 |
> | 模块7-记忆与离线复盘 | "日记本与反思"——记录经验，离线总结 |
> | 数据流图 | "信息流水线"——感知→预测→评估→决策→执行→学习→整合 |
> | IntegratedSystem | "集成系统的完整画像"——组合MainScheduler，在调度器基础上增加模块集成状态 |

---

## 1. 第一性原理依据

本阶段从阶段1-90的所有模块接口出发，建立七大模块集成。核心依据为：

- **A31.x（环境感知模块）**（引用阶段31）：EnvironmentStateVector定义了环境感知的输出接口。模块1-环境感知将阶段11-31的所有感知能力封装为统一接口。

- **A27.x（驱力引擎模块）**（引用阶段27）：DriveEngineState定义了驱力引擎的输出接口。模块2-驱力引擎将阶段21-30的所有驱力计算封装为统一接口。

- **A40.x（自主目标生成模块）**（引用阶段40）：DecisionEngineState定义了决策引擎的输出接口。模块3-自主目标生成将阶段31-40的所有决策能力封装为统一接口。

- **A50.x（多层预测模型模块）**（引用阶段50）：CrossLayerProtocol定义了跨层交互的输出接口。模块4-多层预测模型将阶段41-50的所有预测能力封装为统一接口。

- **A59.x（误差评估反馈模块）**（引用阶段59）：ErrorEvaluatorState定义了误差评估的输出接口。模块5-误差评估反馈将阶段51-60的所有评估能力封装为统一接口。

- **A70.x（双模式模型优化模块）**（引用阶段70）：EvolutionValidationResult定义了进化验证的输出接口。模块6-双模式模型优化将阶段61-70的所有进化能力封装为统一接口。

- **A79.x（记忆与离线复盘模块）**（引用阶段79）：MemoryManagerState定义了记忆管理的输出接口。模块7-记忆与离线复盘将阶段71-80的所有记忆能力封装为统一接口。

本阶段的核心目标是：**定义模块1-环境感知 → EnvironmentStateVector（A94.1），模块2-驱力引擎 → DriveEngineState（A94.2），模块3-自主目标生成 → DecisionEngineState（A94.3），模块4-多层预测模型 → CrossLayerProtocol（A94.4），模块5-误差评估反馈 → ErrorEvaluatorState（A94.5），模块6-双模式模型优化 → EvolutionValidationResult（A94.6），模块7-记忆与离线复盘 → MemoryManagerState（A94.7），数据流图 perceive → predict → evaluate → decide → execute → learn → consolidate（A94.8），封装为IntegratedSystem数据类组合MainScheduler。**

> **去重声明**：本阶段仅定义七大模块的集成接口和数据流图，不重新实现任何模块的内部逻辑。所有模块均通过引用阶段31/27/40/50/59/70/79的接口接入。

---

## 2. 核心数学公式

### 第一部分至第七部分：七大模块接口

#### 2.1 模块1-环境感知（A94.1）

$$\text{Module}_{perception}: o_t \to \text{EnvironmentStateVector} \tag{A94.1}$$

**接口定义**：

| 方法 | 输入 | 输出 | 引用 |
|------|------|------|------|
| perceive() | 观测 $o_t$ | EnvironmentStateVector | 阶段31 |
| update_representation() | 误差信号 | 更新后的表征 | 阶段14 |
| get_prediction() | 表征 $\mu$ | 预测 $\hat{o}$ | 阶段12 |

**模块1的内部组成**：

| 子模块 | 阶段 | 功能 |
|--------|------|------|
| 层级生成模型 | 11 | 构建多层生成模型 |
| 自下而上传播 | 12 | 误差信号向上传播 |
| 自上而下传播 | 13 | 预测信号向下传播 |
| 感知推断 | 14 | 最优表征推断 |
| 表征更新 | 14 | 表征梯度更新 |
| 环境状态向量 | 31 | 统一环境表征 |

- **输入**：环境观测 $o_t$
- **输出**：EnvironmentStateVector
- **约束**：感知延迟 ≤ 10ms

> **白话翻译**：模块1是"眼睛耳朵"——看环境、听信息，构建内部表征。包含层级生成模型、误差传播、预测传播、感知推断等子模块，最终输出统一的环境状态向量。

---

#### 2.2 模块2-驱力引擎（A94.2）

$$\text{Module}_{drive}: \text{EnvironmentStateVector} \to \text{DriveEngineState} \tag{A94.2}$$

**接口定义**：

| 方法 | 输入 | 输出 | 引用 |
|------|------|------|------|
| compute_drives() | 环境状态 | DriveEngineState | 阶段27 |
| get_drive_gaps() | 驱力状态 | 缺口向量 | 阶段28 |
| get_drive_rewards() | 驱力状态 | 奖励信号 | 阶段29 |

**模块2的内部组成**：

| 子模块 | 阶段 | 功能 |
|--------|------|------|
| 自由能分解 | 21 | 精度驱力+广度驱力+能效驱力 |
| 好奇心驱力 | 22 | 信息增益驱动探索 |
| 收益驱力 | 22 | 预期收益驱动利用 |
| 能效驱力 | 23 | 资源效率驱动优化 |
| 驱力优先级 | 25 | 生存>精度>广度>能效 |
| 驱力引擎 | 27 | 统一驱力计算 |

- **输入**：EnvironmentStateVector
- **输出**：DriveEngineState
- **约束**：驱力缺口永不为零（永续性保证）

> **白话翻译**：模块2是"饥渴驱动"——根据环境状态计算三大驱力（精度、广度、能效），产生内在动机。驱力缺口永远不为零，确保系统永远有事做。

---

#### 2.3 模块3-自主目标生成（A94.3）

$$\text{Module}_{decision}: \text{DriveEngineState} \to \text{DecisionEngineState} \tag{A94.3}$$

**接口定义**：

| 方法 | 输入 | 输出 | 引用 |
|------|------|------|------|
| generate_tasks() | 驱力缺口 | 任务列表 | 阶段38 |
| select_task() | 任务列表 | 选定任务 | 阶段39 |
| decide_action() | 选定任务 | 行动 | 阶段40 |

**模块3的内部组成**：

| 子模块 | 阶段 | 功能 |
|--------|------|------|
| 探索策略 | 36 | 信息增益最大化 |
| 利用策略 | 37 | 预期收益最大化 |
| 任务生成 | 38 | 从驱力缺口生成任务 |
| 任务调度 | 39 | 优先级排序和中断 |
| 决策引擎 | 40 | 统一决策接口 |

- **输入**：DriveEngineState
- **输出**：DecisionEngineState
- **约束**：100%自主决策，无外部指令

> **白话翻译**：模块3是"自己定目标"——从驱力缺口生成任务，选择最该做的事，决定怎么行动。完全自主，不需要外部指令。

---

#### 2.4 模块4-多层预测模型（A94.4）

$$\text{Module}_{predictor}: (\mu, o) \to \text{CrossLayerProtocol} \tag{A94.4}$$

**接口定义**：

| 方法 | 输入 | 输出 | 引用 |
|------|------|------|------|
| predict() | 表征 $\mu$ | 预测 $\hat{o}$ | 阶段46-48 |
| update() | 误差信号 | 更新后的模型 | 阶段49 |
| cross_layer_communicate() | 层间信号 | 跨层协议 | 阶段50 |

**模块4的内部组成**：

| 子模块 | 阶段 | 功能 |
|--------|------|------|
| 低层预测器 | 41-44 | 快速感知预测 |
| 中层预测器 | 45-47 | 模式识别预测 |
| 高层预测器 | 48-49 | 抽象概念预测 |
| 跨层协议 | 50 | 层间信息交换 |

- **输入**：表征 $\mu$、观测 $o$
- **输出**：CrossLayerProtocol
- **约束**：三层预测一致性

> **白话翻译**：模块4是"三层预言机"——低层快速感知、中层模式识别、高层抽象概念，三层互相交流，形成统一的预测。

---

#### 2.5 模块5-误差评估反馈（A94.5）

$$\text{Module}_{evaluator}: (\hat{o}, o) \to \text{ErrorEvaluatorState} \tag{A94.5}$$

**接口定义**：

| 方法 | 输入 | 输出 | 引用 |
|------|------|------|------|
| evaluate() | 预测 $\hat{o}$, 观测 $o$ | ErrorEvaluatorState | 阶段59 |
| detect_anomaly() | 误差序列 | 异常标志 | 阶段56 |
| route_feedback() | 误差评估 | 反馈路由 | 阶段57 |

**模块5的内部组成**：

| 子模块 | 阶段 | 功能 |
|--------|------|------|
| 误差标准化 | 51 | 统一误差度量 |
| 误差分解 | 52 | 层级+维度分解 |
| 误差累积 | 53 | 时间窗口累积 |
| 精度加权 | 54 | 注意力加权 |
| 异常检测 | 56 | 统计异常检测 |
| 反馈路由 | 57 | 误差→学习/进化 |
| 误差评估器 | 59 | 统一评估接口 |

- **输入**：预测 $\hat{o}$、观测 $o$
- **输出**：ErrorEvaluatorState
- **约束**：异常检测灵敏度 ≥ 95%

> **白话翻译**：模块5是"裁判吹哨"——比较预测和实际，评估误差，检测异常，把反馈路由到该去的地方（学习或进化）。

---

#### 2.6 模块6-双模式模型优化（A94.6）

$$\text{Module}_{evolution}: \text{ErrorEvaluatorState} \to \text{EvolutionValidationResult} \tag{A94.6}$$

**接口定义**：

| 方法 | 输入 | 输出 | 引用 |
|------|------|------|------|
| micro_adjust() | 误差信号 | 微调结果 | 阶段63 |
| structural_reform() | 长期误差 | 重构结果 | 阶段67 |
| validate() | 新旧模型 | 验证结果 | 阶段70 |

**模块6的内部组成**：

| 子模块 | 阶段 | 功能 |
|--------|------|------|
| 微修正器 | 61-64 | 梯度下降微调 |
| 结构性重构 | 65-69 | 网络结构进化 |
| 进化验证 | 70 | 重构后性能验证 |

- **输入**：ErrorEvaluatorState
- **输出**：EvolutionValidationResult
- **约束**：性能倒退>5%自动回滚

> **白话翻译**：模块6是"两种修法"——小毛病微调（梯度下降），大问题重构（网络进化）。重构后如果性能倒退超过5%，自动回滚到旧模型。

---

#### 2.7 模块7-记忆与离线复盘（A94.7）

$$\text{Module}_{memory}: \text{experience} \to \text{MemoryManagerState} \tag{A94.7}$$

**接口定义**：

| 方法 | 输入 | 输出 | 引用 |
|------|------|------|------|
| store() | 经验 | 存储确认 | 阶段76 |
| retrieve() | 查询条件 | 相关经验 | 阶段77 |
| replay() | 回放条件 | 回放结果 | 阶段79 |

**模块7的内部组成**：

| 子模块 | 阶段 | 功能 |
|--------|------|------|
| 短期记忆 | 71-73 | 工作记忆和经验缓存 |
| 长期记忆 | 74-76 | 持久化存储和检索 |
| 快照管理 | 78 | 模型快照和回滚 |
| 离线复盘 | 79 | 经验回放和整合 |

- **输入**：经验数据
- **输出**：MemoryManagerState
- **约束**：磁盘占用 < 80%

> **白话翻译**：模块7是"日记本与反思"——记录经验（短期+长期记忆），保存模型快照，空闲时翻看日记反思（离线复盘）。

---

### 第八部分：数据流图

#### 2.8 数据流图定义（A94.8）

七大模块通过数据流图串联为统一流水线：

$$\text{perceive} \to \text{predict} \to \text{evaluate} \to \text{decide} \to \text{execute} \to \text{learn} \to \text{consolidate} \tag{A94.8}$$

**数据流图的详细连接**：

```
感知(perceive)
  │ Module1: o_t → EnvironmentStateVector
  │ 输出: 表征μ, 预测ô, 环境状态
  ▼
预测(predict)
  │ Module4: μ → CrossLayerProtocol
  │ 输出: 三层预测, 跨层信号
  ▼
评估(evaluate)
  │ Module5: (ô, o) → ErrorEvaluatorState
  │ 输出: 误差评估, 异常标志, 反馈路由
  ▼
决策(decide)
  │ Module2: EnvironmentStateVector → DriveEngineState
  │ Module3: DriveEngineState → DecisionEngineState
  │ 输出: 驱力状态, 选定任务, 行动
  ▼
执行(execute)
  │ 行动a → 环境 → 新观测o_{t+1}
  │ 输出: 新观测, 执行结果
  ▼
学习(learn)
  │ Module6: ErrorEvaluatorState → EvolutionValidationResult
  │ 输出: 微调/重构结果, 验证结果
  ▼
整合(consolidate)
  │ Module7: experience → MemoryManagerState
  │ 输出: 记忆存储, 经验回放
  │
  └──→ 回到感知(perceive)，下一轮循环
```

**数据流图的关键数据类型**：

| 流转环节 | 数据类型 | 来源模块 | 目标模块 |
|---------|---------|---------|---------|
| perceive→predict | EnvironmentStateVector | 模块1 | 模块4 |
| predict→evaluate | CrossLayerProtocol | 模块4 | 模块5 |
| evaluate→decide | ErrorEvaluatorState | 模块5 | 模块2+3 |
| decide→execute | DecisionEngineState | 模块3 | 环境 |
| execute→learn | 执行结果 | 环境 | 模块6 |
| learn→consolidate | EvolutionValidationResult | 模块6 | 模块7 |
| consolidate→perceive | MemoryManagerState | 模块7 | 模块1 |

**数据流图的旁路连接**：

除主流水线外，还有以下旁路连接：

| 旁路 | 说明 |
|------|------|
| 评估→学习 | 误差直接驱动微修正（不经过决策） |
| 评估→进化 | 长期误差驱动结构性重构 |
| 记忆→预测 | 经验回放辅助预测 |
| 安全→执行 | 安全护栏拦截危险行动 |

- **输入**：环境观测 $o_t$
- **输出**：行动 $a_t$、更新后的全系统状态
- **约束**：数据流完整闭环；无断链

> **白话翻译**：数据流图就是"信息流水线"——看环境→做预测→评误差→定决策→动手做→学经验→存记忆，然后回到看环境，循环往复。每一步的输出就是下一步的输入，形成完整闭环。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A94.1 | Module_perception: o → EnvironmentStateVector | 观测 | 环境状态 | 延迟≤10ms |
| A94.2 | Module_drive: EnvState → DriveEngineState | 环境状态 | 驱力状态 | 缺口永不为零 |
| A94.3 | Module_decision: DriveState → DecisionEngineState | 驱力状态 | 决策状态 | 100%自主 |
| A94.4 | Module_predictor: (μ, o) → CrossLayerProtocol | 表征+观测 | 跨层协议 | 三层一致 |
| A94.5 | Module_evaluator: (ô, o) → ErrorEvaluatorState | 预测+观测 | 误差状态 | 异常检测≥95% |
| A94.6 | Module_evolution: ErrorState → EvolutionValidationResult | 误差状态 | 进化结果 | 倒退>5%回滚 |
| A94.7 | Module_memory: experience → MemoryManagerState | 经验 | 记忆状态 | 磁盘<80% |
| A94.8 | perceive→predict→evaluate→decide→execute→learn→consolidate | 全状态 | 全状态 | 闭环无断链 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np

# 引用阶段93的数据类
from stage_093 import MainScheduler

# 引用七大模块的接口类型
from stage_031 import EnvironmentStateVector
from stage_027 import DriveEngineState
from stage_040 import DecisionEngineState
from stage_050 import CrossLayerProtocol
from stage_059 import ErrorEvaluatorState
from stage_070 import EvolutionValidationResult
from stage_079 import MemoryManagerState

# 引用安全护栏
from stage_090 import SafetyGuard, SafetyGuardState


# ============================================================
# 阶段94 数据类：集成系统
# ============================================================
@dataclass(frozen=True)
class IntegratedSystem:
    """七大模块集成的完整状态描述。

    组合阶段93的MainScheduler，在调度器基础上增加模块集成状态。

    Attributes:
        main_scheduler: 阶段93的主循环调度器（组合字段）
        environment_state: 模块1-环境感知状态
        drive_state: 模块2-驱力引擎状态
        decision_state: 模块3-自主目标生成状态
        predictor_state: 模块4-多层预测模型状态
        evaluator_state: 模块5-误差评估反馈状态
        evolution_state: 模块6-双模式模型优化状态
        memory_state: 模块7-记忆与离线复盘状态
        safety_state: 安全护栏状态
        current_observation: 当前观测
        current_action: 当前行动
        step_count: 总步数
        uptime_seconds: 运行时间

    Invariants:
        - step_count >= 0
        - uptime_seconds >= 0
        - 所有模块状态非空
    """
    main_scheduler: MainScheduler
    environment_state: EnvironmentStateVector
    drive_state: DriveEngineState
    decision_state: DecisionEngineState
    predictor_state: CrossLayerProtocol
    evaluator_state: ErrorEvaluatorState
    evolution_state: EvolutionValidationResult
    memory_state: MemoryManagerState
    safety_state: SafetyGuardState
    current_observation: np.ndarray
    current_action: np.ndarray
    step_count: int = 0
    uptime_seconds: float = 0.0


# ============================================================
# 阶段94 核心接口：集成系统
# ============================================================
class AgentSystem:
    """阶段94的核心接口：七大模块集成的智能体系统。

    实现数据流图 perceive → predict → evaluate → decide →
    execute → learn → consolidate 的完整闭环。

    核心约束：
        - 数据流完整闭环
        - 所有模块正确对接
        - 安全护栏贯穿全流程
    """

    def __init__(
        self,
        scheduler: MainScheduler,
        safety_guard: SafetyGuard,
        config: Optional[Dict] = None
    ):
        """初始化集成系统。

        Args:
            scheduler: 主循环调度器
            safety_guard: 安全护栏
            config: 系统配置字典

        Examples:
            >>> from stage_093 import MainScheduler
            >>> from stage_092 import GlobalDynamicsCalculator
            >>> calc = GlobalDynamicsCalculator()
            >>> scheduler = MainScheduler(global_dynamics=None)
            >>> sg = SafetyGuardExecutor().check_all()
            >>> integrator = SystemIntegrator(scheduler, sg)
            >>> integrator.config is not None
            True
        """
        self.scheduler = scheduler
        self.safety_guard = safety_guard
        self.config = config or {}
        self._step_count = 0
        self._last_objective = 0.0

    def perceive(self, observation: np.ndarray) -> EnvironmentStateVector:
        """模块1：环境感知（A94.1）。

        Args:
            observation: 环境观测，shape (d_obs,)

        Returns:
            环境状态向量

        Examples:
            >>> from stage_093 import MainScheduler
            >>> scheduler = MainScheduler(global_dynamics=None)
            >>> sg = SafetyGuardExecutor().check_all()
            >>> integrator = SystemIntegrator(scheduler, sg)
            >>> state = integrator.perceive(np.array([1.0, 0.5]))
            >>> state.observation.shape == (2,)
            True
        """
        return EnvironmentStateVector(
            observation=observation,
            context=np.zeros_like(observation),
            timestamp=0.0,
            layer=0
        )

    def predict(self, env_state: EnvironmentStateVector) -> CrossLayerProtocol:
        """模块4：多层预测（A94.4）。

        Args:
            env_state: 环境状态向量

        Returns:
            跨层协议（含三层预测）

        Examples:
            >>> from stage_093 import MainScheduler
            >>> scheduler = MainScheduler(global_dynamics=None)
            >>> sg = SafetyGuardExecutor().check_all()
            >>> integrator = SystemIntegrator(scheduler, sg)
            >>> env = integrator.perceive(np.array([1.0]))
            >>> pred = integrator.predict(env)
            >>> pred.surface_prediction.shape == pred.observation.shape
            True
        """
        obs = env_state.observation
        return CrossLayerProtocol(
            observation=obs,
            surface_prediction=obs * 0.9,
            middle_prediction=obs * 0.8,
            ultimate_prediction=obs * 0.7,
            confidence=0.9,
            layer=env_state.layer,
            timestamp=env_state.timestamp
        )

    def evaluate(
        self,
        prediction: CrossLayerProtocol,
        observation: np.ndarray
    ) -> ErrorEvaluatorState:
        """模块5：误差评估（A94.5）。

        Args:
            prediction: 跨层协议
            observation: 实际观测，shape (d_obs,)

        Returns:
            误差评估状态

        Examples:
            >>> from stage_093 import MainScheduler
            >>> scheduler = MainScheduler(global_dynamics=None)
            >>> sg = SafetyGuardExecutor().check_all()
            >>> integrator = SystemIntegrator(scheduler, sg)
            >>> env = integrator.perceive(np.array([1.0]))
            >>> pred = integrator.predict(env)
            >>> eval_state = integrator.evaluate(pred, np.array([1.0]))
            >>> eval_state.global_error >= 0
            True
        """
        error = float(np.mean((prediction.surface_prediction - observation) ** 2))
        return ErrorEvaluatorState(
            global_error=error,
            layer_errors=(error, error * 0.9, error * 0.8),
            error_history=(error,),
            prediction_error=error,
            complexity=1.0,
            free_energy=error + 1.0,
            timestamp=0.0
        )

    def decide(
        self,
        env_state: EnvironmentStateVector,
        evaluator_state: ErrorEvaluatorState
    ) -> DecisionEngineState:
        """模块2+3：驱力计算与决策（A94.2+A94.3）。

        Args:
            env_state: 环境状态向量
            evaluator_state: 误差评估状态

        Returns:
            决策引擎状态

        Examples:
            >>> from stage_093 import MainScheduler
            >>> scheduler = MainScheduler(global_dynamics=None)
            >>> sg = SafetyGuardExecutor().check_all()
            >>> integrator = SystemIntegrator(scheduler, sg)
            >>> env = integrator.perceive(np.array([1.0]))
            >>> pred = integrator.predict(env)
            >>> ev = integrator.evaluate(pred, np.array([1.0]))
            >>> decision = integrator.decide(env, ev)
            >>> decision.selected_action >= 0
            True
        """
        return DecisionEngineState(
            drive_values=(evaluator_state.global_error,),
            drive_weights=(1.0,),
            action_values=(0.5,),
            action_probabilities=(1.0,),
            selected_action=0,
            expected_free_energy=evaluator_state.free_energy,
            confidence=0.9,
            explore_weight=0.3,
            timestamp=0.0
        )

    def execute(self, decision: DecisionEngineState) -> np.ndarray:
        """执行行动。

        Args:
            decision: 决策引擎状态

        Returns:
            执行的行动向量

        Examples:
            >>> from stage_093 import MainScheduler
            >>> scheduler = MainScheduler(global_dynamics=None)
            >>> sg = SafetyGuardExecutor().check_all()
            >>> integrator = SystemIntegrator(scheduler, sg)
            >>> env = integrator.perceive(np.array([1.0]))
            >>> pred = integrator.predict(env)
            >>> ev = integrator.evaluate(pred, np.array([1.0]))
            >>> decision = integrator.decide(env, ev)
            >>> action = integrator.execute(decision)
            >>> action.shape[0] > 0
            True
        """
        return np.array([decision.action_values[decision.selected_action]])

    def learn(self, evaluator_state: ErrorEvaluatorState) -> EvolutionValidationResult:
        """模块6：学习与进化（A94.6）。

        Args:
            evaluator_state: 误差评估状态

        Returns:
            进化验证结果

        Examples:
            >>> from stage_093 import MainScheduler
            >>> scheduler = MainScheduler(global_dynamics=None)
            >>> sg = SafetyGuardExecutor().check_all()
            >>> integrator = SystemIntegrator(scheduler, sg)
            >>> env = integrator.perceive(np.array([1.0]))
            >>> pred = integrator.predict(env)
            >>> ev = integrator.evaluate(pred, np.array([1.0]))
            >>> result = integrator.learn(ev)
            >>> result.is_valid in (True, False)
            True
        """
        return EvolutionValidationResult(
            is_valid=evaluator_state.global_error < 0.5,
            fitness=1.0 / (1.0 + evaluator_state.global_error),
            error=evaluator_state.global_error,
            complexity=evaluator_state.complexity,
            free_energy=evaluator_state.free_energy,
            timestamp=0.0
        )

    def consolidate(
        self,
        experience: Dict,
        evolution_result: EvolutionValidationResult
    ) -> MemoryManagerState:
        """模块7：记忆整合（A94.7）。

        Args:
            experience: 当前经验字典
            evolution_result: 进化结果

        Returns:
            记忆管理状态

        Examples:
            >>> from stage_093 import MainScheduler
            >>> scheduler = MainScheduler(global_dynamics=None)
            >>> sg = SafetyGuardExecutor().check_all()
            >>> integrator = SystemIntegrator(scheduler, sg)
            >>> result = integrator.learn(
            ...     ErrorEvaluatorState(global_error=0.1))
            >>> mem_state = integrator.consolidate({}, result)
            >>> mem_state.current_step >= 0
            True
        """
        return MemoryManagerState(
            model_snapshot=ModelSnapshot(
                consolidation=MemoryConsolidation(
                    offline_replay=OfflineReplay(
                        generative_replay=GenerativeReplay(
                            replay=PriorityReplay(
                                evaluator=SampleValueEvaluator(
                                    ltm=LongTermMemory(
                                        stm=ShortTermMemory(
                                            validation_result=evolution_result,
                                            db_path=":memory:"
                                        ),
                                        db_path=":memory:"
                                    ),
                                    error_mean=0.0, error_var=1.0,
                                    alpha=0.4, beta=0.4, gamma=0.2,
                                    forget_factor=0.99, top_k_ratio=0.3
                                ),
                                alpha=1.0, beta_start=0.4,
                                beta_end=1.0, beta_anneal_steps=10000,
                                batch_size=32
                            ),
                            vae_latent_dim=32,
                            quality_threshold=0.1,
                            avg_reconstruction_error=0.0
                        ),
                        batch_size=128,
                        min_improvement=0.001,
                        max_rounds=10,
                        debounce_steps=5,
                        idle_threshold=10.0
                    ),
                    consolidate_threshold=1000.0,
                    forget_tau_base=86400.0,
                    spacing_factor=2.0,
                    retention_threshold=0.85
                ),
                snapshot_interval=10000,
                max_snapshots=10,
                max_chain_length=5
            ),
            total_samples_stored=0,
            total_samples_consolidated=0,
            total_replays=0,
            total_snapshots=0,
            total_rollbacks=0,
            current_step=self._step_count,
            is_idle=False,
            last_consolidate_step=0,
            last_snapshot_step=0
        )

    def step(self, observation: np.ndarray) -> Dict:
        """执行一步完整数据流（A94.8）。

        perceive → predict → evaluate → decide →
        execute → learn → consolidate

        Args:
            observation: 环境观测，shape (d_obs,)

        Returns:
            包含行动和全系统状态的字典

        Examples:
            >>> from stage_093 import MainScheduler
            >>> scheduler = MainScheduler(global_dynamics=None)
            >>> sg = SafetyGuardExecutor().check_all()
            >>> integrator = SystemIntegrator(scheduler, sg)
            >>> result = integrator.step(np.array([1.0, 0.5]))
            >>> "action" in result
            True
            >>> "objective" in result
            True
        """
        self._step_count += 1
        env_state = self.perceive(observation)
        prediction = self.predict(env_state)
        evaluator_state = self.evaluate(prediction, observation)
        decision = self.decide(env_state, evaluator_state)
        action = self.execute(decision)
        evolution_result = self.learn(evaluator_state)
        mem_state = self.consolidate({}, evolution_result)
        self._last_objective = evaluator_state.free_energy
        return {
            "action": action,
            "decision": decision,
            "evaluator_state": evaluator_state,
            "evolution_result": evolution_result,
            "memory_state": mem_state,
            "objective": self._last_objective,
            "step_count": self._step_count
        }

    def get_state(self) -> IntegratedSystem:
        """获取集成系统的完整状态。

        Returns:
            IntegratedSystem 完整状态

        Examples:
            >>> from stage_093 import MainScheduler
            >>> scheduler = MainScheduler(global_dynamics=None)
            >>> sg = SafetyGuardExecutor().check_all()
            >>> integrator = SystemIntegrator(scheduler, sg)
            >>> _ = integrator.step(np.array([1.0]))
            >>> state = integrator.get_state()
            >>> isinstance(state, IntegratedSystem)
            True
        """
        return IntegratedSystem(
            scheduler=self.scheduler,
            safety_guard=self.safety_guard,
            config=self.config,
            step_count=self._step_count,
            last_objective=self._last_objective,
            is_running=True,
            is_paused=False
        )

    def get_objective(self) -> float:
        """获取当前全局目标值。

        Returns:
            全局目标 J

        Examples:
            >>> from stage_093 import MainScheduler
            >>> scheduler = MainScheduler(global_dynamics=None)
            >>> sg = SafetyGuardExecutor().check_all()
            >>> integrator = SystemIntegrator(scheduler, sg)
            >>> _ = integrator.step(np.array([1.0]))
            >>> obj = integrator.get_objective()
            >>> obj >= 0
            True
        """
        return self._last_objective
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `MainScheduler` | ← 阶段93 | 作为集成系统的调度核心（组合字段） |
| **输入** | `EnvironmentStateVector` | ← 阶段31 | 模块1的输出接口 |
| **输入** | `DriveEngineState` | ← 阶段27 | 模块2的输出接口 |
| **输入** | `DecisionEngineState` | ← 阶段40 | 模块3的输出接口 |
| **输入** | `CrossLayerProtocol` | ← 阶段50 | 模块4的输出接口 |
| **输入** | `ErrorEvaluatorState` | ← 阶段59 | 模块5的输出接口 |
| **输入** | `EvolutionValidationResult` | ← 阶段70 | 模块6的输出接口 |
| **输入** | `MemoryManagerState` | ← 阶段79 | 模块7的输出接口 |
| **输入** | `SafetyGuardState` | ← 阶段90 | 安全护栏状态 |
| **输出** | `IntegratedSystem` | → 阶段95-100 | 七大模块集成的完整系统 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 核心计算 | 所有模块的数值计算 |
| asyncio | 异步调度 | 主循环的异步执行 |
| psutil | 资源监控 | 安全护栏的数据来源 |
| SQLite | 持久化 | 记忆模块的存储后端 |
| NumPy | 数据转换 | 模块间的数据格式转换 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V94.1 | 模块1对接 | perceive()返回EnvironmentStateVector | 接口测试 |
| V94.2 | 模块2对接 | compute_drives()返回DriveEngineState | 接口测试 |
| V94.3 | 模块3对接 | decide()返回DecisionEngineState | 接口测试 |
| V94.4 | 模块4对接 | predict()返回CrossLayerProtocol | 接口测试 |
| V94.5 | 模块5对接 | evaluate()返回ErrorEvaluatorState | 接口测试 |
| V94.6 | 模块6对接 | learn()返回EvolutionValidationResult | 接口测试 |
| V94.7 | 模块7对接 | consolidate()返回MemoryManagerState | 接口测试 |
| V94.8 | 数据流闭环 | perceive→...→consolidate→perceive | 流程测试 |
| V94.9 | 数据类型匹配 | 每步输出类型=下一步输入类型 | 类型测试 |
| V94.10 | 安全护栏贯穿 | 每步都经过安全检查 | 安全测试 |
| V94.11 | 旁路连接 | 评估→学习/进化旁路正常 | 旁路测试 |
| V94.12 | 1000步无异常 | step()连续1000步无报错 | 稳定性测试 |
| V94.13 | IntegratedSystem组合 | `state.main_scheduler` 类型为 `MainScheduler` | 类型检查 |
| V94.14 | 全局目标可获取 | get_objective()返回有效值 | 目标测试 |
| V94.15 | 模块独立性 | 单个模块可独立测试 | 解耦测试 |
| V94.16 | 资源约束 | CPU≤20%, 内存≤20% | 资源测试 |
| V94.17 | 驱力缺口非零 | drive_gaps永不为零向量 | 永续性测试 |
| V94.18 | 自主决策 | 100%决策由驱力驱动 | 自驱性测试 |

---

> **阶段94完** | 阶段94建立了七大模块集成：模块1-环境感知 → EnvironmentStateVector（A94.1），模块2-驱力引擎 → DriveEngineState（A94.2），模块3-自主目标生成 → DecisionEngineState（A94.3），模块4-多层预测模型 → CrossLayerProtocol（A94.4），模块5-误差评估反馈 → ErrorEvaluatorState（A94.5），模块6-双模式模型优化 → EvolutionValidationResult（A94.6），模块7-记忆与离线复盘 → MemoryManagerState（A94.7），数据流图 perceive → predict → evaluate → decide → execute → learn → consolidate（A94.8），封装为IntegratedSystem数据类组合MainScheduler。七大模块通过数据流图串联为完整闭环，从感知到记忆，从驱力到进化，从评估到决策，形成自驱永续进化的统一系统。下一步：阶段95将建立系统启动与初始化流程。
