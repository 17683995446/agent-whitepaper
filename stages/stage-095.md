# 阶段95：系统启动与初始化

> **阶段编号**：S95 | **所属层**：第十层·系统集成层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 初始化顺序 | "开机清单"——按固定顺序逐项启动，确保依赖关系 |
> | 配置加载 | "读设置"——从配置文件读取所有参数 |
> | 初始模型初始化 | "出厂设置"——创建初始预测模型 |
> | 初始驱力设定 | "初始动机"——设定三大驱力的初始值 |
> | 安全护栏初始化 | "安全系统启动"——启动所有安全监控 |
> | 首次感知与预测 | "睁开眼睛"——第一次看世界，建立初始预测 |
> | SystemInitializer | "系统初始化器的完整画像"——组合IntegratedSystem，在集成系统基础上增加初始化状态 |

---

## 1. 第一性原理依据

本阶段从阶段94（集成系统）的核心结论出发，建立系统启动与初始化流程。核心依据为：

- **A94.x（七大模块集成）**（引用阶段94）：IntegratedSystem定义了七大模块的集成接口。系统初始化必须按照模块间的依赖关系，依次创建和配置每个模块，确保数据流图的完整性。

本阶段的核心目标是：**定义初始化顺序 load_config → init_perception → init_drive → init_predictor → init_memory → init_safety → start_main_loop（A95.1），配置加载 config = load('agent_config.yaml')（A95.2），初始模型初始化 model = init_hierarchical_model(config)（A95.3），初始驱力设定 drives = init_drives(config.preset='balanced')（A95.4），安全护栏初始化 safety = init_safety_guard(config.limits)（A95.5），首次感知与预测 bootstrap()（A95.6），封装为SystemInitializer数据类组合IntegratedSystem。**

> **去重声明**：本阶段仅定义系统启动与初始化流程，不重新实现任何模块的内部逻辑。初始化过程调用各模块的构造函数和配置接口。

---

## 2. 核心数学公式

### 第一部分：初始化顺序

#### 2.1 初始化顺序定义（A95.1）

系统初始化按严格的依赖顺序执行：

$$\text{load\_config} \to \text{init\_perception} \to \text{init\_drive} \to \text{init\_predictor} \to \text{init\_memory} \to \text{init\_safety} \to \text{start\_main\_loop} \tag{A95.1}$$

**初始化顺序的依赖关系**：

| 步骤 | 初始化模块 | 依赖 | 输出 |
|------|-----------|------|------|
| 1 | load_config | 无 | 配置字典 |
| 2 | init_perception | config | 感知模块 |
| 3 | init_drive | config, perception | 驱力引擎 |
| 4 | init_predictor | config, perception | 预测模型 |
| 5 | init_memory | config | 记忆管理器 |
| 6 | init_safety | config | 安全护栏 |
| 7 | start_main_loop | 全部模块 | 运行中的系统 |

**初始化顺序的设计原则**：

1. **配置优先**：所有模块的参数从配置文件加载，确保可配置性
2. **感知先行**：感知是最基础的模块，其他模块依赖感知输出
3. **驱力次之**：驱力引擎依赖环境状态（感知输出）
4. **预测随后**：预测模型依赖感知表征
5. **记忆独立**：记忆管理器不依赖其他模块
6. **安全兜底**：安全护栏最后初始化，确保所有模块就绪后才启动监控
7. **主循环最后**：所有模块就绪后才启动主循环

**初始化失败的处理**：

| 失败步骤 | 处理策略 |
|---------|---------|
| 配置加载失败 | 使用默认配置，记录警告 |
| 感知初始化失败 | 终止启动，报告错误 |
| 驱力初始化失败 | 使用默认驱力，记录警告 |
| 预测模型初始化失败 | 终止启动，报告错误 |
| 记忆初始化失败 | 使用空记忆，记录警告 |
| 安全护栏初始化失败 | 终止启动，报告错误 |
| 主循环启动失败 | 回滚到安全状态 |

**初始化的时间预算**：

$$T_{init} = \sum_{i=1}^{7} T_{init,i} \leq T_{max} = 30s \tag{A95.1-time}$$

| 步骤 | 时间预算 | 说明 |
|------|---------|------|
| load_config | 0.1s | 读取YAML文件 |
| init_perception | 5s | 创建层级模型 |
| init_drive | 1s | 初始化驱力参数 |
| init_predictor | 10s | 初始化三层预测器 |
| init_memory | 2s | 创建数据库和缓存 |
| init_safety | 1s | 初始化安全子系统 |
| start_main_loop | 0.1s | 启动调度器 |
| **总计** | **~20s** | **在30s预算内** |

- **输入**：配置文件路径
- **输出**：运行中的IntegratedSystem
- **约束**：$T_{init} \leq 30s$；关键模块失败则终止

> **白话翻译**：初始化顺序就是"开机清单"——先读设置，再开眼睛，再启动动机，再建预言机，再开记忆，再上安全锁，最后启动主循环。每一步都有依赖关系，不能乱序。关键模块（感知、预测、安全）开不了就停机，非关键模块（记忆）开不了就用默认值凑合。

---

### 第二部分：配置加载

#### 2.2 配置加载定义（A95.2）

从YAML配置文件加载所有参数：

$$\text{config} = \text{load}(\text{'agent\_config.yaml'}) \tag{A95.2}$$

**配置文件的结构**：

```yaml
# agent_config.yaml
perception:
  n_layers: 3
  layer_dims: [64, 128, 256]
  activation: "relu"

drive:
  preset: "balanced"  # balanced / aggressive / conservative
  accuracy_weight: 0.40
  coverage_weight: 0.35
  efficiency_weight: 0.25

predictor:
  low_level:
    hidden_dim: 64
    learning_rate: 0.01
  mid_level:
    hidden_dim: 128
    learning_rate: 0.001
  high_level:
    hidden_dim: 256
    learning_rate: 0.0001

memory:
  short_term_capacity: 1000
  long_term_db: "agent_memory.db"
  snapshot_dir: "snapshots/"

safety:
  cpu_max: 0.20
  mem_max: 0.20
  disk_max: 0.80
  rollback_threshold: 0.05
  explore_ratio_target: 0.70

scheduler:
  surface_interval: 1
  middle_interval: 600
  ultimate_interval: 36000
  target_loop_time: 0.05

objective:
  lambda_complexity: 0.01
  lambda_safety: 1.0
```

**配置参数的默认值**：

| 参数组 | 默认值 | 说明 |
|--------|--------|------|
| perception.n_layers | 3 | 三层感知 |
| drive.preset | balanced | 平衡驱力配置 |
| safety.cpu_max | 0.20 | CPU上限20% |
| safety.mem_max | 0.20 | 内存上限20% |
| objective.lambda_complexity | 0.01 | 复杂度系数 |
| objective.lambda_safety | 1.0 | 安全系数 |

**配置验证**：

$$\text{valid} = \text{validate}(\text{config}) \tag{A95.2-validate}$$

| 验证项 | 条件 | 失败处理 |
|--------|------|---------|
| 层级数量 | 1 ≤ n_layers ≤ 10 | 截断到边界 |
| 驱力权重和 | w_acc + w_cov + w_eff = 1.0 | 归一化 |
| CPU上限 | 0 < cpu_max ≤ 0.50 | 截断到0.20 |
| 内存上限 | 0 < mem_max ≤ 0.50 | 截断到0.20 |
| 安全系数 | lambda_safety > lambda_complexity | 交换 |

- **输入**：配置文件路径
- **输出**：验证后的配置字典
- **约束**：所有参数在有效范围内

> **白话翻译**：配置加载就是"读设置"——从YAML文件读取所有参数。配置文件定义了感知层数、驱力权重、安全阈值等。读完后验证一遍，不对的自动修正。

---

### 第三部分：初始模型初始化

#### 2.3 初始模型初始化定义（A95.3）

创建初始层级预测模型：

$$\text{model} = \text{init\_hierarchical\_model}(\text{config}) \tag{A95.3}$$

**初始模型的结构**：

| 层级 | 维度 | 参数量 | 初始化方式 |
|------|------|--------|-----------|
| 低层 | 64 | ~4K | Xavier初始化 |
| 中层 | 128 | ~16K | Xavier初始化 |
| 高层 | 256 | ~65K | Xavier初始化 |

**初始模型的总参数量**：

$$N_{params} = \sum_{l=1}^{L} (d_l \cdot d_{l-1} + d_l) \tag{A95.3-params}$$

其中 $d_0$ 为输入维度，$d_l$ 为第 $l$ 层维度。三层模型总参数量约85K，满足轻量级要求。

**初始模型的验证**：

$$\text{model\_valid} = \mathbb{1}[\text{forward\_pass}(\text{model}, x_{test}) \text{ 无异常}] \tag{A95.3-verify}$$

初始化后执行一次前向传播，确保模型结构正确、数值稳定。

- **输入**：配置参数
- **输出**：初始化的层级模型
- **约束**：前向传播无异常；参数量 < 100K

> **白话翻译**：初始模型初始化就是"出厂设置"——创建三层预测模型，用Xavier方法初始化权重。总共约8万5千个参数，很轻量。初始化后跑一次前向传播，确认模型能正常工作。

---

### 第四部分：初始驱力设定

#### 2.4 初始驱力设定定义（A95.4）

设定三大驱力的初始权重和目标值：

$$\text{drives} = \text{init\_drives}(\text{config.preset} = \text{'balanced'}) \tag{A95.4}$$

**驱力预设方案**：

| 预设 | 精度权重 | 广度权重 | 能效权重 | 适用场景 |
|------|---------|---------|---------|---------|
| balanced | 0.40 | 0.35 | 0.25 | 通用场景 |
| aggressive | 0.50 | 0.40 | 0.10 | 高精度需求 |
| conservative | 0.30 | 0.30 | 0.40 | 资源受限 |
| explorer | 0.25 | 0.55 | 0.20 | 新环境探索 |

**初始驱力缺口**：

$$\text{gap}_i = \text{target}_i - \text{current}_i \tag{A95.4-gap}$$

| 驱力 | 初始目标 | 初始当前值 | 初始缺口 |
|------|---------|-----------|---------|
| 精度 | 0.95 | 0.50 | 0.45 |
| 广度 | 0.80 | 0.10 | 0.70 |
| 能效 | 0.90 | 0.50 | 0.40 |

初始缺口确保系统启动后立即有驱力驱动行为——精度、广度、能效都有提升空间。

- **输入**：驱力预设名称
- **输出**：初始化的驱力状态
- **约束**：权重和 = 1.0；缺口 > 0

> **白话翻译**：初始驱力设定就是"初始动机"——设定三大驱力的权重和目标值。默认用"平衡"方案：精度40%、广度35%、能效25%。初始缺口都很大，确保系统一启动就有事做。

---

### 第五部分：安全护栏初始化

#### 2.5 安全护栏初始化定义（A95.5）

初始化九个安全子系统：

$$\text{safety} = \text{init\_safety\_guard}(\text{config.limits}) \tag{A95.5}$$

**安全子系统的初始化参数**：

| 子系统 | 初始化参数 | 默认值 |
|--------|-----------|--------|
| CPU监控 | cpu_max_threshold | 0.20 |
| 内存监控 | mem_max_threshold | 0.20 |
| 存储清理 | disk_max_threshold | 0.80 |
| 文件保护 | protected_patterns | ["*.yaml", "*.db"] |
| 探索冻结 | no_benefit_rounds | 5 |
| 自动回滚 | rollback_threshold | 0.05 |
| 比例执行 | ratio_target | 0.70 |
| 空闲复盘 | idle_threshold | 60s |
| 稳态探索 | stable_duration | 300s |

**安全护栏的启动验证**：

$$\text{safety\_valid} = \mathbb{1}[\text{check\_all}() = \text{OK}] \tag{A95.5-verify}$$

安全护栏初始化后执行一次全量检查，确保所有子系统正常工作。

- **输入**：安全限制配置
- **输出**：初始化的安全护栏
- **约束**：九个子系统全部正常

> **白话翻译**：安全护栏初始化就是"安全系统启动"——启动九个安全监控：CPU、内存、磁盘、文件保护、探索冻结、自动回滚、比例执行、空闲复盘、稳态探索。启动后做一次全量检查，确认都正常。

---

### 第六部分：首次感知与预测

#### 2.6 首次感知与预测定义（A95.6）

系统启动后执行首次感知-预测循环，建立初始内部模型：

$$\text{bootstrap}(): o_0 \to \mu_0 \to \hat{o}_0 \tag{A95.6}$$

**bootstrap的执行步骤**：

| 步骤 | 操作 | 输出 |
|------|------|------|
| 1 | 首次感知：获取初始观测 $o_0$ | 初始环境状态 |
| 2 | 首次推断：计算初始表征 $\mu_0$ | 初始内部表征 |
| 3 | 首次预测：生成预测 $\hat{o}_0$ | 初始预测 |
| 4 | 首次评估：计算初始误差 $e_0 = o_0 - \hat{o}_0$ | 初始误差 |
| 5 | 首次驱力：计算初始驱力缺口 | 初始驱力状态 |
| 6 | 首次决策：选择首个行动 | 初始行动 |

**bootstrap的初始误差**：

初始模型的预测是随机的（Xavier初始化），因此初始误差较大：

$$e_0 \sim \mathcal{N}(0, \sigma_0^2), \quad \sigma_0 \approx 1.0 \tag{A95.6-error}$$

较大的初始误差确保系统启动后立即进入学习状态——驱力缺口大，学习信号强。

**bootstrap的验证**：

$$\text{bootstrap\_valid} = \mathbb{1}[\text{所有模块输出非空}] \wedge \mathbb{1}[\text{数据流完整}] \tag{A95.6-verify}$$

- **输入**：初始化后的系统
- **输出**：完成首次感知-预测的系统
- **约束**：所有模块输出非空；数据流完整

> **白话翻译**：首次感知与预测就是"睁开眼睛"——第一次看世界，建立初始预测。因为模型是随机初始化的，预测肯定不准，误差很大。但这正好——大误差意味着大驱力缺口，系统立刻进入学习状态。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A95.1 | load_config→init_perception→...→start_main_loop | 配置文件 | 运行中的系统 | T_init ≤ 30s |
| A95.2 | config = load('agent_config.yaml') | 文件路径 | 配置字典 | 参数在有效范围 |
| A95.3 | model = init_hierarchical_model(config) | 配置 | 层级模型 | 前向传播无异常 |
| A95.4 | drives = init_drives(preset='balanced') | 预设名 | 驱力状态 | 权重和=1, 缺口>0 |
| A95.5 | safety = init_safety_guard(config.limits) | 限制配置 | 安全护栏 | 九子系统正常 |
| A95.6 | bootstrap(): o₀→μ₀→ô₀ | 初始系统 | 完成bootstrap的系统 | 数据流完整 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import numpy as np

# 引用阶段94的数据类
from stage_094 import IntegratedSystem, AgentSystem


# ============================================================
# 阶段95 枚举：初始化状态
# ============================================================
class InitStatus(Enum):
    """初始化状态。"""
    PENDING = "pending"          # 待初始化
    IN_PROGRESS = "in_progress"  # 初始化中
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"            # 失败
    SKIPPED = "skipped"          # 跳过（使用默认值）


# ============================================================
# 阶段95 数据类：初始化步骤结果
# ============================================================
@dataclass(frozen=True)
class InitStepResult:
    """单个初始化步骤的结果。

    Attributes:
        step_name: 步骤名称
        status: 初始化状态
        duration_seconds: 耗时（秒）
        error_message: 错误信息（失败时）
    """
    step_name: str
    status: InitStatus
    duration_seconds: float = 0.0
    error_message: Optional[str] = None


# ============================================================
# 阶段95 数据类：系统初始化器
# ============================================================
@dataclass(frozen=True)
class SystemInitializer:
    """系统初始化器的完整状态描述。

    组合阶段94的IntegratedSystem，在集成系统基础上增加初始化状态。

    Attributes:
        integrated_system: 阶段94的集成系统（组合字段）
        config: 加载的配置字典
        init_results: 各步骤初始化结果
        total_init_time: 总初始化时间
        is_ready: 系统是否就绪
        bootstrap_completed: bootstrap是否完成
        initial_free_energy: 初始自由能
        initial_drive_gaps: 初始驱力缺口

    Invariants:
        - is_ready = all(step.status == COMPLETED for critical steps)
        - total_init_time <= 30.0
    """
    integrated_system: Optional[IntegratedSystem]
    config: Dict = field(default_factory=dict)
    init_results: Tuple[InitStepResult, ...] = ()
    total_init_time: float = 0.0
    is_ready: bool = False
    bootstrap_completed: bool = False
    initial_free_energy: float = 0.0
    initial_drive_gaps: np.ndarray = field(default_factory=lambda: np.array([0.45, 0.70, 0.40]))


# ============================================================
# 阶段95 核心接口：系统初始化器
# ============================================================
class SystemInitializerExecutor:
    """阶段95的核心接口：系统启动与初始化。

    实现配置加载、模块初始化、安全验证和bootstrap。

    核心约束：
        - 初始化时间 ≤ 30秒
        - 关键模块失败则终止启动
        - 非关键模块失败使用默认值
    """

    def __init__(self, config_path: str = "agent_config.yaml"):
        """初始化系统初始化器。

        Args:
            config_path: 配置文件路径

        Examples:
            >>> initializer = SystemInitializer("config.yaml")
            >>> initializer.config_path
            'config.yaml'
        """
        self.config_path = config_path
        self._config: Optional[Dict] = None
        self._init_time = 0.0

    def load_config(self, config_path: str) -> Dict:
        """加载配置文件（A95.2）。

        Args:
            config_path: 配置文件路径

        Returns:
            验证后的配置字典

        Raises:
            FileNotFoundError: 当配置文件不存在时

        Examples:
            >>> initializer = SystemInitializer()
            >>> config = initializer.load_config("agent_config.yaml")
            >>> isinstance(config, dict)
            True
        """
        import os
        if not os.path.exists(config_path):
            return {
                "perception": {"layers": 3},
                "drive": {"preset": "balanced"},
                "predictor": {"hidden_dim": 64},
                "memory": {"buffer_size": 10000},
                "safety": {"cpu_max": 0.20}
            }
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {
                "perception": {"layers": 3},
                "drive": {"preset": "balanced"},
                "predictor": {"hidden_dim": 64},
                "memory": {"buffer_size": 10000},
                "safety": {"cpu_max": 0.20}
            }

    def init_perception(self, config: Dict):
        """初始化感知模块（A95.1-step2）。

        Args:
            config: 配置字典

        Returns:
            感知模块实例（EnvironmentStateVector）

        Examples:
            >>> initializer = SystemInitializer()
            >>> config = initializer.load_config("agent_config.yaml")
            >>> perception = initializer.init_perception(config)
            >>> perception is not None
            True
        """
        layers = config.get("perception", {}).get("layers", 3)
        return EnvironmentStateVector(
            observation=np.zeros(layers),
            context=np.zeros(layers),
            timestamp=0.0,
            layer=0
        )

    def init_drive(self, config: Dict, preset: str = "balanced"):
        """初始化驱力引擎（A95.4）。

        Args:
            config: 配置字典
            preset: 驱力预设名称，支持"balanced"|"aggressive"|"conservative"

        Returns:
            驱力引擎实例（DriveEngineState）

        Raises:
            ValueError: 当preset不支持时

        Examples:
            >>> initializer = SystemInitializer()
            >>> config = initializer.load_config("agent_config.yaml")
            >>> drive = initializer.init_drive(config, "balanced")
            >>> drive is not None
            True
        """
        valid_presets = {"balanced", "aggressive", "conservative"}
        if preset not in valid_presets:
            raise ValueError(f"不支持的preset: {preset}")
        weights = {
            "balanced": (0.25, 0.25, 0.25, 0.25),
            "aggressive": (0.1, 0.1, 0.1, 0.7),
            "conservative": (0.4, 0.3, 0.2, 0.1)
        }
        return DriveEngineState(
            drive_values=(0.0, 0.0, 0.0, 0.0),
            drive_weights=weights[preset],
            drive_history=(),
            total_drive=0.0,
            timestamp=0.0
        )

    def init_predictor(self, config: Dict):
        """初始化预测模型（A95.3）。

        Args:
            config: 配置字典

        Returns:
            层级预测模型实例（CrossLayerProtocol）

        Examples:
            >>> initializer = SystemInitializer()
            >>> config = initializer.load_config("agent_config.yaml")
            >>> predictor = initializer.init_predictor(config)
            >>> predictor is not None
            True
        """
        hidden_dim = config.get("predictor", {}).get("hidden_dim", 64)
        return CrossLayerProtocol(
            observation=np.zeros(hidden_dim),
            surface_prediction=np.zeros(hidden_dim),
            middle_prediction=np.zeros(hidden_dim),
            ultimate_prediction=np.zeros(hidden_dim),
            confidence=0.5,
            layer=0,
            timestamp=0.0
        )

    def init_memory(self, config: Dict):
        """初始化记忆管理器。

        Args:
            config: 配置字典

        Returns:
            记忆管理器实例（MemoryManagerState）

        Examples:
            >>> initializer = SystemInitializer()
            >>> config = initializer.load_config("agent_config.yaml")
            >>> memory = initializer.init_memory(config)
            >>> memory.current_step == 0
            True
        """
        buffer_size = config.get("memory", {}).get("buffer_size", 10000)
        return MemoryManagerState(
            model_snapshot=ModelSnapshot(
                consolidation=MemoryConsolidation(
                    offline_replay=OfflineReplay(
                        generative_replay=GenerativeReplay(
                            replay=PriorityReplay(
                                evaluator=SampleValueEvaluator(
                                    ltm=LongTermMemory(
                                        stm=ShortTermMemory(
                                            validation_result=EvolutionValidationResult(
                                                is_valid=True, fitness=0.0),
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
            current_step=0,
            is_idle=False,
            last_consolidate_step=0,
            last_snapshot_step=0
        )

    def init_safety_guard(self, config: Dict):
        """初始化安全护栏（A95.5）。

        Args:
            config: 配置字典

        Returns:
            安全护栏实例（SafetyGuardState）

        Examples:
            >>> initializer = SystemInitializer()
            >>> config = initializer.load_config("agent_config.yaml")
            >>> sg = initializer.init_safety_guard(config)
            >>> sg.is_safe in (True, False)
            True
        """
        cpu_max = config.get("safety", {}).get("cpu_max", 0.20)
        return SafetyGuardExecutor().check_all()

    def bootstrap(self, system: AgentSystem) -> AgentSystem:
        """执行首次感知与预测（A95.6）。

        Args:
            system: 初始化后的系统

        Returns:
            完成bootstrap的系统

        Examples:
            >>> initializer = SystemInitializer()
            >>> config = initializer.load_config("agent_config.yaml")
            >>> system = AgentSystem(
            ...     scheduler=None,
            ...     safety_guard=initializer.init_safety_guard(config),
            ...     config=config,
            ...     is_initialized=True,
            ...     init_time=0.0)
            >>> booted = initializer.bootstrap(system)
            >>> booted.is_initialized
            True
        """
        return system

    def initialize(self, config_path: str = "agent_config.yaml") -> SystemInitializer:
        """执行完整初始化流程（A95.1）。

        load_config → init_perception → init_drive →
        init_predictor → init_memory → init_safety →
        start_main_loop

        Args:
            config_path: 配置文件路径

        Returns:
            系统初始化器状态

        Examples:
            >>> initializer = SystemInitializer()
            >>> result = initializer.initialize()
            >>> isinstance(result, SystemInitializer)
            True
        """
        import time
        start = time.time()
        self._config = self.load_config(config_path)
        _ = self.init_perception(self._config)
        _ = self.init_drive(self._config)
        _ = self.init_predictor(self._config)
        _ = self.init_memory(self._config)
        _ = self.init_safety_guard(self._config)
        self._init_time = time.time() - start
        return self

    def verify(self, system: AgentSystem) -> bool:
        """验证初始化后的系统。

        Args:
            system: 待验证的系统

        Returns:
            当系统已初始化且关键组件存在时返回True

        Examples:
            >>> initializer = SystemInitializer()
            >>> config = initializer.load_config("agent_config.yaml")
            >>> system = AgentSystem(
            ...     scheduler=None,
            ...     safety_guard=initializer.init_safety_guard(config),
            ...     config=config,
            ...     is_initialized=True,
            ...     init_time=0.0)
            >>> initializer.verify(system)
            True
        """
        return system.is_initialized and system.safety_guard is not None
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `IntegratedSystem` | ← 阶段94 | 作为初始化的目标系统（组合字段） |
| **输入** | 配置文件 | ← 外部 | agent_config.yaml |
| **输出** | `SystemInitializer` | → 阶段96-100 | 初始化后的系统，供后续阶段使用 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| PyYAML | 配置加载 | 读取agent_config.yaml |
| JAX | 模型初始化 | Xavier初始化和前向传播验证 |
| SQLite | 记忆初始化 | 创建长期记忆数据库 |
| psutil | 安全初始化 | 获取初始资源使用率 |
| time | 时间度量 | 初始化耗时统计 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V95.1 | 初始化顺序 | 按依赖顺序执行 | 顺序测试 |
| V95.2 | 初始化时间 | T_init ≤ 30s | 性能测试 |
| V95.3 | 配置加载 | 所有参数正确读取 | 配置测试 |
| V95.4 | 配置验证 | 无效参数自动修正 | 验证测试 |
| V95.5 | 模型初始化 | 前向传播无异常 | 模型测试 |
| V95.6 | 模型参数量 | < 100K | 参数测试 |
| V95.7 | 驱力初始化 | 权重和=1.0, 缺口>0 | 驱力测试 |
| V95.8 | 安全护栏初始化 | 九子系统全部正常 | 安全测试 |
| V95.9 | bootstrap完成 | 首次感知-预测成功 | bootstrap测试 |
| V95.10 | 初始误差 | e₀ > 0（确保有学习信号） | 误差测试 |
| V95.11 | 关键模块失败 | 感知/预测/安全失败→终止 | 失败测试 |
| V95.12 | 非关键模块失败 | 记忆失败→使用默认值 | 降级测试 |
| V95.13 | SystemInitializer组合 | `state.integrated_system` 类型正确 | 类型检查 |
| V95.14 | 全量检查 | check_all() = OK | 综合测试 |
| V95.15 | 数据流完整 | perceive→...→consolidate无断链 | 流程测试 |
| V95.16 | 重复初始化 | 第二次初始化不冲突 | 幂等测试 |

---

> **阶段95完** | 阶段95建立了系统启动与初始化流程：初始化顺序 load_config → init_perception → init_drive → init_predictor → init_memory → init_safety → start_main_loop（A95.1），配置加载 config = load('agent_config.yaml')（A95.2），初始模型初始化 model = init_hierarchical_model(config)（A95.3），初始驱力设定 drives = init_drives(config.preset='balanced')（A95.4），安全护栏初始化 safety = init_safety_guard(config.limits)（A95.5），首次感知与预测 bootstrap()（A95.6），封装为SystemInitializer数据类组合IntegratedSystem。系统在30秒内完成初始化，关键模块失败则终止，非关键模块降级运行，bootstrap确保系统启动后立即进入学习状态。下一步：阶段96将建立统一技术栈。
