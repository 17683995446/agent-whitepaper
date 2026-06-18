# 阶段96：统一技术栈

> **阶段编号**：S96 | **所属层**：第十层·系统集成层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | JAX核心框架 | "计算引擎"——自动微分+GPU加速+JIT编译，一统天下 |
> | NumPyro概率编程 | "概率大脑"——变分推断+MCMC，处理不确定性 |
> | Haiku神经网络 | "神经网络工具箱"——JAX原生的NN库 |
> | evosax进化策略 | "进化引擎"——(1+1)-ES+OpenAI-ES，驱动进化 |
> | PureJaxRL强化学习 | "决策教练"——JAX原生RL，策略优化参考 |
> | SQLite+msgpack存储 | "记忆仓库"——轻量持久化，存经验存快照 |
> | Prometheus+Grafana监控 | "仪表盘"——可选的监控可视化 |
> | 技术选型决策树 | "选型指南"——什么场景用什么技术 |
> | TechStack | "技术栈的完整画像"——组合SystemInitializer，在初始化基础上增加技术栈状态 |

---

## 1. 第一性原理依据

本阶段从各阶段的技术选型出发，统一为JAX生态系统。核心依据为：

- **阶段1-10（基础公理层）**：SymPy用于符号推导验证，JAX用于数值计算
- **阶段11-20（预测编码层）**：JAX用于前向/反向传播，NumPyro用于变分推断
- **阶段21-30（驱力引擎层）**：JAX用于驱力计算，NumPyro用于不确定性量化
- **阶段31-40（自主决策层）**：JAX用于决策计算，PureJaxRL用于策略优化
- **阶段41-50（多层预测模型层）**：Haiku用于神经网络构建，JAX用于自动微分
- **阶段51-60（误差评估反馈层）**：JAX用于误差计算，SciPy用于统计检验
- **阶段61-70（双模式进化层）**：evosax用于进化策略，JAX用于适应度计算
- **阶段71-80（记忆与复盘层）**：SQLite用于持久化，msgpack用于序列化
- **阶段81-90（安全护栏层）**：psutil用于资源监控，JAX用于性能评估

本阶段的核心目标是：**定义核心框架 JAX（A96.1），概率编程 NumPyro（A96.2），神经网络 Haiku（A96.3），进化策略 evosax（A96.4），强化学习 PureJaxRL（A96.5），存储层 SQLite+msgpack（A96.6），监控层 Prometheus+Grafana（A96.7），技术选型决策树（A96.8），封装为TechStack数据类组合SystemInitializer。**

> **去重声明**：本阶段仅定义统一技术栈的选型决策和集成验证，不重新实现任何技术组件。各阶段仅列新增技术，统一技术栈在本阶段定义。

---

## 2. 核心数学公式

### 第一部分：核心框架JAX

#### 2.1 核心框架JAX定义（A96.1）

JAX是系统的核心计算框架，提供三大关键能力：

$$\text{JAX} = \text{AutoDiff} + \text{GPU\_Accelerate} + \text{JIT\_Compile} \tag{A96.1}$$

**JAX在系统中的使用场景**：

| 能力 | 使用场景 | 对应阶段 |
|------|---------|---------|
| grad() | 计算 ∂F/∂μ, ∂F/∂a, ∂F/∂W, ∂J/∂θ | 92 |
| jit() | 编译主循环、动力学计算 | 93 |
| vmap() | 批量感知、批量预测 | 94 |
| pmap() | 多设备并行进化 | 69 |
| lax.scan() | 时间序列积分 | 92 |

**JAX的自动微分与全局动力学的对应**：

| 梯度 | JAX实现 | 对应动力学 |
|------|---------|-----------|
| ∂F/∂μ | jax.grad(F, argnums=0) | 感知动力学 A92.1 |
| ∂F/∂a | jax.grad(F, argnums=1) | 行动动力学 A92.2 |
| ∂F/∂W | jax.grad(F, argnums=2) | 学习动力学 A92.3 |
| ∂J/∂θ | jax.grad(J, argnums=0) | 进化动力学 A92.4 |

**JAX的版本和依赖**：

| 组件 | 最低版本 | 说明 |
|------|---------|------|
| jax | 0.4.20 | 核心计算 |
| jaxlib | 0.4.20 | 后端支持 |
| numpy | 1.24+ | 兼容NumPy API |

- **输入**：Python函数
- **输出**：可微分、可编译、可加速的函数
- **约束**：纯函数（无副作用）

> **白话翻译**：JAX是"计算引擎"——自动微分算梯度、GPU加速算得快、JIT编译跑得稳。系统里所有需要算梯度的地方（感知、行动、学习、进化）都用JAX的grad()。主循环用jit()编译一次，后面每步都快。

---

### 第二部分：概率编程NumPyro

#### 2.2 概率编程NumPyro定义（A96.2）

NumPyro提供变分推断和MCMC能力：

$$\text{NumPyro} = \text{VariationalInference} + \text{MCMC} + \text{HierarchicalBayes} \tag{A96.2}$$

**NumPyro在系统中的使用场景**：

| 能力 | 使用场景 | 对应阶段 |
|------|---------|---------|
| SVI | 变分自由能最小化 | 4, 14 |
| MCMC | 后验采样验证 | 11 |
| NUTS | 高效后验探索 | 7 |
| AutoGuide | 自动变分分布 | 14 |

**NumPyro与变分自由能的对应**：

变分自由能 $F = D_{KL}[q(s) \| p(s|o)] + E_q[-\ln p(o|s)]$ 的最小化等价于变分推断的ELBO最大化。NumPyro的SVI（随机变分推断）直接实现了这一优化：

$$\text{SVI}(F) = \text{minimize}_{q} \, F(q, p) \tag{A96.2-svi}$$

- **输入**：概率模型定义
- **输出**：变分后验分布
- **约束**：JAX后端

> **白话翻译**：NumPyro是"概率大脑"——处理不确定性的工具。变分推断最小化自由能，MCMC采样验证后验，层级贝叶斯建模多层预测模型。底层跑在JAX上，自动享受GPU加速。

---

### 第三部分：神经网络Haiku

#### 2.3 神经网络Haiku定义（A96.3）

Haiku是JAX原生的神经网络库：

$$\text{Haiku} = \text{JAX\_Native\_NN} + \text{Stateful\_Module} + \text{RNN\_Support} \tag{A96.3}$$

**Haiku在系统中的使用场景**：

| 能力 | 使用场景 | 对应阶段 |
|------|---------|---------|
| Linear | 权重矩阵 | 15 |
| Conv1D | 时序特征提取 | 44 |
| LSTM | 序列记忆 | 46 |
| LayerNorm | 归一化 | 49 |
| MultiHeadAttention | 注意力机制 | 54 |

**Haiku与层级预测模型的对应**：

| 层级 | Haiku模块 | 参数量 |
|------|----------|--------|
| 低层 | hk.Conv1D + hk.Linear | ~4K |
| 中层 | hk.LSTM + hk.Linear | ~16K |
| 高层 | hk.MultiHeadAttention + hk.Linear | ~65K |

- **输入**：网络结构定义
- **输出**：可训练的神经网络模块
- **约束**：JAX原生，支持jit/vmap/grad

> **白话翻译**：Haiku是"神经网络工具箱"——JAX原生的NN库，支持各种网络层：线性层、卷积层、LSTM、注意力。因为是JAX原生的，自动支持jit编译和自动微分，不需要额外适配。

---

### 第四部分：进化策略evosax

#### 2.4 进化策略evosax定义（A96.4）

evosax提供JAX原生的进化策略：

$$\text{evosax} = (1+1)\text{-ES} + \text{OpenAI-ES} + \text{CMA-ES} \tag{A96.4}$$

**evosax在系统中的使用场景**：

| 策略 | 使用场景 | 对应阶段 |
|------|---------|---------|
| (1+1)-ES | 微修正进化 | 63 |
| OpenAI-ES | 结构性进化 | 67 |
| CMA-ES | 参数空间探索 | 69 |

**evosax与进化动力学的对应**：

A92.4的进化动力学 $\tau_\theta \dot{\theta} = -\partial J / \partial \theta + \sigma_\theta \xi$ 的离散化实现：

- 梯度项 $-\partial J / \partial \theta$：由JAX的grad()计算
- 变异项 $\sigma_\theta \xi$：由evosax的变异算子生成
- 选择：由evosax的选择策略执行

- **输入**：适应度函数 $f(\theta)$
- **输出**：进化后的参数 $\theta_{t+1}$
- **约束**：JAX原生，支持jit

> **白话翻译**：evosax是"进化引擎"——JAX原生的进化策略库。微调用(1+1)-ES（简单高效），大改用OpenAI-ES（全局搜索），精细调参用CMA-ES（协方差自适应）。底层跑在JAX上，进化过程可以JIT编译。

---

### 第五部分：强化学习PureJaxRL

#### 2.5 强化学习PureJaxRL定义（A96.5）

PureJaxRL提供JAX原生的强化学习算法：

$$\text{PureJaxRL} = \text{PPO} + \text{SAC} + \text{DQN} \tag{A96.5}$$

**PureJaxRL在系统中的使用场景**：

| 算法 | 使用场景 | 对应阶段 |
|------|---------|---------|
| PPO | 策略优化参考 | 37 |
| SAC | 连续行动空间 | 40 |
| DQN | 离散行动空间 | 39 |

**PureJaxRL与决策引擎的关系**：

PureJaxRL作为策略优化的参考实现，不直接替代主动推断的决策机制。系统优先使用预期自由能（A7.1）进行决策，PureJaxRL提供策略梯度的参考基准。

- **输入**：环境交互数据
- **输出**：优化后的策略参数
- **约束**：JAX原生，参考实现

> **白话翻译**：PureJaxRL是"决策教练"——JAX原生的强化学习库，提供PPO、SAC、DQN等算法。主要作为参考实现，系统优先用主动推断做决策，RL提供策略梯度的对照基准。

---

### 第六部分：存储层

#### 2.6 存储层定义（A96.6）

SQLite+msgpack提供轻量持久化：

$$\text{Storage} = \text{SQLite} + \text{msgpack} \tag{A96.6}$$

**存储层在系统中的使用场景**：

| 组件 | 使用场景 | 对应阶段 |
|------|---------|---------|
| SQLite | 长期记忆存储 | 76 |
| SQLite | 模型快照存储 | 78 |
| SQLite | 安全日志存储 | 84 |
| msgpack | 经验序列化 | 73 |
| msgpack | 模型参数序列化 | 78 |
| msgpack | 配置序列化 | 95 |

**存储层的数据量估算**：

| 数据类型 | 单条大小 | 日增量 | 72小时总量 |
|---------|---------|--------|-----------|
| 经验记录 | 1KB | ~10MB | ~2GB |
| 模型快照 | 500KB | ~1MB | ~70MB |
| 安全日志 | 100B | ~100KB | ~7MB |
| **总计** | — | ~11MB | ~2.1GB |

72小时产生约2.1GB数据，在磁盘80%上限内（假设磁盘≥10GB）。

- **输入**：待持久化的数据
- **输出**：持久化确认
- **约束**：磁盘占用 < 80%

> **白话翻译**：存储层是"记忆仓库"——SQLite存结构化数据（记忆、快照、日志），msgpack序列化二进制数据（经验、参数、配置）。72小时大概产生2GB数据，磁盘80%上限内够用。

---

### 第七部分：监控层

#### 2.7 监控层定义（A96.7）

Prometheus+Grafana提供可选的监控可视化：

$$\text{Monitoring} = \text{Prometheus} + \text{Grafana} \tag{A96.7}$$

**监控指标**：

| 指标 | 类型 | 采集频率 |
|------|------|---------|
| cpu_usage | Gauge | 1s |
| mem_usage | Gauge | 1s |
| disk_usage | Gauge | 60s |
| free_energy | Gauge | 1step |
| drive_gaps | Gauge | 1step |
| objective_J | Gauge | 1step |
| task_count | Counter | 1step |
| safety_violations | Counter | 1event |

**监控层的可选性**：

Prometheus+Grafana是可选组件——系统可以在无监控环境下独立运行。监控层仅用于开发和调试，不影响核心功能。

- **输入**：系统运行指标
- **输出**：可视化仪表盘
- **约束**：可选组件

> **白话翻译**：监控层是"仪表盘"——Prometheus采集指标，Grafana画图。CPU、内存、自由能、驱力、目标值、任务数、安全违规，一目了然。这是可选的，不影响系统运行。

---

### 第八部分：技术选型决策树

#### 2.8 技术选型决策树定义（A96.8）

技术选型决策树指导开发者在不同场景下选择合适的技术：

```
需要什么能力？
│
├── 数值计算 + 自动微分
│   └── JAX
│       ├── 需要概率推断？ → NumPyro
│       ├── 需要神经网络？ → Haiku
│       ├── 需要进化策略？ → evosax
│       └── 需要强化学习？ → PureJaxRL
│
├── 数据持久化
│   ├── 结构化数据？ → SQLite
│   └── 二进制序列化？ → msgpack
│
├── 资源监控
│   └── psutil
│
├── 符号验证
│   └── SymPy
│
├── 可视化监控
│   └── Prometheus + Grafana
│
└── 配置管理
    └── PyYAML
```

**技术选型的互斥规则**：

| 禁止 | 原因 | 替代 |
|------|------|------|
| PyTorch | 与JAX生态不兼容 | Haiku |
| TensorFlow | 与JAX生态不兼容 | JAX |
| pickle | 安全风险 | msgpack |
| Redis | 过重 | SQLite |
| MongoDB | 过重 | SQLite |

- **输入**：能力需求
- **输出**：推荐技术
- **约束**：全栈JAX生态

> **白话翻译**：技术选型决策树就是"选型指南"——需要算梯度用JAX，需要概率推断用NumPyro，需要神经网络用Haiku，需要进化用evosax，需要RL用PureJaxRL。存数据用SQLite+msgpack，看监控用Prometheus+Grafana。不用PyTorch和TensorFlow，因为跟JAX不兼容。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A96.1 | JAX = AutoDiff + GPU + JIT | Python函数 | 可微分/编译/加速函数 | 纯函数 |
| A96.2 | NumPyro = VI + MCMC + HierBayes | 概率模型 | 变分后验 | JAX后端 |
| A96.3 | Haiku = JAX_NN + Stateful + RNN | 网络定义 | NN模块 | JAX原生 |
| A96.4 | evosax = (1+1)-ES + OpenAI-ES + CMA-ES | 适应度函数 | 进化参数 | JAX原生 |
| A96.5 | PureJaxRL = PPO + SAC + DQN | 交互数据 | 策略参数 | 参考实现 |
| A96.6 | Storage = SQLite + msgpack | 待存储数据 | 持久化确认 | 磁盘<80% |
| A96.7 | Monitoring = Prometheus + Grafana | 运行指标 | 可视化 | 可选 |
| A96.8 | 技术选型决策树 | 能力需求 | 推荐技术 | 全栈JAX |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum

# 引用阶段95的数据类
from stage_095 import SystemInitializer


# ============================================================
# 阶段96 数据类：技术栈验证结果
# ============================================================
@dataclass(frozen=True)
class TechVerificationResult:
    """单个技术组件的验证结果。

    Attributes:
        component: 组件名称
        is_available: 是否可导入
        version: 版本号
        is_compatible: 版本是否兼容
        error_message: 错误信息
    """
    component: str
    is_available: bool
    version: str = "unknown"
    is_compatible: bool = False
    error_message: Optional[str] = None


# ============================================================
# 阶段96 数据类：技术栈
# ============================================================
@dataclass(frozen=True)
class TechStack:
    """统一技术栈的完整状态描述。

    组合阶段95的SystemInitializer，在初始化基础上增加技术栈状态。

    Attributes:
        system_initializer: 阶段95的系统初始化器（组合字段）
        verification_results: 各组件验证结果
        jax_version: JAX版本
        numpyro_version: NumPyro版本
        haiku_version: Haiku版本
        evosax_version: evosax版本
        purejaxrl_version: PureJaxRL版本
        sqlite_version: SQLite版本
        msgpack_version: msgpack版本
        all_compatible: 所有组件是否兼容
        jax_devices: JAX可用设备列表

    Invariants:
        - all_compatible = all(r.is_compatible for r in verification_results)
    """
    system_initializer: SystemInitializer
    verification_results: Tuple[TechVerificationResult, ...] = ()
    jax_version: str = "unknown"
    numpyro_version: str = "unknown"
    haiku_version: str = "unknown"
    evosax_version: str = "unknown"
    purejaxrl_version: str = "unknown"
    sqlite_version: str = "unknown"
    msgpack_version: str = "unknown"
    all_compatible: bool = False
    jax_devices: Tuple[str, ...] = ()


# ============================================================
# 阶段96 核心接口：技术栈验证器
# ============================================================
class TechStackVerifier:
    """阶段96的核心接口：统一技术栈验证器。

    验证所有技术组件的可用性、版本和兼容性。

    核心约束：
        - 所有核心组件可导入
        - 版本满足最低要求
        - JAX设备检测正常
    """

    # 最低版本要求
    MIN_VERSIONS = {
        "jax": "0.4.20",
        "numpyro": "0.13.0",
        "dm_haiku": "0.0.10",
        "evosax": "0.1.4",
        "purejaxrl": "0.0.2",
    }

    def verify_imports(self) -> Dict[str, bool]:
        """验证所有组件是否可导入。

        Returns:
            组件名→是否可导入的字典

        Examples:
            >>> configurator = TechStackConfigurator()
            >>> results = configurator.verify_imports()
            >>> isinstance(results, dict)
            True
            >>> "numpy" in results
            True
        """
        components = [
            "jax", "numpy", "numpyro", "haiku", "evosax",
            "sqlite3", "msgpack", "psutil", "yaml", "sympy"
        ]
        results: Dict[str, bool] = {}
        for name in components:
            try:
                __import__(name)
                results[name] = True
            except ImportError:
                results[name] = False
        return results

    def verify_versions(self) -> Dict[str, str]:
        """获取所有组件的版本号。

        Returns:
            组件名→版本号的字典

        Examples:
            >>> configurator = TechStackConfigurator()
            >>> versions = configurator.verify_versions()
            >>> isinstance(versions, dict)
            True
            >>> "numpy" in versions
            True
        """
        versions: Dict[str, str] = {}
        import importlib
        for name in ["jax", "numpy", "numpyro", "sympy"]:
            try:
                mod = importlib.import_module(name)
                versions[name] = getattr(mod, "__version__", "unknown")
            except Exception:
                versions[name] = "not_installed"
        return versions

    def verify_compatibility(self) -> bool:
        """验证所有组件版本是否兼容。

        Returns:
            当所有已安装组件版本 >= 最低版本要求时返回True

        Examples:
            >>> configurator = TechStackConfigurator()
            >>> ok = configurator.verify_compatibility()
            >>> isinstance(ok, bool)
            True
        """
        versions = self.verify_versions()
        for name, min_ver in self.MIN_VERSIONS.items():
            installed = versions.get(name, "0.0.0")
            if installed == "not_installed":
                continue
            if not self._version_gte(installed, min_ver):
                return False
        return True

    def _version_gte(self, v1: str, v2: str) -> bool:
        """比较版本号 v1 >= v2。

        Args:
            v1: 已安装版本
            v2: 最低版本

        Returns:
            当v1 >= v2时返回True
        """
        def parse(v: str):
            return tuple(int(x) for x in v.split(".") if x.isdigit())
        try:
            return parse(v1) >= parse(v2)
        except Exception:
            return True

    def verify_jax_devices(self) -> List[str]:
        """检测JAX可用设备。

        Returns:
            设备名称列表

        Examples:
            >>> configurator = TechStackConfigurator()
            >>> devices = configurator.verify_jax_devices()
            >>> len(devices) >= 1
            True
        """
        try:
            import jax
            return [str(d) for d in jax.devices()]
        except Exception:
            return ["cpu"]

    def full_verification(self) -> TechStack:
        """执行完整技术栈验证。

        Returns:
            TechStack 技术栈状态

        Examples:
            >>> configurator = TechStackConfigurator()
            >>> stack = configurator.full_verification()
            >>> isinstance(stack, TechStack)
            True
        """
        import time
        start = time.time()
        imports = self.verify_imports()
        versions = self.verify_versions()
        compatible = self.verify_compatibility()
        devices = self.verify_jax_devices()
        elapsed = time.time() - start
        all_ok = all(imports.values()) and compatible and len(devices) >= 1
        return TechStack(
            system_initializer=SystemInitializer(),
            import_status=imports,
            versions=versions,
            compatibility=compatible,
            jax_devices=tuple(devices),
            verification_time=elapsed,
            all_passed=all_ok
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `SystemInitializer` | ← 阶段95 | 作为技术栈验证的基础（组合字段） |
| **输入** | 各阶段技术选型 | ← 阶段1-90 | 统一为JAX生态 |
| **输出** | `TechStack` | → 阶段97-100 | 验证通过的技术栈 |

---

## 6. 开源技术选型

| 技术 | 用途 | 最低版本 |
|------|------|---------|
| JAX | 核心计算 | 0.4.20 |
| NumPyro | 概率编程 | 0.13.0 |
| dm-haiku | 神经网络 | 0.0.10 |
| evosax | 进化策略 | 0.1.4 |
| PureJaxRL | 强化学习 | 0.0.2 |
| SQLite | 持久化 | 3.40+ |
| msgpack | 序列化 | 1.0+ |
| psutil | 资源监控 | 5.9+ |
| PyYAML | 配置管理 | 6.0+ |
| SymPy | 符号验证 | 1.12+ |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V96.1 | JAX导入 | import jax 成功 | 导入测试 |
| V96.2 | JAX版本 | ≥ 0.4.20 | 版本测试 |
| V96.3 | JAX设备 | 至少1个设备（CPU） | 设备测试 |
| V96.4 | JAX自动微分 | jax.grad() 正常 | 梯度测试 |
| V96.5 | JAX JIT | jax.jit() 正常 | 编译测试 |
| V96.6 | NumPyro导入 | import numpyro 成功 | 导入测试 |
| V96.7 | NumPyro SVI | 变分推断正常 | VI测试 |
| V96.8 | Haiku导入 | import haiku 成功 | 导入测试 |
| V96.9 | Haiku网络 | hk.Linear 正常 | 网络测试 |
| V96.10 | evosax导入 | import evosax 成功 | 导入测试 |
| V96.11 | evosax (1+1)-ES | 进化策略正常 | ES测试 |
| V96.12 | PureJaxRL导入 | import purejaxrl 成功 | 导入测试 |
| V96.13 | SQLite | 数据库读写正常 | 存储测试 |
| V96.14 | msgpack | 序列化/反序列化正常 | 序列化测试 |
| V96.15 | 版本兼容 | 所有组件版本满足最低要求 | 兼容性测试 |
| V96.16 | TechStack组合 | `state.system_initializer` 类型正确 | 类型检查 |
| V96.17 | 互斥规则 | 无PyTorch/TensorFlow依赖 | 依赖测试 |
| V96.18 | 端到端JAX | 所有计算可JIT编译 | 全栈测试 |

---

> **阶段96完** | 阶段96建立了统一技术栈：核心框架 JAX（自动微分+GPU加速+JIT编译）（A96.1），概率编程 NumPyro（变分推断+MCMC）（A96.2），神经网络 Haiku（JAX原生NN库）（A96.3），进化策略 evosax（(1+1)-ES+OpenAI-ES）（A96.4），强化学习 PureJaxRL（JAX原生RL）（A96.5），存储层 SQLite+msgpack（轻量持久化）（A96.6），监控层 Prometheus+Grafana（可选）（A96.7），技术选型决策树（A96.8），封装为TechStack数据类组合SystemInitializer。全栈JAX生态，从微分到进化，从概率到网络，从存储到监控，统一技术栈确保系统的一致性和可维护性。下一步：阶段97将建立端到端验证方案。
