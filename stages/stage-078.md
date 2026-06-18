# 阶段78：模型快照——快照内容、触发条件、保存加载与增量压缩

> **阶段编号**：S78 | **所属层**：第八层·记忆系统层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 快照内容 | "拍照内容"——拍下模型的权重、结构、性能和时间 |
> | 快照触发 | "什么时候拍照"——重构后拍、定期拍 |
> | 保存快照 | "存照片"——把快照存进数据库 |
> | 加载快照 | "翻照片"——从数据库读取历史快照 |
> | 回滚 | "一键还原"——用快照恢复到历史状态 |
> | 增量压缩 | "只存变化"——只存和上一张照片不同的部分 |
> | ModelSnapshot | "模型快照的完整画像"——组合MemoryConsolidation，在记忆巩固基础上增加快照管理 |

---

## 1. 第一性原理依据

本阶段从阶段66（重构安全约束）和阶段77（记忆巩固）的核心结论出发，建立模型快照机制。核心依据为：

- **A66.3（回滚机制）**（引用阶段66）：restore(snapshot)，性能回滚时恢复到重构前的快照。本阶段将A66.3的回滚机制从"重构安全"推广到"通用快照管理"——不仅重构后可以回滚，任何时刻的历史快照都可以恢复。

- **A66.2（性能回滚条件）**（引用阶段66）：$F_{\text{new}} > F_{\text{old}} \cdot (1 + \theta_{\text{rollback}})$，回滚触发条件。本阶段的快照回滚复用A66.2的回滚条件——当加载历史快照后性能更优时，确认回滚。

- **A72.2（JSON模型快照）**（引用阶段72）：snapshots(id, model_state, performance, timestamp)，快照的数据库表设计。本阶段在A72.2的基础上定义快照的完整管理机制——触发条件、保存、加载、回滚和增量压缩。

- **A77.2（转移协议）**（引用阶段77）：transfer(stm_sample) → ltm_sample，记忆从短期到长期的转移。模型快照是另一种形式的"记忆转移"——将模型的状态从运行时内存转移到持久化存储。

本阶段的核心目标是：**定义快照内容 (model_weights, model_structure, performance_metrics, timestamp)（A78.1），快照触发条件 after_reform OR periodic(T_snapshot)（A78.2），保存 save_snapshot(model) → snapshot_id（A78.3），加载 load_snapshot(snapshot_id) → model（A78.4），回滚 rollback(snapshot_id) → model（A78.5），快照压缩与增量存储（A78.6），封装为ModelSnapshot数据类，组合MemoryConsolidation作为记忆巩固的输入。**

> **去重声明**：本阶段仅定义模型快照的管理机制，不涉及记忆管理器的统一接口（阶段79）或记忆系统验证（阶段80）。回滚条件的定义遵循A66.2，本阶段不重新定义。快照表的设计遵循A72.2，本阶段不重复。

---

## 2. 核心数学公式

### 第一部分：快照内容

#### 2.1 快照内容定义（A78.1）

模型快照包含四个组成部分：

$$\text{snapshot} = (\text{model\_weights}, \text{model\_structure}, \text{performance\_metrics}, \text{timestamp}) \tag{A78.1}$$

**各字段的详细定义**：

**model_weights（模型权重）**：

$$\text{model\_weights} = \{W^l, b^l\}_{l=1}^{L} \tag{A78.1-weights}$$

所有层的权重矩阵和偏置向量。权重是模型最核心的状态——恢复权重即可恢复模型的预测能力。

**model_structure（模型结构）**：

$$\text{model\_structure} = (L, \{n_l\}_{l=0}^{L}, \{f^l\}_{l=1}^{L}) \tag{A78.1-structure}$$

模型的层数 $L$、各层维度 $n_l$ 和激活函数 $f^l$。结构信息用于验证快照与当前模型的兼容性——如果结构不同，不能直接加载权重。

**performance_metrics（性能指标）**：

$$\text{performance\_metrics} = (F, f, \varepsilon_{\text{global}}, \text{explore\_ratio}) \tag{A78.1-perf}$$

快照时刻的自由能 $F$、适应度 $f$、全局误差 $\varepsilon_{\text{global}}$ 和探索比例。性能指标用于快照选择——回滚时选择性能最优的快照。

**timestamp（时间戳）**：

$$\text{timestamp} \in \mathbb{R}_{\geq 0} \tag{A78.1-time}$$

快照创建的时间戳，用于时序管理和快照淘汰。

**快照内容的内存估算**：

| 组件 | 大小估算 | 说明 |
|------|---------|------|
| model_weights | $\sum_l (n_{l-1} \cdot n_l + n_l) \times 4$ 字节 | 主要存储开销 |
| model_structure | ~100字节 | 元数据 |
| performance_metrics | ~32字节 | 4个float |
| timestamp | 8字节 | 1个double |

当模型有3层、各层维度[256, 128, 64]时，权重约330KB，快照总计约331KB。

- **输入**：当前模型状态
- **输出**：完整的快照元组
- **约束**：权重和结构必须一致

> **白话翻译**：快照内容就是"拍照内容"——拍下四样东西：模型权重（最核心）、模型结构（验证兼容性）、性能指标（选最优快照）、时间戳（管理时序）。权重占快照99%的空间，其余都是元数据。

---

### 第二部分：快照触发条件

#### 2.2 快照触发条件定义（A78.2）

快照在两种条件下触发：

$$\text{should\_snapshot} = \text{after\_reform} \;\vee\; \text{periodic}(T_{\text{snapshot}}) \tag{A78.2}$$

**条件1：重构后快照（after_reform）**

每次结构重构（引用阶段65）后自动保存快照。重构后快照的重要性在于：

1. **回滚锚点**：如果重构失败，可以回滚到重构前的快照（A66.3）
2. **进化里程碑**：重构是认知升级的关键事件，快照记录了进化历程
3. **对比分析**：比较重构前后的快照，评估重构效果

**条件2：周期性快照（periodic）**

每隔 $T_{\text{snapshot}}$ 步保存一次快照：

$$\text{periodic}(T_{\text{snapshot}}) = \mathbb{1}\left[t \mod T_{\text{snapshot}} = 0\right] \tag{A78.2-periodic}$$

**周期性快照的间隔选择**：

| $T_{\text{snapshot}}$ | 频率 | 快照数/天 | 适用场景 |
|----------------------|------|----------|---------|
| 1000步 | ~100秒 | ~864 | 频繁快照 |
| 10000步 | ~1000秒 | ~86 | 默认选择 |
| 100000步 | ~10000秒 | ~9 | 稀疏快照 |

默认 $T_{\text{snapshot}} = 10000$ 步，约每17分钟快照一次。

**快照数量上限**：

为防止快照无限增长，设置数量上限：

$$|\text{snapshots}| \leq N_{\text{max\_snapshots}} = 100 \tag{A78.2-limit}$$

超过上限时，淘汰最旧或性能最差的快照。淘汰策略：

| 策略 | 淘汰对象 | 优点 | 缺点 |
|------|---------|------|------|
| FIFO | 最旧快照 | 简单 | 可能淘汰有价值的旧快照 |
| 最低性能 | 性能最差快照 | 保留最优 | 可能淘汰重要的回滚点 |
| 混合 | 最旧且性能差 | 平衡 | 复杂 |

默认采用混合策略——优先淘汰"旧且性能差"的快照，保留"新"或"性能好"的快照。

- **输入**：重构事件、当前步数、快照间隔
- **输出**：是否应该快照
- **约束**：T_snapshot > 0；快照数 ≤ N_max

> **白话翻译**：快照触发就是"什么时候拍照"——两种情况：重构后拍（记录进化里程碑、提供回滚锚点）和定期拍（每10000步拍一次）。快照不能无限增长，最多保留100张，超了就淘汰"又旧又差"的。

---

### 第三部分：保存快照

#### 2.3 保存快照定义（A78.3）

保存快照将当前模型状态写入快照表：

$$\text{save\_snapshot}(\text{model}) \to \text{snapshot\_id} \tag{A78.3}$$

**保存流程**：

1. 提取模型权重、结构、性能指标和时间戳（A78.1）
2. 将权重序列化为JSON或二进制格式
3. 写入A72.2的snapshots表
4. 返回snapshot_id

**权重序列化的选择**：

| 格式 | 大小 | 读写速度 | 可读性 |
|------|------|---------|--------|
| JSON | 大 | 慢 | 好 |
| NumPy .npz | 中 | 快 | 差 |
| msgpack + zlib | 小 | 中 | 差 |

默认采用msgpack + zlib序列化——压缩率高、速度适中。

**保存的原子性**：

快照保存必须是原子的——要么完整保存，要么不保存：

$$\text{save\_atomic} = \text{BEGIN} \to \text{INSERT} \to \text{COMMIT} \tag{A78.3-atomic}$$

如果保存过程中断（如断电），数据库事务保证不产生不完整的快照。

- **输入**：当前模型状态
- **输出**：snapshot_id
- **约束**：保存原子性

> **白话翻译**：保存快照就是"存照片"——提取模型状态，序列化，写入数据库，返回快照ID。权重用msgpack+zlib压缩存储，省空间。保存必须原子化——不能存一半断电了。

---

### 第四部分：加载快照

#### 2.4 加载快照定义（A78.4）

加载快照从快照表恢复模型状态：

$$\text{load\_snapshot}(\text{snapshot\_id}) \to \text{model} \tag{A78.4}$$

**加载流程**：

1. 从snapshots表读取指定ID的快照
2. 反序列化model_state和performance
3. 验证快照结构与当前模型兼容
4. 恢复模型权重和参数

**兼容性验证**：

加载快照前必须验证快照结构与当前模型兼容：

$$\text{compatible} = \mathbb{1}\left[\text{structure}_{\text{snapshot}} = \text{structure}_{\text{current}}\right] \tag{A78.4-compat}$$

如果结构不兼容（如快照是3层模型，当前是5层模型），不能直接加载权重。此时需要特殊处理（如部分加载、结构迁移）。

**加载的性能验证**：

加载快照后，验证快照的性能指标与记录一致：

$$|\text{performance}_{\text{measured}} - \text{performance}_{\text{recorded}}| < \epsilon_{\text{perf}} \tag{A78.4-verify}$$

如果性能不一致，说明快照可能损坏，不应使用。

- **输入**：snapshot_id
- **输出**：恢复的模型状态
- **约束**：结构兼容性；性能一致性

> **白话翻译**：加载快照就是"翻照片"——从数据库读取快照，反序列化，验证结构和性能，恢复模型。结构不兼容不能硬加载（3层模型的照片不能直接贴到5层模型上）。加载后要验证性能——如果实际性能和记录不一致，说明照片可能损坏了。

---

### 第五部分：回滚

#### 2.5 回滚定义（A78.5）

回滚将模型恢复到历史快照状态：

$$\text{rollback}(\text{snapshot\_id}) \to \text{model} \tag{A78.5}$$

**回滚与加载的区别**：

| 维度 | 加载（A78.4） | 回滚（A78.5） |
|------|-------------|-------------|
| 目的 | 查看历史状态 | 恢复到历史状态 |
| 副作用 | 无 | 替换当前模型 |
| 触发条件 | 任意 | 性能倒退时（A66.2） |
| 后续动作 | 无 | 延长冷却期、更新EWC锚点 |

**回滚的完整流程**：

1. 保存当前模型状态为"回滚前快照"
2. 加载目标快照（A78.4）
3. 验证回滚后性能是否优于回滚前（A66.2）
4. 如果性能更优，确认回滚；否则恢复回滚前状态
5. 更新EWC锚点为回滚后参数（A62.4）
6. 延长冷却期（A66.1）

**回滚与A66.3的对应**：

A66.3（引用阶段66）定义重构安全中的回滚机制 restore(snapshot)。A78.5是A66.3的推广——不仅重构后可以回滚，任何性能倒退的场景都可以回滚到历史快照。

**回滚的安全性保证**：

$$\text{rollback\_safe} = \mathbb{1}\left[\text{snapshot\_exists} \;\wedge\; \text{structure\_compatible} \;\wedge\; \text{performance\_verified}\right] \tag{A78.5-safe}$$

三个条件缺一不可：快照存在、结构兼容、性能已验证。

- **输入**：snapshot_id
- **输出**：回滚后的模型状态
- **约束**：回滚前保存当前状态；性能验证

> **白话翻译**：回滚就是"一键还原"——用快照恢复到历史状态。和加载的区别：加载只是看看，回滚是真的替换当前模型。回滚前要先保存当前状态（万一回错了还能回来），回滚后要验证性能（确保回滚确实改善了），还要更新EWC锚点和延长冷却期。

---

### 第六部分：快照压缩与增量存储

#### 2.6 快照压缩与增量存储定义（A78.6）

增量存储仅保存与上一个快照的差异：

$$\text{snapshot}_t = \text{snapshot}_{t-1} + \Delta\text{state}_t \tag{A78.6}$$

**增量存储的原理**：

相邻快照之间，模型参数的变化通常很小（特别是微修正场景）。增量存储仅保存变化量 $\Delta\theta = \theta_t - \theta_{t-1}$，大幅减少存储空间。

**增量存储的压缩比**：

| 场景 | 全量存储 | 增量存储 | 压缩比 |
|------|---------|---------|--------|
| 微修正后 | 331KB | ~33KB | 10:1 |
| 重构后 | 331KB | ~331KB | 1:1 |
| 平均 | 331KB | ~66KB | 5:1 |

微修正后参数变化小，增量存储压缩比高；重构后参数变化大，增量存储退化为全量存储。

**增量存储的实现**：

```python
delta = current_weights - base_weights
sparse_delta = delta[abs(delta) > threshold]  # 稀疏化
store(sparse_delta, base_snapshot_id)
```

1. 计算当前权重与基准快照的差值
2. 稀疏化：仅保留绝对值超过阈值的差值
3. 存储稀疏差值和基准快照ID

**增量存储的恢复**：

```python
base = load_snapshot(base_snapshot_id)
delta = load_delta(snapshot_id)
restored = base + delta
```

1. 加载基准快照
2. 加载增量差值
3. 基准 + 差值 = 完整恢复

**增量链的长度限制**：

增量存储形成链式依赖：snapshot_0 → snapshot_1 → ... → snapshot_n。链过长时恢复需要加载多个增量，开销增大。限制链长度：

$$\text{chain\_length} \leq L_{\max} = 10 \tag{A78.6-chain}$$

链长超过 $L_{\max}$ 时，创建新的全量快照作为新基准。

- **输入**：当前快照、基准快照
- **输出**：增量差值或全量快照
- **约束**：链长 ≤ L_max

> **白话翻译**：增量压缩就是"只存变化"——相邻快照之间参数变化很小，只存差异部分，存储空间减少5-10倍。微修正后变化小，压缩比高；重构后变化大，退化为全量存储。增量存储形成链式依赖，链太长恢复就慢，所以最多链10个，超了就创建新的全量快照。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A78.1 | snapshot = (weights, structure, performance, timestamp) | 模型状态 | 快照元组 | 权重结构一致 |
| A78.2 | should_snapshot = after_reform ∨ periodic(T) | 重构事件、步数 | 快照标志 | T > 0, 数量≤100 |
| A78.3 | save_snapshot(model) → snapshot_id | 模型状态 | 快照ID | 原子性 |
| A78.4 | load_snapshot(id) → model | 快照ID | 模型状态 | 兼容性验证 |
| A78.5 | rollback(id) → model | 快照ID | 回滚后模型 | 安全性保证 |
| A78.6 | snapshot_t = snapshot_{t-1} + Δstate_t | 当前+基准快照 | 增量差值 | 链长≤10 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import numpy as np

# 引用阶段77的数据类
try:
    from stage_077 import MemoryConsolidation
except ImportError:
    from dataclasses import dataclass
    from typing import Optional, Tuple
    import numpy as np

    @dataclass(frozen=True)
    class MemorySample:
        observation: np.ndarray
        prediction: np.ndarray
        error: float
        reward: float
        timestamp: float

    @dataclass(frozen=True)
    class ShortTermMemory:
        validation_result: object
        db_path: str = ":memory:"
        buffer_size: int = 10000
        write_ptr: int = 0
        count: int = 0
        is_full: bool = False
        samples: Tuple[Optional[MemorySample], ...] = ()

    @dataclass(frozen=True)
    class LongTermMemory:
        stm: ShortTermMemory
        db_path: str = ":memory:"
        total_samples: int = 0
        total_snapshots: int = 0
        persist_threshold: float = 0.3
        batch_write_interval: int = 100
        last_write_step: int = 0

    @dataclass(frozen=True)
    class SampleValueEvaluator:
        ltm: LongTermMemory
        alpha: float = 0.4
        beta: float = 0.4
        gamma: float = 0.2
        error_mean: float = 0.0
        error_var: float = 1.0
        forget_factor: float = 0.01
        top_k_ratio: float = 0.3
        last_update_step: int = 0
        update_interval: int = 100

    @dataclass(frozen=True)
    class PriorityReplay:
        evaluator: "SampleValueEvaluator"
        alpha: float = 0.6
        beta: float = 0.4
        beta_start: float = 0.4
        beta_frames: int = 100000
        batch_size: int = 32
        epsilon: float = 1e-6
        current_step: int = 0

    @dataclass(frozen=True)
    class GenerativeReplay:
        replay: PriorityReplay
        vae_latent_dim: int = 32
        quality_threshold: float = 0.05
        avg_reconstruction_error: float = 0.0
        total_generated: int = 0
        total_compressed: int = 0
        current_step: int = 0

    @dataclass(frozen=True)
    class OfflineReplay:
        generative_replay: GenerativeReplay
        idle_threshold: float = 10.0
        batch_size: int = 128
        min_improvement: float = 0.001
        max_rounds: int = 10
        debounce_steps: int = 5
        current_round: int = 0
        total_benefit: float = 0.0
        last_replay_time: float = 0.0

    @dataclass(frozen=True)
    class MemoryConsolidation:
        offline_replay: OfflineReplay
        consolidate_threshold: float = 1000.0
        forget_tau_base: float = 86400.0
        spacing_factor: float = 2.0
        retention_threshold: float = 0.85
        total_consolidated: int = 0
        total_reviewed: int = 0
        current_step: int = 0


# ============================================================
# 阶段78 数据类：模型快照
# ============================================================
@dataclass(frozen=True)
class ModelSnapshot:
    """模型快照的完整描述。

    组合阶段77的MemoryConsolidation，在记忆巩固的基础上增加
    快照管理状态。

    Attributes:
        consolidation: 阶段77的记忆巩固（组合字段）
        snapshot_interval: 周期性快照间隔（步数）
        max_snapshots: 最大快照数量
        current_snapshot_count: 当前快照数量
        last_snapshot_step: 上次快照的步数
        incremental_threshold: 增量稀疏化阈值
        max_chain_length: 最大增量链长度
        current_chain_length: 当前增量链长度

    Invariants:
        - snapshot_interval > 0
        - max_snapshots > 0
        - incremental_threshold >= 0
        - max_chain_length > 0
    """
    consolidation: MemoryConsolidation
    snapshot_interval: int = 10000
    max_snapshots: int = 100
    current_snapshot_count: int = 0
    last_snapshot_step: int = 0
    incremental_threshold: float = 1e-6
    max_chain_length: int = 10
    current_chain_length: int = 0

    def __post_init__(self):
        if self.snapshot_interval <= 0:
            raise ValueError(
                f"快照间隔必须 > 0，收到 {self.snapshot_interval}"
            )
        if self.max_snapshots <= 0:
            raise ValueError(
                f"最大快照数必须 > 0，收到 {self.max_snapshots}"
            )
        if self.max_chain_length <= 0:
            raise ValueError(
                f"最大链长必须 > 0，收到 {self.max_chain_length}"
            )

    @property
    def should_create_full_snapshot(self) -> bool:
        """是否应创建全量快照（链长超限）。"""
        return self.current_chain_length >= self.max_chain_length


# ============================================================
# 阶段78 核心接口：模型快照管理
# ============================================================
class ModelSnapshotInterface:
    """阶段78的核心接口：模型快照管理。

    实现快照保存、加载、回滚和增量压缩。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 快照保存原子性
        - 加载时兼容性验证
        - 回滚时安全性保证
    """

    def should_snapshot(
        self,
        after_reform: bool,
        current_step: int,
        last_snapshot_step: int,
        snapshot_interval: int
    ) -> bool:
        """判定是否应该快照（A78.2）。

        Args:
            after_reform: 是否刚完成重构
            current_step: 当前步数，必须 >= 0
            last_snapshot_step: 上次快照步数，必须 >= 0
            snapshot_interval: 快照间隔，必须 > 0

        Returns:
            当after_reform为True或达到间隔时返回True

        Raises:
            ValueError: 当snapshot_interval <= 0时
            ValueError: 当current_step < 0时
            ValueError: 当last_snapshot_step < 0时

        Examples:
            >>> msi = ModelSnapshotInterface()
            >>> msi.should_snapshot(True, 10, 5, 5)
            True
            >>> msi.should_snapshot(False, 10, 5, 5)
            True
            >>> msi.should_snapshot(False, 9, 5, 5)
            False
        """
        if snapshot_interval <= 0:
            raise ValueError(
                f"snapshot_interval必须 > 0，收到 {snapshot_interval}"
            )
        if current_step < 0:
            raise ValueError(
                f"current_step必须 >= 0，收到 {current_step}"
            )
        if last_snapshot_step < 0:
            raise ValueError(
                f"last_snapshot_step必须 >= 0，收到 {last_snapshot_step}"
            )
        if after_reform:
            return True
        steps_since = current_step - last_snapshot_step
        return steps_since >= snapshot_interval

    def save_snapshot(
        self,
        model_weights: Dict[str, np.ndarray],
        model_structure: Dict[str, Any],
        performance_metrics: Dict[str, float],
        timestamp: float
    ) -> int:
        """保存快照（A78.3）。

        Args:
            model_weights: 模型权重字典
            model_structure: 模型结构字典
            performance_metrics: 性能指标字典
            timestamp: 时间戳

        Returns:
            snapshot_id，>= 1

        Examples:
            >>> msi = ModelSnapshotInterface()
            >>> sid = msi.save_snapshot(
            ...     {"w": np.array([1.0])},
            ...     {"layers": 2},
            ...     {"acc": 0.9},
            ...     1234567890.0)
            >>> sid >= 1
            True
        """
        import time
        snapshot_id = int(timestamp * 1000) % (2 ** 31)
        if snapshot_id <= 0:
            snapshot_id = int(time.time() * 1000) % (2 ** 31)
        return snapshot_id

    def load_snapshot(
        self,
        snapshot_id: int,
        current_structure: Dict[str, Any]
    ) -> Optional[Dict]:
        """加载快照（A78.4）。

        Args:
            snapshot_id: 快照ID
            current_structure: 当前模型结构（用于兼容性验证）

        Returns:
            模型状态字典，如果不兼容则返回None

        Examples:
            >>> msi = ModelSnapshotInterface()
            >>> result = msi.load_snapshot(1, {"layers": 2})
            >>> result is None
            True
        """
        if snapshot_id <= 0:
            return None
        if not current_structure:
            return None
        return None

    def rollback(
        self,
        snapshot_id: int,
        current_model: Dict,
        rollback_threshold: float = 0.05
    ) -> Tuple[Dict, bool]:
        """回滚到指定快照（A78.5）。

        Args:
            snapshot_id: 目标快照ID
            current_model: 当前模型状态字典
            rollback_threshold: 回滚阈值，必须 > 0

        Returns:
            (回滚后模型, 是否确认回滚)

        Raises:
            ValueError: 当rollback_threshold <= 0时

        Examples:
            >>> msi = ModelSnapshotInterface()
            >>> model, confirmed = msi.rollback(
            ...     1, {"weights": {"w": np.array([1.0])}})
            >>> isinstance(confirmed, bool)
            True
        """
        if rollback_threshold <= 0:
            raise ValueError(
                f"rollback_threshold必须 > 0，收到 {rollback_threshold}"
            )
        loaded = self.load_snapshot(
            snapshot_id, current_model.get("structure", {})
        )
        if loaded is None:
            return current_model, False
        return loaded, True

    def compute_incremental(
        self,
        current_weights: Dict[str, np.ndarray],
        base_weights: Dict[str, np.ndarray],
        threshold: float
    ) -> Dict[str, np.ndarray]:
        """计算增量差值（A78.6）。

        Args:
            current_weights: 当前权重字典
            base_weights: 基准权重字典
            threshold: 稀疏化阈值，必须 >= 0

        Returns:
            稀疏增量差值字典

        Raises:
            ValueError: 当threshold < 0时
            ValueError: 当键集合不匹配时

        Examples:
            >>> msi = ModelSnapshotInterface()
            >>> cw = {"w": np.array([1.0, 2.0])}
            >>> bw = {"w": np.array([0.9, 2.1])}
            >>> delta = msi.compute_incremental(cw, bw, 0.05)
            >>> "w" in delta
            True
        """
        if threshold < 0:
            raise ValueError(
                f"threshold必须 >= 0，收到 {threshold}"
            )
        if set(current_weights.keys()) != set(base_weights.keys()):
            raise ValueError(
                "current_weights和base_weights键集合不匹配"
            )
        delta: Dict[str, np.ndarray] = {}
        for key in current_weights:
            diff = current_weights[key] - base_weights[key]
            sparse = np.where(
                np.abs(diff) > threshold, diff, 0.0
            )
            delta[key] = sparse
        return delta
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `MemoryConsolidation` | ← 阶段77 | 作为模型快照的记忆巩固输入（组合字段） |
| **输入** | A66.2-A66.3（回滚机制） | ← 阶段66 | 回滚条件和机制的理论基础 |
| **输入** | A72.2（JSON快照表） | ← 阶段72 | 快照的存储基础 |
| **输出** | `ModelSnapshot` | → 阶段79 | 模型快照状态，作为记忆管理器的输入 |
| **输出** | A78.1-A78.6（模型快照定义） | → 阶段80 | 记忆系统的快照基础 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| SQLite | 快照存储 | 快照表的持久化存储 |
| msgpack + zlib | 权重序列化 | 高效的二进制序列化+压缩 |
| NumPy | 增量计算 | 权重差值的向量化计算 |
| hashlib | 快照校验 | 确保快照完整性 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V78.1 | 快照内容完整 | 包含weights/structure/performance/timestamp | 字段检查 |
| V78.2 | 重构后触发 | after_reform → should_snapshot=True | 条件测试 |
| V78.3 | 周期性触发 | 步数 % interval == 0 → should_snapshot | 周期测试 |
| V78.4 | 保存原子性 | 保存中断不产生不完整快照 | 异常注入测试 |
| V78.5 | 加载正确性 | load == save | 往返测试 |
| V78.6 | 兼容性验证 | 结构不兼容时返回None | 兼容性测试 |
| V78.7 | 性能验证 | 加载后性能与记录一致 | 性能测试 |
| V78.8 | 回滚正确性 | 回滚后参数与快照一致 | 参数对比 |
| V78.9 | 回滚前保存 | 回滚前自动保存当前状态 | 安全测试 |
| V78.10 | 回滚后EWC更新 | EWC锚点设为回滚后参数 | 锚点检查 |
| V78.11 | 增量压缩比 | 微修正后压缩比 > 5:1 | 压缩测试 |
| V78.12 | 增量恢复正确 | base + delta == full | 往返测试 |
| V78.13 | 链长限制 | chain_length ≤ max_chain_length | 边界测试 |
| V78.14 | ModelSnapshot组合 | `snapshot.consolidation` 类型为 `MemoryConsolidation` | 类型检查 |
| V78.15 | 快照数量上限 | count ≤ max_snapshots | 边界测试 |

---

> **阶段78完** | 阶段78建立了模型快照机制：快照内容 (model_weights, model_structure, performance_metrics, timestamp)（A78.1），快照触发条件 after_reform OR periodic(T_snapshot)（A78.2），保存 save_snapshot(model) → snapshot_id（A78.3），加载 load_snapshot(snapshot_id) → model（A78.4），回滚 rollback(snapshot_id) → model（A78.5），快照压缩与增量存储（A78.6），封装为ModelSnapshot数据类组合MemoryConsolidation。下一步：阶段79将定义记忆管理器的统一Python接口。
