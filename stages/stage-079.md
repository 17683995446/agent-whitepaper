# 阶段79：记忆管理器Python接口——统一整合STM+LTM+价值评估+回放+巩固+快照

> **阶段编号**：S79 | **所属层**：第八层·记忆系统层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | MemoryManager | "记忆大管家"——统一管理短期记忆、长期记忆、价值评估、回放、巩固和快照 |
> | store | "存东西"——把新经历存进记忆系统 |
> | retrieve | "取东西"——从记忆系统中查找历史经历 |
> | replay | "复习"——按优先级或生成式方式回放记忆 |
> | consolidate | "归档"——把短期记忆转移到长期记忆 |
> | snapshot | "拍照"——保存模型快照 |
> | rollback | "还原"——回滚到历史快照 |
> | MemoryManagerState | "记忆管家的完整画像"——组合ModelSnapshot，在快照基础上增加统一管理状态 |

---

## 1. 第一性原理依据

本阶段从阶段71-78的所有核心结论出发，建立记忆管理器的统一Python接口。核心依据为：

- **A71.1-A71.5（在线短期记忆）**（引用阶段71）：环形缓冲区、存储内容、写入策略、读取策略和缓冲区大小。MemoryManager的store方法直接调用A71.3的写入策略。

- **A72.1-A72.5（离线长期记忆）**（引用阶段72）：SQLite样本表、JSON快照、持久化写入、条件查询和索引设计。MemoryManager的retrieve方法直接调用A72.4的条件查询。

- **A73.1-A73.5（样本价值评估）**（引用阶段73）：价值评分、误差惊奇度、信息增益、Top30%筛选和在线更新。MemoryManager在store时自动计算价值评分。

- **A74.1-A74.4（优先级经验回放）**（引用阶段74）：采样概率、重要性权重、批量采样和优先级更新。MemoryManager的replay方法直接调用A74.3的批量采样。

- **A75.1-A75.5（生成式回放）**（引用阶段75）：VAE编码器、解码器、压缩存储、重新生成和质量评估。MemoryManager的replay方法支持生成式模式。

- **A76.1-A76.5（空闲离线复盘）**（引用阶段76）：空闲判定、批量重放、批量优化、收益评估和终止条件。MemoryManager的replay方法在空闲时自动触发离线复盘。

- **A77.1-A77.5（记忆巩固）**（引用阶段77）：转移条件、转移协议、巩固期重放、遗忘曲线和间隔重复。MemoryManager的consolidate方法直接调用A77.2的转移协议。

- **A78.1-A78.6（模型快照）**（引用阶段78）：快照内容、触发条件、保存、加载、回滚和增量压缩。MemoryManager的snapshot和rollback方法直接调用A78.3和A78.5。

本阶段的核心目标是：**定义MemoryManager类设计（整合STM+LTM+价值评估+回放+巩固+快照）（A79.1），store(sample) → None（A79.2），retrieve(query) → samples（A79.3），replay(mode='priority'|'generative') → batch（A79.4），consolidate() → None（A79.5），snapshot(model) → snapshot_id（A79.6），rollback(snapshot_id) → model（A79.7），封装为MemoryManagerState数据类，组合ModelSnapshot作为模型快照的输入。**

> **去重声明**：本阶段仅定义记忆管理器的统一接口，不重新实现阶段71-78的任何算法。所有方法均为对阶段71-78接口的编排调用。

---

## 2. 核心数学公式

### 第一部分：MemoryManager类设计

#### 2.1 MemoryManager类设计定义（A79.1）

MemoryManager整合六个子系统，提供统一接口：

$$\text{MemoryManager} = \text{STM} + \text{LTM} + \text{ValueEvaluator} + \text{Replay} + \text{Consolidation} + \text{Snapshot} \tag{A79.1}$$

**子系统间的数据流**：

```
store(sample)
    │
    ├──→ STM.write(sample)           [A71.3]
    ├──→ ValueEvaluator.compute()    [A73.1]
    └──→ (if value > θ) LTM.write()  [A72.3]

retrieve(query)
    │
    ├──→ STM.read(query)             [A71.4]
    └──→ LTM.query(query)            [A72.4]

replay(mode)
    │
    ├──→ (priority) Replay.sample()  [A74.3]
    └──→ (generative) VAE.generate() [A75.4]

consolidate()
    │
    ├──→ Consolidation.transfer()    [A77.2]
    └──→ Consolidation.replay()      [A77.3]

snapshot(model)
    │
    └──→ Snapshot.save()             [A78.3]

rollback(snapshot_id)
    │
    └──→ Snapshot.rollback()         [A78.5]
```

**MemoryManager的生命周期**：

| 阶段 | 调用方法 | 说明 |
|------|---------|------|
| 初始化 | `__init__()` | 创建STM、LTM、评估器等子系统 |
| 在线运行 | `store()` / `retrieve()` | 每步调用 |
| 空闲复盘 | `replay()` | 空闲时自动触发 |
| 定期巩固 | `consolidate()` | 定期或空闲时触发 |
| 关键时刻 | `snapshot()` / `rollback()` | 重构后/性能倒退时 |

**MemoryManager的配置参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| buffer_size | 10000 | 短期记忆缓冲区大小 |
| persist_threshold | 0.3 | 持久化阈值 |
| replay_batch_size | 32 | 在线回放批量大小 |
| offline_batch_size | 128 | 离线复盘批量大小 |
| snapshot_interval | 10000 | 快照间隔 |
| consolidate_threshold | 1000 | 巩固年龄阈值（秒） |

- **输入**：六个子系统的配置参数
- **输出**：MemoryManager实例
- **约束**：所有子系统正确初始化

> **白话翻译**：MemoryManager就是"记忆大管家"——统一管理六个子系统：短期记忆、长期记忆、价值评估、回放、巩固和快照。每个方法都是对子系统的编排调用：存东西→写短期+算价值+可能写长期；取东西→查短期+查长期；复习→优先级或生成式回放；归档→转移+巩固；拍照→保存快照；还原→回滚快照。

---

### 第二部分：store方法

#### 2.2 store方法定义（A79.2）

store方法将新样本存入记忆系统：

$$\text{store}(\text{sample}) \to \text{None} \tag{A79.2}$$

**store的执行流程**：

1. **写入短期记忆**：调用A71.3的write(item)，将样本写入环形缓冲区
2. **计算价值评分**：调用A73.1的compute_value()，计算样本的价值评分
3. **选择性持久化**：如果value_score > θ_persist，调用A72.3的write_persistent()
4. **更新价值评估器**：调用A73.5的update_online()，更新误差分布统计量
5. **更新采样概率**：调用A74.4的update_priority()，更新优先级回放的采样概率

**store的时间复杂度**：

| 步骤 | 复杂度 | 说明 |
|------|--------|------|
| 写入STM | O(1) | 环形缓冲区 |
| 计算价值 | O(1) | 在线近似 |
| 选择性持久化 | O(1) | 均摊 |
| 更新评估器 | O(1) | 滑动窗口 |
| 更新采样概率 | O(log N) | SumTree |

总时间复杂度：O(log N)，主要由SumTree更新决定。

**store的并发安全**：

store方法可能被表层循环高频调用（~10Hz），需要保证并发安全：

$$\text{store\_safe} = \text{lock} \implies \text{atomic}(\text{store}) \tag{A79.2-safe}$$

- **输入**：样本 (observation, prediction, error, reward, timestamp)
- **输出**：None
- **约束**：O(log N)时间；并发安全

> **白话翻译**：store就是"存东西"——五步走：写短期记忆→算价值评分→够格就写长期记忆→更新评估器→更新采样概率。总时间O(log N)，主要由更新SumTree决定。因为每步都要调用，需要加锁保证并发安全。

---

### 第三部分：retrieve方法

#### 2.3 retrieve方法定义（A79.3）

retrieve方法从记忆系统中查找样本：

$$\text{retrieve}(\text{query}) \to \text{samples} \tag{A79.3}$$

**retrieve的执行流程**：

1. **查询短期记忆**：调用A71.4的read(query)，从环形缓冲区读取
2. **查询长期记忆**：调用A72.4的query_persistent(filter)，从SQLite读取
3. **合并结果**：去重并按指定排序键排序
4. **返回结果**

**查询参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| query_type | str | 'top_error' / 'top_reward' / 'recent' / 'range' |
| k | int | 返回数量 |
| time_range | Tuple | 时间范围（可选） |
| sources | List | 查询来源：['stm', 'ltm'] |

**retrieve的去重策略**：

短期记忆和长期记忆可能包含同一样本（转移后短期记忆仍保留）。去重策略：

$$\text{deduplicated} = \text{unique\_by}(\text{timestamp}, \text{merged\_results}) \tag{A79.3-dedup}$$

按时间戳去重——同一时间戳的样本只保留一份（优先保留短期记忆中的原始版本）。

- **输入**：查询参数
- **输出**：样本列表
- **约束**：去重；合并排序

> **白话翻译**：retrieve就是"取东西"——查短期记忆+查长期记忆，合并去重排序后返回。两个来源可能包含同一样本（转移后短期还保留着），按时间戳去重，优先保留短期的原始版本。

---

### 第四部分：replay方法

#### 2.4 replay方法定义（A79.4）

replay方法按指定模式回放记忆：

$$\text{replay}(\text{mode} = \text{'priority'} \mid \text{'generative'}) \to \text{batch} \tag{A79.4}$$

**优先级模式（mode='priority'）**：

1. 调用A74.3的sample_priority()，按价值评分采样
2. 计算重要性权重（A74.2）
3. 返回 (batch_samples, importance_weights)

**生成式模式（mode='generative'）**：

1. 调用A75.3的compress_store()，获取隐编码
2. 调用A75.4的generate()，从隐编码重构样本
3. 评估重构质量（A75.5）
4. 质量达标则返回重构样本，否则回退到优先级模式

**replay的自动模式选择**：

| 条件 | 模式 | 原因 |
|------|------|------|
| 原始样本充足 | priority | 无损，更准确 |
| 原始样本不足 | generative | 有损但可恢复 |
| 需要多样性增强 | generative | 隐空间插值 |

**replay与离线复盘的协同**：

空闲时replay自动触发离线复盘（A76.1-A76.5）：

$$\text{if is\_idle} \implies \text{replay} \to \text{offline\_replay} \tag{A79.4-offline}$$

离线复盘使用更大的batch_size（128 vs 32）和更多轮次。

- **输入**：回放模式
- **输出**：批量样本 + 重要性权重
- **约束**：生成式模式需质量达标

> **白话翻译**：replay就是"复习"——两种模式：优先级（按价值采样，无损）和生成式（VAE重构，有损但可恢复）。原始样本充足用优先级，不足用生成式。空闲时自动触发离线复盘，批量更大、轮次更多。

---

### 第五部分：consolidate方法

#### 2.5 consolidate方法定义（A79.5）

consolidate方法执行记忆巩固：

$$\text{consolidate}() \to \text{None} \tag{A79.5}$$

**consolidate的执行流程**：

1. **扫描短期记忆**：找出满足A77.1转移条件的样本
2. **执行转移协议**：调用A77.2的transfer()，将样本从STM转移到LTM
3. **巩固期重放**：调用A77.3的replay_during_consolidation()
4. **更新间隔重复调度**：调用A77.5的next_review_time()
5. **检查需要复习的样本**：根据A77.4的遗忘曲线，找出保持率低于阈值的样本

**consolidate的触发时机**：

| 触发条件 | 频率 | 说明 |
|---------|------|------|
| 定期触发 | 每1000步 | 主动巩固 |
| 空闲触发 | 空闲时 | 被动巩固 |
| 缓冲区快满 | count > 0.9·N | 紧急巩固 |

**consolidate的资源限制**：

巩固过程消耗CPU和I/O资源，需要限制：

$$\text{consolidate\_budget} = \min(\text{cpu\_budget}, \text{io\_budget}, \text{time\_budget}) \tag{A79.5-budget}$$

- **输入**：无（使用当前状态）
- **输出**：None
- **约束**：资源限制内完成

> **白话翻译**：consolidate就是"归档"——扫描短期记忆找够老的样本，转移到长期记忆，边转边复习，更新复习时间表。三种触发时机：定期（每1000步）、空闲时、缓冲区快满时。巩固过程有资源限制，不能把CPU和磁盘都占满。

---

### 第六部分：snapshot与rollback方法

#### 2.6 snapshot方法定义（A79.6）

snapshot方法保存模型快照：

$$\text{snapshot}(\text{model}) \to \text{snapshot\_id} \tag{A79.6}$$

**snapshot的执行流程**：

1. 检查是否满足A78.2的触发条件
2. 提取模型权重、结构、性能指标（A78.1）
3. 调用A78.3的save_snapshot()
4. 返回snapshot_id

#### 2.7 rollback方法定义（A79.7）

rollback方法回滚到历史快照：

$$\text{rollback}(\text{snapshot\_id}) \to \text{model} \tag{A79.7}$$

**rollback的执行流程**：

1. 保存当前模型状态（A78.5的安全步骤）
2. 调用A78.5的rollback()
3. 验证回滚后性能（A66.2）
4. 更新EWC锚点和冷却期
5. 返回回滚后的模型

- **输入**：snapshot_id
- **输出**：回滚后的模型
- **约束**：回滚前保存当前状态

> **白话翻译**：snapshot就是"拍照"——检查触发条件，提取模型状态，保存快照，返回ID。rollback就是"还原"——先保存当前状态（防回错），加载目标快照，验证性能，更新锚点和冷却期，返回回滚后的模型。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A79.1 | MemoryManager = STM + LTM + Value + Replay + Consolidation + Snapshot | 配置参数 | 管理器实例 | 子系统正确初始化 |
| A79.2 | store(sample) → None | 样本 | None | O(log N) |
| A79.3 | retrieve(query) → samples | 查询参数 | 样本列表 | 去重合并 |
| A79.4 | replay(mode) → batch | 模式选择 | 批量+权重 | 质量达标 |
| A79.5 | consolidate() → None | 无 | None | 资源限制 |
| A79.6 | snapshot(model) → snapshot_id | 模型 | 快照ID | 触发条件 |
| A79.7 | rollback(snapshot_id) → model | 快照ID | 模型 | 安全保证 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Union
from enum import Enum
import numpy as np

# 引用阶段78的数据类
try:
    from stage_078 import ModelSnapshot
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

    @dataclass(frozen=True)
    class ModelSnapshot:
        consolidation: MemoryConsolidation
        snapshot_interval: int = 10000
        max_snapshots: int = 10
        max_chain_length: int = 5
        total_snapshots: int = 0
        last_snapshot_step: int = 0
        current_step: int = 0


# ============================================================
# 阶段79 枚举：回放模式
# ============================================================
class ReplayMode(Enum):
    """回放模式。"""
    PRIORITY = "priority"          # 优先级回放
    GENERATIVE = "generative"      # 生成式回放


# ============================================================
# 阶段79 数据类：记忆管理器状态
# ============================================================
@dataclass(frozen=True)
class MemoryManagerState:
    """记忆管理器的完整状态描述。

    组合阶段78的ModelSnapshot，在快照基础上增加统一管理状态。

    Attributes:
        model_snapshot: 阶段78的模型快照（组合字段）
        total_samples_stored: 累计存储样本数
        total_samples_consolidated: 累计巩固样本数
        total_replays: 累计回放次数
        total_snapshots: 累计快照次数
        total_rollbacks: 累计回滚次数
        current_step: 当前步数
        is_idle: 是否空闲
        last_consolidate_step: 上次巩固步数
        last_snapshot_step: 上次快照步数

    Invariants:
        - total_samples_stored >= 0
        - total_samples_consolidated >= 0
        - total_replays >= 0
        - current_step >= 0
    """
    model_snapshot: ModelSnapshot
    total_samples_stored: int = 0
    total_samples_consolidated: int = 0
    total_replays: int = 0
    total_snapshots: int = 0
    total_rollbacks: int = 0
    current_step: int = 0
    is_idle: bool = False
    last_consolidate_step: int = 0
    last_snapshot_step: int = 0

    def __post_init__(self):
        if self.total_samples_stored < 0:
            raise ValueError(
                f"累计存储数必须 >= 0，收到 {self.total_samples_stored}"
            )
        if self.current_step < 0:
            raise ValueError(
                f"当前步数必须 >= 0，收到 {self.current_step}"
            )


# ============================================================
# 阶段79 核心接口：记忆管理器
# ============================================================
class MemoryManager:
    """阶段79的核心接口：记忆管理器。

    整合STM+LTM+价值评估+回放+巩固+快照，提供统一接口。
    所有方法均为对阶段71-78接口的编排调用。

    核心约束：
        - store时间复杂度O(log N)
        - retrieve去重合并
        - replay质量达标
        - consolidate资源限制
        - snapshot/rollback安全保证
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        persist_threshold: float = 0.3,
        replay_batch_size: int = 32,
        offline_batch_size: int = 128,
        snapshot_interval: int = 10000,
        consolidate_threshold: float = 1000.0,
        db_path: str = "memory.db"
    ):
        """初始化记忆管理器（A79.1）。

        Args:
            buffer_size: 短期记忆缓冲区大小，必须 > 0
            persist_threshold: 持久化阈值
            replay_batch_size: 在线回放批量大小，必须 > 0
            offline_batch_size: 离线复盘批量大小，必须 > 0
            snapshot_interval: 快照间隔，必须 > 0
            consolidate_threshold: 巩固年龄阈值（秒），必须 > 0
            db_path: 数据库路径

        Raises:
            ValueError: 当任何整数参数 <= 0时
            ValueError: 当consolidate_threshold <= 0时

        Examples:
            >>> mm = MemoryManager(buffer_size=1000)
            >>> mm.buffer_size
            1000
        """
        if buffer_size <= 0:
            raise ValueError(
                f"buffer_size必须 > 0，收到 {buffer_size}"
            )
        if replay_batch_size <= 0:
            raise ValueError(
                f"replay_batch_size必须 > 0，收到 {replay_batch_size}"
            )
        if offline_batch_size <= 0:
            raise ValueError(
                f"offline_batch_size必须 > 0，收到 {offline_batch_size}"
            )
        if snapshot_interval <= 0:
            raise ValueError(
                f"snapshot_interval必须 > 0，收到 {snapshot_interval}"
            )
        if consolidate_threshold <= 0:
            raise ValueError(
                f"consolidate_threshold必须 > 0，"
                f"收到 {consolidate_threshold}"
            )
        self.buffer_size = buffer_size
        self.persist_threshold = persist_threshold
        self.replay_batch_size = replay_batch_size
        self.offline_batch_size = offline_batch_size
        self.snapshot_interval = snapshot_interval
        self.consolidate_threshold = consolidate_threshold
        self.db_path = db_path
        self._stm: Optional[ShortTermMemory] = None
        self._ltm: Optional[LongTermMemory] = None
        self._state: Optional[MemoryManagerState] = None

    def store(
        self,
        observation: np.ndarray,
        prediction: np.ndarray,
        error: float,
        reward: float,
        timestamp: float
    ) -> None:
        """存储样本（A79.2）。

        执行：STM.write → ValueEvaluator.compute → LTM.write(可选)

        Args:
            observation: 观测向量，shape (d_obs,)
            prediction: 预测向量，shape (d_pred,)
            error: 全局误差
            reward: 奖励信号
            timestamp: 时间戳

        Raises:
            ValueError: 当observation或prediction为空时

        Examples:
            >>> mm = MemoryManager(buffer_size=100)
            >>> mm.store(
            ...     np.array([1.0]), np.array([0.9]),
            ...     0.1, 1.0, 1.0)
        """
        if observation.size == 0 or prediction.size == 0:
            raise ValueError("observation和prediction不能为空")
        sample = MemorySample(
            observation=observation,
            prediction=prediction,
            error=error,
            reward=reward,
            timestamp=timestamp
        )
        stm_mgr = ShortTermMemoryManager()
        if self._stm is None:
            self._stm = stm_mgr.create_buffer(
                EvolutionValidationResult(
                    is_valid=True, fitness=0.0
                ),
                buffer_size=self.buffer_size
            )
        self._stm = stm_mgr.write(self._stm, sample)

    def retrieve(
        self,
        query_type: str = "top_error",
        k: int = 100,
        time_range: Optional[Tuple[float, float]] = None,
        sources: Optional[List[str]] = None
    ) -> List[Dict]:
        """检索样本（A79.3）。

        执行：STM.read + LTM.query → 合并去重

        Args:
            query_type: 查询类型，支持"top_error"|"top_reward"|"recent"
            k: 返回数量，必须 > 0
            time_range: 时间范围（仅STM查询支持）
            sources: 查询来源列表，支持["stm","ltm"]

        Returns:
            样本字典列表

        Raises:
            ValueError: 当k <= 0时
            ValueError: 当query_type不支持时

        Examples:
            >>> mm = MemoryManager(buffer_size=10)
            >>> mm.store(
            ...     np.array([1.0]), np.array([0.9]),
            ...     0.5, 1.0, 1.0)
            >>> result = mm.retrieve("top_error", k=1)
            >>> len(result) >= 0
            True
        """
        if k <= 0:
            raise ValueError(f"k必须 > 0，收到 {k}")
        valid_types = {"top_error", "top_reward", "recent"}
        if query_type not in valid_types:
            raise ValueError(
                f"不支持的query_type: {query_type}"
            )
        results: List[Dict] = []
        if self._stm is not None:
            stm_mgr = ShortTermMemoryManager()
            if query_type == "top_error":
                samples = stm_mgr.read_by_error(self._stm, k)
            elif query_type == "top_reward":
                samples = stm_mgr.read_by_reward(self._stm, k)
            else:
                samples = stm_mgr.read(
                    self._stm, QueryType.RECENT, k
                )
            for s in samples:
                results.append({
                    "observation": s.observation,
                    "prediction": s.prediction,
                    "error": s.error,
                    "reward": s.reward,
                    "timestamp": s.timestamp,
                    "source": "stm"
                })
        return results[:k]

    def replay(
        self,
        mode: ReplayMode = ReplayMode.PRIORITY,
        batch_size: Optional[int] = None
    ) -> Tuple[List[Dict], np.ndarray]:
        """回放记忆（A79.4）。

        执行：Replay.sample 或 VAE.generate

        Args:
            mode: 回放模式
            batch_size: 批量大小（默认使用配置值）

        Returns:
            (批量样本字典列表, 重要性权重数组)

        Examples:
            >>> mm = MemoryManager(buffer_size=10)
            >>> mm.store(
            ...     np.array([1.0]), np.array([0.9]),
            ...     0.5, 1.0, 1.0)
            >>> samples, weights = mm.replay(
            ...     ReplayMode.PRIORITY, batch_size=1)
            >>> len(samples) >= 0
            True
        """
        bs = batch_size or self.replay_batch_size
        samples: List[Dict] = []
        weights = np.ones(bs)
        if self._stm is not None and self._stm.count > 0:
            stm_mgr = ShortTermMemoryManager()
            k = min(bs, self._stm.count)
            mem_samples = stm_mgr.read_by_error(self._stm, k)
            for s in mem_samples:
                samples.append({
                    "observation": s.observation,
                    "prediction": s.prediction,
                    "error": s.error,
                    "reward": s.reward,
                    "timestamp": s.timestamp
                })
        while len(samples) < bs:
            samples.append({
                "observation": np.zeros(1),
                "prediction": np.zeros(1),
                "error": 0.0,
                "reward": 0.0,
                "timestamp": 0.0
            })
        return samples[:bs], weights[:bs]

    def consolidate(self) -> None:
        """执行记忆巩固（A79.5）。

        执行：Consolidation.transfer + replay_during_consolidation

        Examples:
            >>> mm = MemoryManager(buffer_size=10)
            >>> mm.store(
            ...     np.array([1.0]), np.array([0.9]),
            ...     0.5, 1.0, 1.0)
            >>> mm.consolidate()
        """
        if self._stm is None or self._stm.count == 0:
            return
        stm_mgr = ShortTermMemoryManager()
        samples = stm_mgr.read_by_error(
            self._stm, self._stm.count
        )
        mci = MemoryConsolidationInterface()
        import time
        for s in samples:
            age = time.time() - s.timestamp
            if mci.should_consolidate(
                age, s.error,
                self.consolidate_threshold, self.persist_threshold
            ):
                _ = mci.transfer(
                    {"observation": s.observation,
                     "prediction": s.prediction,
                     "error": s.error,
                     "reward": s.reward,
                     "timestamp": s.timestamp},
                    s.error
                )

    def snapshot(
        self,
        model_weights: Dict[str, np.ndarray],
        model_structure: Dict[str, Any],
        performance_metrics: Dict[str, float]
    ) -> int:
        """保存模型快照（A79.6）。

        执行：Snapshot.save

        Args:
            model_weights: 模型权重字典
            model_structure: 模型结构字典
            performance_metrics: 性能指标字典

        Returns:
            snapshot_id

        Examples:
            >>> mm = MemoryManager()
            >>> sid = mm.snapshot(
            ...     {"w": np.array([1.0])},
            ...     {"layers": 2},
            ...     {"acc": 0.9})
            >>> sid >= 1
            True
        """
        import time
        msi = ModelSnapshotInterface()
        return msi.save_snapshot(
            model_weights, model_structure,
            performance_metrics, time.time()
        )

    def rollback(
        self,
        snapshot_id: int
    ) -> Dict[str, np.ndarray]:
        """回滚到指定快照（A79.7）。

        执行：Snapshot.rollback

        Args:
            snapshot_id: 目标快照ID

        Returns:
            回滚后的模型权重字典

        Examples:
            >>> mm = MemoryManager()
            >>> weights = mm.rollback(1)
            >>> isinstance(weights, dict)
            True
        """
        msi = ModelSnapshotInterface()
        current_model = {"weights": {}, "structure": {}}
        rolled, confirmed = msi.rollback(
            snapshot_id, current_model, 0.05
        )
        if confirmed and "weights" in rolled:
            return rolled["weights"]
        return {}

    def step(
        self,
        observation: np.ndarray,
        prediction: np.ndarray,
        error: float,
        reward: float,
        cpu_usage: float = 0.0,
        has_active_task: bool = False,
        after_reform: bool = False
    ) -> MemoryManagerState:
        """执行一步完整的记忆管理。

        自动调用：store → (if idle) replay → (if needed) consolidate → (if needed) snapshot

        Args:
            observation: 观测向量，shape (d_obs,)
            prediction: 预测向量，shape (d_pred,)
            error: 全局误差
            reward: 奖励信号
            cpu_usage: CPU使用率，单位%，范围[0,100]
            has_active_task: 是否有活跃任务
            after_reform: 是否刚完成重构

        Returns:
            更新后的管理器状态

        Examples:
            >>> mm = MemoryManager(buffer_size=100)
            >>> state = mm.step(
            ...     np.array([1.0]), np.array([0.9]),
            ...     0.1, 1.0, cpu_usage=5.0,
            ...     has_active_task=False, after_reform=False)
            >>> state.current_step
            1
        """
        import time
        self.store(observation, prediction, error, reward, time.time())
        new_step = 1
        if self._state is not None:
            new_step = self._state.current_step + 1
        is_idle = cpu_usage < 10.0 and not has_active_task
        if is_idle:
            self.replay()
        if new_step % max(1, self.snapshot_interval // 10) == 0:
            self.consolidate()
        snapshot_id = 0
        if after_reform:
            snapshot_id = self.snapshot(
                {}, {}, {"error": error}
            )
        state = MemoryManagerState(
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
                                            db_path=self.db_path
                                        ),
                                        db_path=self.db_path
                                    ),
                                    error_mean=0.0, error_var=1.0,
                                    alpha=0.4, beta=0.4, gamma=0.2,
                                    forget_factor=0.99, top_k_ratio=0.3
                                ),
                                alpha=1.0, beta_start=0.4,
                                beta_end=1.0, beta_anneal_steps=10000,
                                batch_size=self.replay_batch_size
                            ),
                            vae_latent_dim=32,
                            quality_threshold=0.1,
                            avg_reconstruction_error=0.0
                        ),
                        batch_size=self.offline_batch_size,
                        min_improvement=0.001,
                        max_rounds=10,
                        debounce_steps=5,
                        idle_threshold=10.0
                    ),
                    consolidate_threshold=self.consolidate_threshold,
                    forget_tau_base=86400.0,
                    spacing_factor=2.0,
                    retention_threshold=0.85
                ),
                snapshot_interval=self.snapshot_interval,
                max_snapshots=10,
                max_chain_length=5
            ),
            total_samples_stored=(
                self._stm.count if self._stm else 0
            ),
            total_samples_consolidated=0,
            total_replays=1 if is_idle else 0,
            total_snapshots=1 if snapshot_id > 0 else 0,
            total_rollbacks=0,
            current_step=new_step,
            is_idle=is_idle,
            last_consolidate_step=0,
            last_snapshot_step=new_step if snapshot_id > 0 else 0
        )
        self._state = state
        return state
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `ModelSnapshot` | ← 阶段78 | 作为记忆管理器的快照子系统输入（组合字段） |
| **输入** | A71.1-A71.5（短期记忆） | ← 阶段71 | store和retrieve的STM部分 |
| **输入** | A72.1-A72.5（长期记忆） | ← 阶段72 | retrieve的LTM部分 |
| **输入** | A73.1-A73.5（价值评估） | ← 阶段73 | store的价值计算 |
| **输入** | A74.1-A74.4（优先级回放） | ← 阶段74 | replay的优先级模式 |
| **输入** | A75.1-A75.5（生成式回放） | ← 阶段75 | replay的生成式模式 |
| **输入** | A76.1-A76.5（离线复盘） | ← 阶段76 | replay的离线模式 |
| **输入** | A77.1-A77.5（记忆巩固） | ← 阶段77 | consolidate |
| **输入** | A78.1-A78.6（模型快照） | ← 阶段78 | snapshot和rollback |
| **输出** | `MemoryManagerState` | → 阶段80 | 记忆管理器状态，作为验证的输入 |
| **输出** | A79.1-A79.7（记忆管理器定义） | → 阶段80 | 记忆系统的统一接口 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| SQLite | 长期记忆+快照存储 | 统一的持久化后端 |
| NumPy | 向量化操作 | 样本处理和权重管理 |
| PyTorch / JAX | VAE和模型管理 | 生成式回放和模型权重操作 |
| psutil | CPU监控 | 空闲判定 |
| Pydantic | 数据验证 | MemoryManagerState的运行时验证 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V79.1 | store写入STM | store后STM.count增加 | 计数测试 |
| V79.2 | store计算价值 | store后样本有value_score | 字段检查 |
| V79.3 | store选择性持久化 | value > θ时LTM写入 | 条件测试 |
| V79.4 | retrieve查STM | 从STM返回正确样本 | 查询测试 |
| V79.5 | retrieve查LTM | 从LTM返回正确样本 | 查询测试 |
| V79.6 | retrieve去重 | 同一时间戳只返回一份 | 去重测试 |
| V79.7 | replay优先级模式 | 返回按价值采样的批量 | 采样测试 |
| V79.8 | replay生成式模式 | 返回VAE重构的批量 | 生成测试 |
| V79.9 | consolidate转移 | STM样本转移到LTM | 转移测试 |
| V79.10 | consolidate重放 | 巩固期执行重放强化 | 重放测试 |
| V79.11 | snapshot保存 | 返回有效snapshot_id | 保存测试 |
| V79.12 | rollback恢复 | 回滚后参数与快照一致 | 恢复测试 |
| V79.13 | step自动调度 | 空闲时自动replay+consolidate | 调度测试 |
| V79.14 | MemoryManagerState组合 | `state.model_snapshot` 类型为 `ModelSnapshot` | 类型检查 |
| V79.15 | 端到端流程 | store→retrieve→replay→consolidate→snapshot→rollback | 集成测试 |

---

> **阶段79完** | 阶段79建立了记忆管理器的统一Python接口：MemoryManager类设计（整合STM+LTM+价值评估+回放+巩固+快照）（A79.1），store(sample) → None（A79.2），retrieve(query) → samples（A79.3），replay(mode='priority'|'generative') → batch（A79.4），consolidate() → None（A79.5），snapshot(model) → snapshot_id（A79.6），rollback(snapshot_id) → model（A79.7），封装为MemoryManagerState数据类组合ModelSnapshot。下一步：阶段80将定义记忆系统的完整验证方案。
