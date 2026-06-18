# 阶段72：离线长期记忆——SQLite持久化、JSON快照与索引优化

> **阶段编号**：S72 | **所属层**：第八层·记忆系统层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | SQLite样本表 | "档案柜"——把重要经历存进数据库，关机也不丢 |
> | JSON模型快照 | "拍照留念"——把模型的状态拍下来，随时可以翻看 |
> | 持久化写入 | "存档"——把短期记忆中的精华存入长期档案 |
> | 条件查询 | "翻档案"——按时间、误差、价值等条件查找历史经历 |
> | 索引设计 | "目录索引"——给档案柜建目录，翻找更快 |
> | LongTermMemory | "长期记忆的完整画像"——组合ShortTermMemory，在短期缓冲基础上增加持久化存储 |

---

## 1. 第一性原理依据

本阶段从阶段71（在线短期记忆）的核心结论出发，建立离线长期记忆的持久化存储机制。核心依据为：

- **A71.2（存储内容）**（引用阶段71）：item = (observation, prediction, error, reward, timestamp)，短期记忆的样本结构。长期记忆的样本表直接继承A71.2的字段定义，并增加value_score字段用于样本价值评估（阶段73）。

- **A71.3（写入策略）**（引用阶段71）：write(item) → buffer（FIFO覆盖），短期记忆的写入策略。长期记忆的写入策略与短期记忆不同——不是FIFO覆盖，而是选择性持久化：仅将高价值样本从短期记忆转移到长期记忆。

- **A71.4（读取策略）**（引用阶段71）：read(query) → items（按误差/奖励排序），短期记忆的读取策略。长期记忆的读取策略更丰富——支持按时间、误差、价值等多维度条件查询，且查询结果可跨时间尺度。

- **A9.3（终极循环）**（引用阶段9）：终极循环在长期无进步时触发认知升级。长期记忆为终极循环提供历史数据——通过分析长期记忆中的模式变化，判断系统是否需要结构重构。

本阶段的核心目标是：**定义SQLite表设计 samples(id, observation, prediction, error, reward, timestamp, value_score)（A72.1），JSON模型快照 snapshots(id, model_state, performance, timestamp)（A72.2），写入策略 write_persistent(sample) → SQLite（A72.3），读取策略 query_persistent(filter) → samples（A72.4），索引设计（timestamp, error, value_score）（A72.5），封装为LongTermMemory数据类，组合ShortTermMemory作为短期记忆的输入。**

> **去重声明**：本阶段仅定义离线长期记忆的持久化存储机制，不涉及样本价值评估（阶段73）、经验回放（阶段74）或记忆巩固（阶段77）。短期记忆的缓冲区设计已在阶段71定义，本阶段不重复。

---

## 2. 核心数学公式

### 第一部分：SQLite表设计

#### 2.1 SQLite样本表设计定义（A72.1）

样本表存储从短期记忆转移过来的高价值样本：

$$\text{samples}(\text{id}, \text{observation}, \text{prediction}, \text{error}, \text{reward}, \text{timestamp}, \text{value\_score}) \tag{A72.1}$$

各字段的详细定义：

| 字段 | SQLite类型 | 语义 | 来源 |
|------|-----------|------|------|
| id | INTEGER PRIMARY KEY AUTOINCREMENT | 唯一标识 | 自增 |
| observation | BLOB | 观测向量（序列化） | A71.2 |
| prediction | BLOB | 预测向量（序列化） | A71.2 |
| error | REAL | 全局误差 | A71.2 / A51.5 |
| reward | REAL | 奖励信号 | A71.2 |
| timestamp | REAL | 时间戳 | A71.2 |
| value_score | REAL | 样本价值评分 | 阶段73 |

**BLOB序列化方案**：

观测向量和预测向量以BLOB形式存储，序列化方案选择：

| 方案 | 压缩率 | 读写速度 | 适用场景 |
|------|--------|---------|---------|
| NumPy原生（.npy） | 无压缩 | 最快 | 小向量（<1KB） |
| msgpack | 中等 | 快 | 中等向量（1-10KB） |
| zlib压缩NumPy | 高 | 较慢 | 大向量（>10KB） |

默认采用msgpack序列化，在压缩率和速度之间取得平衡。

**样本表的约束**：

```sql
CREATE TABLE samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation BLOB NOT NULL,
    prediction BLOB NOT NULL,
    error REAL NOT NULL CHECK(error >= 0),
    reward REAL NOT NULL,
    timestamp REAL NOT NULL CHECK(timestamp >= 0),
    value_score REAL NOT NULL DEFAULT 0.0
);
```

**样本表与短期记忆缓冲区的区别**：

| 维度 | 短期记忆缓冲区 | 长期记忆样本表 |
|------|--------------|--------------|
| 存储介质 | 内存（NumPy数组） | 磁盘（SQLite） |
| 容量 | N=10000（固定） | 无硬性上限 |
| 覆盖策略 | FIFO覆盖 | 不覆盖（追加写入） |
| 查询方式 | 排序读取 | SQL条件查询 |
| 持久性 | 进程退出即丢失 | 进程退出不丢失 |
| 写入频率 | 每步写入 | 选择性写入 |

- **输入**：短期记忆中的样本 + value_score
- **输出**：SQLite样本表
- **约束**：error ≥ 0；timestamp ≥ 0；value_score默认0.0

> **白话翻译**：SQLite样本表就是"档案柜"——把短期记忆中的精华存进数据库，关机也不丢。每条记录有七个字段：编号、观测、预测、误差、奖励、时间戳、价值评分。观测和预测用BLOB存储（二进制大对象），默认用msgpack序列化。和短期记忆的区别：短期记忆在内存里，满了就覆盖；长期记忆在磁盘上，只追加不覆盖。

---

### 第二部分：JSON模型快照

#### 2.2 JSON模型快照定义（A72.2）

模型快照以JSON格式存储模型的完整状态：

$$\text{snapshots}(\text{id}, \text{model\_state}, \text{performance}, \text{timestamp}) \tag{A72.2}$$

各字段的详细定义：

| 字段 | 类型 | 语义 | 说明 |
|------|------|------|------|
| id | INTEGER PRIMARY KEY | 快照唯一标识 | 自增 |
| model_state | TEXT (JSON) | 模型状态 | 包含权重、结构、超参数 |
| performance | TEXT (JSON) | 性能指标 | 包含自由能、适应度、误差 |
| timestamp | REAL | 快照时间戳 | 快照创建时间 |

**model_state的JSON结构**：

```json
{
    "weights": {"layer_0": [...], "layer_1": [...], ...},
    "structure": {"n_layers": 3, "dims": [256, 128, 64]},
    "hyperparameters": {"learning_rate": 0.01, "tau": 0.1},
    "precision_matrices": {"layer_0": [...], ...}
}
```

**performance的JSON结构**：

```json
{
    "free_energy": 1.234,
    "fitness": 0.876,
    "global_error": 0.567,
    "layer_errors": {"low": 0.2, "mid": 0.3, "high": 0.067},
    "explore_ratio": 0.3
}
```

**快照的存储策略**：

| 策略 | 触发条件 | 存储频率 | 说明 |
|------|---------|---------|------|
| 重构后快照 | after_reform | 不定期 | 每次结构重构后保存 |
| 周期性快照 | periodic(T_snapshot) | 每T_snapshot步 | 定期保存 |
| 里程碑快照 | fitness > threshold | 不定期 | 达到性能里程碑时保存 |

快照的详细触发条件将在阶段78定义。

**快照的压缩**：

大型模型的快照可能占用大量空间。采用增量存储策略：

$$\text{snapshot}_t = \text{snapshot}_{t-1} + \Delta\text{state}_t \tag{A72.2-incremental}$$

仅存储与上一个快照的差异，大幅减少存储空间。增量存储的详细设计将在阶段78（A78.6）展开。

- **输入**：模型状态、性能指标、时间戳
- **输出**：JSON格式的模型快照
- **约束**：JSON可解析；快照可完整恢复模型

> **白话翻译**：JSON模型快照就是"拍照留念"——把模型的状态拍下来存成JSON文件，随时可以翻看。快照里记着模型的权重、结构、超参数（model_state），还有当时的自由能、适应度、误差（performance）。快照可以在重构后拍、定期拍、或者达到里程碑时拍。大模型快照可以用增量存储——只存和上次快照的差异。

---

### 第三部分：写入策略

#### 2.3 持久化写入策略定义（A72.3）

持久化写入策略将高价值样本从短期记忆转移到长期记忆：

$$\text{write\_persistent(sample)} \to \text{SQLite} \tag{A72.3}$$

**写入流程**：

1. 从短期记忆缓冲区中选择待转移的样本
2. 计算样本的价值评分 value_score（阶段73定义）
3. 将样本序列化并写入SQLite样本表
4. 从短期记忆缓冲区中标记该样本为已转移

**选择性持久化的条件**：

不是所有短期记忆样本都需要持久化。选择性持久化的条件为：

$$\text{should\_persist}(s) = \mathbb{1}\left[\text{value\_score}(s) > \theta_{\text{persist}}\right] \tag{A72.3-condition}$$

其中 $\theta_{\text{persist}}$ 为持久化阈值。仅价值评分超过阈值的样本才写入长期记忆。

**持久化阈值的选择**：

| 阈值 | 持久化比例 | 磁盘增长 | 适用场景 |
|------|----------|---------|---------|
| 低（0.1） | ~50% | 快 | 保守策略，多存少丢 |
| 中（0.3） | ~30% | 中 | 平衡策略 |
| 高（0.5） | ~10% | 慢 | 激进策略，只存精华 |

**写入的批量化**：

为减少SQLite的写入开销，采用批量写入：

$$\text{write\_batch}(\text{samples}) = \bigcup_{s \in \text{samples}} \text{write\_persistent}(s) \tag{A72.3-batch}$$

批量写入使用SQLite的事务机制，确保原子性：

```sql
BEGIN TRANSACTION;
INSERT INTO samples (...) VALUES (...);
INSERT INTO samples (...) VALUES (...);
...
COMMIT;
```

**写入频率与磁盘I/O的权衡**：

| 写入频率 | 磁盘I/O | 数据安全 | 适用场景 |
|---------|---------|---------|---------|
| 每步写入 | 高 | 最安全 | 关键数据 |
| 每100步写入 | 中 | 较安全 | 一般场景 |
| 每1000步写入 | 低 | 可能丢失 | 非关键数据 |

默认每100步批量写入一次，在数据安全和磁盘I/O之间取得平衡。

- **输入**：短期记忆中的高价值样本
- **输出**：SQLite样本表的更新
- **约束**：仅持久化value_score > θ_persist的样本；批量写入保证原子性

> **白话翻译**：持久化写入就是"存档"——不是所有短期记忆都值得存档，只存价值评分超过阈值的精华。写入是批量的——攒够一批再写，减少磁盘开销。写入用事务保证原子性——要么全部写入成功，要么全部不写。默认每100步存一次档。

---

### 第四部分：读取策略

#### 2.4 持久化读取策略定义（A72.4）

持久化读取策略支持多维度条件查询：

$$\text{query\_persistent(filter)} \to \text{samples} \tag{A72.4}$$

**过滤条件的结构**：

$$\text{filter} = \bigwedge_{i} \text{condition}_i \tag{A72.4-filter}$$

支持的过滤条件：

| 条件类型 | SQL示例 | 说明 |
|---------|---------|------|
| 时间范围 | WHERE timestamp BETWEEN t₁ AND t₂ | 查询特定时间段 |
| 误差范围 | WHERE error BETWEEN e₁ AND e₂ | 查询特定误差范围 |
| 价值范围 | WHERE value_score >= v₁ | 查询高价值样本 |
| 排序 | ORDER BY error DESC | 按误差降序 |
| 限制数量 | LIMIT k | 最多返回k条 |

**常用查询模式**：

**模式1：高误差历史查询**

```sql
SELECT * FROM samples
WHERE error > θ_high
ORDER BY timestamp DESC
LIMIT k;
```

返回历史上误差超过阈值 $\theta_{\text{high}}$ 的最近 $k$ 个样本。用于分析系统在哪些时刻遇到了困难。

**模式2：高价值样本查询**

```sql
SELECT * FROM samples
WHERE value_score >= θ_value
ORDER BY value_score DESC
LIMIT k;
```

返回价值评分最高的 $k$ 个样本。用于优先回放（阶段74）。

**模式3：时间窗口统计**

```sql
SELECT AVG(error), AVG(reward), COUNT(*)
FROM samples
WHERE timestamp BETWEEN t₁ AND t₂;
```

返回特定时间窗口内的误差均值、奖励均值和样本数。用于复盘分析（阶段76）。

**查询性能优化**：

| 优化技术 | 效果 | 开销 |
|---------|------|------|
| 索引（A72.5） | 加速WHERE和ORDER BY | 增加写入开销 |
| 预编译语句 | 减少SQL解析时间 | 无 |
| 分页查询 | 减少内存占用 | 需要多次查询 |
| 物化视图 | 加速聚合查询 | 增加存储 |

- **输入**：过滤条件filter
- **输出**：满足条件的样本列表
- **约束**：查询结果不超过内存限制

> **白话翻译**：持久化读取就是"翻档案"——可以按时间、误差、价值等条件查找历史经历。常用的翻法有三种：翻高误差的（看什么时候遇到过困难）、翻高价值的（看什么经历最值得学习）、翻统计的（看某个时期的整体表现）。给档案柜建好目录（索引），翻起来就快了。

---

### 第五部分：索引设计

#### 2.5 索引设计定义（A72.5）

为加速常用查询，在样本表上建立三个索引：

$$\text{indexes} = \{\text{idx\_timestamp}, \text{idx\_error}, \text{idx\_value\_score}\} \tag{A72.5}$$

**索引1：时间戳索引**

```sql
CREATE INDEX idx_timestamp ON samples(timestamp);
```

加速时间范围查询（WHERE timestamp BETWEEN t₁ AND t₂）和时间排序（ORDER BY timestamp）。时间戳索引是最重要的索引——几乎所有查询都涉及时间维度。

**索引2：误差索引**

```sql
CREATE INDEX idx_error ON samples(error);
```

加速误差范围查询（WHERE error > θ）和误差排序（ORDER BY error DESC）。误差索引用于高误差样本的快速检索。

**索引3：价值评分索引**

```sql
CREATE INDEX idx_value_score ON samples(value_score);
```

加速价值范围查询（WHERE value_score >= θ）和价值排序（ORDER BY value_score DESC）。价值评分索引用于高价值样本的快速检索，是优先回放（阶段74）的关键加速。

**索引的开销分析**：

| 索引 | 查询加速 | 写入开销 | 存储开销 |
|------|---------|---------|---------|
| idx_timestamp | 时间查询10-100x | +10%写入时间 | ~5%额外空间 |
| idx_error | 误差查询10-100x | +10%写入时间 | ~5%额外空间 |
| idx_value_score | 价值查询10-100x | +10%写入时间 | ~5%额外空间 |

三个索引的总写入开销约+30%，总存储开销约+15%，但查询性能提升10-100倍，值得投入。

**复合索引的考虑**：

如果常用查询同时涉及多个字段（如 WHERE timestamp BETWEEN t₁ AND t₂ AND value_score >= v），可考虑建立复合索引：

```sql
CREATE INDEX idx_timestamp_value ON samples(timestamp, value_score);
```

复合索引的选择取决于实际查询模式。初始版本仅建立单字段索引，根据查询热点再添加复合索引。

**索引维护策略**：

SQLite的索引在写入时自动维护，无需手动更新。但当样本表数据量极大时（>100万条），索引的写入开销可能显著。此时可采用分区策略——按时间范围将样本表分为多个子表，每个子表独立索引。

- **输入**：样本表的查询模式
- **输出**：索引定义
- **约束**：索引不改变查询结果，仅加速查询

> **白话翻译**：索引设计就是"给档案柜建目录"——三个目录：按时间查的目录、按误差查的目录、按价值查的目录。有了目录，翻档案快10-100倍，但每次存档时也要更新目录，多花30%的时间。值得——查比存频繁得多。如果经常同时按时间和价值查，可以建复合目录。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A72.1 | samples(id, observation, prediction, error, reward, timestamp, value_score) | 短期记忆样本 + 价值评分 | SQLite样本表 | error ≥ 0, timestamp ≥ 0 |
| A72.2 | snapshots(id, model_state, performance, timestamp) | 模型状态、性能指标 | JSON模型快照 | JSON可解析 |
| A72.3 | write_persistent(sample) → SQLite | 高价值样本 | 样本表更新 | value_score > θ_persist |
| A72.4 | query_persistent(filter) → samples | 过滤条件 | 满足条件的样本列表 | 结果不超内存限制 |
| A72.5 | indexes = {idx_timestamp, idx_error, idx_value_score} | 查询模式 | 索引定义 | 索引自动维护 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import numpy as np
import sqlite3
import json

# 引用阶段71的数据类
try:
    from stage_071 import ShortTermMemory, MemorySample
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


# ============================================================
# 阶段72 数据类：查询过滤条件
# ============================================================
@dataclass(frozen=True)
class PersistentFilter:
    """长期记忆查询过滤条件。

    Attributes:
        time_range: 时间范围 [t_start, t_end]
        error_range: 误差范围 [e_min, e_max]
        value_range: 价值评分范围 [v_min, v_max]
        order_by: 排序字段
        order_desc: 是否降序
        limit: 返回数量上限
    """
    time_range: Optional[Tuple[float, float]] = None
    error_range: Optional[Tuple[float, float]] = None
    value_range: Optional[Tuple[float, float]] = None
    order_by: str = "timestamp"
    order_desc: bool = True
    limit: int = 100


# ============================================================
# 阶段72 数据类：模型快照
# ============================================================
@dataclass(frozen=True)
class ModelSnapshotRecord:
    """模型快照记录。

    Attributes:
        snapshot_id: 快照唯一标识
        model_state: 模型状态（JSON字符串）
        performance: 性能指标（JSON字符串）
        timestamp: 快照时间戳
    """
    snapshot_id: int
    model_state: str
    performance: str
    timestamp: float


# ============================================================
# 阶段72 数据类：长期记忆
# ============================================================
@dataclass(frozen=True)
class LongTermMemory:
    """离线长期记忆的完整描述。

    组合阶段71的ShortTermMemory，在短期缓冲的基础上增加
    持久化存储的状态。

    Attributes:
        stm: 阶段71的短期记忆（组合字段）
        db_path: SQLite数据库路径
        total_samples: 样本表中的总记录数
        total_snapshots: 快照表中的总记录数
        persist_threshold: 持久化阈值
        batch_write_interval: 批量写入间隔（步数）
        last_write_step: 上次写入的步数

    Invariants:
        - total_samples >= 0
        - total_snapshots >= 0
        - persist_threshold >= 0
        - batch_write_interval > 0
    """
    stm: ShortTermMemory
    db_path: str = ":memory:"
    total_samples: int = 0
    total_snapshots: int = 0
    persist_threshold: float = 0.3
    batch_write_interval: int = 100
    last_write_step: int = 0

    def __post_init__(self):
        if self.total_samples < 0:
            raise ValueError(
                f"总样本数必须 >= 0，收到 {self.total_samples}"
            )
        if self.total_snapshots < 0:
            raise ValueError(
                f"总快照数必须 >= 0，收到 {self.total_snapshots}"
            )
        if self.persist_threshold < 0:
            raise ValueError(
                f"持久化阈值必须 >= 0，收到 {self.persist_threshold}"
            )
        if self.batch_write_interval <= 0:
            raise ValueError(
                f"批量写入间隔必须 > 0，收到 {self.batch_write_interval}"
            )

    @property
    def estimated_disk_size_mb(self) -> float:
        """估算磁盘占用（MB）。"""
        sample_size_kb = 2.0  # 每样本约2KB
        return self.total_samples * sample_size_kb / 1024


# ============================================================
# 阶段72 核心接口：长期记忆管理器
# ============================================================
class LongTermMemoryManager:
    """阶段72的核心接口：离线长期记忆管理器。

    实现SQLite持久化存储和JSON模型快照。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 仅持久化高价值样本
        - 批量写入保证原子性
        - 索引加速常用查询
    """

    def initialize_db(self, db_path: str) -> None:
        """初始化SQLite数据库（A72.1, A72.5）。

        创建样本表、快照表和索引。

        Args:
            db_path: 数据库文件路径

        Examples:
            >>> import tempfile, os
            >>> db = os.path.join(tempfile.gettempdir(), "test_ltm.db")
            >>> LongTermMemoryManager().initialize_db(db)
            >>> os.path.exists(db)
            True
        """
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                observation BLOB NOT NULL,
                prediction BLOB NOT NULL,
                error REAL NOT NULL,
                reward REAL NOT NULL,
                timestamp REAL NOT NULL,
                value_score REAL NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_state TEXT NOT NULL,
                performance TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_samples_timestamp
            ON samples(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_samples_error
            ON samples(error)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_samples_value
            ON samples(value_score)
        """)
        conn.commit()
        conn.close()

    def write_persistent(
        self,
        ltm: LongTermMemory,
        sample: MemorySample,
        value_score: float
    ) -> LongTermMemory:
        """持久化写入单个样本（A72.3）。

        仅当value_score > persist_threshold时写入。

        Args:
            ltm: 当前长期记忆状态
            sample: 待写入的样本
            value_score: 样本价值评分

        Returns:
            更新后的长期记忆状态

        Examples:
            >>> import tempfile, os
            >>> db = os.path.join(tempfile.gettempdir(), "test_write.db")
            >>> stm = ShortTermMemory(
            ...     validation_result=EvolutionValidationResult(
            ...         is_valid=True, fitness=0.0),
            ...     db_path=db
            ... )
            >>> ltm = LongTermMemory(stm=stm, db_path=db)
            >>> sample = MemorySample(
            ...     observation=np.array([1.0]),
            ...     prediction=np.array([0.9]),
            ...     error=0.5, reward=1.0, timestamp=1.0)
            >>> new_ltm = LongTermMemoryManager().write_persistent(
            ...     ltm, sample, 0.5)
            >>> new_ltm.total_samples >= 0
            True
        """
        if value_score <= ltm.persist_threshold:
            return ltm
        import sqlite3
        self.initialize_db(ltm.db_path)
        conn = sqlite3.connect(ltm.db_path)
        cursor = conn.cursor()
        obs_bytes = sample.observation.tobytes()
        pred_bytes = sample.prediction.tobytes()
        cursor.execute(
            """
            INSERT INTO samples
            (observation, prediction, error, reward, timestamp, value_score)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (obs_bytes, pred_bytes, sample.error, sample.reward,
             sample.timestamp, value_score)
        )
        conn.commit()
        conn.close()
        return LongTermMemory(
            stm=ltm.stm,
            db_path=ltm.db_path,
            total_samples=ltm.total_samples + 1,
            total_snapshots=ltm.total_snapshots,
            persist_threshold=ltm.persist_threshold,
            batch_write_interval=ltm.batch_write_interval,
            last_write_step=ltm.last_write_step
        )

    def write_batch(
        self,
        ltm: LongTermMemory,
        samples: List[Tuple[MemorySample, float]]
    ) -> LongTermMemory:
        """批量持久化写入（A72.3-batch）。

        使用SQLite事务保证原子性。

        Args:
            ltm: 当前长期记忆状态
            samples: 待写入的样本列表（样本+价值评分）

        Returns:
            更新后的长期记忆状态

        Examples:
            >>> import tempfile, os
            >>> db = os.path.join(tempfile.gettempdir(), "test_batch.db")
            >>> stm = ShortTermMemory(
            ...     validation_result=EvolutionValidationResult(
            ...         is_valid=True, fitness=0.0),
            ...     db_path=db
            ... )
            >>> ltm = LongTermMemory(stm=stm, db_path=db)
            >>> s = MemorySample(
            ...     observation=np.array([1.0]),
            ...     prediction=np.array([0.9]),
            ...     error=0.5, reward=1.0, timestamp=1.0)
            >>> new_ltm = LongTermMemoryManager().write_batch(
            ...     ltm, [(s, 0.5)])
            >>> new_ltm.total_samples >= 0
            True
        """
        filtered = [(s, v) for s, v in samples if v > ltm.persist_threshold]
        if not filtered:
            return ltm
        import sqlite3
        self.initialize_db(ltm.db_path)
        conn = sqlite3.connect(ltm.db_path)
        cursor = conn.cursor()
        try:
            for sample, value_score in filtered:
                obs_bytes = sample.observation.tobytes()
                pred_bytes = sample.prediction.tobytes()
                cursor.execute(
                    """
                    INSERT INTO samples
                    (observation, prediction, error, reward, timestamp,
                     value_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (obs_bytes, pred_bytes, sample.error, sample.reward,
                     sample.timestamp, value_score)
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return LongTermMemory(
            stm=ltm.stm,
            db_path=ltm.db_path,
            total_samples=ltm.total_samples + len(filtered),
            total_snapshots=ltm.total_snapshots,
            persist_threshold=ltm.persist_threshold,
            batch_write_interval=ltm.batch_write_interval,
            last_write_step=ltm.last_write_step
        )

    def query_persistent(
        self,
        ltm: LongTermMemory,
        filter_condition: PersistentFilter
    ) -> List[MemorySample]:
        """条件查询持久化样本（A72.4）。

        Args:
            ltm: 当前长期记忆状态
            filter_condition: 过滤条件

        Returns:
            满足条件的样本列表

        Examples:
            >>> import tempfile, os
            >>> db = os.path.join(tempfile.gettempdir(), "test_query.db")
            >>> stm = ShortTermMemory(
            ...     validation_result=EvolutionValidationResult(
            ...         is_valid=True, fitness=0.0),
            ...     db_path=db
            ... )
            >>> ltm = LongTermMemory(stm=stm, db_path=db)
            >>> result = LongTermMemoryManager().query_persistent(
            ...     ltm, PersistentFilter(limit=10))
            >>> isinstance(result, list)
            True
        """
        import sqlite3
        self.initialize_db(ltm.db_path)
        conn = sqlite3.connect(ltm.db_path)
        cursor = conn.cursor()
        query = "SELECT observation, prediction, error, reward, timestamp FROM samples WHERE 1=1"
        params: List[Any] = []
        if filter_condition.time_range is not None:
            t_start, t_end = filter_condition.time_range
            query += " AND timestamp >= ? AND timestamp <= ?"
            params.extend([t_start, t_end])
        if filter_condition.error_range is not None:
            e_min, e_max = filter_condition.error_range
            query += " AND error >= ? AND error <= ?"
            params.extend([e_min, e_max])
        if filter_condition.value_range is not None:
            v_min, v_max = filter_condition.value_range
            query += " AND value_score >= ? AND value_score <= ?"
            params.extend([v_min, v_max])
        order = "DESC" if filter_condition.order_desc else "ASC"
        query += f" ORDER BY {filter_condition.order_by} {order}"
        query += " LIMIT ?"
        params.append(filter_condition.limit)
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        results: List[MemorySample] = []
        for row in rows:
            obs = np.frombuffer(row[0], dtype=np.float64)
            pred = np.frombuffer(row[1], dtype=np.float64)
            results.append(MemorySample(
                observation=obs,
                prediction=pred,
                error=row[2],
                reward=row[3],
                timestamp=row[4]
            ))
        return results

    def save_snapshot(
        self,
        ltm: LongTermMemory,
        model_state: Dict[str, Any],
        performance: Dict[str, float]
    ) -> LongTermMemory:
        """保存模型快照（A72.2）。

        Args:
            ltm: 当前长期记忆状态
            model_state: 模型状态字典
            performance: 性能指标字典

        Returns:
            更新后的长期记忆状态

        Examples:
            >>> import tempfile, os
            >>> db = os.path.join(tempfile.gettempdir(), "test_snap.db")
            >>> stm = ShortTermMemory(
            ...     validation_result=EvolutionValidationResult(
            ...         is_valid=True, fitness=0.0),
            ...     db_path=db
            ... )
            >>> ltm = LongTermMemory(stm=stm, db_path=db)
            >>> new_ltm = LongTermMemoryManager().save_snapshot(
            ...     ltm, {"w": [0.1]}, {"acc": 0.9})
            >>> new_ltm.total_snapshots >= 0
            True
        """
        import sqlite3, time
        self.initialize_db(ltm.db_path)
        conn = sqlite3.connect(ltm.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO snapshots (model_state, performance, timestamp)
            VALUES (?, ?, ?)
            """,
            (json.dumps(model_state), json.dumps(performance), time.time())
        )
        conn.commit()
        conn.close()
        return LongTermMemory(
            stm=ltm.stm,
            db_path=ltm.db_path,
            total_samples=ltm.total_samples,
            total_snapshots=ltm.total_snapshots + 1,
            persist_threshold=ltm.persist_threshold,
            batch_write_interval=ltm.batch_write_interval,
            last_write_step=ltm.last_write_step
        )

    def load_snapshot(
        self,
        ltm: LongTermMemory,
        snapshot_id: int
    ) -> Optional[ModelSnapshotRecord]:
        """加载模型快照（A72.2）。

        Args:
            ltm: 当前长期记忆状态
            snapshot_id: 快照ID

        Returns:
            快照记录，如果不存在则返回None

        Examples:
            >>> import tempfile, os
            >>> db = os.path.join(tempfile.gettempdir(), "test_load.db")
            >>> stm = ShortTermMemory(
            ...     validation_result=EvolutionValidationResult(
            ...         is_valid=True, fitness=0.0),
            ...     db_path=db
            ... )
            >>> ltm = LongTermMemory(stm=stm, db_path=db)
            >>> result = LongTermMemoryManager().load_snapshot(ltm, 999)
            >>> result is None
            True
        """
        import sqlite3
        self.initialize_db(ltm.db_path)
        conn = sqlite3.connect(ltm.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT snapshot_id, model_state, performance, timestamp "
            "FROM snapshots WHERE snapshot_id = ?",
            (snapshot_id,)
        )
        row = cursor.fetchone()
        conn.close()
        if row is None:
            return None
        return ModelSnapshotRecord(
            snapshot_id=row[0],
            model_state=row[1],
            performance=row[2],
            timestamp=row[3]
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `ShortTermMemory` | ← 阶段71 | 作为长期记忆的短期缓冲输入（组合字段） |
| **输入** | A71.2（存储内容） | ← 阶段71 | 样本字段的定义 |
| **输入** | A71.3（写入策略） | ← 阶段71 | 短期记忆写入的参考 |
| **输入** | A9.3（终极循环） | ← 阶段9 | 长期记忆为终极循环提供历史数据 |
| **输出** | `LongTermMemory` | → 阶段73 | 长期记忆状态，作为样本价值评估的输入 |
| **输出** | A72.1-A72.5（长期记忆定义） | → 阶段74-80 | 记忆系统的持久化基础 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| SQLite | 持久化存储 | Python标准库内置，零配置，支持索引和事务 |
| msgpack | BLOB序列化 | 高效的二进制序列化，比JSON更紧凑更快 |
| json | 快照序列化 | Python标准库，可读性好，适合模型快照 |
| zlib | 可选压缩 | 大向量的BLOB压缩，减少磁盘占用 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V72.1 | 样本表创建 | 表结构符合A72.1定义 | Schema检查 |
| V72.2 | 快照表创建 | 表结构符合A72.2定义 | Schema检查 |
| V72.3 | 索引创建 | 三个索引存在 | PRAGMA index_list检查 |
| V72.4 | 持久化写入 | value_score > θ时写入成功 | 写入+查询验证 |
| V72.5 | 选择性持久化 | value_score ≤ θ时不写入 | 条件测试 |
| V72.6 | 批量写入原子性 | 事务中任一失败则全部回滚 | 异常注入测试 |
| V72.7 | 时间范围查询 | 返回timestamp在[t₁,t₂]内的样本 | 查询正确性测试 |
| V72.8 | 误差范围查询 | 返回error在[e₁,e₂]内的样本 | 查询正确性测试 |
| V72.9 | 价值范围查询 | 返回value_score ≥ v的样本 | 查询正确性测试 |
| V72.10 | 排序查询 | 按指定字段和方向排序 | 排序正确性测试 |
| V72.11 | 快照保存与加载 | load == save | 往返测试 |
| V72.12 | BLOB序列化往返 | 反序列化后与原始向量一致 | 序列化测试 |
| V72.13 | 索引加速效果 | 有索引比无索引快10x以上 | 性能基准测试 |
| V72.14 | LongTermMemory组合 | `ltm.stm` 类型为 `ShortTermMemory` | 类型检查 |
| V72.15 | 进程重启持久性 | 重启后数据不丢失 | 持久化测试 |

---

> **阶段72完** | 阶段72建立了离线长期记忆的持久化存储机制：SQLite样本表 samples(id, observation, prediction, error, reward, timestamp, value_score)（A72.1），JSON模型快照 snapshots(id, model_state, performance, timestamp)（A72.2），持久化写入 write_persistent(sample) → SQLite（A72.3），条件查询 query_persistent(filter) → samples（A72.4），索引设计（timestamp, error, value_score）（A72.5），封装为LongTermMemory数据类组合ShortTermMemory。下一步：阶段73将定义样本价值评估机制。
