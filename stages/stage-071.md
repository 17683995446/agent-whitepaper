# 阶段71：在线短期记忆——环形缓冲区设计、FIFO写入与排序读取

> **阶段编号**：S71 | **所属层**：第八层·记忆系统层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 环形缓冲区 | "旋转木马"——固定大小的记忆槽，满了就覆盖最旧的 |
> | FIFO覆盖 | "先进先出"——最早的记忆被新记忆挤掉 |
> | 存储内容 | "记忆卡片"——每张卡片记着看到了什么、猜到了什么、错多少、得了多少分 |
> | 误差排序读取 | "先看最离谱的"——按误差大小排列，最出乎意料的排最前 |
> | 奖励排序读取 | "先看最值钱的"——按奖励大小排列，收益最高的排最前 |
> | ShortTermMemory | "短期记忆的完整画像"——组合EvolutionValidationResult，在进化验证基础上增加记忆缓冲 |

---

## 1. 第一性原理依据

本阶段从阶段3（预测编码公理）和阶段51（标准化误差定义）的核心结论出发，建立在线短期记忆的环形缓冲区机制。核心依据为：

- **A3.1（误差信号定义）**（引用阶段3）：$e^{l-1} = r^{l-1} - \hat{r}^{l-1}$，预测误差信号的基本定义。短期记忆的核心存储内容之一即为误差信号——系统需要记住"哪里预测错了"以便后续学习。A3.1为短期记忆中误差字段的语义提供了唯一定义。

- **A51.1（标准化误差向量）**（引用阶段51）：$\boldsymbol{\varepsilon} = [\varepsilon_{\text{low}}, \varepsilon_{\text{mid}}, \varepsilon_{\text{high}}]$，三层预测体系的逐层误差度量。短期记忆中的误差字段采用A51.1的标准化形式——不是原始的欧氏距离，而是精度加权的标准化误差，确保跨层误差的可比性。

- **A51.5（全局误差）**（引用阶段51）：$\varepsilon_{\text{global}} = \sum_l w_l \varepsilon_l$，三层误差的加权聚合。短期记忆中的误差排序以A51.5的全局误差为排序键——全局误差最大的样本优先被读取。

- **A9.1（表层循环）**（引用阶段9）：$\delta_t = F(q_t, o_t)$，表层循环在每一时间步执行感知-行动闭环。短期记忆的写入与表层循环同步——每个时间步产生一个样本写入缓冲区。

本阶段的核心目标是：**定义环形缓冲区设计 buffer[size=N, overwrite=oldest]（A71.1），存储内容 (observation, prediction, error, reward, timestamp)（A71.2），写入策略 write(item) → buffer（FIFO覆盖）（A71.3），读取策略 read(query) → items（按误差/奖励排序）（A71.4），缓冲区大小 N = 10000（可配置）（A71.5），封装为ShortTermMemory数据类，组合EvolutionValidationResult作为进化验证状态的输入。**

> **去重声明**：本阶段仅定义在线短期记忆的缓冲区设计和读写策略，不涉及长期持久化（阶段72）、样本价值评估（阶段73）或经验回放（阶段74）。误差信号的定义遵循A3.1，标准化误差的计算遵循A51.1-A51.5，本阶段不重新定义。

---

## 2. 核心数学公式

### 第一部分：环形缓冲区设计

#### 2.1 环形缓冲区设计定义（A71.1）

环形缓冲区是一个固定大小、循环覆盖的数据结构：

$$\text{buffer}[i] = \begin{cases} \text{item} & \text{if } i = \text{write\_ptr} \mod N \\ \text{previous\_item} & \text{otherwise} \end{cases} \tag{A71.1}$$

其中：
- $N$ 为缓冲区容量（A71.5定义 $N = 10000$）
- $\text{write\_ptr}$ 为写入指针，每写入一个样本递增1
- $\text{overwrite} = \text{oldest}$：当缓冲区满时，覆盖最旧的样本

**环形缓冲区的状态变量**：

| 变量 | 类型 | 含义 |
|------|------|------|
| buffer | Array[N] | 存储样本的数组 |
| write_ptr | int ∈ [0, N) | 下一个写入位置 |
| count | int ∈ [0, N] | 当前缓冲区中的样本数 |
| is_full | bool | 缓冲区是否已满 |

**环形缓冲区的索引映射**：

当缓冲区未满时，有效索引为 $[0, \text{count})$；当缓冲区已满时，有效索引为 $[\text{write\_ptr}, N) \cup [0, \text{write\_ptr})$，即从最旧样本到最新样本的循环序列。

**与线性缓冲区的对比**：

| 维度 | 线性缓冲区 | 环形缓冲区 |
|------|----------|----------|
| 内存分配 | O(N)每次扩容 | O(N)固定 |
| 写入时间 | O(1)均摊 | O(1)最坏 |
| 删除最旧 | O(N)移动 | O(1)指针移动 |
| 缓存友好性 | 差（扩容时复制） | 好（固定位置） |

- **输入**：缓冲区容量 $N$，写入指针位置
- **输出**：环形缓冲区数据结构
- **约束**：$N > 0$；写入指针始终在 $[0, N)$ 范围内

> **白话翻译**：环形缓冲区就是"旋转木马"——N个座位围成一圈，新来的人坐到下一个空位，满了就替换坐得最久的那个人。不需要搬动任何人，只要移动"下一个坐哪"的指针。比线性缓冲区好在：不用扩容复制，不用移动元素，写入时间恒定。

---

### 第二部分：存储内容

#### 2.2 存储内容定义（A71.2）

每个短期记忆样本包含五个字段：

$$\text{item} = (\text{observation}, \text{prediction}, \text{error}, \text{reward}, \text{timestamp}) \tag{A71.2}$$

各字段的详细定义：

**observation（观测向量）**：

$$\text{observation} \in \mathbb{R}^{d_{\text{obs}}}$$

当前时间步的观测输入。在三层预测体系中，observation包含底层观测 $\mathbf{o}_{\text{low}}$、中层趋势 $\mathbf{o}_{\text{mid}}$ 和高层模式 $\mathbf{o}_{\text{high}}$ 的拼接。

**prediction（预测向量）**：

$$\text{prediction} \in \mathbb{R}^{d_{\text{pred}}}$$

模型对当前时间步的预测输出。与observation对应，包含底层预测 $\hat{\mathbf{o}}_{\text{low}}$、中层趋势预测 $\hat{\mathbf{o}}_{\text{mid}}$ 和高层模式预测 $\hat{\mathbf{o}}_{\text{high}}$ 的拼接。

**error（误差标量）**：

$$\text{error} = \varepsilon_{\text{global}} = \sum_l w_l \varepsilon_l \tag{A71.2-error}$$

采用A51.5（引用阶段51）的全局误差作为误差度量。全局误差是三层标准化误差的加权聚合，确保跨层误差的可比性。此字段是后续排序读取（A71.4）和样本价值评估（阶段73）的核心输入。

**reward（奖励标量）**：

$$\text{reward} \in \mathbb{R}$$

当前时间步的外部奖励信号。奖励来源包括：生存奖励（引用阶段2，生存概率的增量）、目标达成奖励（引用阶段7，偏好分布的匹配度）和探索奖励（引用阶段7，认知价值的实现）。

**timestamp（时间戳）**：

$$\text{timestamp} \in \mathbb{R}_{\geq 0}$$

样本产生的时间戳，用于时序排序和遗忘曲线计算（阶段77）。时间戳的精度应足以区分相邻时间步的样本。

**存储内容的内存估算**：

| 字段 | 维度 | 精度 | 单样本内存 |
|------|------|------|----------|
| observation | $d_{\text{obs}}$ | float32 | $4 d_{\text{obs}}$ 字节 |
| prediction | $d_{\text{pred}}$ | float32 | $4 d_{\text{pred}}$ 字节 |
| error | 1 | float32 | 4 字节 |
| reward | 1 | float32 | 4 字节 |
| timestamp | 1 | float64 | 8 字节 |

当 $d_{\text{obs}} = d_{\text{pred}} = 256$ 时，单样本约 2KB，N=10000时总内存约 20MB。

- **输入**：观测、预测、误差、奖励、时间戳
- **输出**：封装的样本元组
- **约束**：error ≥ 0（由A51.5非负性保证）；timestamp单调递增

> **白话翻译**：存储内容就是"记忆卡片"——每张卡片记着五件事：看到了什么（observation）、猜到了什么（prediction）、错多少（error）、得了多少分（reward）、什么时候发生的（timestamp）。误差用全局误差（A51.5），不是简单的欧氏距离，而是三层精度加权的标准化度量。

---

### 第三部分：写入策略

#### 2.3 写入策略定义（A71.3）

写入策略采用FIFO（First-In-First-Out）覆盖方式：

$$\text{write(item)} \to \begin{cases} \text{buffer}[\text{write\_ptr}] = \text{item} \\ \text{write\_ptr} = (\text{write\_ptr} + 1) \mod N \\ \text{count} = \min(\text{count} + 1, N) \end{cases} \tag{A71.3}$$

**写入流程**：

1. 将item写入buffer[write_ptr]位置
2. write_ptr递增并取模N
3. 如果缓冲区未满，count递增
4. 如果缓冲区已满，最旧的样本被自动覆盖（无需显式删除）

**写入的原子性保证**：

写入操作必须是原子的——要么完整写入，要么不写入。在并发场景下，写入操作需要加锁：

$$\text{write\_lock} \implies \text{atomic}(\text{buffer}[\text{write\_ptr}] = \text{item}, \text{write\_ptr} \mathrel{+}= 1) \tag{A71.3-atomic}$$

**写入频率与表层循环的关系**：

由A9.1（引用阶段9），表层循环在每个时间步执行。短期记忆的写入与表层循环同步——每个时间步产生一个样本并写入缓冲区：

$$\text{write\_frequency} = \text{surface\_loop\_frequency} \tag{A71.3-freq}$$

**写入时的可选预处理**：

写入前可对样本进行轻量级预处理：

| 预处理 | 开销 | 效果 |
|--------|------|------|
| 误差归一化 | O(1) | 使不同时间段的误差可比 |
| 奖励裁剪 | O(1) | 防止极端奖励值影响排序 |
| 观测降维 | O(d) | 减少存储开销 |

- **输入**：待写入的样本item
- **输出**：更新后的缓冲区状态
- **约束**：写入时间O(1)；原子性保证

> **白话翻译**：写入策略就是"先进先出"——新记忆写入下一个空位，满了就覆盖最旧的。写入和表层循环同步——每步写一个样本。写入必须原子化——不能写一半被打断。写入前可以做轻量级预处理，比如误差归一化、奖励裁剪。

---

### 第四部分：读取策略

#### 2.4 读取策略定义（A71.4）

读取策略支持按误差或奖励排序返回样本：

$$\text{read(query)} \to \text{items} \tag{A71.4}$$

**查询类型**：

| 查询类型 | 排序键 | 方向 | 适用场景 |
|---------|--------|------|---------|
| top_error | error | 降序 | 优先学习高误差样本 |
| top_reward | reward | 降序 | 优先学习高收益样本 |
| recent | timestamp | 降序 | 获取最近样本 |
| range | timestamp ∈ [t₁, t₂] | 升序 | 时间范围查询 |

**按误差排序读取**：

$$\text{items} = \text{sort}(\text{buffer}, \text{key}=\text{error}, \text{descending})[:k] \tag{A71.4-error}$$

返回误差最大的前 $k$ 个样本。误差排序的物理含义：误差最大的样本是模型"最出乎意料"的经历，优先学习这些样本可以最大程度降低自由能。

**按奖励排序读取**：

$$\text{items} = \text{sort}(\text{buffer}, \text{key}=\text{reward}, \text{descending})[:k] \tag{A71.4-reward}$$

返回奖励最高的前 $k$ 个样本。奖励排序的物理含义：高奖励样本是模型"最有价值"的经历，优先学习这些样本可以最大化期望收益。

**读取的时间复杂度**：

| 查询类型 | 朴素实现 | 优化实现 |
|---------|---------|---------|
| top_error | O(N log N) 排序 | O(N + k log N) 堆选择 |
| top_reward | O(N log N) 排序 | O(N + k log N) 堆选择 |
| recent | O(N) 线性扫描 | O(k) 直接索引 |
| range | O(N) 线性扫描 | O(log N + m) 二分查找 |

其中 $m$ 为时间范围内的样本数。

**读取与写入的并发控制**：

读取操作不应阻塞写入操作。采用读写锁策略：

$$\text{read\_lock} \implies \text{snapshot}(\text{buffer}) \to \text{read\_from\_snapshot} \tag{A71.4-concurrent}$$

读取时先对缓冲区做快照，再从快照中读取，避免读写冲突。

- **输入**：查询类型、排序键、返回数量 $k$
- **输出**：排序后的样本列表
- **约束**：返回数量 $k \leq \text{count}$；读取不阻塞写入

> **白话翻译**：读取策略就是"先看最重要的"——可以按误差排序（最离谱的排最前），也可以按奖励排序（最值钱的排最前），还可以按时间排序（最近的排最前）。读取时先拍快照再读，不阻塞写入。排序用堆选择比全排序快——不需要把所有记忆都排好，只要找到前k个就行。

---

### 第五部分：缓冲区大小

#### 2.5 缓冲区大小定义（A71.5）

缓冲区大小默认为10000，可配置：

$$N = 10000 \quad \text{(可配置)} \tag{A71.5}$$

**缓冲区大小的选择依据**：

| 因素 | 分析 | 结论 |
|------|------|------|
| 表层循环频率 | 约10Hz（引用A9.1） | 10000步 ≈ 1000秒 ≈ 17分钟 |
| 中层循环频率 | 约0.1Hz（引用A9.2） | 10000步覆盖约100个中层周期 |
| 终极循环频率 | 约0.001Hz（引用A9.3） | 10000步覆盖约1个终极周期 |
| 内存开销 | 单样本约2KB | 10000样本约20MB |
| 采样多样性 | 优先回放需要足够样本池 | 10000 > 32 × 300（batch_size × 多样性余量） |

**缓冲区大小与记忆深度的关系**：

缓冲区大小决定了短期记忆的"时间深度"——系统能回溯多远的历史：

$$T_{\text{depth}} = \frac{N}{f_{\text{surface}}} \tag{A71.5-depth}$$

其中 $f_{\text{surface}}$ 为表层循环频率。当 $N = 10000$、$f_{\text{surface}} = 10\text{Hz}$ 时，$T_{\text{depth}} = 1000$ 秒。

**缓冲区大小的动态调整**：

缓冲区大小可根据系统资源动态调整：

$$N_{\text{adjusted}} = \min\left(N_{\text{config}}, \frac{\text{memory\_budget}}{\text{size\_per\_sample}}\right) \tag{A71.5-dynamic}$$

当系统内存紧张时，自动缩小缓冲区以适应资源约束。

**缓冲区大小与回放效果的关系**：

| N | 记忆深度 | 回放多样性 | 内存开销 |
|---|---------|----------|---------|
| 1000 | 100秒 | 低 | 2MB |
| 5000 | 500秒 | 中 | 10MB |
| 10000 | 1000秒 | 高 | 20MB |
| 50000 | 5000秒 | 极高 | 100MB |

N=10000在记忆深度、回放多样性和内存开销之间取得了良好平衡。

- **输入**：配置值或资源预算
- **输出**：缓冲区大小 $N$
- **约束**：$N > 0$；$N$ 为2的幂次时索引计算更高效

> **白话翻译**：缓冲区大小默认10000——大约能记住17分钟的经历。这个数字不是随便定的：足够覆盖100个中层周期和1个终极周期，足够给优先回放提供多样样本池，内存开销也就20MB。如果内存紧张可以缩小，但太小会影响回放效果——样本池不够多样，学来学去都是那几个样本。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A71.1 | buffer[i] = item if i = write_ptr mod N | 容量N、写入指针 | 环形缓冲区 | N > 0 |
| A71.2 | item = (observation, prediction, error, reward, timestamp) | 观测、预测、误差、奖励、时间戳 | 样本元组 | error ≥ 0 |
| A71.3 | write(item) → buffer[write_ptr], ptr = (ptr+1) mod N | 样本 | 更新后缓冲区 | O(1)写入 |
| A71.4 | read(query) → sort(buffer, key) [:k] | 查询类型、排序键、数量k | 排序后样本列表 | k ≤ count |
| A71.5 | N = 10000 (可配置) | 配置值或资源预算 | 缓冲区大小 | N > 0 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum
import numpy as np

# 引用阶段70的数据类
try:
    from stage_070 import EvolutionValidationResult
except ImportError:
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class EvolutionValidationResult:
        is_valid: bool
        fitness: float


# ============================================================
# 阶段71 枚举：查询类型
# ============================================================
class QueryType(Enum):
    """短期记忆读取查询类型。"""
    TOP_ERROR = "top_error"      # 按误差降序
    TOP_REWARD = "top_reward"    # 按奖励降序
    RECENT = "recent"            # 按时间降序
    RANGE = "range"              # 时间范围查询


# ============================================================
# 阶段71 数据类：记忆样本
# ============================================================
@dataclass(frozen=True)
class MemorySample:
    """短期记忆中的单个样本。

    Attributes:
        observation: 观测向量
        prediction: 预测向量
        error: 全局误差（A51.5标准化误差）
        reward: 奖励信号
        timestamp: 时间戳

    Invariants:
        - error >= 0
        - timestamp >= 0
    """
    observation: np.ndarray
    prediction: np.ndarray
    error: float
    reward: float
    timestamp: float

    def __post_init__(self):
        if self.error < 0:
            raise ValueError(
                f"误差必须 >= 0，收到 {self.error}"
            )
        if self.timestamp < 0:
            raise ValueError(
                f"时间戳必须 >= 0，收到 {self.timestamp}"
            )


# ============================================================
# 阶段71 数据类：短期记忆
# ============================================================
@dataclass(frozen=True)
class ShortTermMemory:
    """在线短期记忆的完整描述。

    组合阶段70的EvolutionValidationResult，在进化验证状态的基础上
    增加环形缓冲区的短期记忆。

    Attributes:
        validation_result: 阶段70的进化验证结果（组合字段）
        buffer_size: 缓冲区容量N
        write_ptr: 写入指针
        count: 当前样本数
        is_full: 缓冲区是否已满
        samples: 缓冲区中的样本列表

    Invariants:
        - buffer_size > 0
        - 0 <= write_ptr < buffer_size
        - 0 <= count <= buffer_size
        - is_full == (count == buffer_size)
    """
    validation_result: EvolutionValidationResult
    buffer_size: int = 10000
    write_ptr: int = 0
    count: int = 0
    is_full: bool = False
    samples: Tuple[Optional[MemorySample], ...] = ()

    def __post_init__(self):
        if self.buffer_size <= 0:
            raise ValueError(
                f"缓冲区大小必须 > 0，收到 {self.buffer_size}"
            )
        if not (0 <= self.write_ptr < self.buffer_size):
            raise ValueError(
                f"写入指针必须在 [0, {self.buffer_size}) 范围内，"
                f"收到 {self.write_ptr}"
            )
        if not (0 <= self.count <= self.buffer_size):
            raise ValueError(
                f"样本数必须在 [0, {self.buffer_size}] 范围内，"
                f"收到 {self.count}"
            )

    @property
    def memory_depth_seconds(self) -> float:
        """记忆深度（秒），假设10Hz表层循环。"""
        return self.count / 10.0

    @property
    def utilization(self) -> float:
        """缓冲区利用率。"""
        return self.count / self.buffer_size


# ============================================================
# 阶段71 核心接口：短期记忆管理器
# ============================================================
class ShortTermMemoryManager:
    """阶段71的核心接口：在线短期记忆管理器。

    实现环形缓冲区的写入和读取策略。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 写入时间O(1)
        - FIFO覆盖最旧样本
        - 读取不阻塞写入
    """

    def write(
        self,
        stm: ShortTermMemory,
        item: MemorySample
    ) -> ShortTermMemory:
        """写入样本到环形缓冲区（A71.3）。

        FIFO覆盖策略：缓冲区满时覆盖最旧样本。

        Args:
            stm: 当前短期记忆状态
            item: 待写入的样本

        Returns:
            更新后的短期记忆状态

        Examples:
            >>> stm = ShortTermMemoryManager().create_buffer(
            ...     EvolutionValidationResult(is_valid=True, fitness=0.0)
            ... )
            >>> sample = MemorySample(
            ...     observation=np.array([1.0]),
            ...     prediction=np.array([0.9]),
            ...     error=0.1,
            ...     reward=1.0,
            ...     timestamp=1.0
            ... )
            >>> new_stm = ShortTermMemoryManager().write(stm, sample)
            >>> new_stm.count
            1
        """
        samples_list: List[Optional[MemorySample]] = list(stm.samples)
        if len(samples_list) < stm.buffer_size:
            samples_list.append(item)
        else:
            samples_list[stm.write_ptr] = item
        new_count = min(stm.count + 1, stm.buffer_size)
        new_ptr = (stm.write_ptr + 1) % stm.buffer_size
        new_is_full = new_count == stm.buffer_size
        return ShortTermMemory(
            validation_result=stm.validation_result,
            buffer_size=stm.buffer_size,
            write_ptr=new_ptr,
            count=new_count,
            is_full=new_is_full,
            samples=tuple(samples_list)
        )

    def read(
        self,
        stm: ShortTermMemory,
        query_type: QueryType,
        k: int,
        time_range: Optional[Tuple[float, float]] = None
    ) -> List[MemorySample]:
        """从缓冲区读取样本（A71.4）。

        按查询类型排序返回前k个样本。

        Args:
            stm: 当前短期记忆状态
            query_type: 查询类型
            k: 返回数量
            time_range: 时间范围（仅RANGE查询需要）

        Returns:
            排序后的样本列表

        Raises:
            ValueError: 当k > count时
            ValueError: 当RANGE查询但time_range为None时

        Examples:
            >>> stm = ShortTermMemoryManager().create_buffer(
            ...     EvolutionValidationResult(is_valid=True, fitness=0.0)
            ... )
            >>> stm = ShortTermMemoryManager().write(stm, MemorySample(
            ...     observation=np.array([1.0]), prediction=np.array([0.9]),
            ...     error=0.5, reward=1.0, timestamp=1.0))
            >>> result = ShortTermMemoryManager().read(stm, QueryType.TOP_ERROR, 1)
            >>> len(result)
            1
        """
        if k > stm.count:
            raise ValueError(
                f"请求数量k({k})不能超过当前样本数({stm.count})"
            )
        if query_type == QueryType.RANGE and time_range is None:
            raise ValueError(
                "RANGE查询必须提供time_range参数"
            )
        valid_samples = [s for s in stm.samples if s is not None]
        if query_type == QueryType.TOP_ERROR:
            sorted_samples = sorted(
                valid_samples, key=lambda x: x.error, reverse=True
            )
        elif query_type == QueryType.TOP_REWARD:
            sorted_samples = sorted(
                valid_samples, key=lambda x: x.reward, reverse=True
            )
        elif query_type == QueryType.RECENT:
            sorted_samples = sorted(
                valid_samples, key=lambda x: x.timestamp, reverse=True
            )
        elif query_type == QueryType.RANGE:
            t_start, t_end = time_range
            sorted_samples = [
                s for s in valid_samples
                if t_start <= s.timestamp <= t_end
            ]
            sorted_samples.sort(key=lambda x: x.timestamp)
        else:
            raise ValueError(f"未知的查询类型: {query_type}")
        return sorted_samples[:k]

    def read_by_error(
        self,
        stm: ShortTermMemory,
        k: int
    ) -> List[MemorySample]:
        """按误差降序读取前k个样本（A71.4-error）。

        Args:
            stm: 当前短期记忆状态
            k: 返回数量

        Returns:
            误差最大的k个样本

        Examples:
            >>> stm = ShortTermMemoryManager().create_buffer(
            ...     EvolutionValidationResult(is_valid=True, fitness=0.0)
            ... )
            >>> stm = ShortTermMemoryManager().write(stm, MemorySample(
            ...     observation=np.array([1.0]), prediction=np.array([0.9]),
            ...     error=0.5, reward=1.0, timestamp=1.0))
            >>> result = ShortTermMemoryManager().read_by_error(stm, 1)
            >>> result[0].error
            0.5
        """
        return self.read(stm, QueryType.TOP_ERROR, k)

    def read_by_reward(
        self,
        stm: ShortTermMemory,
        k: int
    ) -> List[MemorySample]:
        """按奖励降序读取前k个样本（A71.4-reward）。

        Args:
            stm: 当前短期记忆状态
            k: 返回数量

        Returns:
            奖励最高的k个样本

        Examples:
            >>> stm = ShortTermMemoryManager().create_buffer(
            ...     EvolutionValidationResult(is_valid=True, fitness=0.0)
            ... )
            >>> stm = ShortTermMemoryManager().write(stm, MemorySample(
            ...     observation=np.array([1.0]), prediction=np.array([0.9]),
            ...     error=0.5, reward=1.0, timestamp=1.0))
            >>> result = ShortTermMemoryManager().read_by_reward(stm, 1)
            >>> result[0].reward
            1.0
        """
        return self.read(stm, QueryType.TOP_REWARD, k)

    def create_buffer(
        self,
        validation_result: EvolutionValidationResult,
        buffer_size: int = 10000
    ) -> ShortTermMemory:
        """创建短期记忆缓冲区（A71.1, A71.5）。

        Args:
            validation_result: 进化验证结果
            buffer_size: 缓冲区大小，必须 > 0

        Returns:
            初始化的短期记忆状态

        Raises:
            ValueError: 当buffer_size <= 0时

        Examples:
            >>> stm = ShortTermMemoryManager().create_buffer(
            ...     EvolutionValidationResult(is_valid=True, fitness=0.0),
            ...     buffer_size=1000
            ... )
            >>> stm.buffer_size
            1000
            >>> stm.count
            0
        """
        if buffer_size <= 0:
            raise ValueError(
                f"缓冲区大小必须 > 0，收到 {buffer_size}"
            )
        return ShortTermMemory(
            validation_result=validation_result,
            buffer_size=buffer_size,
            write_ptr=0,
            count=0,
            is_full=False,
            samples=tuple([None] * buffer_size)
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `EvolutionValidationResult` | ← 阶段70 | 作为短期记忆的进化验证状态输入（组合字段） |
| **输入** | A3.1（误差信号定义） | ← 阶段3 | 误差字段的语义定义 |
| **输入** | A51.1-A51.5（标准化误差） | ← 阶段51 | 误差字段的计算方式 |
| **输入** | A9.1（表层循环） | ← 阶段9 | 写入频率的同步依据 |
| **输出** | `ShortTermMemory` | → 阶段72 | 短期记忆状态，作为长期记忆的输入 |
| **输出** | A71.1-A71.5（短期记忆定义） | → 阶段73-80 | 记忆系统的缓冲区基础 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| NumPy | 环形缓冲区实现 | 支撑固定大小数组的O(1)写入和向量化排序 |
| collections.deque | 备选缓冲区实现 | Python标准库的双端队列，支持maxlen自动FIFO覆盖 |
| heapq | 堆选择算法 | O(N + k log N)的top-k选择，避免全排序 |
| JAX | 向量化排序 | 大规模缓冲区的GPU加速排序 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V71.1 | 环形缓冲区写入 | write(item)后buffer[write_ptr] == item | 单步写入测试 |
| V71.2 | FIFO覆盖 | 缓冲区满后写入覆盖最旧样本 | N+1次写入测试 |
| V71.3 | 写入指针循环 | write_ptr在[0, N)内循环 | 2N次写入测试 |
| V71.4 | 误差非负 | 所有样本error >= 0 | 随机样本测试 |
| V71.5 | 时间戳单调递增 | 相邻样本timestamp递增 | 时序写入测试 |
| V71.6 | 按误差排序读取 | 返回样本按error降序排列 | 排序正确性测试 |
| V71.7 | 按奖励排序读取 | 返回样本按reward降序排列 | 排序正确性测试 |
| V71.8 | 按时间排序读取 | 返回样本按timestamp降序排列 | 排序正确性测试 |
| V71.9 | 时间范围查询 | 返回样本在[t₁, t₂]范围内 | 范围查询测试 |
| V71.10 | 返回数量限制 | 返回样本数 ≤ k | 边界测试 |
| V71.11 | 缓冲区大小可配置 | N可配置且生效 | 不同N值测试 |
| V71.12 | 写入时间O(1) | 单次写入时间与N无关 | 性能基准测试 |
| V71.13 | ShortTermMemory组合 | `stm.validation_result` 类型为 `EvolutionValidationResult` | 类型检查 |
| V71.14 | 非法参数拒绝 | buffer_size ≤ 0 抛出 `ValueError` | 异常测试 |
| V71.15 | 内存开销 | N=10000时内存 < 50MB | 内存监控测试 |

---

> **阶段71完** | 阶段71建立了在线短期记忆的环形缓冲区机制：环形缓冲区设计 buffer[size=N, overwrite=oldest]（A71.1），存储内容 (observation, prediction, error, reward, timestamp)（A71.2），写入策略 write(item) → buffer（FIFO覆盖）（A71.3），读取策略 read(query) → items（按误差/奖励排序）（A71.4），缓冲区大小 N = 10000（可配置）（A71.5），封装为ShortTermMemory数据类组合EvolutionValidationResult。下一步：阶段72将定义离线长期记忆的持久化存储。
