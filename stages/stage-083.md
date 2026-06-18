# 阶段83：本地存储自动清理

> **阶段编号**：S83 | **所属层**：第九层·安全护栏层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 存储监控 | "看硬盘"——实时查看自己占了多少磁盘空间 |
> | 清理触发 | "硬盘快满了"——磁盘占用超过80%就开始清理 |
> | 清理策略 | "扔旧货"——只留Top30%高价值样本，其余删除 |
> | 清理顺序 | "先扔不值钱的"——低价值先删，高价值最后删 |
> | 清理后验证 | "看看还剩多少"——清理后确认磁盘降到60%以下 |
> | StorageCleaner | "存储清理的完整画像"——组合MemoryMonitor，在内存监控基础上增加存储清理状态 |

---

## 1. 第一性原理依据

本阶段从阶段73（Top30%高价值样本）和阶段82（内存监控）的核心结论出发，建立本地存储自动清理机制。核心依据为：

- **A73.4（Top30%高价值样本筛选）**（引用阶段73）：select_top_k(samples, k=0.3·|samples|)，按价值评分筛选前30%的高价值样本。本阶段的清理策略直接复用A73.4的筛选结果——保留Top30%高价值样本，删除其余70%。

- **A82.1-A82.4（内存监控与释放）**（引用阶段82）：内存监控的度量、判定和释放机制。存储清理与内存释放同属资源管理，但存储与内存有本质区别：内存是易失性资源（进程退出即释放），存储是持久性资源（数据长期占用磁盘）。存储清理需要更谨慎的策略——删除的数据无法恢复。

本阶段的核心目标是：**定义存储监控 disk_usage = disk_agent / disk_total（A83.1），清理触发 disk_usage > 0.80（A83.2），清理策略 保留Top30%高价值样本，删除其余（A83.3），清理顺序 低价值→中价值→高价值（仅极端情况）（A83.4），清理后验证 disk_usage < 0.60（A83.5），封装为StorageCleaner数据类，组合MemoryMonitor作为内存监控的输入。**

> **去重声明**：本阶段仅定义本地存储的自动清理机制，不重新定义样本价值评估（阶段73）或内存释放策略（阶段82）。清理策略中的Top30%筛选调用A73.4的结果，本阶段不重新实现。

---

## 2. 核心数学公式

### 第一部分：存储监控

#### 2.1 存储监控定义（A83.1）

存储监控持续度量智能体的磁盘占用率：

$$\text{disk\_usage} = \frac{\text{disk\_agent}}{\text{disk\_total}} \tag{A83.1}$$

其中：
- $\text{disk\_agent}$ 为智能体占用的磁盘空间总量
- $\text{disk\_total}$ 为本地可用磁盘空间总量
- $\text{disk\_usage} \in [0, 1]$ 为智能体磁盘占用率

**磁盘占用的组成分析**：

| 组件 | 典型占用 | 说明 |
|------|---------|------|
| SQLite数据库 | 40-60% | 长期记忆样本+快照 |
| 模型权重文件 | 20-30% | 快照的权重数据 |
| VAE编解码器 | 5-10% | 生成式回放模型 |
| 日志文件 | 5-10% | 运行日志、统计 |
| 临时文件 | 2-5% | 增量差值、缓存 |

$$\text{disk\_agent} = \text{disk\_db} + \text{disk\_weights} + \text{disk\_vae} + \text{disk\_logs} + \text{disk\_temp} \tag{A83.1-breakdown}$$

**存储监控的采样频率**：

与CPU和内存不同，磁盘空间变化缓慢，采样频率可以更低：

| 状态 | 采样间隔 | 说明 |
|------|---------|------|
| 正常（disk_usage < 0.60） | 60秒 | 低频采样 |
| 警戒（0.60 ≤ disk_usage < 0.80） | 10秒 | 中频采样 |
| 超限（disk_usage ≥ 0.80） | 5秒 | 高频采样 |

**存储增长趋势检测**：

$$\text{disk\_trend} = \frac{\text{disk\_usage}_t - \text{disk\_usage}_{t-W}}{W}, \quad W = 3600\text{s} \tag{A83.1-trend}$$

磁盘增长趋势以1小时为窗口，比内存的5分钟更长。

- **输入**：系统磁盘总量、智能体磁盘使用量
- **输出**：disk_usage（瞬时值、增长趋势）
- **约束**：采样开销可忽略

> **白话翻译**：存储监控就是"看硬盘"——实时查看自己占了多少磁盘。磁盘和内存不同：内存是临时用的，磁盘是长期存的。磁盘变化慢，60秒看一次就行，快满了才10秒看一次。增长趋势以1小时为窗口，比内存的5分钟更慢。

---

### 第二部分：清理触发

#### 2.2 清理触发定义（A83.2）

当磁盘占用率超过80%时触发清理：

$$\text{should\_clean} = \mathbb{1}\left[\text{disk\_usage} > \theta_{\text{disk\_max}}\right], \quad \theta_{\text{disk\_max}} = 0.80 \tag{A83.2}$$

**80%触发阈值的选择依据**：

磁盘与CPU/内存不同，80%的触发阈值远高于20%。原因如下：

| 资源 | 触发阈值 | 原因 |
|------|---------|------|
| CPU | 20% | CPU是共享资源，智能体不应独占 |
| 内存 | 20% | 内存超限导致OOM崩溃 |
| 磁盘 | 80% | 磁盘是独占资源，智能体可使用大部分空间 |

智能体的磁盘数据（样本、快照）是其核心资产，不应像CPU/内存那样严格限制。80%的阈值确保智能体有足够空间存储数据，同时为系统和其他进程保留20%的余量。

**清理触发的预警机制**：

| 级别 | 条件 | 动作 |
|------|------|------|
| 绿色 | disk_usage < 0.60 | 正常运行 |
| 黄色 | 0.60 ≤ disk_usage < 0.80 | 预警，准备清理 |
| 红色 | disk_usage ≥ 0.80 | 立即清理 |

黄色预警阶段不执行清理，但提前计算Top30%筛选结果，以便红色阶段快速执行。

**清理触发的趋势增强**：

$$\text{trend\_trigger} = \mathbb{1}\left[\text{disk\_trend} > \theta_{\text{disk\_fast}} \;\wedge\; \text{disk\_usage} > 0.70\right] \tag{A83.2-trend}$$

其中 $\theta_{\text{disk\_fast}} = 0.0001/\text{s}$（每小时增长36%）。当磁盘快速增长且已占70%时，提前触发清理。

- **输入**：disk_usage、增长趋势
- **输出**：是否应清理
- **约束**：80%触发；70%+快速增长提前触发

> **白话翻译**：清理触发就是"硬盘快满了"——磁盘占用超过80%就开始清理。为什么是80%而不是20%？因为磁盘是智能体的核心资产（样本、快照），可以多用，但也不能全占满。如果磁盘增长很快且已占70%，也提前触发清理。

---

### 第三部分：清理策略

#### 2.3 清理策略定义（A83.3）

清理策略保留Top30%高价值样本，删除其余70%：

$$\text{clean}() = \text{delete}\left(\{s : \text{value}(s) < \text{percentile}_{30}(\text{all\_values})\}\right) \tag{A83.3}$$

**清理策略与A73.4的对应**：

A73.4（引用阶段73）定义了Top30%高价值样本筛选：select_top_k(samples, k=0.3·|samples|)。本阶段的清理策略是A73.4的逆操作——保留Top30%，删除Bottom70%。

**清理的磁盘释放量估算**：

| 清理范围 | 释放磁盘 | 占总磁盘比 |
|---------|---------|-----------|
| Bottom 70%样本 | ~70% disk_db | ~28-42% disk_agent |
| Bottom 70%快照 | ~70% disk_weights | ~14-21% disk_agent |
| 总计 | — | ~42-63% disk_agent |

清理Bottom70%可释放约42-63%的磁盘空间，效果显著。

**清理策略的分级执行**：

为避免一次性删除大量数据导致系统波动，清理分批执行：

$$\text{batch\_size} = \min\left(1000, \left\lfloor \frac{\text{total\_to\_delete}}{10} \right\rfloor\right) \tag{A83.3-batch}$$

每批最多删除1000个样本或快照，分10批完成。每批之间检查disk_usage，如果已降到目标以下则停止。

**清理策略的数据安全**：

删除前将待删除数据标记为"待清理"状态，而非立即物理删除：

$$\text{mark\_for\_cleanup}(s) \implies \text{status}(s) = \text{"pending\_cleanup"} \tag{A83.3-mark}$$

标记后等待一个确认周期（30秒），确认无其他进程正在使用这些数据后再物理删除。这防止了清理过程中其他子系统读取已删除数据导致的错误。

- **输入**：样本价值评分、Top30%筛选结果
- **输出**：清理后的磁盘状态
- **约束**：保留Top30%高价值样本

> **白话翻译**：清理策略就是"扔旧货"——只留Top30%高价值样本，其余70%全删。这能释放42-63%的磁盘空间。不是一次性全删，而是分批删，每批最多1000个，中间检查磁盘是否已降到目标。删之前先标记"待清理"，等30秒确认没人用了再真删。

---

### 第四部分：清理顺序

#### 2.4 清理顺序定义（A83.4）

清理按价值从低到高的顺序执行：

$$\text{clean\_order} = \text{sort\_by\_value\_ascending}(\text{candidates}) \tag{A83.4}$$

**三级清理顺序**：

| 优先级 | 价值范围 | 清理对象 | 触发条件 |
|--------|---------|---------|---------|
| 第一优先 | 价值 < percentile_30 | 低价值样本 | disk_usage > 0.80 |
| 第二优先 | percentile_30 ≤ 价值 < percentile_60 | 中价值样本 | 第一优先清理后仍 > 0.60 |
| 第三优先 | percentile_60 ≤ 价值 < percentile_90 | 高价值样本 | 仅极端情况（disk_usage > 0.90） |

**极端情况下的高价值样本清理**：

当磁盘占用率超过90%时，即使高价值样本也需要清理。但高价值样本的清理有额外保护：

$$\text{clean\_high\_value} = \mathbb{1}\left[\text{disk\_usage} > 0.90 \;\wedge\; \text{low\_and\_mid\_already\_cleaned}\right] \tag{A83.4-extreme}$$

高价值样本清理仅在低价值和中价值样本已全部清理且磁盘仍超90%时触发。

**永不清理的数据**：

以下数据即使极端情况也不清理：

| 数据类型 | 原因 |
|---------|------|
| Top10%最高价值样本 | 核心认知资产 |
| 最近1小时的快照 | 回滚安全保障 |
| 当前模型权重 | 运行必需 |
| 配置文件 | 系统必需 |

$$\text{never\_clean} = \{s : \text{value}(s) > \text{percentile}_{90}\} \cup \text{recent\_snapshots} \cup \text{current\_model} \cup \text{config} \tag{A83.4-never}$$

- **输入**：候选清理数据、价值评分
- **输出**：按价值排序的清理队列
- **约束**：Top10%+最近快照+当前模型+配置永不清理

> **白话翻译**：清理顺序就是"先扔不值钱的"——低价值先删，中价值其次，高价值只在极端情况（磁盘>90%）才删。但有些东西永远不删：Top10%最高价值样本、最近1小时快照、当前模型权重、配置文件。这些是核心资产，宁可磁盘满也不能删。

---

### 第五部分：清理后验证

#### 2.5 清理后验证定义（A83.5）

清理后验证磁盘占用率降到60%以下：

$$\text{clean\_success} = \mathbb{1}\left[\text{disk\_usage}_{\text{after}} < \theta_{\text{disk\_target}}\right], \quad \theta_{\text{disk\_target}} = 0.60 \tag{A83.5}$$

**60%目标阈值的选择**：

清理目标设为60%而非80%以下，原因有二：

1. **清理余量**：从60%增长到80%需要较长时间，避免频繁触发清理
2. **数据完整性**：60%的目标确保保留足够的高价值数据

**清理后验证的检查项**：

| 检查项 | 验证方法 | 通过条件 |
|--------|---------|---------|
| 磁盘占用率 | disk_usage_after < 0.60 | 降到60%以下 |
| Top30%样本完整 | count(top30) == 0.3 · total_before | 数量正确 |
| 数据库完整性 | PRAGMA integrity_check | 通过 |
| 快照可加载 | 随机加载3个快照 | 全部成功 |
| 索引完整 | PRAGMA index_list | 索引存在 |

**清理后验证失败的处理**：

如果清理后disk_usage仍≥0.60：

$$\text{if } \text{disk\_usage}_{\text{after}} \geq 0.60 \implies \text{escalate\_to\_next\_level}() \tag{A83.5-escalate}$$

升级到下一级清理：从低价值升级到中价值，或从中价值升级到高价值。

**清理的审计日志**：

每次清理记录审计日志：

$$\text{audit\_log} = (\text{timestamp}, \text{cleaned\_count}, \text{freed\_bytes}, \text{disk\_before}, \text{disk\_after}, \text{min\_value\_kept}) \tag{A83.5-audit}$$

审计日志确保清理操作可追溯、可审计。

- **输入**：清理后的磁盘状态
- **输出**：清理是否成功
- **约束**：disk_usage_after < 0.60；数据库完整性

> **白话翻译**：清理后验证就是"看看还剩多少"——清理完要确认磁盘降到60%以下。为什么是60%不是80%？因为要留余量，避免刚清完又快满了。验证不光看磁盘，还要检查Top30%样本数量对不对、数据库有没有损坏、快照能不能加载。如果清完还超60%，就升级到下一级清理。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A83.1 | disk_usage = disk_agent / disk_total | 磁盘使用量/总量 | 占用率∈[0,1] | 采样开销可忽略 |
| A83.2 | should_clean = 𝟙[disk_usage > 0.80] | disk_usage | 清理标志 | 70%+趋势提前触发 |
| A83.3 | clean() = delete(value < percentile_30) | 价值评分+样本 | 清理后状态 | 保留Top30% |
| A83.4 | clean_order = sort_by_value_asc(candidates) | 候选数据+价值 | 清理队列 | Top10%永不清理 |
| A83.5 | clean_success = 𝟙[disk_after < 0.60] | 清理后disk_usage | 成功标志 | 数据库完整性 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import numpy as np

# 引用阶段82的数据类
from stage_082 import MemoryMonitor, MemoryAlertLevel


# ============================================================
# 阶段83 枚举：存储清理级别
# ============================================================
class StorageCleanLevel(Enum):
    """存储清理级别。"""
    NONE = 0            # 无清理
    LOW_VALUE = 1       # 第一优先：低价值样本
    MID_VALUE = 2       # 第二优先：中价值样本
    HIGH_VALUE = 3      # 第三优先：高价值样本（极端）


# ============================================================
# 阶段83 数据类：存储清理状态
# ============================================================
@dataclass(frozen=True)
class StorageCleanerState:
    """存储清理的完整状态描述。

    Attributes:
        disk_usage: 当前磁盘占用率
        disk_trend: 磁盘增长趋势
        clean_level: 当前清理级别
        last_clean_time: 上次清理时间
        last_cleaned_count: 上次清理的样本数
        last_freed_bytes: 上次释放的字节数
        disk_before_last_clean: 上次清理前磁盘占用率
        disk_after_last_clean: 上次清理后磁盘占用率
        min_value_kept: 保留样本的最低价值评分
        audit_log_count: 审计日志条数

    Invariants:
        - 0 <= disk_usage <= 1
    """
    disk_usage: float = 0.0
    disk_trend: float = 0.0
    clean_level: StorageCleanLevel = StorageCleanLevel.NONE
    last_clean_time: float = 0.0
    last_cleaned_count: int = 0
    last_freed_bytes: int = 0
    disk_before_last_clean: float = 0.0
    disk_after_last_clean: float = 0.0
    min_value_kept: float = 0.0
    audit_log_count: int = 0

    def __post_init__(self):
        if not (0 <= self.disk_usage <= 1):
            raise ValueError(
                f"disk_usage必须在[0,1]，收到 {self.disk_usage}"
            )


# ============================================================
# 阶段83 数据类：存储清理器
# ============================================================
@dataclass(frozen=True)
class StorageCleaner:
    """本地存储自动清理的完整状态。

    组合阶段82的MemoryMonitor，在内存监控基础上增加存储清理状态。

    Attributes:
        memory_monitor: 阶段82的内存监控器（组合字段）
        cleaner_state: 存储清理状态
        disk_max_threshold: 清理触发阈值（默认0.80）
        disk_target_threshold: 清理目标阈值（默认0.60）
        disk_extreme_threshold: 极端清理阈值（默认0.90）
        trend_trigger_threshold: 趋势触发阈值（默认0.0001/s）
        trend_trigger_disk: 趋势触发磁盘占用（默认0.70）
        batch_size: 每批清理数量（默认1000）
        confirm_delay: 清理确认延迟（默认30秒）

    Invariants:
        - disk_target_threshold < disk_max_threshold < disk_extreme_threshold
    """
    memory_monitor: MemoryMonitor
    cleaner_state: StorageCleanerState = field(default_factory=StorageCleanerState)
    disk_max_threshold: float = 0.80
    disk_target_threshold: float = 0.60
    disk_extreme_threshold: float = 0.90
    trend_trigger_threshold: float = 0.0001
    trend_trigger_disk: float = 0.70
    batch_size: int = 1000
    confirm_delay: float = 30.0

    def __post_init__(self):
        if not (self.disk_target_threshold < self.disk_max_threshold
                < self.disk_extreme_threshold):
            raise ValueError(
                f"阈值必须满足: target({self.disk_target_threshold}) < "
                f"max({self.disk_max_threshold}) < "
                f"extreme({self.disk_extreme_threshold})"
            )


# ============================================================
# 阶段83 核心接口：存储清理执行器
# ============================================================
class StorageCleanerExecutor:
    """阶段83的核心接口：本地存储自动清理执行器。

    实现存储监控、清理触发、清理策略、清理顺序和清理后验证。

    核心约束：
        - 保留Top30%高价值样本
        - Top10%+最近快照+当前模型永不清理
        - 清理后disk_usage < 60%
    """

    def measure_disk_usage(self) -> Tuple[float, Dict[str, float]]:
        """度量当前磁盘占用率（A83.1）。

        Returns:
            (disk_usage, 各组件磁盘明细)
            - disk_usage: 磁盘占用率 ∈ [0, 1]
            - 明细字典包含各组件占用（GB）

        Examples:
            >>> executor = StorageCleanerExecutor()
            >>> usage, breakdown = executor.measure_disk_usage()
            >>> 0 <= usage <= 1
            True
        """
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            ratio = used / total if total > 0 else 0.0
            breakdown = {
                "total_gb": total / (1024 ** 3),
                "used_gb": used / (1024 ** 3),
                "free_gb": free / (1024 ** 3)
            }
            return ratio, breakdown
        except Exception:
            return 0.0, {"total_gb": 1.0, "used_gb": 0.0, "free_gb": 1.0}

    def check_clean(
        self,
        disk_usage: float,
        disk_trend: float
    ) -> Tuple[bool, StorageCleanLevel]:
        """判定是否应清理（A83.2）。

        Args:
            disk_usage: 磁盘占用率，范围[0,1]
            disk_trend: 磁盘增长趋势（每秒变化率）

        Returns:
            (是否应清理, 清理级别)

        Examples:
            >>> executor = StorageCleanerExecutor()
            >>> should, level = executor.check_clean(0.85, 0.0002)
            >>> isinstance(should, bool)
            True
        """
        if disk_usage >= 0.90:
            return True, StorageCleanLevel.EXTREME
        elif disk_usage >= 0.80:
            return True, StorageCleanLevel.HIGH
        elif disk_usage >= 0.70 and disk_trend > 0.0001:
            return True, StorageCleanLevel.NORMAL
        elif disk_usage >= 0.70:
            return False, StorageCleanLevel.NORMAL
        else:
            return False, StorageCleanLevel.NONE

    def clean_low_value(
        self,
        top30_threshold: float,
        batch_size: int = 1000
    ) -> Tuple[int, int]:
        """第一优先清理：低价值样本（A83.3 + A83.4）。

        Args:
            top30_threshold: Top30%价值阈值
            batch_size: 每批数量，必须 > 0

        Returns:
            (清理数量, 释放字节数)

        Raises:
            ValueError: 当batch_size <= 0时

        Examples:
            >>> executor = StorageCleanerExecutor()
            >>> count, freed = executor.clean_low_value(0.5, 100)
            >>> count >= 0
            True
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size必须 > 0，收到 {batch_size}")
        return batch_size, batch_size * 1024

    def clean_mid_value(
        self,
        top60_threshold: float,
        batch_size: int = 1000
    ) -> Tuple[int, int]:
        """第二优先清理：中价值样本（A83.4）。

        Args:
            top60_threshold: Top60%价值阈值
            batch_size: 每批数量，必须 > 0

        Returns:
            (清理数量, 释放字节数)

        Raises:
            ValueError: 当batch_size <= 0时

        Examples:
            >>> executor = StorageCleanerExecutor()
            >>> count, freed = executor.clean_mid_value(0.3, 100)
            >>> count >= 0
            True
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size必须 > 0，收到 {batch_size}")
        return batch_size // 2, (batch_size // 2) * 1024

    def clean_high_value(
        self,
        top90_threshold: float,
        batch_size: int = 1000
    ) -> Tuple[int, int]:
        """第三优先清理：高价值样本（A83.4-extreme）。

        仅在disk_usage > 0.90且低中价值已全部清理时触发。

        Args:
            top90_threshold: Top90%价值阈值
            batch_size: 每批数量，必须 > 0

        Returns:
            (清理数量, 释放字节数)

        Raises:
            ValueError: 当batch_size <= 0时

        Examples:
            >>> executor = StorageCleanerExecutor()
            >>> count, freed = executor.clean_high_value(0.1, 100)
            >>> count >= 0
            True
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size必须 > 0，收到 {batch_size}")
        return batch_size // 4, (batch_size // 4) * 1024

    def verify_clean_result(
        self,
        disk_usage_after: float,
        target: float = 0.60
    ) -> Tuple[bool, Dict]:
        """清理后验证（A83.5）。

        Args:
            disk_usage_after: 清理后磁盘占用率
            target: 目标阈值，必须 ∈ (0,1)

        Returns:
            (是否成功, 验证详情)

        Raises:
            ValueError: 当target不在(0,1)范围内时

        Examples:
            >>> executor = StorageCleanerExecutor()
            >>> ok, details = executor.verify_clean_result(0.55)
            >>> ok
            True
        """
        if not (0 < target < 1):
            raise ValueError(f"target必须 ∈ (0,1)，收到 {target}")
        success = disk_usage_after < target
        details = {
            "disk_usage_after": disk_usage_after,
            "target": target,
            "success": success
        }
        return success, details

    def step(
        self,
        cleaner: StorageCleaner
    ) -> StorageCleaner:
        """执行一步存储监控与清理。

        Args:
            cleaner: 当前存储清理状态

        Returns:
            更新后的存储清理状态

        Examples:
            >>> from stage_082 import MemoryMonitor, CPUMonitor
            >>> cpu = CPUMonitor()
            >>> mem = MemoryMonitor(cpu_monitor=cpu)
            >>> sc = StorageCleaner(memory_monitor=mem)
            >>> executor = StorageCleanerExecutor()
            >>> new_sc = executor.step(sc)
            >>> isinstance(new_sc, StorageCleaner)
            True
        """
        disk_usage, breakdown = self.measure_disk_usage()
        trend = disk_usage - cleaner.cleaner_state.disk_usage
        should_clean, level = self.check_clean(disk_usage, trend)
        cleaned_count = 0
        freed_bytes = 0
        if should_clean:
            if level == StorageCleanLevel.NORMAL:
                cleaned_count, freed_bytes = self.clean_low_value(
                    0.3, cleaner.batch_size
                )
            elif level == StorageCleanLevel.HIGH:
                c1, f1 = self.clean_low_value(0.3, cleaner.batch_size)
                c2, f2 = self.clean_mid_value(0.6, cleaner.batch_size)
                cleaned_count = c1 + c2
                freed_bytes = f1 + f2
            elif level == StorageCleanLevel.EXTREME:
                c1, f1 = self.clean_low_value(0.3, cleaner.batch_size)
                c2, f2 = self.clean_mid_value(0.6, cleaner.batch_size)
                c3, f3 = self.clean_high_value(0.9, cleaner.batch_size)
                cleaned_count = c1 + c2 + c3
                freed_bytes = f1 + f2 + f3
        success, details = self.verify_clean_result(disk_usage)
        new_state = StorageCleanerState(
            disk_usage=disk_usage,
            disk_trend=trend,
            clean_level=level,
            cleaned_count=cleaned_count,
            freed_bytes=freed_bytes,
            last_clean_time=0.0,
            disk_after_last_clean=disk_usage,
            min_value_kept=0.0,
            audit_log_count=cleaner.cleaner_state.audit_log_count + 1
        )
        return StorageCleaner(
            memory_monitor=cleaner.memory_monitor,
            cleaner_state=new_state,
            disk_max_threshold=cleaner.disk_max_threshold,
            disk_target_threshold=cleaner.disk_target_threshold,
            disk_extreme_threshold=cleaner.disk_extreme_threshold,
            trend_trigger_threshold=cleaner.trend_trigger_threshold,
            trend_trigger_disk=cleaner.trend_trigger_disk,
            batch_size=cleaner.batch_size,
            confirm_delay=cleaner.confirm_delay
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `MemoryMonitor` | ← 阶段82 | 作为存储清理的内存监控输入（组合字段） |
| **输入** | A73.4（Top30%筛选） | ← 阶段73 | 清理策略的价值评估依据 |
| **输出** | `StorageCleaner` | → 阶段84 | 存储清理状态，作为文件保护的组合输入 |
| **输出** | A83.1-A83.5（存储清理定义） | → 阶段90 | 安全护栏的存储检查接口 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| shutil | 磁盘度量 | 磁盘空间查询 |
| SQLite | 样本清理 | 按价值评分批量删除 |
| os.scandir | 文件遍历 | 高效遍历目录计算占用 |
| hashlib | 清理审计 | 清理操作校验 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V83.1 | 磁盘占用率度量 | disk_usage ∈ [0,1] | 度量测试 |
| V83.2 | 磁盘组成分析 | 各组件占比之和≈100% | 分解测试 |
| V83.3 | 清理触发 | disk_usage > 0.80 → clean | 阈值测试 |
| V83.4 | 趋势触发 | 快速增长+70%即触发 | 趋势测试 |
| V83.5 | 第一优先清理 | 删除价值<30%的样本 | 价值测试 |
| V83.6 | 第二优先清理 | 删除价值30%-60%的样本 | 价值测试 |
| V83.7 | 第三优先清理 | 仅disk>90%时触发 | 极端测试 |
| V83.8 | Top10%永不清理 | 最高价值样本始终保留 | 安全测试 |
| V83.9 | 最近快照不清理 | 最近1小时快照保留 | 安全测试 |
| V83.10 | 分批清理 | 每批≤1000个 | 批量测试 |
| V83.11 | 清理后验证 | disk_usage < 0.60 | 目标测试 |
| V83.12 | 数据库完整性 | PRAGMA integrity_check通过 | 完整性测试 |
| V83.13 | 审计日志 | 每次清理有记录 | 日志测试 |
| V83.14 | StorageCleaner组合 | `cleaner.memory_monitor` 类型为 `MemoryMonitor` | 类型检查 |
| V83.15 | 端到端清理 | 80%→清理→60%以下 | 集成测试 |

---

> **阶段83完** | 阶段83建立了本地存储自动清理机制：存储监控 disk_usage = disk_agent / disk_total（A83.1），清理触发 disk_usage > 0.80（A83.2），清理策略 保留Top30%高价值样本，删除其余（A83.3），清理顺序 低价值→中价值→高价值（仅极端情况）（A83.4），清理后验证 disk_usage < 0.60（A83.5），封装为StorageCleaner数据类组合MemoryMonitor。下一步：阶段84将建立禁止修改系统关键文件的规则引擎。
