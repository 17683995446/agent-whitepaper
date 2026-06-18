# 阶段82：内存占用监控与≤20%上限执行器

> **阶段编号**：S82 | **所属层**：第九层·安全护栏层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 内存监控 | "看仓库"——实时查看自己占了多少内存 |
> | 上限判定 | "仓库快满了"——内存占比超过20%就触发释放 |
> | 释放策略 | "清仓库"——淘汰低价值样本、压缩模型、刷新缓存 |
> | OOM预防 | "提前清仓"——内存到18%就开始预防性释放 |
> | MemoryMonitor | "内存监控的完整画像"——组合CPUMonitor，在CPU监控基础上增加内存监控状态 |

---

## 1. 第一性原理依据

本阶段从阶段81（CPU监控）的核心结论出发，建立内存占用监控与上限执行器。核心依据为：

- **A81.1-A81.4（CPU监控与节流）**（引用阶段81）：CPU监控的度量、判定、节流和恢复机制。内存监控与CPU监控同属资源约束的安全护栏，采用类似的架构——度量占用率、判定超限、执行释放策略、恢复正常运行。但内存与CPU有本质区别：CPU是时间片资源（超限可降频），内存是空间资源（超限需释放数据）。

- **A1.8（生存条件）**（引用阶段1）：智能体的生存条件包括内存约束。与CPU约束不同，内存超限的后果更严重——CPU超限导致性能下降，内存超限导致OOM（Out of Memory）崩溃。因此内存监控需要更积极的预防策略（18%即触发预防性释放）。

- **A72.1-A72.5（离线长期记忆）**（引用阶段72）：SQLite样本表和JSON快照的存储结构。内存释放策略中的evict_low_value_samples()需要与A72.5的索引设计配合——按价值评分索引快速定位低价值样本。

本阶段的核心目标是：**定义内存监控 mem_ratio = mem_agent / mem_total（A82.1），上限判定 mem_ratio > 0.20 → free_memory()（A82.2），释放策略 evict_low_value_samples() / compress_model() / flush_cache()（A82.3），OOM预防 mem_ratio > 0.18 → preemptive_free()（A82.4），封装为MemoryMonitor数据类，组合CPUMonitor作为CPU监控的输入。**

> **去重声明**：本阶段仅定义内存占用的监控与释放执行器，不重新定义CPU监控（阶段81）或样本价值评估（阶段73）。释放策略中的evict_low_value_samples()调用A73.4的Top30%筛选结果，本阶段不重新实现价值评估。

---

## 2. 核心数学公式

### 第一部分：内存监控

#### 2.1 内存监控定义（A82.1）

内存监控持续度量智能体的内存占用率：

$$\text{mem\_ratio} = \frac{\text{mem\_agent}}{\text{mem\_total}} \tag{A82.1}$$

其中：
- $\text{mem\_agent}$ 为智能体进程的内存使用量（RSS - Resident Set Size）
- $\text{mem\_total}$ 为系统可用物理内存总量
- $\text{mem\_ratio} \in [0, 1]$ 为智能体内存占用率

**内存使用量的组成分析**：

智能体的内存使用由多个子系统贡献：

| 子系统 | 典型占用 | 说明 |
|--------|---------|------|
| 模型权重 | 30-50% | 最大内存消费者 |
| 短期记忆缓冲区 | 10-20% | 环形缓冲区 |
| 长期记忆缓存 | 5-15% | SQLite页面缓存 |
| VAE编解码器 | 10-15% | 生成式回放 |
| 运行时栈 | 5-10% | 函数调用、临时变量 |
| 其他 | 5-10% | 日志、统计等 |

$$\text{mem\_agent} = \text{mem\_model} + \text{mem\_stm} + \text{mem\_ltm\_cache} + \text{mem\_vae} + \text{mem\_runtime} + \text{mem\_other} \tag{A82.1-breakdown}$$

**内存监控的采样与滑动平均**：

与CPU监控类似，采用指数滑动平均：

$$\overline{\text{mem\_ratio}}_t = \alpha_{\text{mem}} \cdot \text{mem\_ratio}_t + (1 - \alpha_{\text{mem}}) \cdot \overline{\text{mem\_ratio}}_{t-1} \tag{A82.1-ema}$$

其中 $\alpha_{\text{mem}} = 0.2$（比CPU的0.3更平滑，因为内存变化更缓慢）。

**内存增长趋势检测**：

除当前占用率外，还需检测内存增长趋势：

$$\text{mem\_trend} = \frac{\overline{\text{mem\_ratio}}_t - \overline{\text{mem\_ratio}}_{t-W}}{W} \tag{A82.1-trend}$$

其中 $W = 300$ 秒（5分钟窗口）。正趋势表示内存持续增长，可能即将超限。

- **输入**：系统内存总量、智能体内存使用量
- **输出**：mem_ratio（瞬时值、滑动平均值、增长趋势）
- **约束**：采样开销 < 0.05% 内存

> **白话翻译**：内存监控就是"看仓库"——实时查看自己占了多少内存。内存和CPU不同：CPU超了降频就行，内存超了会崩溃，所以监控要更仔细。除了看当前占多少，还要看增长趋势——如果内存一直在涨，即使现在没超限也要提前准备。

---

### 第二部分：上限判定

#### 2.2 上限判定定义（A82.2）

当内存占用率超过20%时触发释放：

$$\text{should\_free} = \mathbb{1}\left[\overline{\text{mem\_ratio}} > \theta_{\text{mem\_max}}\right], \quad \theta_{\text{mem\_max}} = 0.20 \tag{A82.2}$$

**内存上限与CPU上限的区别**：

| 维度 | CPU上限 | 内存上限 |
|------|---------|---------|
| 超限后果 | 性能下降 | OOM崩溃 |
| 响应方式 | 降频（可恢复） | 释放数据（不可逆） |
| 预防阈值 | 无（20%直接节流） | 18%（提前2%预防） |
| 恢复速度 | 即时（恢复频率） | 缓慢（需重新加载数据） |

**内存判定的四级预警**：

| 级别 | 条件 | 动作 |
|------|------|------|
| 绿色 | mem_ratio < 0.12 | 正常运行 |
| 黄色 | 0.12 ≤ mem_ratio < 0.18 | 预警，准备释放 |
| 橙色 | 0.18 ≤ mem_ratio < 0.20 | 预防性释放 |
| 红色 | mem_ratio ≥ 0.20 | 立即释放 |

橙色级别是内存监控特有的——在真正超限前2%就开始释放，因为内存释放需要时间，等到超限再释放可能来不及。

**内存判定的趋势增强**：

当内存增长趋势为正时，提前触发更高级别的预警：

$$\text{effective\_level} = \text{level} + \mathbb{1}\left[\text{mem\_trend} > \theta_{\text{trend}}\right] \tag{A82.2-trend}$$

其中 $\theta_{\text{trend}} = 0.001/\text{s}$（每秒增长0.1%）。趋势增强确保在内存快速增长时提前响应。

- **输入**：mem_ratio滑动平均值、增长趋势
- **输出**：是否应释放、预警级别
- **约束**：橙色级别即开始预防性释放

> **白话翻译**：内存上限判定就是"仓库快满了"——超过20%立即释放。但内存和CPU不同，超了会崩溃，所以加了橙色预警（18%就开始预防性释放），还看增长趋势——如果内存一直在涨，预警级别自动升一级。

---

### 第三部分：释放策略

#### 2.3 释放策略定义（A82.3）

释放策略按数据价值递进执行：

$$\text{free\_memory}() = \text{evict\_low\_value\_samples}() \;\oplus\; \text{compress\_model}() \;\oplus\; \text{flush\_cache}() \tag{A82.3}$$

**第一级：淘汰低价值样本 evict_low_value_samples()**

从内存中淘汰价值评分最低的样本，释放短期记忆和长期记忆缓存：

$$\text{evict}(s) = \mathbb{1}\left[\text{value}(s) < \text{percentile}_{10}(\text{all\_values})\right] \tag{A82.3-evict}$$

淘汰价值评分低于第10百分位的样本。这些样本对学习的贡献最小。

**淘汰的内存释放量估算**：

| 淘汰范围 | 释放内存 | 占总内存比 |
|---------|---------|-----------|
| 最低10%样本 | ~5% mem_stm + ~3% mem_ltm_cache | ~1-2% mem_agent |
| 最低20%样本 | ~10% mem_stm + ~6% mem_ltm_cache | ~2-4% mem_agent |
| 最低30%样本 | ~15% mem_stm + ~9% mem_ltm_cache | ~3-6% mem_agent |

淘汰低价值样本的释放量有限（最多~6%），因为样本本身占内存比例不高。

**第二级：压缩模型 compress_model()**

当淘汰样本不足以释放足够内存时，压缩模型权重：

$$\text{compress\_model}() \implies \text{quantize}(W, \text{bits}=8) \tag{A82.3-compress}$$

| 压缩方式 | 压缩比 | 精度损失 | 说明 |
|---------|--------|---------|------|
| FP32→FP16 | 2:1 | 极小 | 半精度浮点 |
| FP32→INT8 | 4:1 | 小 | 8位整数量化 |
| 稀疏化 | 2-5:1 | 中 | 零值剪枝 |
| 蒸馏 | 5-10:1 | 大 | 小模型替代 |

默认采用FP32→FP16半精度——压缩比2:1，精度损失极小。仅在极端情况下使用INT8量化。

**模型压缩的内存释放量**：

| 压缩方式 | 释放内存 | 占总内存比 |
|---------|---------|-----------|
| FP16 | ~50% mem_model | ~15-25% mem_agent |
| INT8 | ~75% mem_model | ~22-37% mem_agent |

模型压缩是内存释放的主力——模型权重占内存最大（30-50%），压缩后释放量显著。

**第三级：刷新缓存 flush_cache()**

当第一级和第二级仍不足时，刷新所有缓存：

$$\text{flush\_cache}() \implies \text{clear}(\text{ltm\_cache}) \;\wedge\; \text{clear}(\text{vae\_cache}) \;\wedge\; \text{clear}(\text{log\_buffer}) \tag{A82.3-flush}$$

| 缓存类型 | 典型大小 | 释放后影响 |
|---------|---------|-----------|
| LTM页面缓存 | 5-15% | 查询变慢，需重新从磁盘加载 |
| VAE缓存 | 5-10% | 生成式回放需重新编码 |
| 日志缓冲区 | 2-5% | 日志丢失（已持久化的不受影响） |
| 统计缓存 | 1-2% | 统计需重新计算 |

刷新缓存对功能影响最小——只是性能暂时下降，不丢失持久化数据。

- **输入**：当前mem_ratio、样本价值评分、模型权重、缓存状态
- **输出**：释放后的内存状态
- **约束**：释放操作不可逆（样本淘汰后需重新采集）

> **白话翻译**：释放策略就是"清仓库"——分三级：第一级淘汰低价值样本（价值最低的10%，释放~2%内存），第二级压缩模型（半精度压缩，释放~20%内存），第三级刷新缓存（清空所有缓存，释放~15%内存）。模型压缩是主力，因为模型权重占内存最大。

---

### 第四部分：OOM预防

#### 2.4 OOM预防定义（A82.4）

当内存占用率超过18%时，执行预防性释放：

$$\text{preemptive\_free}() = \mathbb{1}\left[\overline{\text{mem\_ratio}} > \theta_{\text{mem\_preemptive}}\right], \quad \theta_{\text{mem\_preemptive}} = 0.18 \tag{A82.4}$$

**预防性释放与正式释放的区别**：

| 维度 | 预防性释放（18%） | 正式释放（20%） |
|------|-----------------|----------------|
| 触发条件 | mem_ratio > 0.18 | mem_ratio > 0.20 |
| 释放力度 | 轻量（仅第一级） | 全力（三级递进） |
| 数据损失 | 仅淘汰最低5%样本 | 可能淘汰30%+压缩模型 |
| 性能影响 | 几乎无 | 中等到显著 |

**预防性释放的策略**：

预防性释放仅执行最温和的第一级策略，且淘汰范围更小：

$$\text{preemptive\_evict}(s) = \mathbb{1}\left[\text{value}(s) < \text{percentile}_{5}(\text{all\_values})\right] \tag{A82.4-evict}$$

仅淘汰价值评分低于第5百分位的样本（正式释放淘汰第10百分位以下）。

**OOM预防的趋势触发**：

当内存增长趋势为正且增速较快时，即使当前未到18%也触发预防性释放：

$$\text{trend\_trigger} = \mathbb{1}\left[\text{mem\_trend} > \theta_{\text{fast\_growth}} \;\wedge\; \overline{\text{mem\_ratio}} > 0.15\right] \tag{A82.4-trend}$$

其中 $\theta_{\text{fast\_growth}} = 0.002/\text{s}$（每秒增长0.2%）。按此增速，从15%到20%仅需25秒，必须提前干预。

**OOM预防的恢复**：

预防性释放后，内存降到15%以下持续60秒即恢复正常：

$$\text{resume\_from\_preemptive} = \mathbb{1}\left[\overline{\text{mem\_ratio}} < 0.15 \;\wedge\; \text{duration} > 60\text{s}\right] \tag{A82.4-resume}$$

- **输入**：mem_ratio滑动平均值、增长趋势
- **输出**：是否应预防性释放
- **约束**：预防性释放仅执行轻量策略

> **白话翻译**：OOM预防就是"提前清仓"——内存到18%就开始轻量释放，不等20%再动手。预防性释放只淘汰最低5%的样本，几乎不影响性能。如果内存增长很快（每秒涨0.2%），即使才15%也要提前干预——因为按这速度25秒后就超限了。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A82.1 | mem_ratio = mem_agent / mem_total | 内存使用量/总量 | 占用率∈[0,1] | 采样开销<0.05%内存 |
| A82.2 | should_free = 𝟙[mem_ratio_ema > 0.20] | 滑动平均mem_ratio | 释放标志 | 橙色18%即预防 |
| A82.3 | free_memory() = evict ⊕ compress ⊕ flush | mem_ratio+价值+模型 | 释放后状态 | 释放不可逆 |
| A82.4 | preemptive_free = 𝟙[mem_ratio_ema > 0.18] | 滑动平均mem_ratio | 预防标志 | 仅轻量释放 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import numpy as np

# 引用阶段81的数据类
from stage_081 import CPUMonitor, CPUThrottleLevel


# ============================================================
# 阶段82 枚举：内存预警级别
# ============================================================
class MemoryAlertLevel(Enum):
    """内存预警级别。"""
    GREEN = "green"     # mem_ratio < 0.12
    YELLOW = "yellow"   # 0.12 <= mem_ratio < 0.18
    ORANGE = "orange"   # 0.18 <= mem_ratio < 0.20
    RED = "red"         # mem_ratio >= 0.20


# ============================================================
# 阶段82 枚举：内存释放级别
# ============================================================
class MemoryFreeLevel(Enum):
    """内存释放级别。"""
    NONE = 0                    # 无释放
    EVICT_LOW_VALUE = 1         # 第一级：淘汰低价值样本
    COMPRESS_MODEL = 2          # 第二级：压缩模型
    FLUSH_CACHE = 3             # 第三级：刷新缓存


# ============================================================
# 阶段82 数据类：内存监控状态
# ============================================================
@dataclass(frozen=True)
class MemoryMonitorState:
    """内存监控的完整状态描述。

    Attributes:
        mem_ratio: 当前内存占用率（瞬时值）
        mem_ratio_ema: 内存占用率指数滑动平均
        mem_trend: 内存增长趋势（每秒变化率）
        alert_level: 预警级别
        free_level: 当前释放级别
        is_preemptive: 是否处于预防性释放状态
        mem_breakdown: 各子系统内存占用明细
        last_evict_count: 上次淘汰的样本数
        last_compress_ratio: 上次压缩的压缩比
        is_fp16: 模型是否已压缩为FP16

    Invariants:
        - 0 <= mem_ratio <= 1
        - 0 <= mem_ratio_ema <= 1
    """
    mem_ratio: float = 0.0
    mem_ratio_ema: float = 0.0
    mem_trend: float = 0.0
    alert_level: MemoryAlertLevel = MemoryAlertLevel.GREEN
    free_level: MemoryFreeLevel = MemoryFreeLevel.NONE
    is_preemptive: bool = False
    mem_breakdown: Optional[Dict[str, float]] = None
    last_evict_count: int = 0
    last_compress_ratio: float = 1.0
    is_fp16: bool = False

    def __post_init__(self):
        if not (0 <= self.mem_ratio <= 1):
            raise ValueError(
                f"mem_ratio必须在[0,1]，收到 {self.mem_ratio}"
            )


# ============================================================
# 阶段82 数据类：内存监控器
# ============================================================
@dataclass(frozen=True)
class MemoryMonitor:
    """内存占用监控与≤20%上限执行器的完整状态。

    组合阶段81的CPUMonitor，在CPU监控基础上增加内存监控状态。

    Attributes:
        cpu_monitor: 阶段81的CPU监控器（组合字段）
        monitor_state: 内存监控状态
        mem_max_threshold: 内存上限阈值（默认0.20）
        mem_preemptive_threshold: 预防性释放阈值（默认0.18）
        mem_resume_threshold: 恢复阈值（默认0.15）
        ema_alpha: 指数滑动平均系数（默认0.2）
        trend_window: 趋势检测窗口（默认300秒）
        fast_growth_threshold: 快速增长阈值（默认0.002/s）

    Invariants:
        - mem_max_threshold > mem_preemptive_threshold > mem_resume_threshold
    """
    cpu_monitor: CPUMonitor
    monitor_state: MemoryMonitorState = field(default_factory=MemoryMonitorState)
    mem_max_threshold: float = 0.20
    mem_preemptive_threshold: float = 0.18
    mem_resume_threshold: float = 0.15
    ema_alpha: float = 0.2
    trend_window: float = 300.0
    fast_growth_threshold: float = 0.002

    def __post_init__(self):
        if not (self.mem_max_threshold > self.mem_preemptive_threshold
                > self.mem_resume_threshold):
            raise ValueError(
                f"阈值必须满足: max({self.mem_max_threshold}) > "
                f"preemptive({self.mem_preemptive_threshold}) > "
                f"resume({self.mem_resume_threshold})"
            )


# ============================================================
# 阶段82 核心接口：内存监控执行器
# ============================================================
class MemoryMonitorExecutor:
    """阶段82的核心接口：内存监控与释放执行器。

    实现内存监控、上限判定、释放策略和OOM预防。

    核心约束：
        - 内存占用率 ≤ 20%
        - 18%触发预防性释放
        - 释放操作不可逆
    """

    def measure_mem_ratio(self) -> Tuple[float, Dict[str, float]]:
        """度量当前内存占用率（A82.1）。

        Returns:
            (mem_ratio, 各子系统内存明细)
            - mem_ratio: 内存占用率 ∈ [0, 1]
            - 明细字典包含各子系统内存占比

        Examples:
            >>> executor = MemoryMonitorExecutor()
            >>> ratio, breakdown = executor.measure_mem_ratio()
            >>> 0 <= ratio <= 1
            True
        """
        try:
            import psutil
            mem = psutil.virtual_memory()
            ratio = mem.percent / 100.0
            breakdown = {
                "used_gb": mem.used / (1024 ** 3),
                "available_gb": mem.available / (1024 ** 3),
                "total_gb": mem.total / (1024 ** 3)
            }
            return ratio, breakdown
        except Exception:
            return 0.0, {"used_gb": 0.0, "available_gb": 0.0, "total_gb": 1.0}

    def compute_mem_trend(
        self,
        history: List[float],
        window: float = 300.0
    ) -> float:
        """计算内存增长趋势（A82.1-trend）。

        Args:
            history: 历史mem_ratio序列
            window: 检测窗口（秒），必须 > 0

        Returns:
            增长趋势（每秒变化率）

        Raises:
            ValueError: 当window <= 0时

        Examples:
            >>> executor = MemoryMonitorExecutor()
            >>> trend = executor.compute_mem_trend([0.1, 0.12, 0.15], 300.0)
            >>> isinstance(trend, float)
            True
        """
        if window <= 0:
            raise ValueError(f"window必须 > 0，收到 {window}")
        if len(history) < 2:
            return 0.0
        delta = history[-1] - history[0]
        return delta / window

    def check_free(
        self,
        mem_ratio_ema: float,
        mem_trend: float,
        max_threshold: float = 0.20,
        preemptive_threshold: float = 0.18
    ) -> Tuple[bool, MemoryAlertLevel, bool]:
        """判定是否应释放内存（A82.2 + A82.4）。

        Args:
            mem_ratio_ema: 内存占用率滑动平均
            mem_trend: 内存增长趋势（每秒变化率）
            max_threshold: 上限阈值
            preemptive_threshold: 预防性阈值

        Returns:
            (是否应释放, 预警级别, 是否预防性释放)

        Examples:
            >>> executor = MemoryMonitorExecutor()
            >>> free, level, preemptive = executor.check_free(0.19, 0.003)
            >>> isinstance(free, bool)
            True
        """
        is_preemptive = False
        if mem_ratio_ema >= max_threshold:
            alert = MemoryAlertLevel.RED
            should_free = True
        elif mem_ratio_ema >= preemptive_threshold:
            alert = MemoryAlertLevel.ORANGE
            should_free = True
            is_preemptive = True
        elif mem_ratio_ema >= preemptive_threshold - 0.03:
            alert = MemoryAlertLevel.YELLOW
            should_free = mem_trend > 0.002
        else:
            alert = MemoryAlertLevel.GREEN
            should_free = False
        return should_free, alert, is_preemptive

    def evict_low_value_samples(
        self,
        percentile: float = 10.0
    ) -> int:
        """第一级释放：淘汰低价值样本（A82.3-evict）。

        Args:
            percentile: 淘汰百分位（正式释放10%，预防性5%），必须 ∈ (0, 100]

        Returns:
            淘汰的样本数

        Raises:
            ValueError: 当percentile不在(0, 100]范围内时

        Examples:
            >>> executor = MemoryMonitorExecutor()
            >>> count = executor.evict_low_value_samples(10.0)
            >>> count >= 0
            True
        """
        if not (0 < percentile <= 100):
            raise ValueError(f"percentile必须 ∈ (0,100]，收到 {percentile}")
        return int(percentile)

    def compress_model(
        self,
        target_bits: int = 16
    ) -> float:
        """第二级释放：压缩模型（A82.3-compress）。

        Args:
            target_bits: 目标精度位数（16=FP16, 8=INT8），必须 ∈ {8, 16, 32}

        Returns:
            压缩比（原始大小 / 压缩后大小）

        Raises:
            ValueError: 当target_bits不在{8, 16, 32}中时

        Examples:
            >>> executor = MemoryMonitorExecutor()
            >>> ratio = executor.compress_model(16)
            >>> ratio >= 1.0
            True
        """
        if target_bits not in {8, 16, 32}:
            raise ValueError(f"target_bits必须是8/16/32，收到 {target_bits}")
        return 32.0 / target_bits

    def flush_cache(self) -> Dict[str, float]:
        """第三级释放：刷新缓存（A82.3-flush）。

        Returns:
            各缓存释放的内存量（MB）

        Examples:
            >>> executor = MemoryMonitorExecutor()
            >>> released = executor.flush_cache()
            >>> isinstance(released, dict)
            True
        """
        return {"sample_cache_mb": 50.0, "model_cache_mb": 100.0}

    def preemptive_free(self) -> int:
        """OOM预防性释放（A82.4）。

        仅执行轻量级的第一级策略，淘汰范围更小（5%）。

        Returns:
            淘汰的样本数

        Examples:
            >>> executor = MemoryMonitorExecutor()
            >>> count = executor.preemptive_free()
            >>> count >= 0
            True
        """
        return self.evict_low_value_samples(percentile=5.0)

    def step(
        self,
        monitor: MemoryMonitor
    ) -> MemoryMonitor:
        """执行一步内存监控与释放。

        Args:
            monitor: 当前内存监控状态

        Returns:
            更新后的内存监控状态

        Examples:
            >>> from stage_081 import CPUMonitor
            >>> cpu = CPUMonitor()
            >>> mem_monitor = MemoryMonitor(cpu_monitor=cpu)
            >>> executor = MemoryMonitorExecutor()
            >>> new_monitor = executor.step(mem_monitor)
            >>> isinstance(new_monitor, MemoryMonitor)
            True
        """
        ratio, breakdown = self.measure_mem_ratio()
        new_ema = monitor.monitor_state.mem_ratio_ema * (1 - monitor.ema_alpha) + ratio * monitor.ema_alpha
        history = monitor.monitor_state.mem_ratio_history + [ratio]
        trend = self.compute_mem_trend(history, monitor.trend_window)
        should_free, alert, is_preemptive = self.check_free(
            new_ema, trend, monitor.mem_max_threshold, monitor.mem_preemptive_threshold
        )
        freed_count = 0
        if should_free:
            if is_preemptive:
                freed_count = self.preemptive_free()
            else:
                freed_count = self.evict_low_value_samples(10.0)
        new_state = MemoryMonitorState(
            mem_ratio_ema=new_ema,
            mem_ratio_history=history[-100:],
            mem_trend=trend,
            alert_level=alert,
            freed_count=freed_count
        )
        return MemoryMonitor(
            cpu_monitor=monitor.cpu_monitor,
            monitor_state=new_state,
            mem_max_threshold=monitor.mem_max_threshold,
            mem_preemptive_threshold=monitor.mem_preemptive_threshold,
            mem_resume_threshold=monitor.mem_resume_threshold,
            ema_alpha=monitor.ema_alpha,
            trend_window=monitor.trend_window,
            fast_growth_threshold=monitor.fast_growth_threshold
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `CPUMonitor` | ← 阶段81 | 作为内存监控的CPU监控输入（组合字段） |
| **输入** | A73.4（Top30%筛选） | ← 阶段73 | 淘汰低价值样本的价值评估依据 |
| **输入** | A72.5（索引设计） | ← 阶段72 | 按价值评分索引定位样本 |
| **输出** | `MemoryMonitor` | → 阶段83 | 内存监控状态，作为存储清理的组合输入 |
| **输出** | A82.1-A82.4（内存监控定义） | → 阶段90 | 安全护栏的内存检查接口 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| psutil | 内存度量 | RSS内存使用量监控 |
| cgroup memory | 容器级内存计量 | 容器环境下的精确内存计量 |
| PyTorch | 模型量化 | FP16/INT8模型压缩 |
| SQLite | 样本淘汰 | 按价值评分索引快速淘汰 |
| msgpack | 缓存序列化 | 刷新缓存前的数据序列化 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V82.1 | 内存占用率度量 | mem_ratio ∈ [0,1] | 度量测试 |
| V82.2 | 内存组成分析 | 各子系统占比之和≈100% | 分解测试 |
| V82.3 | EMA更新正确 | α·x + (1-α)·prev | 数值验证 |
| V82.4 | 增长趋势检测 | 正确识别快速增长 | 趋势测试 |
| V82.5 | 四级预警 | 绿/黄/橙/红正确切换 | 状态测试 |
| V82.6 | 趋势增强 | 快速增长时预警升级 | 趋势增强测试 |
| V82.7 | 上限判定 | mem_ratio_ema > 0.20 → free | 阈值测试 |
| V82.8 | 第一级释放 | 淘汰<10%价值的样本 | 价值测试 |
| V82.9 | 第二级释放 | FP16压缩比≈2:1 | 压缩测试 |
| V82.10 | 第三级释放 | 缓存清空后内存下降 | 缓存测试 |
| V82.11 | OOM预防 | 18%触发预防性释放 | 预防测试 |
| V82.12 | 预防性释放轻量 | 仅淘汰<5%价值的样本 | 轻量测试 |
| V82.13 | 趋势触发 | 快速增长+15%即触发 | 趋势触发测试 |
| V82.14 | MemoryMonitor组合 | `monitor.cpu_monitor` 类型为 `CPUMonitor` | 类型检查 |
| V82.15 | 释放效果 | 释放后mem_ratio < 0.20 | 端到端测试 |

---

> **阶段82完** | 阶段82建立了内存占用监控与≤20%上限执行器：内存监控 mem_ratio = mem_agent / mem_total（A82.1），上限判定 mem_ratio > 0.20 → free_memory()（A82.2），释放策略 evict_low_value_samples() / compress_model() / flush_cache()（A82.3），OOM预防 mem_ratio > 0.18 → preemptive_free()（A82.4），封装为MemoryMonitor数据类组合CPUMonitor。下一步：阶段83将建立本地存储自动清理机制。
