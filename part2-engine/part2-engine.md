# 自驱永续进化预测修正Agent系统技术方案白皮书

## 第二部分：核心引擎篇（阶段21-40）

---

> **承上启下**：第一部分从第一性原理出发，建立了以自由能最小化为核心的统一数学框架——预测编码提供层级信息处理机制，主动推断提供行动生成机制，进化计算提供结构优化机制，安全约束提供生存保障机制。本部分将理论转化为可执行的工程架构，聚焦三大核心引擎的实现：**环境感知引擎**（阶段21-28）、**驱力引擎**（阶段29-35）、**预测引擎**（阶段36-40）。三大引擎的关系如下：
>
> - **环境感知引擎**：将物理世界的异构信号转化为标准化的环境状态向量 $E_t$，是系统的"感觉器官"
> - **驱力引擎**：基于 $E_t$ 计算三大驱力奖励与价值缺口，生成内生动机，是系统的"动力核心"
> - **预测引擎**：基于 $E_t$ 进行多层预测，输出预测误差与置信度，是系统的"认知核心"
>
> 三者形成闭环：感知→驱力→动机→行动→环境变化→感知，永续运转。

---

## 阶段21：环境感知模块总体架构设计

### 21.1 从理论到工程：感知模块的设计哲学

在第一部分的理论框架中，我们建立了Agent系统的核心动力学方程：

$$\tau_s \dot{\mathbf{s}} = -\nabla_{\mathbf{s}} F(\mathbf{s}, \theta, o)$$

其中观测 $o$ 是感知模块的输出，也是自由能计算的输入。感知模块的质量直接决定了整个系统的上限——如果感知信号充满噪声、维度混乱、信息缺失，那么无论后续的驱力引擎和预测引擎多么精巧，系统都无法做出正确的决策。

从信息论角度，感知模块的核心任务是**最大化互信息** $I(E_t; \text{Environment})$——即环境状态向量 $E_t$ 应尽可能保留环境中的有效信息，同时压缩冗余。这与第一部分阶段5的信息论基础完全一致：

$$I(E_t; \text{Env}) = H(E_t) - H(E_t | \text{Env})$$

好的感知模块应使 $H(E_t)$ 尽可能大（保留信息多样性），同时使 $H(E_t | \text{Env})$ 尽可能小（感知确定性高）。

### 21.2 感知模块的层级架构

感知模块采用四层层级架构，每一层承担明确的信息处理职责：

```
┌─────────────────────────────────────────────────────┐
│                    输出层 (Output Layer)              │
│  输出：标准化环境状态向量 E_t ∈ R^200                 │
│  职责：向量组装、维度对齐、时间戳同步                   │
├─────────────────────────────────────────────────────┤
│                    降噪层 (Denoising Layer)           │
│  输入：标准化指标流                                    │
│  输出：清洁指标流                                      │
│  职责：异常值过滤、缺失值插补、信号平滑、特征缩放        │
├─────────────────────────────────────────────────────┤
│                    标准化层 (Normalization Layer)      │
│  输入：原始采集数据                                    │
│  输出：统一格式指标流                                  │
│  职责：单位统一、格式转换、语义映射、时间对齐            │
├─────────────────────────────────────────────────────┤
│                    采集层 (Collection Layer)           │
│  输入：物理环境信号                                    │
│  输出：原始采集数据                                    │
│  职责：数据源注册、采集调度、缓冲管理、错误恢复          │
└─────────────────────────────────────────────────────┘
```

**层级架构的第一性原理推导**：这一四层架构并非随意设计，而是从预测编码的层级处理原则推导而来。在阶段2中，我们建立了预测编码的层级生成模型 $\hat{\mathbf{r}}^{l-1} = f^l(\mathbf{r}^l, \mathbf{W}^l)$，每一层对下层提供预测并对上层传递误差。感知模块的四层架构遵循同样的信息压缩原则：

- **采集层**对应最底层 $\mathbf{r}^0$：直接接触物理信号，信息量最大但冗余最多
- **标准化层**对应 $\mathbf{r}^1$：消除格式差异，提取语义不变量
- **降噪层**对应 $\mathbf{r}^2$：过滤噪声，保留预测有用的信号
- **输出层**对应 $\mathbf{r}^3$：压缩为紧凑的状态向量，供高层模块使用

### 21.3 采集层架构设计

采集层是感知模块与物理世界的接口，负责从多种数据源获取原始信号。

**数据源注册机制**：每个数据源实现统一的 `DataSource` 接口：

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
import time

@dataclass
class Sample:
    source_id: str
    timestamp: float
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]

class DataSource(ABC):
    @abstractmethod
    def collect(self) -> Optional[Sample]:
        ...

    @abstractmethod
    def get_collect_interval(self) -> float:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def get_schema(self) -> Dict[str, str]:
        ...
```

**采集调度器**：管理多个数据源的采集频率和并发调度：

```python
import asyncio
import heapq
from collections import defaultdict

class CollectionScheduler:
    def __init__(self):
        self.sources: Dict[str, DataSource] = {}
        self.heap: list = []
        self.buffer: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.MAX_ERRORS = 5

    def register(self, source_id: str, source: DataSource):
        self.sources[source_id] = source
        next_time = time.time() + source.get_collect_interval()
        heapq.heappush(self.heap, (next_time, source_id))

    async def run(self):
        while True:
            if not self.heap:
                await asyncio.sleep(0.1)
                continue

            next_time, source_id = self.heap[0]
            now = time.time()
            wait = next_time - now
            if wait > 0:
                await asyncio.sleep(min(wait, 0.05))
                continue

            heapq.heappop(self.heap)
            source = self.sources.get(source_id)
            if source and source.is_available():
                try:
                    sample = source.collect()
                    if sample:
                        await self.buffer.put(sample)
                        self.error_counts[source_id] = 0
                except Exception as e:
                    self.error_counts[source_id] += 1
                    if self.error_counts[source_id] >= self.MAX_ERRORS:
                        source.is_available = lambda: False

            interval = source.get_collect_interval() if source else 60.0
            heapq.heappush(self.heap, (now + interval, source_id))
```

### 21.4 标准化层架构设计

标准化层将异构的原始采集数据转化为统一的中间表示。核心挑战在于：不同数据源的采样频率、数据格式、度量单位各不相同。

**统一数据模型**：

```python
@dataclass
class NormalizedMetric:
    name: str
    value: float
    unit: str
    timestamp: float
    source: str
    confidence: float
    tags: Dict[str, str]

class NormalizationLayer:
    def __init__(self):
        self.unit_converters: Dict[str, callable] = {}
        self.schema_registry: Dict[str, Dict[str, str]] = {}

    def register_schema(self, source_id: str, schema: Dict[str, str]):
        self.schema_registry[source_id] = schema

    def normalize(self, sample: Sample) -> list:
        schema = self.schema_registry.get(sample.source_id, {})
        results = []
        for key, raw_value in sample.metrics.items():
            if not isinstance(raw_value, (int, float)):
                continue
            unit = schema.get(key, "unknown")
            converted = self._convert_unit(raw_value, unit)
            results.append(NormalizedMetric(
                name=f"{sample.source_id}.{key}",
                value=converted,
                unit=self._standard_unit(unit),
                timestamp=sample.timestamp,
                source=sample.source_id,
                confidence=1.0,
                tags=sample.metadata
            ))
        return results

    def _convert_unit(self, value: float, unit: str) -> float:
        converters = {
            "KB": lambda v: v / (1024 ** 2),
            "MB": lambda v: v / 1024,
            "GB": lambda v: v,
            "ms": lambda v: v / 1000,
            "us": lambda v: v / 1e6,
        }
        return converters.get(unit, lambda v: v)(value)

    def _standard_unit(self, unit: str) -> str:
        mapping = {"KB": "GB", "MB": "GB", "GB": "GB",
                   "ms": "s", "us": "s", "s": "s",
                   "%": "%", "count": "count"}
        return mapping.get(unit, unit)
```

**时间对齐机制**：不同数据源的采集时刻不同，需要对齐到统一的时间网格：

```python
import numpy as np

class TimeAligner:
    def __init__(self, grid_interval: float = 1.0):
        self.grid_interval = grid_interval
        self.buffers: Dict[str, list] = {}

    def align(self, metrics: list) -> Dict[str, float]:
        grid_time = round(time.time() / self.grid_interval) * self.grid_interval
        for m in metrics:
            key = m.name
            if key not in self.buffers:
                self.buffers[key] = []
            self.buffers[key].append((m.timestamp, m.value))

        aligned = {}
        for key, buf in self.buffers.items():
            if not buf:
                continue
            recent = [(t, v) for t, v in buf if grid_time - t < self.grid_interval * 3]
            self.buffers[key] = recent[-100:]
            if recent:
                weights = np.array([np.exp(-(grid_time - t) / self.grid_interval) for t, _ in recent])
                values = np.array([v for _, v in recent])
                aligned[key] = np.average(values, weights=weights)
        return aligned
```

### 21.5 降噪层架构设计

降噪层在阶段28中将详细展开，此处给出接口定义：

```python
class DenoisingLayer:
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.stats: Dict[str, Dict[str, float]] = {}

    def process(self, aligned_metrics: Dict[str, float]) -> Dict[str, float]:
        cleaned = {}
        for key, value in aligned_metrics.items():
            self.history[key].append(value)
            if len(self.history[key]) < 3:
                cleaned[key] = value
                continue
            cleaned[key] = self._denoise(key, value)
        return cleaned

    def _denoise(self, key: str, value: float) -> float:
        series = list(self.history[key])
        median = np.median(series)
        mad = np.median(np.abs(np.array(series) - median))
        if mad > 0 and abs(value - median) > 3 * 1.4826 * mad:
            return median
        return value
```

### 21.6 输出层架构设计

输出层将清洁的指标流组装为标准化的环境状态向量 $E_t$：

```python
class EnvironmentStateVector:
    DIMENSION = 200

    def __init__(self):
        self.vector = np.zeros(self.DIMENSION)
        self.dimension_map: Dict[str, tuple] = {}
        self.timestamp: float = 0.0
        self._build_dimension_map()

    def _build_dimension_map(self):
        offset = 0
        sections = {
            "cpu": 30, "memory": 25, "disk": 25, "network": 30,
            "process": 25, "log": 20, "file": 15, "derived": 30
        }
        for section, dim in sections.items():
            self.dimension_map[section] = (offset, offset + dim)
            offset += dim

    def assemble(self, cleaned_metrics: Dict[str, float], timestamp: float) -> np.ndarray:
        self.timestamp = timestamp
        for key, value in cleaned_metrics.items():
            section = key.split(".")[0]
            if section in self.dimension_map:
                start, end = self.dimension_map[section]
                idx = hash(key) % (end - start) + start
                self.vector[idx] = value
        return self.vector.copy()
```

### 21.7 感知频率策略：高频轮询+事件触发双模式

感知模块需要平衡信息完整性和计算开销。我们采用双模式策略：

**模式一：高频轮询（Polling Mode）**

系统指标（CPU、内存等）以固定频率采集，频率根据指标变化速率自适应调整：

$$f_{\text{collect}}^{(i)} = f_{\min} + (f_{\max} - f_{\min}) \cdot \sigma\left(\alpha \cdot \frac{|\Delta x_i|}{\bar{x}_i + \epsilon}\right)$$

其中 $\Delta x_i$ 是指标 $i$ 的近期变化量，$\bar{x}_i$ 是近期均值，$\sigma$ 是sigmoid函数，$\alpha$ 是灵敏度参数。变化越快的指标采集越频繁。

**模式二：事件触发（Event-Driven Mode）**

文件变更、进程事件等离散事件通过回调机制即时感知，无需轮询：

```python
class EventDrivenCollector:
    def __init__(self):
        self.handlers: Dict[str, list] = defaultdict(list)
        self.event_queue: asyncio.Queue = asyncio.Queue()

    def register_handler(self, event_type: str, handler: callable):
        self.handlers[event_type].append(handler)

    async def on_event(self, event_type: str, event_data: dict):
        await self.event_queue.put((event_type, event_data, time.time()))
        for handler in self.handlers.get(event_type, []):
            handler(event_data)

    async def process_events(self):
        while True:
            event_type, data, ts = await self.event_queue.get()
            for handler in self.handlers.get(event_type, []):
                try:
                    handler(data)
                except Exception:
                    pass
```

**双模式融合**：轮询模式提供连续的基线感知，事件触发模式提供离散的突变感知。两者在标准化层汇合，共同构成完整的环境状态。

### 21.8 数据流全景图

```
物理环境
  │
  ├── CPU/Memory/Disk/Network ──→ [系统指标采集器] ──→ 轮询模式
  │                                    │
  ├── syslog/journal/app logs ──→ [日志流解析器] ──→ 事件触发
  │                                    │
  ├── 文件系统变更 ──→ [inotify/watchdog] ──→ 事件触发
  │                                    │
  ├── 进程事件 ──→ [进程监控器] ──→ 轮询+事件
  │                                    │
  └── 网络连接 ──→ [网络感知器] ──→ 轮询模式
                                       │
                              ┌────────▼────────┐
                              │   采集层          │
                              │  Sample{source,   │
                              │   ts, metrics}    │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │   标准化层        │
                              │  NormalizedMetric │
                              │  {name, value,    │
                              │   unit, ts, conf} │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │   降噪层          │
                              │  cleaned_metrics  │
                              │  {key: value}     │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │   输出层          │
                              │  E_t ∈ R^200     │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │  驱力引擎/预测引擎 │
                              └─────────────────┘
```

### 21.9 开源技术选型

| 技术需求 | 开源方案 | 版本建议 | 说明 |
|---------|---------|---------|------|
| 系统指标采集 | psutil | ≥5.9 | 跨平台CPU/内存/磁盘/网络/进程采集 |
| 文件系统监控 | watchdog | ≥3.0 | 基于inotify的跨平台文件监控 |
| Linux原生事件 | pyinotify | ≥0.9 | 直接封装Linux inotify系统调用 |
| 异步框架 | asyncio | 内置 | Python原生异步IO，支持并发采集 |
| 数据缓冲 | collections.deque | 内置 | 高效的固定大小缓冲区 |
| 时间序列对齐 | pandas | ≥2.0 | 时间重采样和对齐 |
| 日志采集 | python-json-logger | ≥2.0 | 结构化日志输出 |

### 21.10 实现路径

1. **采集层实现**：基于psutil实现系统指标采集器，基于watchdog实现文件监控采集器，基于asyncio实现统一调度器
2. **标准化层实现**：定义统一数据模型，实现单位转换和时间对齐
3. **降噪层实现**：实现中值滤波和MAD异常值检测（阶段28详细展开）
4. **输出层实现**：设计200维环境状态向量的维度映射，实现向量组装
5. **双模式融合**：实现轮询模式的自适应频率调整和事件触发模式的回调注册
6. **端到端测试**：验证从物理信号到 $E_t$ 的完整数据流

---

## 阶段22：系统指标采集器实现

### 22.1 采集器的工程目标

系统指标采集器是感知模块采集层的核心组件，负责从操作系统获取CPU、内存、磁盘、网络四大类运行指标。这些指标构成了环境状态向量 $E_t$ 中约110个维度（占55%）的基础数据源，是Agent感知自身运行环境的主要通道。

从第一部分阶段7的控制论基础出发，系统指标采集器对应的是反馈控制系统中的**传感器**——它必须准确、及时、低开销地获取被控对象的状态信息。传感器性能的三个关键指标是：

- **精度**（Accuracy）：采集值与真实值的偏差，由psutil的内核接口保证
- **延迟**（Latency）：从状态变化到采集完成的时间差，由采集频率和调度策略决定
- **开销**（Overhead）：采集过程本身消耗的系统资源，需控制在总资源的1%以内

### 22.2 CPU指标采集

CPU是系统最核心的计算资源，其使用情况直接影响Agent的推理性能和响应速度。

**采集指标清单**：

| 指标名 | 含义 | 采集方式 | 单位 |
|--------|------|---------|------|
| cpu_percent | 总体CPU使用率 | psutil.cpu_percent(interval=None) | % |
| cpu_per_core | 单核使用率 | psutil.cpu_percent(percpu=True) | % |
| cpu_load_1m | 1分钟负载均值 | os.getloadavg()[0] / cpu_count | 比率 |
| cpu_load_5m | 5分钟负载均值 | os.getloadavg()[1] / cpu_count | 比率 |
| cpu_load_15m | 15分钟负载均值 | os.getloadavg()[2] / cpu_count | 比率 |
| cpu_ctx_switches | 上下文切换速率 | psutil.cpu_stats().ctx_switches 差分 | 次/秒 |
| cpu_interrupts | 中断速率 | psutil.cpu_stats().interrupts 差分 | 次/秒 |
| cpu_user | 用户态占比 | psutil.cpu_times_percent().user | % |
| cpu_system | 内核态占比 | psutil.cpu_times_percent().system | % |
| cpu_iowait | IO等待占比 | psutil.cpu_times_percent().iowait | % |
| cpu_steal | 虚拟化偷取占比 | psutil.cpu_times_percent().steal | % |

**实现代码**：

```python
import psutil
import os
import time
from typing import Dict, Any

class CPUCollector(DataSource):
    def __init__(self):
        self._last_ctx = None
        self._last_intr = None
        self._last_time = None
        self._interval = 1.0

    def collect(self) -> Sample:
        now = time.time()
        metrics: Dict[str, Any] = {}

        metrics["cpu_percent"] = psutil.cpu_percent(interval=None)
        per_core = psutil.cpu_percent(percpu=True)
        for i, pct in enumerate(per_core):
            metrics[f"cpu_core_{i}_percent"] = pct

        load_avg = os.getloadavg()
        cpu_count = psutil.cpu_count()
        metrics["cpu_load_1m"] = load_avg[0] / cpu_count
        metrics["cpu_load_5m"] = load_avg[1] / cpu_count
        metrics["cpu_load_15m"] = load_avg[2] / cpu_count

        cpu_times = psutil.cpu_times_percent()
        metrics["cpu_user"] = cpu_times.user
        metrics["cpu_system"] = cpu_times.system
        metrics["cpu_iowait"] = getattr(cpu_times, "iowait", 0.0)
        metrics["cpu_steal"] = getattr(cpu_times, "steal", 0.0)

        stats = psutil.cpu_stats()
        if self._last_ctx is not None and self._last_time is not None:
            dt = now - self._last_time
            if dt > 0:
                metrics["cpu_ctx_switches_rate"] = (stats.ctx_switches - self._last_ctx) / dt
                metrics["cpu_interrupts_rate"] = (stats.interrupts - self._last_intr) / dt
        self._last_ctx = stats.ctx_switches
        self._last_intr = stats.interrupts
        self._last_time = now

        return Sample(
            source_id="cpu",
            timestamp=now,
            metrics=metrics,
            metadata={"cpu_count": cpu_count}
        )

    def get_collect_interval(self) -> float:
        return self._interval

    def is_available(self) -> bool:
        return True

    def get_schema(self) -> Dict[str, str]:
        return {
            "cpu_percent": "%", "cpu_load_1m": "ratio",
            "cpu_load_5m": "ratio", "cpu_load_15m": "ratio",
            "cpu_ctx_switches_rate": "count/s",
            "cpu_interrupts_rate": "count/s",
        }
```

**关键设计决策**：`cpu_percent(interval=None)` 使用非阻塞模式，依赖上一次调用以来的增量计算。这要求采集间隔相对稳定，否则数据会有偏差。我们在调度器中保证采集间隔的稳定性。

### 22.3 内存指标采集

内存指标反映系统的内存压力状况，对Agent的资源调度决策至关重要。

**采集指标清单**：

| 指标名 | 含义 | 采集方式 | 单位 |
|--------|------|---------|------|
| mem_total | 物理内存总量 | psutil.virtual_memory().total | GB |
| mem_used | 已用物理内存 | psutil.virtual_memory().used | GB |
| mem_percent | 内存使用率 | psutil.virtual_memory().percent | % |
| mem_available | 可用内存 | psutil.virtual_memory().available | GB |
| mem_cached | 缓存占用 | psutil.virtual_memory().cached | GB |
| mem_buffers | 缓冲区占用 | psutil.virtual_memory().buffers | GB |
| swap_total | 交换区总量 | psutil.swap_memory().total | GB |
| swap_used | 交换区已用 | psutil.swap_memory().used | GB |
| swap_percent | 交换区使用率 | psutil.swap_memory().percent | % |
| mem_cache_hit_ratio | 缓存命中率 | 1 - (mem_available / (mem_cached + mem_buffers + mem_used)) | 比率 |

**实现代码**：

```python
class MemoryCollector(DataSource):
    def __init__(self):
        self._interval = 2.0
        self._oom_count = 0

    def collect(self) -> Sample:
        now = time.time()
        metrics: Dict[str, Any] = {}
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()

        GB = 1024 ** 3
        metrics["mem_total"] = vm.total / GB
        metrics["mem_used"] = vm.used / GB
        metrics["mem_percent"] = vm.percent
        metrics["mem_available"] = vm.available / GB
        metrics["mem_cached"] = getattr(vm, "cached", 0) / GB
        metrics["mem_buffers"] = getattr(vm, "buffers", 0) / GB

        metrics["swap_total"] = swap.total / GB
        metrics["swap_used"] = swap.used / GB
        metrics["swap_percent"] = swap.percent

        cache_total = getattr(vm, "cached", 0) + getattr(vm, "buffers", 0)
        if vm.used > 0 and cache_total > 0:
            metrics["mem_cache_ratio"] = cache_total / vm.used
        else:
            metrics["mem_cache_ratio"] = 0.0

        if swap.percent > 50:
            metrics["mem_pressure"] = "high"
        elif vm.percent > 85:
            metrics["mem_pressure"] = "medium"
        else:
            metrics["mem_pressure"] = "low"

        return Sample(
            source_id="memory",
            timestamp=now,
            metrics=metrics,
            metadata={}
        )

    def get_collect_interval(self) -> float:
        return self._interval

    def is_available(self) -> bool:
        return True

    def get_schema(self) -> Dict[str, str]:
        return {
            "mem_total": "GB", "mem_used": "GB", "mem_percent": "%",
            "mem_available": "GB", "swap_percent": "%",
        }
```

### 22.4 磁盘指标采集

磁盘IO往往是系统性能瓶颈，尤其是对需要频繁读写模型和日志的Agent系统。

**采集指标清单**：

| 指标名 | 含义 | 采集方式 | 单位 |
|--------|------|---------|------|
| disk_usage_percent | 磁盘使用率 | psutil.disk_usage('/') | % |
| disk_usage_free | 可用空间 | psutil.disk_usage('/') | GB |
| disk_read_bytes_rate | 读取吞吐 | psutil.disk_io_counters() 差分 | MB/s |
| disk_write_bytes_rate | 写入吞吐 | psutil.disk_io_counters() 差分 | MB/s |
| disk_read_count_rate | 读取IOPS | psutil.disk_io_counters() 差分 | 次/秒 |
| disk_write_count_rate | 写入IOPS | psutil.disk_io_counters() 差分 | 次/秒 |
| disk_read_time_rate | 读取耗时 | psutil.disk_io_counters() 差分 | ms/s |
| disk_write_time_rate | 写入耗时 | psutil.disk_io_counters() 差分 | ms/s |
| inode_usage_percent | inode使用率 | os.statvfs | % |

**实现代码**：

```python
class DiskCollector(DataSource):
    def __init__(self, mount_points: list = None):
        self._interval = 3.0
        self._mount_points = mount_points or ["/"]
        self._last_io = None
        self._last_time = None

    def collect(self) -> Sample:
        now = time.time()
        metrics: Dict[str, Any] = {}
        GB = 1024 ** 3

        for mp in self._mount_points:
            try:
                usage = psutil.disk_usage(mp)
                prefix = f"disk_{mp.replace('/', '_').strip('_') or 'root'}"
                metrics[f"{prefix}_usage_percent"] = usage.percent
                metrics[f"{prefix}_usage_free"] = usage.free / GB
            except Exception:
                pass

        io = psutil.disk_io_counters()
        if io and self._last_io and self._last_time:
            dt = now - self._last_time
            if dt > 0:
                MB = 1024 ** 2
                metrics["disk_read_bytes_rate"] = (io.read_bytes - self._last_io.read_bytes) / dt / MB
                metrics["disk_write_bytes_rate"] = (io.write_bytes - self._last_io.write_bytes) / dt / MB
                metrics["disk_read_count_rate"] = (io.read_count - self._last_io.read_count) / dt
                metrics["disk_write_count_rate"] = (io.write_count - self._last_io.write_count) / dt
                metrics["disk_read_time_rate"] = (io.read_time - self._last_io.read_time) / dt
                metrics["disk_write_time_rate"] = (io.write_time - self._last_io.write_time) / dt

                total_io_time = (io.read_time - self._last_io.read_time) + (io.write_time - self._last_io.write_time)
                total_io_count = (io.read_count - self._last_io.read_count) + (io.write_count - self._last_io.write_count)
                metrics["disk_avg_latency"] = total_io_time / max(total_io_count, 1)

        if io:
            self._last_io = io
            self._last_time = now

        try:
            stat = os.statvfs("/")
            total_inodes = stat.f_files
            free_inodes = stat.f_ffree
            if total_inodes > 0:
                metrics["inode_usage_percent"] = (1 - free_inodes / total_inodes) * 100
        except Exception:
            pass

        return Sample(
            source_id="disk",
            timestamp=now,
            metrics=metrics,
            metadata={}
        )

    def get_collect_interval(self) -> float:
        return self._interval

    def is_available(self) -> bool:
        return True

    def get_schema(self) -> Dict[str, str]:
        return {
            "disk_root_usage_percent": "%", "disk_read_bytes_rate": "MB/s",
            "disk_write_bytes_rate": "MB/s", "disk_avg_latency": "ms",
        }
```

### 22.5 网络指标采集

网络指标反映Agent与外界的通信状况，对分布式场景尤为重要。

**采集指标清单**：

| 指标名 | 含义 | 采集方式 | 单位 |
|--------|------|---------|------|
| net_bytes_sent_rate | 发送带宽 | psutil.net_io_counters() 差分 | MB/s |
| net_bytes_recv_rate | 接收带宽 | psutil.net_io_counters() 差分 | MB/s |
| net_packets_sent_rate | 发送包速率 | psutil.net_io_counters() 差分 | 包/秒 |
| net_packets_recv_rate | 接收包速率 | psutil.net_io_counters() 差分 | 包/秒 |
| net_errin_rate | 入错误率 | psutil.net_io_counters() 差分 | 次/秒 |
| net_errout_rate | 出错误率 | psutil.net_io_counters() 差分 | 次/秒 |
| net_dropin_rate | 入丢包率 | psutil.net_io_counters() 差分 | 次/秒 |
| net_dropout_rate | 出丢包率 | psutil.net_io_counters() 差分 | 次/秒 |
| net_connections_count | 连接总数 | psutil.net_connections() | 个 |
| net_tcp_established | 已建立TCP连接 | psutil.net_connections() 筛选 | 个 |
| net_tcp_time_wait | TIME_WAIT连接 | psutil.net_connections() 筛选 | 个 |

**实现代码**：

```python
class NetworkCollector(DataSource):
    def __init__(self):
        self._interval = 2.0
        self._last_io = None
        self._last_time = None

    def collect(self) -> Sample:
        now = time.time()
        metrics: Dict[str, Any] = {}

        io = psutil.net_io_counters()
        if io and self._last_io and self._last_time:
            dt = now - self._last_time
            if dt > 0:
                MB = 1024 ** 2
                metrics["net_bytes_sent_rate"] = (io.bytes_sent - self._last_io.bytes_sent) / dt / MB
                metrics["net_bytes_recv_rate"] = (io.bytes_recv - self._last_io.bytes_recv) / dt / MB
                metrics["net_packets_sent_rate"] = (io.packets_sent - self._last_io.packets_sent) / dt
                metrics["net_packets_recv_rate"] = (io.packets_recv - self._last_io.packets_recv) / dt
                metrics["net_errin_rate"] = (io.errin - self._last_io.errin) / dt
                metrics["net_errout_rate"] = (io.errout - self._last_io.errout) / dt
                metrics["net_dropin_rate"] = (io.dropin - self._last_io.dropin) / dt
                metrics["net_dropout_rate"] = (io.dropout - self._last_io.dropout) / dt

                total_packets = (io.packets_recv - self._last_io.packets_recv)
                total_drops = (io.dropin - self._last_io.dropin) + (io.dropout - self._last_io.dropout)
                if total_packets > 0:
                    metrics["net_drop_ratio"] = total_drops / total_packets
                else:
                    metrics["net_drop_ratio"] = 0.0

        if io:
            self._last_io = io
            self._last_time = now

        try:
            connections = psutil.net_connections(kind='inet')
            metrics["net_connections_count"] = len(connections)
            status_count = {}
            for conn in connections:
                status = conn.status
                status_count[status] = status_count.get(status, 0) + 1
            metrics["net_tcp_established"] = status_count.get("ESTABLISHED", 0)
            metrics["net_tcp_time_wait"] = status_count.get("TIME_WAIT", 0)
            metrics["net_tcp_close_wait"] = status_count.get("CLOSE_WAIT", 0)
        except (psutil.AccessDenied, PermissionError):
            pass

        return Sample(
            source_id="network",
            timestamp=now,
            metrics=metrics,
            metadata={}
        )

    def get_collect_interval(self) -> float:
        return self._interval

    def is_available(self) -> bool:
        return True

    def get_schema(self) -> Dict[str, str]:
        return {
            "net_bytes_sent_rate": "MB/s", "net_bytes_recv_rate": "MB/s",
            "net_drop_ratio": "ratio", "net_tcp_established": "count",
        }
```

### 22.6 高频采集的性能优化

高频采集（1-5秒间隔）对性能有严格要求。关键优化策略：

**1. 差分计算的增量式设计**：所有速率指标通过两次采集的差值计算，避免存储完整历史。

**2. 非阻塞采集**：`cpu_percent(interval=None)` 使用非阻塞模式，但首次调用返回0.0，需要在初始化时做一次预热：

```python
class CollectorWarmup:
    @staticmethod
    def warmup():
        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(percpu=True)
        psutil.disk_io_counters()
        psutil.net_io_counters()
        time.sleep(0.1)
```

**3. 采集开销监控**：采集器自身也需被监控，防止"观察者效应"：

```python
class InstrumentedCollector:
    def __init__(self, collector: DataSource):
        self.collector = collector
        self.collect_times: deque = deque(maxlen=100)

    def collect(self) -> Sample:
        start = time.perf_counter()
        sample = self.collector.collect()
        elapsed = time.perf_counter() - start
        self.collect_times.append(elapsed)
        sample.metadata["collect_duration_ms"] = elapsed * 1000
        return sample

    def get_overhead_ms(self) -> float:
        if not self.collect_times:
            return 0.0
        return np.mean(self.collect_times) * 1000
```

### 22.7 开源技术选型

| 技术需求 | 开源方案 | 说明 |
|---------|---------|------|
| 系统指标采集 | psutil ≥5.9 | 跨平台，API统一，社区活跃 |
| 负载均值 | os.getloadavg() | Python内置，Unix系统调用 |
| 文件系统信息 | os.statvfs() | Python内置，获取inode信息 |
| 高性能计时 | time.perf_counter() | 纳秒级精度，用于性能监控 |
| 数值计算 | numpy | 差分计算、统计聚合 |

### 22.8 实现路径

1. 实现四大采集器（CPU/Memory/Disk/Network），每个独立测试
2. 实现采集调度器，支持频率自适应
3. 实现采集器预热和开销监控
4. 集成到采集层框架，验证端到端数据流
5. 压力测试：在1秒间隔下持续采集24小时，验证稳定性和开销

---

## 阶段23：日志流解析与结构化

### 23.1 日志作为环境感知信号的理论基础

在第一部分的统一框架中，观测 $o_t$ 不仅包括数值型系统指标，还包括符号型事件流——日志就是最重要的符号型信号。从信息论角度，日志携带了系统指标无法捕获的信息：

$$I(\text{Log}; \text{System State}) > I(\text{Metrics}; \text{System State}) \text{ 在异常场景下}$$

当系统正常运行时，日志大多是重复的模板化信息，信息熵低；但当异常发生时，日志中会出现罕见的错误消息和堆栈跟踪，信息熵急剧升高——这正是阶段13异常检测理论中"分布偏移"的典型表现。

### 23.2 系统日志源识别

Agent系统需要监控三类日志源：

**1. 系统日志（syslog/journalctl）**

```python
import subprocess
import asyncio

class SystemLogSource:
    def __init__(self):
        self.journal_process = None

    async def start(self):
        try:
            self.journal_process = await asyncio.create_subprocess_exec(
                "journalctl", "-f", "--output=json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
        except FileNotFoundError:
            self.journal_process = await asyncio.create_subprocess_exec(
                "tail", "-F", "/var/log/syslog",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

    async def read(self) -> dict:
        if not self.journal_process or not self.journal_process.stdout:
            return None
        line = await self.journal_process.stdout.readline()
        if not line:
            return None
        return self._parse_line(line.decode("utf-8", errors="replace").strip())

    def _parse_line(self, line: str) -> dict:
        try:
            import json
            data = json.loads(line)
            return {
                "timestamp": data.get("__REALTIME_TIMESTAMP", ""),
                "level": data.get("PRIORITY", "6"),
                "source": data.get("SYSLOG_IDENTIFIER", "unknown"),
                "message": data.get("MESSAGE", ""),
                "hostname": data.get("_HOSTNAME", ""),
            }
        except json.JSONDecodeError:
            return self._parse_syslog_line(line)

    def _parse_syslog_line(self, line: str) -> dict:
        import re
        pattern = r"^(\w+\s+\d+\s+[\d:]+)\s+(\S+)\s+(\S+?):\s*(.*)$"
        match = re.match(pattern, line)
        if match:
            return {
                "timestamp": match.group(1),
                "source": match.group(3),
                "level": "6",
                "message": match.group(4),
                "hostname": match.group(2),
            }
        return {"timestamp": "", "source": "unknown", "level": "6", "message": line, "hostname": ""}
```

**2. 应用日志**

```python
class ApplicationLogSource:
    def __init__(self, log_path: str, log_format: str = "json"):
        self.log_path = log_path
        self.log_format = log_format
        self._file = None
        self._inode = None

    async def start(self):
        self._file = open(self.log_path, "r")
        self._file.seek(0, 2)
        import os
        self._inode = os.stat(self.log_path).st_ino

    async def read(self) -> list:
        import os
        try:
            if os.stat(self.log_path).st_ino != self._inode:
                self._file.close()
                self._file = open(self.log_path, "r")
                self._inode = os.stat(self.log_path).st_ino
        except FileNotFoundError:
            return []

        lines = self._file.readlines()
        results = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parsed = self._parse(line)
            if parsed:
                results.append(parsed)
        return results

    def _parse(self, line: str) -> dict:
        if self.log_format == "json":
            try:
                import json
                return json.loads(line)
            except json.JSONDecodeError:
                pass
        return self._parse_plain(line)

    def _parse_plain(self, line: str) -> dict:
        import re
        pattern = r"^(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}[.\d]*)\s*\[?(\w+)\]?\s*(\S+)\s*[-:]\s*(.*)$"
        match = re.match(pattern, line)
        if match:
            return {
                "timestamp": match.group(1),
                "level": match.group(2).upper(),
                "source": match.group(3),
                "message": match.group(4),
            }
        return {"timestamp": "", "level": "INFO", "source": "app", "message": line}
```

### 23.3 日志解析器设计：正则模板匹配+动态模式发现

日志解析的核心挑战是：日志格式多样，且可能出现未见过的新格式。我们采用两层解析策略：

**第一层：正则模板匹配**——对已知格式的日志快速解析

```python
import re
from typing import Optional

class LogTemplate:
    def __init__(self, name: str, pattern: str, field_map: dict):
        self.name = name
        self.pattern = re.compile(pattern)
        self.field_map = field_map

class TemplateMatcher:
    def __init__(self):
        self.templates: list = [
            LogTemplate(
                "python_logging",
                r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d+)\s+-\s+(?P<level>\w+)\s+-\s+(?P<source>\S+)\s+-\s+(?P<message>.*)$",
                {}
            ),
            LogTemplate(
                "json_log",
                r'^\{.*\}$',
                {}
            ),
            LogTemplate(
                "syslog_bsd",
                r"^(?P<timestamp>\w+\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(?P<hostname>\S+)\s+(?P<source>\S+?):\s+(?P<message>.*)$",
                {}
            ),
        ]

    def match(self, line: str) -> Optional[dict]:
        for template in self.templates:
            m = template.pattern.match(line)
            if m:
                result = m.groupdict()
                result["template"] = template.name
                return result
        return None
```

**第二层：动态模式发现**——对未知格式日志，使用Drain算法提取日志模板

Drain算法的核心思想是：将相似日志聚合成模板，将具体值替换为通配符 `*`。例如：

```
"Connection timeout to 192.168.1.1:8080" → "Connection timeout to *:*"
"Connection timeout to 10.0.0.1:443"     → "Connection timeout to *:*"
```

```python
class DrainParser:
    def __init__(self, sim_threshold: float = 0.5, max_children: int = 100):
        self.sim_threshold = sim_threshold
        self.max_children = max_children
        self.log_clusters: list = []
        self.token_trees: dict = {}

    def parse(self, line: str) -> dict:
        tokens = line.strip().split()
        if not tokens:
            return {"template": "EMPTY", "params": [], "message": line}

        cluster = self._find_cluster(tokens)
        if cluster:
            template, params = self._extract_params(tokens, cluster["template_tokens"])
            return {"template": " ".join(template), "params": params, "message": line, "cluster_id": cluster["id"]}

        new_cluster = self._create_cluster(tokens)
        return {"template": new_cluster["template"], "params": [], "message": line, "cluster_id": new_cluster["id"]}

    def _find_cluster(self, tokens: list) -> Optional[dict]:
        best_cluster = None
        best_sim = self.sim_threshold
        for cluster in self.log_clusters:
            sim = self._similarity(tokens, cluster["template_tokens"])
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster
        return best_cluster

    def _similarity(self, tokens1: list, tokens2: list) -> float:
        if len(tokens1) != len(tokens2):
            return 0.0
        match = sum(1 for a, b in zip(tokens1, tokens2) if a == b or a == "*" or b == "*")
        return match / max(len(tokens1), 1)

    def _extract_params(self, tokens: list, template_tokens: list) -> tuple:
        result_template = []
        params = []
        for t, tmpl in zip(tokens, template_tokens):
            if tmpl == "*":
                result_template.append("*")
                params.append(t)
            elif t != tmpl:
                result_template.append("*")
                params.append(t)
            else:
                result_template.append(tmpl)
        return result_template, params

    def _create_cluster(self, tokens: list) -> dict:
        cluster_id = len(self.log_clusters)
        cluster = {
            "id": cluster_id,
            "template_tokens": list(tokens),
            "template": " ".join(tokens),
            "count": 1,
        }
        self.log_clusters.append(cluster)
        return cluster
```

### 23.4 日志结构化：统一数据模型

所有日志源解析后统一为结构化格式：

```python
@dataclass
class StructuredLog:
    timestamp: float
    level: str
    source: str
    message: str
    template: str
    params: list
    tags: Dict[str, str]
    cluster_id: int
    raw_line: str

    LEVEL_MAP = {"EMERG": 0, "ALERT": 1, "CRIT": 2, "ERROR": 3,
                 "WARNING": 4, "NOTICE": 5, "INFO": 6, "DEBUG": 7}

    @property
    def level_num(self) -> int:
        return self.LEVEL_MAP.get(self.level.upper(), 6)

    @property
    def is_error(self) -> bool:
        return self.level_num <= 3

    @property
    def is_warning(self) -> bool:
        return self.level_num == 4
```

### 23.5 异常日志提取与聚合

异常日志的提取与聚合是感知模块的关键功能，直接服务于安全驱力的计算：

```python
from collections import defaultdict, Counter

class LogAggregator:
    def __init__(self, window_seconds: int = 300):
        self.window_seconds = window_seconds
        self.error_buffer: deque = deque()
        self.template_counter: Counter = Counter()
        self.source_counter: Counter = Counter()
        self.anomaly_score: float = 0.0

    def add(self, log: StructuredLog):
        now = time.time()
        self.error_buffer.append(log)
        self._evict_old(now)

        if log.is_error:
            self.template_counter[log.template] += 1
            self.source_counter[log.source] += 1
            self._update_anomaly_score(log)

    def _evict_old(self, now: float):
        cutoff = now - self.window_seconds
        while self.error_buffer and self.error_buffer[0].timestamp < cutoff:
            old = self.error_buffer.popleft()
            if old.is_error:
                self.template_counter[old.template] = max(0, self.template_counter[old.template] - 1)
                self.source_counter[old.source] = max(0, self.source_counter[old.source] - 1)

    def _update_anomaly_score(self, log: StructuredLog):
        rarity = 1.0 / (1.0 + self.template_counter[log.template])
        burst_factor = min(len(self.error_buffer) / 10.0, 3.0)
        self.anomaly_score = 0.7 * self.anomaly_score + 0.3 * (rarity * burst_factor)

    def get_metrics(self) -> Dict[str, Any]:
        now = time.time()
        self._evict_old(now)
        recent_errors = [l for l in self.error_buffer if l.is_error]
        recent_warnings = [l for l in self.error_buffer if l.is_warning]

        return {
            "log_error_count_5m": len(recent_errors),
            "log_warning_count_5m": len(recent_warnings),
            "log_total_count_5m": len(self.error_buffer),
            "log_error_rate": len(recent_errors) / max(len(self.error_buffer), 1),
            "log_unique_error_templates": len([k for k, v in self.template_counter.items() if v > 0]),
            "log_anomaly_score": self.anomaly_score,
            "log_top_error_source": self.source_counter.most_common(1)[0][0] if self.source_counter else "none",
        }
```

### 23.6 开源技术选型

| 技术需求 | 开源方案 | 说明 |
|---------|---------|------|
| 结构化日志输出 | python-json-logger ≥2.0 | 将Python logging输出为JSON格式 |
| 日志模板提取 | logparser (Drain) | 学术界主流日志解析算法 |
| 日志流读取 | asyncio.subprocess | 非阻塞读取journalctl/tail输出 |
| 正则匹配 | re | Python内置正则引擎 |
| 计数聚合 | collections.Counter | 高效的计数器数据结构 |

### 23.7 实现路径

1. 实现系统日志源（journalctl/syslog）和应用日志源的读取
2. 实现正则模板匹配器，覆盖常见日志格式
3. 实现Drain算法的简化版本，支持在线日志模板发现
4. 实现日志结构化统一模型
5. 实现异常日志聚合器，输出日志相关指标
6. 集成到采集层框架

---

## 阶段24：文件系统变更监控

### 24.1 文件变更监控的理论意义

在第一部分的主动推断框架中，Agent需要区分"自身行动导致的环境变化"和"外部因素导致的环境变化"。文件系统变更监控正是区分这两者的关键——Agent自身写入的文件是可预测的，而外部进程或用户修改的文件则构成预测误差。

从预测编码的角度，文件变更事件对应底层预测误差 $\mathbf{e}^0$：如果Agent预测某个文件不应被修改，但检测到了修改事件，则产生正的预测误差，需要向上传播以更新高层模型。

### 24.2 基于inotify的实时文件变更监控（Linux）

Linux内核自2.6.13起提供inotify机制，可监控文件的创建、修改、删除、移动等事件。watchdog库封装了inotify，提供跨平台API：

```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent, FileMovedEvent
from dataclasses import dataclass, field
from typing import List
import time

@dataclass
class FileChangeEvent:
    event_type: str
    src_path: str
    dest_path: str = ""
    timestamp: float = 0.0
    is_directory: bool = False
    size: int = 0

class AgentFileHandler(FileSystemEventHandler):
    def __init__(self, event_callback):
        self.event_callback = event_callback
        self._last_events: dict = {}
        self._debounce_ms = 100

    def on_any_event(self, event):
        if event.is_directory:
            return

        now = time.time()
        key = (event.event_type, event.src_path)
        if key in self._last_events:
            if (now - self._last_events[key]) * 1000 < self._debounce_ms:
                return
        self._last_events[key] = now

        change = FileChangeEvent(
            event_type=event.event_type,
            src_path=event.src_path,
            dest_path=getattr(event, "dest_path", ""),
            timestamp=now,
            is_directory=event.is_directory,
        )

        try:
            if event.event_type in ("created", "modified"):
                import os
                change.size = os.path.getsize(event.src_path)
        except OSError:
            pass

        self.event_callback(change)

class FilesystemMonitor(DataSource):
    def __init__(self, watch_paths: List[str], ignore_patterns: List[str] = None):
        self.watch_paths = watch_paths
        self.ignore_patterns = ignore_patterns or ["*.pyc", "__pycache__", ".git", "*.tmp", "*.swp"]
        self.observer = Observer()
        self.event_buffer: asyncio.Queue = asyncio.Queue(maxsize=5000)
        self._handler = AgentFileHandler(self._on_change)
        self._stats = {
            "created": 0, "modified": 0, "deleted": 0, "moved": 0,
            "total": 0
        }
        self._hot_dirs: Counter = Counter()
        self._interval = 5.0

    def _on_change(self, change: FileChangeEvent):
        self._stats[change.event_type] = self._stats.get(change.event_type, 0) + 1
        self._stats["total"] += 1
        dir_path = os.path.dirname(change.src_path)
        self._hot_dirs[dir_path] += 1
        try:
            self.event_buffer.put_nowait(change)
        except asyncio.QueueFull:
            pass

    def start(self):
        for path in self.watch_paths:
            self.observer.schedule(self._handler, path, recursive=True)
        self.observer.start()

    def stop(self):
        self.observer.stop()
        self.observer.join()

    def collect(self) -> Sample:
        now = time.time()
        metrics = {
            "fs_created_count": self._stats.get("created", 0),
            "fs_modified_count": self._stats.get("modified", 0),
            "fs_deleted_count": self._stats.get("deleted", 0),
            "fs_moved_count": self._stats.get("moved", 0),
            "fs_total_changes": self._stats.get("total", 0),
        }

        if self._hot_dirs:
            top_dirs = self._hot_dirs.most_common(5)
            for i, (dir_path, count) in enumerate(top_dirs):
                metrics[f"fs_hotdir_{i}_count"] = count

        self._stats = {k: 0 for k in self._stats}
        self._hot_dirs.clear()

        return Sample(
            source_id="filesystem",
            timestamp=now,
            metrics=metrics,
            metadata={}
        )

    def get_collect_interval(self) -> float:
        return self._interval

    def is_available(self) -> bool:
        return self.observer.is_alive()

    def get_schema(self) -> Dict[str, str]:
        return {
            "fs_created_count": "count", "fs_modified_count": "count",
            "fs_deleted_count": "count", "fs_total_changes": "count",
        }
```

### 24.3 变更事件分类与语义标注

五种变更事件类型对应不同的语义含义：

| 事件类型 | inotify标志 | 语义含义 | 预测误差含义 |
|---------|-----------|---------|------------|
| created | IN_CREATE | 新文件出现 | 新信息产生（正误差） |
| modified | IN_MODIFY | 文件内容变化 | 已有信息更新（小误差） |
| deleted | IN_DELETE | 文件消失 | 信息丢失（负误差） |
| moved | IN_MOVE | 文件移动/重命名 | 信息重组（中性误差） |
| perm_changed | IN_ATTRIB | 权限/属性变更 | 访问控制变化（安全信号） |

```python
class ChangeClassifier:
    def __init__(self):
        self.path_patterns = {
            "config": [r"\.ya?ml$", r"\.json$", r"\.conf$", r"\.ini$", r"\.env$"],
            "code": [r"\.py$", r"\.js$", r"\.go$", r"\.rs$", r"\.java$"],
            "data": [r"\.csv$", r"\.parquet$", r"\.db$", r"\.sqlite$"],
            "log": [r"\.log$", r"/var/log/"],
            "model": [r"\.pt$", r"\.pth$", r"\.onnx$", r"\.bin$"],
            "temp": [r"\.tmp$", r"\.swp$", r"\.cache$", r"__pycache__"],
        }
        self._compiled = {k: [re.compile(p) for p in v] for k, v in self.path_patterns.items()}

    def classify(self, change: FileChangeEvent) -> dict:
        category = "unknown"
        for cat, patterns in self._compiled.items():
            if any(p.search(change.src_path) for p in patterns):
                category = cat
                break

        risk_level = "low"
        if category == "config" and change.event_type in ("modified", "deleted"):
            risk_level = "high"
        elif category == "code" and change.event_type == "modified":
            risk_level = "medium"
        elif category == "model" and change.event_type in ("modified", "deleted"):
            risk_level = "high"
        elif category == "temp":
            risk_level = "ignore"

        return {
            "category": category,
            "risk_level": risk_level,
            "predictable": category in ("log", "temp"),
        }
```

### 24.4 变更频率统计与热点目录识别

热点目录识别帮助Agent理解哪些区域活动最频繁，从而调整注意力分配：

```python
class HotspotDetector:
    def __init__(self, window_seconds: int = 300, top_k: int = 10):
        self.window_seconds = window_seconds
        self.top_k = top_k
        self.events: deque = deque()

    def add_event(self, change: FileChangeEvent):
        self.events.append(change)
        self._evict()

    def _evict(self):
        cutoff = time.time() - self.window_seconds
        while self.events and self.events[0].timestamp < cutoff:
            self.events.popleft()

    def get_hotspots(self) -> list:
        self._evict()
        dir_counts: Counter = Counter()
        for e in self.events:
            dir_path = os.path.dirname(e.src_path)
            dir_counts[dir_path] += 1

        total = sum(dir_counts.values())
        hotspots = []
        for dir_path, count in dir_counts.most_common(self.top_k):
            hotspots.append({
                "path": dir_path,
                "count": count,
                "ratio": count / max(total, 1),
                "activity_level": "high" if count > 50 else "medium" if count > 10 else "low"
            })
        return hotspots

    def get_metrics(self) -> Dict[str, Any]:
        hotspots = self.get_hotspots()
        metrics = {
            "fs_active_dirs": len(hotspots),
            "fs_max_dir_changes": hotspots[0]["count"] if hotspots else 0,
            "fs_change_rate": len(self.events) / self.window_seconds,
        }
        for i, hs in enumerate(hotspots[:3]):
            metrics[f"fs_hotdir_{i}_ratio"] = hs["ratio"]
        return metrics
```

### 24.5 开源技术选型

| 技术需求 | 开源方案 | 说明 |
|---------|---------|------|
| 跨平台文件监控 | watchdog ≥3.0 | 统一API，Linux用inotify，macOS用FSEvents，Windows用ReadDirectoryChangesW |
| Linux原生inotify | pyinotify ≥0.9 | 直接封装inotify系统调用，更细粒度控制 |
| 路径匹配 | fnmatch/re | 文件过滤和分类 |
| 防抖处理 | 自实现 | 合并短时间内对同一文件的多次事件 |

### 24.6 实现路径

1. 基于watchdog实现文件系统监控器，支持多目录递归监控
2. 实现变更事件分类器，标注语义类别和风险等级
3. 实现防抖机制，避免同一文件的高频重复事件
4. 实现热点目录检测器，输出变更频率统计
5. 集成到采集层框架，作为事件触发型数据源

---

## 阶段25：进程行为监控

### 25.1 进程监控与主动推断的关联

在第一部分阶段4的主动推断框架中，Agent的行动 $a_t$ 会改变环境状态 $e_t$。在计算环境中，"行动"的一个重要体现就是进程的创建和执行。进程行为监控帮助Agent区分：

- **自身进程**：Agent自己创建的子进程，行为可预测（预测误差低）
- **已知进程**：系统中常驻的服务进程，行为模式稳定（预测误差低）
- **未知进程**：新出现的进程，行为不可预测（预测误差高，触发安全驱力）

这种区分直接对应阶段1中的核心公式：

$$a^* = \arg\min_a \mathbb{E}[F(\theta, o(a))]$$

Agent需要监控进程行为，才能评估自身行动的效果，并检测外部干扰。

### 25.2 进程树监控

```python
import psutil
from dataclasses import dataclass, field

@dataclass
class ProcessInfo:
    pid: int
    name: str
    cmdline: list
    username: str
    status: str
    create_time: float
    ppid: int
    cpu_percent: float
    memory_percent: float
    num_threads: int
    num_fds: int
    children: list = field(default_factory=list)

class ProcessMonitor(DataSource):
    def __init__(self, self_pid: int = None):
        self._interval = 3.0
        self.self_pid = self_pid or os.getpid()
        self.process_cache: Dict[int, ProcessInfo] = {}
        self.baseline: Dict[str, dict] = {}
        self._last_collect_time = 0.0

    def collect(self) -> Sample:
        now = time.time()
        metrics: Dict[str, Any] = {}

        all_procs = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'username', 'status',
                                          'create_time', 'ppid', 'cpu_percent', 'memory_percent',
                                          'num_threads', 'num_fds']):
            try:
                info = proc.info
                pinfo = ProcessInfo(
                    pid=info['pid'],
                    name=info['name'] or "unknown",
                    cmdline=info['cmdline'] or [],
                    username=info['username'] or "",
                    status=info['status'] or "",
                    create_time=info['create_time'] or 0,
                    ppid=info['ppid'] or 0,
                    cpu_percent=info['cpu_percent'] or 0,
                    memory_percent=info['memory_percent'] or 0,
                    num_threads=info['num_threads'] or 0,
                    num_fds=info.get('num_fds', 0) or 0,
                )
                all_procs.append(pinfo)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        metrics["proc_total_count"] = len(all_procs)

        status_dist = Counter(p.status for p in all_procs)
        metrics["proc_running_count"] = status_dist.get("running", 0)
        metrics["proc_sleeping_count"] = status_dist.get("sleeping", 0)
        metrics["proc_zombie_count"] = status_dist.get("zombie", 0)
        metrics["proc_stopped_count"] = status_dist.get("stopped", 0)

        self_procs = [p for p in all_procs if self._is_self_or_child(p, all_procs)]
        metrics["proc_self_count"] = len(self_procs)
        metrics["proc_self_cpu_total"] = sum(p.cpu_percent for p in self_procs)
        metrics["proc_self_mem_total"] = sum(p.memory_percent for p in self_procs)

        high_cpu = [p for p in all_procs if p.cpu_percent > 50]
        high_mem = [p for p in all_procs if p.memory_percent > 10]
        metrics["proc_high_cpu_count"] = len(high_cpu)
        metrics["proc_high_mem_count"] = len(high_mem)

        known_names = set(self.baseline.keys())
        current_names = set(p.name for p in all_procs)
        new_names = current_names - known_names
        metrics["proc_new_process_types"] = len(new_names)
        metrics["proc_unknown_ratio"] = len(new_names) / max(len(current_names), 1)

        self._update_cache(all_procs)
        self._update_baseline(all_procs)

        return Sample(
            source_id="process",
            timestamp=now,
            metrics=metrics,
            metadata={"new_processes": list(new_names)[:10]}
        )

    def _is_self_or_child(self, proc: ProcessInfo, all_procs: list) -> bool:
        if proc.pid == self.self_pid:
            return True
        pid_map = {p.pid: p for p in all_procs}
        current = proc
        visited = set()
        while current and current.ppid > 0 and current.pid not in visited:
            visited.add(current.pid)
            if current.ppid == self.self_pid:
                return True
            parent = pid_map.get(current.ppid)
            if parent:
                current = parent
            else:
                break
        return False

    def _update_cache(self, all_procs: list):
        self.process_cache = {p.pid: p for p in all_procs}

    def _update_baseline(self, all_procs: list):
        name_stats: Dict[str, list] = defaultdict(list)
        for p in all_procs:
            name_stats[p.name].append(p)

        for name, procs in name_stats.items():
            if name not in self.baseline:
                self.baseline[name] = {
                    "count": len(procs),
                    "avg_cpu": np.mean([p.cpu_percent for p in procs]),
                    "avg_mem": np.mean([p.memory_percent for p in procs]),
                    "first_seen": time.time(),
                }
            else:
                bl = self.baseline[name]
                alpha = 0.1
                bl["count"] = (1 - alpha) * bl["count"] + alpha * len(procs)
                bl["avg_cpu"] = (1 - alpha) * bl["avg_cpu"] + alpha * np.mean([p.cpu_percent for p in procs])
                bl["avg_mem"] = (1 - alpha) * bl["avg_mem"] + alpha * np.mean([p.memory_percent for p in procs])

    def get_collect_interval(self) -> float:
        return self._interval

    def is_available(self) -> bool:
        return True

    def get_schema(self) -> Dict[str, str]:
        return {
            "proc_total_count": "count", "proc_zombie_count": "count",
            "proc_high_cpu_count": "count", "proc_unknown_ratio": "ratio",
        }
```

### 25.3 进程生命周期事件检测

```python
class ProcessLifecycleDetector:
    def __init__(self):
        self.known_pids: Dict[int, dict] = {}
        self.event_queue: deque = deque(maxlen=1000)

    def detect_events(self, current_procs: list) -> list:
        current_pids = {p.pid for p in current_procs}
        previous_pids = set(self.known_pids.keys())

        events = []

        for pid in current_pids - previous_pids:
            proc = next((p for p in current_procs if p.pid == pid), None)
            if proc:
                events.append({
                    "type": "process_created",
                    "pid": pid,
                    "name": proc.name,
                    "ppid": proc.ppid,
                    "timestamp": time.time(),
                })

        for pid in previous_pids - current_pids:
            old_info = self.known_pids[pid]
            events.append({
                "type": "process_terminated",
                "pid": pid,
                "name": old_info.get("name", "unknown"),
                "timestamp": time.time(),
            })

        for proc in current_procs:
            if proc.pid in self.known_pids:
                old_status = self.known_pids[proc.pid].get("status", "")
                if old_status != proc.status:
                    if proc.status == "zombie":
                        events.append({
                            "type": "process_zombie",
                            "pid": proc.pid,
                            "name": proc.name,
                            "timestamp": time.time(),
                        })

        self.known_pids = {
            p.pid: {"name": p.name, "status": p.status, "ppid": p.ppid}
            for p in current_procs
        }

        self.event_queue.extend(events)
        return events
```

### 25.4 进程行为基线与偏差检测

基于第一部分阶段13的异常检测理论，我们为每个已知进程建立行为基线，并检测偏差：

```python
class ProcessBaselineDetector:
    def __init__(self, z_threshold: float = 3.0):
        self.z_threshold = z_threshold
        self.baselines: Dict[str, dict] = {}

    def update_and_check(self, proc: ProcessInfo) -> dict:
        name = proc.name
        if name not in self.baselines:
            self.baselines[name] = {
                "cpu_history": deque(maxlen=100),
                "mem_history": deque(maxlen=100),
                "thread_history": deque(maxlen=100),
            }

        bl = self.baselines[name]
        bl["cpu_history"].append(proc.cpu_percent)
        bl["mem_history"].append(proc.memory_percent)
        bl["thread_history"].append(proc.num_threads)

        anomalies = {}
        for metric_name, history_key in [("cpu", "cpu_history"), ("mem", "mem_history"), ("threads", "thread_history")]:
            history = list(bl[history_key])
            if len(history) < 10:
                continue
            mean = np.mean(history)
            std = np.std(history)
            if std > 0:
                current = history[-1]
                z_score = abs(current - mean) / std
                if z_score > self.z_threshold:
                    anomalies[f"{metric_name}_anomaly"] = {
                        "current": current,
                        "mean": mean,
                        "z_score": z_score,
                    }

        return {
            "process_name": name,
            "pid": proc.pid,
            "is_anomalous": len(anomalies) > 0,
            "anomalies": anomalies,
        }
```

### 25.5 开源技术选型

| 技术需求 | 开源方案 | 说明 |
|---------|---------|------|
| 进程信息采集 | psutil.process_iter() | 跨平台进程信息获取 |
| 进程树遍历 | psutil.Process.children() | 获取子进程关系 |
| 僵尸进程检测 | psutil.Process.status() | 检测zombie状态 |
| 统计计算 | numpy | 均值、标准差、Z-score计算 |

### 25.6 实现路径

1. 实现进程监控器，采集进程列表和资源占用
2. 实现进程树分析，区分自身/已知/未知进程
3. 实现生命周期事件检测器
4. 实现行为基线与偏差检测
5. 集成到采集层框架

---
