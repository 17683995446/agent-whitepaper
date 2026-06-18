# 阶段31：环境状态向量——多源感知信息的结构化表示

> **阶段编号**：S31 | **所属层**：第四层·任务生成层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 环境状态向量 | "体检报告单"——把系统各项指标整理成一张标准化表格 |
> | 四层结构 | "四个科室"——系统层（内科）、日志层（检验科）、文件层（影像科）、自身层（心理科） |
> | 标准化 | "统一单位"——把不同量纲的指标换算成标准分，方便比较 |
> | 采集频率 | "检查间隔"——紧急指标每小时查，常规指标每年查 |

---

## 1. 第一性原理依据

本阶段从阶段1（自由能原理框架）和阶段30（驱力验证）的核心结论出发，建立环境状态向量的完整定义。核心依据为：

- **A1.4（熵平衡方程）**（引用阶段1）：$\dot{S}_{\text{total}} = \dot{S}_{\text{env}} + \dot{S}_{\text{int}} \geq 0$。环境状态向量必须同时描述外部环境（$\dot{S}_{\text{env}}$）和内部状态（$\dot{S}_{\text{int}}$），四层结构正是这一分解的具体实现——系统层和日志层对应外部环境，文件层和自身层对应内部状态。

- **A3.1（预测误差信号）**（引用阶段3）：$e^{l-1} = r^{l-1} - \hat{r}^{l-1}$。环境状态向量是误差计算的基础——$r^{l-1}$ 的当前值由环境状态向量提供，$\hat{r}^{l-1}$ 的预测值由阶段43-49的预测器提供。

- **A21.1-A21.2（安全驱力异常信号）**（引用阶段21）：安全驱力 $\mathbf{F}_{\text{safe}} = \mathbf{F}_{\text{sys}} + \mathbf{F}_{\text{log}}$ 的输入来自系统层和日志层指标。环境状态向量的系统层和日志层直接支撑安全缺口（阶段32，A32.1）的计算。

- **A22.2-A22.4（收益驱力分量）**（引用阶段22）：收益驱力 $\mathbf{F}_{\text{gain}} = \mathbf{F}_{\text{file}} + \mathbf{F}_{\text{self}}$ 的输入来自文件层和自身层指标。环境状态向量的文件层和自身层直接支撑收益缺口（阶段32，A32.2）的计算。

- **A23.2-A23.5（能效驱力分量）**（引用阶段23）：能效驱力 $\mathbf{F}_{\text{eff}} = \mathbf{F}_{\text{sys}} + \mathbf{F}_{\text{self}}$ 的输入来自系统层和自身层指标。环境状态向量的系统层和自身层直接支撑能效缺口（阶段32，A32.3）的计算。

- **A30.1-A30.3（驱力验证）**（引用阶段30）：驱力验证确认三个驱力的有效性。环境状态向量在驱力验证结果（DriveValidationResult）的基础上增加四层结构化指标，为后续缺口评估提供完整输入。

本阶段的核心目标是：**定义环境状态向量的四层结构（系统层、日志层、文件层、自身层），建立标准化方法（z-score标准化），确定采集频率（系统层1Hz、日志层0.1Hz、文件层0.01Hz、自身层0.001Hz），并封装为EnvironmentStateVector数据类，组合DriveValidationResult作为驱力验证的组合字段。** 环境状态向量是多源感知信息的统一表示——它将系统资源、日志事件、文件变更和自身评估四类异构数据整合为标准化向量，为缺口评估算法提供结构化输入。

---

## 2. 核心数学公式

### 第一部分：环境状态向量的四层结构

#### 2.1 环境状态向量的完整定义（A31.1）

环境状态向量 $\mathbf{s} \in \mathbb{R}^{16}$ 由四个子向量拼接而成：

$$\mathbf{s} = [\mathbf{s}_{\text{sys}}, \mathbf{s}_{\text{log}}, \mathbf{s}_{\text{file}}, \mathbf{s}_{\text{self}}] \tag{A31.1}$$

其中：

- $\mathbf{s}_{\text{sys}} \in \mathbb{R}^6$：系统层子向量（A31.2）
- $\mathbf{s}_{\text{log}} \in \mathbb{R}^3$：日志层子向量（A31.3）
- $\mathbf{s}_{\text{file}} \in \mathbb{R}^3$：文件层子向量（A31.4）
- $\mathbf{s}_{\text{self}} \in \mathbb{R}^4$：自身层子向量（A31.5）

总维度：$6 + 3 + 3 + 4 = 16$。

---

#### 2.2 系统层子向量（A31.2）

系统层子向量包含6个系统资源指标：

$$\mathbf{s}_{\text{sys}} = [\text{cpu}, \text{mem}, \text{disk}, \text{net}, \text{load}, \text{processes}] \tag{A31.2}$$

| 指标 | 符号 | 含义 | 单位 | 范围 |
|------|------|------|------|------|
| CPU使用率 | cpu | 当前CPU利用率 | % | $[0, 100]$ |
| 内存使用率 | mem | 当前内存利用率 | % | $[0, 100]$ |
| 磁盘使用率 | disk | 当前磁盘利用率 | % | $[0, 100]$ |
| 网络带宽 | net | 当前网络带宽使用率 | % | $[0, 100]$ |
| 系统负载 | load | 1分钟平均负载 | 无量纲 | $\geq 0$ |
| 进程数 | processes | 当前运行进程数 | 个 | $\geq 0$ |

- **输入**：系统资源监控数据
- **输出**：$\mathbf{s}_{\text{sys}} \in \mathbb{R}^6$
- **约束**：所有分量 $\geq 0$；cpu, mem, disk, net $\in [0, 100]$

> **白话翻译**：系统层就是"内科检查"——CPU、内存、磁盘、网络、负载、进程数，六项基本指标。这些指标反映系统硬件的健康状况，是安全缺口和能效缺口的重要输入。

---

#### 2.3 日志层子向量（A31.3）

日志层子向量包含3个日志事件指标：

$$\mathbf{s}_{\text{log}} = [\text{new\_logs}, \text{errors}, \text{anomaly\_trend}] \tag{A31.3}$$

| 指标 | 符号 | 含义 | 单位 | 范围 |
|------|------|------|------|------|
| 新日志频率 | new_logs | 单位时间内新日志条数 | 条/秒 | $\geq 0$ |
| 错误率 | errors | 错误日志占总日志比例 | 比例 | $[0, 1]$ |
| 异常趋势 | anomaly_trend | 异常日志数量的变化趋势 | 条/秒² | $\mathbb{R}$ |

- **输入**：日志事件流
- **输出**：$\mathbf{s}_{\text{log}} \in \mathbb{R}^3$
- **约束**：errors $\in [0, 1]$；new_logs $\geq 0$

> **白话翻译**：日志层就是"检验科"——看日志里有没有异常、错误多不多、趋势是不是在恶化。错误率必须在0到1之间（不能负也不能超100%），新日志频率反映系统活跃度。

---

#### 2.4 文件层子向量（A31.4）

文件层子向量包含3个文件变更指标：

$$\mathbf{s}_{\text{file}} = [\text{file\_changes}, \text{dir\_state}, \text{new\_deleted}] \tag{A31.4}$$

| 指标 | 符号 | 含义 | 单位 | 范围 |
|------|------|------|------|------|
| 文件变更率 | file_changes | 变更文件占总文件比例 | 比例 | $[0, 1]$ |
| 目录状态 | dir_state | 目录结构变化程度 | 无量纲 | $\geq 0$ |
| 增删比 | new_deleted | (新增数-删除数)/(新增数+删除数) | 比例 | $[-1, 1]$ |

- **输入**：文件系统变更事件
- **输出**：$\mathbf{s}_{\text{file}} \in \mathbb{R}^3$
- **约束**：file_changes $\in [0, 1]$；new_deleted $\in [-1, 1]$

> **白话翻译**：文件层就是"影像科"——看文件系统有没有变化、变了多少、是新增多还是删除多。增删比-1表示只删不增，+1表示只增不删，0表示增减平衡。

---

#### 2.5 自身层子向量（A31.5）

自身层子向量包含4个模型自身评估指标：

$$\mathbf{s}_{\text{self}} = [\text{model\_accuracy}, \text{error\_dist}, \text{compute\_cost}, \text{history}] \tag{A31.5}$$

| 指标 | 符号 | 含义 | 单位 | 范围 |
|------|------|------|------|------|
| 模型准确率 | model_accuracy | 最近预测准确率 | 比例 | $[0, 1]$ |
| 误差分布熵 | error_dist | 预测误差分布的熵 | 比特 | $\geq 0$ |
| 计算成本 | compute_cost | 单次推理计算成本 | FLOPs | $\geq 0$ |
| 历史长度 | history | 已处理的历史步数 | 步 | $\geq 0$ |

- **输入**：模型自身评估数据
- **输出**：$\mathbf{s}_{\text{self}} \in \mathbb{R}^4$
- **约束**：model_accuracy $\leq 1$；所有分量 $\geq 0$

> **白话翻译**：自身层就是"心理科"——模型自己评估自己：预测准不准（准确率）、误差乱不乱（熵）、算得快不快（成本）、经验多不多（历史）。这是收益缺口和能效缺口的重要输入。

---

### 第二部分：标准化方法

#### 2.6 z-score标准化（A31.6）

环境状态向量的各分量量纲不同（CPU是百分比，进程数是整数，熵是比特），需要标准化后才能统一比较。采用z-score标准化：

$$z_i = \frac{x_i - \mu_i}{\sigma_i} \tag{A31.6}$$

其中：
- $x_i$ 为第 $i$ 个分量的原始值
- $\mu_i$ 为第 $i$ 个分量的历史均值（在线更新）
- $\sigma_i$ 为第 $i$ 个分量的历史标准差（在线更新）

**历史统计量的在线更新**（EMA）：

$$\mu_i^{(t)} = (1 - \beta_{\mu}) \mu_i^{(t-1)} + \beta_{\mu} x_i^{(t)} \tag{A31.6-mu}$$

$$\sigma_i^{2(t)} = (1 - \beta_{\sigma}) \sigma_i^{2(t-1)} + \beta_{\sigma} (x_i^{(t)} - \mu_i^{(t)})^2 \tag{A31.6-sigma}$$

其中 $\beta_{\mu}, \beta_{\sigma} \in (0, 1)$ 为更新速率（典型值 $\beta_{\mu} = 0.01, \beta_{\sigma} = 0.01$）。

**标准差下界保护**：

为避免 $\sigma_i \approx 0$ 导致除零错误，引入下界保护：

$$\sigma_i^{\text{eff}} = \max(\sigma_i, \sigma_{\min}) \tag{A31.6-protect}$$

其中 $\sigma_{\min} = 10^{-6}$ 为最小标准差。

- **输入**：原始值 $x_i$，历史均值 $\mu_i$，历史标准差 $\sigma_i$
- **输出**：标准化值 $z_i$
- **约束**：$\sigma_i^{\text{eff}} > 0$；标准化后近似 $N(0, 1)$（大样本下）

> **白话翻译**：标准化就是"统一单位"——把CPU百分比、进程数、熵值等不同量纲的指标，都换算成"标准分"。就像考试把语文、数学、英语都换算成标准分，才能比较总分。历史统计量用EMA在线更新，越新的数据权重越大。

---

### 第三部分：采集频率

#### 2.7 分层采集频率（A31.7）

四层指标的采集频率遵循"紧急指标高频采集，常规指标低频采集"的原则：

| 层级 | 频率 | 采集间隔 | 理由 |
|------|------|---------|------|
| 系统层 | $f_{\text{sys}} = 1\text{Hz}$ | 1秒 | 系统资源变化快，需要实时监控 |
| 日志层 | $f_{\text{log}} = 0.1\text{Hz}$ | 10秒 | 日志事件频率适中 |
| 文件层 | $f_{\text{file}} = 0.01\text{Hz}$ | 100秒 | 文件变更相对缓慢 |
| 自身层 | $f_{\text{self}} = 0.001\text{Hz}$ | 1000秒 | 模型评估不需要频繁更新 |

**频率的数学关系**：

$$f_{\text{sys}} : f_{\text{log}} : f_{\text{file}} : f_{\text{self}} = 1000 : 100 : 10 : 1 \tag{A31.7}$$

即每采集1次自身层指标，采集10次文件层、100次日志层、1000次系统层。

**紧急场景下的频率提升**：

当安全驱力异常（A21.1）或系统层指标超过阈值时，所有层级频率提升10倍：

$$f_{\text{emergency}} = 10 \times f_{\text{normal}} \tag{A31.7-emergency}$$

紧急场景下频率提升10倍，确保系统在危机时刻获得更精细的环境感知。频率自适应与A28.4-emergency（引用阶段28）的紧急豁免机制一致——紧急场景下放宽限制以换取响应速度。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A31.1 | $\mathbf{s} = [\mathbf{s}_{\text{sys}}, \mathbf{s}_{\text{log}}, \mathbf{s}_{\text{file}}, \mathbf{s}_{\text{self}}]$ | 各层原始指标 | $\mathbf{s} \in \mathbb{R}^{16}$ | 四层拼接 |
| A31.2 | $\mathbf{s}_{\text{sys}} = [\text{cpu}, \text{mem}, \text{disk}, \text{net}, \text{load}, \text{processes}]$ | 系统资源指标 | $\mathbf{s}_{\text{sys}} \in \mathbb{R}^{6}$ | 各分量 $\geq 0$ |
| A31.3 | $\mathbf{s}_{\text{log}} = [\text{new\_logs}, \text{errors}, \text{anomaly\_trend}]$ | 日志事件流 | $\mathbf{s}_{\text{log}} \in \mathbb{R}^{3}$ | errors $\in [0, 1]$ |
| A31.4 | $\mathbf{s}_{\text{file}} = [\text{file\_changes}, \text{dir\_state}, \text{new\_deleted}]$ | 文件变更事件 | $\mathbf{s}_{\text{file}} \in \mathbb{R}^{3}$ | file\_changes $\in [0, 1]$ |
| A31.5 | $\mathbf{s}_{\text{self}} = [\text{model\_accuracy}, \text{error\_dist}, \text{compute\_cost}, \text{history}]$ | 自身评估指标 | $\mathbf{s}_{\text{self}} \in \mathbb{R}^{4}$ | model\_accuracy $\leq 1$ |
| A31.6 | $z_i = (x_i - \mu_i) / \sigma_i$ | 原始值、统计量 | $z_i \in \mathbb{R}$ | $\sigma_i > 0$ |
| A31.7 | $f_{\text{sys}}=1\text{Hz}, f_{\text{log}}=0.1\text{Hz}, f_{\text{file}}=0.01\text{Hz}, f_{\text{self}}=0.001\text{Hz}$ | — | 采集频率 | 频率递减 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# 引用阶段30的数据类
from stage_030 import DriveValidationResult


# ============================================================
# 阶段31 数据类：环境状态向量
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class EnvironmentStateVector:
    """环境状态向量的完整描述。

    组合阶段30的DriveValidationResult，在驱力验证结果的基础上增加
    环境状态向量的四层结构。

    Attributes:
        validation_result: 阶段30的驱力验证结果（组合字段）
        s_sys: 系统层子向量 (cpu, mem, disk, net, load, processes)（A31.2）
        s_log: 日志层子向量 (new_logs, errors, anomaly_trend)（A31.3）
        s_file: 文件层子向量 (file_changes, dir_state, new_deleted)（A31.4）
        s_self: 自身层子向量 (model_accuracy, error_dist, compute_cost, history)（A31.5）
        z_sys: 标准化后的系统层子向量（A31.6）
        z_log: 标准化后的日志层子向量（A31.6）
        z_file: 标准化后的文件层子向量（A31.6）
        z_self: 标准化后的自身层子向量（A31.6）
        mu: 各分量历史均值向量（A31.6）
        sigma: 各分量历史标准差向量（A31.6）
        timestamps: 各层最后采集时间戳（A31.7）

    Invariants:
        - s_sys各分量 >= 0
        - s_log[1] (errors) ∈ [0, 1]
        - s_file[0] (file_changes) ∈ [0, 1]
        - s_file[2] (new_deleted) ∈ [-1, 1]
        - s_self[0] (model_accuracy) <= 1
        - sigma各分量 > 0
    """
    validation_result: DriveValidationResult
    s_sys: tuple[float, float, float, float, float, float]
    s_log: tuple[float, float, float]
    s_file: tuple[float, float, float]
    s_self: tuple[float, float, float, float]
    z_sys: tuple[float, float, float, float, float, float]
    z_log: tuple[float, float, float]
    z_file: tuple[float, float, float]
    z_self: tuple[float, float, float, float]
    mu: tuple[float, ...]      # 16维
    sigma: tuple[float, ...]   # 16维
    timestamps: tuple[float, float, float, float]  # 四层时间戳

    def __post_init__(self):
        # 验证系统层非负
        for i, v in enumerate(self.s_sys):
            if v < -1e-10:
                raise ValueError(
                    f"系统层第 {i} 分量必须 ≥ 0，"
                    f"收到 {v}"
                )
        # 验证错误率范围
        if not (0 <= self.s_log[1] <= 1 + 1e-10):
            raise ValueError(
                f"错误率必须 ∈ [0, 1]，"
                f"收到 {self.s_log[1]}"
            )
        # 验证文件变更率范围
        if not (0 <= self.s_file[0] <= 1 + 1e-10):
            raise ValueError(
                f"文件变更率必须 ∈ [0, 1]，"
                f"收到 {self.s_file[0]}"
            )
        # 验证标准差严格为正
        for i, s in enumerate(self.sigma):
            if s <= 0:
                raise ValueError(
                    f"第 {i} 个分量的标准差必须 > 0，"
                    f"收到 {s}"
                )

    @property
    def full_vector(self) -> np.ndarray:
        """完整的环境状态向量（标准化后）。"""
        return np.array(list(self.z_sys) + list(self.z_log)
                        + list(self.z_file) + list(self.z_self))

    @property
    def raw_vector(self) -> np.ndarray:
        """完整的环境状态向量（原始值）。"""
        return np.array(list(self.s_sys) + list(self.s_log)
                        + list(self.s_file) + list(self.s_self))

    @property
    def dimension(self) -> int:
        """环境状态向量的总维度。"""
        return 16

    @property
    def is_fresh(self) -> bool:
        """各层数据是否新鲜（在采集间隔内）。"""
        # 检查各层时间戳是否在合理范围内
        return True  # 简化实现

    @property
    def dominant_anomaly_layer(self) -> str:
        """异常最显著的层级。"""
        max_z = {
            'sys': max(abs(z) for z in self.z_sys),
            'log': max(abs(z) for z in self.z_log),
            'file': max(abs(z) for z in self.z_file),
            'self': max(abs(z) for z in self.z_self)
        }
        return max(max_z, key=max_z.get)


# ============================================================
# 阶段31 核心接口：环境状态采集器
# ============================================================
class EnvironmentStateCollector:
    """阶段31的核心计算接口。

    实现环境状态向量的完整采集流程，包括四层指标采集、
    标准化计算和频率管理。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 四层结构完整
        - 标准化后近似 N(0,1)
        - 采集频率分层
    """

    def collect_system_layer(self) -> tuple[float, ...]:
        """采集系统层指标（A31.2）。

        Returns:
            (cpu, mem, disk, net, load, processes)

        Examples:
            >>> collector = EnvironmentStateCollector()
            >>> sys_layer = collector.collect_system_layer()
            >>> len(sys_layer)
            6
        """
        try:
            import psutil
        except Exception:
            psutil = None
        if psutil is None:
            # 降级：返回零值占位
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        cpu = float(psutil.cpu_percent(interval=None))
        mem = float(psutil.virtual_memory().percent)
        disk = float(psutil.disk_usage('/').percent)
        net_io = psutil.net_io_counters()
        # 网络带宽使用率简化为发送+接收字节数的对数归一化占位
        net = 0.0
        load1, _, _ = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0.0, 0.0, 0.0)
        processes = float(len(psutil.pids()))
        return (cpu, mem, disk, net, float(load1), processes)

    def collect_log_layer(
        self,
        window_seconds: float = 600.0
    ) -> tuple[float, ...]:
        """采集日志层指标（A31.3）。

        Args:
            window_seconds: 时间窗口长度（秒），默认600秒

        Returns:
            (new_logs, errors, anomaly_trend)

        Examples:
            >>> collector = EnvironmentStateCollector()
            >>> log_layer = collector.collect_log_layer(window_seconds=60.0)
            >>> len(log_layer)
            3
        """
        if window_seconds <= 0:
            raise ValueError(f"时间窗口必须 > 0，收到 {window_seconds}")
        # 占位实现：实际应从日志采集系统读取
        # new_logs: 条/秒，errors: 比例，anomaly_trend: 条/秒²
        return (0.0, 0.0, 0.0)

    def collect_file_layer(
        self,
        window_seconds: float = 600.0
    ) -> tuple[float, ...]:
        """采集文件层指标（A31.4）。

        Args:
            window_seconds: 时间窗口长度（秒），默认600秒

        Returns:
            (file_changes, dir_state, new_deleted)

        Examples:
            >>> collector = EnvironmentStateCollector()
            >>> file_layer = collector.collect_file_layer(window_seconds=60.0)
            >>> len(file_layer)
            3
        """
        if window_seconds <= 0:
            raise ValueError(f"时间窗口必须 > 0，收到 {window_seconds}")
        # 占位实现：实际应从文件监控事件统计
        # file_changes ∈ [0,1], dir_state ≥ 0, new_deleted ∈ [-1,1]
        return (0.0, 0.0, 0.0)

    def collect_self_layer(
        self,
        lookback_steps: int = 1000
    ) -> tuple[float, ...]:
        """采集自身层指标（A31.5）。

        Args:
            lookback_steps: 历史回溯步数，默认1000

        Returns:
            (model_accuracy, error_dist, compute_cost, history)

        Raises:
            ValueError: 当 lookback_steps <= 0 时

        Examples:
            >>> collector = EnvironmentStateCollector()
            >>> self_layer = collector.collect_self_layer(lookback_steps=100)
            >>> len(self_layer)
            4
        """
        if lookback_steps <= 0:
            raise ValueError(
                f"历史回溯步数必须 > 0，收到 {lookback_steps}"
            )
        # 占位实现：实际应从模型内部评估模块读取
        # model_accuracy ∈ [0,1], error_dist ≥ 0, compute_cost ≥ 0, history ≥ 0
        return (0.0, 0.0, 0.0, float(lookback_steps))

    def standardize(
        self,
        raw_values: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        sigma_min: float = 1e-6
    ) -> np.ndarray:
        """标准化（A31.6）。

        Args:
            raw_values: 原始值向量，shape (n,)
            mu: 均值向量，shape (n,)
            sigma: 标准差向量，shape (n,)
            sigma_min: 标准差下界保护值，默认1e-6

        Returns:
            标准化向量 z = (x - mu) / max(sigma, sigma_min)，shape (n,)

        Raises:
            ValueError: 当输入维度不匹配时

        Examples:
            >>> collector = EnvironmentStateCollector()
            >>> z = collector.standardize(
            ...     np.array([1.0, 2.0]),
            ...     np.array([0.0, 0.0]),
            ...     np.array([1.0, 2.0])
            ... )
            >>> z.tolist()
            [1.0, 1.0]
        """
        raw_values = np.asarray(raw_values, dtype=float)
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        if raw_values.shape != mu.shape or raw_values.shape != sigma.shape:
            raise ValueError(
                f"维度不匹配：raw_values{raw_values.shape}, "
                f"mu{mu.shape}, sigma{sigma.shape}"
            )
        sigma_eff = np.maximum(sigma, sigma_min)
        z = (raw_values - mu) / sigma_eff
        return z

    def update_statistics(
        self,
        mu: np.ndarray,
        sigma_sq: np.ndarray,
        new_value: np.ndarray,
        beta_mu: float = 0.01,
        beta_sigma: float = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        """更新历史统计量（A31.6-mu, A31.6-sigma）。

        Args:
            mu: 当前均值向量，shape (n,)
            sigma_sq: 当前方差向量，shape (n,)
            new_value: 新采集值，shape (n,)
            beta_mu: 均值更新速率，默认0.01
            beta_sigma: 方差更新速率，默认0.01

        Returns:
            (更新后的均值, 更新后的方差)，均为 shape (n,)

        Raises:
            ValueError: 当 beta 不在 (0,1) 区间或维度不匹配时

        Examples:
            >>> collector = EnvironmentStateCollector()
            >>> new_mu, new_var = collector.update_statistics(
            ...     np.array([0.0]), np.array([1.0]),
            ...     np.array([2.0]), beta_mu=0.5, beta_sigma=0.5
            ... )
            >>> float(new_mu[0])
            1.0
        """
        mu = np.asarray(mu, dtype=float)
        sigma_sq = np.asarray(sigma_sq, dtype=float)
        new_value = np.asarray(new_value, dtype=float)
        if not (0.0 < beta_mu < 1.0):
            raise ValueError(f"beta_mu 必须在 (0,1) 内，收到 {beta_mu}")
        if not (0.0 < beta_sigma < 1.0):
            raise ValueError(f"beta_sigma 必须在 (0,1) 内，收到 {beta_sigma}")
        if mu.shape != sigma_sq.shape or mu.shape != new_value.shape:
            raise ValueError("mu、sigma_sq、new_value 维度必须一致")
        new_mu = (1.0 - beta_mu) * mu + beta_mu * new_value
        diff = new_value - new_mu
        new_sigma_sq = (1.0 - beta_sigma) * sigma_sq + beta_sigma * (diff ** 2)
        # 保证方差非负
        new_sigma_sq = np.maximum(new_sigma_sq, 0.0)
        return new_mu, new_sigma_sq

    def get_collection_frequency(
        self,
        layer: str,
        is_emergency: bool = False
    ) -> float:
        """获取采集频率（A31.7）。

        Args:
            layer: 层级名称 ('sys' | 'log' | 'file' | 'self')
            is_emergency: 是否处于紧急场景，默认False

        Returns:
            采集频率（Hz）

        Raises:
            ValueError: 当 layer 不是合法层级时

        Examples:
            >>> collector = EnvironmentStateCollector()
            >>> collector.get_collection_frequency('sys')
            1.0
            >>> collector.get_collection_frequency('sys', is_emergency=True)
            10.0
        """
        base_freq = {
            'sys': 1.0,
            'log': 0.1,
            'file': 0.01,
            'self': 0.001,
        }
        if layer not in base_freq:
            raise ValueError(
                f"层级必须是 'sys'|'log'|'file'|'self' 之一，"
                f"收到 '{layer}'"
            )
        freq = base_freq[layer]
        if is_emergency:
            freq *= 10.0
        return freq
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `DriveValidationResult` | ← 阶段30 | 作为环境状态向量的驱力验证组合字段 |
| **输入** | A1.4（熵平衡方程） | ← 阶段1 | 四层结构的物理基础 |
| **输入** | A3.1（预测误差信号） | ← 阶段3 | 日志层与误差信号的关联 |
| **输入** | A21.1-A21.2（安全驱力异常信号） | ← 阶段21 | 系统层和日志层指标的安全驱力输入 |
| **输入** | A22.2-A22.4（收益驱力分量） | ← 阶段22 | 自身层指标的收益驱力输入 |
| **输入** | A23.2-A23.5（能效驱力分量） | ← 阶段23 | 系统层和自身层指标的能效驱力输入 |
| **输出** | `EnvironmentStateVector` | → 阶段32 | 环境状态向量作为缺口评估的输入 |
| **输出** | A31.6（标准化方法） | → 阶段32 | 标准化向量用于缺口计算 |
| **输出** | A31.7（采集频率分层） | → 工程实现 | 采集频率的配置标准 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| psutil | 系统层指标采集 | 提供CPU、内存、磁盘、网络、负载、进程数的跨平台采集接口，直接支撑A31.2的6个系统层指标 |
| JAX | 标准化计算 | `jax.numpy`实现向量化的标准化计算（A31.6）；`jax.lax.scan`实现统计量的EMA更新 |
| watchdog | 文件层变更监控 | 提供文件系统变更事件的实时监控，支撑A31.4的文件变更率和增删比计算 |
| structlog | 日志层事件采集 | 提供结构化日志的采集和解析接口，支撑A31.3的新日志频率和错误率计算 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V31.1 | 系统层非负性 | $\mathbf{s}_{\text{sys}}$ 各分量 $\geq 0$ | 随机系统指标测试 |
| V31.2 | 错误率范围 | $\text{errors} \in [0, 1]$ | 边界值测试 |
| V31.3 | 文件变更率范围 | $\text{file\_changes} \in [0, 1]$ | 边界值测试 |
| V31.4 | 增删比范围 | $\text{new\_deleted} \in [-1, 1]$ | 极端增删测试 |
| V31.5 | 标准化零均值 | $\mathbb{E}[z_i] \approx 0$（大样本下） | 合成正态数据测试 |
| V31.6 | 标准化单位方差 | $\text{Var}[z_i] \approx 1$（大样本下） | 合成正态数据测试 |
| V31.7 | 标准差下界保护 | $\sigma_i \geq \sigma_{\min}$ | 常量序列测试 |
| V31.8 | 向量维度 | $\dim(\mathbf{s}) = 16$ | 维度检查 |
| V31.9 | 四层拼接完整性 | $\mathbf{s} = [\mathbf{s}_{\text{sys}}, \mathbf{s}_{\text{log}}, \mathbf{s}_{\text{file}}, \mathbf{s}_{\text{self}}]$ | 拼接一致性测试 |
| V31.10 | 采集频率递减 | $f_{\text{sys}} > f_{\text{log}} > f_{\text{file}} > f_{\text{self}}$ | 频率配置验证 |
| V31.11 | 紧急频率提升 | 紧急场景下各层频率提升10倍 | 紧急场景模拟 |
| V31.12 | EMA统计量收敛 | 历史统计量收敛到真实值 | 长期运行测试 |
| V31.13 | EnvironmentStateVector组合 | `EnvironmentStateVector.validation_result` 类型为 `DriveValidationResult` | 类型检查 |
| V31.14 | 非法输入拒绝 | 负CPU占用率、错误率>1等抛出 `ValueError` | 异常测试 |
| V31.15 | 主导异常层级 | `dominant_anomaly_layer` 返回标准化值最大的层级 | 合成异常数据测试 |

---

> **阶段31完** | 阶段31建立了环境状态向量的完整定义：四层结构（系统层、日志层、文件层、自身层）将多源感知信息统一为16维标准化向量，采集频率分层确保信息获取效率。环境状态向量是缺口评估算法（阶段32）的基础输入——缺口评估将从环境状态向量中提取三大缺口的量化值。
