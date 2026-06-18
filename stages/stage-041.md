# 阶段41：底层模型（高频微预测）——系统级快速预测的目标与特征

> **阶段编号**：S41 | **所属层**：第五层·层级预测层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 底层预测目标 | "显微镜下的观察对象"——系统负载、日志变化、文件活动等最细粒度的指标 |
> | 采集频率 | "采样的快慢"——1Hz看系统指标，0.1Hz看日志，越快越细 |
> | 预测窗口 | "能看多远"——1到10步的短期预测，只看眼前 |
> | 底层特征工程 | "数据预处理"——滑动均值、滑动标准差、差分、变化率，把原始信号变成可预测的特征 |
> | LowLevelTarget | "底层预测的完整画像"——组合决策引擎状态，在决策基础上增加底层预测目标 |

---

## 1. 第一性原理依据

本阶段从阶段11（层级生成模型）和阶段3（预测编码公理）的核心结论出发，建立层级预测体系的最底层——高频微预测模型。核心依据为：

- **A11.1（层级生成模型联合分布）**（引用阶段11）：$p(o, r^{1:L}) = p(o|r^1)\prod_{l=1}^{L}p(r^l|r^{l+1})p(r^{L+1})$，层级生成模型将预测体系分为多个抽象层级。底层模型对应 $p(o|r^1)$——从最细粒度的表征生成观测预测。底层预测目标是层级生成模型中第一层表征 $r^1$ 所编码的信息。

- **A11.4（深度-抽象度关系）**（引用阶段11）：$\text{abstraction}(l) \propto \tau_l$，层级越深，抽象度越高，时间常数越大。底层（$l=1$）具有最小的时间常数 $\tau_1$，对应最快的变化速率——这正是高频微预测的理论基础。底层预测目标的变化速率远高于中层和高层，因此需要更高的采集频率。

- **A3.1（预测误差信号）**（引用阶段3）：$e^{l-1} = r^{l-1} - \hat{r}^{l-1}$，预测误差的定义。底层预测的误差信号 $e^0 = o - \hat{o}$ 直接度量了系统级观测与预测的偏差，是自下而上误差传播（引用阶段12，A12.1）的起点。

- **A3.2（精度加权误差能量）**（引用阶段3）：$\varepsilon_l = e^{l-1,T}\Pi_l e^{l-1}$，精度加权误差能量。底层预测的精度矩阵 $\Pi_1$ 决定了底层误差信号在层级传播中的权重——精度越高，底层误差对中高层的影响越大。

本阶段的核心目标是：**定义底层预测目标（system_load, log_change, file_activity）、确定采集频率（1Hz/0.1Hz）、设定预测窗口（1-10步）、设计底层特征工程（rolling_mean, rolling_std, diff, rate_of_change），并封装为LowLevelTarget数据类，组合DecisionEngineState作为决策引擎状态的输入。** 底层模型是层级预测体系的基础——它提供最细粒度、最高频率的预测信号，是中层趋势预测和高层抽象预测的信号来源。

---

## 2. 核心数学公式

### 第一部分：底层预测目标

#### 2.1 底层预测目标定义（A41.1）

底层预测目标 $\mathbf{y}_{\text{low}} \in \mathbb{R}^{d_{\text{low}}}$ 由三个子向量组成：

$$\mathbf{y}_{\text{low}} = [\mathbf{y}_{\text{sys}}, \mathbf{y}_{\text{log}}, \mathbf{y}_{\text{file}}] \tag{A41.1}$$

其中：

- $\mathbf{y}_{\text{sys}} \in \mathbb{R}^{d_{\text{sys}}}$：系统负载指标向量，包含CPU利用率、内存占用率、磁盘IO速率、网络吞吐量等系统级指标。$d_{\text{sys}}$ 为系统指标维度（典型值 $d_{\text{sys}} = 8$）。

- $\mathbf{y}_{\text{log}} \in \mathbb{R}^{d_{\text{log}}}$：日志变化指标向量，包含日志条目速率、错误日志比例、警告日志比例、异常模式计数等日志级指标。$d_{\text{log}}$ 为日志指标维度（典型值 $d_{\text{log}} = 6$）。

- $\mathbf{y}_{\text{file}} \in \mathbb{R}^{d_{\text{file}}}$：文件活动指标向量，包含文件创建速率、文件修改速率、文件删除速率、目录变化速率等文件级指标。$d_{\text{file}}$ 为文件指标维度（典型值 $d_{\text{file}} = 4$）。

底层预测目标的总维度为：

$$d_{\text{low}} = d_{\text{sys}} + d_{\text{log}} + d_{\text{file}} \tag{A41.1-dim}$$

典型值 $d_{\text{low}} = 18$。

- **输入**：系统原始指标数据、日志流数据、文件系统事件数据
- **输出**：底层预测目标向量 $\mathbf{y}_{\text{low}} \in \mathbb{R}^{d_{\text{low}}}$
- **约束**：各子向量维度固定；$\mathbf{y}_{\text{low}}$ 的各分量非负（速率、比例类指标）

> **白话翻译**：底层预测目标就是"显微镜下看到的东西"——系统负载（CPU忙不忙、内存够不够）、日志变化（报错多不多、异常有没有）、文件活动（文件在不在变、目录在不在改）。这三个维度拼起来就是底层预测的完整目标。

**与层级生成模型的对应**：

在A11.1的层级生成模型中，底层预测目标 $\mathbf{y}_{\text{low}}$ 对应第一层表征 $r^1$ 所生成的观测预测 $\hat{o}$。具体地：

$$\hat{o} = g(r^1) \approx \mathbf{y}_{\text{low}}$$

其中 $g$ 为观测生成函数（引用阶段11，A11.2-obs）。底层预测的目标是最小化预测误差：

$$e^0 = o - \hat{o} = o - g(r^1) \tag{A41.1-error}$$

此误差信号是A12.1（引用阶段12）自下而上误差传播的起点。

**与决策引擎状态的对应**：

底层预测目标 $\mathbf{y}_{\text{low}}$ 的部分分量与决策引擎的环境状态向量（引用阶段31，A31.1）重叠：

| 底层预测目标 | 环境状态向量分量 | 关系 |
|-------------|----------------|------|
| $\mathbf{y}_{\text{sys}}$ | $\mathbf{s}_{\text{sys}}$ | $\mathbf{y}_{\text{sys}}$ 是 $\mathbf{s}_{\text{sys}}$ 的时序扩展 |
| $\mathbf{y}_{\text{log}}$ | $\mathbf{s}_{\text{log}}$ | $\mathbf{y}_{\text{log}}$ 是 $\mathbf{s}_{\text{log}}$ 的变化率 |
| $\mathbf{y}_{\text{file}}$ | $\mathbf{s}_{\text{file}}$ | $\mathbf{y}_{\text{file}}$ 是 $\mathbf{s}_{\text{file}}$ 的活动速率 |

底层预测目标不仅包含当前状态，还包含状态的变化趋势——这是预测模型与状态评估的核心区别。

---

### 第二部分：采集频率

#### 2.2 采集频率定义（A41.2）

底层预测目标的不同子向量具有不同的采集频率，反映了其变化速率的差异：

$$f_{\text{sys}} = 1\text{Hz}, \quad f_{\text{log}} = 0.1\text{Hz}, \quad f_{\text{file}} = 1\text{Hz} \tag{A41.2}$$

对应的采样间隔为：

$$\Delta t_{\text{sys}} = 1\text{s}, \quad \Delta t_{\text{log}} = 10\text{s}, \quad \Delta t_{\text{file}} = 1\text{s} \tag{A41.2-interval}$$

**频率选择的理论依据**：

由A11.4（引用阶段11），$\text{abstraction}(l) \propto \tau_l$，底层的时间常数 $\tau_1$ 最小。根据Nyquist采样定理，采样频率必须至少为信号最高频率分量的2倍：

$$f_{\text{sample}} \geq 2 f_{\text{max}} \tag{A41.2-nyquist}$$

系统负载指标的变化频率最高（秒级波动），因此需要1Hz采样。日志变化相对缓慢（通常以10秒为单位才有显著变化），因此0.1Hz采样足够。文件活动与系统负载同频，1Hz采样。

**多频率对齐**：

不同频率的信号需要对齐到统一的时间网格上。设系统时钟以 $\Delta t_{\text{base}} = 1\text{s}$ 为基准，则：

- 系统负载：每步采样一次（$k_{\text{sys}} = 1$）
- 日志变化：每10步采样一次（$k_{\text{log}} = 10$）
- 文件活动：每步采样一次（$k_{\text{file}} = 1$）

在非采样步，日志变化指标使用最近一次采样值（零阶保持）：

$$\mathbf{y}_{\text{log}}(t) = \mathbf{y}_{\text{log}}\left(\lfloor t / k_{\text{log}} \rfloor \cdot k_{\text{log}}\right) \tag{A41.2-hold}$$

- **输入**：各子向量的变化频率估计
- **输出**：采集频率 $f_{\text{sys}}, f_{\text{log}}, f_{\text{file}}$ 和采样间隔
- **约束**：满足Nyquist采样定理；系统负载和文件活动频率不低于1Hz

> **白话翻译**：采集频率就是"多久看一次"——系统指标每秒看一次（变化快），日志每10秒看一次（变化慢），文件活动每秒看一次（变化快）。不同指标看的频率不同，但都对齐到1秒的统一时钟上。日志在不采样的时候，用上一次的值顶替。

---

### 第三部分：预测窗口

#### 2.3 预测窗口定义（A41.3）

底层预测的预测窗口 $h_{\text{low}}$ 定义为预测步数范围：

$$h_{\text{low}} \in [1, 10] \tag{A41.3}$$

以系统负载的1Hz频率为基准，预测窗口对应的时间范围为：

$$T_{\text{predict}} = h_{\text{low}} \cdot \Delta t_{\text{base}} \in [1\text{s}, 10\text{s}] \tag{A41.3-time}$$

**预测窗口的选择依据**：

底层预测窗口的选择由两个因素决定：

1. **预测精度衰减**：预测步数越多，预测误差越大。设第 $h$ 步的预测误差方差为 $\sigma^2(h)$，则：

$$\sigma^2(h) = \sigma^2_0 \cdot (1 + \alpha h) \tag{A41.3-error}$$

其中 $\sigma^2_0$ 为一步预测误差方差，$\alpha > 0$ 为误差增长系数。当 $\sigma^2(h) > \sigma^2_{\text{threshold}}$ 时，预测不再可靠。由此确定预测窗口上界 $h_{\max} = 10$。

2. **决策响应时间**：底层预测需要为决策引擎提供足够的前瞻时间。由阶段39（动态中断机制），最小执行时间 $T_{\text{min\_exec}} = 5$ 步。底层预测窗口至少覆盖 $T_{\text{min\_exec}}$，即 $h_{\text{low}} \geq 5$。

**多步预测的输出格式**：

对于预测窗口 $h_{\text{low}}$，底层模型输出一个预测序列：

$$\hat{\mathbf{Y}}_{\text{low}} = [\hat{\mathbf{y}}_{\text{low}}(t+1), \hat{\mathbf{y}}_{\text{low}}(t+2), \ldots, \hat{\mathbf{y}}_{\text{low}}(t+h_{\text{low}})] \tag{A41.3-output}$$

其中 $\hat{\mathbf{y}}_{\text{low}}(t+h) \in \mathbb{R}^{d_{\text{low}}}$ 为第 $h$ 步的预测值。预测序列的总维度为 $h_{\text{low}} \times d_{\text{low}}$。

- **输入**：当前时刻 $t$，预测步数 $h_{\text{low}} \in [1, 10]$
- **输出**：预测序列 $\hat{\mathbf{Y}}_{\text{low}} \in \mathbb{R}^{h_{\text{low}} \times d_{\text{low}}}$
- **约束**：$1 \leq h_{\text{low}} \leq 10$；预测误差方差随步数单调递增

> **白话翻译**：预测窗口就是"能看多远"——底层只看1到10步（1到10秒），因为太远了看不准。预测误差随步数增长，10步之外就不靠谱了。底层预测输出一个序列：第1秒预测什么、第2秒预测什么、……、第10秒预测什么。

---

### 第四部分：底层特征工程

#### 2.4 底层特征工程定义（A41.4）

底层特征工程将原始预测目标 $\mathbf{y}_{\text{low}}$ 转化为更适合预测模型处理的特征向量 $\mathbf{x}_{\text{low}}$。四种特征变换定义如下：

**滑动均值（rolling_mean）**：

$$\text{rolling\_mean}(\mathbf{y}, w) = \frac{1}{w}\sum_{i=0}^{w-1} \mathbf{y}(t-i) \tag{A41.4-mean}$$

其中 $w$ 为窗口大小（典型值 $w = 5$）。滑动均值平滑短期波动，提取局部趋势。

**滑动标准差（rolling_std）**：

$$\text{rolling\_std}(\mathbf{y}, w) = \sqrt{\frac{1}{w}\sum_{i=0}^{w-1}\left(\mathbf{y}(t-i) - \text{rolling\_mean}(\mathbf{y}, w)\right)^2} \tag{A41.4-std}$$

滑动标准差度量局部波动性——波动性突增往往预示异常。

**差分（diff）**：

$$\text{diff}(\mathbf{y}) = \mathbf{y}(t) - \mathbf{y}(t-1) \tag{A41.4-diff}$$

差分提取一阶变化量，将非平稳序列转化为近似平稳序列，是ARIMA模型（阶段42，A42.1）的输入要求。

**变化率（rate_of_change）**：

$$\text{roc}(\mathbf{y}) = \frac{\mathbf{y}(t) - \mathbf{y}(t-1)}{|\mathbf{y}(t-1)| + \epsilon} \tag{A41.4-roc}$$

其中 $\epsilon = 10^{-8}$ 为数值稳定项。变化率度量相对变化幅度，消除量纲影响。

**特征向量的组装**：

对每个原始指标 $\mathbf{y}_i$（$i = 1, \ldots, d_{\text{low}}$），生成4个特征：

$$\mathbf{x}_i = [\text{rolling\_mean}(\mathbf{y}_i, w), \text{rolling\_std}(\mathbf{y}_i, w), \text{diff}(\mathbf{y}_i), \text{roc}(\mathbf{y}_i)] \tag{A41.4-feature}$$

完整特征向量为所有指标特征的拼接：

$$\mathbf{x}_{\text{low}} = [\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_{d_{\text{low}}}] \in \mathbb{R}^{4 d_{\text{low}}} \tag{A41.4-assemble}$$

特征维度 $d_{\text{feat}} = 4 d_{\text{low}} = 72$（典型值）。

- **输入**：原始预测目标时间序列 $\{\mathbf{y}_{\text{low}}(t-w+1), \ldots, \mathbf{y}_{\text{low}}(t)\}$
- **输出**：特征向量 $\mathbf{x}_{\text{low}} \in \mathbb{R}^{4 d_{\text{low}}}$
- **约束**：窗口 $w \geq 2$；差分和变化率需要至少2个历史值

> **白话翻译**：特征工程就是"把原始数据变成模型好消化的形式"——滑动均值看趋势、滑动标准差看波动、差分看变化量、变化率看相对变化。每个原始指标变出4个特征，18个指标变成72个特征，信息更丰富，模型更好用。

**特征工程与预测编码的对应**：

在预测编码框架（引用阶段3）中，特征工程对应从原始观测 $o$ 到内部表征 $r^1$ 的编码过程。四种特征变换可以理解为对原始信号的不同"观察角度"：

| 特征变换 | 预测编码对应 | 物理含义 |
|---------|------------|---------|
| rolling_mean | 低通滤波 | 提取慢变化成分（对应高层表征的预测） |
| rolling_std | 波动性编码 | 提取不确定性信号（对应精度矩阵 $\Pi$ 的输入） |
| diff | 预测残差 | 提取新信息（对应误差信号 $e$） |
| roc | 归一化残差 | 提取相对新信息（对应精度加权误差 $\Pi e$） |

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A41.1 | $\mathbf{y}_{\text{low}} = [\mathbf{y}_{\text{sys}}, \mathbf{y}_{\text{log}}, \mathbf{y}_{\text{file}}]$ | 系统指标、日志流、文件事件 | $\mathbf{y}_{\text{low}} \in \mathbb{R}^{d_{\text{low}}}$ | $d_{\text{low}} = d_{\text{sys}} + d_{\text{log}} + d_{\text{file}}$ |
| A41.2 | $f_{\text{sys}} = 1\text{Hz}, f_{\text{log}} = 0.1\text{Hz}, f_{\text{file}} = 1\text{Hz}$ | 各子向量变化频率 | 采集频率和采样间隔 | 满足Nyquist定理 |
| A41.3 | $h_{\text{low}} \in [1, 10]$ | 预测步数 | 预测序列 $\hat{\mathbf{Y}}_{\text{low}} \in \mathbb{R}^{h \times d_{\text{low}}}$ | 误差方差随步数递增 |
| A41.4 | $\mathbf{x}_{\text{low}} = [\text{mean}, \text{std}, \text{diff}, \text{roc}]$ 对每个指标 | 原始时间序列 | $\mathbf{x}_{\text{low}} \in \mathbb{R}^{4 d_{\text{low}}}$ | 窗口 $w \geq 2$ |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np

# 引用阶段40的数据类
from stage_040 import DecisionEngineState


# ============================================================
# 阶段41 枚举：采集频率类型
# ============================================================
class SamplingFrequency(Enum):
    """底层预测目标的采集频率类型（A41.2）。"""
    SYSTEM_1HZ = "1hz"       # 系统负载：1Hz
    LOG_01HZ = "0.1hz"       # 日志变化：0.1Hz
    FILE_1HZ = "1hz"         # 文件活动：1Hz


# ============================================================
# 阶段41 数据类：底层预测目标
# ============================================================
@dataclass(frozen=True)
class LowLevelTarget:
    """底层预测目标的完整描述。

    组合阶段40的DecisionEngineState，在决策引擎状态的基础上增加
    底层预测目标——系统负载、日志变化、文件活动三个子向量，
    以及采集频率、预测窗口、特征工程参数。

    Attributes:
        engine_state: 阶段40的决策引擎状态（组合字段）
        system_load: 系统负载指标向量 y_sys ∈ R^{d_sys}
        log_change: 日志变化指标向量 y_log ∈ R^{d_log}
        file_activity: 文件活动指标向量 y_file ∈ R^{d_file}
        sys_sampling_hz: 系统负载采集频率（默认1Hz）
        log_sampling_hz: 日志变化采集频率（默认0.1Hz）
        file_sampling_hz: 文件活动采集频率（默认1Hz）
        prediction_horizon: 预测窗口 h_low ∈ [1, 10]
        feature_window: 特征工程滑动窗口大小 w
        feature_vector: 特征向量 x_low ∈ R^{4*d_low}

    Invariants:
        - system_load 维度 d_sys >= 1
        - log_change 维度 d_log >= 1
        - file_activity 维度 d_file >= 1
        - sys_sampling_hz > 0
        - log_sampling_hz > 0
        - file_sampling_hz > 0
        - 1 <= prediction_horizon <= 10
        - feature_window >= 2
    """
    engine_state: DecisionEngineState
    system_load: np.ndarray
    log_change: np.ndarray
    file_activity: np.ndarray
    sys_sampling_hz: float = 1.0
    log_sampling_hz: float = 0.1
    file_sampling_hz: float = 1.0
    prediction_horizon: int = 5
    feature_window: int = 5
    feature_vector: Optional[np.ndarray] = None

    def __post_init__(self):
        # 验证系统负载维度
        if self.system_load.ndim != 1 or len(self.system_load) < 1:
            raise ValueError(
                f"系统负载必须为一维非空向量，收到形状 {self.system_load.shape}"
            )
        # 验证日志变化维度
        if self.log_change.ndim != 1 or len(self.log_change) < 1:
            raise ValueError(
                f"日志变化必须为一维非空向量，收到形状 {self.log_change.shape}"
            )
        # 验证文件活动维度
        if self.file_activity.ndim != 1 or len(self.file_activity) < 1:
            raise ValueError(
                f"文件活动必须为一维非空向量，收到形状 {self.file_activity.shape}"
            )
        # 验证采集频率
        if self.sys_sampling_hz <= 0:
            raise ValueError(
                f"系统负载采集频率必须 > 0，收到 {self.sys_sampling_hz}"
            )
        if self.log_sampling_hz <= 0:
            raise ValueError(
                f"日志变化采集频率必须 > 0，收到 {self.log_sampling_hz}"
            )
        if self.file_sampling_hz <= 0:
            raise ValueError(
                f"文件活动采集频率必须 > 0，收到 {self.file_sampling_hz}"
            )
        # 验证预测窗口
        if not (1 <= self.prediction_horizon <= 10):
            raise ValueError(
                f"预测窗口必须在 [1, 10] 范围内，收到 {self.prediction_horizon}"
            )
        # 验证特征窗口
        if self.feature_window < 2:
            raise ValueError(
                f"特征窗口必须 >= 2，收到 {self.feature_window}"
            )

    @property
    def target_vector(self) -> np.ndarray:
        """完整底层预测目标向量 y_low（A41.1）。"""
        return np.concatenate([self.system_load, self.log_change, self.file_activity])

    @property
    def target_dim(self) -> int:
        """底层预测目标维度 d_low（A41.1-dim）。"""
        return len(self.system_load) + len(self.log_change) + len(self.file_activity)

    @property
    def feature_dim(self) -> int:
        """特征向量维度 4*d_low（A41.4-assemble）。"""
        return 4 * self.target_dim

    @property
    def base_interval(self) -> float:
        """基准采样间隔 Δt_base = 1/max(f)（A41.2-interval）。"""
        return 1.0 / max(self.sys_sampling_hz, self.file_sampling_hz)

    @property
    def log_hold_steps(self) -> int:
        """日志零阶保持步数 k_log（A41.2-hold）。"""
        return max(1, int(round(self.sys_sampling_hz / self.log_sampling_hz)))


# ============================================================
# 阶段41 核心接口：底层特征工程
# ============================================================
class LowLevelFeatureEngine:
    """阶段41的核心计算接口。

    实现底层特征工程的四种变换：滑动均值、滑动标准差、
    差分、变化率，以及特征向量的组装。

    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 滑动窗口 w >= 2
        - 差分和变化率需要至少2个历史值
        - 变化率分母加 epsilon 防止除零
    """

    def rolling_mean(
        self,
        series: np.ndarray,
        window: int
    ) -> np.ndarray:
        """计算滑动均值（A41.4-mean）。

        Args:
            series: 时间序列，形状 (T, d)
            window: 滑动窗口大小 w >= 2

        Returns:
            滑动均值序列，形状 (T, d)

        Raises:
            ValueError: 当 window < 2 时

        Examples:
            >>> engine = LowLevelFeatureEngine()
            >>> s = np.array([[1.0], [2.0], [3.0], [4.0]])
            >>> engine.rolling_mean(s, window=2).tolist()
            [[1.0], [1.5], [2.5], [3.5]]
        """
        if window < 2:
            raise ValueError(f"窗口大小必须 >= 2，收到 {window}")
        series = np.asarray(series, dtype=float)
        # 使用累积和实现高效滑动均值
        if series.ndim == 1:
            series = series.reshape(-1, 1)
        T, d = series.shape
        result = np.empty_like(series)
        cumsum = np.cumsum(series, axis=0)
        result[:window - 1] = cumsum[:window - 1] / np.arange(1, window)[:, None]
        result[window - 1:] = (cumsum[window - 1:] - cumsum[:T - window + 1]) / window
        return result

    def rolling_std(
        self,
        series: np.ndarray,
        window: int
    ) -> np.ndarray:
        """计算滑动标准差（A41.4-std）。

        Args:
            series: 时间序列，形状 (T, d)
            window: 滑动窗口大小 w >= 2

        Returns:
            滑动标准差序列，形状 (T, d)

        Raises:
            ValueError: 当 window < 2 时

        Examples:
            >>> engine = LowLevelFeatureEngine()
            >>> s = np.array([[1.0], [2.0], [3.0], [4.0]])
            >>> stds = engine.rolling_std(s, window=2)
            >>> stds.shape
            (4, 1)
        """
        if window < 2:
            raise ValueError(f"窗口大小必须 >= 2，收到 {window}")
        series = np.asarray(series, dtype=float)
        if series.ndim == 1:
            series = series.reshape(-1, 1)
        T, d = series.shape
        result = np.empty_like(series)
        for t in range(T):
            start = max(0, t - window + 1)
            segment = series[start:t + 1]
            result[t] = np.std(segment, axis=0, ddof=0)
        return result

    def diff(
        self,
        series: np.ndarray
    ) -> np.ndarray:
        """计算差分（A41.4-diff）。

        Args:
            series: 时间序列，形状 (T, d)

        Returns:
            差分序列，形状 (T-1, d)

        Raises:
            ValueError: 当序列长度 < 2 时

        Examples:
            >>> engine = LowLevelFeatureEngine()
            >>> s = np.array([[1.0], [3.0], [6.0]])
            >>> engine.diff(s).tolist()
            [[2.0], [3.0]]
        """
        series = np.asarray(series, dtype=float)
        if series.shape[0] < 2:
            raise ValueError(f"序列长度必须 >= 2，收到 {series.shape[0]}")
        return np.diff(series, axis=0)

    def rate_of_change(
        self,
        series: np.ndarray,
        epsilon: float = 1e-8
    ) -> np.ndarray:
        """计算变化率（A41.4-roc）。

        Args:
            series: 时间序列，形状 (T, d)
            epsilon: 数值稳定项 > 0

        Returns:
            变化率序列，形状 (T-1, d)

        Raises:
            ValueError: 当序列长度 < 2 或 epsilon <= 0 时

        Examples:
            >>> engine = LowLevelFeatureEngine()
            >>> s = np.array([[1.0], [3.0], [6.0]])
            >>> engine.rate_of_change(s, epsilon=1e-8).tolist()
            [[2.0], [1.0]]
        """
        series = np.asarray(series, dtype=float)
        if series.shape[0] < 2:
            raise ValueError(f"序列长度必须 >= 2，收到 {series.shape[0]}")
        if epsilon <= 0:
            raise ValueError(f"epsilon 必须 > 0，收到 {epsilon}")
        diff = np.diff(series, axis=0)
        denom = np.abs(series[:-1]) + epsilon
        return diff / denom

    def extract_features(
        self,
        target: LowLevelTarget,
        history: np.ndarray
    ) -> np.ndarray:
        """提取完整特征向量（A41.4-assemble）。

        对每个原始指标生成4个特征（均值、标准差、差分、变化率），
        取最后有效值组装为完整特征向量。

        Args:
            target: 底层预测目标
            history: 历史时间序列，形状 (T, d_low)

        Returns:
            特征向量 x_low ∈ R^{4*d_low}

        Raises:
            ValueError: 当历史长度不足时

        Examples:
            >>> from dataclasses import dataclass
            >>> @dataclass
            ... class MockTarget:
            ...     feature_window = 3
            >>> engine = LowLevelFeatureEngine()
            >>> h = np.array([[1.0], [2.0], [3.0], [4.0]])
            >>> feat = engine.extract_features(MockTarget(), h)
            >>> feat.shape[0]
            4
        """
        history = np.asarray(history, dtype=float)
        if history.ndim == 1:
            history = history.reshape(-1, 1)
        T, d = history.shape
        w = target.feature_window
        if T < w:
            raise ValueError(f"历史长度 {T} 不足窗口 {w}")
        mean_feat = self.rolling_mean(history, w)[-1]
        std_feat = self.rolling_std(history, w)[-1]
        diff_feat = self.diff(history)[-1] if T > 1 else np.zeros(d)
        roc_feat = self.rate_of_change(history)[-1] if T > 1 else np.zeros(d)
        return np.concatenate([mean_feat, std_feat, diff_feat, roc_feat])
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `DecisionEngineState` | ← 阶段40 | 作为底层预测目标的决策引擎状态基础（组合字段） |
| **输入** | A11.1（层级生成模型联合分布） | ← 阶段11 | 底层预测目标对应第一层表征的观测预测 |
| **输入** | A11.4（深度-抽象度关系） | ← 阶段11 | 底层时间常数最小，需要高频采样的理论依据 |
| **输入** | A3.1（预测误差信号） | ← 阶段3 | 底层预测误差的定义和传播起点 |
| **输入** | A3.2（精度加权误差能量） | ← 阶段3 | 底层精度矩阵决定误差传播权重 |
| **输出** | `LowLevelTarget` | → 阶段42 | 底层预测目标数据，包含DecisionEngineState组合 |
| **输出** | A41.1（底层预测目标） | → 阶段42/43 | 底层模型算法的输入目标 |
| **输出** | A41.2（采集频率） | → 阶段42/50 | 多频率对齐和时间尺度协议的依据 |
| **输出** | A41.3（预测窗口） | → 阶段42/43 | 预测步数和预测序列的格式定义 |
| **输出** | A41.4（底层特征工程） | → 阶段42 | 底层模型算法的特征输入 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| pandas | 时间序列特征工程 | `rolling()`直接支撑滑动均值（A41.4-mean）和滑动标准差（A41.4-std）的计算，`diff()`支撑差分（A41.4-diff），`pct_change()`支撑变化率（A41.4-roc） |
| NumPy | 向量化计算 | 特征向量的组装（A41.4-assemble）和多频率对齐（A41.2-hold） |
| JAX | 自动微分 | 特征工程的可微实现，支撑端到端梯度传播 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V41.1 | 目标向量组装 | `target_vector == [system_load, log_change, file_activity]` | 拼接验证 |
| V41.2 | 目标维度 | `target_dim == d_sys + d_log + d_file` | 维度断言 |
| V41.3 | 采集频率正值 | 所有采样频率 > 0 | 边界测试 |
| V41.4 | 预测窗口范围 | `1 <= prediction_horizon <= 10` | 边界测试 |
| V41.5 | 特征窗口最小值 | `feature_window >= 2` | 边界测试 |
| V41.6 | 滑动均值正确性 | 与手动计算结果在容差 $10^{-10}$ 内一致 | 随机序列测试 |
| V41.7 | 滑动标准差正确性 | 与手动计算结果在容差 $10^{-10}$ 内一致 | 随机序列测试 |
| V41.8 | 差分正确性 | `diff(y)[t] == y[t] - y[t-1]` | 逐元素验证 |
| V41.9 | 变化率正确性 | `roc(y)[t] == (y[t] - y[t-1]) / (|y[t-1]| + eps)` | 逐元素验证 |
| V41.10 | 变化率除零保护 | `y[t-1] = 0` 时不抛异常，返回有限值 | 零值测试 |
| V41.11 | 特征向量维度 | `feature_dim == 4 * target_dim` | 维度断言 |
| V41.12 | 日志零阶保持 | 非采样步使用最近采样值 | 步进验证 |
| V41.13 | LowLevelTarget组合 | `target.engine_state` 类型为 `DecisionEngineState` | 类型检查 |
| V41.14 | 非法输入拒绝 | 维度为0、频率≤0、窗口<2等抛出 `ValueError` | 异常测试 |
| V41.15 | Nyquist采样满足 | 所有采集频率 >= 2倍信号最高频率 | 频谱分析验证 |

---

> **阶段41完** | 阶段41建立了底层高频微预测的目标体系：预测目标包含系统负载、日志变化、文件活动三个子向量（A41.1），采集频率分别为1Hz/0.1Hz/1Hz（A41.2），预测窗口1-10步（A41.3），特征工程包含滑动均值、滑动标准差、差分、变化率四种变换（A41.4），封装为LowLevelTarget数据类组合DecisionEngineState。下一步：阶段42将从本阶段的底层预测目标出发，设计底层模型的算法——在线ARIMA和卡尔曼滤波。
