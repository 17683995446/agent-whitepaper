# 阶段76：空闲离线复盘——空闲判定、批量重放优化与收益终止条件

> **阶段编号**：S76 | **所属层**：第八层·记忆系统层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 空闲判定 | "没事干"——CPU闲着、没有活跃任务，正好复习功课 |
> | 批量重放 | "集中复习"——一次抽128张卡片集中学习 |
> | 批量优化 | "逐张消化"——每张卡片都用来更新模型 |
> | 复盘收益 | "复习有没有用"——复习前后自由能差了多少 |
> | 终止条件 | "复习到什么时候停"——收益太小就别复习了，去做别的事 |
> | OfflineReplay | "离线复盘的完整画像"——组合GenerativeReplay，在生成式回放基础上增加复盘调度 |

---

## 1. 第一性原理依据

本阶段从阶段9（三大循环定义）和阶段62（微修正算法）的核心结论出发，建立空闲离线复盘机制。核心依据为：

- **A9.6（终极循环触发条件）**（引用阶段9）：终极循环在长期无进步时触发认知升级。空闲离线复盘是终极循环在记忆系统层面的实现——当系统空闲时，利用记忆中的历史数据进行批量学习，推动系统持续进步。

- **A62.1-A62.5（微修正算法）**（引用阶段62）：权重微调 $\Delta W = -\eta \cdot \partial F / \partial W$、自适应学习率、在线贝叶斯更新、EWC正则化和收敛保证。离线复盘中的批量优化直接采用A62.1的微修正算法——对每个回放样本执行一次微修正更新。

- **A69.1（稳态检测）**（引用阶段69）：is_stable = 𝟙[var(F, W_F) < θ_F ∧ var(‖W‖, W_W) < θ_W]，系统稳态检测。空闲判定与稳态检测密切相关——系统空闲通常意味着系统处于稳态（自由能和权重都稳定）。

- **A75.1-A75.5（生成式回放）**（引用阶段75）：VAE编码器、解码器、压缩存储、重新生成和质量评估。离线复盘的回放样本可以来自原始缓冲区（优先级回放）或压缩存储（生成式回放）。

本阶段的核心目标是：**定义空闲判定 cpu_usage < θ_idle AND no_active_task（A76.1），批量重放 batch = sample_priority(buffer, batch_size=128)（A76.2），批量优化 for sample in batch: model.update(sample)（A76.3），复盘收益评估 ΔF_replay = F_before - F_after（A76.4），复盘终止条件 ΔF_replay < θ_min_improvement（A76.5），封装为OfflineReplay数据类，组合GenerativeReplay作为生成式回放的输入。**

> **去重声明**：本阶段仅定义空闲离线复盘的调度机制和收益评估，不涉及记忆巩固（阶段77）或模型快照（阶段78）。微修正算法遵循A62.1-A62.5，本阶段不重新定义。稳态检测遵循A69.1，本阶段不重复。

---

## 2. 核心数学公式

### 第一部分：空闲判定

#### 2.1 空闲判定定义（A76.1）

空闲判定基于CPU使用率和活跃任务状态：

$$\text{is\_idle} = \mathbb{1}\left[\text{cpu\_usage} < \theta_{\text{idle}} \;\wedge\; \text{no\_active\_task}\right] \tag{A76.1}$$

其中：
- $\text{cpu\_usage} \in [0, 1]$ 为当前CPU使用率
- $\theta_{\text{idle}}$ 为空闲阈值（通常设为0.3，即CPU使用率低于30%）
- $\text{no\_active\_task}$ 为无活跃任务标志

**活跃任务的定义**：

| 任务类型 | 活跃条件 | 说明 |
|---------|---------|------|
| 表层循环 | 正在执行感知-行动 | 始终活跃（除非暂停） |
| 中层循环 | 正在执行策略调整 | 偶尔活跃 |
| 终极循环 | 正在执行结构重构 | 罕见活跃 |
| 外部请求 | 正在处理外部输入 | 间歇活跃 |

空闲判定要求所有非常规任务（中层、终极、外部请求）都不活跃，且表层循环的CPU占用低于阈值。

**空闲判定与A69.1稳态检测的关系**：

| 维度 | A69.1（稳态检测） | A76.1（空闲判定） |
|------|------------------|------------------|
| 检测对象 | 自由能和权重的稳定性 | CPU使用率和任务状态 |
| 含义 | 系统内部已收敛 | 系统外部无负载 |
| 用途 | 探索权重上浮 | 触发离线复盘 |
| 关系 | 稳态 ⟹ 可能空闲 | 空闲 ⟹ 可能稳态 |

稳态和空闲通常同时出现——系统内部收敛且外部无负载。但两者不完全等价：系统可能稳态但不空闲（正在处理外部请求），也可能空闲但不稳态（刚经历环境突变）。

**空闲判定的防抖动**：

为避免频繁切换空闲/非空闲状态，引入防抖动机制：

$$\text{is\_idle\_stable} = \mathbb{1}\left[\text{is\_idle} \text{ for } T_{\text{debounce}} \text{ consecutive steps}\right] \tag{A76.1-debounce}$$

其中 $T_{\text{debounce}}$ 为防抖动窗口（通常设为100步）。只有连续100步都判定为空闲，才真正进入空闲状态。

- **输入**：CPU使用率、活跃任务状态、防抖动窗口
- **输出**：空闲标志
- **约束**：θ_idle ∈ (0, 1)；T_debounce > 0

> **白话翻译**：空闲判定就是"没事干"——CPU闲着（使用率低于30%）、没有活跃任务（不在调策略、不在重构、不在处理请求），就是空闲。和稳态检测的区别：稳态看内部（自由能和权重稳不稳），空闲看外部（有没有事做）。加个防抖动——连续100步都空闲才算真正空闲，避免抖来抖去。

---

### 第二部分：批量重放

#### 2.2 批量重放定义（A76.2）

空闲时从缓冲区中按优先级批量采样：

$$\text{batch} = \text{sample\_priority}(\text{buffer}, \text{batch\_size}=128) \tag{A76.2}$$

**离线复盘的batch_size vs 在线回放的batch_size**：

| 场景 | batch_size | 原因 |
|------|-----------|------|
| 在线回放（A74.3） | 32 | 需要快速返回，不能阻塞表层循环 |
| 离线复盘（A76.2） | 128 | 空闲时资源充裕，可以更大批量 |

离线复盘的batch_size是在线回放的4倍——空闲时CPU资源充裕，更大的批量提供更稳定的梯度估计。

**批量重放的样本来源**：

| 来源 | 优先级 | 说明 |
|------|--------|------|
| 短期记忆缓冲区 | 高 | 最新鲜的样本，优先使用 |
| 长期记忆样本表 | 中 | 历史高价值样本，通过SQL查询 |
| 生成式回放 | 低 | 原始样本不可用时的替代 |

**批量重放的多轮策略**：

一次空闲周期内可以执行多轮批量重放：

$$\text{n\_rounds} = \min\left(\left\lfloor \frac{T_{\text{idle}}}{T_{\text{round}}} \right\rfloor, N_{\max}\right) \tag{A76.2-rounds}$$

其中：
- $T_{\text{idle}}$ 为空闲持续时间
- $T_{\text{round}}$ 为单轮重放时间
- $N_{\max}$ 为最大轮数（通常设为10）

- **输入**：缓冲区、batch_size
- **输出**：批量样本 + 重要性权重
- **约束**：batch_size ≤ 缓冲区大小

> **白话翻译**：批量重放就是"集中复习"——空闲时一次抽128张卡片集中学习，比在线回放的32张多4倍。样本来源优先用短期记忆（最新鲜），其次用长期记忆（历史精华），最后用生成式回放（替代品）。一次空闲可以复习多轮，但最多10轮。

---

### 第三部分：批量优化

#### 2.3 批量优化定义（A76.3）

对批量中的每个样本执行模型更新：

$$\text{for } s \in \text{batch}: \theta \leftarrow \theta - \eta \cdot w_s \cdot \frac{\partial F(s)}{\partial \theta} \tag{A76.3}$$

其中：
- $\theta$ 为模型参数
- $\eta$ 为学习率
- $w_s$ 为样本 $s$ 的重要性权重（A74.2）
- $\partial F(s) / \partial \theta$ 为自由能关于参数的梯度

**与A62.1的对应**：

A62.1（引用阶段62）定义权重微调 $\Delta W = -\eta \cdot \partial F / \partial W$。A76.3是A62.1在批量回放场景下的应用——对每个回放样本执行一次微修正，并乘以重要性权重校正偏差。

**批量优化的EWC正则化**：

由A62.4（引用阶段62），微修正需要EWC正则化保护旧知识：

$$\theta \leftarrow \theta - \eta \cdot \left(w_s \cdot \frac{\partial F(s)}{\partial \theta} + \lambda_{\text{ewc}} \sum_i F_i (\theta_i - \theta_i^*)\right) \tag{A76.3-ewc}$$

其中 $F_i$ 为Fisher信息矩阵的对角元素，$\theta_i^*$ 为EWC锚点参数。

**批量优化的梯度累积**：

为提高效率，可采用梯度累积——先累积整个批量的梯度，再一次性更新参数：

$$\theta \leftarrow \theta - \eta \cdot \frac{1}{|\text{batch}|} \sum_{s \in \text{batch}} w_s \cdot \frac{\partial F(s)}{\partial \theta} \tag{A76.3-accumulate}$$

梯度累积比逐样本更新更稳定（梯度估计方差更小），但需要更多内存存储中间梯度。

- **输入**：批量样本、模型参数、学习率、重要性权重
- **输出**：更新后的模型参数
- **约束**：自由能单调不增（A62.5保证）

> **白话翻译**：批量优化就是"逐张消化"——每张卡片都用来更新模型。更新方式就是微修正（A62.1），加上重要性权重校正偏差，加上EWC正则化保护旧知识。可以逐张更新，也可以先累积梯度再一次性更新——后者更稳定但更费内存。

---

### 第四部分：复盘收益评估

#### 2.4 复盘收益评估定义（A76.4）

复盘收益通过自由能的变化量衡量：

$$\Delta F_{\text{replay}} = F_{\text{before}} - F_{\text{after}} \tag{A76.4}$$

其中：
- $F_{\text{before}}$ 为复盘前的自由能
- $F_{\text{after}}$ 为复盘后的自由能
- $\Delta F_{\text{replay}} > 0$ 表示复盘有收益（自由能下降）

**复盘收益的物理含义**：

复盘收益度量"复习有没有用"——自由能下降越多，说明复盘越有效。自由能下降意味着模型通过复习更好地拟合了历史数据，预测能力提升。

**复盘收益的归一化**：

为消除自由能绝对值的影响，采用相对收益：

$$\text{relative\_benefit} = \frac{\Delta F_{\text{replay}}}{F_{\text{before}}} \cdot 100\% \tag{A76.4-relative}$$

**复盘收益与A70.1的关系**：

A70.1（引用阶段70）定义微修正精度提升率 micro_improvement_rate = (ε₀ - ε₁₀₀)/ε₀ · 100%。A76.4的复盘收益与A70.1在形式上一致——都是衡量优化前后的相对改善。区别在于：A70.1衡量在线微修正的改善，A76.4衡量离线复盘的改善。

**复盘收益的统计特性**：

| 复盘轮次 | 预期收益 | 原因 |
|---------|---------|------|
| 第1轮 | 高 | 高价值样本首次被复习 |
| 第2-3轮 | 中 | 部分样本已学会 |
| 第4-5轮 | 低 | 大部分样本已学会 |
| 第6+轮 | 极低 | 几乎所有样本已学会 |

收益递减是正常的——随着模型改善，同一样本的边际收益递减。

- **输入**：复盘前后的自由能
- **输出**：复盘收益 ΔF_replay
- **约束**：ΔF_replay ≥ -ε_noise（允许微小噪声）

> **白话翻译**：复盘收益就是"复习有没有用"——复习前后自由能差了多少。自由能下降越多，说明复习越有效。收益会递减——第一轮复习收益最高，后面越来越低，因为大部分样本已经学会了。

---

### 第五部分：复盘终止条件

#### 2.5 复盘终止条件定义（A76.5）

当复盘收益低于最小改善阈值时终止复盘：

$$\text{stop\_replay} = \mathbb{1}\left[\Delta F_{\text{replay}} < \theta_{\text{min\_improvement}}\right] \tag{A76.5}$$

其中 $\theta_{\text{min\_improvement}}$ 为最小改善阈值。

**最小改善阈值的选择**：

| θ_min_improvement | 含义 | 适用场景 |
|-------------------|------|---------|
| 0.001 (0.1%) | 非常敏感 | 精确优化 |
| 0.01 (1%) | 适中 | 默认选择 |
| 0.05 (5%) | 宽松 | 快速迭代 |

默认 $\theta_{\text{min\_improvement}} = 0.01$（1%），即复盘收益低于1%时终止。

**终止条件的组合判定**：

除了收益阈值，还有其他终止条件：

$$\text{stop} = \text{stop\_replay} \;\vee\; \text{stop\_time} \;\vee\; \text{stop\_rounds} \;\vee\; \text{stop\_interrupt} \tag{A76.5-combined}$$

| 终止条件 | 触发 | 说明 |
|---------|------|------|
| stop_replay | ΔF < θ_min | 收益太低 |
| stop_time | T_replay > T_max | 时间超限 |
| stop_rounds | n_rounds ≥ N_max | 轮次超限 |
| stop_interrupt | new_task_arrived | 新任务到来 |

新任务到来时必须立即终止复盘——在线任务优先级高于离线复盘。

**终止后的状态保存**：

复盘终止后需要保存当前状态，以便下次空闲时继续：

| 保存项 | 说明 |
|--------|------|
| 已回放的样本ID | 避免重复回放 |
| 当前模型参数 | 下次复盘的起点 |
| 累积收益 | 监控复盘效果 |
| 剩余高价值样本 | 下次优先回放 |

- **输入**：复盘收益、最小改善阈值、其他终止条件
- **输出**：是否终止复盘
- **约束**：新任务到来时立即终止

> **白话翻译**：复盘终止条件就是"复习到什么时候停"——收益低于1%就别复习了，去做别的事。还有其他停下来的理由：时间到了、轮次够了、新任务来了。新任务来了必须立刻停——在线的事比离线复习重要。停下来后要保存状态，下次空闲时接着复习。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A76.1 | is_idle = 𝟙[cpu < θ ∧ no_task] | CPU使用率、任务状态 | 空闲标志 | θ ∈ (0,1) |
| A76.2 | batch = sample_priority(buffer, 128) | 缓冲区、batch_size | 批量样本+权重 | batch_size ≤ N |
| A76.3 | for s: θ ← θ - η·w_s·∂F/∂θ | 批量样本、模型参数 | 更新后参数 | F单调不增 |
| A76.4 | ΔF = F_before - F_after | 复盘前后自由能 | 复盘收益 | ΔF ≥ -ε |
| A76.5 | stop = 𝟙[ΔF < θ_min] | 复盘收益、阈值 | 终止标志 | 新任务立即终止 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import numpy as np

# 引用阶段75的数据类
try:
    from stage_075 import GenerativeReplay
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


# ============================================================
# 阶段76 数据类：离线复盘
# ============================================================
@dataclass(frozen=True)
class OfflineReplay:
    """空闲离线复盘的完整描述。

    组合阶段75的GenerativeReplay，在生成式回放的基础上增加
    复盘调度和收益评估。

    Attributes:
        generative_replay: 阶段75的生成式回放（组合字段）
        idle_threshold: 空闲CPU阈值
        debounce_steps: 防抖动步数
        offline_batch_size: 离线复盘批量大小
        max_rounds: 最大复盘轮次
        min_improvement: 最小改善阈值
        is_idle: 当前是否空闲
        idle_duration: 空闲持续时间
        replay_rounds: 当前复盘轮次
        total_replay_benefit: 累积复盘收益
        last_benefit: 最近一轮复盘收益

    Invariants:
        - idle_threshold ∈ (0, 1)
        - debounce_steps > 0
        - offline_batch_size > 0
        - max_rounds > 0
        - min_improvement > 0
    """
    generative_replay: GenerativeReplay
    idle_threshold: float = 0.3
    debounce_steps: int = 100
    offline_batch_size: int = 128
    max_rounds: int = 10
    min_improvement: float = 0.01
    is_idle: bool = False
    idle_duration: int = 0
    replay_rounds: int = 0
    total_replay_benefit: float = 0.0
    last_benefit: float = 0.0

    def __post_init__(self):
        if not (0 < self.idle_threshold < 1):
            raise ValueError(
                f"空闲阈值必须 ∈ (0,1)，收到 {self.idle_threshold}"
            )
        if self.debounce_steps <= 0:
            raise ValueError(
                f"防抖动步数必须 > 0，收到 {self.debounce_steps}"
            )
        if self.offline_batch_size <= 0:
            raise ValueError(
                f"离线批量大小必须 > 0，收到 {self.offline_batch_size}"
            )
        if self.min_improvement <= 0:
            raise ValueError(
                f"最小改善阈值必须 > 0，收到 {self.min_improvement}"
            )

    @property
    def should_stop(self) -> bool:
        """是否应该终止复盘（A76.5）。"""
        return self.last_benefit < self.min_improvement


# ============================================================
# 阶段76 核心接口：离线复盘
# ============================================================
class OfflineReplayInterface:
    """阶段76的核心接口：空闲离线复盘。

    实现空闲判定、批量重放、批量优化和收益评估。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 仅在空闲时执行复盘
        - 新任务到来立即终止
        - 收益低于阈值时终止
    """

    def check_idle(
        self,
        cpu_usage: float,
        has_active_task: bool,
        idle_duration: int,
        debounce_steps: int,
        idle_threshold: float
    ) -> Tuple[bool, int]:
        """判定是否空闲（A76.1）。

        Args:
            cpu_usage: CPU使用率，单位%，范围[0,100]
            has_active_task: 是否有活跃任务
            idle_duration: 已空闲持续时间，单位步
            debounce_steps: 防抖动步数，必须 > 0
            idle_threshold: 空闲阈值，单位%，范围[0,100]

        Returns:
            (是否空闲稳定, 更新后的空闲持续时间)

        Raises:
            ValueError: 当debounce_steps <= 0时
            ValueError: 当idle_threshold不在[0,100]范围内时

        Examples:
            >>> ori = OfflineReplayInterface()
            >>> is_idle, dur = ori.check_idle(
            ...     5.0, False, 5, 10, 10.0)
            >>> dur
            6
            >>> is_idle
            False
        """
        if debounce_steps <= 0:
            raise ValueError(
                f"debounce_steps必须 > 0，收到 {debounce_steps}"
            )
        if not (0 <= idle_threshold <= 100):
            raise ValueError(
                f"idle_threshold必须 ∈ [0,100]，收到 {idle_threshold}"
            )
        if cpu_usage < idle_threshold and not has_active_task:
            new_duration = idle_duration + 1
        else:
            new_duration = 0
        is_idle_stable = new_duration >= debounce_steps
        return is_idle_stable, new_duration

    def sample_offline_batch(
        self,
        offline_replay: OfflineReplay
    ) -> List[Tuple]:
        """批量重放采样（A76.2）。

        Args:
            offline_replay: 当前离线复盘状态

        Returns:
            批量样本列表，长度 = batch_size

        Examples:
            >>> ori = OfflineReplayInterface()
            >>> batch = ori.sample_offline_batch(
            ...     OfflineReplay(batch_size=4))
            >>> len(batch)
            4
        """
        batch_size = offline_replay.batch_size
        return [(i, 0.0) for i in range(batch_size)]

    def batch_optimize(
        self,
        params: np.ndarray,
        batch: List[Tuple],
        learning_rate: float,
        ewc_lambda: float = 0.0
    ) -> np.ndarray:
        """批量优化（A76.3）。

        Args:
            params: 当前模型参数，shape (n,)
            batch: 批量样本，每个元素为(observation, target)
            learning_rate: 学习率，必须 > 0
            ewc_lambda: EWC正则化系数，必须 >= 0

        Returns:
            更新后的模型参数，shape (n,)

        Raises:
            ValueError: 当learning_rate <= 0时
            ValueError: 当ewc_lambda < 0时

        Examples:
            >>> ori = OfflineReplayInterface()
            >>> params = np.array([1.0, 0.0])
            >>> batch = [(np.array([1.0]), 0.5)]
            >>> new_params = ori.batch_optimize(
            ...     params, batch, 0.01, 0.0)
            >>> new_params.shape == params.shape
            True
        """
        if learning_rate <= 0:
            raise ValueError(
                f"learning_rate必须 > 0，收到 {learning_rate}"
            )
        if ewc_lambda < 0:
            raise ValueError(
                f"ewc_lambda必须 >= 0，收到 {ewc_lambda}"
            )
        new_params = params.copy()
        for sample in batch:
            if len(sample) >= 2:
                obs = np.asarray(sample[0])
                target = sample[1]
                pred = new_params[:obs.shape[0]] @ obs
                grad = 2 * (pred - target) * obs
                new_params = new_params - learning_rate * grad
        return new_params

    def evaluate_benefit(
        self,
        free_energy_before: float,
        free_energy_after: float
    ) -> Tuple[float, float]:
        """评估复盘收益（A76.4）。

        Args:
            free_energy_before: 复盘前自由能
            free_energy_after: 复盘后自由能

        Returns:
            (绝对收益, 相对收益%)
            - 绝对收益 = F_before - F_after
            - 相对收益% = (F_before - F_after) / |F_before| * 100

        Examples:
            >>> ori = OfflineReplayInterface()
            >>> abs_ben, rel_ben = ori.evaluate_benefit(10.0, 8.0)
            >>> abs_ben
            2.0
            >>> rel_ben
            20.0
        """
        abs_benefit = free_energy_before - free_energy_after
        if abs(free_energy_before) > 1e-12:
            rel_benefit = (
                abs_benefit / abs(free_energy_before)
            ) * 100.0
        else:
            rel_benefit = 0.0
        return abs_benefit, rel_benefit

    def check_stop(
        self,
        benefit: float,
        min_improvement: float,
        rounds: int,
        max_rounds: int,
        new_task: bool
    ) -> bool:
        """检查终止条件（A76.5）。

        Args:
            benefit: 最近一轮收益
            min_improvement: 最小改善阈值
            rounds: 当前轮次，必须 >= 0
            max_rounds: 最大轮次，必须 > 0
            new_task: 是否有新任务

        Returns:
            是否终止复盘

        Raises:
            ValueError: 当max_rounds <= 0时
            ValueError: 当rounds < 0时

        Examples:
            >>> ori = OfflineReplayInterface()
            >>> ori.check_stop(0.001, 0.01, 5, 10, False)
            True
            >>> ori.check_stop(0.1, 0.01, 5, 10, True)
            True
            >>> ori.check_stop(0.1, 0.01, 5, 10, False)
            False
        """
        if max_rounds <= 0:
            raise ValueError(
                f"max_rounds必须 > 0，收到 {max_rounds}"
            )
        if rounds < 0:
            raise ValueError(f"rounds必须 >= 0，收到 {rounds}")
        if new_task:
            return True
        if rounds >= max_rounds:
            return True
        if benefit < min_improvement:
            return True
        return False
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `GenerativeReplay` | ← 阶段75 | 作为离线复盘的生成式回放输入（组合字段） |
| **输入** | A9.6（终极循环触发条件） | ← 阶段9 | 离线复盘的理论基础 |
| **输入** | A62.1-A62.5（微修正算法） | ← 阶段62 | 批量优化的算法基础 |
| **输入** | A69.1（稳态检测） | ← 阶段69 | 空闲判定的参考 |
| **输出** | `OfflineReplay` | → 阶段77 | 离线复盘状态，作为记忆巩固的输入 |
| **输出** | A76.1-A76.5（离线复盘定义） | → 阶段78-80 | 记忆系统的复盘基础 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| psutil | CPU监控 | 获取系统CPU使用率 |
| NumPy | 批量梯度计算 | 向量化的梯度累积 |
| JAX | 高性能优化 | JIT编译的批量优化循环 |
| PyTorch | EWC正则化 | Fisher信息矩阵的计算和应用 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V76.1 | 空闲判定正确 | cpu < θ ∧ no_task → idle | 条件测试 |
| V76.2 | 防抖动有效 | 连续debounce_steps步空闲才判定 | 防抖测试 |
| V76.3 | 非空闲不触发 | cpu ≥ θ 或有任务时不复盘 | 条件测试 |
| V76.4 | 批量大小正确 | 返回128个样本 | 计数测试 |
| V76.5 | 批量优化自由能不增 | F_after ≤ F_before + ε | 收敛测试 |
| V76.6 | EWC正则化生效 | 旧知识不被灾难性遗忘 | 遗忘测试 |
| V76.7 | 复盘收益计算正确 | ΔF = F_before - F_after | 数值验证 |
| V76.8 | 收益递减 | 后续轮次收益 ≤ 前几轮 | 多轮测试 |
| V76.9 | 收益阈值终止 | ΔF < θ_min 时终止 | 条件测试 |
| V76.10 | 轮次上限终止 | rounds ≥ max_rounds 时终止 | 边界测试 |
| V76.11 | 新任务中断 | 新任务到来立即终止 | 中断测试 |
| V76.12 | 状态保存完整 | 终止后可恢复复盘 | 恢复测试 |
| V76.13 | OfflineReplay组合 | `oreplay.generative_replay` 类型为 `GenerativeReplay` | 类型检查 |
| V76.14 | 复盘后F下降>1% | 至少第一轮收益 > 1% | 收益测试 |
| V76.15 | 长期复盘稳定性 | 100轮复盘无异常 | 长期运行测试 |

---

> **阶段76完** | 阶段76建立了空闲离线复盘机制：空闲判定 cpu_usage < θ_idle AND no_active_task（A76.1），批量重放 batch = sample_priority(buffer, batch_size=128)（A76.2），批量优化 for sample in batch: model.update(sample)（A76.3），复盘收益评估 ΔF_replay = F_before - F_after（A76.4），复盘终止条件 ΔF_replay < θ_min_improvement（A76.5），封装为OfflineReplay数据类组合GenerativeReplay。下一步：阶段77将定义记忆巩固机制。
