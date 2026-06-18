# 阶段77：记忆巩固——快慢转移、遗忘曲线与间隔重复调度

> **阶段编号**：S77 | **所属层**：第八层·记忆系统层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 快速→慢速转移 | "短期记忆转长期"——经历够久、够重要，就从临时笔记转进档案柜 |
> | 转移协议 | "归档流程"——从短期记忆转移到长期记忆的具体步骤 |
> | 巩固期重放强化 | "边归档边复习"——转移过程中反复复习，确保记忆牢固 |
> | 遗忘曲线 | "记忆衰减规律"——不复习就忘，忘的速度先快后慢 |
> | 间隔重复 | "什么时候该复习"——在快忘的时候及时复习，用最少时间保持记忆 |
> | MemoryConsolidation | "记忆巩固的完整画像"——组合OfflineReplay，在离线复盘基础上增加巩固调度 |

---

## 1. 第一性原理依据

本阶段从阶段17（快感知-慢学习）和阶段76（空闲离线复盘）的核心结论出发，建立记忆巩固机制。核心依据为：

- **A17.1-A17.5（快感知-慢学习）**（引用阶段17）：感知时间常数 $\tau_{\text{perceive}} = 1/\kappa_\mu$、学习时间常数 $\tau_{\text{learn}} = 1/\kappa_W$、分离比 $\tau_{\text{learn}} / \tau_{\text{perceive}} \geq 100$、快感知算法和慢学习算法。记忆巩固是快感知-慢学习在记忆系统层面的体现——短期记忆对应"快感知"（快速写入、快速遗忘），长期记忆对应"慢学习"（缓慢写入、缓慢遗忘），巩固过程就是从快到慢的转移。

- **A76.1-A76.5（空闲离线复盘）**（引用阶段76）：空闲判定、批量重放、批量优化、收益评估和终止条件。记忆巩固利用离线复盘的批量优化机制——在巩固期间对转移中的样本进行重放强化，确保记忆从短期成功转移到长期。

- **A72.1-A72.5（离线长期记忆）**（引用阶段72）：SQLite样本表、JSON快照、写入策略、读取策略和索引设计。记忆巩固的转移目标即为A72.1的长期记忆样本表——样本从短期缓冲区转移到长期样本表。

- **A5.18（时间尺度分离）**（引用阶段5）：$\tau_A \ll \tau_C$，准确性与复杂性的时间尺度分离。记忆巩固体现了时间尺度分离——短期记忆的时间常数 $\tau_{\text{STM}}$ 远小于长期记忆的时间常数 $\tau_{\text{LTM}}$。

本阶段的核心目标是：**定义快速记忆→慢速记忆转移条件 age(sample) > T_consolidate（A77.1），转移协议 transfer(stm_sample) → ltm_sample（A77.2），巩固期间的重放强化 replay_during_consolidation()（A77.3），遗忘曲线 R(t) = exp(-t/τ_forget)（A77.4），间隔重复调度 next_review = f(R, value, age)（A77.5），封装为MemoryConsolidation数据类，组合OfflineReplay作为离线复盘的输入。**

> **去重声明**：本阶段仅定义记忆巩固的转移机制和间隔重复调度，不涉及模型快照（阶段78）或记忆管理器接口（阶段79）。快感知-慢学习的算法已在阶段17定义，本阶段不重复。离线复盘的批量优化遵循A76.3，本阶段不重新定义。

---

## 2. 核心数学公式

### 第一部分：快速→慢速转移条件

#### 2.1 转移条件定义（A77.1）

样本从短期记忆转移到长期记忆需要满足年龄条件：

$$\text{should\_consolidate}(s) = \mathbb{1}\left[\text{age}(s) > T_{\text{consolidate}}\right] \tag{A77.1}$$

其中：
- $\text{age}(s) = t_{\text{current}} - s.\text{timestamp}$ 为样本的年龄
- $T_{\text{consolidate}}$ 为巩固阈值时间

**巩固阈值的选择**：

| $T_{\text{consolidate}}$ | 对应步数 | 说明 |
|--------------------------|---------|------|
| 100秒 | ~1000步 | 快速巩固 |
| 500秒 | ~5000步 | 中等巩固 |
| 1000秒 | ~10000步 | 慢速巩固（默认） |

默认 $T_{\text{consolidate}} = 1000$ 秒，约等于短期记忆缓冲区（N=10000步）的时间深度。这意味着样本在短期记忆中存在了整个缓冲区周期后，才被考虑转移到长期记忆。

**转移条件的组合判定**：

仅年龄条件不够——还需要考虑样本的价值评分：

$$\text{should\_consolidate}(s) = \mathbb{1}\left[\text{age}(s) > T_{\text{consolidate}} \;\wedge\; \text{value}(s) > \theta_{\text{persist}}\right] \tag{A77.1-combined}$$

其中 $\theta_{\text{persist}}$ 为A72.3（引用阶段72）的持久化阈值。两个条件缺一不可：年龄足够大（不是刚产生的噪声），价值足够高（值得长期保存）。

**与A17.1-A17.5的对应**：

A17.1-A17.5（引用阶段17）定义了快感知-慢学习的时间尺度分离。A77.1的转移条件体现了同样的分离——短期记忆的时间常数 $\tau_{\text{STM}} \approx T_{\text{consolidate}}$，长期记忆的时间常数 $\tau_{\text{LTM}} \to \infty$（不自动遗忘）。转移条件 $T_{\text{consolidate}}$ 对应快-慢分离的临界点。

- **输入**：样本年龄、巩固阈值、价值评分
- **输出**：是否应该巩固
- **约束**：T_consolidate > 0

> **白话翻译**：转移条件就是"短期记忆转长期"——样本在短期记忆中待够久（>1000秒）且价值够高（>持久化阈值），就转移到长期记忆。两个条件缺一不可：刚产生的可能是噪声（需要等一等），低价值的不值得长期保存（需要筛选）。这和快感知-慢学习是一个道理——快的东西要经过时间检验才能变成慢的。

---

### 第二部分：转移协议

#### 2.2 转移协议定义（A77.2）

转移协议定义从短期记忆到长期记忆的具体步骤：

$$\text{transfer}(\text{stm\_sample}) \to \text{ltm\_sample} \tag{A77.2}$$

**转移流程**：

1. **选择待转移样本**：从短期缓冲区中选择满足A77.1条件的样本
2. **计算价值评分**：按A73.1计算value_score
3. **序列化**：将observation和prediction序列化为BLOB
4. **写入长期记忆**：执行A72.3的write_persistent
5. **标记短期记忆**：在短期缓冲区中标记该样本为已转移
6. **更新索引**：确保A72.5的索引包含新写入的样本

**转移的原子性**：

转移操作必须是原子的——要么完整转移，要么不转移：

$$\text{transfer\_atomic}(s) = \text{write\_persistent}(s) \;\wedge\; \text{mark\_transferred}(s) \tag{A77.2-atomic}$$

如果写入长期记忆失败（如磁盘满），短期记忆中的样本不标记为已转移，下次可以重试。

**转移的批量处理**：

为减少I/O开销，转移采用批量处理：

$$\text{transfer\_batch}(\{s_1, ..., s_k\}) = \bigcup_{i=1}^{k} \text{transfer}(s_i) \tag{A77.2-batch}$$

批量转移使用SQLite事务，保证原子性。

**转移后的短期记忆处理**：

转移后的样本在短期缓冲区中的处理策略：

| 策略 | 说明 | 优点 | 缺点 |
|------|------|------|------|
| 立即删除 | 释放缓冲区空间 | 节省内存 | 丢失原始数据 |
| 标记不覆盖 | 保留但不再覆盖 | 保留原始数据 | 占用空间 |
| 延迟删除 | 保留一段时间后删除 | 平衡 | 复杂 |

默认采用"标记不覆盖"策略——转移后的样本保留在短期缓冲区中，但不会被FIFO覆盖优先选中，直到自然被新样本挤出。

- **输入**：短期记忆样本
- **输出**：长期记忆中的新记录
- **约束**：转移原子性；批量事务

> **白话翻译**：转移协议就是"归档流程"——从短期记忆转移到长期记忆的六个步骤：选样本→算评分→序列化→写入长期→标记短期→更新索引。整个过程必须原子化——要么全部完成，要么全部不做。批量归档用事务保证原子性。归档后的样本在短期记忆中标记"已归档"，不会被优先覆盖。

---

### 第三部分：巩固期间的重放强化

#### 2.3 巩固期重放强化定义（A77.3）

在样本转移过程中，对转移中的样本进行重放强化：

$$\text{replay\_during\_consolidation}() \tag{A77.3}$$

**重放强化的物理含义**：

神经科学中的记忆巩固理论（系统巩固理论）认为，记忆从海马体（短期）转移到新皮层（长期）的过程中，需要反复"回放"——海马体在睡眠期间重新激活记忆痕迹，驱动新皮层逐步学习。A77.3的计算实现模拟了这一过程——在样本从短期记忆转移到长期记忆时，通过离线复盘反复回放这些样本，确保模型对它们的记忆从"短期缓存"转化为"长期权重"。

**重放强化的流程**：

1. 收集待巩固的样本集合 $S_{\text{consolidate}}$
2. 对 $S_{\text{consolidate}}$ 执行多轮离线复盘（A76.2-A76.3）
3. 评估每轮复盘的收益（A76.4）
4. 收益低于阈值时停止（A76.5）
5. 将巩固后的样本写入长期记忆

**重放强化与A76.3的关系**：

A76.3（引用阶段76）定义批量优化 for sample in batch: model.update(sample)。A77.3的重放强化直接复用A76.3的批量优化机制——区别仅在于样本来源：A76.3从整个缓冲区采样，A77.3仅从待巩固样本中采样。

**重放强化的轮次**：

| 样本价值 | 重放轮次 | 说明 |
|---------|---------|------|
| 高（>0.7） | 5轮 | 高价值样本需要更多巩固 |
| 中（0.3-0.7） | 3轮 | 中等价值样本适度巩固 |
| 低（<0.3） | 1轮 | 低价值样本最少巩固 |

- **输入**：待巩固样本集合
- **输出**：巩固后的模型状态
- **约束**：每轮收益递减；总收益有上界

> **白话翻译**：巩固期重放强化就是"边归档边复习"——样本从短期记忆转移到长期记忆时，反复复习这些样本，确保模型真正学会了。这模拟了大脑睡眠时的记忆巩固——海马体重放记忆，驱动新皮层学习。高价值样本复习5轮，中等3轮，低价值1轮。复用离线复盘的批量优化机制。

---

### 第四部分：遗忘曲线

#### 2.4 遗忘曲线定义（A77.4）

遗忘曲线描述记忆保持率随时间的衰减：

$$R(t) = \exp\left(-\frac{t}{\tau_{\text{forget}}}\right) \tag{A77.4}$$

其中：
- $R(t) \in (0, 1]$ 为记忆保持率
- $t$ 为距上次复习的时间
- $\tau_{\text{forget}} > 0$ 为遗忘时间常数

**遗忘时间常数的选择**：

| $\tau_{\text{forget}}$ | 半衰期 | 适用记忆类型 |
|----------------------|--------|------------|
| 100秒 | 69秒 | 短期记忆 |
| 1000秒 | 693秒 | 工作记忆 |
| 10000秒 | 6931秒 | 长期记忆（默认） |
| ∞ | ∞ | 永久记忆 |

默认 $\tau_{\text{forget}} = 10000$ 秒（约2.8小时），对应长期记忆的遗忘时间常数。

**遗忘曲线与价值评分的关系**：

高价值样本的遗忘时间常数应更大（忘得更慢）：

$$\tau_{\text{forget}}(s) = \tau_{\text{base}} \cdot (1 + \kappa \cdot \text{value}(s)) \tag{A77.4-value}$$

其中 $\kappa > 0$ 为价值调制系数。价值越高的样本，遗忘越慢。

**遗忘曲线与复习的关系**：

每次复习后，记忆保持率重置为1，遗忘时间常数增大（间隔效应）：

$$R(t_{\text{after\_review}}) = 1 \tag{A77.4-reset}$$

$$\tau_{\text{forget}}^{(n+1)} = \tau_{\text{forget}}^{(n)} \cdot \phi, \quad \phi > 1 \tag{A77.4-spacing}$$

其中 $\phi$ 为间隔扩展因子（通常 $\phi = 2$），每次复习后遗忘时间常数翻倍。这解释了间隔重复的有效性——复习间隔越来越长，但记忆保持率不降。

**遗忘曲线的数值示例**：

| 复习次数 | $\tau_{\text{forget}}$ | 保持率>0.9的时间 | 保持率>0.5的时间 |
|---------|----------------------|----------------|----------------|
| 0 | 10000秒 | 1053秒 | 6931秒 |
| 1 | 20000秒 | 2107秒 | 13863秒 |
| 2 | 40000秒 | 4214秒 | 27726秒 |
| 3 | 80000秒 | 8427秒 | 55452秒 |

- **输入**：距上次复习的时间、遗忘时间常数
- **输出**：记忆保持率 R ∈ (0, 1]
- **约束**：τ_forget > 0；R单调递减

> **白话翻译**：遗忘曲线就是"记忆衰减规律"——不复习就忘，忘的速度先快后慢（指数衰减）。高价值的记忆忘得慢（τ更大），每次复习后忘得更慢（τ翻倍）。这就是间隔重复的原理：在快忘的时候及时复习，复习间隔越来越长，但记忆保持率不降。

---

### 第五部分：间隔重复调度

#### 2.5 间隔重复调度定义（A77.5）

间隔重复调度决定每个样本的下次复习时间：

$$t_{\text{next\_review}} = f(R, \text{value}, \text{age}) \tag{A77.5}$$

**调度策略**：

当记忆保持率低于阈值时触发复习：

$$t_{\text{next\_review}} = t_{\text{last\_review}} + \tau_{\text{forget}} \cdot \ln\left(\frac{1}{R_{\text{threshold}}}\right) \tag{A77.5-schedule}$$

其中 $R_{\text{threshold}}$ 为保持率阈值（通常设为0.85，即保持率低于85%时需要复习）。

**调度的优先级排序**：

当多个样本同时需要复习时，按优先级排序：

$$\text{priority}(s) = \text{value}(s) \cdot (1 - R(s)) \tag{A77.5-priority}$$

优先复习"价值高且快忘了"的样本。

**间隔重复的时间表**：

| 复习次数 | 间隔 | 累计时间 | 说明 |
|---------|------|---------|------|
| 1 | 1000秒 | 1000秒 | 首次复习 |
| 2 | 2000秒 | 3000秒 | 间隔翻倍 |
| 3 | 4000秒 | 7000秒 | 继续翻倍 |
| 4 | 8000秒 | 15000秒 | 继续翻倍 |
| 5 | 16000秒 | 31000秒 | 继续翻倍 |

5次复习后，样本的记忆保持率在31秒时仍>85%，基本实现了长期记忆。

**间隔重复与离线复盘的协同**：

间隔重复调度的复习任务在离线复盘时执行：

1. 离线复盘开始时，检查哪些样本的保持率低于阈值
2. 将这些样本加入复盘的采样池
3. 在批量重放中优先采样这些样本
4. 复习后更新遗忘时间常数和下次复习时间

- **输入**：记忆保持率、价值评分、样本年龄
- **输出**：下次复习时间
- **约束**：复习间隔单调递增

> **白话翻译**：间隔重复调度就是"什么时候该复习"——在保持率低于85%时及时复习，复习间隔越来越长（翻倍）。多个样本同时需要复习时，优先复习"价值高且快忘了"的。复习任务在离线复盘时执行——空闲时检查哪些记忆快忘了，优先复习这些。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A77.1 | should_consolidate = 𝟙[age > T ∧ value > θ] | 样本年龄、价值 | 巩固标志 | T > 0 |
| A77.2 | transfer(stm_sample) → ltm_sample | 短期记忆样本 | 长期记忆记录 | 原子性 |
| A77.3 | replay_during_consolidation() | 待巩固样本 | 巩固后模型 | 收益递减 |
| A77.4 | R(t) = exp(-t/τ_forget) | 时间、遗忘常数 | 保持率 ∈ (0,1] | τ > 0 |
| A77.5 | t_next = t_last + τ·ln(1/R_threshold) | 保持率、价值、年龄 | 下次复习时间 | 间隔递增 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import numpy as np

# 引用阶段76的数据类
try:
    from stage_076 import OfflineReplay
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


# ============================================================
# 阶段77 数据类：记忆巩固
# ============================================================
@dataclass(frozen=True)
class MemoryConsolidation:
    """记忆巩固的完整描述。

    组合阶段76的OfflineReplay，在离线复盘的基础上增加
    记忆巩固的调度状态。

    Attributes:
        offline_replay: 阶段76的离线复盘（组合字段）
        consolidate_threshold: 巩固年龄阈值（秒）
        forget_tau_base: 基础遗忘时间常数
        forget_value_kappa: 价值调制系数
        spacing_factor: 间隔扩展因子
        retention_threshold: 保持率阈值
        review_count: 当前复习轮次
        last_consolidate_time: 上次巩固时间

    Invariants:
        - consolidate_threshold > 0
        - forget_tau_base > 0
        - forget_value_kappa >= 0
        - spacing_factor > 1
        - 0 < retention_threshold < 1
    """
    offline_replay: OfflineReplay
    consolidate_threshold: float = 1000.0
    forget_tau_base: float = 10000.0
    forget_value_kappa: float = 1.0
    spacing_factor: float = 2.0
    retention_threshold: float = 0.85
    review_count: int = 0
    last_consolidate_time: float = 0.0

    def __post_init__(self):
        if self.consolidate_threshold <= 0:
            raise ValueError(
                f"巩固阈值必须 > 0，收到 {self.consolidate_threshold}"
            )
        if self.forget_tau_base <= 0:
            raise ValueError(
                f"遗忘时间常数必须 > 0，收到 {self.forget_tau_base}"
            )
        if self.spacing_factor <= 1:
            raise ValueError(
                f"间隔扩展因子必须 > 1，收到 {self.spacing_factor}"
            )
        if not (0 < self.retention_threshold < 1):
            raise ValueError(
                f"保持率阈值必须 ∈ (0,1)，收到 {self.retention_threshold}"
            )

    @property
    def forget_tau(self) -> float:
        """当前遗忘时间常数（考虑间隔效应）。"""
        return self.forget_tau_base * (self.spacing_factor ** self.review_count)


# ============================================================
# 阶段77 核心接口：记忆巩固
# ============================================================
class MemoryConsolidationInterface:
    """阶段77的核心接口：记忆巩固。

    实现转移条件判定、转移协议、巩固期重放、遗忘曲线和间隔重复。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 转移条件：年龄+价值双重判定
        - 遗忘曲线：指数衰减
        - 间隔重复：复习间隔单调递增
    """

    def should_consolidate(
        self,
        sample_age: float,
        sample_value: float,
        age_threshold: float,
        value_threshold: float
    ) -> bool:
        """判定是否应该巩固（A77.1）。

        Args:
            sample_age: 样本年龄（秒），必须 >= 0
            sample_value: 样本价值评分
            age_threshold: 年龄阈值（秒），必须 >= 0
            value_threshold: 价值阈值

        Returns:
            当年龄 > age_threshold 且价值 > value_threshold 时返回True

        Raises:
            ValueError: 当sample_age < 0时
            ValueError: 当age_threshold < 0时

        Examples:
            >>> mci = MemoryConsolidationInterface()
            >>> mci.should_consolidate(100.0, 0.8, 60.0, 0.5)
            True
            >>> mci.should_consolidate(30.0, 0.8, 60.0, 0.5)
            False
        """
        if sample_age < 0:
            raise ValueError(
                f"sample_age必须 >= 0，收到 {sample_age}"
            )
        if age_threshold < 0:
            raise ValueError(
                f"age_threshold必须 >= 0，收到 {age_threshold}"
            )
        return (sample_age > age_threshold) and (sample_value > value_threshold)

    def transfer(
        self,
        stm_sample: Dict,
        value_score: float
    ) -> Dict:
        """执行转移协议（A77.2）。

        Args:
            stm_sample: 短期记忆样本字典
            value_score: 价值评分

        Returns:
            长期记忆记录字典，包含原始字段+ltm_timestamp+value_score

        Examples:
            >>> mci = MemoryConsolidationInterface()
            >>> record = mci.transfer(
            ...     {"observation": np.array([1.0])}, 0.9)
            >>> "ltm_timestamp" in record
            True
            >>> record["value_score"]
            0.9
        """
        import time
        ltm_record = dict(stm_sample)
        ltm_record["ltm_timestamp"] = time.time()
        ltm_record["value_score"] = value_score
        return ltm_record

    def replay_during_consolidation(
        self,
        samples: List[Dict],
        model_params: np.ndarray,
        n_rounds: int = 3
    ) -> Tuple[np.ndarray, float]:
        """巩固期重放强化（A77.3）。

        Args:
            samples: 待巩固样本列表
            model_params: 当前模型参数，shape (n,)
            n_rounds: 重放轮次，必须 > 0

        Returns:
            (更新后参数, 总收益)

        Raises:
            ValueError: 当n_rounds <= 0时

        Examples:
            >>> mci = MemoryConsolidationInterface()
            >>> params = np.array([1.0, 0.0])
            >>> samples = [{"observation": np.array([1.0]), "target": 0.5}]
            >>> new_params, benefit = mci.replay_during_consolidation(
            ...     samples, params, n_rounds=2)
            >>> new_params.shape == params.shape
            True
        """
        if n_rounds <= 0:
            raise ValueError(
                f"n_rounds必须 > 0，收到 {n_rounds}"
            )
        new_params = model_params.copy()
        total_benefit = 0.0
        lr = 0.01
        for _ in range(n_rounds):
            for s in samples:
                obs = np.asarray(s.get("observation", np.zeros(1)))
                target = s.get("target", 0.0)
                pred = new_params[:obs.shape[0]] @ obs
                error_before = float((pred - target) ** 2)
                grad = 2 * (pred - target) * obs
                new_params = new_params - lr * grad
                pred_after = new_params[:obs.shape[0]] @ obs
                error_after = float((pred_after - target) ** 2)
                total_benefit += max(0.0, error_before - error_after)
        return new_params, total_benefit

    def retention(
        self,
        time_since_review: float,
        forget_tau: float
    ) -> float:
        """计算记忆保持率（A77.4）。

        R(t) = exp(-t / τ_forget)

        Args:
            time_since_review: 距上次复习的时间（秒），必须 >= 0
            forget_tau: 遗忘时间常数（秒），必须 > 0

        Returns:
            保持率 R ∈ (0, 1]

        Raises:
            ValueError: 当time_since_review < 0时
            ValueError: 当forget_tau <= 0时

        Examples:
            >>> mci = MemoryConsolidationInterface()
            >>> mci.retention(0.0, 100.0)
            1.0
            >>> mci.retention(100.0, 100.0) < 1.0
            True
        """
        if time_since_review < 0:
            raise ValueError(
                f"time_since_review必须 >= 0，收到 {time_since_review}"
            )
        if forget_tau <= 0:
            raise ValueError(
                f"forget_tau必须 > 0，收到 {forget_tau}"
            )
        return float(np.exp(-time_since_review / forget_tau))

    def next_review_time(
        self,
        last_review: float,
        forget_tau: float,
        retention_threshold: float
    ) -> float:
        """计算下次复习时间（A77.5）。

        t_next = t_last + τ * ln(1 / R_threshold)

        Args:
            last_review: 上次复习时间（时间戳）
            forget_tau: 遗忘时间常数（秒），必须 > 0
            retention_threshold: 保持率阈值，必须 ∈ (0,1)

        Returns:
            下次复习时间（时间戳）

        Raises:
            ValueError: 当forget_tau <= 0时
            ValueError: 当retention_threshold不在(0,1)范围内时

        Examples:
            >>> mci = MemoryConsolidationInterface()
            >>> t_next = mci.next_review_time(0.0, 100.0, 0.85)
            >>> t_next > 0
            True
        """
        if forget_tau <= 0:
            raise ValueError(
                f"forget_tau必须 > 0，收到 {forget_tau}"
            )
        if not (0 < retention_threshold < 1):
            raise ValueError(
                f"retention_threshold必须 ∈ (0,1)，"
                f"收到 {retention_threshold}"
            )
        delta_t = forget_tau * np.log(1.0 / retention_threshold)
        return float(last_review + delta_t)

    def review_priority(
        self,
        sample_value: float,
        retention: float
    ) -> float:
        """计算复习优先级（A77.5-priority）。

        priority = value * (1 - retention)

        Args:
            sample_value: 样本价值评分，必须 >= 0
            retention: 当前保持率，必须 ∈ [0,1]

        Returns:
            复习优先级 >= 0

        Raises:
            ValueError: 当sample_value < 0时
            ValueError: 当retention不在[0,1]范围内时

        Examples:
            >>> mci = MemoryConsolidationInterface()
            >>> mci.review_priority(1.0, 0.5)
            0.5
            >>> mci.review_priority(0.0, 0.0)
            0.0
        """
        if sample_value < 0:
            raise ValueError(
                f"sample_value必须 >= 0，收到 {sample_value}"
            )
        if not (0 <= retention <= 1):
            raise ValueError(
                f"retention必须 ∈ [0,1]，收到 {retention}"
            )
        return float(sample_value * (1.0 - retention))
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `OfflineReplay` | ← 阶段76 | 作为记忆巩固的离线复盘输入（组合字段） |
| **输入** | A17.1-A17.5（快感知-慢学习） | ← 阶段17 | 巩固的理论基础 |
| **输入** | A72.1-A72.5（长期记忆） | ← 阶段72 | 转移的目标存储 |
| **输入** | A5.18（时间尺度分离） | ← 阶段5 | 快-慢分离的原理 |
| **输出** | `MemoryConsolidation` | → 阶段78 | 记忆巩固状态，作为模型快照的输入 |
| **输出** | A77.1-A77.5（记忆巩固定义） | → 阶段79-80 | 记忆系统的巩固基础 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| SQLite | 转移写入 | 长期记忆的持久化存储 |
| NumPy | 遗忘曲线计算 | 指数衰减的向量化计算 |
| heapq | 复习优先级队列 | 按优先级排序的复习调度 |
| APScheduler | 定时复习调度 | 间隔重复的定时触发 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V77.1 | 转移条件判定 | age > T ∧ value > θ → consolidate | 条件测试 |
| V77.2 | 年龄不足不转移 | age ≤ T 时不转移 | 边界测试 |
| V77.3 | 价值不足不转移 | value ≤ θ 时不转移 | 边界测试 |
| V77.4 | 转移原子性 | 写入失败时短期记忆不标记 | 异常注入测试 |
| V77.5 | 遗忘曲线正确 | R(t) = exp(-t/τ) | 数值验证 |
| V77.6 | 保持率单调递减 | R(t₁) > R(t₂) 当 t₁ < t₂ | 单调性测试 |
| V77.7 | 价值调制遗忘 | 高价值样本τ更大 | 条件测试 |
| V77.8 | 间隔扩展 | 每次复习后τ翻倍 | 间隔测试 |
| V77.9 | 复习时间正确 | t_next = t_last + τ·ln(1/R_threshold) | 数值验证 |
| V77.10 | 复习间隔递增 | 后次间隔 > 前次间隔 | 单调性测试 |
| V77.11 | 复习优先级 | value高且R低 → 优先级高 | 排序测试 |
| V77.12 | 巩固期重放收益 | 重放后F下降 | 收益测试 |
| V77.13 | MemoryConsolidation组合 | `consolidation.offline_replay` 类型为 `OfflineReplay` | 类型检查 |
| V77.14 | 长期依赖测试 | 5次复习后保持率>0.85 | 长期测试 |
| V77.15 | 巩固不丢数据 | 转移后长期记忆可查询 | 完整性测试 |

---

> **阶段77完** | 阶段77建立了记忆巩固机制：快速→慢速转移条件 age(sample) > T_consolidate（A77.1），转移协议 transfer(stm_sample) → ltm_sample（A77.2），巩固期重放强化 replay_during_consolidation()（A77.3），遗忘曲线 R(t) = exp(-t/τ_forget)（A77.4），间隔重复调度 next_review = f(R, value, age)（A77.5），封装为MemoryConsolidation数据类组合OfflineReplay。下一步：阶段78将定义模型快照机制。
