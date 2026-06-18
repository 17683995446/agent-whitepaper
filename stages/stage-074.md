# 阶段74：优先级经验回放——价值驱动采样、偏差校正与优先级更新

> **阶段编号**：S74 | **所属层**：第八层·记忆系统层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 优先级采样 | "挑重要的学"——价值越高的经历，被抽中复习的概率越大 |
> | 重要性权重 | "校正偏心"——因为挑着学导致偏差，用权重校正回来 |
> | 批量采样 | "一次抽32张卡片"——每次复习抽一批样本，不是一个个来 |
> | 优先级更新 | "更新评分"——复习后重新评估样本的价值，调整下次被抽中的概率 |
> | PriorityReplay | "优先级回放的完整画像"——组合SampleValueEvaluator，在价值评估基础上增加回放策略 |

---

## 1. 第一性原理依据

本阶段从阶段73（样本价值评估）的核心结论出发，建立优先级经验回放机制。核心依据为：

- **A73.1（价值评分）**（引用阶段73）：value = α·error_surprise + β·info_gain + γ·reward_magnitude，样本价值的综合评分。优先级回放的核心思想即为"价值越高的样本越优先被回放"——A73.1的价值评分直接驱动采样概率。

- **A73.4（Top30%筛选）**（引用阶段73）：select_top_k(samples, k=0.3·|samples|)，高价值样本的筛选。优先级回放不是简单的Top-k筛选——而是按价值评分的概率采样，保证高价值样本被更频繁地回放，同时低价值样本也有被采样的机会。

- **A7.5（认知价值=互信息）**（引用阶段7）：认知价值度量策略的信息增益。优先级回放与认知价值在精神上一致——都是优先选择信息增益最大的样本/策略。但优先级回放是样本层面的实现，A7.5是策略层面的定义。

- **A9.2（中层循环）**（引用阶段9）：中层循环在表层循环无法有效降低自由能时触发策略调整。优先级回放为中层循环提供学习素材——通过回放高价值样本，帮助模型更快地适应新策略。

本阶段的核心目标是：**定义采样概率 P(sample_i) ∝ value_i^α / Σ_j value_j^α（A74.1），重要性权重 w_i = (N·P_i)^{-β}（偏差校正）（A74.2），批量采样 batch = sample_priority(buffer, batch_size=32)（A74.3），优先级更新 update_priority(sample, new_value)（A74.4），封装为PriorityReplay数据类，组合SampleValueEvaluator作为价值评估的输入。**

> **去重声明**：本阶段仅定义优先级经验回放的采样策略和偏差校正，不涉及生成式回放（阶段75）或离线复盘（阶段76）。价值评分的计算遵循A73.1，本阶段不重新定义。

---

## 2. 核心数学公式

### 第一部分：采样概率

#### 2.1 采样概率定义（A74.1）

采样概率与样本价值的α次幂成正比：

$$P(\text{sample}_i) = \frac{\text{value}_i^\alpha}{\sum_j \text{value}_j^\alpha} \tag{A74.1}$$

其中：
- $\text{value}_i$ 为样本 $i$ 的价值评分（A73.1定义）
- $\alpha \in [0, +\infty)$ 为优先级指数，控制优先级的强度
- $\sum_j$ 对缓冲区中所有样本求和

**优先级指数α的物理含义**：

| α值 | 采样策略 | 效果 |
|-----|---------|------|
| α = 0 | 均匀采样 | 所有样本等概率，无优先级 |
| α = 0.5 | 中等优先级 | 高价值样本略多于均匀采样 |
| α = 1 | 完全优先级 | 采样概率与价值成正比 |
| α → ∞ | 贪心采样 | 只采样价值最高的样本 |

**α的选择**：

默认 $\alpha = 0.6$，在优先级和多样性之间取得平衡。$\alpha$ 太小退化为均匀采样（浪费价值信息），$\alpha$ 太大退化为贪心采样（缺乏多样性）。

**采样概率的数值示例**：

假设缓冲区有5个样本，价值评分分别为 [0.1, 0.3, 0.5, 0.7, 0.9]：

| 样本 | value | P(α=0) | P(α=0.6) | P(α=1.0) |
|------|-------|--------|----------|----------|
| 1 | 0.1 | 0.20 | 0.08 | 0.04 |
| 2 | 0.3 | 0.20 | 0.16 | 0.13 |
| 3 | 0.5 | 0.20 | 0.22 | 0.22 |
| 4 | 0.7 | 0.20 | 0.26 | 0.30 |
| 5 | 0.9 | 0.20 | 0.28 | 0.31 |

α=0.6时，高价值样本的采样概率约为低价值样本的3.5倍（0.28/0.08），而非完全优先级下的7.8倍（0.31/0.04）。

**采样概率的实现——比例式优先级**：

为避免value=0的样本永远不被采样，采用比例式优先级：

$$P(\text{sample}_i) = \frac{(\text{value}_i + \epsilon)^\alpha}{\sum_j (\text{value}_j + \epsilon)^\alpha} \tag{A74.1-eps}$$

其中 $\epsilon > 0$ 为小常数（通常 $\epsilon = 10^{-6}$），确保零价值样本也有非零采样概率。

- **输入**：所有样本的价值评分、优先级指数α
- **输出**：采样概率分布 P
- **约束**：Σ P_i = 1；P_i > 0

> **白话翻译**：采样概率就是"挑重要的学"——价值越高的样本，被抽中复习的概率越大。α控制"偏心"程度：α=0完全不偏心（均匀采样），α=1按价值比例偏心，α→∞只挑最贵的。默认α=0.6，适度偏心。加个小常数ε防止零价值样本永远不被抽中。

---

### 第二部分：重要性权重

#### 2.2 重要性权重定义（A74.2）

优先级采样引入了偏差——高价值样本被过度采样。重要性权重校正此偏差：

$$w_i = \left(\frac{1}{N \cdot P_i}\right)^\beta \tag{A74.2}$$

其中：
- $N$ 为缓冲区中的样本总数
- $P_i$ 为样本 $i$ 的采样概率（A74.1）
- $\beta \in [0, 1]$ 为重要性采样指数

**重要性权重的推导**：

在均匀采样下，每个样本的采样概率为 $1/N$。在优先级采样下，采样概率为 $P_i$。重要性权重补偿两种采样策略的差异：

$$w_i = \frac{P_{\text{uniform}}(i)}{P_{\text{priority}}(i)} = \frac{1/N}{P_i} \tag{A74.2-derivation}$$

引入指数 $\beta$ 控制校正强度：

- $\beta = 0$：不校正（完全接受优先级偏差）
- $\beta = 1$：完全校正（消除优先级偏差）

**β的退火策略**：

训练初期优先级偏差有利于快速学习，不需要强校正；训练后期需要更准确的梯度估计，需要强校正。因此β从0线性退火到1：

$$\beta_t = \beta_0 + (1 - \beta_0) \cdot \frac{t}{T_{\text{anneal}}} \tag{A74.2-anneal}$$

其中 $T_{\text{anneal}}$ 为退火总步数。

**重要性权重的归一化**：

为防止权重过大导致梯度爆炸，对重要性权重进行归一化：

$$\hat{w}_i = \frac{w_i}{\max_j w_j} \tag{A74.2-normalize}$$

归一化后 $\hat{w}_i \in (0, 1]$，最大权重为1。

**重要性权重的数值示例**：

| 样本 | P_i | 1/(N·P_i) | w(β=0) | w(β=0.5) | w(β=1) |
|------|-----|-----------|--------|----------|--------|
| 高价值 | 0.28 | 0.71 | 1.00 | 0.84 | 0.71 |
| 中价值 | 0.22 | 0.91 | 1.00 | 0.95 | 0.91 |
| 低价值 | 0.08 | 2.50 | 1.00 | 1.58 | 2.50 |

β=1时，低价值样本的权重最大（2.50），因为它被欠采样了，需要更大的权重来补偿。

- **输入**：采样概率P_i、样本总数N、校正指数β
- **输出**：重要性权重 w_i ∈ (0, +∞)
- **约束**：β ∈ [0, 1]；归一化后 w_i ∈ (0, 1]

> **白话翻译**：重要性权重就是"校正偏心"——因为挑着学导致高价值样本被过度采样，低价值样本被欠采样。校正方法：被欠采样的样本给更大的权重，被过度采样的样本给更小的权重。β控制校正强度：β=0不校正，β=1完全校正。训练初期不校正（偏心有利于快速学习），训练后期逐渐校正（需要更准确的梯度）。

---

### 第三部分：批量采样

#### 2.3 批量采样定义（A74.3）

批量采样从缓冲区中按优先级概率抽取一批样本：

$$\text{batch} = \text{sample\_priority}(\text{buffer}, \text{batch\_size}=32) \tag{A74.3}$$

**采样流程**：

1. 计算缓冲区中所有样本的采样概率 $P_i$（A74.1）
2. 按概率分布随机采样batch_size个样本（有放回）
3. 计算每个采样样本的重要性权重 $w_i$（A74.2）
4. 返回 (batch_samples, importance_weights)

**采样算法的选择**：

| 算法 | 预处理 | 采样 | 更新 | 适用场景 |
|------|--------|------|------|---------|
| 朴素随机采样 | O(N) | O(batch) | O(1) | 小缓冲区 |
| 分段线性采样 | O(N) | O(batch·logN) | O(N) | 中等缓冲区 |
| SumTree采样 | O(N) | O(batch·logN) | O(logN) | 大缓冲区 |

默认采用SumTree（线段树）实现，预处理O(N)，采样O(batch·logN)，优先级更新O(logN)。

**SumTree数据结构**：

SumTree是一棵二叉树，叶节点存储样本的优先级，非叶节点存储子节点优先级之和：

$$\text{tree}[i] = \text{tree}[2i] + \text{tree}[2i+1] \tag{A74.3-sumtree}$$

采样时，在[0, total_priority]范围内生成随机数，沿树向下查找对应的叶节点。

**batch_size的选择**：

| batch_size | 梯度估计 | 计算开销 | 适用场景 |
|-----------|---------|---------|---------|
| 16 | 高方差 | 低 | 快速迭代 |
| 32 | 适中 | 适中 | 默认选择 |
| 64 | 低方差 | 高 | 精确更新 |
| 128 | 极低方差 | 很高 | 离线复盘（阶段76） |

默认batch_size=32，在梯度估计质量和计算开销之间取得平衡。

- **输入**：缓冲区、batch_size
- **输出**：批量样本 + 重要性权重
- **约束**：batch_size ≤ N；有放回采样

> **白话翻译**：批量采样就是"一次抽32张卡片"——按优先级概率从缓冲区中抽一批样本，同时计算重要性权重。用SumTree（线段树）加速采样——预处理一次O(N)，之后每次采样O(logN)。32张卡片是默认选择——太少梯度不稳定，太多计算太贵。

---

### 第四部分：优先级更新

#### 2.4 优先级更新定义（A74.4）

回放后根据新的TD误差或价值评分更新样本的优先级：

$$\text{update\_priority}(\text{sample}, \text{new\_value}) \tag{A74.4}$$

**优先级更新的流程**：

1. 用当前模型对回放样本重新计算价值评分
2. 用新价值评分更新采样概率
3. 更新SumTree中的对应叶节点

**优先级更新的物理含义**：

一个样本被回放后，模型可能已经学会了该样本包含的信息——此时该样本的信息增益降低，价值评分下降，下次被采样的概率也降低。反之，如果模型仍然无法很好地处理该样本，价值评分保持高位，继续被频繁采样。

**优先级更新的频率**：

| 策略 | 频率 | 开销 | 效果 |
|------|------|------|------|
| 每次回放后更新 | 高 | 高 | 最精确 |
| 每batch后更新 | 中 | 中 | 平衡 |
| 每N步更新 | 低 | 低 | 近似 |

默认每batch后更新一次，在精确性和开销之间取得平衡。

**优先级更新与A73.5的关系**：

A73.5（引用阶段73）定义价值评估的在线更新。A74.4的优先级更新是A73.5在回放场景下的具体应用——回放后重新评估样本价值，并据此调整采样概率。

- **输入**：样本、新价值评分
- **输出**：更新后的SumTree/采样概率
- **约束**：更新后概率分布仍归一化

> **白话翻译**：优先级更新就是"更新评分"——复习后重新评估样本的价值。如果模型已经学会了这个样本，价值下降，下次被抽中的概率也下降；如果还是学不会，价值保持高位，继续被频繁复习。默认每批复习后更新一次评分。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A74.1 | P(i) = value_i^α / Σ value_j^α | 价值评分、α | 采样概率 | ΣP=1, P>0 |
| A74.2 | w_i = (1/(N·P_i))^β | 采样概率、N、β | 重要性权重 | β∈[0,1] |
| A74.3 | batch = sample_priority(buffer, 32) | 缓冲区、batch_size | 批量样本+权重 | batch_size≤N |
| A74.4 | update_priority(sample, new_value) | 样本、新价值 | 更新后概率分布 | 归一化 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np

# 引用阶段73的数据类
try:
    from stage_073 import SampleValueEvaluator
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


# ============================================================
# 阶段74 数据类：优先级回放
# ============================================================
@dataclass(frozen=True)
class PriorityReplay:
    """优先级经验回放的完整描述。

    组合阶段73的SampleValueEvaluator，在价值评估的基础上增加
    优先级采样和偏差校正的状态。

    Attributes:
        evaluator: 阶段73的样本价值评估器（组合字段）
        alpha: 优先级指数
        beta: 重要性采样指数
        beta_start: β初始值
        beta_frames: β退火总步数
        batch_size: 批量采样大小
        epsilon: 零价值保护常数
        current_step: 当前步数

    Invariants:
        - alpha >= 0
        - 0 <= beta <= 1
        - batch_size > 0
        - epsilon > 0
    """
    evaluator: SampleValueEvaluator
    alpha: float = 0.6
    beta: float = 0.4
    beta_start: float = 0.4
    beta_frames: int = 100000
    batch_size: int = 32
    epsilon: float = 1e-6
    current_step: int = 0

    def __post_init__(self):
        if self.alpha < 0:
            raise ValueError(
                f"优先级指数必须 >= 0，收到 {self.alpha}"
            )
        if not (0 <= self.beta <= 1):
            raise ValueError(
                f"重要性采样指数必须 ∈ [0,1]，收到 {self.beta}"
            )
        if self.batch_size <= 0:
            raise ValueError(
                f"批量大小必须 > 0，收到 {self.batch_size}"
            )
        if self.epsilon <= 0:
            raise ValueError(
                f"保护常数必须 > 0，收到 {self.epsilon}"
            )

    @property
    def beta_annealed(self) -> float:
        """退火后的β值（A74.2-anneal）。"""
        progress = min(1.0, self.current_step / max(1, self.beta_frames))
        return self.beta_start + (1.0 - self.beta_start) * progress


# ============================================================
# 阶段74 核心接口：优先级经验回放
# ============================================================
class PriorityReplayInterface:
    """阶段74的核心接口：优先级经验回放。

    实现优先级采样、重要性权重和优先级更新。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 采样概率归一化
        - 重要性权重归一化
        - 优先级更新后概率归一化
    """

    def compute_sampling_probabilities(
        self,
        values: np.ndarray,
        alpha: float,
        epsilon: float = 1e-6
    ) -> np.ndarray:
        """计算采样概率（A74.1）。

        P(i) = (value_i + ε)^α / Σ(value_j + ε)^α

        Args:
            values: 价值评分数组，shape (n,)
            alpha: 优先级指数，必须 >= 0
            epsilon: 零价值保护常数，必须 > 0

        Returns:
            采样概率数组，shape (n,)，和为1

        Raises:
            ValueError: 当alpha < 0时
            ValueError: 当epsilon <= 0时
            ValueError: 当values为空时

        Examples:
            >>> sampler = PriorityReplaySampler()
            >>> probs = sampler.compute_sampling_probabilities(
            ...     np.array([1.0, 2.0, 3.0]), alpha=1.0)
            >>> np.isclose(np.sum(probs), 1.0)
            True
            >>> probs[2] > probs[0]
            True
        """
        if alpha < 0:
            raise ValueError(f"alpha必须 >= 0，收到 {alpha}")
        if epsilon <= 0:
            raise ValueError(f"epsilon必须 > 0，收到 {epsilon}")
        if values.size == 0:
            raise ValueError("values不能为空数组")
        scaled = (values + epsilon) ** alpha
        total = np.sum(scaled)
        if total <= 0:
            return np.ones_like(values) / values.size
        return scaled / total

    def compute_importance_weights(
        self,
        probabilities: np.ndarray,
        n_samples: int,
        beta: float
    ) -> np.ndarray:
        """计算重要性权重（A74.2）。

        w_i = (1/(N·P_i))^β, 归一化为 w_i/max(w)

        Args:
            probabilities: 采样概率数组，shape (n,)
            n_samples: 样本总数N，必须 > 0
            beta: 重要性采样指数，必须 ∈ [0, 1]

        Returns:
            归一化重要性权重数组，shape (n,)，max=1.0

        Raises:
            ValueError: 当beta不在[0,1]范围内时
            ValueError: 当n_samples <= 0时
            ValueError: 当probabilities包含0时

        Examples:
            >>> sampler = PriorityReplaySampler()
            >>> weights = sampler.compute_importance_weights(
            ...     np.array([0.5, 0.5]), n_samples=2, beta=0.5)
            >>> np.isclose(np.max(weights), 1.0)
            True
        """
        if not (0 <= beta <= 1):
            raise ValueError(f"beta必须 ∈ [0,1]，收到 {beta}")
        if n_samples <= 0:
            raise ValueError(f"n_samples必须 > 0，收到 {n_samples}")
        if np.any(probabilities <= 0):
            raise ValueError("probabilities必须全部 > 0")
        weights = (1.0 / (n_samples * probabilities)) ** beta
        max_w = np.max(weights)
        if max_w > 0:
            weights = weights / max_w
        return weights

    def sample_batch(
        self,
        replay: PriorityReplay,
        buffer_values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """批量采样（A74.3）。

        Args:
            replay: 当前回放状态
            buffer_values: 缓冲区中所有样本的价值评分，shape (n,)

        Returns:
            (采样索引, 采样概率, 重要性权重)
            - 采样索引: shape (batch_size,)
            - 采样概率: shape (batch_size,)
            - 重要性权重: shape (batch_size,)

        Raises:
            ValueError: 当buffer_values为空时
            ValueError: 当batch_size > len(buffer_values)时

        Examples:
            >>> sampler = PriorityReplaySampler()
            >>> ltm = LongTermMemory(
            ...     stm=ShortTermMemory(
            ...         validation_result=EvolutionValidationResult(
            ...             is_valid=True, fitness=0.0),
            ...         db_path=":memory:"
            ...     ),
            ...     db_path=":memory:"
            ... )
            >>> ev = SampleValueEvaluator(
            ...     ltm=ltm, error_mean=0.0, error_var=1.0,
            ...     alpha=0.4, beta=0.4, gamma=0.2,
            ...     forget_factor=0.99, top_k_ratio=0.3)
            >>> replay = PriorityReplay(
            ...     evaluator=ev, alpha=1.0, beta_start=0.4,
            ...     beta_end=1.0, beta_anneal_steps=10000,
            ...     batch_size=2)
            >>> values = np.array([1.0, 2.0, 3.0, 4.0])
            >>> idx, probs, weights = sampler.sample_batch(replay, values)
            >>> len(idx) == replay.batch_size
            True
        """
        if buffer_values.size == 0:
            raise ValueError("buffer_values不能为空")
        if replay.batch_size > buffer_values.size:
            raise ValueError(
                f"batch_size({replay.batch_size})不能超过"
                f"buffer大小({buffer_values.size})"
            )
        probs = self.compute_sampling_probabilities(
            buffer_values, replay.alpha
        )
        indices = np.random.choice(
            len(buffer_values),
            size=replay.batch_size,
            replace=False,
            p=probs
        )
        sampled_probs = probs[indices]
        current_beta = replay.current_beta
        weights = self.compute_importance_weights(
            sampled_probs, len(buffer_values), current_beta
        )
        return indices, sampled_probs, weights

    def update_priority(
        self,
        replay: PriorityReplay,
        indices: np.ndarray,
        new_values: np.ndarray
    ) -> PriorityReplay:
        """更新优先级（A74.4）。

        Args:
            replay: 当前回放状态
            indices: 待更新的样本索引，shape (k,)
            new_values: 新的价值评分，shape (k,)

        Returns:
            更新后的回放状态

        Raises:
            ValueError: 当indices和new_values长度不匹配时

        Examples:
            >>> sampler = PriorityReplaySampler()
            >>> ltm = LongTermMemory(
            ...     stm=ShortTermMemory(
            ...         validation_result=EvolutionValidationResult(
            ...             is_valid=True, fitness=0.0),
            ...         db_path=":memory:"
            ...     ),
            ...     db_path=":memory:"
            ... )
            >>> ev = SampleValueEvaluator(
            ...     ltm=ltm, error_mean=0.0, error_var=1.0,
            ...     alpha=0.4, beta=0.4, gamma=0.2,
            ...     forget_factor=0.99, top_k_ratio=0.3)
            >>> replay = PriorityReplay(
            ...     evaluator=ev, alpha=1.0, beta_start=0.4,
            ...     beta_end=1.0, beta_anneal_steps=10000,
            ...     batch_size=2)
            >>> new_replay = sampler.update_priority(
            ...     replay, np.array([0]), np.array([5.0]))
            >>> new_replay.total_samples == replay.total_samples
            True
        """
        if len(indices) != len(new_values):
            raise ValueError(
                f"indices长度({len(indices)})与new_values长度"
                f"({len(new_values)})不匹配"
            )
        return PriorityReplay(
            evaluator=replay.evaluator,
            alpha=replay.alpha,
            beta_start=replay.beta_start,
            beta_end=replay.beta_end,
            beta_anneal_steps=replay.beta_anneal_steps,
            current_beta=replay.current_beta,
            batch_size=replay.batch_size,
            total_samples=replay.total_samples,
            total_updates=replay.total_updates + len(indices)
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `SampleValueEvaluator` | ← 阶段73 | 作为优先级回放的价值评估输入（组合字段） |
| **输入** | A73.1（价值评分） | ← 阶段73 | 采样概率的驱动信号 |
| **输入** | A7.5（认知价值） | ← 阶段7 | 优先级回放的理论基础 |
| **输入** | A9.2（中层循环） | ← 阶段9 | 回放为策略调整提供学习素材 |
| **输出** | `PriorityReplay` | → 阶段75 | 优先级回放状态，作为生成式回放的输入 |
| **输出** | A74.1-A74.4（优先级回放定义） | → 阶段76-80 | 记忆系统的回放基础 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| NumPy | 概率计算和采样 | np.random.choice支持概率采样 |
| Cython | SumTree加速 | 线段树的高性能实现 |
| JAX | 向量化采样 | 大规模缓冲区的GPU加速采样 |
| PyTorch | 权重校正 | 重要性权重在梯度更新中的应用 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V74.1 | 采样概率归一化 | ΣP_i = 1.0 | 求和验证 |
| V74.2 | 采样概率非负 | P_i > 0 对所有样本 | 数值验证 |
| V74.3 | 高价值高概率 | value_i > value_j ⟹ P_i > P_j | 条件测试 |
| V74.4 | α=0均匀采样 | P_i = 1/N 对所有i | 边界测试 |
| V74.5 | 重要性权重正确 | w_i = (1/(N·P_i))^β | 数值验证 |
| V74.6 | 权重归一化 | max(w) = 1.0 | 归一化验证 |
| V74.7 | β退火 | β从β_start线性增长到1 | 退火测试 |
| V74.8 | 批量采样大小 | 返回batch_size个样本 | 计数测试 |
| V74.9 | 优先级更新生效 | 更新后采样概率改变 | 更新前后对比 |
| V74.10 | SumTree正确性 | 树结构和与total_priority一致 | 结构验证 |
| V74.11 | 零价值保护 | value=0的样本P > 0 | 边界测试 |
| V74.12 | PriorityReplay组合 | `replay.evaluator` 类型为 `SampleValueEvaluator` | 类型检查 |
| V74.13 | 非法α拒绝 | α < 0 抛出 `ValueError` | 异常测试 |
| V74.14 | 非法β拒绝 | β ∉ [0,1] 抛出 `ValueError` | 异常测试 |
| V74.15 | 长期回放稳定性 | 10000步回放无异常 | 长期运行测试 |

---

> **阶段74完** | 阶段74建立了优先级经验回放机制：采样概率 P(sample_i) ∝ value_i^α / Σ value_j^α（A74.1），重要性权重 w_i = (N·P_i)^{-β}（偏差校正）（A74.2），批量采样 batch = sample_priority(buffer, batch_size=32)（A74.3），优先级更新 update_priority(sample, new_value)（A74.4），封装为PriorityReplay数据类组合SampleValueEvaluator。下一步：阶段75将定义生成式回放机制。
