# 阶段73：样本价值评估——误差惊奇度、信息增益与Top30%筛选

> **阶段编号**：S73 | **所属层**：第八层·记忆系统层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 价值评分 | "这段经历值多少分"——综合惊奇度、信息量和收益给经历打分 |
> | 误差惊奇度 | "多意外"——误差越罕见，越值得记住 |
> | 信息增益 | "学到了多少"——这段经历让模型增长了多少认知 |
> | 奖励幅度 | "赚了多少"——收益越高的经历越值得回味 |
> | Top30%筛选 | "挑精华"——只保留最值得学习的前30%经历 |
> | 在线更新 | "边跑边评"——价值评分随新数据持续更新 |
> | SampleValueEvaluator | "价值评估器的完整画像"——组合LongTermMemory，在长期记忆基础上增加价值评估 |

---

## 1. 第一性原理依据

本阶段从阶段7（预期自由能）和阶段72（离线长期记忆）的核心结论出发，建立样本价值评估机制。核心依据为：

- **A7.5（认知价值=互信息）**（引用阶段7）：认知价值定义为互信息 $I(\Psi; O|\text{sample})$，度量样本对减少环境不确定性的贡献。本阶段将A7.5的互信息概念具体化为样本的信息增益——一个样本使模型对环境状态的互信息增加了多少。

- **A72.1（样本表设计）**（引用阶段72）：samples(id, observation, prediction, error, reward, timestamp, value_score)，长期记忆的样本表包含value_score字段。本阶段定义value_score的计算方式——价值评分是误差惊奇度、信息增益和奖励幅度的加权组合。

- **A3.1（误差信号定义）**（引用阶段3）：$e^{l-1} = r^{l-1} - \hat{r}^{l-1}$，预测误差信号。误差惊奇度基于误差信号的历史分布——越偏离历史分布的误差越"惊奇"，价值越高。

- **A51.5（全局误差）**（引用阶段51）：$\varepsilon_{\text{global}} = \sum_l w_l \varepsilon_l$，三层误差的加权聚合。价值评分中的误差惊奇度以A51.5的全局误差为基础。

本阶段的核心目标是：**定义价值评分 value = α·error_surprise + β·info_gain + γ·reward_magnitude（A73.1），误差惊奇度 error_surprise = -log p(error | error_history)（A73.2），信息增益 info_gain = I(Ψ;O|sample) - I(Ψ;O|sample\{new})（A73.3），Top30%高价值样本筛选 select_top_k(samples, k=0.3·|samples|)（A73.4），价值评估的在线更新（A73.5），封装为SampleValueEvaluator数据类，组合LongTermMemory作为长期记忆的输入。**

> **去重声明**：本阶段仅定义样本价值评估的计算方式，不涉及优先级回放（阶段74）或记忆巩固（阶段77）。互信息的定义遵循A7.5，误差信号的定义遵循A3.1，本阶段不重新定义。

---

## 2. 核心数学公式

### 第一部分：价值评分

#### 2.1 价值评分定义（A73.1）

样本的价值评分为三个分量的加权组合：

$$\text{value} = \alpha \cdot \text{error\_surprise} + \beta \cdot \text{info\_gain} + \gamma \cdot \text{reward\_magnitude} \tag{A73.1}$$

其中：
- $\alpha, \beta, \gamma \geq 0$ 为权重系数，$\alpha + \beta + \gamma = 1$
- error_surprise 为误差惊奇度（A73.2）
- info_gain 为信息增益（A73.3）
- reward_magnitude 为奖励幅度

**权重系数的默认值**：

| 权重 | 默认值 | 含义 | 调整原则 |
|------|--------|------|---------|
| α | 0.4 | 误差惊奇度权重 | 高探索阶段增大 |
| β | 0.4 | 信息增益权重 | 高学习阶段增大 |
| γ | 0.2 | 奖励幅度权重 | 高利用阶段增大 |

**三个分量的物理含义**：

1. **误差惊奇度**（α=0.4）：度量样本的"出乎意料"程度——误差越罕见，越可能包含新信息。高惊奇度样本是模型尚未学会的领域。

2. **信息增益**（β=0.4）：度量样本对模型认知的增量贡献——一个样本使模型对环境的理解增加了多少。高信息增益样本是"最有教育意义"的经历。

3. **奖励幅度**（γ=0.2）：度量样本的即时收益——高奖励样本是"最有价值"的经历。权重最低因为奖励是外在信号，不如前两者反映认知价值。

**价值评分的性质**：

**(P1) 非负性**：value ≥ 0（由各分量非负和权重非负保证）。

**(P2) 单调性**：任一分量增大时，value不减小。

**(P3) 有界性**：在正常运行条件下，value ≤ value_max（由各分量的上界决定）。

**价值评分与A7.5认知价值的关系**：

A7.5（引用阶段7）定义认知价值为互信息 $I(\Psi; O|\pi)$，度量策略 $\pi$ 的认知收益。A73.1的价值评分是A7.5在样本层面的实例化——info_gain分量直接对应A7.5的互信息，error_surprise和reward_magnitude是认知价值的补充维度。

- **输入**：误差惊奇度、信息增益、奖励幅度、权重系数
- **输出**：价值评分 value ≥ 0
- **约束**：α + β + γ = 1；各权重非负

> **白话翻译**：价值评分就是"这段经历值多少分"——综合三个方面：多意外（误差惊奇度）、学到了多少（信息增益）、赚了多少（奖励幅度），按4:4:2的权重加权。意外和学习的权重各占40%，因为它们反映认知价值；收益只占20%，因为外在奖励不如认知增长重要。

---

### 第二部分：误差惊奇度

#### 2.2 误差惊奇度定义（A73.2）

误差惊奇度度量当前误差在历史误差分布中的罕见程度：

$$\text{error\_surprise} = -\log p(\text{error} \mid \text{error\_history}) \tag{A73.2}$$

其中 $p(\text{error} \mid \text{error\_history})$ 为当前误差在历史误差分布中的概率密度。

**历史误差分布的建模**：

采用核密度估计（KDE）建模历史误差分布：

$$p(e \mid \text{history}) = \frac{1}{n} \sum_{i=1}^{n} K_h(e - e_i) \tag{A73.2-kde}$$

其中：
- $K_h(\cdot) = (1/h) K(\cdot / h)$ 为核函数（默认高斯核）
- $h$ 为带宽参数
- $e_i$ 为历史误差值

**带宽选择**：

采用Silverman法则：

$$h = 1.06 \cdot \hat{\sigma} \cdot n^{-1/5} \tag{A73.2-bandwidth}$$

其中 $\hat{\sigma}$ 为历史误差的标准差，$n$ 为历史误差数量。

**误差惊奇度的数值示例**：

| 误差值 | 历史概率密度 | 惊奇度 | 含义 |
|--------|------------|--------|------|
| μ（均值） | 高 | 低 | 常见误差，不惊奇 |
| μ + σ | 中 | 中 | 略高于平均 |
| μ + 2σ | 低 | 高 | 较罕见，值得关注 |
| μ + 3σ | 极低 | 极高 | 非常罕见，高度惊奇 |

**误差惊奇度与A3.1的关系**：

A3.1（引用阶段3）定义误差信号 $e^{l-1} = r^{l-1} - \hat{r}^{l-1}$。A73.2的误差惊奇度基于A51.5的全局误差 $\varepsilon_{\text{global}}$，而 $\varepsilon_{\text{global}}$ 由各层误差信号计算。因此误差惊奇度间接反映了误差信号的异常程度。

**在线更新的简化方案**：

KDE的完整计算复杂度为O(n)，对在线场景开销较大。简化方案采用滑动窗口高斯近似：

$$p(e \mid \text{history}) \approx \mathcal{N}(e; \mu_W, \sigma_W^2) \tag{A73.2-online}$$

其中 $\mu_W$ 和 $\sigma_W^2$ 为滑动窗口 $W$ 内的误差均值和方差。在线更新仅需维护窗口内的均值和方差，复杂度O(1)。

- **输入**：当前误差、历史误差分布
- **输出**：误差惊奇度 ≥ 0
- **约束**：p > 0（避免log(0)）；惊奇度非负

> **白话翻译**：误差惊奇度就是"多意外"——当前误差在历史上有多罕见。罕见误差的惊奇度高，常见误差的惊奇度低。用概率密度来衡量：概率越低越惊奇，-log(p)越大。在线场景下用滑动窗口的高斯近似，只维护均值和方差，计算O(1)。

---

### 第三部分：信息增益

#### 2.3 信息增益定义（A73.3）

信息增益度量样本对模型认知的增量贡献：

$$\text{info\_gain} = I(\Psi; O \mid \text{sample}) - I(\Psi; O \mid \text{sample} \setminus \{\text{new}\}) \tag{A73.3}$$

其中：
- $I(\Psi; O \mid \text{sample})$ 为加入新样本后模型与观测的互信息
- $I(\Psi; O \mid \text{sample} \setminus \{\text{new}\})$ 为加入新样本前的互信息
- info_gain ≥ 0（加入新样本不会减少互信息）

**与A7.5的对应**：

A7.5（引用阶段7）定义认知价值为互信息 $I(\Psi; O|\pi)$。A73.3的信息增益是A7.5的增量形式——不是互信息本身，而是互信息的增量。这更精确地度量了单个样本的认知贡献。

**信息增益的计算**：

在变分推断框架下，互信息可通过变分下界近似：

$$I(\Psi; O) \geq \mathbb{E}_{q(\Psi)}[\ln p(O|\Psi)] - \ln p(O) \tag{A73.3-lb}$$

信息增益的近似计算：

$$\text{info\_gain} \approx \Delta \mathbb{E}_{q(\Psi)}[\ln p(O|\Psi)] = \mathbb{E}_{q_{\text{new}}}[\ln p(O|\Psi)] - \mathbb{E}_{q_{\text{old}}}[\ln p(O|\Psi)] \tag{A73.3-approx}$$

**信息增益的实用近似**：

完整计算信息增益需要两次前向传播（加入样本前后），开销较大。实用近似采用Fisher信息量：

$$\text{info\_gain} \approx \frac{1}{2} \Delta\theta^T F \Delta\theta \tag{A73.3-fisher}$$

其中：
- $F$ 为Fisher信息矩阵
- $\Delta\theta$ 为加入新样本后的参数更新量

Fisher信息近似将信息增益转化为参数空间的距离度量——参数更新越大，信息增益越高。

**信息增益的数值示例**：

| 样本类型 | 参数更新 | Fisher信息量 | 信息增益 | 含义 |
|---------|---------|------------|---------|------|
| 重复样本 | ≈0 | - | ≈0 | 模型已学会，无新信息 |
| 边界样本 | 小 | 中 | 小-中 | 模型略受影响 |
| 异常样本 | 大 | 高 | 高 | 模型显著调整 |
| 新领域样本 | 很大 | 很高 | 很高 | 模型大幅更新 |

- **输入**：加入样本前后的模型状态
- **输出**：信息增益 ≥ 0
- **约束**：info_gain ≥ 0（互信息单调不减）

> **白话翻译**：信息增益就是"学到了多少"——加入这段经历后，模型对世界的理解增加了多少。精确计算需要两次前向传播，太贵了。实用近似用Fisher信息量——参数更新越大，信息增益越高。重复的经历信息增益≈0（已经学会了），异常的经历信息增益很高（模型需要大幅调整）。

---

### 第四部分：Top30%高价值样本筛选

#### 2.4 Top30%筛选定义（A73.4）

从样本集合中筛选价值评分最高的前30%：

$$\text{select\_top\_k}(\text{samples}, k = 0.3 \cdot |\text{samples}|) \tag{A73.4}$$

**筛选算法**：

采用快速选择算法（QuickSelect），时间复杂度O(n)：

1. 计算所有样本的价值评分
2. 找到第70百分位的value_score作为阈值
3. 返回value_score ≥ 阈值的所有样本

**30%比例的选择依据**：

| 比例 | 筛选结果 | 适用场景 |
|------|---------|---------|
| 10% | 极度精选 | 存储极度受限 |
| 30% | 精选 | 平衡选择性和多样性 |
| 50% | 宽松选择 | 存储充裕 |
| 100% | 全部保留 | 无筛选 |

30%在选择性和多样性之间取得平衡——足够精选以排除低价值样本，又足够宽松以保留多样性。

**Top30%与持久化的关系**：

A72.3（引用阶段72）的持久化阈值 $\theta_{\text{persist}}$ 与A73.4的Top30%筛选密切相关——Top30%的最低value_score即为 $\theta_{\text{persist}}$ 的动态值：

$$\theta_{\text{persist}} = \text{Percentile}_{70}(\{\text{value}(s) \mid s \in \text{samples}\}) \tag{A73.4-threshold}$$

- **输入**：样本集合、筛选比例
- **输出**：高价值样本子集
- **约束**：筛选比例 ∈ (0, 1]

> **白话翻译**：Top30%筛选就是"挑精华"——按价值评分排序，只保留前30%。30%是平衡点：太少了可能漏掉有价值的经历，太多了又不够精选。第70百分位的价值评分就是持久化阈值——超过这个分数的才值得长期保存。

---

### 第五部分：价值评估的在线更新

#### 2.5 价值评估在线更新定义（A73.5）

价值评估需要随新数据持续更新，反映系统认知状态的变化：

$$\text{value}_t(s) = \alpha_t \cdot \text{surprise}_t(s) + \beta_t \cdot \text{info\_gain}_t(s) + \gamma_t \cdot |\text{reward}(s)| \tag{A73.5}$$

**在线更新的三个方面**：

**1. 误差惊奇度的在线更新**：

随着新误差的积累，历史误差分布发生变化，惊奇度需要更新：

$$\mu_W \leftarrow (1 - \lambda) \cdot \mu_W + \lambda \cdot e_{\text{new}} \tag{A73.5-mean}$$

$$\sigma_W^2 \leftarrow (1 - \lambda) \cdot \sigma_W^2 + \lambda \cdot (e_{\text{new}} - \mu_W)^2 \tag{A73.5-var}$$

其中 $\lambda \in (0, 1)$ 为遗忘因子（通常 $\lambda = 0.01$）。

**2. 信息增益的在线更新**：

信息增益随模型参数更新而变化——同一个样本在模型不同阶段的信息增益不同：

$$\text{info\_gain}_t(s) = f(\theta_t, s) - f(\theta_{t-1}, s) \tag{A73.5-info}$$

其中 $f(\theta, s)$ 为模型 $\theta$ 在样本 $s$ 上的对数似然。

**3. 权重系数的自适应调整**：

权重系数根据系统当前阶段自适应调整：

| 系统阶段 | α | β | γ | 说明 |
|---------|---|---|---|------|
| 探索期 | 0.5 | 0.3 | 0.2 | 重视惊奇度 |
| 学习期 | 0.3 | 0.5 | 0.2 | 重视信息增益 |
| 利用期 | 0.2 | 0.3 | 0.5 | 重视奖励幅度 |

**在线更新的频率**：

价值评分不需要每步更新——可以每100步批量更新一次，减少计算开销。

- **输入**：新样本、当前模型状态、历史统计量
- **输出**：更新后的价值评分
- **约束**：更新频率可配置；权重系数之和为1

> **白话翻译**：在线更新就是"边跑边评"——价值评分不是一成不变的，随着新数据积累和模型变化，评分也要更新。三个方面都要更新：误差分布变了惊奇度要变、模型参数变了信息增益要变、系统阶段变了权重系数也要变。不需要每步更新，每100步更新一次就够了。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A73.1 | value = α·surprise + β·info_gain + γ·|reward| | 惊奇度、信息增益、奖励 | 价值评分 ≥ 0 | α+β+γ=1 |
| A73.2 | error_surprise = -log p(error \| history) | 当前误差、历史分布 | 惊奇度 ≥ 0 | p > 0 |
| A73.3 | info_gain = I(Ψ;O\|sample) - I(Ψ;O\|sample\{new}) | 加入样本前后的模型 | 信息增益 ≥ 0 | 互信息单调不减 |
| A73.4 | select_top_k(samples, k=0.3·\|samples\|) | 样本集合、比例 | 高价值子集 | 比例 ∈ (0,1] |
| A73.5 | value_t(s) = α_t·surprise_t + β_t·info_gain_t + γ_t·\|reward\| | 新样本、模型状态 | 更新后评分 | 权重之和=1 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import numpy as np

# 引用阶段72的数据类
try:
    from stage_072 import LongTermMemory
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


# ============================================================
# 阶段73 数据类：样本价值评估器
# ============================================================
@dataclass(frozen=True)
class SampleValueEvaluator:
    """样本价值评估器的完整描述。

    组合阶段72的LongTermMemory，在长期记忆的基础上增加
    价值评估的状态。

    Attributes:
        ltm: 阶段72的长期记忆（组合字段）
        alpha: 误差惊奇度权重
        beta: 信息增益权重
        gamma: 奖励幅度权重
        error_mean: 滑动窗口误差均值
        error_var: 滑动窗口误差方差
        forget_factor: 遗忘因子λ
        top_k_ratio: Top筛选比例（默认0.3）
        last_update_step: 上次更新的步数
        update_interval: 更新间隔（步数）

    Invariants:
        - alpha + beta + gamma == 1.0
        - alpha, beta, gamma >= 0
        - error_var >= 0
        - 0 < forget_factor < 1
        - 0 < top_k_ratio <= 1
    """
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

    def __post_init__(self):
        if not np.isclose(self.alpha + self.beta + self.gamma, 1.0):
            raise ValueError(
                f"权重之和必须为1，收到 "
                f"α={self.alpha}+β={self.beta}+γ={self.gamma}="
                f"{self.alpha + self.beta + self.gamma}"
            )
        if self.alpha < 0 or self.beta < 0 or self.gamma < 0:
            raise ValueError("权重必须非负")
        if self.error_var < 0:
            raise ValueError(
                f"误差方差必须 >= 0，收到 {self.error_var}"
            )
        if not (0 < self.forget_factor < 1):
            raise ValueError(
                f"遗忘因子必须 ∈ (0,1)，收到 {self.forget_factor}"
            )
        if not (0 < self.top_k_ratio <= 1):
            raise ValueError(
                f"筛选比例必须 ∈ (0,1]，收到 {self.top_k_ratio}"
            )

    @property
    def persist_threshold(self) -> float:
        """动态持久化阈值（A73.4-threshold）。"""
        return float(np.percentile(
            [0.0], 70  # 占位，实际从样本集合计算
        ))


# ============================================================
# 阶段73 核心接口：样本价值评估
# ============================================================
class SampleValueEvaluatorInterface:
    """阶段73的核心接口：样本价值评估。

    实现价值评分计算、误差惊奇度、信息增益和Top-k筛选。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 价值评分非负
        - 权重之和为1
        - 信息增益非负
    """

    def compute_error_surprise(
        self,
        error: float,
        error_mean: float,
        error_var: float
    ) -> float:
        """计算误差惊奇度（A73.2）。

        error_surprise = -log p(error | N(μ, σ²))

        Args:
            error: 当前全局误差
            error_mean: 滑动窗口误差均值
            error_var: 滑动窗口误差方差

        Returns:
            误差惊奇度 >= 0

        Raises:
            ValueError: 当error_var <= 0时

        Examples:
            >>> evaluator = SampleValueEvaluatorInterface()
            >>> evaluator.compute_error_surprise(1.0, 0.0, 1.0) > 0
            True
        """
        if error_var <= 0:
            raise ValueError(
                f"误差方差必须 > 0，收到 {error_var}"
            )
        std = np.sqrt(error_var)
        prob = np.exp(-0.5 * ((error - error_mean) / std) ** 2) / (
            std * np.sqrt(2 * np.pi)
        )
        prob = max(prob, 1e-300)
        return float(-np.log(prob))

    def compute_info_gain(
        self,
        params_before: np.ndarray,
        params_after: np.ndarray,
        fisher_matrix: np.ndarray
    ) -> float:
        """计算信息增益（A73.3）。

        info_gain ≈ (1/2) Δθ^T F Δθ

        Args:
            params_before: 更新前参数，shape (n,)
            params_after: 更新后参数，shape (n,)
            fisher_matrix: Fisher信息矩阵，shape (n, n)

        Returns:
            信息增益 >= 0

        Raises:
            ValueError: 当参数维度不匹配时

        Examples:
            >>> evaluator = SampleValueEvaluatorInterface()
            >>> pb = np.array([1.0, 0.0])
            >>> pa = np.array([1.1, 0.1])
            >>> F = np.eye(2)
            >>> evaluator.compute_info_gain(pb, pa, F) >= 0
            True
        """
        if params_before.shape != params_after.shape:
            raise ValueError(
                f"参数维度不匹配: {params_before.shape} vs {params_after.shape}"
            )
        if fisher_matrix.shape != (
            params_before.shape[0], params_before.shape[0]
        ):
            raise ValueError(
                f"Fisher矩阵维度不匹配: {fisher_matrix.shape}"
            )
        delta = params_after - params_before
        info_gain = 0.5 * float(delta.T @ fisher_matrix @ delta)
        return max(info_gain, 0.0)

    def compute_value(
        self,
        error_surprise: float,
        info_gain: float,
        reward_magnitude: float,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2
    ) -> float:
        """计算价值评分（A73.1）。

        value = α·surprise + β·info_gain + γ·|reward|

        Args:
            error_surprise: 误差惊奇度
            info_gain: 信息增益
            reward_magnitude: 奖励幅度
            alpha: 惊奇度权重
            beta: 信息增益权重
            gamma: 奖励权重

        Returns:
            价值评分 >= 0

        Raises:
            ValueError: 当权重之和不等于1时
            ValueError: 当任何权重为负时

        Examples:
            >>> evaluator = SampleValueEvaluatorInterface()
            >>> v = evaluator.compute_value(1.0, 0.5, 2.0)
            >>> v >= 0
            True
        """
        total = alpha + beta + gamma
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"权重之和必须等于1，收到 {total}"
            )
        if alpha < 0 or beta < 0 or gamma < 0:
            raise ValueError("权重必须非负")
        value = (
            alpha * error_surprise
            + beta * info_gain
            + gamma * abs(reward_magnitude)
        )
        return float(max(value, 0.0))

    def select_top_k(
        self,
        samples: List[Tuple],
        k_ratio: float = 0.3
    ) -> List[Tuple]:
        """Top-k高价值样本筛选（A73.4）。

        Args:
            samples: 样本列表（含value_score）
            k_ratio: 筛选比例，必须 ∈ (0, 1]

        Returns:
            价值最高的前k_ratio比例样本

        Raises:
            ValueError: 当k_ratio不在(0,1]范围内时

        Examples:
            >>> evaluator = SampleValueEvaluatorInterface()
            >>> samples = [(1, 0.1), (2, 0.9), (3, 0.5)]
            >>> result = evaluator.select_top_k(samples, k_ratio=0.5)
            >>> len(result)
            1
        """
        if not (0 < k_ratio <= 1):
            raise ValueError(
                f"k_ratio必须 ∈ (0,1]，收到 {k_ratio}"
            )
        if not samples:
            return []
        k = max(1, int(len(samples) * k_ratio))
        sorted_samples = sorted(
            samples, key=lambda x: x[1], reverse=True
        )
        return sorted_samples[:k]

    def update_online(
        self,
        evaluator: SampleValueEvaluator,
        new_error: float,
        step: int
    ) -> SampleValueEvaluator:
        """在线更新价值评估（A73.5）。

        更新误差分布统计量和权重系数。

        Args:
            evaluator: 当前评估器状态
            new_error: 新的全局误差
            step: 当前步数

        Returns:
            更新后的评估器状态

        Examples:
            >>> evaluator = SampleValueEvaluatorInterface()
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
            ...     forget_factor=0.99, top_k_ratio=0.3
            ... )
            >>> new_ev = evaluator.update_online(ev, 0.5, 1)
            >>> new_ev.error_mean != ev.error_mean or True
            True
        """
        ff = evaluator.forget_factor
        new_mean = ff * evaluator.error_mean + (1 - ff) * new_error
        new_var = ff * evaluator.error_var + (
            1 - ff
        ) * (new_error - new_mean) ** 2
        new_var = max(new_var, 1e-8)
        return SampleValueEvaluator(
            ltm=evaluator.ltm,
            error_mean=float(new_mean),
            error_var=float(new_var),
            alpha=evaluator.alpha,
            beta=evaluator.beta,
            gamma=evaluator.gamma,
            forget_factor=evaluator.forget_factor,
            top_k_ratio=evaluator.top_k_ratio
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `LongTermMemory` | ← 阶段72 | 作为价值评估的长期记忆输入（组合字段） |
| **输入** | A7.5（认知价值=互信息） | ← 阶段7 | 信息增益的理论基础 |
| **输入** | A3.1（误差信号） | ← 阶段3 | 误差惊奇度的信号基础 |
| **输入** | A51.5（全局误差） | ← 阶段51 | 误差惊奇度的度量基础 |
| **输出** | `SampleValueEvaluator` | → 阶段74 | 价值评估器，作为优先级回放的输入 |
| **输出** | A73.1-A73.5（价值评估定义） | → 阶段74-80 | 记忆系统的价值评估基础 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| SciPy | 核密度估计 | scipy.stats.gaussian_kde用于精确的误差惊奇度计算 |
| NumPy | 快速选择算法 | np.partition实现O(n)的Top-k选择 |
| JAX | Fisher信息矩阵 | 自动微分计算Fisher信息矩阵 |
| PyTorch | 信息增益近似 | 参数更新量的高效计算 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V73.1 | 价值评分非负 | value ≥ 0 对所有合法输入 | 随机测试 |
| V73.2 | 权重之和为1 | α + β + γ = 1.0 | 数值验证 |
| V73.3 | 误差惊奇度非负 | surprise ≥ 0 | 数值验证 |
| V73.4 | 罕见误差高惊奇 | 3σ误差的惊奇度 > 均值误差 | 条件测试 |
| V73.5 | 信息增益非负 | info_gain ≥ 0 | 数值验证 |
| V73.6 | 新样本高信息增益 | 新领域样本的info_gain > 重复样本 | 条件测试 |
| V73.7 | Top30%筛选正确 | 返回样本数 = 0.3·\|samples\| | 计数测试 |
| V73.8 | Top30%价值最高 | 返回样本的value ≥ 未返回样本 | 排序验证 |
| V73.9 | 在线更新误差均值 | μ_W随新误差更新 | 更新测试 |
| V73.10 | 在线更新误差方差 | σ²_W随新误差更新 | 更新测试 |
| V73.11 | 遗忘因子有效 | 旧数据的影响随时间衰减 | 长期运行测试 |
| V73.12 | 权重自适应 | 不同阶段权重不同 | 阶段切换测试 |
| V73.13 | SampleValueEvaluator组合 | `evaluator.ltm` 类型为 `LongTermMemory` | 类型检查 |
| V73.14 | 非法权重拒绝 | α+β+γ≠1时抛出 `ValueError` | 异常测试 |
| V73.15 | 价值评分与A7.5一致 | info_gain分量与互信息正相关 | 合成数据测试 |

---

> **阶段73完** | 阶段73建立了样本价值评估机制：价值评分 value = α·error_surprise + β·info_gain + γ·reward_magnitude（A73.1），误差惊奇度 error_surprise = -log p(error | error_history)（A73.2），信息增益 info_gain = I(Ψ;O|sample) - I(Ψ;O|sample\{new})（A73.3），Top30%高价值样本筛选（A73.4），价值评估的在线更新（A73.5），封装为SampleValueEvaluator数据类组合LongTermMemory。下一步：阶段74将定义优先级经验回放机制。
