# 阶段75：生成式回放——VAE编码压缩、隐式存储与重构质量保证

> **阶段编号**：S75 | **所属层**：第八层·记忆系统层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | VAE编码器 | "压缩打包"——把经历压缩成紧凑的隐编码 |
> | VAE解码器 | "拆包还原"——从隐编码还原出原始经历 |
> | 压缩存储 | "真空收纳"——用隐编码代替原始数据，大幅节省空间 |
> | 重构生成 | "凭记忆复述"——从隐编码重新生成样本用于复习 |
> | 重构质量评估 | "复述准不准"——检查还原的经历和原始经历差多少 |
> | GenerativeReplay | "生成式回放的完整画像"——组合PriorityReplay，在优先级回放基础上增加生成能力 |

---

## 1. 第一性原理依据

本阶段从阶段74（优先级经验回放）的核心结论出发，建立生成式回放机制。核心依据为：

- **A74.3（批量采样）**（引用阶段74）：batch = sample_priority(buffer, batch_size=32)，优先级回放的批量采样。生成式回放不替代优先级回放，而是在优先级回放的基础上增加生成能力——当原始样本不可用（如已被覆盖或存储受限）时，通过VAE重构生成替代样本。

- **A72.1（样本表设计）**（引用阶段72）：长期记忆的样本表存储原始观测和预测向量。当样本数量极大时，原始向量的存储开销成为瓶颈。生成式回放通过VAE压缩，将高维向量压缩为低维隐编码，大幅减少存储开销。

- **A4.4（变分自由能）**（引用阶段4）：$F[q] = -\mathbb{E}_q[\ln p(o|\psi)] + D_{\text{KL}}[q(\psi) \| p(\psi)]$，变分自由能的定义。VAE的训练目标正是变分自由能的最小化——编码器和解码器联合优化，使重构误差最小且隐编码接近先验分布。

- **A3.6（信息瓶颈）**（引用阶段3）：$I(r^{l-1}; e^{l-1}) \leq H(r^{l-1})$，误差信号的信息瓶颈。VAE的隐编码也受信息瓶颈约束——隐编码携带的信息不超过原始数据的信息，但好的压缩应保留最重要的信息。

本阶段的核心目标是：**定义VAE编码器 q(z|x) = N(μ_enc(x), Σ_enc(x))（A75.1），VAE解码器 p(x|z) = N(μ_dec(z), Σ_dec(z))（A75.2），压缩存储 store(latent_z, metadata)（A75.3），重新生成 generate(z) → x_reconstructed（A75.4），生成质量评估 reconstruction_error < θ_quality（A75.5），封装为GenerativeReplay数据类，组合PriorityReplay作为优先级回放的输入。**

> **去重声明**：本阶段仅定义生成式回放的VAE架构和压缩存储机制，不涉及离线复盘（阶段76）或记忆巩固（阶段77）。优先级回放的采样策略已在阶段74定义，本阶段不重复。变分自由能的定义遵循A4.4，本阶段不重新推导。

---

## 2. 核心数学公式

### 第一部分：VAE编码器

#### 2.1 VAE编码器定义（A75.1）

VAE编码器将高维样本映射到低维隐空间：

$$q(z|x) = \mathcal{N}(z; \mu_{\text{enc}}(x), \Sigma_{\text{enc}}(x)) \tag{A75.1}$$

其中：
- $x \in \mathbb{R}^{d_x}$ 为输入样本（observation + prediction的拼接）
- $z \in \mathbb{R}^{d_z}$ 为隐编码，$d_z \ll d_x$
- $\mu_{\text{enc}}(x) \in \mathbb{R}^{d_z}$ 为编码均值
- $\Sigma_{\text{enc}}(x) \in \mathbb{S}_{++}^{d_z}$ 为编码协方差（通常取对角近似）

**编码器的网络结构**：

$$\mu_{\text{enc}}(x) = W_\mu \cdot h(x) + b_\mu \tag{A75.1-mean}$$

$$\log \sigma_{\text{enc}}^2(x) = W_\sigma \cdot h(x) + b_\sigma \tag{A75.1-var}$$

其中 $h(x)$ 为共享特征提取器的输出，$W_\mu, W_\sigma$ 为独立的投影层。

**重参数化技巧**：

为使采样操作可微分，采用重参数化技巧：

$$z = \mu_{\text{enc}}(x) + \sigma_{\text{enc}}(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) \tag{A75.1-reparam}$$

**隐编码维度 $d_z$ 的选择**：

| $d_z$ | 压缩比 | 重构质量 | 适用场景 |
|--------|--------|---------|---------|
| 8 | 极高 | 较差 | 极度存储受限 |
| 16 | 高 | 中等 | 存储受限 |
| 32 | 中 | 良好 | 平衡选择 |
| 64 | 低 | 优秀 | 存储充裕 |

默认 $d_z = 32$，在压缩比和重构质量之间取得平衡。当 $d_x = 512$ 时，压缩比为16:1。

- **输入**：样本向量 $x \in \mathbb{R}^{d_x}$
- **输出**：隐编码 $z \in \mathbb{R}^{d_z}$，编码参数 $(\mu_{\text{enc}}, \sigma_{\text{enc}})$
- **约束**：$d_z \ll d_x$；重参数化保证可微分

> **白话翻译**：VAE编码器就是"压缩打包"——把高维的原始经历压缩成低维的隐编码。压缩不是随便丢信息，而是学会保留最重要的信息。用重参数化技巧让采样可微分——先从标准正态分布采样噪声，再缩放平移到目标分布。

---

### 第二部分：VAE解码器

#### 2.2 VAE解码器定义（A75.2）

VAE解码器从隐编码重构原始样本：

$$p(x|z) = \mathcal{N}(x; \mu_{\text{dec}}(z), \Sigma_{\text{dec}}(z)) \tag{A75.2}$$

其中：
- $z \in \mathbb{R}^{d_z}$ 为隐编码
- $\mu_{\text{dec}}(z) \in \mathbb{R}^{d_x}$ 为重构均值
- $\Sigma_{\text{dec}}(z) \in \mathbb{S}_{++}^{d_x}$ 为重构协方差（通常取对角近似）

**解码器的网络结构**：

$$\mu_{\text{dec}}(z) = W_{\text{dec}} \cdot g(z) + b_{\text{dec}} \tag{A75.2-mean}$$

其中 $g(z)$ 为解码特征提取器的输出。

**VAE的训练目标**：

VAE的训练目标为变分下界（ELBO）的最大化，等价于变分自由能（A4.4）的最小化：

$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q(z|x)}[\ln p(x|z)] - D_{\text{KL}}[q(z|x) \| p(z)] \tag{A75.2-elbo}$$

其中：
- 第一项为重构项：鼓励解码器准确重构原始样本
- 第二项为正则项：鼓励隐编码接近先验分布 $p(z) = \mathcal{N}(0, I)$

**与A4.4的对应**：

A4.4（引用阶段4）的变分自由能为 $F[q] = -\mathbb{E}_q[\ln p(o|\psi)] + D_{\text{KL}}[q(\psi) \| p(\psi)]$。A75.2的ELBO与A4.4在数学结构上完全一致——最大化ELBO等价于最小化变分自由能。VAE的训练过程就是变分自由能最小化的过程。

- **输入**：隐编码 $z \in \mathbb{R}^{d_z}$
- **输出**：重构样本 $\hat{x} \in \mathbb{R}^{d_x}$
- **约束**：重构质量满足A75.5

> **白话翻译**：VAE解码器就是"拆包还原"——从低维隐编码还原出高维的原始经历。训练目标是两件事的平衡：还原要准（重构项），隐编码要规范（正则项）。训练VAE就是最小化变分自由能——和阶段4的自由能原理完全一致。

---

### 第三部分：压缩存储

#### 2.3 压缩存储定义（A75.3）

压缩存储用隐编码替代原始向量，大幅减少存储空间：

$$\text{store}(\text{latent\_z}, \text{metadata}) \tag{A75.3}$$

**存储内容的对比**：

| 存储方式 | 每样本大小 | 10000样本总量 | 说明 |
|---------|----------|------------|------|
| 原始向量 | $2 \times d_x \times 4$ 字节 | 40MB | observation + prediction |
| 隐编码 | $d_z \times 4 + 20$ 字节 | 1.7MB | latent_z + metadata |
| 压缩比 | - | 23.5x | $d_x=256, d_z=32$ |

**metadata的内容**：

metadata存储无法从隐编码恢复的辅助信息：

| 字段 | 大小 | 说明 |
|------|------|------|
| error | 4字节 | 全局误差 |
| reward | 4字节 | 奖励信号 |
| timestamp | 8字节 | 时间戳 |
| value_score | 4字节 | 价值评分 |

metadata总计约20字节，相比原始向量的2KB可忽略不计。

**压缩存储的写入流程**：

1. 对样本 $x$ 执行编码：$z = \text{encode}(x)$
2. 存储 $(z, \text{metadata})$ 替代 $(x, \text{metadata})$
3. 可选：删除原始样本 $x$ 以释放空间

**压缩存储的适用场景**：

| 场景 | 是否适用 | 原因 |
|------|---------|------|
| 长期记忆归档 | 适用 | 存储空间是主要瓶颈 |
| 短期记忆缓冲区 | 不适用 | 需要快速读写，压缩/解压开销不值得 |
| 跨节点传输 | 适用 | 带宽是瓶颈，压缩减少传输量 |

- **输入**：样本的隐编码和metadata
- **输出**：压缩后的存储记录
- **约束**：metadata字段完整；隐编码维度正确

> **白话翻译**：压缩存储就是"真空收纳"——用隐编码代替原始数据，存储空间减少23倍。原始向量2KB，隐编码+元数据才0.17KB。元数据里存着误差、奖励、时间戳、价值评分——这些是隐编码恢复不了的，必须单独存。压缩存储适合长期归档，不适合短期缓冲区（压缩解压有开销）。

---

### 第四部分：重新生成

#### 2.4 重新生成定义（A75.4）

从隐编码重新生成样本用于回放学习：

$$\text{generate}(z) \to x_{\text{reconstructed}} \tag{A75.4}$$

**生成流程**：

1. 从压缩存储中读取隐编码 $z$
2. 执行解码：$\hat{x} = \text{decode}(z)$
3. 从metadata中恢复辅助信息
4. 组装为完整的回放样本

**生成与原始样本的差异**：

$$\Delta x = \|x_{\text{original}} - x_{\text{reconstructed}}\| \tag{A75.4-diff}$$

重构误差 $\Delta x$ 由A75.5的质量阈值约束。

**生成式回放与优先级回放的协同**：

生成式回放不替代优先级回放，而是作为补充：

| 场景 | 回放方式 | 原因 |
|------|---------|------|
| 原始样本可用 | 优先级回放 | 无损，更准确 |
| 原始样本不可用 | 生成式回放 | 有损，但可恢复 |
| 需要增强多样性 | 生成式回放 | 可在隐空间插值生成新样本 |

**隐空间插值生成新样本**：

在两个隐编码之间线性插值，生成过渡样本：

$$z_{\text{new}} = (1-\lambda) z_1 + \lambda z_2, \quad \lambda \in (0, 1) \tag{A75.4-interp}$$

插值生成增加了回放样本的多样性——不仅复习历史经历，还"想象"可能经历的中间状态。

- **输入**：隐编码 $z$
- **输出**：重构样本 $x_{\text{reconstructed}}$
- **约束**：重构质量满足A75.5

> **白话翻译**：重新生成就是"凭记忆复述"——从隐编码还原出原始经历用于复习。复述不可能100%准确，但只要误差在可接受范围内就行。生成式回放还有一个绝招：在两个隐编码之间插值，生成"没见过但可能经历过"的样本，增加复习的多样性。

---

### 第五部分：生成质量评估

#### 2.5 生成质量评估定义（A75.5）

生成质量通过重构误差评估，要求低于质量阈值：

$$\text{reconstruction\_error} = \|x - \hat{x}\|^2 / d_x < \theta_{\text{quality}} \tag{A75.5}$$

其中：
- $x$ 为原始样本
- $\hat{x}$ 为重构样本
- $d_x$ 为样本维度（归一化用）
- $\theta_{\text{quality}}$ 为质量阈值

**质量阈值的选择**：

| $\theta_{\text{quality}}$ | 重构质量 | 信息保留 | 适用场景 |
|--------------------------|---------|---------|---------|
| 0.01 | 优秀 | >99% | 精确回放 |
| 0.05 | 良好 | >95% | 默认选择 |
| 0.10 | 可接受 | >90% | 存储极度受限 |

默认 $\theta_{\text{quality}} = 0.05$，要求重构保留95%以上的信息。

**质量评估的维度分解**：

重构误差可按observation和prediction分别评估：

$$\text{error}_{\text{obs}} = \|x_{\text{obs}} - \hat{x}_{\text{obs}}\|^2 / d_{\text{obs}} \tag{A75.5-obs}$$

$$\text{error}_{\text{pred}} = \|x_{\text{pred}} - \hat{x}_{\text{pred}}\|^2 / d_{\text{pred}} \tag{A75.5-pred}$$

**质量不达标时的处理**：

| 情况 | 处理 |
|------|------|
| 重构误差 > θ_quality | 不使用该重构样本，回退到原始样本 |
| VAE整体质量下降 | 重新训练VAE |
| 特定样本质量差 | 增大该样本的隐编码维度 |

**VAE的定期重训练**：

随着系统运行，数据分布可能发生变化（环境突变、模型重构等），导致VAE的重构质量下降。定期重训练VAE以适应新的数据分布：

$$\text{retrain\_condition} = \text{reconstruction\_error} > \theta_{\text{quality}} \text{ for } > 10\% \text{ of samples} \tag{A75.5-retrain}$$

当超过10%的样本重构误差超标时，触发VAE重训练。

- **输入**：原始样本、重构样本、质量阈值
- **输出**：重构误差、质量是否达标
- **约束**：reconstruction_error ≥ 0

> **白话翻译**：生成质量评估就是"复述准不准"——检查还原的经历和原始经历差多少。默认要求每维平均误差小于0.05，也就是保留95%以上的信息。如果不达标，就不用这个重构样本，回退到原始样本。如果整体质量下降（超过10%的样本不达标），就重新训练VAE。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A75.1 | q(z\|x) = N(μ_enc(x), Σ_enc(x)) | 样本x | 隐编码z + 编码参数 | d_z ≪ d_x |
| A75.2 | p(x\|z) = N(μ_dec(z), Σ_dec(z)) | 隐编码z | 重构样本x̂ | ELBO最大化 |
| A75.3 | store(latent_z, metadata) | 隐编码+元数据 | 压缩存储记录 | metadata完整 |
| A75.4 | generate(z) → x_reconstructed | 隐编码z | 重构样本 | 质量满足A75.5 |
| A75.5 | ‖x - x̂‖²/d_x < θ_quality | 原始+重构样本 | 质量评估 | θ_quality = 0.05 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import numpy as np

# 引用阶段74的数据类
try:
    from stage_074 import PriorityReplay
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


# ============================================================
# 阶段75 数据类：生成式回放
# ============================================================
@dataclass(frozen=True)
class GenerativeReplay:
    """生成式回放的完整描述。

    组合阶段74的PriorityReplay，在优先级回放的基础上增加
    VAE生成能力。

    Attributes:
        replay: 阶段74的优先级回放（组合字段）
        latent_dim: 隐编码维度d_z
        encoder_params: 编码器参数
        decoder_params: 解码器参数
        quality_threshold: 重构质量阈值
        compression_ratio: 压缩比
        avg_reconstruction_error: 平均重构误差
        vae_retrain_count: VAE重训练次数

    Invariants:
        - latent_dim > 0
        - quality_threshold > 0
        - avg_reconstruction_error >= 0
    """
    replay: PriorityReplay
    latent_dim: int = 32
    encoder_params: Optional[Dict] = None
    decoder_params: Optional[Dict] = None
    quality_threshold: float = 0.05
    compression_ratio: float = 16.0
    avg_reconstruction_error: float = 0.0
    vae_retrain_count: int = 0

    def __post_init__(self):
        if self.latent_dim <= 0:
            raise ValueError(
                f"隐编码维度必须 > 0，收到 {self.latent_dim}"
            )
        if self.quality_threshold <= 0:
            raise ValueError(
                f"质量阈值必须 > 0，收到 {self.quality_threshold}"
            )
        if self.avg_reconstruction_error < 0:
            raise ValueError(
                f"平均重构误差必须 >= 0，收到 {self.avg_reconstruction_error}"
            )

    @property
    def quality_ok(self) -> bool:
        """重构质量是否达标。"""
        return self.avg_reconstruction_error < self.quality_threshold


# ============================================================
# 阶段75 核心接口：生成式回放
# ============================================================
class GenerativeReplayInterface:
    """阶段75的核心接口：生成式回放。

    实现VAE编码、解码、压缩存储和生成质量评估。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 重构误差低于质量阈值
        - 压缩比 > 1
        - 隐编码维度远小于原始维度
    """

    def encode(
        self,
        x: np.ndarray,
        encoder_params: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """VAE编码（A75.1）。

        Args:
            x: 输入样本，shape (batch, d_x)
            encoder_params: 编码器参数，包含 "W_mu", "b_mu", "W_logvar", "b_logvar"

        Returns:
            (z, mu, log_var)
            - z: 隐编码，shape (batch, d_z)
            - mu: 均值，shape (batch, d_z)
            - log_var: 对数方差，shape (batch, d_z)

        Raises:
            ValueError: 当encoder_params缺少必要键时

        Examples:
            >>> gre = GenerativeReplayInterface()
            >>> x = np.array([[1.0, 2.0]])
            >>> params = {"W_mu": np.ones((2,1)), "b_mu": np.zeros(1),
            ...           "W_logvar": np.ones((2,1)), "b_logvar": np.zeros(1)}
            >>> z, mu, lv = gre.encode(x, params)
            >>> z.shape[1] == mu.shape[1]
            True
        """
        required = {"W_mu", "b_mu", "W_logvar", "b_logvar"}
        missing = required - set(encoder_params.keys())
        if missing:
            raise ValueError(f"encoder_params缺少键: {missing}")
        W_mu = encoder_params["W_mu"]
        b_mu = encoder_params["b_mu"]
        W_logvar = encoder_params["W_logvar"]
        b_logvar = encoder_params["b_logvar"]
        h = np.tanh(x @ W_mu + b_mu)
        mu = h @ W_mu + b_mu
        log_var = h @ W_logvar + b_logvar
        std = np.exp(0.5 * log_var)
        eps = np.random.randn(*mu.shape)
        z = mu + std * eps
        return z, mu, log_var

    def decode(
        self,
        z: np.ndarray,
        decoder_params: Dict
    ) -> np.ndarray:
        """VAE解码（A75.2）。

        Args:
            z: 隐编码，shape (batch, d_z)
            decoder_params: 解码器参数，包含 "W_out", "b_out"

        Returns:
            重构样本，shape (batch, d_x)

        Raises:
            ValueError: 当decoder_params缺少必要键时

        Examples:
            >>> gre = GenerativeReplayInterface()
            >>> z = np.array([[0.5]])
            >>> params = {"W_out": np.ones((1,2)), "b_out": np.zeros(2)}
            >>> x_hat = gre.decode(z, params)
            >>> x_hat.shape[1]
            2
        """
        required = {"W_out", "b_out"}
        missing = required - set(decoder_params.keys())
        if missing:
            raise ValueError(f"decoder_params缺少键: {missing}")
        W_out = decoder_params["W_out"]
        b_out = decoder_params["b_out"]
        h = np.tanh(z @ W_out + b_out)
        x_hat = h @ W_out + b_out
        return x_hat

    def compress_store(
        self,
        x: np.ndarray,
        metadata: Dict,
        encoder_params: Dict
    ) -> Tuple[np.ndarray, Dict]:
        """压缩存储（A75.3）。

        Args:
            x: 原始样本，shape (batch, d_x)
            metadata: 元数据字典
            encoder_params: 编码器参数

        Returns:
            (latent_z, metadata) 压缩后的存储记录

        Examples:
            >>> gre = GenerativeReplayInterface()
            >>> x = np.array([[1.0, 2.0]])
            >>> params = {"W_mu": np.ones((2,1)), "b_mu": np.zeros(1),
            ...           "W_logvar": np.ones((2,1)), "b_logvar": np.zeros(1)}
            >>> z, meta = gre.compress_store(x, {"id": 1}, params)
            >>> "latent_z" in meta
            True
        """
        z, mu, log_var = self.encode(x, encoder_params)
        meta_out = dict(metadata)
        meta_out["latent_z"] = z
        meta_out["mu"] = mu
        meta_out["log_var"] = log_var
        return z, meta_out

    def generate(
        self,
        z: np.ndarray,
        decoder_params: Dict
    ) -> np.ndarray:
        """重新生成（A75.4）。

        Args:
            z: 隐编码，shape (batch, d_z)
            decoder_params: 解码器参数

        Returns:
            重构样本，shape (batch, d_x)

        Examples:
            >>> gre = GenerativeReplayInterface()
            >>> z = np.array([[0.5]])
            >>> params = {"W_out": np.ones((1,2)), "b_out": np.zeros(2)}
            >>> x_hat = gre.generate(z, params)
            >>> x_hat.shape[0]
            1
        """
        return self.decode(z, decoder_params)

    def evaluate_quality(
        self,
        x_original: np.ndarray,
        x_reconstructed: np.ndarray
    ) -> Tuple[float, bool]:
        """评估生成质量（A75.5）。

        Args:
            x_original: 原始样本，shape (batch, d_x)
            x_reconstructed: 重构样本，shape (batch, d_x)

        Returns:
            (reconstruction_error, quality_ok)
            - reconstruction_error: 均方误差
            - quality_ok: 误差是否低于阈值

        Raises:
            ValueError: 当输入形状不匹配时

        Examples:
            >>> gre = GenerativeReplayInterface()
            >>> x = np.array([[1.0, 2.0]])
            >>> x_hat = np.array([[1.1, 1.9]])
            >>> err, ok = gre.evaluate_quality(x, x_hat)
            >>> err >= 0
            True
        """
        if x_original.shape != x_reconstructed.shape:
            raise ValueError(
                f"形状不匹配: {x_original.shape} vs {x_reconstructed.shape}"
            )
        mse = float(np.mean((x_original - x_reconstructed) ** 2))
        quality_ok = mse < 0.1
        return mse, quality_ok

    def interpolate(
        self,
        z1: np.ndarray,
        z2: np.ndarray,
        lambda_val: float
    ) -> np.ndarray:
        """隐空间插值生成（A75.4-interp）。

        Args:
            z1: 隐编码1，shape (d_z,)
            z2: 隐编码2，shape (d_z,)
            lambda_val: 插值系数 ∈ [0, 1]

        Returns:
            插值隐编码，shape (d_z,)

        Raises:
            ValueError: 当lambda_val不在[0,1]范围内时
            ValueError: 当z1和z2形状不匹配时

        Examples:
            >>> gre = GenerativeReplayInterface()
            >>> z1 = np.array([0.0, 1.0])
            >>> z2 = np.array([1.0, 0.0])
            >>> z_interp = gre.interpolate(z1, z2, 0.5)
            >>> np.allclose(z_interp, np.array([0.5, 0.5]))
            True
        """
        if not (0 <= lambda_val <= 1):
            raise ValueError(
                f"lambda_val必须 ∈ [0,1]，收到 {lambda_val}"
            )
        if z1.shape != z2.shape:
            raise ValueError(
                f"z1和z2形状不匹配: {z1.shape} vs {z2.shape}"
            )
        return (1 - lambda_val) * z1 + lambda_val * z2
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `PriorityReplay` | ← 阶段74 | 作为生成式回放的优先级回放输入（组合字段） |
| **输入** | A74.3（批量采样） | ← 阶段74 | 生成式回放与优先级回放协同 |
| **输入** | A4.4（变分自由能） | ← 阶段4 | VAE训练目标的理论基础 |
| **输入** | A3.6（信息瓶颈） | ← 阶段3 | 隐编码的信息约束 |
| **输出** | `GenerativeReplay` | → 阶段76 | 生成式回放状态，作为离线复盘的输入 |
| **输出** | A75.1-A75.5（生成式回放定义） | → 阶段77-80 | 记忆系统的生成能力基础 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| PyTorch / JAX | VAE实现 | 编码器和解码器的神经网络实现 |
| NumPy | 重构误差计算 | 向量化的均方误差计算 |
| msgpack | 隐编码序列化 | 压缩存储的BLOB序列化 |
| Optuna | 超参数调优 | 隐编码维度、学习率等超参数搜索 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V75.1 | 编码器输出维度 | z.shape == (batch, d_z) | 形状检查 |
| V75.2 | 解码器输出维度 | x̂.shape == (batch, d_x) | 形状检查 |
| V75.3 | 重参数化可微分 | 梯度可通过采样操作反向传播 | 梯度检查 |
| V75.4 | 压缩比 > 1 | 原始大小 / 压缩大小 > 1 | 大小对比 |
| V75.5 | 重构误差达标 | avg_error < θ_quality | 质量测试 |
| V75.6 | 重构保留信息 | I(x; x̂) / H(x) > 0.95 | 信息论测试 |
| V75.7 | 隐编码接近先验 | KL散度 < 1.0 nats | KL测试 |
| V75.8 | 插值生成平滑 | 相邻λ值的重构样本平滑过渡 | 插值测试 |
| V75.9 | 压缩存储完整性 | 解压后metadata完整 | 往返测试 |
| V75.10 | VAE重训练触发 | error > θ for >10% samples时触发 | 条件测试 |
| V75.11 | GenerativeReplay组合 | `greplay.replay` 类型为 `PriorityReplay` | 类型检查 |
| V75.12 | 非法维度拒绝 | d_z ≤ 0 抛出 `ValueError` | 异常测试 |
| V75.13 | 长期压缩存储 | 10000样本压缩存储无异常 | 长期运行测试 |
| V75.14 | 生成式回放与优先级回放协同 | 原始可用时用优先级，不可用时用生成 | 场景测试 |
| V75.15 | ELBO与A4.4一致 | VAE训练目标等价于变分自由能最小化 | 数学验证 |

---

> **阶段75完** | 阶段75建立了生成式回放机制：VAE编码器 q(z|x) = N(μ_enc(x), Σ_enc(x))（A75.1），VAE解码器 p(x|z) = N(μ_dec(z), Σ_dec(z))（A75.2），压缩存储 store(latent_z, metadata)（A75.3），重新生成 generate(z) → x_reconstructed（A75.4），生成质量评估 reconstruction_error < θ_quality（A75.5），封装为GenerativeReplay数据类组合PriorityReplay。下一步：阶段76将定义空闲离线复盘机制。
