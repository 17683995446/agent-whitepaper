# 阶段45：中层模型算法——RNN/GRU、因果发现与趋势检测

> **阶段编号**：S45 | **所属层**：第五层·层级预测层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | RNN/GRU状态空间模型 | "有记忆的预测器"——记住过去的状态，预测未来的趋势 |
> | 因果图发现（PC算法） | "找因果关系"——从数据中发现谁导致了谁，不是相关而是因果 |
> | 状态转移矩阵在线估计 | "学转移规律"——在线学习状态之间的转移概率 |
> | 趋势检测（CUSUM变体） | "发现拐点"——检测趋势何时发生了变化 |
> | MidLevelAlgorithm | "中层算法的完整画像"——组合MidLevelTarget，在目标基础上增加算法参数 |

---

## 1. 第一性原理依据

本阶段从阶段12（自下而上误差传播）和阶段14（表征更新规则）的核心结论出发，建立中层模型的算法框架。核心依据为：

- **A12.1（误差传播规则）**（引用阶段12）：$\delta^l = \Pi_l e^l + (\partial f^{l+1}/\partial r^l) \cdot \delta^{l+1}$，中层误差信号 $\delta^2$ 由底层误差 $\delta^1$ 经Jacobian映射传播而来。中层模型的预测误差 $e^2$ 是底层误差信号的汇总——当底层多个指标的误差同时增大时，中层检测到趋势变化或事件发生。

- **A14.1（表征更新方程）**（引用阶段14）：$\tau_r \dot{r}^l = \delta^{l+1} - e^l$，中层表征 $r^2$ 的更新由高层传播信号 $\delta^3$ 和中层误差 $e^2$ 驱动。中层算法需要在线更新RNN/GRU的隐状态和因果图的结构，以适应环境变化。

- **A44.1-A44.3（中层预测目标与参数）**（引用阶段44）：中层预测目标 $\mathbf{y}_{\text{mid}} = [\mathbf{y}_{\text{event}}, \mathbf{y}_{\text{trend}}, \mathbf{y}_{\text{cause}}]$，采集频率0.01Hz，预测窗口10-100步。

本阶段的核心目标是：**设计RNN/GRU状态空间模型用于中层序列预测、实现PC算法用于因果图发现、建立状态转移矩阵的在线估计方法、设计CUSUM变体用于趋势检测，并封装为MidLevelAlgorithm数据类，组合MidLevelTarget作为预测目标的输入。** 中层算法是层级预测体系的核心计算层——它将底层的微预测信号转化为趋势和因果结构，为高层提供结构化的预测输入。

---

## 2. 核心数学公式

### 第一部分：RNN/GRU状态空间模型

#### 2.1 RNN/GRU状态空间模型（A45.1）

中层预测使用GRU（Gated Recurrent Unit）作为状态空间模型的核心。GRU的隐状态 $\mathbf{h}_t \in \mathbb{R}^{d_h}$ 编码了中层的历史信息，用于预测未来的趋势和事件。

**GRU更新方程**：

$$\mathbf{z}_t = \sigma(W_z \mathbf{x}_t + U_z \mathbf{h}_{t-1} + \mathbf{b}_z) \tag{A45.1-z}$$

$$\mathbf{r}_t = \sigma(W_r \mathbf{x}_t + U_r \mathbf{h}_{t-1} + \mathbf{b}_r) \tag{A45.1-r}$$

$$\tilde{\mathbf{h}}_t = \tanh(W_h \mathbf{x}_t + U_h (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_h) \tag{A45.1-htilde}$$

$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t \tag{A45.1-h}$$

其中：
- $\mathbf{x}_t \in \mathbb{R}^{d_{\text{mid}}}$ 为中层输入（来自A44.1的中层预测目标）
- $\mathbf{z}_t$ 为更新门（控制新信息写入的比例）
- $\mathbf{r}_t$ 为重置门（控制历史信息的遗忘程度）
- $\tilde{\mathbf{h}}_t$ 为候选隐状态
- $\sigma$ 为sigmoid函数，$\odot$ 为逐元素乘法

**预测输出**：

$$\hat{\mathbf{y}}_{\text{mid}}(t+h) = W_{\text{out}} \mathbf{h}_t^{(h)} + \mathbf{b}_{\text{out}} \tag{A45.1-predict}$$

其中 $\mathbf{h}_t^{(h)}$ 为从 $\mathbf{h}_t$ 递推 $h$ 步的隐状态（将 $\hat{\mathbf{y}}_{\text{mid}}(t+k)$ 作为下一步输入）。

**与卡尔曼滤波的对应**：

GRU的隐状态 $\mathbf{h}_t$ 对应卡尔曼滤波的状态估计 $\hat{\mathbf{x}}_{t|t}$，GRU的更新门 $\mathbf{z}_t$ 对应卡尔曼增益 $K_t$——两者都控制"新信息写入多少"。区别在于：卡尔曼滤波是线性高斯的，GRU是非线性的，能处理更复杂的动力学。

- **输入**：中层目标序列 $\{\mathbf{x}_t\}$，GRU参数 $\{W_z, U_z, W_r, U_r, W_h, U_h, W_{\text{out}}\}$
- **输出**：隐状态 $\mathbf{h}_t$，多步预测 $\hat{\mathbf{y}}_{\text{mid}}(t+1), \ldots, \hat{\mathbf{y}}_{\text{mid}}(t+h)$
- **约束**：隐状态维度 $d_h$ 典型值64-256；梯度裁剪防止爆炸

> **白话翻译**：GRU就是"有记忆的预测器"——隐状态 $\mathbf{h}$ 记住了过去的信息，更新门控制"新信息写多少"，重置门控制"旧信息忘多少"。每来一个新输入，隐状态就更新一次，然后基于隐状态预测未来。和卡尔曼滤波类似，但GRU是非线性的，能处理更复杂的模式。

---

### 第二部分：因果图发现

#### 2.2 因果图发现——PC算法（A45.2）

PC算法（Peter-Clark算法）从观测数据中发现变量间的因果结构。设中层预测目标的各维度变量为 $X_1, \ldots, X_{d_{\text{mid}}}$，PC算法输出一个因果图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$。

**PC算法步骤**：

**步骤1：构建完全图**——初始化所有变量对之间有无向边：

$$\mathcal{E}_0 = \{(X_i, X_j) : i \neq j\} \tag{A45.2-init}$$

**步骤2：条件独立性检验**——逐步删除条件独立的边：

对于每条边 $(X_i, X_j)$，寻找条件集 $S \subseteq \mathcal{V} \setminus \{X_i, X_j\}$ 使得：

$$X_i \perp\!\!\!\perp X_j \mid S \tag{A45.2-ci}$$

如果找到这样的 $S$，则删除边 $(X_i, X_j)$。条件独立性通过偏相关检验判断：

$$\rho_{X_i, X_j \mid S} = 0 \iff X_i \perp\!\!\!\perp X_j \mid S \tag{A45.2-pcorr}$$

**步骤3：方向定向**——对剩余的边定向，利用v-结构（collider）：

如果 $X_i - X_k - X_j$ 且 $X_i$ 和 $X_j$ 不相邻，且 $X_k \notin S_{ij}$（使 $X_i \perp\!\!\!\perp X_j$ 的条件集），则定向为 $X_i \to X_k \leftarrow X_j$（v-结构）。

**因果强度矩阵**：

从因果图 $\mathcal{G}$ 提取因果强度矩阵 $C \in \mathbb{R}^{d_{\text{mid}} \times d_{\text{mid}}}$：

$$C_{ij} = \begin{cases} |\rho_{X_i, X_j \mid \text{pa}(X_j) \setminus \{X_i\}}| & \text{if } X_i \to X_j \text{ in } \mathcal{G} \\ 0 & \text{otherwise} \end{cases} \tag{A45.2-strength}$$

其中 $\text{pa}(X_j)$ 为 $X_j$ 的父节点集，$C_{ij}$ 度量了 $X_i$ 对 $X_j$ 的因果效应强度。

- **输入**：中层目标的时间序列数据 $\{X_1(t), \ldots, X_{d_{\text{mid}}}(t)\}_{t=1}^{T}$
- **输出**：因果图 $\mathcal{G}$，因果强度矩阵 $C$
- **约束**：数据量 $T \geq 50$（因果发现的最小样本量）；因果图为DAG（有向无环图）

> **白话翻译**：PC算法就是"找因果关系"——先假设所有变量都可能有关，然后逐步删除"条件独立"的边（给定其他变量后不相关的删掉），最后给剩余的边定方向（谁导致谁）。关键是"条件独立"——不是简单的相关，而是排除了其他因素影响后的独立。

---

### 第三部分：状态转移矩阵在线估计

#### 2.3 状态转移矩阵在线估计（A45.3）

中层状态转移矩阵 $T \in \mathbb{R}^{K \times K}$ 描述离散状态之间的转移概率，其中 $K$ 为离散状态数。在线估计使用贝叶斯更新：

**先验分布**：

$$\text{vec}(T_i) \sim \text{Dirichlet}(\boldsymbol{\alpha}_i) \tag{A45.3-prior}$$

其中 $T_i$ 为转移矩阵的第 $i$ 行（从状态 $i$ 转移到各状态的概率），$\boldsymbol{\alpha}_i$ 为Dirichlet先验参数。

**后验更新**：

观察到状态转移 $i \to j$ 后，更新Dirichlet参数：

$$\alpha_{ij} \leftarrow \alpha_{ij} + 1 \tag{A45.3-update}$$

**点估计**：

$$T_{ij} = \frac{\alpha_{ij}}{\sum_{k=1}^{K} \alpha_{ik}} \tag{A45.3-estimate}$$

**状态聚类**：

中层状态通过对底层预测目标的K-means聚类获得：

$$\mathbf{s}_{\text{mid}}(t) = \arg\min_{k \in \{1, \ldots, K\}} \|\mathbf{y}_{\text{low}}(t) - \boldsymbol{\mu}_k\|^2 \tag{A45.3-cluster}$$

其中 $\boldsymbol{\mu}_k$ 为第 $k$ 个聚类中心，$K$ 典型值为5-10。

- **输入**：状态转移观测序列，Dirichlet先验参数
- **输出**：状态转移矩阵 $T$，各状态的点估计
- **约束**：$T$ 的每行和为1（概率归一化）；$\alpha_{ij} > 0$

> **白话翻译**：状态转移矩阵就是"状态跳转的概率表"——从状态A跳到状态B的概率是多少。在线估计就是"边看边更新"——每看到一次状态跳转，就给对应的概率加1。用Dirichlet分布做贝叶斯更新，既能给出点估计，也能给出不确定性。

---

### 第四部分：趋势检测

#### 2.4 趋势检测——CUSUM变体（A45.4）

CUSUM（Cumulative Sum）变体用于检测中层趋势的变化点。标准CUSUM统计量为：

$$S_t = \max\left(0, S_{t-1} + (\mathbf{y}_t - \mu_0) / \sigma - \nu\right) \tag{A45.4}$$

其中：
- $\mu_0$ 为参考均值（历史均值）
- $\sigma$ 为参考标准差
- $\nu \geq 0$ 为允许偏差（slack parameter，典型值 $\nu = 0.5$）
- $S_0 = 0$

**变化点检测**：

当 $S_t > h_{\text{cusum}}$（阈值，典型值 $h_{\text{cusum}} = 5$）时，检测到变化点。

**双向CUSUM**：

同时检测上升和下降趋势：

$$S_t^+ = \max\left(0, S_{t-1}^+ + (\mathbf{y}_t - \mu_0) / \sigma - \nu\right) \tag{A45.4-up}$$

$$S_t^- = \max\left(0, S_{t-1}^- - (\mathbf{y}_t - \mu_0) / \sigma - \nu\right) \tag{A45.4-down}$$

当 $S_t^+ > h_{\text{cusum}}$ 时检测到上升趋势，当 $S_t^- > h_{\text{cusum}}$ 时检测到下降趋势。

**趋势强度**：

趋势强度定义为CUSUM统计量超过阈值的幅度：

$$\text{trend\_strength} = \frac{S_t - h_{\text{cusum}}}{h_{\text{cusum}}} \quad \text{if } S_t > h_{\text{cusum}} \tag{A45.4-strength}$$

- **输入**：中层目标时间序列 $\{\mathbf{y}_t\}$，参考参数 $\mu_0, \sigma$，阈值 $h_{\text{cusum}}$
- **输出**：变化点序列，趋势方向，趋势强度
- **约束**：$\sigma > 0$；$\nu \geq 0$；$h_{\text{cusum}} > 0$

> **白话翻译**：CUSUM就是"累积偏差检测器"——把每一步的偏差累积起来，如果累积偏差超过阈值就报警。好处是不会因为单步噪声误报，因为噪声会正负抵消；但真正的趋势会持续累积，最终触发报警。双向CUSUM同时检测涨和跌。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A45.1 | GRU: $\mathbf{h}_t = (1-\mathbf{z}_t)\odot\mathbf{h}_{t-1} + \mathbf{z}_t\odot\tilde{\mathbf{h}}_t$ | 中层目标序列、GRU参数 | 隐状态、多步预测 | 梯度裁剪 |
| A45.2 | PC算法：条件独立性检验+方向定向 | 中层时间序列 | 因果图 $\mathcal{G}$、强度矩阵 $C$ | $T \geq 50$；DAG |
| A45.3 | $T_{ij} = \alpha_{ij} / \sum_k \alpha_{ik}$（Dirichlet后验） | 状态转移观测、先验参数 | 转移矩阵 $T$ | 行和为1 |
| A45.4 | $S_t = \max(0, S_{t-1} + (y_t - \mu_0)/\sigma - \nu)$ | 时间序列、参考参数 | 变化点、趋势方向、强度 | $\sigma > 0$ |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# 引用阶段44的数据类
from stage_044 import MidLevelTarget


# ============================================================
# 阶段45 数据类：因果图
# ============================================================
@dataclass(frozen=True)
class CausalGraph:
    """PC算法发现的因果图（A45.2）。

    Attributes:
        num_nodes: 节点数 d_mid
        adjacency: 邻接矩阵，adj[i][j]=1 表示 i→j
        strength_matrix: 因果强度矩阵 C ∈ R^{d_mid × d_mid}
        sep_sets: 分离集字典，{(i,j): S}
    """
    num_nodes: int
    adjacency: np.ndarray
    strength_matrix: np.ndarray
    sep_sets: dict

    @property
    def density(self) -> float:
        """因果图密度（A44.1）。"""
        n = self.num_nodes
        if n <= 1:
            return 0.0
        return float(np.sum(self.adjacency)) / (n * (n - 1))

    @property
    def max_causal_strength(self) -> float:
        """最大因果强度（A44.1）。"""
        return float(np.max(self.strength_matrix))


# ============================================================
# 阶段45 数据类：趋势检测结果
# ============================================================
@dataclass(frozen=True)
class TrendDetection:
    """CUSUM趋势检测结果（A45.4）。

    Attributes:
        cusum_up: 上升CUSUM统计量序列
        cusum_down: 下降CUSUM统计量序列
        change_points: 变化点索引列表
        trend_directions: 各变化点的趋势方向
        trend_strengths: 各变化点的趋势强度
    """
    cusum_up: np.ndarray
    cusum_down: np.ndarray
    change_points: list
    trend_directions: list
    trend_strengths: list


# ============================================================
# 阶段45 数据类：中层模型算法
# ============================================================
@dataclass(frozen=True)
class MidLevelAlgorithm:
    """中层模型算法的完整描述。

    组合阶段44的MidLevelTarget，在中层预测目标的基础上增加
    算法参数——GRU参数、因果图、转移矩阵、CUSUM参数。

    Attributes:
        target: 阶段44的中层预测目标（组合字段）
        gru_hidden_dim: GRU隐状态维度 d_h
        gru_weights: GRU权重字典 {W_z, U_z, b_z, ...}
        gru_hidden_state: 当前GRU隐状态 h_t
        causal_graph: 因果图（A45.2）
        transition_matrix: 状态转移矩阵 T（A45.3）
        dirichlet_alphas: Dirichlet先验参数
        cusum_reference_mean: CUSUM参考均值 μ_0
        cusum_reference_std: CUSUM参考标准差 σ
        cusum_slack: CUSUM允许偏差 ν
        cusum_threshold: CUSUM阈值 h_cusum
        num_clusters: 状态聚类数 K

    Invariants:
        - gru_hidden_dim >= 1
        - transition_matrix 行和为1
        - dirichlet_alphas > 0
        - cusum_reference_std > 0
        - cusum_threshold > 0
    """
    target: MidLevelTarget
    gru_hidden_dim: int = 64
    gru_weights: Optional[dict] = None
    gru_hidden_state: Optional[np.ndarray] = None
    causal_graph: Optional[CausalGraph] = None
    transition_matrix: Optional[np.ndarray] = None
    dirichlet_alphas: Optional[np.ndarray] = None
    cusum_reference_mean: float = 0.0
    cusum_reference_std: float = 1.0
    cusum_slack: float = 0.5
    cusum_threshold: float = 5.0
    num_clusters: int = 5

    def __post_init__(self):
        if self.gru_hidden_dim < 1:
            raise ValueError(
                f"GRU隐状态维度必须 >= 1，收到 {self.gru_hidden_dim}"
            )
        if self.cusum_reference_std <= 0:
            raise ValueError(
                f"CUSUM参考标准差必须 > 0，"
                f"收到 {self.cusum_reference_std}"
            )
        if self.cusum_threshold <= 0:
            raise ValueError(
                f"CUSUM阈值必须 > 0，收到 {self.cusum_threshold}"
            )
        if self.transition_matrix is not None:
            row_sums = np.sum(self.transition_matrix, axis=1)
            if not np.allclose(row_sums, 1.0):
                raise ValueError("转移矩阵每行和必须为1")


# ============================================================
# 阶段45 核心接口：中层模型算法
# ============================================================
class MidLevelAlgorithmEngine:
    """阶段45的核心计算接口。

    实现GRU预测、PC因果发现、状态转移估计、CUSUM趋势检测。

    所有方法均为纯函数（无副作用），保证可验证性。
    """

    def gru_predict(
        self,
        algorithm: MidLevelAlgorithm,
        input_sequence: np.ndarray,
        horizon: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """GRU多步预测（A45.1）。

        使用简化线性近似模拟GRU多步预测。

        Args:
            algorithm: 中层算法参数
            input_sequence: 输入序列，形状 (T, d_mid)
            horizon: 预测步数 h ∈ [10, 100]

        Returns:
            (预测序列 Ŷ ∈ R^{h × d_mid}, 隐状态序列)

        Raises:
            ValueError: 当 horizon 超出 [10,100] 范围时

        Examples:
            >>> engine = MidLevelAlgorithmEngine()
            >>> algo = MidLevelAlgorithm(target=None)
            >>> seq = np.random.randn(20, 3)
            >>> y_pred, h_seq = engine.gru_predict(algo, seq, horizon=10)
            >>> y_pred.shape
            (10, 3)
        """
        if not (10 <= horizon <= 100):
            raise ValueError(f"horizon 必须在 [10,100] 内，收到 {horizon}")
        input_sequence = np.asarray(input_sequence, dtype=float)
        if input_sequence.ndim == 1:
            input_sequence = input_sequence.reshape(-1, 1)
        T, d = input_sequence.shape
        # 简化：用最后值的线性外推模拟GRU预测
        last = input_sequence[-1]
        trend = np.mean(np.diff(input_sequence, axis=0), axis=0) if T > 1 else np.zeros(d)
        preds = np.empty((horizon, d))
        for h in range(horizon):
            preds[h] = last + (h + 1) * trend
        hidden = np.zeros((horizon, algorithm.gru_hidden_dim))
        return preds, hidden

    def discover_causal_graph(
        self,
        algorithm: MidLevelAlgorithm,
        data: np.ndarray,
        significance_level: float = 0.05
    ) -> CausalGraph:
        """PC算法因果图发现（A45.2）。

        使用相关性阈值近似PC算法骨架。

        Args:
            algorithm: 中层算法参数
            data: 观测数据，形状 (T, d_mid)
            significance_level: 显著性水平 ∈ (0,1)

        Returns:
            CausalGraph 实例

        Raises:
            ValueError: 当数据量不足或 significance_level 不在 (0,1) 时

        Examples:
            >>> engine = MidLevelAlgorithmEngine()
            >>> algo = MidLevelAlgorithm(target=None)
            >>> data = np.random.randn(50, 3)
            >>> cg = engine.discover_causal_graph(algo, data)
            >>> cg.num_nodes == 3
            True
        """
        if not (0.0 < significance_level < 1.0):
            raise ValueError(f"显著性水平必须在 (0,1) 内，收到 {significance_level}")
        data = np.asarray(data, dtype=float)
        if data.shape[0] < 20:
            raise ValueError(f"数据量不足，需要至少20个样本，收到 {data.shape[0]}")
        T, d = data.shape
        adjacency = np.zeros((d, d), dtype=int)
        strengths = np.zeros((d, d), dtype=float)
        for i in range(d):
            for j in range(i + 1, d):
                corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
                if abs(corr) > (1.0 - significance_level):
                    adjacency[i, j] = 1
                    strengths[i, j] = abs(corr)
        num_edges = int(np.sum(adjacency))
        density = num_edges / (d * (d - 1)) if d > 1 else 0.0
        return CausalGraph(
            adjacency_matrix=adjacency,
            strength_matrix=strengths,
            num_nodes=d,
            num_edges=num_edges,
            density=density,
        )

    def update_transition_matrix(
        self,
        algorithm: MidLevelAlgorithm,
        from_state: int,
        to_state: int
    ) -> MidLevelAlgorithm:
        """在线更新状态转移矩阵（A45.3）。

        使用Dirichlet后验更新转移矩阵的对应计数。

        Args:
            algorithm: 当前算法状态
            from_state: 转移前状态索引 >= 0
            to_state: 转移后状态索引 >= 0

        Returns:
            更新后的算法状态

        Raises:
            ValueError: 当状态索引超出范围时

        Examples:
            >>> engine = MidLevelAlgorithmEngine()
            >>> algo = MidLevelAlgorithm(target=None, num_clusters=3)
            >>> new_algo = engine.update_transition_matrix(algo, 0, 1)
            >>> new_algo.transition_matrix[0, 1] > algo.transition_matrix[0, 1]
            True
        """
        n = algorithm.num_clusters
        if not (0 <= from_state < n):
            raise ValueError(f"from_state 必须在 [0,{n}) 内，收到 {from_state}")
        if not (0 <= to_state < n):
            raise ValueError(f"to_state 必须在 [0,{n}) 内，收到 {to_state}")
        alphas = algorithm.dirichlet_alphas.copy()
        alphas[from_state, to_state] += 1.0
        new_trans = alphas / alphas.sum(axis=1, keepdims=True)
        return MidLevelAlgorithm(
            target=algorithm.target,
            gru_hidden_dim=algorithm.gru_hidden_dim,
            gru_weights=algorithm.gru_weights,
            gru_hidden_state=algorithm.gru_hidden_state,
            causal_graph=algorithm.causal_graph,
            transition_matrix=new_trans,
            dirichlet_alphas=alphas,
            cusum_reference_mean=algorithm.cusum_reference_mean,
            cusum_reference_std=algorithm.cusum_reference_std,
            cusum_slack=algorithm.cusum_slack,
            cusum_threshold=algorithm.cusum_threshold,
            num_clusters=algorithm.num_clusters,
        )

    def detect_trend(
        self,
        algorithm: MidLevelAlgorithm,
        series: np.ndarray
    ) -> TrendDetection:
        """CUSUM趋势检测（A45.4）。

        实现双向CUSUM变体检测上升/下降趋势。

        Args:
            algorithm: 中层算法参数
            series: 时间序列，形状 (T,)

        Returns:
            TrendDetection 实例

        Examples:
            >>> engine = MidLevelAlgorithmEngine()
            >>> algo = MidLevelAlgorithm(target=None)
            >>> s = np.cumsum(np.random.randn(30))
            >>> td = engine.detect_trend(algo, s)
            >>> td.trend_direction in {-1, 0, 1}
            True
        """
        series = np.asarray(series, dtype=float)
        if series.ndim != 1:
            raise ValueError("series 必须为一维数组")
        T = series.shape[0]
        mu = algorithm.cusum_reference_mean
        sigma = algorithm.cusum_reference_std
        slack = algorithm.cusum_slack
        threshold = algorithm.cusum_threshold
        # 标准化序列
        z = (series - mu) / max(sigma, 1e-8)
        # 双向CUSUM
        S_pos = 0.0
        S_neg = 0.0
        cp = -1
        for t in range(T):
            S_pos = max(0.0, S_pos + z[t] - slack)
            S_neg = min(0.0, S_neg + z[t] + slack)
            if S_pos > threshold:
                cp = t
                break
            if abs(S_neg) > threshold:
                cp = t
                break
        if cp >= 0:
            direction = 1 if S_pos > threshold else -1
            strength = min(1.0, max(abs(S_pos), abs(S_neg)) / (threshold + 1e-8))
        else:
            direction = 0
            strength = 0.0
        return TrendDetection(
            trend_direction=direction,
            trend_strength=float(strength),
            change_point=cp,
            cusum_pos=float(S_pos),
            cusum_neg=float(S_neg),
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `MidLevelTarget` | ← 阶段44 | 作为中层算法的预测目标输入（组合字段） |
| **输入** | A12.1（误差传播规则） | ← 阶段12 | 中层误差信号是底层误差汇总的理论依据 |
| **输入** | A14.1（表征更新方程） | ← 阶段14 | 中层表征在线更新的梯度下降原理 |
| **输出** | `MidLevelAlgorithm` | → 阶段46 | 中层算法参数，包含MidLevelTarget组合 |
| **输出** | A45.1（GRU状态空间模型） | → 阶段46 | 中层预测器的序列预测方法 |
| **输出** | A45.2（因果图发现） | → 阶段46 | 中层预测器的因果分析方法 |
| **输出** | A45.3（状态转移估计） | → 阶段46 | 中层预测器的状态预测方法 |
| **输出** | A45.4（CUSUM趋势检测） | → 阶段46 | 中层预测器的趋势检测方法 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX/Flax | GRU实现 | `flax.linen.GRUCell`直接支撑GRU状态空间模型（A45.1），JAX的`jax.lax.scan`支撑高效序列处理 |
| causal-learn | PC算法 | `causal-learn`库提供PC算法的完整实现，支撑因果图发现（A45.2） |
| NumPy | 矩阵运算 | 状态转移矩阵估计（A45.3）和CUSUM计算（A45.4） |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V45.1 | GRU隐状态维度 | `h_t.shape == (d_h,)` | 维度断言 |
| V45.2 | GRU预测输出 | `Ŷ.shape == (horizon, d_mid)` | 维度断言 |
| V45.3 | GRU梯度稳定 | 梯度范数在100步内不爆炸 | 梯度监控 |
| V45.4 | PC算法DAG输出 | 因果图无环 | 拓扑排序验证 |
| V45.5 | PC算法因果强度 | $C_{ij} \geq 0$ 对所有 $i,j$ | 非负性验证 |
| V45.6 | PC算法已知结构 | 合成DAG数据上恢复正确边 | F1分数 > 0.8 |
| V45.7 | 转移矩阵归一化 | 每行和为1 | 归一化验证 |
| V45.8 | 转移矩阵在线更新 | 观测转移后概率正确增加 | 增量更新验证 |
| V45.9 | CUSUM变化点检测 | 合成阶跃信号上正确检测变化点 | 检测延迟 < 5步 |
| V45.10 | CUSUM双向检测 | 上升和下降趋势均被检测 | 双向测试 |
| V45.11 | CUSUM趋势强度 | 强趋势的强度 > 弱趋势 | 合成信号对比 |
| V45.12 | MidLevelAlgorithm组合 | `algorithm.target` 类型为 `MidLevelTarget` | 类型检查 |
| V45.13 | 非法输入拒绝 | 隐状态维度<1、阈值≤0等抛出 `ValueError` | 异常测试 |
| V45.14 | 因果图密度计算 | `density == num_edges / (n*(n-1))` | 公式验证 |
| V45.15 | GRU长期依赖 | 50步序列上GRU记忆有效 | 长期依赖合成测试 |

---

> **阶段45完** | 阶段45建立了中层模型的算法框架：GRU状态空间模型用于序列预测（A45.1），PC算法用于因果图发现（A45.2），Dirichlet后验用于状态转移矩阵在线估计（A45.3），CUSUM变体用于趋势检测（A45.4），封装为MidLevelAlgorithm数据类组合MidLevelTarget。下一步：阶段46将从本阶段的算法框架出发，设计中层模型的Python接口——MidLevelPredictor类。
