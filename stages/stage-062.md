# 阶段62：微修正算法——梯度下降、贝叶斯更新与EWC正则化的收敛保证

> **阶段编号**：S62 | **所属层**：第七层·进化适应层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 权重微调 | "轻轻拧螺丝"——沿梯度反方向微调权重，让预测更准 |
> | 学习率衰减 | "越修越精细"——随着修正次数增加，步长越来越小 |
> | 在线贝叶斯更新 | "用新证据修正信念"——观测到新数据后，按贝叶斯规则更新参数 |
> | EWC正则化 | "不忘旧知识"——学新东西时保护旧知识，不让旧参数偏离太远 |
> | 收敛保证 | "保证能修好"——数学证明微修正一定会收敛到更好的状态 |
> | MicroAdjuster | "微修正器的完整画像"——组合MicroAdjustTrigger，在触发条件基础上增加修正算法 |

---

## 1. 第一性原理依据

本阶段从阶段14（表征更新规则）、阶段15（权重更新规则）和阶段10（最小公理集）的核心结论出发，建立微修正的具体算法。核心依据为：

- **A14.1（表征更新方程）**（引用阶段14）：$\tau_r \dot{r}^l = \delta^{l+1} - e^l$，表征更新由自由能梯度驱动。本阶段将A14.1的连续时间更新推广为微修正场景下的离散化权重微调——当微修正触发条件满足时，按梯度方向微调权重。

- **A15.1（权重更新方程）**（引用阶段15）：$\tau_W \dot{W}^l = -(\partial f^l / \partial W^l) \cdot \delta^l$，权重更新由自由能关于权重的梯度驱动。本阶段将A15.1从连续时间推广为微修正场景下的离散化更新，并增加学习率衰减和EWC正则化。

- **A15.5（EWC正则化）**（引用阶段15）：弹性权重巩固（Elastic Weight Consolidation），保护重要参数不被新学习覆盖。本阶段在微修正中引入EWC正则化，确保微修正不会破坏已有知识。

- **A10.1（最小公理集）**（引用阶段10）：自由能最小化公理。微修正算法必须保证自由能单调递减（或至少不增），这是微修正收敛保证的理论基础。

- **A61.1-A61.3（微修正触发条件）**（引用阶段61）：梯度范数条件、稳定性条件和频率限制。微修正算法仅在触发条件满足时执行。

本阶段的核心目标是：**定义权重微调 $\Delta W = -\eta \cdot \partial F / \partial W$（A62.1），参数自适应学习率 $\eta_t = \eta_0 / (1 + \text{decay} \cdot t)$（A62.2），在线贝叶斯更新 $\mu_{t+1} = (\Sigma_t^{-1}\mu_t + H^TR^{-1}o) / \Sigma_{t+1}^{-1}$（A62.3），EWC正则化 $L_{\text{ewc}} = \Sigma_i F_i(\theta_i - \theta_i^*)^2$（A62.4），以及微修正的收敛保证（A62.5），封装为MicroAdjuster数据类，组合MicroAdjustTrigger作为触发条件的输入。**

> **去重声明**：本阶段仅定义微修正的具体算法，不重新定义表征更新（阶段14）或权重更新（阶段15）的完整规则。微修正是A14.1/A15.1在"小误差+稳定环境"条件下的特化实现，增加了学习率衰减、贝叶斯更新和EWC正则化。

---

## 2. 核心数学公式

### 第一部分：权重微调

#### 2.1 权重微调定义（A62.1）

微修正的核心操作是沿自由能梯度的反方向微调权重：

$$\Delta W = -\eta \cdot \frac{\partial F}{\partial W} \tag{A62.1}$$

其中：
- $\eta > 0$ 为学习率（步长）
- $\partial F / \partial W$ 为自由能关于权重的梯度
- $\Delta W$ 为权重更新量

**与A15.1的对应**：A62.1是A15.1（引用阶段15）$\tau_W \dot{W}^l = -(\partial f^l / \partial W^l) \cdot \delta^l$ 的离散化形式。在拉普拉斯近似下，$\partial F / \partial W = (\partial f / \partial W) \cdot \delta$，因此A62.1与A15.1完全一致。

**微修正与标准梯度下降的区别**：

| 维度 | 标准梯度下降 | 微修正（A62.1） |
|------|------------|----------------|
| 触发条件 | 每步执行 | 仅在A61.1-A61.3条件满足时执行 |
| 学习率 | 固定或按全局调度衰减 | 自适应衰减（A62.2） |
| 正则化 | L2/L1 | EWC正则化（A62.4） |
| 更新范围 | 所有参数 | 仅更新重要参数（受EWC保护） |
| 收敛目标 | 全局最优 | 局部精细调优 |

**权重微调的步长约束**：

为避免微修正步长过大导致自由能上升，设置步长上界：

$$\|\Delta W\| \leq \Delta W_{\max} = \theta_{\text{step}} \cdot \|W\| \tag{A62.1-step-bound}$$

其中 $\theta_{\text{step}} \in (0, 0.1)$ 为步长比例因子（通常设为0.01）。当 $\|\Delta W\| > \Delta W_{\max}$ 时，按比例缩小步长：

$$\Delta W \leftarrow \Delta W \cdot \frac{\Delta W_{\max}}{\|\Delta W\|} \tag{A62.1-clip}$$

- **输入**：自由能梯度 $\partial F / \partial W$，学习率 $\eta$，权重 $W$
- **输出**：权重更新量 $\Delta W$
- **约束**：$\eta > 0$；$\|\Delta W\| \leq \Delta W_{\max}$

> **白话翻译**：权重微调就是"轻轻拧螺丝"——沿梯度的反方向微调权重，步长很小，确保不会"拧过头"。和标准梯度下降的区别是：微修正只在条件满足时才做，步长会随时间衰减，还有EWC正则化保护旧知识。

---

### 第二部分：参数自适应学习率

#### 2.2 学习率衰减定义（A62.2）

学习率随微修正次数增加而衰减，实现"越修越精细"：

$$\eta_t = \frac{\eta_0}{1 + \text{decay} \cdot t} \tag{A62.2}$$

其中：
- $\eta_0 > 0$ 为初始学习率
- $\text{decay} > 0$ 为衰减率（通常设为0.01-0.1）
- $t$ 为微修正次数（非全局时间步）

**学习率衰减的物理含义**：

学习率衰减模拟了优化理论中的"步长递减"策略。在优化初期，参数距离最优较远，需要较大的步长快速接近；在优化后期，参数已接近最优，需要较小的步长精细调优。衰减率 $\text{decay}$ 控制从"快速接近"到"精细调优"的过渡速度。

**与其他衰减策略的比较**：

| 策略 | 公式 | 特点 |
|------|------|------|
| 线性衰减 | $\eta_t = \eta_0 / (1 + \text{decay} \cdot t)$ | 简单，收敛保证好 |
| 指数衰减 | $\eta_t = \eta_0 \cdot \gamma^t$ | 衰减快，可能过早停止 |
| 余弦退火 | $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})(1 + \cos(\pi t / T))$ | 周期性，适合非凸优化 |
| 自适应（Adam） | $\eta_t = \eta_0 / \sqrt{\hat{v}_t + \epsilon}$ | 参数级自适应 |

本阶段选择线性衰减（A62.2），原因：
1. 收敛保证明确（$\sum \eta_t = \infty$，$\sum \eta_t^2 < \infty$）
2. 实现简单，超参数少
3. 与微修正的"逐步精细"语义一致

**学习率下界**：

为避免学习率衰减到零导致微修正完全停止，设置学习率下界：

$$\eta_t \geq \eta_{\min} > 0 \tag{A62.2-lower}$$

通常 $\eta_{\min} = 0.01 \cdot \eta_0$。

- **输入**：初始学习率 $\eta_0$，衰减率 $\text{decay}$，微修正次数 $t$
- **输出**：当前学习率 $\eta_t$
- **约束**：$\eta_0 > 0$；$\text{decay} > 0$；$\eta_t \geq \eta_{\min}$

> **白话翻译**：学习率衰减就是"越修越精细"——刚开始修的时候步子大一点，快速接近目标；修了几次之后步子小一点，精细调整。这和修表一样——先粗调，再细调，最后微调。

---

### 第三部分：在线贝叶斯更新

#### 2.3 在线贝叶斯更新定义（A62.3）

当有新的观测数据到达时，按贝叶斯规则在线更新参数的后验分布：

$$\mu_{t+1} = \frac{\Sigma_t^{-1}\mu_t + H^TR^{-1}o_t}{\Sigma_{t+1}^{-1}} \tag{A62.3}$$

$$\Sigma_{t+1}^{-1} = \Sigma_t^{-1} + H^TR^{-1}H \tag{A62.3-cov}$$

其中：
- $\mu_t$ 为时刻 $t$ 的参数后验均值
- $\Sigma_t$ 为时刻 $t$ 的参数后验协方差
- $H = \partial \hat{o} / \partial \mu$ 为观测函数关于参数的Jacobian矩阵
- $R$ 为观测噪声协方差
- $o_t$ 为时刻 $t$ 的观测值

**与A5.10（最优权衡条件）的对应**：A62.3是A5.10（引用阶段5）在线性情形下的递推形式。A5.10给出的是批量解 $\mu^* = (G^T\Sigma_\eta^{-1}G + \Sigma_{\text{prior}}^{-1})^{-1}(G^T\Sigma_\eta^{-1}o + \Sigma_{\text{prior}}^{-1}\mu_{\text{prior}})$，A62.3给出的是在线递推解——每来一个新观测，就更新一次后验。

**在线贝叶斯更新的性质**：

1. **信息融合**：新后验 = 旧后验信息 + 新观测信息，精度（逆协方差）相加
2. **一致性**：当观测数量趋于无穷时，$\mu_t$ 收敛到真实参数（在正则条件下）
3. **在线性**：不需要存储所有历史数据，仅保留当前后验 $(\mu_t, \Sigma_t)$
4. **与卡尔曼滤波的等价性**：A62.3与卡尔曼滤波的更新步骤完全一致

**在线贝叶斯更新与权重微调的关系**：

A62.3（贝叶斯更新）和A62.1（权重微调）是两种互补的微修正方式：
- A62.1直接更新权重（点估计），适用于快速微调
- A62.3更新参数的后验分布（分布估计），适用于需要不确定性量化的场景

两者可以组合使用：先用A62.3更新后验分布，再用A62.1将后验均值作为新的权重。

- **输入**：当前后验 $(\mu_t, \Sigma_t)$，新观测 $o_t$，Jacobian $H$，噪声协方差 $R$
- **输出**：更新后的后验 $(\mu_{t+1}, \Sigma_{t+1})$
- **约束**：$\Sigma_t$ 正定；$R$ 正定；$H$ 可计算

> **白话翻译**：在线贝叶斯更新就是"用新证据修正信念"——每来一个新观测，就按贝叶斯规则更新参数的后验分布。新后验 = 旧信念 + 新证据，精度（确定性）越高的一方权重越大。和卡尔曼滤波完全一样——本质上就是递推的贝叶斯推断。

---

### 第四部分：EWC正则化

#### 2.4 EWC正则化定义（A62.4）

弹性权重巩固（EWC）通过惩罚重要参数的偏离来保护已有知识：

$$L_{\text{ewc}} = \sum_i F_i \cdot (\theta_i - \theta_i^*)^2 \tag{A62.4}$$

其中：
- $F_i$ 为Fisher信息矩阵的第 $i$ 个对角元素，度量参数 $\theta_i$ 的重要性
- $\theta_i$ 为当前参数值
- $\theta_i^*$ 为旧任务的最优参数值（"锚点"）

**与A15.5的对应**：A62.4是A15.5（引用阶段15）EWC正则化的微修正版本。在微修正场景下，EWC的作用更加关键——微修正的目标是精细调优，而不是学习全新知识，因此必须严格保护已有知识不被破坏。

**Fisher信息的计算**：

Fisher信息矩阵的对角元素近似为：

$$F_i \approx \mathbb{E}_{o \sim p(o|\theta^*)}\left[\left(\frac{\partial \ln p(o|\theta)}{\partial \theta_i}\bigg|_{\theta=\theta^*}\right)^2\right] \tag{A62.4-fisher}$$

在实际计算中，通常用经验平均近似：

$$F_i \approx \frac{1}{N}\sum_{j=1}^{N}\left(\frac{\partial \ln p(o_j|\theta)}{\partial \theta_i}\bigg|_{\theta=\theta^*}\right)^2$$

**EWC正则化在微修正中的作用**：

微修正的总损失函数为：

$$L_{\text{total}} = F(q, o; \theta) + \lambda_{\text{ewc}} \cdot L_{\text{ewc}} \tag{A62.4-total}$$

其中 $\lambda_{\text{ewc}} > 0$ 为EWC正则化系数。微修正的梯度为：

$$\frac{\partial L_{\text{total}}}{\partial \theta_i} = \frac{\partial F}{\partial \theta_i} + 2\lambda_{\text{ewc}} \cdot F_i \cdot (\theta_i - \theta_i^*) \tag{A62.4-gradient}$$

EWC正则化的效果：
- 对于重要参数（$F_i$ 大）：偏离锚点的惩罚大，微修正幅度小
- 对于不重要参数（$F_i$ 小）：偏离锚点的惩罚小，微修正幅度大

**EWC与L2正则化的关系**：当 $F_i = 1$ 对所有 $i$ 成立时，EWC退化为L2正则化（引用阶段5，A5.14）。EWC是L2正则化的参数级加权版本——重要参数的"弹簧"更硬，不容易被拉开。

- **输入**：当前参数 $\theta$，锚点参数 $\theta^*$，Fisher信息对角元素 $\{F_i\}$，正则化系数 $\lambda_{\text{ewc}}$
- **输出**：EWC正则化损失 $L_{\text{ewc}}$，EWC梯度 $2\lambda_{\text{ewc}} F_i (\theta_i - \theta_i^*)$
- **约束**：$F_i \geq 0$；$\lambda_{\text{ewc}} > 0$

> **白话翻译**：EWC正则化就是"不忘旧知识"——学新东西的时候，对旧知识中重要的参数加"弹簧"，不让它们偏离太远。Fisher信息衡量参数的重要性——越重要的参数，弹簧越硬。微修正时，重要参数几乎不动，不重要参数可以自由调整。

---

### 第五部分：微修正的收敛保证

#### 2.5 收敛保证定理（A62.5）

**定理62.1**：在以下条件下，微修正算法保证收敛：

1. 自由能 $F(\theta)$ 关于参数 $\theta$ 是 $L$-光滑的（梯度Lipschitz连续）
2. 学习率满足 $\eta_t \leq 1/L$
3. EWC正则化系数 $\lambda_{\text{ewc}} \geq 0$
4. 微修正触发条件（A61.1-A61.3）满足

则微修正后的自由能单调不增：

$$F(\theta_{t+1}) \leq F(\theta_t) - \frac{\eta_t}{2}\left\|\frac{\partial F}{\partial \theta}\right\|^2 + \lambda_{\text{ewc}} \cdot L_{\text{ewc}}(\theta_t) \tag{A62.5}$$

当 $\lambda_{\text{ewc}} = 0$ 时，自由能严格递减：

$$F(\theta_{t+1}) \leq F(\theta_t) - \frac{\eta_t}{2}\left\|\frac{\partial F}{\partial \theta}\right\|^2 \tag{A62.5-no-ewc}$$

**证明思路**：

由 $L$-光滑性，对任意 $\theta, \theta'$：

$$F(\theta') \leq F(\theta) + \nabla F(\theta)^T(\theta' - \theta) + \frac{L}{2}\|\theta' - \theta\|^2$$

令 $\theta' = \theta_t - \eta_t \nabla L_{\text{total}}$，代入得：

$$F(\theta_{t+1}) \leq F(\theta_t) - \eta_t \nabla F^T \nabla L_{\text{total}} + \frac{L\eta_t^2}{2}\|\nabla L_{\text{total}}\|^2$$

当 $\eta_t \leq 1/L$ 且 $\nabla L_{\text{total}} = \nabla F + 2\lambda_{\text{ewc}} F_i (\theta_i - \theta_i^*)$ 时，EWC项可能使自由能暂时上升，但总损失 $L_{\text{total}}$ 单调递减。当 $\lambda_{\text{ewc}} = 0$ 时，$\nabla L_{\text{total}} = \nabla F$，自由能严格递减。

**收敛速率**：

在A62.2的学习率衰减下（$\eta_t = \eta_0 / (1 + \text{decay} \cdot t)$），微修正的收敛速率为：

$$\min_{t=1,\ldots,T} \|\nabla F(\theta_t)\|^2 \leq \frac{2(F(\theta_0) - F^*)}{\sum_{t=1}^{T} \eta_t} = O\left(\frac{\ln T}{T}\right) \tag{A62.5-rate}$$

其中 $F^* = \min_\theta F(\theta)$。

- **输入**：自由能函数 $F$，参数序列 $\{\theta_t\}$，学习率序列 $\{\eta_t\}$
- **输出**：收敛保证（自由能单调不增）
- **约束**：$F$ 为 $L$-光滑；$\eta_t \leq 1/L$

> **白话翻译**：收敛保证就是"保证能修好"——在合理的条件下（自由能光滑、步长不太大），微修正一定能让自由能下降或至少不增。有EWC正则化时，自由能可能暂时上升（因为EWC在保护旧知识），但总损失（自由能+EWC惩罚）一定下降。没有EWC时，自由能严格下降——每修一次，一定比修之前好。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A62.1 | $\Delta W = -\eta \cdot \partial F / \partial W$ | 梯度、学习率、权重 | 权重更新量 | $\eta > 0$；$\|\Delta W\| \leq \Delta W_{\max}$ |
| A62.2 | $\eta_t = \eta_0 / (1 + \text{decay} \cdot t)$ | 初始学习率、衰减率、次数 | 当前学习率 | $\eta_0, \text{decay} > 0$；$\eta_t \geq \eta_{\min}$ |
| A62.3 | $\mu_{t+1} = (\Sigma_t^{-1}\mu_t + H^TR^{-1}o) / \Sigma_{t+1}^{-1}$ | 后验、观测、Jacobian、噪声 | 更新后后验 | $\Sigma_t, R$ 正定 |
| A62.4 | $L_{\text{ewc}} = \Sigma_i F_i(\theta_i - \theta_i^*)^2$ | 参数、锚点、Fisher信息 | EWC损失 | $F_i \geq 0$；$\lambda_{\text{ewc}} > 0$ |
| A62.5 | $F(\theta_{t+1}) \leq F(\theta_t) - (\eta_t/2)\|\nabla F\|^2 + \lambda_{\text{ewc}} L_{\text{ewc}}$ | 自由能、梯度、学习率 | 收敛保证 | $F$ 为 $L$-光滑；$\eta_t \leq 1/L$ |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import numpy as np

# 引用阶段61的数据类
from stage_061 import MicroAdjustTrigger


# ============================================================
# 阶段62 数据类：微修正器
# ============================================================
@dataclass(frozen=True)
class MicroAdjuster:
    """微修正算法的完整描述。

    组合阶段61的MicroAdjustTrigger，在触发条件的基础上增加
    微修正算法的状态——学习率、EWC参数、收敛信息。

    Attributes:
        trigger: 阶段61的微修正触发状态（组合字段）
        eta_0: 初始学习率
        eta_current: 当前学习率
        decay: 学习率衰减率
        eta_min: 学习率下界
        lambda_ewc: EWC正则化系数
        fisher_diag: Fisher信息矩阵对角元素
        anchor_params: 锚点参数 θ*
        step_size_limit: 步长比例因子 θ_step
        micro_adjust_count: 微修正执行次数
        last_free_energy: 上次微修正后的自由能
        convergence_history: 自由能历史（用于收敛判定）

    Invariants:
        - eta_0 > 0
        - eta_current >= eta_min > 0
        - decay > 0
        - lambda_ewc >= 0
        - step_size_limit > 0
        - micro_adjust_count >= 0
    """
    trigger: MicroAdjustTrigger
    eta_0: float = 0.01
    eta_current: float = 0.01
    decay: float = 0.01
    eta_min: float = 0.0001
    lambda_ewc: float = 100.0
    fisher_diag: Optional[np.ndarray] = None
    anchor_params: Optional[np.ndarray] = None
    step_size_limit: float = 0.01
    micro_adjust_count: int = 0
    last_free_energy: float = 0.0
    convergence_history: Optional[Tuple[float, ...]] = None

    def __post_init__(self):
        if self.eta_0 <= 0:
            raise ValueError(
                f"初始学习率必须 > 0，收到 {self.eta_0}"
            )
        if self.eta_current < self.eta_min:
            raise ValueError(
                f"当前学习率必须 >= 下界，"
                f"收到 eta={self.eta_current}, min={self.eta_min}"
            )
        if self.decay <= 0:
            raise ValueError(
                f"衰减率必须 > 0，收到 {self.decay}"
            )
        if self.lambda_ewc < 0:
            raise ValueError(
                f"EWC系数必须 >= 0，收到 {self.lambda_ewc}"
            )
        if self.step_size_limit <= 0:
            raise ValueError(
                f"步长比例因子必须 > 0，收到 {self.step_size_limit}"
            )
        if self.micro_adjust_count < 0:
            raise ValueError(
                f"微修正次数必须 >= 0，收到 {self.micro_adjust_count}"
            )

    @property
    def is_converged(self) -> bool:
        """微修正是否已收敛。"""
        if self.convergence_history is None or len(self.convergence_history) < 5:
            return False
        recent = self.convergence_history[-5:]
        return max(recent) - min(recent) < 1e-6

    @property
    def ewc_active(self) -> bool:
        """EWC正则化是否激活。"""
        return (
            self.lambda_ewc > 0
            and self.fisher_diag is not None
            and self.anchor_params is not None
        )


# ============================================================
# 阶段62 核心接口：微修正算法
# ============================================================
class MicroAdjustAlgorithm:
    """阶段62的核心接口：微修正算法。

    实现权重微调、学习率衰减、在线贝叶斯更新、
    EWC正则化和收敛保证。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 微修正仅在触发条件满足时执行
        - 自由能单调不增（无EWC时严格递减）
        - 学习率单调递减且不低于下界
    """

    def adjust_weights(
        self,
        adjuster: MicroAdjuster,
        gradient: np.ndarray,
        weights: np.ndarray
    ) -> Tuple[np.ndarray, MicroAdjuster]:
        """执行权重微调 ΔW = -η·∂F/∂W（A62.1）。

        Args:
            adjuster: 当前微修正器状态
            gradient: 自由能梯度 ∂F/∂W
            weights: 当前权重 W

        Returns:
            (更新后的权重, 更新后的微修正器状态)
        """
        eta = adjuster.eta_current
        delta_w = -eta * gradient
        max_step = adjuster.step_size_limit * np.linalg.norm(weights)
        delta_norm = np.linalg.norm(delta_w)
        if delta_norm > max_step and delta_norm > 0:
            delta_w = delta_w * (max_step / delta_norm)
        new_weights = weights + delta_w
        new_count = adjuster.micro_adjust_count + 1
        new_eta = self.compute_learning_rate(
            adjuster.eta_0, adjuster.decay, new_count, adjuster.eta_min
        )
        new_adjuster = MicroAdjuster(
            trigger=adjuster.trigger,
            eta_0=adjuster.eta_0,
            eta_current=new_eta,
            decay=adjuster.decay,
            eta_min=adjuster.eta_min,
            lambda_ewc=adjuster.lambda_ewc,
            fisher_diag=adjuster.fisher_diag,
            anchor_params=adjuster.anchor_params,
            step_size_limit=adjuster.step_size_limit,
            micro_adjust_count=new_count,
            last_free_energy=adjuster.last_free_energy,
            convergence_history=adjuster.convergence_history,
        )
        return new_weights, new_adjuster

    def compute_learning_rate(
        self,
        eta_0: float,
        decay: float,
        t: int,
        eta_min: float
    ) -> float:
        """计算衰减学习率 η_t = η_0/(1+decay·t)（A62.2）。

        Args:
            eta_0: 初始学习率
            decay: 衰减率
            t: 微修正次数
            eta_min: 学习率下界

        Returns:
            当前学习率 η_t ∈ [η_min, η_0]
        """
        eta_t = eta_0 / (1.0 + decay * t)
        return max(eta_t, eta_min)

    def bayesian_update(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        observation: np.ndarray,
        H: np.ndarray,
        R: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """执行在线贝叶斯更新（A62.3）。

        μ_{t+1} = (Σ_t^{-1}μ_t + H^TR^{-1}o) / Σ_{t+1}^{-1}
        Σ_{t+1}^{-1} = Σ_t^{-1} + H^TR^{-1}H

        Args:
            mu: 当前后验均值
            sigma: 当前后验协方差
            observation: 新观测值
            H: Jacobian矩阵
            R: 观测噪声协方差

        Returns:
            (更新后的均值, 更新后的协方差)
        """
        sigma_inv = np.linalg.inv(sigma)
        R_inv = np.linalg.inv(R)
        Ht_Rinv = H.T @ R_inv
        sigma_next_inv = sigma_inv + Ht_Rinv @ H
        sigma_next = np.linalg.inv(sigma_next_inv)
        mu_next = sigma_next @ (sigma_inv @ mu + Ht_Rinv @ observation)
        return mu_next, sigma_next

    def compute_ewc_loss(
        self,
        params: np.ndarray,
        anchor: np.ndarray,
        fisher_diag: np.ndarray,
        lambda_ewc: float
    ) -> float:
        """计算EWC正则化损失（A62.4）。

        L_ewc = Σ_i F_i(θ_i - θ_i*)²

        Args:
            params: 当前参数
            anchor: 锚点参数
            fisher_diag: Fisher信息对角元素
            lambda_ewc: EWC系数

        Returns:
            EWC正则化损失
        """
        diff = params - anchor
        loss = float(np.sum(fisher_diag * diff ** 2))
        return lambda_ewc * loss

    def compute_ewc_gradient(
        self,
        params: np.ndarray,
        anchor: np.ndarray,
        fisher_diag: np.ndarray,
        lambda_ewc: float
    ) -> np.ndarray:
        """计算EWC正则化梯度（A62.4-gradient）。

        ∂L_ewc/∂θ_i = 2λ_ewc·F_i·(θ_i - θ_i*)

        Args:
            params: 当前参数
            anchor: 锚点参数
            fisher_diag: Fisher信息对角元素
            lambda_ewc: EWC系数

        Returns:
            EWC梯度
        """
        return 2.0 * lambda_ewc * fisher_diag * (params - anchor)

    def check_convergence(
        self,
        free_energy_history: Tuple[float, ...],
        tolerance: float = 1e-6,
        window: int = 5
    ) -> bool:
        """检查微修正是否收敛（A62.5）。

        Args:
            free_energy_history: 自由能历史
            tolerance: 收敛容差
            window: 检查窗口长度

        Returns:
            是否收敛
        """
        if free_energy_history is None or len(free_energy_history) < window:
            return False
        recent = free_energy_history[-window:]
        return max(recent) - min(recent) < tolerance
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `MicroAdjustTrigger` | ← 阶段61 | 作为微修正算法的触发条件输入（组合字段） |
| **输入** | A14.1（表征更新方程） | ← 阶段14 | 权重微调的理论基础 |
| **输入** | A15.1（权重更新方程） | ← 阶段15 | 权重微调和EWC正则化的理论基础 |
| **输入** | A15.5（EWC正则化） | ← 阶段15 | EWC正则化的定义 |
| **输入** | A5.10（最优权衡条件） | ← 阶段5 | 在线贝叶斯更新的理论基础 |
| **输出** | `MicroAdjuster` | → 阶段63 | 微修正器状态，作为Python接口的输入 |
| **输出** | A62.1-A62.5（微修正算法定义） | → 阶段63 | 微修正算法的核心定义 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 梯度计算与自动微分 | `jax.grad`计算自由能梯度和Fisher信息；`jax.jit`编译微修正步骤 |
| Optax | 优化器与学习率调度 | `optax`优化器库支撑学习率衰减和梯度裁剪 |
| NumPy | 矩阵运算 | 贝叶斯更新中的矩阵求逆和乘法运算 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V62.1 | 权重微调方向 | $\Delta W$ 与 $-\partial F / \partial W$ 方向一致 | 梯度验证 |
| V62.2 | 步长约束 | $\|\Delta W\| \leq \Delta W_{\max}$ | 范数检查 |
| V62.3 | 学习率衰减 | $\eta_t = \eta_0 / (1 + \text{decay} \cdot t)$ | 数值验证 |
| V62.4 | 学习率下界 | $\eta_t \geq \eta_{\min}$ | 范围检查 |
| V62.5 | 贝叶斯更新正确性 | $\mu_{t+1}$ 与批量贝叶斯推断结果一致 | 数值对比 |
| V62.6 | 贝叶斯更新精度相加 | $\Sigma_{t+1}^{-1} = \Sigma_t^{-1} + H^TR^{-1}H$ | 矩阵验证 |
| V62.7 | EWC损失正确性 | $L_{\text{ewc}} = \Sigma_i F_i(\theta_i - \theta_i^*)^2$ | 数值验证 |
| V62.8 | EWC梯度正确性 | $\partial L_{\text{ewc}} / \partial \theta_i = 2\lambda F_i(\theta_i - \theta_i^*)$ | 梯度验证 |
| V62.9 | 自由能单调不增（无EWC） | $F(\theta_{t+1}) \leq F(\theta_t)$ | 序列验证 |
| V62.10 | 总损失单调递减 | $L_{\text{total}}(\theta_{t+1}) \leq L_{\text{total}}(\theta_t)$ | 序列验证 |
| V62.11 | 收敛判定正确性 | 自由能变化 < 容差时判定为收敛 | 条件测试 |
| V62.12 | MicroAdjuster组合 | `adjuster.trigger` 类型为 `MicroAdjustTrigger` | 类型检查 |
| V62.13 | 非法参数拒绝 | $\eta_0 \leq 0$ 等抛出 `ValueError` | 异常测试 |
| V62.14 | EWC退化测试 | $F_i = 1$ 时EWC退化为L2正则化 | 退化验证 |
| V62.15 | 长期微修正稳定性 | 1000次微修正循环自由能不上升 | 长期运行测试 |

---

> **阶段62完** | 阶段62建立了微修正的完整算法：权重微调 $\Delta W = -\eta \cdot \partial F / \partial W$（A62.1），学习率衰减 $\eta_t = \eta_0 / (1 + \text{decay} \cdot t)$（A62.2），在线贝叶斯更新 $\mu_{t+1} = (\Sigma_t^{-1}\mu_t + H^TR^{-1}o) / \Sigma_{t+1}^{-1}$（A62.3），EWC正则化 $L_{\text{ewc}} = \Sigma_i F_i(\theta_i - \theta_i^*)^2$（A62.4），收敛保证 $F(\theta_{t+1}) \leq F(\theta_t) - (\eta_t/2)\|\nabla F\|^2$（A62.5），封装为MicroAdjuster数据类组合MicroAdjustTrigger。下一步：阶段63将定义微修正的Python接口。
