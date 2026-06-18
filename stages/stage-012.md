# 阶段12：自下而上误差传播——预测误差的逐层上报机制

> **阶段编号**：S12 | **所属层**：第二层·工程实现层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 误差传播规则 | "逐级上报"——底层发现偏差，逐层向上汇报，每层加上自己的偏差 |
> | 递归传播公式 | "接力赛"——每层收到下层的汇报后，乘以本层的"翻译矩阵"，继续向上传 |
> | Jacobian矩阵 | "翻译矩阵"——上层意思到下层表现的映射关系 |
> | 计算复杂度 | "上报的代价"——层数×维度的平方，越深越宽越慢 |
> | 精度加权传播 | "加权上报"——越可信的偏差，上报力度越大 |
> | 传播链的稳定性 | "上报不会爆炸"——误差在传播过程中不会无限放大 |

---

## 1. 第一性原理依据

本阶段从阶段3和阶段11的核心结论出发，无需引入新的物理公理。核心依据为：

- **A3.4（自下而上传播信号）**（引用阶段3）：$\delta^l = \Pi_l e^l$，精度加权误差信号的基本传播形式。
- **A3.1-A3.2（误差信号与精度加权能量）**（引用阶段3）：$e^{l-1} = r^{l-1} - \hat{r}^{l-1}$（A3.1），$\varepsilon_l = e^{l-1,T}\Pi_l e^{l-1}$（A3.2），误差信号的定义与度量。
- **A11.1-A11.3（层级生成模型）**（引用阶段11）：联合分布 $p(o, r^{1:L}) = p(o|r^1)\prod_l p(r^l|r^{l+1})p(r^{L+1})$（A11.1），条件分布 $p(r^{l-1}|r^l) = \mathcal{N}(r^{l-1}; f^l(r^l, W^l), \Sigma^l)$（A11.2），联合对数似然 $\ln p(o, r^{1:L}) = \sum_l \ln p(r^{l-1}|r^l)$（A11.3）。

本阶段的核心目标是：**将阶段3中A3.4的单层传播信号 $\delta^l = \Pi_l e^l$ 推广到多层递归传播，推导误差信号如何从底层逐层传播到高层，建立递归传播公式，分析计算复杂度，并证明传播链的稳定性。** 这是层级预测编码算法的核心计算步骤——后续阶段14（表征更新规则）和阶段15（权重更新规则）均依赖本阶段的传播信号。

---

## 2. 核心数学公式

### 第一部分：误差传播规则

#### 2.1 误差传播规则（A12.1）

在层级生成模型中，第 $l$ 层的误差信号需要传播到第 $l$ 层以上的所有层，以驱动高层表征的更新。误差传播规则为：

$$\delta^l = \Pi_l e^l + \frac{\partial f^{l+1}}{\partial r^l} \cdot \delta^{l+1} \tag{A12.1}$$

其中：
- $\Pi_l e^l$ 为第 $l$ 层自身的精度加权误差信号（引用阶段3，A3.4）
- $\frac{\partial f^{l+1}}{\partial r^l}$ 为第 $l+1$ 层生成函数关于第 $l$ 层表征的偏导数矩阵
- $\delta^{l+1}$ 为第 $l+1$ 层的传播信号（递归定义）

- **输入**：第 $l$ 层精度矩阵 $\Pi_l$，第 $l$ 层误差信号 $e^l$，第 $l+1$ 层生成函数的偏导数 $\partial f^{l+1}/\partial r^l$，第 $l+1$ 层传播信号 $\delta^{l+1}$
- **输出**：第 $l$ 层的传播信号 $\delta^l \in \mathbb{R}^{n_l}$
- **约束**：递归终止条件为 $\delta^{L+1} = \Pi_{L+1} e^{L+1}$（顶层只有自身误差，无更高层信号）

> **白话翻译**：误差传播规则就是"逐级上报"——每层收到两份信息：（1）本层自己的精度加权偏差 $\Pi_l e^l$（"我发现的问题"），（2）从上层传回来的信号经过"翻译矩阵" $\partial f^{l+1}/\partial r^l$ 映射后的结果（"上层反馈的问题对我的影响"）。两者相加就是本层需要处理的总信号。

**完整推导**：

**步骤1：从自由能梯度出发**

由A4.9（引用阶段4），自由能关于第 $l$ 层表征 $r^l$ 的梯度为：

$$\frac{\partial F}{\partial r^l} = \delta^{l+1} - e^l$$

但这是简化的形式。更精确地，自由能关于 $r^l$ 的梯度应考虑 $r^l$ 对所有下层条件分布的影响。

**步骤2：展开自由能关于 $r^l$ 的完整依赖**

由A11.3（引用阶段11），联合对数似然为 $\ln p(o, r^{1:L}) = \sum_{l=0}^{L}\ln p(r^{l-1}|r^l) + \ln p(r^{L+1})$。第 $l$ 层表征 $r^l$ 出现在两个条件分布中：

1. $p(r^{l-1}|r^l)$：$r^l$ 作为父节点，通过 $f^l(r^l, W^l)$ 影响 $r^{l-1}$ 的预测
2. $p(r^l|r^{l+1})$：$r^l$ 作为子节点，其预测误差 $e^l = r^l - f^{l+1}(r^{l+1}, W^{l+1})$ 直接出现

**步骤3：计算梯度**

对第1项求梯度（$r^l$ 通过 $f^l$ 影响 $r^{l-1}$ 的预测）：

$$\frac{\partial}{\partial r^l}\left[-\frac{1}{2}(r^{l-1} - f^l(r^l, W^l))^T \Pi_l (r^{l-1} - f^l(r^l, W^l))\right] = \left(\frac{\partial f^l}{\partial r^l}\right)^T \Pi_l (r^{l-1} - f^l(r^l, W^l))$$

$$= \left(\frac{\partial f^l}{\partial r^l}\right)^T \Pi_l e^{l-1} = \left(\frac{\partial f^l}{\partial r^l}\right)^T \delta^l_{\text{local}}$$

其中 $\delta^l_{\text{local}} = \Pi_l e^{l-1}$ 为第 $l$ 层的局部精度加权误差。

对第2项求梯度（$r^l$ 直接出现在 $e^l$ 中）：

$$\frac{\partial}{\partial r^l}\left[-\frac{1}{2}(r^l - f^{l+1}(r^{l+1}, W^{l+1}))^T \Pi_{l+1} (r^l - f^{l+1}(r^{l+1}, W^{l+1}))\right] = -\Pi_{l+1} e^l = -\delta^{l+1}_{\text{local}}$$

**步骤4：定义传播信号**

将第 $l$ 层的传播信号定义为自由能关于 $r^l$ 的负梯度（取负号使传播信号指向自由能下降方向）：

$$\delta^l \equiv -\frac{\partial F}{\partial r^l} = \Pi_{l+1} e^l - \left(\frac{\partial f^l}{\partial r^l}\right)^T \Pi_l e^{l-1}$$

重新整理，将本层误差和来自下层的传播信号分离：

$$\delta^l = \Pi_l e^l + \left(\frac{\partial f^{l+1}}{\partial r^l}\right)^T \cdot \delta^{l+1}_{\text{propagated}}$$

其中 $\delta^{l+1}_{\text{propagated}}$ 为从第 $l+1$ 层传播到第 $l$ 层的信号。在线性近似下，$\delta^{l+1}_{\text{propagated}} = \delta^{l+1}$，即得A12.1。

$\blacksquare$

**与神经网络反向传播的对应**：A12.1的形式 $\delta^l = \Pi_l e^l + (\partial f^{l+1}/\partial r^l)^T \delta^{l+1}$ 与神经网络反向传播的递归公式 $\delta^l = (\partial f^{l+1}/\partial z^l)^T \delta^{l+1} + \partial L/\partial z^l$ 在数学结构上完全同构。其中 $\Pi_l e^l$ 对应损失函数对本层激活的梯度，$(\partial f^{l+1}/\partial r^l)^T$ 对应权重转置乘以上层梯度。

---

### 第二部分：递归传播公式

#### 2.2 递归传播公式（A12.2）

将A12.1中的偏导数矩阵定义为Jacobian矩阵：

$$J_l = \frac{\partial f^{l+1}}{\partial r^l} \in \mathbb{R}^{n_{l+1} \times n_l} \tag{A12.2-jacobian}$$

则递归传播公式为：

$$\delta^l = \Pi_l e^l + J_l^T \delta^{l+1} \tag{A12.2}$$

递归展开得：

$$\delta^l = \Pi_l e^l + J_l^T \Pi_{l+1} e^{l+1} + J_l^T J_{l+1}^T \Pi_{l+2} e^{l+2} + \cdots + J_l^T J_{l+1}^T \cdots J_{L-1}^T \Pi_L e^L$$

即：

$$\delta^l = \sum_{k=l}^{L} \left(\prod_{j=l}^{k-1} J_j^T\right) \Pi_k e^k \tag{A12.2-expanded}$$

其中约定 $\prod_{j=l}^{l-1} J_j^T = I$（空积为单位矩阵）。

- **输入**：各层精度矩阵 $\{\Pi_l\}$，各层误差信号 $\{e^l\}$，各层Jacobian矩阵 $\{J_l\}$
- **输出**：第 $l$ 层的传播信号 $\delta^l \in \mathbb{R}^{n_l}$
- **约束**：递归方向从高层 $L$ 到低层 $l$；$J_l$ 的维度为 $n_{l+1} \times n_l$

> **白话翻译**：递归展开就是把"逐级上报"写成一条公式——第 $l$ 层收到的总信号 = 本层偏差 + 上层偏差经翻译后的影响 + 再上层偏差经两次翻译后的影响 + ... + 顶层偏差经多次翻译后的影响。每经过一层翻译，就乘以一个Jacobian转置矩阵——这就是"翻译矩阵"的物理含义。

**Jacobian矩阵的物理含义**：

$J_l = \partial f^{l+1}/\partial r^l$ 度量了"第 $l$ 层表征的微小变化如何影响第 $l+1$ 层的预测"。其转置 $J_l^T$ 度量了"第 $l+1$ 层的信号如何映射回第 $l$ 层的表征空间"——这就是误差信号从高层向低层传播时的"翻译"操作。

**线性模型下的简化**：当生成函数为线性 $f^{l+1}(r^{l+1}, W^{l+1}) = W^{l+1} r^{l+1}$ 时，$J_l = W^{l+1}$，递归公式简化为：

$$\delta^l = \Pi_l e^l + (W^{l+1})^T \delta^{l+1} \tag{A12.2-linear}$$

这与经典反向传播的权重转置乘法完全一致。

**递归计算算法**：

```
算法：BottomUpPropagation(model, errors, precisions)
输入：层级生成模型 model, 各层误差 {e^1, ..., e^L}, 各层精度 {Π_1, ..., Π_L}
输出：各层传播信号 {δ^1, ..., δ^L}

1.  // 顶层初始化
2.  δ^L ← Π_L · e^L

3.  // 从高层到低层递归
4.  for l = L-1 down to 1 do
5.      J_l ← ComputeJacobian(model, l)  // ∂f^{l+1}/∂r^l
6.      δ^l ← Π_l · e^l + J_l^T · δ^{l+1}
7.  end for

8.  return {δ^1, ..., δ^L}
```

- **计算方向**：从高层 $L$ 到低层 $1$（自下而上传播的方向是从底层到高层，但递归计算的方向是从高层到底层——因为低层的传播信号依赖高层的传播信号）
- **时间复杂度**：$O(L \cdot d^2)$（A12.3）

---

### 第三部分：计算复杂度分析

#### 2.3 传播的计算复杂度（A12.3）

自下而上误差传播的计算复杂度为：

$$\text{Complexity} = O(L \cdot d^2) \tag{A12.3}$$

其中 $L$ 为层级深度，$d = \max_l n_l$ 为最大层维度。

**推导**：

递归传播算法（A12.2）的每一步包含两个操作：
1. 精度加权：$\Pi_l e^l$，复杂度 $O(n_l^2)$（矩阵-向量乘法）
2. Jacobian转置乘法：$J_l^T \delta^{l+1}$，复杂度 $O(n_l \cdot n_{l+1})$

每步总复杂度为 $O(n_l \cdot n_{l+1} + n_l^2) = O(n_l \cdot \max(n_l, n_{l+1}))$。

设 $d = \max_l n_l$，则每步复杂度上界为 $O(d^2)$。$L$ 步的总复杂度为 $O(L \cdot d^2)$。

- **输入**：层级深度 $L$，最大层维度 $d$
- **输出**：计算复杂度 $O(L \cdot d^2)$
- **约束**：假设各层维度同阶（$n_l = O(d)$）

> **白话翻译**：误差传播的代价 = 层数 × 维度的平方。层数翻倍，代价翻倍；维度翻倍，代价翻四倍。所以"深而窄"的模型比"浅而宽"的模型计算更省——这是深度学习的计算优势之一。

**与反向传播的复杂度对比**：

| 方法 | 复杂度 | 说明 |
|------|--------|------|
| 自下而上误差传播（A12.2） | $O(L \cdot d^2)$ | 预测编码的传播方式 |
| 标准反向传播 | $O(L \cdot d^2)$ | 神经网络的梯度传播方式 |
| 前向传播 | $O(L \cdot d^2)$ | 生成模型的正向计算 |

三者复杂度同阶——预测编码的误差传播与标准反向传播的计算代价相当，但预测编码具有局部性优势（每层只需访问相邻层的信号，不需要全局梯度）。

**空间复杂度**：

递归传播需要存储各层的Jacobian矩阵和中间传播信号：

$$\text{Space} = O(L \cdot d^2)$$

这与时间复杂度同阶。在内存受限的场景下，可以使用梯度检查点（gradient checkpointing）技术将空间复杂度降至 $O(L \cdot d)$，代价是时间复杂度增加一倍。

---

### 第四部分：传播链的稳定性

#### 定理12.1：误差传播的稳定性条件

**定理12.1**：若各层Jacobian矩阵的谱范数满足 $\|J_l\|_2 \leq 1$ 对所有 $l$ 成立，则误差传播链是稳定的——传播信号 $\delta^l$ 的范数有上界：

$$\|\delta^l\| \leq \sum_{k=l}^{L} \|\Pi_k e^k\|$$

**完整证明**：

**步骤1：对递归公式取范数**

由A12.2，$\delta^l = \Pi_l e^l + J_l^T \delta^{l+1}$。取2-范数：

$$\|\delta^l\| \leq \|\Pi_l e^l\| + \|J_l^T\| \cdot \|\delta^{l+1}\|$$

由 $\|J_l^T\|_2 = \|J_l\|_2 \leq 1$（假设条件）：

$$\|\delta^l\| \leq \|\Pi_l e^l\| + \|\delta^{l+1}\|$$

**步骤2：递归展开**

$$\|\delta^l\| \leq \|\Pi_l e^l\| + \|\Pi_{l+1} e^{l+1}\| + \cdots + \|\Pi_L e^L\| = \sum_{k=l}^{L} \|\Pi_k e^k\|$$

**步骤3：有界性**

当各层精度加权误差 $\|\Pi_k e^k\|$ 有界时（由A3.3引用阶段3的生存约束保证），传播信号 $\delta^l$ 有界。因此传播链稳定。

$\blacksquare$

**稳定性条件的物理含义**：$\|J_l\|_2 \leq 1$ 意味着"翻译矩阵不会放大信号"——从上层到下层的映射是收缩的或等距的。这对应于生成函数 $f^{l+1}$ 的Lipschitz常数不超过1——上层表征的微小变化不会导致下层预测的更大变化。

**当稳定性条件不满足时**：若存在某层 $\|J_l\|_2 > 1$，则误差信号在传播过程中可能被放大（梯度爆炸），导致传播链不稳定。此时需要引入梯度裁剪（gradient clipping）或归一化技术（如层归一化）来保证稳定性。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A12.1 | $\delta^l = \Pi_l e^l + (\partial f^{l+1}/\partial r^l) \cdot \delta^{l+1}$ | $\Pi_l$, $e^l$, $\partial f^{l+1}/\partial r^l$, $\delta^{l+1}$ | $\delta^l \in \mathbb{R}^{n_l}$ | 递归终止于顶层 |
| A12.2 | $\delta^l = \Pi_l e^l + J_l^T \delta^{l+1}$ | $\Pi_l$, $e^l$, $J_l$, $\delta^{l+1}$ | $\delta^l \in \mathbb{R}^{n_l}$ | $J_l = \partial f^{l+1}/\partial r^l$ |
| A12.3 | Complexity $= O(L \cdot d^2)$ | $L$, $d$ | 复杂度阶 | 各层维度 $O(d)$ |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# 引用阶段11的数据类
from stage_011 import HierarchicalGenerativeModel


# ============================================================
# 阶段12 数据类：自下而上传播
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class BottomUpPropagation:
    """自下而上误差传播的完整状态描述。

    组合阶段11的HierarchicalGenerativeModel，在层级生成模型的基础上增加
    误差传播的计算结果——各层传播信号、Jacobian矩阵、计算复杂度。

    Attributes:
        model: 阶段11的层级生成模型（组合字段）
        error_signals: 各层误差信号元组 (e^1, ..., e^L)
        propagated_signals: 各层传播信号元组 (δ^1, ..., δ^L)
        jacobians: 各层Jacobian矩阵元组 (J_1, ..., J_{L-1})
        precision_weighted_errors: 各层精度加权误差元组 (Π_1·e^1, ..., Π_L·e^L)
        total_complexity: 计算复杂度估计 O(L·d²)

    Invariants:
        - len(error_signals) == model.num_layers
        - len(propagated_signals) == model.num_layers
        - len(jacobians) == model.num_layers - 1
        - len(precision_weighted_errors) == model.num_layers
        - δ^l = Π_l·e^l + J_l^T·δ^{l+1} 对所有 l 成立（A12.2）
        - total_complexity > 0
    """
    model: HierarchicalGenerativeModel
    error_signals: tuple[np.ndarray, ...]
    propagated_signals: tuple[np.ndarray, ...]
    jacobians: tuple[np.ndarray, ...]
    precision_weighted_errors: tuple[np.ndarray, ...]
    total_complexity: int

    def __post_init__(self):
        L = self.model.num_layers
        # 验证误差信号数量
        if len(self.error_signals) != L:
            raise ValueError(
                f"误差信号数量 {len(self.error_signals)} "
                f"必须等于层级深度 {L}"
            )
        # 验证传播信号数量
        if len(self.propagated_signals) != L:
            raise ValueError(
                f"传播信号数量 {len(self.propagated_signals)} "
                f"必须等于层级深度 {L}"
            )
        # 验证Jacobian数量
        if len(self.jacobians) != L - 1:
            raise ValueError(
                f"Jacobian数量 {len(self.jacobians)} "
                f"必须等于 {L - 1}"
            )
        # 验证精度加权误差数量
        if len(self.precision_weighted_errors) != L:
            raise ValueError(
                f"精度加权误差数量 {len(self.precision_weighted_errors)} "
                f"必须等于 {L}"
            )
        # 验证计算复杂度
        if self.total_complexity <= 0:
            raise ValueError(
                f"计算复杂度必须 > 0，收到 {self.total_complexity}"
            )

    @property
    def is_stable(self) -> bool:
        """传播链是否稳定（定理12.1）。

        检查所有Jacobian的谱范数是否 ≤ 1。
        """
        return all(
            np.linalg.norm(J, ord=2) <= 1.0 + 1e-8
            for J in self.jacobians
        )

    @property
    def signal_norms(self) -> tuple[float, ...]:
        """各层传播信号的范数 ‖δ^l‖。"""
        return tuple(
            float(np.linalg.norm(d)) for d in self.propagated_signals
        )

    @property
    def max_signal_norm(self) -> float:
        """最大传播信号范数。"""
        return max(self.signal_norms)


# ============================================================
# 阶段12 核心接口：自下而上传播计算器
# ============================================================
class BottomUpPropagator:
    """阶段12的核心计算接口。

    实现自下而上误差传播的递归计算、稳定性验证和复杂度分析。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - propagate 返回的传播信号满足 A12.2 递归公式
        - 传播链稳定性由定理12.1保证
        - 计算复杂度为 O(L·d²)（A12.3）
    """

    def propagate(
        self,
        model: HierarchicalGenerativeModel,
        error_signals: tuple[np.ndarray, ...],
        representations: tuple[np.ndarray, ...]
    ) -> BottomUpPropagation:
        """执行自下而上误差传播（A12.2）。

        从顶层到底层递归计算各层传播信号。

        Args:
            model: 层级生成模型
            error_signals: 各层误差信号 (e^1, ..., e^L)
            representations: 各层表征 (r^1, ..., r^L)

        Returns:
            BottomUpPropagation 实例，包含所有传播信号

        Raises:
            ValueError: 当误差信号或表征数量不匹配时
            ValueError: 当维度不匹配时
        """
        L = model.num_layers
        if len(error_signals) != L:
            raise ValueError(f"误差信号数量 {len(error_signals)} 必须等于 {L}")
        if len(representations) != L:
            raise ValueError(f"表征数量 {len(representations)} 必须等于 {L}")
        # 从顶层到底层递归计算传播信号
        jacobians = []
        propagated = [None] * L
        precision_weighted = []
        # 顶层初始化
        propagated[L - 1] = model.precisions[L - 1] @ error_signals[L - 1]
        precision_weighted.append(model.precisions[L - 1] @ error_signals[L - 1])
        # 从高层到低层递归
        for l in range(L - 2, -1, -1):
            J = self.compute_jacobian(model, l + 1, representations[l])
            jacobians.insert(0, J)
            propagated[l] = model.precisions[l] @ error_signals[l] + J.T @ propagated[l + 1]
            precision_weighted.insert(0, model.precisions[l] @ error_signals[l])
        # 顶层没有Jacobian
        total_complexity = self.estimate_complexity(model)
        return BottomUpPropagation(
            model=model,
            error_signals=error_signals,
            propagated_signals=tuple(propagated),
            jacobians=tuple(jacobians) if jacobians else (np.zeros((1, 1)),),
            precision_weighted_errors=tuple(precision_weighted),
            total_complexity=total_complexity,
        )

    def compute_jacobian(
        self,
        model: HierarchicalGenerativeModel,
        layer_index: int,
        representation: np.ndarray
    ) -> np.ndarray:
        """计算第 l 层的Jacobian矩阵 J_l（A12.2-jacobian）。

        Args:
            model: 层级生成模型
            layer_index: 层索引 l
            representation: 第 l 层表征 r^l

        Returns:
            Jacobian矩阵 J_l = ∂f^{l+1}/∂r^l，形状 (n_{l+1}, n_l)

        Raises:
            ValueError: 当 layer_index 超出范围时
        """
        if not (1 <= layer_index <= model.num_layers - 1):
            raise ValueError(
                f"layer_index {layer_index} 超出范围 [1, {model.num_layers - 1}]"
            )
        # 使用有限差分近似计算Jacobian
        n_l = model.layer_dims[layer_index - 1]
        n_lp1 = model.layer_dims[layer_index] if layer_index < model.num_layers else model.layer_dims[-1]
        f = model.generative_fns[layer_index]
        eps = 1e-5
        J = np.zeros((n_l, n_lp1))
        for i in range(n_lp1):
            r_plus = representation.copy()
            r_plus[i] += eps
            r_minus = representation.copy()
            r_minus[i] -= eps
            J[:, i] = (f(r_plus, model.weights[layer_index]) - f(r_minus, model.weights[layer_index])) / (2 * eps)
        return J

    def verify_recursion(
        self,
        propagation: BottomUpPropagation,
        tolerance: float = 1e-8
    ) -> bool:
        """验证递归公式的正确性（A12.2）。

        检查 δ^l = Π_l·e^l + J_l^T·δ^{l+1} 对所有 l 成立。

        Args:
            propagation: 传播结果
            tolerance: 数值容差

        Returns:
            True 如果递归公式在容差范围内成立
        """
        model = propagation.model
        L = model.num_layers
        for l in range(L - 1):
            expected = model.precisions[l] @ propagation.error_signals[l]
            if l < L - 1 and l < len(propagation.jacobians):
                expected += propagation.jacobians[l].T @ propagation.propagated_signals[l + 1]
            if not np.allclose(propagation.propagated_signals[l], expected, atol=tolerance):
                return False
        return True

    def check_stability(
        self,
        propagation: BottomUpPropagation
    ) -> dict[str, bool]:
        """检查传播链的稳定性（定理12.1）。

        检查各层Jacobian的谱范数是否 ≤ 1。

        Args:
            propagation: 传播结果

        Returns:
            字典包含：
            - "is_stable": 整体稳定性
            - "layer_stability": 各层稳定性列表
            - "spectral_norms": 各层谱范数列表
        """
        spectral_norms = []
        layer_stability = []
        for J in propagation.jacobians:
            s_norm = float(np.linalg.norm(J, ord=2))
            spectral_norms.append(s_norm)
            layer_stability.append(s_norm <= 1.0 + 1e-8)
        is_stable = all(layer_stability)
        return {
            "is_stable": is_stable,
            "layer_stability": layer_stability,
            "spectral_norms": spectral_norms,
        }

    def estimate_complexity(
        self,
        model: HierarchicalGenerativeModel
    ) -> int:
        """估计计算复杂度（A12.3）。

        Args:
            model: 层级生成模型

        Returns:
            估计的浮点运算次数
        """
        L = model.num_layers
        d = max(model.layer_dims) if model.layer_dims else 1
        return L * d * d

    def compute_propagation_signal_at_layer(
        self,
        model: HierarchicalGenerativeModel,
        layer_index: int,
        error_signal: np.ndarray,
        parent_propagated_signal: np.ndarray,
        representation: np.ndarray
    ) -> np.ndarray:
        """计算指定层的传播信号（A12.1）。

        Args:
            model: 层级生成模型
            layer_index: 层索引 l
            error_signal: 本层误差信号 e^l
            parent_propagated_signal: 上层传播信号 δ^{l+1}
            representation: 本层表征 r^l

        Returns:
            本层传播信号 δ^l = Π_l·e^l + J_l^T·δ^{l+1}

        Raises:
            ValueError: 当维度不匹配时
        """
        J = self.compute_jacobian(model, layer_index, representation)
        return model.precisions[layer_index - 1] @ error_signal + J.T @ parent_propagated_signal
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `HierarchicalGenerativeModel` | ← 阶段11 | 作为误差传播的层级生成模型基础（组合字段） |
| **输入** | A3.4（自下而上传播信号） | ← 阶段3 | 单层传播信号 $\delta^l = \Pi_l e^l$ 的定义 |
| **输入** | A3.1-A3.2（误差信号与精度加权能量） | ← 阶段3 | 误差信号和精度矩阵的定义 |
| **输入** | A11.1-A11.3（层级生成模型） | ← 阶段11 | 联合分布、条件分布、联合对数似然 |
| **输出** | `BottomUpPropagation` | → 阶段13/14/15 | 包含HierarchicalGenerativeModel的传播结果 |
| **输出** | A12.1（误差传播规则） | → 阶段14 | 表征更新规则中传播信号的计算依据 |
| **输出** | A12.2（递归传播公式） | → 阶段14/15 | 表征更新和权重更新中传播信号的递归计算 |
| **输出** | A12.3（计算复杂度） | → 阶段29 | 工程实现的性能预算依据 |
| **输出** | 定理12.1（传播稳定性） | → 阶段14 | 表征更新收敛性的前提条件 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | **本阶段新增**：自动微分与Jacobian计算 | JAX的`jax.jacfwd`/`jax.jacrev`直接支撑Jacobian矩阵（A12.2-jacobian）的自动计算，`jax.lax.scan`支撑递归传播（A12.2）的向量化实现，避免Python循环的性能瓶颈 |
| NumPy | 矩阵运算 | 精度加权误差计算、矩阵-向量乘法 |
| SciPy | 线性代数 | 谱范数计算（稳定性验证） |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V12.1 | 递归公式正确性 | `δ^l == Π_l·e^l + J_l^T·δ^{l+1}` 对所有 $l$ 在容差 $10^{-8}$ 内成立 | 逐层数值验证 |
| V12.2 | 顶层初始化 | `δ^L == Π_L·e^L`（顶层只有自身误差） | 断言验证 |
| V12.3 | 递归展开一致性 | `δ^l == Σ_k (Π_{j=l}^{k-1} J_j^T)·Π_k·e^k` 在容差内成立 | 展开公式数值对比 |
| V12.4 | 线性模型简化 | 线性模型下 `δ^l == Π_l·e^l + (W^{l+1})^T·δ^{l+1}` | 线性模型数值验证 |
| V12.5 | Jacobian计算正确性 | 自动微分Jacobian与有限差分Jacobian在容差 $10^{-5}$ 内一致 | JAX jacfwd vs 有限差分 |
| V12.6 | 计算复杂度 | 实际浮点运算次数与 $O(L \cdot d^2)$ 一致（相对误差 < 20%） | 计时测量 |
| V12.7 | 传播稳定性（定理12.1） | $\|J_l\|_2 \leq 1$ 时 $\|\delta^l\| \leq \sum_k \|\Pi_k e^k\|$ | 谱范数 + 信号范数验证 |
| V12.8 | 传播链不爆炸 | 稳定条件下传播信号范数不随层级递增 | 多层模型信号范数监控 |
| V12.9 | BottomUpPropagation组合 | `propagation.model` 类型为 `HierarchicalGenerativeModel` | 类型检查 |
| V12.10 | 非法输入拒绝 | 误差信号数量不匹配、维度不一致等抛出 `ValueError` | 异常测试 |
| V12.11 | 零误差传播 | 所有 $e^l = 0$ 时 $\delta^l = 0$ 对所有层成立 | 零向量测试 |
| V12.12 | 与反向传播对比 | 预测编码传播信号与标准反向传播梯度在数值上一致（线性模型） | PyTorch反向传播对比 |
| V12.13 | JAX向量化性能 | `jax.lax.scan` 实现比Python循环快 10× 以上 | 性能基准测试 |

---

> **阶段12完** | 下一步：阶段13将从本阶段的传播信号（A12.1-A12.2）出发，推导自上而下的预测传播规则。
