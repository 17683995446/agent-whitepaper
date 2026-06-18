# 阶段13：自上而下预测传播——高层预测的逐层下达机制

> **阶段编号**：S13 | **所属层**：第二层·工程实现层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 预测传播规则 | "逐级下达指令"——高层根据自己对世界的理解，逐层向下预测每层应该看到什么 |
> | 精度加权预测 | "带置信度的指令"——越可信的预测，下达的力度越大 |
> | 预测与误差的交互 | "指令与反馈的平衡"——实际感受 = 预测指令 + 误差修正 |
| 预测传播方向 | "从抽象到具体"——从"这是一只猫"到"应该有尖耳朵"到"应该有三角形像素" |
| 预测的级联效应 | "一环扣一环"——顶层预测变化，所有底层的预测都跟着变 |
| 预测精度调制 | "注意力调节"——对重要层级的预测赋予更高权重 |

---

## 1. 第一性原理依据

本阶段从阶段3和阶段11的核心结论出发，无需引入新的物理公理。核心依据为：

- **A3.5（自上而下预测）**（引用阶段3）：$\hat{r}^{l-1} = f^l(r^l, W^l)$，高层表征对低层表征的预测。
- **A3.4（自下而上传播信号）**（引用阶段3）：$\delta^l = \Pi_l e^l$，精度加权误差信号。
- **A11.2（条件分布）**（引用阶段11）：$p(r^{l-1}|r^l) = \mathcal{N}(r^{l-1}; f^l(r^l, W^l), \Sigma^l)$，条件分布的均值即为自上而下预测。

本阶段的核心目标是：**将阶段3中A3.5的单层预测推广到多层级联预测，推导预测信号如何从顶层逐层传播到底层，建立精度加权预测和预测-误差交互规则，为阶段14的表征更新提供完整的信号基础。** 自上而下预测传播与阶段12的自下而上误差传播构成预测编码的双向信号流——两者共同决定了每层表征的更新方向。

---

## 2. 核心数学公式

### 第一部分：预测传播规则

#### 2.1 预测传播规则（A13.1）

自上而下预测传播规则为：第 $l$ 层基于自身表征 $r^l$ 和权重 $W^l$，对第 $l-1$ 层生成预测：

$$\hat{r}^{l-1} = f^l(r^l, W^l) \tag{A13.1}$$

这是A3.5（引用阶段3）的直接引用，但此处强调其**级联传播**特性：顶层表征 $r^L$ 的预测逐层向下传播，每一层的预测成为下一层预测的输入。

级联预测链为：

$$r^L \xrightarrow{f^L} \hat{r}^{L-1} \xrightarrow{f^{L-1}} \hat{r}^{L-2} \xrightarrow{f^{L-2}} \cdots \xrightarrow{f^2} \hat{r}^1 \xrightarrow{g} \hat{o} \tag{A13.1-cascade}$$

其中 $\hat{o} = g(\hat{r}^1)$ 为对观测的最终预测。

- **输入**：第 $l$ 层表征 $r^l \in \mathbb{R}^{n_l}$，第 $l$ 层权重 $W^l \in \mathbb{R}^{n_{l-1} \times n_l}$
- **输出**：对第 $l-1$ 层的预测 $\hat{r}^{l-1} \in \mathbb{R}^{n_{l-1}}$
- **约束**：$f^l$ 可微（保证梯度传播）；传播方向从高层 $L$ 到底层 $1$

> **白话翻译**：预测传播就是"逐级下达指令"——高层说"我认为这是猫"，中层就预测"那应该有尖耳朵和长尾巴"，底层就预测"那应该有三角形和曲线的像素"。每一层的预测都基于上一层的"指令"，逐层从抽象翻译到具体。

**与A11.2的对应**：A13.1的预测 $\hat{r}^{l-1}$ 恰好是条件分布 $p(r^{l-1}|r^l) = \mathcal{N}(r^{l-1}; f^l(r^l, W^l), \Sigma^l)$ 的均值。因此，自上而下预测就是条件分布的"最佳猜测"——在给定高层表征的条件下，对低层表征的最可能值的估计。

**级联预测的计算算法**：

```
算法：TopDownPrediction(model, top_representation)
输入：层级生成模型 model, 顶层表征 r^L
输出：各层预测 {r̂^1, ..., r̂^{L-1}}, 观测预测 ô

1.  // 从顶层到底层逐层预测
2.  r_current ← top_representation  // r^L
3.  predictions ← []
4.  for l = L down to 2 do
5.      r̂^{l-1} ← f^l(r_current, W^l)
6.      predictions.append(r̂^{l-1})
7.      r_current ← r̂^{l-1}  // 预测作为下一层的输入
8.  end for
9.
10. // 观测预测
11. ô ← g(predictions[0])  // g(r̂^1)
12.
13. return predictions, ô
```

**注意**：级联预测使用预测值 $\hat{r}^{l-1}$（而非实际表征 $r^{l-1}$）作为下一层的输入。这是"纯预测"模式——假设所有层的表征都完美匹配高层的预测。在实际系统中，表征 $r^{l-1}$ 可能偏离预测 $\hat{r}^{l-1}$，此时使用实际表征 $r^{l-1}$ 作为下一层预测的输入更准确。

---

### 第二部分：精度加权预测

#### 2.2 精度加权预测（A13.2）

与误差信号的精度加权类似（引用阶段3，A3.4），预测信号也需要精度加权——精度越高的层，其预测在整体推断中的权重越大：

$$\Pi_l \cdot \hat{r}^{l-1} = \Pi_l \cdot f^l(r^l, W^l) \tag{A13.2}$$

- **输入**：第 $l$ 层精度矩阵 $\Pi_l \in \mathbb{S}_{++}^{n_{l-1}}$，预测 $\hat{r}^{l-1} \in \mathbb{R}^{n_{l-1}}$
- **输出**：精度加权预测 $\Pi_l \hat{r}^{l-1} \in \mathbb{R}^{n_{l-1}}$
- **约束**：$\Pi_l$ 正定

> **白话翻译**：精度加权预测就是"带置信度的指令"——越可信的层级（精度越高），其预测在整体推断中占的比重越大。就像专家的意见比外行的意见权重更大——不是所有层的"指令"都同等重要。

**精度加权预测的物理含义**：

精度矩阵 $\Pi_l = (\Sigma^l)^{-1}$ 度量了第 $l$ 层条件分布的"锐度"——$\Pi_l$ 越大，条件分布 $p(r^{l-1}|r^l)$ 越集中（预测越确定），预测的权重越大。反之，$\Pi_l$ 越小，条件分布越分散（预测越不确定），预测的权重越小。

在神经科学中，精度加权对应**注意力机制**——提高某层的精度等价于将注意力资源分配给该层。A13.2的形式 $\Pi_l \hat{r}^{l-1}$ 说明：注意力不仅调制误差信号的权重（A3.4），也调制预测信号的权重——两者是同一枚硬币的两面。

**精度加权预测与自由能的关系**：

由A11.2-log（引用阶段11），条件分布的对数似然为：

$$\ln p(r^{l-1}|r^l) = -\frac{1}{2}(r^{l-1} - \hat{r}^{l-1})^T \Pi_l (r^{l-1} - \hat{r}^{l-1}) + \text{const}$$

对 $r^{l-1}$ 求梯度：

$$\frac{\partial \ln p(r^{l-1}|r^l)}{\partial r^{l-1}} = -\Pi_l (r^{l-1} - \hat{r}^{l-1}) = -\Pi_l e^{l-1}$$

对 $\hat{r}^{l-1}$ 求梯度：

$$\frac{\partial \ln p(r^{l-1}|r^l)}{\partial \hat{r}^{l-1}} = \Pi_l (r^{l-1} - \hat{r}^{l-1}) = \Pi_l e^{l-1} = \delta^l$$

因此，**精度加权预测 $\Pi_l \hat{r}^{l-1}$ 的梯度效应等价于传播信号 $\delta^l$ 的负值**——预测越准确（$e^{l-1}$ 越小），对数似然的梯度越小，表征更新的驱动力越小。

---

### 第三部分：预测与误差的交互

#### 2.3 预测与误差的交互规则（A13.3）

在预测编码中，每层表征 $r^{l-1}$ 由两部分决定：自上而下的预测 $\hat{r}^{l-1}$ 和自下而上的误差修正 $\Pi_l^{-1} \delta^l$。两者的交互规则为：

$$r^{l-1} = \hat{r}^{l-1} + \Pi_l^{-1} \delta^l \tag{A13.3}$$

- **输入**：预测 $\hat{r}^{l-1} \in \mathbb{R}^{n_{l-1}}$，传播信号 $\delta^l \in \mathbb{R}^{n_l}$，精度矩阵 $\Pi_l \in \mathbb{S}_{++}^{n_{l-1}}$
- **输出**：更新后的表征 $r^{l-1} \in \mathbb{R}^{n_{l-1}}$
- **约束**：$\Pi_l$ 正定可逆；$\delta^l$ 维度与 $\Pi_l$ 匹配

> **白话翻译**：实际感受 = 预测指令 + 误差修正。预测是"我认为你应该看到什么"，误差修正是"实际与你预测的偏差经精度调整后的修正"。精度越高（$\Pi_l$ 越大），误差修正的权重越小（$\Pi_l^{-1}$ 越小）——因为预测已经很准了，不需要大修正。精度越低，误差修正的权重越大——因为预测不可靠，需要更多依赖实际反馈。

**完整推导**：

**步骤1：从误差信号定义出发**

由A3.1（引用阶段3），误差信号为 $e^{l-1} = r^{l-1} - \hat{r}^{l-1}$。因此：

$$r^{l-1} = \hat{r}^{l-1} + e^{l-1}$$

**步骤2：用传播信号表示误差**

由A3.4（引用阶段3），$\delta^l = \Pi_l e^{l-1}$（注意：A3.4中的 $\delta^l$ 对应第 $l$ 层的精度加权误差，其中 $e^l$ 是第 $l$ 层的误差信号。此处我们使用 $\delta^l = \Pi_l e^{l-1}$，其中 $e^{l-1}$ 是第 $l-1$ 层的误差信号，$\Pi_l$ 是第 $l$ 层的精度矩阵）。

由 $\Pi_l$ 正定可逆：

$$e^{l-1} = \Pi_l^{-1} \delta^l$$

**步骤3：代入**

$$r^{l-1} = \hat{r}^{l-1} + \Pi_l^{-1} \delta^l$$

$\blacksquare$

**A13.3的深层含义**：

A13.3揭示了预测编码的核心计算结构——每层表征是"自上而下预测"和"自下而上误差修正"的加权和：

- 当 $\Pi_l$ 很大（高精度/高置信度）时，$\Pi_l^{-1}$ 很小，误差修正的权重很小——表征主要由预测决定。这对应"相信预测"的模式——当预测很可靠时，不需要太多修正。
- 当 $\Pi_l$ 很小（低精度/低置信度）时，$\Pi_l^{-1}$ 很大，误差修正的权重很大——表征主要由误差信号决定。这对应"相信数据"的模式——当预测不可靠时，需要更多依赖实际观测。

**与贝叶斯推断的对应**：

A13.3的形式与贝叶斯后验的"先验+似然"结构完全同构：

| 预测编码 | 贝叶斯推断 |
|---------|-----------|
| $\hat{r}^{l-1}$（自上而下预测） | 先验均值 $\mu_{\text{prior}}$ |
| $\Pi_l^{-1} \delta^l$（误差修正） | 似然修正 $(\Sigma_{\text{like}}^{-1} \cdot \text{data})$ |
| $r^{l-1}$（更新后表征） | 后验均值 $\mu_{\text{post}}$ |
| $\Pi_l$（精度矩阵） | 先验精度 + 似然精度 |

在贝叶斯推断中，后验均值 = 先验均值 + 先验精度⁻¹ × 似然修正。A13.3的 $r^{l-1} = \hat{r}^{l-1} + \Pi_l^{-1}\delta^l$ 与此完全一致——预测编码的每层更新就是一次局部贝叶斯推断。

**A13.3的变分解释**：

由A4.9（引用阶段4），自由能关于 $r^{l-1}$ 的梯度为 $\partial F/\partial r^{l-1} = \delta^l - e^{l-1}$。在自由能最小化的稳态（$\partial F/\partial r^{l-1} = 0$）：

$$\delta^l = e^{l-1}$$

即 $\Pi_l e^{l-1} = e^{l-1}$，这意味着 $\Pi_l = I$（精度矩阵为单位矩阵）或 $e^{l-1} = 0$（预测完全准确）。一般情况下，稳态条件为 $e^{l-1} = 0$，即 $r^{l-1} = \hat{r}^{l-1}$——表征完美匹配预测。

在非稳态下，A13.3给出了表征的瞬时值——它是预测和误差修正的动态平衡。

---

### 第四部分：预测传播与误差传播的对偶性

#### 定理13.1：预测-误差对偶性

**定理13.1**：在层级生成模型中，自上而下预测传播和自下而上误差传播构成对偶关系：

1. **方向对偶**：预测从高层到低层（A13.1），误差从低层到高层（A12.2）
2. **精度对偶**：预测被精度加权（A13.2），误差也被精度加权（A3.4）
3. **交互对偶**：表征 = 预测 + 精度⁻¹ × 误差（A13.3），误差 = 表征 - 预测（A3.1）

**证明**：

**方向对偶**：A13.1的传播方向为 $r^L \to \hat{r}^{L-1} \to \cdots \to \hat{r}^1 \to \hat{o}$（从高到低），A12.2的传播方向为 $\delta^1 \to \delta^2 \to \cdots \to \delta^L$（从低到高）。两者方向相反。

**精度对偶**：A13.2的精度加权预测为 $\Pi_l \hat{r}^{l-1}$，A3.4的精度加权误差为 $\Pi_l e^{l-1} = \delta^l$。两者使用相同的精度矩阵 $\Pi_l$，但作用于不同的信号（预测 vs 误差）。

**交互对偶**：A13.3给出 $r^{l-1} = \hat{r}^{l-1} + \Pi_l^{-1}\delta^l$，A3.1给出 $e^{l-1} = r^{l-1} - \hat{r}^{l-1}$。两者互为逆运算——A13.3从预测和误差合成表征，A3.1从表征和预测分解误差。

$\blacksquare$

**对偶性的物理意义**：预测编码系统的信息流是双向的——自上而下的预测传达"应该看到什么"，自下而上的误差传达"实际与预期的偏差"。两者在每层交汇，共同决定表征的更新方向。这种对偶性是预测编码区别于纯前馈模型（如标准神经网络）的核心特征。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A13.1 | $\hat{r}^{l-1} = f^l(r^l, W^l)$ | $r^l$, $W^l$ | $\hat{r}^{l-1} \in \mathbb{R}^{n_{l-1}}$ | $f^l$ 可微；引用A3.5 |
| A13.2 | $\Pi_l \cdot \hat{r}^{l-1} = \Pi_l \cdot f^l(r^l, W^l)$ | $\Pi_l$, $\hat{r}^{l-1}$ | $\Pi_l \hat{r}^{l-1} \in \mathbb{R}^{n_{l-1}}$ | $\Pi_l$ 正定 |
| A13.3 | $r^{l-1} = \hat{r}^{l-1} + \Pi_l^{-1} \delta^l$ | $\hat{r}^{l-1}$, $\delta^l$, $\Pi_l$ | $r^{l-1} \in \mathbb{R}^{n_{l-1}}$ | $\Pi_l$ 正定可逆 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# 引用阶段12的数据类
from stage_012 import BottomUpPropagation


# ============================================================
# 阶段13 数据类：自上而下传播
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class TopDownPropagation:
    """自上而下预测传播的完整状态描述。

    组合阶段12的BottomUpPropagation，在自下而上误差传播的基础上增加
    自上而下预测传播的计算结果——各层预测、精度加权预测、预测-误差交互。

    Attributes:
        bottom_up: 阶段12的自下而上传播结果（组合字段）
        predictions: 各层预测元组 (r̂^1, ..., r̂^{L-1})
        precision_weighted_predictions: 各层精度加权预测元组 (Π_1·r̂^1, ..., Π_{L-1}·r̂^{L-2})
        obs_prediction: 观测预测 ô = g(r̂^1)
        updated_representations: 预测-误差交互后的表征元组 (r^1, ..., r^{L-1})

    Invariants:
        - len(predictions) == model.num_layers - 1
        - len(precision_weighted_predictions) == model.num_layers - 1
        - len(updated_representations) == model.num_layers - 1
        - r^{l-1} = r̂^{l-1} + Π_l^{-1}·δ^l 对所有层成立（A13.3）
    """
    bottom_up: BottomUpPropagation
    predictions: tuple[np.ndarray, ...]
    precision_weighted_predictions: tuple[np.ndarray, ...]
    obs_prediction: np.ndarray
    updated_representations: tuple[np.ndarray, ...]

    def __post_init__(self):
        L = self.bottom_up.model.num_layers
        # 验证预测数量
        if len(self.predictions) != L - 1:
            raise ValueError(
                f"预测数量 {len(self.predictions)} "
                f"必须等于 {L - 1}"
            )
        # 验证精度加权预测数量
        if len(self.precision_weighted_predictions) != L - 1:
            raise ValueError(
                f"精度加权预测数量 {len(self.precision_weighted_predictions)} "
                f"必须等于 {L - 1}"
            )
        # 验证更新表征数量
        if len(self.updated_representations) != L - 1:
            raise ValueError(
                f"更新表征数量 {len(self.updated_representations)} "
                f"必须等于 {L - 1}"
            )

    @property
    def prediction_errors(self) -> tuple[np.ndarray, ...]:
        """各层预测误差 e^{l-1} = r^{l-1} - r̂^{l-1}（A3.1）。"""
        model = self.bottom_up.model
        reprs = self.updated_representations
        preds = self.predictions
        return tuple(
            r - p for r, p in zip(reprs, preds)
        )

    @property
    def dual_flow_consistent(self) -> bool:
        """预测-误差对偶性是否成立（定理13.1）。

        检查 r^{l-1} == r̂^{l-1} + Π_l^{-1}·δ^l 对所有层成立。
        """
        model = self.bottom_up.model
        precisions = model.precisions
        deltas = self.bottom_up.propagated_signals
        reprs = self.updated_representations
        preds = self.predictions

        for l in range(len(reprs)):
            pi_inv = np.linalg.inv(precisions[l])
            expected = preds[l] + pi_inv @ deltas[l + 1]
            if not np.allclose(reprs[l], expected, atol=1e-8):
                return False
        return True


# ============================================================
# 阶段13 核心接口：自上而下传播计算器
# ============================================================
class TopDownPropagator:
    """阶段13的核心计算接口。

    实现自上而下预测传播的级联计算、精度加权和预测-误差交互。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - propagate 返回的预测满足 A13.1 级联传播
        - 精度加权预测满足 A13.2
        - 预测-误差交互满足 A13.3
        - 预测-误差对偶性满足定理13.1
    """

    def propagate(
        self,
        bottom_up: BottomUpPropagation,
        top_representation: np.ndarray
    ) -> TopDownPropagation:
        """执行自上而下预测传播（A13.1级联）。

        从顶层到底层逐层计算预测，然后计算预测-误差交互。

        Args:
            bottom_up: 阶段12的自下而上传播结果
            top_representation: 顶层表征 r^L

        Returns:
            TopDownPropagation 实例

        Raises:
            ValueError: 当顶层表征维度不匹配时
        """
        model = bottom_up.model
        L = model.num_layers
        if top_representation.shape[0] != model.layer_dims[-1]:
            raise ValueError(
                f"顶层表征维度 {top_representation.shape[0]} 必须等于 {model.layer_dims[-1]}"
            )
        predictions, obs_prediction = self.cascade_predict(model, top_representation)
        precision_weighted = []
        updated = []
        deltas = bottom_up.propagated_signals
        for l in range(L - 1):
            pi = model.precisions[l]
            pw = pi @ predictions[l]
            precision_weighted.append(pw)
            pi_inv = np.linalg.inv(pi)
            upd = predictions[l] + pi_inv @ deltas[l + 1]
            updated.append(upd)
        return TopDownPropagation(
            bottom_up=bottom_up,
            predictions=predictions,
            precision_weighted_predictions=tuple(precision_weighted),
            obs_prediction=obs_prediction,
            updated_representations=tuple(updated),
        )

    def predict_at_layer(
        self,
        model: 'HierarchicalGenerativeModel',
        layer_index: int,
        parent_representation: np.ndarray
    ) -> np.ndarray:
        """计算指定层的预测（A13.1）。

        Args:
            model: 层级生成模型
            layer_index: 目标层索引 l-1
            parent_representation: 父层表征 r^l

        Returns:
            预测 r̂^{l-1} = f^l(r^l, W^l)

        Raises:
            ValueError: 当维度不匹配时
        """
        if not (1 <= layer_index <= model.num_layers):
            raise ValueError(
                f"layer_index {layer_index} 超出范围 [1, {model.num_layers}]"
            )
        return model.generative_fns[layer_index - 1](parent_representation, model.weights[layer_index - 1])

    def precision_weighted_predict(
        self,
        model: 'HierarchicalGenerativeModel',
        layer_index: int,
        prediction: np.ndarray
    ) -> np.ndarray:
        """计算精度加权预测（A13.2）。

        Args:
            model: 层级生成模型
            layer_index: 层索引 l
            prediction: 预测 r̂^{l-1}

        Returns:
            精度加权预测 Π_l · r̂^{l-1}
        """
        return model.precisions[layer_index - 1] @ prediction

    def compute_updated_representation(
        self,
        prediction: np.ndarray,
        propagated_signal: np.ndarray,
        precision: np.ndarray
    ) -> np.ndarray:
        """计算预测-误差交互后的表征（A13.3）。

        r^{l-1} = r̂^{l-1} + Π_l^{-1} · δ^l

        Args:
            prediction: 预测 r̂^{l-1}
            propagated_signal: 传播信号 δ^l
            precision: 精度矩阵 Π_l

        Returns:
            更新后的表征 r^{l-1}

        Raises:
            ValueError: 当精度矩阵不正定时
            ValueError: 当维度不匹配时
        """
        try:
            np.linalg.cholesky(precision)
        except np.linalg.LinAlgError:
            raise ValueError("精度矩阵必须正定")
        precision_inv = np.linalg.inv(precision)
        return prediction + precision_inv @ propagated_signal

    def verify_dual_consistency(
        self,
        propagation: TopDownPropagation,
        tolerance: float = 1e-8
    ) -> bool:
        """验证预测-误差对偶性（定理13.1）。

        Args:
            propagation: 自上而下传播结果
            tolerance: 数值容差

        Returns:
            True 如果对偶性成立
        """
        return propagation.dual_flow_consistent

    def cascade_predict(
        self,
        model: 'HierarchicalGenerativeModel',
        top_representation: np.ndarray
    ) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
        """执行级联预测（A13.1-cascade）。

        从顶层到底层逐层预测，返回各层预测和观测预测。

        Args:
            model: 层级生成模型
            top_representation: 顶层表征 r^L

        Returns:
            (predictions, obs_prediction) 元组
            - predictions: 各层预测 (r̂^1, ..., r̂^{L-1})
            - obs_prediction: 观测预测 ô = g(r̂^1)
        """
        L = model.num_layers
        predictions = []
        r_current = top_representation
        for l in range(L - 1, 0, -1):
            r_hat = model.generative_fns[l](r_current, model.weights[l])
            predictions.insert(0, r_hat)
            r_current = r_hat
        obs_prediction = model.obs_generative_fn(predictions[0]) if predictions else model.obs_generative_fn(top_representation)
        return tuple(predictions), obs_prediction
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `BottomUpPropagation` | ← 阶段12 | 作为自上而下传播的误差信号基础（组合字段） |
| **输入** | A3.5（自上而下预测） | ← 阶段3 | 预测函数 $\hat{r}^{l-1} = f^l(r^l, W^l)$ 的定义 |
| **输入** | A3.4（自下而上传播信号） | ← 阶段3 | 精度加权误差信号 $\delta^l = \Pi_l e^l$ |
| **输入** | A11.2（条件分布） | ← 阶段11 | 预测函数作为条件分布均值的概率基础 |
| **输出** | `TopDownPropagation` | → 阶段14/15 | 包含BottomUpPropagation的预测传播结果 |
| **输出** | A13.1（预测传播规则） | → 阶段14 | 表征更新中预测信号的计算 |
| **输出** | A13.2（精度加权预测） | → 阶段14 | 精度调制表征更新的权重 |
| **输出** | A13.3（预测-误差交互） | → 阶段14 | 表征更新的核心计算规则 |
| **输出** | 定理13.1（预测-误差对偶性） | → 阶段14/29 | 双向信号流的统一理解 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 自动微分与向量化 | 级联预测（A13.1-cascade）的`jax.lax.scan`向量化实现；预测-误差交互（A13.3）的JIT编译 |
| NumPy | 矩阵运算 | 精度矩阵求逆、矩阵-向量乘法 |
| SciPy | 线性代数 | 精度矩阵Cholesky分解（高效求逆） |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V13.1 | 预测传播正确性 | `r̂^{l-1} == f^l(r^l, W^l)` 对所有层成立 | 随机表征 + 生成函数验证 |
| V13.2 | 级联预测一致性 | 级联预测与逐层手动计算在容差 $10^{-8}$ 内一致 | 多层模型数值验证 |
| V13.3 | 精度加权预测 | `Π_l·r̂^{l-1} == Π_l·f^l(r^l, W^l)` 对所有层成立 | 精度矩阵乘法验证 |
| V13.4 | 预测-误差交互 | `r^{l-1} == r̂^{l-1} + Π_l^{-1}·δ^l` 对所有层在容差内成立 | A13.3数值验证 |
| V13.5 | 交互与误差定义一致 | `e^{l-1} == r^{l-1} - r̂^{l-1}` 与A3.1一致 | 误差信号对比验证 |
| V13.6 | 预测-误差对偶性（定理13.1） | 方向对偶、精度对偶、交互对偶同时成立 | 综合验证 |
| V13.7 | 稳态条件 | 稳态时 `r^{l-1} == r̂^{l-1}`（预测完全匹配） | 收敛后表征-预测对比 |
| V13.8 | 高精度时预测主导 | $\Pi_l$ 很大时 $r^{l-1} \approx \hat{r}^{l-1}$ | 大精度数值验证 |
| V13.9 | 低精度时误差主导 | $\Pi_l$ 很小时 $r^{l-1} \approx \hat{r}^{l-1} + \Pi_l^{-1}\delta^l$（误差修正大） | 小精度数值验证 |
| V13.10 | TopDownPropagation组合 | `propagation.bottom_up` 类型为 `BottomUpPropagation` | 类型检查 |
| V13.11 | 非法输入拒绝 | 维度不匹配、精度矩阵不正定等抛出 `ValueError` | 异常测试 |
| V13.12 | 观测预测正确性 | `ô == g(r̂^1)` 与生成函数一致 | 观测预测验证 |
| V13.13 | 贝叶斯推断对应 | 预测-误差交互与高斯后验更新公式一致 | 解析对比验证 |

---

> **阶段13完** | 下一步：阶段14将从本阶段的预测-误差交互规则（A13.3）和阶段12的传播信号（A12.2）出发，推导表征更新规则（感知推断）。
