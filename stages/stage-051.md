# 阶段51：预测误差的标准化定义——三层预测体系的逐层误差度量与全局聚合

> **阶段编号**：S51 | **所属层**：第六层·误差评估层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 逐层误差向量 | "各楼层的偏差清单"——底层、中层、高层各有各的误差分数 |
> | 底层误差 | "细节偏差"——短期微预测和实际观测的距离 |
> | 中层误差 | "趋势偏差"——中期趋势预测和实际趋势的距离 |
> | 高层误差 | "战略偏差"——长期抽象预测和实际模式的距离 |
> | 全局误差 | "总偏差"——三层偏差按重要性加权汇总 |
> | StandardizedError | "标准化误差的完整画像"——组合CrossLayerProtocol，在跨层协议基础上增加标准化误差 |

---

## 1. 第一性原理依据

本阶段从阶段3（预测编码公理）和阶段50（跨层交互协议）的核心结论出发，建立三层预测体系的标准化误差定义。核心依据为：

- **A3.1（误差信号定义）**（引用阶段3）：$e^{l-1} = r^{l-1} - \hat{r}^{l-1}$，预测误差信号的基本定义。本阶段将A3.1从通用层级架构具体化为三层预测体系的逐层误差——底层、中层、高层各有独立的误差信号。

- **A3.2（精度加权误差能量）**（引用阶段3）：$\varepsilon_l = e^{l-1,T} \Pi_l e^{l-1}$，精度加权误差能量的定义。本阶段的逐层误差采用A3.2的形式——不是简单的欧氏距离，而是精度加权的二次型。

- **A50.1（跨层误差传播协议）**（引用阶段50）：$\mathbf{e}_{\text{mid}} = W_{\text{low→mid}} \cdot \Pi_{\text{low}} \mathbf{e}_{\text{low}}$，跨层误差传播规则。本阶段在A50.1的传播结果基础上，定义各层独立的标准化误差度量——传播协议解决"误差如何跨层传递"，本阶段解决"误差如何统一度量"。

- **A50.3（注意力调制协议）**（引用阶段50）：$\text{precision\_weight}_l = \text{softmax}(\beta \cdot \text{error\_magnitude}_l)$，注意力权重分配。本阶段的全局误差聚合权重 $w_l$ 与A50.3的注意力权重概念一致——按重要性加权汇总。

本阶段的核心目标是：**定义逐层误差向量 $\boldsymbol{\varepsilon} = [\varepsilon_{\text{low}}, \varepsilon_{\text{mid}}, \varepsilon_{\text{high}}]$，将三层预测体系各层的预测-观测偏差统一为标准化度量，并通过加权聚合形成全局误差 $\varepsilon_{\text{global}}$，封装为StandardizedError数据类，组合CrossLayerProtocol作为跨层协议的输入。** 标准化误差是误差评估体系的基石——它为后续的累积策略（阶段52）、反馈生成（阶段53）、异常检测（阶段55）提供统一的度量基础。

> **去重声明**：本阶段仅定义标准化误差的度量方式，不涉及时间累积（阶段52）、反馈信号（阶段53）或异常检测（阶段55）。跨层误差传播机制已在阶段50定义，本阶段不重复。

---

## 2. 核心数学公式

### 第一部分：逐层误差向量

#### 2.1 逐层误差向量定义（A51.1）

三层预测体系的误差用向量统一表示：

$$\boldsymbol{\varepsilon} = [\varepsilon_{\text{low}}, \varepsilon_{\text{mid}}, \varepsilon_{\text{high}}] \in \mathbb{R}_{\geq 0}^3 \tag{A51.1}$$

其中每个分量 $\varepsilon_l \geq 0$ 为对应层级的标准化误差能量。

**与A3.2的对应**：逐层误差向量的每个分量都是A3.2（引用阶段3）精度加权误差能量的实例化——$\varepsilon_l = \mathbf{e}_l^T \Pi_l \mathbf{e}_l$，其中 $\mathbf{e}_l$ 为第 $l$ 层的误差信号，$\Pi_l$ 为第 $l$ 层的精度矩阵。

**逐层误差向量的性质**：

**(P1) 非负性**：$\varepsilon_l \geq 0$ 对所有 $l \in \{\text{low, mid, high}\}$ 成立（由A3.2的非负性保证）。

**(P2) 零点一致性**：$\boldsymbol{\varepsilon} = \mathbf{0}$ 当且仅当三层预测完全准确——底层观测与底层预测一致、中层趋势与中层预测一致、高层模式与高层预测一致。

**(P3) 有界性**：在正常运行条件下，$\varepsilon_l \leq \varepsilon_l^{\max}$，其中 $\varepsilon_l^{\max}$ 由生存约束（引用阶段3，A3.3）决定。

- **输入**：三层预测器的输出（预测值和观测值），精度矩阵
- **输出**：逐层误差向量 $\boldsymbol{\varepsilon} \in \mathbb{R}_{\geq 0}^3$
- **约束**：各分量非负；与A3.2的精度加权形式一致

> **白话翻译**：逐层误差向量就是"各楼层的偏差清单"——底层、中层、高层各有各的误差分数，三个分数组成一个向量。每个分数都是"精度加权的偏差大小"——越可信的层级，偏差的权重越大。三个分数都是零，说明三层预测全部准确。

---

### 第二部分：底层误差

#### 2.2 底层误差定义（A51.2）

底层误差度量短期微预测与实际观测的偏差：

$$\varepsilon_{\text{low}} = \|\mathbf{o}_{\text{low}} - \hat{\mathbf{o}}_{\text{low}}\|^2 \tag{A51.2}$$

其中：
- $\mathbf{o}_{\text{low}} \in \mathbb{R}^{d_{\text{low}}}$ 为底层实际观测向量
- $\hat{\mathbf{o}}_{\text{low}} \in \mathbb{R}^{d_{\text{low}}}$ 为底层预测向量
- $d_{\text{low}}$ 为底层观测维度

**与A3.1的对应**：A51.2是A3.1（引用阶段3）$e^{l-1} = r^{l-1} - \hat{r}^{l-1}$ 在底层（$l=0$）的具体实例化。当 $\Pi_{\text{low}} = I$（各向同性精度）时，A51.2与A3.2完全等价：$\varepsilon_{\text{low}} = \mathbf{e}_{\text{low}}^T \mathbf{e}_{\text{low}} = \|\mathbf{e}_{\text{low}}\|^2$。

**底层误差的物理含义**：

底层预测器（引用阶段43）以1秒为间隔进行微预测，底层误差反映"系统对即时环境的预测有多准"。底层误差突增通常意味着：
1. 环境发生了突变（如传感器故障、外部冲击）
2. 底层模型参数过期（需要权重更新）
3. 观测噪声异常增大

**底层误差的尺度归一化**：

为消除维度 $d_{\text{low}}$ 的影响，底层误差可归一化为每维平均误差：

$$\bar{\varepsilon}_{\text{low}} = \frac{\varepsilon_{\text{low}}}{d_{\text{low}}} \tag{A51.2-norm}$$

归一化后 $\bar{\varepsilon}_{\text{low}}$ 可跨维度比较。

- **输入**：底层观测 $\mathbf{o}_{\text{low}}$，底层预测 $\hat{\mathbf{o}}_{\text{low}}$
- **输出**：$\varepsilon_{\text{low}} \in \mathbb{R}_{\geq 0}$
- **约束**：$\mathbf{o}_{\text{low}}$ 和 $\hat{\mathbf{o}}_{\text{low}}$ 维度一致

> **白话翻译**：底层误差就是"细节偏差"——短期微预测和实际观测的欧氏距离平方。距离越大，说明短期预测越不准。底层误差突增意味着"刚刚发生的事和预期不一样"——可能是环境突变，也可能是模型过期。

---

### 第三部分：中层误差

#### 2.3 中层误差定义（A51.3）

中层误差度量中期趋势预测与实际趋势的偏差：

$$\varepsilon_{\text{mid}} = \|\mathbf{o}_{\text{mid}} - \hat{\mathbf{o}}_{\text{mid}}\|^2 \tag{A51.3}$$

其中：
- $\mathbf{o}_{\text{mid}} \in \mathbb{R}^{d_{\text{mid}}}$ 为中层实际趋势向量（由底层观测聚合得到，引用阶段44）
- $\hat{\mathbf{o}}_{\text{mid}} \in \mathbb{R}^{d_{\text{mid}}}$ 为中层趋势预测向量
- $d_{\text{mid}}$ 为中层趋势维度

**与A50.1的对应**：中层实际趋势 $\mathbf{o}_{\text{mid}}$ 由底层观测通过聚合映射得到（引用阶段50，A50.1-low-mid）。因此中层误差间接反映了底层误差的聚合效果——如果底层误差持续偏高，中层实际趋势与预测趋势的偏差也会增大。

**中层误差的物理含义**：

中层预测器（引用阶段46）以10秒为间隔进行趋势预测，中层误差反映"系统对中期趋势的预测有多准"。中层误差突增通常意味着：
1. 趋势方向判断错误（如上升趋势误判为下降）
2. 因果结构发生变化（引用阶段46的因果图需要更新）
3. 底层误差持续累积传导到中层

**中层误差与底层误差的独立性**：

虽然中层观测来自底层聚合，但中层误差与底层误差在统计上可以独立——当底层预测准确但聚合映射有偏差时，$\varepsilon_{\text{low}} \approx 0$ 但 $\varepsilon_{\text{mid}} > 0$；反之，底层预测偏差可能在中层聚合后被平滑，$\varepsilon_{\text{low}} > 0$ 但 $\varepsilon_{\text{mid}} \approx 0$。

- **输入**：中层趋势观测 $\mathbf{o}_{\text{mid}}$，中层趋势预测 $\hat{\mathbf{o}}_{\text{mid}}$
- **输出**：$\varepsilon_{\text{mid}} \in \mathbb{R}_{\geq 0}$
- **约束**：$\mathbf{o}_{\text{mid}}$ 和 $\hat{\mathbf{o}}_{\text{mid}}$ 维度一致

> **白话翻译**：中层误差就是"趋势偏差"——中期趋势预测和实际趋势的欧氏距离平方。趋势方向判断错了、因果结构变了、底层误差累积传导，都会让中层误差增大。中层误差和底层误差可以独立——底层准不代表趋势对，趋势对也不代表细节准。

---

### 第四部分：高层误差

#### 2.4 高层误差定义（A51.4）

高层误差度量长期抽象预测与实际模式的偏差：

$$\varepsilon_{\text{high}} = \|\mathbf{o}_{\text{high}} - \hat{\mathbf{o}}_{\text{high}}\|^2 \tag{A51.4}$$

其中：
- $\mathbf{o}_{\text{high}} \in \mathbb{R}^{d_{\text{high}}}$ 为高层实际抽象模式向量（由中层趋势聚合得到，引用阶段47）
- $\hat{\mathbf{o}}_{\text{high}} \in \mathbb{R}^{d_{\text{high}}}$ 为高层抽象预测向量
- $d_{\text{high}}$ 为高层抽象维度

**与A50.1的对应**：高层实际模式 $\mathbf{o}_{\text{high}}$ 由中层趋势通过聚合映射得到（引用阶段50，A50.1-mid-high）。高层误差间接反映了中层误差的聚合效果。

**高层误差的物理含义**：

高层预测器（引用阶段49）以100秒为间隔进行抽象预测，高层误差反映"系统对长期模式的预测有多准"。高层误差突增通常意味着：
1. 长期模式发生了根本性变化（如从周期模式变为随机模式）
2. 慢特征发生了突变（引用阶段48的SFA特征漂移）
3. 策略评估严重偏离实际（引用阶段48的策略搜索结果失效）

**高层误差的特殊性**：

高层误差的变化最慢但影响最深远——高层预测涉及系统的"大方向"，一旦出错，底层和中层的预测都会受到自上而下预测传播（引用阶段50，A50.2）的负面影响。

- **输入**：高层模式观测 $\mathbf{o}_{\text{high}}$，高层模式预测 $\hat{\mathbf{o}}_{\text{high}}$
- **输出**：$\varepsilon_{\text{high}} \in \mathbb{R}_{\geq 0}$
- **约束**：$\mathbf{o}_{\text{high}}$ 和 $\hat{\mathbf{o}}_{\text{high}}$ 维度一致

> **白话翻译**：高层误差就是"战略偏差"——长期抽象预测和实际模式的欧氏距离平方。长期模式变了、慢特征漂移了、策略评估失效了，都会让高层误差增大。高层误差变化最慢但影响最大——"大方向"错了，下面全跟着错。

---

### 第五部分：全局误差

#### 2.5 全局误差定义（A51.5）

全局误差是三层误差的加权聚合：

$$\varepsilon_{\text{global}} = w_{\text{low}} \cdot \varepsilon_{\text{low}} + w_{\text{mid}} \cdot \varepsilon_{\text{mid}} + w_{\text{high}} \cdot \varepsilon_{\text{high}} \tag{A51.5}$$

其中：
- $w_{\text{low}}, w_{\text{mid}}, w_{\text{high}} \geq 0$ 为各层权重
- $w_{\text{low}} + w_{\text{mid}} + w_{\text{high}} = 1$（权重归一化）

**权重与A50.3注意力权重的关系**：

A51.5的聚合权重 $w_l$ 与A50.3（引用阶段50）的注意力权重 $\text{precision\_weight}_l$ 概念一致——都是按重要性分配权重。区别在于：A50.3的注意力权重是动态的（根据误差大小实时调整），A51.5的聚合权重可以是静态的（预设权重）或动态的（与注意力权重同步）。

**默认权重方案**：

| 方案 | $w_{\text{low}}$ | $w_{\text{mid}}$ | $w_{\text{high}}$ | 适用场景 |
|------|---------|---------|----------|---------|
| 均匀权重 | 1/3 | 1/3 | 1/3 | 不区分层级重要性 |
| 底层优先 | 0.5 | 0.3 | 0.2 | 关注即时预测精度 |
| 高层优先 | 0.2 | 0.3 | 0.5 | 关注长期模式准确性 |
| 动态权重 | A50.3 | A50.3 | A50.3 | 按误差大小动态分配 |

**全局误差与自由能的关系**：

由A4.8（引用阶段4）的拉普拉斯近似，自由能 $F \approx \varepsilon_0 + \frac{1}{2}\sum_{l=1}^{L} \varepsilon_l + \text{complexity}$。在三层体系中，$\varepsilon_{\text{global}}$ 是自由能准确性项的标准化近似——全局误差越大，自由能越高，系统越"惊讶"。

**全局误差的性质**：

**(P1) 非负性**：$\varepsilon_{\text{global}} \geq 0$（由各分量非负和权重非负保证）。

**(P2) 零点一致性**：$\varepsilon_{\text{global}} = 0$ 当且仅当 $\boldsymbol{\varepsilon} = \mathbf{0}$（由权重严格正保证）。

**(P3) 单调性**：任意 $\varepsilon_l$ 增大时，$\varepsilon_{\text{global}}$ 不减小。

- **输入**：逐层误差向量 $\boldsymbol{\varepsilon}$，权重向量 $\mathbf{w} = [w_{\text{low}}, w_{\text{mid}}, w_{\text{high}}]$
- **输出**：$\varepsilon_{\text{global}} \in \mathbb{R}_{\geq 0}$
- **约束**：权重归一化；各分量非负

> **白话翻译**：全局误差就是"总偏差"——三层偏差按重要性加权汇总。权重可以是固定的（比如"底层更重要"），也可以是动态的（哪层误差大就多关注哪层）。全局误差越大，系统越"惊讶"，自由能越高。全局误差为零，说明三层预测全部准确。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A51.1 | $\boldsymbol{\varepsilon} = [\varepsilon_{\text{low}}, \varepsilon_{\text{mid}}, \varepsilon_{\text{high}}]$ | 三层预测-观测对、精度矩阵 | 逐层误差向量 $\in \mathbb{R}_{\geq 0}^3$ | 各分量非负 |
| A51.2 | $\varepsilon_{\text{low}} = \|\mathbf{o}_{\text{low}} - \hat{\mathbf{o}}_{\text{low}}\|^2$ | 底层观测、底层预测 | 底层误差 $\in \mathbb{R}_{\geq 0}$ | 维度一致 |
| A51.3 | $\varepsilon_{\text{mid}} = \|\mathbf{o}_{\text{mid}} - \hat{\mathbf{o}}_{\text{mid}}\|^2$ | 中层趋势观测、中层趋势预测 | 中层误差 $\in \mathbb{R}_{\geq 0}$ | 维度一致 |
| A51.4 | $\varepsilon_{\text{high}} = \|\mathbf{o}_{\text{high}} - \hat{\mathbf{o}}_{\text{high}}\|^2$ | 高层模式观测、高层模式预测 | 高层误差 $\in \mathbb{R}_{\geq 0}$ | 维度一致 |
| A51.5 | $\varepsilon_{\text{global}} = \sum_l w_l \varepsilon_l$ | 逐层误差向量、权重向量 | 全局误差 $\in \mathbb{R}_{\geq 0}$ | 权重归一化 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum
import numpy as np

# 引用阶段50的数据类
from stage_050 import CrossLayerProtocol


# ============================================================
# 阶段51 枚举：权重方案
# ============================================================
class WeightScheme(Enum):
    """全局误差聚合权重方案。"""
    UNIFORM = "uniform"          # 均匀权重 [1/3, 1/3, 1/3]
    LOW_PRIORITY = "low_priority"    # 底层优先 [0.5, 0.3, 0.2]
    HIGH_PRIORITY = "high_priority"  # 高层优先 [0.2, 0.3, 0.5]
    DYNAMIC = "dynamic"          # 动态权重（与A50.3注意力权重同步）


# ============================================================
# 阶段51 数据类：标准化误差
# ============================================================
@dataclass(frozen=True)
class StandardizedError:
    """三层预测体系的标准化误差描述。

    组合阶段50的CrossLayerProtocol，在跨层协议的基础上增加
    标准化误差度量——逐层误差向量和全局误差。

    Attributes:
        cross_layer_protocol: 阶段50的跨层交互协议（组合字段）
        epsilon_low: 底层误差 ε_low
        epsilon_mid: 中层误差 ε_mid
        epsilon_high: 高层误差 ε_high
        weights: 聚合权重 [w_low, w_mid, w_high]
        weight_scheme: 权重方案
        epsilon_global: 全局误差 ε_global
        observation_low: 底层实际观测
        observation_mid: 中层实际趋势
        observation_high: 高层实际模式
        prediction_low: 底层预测
        prediction_mid: 中层趋势预测
        prediction_high: 高层模式预测
        timestamp: 误差计算时间戳

    Invariants:
        - epsilon_low, epsilon_mid, epsilon_high >= 0
        - weights 之和为 1
        - epsilon_global == Σ w_l · ε_l
        - 各观测与预测维度一致
    """
    cross_layer_protocol: CrossLayerProtocol
    epsilon_low: float
    epsilon_mid: float
    epsilon_high: float
    weights: np.ndarray
    weight_scheme: WeightScheme = WeightScheme.UNIFORM
    epsilon_global: float = 0.0
    observation_low: Optional[np.ndarray] = None
    observation_mid: Optional[np.ndarray] = None
    observation_high: Optional[np.ndarray] = None
    prediction_low: Optional[np.ndarray] = None
    prediction_mid: Optional[np.ndarray] = None
    prediction_high: Optional[np.ndarray] = None
    timestamp: float = 0.0

    def __post_init__(self):
        # 验证各层误差非负
        for name, val in [("epsilon_low", self.epsilon_low),
                          ("epsilon_mid", self.epsilon_mid),
                          ("epsilon_high", self.epsilon_high)]:
            if val < 0:
                raise ValueError(
                    f"{name} 必须 >= 0，收到 {val}"
                )
        # 验证权重
        if self.weights is not None:
            if len(self.weights) != 3:
                raise ValueError(
                    f"权重向量长度必须为3，收到 {len(self.weights)}"
                )
            if not np.allclose(np.sum(self.weights), 1.0):
                raise ValueError(
                    f"权重之和必须为1，收到 {np.sum(self.weights)}"
                )
            if np.any(self.weights < 0):
                raise ValueError("权重必须非负")
        # 验证全局误差一致性
        expected_global = (
            self.weights[0] * self.epsilon_low
            + self.weights[1] * self.epsilon_mid
            + self.weights[2] * self.epsilon_high
        )
        if not np.allclose(self.epsilon_global, expected_global, atol=1e-6):
            raise ValueError(
                f"全局误差与加权和不一致："
                f"ε_global={self.epsilon_global}，"
                f"加权和={expected_global}"
            )

    @property
    def epsilon_vector(self) -> np.ndarray:
        """逐层误差向量（A51.1）。"""
        return np.array([self.epsilon_low, self.epsilon_mid, self.epsilon_high])

    @property
    def dominant_error_layer(self) -> str:
        """误差最大的层级。"""
        layers = ["low", "mid", "high"]
        errors = [self.epsilon_low, self.epsilon_mid, self.epsilon_high]
        return layers[int(np.argmax(errors))]

    @property
    def is_zero(self) -> bool:
        """三层误差是否全部为零。"""
        return (self.epsilon_low == 0.0
                and self.epsilon_mid == 0.0
                and self.epsilon_high == 0.0)

    @property
    def normalized_errors(self) -> np.ndarray:
        """每维平均误差（A51.2-norm推广）。"""
        dims = []
        for obs in [self.observation_low, self.observation_mid, self.observation_high]:
            if obs is not None:
                dims.append(len(obs))
            else:
                dims.append(1)
        errors = self.epsilon_vector
        dims_arr = np.array(dims, dtype=float)
        return errors / dims_arr


# ============================================================
# 阶段51 核心接口：标准化误差计算器
# ============================================================
class StandardizedErrorCalculator:
    """阶段51的核心接口：标准化误差计算器。

    计算三层预测体系的逐层误差和全局误差。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 逐层误差非负
        - 全局误差等于加权和
        - 权重归一化
    """

    def compute_layer_error(
        self,
        observation: np.ndarray,
        prediction: np.ndarray
    ) -> float:
        """计算单层误差（A51.2/A51.3/A51.4）。

        ε_l = ||o_l - ô_l||²

        Args:
            observation: 实际观测/趋势/模式
            prediction: 预测值

        Returns:
            误差 ε_l >= 0

        Raises:
            ValueError: 当观测和预测维度不匹配时
        """
        if observation.shape != prediction.shape:
            raise ValueError(
                f"观测和预测维度不匹配：{observation.shape} vs {prediction.shape}"
            )
        diff = observation - prediction
        return float(np.dot(diff, diff))

    def compute_global_error(
        self,
        epsilon_vector: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """计算全局误差（A51.5）。

        ε_global = Σ w_l · ε_l

        Args:
            epsilon_vector: 逐层误差向量 [ε_low, ε_mid, ε_high]
            weights: 聚合权重 [w_low, w_mid, w_high]

        Returns:
            全局误差 ε_global >= 0

        Raises:
            ValueError: 当 epsilon_vector 长度不为3时
            ValueError: 当权重不归一化时
        """
        if epsilon_vector.shape != (3,):
            raise ValueError(
                f"epsilon_vector 长度必须为3，收到 {epsilon_vector.shape}"
            )
        if weights.shape != (3,):
            raise ValueError(
                f"权重向量长度必须为3，收到 {weights.shape}"
            )
        if not np.allclose(np.sum(weights), 1.0):
            raise ValueError(
                f"权重之和必须为1，收到 {np.sum(weights)}"
            )
        if np.any(weights < 0):
            raise ValueError("权重必须非负")
        return float(np.dot(weights, epsilon_vector))

    def get_weights(
        self,
        scheme: WeightScheme,
        protocol: Optional[CrossLayerProtocol] = None
    ) -> np.ndarray:
        """获取聚合权重。

        Args:
            scheme: 权重方案
            protocol: 跨层协议（动态权重方案需要）

        Returns:
            权重向量 [w_low, w_mid, w_high]

        Raises:
            ValueError: 当动态权重方案但protocol为None时
        """
        if scheme == WeightScheme.UNIFORM:
            return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        elif scheme == WeightScheme.LOW_PRIORITY:
            return np.array([0.5, 0.3, 0.2])
        elif scheme == WeightScheme.HIGH_PRIORITY:
            return np.array([0.2, 0.3, 0.5])
        elif scheme == WeightScheme.DYNAMIC:
            if protocol is None:
                raise ValueError(
                    "动态权重方案需要提供 CrossLayerProtocol"
                )
            # 使用 A50.3 注意力权重概念：softmax 归一化
            # 从 protocol 中提取 error_magnitude（如果可用）
            # 否则回退到均匀权重
            error_magnitude = getattr(protocol, 'error_magnitude', None)
            if error_magnitude is None:
                return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
            exp_vals = np.exp(error_magnitude - np.max(error_magnitude))
            weights = exp_vals / np.sum(exp_vals)
            return weights
        else:
            raise ValueError(f"未知的权重方案：{scheme}")

    def evaluate(
        self,
        protocol: CrossLayerProtocol,
        obs_low: np.ndarray,
        pred_low: np.ndarray,
        obs_mid: np.ndarray,
        pred_mid: np.ndarray,
        obs_high: np.ndarray,
        pred_high: np.ndarray,
        weight_scheme: WeightScheme = WeightScheme.UNIFORM
    ) -> StandardizedError:
        """完整评估三层标准化误差（A51.1-A51.5）。

        Args:
            protocol: 跨层交互协议
            obs_low/pred_low: 底层观测/预测
            obs_mid/pred_mid: 中层观测/预测
            obs_high/pred_high: 高层观测/预测
            weight_scheme: 权重方案

        Returns:
            StandardizedError 实例

        Raises:
            ValueError: 当任意观测-预测维度不匹配时
        """
        # 验证维度一致性
        for name, obs, pred in [
            ("底层", obs_low, pred_low),
            ("中层", obs_mid, pred_mid),
            ("高层", obs_high, pred_high),
        ]:
            if obs.shape != pred.shape:
                raise ValueError(
                    f"{name}观测-预测维度不匹配：{obs.shape} vs {pred.shape}"
                )

        # 计算逐层误差
        epsilon_low = self.compute_layer_error(obs_low, pred_low)
        epsilon_mid = self.compute_layer_error(obs_mid, pred_mid)
        epsilon_high = self.compute_layer_error(obs_high, pred_high)

        # 获取权重
        weights = self.get_weights(weight_scheme, protocol)

        # 计算全局误差
        epsilon_vector = np.array([epsilon_low, epsilon_mid, epsilon_high])
        epsilon_global = self.compute_global_error(epsilon_vector, weights)

        return StandardizedError(
            cross_layer_protocol=protocol,
            epsilon_low=epsilon_low,
            epsilon_mid=epsilon_mid,
            epsilon_high=epsilon_high,
            weights=weights,
            weight_scheme=weight_scheme,
            epsilon_global=epsilon_global,
            observation_low=obs_low,
            observation_mid=obs_mid,
            observation_high=obs_high,
            prediction_low=pred_low,
            prediction_mid=pred_mid,
            prediction_high=pred_high,
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `CrossLayerProtocol` | ← 阶段50 | 作为标准化误差的跨层协议基础（组合字段） |
| **输入** | A3.1（误差信号定义） | ← 阶段3 | 逐层误差的理论基础 |
| **输入** | A3.2（精度加权误差能量） | ← 阶段3 | 误差度量的精度加权形式 |
| **输入** | A50.1（跨层误差传播） | ← 阶段50 | 各层误差的传播来源 |
| **输入** | A50.3（注意力调制协议） | ← 阶段50 | 动态权重方案的来源 |
| **输出** | `StandardizedError` | → 阶段52 | 标准化误差，作为误差累积的输入 |
| **输出** | A51.1-A51.5（标准化误差定义） | → 阶段52-60 | 误差评估体系的度量基础 |
| **输出** | `StandardizedErrorCalculator` | → 阶段59 | 误差计算器的接口 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 误差计算JIT编译 | `jax.jit`编译逐层误差和全局误差计算，实现端到端加速 |
| NumPy | 向量化误差计算 | 支撑逐层误差的向量化计算和权重聚合 |
| Pydantic | 数据验证 | 支撑StandardizedError的运行时类型检查和一致性验证 |
| structlog | 误差审计 | 结构化日志库，支撑误差计算的完整审计追踪 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V51.1 | 底层误差计算正确性 | $\varepsilon_{\text{low}} = \|\mathbf{o}_{\text{low}} - \hat{\mathbf{o}}_{\text{low}}\|^2$ | 数值对比 |
| V51.2 | 中层误差计算正确性 | $\varepsilon_{\text{mid}} = \|\mathbf{o}_{\text{mid}} - \hat{\mathbf{o}}_{\text{mid}}\|^2$ | 数值对比 |
| V51.3 | 高层误差计算正确性 | $\varepsilon_{\text{high}} = \|\mathbf{o}_{\text{high}} - \hat{\mathbf{o}}_{\text{high}}\|^2$ | 数值对比 |
| V51.4 | 全局误差等于加权和 | $\varepsilon_{\text{global}} = \sum_l w_l \varepsilon_l$ | 数值对比 |
| V51.5 | 逐层误差非负 | $\varepsilon_l \geq 0$ 对所有合法输入 | 随机测试 |
| V51.6 | 零点一致性 | $\boldsymbol{\varepsilon} = \mathbf{0}$ 当且仅当预测完全准确 | 零误差测试 |
| V51.7 | 权重归一化 | $w_{\text{low}} + w_{\text{mid}} + w_{\text{high}} = 1$ | 求和验证 |
| V51.8 | 均匀权重方案 | 权重 = [1/3, 1/3, 1/3] | 方案测试 |
| V51.9 | 动态权重方案 | 权重与A50.3注意力权重一致 | 对比验证 |
| V51.10 | 维度不匹配拒绝 | 观测-预测维度不一致时抛出 `ValueError` | 异常测试 |
| V51.11 | 负误差拒绝 | $\varepsilon_l < 0$ 时抛出 `ValueError` | 异常测试 |
| V51.12 | 权重非法拒绝 | 权重之和≠1或含负值时抛出 `ValueError` | 异常测试 |
| V51.13 | StandardizedError组合 | `error.cross_layer_protocol` 类型为 `CrossLayerProtocol` | 类型检查 |
| V51.14 | 全局误差与自由能一致性 | $\varepsilon_{\text{global}}$ 与自由能准确性项正相关 | 合成数据测试 |
| V51.15 | 端到端误差评估 | 1000步三层预测循环无异常 | 长期运行测试 |

---

> **阶段51完** | 阶段51建立了三层预测体系的标准化误差定义：逐层误差向量 $\boldsymbol{\varepsilon} = [\varepsilon_{\text{low}}, \varepsilon_{\text{mid}}, \varepsilon_{\text{high}}]$ 统一度量三层的预测-观测偏差（A51.1），底层误差度量短期微预测精度（A51.2），中层误差度量中期趋势预测精度（A51.3），高层误差度量长期抽象预测精度（A51.4），全局误差通过加权聚合统一三层偏差（A51.5），封装为StandardizedError数据类组合CrossLayerProtocol。下一步：阶段52将定义误差的时间窗口与累积策略。
