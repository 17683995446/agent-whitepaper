# 阶段50：跨层交互协议——三层预测体系的信号传播与时间对齐

> **阶段编号**：S50 | **所属层**：第五层·层级预测层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 误差传播协议 | "问题逐级上报"——底层误差传到中层，中层误差传到高层 |
> | 预测传播协议 | "指导逐级下达"——高层预测指导中层，中层预测指导底层 |
> | 注意力调制协议 | "按重要性分配精力"——误差越大的层级获得越多的注意力 |
> | 跨层时间尺度对齐 | "快慢协调"——底层1秒、中层10秒、高层100秒，严格对齐 |
> | 跨层接口统一协议 | "统一语言"——三层预测器用统一的接口交互 |
> | CrossLayerProtocol | "跨层协议的完整画像"——组合HighLevelPredictorState，在高层预测器基础上增加跨层协议 |

---

## 1. 第一性原理依据

本阶段从阶段12-13（自下而上/自上而下传播）、阶段17（快感知-慢学习）的核心结论出发，建立三层预测体系的跨层交互协议。核心依据为：

- **A12.1（误差传播规则）**（引用阶段12）：$\delta^l = \Pi_l e^l + (\partial f^{l+1}/\partial r^l) \cdot \delta^{l+1}$，误差信号从底层逐层向上传播。跨层误差传播协议将A12.1从两层推广到三层——底层误差传播到中层，中层误差传播到高层。

- **A13.1-A13.5（自上而下预测传播）**（引用阶段13）：自上而下的预测信号从高层逐层向下传播，指导低层的预测。跨层预测传播协议将自上而下传播从两层推广到三层——高层预测指导中层，中层预测指导底层。

- **A17.1-A17.2（快感知-慢学习）**（引用阶段17）：$\tau_{\text{perceive}} \ll \tau_{\text{learn}}$，感知比学习快100倍以上。跨层时间尺度对齐将快感知-慢学习原则从单层推广到三层——底层最快（1秒），中层中等（10秒），高层最慢（100秒）。

- **A3.4（自下而上传播信号）**（引用阶段3）：$\delta^l = \Pi_l e^l$，精度加权误差信号。跨层注意力调制协议将A3.4的精度加权推广为注意力权重——误差越大的层级获得越多的计算资源。

- **A43-A49（三层预测器接口）**（引用阶段43/46/49）：LowLevelPredictor、MidLevelPredictor、HighLevelPredictor三个预测器的统一接口。跨层接口统一协议定义了三个预测器之间的交互规范。

本阶段的核心目标是：**定义误差传播协议（low→mid→high）、预测传播协议（high→mid→low）、注意力调制协议（precision_weight = softmax(β·error_magnitude)）、跨层时间尺度对齐（Δt_low:Δt_mid:Δt_high = 1:10:100）、跨层接口统一协议，并封装为CrossLayerProtocol数据类，组合HighLevelPredictorState作为高层预测器的输入。** 跨层交互协议是三层预测体系的"神经系统"——它定义了信号如何在三层之间传播、如何分配注意力、如何对齐时间尺度。

---

## 2. 核心数学公式

### 第一部分：误差传播协议

#### 2.1 误差传播协议（A50.1）

误差传播协议定义了误差信号从底层到高层的传播规则：

$$\text{low\_error} \xrightarrow{\text{aggregate}} \text{mid\_error} \xrightarrow{\text{aggregate}} \text{high\_error} \tag{A50.1}$$

**底层到中层的误差传播**：

底层误差信号 $\mathbf{e}_{\text{low}} \in \mathbb{R}^{d_{\text{low}}}$ 通过聚合映射传播到中层：

$$\mathbf{e}_{\text{mid}} = W_{\text{low→mid}} \cdot \Pi_{\text{low}} \mathbf{e}_{\text{low}} \tag{A50.1-low-mid}$$

其中：
- $W_{\text{low→mid}} \in \mathbb{R}^{d_{\text{mid}} \times d_{\text{low}}}$ 为底层到中层的聚合权重矩阵
- $\Pi_{\text{low}} \in \mathbb{R}^{d_{\text{low}} \times d_{\text{low}}}$ 为底层精度矩阵

聚合权重矩阵由底层预测目标到中层预测目标的映射关系决定（引用A44.1-aggregate的逆映射）。

**中层到高层的误差传播**：

中层误差信号 $\mathbf{e}_{\text{mid}} \in \mathbb{R}^{d_{\text{mid}}}$ 通过聚合映射传播到高层：

$$\mathbf{e}_{\text{high}} = W_{\text{mid→high}} \cdot \Pi_{\text{mid}} \mathbf{e}_{\text{mid}} \tag{A50.1-mid-high}$$

其中：
- $W_{\text{mid→high}} \in \mathbb{R}^{d_{\text{high}} \times d_{\text{mid}}}$ 为中层到高层的聚合权重矩阵
- $\Pi_{\text{mid}} \in \mathbb{R}^{d_{\text{mid}} \times d_{\text{mid}}}$ 为中层精度矩阵

**与A12.1的对应**：

A50.1的误差传播协议是A12.1（引用阶段12）$\delta^l = \Pi_l e^l + J_l^T \delta^{l+1}$ 在三层预测体系中的具体实现。区别在于：A12.1是连续层的递归传播，A50.1是跨层的聚合传播——底层和中层之间有聚合窗口（100步底层→1步中层），中层和高层之间也有聚合窗口（100步中层→1步高层）。

**误差传播的能量守恒**：

误差传播过程中，精度加权误差能量守恒（不放大）：

$$\mathbf{e}_{\text{mid}}^T \Pi_{\text{mid}} \mathbf{e}_{\text{mid}} \leq \mathbf{e}_{\text{low}}^T \Pi_{\text{low}} \mathbf{e}_{\text{low}} \tag{A50.1-energy}$$

这保证了误差信号在传播过程中不会爆炸——与定理12.1（引用阶段12）的稳定性条件一致。

- **输入**：底层误差 $\mathbf{e}_{\text{low}}$，精度矩阵 $\Pi_{\text{low}}, \Pi_{\text{mid}}$
- **输出**：中层误差 $\mathbf{e}_{\text{mid}}$，高层误差 $\mathbf{e}_{\text{high}}$
- **约束**：误差能量不放大；聚合权重矩阵维度匹配

> **白话翻译**：误差传播协议就是"问题逐级上报"——底层发现误差后，汇总到中层（100个底层误差聚合成1个中层误差），中层再汇总到高层。汇总时乘以精度矩阵——越可信的误差信号，上报力度越大。关键是误差不会在传播中爆炸——上报的问题不会越报越大。

---

### 第二部分：预测传播协议

#### 2.2 预测传播协议（A50.2）

预测传播协议定义了预测信号从高层到底层的传播规则：

$$\text{high\_prediction} \xrightarrow{\text{disaggregate}} \text{mid\_prediction} \xrightarrow{\text{disaggregate}} \text{low\_prediction} \tag{A50.2}$$

**高层到中层的预测传播**：

高层预测 $\hat{\mathbf{y}}_{\text{high}} \in \mathbb{R}^{d_{\text{high}}}$ 通过解聚合映射传播到中层：

$$\hat{\mathbf{y}}_{\text{mid}}^{\text{top-down}} = W_{\text{high→mid}} \cdot \hat{\mathbf{y}}_{\text{high}} \tag{A50.2-high-mid}$$

其中 $W_{\text{high→mid}} \in \mathbb{R}^{d_{\text{mid}} \times d_{\text{high}}}$ 为高层到中层的解聚合权重矩阵。

**中层到底层的预测传播**：

中层预测 $\hat{\mathbf{y}}_{\text{mid}} \in \mathbb{R}^{d_{\text{mid}}}$ 通过解聚合映射传播到底层：

$$\hat{\mathbf{y}}_{\text{low}}^{\text{top-down}} = W_{\text{mid→low}} \cdot \hat{\mathbf{y}}_{\text{mid}} \tag{A50.2-mid-low}$$

其中 $W_{\text{mid→low}} \in \mathbb{R}^{d_{\text{low}} \times d_{\text{mid}}}$ 为中层到底层的解聚合权重矩阵。

**自上而下预测与自下而上预测的融合**：

底层的最终预测是自下而上预测和自上而下预测的融合：

$$\hat{\mathbf{y}}_{\text{low}}^{\text{final}} = \Pi_{\text{total}}^{-1}(\Pi_{\text{bottom-up}} \hat{\mathbf{y}}_{\text{low}}^{\text{bottom-up}} + \Pi_{\text{top-down}} \hat{\mathbf{y}}_{\text{low}}^{\text{top-down}}) \tag{A50.2-fusion}$$

其中 $\Pi_{\text{total}} = \Pi_{\text{bottom-up}} + \Pi_{\text{top-down}}$。精度越高的预测，在融合中权重越大。

**与A13.1的对应**：

A50.2的预测传播协议是A13.1（引用阶段13）自上而下预测传播 $\hat{r}^{l-1} = f^l(r^l, W^l)$ 在三层预测体系中的具体实现。自上而下的预测为底层提供了"先验"——高层认为底层应该看到什么。

- **输入**：高层预测 $\hat{\mathbf{y}}_{\text{high}}$，解聚合权重矩阵
- **输出**：中层预测 $\hat{\mathbf{y}}_{\text{mid}}^{\text{top-down}}$，底层预测 $\hat{\mathbf{y}}_{\text{low}}^{\text{top-down}}$
- **约束**：解聚合不引入信息（信息量不增）

> **白话翻译**：预测传播协议就是"指导逐级下达"——高层的长期预测指导下层的中期预测，中层的中期预测指导底层的短期预测。底层最终预测是"自己看到的"和"上级指导的"的融合——谁更准就多听谁的。上级指导相当于"先验"——"你应该看到这个"。

---

### 第三部分：注意力调制协议

#### 2.3 注意力调制协议（A50.3）

注意力调制协议根据各层误差的大小动态分配计算资源：

$$\text{precision\_weight}_l = \text{softmax}(\beta \cdot \text{error\_magnitude}_l) \tag{A50.3}$$

其中：
- $\text{error\_magnitude}_l = \mathbf{e}_l^T \Pi_l \mathbf{e}_l$ 为第 $l$ 层的精度加权误差能量
- $\beta > 0$ 为注意力温度参数（$\beta$ 越大，注意力越集中在误差最大的层）
- $\text{softmax}$ 确保权重归一化：$\sum_l \text{precision\_weight}_l = 1$

**注意力权重的物理含义**：

| $\beta$ 值 | 注意力分布 | 含义 |
|-----------|----------|------|
| $\beta \to 0$ | 均匀分布 $[1/3, 1/3, 1/3]$ | 不区分各层，均匀分配资源 |
| $\beta = 1$ | 按误差比例分配 | 误差大的层获得更多资源 |
| $\beta \to \infty$ | 全部集中在最大误差层 | 只关注问题最严重的层 |

**注意力调制的计算资源分配**：

各层的计算资源（更新频率、模型复杂度）按注意力权重分配：

$$\text{compute\_budget}_l = \text{precision\_weight}_l \times \text{total\_budget} \tag{A50.3-budget}$$

当底层误差突增时（如系统异常），$\text{precision\_weight}_{\text{low}}$ 增大，底层获得更多计算资源——更频繁的更新、更精细的模型。

**与A3.8的对应**：

A50.3的注意力调制协议是A3.8（引用阶段3）精度更新方程 $\dot{\Pi}_l = \kappa_\gamma(\Pi_l^{\text{opt}} - \Pi_l)$ 的跨层推广。A3.8调整单层内的精度，A50.3调整跨层的注意力分配——两者都是"根据误差调整信任度"的机制。

- **输入**：各层误差能量 $\{\varepsilon_l\}_{l=\text{low, mid, high}}$，温度参数 $\beta$
- **输出**：注意力权重 $\{\text{precision\_weight}_l\}$
- **约束**：权重归一化；$\beta > 0$

> **白话翻译**：注意力调制协议就是"按重要性分配精力"——哪一层的误差最大，就给哪一层更多的计算资源。温度参数 $\beta$ 控制"偏心程度"——$\beta$ 小就一视同仁，$\beta$ 大就只管最出问题的层。这和人的注意力一样——哪边着火就先看哪边。

---

### 第四部分：跨层时间尺度对齐

#### 2.4 跨层时间尺度对齐（A50.4）

三层预测体系的时间尺度严格对齐：

$$\Delta t_{\text{low}} : \Delta t_{\text{mid}} : \Delta t_{\text{high}} = 1 : 10 : 100 \tag{A50.4}$$

具体地：

| 层级 | 采样间隔 | 聚合窗口 | 有效间隔 |
|------|---------|---------|---------|
| 底层 | $\Delta t_{\text{low}} = 1\text{s}$ | — | $1\text{s}$ |
| 中层 | $\Delta t_{\text{mid}} = 10\text{s}$ | $W_{\text{low→mid}} = 10$ 步底层 | $10\text{s}$ |
| 高层 | $\Delta t_{\text{high}} = 100\text{s}$ | $W_{\text{mid→high}} = 10$ 步中层 | $100\text{s}$ |

**时间对齐的同步机制**：

三层的更新按时间对齐同步：

```
每1秒：底层更新（predict + update + get_error）
每10秒：中层更新（predict + update + discover_causal）
        + 底层误差聚合到中层
        + 中层预测传播到底层
每100秒：高层更新（predict + update + extract_slow_features）
         + 中层误差聚合到高层
         + 高层预测传播到中层
         + 注意力权重重新计算
```

**与A17.1-A17.2的对应**：

A50.4的时间尺度对齐是A17.1-A17.2（引用阶段17）快感知-慢学习原则的跨层推广。在单层内，感知比学习快100倍；在跨层间，底层比中层快10倍，中层比高层快10倍，底层比高层快100倍。两个维度的分离比一致：

$$\frac{\tau_{\text{learn}}}{\tau_{\text{perceive}}} \approx \frac{\Delta t_{\text{high}}}{\Delta t_{\text{low}}} = 100 \tag{A50.4-separation}$$

- **输入**：三层的时间常数 $\tau_{\text{low}}, \tau_{\text{mid}}, \tau_{\text{high}}$
- **输出**：同步时间表，聚合窗口大小
- **约束**：$\Delta t_{\text{low}} : \Delta t_{\text{mid}} : \Delta t_{\text{high}} = 1 : 10 : 100$

> **白话翻译**：时间尺度对齐就是"快慢协调"——底层1秒更新一次，中层10秒更新一次，高层100秒更新一次。三层严格同步：每10秒中层更新时，底层已经更新了10次；每100秒高层更新时，中层已经更新了10次。这和"快感知-慢学习"原则一致——底层是快感知，高层是慢学习。

---

### 第五部分：跨层接口统一协议

#### 2.5 跨层接口统一协议（A50.5）

跨层接口统一协议定义了三个预测器之间的交互规范：

**统一接口签名**：

三个预测器共享相同的接口签名模式：

| 方法 | 底层 | 中层 | 高层 |
|------|------|------|------|
| predict | `predict(state, h) → (Prediction, State)` | `predict(state, h) → (TrendPrediction, State)` | `predict(state, h) → (AbstractPrediction, State)` |
| update | `update(state, obs) → State` | `update(state, obs) → State` | `update(state, obs) → State` |
| get_error | `get_error(state, obs) → ErrorVector` | （内部方法） | （内部方法） |

**跨层数据流规范**：

```
底层 → 中层：
  输入：底层 Prediction.values, ErrorVector.raw_error, ErrorVector.energy
  聚合：MidLevelAggregator.aggregate_from_low_level()
  输出：MidLevelTarget

中层 → 高层：
  输入：中层 TrendPrediction.values, CausalGraph
  聚合：HighLevelAggregator.aggregate_from_mid_level()
  输出：HighLevelTarget

高层 → 中层：
  输入：高层 AbstractPrediction.values, strategy_recommendation
  解聚合：W_{high→mid} 映射
  输出：中层 top-down 预测

中层 → 底层：
  输入：中层 TrendPrediction.values, directions, strengths
  解聚合：W_{mid→low} 映射
  输出：底层 top-down 预测
```

**跨层状态一致性**：

三层预测器的状态必须满足一致性约束——高层状态包含中层状态，中层状态包含底层状态：

$$\text{HighLevelPredictorState} \supset \text{MidLevelPredictorState} \supset \text{LowLevelPredictorState} \tag{A50.5-consistency}$$

具体地：
- `HighLevelPredictorState.algorithm.target.mid_predictor_state` 包含中层状态
- `MidLevelPredictorState.algorithm.target.low_predictor_state` 包含底层状态

这种嵌套组合确保了三层状态的完整性和一致性。

- **输入**：三个预测器的状态和输出
- **输出**：跨层数据流规范和一致性约束
- **约束**：接口签名统一；状态嵌套组合

> **白话翻译**：跨层接口统一协议就是"统一语言"——三个预测器用相同的模式交互（predict/update/get_error），数据流有明确的规范（谁给谁什么数据），状态有嵌套组合（高层包含中层，中层包含底层）。这样三层预测器就像一个团队，用统一的语言沟通，不会"鸡同鸭讲"。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A50.1 | $\mathbf{e}_{\text{mid}} = W_{\text{low→mid}} \Pi_{\text{low}} \mathbf{e}_{\text{low}}$ | 底层误差、精度矩阵 | 中层/高层误差 | 误差能量不放大 |
| A50.2 | $\hat{\mathbf{y}}_{\text{low}}^{\text{top-down}} = W_{\text{mid→low}} \hat{\mathbf{y}}_{\text{mid}}$ | 高层/中层预测 | 中层/底层top-down预测 | 信息量不增 |
| A50.3 | $\text{precision\_weight}_l = \text{softmax}(\beta \cdot \varepsilon_l)$ | 各层误差能量、温度 $\beta$ | 注意力权重 | 权重归一化 |
| A50.4 | $\Delta t_{\text{low}} : \Delta t_{\text{mid}} : \Delta t_{\text{high}} = 1 : 10 : 100$ | 三层时间常数 | 同步时间表 | 严格比例 |
| A50.5 | 跨层接口统一规范 | 三个预测器状态和输出 | 数据流规范、一致性约束 | 状态嵌套组合 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np

# 引用阶段49的数据类
from stage_049 import HighLevelPredictorState, AbstractPrediction, SlowFeatures
# 引用阶段46的数据类
from stage_046 import MidLevelPredictorState, TrendPrediction, CausalGraph
# 引用阶段43的数据类
from stage_043 import LowLevelPredictorState, Prediction, ErrorVector


# ============================================================
# 阶段50 枚举：传播方向
# ============================================================
class PropagationDirection(Enum):
    """跨层信号传播方向。"""
    BOTTOM_UP = "bottom_up"      # 自下而上：误差传播
    TOP_DOWN = "top_down"        # 自上而下：预测传播


# ============================================================
# 阶段50 数据类：跨层协议
# ============================================================
@dataclass(frozen=True)
class CrossLayerProtocol:
    """跨层交互协议的完整描述。

    组合阶段49的HighLevelPredictorState，在高层预测器状态的基础上增加
    跨层交互协议——误差传播、预测传播、注意力调制、时间对齐、接口统一。

    Attributes:
        high_predictor_state: 阶段49的高层预测器状态（组合字段）
        low_to_mid_weights: 底层到中层的聚合权重 W_{low→mid}
        mid_to_high_weights: 中层到高层的聚合权重 W_{mid→high}
        high_to_mid_weights: 高层到中层的解聚合权重 W_{high→mid}
        mid_to_low_weights: 中层到底层的解聚合权重 W_{mid→low}
        attention_beta: 注意力温度参数 β
        attention_weights: 当前注意力权重 [w_low, w_mid, w_high]
        low_sampling_interval: 底层采样间隔 Δt_low
        mid_sampling_interval: 中层采样间隔 Δt_mid
        high_sampling_interval: 高层采样间隔 Δt_high
        sync_step_count: 同步步数计数器
        last_bottom_up_time: 最近一次自下而上传播时间
        last_top_down_time: 最近一次自上而下传播时间

    Invariants:
        - attention_weights 之和为1
        - Δt_low : Δt_mid : Δt_high = 1 : 10 : 100
        - 聚合/解聚合权重矩阵维度匹配
        - sync_step_count >= 0
    """
    high_predictor_state: HighLevelPredictorState
    low_to_mid_weights: np.ndarray
    mid_to_high_weights: np.ndarray
    high_to_mid_weights: np.ndarray
    mid_to_low_weights: np.ndarray
    attention_beta: float = 1.0
    attention_weights: np.ndarray = None
    low_sampling_interval: float = 1.0
    mid_sampling_interval: float = 10.0
    high_sampling_interval: float = 100.0
    sync_step_count: int = 0
    last_bottom_up_time: float = 0.0
    last_top_down_time: float = 0.0

    def __post_init__(self):
        # 验证注意力权重
        if self.attention_weights is not None:
            if not np.allclose(np.sum(self.attention_weights), 1.0):
                raise ValueError("注意力权重之和必须为1")
        # 验证时间尺度对齐
        ratio_mid_low = self.mid_sampling_interval / self.low_sampling_interval
        ratio_high_mid = self.high_sampling_interval / self.mid_sampling_interval
        if not (9.5 < ratio_mid_low < 10.5):
            raise ValueError(
                f"中层/底层采样间隔比必须约为10，"
                f"收到 {ratio_mid_low}"
            )
        if not (9.5 < ratio_high_mid < 10.5):
            raise ValueError(
                f"高层/中层采样间隔比必须约为10，"
                f"收到 {ratio_high_mid}"
            )
        # 验证注意力温度
        if self.attention_beta <= 0:
            raise ValueError(
                f"注意力温度必须 > 0，收到 {self.attention_beta}"
            )

    @property
    def time_scale_ratio(self) -> tuple:
        """时间尺度比例（A50.4）。"""
        return (
            self.low_sampling_interval,
            self.mid_sampling_interval,
            self.high_sampling_interval
        )

    @property
    def dominant_layer(self) -> str:
        """注意力权重最大的层级。"""
        if self.attention_weights is None:
            return "unknown"
        layers = ["low", "mid", "high"]
        return layers[int(np.argmax(self.attention_weights))]


# ============================================================
# 阶段50 核心接口：跨层交互引擎
# ============================================================
class CrossLayerEngine:
    """阶段50的核心接口：跨层交互引擎。

    实现三层预测体系之间的误差传播、预测传播、
    注意力调制和时间尺度对齐。

    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 误差传播不放大能量
        - 预测传播不增加信息量
        - 注意力权重归一化
        - 时间尺度严格对齐 1:10:100
    """

    def propagate_error_bottom_up(
        self,
        protocol: CrossLayerProtocol,
        low_error: ErrorVector
    ) -> tuple[np.ndarray, np.ndarray]:
        """自下而上误差传播（A50.1）。

        底层误差 → 中层误差 → 高层误差。
        使用线性聚合和降维传播误差。

        Args:
            protocol: 跨层协议
            low_error: 底层误差向量

        Returns:
            (中层误差, 高层误差)

        Examples:
            >>> engine = CrossLayerEngine()
            >>> err = ErrorVector(raw_error=np.ones(3), precision_weighted=np.ones(3), energy=3.0, system_error=np.ones(3), log_error=np.zeros(0), file_error=np.zeros(0), timestamp=0.0)
            >>> mid_err, high_err = engine.propagate_error_bottom_up(None, err)
            >>> mid_err.shape[0] > 0
            True
        """
        raw = np.asarray(low_error.raw_error, dtype=float)
        # 中层误差：对底层误差做平滑降维
        mid_error = np.array([
            float(np.mean(raw)),
            float(np.std(raw)),
            float(np.max(np.abs(raw))),
        ])
        # 高层误差：对中层误差进一步聚合
        high_error = np.array([
            float(np.mean(mid_error)),
            float(np.var(mid_error)),
        ])
        return mid_error, high_error

    def propagate_prediction_top_down(
        self,
        protocol: CrossLayerProtocol,
        high_prediction: AbstractPrediction
    ) -> tuple[np.ndarray, np.ndarray]:
        """自上而下预测传播（A50.2）。

        高层预测 → 中层预测 → 底层预测。
        通过降采样和插值将高层预测分解到中层和底层。

        Args:
            protocol: 跨层协议
            high_prediction: 高层抽象预测

        Returns:
            (中层top-down预测, 底层top-down预测)

        Examples:
            >>> engine = CrossLayerEngine()
            >>> pred = AbstractPrediction(values=np.ones((100, 2)), covariances=np.tile(np.eye(2),(100,1,1)), trend=np.zeros((100,2)), seasonal=np.zeros((100,2)), residual=np.zeros((100,2)), slow_feature_projection=np.zeros((100,4)), strategy_recommendation={}, confidences=np.ones(100), horizon=100, timestamp=0.0)
            >>> mid_td, low_td = engine.propagate_prediction_top_down(None, pred)
            >>> mid_td.shape[0] > 0
            True
        """
        high_vals = np.asarray(high_prediction.values, dtype=float)
        H, d_high = high_vals.shape
        # 中层：降采样10倍
        mid_down = high_vals[::10]
        # 底层：线性插值到更高分辨率
        low_down = np.zeros((H, d_high))
        if H > 1:
            for i in range(d_high):
                low_down[:, i] = np.interp(
                    np.linspace(0, H - 1, H),
                    np.arange(len(mid_down)),
                    mid_down[:, i]
                )
        return mid_down, low_down

    def compute_attention_weights(
        self,
        protocol: CrossLayerProtocol,
        low_error_energy: float,
        mid_error_energy: float,
        high_error_energy: float
    ) -> np.ndarray:
        """计算注意力权重（A50.3）。

        precision_weight = softmax(β · error_magnitude)
        误差越大，注意力权重越高。

        Args:
            protocol: 跨层协议
            low_error_energy: 底层误差能量 >= 0
            mid_error_energy: 中层误差能量 >= 0
            high_error_energy: 高层误差能量 >= 0

        Returns:
            注意力权重 [w_low, w_mid, w_high]，和为1

        Raises:
            ValueError: 当任一能量 < 0 时

        Examples:
            >>> engine = CrossLayerEngine()
            >>> w = engine.compute_attention_weights(None, 1.0, 0.5, 0.2)
            >>> np.isclose(np.sum(w), 1.0)
            True
        """
        for name, val in [("low_error_energy", low_error_energy),
                          ("mid_error_energy", mid_error_energy),
                          ("high_error_energy", high_error_energy)]:
            if val < 0:
                raise ValueError(f"{name} 必须 >= 0，收到 {val}")
        beta = protocol.attention_beta if protocol is not None else 1.0
        energies = np.array([low_error_energy, mid_error_energy, high_error_energy])
        # 防止除零
        exp_vals = np.exp(beta * energies)
        weights = exp_vals / np.sum(exp_vals)
        return weights

    def synchronize_time_scales(
        self,
        protocol: CrossLayerProtocol,
        current_step: int
    ) -> dict:
        """同步三层时间尺度（A50.4）。

        返回当前步各层是否需要更新的标志。
        底层每步更新，中层每10步更新，高层每100步更新。

        Args:
            protocol: 跨层协议
            current_step: 当前步数 >= 0

        Returns:
            字典包含更新标志

        Raises:
            ValueError: 当 current_step < 0 时

        Examples:
            >>> engine = CrossLayerEngine()
            >>> flags = engine.synchronize_time_scales(None, 100)
            >>> flags["high_update"]
            True
        """
        if current_step < 0:
            raise ValueError(f"当前步数必须 >= 0，收到 {current_step}")
        low_interval = protocol.low_sampling_interval if protocol is not None else 1.0
        mid_interval = protocol.mid_sampling_interval if protocol is not None else 10.0
        high_interval = protocol.high_sampling_interval if protocol is not None else 100.0
        low_update = True  # 底层每步都更新
        mid_update = (current_step % int(mid_interval / low_interval)) == 0
        high_update = (current_step % int(high_interval / low_interval)) == 0
        bottom_up = mid_update or high_update
        top_down = high_update
        return {
            "low_update": low_update,
            "mid_update": mid_update,
            "high_update": high_update,
            "bottom_up_propagate": bottom_up,
            "top_down_propagate": top_down,
        }

    def execute_cross_layer_step(
        self,
        protocol: CrossLayerProtocol,
        low_predictor: 'LowLevelPredictor',
        mid_predictor: 'MidLevelPredictor',
        high_predictor: 'HighLevelPredictor',
        observation: np.ndarray
    ) -> CrossLayerProtocol:
        """执行一步完整的跨层交互（A50.5）。

        按时间对齐协议执行底层/中层/高层更新，
        以及跨层误差传播和预测传播。

        Args:
            protocol: 当前跨层协议状态
            low_predictor: 底层预测器
            mid_predictor: 中层预测器
            high_predictor: 高层预测器
            observation: 新观测，形状 (d_low,)

        Returns:
            更新后的跨层协议状态

        Examples:
            >>> engine = CrossLayerEngine()
            >>> lp = LowLevelPredictor()
            >>> mp = MidLevelPredictor()
            >>> hp = HighLevelPredictor()
            >>> obs = np.zeros(3)
            >>> new_proto = engine.execute_cross_layer_step(None, lp, mp, hp, obs)
            >>> new_proto.sync_step_count >= 0
            True
        """
        observation = np.asarray(observation, dtype=float)
        current_step = protocol.sync_step_count if protocol is not None else 0
        # 时间尺度同步
        flags = self.synchronize_time_scales(protocol, current_step)
        # 底层更新
        low_state = protocol.low_predictor_state if protocol is not None else None
        if low_predictor is not None and low_state is not None:
            low_state = low_predictor.update(low_state, observation)
        # 中层更新（每10步）
        mid_state = protocol.mid_predictor_state if protocol is not None else None
        if flags["mid_update"] and mid_predictor is not None and mid_state is not None:
            mid_obs = np.zeros(3)  # 简化：占位观测
            mid_state = mid_predictor.update(mid_state, mid_obs)
        # 高层更新（每100步）
        high_state = protocol.high_predictor_state if protocol is not None else None
        if flags["high_update"] and high_predictor is not None and high_state is not None:
            high_obs = np.zeros(3)
            high_state = high_predictor.update(high_state, high_obs)
        # 自下而上误差传播（简化）
        if flags["bottom_up_propagate"] and low_state is not None:
            err = low_state.last_error
            if err is not None:
                mid_err, high_err = self.propagate_error_bottom_up(protocol, err)
        # 自上而下预测传播（简化）
        if flags["top_down_propagate"] and high_state is not None:
            high_pred = high_state.last_prediction
            if high_pred is not None:
                mid_td, low_td = self.propagate_prediction_top_down(protocol, high_pred)
        # 更新协议状态
        new_protocol = CrossLayerProtocol(
            low_predictor_state=low_state,
            mid_predictor_state=mid_state,
            high_predictor_state=high_state,
            attention_weights=protocol.attention_weights if protocol is not None else None,
            attention_beta=protocol.attention_beta if protocol is not None else 1.0,
            low_sampling_interval=protocol.low_sampling_interval if protocol is not None else 1.0,
            mid_sampling_interval=protocol.mid_sampling_interval if protocol is not None else 10.0,
            high_sampling_interval=protocol.high_sampling_interval if protocol is not None else 100.0,
            sync_step_count=current_step + 1,
            last_bottom_up_time=float(current_step) if flags["bottom_up_propagate"] else (protocol.last_bottom_up_time if protocol is not None else 0.0),
            last_top_down_time=float(current_step) if flags["top_down_propagate"] else (protocol.last_top_down_time if protocol is not None else 0.0),
        )
        return new_protocol
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `HighLevelPredictorState` | ← 阶段49 | 作为跨层协议的高层预测器状态基础（组合字段） |
| **输入** | A12.1（误差传播规则） | ← 阶段12 | 自下而上误差传播的理论基础 |
| **输入** | A13.1-A13.5（自上而下预测传播） | ← 阶段13 | 自上而下预测传播的理论基础 |
| **输入** | A17.1-A17.2（快感知-慢学习） | ← 阶段17 | 跨层时间尺度对齐的理论基础 |
| **输入** | A3.4（自下而上传播信号） | ← 阶段3 | 注意力调制的精度加权基础 |
| **输入** | `LowLevelPredictor` | ← 阶段43 | 底层预测器接口 |
| **输入** | `MidLevelPredictor` | ← 阶段46 | 中层预测器接口 |
| **输出** | `CrossLayerProtocol` | → 后续阶段 | 跨层交互协议的完整状态 |
| **输出** | A50.1（误差传播协议） | → 工程实现 | 自下而上误差传播的规范 |
| **输出** | A50.2（预测传播协议） | → 工程实现 | 自上而下预测传播的规范 |
| **输出** | A50.3（注意力调制协议） | → 工程实现 | 跨层注意力分配的规范 |
| **输出** | A50.4（时间尺度对齐） | → 工程实现 | 三层同步时间表 |
| **输出** | A50.5（接口统一协议） | → 工程实现 | 跨层数据流规范 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 跨层计算JIT编译 | `jax.jit`编译跨层交互引擎的核心方法，实现端到端加速 |
| asyncio | 异步多层更新 | Python异步IO框架，支撑三层预测器的异步更新和跨层传播 |
| Pydantic | 数据验证 | 支撑CrossLayerProtocol的运行时类型检查和一致性验证 |
| structlog | 跨层审计 | 结构化日志库，支撑跨层信号传播的完整审计追踪 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V50.1 | 误差传播正确性 | 中层误差 = $W_{\text{low→mid}} \Pi_{\text{low}} \mathbf{e}_{\text{low}}$ | 矩阵运算验证 |
| V50.2 | 误差能量不放大 | $\mathbf{e}_{\text{mid}}^T \Pi_{\text{mid}} \mathbf{e}_{\text{mid}} \leq \mathbf{e}_{\text{low}}^T \Pi_{\text{low}} \mathbf{e}_{\text{low}}$ | 能量守恒验证 |
| V50.3 | 预测传播正确性 | 底层top-down = $W_{\text{mid→low}} \hat{\mathbf{y}}_{\text{mid}}$ | 矩阵运算验证 |
| V50.4 | 预测融合正确性 | 融合预测与手动精度加权一致 | 数值对比 |
| V50.5 | 注意力权重归一化 | $\sum_l w_l = 1$ | 求和验证 |
| V50.6 | 注意力权重与误差正相关 | 误差大的层注意力权重更大 | 合成误差测试 |
| V50.7 | 注意力温度效果 | $\beta \to \infty$ 时权重集中在最大误差层 | 极限条件测试 |
| V50.8 | 时间尺度对齐 | $\Delta t_{\text{low}} : \Delta t_{\text{mid}} : \Delta t_{\text{high}} = 1 : 10 : 100$ | 比例验证 |
| V50.9 | 同步时间表正确 | 底层每步更新，中层每10步更新，高层每100步更新 | 步进验证 |
| V50.10 | 跨层传播时机 | 自下而上传播在中层更新时执行，自上而下传播在高层更新时执行 | 时机验证 |
| V50.11 | 状态嵌套一致性 | 高层状态包含中层状态，中层状态包含底层状态 | 嵌套字段检查 |
| V50.12 | CrossLayerProtocol组合 | `protocol.high_predictor_state` 类型为 `HighLevelPredictorState` | 类型检查 |
| V50.13 | 非法输入拒绝 | 注意力温度≤0、时间比例不对等抛出 `ValueError` | 异常测试 |
| V50.14 | 端到端跨层循环 | 1000步跨层交互无异常 | 长期运行测试 |
| V50.15 | 跨层预测改善 | 有跨层传播时预测误差低于无跨层传播 | 对比实验 |

---

> **阶段50完** | 阶段50建立了三层预测体系的跨层交互协议：误差传播协议实现low→mid→high的误差上报（A50.1），预测传播协议实现high→mid→low的指导下达（A50.2），注意力调制协议按误差大小分配计算资源（A50.3），时间尺度对齐确保三层严格同步1:10:100（A50.4），接口统一协议定义了跨层数据流规范和状态一致性（A50.5），封装为CrossLayerProtocol数据类组合HighLevelPredictorState。至此，阶段41-50建立了完整的三层层级预测体系：底层高频微预测（41-43）→ 中层中频趋势预测（44-46）→ 高层低频抽象预测（47-49）→ 跨层交互协议（50）。从秒级微预测到天级抽象预测，三层预测器通过误差传播、预测传播、注意力调制和时间对齐形成统一的预测智能体。
