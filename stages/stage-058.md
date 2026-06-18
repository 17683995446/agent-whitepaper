# 阶段58：误差反馈到模型的映射——微修正触发、结构重构与自适应增益

> **阶段编号**：S58 | **所属层**：第六层·误差评估层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 小误差→微修正 | "小毛病微调"——误差小就只做细微调整 |
> | 大误差→结构重构 | "大毛病大修"——误差大就需要重构模型结构 |
> | 误差方向→更新方向 | "朝哪个方向修"——误差的梯度决定修正方向 |
> | 反馈增益自适应 | "修多少看情况"——根据误差大小和趋势调整修正力度 |
> | ErrorModelMapping | "误差-模型映射的完整画像"——组合ErrorDriveMapping，在驱力映射基础上增加模型映射 |

---

## 1. 第一性原理依据

本阶段从阶段14（表征更新规则）、阶段15（权重更新规则）和阶段57（误差反馈到驱力的映射）的核心结论出发，建立误差反馈到模型的映射机制。核心依据为：

- **A14.1（表征更新方程）**（引用阶段14）：$\tau_r \dot{r}^l = \delta^{l+1} - e^l$，表征更新由自由能梯度驱动。本阶段将A14.1从连续时间推广到误差驱动的触发机制——小误差触发微修正（连续时间梯度的离散近似），大误差触发结构重构（超出梯度更新的范围）。

- **A15.1（权重更新方程）**（引用阶段15）：$\tau_W \dot{W}^l = -\frac{\partial f^l}{\partial W^l} \cdot \delta^l$，权重更新由自由能关于权重的梯度驱动。本阶段将A15.1从连续时间推广到误差驱动的触发机制——小误差触发微修正（Hebbian学习），大误差触发结构重构（需要重新搜索模型空间）。

- **A57.1-A57.4（误差反馈到驱力的映射）**（引用阶段57）：异常→安全驱力、偏移→安全驱力、误差下降→收益驱力、误差上升→收益缺口。本阶段在驱力映射的基础上增加模型映射——驱力映射决定"是否行动"，模型映射决定"如何行动"。

- **A6.5（感知方程）**（引用阶段6）：$\dot{\mu} = -\kappa_\mu \partial F / \partial \mu$，感知更新的梯度流方程。本阶段的微修正触发是A6.5的误差驱动版本——当误差小时，按梯度微修正；当误差大时，梯度更新不够，需要结构重构。

本阶段的核心目标是：**定义小误差→微修正触发 $\mu_{\text{update}} = -\kappa \partial F / \partial \mu$、大误差→结构重构触发、误差方向→更新方向 $\partial F / \partial \mu \propto \varepsilon \cdot \partial \hat{o} / \partial \mu$、反馈增益自适应 $\text{gain} = f(\varepsilon_{\text{magnitude}}, \varepsilon_{\text{trend}})$，封装为ErrorModelMapping数据类，组合ErrorDriveMapping作为驱力映射的输入。** 误差反馈到模型的映射是"感知→学习"的关键环节——它将误差评估的结果转化为模型更新的具体操作。

> **去重声明**：本阶段仅定义误差反馈到模型的映射规则，不重新定义表征更新（阶段14）或权重更新（阶段15）的具体算法。结构重构的详细机制将在阶段64定义，本阶段仅定义触发条件。

---

## 2. 核心数学公式

### 第一部分：小误差→微修正触发

#### 2.1 小误差→微修正触发定义（A58.1）

当误差较小时，触发基于自由能梯度的微修正：

$$\mu_{\text{update}} = -\kappa \cdot \frac{\partial F}{\partial \mu} \tag{A58.1}$$

其中：
- $\kappa > 0$ 为更新步长（学习率）
- $\mu$ 为模型参数（表征或权重）
- $\partial F / \partial \mu$ 为自由能关于参数的梯度

**微修正触发的条件**：

微修正仅在误差较小时触发——"较小时"定义为：

$$\varepsilon_{\text{weighted}} < \varepsilon_{\text{micro\_threshold}} \tag{A58.1-condition}$$

其中 $\varepsilon_{\text{micro\_threshold}}$ 为微修正阈值（通常设为稳态误差均值的2倍）。

**与A14.1/A15.1的对应**：

A58.1是A14.1（引用阶段14）$\tau_r \dot{r}^l = -\partial F / \partial r^l$ 和A15.1（引用阶段15）$\tau_W \dot{W}^l = -\partial F / \partial W^l$ 的误差驱动触发版本。区别在于：
- A14.1/A15.1：每步都执行梯度更新（连续时间）
- A58.1：仅在误差较小时执行梯度更新（误差驱动触发）

**微修正的步长选择**：

步长 $\kappa$ 应与误差大小成反比——误差越小，步长越小（精细调整）：

$$\kappa = \kappa_0 \cdot \frac{\varepsilon_{\text{micro\_threshold}}}{\varepsilon_{\text{weighted}} + \varepsilon_{\text{micro\_threshold}}} \tag{A58.1-step-size}$$

当 $\varepsilon_{\text{weighted}} \to 0$ 时，$\kappa \to \kappa_0$（正常步长）；当 $\varepsilon_{\text{weighted}} \to \varepsilon_{\text{micro\_threshold}}$ 时，$\kappa \to \kappa_0 / 2$（步长减半）。

- **输入**：自由能梯度 $\partial F / \partial \mu$，步长 $\kappa$，误差 $\varepsilon_{\text{weighted}}$
- **输出**：参数更新量 $\mu_{\text{update}}$
- **约束**：$\kappa > 0$；$\varepsilon_{\text{weighted}} < \varepsilon_{\text{micro\_threshold}}$

> **白话翻译**：小误差→微修正就是"小毛病微调"——误差小时，按自由能梯度的反方向做细微调整。步长和误差大小成反比——误差越小，调整越精细。这和阶段14/15的梯度更新一样，但只在误差小时才触发——误差大时需要更激进的措施。

---

### 第二部分：大误差→结构重构触发

#### 2.2 大误差→结构重构触发定义（A58.2）

当误差较大时，梯度微修正不足以恢复系统性能，需要触发结构重构：

$$\varepsilon_{\text{weighted}} \geq \varepsilon_{\text{restructure\_threshold}} \implies \text{trigger\_restructure} \tag{A58.2}$$

其中 $\varepsilon_{\text{restructure\_threshold}}$ 为结构重构阈值（通常设为稳态误差均值的5倍）。

**结构重构的触发层次**：

| 误差范围 | 触发动作 | 说明 |
|---------|---------|------|
| $\varepsilon < \varepsilon_{\text{micro}}$ | 微修正（A58.1） | 梯度更新 |
| $\varepsilon_{\text{micro}} \leq \varepsilon < \varepsilon_{\text{restructure}}$ | 增强更新 | 增大步长+扩大搜索范围 |
| $\varepsilon \geq \varepsilon_{\text{restructure}}$ | 结构重构 | 重新搜索模型空间 |

**结构重构的具体操作**：

结构重构的详细机制将在阶段64定义。本阶段仅定义触发条件，具体操作包括：
1. 模型结构搜索（增加/删除隐层单元）
2. 超参数重新优化（学习率、正则化系数等）
3. 知识蒸馏（从旧模型迁移知识到新模型）

**与A57.1-A57.2的协同**：

大误差触发结构重构的同时，也会触发安全驱力（通过A57.1的异常→安全驱力映射）。两者的协同：
- 安全驱力：降低风险暴露（防御性）
- 结构重构：修复模型缺陷（进取性）

- **输入**：误差 $\varepsilon_{\text{weighted}}$，重构阈值 $\varepsilon_{\text{restructure\_threshold}}$
- **输出**：结构重构触发标志
- **约束**：$\varepsilon_{\text{restructure\_threshold}} > \varepsilon_{\text{micro\_threshold}}$

> **白话翻译**：大误差→结构重构就是"大毛病大修"——误差大到梯度微修正不够用时，需要重构模型结构。重构比微修正更激进——可能增加模型容量、重新优化超参数、甚至重建模型。重构的详细机制在阶段64定义，本阶段只定义"什么时候该重构"。

---

### 第三部分：误差方向→更新方向

#### 2.3 误差方向→更新方向定义（A58.3）

模型参数的更新方向由误差信号和预测函数的梯度共同决定：

$$\frac{\partial F}{\partial \mu} \propto \varepsilon \cdot \frac{\partial \hat{o}}{\partial \mu} \tag{A58.3}$$

其中：
- $\varepsilon = o - \hat{o}$ 为误差信号
- $\partial \hat{o} / \partial \mu$ 为预测函数关于参数的梯度
- $\mu$ 为模型参数

**与A15.1的对应**：

A58.3是A15.1（引用阶段15）$\tau_W \dot{W}^l = -(\partial f^l / \partial W^l) \cdot \delta^l$ 的简化形式。在二次型自由能下：

$$\frac{\partial F}{\partial \mu} = \frac{\partial}{\partial \mu}\left[\frac{1}{2}(o - \hat{o})^T \Pi (o - \hat{o})\right] = -\Pi \varepsilon \cdot \frac{\partial \hat{o}}{\partial \mu} \tag{A58.3-derivation}$$

因此更新方向 = $-\partial F / \partial \mu = \Pi \varepsilon \cdot \partial \hat{o} / \partial \mu$，即精度加权误差 × 预测梯度。

**更新方向的物理含义**：

更新方向由两个因素决定：
1. **误差方向 $\varepsilon$**：预测偏离观测的方向——"哪里错了"
2. **预测梯度 $\partial \hat{o} / \partial \mu$**：参数对预测的影响——"怎么改能修正"

两者相乘得到"朝哪个方向修"——误差方向告诉我们"哪里错了"，预测梯度告诉我们"怎么改能修正"。

**逐层更新方向**：

对每一层分别计算更新方向：

$$\Delta \mu_l \propto \Pi_l \varepsilon_l \cdot \frac{\partial \hat{o}_l}{\partial \mu_l}, \quad l \in \{\text{low, mid, high}\} \tag{A58.3-per-layer}$$

- **输入**：误差信号 $\varepsilon$，预测梯度 $\partial \hat{o} / \partial \mu$，精度矩阵 $\Pi$
- **输出**：参数更新方向 $\Delta \mu$
- **约束**：更新方向与自由能梯度方向一致

> **白话翻译**：误差方向→更新方向就是"朝哪个方向修"——误差方向告诉我们"哪里错了"，预测梯度告诉我们"怎么改能修正"。两者相乘得到修正方向。精度加权意味着"越可信的误差，修正力度越大"——和A3.4的自下而上传播信号一致。

---

### 第四部分：反馈增益自适应

#### 2.4 反馈增益自适应定义（A58.4）

反馈增益根据误差大小和趋势自适应调整：

$$\text{gain} = f(\varepsilon_{\text{magnitude}}, \varepsilon_{\text{trend}}) \tag{A58.4}$$

具体形式为：

$$\text{gain} = \text{gain}_{\text{base}} \cdot \left(1 + \eta \cdot \frac{\varepsilon_{\text{magnitude}}}{\varepsilon_{\text{baseline}}}\right) \cdot \left(1 + \zeta \cdot \max(0, \varepsilon_{\text{trend}} - 1)\right) \tag{A58.4-formula}$$

其中：
- $\text{gain}_{\text{base}} > 0$ 为基础增益
- $\eta \geq 0$ 为误差大小增益系数
- $\zeta \geq 0$ 为误差趋势增益系数
- $\varepsilon_{\text{magnitude}}$ 为当前误差大小
- $\varepsilon_{\text{baseline}}$ 为误差基线
- $\varepsilon_{\text{trend}}$ 为误差趋势（引用A52.4的多尺度趋势）

**增益自适应的物理含义**：

| 条件 | 增益变化 | 含义 |
|------|---------|------|
| $\varepsilon_{\text{magnitude}}$ 大 | 增益增大 | 误差大→更激进的更新 |
| $\varepsilon_{\text{magnitude}}$ 小 | 增益不变 | 误差小→正常更新 |
| $\varepsilon_{\text{trend}} > 1$（恶化） | 增益增大 | 误差在恶化→加速更新 |
| $\varepsilon_{\text{trend}} < 1$（改善） | 增益不变 | 误差在改善→维持当前策略 |
| $\varepsilon_{\text{trend}} \approx 1$（稳定） | 增益不变 | 误差稳定→维持当前策略 |

**增益上界**：

为避免增益过大导致更新不稳定，设置增益上界：

$$\text{gain} \leq \text{gain}_{\max} \tag{A58.4-upper}$$

通常 $\text{gain}_{\max} = 10 \cdot \text{gain}_{\text{base}}$。

**增益与A58.1步长的关系**：

A58.4的自适应增益与A58.1的步长 $\kappa$ 的关系为：

$$\kappa_{\text{effective}} = \kappa_0 \cdot \text{gain} \tag{A58.4-kappa}$$

即有效步长 = 基础步长 × 自适应增益。

- **输入**：误差大小 $\varepsilon_{\text{magnitude}}$，误差趋势 $\varepsilon_{\text{trend}}$，基线，增益参数
- **输出**：自适应增益 $\text{gain} \in [\text{gain}_{\text{base}}, \text{gain}_{\max}]$
- **约束**：$\text{gain} \geq \text{gain}_{\text{base}}$；$\text{gain} \leq \text{gain}_{\max}$

> **白话翻译**：反馈增益自适应就是"修多少看情况"——误差大就多修，误差小就少修；误差在恶化就加速修，误差在改善就维持当前修法。增益有上界，避免"修过头"。这和人的学习一样——发现大问题就下大力气改，小问题就微调。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A58.1 | $\mu_{\text{update}} = -\kappa \cdot \partial F / \partial \mu$ | 自由能梯度、步长、误差 | 参数更新量 | $\varepsilon < \varepsilon_{\text{micro}}$ |
| A58.2 | $\varepsilon \geq \varepsilon_{\text{restructure}} \implies$ 触发重构 | 误差、重构阈值 | 触发标志 | $\varepsilon_{\text{restructure}} > \varepsilon_{\text{micro}}$ |
| A58.3 | $\partial F / \partial \mu \propto \varepsilon \cdot \partial \hat{o} / \partial \mu$ | 误差信号、预测梯度 | 更新方向 | 与自由能梯度一致 |
| A58.4 | $\text{gain} = f(\varepsilon_{\text{magnitude}}, \varepsilon_{\text{trend}})$ | 误差大小、趋势、增益参数 | 自适应增益 | $\text{gain} \in [\text{gain}_{\text{base}}, \text{gain}_{\max}]$ |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict
from enum import Enum
import numpy as np

# 引用阶段57的数据类
from stage_057 import ErrorDriveMapping


# ============================================================
# 阶段58 枚举：更新类型
# ============================================================
class UpdateType(Enum):
    """模型更新类型。"""
    MICRO_CORRECTION = "micro_correction"    # 微修正（A58.1）
    ENHANCED_UPDATE = "enhanced_update"      # 增强更新
    STRUCTURE_RESTRUCTURE = "structure_restructure"  # 结构重构（A58.2）


# ============================================================
# 阶段58 数据类：误差-模型映射
# ============================================================
@dataclass(frozen=True)
class ErrorModelMapping:
    """误差反馈到模型的映射描述。

    组合阶段57的ErrorDriveMapping，在驱力映射的基础上增加
    模型映射——微修正触发、结构重构触发、更新方向、自适应增益。

    Attributes:
        drive_mapping: 阶段57的驱力映射（组合字段）
        micro_threshold: 微修正阈值 ε_micro
        restructure_threshold: 结构重构阈值 ε_restructure
        current_update_type: 当前更新类型
        update_direction: 参数更新方向
        adaptive_gain: 自适应增益
        gain_base: 基础增益
        gain_max: 增益上界
        eta: 误差大小增益系数
        zeta: 误差趋势增益系数
        restructure_triggered: 是否触发了结构重构
        kappa: 当前有效步长

    Invariants:
        - micro_threshold > 0
        - restructure_threshold > micro_threshold
        - adaptive_gain >= gain_base
        - adaptive_gain <= gain_max
        - eta, zeta >= 0
    """
    drive_mapping: ErrorDriveMapping
    micro_threshold: float = 2.0
    restructure_threshold: float = 10.0
    current_update_type: UpdateType = UpdateType.MICRO_CORRECTION
    update_direction: Optional[np.ndarray] = None
    adaptive_gain: float = 1.0
    gain_base: float = 1.0
    gain_max: float = 10.0
    eta: float = 1.0
    zeta: float = 1.0
    restructure_triggered: bool = False
    kappa: float = 0.01

    def __post_init__(self):
        if self.micro_threshold <= 0:
            raise ValueError(
                f"微修正阈值必须 > 0，收到 {self.micro_threshold}"
            )
        if self.restructure_threshold <= self.micro_threshold:
            raise ValueError(
                f"结构重构阈值必须 > 微修正阈值，"
                f"收到 restructure={self.restructure_threshold}, "
                f"micro={self.micro_threshold}"
            )
        if self.adaptive_gain < self.gain_base:
            raise ValueError(
                f"自适应增益必须 >= 基础增益，"
                f"收到 gain={self.adaptive_gain}, base={self.gain_base}"
            )
        if self.adaptive_gain > self.gain_max:
            raise ValueError(
                f"自适应增益必须 <= 增益上界，"
                f"收到 gain={self.adaptive_gain}, max={self.gain_max}"
            )
        if self.eta < 0 or self.zeta < 0:
            raise ValueError("eta 和 zeta 必须 >= 0")

    @property
    def effective_step_size(self) -> float:
        """有效步长 = 基础步长 × 自适应增益（A58.4-kappa）。"""
        return self.kappa * self.adaptive_gain

    @property
    def is_micro_correction(self) -> bool:
        """是否为微修正模式。"""
        return self.current_update_type == UpdateType.MICRO_CORRECTION

    @property
    def is_restructure(self) -> bool:
        """是否为结构重构模式。"""
        return self.current_update_type == UpdateType.STRUCTURE_RESTRUCTURE


# ============================================================
# 阶段58 核心接口：误差-模型映射器
# ============================================================
class ErrorModelMapper:
    """阶段58的核心接口：误差-模型映射器。

    将误差反馈映射到模型更新操作——微修正、结构重构、
    更新方向和自适应增益。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 微修正阈值 < 结构重构阈值
        - 自适应增益 ∈ [gain_base, gain_max]
        - 更新方向与自由能梯度方向一致
    """

    def determine_update_type(
        self,
        error: float,
        micro_threshold: float,
        restructure_threshold: float
    ) -> UpdateType:
        """确定更新类型（A58.1/A58.2）。

        Args:
            error: 当前精度加权误差
            micro_threshold: 微修正阈值
            restructure_threshold: 结构重构阈值

        Returns:
            更新类型

        Raises:
            ValueError: 当阈值不合法时
        """
        if micro_threshold <= 0:
            raise ValueError(f"微修正阈值必须 > 0，收到 {micro_threshold}")
        if restructure_threshold <= micro_threshold:
            raise ValueError(
                f"结构重构阈值必须 > 微修正阈值，"
                f"收到 restructure={restructure_threshold}, micro={micro_threshold}"
            )
        if error < micro_threshold:
            return UpdateType.MICRO_CORRECTION
        elif error < restructure_threshold:
            return UpdateType.ENHANCED_UPDATE
        else:
            return UpdateType.STRUCTURE_RESTRUCTURE

    def compute_update_direction(
        self,
        error_signal: np.ndarray,
        prediction_gradient: np.ndarray,
        precision: np.ndarray
    ) -> np.ndarray:
        """计算更新方向（A58.3）。

        Δμ ∝ Π·ε · ∂ô/∂μ

        Args:
            error_signal: 误差信号 ε
            prediction_gradient: 预测梯度 ∂ô/∂μ
            precision: 精度矩阵 Π

        Returns:
            更新方向
        """
        eps = np.asarray(error_signal)
        grad = np.asarray(prediction_gradient)
        Pi = np.asarray(precision)
        # 精度加权误差
        weighted_error = Pi @ eps if Pi.ndim >= 2 else Pi * eps
        # 更新方向 = 精度加权误差 * 预测梯度
        update_dir = weighted_error * grad
        return update_dir

    def compute_adaptive_gain(
        self,
        error_magnitude: float,
        error_trend: float,
        error_baseline: float,
        gain_base: float,
        gain_max: float,
        eta: float,
        zeta: float
    ) -> float:
        """计算自适应增益（A58.4）。

        gain = gain_base · (1 + η·ε_mag/ε_baseline) · (1 + ζ·max(0, trend-1))

        Args:
            error_magnitude: 误差大小
            error_trend: 误差趋势
            error_baseline: 误差基线
            gain_base: 基础增益
            gain_max: 增益上界
            eta: 误差大小增益系数
            zeta: 误差趋势增益系数

        Returns:
            自适应增益 ∈ [gain_base, gain_max]
        """
        if gain_base <= 0 or gain_max < gain_base:
            raise ValueError(
                f"增益参数非法: gain_base={gain_base}, gain_max={gain_max}"
            )
        if eta < 0 or zeta < 0:
            raise ValueError(f"eta 和 zeta 必须 >= 0，收到 eta={eta}, zeta={zeta}")
        if error_baseline <= 0:
            raise ValueError(f"误差基线必须 > 0，收到 {error_baseline}")
        scale_factor = 1.0 + eta * (error_magnitude / error_baseline)
        trend_factor = 1.0 + zeta * max(0.0, error_trend - 1.0)
        gain = gain_base * scale_factor * trend_factor
        return float(min(gain, gain_max))

    def step(
        self,
        mapping: ErrorModelMapping,
        error: float,
        error_signal: np.ndarray,
        prediction_gradient: np.ndarray,
        precision: np.ndarray,
        error_trend: float
    ) -> ErrorModelMapping:
        """执行一步完整的误差-模型映射。

        Args:
            mapping: 当前映射状态
            error: 当前精度加权误差
            error_signal: 误差信号
            prediction_gradient: 预测梯度
            precision: 精度矩阵
            error_trend: 误差趋势

        Returns:
            更新后的ErrorModelMapping
        """
        from dataclasses import replace
        update_type = self.determine_update_type(
            error, mapping.micro_threshold, mapping.restructure_threshold
        )
        update_dir = self.compute_update_direction(
            error_signal, prediction_gradient, precision
        )
        adaptive_gain = self.compute_adaptive_gain(
            error_magnitude=abs(error),
            error_trend=error_trend,
            error_baseline=mapping.micro_threshold,
            gain_base=mapping.gain_base,
            gain_max=mapping.gain_max,
            eta=mapping.eta,
            zeta=mapping.zeta,
        )
        restructure_triggered = update_type == UpdateType.STRUCTURE_RESTRUCTURE
        return replace(
            mapping,
            current_update_type=update_type,
            update_direction=update_dir,
            adaptive_gain=adaptive_gain,
            restructure_triggered=restructure_triggered,
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `ErrorDriveMapping` | ← 阶段57 | 作为模型映射的驱力映射输入（组合字段） |
| **输入** | A14.1（表征更新方程） | ← 阶段14 | 微修正的理论基础 |
| **输入** | A15.1（权重更新方程） | ← 阶段15 | 微修正和更新方向的理论基础 |
| **输入** | A6.5（感知方程） | ← 阶段6 | 梯度流的理论基础 |
| **输出** | `ErrorModelMapping` | → 阶段59 | 误差-模型映射状态，作为Python接口的输入 |
| **输出** | A58.1-A58.4（模型映射定义） | → 阶段59-60 | 模型映射的基础定义 |
| **输出** | 结构重构触发信号 | → 阶段64 | 触发结构重构的条件 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 梯度计算与JIT编译 | `jax.grad`自动微分计算自由能梯度，`jax.jit`编译更新方向计算 |
| Optax | 自适应步长 | `optax`优化器库支撑自适应增益的步长管理 |
| NumPy | 更新方向计算 | 支撑精度加权误差与预测梯度的矩阵运算 |
| Pydantic | 数据验证 | 支撑ErrorModelMapping的运行时类型检查 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V58.1 | 微修正触发正确性 | $\varepsilon < \varepsilon_{\text{micro}}$ 时触发微修正 | 条件测试 |
| V58.2 | 微修正更新方向 | $\mu_{\text{update}} = -\kappa \cdot \partial F / \partial \mu$ | 梯度验证 |
| V58.3 | 结构重构触发正确性 | $\varepsilon \geq \varepsilon_{\text{restructure}}$ 时触发重构 | 条件测试 |
| V58.4 | 增强更新触发 | $\varepsilon_{\text{micro}} \leq \varepsilon < \varepsilon_{\text{restructure}}$ 时增强更新 | 条件测试 |
| V58.5 | 更新方向正确性 | $\Delta \mu \propto \Pi \varepsilon \cdot \partial \hat{o} / \partial \mu$ | 数值对比 |
| V58.6 | 更新方向与梯度一致 | 更新方向 = $-\partial F / \partial \mu$ | 梯度验证 |
| V58.7 | 自适应增益范围 | $\text{gain} \in [\text{gain}_{\text{base}}, \text{gain}_{\max}]$ | 范围检查 |
| V58.8 | 误差大时增益增大 | $\varepsilon_{\text{magnitude}}$ 增大时 gain 增大 | 单调性测试 |
| V58.9 | 趋势恶化时增益增大 | $\varepsilon_{\text{trend}} > 1$ 时 gain 增大 | 条件测试 |
| V58.10 | 趋势改善时增益不变 | $\varepsilon_{\text{trend}} < 1$ 时 gain 不增 | 条件测试 |
| V58.11 | 有效步长正确 | $\kappa_{\text{effective}} = \kappa_0 \cdot \text{gain}$ | 数值验证 |
| V58.12 | ErrorModelMapping组合 | `mapping.drive_mapping` 类型为 `ErrorDriveMapping` | 类型检查 |
| V58.13 | 非法输入拒绝 | 阈值不合法、增益参数非法抛出 `ValueError` | 异常测试 |
| V58.14 | 阈值层次正确 | $\varepsilon_{\text{restructure}} > \varepsilon_{\text{micro}} > 0$ | 参数验证 |
| V58.15 | 端到端模型映射 | 1000步误差变化循环无异常 | 长期运行测试 |

---

> **阶段58完** | 阶段58建立了误差反馈到模型的映射机制：小误差→微修正触发 $\mu_{\text{update}} = -\kappa \partial F / \partial \mu$（A58.1），大误差→结构重构触发（A58.2），误差方向→更新方向 $\partial F / \partial \mu \propto \varepsilon \cdot \partial \hat{o} / \partial \mu$（A58.3），反馈增益自适应 $\text{gain} = f(\varepsilon_{\text{magnitude}}, \varepsilon_{\text{trend}})$（A58.4），封装为ErrorModelMapping数据类组合ErrorDriveMapping。下一步：阶段59将定义误差评估模块的统一Python接口。
