# 阶段57：误差反馈到驱力的映射——从异常/偏移信号到驱力激活

> **阶段编号**：S57 | **所属层**：第六层·误差评估层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 异常→安全驱力 | "出问题了→避险本能激活"——异常越严重，避险冲动越强 |
> | 偏移→安全驱力 | "环境变了→警惕性提高"——偏移越大，警惕性越高 |
> | 误差下降→收益驱力 | "变好了→继续赚钱"——误差下降越多，收益驱力越强 |
> | 误差上升→收益缺口 | "变差了→差距拉大"——误差上升越多，收益缺口越大 |
> | ErrorDriveMapping | "误差-驱力映射的完整画像"——组合DistributionShiftDetector，在分布偏移检测基础上增加驱力映射 |

---

## 1. 第一性原理依据

本阶段从阶段21（安全驱力）、阶段29（驱力-自由能统一）和阶段56（分布偏移检测）的核心结论出发，建立误差反馈到驱力的映射机制。核心依据为：

- **A21.4（安全驱力与自由能）**（引用阶段21）：$R_{\text{safe}} = -\alpha_s \cdot \max(0, F - F_{\text{baseline}})$，安全驱力在自由能偏离基线时激活。本阶段将A21.4从自由能偏离推广到异常误差和分布偏移——异常误差和分布偏移都是自由能偏离的具体表现。

- **A29.1（安全驱力=自由能异常项）**（引用阶段29）：$R_{\text{safe}} \propto -(F - F_{\text{baseline}})$，安全驱力度量自由能偏离基线的程度。本阶段将A29.1具体化为异常误差→安全驱力和分布偏移→安全驱力两条映射路径。

- **A29.2（收益驱力=自由能下降率）**（引用阶段29）：$R_{\text{gain}} \propto -\dot{F}$，收益驱力与自由能下降率成正比。本阶段将A29.2具体化为误差下降→收益驱力激活和误差上升→收益缺口增大两条映射路径。

- **A55.4（组合异常分数）**（引用阶段55）：$\text{anomaly\_score} = w_1 \cdot \text{threshold} + w_2 \cdot \text{CUSUM} + w_3 \cdot \text{EWMA}$，组合异常分数为安全驱力提供量化输入。

- **A56.4（偏移程度量化）**（引用阶段56）：$\text{shift\_magnitude} = D / D_{\text{threshold}}$，偏移程度为安全驱力提供分布偏移的量化输入。

本阶段的核心目标是：**定义异常误差→安全驱力激活 $R_{\text{safe}} += \alpha \cdot \text{anomaly\_score}$、分布偏移→安全驱力激活 $R_{\text{safe}} += \beta \cdot \text{shift\_magnitude}$、误差下降→收益驱力激活 $R_{\text{gain}} += \gamma \cdot |\Delta F|$、误差上升→收益缺口增大 $\Delta V_{\text{gain}} += \delta \cdot \Delta F$，封装为ErrorDriveMapping数据类，组合DistributionShiftDetector作为分布偏移检测的输入。** 误差反馈到驱力的映射是"感知→行动"的关键环节——它将误差评估的结果转化为驱力系统的激活信号。

> **去重声明**：本阶段仅定义误差反馈到驱力的映射规则，不重新定义驱力本身（阶段21/24/29已定义）或模型更新规则（阶段58）。驱力的完整计算公式在阶段21/24/29，本阶段仅定义映射的增量。

---

## 2. 核心数学公式

### 第一部分：异常误差→安全驱力

#### 2.1 异常误差→安全驱力映射（A57.1）

异常误差触发安全驱力的增量激活：

$$R_{\text{safe}} \mathrel{+}= \alpha \cdot \text{anomaly\_score} \tag{A57.1}$$

其中：
- $\alpha > 0$ 为异常-安全驱力映射系数
- $\text{anomaly\_score}$ 为组合异常分数（引用A55.4）

**与A21.4的对应**：

A57.1是A21.4（引用阶段21）$R_{\text{safe}} = -\alpha_s \cdot \max(0, F - F_{\text{baseline}})$ 的具体化——异常分数 $\text{anomaly\_score}$ 是自由能偏离 $F - F_{\text{baseline}}$ 的量化近似。当异常分数增大时，自由能偏离基线，安全驱力被激活。

**映射系数 $\alpha$ 的选择**：

$\alpha$ 控制异常对安全驱力的影响强度：

| $\alpha$ | 含义 | 适用场景 |
|----------|------|---------|
| 小（如0.1） | 异常对安全驱力影响弱 | 环境噪声大，需要容忍偶尔的异常 |
| 中（如1.0） | 异常对安全驱力影响适中 | 一般场景 |
| 大（如5.0） | 异常对安全驱力影响强 | 安全关键场景，异常必须立即响应 |

**逐层异常→安全驱力**：

不同层的异常对安全驱力的影响可以不同：

$$R_{\text{safe}} \mathrel{+}= \sum_l \alpha_l \cdot \text{anomaly\_score}_l \tag{A57.1-per-layer}$$

高层异常（长期模式变化）对安全驱力的影响通常大于底层异常（短期噪声）。

- **输入**：异常分数 $\text{anomaly\_score}$，映射系数 $\alpha$
- **输出**：安全驱力增量 $\Delta R_{\text{safe}} \in \mathbb{R}_{\geq 0}$
- **约束**：$\alpha > 0$；$\text{anomaly\_score} \geq 0$

> **白话翻译**：异常→安全驱力就是"出问题了→避险本能激活"——异常分数越高，安全驱力增量越大。$\alpha$ 是"避险灵敏度"——$\alpha$ 大则一有异常就强烈避险，$\alpha$ 小则对异常更容忍。高层异常比底层异常对安全驱力的影响更大——"大方向出问题"比"细节出问题"更可怕。

---

### 第二部分：分布偏移→安全驱力

#### 2.2 分布偏移→安全驱力映射（A57.2）

分布偏移触发安全驱力的增量激活：

$$R_{\text{safe}} \mathrel{+}= \beta \cdot \text{shift\_magnitude} \tag{A57.2}$$

其中：
- $\beta > 0$ 为偏移-安全驱力映射系数
- $\text{shift\_magnitude}$ 为偏移程度（引用A56.4）

**与A29.1的对应**：

A57.2是A29.1（引用阶段29）$R_{\text{safe}} \propto -(F - F_{\text{baseline}})$ 的分布偏移具体化——分布偏移意味着环境发生了根本性变化，自由能必然偏离基线，安全驱力应被激活。

**偏移→安全驱力与异常→安全驱力的区别**：

| 特性 | 异常→安全驱力 | 偏移→安全驱力 |
|------|-------------|-------------|
| 检测对象 | 误差值异常 | 误差分布变化 |
| 触发条件 | 误差超过阈值 | 分布形状改变 |
| 响应速度 | 快（1-5步） | 慢（需要足够样本） |
| 影响范围 | 局部 | 全局 |
| 物理含义 | "当前出问题了" | "环境变了" |

**偏移→安全驱力的特殊意义**：

分布偏移比异常误差更值得关注——异常可能是暂时的噪声，但分布偏移意味着环境发生了根本性变化，系统的模型可能已经过时。因此 $\beta$ 通常应大于 $\alpha$：

$$\beta > \alpha \tag{A57.2-beta-gt-alpha}$$

- **输入**：偏移程度 $\text{shift\_magnitude}$，映射系数 $\beta$
- **输出**：安全驱力增量 $\Delta R_{\text{safe}} \in \mathbb{R}_{\geq 0}$
- **约束**：$\beta > 0$；$\text{shift\_magnitude} \geq 0$

> **白话翻译**：偏移→安全驱力就是"环境变了→警惕性提高"——偏移程度越大，安全驱力增量越大。和异常→安全驱力的区别是：异常是"当前出问题了"（可能是暂时的），偏移是"环境变了"（根本性的）。所以偏移对安全驱力的影响通常比异常更大——"环境变了"比"暂时出问题"更值得警惕。

---

### 第三部分：误差下降→收益驱力

#### 2.3 误差下降→收益驱力映射（A57.3）

误差下降触发收益驱力的增量激活：

$$R_{\text{gain}} \mathrel{+}= \gamma \cdot |\Delta F| \quad (\Delta F < 0) \tag{A57.3}$$

其中：
- $\gamma > 0$ 为误差下降-收益驱力映射系数
- $\Delta F$ 为自由能变化量（$\Delta F < 0$ 表示误差下降）

**与A29.2的对应**：

A57.3是A29.2（引用阶段29）$R_{\text{gain}} \propto -\dot{F}$ 的离散时间具体化——自由能下降率 $-\dot{F}$ 对应离散时间的变化量 $|\Delta F|$。

**映射系数 $\gamma$ 的选择**：

| $\gamma$ | 含义 | 适用场景 |
|----------|------|---------|
| 小（如0.1） | 误差下降对收益驱力影响弱 | 保守策略，不因小改善而冒险 |
| 中（如1.0） | 误差下降对收益驱力影响适中 | 一般场景 |
| 大（如3.0） | 误差下降对收益驱力影响强 | 激进策略，积极利用改善 |

**收益驱力的物理含义**：

收益驱力激活意味着"系统在变好，应继续当前策略"——误差下降说明当前策略有效，收益驱力鼓励系统继续执行当前策略。这与A53.1的正向反馈一致——正向反馈激活收益驱力。

- **输入**：自由能变化量 $\Delta F$（$\Delta F < 0$），映射系数 $\gamma$
- **输出**：收益驱力增量 $\Delta R_{\text{gain}} \in \mathbb{R}_{\geq 0}$
- **约束**：$\gamma > 0$；仅当 $\Delta F < 0$ 时激活

> **白话翻译**：误差下降→收益驱力就是"变好了→继续赚钱"——误差下降越多，收益驱力增量越大。$\gamma$ 是"赚钱灵敏度"——$\gamma$ 大则一有改善就积极行动，$\gamma$ 小则更保守。收益驱力激活意味着"当前策略有效，继续执行"。

---

### 第四部分：误差上升→收益缺口

#### 2.4 误差上升→收益缺口映射（A57.4）

误差上升导致收益缺口增大：

$$\Delta V_{\text{gain}} \mathrel{+}= \delta \cdot \Delta F \quad (\Delta F > 0) \tag{A57.4}$$

其中：
- $\delta > 0$ 为误差上升-收益缺口映射系数
- $\Delta V_{\text{gain}}$ 为收益缺口的增量
- $\Delta F > 0$ 表示自由能上升（误差恶化）

**收益缺口的物理含义**：

收益缺口 $V_{\text{gain}}$ 度量系统"本应达到但未达到的收益水平"——误差上升意味着系统偏离了最优状态，收益缺口增大。收益缺口与安全驱力不同：

| 信号 | 物理含义 | 行动倾向 |
|------|---------|---------|
| 安全驱力 $R_{\text{safe}}$ | "出问题了" | 回避/防御 |
| 收益缺口 $\Delta V_{\text{gain}}$ | "没赚到" | 探索/改进 |

**收益缺口与安全驱力的协同**：

误差上升同时触发安全驱力（通过A57.1的异常检测）和收益缺口（通过A57.4）。两者协同作用：
- 安全驱力驱动"防御性响应"——降低风险暴露
- 收益缺口驱动"进取性响应"——寻找更好的策略

**映射系数 $\delta$ 的选择**：

$\delta$ 控制误差上升对收益缺口的影响强度。$\delta$ 越大，系统对"没赚到"越敏感，越倾向于探索新策略。

- **输入**：自由能变化量 $\Delta F$（$\Delta F > 0$），映射系数 $\delta$
- **输出**：收益缺口增量 $\Delta V_{\text{gain}} \in \mathbb{R}_{\geq 0}$
- **约束**：$\delta > 0$；仅当 $\Delta F > 0$ 时激活

> **白话翻译**：误差上升→收益缺口就是"变差了→差距拉大"——误差上升越多，收益缺口越大。和安全驱力的区别是：安全驱力说"出问题了，快躲开"，收益缺口说"没赚到，想办法赚回来"。两者一起作用——既防御又进取。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A57.1 | $R_{\text{safe}} += \alpha \cdot \text{anomaly\_score}$ | 异常分数、$\alpha$ | 安全驱力增量 $\geq 0$ | $\alpha > 0$ |
| A57.2 | $R_{\text{safe}} += \beta \cdot \text{shift\_magnitude}$ | 偏移程度、$\beta$ | 安全驱力增量 $\geq 0$ | $\beta > 0$ |
| A57.3 | $R_{\text{gain}} += \gamma \cdot |\Delta F|$（$\Delta F < 0$） | 自由能变化量、$\gamma$ | 收益驱力增量 $\geq 0$ | $\gamma > 0$，$\Delta F < 0$ |
| A57.4 | $\Delta V_{\text{gain}} += \delta \cdot \Delta F$（$\Delta F > 0$） | 自由能变化量、$\delta$ | 收益缺口增量 $\geq 0$ | $\delta > 0$，$\Delta F > 0$ |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict
from enum import Enum
import numpy as np

# 引用阶段56的数据类
from stage_056 import DistributionShiftDetector, ShiftReport


# ============================================================
# 阶段57 数据类：误差-驱力映射
# ============================================================
@dataclass(frozen=True)
class ErrorDriveMapping:
    """误差反馈到驱力的映射描述。

    组合阶段56的DistributionShiftDetector，在分布偏移检测的基础上增加
    误差反馈到驱力的映射——异常→安全驱力、偏移→安全驱力、
    误差下降→收益驱力、误差上升→收益缺口。

    Attributes:
        shift_detector: 阶段56的分布偏移检测器（组合字段）
        alpha: 异常→安全驱力映射系数
        beta: 偏移→安全驱力映射系数
        gamma: 误差下降→收益驱力映射系数
        delta: 误差上升→收益缺口映射系数
        safe_drive_increment: 安全驱力增量
        gain_drive_increment: 收益驱力增量
        gain_gap_increment: 收益缺口增量
        anomaly_contribution: 异常对安全驱力的贡献
        shift_contribution: 偏移对安全驱力的贡献
        last_delta_F: 最近一次自由能变化量

    Invariants:
        - alpha, beta, gamma, delta > 0
        - safe_drive_increment >= 0
        - gain_drive_increment >= 0
        - gain_gap_increment >= 0
        - beta > alpha（偏移比异常更重要）
    """
    shift_detector: DistributionShiftDetector
    alpha: float = 1.0
    beta: float = 2.0
    gamma: float = 1.0
    delta: float = 1.0
    safe_drive_increment: float = 0.0
    gain_drive_increment: float = 0.0
    gain_gap_increment: float = 0.0
    anomaly_contribution: float = 0.0
    shift_contribution: float = 0.0
    last_delta_F: float = 0.0

    def __post_init__(self):
        if self.alpha <= 0:
            raise ValueError(
                f"alpha 必须 > 0，收到 {self.alpha}"
            )
        if self.beta <= 0:
            raise ValueError(
                f"beta 必须 > 0，收到 {self.beta}"
            )
        if self.gamma <= 0:
            raise ValueError(
                f"gamma 必须 > 0，收到 {self.gamma}"
            )
        if self.delta <= 0:
            raise ValueError(
                f"delta 必须 > 0，收到 {self.delta}"
            )
        if self.safe_drive_increment < 0:
            raise ValueError(
                f"安全驱力增量必须 >= 0，收到 {self.safe_drive_increment}"
            )
        if self.gain_drive_increment < 0:
            raise ValueError(
                f"收益驱力增量必须 >= 0，收到 {self.gain_drive_increment}"
            )
        if self.gain_gap_increment < 0:
            raise ValueError(
                f"收益缺口增量必须 >= 0，收到 {self.gain_gap_increment}"
            )

    @property
    def total_safe_increment(self) -> float:
        """安全驱力总增量。"""
        return self.anomaly_contribution + self.shift_contribution

    @property
    def dominant_drive(self) -> str:
        """主导驱力类型。"""
        if self.safe_drive_increment > self.gain_drive_increment:
            return "safe"
        else:
            return "gain"


# ============================================================
# 阶段57 核心接口：误差-驱力映射器
# ============================================================
class ErrorDriveMapper:
    """阶段57的核心接口：误差-驱力映射器。

    将异常误差、分布偏移和自由能变化映射到驱力增量。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 映射系数 > 0
        - 驱力增量 >= 0
        - beta > alpha
    """

    def map_anomaly_to_safe(
        self,
        anomaly_score: float,
        alpha: float
    ) -> float:
        """异常→安全驱力映射（A57.1）。

        R_safe += α · anomaly_score

        Args:
            anomaly_score: 组合异常分数
            alpha: 映射系数

        Returns:
            安全驱力增量 >= 0

        Raises:
            ValueError: 当 alpha <= 0 时
        """
        if alpha <= 0:
            raise ValueError(f"alpha 必须 > 0，收到 {alpha}")
        return alpha * max(float(anomaly_score), 0.0)

    def map_shift_to_safe(
        self,
        shift_magnitude: float,
        beta: float
    ) -> float:
        """偏移→安全驱力映射（A57.2）。

        R_safe += β · shift_magnitude

        Args:
            shift_magnitude: 偏移程度
            beta: 映射系数

        Returns:
            安全驱力增量 >= 0

        Raises:
            ValueError: 当 beta <= 0 时
        """
        if beta <= 0:
            raise ValueError(f"beta 必须 > 0，收到 {beta}")
        return beta * max(float(shift_magnitude), 0.0)

    def map_error_decrease_to_gain(
        self,
        delta_F: float,
        gamma: float
    ) -> float:
        """误差下降→收益驱力映射（A57.3）。

        R_gain += γ · |ΔF|（ΔF < 0时）

        Args:
            delta_F: 自由能变化量
            gamma: 映射系数

        Returns:
            收益驱力增量 >= 0（仅当ΔF < 0时非零）
        """
        if gamma <= 0:
            raise ValueError(f"gamma 必须 > 0，收到 {gamma}")
        if delta_F < 0:
            return gamma * abs(float(delta_F))
        return 0.0

    def map_error_increase_to_gap(
        self,
        delta_F: float,
        delta: float
    ) -> float:
        """误差上升→收益缺口映射（A57.4）。

        ΔV_gain += δ · ΔF（ΔF > 0时）

        Args:
            delta_F: 自由能变化量
            delta: 映射系数

        Returns:
            收益缺口增量 >= 0（仅当ΔF > 0时非零）
        """
        if delta <= 0:
            raise ValueError(f"delta 必须 > 0，收到 {delta}")
        if delta_F > 0:
            return delta * float(delta_F)
        return 0.0

    def step(
        self,
        mapping: ErrorDriveMapping,
        anomaly_score: float,
        shift_magnitude: float,
        delta_F: float
    ) -> ErrorDriveMapping:
        """执行一步完整的误差-驱力映射。

        Args:
            mapping: 当前映射状态
            anomaly_score: 当前异常分数
            shift_magnitude: 当前偏移程度
            delta_F: 当前自由能变化量

        Returns:
            更新后的ErrorDriveMapping
        """
        from dataclasses import replace
        anomaly_contrib = self.map_anomaly_to_safe(anomaly_score, mapping.alpha)
        shift_contrib = self.map_shift_to_safe(shift_magnitude, mapping.beta)
        safe_increment = anomaly_contrib + shift_contrib
        gain_increment = self.map_error_decrease_to_gain(delta_F, mapping.gamma)
        gap_increment = self.map_error_increase_to_gap(delta_F, mapping.delta)
        return replace(
            mapping,
            safe_drive_increment=safe_increment,
            gain_drive_increment=gain_increment,
            gain_gap_increment=gap_increment,
            anomaly_contribution=anomaly_contrib,
            shift_contribution=shift_contrib,
            last_delta_F=delta_F,
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `DistributionShiftDetector` | ← 阶段56 | 作为驱力映射的分布偏移检测输入（组合字段） |
| **输入** | A21.4（安全驱力与自由能） | ← 阶段21 | 安全驱力激活的理论基础 |
| **输入** | A29.1-A29.2（驱力-自由能统一） | ← 阶段29 | 驱力映射的理论基础 |
| **输入** | A55.4（组合异常分数） | ← 阶段55 | 异常→安全驱力的输入 |
| **输入** | A56.4（偏移程度量化） | ← 阶段56 | 偏移→安全驱力的输入 |
| **输出** | `ErrorDriveMapping` | → 阶段58 | 误差-驱力映射状态，作为模型映射的输入 |
| **输出** | A57.1-A57.4（驱力映射定义） | → 阶段59-60 | 驱力映射的基础定义 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 映射计算JIT编译 | `jax.jit`编译驱力映射的纯函数 |
| NumPy | 映射系数管理 | 支撑逐层映射系数的向量化计算 |
| Pydantic | 数据验证 | 支撑ErrorDriveMapping的运行时类型检查 |
| structlog | 驱力映射审计 | 结构化日志库，支撑驱力映射的完整审计追踪 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V57.1 | 异常→安全驱力正确性 | $\Delta R_{\text{safe}} = \alpha \cdot \text{anomaly\_score}$ | 数值验证 |
| V57.2 | 偏移→安全驱力正确性 | $\Delta R_{\text{safe}} = \beta \cdot \text{shift\_magnitude}$ | 数值验证 |
| V57.3 | 误差下降→收益驱力正确性 | $\Delta R_{\text{gain}} = \gamma \cdot |\Delta F|$（$\Delta F < 0$） | 数值验证 |
| V57.4 | 误差上升→收益缺口正确性 | $\Delta V_{\text{gain}} = \delta \cdot \Delta F$（$\Delta F > 0$） | 数值验证 |
| V57.5 | 安全驱力增量非负 | $\Delta R_{\text{safe}} \geq 0$ | 随机测试 |
| V57.6 | 收益驱力增量非负 | $\Delta R_{\text{gain}} \geq 0$ | 随机测试 |
| V57.7 | 收益缺口增量非负 | $\Delta V_{\text{gain}} \geq 0$ | 随机测试 |
| V57.8 | beta > alpha | $\beta > \alpha$ | 参数验证 |
| V57.9 | 误差不变时无增量 | $\Delta F = 0$ 时收益驱力和缺口增量均为零 | 边界测试 |
| V57.10 | 异常分数为零时无安全增量 | anomaly_score = 0 时异常贡献为零 | 边界测试 |
| V57.11 | ErrorDriveMapping组合 | `mapping.shift_detector` 类型为 `DistributionShiftDetector` | 类型检查 |
| V57.12 | 非法输入拒绝 | $\alpha, \beta, \gamma, \delta \leq 0$ 抛出 `ValueError` | 异常测试 |
| V57.13 | 逐层映射正确性 | 各层分别映射，结果一致 | 逐层测试 |
| V57.14 | 驱力映射与A29一致 | 映射结果与A29.1-A29.2的理论预测一致 | 理论对比 |
| V57.15 | 端到端驱力映射 | 1000步误差变化循环无异常 | 长期运行测试 |

---

> **阶段57完** | 阶段57建立了误差反馈到驱力的映射机制：异常误差→安全驱力激活 $R_{\text{safe}} += \alpha \cdot \text{anomaly\_score}$（A57.1），分布偏移→安全驱力激活 $R_{\text{safe}} += \beta \cdot \text{shift\_magnitude}$（A57.2），误差下降→收益驱力激活 $R_{\text{gain}} += \gamma \cdot |\Delta F|$（A57.3），误差上升→收益缺口增大 $\Delta V_{\text{gain}} += \delta \cdot \Delta F$（A57.4），封装为ErrorDriveMapping数据类组合DistributionShiftDetector。下一步：阶段58将定义误差反馈到模型的映射。
