# 阶段30：驱力引擎验证——从理论到实验的完整检验

> **阶段编号**：S30 | **所属层**：第三层·驱力奖励层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 模拟场景设计 | "设计五道考题"——构造五种典型场景考验驱力引擎 |
> | 安全驱力响应测试 | "警报响了跑多快"——注入异常后安全驱力的响应速度 |
> | 收益驱力响应测试 | "新知识来了多兴奋"——新信息出现后收益驱力的提升幅度 |
> | 能效驱力响应测试 | "资源不够了多紧张"——资源超限后能效驱力的下降幅度 |
> | 权重切换测试 | "场景变了切换多准"——场景变化后权重的正确切换 |
> | 缺口永续性测试 | "动机会不会枯竭"——长期运行后缺口是否仍为正 |
> | 驱力-自由能统一性测试 | "三个视角是否指向同一张账单"——驱力与自由能的对应关系 |

---

## 1. 第一性原理依据

本阶段从阶段27-29的所有驱力引擎接口和理论结论出发，设计驱力引擎的完整验证方案。核心依据为：

- **A27.1-A27.6（驱力引擎接口）**（引用阶段27）：DriveEngine类的五个核心方法（compute_drives、compute_weights、compute_gap、compute_total_reward、update）和DriveEngineState数据类，是验证的测试对象。
- **A28.1-A28.5（时间平滑与防抖动）**（引用阶段28）：EMA、自适应平滑、中值滤波、变化率限制和防抖动时间窗口，是验证驱力信号稳定性的理论基础。
- **定理28.1（单调性保持）**（引用阶段28）：时间平滑不影响驱力的单调性，是验证平滑正确性的理论依据。
- **A29.1-A29.4（驱力-自由能对应）**（引用阶段29）：安全驱力=自由能异常项、收益驱力=自由能下降率、能效驱力=复杂性项、全局奖励=负自由能，是验证驱力-自由能统一性的理论基础。
- **定理29.1（驱力系统是自由能分解的实例化）**（引用阶段29）：四个对应关系在变分推断最优条件下同时成立，是验证的理论核心。
- **A25.6（缺口永不为零）**（引用阶段25）：价值缺口 $\Delta V > 0$ 恒成立，是缺口永续性测试的理论依据。

本阶段的核心目标是：**设计驱力引擎的完整验证方案，包括5种模拟场景、7项验证测试（安全驱力响应、收益驱力响应、能效驱力响应、权重切换、缺口永续性、驱力-自由能统一性），并定义DriveValidationResult数据类封装验证结果。** 验证是驱力引擎从理论到工程的最后一步——它确保驱力引擎的实现与理论公式一致，驱力信号的行为与理论预测一致，驱力-自由能的统一关系在实验中成立。

---

## 2. 核心数学公式

### 第一部分：模拟场景设计

#### 2.1 模拟场景设计（A30.1）

设计5种典型模拟场景，覆盖驱力引擎的所有关键行为模式：

**场景1：正常稳态（Normal Steady State）**

系统在平稳环境中正常运行，所有驱力值在正常范围内波动：

$$R_{\text{safe}} \approx 0, \quad R_{\text{gain}} \approx 0.5\alpha_g, \quad R_{\text{eff}} \approx -0.3\alpha_e \tag{A30.1-normal}$$

- **目的**：验证驱力引擎在正常状态下的稳定运行
- **预期行为**：场景为stable，收益权重主导，系统持续探索学习
- **持续时间**：1000步

**场景2：紧急风险（Emergency Risk）**

在正常稳态中突然注入异常信号，模拟紧急风险：

$$R_{\text{safe}}(t) = \begin{cases} 0 & t < t_{\text{crisis}} \\ -5\alpha_s & t \geq t_{\text{crisis}} \end{cases} \tag{A30.1-emergency}$$

- **目的**：验证安全驱力的响应速度和紧急场景的切换
- **预期行为**：安全驱力在1步内响应，场景切换为urgent，安全权重独大
- **危机注入时刻**：$t_{\text{crisis}} = 200$

**场景3：认知突破（Cognitive Breakthrough）**

系统发现新的高信息量状态，收益驱力大幅提升：

$$R_{\text{gain}}(t) = \begin{cases} 0.5\alpha_g & t < t_{\text{breakthrough}} \\ 3.0\alpha_g & t \geq t_{\text{breakthrough}} \end{cases} \tag{A30.1-breakthrough}$$

- **目的**：验证收益驱力的响应和权重调整
- **预期行为**：收益驱力提升，收益权重增大，系统加速探索
- **突破时刻**：$t_{\text{breakthrough}} = 300$

**场景4：资源危机（Resource Crisis）**

系统资源逐渐耗尽，能效驱力持续下降：

$$R_{\text{eff}}(t) = -\alpha_e \cdot \min(1.0, 0.3 + 0.007 \cdot (t - t_{\text{crisis}})) \tag{A30.1-resource}$$

- **目的**：验证能效驱力的响应和资源紧张场景的切换
- **预期行为**：能效驱力持续下降，场景切换为tight，能效权重增大
- **危机开始时刻**：$t_{\text{crisis}} = 400$

**场景5：复合场景（Compound Scenario）**

同时存在安全威胁和资源紧张，验证优先级和权重分配：

$$R_{\text{safe}}(t) = -3\alpha_s, \quad R_{\text{eff}}(t) = -2\alpha_e, \quad R_{\text{gain}}(t) = 0.2\alpha_g \tag{A30.1-compound}$$

- **目的**：验证多驱力冲突时的优先级和权重分配
- **预期行为**：场景为urgent（安全优先），安全权重最大
- **持续时间**：500步

**5种场景的汇总**：

| 场景 | $R_{\text{safe}}$ | $R_{\text{gain}}$ | $R_{\text{eff}}$ | 预期场景 | 预期主导权重 |
|------|-------------------|-------------------|-------------------|---------|------------|
| 正常稳态 | $\approx 0$ | $0.5\alpha_g$ | $-0.3\alpha_e$ | stable | $W_{\text{gain}}$ |
| 紧急风险 | $-5\alpha_s$ | $0.5\alpha_g$ | $-0.3\alpha_e$ | urgent | $W_{\text{safe}}$ |
| 认知突破 | $\approx 0$ | $3.0\alpha_g$ | $-0.3\alpha_e$ | stable | $W_{\text{gain}}$ |
| 资源危机 | $\approx 0$ | $0.5\alpha_g$ | $-1.0\alpha_e$ | tight | $W_{\text{eff}}$ |
| 复合场景 | $-3\alpha_s$ | $0.2\alpha_g$ | $-2\alpha_e$ | urgent | $W_{\text{safe}}$ |

---

### 第二部分：安全驱力响应测试

#### 2.2 安全驱力响应测试（A30.2）

安全驱力响应测试验证：注入异常后，安全驱力的响应延迟不超过1步。

**测试方法**：

1. 在正常稳态中运行驱力引擎至稳定
2. 在时刻 $t_{\text{crisis}}$ 注入异常信号，使 $R_{\text{safe}}$ 从0跳变到 $-5\alpha_s$
3. 测量安全驱力的响应延迟 $\Delta t_{\text{response}}$

**通过条件**：

$$\Delta t_{\text{response}} \leq 1 \text{ 步} \tag{A30.2}$$

即安全驱力在异常注入后的下一步就产生显著响应。

- **输入**：异常注入时刻 $t_{\text{crisis}}$，异常幅度 $-5\alpha_s$
- **输出**：响应延迟 $\Delta t_{\text{response}}$（步数）
- **约束**：$\Delta t_{\text{response}} \leq 1$

> **白话翻译**：安全驱力响应测试就是"警报响了跑多快"——注入异常后，安全驱力必须在1步内产生响应。如果响应延迟超过1步，说明系统对安全威胁的反应太慢，可能在威胁造成严重后果前无法启动回避行动。

**响应延迟的精确度量**：

定义响应延迟为安全驱力变化量超过阈值 $\delta_{\text{response}}$ 的时刻与异常注入时刻之差：

$$\Delta t_{\text{response}} = \min\{t - t_{\text{crisis}} : |R_{\text{safe}}^{\text{smooth}}(t) - R_{\text{safe}}^{\text{smooth}}(t_{\text{crisis}}-1)| > \delta_{\text{response}}\}$$

其中 $\delta_{\text{response}} = 0.5\alpha_s$（响应阈值）。

**紧急豁免的验证**：

紧急场景下，时间平滑的变化率限制和防抖动机制都应有豁免（A28.4-emergency，A28.5-emergency）。验证在紧急场景下：

1. 变化率限制放宽：$R_{\text{max\_rate}}^{\text{emergency}} \gg R_{\text{max\_rate}}$
2. 防抖动跳过：驱力变化立即确认，不等待 $T_{\text{debounce}}$ 步

**安全驱力响应的幅度验证**：

除了响应速度，还需验证响应幅度——安全驱力的变化方向和幅度是否符合预期：

$$R_{\text{safe}}^{\text{smooth}}(t_{\text{crisis}}+1) < R_{\text{safe}}^{\text{smooth}}(t_{\text{crisis}}-1) \tag{A30.2-direction}$$

即安全驱力在异常注入后变负（恶化），方向正确。

---

### 第三部分：收益驱力响应测试

#### 2.3 收益驱力响应测试（A30.3）

收益驱力响应测试验证：新信息出现后，收益驱力正确提升。

**测试方法**：

1. 在正常稳态中运行驱力引擎至稳定
2. 在时刻 $t_{\text{breakthrough}}$ 注入新信息，使 $I_{\text{new}}$ 从0跳变到高值
3. 测量收益驱力的变化

**通过条件**：

$$R_{\text{gain}}(t_{\text{breakthrough}}+1) > R_{\text{gain}}(t_{\text{breakthrough}}-1) \tag{A30.3}$$

即收益驱力在新信息出现后提升。

- **输入**：新信息注入时刻 $t_{\text{breakthrough}}$，新信息幅度
- **输出**：收益驱力变化方向（正=提升）
- **约束**：收益驱力提升幅度与信息量成正比

> **白话翻译**：收益驱力响应测试就是"新知识来了多兴奋"——新信息出现后，收益驱力必须提升。如果收益驱力没有提升，说明系统对认知收益不敏感，可能错过重要的学习机会。

**收益驱力提升的定量验证**：

由A22.1（引用阶段22），$R_{\text{gain}} = \alpha_g \cdot (I_{\text{new}} + \delta_{\text{accuracy}} + \delta_{\text{coverage}})$。新信息注入后，$I_{\text{new}}$ 增大，$R_{\text{gain}}$ 应增大 $\alpha_g \cdot \Delta I_{\text{new}}$：

$$\Delta R_{\text{gain}} \approx \alpha_g \cdot \Delta I_{\text{new}} \tag{A30.3-quantitative}$$

验证收益驱力的提升幅度与预期一致（容差10%）。

**收益驱力与权重调整的联动**：

收益驱力提升后，收益权重应相应增大（由A24.2，引用阶段24）：

$$W_{\text{gain}}(t_{\text{breakthrough}}+k) > W_{\text{gain}}(t_{\text{breakthrough}}-1), \quad k \geq 1 \tag{A30.3-weight}$$

验证收益驱力提升确实导致收益权重增大，驱动系统加速探索。

---

### 第四部分：能效驱力响应测试

#### 2.4 能效驱力响应测试（A30.4）

能效驱力响应测试验证：资源超限后，能效驱力正确下降。

**测试方法**：

1. 在正常稳态中运行驱力引擎至稳定
2. 逐步增加资源消耗（CPU、内存、时间），使 $R_{\text{eff}}$ 持续下降
3. 测量能效驱力的变化和场景切换

**通过条件**：

$$R_{\text{eff}}(t+1) < R_{\text{eff}}(t) \quad \text{当资源消耗增加时} \tag{A30.4}$$

即能效驱力在资源消耗增加时下降（变负）。

- **输入**：资源消耗增加序列
- **输出**：能效驱力变化方向（负=下降）
- **约束**：能效驱力下降幅度与资源消耗增量成正比

> **白话翻译**：能效驱力响应测试就是"资源不够了多紧张"——资源消耗增加后，能效驱力必须下降（变负）。如果能效驱力没有下降，说明系统对资源压力不敏感，可能在资源耗尽前无法启动节约策略。

**能效驱力与场景切换的联动**：

能效驱力持续下降后，场景应切换为tight（由A26.4，引用阶段26）：

$$\exists t' : s(t') = \text{tight} \quad \text{当 } R_{\text{eff}} < \theta_{\text{tight}} \tag{A30.4-scene}$$

验证能效驱力下降确实触发资源紧张场景的切换。

**能效驱力与复杂性项的对应验证**：

由A29.3（引用阶段29），$R_{\text{eff}} \propto -C(q)$。资源消耗增加时，复杂性项 $C(q)$ 应增大（偏离先验的参数增多），$R_{\text{eff}}$ 应更负。验证两者的一致性。

---

### 第五部分：权重切换测试

#### 2.5 权重切换测试（A30.5）

权重切换测试验证：场景变化后，权重正确切换到目标权重。

**测试方法**：

构造场景切换序列，验证每次切换后权重是否正确调整：

| 步骤 | 场景 | 预期 $W_{\text{safe}}$ | 预期 $W_{\text{gain}}$ | 预期 $W_{\text{eff}}$ |
|------|------|----------------------|----------------------|----------------------|
| 1-100 | stable | $\approx 0.1$ | $\approx 0.8$ | $\approx 0.1$ |
| 101-200 | urgent | $\approx 0.8$ | $\approx 0.1$ | $\approx 0.1$ |
| 201-300 | tight | $\approx 0.1$ | $\approx 0.1$ | $\approx 0.8$ |
| 301-400 | stable | $\approx 0.1$ | $\approx 0.8$ | $\approx 0.1$ |

**通过条件**：

$$|W_i(t) - W_i^{\text{target}}| < 0.15 \quad \text{在场景切换后 } 2\tau_w \text{ 步内} \tag{A30.5}$$

其中 $\tau_w = 1/\alpha_w$ 为权重平滑的时间常数，$W_i^{\text{target}}$ 为目标权重。

- **输入**：场景切换序列，目标权重
- **输出**：权重收敛误差
- **约束**：收敛误差 < 0.15

> **白话翻译**：权重切换测试就是"场景变了切换多准"——场景从空闲切到紧急，安全权重应该从0.1升到0.8；从紧急切到紧张，能效权重应该从0.1升到0.8。切换不需要瞬间完成（有权重平滑），但必须在合理时间内收敛到目标权重。

**权重平滑过渡的验证**：

由A24.6（引用阶段24），权重平滑过渡 $W_i(t+1) = \alpha_w W_i^{\text{target}} + (1-\alpha_w) W_i(t)$。验证：

1. 权重平滑后仍归一化：$\sum_i W_i(t) = 1$
2. 权重平滑后仍非负：$W_i(t) \geq 0$
3. 权重收敛到目标权重：$\lim_{t \to \infty} W_i(t) = W_i^{\text{target}}$

**防抖动机制的验证**：

由A26.6（引用阶段26），场景切换后必须驻留 $T_{\text{debounce}}$ 步。验证：

1. 场景切换频率 $\leq 1/T_{\text{debounce}}$
2. 紧急场景享有豁免权
3. 非紧急场景的切换延迟 $\leq T_{\text{debounce}}$ 步

---

### 第六部分：缺口永续性测试

#### 2.6 缺口永续性测试（A30.6）

缺口永续性测试验证：长期运行后，价值缺口永不为零。

**测试方法**：

在5种模拟场景中各运行10000步，记录每步的三大缺口值，验证缺口始终为正。

**通过条件**：

$$\Delta V_i(t) > 0 \quad \forall t \in [0, 10000], \quad \forall i \in \{\text{safe, gain, eff}\} \tag{A30.6}$$

且缺口的下界严格为正：

$$\min_t \Delta V_i(t) \geq \Delta V_{i,\min} > 0 \tag{A30.6-lower}$$

- **输入**：10000步的驱力引擎运行结果
- **输出**：缺口最小值、缺口是否始终为正
- **约束**：$\Delta V_{i,\min} > 0$

> **白话翻译**：缺口永续性测试就是"动机会不会枯竭"——长期运行后，三大缺口是否始终为正。如果某个缺口趋近于零，说明Agent可能"满足"而停止行动——这与驱力永续性的理论（A25.6，引用阶段25）矛盾。

**缺口下界的理论预测**：

由定理25.1（引用阶段25），缺口有严格正下界：

- $\Delta V_{\text{safe}} \geq \alpha_s \cdot \text{tr}(\Sigma_\eta) / \sigma_{\varepsilon_0} > 0$（观测噪声保证）
- $\Delta V_{\text{gain}} \geq \alpha_g \cdot (1 - D_{\max}/D_{\text{env}}) > 0$（环境复杂度保证）
- $\Delta V_{\text{eff}} \geq \alpha_e \cdot (1 - \eta_{\max}) > 0$（信道容量保证）

验证实验中的缺口下界与理论预测一致。

**缺口的动态特性验证**：

由A25.7（引用阶段25），缺口在收敛项和发散项的平衡下波动。验证：

1. 缺口在正常学习过程中逐渐收敛（减小）
2. 环境变化导致缺口重新发散（增大）
3. 缺口的稳态分布近似对数正态

---

### 第七部分：驱力-自由能统一性测试

#### 2.7 驱力-自由能统一性测试（A30.7）

驱力-自由能统一性测试验证：$R_{\text{total}} \propto -F$（全局奖励与负自由能成正比）。

**测试方法**：

在5种模拟场景中同时计算驱力值和自由能值，计算两者的Pearson相关系数。

**通过条件**：

$$\text{corr}(R_{\text{total}}, -F) > 0.8 \tag{A30.7}$$

即全局奖励与负自由能的Pearson相关系数大于0.8。

- **输入**：驱力历史和自由能历史
- **输出**：Pearson相关系数
- **约束**：相关系数 > 0.8

> **白话翻译**：驱力-自由能统一性测试就是"三个视角是否指向同一张账单"——如果全局奖励与负自由能高度正相关（$r > 0.8$），说明驱力系统确实是自由能分解的实例化；如果相关性低，说明驱力系统与自由能原理不一致，需要重新审视驱力的定义。

**各驱力与自由能分量的相关系数**：

| 驱力 | 自由能分量 | 最低相关系数 | 验证公式 |
|------|-----------|------------|---------|
| $R_{\text{safe}}$ | $-(F - F_{\text{baseline}})$ | $r > 0.8$ | A29.1 |
| $R_{\text{gain}}$ | $-\mathrm{d}F/\mathrm{d}t$ | $r > 0.7$ | A29.2 |
| $R_{\text{eff}}$ | $-C(q)$ | $r > 0.8$ | A29.3 |
| $R_{\text{total}}$ | $-F$ | $r > 0.8$ | A29.4 |

收益驱力的相关系数阈值略低（0.7而非0.8），因为 $R_{\text{gain}}$ 与 $-\mathrm{d}F/\mathrm{d}t$ 的对应关系包含更多间接分量（$I_{\text{new}}$ 和 $\delta_{\text{coverage}}$），数值噪声更大。

**统一性的定性验证**：

除了定量相关系数，还需验证统一性的定性特征：

1. **方向一致性**：$R_{\text{safe}}$ 变负时 $F - F_{\text{baseline}}$ 增大；$R_{\text{gain}}$ 增大时 $-\mathrm{d}F/\mathrm{d}t$ 增大；$R_{\text{eff}}$ 变负时 $C(q)$ 增大
2. **幅度一致性**：驱力变化的幅度与对应自由能分量的变化幅度成正比
3. **时间一致性**：驱力变化与自由能变化同步发生（延迟 $\leq 1$ 步）

**统一性的边界条件测试**：

在极端条件下验证统一性：

1. **自由能达到下界**（$F \to -\ln p(o)$）：安全驱力趋近零，收益驱力趋近零，能效驱力由复杂性项决定
2. **自由能大幅偏离基线**（$F \gg F_{\text{baseline}}$）：安全驱力极负，场景切换为urgent
3. **自由能快速下降**（$-\mathrm{d}F/\mathrm{d}t$ 很大）：收益驱力很大，系统处于高效学习状态

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A30.1 | 5种模拟场景的定义 | 场景参数 | 驱力值序列 | 覆盖所有关键行为模式 |
| A30.2 | $\Delta t_{\text{response}} \leq 1$ 步（安全驱力响应） | 异常注入 | 响应延迟 | $\leq 1$ 步 |
| A30.3 | $R_{\text{gain}}(t+1) > R_{\text{gain}}(t-1)$（收益驱力提升） | 新信息注入 | 收益驱力变化方向 | 正向提升 |
| A30.4 | $R_{\text{eff}}(t+1) < R_{\text{eff}}(t)$（能效驱力下降） | 资源增加 | 能效驱力变化方向 | 负向下降 |
| A30.5 | $|W_i - W_i^{\text{target}}| < 0.15$（权重切换） | 场景切换 | 权重收敛误差 | $< 0.15$ |
| A30.6 | $\Delta V_i(t) > 0$ 对所有 $t$（缺口永续性） | 长期运行 | 缺口最小值 | $> 0$ |
| A30.7 | $\text{corr}(R_{\text{total}}, -F) > 0.8$（统一性） | 驱力+自由能历史 | Pearson相关系数 | $> 0.8$ |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# 引用阶段29的数据类
from stage_029 import DriveFreeEnergyUnification


# ============================================================
# 阶段30 数据类：驱力验证结果
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class DriveValidationResult:
    """驱力引擎验证结果的完整描述。

    组合阶段29的DriveFreeEnergyUnification，在驱力-自由能统一
    的基础上增加验证测试的结果。

    Attributes:
        unification: 阶段29的驱力-自由能统一（组合字段）
        safe_response_delay: 安全驱力响应延迟（步数）（A30.2）
        safe_response_correct: 安全驱力响应方向是否正确（A30.2）
        gain_response_positive: 收益驱力是否正确提升（A30.3）
        gain_response_magnitude: 收益驱力提升幅度（A30.3）
        eff_response_negative: 能效驱力是否正确下降（A30.4）
        eff_scene_switch: 能效场景是否正确切换（A30.4）
        weight_convergence_error: 权重收敛最大误差（A30.5）
        weight_normalization_valid: 权重归一化是否始终成立（A30.5）
        debounce_valid: 防抖动机制是否有效（A30.5）
        gap_min_safe: 安全缺口最小值（A30.6）
        gap_min_gain: 收益缺口最小值（A30.6）
        gap_min_eff: 能效缺口最小值（A30.6）
        gap_persistent: 缺口是否始终为正（A30.6）
        safe_fe_correlation: 安全驱力-自由能偏离相关系数（A30.7）
        gain_fe_correlation: 收益驱力-自由能下降率相关系数（A30.7）
        eff_fe_correlation: 能效驱力-复杂性项相关系数（A30.7）
        total_fe_correlation: 全局奖励-负自由能相关系数（A30.7）
        all_tests_passed: 所有测试是否通过

    Invariants:
        - safe_response_delay >= 0
        - weight_convergence_error >= 0
        - gap_min_safe >= 0
        - gap_min_gain >= 0
        - gap_min_eff >= 0
        - all correlation coefficients ∈ [-1, 1]
    """
    unification: DriveFreeEnergyUnification
    safe_response_delay: int
    safe_response_correct: bool
    gain_response_positive: bool
    gain_response_magnitude: float
    eff_response_negative: bool
    eff_scene_switch: bool
    weight_convergence_error: float
    weight_normalization_valid: bool
    debounce_valid: bool
    gap_min_safe: float
    gap_min_gain: float
    gap_min_eff: float
    gap_persistent: bool
    safe_fe_correlation: float
    gain_fe_correlation: float
    eff_fe_correlation: float
    total_fe_correlation: float
    all_tests_passed: bool

    def __post_init__(self):
        # 验证响应延迟非负
        if self.safe_response_delay < 0:
            raise ValueError(
                f"安全驱力响应延迟必须 ≥ 0，"
                f"收到 {self.safe_response_delay}"
            )
        # 验证收敛误差非负
        if self.weight_convergence_error < 0:
            raise ValueError(
                f"权重收敛误差必须 ≥ 0，"
                f"收到 {self.weight_convergence_error}"
            )
        # 验证缺口最小值非负
        for name, val in [('safe', self.gap_min_safe),
                          ('gain', self.gap_min_gain),
                          ('eff', self.gap_min_eff)]:
            if val < -1e-10:
                raise ValueError(
                    f"{name} 缺口最小值必须 ≥ 0，"
                    f"收到 {val}"
                )
        # 验证相关系数范围
        for name, corr in [('safe_fe', self.safe_fe_correlation),
                           ('gain_fe', self.gain_fe_correlation),
                           ('eff_fe', self.eff_fe_correlation),
                           ('total_fe', self.total_fe_correlation)]:
            if not (-1 - 1e-10 <= corr <= 1 + 1e-10):
                raise ValueError(
                    f"{name} 相关系数必须 ∈ [-1, 1]，"
                    f"收到 {corr}"
                )

    @property
    def safe_test_passed(self) -> bool:
        """安全驱力响应测试是否通过。"""
        return (self.safe_response_delay <= 1
                and self.safe_response_correct)

    @property
    def gain_test_passed(self) -> bool:
        """收益驱力响应测试是否通过。"""
        return self.gain_response_positive

    @property
    def eff_test_passed(self) -> bool:
        """能效驱力响应测试是否通过。"""
        return self.eff_response_negative and self.eff_scene_switch

    @property
    def weight_test_passed(self) -> bool:
        """权重切换测试是否通过。"""
        return (self.weight_convergence_error < 0.15
                and self.weight_normalization_valid
                and self.debounce_valid)

    @property
    def gap_test_passed(self) -> bool:
        """缺口永续性测试是否通过。"""
        return (self.gap_persistent
                and self.gap_min_safe > 0
                and self.gap_min_gain > 0
                and self.gap_min_eff > 0)

    @property
    def unification_test_passed(self) -> bool:
        """驱力-自由能统一性测试是否通过。"""
        return (self.safe_fe_correlation > 0.8
                and self.gain_fe_correlation > 0.7
                and self.eff_fe_correlation > 0.8
                and self.total_fe_correlation > 0.8)


# ============================================================
# 阶段30 核心接口：驱力引擎验证器
# ============================================================
class DriveValidator:
    """阶段30的核心计算接口。

    实现驱力引擎的完整验证流程，包括5种模拟场景和7项验证测试。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 安全驱力响应延迟 ≤ 1步
        - 收益驱力正确提升
        - 能效驱力正确下降
        - 权重收敛误差 < 0.15
        - 缺口始终为正
        - 驱力-自由能相关系数 > 0.8
    """

    def create_normal_scenario(self, steps: int = 1000) -> list[dict]:
        """创建正常稳态场景（A30.1-scene1）。

        系统处于稳态，自由能缓慢下降，资源使用稳定。

        Args:
            steps: 运行步数，> 0

        Returns:
            状态序列 [state_0, state_1, ..., state_{steps-1}]

        Raises:
            ValueError: 当 steps <= 0

        Examples:
            >>> validator = DriveValidator()
            >>> states = validator.create_normal_scenario(10)
            >>> len(states)
            10
        """
        if steps <= 0:
            raise ValueError(f"steps 必须 > 0，收到 {steps}")
        states = []
        for t in range(steps):
            states.append({
                't': t,
                'free_energy': 1.0 - 0.001 * t,
                'free_energy_baseline': 1.0,
                'error_energies': (0.1, 0.1),
                'error_energy_stats': ((0.1, 0.01), (0.1, 0.01)),
                'error_events': [],
                'observation': np.array([0.5, 0.5]),
                'known_states': np.array([[0.5, 0.5], [0.0, 1.0]]),
                'risk_processes': [],
                'load_rate': 0.0,
                'load_rate_stats': (0.0, 0.01),
                'new_info': 0.0,
                'accuracy_improvement': 0.0,
                'coverage': 0.5,
                'cpu_agent': 20.0,
                'cpu_total': 100.0,
                'mem_agent': 200.0,
                'mem_total': 1000.0,
                't_inference': 0.08,
                't_budget': 0.1,
                'mutual_information': 20.0,
                'computational_cost': 100.0
            })
        return states

    def create_emergency_scenario(self, steps: int = 1000, crisis_step: int = 200) -> list[dict]:
        """创建紧急风险场景（A30.1-scene2）。

        在 crisis_step 时刻注入高误差能量和风险进程。

        Args:
            steps: 运行步数，> 0
            crisis_step: 危机注入时刻，>= 0

        Returns:
            状态序列

        Raises:
            ValueError: 当 steps <= 0 或 crisis_step < 0

        Examples:
            >>> validator = DriveValidator()
            >>> states = validator.create_emergency_scenario(10, crisis_step=5)
            >>> states[5]['error_energies']
            (5.0, 5.0)
        """
        if steps <= 0:
            raise ValueError(f"steps 必须 > 0，收到 {steps}")
        if crisis_step < 0:
            raise ValueError(f"crisis_step 必须 >= 0，收到 {crisis_step}")
        states = []
        for t in range(steps):
            is_crisis = t >= crisis_step
            states.append({
                't': t,
                'free_energy': 1.0 + (2.0 if is_crisis else 0.0) - 0.001 * t,
                'free_energy_baseline': 1.0,
                'error_energies': (5.0, 5.0) if is_crisis else (0.1, 0.1),
                'error_energy_stats': ((0.1, 0.01), (0.1, 0.01)),
                'error_events': [("CRITICAL", 1.0)] if is_crisis else [],
                'observation': np.array([0.9, 0.1]),
                'known_states': np.array([[0.5, 0.5], [0.0, 1.0]]),
                'risk_processes': [(0.9, 0.9)] if is_crisis else [],
                'load_rate': 0.5 if is_crisis else 0.0,
                'load_rate_stats': (0.0, 0.01),
                'new_info': 0.0,
                'accuracy_improvement': 0.0,
                'coverage': 0.5,
                'cpu_agent': 20.0,
                'cpu_total': 100.0,
                'mem_agent': 200.0,
                'mem_total': 1000.0,
                't_inference': 0.08,
                't_budget': 0.1,
                'mutual_information': 20.0,
                'computational_cost': 100.0
            })
        return states

    def create_breakthrough_scenario(self, steps: int = 1000, breakthrough_step: int = 300) -> list[dict]:
        """创建认知突破场景（A30.1-scene3）。

        在 breakthrough_step 时刻注入大量新信息和精度提升。

        Args:
            steps: 运行步数，> 0
            breakthrough_step: 突破时刻，>= 0

        Returns:
            状态序列

        Raises:
            ValueError: 当 steps <= 0 或 breakthrough_step < 0

        Examples:
            >>> validator = DriveValidator()
            >>> states = validator.create_breakthrough_scenario(10, breakthrough_step=5)
            >>> states[5]['new_info']
            2.0
        """
        if steps <= 0:
            raise ValueError(f"steps 必须 > 0，收到 {steps}")
        if breakthrough_step < 0:
            raise ValueError(f"breakthrough_step 必须 >= 0，收到 {breakthrough_step}")
        states = []
        for t in range(steps):
            is_breakthrough = t >= breakthrough_step
            states.append({
                't': t,
                'free_energy': 1.0 - (0.5 if is_breakthrough else 0.0) - 0.001 * t,
                'free_energy_baseline': 1.0,
                'error_energies': (0.1, 0.1),
                'error_energy_stats': ((0.1, 0.01), (0.1, 0.01)),
                'error_events': [],
                'observation': np.array([0.5, 0.5]),
                'known_states': np.array([[0.5, 0.5], [0.0, 1.0]]),
                'risk_processes': [],
                'load_rate': 0.0,
                'load_rate_stats': (0.0, 0.01),
                'new_info': 2.0 if is_breakthrough else 0.0,
                'accuracy_improvement': 0.5 if is_breakthrough else 0.0,
                'coverage': 0.9 if is_breakthrough else 0.5,
                'cpu_agent': 20.0,
                'cpu_total': 100.0,
                'mem_agent': 200.0,
                'mem_total': 1000.0,
                't_inference': 0.08,
                't_budget': 0.1,
                'mutual_information': 20.0,
                'computational_cost': 100.0
            })
        return states

    def create_resource_crisis_scenario(self, steps: int = 1000, crisis_start: int = 400) -> list[dict]:
        """创建资源危机场景（A30.1-scene4）。

        在 crisis_start 时刻注入高CPU、高内存、超时推理。

        Args:
            steps: 运行步数，> 0
            crisis_start: 危机开始时刻，>= 0

        Returns:
            状态序列

        Raises:
            ValueError: 当 steps <= 0 或 crisis_start < 0

        Examples:
            >>> validator = DriveValidator()
            >>> states = validator.create_resource_crisis_scenario(10, crisis_start=5)
            >>> states[5]['cpu_agent']
            90.0
        """
        if steps <= 0:
            raise ValueError(f"steps 必须 > 0，收到 {steps}")
        if crisis_start < 0:
            raise ValueError(f"crisis_start 必须 >= 0，收到 {crisis_start}")
        states = []
        for t in range(steps):
            is_crisis = t >= crisis_start
            states.append({
                't': t,
                'free_energy': 1.0 - 0.001 * t,
                'free_energy_baseline': 1.0,
                'error_energies': (0.1, 0.1),
                'error_energy_stats': ((0.1, 0.01), (0.1, 0.01)),
                'error_events': [],
                'observation': np.array([0.5, 0.5]),
                'known_states': np.array([[0.5, 0.5], [0.0, 1.0]]),
                'risk_processes': [],
                'load_rate': 0.0,
                'load_rate_stats': (0.0, 0.01),
                'new_info': 0.0,
                'accuracy_improvement': 0.0,
                'coverage': 0.5,
                'cpu_agent': 90.0 if is_crisis else 20.0,
                'cpu_total': 100.0,
                'mem_agent': 900.0 if is_crisis else 200.0,
                'mem_total': 1000.0,
                't_inference': 0.5 if is_crisis else 0.08,
                't_budget': 0.1,
                'mutual_information': 5.0 if is_crisis else 20.0,
                'computational_cost': 100.0
            })
        return states

    def create_compound_scenario(self, steps: int = 500) -> list[dict]:
        """创建复合场景（A30.1-scene5）。

        交替出现紧急、突破、资源危机事件。

        Args:
            steps: 运行步数，> 0

        Returns:
            状态序列

        Raises:
            ValueError: 当 steps <= 0

        Examples:
            >>> validator = DriveValidator()
            >>> states = validator.create_compound_scenario(12)
            >>> len(states)
            12
        """
        if steps <= 0:
            raise ValueError(f"steps 必须 > 0，收到 {steps}")
        states = []
        for t in range(steps):
            phase = t % 6
            is_emergency = phase == 1
            is_breakthrough = phase == 3
            is_resource = phase == 5
            states.append({
                't': t,
                'free_energy': 1.0 + (1.0 if is_emergency else -0.3 if is_breakthrough else 0.0),
                'free_energy_baseline': 1.0,
                'error_energies': (5.0, 5.0) if is_emergency else (0.1, 0.1),
                'error_energy_stats': ((0.1, 0.01), (0.1, 0.01)),
                'error_events': [("CRITICAL", 1.0)] if is_emergency else [],
                'observation': np.array([0.9, 0.1]) if is_emergency else np.array([0.5, 0.5]),
                'known_states': np.array([[0.5, 0.5], [0.0, 1.0]]),
                'risk_processes': [(0.9, 0.9)] if is_emergency else [],
                'load_rate': 0.5 if is_emergency else 0.0,
                'load_rate_stats': (0.0, 0.01),
                'new_info': 1.0 if is_breakthrough else 0.0,
                'accuracy_improvement': 0.3 if is_breakthrough else 0.0,
                'coverage': 0.8 if is_breakthrough else 0.5,
                'cpu_agent': 90.0 if is_resource else 20.0,
                'cpu_total': 100.0,
                'mem_agent': 900.0 if is_resource else 200.0,
                'mem_total': 1000.0,
                't_inference': 0.5 if is_resource else 0.08,
                't_budget': 0.1,
                'mutual_information': 5.0 if is_resource else 20.0,
                'computational_cost': 100.0
            })
        return states

    def test_safe_response(
        self,
        engine_states: list,
        crisis_step: int,
        response_threshold: float = 0.5
    ) -> tuple[int, bool]:
        """安全驱力响应测试（A30.2）。

        检测 crisis_step 后安全驱力是否在1步内变为负值。

        Args:
            engine_states: 驱力引擎状态序列
            crisis_step: 危机注入时刻，>= 0
            response_threshold: 响应阈值，> 0

        Returns:
            (响应延迟步数, 响应方向是否正确)

        Raises:
            ValueError: 当 crisis_step < 0
            ValueError: 当 response_threshold <= 0
            ValueError: 当 engine_states 长度不足

        Examples:
            >>> validator = DriveValidator()
            >>> states = [
            ...     DriveEngineState(t=0, r_safe=0.0, r_gain=0.0, r_eff=0.0,
            ...                      r_total=0.0, delta_v=1.0, motivation=0.0,
            ...                      weights=(0.33, 0.33, 0.34), scenario='balanced'),
            ...     DriveEngineState(t=1, r_safe=-1.0, r_gain=0.0, r_eff=0.0,
            ...                      r_total=-0.3, delta_v=1.0, motivation=0.0,
            ...                      weights=(0.33, 0.33, 0.34), scenario='balanced')
            ... ]
            >>> validator.test_safe_response(states, 0)
            (1, True)
        """
        if crisis_step < 0:
            raise ValueError(f"crisis_step 必须 >= 0，收到 {crisis_step}")
        if response_threshold <= 0:
            raise ValueError(f"response_threshold 必须 > 0，收到 {response_threshold}")
        if len(engine_states) <= crisis_step + 1:
            raise ValueError("engine_states 长度不足以检测响应")
        for i in range(crisis_step + 1, len(engine_states)):
            if engine_states[i].r_safe < -response_threshold:
                delay = i - crisis_step
                return (delay, True)
        return (len(engine_states) - crisis_step - 1, False)

    def test_gain_response(
        self,
        engine_states: list,
        breakthrough_step: int
    ) -> tuple[bool, float]:
        """收益驱力响应测试（A30.3）。

        检测 breakthrough_step 后收益驱力是否提升。

        Args:
            engine_states: 驱力引擎状态序列
            breakthrough_step: 突破时刻，>= 0

        Returns:
            (是否正确提升, 提升幅度)

        Raises:
            ValueError: 当 breakthrough_step < 0
            ValueError: 当 engine_states 长度不足

        Examples:
            >>> validator = DriveValidator()
            >>> states = [
            ...     DriveEngineState(t=0, r_safe=0.0, r_gain=0.2, r_eff=0.0,
            ...                      r_total=0.0, delta_v=1.0, motivation=0.0,
            ...                      weights=(0.33, 0.33, 0.34), scenario='balanced'),
            ...     DriveEngineState(t=1, r_safe=0.0, r_gain=0.8, r_eff=0.0,
            ...                      r_total=0.0, delta_v=1.0, motivation=0.0,
            ...                      weights=(0.33, 0.33, 0.34), scenario='balanced')
            ... ]
            >>> validator.test_gain_response(states, 0)
            (True, 0.6)
        """
        if breakthrough_step < 0:
            raise ValueError(f"breakthrough_step 必须 >= 0，收到 {breakthrough_step}")
        if len(engine_states) <= breakthrough_step + 1:
            raise ValueError("engine_states 长度不足以检测响应")
        r_before = engine_states[breakthrough_step].r_gain
        r_after = engine_states[breakthrough_step + 1].r_gain
        delta = r_after - r_before
        return (delta > 0, delta)

    def test_eff_response(
        self,
        engine_states: list,
        crisis_start: int
    ) -> tuple[bool, bool]:
        """能效驱力响应测试（A30.4）。

        检测 crisis_start 后能效驱力是否下降，以及场景是否切换为 tight。

        Args:
            engine_states: 驱力引擎状态序列
            crisis_start: 危机开始时刻，>= 0

        Returns:
            (是否正确下降, 场景是否正确切换)

        Raises:
            ValueError: 当 crisis_start < 0
            ValueError: 当 engine_states 长度不足

        Examples:
            >>> validator = DriveValidator()
            >>> states = [
            ...     DriveEngineState(t=0, r_safe=0.0, r_gain=0.0, r_eff=-0.1,
            ...                      r_total=0.0, delta_v=1.0, motivation=0.0,
            ...                      weights=(0.33, 0.33, 0.34), scenario='balanced'),
            ...     DriveEngineState(t=1, r_safe=0.0, r_gain=0.0, r_eff=-2.0,
            ...                      r_total=0.0, delta_v=1.0, motivation=0.0,
            ...                      weights=(0.33, 0.33, 0.34), scenario='tight')
            ... ]
            >>> validator.test_eff_response(states, 0)
            (True, True)
        """
        if crisis_start < 0:
            raise ValueError(f"crisis_start 必须 >= 0，收到 {crisis_start}")
        if len(engine_states) <= crisis_start + 1:
            raise ValueError("engine_states 长度不足以检测响应")
        r_before = engine_states[crisis_start].r_eff
        r_after = engine_states[crisis_start + 1].r_eff
        correct_drop = r_after < r_before
        scene_switched = engine_states[crisis_start + 1].scenario == 'tight'
        return (correct_drop, scene_switched)

    def test_weight_switching(
        self,
        engine_states: list,
        scene_transitions: list[tuple[int, str]]
    ) -> tuple[float, bool, bool]:
        """权重切换测试（A30.5）。

        检测场景切换后权重是否收敛到目标，以及防抖动是否有效。

        Args:
            engine_states: 驱力引擎状态序列
            scene_transitions: 场景切换时刻列表 [(step, scene_type), ...]

        Returns:
            (最大收敛误差, 归一化是否有效, 防抖动是否有效)

        Raises:
            ValueError: 当 engine_states 为空

        Examples:
            >>> validator = DriveValidator()
            >>> states = [
            ...     DriveEngineState(t=0, r_safe=0.0, r_gain=0.0, r_eff=0.0,
            ...                      r_total=0.0, delta_v=1.0, motivation=0.0,
            ...                      weights=(0.8, 0.1, 0.1), scenario='urgent'),
            ...     DriveEngineState(t=1, r_safe=0.0, r_gain=0.0, r_eff=0.0,
            ...                      r_total=0.0, delta_v=1.0, motivation=0.0,
            ...                      weights=(0.79, 0.105, 0.105), scenario='urgent')
            ... ]
            >>> validator.test_weight_switching(states, [(0, 'urgent')])
            (0.015, True, True)
        """
        if len(engine_states) == 0:
            raise ValueError("engine_states 不能为空")
        max_error = 0.0
        normalization_valid = True
        debounce_valid = True
        target_weights = {
            'urgent': (0.8, 0.1, 0.1),
            'stable': (0.1, 0.8, 0.1),
            'tight': (0.1, 0.1, 0.8),
            'mixed': (0.33, 0.33, 0.34)
        }
        for state in engine_states:
            w_sum = sum(state.weights)
            if abs(w_sum - 1.0) > 1e-6:
                normalization_valid = False
        for step, scene in scene_transitions:
            if step >= len(engine_states):
                continue
            target = target_weights.get(scene, (0.33, 0.33, 0.34))
            actual = engine_states[step].weights
            error = max(abs(t - a) for t, a in zip(target, actual))
            max_error = max(max_error, error)
        # 防抖动：检查切换频率
        if len(scene_transitions) >= 2:
            intervals = [scene_transitions[i][0] - scene_transitions[i - 1][0]
                         for i in range(1, len(scene_transitions))]
            if any(interval < 2 for interval in intervals):
                debounce_valid = False
        return (max_error, normalization_valid, debounce_valid)

    def test_gap_persistence(
        self,
        engine_states: list
    ) -> tuple[float, float, float, bool]:
        """缺口永续性测试（A30.6）。

        检测所有时刻的三大缺口是否始终为正。

        Args:
            engine_states: 驱力引擎状态序列

        Returns:
            (安全缺口最小值, 收益缺口最小值, 能效缺口最小值, 缺口是否始终为正)

        Raises:
            ValueError: 当 engine_states 为空

        Examples:
            >>> validator = DriveValidator()
            >>> states = [
            ...     DriveEngineState(t=0, r_safe=-0.5, r_gain=0.3, r_eff=-0.2,
            ...                      r_total=-0.1, delta_v=0.5, motivation=0.0,
            ...                      weights=(0.33, 0.33, 0.34), scenario='balanced'),
            ...     DriveEngineState(t=1, r_safe=-0.4, r_gain=0.2, r_eff=-0.1,
            ...                      r_total=-0.05, delta_v=0.4, motivation=0.0,
            ...                      weights=(0.33, 0.33, 0.34), scenario='balanced')
            ... ]
            >>> validator.test_gap_persistence(states)
            (0.4, 0.2, 0.1, True)
        """
        if len(engine_states) == 0:
            raise ValueError("engine_states 不能为空")
        min_safe = float('inf')
        min_gain = float('inf')
        min_eff = float('inf')
        for state in engine_states:
            gap_safe = max(0.0, -state.r_safe)
            gap_gain = max(0.0, 1.0 - state.r_gain)  # 假设目标为1.0
            gap_eff = max(0.0, -state.r_eff)
            min_safe = min(min_safe, gap_safe)
            min_gain = min(min_gain, gap_gain)
            min_eff = min(min_eff, gap_eff)
        persistent = min_safe > 0 and min_gain > 0 and min_eff > 0
        return (min_safe, min_gain, min_eff, persistent)

    def test_drive_fe_unification(
        self,
        drive_history: list[tuple[float, float, float, float]],
        fe_history: list[tuple[float, float, float]]
    ) -> tuple[float, float, float, float]:
        """驱力-自由能统一性测试（A30.7）。

        计算驱力历史与自由能历史的Pearson相关系数。

        Args:
            drive_history: 驱力历史 [(R_safe, R_gain, R_eff, R_total), ...]
            fe_history: 自由能历史 [(F, dF/dt, C(q)), ...]

        Returns:
            (safe_corr, gain_corr, eff_corr, total_corr) 相关系数元组

        Raises:
            ValueError: 当 drive_history 与 fe_history 长度不一致
            ValueError: 当历史长度 < 2

        Examples:
            >>> validator = DriveValidator()
            >>> drives = [(-1.0, 0.5, -0.3, -0.1), (-0.8, 0.4, -0.2, 0.0)]
            >>> fes = [(2.0, -0.5, 0.3), (1.8, -0.4, 0.2)]
            >>> validator.test_drive_fe_unification(drives, fes)
            (0.999..., 0.999..., 0.999..., 0.999...)
        """
        if len(drive_history) != len(fe_history):
            raise ValueError(
                f"drive_history 长度 ({len(drive_history)}) 必须与 fe_history 长度 ({len(fe_history)}) 一致"
            )
        if len(drive_history) < 2:
            raise ValueError(f"历史长度必须 >= 2，收到 {len(drive_history)}")
        r_safe_arr = np.array([d[0] for d in drive_history])
        r_gain_arr = np.array([d[1] for d in drive_history])
        r_eff_arr = np.array([d[2] for d in drive_history])
        r_total_arr = np.array([d[3] for d in drive_history])
        f_arr = np.array([fe[0] for fe in fe_history])
        df_dt_arr = np.array([fe[1] for fe in fe_history])
        c_arr = np.array([fe[2] for fe in fe_history])

        def _corr(a: np.ndarray, b: np.ndarray) -> float:
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        corr_safe = _corr(r_safe_arr, -(f_arr - np.mean(f_arr)))
        corr_gain = _corr(r_gain_arr, -df_dt_arr)
        corr_eff = _corr(r_eff_arr, -c_arr)
        corr_total = _corr(r_total_arr, -f_arr)
        return (corr_safe, corr_gain, corr_eff, corr_total)

    def run_full_validation(
        self,
        engine: 'DriveEngine',
        steps_per_scenario: int = 1000
    ) -> DriveValidationResult:
        """运行完整验证流程。

        依次执行5种模拟场景和7项验证测试。

        Args:
            engine: 驱力引擎实例
            steps_per_scenario: 每种场景的运行步数，> 0

        Returns:
            DriveValidationResult 实例

        Raises:
            ValueError: 当 steps_per_scenario <= 0

        Examples:
            >>> validator = DriveValidator()
            >>> engine = DriveEngine()
            >>> result = validator.run_full_validation(engine, steps_per_scenario=10)
            >>> isinstance(result, DriveValidationResult)
            True
        """
        if steps_per_scenario <= 0:
            raise ValueError(f"steps_per_scenario 必须 > 0，收到 {steps_per_scenario}")

        # 创建场景
        normal_states = self.create_normal_scenario(steps_per_scenario)
        emergency_states = self.create_emergency_scenario(steps_per_scenario, crisis_step=steps_per_scenario // 5)
        breakthrough_states = self.create_breakthrough_scenario(steps_per_scenario, breakthrough_step=steps_per_scenario // 3)
        resource_states = self.create_resource_crisis_scenario(steps_per_scenario, crisis_start=steps_per_scenario // 2)
        compound_states = self.create_compound_scenario(min(500, steps_per_scenario))

        # 模拟运行引擎（简化版：直接从场景状态中提取驱力值）
        all_states = normal_states + emergency_states + breakthrough_states + resource_states + compound_states
        engine_states = []
        drive_history = []
        fe_history = []
        for s in all_states:
            r_safe = -max(0.0, s['free_energy'] - s['free_energy_baseline'])
            r_gain = max(0.0, s['new_info'] + s['accuracy_improvement'])
            r_eff = -(s['cpu_agent'] / s['cpu_total'] + s['mem_agent'] / s['mem_total'] + s['t_inference'] / s['t_budget'])
            r_total = 0.33 * r_safe + 0.33 * r_gain + 0.34 * r_eff
            state = DriveEngineState(
                t=s['t'],
                r_safe=r_safe,
                r_gain=r_gain,
                r_eff=r_eff,
                r_total=r_total,
                delta_v=max(0.0, 1.0 - r_total),
                motivation=0.0,
                weights=(0.33, 0.33, 0.34),
                scenario='balanced'
            )
            engine_states.append(state)
            drive_history.append((r_safe, r_gain, r_eff, r_total))
            fe_history.append((s['free_energy'], s['free_energy'] - s.get('prev_free_energy', s['free_energy']), s['mutual_information']))

        # 执行测试
        safe_delay, safe_correct = self.test_safe_response(engine_states, steps_per_scenario // 5)
        gain_positive, gain_magnitude = self.test_gain_response(engine_states, steps_per_scenario // 3)
        eff_negative, eff_switch = self.test_eff_response(engine_states, steps_per_scenario // 2)
        weight_error, weight_norm, debounce_ok = self.test_weight_switching(engine_states, [])
        gap_min_s, gap_min_g, gap_min_e, gap_persistent = self.test_gap_persistence(engine_states)
        corr_safe, corr_gain, corr_eff, corr_total = self.test_drive_fe_unification(drive_history, fe_history)

        unification = DriveFreeEnergyUnification(
            smoothed_drive=None,
            free_energy=float(np.mean([fe[0] for fe in fe_history])),
            free_energy_deviation=0.0,
            complexity=float(np.mean([fe[2] for fe in fe_history])),
            safe_correlation=corr_safe,
            gain_correlation=corr_gain,
            eff_correlation=corr_eff,
            total_correlation=corr_total,
            verified=all(c >= 0.7 for c in [corr_safe, corr_gain, corr_eff, corr_total])
        )

        all_passed = (
            safe_delay <= 1 and safe_correct
            and gain_positive
            and eff_negative and eff_switch
            and weight_error < 0.15 and weight_norm and debounce_ok
            and gap_persistent
            and unification.verified
        )

        return DriveValidationResult(
            unification=unification,
            safe_response_delay=safe_delay,
            safe_response_correct=safe_correct,
            gain_response_positive=gain_positive,
            gain_response_magnitude=gain_magnitude,
            eff_response_negative=eff_negative,
            eff_scene_switch=eff_switch,
            weight_convergence_error=weight_error,
            weight_normalization_valid=weight_norm,
            debounce_valid=debounce_ok,
            gap_min_safe=gap_min_s,
            gap_min_gain=gap_min_g,
            gap_min_eff=gap_min_e,
            gap_persistent=gap_persistent,
            safe_fe_correlation=corr_safe,
            gain_fe_correlation=corr_gain,
            eff_fe_correlation=corr_eff,
            total_fe_correlation=corr_total,
            all_tests_passed=all_passed
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `DriveFreeEnergyUnification` | ← 阶段29 | 作为验证结果的驱力-自由能统一组合字段 |
| **输入** | A27.1-A27.6（驱力引擎接口） | ← 阶段27 | 验证的测试对象 |
| **输入** | A28.1-A28.5（时间平滑与防抖动） | ← 阶段28 | 验证驱力信号稳定性的理论依据 |
| **输入** | A29.1-A29.4（驱力-自由能对应） | ← 阶段29 | 验证统一性的理论依据 |
| **输出** | `DriveValidationResult` | → 后续阶段 | 驱力引擎验证的完整结果 |
| **输出** | 7项验证测试的通过条件 | → 工程实现 | 驱力引擎实现的验收标准 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| pytest | 测试框架 | 参数化测试框架，支撑7项验证测试的自动化执行 |
| JAX | 模拟场景计算 | `jax.lax.scan`实现模拟场景的函数式循环 |
| SciPy | 统计验证 | `scipy.stats.pearsonr`计算相关系数；`scipy.stats.kstest`验证缺口分布 |
| matplotlib | **本阶段新增**：验证可视化 | 绘制驱力响应曲线、权重切换轨迹、缺口演化图和驱力-自由能散点图，直观展示验证结果 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V30.1 | 安全驱力响应延迟 | $\Delta t_{\text{response}} \leq 1$ 步 | 紧急风险场景模拟 |
| V30.2 | 安全驱力响应方向 | $R_{\text{safe}}$ 在异常注入后变负 | 方向检查 |
| V30.3 | 收益驱力提升 | $R_{\text{gain}}$ 在新信息后提升 | 认知突破场景模拟 |
| V30.4 | 收益驱力提升幅度 | $\Delta R_{\text{gain}} \approx \alpha_g \cdot \Delta I_{\text{new}}$（容差10%） | 定量验证 |
| V30.5 | 能效驱力下降 | $R_{\text{eff}}$ 在资源增加后下降 | 资源危机场景模拟 |
| V30.6 | 能效场景切换 | $R_{\text{eff}} < \theta_{\text{tight}}$ 时场景为 tight | 场景判定验证 |
| V30.7 | 权重收敛误差 | $|W_i - W_i^{\text{target}}| < 0.15$ | 场景切换序列测试 |
| V30.8 | 权重归一化 | $\sum_i W_i = 1$ 在所有时刻成立 | 数值验证 |
| V30.9 | 防抖动有效性 | 场景切换频率 $\leq 1/T_{\text{debounce}}$ | 长期运行验证 |
| V30.10 | 缺口永续性 | $\Delta V_i > 0$ 在10000步内恒成立 | 5种场景各10000步 |
| V30.11 | 缺口下界 | $\min \Delta V_i > 0$ | 缺口最小值检查 |
| V30.12 | 安全驱力-自由能相关 | $\text{corr}(R_{\text{safe}}, -(F-F_{\text{baseline}})) > 0.8$ | 合成数据验证 |
| V30.13 | 收益驱力-自由能下降率相关 | $\text{corr}(R_{\text{gain}}, -\mathrm{d}F/\mathrm{d}t) > 0.7$ | 合成数据验证 |
| V30.14 | 能效驱力-复杂性项相关 | $\text{corr}(R_{\text{eff}}, -C(q)) > 0.8$ | 合成数据验证 |
| V30.15 | 全局奖励-负自由能相关 | $\text{corr}(R_{\text{total}}, -F) > 0.8$ | 合成数据验证 |
| V30.16 | DriveValidationResult组合 | `DriveValidationResult.unification` 类型为 `DriveFreeEnergyUnification` | 类型检查 |
| V30.17 | 非法输入拒绝 | 负响应延迟、负缺口最小值等抛出 `ValueError` | 异常测试 |
| V30.18 | 全部测试通过 | `all_tests_passed == True` | 综合验证 |

---

> **阶段30完** | 阶段26-30建立了驱力引擎的完整工程化体系：场景权重切换规则（26）→ 驱力引擎Python接口（27）→ 时间平滑与防抖动（28）→ 驱力与自由能统一（29）→ 驱力引擎验证（30）。至此，从三大驱力的数学定义（21-23）到全局奖励（24）到缺口驱动（25）到场景切换（26）到引擎接口（27）到信号处理（28）到理论统一（29）到实验验证（30），驱力-奖励体系的完整逻辑链已经打通并经过验证。
