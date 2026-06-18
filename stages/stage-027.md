# 阶段27：驱力引擎Python接口——三大驱力的统一计算引擎

> **阶段编号**：S27 | **所属层**：第三层·驱力奖励层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 驱力引擎 | "大脑的总指挥"——统一管理安全、收益、能效三大驱力的计算 |
> | compute_drives | "测三项指标"——计算安全、收益、能效三大驱力值 |
> | compute_weights | "分配注意力"——根据场景分配三大驱力的权重 |
> | compute_gap | "量差距"——计算当前状态与目标的差距 |
> | compute_total_reward | "算总分"——加权求和得到全局奖励 |
> | update | "更新状态"——一步完成所有计算并更新引擎状态 |

---

## 1. 第一性原理依据

本阶段从阶段21-26的所有驱力相关公式出发，设计驱力引擎的统一Python接口。核心依据为：

- **A21.1-A21.5（安全驱力）**（引用阶段21）：安全驱力的定义、异常信号、权重自适应、自由能关系和时间平滑，是驱力引擎安全分量的计算基础。
- **A22.1-A22.4（收益驱力）**（引用阶段22）：收益驱力的定义、新信息维度、精度提升和认知覆盖度，是驱力引擎收益分量的计算基础。
- **A23.1-A23.6（能效驱力）**（引用阶段23）：能效驱力的定义、CPU/内存/时间/冗余度四项指标、与复杂性项的关系，是驱力引擎能效分量的计算基础。
- **A24.1-A24.6（全局奖励与动态权重）**（引用阶段24）：全局奖励公式、动态权重自适应、三种典型场景、权重平滑过渡，是驱力引擎权重和总奖励的计算基础。
- **A25.1-A25.6（缺口驱动）**（引用阶段25）：价值缺口的定义、三大缺口分解、缺口→动机映射、缺口永不为零，是驱力引擎缺口计算的基础。
- **A26.1-A26.6（场景权重切换）**（引用阶段26）：场景分类器、三种场景判定、混合场景插值、防抖动机制，是驱力引擎场景识别和权重切换的基础。

本阶段的核心目标是：**设计DriveEngine类，整合三大驱力计算、动态权重分配、缺口计算和全局奖励计算为统一的接口，提供compute_drives、compute_weights、compute_gap、compute_total_reward和update五个核心方法，并定义DriveEngineState数据类封装引擎的完整状态。** 驱力引擎是阶段21-26所有数学公式的工程化实现——它将分散的公式整合为可调用、可测试、可组合的Python接口。

---

## 2. 核心数学公式

### 第一部分：DriveEngine类设计

#### 2.1 DriveEngine类设计（A27.1）

DriveEngine类是驱力引擎的核心入口，整合三大驱力+权重+缺口的完整计算流程：

$$\text{DriveEngine} = (\text{SafetyCalculator}, \text{GainCalculator}, \text{EfficiencyCalculator}, \text{SceneSwitcher}, \text{GapCalculator}) \tag{A27.1}$$

其中：
- $\text{SafetyCalculator}$：安全驱力计算器（引用阶段21，A21.1-A21.5）
- $\text{GainCalculator}$：收益驱力计算器（引用阶段22，A22.1-A22.4）
- $\text{EfficiencyCalculator}$：能效驱力计算器（引用阶段23，A23.1-A23.6）
- $\text{SceneSwitcher}$：场景权重切换器（引用阶段26，A26.1-A26.6）
- $\text{GapCalculator}$：缺口计算器（引用阶段25，A25.1-A25.6）

- **输入**：系统状态 state（包含所有驱力计算所需的原始数据）
- **输出**：DriveEngineState（包含三大驱力值、权重、缺口、总奖励、场景等完整信息）
- **约束**：所有计算均为纯函数（无副作用）；状态更新通过创建新的DriveEngineState实例实现

> **白话翻译**：驱力引擎就是"大脑的总指挥中心"——它把安全、收益、能效三个计算器，加上场景识别和缺口计算，整合成一个统一的接口。调用者只需要传入系统状态，引擎就能一次性算出所有驱力相关的值。

**DriveEngine的设计原则**：

1. **单一入口**：所有驱力计算通过DriveEngine统一入口，避免调用者直接操作底层计算器
2. **不可变状态**：每次update返回新的DriveEngineState，不修改旧状态，保证线程安全和可追溯性
3. **纯函数计算**：compute_drives、compute_weights、compute_gap、compute_total_reward均为纯函数，输出仅依赖输入
4. **组合优于继承**：DriveEngine通过组合（而非继承）各计算器实现功能，符合SOLID原则
5. **可验证性**：每个方法都有明确的输入输出类型和约束，支持自动化验证

**DriveEngine的计算流程**：

```
state → compute_drives(state) → (R_safe, R_gain, R_eff)
                                      ↓
                    compute_weights(drives, scene) → (W_safe, W_gain, W_eff)
                                      ↓
                    compute_gap(drives, targets) → (ΔV_safe, ΔV_gain, ΔV_eff)
                                      ↓
                    compute_total_reward(drives, weights) → R_total
                                      ↓
                    update(drives, scene, targets) → DriveEngineState
```

**DriveEngine的参数配置**：

| 参数 | 来源 | 默认值 | 说明 |
|------|------|--------|------|
| $\alpha_s$ | A21.1 | 1.0 | 安全驱力缩放系数 |
| $\alpha_g$ | A22.1 | 1.0 | 收益驱力缩放系数 |
| $\alpha_e$ | A23.1 | 1.0 | 能效驱力缩放系数 |
| $\beta$ | A24.2 | 1.0 | 权重锐度参数 |
| $\alpha_w$ | A24.6 | 0.1 | 权重更新速率 |
| $\gamma_s$ | A21.5 | 0.9 | 安全驱力平滑系数 |
| $\theta_{\text{urgent}}$ | A26.2 | $-3\alpha_s$ | 紧急风险阈值 |
| $\theta_{\text{stable}}$ | A26.3 | $-0.5\alpha_s$ | 稳定阈值 |
| $\theta_{\text{idle}}$ | A26.3 | $0.3\alpha_g$ | 空闲阈值 |
| $\theta_{\text{tight}}$ | A26.4 | $-2.0\alpha_e$ | 资源紧张阈值 |
| $T_{\text{debounce}}$ | A26.6 | 20 | 最小驻留时间 |

---

### 第二部分：compute_drives接口

#### 2.2 compute_drives(state) → (R_safe, R_gain, R_eff)（A27.2）

compute_drives从系统状态计算三大驱力值：

$$(R_{\text{safe}}, R_{\text{gain}}, R_{\text{eff}}) = \text{compute\_drives}(\text{state}) \tag{A27.2}$$

其中：
- $R_{\text{safe}} = -\alpha_s \cdot \sum_i w_i \cdot \text{anomaly}_i$（引用阶段21，A21.1）
- $R_{\text{gain}} = \alpha_g \cdot (I_{\text{new}} + \delta_{\text{accuracy}} + \delta_{\text{coverage}})$（引用阶段22，A22.1）
- $R_{\text{eff}} = -\alpha_e \cdot (\text{cpu\_ratio} + \text{mem\_ratio} + \text{time\_cost} + \text{redundancy})$（引用阶段23，A23.1）

- **输入**：系统状态 state，包含异常信号、认知指标、资源指标等原始数据
- **输出**：三大驱力值元组 $(R_{\text{safe}}, R_{\text{gain}}, R_{\text{eff}})$
- **约束**：$R_{\text{safe}} \leq 0$；$R_{\text{eff}} \leq 0$；$R_{\text{gain}}$ 通常 $\geq 0$

> **白话翻译**：compute_drives就是"测三项指标"——从系统状态中提取异常信号、认知指标和资源指标，分别计算安全驱力、收益驱力和能效驱力。这是驱力引擎最基础的计算步骤。

**state的数据结构**：

系统状态 state 包含以下字段：

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| error_energies | tuple[float, ...] | 阶段3，A3.2 | 各层精度加权误差能量 |
| error_events | list[tuple[str, float]] | 系统日志 | 错误事件序列 |
| observation | np.ndarray | 感知层 | 当前观测 |
| known_states | np.ndarray | 记忆库 | 已知状态库 |
| risk_processes | list[tuple[float, float]] | 进程监控 | 风险进程列表 |
| load_rate | float | 系统监控 | 负载变化率 |
| new_info | float | 阶段7，A7.5 | 新信息维度 |
| accuracy_improvement | float | 阶段5，A5.1 | 精度提升量 |
| coverage | float | 阶段22，A22.4 | 认知覆盖度 |
| cpu_ratio | float | 系统监控 | CPU占用率 |
| mem_ratio | float | 系统监控 | 内存占用率 |
| time_cost | float | 推理计时 | 推理耗时比 |
| redundancy | float | 阶段23，A23.5 | 计算冗余度 |

**compute_drives的计算步骤**：

1. **安全驱力**：调用SafetyCalculator.compute_anomalies提取五类异常信号（A21.2），调用compute_weights计算自适应权重（A21.3），调用compute_safe_reward计算安全驱力（A21.1），调用compute_safe_reward_from_free_energy验证一致性（A21.4），调用smooth_safe_reward进行时间平滑（A21.5）
2. **收益驱力**：从state提取新信息维度、精度提升和认知覆盖度，按A22.1计算收益驱力
3. **能效驱力**：从state提取CPU/内存/时间/冗余度四项指标，按A23.1计算能效驱力

**compute_drives的不变量**：

- $R_{\text{safe}} \leq 0$：安全驱力恒非正（A21.1的约束）
- $R_{\text{eff}} \leq 0$：能效驱力恒非正（A23.1的约束）
- 安全驱力的A21.1形式与A21.4形式在正常范围内数值一致（A21.4的等价性论证）

---

### 第三部分：compute_weights接口

#### 2.3 compute_weights(drives, scene) → (W_safe, W_gain, W_eff)（A27.3）

compute_weights根据驱力值和场景计算动态权重：

$$(W_{\text{safe}}, W_{\text{gain}}, W_{\text{eff}}) = \text{compute\_weights}(\text{drives}, \text{scene}) \tag{A27.3}$$

其中权重的计算流程为：

1. **场景识别**：调用SceneSwitcher.classify_scene判定当前场景（A26.1）
2. **目标权重**：根据场景获取目标权重向量（A26.1-defaults）
3. **混合插值**：若为混合场景，调用interpolate_weights计算插值权重（A26.5）
4. **权重平滑**：调用smooth_weights进行权重平滑过渡（A24.6）
5. **防抖动**：调用apply_debounce确保场景切换的稳定性（A26.6）

- **输入**：驱力值 $(R_{\text{safe}}, R_{\text{gain}}, R_{\text{eff}})$，场景信息（含上一时刻场景和切换时间）
- **输出**：动态权重元组 $(W_{\text{safe}}, W_{\text{gain}}, W_{\text{eff}})$
- **约束**：$W_i \geq 0$；$\sum_i W_i = 1$

> **白话翻译**：compute_weights就是"分配注意力"——根据当前场景和驱力值，决定安全、收益、能效各占多少权重。紧急时安全权重高，空闲时收益权重高，缺资源时能效权重高，混合时按比例分配。

**compute_weights的计算步骤**：

1. 调用SceneWeightSwitcherCalculator.classify_scene，传入三大驱力值和阈值参数，获取场景标识 $s$
2. 调用apply_debounce，判断场景是否可以切换（A26.6）
3. 根据有效场景获取目标权重 $\mathbf{W}^s$（A26.1-defaults）
4. 若为混合场景，调用compute_memberships和interpolate_weights（A26.5）
5. 调用smooth_weights，从当前权重平滑过渡到目标权重（A24.6）
6. 返回平滑后的权重

**compute_weights的不变量**：

- 权重归一化：$\sum_i W_i = 1$（由softmax和插值的归一化保持性保证）
- 权重非负：$W_i \geq 0$（由softmax和插值的非负性保持性保证）
- 场景切换频率：$\leq 1/T_{\text{debounce}}$（由防抖动机制保证）

---

### 第四部分：compute_gap接口

#### 2.4 compute_gap(drives, targets) → (ΔV_safe, ΔV_gain, ΔV_eff)（A27.4）

compute_gap根据驱力值和目标值计算三大缺口：

$$(\Delta V_{\text{safe}}, \Delta V_{\text{gain}}, \Delta V_{\text{eff}}) = \text{compute\_gap}(\text{drives}, \text{targets}) \tag{A27.4}$$

其中：
- $\Delta V_{\text{safe}} = -R_{\text{safe}} = \alpha_s \cdot \sum_i w_i \cdot \text{anomaly}_i$（引用阶段25，A25.2）
- $\Delta V_{\text{gain}} = V_{\text{gain,target}} - R_{\text{gain}}$（引用阶段25，A25.3）
- $\Delta V_{\text{eff}} = -R_{\text{eff}} = \alpha_e \cdot (\text{cpu} + \text{mem} + \text{time} + \text{redundancy})$（引用阶段25，A25.4）

- **输入**：驱力值 $(R_{\text{safe}}, R_{\text{gain}}, R_{\text{eff}})$，目标值 $(V_{\text{safe,target}}, V_{\text{gain,target}}, V_{\text{eff,target}})$
- **输出**：三大缺口元组 $(\Delta V_{\text{safe}}, \Delta V_{\text{gain}}, \Delta V_{\text{eff}})$
- **约束**：$\Delta V_i \geq 0$；$\Delta V_i > 0$（缺口永不为零，A25.6）

> **白话翻译**：compute_gap就是"量差距"——计算当前驱力值与终极目标之间的差距。安全缺口 = 当前异常的总量，收益缺口 = 离全知还差多远，能效缺口 = 当前资源浪费的总量。三个缺口都永远为正——这是Agent永不停止行动的根本原因。

**目标值的默认设置**：

| 目标 | 默认值 | 物理含义 |
|------|--------|---------|
| $V_{\text{safe,target}}$ | 0 | 零异常——绝对安全 |
| $V_{\text{gain,target}}$ | $\alpha_g \cdot (H[O] + \Delta F_{\max} + 1)$ | 全知——完全理解环境 |
| $V_{\text{eff,target}}$ | 0 | 零资源消耗——完美效率 |

**compute_gap的不变量**：

- 缺口非负：$\Delta V_i \geq 0$（由驱力的符号约束保证）
- 缺口永不为零：$\Delta V_i > 0$（由A25.6保证——热力学第二定律的推论）
- 缺口-动机单调性：$\text{motivation}_i = f(\Delta V_i)$ 单调递增（A25.5）

---

### 第五部分：compute_total_reward接口

#### 2.5 compute_total_reward(drives, weights) → R_total（A27.5）

compute_total_reward根据驱力值和权重计算全局总奖励：

$$R_{\text{total}} = \text{compute\_total\_reward}(\text{drives}, \text{weights}) = W_{\text{safe}} \cdot R_{\text{safe}} + W_{\text{gain}} \cdot R_{\text{gain}} + W_{\text{eff}} \cdot R_{\text{eff}} \tag{A27.5}$$

- **输入**：驱力值 $(R_{\text{safe}}, R_{\text{gain}}, R_{\text{eff}})$，权重 $(W_{\text{safe}}, W_{\text{gain}}, W_{\text{eff}})$
- **输出**：$R_{\text{total}} \in \mathbb{R}$
- **约束**：$W_i \geq 0$；$\sum_i W_i = 1$

> **白话翻译**：compute_total_reward就是"算总分"——安全成绩×安全权重 + 收益成绩×收益权重 + 能效成绩×能效权重 = 总分。总分为正表示整体向好，为负表示整体向坏。

**compute_total_reward的符号分析**：

由A24.1（引用阶段24），全局奖励的符号取决于权重分配和驱力大小：

| 条件 | $R_{\text{total}}$ 符号 | 含义 |
|------|----------------------|------|
| $W_{\text{gain}} R_{\text{gain}} > |W_{\text{safe}} R_{\text{safe}} + W_{\text{eff}} R_{\text{eff}}|$ | 正 | 收益超过代价 |
| $W_{\text{gain}} R_{\text{gain}} < |W_{\text{safe}} R_{\text{safe}} + W_{\text{eff}} R_{\text{eff}}|$ | 负 | 代价超过收益 |
| 两者相等 | 零 | 收益与代价平衡 |

**compute_total_reward与EFE的等价性**：

由定理24.1（引用阶段24），$R_{\text{total}} \propto -G(\pi)$，即最大化全局奖励等价于最小化预期自由能。compute_total_reward的计算结果可直接与自由能计算结果对比验证。

---

### 第六部分：update接口

#### 2.6 update(drives, scene, targets) → DriveEngineState（A27.6）

update是驱力引擎的主入口，一步完成所有计算并更新引擎状态：

$$\text{DriveEngineState}_{t+1} = \text{update}(\text{state}_t) \tag{A27.6}$$

update的完整计算流程：

1. compute_drives(state) → $(R_{\text{safe}}, R_{\text{gain}}, R_{\text{eff}})$
2. compute_weights(drives, scene) → $(W_{\text{safe}}, W_{\text{gain}}, W_{\text{eff}})$
3. compute_gap(drives, targets) → $(\Delta V_{\text{safe}}, \Delta V_{\text{gain}}, \Delta V_{\text{eff}})$
4. compute_total_reward(drives, weights) → $R_{\text{total}}$
5. 封装为DriveEngineState

- **输入**：系统状态 state（包含所有原始数据 + 上一时刻的引擎状态）
- **输出**：新的DriveEngineState实例
- **约束**：不修改旧状态；所有不变量在新状态中保持

> **白话翻译**：update就是"一键更新"——传入系统状态，引擎自动完成驱力计算、权重分配、缺口计算和总奖励计算，返回完整的引擎状态。调用者不需要关心内部计算步骤，只需调用update即可获得所有驱力相关的信息。

**DriveEngineState的数据结构**：

DriveEngineState封装驱力引擎的完整状态，组合SceneWeightSwitcher（引用阶段26）：

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| scene_switcher | SceneWeightSwitcher | 阶段26 | 场景权重切换器（组合字段） |
| r_safe | float | A27.2 | 安全驱力值 |
| r_gain | float | A27.2 | 收益驱力值 |
| r_eff | float | A27.2 | 能效驱力值 |
| weights | tuple[float, float, float] | A27.3 | 动态权重 |
| delta_v_safe | float | A27.4 | 安全缺口 |
| delta_v_gain | float | A27.4 | 收益缺口 |
| delta_v_eff | float | A27.4 | 能效缺口 |
| total_reward | float | A27.5 | 全局总奖励 |
| timestamp | int | 外部 | 时间戳 |

**DriveEngineState的不变量**：

1. $R_{\text{safe}} \leq 0$（安全驱力非正）
2. $R_{\text{eff}} \leq 0$（能效驱力非正）
3. $W_i \geq 0$，$\sum_i W_i = 1$（权重归一化非负）
4. $\Delta V_i \geq 0$（缺口非负）
5. $R_{\text{total}} = \sum_i W_i \cdot R_i$（总奖励一致性）

**update的幂等性**：

update不是幂等的——每次调用都会根据新的系统状态产生新的DriveEngineState。但update是确定性的——相同的输入总是产生相同的输出。这保证了驱力引擎的可复现性。

**update的性能考虑**：

update的计算复杂度主要取决于：
1. compute_drives：$O(L)$（$L$ 为预测编码层数，来自A21.2-sys的误差能量计算）
2. compute_weights：$O(1)$（场景判定和softmax计算为常数时间）
3. compute_gap：$O(1)$（缺口计算为简单算术运算）
4. compute_total_reward：$O(1)$（加权和计算为常数时间）

总体复杂度为 $O(L)$，主要由安全驱力中的分层误差能量计算决定。对于典型的预测编码网络（$L \leq 10$），update的计算时间在微秒级别，满足实时性要求。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A27.1 | DriveEngine = (SafetyCalc, GainCalc, EffCalc, SceneSwitcher, GapCalc) | 配置参数 | DriveEngine实例 | 组合模式 |
| A27.2 | $(R_{\text{safe}}, R_{\text{gain}}, R_{\text{eff}})$ = compute_drives(state) | 系统状态 | 三大驱力值 | $R_{\text{safe}} \leq 0$；$R_{\text{eff}} \leq 0$ |
| A27.3 | $(W_{\text{safe}}, W_{\text{gain}}, W_{\text{eff}})$ = compute_weights(drives, scene) | 驱力值, 场景 | 动态权重 | $W_i \geq 0$；$\sum W_i = 1$ |
| A27.4 | $(\Delta V_{\text{safe}}, \Delta V_{\text{gain}}, \Delta V_{\text{eff}})$ = compute_gap(drives, targets) | 驱力值, 目标值 | 三大缺口 | $\Delta V_i \geq 0$；$\Delta V_i > 0$ |
| A27.5 | $R_{\text{total}}$ = compute_total_reward(drives, weights) | 驱力值, 权重 | $R_{\text{total}} \in \mathbb{R}$ | $\sum W_i = 1$ |
| A27.6 | DriveEngineState$_{t+1}$ = update(state$_t$) | 系统状态 | DriveEngineState | 所有不变量保持 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# 引用阶段21-26的数据类和计算器
from stage_021 import SafetyDriveCalculator
from stage_022 import GainDriveCalculator
from stage_023 import EfficiencyDriveCalculator
from stage_026 import SceneWeightSwitcher, SceneType, SceneWeightSwitcherCalculator


# ============================================================
# 阶段27 数据类：驱力引擎状态
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class DriveEngineState:
    """驱力引擎的完整状态描述。

    组合阶段26的SceneWeightSwitcher，在场景权重切换的基础上增加
    三大驱力值、缺口和全局奖励的完整计算结果。

    Attributes:
        scene_switcher: 阶段26的场景权重切换器（组合字段）
        r_safe: 安全驱力值（A27.2）
        r_gain: 收益驱力值（A27.2）
        r_eff: 能效驱力值（A27.2）
        weights: 动态权重元组 (W_safe, W_gain, W_eff)（A27.3）
        delta_v_safe: 安全缺口（A27.4）
        delta_v_gain: 收益缺口（A27.4）
        delta_v_eff: 能效缺口（A27.4）
        total_reward: 全局总奖励（A27.5）
        timestamp: 时间戳

    Invariants:
        - r_safe <= 0
        - r_eff <= 0
        - all(w >= 0 for w in weights) and sum(weights) ≈ 1
        - delta_v_safe >= 0
        - delta_v_gain >= 0
        - delta_v_eff >= 0
        - total_reward == weights[0]*r_safe + weights[1]*r_gain + weights[2]*r_eff
    """
    scene_switcher: SceneWeightSwitcher
    r_safe: float
    r_gain: float
    r_eff: float
    weights: tuple[float, float, float]
    delta_v_safe: float
    delta_v_gain: float
    delta_v_eff: float
    total_reward: float
    timestamp: int

    def __post_init__(self):
        # 验证安全驱力非正
        if self.r_safe > 1e-10:
            raise ValueError(
                f"安全驱力必须 ≤ 0，"
                f"收到 {self.r_safe}"
            )
        # 验证能效驱力非正
        if self.r_eff > 1e-10:
            raise ValueError(
                f"能效驱力必须 ≤ 0，"
                f"收到 {self.r_eff}"
            )
        # 验证权重归一化
        for i, w in enumerate(self.weights):
            if w < -1e-10:
                raise ValueError(
                    f"第 {i} 个权重必须 ≥ 0，"
                    f"收到 {w}"
                )
        w_sum = sum(self.weights)
        if not (0.99 <= w_sum <= 1.01):
            raise ValueError(
                f"权重之和必须 ≈ 1，"
                f"收到 {w_sum}"
            )
        # 验证缺口非负
        if self.delta_v_safe < -1e-10:
            raise ValueError(
                f"安全缺口必须 ≥ 0，"
                f"收到 {self.delta_v_safe}"
            )
        if self.delta_v_gain < -1e-10:
            raise ValueError(
                f"收益缺口必须 ≥ 0，"
                f"收到 {self.delta_v_gain}"
            )
        if self.delta_v_eff < -1e-10:
            raise ValueError(
                f"能效缺口必须 ≥ 0，"
                f"收到 {self.delta_v_eff}"
            )
        # 验证总奖励一致性
        expected = (self.weights[0] * self.r_safe
                    + self.weights[1] * self.r_gain
                    + self.weights[2] * self.r_eff)
        if not np.isclose(self.total_reward, expected, atol=1e-8):
            raise ValueError(
                f"总奖励不一致: 期望 {expected}，"
                f"收到 {self.total_reward}"
            )

    @property
    def scene(self) -> SceneType:
        """当前场景。"""
        return self.scene_switcher.scene

    @property
    def total_gap(self) -> float:
        """总价值缺口。"""
        return self.delta_v_safe + self.delta_v_gain + self.delta_v_eff

    @property
    def dominant_drive(self) -> str:
        """主导驱力（权重最大的驱力）。"""
        drive_names = ['safe', 'gain', 'eff']
        return drive_names[int(np.argmax(self.weights))]

    @property
    def is_healthy(self) -> bool:
        """系统是否健康（安全驱力接近零且总奖励为正）。"""
        return self.r_safe > -0.5 and self.total_reward > 0


# ============================================================
# 阶段27 核心接口：驱力引擎
# ============================================================
class DriveEngine:
    """阶段27的核心计算接口。

    整合三大驱力计算、动态权重分配、缺口计算和全局奖励计算
    为统一的接口。所有计算均为纯函数（无副作用），保证可验证性。

    核心约束：
        - R_safe ≤ 0（安全驱力恒非正）
        - R_eff ≤ 0（能效驱力恒非正）
        - 权重归一化 ∑W_i = 1
        - 缺口非负 ΔV_i ≥ 0
        - 总奖励一致性 R_total = ∑W_i · R_i
    """

    def __init__(
        self,
        alpha_s: float = 1.0,
        alpha_g: float = 1.0,
        alpha_e: float = 1.0,
        beta: float = 1.0,
        alpha_w: float = 0.1,
        gamma_s: float = 0.9,
        theta_urgent: Optional[float] = None,
        theta_stable: Optional[float] = None,
        theta_idle: Optional[float] = None,
        theta_tight: Optional[float] = None,
        t_debounce: int = 20
    ):
        """初始化驱力引擎。

        Args:
            alpha_s: 安全驱力缩放系数
            alpha_g: 收益驱力缩放系数
            alpha_e: 能效驱力缩放系数
            beta: 权重锐度参数
            alpha_w: 权重更新速率
            gamma_s: 安全驱力平滑系数
            theta_urgent: 紧急风险阈值（默认 -3*alpha_s）
            theta_stable: 稳定阈值（默认 -0.5*alpha_s）
            theta_idle: 空闲阈值（默认 0.3*alpha_g）
            theta_tight: 资源紧张阈值（默认 -2.0*alpha_e）
            t_debounce: 最小驻留时间
        """
        if alpha_s <= 0:
            raise ValueError(f"alpha_s 必须 > 0，收到 {alpha_s}")
        if alpha_g <= 0:
            raise ValueError(f"alpha_g 必须 > 0，收到 {alpha_g}")
        if alpha_e <= 0:
            raise ValueError(f"alpha_e 必须 > 0，收到 {alpha_e}")
        if beta <= 0:
            raise ValueError(f"beta 必须 > 0，收到 {beta}")
        if alpha_w <= 0:
            raise ValueError(f"alpha_w 必须 > 0，收到 {alpha_w}")
        if not (0 <= gamma_s < 1):
            raise ValueError(f"gamma_s 必须 ∈ [0, 1)，收到 {gamma_s}")
        if t_debounce <= 0:
            raise ValueError(f"t_debounce 必须 > 0，收到 {t_debounce}")

        self.alpha_s = alpha_s
        self.alpha_g = alpha_g
        self.alpha_e = alpha_e
        self.beta = beta
        self.alpha_w = alpha_w
        self.gamma_s = gamma_s
        self.theta_urgent = theta_urgent if theta_urgent is not None else -3.0 * alpha_s
        self.theta_stable = theta_stable if theta_stable is not None else -0.5 * alpha_s
        self.theta_idle = theta_idle if theta_idle is not None else 0.3 * alpha_g
        self.theta_tight = theta_tight if theta_tight is not None else -2.0 * alpha_e
        self.t_debounce = t_debounce

        # 组合底层计算器
        self.safety_calc = SafetyDriveCalculator()
        self.gain_calc = GainDriveCalculator()
        self.efficiency_calc = EfficiencyDriveCalculator()
        self.scene_calc = SceneWeightSwitcherCalculator()

    def compute_drives(
        self,
        state: dict
    ) -> tuple[float, float, float]:
        """计算三大驱力值（A27.2）。

        从系统状态中提取异常信号、认知指标和资源指标，
        分别计算安全驱力、收益驱力和能效驱力。

        Args:
            state: 系统状态字典，包含以下字段：
                - error_energies: tuple[float, ...] 各层精度加权误差能量
                - error_energy_stats: tuple[tuple[float, float], ...] 各层历史统计量
                - error_events: list[tuple[str, float]] 错误事件序列
                - observation: np.ndarray 当前观测
                - known_states: np.ndarray 已知状态库
                - risk_processes: list[tuple[float, float]] 风险进程列表
                - load_rate: float 负载变化率
                - load_rate_stats: tuple[float, float] 负载变化率统计量
                - free_energy: float 当前自由能
                - free_energy_baseline: float 自由能基线
                - new_info: float 新信息维度
                - accuracy_improvement: float 精度提升量
                - coverage: float 认知覆盖度
                - cpu_ratio: float CPU占用率
                - mem_ratio: float 内存占用率
                - time_cost: float 推理耗时比
                - redundancy: float 计算冗余度

        Returns:
            (R_safe, R_gain, R_eff) 三大驱力值元组
            - R_safe <= 0（安全驱力恒非正）
            - R_gain 通常 >= 0（收益驱力）
            - R_eff <= 0（能效驱力恒非正）

        Raises:
            ValueError: 当state缺少必要字段时
            ValueError: 当alpha_s, alpha_g, alpha_e <= 0

        Examples:
            >>> engine = DriveEngine()
            >>> state = {
            ...     'error_energies': (0.5, 0.3),
            ...     'error_energy_stats': ((0.4, 0.1), (0.3, 0.05)),
            ...     'error_events': [],
            ...     'observation': np.array([1.0, 0.0]),
            ...     'known_states': np.array([[1.0, 0.0], [0.0, 1.0]]),
            ...     'risk_processes': [],
            ...     'load_rate': 0.0,
            ...     'load_rate_stats': (0.0, 0.01),
            ...     'free_energy': 1.5,
            ...     'free_energy_baseline': 1.0,
            ...     'new_info': 0.5,
            ...     'accuracy_improvement': 0.1,
            ...     'coverage': 0.7,
            ...     'cpu_ratio': 0.3,
            ...     'mem_ratio': 0.3,
            ...     'time_cost': 0.15,
            ...     'redundancy': 0.1
            ... }
            >>> r_safe, r_gain, r_eff = engine.compute_drives(state)
            >>> r_safe <= 0
            True
            >>> r_eff <= 0
            True
        """
        # 验证必要字段
        required_fields = [
            'error_energies', 'error_energy_stats', 'error_events',
            'observation', 'known_states', 'risk_processes', 'load_rate',
            'load_rate_stats', 'new_info', 'accuracy_improvement', 'coverage',
            'cpu_ratio', 'mem_ratio', 'time_cost', 'redundancy'
        ]
        for field in required_fields:
            if field not in state:
                raise ValueError(f"state 缺少必要字段: {field}")

        # 安全驱力：提取异常信号 → 计算权重 → 计算安全驱力
        anomalies = self.safety_calc.compute_anomalies(
            error_energies=state['error_energies'],
            error_energy_stats=state['error_energy_stats'],
            error_events=state['error_events'],
            observation=state['observation'],
            known_states=state['known_states'],
            risk_processes=state['risk_processes'],
            load_rate=state['load_rate'],
            load_rate_stats=state['load_rate_stats']
        )
        weights = self.safety_calc.compute_weights(anomalies, beta_s=self.beta)
        r_safe = self.safety_calc.compute_safe_reward(
            anomalies, weights, alpha_s=self.alpha_s
        )

        # 若提供了自由能信息，可验证一致性（可选）
        if 'free_energy' in state and 'free_energy_baseline' in state:
            r_safe_fe = self.safety_calc.compute_safe_reward_from_free_energy(
                state['free_energy'], state['free_energy_baseline'], self.alpha_s
            )
            # 在正常范围内允许一定容差，此处仅做计算，不强制校验

        # 收益驱力
        r_gain = self.gain_calc.compute_gain_reward(
            new_information=state['new_info'],
            accuracy_improvement=state['accuracy_improvement'],
            coverage=state['coverage'],
            alpha_g=self.alpha_g
        )

        # 能效驱力
        r_eff = self.efficiency_calc.compute_eff_reward(
            cpu_ratio=state['cpu_ratio'],
            mem_ratio=state['mem_ratio'],
            time_cost=state['time_cost'],
            redundancy=state['redundancy'],
            alpha_e=self.alpha_e
        )

        return (r_safe, r_gain, r_eff)

    def compute_weights(
        self,
        drives: tuple[float, float, float],
        scene_context: Optional[dict] = None
    ) -> tuple[float, float, float]:
        """计算动态权重（A27.3）。

        根据当前场景和驱力值，决定安全、收益、能效各占多少权重。
        紧急时安全权重高，空闲时收益权重高，缺资源时能效权重高，混合时按比例分配。

        Args:
            drives: 三大驱力值 (R_safe, R_gain, R_eff)
            scene_context: 场景上下文（含上一时刻场景和切换时间），可选字段：
                - prev_scene: SceneType 上一时刻场景
                - scene_start_time: int 场景开始时间戳
                - current_time: int 当前时间戳

        Returns:
            (W_safe, W_gain, W_eff) 动态权重元组
            - W_i >= 0（权重非负）
            - sum(W_i) ≈ 1（权重归一化）

        Raises:
            ValueError: 当drives格式不正确

        Examples:
            >>> engine = DriveEngine()
            >>> weights = engine.compute_weights((-3.0, 0.5, -0.3))
            >>> sum(weights)
            1.0
            >>> weights[0] > weights[1]  # 紧急场景安全权重高
            True
        """
        if len(drives) != 3:
            raise ValueError(f"drives 必须为长度3的元组，收到长度 {len(drives)}")

        r_safe, r_gain, r_eff = drives

        # 场景上下文默认值
        if scene_context is None:
            scene_context = {}
        prev_scene = scene_context.get('prev_scene', SceneType.MIXED)
        scene_start_time = scene_context.get('scene_start_time', 0)
        current_time = scene_context.get('current_time', 0)
        prev_weights = scene_context.get('prev_weights', (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))

        # 1. 场景识别
        new_scene = self.scene_calc.classify_scene(
            r_safe, r_gain, r_eff,
            self.theta_urgent, self.theta_stable,
            self.theta_idle, self.theta_tight
        )

        # 2. 防抖动
        effective_scene, _ = self.scene_calc.apply_debounce(
            new_scene, prev_scene, scene_start_time, current_time, self.t_debounce
        )

        # 3. 目标权重
        scene_weights = {
            SceneType.URGENT: (0.8, 0.1, 0.1),
            SceneType.STABLE: (0.1, 0.8, 0.1),
            SceneType.TIGHT: (0.1, 0.1, 0.8),
            SceneType.MIXED: (0.33, 0.33, 0.34)
        }

        # 4. 混合插值
        if effective_scene == SceneType.MIXED:
            memberships = self.scene_calc.compute_memberships(
                r_safe, r_gain, r_eff,
                self.theta_urgent, self.theta_stable,
                self.theta_idle, self.theta_tight
            )
            target_weights = self.scene_calc.interpolate_weights(memberships, scene_weights)
        else:
            target_weights = scene_weights[effective_scene]

        # 5. 权重平滑过渡 (A24.6)
        w_safe = (1.0 - self.alpha_w) * prev_weights[0] + self.alpha_w * target_weights[0]
        w_gain = (1.0 - self.alpha_w) * prev_weights[1] + self.alpha_w * target_weights[1]
        w_eff = (1.0 - self.alpha_w) * prev_weights[2] + self.alpha_w * target_weights[2]

        # 归一化
        total = w_safe + w_gain + w_eff
        if total > 0:
            w_safe /= total
            w_gain /= total
            w_eff /= total
        else:
            w_safe = w_gain = w_eff = 1.0 / 3.0

        return (w_safe, w_gain, w_eff)

    def compute_gap(
        self,
        drives: tuple[float, float, float],
        targets: Optional[tuple[float, float, float]] = None
    ) -> tuple[float, float, float]:
        """计算三大缺口（A27.4）。

        计算当前驱力值与终极目标之间的差距。
        三个缺口都永远为正——这是Agent永不停止行动的根本原因。

        Args:
            drives: 三大驱力值 (R_safe, R_gain, R_eff)
            targets: 目标值 (V_safe_target, V_gain_target, V_eff_target)
                     默认为 (0, None, 0)，None表示自动计算全知目标
                     - V_safe_target: 零异常——绝对安全，默认0
                     - V_gain_target: 全知——完全理解环境，默认自动计算
                     - V_eff_target: 零资源消耗——完美效率，默认0

        Returns:
            (ΔV_safe, ΔV_gain, ΔV_eff) 三大缺口元组
            - ΔV_i >= 0（缺口非负）
            - ΔV_i > 0（缺口永不为零，热力学第二定律推论）

        Raises:
            ValueError: 当drives格式不正确

        Examples:
            >>> engine = DriveEngine()
            >>> gaps = engine.compute_gap((-1.0, 0.5, -0.3))
            >>> all(g >= 0 for g in gaps)
            True
        """
        if len(drives) != 3:
            raise ValueError(f"drives 必须为长度3的元组，收到长度 {len(drives)}")

        r_safe, r_gain, r_eff = drives

        # 默认目标值
        if targets is None:
            targets = (0.0, None, 0.0)

        v_safe_target = targets[0] if targets[0] is not None else 0.0
        # 收益目标：全知目标，默认自动计算
        if targets[1] is not None:
            v_gain_target = targets[1]
        else:
            # 默认全知目标：alpha_g * (H[O] + DeltaF_max + 1)
            # 此处用 state 中无法直接获取的熵和最大自由能差，采用启发式默认值
            v_gain_target = self.alpha_g * 10.0
        v_eff_target = targets[2] if targets[2] is not None else 0.0

        # 计算缺口 (A27.4)
        delta_v_safe = max(0.0, v_safe_target - r_safe)
        # 安全驱力 r_safe <= 0，所以 delta_v_safe = -r_safe
        delta_v_safe = max(delta_v_safe, -r_safe)

        delta_v_gain = max(0.0, v_gain_target - r_gain)

        delta_v_eff = max(0.0, v_eff_target - r_eff)
        # 能效驱力 r_eff <= 0，所以 delta_v_eff = -r_eff
        delta_v_eff = max(delta_v_eff, -r_eff)

        # 确保缺口永不为零 (A25.6)
        eps = 1e-10
        delta_v_safe = max(delta_v_safe, eps)
        delta_v_gain = max(delta_v_gain, eps)
        delta_v_eff = max(delta_v_eff, eps)

        return (delta_v_safe, delta_v_gain, delta_v_eff)

    def compute_total_reward(
        self,
        drives: tuple[float, float, float],
        weights: tuple[float, float, float]
    ) -> float:
        """计算全局总奖励（A27.5）。

        安全成绩×安全权重 + 收益成绩×收益权重 + 能效成绩×能效权重 = 总分。
        总分为正表示整体向好，为负表示整体向坏。

        由定理24.1，最大化全局奖励等价于最小化预期自由能：
        R_total ∝ -G(π)

        Args:
            drives: 三大驱力值 (R_safe, R_gain, R_eff)
            weights: 动态权重 (W_safe, W_gain, W_eff)

        Returns:
            R_total: 全局总奖励
            - R_total > 0: 收益超过代价
            - R_total < 0: 代价超过收益
            - R_total = 0: 收益与代价平衡

        Raises:
            ValueError: 当weights不归一化或存在负值

        Examples:
            >>> engine = DriveEngine()
            >>> r_total = engine.compute_total_reward(
            ...     (-1.0, 0.5, -0.3),
            ...     (0.5, 0.3, 0.2)
            ... )
            >>> isinstance(r_total, float)
            True
        """
        if len(drives) != 3:
            raise ValueError(f"drives 必须为长度3的元组，收到长度 {len(drives)}")
        if len(weights) != 3:
            raise ValueError(f"weights 必须为长度3的元组，收到长度 {len(weights)}")

        # 验证权重非负
        for i, w in enumerate(weights):
            if w < -1e-10:
                raise ValueError(f"第 {i} 个权重必须 ≥ 0，收到 {w}")

        # 验证权重归一化
        w_sum = sum(weights)
        if not (0.99 <= w_sum <= 1.01):
            raise ValueError(f"权重之和必须 ≈ 1，收到 {w_sum}")

        r_safe, r_gain, r_eff = drives
        w_safe, w_gain, w_eff = weights

        return w_safe * r_safe + w_gain * r_gain + w_eff * r_eff

    def update(
        self,
        state: dict,
        prev_engine_state: Optional[DriveEngineState] = None
    ) -> DriveEngineState:
        """更新驱力引擎状态（A27.6）。

        一步完成所有计算并返回新的引擎状态。
        传入系统状态，引擎自动完成驱力计算、权重分配、缺口计算和总奖励计算。

        计算流程：
            state → compute_drives → (R_safe, R_gain, R_eff)
                         ↓
            compute_weights(drives, scene) → (W_safe, W_gain, W_eff)
                         ↓
            compute_gap(drives, targets) → (ΔV_safe, ΔV_gain, ΔV_eff)
                         ↓
            compute_total_reward(drives, weights) → R_total
                         ↓
            DriveEngineState

        Args:
            state: 系统状态字典，包含所有原始数据
            prev_engine_state: 上一时刻的引擎状态（用于时间平滑和防抖动）
                               首次调用时为None

        Returns:
            新的DriveEngineState实例（不可变，保证线程安全和可追溯性）

        Raises:
            ValueError: 当state缺少必要字段
            ValueError: 当不变量验证失败

        Examples:
            >>> engine = DriveEngine()
            >>> state = {
            ...     'error_energies': (0.5, 0.3),
            ...     'error_energy_stats': ((0.4, 0.1), (0.3, 0.05)),
            ...     'error_events': [],
            ...     'observation': np.array([1.0, 0.0]),
            ...     'known_states': np.array([[1.0, 0.0], [0.0, 1.0]]),
            ...     'risk_processes': [],
            ...     'load_rate': 0.0,
            ...     'load_rate_stats': (0.0, 0.01),
            ...     'free_energy': 1.5,
            ...     'free_energy_baseline': 1.0,
            ...     'new_info': 0.5,
            ...     'accuracy_improvement': 0.1,
            ...     'coverage': 0.7,
            ...     'cpu_ratio': 0.3,
            ...     'mem_ratio': 0.3,
            ...     'time_cost': 0.15,
            ...     'redundancy': 0.1
            ... }
            >>> new_state = engine.update(state)
            >>> isinstance(new_state, DriveEngineState)
            True
            >>> new_state.r_safe <= 0
            True
            >>> abs(sum(new_state.weights) - 1.0) < 1e-6
            True
        """
        # 1. 计算三大驱力
        drives = self.compute_drives(state)
        r_safe, r_gain, r_eff = drives

        # 2. 准备场景上下文
        scene_context = {}
        timestamp = state.get('timestamp', 0)
        if prev_engine_state is not None:
            scene_context['prev_scene'] = prev_engine_state.scene
            scene_context['scene_start_time'] = prev_engine_state.scene_switcher.t_last_switch
            scene_context['current_time'] = timestamp
            scene_context['prev_weights'] = prev_engine_state.weights
            # 时间平滑安全驱力 (A21.5)
            r_safe = self.safety_calc.smooth_safe_reward(
                r_safe, prev_engine_state.r_safe, self.gamma_s
            )
            drives = (r_safe, r_gain, r_eff)
        else:
            scene_context['prev_scene'] = SceneType.MIXED
            scene_context['scene_start_time'] = 0
            scene_context['current_time'] = timestamp
            scene_context['prev_weights'] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

        # 3. 计算动态权重
        weights = self.compute_weights(drives, scene_context)

        # 4. 计算缺口
        gaps = self.compute_gap(drives)
        delta_v_safe, delta_v_gain, delta_v_eff = gaps

        # 5. 计算总奖励
        total_reward = self.compute_total_reward(drives, weights)

        # 6. 构建 SceneWeightSwitcher
        memberships = self.scene_calc.compute_memberships(
            r_safe, r_gain, r_eff,
            self.theta_urgent, self.theta_stable,
            self.theta_idle, self.theta_tight
        )
        new_scene = self.scene_calc.classify_scene(
            r_safe, r_gain, r_eff,
            self.theta_urgent, self.theta_stable,
            self.theta_idle, self.theta_tight
        )
        effective_scene, switched = self.scene_calc.apply_debounce(
            new_scene,
            scene_context['prev_scene'],
            scene_context['scene_start_time'],
            timestamp,
            self.t_debounce
        )
        t_last_switch = scene_context['scene_start_time']
        if switched:
            t_last_switch = timestamp

        # 构造目标权重
        scene_weights = {
            SceneType.URGENT: (0.8, 0.1, 0.1),
            SceneType.STABLE: (0.1, 0.8, 0.1),
            SceneType.TIGHT: (0.1, 0.1, 0.8),
            SceneType.MIXED: (0.33, 0.33, 0.34)
        }
        target_weights = scene_weights[effective_scene]

        # 组合 ValueGap 和 GlobalReward 需要 stage_024/025 的导入
        # 为保持自洽，此处直接构造 SceneWeightSwitcher，不嵌套引用其他阶段数据类
        from stage_025 import ValueGap
        from stage_024 import GlobalReward

        global_reward = GlobalReward(
            r_safe=r_safe,
            r_gain=r_gain,
            r_eff=r_eff,
            weights=weights,
            total_reward=total_reward
        )
        value_gap = ValueGap(
            global_reward=global_reward,
            v_target=delta_v_safe + delta_v_gain + delta_v_eff,
            v_safe_target=0.0,
            v_gain_target=self.alpha_g * 10.0,
            v_eff_target=0.0,
            delta_v=delta_v_safe + delta_v_gain + delta_v_eff,
            delta_v_safe=delta_v_safe,
            delta_v_gain=delta_v_gain,
            delta_v_eff=delta_v_eff,
            motivations=(delta_v_safe, delta_v_gain, delta_v_eff),
            motivation_fn='identity'
        )
        scene_switcher = SceneWeightSwitcher(
            value_gap=value_gap,
            scene=effective_scene,
            target_weights=target_weights,
            theta_urgent=self.theta_urgent,
            theta_stable=self.theta_stable,
            theta_idle=self.theta_idle,
            theta_tight=self.theta_tight,
            memberships=memberships,
            t_last_switch=t_last_switch,
            t_debounce=self.t_debounce,
            pending_scene=new_scene if new_scene != effective_scene else None
        )

        # 7. 封装为 DriveEngineState
        return DriveEngineState(
            scene_switcher=scene_switcher,
            r_safe=r_safe,
            r_gain=r_gain,
            r_eff=r_eff,
            weights=weights,
            delta_v_safe=delta_v_safe,
            delta_v_gain=delta_v_gain,
            delta_v_eff=delta_v_eff,
            total_reward=total_reward,
            timestamp=timestamp
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `SceneWeightSwitcher` | ← 阶段26 | 作为驱力引擎状态的场景权重切换器组合字段 |
| **输入** | A21.1-A21.5（安全驱力） | ← 阶段21 | compute_drives的安全分量计算 |
| **输入** | A22.1-A22.4（收益驱力） | ← 阶段22 | compute_drives的收益分量计算 |
| **输入** | A23.1-A23.6（能效驱力） | ← 阶段23 | compute_drives的能效分量计算 |
| **输入** | A24.1-A24.6（全局奖励与权重） | ← 阶段24 | compute_weights和compute_total_reward的计算基础 |
| **输入** | A25.1-A25.6（缺口驱动） | ← 阶段25 | compute_gap的计算基础 |
| **输出** | `DriveEngineState` | → 阶段28 | 驱力引擎状态作为时间平滑的输入 |
| **输出** | DriveEngine接口 | → 阶段30 | 驱力引擎的完整接口用于验证 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 驱力计算与向量化 | `jax.vmap`批量计算驱力值；`jax.lax.scan`实现update的函数式循环 |
| pydantic | **本阶段新增**：状态验证 | DriveEngineState的数据验证框架，自动检查所有不变量（驱力符号、权重归一化、缺口非负、总奖励一致性），替代手写__post_init__验证逻辑 |
| Optuna | 超参数优化 | 自动调优 $\alpha_s, \alpha_g, \alpha_e, \beta, \alpha_w$ 等超参数 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V27.1 | 安全驱力非正 | $R_{\text{safe}} \leq 0$ 对所有合法state成立 | 随机状态测试 |
| V27.2 | 能效驱力非正 | $R_{\text{eff}} \leq 0$ 对所有合法state成立 | 随机状态测试 |
| V27.3 | 权重归一化 | $\sum_i W_i = 1$ 对所有合法输入成立 | 数值验证 |
| V27.4 | 权重非负 | $W_i \geq 0$ 对所有合法输入成立 | 边界值测试 |
| V27.5 | 缺口非负 | $\Delta V_i \geq 0$ 对所有合法输入成立 | 随机驱力值测试 |
| V27.6 | 总奖励一致性 | $R_{\text{total}} = \sum_i W_i \cdot R_i$ | 数值一致性验证 |
| V27.7 | update完整性 | update返回的DriveEngineState包含所有字段 | 字段完整性检查 |
| V27.8 | update确定性 | 相同输入产生相同输出 | 重复调用验证 |
| V27.9 | 场景切换正确性 | 驱力值变化时场景正确切换 | 场景边界测试 |
| V27.10 | DriveEngineState组合 | `DriveEngineState.scene_switcher` 类型为 `SceneWeightSwitcher` | 类型检查 |
| V27.11 | 非法输入拒绝 | 缺少必要字段、参数越界等抛出 `ValueError` | 异常测试 |
| V27.12 | 性能基准 | update单次调用 < 1ms（$L \leq 10$） | 性能基准测试 |

---

> **阶段27完** | 下一步：阶段28将从本阶段的驱力引擎状态出发，设计驱力信号的时间平滑与防抖动机制，确保驱力信号的稳定性和响应速度。
