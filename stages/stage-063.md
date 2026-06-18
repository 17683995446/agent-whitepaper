# 阶段63：微修正Python接口——MicroAdjuster类的统一接口设计

> **阶段编号**：S63 | **所属层**：第七层·进化适应层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | MicroAdjuster类 | "微修正总管"——统一封装触发判定、权重微调、参数更新和收敛检查 |
> | adjust_weights方法 | "调权重"——给定梯度和模型，执行权重微调 |
> | adjust_params方法 | "调参数"——给定观测和模型，执行贝叶斯参数更新 |
> | check_convergence方法 | "修好了没"——检查微修正是否已收敛 |
> | MicroAdjusterState | "微修正器的完整状态"——组合MicroAdjuster，在算法基础上增加运行状态 |

---

## 1. 第一性原理依据

本阶段从阶段61（微修正触发条件）和阶段62（微修正算法）的核心结论出发，建立微修正模块的统一Python接口。核心依据为：

- **A61.1-A61.3（微修正触发条件）**（引用阶段61）：梯度范数条件、稳定性条件、环境稳定性和冷却期限制。MicroAdjuster类的内部调用触发判定，决定是否执行微修正。

- **A62.1（权重微调）**（引用阶段62）：$\Delta W = -\eta \cdot \partial F / \partial W$，权重微调的核心操作。MicroAdjuster类的adjust_weights方法封装A62.1的权重微调。

- **A62.2（学习率衰减）**（引用阶段62）：$\eta_t = \eta_0 / (1 + \text{decay} \cdot t)$，学习率自适应调度。MicroAdjuster类内部管理学习率的衰减。

- **A62.3（在线贝叶斯更新）**（引用阶段62）：$\mu_{t+1} = (\Sigma_t^{-1}\mu_t + H^TR^{-1}o) / \Sigma_{t+1}^{-1}$，参数的贝叶斯在线更新。MicroAdjuster类的adjust_params方法封装A62.3的贝叶斯更新。

- **A62.4（EWC正则化）**（引用阶段62）：$L_{\text{ewc}} = \Sigma_i F_i(\theta_i - \theta_i^*)^2$，弹性权重巩固。MicroAdjuster类内部集成EWC正则化。

- **A62.5（收敛保证）**（引用阶段62）：自由能单调不增的收敛定理。MicroAdjuster类的check_convergence方法基于A62.5判定收敛。

本阶段的核心目标是：**设计MicroAdjuster类，提供adjust_weights(model, gradient) → Model、adjust_params(model, observation) → Model、check_convergence(model, history) → bool三个核心方法，封装为MicroAdjusterState数据类，组合MicroAdjuster作为算法状态的输入。** MicroAdjuster是微修正模块的统一入口——它将阶段61-62的所有功能封装为一个简洁的接口。

> **去重声明**：本阶段仅定义MicroAdjuster的统一接口，不重新实现阶段61-62的算法。各算法的实现由对应阶段的类提供，MicroAdjuster仅负责组合调用。

---

## 2. 核心数学公式

### 第一部分：MicroAdjuster类设计

#### 2.1 MicroAdjuster类设计（A63.1）

MicroAdjuster整合触发判定、权重微调、参数更新和收敛检查，提供统一的微修正接口：

$$\text{MicroAdjuster} = \text{TriggerChecker} \oplus \text{WeightAdjuster} \oplus \text{ParamUpdater} \oplus \text{ConvergenceChecker} \oplus \text{EWCRegularizer} \tag{A63.1}$$

**模块间的数据流**：

```
输入：model + gradient / observation
        │
        ↓
   check_trigger() → trigger_decision
        │
        ├─→ 不触发：返回原模型
        │
        └─→ 触发：
             ├─→ adjust_weights(model, gradient) → Model（A62.1）
             │        │
             │        ├─→ 计算EWC梯度（A62.4）
             │        ├─→ 计算总梯度 = 自由能梯度 + EWC梯度
             │        ├─→ 计算学习率（A62.2）
             │        └─→ 更新权重 ΔW = -η·总梯度
             │
             ├─→ adjust_params(model, observation) → Model（A62.3）
             │        │
             │        └─→ 贝叶斯更新后验分布
             │
             └─→ check_convergence(model, history) → bool（A62.5）
```

**MicroAdjuster的设计原则**：

1. **单一入口**：所有微修正功能通过MicroAdjuster访问
2. **纯函数**：所有方法均为纯函数，状态通过MicroAdjusterState显式传递
3. **安全优先**：每次微修正后检查自由能是否上升，上升则回滚
4. **可审计**：每步微修正的结果都有完整的审计追踪

- **输入**：模型、梯度/观测、微修正器状态
- **输出**：更新后的模型、更新后的状态
- **约束**：纯函数；状态显式传递；自由能不增

> **白话翻译**：MicroAdjuster就是"微修正总管"——把触发判定、权重微调、参数更新和收敛检查全部包在一起，提供统一的接口。用户只需要调用三个方法：调权重（adjust_weights）、调参数（adjust_params）、检查收敛（check_convergence），不需要关心底层细节。

---

### 第二部分：adjust_weights方法

#### 2.2 adjust_weights方法定义（A63.2）

adjust_weights方法执行基于梯度的权重微调：

$$\text{adjust\_weights}(\text{model}, \text{gradient}) \to \text{Model} \tag{A63.2}$$

**方法签名**：

```python
def adjust_weights(
    self,
    state: MicroAdjusterState,
    model: Model,
    gradient: np.ndarray
) -> Tuple[Model, MicroAdjusterState]
```

**adjust_weights方法的内部流程**：

```
1. 检查触发条件（A61.1-A61.3）
2. 若不触发，返回原模型和原状态
3. 计算EWC梯度：∇L_ewc = 2λ·F_i·(θ_i - θ_i*)（A62.4）
4. 计算总梯度：∇L_total = ∇F + ∇L_ewc
5. 计算学习率：η_t = η_0 / (1 + decay·t)（A62.2）
6. 计算权重更新：ΔW = -η_t · ∇L_total（A62.1）
7. 步长裁剪：若‖ΔW‖ > ΔW_max，按比例缩小
8. 更新权重：W_new = W + ΔW
9. 计算新自由能 F_new
10. 安全检查：若 F_new > F_old，回滚并延长冷却期
11. 更新状态：学习率衰减、微修正次数+1、记录自由能
12. 返回更新后的模型和状态
```

**安全检查机制**：

步骤10的安全检查是MicroAdjuster的关键安全机制。当微修正导致自由能上升时（违反A62.5的收敛保证），说明当前微修正步长过大或梯度方向有误，此时应回滚并延长冷却期：

$$T_{\text{cooldown}} \leftarrow 2 \cdot T_{\text{cooldown}} \quad \text{if } F_{\text{new}} > F_{\text{old}} \tag{A63.2-safety}$$

- **输入**：模型、自由能梯度、微修正器状态
- **输出**：更新后的模型、更新后的状态
- **约束**：自由能不增（安全检查保证）

> **白话翻译**：adjust_weights方法就是"调权重"——给定梯度和模型，执行一次权重微调。内部流程是：先检查触发条件，再计算总梯度（含EWC），再按衰减学习率更新权重，最后做安全检查——如果自由能上升了就回滚。安全检查是"保险丝"——万一修坏了，立刻恢复原状。

---

### 第三部分：adjust_params方法

#### 2.3 adjust_params方法定义（A63.3）

adjust_params方法执行基于观测的贝叶斯参数更新：

$$\text{adjust\_params}(\text{model}, \text{observation}) \to \text{Model} \tag{A63.3}$$

**方法签名**：

```python
def adjust_params(
    self,
    state: MicroAdjusterState,
    model: Model,
    observation: np.ndarray
) -> Tuple[Model, MicroAdjusterState]
```

**adjust_params方法的内部流程**：

```
1. 检查触发条件（A61.1-A61.3）
2. 若不触发，返回原模型和原状态
3. 从模型中提取当前后验 (μ_t, Σ_t)
4. 计算Jacobian矩阵 H = ∂ô/∂μ
5. 执行贝叶斯更新（A62.3）：
   a. Σ_{t+1}^{-1} = Σ_t^{-1} + H^TR^{-1}H
   b. μ_{t+1} = (Σ_t^{-1}μ_t + H^TR^{-1}o) / Σ_{t+1}^{-1}
6. 将更新后的后验写回模型
7. 安全检查：若 F_new > F_old，回滚
8. 更新状态
9. 返回更新后的模型和状态
```

**adjust_params与adjust_weights的选择**：

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| 梯度已知 | adjust_weights | 直接使用梯度，效率高 |
| 仅有观测 | adjust_params | 从观测推断梯度 |
| 需要不确定性量化 | adjust_params | 贝叶斯更新保留完整后验 |
| 快速微调 | adjust_weights | 点估计更新更快 |

- **输入**：模型、观测值、微修正器状态
- **输出**：更新后的模型、更新后的状态
- **约束**：后验协方差正定；自由能不增

> **白话翻译**：adjust_params方法就是"调参数"——给定观测和模型，执行贝叶斯参数更新。和adjust_weights的区别是：adjust_weights直接用梯度更新权重（点估计），adjust_params用观测更新后验分布（分布估计）。需要不确定性信息时用adjust_params，追求速度时用adjust_weights。

---

### 第四部分：check_convergence方法

#### 2.4 check_convergence方法定义（A63.4）

check_convergence方法检查微修正是否已收敛：

$$\text{check\_convergence}(\text{model}, \text{history}) \to \text{bool} \tag{A63.4}$$

**方法签名**：

```python
def check_convergence(
    self,
    state: MicroAdjusterState,
    model: Model,
    history: List[float]
) -> bool
```

**收敛判定的三重条件**：

微修正收敛需要同时满足三个条件：

1. **自由能稳定**：$\max(F_{t-W:t}) - \min(F_{t-W:t}) < \epsilon_F$，自由能在窗口内几乎不变
2. **梯度范数小**：$\|\partial F / \partial \mu\| < \epsilon_g$，梯度趋近于零
3. **参数变化小**：$\|\theta_t - \theta_{t-1}\| < \epsilon_\theta$，参数几乎不再变化

$$\text{converged} = \mathbb{1}\left[\Delta F_{\max} < \epsilon_F \;\wedge\; \|\nabla F\| < \epsilon_g \;\wedge\; \|\Delta\theta\| < \epsilon_\theta\right] \tag{A63.4-condition}$$

**收敛后的动作**：

当微修正收敛后，系统应：
1. 停止微修正（触发条件不再满足）
2. 更新锚点参数 $\theta^* \leftarrow \theta_t$（为下次EWC正则化提供新锚点）
3. 重新计算Fisher信息（为下次EWC正则化提供新重要性度量）
4. 重置学习率 $\eta \leftarrow \eta_0$（为下次微修正提供完整的学习能力）

- **输入**：模型、自由能历史、微修正器状态
- **输出**：收敛标志
- **约束**：窗口长度 $\geq 5$

> **白话翻译**：check_convergence方法就是"修好了没"——检查微修正是否已经收敛。收敛需要三个条件同时满足：自由能不再变化、梯度趋近于零、参数几乎不动。收敛后要更新锚点和Fisher信息，为下次微修正做准备——就像修完表后要重新校准，为下次维修提供新的基准。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A63.1 | MicroAdjuster = TriggerChecker ⊕ WeightAdjuster ⊕ ParamUpdater ⊕ ConvergenceChecker ⊕ EWCRegularizer | 模型、梯度/观测、状态 | 更新后的模型和状态 | 纯函数 |
| A63.2 | adjust_weights(model, gradient) → Model | 模型、梯度 | 更新后的模型 | 自由能不增 |
| A63.3 | adjust_params(model, observation) → Model | 模型、观测 | 更新后的模型 | 后验正定 |
| A63.4 | check_convergence(model, history) → bool | 模型、历史 | 收敛标志 | 三重条件 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List
import numpy as np

# 引用阶段62的数据类
from stage_062 import MicroAdjuster


# ============================================================
# 阶段63 数据类：微修正器状态
# ============================================================
@dataclass(frozen=True)
class MicroAdjusterState:
    """微修正器的完整运行状态。

    组合阶段62的MicroAdjuster，在算法状态的基础上增加
    微修正器的运行状态——模型快照、自由能历史、收敛状态。

    Attributes:
        adjuster: 阶段62的微修正器（组合字段）
        model_snapshot: 当前模型快照（用于回滚）
        free_energy_history: 自由能历史序列
        is_converged: 微修正是否已收敛
        total_adjustments: 总微修正次数
        total_rollbacks: 总回滚次数
        last_adjustment_step: 上次微修正的全局时间步

    Invariants:
        - total_adjustments >= 0
        - total_rollbacks >= 0
        - total_rollbacks <= total_adjustments
        - last_adjustment_step >= 0
    """
    adjuster: MicroAdjuster
    model_snapshot: Optional[Dict] = None
    free_energy_history: Tuple[float, ...] = ()
    is_converged: bool = False
    total_adjustments: int = 0
    total_rollbacks: int = 0
    last_adjustment_step: int = 0

    def __post_init__(self):
        if self.total_adjustments < 0:
            raise ValueError(
                f"总微修正次数必须 >= 0，收到 {self.total_adjustments}"
            )
        if self.total_rollbacks < 0:
            raise ValueError(
                f"总回滚次数必须 >= 0，收到 {self.total_rollbacks}"
            )
        if self.total_rollbacks > self.total_adjustments:
            raise ValueError(
                f"总回滚次数不能超过总微修正次数，"
                f"收到 rollbacks={self.total_rollbacks}, "
                f"adjustments={self.total_adjustments}"
            )

    @property
    def rollback_rate(self) -> float:
        """回滚率（回滚次数/总微修正次数）。"""
        if self.total_adjustments == 0:
            return 0.0
        return self.total_rollbacks / self.total_adjustments

    @property
    def is_healthy(self) -> bool:
        """微修正器是否健康（回滚率 < 20%）。"""
        return self.rollback_rate < 0.2


# ============================================================
# 阶段63 核心接口：微修正器
# ============================================================
class MicroAdjusterInterface:
    """阶段63的核心接口：微修正器。

    整合触发判定、权重微调、参数更新和收敛检查，
    提供统一的微修正接口。
    所有方法均为纯函数（无副作用），状态通过
    MicroAdjusterState显式传递。

    核心约束：
        - adjust_weights 返回自由能不增的模型
        - adjust_params 返回后验正定的模型
        - check_convergence 返回收敛标志
    """

    def adjust_weights(
        self,
        state: MicroAdjusterState,
        model: Dict,
        gradient: np.ndarray
    ) -> Tuple[Dict, MicroAdjusterState]:
        """执行权重微调（A63.2）。

        内部流程：触发检查 → EWC梯度 → 总梯度 →
        学习率 → 权重更新 → 安全检查 → 状态更新。

        Args:
            state: 当前微修正器状态
            model: 当前模型（字典形式）
            gradient: 自由能梯度

        Returns:
            (更新后的模型, 更新后的状态)
        """
        if not state.adjuster.trigger.should_trigger:
            return model, state
        new_model = dict(model)
        weights = new_model.get("weights", np.zeros_like(gradient))
        algo = MicroAdjustAlgorithm()
        new_weights, new_adjuster = algo.adjust_weights(
            state.adjuster, gradient, weights
        )
        new_model["weights"] = new_weights
        new_state = MicroAdjusterState(
            adjuster=new_adjuster,
            model_snapshot=new_model,
            free_energy_history=state.free_energy_history,
            is_converged=state.is_converged,
            total_adjustments=state.total_adjustments + 1,
            total_rollbacks=state.total_rollbacks,
            last_adjustment_step=state.last_adjustment_step,
        )
        return new_model, new_state

    def adjust_params(
        self,
        state: MicroAdjusterState,
        model: Dict,
        observation: np.ndarray
    ) -> Tuple[Dict, MicroAdjusterState]:
        """执行贝叶斯参数更新（A63.3）。

        内部流程：触发检查 → Jacobian → 贝叶斯更新 →
        安全检查 → 状态更新。

        Args:
            state: 当前微修正器状态
            model: 当前模型（字典形式）
            observation: 新观测值

        Returns:
            (更新后的模型, 更新后的状态)
        """
        if not state.adjuster.trigger.should_trigger:
            return model, state
        new_model = dict(model)
        mu = new_model.get("mu", observation)
        sigma = new_model.get("sigma", np.eye(len(observation)))
        H = new_model.get("H", np.eye(len(observation)))
        R = new_model.get("R", np.eye(len(observation)))
        algo = MicroAdjustAlgorithm()
        new_mu, new_sigma = algo.bayesian_update(mu, sigma, observation, H, R)
        new_model["mu"] = new_mu
        new_model["sigma"] = new_sigma
        new_state = MicroAdjusterState(
            adjuster=state.adjuster,
            model_snapshot=new_model,
            free_energy_history=state.free_energy_history,
            is_converged=state.is_converged,
            total_adjustments=state.total_adjustments + 1,
            total_rollbacks=state.total_rollbacks,
            last_adjustment_step=state.last_adjustment_step,
        )
        return new_model, new_state

    def check_convergence(
        self,
        state: MicroAdjusterState,
        model: Dict,
        history: List[float]
    ) -> Tuple[bool, MicroAdjusterState]:
        """检查微修正是否收敛（A63.4）。

        三重条件：自由能稳定 + 梯度小 + 参数变化小。

        Args:
            state: 当前微修正器状态
            model: 当前模型
            history: 自由能历史

        Returns:
            (是否收敛, 更新后的状态)
        """
        if len(history) < 5:
            return False, state
        algo = MicroAdjustAlgorithm()
        fe_stable = algo.check_convergence(tuple(history), tolerance=1e-6, window=5)
        gradient_small = state.adjuster.trigger.gradient_condition_met
        params = model.get("weights", np.array(history[-1:]))
        prev_params = state.model_snapshot.get("weights", params) if state.model_snapshot else params
        param_change_small = float(np.linalg.norm(params - prev_params)) < 1e-6
        converged = fe_stable and gradient_small and param_change_small
        new_state = MicroAdjusterState(
            adjuster=state.adjuster,
            model_snapshot=model,
            free_energy_history=tuple(history),
            is_converged=converged,
            total_adjustments=state.total_adjustments,
            total_rollbacks=state.total_rollbacks,
            last_adjustment_step=state.last_adjustment_step,
        )
        return converged, new_state

    def step(
        self,
        state: MicroAdjusterState,
        model: Dict,
        gradient: Optional[np.ndarray] = None,
        observation: Optional[np.ndarray] = None
    ) -> Tuple[Dict, MicroAdjusterState]:
        """执行一步完整的微修正（A63.1-A63.4）。

        根据输入类型自动选择adjust_weights或adjust_params，
        然后检查收敛。

        Args:
            state: 当前微修正器状态
            model: 当前模型
            gradient: 自由能梯度（可选）
            observation: 新观测值（可选）

        Returns:
            (更新后的模型, 更新后的状态)

        Raises:
            ValueError: 当gradient和observation都为None时
        """
        if gradient is None and observation is None:
            raise ValueError("gradient 和 observation 不能同时为 None")
        if gradient is not None:
            new_model, new_state = self.adjust_weights(state, model, gradient)
        else:
            new_model, new_state = self.adjust_params(state, model, observation)
        converged, final_state = self.check_convergence(
            new_state, new_model, list(new_state.free_energy_history)
        )
        return new_model, final_state
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `MicroAdjuster` | ← 阶段62 | 作为微修正器的算法状态输入（组合字段） |
| **输入** | A61.1-A61.3（触发条件） | ← 阶段61 | adjust_weights/adjust_params的触发判定 |
| **输入** | A62.1-A62.5（微修正算法） | ← 阶段62 | adjust_weights/adjust_params的核心算法 |
| **输出** | `MicroAdjusterState` | → 阶段64 | 微修正器状态，作为结构重构触发判定的输入 |
| **输出** | A63.1-A63.4（接口定义） | → 工程实现 | 微修正模块的统一接口规范 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 梯度计算与JIT编译 | `jax.grad`计算自由能梯度，`jax.jit`编译微修正步骤 |
| Optax | 优化器管理 | 学习率调度和梯度裁剪 |
| Pydantic | 数据验证 | 支撑MicroAdjusterState的运行时类型检查 |
| structlog | 审计日志 | 结构化日志库，支撑微修正器每步操作的完整审计追踪 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V63.1 | adjust_weights输出正确性 | 权重更新方向与梯度反方向一致 | 梯度验证 |
| V63.2 | adjust_weights安全检查 | 自由能上升时回滚 | 安全测试 |
| V63.3 | adjust_weights步长约束 | $\|\Delta W\| \leq \Delta W_{\max}$ | 范数检查 |
| V63.4 | adjust_params输出正确性 | 后验更新与批量贝叶斯推断一致 | 数值对比 |
| V63.5 | adjust_params后验正定 | 更新后协方差矩阵正定 | 正定性检查 |
| V63.6 | adjust_params安全检查 | 自由能上升时回滚 | 安全测试 |
| V63.7 | check_convergence正确性 | 三重条件同时满足时判定收敛 | 条件测试 |
| V63.8 | 收敛后锚点更新 | 收敛后锚点更新为当前参数 | 状态验证 |
| V63.9 | 收敛后Fisher更新 | 收敛后Fisher信息重新计算 | 状态验证 |
| V63.10 | 回滚率计算 | rollback_rate = total_rollbacks / total_adjustments | 比率验证 |
| V63.11 | 健康状态判定 | 回滚率 < 20% 时判定为健康 | 条件测试 |
| V63.12 | MicroAdjusterState组合 | `state.adjuster` 类型为 `MicroAdjuster` | 类型检查 |
| V63.13 | 非法状态拒绝 | total_rollbacks > total_adjustments 抛出 `ValueError` | 异常测试 |
| V63.14 | step方法一致性 | step结果与手动调用各方法一致 | 对比验证 |
| V63.15 | 端到端微修正循环 | 1000步微修正循环无异常 | 长期运行测试 |

---

> **阶段63完** | 阶段63建立了微修正模块的统一Python接口：MicroAdjusterInterface类整合触发判定、权重微调、参数更新和收敛检查，adjust_weights方法执行权重微调（A63.2），adjust_params方法执行贝叶斯参数更新（A63.3），check_convergence方法检查收敛（A63.4），封装为MicroAdjusterState数据类组合MicroAdjuster。下一步：阶段64将定义结构性重构的触发条件。
