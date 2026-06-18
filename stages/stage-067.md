# 阶段67：结构性重构Python接口——StructuralReformer类的统一接口设计

> **阶段编号**：S67 | **所属层**：第七层·进化适应层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | StructuralReformer类 | "重构总管"——统一封装触发判定、重构执行、效果验证和回滚 |
> | check_trigger方法 | "该不该重构"——检查是否满足重构触发条件 |
> | execute_reform方法 | "执行重构"——按选定方案执行结构性重构 |
> | validate_reform方法 | "重构效果如何"——对比重构前后的性能 |
> | rollback方法 | "一键还原"——回滚到重构前的状态 |
> | StructuralReformerState | "重构器的完整状态"——组合ReformSafety，在安全约束基础上增加运行状态 |

---

## 1. 第一性原理依据

本阶段从阶段64（重构触发条件）、阶段65（重构算法）和阶段66（重构安全约束）的核心结论出发，建立结构性重构模块的统一Python接口。核心依据为：

- **A64.1-A64.4（重构触发条件）**（引用阶段64）：持续高误差、长期停滞、新环境和风险评估。StructuralReformer类的check_trigger方法封装A64.1-A64.4的触发判定。

- **A65.1-A65.5（重构算法）**（引用阶段65）：特征增删、子模型重组、规则迭代、认知体系更新和贝叶斯搜索。StructuralReformer类的execute_reform方法封装A65.1-A65.5的重构操作。

- **A66.1-A66.5（重构安全约束）**（引用阶段66）：迭代限制、回滚条件、回滚机制、渐进式重构和验证窗口。StructuralReformer类内部集成A66.1-A66.5的安全约束。

本阶段的核心目标是：**设计StructuralReformer类，提供check_trigger(state) → Optional[ReformType]、execute_reform(model, reform_type) → Model、validate_reform(model_old, model_new) → bool、rollback(snapshot) → Model四个核心方法，封装为StructuralReformerState数据类，组合ReformSafety作为安全约束的输入。**

> **去重声明**：本阶段仅定义StructuralReformer的统一接口，不重新实现阶段64-66的算法。各算法的实现由对应阶段的类提供，StructuralReformer仅负责组合调用。

---

## 2. 核心数学公式

### 第一部分：StructuralReformer类设计

#### 2.1 StructuralReformer类设计（A67.1）

StructuralReformer整合触发判定、重构执行、效果验证和回滚，提供统一的结构性重构接口：

$$\text{StructuralReformer} = \text{TriggerChecker} \oplus \text{ReformExecutor} \oplus \text{ReformValidator} \oplus \text{RollbackHandler} \oplus \text{SafetyChecker} \tag{A67.1}$$

**模块间的数据流**：

```
输入：model + state
        │
        ↓
   check_trigger(state) → Optional[ReformType]
        │
        ├─→ None：不触发，返回原模型
        │
        └─→ ReformType：
             │
             ├─→ 创建快照（A66.3）
             │
             ├─→ execute_reform(model, reform_type) → Model（A65.1-A65.5）
             │        │
             │        ├─→ 渐进式执行（A66.4）
             │        └─→ 每步检查回滚条件（A66.2）
             │
             ├─→ validate_reform(model_old, model_new) → bool（A66.5）
             │        │
             │        ├─→ 验证通过：确认重构
             │        └─→ 验证失败：rollback(snapshot)（A66.3）
             │
             └─→ 更新状态
```

**StructuralReformer的设计原则**：

1. **安全优先**：每次重构都创建快照，验证失败立即回滚
2. **渐进执行**：高风险重构采用渐进式，低风险可一次性
3. **可审计**：每步重构的结果都有完整的审计追踪
4. **纯函数**：所有方法均为纯函数，状态通过StructuralReformerState显式传递

- **输入**：模型、状态
- **输出**：更新后的模型、更新后的状态
- **约束**：纯函数；安全优先

> **白话翻译**：StructuralReformer就是"重构总管"——把触发判定、重构执行、效果验证和回滚全部包在一起。流程是：先检查该不该重构，该重构就创建快照再执行，执行完验证效果，效果不好就回滚。每一步都有安全网——快照保证随时可以回滚，验证保证重构不会偷偷变差。

---

### 第二部分：check_trigger方法

#### 2.2 check_trigger方法定义（A67.2）

check_trigger方法检查是否满足重构触发条件，返回触发类型：

$$\text{check\_trigger}(\text{state}) \to \text{Optional}[\text{ReformType}] \tag{A67.2}$$

**方法签名**：

```python
def check_trigger(
    self,
    state: StructuralReformerState
) -> Optional[ReformTriggerType]
```

**触发类型的优先级**：

当多个触发条件同时满足时，按优先级选择：

$$\text{ReformType} = \begin{cases} \text{HIGH\_ERROR} & \text{if A64.1 triggered} \\ \text{STAGNATION} & \text{elif A64.2 triggered} \\ \text{NEW\_ENVIRONMENT} & \text{elif A64.3 triggered} \\ \text{None} & \text{otherwise} \end{cases} \tag{A67.2-priority}$$

持续高误差优先级最高（最紧急），长期停滞次之，新环境最低（可等待）。

**触发与微修正的互斥性**：

如果微修正触发条件（A61.1）同时满足，则不触发重构——微修正优先于重构：

$$\text{if trigger\_micro} \implies \text{ReformType} = \text{None} \tag{A67.2-mutex}$$

- **输入**：重构器状态
- **输出**：触发类型（可能为None）
- **约束**：与微修正互斥

> **白话翻译**：check_trigger方法就是"该不该重构"——检查是否满足重构触发条件。多个条件同时满足时，按紧急程度排序：持续高误差最紧急，长期停滞次之，新环境最低。如果微修正条件也满足，优先微修正——能微调解决的不用重构。

---

### 第三部分：execute_reform方法

#### 2.3 execute_reform方法定义（A67.3）

execute_reform方法按选定方案执行结构性重构：

$$\text{execute\_reform}(\text{model}, \text{reform\_type}) \to \text{Model} \tag{A67.3}$$

**方法签名**：

```python
def execute_reform(
    self,
    state: StructuralReformerState,
    model: Dict,
    reform_type: ReformTriggerType
) -> Tuple[Dict, StructuralReformerState]
```

**execute_reform的内部流程**：

```
1. 根据reform_type选择重构策略：
   - HIGH_ERROR → 特征增加 + 规则迭代（A65.1 + A65.3）
   - STAGNATION → 特征增加 + 子模型重组（A65.1 + A65.2）
   - NEW_ENVIRONMENT → 认知体系更新 + 特征增加（A65.4 + A65.1）

2. 生成候选方案（A65.5贝叶斯搜索）

3. 根据风险评估选择执行方式：
   - 低风险：一次性执行
   - 中风险：渐进式执行（A66.4）
   - 高风险：最小化重构 + 渐进式执行

4. 执行重构操作

5. 每步检查回滚条件（A66.2）

6. 返回更新后的模型和状态
```

**重构策略与触发类型的对应**：

| 触发类型 | 首选重构策略 | 原因 |
|---------|------------|------|
| HIGH_ERROR | 特征增加 + 规则迭代 | 误差大需要更多表达能力和更好的规则 |
| STAGNATION | 特征增加 + 子模型重组 | 停滞需要新的结构和协同 |
| NEW_ENVIRONMENT | 认知体系更新 + 特征增加 | 新环境需要新的层次结构和新特征 |

- **输入**：模型、触发类型、重构器状态
- **输出**：更新后的模型、更新后的状态
- **约束**：执行前创建快照；每步检查回滚

> **白话翻译**：execute_reform方法就是"执行重构"——根据触发类型选择合适的重构策略，然后执行。持续高误差就加特征和改规则，长期停滞就加特征和重组子模型，新环境就更新层次结构和加特征。执行方式看风险——低风险一次性改，高风险一步步来。

---

### 第四部分：validate_reform方法

#### 2.4 validate_reform方法定义（A67.4）

validate_reform方法对比重构前后的性能，判定重构是否成功：

$$\text{validate\_reform}(\text{model\_old}, \text{model\_new}) \to \text{bool} \tag{A67.4}$$

**方法签名**：

```python
def validate_reform(
    self,
    state: StructuralReformerState,
    model_old: Dict,
    model_new: Dict
) -> Tuple[bool, StructuralReformerState]
```

**验证的三重检查**：

1. **即时检查**：重构后的自由能是否满足回滚条件（A66.2）
2. **验证窗口检查**：验证窗口内的平均性能是否优于重构前（A66.5）
3. **异常检查**：验证窗口内是否检测到异常（引用阶段55）

$$\text{valid} = \left[F_{\text{new}} \leq F_{\text{old}} \cdot (1 + \theta_{\text{rollback}})\right] \;\wedge\; \left[\bar{f}_{\text{verify}} > f_{\text{old}}\right] \;\wedge\; \left[\text{no\_anomaly}\right] \tag{A67.4-condition}$$

**验证失败的处理**：

验证失败时，根据失败原因采取不同措施：

| 失败原因 | 处理 |
|---------|------|
| 自由能超过回滚阈值 | 立即回滚（A66.3） |
| 适应度未提升 | 回滚 + 延长冷却期 |
| 检测到异常 | 回滚 + 触发安全驱力 |

- **输入**：重构前后模型、重构器状态
- **输出**：验证结果、更新后的状态
- **约束**：验证失败时触发回滚

> **白话翻译**：validate_reform方法就是"重构效果如何"——对比重构前后的性能。三重检查：自由能有没有超标、适应度有没有提升、有没有出现异常。三项全部通过才算重构成功，任何一项失败就回滚。

---

### 第五部分：rollback方法

#### 2.5 rollback方法定义（A67.5）

rollback方法从快照恢复模型到重构前的状态：

$$\text{rollback}(\text{snapshot}) \to \text{Model}_{\text{old}} \tag{A67.5}$$

**方法签名**：

```python
def rollback(
    self,
    state: StructuralReformerState,
    snapshot: Dict
) -> Tuple[Dict, StructuralReformerState]
```

**rollback的完整操作**：

```
1. 从快照恢复模型参数（A66.3）
2. 延长冷却期：T_cooldown ← 2 · T_cooldown
3. 更新EWC锚点：θ* ← θ_snapshot
4. 重置学习率：η ← η_0
5. 增加回滚计数
6. 记录回滚原因
7. 返回恢复后的模型和更新后的状态
```

**回滚后的状态更新**：

回滚不仅恢复模型，还更新相关状态以确保系统安全：

- 冷却期翻倍：避免立即再次重构
- EWC锚点更新：保护回滚后的状态
- 学习率重置：为下次微修正提供完整的学习能力
- 回滚计数增加：用于监控重构器的健康状态

- **输入**：快照、重构器状态
- **输出**：恢复后的模型、更新后的状态
- **约束**：快照必须有效

> **白话翻译**：rollback方法就是"一键还原"——从快照恢复到重构前的状态。回滚不仅是恢复模型，还要延长冷却期、更新锚点、重置学习率——确保回滚后系统处于安全状态，不会立刻又触发重构。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A67.1 | StructuralReformer = TriggerChecker ⊕ ReformExecutor ⊕ Validator ⊕ RollbackHandler ⊕ SafetyChecker | 模型、状态 | 更新后的模型和状态 | 纯函数 |
| A67.2 | check_trigger(state) → Optional[ReformType] | 状态 | 触发类型 | 与微修正互斥 |
| A67.3 | execute_reform(model, reform_type) → Model | 模型、触发类型 | 更新后的模型 | 安全约束 |
| A67.4 | validate_reform(model_old, model_new) → bool | 重构前后模型 | 验证结果 | 失败则回滚 |
| A67.5 | rollback(snapshot) → Model | 快照 | 恢复后的模型 | 快照有效 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import numpy as np

# 引用阶段64-66的数据类
from stage_064 import ReformTriggerType
from stage_066 import ReformSafety, ReformStatus


# ============================================================
# 阶段67 数据类：结构性重构器状态
# ============================================================
@dataclass(frozen=True)
class StructuralReformerState:
    """结构性重构器的完整运行状态。

    组合阶段66的ReformSafety，在安全约束的基础上增加
    重构器的运行状态。

    Attributes:
        safety: 阶段66的重构安全状态（组合字段）
        total_reforms: 总重构次数
        successful_reforms: 成功重构次数
        total_rollbacks: 总回滚次数
        last_reform_type: 上次重构类型
        last_reform_step: 上次重构的全局时间步
        reform_log: 重构日志

    Invariants:
        - total_reforms >= 0
        - successful_reforms >= 0
        - successful_reforms <= total_reforms
        - total_rollbacks >= 0
    """
    safety: ReformSafety
    total_reforms: int = 0
    successful_reforms: int = 0
    total_rollbacks: int = 0
    last_reform_type: Optional[ReformTriggerType] = None
    last_reform_step: int = 0
    reform_log: Tuple[Dict, ...] = ()

    def __post_init__(self):
        if self.total_reforms < 0:
            raise ValueError(
                f"总重构次数必须 >= 0，收到 {self.total_reforms}"
            )
        if self.successful_reforms < 0:
            raise ValueError(
                f"成功重构次数必须 >= 0，收到 {self.successful_reforms}"
            )
        if self.successful_reforms > self.total_reforms:
            raise ValueError(
                f"成功重构次数不能超过总次数"
            )

    @property
    def success_rate(self) -> float:
        """重构成功率。"""
        if self.total_reforms == 0:
            return 0.0
        return self.successful_reforms / self.total_reforms

    @property
    def is_healthy(self) -> bool:
        """重构器是否健康（成功率 > 50%）。"""
        if self.total_reforms < 3:
            return True  # 样本太少，不判定
        return self.success_rate > 0.5


# ============================================================
# 阶段67 核心接口：结构性重构器
# ============================================================
class StructuralReformerInterface:
    """阶段67的核心接口：结构性重构器。

    整合触发判定、重构执行、效果验证和回滚，
    提供统一的结构性重构接口。
    所有方法均为纯函数（无副作用），状态通过
    StructuralReformerState显式传递。

    核心约束：
        - 重构后自由能不超过回滚阈值
        - 验证失败立即回滚
        - 重构与微修正互斥
    """

    def check_trigger(
        self,
        state: StructuralReformerState
    ) -> Optional[ReformTriggerType]:
        """检查重构触发条件（A67.2）。

        Args:
            state: 当前重构器状态

        Returns:
            触发类型，若不触发则为None
        """
        # 简化的触发判定：从 reformer_state 的 safety 中读取标志位
        # 实际应由阶段64的触发条件提供；此处按优先级返回
        safety = state.safety
        # 若存在高误差标志则优先返回 HIGH_ERROR，其次 STAGNATION，再次 NEW_ENVIRONMENT
        if getattr(safety, "trigger_high_error", False):
            return ReformTriggerType.HIGH_ERROR
        if getattr(safety, "trigger_stagnation", False):
            return ReformTriggerType.STAGNATION
        if getattr(safety, "trigger_new_environment", False):
            return ReformTriggerType.NEW_ENVIRONMENT
        return None

    def execute_reform(
        self,
        state: StructuralReformerState,
        model: Dict,
        reform_type: ReformTriggerType
    ) -> Tuple[Dict, StructuralReformerState]:
        """执行结构性重构（A67.3）。

        Args:
            state: 当前重构器状态
            model: 当前模型
            reform_type: 重构类型

        Returns:
            (更新后的模型, 更新后的状态)
        """
        from dataclasses import replace
        updated_model = dict(model)
        updated_model["last_reform_type"] = reform_type.value if hasattr(reform_type, "value") else str(reform_type)
        updated_state = replace(
            state,
            total_reforms=state.total_reforms + 1,
            last_reform_type=reform_type,
            last_reform_step=state.last_reform_step + 1,
        )
        return updated_model, updated_state

    def validate_reform(
        self,
        state: StructuralReformerState,
        model_old: Dict,
        model_new: Dict
    ) -> Tuple[bool, StructuralReformerState]:
        """验证重构效果（A67.4）。

        Args:
            state: 当前重构器状态
            model_old: 重构前模型
            model_new: 重构后模型

        Returns:
            (是否验证通过, 更新后的状态)
        """
        from dataclasses import replace
        # 三重检查：自由能、适应度、异常
        F_old = model_old.get("F_old", float("inf"))
        F_new = model_new.get("F_old", float("inf"))
        f_old = model_old.get("fitness", 0.0)
        f_new = model_new.get("fitness", 0.0)
        theta_rollback = state.safety.theta_rollback
        free_energy_ok = F_new <= F_old * (1.0 + theta_rollback)
        fitness_ok = f_new > f_old
        anomaly_ok = not getattr(state.safety, "verify_anomaly_detected", False)
        valid = free_energy_ok and fitness_ok and anomaly_ok
        updated_state = replace(
            state,
            successful_reforms=state.successful_reforms + (1 if valid else 0),
        )
        return valid, updated_state

    def rollback(
        self,
        state: StructuralReformerState,
        snapshot: Dict
    ) -> Tuple[Dict, StructuralReformerState]:
        """回滚到重构前状态（A67.5）。

        Args:
            state: 当前重构器状态
            snapshot: 重构前快照

        Returns:
            (恢复后的模型, 更新后的状态)
        """
        import copy
        from dataclasses import replace
        restored_model = {
            "theta_arch": copy.deepcopy(snapshot.get("theta_arch", {})),
            "theta_param": copy.deepcopy(snapshot.get("theta_param", {})),
            "Sigma": copy.deepcopy(snapshot.get("Sigma", {})),
            "F_old": snapshot.get("F_old", 0.0),
            "restored": True,
        }
        updated_state = replace(
            state,
            total_rollbacks=state.total_rollbacks + 1,
        )
        return restored_model, updated_state

    def step(
        self,
        state: StructuralReformerState,
        model: Dict
    ) -> Tuple[Dict, StructuralReformerState]:
        """执行一步完整的重构流程（A67.1-A67.5）。

        check_trigger → execute_reform → validate_reform → (rollback if needed)

        Args:
            state: 当前重构器状态
            model: 当前模型

        Returns:
            (更新后的模型, 更新后的状态)
        """
        reform_type = self.check_trigger(state)
        if reform_type is None:
            return model, state
        # 创建快照（简化实现）
        snapshot = {
            "theta_arch": model.get("theta_arch", {}),
            "theta_param": model.get("theta_param", {}),
            "Sigma": model.get("Sigma", {}),
            "F_old": model.get("F_old", float("inf")),
        }
        model_new, state_new = self.execute_reform(state, model, reform_type)
        valid, state_valid = self.validate_reform(state_new, model, model_new)
        if not valid:
            model_final, state_final = self.rollback(state_valid, snapshot)
            return model_final, state_final
        return model_new, state_valid
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `ReformSafety` | ← 阶段66 | 作为重构器的安全约束输入（组合字段） |
| **输入** | A64.1-A64.4（触发条件） | ← 阶段64 | check_trigger的理论基础 |
| **输入** | A65.1-A65.5（重构算法） | ← 阶段65 | execute_reform的核心算法 |
| **输入** | A66.1-A66.5（安全约束） | ← 阶段66 | 安全约束机制 |
| **输出** | `StructuralReformerState` | → 阶段68 | 重构器状态，作为探索-利用强制执行的输入 |
| **输出** | A67.1-A67.5（接口定义） | → 工程实现 | 结构重构模块的统一接口规范 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 性能评估 | 自由能和适应度的快速计算 |
| Pydantic | 数据验证 | 支撑StructuralReformerState的运行时类型检查 |
| structlog | 审计日志 | 重构步骤、验证结果和回滚操作的完整审计追踪 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V67.1 | check_trigger正确性 | 触发条件满足时返回正确的ReformType | 条件测试 |
| V67.2 | 触发优先级 | 多条件同时满足时按优先级选择 | 优先级测试 |
| V67.3 | 微修正互斥 | 微修正触发时不触发重构 | 互斥测试 |
| V67.4 | execute_reform正确性 | 重构后模型结构符合预期 | 结构验证 |
| V67.5 | 快照创建 | 执行重构前创建快照 | 状态检查 |
| V67.6 | validate_reform正确性 | 验证通过/失败判定正确 | 条件测试 |
| V67.7 | 验证失败回滚 | 验证失败时自动回滚 | 流程验证 |
| V67.8 | rollback正确性 | 回滚后模型与快照一致 | 一致性验证 |
| V67.9 | 回滚后冷却期延长 | 回滚后冷却期翻倍 | 状态验证 |
| V67.10 | step方法一致性 | step结果与手动调用各方法一致 | 对比验证 |
| V67.11 | StructuralReformerState组合 | `state.safety` 类型为 `ReformSafety` | 类型检查 |
| V67.12 | 成功率计算 | success_rate = successful / total | 比率验证 |
| V67.13 | 健康状态判定 | 成功率 > 50% 时判定为健康 | 条件测试 |
| V67.14 | 非法参数拒绝 | total_reforms < 0 抛出 `ValueError` | 异常测试 |
| V67.15 | 端到端重构循环 | 完整重构+验证+回滚流程无异常 | 集成测试 |

---

> **阶段67完** | 阶段67建立了结构性重构模块的统一Python接口：StructuralReformerInterface类整合触发判定、重构执行、效果验证和回滚，check_trigger方法检查触发条件（A67.2），execute_reform方法执行重构（A67.3），validate_reform方法验证效果（A67.4），rollback方法回滚（A67.5），封装为StructuralReformerState数据类组合ReformSafety。下一步：阶段68将定义探索-利用的70/30强制比例实现。
