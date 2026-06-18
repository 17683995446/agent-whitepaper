# 阶段66：重构安全约束——迭代限制、性能回滚与渐进式重构

> **阶段编号**：S66 | **所属层**：第七层·进化适应层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 最大迭代次数限制 | "最多改三次"——重构不能无限次尝试，最多3次 |
> | 性能回滚条件 | "改坏了就撤"——重构后性能倒退就自动回滚 |
> | 回滚机制 | "一键还原"——恢复到重构前的快照 |
> | 渐进式重构 | "一次只改一个"——不一次性大改，而是一步步来 |
> | 验证窗口 | "改完先看看"——重构后观察一段时间再确认 |
> | ReformSafety | "重构安全的完整画像"——组合StructuralReformer，在重构器基础上增加安全约束 |

---

## 1. 第一性原理依据

本阶段从阶段65（结构性重构算法）和阶段8（适应度单调递增定理）的核心结论出发，建立重构的安全约束机制。核心依据为：

- **定理8.1（适应度单调递增）**（引用阶段8）：$\mathbb{E}_{p_{k+1}}[f(\theta)] \geq \mathbb{E}_{p_k}[f(\theta)]$，种群平均适应度单调递增。本阶段将定理8.1从种群级推广到单体重构——每次重构后的适应度必须不低于重构前，否则回滚。

- **A62.5（微修正收敛保证）**（引用阶段62）：自由能单调不增的收敛保证。本阶段将类似的收敛保证推广到结构重构——重构后的自由能不能超过回滚阈值。

- **A64.4（重构风险评估）**（引用阶段64）：risk = f(complexity, uncertainty, resource_cost)。安全约束的严格程度与风险评估挂钩——高风险重构需要更严格的安全约束。

- **A65.1-A65.5（结构性重构算法）**（引用阶段65）：特征增删、子模型重组、规则迭代、认知体系更新和贝叶斯搜索。安全约束是这些重构算法的"安全带"——确保重构不会导致性能倒退。

本阶段的核心目标是：**定义最大迭代次数限制 max_iterations = 3（A66.1），性能回滚条件 $F_{\text{new}} > F_{\text{old}} \cdot (1 + \theta_{\text{rollback}})$（A66.2），回滚机制 restore(snapshot)（A66.3），渐进式重构（一次只改一个组件）（A66.4），重构后的验证窗口 $T_{\text{verify}}$（A66.5），封装为ReformSafety数据类，组合StructuralReformer作为重构器的输入。**

> **去重声明**：本阶段仅定义重构的安全约束机制，不重新定义重构算法（阶段65）或微修正的安全检查（阶段63）。重构安全约束是阶段65重构算法的"安全层"，确保重构不会导致灾难性后果。

---

## 2. 核心数学公式

### 第一部分：最大迭代次数限制

#### 2.1 最大迭代次数限制定义（A66.1）

单次重构触发最多执行3次迭代尝试：

$$N_{\text{iteration}} \leq \text{max\_iterations} = 3 \tag{A66.1}$$

**迭代次数限制的必要性**：

如果没有迭代次数限制，重构可能陷入无限循环——每次重构后性能不满足要求，又触发新的重构，如此反复。迭代次数限制确保重构有明确的终止条件：

| 迭代次数 | 状态 | 动作 |
|---------|------|------|
| 1 | 首次尝试 | 执行重构，验证性能 |
| 2 | 首次失败 | 尝试不同的重构方案 |
| 3 | 二次失败 | 最后一次尝试 |
| >3 | 三次失败 | 放弃重构，维持原模型 |

**迭代次数与风险的关系**：

高风险重构（A64.4 risk ≥ 0.7）的迭代次数应更少：

$$\text{max\_iterations}(\text{risk}) = \begin{cases} 3 & \text{risk} < 0.3 \\ 2 & 0.3 \leq \text{risk} < 0.7 \\ 1 & \text{risk} \geq 0.7 \end{cases} \tag{A66.1-risk}$$

- **输入**：风险评估值
- **输出**：最大迭代次数
- **约束**：max_iterations ≥ 1

> **白话翻译**：最大迭代次数限制就是"最多改三次"——重构不能无限次尝试，最多3次。风险越高，允许尝试的次数越少——高风险只允许试1次，低风险可以试3次。三次都改不好就放弃，维持原状比改坏了强。

---

### 第二部分：性能回滚条件

#### 2.2 性能回滚条件定义（A66.2）

当重构后的自由能超过回滚阈值时，自动回滚到重构前的状态：

$$\text{should\_rollback} = \mathbb{1}\left[F_{\text{new}} > F_{\text{old}} \cdot (1 + \theta_{\text{rollback}})\right] \tag{A66.2}$$

其中：
- $F_{\text{new}}$ 为重构后的自由能
- $F_{\text{old}}$ 为重构前的自由能
- $\theta_{\text{rollback}} > 0$ 为回滚容差（通常设为0.05，即允许5%的性能下降）

**回滚容差的物理含义**：

$\theta_{\text{rollback}}$ 允许重构后有轻微的性能下降——这是因为重构后的模型可能需要短暂的适应期（参数未充分优化），轻微的性能下降是可接受的。但超过容差的性能下降说明重构方向错误，必须回滚。

**回滚容差与风险的关系**：

高风险重构的回滚容差应更严格：

$$\theta_{\text{rollback}}(\text{risk}) = \theta_{\text{base}} \cdot (1 - \text{risk}) \tag{A66.2-risk}$$

风险越高，允许的性能下降越小——高风险重构几乎没有容错空间。

**回滚条件的层次**：

| 性能变化 | 判定 | 动作 |
|---------|------|------|
| $F_{\text{new}} < F_{\text{old}}$ | 性能提升 | 接受重构 |
| $F_{\text{old}} \leq F_{\text{new}} \leq F_{\text{old}}(1 + \theta_{\text{rollback}})$ | 轻微下降 | 进入验证窗口（A66.5） |
| $F_{\text{new}} > F_{\text{old}}(1 + \theta_{\text{rollback}})$ | 显著下降 | 立即回滚 |

- **输入**：重构前后自由能，回滚容差
- **输出**：回滚标志
- **约束**：$\theta_{\text{rollback}} > 0$

> **白话翻译**：性能回滚条件就是"改坏了就撤"——重构后如果性能显著下降（超过容差），立刻回滚到重构前的状态。容差允许轻微下降（5%以内），因为新结构可能需要短暂适应。但超过容差就说明重构方向错了，必须立刻撤回。

---

### 第三部分：回滚机制

#### 2.3 回滚机制定义（A66.3）

回滚机制在重构前保存模型快照，回滚时恢复快照：

$$\text{restore}(\text{snapshot}) \to \text{Model}_{\text{old}} \tag{A66.3}$$

**快照的内容**：

快照保存重构前的完整模型状态：

$$\text{snapshot} = \{\theta_{\text{arch}}, \theta_{\text{param}}, \Sigma, F_{\text{old}}, t_{\text{snapshot}}\} \tag{A66.3-snapshot}$$

包括架构参数、权重参数、后验协方差、自由能值和时间戳。

**快照的存储策略**：

| 策略 | 存储 | 优点 | 缺点 |
|------|------|------|------|
| 完整快照 | 所有参数 | 回滚完整 | 内存开销大 |
| 增量快照 | 仅变化部分 | 内存开销小 | 回滚需重建 |
| 压缩快照 | 压缩后参数 | 折中 | 压缩/解压开销 |

本阶段选择完整快照策略，原因：
1. 回滚的完整性保证——不丢失任何状态
2. 实现简单——直接复制整个模型
3. 重构频率低——快照的内存开销可接受

**回滚后的状态恢复**：

回滚不仅恢复模型参数，还恢复相关状态：

$$\text{rollback} = \text{restore}(\text{snapshot}) + \text{reset}(\text{cooldown}) + \text{update}(\text{EWC\_anchor}) \tag{A66.3-rollback}$$

1. 恢复模型参数（从快照）
2. 重置冷却期（延长冷却期，避免立即再次重构）
3. 更新EWC锚点（将锚点设为回滚后的参数，保护当前状态）

- **输入**：重构前快照
- **输出**：恢复后的模型
- **约束**：快照在重构前创建

> **白话翻译**：回滚机制就是"一键还原"——重构前先拍快照，改坏了就恢复快照。快照保存完整的模型状态，回滚时一键恢复到重构前。回滚后还要延长冷却期，避免刚回滚又立刻重构——给系统一个喘息的机会。

---

### 第四部分：渐进式重构

#### 2.4 渐进式重构定义（A66.4）

渐进式重构一次只修改一个组件，避免同时修改多个组件导致不可预测的后果：

$$\text{reform\_step}_k = \text{apply\_one\_operation}(\text{reform\_plan}[k]) \tag{A66.4}$$

**渐进式重构的流程**：

```
1. 生成重构计划：reform_plan = [op_1, op_2, ..., op_K]
2. 保存快照
3. for k = 1 to K:
   a. 执行第k个操作：reform_step_k
   b. 评估性能：F_new
   c. 检查回滚条件：F_new > F_old * (1 + θ_rollback)?
   d. 若需回滚：restore(snapshot)，终止重构
   e. 更新快照：snapshot = current_state
4. 进入验证窗口
```

**渐进式重构的优势**：

1. **可归因性**：如果某步导致性能下降，可以精确定位是哪个操作出了问题
2. **安全性**：每步都检查性能，及时发现并回滚
3. **可调试性**：每步的结果都可以独立验证

**渐进式重构与一次性重构的比较**：

| 维度 | 渐进式重构 | 一次性重构 |
|------|----------|----------|
| 安全性 | 高（每步检查） | 低（整体检查） |
| 效率 | 低（多次评估） | 高（一次评估） |
| 可归因性 | 高（精确定位） | 低（无法定位） |
| 适用场景 | 高风险重构 | 低风险重构 |

- **输入**：重构计划（操作列表）
- **输出**：逐步重构后的模型
- **约束**：每步后检查回滚条件

> **白话翻译**：渐进式重构就是"一次只改一个"——不一次性大改，而是一步步来。每改一个组件就检查一次性能，改坏了立刻回滚。好处是能精确定位问题——如果改到第三个组件时性能下降，就知道是第三个组件的问题。坏处是效率低——每次都要评估性能。高风险重构必须用渐进式，低风险可以一次性改。

---

### 第五部分：重构后的验证窗口

#### 2.5 验证窗口定义（A66.5）

重构完成后，需要观察一段时间（验证窗口）才能确认重构效果：

$$T_{\text{verify}} = \max\left(T_{\text{base}}, \frac{T_{\text{stagnation}}}{5}\right) \tag{A66.5}$$

其中 $T_{\text{base}}$ 为基础验证时间（通常50步），$T_{\text{stagnation}}$ 为停滞判定时间（引用A64.2）。

**验证窗口内的监控指标**：

| 指标 | 阈值 | 不满足时的动作 |
|------|------|--------------|
| 自由能 | $F_t \leq F_{\text{old}} \cdot (1 + \theta_{\text{rollback}})$ | 回滚 |
| 适应度 | $f_t \geq f_{\text{old}}$ | 回滚 |
| 误差率 | 异常检测无异常 | 警告 |
| 资源消耗 | 不超过预算 | 警告 |

**验证窗口的确认条件**：

验证窗口结束时，如果以下条件全部满足，则确认重构成功：

$$\text{reform\_confirmed} = \mathbb{1}\left[\bar{F}_{\text{verify}} < F_{\text{old}} \;\wedge\; \bar{f}_{\text{verify}} > f_{\text{old}} \;\wedge\; \text{no\_anomaly}\right] \tag{A66.5-confirm}$$

其中 $\bar{F}_{\text{verify}}$ 和 $\bar{f}_{\text{verify}}$ 为验证窗口内的平均自由能和平均适应度。

- **输入**：验证窗口长度，监控指标
- **输出**：重构确认标志
- **约束**：$T_{\text{verify}} \geq T_{\text{base}}$

> **白话翻译**：验证窗口就是"改完先看看"——重构完成后不能立刻宣布成功，要观察一段时间确认效果。验证窗口内持续监控自由能、适应度和异常情况，全部正常才确认重构成功。如果验证窗口内出现问题，仍然可以回滚——安全网一直都在。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A66.1 | $N_{\text{iteration}} \leq \text{max\_iterations} = 3$ | 风险评估值 | 最大迭代次数 | max_iterations ≥ 1 |
| A66.2 | $\text{rollback} = \mathbb{1}[F_{\text{new}} > F_{\text{old}} \cdot (1 + \theta_{\text{rollback}})]$ | 重构前后自由能、容差 | 回滚标志 | $\theta_{\text{rollback}} > 0$ |
| A66.3 | restore(snapshot) → Model_old | 快照 | 恢复后的模型 | 快照在重构前创建 |
| A66.4 | reform_step_k = apply_one_operation(plan[k]) | 重构计划 | 逐步重构结果 | 每步检查回滚 |
| A66.5 | $T_{\text{verify}} = \max(T_{\text{base}}, T_{\text{stagnation}}/5)$ | 基础时间、停滞时间 | 验证窗口长度 | $T_{\text{verify}} \geq T_{\text{base}}$ |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from enum import Enum
import numpy as np

# 引用阶段65的数据类
from stage_065 import StructuralReformer


# ============================================================
# 阶段66 枚举：重构状态
# ============================================================
class ReformStatus(Enum):
    """重构执行状态。"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"
    VERIFIED = "verified"
    FAILED = "failed"


# ============================================================
# 阶段66 数据类：重构安全
# ============================================================
@dataclass(frozen=True)
class ReformSafety:
    """重构安全约束的完整描述。

    组合阶段65的StructuralReformer，在重构器的基础上增加
    安全约束的状态——迭代计数、快照、回滚状态、验证状态。

    Attributes:
        reformer: 阶段65的结构性重构器（组合字段）
        max_iterations: 最大迭代次数
        current_iteration: 当前迭代次数
        theta_rollback: 回滚容差
        snapshot: 重构前快照
        reform_status: 重构执行状态
        T_verify: 验证窗口长度
        verify_steps_elapsed: 验证窗口已过步数
        verify_free_energy_avg: 验证窗口内平均自由能
        verify_fitness_avg: 验证窗口内平均适应度
        verify_anomaly_detected: 验证窗口内是否检测到异常
        rollback_count: 总回滚次数
        is_progressive: 是否渐进式重构
        progressive_step: 渐进式重构当前步骤

    Invariants:
        - max_iterations >= 1
        - current_iteration >= 0
        - current_iteration <= max_iterations
        - theta_rollback > 0
        - T_verify > 0
        - rollback_count >= 0
    """
    reformer: StructuralReformer
    max_iterations: int = 3
    current_iteration: int = 0
    theta_rollback: float = 0.05
    snapshot: Optional[Dict] = None
    reform_status: ReformStatus = ReformStatus.NOT_STARTED
    T_verify: int = 50
    verify_steps_elapsed: int = 0
    verify_free_energy_avg: float = 0.0
    verify_fitness_avg: float = 0.0
    verify_anomaly_detected: bool = False
    rollback_count: int = 0
    is_progressive: bool = True
    progressive_step: int = 0

    def __post_init__(self):
        if self.max_iterations < 1:
            raise ValueError(
                f"最大迭代次数必须 >= 1，收到 {self.max_iterations}"
            )
        if self.current_iteration < 0:
            raise ValueError(
                f"当前迭代次数必须 >= 0，收到 {self.current_iteration}"
            )
        if self.current_iteration > self.max_iterations:
            raise ValueError(
                f"当前迭代次数不能超过最大值，"
                f"收到 current={self.current_iteration}, "
                f"max={self.max_iterations}"
            )
        if self.theta_rollback <= 0:
            raise ValueError(
                f"回滚容差必须 > 0，收到 {self.theta_rollback}"
            )
        if self.T_verify <= 0:
            raise ValueError(
                f"验证窗口长度必须 > 0，收到 {self.T_verify}"
            )

    @property
    def iterations_remaining(self) -> int:
        """剩余迭代次数。"""
        return self.max_iterations - self.current_iteration

    @property
    def is_in_verification(self) -> bool:
        """是否在验证窗口中。"""
        return (
            self.reform_status == ReformStatus.COMPLETED
            and self.verify_steps_elapsed < self.T_verify
        )

    @property
    def is_reform_confirmed(self) -> bool:
        """重构是否已确认成功。"""
        return self.reform_status == ReformStatus.VERIFIED


# ============================================================
# 阶段66 核心接口：重构安全约束
# ============================================================
class ReformSafetyChecker:
    """阶段66的核心接口：重构安全约束。

    实现迭代限制、性能回滚、回滚机制、渐进式重构
    和验证窗口。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 重构后自由能不超过回滚阈值
        - 迭代次数不超过最大值
        - 验证窗口内持续监控
    """

    def check_iteration_limit(
        self,
        safety: ReformSafety
    ) -> bool:
        """检查迭代次数是否超限（A66.1）。

        Args:
            safety: 当前安全状态

        Returns:
            是否仍在迭代限制内
        """
        return safety.current_iteration < safety.max_iterations

    def check_rollback_condition(
        self,
        F_new: float,
        F_old: float,
        theta_rollback: float
    ) -> bool:
        """检查性能回滚条件（A66.2）。

        F_new > F_old * (1 + θ_rollback) → 回滚

        Args:
            F_new: 重构后自由能
            F_old: 重构前自由能
            theta_rollback: 回滚容差

        Returns:
            是否需要回滚
        """
        return F_new > F_old * (1.0 + theta_rollback)

    def create_snapshot(
        self,
        model: Dict
    ) -> Dict:
        """创建模型快照（A66.3前置）。

        Args:
            model: 当前模型

        Returns:
            模型快照
        """
        import copy, time
        return {
            "theta_arch": copy.deepcopy(model.get("theta_arch", {})),
            "theta_param": copy.deepcopy(model.get("theta_param", {})),
            "Sigma": copy.deepcopy(model.get("Sigma", {})),
            "F_old": model.get("F_old", 0.0),
            "t_snapshot": time.time(),
        }

    def restore_snapshot(
        self,
        snapshot: Dict
    ) -> Dict:
        """从快照恢复模型（A66.3）。

        Args:
            snapshot: 模型快照

        Returns:
            恢复后的模型
        """
        import copy
        return {
            "theta_arch": copy.deepcopy(snapshot.get("theta_arch", {})),
            "theta_param": copy.deepcopy(snapshot.get("theta_param", {})),
            "Sigma": copy.deepcopy(snapshot.get("Sigma", {})),
            "F_old": snapshot.get("F_old", 0.0),
            "restored_from": snapshot.get("t_snapshot", None),
        }

    def execute_progressive_step(
        self,
        safety: ReformSafety,
        model: Dict,
        operation_index: int
    ) -> Tuple[Dict, ReformSafety]:
        """执行渐进式重构的一步（A66.4）。

        Args:
            safety: 当前安全状态
            model: 当前模型
            operation_index: 操作索引

        Returns:
            (更新后的模型, 更新后的安全状态)
        """
        # 标记重构进行中
        from dataclasses import replace
        updated_safety = replace(
            safety,
            reform_status=ReformStatus.IN_PROGRESS,
            progressive_step=operation_index + 1,
        )
        # 模拟执行一步操作：此处仅对模型做浅层标记，实际应由调用方注入具体操作
        updated_model = dict(model)
        updated_model["_progressive_step"] = operation_index + 1
        return updated_model, updated_safety

    def check_verification(
        self,
        safety: ReformSafety,
        F_old: float,
        f_old: float
    ) -> Tuple[bool, ReformSafety]:
        """检查验证窗口条件（A66.5）。

        Args:
            safety: 当前安全状态
            F_old: 重构前自由能
            f_old: 重构前适应度

        Returns:
            (是否确认成功, 更新后的安全状态)
        """
        from dataclasses import replace
        confirmed = (
            safety.verify_free_energy_avg < F_old
            and safety.verify_fitness_avg > f_old
            and not safety.verify_anomaly_detected
        )
        new_status = ReformStatus.VERIFIED if confirmed else ReformStatus.FAILED
        updated_safety = replace(
            safety,
            reform_status=new_status,
        )
        return confirmed, updated_safety
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `StructuralReformer` | ← 阶段65 | 作为安全约束的重构器输入（组合字段） |
| **输入** | 定理8.1（适应度单调递增） | ← 阶段8 | 回滚条件的理论基础 |
| **输入** | A64.4（重构风险评估） | ← 阶段64 | 迭代次数与风险的关系 |
| **输出** | `ReformSafety` | → 阶段67 | 安全约束状态，作为Python接口的输入 |
| **输出** | A66.1-A66.5（安全约束定义） | → 阶段67 | 重构安全约束的核心定义 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 性能评估 | 自由能和适应度的快速计算 |
| Pydantic | 快照验证 | 快照数据的运行时类型检查 |
| structlog | 审计日志 | 重构步骤和回滚操作的完整审计追踪 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V66.1 | 迭代次数限制 | current_iteration ≤ max_iterations | 计数验证 |
| V66.2 | 风险与迭代次数关系 | 高风险时 max_iterations 更小 | 条件测试 |
| V66.3 | 回滚条件正确性 | $F_{\text{new}} > F_{\text{old}}(1 + \theta)$ 时触发回滚 | 条件测试 |
| V66.4 | 回滚容差与风险关系 | 高风险时容差更小 | 条件测试 |
| V66.5 | 快照创建完整性 | 快照包含所有必要状态 | 字段检查 |
| V66.6 | 快照恢复正确性 | 恢复后模型与快照一致 | 一致性验证 |
| V66.7 | 渐进式重构安全性 | 每步后检查回滚条件 | 流程验证 |
| V66.8 | 渐进式回滚定位 | 回滚时精确定位问题步骤 | 归因测试 |
| V66.9 | 验证窗口长度 | $T_{\text{verify}} \geq T_{\text{base}}$ | 范围检查 |
| V66.10 | 验证确认条件 | 三条件同时满足时确认 | 条件测试 |
| V66.11 | ReformSafety组合 | `safety.reformer` 类型为 `StructuralReformer` | 类型检查 |
| V66.12 | 非法参数拒绝 | max_iterations < 1 抛出 `ValueError` | 异常测试 |
| V66.13 | 回滚后冷却期延长 | 回滚后冷却期翻倍 | 状态验证 |
| V66.14 | 端到端安全重构 | 完整重构+回滚流程无异常 | 集成测试 |

---

> **阶段66完** | 阶段66建立了重构安全约束的完整机制：最大迭代次数限制 max_iterations = 3（A66.1），性能回滚条件 $F_{\text{new}} > F_{\text{old}} \cdot (1 + \theta_{\text{rollback}})$（A66.2），回滚机制 restore(snapshot)（A66.3），渐进式重构（A66.4），验证窗口 $T_{\text{verify}}$（A66.5），封装为ReformSafety数据类组合StructuralReformer。下一步：阶段67将定义结构性重构的Python接口。
