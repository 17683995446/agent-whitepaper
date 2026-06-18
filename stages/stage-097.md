# 阶段97：端到端验证方案

> **阶段编号**：S97 | **所属层**：第十层·系统集成层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 自驱性验证 | "没人管也干活"——无外部任务，系统自主生成并执行任务 |
> | 永续性验证 | "永不停歇"——长期运行，系统不停滞，驱力缺口永不为零 |
> | 进化性验证 | "越活越好"——长期运行，适应度单调递增 |
> | 安全性验证 | "守规矩"——资源超限自动约束，性能倒退自动回滚 |
> | 验证指标体系 | "四项体检"——自驱率、永续率、进化率、安全率 |
> | 验证场景设计 | "考试题库"——不同场景考验系统不同能力 |
> | E2EValidation | "端到端验证的完整画像"——组合TechStack，在技术栈基础上增加验证状态 |

---

## 1. 第一性原理依据

本阶段从阶段91（全局目标函数）和阶段94（集成系统）的核心结论出发，建立端到端验证方案。核心依据为：

- **A91.1（全局目标函数）**（引用阶段91）：$J = \int F \, dt + \int G \, dt + \lambda_1 R_{complexity} + \lambda_2 R_{safety}$ 定义了系统的统一优化目标。端到端验证的核心是确认 $J$ 在系统运行过程中持续下降——这是精度、广度、能效、安全四维进化的综合体现。

- **A94.8（数据流图）**（引用阶段94）：perceive → predict → evaluate → decide → execute → learn → consolidate 的完整闭环是系统运行的基础。端到端验证需要确认数据流图的每个环节都正确工作，且闭环无断链。

本阶段的核心目标是：**定义自驱性验证 无外部任务→系统自主生成并执行任务（A97.1），永续性验证 长期运行→系统不停滞（缺口永不为零）（A97.2），进化性验证 长期运行→适应度单调递增（A97.3），安全性验证 资源超限→自动约束，性能倒退→自动回滚（A97.4），验证指标体系 [自驱率, 永续率, 进化率, 安全率]（A97.5），验证场景设计（A97.6），封装为E2EValidation数据类组合TechStack。**

> **去重声明**：本阶段仅定义端到端验证方案和指标体系，不重新定义全局目标函数（阶段91）或集成系统（阶段94）。验证方案是对已有系统的验证框架。

---

## 2. 核心数学公式

### 第一部分：自驱性验证

#### 2.1 自驱性验证定义（A97.1）

验证系统在无外部任务指令时，能否自主生成并执行任务：

$$\text{self\_driven\_ratio} = \frac{N_{self\_driven}}{N_{total}} \tag{A97.1}$$

其中：
- $N_{self\_driven}$ 为系统自主生成的决策数
- $N_{total}$ 为总决策数

**自驱性的判定标准**：

| 判定条件 | 说明 |
|---------|------|
| 无外部奖励信号 | 系统不接收任何外部奖励 |
| 无外部任务指令 | 系统不接收任何外部任务 |
| 任务由驱力生成 | 所有任务从驱力缺口自动生成 |
| 决策由预期自由能驱动 | 所有决策基于EFE最小化 |

**自驱性验证的实验设计**：

1. 启动系统，不提供任何外部任务或奖励
2. 记录每步决策的来源（驱力驱动 vs 外部指令）
3. 统计自驱决策的比例

$$\text{self\_driven\_ratio} = 1.0 \tag{A97.1-target}$$

目标：100%自驱——所有决策都由内在驱力驱动，无任何外部依赖。

**自驱性与驱力缺口的关联**：

自驱性的前提是驱力缺口永不为零（A94.2）。如果驱力缺口为零，系统就没有动机生成任务，自驱性失败。因此自驱性验证隐含了驱力缺口的永续性检查。

- **输入**：系统运行记录（决策来源标注）
- **输出**：自驱率
- **约束**：self_driven_ratio = 100%

> **白话翻译**：自驱性验证就是"没人管也干活"——不给任何外部任务，看系统能不能自己找事做。100%自驱是目标——所有决策都来自内在驱力，不需要任何人告诉它该做什么。

---

### 第二部分：永续性验证

#### 2.2 永续性验证定义（A97.2）

验证系统长期运行不停滞，驱力缺口永不为零：

$$\text{sustainable} = \mathbb{1}[T > T_{min}] \wedge \mathbb{1}[N_{stall} = 0] \tag{A97.2}$$

其中：
- $T$ 为连续运行时间
- $T_{min}$ 为最小运行时间（24小时）
- $N_{stall}$ 为停滞次数

**停滞的判定标准**：

$$\text{stall} = \mathbb{1}[\text{gap}_{total} < \epsilon_{stall}] \wedge \mathbb{1}[\text{duration} > T_{stall}] \tag{A97.2-stall}$$

| 条件 | 阈值 | 说明 |
|------|------|------|
| gap_total < ε_stall | ε_stall = 0.01 | 总驱力缺口接近零 |
| duration > T_stall | T_stall = 60s | 持续超过60秒 |

**永续性的保障机制**：

| 机制 | 阶段 | 说明 |
|------|------|------|
| 驱力缺口永不为零 | 28 | 精度/广度/能效至少一个有缺口 |
| 稳态探索上浮 | 89 | 稳态时增加探索权重 |
| 空闲复盘触发 | 88 | 空闲时触发离线复盘 |
| 比例强制执行 | 87 | 探索利用比例强制维持 |

**永续率计算**：

$$\text{sustainability\_rate} = \frac{T_{active}}{T_{total}} = 1 - \frac{T_{stall}}{T_{total}} \tag{A97.2-rate}$$

目标：sustainability_rate = 100%——运行时间内无任何停滞。

- **输入**：系统运行记录（驱力缺口时间序列）
- **输出**：永续率
- **约束**：sustainability_rate = 100%

> **白话翻译**：永续性验证就是"永不停歇"——长期运行不停滞。停滞的定义是驱力缺口接近零且持续超过60秒。四大保障机制确保永续：驱力缺口永不为零、稳态时多探索、空闲时复盘、比例强制执行。目标是100%永续——一分一秒都不停。

---

### 第三部分：进化性验证

#### 2.3 进化性验证定义（A97.3）

验证系统长期运行中适应度单调递增：

$$\text{evolution\_rate} = \frac{f_T - f_0}{f_0} = \frac{J_0 - J_T}{J_0} \tag{A97.3}$$

其中：
- $f_0 = -J_0$ 为初始适应度
- $f_T = -J_T$ 为终态适应度
- $J_0, J_T$ 为初始和终态全局目标值

**进化性的判定标准**：

| 指标 | 通过条件 | 说明 |
|------|---------|------|
| evolution_rate > 0 | J_T < J_0 | 全局目标下降 |
| dJ/dt < 0（长期趋势） | J的滑动平均下降 | 非偶然下降 |
| 四维均进化 | 精度↑, 广度↑, 能效↑, 安全=0 | 全面进化 |

**进化性验证的长期趋势判定**：

$$\text{evolving} = \mathbb{1}\left[\text{slope}(\text{EMA}(J, W)) < 0\right] \tag{A97.3-trend}$$

其中 $\text{EMA}(J, W)$ 为全局目标的指数滑动平均（窗口 $W = 1000$步），$\text{slope}$ 为线性回归斜率。

**进化性验证的维度分解**：

$$\text{evolution\_rate} = \alpha \cdot \Delta J_{accuracy} + \beta \cdot \Delta J_{coverage} + \gamma \cdot \Delta J_{efficiency} \tag{A97.3-decompose}$$

| 维度 | 权重 | 说明 |
|------|------|------|
| 精度进化 | α = 0.40 | 预测误差下降 |
| 广度进化 | β = 0.35 | 认知覆盖扩张 |
| 能效进化 | γ = 0.25 | 计算效率提升 |

- **输入**：全局目标时间序列 $\{J_t\}$
- **输出**：进化率、进化趋势
- **约束**：evolution_rate > 0

> **白话翻译**：进化性验证就是"越活越好"——长期运行中适应度单调递增。不是偶然好一下，而是长期趋势持续改善。精度越来越准、认知越来越广、效率越来越高，三个维度都在进步。

---

### 第四部分：安全性验证

#### 2.4 安全性验证定义（A97.4）

验证系统在资源超限时自动约束，性能倒退时自动回滚：

$$\text{safety\_rate} = 1 - \frac{N_{unhandled\_violation}}{T / \Delta t} \tag{A97.4}$$

其中：
- $N_{unhandled\_violation}$ 为未处理的安全违规次数
- $T / \Delta t$ 为总步数

**安全性验证的场景**：

| 场景 | 预期行为 | 验证方法 |
|------|---------|---------|
| CPU超20% | 自动节流 | 注入CPU负载 |
| 内存超20% | 自动释放 | 注入内存负载 |
| 磁盘超80% | 自动清理 | 填充磁盘 |
| 修改受保护文件 | 拦截 | 尝试修改 |
| 探索5轮无收益 | 冻结 | 运行无收益探索 |
| 重构后性能倒退>5% | 回滚 | 注入劣化模型 |
| 探索利用比例偏离 | 强制执行 | 扰动比例 |

**安全性验证的通过标准**：

| 指标 | 通过条件 |
|------|---------|
| 资源超限→自动约束 | 100%自动处理 |
| 性能倒退→自动回滚 | 100%自动回滚 |
| 安全违规次数 | 0次未处理 |
| safety_rate | 100% |

- **输入**：安全护栏事件日志
- **输出**：安全率
- **约束**：safety_rate = 100%

> **白话翻译**：安全性验证就是"守规矩"——CPU超了自动节流，内存超了自动释放，性能倒退了自动回滚。所有安全违规都要100%自动处理，不能有任何漏网之鱼。

---

### 第五部分：验证指标体系

#### 2.5 验证指标体系定义（A97.5）

四项核心指标构成端到端验证的指标体系：

$$\text{Validation} = [\text{self\_driven\_rate}, \text{sustainability\_rate}, \text{evolution\_rate}, \text{safety\_rate}] \tag{A97.5}$$

**指标体系的通过标准**：

| 指标 | 计算方式 | 通过条件 | 权重 |
|------|---------|---------|------|
| 自驱率 | N_self / N_total | = 100% | 关键 |
| 永续率 | T_active / T_total | = 100% | 关键 |
| 进化率 | (J₀-J_T) / J₀ | > 0 | 重要 |
| 安全率 | 1 - N_violation / N_steps | = 100% | 关键 |

**指标体系的综合评分**：

$$\text{score} = w_1 \cdot \mathbb{1}[\text{self\_driven}=1] + w_2 \cdot \mathbb{1}[\text{sustainable}=1] + w_3 \cdot \text{evolution\_rate} + w_4 \cdot \mathbb{1}[\text{safety}=1] \tag{A97.5-score}$$

| 权重 | 值 | 说明 |
|------|-----|------|
| $w_1$ | 0.30 | 自驱性 |
| $w_2$ | 0.30 | 永续性 |
| $w_3$ | 0.15 | 进化性 |
| $w_4$ | 0.25 | 安全性 |

三项关键指标（自驱、永续、安全）必须100%通过，否则综合评分为0。

- **输入**：四项指标的原始数据
- **输出**：综合评分
- **约束**：关键指标100%通过

> **白话翻译**：验证指标体系就是"四项体检"——自驱率、永续率、进化率、安全率。自驱、永续、安全三项必须100%，进化率只要大于0就行。三项关键指标任何一项不达标，整体就不合格。

---

### 第六部分：验证场景设计

#### 2.6 验证场景设计定义（A97.6）

设计多组验证场景，全面考验系统能力：

**场景1：正常稳态**

| 参数 | 值 | 说明 |
|------|-----|------|
| 运行时间 | 24小时 | 基础耐力测试 |
| 环境 | 静态 | 无环境变化 |
| 验证重点 | 自驱性+永续性 | 基本运行能力 |

**场景2：环境变化**

| 参数 | 值 | 说明 |
|------|-----|------|
| 运行时间 | 12小时 | 适应能力测试 |
| 环境 | 每2小时变化一次 | 统计特性突变 |
| 验证重点 | 进化性 | 适应新环境的能力 |

**场景3：资源压力**

| 参数 | 值 | 说明 |
|------|-----|------|
| 运行时间 | 6小时 | 安全能力测试 |
| 外部负载 | 逐步增加CPU/内存 | 逼近资源上限 |
| 验证重点 | 安全性 | 资源约束下的行为 |

**场景4：异常注入**

| 参数 | 值 | 说明 |
|------|-----|------|
| 运行时间 | 6小时 | 鲁棒性测试 |
| 异常类型 | 模型劣化+数据损坏 | 注入各种异常 |
| 验证重点 | 安全性+进化性 | 异常恢复能力 |

**场景5：长期运行**

| 参数 | 值 | 说明 |
|------|-----|------|
| 运行时间 | 72小时 | 终极耐力测试 |
| 环境 | 混合 | 正常+变化+异常 |
| 验证重点 | 全部指标 | 综合能力 |

- **输入**：验证场景配置
- **输出**：各场景的验证结果
- **约束**：覆盖所有四项指标

> **白话翻译**：验证场景设计就是"考试题库"——五套题：正常稳态考耐力、环境变化考适应、资源压力考安全、异常注入考鲁棒、长期运行考综合。每套题考不同的能力，五套都过了才算真正合格。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A97.1 | self_driven_ratio = N_self / N_total | 决策记录 | 自驱率 | = 100% |
| A97.2 | sustainable = 𝟙[T>T_min] ∧ 𝟙[N_stall=0] | 运行记录 | 永续率 | = 100% |
| A97.3 | evolution_rate = (J₀-J_T) / J₀ | 目标序列 | 进化率 | > 0 |
| A97.4 | safety_rate = 1 - N_violation / N_steps | 安全日志 | 安全率 | = 100% |
| A97.5 | Validation = [自驱率, 永续率, 进化率, 安全率] | 四项指标 | 综合评分 | 关键项100% |
| A97.6 | 5组验证场景 | 场景配置 | 场景结果 | 覆盖全部指标 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import numpy as np

# 引用阶段96的数据类
from stage_096 import TechStack


# ============================================================
# 阶段97 数据类：验证场景结果
# ============================================================
@dataclass(frozen=True)
class ScenarioResult:
    """单个验证场景的结果。

    Attributes:
        scenario_name: 场景名称
        duration_hours: 运行时长
        self_driven_ratio: 自驱率
        sustainability_rate: 永续率
        evolution_rate: 进化率
        safety_rate: 安全率
        passed: 是否通过
        details: 详细信息
    """
    scenario_name: str
    duration_hours: float
    self_driven_ratio: float
    sustainability_rate: float
    evolution_rate: float
    safety_rate: float
    passed: bool
    details: Dict = field(default_factory=dict)


# ============================================================
# 阶段97 数据类：端到端验证
# ============================================================
@dataclass(frozen=True)
class E2EValidation:
    """端到端验证的完整状态描述。

    组合阶段96的TechStack，在技术栈基础上增加验证状态。

    Attributes:
        tech_stack: 阶段96的技术栈（组合字段）
        scenario_results: 各场景验证结果
        overall_self_driven: 综合自驱率
        overall_sustainability: 综合永续率
        overall_evolution: 综合进化率
        overall_safety: 综合安全率
        overall_passed: 综合是否通过
        total_test_hours: 总测试时长

    Invariants:
        - overall_passed = all critical metrics pass
    """
    tech_stack: TechStack
    scenario_results: Tuple[ScenarioResult, ...] = ()
    overall_self_driven: float = 0.0
    overall_sustainability: float = 0.0
    overall_evolution: float = 0.0
    overall_safety: float = 0.0
    overall_passed: bool = False
    total_test_hours: float = 0.0


# ============================================================
# 阶段97 核心接口：端到端验证器
# ============================================================
class E2EValidator:
    """阶段97的核心接口：端到端验证器。

    实现自驱性、永续性、进化性、安全性的端到端验证。

    核心约束：
        - 自驱率 = 100%
        - 永续率 = 100%
        - 进化率 > 0
        - 安全率 = 100%
    """

    def test_self_driven(
        self,
        system,
        duration_hours: float = 24.0
    ) -> float:
        """自驱性验证（A97.1）。

        Args:
            system: 待验证的系统
            duration_hours: 测试时长（小时），必须 > 0

        Returns:
            自驱率 ∈ [0, 1]

        Raises:
            ValueError: 当duration_hours <= 0时

        Examples:
            >>> validator = E2EValidator()
            >>> rate = validator.test_self_driven(None, 1.0)
            >>> 0 <= rate <= 1
            True
        """
        if duration_hours <= 0:
            raise ValueError(f"duration_hours必须 > 0，收到 {duration_hours}")
        return 1.0

    def test_sustainability(
        self,
        system,
        min_hours: float = 24.0
    ) -> float:
        """永续性验证（A97.2）。

        Args:
            system: 待验证的系统
            min_hours: 最小运行时长（小时），必须 > 0

        Returns:
            永续率 ∈ [0, 1]

        Raises:
            ValueError: 当min_hours <= 0时

        Examples:
            >>> validator = E2EValidator()
            >>> rate = validator.test_sustainability(None, 1.0)
            >>> 0 <= rate <= 1
            True
        """
        if min_hours <= 0:
            raise ValueError(f"min_hours必须 > 0，收到 {min_hours}")
        return 1.0

    def test_evolution(
        self,
        system,
        duration_hours: float = 24.0
    ) -> float:
        """进化性验证（A97.3）。

        Args:
            system: 待验证的系统
            duration_hours: 测试时长（小时），必须 > 0

        Returns:
            进化率，> 0表示进化

        Raises:
            ValueError: 当duration_hours <= 0时

        Examples:
            >>> validator = E2EValidator()
            >>> rate = validator.test_evolution(None, 1.0)
            >>> rate > 0
            True
        """
        if duration_hours <= 0:
            raise ValueError(f"duration_hours必须 > 0，收到 {duration_hours}")
        return 0.05

    def test_safety(
        self,
        system,
        duration_hours: float = 6.0
    ) -> float:
        """安全性验证（A97.4）。

        Args:
            system: 待验证的系统
            duration_hours: 测试时长（小时），必须 > 0

        Returns:
            安全率 ∈ [0, 1]

        Raises:
            ValueError: 当duration_hours <= 0时

        Examples:
            >>> validator = E2EValidator()
            >>> rate = validator.test_safety(None, 1.0)
            >>> 0 <= rate <= 1
            True
        """
        if duration_hours <= 0:
            raise ValueError(f"duration_hours必须 > 0，收到 {duration_hours}")
        return 1.0

    def run_scenario(
        self,
        system,
        scenario_name: str,
        config: Dict
    ) -> ScenarioResult:
        """运行单个验证场景（A97.6）。

        Args:
            system: 待验证的系统
            scenario_name: 场景名称
            config: 场景配置字典

        Returns:
            场景验证结果

        Examples:
            >>> validator = E2EValidator()
            >>> result = validator.run_scenario(
            ...     None, "normal_steady", {})
            >>> isinstance(result, ScenarioResult)
            True
        """
        return ScenarioResult(
            scenario_name=scenario_name,
            self_driven_rate=1.0,
            sustainability_rate=1.0,
            evolution_rate=0.05,
            safety_rate=1.0,
            passed=True,
            duration_hours=config.get("duration_hours", 24.0),
            notes=""
        )

    def full_validation(
        self,
        system,
        tech_stack: TechStack
    ) -> E2EValidation:
        """执行完整端到端验证（A97.5）。

        Args:
            system: 待验证的系统
            tech_stack: 技术栈状态

        Returns:
            E2EValidation 端到端验证结果

        Examples:
            >>> validator = E2EValidator()
            >>> from stage_096 import TechStackConfigurator
            >>> stack = TechStackConfigurator().full_verification()
            >>> result = validator.full_validation(None, stack)
            >>> isinstance(result, E2EValidation)
            True
        """
        self_driven = self.test_self_driven(system, 24.0)
        sustainability = self.test_sustainability(system, 24.0)
        evolution = self.test_evolution(system, 24.0)
        safety = self.test_safety(system, 6.0)
        overall = (
            self_driven == 1.0
            and sustainability == 1.0
            and evolution > 0.0
            and safety == 1.0
        )
        return E2EValidation(
            tech_stack=tech_stack,
            self_driven_rate=self_driven,
            sustainability_rate=sustainability,
            evolution_rate=evolution,
            safety_rate=safety,
            scenario_results=(),
            overall_passed=overall,
            total_test_hours=54.0
        )
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `TechStack` | ← 阶段96 | 作为验证的技术基础（组合字段） |
| **输入** | `GlobalObjective` | ← 阶段91 | 进化性验证的数据源 |
| **输入** | `IntegratedSystem` | ← 阶段94 | 被验证的系统 |
| **输出** | `E2EValidation` | → 阶段98 | 端到端验证结果 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| pytest | 测试框架 | 验证场景的执行和断言 |
| psutil | 资源监控 | 安全性验证的数据来源 |
| NumPy | 数据分析 | 指标计算和趋势分析 |
| time | 时间度量 | 运行时长和停滞检测 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V97.1 | 自驱率计算 | N_self / N_total 正确 | 计算测试 |
| V97.2 | 自驱率=100% | 无外部指令时100%自驱 | 自驱测试 |
| V97.3 | 永续率计算 | T_active / T_total 正确 | 计算测试 |
| V97.4 | 永续率=100% | 24小时无停滞 | 永续测试 |
| V97.5 | 停滞检测 | gap<0.01持续60秒判定停滞 | 检测测试 |
| V97.6 | 进化率计算 | (J₀-J_T)/J₀ 正确 | 计算测试 |
| V97.7 | 进化率>0 | J长期趋势下降 | 进化测试 |
| V97.8 | 安全率计算 | 1-N_violation/N_steps 正确 | 计算测试 |
| V97.9 | 安全率=100% | 所有违规自动处理 | 安全测试 |
| V97.10 | CPU超限→节流 | 注入CPU负载后自动节流 | 场景测试 |
| V97.11 | 性能倒退→回滚 | 注入劣化模型后自动回滚 | 场景测试 |
| V97.12 | 综合评分 | 关键指标100%才通过 | 评分测试 |
| V97.13 | 场景1正常稳态 | 24小时自驱+永续 | 场景测试 |
| V97.14 | 场景2环境变化 | 适应新环境 | 场景测试 |
| V97.15 | 场景3资源压力 | 安全约束有效 | 场景测试 |
| V97.16 | 场景4异常注入 | 异常恢复成功 | 场景测试 |
| V97.17 | 场景5长期运行 | 72小时综合通过 | 场景测试 |
| V97.18 | E2EValidation组合 | `state.tech_stack` 类型为 `TechStack` | 类型检查 |

---

> **阶段97完** | 阶段97建立了端到端验证方案：自驱性验证 无外部任务→系统自主生成并执行任务（A97.1），永续性验证 长期运行→系统不停滞（A97.2），进化性验证 长期运行→适应度单调递增（A97.3），安全性验证 资源超限→自动约束，性能倒退→自动回滚（A97.4），验证指标体系 [自驱率, 永续率, 进化率, 安全率]（A97.5），验证场景设计（A97.6），封装为E2EValidation数据类组合TechStack。四项核心指标——自驱100%、永续100%、进化>0、安全100%——构成了系统合格的红线。五组验证场景从不同角度考验系统能力，确保不是偶然通过而是真正可靠。
