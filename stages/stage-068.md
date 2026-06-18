# 阶段68：探索-利用的70/30强制比例实现——进化层的强制执行机制

> **阶段编号**：S68 | **所属层**：第七层·进化适应层
>
> **去重声明**：70/30的理论基础在阶段7（A7.7 EFE分解=外禀+认知，A7.10经验比例标注），工程实现在阶段38（A38.3比例强制执行器）。本阶段是进化层的强制执行——将70/30比例从任务选择层提升到模型进化层，确保微修正与结构重构的比例也遵循70/30。任何关于70/30的理论推导和公理依据均引用阶段7，任务层的工程实现引用阶段38，本阶段不重复。
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 进化层探索比例 | "微调和重构的比例"——微修正70%，结构重构30% |
> | 强制探索机制 | "不够就强制"——探索比例低于30%时强制触发重构 |
> | 探索多样性保证 | "不重复探索"——每次探索不同的结构方向 |
> | 探索结果评估与吸收 | "探索完要总结"——评估探索结果并吸收有效发现 |
> | ExploreExploitEnforcer | "比例执行器的完整画像"——组合StructuralReformerState，在重构器基础上增加比例控制 |

---

## 1. 第一性原理依据

本阶段从阶段38（70/30双决策机制）和阶段7（EFE分解）的核心结论出发，建立进化层的探索-利用强制比例机制。核心依据为：

- **A38.3（比例强制执行器）**（引用阶段38）：count_exploit / count_total = 0.7 ± 0.05，任务选择层的70/30比例强制执行器。本阶段将A38.3从任务选择层提升到模型进化层——微修正（利用）占70%，结构重构（探索）占30%。

- **A7.7（EFE完整分解）**（引用阶段7）：$G(\pi) = E_{\text{ext}}(\pi) - E_{\text{epi}}(\pi)$，预期自由能分解为外禀价值（利用）与认知价值（探索）。70/30比例是外禀价值与认知价值在典型场景下的相对权重。

- **A7.10（70/30经验比例标注）**（引用阶段7）：$E_{\text{ext}}/E_{\text{epi}} \approx 70/30$ 是经验参数。本阶段将这一经验观察从任务选择推广到模型进化。

- **A67.1-A67.5（结构性重构Python接口）**（引用阶段67）：StructuralReformerState记录了重构的完整状态。探索-利用强制比例需要基于重构状态计算当前比例。

本阶段的核心目标是：**定义进化层探索比例 explore_ratio = count_explore / count_total（A68.1），强制探索机制 if explore_ratio < 0.3 then force_explore()（A68.2），探索任务的多样性保证（A68.3），探索结果的评估与吸收（A68.4），封装为ExploreExploitEnforcer数据类，组合StructuralReformerState作为重构器状态的输入。**

> **去重声明**：理论基础在阶段7，工程实现在阶段38，本阶段是进化层的强制执行。本阶段不重新推导70/30的理论依据（引用阶段7），不重新实现任务层的比例强制执行器（引用阶段38），仅将70/30比例从任务选择层推广到模型进化层。

---

## 2. 核心数学公式

### 第一部分：进化层探索比例

#### 2.1 进化层探索比例定义（A68.1）

进化层的探索比例定义为结构重构次数占总进化操作次数的比例：

$$\text{explore\_ratio} = \frac{\text{count\_explore}}{\text{count\_total}} = \frac{N_{\text{reform}}}{N_{\text{micro}} + N_{\text{reform}}} \tag{A68.1}$$

其中：
- $N_{\text{reform}}$ 为结构重构执行次数
- $N_{\text{micro}}$ 为微修正执行次数
- $N_{\text{total}} = N_{\text{micro}} + N_{\text{reform}}$ 为总进化操作次数

**探索比例的目标范围**：

$$\text{explore\_ratio} \in [0.25, 0.35] \tag{A68.1-target}$$

即探索比例应在25%-35%之间，以30%为中心，±5%为容差。

**探索比例的滑动窗口计算**：

为避免早期比例波动过大，使用滑动窗口计算近期探索比例：

$$\text{explore\_ratio}_W = \frac{\sum_{t=t-W+1}^{t} \mathbb{1}[\text{reform}_t]}{W} \tag{A68.1-window}$$

其中 $W$ 为滑动窗口长度（通常设为100-200步）。

**与A38.3的对应**：A68.1是A38.3（引用阶段38）在进化层的对应。A38.3衡量任务选择层的探索比例，A68.1衡量模型进化层的探索比例。两者使用相同的70/30目标比例，但计算对象不同。

- **输入**：微修正次数、重构次数、窗口长度
- **输出**：探索比例 explore_ratio ∈ [0, 1]
- **约束**：count_total > 0

> **白话翻译**：进化层探索比例就是"微调和重构的比例"——结构重构次数占总进化操作次数的比例。目标是30%左右——70%的时间做微修正（利用已知知识），30%的时间做结构重构（探索新结构）。用滑动窗口计算近期比例，避免早期波动。

---

### 第二部分：强制探索机制

#### 2.2 强制探索机制定义（A68.2）

当探索比例低于30%时，强制触发结构重构：

$$\text{if } \text{explore\_ratio} < 0.3 \implies \text{force\_explore}() \tag{A68.2}$$

**强制探索的触发条件**：

$$\text{force\_explore} = \mathbb{1}\left[\text{explore\_ratio}_W < 0.25 \;\wedge\; \text{steps\_since\_last\_reform} > T_{\text{min\_explore}}\right] \tag{A68.2-condition}$$

其中 $T_{\text{min\_explore}}$ 为最小探索间隔（通常设为50步），避免过于频繁的强制探索。

**强制探索与自然触发的区别**：

| 维度 | 自然触发（A64.1-A64.3） | 强制探索（A68.2） |
|------|----------------------|------------------|
| 触发原因 | 误差大/停滞/环境变化 | 探索比例过低 |
| 重构方向 | 针对具体问题 | 随机或多样性驱动 |
| 风险评估 | 必须通过 | 可以放宽 |
| 紧急程度 | 高（问题已出现） | 低（预防性） |

**强制探索的重构方向**：

强制探索的重构方向不是针对具体问题，而是为了增加结构多样性：

1. **随机特征增加**：随机选择一个位置增加特征维度
2. **随机子模型重组**：随机选择两个子模型尝试合并
3. **随机规则迭代**：随机选择一个规则进行变异

强制探索的重构仍然受安全约束（阶段66）保护——即使重构方向是随机的，也必须通过回滚条件检查。

- **输入**：探索比例、最小探索间隔
- **输出**：强制探索标志
- **约束**：探索比例 < 0.25 时触发；间隔 > $T_{\text{min\_explore}}$

> **白话翻译**：强制探索机制就是"不够就强制"——当探索比例低于25%时，强制触发一次结构重构。和自然触发的区别是：自然触发是因为出了问题（误差大、停滞、环境变化），强制探索是因为探索不够（预防性）。强制探索的重构方向是随机的——不是为了解决具体问题，而是为了增加结构多样性。但安全约束仍然有效——改坏了还是要回滚。

---

### 第三部分：探索任务的多样性保证

#### 2.3 探索多样性保证定义（A68.3）

强制探索的重构方向应保证多样性，避免反复探索相似的结构方向：

$$\text{diversity}(\text{reform}_k) = \min_{j < k} d(\text{reform}_k, \text{reform}_j) \tag{A68.3}$$

其中 $d(\cdot, \cdot)$ 为重构操作之间的距离度量。

**重构操作的距离度量**：

$$d(\text{reform}_1, \text{reform}_2) = \|\text{embedding}(\text{reform}_1) - \text{embedding}(\text{reform}_2)\|_2 \tag{A68.3-distance}$$

其中 embedding(reform) 将重构操作映射为向量：

| 操作类型 | embedding |
|---------|-----------|
| 特征增加（位置 $p$） | $[1, 0, 0, 0, p/N_{\text{layers}}]$ |
| 特征删除（位置 $p$） | $[0, 1, 0, 0, p/N_{\text{layers}}]$ |
| 子模型合并 | $[0, 0, 1, 0, 0]$ |
| 子模型拆分 | $[0, 0, 0, 1, 0]$ |
| 规则迭代 | $[0.5, 0.5, 0, 0, 0]$ |

**多样性保证的采样策略**：

强制探索时，从候选重构操作中采样，采样概率与多样性成正比：

$$P(\text{reform}_k) \propto \text{diversity}(\text{reform}_k) \tag{A68.3-sampling}$$

即与历史重构操作越不同的候选操作，被选中的概率越高。

- **输入**：候选重构操作列表、历史重构操作
- **输出**：选定的重构操作
- **约束**：选定的操作与历史操作足够不同

> **白话翻译**：探索多样性保证就是"不重复探索"——每次强制探索时，选择和之前不同的重构方向。用距离度量衡量"不同"——操作类型不同、位置不同都算不同。采样时优先选"最不同"的方向，避免反复探索相似的结构。

---

### 第四部分：探索结果的评估与吸收

#### 2.4 探索结果评估与吸收定义（A68.4）

强制探索的结果需要评估，有效的发现应被吸收到模型中：

$$\text{absorb}(\text{reform\_result}) = \begin{cases} \text{keep} & \text{if } f_{\text{new}} > f_{\text{old}} \\ \text{discard} & \text{if } f_{\text{new}} \leq f_{\text{old}} \end{cases} \tag{A68.4}$$

**探索结果的评估指标**：

| 指标 | 计算方式 | 通过条件 |
|------|---------|---------|
| 适应度提升 | $\Delta f = f_{\text{new}} - f_{\text{old}}$ | $\Delta f > 0$ |
| 自由能下降 | $\Delta F = F_{\text{new}} - F_{\text{old}}$ | $\Delta F < 0$ |
| 新结构稳定性 | 验证窗口内自由能方差 | 方差 < 阈值 |
| 知识保留 | EWC正则化损失 | $L_{\text{ewc}} < L_{\text{ewc,max}}$ |

**探索结果的吸收流程**：

```
1. 执行强制探索重构
2. 进入验证窗口（A66.5）
3. 评估探索结果：
   a. 适应度是否提升？
   b. 自由能是否下降？
   c. 新结构是否稳定？
   d. 旧知识是否保留？
4. 若全部通过：吸收（保留新结构）
5. 若任一失败：丢弃（回滚到旧结构）
6. 记录探索结果到历史（无论成功失败）
```

**失败探索的价值**：

即使探索失败（适应度未提升），探索结果也有价值——它告诉我们"这个方向不行"，避免将来重复探索相似方向。失败探索的信息被记录到多样性保证系统中（A68.3），作为未来采样的参考。

- **输入**：探索重构结果、验证窗口数据
- **输出**：吸收/丢弃决策
- **约束**：失败探索仍记录到历史

> **白话翻译**：探索结果评估与吸收就是"探索完要总结"——评估探索的结果是否有效，有效的就吸收（保留新结构），无效的就丢弃（回滚到旧结构）。即使探索失败也有价值——它告诉我们"这个方向不行"，下次就不会再试了。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A68.1 | explore_ratio = count_explore / count_total | 微修正次数、重构次数 | 探索比例 ∈ [0, 1] | count_total > 0 |
| A68.2 | if explore_ratio < 0.3 then force_explore() | 探索比例、间隔 | 强制探索标志 | 比例 < 0.25 且间隔 > $T_{\min}$ |
| A68.3 | diversity(reform_k) = min_j d(reform_k, reform_j) | 候选操作、历史操作 | 多样性值 | 采样概率与多样性成正比 |
| A68.4 | absorb if f_new > f_old, discard otherwise | 探索结果、验证数据 | 吸收/丢弃 | 失败仍记录 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List
import numpy as np

# 引用阶段67的数据类
from stage_067 import StructuralReformerState
from stage_064 import ReformTriggerType


# ============================================================
# 阶段68 数据类：探索-利用强制执行器
# ============================================================
@dataclass(frozen=True)
class ExploreExploitEnforcer:
    """探索-利用强制执行器的完整描述。

    组合阶段67的StructuralReformerState，在重构器状态的基础上增加
    探索-利用比例控制的状态。

    Attributes:
        reformer_state: 阶段67的重构器状态（组合字段）
        count_micro: 微修正执行次数
        count_reform: 结构重构执行次数
        explore_ratio: 当前探索比例
        explore_ratio_window: 滑动窗口探索比例
        window_size: 滑动窗口长度
        target_ratio: 目标探索比例（0.3）
        ratio_tolerance: 比例容差（0.05）
        T_min_explore: 最小探索间隔
        steps_since_last_reform: 距上次重构的步数
        force_explore_active: 强制探索是否激活
        reform_history_embeddings: 历史重构操作的embedding
        last_explore_result: 上次探索结果

    Invariants:
        - count_micro >= 0
        - count_reform >= 0
        - target_ratio ∈ (0, 1)
        - ratio_tolerance ∈ (0, 0.5)
    """
    reformer_state: StructuralReformerState
    count_micro: int = 0
    count_reform: int = 0
    explore_ratio: float = 0.0
    explore_ratio_window: float = 0.0
    window_size: int = 100
    target_ratio: float = 0.3
    ratio_tolerance: float = 0.05
    T_min_explore: int = 50
    steps_since_last_reform: int = 100
    force_explore_active: bool = False
    reform_history_embeddings: Optional[np.ndarray] = None
    last_explore_result: Optional[Dict] = None

    def __post_init__(self):
        if self.count_micro < 0:
            raise ValueError(
                f"微修正次数必须 >= 0，收到 {self.count_micro}"
            )
        if self.count_reform < 0:
            raise ValueError(
                f"重构次数必须 >= 0，收到 {self.count_reform}"
            )
        if not (0 < self.target_ratio < 1):
            raise ValueError(
                f"目标比例必须 ∈ (0,1)，收到 {self.target_ratio}"
            )

    @property
    def count_total(self) -> int:
        """总进化操作次数。"""
        return self.count_micro + self.count_reform

    @property
    def is_ratio_healthy(self) -> bool:
        """探索比例是否在目标范围内。"""
        return abs(self.explore_ratio_window - self.target_ratio) <= self.ratio_tolerance

    @property
    def should_force_explore(self) -> bool:
        """是否应强制探索。"""
        return (
            self.explore_ratio_window < self.target_ratio - self.ratio_tolerance
            and self.steps_since_last_reform >= self.T_min_explore
        )


# ============================================================
# 阶段68 核心接口：探索-利用强制执行器
# ============================================================
class ExploreExploitEnforcerInterface:
    """阶段68的核心接口：探索-利用强制执行器。

    实现进化层的70/30比例强制执行。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 探索比例 ∈ [0.25, 0.35]
        - 强制探索受安全约束保护
        - 探索方向保证多样性
    """

    def compute_explore_ratio(
        self,
        count_micro: int,
        count_reform: int
    ) -> float:
        """计算探索比例（A68.1）。

        explore_ratio = count_reform / (count_micro + count_reform)

        Args:
            count_micro: 微修正次数
            count_reform: 重构次数

        Returns:
            探索比例 ∈ [0, 1]
        """
        total = count_micro + count_reform
        if total == 0:
            return 0.0
        return count_reform / total

    def check_force_explore(
        self,
        enforcer: ExploreExploitEnforcer
    ) -> bool:
        """检查是否需要强制探索（A68.2）。

        Args:
            enforcer: 当前执行器状态

        Returns:
            是否需要强制探索
        """
        return (
            enforcer.explore_ratio_window < enforcer.target_ratio - enforcer.ratio_tolerance
            and enforcer.steps_since_last_reform >= enforcer.T_min_explore
        )

    def select_diverse_reform(
        self,
        candidates: List[Dict],
        history_embeddings: np.ndarray
    ) -> Dict:
        """选择多样性最高的重构操作（A68.3）。

        Args:
            candidates: 候选重构操作列表
            history_embeddings: 历史重构embedding

        Returns:
            选定的重构操作
        """
        if not candidates:
            raise ValueError("候选重构操作列表不能为空")
        if history_embeddings is None or len(history_embeddings) == 0:
            return candidates[0]
        max_diversity = -1.0
        best_candidate = candidates[0]
        for candidate in candidates:
            emb = candidate.get("embedding")
            if emb is None:
                continue
            distances = np.linalg.norm(history_embeddings - emb, axis=1)
            diversity = float(np.min(distances)) if len(distances) > 0 else float("inf")
            if diversity > max_diversity:
                max_diversity = diversity
                best_candidate = candidate
        return best_candidate

    def evaluate_explore_result(
        self,
        fitness_old: float,
        fitness_new: float,
        free_energy_old: float,
        free_energy_new: float,
        ewc_loss: float
    ) -> Tuple[bool, Dict]:
        """评估探索结果（A68.4）。

        Args:
            fitness_old: 重构前适应度
            fitness_new: 重构后适应度
            free_energy_old: 重构前自由能
            free_energy_new: 重构后自由能
            ewc_loss: EWC正则化损失

        Returns:
            (是否吸收, 评估详情)
        """
        absorb = fitness_new > fitness_old
        details = {
            "fitness_delta": fitness_new - fitness_old,
            "free_energy_delta": free_energy_new - free_energy_old,
            "ewc_loss": ewc_loss,
            "absorb": absorb,
        }
        return absorb, details

    def step(
        self,
        enforcer: ExploreExploitEnforcer,
        model: Dict
    ) -> Tuple[Dict, ExploreExploitEnforcer]:
        """执行一步探索-利用强制执行（A68.1-A68.4）。

        Args:
            enforcer: 当前执行器状态
            model: 当前模型

        Returns:
            (更新后的模型, 更新后的执行器状态)
        """
        from dataclasses import replace
        force = self.check_force_explore(enforcer)
        updated_enforcer = replace(
            enforcer,
            force_explore_active=force,
        )
        return model, updated_enforcer
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `StructuralReformerState` | ← 阶段67 | 作为执行器的重构器状态输入（组合字段） |
| **输入** | A38.3（比例强制执行器） | ← 阶段38 | 任务层70/30的工程实现参考 |
| **输入** | A7.7/A7.10（EFE分解与70/30） | ← 阶段7 | 70/30的理论基础 |
| **输出** | `ExploreExploitEnforcer` | → 阶段69 | 执行器状态，作为稳态探索上浮的输入 |
| **输出** | A68.1-A68.4（比例强制执行定义） | → 阶段70 | 验证测试的依据 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 比例计算与采样 | 探索比例的快速计算和多样性采样 |
| NumPy | 距离计算 | 重构操作embedding的距离度量 |
| Pydantic | 数据验证 | 支撑ExploreExploitEnforcer的运行时类型检查 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V68.1 | 探索比例计算 | explore_ratio = count_reform / count_total | 数值验证 |
| V68.2 | 滑动窗口比例 | 窗口内比例与全局比例趋势一致 | 窗口验证 |
| V68.3 | 强制探索触发 | explore_ratio < 0.25 时触发 | 条件测试 |
| V68.4 | 强制探索间隔 | 距上次重构 < $T_{\min}$ 时不触发 | 间隔测试 |
| V68.5 | 多样性采样 | 选定操作与历史操作距离最大 | 距离验证 |
| V68.6 | 探索结果评估 | 适应度提升时吸收，否则丢弃 | 条件测试 |
| V68.7 | 失败探索记录 | 失败探索仍记录到历史 | 历史检查 |
| V68.8 | 长期比例维持 | 1000步后探索比例 ∈ [0.25, 0.35] | 长期运行测试 |
| V68.9 | ExploreExploitEnforcer组合 | `enforcer.reformer_state` 类型为 `StructuralReformerState` | 类型检查 |
| V68.10 | 非法参数拒绝 | target_ratio ∉ (0,1) 抛出 `ValueError` | 异常测试 |
| V68.11 | 强制探索安全约束 | 强制探索仍受回滚条件保护 | 安全测试 |
| V68.12 | 与任务层70/30一致性 | 进化层比例与任务层比例目标一致 | 一致性验证 |

---

> **阶段68完** | 阶段68建立了进化层的70/30强制比例实现：进化层探索比例 explore_ratio = count_explore / count_total（A68.1），强制探索机制 if explore_ratio < 0.3 then force_explore()（A68.2），探索多样性保证 diversity(reform_k) = min_j d(reform_k, reform_j)（A68.3），探索结果评估与吸收（A68.4），封装为ExploreExploitEnforcer数据类组合StructuralReformerState。下一步：阶段69将定义稳态环境探索权重上浮。
