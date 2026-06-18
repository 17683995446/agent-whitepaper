# 阶段7：预期自由能定义——外禀价值与认知价值

> **阶段编号**：S7 | **所属层**：第一层·基础公理层
>
> **去重声明**：本阶段是预期自由能（Expected Free Energy, EFE）的**全局唯一定义点**。后续所有阶段（8/9/10/18/29/31等）引用预期自由能、外禀价值、认知价值、策略选择时，均指向本阶段。任何其他阶段不得重新定义或推导预期自由能。变分自由能的全局唯一定义见阶段4（A4.4），自由能分解见阶段5（A5.1-A5.18），主动推断公理见阶段6（A6.4-A6.5）。本阶段仅引用，不重复。
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 预期自由能 | "选哪条路的总账单预估"——还没走，先算算每条路大概要花多少代价 |
> | 外禀价值 | "这条路离目标多远"——走这条路能不能到达想去的地方 |
> | 认知价值 | "这条路让我多了解世界"——走这条路能学到多少新东西 |
> | 策略选择 | "选最划算的那条路"——既接近目标，又学到东西 |
> | 探索-利用 | "不是对立的，是同一枚硬币的两面"——EFE的两个分量自动平衡 |
> | 策略后验 | "各条路的选票"——账单越低的路，得票越多 |
> | 偏好分布 | "想去的地方长什么样"——编码了什么观测是"好的" |

---

## 1. 第一性原理依据

本阶段从阶段6、阶段4和阶段5的核心结论出发，无需引入新的物理公理。核心依据为：

- **A6.4-A6.5（主动推断公理）**（引用阶段6）：行动方程 $\dot{a} = -\kappa_a \partial F / \partial a$ 与感知-行动闭环 $\dot{a} = -\kappa_a \partial F / \partial a$，$\dot{\mu} = -\kappa_\mu \partial F / \partial \mu$。本阶段将行动方程从"即时梯度下降"推广到"策略级规划"——不是对当前行动做梯度下降，而是对所有可能策略做预期评估。
- **A4.4（变分自由能全局唯一定义）**（引用阶段4）：$F[q] = -\mathbb{E}_q[\ln p(o|\psi)] + D_{\text{KL}}[q(\psi) \| p(\psi)]$，预期自由能从此定义出发推广到策略选择场景。
- **A5.1-A5.18（自由能分解）**（引用阶段5）：准确性项 $A(q)$ 与复杂性项 $C(q)$ 的严格分析、梯度计算、Pareto最优权衡。本阶段将准确性-复杂性分解推广为外禀价值-认知价值分解。
- **定理6.1（主动行动生存优势）**（引用阶段6）：主动预测者的生存概率严格高于被动预测者，为策略选择提供生存驱力的理论基础。

本阶段的核心目标是回答：**Agent如何在多个可能策略之间做出选择？** 阶段6的行动方程（A6.4）是即时梯度下降——只看当前步的自由能梯度。但实际决策需要"向前看"：评估每个策略的预期后果，选择预期自由能最低的策略。本阶段将从变分自由能（A4.4）出发，完整推导预期自由能的定义、分解、策略选择规则和计算近似。

---

## 2. 核心数学公式

### 第一部分：从变分自由能到预期自由能

#### 2.1 预期自由能的全局唯一定义（A7.1）⭐

**A7.1 是预期自由能的全局唯一定义点。** 后续所有阶段引用"预期自由能"或"EFE"时，均指向此定义。

预期自由能将变分自由能（A4.4，引用阶段4）从"给定观测下的自由能"推广到"策略条件下的预期自由能"。给定策略 $\pi$，预期自由能定义为：

$$G(\pi) = \mathbb{E}_{q(\psi, o|\pi)}\left[\ln q(\psi) - \ln p(\psi, o|\pi)\right] \tag{A7.1}$$

其中：
- $q(\psi, o|\pi) = q(\psi|o, \pi) \, q(o|\pi)$：策略条件下的变分联合分布
- $p(\psi, o|\pi) = p(o|\psi, \pi) \, p(\psi)$：策略条件下的生成模型联合分布
- $\mathbb{E}_{q(\psi, o|\pi)}[\cdot]$：在变分联合分布下的期望

- **输入**：变分联合分布 $q(\psi, o|\pi)$，生成模型 $p(\psi, o|\pi)$，策略 $\pi$
- **输出**：$G(\pi) \in \mathbb{R}$
- **约束**：$q(\psi, o|\pi)$ 为合法概率分布；$p(\psi, o|\pi)$ 为合法概率分布

> **白话翻译**：预期自由能就是"如果我选了策略 $\pi$，未来的总账单预期是多少"。它把变分自由能从"看到观测后的账单"推广到"还没看到观测、但预期会看到什么观测的账单"。关键区别：变分自由能是事后算账（观测已给定），预期自由能是事前算账（观测还没来，取期望）。

**推导：从A4.4出发**

**步骤1：回顾变分自由能**

由A4.4（引用阶段4），给定观测 $o$ 的变分自由能为：

$$F[q] = \mathbb{E}_q[\ln q(\psi)] - \mathbb{E}_q[\ln p(o, \psi)]$$

**步骤2：引入策略条件**

在策略选择场景中，观测 $o$ 不是给定的，而是取决于策略 $\pi$。因此需要将自由能推广到策略条件下的期望。将 $o$ 视为随机变量（由策略 $\pi$ 和环境共同决定），在变分联合分布 $q(\psi, o|\pi)$ 下取期望：

$$G(\pi) = \mathbb{E}_{q(\psi, o|\pi)}\left[\ln q(\psi) - \ln p(\psi, o|\pi)\right]$$

**步骤3：与变分自由能的关系**

当观测 $o$ 给定时（即 $q(o|\pi) = \delta(o - o_0)$），预期自由能退化为变分自由能：

$$G(\pi)\big|_{o = o_0} = \mathbb{E}_{q(\psi|o_0, \pi)}\left[\ln q(\psi) - \ln p(\psi, o_0|\pi)\right] = F[q]$$

因此预期自由能是变分自由能在策略选择场景下的自然推广——从"事后算账"到"事前预估"。

$\blacksquare$

#### 2.2 EFE的第一步分解（A7.2）

将A7.1中的联合分布分解，得到EFE的第一步分解：

$$G(\pi) = \underbrace{\mathbb{E}_{q(\psi, o|\pi)}\left[\ln q(\psi) - \ln p(o|\pi)\right]}_{\text{外禀价值相关项}} + \underbrace{\mathbb{E}_{q(\psi, o|\pi)}\left[-\ln p(\psi|o, \pi)\right]}_{\text{认知价值相关项}} \tag{A7.2}$$

- **输入**：同A7.1
- **输出**：$G(\pi) = G_{\text{ext-related}} + G_{\text{epi-related}}$
- **约束**：$p(\psi, o|\pi) = p(o|\pi) \, p(\psi|o, \pi)$（贝叶斯分解）

> **白话翻译**：预期自由能可以拆成两笔账：第一笔跟"观测的预期意外程度"有关，第二笔跟"看到观测后对状态的后验不确定性"有关。前者是"我预期会多意外"，后者是"我预期会多不确定"。

**完整推导**：

**步骤1：展开A7.1**

$$G(\pi) = \mathbb{E}_{q(\psi, o|\pi)}\left[\ln q(\psi) - \ln p(\psi, o|\pi)\right]$$

**步骤2：分解联合分布**

$$p(\psi, o|\pi) = p(o|\pi) \, p(\psi|o, \pi)$$

代入：

$$G(\pi) = \mathbb{E}_{q(\psi, o|\pi)}\left[\ln q(\psi) - \ln p(o|\pi) - \ln p(\psi|o, \pi)\right]$$

$$= \mathbb{E}_{q(\psi, o|\pi)}\left[\ln q(\psi) - \ln p(o|\pi)\right] + \mathbb{E}_{q(\psi, o|\pi)}\left[-\ln p(\psi|o, \pi)\right]$$

$\blacksquare$

---

### 第二部分：外禀价值（利用项）

#### 2.3 外禀价值的定义（A7.3）

外禀价值（Extrinsic Value）是预期自由能中与"观测的预期意外程度"相关的项。由A7.2的第一项，利用 $q(\psi, o|\pi) = q(\psi|o, \pi) \, q(o|\pi)$ 对第一项做进一步分解：

$$\mathbb{E}_{q(\psi, o|\pi)}\left[\ln q(\psi) - \ln p(o|\pi)\right] = \mathbb{E}_{q(o|\pi)}\left[\mathbb{E}_{q(\psi|o,\pi)}[\ln q(\psi)] - \ln p(o|\pi)\right]$$

外禀价值定义为：

$$E_{\text{ext}}(\pi) = \mathbb{E}_{q(o|\pi)}\left[-\ln p(o|\pi)\right] = -\mathbb{E}_{q(o|\pi)}\left[\ln p(o|\pi)\right] \tag{A7.3}$$

- **输入**：策略条件下的观测边际分布 $q(o|\pi)$，偏好分布 $p(o|\pi)$
- **输出**：$E_{\text{ext}}(\pi) \in \mathbb{R}_{\geq 0}$
- **约束**：$p(o|\pi)$ 为合法概率分布；$E_{\text{ext}}(\pi) \geq 0$（负对数似然的期望非负）

> **白话翻译**：外禀价值衡量"这个策略让我离目标多远"——如果策略产生的观测与偏好分布高度吻合，$p(o|\pi)$ 大，$-\ln p(o|\pi)$ 小，外禀价值低；如果观测与偏好不符，$p(o|\pi)$ 小，$-\ln p(o|\pi)$ 大，外禀价值高。**最小化外禀价值 = 最大化观测与偏好的吻合度 = 利用（exploitation）。**

**与A4.5的连接**：外禀价值对应变分自由能中的准确性项（A4.5第一项，引用阶段4）在策略条件下的期望——不是"给定观测的准确性"，而是"预期观测的准确性"。

#### 2.4 偏好分布下的外禀价值分解（A7.4）

当引入偏好分布 $p(o)$（编码了"什么样的观测是好的"）时，外禀价值可分解为KL散度与熵之和：

$$E_{\text{ext}}(\pi) = D_{\text{KL}}\left[q(o|\pi) \| p(o)\right] + H\left[q(o|\pi)\right] \tag{A7.4}$$

其中：
- $D_{\text{KL}}[q(o|\pi) \| p(o)] = \mathbb{E}_{q(o|\pi)}\left[\ln \frac{q(o|\pi)}{p(o)}\right] \geq 0$：策略观测分布与偏好分布之间的KL散度
- $H[q(o|\pi)] = -\mathbb{E}_{q(o|\pi)}[\ln q(o|\pi)]$：策略观测分布的Shannon熵

- **输入**：策略观测分布 $q(o|\pi)$，偏好分布 $p(o)$
- **输出**：$E_{\text{ext}}(\pi) = D_{\text{KL}} + H$
- **约束**：$D_{\text{KL}} \geq 0$；$H \geq 0$

> **白话翻译**：外禀价值 = "策略观测与偏好的偏离"（KL散度）+ "策略观测的不确定性"（熵）。KL散度衡量"这个策略让我离目标多远"——偏离偏好越远，此项越大；熵衡量"策略产生的观测有多随机"——越随机，此项越大。**最小化外禀价值 = 让观测既接近偏好，又不太随机。**

**完整推导**：

**步骤1：展开 $-\ln p(o|\pi)$**

在偏好分布设定下，$p(o|\pi) = p(o)$（偏好分布不依赖策略，详见A7.15），因此：

$$E_{\text{ext}}(\pi) = -\mathbb{E}_{q(o|\pi)}[\ln p(o)]$$

**步骤2：引入 $q(o|\pi)$**

$$-\mathbb{E}_{q(o|\pi)}[\ln p(o)] = -\mathbb{E}_{q(o|\pi)}[\ln q(o|\pi)] + \mathbb{E}_{q(o|\pi)}[\ln q(o|\pi)] - \mathbb{E}_{q(o|\pi)}[\ln p(o)]$$

$$= H[q(o|\pi)] + D_{\text{KL}}[q(o|\pi) \| p(o)]$$

$\blacksquare$

**外禀价值的两个分量**：

| 分量 | 公式 | 含义 | 符号 |
|------|------|------|------|
| KL散度项 | $D_{\text{KL}}[q(o\|\pi) \| p(o)]$ | 策略观测与偏好的偏离 | $\geq 0$，偏好吻合时为0 |
| 熵项 | $H[q(o\|\pi)]$ | 策略观测的不确定性 | $\geq 0$，确定性策略时为0 |

---

### 第三部分：认知价值（探索项）

#### 2.5 认知价值的定义（A7.5）

认知价值（Epistemic Value）是预期自由能中与"信息增益"相关的项。由A7.2的第二项：

$$E_{\text{epi}}(\pi) = \mathbb{E}_{q(\psi, o|\pi)}\left[\ln q(\psi|o, \pi) - \ln q(\psi)\right] \tag{A7.5}$$

利用 $q(\psi, o|\pi) = q(\psi|o, \pi) \, q(o|\pi)$，可重写为：

$$E_{\text{epi}}(\pi) = \mathbb{E}_{q(o|\pi)}\left[D_{\text{KL}}[q(\psi|o, \pi) \| q(\psi)]\right] = I_q(\Psi; O | \pi)$$

其中 $I_q(\Psi; O | \pi)$ 为策略条件下隐状态 $\Psi$ 与观测 $O$ 之间的互信息。

- **输入**：变分后验 $q(\psi|o, \pi)$，变分先验 $q(\psi)$，策略 $\pi$
- **输出**：$E_{\text{epi}}(\pi) \in \mathbb{R}_{\geq 0}$
- **约束**：$E_{\text{epi}}(\pi) \geq 0$（KL散度的期望非负）；$E_{\text{epi}}(\pi) = 0$ 当且仅当 $q(\psi|o, \pi) = q(\psi)$ 几乎处处成立（观测不提供关于状态的任何信息）

> **白话翻译**：认知价值衡量"这个策略让我多了解世界"——如果执行策略后，观测能大幅更新对状态的信念（$q(\psi|o, \pi)$ 与 $q(\psi)$ 差异大），认知价值高；如果观测对信念没有影响，认知价值为零。**最大化认知价值 = 最大化信息增益 = 探索（exploration）。**

**完整推导**：

**步骤1：从A7.2的第二项出发**

$$\mathbb{E}_{q(\psi, o|\pi)}\left[-\ln p(\psi|o, \pi)\right]$$

此项目前不是认知价值的直接定义。认知价值来自A7.1中 $\ln q(\psi)$ 项与 $\ln p(\psi|o, \pi)$ 项的组合。重新审视A7.1的完整分解：

$$G(\pi) = \mathbb{E}_{q(\psi, o|\pi)}\left[\ln q(\psi) - \ln p(o|\pi) - \ln p(\psi|o, \pi)\right]$$

将 $\ln q(\psi)$ 与 $-\ln p(\psi|o, \pi)$ 组合：

$$\mathbb{E}_{q(\psi, o|\pi)}\left[\ln q(\psi) - \ln p(\psi|o, \pi)\right]$$

在变分推断框架中，$p(\psi|o, \pi)$ 是真实后验（不可计算），用变分后验 $q(\psi|o, \pi)$ 近似。当 $q(\psi|o, \pi) \approx p(\psi|o, \pi)$ 时（变分推断的目标，引用阶段4，A4.4）：

$$\mathbb{E}_{q(\psi, o|\pi)}\left[\ln q(\psi) - \ln q(\psi|o, \pi)\right] = -\mathbb{E}_{q(o|\pi)}\left[D_{\text{KL}}[q(\psi|o, \pi) \| q(\psi)]\right] = -E_{\text{epi}}(\pi)$$

因此认知价值 $E_{\text{epi}}(\pi) = \mathbb{E}_{q(o|\pi)}[D_{\text{KL}}[q(\psi|o, \pi) \| q(\psi)]]$。

**步骤2：证明等价于互信息**

$$E_{\text{epi}}(\pi) = \mathbb{E}_{q(o|\pi)}\left[D_{\text{KL}}[q(\psi|o, \pi) \| q(\psi)]\right]$$

$$= \mathbb{E}_{q(o|\pi)}\left[\mathbb{E}_{q(\psi|o,\pi)}\left[\ln \frac{q(\psi|o, \pi)}{q(\psi)}\right]\right]$$

$$= \mathbb{E}_{q(\psi, o|\pi)}\left[\ln \frac{q(\psi|o, \pi)}{q(\psi)}\right]$$

$$= \mathbb{E}_{q(\psi, o|\pi)}\left[\ln \frac{q(\psi, o|\pi)}{q(\psi) \, q(o|\pi)}\right]$$

$$= I_q(\Psi; O | \pi)$$

这正是互信息的定义——策略条件下隐状态与观测之间的互信息。

$\blacksquare$

#### 2.6 认知价值的信息论统一（A7.6）

认知价值具有信息论中的多重等价解释：

$$E_{\text{epi}}(\pi) = I_q(\Psi; O | \pi) = \mathbb{E}_{q(o|\pi)}\left[D_{\text{KL}}[q(\psi|o, \pi) \| q(\psi)]\right] = H[q(\psi)] - \mathbb{E}_{q(o|\pi)}[H[q(\psi|o, \pi)]] \tag{A7.6}$$

即：

$$\text{认知价值} = \text{互信息} = \text{信息增益} = \text{好奇心}$$

- **输入**：变分分布 $q(\psi)$，$q(\psi|o, \pi)$，$q(o|\pi)$
- **输出**：$E_{\text{epi}}(\pi) \in \mathbb{R}_{\geq 0}$
- **约束**：四种表述在数学上严格等价

> **白话翻译**：认知价值有四种等价的说法：（1）"观测能告诉我多少关于状态的信息"（互信息）；（2）"看到观测后，信念改变了多少"（信息增益/KL散度期望）；（3）"观测前的不确定性减去观测后的不确定性"（熵减）；（4）"好奇心"——越不确定的地方越想去看看。这四种说法是同一枚硬币的四个面——信息论把它们统一了。

**等价性证明**：

**互信息 = KL散度期望**：已在A7.5中证明。

**互信息 = 熵减**：

$$I_q(\Psi; O | \pi) = H[q(\psi)] - H[q(\psi) | q(o|\pi)]$$

其中 $H[q(\psi) | q(o|\pi)] = \mathbb{E}_{q(o|\pi)}[H[q(\psi|o, \pi)]]$ 为条件熵。

由互信息的定义：

$$I_q(\Psi; O | \pi) = H[q(\psi)] - H[q(\psi) | q(o|\pi)]$$

$$= H[q(\psi)] - \mathbb{E}_{q(o|\pi)}[H[q(\psi|o, \pi)]]$$

因此认知价值 = 先验不确定性 - 后验不确定性的期望 = 不确定性的预期减少量。

$\blacksquare$

---

### 第四部分：EFE的完整分解

#### 2.7 EFE的完整分解（A7.7）⭐

综合A7.3-A7.6，预期自由能可分解为外禀价值与认知价值之差：

$$G(\pi) = E_{\text{ext}}(\pi) - E_{\text{epi}}(\pi) \tag{A7.7}$$

展开为：

$$G(\pi) = \underbrace{D_{\text{KL}}[q(o|\pi) \| p(o)] + H[q(o|\pi)]}_{\text{外禀价值 } E_{\text{ext}}(\pi)} - \underbrace{I_q(\Psi; O | \pi)}_{\text{认知价值 } E_{\text{epi}}(\pi)}$$

- **输入**：策略 $\pi$，变分分布 $q$，偏好分布 $p(o)$
- **输出**：$G(\pi) \in \mathbb{R}$
- **约束**：$E_{\text{ext}}(\pi) \geq 0$；$E_{\text{epi}}(\pi) \geq 0$；$G(\pi)$ 可正可负

> **白话翻译**：预期自由能 = 外禀价值 - 认知价值。"外禀价值"是利用的代价——偏离偏好越远，代价越高；"认知价值"是探索的收益——学到越多，收益越高。**最小化EFE = 最小化利用代价 + 最大化探索收益。** 注意符号：认知价值是"减去的"——信息增益降低预期自由能，鼓励探索。

**完整推导**：

**步骤1：从A7.1出发**

$$G(\pi) = \mathbb{E}_{q(\psi, o|\pi)}\left[\ln q(\psi) - \ln p(\psi, o|\pi)\right]$$

**步骤2：分解联合分布**

$$p(\psi, o|\pi) = p(o|\pi) \, p(\psi|o, \pi)$$

代入：

$$G(\pi) = \mathbb{E}_{q(\psi, o|\pi)}\left[\ln q(\psi) - \ln p(o|\pi)\right] + \mathbb{E}_{q(\psi, o|\pi)}\left[-\ln p(\psi|o, \pi)\right]$$

**步骤3：处理第一项——外禀价值**

$$\mathbb{E}_{q(\psi, o|\pi)}\left[\ln q(\psi) - \ln p(o|\pi)\right]$$

对 $\psi$ 积分（利用 $\int q(\psi|o, \pi) \, \mathrm{d}\psi = 1$）：

$$= \mathbb{E}_{q(o|\pi)}\left[\mathbb{E}_{q(\psi|o,\pi)}[\ln q(\psi)] - \ln p(o|\pi)\right]$$

在变分近似 $q(\psi|o, \pi) \approx p(\psi|o, \pi)$ 下，$\mathbb{E}_{q(\psi|o,\pi)}[\ln q(\psi)]$ 项需要进一步处理。利用 $q(\psi) = \int q(\psi|o, \pi) q(o|\pi) \, \mathrm{d}o$，将 $\mathbb{E}_{q(\psi|o,\pi)}[\ln q(\psi)]$ 与 $-\ln p(o|\pi)$ 组合：

$$= \mathbb{E}_{q(o|\pi)}[-\ln p(o|\pi)] + \mathbb{E}_{q(\psi, o|\pi)}[\ln q(\psi)]$$

第一项即为外禀价值 $E_{\text{ext}}(\pi) = -\mathbb{E}_{q(o|\pi)}[\ln p(o|\pi)]$（A7.3）。

**步骤4：处理第二项——认知价值的负值**

$$\mathbb{E}_{q(\psi, o|\pi)}\left[-\ln p(\psi|o, \pi)\right]$$

在变分近似下，$p(\psi|o, \pi) \approx q(\psi|o, \pi)$：

$$\approx \mathbb{E}_{q(\psi, o|\pi)}\left[-\ln q(\psi|o, \pi)\right]$$

**步骤5：合并步骤3和步骤4**

$$G(\pi) = \mathbb{E}_{q(o|\pi)}[-\ln p(o|\pi)] + \mathbb{E}_{q(\psi, o|\pi)}[\ln q(\psi)] - \mathbb{E}_{q(\psi, o|\pi)}[\ln q(\psi|o, \pi)]$$

$$= E_{\text{ext}}(\pi) + \mathbb{E}_{q(\psi, o|\pi)}\left[\ln q(\psi) - \ln q(\psi|o, \pi)\right]$$

$$= E_{\text{ext}}(\pi) - \mathbb{E}_{q(o|\pi)}\left[D_{\text{KL}}[q(\psi|o, \pi) \| q(\psi)]\right]$$

$$= E_{\text{ext}}(\pi) - E_{\text{epi}}(\pi)$$

$\blacksquare$

#### 2.8 策略选择规则（A7.8）

最优策略选择预期自由能最小的策略：

$$\pi^* = \arg\min_{\pi} G(\pi) = \arg\min_{\pi}\left(E_{\text{ext}}(\pi) - E_{\text{epi}}(\pi)\right) \tag{A7.8}$$

等价于：

$$\pi^* = \arg\min_{\pi}\left(D_{\text{KL}}[q(o|\pi) \| p(o)] + H[q(o|\pi)] - I_q(\Psi; O | \pi)\right)$$

- **输入**：策略空间 $\Pi$，预期自由能 $G(\pi)$ 对所有 $\pi \in \Pi$
- **输出**：最优策略 $\pi^* \in \Pi$
- **约束**：策略空间 $\Pi$ 非空；$G(\pi)$ 对所有 $\pi$ 可计算

> **白话翻译**：选最接近目标且最能学到东西的策略。具体地，选让"偏离偏好的代价 + 观测随机性 - 信息增益"最小的策略。这个规则自动平衡利用（减少偏离偏好）和探索（增加信息增益），不需要额外的探索机制。

**与A6.4的连接**：A6.4（引用阶段6）是即时行动方程——对当前行动做梯度下降。A7.8是策略级选择——对所有可能策略做预期评估后选择最优。当策略空间退化为单步行动时，A7.8退化为A6.4。

---

### 第五部分：探索-利用的统一

#### 2.9 探索-利用不是对立的（A7.9）

在预期自由能框架中，探索与利用不是需要额外机制来平衡的对立目标，而是EFE的两个自然分量：

$$\underbrace{G(\pi)}_{\text{预期自由能}} = \underbrace{E_{\text{ext}}(\pi)}_{\text{利用项（外禀价值）}} - \underbrace{E_{\text{epi}}(\pi)}_{\text{探索项（认知价值）}} \tag{A7.9}$$

- **输入**：EFE分解（A7.7）
- **输出**：探索-利用统一框架
- **约束**：两项均非负；最小化 $G$ 自动平衡两项

> **白话翻译**：在传统强化学习中，探索和利用是"鱼与熊掌"——需要 $\epsilon$-贪心、UCB等额外机制来平衡。但在预期自由能框架中，探索和利用是"同一枚硬币的两面"——它们是同一个目标函数（EFE）的两个分量，最小化EFE自动平衡两者。不需要额外的探索机制，不需要手动调参数。

**与传统方法的对比**：

| 方法 | 探索机制 | 利用机制 | 平衡方式 |
|------|---------|---------|---------|
| $\epsilon$-贪心 | 以概率 $\epsilon$ 随机行动 | 以概率 $1-\epsilon$ 贪心行动 | 手动调 $\epsilon$ |
| UCB | 不确定性 bonuses | 均值估计 | 手动调置信系数 |
| Thompson采样 | 从后验采样 | 后验均值 | 采样过程隐式平衡 |
| **EFE（本阶段）** | **认知价值 $E_{\text{epi}}$（自动）** | **外禀价值 $E_{\text{ext}}$（自动）** | **最小化 $G$ 自动平衡** |

#### 2.10 70/30比例标注（A7.10）

在实际应用中，外禀价值与认知价值的相对权重常被经验性地设定为约70/30的比例。**必须明确标注：这是经验参数，非公理推导结论。**

$$\text{经验比例：} \frac{E_{\text{ext}}(\pi^*)}{E_{\text{epi}}(\pi^*)} \approx \frac{70}{30} \tag{A7.10}$$

- **输入**：经验数据
- **输出**：经验比例 ≈ 70/30
- **约束**：**此比例为经验参数，非公理推导结论。** 具体比例取决于环境结构、偏好分布的锐度、变分分布的不确定性等因素。在确定性环境中（$H[q(o|\pi)] \approx 0$），外禀价值占主导；在高不确定性环境中，认知价值占主导。

> **白话翻译**：有人说"利用占70%，探索占30%"——这只是经验观察，不是定理。在熟悉的环境中（不确定性低），利用自然占主导；在陌生的环境中（不确定性高），探索自然占主导。EFE框架自动调节比例，不需要人为设定。

**比例的理论分析**：

由A7.7，$G(\pi) = E_{\text{ext}}(\pi) - E_{\text{epi}}(\pi)$。在最优策略 $\pi^*$ 处，两者的相对大小取决于：

1. **偏好分布的锐度**：$p(o)$ 越集中（偏好越明确），$D_{\text{KL}}[q(o|\pi) \| p(o)]$ 越大，外禀价值越占主导。
2. **变分分布的不确定性**：$H[q(\psi)]$ 越大（对世界越不确定），$I_q(\Psi; O | \pi)$ 的上界越大，认知价值越占主导。
3. **环境的随机性**：环境越随机，$H[q(o|\pi)]$ 越大，外禀价值中的熵项越占主导。

因此，70/30不是普遍规律，而是在"中等偏好锐度 + 中等不确定性"的典型场景下的经验观察。

#### 2.11 策略后验（A7.11）

策略选择可以通过策略后验分布实现。策略后验定义为EFE的softmax变换：

$$q(\pi) = \sigma(-\gamma \cdot G(\pi)) = \frac{\exp(-\gamma \cdot G(\pi))}{\sum_{\pi'} \exp(-\gamma \cdot G(\pi'))} \tag{A7.11}$$

其中 $\gamma > 0$ 为策略精度（inverse temperature），控制策略选择的确定性程度。

- **输入**：预期自由能 $G(\pi)$ 对所有 $\pi \in \Pi$，策略精度 $\gamma > 0$
- **输出**：策略后验分布 $q(\pi)$，满足 $q(\pi) \geq 0$，$\sum_\pi q(\pi) = 1$
- **约束**：$\gamma > 0$；$\gamma \to \infty$ 时 $q(\pi) \to \delta(\pi - \pi^*)$（确定性选择）；$\gamma \to 0$ 时 $q(\pi) \to 1/|\Pi|$（均匀随机选择）

> **白话翻译**：策略后验就是"各条路的选票"——EFE越低的路，得票越多。$\gamma$ 是"投票的认真程度"：$\gamma$ 大意味着只投最低EFE的路（确定性选择），$\gamma$ 小意味着随便投（随机选择）。$\gamma$ 不是超参数——它本身也可以被推断（学习"该多认真地选"）。

**策略后验与Boltzmann分布的关系**：A7.11的形式与统计力学中的Boltzmann分布完全同构——$G(\pi)$ 对应能量，$\gamma$ 对应逆温度，$q(\pi)$ 对应平衡态分布。最小化EFE等价于最小化"自由能"，策略后验是"自由能最低"的Boltzmann分布。

---

### 第六部分：计算近似

#### 2.12 蒙特卡洛近似（A7.12）

预期自由能的蒙特卡洛近似通过从变分分布中采样来估计期望：

$$G(\pi) \approx \frac{1}{K}\sum_{k=1}^{K} \left[\ln q(\psi_k) - \ln p(\psi_k, o_k | \pi)\right], \quad (\psi_k, o_k) \sim q(\psi, o|\pi) \tag{A7.12}$$

- **输入**：采样数 $K \in \mathbb{Z}_{> 0}$，变分联合分布 $q(\psi, o|\pi)$，生成模型 $p(\psi, o|\pi)$
- **输出**：$G(\pi)$ 的蒙特卡洛估计 $\hat{G}(\pi) \in \mathbb{R}$
- **约束**：$K > 0$；估计方差为 $O(1/K)$；采样 $(\psi_k, o_k)$ 独立同分布

> **白话翻译**：蒙特卡洛近似就是"采样算平均"——从变分分布中随机抽 $K$ 个样本，算每个样本的"账单"，再取平均。样本越多，估计越准，但计算越慢。这是最直接的近似方法，适用于任何分布形式。

**方差分析**：设 $X_k = \ln q(\psi_k) - \ln p(\psi_k, o_k | \pi)$，则 $\hat{G} = \frac{1}{K}\sum_k X_k$，$\text{Var}[\hat{G}] = \text{Var}[X] / K$。当 $\text{Var}[X]$ 较大时（如多模态分布），需要更多样本才能获得可靠估计。

#### 2.13 变分近似（A7.13）

预期自由能的变分近似将其分解为变分自由能与信息增益之和：

$$G(\pi) \approx F[q] + \mathbb{E}_{q(o|\pi)}\left[D_{\text{KL}}[q(\psi|o, \pi) \| q(\psi)]\right] \tag{A7.13}$$

即：

$$G(\pi) \approx F[q] + E_{\text{epi}}(\pi)$$

- **输入**：变分自由能 $F[q]$（引用阶段4，A4.4），认知价值 $E_{\text{epi}}(\pi)$（A7.5）
- **输出**：$G(\pi)$ 的变分近似
- **约束**：近似精度取决于变分分布 $q$ 与真实后验 $p(\psi|o)$ 的接近程度

> **白话翻译**：变分近似把预期自由能拆成"当前账单"（变分自由能）加"预期信息增益"（认知价值）。当前账单是已知的（阶段4已算），信息增益需要预估（看到新观测后信念会改变多少）。这个近似的好处是：不需要采样联合分布，只需要算变分自由能和信息增益。

**推导**：

由A7.1：

$$G(\pi) = \mathbb{E}_{q(\psi, o|\pi)}[\ln q(\psi) - \ln p(\psi, o|\pi)]$$

分解 $p(\psi, o|\pi) = p(o|\pi) p(\psi|o, \pi)$：

$$G(\pi) = \mathbb{E}_{q(\psi, o|\pi)}[\ln q(\psi) - \ln p(o|\pi) - \ln p(\psi|o, \pi)]$$

在变分近似 $p(\psi|o, \pi) \approx q(\psi|o, \pi)$ 下：

$$G(\pi) \approx \mathbb{E}_{q(\psi, o|\pi)}[\ln q(\psi) - \ln p(o|\pi) - \ln q(\psi|o, \pi)]$$

$$= \underbrace{\mathbb{E}_{q(\psi|o, \pi)}[\ln q(\psi) - \ln p(o|\pi) - \ln q(\psi|o, \pi)]}_{F[q] \text{ 的推广}} + \text{修正项}$$

在观测 $o$ 给定时，第一项退化为变分自由能 $F[q]$（引用阶段4，A4.4）。对 $o$ 取期望后：

$$G(\pi) \approx F[q] + E_{\text{epi}}(\pi)$$

$\blacksquare$

---

### 第七部分：决策论统一

#### 2.14 EFE与期望效用理论的关系（A7.14）

预期自由能与期望效用理论（Expected Utility Theory, EUT）有深刻的联系。在EUT中，决策者选择最大化期望效用的策略：

$$\pi_{\text{EUT}}^* = \arg\max_{\pi} \mathbb{E}_{p(o|\pi)}[U(o)]$$

其中 $U(o)$ 为效用函数。在EFE框架中：

$$\pi_{\text{EFE}}^* = \arg\min_{\pi} G(\pi) = \arg\min_{\pi}\left(E_{\text{ext}}(\pi) - E_{\text{epi}}(\pi)\right)$$

两者的关系为：

$$G(\pi) = \underbrace{-\mathbb{E}_{q(o|\pi)}[\ln p(o)]}_{\text{期望效用（负值）}} + \underbrace{H[q(o|\pi)] - I_q(\Psi; O | \pi)}_{\text{EFE额外项}} \tag{A7.14}$$

- **输入**：EFE分解（A7.7），效用函数 $U(o) = \ln p(o)$
- **输出**：EFE与EUT的定量关系
- **约束**：当 $H[q(o|\pi)] = 0$ 且 $I_q(\Psi; O | \pi) = 0$ 时，EFE退化为EUT

> **白话翻译**：期望效用理论只看"结果好不好"（效用），不管"过程学没学到"（信息）。EFE比EUT多了两项：观测的随机性（熵）和信息增益。当环境完全确定且没有可学的东西时，EFE = 期望效用（负值）。所以EFE是EUT的推广——在确定性的退化情形下回到EUT，但在一般情况下自动加入探索动力。

**EFE相对于EUT的优越性**：

1. **自动探索**：EUT需要额外机制（如Thompson采样、信息价值采样）来鼓励探索。EFE的认知价值项自动提供探索动力。
2. **不确定性感知**：EUT的效用函数 $U(o)$ 不区分"确定的好结果"和"可能好也可能坏的结果"。EFE通过熵项 $H[q(o|\pi)]$ 区分——确定性好的策略优于随机可能好的策略。
3. **信念依赖**：EUT的期望基于真实分布 $p(o|\pi)$（通常未知）。EFE的期望基于变分分布 $q(o|\pi)$（可计算），且认知价值鼓励减少信念与真实的差距。

#### 2.15 偏好分布的设定（A7.15）

偏好分布 $p(o)$ 编码了"什么样的观测是好的"。其设定方式为：

$$p(o) = p_{\text{prior}}(o) \tag{A7.15}$$

即偏好分布等于观测的先验分布。这一设定的物理依据：

- 先验分布 $p_{\text{prior}}(o)$ 编码了系统"期望看到的观测"——由生存相容集（引用阶段2，A2.2）和先验信念（引用阶段4，A4.4）共同决定。
- 当观测与先验一致时，$-\ln p(o)$ 最小，自由能最低（引用阶段4，A4.5），生存概率最高（引用阶段4，定理4.2）。
- 因此"偏好"不是人为设定的，而是从生存必要性和先验信念中自然推出的。

- **输入**：观测先验 $p_{\text{prior}}(o)$
- **输出**：偏好分布 $p(o) = p_{\text{prior}}(o)$
- **约束**：$p(o)$ 为合法概率分布；$p(o)$ 与生存相容集一致

> **白话翻译**：偏好分布就是"系统天生想看到的观测"——不是人手动设定的"奖励"，而是从生存需求和先验信念中自然推出的。系统偏好那些"让自由能最低的观测"，而自由能最低的观测恰好是先验最预期的观测。所以"偏好 = 先验"——系统想看到它预期会看到的东西，因为预期中的东西意味着安全。

**偏好学习**：当先验 $p_{\text{prior}}(o)$ 不精确时，可以通过经验数据更新偏好分布。偏好学习等价于先验更新——通过变分自由能最小化（引用阶段4，A4.4）自动调整先验，使偏好更准确地反映生存需求。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A7.1 | $G(\pi) = \mathbb{E}_q[\ln q(\psi) - \ln p(\psi, o\|\pi)]$ | $q(\psi, o\|\pi)$, $p(\psi, o\|\pi)$, $\pi$ | $G(\pi) \in \mathbb{R}$ | EFE全局唯一定义点 ⭐ |
| A7.2 | $G(\pi) = \mathbb{E}_q[\ln q(\psi) - \ln p(o\|\pi)] + \mathbb{E}_q[-\ln p(\psi\|o,\pi)]$ | 同A7.1 | 两项之和 | $p(\psi, o\|\pi) = p(o\|\pi)p(\psi\|o,\pi)$ |
| A7.3 | $E_{\text{ext}}(\pi) = -\mathbb{E}_{q(o\|\pi)}[\ln p(o\|\pi)]$ | $q(o\|\pi)$, $p(o\|\pi)$ | $E_{\text{ext}} \in \mathbb{R}_{\geq 0}$ | 外禀价值定义 |
| A7.4 | $E_{\text{ext}}(\pi) = D_{\text{KL}}[q(o\|\pi) \| p(o)] + H[q(o\|\pi)]$ | $q(o\|\pi)$, $p(o)$ | $D_{\text{KL}} + H$ | $D_{\text{KL}} \geq 0$；$H \geq 0$ |
| A7.5 | $E_{\text{epi}}(\pi) = \mathbb{E}_{q(o\|\pi)}[D_{\text{KL}}[q(\psi\|o,\pi) \| q(\psi)]] = I_q(\Psi;O\|\pi)$ | $q(\psi\|o,\pi)$, $q(\psi)$, $q(o\|\pi)$ | $E_{\text{epi}} \in \mathbb{R}_{\geq 0}$ | 认知价值 = 互信息 |
| A7.6 | $E_{\text{epi}} = I_q = \text{信息增益} = H[q(\psi)] - \mathbb{E}_{q(o\|\pi)}[H[q(\psi\|o,\pi)]]$ | 同A7.5 | 四种等价表述 | 信息论统一 |
| A7.7 | $G(\pi) = E_{\text{ext}}(\pi) - E_{\text{epi}}(\pi)$ | $E_{\text{ext}}$, $E_{\text{epi}}$ | $G(\pi) \in \mathbb{R}$ | EFE完整分解 ⭐ |
| A7.8 | $\pi^* = \arg\min G(\pi) = \arg\min(E_{\text{ext}} - E_{\text{epi}})$ | $G(\pi)$ 对所有 $\pi$ | $\pi^* \in \Pi$ | 策略选择规则 |
| A7.9 | 探索-利用 = EFE的两个分量 | EFE分解（A7.7） | 统一框架 | 不需要额外平衡机制 |
| A7.10 | $E_{\text{ext}}/E_{\text{epi}} \approx 70/30$ | 经验数据 | 经验比例 | **经验参数，非公理推导** |
| A7.11 | $q(\pi) = \sigma(-\gamma \cdot G(\pi))$ | $G(\pi)$, $\gamma > 0$ | $q(\pi) \in \Delta^{|\Pi|-1}$ | $\gamma \to \infty$：确定性；$\gamma \to 0$：均匀 |
| A7.12 | $G(\pi) \approx \frac{1}{K}\sum_k [\ln q(\psi_k) - \ln p(\psi_k, o_k\|\pi)]$ | $K$, $q(\psi, o\|\pi)$, $p(\psi, o\|\pi)$ | $\hat{G}(\pi) \in \mathbb{R}$ | 方差 $O(1/K)$ |
| A7.13 | $G(\pi) \approx F[q] + E_{\text{epi}}(\pi)$ | $F[q]$（A4.4）, $E_{\text{epi}}$（A7.5） | $G(\pi)$ 近似值 | 变分近似精度 |
| A7.14 | $G(\pi) = -\mathbb{E}_{q(o\|\pi)}[\ln p(o)] + H[q(o\|\pi)] - I_q(\Psi;O\|\pi)$ | EFE分解, 效用 $U(o)=\ln p(o)$ | EFE与EUT关系 | 确定性+无信息时退化为EUT |
| A7.15 | $p(o) = p_{\text{prior}}(o)$ | 观测先验 | 偏好分布 | 偏好 = 先验；与生存相容集一致 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field, replace
from typing import Optional
import numpy as np

# 引用阶段6的数据类
from stage_006 import ActiveInferenceState


# ============================================================
# 阶段7 数据类：预期自由能
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class ExpectedFreeEnergy:
    """预期自由能的完整描述。

    组合阶段6的ActiveInferenceState，在主动推断状态的基础上增加
    预期自由能的外禀价值-认知价值分解结构。

    Attributes:
        active_inference_state: 阶段6的主动推断状态（组合字段）
        extrinsic_value: 外禀价值 E_ext(π) = -E_q[ln p(o|π)]（A7.3）
        kl_divergence_term: 外禀价值的KL散度项 D_KL[q(o|π)||p(o)]（A7.4第一项）
        entropy_term: 外禀价值的熵项 H[q(o|π)]（A7.4第二项）
        epistemic_value: 认知价值 E_epi(π) = I_q(Ψ;O|π)（A7.5）
        prior_entropy: 先验熵 H[q(ψ)]（A7.6展开项）
        expected_posterior_entropy: 预期后验熵 E_q[H[q(ψ|o,π)]]（A7.6展开项）
        policy_precision: 策略精度 γ > 0（A7.11）
        total_efe: 预期自由能总值 G(π) = E_ext - E_epi（A7.7）

    Invariants:
        - extrinsic_value == kl_divergence_term + entropy_term
        - epistemic_value == prior_entropy - expected_posterior_entropy
        - total_efe == extrinsic_value - epistemic_value
        - kl_divergence_term >= 0
        - entropy_term >= 0
        - epistemic_value >= 0
        - policy_precision > 0
    """
    active_inference_state: ActiveInferenceState
    extrinsic_value: float
    kl_divergence_term: float
    entropy_term: float
    epistemic_value: float
    prior_entropy: float
    expected_posterior_entropy: float
    policy_precision: float
    total_efe: float

    def __post_init__(self):
        # 验证外禀价值分解一致性
        ext_sum = self.kl_divergence_term + self.entropy_term
        if not np.isclose(ext_sum, self.extrinsic_value, atol=1e-8):
            raise ValueError(
                f"外禀价值分解不一致: D_KL + H = {ext_sum}, "
                f"但 E_ext = {self.extrinsic_value}"
            )
        # 验证认知价值分解一致性
        epi_sum = self.prior_entropy - self.expected_posterior_entropy
        if not np.isclose(epi_sum, self.epistemic_value, atol=1e-8):
            raise ValueError(
                f"认知价值分解不一致: H[prior] - E[H[post]] = {epi_sum}, "
                f"但 E_epi = {self.epistemic_value}"
            )
        # 验证EFE完整分解
        efe_sum = self.extrinsic_value - self.epistemic_value
        if not np.isclose(efe_sum, self.total_efe, atol=1e-8):
            raise ValueError(
                f"EFE分解不一致: E_ext - E_epi = {efe_sum}, "
                f"但 G(π) = {self.total_efe}"
            )
        # 验证非负性
        if self.kl_divergence_term < -1e-10:
            raise ValueError(
                f"KL散度项必须 ≥ 0，收到 {self.kl_divergence_term}"
            )
        if self.entropy_term < -1e-10:
            raise ValueError(
                f"熵项必须 ≥ 0，收到 {self.entropy_term}"
            )
        if self.epistemic_value < -1e-10:
            raise ValueError(
                f"认知价值必须 ≥ 0，收到 {self.epistemic_value}"
            )
        # 验证策略精度
        if self.policy_precision <= 0:
            raise ValueError(
                f"策略精度 γ 必须 > 0，收到 {self.policy_precision}"
            )

    @property
    def exploration_exploitation_ratio(self) -> float:
        """探索-利用比 E_epi / E_ext（A7.10的经验比例参考）。"""
        if self.extrinsic_value < 1e-10:
            return float('inf')
        return self.epistemic_value / self.extrinsic_value

    @property
    def policy_posterior_log(self) -> float:
        """策略后验对数 ln q(π) ∝ -γ·G(π)（A7.11，未归一化）。"""
        return -self.policy_precision * self.total_efe

    @property
    def is_empirical_ratio(self) -> bool:
        """探索-利用比是否接近经验70/30比例（A7.10）。

        注意：70/30是经验参数，非公理推导结论。
        """
        ratio = self.exploration_exploitation_ratio
        return 0.2 < ratio < 0.6  # 宽松范围，约30/70到60/40


# ============================================================
# 阶段7 核心接口：预期自由能计算器
# ============================================================
class ExpectedFreeEnergyCalculator:
    """阶段7的核心计算接口。

    实现预期自由能的计算、分解、策略选择和计算近似。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - compute_efe 返回值满足 G(π) = E_ext - E_epi
        - E_ext = D_KL + H
        - E_epi = I_q(Ψ;O|π) >= 0
        - 策略后验 q(π) = σ(-γ·G(π)) 为合法概率分布
    """

    def compute_efe(
        self,
        active_inference_state: ActiveInferenceState,
        policy: int,
        q_observation_given_policy: callable,
        p_preference: callable,
        q_state_given_obs_policy: callable,
        q_state: callable,
        policy_precision: float
    ) -> ExpectedFreeEnergy:
        """计算预期自由能 G(π)（A7.1，全局唯一定义点）。

        Args:
            active_inference_state: 阶段6的主动推断状态（组合字段）
            policy: 策略索引 π
            q_observation_given_policy: 策略条件观测分布 q(o|π) 的采样/对数概率函数
            p_preference: 偏好分布 p(o) 的对数概率函数
            q_state_given_obs_policy: 变分后验 q(ψ|o,π) 的采样/对数概率函数
            q_state: 变分先验 q(ψ) 的对数概率函数
            policy_precision: 策略精度 γ > 0

        Returns:
            ExpectedFreeEnergy 实例，包含完整的EFE分解信息。

        Raises:
            ValueError: 当 policy_precision <= 0 时
            TypeError: 当 active_inference_state 不是 ActiveInferenceState 实例时

        Examples:
            >>> calc = ExpectedFreeEnergyCalculator()
            >>> # efe = calc.compute_efe(state, 0, q_o, p_pref, q_psi_o, q_psi, 1.0)
            >>> # doctest: +SKIP
        """
        if not isinstance(active_inference_state, ActiveInferenceState):
            raise TypeError(
                f"active_inference_state 必须为 ActiveInferenceState 实例，"
                f"收到 {type(active_inference_state).__name__}"
            )
        if policy_precision <= 0:
            raise ValueError(
                f"策略精度 γ 必须 > 0，收到 {policy_precision}"
            )

        # 计算外禀价值 E_ext(π)（A7.3/A7.4）
        e_ext, kl_div, entropy = self.compute_extrinsic_value(
            q_observation_given_policy, p_preference
        )

        # 计算认知价值 E_epi(π)（A7.5/A7.6）
        e_epi, h_prior, e_h_posterior = self.compute_epistemic_value(
            q_state_given_obs_policy, q_state, q_observation_given_policy
        )

        # A7.7: G(π) = E_ext(π) - E_epi(π)
        total_efe = e_ext - e_epi

        return ExpectedFreeEnergy(
            active_inference_state=active_inference_state,
            extrinsic_value=e_ext,
            kl_divergence_term=kl_div,
            entropy_term=entropy,
            epistemic_value=e_epi,
            prior_entropy=h_prior,
            expected_posterior_entropy=e_h_posterior,
            policy_precision=policy_precision,
            total_efe=total_efe,
        )

    def compute_extrinsic_value(
        self,
        q_observation_given_policy: callable,
        p_preference: callable,
        num_samples: int = 1000
    ) -> tuple[float, float, float]:
        """计算外禀价值 E_ext(π)（A7.3/A7.4）。

        Args:
            q_observation_given_policy: 策略条件观测分布 q(o|π)
            p_preference: 偏好分布 p(o)
            num_samples: 蒙特卡洛采样数

        Returns:
            (E_ext, D_KL, H) 元组
            - E_ext = D_KL[q(o|π)||p(o)] + H[q(o|π)]
            - D_KL >= 0
            - H >= 0

        Raises:
            ValueError: 当 num_samples <= 0 时

        Examples:
            >>> calc = ExpectedFreeEnergyCalculator()
            >>> # 高斯分布示例
            >>> # q_o = lambda: (np.random.randn(2), lambda x: -0.5*np.sum(x**2))
            >>> # e_ext, kl, h = calc.compute_extrinsic_value(q_o, p_pref, 1000)
            >>> # doctest: +SKIP
        """
        if num_samples <= 0:
            raise ValueError(f"采样数必须 > 0，收到 {num_samples}")

        # A7.3/A7.4: E_ext(π) = D_KL[q(o|π)||p(o)] + H[q(o|π)]
        # 蒙特卡洛估计
        log_q_sum = 0.0
        log_p_sum = 0.0
        for _ in range(num_samples):
            # q_observation_given_policy() 返回 (sample, log_prob) 元组
            o_sample, log_q_o = q_observation_given_policy()
            log_p_o = p_preference(o_sample)
            log_q_sum += log_q_o
            log_p_sum += log_p_o

        avg_log_q = log_q_sum / num_samples
        avg_log_p = log_p_sum / num_samples

        # H[q(o|π)] = -E_q[ln q(o|π)]
        entropy = -avg_log_q
        # D_KL[q(o|π)||p(o)] = E_q[ln q(o|π)/p(o)] = E_q[ln q(o|π)] - E_q[ln p(o)]
        kl_divergence = avg_log_q - avg_log_p
        # E_ext = D_KL + H
        e_ext = kl_divergence + entropy

        # 数值修正: KL散度不应为负
        kl_divergence = max(kl_divergence, 0.0)
        entropy = max(entropy, 0.0)
        e_ext = kl_divergence + entropy

        return (e_ext, kl_divergence, entropy)

    def compute_epistemic_value(
        self,
        q_state_given_obs_policy: callable,
        q_state: callable,
        q_observation_given_policy: callable,
        num_samples: int = 1000
    ) -> tuple[float, float, float]:
        """计算认知价值 E_epi(π)（A7.5/A7.6）。

        Args:
            q_state_given_obs_policy: 变分后验 q(ψ|o,π)
            q_state: 变分先验 q(ψ)
            q_observation_given_policy: 策略条件观测分布 q(o|π)
            num_samples: 蒙特卡洛采样数

        Returns:
            (E_epi, H_prior, E_H_posterior) 元组
            - E_epi = I_q(Ψ;O|π) = H[q(ψ)] - E_q[H[q(ψ|o,π)]]
            - H_prior = H[q(ψ)]
            - E_H_posterior = E_q[H[q(ψ|o,π)]]
            - E_epi >= 0

        Raises:
            ValueError: 当 num_samples <= 0 时

        Examples:
            >>> calc = ExpectedFreeEnergyCalculator()
            >>> # e_epi, h_prior, e_h_post = calc.compute_epistemic_value(
            >>> #     q_psi_o, q_psi, q_o, 1000)  # doctest: +SKIP
        """
        if num_samples <= 0:
            raise ValueError(f"采样数必须 > 0，收到 {num_samples}")

        # A7.5/A7.6: E_epi(π) = I_q(Ψ;O|π) = H[q(ψ)] - E_q(o|π)[H[q(ψ|o,π)]]
        # = E_q(o|π)[D_KL[q(ψ|o,π)||q(ψ)]]

        # 步骤1: 计算先验熵 H[q(ψ)]
        # q_state() 返回 (sample, log_prob) 元组
        log_q_prior_sum = 0.0
        for _ in range(num_samples):
            _, log_q_psi = q_state()
            log_q_prior_sum += log_q_psi
        h_prior = -log_q_prior_sum / num_samples

        # 步骤2: 计算预期后验熵 E_q(o|π)[H[q(ψ|o,π)]]
        # 和认知价值 E_epi = E_q[D_KL[q(ψ|o,π)||q(ψ)]]
        h_posterior_sum = 0.0
        kl_sum = 0.0
        for _ in range(num_samples):
            # 采样观测 o ~ q(o|π)
            o_sample, _ = q_observation_given_policy()
            # 计算后验熵 H[q(ψ|o,π)] 和 KL[q(ψ|o,π)||q(ψ)]
            # q_state_given_obs_policy(o) 返回 (sample, log_prob_posterior)
            _, log_q_posterior = q_state_given_obs_policy(o_sample)
            log_q_prior_at_psi = q_state()  # 重新采样先验用于KL估计

            # 使用蒙特卡洛估计后验熵
            inner_log_sum = 0.0
            inner_kl_sum = 0.0
            num_inner = max(1, num_samples // 10)
            for _ in range(num_inner):
                psi_sample, log_q_post = q_state_given_obs_policy(o_sample)
                _, log_q_pri = q_state()
                inner_log_sum += log_q_post
                inner_kl_sum += log_q_post - log_q_pri

            h_posterior_sample = -inner_log_sum / num_inner
            kl_sample = inner_kl_sum / num_inner

            h_posterior_sum += h_posterior_sample
            kl_sum += kl_sample

        e_h_posterior = h_posterior_sum / num_samples

        # A7.6: E_epi = H[q(ψ)] - E_q[H[q(ψ|o,π)]] = I_q(Ψ;O|π)
        e_epi = h_prior - e_h_posterior

        # 数值修正: 认知价值不应为负
        e_epi = max(e_epi, 0.0)

        return (e_epi, h_prior, e_h_posterior)

    def select_policy(
        self,
        efe_values: list[ExpectedFreeEnergy],
        policy_precision: float
    ) -> int:
        """选择最优策略 π* = argmin G(π)（A7.8）。

        Args:
            efe_values: 各策略的ExpectedFreeEnergy列表
            policy_precision: 策略精度 γ > 0

        Returns:
            最优策略索引 π*

        Raises:
            ValueError: 当 efe_values 为空或 policy_precision <= 0 时

        Examples:
            >>> calc = ExpectedFreeEnergyCalculator()
            >>> # pi_star = calc.select_policy([efe1, efe2, efe3], 1.0)
            >>> # doctest: +SKIP
        """
        if not efe_values:
            raise ValueError("策略EFE列表不能为空")
        if policy_precision <= 0:
            raise ValueError(
                f"策略精度 γ 必须 > 0，收到 {policy_precision}"
            )

        # A7.8: π* = argmin G(π)
        min_idx = 0
        min_efe = efe_values[0].total_efe
        for i, efe in enumerate(efe_values):
            if efe.total_efe < min_efe:
                min_efe = efe.total_efe
                min_idx = i
        return min_idx

    def compute_policy_posterior(
        self,
        efe_values: list[ExpectedFreeEnergy],
        policy_precision: float
    ) -> np.ndarray:
        """计算策略后验分布 q(π) = σ(-γ·G(π))（A7.11）。

        Args:
            efe_values: 各策略的ExpectedFreeEnergy列表
            policy_precision: 策略精度 γ > 0

        Returns:
            策略后验分布 q(π)，形状 (num_policies,)，满足 sum = 1

        Raises:
            ValueError: 当 efe_values 为空或 policy_precision <= 0 时

        Examples:
            >>> calc = ExpectedFreeEnergyCalculator()
            >>> # q_pi = calc.compute_policy_posterior([efe1, efe2], 1.0)
            >>> # doctest: +SKIP
        """
        if not efe_values:
            raise ValueError("策略EFE列表不能为空")
        if policy_precision <= 0:
            raise ValueError(
                f"策略精度 γ 必须 > 0，收到 {policy_precision}"
            )

        # A7.11: q(π) = σ(-γ·G(π)) = softmax(-γ·G(π))
        efe_array = np.array([efe.total_efe for efe in efe_values])
        # 数值稳定的 softmax: 减去最大值
        logits = -policy_precision * efe_array
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(logits_shifted)
        q_pi = exp_logits / np.sum(exp_logits)

        return q_pi

    def monte_carlo_efe(
        self,
        policy: int,
        q_joint: callable,
        p_joint: callable,
        num_samples: int = 1000
    ) -> float:
        """蒙特卡洛近似计算EFE（A7.12）。

        G(π) ≈ (1/K)Σ_k [ln q(ψ_k) - ln p(ψ_k, o_k|π)]

        Args:
            policy: 策略索引 π
            q_joint: 变分联合分布 q(ψ, o|π) 的采样和对数概率函数
            p_joint: 生成模型联合分布 p(ψ, o|π) 的对数概率函数
            num_samples: 采样数 K，必须 > 0

        Returns:
            EFE的蒙特卡洛估计

        Raises:
            ValueError: 当 num_samples <= 0 时

        Examples:
            >>> calc = ExpectedFreeEnergyCalculator()
            >>> # G_hat = calc.monte_carlo_efe(0, q_joint, p_joint, 1000)
            >>> # doctest: +SKIP
        """
        if num_samples <= 0:
            raise ValueError(f"采样数 K 必须 > 0，收到 {num_samples}")

        # A7.12: G(π) ≈ (1/K)Σ_k [ln q(ψ_k) - ln p(ψ_k, o_k|π)]
        total = 0.0
        for _ in range(num_samples):
            # q_joint() 返回 (psi_sample, o_sample, log_q_joint) 元组
            psi_k, o_k, log_q = q_joint()
            log_p = p_joint(psi_k, o_k)
            total += log_q - log_p

        return total / num_samples

    def variational_efe_approx(
        self,
        free_energy: float,
        epistemic_value: float
    ) -> float:
        """变分近似计算EFE（A7.13）。

        G(π) ≈ F[q] + E_epi(π)

        Args:
            free_energy: 变分自由能 F[q]（引用阶段4，A4.4）
            epistemic_value: 认知价值 E_epi(π)（A7.5）

        Returns:
            EFE的变分近似值

        Examples:
            >>> calc = ExpectedFreeEnergyCalculator()
            >>> calc.variational_efe_approx(5.0, 2.0)
            7.0
            >>> calc.variational_efe_approx(0.0, 0.0)
            0.0
        """
        # A7.13: G(π) ≈ F[q] + E_epi(π)
        if np.isnan(free_energy) or np.isnan(epistemic_value):
            raise ValueError("自由能和认知价值不能为 NaN")
        return free_energy + epistemic_value

    def efe_to_eut_relation(
        self,
        efe: ExpectedFreeEnergy
    ) -> tuple[float, float]:
        """计算EFE与期望效用理论的关系（A7.14）。

        Args:
            efe: ExpectedFreeEnergy实例

        Returns:
            (expected_utility, efe_extra) 元组
            - expected_utility = -E_q[ln p(o)]（期望效用，负值）
            - efe_extra = H[q(o|π)] - I_q(Ψ;O|π)（EFE额外项）

        Examples:
            >>> calc = ExpectedFreeEnergyCalculator()
            >>> # eu, extra = calc.efe_to_eut_relation(efe)  # doctest: +SKIP
        """
        # A7.14: G(π) = -E_q[ln p(o)] + H[q(o|π)] - I_q(Ψ;O|π)
        # 由 A7.4: -E_q[ln p(o)] = D_KL[q(o|π)||p(o)] + H[q(o|π)] = E_ext
        # 因此: expected_utility = E_ext = -E_q[ln p(o)]
        # efe_extra = H[q(o|π)] - I_q(Ψ;O|π) = entropy_term - epistemic_value
        # 验证: expected_utility + efe_extra = E_ext + H - E_epi = G(π) + H
        # 注意: 此分解不是 G = eu + extra，而是 G = eu - epi + H = eu + extra
        expected_utility = efe.extrinsic_value  # -E_q[ln p(o)] = E_ext
        efe_extra = efe.entropy_term - efe.epistemic_value

        return (expected_utility, efe_extra)

    def verify_efe_decomposition(
        self,
        efe: ExpectedFreeEnergy,
        tolerance: float = 1e-8
    ) -> bool:
        """验证EFE分解的一致性（A7.7 = A7.3 - A7.5）。

        Args:
            efe: ExpectedFreeEnergy实例
            tolerance: 数值容差

        Returns:
            True 如果 G(π) == E_ext - E_epi 在容差范围内一致，False 否则

        Examples:
            >>> calc = ExpectedFreeEnergyCalculator()
            >>> # calc.verify_efe_decomposition(efe, 1e-8)  # doctest: +SKIP
        """
        # A7.7: G(π) = E_ext(π) - E_epi(π)
        expected = efe.extrinsic_value - efe.epistemic_value
        return np.isclose(efe.total_efe, expected, atol=tolerance)

    def verify_policy_posterior_normalized(
        self,
        policy_posterior: np.ndarray,
        tolerance: float = 1e-8
    ) -> bool:
        """验证策略后验分布的归一化（A7.11）。

        Args:
            policy_posterior: 策略后验分布，形状 (num_policies,)
            tolerance: 数值容差

        Returns:
            True 如果 sum(q(π)) ≈ 1 且所有 q(π) >= 0，False 否则

        Examples:
            >>> calc = ExpectedFreeEnergyCalculator()
            >>> q_pi = np.array([0.3, 0.5, 0.2])
            >>> calc.verify_policy_posterior_normalized(q_pi)
            True
            >>> q_bad = np.array([0.3, 0.5, 0.3])
            >>> calc.verify_policy_posterior_normalized(q_bad)
            False
            >>> q_neg = np.array([-0.1, 0.6, 0.5])
            >>> calc.verify_policy_posterior_normalized(q_neg)
            False
        """
        if policy_posterior.ndim != 1:
            return False
        if len(policy_posterior) == 0:
            return False
        # A7.11: q(π) 为合法概率分布
        if np.any(policy_posterior < -tolerance):
            return False
        total = float(np.sum(policy_posterior))
        return np.isclose(total, 1.0, atol=tolerance)

    def decompose_efe(
        self,
        efe: ExpectedFreeEnergy
    ) -> dict:
        """将EFE分解为所有子分量（A7.7 = A7.3 - A7.5）。

        返回包含所有EFE分解分量的字典，便于分析和可视化。

        Args:
            efe: ExpectedFreeEnergy实例

        Returns:
            包含以下键的字典:
            - 'total_efe': G(π) = E_ext - E_epi（A7.7）
            - 'extrinsic_value': E_ext(π)（A7.3）
            - 'epistemic_value': E_epi(π)（A7.5）
            - 'kl_divergence': D_KL[q(o|π)||p(o)]（A7.4第一项）
            - 'entropy': H[q(o|π)]（A7.4第二项）
            - 'prior_entropy': H[q(ψ)]（A7.6展开项）
            - 'expected_posterior_entropy': E_q[H[q(ψ|o,π)]]（A7.6展开项）
            - 'exploration_exploitation_ratio': E_epi/E_ext（A7.10参考）

        Raises:
            TypeError: 当 efe 不是 ExpectedFreeEnergy 实例时

        Examples:
            >>> calc = ExpectedFreeEnergyCalculator()
            >>> # components = calc.decompose_efe(efe)  # doctest: +SKIP
        """
        if not isinstance(efe, ExpectedFreeEnergy):
            raise TypeError(
                f"efe 必须为 ExpectedFreeEnergy 实例，"
                f"收到 {type(efe).__name__}"
            )
        return {
            'total_efe': efe.total_efe,
            'extrinsic_value': efe.extrinsic_value,
            'epistemic_value': efe.epistemic_value,
            'kl_divergence': efe.kl_divergence_term,
            'entropy': efe.entropy_term,
            'prior_entropy': efe.prior_entropy,
            'expected_posterior_entropy': efe.expected_posterior_entropy,
            'exploration_exploitation_ratio': efe.exploration_exploitation_ratio,
        }

    def set_preference(
        self,
        preference_mean: np.ndarray,
        preference_cov: np.ndarray
    ) -> callable:
        """设定偏好分布 p(o)（A7.15）。

        偏好分布等于观测的先验分布: p(o) = p_prior(o)。
        此方法构造高斯偏好分布的对数概率函数。

        Args:
            preference_mean: 偏好分布均值 o_pref，形状 (d_o,)
            preference_cov: 偏好分布协方差 Σ_pref，形状 (d_o, d_o)，对称正定

        Returns:
            偏好分布的对数概率函数 p_preference(o) → float

        Raises:
            ValueError: 当 preference_cov 不正定或维度不匹配时

        Examples:
            >>> calc = ExpectedFreeEnergyCalculator()
            >>> mu_pref = np.array([1.0, 0.0])
            >>> Sigma_pref = np.eye(2)
            >>> log_p = calc.set_preference(mu_pref, Sigma_pref)
            >>> # log_p(np.array([1.0, 0.0]))  # 在偏好均值处对数概率最高
            >>> # doctest: +SKIP
        """
        d_o = preference_mean.shape[0]
        if preference_cov.shape != (d_o, d_o):
            raise ValueError(
                f"preference_cov 形状 {preference_cov.shape} "
                f"与均值维度 {d_o} 不匹配"
            )
        if not np.allclose(preference_cov, preference_cov.T):
            raise ValueError("preference_cov 必须对称")
        try:
            np.linalg.cholesky(preference_cov)
        except np.linalg.LinAlgError:
            raise ValueError("preference_cov 必须正定")

        # A7.15: p(o) = p_prior(o)，此处构造高斯偏好分布
        pref_prec = np.linalg.inv(preference_cov)
        _, logdet_pref = np.linalg.slogdet(preference_cov)
        const = -0.5 * d_o * np.log(2 * np.pi) - 0.5 * logdet_pref

        def p_preference(o: np.ndarray) -> float:
            """偏好分布的对数概率 ln p(o)。"""
            diff = o - preference_mean
            return const - 0.5 * float(diff @ pref_prec @ diff)

        return p_preference
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `ActiveInferenceState` | ← 阶段6 | 作为预期自由能的主动推断状态基础（组合字段） |
| **输入** | A6.4-A6.5（主动推断公理） | ← 阶段6 | 行动方程与感知-行动闭环，EFE从即时梯度推广到策略级规划 |
| **输入** | A4.4（变分自由能全局唯一定义） | ← 阶段4 | EFE推导的起点；变分近似（A7.13）中的 $F[q]$ |
| **输入** | A5.1-A5.18（自由能分解） | ← 阶段5 | 准确性-复杂性分解推广为外禀价值-认知价值分解 |
| **输入** | 定理6.1（主动行动生存优势） | ← 阶段6 | 策略选择的生存驱力理论基础 |
| **输入** | A2.2（生存相容集） | ← 阶段2 | 偏好分布 $p(o)$ 与生存相容集的对应（A7.15） |
| **输出** | `ExpectedFreeEnergy` | → 阶段8/9/10/18/29/31 | 包含ActiveInferenceState的EFE分解数据 |
| **输出** | A7.1（EFE全局唯一定义） | → 阶段8/9/10/18/29/31 | 预期自由能的全局唯一定义点 ⭐ |
| **输出** | A7.7（EFE完整分解） | → 阶段8/9/10 | 外禀价值-认知价值分解的理论基础 |
| **输出** | A7.8（策略选择规则） | → 阶段8/9/18 | 策略选择的全局唯一规则 |
| **输出** | A7.11（策略后验） | → 阶段8/9/18 | 策略后验的计算公式 |
| **输出** | A7.12-A7.13（计算近似） | → 阶段9/10 | EFE的数值计算方法 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 自动微分与采样 | EFE各分量（A7.3-A7.6）的自动微分验证；蒙特卡洛采样（A7.12）的向量化加速 |
| SciPy | 优化与数值积分 | 策略选择（A7.8）的数值优化；KL散度与熵的数值计算 |
| NumPyro | 变分推断与采样 | 变分分布 $q(\psi\|o, \pi)$ 的推断；蒙特卡洛EFE估计（A7.12）的采样引擎 |
| **pymdp** | **主动推断框架** | **本阶段新增**：pymdp提供预期自由能的参考实现，包括策略条件观测分布 $q(o\|\pi)$ 的计算、外禀价值与认知价值的分解、策略后验 $q(\pi) = \sigma(-\gamma G(\pi))$ 的计算，直接支撑A7.1-A7.11的实现与验证 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V7.1 | EFE定义 | $G(\pi) = \mathbb{E}_q[\ln q(\psi) - \ln p(\psi, o\|\pi)]$ 对所有合法输入成立 | 蒙特卡洛期望估计 + 解析对比 |
| V7.2 | EFE第一步分解 | A7.2的两项之和等于A7.1 | 数值一致性验证 |
| V7.3 | 外禀价值定义 | $E_{\text{ext}} = -\mathbb{E}_{q(o\|\pi)}[\ln p(o\|\pi)]$ 与蒙特卡洛估计一致 | 蒙特卡洛对比验证 |
| V7.4 | 外禀价值分解 | $E_{\text{ext}} = D_{\text{KL}} + H$ 对所有合法输入成立 | 数值一致性验证 |
| V7.5 | KL散度非负 | $D_{\text{KL}}[q(o\|\pi) \| p(o)] \geq 0$ 对所有策略成立 | 随机分布测试 |
| V7.6 | 认知价值定义 | $E_{\text{epi}} = \mathbb{E}_{q(o\|\pi)}[D_{\text{KL}}[q(\psi\|o,\pi) \| q(\psi)]]$ 与蒙特卡洛估计一致 | 蒙特卡洛对比验证 |
| V7.7 | 认知价值 = 互信息 | $E_{\text{epi}} = I_q(\Psi; O\|\pi)$ 对所有策略成立 | 互信息数值计算对比 |
| V7.8 | 认知价值 = 熵减 | $E_{\text{epi}} = H[q(\psi)] - \mathbb{E}_{q(o\|\pi)}[H[q(\psi\|o,\pi)]]$ 对所有策略成立 | 熵计算对比验证 |
| V7.9 | 认知价值非负 | $E_{\text{epi}} \geq 0$ 对所有策略成立 | 随机策略测试 |
| V7.10 | EFE完整分解 | $G(\pi) = E_{\text{ext}} - E_{\text{epi}}$ 对所有策略成立 | 数值一致性验证 |
| V7.11 | 策略选择规则 | $\pi^* = \arg\min G(\pi)$ 在非平凡策略空间中选出合理策略 | 合成环境测试 |
| V7.12 | 策略后验归一化 | $\sum_\pi q(\pi) = 1$ 且 $q(\pi) \geq 0$ 对所有 $\gamma > 0$ 成立 | 数值验证 |
| V7.13 | 策略后验极限 | $\gamma \to \infty$ 时 $q(\pi) \to \delta(\pi - \pi^*)$；$\gamma \to 0$ 时 $q(\pi) \to 1/\|\Pi\|$ | 极限行为数值验证 |
| V7.14 | 蒙特卡洛近似精度 | 蒙特卡洛EFE（A7.12）与解析EFE的相对误差 < 5%（$K \geq 10000$） | 大样本蒙特卡洛对比 |
| V7.15 | 变分近似精度 | 变分EFE（A7.13）与精确EFE的相对误差 < 10%（单模态后验） | 合成高斯后验测试 |
| V7.16 | EFE-EUT关系 | 确定性+无信息时 $G(\pi) \to -\mathbb{E}[\ln p(o)]$（退化为EUT） | 极限情形数值验证 |
| V7.17 | 偏好-先验一致性 | $p(o) = p_{\text{prior}}(o)$ 与生存相容集一致 | 解析验证 |
| V7.18 | ExpectedFreeEnergy组合 | `ExpectedFreeEnergy.active_inference_state` 类型为 `ActiveInferenceState` | 类型检查 |
| V7.19 | 非法输入拒绝 | $\gamma \leq 0$、KL散度为负、维度不匹配等抛出 `ValueError` | 异常测试 |
| V7.20 | pymdp集成验证 | 使用pymdp的EFE实现验证A7.1-A7.11的计算结果 | pymdp数值对比 |

---

> **阶段7完** | 下一步：阶段8将从本阶段的EFE定义（A7.1）和策略选择规则（A7.8）出发，展开规划推理与策略树的推导。
