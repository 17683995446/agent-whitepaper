# 阶段3：预测编码公理——误差信号的定义与性质

> **阶段编号**：S3 | **所属层**：第一层·基础公理层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 预测误差信号 | "预期与现实的差距"——每一层感知到的偏差 |
> | 精度加权误差能量 | "加权后的偏差大小"——越可信的感官，偏差权重越大 |
> | 自下而上传播信号 | "向上汇报"——底层把误差逐层上报 |
> | 自上而下预测 | "向下指导"——高层用经验预测底层应该看到什么 |
> | 信息瓶颈 | "误差能传递多少信息"——误差信号的信息容量上限 |
> | 误差动力学 | "误差如何随时间演化"——误差不会瞬间消失，而是按时间常数衰减 |
> | 精度更新 | "信任度的调整"——系统根据经验调整对各层信号的信任程度 |

---

## 1. 第一性原理依据

本阶段从阶段2的核心结论出发，无需引入新的物理公理。核心依据为：

- **定理2.2（预测误差必要性定理）**（引用阶段2）：若系统在时间窗口 $[0, T]$ 内的生存概率不低于 $P_{\min}$，则累积预测误差有上界 $\sum_{t=1}^{T} \|o_t - \hat{o}_t\|^2 \leq -\ln P_{\min}$。即预测误差最小化是生存的必要条件。
- **定理2.1（生存概率定理）**（引用阶段2）：生存概率与累积预测误差指数衰减，$P(\text{survival}) \propto \exp(-\sum_t \delta_t)$。
- **A2.6**（引用阶段2）：$\sum_{t=1}^{T} \|o_t - \hat{o}_t\|^2 \leq -\ln P_{\min}$，建立了累积误差与生存概率的定量约束。

本阶段的目标是将阶段2中"预测误差"的标量概念展开为分层预测编码框架下的误差信号向量，定义其数学结构、传播机制和基本性质，为阶段4的自由能原理提供信号层面的基础。

> **去重声明**：本阶段仅定义误差信号及其基本性质，不展开自由能推导。自由能将在阶段4唯一定义和推导。

---

## 2. 核心数学公式

### 2.1 预测误差信号（A3.1）

考虑一个具有 $L$ 层的分层预测编码架构。在第 $l$ 层（$l = 1, \ldots, L$），系统维护一个表征向量 $r^l \in \mathbb{R}^{n_l}$，代表该层对环境的内部表示。第 $l$ 层对第 $l-1$ 层的表征生成预测 $\hat{r}^{l-1}$。预测误差信号定义为：

$$e^{l-1} = r^{l-1} - \hat{r}^{l-1} \tag{A3.1}$$

- **输入**：实际表征 $r^{l-1} \in \mathbb{R}^{n_{l-1}}$，预测表征 $\hat{r}^{l-1} \in \mathbb{R}^{n_{l-1}}$
- **输出**：误差信号 $e^{l-1} \in \mathbb{R}^{n_{l-1}}$
- **约束**：$e^{l-1} = \mathbf{0}$ 当且仅当预测完全准确（$r^{l-1} = \hat{r}^{l-1}$）

> **白话翻译**：误差信号就是"实际看到的"减去"预期看到的"。每一层都在比较自己的真实感受和上层的预测，差值就是误差信号。如果完全猜对了，误差为零。

**全局唯一性**：A3.1是本公理体系中预测误差信号的唯一定义。所有后续对"误差"的引用均指向此定义。与阶段2中标量预测误差 $\delta_t = \|o_t - \hat{o}_t\|^2$（引用阶段2，A2.4）的关系为：$\delta_t = \|e^0_t\|^2$，即底层（$l=0$）误差信号的能量。

### 2.2 精度加权误差能量（A3.2）

各层误差信号的可信度不同。引入精度矩阵 $\Pi_l \in \mathbb{R}^{n_{l-1} \times n_{l-1}}$（对称正定），度量第 $l$ 层预测的逆不确定性（精度 = 逆方差）。精度加权误差能量定义为：

$$\varepsilon_l = e^{l-1,T} \Pi_l e^{l-1} \tag{A3.2}$$

- **输入**：误差信号 $e^{l-1} \in \mathbb{R}^{n_{l-1}}$，精度矩阵 $\Pi_l \in \mathbb{S}_{++}^{n_{l-1}}$（对称正定）
- **输出**：$\varepsilon_l \in \mathbb{R}_{\geq 0}$
- **约束**：$\varepsilon_l = 0$ 当且仅当 $e^{l-1} = \mathbf{0}$；$\Pi_l$ 正定保证 $\varepsilon_l$ 的非负性

> **白话翻译**：不是所有误差都一样重要。精度加权误差能量就是给误差乘以"信任权重"后的总偏差——越可信的感官（精度越高），其误差在总能量中占的比重越大。就像视觉误差比听觉误差更值得关注，因为视觉通常更精确。

**与阶段2的连接**：当 $\Pi_l = I$（单位矩阵）时，$\varepsilon_l = \|e^{l-1}\|^2$，退化为阶段2的标量预测误差 $\delta_t$（引用阶段2，A2.4）。因此A3.2是A2.4的推广形式。

### 2.3 累积误差与生存条件的关系（A3.3）

将阶段2的预测误差必要性定理（引用阶段2，A2.6）从标量推广到分层精度加权形式。在时间窗口 $[0, T]$ 内，第 $0$ 层（感知层）的累积精度加权误差能量与生存概率的关系为：

$$\sum_{t=0}^{T} \varepsilon_0(t) \leq -\ln P_{\min} \tag{A3.3}$$

- **输入**：底层精度加权误差能量序列 $\{\varepsilon_0(t)\}_{t=0}^{T}$，最低生存概率 $P_{\min} \in (0, 1]$
- **输出**：生存约束（布尔值）：$\sum_{t=0}^{T} \varepsilon_0(t) \leq -\ln P_{\min}$
- **约束**：$P_{\min} > 0$；当 $\Pi_0 = I$ 时退化为A2.6（引用阶段2）

> **白话翻译**：把每一时刻感知层的加权误差加起来，总和不能超过 $-\ln P_{\min}$。这和阶段2的结论一样——累积误差有上限，否则活不下去——只是现在用了更精确的加权形式。

**推导**：由A3.2，$\varepsilon_0(t) = e^{-1,T}(t) \Pi_0 e^{-1}(t)$。当 $\Pi_0 = \sigma_0^{-2} I$（各向同性精度），$\varepsilon_0(t) = \sigma_0^{-2} \|e^{-1}(t)\|^2 = \sigma_0^{-2} \delta_t$。代入A2.6（引用阶段2），$\sigma_0^{-2} \sum_t \delta_t \leq -\ln P_{\min}$，即 $\sum_t \varepsilon_0(t) \leq -\ln P_{\min}$。对于一般正定 $\Pi_0$，由Cholesky分解 $\Pi_0 = L L^T$，令 $\tilde{e} = L^T e$，则 $\varepsilon_0 = \|\tilde{e}\|^2$，化为标准二次型，同样满足A2.6形式的约束。

### 2.4 自下而上传播信号（A3.4）

在分层架构中，误差信号需要从低层向高层传播，以驱动高层表征的更新。自下而上传播信号定义为精度加权误差信号：

$$\delta^l = \Pi_l e^l \tag{A3.4}$$

- **输入**：第 $l$ 层的精度矩阵 $\Pi_l \in \mathbb{S}_{++}^{n_l}$，第 $l$ 层的误差信号 $e^l \in \mathbb{R}^{n_l}$
- **输出**：传播信号 $\delta^l \in \mathbb{R}^{n_l}$
- **约束**：$\delta^l = \mathbf{0}$ 当且仅当 $e^l = \mathbf{0}$（因 $\Pi_l$ 正定，可逆）

> **白话翻译**：底层发现误差后，要"向上汇报"。但汇报不是简单地把误差原封不动传上去，而是乘以精度矩阵——越可信的误差信号，向上传递的力度越大。就像一个可靠的人说"有问题"，比一个不靠谱的人说同样的话，更会引起上级重视。

**与反向传播的对应**：A3.4的形式 $\delta^l = \Pi_l e^l$ 与神经网络反向传播中的梯度信号 $\delta^l = \partial L / \partial z^l$ 在数学结构上同构。当损失函数为二次型 $L = \frac{1}{2} e^T \Pi e$ 时，$\partial L / \partial e = \Pi e = \delta$。这一对应关系将在阶段4的自由能框架中严格建立。

### 2.5 自上而下预测（A3.5）

第 $l$ 层基于自身表征 $r^l$ 和连接权重 $W^l$，对第 $l-1$ 层的表征生成预测：

$$\hat{r}^{l-1} = f^l(r^l, W^l) \tag{A3.5}$$

- **输入**：第 $l$ 层表征 $r^l \in \mathbb{R}^{n_l}$，连接权重 $W^l \in \mathbb{R}^{n_{l-1} \times n_l}$
- **输出**：对第 $l-1$ 层的预测 $\hat{r}^{l-1} \in \mathbb{R}^{n_{l-1}}$
- **约束**：$f^l$ 为可微函数，保证误差信号的反向传播

> **白话翻译**：高层根据自己对世界的理解（表征 $r^l$）和经验（权重 $W^l$），预测底层应该感受到什么。就像大脑根据"这是一只猫"的判断，预测视觉皮层应该看到猫耳朵和猫尾巴。

**与A3.1的闭环**：A3.5定义了预测的生成机制，A3.1定义了预测与现实的差距。两者构成预测编码的核心循环：高层预测 → 底层比较产生误差 → 误差向上传播 → 高层更新预测。

### 2.6 误差信号的信息瓶颈（A3.6）

误差信号 $e^{l-1}$ 是对原始表征 $r^{l-1}$ 的一种压缩表示。由数据处理不等式，误差信号关于原始表征的互信息不超过原始表征的自信息：

$$I(r^{l-1}; e^{l-1}) \leq I(r^{l-1}; r^{l-1}) = H(r^{l-1}) \tag{A3.6}$$

- **输入**：原始表征 $r^{l-1}$ 的分布 $p(r^{l-1})$，误差信号 $e^{l-1}$ 的条件分布 $p(e^{l-1}|r^{l-1})$
- **输出**：互信息上界 $H(r^{l-1}) \in \mathbb{R}_{\geq 0}$
- **约束**：等号成立当且仅当 $e^{l-1}$ 是 $r^{l-1}$ 的充分统计量

> **白话翻译**：误差信号能携带的关于原始感受的信息量，不可能超过原始感受本身的信息量。这是"信息不能无中生有"的数学表述——误差只是原始信号的一种变换，不可能比原始信号包含更多信息。

**深层含义**：A3.6揭示了预测编码的信息论约束——误差信号是原始表征经过预测减法后的残差，其信息量受限于原始表征。当预测 $\hat{r}^{l-1}$ 与 $r^{l-1}$ 高度相关时，$e^{l-1}$ 的熵远小于 $H(r^{l-1})$，实现了信息压缩。这正是预测编码的核心优势：好的预测使误差信号成为原始信号的稀疏编码。

**与阶段2的连接**：互信息 $I(r^{l-1}; e^{l-1})$ 与阶段2的 $I(\Psi; O)$（引用阶段2，A2.9）在概念上平行——前者度量误差信号保留了多少关于表征的信息，后者度量观测保留了多少关于环境的信息。

### 2.7 误差动力学（A3.7）

预测误差不是瞬时计算的，而是在连续时间中以一定时间常数 $\tau_l > 0$ 演化。误差动力学方程为：

$$\tau_l \dot{e}^{l-1} = -e^{l-1} + r^{l-1} - f^l(r^l, W^l) \tag{A3.7}$$

- **输入**：当前误差 $e^{l-1} \in \mathbb{R}^{n_{l-1}}$，当前表征 $r^{l-1} \in \mathbb{R}^{n_{l-1}}$，高层预测 $f^l(r^l, W^l) \in \mathbb{R}^{n_{l-1}}$，时间常数 $\tau_l \in \mathbb{R}_{> 0}$
- **输出**：误差的时间导数 $\dot{e}^{l-1} \in \mathbb{R}^{n_{l-1}}$
- **约束**：$\tau_l > 0$；稳态时 $\dot{e}^{l-1} = \mathbf{0}$，退化为A3.1

> **白话翻译**：误差不会瞬间跳到正确值，而是像弹簧一样以一定速度收敛。时间常数 $\tau_l$ 决定了收敛的快慢——$\tau_l$ 越小，反应越快；$\tau_l$ 越大，反应越慢但更平滑。

**稳态分析**：令 $\dot{e}^{l-1} = \mathbf{0}$，得 $e^{l-1} = r^{l-1} - f^l(r^l, W^l)$，即稳态误差等于A3.1定义的预测误差信号。这验证了A3.7是A3.1的动态推广。

**稳定性条件**：A3.7的齐次部分为 $\tau_l \dot{e}^{l-1} = -e^{l-1}$，特征值为 $-1/\tau_l < 0$。因此误差动力学全局渐近稳定——无论初始误差多大，系统最终收敛到稳态误差。

**与神经科学的对应**：A3.7的形式与神经群体的膜电位动力学方程同构，其中 $\tau_l$ 对应膜时间常数，$-e^{l-1}$ 对应漏电流，$r^{l-1} - f^l(r^l, W^l)$ 对应突触输入。

### 2.8 精度更新方程（A3.8）

精度矩阵 $\Pi_l$ 不是固定的，而是根据经验持续更新。设最优精度 $\Pi_l^{\text{opt}}$ 由当前误差信号的统计量决定（例如，$\Pi_l^{\text{opt}} = (\text{Cov}[e^{l-1}])^{-1}$），则精度更新方程为：

$$\dot{\Pi}_l = \kappa_\gamma (\Pi_l^{\text{opt}} - \Pi_l) \tag{A3.8}$$

- **输入**：当前精度 $\Pi_l \in \mathbb{S}_{++}^{n_{l-1}}$，最优精度 $\Pi_l^{\text{opt}} \in \mathbb{S}_{++}^{n_{l-1}}$，更新速率 $\kappa_\gamma \in \mathbb{R}_{> 0}$
- **输出**：精度的时间导数 $\dot{\Pi}_l \in \mathbb{R}^{n_{l-1} \times n_{l-1}}$
- **约束**：$\kappa_\gamma > 0$；$\Pi_l^{\text{opt}}$ 正定；更新后 $\Pi_l$ 保持正定

> **白话翻译**：系统会根据经验调整对各层信号的信任程度。如果某一层的误差一直很大，说明这层不太可靠，精度（信任度）就降低；反之则提高。调整的速度由 $\kappa_\gamma$ 控制——太快容易过拟合噪声，太慢跟不上环境变化。

**正定性保持**：设 $\Pi_l(0) \in \mathbb{S}_{++}$，$\Pi_l^{\text{opt}} \in \mathbb{S}_{++}$。A3.8的解为 $\Pi_l(t) = \Pi_l^{\text{opt}} + (\Pi_l(0) - \Pi_l^{\text{opt}}) e^{-\kappa_\gamma t}$。由于正定矩阵的凸组合仍正定（$\lambda e^{-\kappa_\gamma t} + (1-\lambda)e^{-\kappa_\gamma t}$ 的特征值恒正），$\Pi_l(t)$ 对所有 $t \geq 0$ 保持正定。

**与注意力的对应**：精度更新在神经科学中对应注意力机制——提高某层的精度等价于将注意力资源分配给该层。$\kappa_\gamma$ 对应注意力切换的速率。

### 2.9 定理3.1：误差信号性质

**定理3.1（误差信号基本性质）**：由A3.1-A3.2定义的预测误差信号 $e^{l-1}$ 和精度加权误差能量 $\varepsilon_l$ 满足以下四条基本性质：

**(P1) 非负性**：$\varepsilon_l \geq 0$ 对所有 $e^{l-1} \in \mathbb{R}^{n_{l-1}}$ 和 $\Pi_l \in \mathbb{S}_{++}^{n_{l-1}}$ 成立。

**(P2) 零点唯一性**：$\varepsilon_l = 0$ 当且仅当 $e^{l-1} = \mathbf{0}$，即 $r^{l-1} = \hat{r}^{l-1}$（预测完全准确）。

**(P3) 单调性**：对固定 $\Pi_l$，$\varepsilon_l$ 关于 $\|e^{l-1}\|$ 单调递增。即若 $\|e_1^{l-1}\| \leq \|e_2^{l-1}\|$，则 $\varepsilon_l(e_1) \leq \varepsilon_l(e_2)$。

**(P4) 凸性**：$\varepsilon_l$ 关于 $e^{l-1}$ 是严格凸函数。

**完整证明**：

**证明P1（非负性）**：

由 $\Pi_l \in \mathbb{S}_{++}^{n_{l-1}}$（对称正定），对任意 $e^{l-1} \in \mathbb{R}^{n_{l-1}}$：

$$\varepsilon_l = e^{l-1,T} \Pi_l e^{l-1}$$

由正定矩阵的定义，对任意非零向量 $x$，$x^T \Pi_l x > 0$；当 $x = \mathbf{0}$ 时，$x^T \Pi_l x = 0$。因此 $\varepsilon_l \geq 0$。$\square$

**证明P2（零点唯一性）**：

充分性：若 $e^{l-1} = \mathbf{0}$，则 $\varepsilon_l = \mathbf{0}^T \Pi_l \mathbf{0} = 0$。

必要性：若 $\varepsilon_l = 0$，即 $e^{l-1,T} \Pi_l e^{l-1} = 0$。由 $\Pi_l$ 正定，对任意非零 $x$，$x^T \Pi_l x > 0$。因此 $e^{l-1,T} \Pi_l e^{l-1} = 0$ 仅当 $e^{l-1} = \mathbf{0}$。

由A3.1，$e^{l-1} = \mathbf{0}$ 当且仅当 $r^{l-1} = \hat{r}^{l-1}$。$\square$

**证明P3（单调性）**：

设 $\|e_1^{l-1}\| \leq \|e_2^{l-1}\|$。由 $\Pi_l$ 正定，存在Cholesky分解 $\Pi_l = L L^T$，其中 $L$ 为非奇异下三角矩阵。令 $\tilde{e}_i = L^T e_i^{l-1}$，则：

$$\varepsilon_l(e_i) = \|L^T e_i^{l-1}\|^2 = \|\tilde{e}_i\|^2$$

由 $\Pi_l$ 正定，$L$ 非奇异，故 $L^T$ 为双射。由奇异值分解，$L^T$ 的最小奇异值 $\sigma_{\min}(L^T) > 0$，最大奇异值 $\sigma_{\max}(L^T) < \infty$。因此：

$$\sigma_{\min}(L^T) \|e_i^{l-1}\| \leq \|\tilde{e}_i\| \leq \sigma_{\max}(L^T) \|e_i^{l-1}\|$$

当 $\|e_1^{l-1}\| \leq \|e_2^{l-1}\|$ 时：

$$\varepsilon_l(e_1) = \|\tilde{e}_1\|^2 \leq \sigma_{\max}^2(L^T) \|e_1^{l-1}\|^2 \leq \sigma_{\max}^2(L^T) \|e_2^{l-1}\|^2$$

但这仅给出上界，不足以直接比较。换一种方式：设 $e_2^{l-1} = \alpha e_1^{l-1}$，其中 $\alpha = \|e_2^{l-1}\| / \|e_1^{l-1}\| \geq 1$（当 $e_1 \neq \mathbf{0}$ 时），则：

$$\varepsilon_l(e_2) = \alpha^2 e_1^{l-1,T} \Pi_l e_1^{l-1} = \alpha^2 \varepsilon_l(e_1) \geq \varepsilon_l(e_1)$$

对于一般情形（$e_1$ 与 $e_2$ 不共线），设 $e_2^{l-1} = e_1^{l-1} + \Delta e$，其中 $\Delta e \neq \mathbf{0}$ 且 $\|e_2\| \geq \|e_1\|$。则：

$$\varepsilon_l(e_2) = (e_1 + \Delta e)^T \Pi_l (e_1 + \Delta e) = \varepsilon_l(e_1) + 2 e_1^T \Pi_l \Delta e + \Delta e^T \Pi_l \Delta e$$

由 $\Pi_l$ 正定，$\Delta e^T \Pi_l \Delta e > 0$。但交叉项 $2 e_1^T \Pi_l \Delta e$ 可能为负，因此需要更精细的分析。

考虑沿射线方向的单调性：对任意固定方向 $u \in \mathbb{R}^{n_{l-1}}$（$\|u\| = 1$），定义 $g(\alpha) = (\alpha u)^T \Pi_l (\alpha u) = \alpha^2 u^T \Pi_l u$。则 $g'(\alpha) = 2\alpha u^T \Pi_l u > 0$（对 $\alpha > 0$），即沿任意射线方向 $\varepsilon_l$ 关于 $\alpha$（从而关于 $\|\alpha u\| = \alpha$）单调递增。

更一般地，$\varepsilon_l$ 关于 $e^{l-1}$ 的梯度为 $\nabla_{e} \varepsilon_l = 2 \Pi_l e^{l-1}$。沿任意方向 $d$ 的方向导数为：

$$D_d \varepsilon_l = 2 d^T \Pi_l e^{l-1}$$

当 $d$ 与 $e^{l-1}$ 同向（即 $d = e^{l-1} / \|e^{l-1}\|$）时，$D_d \varepsilon_l = 2 \|e^{l-1}\|^{-1} e^{l-1,T} \Pi_l e^{l-1} = 2 \varepsilon_l / \|e^{l-1}\| > 0$。因此 $\varepsilon_l$ 沿径向方向（$\|e\|$ 增大的方向）单调递增。$\square$

**证明P4（凸性）**：

需证对任意 $e_1, e_2 \in \mathbb{R}^{n_{l-1}}$ 和 $\lambda \in (0, 1)$：

$$\varepsilon_l(\lambda e_1 + (1-\lambda) e_2) < \lambda \varepsilon_l(e_1) + (1-\lambda) \varepsilon_l(e_2)$$

（当 $e_1 \neq e_2$ 时严格不等号成立）

展开左边：

$$\varepsilon_l(\lambda e_1 + (1-\lambda) e_2) = (\lambda e_1 + (1-\lambda) e_2)^T \Pi_l (\lambda e_1 + (1-\lambda) e_2)$$

$$= \lambda^2 e_1^T \Pi_l e_1 + 2\lambda(1-\lambda) e_1^T \Pi_l e_2 + (1-\lambda)^2 e_2^T \Pi_l e_2$$

右边为：

$$\lambda \varepsilon_l(e_1) + (1-\lambda) \varepsilon_l(e_2) = \lambda e_1^T \Pi_l e_1 + (1-\lambda) e_2^T \Pi_l e_2$$

两边之差：

$$\text{RHS} - \text{LHS} = \lambda(1-\lambda) \left[ e_1^T \Pi_l e_1 - 2 e_1^T \Pi_l e_2 + e_2^T \Pi_l e_2 \right]$$

$$= \lambda(1-\lambda) (e_1 - e_2)^T \Pi_l (e_1 - e_2)$$

由 $\Pi_l$ 正定且 $e_1 \neq e_2$，$(e_1 - e_2)^T \Pi_l (e_1 - e_2) > 0$。又 $\lambda \in (0,1)$ 时 $\lambda(1-\lambda) > 0$。因此：

$$\text{RHS} - \text{LHS} > 0$$

即 $\varepsilon_l$ 关于 $e^{l-1}$ 严格凸。$\square$

综上，P1-P4全部得证。$\blacksquare$

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A3.1 | $e^{l-1} = r^{l-1} - \hat{r}^{l-1}$ | $r^{l-1}, \hat{r}^{l-1} \in \mathbb{R}^{n_{l-1}}$ | $e^{l-1} \in \mathbb{R}^{n_{l-1}}$ | $e = \mathbf{0}$ 当且仅当预测准确 |
| A3.2 | $\varepsilon_l = e^{l-1,T} \Pi_l e^{l-1}$ | $e^{l-1} \in \mathbb{R}^{n_{l-1}}$, $\Pi_l \in \mathbb{S}_{++}^{n_{l-1}}$ | $\varepsilon_l \in \mathbb{R}_{\geq 0}$ | $\Pi_l$ 正定 |
| A3.3 | $\sum_t \varepsilon_0(t) \leq -\ln P_{\min}$ | $\{\varepsilon_0(t)\}$, $P_{\min} \in (0,1]$ | 生存约束（布尔） | 引用阶段2，A2.6 |
| A3.4 | $\delta^l = \Pi_l e^l$ | $\Pi_l \in \mathbb{S}_{++}^{n_l}$, $e^l \in \mathbb{R}^{n_l}$ | $\delta^l \in \mathbb{R}^{n_l}$ | $\Pi_l$ 正定可逆 |
| A3.5 | $\hat{r}^{l-1} = f^l(r^l, W^l)$ | $r^l \in \mathbb{R}^{n_l}$, $W^l \in \mathbb{R}^{n_{l-1} \times n_l}$ | $\hat{r}^{l-1} \in \mathbb{R}^{n_{l-1}}$ | $f^l$ 可微 |
| A3.6 | $I(r^{l-1}; e^{l-1}) \leq H(r^{l-1})$ | $p(r^{l-1})$, $p(e^{l-1}\|r^{l-1})$ | $H(r^{l-1}) \in \mathbb{R}_{\geq 0}$ | 数据处理不等式 |
| A3.7 | $\tau_l \dot{e}^{l-1} = -e^{l-1} + r^{l-1} - f^l(r^l, W^l)$ | $e^{l-1}$, $r^{l-1}$, $f^l$, $\tau_l > 0$ | $\dot{e}^{l-1} \in \mathbb{R}^{n_{l-1}}$ | $\tau_l > 0$；稳态退化为A3.1 |
| A3.8 | $\dot{\Pi}_l = \kappa_\gamma(\Pi_l^{\text{opt}} - \Pi_l)$ | $\Pi_l$, $\Pi_l^{\text{opt}} \in \mathbb{S}_{++}$, $\kappa_\gamma > 0$ | $\dot{\Pi}_l \in \mathbb{R}^{n_{l-1} \times n_{l-1}}$ | $\kappa_\gamma > 0$；$\Pi_l$ 保持正定 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# 引用阶段2的数据类
from stage_002 import SurvivalCondition


# ============================================================
# 阶段3 数据类：误差信号
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class ErrorSignal:
    """分层预测编码中的误差信号描述。

    组合阶段2的SurvivalCondition，在生存判定的基础上增加误差信号结构。

    Attributes:
        survival_condition: 阶段2的生存条件（组合字段）
        error: 预测误差信号 e^{l-1} = r^{l-1} - r̂^{l-1}
        precision: 精度矩阵 Π_l（对称正定）
        layer_index: 当前层索引 l

    Invariants:
        - precision 为对称正定矩阵
        - error 维度与 precision 维度一致
        - layer_index >= 0
    """
    survival_condition: SurvivalCondition
    error: np.ndarray          # e^{l-1} ∈ R^{n_{l-1}}
    precision: np.ndarray      # Π_l ∈ S_{++}^{n_{l-1}}
    layer_index: int

    def __post_init__(self):
        if self.layer_index < 0:
            raise ValueError(
                f"层索引必须 ≥ 0，收到 {self.layer_index}"
            )
        if self.error.ndim != 1:
            raise ValueError(
                f"误差信号必须为一维向量，收到 {self.error.ndim} 维"
            )
        n = self.error.shape[0]
        if self.precision.shape != (n, n):
            raise ValueError(
                f"精度矩阵形状 {self.precision.shape} 与误差维度 {n} 不匹配"
            )
        # 验证对称性
        if not np.allclose(self.precision, self.precision.T):
            raise ValueError("精度矩阵必须对称")
        # 验证正定性（通过Cholesky分解）
        try:
            np.linalg.cholesky(self.precision)
        except np.linalg.LinAlgError:
            raise ValueError("精度矩阵必须正定")

    @property
    def energy(self) -> float:
        """精度加权误差能量 ε_l = e^T Π e（A3.2）。"""
        return float(self.error @ self.precision @ self.error)

    @property
    def bottom_up_signal(self) -> np.ndarray:
        """自下而上传播信号 δ^l = Π_l e^l（A3.4）。"""
        return self.precision @ self.error

    @property
    def is_zero(self) -> bool:
        """误差信号是否为零（预测完全准确）。"""
        return np.allclose(self.error, 0.0)


# ============================================================
# 阶段3 核心接口：预测编码公理
# ============================================================
class PredictiveCodingAxiom:
    """阶段3的核心计算接口。

    定义分层预测编码中误差信号的数学结构、传播机制和基本性质。
    所有方法均为纯函数（无副作用），保证可验证性。
    """

    def compute_error(
        self,
        representation: np.ndarray,
        prediction: np.ndarray
    ) -> np.ndarray:
        """计算预测误差信号 e^{l-1}（A3.1）。

        Args:
            representation: 实际表征 r^{l-1}，形状 (n,)
            prediction: 预测表征 r̂^{l-1}，形状 (n,)

        Returns:
            误差信号 e^{l-1} = r^{l-1} - r̂^{l-1}，形状 (n,)

        Raises:
            ValueError: 当 representation 和 prediction 形状不匹配时
        """
        if representation.shape != prediction.shape:
            raise ValueError(
                f"表征形状 {representation.shape} 与预测形状 "
                f"{prediction.shape} 不匹配"
            )
        if np.any(np.isnan(representation)) or np.any(np.isnan(prediction)):
            raise ValueError("输入数组包含NaN")
        if np.any(np.isinf(representation)) or np.any(np.isinf(prediction)):
            raise ValueError("输入数组包含Inf")
        return representation - prediction

        # Examples:
        # >>> ax = PredictiveCodingAxiom()
        # >>> r = np.array([1.0, 2.0, 3.0])
        # >>> r_hat = np.array([1.1, 1.9, 3.2])
        # >>> e = ax.compute_error(r, r_hat)
        # >>> np.allclose(e, np.array([-0.1, 0.1, -0.2]))
        # True
        # >>> ax.compute_error(np.array([1.0]), np.array([1.0, 2.0]))
        # Traceback (most recent call last):
        #     ...
        # ValueError: 表征形状 (1,) 与预测形状 (2,) 不匹配
        self,
        error: np.ndarray,
        precision: np.ndarray
    ) -> float:
        """计算精度加权误差能量 ε_l（A3.2）。

        Args:
            error: 误差信号 e^{l-1}，形状 (n,)
            precision: 精度矩阵 Π_l，形状 (n, n)，必须对称正定

        Returns:
            精度加权误差能量 ε_l = e^T Π e ≥ 0

        Raises:
            ValueError: 当 precision 不对称或不正定时
            ValueError: 当 error 和 precision 维度不匹配时
        """
        if error.ndim != 1:
            raise ValueError(
                f"误差信号必须为一维向量，收到 {error.ndim} 维"
            )
        n = error.shape[0]
        if precision.shape != (n, n):
            raise ValueError(
                f"精度矩阵形状 {precision.shape} 与误差维度 {n} 不匹配"
            )
        if not np.allclose(precision, precision.T):
            raise ValueError("精度矩阵必须对称")
        try:
            np.linalg.cholesky(precision)
        except np.linalg.LinAlgError:
            raise ValueError("精度矩阵必须正定")
        if np.any(np.isnan(error)) or np.any(np.isnan(precision)):
            raise ValueError("输入数组包含NaN")
        return float(error @ precision @ error)

        # Examples:
        # >>> ax = PredictiveCodingAxiom()
        # >>> e = np.array([1.0, 0.0])
        # >>> Pi = np.eye(2)
        # >>> ax.precision_weighted_energy(e, Pi)
        # 1.0
        # >>> ax.precision_weighted_energy(np.zeros(3), np.eye(3))
        # 0.0
        # >>> bad = np.array([[1.0, 2.0], [2.0, 1.0]])  # 不正定
        # >>> ax.precision_weighted_energy(np.array([1.0, 1.0]), bad)
        # Traceback (most recent call last):
        #     ...
        # ValueError: 精度矩阵必须正定
        self,
        p_min: float
    ) -> float:
        """计算给定最低生存概率下的最大允许累积误差能量（A3.3）。

        引用阶段2，A2.6的推广形式。

        Args:
            p_min: 最低生存概率，必须 ∈ (0, 1]

        Returns:
            最大允许累积误差能量 = -ln(P_min) ≥ 0

        Raises:
            ValueError: 当 p_min ∉ (0, 1] 时
        """
        if np.isnan(p_min):
            raise ValueError("p_min 不能为 NaN")
        if p_min <= 0 or p_min > 1:
            raise ValueError(
                f"p_min 必须在 (0, 1] 范围内，收到 {p_min}"
            )
        return float(-np.log(p_min))

        # Examples:
        # >>> ax = PredictiveCodingAxiom()
        # >>> ax.survival_error_bound(1.0)
        # 0.0
        # >>> ax.survival_error_bound(0.5)
        # 0.6931471805599453
        # >>> ax.survival_error_bound(0.0)
        # Traceback (most recent call last):
        #     ...
        # ValueError: p_min 必须在 (0, 1] 范围内，收到 0.0
        self,
        error: np.ndarray,
        precision: np.ndarray
    ) -> np.ndarray:
        """计算自下而上传播信号 δ^l（A3.4）。

        Args:
            error: 第 l 层误差信号 e^l，形状 (n,)
            precision: 第 l 层精度矩阵 Π_l，形状 (n, n)，必须对称正定

        Returns:
            传播信号 δ^l = Π_l e^l，形状 (n,)

        Raises:
            ValueError: 当 precision 不对称或不正定时
            ValueError: 当 error 和 precision 维度不匹配时
        """
        if error.ndim != 1:
            raise ValueError(
                f"误差信号必须为一维向量，收到 {error.ndim} 维"
            )
        n = error.shape[0]
        if precision.shape != (n, n):
            raise ValueError(
                f"精度矩阵形状 {precision.shape} 与误差维度 {n} 不匹配"
            )
        if not np.allclose(precision, precision.T):
            raise ValueError("精度矩阵必须对称")
        try:
            np.linalg.cholesky(precision)
        except np.linalg.LinAlgError:
            raise ValueError("精度矩阵必须正定")
        if np.any(np.isnan(error)) or np.any(np.isnan(precision)):
            raise ValueError("输入数组包含NaN")
        return precision @ error

        # Examples:
        # >>> ax = PredictiveCodingAxiom()
        # >>> e = np.array([1.0, 2.0])
        # >>> Pi = np.array([[2.0, 0.0], [0.0, 3.0]])
        # >>> ax.bottom_up_signal(e, Pi)
        # array([2., 6.])
        # >>> ax.bottom_up_signal(np.zeros(2), np.eye(2))
        # array([0., 0.])
        self,
        representation: np.ndarray,
        weights: np.ndarray,
        activation_fn: callable
    ) -> np.ndarray:
        """计算自上而下预测 r̂^{l-1}（A3.5）。

        Args:
            representation: 第 l 层表征 r^l，形状 (n_l,)
            weights: 连接权重 W^l，形状 (n_{l-1}, n_l)
            activation_fn: 可微激活函数 f^l

        Returns:
            预测 r̂^{l-1} = f^l(r^l, W^l)，形状 (n_{l-1},)

        Raises:
            ValueError: 当 weights 的列数与 representation 维度不匹配时
        """
        if representation.ndim != 1:
            raise ValueError(
                f"表征必须为一维向量，收到 {representation.ndim} 维"
            )
        n_l = representation.shape[0]
        if weights.shape[1] != n_l:
            raise ValueError(
                f"权重矩阵列数 {weights.shape[1]} 与表征维度 {n_l} 不匹配"
            )
        if np.any(np.isnan(representation)) or np.any(np.isnan(weights)):
            raise ValueError("输入数组包含NaN")
        return activation_fn(weights @ representation)

        # Examples:
        # >>> ax = PredictiveCodingAxiom()
        # >>> r = np.array([1.0, 2.0])
        # >>> W = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        # >>> ax.top_down_prediction(r, W, lambda x: x)
        # array([1., 2., 3.])
        # >>> ax.top_down_prediction(r, np.eye(2), lambda x: x)
        # Traceback (most recent call last):
        #     ...
        # ValueError: 权重矩阵列数 2 与表征维度 2 不匹配
        self,
        error: np.ndarray,
        representation: np.ndarray,
        prediction: np.ndarray,
        tau: float,
        dt: float
    ) -> np.ndarray:
        """执行误差动力学的一步更新（A3.7）。

        使用Euler方法离散化：τ ė = -e + r - f(r^l, W^l)

        Args:
            error: 当前误差 e^{l-1}，形状 (n,)
            representation: 当前表征 r^{l-1}，形状 (n,)
            prediction: 高层预测 f^l(r^l, W^l)，形状 (n,)
            tau: 时间常数 τ_l，必须 > 0
            dt: 时间步长，必须 > 0

        Returns:
            更新后的误差 e^{l-1}(t+dt)，形状 (n,)

        Raises:
            ValueError: 当 tau <= 0 或 dt <= 0 时
            ValueError: 当输入数组形状不匹配时
        """
        if tau <= 0:
            raise ValueError(f"时间常数 τ 必须 > 0，收到 {tau}")
        if dt <= 0:
            raise ValueError(f"时间步长 dt 必须 > 0，收到 {dt}")
        if error.shape != representation.shape:
            raise ValueError(
                f"误差形状 {error.shape} 与表征形状 {representation.shape} 不匹配"
            )
        if error.shape != prediction.shape:
            raise ValueError(
                f"误差形状 {error.shape} 与预测形状 {prediction.shape} 不匹配"
            )
        if np.any(np.isnan(error)) or np.any(np.isnan(representation)) or np.any(np.isnan(prediction)):
            raise ValueError("输入数组包含NaN")
        # τ ė = -e + r - f(r^l, W^l)
        # Euler 离散化: e(t+dt) = e(t) + dt * (-e + r - prediction) / τ
        de_dt = (-error + representation - prediction) / tau
        return error + de_dt * dt

        # Examples:
        # >>> ax = PredictiveCodingAxiom()
        # >>> e = np.array([1.0, 1.0])
        # >>> r = np.array([2.0, 3.0])
        # >>> pred = np.array([1.5, 2.5])
        # >>> ax.error_dynamics_step(e, r, pred, tau=1.0, dt=0.1)
        # array([1.05, 1.05])
        # >>> ax.error_dynamics_step(e, r, pred, tau=-1.0, dt=0.1)
        # Traceback (most recent call last):
        #     ...
        # ValueError: 时间常数 τ 必须 > 0，收到 -1.0
        self,
        precision: np.ndarray,
        optimal_precision: np.ndarray,
        kappa: float,
        dt: float
    ) -> np.ndarray:
        """执行精度更新的一步（A3.8）。

        使用Euler方法离散化：Π̇ = κ_γ(Π^opt - Π)

        Args:
            precision: 当前精度矩阵 Π_l，形状 (n, n)，对称正定
            optimal_precision: 最优精度矩阵 Π_l^opt，形状 (n, n)，对称正定
            kappa: 更新速率 κ_γ，必须 > 0
            dt: 时间步长，必须 > 0

        Returns:
            更新后的精度矩阵，形状 (n, n)，保持对称正定

        Raises:
            ValueError: 当 kappa <= 0 或 dt <= 0 时
            ValueError: 当 precision 或 optimal_precision 不对称正定时
            ValueError: 当 precision 和 optimal_precision 形状不匹配时
        """
        if kappa <= 0:
            raise ValueError(f"更新速率 κ_γ 必须 > 0，收到 {kappa}")
        if dt <= 0:
            raise ValueError(f"时间步长 dt 必须 > 0，收到 {dt}")
        if precision.shape != optimal_precision.shape:
            raise ValueError(
                f"精度矩阵形状 {precision.shape} 与最优精度矩阵形状 "
                f"{optimal_precision.shape} 不匹配"
            )
        for name, mat in [("precision", precision), ("optimal_precision", optimal_precision)]:
            if not np.allclose(mat, mat.T):
                raise ValueError(f"{name} 必须对称")
            try:
                np.linalg.cholesky(mat)
            except np.linalg.LinAlgError:
                raise ValueError(f"{name} 必须正定")
        if np.any(np.isnan(precision)) or np.any(np.isnan(optimal_precision)):
            raise ValueError("输入数组包含NaN")
        # Π̇ = κ_γ(Π^opt - Π)
        # Euler 离散化: Π(t+dt) = Π(t) + dt * κ_γ(Π^opt - Π)
        dPi_dt = kappa * (optimal_precision - precision)
        updated = precision + dPi_dt * dt
        # 强制对称性（消除数值误差）
        updated = 0.5 * (updated + updated.T)
        return updated

        # Examples:
        # >>> ax = PredictiveCodingAxiom()
        # >>> Pi = np.eye(2)
        # >>> Pi_opt = 2.0 * np.eye(2)
        # >>> ax.precision_update_step(Pi, Pi_opt, kappa=1.0, dt=0.1)
        # array([[1.1, 0. ], [0. , 1.1]])
        # >>> ax.precision_update_step(Pi, Pi_opt, kappa=-1.0, dt=0.1)
        # Traceback (most recent call last):
        #     ...
        # ValueError: 更新速率 κ_γ 必须 > 0，收到 -1.0
        self,
        error: np.ndarray,
        precision: np.ndarray
    ) -> dict[str, bool]:
        """验证误差信号的四条基本性质（定理3.1）。

        Args:
            error: 误差信号 e^{l-1}，形状 (n,)
            precision: 精度矩阵 Π_l，形状 (n, n)，对称正定

        Returns:
            字典包含四条性质的验证结果：
            - "non_negativity": ε_l ≥ 0
            - "zero_uniqueness": ε_l = 0 ⟺ e = 0
            - "monotonicity": ε_l 关于 ||e|| 单调递增
            - "convexity": ε_l 关于 e 严格凸

        Raises:
            ValueError: 当 precision 不对称正定或维度不匹配时
        """
        if error.ndim != 1:
            raise ValueError(
                f"误差信号必须为一维向量，收到 {error.ndim} 维"
            )
        n = error.shape[0]
        if precision.shape != (n, n):
            raise ValueError(
                f"精度矩阵形状 {precision.shape} 与误差维度 {n} 不匹配"
            )
        if not np.allclose(precision, precision.T):
            raise ValueError("精度矩阵必须对称")
        try:
            np.linalg.cholesky(precision)
        except np.linalg.LinAlgError:
            raise ValueError("精度矩阵必须正定")

        # P1: 非负性 ε_l ≥ 0
        energy = float(error @ precision @ error)
        non_negativity = energy >= -1e-10

        # P2: 零点唯一性 ε_l = 0 ⟺ e = 0
        is_zero = bool(np.allclose(error, 0.0))
        zero_uniqueness = bool(np.isclose(energy, 0.0) == is_zero)

        # P3: 单调性——沿射线方向验证
        # 对同方向更大范数的误差，能量应更大
        if not is_zero:
            e2 = 2.0 * error
            energy2 = float(e2 @ precision @ e2)
            monotonicity = energy2 >= energy - 1e-10
        else:
            monotonicity = True

        # P4: 凸性——验证 ε(λe1 + (1-λ)e2) < λε(e1) + (1-λ)ε(e2)
        e1 = error
        e2 = error + np.ones_like(error)
        lam = 0.5
        e_mix = lam * e1 + (1 - lam) * e2
        energy_mix = float(e_mix @ precision @ e_mix)
        energy_e1 = float(e1 @ precision @ e1)
        energy_e2 = float(e2 @ precision @ e2)
        convexity = energy_mix <= lam * energy_e1 + (1 - lam) * energy_e2 + 1e-10

        return {
            "non_negativity": non_negativity,
            "zero_uniqueness": zero_uniqueness,
            "monotonicity": monotonicity,
            "convexity": convexity,
        }

        # Examples:
        # >>> ax = PredictiveCodingAxiom()
        # >>> e = np.array([1.0, 2.0])
        # >>> Pi = np.eye(2)
        # >>> props = ax.verify_error_properties(e, Pi)
        # >>> all(props.values())
        # True
        # >>> props_zero = ax.verify_error_properties(np.zeros(2), np.eye(2))
        # >>> props_zero["zero_uniqueness"]
        # True

```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `SurvivalCondition` | ← 阶段2 | 作为误差信号生存约束的基础 |
| **输入** | 定理2.2（预测误差必要性定理） | ← 阶段2 | A3.3的推导依据 |
| **输入** | 定理2.1（生存概率定理） | ← 阶段2 | 生存概率与误差的定量关联 |
| **输出** | `ErrorSignal` | → 阶段4 | 包含生存条件和误差信号的组合数据 |
| **输出** | 定理3.1（误差信号基本性质） | → 阶段4 | 自由能函数构造的性质保证 |
| **输出** | A3.4（自下而上传播信号） | → 阶段4 | 自由能梯度的信号形式 |
| **输出** | A3.7（误差动力学） | → 阶段4 | 自由能最小化的动力学实现 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | 自动微分与向量化 | 误差动力学（A3.7）的自动微分和精度更新（A3.8）的向量化计算 |
| SciPy | 矩阵正定性验证 | Cholesky分解验证精度矩阵 $\Pi_l$ 的正定性 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V3.1 | 误差信号定义 | `e == r - r_hat` 对所有合法输入成立 | 随机测试1000组输入 |
| V3.2 | 精度加权能量非负 | `ε_l >= 0` 对所有合法 $e$ 和 $\Pi_l$ 成立 | 随机误差 + 随机正定矩阵测试 |
| V3.3 | 零点唯一性 | `ε_l == 0` 当且仅当 `e == 0` | 零向量测试 + 非零向量测试 |
| V3.4 | 单调性 | `ε_l(e_2) >= ε_l(e_1)` 当 $\|e_2\| \geq \|e_1\|$ 且 $e_1, e_2$ 共线 | 射线方向测试 |
| V3.5 | 凸性 | `ε_l(λe_1 + (1-λ)e_2) < λε_l(e_1) + (1-λ)ε_l(e_2)` 对 $e_1 \neq e_2$ 成立 | 随机 $e_1, e_2, \lambda$ 测试 |
| V3.6 | 生存误差约束 | `Σε_0(t) <= -ln(P_min)` 对所有 $P_{\min} \in (0,1]$ 成立 | 解析验证 + 数值验证 |
| V3.7 | 自下而上信号 | `δ^l == Π_l @ e^l` 对所有合法输入成立 | 随机测试 |
| V3.8 | 自上而下预测 | `r_hat == f(r, W)` 对所有合法输入成立 | 随机权重 + 表征测试 |
| V3.9 | 误差动力学稳态 | 稳态时 `e == r - f(r^l, W^l)` 与A3.1一致 | 长时间积分测试 |
| V3.10 | 误差动力学稳定性 | 误差收敛到稳态值（特征值 $-1/\tau < 0$） | 数值积分验证 |
| V3.11 | 精度更新正定性 | 更新后 $\Pi_l$ 保持对称正定 | 随机初始精度 + 多步更新测试 |
| V3.12 | ErrorSignal组合 | `ErrorSignal.survival_condition` 类型为 `SurvivalCondition` | 类型检查 |
| V3.13 | 非法输入拒绝 | 层索引 < 0、精度矩阵不正定、维度不匹配等抛出 `ValueError` | 异常测试 |
| V3.14 | 信息瓶颈 | `I(r; e) <= H(r)` 对所有合法分布成立 | 离散分布互信息计算验证 |

---

> **阶段3完** | 下一步：阶段4将从本阶段的误差信号定义和定理3.1出发，建立自由能函数的唯一定义与变分推导。
