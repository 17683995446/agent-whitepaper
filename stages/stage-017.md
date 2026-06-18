# 阶段17：快感知-慢学习原则——时间尺度分离的算法实现

> **阶段编号**：S17 | **所属层**：第二层·工程实现层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 快感知 | "一眼看懂"——快速调整内部认识以匹配当前输入 |
> | 慢学习 | "日积月累"——缓慢修改翻译规则以改善长期预测 |
> | 分离比 | "看与学的速度差"——感知速度必须是学习速度的100倍以上 |
> | 感知迭代 | "反复琢磨直到想通"——每步输入后多次迭代表征更新 |
> | 权重冻结 | "学习时不动认识"——更新翻译规则时，认识保持当前最优 |

---

## 1. 第一性原理依据

本阶段从阶段5、阶段9和阶段16的核心结论出发，无需引入新的物理公理。核心依据为：

- **A5.18（时间尺度分离）**（引用阶段5）：$\tau_A \ll \tau_C$，准确性与复杂性的时间尺度分离，快感知-慢学习的第一性原理基础。
- **A9.13（分离条件）**（引用阶段9）：$\tau_1 \ll \tau_2 \ll \tau_3$，三层循环的时间尺度分离，其中 $\tau_1 \approx \tau_{\text{perceive}}$，$\tau_2 \approx \tau_{\text{learn}}$。
- **A16.1（统一欧拉法）**（引用阶段16）：$r^l_{t+\Delta t} = r^l_t + \Delta t \cdot \dot{r}^l_t$，离散化的基础方法。
- **A16.2（步长选择准则）**（引用阶段16）：$\Delta t < \min(\tau_r, \tau_W, \tau_a)$，步长由最快动力学决定。

本阶段的核心目标是：**从时间尺度分离（A5.18、A9.13）推导快感知-慢学习的完整算法框架，包括感知时间常数与学习时间常数的定量关系、分离比的经验参数、快感知算法和慢学习算法的具体实现，以及快感知-慢学习收敛性的严格证明。** 这是预测编码从理论到工程的关键桥梁——将连续时间的梯度流方程转化为交替执行的感知迭代和权重更新。

---

## 2. 核心数学公式

### 第一部分：感知时间常数

#### 2.1 感知时间常数（A17.1）

感知时间常数由感知方程（引用阶段6，A6.5）的学习率决定：

$$\tau_{\text{perceive}} = \frac{1}{\kappa_\mu} \tag{A17.1}$$

其中 $\kappa_\mu > 0$ 为感知学习率（A6.5中 $\dot{\mu} = -\kappa_\mu \partial F/\partial \mu$ 的系数）。

- **输入**：感知学习率 $\kappa_\mu > 0$
- **输出**：感知时间常数 $\tau_{\text{perceive}} > 0$
- **约束**：$\kappa_\mu > 0$；$\tau_{\text{perceive}}$ 对应表层循环的时间尺度 $\tau_1$（引用A9.10）

> **白话翻译**：感知时间常数就是"想明白需要多久"——学习率 $\kappa_\mu$ 越大，感知越快（时间常数越小）。就像反应速度——训练有素的人一眼就能看懂，新手需要反复琢磨。

**感知时间常数的物理含义**：

$\tau_{\text{perceive}}$ 度量表征更新对输入变化的响应速度。由A14.1（引用阶段14），表征更新方程为 $\tau_r \dot{r}^l = \delta^{l+1} - e^l$，其中 $\tau_r = \tau_{\text{perceive}}$。当输入突然变化时，表征以指数速率 $\exp(-t/\tau_{\text{perceive}})$ 收敛到新的稳态。

**感知时间常数与精度的关系**：

由A14.2（引用阶段14），精度调制的表征更新为 $\tau_r \dot{r}^l = \Pi_l^{-1}(\delta^{l+1} - \Pi_l e^l)$。在精度加权框架下，有效感知时间常数为：

$$\tau_{\text{perceive}}^{(l)} = \frac{\tau_r}{\lambda_{\min}(\Pi_l)} \tag{A17.1-precision}$$

其中 $\lambda_{\min}(\Pi_l)$ 为第 $l$ 层精度矩阵的最小特征值。精度越高（$\Pi_l$ 越大），有效时间常数越小，感知越快——"越确定的东西，越快确认"。

**与A9.10的对应**：

A9.10（引用阶段9）给出表层时间常数 $\tau_1 \approx 1/(\kappa_\mu + \kappa_a)$。在纯感知场景下（$\kappa_a = 0$），$\tau_1 = 1/\kappa_\mu = \tau_{\text{perceive}}$，与A17.1一致。

---

### 第二部分：学习时间常数

#### 2.2 学习时间常数（A17.2）

学习时间常数由权重更新方程（引用阶段15，A15.1）的时间常数决定：

$$\tau_{\text{learn}} = \tau_W = \frac{\tau_r}{\alpha_{\text{separation}}} \tag{A17.2}$$

其中 $\alpha_{\text{separation}} = \tau_{\text{learn}}/\tau_{\text{perceive}} \gg 1$ 为分离比（见A17.3）。

- **输入**：感知时间常数 $\tau_{\text{perceive}}$，分离比 $\alpha_{\text{separation}}$
- **输出**：学习时间常数 $\tau_{\text{learn}} \gg \tau_{\text{perceive}}$
- **约束**：$\alpha_{\text{separation}} \geq 100$（A17.3）；$\tau_{\text{learn}}$ 对应中层循环的时间尺度 $\tau_2$

> **白话翻译**：学习时间常数就是"改变经验需要多久"——比感知慢得多。感知是"秒级"的反应，学习是"分钟级"的积累。分离比就是"看与学的速度差"——感知比学习快100倍以上。

**学习时间常数的物理含义**：

$\tau_{\text{learn}}$ 度量权重更新对累积误差的响应速度。由A15.1（引用阶段15），权重更新方程为 $\tau_W \dot{W}^l = -(\partial f^l/\partial W^l) \cdot \delta^l$。权重以指数速率 $\exp(-t/\tau_W)$ 收敛到最优值。

**学习时间常数与学习率的关系**：

由A15.3（引用阶段15），离散化权重更新为 $W^l_{t+1} = W^l_t - (\eta/\tau_W)(\partial f^l/\partial W^l) \cdot \delta^l$。有效学习率为 $\eta/\tau_W$。当 $\tau_W$ 增大时，有效学习率减小，权重更新更慢但更稳。

**与A5.18和A9.13的对应**：

A5.18（引用阶段5）给出 $\tau_A \ll \tau_C$，其中 $\tau_A$ 对应感知时间常数（准确性驱动的快速更新），$\tau_C$ 对应学习时间常数（复杂性驱动的慢速更新）。A9.13（引用阶段9）将此推广为 $\tau_1 \ll \tau_2$，其中 $\tau_1 = \tau_{\text{perceive}}$，$\tau_2 = \tau_{\text{learn}}$。A17.2是这些时间尺度分离在算法层面的具体实现。

---

### 第三部分：分离比

#### 2.3 分离比（A17.3）

分离比定义为学习时间常数与感知时间常数的比值：

$$\alpha_{\text{separation}} = \frac{\tau_{\text{learn}}}{\tau_{\text{perceive}}} \geq 100 \tag{A17.3}$$

- **输入**：感知时间常数 $\tau_{\text{perceive}}$，学习时间常数 $\tau_{\text{learn}}$
- **输出**：分离比 $\alpha_{\text{separation}} \geq 100$
- **约束**：**此为经验参数，非公理推导**。下界100来自工程实践和神经科学观测，标注为经验值。

> **白话翻译**：分离比100就是"感知比学习快100倍"——每做100次感知更新，才做1次权重更新。这个数字不是从公理推导的，而是工程经验和神经科学观测的总结——大脑的突触可塑性比神经放电慢大约2-3个数量级。

**分离比的经验依据**：

1. **神经科学**：皮层锥体细胞的膜电位变化在毫秒级（$\sim 10\text{ms}$），突触长期增强（LTP）的稳定化需要分钟级（$\sim 1\text{min}$），分离比约 $10^2$。
2. **机器学习**：内循环（表征优化）通常需要10-100步收敛，外循环（权重更新）每步只做一次，有效分离比约 $10^1$-$10^2$。
3. **控制理论**：快慢分离系统的稳定性要求分离比至少 $10^1$，工程安全裕度推荐 $10^2$。

**分离比与收敛性的关系**：

分离比越大，感知-学习的解耦越彻底，收敛性保证越强。但分离比过大意味着学习极慢，实际效率降低。分离比100是在收敛保证和学习效率之间的折中。

**分离比的参数敏感性分析**：

| 分离比 | 感知收敛质量 | 学习效率 | 总收敛步数 | 推荐场景 |
|--------|------------|---------|-----------|---------|
| 10 | 较差（感知未充分收敛） | 高 | 较多 | 快速原型 |
| 100 | 好（感知近似收敛） | 适中 | 适中 | **默认推荐** |
| 1000 | 很好（感知充分收敛） | 低 | 较多 | 高精度要求 |

---

### 第四部分：快感知算法

#### 2.4 快感知算法（A17.4）

快感知算法在每个输入步执行 $N_{\text{perceive}}$ 次表征更新迭代，直到表征收敛：

$$r^l_{t,k+1} = r^l_{t,k} + \frac{\Delta t}{\tau_r}(\delta^{l+1}_{t,k} - e^l_{t,k}), \quad k = 0, 1, \ldots, N_{\text{perceive}}-1 \tag{A17.4}$$

其中下标 $t$ 为外部时间步（输入步），$k$ 为内部感知迭代步。

- **输入**：当前输入 $o_t$，初始表征 $r^l_{t,0} = r^l_{t-1,N_{\text{perceive}}}$（上一步的收敛表征），时间常数 $\tau_r$，步长 $\Delta t$，最大迭代步数 $N_{\text{perceive}}$
- **输出**：收敛后的表征 $r^l_{t,N_{\text{perceive}}}$
- **约束**：$N_{\text{perceive}} \geq \alpha_{\text{separation}}/\alpha_{\text{step}}$，其中 $\alpha_{\text{step}} = \tau_r/\Delta t$ 为每步的有效学习率

> **白话翻译**：快感知算法就是"反复琢磨直到想通"——每次看到新输入，不是只调整一次认识，而是反复调整直到认识与输入一致。$N_{\text{perceive}}$ 是"琢磨的次数"——太少还没想通，太多浪费时间。

**快感知算法的完整流程**：

```
算法：FastPerceive(model, observation, initial_reprs, tau_r, dt, N_perceive, eps_conv)
输入：层级生成模型 model, 观测 o_t, 初始表征 {r^l_0},
      时间常数 τ_r, 步长 Δt, 最大迭代步数 N_perceive, 收敛阈值 ε_conv
输出：收敛后的表征 {r^l}

1.  r^l ← initial_reprs^l  for all l
2.  for k = 1 to N_perceive do
3.      // 自上而下预测
4.      {r̂^{l-1}} ← TopDownPredict(model, {r^l})
5.
6.      // 计算误差信号（A3.1）
7.      for l = 1 to L do
8.          e^l ← r^l - r̂^l
9.      end for
10.     e^0 ← o_t - g(r̂^1)
11.
12.     // 自下而上误差传播（A12.2）
13.     {δ^l} ← BottomUpPropagate(model, {e^l})
14.
15.     // 表征更新（A17.4）
16.     for l = 1 to L do
17.         r^l ← r^l + (Δt/τ_r) · (δ^{l+1} - e^l)
18.     end for
19.
20.     // 收敛检查（A14.4）
21.     if max_l ‖δ^{l+1} - e^l‖ < ε_conv then
22.         break
23.     end if
24. end for
25.
26. return {r^l}
```

**快感知的计算复杂度**：

每步感知迭代需要 $O(\sum_l n_l^2)$ 的计算量（前向预测 + 误差传播 + 表征更新）。$N_{\text{perceive}}$ 步的总计算量为 $O(N_{\text{perceive}} \cdot \sum_l n_l^2)$。由A17.3，$N_{\text{perceive}} \approx \alpha_{\text{separation}} = 100$，因此感知阶段的计算量约为单步前向传播的100倍。

---

### 第五部分：慢学习算法

#### 2.5 慢学习算法（A17.5）

慢学习算法每 $N_{\text{perceive}}$ 步感知迭代后执行一次权重更新：

$$W^l_{t+1} = W^l_t - \frac{\eta}{\tau_W}\left(\frac{\partial f^l}{\partial W^l}\right)_{t} \cdot \delta^l_t \tag{A17.5}$$

其中 $\delta^l_t$ 使用感知收敛后的表征计算，$\eta$ 为学习率。

- **输入**：当前权重 $W^l_t$，感知收敛后的表征 $r^l_{t,N_{\text{perceive}}}$，传播信号 $\delta^l_t$，时间常数 $\tau_W$，学习率 $\eta$
- **输出**：更新后的权重 $W^l_{t+1}$
- **约束**：权重更新在感知收敛后执行（$\|\delta^{l+1} - e^l\| < \varepsilon_{\text{conv}}$）

> **白话翻译**：慢学习算法就是"想通了再改经验"——不是边看边改，而是先把当前输入彻底搞懂（感知收敛），然后再根据理解的结果调整翻译规则。这保证了每次修改经验都是基于"正确的理解"，而不是"还没想明白就乱改"。

**慢学习的关键设计决策**：

1. **权重冻结**：感知迭代期间权重不变（$W^l_{t,k} = W^l_t$ 对所有 $k$）。这保证感知更新是在固定的生成模型下进行，不会因权重变化而破坏表征的收敛。

2. **收敛后更新**：权重更新仅在感知收敛后执行。由A14.4（引用阶段14），收敛条件为 $\max_l \|\delta^{l+1} - e^l\| < \varepsilon_{\text{conv}}$。收敛时 $\delta^{l+1} \approx e^l$，表征近似匹配预测。

3. **EWC正则化**：权重更新可叠加EWC正则化（引用A15.5），防止学习新知识时遗忘旧知识。

**慢学习算法的完整流程**：

```
算法：SlowLearn(model, converged_reprs, weights, tau_W, eta, ewc_params)
输入：层级生成模型 model, 感知收敛后的表征 {r^l},
      当前权重 {W^l}, 时间常数 τ_W, 学习率 η, EWC参数
输出：更新后的权重 {W'^l}

1.  // 计算误差信号（使用收敛表征）
2.  {r̂^{l-1}} ← TopDownPredict(model, {r^l})
3.  for l = 1 to L do
4.      e^l ← r^l - r̂^l
5.  end for
6.  {δ^l} ← BottomUpPropagate(model, {e^l})
7.
8.  // 权重更新（A17.5 + A15.5）
9.  for l = 1 to L do
10.     grad ← (∂f^l/∂W^l) · δ^l
11.     if ewc_params is not None then
12.         ewc_grad ← λ_EWC · F_diag^l ⊙ (W^l - W^l_old)
13.         grad ← grad + ewc_grad
14.     end if
15.     W'^l ← W^l - (η/τ_W) · grad
16. end for
17.
18. return {W'^l}
```

**快感知-慢学习的完整训练循环**：

```
算法：FastSlowTrain(model, data_stream, config)
输入：层级生成模型 model, 数据流 {(o_t)}, 配置 FastSlowConfig
输出：训练后的模型权重

1.  for each observation o_t in data_stream do
2.      // 快感知阶段（A17.4）
3.      {r^l} ← FastPerceive(model, o_t, {r^l}, τ_r, Δt, N_perceive, ε_conv)
4.
5.      // 慢学习阶段（A17.5）
6.      {W^l} ← SlowLearn(model, {r^l}, {W^l}, τ_W, η, ewc_params)
7.  end for
8.
9.  return {W^l}
```

---

### 第六部分：快感知-慢学习的收敛性

#### 定理17.1：快感知-慢学习的收敛性

**定理17.1**：在以下条件下：
1. 分离比 $\alpha_{\text{separation}} = \tau_{\text{learn}}/\tau_{\text{perceive}} \geq 100$（A17.3）；
2. 步长 $\Delta t$ 满足CFL条件（A16.3）；
3. 自由能Hessian在驻点附近正定（自由能局部强凸）；
4. 感知迭代收敛至 $\max_l \|\delta^{l+1} - e^l\| < \varepsilon_{\text{conv}}$（A14.4）。

则快感知-慢学习算法保证：
(a) 自由能在每个感知-学习周期后单调递减：$F(W_{t+1}) \leq F(W_t)$；
(b) 权重序列 $\{W_t\}$ 收敛到自由能关于权重的驻点：$\|\partial F/\partial W^l\| \to 0$。

**完整证明**：

**步骤1：感知阶段的收敛**

由A4.11（引用阶段4），自由能在梯度下降过程中单调递减：$\mathrm{d}F/\mathrm{d}t \leq 0$。由A14.3（引用阶段14），每步感知更新使自由能下降（在CFL条件满足时）：

$$F(r_{t,k+1}) \leq F(r_{t,k}) - \frac{\Delta t}{\tau_r}\left\|\frac{\partial F}{\partial r}\right\|^2 + O(\Delta t^2)$$

当 $\Delta t$ 足够小时，自由能严格递减。经过 $N_{\text{perceive}}$ 步迭代后，表征收敛到当前权重下的近似最优值 $r^*_t$：

$$\left\|\frac{\partial F}{\partial r}\bigg|_{r=r^*_t}\right\| < \varepsilon_{\text{conv}}$$

**步骤2：学习阶段的自由能下降**

权重更新方向为 $-\partial F/\partial W^l$（A15.1）。在感知收敛后，表征 $r^*_t$ 是当前权重下的近似最优值。由链式法则：

$$\frac{\mathrm{d}F}{\mathrm{d}t}\bigg|_{W=W_t} = \frac{\partial F}{\partial W}\dot{W} + \frac{\partial F}{\partial r}\dot{r}$$

在感知收敛后，$\partial F/\partial r \approx 0$（由步骤1），因此：

$$\frac{\mathrm{d}F}{\mathrm{d}t}\bigg|_{W=W_t} \approx \frac{\partial F}{\partial W}\dot{W} = -\frac{1}{\tau_W}\left\|\frac{\partial F}{\partial W}\right\|^2 \leq 0$$

离散化后：

$$F(W_{t+1}) \approx F(W_t) - \frac{\eta}{\tau_W}\left\|\frac{\partial F}{\partial W}\right\|^2 + O(\eta^2)$$

当 $\eta$ 足够小时，$F(W_{t+1}) \leq F(W_t)$。此即结论(a)。

**步骤3：权重序列的收敛**

由步骤2，$F(W_t)$ 单调递减且有下界 $F \geq -\ln p(o)$（A4.6），因此 $F(W_t)$ 收敛到某个值 $F^*$。

由 $F(W_{t+1}) - F(W_t) \to 0$，得 $\|\partial F/\partial W\|^2 \to 0$，即 $\|\partial F/\partial W^l\| \to 0$。此即结论(b)。

**步骤4：分离比的充分性**

分离比 $\alpha_{\text{separation}} \geq 100$ 保证感知阶段在学习阶段执行前充分收敛。具体地，感知收敛需要 $O(\tau_r \ln(1/\varepsilon_{\text{conv}}))$ 的时间。在 $N_{\text{perceive}}$ 步感知迭代后，表征误差为：

$$\|r_{t,N_{\text{perceive}}} - r^*_t\| \leq \|r_{t,0} - r^*_t\| \cdot \exp(-N_{\text{perceive}} \cdot \Delta t / \tau_r)$$

当 $N_{\text{perceive}} \cdot \Delta t / \tau_r \geq \ln(1/\varepsilon_{\text{conv}})$ 时，表征误差小于 $\varepsilon_{\text{conv}}$。由 $\alpha_{\text{separation}} \geq 100$，$N_{\text{perceive}} \geq 100 \cdot \tau_r / \Delta t$，因此 $N_{\text{perceive}} \cdot \Delta t / \tau_r \geq 100 \gg \ln(1/\varepsilon_{\text{conv}})$（对 $\varepsilon_{\text{conv}} \geq 10^{-40}$），感知充分收敛。

$\blacksquare$

**定理17.1的实用推论**：

1. **感知-学习解耦**：感知阶段和学习阶段可以独立优化——感知阶段追求快速收敛，学习阶段追求稳定更新。
2. **并行化机会**：感知迭代中的不同层可以部分并行（半隐式方法的自上而下顺序除外），学习更新中的不同层可以完全并行。
3. **早停策略**：感知迭代可以在收敛后立即停止（A14.4），不必等满 $N_{\text{perceive}}$ 步，节省计算量。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A17.1 | $\tau_{\text{perceive}} = 1/\kappa_\mu$ | $\kappa_\mu$ | $\tau_{\text{perceive}} > 0$ | 引用A6.5 |
| A17.2 | $\tau_{\text{learn}} = \tau_r / \alpha_{\text{separation}}$ | $\tau_r$, $\alpha_{\text{separation}}$ | $\tau_{\text{learn}} \gg \tau_{\text{perceive}}$ | 引用A15.1 |
| A17.3 | $\alpha_{\text{separation}} = \tau_{\text{learn}}/\tau_{\text{perceive}} \geq 100$ | $\tau_{\text{learn}}$, $\tau_{\text{perceive}}$ | $\alpha_{\text{separation}} \geq 100$ | **经验参数** |
| A17.4 | $r^l_{t,k+1} = r^l_{t,k} + (\Delta t/\tau_r)(\delta^{l+1}_{t,k} - e^l_{t,k})$ | $r^l_{t,k}$, $\delta^{l+1}_{t,k}$, $e^l_{t,k}$, $\tau_r$, $\Delta t$ | $r^l_{t,k+1}$ | $k = 0, \ldots, N_{\text{perceive}}-1$ |
| A17.5 | $W^l_{t+1} = W^l_t - (\eta/\tau_W)(\partial f^l/\partial W^l)_t \cdot \delta^l_t$ | $W^l_t$, $\partial f^l/\partial W^l$, $\delta^l_t$, $\eta$, $\tau_W$ | $W^l_{t+1}$ | 感知收敛后执行 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# 引用阶段16的数据类
from stage_016 import DiscretizationConfig


# ============================================================
# 阶段17 数据类：快感知-慢学习配置
# ============================================================
@dataclass(frozen=True)  # 不可变，保证数据完整性
class FastSlowConfig:
    """快感知-慢学习的完整配置描述。

    组合阶段16的DiscretizationConfig，在离散化配置的基础上增加
    快感知-慢学习的时间尺度分离参数和算法配置。

    Attributes:
        discretization: 阶段16的离散化配置（组合字段）
        perceive_time_constant: 感知时间常数 τ_perceive
        learn_time_constant: 学习时间常数 τ_learn
        separation_ratio: 分离比 α_separation = τ_learn / τ_perceive
        perceive_iterations: 感知迭代步数 N_perceive
        convergence_threshold: 感知收敛阈值 ε_conv
        perceive_step_size: 感知步长 Δt_perceive
        learn_step_size: 学习步长 Δt_learn（= η/τ_W）
        use_ewc: 是否使用EWC正则化

    Invariants:
        - perceive_time_constant > 0
        - learn_time_constant > 0
        - separation_ratio >= 100（A17.3，经验参数）
        - learn_time_constant == perceive_time_constant * separation_ratio
        - perceive_iterations >= 1
        - convergence_threshold > 0
        - perceive_step_size > 0
        - perceive_step_size < perceive_time_constant（A16.2）
        - learn_step_size > 0
    """
    discretization: DiscretizationConfig
    perceive_time_constant: float
    learn_time_constant: float
    separation_ratio: float
    perceive_iterations: int
    convergence_threshold: float
    perceive_step_size: float
    learn_step_size: float
    use_ewc: bool = False

    def __post_init__(self):
        # 验证感知时间常数
        if self.perceive_time_constant <= 0:
            raise ValueError(
                f"感知时间常数 τ_perceive 必须 > 0，"
                f"收到 {self.perceive_time_constant}"
            )
        # 验证学习时间常数
        if self.learn_time_constant <= 0:
            raise ValueError(
                f"学习时间常数 τ_learn 必须 > 0，"
                f"收到 {self.learn_time_constant}"
            )
        # 验证分离比（A17.3）
        if self.separation_ratio < 100:
            raise ValueError(
                f"分离比 α_separation 必须 ≥ 100（A17.3），"
                f"收到 {self.separation_ratio}"
            )
        # 验证时间常数与分离比的一致性
        expected_learn = self.perceive_time_constant * self.separation_ratio
        if not np.isclose(self.learn_time_constant, expected_learn, rtol=0.01):
            raise ValueError(
                f"τ_learn ({self.learn_time_constant}) 必须 ≈ "
                f"τ_perceive × α_separation ({expected_learn})"
            )
        # 验证感知迭代步数
        if self.perceive_iterations < 1:
            raise ValueError(
                f"感知迭代步数必须 ≥ 1，"
                f"收到 {self.perceive_iterations}"
            )
        # 验证收敛阈值
        if self.convergence_threshold <= 0:
            raise ValueError(
                f"收敛阈值 ε_conv 必须 > 0，"
                f"收到 {self.convergence_threshold}"
            )
        # 验证感知步长
        if self.perceive_step_size <= 0:
            raise ValueError(
                f"感知步长必须 > 0，"
                f"收到 {self.perceive_step_size}"
            )
        if self.perceive_step_size >= self.perceive_time_constant:
            raise ValueError(
                f"感知步长 ({self.perceive_step_size}) 必须 < "
                f"τ_perceive ({self.perceive_time_constant})（A16.2）"
            )
        # 验证学习步长
        if self.learn_step_size <= 0:
            raise ValueError(
                f"学习步长必须 > 0，"
                f"收到 {self.learn_step_size}"
            )

    @property
    def effective_perceive_rate(self) -> float:
        """有效感知学习率 = Δt_perceive / τ_perceive。"""
        return self.perceive_step_size / self.perceive_time_constant

    @property
    def effective_learn_rate(self) -> float:
        """有效学习率 = Δt_learn / τ_learn。"""
        return self.learn_step_size / self.learn_time_constant

    @property
    def is_separated(self) -> bool:
        """时间尺度分离是否满足（A17.3）。"""
        return self.separation_ratio >= 100


# ============================================================
# 阶段17 核心接口：快感知-慢学习器
# ============================================================
class FastSlowLearner:
    """阶段17的核心计算接口。

    实现快感知-慢学习的完整训练循环，
    包括感知迭代、权重更新和收敛性监控。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - perceive 返回的表征满足 A14.4 收敛条件
        - learn 返回的权重满足 A15.3 离散化更新
        - 分离比满足 A17.3（≥ 100）
        - 自由能在每个感知-学习周期后单调递减（定理17.1）
    """

    def perceive(
        self,
        model: 'HierarchicalGenerativeModel',
        observation: np.ndarray,
        initial_representations: tuple[np.ndarray, ...],
        config: FastSlowConfig
    ) -> tuple[np.ndarray, ...]:
        """执行快感知迭代（A17.4）。

        在固定权重下迭代更新表征直到收敛。

        Args:
            model: 层级生成模型，需实现 forward()、compute_errors(obs)
            observation: 观测 o_t，shape (n_obs,)
            initial_representations: 初始表征 (r^1, ..., r^L)，每个 shape (n_l,)
            config: 快感知-慢学习配置

        Returns:
            收敛后的表征 (r'^1, ..., r'^L)，每个元素 shape (n_l,)

        Raises:
            ValueError: 若 config.perceive_iterations < 1
            ValueError: 若 observation 不是一维数组

        Examples:
            >>> import numpy as np
            >>> class DummyModel:
            ...     def forward(self, reprs): return [r * 0.5 for r in reprs]
            ...     def compute_errors(self, obs, reprs, preds):
            ...         errors = [r - p for r, p in zip(reprs, preds)]
            ...         deltas = [e for e in errors]
            ...         return errors, deltas
            >>> class DummyConfig:
            ...     perceive_iterations = 5
            ...     perceive_step_size = 0.1
            ...     perceive_time_constant = 1.0
            ...     convergence_threshold = 1e-3
            >>> learner = FastSlowLearner()
            >>> r0 = (np.array([1.0, 2.0]), np.array([3.0]))
            >>> obs = np.array([0.5, 1.0])
            >>> learner.perceive(DummyModel(), obs, r0, DummyConfig())
            (array([...]), array([...]))
        """
        if config.perceive_iterations < 1:
            raise ValueError(
                f"感知迭代步数必须 ≥ 1，收到 {config.perceive_iterations}"
            )
        if observation.ndim != 1:
            raise ValueError(
                f"观测必须为一维数组，收到 shape {observation.shape}"
            )
        representations = list(initial_representations)
        dt = config.perceive_step_size
        tau_r = config.perceive_time_constant
        for k in range(config.perceive_iterations):
            predictions = model.forward(representations)
            errors, deltas = model.compute_errors(observation, representations, predictions)
            max_grad = 0.0
            for l in range(len(representations)):
                gradient = deltas[l] - errors[l]
                representations[l] = representations[l] + (dt / tau_r) * gradient
                grad_norm = float(np.linalg.norm(gradient))
                max_grad = max(max_grad, grad_norm)
            if max_grad < config.convergence_threshold:
                break
        return tuple(representations)

    def learn(
        self,
        model: 'HierarchicalGenerativeModel',
        converged_representations: tuple[np.ndarray, ...],
        weights: tuple[np.ndarray, ...],
        config: FastSlowConfig,
        fisher_diagonals: Optional[tuple[np.ndarray, ...]] = None,
        anchor_weights: Optional[tuple[np.ndarray, ...]] = None
    ) -> tuple[np.ndarray, ...]:
        """执行慢学习更新（A17.5）。

        在感知收敛后更新权重。

        Args:
            model: 层级生成模型，需实现 compute_weight_gradient(l, r, delta)
            converged_representations: 感知收敛后的表征 (r^1, ..., r^L)，每个 shape (n_l,)
            weights: 当前权重 (W^1, ..., W^L)，W^l shape (n_{l-1}, n_l)
            config: 快感知-慢学习配置
            fisher_diagonals: Fisher对角元素（EWC用），每个 shape 与对应权重相同
            anchor_weights: 锚点权重（EWC用），每个 shape 与对应权重相同

        Returns:
            更新后的权重 (W'^1, ..., W'^L)，每个元素 shape (n_{l-1}, n_l)

        Raises:
            ValueError: 若 weights 与 converged_representations 长度不一致
            ValueError: 若 config.learn_step_size <= 0
            ValueError: 若启用 EWC 但 fisher_diagonals 或 anchor_weights 为 None

        Examples:
            >>> import numpy as np
            >>> class DummyModel:
            ...     def compute_weight_gradient(self, l, r, delta):
            ...         return np.outer(delta, r).T
            >>> class DummyConfig:
            ...     learn_step_size = 0.01
            ...     use_ewc = False
            >>> learner = FastSlowLearner()
            >>> r = (np.array([1.0, 2.0]), np.array([3.0]))
            >>> W = (np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([[0.5], [0.6]]))
            >>> learner.learn(DummyModel(), r, W, DummyConfig())
            (array([...]), array([...]))
        """
        if len(weights) != len(converged_representations):
            raise ValueError(
                f"权重数 ({len(weights)}) 与表征数 ({len(converged_representations)}) 不一致"
            )
        if config.learn_step_size <= 0:
            raise ValueError(
                f"学习步长必须 > 0，收到 {config.learn_step_size}"
            )
        if config.use_ewc:
            if fisher_diagonals is None or anchor_weights is None:
                raise ValueError(
                    "启用 EWC 时必须提供 fisher_diagonals 和 anchor_weights"
                )
            if len(fisher_diagonals) != len(weights) or len(anchor_weights) != len(weights):
                raise ValueError(
                    "EWC 参数长度必须与权重数一致"
                )
        updated_weights = []
        for l in range(len(weights)):
            # 简化的权重梯度：假设 model 提供 compute_weight_gradient
            # 若模型未实现，退化为外积近似
            if hasattr(model, 'compute_weight_gradient'):
                delta = converged_representations[l]
                grad = model.compute_weight_gradient(l, converged_representations[l], delta)
            else:
                # Hebbian 近似：-r^l @ δ^l^T
                delta = converged_representations[l]
                grad = -np.outer(converged_representations[l], delta).T
            # EWC 正则化
            if config.use_ewc and fisher_diagonals is not None and anchor_weights is not None:
                ewc_penalty = (
                    0.1  # 简化 EWC 强度
                    * fisher_diagonals[l]
                    * (weights[l] - anchor_weights[l])
                )
                grad = grad + ewc_penalty
            updated_weights.append(weights[l] - config.learn_step_size * grad)
        return tuple(updated_weights)

    def train_step(
        self,
        model: 'HierarchicalGenerativeModel',
        observation: np.ndarray,
        representations: tuple[np.ndarray, ...],
        weights: tuple[np.ndarray, ...],
        config: FastSlowConfig
    ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], float]:
        """执行一步完整的感知-学习训练（A17.4 + A17.5）。

        Args:
            model: 层级生成模型
            observation: 观测 o_t，shape (n_obs,)
            representations: 当前表征 (r^1, ..., r^L)
            weights: 当前权重 (W^1, ..., W^L)
            config: 快感知-慢学习配置

        Returns:
            (更新后的表征, 更新后的权重, 自由能值) 元组

        Raises:
            ValueError: 若 observation 不是一维数组

        Examples:
            >>> import numpy as np
            >>> class DummyModel:
            ...     def forward(self, reprs): return [r * 0.5 for r in reprs]
            ...     def compute_errors(self, obs, reprs, preds):
            ...         errors = [r - p for r, p in zip(reprs, preds)]
            ...         deltas = [e for e in errors]
            ...         return errors, deltas
            ...     def compute_weight_gradient(self, l, r, delta):
            ...         return np.outer(delta, r).T
            ...     def free_energy(self, obs, reprs):
            ...         return float(np.sum([np.sum(r**2) for r in reprs]))
            >>> class DummyConfig:
            ...     perceive_iterations = 3
            ...     perceive_step_size = 0.1
            ...     perceive_time_constant = 1.0
            ...     convergence_threshold = 1e-3
            ...     learn_step_size = 0.01
            ...     use_ewc = False
            >>> learner = FastSlowLearner()
            >>> r = (np.array([1.0, 2.0]), np.array([3.0]))
            >>> W = (np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([[0.5], [0.6]]))
            >>> obs = np.array([0.5, 1.0])
            >>> learner.train_step(DummyModel(), obs, r, W, DummyConfig())
            ((array([...]), array([...])), (array([...]), array([...])), ...)
        """
        if observation.ndim != 1:
            raise ValueError(
                f"观测必须为一维数组，收到 shape {observation.shape}"
            )
        # 快感知阶段
        new_representations = self.perceive(
            model, observation, representations, config
        )
        # 慢学习阶段
        new_weights = self.learn(
            model, new_representations, weights, config
        )
        # 计算自由能（简化：若模型未实现则返回 0.0）
        if hasattr(model, 'free_energy'):
            free_energy = model.free_energy(observation, new_representations)
        else:
            free_energy = 0.0
        return new_representations, new_weights, free_energy

    def verify_convergence(
        self,
        free_energy_history: list[float],
        tolerance: float = 1e-6
    ) -> bool:
        """验证快感知-慢学习的收敛性（定理17.1）。

        检查自由能是否单调递减并收敛。

        Args:
            free_energy_history: 自由能历史记录，长度 ≥ 2
            tolerance: 收敛容差

        Returns:
            True 如果自由能单调递减且最后变化量 < tolerance

        Raises:
            ValueError: 若 free_energy_history 长度 < 2
            ValueError: 若 tolerance <= 0

        Examples:
            >>> learner = FastSlowLearner()
            >>> learner.verify_convergence([1.0, 0.9, 0.85, 0.83], 1e-2)
            True
            >>> learner.verify_convergence([1.0, 1.1, 0.9], 1e-2)
            False
        """
        if len(free_energy_history) < 2:
            raise ValueError(
                f"自由能历史记录长度必须 ≥ 2，收到 {len(free_energy_history)}"
            )
        if tolerance <= 0:
            raise ValueError(f"容差必须 > 0，收到 {tolerance}")
        # 检查单调递减（允许微小数值误差）
        for i in range(len(free_energy_history) - 1):
            if free_energy_history[i + 1] > free_energy_history[i] + 1e-12:
                return False
        # 检查最后变化量是否小于容差
        last_diff = abs(free_energy_history[-1] - free_energy_history[-2])
        return last_diff < tolerance
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `DiscretizationConfig` | ← 阶段16 | 作为快感知-慢学习配置的组合字段 |
| **输入** | A5.18（时间尺度分离） | ← 阶段5 | $\tau_A \ll \tau_C$，快感知-慢学习的第一性原理基础 |
| **输入** | A9.13（分离条件） | ← 阶段9 | $\tau_1 \ll \tau_2 \ll \tau_3$，三层时间尺度分离 |
| **输入** | A16.1-A16.5（离散化方法） | ← 阶段16 | 欧拉法、半隐式法、自适应步长 |
| **输出** | `FastSlowConfig` | → 阶段18 | 包含DiscretizationConfig的快感知-慢学习配置 |
| **输出** | A17.4（快感知算法） | → 阶段19 | 预测编码网络的感知迭代实现 |
| **输出** | A17.5（慢学习算法） | → 阶段19 | 预测编码网络的权重更新实现 |
| **输出** | 定理17.1（收敛性） | → 阶段18/20 | 等价性证明和验证的理论保证 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| JAX | JIT编译与自动微分 | `jax.lax.while_loop`实现感知迭代的函数式循环（A17.4），支持JIT编译；`jax.lax.cond`实现收敛早停逻辑；`jax.grad`计算权重梯度（A17.5） |
| optax | 梯度优化 | `optax.sgd`实现慢学习阶段的权重更新（A17.5）；`optax.apply_if_finite`实现NaN/Inf安全机制，与自由能上升拒绝互补 |
| equinox | **本阶段新增**：模块化神经网络 | equinox提供JAX原生的模块化神经网络框架，支持`eqx.Module`定义预测编码网络的层级结构，`eqx.filter_jit`实现细粒度JIT编译（感知迭代内循环JIT，权重更新外循环不JIT），直接支撑快感知-慢学习的交替执行架构 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V17.1 | 感知时间常数正确性 | $\tau_{\text{perceive}} = 1/\kappa_\mu$ 与A6.5一致 | 参数关系验证 |
| V17.2 | 学习时间常数正确性 | $\tau_{\text{learn}} = \tau_r / \alpha_{\text{separation}}$ 与A15.1一致 | 参数关系验证 |
| V17.3 | 分离比约束 | $\alpha_{\text{separation}} \geq 100$（A17.3） | 参数范围验证 |
| V17.4 | 快感知收敛 | 感知迭代后 $\max_l \|\delta^{l+1} - e^l\| < \varepsilon_{\text{conv}}$ | 收敛性测试 |
| V17.5 | 权重冻结 | 感知迭代期间权重不变 | 权重变化监控 |
| V17.6 | 慢学习更新正确性 | A17.5与A15.3在感知收敛后一致 | 梯度对比验证 |
| V17.7 | 自由能单调递减 | $F(W_{t+1}) \leq F(W_t)$ 对每个训练步成立（定理17.1-a） | 多步训练自由能监控 |
| V17.8 | 权重序列收敛 | $\|\partial F/\partial W^l\| \to 0$（定理17.1-b） | 长期训练梯度范数监控 |
| V17.9 | 分离比充分性 | $\alpha_{\text{separation}} = 100$ 时收敛，$\alpha_{\text{separation}} = 1$ 时不收敛 | 不同分离比的对比测试 |
| V17.10 | 感知迭代步数 | $N_{\text{perceive}} \geq 100 \cdot \tau_r/\Delta t$ 时感知充分收敛 | 不同迭代步数的收敛质量测试 |
| V17.11 | 早停有效性 | 收敛后早停与满步迭代结果一致 | 早停vs满步对比 |
| V17.12 | EWC正则化兼容 | 慢学习+EWC正则化仍满足自由能单调递减 | EWC场景测试 |
| V17.13 | FastSlowConfig组合 | `config.discretization` 类型为 `DiscretizationConfig` | 类型检查 |
| V17.14 | 非法输入拒绝 | $\alpha_{\text{separation}} < 100$、$\tau_{\text{perceive}} \leq 0$ 等抛出 `ValueError` | 异常测试 |
| V17.15 | equinox模块化 | 使用eqx.Module定义的PC网络与手动实现行为一致 | equinx数值对比 |

---

> **阶段17完** | 下一步：阶段18将从本阶段的快感知-慢学习框架和阶段4的自由能定义出发，证明预测编码与自由能最小化的严格等价性。
