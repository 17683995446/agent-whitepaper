# 自驱永续进化预测修正Agent系统技术方案白皮书

## 第一部分：基础理论篇（阶段1-20）

---

> **核心主张**：智能系统的本质是预测机器——它通过持续最小化预测误差来维持自身存在，通过主动推断来改造环境使之符合预测，通过进化计算来优化自身结构使之适应更复杂的预测任务。本部分从第一性原理出发，逐步构建自驱永续进化预测修正Agent系统的完整理论基石。

---

## 阶段1：第一性原理推导——从物理世界到认知系统

### 1.1 从热力学第二定律出发

热力学第二定律告诉我们，一个孤立系统的熵（无序度）总是趋向增大：

$$\Delta S_{\text{isolated}} \geq 0$$

这意味着任何物理系统若不与外界进行能量和物质交换，终将走向热力学平衡——即最大熵状态，也就是"死亡"。生命系统之所以能够维持高度有序的低熵状态，是因为它不是孤立系统，而是开放系统。生命通过持续从环境中汲取自由能（低熵能量）并排出高熵废热，来抵抗自身的熵增：

$$\Delta S_{\text{system}} = \Delta S_{\text{internal}} + \Delta S_{\text{exchange}}$$

其中 $\Delta S_{\text{internal}} \geq 0$（内部熵产生不可逆），而 $\Delta S_{\text{exchange}} < 0$（通过与环境的负熵交换来抵消内部熵增）。

薛定谔在《生命是什么》中首次指出：**生命以负熵为食**。这一洞察的深层含义是——生命系统必须持续地、准确地预测环境中自由能的分布，才能有效地获取负熵。一个不能预测环境变化的生物体，将无法在熵增的洪流中维持自身的有序结构。

### 1.2 预测误差最小化推导智能的必要性

让我们将上述物理直觉形式化。设生命系统在时刻 $t$ 的状态为 $s_t$，环境状态为 $e_t$。系统需要维持自身状态在一个"生存相容"的集合 $\mathcal{S}_{\text{viable}}$ 内：

$$s_t \in \mathcal{S}_{\text{viable}}, \quad \forall t$$

为了做到这一点，系统必须对环境状态 $e_t$ 做出适当的响应 $a_t = \pi(s_t, e_t)$。但环境状态并非直接可观测——系统只能通过感觉输入 $o_t = g(e_t) + \eta$ 来间接推断环境（其中 $\eta$ 为噪声）。

因此，系统必须建立一个内部模型 $p(o_t | \theta)$ 来预测感觉输入。如果预测准确，系统就能提前做出适应性行为；如果预测失败，系统将面临生存风险。预测误差 $\|o_t - \hat{o}_t\|^2$ 直接关联到系统的生存概率：

$$P(\text{survival}) \propto \exp\left(-\sum_t \|o_t - \hat{o}_t\|^2\right)$$

这给出了一个根本性结论：**预测误差最小化不是智能系统的可选特征，而是生存的必要条件**。任何持续存在的系统，必然是在某种程度上成功最小化预测误差的系统——这被称为"选择性压力下的必然性"。

### 1.3 从预测到行动：主动推断的萌芽

仅有预测是不够的。被动预测者只能适应环境，无法改变环境。但生命系统可以通过行动改变感觉输入本身——这就是主动推断的核心思想：

$$o_t = g(e_t, a_t) + \eta$$

行动 $a_t$ 成为改变感觉输入的变量。系统不仅可以通过更新内部模型 $\theta$ 来减少预测误差（感知），还可以通过选择行动 $a_t$ 来使感觉输入更符合预测（行动）。这构成了感知-行动闭环：

$$\underbrace{\theta^* = \arg\min_\theta \|o - f(\theta)\|^2}_{\text{感知：修正模型}} \quad \text{与} \quad \underbrace{a^* = \arg\min_a \|g(e, a) - f(\theta^*)\|^2}_{\text{行动：改造输入}}$$

### 1.4 Agent系统的物理层基础假设

基于以上推导，我们建立Agent系统的三条基础假设：

**假设1（存在性假设）**：Agent系统必须持续最小化其预测误差（等价于最小化自由能），否则将退化为无序状态。

$$\frac{d}{dt}F(\theta, o) \leq 0$$

**假设2（主动性假设）**：Agent系统不仅通过更新内部模型来适应环境，还通过主动行动来改造环境，使环境更符合其预测。

$$a^* = \arg\min_a \mathbb{E}[F(\theta, o(a))]$$

**假设3（进化性假设）**：Agent系统的内部模型结构 $\theta$ 不是固定的，而是在选择压力下持续进化，以适应更复杂的预测任务。

$$\theta_{t+1} = \text{Evolve}(\theta_t, \text{fitness} = -F(\theta_t, o_t))$$

### 1.5 开源技术选型

| 技术需求 | 开源方案 | 说明 |
|---------|---------|------|
| 数值计算基础 | NumPy / JAX | JAX支持自动微分与GPU加速，适合大规模自由能计算 |
| 概率编程 | Pyro / NumPyro | 基于PyTorch/NumPy的贝叶斯推断框架 |
| 符号计算 | SymPy | 用于理论公式的符号推导与验证 |
| 仿真环境 | OpenAI Gym / PettingZoo | 提供标准化的Agent-环境交互接口 |

### 1.6 实现路径

1. **物理层建模**：使用JAX实现自由能计算核心，建立 $F(\theta, o)$ 的可微分计算图
2. **感知模块**：基于NumPyro实现变分推断，完成内部模型对观测的后验更新
3. **行动模块**：实现预期自由能最小化的策略选择，作为Agent的行动生成器
4. **进化模块**：基于(1+1)-ES实现模型参数的在线进化，以预测误差的负值为适应度
5. **闭环验证**：在Gym环境中验证感知-行动闭环的自稳性

---

## 阶段2：预测编码理论（Predictive Coding）数学基础

### 2.1 预测编码的核心思想

预测编码（Predictive Coding）是认知神经科学中最重要的计算理论之一，由Rao & Ballard（1999）系统形式化。其核心主张是：大脑不是一个被动接收信息的处理器，而是一个主动预测输入的生成模型。大脑只处理"预测误差"——即实际输入与预测输入之间的差异——而非输入本身。

这一思想有着深刻的信息论基础：如果环境具有统计规律性，那么大部分输入都可以被预测，只有不可预测的部分（预测误差）才需要被传输和处理。这是一种高效的信息压缩策略。

### 2.2 Rao & Ballard预测编码框架的数学形式化

考虑一个层级生成模型。设第 $l$ 层的表征为 $\mathbf{r}^l$，第 $l$ 层对第 $l-1$ 层的预测为：

$$\hat{\mathbf{r}}^{l-1} = f^l(\mathbf{r}^l, \mathbf{W}^l)$$

其中 $f^l$ 是第 $l$ 层的生成函数（通常取非线性变换），$\mathbf{W}^l$ 是生成权重矩阵。

第 $l-1$ 层的预测误差定义为：

$$\mathbf{e}^{l-1} = \mathbf{r}^{l-1} - \hat{\mathbf{r}}^{l-1} = \mathbf{r}^{l-1} - f^l(\mathbf{r}^l, \mathbf{W}^l)$$

整个层级模型的能量函数（预测误差的总和）为：

$$E = \sum_{l=0}^{L} \left\| \mathbf{r}^{l-1} - f^l(\mathbf{r}^l, \mathbf{W}^l) \right\|^2_{\Sigma_l^{-1}} + \sum_{l=1}^{L} \left\| \mathbf{r}^l \right\|^2_{\Lambda_l^{-1}}$$

其中第一项是各层预测误差的加权平方和（$\Sigma_l$ 是噪声协方差矩阵），第二项是表征的先验正则化项（$\Lambda_l$ 是先验协方差矩阵）。$\| \mathbf{x} \|_{\Sigma}^2 = \mathbf{x}^T \Sigma \mathbf{x}$ 是马氏距离。

注意：对于最底层 $l=0$，$\mathbf{r}^0 = \mathbf{x}$ 是感觉输入本身。

### 2.3 误差信号的多层推广

预测误差在层级中双向传播：

**自下而上（Bottom-up）**：预测误差 $\mathbf{e}^{l}$ 从低层传向高层，告知高层其预测的偏差：

$$\mathbf{e}^{l} = \mathbf{r}^{l} - f^{l+1}(\mathbf{r}^{l+1}, \mathbf{W}^{l+1})$$

**自上而下（Top-down）**：预测 $\hat{\mathbf{r}}^{l}$ 从高层传向低层，提供对低层表征的先验约束：

$$\hat{\mathbf{r}}^{l} = f^{l+1}(\mathbf{r}^{l+1}, \mathbf{W}^{l+1})$$

这种双向信息流形成了一种"协商"过程：高层不断修正其预测以匹配低层的表征，低层不断修正其表征以匹配高层的预测，最终达到一致——即预测误差最小化的不动点。

### 2.4 自由能函数的推导

从概率角度，预测编码可以理解为变分自由能的最小化。设生成模型为：

$$p(\mathbf{x}, \mathbf{r}) = p(\mathbf{x} | \mathbf{r}^0) \prod_{l=0}^{L-1} p(\mathbf{r}^l | \mathbf{r}^{l+1})$$

其中 $\mathbf{r} = (\mathbf{r}^0, \mathbf{r}^1, \ldots, \mathbf{r}^L)$ 是所有层级的表征。

系统需要推断后验 $p(\mathbf{r} | \mathbf{x})$，但由于高维积分不可解析计算，引入变分分布 $q(\mathbf{r})$ 来近似。变分自由能定义为：

$$F = -\ln p(\mathbf{x}) + D_{\text{KL}}[q(\mathbf{r}) \| p(\mathbf{r} | \mathbf{x})]$$

由于 $D_{\text{KL}} \geq 0$，有 $F \geq -\ln p(\mathbf{x})$，即变分自由能是负对数似然的上界。最小化自由能等价于最大化模型证据（边际似然）。

将自由能展开：

$$F = \underbrace{-\mathbb{E}_{q}[\ln p(\mathbf{x} | \mathbf{r})]}_{\text{预测误差项}} + \underbrace{D_{\text{KL}}[q(\mathbf{r}) \| p(\mathbf{r})]}_{\text{复杂性项}}$$

在假设高斯噪声和高斯先验的条件下，上式退化为：

$$F = \frac{1}{2}\sum_{l=0}^{L} \left\| \mathbf{r}^{l-1} - f^l(\mathbf{r}^l, \mathbf{W}^l) \right\|^2_{\Sigma_l^{-1}} + \frac{1}{2}\sum_{l=1}^{L} \left\| \mathbf{r}^l \right\|^2_{\Lambda_l^{-1}} + \text{const}$$

这与2.2节的能量函数完全一致，验证了预测编码与变分自由能最小化的等价性。

### 2.5 梯度下降修正规则

对能量函数 $E$ 分别关于表征 $\mathbf{r}^l$ 和权重 $\mathbf{W}^l$ 求梯度，得到预测编码的核心更新规则：

**表征更新（感知推断）**：

$$\dot{\mathbf{r}}^l = -\frac{\partial E}{\partial \mathbf{r}^l} = -\Sigma_l^{-1} \mathbf{e}^l + \frac{\partial f^{l}(\mathbf{r}^l, \mathbf{W}^l)^T}{\partial \mathbf{r}^l} \Sigma_{l-1}^{-1} \mathbf{e}^{l-1} - \Lambda_l^{-1} \mathbf{r}^l$$

其中第一项是本层预测误差的抑制信号，第二项是来自低层预测误差的驱动信号（经生成函数的Jacobian矩阵传播），第三项是先验正则化。

**权重更新（模型学习）**：

$$\dot{\mathbf{W}}^l = -\frac{\partial E}{\partial \mathbf{W}^l} = \frac{\partial f^{l}(\mathbf{r}^l, \mathbf{W}^l)^T}{\partial \mathbf{W}^l} \Sigma_{l-1}^{-1} \mathbf{e}^{l-1}$$

权重更新仅依赖于预测误差和生成函数对权重的梯度——这就是Hebb学习的推广形式：**连接强度的调整与突触前活动和突触后误差的乘积成正比**。

### 2.6 连续时间动力学的离散化

在实际实现中，连续时间微分方程需要离散化。使用欧拉法：

$$\mathbf{r}^l_{t+1} = \mathbf{r}^l_t - \kappa_r \frac{\partial E}{\partial \mathbf{r}^l}\bigg|_t$$

$$\mathbf{W}^l_{t+1} = \mathbf{W}^l_t - \kappa_W \frac{\partial E}{\partial \mathbf{W}^l}\bigg|_t$$

其中 $\kappa_r$ 和 $\kappa_W$ 分别是表征和权重的学习率。关键观察：**表征更新在每次观测后快速迭代（感知推断），而权重更新在多个观测上缓慢积累（模型学习）**——这对应了"快感知、慢学习"的认知原则。

### 2.7 开源技术选型

| 技术需求 | 开源方案 | 说明 |
|---------|---------|------|
| 预测编码框架 | predify (PyTorch) | 专用于预测编码层级网络的PyTorch库 |
| 自动微分 | JAX / PyTorch | 支持任意能量函数的自动梯度计算 |
| 变分推断 | Pyro / NumPyro | 用于变分自由能的优化 |
| 神经网络构建 | Haiku (JAX) / nn.Module (PyTorch) | 构建层级生成模型 |

### 2.8 实现路径

1. **层级生成模型**：使用PyTorch构建 $L$ 层生成网络，每层包含生成函数 $f^l$ 和误差计算单元
2. **双向传播**：实现自下而上的误差传播和自上而下的预测传播
3. **迭代推断**：在每次观测后，对表征进行 $K$ 步梯度下降迭代（$K$ 通常取10-50）
4. **权重学习**：在推断收敛后，执行一步权重更新
5. **验证**：在MNIST等数据集上验证预测编码的重构能力和稀疏表征特性

---

## 阶段3：自由能原理（Free Energy Principle）形式化

### 3.1 自由能原理的物理直觉

Karl Friston的自由能原理（Free Energy Principle, FEP）是21世纪认知科学最具影响力的理论框架之一。其核心主张可以用一句话概括：**任何与外界环境交互的自组织系统，必须最小化其自由能**。

这一原理的物理直觉来自统计力学。在统计力学中，自由能定义为：

$$F = U - TS$$

其中 $U$ 是内能，$T$ 是温度，$S$ 是熵。系统通过最小化自由能达到热力学平衡。Friston将这一思想推广到生物系统：生物系统不是走向最大熵（死亡），而是通过主动维持低自由能来远离平衡态——这正是生命的本质特征。

### 3.2 变分自由能的核心推导

设Agent的内部状态为 $s$，外部环境状态为 $\psi$，观测为 $o$，行动为 $a$。Agent无法直接观测 $\psi$，只能通过 $o$ 间接推断。

生成模型描述了观测如何由外部状态产生：

$$p(o, \psi) = p(o | \psi) p(\psi)$$

Agent需要推断后验 $p(\psi | o)$，但精确推断通常不可解。引入变分分布 $q(\psi)$ 近似后验，定义变分自由能：

$$F[q] = \underbrace{\mathbb{E}_{q(\psi)}[\ln q(\psi) - \ln p(o, \psi)]}_{\text{变分自由能}}$$

展开可得：

$$F[q] = \underbrace{D_{\text{KL}}[q(\psi) \| p(\psi | o)]}_{\geq 0} - \underbrace{\ln p(o)}_{\text{模型证据}}$$

由于KL散度非负，$F[q] \geq -\ln p(o)$。最小化自由能同时实现了两个目标：

1. **最小化KL散度**：使变分分布 $q(\psi)$ 逼近真实后验 $p(\psi | o)$——即精确感知
2. **最大化模型证据**：使生成模型 $p(o)$ 更好地解释观测——即模型选择

进一步展开变分自由能：

$$F[q] = \underbrace{-\mathbb{E}_{q(\psi)}[\ln p(o | \psi)]}_{\text{准确性（Accuracy）}} + \underbrace{D_{\text{KL}}[q(\psi) \| p(\psi)]}_{\text{复杂性（Complexity）}}$$

- **准确性项**：衡量模型对观测的拟合程度（预测误差的负对数似然）
- **复杂性项**：衡量后验偏离先验的程度（学习代价）

这揭示了自由能最小化的深层含义：**在拟合数据（准确性）与保持简单（复杂性）之间寻找最优平衡**——这正是贝叶斯奥卡姆剃刀的数学表述。

### 3.3 预期自由能

变分自由能描述的是Agent对当前观测的"惊讶程度"。但Agent还需要评估未来行动的后果——这就需要预期自由能（Expected Free Energy, EFE）。

给定策略 $\pi$（行动序列），预期自由能定义为对未来观测的变分自由能的期望：

$$G(\pi) = \mathbb{E}_{q(o, \psi | \pi)}[F[q]] = \mathbb{E}_{q(o, \psi | \pi)}[\ln q(\psi) - \ln p(o, \psi)]$$

将预期自由能展开为两个可解释的项：

$$G(\pi) = \underbrace{-\mathbb{E}_{q(o|\pi)}[\ln p(o)]}_{\text{外禀价值（Pragmatic）}} + \underbrace{\mathbb{E}_{q(o|\pi)}[D_{\text{KL}}[q(\psi|o,\pi) \| q(\psi|\pi)]]}_{\text{认知价值（Epistemic）}}$$

- **外禀价值**：期望观测与偏好分布的匹配程度——Agent倾向于到达其偏好的状态（如觅食、避险）
- **认知价值**：期望获得的信息增益——Agent倾向于选择能减少不确定性的行动（如探索、观察）

认知价值可以进一步写为互信息形式：

$$\text{认知价值} = \mathbb{E}_{q(o|\pi)}[D_{\text{KL}}[q(\psi|o,\pi) \| q(\psi|\pi)]] = I(\Psi; O | \pi)$$

这意味着**最优策略同时最大化外禀价值（满足偏好）和认知价值（获取信息）**——主动推断自然地实现了探索-利用的平衡。

### 3.4 从自由能最小化推导主动推断

主动推断的核心逻辑链：

1. Agent的目标是最小化变分自由能 $F$
2. 自由能依赖于观测 $o$，而观测依赖于行动 $a$
3. 因此Agent可以通过选择行动来控制自由能：$a^* = \arg\min_a F[q, o(a)]$
4. 但行动的效果需要在未来才能观测到，因此需要最小化预期自由能 $G(\pi)$

策略选择规则：

$$\pi^* = \arg\min_\pi G(\pi)$$

在离散时间、有限策略空间的设定下，策略的后验概率可以通过softmax计算：

$$q(\pi) = \sigma\left(-\gamma \cdot G(\pi)\right)$$

其中 $\gamma$ 是逆温度参数（控制策略选择的随机性），$\sigma$ 是softmax函数。

### 3.5 Agent系统如何实现自由能最小化闭环

完整的自由能最小化闭环包含三个相互耦合的过程：

**感知（Perception）**：更新变分分布 $q(\psi)$ 以最小化当前自由能：

$$q^*(\psi) = \arg\min_q F[q, o]$$

**学习（Learning）**：更新生成模型参数 $\theta$ 以最小化长期平均自由能：

$$\theta^* = \arg\min_\theta \mathbb{E}_{p(o)}[F[q, o; \theta]]$$

**行动（Action）**：选择行动以最小化预期自由能：

$$a^* = \arg\min_a G(a)$$

三者形成闭环：感知提供对当前状态的最优估计，学习优化内部模型以更好地预测未来，行动改变环境以使预测更易满足。

### 3.6 自由能的拉普拉斯近似

在实际计算中，变分分布 $q(\psi)$ 通常假设为高斯分布（拉普拉斯近似）：

$$q(\psi) = \mathcal{N}(\psi; \mu, \Sigma)$$

在此近似下，变分自由能的展开为：

$$F \approx \underbrace{-\ln p(o | \mu)}_{\text{准确性}} + \underbrace{\frac{1}{2} \text{tr}(\Sigma^{-1} \Sigma_0) + \frac{1}{2}(\mu - \mu_0)^T \Sigma_0^{-1} (\mu - \mu_0) + \text{const}}_{\text{复杂性}}$$

其中 $\mu_0, \Sigma_0$ 是先验参数。对 $\mu$ 和 $\Sigma$ 分别求极值，得到：

$$\mu^* = \arg\min_\mu F(\mu) \quad \text{（通过梯度下降迭代求解）}$$

$$\Sigma^* \approx H(\mu^*)^{-1} \quad \text{（Hessian矩阵的逆）}$$

其中 $H(\mu)$ 是自由能关于 $\mu$ 的Hessian矩阵。这表明**后验不确定性由自由能景观的曲率决定**——在自由能陡峭的区域（确定性高的预测），后验不确定性小；在自由能平坦的区域（模糊的预测），后验不确定性大。

### 3.7 开源技术选型

| 技术需求 | 开源方案 | 说明 |
|---------|---------|------|
| 自由能计算 | pymdp | Friston实验室官方主动推断Python库 |
| 变分推断 | Pyro / NumPyro | 灵活的变分推断框架 |
| 拉普拉斯近似 | JAX (自定义) | 利用JAX的Hessian自动计算 |
| 生成模型构建 | PyTorch / JAX | 构建参数化生成模型 |

### 3.8 实现路径

1. **变分自由能模块**：使用JAX实现拉普拉斯近似下的自由能计算，支持自动Hessian计算
2. **感知推断**：实现基于梯度下降的变分分布更新，支持在线迭代
3. **预期自由能计算**：实现EFE的两项分解（外禀价值+认知价值），支持策略评估
4. **闭环集成**：将感知、学习、行动三个模块集成为完整的自由能最小化闭环
5. **验证**：在T-Maze等经典主动推断任务中验证系统的探索-利用平衡行为

---

## 阶段4：主动推断（Active Inference）框架

### 4.1 感知-行动闭环的数学描述

主动推断是自由能原理的行动扩展——它不仅解释了Agent如何感知世界，还解释了Agent如何在世界中行动。其核心方程组描述了感知-行动闭环的完整动力学：

**感知方程（状态推断）**：

$$\dot{\mu} = -\kappa_\mu \frac{\partial F}{\partial \mu}$$

其中 $\mu$ 是对外部状态的变分估计，$\kappa_\mu$ 是感知更新率。

**行动方程（主动推断）**：

$$\dot{a} = -\kappa_a \frac{\partial F}{\partial a} = -\kappa_a \frac{\partial F}{\partial o} \frac{\partial o}{\partial a}$$

行动通过改变观测 $o$ 来减少自由能。$\frac{\partial o}{\partial a}$ 是感觉输入对行动的偏导数，描述了行动如何影响观测——即Agent的"运动-感觉"映射。

**学习方程（参数更新）**：

$$\dot{\theta} = -\kappa_\theta \frac{\partial F}{\partial \theta}$$

三个方程的耦合形成了完整的主动推断动力学：感知提供状态估计，行动改变环境，学习优化模型。

### 4.2 离散状态空间中的主动推断

在离散状态空间中，主动推断的形式化更加清晰。设：

- 隐藏状态空间：$\mathcal{S} = \{s_1, s_2, \ldots, s_N\}$
- 观测空间：$\mathcal{O} = \{o_1, o_2, \ldots, o_M\}$
- 行动空间：$\mathcal{A} = \{a_1, a_2, \ldots, a_K\}$
- 策略空间：$\Pi = \{\pi_1, \pi_2, \ldots, \pi_P\}$（每个策略是行动序列）

生成模型由以下概率矩阵参数化：

- **A矩阵**（似然映射）：$A_{ij} = P(o=i | s=j)$，从状态到观测的映射
- **B矩阵**（转移映射）：$B_{ijk} = P(s_t=i | s_{t-1}=j, a=k)$，行动条件下的状态转移
- **C向量**（偏好分布）：$C_i = \ln P(o=i)$，Agent对观测的先验偏好
- **D向量**（初始状态）：$D_i = P(s_0=i)$，初始状态的先验

### 4.3 策略选择：最小化预期自由能

对于策略 $\pi$，预期自由能 $G(\pi)$ 的完整计算如下：

$$G(\pi) = \underbrace{\sum_{\tau} \mathbb{E}_{q(o_\tau|\pi)}[\ln q(o_\tau|\pi) - \ln C]}_{\text{外禀价值：偏好满足}} - \underbrace{\sum_{\tau} \mathbb{E}_{q(o_\tau|\pi)}[D_{\text{KL}}[q(s_\tau|o_\tau,\pi) \| q(s_\tau|\pi)]]}_{\text{认知价值：信息增益}}$$

外禀价值项可以重写为：

$$\sum_\tau \mathbb{E}_{q(o_\tau|\pi)}[\ln q(o_\tau|\pi) - \ln C] = \sum_\tau D_{\text{KL}}[q(o_\tau|\pi) \| C] + \text{const}$$

这表明外禀价值是期望观测分布与偏好分布之间的KL散度——Agent倾向于选择能使其观测分布接近偏好分布的策略。

认知价值项是未来每个时间步的互信息之和——Agent倾向于选择能最大程度减少状态不确定性的策略。

策略后验通过softmax计算：

$$q(\pi) = \frac{\exp(-\gamma \cdot G(\pi) - \ln E(\pi))}{\sum_{\pi'} \exp(-\gamma \cdot G(\pi') - \ln E(\pi'))}$$

其中 $E(\pi)$ 是策略的先验（习惯），$\gamma$ 是精度参数。

### 4.4 从主动推断推导Agent的自驱行为生成

主动推断框架的一个革命性特征是：**行为不需要外部奖励信号驱动**。行为的驱动力来自自由能最小化——这是一个完全内生的动力。

具体来说，Agent的自驱行为来自两个来源：

**1. 偏好驱动（外禀动机）**：Agent有先验偏好 $C$（如"偏好吃饱"），通过最小化外禀价值来满足偏好。这类似于强化学习中的外在奖励，但关键区别是：偏好是生成模型的一部分，而非外部注入的信号。

**2. 不确定性驱动（认知动机）**：Agent有减少不确定性的内在倾向，通过最大化认知价值来探索环境。这类似于强化学习中的内在奖励（好奇心），但在主动推断中，它是自由能最小化的自然推论，无需额外设计。

自驱行为的生成过程：

$$\text{观测 } o_t \xrightarrow{\text{感知}} q(s_t) \xrightarrow{\text{规划}} q(\pi) \xrightarrow{\text{行动}} a_t \xrightarrow{\text{环境}} o_{t+1} \rightarrow \cdots$$

这个循环完全由自由能最小化驱动，无需任何外部奖励设计——这正是"自驱"的数学基础。

### 4.5 连续时间主动推断

在连续时间设定中，主动推断采用广义坐标描述。设外部状态 $\psi(t)$ 的广义坐标为 $\tilde{\psi} = (\psi, \psi', \psi'', \ldots)$，其中 $\psi^{(n)}$ 是 $\psi$ 的第 $n$ 阶时间导数。

广义自由能：

$$\tilde{F} = \frac{1}{2}(\tilde{o} - \tilde{g}(\tilde{\psi}))^T \tilde{\Sigma}_o^{-1} (\tilde{o} - \tilde{g}(\tilde{\psi})) + \frac{1}{2}(\tilde{\psi} - \tilde{f}(\tilde{\psi}))^T \tilde{\Sigma}_s^{-1} (\tilde{\psi} - \tilde{f}(\tilde{\psi}))$$

其中 $\tilde{g}$ 是观测函数的广义形式，$\tilde{f}$ 是状态演化函数的广义形式。

感知和行动的更新方程：

$$\dot{\tilde{\mu}} = D\tilde{\mu} - \kappa_\mu \partial_{\tilde{\mu}} \tilde{F}$$

$$\dot{a} = -\kappa_a \partial_a \tilde{F} = -\kappa_a (\partial_a \tilde{o})^T \tilde{\Sigma}_o^{-1} (\tilde{o} - \tilde{g}(\tilde{\mu}))$$

其中 $D$ 是移位算子（将广义坐标降一阶：$D(\psi, \psi', \psi'', \ldots) = (\psi', \psi'', \ldots)$）。

### 4.6 开源实现：pymdp库

pymdp是Friston实验室开发的Python主动推断库，支持离散状态空间的完整主动推断流程：

```python
from pymdp import Agent
from pymdp import utils

A = utils.create_A_matrix(num_obs, num_states, num_modalities, num_factors)
B = utils.create_B_matrix(num_states, num_actions, num_factors)
C = utils.create_C_vector(num_obs, num_modalities)
D = utils.create_D_vector(num_states, num_factors)

agent = Agent(A=A, B=B, C=C, D=D)

observation = env.step(initial_action)
qx = agent.infer_states(observation)
qpi = agent.infer_policies()
action = agent.sample_action()
```

### 4.7 实现路径

1. **离散主动推断**：基于pymdp实现完整的POMDP主动推断Agent
2. **预期自由能计算**：实现EFE的精确计算，包括外禀价值和认知价值两项
3. **策略选择**：实现基于softmax的策略后验计算，支持精度调节
4. **连续主动推断**：使用JAX实现广义坐标下的连续时间主动推断
5. **自驱验证**：在无外部奖励的环境中，验证Agent的自驱探索行为和偏好满足行为

---

## 阶段5：信息论基础——熵、互信息、KL散度

### 5.1 香农熵与不确定性量化

信息论由Claude Shannon于1948年创立，为"信息"这一直觉概念提供了严格的数学基础。信息论的核心量是香农熵：

$$H(X) = -\sum_{x} p(x) \ln p(x)$$

对于连续随机变量，微分熵定义为：

$$h(X) = -\int p(x) \ln p(x) dx$$

熵度量了随机变量的不确定性。几个关键性质：

1. **非负性**：$H(X) \geq 0$（离散情况）
2. **最大熵**：在有限支撑集上，均匀分布达到最大熵
3. **可加性**：对独立随机变量，$H(X, Y) = H(X) + H(Y)$
4. **条件熵**：$H(X|Y) = H(X, Y) - H(Y) \leq H(X)$（条件减少不确定性）

在Agent系统中的意义：**熵度量了Agent对环境状态的无知程度**。Agent的目标是减少对环境状态的不确定性——即最小化 $H(S|O)$（给定观测后状态的条件熵）。这与自由能最小化直接对应：

$$F \geq H(S|O) = -\mathbb{E}_{p(o,s)}[\ln p(s|o)]$$

变分自由能是条件熵的上界，最小化自由能自然地最小化了条件熵。

### 5.2 互信息与信息增益

互信息度量了两个随机变量之间的统计依赖程度：

$$I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = \sum_{x,y} p(x,y) \ln \frac{p(x,y)}{p(x)p(y)}$$

互信息的关键性质：

1. **对称性**：$I(X; Y) = I(Y; X)$
2. **非负性**：$I(X; Y) \geq 0$，等号当且仅当 $X$ 和 $Y$ 独立
3. **与KL散度的关系**：$I(X; Y) = D_{\text{KL}}[p(X,Y) \| p(X)p(Y)]$
4. **数据处理不等式**：若 $X \rightarrow Y \rightarrow Z$ 构成马尔可夫链，则 $I(X; Z) \leq I(X; Y)$

在Agent系统中的意义：**互信息度量了观测对状态的信息增益**。主动推断中的认知价值正是互信息：

$$\text{认知价值} = I(\Psi; O | \pi) = \mathbb{E}_{q(o|\pi)}[D_{\text{KL}}[q(\psi|o,\pi) \| q(\psi|\pi)]]$$

Agent选择最大化信息增益的策略——即选择能提供最多关于隐藏状态信息的行动。这就是"好奇心"的信息论定义。

### 5.3 KL散度与模型差异度量

KL散度（Kullback-Leibler散度）度量了两个概率分布之间的"距离"：

$$D_{\text{KL}}(p \| q) = \sum_{x} p(x) \ln \frac{p(x)}{q(x)} = \mathbb{E}_{p}\left[\ln \frac{p(x)}{q(x)}\right]$$

KL散度的关键性质：

1. **非负性**：$D_{\text{KL}}(p \| q) \geq 0$，等号当且仅当 $p = q$（Gibbs不等式）
2. **非对称性**：$D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$，因此不是真正的距离度量
3. **与交叉熵的关系**：$D_{\text{KL}}(p \| q) = H(p, q) - H(p)$，其中 $H(p, q) = -\sum p \ln q$ 是交叉熵
4. **与互信息的关系**：$I(X; Y) = D_{\text{KL}}[p(X,Y) \| p(X)p(Y)]$
5. **凸性**：$D_{\text{KL}}(p \| q)$ 关于 $(p, q)$ 联合凸

KL散度在Agent系统中的核心角色：

- **变分推断**：$D_{\text{KL}}[q(\psi) \| p(\psi|o)]$ 度量变分分布与真实后验的差距
- **自由能分解**：$F = D_{\text{KL}}[q(\psi) \| p(\psi|o)] - \ln p(o)$
- **模型学习**：$D_{\text{KL}}[q(\psi) \| p(\psi)]$ 度量后验偏离先验的程度（复杂性代价）
- **策略评估**：$D_{\text{KL}}[q(o|\pi) \| C]$ 度量期望观测偏离偏好的程度

### 5.4 信息论在Agent系统中的应用映射

将信息论的核心量映射到Agent系统的功能模块：

| 信息论量 | Agent系统映射 | 功能 |
|---------|-------------|------|
| $H(S)$ | 环境先验不确定性 | Agent对环境的初始无知 |
| $H(S\|O)$ | 后验不确定性 | 感知后的剩余无知 |
| $I(S; O)$ | 感知信息增益 | 观测对状态的信息量 |
| $D_{\text{KL}}[q\|p]$ | 模型偏差 | 内部模型与真实模型的差距 |
| $I(S; O\|\pi)$ | 行动信息增益 | 特定行动的信息获取量 |

信息论视角下的Agent目标函数：

$$\text{Agent目标} = \min_\theta D_{\text{KL}}[q_\theta(\psi|o) \| p(\psi|o)] + \lambda \cdot I(\Psi; O | \pi)$$

第一项是感知精度（最小化变分近似误差），第二项是认知价值（最大化信息获取）。

### 5.5 信息论与自由能的统一

自由能可以完全用信息论量表达：

$$F = \underbrace{H(q)}_{\text{变分熵}} - \underbrace{\mathbb{E}_q[\ln p(o, \psi)]}_{\text{期望对数联合}}$$

或者等价地：

$$F = \underbrace{-\ln p(o)}_{\text{惊讶度}} + \underbrace{D_{\text{KL}}[q(\psi) \| p(\psi|o)]}_{\text{推断误差}}$$

这表明自由能最小化等价于：
1. 最小化对观测的惊讶度（使模型更好地解释数据）
2. 最小化变分推断的近似误差（使内部表征更接近真实后验）

### 5.6 信息瓶颈理论

信息瓶颈（Information Bottleneck）理论为Agent的表征学习提供了信息论框架。给定输入 $X$ 和相关目标 $Y$，寻找压缩表征 $T$ 使得：

$$\min_{p(t|x)} I(X; T) - \beta \cdot I(T; Y)$$

第一项鼓励压缩（遗忘无关信息），第二项鼓励保留与目标相关的信息。参数 $\beta$ 控制压缩-保留的权衡。

在Agent系统中，$X$ 是原始观测，$Y$ 是隐藏状态，$T$ 是内部表征。信息瓶颈原则说明：**最优内部表征应在保留预测相关信息的同时最大化压缩**——这解释了为什么大脑的表征是稀疏和高效的。

### 5.7 开源技术选型

| 技术需求 | 开源方案 | 说明 |
|---------|---------|------|
| 信息论计算 | dit (Python) | 离散分布的信息论量计算库 |
| 互信息估计 | NPEET / scikit-learn | 连续变量的互信息非参数估计 |
| KL散度计算 | scipy.stats.entropy | 离散KL散度；PyTorch分布模块 |
| 信息瓶颈 | IB_neural (PyTorch) | 神经网络信息瓶颈实现 |

### 5.8 实现路径

1. **信息论工具箱**：封装熵、互信息、KL散度的计算函数，支持离散和连续分布
2. **信息增益估计**：实现基于kNN的互信息估计器，用于连续状态空间
3. **信息瓶颈模块**：实现信息瓶颈正则化，作为Agent表征学习的约束
4. **信息论诊断**：实现Agent内部信息流的监控工具，量化感知、行动各环节的信息传递
5. **验证**：在简单环境中验证信息增益驱动的探索行为与随机探索的效率差异

---

## 阶段6：动力系统理论——吸引子、分岔、混沌

### 6.1 相空间与吸引子概念

动力系统理论为理解Agent认知状态的演化提供了几何语言。一个连续时间动力系统由微分方程描述：

$$\dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u})$$

其中 $\mathbf{x} \in \mathbb{R}^n$ 是状态向量，$\mathbf{u}$ 是控制参数（或外部输入），$f$ 是向量场。系统的相空间是所有可能状态的集合 $\mathbb{R}^n$，系统在相空间中的轨迹称为轨道。

**吸引子**是相空间中的特殊结构，它"吸引"附近的轨道。形式化地，集合 $\mathcal{A}$ 是吸引子，如果存在邻域 $\mathcal{U} \supset \mathcal{A}$ 使得：

$$\lim_{t \to \infty} \text{dist}(\phi_t(\mathbf{x}_0), \mathcal{A}) = 0, \quad \forall \mathbf{x}_0 \in \mathcal{U}$$

其中 $\phi_t$ 是流映射，$\text{dist}$ 是点到集合的距离。

吸引子的类型：

1. **不动点（Fixed Point）**：$f(\mathbf{x}^*) = 0$，系统收敛到单一状态。在Agent系统中，不动点对应稳态认知——对环境的稳定表征。
2. **极限环（Limit Cycle）**：周期性轨道，系统在相空间中沿闭合曲线运动。对应Agent的节律性行为（如昼夜节律、呼吸）。
3. **环面（Torus）**：准周期运动，多个不可约频率的叠加。对应Agent的多时间尺度行为。
4. **奇怪吸引子（Strange Attractor）**：分形结构的混沌吸引子。对应Agent的创造性、不可预测行为。

### 6.2 分岔理论与系统相变

分岔是动力系统的定性行为随参数变化而突然改变的现象。考虑参数化系统：

$$\dot{\mathbf{x}} = f(\mathbf{x}, \mu)$$

当参数 $\mu$ 穿过临界值 $\mu_c$ 时，系统的吸引子结构发生质变——这就是分岔。

**鞍结分岔（Saddle-Node Bifurcation）**：

$$\dot{x} = \mu - x^2$$

- 当 $\mu > 0$：存在两个不动点 $x^* = \pm\sqrt{\mu}$（稳定和不稳定各一个）
- 当 $\mu = 0$：两个不动点合并消失
- 当 $\mu < 0$：无不动点

**跨临界分岔（Transcritical Bifurcation）**：

$$\dot{x} = \mu x - x^2$$

稳定不动点与不稳定不动点在 $\mu = 0$ 处交换稳定性。

**叉式分岔（Pitchfork Bifurcation）**：

$$\dot{x} = \mu x - x^3$$

- 当 $\mu < 0$：唯一稳定不动点 $x^* = 0$
- 当 $\mu > 0$：$x^* = 0$ 变为不稳定，出现两个新稳定不动点 $x^* = \pm\sqrt{\mu}$

在Agent系统中的意义：**分岔对应认知相变**。当环境参数（如不确定性水平）超过临界值时，Agent的认知状态可能从单一稳态跳变为多稳态——这解释了"顿悟"、"范式转换"等认知现象。Agent的进化过程本质上是在参数空间中寻找远离分岔边界的稳健工作点。

### 6.3 Lyapunov稳定性分析

Lyapunov稳定性理论为判断系统行为提供了强有力的工具，无需求解微分方程。

**Lyapunov函数**：若存在正定函数 $V(\mathbf{x})$ 使得 $\dot{V}(\mathbf{x}) \leq 0$，则不动点 $\mathbf{x}^* = 0$ 是稳定的。若 $\dot{V}(\mathbf{x}) < 0$（严格负定），则渐近稳定。

**Lyapunov指数**：度量相邻轨道的指数分离率：

$$\lambda_i = \lim_{t \to \infty} \frac{1}{t} \ln \frac{\|\delta \mathbf{x}_i(t)\|}{\|\delta \mathbf{x}_i(0)\|}$$

其中 $\delta \mathbf{x}_i$ 是沿第 $i$ 个方向的扰动。Lyapunov指数谱 $\{\lambda_1, \lambda_2, \ldots, \lambda_n\}$ 完全刻画了系统的动力学性质：

- 所有 $\lambda_i < 0$：系统收敛到不动点
- 存在 $\lambda_i = 0$ 且其余 $< 0$：极限环
- 最大 $\lambda_1 > 0$：混沌

在Agent系统中，**自由能本身就是一个Lyapunov函数**：

$$\dot{F} = \nabla F \cdot \dot{\mathbf{x}} = -\kappa \|\nabla F\|^2 \leq 0$$

这保证了自由能最小化过程的稳定性——Agent的认知状态必然收敛到自由能的局部极小值（吸引子）。

### 6.4 Agent认知状态的动力系统建模

将Agent的认知动力学建模为动力系统：

$$\dot{\mathbf{s}} = -\nabla_{\mathbf{s}} F(\mathbf{s}, \theta) + \xi(t)$$

其中 $\mathbf{s}$ 是认知状态（内部表征），$F$ 是自由能函数，$\xi(t)$ 是噪声项。

这个模型的关键性质：

1. **多稳态**：自由能景观可能有多个局部极小值，对应不同的认知模式
2. **噪声诱导转换**：噪声可以使系统从一个吸引子跳到另一个吸引子，对应认知状态切换
3. **分岔控制**：模型参数 $\theta$ 的变化可以触发分岔，对应认知相变
4. **混沌边缘**：在有序与混沌之间的临界状态，系统具有最大的计算能力和适应性

**临界性假说**：生物认知系统倾向于运行在临界状态附近——既不完全有序（缺乏灵活性），也不完全混沌（缺乏稳定性）。这种"混沌边缘"状态最大化了信息处理能力：

$$\text{适应性} \propto \text{灵活性} \times \text{稳定性}$$

在临界点附近，系统同时具有高灵活性（可以快速切换状态）和高稳定性（可以维持当前状态）。

### 6.5 开源技术选型

| 技术需求 | 开源方案 | 说明 |
|---------|---------|------|
| 动力系统仿真 | SciPy (odeint/solve_ivp) | 常微分方程数值求解 |
| 分岔分析 | PyDSTool / AUTO-07p | 分岔分析与延续方法 |
| Lyapunov指数 | nolds (Python) | 非线性动力系统特征量计算 |
| 相空间重构 | TISEAN / pyTisean | 时间序列的非线性分析 |
| 吸引子可视化 | Matplotlib / Plotly | 相空间轨迹与吸引子可视化 |

### 6.6 实现路径

1. **自由能景观建模**：使用JAX构建自由能函数的参数化模型，支持梯度计算和Hessian分析
2. **吸引子识别**：实现不动点搜索算法（牛顿法+延拓法），识别自由能景观的吸引子结构
3. **分岔分析**：实现参数延续算法，追踪吸引子随模型参数变化的分岔曲线
4. **Lyapunov分析**：实现Lyapunov指数谱的数值计算，评估认知动力学的稳定性
5. **临界性检测**：实现基于关联长度和雪崩分布的临界性检测算法
6. **验证**：在简单认知任务中验证Agent认知状态的多稳态和分岔行为

---

## 阶段7：控制论基础——反馈、前馈、自适应控制

### 7.1 负反馈与系统稳定性

控制论（Cybernetics）由Norbert Wiener创立，研究系统如何通过反馈来调节自身行为。其核心概念是负反馈：系统输出与期望值之间的偏差被用来调整输入，从而减小偏差。

经典负反馈控制系统：

$$u(t) = K \cdot e(t) = K \cdot (r(t) - y(t))$$

其中 $r(t)$ 是参考信号（期望输出），$y(t)$ 是实际输出，$e(t) = r(t) - y(t)$ 是误差信号，$K$ 是增益，$u(t)$ 是控制输入。

闭环传递函数：

$$G_{cl}(s) = \frac{KG(s)}{1 + KG(s)}$$

稳定性条件：闭环特征方程 $1 + KG(s) = 0$ 的所有根具有负实部。

在Agent系统中，**自由能本身就是误差信号**，自由能最小化就是负反馈控制：

$$\dot{\theta} = -K \cdot \nabla_\theta F(\theta, o)$$

其中 $F$ 是自由能（误差），$\theta$ 是模型参数（控制变量），$K$ 是学习率（增益）。

### 7.2 PID控制与Agent误差修正

PID（比例-积分-微分）控制是最广泛使用的控制策略：

$$u(t) = K_p e(t) + K_i \int_0^t e(\tau) d\tau + K_d \frac{de(t)}{dt}$$

三项的物理意义：

- **比例项（P）**：$K_p e(t)$——当前误差的直接响应，产生即时修正力
- **积分项（I）**：$K_i \int e(\tau) d\tau$——累积误差的响应，消除稳态偏差
- **微分项（D）**：$K_d \dot{e}(t)$——误差变化率的响应，提供阻尼和预判

PID控制与Agent预测编码的对应关系：

| PID项 | 预测编码对应 | Agent功能 |
|-------|------------|----------|
| P项 | 当前预测误差 $\mathbf{e}^0$ | 即时感知修正 |
| I项 | 累积预测误差 $\int \mathbf{e}^0 dt$ | 长期模型更新（学习） |
| D项 | 误差变化率 $\dot{\mathbf{e}}^0$ | 趋势预判（预测） |

Agent的PID式预测修正规则：

$$\Delta \theta = -\alpha_p \nabla_\theta F_t - \alpha_i \sum_{\tau=0}^{t} \nabla_\theta F_\tau - \alpha_d (\nabla_\theta F_t - \nabla_\theta F_{t-1})$$

其中 $\alpha_p, \alpha_i, \alpha_d$ 分别是比例、积分、微分学习率。

### 7.3 模型预测控制（MPC）

模型预测控制（Model Predictive Control）是现代控制理论的核心方法，其思想是：**利用内部模型预测未来，选择使预测输出最优的控制输入**。

MPC的数学形式化：

$$\min_{\mathbf{u}_{t:t+H}} \sum_{k=0}^{H} \left[ \|y_{t+k} - r_{t+k}\|_Q^2 + \|u_{t+k}\|_R^2 \right]$$

约束条件：

$$x_{t+k+1} = f(x_{t+k}, u_{t+k}), \quad y_{t+k} = h(x_{t+k})$$

$$u_{\min} \leq u_{t+k} \leq u_{\max}, \quad x_{\min} \leq x_{t+k} \leq x_{\max}$$

其中 $H$ 是预测时域，$Q$ 是输出误差权重，$R$ 是控制代价权重。

MPC与主动推断的对应关系：

| MPC概念 | 主动推断对应 |
|---------|------------|
| 内部模型 $f, h$ | 生成模型 $p(o|\psi)$ |
| 参考轨迹 $r$ | 偏好分布 $C$ |
| 预测时域 $H$ | 策略规划时域 |
| 代价函数 | 预期自由能 $G(\pi)$ |
| 约束 | 状态-行动的物理限制 |

关键区别：MPC使用确定性优化，主动推断使用概率推断。主动推断可以看作**概率化的MPC**——它不仅考虑期望代价，还考虑不确定性（认知价值）。

### 7.4 自适应控制与在线参数调整

自适应控制处理系统参数未知或时变的情况。其核心思想是：**同时估计系统参数和控制状态**——这与主动推断的"同时感知和学习"完全一致。

**模型参考自适应控制（MRAC）**：

设被控系统为 $\dot{y} = a \cdot y + b \cdot u$（参数 $a, b$ 未知），参考模型为 $\dot{y}_m = a_m \cdot y_m + b_m \cdot r$。

控制律：$u = \hat{\theta}^T \omega$，其中 $\hat{\theta}$ 是可调参数，$\omega$ 是回归向量。

参数更新律（基于Lyapunov稳定性）：

$$\dot{\hat{\theta}} = -\Gamma \omega e$$

其中 $e = y - y_m$ 是跟踪误差，$\Gamma$ 是正定自适应增益矩阵。

**自适应控制与Agent在线学习的统一**：

Agent的在线学习本质上是自适应控制——模型参数 $\theta$ 对应控制器参数，预测误差对应跟踪误差，梯度下降对应自适应律。Lyapunov稳定性理论为Agent的在线学习提供了稳定性保证：

$$V = F(\theta) \geq 0, \quad \dot{V} = \nabla F \cdot \dot{\theta} = -\kappa \|\nabla F\|^2 \leq 0$$

自由能作为Lyapunov函数，保证了Agent在线学习的稳定收敛。

### 7.5 鲁棒控制与Agent安全

鲁棒控制处理模型不确定性——当内部模型不完全准确时，如何保证系统仍然稳定。

$H_\infty$ 控制的核心思想：**最坏情况下的最优控制**。

$$\min_{K} \max_{\Delta} \|T_{zw}(K, \Delta)\|_\infty$$

其中 $K$ 是控制器，$\Delta$ 是模型不确定性，$T_{zw}$ 是从扰动 $w$ 到输出 $z$ 的传递函数。

在Agent系统中，鲁棒性对应**安全性**：即使内部模型有误差，Agent也不应做出危险行为。实现方式：

1. **保守策略**：在不确定时选择更安全的行动
2. **约束优化**：在自由能最小化中加入安全约束
3. **多重模型**：维护多个内部模型，考虑最坏情况

### 7.6 开源技术选型

| 技术需求 | 开源方案 | 说明 |
|---------|---------|------|
| 控制系统仿真 | python-control | 经典与现代控制理论工具箱 |
| MPC实现 | CasADi / do-mpc | 非线性MPC求解器 |
| 自适应控制 | 自定义 (JAX) | 基于JAX的自适应律实现 |
| 鲁棒控制 | python-control | 支持$H_\infty$综合 |
| 系统辨识 | SIPPY / sysid | 系统参数在线辨识 |

### 7.7 实现路径

1. **PID修正模块**：实现预测误差的PID式修正，支持比例/积分/微分三项独立调节
2. **MPC规划模块**：基于CasADi实现非线性MPC，作为Agent的规划核心
3. **自适应学习模块**：实现基于Lyapunov的参数自适应律，保证在线学习稳定性
4. **鲁棒安全模块**：实现约束MPC，在自由能最小化中加入安全约束
5. **验证**：在控制基准任务（倒立摆、轨迹跟踪）中验证Agent的控制性能和鲁棒性

---

## 阶段8：进化计算理论——选择、变异、遗传

### 8.1 自然选择算法的数学描述

达尔文自然选择的三要素——变异、遗传、选择——可以形式化为优化算法。设种群为 $\mathcal{P} = \{\theta_1, \theta_2, \ldots, \theta_N\}$，适应度函数为 $\Phi: \Theta \to \mathbb{R}$。

自然选择的数学描述：

$$\mathcal{P}_{t+1} = \text{Select}(\text{Mutate}(\text{Crossover}(\mathcal{P}_t)))$$

选择算子的数学描述（轮盘赌选择）：

$$P(\theta_i \text{ 被选中}) = \frac{\Phi(\theta_i)}{\sum_{j=1}^{N} \Phi(\theta_j)}$$

更一般地，使用Boltzmann选择：

$$P(\theta_i) = \frac{\exp(\beta \cdot \Phi(\theta_i))}{\sum_j \exp(\beta \cdot \Phi(\theta_j))}$$

其中 $\beta$ 是逆温度参数，$\beta \to \infty$ 时退化为贪婪选择，$\beta \to 0$ 时退化为均匀选择。

### 8.2 变异算子与探索空间

变异是进化的驱动力——它产生新的候选解。变异算子的设计决定了搜索的性质：

**高斯变异**：

$$\theta' = \theta + \sigma \cdot \mathcal{N}(0, I)$$

其中 $\sigma$ 是变异步长。$\sigma$ 太大导致随机搜索，太小导致早熟收敛。

**自适应变异步长**（1/5成功法则）：

$$\sigma_{t+1} = \begin{cases} \sigma_t \cdot c^{1/d} & \text{if } p_{\text{success}} > 1/5 \\ \sigma_t / c^{1/d} & \text{if } p_{\text{success}} < 1/5 \\ \sigma_t & \text{if } p_{\text{success}} = 1/5 \end{cases}$$

其中 $p_{\text{success}}$ 是变异成功率，$c \approx 0.817$，$d$ 是维度。这一法则的理论基础是：最优变异步长应使约1/5的变异产生更优后代。

**协方差自适应变异**（CMA-ES的核心）：

$$\theta' \sim \mathcal{N}(\mathbf{m}, \sigma^2 \mathbf{C})$$

其中 $\mathbf{m}$ 是分布均值，$\mathbf{C}$ 是协方差矩阵。CMA-ES通过利用成功变异的方向信息来自适应调整协方差矩阵，实现高效的定向搜索。

### 8.3 遗传算法在模型结构搜索中的应用

在Agent系统中，进化计算不仅用于参数优化，更关键的是用于模型结构搜索（Architecture Search）。

设模型结构为 $\alpha \in \mathcal{A}$（如网络层数、每层宽度、连接拓扑），参数为 $\theta \in \Theta_\alpha$。结构搜索的目标是：

$$\alpha^* = \arg\max_{\alpha \in \mathcal{A}} \Phi(\alpha) = \arg\max_{\alpha} \mathbb{E}_{\theta \sim p(\theta|\alpha)}[-F(\theta, \alpha)]$$

遗传算法的结构搜索流程：

1. **编码**：将模型结构 $\alpha$ 编码为"染色体"（如邻接矩阵、变长序列）
2. **交叉**：两个父代结构的重组（如均匀交叉、单点交叉）
3. **变异**：结构的随机修改（如增删层、改变连接）
4. **选择**：基于适应度（负自由能）选择优秀结构
5. **评估**：训练每个候选结构并评估其自由能

### 8.4 (1+1)-ES与Agent模型进化

(1+1)-ES（Evolution Strategy）是最简单的进化策略，但理论分析最为清晰，非常适合Agent的在线进化：

算法：

```
初始化: θ_0, σ_0
For t = 0, 1, 2, ...:
    θ' = θ_t + σ_t · N(0, I)          // 变异
    If Φ(θ') > Φ(θ_t):                 // 选择
        θ_{t+1} = θ'                    // 接受
    Else:
        θ_{t+1} = θ_t                   // 拒绝
    更新 σ_t → σ_{t+1}                  // 步长自适应
```

(1+1)-ES的收敛性：在球面函数 $f(\theta) = -\|\theta\|^2$ 上，(1+1)-ES的期望收敛速率为：

$$\mathbb{E}[\ln f(\theta_t)] = O(t)$$

即对数线性收敛。这一收敛速度虽然慢于梯度方法，但(1+1)-ES不需要梯度信息——这在梯度难以计算或不可靠时（如非凸景观、噪声适应度）具有显著优势。

**Agent在线进化的实现**：

在Agent系统中，(1+1)-ES用于模型参数的在线进化：

$$\theta_{t+1} = \begin{cases} \theta_t + \sigma_t \xi_t & \text{if } F(\theta_t + \sigma_t \xi_t) < F(\theta_t) \\ \theta_t & \text{otherwise} \end{cases}$$

其中适应度是负自由能 $-F$，$\xi_t \sim \mathcal{N}(0, I)$ 是随机扰动。

这种进化策略与梯度下降的关键区别：
- 梯度下降：沿着最速下降方向，快速但可能陷入局部极小
- (1+1)-ES：随机探索方向，慢但能跳出局部极小

两者可以互补：**梯度下降负责快速适应（感知推断），进化策略负责长期优化（结构进化）**。

### 8.5 进化与学习的协同

Baldwin效应描述了学习与进化的协同：个体学习可以引导进化方向——即使习得特征不遗传，学习使个体在更广的适应度景观中"存活"，从而为进化提供更多时间来发现基因编码的等效解。

在Agent系统中，Baldwin效应的实现：

1. **学习阶段**：Agent通过梯度下降（感知推断）快速适应当前环境
2. **评估阶段**：评估学习后的自由能作为适应度
3. **进化阶段**：基于适应度进行参数变异和选择
4. **遗传阶段**：将进化后的参数作为下一代的初始值

这种"学习引导进化、进化加速学习"的协同机制，使Agent能够同时实现快速适应和长期优化。

### 8.6 开源技术选型

| 技术需求 | 开源方案 | 说明 |
|---------|---------|------|
| 进化策略 | evosax (JAX) | JAX原生进化策略库，支持(1+1)-ES, CMA-ES等 |
| 遗传算法 | DEAP | 通用进化计算框架 |
| CMA-ES | pycma | CMA-ES官方实现 |
| 神经架构搜索 | NAS-Bench / DARTS | 可微分架构搜索 |
| 进化强化学习 | OpenAI ES / PPO-ES | 进化策略与强化学习结合 |

### 8.7 实现路径

1. **进化策略核心**：基于evosax实现(1+1)-ES和CMA-ES，以负自由能为适应度
2. **在线进化循环**：将进化策略嵌入Agent的感知-行动循环，实现边运行边进化
3. **结构搜索**：实现基于遗传算法的模型结构搜索，支持增删层和修改连接
4. **学习-进化协同**：实现Baldwin效应的在线版本，学习引导进化方向
5. **验证**：在非凸优化基准和动态环境中验证进化策略的跳出局部极小能力

---

## 阶段9：强化学习理论——MDP、策略梯度、值函数

### 9.1 马尔可夫决策过程形式化

马尔可夫决策过程（MDP）是强化学习的数学基础，定义为元组 $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$：

- 状态空间：$\mathcal{S}$
- 行动空间：$\mathcal{A}$
- 转移概率：$P(s'|s,a) = \Pr(s_{t+1}=s'|s_t=s, a_t=a)$
- 奖励函数：$R(s,a) = \mathbb{E}[r_t|s_t=s, a_t=a]$
- 折扣因子：$\gamma \in [0, 1)$

策略 $\pi(a|s)$ 定义了在每个状态下选择行动的概率分布。MDP的目标是找到最优策略 $\pi^*$ 最大化期望累积奖励：

$$\pi^* = \arg\max_\pi \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]$$

**POMDP扩展**：在Agent系统中，状态不是完全可观测的。部分可观测MDP（POMDP）引入观测空间 $\mathcal{O}$ 和观测函数 $O(o|s,a)$：

$$\text{POMDP} = (\mathcal{S}, \mathcal{A}, \mathcal{O}, P, R, O, \gamma)$$

Agent需要维护信念状态 $b(s) = p(s_t=s|o_{0:t}, a_{0:t-1})$，信念更新通过贝叶斯规则：

$$b'(s') = \eta \cdot O(o|s',a) \sum_s P(s'|s,a) b(s)$$

其中 $\eta$ 是归一化常数。POMDP与主动推断的生成模型完全对应。

### 9.2 Bellman方程与值迭代

**状态值函数**：

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t \bigg| s_0 = s\right]$$

**行动值函数（Q函数）**：

$$Q^\pi(s,a) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t \bigg| s_0 = s, a_0 = a\right]$$

**Bellman期望方程**：

$$V^\pi(s) = \sum_a \pi(a|s) \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')\right]$$

$$Q^\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \sum_{a'} \pi(a'|s') Q^\pi(s',a')$$

**Bellman最优方程**：

$$V^*(s) = \max_a \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s')\right]$$

$$Q^*(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \max_{a'} Q^*(s',a')$$

**值迭代算法**：

$$V_{k+1}(s) = \max_a \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s')\right]$$

值迭代是压缩映射，由Banach不动点定理保证收敛：

$$\|V_{k+1} - V^*\| \leq \gamma \|V_k - V^*\|$$

### 9.3 策略梯度定理

策略梯度方法直接优化策略参数 $\theta$，无需显式计算值函数。

**策略梯度定理**：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \ln \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s,a)\right]$$

其中 $J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_t \gamma^t r_t\right]$ 是目标函数。

**REINFORCE算法**（Monte Carlo策略梯度）：

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T} \nabla_\theta \ln \pi_\theta(a_t^i|s_t^i) \cdot G_t^i$$

其中 $G_t^i = \sum_{k=t}^{T} \gamma^{k-t} r_k^i$ 是从时刻 $t$ 开始的累积奖励。

**优势函数与方差缩减**：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \ln \pi_\theta(a|s) \cdot A^{\pi_\theta}(s,a)\right]$$

其中优势函数 $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ 度量行动 $a$ 相对于平均水平的优势。使用优势函数替代Q值可以显著减小梯度方差。

**PPO（Proximal Policy Optimization）**：

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是策略比率。PPO通过裁剪策略更新幅度来保证训练稳定性。

### 9.4 内在奖励与自驱Agent的关联

传统强化学习依赖外部奖励信号，但自驱Agent需要内在动机。内在奖励的几种形式：

**1. 好奇心驱动（预测误差）**：

$$r_t^{\text{int}} = \|o_{t+1} - \hat{o}_{t+1}\|^2$$

Agent对预测误差大的状态/行动感兴趣——这与预测编码中的预测误差信号直接对应。

**2. 信息增益驱动（贝叶斯好奇心）**：

$$r_t^{\text{int}} = D_{\text{KL}}[p(s|o_{t+1}) \| p(s|o_t)]$$

Agent对能最大程度减少不确定性的行动感兴趣——这与主动推断的认知价值对应。

**3. 赋能驱动（状态访问熵）**：

$$r_t^{\text{int}} = H\left(\frac{1}{T}\sum_t p(s_t|a_{0:t})\right)$$

Agent倾向于到达能到达最多不同状态的位置——最大化对状态空间的控制力。

**统一视角**：所有内在奖励都可以从自由能原理推导出来——它们是自由能最小化的不同方面。好奇心驱动对应准确性项，信息增益对应复杂性项，赋能对应模型证据的边际化。

### 9.5 强化学习与主动推断的统一

主动推断可以看作强化学习的贝叶斯推广：

| 强化学习概念 | 主动推断对应 |
|------------|------------|
| 奖励 $r$ | 偏好 $C = \ln p(o)$ |
| 值函数 $V(s)$ | 负预期自由能 $-G(\pi)$ |
| 策略 $\pi(a|s)$ | 策略后验 $q(\pi)$ |
| 探索（$\epsilon$-greedy） | 认知价值（信息增益） |
| 利用（贪婪策略） | 外禀价值（偏好满足） |
| 模型无关 | 模型无关主动推断 |
| 模型基础 | 生成模型 $p(o,\psi)$ |

关键区别：主动推断中，探索不是需要手动平衡的附加项，而是自由能最小化的自然推论。这解决了强化学习中探索-利用权衡需要人工调参的问题。

### 9.6 开源技术选型

| 技术需求 | 开源方案 | 说明 |
|---------|---------|------|
| 强化学习框架 | Stable-Baselines3 / CleanRL | 标准RL算法实现 |
| PPO实现 | CleanRL | 单文件PPO实现，清晰易读 |
| 内在奖励 | intrinsic-reward (GitHub) | 好奇心、RND等内在奖励实现 |
| POMDP求解 | pomdp-py | POMDP求解框架 |
| 分布式RL | Ray RLlib | 大规模分布式RL训练 |

### 9.7 实现路径

1. **MDP核心**：使用Gymnasium构建Agent-环境交互接口，支持POMDP扩展
2. **值函数学习**：实现DQN及其变体（Double DQN, Dueling DQN）
3. **策略梯度**：实现PPO算法，作为Agent策略优化的基线
4. **内在奖励模块**：实现好奇心驱动和信息增益驱动的内在奖励
5. **主动推断-RL统一**：实现基于自由能的统一目标函数，同时包含外禀和认知价值
6. **验证**：在Atari和MuJoCo环境中对比外在奖励、内在奖励和主动推断的性能

---

## 阶段10：贝叶斯推断与在线学习

### 10.1 贝叶斯定理与后验更新

贝叶斯定理是概率论的基本定理，也是Agent感知过程的数学基础：

$$p(\theta | \mathcal{D}) = \frac{p(\mathcal{D} | \theta) p(\theta)}{p(\mathcal{D})}$$

其中：
- $p(\theta)$ 是先验分布——观测数据前的信念
- $p(\mathcal{D} | \theta)$ 是似然函数——给定参数时数据的概率
- $p(\theta | \mathcal{D})$ 是后验分布——观测数据后的更新信念
- $p(\mathcal{D}) = \int p(\mathcal{D} | \theta) p(\theta) d\theta$ 是边际似然（模型证据）

贝叶斯更新的核心思想：**后验正比于似然乘以先验**。每获得新数据，先验就被更新为后验，后验又成为下一次更新的先验——这就是在线学习的贝叶斯框架。

**序贯贝叶斯更新**：

$$p(\theta | \mathcal{D}_{1:t}) \propto p(d_t | \theta) \cdot p(\theta | \mathcal{D}_{1:t-1})$$

每一步只需用新数据 $d_t$ 更新前一步的后验，无需重新处理所有历史数据——这对Agent的在线学习至关重要。

### 10.2 共轭先验与高效计算

共轭先验使得后验分布与先验分布属于同一分布族，从而避免了积分计算：

**Beta-Bernoulli共轭**：

先验：$\theta \sim \text{Beta}(\alpha, \beta)$

似然：$d_i \sim \text{Bernoulli}(\theta)$

后验：$\theta | \mathcal{D} \sim \text{Beta}(\alpha + n_1, \beta + n_0)$

其中 $n_1$ 是成功次数，$n_0$ 是失败次数。

**Gaussian-Gaussian共轭**：

先验：$\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$

似然：$d_i \sim \mathcal{N}(\mu, \sigma^2)$（$\sigma^2$ 已知）

后验：$\mu | \mathcal{D} \sim \mathcal{N}(\mu_n, \sigma_n^2)$

其中：

$$\sigma_n^2 = \left(\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}\right)^{-1}, \quad \mu_n = \sigma_n^2 \left(\frac{\mu_0}{\sigma_0^2} + \frac{n\bar{d}}{\sigma^2}\right)$$

后验精度（方差倒数）等于先验精度与似然精度之和——**信息以精度加权的方式累积**。

**Dirichlet-Multinomial共轭**：

先验：$\boldsymbol{\theta} \sim \text{Dir}(\boldsymbol{\alpha})$

似然：$\mathbf{d} \sim \text{Mult}(n, \boldsymbol{\theta})$

后验：$\boldsymbol{\theta} | \mathcal{D} \sim \text{Dir}(\boldsymbol{\alpha} + \mathbf{n})$

在Agent系统中，Dirichlet-Multinomial共轭直接用于pymdp中的A矩阵和B矩阵的在线学习。

### 10.3 在线贝叶斯更新算法

在线贝叶斯更新是Agent实时学习的核心。算法框架：

```
初始化: 先验 p(θ)
For t = 1, 2, ...:
    1. 接收新观测 d_t
    2. 计算似然 p(d_t | θ)
    3. 更新后验: p(θ | D_{1:t}) ∝ p(d_t | θ) · p(θ | D_{1:t-1})
    4. 提取点估计: θ̂_t = E[θ | D_{1:t}]
    5. 使用θ̂_t进行决策
```

**遗忘因子**：在非平稳环境中，旧数据可能不再相关。引入遗忘因子 $\lambda \in (0, 1]$：

$$p(\theta | \mathcal{D}_{1:t}) \propto p(d_t | \theta) \cdot p(\theta | \mathcal{D}_{1:t-1})^\lambda$$

等价于对旧后验"升温"（增大方差），使Agent逐渐遗忘旧信息。$\lambda = 1$ 是标准贝叶斯更新（无遗忘），$\lambda < 1$ 引入遗忘。

**滑动窗口**：另一种处理非平稳性的方法是只使用最近 $W$ 个数据点：

$$p(\theta | \mathcal{D}_{t-W+1:t}) \propto \prod_{i=t-W+1}^{t} p(d_i | \theta) \cdot p(\theta)$$

### 10.4 粒子滤波与序贯蒙特卡洛

当共轭先验不可用或后验分布非高斯时，需要使用近似推断方法。粒子滤波（Particle Filtering）是序贯蒙特卡洛（Sequential Monte Carlo, SMC）方法的核心。

**基本思想**：用一组加权粒子 $\{(\theta^{(i)}, w^{(i)})\}_{i=1}^{N}$ 近似后验分布：

$$p(\theta | \mathcal{D}_{1:t}) \approx \sum_{i=1}^{N} w_t^{(i)} \delta(\theta - \theta_t^{(i)})$$

**序贯重要性采样（SIS）**：

1. **预测**：从提议分布采样 $\theta_t^{(i)} \sim q(\theta_t | \theta_{t-1}^{(i)}, d_t)$
2. **更新权重**：$w_t^{(i)} \propto w_{t-1}^{(i)} \cdot \frac{p(d_t | \theta_t^{(i)}) p(\theta_t^{(i)} | \theta_{t-1}^{(i)})}{q(\theta_t | \theta_{t-1}^{(i)}, d_t)}$
3. **归一化**：$w_t^{(i)} \leftarrow w_t^{(i)} / \sum_j w_t^{(j)}$

**重采样**：当有效粒子数 $N_{\text{eff}} = 1 / \sum (w^{(i)})^2$ 低于阈值时，进行重采样以避免权重退化：

- 系统重采样：按权重比例复制粒子
- 残差重采样：确定性+随机性组合

**Rao-Blackwellized粒子滤波**：对部分变量使用解析推断，其余变量使用粒子滤波——这可以显著减少所需粒子数。

### 10.5 变分在线学习

变分推断提供了另一种在线贝叶斯更新的途径。在变分框架下，后验近似为参数化分布 $q_\phi(\theta)$，通过最小化变分自由能来优化参数 $\phi$：

$$\phi_{t+1} = \phi_t - \eta \nabla_\phi \left[-\mathbb{E}_{q_\phi}[\ln p(d_t | \theta)] + D_{\text{KL}}[q_\phi(\theta) \| q_{\phi_t}(\theta)]\right]$$

注意第二项的KL散度是相对于上一步的后验（而非先验）——这实现了"自然"的在线更新，新后验不会偏离旧后验太远。

**自然梯度**：在变分参数空间中使用Fisher信息矩阵加权的梯度：

$$\tilde{\nabla}_\phi F = \mathbf{F}^{-1} \nabla_\phi F, \quad \mathbf{F} = \mathbb{E}_{q_\phi}[\nabla_\phi \ln q_\phi \nabla_\phi \ln q_\phi^T]$$

自然梯度考虑了参数空间的几何结构，收敛速度远快于普通梯度。

### 10.6 开源技术选型

| 技术需求 | 开源方案 | 说明 |
|---------|---------|------|
| 概率编程 | Pyro / NumPyro | 支持SVI、HMC、SMC等推断算法 |
| 粒子滤波 | particles (Python) | SMC和粒子滤波库 |
| 在线变分推断 | NumPyro (SVI) | 支持自定义变分族和优化器 |
| 贝叶斯神经网络 | BayesianTorch / Fortuna | 贝叶斯神经网络实现 |
| 自然梯度 | PyTorch (自定义) | 基于Fisher信息的自然梯度 |

### 10.7 实现路径

1. **共轭更新模块**：实现Beta-Bernoulli、Gaussian-Gaussian、Dirichlet-Multinomial的在线更新
2. **粒子滤波模块**：基于particles库实现序贯蒙特卡洛，支持重采样和Rao-Blackwell化
3. **变分在线学习**：基于NumPyro实现在线变分推断，支持自然梯度
4. **遗忘机制**：实现遗忘因子和滑动窗口两种非平稳适应机制
5. **验证**：在非平稳环境中（如分布漂移）验证在线贝叶斯更新的跟踪性能

---

## 阶段11：多臂老虎机与探索-利用权衡

### 11.1 随机多臂老虎机问题

多臂老虎机（Multi-Armed Bandit, MAB）问题是探索-利用权衡的最简形式化。设有 $K$ 个臂（行动选项），每个臂 $i$ 的奖励服从未知分布 $P_i$，均值为 $\mu_i$。Agent在每轮选择一个臂，获得该臂的随机奖励，目标是最大化累积奖励。

**遗憾（Regret）** 是衡量策略优劣的核心指标：

$$R_T = T\mu^* - \mathbb{E}\left[\sum_{t=1}^{T} r_t\right] = \sum_{i=1}^{K} \Delta_i \cdot \mathbb{E}[n_i(T)]$$

其中 $\mu^* = \max_i \mu_i$ 是最优臂的期望奖励，$\Delta_i = \mu^* - \mu_i$ 是臂 $i$ 的次优性间隙，$n_i(T)$ 是前 $T$ 轮中选择臂 $i$ 的次数。

**最优遗憾下界**（Lai & Robbins, 1985）：对于任何一致性策略，遗憾满足：

$$\liminf_{T \to \infty} \frac{R_T}{\ln T} \geq \sum_{i: \mu_i < \mu^*} \frac{\Delta_i}{D_{\text{KL}}(P_i \| P_{i^*})}$$

这意味着最优遗憾增长率为 $O(\ln T)$——对数渐近是最优的。

### 11.2 UCB算法

UCB（Upper Confidence Bound）算法是解决MAB问题的经典方法，其核心思想是：**选择置信上界最大的臂**——既考虑当前估计的均值（利用），又考虑估计的不确定性（探索）。

**UCB1算法**：

$$a_t = \arg\max_{i} \left[\hat{\mu}_i(t) + c \sqrt{\frac{\ln t}{n_i(t)}}\right]$$

其中 $\hat{\mu}_i(t)$ 是臂 $i$ 的样本均值，$n_i(t)$ 是臂 $i$ 被选择的次数，$c$ 是探索参数（通常取 $\sqrt{2}$）。

UCB1的遗憾上界：

$$R_T \leq \sum_{i: \Delta_i > 0} \left[\frac{8\ln T}{\Delta_i} + \left(1 + \frac{\pi^2}{3}\right)\Delta_i\right] = O\left(\sum_i \frac{\ln T}{\Delta_i}\right)$$

这达到了Lai-Robbins下界的量级（对数遗憾），是最优的。

**KL-UCB算法**：使用KL散度替代平方根项，更精确地刻画不确定性：

$$a_t = \arg\max_{i} \max\left\{\mu : D_{\text{KL}}(\hat{P}_i \| P_\mu) \leq \frac{f(t)}{n_i(t)}\right\}$$

其中 $f(t) = \ln t + 3\ln\ln t$。KL-UCB渐近匹配Lai-Robbins下界。

### 11.3 Thompson采样

Thompson采样（Thompson Sampling）是贝叶斯方法解决MAB问题的经典算法，其思想极其优雅：**按照后验概率选择最优臂**。

算法流程：

```
初始化: 每个臂的先验参数
For t = 1, 2, ...:
    1. 对每个臂 i，从后验分布采样: θ_i ~ p(θ_i | D_{1:t-1})
    2. 选择采样值最大的臂: a_t = argmax_i θ_i
    3. 观测奖励 r_t
    4. 更新选中臂的后验: p(θ_{a_t} | D_{1:t})
```

对于Bernoulli老虎机（奖励为0或1），使用Beta先验：

$$\theta_i \sim \text{Beta}(\alpha_i, \beta_i)$$

更新规则：若选中臂 $i$ 且获得奖励1，则 $\alpha_i \leftarrow \alpha_i + 1$；若奖励0，则 $\beta_i \leftarrow \beta_i + 1$。

Thompson采样的遗憾上界（Bernoulli情况）：

$$R_T \leq O\left(\sum_{i: \Delta_i > 0} \frac{\ln T}{\Delta_i}\right)$$

与UCB同量级，但实践中Thompson采样通常表现更好，因为其探索更加"智能"——探索频率自动适应臂的不确定性。

**Thompson采样与主动推断的关系**：Thompson采样可以理解为主动推断在MAB问题上的特化。策略选择 $q(\pi) \propto \exp(-\gamma G(\pi))$ 在 $\gamma \to \infty$ 时退化为贪心策略，在有限 $\gamma$ 时自然产生探索行为——这与Thompson采样的随机策略选择本质相同。

### 11.4 上下文老虎机

上下文老虎机（Contextual Bandit）在MAB基础上引入上下文信息：每轮Agent观测到上下文 $\mathbf{x}_t$，需要根据上下文选择臂。

$$a_t = \arg\max_i \mathbb{E}[r_t | \mathbf{x}_t, a_t = i]$$

**LinUCB算法**：假设奖励是上下文的线性函数：

$$\mathbb{E}[r_t | \mathbf{x}_t, a_t = i] = \mathbf{x}_t^T \boldsymbol{\theta}_i$$

UCB策略：

$$a_t = \arg\max_i \left[\mathbf{x}_t^T \hat{\boldsymbol{\theta}}_i + c \sqrt{\mathbf{x}_t^T \mathbf{A}_i^{-1} \mathbf{x}_t}\right]$$

其中 $\hat{\boldsymbol{\theta}}_i = \mathbf{A}_i^{-1} \mathbf{b}_i$ 是岭回归估计，$\mathbf{A}_i = \mathbf{I} + \sum_{\tau \in \mathcal{T}_i} \mathbf{x}_\tau \mathbf{x}_\tau^T$ 是设计矩阵加单位阵，$\mathbf{b}_i = \sum_{\tau \in \mathcal{T}_i} r_\tau \mathbf{x}_\tau$ 是奖励向量。

**NeuralUCB**：用神经网络替代线性模型，最后一层的特征用于计算不确定性：

$$a_t = \arg\max_i \left[f(\mathbf{x}_t, i; \hat{\theta}) + c \sqrt{\mathbf{g}_i^T \mathbf{A}^{-1} \mathbf{g}_i}\right]$$

其中 $\mathbf{g}_i = \nabla_\theta f(\mathbf{x}_t, i; \hat{\theta})$ 是梯度向量，$\mathbf{A}$ 是梯度的外积累加。

### 11.5 70/30探索利用比例的理论依据

在Agent系统设计中，我们提出70/30的探索-利用比例。这一比例的理论依据来自以下分析：

**1. 遗憾最优性分析**：在MAB问题中，最优策略将约 $O(\ln T)$ 次探索分配给次优臂，其余时间用于利用。对于有限时间 $T$，探索比例约为：

$$\rho_{\text{explore}} \approx \frac{K \ln T}{T}$$

当 $T$ 较小时（如 $T = 100, K = 10$），$\rho \approx 0.46$；当 $T$ 较大时（如 $T = 1000$），$\rho \approx 0.07$。70/30比例对应中等时间尺度上的平衡。

**2. 信息论分析**：在主动推断中，预期自由能的分解为：

$$G(\pi) = \underbrace{D_{\text{KL}}[q(o|\pi) \| C]}_{\text{外禀（利用）}} - \underbrace{I(\Psi; O|\pi)}_{\text{认知（探索）}}$$

最优策略的探索-利用比例取决于两项的相对量级。在环境不确定性较高时（Agent早期），认知价值占主导（探索为主）；在不确定性较低时（Agent后期），外禀价值占主导（利用为主）。70/30比例是中等不确定性下的经验最优值。

**3. 自适应比例**：更精确的实现不是固定比例，而是根据不确定性自适应调整：

$$\rho_{\text{explore}}(t) = \frac{H(\Psi | O_{1:t})}{H(\Psi | O_{1:0})}$$

即当前不确定性占总不确定性的比例。随着Agent学习，不确定性降低，探索比例自然下降——从初始的高探索逐渐过渡到高利用。

### 11.6 开源技术选型

| 技术需求 | 开源方案 | 说明 |
|---------|---------|------|
| MAB框架 | BayesianOptimization / SMAC | 贝叶斯优化与老虎机 |
| 上下文老虎机 | Vowpal Wabbit | 工业级上下文老虎机实现 |
| Thompson采样 | Pyro / NumPyro | 贝叶斯后验采样 |
| NeuralUCB | 自定义 (PyTorch) | 基于神经网络的不确定性估计 |
| 自适应探索 | 自定义 | 基于不确定性的探索比例调节 |

### 11.7 实现路径

1. **MAB核心**：实现UCB1、KL-UCB、Thompson采样三种策略
2. **上下文老虎机**：实现LinUCB和NeuralUCB，支持上下文感知的臂选择
3. **自适应探索模块**：实现基于不确定性的探索比例自适应调节
4. **Agent集成**：将MAB策略作为Agent行动选择的子模块，用于探索-利用权衡
5. **验证**：在合成和真实数据集上对比不同策略的遗憾性能

---

## 阶段12：时序预测理论——ARIMA、状态空间模型

### 12.1 ARIMA模型的数学基础

ARIMA（AutoRegressive Integrated Moving Average）模型是时序预测的经典方法，由三个组件构成：

**AR(p) 自回归部分**：

$$X_t = c + \sum_{i=1}^{p} \phi_i X_{t-i} + \varepsilon_t$$

其中 $\phi_i$ 是自回归系数，$\varepsilon_t \sim \text{WN}(0, \sigma^2)$ 是白噪声。

**MA(q) 滑动平均部分**：

$$X_t = \mu + \varepsilon_t + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j}$$

其中 $\theta_j$ 是滑动平均系数。

**ARMA(p,q) 组合模型**：

$$X_t = c + \sum_{i=1}^{p} \phi_i X_{t-i} + \varepsilon_t + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j}$$

**ARIMA(p,d,q)**：对非平稳序列进行 $d$ 阶差分使其平稳：

$$\Delta^d X_t = (1-B)^d X_t$$

其中 $B$ 是后移算子（$BX_t = X_{t-1}$）。差分后的序列用ARMA(p,q)建模。

**平稳性条件**：AR部分的特征方程 $1 - \phi_1 z - \phi_2 z^2 - \cdots - \phi_p z^p = 0$ 的所有根在单位圆外。

**可逆性条件**：MA部分的特征方程 $1 + \theta_1 z + \theta_2 z^2 + \cdots + \theta_q z^q = 0$ 的所有根在单位圆外。

**参数估计**：使用最大似然估计。对于ARMA(p,q)，似然函数为：

$$\ln L(\phi, \theta, \sigma^2) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{t=1}^{n} \varepsilon_t^2$$

其中 $\varepsilon_t$ 通过递推计算。

### 12.2 状态空间模型与卡尔曼滤波

状态空间模型提供了更灵活的时序建模框架，也是Agent多层预测架构的理论基础。

**状态空间表示**：

观测方程：$y_t = \mathbf{Z}_t^T \boldsymbol{\alpha}_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, H_t)$

状态方程：$\boldsymbol{\alpha}_{t+1} = \mathbf{T}_t \boldsymbol{\alpha}_t + \mathbf{R}_t \boldsymbol{\eta}_t, \quad \boldsymbol{\eta}_t \sim \mathcal{N}(0, Q_t)$

其中 $\boldsymbol{\alpha}_t$ 是不可观测的状态向量，$\mathbf{Z}_t$ 是观测矩阵，$\mathbf{T}_t$ 是状态转移矩阵，$\mathbf{R}_t$ 是噪声选择矩阵。

**卡尔曼滤波**：状态空间模型的最优递推估计。

预测步骤：

$$\hat{\boldsymbol{\alpha}}_{t|t-1} = \mathbf{T}_{t-1} \hat{\boldsymbol{\alpha}}_{t-1|t-1}$$

$$\mathbf{P}_{t|t-1} = \mathbf{T}_{t-1} \mathbf{P}_{t-1|t-1} \mathbf{T}_{t-1}^T + \mathbf{R}_{t-1} Q_{t-1} \mathbf{R}_{t-1}^T$$

更新步骤：

$$\mathbf{K}_t = \mathbf{P}_{t|t-1} \mathbf{Z}_t (\mathbf{Z}_t^T \mathbf{P}_{t|t-1} \mathbf{Z}_t + H_t)^{-1}$$

$$\hat{\boldsymbol{\alpha}}_{t|t} = \hat{\boldsymbol{\alpha}}_{t|t-1} + \mathbf{K}_t (y_t - \mathbf{Z}_t^T \hat{\boldsymbol{\alpha}}_{t|t-1})$$

$$\mathbf{P}_{t|t} = (\mathbf{I} - \mathbf{K}_t \mathbf{Z}_t^T) \mathbf{P}_{t|t-1}$$

其中 $\mathbf{K}_t$ 是卡尔曼增益，$y_t - \mathbf{Z}_t^T \hat{\boldsymbol{\alpha}}_{t|t-1}$ 是预测误差（创新）。

**卡尔曼滤波与预测编码的对应**：

| 卡尔曼滤波 | 预测编码 |
|-----------|---------|
| 预测步骤 | 自上而下的预测传播 |
| 更新步骤 | 自下而上的误差修正 |
| 卡尔曼增益 | 精度加权（不确定性高的层获得更多修正） |
| 预测误差（创新） | 预测误差信号 |
| 协方差矩阵 $\mathbf{P}$ | 后验不确定性 |

**扩展卡尔曼滤波（EKF）**：对于非线性状态空间模型，使用一阶Taylor展开线性化：

$$y_t = h(\boldsymbol{\alpha}_t) + \varepsilon_t, \quad \boldsymbol{\alpha}_{t+1} = f(\boldsymbol{\alpha}_t) + \boldsymbol{\eta}_t$$

EKF用Jacobian矩阵替代线性模型中的常数矩阵：

$$\mathbf{Z}_t = \frac{\partial h}{\partial \boldsymbol{\alpha}}\bigg|_{\hat{\boldsymbol{\alpha}}_{t|t-1}}, \quad \mathbf{T}_t = \frac{\partial f}{\partial \boldsymbol{\alpha}}\bigg|_{\hat{\boldsymbol{\alpha}}_{t|t}}$$

### 12.3 在线时序预测算法

Agent需要在线进行时序预测——每获得一个新数据点就更新预测模型。

**在线ARIMA更新**：

1. **递推最小二乘**：对AR部分参数进行在线更新

$$\hat{\boldsymbol{\phi}}_{t+1} = \hat{\boldsymbol{\phi}}_t + \frac{\mathbf{P}_t \mathbf{x}_t}{1 + \mathbf{x}_t^T \mathbf{P}_t \mathbf{x}_t}(y_t - \mathbf{x}_t^T \hat{\boldsymbol{\phi}}_t)$$

$$\mathbf{P}_{t+1} = \mathbf{P}_t - \frac{\mathbf{P}_t \mathbf{x}_t \mathbf{x}_t^T \mathbf{P}_t}{1 + \mathbf{x}_t^T \mathbf{P}_t \mathbf{x}_t}$$

2. **遗忘因子**：引入遗忘因子 $\lambda$ 使旧数据权重衰减

$$\mathbf{P}_{t+1} = \frac{1}{\lambda}\left(\mathbf{P}_t - \frac{\mathbf{P}_t \mathbf{x}_t \mathbf{x}_t^T \mathbf{P}_t}{\lambda + \mathbf{x}_t^T \mathbf{P}_t \mathbf{x}_t}\right)$$

**在线状态空间更新**：卡尔曼滤波本身就是在线算法——每步只需 $O(n^2)$ 计算（$n$ 是状态维度）。

**变阶模型选择**：在线调整ARIMA的阶数 (p,d,q)：

- 使用BIC准则：$\text{BIC} = -2\ln L + k \ln n$
- 定期重新评估阶数选择
- 使用滑动窗口检测结构性变化

### 12.4 Agent多层时序预测架构映射

Agent系统需要在不同时间尺度上进行预测——从毫秒级的感知预测到小时/天级的环境趋势预测。

**三层时序预测架构**：

| 层级 | 时间尺度 | 模型 | 功能 |
|------|---------|------|------|
| 快速层 | 毫秒-秒 | 卡尔曼滤波 | 即时感知预测 |
| 中速层 | 秒-分钟 | ARIMA/RNN | 短期趋势预测 |
| 慢速层 | 分钟-小时 | 层级贝叶斯 | 长期环境建模 |

**层级时间尺度分离**：不同层级的更新频率不同：

$$\tau_{\text{fast}} \ll \tau_{\text{medium}} \ll \tau_{\text{slow}}$$

快速层每步更新，中速层每 $N_{\text{med}}$ 步更新，慢速层每 $N_{\text{slow}}$ 步更新。这种多时间尺度架构对应了大脑中不同脑区的处理速度差异。

**跨层信息传递**：

- 快速层向中速层传递：短期预测误差的统计量（均值、方差）
- 中速层向慢速层传递：中期趋势特征
- 慢速层向中速层传递：长期先验（调节中速层的模型参数）
- 中速层向快速层传递：短期趋势预测（作为快速层的参考信号）

### 12.5 开源技术选型

| 技术需求 | 开源方案 | 说明 |
|---------|---------|------|
| ARIMA | statsmodels | 完整的ARIMA建模与预测 |
| 状态空间模型 | statsmodels / FilterPy | 状态空间模型与卡尔曼滤波 |
| 在线时序预测 | River | 在线机器学习库，支持时序预测 |
| 神经时序模型 | PyTorch Forecasting | 基于Transformer的时序预测 |
| 异常检测 | ADTK / scikit-learn | 时序异常检测工具 |

### 12.6 实现路径

1. **ARIMA模块**：基于statsmodels实现ARIMA的自动阶数选择和在线预测
2. **卡尔曼滤波模块**：基于FilterPy实现标准EKF和UKF，支持非线性状态空间
3. **在线更新**：基于River实现递推参数更新，支持遗忘因子
4. **多层架构**：实现三层时序预测架构，支持跨层信息传递
5. **验证**：在合成时序和非平稳时序数据上验证多层预测的准确性和适应性

---

## 阶段13：异常检测理论——统计检验、分布偏移

### 13.1 统计假设检验基础

异常检测是Agent安全驱力的核心——当环境偏离预期时，Agent必须及时检测并响应。

**假设检验框架**：

- 零假设 $H_0$：系统处于正常状态
- 备择假设 $H_1$：系统处于异常状态

**第一类错误**（假阳性）：$H_0$ 为真时拒绝 $H_0$，概率 $\alpha$（显著性水平）

**第二类错误**（假阴性）：$H_1$ 为真时接受 $H_0$，概率 $\beta$

**检验功效**：$1 - \beta$，正确检测异常的概率

**似然比检验**：最优检验统计量（Neyman-Pearson引理）：

$$\Lambda = \frac{\sup_{\theta \in \Theta_0} L(\theta)}{\sup_{\theta \in \Theta} L(\theta)} = \frac{L(\hat{\theta}_0)}{L(\hat{\theta})}$$

拒绝域：$\Lambda < c$，其中 $c$ 由显著性水平 $\alpha$ 决定。

在Agent系统中，似然比检验对应：**当前观测是否可以用现有生成模型解释？** 如果不能（似然比过低），则检测到异常。

### 13.2 均值漂移检测（CUSUM、EWMA）

**CUSUM（累积和）控制图**：

CUSUM检测序列均值的持续偏移。定义累积和：

$$S_t^+ = \max(0, S_{t-1}^+ + (x_t - \mu_0 - k))$$

$$S_t^- = \max(0, S_{t-1}^- - (x_t - \mu_0 + k))$$

其中 $\mu_0$ 是目标均值，$k$ 是参考值（通常取 $k = \delta/2$，$\delta$ 是要检测的最小偏移量）。

报警条件：$S_t^+ > h$ 或 $S_t^- > h$，$h$ 是决策阈值。

CUSUM的理论性质：对于从时刻 $\nu$ 开始的均值偏移 $\delta$，平均检测延迟为：

$$\text{ARL}_1 \approx \frac{h}{\delta - k} + O(1)$$

而误报的平均运行长度为：

$$\text{ARL}_0 \approx \frac{e^{2kh/\sigma^2}}{2k^2/\sigma^2}$$

**EWMA（指数加权移动平均）控制图**：

$$z_t = \lambda x_t + (1-\lambda) z_{t-1}$$

控制限：

$$\text{UCL} = \mu_0 + L\sigma\sqrt{\frac{\lambda}{2-\lambda}[1-(1-\lambda)^{2t}]}$$

$$\text{LCL} = \mu_0 - L\sigma\sqrt{\frac{\lambda}{2-\lambda}[1-(1-\lambda)^{2t}]}$$

其中 $\lambda \in (0,1]$ 是平滑参数，$L$ 是控制限宽度（通常取3）。

EWMA对微小均值偏移特别敏感，且 $\lambda$ 越小对历史数据权重越大，越适合检测小偏移。

**CUSUM与EWMA在Agent中的应用**：

- **预测误差监控**：对预测误差序列应用CUSUM/EWMA，检测模型性能退化
- **环境变化检测**：对环境观测统计量应用CUSUM/EWMA，检测分布偏移
- **安全阈值监控**：对Agent关键指标应用EWMA，确保不超出安全范围

### 13.3 分布偏移检测（KS检验、MMD）

当不仅均值发生偏移，而是整个分布发生变化时，需要更强的检测方法。

**Kolmogorov-Smirnov（KS）检验**：

KS检验度量两个经验分布函数的最大距离：

$$D_{n,m} = \sup_x |F_n(x) - G_m(x)|$$

其中 $F_n$ 和 $G_m$ 分别是样本量为 $n$ 和 $m$ 的经验分布函数。

在零假设（同分布）下：

$$P(D_{n,m} > c) \leq 2\exp\left(-\frac{2c^2 nm}{n+m}\right)$$

KS检验的优点：非参数、对分布形状无假设。缺点：对尾部差异不敏感。

**最大均值差异（Maximum Mean Discrepancy, MMD）**：

MMD是再生核希尔伯特空间（RKHS）中的分布距离：

$$\text{MMD}^2(P, Q) = \mathbb{E}_{x,x' \sim P}[k(x,x')] - 2\mathbb{E}_{x \sim P, y \sim Q}[k(x,y)] + \mathbb{E}_{y,y' \sim Q}[k(y,y')]$$

其中 $k$ 是正定核函数（通常取高斯核 $k(x,y) = \exp(-\|x-y\|^2/(2\sigma^2))$）。

无偏估计量：

$$\widehat{\text{MMD}}^2 = \frac{1}{n(n-1)}\sum_{i \neq j} k(x_i, x_j) + \frac{1}{m(m-1)}\sum_{i \neq j} k(y_i, y_j) - \frac{2}{nm}\sum_{i,j} k(x_i, y_j)$$

MMD检验的阈值通过核矩阵的特征值分解或置换检验确定。

**在线分布偏移检测**：

使用滑动窗口方法：维护参考窗口 $W_{\text{ref}}$ 和检测窗口 $W_{\text{test}}$，计算两个窗口之间的MMD：

$$\text{MMD}^2(W_{\text{ref}}, W_{\text{test}}) \gtrless \eta$$

当MMD超过阈值 $\eta$ 时，触发分布偏移警报。

### 13.4 Agent安全驱力的异常检测映射

在Agent系统中，异常检测直接映射到安全驱力——这是自由能原理中"避免意外"的体现。

**安全驱力的数学描述**：

$$\text{Safety}(t) = -D_{\text{KL}}[p(o_t | \theta) \| p(o_t | \theta_{\text{safe}})]$$

其中 $p(o_t | \theta_{\text{safe}})$ 是安全状态下的观测分布。当安全驱力低于阈值时，Agent进入"警惕模式"：

1. **增加探索**：提高认知价值的权重，获取更多环境信息
2. **保守行动**：选择风险更低的策略
3. **模型更新**：加速内部模型的在线学习
4. **层级警报**：向上层预测模块发送异常信号

**多尺度异常检测**：

| 层级 | 检测方法 | 响应时间 | 功能 |
|------|---------|---------|------|
| 感知层 | 预测误差阈值 | 毫秒 | 即时异常检测 |
| 趋势层 | CUSUM/EWMA | 秒 | 均值漂移检测 |
| 分布层 | KS/MMD | 分钟 | 分布偏移检测 |
| 结构层 | 因果图变化 | 小时 | 结构性变化检测 |

### 13.5 开源技术选型

| 技术需求 | 开源方案 | 说明 |
|---------|---------|------|
| 统计检验 | SciPy (stats模块) | KS检验、t检验等 |
| CUSUM/EWMA | 自定义 / SPC库 | 统计过程控制工具 |
| MMD计算 | ott-jax | 最优传输与MMD计算 |
| 在线异常检测 | River / PyOD | 在线异常检测框架 |
| 置换检验 | 自定义 | 基于NumPy的快速置换检验 |

### 13.6 实现路径

1. **预测误差监控**：实现基于阈值的即时异常检测
2. **CUSUM/EWMA模块**：实现均值漂移的在线检测，支持自适应阈值
3. **MMD检测模块**：实现基于核方法的分布偏移在线检测
4. **安全驱力集成**：将异常检测与Agent的安全驱力模块集成
5. **多尺度检测**：实现从毫秒到小时的多尺度异常检测体系
6. **验证**：在分布漂移基准数据集上验证检测延迟和误报率

---

## 阶段14：因果推断基础——因果图、do-演算

### 14.1 因果图模型（DAG）

因果图模型用有向无环图（DAG）表示变量之间的因果关系。图 $\mathcal{G} = (\mathbf{V}, \mathbf{E})$ 中，节点 $V_i$ 表示随机变量，有向边 $V_i \to V_j$ 表示 $V_i$ 是 $V_j$ 的直接原因。

**因果图的结构假设**：

1. **因果马尔可夫条件**：每个变量在给定其直接原因（父节点）的条件下，与非后代变量独立

$$V_i \perp\!\!\!\perp \text{NonDesc}(V_i) | \text{Pa}(V_i)$$

2. **因果充分性**：所有共同原因都被纳入图中（无隐混淆因子）
3. **因果忠诚性**：条件独立性仅由因果马尔可夫条件蕴含，不存在偶然的独立

**因果图的因子分解**：

$$P(V_1, V_2, \ldots, V_n) = \prod_{i=1}^{n} P(V_i | \text{Pa}(V_i))$$

**d-分离准则**：判断因果图中条件独立性的图论准则。路径 $p$ 被节点集 $\mathbf{Z}$ 阻断，当且仅当：

- 路径中存在链 $A \to B \to C$ 或叉 $A \leftarrow B \to C$，且 $B \in \mathbf{Z}$
- 路径中存在对撞 $A \to B \leftarrow C$，且 $B \notin \mathbf{Z}$ 且 $B$ 的后代不在 $\mathbf{Z}$ 中

若 $\mathbf{Z}$ 阻断了 $X$ 和 $Y$ 之间的所有路径，则 $X \perp\!\!\!\perp Y | \mathbf{Z}$。

### 14.2 do-演算与干预效应

Judea Pearl的do-演算是因果推断的核心工具，它区分了"观察"和"干预"：

- **观察** $P(Y | X=x)$：在自然条件下，观察到 $X=x$ 时 $Y$ 的分布
- **干预** $P(Y | \text{do}(X=x))$：强制设定 $X=x$ 时 $Y$ 的分布

**干预的图论操作**：$\text{do}(X=x)$ 对应从因果图中删除所有指向 $X$ 的边（切断 $X$ 的所有原因），然后设定 $X=x$：

$$P(Y | \text{do}(X=x)) = \sum_{\mathbf{V} \setminus \{X,Y\}} \prod_{V_i \neq X} P(V_i | \text{Pa}(V_i)) \bigg|_{X=x}$$

**do-演算的三条规则**：

**规则1（插入/删除观察）**：若 $Y \perp\!\!\!\perp Z | X, \mathbf{W}$ 在 $\mathcal{G}_{\overline{X}}$ 中成立：

$$P(Y | \text{do}(X), Z, \mathbf{W}) = P(Y | \text{do}(X), \mathbf{W})$$

**规则2（交换干预与观察）**：若 $Y \perp\!\!\!\perp Z | X, \mathbf{W}$ 在 $\mathcal{G}_{\overline{X}\underline{Z}}$ 中成立：

$$P(Y | \text{do}(X), \text{do}(Z), \mathbf{W}) = P(Y | \text{do}(X), Z, \mathbf{W})$$

**规则3（插入/删除干预）**：若 $Y \perp\!\!\!\perp Z | X, \mathbf{W}$ 在 $\mathcal{G}_{\overline{X}, \overline{Z(W)}}$ 中成立：

$$P(Y | \text{do}(X), \text{do}(Z), \mathbf{W}) = P(Y | \text{do}(X), \mathbf{W})$$

**后门准则**：若变量集 $\mathbf{Z}$ 满足后门准则（阻断 $X$ 到 $Y$ 的所有后门路径），则：

$$P(Y | \text{do}(X=x)) = \sum_{\mathbf{z}} P(Y | X=x, \mathbf{Z}=\mathbf{z}) P(\mathbf{Z}=\mathbf{z})$$

**前门准则**：当后门准则不满足时，若存在中介变量集 $\mathbf{M}$ 满足前门准则：

$$P(Y | \text{do}(X=x)) = \sum_{\mathbf{m}} P(\mathbf{M}=\mathbf{m} | X=x) \sum_{x'} P(Y | X=x', \mathbf{M}=\mathbf{m}) P(X=x')$$

### 14.3 因果发现算法（PC、FCI）

因果发现是从观测数据中学习因果图结构的过程。

**PC算法**（约束型因果发现）：

1. **骨架发现**：从完全图开始，逐步删除条件独立性检验不通过的边
   - 对每对 $(X_i, X_j)$，检验 $X_i \perp\!\!\!\perp X_j | \mathbf{S}$（$\mathbf{S}$ 是其余变量的子集）
   - 若存在 $\mathbf{S}$ 使条件独立成立，删除边 $X_i - X_j$，记录 $\mathbf{S}$ 为分离集
2. **方向确定**：对每个三元组 $X_i - X_k - X_j$（$X_i$ 和 $X_j$ 不相邻）
   - 若 $X_k \notin \text{Sep}(X_i, X_j)$，定向为 $X_i \to X_k \leftarrow X_j$（对撞结构）
   - 通过Meek规则传播方向

PC算法的复杂度：最坏 $O(n^{d+2})$，其中 $d$ 是最大条件集大小。

**FCI算法**（含隐混淆因子的因果发现）：

FCI是PC的推广，允许隐混淆因子的存在。FCI输出部分祖先图（PAG），使用 $\circ \to$、$\circ - \circ$ 等标记表示方向的不确定性。

**基于分数的因果发现**：

NOTEARS算法将因果发现转化为连续优化：

$$\min_{\mathbf{W}} \frac{1}{2n}\|\mathbf{X} - \mathbf{X}\mathbf{W}\|_F^2 + \lambda \|\mathbf{W}\|_1 \quad \text{s.t.} \quad \text{tr}(e^{\mathbf{W} \circ \mathbf{W}}) - d = 0$$

其中DAG约束 $\text{tr}(e^{\mathbf{W} \circ \mathbf{W}}) - d = 0$ 保证结果是无环的，$\circ$ 是Hadamard积。

### 14.4 Agent认知中的因果关系建模

因果推断在Agent系统中的核心作用：**使Agent能够区分相关与因果，从而做出有效的干预决策**。

**Agent的因果模型层次**：

1. **关联层**：$P(Y | X=x)$——观察到 $X=x$ 时 $Y$ 的概率（被动感知）
2. **干预层**：$P(Y | \text{do}(X=x))$——设定 $X=x$ 时 $Y$ 的概率（主动行动）
3. **反事实层**：$P(Y_{X=x} | X=x', Y=y)$——若当时做了不同选择会怎样（反思学习）

**因果模型与生成模型的关系**：

Agent的生成模型 $p(o, \psi)$ 本质上是一个因果图——它描述了隐藏状态如何因果地产生观测。因果发现对应于Agent从观测中学习生成模型的结构。

**因果驱动的行动选择**：

在主动推断中，Agent通过 $\text{do}(a)$ 改变环境。因果模型使Agent能够预测行动的因果效应：

$$P(o' | \text{do}(a)) = \sum_{\psi} P(o' | \psi, a) P(\psi | o)$$

这比纯关联预测 $P(o' | a)$ 更准确，因为它正确处理了混淆因子。

**反事实学习**：

Agent可以利用反事实推理进行反思学习——评估"如果当时做了不同选择，结果会怎样"：

$$\text{反事实遗憾} = r_{\text{actual}} - \mathbb{E}[r_{\text{counterfactual}} | \text{do}(a' \neq a)]$$

这种反事实评估为Agent的策略改进提供了更丰富的信号。

### 14.5 开源技术选型

| 技术需求 | 开源方案 | 说明 |
|---------|---------|------|
| 因果发现 | causal-learn (CMU) | PC、FCI等因果发现算法 |
| 因果推断 | DoWhy (Microsoft) | do-演算、反事实推理 |
| 因果效应估计 | EconML (Microsoft) | 因果效应的机器学习估计 |
| DAG约束优化 | NOTEARS | 连续优化因果发现 |
| 因果图可视化 | networkx / pygraphviz | 因果图的可视化与分析 |

### 14.6 实现路径

1. **因果图构建**：基于causal-learn实现PC/FCI因果发现，从Agent观测数据中学习因果结构
2. **干预推理**：基于DoWhy实现do-演算，支持后门/前门准则的自动识别
3. **反事实模块**：实现反事实推理，支持Agent的反思学习
4. **因果-生成模型统一**：将因果图与生成模型统一，使Agent的内部模型具有因果语义
5. **验证**：在标准因果推断基准（如IHDP、Lalonde）上验证因果效应估计的准确性

---

## 阶段15：元学习理论——学会学习

### 15.1 元学习的形式化定义

元学习（Meta-Learning）是"学会学习"的理论框架——系统不仅学习解决特定任务，还学习如何更高效地学习新任务。

**形式化定义**：设任务分布 $\mathcal{T} \sim P(\mathcal{T})$，每个任务 $\mathcal{T}_i$ 由数据分布 $P_i(\mathbf{x}, y)$ 定义。元学习的目标是找到元参数 $\omega$，使得对于从 $P(\mathcal{T})$ 采样的新任务 $\mathcal{T}_j$，系统可以快速适应：

$$\omega^* = \arg\min_\omega \mathbb{E}_{\mathcal{T} \sim P(\mathcal{T})} \left[\mathcal{L}_{\mathcal{T}}(\theta_j^*)\right]$$

其中 $\theta_j^* = \text{Adapt}(\omega, \mathcal{D}_j^{\text{train}})$ 是在任务 $\mathcal{T}_j$ 的训练集上适应后的参数。

**元学习的三层次**：

1. **元学习器**：学习如何更新参数（学习规则）
2. **基础学习器**：在特定任务上学习参数
3. **任务分布**：定义了"学习"本身的评价标准

### 15.2 MAML算法

MAML（Model-Agnostic Meta-Learning）是最具影响力的元学习算法，由Finn et al.（2017）提出。

**核心思想**：寻找一个参数初始化 $\theta$，使得对于任何新任务，只需一步或少步梯度下降就能达到好的性能。

**MAML的目标函数**：

$$\min_\theta \sum_{\mathcal{T}_i \sim P(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(U_{\mathcal{T}_i}(\theta))$$

其中 $U_{\mathcal{T}_i}(\theta) = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{\text{train}}(\theta)$ 是一步梯度更新。

**MAML的梯度计算**（通过梯度更新链的元梯度）：

$$\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(U_{\mathcal{T}_i}(\theta)) = \nabla_{U_\theta} \mathcal{L}_{\mathcal{T}_i}^{\text{test}} \cdot \left(I - \alpha \nabla_\theta^2 \mathcal{L}_{\mathcal{T}_i}^{\text{train}}(\theta)\right)$$

其中第二项涉及Hessian矩阵。在实践中，常用一阶近似（FOMAML）忽略Hessian项：

$$\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(U_{\mathcal{T}_i}(\theta)) \approx \nabla_{U_\theta} \mathcal{L}_{\mathcal{T}_i}^{\text{test}}$$

**MAML算法流程**：

```
初始化: 元参数 θ
For 每个元训练迭代:
    1. 采样任务批次 {T_i}
    2. For 每个任务 T_i:
        a. 计算训练损失 L_train(T_i, θ)
        b. 内循环更新: θ_i' = θ - α ∇_θ L_train(T_i, θ)
        c. 计算测试损失 L_test(T_i, θ_i')
    3. 元更新: θ ← θ - β ∇_θ Σ_i L_test(T_i, θ_i')
```

**Reptile**：MAML的简化版本，不需要二阶梯度：

$$\theta \leftarrow \theta + \epsilon (\theta_i' - \theta)$$

其中 $\theta_i'$ 是在任务 $\mathcal{T}_i$ 上训练 $K$ 步后的参数。Reptile收敛到与MAML类似的解，但计算效率更高。

### 15.3 在线元学习

在线元学习处理任务序列在线到达的情况——Agent必须在有限交互中快速适应新任务，同时利用过去的经验。

**在线元学习框架**：

在时刻 $t$，Agent面对任务 $\mathcal{T}_t$，使用当前元参数 $\omega_t$ 快速适应：

$$\theta_t = \text{Adapt}(\omega_t, \mathcal{D}_t^{\text{train}})$$

在任务 $\mathcal{T}_t$ 上获得损失 $\mathcal{L}_t(\theta_t)$ 后，更新元参数：

$$\omega_{t+1} = \omega_t - \eta_t \nabla_\omega \mathcal{L}_t(\theta_t)$$

**遗憾分析**：在线元学习的遗憾定义为：

$$R_T = \sum_{t=1}^{T} \left[\mathcal{L}_t(\theta_t) - \mathcal{L}_t(\theta_t^*)\right]$$

其中 $\theta_t^*$ 是任务 $\mathcal{T}_t$ 上的最优参数。目标是最小化 $R_T$，使其增长尽可能慢（理想为 $O(\sqrt{T})$ 或 $O(\ln T)$）。

**Follow the Meta-Leader（FTML）**：

$$\omega_{t+1} = \arg\min_\omega \sum_{\tau=1}^{t} \mathcal{L}_\tau(\text{Adapt}(\omega, \mathcal{D}_\tau^{\text{train}})) + \frac{1}{2\eta}\|\omega - \omega_0\|^2$$

FTML是在线元学习的后悔最小化策略，类似于在线学习的Follow the Regularized Leader。

### 15.4 Agent如何实现"学会更好地进化"

在自驱永续进化Agent系统中，元学习赋予Agent"学会更好地进化"的能力——不仅进化模型参数，还进化学习规则本身。

**三层进化架构**：

| 层级 | 进化对象 | 时间尺度 | 方法 |
|------|---------|---------|------|
| 快速适应 | 模型参数 $\theta$ | 单任务 | 梯度下降/贝叶斯更新 |
| 元学习 | 学习规则 $\omega$ | 跨任务 | MAML/Reptile |
| 结构进化 | 模型架构 $\alpha$ | 长期 | (1+1)-ES/遗传算法 |

**元学习的进化实现**：

将学习率、正则化系数、更新规则等超参数作为元参数 $\omega$，通过进化策略优化：

$$\omega_{t+1} = \omega_t + \sigma \xi_t, \quad \text{if } F(\text{Adapt}(\omega_t + \sigma\xi_t)) < F(\text{Adapt}(\omega_t))$$

这种"进化学习规则"的方法使Agent能够自动发现最优的学习策略，而无需人工设计。

**自指进化**：更激进的想法是让Agent进化自身的进化算法——即元-元学习。虽然理论上可行（通过嵌套的元学习），但实践中需要谨慎控制递归深度以避免不稳定。

**元学习与自由能原理的统一**：

在自由能框架下，元学习对应于**优化变分推断的推断策略**——不仅最小化当前自由能，还优化自由能最小化的过程本身：

$$\omega^* = \arg\min_\omega \mathbb{E}_{\mathcal{T}}\left[\min_\theta F(\theta, \omega; \mathcal{T})\right]$$

其中 $\omega$ 控制推断过程的超参数（如学习率、迭代次数、变分族的选择等）。

### 15.5 开源技术选型

| 技术需求 | 开源方案 | 说明 |
|---------|---------|------|
| 元学习框架 | learn2learn | PyTorch元学习库，支持MAML等 |
| MAML实现 | torchmeta | 少样本元学习基准与算法 |
| 在线元学习 | 自定义 (JAX) | 基于JAX的在线元学习实现 |
| AutoML | AutoPyTorch / FLAML | 自动超参数优化 |
| 进化元学习 | evosax (JAX) | 用ES优化学习规则 |

### 15.6 实现路径

1. **MAML核心**：基于learn2learn实现MAML和Reptile，支持任意基础学习器
2. **在线元学习**：实现在线元学习框架，支持任务序列的增量适应
3. **元学习规则进化**：实现基于ES的学习规则进化，自动发现最优超参数
4. **三层进化集成**：将快速适应、元学习、结构进化集成为统一的三层架构
5. **验证**：在少样本学习基准（Omniglot、Mini-ImageNet）和非平稳环境中验证元学习的适应速度

---

## 阶段16：神经可塑性仿生——Hebb学习、STDP

### 16.1 Hebb学习规则

Donald Hebb于1949年提出的假说是神经可塑性理论的奠基石："一起激发的神经元连在一起"（Neurons that fire together, wire together）。其数学形式化：

$$\Delta w_{ij} = \eta \cdot x_i \cdot y_j$$

其中 $w_{ij}$ 是从神经元 $i$ 到神经元 $j$ 的突触权重，$x_i$ 是突触前活动，$y_j$ 是突触后活动，$\eta$ 是学习率。

Hebb规则的问题：**权重只增不减**，导致所有连接最终饱和。为解决这一问题，引入多种修正：

**Oja规则**（归一化Hebb学习）：

$$\Delta w_{ij} = \eta (x_i y_j - y_j^2 w_{ij})$$

Oja规则的第一项是标准Hebb学习，第二项是遗忘项（防止权重无限增长）。可以证明，Oja规则使权重收敛到输入协方差矩阵的第一主成分——即**Oja规则实现了在线PCA**。

**Covariance规则**：

$$\Delta w_{ij} = \eta (x_i - \bar{x}_i)(y_j - \bar{y}_j)$$

使用去均值后的活动，使权重可以增减。这等价于输入与输出的协方差，连接了Hebb学习与统计相关性。

**BCM规则**（Bienenstock-Cooper-Munro）：

$$\Delta w_{ij} = \eta \cdot x_i \cdot y_j (y_j - \theta_M)$$

其中 $\theta_M = \mathbb{E}[y_j^2]$ 是滑动阈值（突触后活动的均方）。当 $y_j > \theta_M$ 时权重增强，$y_j < \theta_M$ 时权重减弱。BCM规则实现了**选择性**——神经元对特定输入模式变得选择性响应，同时保持对其他模式的抑制。

### 16.2 STDP（脉冲时间依赖可塑性）

STDP是Hebb学习的精化，它考虑了突触前后脉冲的精确时间关系。其核心发现：**突触前脉冲先于突触后脉冲时，突触增强；反之则减弱**。

**STDP窗口函数**：

$$\Delta w = \begin{cases} A_+ \exp(-\Delta t / \tau_+) & \text{if } \Delta t > 0 \quad (\text{突触前先于突触后}) \\ -A_- \exp(\Delta t / \tau_-) & \text{if } \Delta t < 0 \quad (\text{突触后先于突触前}) \end{cases}$$

其中 $\Delta t = t_{\text{post}} - t_{\text{pre}}$ 是突触后脉冲与突触前脉冲的时间差，$A_+, A_-$ 是学习率幅度，$\tau_+, \tau_-$ 是时间常数（通常约10-20ms）。

**STDP的计算功能**：

1. **因果学习**：只有因果相关的输入-输出对被增强——突触前活动"导致"突触后活动的连接被强化
2. **时序编码**：STDP自然地学习输入的时序结构
3. **竞争学习**：不同突触竞争驱动突触后神经元，形成稀疏表征

**STDP与预测编码的对应**：

在预测编码框架中，STDP对应于权重更新规则：

$$\Delta W^l \propto e^{l-1} \cdot (r^l)^T$$

即低层预测误差与高层表征的外积。这可以理解为：当高层表征成功预测低层活动时（$e^{l-1}$ 小），连接增强；当预测失败时（$e^{l-1}$ 大且方向相反），连接减弱。STDP的时间窗口确保了这种因果性——只有先于预测误差的表征才被增强。

### 16.3 突触巩固与遗忘

**稳定性-可塑性困境**：学习新信息的同时不能灾难性地遗忘旧信息——这是Agent系统的核心挑战。

**突触巩固机制**：

1. **突触标记与捕获**：近期激活的突触被"标记"，标记的突触优先捕获巩固因子（蛋白质），从而实现选择性巩固
2. **权重固化**：权重越大，改变越困难——实现方式：

$$\Delta w_{ij} = \eta \cdot f(w_{ij}) \cdot x_i \cdot y_j, \quad f(w) = \frac{1}{1 + (w/w_0)^n}$$

其中 $f(w)$ 是巩固因子，大权重（已巩固的连接）更难改变。

3. **Elastic Weight Consolidation（EWC）**：基于Fisher信息矩阵的巩固方法：

$$\mathcal{L}_{\text{EWC}} = \mathcal{L}_{\text{new}} + \frac{\lambda}{2} \sum_i F_i (w_i - w_i^*)^2$$

其中 $F_i$ 是Fisher信息的对角元素（度量参数 $w_i$ 对输出的重要性），$w_i^*$ 是旧任务的最优参数。EWC使重要参数的偏移受到惩罚，从而保护旧知识。

**互补学习系统理论**：

海马体（快速学习、临时存储）与新皮层（慢速学习、长期存储）的互补：

- **海马体**：高可塑性，快速编码新经验，但容量有限、容易干扰
- **新皮层**：低可塑性，慢速整合新知识到已有结构，但容量大、抗干扰

**经验回放**：在"离线"状态下，海马体重放旧经验，使新皮层逐步整合——这对应了Agent的"睡眠学习"机制：

$$\Delta w = \eta_{\text{replay}} \sum_{k \in \text{replay buffer}} \nabla_w \mathcal{L}(x_k, y_k)$$

### 16.4 Agent模型参数的仿生更新规则

将神经可塑性原理映射到Agent的参数更新规则：

**仿生更新规则**：

$$\theta_{t+1} = \theta_t - \eta \cdot g(\theta_t) \cdot \nabla_\theta F_t$$

其中 $g(\theta)$ 是可塑性门控函数，控制每个参数的可塑性：

$$g(\theta_i) = \frac{1}{1 + (\theta_i / \theta_0)^n + \text{age}_i / \tau_{\text{consolidate}}}$$

- $(\theta_i / \theta_0)^n$：大权重更难改变（权重固化）
- $\text{age}_i / \tau_{\text{consolidate}}$：旧参数更难改变（时间巩固）

**STDP式更新**：在预测编码框架中，使用时间差分代替同时活动：

$$\Delta W^l = \eta \left(e^{l-1}_t \cdot (r^l_{t-\delta})^T - \lambda_{\text{decay}} W^l\right)$$

其中 $\delta$ 是时间延迟（对应STDP的时间窗口），$\lambda_{\text{decay}}$ 是衰减项。

**三阶段学习循环**：

1. **在线学习**（快速可塑性）：使用当前数据更新参数，高学习率
2. **经验回放**（中速巩固）：从回放缓冲区采样，中等学习率
3. **结构进化**（慢速优化）：使用进化策略调整架构，低变异率

### 16.5 开源技术选型

| 技术需求 | 开源方案 | 说明 |
|---------|---------|------|
| 脉冲神经网络 | Norse / snnTorch | 基于PyTorch的脉冲神经网络 |
| STDP仿真 | BindsNET | 脉冲神经网络与STDP学习 |
| 持续学习 | Avalanche | 持续学习基准与算法（EWC等） |
| 经验回放 | 自定义 (Replay Buffer) | 基于优先级的回放缓冲区 |
| 可塑性规则 | 自定义 (JAX) | 仿生可塑性门控函数 |

### 16.6 实现路径

1. **仿生更新规则**：实现基于权重固化和时间巩固的可塑性门控
2. **STDP式学习**：在预测编码框架中实现时间差分式权重更新
3. **持续学习模块**：基于Avalanche实现EWC等持续学习算法
4. **经验回放系统**：实现优先级经验回放，支持在线和离线学习
5. **三阶段循环**：实现在线学习-经验回放-结构进化的三阶段学习循环
6. **验证**：在持续学习基准（Permuted MNIST、Split CIFAR）上验证抗遗忘性能

---

## 阶段17：记忆系统理论——工作记忆、长期记忆

### 17.1 Atkinson-Shiffrin记忆模型

Atkinson-Shiffrin模型（1968）是记忆系统的经典理论框架，将人类记忆分为三个存储：

**感觉记忆**：瞬时存储，容量大但持续时间极短（<1秒）

$$\text{容量} \approx \infty, \quad \text{持续时间} \approx 200\text{ms}$$

**短时记忆/工作记忆**：有限容量，有限持续时间

$$\text{容量} \approx 7 \pm 2 \text{ 项}, \quad \text{持续时间} \approx 15\text{-}30\text{s}$$

**长时记忆**：近乎无限容量，持久存储

$$\text{容量} \approx \infty, \quad \text{持续时间} \approx \text{数年至终身}$$

信息流动：感觉记忆 → 工作记忆 → 长时记忆，通过注意力和复述机制控制。

**Baddeley工作记忆模型**（扩展）：

- **中央执行系统**：注意力控制，协调各子系统
- **语音回路**：处理语言信息
- **视空间模板**：处理视觉和空间信息
- **情景缓冲区**：整合多模态信息的临时存储

### 17.2 工作记忆的容量限制

工作记忆的容量限制（Miller's Law: 7±2项，Cowan: 4±1项）不是简单的"槽位"限制，而是由注意力和干扰共同决定。

**资源分配模型**：工作记忆容量由有限的注意资源决定，每个项目分配的资源与项目数成反比：

$$\text{精度}(n) = \frac{R}{n}$$

其中 $R$ 是总注意资源，$n$ 是项目数。项目越多，每个项目的表征精度越低。

**连续资源模型**：

$$p(\hat{x} | x, n) = \frac{1}{2\pi I_0(\kappa_n)} \exp(\kappa_n \cos(\hat{x} - x))$$

其中 $\kappa_n = \kappa_0 / n$ 是von Mises分布的浓度参数（精度），$I_0$ 是修正Bessel函数。这描述了工作记忆中角度信息的概率分布——项目数越多，记忆精度越低。

**干扰理论**：工作记忆的遗忘不是时间衰减，而是项目间的干扰：

$$\text{回忆概率} \propto \exp(-\alpha \cdot \text{相似度}_{\text{probe, distractor}})$$

与探测项越相似的干扰项，越容易产生回忆错误。

**在Agent系统中的意义**：工作记忆的容量限制意味着Agent不能同时处理所有信息——需要注意力机制选择最相关的信息进入工作记忆。

### 17.3 记忆巩固与重放机制

**系统巩固理论**：记忆从海马体（快速、临时）逐渐转移到新皮层（慢速、永久）的过程。

**巩固的数学模型**：

设海马体表征为 $\mathbf{h}_t$，新皮层表征为 $\mathbf{c}_t$。巩固过程通过"教学"信号实现：

$$\mathbf{c}_{t+1} = \mathbf{c}_t + \eta_c (\mathbf{h}_t - \mathbf{c}_t)$$

新皮层逐步逼近海马体的表征，但速度更慢（$\eta_c \ll \eta_h$）。

**经验重放**：在"离线"状态（如Agent空闲时），从长期记忆中采样旧经验进行重放：

$$\mathcal{L}_{\text{replay}} = \mathbb{E}_{(x,y) \sim \mathcal{M}} [\mathcal{L}(f_\theta(x), y)]$$

其中 $\mathcal{M}$ 是记忆缓冲区。

**优先级重放**：不是均匀采样，而是根据"惊讶度"优先重放：

$$P(\text{sample } (x_i, y_i)) \propto |\delta_i| + \epsilon$$

其中 $\delta_i$ 是TD误差或预测误差，$\epsilon$ 是小常数保证探索。预测误差大的经验被更频繁地重放——这与主动推断中"最大化认知价值"一致。

**互补生成回放**：不是存储原始数据，而是存储生成模型的参数，通过生成模型"重新生成"旧经验：

$$\hat{x}_i \sim p_\theta(x | z_i), \quad z_i \sim p(z)$$

这种方式存储效率更高（只需存储隐变量 $z_i$），且生成过程自然地融入了模型的当前知识。

### 17.4 Agent双模式记忆架构映射

基于以上理论，设计Agent的双模式记忆架构：

**模式1：快速记忆（海马体类比）**

| 属性 | 实现 |
|------|------|
| 存储 | 环形缓冲区（容量 $N_{\text{fast}}$） |
| 编码 | 原始观测 + 预测误差 |
| 检索 | 最近优先 + 预测误差优先 |
| 更新 | 每步更新，高可塑性 |
| 功能 | 即时情境感知、短期预测 |

**模式2：慢速记忆（新皮层类比）**

| 属性 | 实现 |
|------|------|
| 存储 | 生成模型参数 + 关键隐变量 |
| 编码 | 压缩表征（VAE/GM隐变量） |
| 检索 | 基于相似度的联想检索 |
| 更新 | 经验回放驱动，低可塑性 |
| 功能 | 长期知识、抽象概念、预测先验 |

**双模式交互**：

1. **编码**：新观测先进入快速记忆，同时提取特征存入慢速记忆
2. **检索**：快速记忆提供即时上下文，慢速记忆提供长期先验
3. **巩固**：空闲时，快速记忆中的经验通过回放转移至慢速记忆
4. **遗忘**：快速记忆采用FIFO策略，慢速记忆通过生成模型压缩旧经验

**记忆容量的信息论分析**：

快速记忆容量 $N_{\text{fast}}$ 的选择基于信息论：存储最近 $N_{\text{fast}}$ 个观测应足以覆盖当前环境的典型变化时间尺度：

$$N_{\text{fast}} \approx \frac{\tau_{\text{environment}}}{\Delta t}$$

其中 $\tau_{\text{environment}}$ 是环境变化的特征时间尺度，$\Delta t$ 是采样间隔。

### 17.5 开源技术选型

| 技术需求 | 开源方案 | 说明 |
|---------|---------|------|
| 经验回放 | Stable-Baselines3 (ReplayBuffer) | 优先级经验回放 |
| 生成式回放 | VAE (PyTorch) | 变分自编码器用于生成式回放 |
| 持续学习 | Avalanche | 持续学习框架，含多种记忆策略 |
| 向量检索 | FAISS / ChromaDB | 高效相似度检索 |
| 记忆管理 | 自定义 | 双模式记忆管理器 |

### 17.6 实现路径

1. **快速记忆模块**：实现环形缓冲区，支持FIFO和预测误差优先的替换策略
2. **慢速记忆模块**：基于生成模型的压缩存储，支持联想检索
3. **优先级回放**：实现基于预测误差的优先级经验回放
4. **巩固机制**：实现快速记忆到慢速记忆的定期巩固
5. **双模式集成**：将双模式记忆集成到Agent的感知-行动循环中
6. **验证**：在需要长期依赖的任务（如T-Maze、记忆游戏）中验证记忆系统的有效性

---

## 阶段18：注意力机制与资源分配

### 18.1 注意力的计算模型

注意力是认知系统的核心资源分配机制——它决定了有限的计算资源如何在众多信息源之间分配。

**注意力的计算定义**：给定输入集合 $\{x_1, x_2, \ldots, x_N\}$，注意力机制输出加权聚合：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

其中 $Q$ 是查询矩阵，$K$ 是键矩阵，$V$ 是值矩阵，$d_k$ 是键的维度。

**缩放点积注意力**的直觉：查询 $Q$ 与键 $K$ 的点积度量相似度，softmax归一化为概率分布（注意力权重），加权求和值 $V$ 得到输出。

**自注意力**：$Q = XW_Q, K = XW_K, V = XW_V$，输入自身决定注意力分配。

**多头注意力**：并行多个注意力头，捕获不同类型的依赖关系：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$$

$$\text{head}_i = \text{Attention}(QW_Q^i, KW_K^i, VW_V^i)$$

### 18.2 自顶向下与自底向上注意力

**自底向上注意力（刺激驱动）**：

由输入的物理显著性驱动，不依赖任务需求。显著性图模型：

$$S(x, y) = \sum_{f} w_f \cdot \text{Feature}_f(x, y)$$

其中 $\text{Feature}_f$ 是第 $f$ 个特征通道（颜色、方向、强度等），$w_f$ 是特征权重。显著性最高的位置吸引注意力。

**自顶向下注意力（目标驱动）**：

由当前任务和目标驱动，主动选择与任务相关的信息。在计算模型中：

$$\text{Top-down attention} = \text{softmax}\left(\frac{Q_{\text{task}} \cdot K^T}{\sqrt{d_k}}\right)$$

其中 $Q_{\text{task}}$ 是任务相关的查询向量，它"询问"输入中哪些信息与当前任务相关。

**两种注意力的交互**：

$$\text{Total attention} = \alpha \cdot \text{Bottom-up} + (1-\alpha) \cdot \text{Top-down}$$

其中 $\alpha \in [0,1]$ 是平衡参数。在Agent系统中：
- 环境异常时（高预测误差），$\alpha$ 增大——更多关注外部刺激
- 执行任务时，$\alpha$ 减小——更多关注任务目标

**预测编码中的注意力**：

在预测编码框架中，注意力等价于**精度加权**——预测误差的精度（逆方差）决定了其对上层的影响权重：

$$\text{Precision-weighted error} = \Pi^{l-1} \cdot e^{l-1} = \Sigma_{l-1}^{-1} \cdot e^{l-1}$$

高精度（低不确定性）的预测误差获得更多注意力，低精度（高不确定性）的预测误差被抑制。这实现了"注意确定性高的信号"的原则——与自顶向下注意力的"注意与任务相关的信号"互补。

### 18.3 基于显著性的资源分配

Agent的计算资源（算力、内存、时间）是有限的，需要根据信息的显著性进行分配。

**显著性驱动的资源分配**：

$$\text{Resource}(x_i) = \frac{\text{Salience}(x_i)}{\sum_j \text{Salience}(x_j)} \cdot R_{\text{total}}$$

显著性度量：

1. **预测误差显著性**：$\text{Salience}_1(x_i) = \|e_i\|^2$——预测误差大的输入更显著
2. **信息增益显著性**：$\text{Salience}_2(x_i) = I(\Psi; x_i)$——信息增益大的输入更显著
3. **任务相关性显著性**：$\text{Salience}_3(x_i) = \text{Relevance}(x_i, \text{task})$——与当前任务相关的输入更显著
4. **新奇性显著性**：$\text{Salience}_4(x_i) = -\ln p(x_i)$——低概率（新奇）的输入更显著

**综合显著性**：

$$\text{Salience}(x_i) = \sum_{k=1}^{4} w_k \cdot \text{Salience}_k(x_i)$$

权重 $w_k$ 由当前上下文动态调整：
- 安全模式：$w_1$（预测误差）和 $w_4$（新奇性）权重增大
- 任务模式：$w_3$（任务相关性）权重增大
- 探索模式：$w_2$（信息增益）权重增大

### 18.4 Agent算力资源的注意力分配

在Agent系统中，注意力机制不仅用于信息选择，还用于计算资源的动态分配。

**计算预算分配**：

设Agent总计算预算为 $B$，需要分配给 $M$ 个模块（感知、预测、规划、学习等）：

$$B = \sum_{m=1}^{M} b_m, \quad b_m \geq b_m^{\min}$$

最优分配基于各模块的边际效用：

$$b_m^* = \arg\max_{b_m} \sum_m U_m(b_m) \quad \text{s.t.} \sum_m b_m = B$$

其中 $U_m(b_m)$ 是模块 $m$ 获得计算预算 $b_m$ 时的效用（如预测精度提升、策略质量改善等）。

**动态预算调整**：根据环境状态动态调整预算分配：

$$b_m(t+1) = b_m(t) + \eta \left[\frac{\partial U_m}{\partial b_m}\bigg|_{b_m(t)} - \bar{\lambda}\right]$$

其中 $\bar{\lambda}$ 是预算的影子价格（拉格朗日乘子），确保总预算约束满足。边际效用高于平均的模块获得更多预算，低于平均的模块减少预算。

**自适应计算**：不同输入使用不同计算量——简单输入快速处理，复杂输入深度处理：

$$\text{Compute}(x) = \begin{cases} \text{Shallow}(x) & \text{if } \text{Salience}(x) < \theta_1 \\ \text{Medium}(x) & \text{if } \theta_1 \leq \text{Salience}(x) < \theta_2 \\ \text{Deep}(x) & \text{if } \text{Salience}(x) \geq \theta_2 \end{cases}$$

这实现了"早退机制"（early exit）——简单输入不需要通过所有计算层。

### 18.5 开源技术选型

| 技术需求 | 开源方案 | 说明 |
|---------|---------|------|
| Transformer注意力 | PyTorch (nn.MultiheadAttention) | 标准注意力实现 |
| 高效注意力 | Flash Attention / xFormers | 内存高效的注意力计算 |
| 稀疏注意力 | Longformer / BigBird | 长序列的稀疏注意力 |
| 自适应计算 | Universal Transformer | 自适应计算时间的Transformer |
| 资源调度 | Ray | 分布式计算资源调度 |

### 18.6 实现路径

1. **精度加权注意力**：在预测编码框架中实现基于精度的误差加权
2. **双路注意力**：实现自顶向下和自底向上注意力的融合
3. **显著性计算**：实现综合显著性度量，支持动态权重调整
4. **计算预算分配**：实现基于边际效用的动态预算分配
5. **自适应计算**：实现基于显著性的早退机制
6. **验证**：在多任务环境中验证注意力机制对计算效率的提升

---

## 阶段19：层级预测处理——从微观到宏观

### 19.1 层级生成模型

层级生成模型是大脑信息处理的核心架构——从低级感觉特征到高级抽象概念的层级组织。

**层级生成模型的数学描述**：

设层级有 $L$ 层，第 $l$ 层的隐变量为 $\mathbf{z}^l$，生成过程为：

$$\mathbf{z}^l \sim p(\mathbf{z}^l | \mathbf{z}^{l+1}), \quad l = L-1, L-2, \ldots, 0$$

$$\mathbf{x} = \mathbf{z}^0 \sim p(\mathbf{x} | \mathbf{z}^1)$$

其中 $\mathbf{z}^L$ 是最高层抽象表征，$\mathbf{x}$ 是观测数据。每一层的分布由上层隐变量条件化：

$$p(\mathbf{z}^l | \mathbf{z}^{l+1}) = \mathcal{N}(f^l(\mathbf{z}^{l+1}; \theta^l), \Sigma^l)$$

联合分布的因子分解：

$$p(\mathbf{x}, \mathbf{z}^1, \ldots, \mathbf{z}^L) = p(\mathbf{x} | \mathbf{z}^1) \prod_{l=1}^{L-1} p(\mathbf{z}^l | \mathbf{z}^{l+1}) p(\mathbf{z}^L)$$

**层级VAE（HVAE）**：层级生成模型的变分自编码器实现：

$$q(\mathbf{z}^{1:L} | \mathbf{x}) = \prod_{l=1}^{L} q(\mathbf{z}^l | \mathbf{z}^{l-1}, \mathbf{x})$$

ELBO：

$$\ln p(\mathbf{x}) \geq \mathbb{E}_{q}\left[\ln p(\mathbf{x} | \mathbf{z}^1)\right] - \sum_{l=1}^{L} D_{\text{KL}}\left[q(\mathbf{z}^l | \mathbf{z}^{l-1}) \| p(\mathbf{z}^l | \mathbf{z}^{l+1})\right]$$

### 19.2 自下而上误差传播与自上而下预测传播

层级预测处理的核心是双向信息流：

**自下而上（Bottom-up）**：预测误差从低层传向高层

$$\mathbf{e}^l = \mathbf{z}^l - f^{l+1}(\mathbf{z}^{l+1}; \theta^{l+1})$$

低层无法解释的残差信息向上传递，请求更高层的解释。

**自上而下（Top-down）**：预测从高层传向低层

$$\hat{\mathbf{z}}^l = f^{l+1}(\mathbf{z}^{l+1}; \theta^{l+1})$$

高层提供对低层活动的先验约束（预测），指导低层的表征。

**消息传递方程**：

在变分推断框架下，层级消息传递可以写为：

$$q(\mathbf{z}^l) \propto \exp\left[\mathbb{E}_{q(\mathbf{z}^{l-1})}[\ln p(\mathbf{z}^{l-1} | \mathbf{z}^l)] + \mathbb{E}_{q(\mathbf{z}^{l+1})}[\ln p(\mathbf{z}^l | \mathbf{z}^{l+1})]\right]$$

第一项来自自下而上的误差信号（似然），第二项来自自上而下的预测信号（先验）。

**精度加权**：每一层的消息由精度（逆方差）加权：

$$\text{Bottom-up message}^l = \Pi^{l-1} \cdot \mathbf{e}^{l-1}$$

$$\text{Top-down message}^l = \Pi^l \cdot \hat{\mathbf{z}}^l$$

高精度层的消息权重更大——这实现了注意力的精度加权机制。

### 19.3 抽象层级的时间尺度分离

层级预测处理的一个关键性质：**不同抽象层级对应不同的时间尺度**。

**时间尺度分离原理**：高层表征变化缓慢（捕获长期趋势和抽象概念），低层表征变化快速（捕获即时感觉和细节特征）。

数学描述：设第 $l$ 层的时间常数为 $\tau_l$，则：

$$\tau_0 \ll \tau_1 \ll \tau_2 \ll \cdots \ll \tau_L$$

在连续时间模型中：

$$\tau_l \dot{\mathbf{z}}^l = -\frac{\partial F}{\partial \mathbf{z}^l}$$

高层的时间常数 $\tau_l$ 更大，意味着其更新更慢——这对应了"慢特征"原则。

**慢特征分析（Slow Feature Analysis, SFA）**：

SFA从时序信号中提取缓慢变化的特征：

$$\min_{f} \mathbb{E}_t\left[\left(\frac{d}{dt}f(\mathbf{x}(t))\right)^2\right] \quad \text{s.t.} \quad \mathbb{E}[f] = 0, \quad \mathbb{E}[f^2] = 1, \quad f \text{ 依次正交}$$

SFA的输出按变化速度排序——最慢的特征对应最高层的抽象表征。

**时间尺度分离的计算优势**：

1. **稳定性**：高层慢变化提供稳定的先验，防止低层噪声传播
2. **泛化性**：慢变化的抽象特征跨情境稳定，支持迁移学习
3. **效率**：高层不需要每步更新，节省计算资源
4. **层次规划**：高层制定长期策略，低层执行即时行动

### 19.4 Agent三层预测架构的理论基础

基于层级预测处理理论，设计Agent的三层预测架构：

**第一层：感知预测层（快速、低级）**

- 时间尺度：$\tau_0 \sim 10\text{ms} - 1\text{s}$
- 功能：预测即时感觉输入
- 模型：卡尔曼滤波 / 短期ARIMA
- 表征：感觉特征（视觉、听觉等）
- 更新频率：每步

**第二层：情境预测层（中速、中级）**

- 时间尺度：$\tau_1 \sim 1\text{s} - 10\text{min}$
- 功能：预测当前情境的演变
- 模型：RNN / 状态空间模型
- 表征：情境状态（位置、任务阶段、社交情境等）
- 更新频率：每 $N_1$ 步

**第三层：语义预测层（慢速、高级）**

- 时间尺度：$\tau_2 \sim 10\text{min} - 1\text{day}$
- 功能：预测长期环境趋势和抽象概念
- 模型：层级贝叶斯 / 慢特征网络
- 表征：抽象概念（目标、策略、世界模型等）
- 更新频率：每 $N_2$ 步

**跨层交互**：

| 方向 | 信息内容 | 机制 |
|------|---------|------|
| 自下而上 | 预测误差 $\mathbf{e}^l$ | 精度加权的误差传播 |
| 自下而上 | 异常信号 | 超阈值误差触发上层更新 |
| 自上而下 | 预测 $\hat{\mathbf{z}}^l$ | 先验约束下层表征 |
| 自上而下 | 注意力权重 | 精度调制控制下层增益 |
| 横向 | 上下文信息 | 同层内上下文调制 |

**层级自由能**：

$$F_{\text{total}} = \sum_{l=0}^{L} F_l = \sum_{l=0}^{L} \left[-\mathbb{E}_{q}[\ln p(\mathbf{z}^{l-1} | \mathbf{z}^l)] + D_{\text{KL}}[q(\mathbf{z}^l) \| p(\mathbf{z}^{l+1})]\right]$$

总自由能是各层自由能之和。每层独立最小化自身自由能，同时通过消息传递与其他层协调。

### 19.5 开源技术选型

| 技术需求 | 开源方案 | 说明 |
|---------|---------|------|
| 层级VAE | PyTorch (自定义) | 层级变分自编码器 |
| 慢特征分析 | SFA (Python) | 慢特征提取 |
| 层级RNN | Hierarchical RNN (PyTorch) | 多时间尺度RNN |
| 消息传递 | PyTorch Geometric | 图神经网络消息传递框架 |
| 层级规划 | Option-Critic / FeUdal Networks | 层级强化学习 |

### 19.6 实现路径

1. **层级生成模型**：实现三层生成模型，支持自下而上和自上而下的消息传递
2. **精度加权**：实现基于精度的误差加权和注意力调制
3. **时间尺度分离**：实现不同层的差异化更新频率
4. **慢特征提取**：实现SFA模块，自动提取慢变化特征作为高层表征
5. **层级规划**：实现基于层级模型的层级规划（高层策略+低层执行）
6. **验证**：在需要多时间尺度预测的任务中验证层级架构的优势

---

## 阶段20：系统统一数学框架

### 20.1 将所有理论统一为单一数学框架

前19个阶段分别从不同角度建立了Agent系统的理论基础。现在，我们将这些理论统一为一个自洽的数学框架。

**统一框架的核心观察**：所有理论都可以从自由能最小化原理推导出来——自由能原理是"万物理论"。

| 阶段 | 理论 | 在统一框架中的角色 |
|------|------|-----------------|
| 1 | 第一性原理 | 物理动机：生存=自由能最小化 |
| 2 | 预测编码 | 自由能最小化的层级实现 |
| 3 | 自由能原理 | 统一框架的核心公理 |
| 4 | 主动推断 | 自由能最小化的行动扩展 |
| 5 | 信息论 | 自由能的信息论分解 |
| 6 | 动力系统 | 自由能景观的几何性质 |
| 7 | 控制论 | 自由能最小化的控制实现 |
| 8 | 进化计算 | 自由能景观上的全局搜索 |
| 9 | 强化学习 | 自由能最小化的策略优化视角 |
| 10 | 贝叶斯推断 | 自由能最小化的推断算法 |
| 11 | 多臂老虎机 | 自由能最小化的探索-利用权衡 |
| 12 | 时序预测 | 自由能最小化的时序建模 |
| 13 | 异常检测 | 自由能突变的检测与响应 |
| 14 | 因果推断 | 生成模型的因果语义扩展 |
| 15 | 元学习 | 自由能最小化过程本身的优化 |
| 16 | 神经可塑性 | 自由能最小化的仿生更新规则 |
| 17 | 记忆系统 | 自由能最小化的时间扩展 |
| 18 | 注意力 | 自由能最小化的资源分配 |
| 19 | 层级预测 | 自由能最小化的层级架构 |

### 20.2 全局目标函数的推导

**Agent的全局目标函数**：

$$\mathcal{J} = \underbrace{\int_0^T F(\theta_t, o_t) dt}_{\text{累积自由能}} + \underbrace{\int_0^T G(\pi_t) dt}_{\text{累积预期自由能}} + \underbrace{\lambda_1 R_{\text{complexity}}(\theta)}_{\text{模型复杂度}} + \underbrace{\lambda_2 R_{\text{safety}}(o)}_{\text{安全约束}}$$

各项的含义：

1. **累积自由能**：Agent在时间 $[0,T]$ 内的感知误差总和——越小越好
2. **累积预期自由能**：Agent策略选择的长期代价——越小越好
3. **模型复杂度正则化**：防止模型过拟合——奥卡姆剃刀
4. **安全约束**：确保Agent不进入危险状态——生存底线

**自由能的展开**：

$$F(\theta, o) = \underbrace{-\mathbb{E}_{q_\theta}[\ln p(o | \psi)]}_{\text{准确性}} + \underbrace{D_{\text{KL}}[q_\theta(\psi) \| p(\psi)]}_{\text{复杂性}}$$

**预期自由能的展开**：

$$G(\pi) = \underbrace{D_{\text{KL}}[q(o|\pi) \| C]}_{\text{外禀价值}} - \underbrace{I(\Psi; O|\pi)}_{\text{认知价值}}$$

**安全约束**：

$$R_{\text{safety}}(o) = \max(0, -\ln p(o | \theta_{\text{safe}}) - \ln \eta_{\text{safe}})$$

当观测的安全对数似然低于阈值 $\ln \eta_{\text{safe}}$ 时，安全约束被激活，产生惩罚项。

### 20.3 系统动力学的微分方程描述

Agent系统的完整动力学由以下耦合微分方程组描述：

**感知动力学**（状态推断）：

$$\tau_s \dot{\mathbf{s}} = -\nabla_{\mathbf{s}} F(\mathbf{s}, \theta, o)$$

**行动动力学**（主动推断）：

$$\tau_a \dot{\mathbf{a}} = -\nabla_{\mathbf{a}} G(\mathbf{a}, \mathbf{s}, \theta)$$

**学习动力学**（参数更新）：

$$\tau_\theta \dot{\theta} = -\nabla_{\theta} F(\mathbf{s}, \theta, o) + \sigma_\theta \xi_\theta(t)$$

**进化动力学**（结构搜索）：

$$\tau_\alpha \dot{\alpha} = -\nabla_{\alpha} \mathbb{E}[F(\theta^*(\alpha), \alpha)] + \sigma_\alpha \xi_\alpha(t)$$

**注意力动力学**（资源分配）：

$$\tau_w \dot{w}_m = \frac{\partial U_m}{\partial b_m} - \bar{\lambda}$$

**记忆动力学**（巩固与遗忘）：

$$\tau_c \dot{\mathbf{c}} = \eta_c (\mathbf{h} - \mathbf{c})$$

其中 $\tau_s \ll \tau_a \ll \tau_\theta \ll \tau_\alpha$ 体现了多时间尺度分离，$\xi_\theta(t)$ 和 $\xi_\alpha(t)$ 是随机噪声项（对应进化的变异）。

**层级耦合**：对于 $L$ 层层级模型，感知动力学扩展为：

$$\tau_l \dot{\mathbf{z}}^l = -\frac{\partial F}{\partial \mathbf{z}^l} = \Pi^{l-1} \mathbf{e}^{l-1} - \Pi^l \mathbf{e}^l, \quad l = 0, 1, \ldots, L$$

其中 $\Pi^l = (\Sigma^l)^{-1}$ 是精度矩阵，$\mathbf{e}^l = \mathbf{z}^l - f^{l+1}(\mathbf{z}^{l+1})$ 是预测误差。

### 20.4 稳定性、收敛性、有界性证明

**定理1（自由能单调递减）**：在感知和行动动力学下，自由能单调递减：

$$\frac{dF}{dt} = \nabla_s F \cdot \dot{\mathbf{s}} + \nabla_a F \cdot \dot{\mathbf{a}} = -\frac{1}{\tau_s}\|\nabla_s F\|^2 - \frac{1}{\tau_a}\|\nabla_a F\|^2 \leq 0$$

证明：直接由梯度下降动力学得出。$\square$

**推论**：自由能是Agent感知-行动动力学的Lyapunov函数，系统收敛到自由能的局部极小值（吸引子）。

**定理2（参数收敛性）**：在强凸自由能景观下（$F$ 关于 $\theta$ 的Hessian矩阵 $H \succeq \mu I$，$\mu > 0$），学习动力学以速率 $O(e^{-\mu t/\tau_\theta})$ 收敛到全局极小值。

证明：强凸条件下，梯度下降的收敛速率由条件数决定：

$$\|\theta_t - \theta^*\| \leq e^{-\mu t / \tau_\theta} \|\theta_0 - \theta^*\|$$

$\square$

**定理3（有界性）**：在安全约束下，Agent的观测始终保持在安全集合内：

$$\mathcal{S}_{\text{safe}} = \{o : -\ln p(o | \theta_{\text{safe}}) \leq \ln \eta_{\text{safe}}\}$$

证明：安全约束 $R_{\text{safety}}$ 在观测接近安全边界时产生无限大的惩罚力，将Agent推回安全区域。形式化地：

$$\dot{\mathbf{a}} = -\nabla_a G - \lambda_2 \nabla_a R_{\text{safety}}$$

当 $o \to \partial \mathcal{S}_{\text{safe}}$ 时，$\nabla_a R_{\text{safety}} \to \infty$，行动被强制拉回。$\square$

**定理4（进化收敛性）**：在(1+1)-ES下，自由能的期望值单调递减：

$$\mathbb{E}[F(\theta_{t+1})] \leq F(\theta_t)$$

证明：(1+1)-ES只接受使自由能降低的变异，因此 $\mathbb{E}[F(\theta_{t+1})] \leq F(\theta_t)$。$\square$

**定理5（层级稳定性）**：在时间尺度分离条件下（$\tau_0 \ll \tau_1 \ll \cdots \ll \tau_L$），层级动力学的各层可以独立分析稳定性——低层快速收敛到由高层确定的吸引子，高层缓慢调整吸引子位置。

证明：基于奇异摄动理论（Tikhonov定理），当 $\tau_l / \tau_{l+1} \to 0$ 时，低层动力学可以"绝热消除"——将低层视为已收敛到高层确定的准稳态。$\square$

### 20.5 统一框架的算法实现

将统一数学框架转化为可执行的算法：

```
Algorithm: Self-Driven Perpetual Evolution Agent

初始化:
    生成模型参数 θ, 结构 α
    三层隐变量 z^0, z^1, z^2
    快速记忆 M_fast, 慢速记忆 M_slow
    注意力权重 w_1, ..., w_M

Main Loop (每步):
    1. 感知 (时间尺度 τ_s):
       a. 接收观测 o_t
       b. 层级预测编码 (K步迭代):
          For l = 0, 1, ..., L:
              e^l = z^l - f^{l+1}(z^{l+1})     // 预测误差
              z^l ← z^l - κ_s · (∂F/∂z^l)       // 精度加权更新
       c. 异常检测: if ||e^0|| > θ_anomaly → 触发安全模式

    2. 行动 (时间尺度 τ_a):
       a. 计算预期自由能 G(π) for each π
       b. 策略选择: q(π) = softmax(-γ·G(π))
       c. 采样行动: a_t ~ q(π)
       d. 执行行动, 获取新观测

    3. 学习 (时间尺度 τ_θ, 每N_θ步):
       a. 梯度更新: θ ← θ - κ_θ · ∇_θ F
       b. 可塑性门控: Δθ *= g(θ) (仿生巩固)
       c. 在线贝叶斯更新 (共轭先验)

    4. 进化 (时间尺度 τ_α, 每N_α步):
       a. 变异: θ' = θ + σ·ξ
       b. 选择: if F(θ') < F(θ) → θ ← θ'
       c. 步长自适应: σ ← adapt(σ, success_rate)

    5. 记忆 (持续):
       a. 快速记忆: M_fast.push(o_t, e_t)
       b. 巩固: if idle → M_slow.consolidate(M_fast)
       c. 回放: if idle → replay from M_slow

    6. 注意力 (持续):
       a. 计算显著性: Salience(x_i) = Σ w_k · Salience_k(x_i)
       b. 分配计算预算: b_m ∝ ∂U_m/∂b_m
       c. 精度更新: Π^l ← adapt_precision(e^l)
```

### 20.6 开源技术选型（统一框架）

| 技术需求 | 开源方案 | 说明 |
|---------|---------|------|
| 核心框架 | JAX | 统一自动微分、JIT编译、GPU加速 |
| 概率编程 | NumPyro | 变分推断、MCMC、SMC |
| 神经网络 | Haiku / Equinox | JAX原生神经网络库 |
| 进化策略 | evosax | JAX原生ES库 |
| 强化学习 | PureJaxRL | JAX原生RL库 |
| 分布式 | Ray / JAX pmap | 分布式计算 |
| 仿真环境 | Gymnasium / JaxMARL | 标准化环境接口 |

### 20.7 实现路径

1. **核心框架搭建**：基于JAX构建统一计算框架，支持自动微分和GPU加速
2. **自由能计算引擎**：实现通用自由能计算模块，支持任意生成模型
3. **层级预测编码**：实现三层预测编码网络，支持双向消息传递
4. **主动推断循环**：实现感知-行动-学习的闭环，支持多时间尺度
5. **进化模块集成**：将(1+1)-ES/CMA-ES集成到主循环
6. **记忆与注意力**：集成双模式记忆和显著性驱动的注意力
7. **安全模块**：实现基于异常检测的安全约束
8. **端到端验证**：在复杂环境中验证完整系统的自驱性、永续性和进化性

### 20.8 理论总结与展望

本部分从第一性原理出发，逐步构建了自驱永续进化预测修正Agent系统的完整理论框架。核心逻辑链如下：

**物理层**：热力学第二定律 → 生命必须预测 → 预测误差最小化 = 生存

**数学层**：自由能原理 → 变分自由能最小化 = 感知 + 学习 + 行动

**算法层**：预测编码 + 主动推断 + 进化计算 = 自驱永续进化Agent

**架构层**：层级预测 + 双模式记忆 + 注意力分配 = 高效可扩展系统

**保证层**：Lyapunov稳定性 + 收敛性 + 有界性 = 可靠安全系统

这一理论框架的关键创新在于：

1. **自驱性**：无需外部奖励设计，行为完全由自由能最小化驱动
2. **永续性**：通过进化计算持续优化，不会收敛到次优解后停滞
3. **进化性**：学习规则本身可以被优化，实现"学会更好地进化"
4. **安全性**：自由能突变检测 + 安全约束，确保Agent不会进入危险状态
5. **统一性**：所有模块从同一数学原理（自由能最小化）推导，保证理论自洽

后续部分将基于此理论基础，详细设计系统架构、实现细节和工程方案。

---

> **第一部分完**

> 本部分建立了自驱永续进化预测修正Agent系统的完整理论基础。从热力学第二定律到自由能原理，从预测编码到主动推断，从贝叶斯推断到进化计算，所有理论最终统一于自由能最小化这一单一数学框架。这一框架不仅提供了理论自洽性，更重要的是为工程实现提供了明确的数学目标和算法路径。
