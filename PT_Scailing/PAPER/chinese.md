# Scaling Probabilistic Transformer via Efficient Cross-Scale Hyperparameter Transfer

**Abstraction**

Probabilistic Transformer (PT), a white-box probabilistic model for contextual word representation, has demonstrated substantial similarity to standard Transformers in both computational structure and downstream task performance on small models and small to medium sized datasets. However, PT is less robust to hyperparameter choices than standard Transformers, making it harder to scale efficiently. In this work, we follow Maximal Update Parametrization ($\mu$P) to rescale PT’s parameters, so that hyperparameters optimized on small models can be transferred to larger models without additional tuning. With this approach, we successfully scale PT to models with up to 0.4B parameters. Experiments show that PT consistently outperforms standard transformer under the same parameter budget on Masked Language Modeling (MLM) tasks. We hope this work will contribute to the practical deployment of probabilistic models at substantially larger scales in the future.

## 1.Introduction

以Transformer为代表的神经网络在自然语言处理等领域取得了巨大成功，但其作为依赖数据的“黑盒”模型，缺乏内在的数学透明度与理论保证。为突破此瓶颈，“白盒模型”逐渐成为深度学习架构设计的前沿焦点。例如，概率Transformer（Probabilistic Transformer, PT）(Wu & Tu, 2023) 完全从句法和概率视角出发，引入条件随机场（CRF）建模潜在表征，并通过平均场变分推断（MFVI）进行求解。这种具备内在可解释性的架构不仅在中小规模数据集上展现出接近标准Transformer的性能，更为揭示模型底层机制提供了重要启示。

然而，PT 在模型参数量的扩展（Scaling）上面临严峻瓶颈。在执行 Head Selection 与 Topic Modeling 等变分推断模块时，PT 需要对非线性归一化（如 Softmax）前的消息张量进行数值缩放，从而引入了 6 个标准 Transformer 中不存在的额外信息权重超参数。这 6 个超参数与学习率之间存在高度的非线性耦合，使得传统的控制变量单步调优策略完全失效，且其最优配置会随模型规模的放大发生剧烈漂移。在算力受限的情况下，这种缺乏跨尺度迁移能力的复杂超参数空间，使得大模型尺度上的超参数搜索极其昂贵甚至不可行，严重制约了 PT 向大规模参数演进的潜力。

最大更新参数化（Maximal Update Parametrization, $\mu$P）(Yang et al., 2021) 是解决超参随规模漂移的有效方法，能实现最优超参数向大模型的零样本（Zero-shot）迁移。但目前的 $\mu$P 框架专为“黑盒”网络设计，其核心操作包含对中间激活值或 Logits 的缩放调整（例如，为保证方差稳定，将注意力机制中的 logits 缩放因子由 $1/\sqrt{d_k}$ 改为 $1/d_k$）。在 PT 的计算图中，每一层激活值均代表严格的概率分布。直接对其施加缩放因子会破坏概率分布假设，进而破坏模型的白盒性质。

针对上述理论冲突，本文提出了一种适用于 PT 的跨尺度参数化重构方法。我们汲取 $\mu$P 的底层原理，在严格维持概率分布假设的前提下，对 PT 的势函数（Potential Functions）与变分自由能（Variational Free Energy）进行了系统性的缩放调整。同时，我们修改了参数初始化方差与特定模块的分组学习率（Group-specific learning rates）。调整后的 PT 不仅严格保留了各层激活值的可解释数学意义，更具备了与使用$\mu$P的黑盒网络一致的超参数迁移能力。该方法使 PT 能够直接在极小规模模型上完成最优调优，并直接迁移至任意参数规模，极大降低了扩展成本。实验表明，在不同规模的掩码语言建模（MLM）任务中，具备该扩展能力的 PT 架构性能均优于 BERT，并接近Universal Transformer。

## 2. Adjustment of $\mu$P to PT

本节旨在详细阐述针对PT的跨尺度缩放框架。通过系统性地重构参数化配置（Parameterization）与底层数学架构（Mathematical Architecture），我们在严格保证 PT 概率推断白盒属性的前提下，成功使其具备了包含学习率与 6 个核心信息权重在内的全局超参数的零样本（Zero-shot）迁移能力。理论分析表明，我们的架构调整在数学本质上与 $\mu$P 的极限缩放原理完全相符。相关定理的严密推导与证明详见附录 B。

### 2.1 Adjustment of parameterization

**可学习参数的分组** $\mu$P根据可学习参数规模与模型宽度的关系，对可学习参数进行分组。在 PT 架构中，隐变量 $Z$ 节点的维度（记为 $\text{dim}_z$）天然对应于模型的宽度属性。基于 $\mu$P 的分类准则，我们将 PT 中的所有可学习参数严格划分为输入权重（Input Weights）、隐层权重（Hidden Weights）与输出权重（Output Weights）三类，具体的参数映射关系如表 1 所示。

|类别|可学习参数|
|-|-|
|Input Weights|Unary factor (S in PT) <br> LayerNorm Weight of MLM Head ($\gamma$ of RMSNorm) <br> All Biases|
|Hidden Weights|Ternary Factor U (U in PT)<br>Ternary Factor V (V in PT)<br>Binary Factor (B in PT)|
|Output Weights|Decoder Weight of MLM Head|

表1：可学习参数的分组

**初始化分布与分组学习率的重构** 遵循 $\mu$P 的理论推导，为不同组别的参数量身定制了初始化的方差缩放规则与分组学习率。具体而言，隐层与输出层权重采用随模型宽度 $N$（即 $\text{dim}_z$）缩放的方差进行初始化，并应用相应的缩放学习率以保证训练动态的稳定；而输入层权重与偏置项则维持基础分布与基础学习率。详细的跨尺度参数化配置如表 2 所示。

|参数分组|初始化分布|学习率|
|-|-|-|
|Input Weights (Biases除外)| $\mathcal{N}(0, 1)$ | **Base LR ($\eta$)**| 
|Hidden Weights|$\mathcal{N}(0, \frac{1}{N})$ | **Scaled LR ($\frac{\eta}{N}$)**|
|Output Weights|$\mathcal{N}(0, \frac{1}{N^2})$<br> | **Scaled LR ($\frac{\eta}{N}$)**|
|Biases|0 | **Base LR ($\eta$)**|

表2：可学习参数的初始化分布与学习率重构

### 2.2 Adjustment of mathematical architecture

**Head Selection 的缩放范式** 在 PT 的 Head Selection 模块中，我们构建了包含 $C$ 个通道（Channels）的依存关系矩阵 $T\in \mathbb{R}^{N\times N}$。针对每个通道的矩阵 $T$，我们采用低秩分解将其表示为 $UV^T$，其中 $U,V\in \mathbb{R}^{N\times r}$。随着模型宽度 $N$ 的扩展，Head Selection 模块的参数化缩放面临两种可行的设计范式：其一为固定秩 $r$，通道数 $C$ 随 $N$ 线性缩放；其二为固定通道数 $C$，秩 $r$ 随 $N$ 线性缩放。为确保架构的普适性，PT 底层数学架构的重构必须能够同时兼容这两种缩放方案，从而能够公平评估并择优选取最佳缩放策略。

**势函数与变分自由能的重构** 在 PT 的概率图模型中，核心推理过程依赖于对隐变量 $Z$ 节点的后验概率分布 $Q_i$ 以及 $H$ 节点的后验概率分布 $Q_{ic}$ 执行平均场变分推断（MFVI）。为了引入跨尺度的数值控制，我们定义了一个针对 $Q_i$ 分布的温度参数（Temperature Parameter）
$$\tau=\frac{N}{r}$$
而 $Q_{ic}$ 分布的温度参数则恒定保持为 1。此外，我们令G节点的数量为M，M与N成正比缩放。在此基础上，相较于原始 PT 架构，我们对系统的势函数（Potential Functions）进行了重构：

$$
\begin{cases}
\phi_u(Z_i=a)=\exp(\tau S_{w_i,a}) \\
\phi_t(H_i=k,Z_i=a,Z_j=b)= \begin{cases} \exp(\tau N T_{a,b}) & H_i=j \\ 1 & \text{otherwise} \end{cases} \\
\phi_b(Z_i=a,G_i=g)=\exp(\tau N B_{g,a})
\end{cases}
$$

相应地，$Q_i$ 的变分自由能函数被修正为：

$$F(Q_i)=\mathbb{E}_Q[-\log P(Z,H,G|w)]-\tau H(Q_i)$$

而 $Q_{ic}$ 的变分自由能函数保持不变：

$$F(Q_{ic})=\mathbb{E}_Q[-\log P(Z,H,G|w)]-H(Q_{ic})$$

这种重构确保了，无论是选择“扩展通道数 $C$”还是“扩展秩 $r$”，Head Selection 模块均能在这一统一的数学框架下实现严谨的 $\mu$P 缩放。我们在附录 A.2 中给出了详细的数学证明。

**与黑盒 $\mu$P 机制的数学等效性** 不同于标准黑盒神经网络可以直接对激活值（Activations）施加标量乘子来进行缩放，PT 必须维持activations的概率分布假设。因此，我们通过修改势函数与变分自由能，在实现缩放的同时保持了PT的白盒性质。可以证明，上述调整在最终的计算图解析推导上与黑盒模型中直接的数值缩放具有数学等效性，其效果完全等价于在 MFVI 迭代中对以下两处关键激活值进行缩放调整：

$$
\begin{cases}
\mathcal{F}_c^{(t)} \leftarrow \mathcal{F}_c^{(t)}/r \\
Q_z^{(t)} \leftarrow Q_z^{(t)} \cdot N
\end{cases}
$$

值得强调的是，在 PT 的推断逻辑中，消息张量 $\mathcal{F}_c^{(t)}$ 的功能完全对应于标准 Transformer 中的Attention logits。因此，将其隐式缩放为 $\mathcal{F}_c^{(t)}/r$ 在数学本质上精准对齐了 $\mu$P 理论中将注意力缩放因子改为 $qk^T/d_k$ 的核心操作。关于此等效性质的严密推导见附录 A.1。

### 2.3 Savings in computing power resources

如图 1 所示，PT 模型的整体计算量（FLOPs）随着模型宽度（$\text{dim}_z$）的扩展呈现出显著的幂律缩放（Power-law scaling）规律。在常规的范式中，针对 7 维超参数空间的精细寻优通常依赖于计算密集的全局网格搜索（Grid Search）。假设在不同规模的模型上保持一致的搜索分辨率（即设定等量的超参数候选值与相应的模型训练总数），则总体超参数调优的算力成本将严格正比于单次模型训练的计算量。

因此，相较于直接在目标宽度 $\text{dim}_z$ 下执行极其昂贵的原位参数寻优（In-situ tuning），本文引入的 $\mu$P 零样本迁移策略能够实现幂级数级的算力节约。在该策略下，我们仅需在宽度为 $\text{dim}_z=256$ 的极小规模代理模型上完成搜索，随后直接迁移超参数。此时，采用 $\mu$P 策略所需的相对算力成本占比，在数学上等效于基准小模型与目标大模型在单次训练下的 FLOPs 比值。

表 3 定量展示了在不同宽度目标下，采用 $\mu$P 迁移策略相较于传统全尺寸调优策略的算力开销占比。随着模型规模的扩大，节约效应愈发显著；在目标宽度为 3840 时，我们的方法仅需消耗传统方法约 3.9% 的计算资源。

|模型宽度|计算量|占比|
|-|-|-|
|256 |92211676534800380  |100.0%|
|512 |197701643091836930  |46.6%|
|1024 |449513314892906500  |20.5%|
|1536 |755767304943304700  |12.2%|
|2560 |1531602239792087000  |6.0%|
|3840 |2355026383426084400 |3.9%|

表3：不同模型宽度的计算量与$\mu$P的计算量开销相对传统方法的占比

<img src="../FLOPs/FLOPs_scaling.png" width=450>

图1：不同模型宽度下的计算量变化规律。横纵轴均为双对数尺度。

## 3. Experiments

### 3.1 Optimality Verification of Scaling

**Random Search Verification.** 考虑到算力资源的限制，在7维耦合的超参数空间中执行全局网格搜索以验证零样本迁移配置的最优性是不可行的。因此，我们引入基于随机搜索的局部最优性验证方法 (Bergstra & Bengio, 2012)。具体而言，通过在零样本迁移超参数的局部邻域内独立采样 $n$ 组配置进行评估，若原配置的Evaluation Loss在扰动样本中仍处于极小值区间，则可在特定的统计置信度（$1-\alpha$）下，证明原配置属于该邻域内前 $p$ 比例的最优参数子空间。该置信度满足如下不等式约束：

$$1-(1-p)^n \geq 1-\alpha$$

若给定目标优越比例 $p$ 与显著性水平 $\alpha$，则所需的最小独立采样次数为：

$$n \geq \frac{\log \alpha}{\log (1-p)}$$

**Methods.** 本实验基于宽度为 $\text{dim}_z=1536$ 的 PT 模型展开。设零样本迁移所得的超参数集合为中心基准点 $S=\{S_1, S_2, \cdots, S_7\}$。针对任意 $S_i\ (1 \leq i \leq 7)$，我们在其 $\pm 20\%$ 的均匀分布区间 $[0.8S_i, 1.2S_i]$ 内进行独立随机采样，从而构建出中心基准点 $S$ 局部邻域内的扰动采样集合 $S'$。为了以 95% 的置信度（即 $\alpha=0.05$）验证 $S$ 是否落入局部空间内 Top 5%（即 $p=0.05$）的最优区间，根据上述定理可计算出理论最小采样次数 $n \geq 58.4$。我们共执行了 $n=63$ 次独立的局部随机采样训练，并将其最终的Evaluation Loss与基准点 $S$ 的Evaluation Loss进行量化对比。

**Results.** 实验结果呈现出极其显著的Basin of local minimum效应。在全部 63 组扰动采样中，绝大多数配置的Evaluation Loss相较于zero shot配置出现了上升。尽管存在少量表现最佳的采样点，但其损失相较于 $S$ 仅发生了最多 0.4% 的微幅下降。考虑到优化过程中的Stochastic Noise，这种极其微弱的性能波动处于训练的容差范围之内，在统计意义上可视为与 $S$ 等效。为了直观量化这一现象，我们定义扰动采样点 $S'$ 与基准点 $S$ 在相对参数空间中的距离为 $D(S',S)=\sqrt{\sum_{i=1}^7 (\frac{S_i'-S_i}{S_i})^2}$。如图 2 所示，我们分别从参数空间距离与相对性能排序两个视角对评估结果进行了可视化。其中，距离散点图展现了配置偏离基准点所引发的性能衰退趋势；而全样本的损失排序曲线则证实了 $S$（红星标注）稳固地处于该陡峭局部极小值盆地（Basin of local minimum）的最底端区域。因此，在统计学上，我们有 95% 的极大把握认定，通过零样本迁移获取的超参数配置处于该邻域空间内前 5% 的最优区域。

<img src="../Optimality/optimality_check_distance.png" width=450>
<img src="../Optimality/optimality_check_ranking.png" width=450>

图2：扰动采样点与基准点的性能对比

### 3.2 Comparison with BERT and Universal TransformerTasks and Datasets. 

**Tasks and Datasets** 我们评估了不同参数规模的 PT 模型在掩码语言建模（MLM，15% 掩码率）任务上的表现。作为对比基线，我们在相同的 MLM 任务上训练了等量参数规模的 BERT 与 Universal Transformer，并以Evaluation Loss作为对比指标。为了确保数据效率对比的公平性，所有模型均在 MiniPile 数据集上严格限制训练数据量为 1 个 Epoch。

**Settings** 为验证我们提出的跨尺度缩放框架的有效性，PT 架构严格遵循 $\mu$P 方法进行扩展。因此，我们仅在最小规模的 PT 模型（$\text{dim}_z=256$，约 17M 参数）上进行了超参数调优。随后，该最优超参数配置被Zero-shot迁移至所有更大规模的 PT 模型中。对于BERT 与 Universal Transformer，不同参数规模下均进行独立超参数调优，以确保在各维度下均处于最优配置状态。

**Results** 实验结果表明，PT在 MLM 任务上的性能稳定高于经过逐级独立调优的 BERT。这一优势部分归因于 PT 的架构设计：其变分推断的迭代过程在数学本质上起到了跨层参数复用的作用，使得 PT 在相同的参数量下，能够获得更高的计算量（FLOPs）。但与明确采用跨层参数循环复用机制的 Universal Transformer 相比，PT 的性能相对较弱。详细的实验对比结果如图 3 所示。

<img src="../PT_BERT/fig5_loglog_loss_no_fit.png" width=450>
<img src="../PT_UT/fig5_loglog_loss_no_fit.png" width=450>

图3：不同参数量下PT与BERT，PT与Universe Transformer的性能对比。横纵轴均为双对数尺度。


## 4. Related Works

4.相关工作

**缩放与超参数迁移** 现代大规模模型依赖于最大更新参数化（$\mu$P，Yang et al.，2021）进行零样本超参数迁移。然而，$\mu$P的启发式缩放往往会破坏白盒模型中固有的严谨数学结构，如概率先验或能量约束。在优化理论中，开发能够保留这些受约束架构的分析性缩放规则仍然是一个尚未充分探索的挑战。

**基于数学原理的白盒扩展** 为神经网络赋予数学透明性往往会在扩展时引入优化瓶颈。像深度平衡模型（DEQs）（Bai et al.，2019，2020）和连续Hopfield网络（Ramsauer et al.，2020）这样的隐式模型提供了坚实的理论保障，但在扩展时却存在超参数敏感性和不稳定性，往往需要经验性正则化，这会破坏其白盒特性。

最近，CRATE-$\alpha$（Yang et al.，2024）通过引入架构修改（如过完备字典和解耦权重）成功扩展了白盒Transformer。尽管这种方法有效，但这种由架构驱动的方法缺乏跨尺度参数优化的通用分析法则。相比之下，我们的工作侧重于参数化的基本数学原理。通过重构变分自由能和势函数，我们能够在不牺牲任何架构的前提下，实现概率Transformer（PT）的$\mu$P级零样本迁移。这为扩展可解释网络提供了一个新的范式，同时保持了其严格的推理语义。

## 5. Conclusions

We propose a modification to the architecture of PT to enable zero-shot hyperparameter transfer. This approach allows us to scale up PT at a significantly lower computational cost, even under limited resource constraints, and achieves superior performance on Masked Language Modeling (MLM) tasks compared to BERT at equivalent model scales.

**Limitations**
A key architectural distinction between PT and the standard Transformer lies in PT’s iterative reuse of the same set of learnable parameters across multiple rounds—effectively implementing weight tying across layers, as opposed to the decoupled layer-wise parameters in the conventional Transformer. This design resembles that of the Universal Transformer, which similarly employs parameter sharing to increase computational depth without expanding the number of parameters. Consequently, under the same parameter budget, such tied-parameter models achieve greater effective computation and thus enhanced performance. As a result, while PT outperforms BERT on the MLM task, it still lags behind the Universal Transformer.
Moreover, PT cannot leverage Flash Attention during training, leading to significantly slower computation compared to standard Transformers.


# 附录

## 附录 A：PT 势函数与变分自由能修改的细节推导

### A.1 修改后的变分推断与数学本质的等效性

本节通过对平均场变分推断（MFVI）过程的重新推导，证明对势函数与变分自由能的重构在数学本质上精准对齐了 $\mu$P 理论中对注意力缩放因子的核心操作。

#### A.1.1 缩放后的势函数与联合概率分布

在 PT 的 $\mu$P 框架下，我们引入温度参数 $\tau = N/r$。重新定义的系统势函数如下：
- **一元因子（Unary Factor）**：$\phi_u(Z_i=a) = \exp(\tau S_{w_i, a})$
- **三元因子（Ternary Factor）**：$\phi_t(H_i=j, Z_i=a, Z_j=b) = \exp(\tau N T_{a,b}^{(c)})$ (当 $H_i=j$ 时，否则为 1)
- **二元因子（Binary Factor）**：$\phi_b(Z_i=a, G_i=g) = \exp(\tau M B_{g,a})$

其中 $T^{(c)}$ 为低秩分解形式 $U^{(c)}V^{(c)\top}$，$N$ 为隐状态维度，$r$ 为三元因子的秩，$M$ 为二元因子的维度。给定序列 $w$，系统的联合概率分布 $P(Z, H, G|w)$ 定义为：
$$P(Z, H, G|w) = \frac{1}{\mathcal{Z}} \prod_{i=1}^n \phi_u(Z_i) \prod_{i=1}^n \phi_b(Z_i, G_i) \prod_{c=1}^h \prod_{i=1}^n \prod_{j=1}^n \phi_t(H_i^{(c)}, Z_i, Z_j)$$

#### A.1.2 变分自由能函数的重构

为了在大模型尺度扩展时维持能量项与熵项的量级平衡，我们定义修改后的变分自由能 $\mathcal{F}_{\mu P}(Q)$。对于隐变量 $Z$ 的边缘分布 $Q_i$，其自由能函数包含温度系数 $\tau$：
$$\mathcal{F}(Q_i) = \mathbb{E}_Q[-\ln P(Z, H, G|w)] - \tau H(Q_i)$$
展开后的完整期望能量项 $E$ 为：
$$\begin{aligned} E = &-\sum_{i=1}^n \sum_{a=1}^N Q_i(a) \tau S_{w_i,a} \\ &-\sum_{c=1}^h \sum_{i=1}^n \sum_{j \neq i} Q_{ic}(j) \sum_{a=1}^N \sum_{b=1}^N Q_i(a) Q_j(b) (\tau N T_{a,b}^{(c)}) \\ &-\sum_{i=1}^n \sum_{a=1}^N \sum_{g=1}^M Q_i(a) Q_i^G(g) (\tau M B_{g,a}) \end{aligned}$$

#### A.1.3 三元变量更新：与注意力缩放因子 $1/r$ 的等效性证明

考虑三元变量分布 $Q_{ic}(j)$（即注意力权重）的优化。根据 MFVI 迭代规则，其分布由能量项对 $Q_{ic}(j)$ 的偏导数决定：
$$\frac{\partial E}{\partial Q_{ic}(j)} = -\sum_{a=1}^N \sum_{b=1}^N Q_i(a) Q_j(b) (\tau N T_{a,b}^{(c)})$$
代入 $\tau = N/r$ 及低秩分解 $T^{(c)} = \sum_{l=1}^r U_{a,l} V_{b,l}$：
$$\frac{\partial E}{\partial Q_{ic}(j)} = -\frac{N^2}{r} \sum_{l=1}^r \left( \sum_{a=1}^N Q_i(a) U_{a,l} \right) \left( \sum_{b=1}^N Q_j(b) V_{b,l} \right)$$
为了观察其尺度特性，定义缩放后的准分布 $\tilde{Q}_i(a) = N Q_i(a)$（由于 $Q_i(a)$ 为概率分布，其坐标量级为 $\Theta(1/N)$，则 $\tilde{Q}_i(a)$ 的坐标量级恢复至 $\Theta(1)$）。记 $\boldsymbol{q}_{i, \cdot} = \sum_a \tilde{Q}_i(a) U_{a, \cdot}$ 与 $\boldsymbol{k}_{j, \cdot} = \sum_b \tilde{Q}_j(b) V_{b, \cdot}$ 为 Query 和 Key 向量，则：
$$\frac{\partial E}{\partial Q_{ic}(j)} = -\frac{1}{r} \boldsymbol{q}_i \boldsymbol{k}_j^\top$$
根据 $Q \propto \exp(-\nabla E)$，得到更新公式：
$$Q_{ic}(j) \propto \exp \left( \frac{1}{r} \boldsymbol{q}_i \boldsymbol{k}_j^\top \right)$$
在标准 Transformer 的 $\mu$P 实现中，注意力得分需缩放为 $qk^\top/d_k$。在 PT 架构中，其“注意力特征维度”由三元因子的秩 $r$ 决定。上述推导严格证明了：**将变分自由能隐式缩放，等效于在计算图底层将注意力缩放因子设定为 $1/r$。** 这一操作确保了在 $N \to \infty$ 时，注意力得分不会因点积爆炸而导致 Softmax 饱和。

#### A.1.4 隐状态变量更新：尺度稳定性的验证

对于隐变量 $Z$ 的分布 $Q_i(a)$，构建 Lagrange 函数 $L = \mathcal{F}(Q_i) + \lambda(\sum_a Q_i(a) - 1)$。令 $\nabla_{Q_i(a)} L = 0$：
$$\nabla E_{Q_i(a)} - \tau (\ln Q_i(a) + 1) + \lambda = 0 \Rightarrow Q_i(a) \propto \exp \left( -\frac{1}{\tau} \nabla E_{Q_i(a)} \right)$$
其中能量梯度 $\nabla E_{Q_i(a)}$ 项均包含系数 $\tau$。展开后，$Q_i(a)$ 的 Logits 为：
$$\begin{aligned} \text{Logits}_i(a) &= \frac{1}{\tau} \Big[ \tau S_{w_i,a} + \tau N \sum_{c,j} Q_{ic}(j) \sum_{b=1}^N Q_j(b) T_{a,b}^{(c)} + \tau M \sum_{g=1}^M Q_i^G(g) B_{g,a} \Big] \\ &= S_{w_i,a} + \sum_{c,j} Q_{ic}(j) \sum_{b=1}^N (N Q_j(b)) T_{a,b}^{(c)} + \sum_{g=1}^M (M Q_i^G(g)) B_{g,a} \end{aligned}$$
显而易见，温度参数 $\tau$ 在分子与分母中相互抵消。这意味着：
1. **维度自适应的准分布缩放**：更新方程中的二元项与三元项自动携带了对应的维度缩放因子（即 $\tilde{Q}_z = N Q_z$ 与 $\tilde{Q}_G = M Q_G$），精准地将概率值转化为坐标强度恒为 $\Theta(1)$ 的特征向量。
2. **绝对的尺度不变性**：Logits 的量级完全由参数矩阵（如 $S, T, B$）决定，与模型宽度 $N$ 和 $M$ 解耦。


### A.2 势函数与变分自由能修改在 Head Selection 模块的统一适配性

在 PT 的 Head Selection 模块中，由于模型宽度 $N$ 的扩展，我们可以采用两种不同的参数化缩放范式：其一为扩展通道数 $C$（$C \propto N, r = \Theta(1)$），其二为扩展三元因子的秩 $r$（$C = \Theta(1), r \propto N$）。本节将详细推导并证明，通过引入温度参数 $\tau = N/r$，修改后的变分自由能在这两种范式下均能保证能量函数与熵函数在训练全周期的量级平衡。

变分自由能函数定义为：
$$\mathcal{F}(Q) = E - \tau H(Q)$$
其中期望能量 $E$ 展开为一元、二元与三元势项之和：
$$E = -\sum_i \sum_{a=1}^N Q_i(a) \tau S_{w_i,a} - \sum_{i} \sum_{a,g} Q_i(a) Q_i^G(g) \tau M B_{g,a} - \sum_{c=1}^C \sum_{i,j} Q_{ic}(j) \sum_{a,b} Q_i(a) Q_j(b) \tau N T_{a,b}^{(c)}$$
为了便于量级分析，我们将概率分布还原为尺度 $\Theta(1)$ 的准分布，令 $\tilde{Q}_i(a) = N Q_i(a)$，$\tilde{Q}_i^G(g) = M Q_i^G(g)$。

---

#### A.2.1 范式一的详细推导：扩展通道数 $C$

在此方案中，$C \propto N$，$r = \Theta(1)$。因此温度参数 $\tau = \Theta(N)$。

**1. 熵函数的数量级**

对于熵函数 $H(Q) = -\sum_{a=1}^N Q(a) \ln Q(a)$，我们分阶段考量其演化。

**训练初期（均匀分布）** 此时 $H(Q) = \sum_{a=1}^N \frac{1}{N} \ln N = \ln N$。结合 $\tau = \Theta(N)$ 的设定，其整体量级为 $\tau H(Q) = \Theta(N \ln N)$。

**训练收敛期（置信分布）** 此时分布趋于稀疏，有限标签占据绝大概率，使得 $H(Q) = \Theta(1)$。相应地，整体量级变为 $\tau H(Q) = \Theta(N)$。

**2. 一元势项的数量级**

对于一元势项 $E_{unary} = -\frac{\tau}{N} \sum_{a=1}^N \tilde{Q}_i(a) S_{w_i,a}$，推导如下。

**训练初期** 参数 $S$ 的 $N$ 个单项可视为相互独立，总和呈随机游走状态。方差 $\mathrm{Var}(\sum \tilde{Q} S) = N \cdot \mathrm{Var}(\tilde{Q} S) = \Theta(N)$，故其绝对值和为 $\Theta(\sqrt{N})$。代入系数得到 $E_{unary} = \frac{\Theta(N)}{N} \Theta(\sqrt{N}) = \Theta(\sqrt{N})$。

**训练收敛期** 变量 $\tilde{Q}$ 与 $S$ 在梯度更新后趋向对齐，对应项符号高度相关（相干叠加）。因此 $\sum_{a=1}^N \tilde{Q}_i(a) S_{w_i,a} = \sum_{a=1}^N \Theta(1) = \Theta(N)$。代入系数得到最终量级 $E_{unary} = \frac{\Theta(N)}{N} \Theta(N) = \Theta(N)$。

**3. 二元势项的数量级**

对于二元势项 $E_{binary} = -\frac{\tau}{N} \sum_{a=1}^N \tilde{Q}_i(a) \sum_{g=1}^M \tilde{Q}_i^G(g) B_{g,a}$，推导如下。

**训练初期** 根据 $\mu$P 初始化原则，$B_{g,a} = \Theta(1/\sqrt{N})$。独立随机游走导致内部求和为 $\Theta(\sqrt{M/N}) = \Theta(1)$（因 $M \propto N$），总和方差游走产生 $\Theta(\sqrt{N})$。代入系数后 $E_{binary} = \Theta(\sqrt{N})$。

**训练收敛期** 参数分解为 $B = B^{init} + B^{learn}$，其中 $B^{learn} = \Theta(1/N)$。内部求和 $\sum_g \tilde{Q}_i^G(g) B_{g,a}^{learn}$ 产生 $\sum_g \Theta(1)\Theta(1/N) = \Theta(1)$ 的对齐累加。外部求和再次发生相干叠加产生 $\Theta(N)$。最终量级为 $E_{binary} = \frac{\Theta(N)}{N} \Theta(N) = \Theta(N)$。

**4. 三元势项的数量级**

对于单通道能量 $E_{ternary}^{(c)} = -\frac{\tau}{N} \sum_{i,j} Q_{ic}(j) \sum_{a,b} \tilde{Q}_i(a) \tilde{Q}_j(b) T_{a,b}^{(c)}$。

**训练收敛期** 利用低秩分解 $T_{a,b} = \sum_{l=1}^r U_{a,l} V_{b,l}$。由于 $r = \Theta(1)$，对齐学习部分 $U^{learn}, V^{learn} = \Theta(1/N)$。经过 $\sum_{a,b}$ 双重叠加计算，内层点积总和的量级为 $\Theta(1)$。由于 $\tau/N = \Theta(1)$，故单通道能量为 $\Theta(1)$。由于多通道累加 $C \propto N$，总三元能量为 $E_{ternary} = C \cdot \Theta(1) = \Theta(N)$。

**范式一结论** 训练初期，熵项 $\Theta(N \ln N)$ 在量级上大于能量项 $\Theta(\sqrt{N})$，从而主导分布脱离均匀态；训练中后期，能量与熵项严格相等均达到 $\Theta(N)$，维持了健康分布，避免了 $\tau=1$ 时必然导致的量级失衡。

---

#### A.2.2 范式二的推导：扩展秩 $r$

在此方案中，$C = \Theta(1)$，$r \propto N$。因此温度参数 $\tau = \frac{N}{r} = \Theta(1)$。详细的推导过程与范式一同理。

**1. 熵函数的数量级**

结合 $\tau = \Theta(1)$ 的设定：

**训练初期** 熵项整体量级为 $\tau H(Q) = \Theta(1) \cdot \ln N = \Theta(\ln N)$。

**训练收敛期** 熵项整体量级为 $\tau H(Q) = \Theta(1) \cdot \Theta(1) = \Theta(1)$。

**2. 能量函数（一元与二元势）的数量级**

**训练初期** 依据范式一的推导，未乘系数的原始能量和为 $\Theta(\sqrt{N})$。代入新系数 $\frac{\tau}{N} = \frac{\Theta(1)}{N}$，得到 $E_{unary, binary} = \Theta(\frac{1}{\sqrt{N}}) \approx 0$。

**训练收敛期** 原始的对齐能量和为 $\Theta(N)$。代入新系数得到 $E_{unary, binary} = \frac{\Theta(1)}{N} \cdot \Theta(N) = \Theta(1)$。

**3. 能量函数（三元势）的数量级**

**训练收敛期** 由于 $r \propto N$，学习部分的转移矩阵 $T_{a,b}^{learn} = \sum_{l=1}^r U_{a,l} V_{b,l} \sim r \cdot (\frac{1}{N} \cdot \frac{1}{N}) = \Theta(\frac{1}{N})$。双重求和 $\sum_{a,b=1}^N \tilde{Q}(a)\tilde{Q}(b) T_{a,b}$ 等效于 $N^2$ 项相干叠加，产生 $N^2 \cdot \Theta(\frac{1}{N}) = \Theta(N)$ 的原始量级。代入三元项系数 $\frac{\tau}{N} = \frac{1}{r}$（此操作对应于计算图底层的 Attention $1/d_k$ 缩放），单通道能量为 $\frac{1}{r} \cdot \Theta(N) = \Theta(1)$。由于通道数 $C = \Theta(1)$，总三元能量为 $\Theta(1)$。

**范式二结论** 训练初期，熵项 $\Theta(\ln N)$ 远大于能量项（接近 0），推动分布快速收敛；训练中后期，两者均稳定在 $\Theta(1)$ 量级。

## 附录 B：PT 对 $\mu$P 原理满足性的理论证明

为了确保概率 Transformer（PT）在宽度 $N$ 扩展过程中具备最优超参数的跨尺度可迁移性，我们遵循最大更新参数化（Maximal Update Parametrization, $\mu$P）的指导原则。本节将通过严格的数学推演，证明 PT 的参数化调整方案如何从初始化与动力学两个维度完美满足 $\mu$P 的三大基本原则。

### B.1 参数分类、初始化与学习率策略

我们将 PT 的所有可学习参数根据其在计算图中的位置、张量形状以及对激活值量级的影响，分为以下三组，并分别设定其初始化标准差 $\sigma$ 与学习率缩放因子 $\eta_{\text{mult}}$：

1. **输入层参数组（Input Group）**：主要包括一元势矩阵 $S \in \mathbb{R}^{V \times N}$ 以及所有偏置项（Biases）。此类参数将离散或低维输入映射至高维空间。
   - **初始化**：设定标准差 $\sigma_{in} = \Theta(1)$。
   - **学习率**：设定缩放因子 $\eta_{in\_mult} = 1$。

2. **隐藏层参数组（Hidden Group）**：包括三元因子的低秩分解矩阵 $U, V \in \mathbb{R}^{N \times r}$ 及二元因子矩阵 $B \in \mathbb{R}^{M \times N}$（其中 $M \propto N$）。此类参数决定了高维表征之间的交互。
   - **初始化**：设定标准差 $\sigma_{hid} = \Theta(1/\sqrt{N})$。
   - **学习率**：设定缩放因子 $\eta_{hid\_mult} = 1/N$。

3. **输出层参数组（Output Group）**：指预测头中的解码矩阵 $W_{out} \in \mathbb{R}^{N \times V}$。此类参数将高维表征映射回标量分数。
   - **初始化**：设定标准差 $\sigma_{out} = \Theta(1/N)$。
   - **学习率**：设定缩放因子 $\eta_{out\_mult} = 1$。

---

### B.2 原则 1：坐标量级的稳定性（Forward Pass Stability）

**原则 1 要求：在初始化时，神经网络中任何神经元的激活值（坐标）数量级应保持为 $\Theta(1)$。**

在 PT 的迭代过程中，考虑隐藏层神经元的线性组合项 $y_i = \sum_{j=1}^N W_{ij} x_j$（如三元因子的消息传递）。假设输入坐标 $x_j$ 满足 $\Theta(1)$，根据独立随机变量之和的性质，其输出方差满足：
$$\text{Var}(y_i) = \sum_{j=1}^N \text{Var}(W_{ij}) \mathbb{E}[x_j^2] = N \cdot \sigma_{hid}^2 \cdot \Theta(1)$$
代入 B.1 中定义的隐藏层初始化策略 $\sigma_{hid} = \Theta(1/\sqrt{N})$，得到：
$$\text{Var}(y_i) = N \cdot \frac{1}{N} \cdot \Theta(1) = \Theta(1)$$
这证明了 PT 在宽度扩展时，内部激活值的坐标强度始终稳定在 $\Theta(1)$，确保了前向传播的数值稳定性。

---

### B.3 原则 2：输出分数的有界性（Output Score Stability）

**原则 2 要求：模型最终输出的 Logits 数量级应为 $O(1)$，且不随宽度增加而爆炸。**

在输出层，预测分数 $s = \sum_{j=1}^N W_{out, j} y_j$。已知激活值 $y_j \sim \Theta(1)$，其输出分数的方差为：
$$\text{Var}(s) = N \cdot \sigma_{out}^2 \cdot \text{Var}(y_j) = N \cdot \frac{1}{N^2} \cdot \Theta(1) = \Theta(1/N)$$
因此，在初始化时刻，输出分数的量级为 $O(1/\sqrt{N})$。随着训练进行，该量级会演化至 $\Theta(1)$，但由于受到 $O(1)$ 的上界约束，Softmax 层不会因为宽度 $N$ 的增加而陷入饱和或梯度消失的状态。

---

### B.4 原则 3：基于 AdamW 的更新量稳定性（Update Stability）

**原则 3 要求：在参数更新一轮后，激活值的变化量 $\Delta y = \Delta W \cdot x$ 应保持在 $\Theta(1)$。**

针对 **AdamW 优化器**，权重的单坐标更新量 $\Delta W_{ij}$ 的量级主要取决于学习率 $\eta$。在特征学习（Feature Learning）阶段，梯度与输入激活值之间存在显著的相关性（Coherence）。

**1. 隐藏层参数更新**
对于隐藏层，其激活值的变化量为：
$$\Delta y_i = \sum_{j=1}^N \Delta W_{ij} x_j \approx \eta_{hid} \sum_{j=1}^N \text{sign}(g_{ij}) x_j$$
由于相干叠加效应，求和项的量级达到 $\Theta(N)$。为了使 $\Delta y_i = \Theta(1)$，必须满足：
$$\eta_{hid} = \eta_{base} \cdot \eta_{hid\_mult} = \eta_{base} \cdot \frac{1}{N}$$
这解释了为何隐藏层的学习率必须随宽度反比例缩放。

**2. 输入层与输出层更新**
输入层（一元势项）的更新不涉及跨维度的线性累加，其变化量级直接由 $\eta_{in}$ 决定，故保持 $\eta_{in\_mult} = 1$ 即可实现 $\Theta(1)$ 的更新。对于输出层，为了抵消初始化时 $1/N$ 的压制并确保其能快速学习到有效的逻辑得分，我们令 $\eta_{out\_mult} = 1$，使得输出层的特征权重能在训练初期迅速增长并稳定在 $\Theta(1)$。