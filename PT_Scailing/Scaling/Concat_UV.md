# 重新定义：耦合后的全局交互矩阵

为了推导清晰，我们首先定义堆叠矩阵（Stacked Matrices）和耦合交互矩阵（Coupled Interaction Matrix）。

- 令 $\mathbf{U} \in \mathbb{R}^{d \times (Cr)}$ 为所有通道投影矩阵的拼接：$\mathbf{U} = [U^{(1)} \dots U^{(C)}]$
- 令 $\mathbf{V} \in \mathbb{R}^{d \times (Cr)}$ 为所有通道投影矩阵的拼接：$\mathbf{V} = [V^{(1)} \dots V^{(C)}]$
- 令 $\mathbf{W}_{mix} \in \mathbb{R}^{(Cr) \times (Cr)}$ 为通道混合矩阵。

此时，原先分解的 $T^{(c)}$ 不再独立存在，而是合并为一个全局的耦合三元势矩阵 $\mathbf{T}_{\text{coupled}}$：

$$
(\mathbf{T}_{\text{coupled}})_{a,b} = (\mathbf{U} \mathbf{W}_{mix} \mathbf{V}^T)_{a,b}
= \sum_{P=1}^{Cr} \sum_{Q=1}^{Cr} \mathbf{U}_{a,P} (\mathbf{W}_{mix})_{P,Q} \mathbf{V}_{b,Q}
$$

---

# MFVI 完整推导

变分自由能函数：
$$
F(Q) = \cdots \text{(一元/二元势不变)} \cdots - \sum_{i=1}^n \sum_{j \ne i} \sum_{a=1}^d \sum_{b=1}^d Q_i(a) Q_j(b) (\mathbf{T}_{\text{coupled}})_{a,b} - H(Q)
$$

我们直接对 $F(Q)$ 关于 $Q_i(a)$ 求导。根据全微分公式，$\nabla E$ 分为两部分：

1. $i$ 作为 Child (Query 端)：$i$ 主动去关注 $j$。对应矩阵 $\mathbf{T}$ 的第一个下标 $a$。
2. $i$ 作为 Head (Key 端)：$k$ 主动来关注 $i$。对应矩阵 $\mathbf{T}$ 的第二个下标 $a$。

$$
\text{Logits}_i(a) = -\nabla E = \nabla_{\text{unary}} + \nabla_{\text{binary}} + \underbrace{\nabla_{\text{child}}}_{\text{Term 1}} + \underbrace{\nabla_{\text{head}}}_{\text{Term 2}}
$$

---

## 推导 $\nabla_{\text{child}}$ ($i$ 关注 $j$)

这一项对应原式中的 $\sum_j \sum_b Q_j(b) T_{a,b}$ (合并了各个channel c)。

$$
\nabla_{\text{child}}(a) = \sum_{j \ne i} \sum_b Q_j(b) (\mathbf{T}_{\text{coupled}})_{a,b}
$$

代入 $\mathbf{T}_{\text{coupled}} = \mathbf{U} \mathbf{W}_{mix} \mathbf{V}^T$：

$$
= \sum_{j \ne i} \sum_b Q_j(b) \left( \sum_{P,Q} \mathbf{U}_{a,P} (\mathbf{W}_{mix})_{P,Q} \mathbf{V}_{b,Q} \right)
$$

写成矩阵向量形式（令 $q_j$ 为 $Q_j$ 的向量表示）：

$$
= \sum_{j \ne i} \left( \mathbf{U} \mathbf{W}_{mix} \mathbf{V}^T q_j \right)_a
= \left( \mathbf{U} \cdot \mathbf{W}_{mix} \cdot \text{Concat} \left[ \sum_j Q_j V^{(c)} \right] \right)_a
$$

> **物理意义**：$i$ 作为 Query，去查询经过混合后的全局 Key/Value 信息。这与上一轮推导一致。

---

## 推导 $\nabla_{\text{head}}$ ($k$ 关注 $i$)

这一项对应原式中的 $\sum_k \sum_b Q_k(b) T_{b,a}$ (合并了各个channel c)。

$$
\nabla_{\text{head}}(a) = \sum_{k \ne i} \sum_{b=1}^d Q_k(b) (\mathbf{T}_{\text{coupled}})_{b,a}
$$

代入 $\mathbf{T}_{\text{coupled}}$ 的元素表达式：

$$
= \sum_{k \ne i} \sum_b Q_k(b) \left( \sum_{P,Q} \mathbf{U}_{b,P} (\mathbf{W}_{mix})_{P,Q} \mathbf{V}_{a,Q} \right)
$$

这里关键在于 $\mathbf{V}_{a,Q}$ 是 $\mathbf{V}$ 的第 $a$ 行第 $Q$ 列，即 $\mathbf{V}^T$ 的第 $Q$ 行第 $a$ 列。我们要凑出关于 index $a$ 的向量形式，必须进行转置处理。

我们将上式视为向量 $x$ 与 $q_k$ 的内积：

$$
(\mathbf{T}_{\text{coupled}})_{b,a} = e_b^T (\mathbf{U} \mathbf{W}_{mix} \mathbf{V}^T) e_a
= \left( (\mathbf{U} \mathbf{W}_{mix} \mathbf{V}^T)^T e_b \right)_a
$$

$$
= \left( \mathbf{V} \mathbf{W}_{mix}^T \mathbf{U}^T e_b \right)_a
$$

代回求和式：

$$
\nabla_{\text{head}} = \sum_{k \ne i} \mathbf{V} \mathbf{W}_{mix}^T \mathbf{U}^T q_k
$$

---

# 最终 MFVI 更新公式

$$
\text{Logits}_i(a) = \text{Unary} + \text{Binary} + \underbrace{\left( \mathbf{U} \mathbf{W}_{mix} \cdot \text{Concat}\left[ \sum_j Q_j V^{(c)} \right] \right)_a}_{\text{Child Term}} + \underbrace{\left( \mathbf{V} \mathbf{W}_{mix}^T \mathbf{U}^T \cdot \sum_{k \ne i} q_k \right)_a}_{\text{Head Term}}
$$

---

# Logits

MFVI 更新公式为：

$$
\text{Logits}_i(a) = -\nabla E = S_{w_i,a} + \sum_g Q_i^G(g) B_{g,a} + \text{Term}_{\text{child}} + \text{Term}_{\text{head}}
$$

## Term Child

$$
\text{Term}_{\text{child}} = \sum_{j \ne i} \sum_b Q_j(b) (\mathbf{T}_{\text{coupled}})_{a,b}
$$

代入 $\mathbf{T}_{\text{coupled}} = \mathbf{U} \mathbf{W}_{mix} \mathbf{V}^T$：

$$
\text{Term}_{\text{child}} = \left( \mathbf{U} \cdot \mathbf{W}_{mix} \cdot \sum_{j \ne i} (\mathbf{V}^T Q_j^T) \right)_a
$$

### 矩阵分块形式：
这里的 $\sum (\mathbf{V}^T Q_j^T)$ 实际上是对所有通道的 Value 进行加权聚合。

$$
\text{Term}_{\text{child}} = \mathbf{U} \cdot \mathbf{W}_{mix} \cdot 
\begin{bmatrix}
\sum_j \text{Attn}_{ij}^{(1)} (Q_j V^{(1)})^T \\
\vdots \\
\sum_j \text{Attn}_{ij}^{(C)} (Q_j V^{(C)})^T
\end{bmatrix}
$$

---

## Term Head


$$
\text{Term}_{\text{head}} = \sum_{k \ne i} \sum_b Q_k(b) (\mathbf{T}_{\text{coupled}})_{b,a}
$$

注意下标顺序是 $(b, a)$。所以需要 $\mathbf{T}_{\text{coupled}}^T$ 的第 $(a, b)$ 项。

$$
(\mathbf{T}_{\text{coupled}})^T = (\mathbf{U} \mathbf{W}_{mix} \mathbf{V}^T)^T = \mathbf{V} \mathbf{W}_{mix}^T \mathbf{U}^T
$$

代入求和式：

$$
\text{Term}_{\text{head}} = \sum_{k \ne i} \left( \mathbf{V} \mathbf{W}_{mix}^T \mathbf{U}^T Q_k^T \right)_a
$$

### 矩阵分块形式：

$$
\text{Term}_{\text{head}} = \mathbf{V} \cdot \mathbf{W}_{mix}^T \cdot 
\begin{bmatrix}
\sum_k \text{Attn}_{ki}^{(1)} (Q_k U^{(1)})^T \\
\vdots \\
\sum_k \text{Attn}_{ki}^{(C)} (Q_k U^{(C)})^T
\end{bmatrix}
$$

---

## 最终完整表达式

$$
\text{Logits}_i = S_{w_i} + \sum_g Q_i^G(g) B_g + \mathbf{U} \mathbf{W}_{mix} \mathbf{v}_{agg} + \mathbf{V} \mathbf{W}_{mix}^T \mathbf{u}_{agg}
$$

其中两个聚合向量（Aggregated Vectors）定义如下（维度均为 $Cr \times 1$）：

$$
\mathbf{v}_{agg} = \text{Concat} \begin{bmatrix}
\sum_j Q_{ic}(j) (Q_j V^{(1)}) \\
\vdots \\
\sum_j Q_{ic}(j) (Q_j V^{(C)})
\end{bmatrix} \quad \text{(收集 Value)}
$$

$$
\mathbf{u}_{agg} = \text{Concat} \begin{bmatrix}
\sum_k Q_{kc}(i) (Q_k U^{(1)}) \\
\vdots \\
\sum_k Q_{kc}(i) (Q_k U^{(C)})
\end{bmatrix} \quad \text{(收集 Query)}
$$