# 原版PT的MFVI推导

### 符号约定
$d$:`dim_z`
$m$:`dim_g`
$n$:`seq_len`

### 势函数

$\phi_u(Z_i=a)=\exp(S_{w_i,a})$
$\phi_t(H_i=k,Z_i=a,Z_j=b)=\exp(T_{a,b}) (H_i=j,否则为1)$
$\phi_b(Z_i=a,G_i=g)=\exp(B_{g,a})$

### 联合概率分布

给定序列$w$（坐标为$i \in 1 \sim n$）,给定各位置$i$的$Z_i,G_i,H_i$，此取值概率为：$P(Z,H,G|w)=\text{Normalization}\{\Pi_{i=1}^n\phi_u(Z_i)*\Pi_{i=1}^n\phi_b(Z_i,G_i)*\Pi_{c=1}^h\Pi_{i=1}^n\Pi_{j=1}^n\phi_t(H_i^{(c)},Z_i,Z_j) \}$

### MFVI

**变分自由能函数**：$F(Q)=E_Q[-\log P(Z,H,G|w)]-H(Q)$
$=-\sum_{i=1}^n(\sum_{a=1}^dQ_i(Z_i=a)S_{w_i,a})\\ -\sum_{c=1}^C\sum_{i=1}^n(\sum_{j\neq i}\sum_{a=1}^d\sum_{b=1}^dQ_{ic}(H_i=j)Q_i(Z_i=a)Q_j(Z_j=b)T_{a,b}^{(c)}) \\ -\sum_{i=1}^n(\sum_{a=1}^d\sum_{g=1}^mQ_i(Z_i=a)Q_i^G(G_i=g)B_{g,a} )\\ +H(Q) $

**MFVI**: 求$Q_i(a)$分布使得$F(Q)$取得最小值，且$\sum_aQ_i(a)=1$。以此构建Lagrange函数，使其对$Q_i(a)$偏导为0，解出一次迭代后的$Q_i(a)$。

**Lagrange**:$L=F(Q)+\lambda(\sum_zQ_i(a)-1)$
偏导为0$\Rightarrow$ $\nabla E+\ln Q_i(a)+1+\lambda=0 $ (其中$\nabla E$为$E_Q[-\log P(Z,H,G|w)]$对$Q_i(a)$的偏导，$\ln Q_i(a)+1$为$H(Q)$对$Q_i(a)$的偏导)

**$\nabla E_{Q_{ic}(j)}$的求解与$Q_{ic}(j)$的更新：**

$\frac{\partial E }{\partial Q_{ic}(j) }=\sum_a\sum_b Q_i(a)Q_j(b)T_{a,b}(c) $
$=\sum_a\sum_b Q_i(a)Q_j(b)(\sum_lU_{a,l}V_{b,l}) $
$=\sum_l(\sum_aQ_i(a)U_{a,l})(\sum_bQ_j(b)V_{b,l}) $
令$\sum_aQ_i(a)U_{a,l}=q_{i,l}$,$\sum_bQ_j(b)V_{b,l}=k_{j,l}$
则$\nabla E_{Q_{ic}(j)}=\sum_{l=1}^rq_{i,l}k_{j,l}=-(\vec{q_i}\vec{k_j^T})$

由$Q\propto \exp(-\nabla E)$得：$Q_{ic}(j)\propto \exp(\vec{q_i}\vec{k_j^T}) $

**$\nabla E_{Q_{i}(a)}$的求解与$Q_{i}(a)$的更新：**

对$F$,只考虑含$Q_i(a)$的项：
$F_i(Q) = -Q_i(a)S_{w_i,a} \\ -\sum_c \sum_{j \ne i} Q_{ic}(j) \sum_b Q_i(a)Q_j(b)T^{(c)}_{ab}\\ - \sum_c \sum_{k \ne i} Q_{kc}(i) \sum_b Q_k(b)Q_i(a)T^{(c)}_{ba}\\ - \sum_g Q_i(a)Q_i^G(g)B_{g,a}$

$\nabla_{unary}=-S_{w_i,a}$
$\nabla_{child}=-\sum_c\sum_{j\neq i}Q_{ic}(j)(\sum_b Q_j(b)T_{a,b}^{(c)}) $
$\nabla_{head}=-\sum_c\sum_{k\neq i}Q_{kc}(i)(\sum_b Q_k(b)T_{b,a}^{(c)}) $
$\nabla_{binary}=-\sum_gQ_i^G(g)G_{g,a} $
$\therefore \nabla E=\nabla_{unary}+\nabla_{child}+\nabla_{head}+\nabla_{binary} $
代入Lagrange函数，令其偏导为0$\Rightarrow$ $\ln Q_i(a)=-\nabla E-(1+\lambda)$
$\Rightarrow Q_i(a)\propto \exp(-\nabla E) $
$\Rightarrow Q_i(a)=\text{Softmax}(-\nabla E)$

即：$\text{Logits}_i(a)=-\nabla E$
$=S_{w_i,a}+\sum_c\sum_{j\neq i}Q_{ic}(j)(\sum_b Q_j(b)T_{a,b}^{(c)})+\sum_c\sum_{k\neq i}Q_{kc}(i)(\sum_b Q_k(b)T_{b,a}^{(c)})+\sum_gQ_i^G(g)B_{g,a} $

考虑$T$矩阵分解：

$\sum_{j\neq i}Q_{ic}(j)(\sum_b Q_j(b)T_{a,b}^{(c)})=\sum_j \text{Attn}_{ij}(Q_jT)=\sum_j \text{Attn}_{ij}(Q_jV)U^T $
$\sum_{k\neq i}Q_{kc}(i)(\sum_b Q_k(b)T_{b,a}^{(c)})=\sum_k \text{Attn}_{ki}(Q_kT)=\sum_k \text{Attn}_{ki}(Q_kU)V^T $


相应代码：`qz_new = Softmax(Input_S + (Attn_Child + Attn_Head) + FFN_Output)`

---

# muP版 PT的MFVI推导

### 势函数 (muP Scaled)

$\phi_u(Z_i=a)=\exp(S_{w_i,a})$
$\phi_t(H_i=k,Z_i=a,Z_j=b)=\exp(\frac{d^2}{r} T_{a,b}) (H_i=j,否则为1)$
$\phi_b(Z_i=a,G_i=g)=\exp(d \cdot m B_{g,a})$

### 联合概率分布

给定序列$w$（坐标为$i \in 1 \sim n$）,给定各位置$i$的$Z_i,G_i,H_i$，此取值概率为：$P(Z,H,G|w)=\text{Normalization}\{\Pi_{i=1}^n\phi_u(Z_i)*\Pi_{i=1}^n\phi_b(Z_i,G_i)*\Pi_{c=1}^h\Pi_{i=1}^n\Pi_{j=1}^n\phi_t(H_i^{(c)},Z_i,Z_j) \}$

### MFVI

**变分自由能函数**：$F(Q)=E_Q[-\log P(Z,H,G|w)]-H(Q)$
$=-\sum_{i=1}^n(\sum_{a=1}^dQ_i(Z_i=a)S_{w_i,a})\\ -\sum_{c=1}^C\sum_{i=1}^n(\sum_{j\neq i}\sum_{a=1}^d\sum_{b=1}^dQ_{ic}(H_i=j)Q_i(Z_i=a)Q_j(Z_j=b)(\frac{d^2}{r}T_{a,b}^{(c)})) \\ -\sum_{i=1}^n(\sum_{a=1}^d\sum_{g=1}^mQ_i(Z_i=a)Q_i^G(G_i=g)(d \cdot m B_{g,a}) )\\ +H(Q) $

**MFVI**: 求$Q_i(a)$分布使得$F(Q)$取得最小值，且$\sum_aQ_i(a)=1$。以此构建Lagrange函数，使其对$Q_i(a)$偏导为0，解出一次迭代后的$Q_i(a)$。

**Lagrange**:$L=F(Q)+\lambda(\sum_zQ_i(a)-1)$
偏导为0$\Rightarrow$ $\nabla E+\ln Q_i(a)+1+\lambda=0 $ (其中$\nabla E$为$E_Q[-\log P(Z,H,G|w)]$对$Q_i(a)$的偏导，$\ln Q_i(a)+1$为$H(Q)$对$Q_i(a)$的偏导)

**$\nabla E_{Q_{ic}(j)}$的求解与$Q_{ic}(j)$的更新：**

$\frac{\partial E }{\partial Q_{ic}(j) }=\sum_a\sum_b Q_i(a)Q_j(b)(\frac{d^2}{r}T_{a,b}(c)) $
引入代码变量 $q^{\text{code}}=d Q$，则 $Q=q^{\text{code}}/d$：
$=\frac{d^2}{r} \sum_a\sum_b \frac{q_i^{\text{code}}(a)}{d} \frac{q_j^{\text{code}}(b)}{d} T_{a,b}^{(c)} $
$=\frac{1}{r} \sum_a\sum_b q_i^{\text{code}}(a) q_j^{\text{code}}(b) (\sum_l U_{a,l}V_{b,l}) $
$=\frac{1}{r} \sum_l(\sum_a q_i^{\text{code}}(a)U_{a,l})(\sum_b q_j^{\text{code}}(b)V_{b,l}) $
令$\sum_a q_i^{\text{code}}(a)U_{a,l}=q_{i,l}^{\text{proj}}$,$\sum_b q_j^{\text{code}}(b)V_{b,l}=k_{j,l}^{\text{proj}}$
则$\nabla E_{Q_{ic}(j)}=\frac{1}{r} \sum_{l=1}^r q_{i,l}^{\text{proj}} k_{j,l}^{\text{proj}} = - \frac{1}{r} (\vec{q_i^{\text{proj}}}\vec{k_j^{\text{proj}T}})$

由$Q\propto \exp(-\nabla E)$得：$Q_{ic}(j)\propto \exp(\frac{1}{r} \vec{q_i^{\text{proj}}}\vec{k_j^{\text{proj}T}}) $

**$\nabla E_{Q_{i}(a)}$的求解与$Q_{i}(a)$的更新：**

对$F$,只考虑含$Q_i(a)$的项：
$F_i(Q) = -Q_i(a)S_{w_i,a} \\ -\sum_c \sum_{j \ne i} Q_{ic}(j) \sum_b Q_i(a)Q_j(b)(\frac{d^2}{r}T^{(c)}_{ab})\\ - \sum_c \sum_{k \ne i} Q_{kc}(i) \sum_b Q_k(b)Q_i(a)(\frac{d^2}{r}T^{(c)}_{ba})\\ - \sum_g Q_i(a)Q_i^G(g)(d \cdot m B_{g,a})$

引入代码变量 $q^{\text{code}}=d Q$ 及 $q^{G,\text{code}}=m Q^G$：

$\nabla_{unary}=-S_{w_i,a}$
$\nabla_{child}=-\sum_c\sum_{j\neq i}Q_{ic}(j)(\sum_b \frac{q_j^{\text{code}}(b)}{d} \frac{d^2}{r} T_{a,b}^{(c)}) = -\frac{d}{r} \sum_c\sum_{j\neq i}Q_{ic}(j)(\sum_b q_j^{\text{code}}(b)T_{a,b}^{(c)}) $
$\nabla_{head}=-\sum_c\sum_{k\neq i}Q_{kc}(i)(\sum_b \frac{q_k^{\text{code}}(b)}{d} \frac{d^2}{r} T_{b,a}^{(c)}) = -\frac{d}{r} \sum_c\sum_{k\neq i}Q_{kc}(i)(\sum_b q_k^{\text{code}}(b)T_{b,a}^{(c)})$
$\nabla_{binary}=-\sum_g \frac{q_i^{G,\text{code}}(g)}{m} (d \cdot m B_{g,a}) = -d \sum_g q_i^{G,\text{code}}(g) B_{g,a}$

$\therefore \nabla E=\nabla_{unary}+\nabla_{child}+\nabla_{head}+\nabla_{binary} $
代入Lagrange函数，令其偏导为0$\Rightarrow$ $\ln Q_i(a)=-\nabla E-(1+\lambda)$
$\Rightarrow Q_i(a)\propto \exp(-\nabla E) $
$\Rightarrow Q_i(a)=\text{Softmax}(-\nabla E)$

即：$\text{Logits}_i(a)=-\nabla E$
$=S_{w_i,a} + \frac{d}{r}\sum_c\sum_{j\neq i}Q_{ic}(j)(\sum_b q_j^{\text{code}}(b)T_{a,b}^{(c)}) + \frac{d}{r}\sum_c\sum_{k\neq i}Q_{kc}(i)(\sum_b q_k^{\text{code}}(b)T_{b,a}^{(c)}) + d \sum_g q_i^{G,\text{code}}(g)B_{g,a} $

考虑$T$矩阵分解及代码对应（常数系数 $d$ 被吸收到 Scaling 或作为能量补偿体现）：

$\text{Term}_{child} \propto \sum_j \text{Attn}_{ij}(q_j^{\text{code}}V)U^T $
$\text{Term}_{head} \propto \sum_k \text{Attn}_{ki}(q_k^{\text{code}}U)V^T $
$\text{Term}_{binary} \propto q^{G,\text{code}} B $

相应代码：
`message_F = (qz_u @ qz_v.T) / ternary_rank`
`qz_new = Softmax(Input_S + (Attn_Child + Attn_Head) + FFN_Output)`