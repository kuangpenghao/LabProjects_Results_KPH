# 原版PT的MFVI推导

### 符号约定
$d$:`dim_z`
$m$:`dim_g`
$n$:`seq_len`
$r$:`ternary_rank`

### 势函数

$\phi_u(Z_i=a)=\exp(S_{w_i,a})$
$\phi_t(H_i=k,Z_i=a,Z_j=b)=\exp(T_{a,b}) (H_i=j,否则为1)$
$\phi_b(Z_i=a,G_i=g)=\exp(B_{g,a})$

### 联合概率分布

给定序列$w$（坐标为$i \in 1 \sim n$）,给定各位置$i$的$Z_i,G_i,H_i$，此取值概率为：$P(Z,H,G|w)=\text{Normalization}\{\Pi_{i=1}^n\phi_u(Z_i)*\Pi_{i=1}^n\phi_b(Z_i,G_i)*\Pi_{c=1}^h\Pi_{i=1}^n\Pi_{j=1}^n\phi_t(H_i^{(c)},Z_i,Z_j) \}$

### MFVI

**变分自由能函数**：$F(Q)=E_Q[-\log P(Z,H,G|w)]-H(Q)$
$=-\sum_{i=1}^n(\sum_{a=1}^dQ_i(Z_i=a)S_{w_i,a})\\ -\sum_{c=1}^C\sum_{i=1}^n(\sum_{j\neq i}\sum_{a=1}^d\sum_{b=1}^dQ_{ic}(H_i=j)Q_i(Z_i=a)Q_j(Z_j=b)T_{a,b}^{(c)}) \\ -\sum_{i=1}^n(\sum_{a=1}^d\sum_{g=1}^mQ_i(Z_i=a)Q_i^G(G_i=g)B_{g,a} )\\ -H(Q) $

**MFVI**: 求$Q_i(a)$分布使得$F(Q)$取得最小值，且$\sum_aQ_i(a)=1$。以此构建Lagrange函数，使其对$Q_i(a)$偏导为0，解出一次迭代后的$Q_i(a)$。

**Lagrange**:$L=F(Q)+\lambda(\sum_zQ_i(a)-1)$
偏导为0$\Rightarrow$ $\nabla E-\ln Q_i(a)-1+\lambda=0 $ (其中$\nabla E$为$E_Q[-\log P(Z,H,G|w)]$对$Q_i(a)$的偏导，$\ln Q_i(a)+1$为$H(Q)$对$Q_i(a)$的偏导)

**$\nabla E_{Q_{ic}(j)}$的求解与$Q_{ic}(j)$的更新：**

$\frac{\partial E }{\partial Q_{ic}(j) }=-\sum_a\sum_b Q_i(a)Q_j(b)T_{a,b}(c) $
$=-\sum_a\sum_b Q_i(a)Q_j(b)(\sum_lU_{a,l}V_{b,l}) $
$=-\sum_l(\sum_aQ_i(a)U_{a,l})(\sum_bQ_j(b)V_{b,l}) $
令$\sum_aQ_i(a)U_{a,l}=q_{i,l}$,$\sum_bQ_j(b)V_{b,l}=k_{j,l}$
则$\nabla E_{Q_{ic}(j)}=-\sum_{l=1}^rq_{i,l}k_{j,l}=-(\vec{q_i}\vec{k_j^T})$

由$Q\propto \exp(-\nabla E)$得：$Q_{ic}(j)\propto \exp(\vec{q_i}\vec{k_j^T}) $

**$\nabla E_{Q_{i}(a)}$的求解与$Q_{i}(a)$的更新：**

对$F$,只考虑含$Q_i(a)$的项：
$F_i(Q) = -Q_i(a)S_{w_i,a} \\ -\sum_c \sum_{j \ne i} Q_{ic}(j) \sum_b Q_i(a)Q_j(b)T^{(c)}_{ab}\\ - \sum_c \sum_{k \ne i} Q_{kc}(i) \sum_b Q_k(b)Q_i(a)T^{(c)}_{ba}\\ - \sum_g Q_i(a)Q_i^G(g)B_{g,a}$

$\nabla_{unary}=-S_{w_i,a}$
$\nabla_{child}=-\sum_c\sum_{j\neq i}Q_{ic}(j)(\sum_b Q_j(b)T_{a,b}^{(c)}) $
$\nabla_{head}=-\sum_c\sum_{k\neq i}Q_{kc}(i)(\sum_b Q_k(b)T_{b,a}^{(c)}) $
$\nabla_{binary}=-\sum_gQ_i^G(g)G_{g,a} $
$\therefore \nabla E=\nabla_{unary}+\nabla_{child}+\nabla_{head}+\nabla_{binary} $
代入Lagrange函数，令其偏导为0$\Rightarrow$ $\ln Q_i(a)=-\nabla E+1-\lambda$
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

$\tau=\frac{d}{r}$，为$Q_i$分布的温度参数。$Q_{ic}$分布的温度参数仍为1。后续进行推导解释。
$\phi_u(Z_i=a)=\exp(\tau S_{w_i,a})$
$\phi_t(H_i=k,Z_i=a,Z_j=b)=\exp(\tau d T_{a,b}) (H_i=j,否则为1)$
$\phi_b(Z_i=a,G_i=g)=\exp(\tau m B_{g,a})$

### 联合概率分布

给定序列$w$（坐标为$i \in 1 \sim n$）,给定各位置$i$的$Z_i,G_i,H_i$，此取值概率为：$P(Z,H,G|w)=\text{Normalization}\{\Pi_{i=1}^n\phi_u(Z_i)*\Pi_{i=1}^n\phi_b(Z_i,G_i)*\Pi_{c=1}^h\Pi_{i=1}^n\Pi_{j=1}^n\phi_t(H_i^{(c)},Z_i,Z_j) \}$

### MFVI

**变分自由能函数**：$F(Q)=E_Q[-\log P(Z,H,G|w)]-\tau H(Q)$(添加了系数$\tau$，后续进行推导)
$=-\sum_{i=1}^n(\sum_{a=1}^dQ_i(Z_i=a)\tau S_{w_i,a})\\ -\sum_{c=1}^C\sum_{i=1}^n(\sum_{j\neq i}\sum_{a=1}^d\sum_{b=1}^dQ_{ic}(H_i=j)Q_i(Z_i=a)Q_j(Z_j=b)(\tau dT_{a,b}^{(c)})) \\ -\sum_{i=1}^n(\sum_{a=1}^d\sum_{g=1}^mQ_i(Z_i=a)Q_i^G(G_i=g)(\tau m B_{g,a}) )\\ -\tau H(Q) $

**MFVI**: 求$Q_i(a)$分布使得$F(Q)$取得最小值，且$\sum_aQ_i(a)=1$。以此构建Lagrange函数，使其对$Q_i(a)$偏导为0，解出一次迭代后的$Q_i(a)$。

**Lagrange**:$L=F(Q)+\lambda(\sum_zQ_i(a)-1)$
偏导为0$\Rightarrow$ $\nabla E-\tau (\ln Q_i(a)+1)+\lambda=0 $ (其中$\nabla E$为$E_Q[-\log P(Z,H,G|w)]$对$Q_i(a)$的偏导，$\ln Q_i(a)+1$为$H(Q)$对$Q_i(a)$的偏导)

**$\nabla E_{Q_{ic}(j)}$的求解与$Q_{ic}(j)$的更新：**

$\frac{\partial E }{\partial Q_{ic}(j) }=-\sum_a\sum_b Q_i(a)Q_j(b)(\tau dT_{a,b}(c)) $
$=-\tau d \sum_a\sum_b Q_i(a)Q_j(b)(\sum_lU_{a,l}V_{b,l}) $
$=-\tau d \sum_l(\sum_aQ_i(a)U_{a,l})(\sum_bQ_j(b)V_{b,l}) $
令$\sum_aQ_i(a)U_{a,l}=q_{i,l}$,$\sum_bQ_j(b)V_{b,l}=k_{j,l}$
则$\nabla E_{Q_{ic}(j)}=-\tau d \sum_{l=1}^rq_{i,l}k_{j,l}=-(\frac{d^2}{r} \vec{q_i}\vec{k_j^T})$

由$Q\propto \exp(-\nabla E)$得：$Q_{ic}(j)\propto \exp(\frac{d^2}{r} \vec{q_i}\vec{k_j^T}) $

**$\nabla E_{Q_{i}(a)}$的求解与$Q_{i}(a)$的更新：**

对$F$,只考虑含$Q_i(a)$的项：
$F_i(Q) = -Q_i(a)\tau S_{w_i,a} \\ -\sum_c \sum_{j \ne i} Q_{ic}(j) \sum_b Q_i(a)Q_j(b)(\tau dT^{(c)}_{ab})\\ - \sum_c \sum_{k \ne i} Q_{kc}(i) \sum_b Q_k(b)Q_i(a)(\tau dT^{(c)}_{ba})\\ - \sum_g Q_i(a)Q_i^G(g)(\tau m B_{g,a})$

$\nabla_{unary}=-\tau S_{w_i,a}$
$\nabla_{child}=-\sum_c\sum_{j\neq i}Q_{ic}(j)(\sum_b Q_j(b) \tau d T_{a,b}^{(c)}) $
$\nabla_{head}=-\sum_c\sum_{k\neq i}Q_{kc}(i)(\sum_b Q_k(b) \tau d T_{b,a}^{(c)}) $
$\nabla_{binary}=-\sum_gQ_i^G(g)(\tau m B_{g,a}) $
$\therefore \nabla E=\nabla_{unary}+\nabla_{child}+\nabla_{head}+\nabla_{binary} $
代入Lagrange函数，令其偏导为0$\Rightarrow$ $\ln Q_i(a)=-\frac{1}{\tau}\nabla E+1-\frac{1}{\tau}\lambda$
$\Rightarrow Q_i(a)\propto \exp(-\frac{1}{\tau}\nabla E) $
$\Rightarrow Q_i(a)=\text{Softmax}(-\frac{1}{\tau}\nabla E)$

即：$\text{Logits}_i(a)=-\frac{1}{\tau}\nabla E$
$=\frac{1}{\tau}[\tau S_{w_i,a}+\tau d\sum_c\sum_{j\neq i}Q_{ic}(j)(\sum_b Q_j(b)T_{a,b}^{(c)})+\tau d\sum_c\sum_{k\neq i}Q_{kc}(i)(\sum_b Q_k(b)T_{b,a}^{(c)})+ \tau m \sum_gQ_i^G(g)B_{g,a}] $
$=S_{w_i,a}+d\sum_c\sum_{j\neq i}Q_{ic}(j)(\sum_b Q_j(b)T_{a,b}^{(c)})+d\sum_c\sum_{k\neq i}Q_{kc}(i)(\sum_b Q_k(b)T_{b,a}^{(c)})+m \sum_gQ_i^G(g)B_{g,a}$

考虑$T$矩阵分解：

$\sum_{j\neq i}Q_{ic}(j)(\sum_b Q_j(b)T_{a,b}^{(c)})=\sum_j \text{Attn}_{ij}(Q_jT)=\sum_j \text{Attn}_{ij}(Q_jV)U^T $
$\sum_{k\neq i}Q_{kc}(i)(\sum_b Q_k(b)T_{b,a}^{(c)})=\sum_k \text{Attn}_{ki}(Q_kT)=\sum_k \text{Attn}_{ki}(Q_kU)V^T $

---

# 代码实现的契合性检验

有关特殊缩放的代码：

```
class SquaredSoftmax(nn.Module):
    def __init__(self, dim=-1, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
    
    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        hidden_states = hidden_states.pow(2)
        # Energy Recovery: scale by dim
        hidden_states = F.normalize(hidden_states, p=1, dim=self.dim, eps=self.eps) * hidden_states.shape[self.dim]
        return hidden_states.to(input_dtype)
```

```
class AbsNormalization(nn.Module):
    def __init__(self, dim=-1, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
    
    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        hidden_states = F.relu(hidden_states)
        # Energy Recovery: scale by dim
        hidden_states = F.normalize(hidden_states, p=1, dim=self.dim, eps=self.eps) * hidden_states.shape[self.dim]
        return hidden_states.to(input_dtype)
```

即：`potential_function_z`为`square`,`potential_function_g`为`abs`，归一化时增加了`* hidden_states.shape[self.dim]`

原始代码中，涉及到归一化操作的变量为：`qz`：相当于$Q_i(a)$； `qg`：相当于$Q_i^G(g)$。原始代码中，经过归一化相当于：`qz`与`qg`显式代表了概率分布。

修改后的代码中：`qz`和`qg`分别乘上了$d$和$m$，即不再显示代表概率分布，而是代表概率分布乘宽度。

结合MFVI推导：

**$\nabla E_{Q_{ic}(j)}$的求解与$Q_{ic}(j)$的更新：**

$\frac{\partial E }{\partial Q_{ic}(j) }=-\sum_a\sum_b Q_i(a)Q_j(b)(\frac{d^2}{r}T_{a,b}(c)) $
$=-\frac{1}{r}\sum_a\sum_b (dQ_i(a))(dQ_j(b))(T_{a,b}(c))$
$=-\frac{1}{r} \sum_l(\sum_a dQ_i(a)U_{a,l})(\sum_b dQ_j(b)V_{b,l}) $
令 $\sum_a dQ_i(a)U_{a,l}=q_{i,l}$,$\sum_b dQ_j(b)V_{b,l}=k_{j,l}$
则 $\nabla E_{Q_{ic}(j)}=-\frac{1}{r} \sum_{l=1}^rq_{i,l}k_{j,l}=-(\frac{1}{r} \vec{q_i}\vec{k_j^T})$

此时`qz`即代表$dQ_i(a)$整体。原始代码中的`Message_F`即为$-\vec{q_i}\vec{k_j^T}$。为保持与MFVI推断的一致性，需将`Message_F`额外除以r（ternary_rank）。此时代码中`Message_F`即代表$-(\frac{1}{r} \vec{q_i}\vec{k_j^T})$，归一化后得$Q_ic(j)$，对应的变量为`qh`。

其余与原始MFVI推导一致。

**$\nabla E_{Q_{i}(a)}$的求解与$Q_{i}(a)$的更新：**

$\text{Logits}_i(a)=-\frac{1}{\tau}\nabla E$
$=\frac{1}{\tau}[\tau S_{w_i,a}+\tau d\sum_c\sum_{j\neq i}Q_{ic}(j)(\sum_b Q_j(b)T_{a,b}^{(c)})+\tau d\sum_c\sum_{k\neq i}Q_{kc}(i)(\sum_b Q_k(b)T_{b,a}^{(c)})+ \tau m \sum_gQ_i^G(g)B_{g,a}] $
$=\frac{1}{\tau}[\tau S_{w_i,a}+\tau \sum_c\sum_{j\neq i}Q_{ic}(j)(\sum_b dQ_j(b)T_{a,b}^{(c)})+\tau \sum_c\sum_{k\neq i}Q_{kc}(i)(\sum_b dQ_k(b)T_{b,a}^{(c)})+ \tau  \sum_g mQ_i^G(g)B_{g,a}] $
$=S_{w_i,a}+\sum_c\sum_{j\neq i}Q_{ic}(j)(\sum_b dQ_j(b)T_{a,b}^{(c)})+\sum_c\sum_{k\neq i}Q_{kc}(i)(\sum_b dQ_k(b)T_{b,a}^{(c)})+ \sum_g mQ_i^G(g)B_{g,a} $

其中$dQ_j$,$dQ_k(b)$,$mQ_i^G(g)$仍对应代码中的`qz` `qg`，这两者仍然不再显示表示概率分布，但上式最终计算结果为概率分布。

Conclusion: 经过调整后，仍然能够符合MFVI的推导步骤。区别在于缩放因子被合入相关变量中。

代码总共进行了两处与原始PT不同的特殊缩放：
1.Normalization函数中全部乘`self.dim[-1]`，对应了`qz` `qg`意义的改变 
2.`Message_F`额外除以$r$，对应了凑齐$\nabla E_{Q_{ic}(j)}$的表达式。


---

# $Q_i$温度参数$\tau$合理性的推导（1）

在此推导第一种scaling方案：`num_channels`随$d$线性缩放，$r$保持不变。

推导结论为：能量函数与熵函数在训练中后期的数量级都为$\Theta(d)$，二者保持了数量级的平衡。而在训练前期则熵函数数量级相对较大，有利于训练初期快速脱离均匀分布状态。

**变分自由能函数**：$F(Q)=E_Q[-\log P(Z,H,G|w)]-\tau H(Q)$(添加了系数$\tau$，后续进行推导)
$=-\sum_{i=1}^n(\sum_{a=1}^dQ_i(Z_i=a)\tau S_{w_i,a})\\ -\sum_{c=1}^C\sum_{i=1}^n(\sum_{j\neq i}\sum_{a=1}^d\sum_{b=1}^dQ_{ic}(H_i=j)Q_i(Z_i=a)Q_j(Z_j=b)(\tau dT_{a,b}^{(c)})) \\ -\sum_{i=1}^n(\sum_{a=1}^d\sum_{g=1}^mQ_i(Z_i=a)Q_i^G(G_i=g)(\tau m B_{g,a}) )\\ -\tau H(Q) $

### 熵函数的数量级推导

$H(Q)=-\sum_{a=1}^dQ(a)\ln Q(a) $

**训练初期的数量级**(按均匀分布计)：$\sum_{a=1}^d \frac{1}{d}\ln \frac{1}{d}=\ln d $
$\tau H(Q)=\Theta(\frac{d}{r}\ln d)=\Theta(d\ln d) $

**训练收敛期的数量级**：
收敛期趋向于：有限K个标签占据绝大多数概率累积值，剩余标签概率极小。
$H(Q)=\Theta(\ln K)=\Theta(1) $
$\tau H(Q)=\Theta(d) $

### 能量函数一元势项的数量级推导

$E=-\sum_{a=1}^dQ_i(a)(\tau S_{w_i,a}) $
$=-\frac{1}{r}\sum_{a=1}^dq_i^{code}(a)S_{w_i,a} $(此处$q_i^{code}(a)$即指代代码中的`qz`，代表$dQ_i(a) $)

**训练初期的数量级**(按均匀分布计)：
训练初期,S的d个求和单项视为互相独立，则S的方差为d个单项的方差直接相加：$Var(E)=dVar(q_i^{code}(a)S_{w_i,a}) $
根据参数化方法，$S_{w_i,a}$数量级为$\Theta(1)$; $q_i^{code}(a)=dQ_i(a)=d\Theta(\frac{1}{d})=\Theta(1)$
$Var(E)=d\Theta(1)=\Theta(d)\Rightarrow |E|=\Theta(\sqrt{d}) $

**训练收敛期的数量级**：
训练收敛期时，$q$与$S_{w_i}$趋向对齐，即向量计算点积时，各对应项符号高度相关而非独立，所以视为同方向相加，无明显的正负抵消（在二元势部分进行了说明）。所以认为 $ \sum_{a=1}^dq_i^{code}(a)S_{w_i,a} $的d个累加项$x_a$：$x_a=\Theta(1)\Theta(1)=\Theta(1) $,所以$\sum_{a=1}^dq_i^{code}(a)S_{w_i,a}=\Theta(d)$, 所以$E=\Theta(d)$

### 能量函数二元势项的数量级推导

$E=-\sum_{i}{a}{g}Q_i(a)Q_i^G(g)(\tau mB_{g,a}) $
$=-\sum_{a}\sum_g q^{code}(a)q^{G,code}(g)B_{g,a} $（由参数化方法，B与三元势的T相关矩阵数量级为$\Theta(\frac{1}{\sqrt{d}})$）

**训练初期的数量级**(按均匀分布计)：
$Var(E)=\sum_{a,g} Var(q^{code}(a)q^{G,code}(g)B_{g,a})=\sum_{a,g} (\Theta(1)\Theta(1)\Theta(\frac{1}{d}))=\Theta(d^2)\Theta(\frac{1}{d})=\Theta(d) $
$\Rightarrow |E|=\sqrt{d} $

**训练收敛期的数量级**：
$E=-\sum_a q^{code}(a)\sum_g q^{G,code}B_{g,a} $
$\sum_g q^{G,code}B_{g,a}$中，向量点积的各累计项符号高度相关（下面为说明），视为同方向相加，无正负抵消。
求$\sum_g q^{G,code}B_{g,a}$的数量级：
由参数化方案得：$B_{g,a}=B_{g,a}^{init}+B_{g,a}^{learn}$，其中$B_{g,a}^{init}=\Theta(\frac{1}{\sqrt{d}})$, $B_{g,a}^{learn}=\Theta(\frac{1}{d})$
$\sum_g q^{G,code}B_{g,a}=\sum_g q^{G,code}B_{g,a}^{init}+q^{G,code}B_{g,a}^{learn} $
第一项数量级的推导参照独立分布，解出方差为$\Theta(1)$，即数量级为$\Theta(1)$。
第二项中，由梯度计算可得：$\Delta B_{g,a}$正比于$q$，即：趋向于同q对齐。
所以第二项的数量级为：$\sum_g \Theta(1)\Theta(\frac{1}{d})=\Theta(1)$

所以$\sum_g q^{G,code}B_{g,a}$的总数量级为$\Theta(1)$
$E$的数量级为：$\sum_a\Theta(1)\Theta(1)=\Theta(d)$

### 能量函数三元势项的数量级推导

$E=-\sum_{c=1}^C\sum_{i}\sum_jQ_{ic}(j)(\sum_{a=1}^d\sum_{b=1}^dQ_i(a)Q_j(b)\tau d T_{a,b}^{(c)}) $

单个channel能量：$E=-\sum_{i,j}Q_{ic}(j)(\sum_{a=1}^d\sum_{b=1}^dQ_i(a)Q_j(b)\tau d T_{a,b}^{(c)})=-\sum_{i,j}Q_ic(j)\sum_{a=1}^d\sum_{b=1}^dq_i^{code}(a)q_j^{code}(b)T_{a,b} $

**训练初期的数量级**：结果同一元、二元一样，过程略
**训练收敛期的数量级**：
$\sum_{a=1}^d\sum_{b=1}^dq_i^{code}(a)q_j^{code}(b)T_{a,b}=\sum_{l=1}^r\sum_{a=1}^d\sum_{b=1}^d(q_i^{code}U_{a,l})(V_{b,l}q_j^{code})=\sum_{l=1}^r(\sum_{a=1}^dq_i^{code}U_{a,l})(\sum_{b=1}^dq_j^{code}V_{b,l}) $
根据梯度计算，$U_{a,l}$的梯度与$q_i^{code}$成正比，$V_{b,l}$的梯度与$q_j^{code}$成正比，对应了$U,V$更新量的对齐关系。
$U_{a,l}=U_{a,l}^{init}+U_{a,l}^{learn}$,$V_{b,l}=V_{b,l}^{init}+V_{b,l}^{learn}$，后者有对齐关系，前者为完全随机。

$(\sum_{a=1}^dq_i^{code}U_{a,l})(\sum_{b=1}^dq_j^{code}V_{b,l})=[(\sum_{a=1}^dq_i^{code}U_{a,l}^{init})+(\sum_{a=1}^dq_i^{code}U_{a,l}^{learn})][(\sum_{b=1}^dq_j^{code}V_{b,l}^{init})+(\sum_{b=1}^dq_j^{code}V_{b,l}^{learn})]$
共有4项：1个init相乘项，2个交叉项，1个learn相乘项。
对于init相乘项，分析$\sum_{a=1}^dq_i^{code}U_{a,l}^{init}$方差可得数量级为$\Theta(1)$,$V$同理，所以相乘项数量级为$\Theta(1)$。
由于$U^{init}$与$U^{learn}$独立，且$U^{learn}$数量级为$\Theta(\frac{1}{d})<\Theta(\frac{1}{\sqrt{d}})$，故交叉相乘项数量级小于init相乘项，省略。
对于learn相乘项：$\sum_{a=1}^dq_i^{code}U_{a,l}^{learn}=\sum_{a=1}^d\Theta(1)\Theta(\frac{1}{d})=\Theta(1) $,故相乘项数量级为$\Theta(1)$
综上，$\sum_{a=1}^d\sum_{b=1}^dq_i^{code}(a)q_j^{code}(b)T_{a,b}$数量级为$\Theta(1)$

所以 $-\sum_{i,j}Q_{ic}(j)\sum_{a=1}^d\sum_{b=1}^dq_i^{code}(a)q_j^{code}(b)T_{a,b}=-\sum_{i,j}Q_{ic}(j)\Theta(1)=\Theta(1) $(等于取平均)

考虑多个channel合并：
`Message_F`在代码中额外除以`ternary_rank`=$\Theta(\frac{d}{C})$进行额外修正：单个channel能量最终数量级为$\Theta(\frac{C}{d})\Theta(1)=\Theta(\frac{C}{d})$
所有channel能量累加：$\Theta(\frac{C^2}{d})=\Theta(d)$ (注：$\Theta(C)=\Theta(d) $)

### Conclusion

**训练初期**：能量函数每项的数量级都为$\sqrt{d}$，总数量级为$\sqrt{d}$；熵函数数量级为$d\ln d$，大于能量函数数量级。即：训练初期的效果更偏向熵函数的降低
**训练中后期**：能量函数每项的数量级都为$d$，总数量级为$d$；熵函数数量级为$d$，二者相等。即：在$\tau$的当前取值下，随$d$变化，二者不会出现明显的大小失衡，使得分布与分布的熵都能维持在较为健康的状态直至训练收敛。而原先$\tau=1$的情况下则无法满足二者的均衡。

---

# $Q_i$温度参数 $\tau$ 合理性推导（2）

在此推导第二种 scaling 方案：`num_channels` ($C$) 保持不变（即 $\Theta(1)$），`ternary_rank` ($r$) 随 $d$ 线性缩放（即 $r = \Theta(d)$）。

### 前置条件

根据定义 $\tau = \frac{d}{r}$。由于在此方案中 $r \propto d$，故温度系数 $\tau$ 的数量级为：

$\tau = \Theta\left(\frac{d}{d}\right) = \Theta(1)$

### 推导结论

能量函数与熵函数在训练收敛期的数量级都为 $\Theta(1)$，二者保持了数量级的平衡。而在训练初期，熵函数数量级为 $\Theta(\ln d)$，相对于能量函数（接近 0）较大，有利于训练初期快速脱离均匀分布状态。

### 熵函数的数量级推导

$H(Q) = -\sum_{a=1}^d Q(a) \ln Q(a)$

**训练初期的数量级**（按均匀分布计）：

$H(Q) = \sum_{a=1}^d \frac{1}{d} \ln d = \ln d$

结合 $\tau = \Theta(1)$：

$\tau H(Q) = \Theta(1) \cdot \ln d = \Theta(\ln d)$

**训练收敛期的数量级**：

收敛期趋向于有限 $K$ 个标签占据绝大多数概率累积值（置信分布）。

$H(Q) = \Theta(\ln K) = \Theta(1)$

$\tau H(Q) = \Theta(1) \cdot \Theta(1) = \Theta(1)$

### 能量函数一元势项的数量级推导

$E = -\sum_{a=1}^d Q_i(a) (\tau S_{w_i,a}) = -\frac{\tau}{d} \sum_{a=1}^d q_i^{code}(a) S_{w_i,a}$

**训练初期的数量级**（按均匀分布计）：

求和项 $\sum qS$ 视为 $d$ 个独立变量的随机游走。

$\mathrm{Var}(\mathrm{Sum}) \sim d \cdot \mathrm{Var}(q \cdot S) = \Theta(d) \Rightarrow |\mathrm{Sum}| = \Theta(\sqrt{d})$

$E = \frac{\Theta(1)}{d} \cdot \Theta(\sqrt{d}) = \Theta\left(\frac{1}{\sqrt{d}}\right) \approx 0$

**训练收敛期的数量级**：

$q$ 与 $S$ 趋向对齐，视为同方向相加（相干叠加）。

$\sum_{a=1}^d q_i^{code}(a) S_{w_i,a} = \sum_{a=1}^d \Theta(1)\Theta(1) = \Theta(d)$

$E = \frac{\tau}{d} \cdot \Theta(d) = \frac{\Theta(1)}{d} \cdot \Theta(d) = \Theta(1)$

### 能量函数二元势项的数量级推导

$E = -\sum_a \sum_g Q_i(a) Q_i^G(g) (\tau r B_{g,a}) \\- \frac{\tau}{d} \sum_a q^{code}(a) \sum_g q^{G, code}(g) B_{g,a}$

**训练初期的数量级**：

同理，随机游走导致总和为 $\Theta(\sqrt{d})$ 量级，被分母 $d$ 抑制。

$E = \Theta\left(\frac{1}{\sqrt{d}}\right) \approx 0$

**训练收敛期的数量级**：

对于内部求和 $\sum_g q^{G, code} B_{g,a}$：由 $\mu P$ 参数化，$B = B^{init} + B^{learn}$。

1. **Init 部分**（$\frac{1}{\sqrt{d}}$）：随机游走 → $\Theta(1)$。
2. **Learn 部分**（$\frac{1}{d}$）：对齐叠加 $\sum_g \Theta(1)\Theta(\frac{1}{d}) = \Theta(1)$。故内部求和整体为 $\Theta(1)$。

对于外部求和 $\sum_a$：

$\sum_a q^{code}(a) \cdot |\mathrm{Inner\ Sum}| \sim \Theta(1) \cdot \Theta(d) = \Theta(d)$

代入总公式：

$E = \frac{\tau}{d} \cdot \Theta(d) = \frac{\Theta(1)}{d} \cdot \Theta(d) = \Theta(1)$

### 能量函数三元势项的数量级推导

$E = -\sum_{c=1}^C \sum_{i,j} Q_{ic}(j) \left( \sum_{a,b} Q_i(a) Q_j(b) \tau r T_{a,b}^{(c)} \right)$

在代码中，这对应于 `logits` 计算并加权。单个 Channel 的能量项（Logits 部分）：

$\text{Logits} = \frac{\sum_{a,b} q_i^{code} q_j^{code} T_{a,b}}{\text{ternary\_rank}}$

> （注：这里 $r \cdot d$ 系数被吸收到 logits 的定义与 $Q$ 的归一化中，最终表现为上述形式）

#### 核心分析：Raw Sum $\sum q q T$ 的数量级

在此方案中，Rank $r \propto d$，即 $r = \Theta(d)$。$T_{a,b} = \sum_{l=1}^r U_{a,l} V_{b,l}$。

**训练收敛期的数量级**：

1. $T$ 矩阵的元素量级：只看对齐的 Learn 部分（主导能量）：$U^{\mathrm{learn}}, V^{\mathrm{learn}} \sim \Theta(\frac{1}{d})$

$T_{a,b}^{\mathrm{learn}} = \sum_{l=1}^r U_{a,l} V_{b,l} \sim r \cdot \frac{1}{d} \cdot \frac{1}{d} = \Theta(d) \cdot \frac{1}{d^2} = \Theta\left(\frac{1}{d}\right)$

2. Raw Sum 的求和：

$\sum_{a=1}^d \sum_{b=1}^d q(a) q(b) T_{a,b} \sim d^2 \cdot \Theta(1) \cdot \Theta(1) \cdot \Theta\left(\frac{1}{d}\right) = \Theta(d)$

> （注：这里 $d^2$ 个项相干叠加）

##### 归一化后的 Logits 数量级：

$\text{Logits} = \frac{\text{Raw Sum}}{\text{ternary rank}} = \frac{\Theta(d)}{\Theta(d)} = \Theta(1)$

### 总能量：

$E = -\sum_{c=1}^C \sum_j Q_{ic}(j) \cdot \text{Logits}$

由于 $C$ 保持不变（$\Theta(1)$），且 $\sum Q_{ic} = 1$：

$E = C \cdot \Theta(1) = \Theta(1)$

## Conclusion

### 训练初期：
- **熵函数**：$\Theta(\ln d)$
- **能量函数**：$\Theta\left(\frac{1}{\sqrt{d}}\right)$（接近 0）
- **结论**：熵函数占主导地位（$\ln d \gg 0$），推动分布接近均匀分布。

### 训练中后期：
- **熵函数**：$\Theta(1)$
- **能量函数**：$\Theta(1)$


---