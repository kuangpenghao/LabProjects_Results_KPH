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

# 两个温度参数$\tau$与$1$合理性的推导