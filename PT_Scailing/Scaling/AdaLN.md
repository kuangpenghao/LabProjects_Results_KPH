**输入**: `input_ids`, `unary_potentials`
**参数**:
*   `Shared Weights`: $W_u, W_v, W_{topic}$ (所有层共享)
*   `AdaLN Params`: $\Gamma \in \mathbb{R}^{L \times N}, \text{B} \in \mathbb{R}^{L \times N}$ ($L$为迭代次数, $N$为dim\_z)

1.  初始化 $qz_0 = \text{unary\_potentials}$
2.  循环 $t = 0 \to L-1$:
    *   **提取当前层的调制参数**:
        $\gamma_t = \Gamma[t]$, $\beta_t = \text{B}[t]$
    *   **归一化 (保持原逻辑 + Energy Recovery)**:
        $z_{norm} = \text{Norm}(qz_t) \times \text{dim\_z}$
    *   **[关键步骤] AdaLN 调制**:
        $z_{mod} = z_{norm} \cdot (1 + \gamma_t) + \beta_t$
    *   **计算功能流 (使用调制后的 $z_{mod}$)**:
        $m_1 = \text{HeadSelection}(z_{mod}, W_u, W_v)$
        $m_2 = \text{TopicModeling}(z_{mod}, W_{topic})$
    *   **状态更新 (Structural Update)**:
        $qz_{t+1} = (m_1 + m_2 + \text{unary}) / \text{regularize\_z}$
        $qz_{t+1} = \text{Damp}(qz_{t+1}, qz_t)$
3.  输出 $qz_L$