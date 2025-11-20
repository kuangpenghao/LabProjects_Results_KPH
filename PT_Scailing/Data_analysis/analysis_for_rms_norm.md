结论：use_rms_norm开关打开会造成巨大的方差增加，由此导致了该开关打开条件下，其他开关开闭情况对loss造成的影响急剧增大。

开关名称约定：
* 开关1：untie attention weights
* 开关2：use std residual
* 开关3：use rms norm

# 分析1：开关1 & 开关3
两种情况的比对（开关1始终打开，开关3开或者关）
<img src="comparison_analysis_13/comparison_boxplot.png">
<img src="comparison_analysis_13/comparison_ecdf.png">
<img src="comparison_analysis_13/comparison_hist_kde.png">
<img src="comparison_analysis_13/comparison_stats_bar.png">
<img src="comparison_analysis_13/comparison_violin.png">

# 分析2：开关1 & 开关2 & 开关3
两种情况的比对（开关1、2始终打开，开关3开或者关）
<img src="comparison_analysis_123/comparison_boxplot.png">
<img src="comparison_analysis_123/comparison_ecdf.png">
<img src="comparison_analysis_123/comparison_hist_kde.png">
<img src="comparison_analysis_123/comparison_stats_bar.png">
<img src="comparison_analysis_123/comparison_violin.png">

# 结论
开关3本身并不造成loss的普遍升降，而是改变了各开关条件下loss分布的std。所以可以解释为什么从均值上表现相对不好，但最好的实验几乎都是开关3打开。

# 疑点解答1
疑点：为何开关3普遍关闭造成shap value较小
<img src="shap_summary.png">

shap value描述的是：该开关采取开闭动作本身造成的loss边际值改变，而非总loss的绝对值改变。
从分布图中可看出，开关3打开会造成大量的loss急剧增高现象。在此情况下，其他开关不变，仅开关3关闭，就可能造成极其明显的loss下降。但从总loss值看，可能关闭后的loss值并不会很低，只是相对之前的严重病态loss值下降较多。

# 疑点解答2
疑点：为何右下角为蓝点
<img src="switch13.png">

仍是研究shap值的问题。纵轴为：开关1的shap值。对于所有蓝点，开关1的开闭能够有稳定的作用，表现为两簇蓝点差别较大（开关1开不开对loss造成的边际影响较大）。对于所有红点，由于rms norm开关打开，loss值的分布开始不稳定，开关1作为loss值减小的最显著开关，其减弱loss的作用收到了严重的噪音影响，不再稳定。

有一些能够佐证的结果：
<img src="interaction_analysis/use_rms_norm_vs_use_std_residual.png">
<img src="interaction_analysis/use_gated_ffn_vs_use_rms_norm.png">
<img src="interaction_analysis/use_rms_norm_vs_use_single_message_attn.png">
<img src="interaction_analysis/use_ffn_after_attn_vs_use_rms_norm.png">
<img src="interaction_analysis/untie_layerwise_weights_vs_use_rms_norm.png">
<img src="interaction_analysis/untie_ffn_weights_vs_use_rms_norm.png">
<img src="interaction_analysis/untie_attn_weights_vs_use_rms_norm.png">
<img src="interaction_analysis/no_vo_rope_vs_use_rms_norm.png">
<img src="interaction_analysis/no_mask_diagonal_vs_use_rms_norm.png">

可以看出，在开关本身对loss边际影响不大的情况下，开关3的开闭使方差明显增加。在开关1的案例中也有所体现，并且具体体现为开关1的边际作用发生了噪音影响。

# 疑点解答3
疑点：前五个开关中，其他开关的明显作用，体现在std大，还是单纯能对loss造成减少？
<img src="shap_summary.png">


### 开关12
<img src="comparison_analysis_12/comparison_stats_bar.png">
<img src="comparison_analysis_12/comparison_violin.png">

### 开关13
<img src="comparison_analysis_13/comparison_stats_bar.png">
<img src="comparison_analysis_13/comparison_violin.png">

### 开关14
<img src="comparison_analysis_14/comparison_stats_bar.png">
<img src="comparison_analysis_14/comparison_violin.png">

### 开关15
<img src="comparison_analysis_15/comparison_stats_bar.png">
<img src="comparison_analysis_15/comparison_violin.png">

### 开关1
<img src="comparison_analysis_1/comparison_stats_bar.png">
<img src="comparison_analysis_1/comparison_violin.png">

### 开关2
<img src="comparison_analysis_2/comparison_stats_bar.png">
<img src="comparison_analysis_2/comparison_violin.png">

### 开关3
<img src="comparison_analysis_3/comparison_stats_bar.png">
<img src="comparison_analysis_3/comparison_violin.png">

### 开关4
<img src="comparison_analysis_4/comparison_stats_bar.png">
<img src="comparison_analysis_4/comparison_violin.png">

### 开关5
<img src="comparison_analysis_5/comparison_stats_bar.png">
<img src="comparison_analysis_5/comparison_violin.png">