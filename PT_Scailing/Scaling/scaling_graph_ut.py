import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# 数据：PT 模型
pt_params = np.array([17072128, 35422976, 76056832, 121933568, 229415680,393259520])#
pt_loss = np.array([2.51745, 2.12855, 1.86028, 1.76296, 1.6409,1.55])#
pt_ppl = np.exp(pt_loss)

# 数据：UT 模型
bert_params = np.array([17537025,36748033,57665025,80296193,187823361,322638081])#
bert_loss = np.array([2.43, 2.08209, 1.90514, 1.80536, 1.6562,1.553])#
bert_ppl = np.exp(bert_loss)

# 设置图形大小和样式
plt.rcParams.update({'font.size': 12})


def linear_fit_and_plot(ax, x, y, color, label_prefix, marker):
    """在线性坐标系中绘制散点并叠加直线拟合 y = ax + b"""
    ax.plot(x, y, marker=marker, color=color, label=label_prefix)
    if len(x) > 1:
        coeffs = np.polyfit(x, y, 1)
        x_fit = np.linspace(x.min(), x.max(), 300)
        y_fit = np.polyval(coeffs, x_fit)
        # 计算 R² 值
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        sign = '+' if coeffs[1] >= 0 else '-'
        expr = f'{label_prefix} fit: y = {coeffs[0]:.2e}x {sign} {abs(coeffs[1]):.3f}, R²={r2:.4f}'
        ax.plot(x_fit, y_fit, '--', color=color, alpha=0.8, label=expr)


def loglog_fit_and_plot(ax, x, y, color, label_prefix, marker):
    """在 log-log 坐标系中绘制散点并叠加幂律拟合 y = C·x^α"""
    ax.loglog(x, y, marker=marker, color=color, label=label_prefix)
    if len(x) > 1:
        log_x = np.log(x)
        log_y = np.log(y)
        coeffs = np.polyfit(log_x, log_y, 1)
        alpha = coeffs[0]
        C = np.exp(coeffs[1])
        x_fit = np.linspace(x.min(), x.max(), 300)
        y_fit = C * x_fit ** alpha
        # 计算 R² 值
        y_pred = C * x ** alpha
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        expr = f'{label_prefix} fit: y = {C:.2e}·x^({alpha:.3f}), R²={r2:.4f}'
        ax.loglog(x_fit, y_fit, '--', color=color, alpha=0.8, label=expr)


def scatter_plot(ax, x, y, color, label_prefix, marker):
    """仅绘制散点，不进行拟合"""
    ax.plot(x, y, marker=marker, linestyle='-', color=color, label=label_prefix)

'''
# 图1: eval_loss vs parameters (linear scale)
fig, ax = plt.subplots(figsize=(8, 6))
scatter_plot(ax, pt_params, pt_loss, 'C0', 'PT', 'o')
scatter_plot(ax, bert_params, bert_loss, 'C1', 'BERT', 's')
ax.set_xlabel('Parameters')
ax.set_ylabel('Eval Loss')
ax.set_title('Eval Loss vs Parameters (Linear Scale)')
ax.legend(fontsize=10)
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M'))
ax.tick_params(axis='x', rotation=30)
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('fig1_linear_loss.png', dpi=300)
plt.show()


# 图2: eval_loss vs parameters (log-log scale)
fig, ax = plt.subplots(figsize=(8, 6))
loglog_fit_and_plot(ax, pt_params, pt_loss, 'C0', 'PT', 'o')
loglog_fit_and_plot(ax, bert_params, bert_loss, 'C1', 'BERT', 's')
ax.set_xlabel('Parameters (log scale)')
ax.set_ylabel('Eval Loss (log scale)')
ax.set_title('Eval Loss vs Parameters (Log-Log Scale)')
ax.legend(fontsize=10)
all_x_loss = np.sort(np.concatenate([pt_params, bert_params]))
ax.set_xticks(all_x_loss)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
ax.tick_params(axis='x', rotation=30)
ax.grid(True, which="both", linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('fig2_loglog_loss.png', dpi=300)
plt.show()

''''''
# 图3: perplexity vs parameters (linear scale)
fig, ax = plt.subplots(figsize=(8, 6))
scatter_plot(ax, pt_params, pt_ppl, 'C0', 'PT', 'o')
scatter_plot(ax, bert_params, bert_ppl, 'C1', 'BERT', 's')
ax.set_xlabel('Parameters')
ax.set_ylabel('Perplexity')
ax.set_title('Perplexity vs Parameters (Linear Scale)')
ax.legend(fontsize=10)
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M'))
ax.tick_params(axis='x', rotation=30)
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('fig3_linear_ppl.png', dpi=300)
plt.show()


# 图4: perplexity vs parameters (log-log scale)
fig, ax = plt.subplots(figsize=(8, 6))
loglog_fit_and_plot(ax, pt_params, pt_ppl, 'C0', 'PT', 'o')
loglog_fit_and_plot(ax, bert_params, bert_ppl, 'C1', 'BERT', 's')
ax.set_xlabel('Parameters (log scale)')
ax.set_ylabel('Perplexity (log scale)')
ax.set_title('Perplexity vs Parameters (Log-Log Scale)')
ax.legend(fontsize=10)
all_x_ppl = np.sort(np.concatenate([pt_params, bert_params]))
ax.set_xticks(all_x_ppl)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
ax.tick_params(axis='x', rotation=30)
ax.grid(True, which="both", linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('fig4_loglog_ppl.png', dpi=300)
plt.show()
'''

''''''
# 图5: eval_loss vs parameters (log-log scale, no fit line)
fig, ax = plt.subplots(figsize=(8, 6))
scatter_plot(ax, pt_params, pt_loss, 'C0', 'PT', 'o')
scatter_plot(ax, bert_params, bert_loss, 'C1', 'UT', 's')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Parameters (log scale)')
ax.set_ylabel('Eval Loss (log scale)')
ax.set_title('Eval Loss vs Parameters (Log-Log Scale)')
ax.legend(fontsize=10)
all_x_loss = np.sort(np.concatenate([pt_params, bert_params]))
ax.set_xticks(all_x_loss)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
ax.tick_params(axis='x', rotation=30)
ax.grid(True, which="both", linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(r'D:\_SHTU_\TKW_Lab\LabProjects_Results_KPH\PT_Scailing\PT_UT\fig5_loglog_loss_no_fit.png', dpi=300)
plt.show()

# 图6: perplexity vs parameters (log-log scale, no fit line)
fig, ax = plt.subplots(figsize=(8, 6))
scatter_plot(ax, pt_params, pt_ppl, 'C0', 'PT', 'o')
scatter_plot(ax, bert_params, bert_ppl, 'C1', 'UT', 's')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Parameters (log scale)')
ax.set_ylabel('Perplexity (log scale)')
ax.set_title('Perplexity vs Parameters (Log-Log Scale)')
ax.legend(fontsize=10)
all_x_ppl = np.sort(np.concatenate([pt_params, bert_params]))
ax.set_xticks(all_x_ppl)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
ax.tick_params(axis='x', rotation=30)
ax.grid(True, which="both", linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(r'D:\_SHTU_\TKW_Lab\LabProjects_Results_KPH\PT_Scailing\PT_UT\fig6_loglog_ppl_no_fit.png', dpi=300) #
plt.show()