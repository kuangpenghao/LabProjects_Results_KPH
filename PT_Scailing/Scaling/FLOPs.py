import matplotlib.pyplot as plt
import numpy as np

# 数据：横坐标为序列长度（或相关维度），纵坐标为 FLOPs
x = np.array([256, 512, 1024, 1536, 2560,3840])
y = np.array([92211676534800380, 197701643091836930, 449513314892906500, 755767304943304700, 1531602239792087000,2355026383426084400])

def loglog_fit_and_plot(ax, x, y, color, label_prefix, marker):
    """在 log-log 坐标系中绘制散点并叠加幂律拟合 y = C * x^alpha"""
    ax.loglog(x, y, marker=marker, linestyle='None', color=color, label=label_prefix)
    
    if len(x) > 1:
        log_x = np.log(x)
        log_y = np.log(y)
        coeffs = np.polyfit(log_x, log_y, 1)
        alpha = coeffs[0]
        C = np.exp(coeffs[1])
        
        # 生成拟合曲线的数据点
        x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
        y_fit = C * x_fit**alpha
        
        # 计算 R^2
        y_pred = C * x**alpha
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - (ss_res / ss_tot)
        
        # 将拟合曲线表达式中的 C 使用科学计数法
        label_fit = f'Fit: $y = {C:.2e} \cdot x^{{{alpha:.3f}}}$, $R^2={r2:.4f}$'
        ax.loglog(x_fit, y_fit, '--', color=color, alpha=0.8, label=label_fit)

# 设置绘图样式
plt.rcParams.update({'font.size': 10})
fig, ax = plt.subplots(figsize=(8, 6))

loglog_fit_and_plot(ax, x, y, 'C0', 'FLOPs Data', 'o')

ax.set_xlabel('dim_z (log scale)')
ax.set_ylabel('FLOPs (log scale)')
ax.set_title('FLOPs vs dim_z (Log-Log Scale)')

# 标注横轴具体数值
ax.set_xticks(x)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

ax.grid(True, which="both", linestyle='--', alpha=0.6)
ax.legend()

plt.tight_layout()
# 保存图像到新建的 FLOPs 文件夹
plt.savefig(r'd:\_SHTU_\TKW_Lab\LabProjects_Results_KPH\PT_Scailing\FLOPs\FLOPs_scaling.png', dpi=300)
plt.show()
