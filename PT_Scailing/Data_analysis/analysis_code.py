# -*- coding: utf-8 -*-
"""
调参结果分析模板
输入：调参结果.xlsx（包含10个布尔开关列 + eval/loss, eval/accuracy）
输出：特征重要性、交互效应、推荐配置

新增功能：
1. analyze_fixed_switches(df, switch_list, value_list, out_dir, loss_col)
   - 固定指定开关的取值，分析满足条件的实验记录的 loss 分布情况
   - 生成箱线图、小提琴图、直方图+KDE、ECDF 等可视化
   - 输出统计摘要（均值、方差、极值等）为 Excel 文件

2. compare_fixed_switches(df, conditions_list, out_dir, loss_col)
   - 对比多组固定开关条件下的 loss 分布情况
   - 生成联合的比较图表：并排箱线图/小提琴图、重叠直方图+KDE、ECDF对比、统计量柱状图
   - 支持多条件同时比较分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
import itertools
import os


def load_data(path="D:\_SHTU_\TKW_Lab\LabProjects_Results_KPH\PT_Scailing\Data_analysis\sweep.xlsx"):
    df = pd.read_excel(path)

    # 自动识别10个开关列（根据你提供的列名）
    switch_cols = [
        'no_mask_diagonal',
        'no_vo_rope',
        'untie_attn_weights',
        'untie_ffn_weights',
        'untie_layerwise_weights',
        'use_ffn_after_attn',
        'use_gated_ffn',
        'use_rms_norm',
        'use_single_message_attn',
        'use_std_residual'
    ]

    missing = set(switch_cols) - set(df.columns)
    if missing:
        raise ValueError(f"缺失开关列: {missing}")

    df_switch = df[switch_cols].astype(int)
    y_loss = df['eval/loss']
    y_acc = df['eval/accuracy']

    print(f"✅ 成功加载 {len(df)} 条实验记录")
    return df, switch_cols, df_switch, y_loss, y_acc


def train_xgb_and_shap(X, y, n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42):
    print("\n🤖 正在训练 XGBoost 模型并计算 SHAP 值...")
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文（可选）
    plt.rcParams['axes.unicode_minus'] = False

    return model, explainer, shap_values


def plot_shap_summary(shap_values, X):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("SHAP 特征重要性（对 eval/loss 的影响）")
    plt.tight_layout()
    plt.savefig("shap_importance.png", dpi=150)
    plt.show()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP 汇总图（红=开关开，蓝=开关关）")
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150)
    plt.show()


def top5_recommendation(df, switch_cols):
    # 计算并保存 Top-5 到 Excel（不直接输出到终端）
    top5 = df.nlargest(390, 'eval/accuracy')[switch_cols + ['eval/accuracy', 'eval/loss', 'Name']]
    #top5.to_excel("top5_recommendation.xlsx", index=False)

    # 保存 Top-5 中各开关开启比例
    top5_switch_mean = top5[switch_cols].mean().sort_values(ascending=False)
    top5_switch_mean.to_frame(name='fraction_on').to_excel("top390_switch_fraction.xlsx")

    return top5


def compute_shap_interactions(explainer, X, switch_cols):
    print("\n🔄 正在计算 SHAP 交互值（可能较慢）...")
    shap_interaction = explainer.shap_interaction_values(X)
    interaction_matrix = np.abs(shap_interaction).mean(axis=0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(interaction_matrix, xticklabels=switch_cols, yticklabels=switch_cols, cmap="Blues")
    plt.title("SHAP 交互强度（绝对值均值）")
    plt.tight_layout()
    plt.savefig("shap_interactions.png", dpi=150)
    plt.show()

    return shap_interaction, interaction_matrix


def pairwise_scan(df, X, shap_values, shap_interaction, switch_cols, out_dir="interaction_analysis"):
    print("\n🔍 正在执行全局交互扫描（每开关 vs 其余9个）...")
    os.makedirs(out_dir, exist_ok=True)

    interaction_summary = []

    for main_feat, other_feat in itertools.combinations(switch_cols, 2):
        group = df.groupby([main_feat, other_feat])['eval/loss'].agg(['mean', 'count'])
        if group.empty:
            continue

        best_combo = group.loc[group['mean'].idxmin()]
        best_val = best_combo['mean']
        best_count = int(best_combo['count'])
        best_config = group['mean'].idxmin()  # (switch1_val, switch2_val)

        try:
            idx1 = X.columns.get_loc(main_feat)
            idx2 = X.columns.get_loc(other_feat)
            interaction_strength = np.abs(shap_interaction[:, idx1, idx2]).mean()
        except Exception:
            interaction_strength = np.nan

        plt.figure(figsize=(6, 4))
        shap.dependence_plot(
            main_feat,
            shap_values,
            X,
            interaction_index=other_feat,
            show=False,
            x_jitter=0.3,
            alpha=0.7
        )
        plt.title(f"{main_feat} vs {other_feat}")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{main_feat}_vs_{other_feat}.png", dpi=120)
        plt.close()

        interaction_summary.append({
            'switch1': main_feat,
            'switch2': other_feat,
            'best_loss': best_val,
            'best_count': best_count,
            'switch1_val': best_config[0],
            'switch2_val': best_config[1],
            'interaction_strength': interaction_strength
        })

    summary_df = pd.DataFrame(interaction_summary)
    summary_df = summary_df.sort_values('best_loss')
    summary_df.to_excel("interaction_summary_pairs.xlsx", index=False)

    # 不在终端打印表格，直接返回 DataFrame（文件已保存）
    print(f"✅ 全局交互扫描完成，摘要已保存为 interaction_summary_pairs.xlsx（共 {len(interaction_summary)} 条）")
    return summary_df


def three_way_scan(df, X, shap_interaction, switch_cols, out_dir="interaction_analysis"):
    print("\n🔎 正在执行三元组合扫描（每 3 个开关一组）...")
    interaction_summary_three = []

    for trio in itertools.combinations(switch_cols, 3):
        a, b, c = trio
        group3 = df.groupby([a, b, c])['eval/loss'].agg(['mean', 'count'])
        if group3.empty:
            continue

        best_combo3 = group3.loc[group3['mean'].idxmin()]
        best_val3 = best_combo3['mean']
        best_count3 = int(best_combo3['count'])
        best_config3 = group3['mean'].idxmin()  # (a_val, b_val, c_val)

        try:
            i1 = X.columns.get_loc(a)
            i2 = X.columns.get_loc(b)
            i3 = X.columns.get_loc(c)
            pair12 = np.abs(shap_interaction[:, i1, i2]).mean()
            pair13 = np.abs(shap_interaction[:, i1, i3]).mean()
            pair23 = np.abs(shap_interaction[:, i2, i3]).mean()
            tri_interaction_strength = np.mean([pair12, pair13, pair23])
        except Exception:
            tri_interaction_strength = np.nan

        try:
            df_plot = group3.reset_index()
            df_plot['combo'] = df_plot.apply(lambda r: f"{int(r[a])}{int(r[b])}{int(r[c])}", axis=1)
            df_plot = df_plot.sort_values('mean')

            plt.figure(figsize=(8, 4))
            sns.barplot(data=df_plot, x='combo', y='mean', palette='viridis')
            plt.xlabel(f"组合({a},{b},{c}) 设置 (a b c)")
            plt.ylabel('mean eval/loss')
            plt.title(f"三元组合 Loss: {a} {b} {c}")
            plt.tight_layout()
            plt.savefig(f"{out_dir}/three_{a}_{b}_{c}.png", dpi=140)
            plt.close()
        except Exception:
            pass

        interaction_summary_three.append({
            'switch1': a,
            'switch2': b,
            'switch3': c,
            'best_loss': best_val3,
            'best_count': best_count3,
            'switch1_val': best_config3[0],
            'switch2_val': best_config3[1],
            'switch3_val': best_config3[2],
            'interaction_strength': tri_interaction_strength
        })

    if interaction_summary_three:
        summary3_df = pd.DataFrame(interaction_summary_three)
        summary3_df = summary3_df.sort_values('best_loss')
        summary3_df.to_excel("interaction_summary_three.xlsx", index=False)
        print(f"✅ 三元组合扫描完成，摘要已保存为 interaction_summary_three.xlsx（共 {len(interaction_summary_three)} 条）")
        return summary3_df
    else:
        print("⚠️ 未生成任何三元组合摘要（数据可能不足）")
        return pd.DataFrame()


def compute_three_way_model_interactions(model, X, switch_cols, sample_size=200, out_file="three_way_model_interactions.xlsx"):
    """
    计算基于模型预测的三元交互强度（对每个三元组，使用有限差分在样本子集上计算三阶交互量）

    Returns a DataFrame with columns: switch1, switch2, switch3, model_interaction_strength
    """
    print("\n🔢 计算基于模型的三元交互强度（可能较慢）...")
    n_samples = len(X)
    sample_size = min(sample_size, n_samples)
    rng = np.random.default_rng(42)

    results = []
    combos = list(itertools.product([0, 1], repeat=3))
    coeffs = np.array([(-1) ** (3 - sum(bits)) for bits in combos])

    # 预转换 X 为 numpy array 以加速复制和修改
    X_vals = X.values
    for trio in itertools.combinations(switch_cols, 3):
        a, b, c = trio
        i1 = X.columns.get_loc(a)
        i2 = X.columns.get_loc(b)
        i3 = X.columns.get_loc(c)

        # 随机抽样索引
        if n_samples <= sample_size:
            sample_idx = np.arange(n_samples)
        else:
            sample_idx = rng.choice(n_samples, size=sample_size, replace=False)

        tri_effects = []
        try:
            for s_idx in sample_idx:
                base = X_vals[s_idx].reshape(1, -1)
                variants = np.repeat(base, 8, axis=0).copy()
                for k, bits in enumerate(combos):
                    variants[k, i1] = bits[0]
                    variants[k, i2] = bits[1]
                    variants[k, i3] = bits[2]

                preds = model.predict(variants)
                tri_val = np.sum(coeffs * preds) / 8.0
                tri_effects.append(tri_val)

            model_strength = float(np.mean(np.abs(tri_effects)))
        except Exception:
            model_strength = np.nan

        results.append({
            'switch1': a,
            'switch2': b,
            'switch3': c,
            'model_interaction_strength': model_strength
        })

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values('model_interaction_strength', ascending=False)
        res_df.to_excel(out_file, index=False)
        print(f"✅ 三元模型交互强度计算完成，结果已保存为 {out_file}（共 {len(res_df)} 条）")
    else:
        print("⚠️ 未计算到任何三元模型交互强度（数据可能不足）")

    return res_df


def analyze_fixed_switches(df, switch_list, value_list, out_dir="fixed_switch_analysis", loss_col='eval/loss'):
    """
    给定若干开关名和对应的取值（0/1），过滤出满足这些固定取值的实验记录，
    并对这些记录的 `loss_col` 列做分布分析和可视化：箱线图、直方图+KDE、小提琴图、ECDF，
    同时输出统计量（count, mean, std, min, 25%, 50%, 75%, max）为 Excel 文件。

    参数:
    - df: 原始 DataFrame（包含开关列与 `loss_col`）
    - switch_list: 开关名称列表，例如 ['use_gated_ffn', 'use_rms_norm']
    - value_list: 与之等长的取值列表，例如 [1, 0]
    - out_dir: 输出目录（图片与统计表保存到此处）
    - loss_col: 要分析的损失列名（默认 'eval/loss'）

    返回:
    - stats_df: 包含统计量的 DataFrame
    - subset_df: 过滤后的子集 DataFrame
    """
    import math

    os.makedirs(out_dir, exist_ok=True)

    # 校验输入
    if len(switch_list) != len(value_list):
        raise ValueError("switch_list 与 value_list 必须等长")

    for sw in switch_list:
        if sw not in df.columns:
            raise ValueError(f"开关列不存在: {sw}")

    # 依次过滤
    mask = np.ones(len(df), dtype=bool)
    for sw, val in zip(switch_list, value_list):
        mask &= (df[sw] == int(val))

    subset = df[mask].copy()
    n = len(subset)
    if n == 0:
        print(f"⚠️ 无符合条件的记录: {list(zip(switch_list, value_list))}")
        return pd.DataFrame(), subset

    # 统计量
    stats = subset[loss_col].describe()
    stats = stats.rename({"25%": "q1", "50%": "median", "75%": "q3"})
    stats_dict = {
        'count': int(stats['count']),
        'mean': float(stats['mean']),
        'std': float(stats['std']) if not np.isnan(stats['std']) else np.nan,
        'min': float(stats['min']),
        'q1': float(stats['q1']),
        'median': float(stats['median']),
        'q3': float(stats['q3']),
        'max': float(stats['max'])
    }
    stats_df = pd.DataFrame([stats_dict])
    stats_df['condition'] = ",".join([f"{s}={int(v)}" for s, v in zip(switch_list, value_list)])
    stats_df = stats_df.set_index('condition')
    stats_df.to_excel(os.path.join(out_dir, "fixed_switch_stats.xlsx"))

    # 几种可视化
    vals = subset[loss_col].dropna()

    # 1) 箱线图 + 小提琴图（并排）
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(x=vals, color='skyblue')
    plt.title('箱线图')
    plt.xlabel(loss_col)

    plt.subplot(1, 2, 2)
    sns.violinplot(x=vals, color='lightgreen')
    plt.title('小提琴图')
    plt.xlabel(loss_col)
    plt.suptitle(f"固定: {';'.join([f'{s}={int(v)}' for s,v in zip(switch_list,value_list)])} (n={n})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(out_dir, "box_violin.png"), dpi=150)
    plt.close()

    # 2) 直方图 + KDE
    plt.figure(figsize=(8, 5))
    sns.histplot(vals, kde=True, stat='density', color='cornflowerblue', bins=30)
    plt.axvline(stats_dict['mean'], color='red', linestyle='--', label=f"mean={stats_dict['mean']:.4f}")
    plt.axvline(stats_dict['median'], color='orange', linestyle='-.', label=f"median={stats_dict['median']:.4f}")
    plt.legend()
    plt.title('直方图 + KDE')
    plt.xlabel(loss_col)
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_kde.png"), dpi=150)
    plt.close()

    # 3) 单独的 KDE 曲线
    plt.figure(figsize=(8, 4))
    try:
        sns.kdeplot(vals, fill=True, color='purple')
        plt.title('概率密度估计 (KDE)')
        plt.xlabel(loss_col)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "kde.png"), dpi=150)
    except Exception:
        # KDE 可能在样本太少时失败，降级为 histogram
        plt.clf()
        sns.histplot(vals, bins=30, color='gray')
        plt.title('Histogram (fallback for KDE)')
        plt.savefig(os.path.join(out_dir, "kde_fallback_hist.png"), dpi=150)
    plt.close()

    # 4) ECDF
    plt.figure(figsize=(8, 4))
    try:
        sns.ecdfplot(vals, color='teal')
        plt.title('经验累积分布函数 (ECDF)')
        plt.xlabel(loss_col)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ecdf.png"), dpi=150)
    except Exception:
        pass
    plt.close()

    # 保存子集为 csv 以便进一步分析
    subset.to_csv(os.path.join(out_dir, "filtered_subset.csv"), index=False)

    print(f"✅ 已对条件 {list(zip(switch_list, value_list))} 的 {n} 条记录完成分析，输出保存在 {out_dir}/")
    return stats_df, subset


def compare_fixed_switches(df, conditions_list, out_dir="compare_fixed_switches", loss_col='eval/loss'):
    """
    对比多组固定开关条件下的 loss 分布情况，生成联合的比较图表。
    
    参数:
    - df: 原始 DataFrame（包含开关列与 `loss_col`）
    - conditions_list: 条件列表，每个元素是一个字典，包含：
        {
            'switch_list': ['开关1', '开关2', ...],
            'value_list': [取值1, 取值2, ...],
            'label': '条件标签'  # 可选，用于图例显示
        }
    - out_dir: 输出目录（图片与统计表保存到此处）
    - loss_col: 要分析的损失列名（默认 'eval/loss'）
    
    返回:
    - combined_stats_df: 包含所有条件统计量的对比 DataFrame
    - combined_data: 包含所有条件数据的 DataFrame（用于进一步分析）
    """
    os.makedirs(out_dir, exist_ok=True)
    
    all_stats = []
    all_data = []
    
    # 为每个条件提取数据和统计量
    for i, condition in enumerate(conditions_list):
        switch_list = condition['switch_list']
        value_list = condition['value_list']
        label = condition.get('label', f"条件{i+1}: " + ",".join([f"{s}={v}" for s, v in zip(switch_list, value_list)]))
        
        # 过滤数据
        mask = np.ones(len(df), dtype=bool)
        for sw, val in zip(switch_list, value_list):
            if sw not in df.columns:
                raise ValueError(f"开关列不存在: {sw}")
            mask &= (df[sw] == int(val))
        
        subset = df[mask].copy()
        n = len(subset)
        
        if n == 0:
            print(f"⚠️ 条件 {label} 无符合条件的记录")
            continue
            
        # 统计量
        vals = subset[loss_col].dropna()
        stats = vals.describe()
        stats_dict = {
            'condition': label,
            'count': int(stats['count']),
            'mean': float(stats['mean']),
            'std': float(stats['std']) if not np.isnan(stats['std']) else np.nan,
            'min': float(stats['min']),
            'q1': float(stats['25%']),
            'median': float(stats['50%']),
            'q3': float(stats['75%']),
            'max': float(stats['max'])
        }
        all_stats.append(stats_dict)
        
        # 为联合分析准备数据
        subset_for_plot = subset[[loss_col]].copy()
        subset_for_plot['condition'] = label
        subset_for_plot['condition_idx'] = i
        all_data.append(subset_for_plot)
    
    if not all_stats:
        print("⚠️ 没有任何条件产生有效数据")
        return pd.DataFrame(), pd.DataFrame()
    
    # 合并数据
    combined_stats_df = pd.DataFrame(all_stats)
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # 保存统计对比表
    combined_stats_df.to_excel(os.path.join(out_dir, "comparison_stats.xlsx"), index=False)
    
    # 1) 并排箱线图比较
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=combined_data, x='condition', y=loss_col, palette='Set2')
    plt.title('多条件 Loss 分布对比 (箱线图)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_boxplot.png"), dpi=150)
    plt.close()
    
    # 2) 并排小提琴图比较
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=combined_data, x='condition', y=loss_col, palette='Set3')
    plt.title('多条件 Loss 分布对比 (小提琴图)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_violin.png"), dpi=150)
    plt.close()
    
    # 3) 重叠的直方图+KDE
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, len(conditions_list)))
    for i, (_, group) in enumerate(combined_data.groupby('condition')):
        vals = group[loss_col].dropna()
        plt.hist(vals, bins=30, alpha=0.6, label=group['condition'].iloc[0], 
                color=colors[i], density=True)
        try:
            sns.kdeplot(vals, color=colors[i], linewidth=2)
        except:
            pass
    plt.xlabel(loss_col)
    plt.ylabel('Density')
    plt.title('多条件 Loss 分布对比 (重叠直方图+KDE)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_hist_kde.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4) ECDF 对比
    plt.figure(figsize=(10, 6))
    for i, (_, group) in enumerate(combined_data.groupby('condition')):
        vals = group[loss_col].dropna()
        try:
            sns.ecdfplot(vals, label=group['condition'].iloc[0], color=colors[i], linewidth=2)
        except:
            pass
    plt.xlabel(loss_col)
    plt.ylabel('累积概率')
    plt.title('多条件 Loss 分布对比 (ECDF)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_ecdf.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5) 统计量对比柱状图
    plt.figure(figsize=(15, 5))
    metrics = ['mean', 'std', 'median']
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        plt.bar(range(len(all_stats)), [s[metric] for s in all_stats], color=colors[:len(all_stats)])
        plt.title(f'{metric.capitalize()} 对比')
        plt.xticks(range(len(all_stats)), [s['condition'] for s in all_stats], rotation=45, ha='right')
        plt.ylabel(metric)
    plt.suptitle('关键统计量对比')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_stats_bar.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 保存合并数据为 CSV
    combined_data.to_csv(os.path.join(out_dir, "comparison_data.csv"), index=False)
    
    print(f"✅ 已完成 {len(conditions_list)} 个条件的对比分析，输出保存在 {out_dir}/")
    print("📊 生成的对比图表：")
    print("  - comparison_boxplot.png: 箱线图对比")
    print("  - comparison_violin.png: 小提琴图对比") 
    print("  - comparison_hist_kde.png: 重叠直方图+KDE对比")
    print("  - comparison_ecdf.png: ECDF对比")
    print("  - comparison_stats_bar.png: 统计量柱状图对比")
    
    return combined_stats_df, combined_data


def main():
    df, switch_cols, df_switch, y_loss, y_acc = load_data()

    X = df_switch.copy()
    y = y_loss
    model, explainer, shap_values = train_xgb_and_shap(X, y)

    #plot_shap_summary(shap_values, X)
    #top5 = top5_recommendation(df, switch_cols)
    #shap_interaction, interaction_matrix = compute_shap_interactions(explainer, X, switch_cols)
    #three_model_df = compute_three_way_model_interactions(model, X, switch_cols, sample_size=200, out_file="three_way_model_interactions.xlsx")
    #summary2_df = pairwise_scan(df, X, shap_values, shap_interaction, switch_cols)
    #summary3_df = three_way_scan(df, X, shap_interaction, switch_cols)
    
    # 单独分析示例
    #stats_df, subset_df = analyze_fixed_switches(df, ['untie_attn_weights', 'use_rms_norm'], [1, 0], out_dir="fixed_10")
    #stats_df, subset_df = analyze_fixed_switches(df, ['untie_attn_weights', 'use_rms_norm'], [1, 1], out_dir="fixed_11")
    
    # 联合对比分析示例
    '''
    conditions = [
        {
            'switch_list': ['untie_attn_weights', 'use_rms_norm'],
            'value_list': [1, 0],
            'label': 'untie_attn=1, rms_norm=0'
        },
        {
            'switch_list': ['untie_attn_weights', 'use_rms_norm'],
            'value_list': [1, 1],
            'label': 'untie_attn=1, rms_norm=1'
        }
    ]
    compare_stats_df, compare_data = compare_fixed_switches(df, conditions, out_dir="comparison_analysis_13")

    conditions = [
        {
            'switch_list': ['untie_attn_weights', 'use_rms_norm', 'use_std_residual'],
            'value_list': [1, 0, 1],
            'label': 'untie_attn=1, rms_norm=0, std_residual=1'
        },
        {
            'switch_list': ['untie_attn_weights', 'use_rms_norm'],
            'value_list': [1, 1, 1],
            'label': 'untie_attn=1, rms_norm=1, std_residual=1'
        }
    ]
    compare_stats_df, compare_data = compare_fixed_switches(df, conditions, out_dir="comparison_analysis_123")
    '''

    conditions = [
        {
            'switch_list': ['untie_layerwise_weights'],
            'value_list': [1],
            'label': 'untie_layerwise_weights=1'
        },
        {
            'switch_list': ['untie_layerwise_weights'],
            'value_list': [0],
            'label': 'untie_layerwise_weights=0'
        }
    ]
    compare_stats_df, compare_data = compare_fixed_switches(df, conditions, out_dir="comparison_analysis_5")

    


if __name__ == "__main__":
    main()


