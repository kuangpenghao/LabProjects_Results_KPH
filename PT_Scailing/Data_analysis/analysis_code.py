# -*- coding: utf-8 -*-
"""
调参结果分析模板
输入：调参结果.xlsx（包含10个布尔开关列 + eval/loss, eval/accuracy）
输出：特征重要性、交互效应、推荐配置
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


def load_data(path="调参结果.xlsx"):
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
    top5 = df.nlargest(5, 'eval/accuracy')[switch_cols + ['eval/accuracy', 'eval/loss', 'Name']]
    top5.to_excel("top5_recommendation.xlsx", index=False)

    # 保存 Top-5 中各开关开启比例
    top5_switch_mean = top5[switch_cols].mean().sort_values(ascending=False)
    top5_switch_mean.to_frame(name='fraction_on').to_excel("top5_switch_fraction.xlsx")

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


def main():
    df, switch_cols, df_switch, y_loss, y_acc = load_data()

    X = df_switch.copy()
    y = y_loss

    model, explainer, shap_values = train_xgb_and_shap(X, y)
    plot_shap_summary(shap_values, X)
    top5 = top5_recommendation(df, switch_cols)
    shap_interaction, interaction_matrix = compute_shap_interactions(explainer, X, switch_cols)
    three_model_df = compute_three_way_model_interactions(model, X, switch_cols, sample_size=200, out_file="three_way_model_interactions.xlsx")
    summary2_df = pairwise_scan(df, X, shap_values, shap_interaction, switch_cols)
    summary3_df = three_way_scan(df, X, shap_interaction, switch_cols)


if __name__ == "__main__":
    main()


