import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from scipy.stats import wasserstein_distance
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 加载并精确对齐 4 份 CSV 数据
# ==========================================
# 替换为你的真实路径目录
exp_dir = "/home/yxy05/code/research_tse/examples/audio/librimix/exp/TSE_BSRNN_SPK_EMB_Confusion"

csv_prior_cos = os.path.join(
    exp_dir, "target_confusion_data.csv")  # 包含 cos(e1, s1), cos(e1, s2)
csv_prior_dist = os.path.join(
    exp_dir, "prior_l2_distance.csv")  # 包含 Dist_e1_s1, Dist_e1_s2
csv_post_cos = os.path.join(exp_dir, "usef_target_confusion_data.csv"
                            )  # 包含 cos(F_post, F_tgt), cos(F_post, F_int)
csv_post_oracle = os.path.join(
    exp_dir, "master_usef_oracle_dataframe.csv"
)  # 包含 Dist_Tgt, Dist_Int, Real_SISNRi, Oracle_SISNRi

print("🔄 正在加载并对齐大一统数据 (2x2 Ablation Matrix)...")
df_prior_c = pd.read_csv(csv_prior_cos)[[
    'Utterance', 'cos(e1, s1)', 'cos(e1, s2)'
]]
df_prior_d = pd.read_csv(csv_prior_dist)[[
    'Utterance', 'Dist_e1_s1', 'Dist_e1_s2'
]]
df_post_c = pd.read_csv(csv_post_cos)[[
    'Utterance', 'cos(F_post, F_tgt)', 'cos(F_post, F_int)'
]]
df_post_d = pd.read_csv(csv_post_oracle)[[
    'Utterance', 'Dist_Tgt', 'Dist_Int', 'Real_SISNRi', 'Oracle_SISNRi'
]]

# 连续 Merge，确保每一行样本在四个空间都有绝对对齐的数据
df = pd.merge(df_prior_c, df_prior_d, on='Utterance', how='inner')
df = pd.merge(df, df_post_c, on='Utterance', how='inner')
df = pd.merge(df, df_post_d, on='Utterance', how='inner')
print(f"✅ 数据对齐完成！有效测试样本总数: {len(df)}")


# ==========================================
# 2. 严苛的物理状态分类 (Oracle 相对潜力法)
# ==========================================
def categorize_true_state(row):
    real = row['Real_SISNRi']
    oracle = row['Oracle_SISNRi']
    delta = oracle - real
    delta_threshold = 2.0

    if oracle < 0.0:
        return '3. Inherent Failure (Dead)'
    elif delta >= delta_threshold:
        return '2. Recoverable Confusion (Cue Fail)'
    elif real > 5.0 and delta < delta_threshold:
        return '1. Safe (Reached Potential)'
    else:
        return '4. Marginal / Others'


df['True_State'] = df.apply(categorize_true_state, axis=1)

states_to_plot = [
    '1. Safe (Reached Potential)', '2. Recoverable Confusion (Cue Fail)',
    '3. Inherent Failure (Dead)'
]
palette_dict = {
    '1. Safe (Reached Potential)': '#2ca02c',
    '2. Recoverable Confusion (Cue Fail)': '#ff7f0e',
    '3. Inherent Failure (Dead)': '#d62728',
    '4. Marginal / Others': '#7f7f7f'
}


# ==========================================
# 3. 统计学指标量化分析
# ==========================================
def quantify_separability(df, feature_x, feature_y, is_distance=False):
    target_states = [
        '1. Safe (Reached Potential)', '2. Recoverable Confusion (Cue Fail)'
    ]
    subset = df[df['True_State'].isin(target_states)].copy()

    X = subset[[feature_x, feature_y]].values
    y = np.where(subset['True_State'] == '2. Recoverable Confusion (Cue Fail)',
                 1, 0)

    results = {}

    # Linear AUC
    clf = LogisticRegression(class_weight='balanced', random_state=42)
    clf.fit(X, y)
    y_pred_proba = clf.predict_proba(X)[:, 1]
    results['AUC_Score'] = roc_auc_score(y, y_pred_proba)

    # Silhouette Score
    if len(np.unique(y)) > 1:
        results['Silhouette'] = silhouette_score(X, y)
    else:
        results['Silhouette'] = np.nan

    # Wasserstein Dist
    if is_distance:
        proj_safe = X[y == 0][:, 1] - X[y == 0][:, 0]
        proj_fatal = X[y == 1][:, 1] - X[y == 1][:, 0]
    else:
        proj_safe = X[y == 0][:, 0] - X[y == 0][:, 1]
        proj_fatal = X[y == 1][:, 0] - X[y == 1][:, 1]
    results['Wasserstein_Dist'] = wasserstein_distance(proj_safe, proj_fatal)

    return results


print("\n🚀 正在计算核心相关性指标 (剔除死局样本)...")
m_prior_cos = quantify_separability(df,
                                    'cos(e1, s1)',
                                    'cos(e1, s2)',
                                    is_distance=False)
m_prior_l2 = quantify_separability(df,
                                   'Dist_e1_s1',
                                   'Dist_e1_s2',
                                   is_distance=True)
m_post_cos = quantify_separability(df,
                                   'cos(F_post, F_tgt)',
                                   'cos(F_post, F_int)',
                                   is_distance=False)
m_post_l2 = quantify_separability(df, 'Dist_Tgt', 'Dist_Int', is_distance=True)

print(
    f"\n{'Metric':<20} | {'Prior (Cos)':<13} | {'Prior (L2)':<13} | {'Posterior (Cos)':<16} | {'Posterior (L2)':<16}"
)
print("-" * 88)


def print_row(metric_name, key):
    print(
        f"{metric_name:<20} | {m_prior_cos[key]:<13.4f} | {m_prior_l2[key]:<13.4f} | {m_post_cos[key]:<16.4f} | {m_post_l2[key]:<16.4f}"
    )


print_row("Linear AUC (↑)", "AUC_Score")
print_row("Silhouette (↑)", "Silhouette")
print_row("Wasserstein (↑)", "Wasserstein_Dist")
print("-" * 88)

# ==========================================
# 4. 绘图引擎 (一键出四图)
# ==========================================
sns.set_theme(style="whitegrid", font_scale=1.2)


def plot_1x3_space(df, x_col, y_col, is_distance, save_name, title_prefix):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    if is_distance:
        limit_min = 0.0
        limit_max = max(df[x_col].max(), df[y_col].max()) * 1.05
    else:
        limit_min = min(df[x_col].min(), df[y_col].min()) - 0.1
        limit_max = max(df[x_col].max(), df[y_col].max()) + 0.1

    for i, state in enumerate(states_to_plot):
        ax = axes[i]
        subset = df[df["True_State"] == state]

        color = palette_dict[state]
        marker = 'o' if i == 0 else ('*' if i == 1 else 'X')

        if len(subset) > 5:
            sns.kdeplot(data=subset,
                        x=x_col,
                        y=y_col,
                        levels=5,
                        color=color,
                        alpha=0.2,
                        fill=True,
                        ax=ax)

        sns.scatterplot(data=subset,
                        x=x_col,
                        y=y_col,
                        color=color,
                        marker=marker,
                        s=120 if i == 0 else 180,
                        alpha=0.8,
                        edgecolor="white" if i == 0 else None,
                        ax=ax)

        ax.plot([limit_min, limit_max], [limit_min, limit_max],
                color='#c0392b',
                linestyle='--',
                linewidth=2)
        ax.set_xlim(limit_min, limit_max)
        ax.set_ylim(limit_min, limit_max)

        title_text = f"{state.split('(')[0].strip()}\n(N={len(subset)})"
        ax.set_title(title_text, fontweight="bold", fontsize=15)

        ax.set_xlabel(f"{title_prefix} to Target", fontweight="bold")
        if i == 0:
            ax.set_ylabel(f"{title_prefix} to Interferer", fontweight="bold")

    plt.suptitle(f"{title_prefix} Space Decoupled by Rescue Potential",
                 fontweight="bold",
                 fontsize=18,
                 y=1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(exp_dir, save_name), dpi=300)
    plt.close(fig)
    print(f"🎨 已生成图像: {save_name}")


print("\n🚀 正在生成 2x2 对比诊断图...")
plot_1x3_space(df, 'cos(e1, s1)', 'cos(e1, s2)', False, "plot_1_Prior_Cos.png",
               "Prior Cosine")
plot_1x3_space(df, 'Dist_e1_s1', 'Dist_e1_s2', True, "plot_2_Prior_L2.png",
               "Prior L2 Distance")
plot_1x3_space(df, 'cos(F_post, F_tgt)', 'cos(F_post, F_int)', False,
               "plot_3_Post_Cos.png", "Posterior Cosine")
plot_1x3_space(df, 'Dist_Tgt', 'Dist_Int', True, "plot_4_Post_L2.png",
               "Posterior L2 Distance")

print("\n🎉 大一统数据整合与画图执行完毕！现在你的论据坚不可摧！")
