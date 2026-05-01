import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from scipy.stats import wasserstein_distance
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 加载并对齐数据 (复用之前的对齐逻辑)
# ==========================================
csv_prior = "/home/yxy05/code/research_tse/examples/audio/librimix/exp/TSE_BSRNN_SPK_EMB_Confusion/target_confusion_data.csv"  # 包含 cos(e1, s1), cos(e1, s2)
csv_post_cos = "/home/yxy05/code/research_tse/examples/audio/librimix/exp/TSE_BSRNN_SPK_EMB_Confusion/usef_target_confusion_data.csv"  # 包含 cos(F_post, F_tgt), cos(F_post, F_int)
csv_oracle = "/home/yxy05/code/research_tse/examples/audio/librimix/exp/TSE_BSRNN_SPK_EMB_Confusion/master_usef_oracle_dataframe.csv"  # 包含 Dist_Tgt, Dist_Int, Real_SISNRi, Oracle_SISNRi

df_prior = pd.read_csv(csv_prior)[['Utterance', 'cos(e1, s1)', 'cos(e1, s2)']]
df_post_cos = pd.read_csv(csv_post_cos)[[
    'Utterance', 'cos(F_post, F_tgt)', 'cos(F_post, F_int)'
]]
df_oracle = pd.read_csv(csv_oracle)[[
    'Utterance', 'Dist_Tgt', 'Dist_Int', 'Real_SISNRi', 'Oracle_SISNRi'
]]

df = pd.merge(df_prior, df_post_cos, on='Utterance', how='inner')
df = pd.merge(df, df_oracle, on='Utterance', how='inner')


# ==========================================
# 2. 严苛的物理状态分类 (Oracle 潜力法)
# ==========================================
def categorize_true_state(row):
    real = row['Real_SISNRi']
    oracle = row['Oracle_SISNRi']
    delta = oracle - real
    delta_threshold = 5.0

    if oracle < 0.0:
        return '3. Inherent Failure (Dead)'
    elif delta >= delta_threshold:
        return '2. Recoverable Confusion (Cue Fail)'
    elif real > 5.0 and delta < delta_threshold:
        return '1. Safe (Reached Potential)'
    else:
        return '4. Marginal / Others'


df['True_State'] = df.apply(categorize_true_state, axis=1)


# ==========================================
# 3. 量化评估函数
# ==========================================
def quantify_separability(df, feature_x, feature_y, is_distance=False):
    """
    计算二维特征空间内的可分性指标。
    正样本(0): Safe；负样本(1): Recoverable Confusion
    """
    # 🎯 核心过滤：只对比真正的 Safe 和 纯粹的混淆样本
    target_states = [
        '1. Safe (Reached Potential)', '2. Recoverable Confusion (Cue Fail)'
    ]
    subset = df[df['True_State'].isin(target_states)].copy()

    X = subset[[feature_x, feature_y]].values
    y = np.where(subset['True_State'] == '2. Recoverable Confusion (Cue Fail)',
                 1, 0)

    results = {}

    # 1. Linear AUC (线性可分性)
    clf = LogisticRegression(class_weight='balanced', random_state=42)
    clf.fit(X, y)
    y_pred_proba = clf.predict_proba(X)[:, 1]
    results['AUC_Score'] = roc_auc_score(y, y_pred_proba)

    # 2. Silhouette Score (轮廓系数)
    if len(np.unique(y)) > 1:
        results['Silhouette'] = silhouette_score(X, y)
    else:
        results['Silhouette'] = np.nan

    # 3. Wasserstein Distance (投影分布差异)
    if is_distance:
        # Dist_Int - Dist_Tgt (越大越像目标)
        proj_safe = X[y == 0][:, 1] - X[y == 0][:, 0]
        proj_fatal = X[y == 1][:, 1] - X[y == 1][:, 0]
    else:
        # Cos_Tgt - Cos_Int (越大越像目标)
        proj_safe = X[y == 0][:, 0] - X[y == 0][:, 1]
        proj_fatal = X[y == 1][:, 0] - X[y == 1][:, 1]

    results['Wasserstein_Dist'] = wasserstein_distance(proj_safe, proj_fatal)

    return results


# ==========================================
# 4. 执行量化并打印学术表格
# ==========================================
print("🚀 正在计算核心相关性指标 (基于 Oracle 潜力标签)...")
metrics_prior = quantify_separability(df,
                                      'cos(e1, s1)',
                                      'cos(e1, s2)',
                                      is_distance=False)
metrics_post_cos = quantify_separability(df,
                                         'cos(F_post, F_tgt)',
                                         'cos(F_post, F_int)',
                                         is_distance=False)
metrics_post_dist = quantify_separability(df,
                                          'Dist_Tgt',
                                          'Dist_Int',
                                          is_distance=True)

print(
    f"\n{'Metric':<20} | {'Prior (Cos)':<15} | {'Posterior (Cos)':<17} | {'Posterior (Dist)':<17}"
)
print("-" * 75)


# 格式化输出函数
def print_row(metric_name, key):
    val1 = metrics_prior[key]
    val2 = metrics_post_cos[key]
    val3 = metrics_post_dist[key]
    print(f"{metric_name:<20} | {val1:<15.4f} | {val2:<17.4f} | {val3:<17.4f}")


print_row("Linear AUC (↑)", "AUC_Score")
print_row("Silhouette (↑)", "Silhouette")
print_row("Wasserstein Dist (↑)", "Wasserstein_Dist")
print("-" * 75)
