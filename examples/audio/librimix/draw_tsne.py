import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import warnings

warnings.filterwarnings('ignore')

# ========================================================
# 1. 设定路径与参数
# ========================================================
exp_dir = "exp/TSE_BSRNN_SPK_EMB_Confusion"  # 请替换为实际路径
feat_base_dir = os.path.join(exp_dir, "exp1_tsne_features")
output_dir = os.path.join(exp_dir, "exp1_tsne_anchor_plots_v2")
os.makedirs(output_dir, exist_ok=True)

mix_folders = sorted(glob.glob(os.path.join(feat_base_dir, "mix_*")))
print(f"🔍 共找到 {len(mix_folders)} 个 Mixture，准备进行高维降维打击...\n")


def get_state(delta):
    if delta < 2.0: return "1. Safe (< 2dB)"
    elif delta <= 5.0: return "2. Marginal (2~5dB)"
    else: return "3. Failure (> 5dB)"


palette = {
    "1. Safe (< 2dB)": "#2ca02c",  # 绿色
    "2. Marginal (2~5dB)": "#ff7f0e",  # 橙色
    "3. Failure (> 5dB)": "#d62728"  # 红色
}

# ========================================================
# 2. 降维与绘图主循环
# ========================================================
for mix_dir in mix_folders:
    mix_name = os.path.basename(mix_dir)

    # --- A. 加载 50 个随机 Cue 的特征 ---
    try:
        prior_feats = np.load(os.path.join(mix_dir, "prior_features.npy"))
        pmap_feats = np.load(os.path.join(mix_dir, "pmap_features.npy"))
        post_feats = np.load(os.path.join(mix_dir, "post_concat_features.npy"))
        df_labels = pd.read_csv(os.path.join(mix_dir, "labels.csv"))

        # --- B. 加载 Oracle 上帝锚点 ---
        anchors = np.load(os.path.join(mix_dir, "oracle_anchors.npz"))
    except Exception as e:
        print(f"⚠️ 跳过 {mix_name}: 数据不全 ({e})")
        continue

    N = len(df_labels)
    if N < 10: continue

    df_labels['State'] = df_labels['Delta_SISNRi'].apply(get_state)

    # ========================================================
    # 🧠 核心操作 1：将 50 个样本与 2 个锚点拼接 (N = 52)
    # ========================================================
    # 顺序：[50个样本] + [1个Target锚点] + [1个Interferer锚点]
    prior_all = np.vstack([
        prior_feats, anchors['tgt_prior'].reshape(1, -1),
        anchors['int_prior'].reshape(1, -1)
    ])
    pmap_all = np.vstack([
        pmap_feats, anchors['tgt_pmap'].reshape(1, -1),
        anchors['int_pmap'].reshape(1, -1)
    ])
    post_all = np.vstack([
        post_feats, anchors['tgt_post'].reshape(1, -1),
        anchors['int_post'].reshape(1, -1)
    ])

    # 对 Prior 进行 L2 归一化 (保护 ECAPA-TDNN 的余弦流形)
    prior_all = normalize(prior_all, norm='l2', axis=1)

    # ========================================================
    # 🧠 核心操作 2：t-SNE 独立降维 (带锚点的全局视野)
    # ========================================================
    perp = min(12, N - 1)

    tsne_prior = TSNE(n_components=2,
                      perplexity=perp,
                      random_state=42,
                      init='pca',
                      learning_rate='auto')
    tsne_pmap = TSNE(n_components=2,
                     perplexity=perp,
                     random_state=42,
                     init='pca',
                     learning_rate='auto')
    tsne_post = TSNE(n_components=2,
                     perplexity=perp,
                     random_state=42,
                     init='pca',
                     learning_rate='auto')

    prior_2d = tsne_prior.fit_transform(prior_all)
    pmap_2d = tsne_pmap.fit_transform(pmap_all)
    post_2d = tsne_post.fit_transform(post_all)

    # 拆分出 50 个样本的坐标
    df_labels['Prior_X'], df_labels['Prior_Y'] = prior_2d[:N, 0], prior_2d[:N,
                                                                           1]
    df_labels['Pmap_X'], df_labels['Pmap_Y'] = pmap_2d[:N, 0], pmap_2d[:N, 1]
    df_labels['Post_X'], df_labels['Post_Y'] = post_2d[:N, 0], post_2d[:N, 1]

    # 拆分出 2 个锚点的坐标
    tgt_anchor_prior, int_anchor_prior = prior_2d[N], prior_2d[N + 1]
    tgt_anchor_pmap, int_anchor_pmap = pmap_2d[N], pmap_2d[N + 1]
    tgt_anchor_post, int_anchor_post = post_2d[N], post_2d[N + 1]

    # ========================================================
    # 🖼️ 绘图引擎 (1x3 演化图)
    # ========================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    spaces = [("1. Prior Space (Identity)", df_labels['Prior_X'],
               df_labels['Prior_Y'], tgt_anchor_prior, int_anchor_prior),
              ("2. Pmap Space (Energy Mask)", df_labels['Pmap_X'],
               df_labels['Pmap_Y'], tgt_anchor_pmap, int_anchor_pmap),
              ("3. Post-Concat Space (Final Decision)", df_labels['Post_X'],
               df_labels['Post_Y'], tgt_anchor_post, int_anchor_post)]

    for i, (title, x, y, tgt_anc, int_anc) in enumerate(spaces):
        ax = axes[i]

        # 1. 画 50 个提取样本
        sns.scatterplot(data=df_labels,
                        x=x,
                        y=y,
                        hue='State',
                        palette=palette,
                        s=120,
                        alpha=0.8,
                        edgecolor='white',
                        ax=ax,
                        legend=(i == 0))

        # 2. 画 Oracle Target 锚点 (巨大的绿色五角星)
        ax.scatter(tgt_anc[0],
                   tgt_anc[1],
                   marker='*',
                   s=600,
                   color='#2ecc71',
                   edgecolor='black',
                   linewidth=1.5,
                   label='Oracle Target' if i == 0 else "",
                   zorder=10)

        # 3. 画 Oracle Interferer 锚点 (巨大的红色叉叉)
        ax.scatter(int_anc[0],
                   int_anc[1],
                   marker='X',
                   s=400,
                   color='#c0392b',
                   edgecolor='black',
                   linewidth=1.5,
                   label='Oracle Interferer' if i == 0 else "",
                   zorder=10)

        # 隐藏坐标轴刻度 (符合顶会规范)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(title, fontweight="bold", fontsize=14)

    # 调整图例
    axes[0].legend(loc='upper right',
                   bbox_to_anchor=(1.0, -0.05),
                   ncol=2,
                   fontsize=11)

    plt.suptitle(
        f"Feature Assimilation Trajectory (Micro-Ablation): {mix_name}",
        fontweight="bold",
        fontsize=16,
        y=1.02)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"tsne_trajectory_{mix_name}.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 生成 {mix_name} 完成。")

print(f"\n🎉 t-SNE 演化神图全部生成完毕！请去 {output_dir} 目录查看！")
