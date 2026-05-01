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
exp_dir = "exp/TSE_BSRNN_SPK_EMB_Confusion"  # 请替换为你的实际路径
feat_base_dir = os.path.join(exp_dir, "exp1_tsne_features")
output_dir = os.path.join(exp_dir, "exp1_tsne_global_highlight_plots")
os.makedirs(output_dir, exist_ok=True)

mix_folders = sorted(glob.glob(os.path.join(feat_base_dir, "mix_*")))
print(f"🔍 共找到 {len(mix_folders)} 个 Mixture，开始构建高维宇宙...\n")


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
# 2. 收集宇宙中所有的点 (Global Aggregation)
# ========================================================
all_prior, all_pmap, all_post = [], [], []
metadata = []  # 记录每一个点的身世（归属哪个Mixture，是样本还是锚点，什么状态）

for mix_dir in mix_folders:
    mix_name = os.path.basename(mix_dir)
    try:
        prior_feats = np.load(os.path.join(mix_dir, "prior_features.npy"))
        pmap_feats = np.load(os.path.join(mix_dir, "pmap_features.npy"))
        post_feats = np.load(os.path.join(mix_dir, "post_concat_features.npy"))
        df_labels = pd.read_csv(os.path.join(mix_dir, "labels.csv"))
        anchors = np.load(os.path.join(mix_dir, "oracle_anchors.npz"))
    except Exception as e:
        continue

    N = len(df_labels)
    if N < 10: continue

    # A. 收集这 50 个 Sample
    all_prior.append(prior_feats)
    all_pmap.append(pmap_feats)
    all_post.append(post_feats)

    for idx, row in df_labels.iterrows():
        metadata.append({
            "Mix_Name": mix_name,
            "Type": "Sample",
            "State": get_state(row["Delta_SISNRi"])
        })

    # B. 收集 2 个上帝锚点 (Target & Interferer)
    all_prior.append(anchors['tgt_prior'].reshape(1, -1))
    all_prior.append(anchors['int_prior'].reshape(1, -1))

    all_pmap.append(anchors['tgt_pmap'].reshape(1, -1))
    all_pmap.append(anchors['int_pmap'].reshape(1, -1))

    all_post.append(anchors['tgt_post'].reshape(1, -1))
    all_post.append(anchors['int_post'].reshape(1, -1))

    metadata.append({
        "Mix_Name": mix_name,
        "Type": "Anchor_Target",
        "State": "Anchor"
    })
    metadata.append({
        "Mix_Name": mix_name,
        "Type": "Anchor_Interferer",
        "State": "Anchor"
    })

# 垂直拼接成宇宙大矩阵
X_prior = np.vstack(all_prior)
X_pmap = np.vstack(all_pmap)
X_post = np.vstack(all_post)
df_meta = pd.DataFrame(metadata)

# ⚠️ 极其关键：Prior 必须做 L2 归一化以保护 Cosine 流形！
X_prior = normalize(X_prior, norm='l2', axis=1)

print(f"🌌 宇宙构建完成！总星体数 (Samples+Anchors): {len(df_meta)}")

# ========================================================
# 3. 全局 t-SNE 降维 (统一的世界坐标系)
# ========================================================
print("⏳ 正在进行全局 t-SNE 降维 (这可能需要十几秒)...")
# 此时样本量有几百个，perplexity 可以放心用默认的 30，或者稍小一点(如 20)凸显局部
perp = min(30, len(df_meta) // 4)

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

df_meta['Prior_X'], df_meta['Prior_Y'] = tsne_prior.fit_transform(X_prior).T
df_meta['Pmap_X'], df_meta['Pmap_Y'] = tsne_pmap.fit_transform(X_pmap).T
df_meta['Post_X'], df_meta['Post_Y'] = tsne_post.fit_transform(X_post).T

print("✅ 全局降维完成！开始依次打光渲染各 Mixture...\n")

# ========================================================
# 4. 依次打光 (Highlight) 绘图
# ========================================================
unique_mixes = df_meta['Mix_Name'].unique()

for target_mix in unique_mixes:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # 拆分当前 Mixture 和 其他星尘 (Background)
    # 背景星尘：不属于当前 mix 的所有普通样本 (为了画面干净，不画别人的锚点)
    df_bg = df_meta[(df_meta['Mix_Name'] != target_mix)
                    & (df_meta['Type'] == 'Sample')]

    # 当前主角 (Foreground)
    df_fg_samples = df_meta[(df_meta['Mix_Name'] == target_mix)
                            & (df_meta['Type'] == 'Sample')]
    tgt_anchor = df_meta[(df_meta['Mix_Name'] == target_mix)
                         & (df_meta['Type'] == 'Anchor_Target')].iloc[0]
    int_anchor = df_meta[(df_meta['Mix_Name'] == target_mix)
                         & (df_meta['Type'] == 'Anchor_Interferer')].iloc[0]

    spaces = [("1. Prior Space (Identity)", 'Prior_X', 'Prior_Y'),
              ("2. Pmap Space (Energy Mask)", 'Pmap_X', 'Pmap_Y'),
              ("3. Post-Concat Space (Decision)", 'Post_X', 'Post_Y')]

    for i, (title, cx, cy) in enumerate(spaces):
        ax = axes[i]

        # 1. 铺设全局背景星尘 (全宇宙)
        ax.scatter(df_bg[cx],
                   df_bg[cy],
                   c='lightgray',
                   s=30,
                   alpha=0.3,
                   edgecolors='none',
                   zorder=1)

        # 2. 绘制当前 Mixture 的 50 个样本 (高亮)
        sns.scatterplot(data=df_fg_samples,
                        x=cx,
                        y=cy,
                        hue='State',
                        palette=palette,
                        s=120,
                        alpha=0.9,
                        edgecolor='white',
                        ax=ax,
                        legend=(i == 0),
                        zorder=5)

        # 3. 绘制上帝锚点 (Target)
        ax.scatter(tgt_anchor[cx],
                   tgt_anchor[cy],
                   marker='*',
                   s=700,
                   color='#2ecc71',
                   edgecolor='black',
                   linewidth=1.5,
                   label='Oracle Target' if i == 0 else "",
                   zorder=10)

        # 4. 绘制上帝锚点 (Interferer)
        ax.scatter(int_anchor[cx],
                   int_anchor[cy],
                   marker='X',
                   s=450,
                   color='#c0392b',
                   edgecolor='black',
                   linewidth=1.5,
                   label='Oracle Interferer' if i == 0 else "",
                   zorder=10)

        # 样式美化
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(title, fontweight="bold", fontsize=14)

    # 图例处理
    axes[0].legend(loc='upper right',
                   bbox_to_anchor=(1.0, -0.05),
                   ncol=2,
                   fontsize=11)

    plt.suptitle(f"Global Context & Local Trajectory: {target_mix}",
                 fontweight="bold",
                 fontsize=16,
                 y=1.02)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"global_tsne_{target_mix}.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"🌟 渲染完成: {target_mix}")

print(f"\n🎉 惊天神图全部搞定！请移步 {output_dir} 见证奇迹！")
