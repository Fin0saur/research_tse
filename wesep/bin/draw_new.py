import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# ==========================================
# 1. 配置路径与读取数据
# ==========================================
# ⚠️ 替换为你实际的 CSV 路径
csv_path = "/home/yxy05/code/research_tse/examples/audio/librimix/exp/TSE_BSRNN_SPK_EMB_Confusion/target_confusion_data.csv"
save_path = "/home/yxy05/code/research_tse/examples/audio/librimix/exp/TSE_BSRNN_SPK_EMB_Confusion/plot_Target_Confusion_Split_KDE_v2.png"

print(f"正在读取数据: {csv_path}")
df = pd.read_csv(csv_path)


# ==========================================
# 2. 状态打标 (Binning Logic)
# ==========================================
def categorize_snri(snri):
    if snri > 10.0: return "Safe (> 10dB)"
    elif snri > 0.0: return "Marginal (0~10dB)"
    else: return "Fatal Confusion (< 0dB)"


# 根据你的 Output_SISNRi 列生成 State 列
df['State'] = df['Output_SISNRi'].apply(categorize_snri)

# ==========================================
# 3. 绘图样式与调色板设置
# ==========================================
sns.set_theme(style="whitegrid", font_scale=1.2)

states_order = [
    "Safe (> 10dB)", "Marginal (0~10dB)", "Fatal Confusion (< 0dB)"
]

palette = {
    "Safe (> 10dB)": "#2ca02c",
    "Marginal (0~10dB)": "#98df8a",
    "Fatal Confusion (< 0dB)": "#d62728"
}
markers = {
    "Safe (> 10dB)": "o",
    "Marginal (0~10dB)": "o",
    "Fatal Confusion (< 0dB)": "*"
}

# ==========================================
# 4. 创建 1x3 共享坐标轴的画布
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

# 动态获取坐标轴范围
x_col, y_col = "cos(e1, s1)", "cos(e1, s2)"
limit_min = min(df[x_col].min(), df[y_col].min()) - 0.05
limit_max = max(df[x_col].max(), df[y_col].max()) + 0.05

for i, state in enumerate(states_order):
    ax = axes[i]
    subset = df[df["State"] == state]

    # --- 画法升级：散点 + KDE等高线 ---

    # 1. 画底层的高密度等高线 (展现扎堆趋势)
    if len(subset) > 5:  # KDE需要足够的数据点
        sns.kdeplot(
            data=subset,
            x=x_col,
            y=y_col,
            levels=5,  # 画5层等高线
            color=palette[state],
            alpha=0.3,  # 半透明热力圈
            fill=True,
            ax=ax)

    # 2. 画顶层的散点图 (展现具体的离群点)
    sns.scatterplot(
        data=subset,
        x=x_col,
        y=y_col,
        color=palette[state],
        marker=markers[state],
        s=120 if state != "Fatal Confusion (< 0dB)" else 180,  # 把红星再放大一点！
        alpha=0.8,
        edgecolor="white" if state == "Safe (> 5dB)" else None,  # 给绿点加白边，显得更精致
        ax=ax)

    # 3. 绘制 y = x 对角虚线 (判定边界)
    ax.plot([limit_min, limit_max], [limit_min, limit_max],
            color='#c0392b',
            linestyle='--',
            linewidth=2,
            zorder=0)

    # 4. 设置属性
    ax.set_xlim(limit_min, limit_max)
    ax.set_ylim(limit_min, limit_max)
    ax.set_title(f"{state}\n(N={len(subset)})", fontweight="bold", fontsize=15)

    # LaTeX 格式的标签
    ax.set_xlabel(f"Similarity to Target\n${x_col}$", fontweight="bold")
    if i == 0:
        ax.set_ylabel(f"Similarity to Interferer\n${y_col}$",
                      fontweight="bold")
    else:
        ax.set_ylabel("")

# ==========================================
# 5. 调整布局并保存
# ==========================================
plt.suptitle("Target Confusion Decoupled by Extraction Quality",
             fontweight="bold",
             fontsize=18,
             y=1.05)
plt.tight_layout()

os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"🎉 带 KDE 热力圈的三联图已保存至: {save_path}")
