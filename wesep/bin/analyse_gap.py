import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json

# ⚠️ 请把这里的路径替换成你实际存放 csv 的 exp_dir
exp_dir = "/home/yxy05/code/research_tse/examples/audio/librimix/exp/TSE_BSRNN_SPK_EMB"
csv_path = os.path.join(exp_dir, "diagnostic_three_stages.csv")


def analyze_gap_distribution():
    print(f"📥 正在读取数据: {csv_path}")
    df = pd.read_csv(csv_path)

    # 1. 修复 Bug：去重并重置索引，确保 1对1 映射，避免变 2D 数组
    df_zero = df[df["Method"] == "1_Static_Zero"].drop_duplicates(
        subset=["Utterance"]).set_index("Utterance")
    df_dyn = df[df["Method"] == "3_Dynamic"].drop_duplicates(
        subset=["Utterance"]).set_index("Utterance")

    df_compare = df_dyn[["Output_SISNRi"]].join(df_zero[["Output_SISNRi"]],
                                                lsuffix="_Dyn",
                                                rsuffix="_Zero")
    df_compare["SNRi_Gap"] = df_compare["Output_SISNRi_Dyn"] - df_compare[
        "Output_SISNRi_Zero"]

    # 将索引还原为普通的列，这步极其关键，能彻底解决 seaborn 的 2D 报错
    df_compare = df_compare.reset_index()

    total_utts = len(df_compare)
    print(f"✅ 成功对齐了 {total_utts} 条测试音频的对比数据！\n")

    # 2. 定义 Gap 区间
    bins = [-float('inf'), 0, 1, 3, float('inf')]
    labels = [
        "< 0 dB (Regression)", "0 ~ 1 dB (Marginal)", "1 ~ 3 dB (Moderate)",
        "> 3 dB (High Gap / Hard Cases)"
    ]

    df_compare["Gap_Category"] = pd.cut(df_compare["SNRi_Gap"],
                                        bins=bins,
                                        labels=labels)

    # 3. 统计占比
    dist_counts = df_compare["Gap_Category"].value_counts().sort_index()
    dist_ratios = (dist_counts / total_utts * 100).round(2)

    print("=========================================")
    print("📊 动态交互提升 (Dynamic vs Zero) Gap 分布占比：")
    print("=========================================")
    for label in labels:
        count = dist_counts[label]
        ratio = dist_ratios[label]
        print(f" {label:<30} : {count:>5} 句话 ({ratio:>5.2f}%)")
    print("=========================================\n")

    # ==========================================================
    # 🎧 新增功能：为这 4 类各自随机抽取 10 个代表性样本，生成听音清单
    # ==========================================================
    sampled_list = []
    for label in labels:
        cat_df = df_compare[df_compare["Gap_Category"] == label]
        # 如果不够 10 个就有多少拿多少
        sample_size = min(10, len(cat_df))
        if sample_size > 0:
            # 随机抽样
            sampled_df = cat_df.sample(n=sample_size, random_state=42)
            for _, row in sampled_df.iterrows():
                sampled_list.append({
                    "Category":
                    label,
                    "Utterance_ID":
                    row["Utterance"],
                    "Gap_Gain_dB":
                    round(row["SNRi_Gap"], 2),
                    "Dyn_SNRi":
                    round(row["Output_SISNRi_Dyn"], 2),
                    "Zero_SNRi":
                    round(row["Output_SISNRi_Zero"], 2)
                })

    # 保存听音清单 CSV
    listen_df = pd.DataFrame(sampled_list)
    listen_csv_path = os.path.join(
        exp_dir, "sampled_10_per_category_for_listening.csv")
    listen_df.to_csv(listen_csv_path, index=False)
    print(f"🎧 听音清单已生成！请查看: {listen_csv_path}")
    print(f"   (你可以根据里面的 Utterance_ID 去 exp_dir/audio 找生成的 wav 听一听)\n")

    # 4. 绘图：确保传入的是一维数据
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 使用 df_compare["SNRi_Gap"].astype(float) 确保绝对安全
    df_compare["SNRi_Gap"] = df_compare["SNRi_Gap"].astype(float)
    sns.histplot(data=df_compare,
                 x="SNRi_Gap",
                 bins=50,
                 kde=True,
                 color="#3498db",
                 ax=ax1)

    ax1.axvline(x=0,
                color='red',
                linestyle='--',
                linewidth=2,
                label="0 dB (No Gain)")
    ax1.axvline(x=3,
                color='orange',
                linestyle='--',
                linewidth=2,
                label="3 dB (High Gap Threshold)")
    ax1.set_title("Distribution of SNRi Gap (Dynamic vs Zero)",
                  fontweight="bold")
    ax1.set_xlabel("SNRi Improvement Gap (dB)")
    ax1.set_ylabel("Number of Utterances")
    ax1.legend()

    # 右图饼图
    colors = ['#e74c3c', '#bdc3c7', '#3498db', '#f39c12']
    explode = (0, 0, 0, 0.1)

    # 防止某类数量为 0 导致画饼图报错
    valid_counts = [dist_counts[l] for l in labels]
    ax2.pie(valid_counts,
            labels=labels,
            autopct='%1.1f%%',
            startangle=140,
            colors=colors,
            explode=explode,
            shadow=True)
    ax2.set_title("Proportion of Gap Categories", fontweight="bold")

    plt.tight_layout()
    plot_path = os.path.join(exp_dir, "plot_Gap_Distribution.png")
    fig.savefig(plot_path, dpi=300)
    print(f"🎨 Gap 分布图已保存至: {plot_path}")


if __name__ == "__main__":
    analyze_gap_distribution()
