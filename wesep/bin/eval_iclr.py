import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

def advanced_linear_probe(feat_base_dir, feature_type="post_concat", success_th=3.0, fail_th=0.0):
    mix_folders = sorted(glob.glob(os.path.join(feat_base_dir, "mix_*")))
    print(f"🔍 正在加载 {feature_type} 数据...")

    X_list, sisnr_list = [], []
    for d in mix_folders:
        csv_path = os.path.join(d, "labels.csv")
        feat_path = os.path.join(d, f"{feature_type}_features.npy")
        if not os.path.exists(csv_path) or not os.path.exists(feat_path): continue
        
        df = pd.read_csv(csv_path)
        feats = np.load(feat_path)
        print(feats.shape)
        n_samples = len(df)
        if len(feats) >= n_samples:
            X_list.append(feats[:n_samples])
            sisnr_list.extend(df['Dynamic_SISNRi'].values)

    X_all = np.vstack(X_list)
    Y_all = np.array(sisnr_list)

    # 1. 划分阵营
    mask_success = Y_all > success_th
    mask_fail = Y_all < fail_th
    
    X_success, Y_success = X_all[mask_success], Y_all[mask_success]
    X_fail, Y_fail = X_all[mask_fail], Y_all[mask_fail]
    
    print(f"⚖️ 原始分布: 成功 {len(X_success)}, 失败 {len(X_fail)}")
    
    # 2. 强制平衡类别 (Under-sampling)
    min_len = min(len(X_success), len(X_fail))
    if min_len == 0: return
    
    # 随机抽取同样数量的成功样本
    idx_success = np.random.choice(len(X_success), min_len, replace=False)
    X_success_bal = X_success[idx_success]
    Y_success_bal = Y_success[idx_success]
    
    X_bal = np.vstack([X_success_bal, X_fail])
    # 标签：1 为成功，0 为失败
    Labels_bal = np.concatenate([np.ones(min_len), np.zeros(min_len)])
    Y_real_bal = np.concatenate([Y_success_bal, Y_fail])
    
    print(f"⚖️ 平衡后分布: 成功 {min_len}, 失败 {min_len} (共 {len(X_bal)} 个训练样本)")

    # 3. 划分训练/测试集 (我们不能在训练 Linear Probe 的数据上测 PCC)
    X_train, X_test, L_train, L_test, _, Y_test = train_test_split(
        X_bal, Labels_bal, Y_real_bal, test_size=0.3, random_state=42
    )

    # 4. 数据预处理：极其重要！压制三万维的噪音
    print("🛠️ 正在进行特征标准化 (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # (可选) PCA 降维，如果你发现 Logistic 还是不行，把这三行解除注释
    # print("🛠️ 正在进行 PCA 降维...")
    # pca = PCA(n_components=512) # 将 3万维 浓缩成 512维
    # X_train_scaled = pca.fit_transform(X_train_scaled)
    # X_test_scaled = pca.transform(X_test_scaled)

    # 5. 训练带有强 L2 正则化的逻辑回归
    print("🧠 正在训练线性探测器 (Logistic Regression)...")
    clf = LogisticRegression(max_iter=1000, C=0.01, solver='liblinear') # C越小，正则化越强，越抗噪
    clf.fit(X_train_scaled, L_train)

    # 6. 在测试集上进行验证
    # decision_function 返回的是距离超平面的连续距离，正是我们需要的“置信度得分”！
    confidence_scores = clf.decision_function(X_test_scaled)

    auroc = roc_auc_score(L_test, confidence_scores)
    pcc, p_val = pearsonr(confidence_scores, Y_test)

    print("\n" + "="*40)
    print("🏆 【改进版】探测指标结果：")
    print(f"🔹 曲线下面积 (AUROC): {auroc:.4f}")
    print(f"🔹 皮尔逊相关系数 (PCC): {pcc:.4f} (p-value: {p_val:.2e})")
    print("="*40 + "\n")

if __name__ == "__main__":
    feat_dir = "/data2/yxy05/exp/TSE_BSRNN_SPK_EMB_tSNE_v2/exp1_tsne_features" # 修改为你的路径
    print(">>> 进阶测试: Post-Concat <<<")
    advanced_linear_probe(feat_dir, "post_concat", success_th=5.0, fail_th=0.0)