import os
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix

def load_global_data(feat_base_dir, feature_name="post_concat"):
    """加载数据并保持完整的 32896 维"""
    print(f"🔍 正在加载全局特征: {feature_name}...")
    mix_folders = sorted(glob.glob(os.path.join(feat_base_dir, "mix_*")))
    
    X_list, Y_list = [], []
    for d in mix_folders:
        csv_path = os.path.join(d, "labels.csv")
        feat_path = os.path.join(d, f"{feature_name}_features.npy")
        if not os.path.exists(csv_path) or not os.path.exists(feat_path): continue
        
        df = pd.read_csv(csv_path)
        feats = np.load(feat_path) # [N, 32896]
        print(f"feats_shape:{feats.shape}")
        n_samples = len(df)
        if len(feats) >= n_samples:
            X_list.append(feats[:n_samples])
            Y_list.extend(df['Dynamic_SISNRi'].values)
    print(f"shape:{feats.shape}")
    X_all = np.vstack(X_list)
    Y_all = np.array(Y_list)
    print(f"📦 成功加载 {len(X_all)} 个样本. 全局特征维度: {X_all.shape[1]}")
    return X_all, Y_all

def run_global_baseline(X, Y, success_th=5.0, fail_th=0.0):
    # 1. 剔除中间模糊地带，提取明确的成功与失败样本
    mask_eval = (Y >= success_th) | (Y <= fail_th)
    X_eval = X[mask_eval]
    Y_eval = Y[mask_eval]
    
    # 标签二值化：1 为成功，0 为失败
    L_eval = (Y_eval >= success_th).astype(int)
    print(f"⚖️ 参与 Baseline 测试的样本分布: 成功 {np.sum(L_eval==1)} 个, 失败 {np.sum(L_eval==0)} 个")

    # 2. 划分训练集和测试集 (7:3)
    X_train, X_test, L_train, L_test = train_test_split(
        X_eval, L_eval, test_size=0.3, random_state=42, stratify=L_eval
    )

    # 3. 特征标准化 (对抗 32896 维的关键)
    print("🛠️ 正在进行 StandardScaler 标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. 训练全局线性探针
    print("🧠 正在训练 32896 维的全局 Logistic Regression...")
    # C=0.01 是强正则化，防止三万维特征立刻过拟合到训练集
    # class_weight='balanced' 保证分类器不会因为成功样本多就全猜成功
    probe = LogisticRegression(max_iter=1000, C=0.01, solver='liblinear', class_weight='balanced')
    probe.fit(X_train_scaled, L_train)

    # 5. 在测试集上进行验证
    print("✅ 训练完成！正在测试集上评估...")
    
    # 输出预测的概率得分 (用于算 AUROC)
    scores_test = probe.decision_function(X_test_scaled)
    # 输出预测的硬标签 0 或 1 (用于算 Accuracy)
    preds_test = probe.predict(X_test_scaled)

    # 6. 计算详细指标
    auroc = roc_auc_score(L_test, scores_test)
    accuracy = accuracy_score(L_test, preds_test)
    conf_matrix = confusion_matrix(L_test, preds_test)

    print("\n" + "="*50)
    print("🏆 【全局 32896 维 Baseline】 测试集表现：")
    print(f"🔹 预测正确率 (Accuracy): {accuracy * 100:.2f}%")
    print(f"🔹 曲线下面积 (AUROC):   {auroc:.4f}")
    print("\n📊 混淆矩阵 (Confusion Matrix):")
    print("                 预测为失败(0)  预测为成功(1)")
    print(f"真实为失败(0):      {conf_matrix[0][0]:<10}    {conf_matrix[0][1]:<10}")
    print(f"真实为成功(1):      {conf_matrix[1][0]:<10}    {conf_matrix[1][1]:<10}")
    print("\n📄 详细分类报告:")
    print(classification_report(L_test, preds_test, target_names=["Failure (0)", "Success (1)"]))
    print("="*50 + "\n")

if __name__ == "__main__":
    FEAT_BASE_DIR = "/data2/yxy05/exp/TSE_BSRNN_SPK_EMB_tSNE_v2/exp1_tsne_features" # 修改为你的路径
    
    # 你可以把 pmap 换成 post_concat 对比一下
    X, Y = load_global_data(FEAT_BASE_DIR, feature_name="pmap") 
    run_global_baseline(X, Y, success_th=5.0, fail_th=0.0)