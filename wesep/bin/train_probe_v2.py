"""
探针训练脚本 - Direct Probe 结构 (prior/pmap/post 特征专用)
在 prior / pmap / post_concat 特征上训练分类头，预测 C0/C1 质量类别

方案A (二分类):
  C0 (Success): Dynamic_SISNRi >= 1
  C1 (Failure): Dynamic_SISNRi < 1

使用 InfiniteBalancedSampler 实现类别平衡采样
"""

from __future__ import print_function

import os
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


# ========================================================
# Infinite Balanced Sampler
# ========================================================
class InfiniteBalancedSampler:
    """
    为每个样本赋予与其类别频率成反比的权重，确保每个 batch 近似等量抽取各类别样本。
    采样为有放回，来自单一无限随机序列（固定种子，无 epoch 重洗牌）。
    """
    def __init__(self, labels, batch_size, seed=42):
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)

        labels = np.array(labels)
        self.classes, counts = np.unique(labels, return_counts=True)
        # 类别权重 = 1 / 频率
        class_weights = 1.0 / counts
        class_weights /= class_weights.sum()  # 归一化
        self.weights = np.zeros(len(labels))
        for cls, w in zip(self.classes, class_weights):
            self.weights[labels == cls] = w
        # 归一化为概率
        self.weights /= self.weights.sum()

        # 预生成一个大索引序列，避免每次采样重建
        self.prefetched_indices = self.rng.choice(
            len(labels), size=len(labels) * 1000, replace=True, p=self.weights)
        self.ptr = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.ptr + self.batch_size > len(self.prefetched_indices):
            self.ptr = 0
        batch = self.prefetched_indices[self.ptr:self.ptr + self.batch_size]
        self.ptr += self.batch_size
        return torch.LongTensor(batch)

    def __len__(self):
        return len(self.prefetched_indices)


# ========================================================
# Direct Probe 分类头 (MLP 版本，适合 [B, D] 特征)
# ========================================================
class DirectProbeMLP(nn.Module):
    """
    输入:
      - act: [B, D]  激活特征 (prior/pmap/post_concat)

    结构 (参考 plan.md Direct probe):
      act -> Linear(D -> 256) -> ReLU -> Dropout
            -> Linear(256 -> 256) -> ReLU -> Dropout
            -> Linear(256 -> num_classes)
    """
    def __init__(self, feat_dim, num_classes=2, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, act, spk_emb=None):
        return self.mlp(act)


# ========================================================
# 数据加载
# ========================================================
def load_dataset(exp_dir, split, feat_type='prior'):
    """加载指定数据集的特征和标签"""
    feat_dir = os.path.join(exp_dir, f'{split}_features')

    feat_file_map = {
        'prior': 'prior_features.npy',
        'pmap': 'pmap_features.npy',
        'post': 'post_concat_features.npy',
    }

    feat_path = os.path.join(feat_dir, feat_file_map[feat_type])
    labels_path = os.path.join(feat_dir, 'labels.csv')

    X = np.load(feat_path)
    df = pd.read_csv(labels_path)

    # 方案A: Dynamic_SISNRi >= 1 -> C0 (0), 否则 C1 (1)
    labels = (df['Dynamic_SISNRi'].values < 1.0).astype(np.int64)

    return X, labels, df


# ========================================================
# 训练 & 评估
# ========================================================
def train_one_epoch(model, optimizer, sampler_iter, data_X, data_y, device, batch_size=128):
    model.train()
    total_loss = 0
    n_batches = 0
    criterion = nn.CrossEntropyLoss()

    for _ in range(500):  # 每个 epoch 固定 500 个 batch
        try:
            indices = next(sampler_iter)
        except StopIteration:
            sampler_iter = iter(InfiniteBalancedSampler(
                data_y.numpy(), batch_size=batch_size, seed=42))
            indices = next(sampler_iter)

        x_batch = data_X[indices].to(device)
        y_batch = data_y[indices].to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, X, y, device):
    model.eval()
    X_t = torch.FloatTensor(X).to(device)
    logits = model(X_t)
    preds = logits.argmax(dim=1).cpu().numpy()
    y_np = y.numpy() if isinstance(y, torch.Tensor) else y

    macro_f1 = f1_score(y_np, preds, average='macro', zero_division=0)
    return macro_f1, preds


def main():
    parser = argparse.ArgumentParser(description='探针训练 - Direct Probe (prior/pmap/post)')
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='实验目录，包含 train_features/, val_features/, test_features/')
    parser.add_argument('--feat_type', type=str, default='prior',
                        choices=['prior', 'pmap', 'post'],
                        help='特征类型: prior (speaker embedding), pmap (energy mask), post (post_concat)')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default=None,
                        help='模型保存目录，默认为 exp_dir/probe_results/{feat_type}/')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    save_dir = args.save_dir or os.path.join(args.exp_dir, 'probe_results', args.feat_type)
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(f"  Direct Probe Training: {args.feat_type}")
    print(f"  Device: {device}")
    print(f"  Save dir: {save_dir}")
    print("=" * 60)

    # 加载数据
    print("\n[1] 加载数据 ...")
    X_train, y_train, df_train = load_dataset(args.exp_dir, 'train', args.feat_type)
    X_val, y_val, df_val = load_dataset(args.exp_dir, 'val', args.feat_type)
    X_test, y_test, df_test = load_dataset(args.exp_dir, 'test', args.feat_type)

    print(f"\n  原始数据分布:")
    print(f"  Train: C0={sum(y_train==0)}, C1={sum(y_train==1)}")
    print(f"  Val:   C0={sum(y_val==0)}, C1={sum(y_val==1)}")
    print(f"  Test:  C0={sum(y_test==0)}, C1={sum(y_test==1)}")

    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 转为 Tensor
    data_X_train = torch.FloatTensor(X_train)
    data_y_train = torch.LongTensor(y_train)
    data_X_val = torch.FloatTensor(X_val)
    data_y_val = torch.LongTensor(y_val)
    data_X_test = torch.FloatTensor(X_test)
    data_y_test = torch.LongTensor(y_test)

    feat_dim = X_train.shape[1]
    print(f"\n  特征维度: {feat_dim}")

    # 构建模型
    model = DirectProbeMLP(
        feat_dim=feat_dim,
        num_classes=2,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 打印模型参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {n_params / 1e6:.2f}M")

    # Infinite Balanced Sampler
    sampler = InfiniteBalancedSampler(y_train, batch_size=args.batch_size, seed=args.seed)
    sampler_iter = iter(sampler)

    # 训练
    print(f"\n[2] 开始训练 ({args.epochs} epochs) ...")
    best_f1 = 0
    best_epoch = 0
    train_losses = []
    val_f1s = []

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(
            model, optimizer, sampler_iter, data_X_train, data_y_train, device, args.batch_size
        )
        train_losses.append(avg_loss)

        # 每 5 个 epoch 验证一次
        if epoch % 5 == 0 or epoch == args.epochs:
            val_f1, _ = evaluate(model, X_val, y_val, device)
            val_f1s.append(val_f1)
            print(f"  Epoch {epoch:3d}: train_loss={avg_loss:.4f}, val_macro_f1={val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
                torch.save(scaler, os.path.join(save_dir, 'scaler.pt'))
                print(f"    ★ New best model! macro_f1={best_f1:.4f}")

    print(f"\n[3] 最终评估 ...")
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pt')))
    test_f1, test_preds = evaluate(model, X_test, y_test, device)

    print(f"\n  Best epoch: {best_epoch}, Best val macro_f1: {best_f1:.4f}")
    print(f"  Test macro_f1: {test_f1:.4f}")

    print(f"\n  Classification Report (Test):")
    print(classification_report(y_test, test_preds, target_names=['C0_Success', 'C1_Failure']))

    print(f"\n  Confusion Matrix (Test):")
    cm = confusion_matrix(y_test, test_preds)
    print(f"  [[TN={cm[0,0]:4d}, FP={cm[0,1]:4d}]")
    print(f"   [FN={cm[1,0]:4d}, TP={cm[1,1]:4d}]]")

    # 保存训练曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, 'b-', linewidth=0.8)
    axes[0].set_title('Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')

    eval_epochs = list(range(5, args.epochs + 1, 5))
    if len(eval_epochs) != len(val_f1s):
        eval_epochs = list(range(1, args.epochs + 1))[::5]
        if len(eval_epochs) != len(val_f1s):
            eval_epochs = list(range(1, len(val_f1s) * 5 + 1, 5))
    axes[1].plot(eval_epochs[:len(val_f1s)], val_f1s, 'c-', linewidth=0.8)
    axes[1].set_title('Val Macro F1')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Macro F1')

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()

    # 保存结果汇总
    results = {
        'feat_type': args.feat_type,
        'feat_dim': feat_dim,
        'best_epoch': best_epoch,
        'best_val_f1': float(best_f1),
        'test_macro_f1': float(test_f1),
        'n_params': n_params,
        'train_c0': int(sum(y_train == 0)),
        'train_c1': int(sum(y_train == 1)),
        'test_c0': int(sum(y_test == 0)),
        'test_c1': int(sum(y_test == 1)),
    }
    pd.DataFrame([results]).to_csv(
        os.path.join(save_dir, 'results.csv'), index=False
    )

    print(f"\n  结果已保存至 {save_dir}")
    print(f"  Test Macro F1: {test_f1:.4f}")


if __name__ == '__main__':
    main()
