import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

# ==========================================
# 1. 探针网络架构 (严格复刻 Notion 文档)
# ==========================================
class DirectProbe(nn.Module):
    def __init__(self, num_classes=2):
        super(DirectProbe, self).__init__()
        
        # 2D Conv on (C, T) jointly
        # 输入: [B, 128, 32, T] -> 输出: [B, 256, 32, T]
        self.conv2d_block = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # 消除 T 维度 -> [B, 256, 32, 1]
        self.pool_t = nn.AdaptiveAvgPool2d((32, 1))
        
        # 1D Conv on C
        # 输入: [B, 256, 32] -> 输出: [B, 256, 32]
        self.conv1d_block = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # 消除 C 维度 -> [B, 256, 1]
        self.pool_c = nn.AdaptiveAvgPool1d(1)
        
        # 映射到共享空间
        self.act_proj = nn.Linear(256, 256)
        self.spk_proj = nn.Linear(192, 256)
        
        # 拼接后的 MLP 分类器
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, activation, spk_emb):
        """
        activation: [B, 32, 128, T]  (C=32, D=128, T)
        spk_emb:    [B, 192]
        """
        # 1. Permute -> [B, 128, 32, T]
        x = activation.permute(0, 2, 1, 3)
        
        # 2. 2D Conv
        x = self.conv2d_block(x)
        
        # 3. AdaptiveAvgPool2d -> [B, 256, 32, 1], 然后 squeeze 掉最后一个维度
        x = self.pool_t(x).squeeze(-1) # -> [B, 256, 32]
        
        # 4. 1D Conv
        x = self.conv1d_block(x)
        
        # 5. AdaptiveAvgPool1d -> [B, 256, 1], 然后 squeeze
        x = self.pool_c(x).squeeze(-1) # -> [B, 256]
        
        # 6. 映射与拼接
        act_repr = self.act_proj(x)       # [B, 256]
        spk_repr = self.spk_proj(spk_emb) # [B, 256]
        
        fused_repr = torch.cat([act_repr, spk_repr], dim=1) # [B, 512]
        
        # 7. 分类输出 [B, 2] (不加 Softmax，因为后面用 CrossEntropyLoss)
        logits = self.mlp(fused_repr)
        return logits


# ==========================================
# 2. 虚拟数据集与平衡采样器
# ==========================================
class ProbeDataset(Dataset):
    def __init__(self, num_samples, p_success=0.91, seq_len=100):
        """
        虚拟数据集，用于模拟你的真实特征。
        你需要替换这里的逻辑，读取你本地的 .npy 或 .pt 文件。
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        
        # 按照 Notion 文档的比例模拟标签 (91% 成功(0), 8% 混淆(1))
        # 注意：文档里 Success是0，Confusion是1
        labels = np.random.choice([0, 1], size=num_samples, p=[p_success, 1 - p_success])
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 模拟 activation: [32, 128, T]
        # 在实际代码中，请用 np.load() 或 torch.load() 读取你的特征
        act = torch.randn(32, 128, self.seq_len)
        
        # 模拟 ECAPA-TDNN 说话人嵌入: [192]
        spk = torch.randn(192)
        
        return act, spk, self.labels[idx]

def get_balanced_sampler(dataset):
    """
    实现 Notion 文档中的 InfiniteBalancedSampler 逻辑
    为每个样本赋予与其类别频率成反比的权重
    """
    labels = dataset.labels.numpy()
    class_counts = np.bincount(labels)
    # 类别权重 = 1 / 类别数量
    class_weights = 1.0 / class_counts
    # 为每个样本分配权重
    sample_weights = class_weights[labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True # 必须有放回，否则小类会被抽干
    )
    return sampler


# ==========================================
# 3. 训练与验证引擎
# ==========================================
def train_probe():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")
    
    # 1. 准备数据 (模拟 train-100+bias 和 test集 的数量比例)
    print("📦 正在准备数据集...")
    train_dataset = ProbeDataset(num_samples=10000, p_success=0.91)
    test_dataset = ProbeDataset(num_samples=2000, p_success=0.91)
    
    train_sampler = get_balanced_sampler(train_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler, num_workers=4)
    # 验证集不需要平衡采样
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # 2. 初始化模型、损失函数和优化器
    model = DirectProbe(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss() # 普通无权重 Loss，因为 sampler 已经做过平衡了
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    epochs = 10
    
    print("🔥 开始训练探针...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (act, spk, labels) in enumerate(train_loader):
            act, spk, labels = act.to(device), spk.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(act, spk)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # --- 验证阶段 ---
        model.eval()
        all_labels = []
        all_probs = []
        all_preds = []
        
        with torch.no_grad():
            for act, spk, labels in test_loader:
                act, spk, labels = act.to(device), spk.to(device), labels.to(device)
                logits = model(act, spk)
                
                # 取类别 1 (混淆) 的概率用于计算 AUROC
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)
                
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                
        # 计算指标
        val_auroc = roc_auc_score(all_labels, all_probs)
        val_acc = accuracy_score(all_labels, all_preds)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} | Val ACC: {val_acc*100:.2f}% | Val AUROC: {val_auroc:.4f}")
        
    print("\n✅ 训练完成！详细分类报告 (Test Set):")
    print(classification_report(all_labels, all_preds, target_names=["C0: Success", "C1: Confusion"]))

if __name__ == "__main__":
    train_probe()