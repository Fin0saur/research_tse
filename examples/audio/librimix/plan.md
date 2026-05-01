# 预测网络结构与消融

# **probe_experiment — 探针实验：从内部表征诊断 TSE 提取质量**

## **概述**

本实验研究基线 BSRNN 目标说话人提取模型的中间层激活能否在**无参考信号**的条件下预测提取质量。我们在冻结的 BSRNN 的 7 个内部检查点上附加轻量级探针分类器，训练它们将每次提取分为基于 SI-SNRi 的 3 个质量类别。

**核心研究问题**：冻结的 TSE 模型的内部表征（说话人嵌入 + 融合/分离层特征）能否揭示提取是否成功、发生说话人混淆或完全失败？

**实际意义**：如果探针成功，我们可以构建一个与推理并行运行的置信度估计器——实现拒绝输出、使用不同注册语音重试、或级联到更强模型——所有这些均不需要真实参考信号。

## **实验原理**

### **三分类质量方案**

阈值：BETA=1（SI-SNRi 边界）：

| **标签** | **名称** | **条件** |
| --- | --- | --- |
| 0 | 成功 (Success) | SI-SNRi >= 1 |
| 1 | 说话人混淆 (Speaker Confusion) | SI-SNRi < 1 且干扰信号 SI-SNRi >= 1 |
| 2 | 完全失败 (Total Failure) | SI-SNRi < 1 且干扰信号 SI-SNRi < 1 |

为减小分类头的训练难度，实际分类时，我们仅仅使用0和1这两类。

### **数据策略**

**说话人不重叠的数据划分**（标准 LibriMix）：

- train-100：251 位说话人，13,900 个混合，27,800 个样本
- dev：40 位说话人，3,000 个混合，6,000 个样本
- test：40 位说话人，3,000 个混合，6,000 个样本
- 任意两个子集之间无说话人重叠

**偏差注册数据**：`data/enroll_bias/dev/bias_enroll20_seed42.jsonl` 提供 120,000 个样本（每个 dev 混合 20 种不同注册 x 3,000 混合 x 2 说话人）。推理结果在 `exp/AAAI-2026/baseline/inference/bias_set_baseline_dev/infer_fused_emb.jsonl`。

**训练集**：train-100 (27,800) + 全部偏差 dev (120,000) = 147,800 个样本

- C0（成功）：134,048 (90.7%)
- C1（混淆）：12,349 (8.4%)
- C2（失败）：1,403 (0.9%)

**验证集**：test (6,000 个样本)

- C0：5,481 (91.4%)
- C1：439 (7.3%)
- C2：80 (1.3%)

**类别平衡策略**：`InfiniteBalancedSampler` 为每个样本赋予与其类别频率成反比的权重，确保每个 batch 近似等量抽取 C0/C1/C2 样本。采样为有放回，来自单一无限随机序列（固定种子，无 epoch 重洗牌）。损失函数使用普通无权重的 `CrossEntropyLoss`。

**数据重叠说明**：偏差 dev 数据使用与 dev 集相同的 3,000 个混合（仅注册语音不同）。为避免泄漏，使用 test 作为验证集。train-100 与 test 无混合/说话人重叠。

### **探针架构**

每个探针头接收：

- `activation`：`[B, 32, 128, T]` — 某一 BSRNN 层的完整时序激活（C=nband=32, D=feature_dim=128, T=时间帧数）
- `spk_emb`：`[B, 192]` — 冻结的 ECAPA-TDNN 说话人嵌入

**设计原则**：从物理意义看，`[B, C, D, T]` 维度的激活必须完整融合 C（频率子带）和 T（时间）两个维度，才能形成对该激活层说话人相关信息的完整表征。只有在形成完整表征之后，才应当将其与 `spk_emb` 映射到同一空间做比较。

当前代码中支持两类探针头：

- **Direct probe**：直接对 `[B, 32, 128, T]` 激活做 2D/1D 卷积汇聚，再与 `spk_emb` 拼接分类
- **Mask-proxy probe**：先从激活预测复数 mask，作用到 mixture subband STFT 上得到 pseudo-separated magnitude，再抽取 speaker proxy embedding 与 `spk_emb` 比较后分类

### **Direct probe（~1.11M 参数/头）**

```
输入：activation [B, C=32, D=128, T] + spk_emb [B, 192]

1. Permute → [B, D=128, C=32, T]
   将 feature_dim 作为 Conv2d 输入通道，(nband, T) 作为空间维度

2. 2D Conv on (C, T) jointly（~665K 参数）：
   Conv2d(128, 192, k=3, pad=1) + BN + ReLU
   Conv2d(192, 256, k=3, pad=1) + BN + ReLU
   → [B, 256, 32, T]
   在频率和时间维度上同时建模局部交互模式

3. AdaptiveAvgPool2d((32, 1))：消除 T 维度
   → [B, 256, 32]

4. 1D Conv on C（~197K 参数）：
   Conv1d(256, 256, k=3, pad=1) + BN + ReLU
   → [B, 256, 32]
   融合跨频率子带的信息

5. AdaptiveAvgPool1d(1)：消除 C 维度
   → [B, 256]
   此时已获得融合了全部 (C, D, T) 信息的完整激活表征

6. 映射到共享空间 + 拼接 + 分类（~247K 参数）：
   act_proj  = Linear(256, 256)   → act_repr  [B, 256]
   spk_proj  = Linear(192, 256)   → spk_repr  [B, 256]
   concat(act_repr, spk_repr)     → [B, 512]
   MLP(512, 256) + ReLU + Dropout(0.3)
   Linear(256, num_classes)
```

参数量：~1.11M/头，7 头 = ~7.77M 总计。所有头独立（无共享参数）。

### **Mask-proxy probe（~1.88M 参数/头 +  speaker encoder）**

动机：`sep_layer_k` 的判别信息可能不是“抽象 pooled activation 是否像目标说话人”，而是“这一层激活若被解码成一个伪分离结果，它在 speaker space 中更像谁”。因此 `mask_proxy` 不再直接对 activation 做全局分类，而是先显式走一条“激活 -> 伪分离频谱 -> speaker proxy”的路径。

```
输入：activation [B, nband=32, feature_dim=128, T]
    + subband_mix_spec: 32 个复数子带频谱
    + spk_emb [B, 192]

1. MaskHead（每个头独立）
   activation -> [B*nband, 128, T]
   GroupNorm(1, 128)
   5 x Conv1d(128/512 -> 512, k=mask_kernel_size) + Tanh
   每个 band 一个独立的 projection: Conv1d(512, band_width * 4, k=1)

2. 生成 gated complex mask
   out -> [B, 2, 2, band_width, T]
   mask = tanh_part * sigmoid_part
   分成 real/imag，与该 band 的复数 mix STFT 相乘

3. 拼接所有 band
   -> pseudo-separated complex spec [B, F, T]
   -> 取 magnitude [B, F, T]

4. Shared SpeakerProxyEncoder（7 个头共享 1 份）
   Conv1d(F, 256, k=5, p=2) + BN + ReLU
   Conv1d(256, 256, k=3, s=2, p=1) + BN + ReLU
   AdaptiveAvgPool1d(1)
   Linear(256, 192) + ReLU
   -> proxy_emb [B, 192]

5. 投影 + 分类
   proxy_proj = Linear(192, 256) -> proj_proxy [B, 256]
   spk_proj   = Linear(192, 256) -> proj_spk   [B, 256]
   concat(proj_proxy, proj_spk)  -> [B, 512]
   MLP(512, 256) + ReLU + Dropout(0.3)
   Linear(256, num_classes)
```

设计理由：

- `MaskHead` 强迫 probe 从 `sep_layer_k` 激活中显式恢复“它想把 mixture 往哪个说话人方向分离”
- `SpeakerProxyEncoder` 将伪分离结果压到一个稳定的 speaker proxy 空间
- 相比 direct probe，`mask_proxy` 更贴近 `C0/C1` 的物理语义：不是问“激活像不像错误样本”，而是问“这层激活解码出的伪说话人更像 target 还是 interference”

实现说明：

- 默认 `mask_kernel_size=1`，即逐帧点式 `1x1 Conv1d`
- 可通过 `-mask_kernel_size 3` 等方式为 `MaskHead` 引入局部时间感受野
- 当只关注单层时，可直接使用 `-probe_layers sep_layer_2` 仅训练/评估该层

参数量说明：

- `MaskProxyProbeClassifier` 每头约 **1.88M**
- `SpeakerProxyEncoder`

### **Mask-proxy 的可选 speaker 辅助损失**

在 `mask_proxy` 模式下，可以通过脚本参数开启一个**类条件说话人排序损失**，用于显式约束 proxy embedding 在 speaker space 中朝正确的说话人方向偏移。该损失与原有分类损失并行存在，只在训练时使用，推理时不引入额外分支。

训练时可用监督：

- `proxy_emb`：由 `MaskHead + SpeakerProxyEncoder` 从 `sep_layer_k` 激活恢复出的 speaker proxy 表征
- `target_emb`：冻结 baseline BSRNN 内部 speaker encoder 对 `target_chunk_path` 的编码
- `interf_emb`：同一冻结 speaker encoder 对 `interf_chunk_path` 的编码

实现与当前代码一致，可写为：

```
z = F.normalize(proxy_emb, dim=-1)
e_t = F.normalize(target_emb, dim=-1)
e_i = F.normalize(interf_emb, dim=-1)

s_t = (z * e_t).sum(dim=-1)
s_i = (z * e_i).sum(dim=-1)

sign = torch.where(labels == 0, 1.0, -1.0)  # C0=success, C1=interference
L_spk = softplus(margin - sign * (s_t - s_i)).mean()
```

等价地按类别展开：

```
C0 (success):      L_spk = softplus(margin - (s_t - s_i))
C1 (interference): L_spk = softplus(margin - (s_i - s_t))
```

其中：

- `margin` 默认 `0.15`
- `C0` 时希望 `s_t - s_i` 尽可能大于 `margin`
- `C1` 时希望 `s_i - s_t` 尽可能大于 `margin`
- `softplus` 版本是平滑排序损失，不是硬 margin hinge

当前每个启用 head 的总训练损失为：

```
L_head = L_cls + lambda_spk * L_spk
```

其中 `lambda_spk = speaker_aux_weight`，默认 `0.3`。最终总损失为所有启用 probe heads 的 `L_head` 之和。

设计动机：

- 分类损失只要求 probe 区分 `C0/C1`
- speaker 辅助损失进一步要求中间 proxy 表征在同一 speaker space 中“更像哪一个说话人”

### **残差激活模式（Residual）**

动机：`fused_emb` 是 SpeakerFuseLayer（FiLM）融合说话人嵌入后的初始表征。各 `sep_layer_k` 在此基础上逐层精炼。减去 `fused_emb` 后，残差只保留分离层新增的信息增量，去除了所有层共享的初始融合信号。

```
# compute_residual_activations() in train_probe_v3.py
fused_emb = activations['fused_emb']  # [B, 32, 128, T]
for k in range(num_sep_layers):
    name = f'sep_layer_{k}'
    activations[name] = activations[name] - fused_emb  # 残差
# fused_emb 本身保持不变（自身残差无意义）
```

分类头结构与 base 完全相同（~1.11M/头），仅输入激活值不同。

### **Enrollment Encoder 分支（Enroll）**

动机：观察到部分提取错误样例在 `spk_emb`（ECAPA-TDNN 192 维）处就展现出异常特征，说明预训练说话人模型可能存在盲区。引入独立的 enrollment 编码器，直接从 fbank 特征学习说话人表征，不依赖冻结的 ECAPA-TDNN。

**EnrollmentEncoder 架构**（~2.32M 参数）：

```
输入：enroll_fbank [B, T', 80]  (enrollment 语音的 fbank 特征)

1. 转置 → [B, 80, T'] (Conv1d 要求通道在前)

2. 6 层 Conv1d 堆叠：
   前 4 层 stride=1（不降采样，在全时间分辨率上建立通道容量）：
     Conv1d(80,  128, k=5, s=1, p=2) + BN + ReLU  → [B, 128, T']
     Conv1d(128, 256, k=3, s=1, p=1) + BN + ReLU  → [B, 256, T']
     Conv1d(256, 256, k=3, s=1, p=1) + BN + ReLU  → [B, 256, T']
     Conv1d(256, 384, k=3, s=1, p=1) + BN + ReLU  → [B, 384, T']
   后 2 层 stride=2（逐步时间压缩）：
     Conv1d(384, 512, k=3, s=2, p=1) + BN + ReLU  → [B, 512, T'/2]
     Conv1d(512, 512, k=3, s=2, p=1) + BN + ReLU  → [B, 512, T'/4]

3. AdaptiveAvgPool1d(1) → [B, 512]

4. 2 层 MLP 投影到共享空间：
   Linear(512, 384) + ReLU
   Linear(384, 256)           → enroll_repr [B, 256]
```

设计理由：

- 前 4 层 stride=1 保持全时间分辨率，让网络充分学习局部频谱/时间模式
- 后 2 层 stride=2 逐步压缩，避免一次性丢失时间信息
- 参数量 ~2.32M，保证 enrollment 路径有足够容量学到有意义的表征
- 第一层 k=5 宽核覆盖更大时间窗口，后续层 k=3

**Enroll 变体分类头结构**（~3.49M/头）：

```
输入：activation [B, 32, 128, T] + spk_emb [B, 192] + enroll_fbank [B, T', 80]

Stages 1-4：与 base 相同（2D Conv + Pool T + 1D Conv + Pool C → act_repr [B, 256]）

Stage 5：三路映射 + 拼接 + 分类
  act_proj    = Linear(256, 256)  → act_repr    [B, 256]
  spk_proj    = Linear(192, 256)  → spk_repr    [B, 256]
  enroll_enc  = EnrollmentEncoder → enroll_repr [B, 256]
  concat(act_repr, spk_repr, enroll_repr)       → [B, 768]
  MLP(768, 256) + ReLU + Dropout(0.3)
  Linear(256, num_classes)
```

参数量：~3.49M/头（含 EnrollmentEncoder ~2.32M），7 头 = ~24.46M 总计。
注意 EnrollmentEncoder 是每个头各自独立一份（不共享），因为各层探针可能需要从 enrollment 中关注不同信息。

### **实验变体与消融结果**

在基线探针（v4_2cls direct）的基础上，设计了多个变体实验来探索不同信息来源与 probe 形式对判别力的影响：

| **变体** | **标识** | **激活输入** | **分类头输入** | **参数量/头** | **假设** | 最好结果（macro-f1） | 结果总结 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Base** | `v4_2cls` | `sep_layer_k` 原始激活 | activation→act_repr(256) + spk_repr(256) = 512 | ~1.11M | 基线 | sep_layer_2: 62.8%           sep_layer_3: 65.3%          sep_layer_4: 67.2% | 我们希望进行修复的层要尽量早，所以最好是使用sep_layer_2进行修复，根据观察，我觉得macro-f1 达到70%时这个分类器的结果拿来修复会是比较有帮助的结果 |
| **Residual** | `v4_2cls_residual` | `sep_layer_k - fused_emb` 残差 | activation→act_repr(256) + spk_repr(256) = 512 | ~1.11M | 移除初始融合表征后，残差更能展现分离层学到的声学特征 | sep_layer_2: 60.4%           sep_layer_3: 63.8%          sep_layer_4: 64.4% | 不如base |
| **Enroll** | `v4_2cls_enroll` | `sep_layer_k` 原始激活 | activation→act_repr(256) + spk_repr(256) + speech enroll→enroll_repr(256) = 768 | ~3.49M | 引入独立 enrollment 编码器，弥补预训练 ECAPA-TDNN 可能遗漏的说话人信息 | sep_layer_2: 62.6%           sep_layer_3: 65.8%          sep_layer_4: 66.8% | 感觉相较于base基本上是半斤八两，带来的影响并没有多余的影响，感觉没用 |
| **Mask proxy** | `v4_2cls_mask_proxy` | `sep_layer_k` 原始激活 + mixture subband STFT | activation→proj_proxy(256) + proj_spk(256) = 512 | ~1.88M + shared encoder | 先显式恢复伪分离说话人，再做 target/interference 判别 | sep_layer_2: 63.8%           sep_layer_3: 65.3%          sep_layer_4: 66.7% | 从总体训练结果看相较于base，在sep_layer_2上有提升，但提升也相当有限（但从具体log看，sep_layer_2上波动变小，感觉有一定稳定训练作用） |
| **Mask proxy+aux loss** |  | `sep_layer_k` 原始激活 + mixture subband STFT | activation→proj_proxy(256) + proj_spk(256) = 512 | ~1.88M + shared encoder | 增加了基于余弦相似度的说话人损失 | sep_layer_2: 63.6%           sep_layer_3: 64.1%          sep_layer_4: 65.5% | 相较于mask proxy感觉没有什么变化 |

1.目前得到的明确结论似乎仅有： sep的深层更有利于进行分类

2.训练的分类头都还没有达到期望的水平，几个消融目前没有展现出明显影响。