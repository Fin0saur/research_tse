"""
基于原始推理逻辑的：在线 CNN 探针训练脚本
直接在 BSRNN 推理的同时，将提取到的特征送入后接的探针进行训练。
"""

from __future__ import print_function

import os
import time
import types
import fire
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from wesep.dataset.dataset import Dataset
from wesep.dataset.collate import BASE_COLLECT_KEYS, build_collect_keys, tse_collate_fn, AUX_KEY_MAP
import numpy as np
import pandas as pd
from wesep.models import get_model
from wesep.utils.checkpoint import load_pretrained_model
from wesep.utils.score import cal_SISNRi
from wesep.utils.utils import get_logger, parse_config_or_kwargs, set_seed
from wesep.utils.file_utils import load_yaml
from wesep.modules.speaker.encoder import Fbank_kaldi, SpeakerEncoder

import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# ==========================================
# 1. 探针网络定义 (直接贴进你的脚本)
# ==========================================
class PmapCNNProbe(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.pool_t = nn.AdaptiveAvgPool2d((32, 1))
        self.conv1d = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.pool_c = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, pmap):
        # 确保 pmap 形状为 [B, 32, 128, T]
        x = pmap.permute(0, 2, 1, 3) # -> [B, 128, 32, T]
        x = self.conv2d(x)
        x = self.pool_t(x).squeeze(-1)
        x = self.conv1d(x)
        x = self.pool_c(x).squeeze(-1)
        return self.classifier(x)

# ==========================================
# 2. 爆改后的在线训练主函数
# ==========================================
def train_online_probe(
    config="confs/conf.yaml",
    dataset_split="train",  # 改为默认跑 train
    epochs=10,
    lr=1e-3,
    accum_steps=16, # 梯度累加，模拟 batch_size=16
    **kwargs
):
    configs = parse_config_or_kwargs(config, **kwargs)
    set_seed(configs["seed"])
    gpu = configs["gpus"]
    device = torch.device(f"cuda:{gpu}" if gpu >= 0 else "cpu")

    if 'spk_model_init' in configs['model_args']['tse_model']:
        configs['model_args']['tse_model']['spk_model_init'] = False

    # -------------- 初始化冻结的 BSRNN --------------
    model = get_model(configs["model"]["tse_model"])(configs["model_args"]["tse_model"])
    load_pretrained_model(model, configs["checkpoint"])
    model = model.to(device)
    
    # 【极其重要】：彻底冻结 BSRNN，我们只训练探针！
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    spk_model_conf = configs["model_args"]["tse_model"]["speaker"].get("speaker_model", None)
    fbank_extractor = Fbank_kaldi(**spk_model_conf['fbank']).to(device)
    spk_extractor = SpeakerEncoder(spk_model_conf['speaker_encoder']).to(device)
    for param in fbank_extractor.parameters(): param.requires_grad = False
    for param in spk_extractor.parameters(): param.requires_grad = False
    fbank_extractor.eval()
    spk_extractor.eval()

    # -------------- 初始化我们要训练的探针 --------------
    probe = PmapCNNProbe(num_classes=2).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    
    # 类别不平衡补偿：给混淆(C1)样本更高的权重
    class_weights = torch.tensor([0.1, 0.9]).to(device) 
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    logger = get_logger(configs["exp_dir"], f"online_probe_train.log")
    logger.info("🚀 启动在线 CNN 探针训练！BSRNN 已冻结。")

    # -------------- 数据集配置 --------------
    data_file_key, cues_file_key = "train_data", "train_cues"
    dataset = Dataset(configs["data_type"], configs[data_file_key], configs["dataset_args"],
                      state=dataset_split, repeat_dataset=False, cues_yaml=configs.get(cues_file_key, None))
    collect_keys = build_collect_keys(load_yaml(configs[cues_file_key]), configs["dataset_args"], BASE_COLLECT_KEYS)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda batch: tse_collate_fn(batch, collect_keys))

    # -------------- 你的 Hook 逻辑 --------------
    hooked_features = {}
    target_module = model.module if hasattr(model, 'module') else model
    original_usef_post = target_module.spk_ft.usef.post

    def hooked_usef_post(self, mix_repr, feat_repr):
        # 【爆改点】：不再转 cpu()，直接保留在 GPU 上，保留梯度所需信息
        hooked_features['pmap'] = feat_repr.detach() 
        out_concat = original_usef_post(mix_repr, feat_repr)
        return out_concat

    target_module.spk_ft.usef.post = types.MethodType(hooked_usef_post, target_module.spk_ft.usef)

    # -------------- 开始 Epoch 循环 --------------
    for epoch in range(epochs):
        probe.train() # 探针进入训练模式
        total_loss = 0
        optimizer.zero_grad()
        
        for i, batch in enumerate(dataloader):
            mix = batch["wav_mix"][0:1].float().to(device)
            target = batch["wav_target"][0:1].float().to(device)
            
            # 提取 cue 逻辑 (与你原来一模一样)
            cue_key = "audio_aux" if ("audio_aux" in batch and batch["audio_aux"] is not None) else list(AUX_KEY_MAP.values())[0]
            cue = batch[cue_key].float().to(device)
            
            if cue.shape[-1] < 400 or target.shape[-1] < 400 or mix.shape[-1] < 400: continue
            if cue.dim() == 3: cue = cue[0, 0, :]
            cue = cue.unsqueeze(0)

            # ==========================================
            # 第一步：过一遍 BSRNN 获取动态结果和特征 (不计算梯度)
            # ==========================================
            with torch.no_grad():
                _ = model(mix, [cue]) # 触发 hook
                out_dynamic = model(mix, [cue])[0].detach().cpu().numpy().flatten()
                
                # 算分打标签 (C0 成功=0, C1 失败=1)
                ref_np = target.view(-1).cpu().numpy().flatten()
                mix_np = mix.view(-1).cpu().numpy().flatten()
                end_s = min(len(out_dynamic), len(ref_np), len(mix_np))
                dyn_snri, _ = cal_SISNRi(out_dynamic[:end_s], ref_np[:end_s], mix_np[:end_s])
                
                # 标签：如果 SISNRi < 1.0，则认为是失败(1)，否则成功(0)
                label = torch.tensor([1 if dyn_snri < 1.0 else 0], dtype=torch.long).to(device)

            # ==========================================
            # 第二步：将特征送入探针进行训练 (计算梯度)
            # ==========================================
            pmap_t = hooked_features.get('pmap', None)
            if pmap_t is None: continue
            
            # pmap_t 维度是 [1, 32, 128, T]
            logits = probe(pmap_t)
            
            # 计算 Loss (加入梯度累加逻辑)
            loss = criterion(logits, label)
            loss = loss / accum_steps
            loss.backward()

            total_loss += loss.item() * accum_steps # 记录真实 loss
            
            # 累加满足条件后，更新参数
            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()


            logger.info(f"Epoch [{epoch}/{epochs}] Step [{i+1}] | Loss: {loss.item()*accum_steps:.4f} | Dynamic SISNRi: {dyn_snri:.2f} -> Label: {label.item()}")

        # Epoch 结束保存模型
        save_path = os.path.join(configs["exp_dir"], f"cnn_probe_epoch_{epoch}.pt")
        torch.save(probe.state_dict(), save_path)
        logger.info(f"💾 Epoch {epoch} 训练完成，模型已保存至 {save_path}")

if __name__ == "__main__":
    fire.Fire(train_online_probe)