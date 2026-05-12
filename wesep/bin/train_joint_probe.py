# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
#
# Module C: Three Independent Probes for TSE Quality Prediction (Joint Fine-Tuning Mode)
# 解冻主干网络浅层，分别独立训练 prior/pmap/post_concat 三个探针，并利用探针梯度微调主干

from __future__ import print_function

import os
import types
import json
import random
import time
from pprint import pformat

import fire
import numpy as np
import soundfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import auraloss
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# [🚨 新增] 引入 sklearn 的 AUC 计算
from sklearn.metrics import f1_score, roc_auc_score 

import wesep.utils.schedulers as schedulers
from wesep.dataset.dataset import Dataset
from wesep.dataset.collate import (
    BASE_COLLECT_KEYS,
    build_collect_keys,
    tse_collate_fn,
    AUX_KEY_MAP,
)
from wesep.models import get_model
from wesep.utils.checkpoint import load_pretrained_model, save_checkpoint
from wesep.utils.executor import Executor
from wesep.utils.utils import parse_config_or_kwargs, set_seed, setup_logger, get_logger
from wesep.utils.file_utils import load_yaml
from wesep.modules.speaker.encoder import Fbank_kaldi

import tableprint as tp

MAX_NUM_log_files = 100


# ========================================================
# 三种探针网络结构（保持不变）
# ========================================================
class PriorProbe(nn.Module):
    def __init__(self, input_dim=192, hidden_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
    def forward(self, prior_emb): return self.proj(prior_emb)

class PmapProbe(nn.Module):
    def __init__(self, in_channels=128, f_dim=257):
        super().__init__()
        self.f_dim = f_dim
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=3, padding=1), nn.BatchNorm2d(192), nn.ReLU(),
            nn.Conv2d(192, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.pool_t = nn.AdaptiveAvgPool2d((f_dim, 1))
        self.conv1d = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU()
        )
        self.pool_c = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2)
        )
    def forward(self, pmap):
        x = pmap.permute(0, 1, 3, 2).contiguous()
        x = self.conv2d(x)
        x = self.pool_t(x).squeeze(-1)
        x = self.conv1d(x)
        x = self.pool_c(x).squeeze(-1)
        return self.classifier(x)

class PostConcatProbe(nn.Module):
    def __init__(self, in_channels=256, f_dim=257):
        super().__init__()
        self.f_dim = f_dim
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=3, padding=1), nn.BatchNorm2d(192), nn.ReLU(),
            nn.Conv2d(192, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.pool_t = nn.AdaptiveAvgPool2d((f_dim, 1))
        self.conv1d = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU()
        )
        self.pool_c = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2)
        )
    def forward(self, post_concat):
        x = post_concat.permute(0, 1, 3, 2).contiguous()
        x = self.conv2d(x)
        x = self.pool_t(x).squeeze(-1)
        x = self.conv1d(x)
        x = self.pool_c(x).squeeze(-1)
        return self.classifier(x)


# ========================================================
# 平衡采样器
# ========================================================
class BalancedSamplerWrapper:
    def __init__(self, labels_list, world_size=1, local_rank=0, seed=42):
        self.labels = np.array(labels_list)
        self.class_counts = np.bincount(self.labels)
        self.class_weights = 1.0 / self.class_counts
        self.sample_weights = self.class_weights[self.labels]

        g = torch.Generator()
        g.manual_seed(seed + local_rank)
        
        num_samples_per_gpu = len(self.sample_weights) // max(1, world_size)

        self.sampler = WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=num_samples_per_gpu,
            replacement=True,
            generator=g
        )

    def get_sampler(self):
        return self.sampler


# ========================================================
# 数据集封装
# ========================================================
class ScoredDataset(torch.utils.data.Dataset):
    def __init__(self, scored_jsonl_files, spk2id_dict=None):
        self.samples = []
        self.labels = []
        for part_file in scored_jsonl_files:
            with open(part_file, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    self.samples.append(item)
                    self.labels.append(item.get('label', 0))
        self.labels = np.array(self.labels)
        print(f"[ScoredDataset] Loaded {len(self.samples)} samples. C0={np.sum(self.labels==0)}, C1={np.sum(self.labels==1)}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        mix_wav, _ = soundfile.read(item['mix']['default'][0], dtype='float32')
        cue_wav, _ = soundfile.read(item['audio_spk1'], dtype='float32')
        target_wav, _ = soundfile.read(item['src'][item['spk'][0]][0], dtype='float32')
        return {
            'mix': torch.tensor(mix_wav),
            'cue': torch.tensor(cue_wav),
            'target': torch.tensor(target_wav),
            'label': item.get('label', 0),
            'key': item['key']
        }

def pad_collate_fn(batch, chunk_len=48000):
    mix_batch, cue_batch, target_batch, labels, keys = [], [], [], [], []
    for item in batch:
        mix_batch.append(item['mix'][:chunk_len])
        cue_batch.append(item['cue'][:chunk_len])
        target_batch.append(item['target'][:chunk_len])
        labels.append(item['label'])
        keys.append(item['key'])

    mix_max = max(x.shape[0] for x in mix_batch)
    cue_max = max(x.shape[0] for x in cue_batch)
    tgt_max = max(x.shape[0] for x in target_batch)

    for i in range(len(mix_batch)):
        mix_pad, cue_pad, target_pad = torch.zeros(mix_max), torch.zeros(cue_max), torch.zeros(tgt_max)
        mix_pad[:mix_batch[i].shape[0]] = mix_batch[i]
        cue_pad[:cue_batch[i].shape[0]] = cue_batch[i]
        target_pad[:target_batch[i].shape[0]] = target_batch[i]
        mix_batch[i], cue_batch[i], target_batch[i] = mix_pad, cue_pad, target_pad

    return {
        'mix': torch.stack(mix_batch).unsqueeze(1),
        'cue': torch.stack(cue_batch).unsqueeze(1),
        'target': torch.stack(target_batch).unsqueeze(1),
        'label': torch.tensor(labels, dtype=torch.long),
        'key': keys
    }


def load_speaker_encoder(pretrained_path=None, freeze=True):
    from wespeaker.models.speaker_model import get_speaker_model
    fbank = Fbank_kaldi(num_mel_bins=80, frame_shift=10, frame_length=25, dither=1.0, sample_rate=16000)
    spk_model = get_speaker_model('ECAPA_TDNN_GLOB_c512')(embed_dim=192, feat_dim=80, pooling_func='ASTP')
    if pretrained_path and os.path.exists(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location='cpu')
        state = spk_model.state_dict()
        for key in list(ckpt.keys()):
            if key not in state: del ckpt[key]
        spk_model.load_state_dict(ckpt)
    spk_model.eval()
    if freeze:
        for p in spk_model.parameters(): p.requires_grad = False
    return fbank, spk_model

def extract_prior(fbank, spk_model, cue_audio):
    emb = spk_model(fbank(cue_audio))
    return emb[-1] if isinstance(emb, tuple) else emb


# ========================================================
# 特征截取引擎 
# ========================================================
class FeatureHookManager:
    def __init__(self, model, fbank_extractor, spk_extractor):
        self.model = model
        self.fbank_extractor = fbank_extractor
        self.spk_extractor = spk_extractor
        self.hooked_features = {}
        self._setup_hooks()

    def _setup_hooks(self):
        target_model = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(target_model, 'spk_ft') and hasattr(target_model.spk_ft, 'usef'):
            usef_obj = target_model.spk_ft.usef
            hooked_features = self.hooked_features
            _original_compute = usef_obj.compute
            _original_post = usef_obj.post

            def hooked_compute(_, enroll_spec, mix_spec):
                enroll_out, mix_out = _original_compute(enroll_spec, mix_spec)
                hooked_features['pmap'] = enroll_out 
                return enroll_out, mix_out

            def hooked_post(_, mix_repr, feat_repr):
                out = _original_post(mix_repr, feat_repr)
                hooked_features['post_concat'] = out 
                return out

            usef_obj.compute = types.MethodType(hooked_compute, usef_obj)
            usef_obj.post = types.MethodType(hooked_post, usef_obj)

    def get_hooked_features(self):
        return self.hooked_features.get('pmap'), self.hooked_features.get('post_concat')

    def clear_features(self):
        self.hooked_features.clear()


# ========================================================
# 训练器
# ========================================================
class SingleProbeTrainer:
    def __init__(self, model, probe, hook_manager, device, optimizer, scaler, log_interval=100):
        self.model = model
        self.probe = probe
        self.hook_manager = hook_manager
        self.device = device
        self.optimizer = optimizer
        self.scaler = scaler
        self.log_interval = log_interval
        self.global_step = 0

    def train_epoch(self, dataloader, epoch, logger, max_batches=None):
        self.model.train()  
        self.probe.train()

        losses, sisdr_losses, cls_losses = [], [], []
        all_preds, all_labels = [], []
        sisdr_loss_fn = auraloss.time.SISDRLoss()

        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches: break
            t0 = time.time()
            mix = batch['mix'].float().to(self.device)
            cue = batch['cue'].float().to(self.device)
            target = batch['target'].float().to(self.device)
            labels = batch['label'].long().to(self.device)

            self.hook_manager.clear_features()

            out_wav = self.model(mix, [cue])

            pmap, post_concat = self.hook_manager.get_hooked_features()
            features = None 
            real_probe = self.probe.module if hasattr(self.probe, 'module') else self.probe
            
            if isinstance(real_probe, PmapProbe): features = pmap
            elif isinstance(real_probe, PostConcatProbe): features = post_concat
            elif isinstance(real_probe, PriorProbe):
                features = extract_prior(self.hook_manager.fbank_extractor, self.hook_manager.spk_extractor, cue.squeeze(1)).to(self.device)
            else:
                raise ValueError(f"[Train Epoch] Unknown probe type: {type(real_probe)}")

            if features is None: continue

            logits = self.probe(features)
            loss_cls = F.cross_entropy(logits, labels)
            loss_sisdr = sisdr_loss_fn(out_wav.squeeze(1), target.squeeze(1)).mean()

            lambda_cls = 0.5 
            loss = loss_sisdr + lambda_cls * loss_cls

            losses.append(loss.item())
            sisdr_losses.append(loss_sisdr.item())
            cls_losses.append(loss_cls.item())

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.probe.parameters()), 5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.global_step += 1

            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = sum(losses[-self.log_interval:]) / self.log_interval
                avg_sisdr = sum(sisdr_losses[-self.log_interval:]) / self.log_interval
                avg_cls = sum(cls_losses[-self.log_interval:]) / self.log_interval
                batch_time = time.time() - t0
                
                lr_probe = self.optimizer.param_groups[0]["lr"]
                lr_bsrnn = self.optimizer.param_groups[1]["lr"]
                
                logger.info(tp.row(
                    ("TRAIN", epoch, batch_idx + 1,
                     f"Tot:{avg_loss:.2f} SI:{avg_sisdr:.2f} Cls:{avg_cls:.3f} ({batch_time:.2f}s)",
                     f"P:{lr_probe:.0e}/M:{lr_bsrnn:.0e}"),
                    width=10, style="grid"))

        avg_loss = sum(losses) / len(losses)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        f1 = self._compute_macro_f1(all_preds, all_labels)
        return avg_loss, f1

    def _compute_macro_f1(self, preds, labels):
        return f1_score(labels.numpy(), preds.numpy(), average='macro', zero_division=0)

    @torch.no_grad()
    def evaluate(self, dataloader, logger, max_batches=None):
        self.model.eval()
        self.probe.eval()
        
        all_preds, all_probs, all_labels, sisdr_vals = [], [], [], []
        sisdr_loss_fn = auraloss.time.SISDRLoss()

        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches: break
            mix = batch['mix'].float().to(self.device)
            cue = batch['cue'].float().to(self.device)
            target = batch['target'].float().to(self.device)
            labels = batch['label'].long().to(self.device)

            self.hook_manager.clear_features()
            out_wav = self.model(mix, [cue])

            pmap, post_concat = self.hook_manager.get_hooked_features()
            features = None 
            real_probe = self.probe.module if hasattr(self.probe, 'module') else self.probe
            
            if isinstance(real_probe, PmapProbe): features = pmap
            elif isinstance(real_probe, PostConcatProbe): features = post_concat
            elif isinstance(real_probe, PriorProbe):
                features = extract_prior(self.hook_manager.fbank_extractor, self.hook_manager.spk_extractor, cue.squeeze(1)).to(self.device)
            else:
                raise ValueError(f"[Evaluate] Unknown probe type: {type(real_probe)}")

            if features is None: continue

            logits = self.probe(features)
            
            # [🚨 新增] 记录概率用于 AUC 计算
            probs = torch.softmax(logits, dim=1)[:, 1] 
            all_probs.append(probs.cpu())
            
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            sisdr_val = -sisdr_loss_fn(out_wav.squeeze(1), target.squeeze(1))
            sisdr_vals.append(sisdr_val.mean().item())

        self.probe.train()
        all_preds = torch.cat(all_preds)
        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)
        
        avg_sisdr = sum(sisdr_vals) / len(sisdr_vals) if sisdr_vals else float('nan')
        
        # 1. 常规 0.5 阈值下的 F1
        f1_05 = self._compute_macro_f1(all_preds, all_labels)
        
        labels_np = all_labels.numpy()
        probs_np = all_probs.numpy()

        # 2. [🚨 新增] 计算 ROC-AUC
        try:
            auc = roc_auc_score(labels_np, probs_np)
        except ValueError:
            auc = 0.0  # 防止验证集单一类别报错

        # 3. [🚨 新增] 动态寻找最佳阈值 (只作为参考记录，不影响训练)
        best_f1_search = 0.0
        best_thresh = 0.5
        # 考虑到模型非常激进，我们在 0.5 到 0.99 之间细致搜索
        for thresh in np.arange(0.5, 0.99, 0.01):
            preds_thresh = (probs_np > thresh).astype(int)
            f1_thresh = f1_score(labels_np, preds_thresh, average='macro', zero_division=0)
            if f1_thresh > best_f1_search:
                best_f1_search = f1_thresh
                best_thresh = thresh

        if logger:
            logger.info(f"[EVAL] SISDR: {avg_sisdr:.2f}dB | C0={int((all_labels==0).sum())} C1={int((all_labels==1).sum())}")
            logger.info(f"       => F1(0.5): {f1_05:.4f} | ROC-AUC: {auc:.4f} | Best F1: {best_f1_search:.4f} (at thresh {best_thresh:.2f})")

        return f1_05, auc, best_f1_search, best_thresh, avg_sisdr


# ========================================================
# 主训练函数
# ========================================================
def train_probe(
    config="confs/tse_bsrnn_spk.yaml",
    scored_data_dir="/mnt/code/research_tse/examples/audio/librimix/data/clean/enroll_bias/train-100",
    checkpoint=None,
    ecapa_pretrained=None,
    probe_type="prior",
    num_epochs=50,
    lr_probe=1e-4,     
    lr_bsrnn=1e-6,     
    batch_size=4,
    sample_num_per_epoch=20000,
    eval_data_dir=None,
    eval_max_batches=1000,
    log_interval=10,
    save_dir="exp/probe_single",
    **kwargs
):
    configs = parse_config_or_kwargs(config, **kwargs)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    gpu = int(configs["gpus"].split(',')[rank])
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    dist.init_process_group(backend="nccl")

    probe_save_dir = os.path.join(save_dir, f"joint_probe_{probe_type}")
    os.makedirs(probe_save_dir, exist_ok=True)
    os.makedirs(os.path.join(probe_save_dir, "models"), exist_ok=True)
    logger = get_logger(probe_save_dir, "train.log")

    logger.info(f"<== Joint Training {probe_type} probe & BSRNN ==>")
    set_seed(configs["seed"])

    if 'model_args' in configs and 'tse_model' in configs['model_args']:
        if 'sv_head' in configs['model_args']['tse_model'].get('speaker', {}):
            configs['model_args']['tse_model']['speaker']['sv_head']['enabled'] = False

    model = get_model(configs["model"]["tse_model"])(configs["model_args"]["tse_model"])
    if checkpoint:
        load_pretrained_model(model, checkpoint)
    
    for p in model.parameters(): p.requires_grad = True
    model = model.to(device)
    if world_size > 1: model = DDP(model, device_ids=[gpu], find_unused_parameters=True)

    f_dim = configs['model_args']['tse_model']['separator']['win'] // 2 + 1
    emb_dim = configs['model_args']['tse_model']['separator']['feature_dim'] // 2
    if ecapa_pretrained is None: ecapa_pretrained = './wespeaker_models/voxceleb_ECAPA512/avg_model.pt'
    fbank_extractor, spk_extractor = load_speaker_encoder(ecapa_pretrained, freeze=True)
    fbank_extractor, spk_extractor = fbank_extractor.to(device), spk_extractor.to(device)

    if probe_type == "prior": probe = PriorProbe(input_dim=192, hidden_dim=256).to(device)
    elif probe_type == "pmap": probe = PmapProbe(in_channels=emb_dim, f_dim=f_dim).to(device)
    elif probe_type == "post": probe = PostConcatProbe(in_channels=emb_dim * 2, f_dim=f_dim).to(device)
    for p in probe.parameters(): p.requires_grad = True
    if world_size > 1: probe = DDP(probe, device_ids=[gpu])

    hook_manager = FeatureHookManager(model, fbank_extractor, spk_extractor)

    optimizer = torch.optim.AdamW([
        {'params': probe.parameters(), 'lr': lr_probe},
        {'params': model.parameters(), 'lr': lr_bsrnn}
    ], weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=configs.get("enable_amp", False))

    all_part_files = sorted([os.path.join(scored_data_dir, f) for f in os.listdir(scored_data_dir) if f.startswith('bias_enroll20_scored.jsonl.part')])
    if not all_part_files: all_part_files = sorted([os.path.join(scored_data_dir, f) for f in os.listdir(scored_data_dir) if f.endswith('.jsonl')])
    
    dataset = ScoredDataset(all_part_files)
    
    sampler = BalancedSamplerWrapper(
        labels_list=dataset.labels,
        world_size=world_size,
        local_rank=rank,  
        seed=configs["seed"]
    ).get_sampler()

    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, collate_fn=lambda b: pad_collate_fn(b, chunk_len=48000), pin_memory=True)

    eval_dataloader = None
    if eval_data_dir:
        eval_files = sorted([os.path.join(eval_data_dir, f) for f in os.listdir(eval_data_dir) if f.startswith('bias_enroll20_scored.jsonl.part')])
        if not eval_files: eval_files = sorted([os.path.join(eval_data_dir, f) for f in os.listdir(eval_data_dir) if f.endswith('.jsonl') and 'scored' not in f])
        if eval_files:
            eval_dataset = ScoredDataset(eval_files)
            eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=4, collate_fn=lambda b: pad_collate_fn(b, chunk_len=48000), pin_memory=True, shuffle=True)

    trainer = SingleProbeTrainer(model=model, probe=probe, hook_manager=hook_manager, device=device, optimizer=optimizer, scaler=scaler, log_interval=log_interval)

    # [🚨 修改] 将保存判断标准从 best_f1 改为 best_auc
    best_auc = 0.0  
    
    local_sample_num = sample_num_per_epoch // max(1, world_size)
    max_batches = local_sample_num // batch_size
    
    if rank == 0:
        logger.info(f"Total samples per epoch: {sample_num_per_epoch}")
        logger.info(f"Local samples per GPU: {local_sample_num} (max_batches={max_batches})")
    
    for epoch in range(1, num_epochs + 1):
        if world_size > 1 and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)

        train_loss, train_f1 = trainer.train_epoch(dataloader, epoch, logger, max_batches)

        if rank == 0:
            if eval_dataloader is not None:
                # 接收修改后的返回值
                f1_05, auc, best_f1_search, best_thresh, eval_sisdr = trainer.evaluate(eval_dataloader, logger, eval_max_batches)
                logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
                
                # [🚨 修改] 使用 AUC 作为唯一的保存评价标准
                save_metric = auc 
            else:
                logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
                save_metric = train_f1 # 如果没验证集，勉强用 train_f1

            if save_metric > best_auc:
                best_auc = save_metric
                save_path = os.path.join(probe_save_dir, "models", f"best_{probe_type}_joint.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'probe_state_dict': probe.module.state_dict() if hasattr(probe, 'module') else probe.state_dict(),
                    'best_auc': best_auc
                }, save_path)
                logger.info(f"   => Saved Best Model based on AUC! (AUC: {best_auc:.4f})")

    if world_size > 1: dist.destroy_process_group()
    if rank == 0: logger.info(f"Joint Training completed! Best AUC: {best_auc:.4f}")

if __name__ == "__main__":
    fire.Fire(train_probe)