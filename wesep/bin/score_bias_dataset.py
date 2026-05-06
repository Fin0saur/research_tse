"""
全量数据预打分脚本 (Pre-scoring for Joint Training)
目的：用冻结的 BSRNN 跑遍偏差数据集，计算 SI-SNRi，
并打上 0(成功) 或 1(失败) 的标签，生成 scored.jsonl 供联合训练使用。

适配 bias_enroll20.jsonl 格式：
  - key: 样本 ID
  - spk: [spk1_id, spk2_id]
  - mix: {"default": [mix_wav_path]}
  - src: {spk_id: [src_wav_path], ...}
  - audio_spk1: 注册音频路径 for spk1
  - audio_spk2: 注册音频路径 for spk2
"""

from __future__ import print_function

import os
import time
import json
import fire
import torch
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

from wesep.models import get_model
from wesep.utils.checkpoint import load_pretrained_model
from wesep.utils.score import cal_SISNRi
from wesep.utils.utils import get_logger, parse_config_or_kwargs, set_seed

import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


class BiasDataset(Dataset):
    """
    直接加载 bias_enroll20.jsonl 格式的数据集
    """

    def __init__(self, jsonl_path):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def load_audio(path, expected_sr=16000):
    """加载音频文件，返回 [1, T] tensor"""
    try:
        audio, sr = sf.read(path, dtype='float32')
        # soundfile 返回 numpy array
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)  # [1, T]
        else:
            audio = audio[:, 0:1].T  # [T, C] -> [1, T], 取第一通道
        # 重采样到 expected_sr
        if sr != expected_sr:
            import librosa
            audio = librosa.resample(audio.squeeze(0), orig_sr=sr, target_sr=expected_sr)
            audio = np.expand_dims(audio, axis=0)
        # 转换为 tensor
        audio = torch.from_numpy(audio.astype(np.float32))
        return audio, sr
    except Exception as e:
        print(f"Warning: Failed to load {path}: {e}")
        return None, None


def score_dataset(
    config="confs/conf.yaml",
    input_jsonl="data/train-100_bias/bias_enroll20.jsonl",
    output_jsonl="data/train-100_bias/train_scored.jsonl",
    num_shards=1,
    shard_id=0,
    **kwargs
):
    start = time.time()
    configs = parse_config_or_kwargs(config, **kwargs)
    set_seed(configs["seed"] + shard_id)
    gpu = configs["gpus"]
    device = torch.device(f"cuda:{gpu}" if gpu >= 0 else "cpu")

    logger = get_logger(configs["exp_dir"], f"scoring_shard_{shard_id}.log")
    logger.info(f"🚀 [Shard {shard_id}/{num_shards}] 启动预打分引擎...")

    # ==========================================
    # 1. 初始化并彻底冻结 BSRNN
    # ==========================================
    if 'spk_model_init' in configs['model_args']['tse_model']:
        configs['model_args']['tse_model']['spk_model_init'] = False
    model = get_model(configs["model"]["tse_model"])(configs["model_args"]["tse_model"])
    load_pretrained_model(model, configs["checkpoint"])
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # ==========================================
    # 2. 准备数据集
    # ==========================================
    logger.info(f"📄 加载原始 JSONL: {input_jsonl}")
    dataset = BiasDataset(input_jsonl)
    total_samples = len(dataset)
    logger.info(f"📦 共有 {total_samples} 条数据需要打分。")

    # 多卡分片：计算当前 shard 要处理的样本索引
    if num_shards > 1:
        shard_indices = [i for i in range(total_samples) if i % num_shards == shard_id]
    else:
        shard_indices = list(range(total_samples))

    logger.info(f"   当前分片处理 {len(shard_indices)} 条样本 (shard {shard_id}/{num_shards})")

    # ==========================================
    # 3. 开始前向推理与打分
    # ==========================================
    shard_output_path = f"{output_jsonl}.part{shard_id}" if num_shards > 1 else output_jsonl

    scored_count = 0
    c0_count = 0
    c1_count = 0
    skipped_count = 0

    with open(shard_output_path, "w", encoding="utf-8") as out_f, torch.no_grad():
        for local_i, global_idx in enumerate(shard_indices):
            record = dataset[global_idx]
            key = record.get("key", f"unknown_{global_idx}")
            spk_ids = record.get("spk", [])
            if isinstance(spk_ids, str):
                spk_ids = [spk_ids]

            # 加载 mix 音频
            mix_paths = record.get("mix", {}).get("default", [])
            if not mix_paths:
                logger.warning(f"[{global_idx}] {key} 缺少 mix 路径，跳过")
                skipped_count += 1
                continue
            mix_wav, _ = load_audio(mix_paths[0])
            if mix_wav is None:
                logger.warning(f"[{global_idx}] {key} mix 加载失败，跳过")
                skipped_count += 1
                continue
            mix = mix_wav.to(device)

            # 加载 target 音频 (第一个说话人的源音频)
            src_dict = record.get("src", {})
            target_path = None
            if spk_ids and spk_ids[0] in src_dict:
                src_paths = src_dict[spk_ids[0]]
                if src_paths:
                    target_path = src_paths[0]
            if target_path is None:
                logger.warning(f"[{global_idx}] {key} 缺少 target 路径，跳过")
                skipped_count += 1
                continue
            target_wav, _ = load_audio(target_path)
            if target_wav is None:
                logger.warning(f"[{global_idx}] {key} target 加载失败，跳过")
                skipped_count += 1
                continue
            target = target_wav.to(device)

            # 加载 enrollment 音频 (audio_spk1, audio_spk2, ...)
            enrollment_list = []
            for spk_idx, spk_id in enumerate(spk_ids, start=1):
                enroll_key = f"audio_spk{spk_idx}"
                enroll_path = record.get(enroll_key)
                if enroll_path is None:
                    logger.warning(f"[{global_idx}] {key} 缺少 {enroll_key}，跳过")
                    enrollment_list = None
                    break
                enroll_wav, _ = load_audio(enroll_path)
                if enroll_wav is None:
                    logger.warning(f"[{global_idx}] {key} {enroll_key} 加载失败，跳过")
                    enrollment_list = None
                    break
                enrollment_list.append(enroll_wav.squeeze(0))  # [T]

            if enrollment_list is None:
                skipped_count += 1
                continue

            # 长度校验
            min_len = 400
            for enroll in enrollment_list:
                if enroll.shape[-1] < min_len:
                    logger.warning(f"[{global_idx}] {key} enrollment 太短，跳过")
                    skipped_count += 1
                    enrollment_list = None
                    break
            if enrollment_list is None:
                continue
            if target.shape[-1] < min_len or mix.shape[-1] < min_len:
                logger.warning(f"[{global_idx}] {key} target/mix 太短，跳过")
                skipped_count += 1
                continue

            # enrollment 长度可能不同，直接取第一个
            cue = enrollment_list[0].unsqueeze(0).to(device)  # [1, T]

            # 推理
            try:
                out_dynamic = model(mix, [cue])[0].cpu().numpy().flatten()
                if np.max(np.abs(out_dynamic)) > 0:
                    out_dynamic = out_dynamic / np.max(np.abs(out_dynamic)) * 0.9

                ref_np = target.view(-1).cpu().numpy().flatten()
                mix_np = mix.view(-1).cpu().numpy().flatten()

                # 算分
                end_s = min(len(out_dynamic), len(ref_np), len(mix_np))
                dyn_snri, _ = cal_SISNRi(out_dynamic[:end_s], ref_np[:end_s], mix_np[:end_s])

                # 打标签: C0 (成功) = dyn_snri >= 1.0, C1 (失败) = dyn_snri < 1.0
                label = 1 if dyn_snri < 1.0 else 0

                record["Dynamic_SISNRi"] = float(dyn_snri)
                record["label"] = label

                if label == 1:
                    c1_count += 1
                else:
                    c0_count += 1

            except Exception as e:
                logger.warning(f"[{global_idx}] {key} 推理失败: {e}，跳过")
                skipped_count += 1
                continue

            # 写入结果
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            scored_count += 1


            logger.info(f"   [Shard {shard_id}] 进度: {scored_count}/{len(shard_indices)} | C0={c0_count}, C1={c1_count}")

    end = time.time()
    logger.info(f"✅ [Shard {shard_id}] 打分完成！耗时: {(end - start)/3600:.2f} 小时。")
    logger.info(f"   统计 -> C0(成功): {c0_count} | C1(失败/混淆): {c1_count} | 跳过: {skipped_count}")
    logger.info(f"   分片文件已保存至: {shard_output_path}")


if __name__ == "__main__":
    fire.Fire(score_dataset)