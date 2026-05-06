"""
收集训练/验证/测试集上的推理中间特征 (prior, pmap, post_concat)
专为 CNN Probe (Direct Probe) 设计：
1. 保留完整的时空维度 [C, D, T]，不进行平均或展平。
2. 采用 "一音频一文件" 的保存策略解决时间维度变长问题。
"""

from __future__ import print_function

import os
import time
import types
import fire
import torch
from torch.utils.data import DataLoader

from wesep.dataset.dataset import Dataset
from wesep.dataset.collate import (
    BASE_COLLECT_KEYS,
    build_collect_keys,
    tse_collate_fn,
    AUX_KEY_MAP,
)
import numpy as np
import pandas as pd
from wesep.models import get_model
from wesep.utils.checkpoint import load_pretrained_model
from wesep.utils.score import cal_SISNRi
from wesep.utils.utils import (
    get_logger,
    parse_config_or_kwargs,
    set_seed,
)
from wesep.utils.file_utils import load_yaml

from wesep.modules.speaker.encoder import Fbank_kaldi, SpeakerEncoder

import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


def collect_features_for_cnn_probe(
    config="confs/conf.yaml",
    dataset_split="test",  # "train", "val", "test"
    num_shards=1,
    shard_id=0,
    **kwargs
):
    start = time.time()
    configs = parse_config_or_kwargs(config, **kwargs)

    rank = 0
    set_seed(configs["seed"] + rank)
    gpu = configs["gpus"]
    device = (torch.device("cuda:{}".format(gpu))
              if gpu >= 0 else torch.device("cpu"))

    if 'spk_model_init' in configs['model_args']['tse_model']:
        configs['model_args']['tse_model']['spk_model_init'] = False

    # 初始化模型
    model = get_model(configs["model"]["tse_model"])(
        configs["model_args"]["tse_model"])
    model_path = os.path.join(configs["checkpoint"])
    load_pretrained_model(model, model_path)

    logger = get_logger(configs["exp_dir"], f"collect_feat_{dataset_split}_shard_{shard_id}.log")
    logger.info(
        f"🚀 [Shard {shard_id}/{num_shards}] 收集 {dataset_split} 原始高维特征 (For CNN Probe)，Load checkpoint from {model_path}"
    )

    # ==========================================
    # 创建特征保存目录结构
    # ==========================================
    base_save_dir = os.path.join(configs["exp_dir"], f"{dataset_split}_features_cnn")
    tensors_dir = os.path.join(base_save_dir, "tensors")
    os.makedirs(tensors_dir, exist_ok=True)

    model = model.to(device)
    model.eval()

    # 初始化声纹探针
    spk_conf_all = configs["model_args"]["tse_model"]["speaker"]
    spk_model_conf = spk_conf_all.get("speaker_model", None)
    fbank_extractor = Fbank_kaldi(**spk_model_conf['fbank']).to(device)
    spk_extractor = SpeakerEncoder(
        spk_model_conf['speaker_encoder']).to(device)
    fbank_extractor.eval()
    spk_extractor.eval()

    # 选择数据集
    data_key_map = {
        "train": ("train_data", "train_cues"),
        "val": ("val_data", "val_cues"),
        "test": ("test_data", "test_cues"),
    }
    data_file_key, cues_file_key = data_key_map[dataset_split]

    dataset = Dataset(
        configs["data_type"],
        configs[data_file_key],
        configs["dataset_args"],
        state=dataset_split,
        repeat_dataset=False,
        cues_yaml=configs.get(cues_file_key, None),
    )
    collect_keys = build_collect_keys(
        load_yaml(configs[cues_file_key]),
        configs["dataset_args"],
        BASE_COLLECT_KEYS,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=lambda batch: tse_collate_fn(batch, collect_keys))

    # Hook 机制：捕获 pmap 和 post_concat
    hooked_features = {}
    target_module = model.module if hasattr(model, 'module') else model

    if hasattr(target_module, 'spk_configs') and target_module.spk_configs['features']['usef']['enabled']:
        original_usef_post = target_module.spk_ft.usef.post

        def hooked_usef_post(self, mix_repr, feat_repr):
            # 保留完整的维度，不破坏计算图 (但为了存储，我们存 detached cpu 副本)
            hooked_features['pmap'] = feat_repr.detach().cpu()
            out_concat = original_usef_post(mix_repr, feat_repr)
            hooked_features['post_concat'] = out_concat.detach().cpu()
            return out_concat

        target_module.spk_ft.usef.post = types.MethodType(
            hooked_usef_post, target_module.spk_ft.usef)

    # 获取总数
    with open(configs[data_file_key], "r", encoding="utf-8") as f:
        total_samples = sum(1 for _ in f)
    logger.info(
        f"🔥 Total: {total_samples} samples. "
        f"当前分片只处理 i % {num_shards} == {shard_id} 的样本。"
    )

    all_metadata = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # 分片逻辑
            if i % num_shards != shard_id:
                continue

            mix = batch["wav_mix"][0:1].float().to(device)
            target = batch["wav_target"][0:1].float().to(device)
            spk = batch["spk"]
            key = batch["key"][0]

            # 提取 cue
            cue_key = "audio_aux"
            if cue_key not in batch or batch[cue_key] is None:
                cue_key = None
                for k in AUX_KEY_MAP.values():
                    if k in batch and batch[k] is not None:
                        cue_key = k
                        break
                if cue_key is None:
                    logger.warning(f"[{i}] Batch {key} 缺少 cue，跳过")
                    continue
            cue = batch[cue_key].float().to(device)

            # 检查音频长度
            min_len = 400
            if cue.shape[-1] < min_len or target.shape[-1] < min_len or mix.shape[-1] < min_len:
                logger.warning(f"[{i}] Batch {key} 音频太短，跳过")
                continue

            if cue.dim() == 3:
                cue = cue[0, 0, :]
            cue = cue.unsqueeze(0)

            # -----------------------------------------------------
            # 1. 提取 Prior (Speaker Embedding) -> 形状 [192]
            # -----------------------------------------------------
            fb = fbank_extractor(cue)
            emb = spk_extractor(fb)
            if isinstance(emb, (tuple, list)):
                emb = emb[-1]
            # 挤掉 batch 维度，变成一维向量
            prior_t = emb.squeeze(0).cpu() 

            # -----------------------------------------------------
            # 2. 计算 SI-SNRi 分数
            # -----------------------------------------------------
            out_oracle = model(mix, [target])[0].detach().cpu().numpy().flatten()
            if np.max(np.abs(out_oracle)) > 0:
                out_oracle = out_oracle / np.max(np.abs(out_oracle)) * 0.9

            _ = model(mix, [cue]) # 触发 hook
            out_dynamic = model(mix, [cue])[0].detach().cpu().numpy().flatten()
            if np.max(np.abs(out_dynamic)) > 0:
                out_dynamic = out_dynamic / np.max(np.abs(out_dynamic)) * 0.9

            ref_np = target.view(-1).cpu().numpy().flatten()
            mix_np = mix.view(-1).cpu().numpy().flatten()

            end_s = min(len(out_oracle), len(ref_np), len(mix_np))
            oracle_snri, _ = cal_SISNRi(out_oracle[:end_s], ref_np[:end_s], mix_np[:end_s])
            dyn_snri, _ = cal_SISNRi(out_dynamic[:end_s], ref_np[:end_s], mix_np[:end_s])

            # -----------------------------------------------------
            # 3. 提取 Pmap 和 Post_Concat (保留原始维度) -> 形状 [32, 128, T]
            # -----------------------------------------------------
            pmap_t = hooked_features.get('pmap', None)
            if pmap_t is None:
                logger.warning(f"[{i}] Batch {key} 未捕获 pmap，跳过")
                continue
            
            # 剥离 Batch 维度: [1, 32, 128, T] -> [32, 128, T]
            pmap_t = pmap_t.squeeze(0).half() # 转为 float16 节省硬盘空间 (可选)

            post_t = hooked_features.get('post_concat', None)
            post_t = post_t.squeeze(0).half() # 转为 float16

            # -----------------------------------------------------
            # 4. 一音频一文件落盘策略
            # -----------------------------------------------------
            # 为了减少文件数量，把这三个张量打包成一个字典保存
            tensor_dict = {
                "prior": prior_t,
                "pmap": pmap_t,
                "post_concat": post_t
            }
            
            tensor_filename = f"{key}.pt"
            tensor_filepath = os.path.join(tensors_dir, tensor_filename)
            torch.save(tensor_dict, tensor_filepath)

            # -----------------------------------------------------
            # 5. 记录元数据
            # -----------------------------------------------------
            all_metadata.append({
                "key": key,
                "spk_id": str(spk[0]) if isinstance(spk[0], int) else spk[0],
                "Oracle_SISNRi": oracle_snri,
                "Dynamic_SISNRi": dyn_snri,
                "Delta_SISNRi": oracle_snri - dyn_snri,
                "tensor_path": os.path.join("tensors", tensor_filename) # 记录相对路径
            })
            
            if (i + 1) % 50 == 0:
                logger.info(f"   [Shard {shard_id}] Processed {i + 1}/{total_samples}")

    # 汇总保存标签文件
    logger.info(f"💾 保存 Labels 到 {base_save_dir} ...")
    pd.DataFrame(all_metadata).to_csv(os.path.join(base_save_dir, "labels.csv"), index=False)

    end = time.time()
    logger.info(
        f"✅ [Shard {shard_id}] {dataset_split} 原始高维特征收集完成！\n"
        f"共 {len(all_metadata)} 个样本，保存至 {base_save_dir}，耗时: {end - start:.1f}s"
    )

if __name__ == "__main__":
    fire.Fire(collect_features_for_cnn_probe)