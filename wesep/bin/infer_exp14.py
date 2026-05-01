from __future__ import print_function

import os
import time
import types
import json
import random

import fire
import soundfile
import torch
import torch.nn.functional as F
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

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


def infer(config="confs/conf.yaml", **kwargs):
    start = time.time()
    configs = parse_config_or_kwargs(config, **kwargs)

    rank = 0
    set_seed(configs["seed"] + rank)
    gpu = configs["gpus"]
    device = (torch.device("cuda:{}".format(gpu))
              if gpu >= 0 else torch.device("cpu"))

    if 'spk_model_init' in configs['model_args']['tse_model']:
        configs['model_args']['tse_model']['spk_model_init'] = False

    model = get_model(configs["model"]["tse_model"])(
        configs["model_args"]["tse_model"])
    model_path = os.path.join(configs["checkpoint"])
    load_pretrained_model(model, model_path)

    logger = get_logger(configs["exp_dir"], "infer_exp1_features_anchors.log")
    logger.info(f"Load checkpoint from {model_path}")

    save_feat_dir = os.path.join(configs["exp_dir"], "exp1_tsne_features")
    os.makedirs(save_feat_dir, exist_ok=True)

    model = model.to(device)
    model.eval()

    # ========================================================
    # 🚀 初始化外部先验探针 (Speaker Encoder)
    # ========================================================
    try:
        spk_conf_all = configs["model_args"]["tse_model"]["speaker"]
        spk_model_conf = spk_conf_all.get("speaker_model", None)
        fbank_extractor = Fbank_kaldi(**spk_model_conf['fbank']).to(device)
        spk_extractor = SpeakerEncoder(
            spk_model_conf['speaker_encoder']).to(device)
        fbank_extractor.eval()
        spk_extractor.eval()
        logger.info("✅ 成功初始化外部 Fbank 与 SpeakerEncoder 作为 Prior 提取器！")
    except Exception as e:
        logger.error(f"初始化声纹探针失败: {e}")
        return

    test_dataset = Dataset(
        configs["data_type"],
        configs["test_data"],
        configs["dataset_args"],
        state="test",
        repeat_dataset=False,
        cues_yaml=configs.get("test_cues", None),
    )
    test_collect_keys = build_collect_keys(
        load_yaml(configs["test_cues"]),
        configs["dataset_args"],
        BASE_COLLECT_KEYS,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=lambda batch: tse_collate_fn(batch, test_collect_keys))

    # ========================================================
    # 🔪 Hook 注入：同时拦截 Pmap 与 拼接后的 Post_Concat
    # ========================================================
    hooked_features = {}
    target_module = model.module if hasattr(model, 'module') else model

    if hasattr(target_module, 'spk_configs'
               ) and target_module.spk_configs['features']['usef']['enabled']:
        original_usef_post = target_module.spk_ft.usef.post

        def hooked_usef_post(self, mix_repr, feat_repr):
            # 💥 拦截 1：进入融合前的 Pmap
            hooked_features['pmap'] = feat_repr.detach().clone()

            # 执行原本的拼接/融合逻辑
            out_concat = original_usef_post(mix_repr, feat_repr)

            # 💥 拦截 2：融合之后的 Post_Concat
            hooked_features['post_concat'] = out_concat.detach().clone()
            return out_concat

        target_module.spk_ft.usef.post = types.MethodType(
            hooked_usef_post, target_module.spk_ft.usef)
        logger.info("🔪 成功注入 USEF Hook！Pmap 与 Post_Concat 已被同步监听。")

    try:
        with open(
                "/home/yxy05/code/research_tse/examples/audio/librimix/data/clean/test/cues/audio.json",
                "r",
                encoding="utf-8") as f:
            spk_audio_dict = json.load(f)
    except Exception as e:
        return

    # ========================================================
    # 🧠 辅助函数：一键提取三阶段特征并降维
    # ========================================================
    def extract_three_stage_features(audio_tensor, mixture_tensor):
        # 1. Prior
        fb = fbank_extractor(audio_tensor)
        emb = spk_extractor(fb)
        if isinstance(emb, (tuple, list)): emb = emb[-1]
        prior_v = emb.view(-1).detach().cpu().numpy()

        # 2. 跑主干网络触发 Hook
        _ = model(mixture_tensor, [audio_tensor])

        # 3. 截获并 Time Pooling 降维
        pmap_t = hooked_features['pmap'].clone()
        if pmap_t.dim() > 2: pmap_t = pmap_t.mean(dim=-1)
        pmap_v = pmap_t.view(-1).cpu().numpy()

        post_t = hooked_features['post_concat'].clone()
        if post_t.dim() > 2: post_t = post_t.mean(dim=-1)
        post_v = post_t.view(-1).cpu().numpy()

        return prior_v, pmap_v, post_v

    # ========================================================
    # 🏃 推断主循环
    # ========================================================
    num_mix_to_test = 15
    num_samples_per_mix = 50

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            if i >= num_mix_to_test: break

            mix, cues, target = extract_model_inputs(batch, device)
            spk = batch["spk"]
            key = batch["key"][0]
            target_spk_id = str(spk[0])

            mix_spk1 = mix[0:1, :, :]
            target_spk1 = target[0:1, :, :] if target.dim() == 3 else target[
                0:1, 0:1, :]

            # 正确获取干扰源，如果数据集只有单轨道 target，就用 mix 减去 target 得到干扰源
            if target.shape[0] > 1:
                interf_spk1 = target[
                    1:2, :, :] if target.dim() == 3 else target[1:2, 0:1, :]
            else:
                interf_spk1 = mix_spk1 - target_spk1
                print("minus")

            ref_1 = target_spk1.detach().cpu().numpy().flatten()
            mix_1 = mix_spk1.detach().cpu().numpy().flatten()

            available_utts = spk_audio_dict.get(target_spk_id, [])
            if len(available_utts) == 0: continue

            # ==========================================
            # 👑 建立上帝锚点 (Oracle Anchors)
            # ==========================================
            oracle_tgt_prior, oracle_tgt_pmap, oracle_tgt_post = extract_three_stage_features(
                target_spk1.view(1, -1), mix_spk1)
            oracle_int_prior, oracle_int_pmap, oracle_int_post = extract_three_stage_features(
                interf_spk1.view(1, -1), mix_spk1)

            # 顺手把 Oracle 的上限 SNRi 算出来
            out_oracle = model(
                mix_spk1,
                [target_spk1.view(1, -1)])[0].detach().cpu().numpy().flatten()
            if np.max(np.abs(out_oracle)) > 0:
                out_oracle = out_oracle / np.max(np.abs(out_oracle)) * 0.9
            end_s = min(len(out_oracle), len(ref_1), len(mix_1))
            oracle_snri, _ = cal_SISNRi(out_oracle[:end_s], ref_1[:end_s],
                                        mix_1[:end_s])

            logger.info(
                f"🚀 轰炸 Mixture {i+1} [{key}] | Oracle 上限: {oracle_snri:.1f}dB")

            mix_prior_list = []
            mix_pmap_list = []
            mix_post_list = []
            mix_labels_list = []

            for idx in range(num_samples_per_mix):
                # 随机裁剪
                item = random.choice(available_utts)
                wav_path = item["path"]
                try:
                    enroll_wav, sr = soundfile.read(wav_path)
                    total_len = len(enroll_wav)
                    min_len = min(sr * 1, total_len)
                    if total_len > min_len:
                        crop_len = random.randint(min_len, total_len)
                        start_idx = random.randint(0, total_len - crop_len)
                        crop_wav = enroll_wav[start_idx:start_idx + crop_len]
                    else:
                        crop_wav = enroll_wav
                    current_cue = torch.tensor(
                        crop_wav, dtype=torch.float32).to(device).view(1, -1)
                except Exception as e:
                    continue

                # 💡 提取 50 次的随机样本三阶段特征
                prior_v, pmap_v, post_v = extract_three_stage_features(
                    current_cue, mix_spk1)

                # 算成绩与 Delta
                outputs_dynamic = model(mix_spk1, [current_cue])
                out_np_dyn = outputs_dynamic[0].detach().cpu().numpy().flatten(
                )
                if np.max(np.abs(out_np_dyn)) > 0:
                    out_np_dyn = out_np_dyn / np.max(np.abs(out_np_dyn)) * 0.9
                end_d = min(len(out_np_dyn), len(ref_1), len(mix_1))
                dyn_snri, _ = cal_SISNRi(out_np_dyn[:end_d], ref_1[:end_d],
                                         mix_1[:end_d])

                delta_snri = oracle_snri - dyn_snri

                mix_prior_list.append(prior_v)
                mix_pmap_list.append(pmap_v)
                mix_post_list.append(post_v)
                mix_labels_list.append({
                    "Sample_Idx": idx,
                    "Dynamic_SISNRi": dyn_snri,
                    "Delta_SISNRi": delta_snri
                })

            # 💾 保存至文件夹
            mix_dir = os.path.join(save_feat_dir, f"mix_{i+1:03d}_{key}")
            os.makedirs(mix_dir, exist_ok=True)

            # 保存 50 组散点
            np.save(os.path.join(mix_dir, "prior_features.npy"),
                    np.array(mix_prior_list))
            np.save(os.path.join(mix_dir, "pmap_features.npy"),
                    np.array(mix_pmap_list))
            np.save(os.path.join(mix_dir, "post_concat_features.npy"),
                    np.array(mix_post_list))
            pd.DataFrame(mix_labels_list).to_csv(os.path.join(
                mix_dir, "labels.csv"),
                                                 index=False)

            # 👑 保存 Oracle 上帝锚点
            np.savez(os.path.join(mix_dir, "oracle_anchors.npz"),
                     tgt_prior=oracle_tgt_prior,
                     tgt_pmap=oracle_tgt_pmap,
                     tgt_post=oracle_tgt_post,
                     int_prior=oracle_int_prior,
                     int_pmap=oracle_int_pmap,
                     int_post=oracle_int_post)

            logger.info(f"   └── 保存完毕: 三阶段特征矩阵 + Oracle Anchors -> {mix_dir}")

    end = time.time()
    logger.info(f"🎉 全部特征及锚点提取完成！耗时: {end - start:.1f}s.")


def extract_model_inputs(batch, device):
    if "wav_mix" not in batch: raise RuntimeError("Missing wav_mix")
    if "wav_target" not in batch: raise RuntimeError("Missing wav_target")
    mix = batch["wav_mix"].float().to(device)
    target = batch["wav_target"].float().to(device)
    cues = [
        batch[k].float().to(device) for k in AUX_KEY_MAP.values()
        if k in batch and batch[k] is not None
    ]
    return mix, cues if len(cues) > 0 else None, target


if __name__ == "__main__":
    fire.Fire(infer)
