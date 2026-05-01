from __future__ import print_function

import os
import time
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

import pandas as pd

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


# ========================================================
# 🚀 核心替换：使用相对 L2 绝对误差来丈量前端 Embedding
# ========================================================
def calc_rel_distance(feat_est, feat_ref):
    """
    计算推断特征与参考特征之间的相对 L2 距离。
    """
    diff = feat_est - feat_ref
    error_norm = torch.norm(diff, p=2)
    ref_norm = torch.norm(feat_ref, p=2)
    rel_dist = (error_norm / (ref_norm + 1e-8)).item()
    return rel_dist


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

    logger = get_logger(configs["exp_dir"], "infer_prior_l2_distance.log")
    logger.info(f"Load checkpoint from {model_path}")

    model = model.to(device)
    model.eval()

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

    with open(configs["test_data"], "r", encoding="utf-8") as f:
        test_iter = sum(1 for _ in f)
    logger.info(f"Test number: {test_iter}")

    try:
        spk_conf_all = configs["model_args"]["tse_model"]["speaker"]
        spk_model_conf = spk_conf_all.get("speaker_model", None)

        if spk_model_conf is None:
            raise ValueError("配置中缺少 speaker_model，请检查 yaml 层级！")

        fbank_extractor = Fbank_kaldi(**spk_model_conf['fbank']).to(device)
        spk_extractor = SpeakerEncoder(
            spk_model_conf['speaker_encoder']).to(device)

        fbank_extractor.eval()
        spk_extractor.eval()
        logger.info("✅ 成功初始化外部 Fbank 与 SpeakerEncoder 作为先验探针！")
    except Exception as e:
        logger.error(f"初始化声纹探针失败，错误信息: {e}")
        return

    plot_data_list = []

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            # if i >= 1000: break

            mix, cues, target = extract_model_inputs(batch, device)
            key = batch["key"]

            mix_spk1 = mix[0:1, :, :]
            target_spk1 = target[0:1, :, :] if target.dim() == 3 else target[
                0:1, 0:1, :]
            interf_spk1 = target[1:2, :, :] if target.dim() == 3 else target[
                1:2, 0:1, :]

            cues_spk1 = [cues[0][0:1, :]] if cues is not None else None
            if cues_spk1 is None or cues_spk1[0].shape[-1] == 0:
                continue

            ref_1 = target_spk1.detach().cpu().numpy().flatten()
            mix_1 = mix_spk1.detach().cpu().numpy().flatten()

            # ==========================================
            # 👼 提取 Enrollment, Target, 以及 Interferer 的 Embedding
            # ==========================================
            try:
                enroll_audio = cues_spk1[0].view(1, -1)
                target_audio = target_spk1.view(1, -1)
                interf_audio = interf_spk1.view(1, -1)

                enroll_fb = fbank_extractor(enroll_audio)
                target_fb = fbank_extractor(target_audio)
                interf_fb = fbank_extractor(interf_audio)

                emb_enroll = spk_extractor(enroll_fb)
                emb_target = spk_extractor(target_fb)
                emb_interf = spk_extractor(interf_fb)

                if isinstance(emb_enroll, (tuple, list)):
                    emb_enroll = emb_enroll[-1]
                if isinstance(emb_target, (tuple, list)):
                    emb_target = emb_target[-1]
                if isinstance(emb_interf, (tuple, list)):
                    emb_interf = emb_interf[-1]

                emb_enroll = emb_enroll.view(1, -1)
                emb_target = emb_target.view(1, -1)
                emb_interf = emb_interf.view(1, -1)

                # --- 🎯 核心计算：从 Cosine 替换为 L2 距离 ---
                dist_target = calc_rel_distance(emb_enroll, emb_target)
                dist_interf = calc_rel_distance(emb_enroll, emb_interf)

            except Exception as e:
                logger.error(f"样本 {key[0]} 特征提取失败，跳过。错误: {e}")
                continue

            # 获取真实 SI-SNRi
            outputs_dyn = model(mix_spk1, cues_spk1)
            ests_dyn = outputs_dyn[0].detach().cpu().numpy().flatten()
            if np.max(np.abs(ests_dyn)) > 0:
                ests_dyn = ests_dyn / np.max(np.abs(ests_dyn)) * 0.9
            end_idx = min(len(ests_dyn), len(ref_1), len(mix_1))

            _, snri_value = cal_SISNRi(ests_dyn[:end_idx], ref_1[:end_idx],
                                       mix_1[:end_idx])

            plot_data_list.append({
                "Utterance": key[0],
                "Dist_e1_s1": dist_target,
                "Dist_e1_s2": dist_interf,
                "Output_SISNRi": snri_value
            })

            logger.info(
                f"Processed {i+1}/{test_iter} | L2(e1,s1): {dist_target:.3f} | L2(e1,s2): {dist_interf:.3f} | SNRi: {snri_value:+.1f}dB"
            )

    end = time.time()
    logger.info(f"Total Inference Time: {end - start:.1f}s")

    # ========================================================
    # 💾 数据保存 (不画图)
    # ========================================================
    logger.info("💾 正在保存 Prior L2 Distance 数据...")
    df = pd.DataFrame(plot_data_list)
    csv_path = os.path.join(configs["exp_dir"], "prior_l2_distance.csv")
    df.to_csv(csv_path, index=False)

    logger.info(f"🎉 数据已成功保存至: {csv_path}")


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
