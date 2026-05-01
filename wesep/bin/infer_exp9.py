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
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

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

    logger = get_logger(configs["exp_dir"], "infer_target_confusion.log")
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

            # --- 🚀 完美修正：直接获取纯净的 Ground Truth 干扰源 ---
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
                interf_audio = interf_spk1.view(1, -1)  # 使用纯净的干扰源

                enroll_fb = fbank_extractor(enroll_audio)
                target_fb = fbank_extractor(target_audio)
                interf_fb = fbank_extractor(interf_audio)

                emb_enroll = spk_extractor(enroll_fb)
                emb_target = spk_extractor(target_fb)
                emb_interf = spk_extractor(interf_fb)

                if isinstance(emb_enroll, tuple) or isinstance(
                        emb_enroll, list):
                    emb_enroll = emb_enroll[-1]
                if isinstance(emb_target, tuple) or isinstance(
                        emb_target, list):
                    emb_target = emb_target[-1]
                if isinstance(emb_interf, tuple) or isinstance(
                        emb_interf, list):
                    emb_interf = emb_interf[-1]

                emb_enroll = emb_enroll.view(1, -1)
                emb_target = emb_target.view(1, -1)
                emb_interf = emb_interf.view(1, -1)

                # --- 分别计算目标相似度与干扰相似度 ---
                sim_target = F.cosine_similarity(emb_enroll,
                                                 emb_target,
                                                 dim=-1).item()
                sim_interf = F.cosine_similarity(emb_enroll,
                                                 emb_interf,
                                                 dim=-1).item()

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

            # 记录用于 2D 散点图的三元组数据
            plot_data_list.append({
                "Utterance": key[0],
                "cos(e1, s1)": sim_target,
                "cos(e1, s2)": sim_interf,
                "Output_SISNRi": snri_value
            })

            # if (i + 1) % 50 == 0:
            logger.info(
                f"Processed {i+1}/{test_iter} | cos(e1,s1): {sim_target:.3f} | cos(e1,s2): {sim_interf:.3f} | SNRi: {snri_value:+.1f}dB"
            )

    end = time.time()
    logger.info(f"Total Inference Time: {end - start:.1f}s")

    # ========================================================
    # 🎨 核心修改 3：复现论文的分 Bin 散点图
    # ========================================================
    logger.info("🎨 正在生成 Target Confusion 分析图...")
    df = pd.DataFrame(plot_data_list)
    df.to_csv(os.path.join(configs["exp_dir"], "target_confusion_data.csv"),
              index=False)

    # 🌟 状态打标 (Binning Logic)
    def categorize_snri(snri):
        if snri > 5.0:
            return "Safe (> 5dB)"
        elif snri > 0.0:
            return "Marginal (0~5dB)"
        else:
            return "Fatal Confusion (< 0dB)"

    df['State'] = df['Output_SISNRi'].apply(categorize_snri)

    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=(9, 7))

    # 映射颜色与形状 (复刻原论文质感)
    palette = {
        "Safe (> 5dB)": "#2ca02c",  # 深绿色
        "Marginal (0~5dB)": "#98df8a",  # 浅绿色
        "Fatal Confusion (< 0dB)": "#d62728"  # 红色预警
    }
    markers = {
        "Safe (> 5dB)": "o",  # 圆点
        "Marginal (0~5dB)": "o",  # 圆点
        "Fatal Confusion (< 0dB)": "*"  # 报警星星
    }

    # 绘制分层散点
    sns.scatterplot(
        data=df,
        x="cos(e1, s1)",
        y="cos(e1, s2)",
        hue="State",
        style="State",
        palette=palette,
        markers=markers,
        s=120,  # 增大点体积
        alpha=0.8  # 增加透明度展示重叠度
    )

    # 绘制 y = x 对角虚线 (判定边界)
    limit_min = min(df["cos(e1, s1)"].min(), df["cos(e1, s2)"].min()) - 0.1
    limit_max = max(df["cos(e1, s1)"].max(), df["cos(e1, s2)"].max()) + 0.1
    plt.plot([limit_min, limit_max], [limit_min, limit_max],
             color='#c0392b',
             linestyle='--',
             linewidth=2.5,
             zorder=0)

    plt.xlim(limit_min, limit_max)
    plt.ylim(limit_min, limit_max)

    plt.title("Target Confusion in Latent Space (Reference-Free Insight)",
              fontweight="bold",
              fontsize=15)
    plt.xlabel("Similarity to Target: $cos(e_1, s_1)$",
               fontweight="bold",
               fontsize=13)
    plt.ylabel("Similarity to Interferer: $cos(e_1, s_2)$",
               fontweight="bold",
               fontsize=13)

    # 优化图例
    plt.legend(loc='upper left',
               frameon=True,
               shadow=True,
               title="Extraction State")

    plt.savefig(os.path.join(configs["exp_dir"],
                             "plot_1_Target_Confusion_Scatter.png"),
                dpi=300,
                bbox_inches='tight')
    plt.close()

    logger.info("🎉 图表生成完毕！快去查看那些突破红线的星星吧！")


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
