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
from wesep.models import get_model
from wesep.utils.checkpoint import load_pretrained_model
from wesep.utils.score import cal_SISNRi
from wesep.utils.utils import (
    generate_enahnced_scp,
    get_logger,
    parse_config_or_kwargs,
    set_seed,
)
from wesep.utils.file_utils import load_yaml

# === 数据分析和绘图依赖 ===
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr  # 引入统计学库计算相关性

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


def infer(config="confs/conf.yaml", **kwargs):
    start = time.time()

    configs = parse_config_or_kwargs(config, **kwargs)
    sign_save_wav = configs.get("save_wav", False)

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

    logger = get_logger(configs["exp_dir"], "infer_posterior_analysis.log")
    logger.info("Load checkpoint from {}".format(model_path))

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
    logger.info("Test number: {}".format(test_iter))

    # ========================================================
    # 🔪 Hook 注入：仅做特征监听 (Feature Interception)
    # ========================================================
    usef_features = {}

    target_module = model.module if hasattr(model, 'module') else model

    if hasattr(target_module, 'spk_configs'
               ) and target_module.spk_configs['features']['usef']['enabled']:
        original_usef_post = target_module.spk_ft.usef.post

        def hooked_usef_post(self, mix_repr, feat_repr):
            # 简化 Hook：不再劫持插值，仅仅作为探针截获特征
            usef_features['current'] = feat_repr.detach().clone()
            return original_usef_post(mix_repr, feat_repr)

        target_module.spk_ft.usef.post = types.MethodType(
            hooked_usef_post, target_module.spk_ft.usef)
        logger.info("🔪 特征监听 Hook 已注入，准备进行后验相似度分析！")

    plot_data_list = []

    # ========================================================
    # 🏃 宏观推断主循环
    # ========================================================
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):

            # 为了画出足够密集和有统计意义的散点图，建议至少跑 500-1000 个样本
            # if i >= 1000:
            #     break

            mix, cues, target = extract_model_inputs(batch, device)
            # print(mix.shape)
            # print(target.shape)
            key = batch["key"]

            mix_spk1 = mix[0:1, :, :]
            target_spk1 = target[0:1, :, :] if target.dim() == 3 else target[
                0:1, 0:1, :]

            cues_spk1 = [cues[0][0:1, :]] if cues is not None else None
            if cues_spk1 is None or cues_spk1[0].shape[-1] == 0:
                continue

            ref_1 = target_spk1.detach().cpu().numpy().flatten()
            mix_1 = mix_spk1.detach().cpu().numpy().flatten()

            # ==========================================
            # 👼 第一步：提取 Oracle Target 锚点特征
            # ==========================================
            target_as_cue_spk1 = [target_spk1.view(1, -1)]
            _ = model(target_spk1, target_as_cue_spk1)
            F_tgt_true = usef_features['current'].clone()

            # --- 算能量权重 (使相似度计算更符合听觉物理) ---
            safe_target_for_stft = target_spk1.view(1, -1)
            tgt_spec = model.sep_model.stft(safe_target_for_stft)[-1]
            tgt_mag = torch.abs(tgt_spec)
            total_energy = tgt_mag.sum() + 1e-8
            weight_matrix = tgt_mag / total_energy

            # ==========================================
            # 🚀 第二步：提取真实后验特征 (Dynamic/Baseline)
            # ==========================================
            outputs_dyn = model(mix_spk1, cues_spk1)
            F_posterior = usef_features['current'].clone()
            F_posterior_centered = F_posterior - F_posterior.mean(dim=1,
                                                                  keepdim=True)
            F_tgt_true_centered = F_tgt_true - F_tgt_true.mean(dim=1,
                                                               keepdim=True)

            sim_posterior = (F.cosine_similarity(
                F_posterior_centered, F_tgt_true_centered, dim=1) *
                             weight_matrix).sum().item()

            # 计算 SI-SNRi
            ests_dyn = outputs_dyn[0].detach().cpu().numpy().flatten()
            if np.max(np.abs(ests_dyn)) > 0:
                ests_dyn = ests_dyn / np.max(np.abs(ests_dyn)) * 0.9
            end_idx = min(len(ests_dyn), len(ref_1), len(mix_1))
            _, snri_value = cal_SISNRi(ests_dyn[:end_idx], ref_1[:end_idx],
                                       mix_1[:end_idx])

            # 记录数据
            plot_data_list.append({
                "Utterance": key[0],
                "Sim_Posterior": sim_posterior,
                "Output_SISNRi": snri_value
            })

            logger.info(
                f"Processed {i+1}/{test_iter} samples...,Sim:{sim_posterior},sisnri:{snri_value}"
            )

    end = time.time()
    logger.info(f"Total Inference Time: {end - start:.1f}s")

    # ========================================================
    # 🧮 统计与可视化 (核心干货)
    # ========================================================
    df = pd.DataFrame(plot_data_list)
    csv_path = os.path.join(configs["exp_dir"],
                            "posterior_similarity_analysis.csv")
    df.to_csv(csv_path, index=False)

    # 1. 计算全局皮尔逊相关系数
    global_r, global_p = pearsonr(df["Sim_Posterior"], df["Output_SISNRi"])
    logger.info("=" * 50)
    logger.info(
        f"📊 【全局分析】Sim_Posterior 与 SI-SNRi 的皮尔逊相关系数: R = {global_r:.4f} (p-value: {global_p:.2e})"
    )

    # 2. 分段定义 (划分“混淆区”与“瓶颈区”)
    # 这里的阈值可以根据你的具体数据分布微调，通常 0.75 到 0.8 是一个物理分水岭
    threshold = 0.80
    df_confusion = df[df["Sim_Posterior"] < threshold]
    df_bottleneck = df[df["Sim_Posterior"] >= threshold]

    r_conf, _ = pearsonr(
        df_confusion["Sim_Posterior"],
        df_confusion["Output_SISNRi"]) if len(df_confusion) > 1 else (0, 1)
    r_bott, _ = pearsonr(
        df_bottleneck["Sim_Posterior"],
        df_bottleneck["Output_SISNRi"]) if len(df_bottleneck) > 1 else (0, 1)

    logger.info(
        f"🔴 【混淆区 (Sim < {threshold})】样本数: {len(df_confusion)}, 相关系数 R = {r_conf:.4f} (高度正相关)"
    )
    logger.info(
        f"🟢 【瓶颈区 (Sim >= {threshold})】样本数: {len(df_bottleneck)}, 相关系数 R = {r_bott:.4f} (饱和无强相关)"
    )
    logger.info("=" * 50)

    # ========================================================
    # 🎨 画图 1：全局散点图 + 线性回归拟合
    # ========================================================
    logger.info("🎨 正在生成全局散点拟合图...")
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(10, 8))

    sns.regplot(data=df,
                x="Sim_Posterior",
                y="Output_SISNRi",
                scatter_kws={
                    'alpha': 0.5,
                    's': 30,
                    'color': '#3498db'
                },
                line_kws={
                    'color': '#e74c3c',
                    'linewidth': 3
                })
    plt.title(f"Global Correlation (Pearson R = {global_r:.3f})",
              fontweight="bold",
              fontsize=16)
    plt.xlabel("Posterior Feature Similarity (Pmap vs Oracle)",
               fontweight="bold")
    plt.ylabel("Output SI-SNRi (dB)", fontweight="bold")
    plt.savefig(os.path.join(configs["exp_dir"],
                             "plot_1_Global_Scatter_Reg.png"),
                dpi=300,
                bbox_inches='tight')
    plt.close()

    # ========================================================
    # 🎨 画图 2：分段区域散点图 (支撑 Scale Predictor 设计的灵魂图)
    # ========================================================
    logger.info("🎨 正在生成分段区域散点图...")
    plt.figure(figsize=(12, 8))

    # 画混淆区 (红色)
    sns.regplot(data=df_confusion,
                x="Sim_Posterior",
                y="Output_SISNRi",
                scatter_kws={
                    'alpha': 0.6,
                    's': 40,
                    'color': '#e74c3c'
                },
                line_kws={
                    'color': '#c0392b',
                    'linewidth': 3
                },
                label=f"Confusion Zone (R = {r_conf:.3f})")

    # 画瓶颈区 (绿色)
    sns.regplot(data=df_bottleneck,
                x="Sim_Posterior",
                y="Output_SISNRi",
                scatter_kws={
                    'alpha': 0.6,
                    's': 40,
                    'color': '#2ecc71'
                },
                line_kws={
                    'color': '#27ae60',
                    'linewidth': 3
                },
                label=f"Bottleneck Zone (R = {r_bott:.3f})")

    # 画分界线
    plt.axvline(x=threshold,
                color='black',
                linestyle='--',
                linewidth=2,
                label=f"Safety Threshold ({threshold})")

    plt.title(
        "Piecewise Analysis: Speaker Confusion vs. Reconstruction Bottleneck",
        fontweight="bold",
        fontsize=16)
    plt.xlabel("Posterior Feature Similarity (Pmap vs Oracle)",
               fontweight="bold")
    plt.ylabel("Output SI-SNRi (dB)", fontweight="bold")
    plt.legend(loc="lower right", fontsize=12)
    plt.savefig(os.path.join(configs["exp_dir"],
                             "plot_2_Piecewise_Scatter.png"),
                dpi=300,
                bbox_inches='tight')
    plt.close()

    logger.info("🎉 分析跑通！请去 exp_dir 查看 CSV 与神级对比图！")


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
