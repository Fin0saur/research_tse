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
    get_logger,
    parse_config_or_kwargs,
    set_seed,
)
from wesep.utils.file_utils import load_yaml

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


# ========================================================
# 🚀 终极真理指标：相对 L2 绝对误差 (Relative L2 Distance)
# ========================================================
def calc_rel_distance(feat_est, feat_ref):
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

    logger = get_logger(configs["exp_dir"],
                        "infer_usef_distance_and_oracle.log")
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

    usef_features = {}
    target_module = model.module if hasattr(model, 'module') else model

    if hasattr(target_module, 'spk_configs'
               ) and target_module.spk_configs['features']['usef']['enabled']:
        original_usef_post = target_module.spk_ft.usef.post

        def hooked_usef_post(self, mix_repr, feat_repr):
            usef_features['current'] = feat_repr.detach().clone()
            return original_usef_post(mix_repr, feat_repr)

        target_module.spk_ft.usef.post = types.MethodType(
            hooked_usef_post, target_module.spk_ft.usef)
        logger.info("🔪 Usef 监听 Hook 已注入！")

    plot_data_list = []
    test_iter = 3000

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            # if i >= 1000: break

            mix, cues, target = extract_model_inputs(batch, device)
            key = batch["key"]

            mix_spk1 = mix[0:1, :, :]
            target_spk1 = target[0:1, :, :] if target.dim() == 3 else target[
                0:1, 0:1, :]

            if target.shape[0] > 1:
                interf_spk1 = target[
                    1:2, :, :] if target.dim() == 3 else target[1:2, 0:1, :]
            else:
                interf_spk1 = mix_spk1 - target_spk1

            cues_spk1 = [cues[0][0:1, :]] if cues is not None else None
            if cues_spk1 is None or cues_spk1[0].shape[-1] == 0:
                continue

            ref_1 = target_spk1.detach().cpu().numpy().flatten()
            mix_1 = mix_spk1.detach().cpu().numpy().flatten()

            # ==========================================
            # 🏃 1. Oracle Target 提取与特征截获
            # ==========================================
            target_as_cue = [target_spk1.view(1, -1)]
            outputs_tgt = model(mix_spk1, target_as_cue)  # 保留完整输出算 Oracle SNRi
            F_tgt_true = usef_features['current'].clone()

            # ==========================================
            # 🏃 2. Oracle Interferer 特征截获
            # ==========================================
            interf_as_cue = [interf_spk1.view(1, -1)]
            _ = model(mix_spk1, interf_as_cue)
            F_int_true = usef_features['current'].clone()

            # ==========================================
            # 🏃 3. 真实 Dynamic 推理与特征截获
            # ==========================================
            outputs_dyn = model(mix_spk1, cues_spk1)
            F_posterior = usef_features['current'].clone()

            # --- 物理距离计算 ---
            dist_tgt = calc_rel_distance(F_posterior, F_tgt_true)
            dist_int = calc_rel_distance(F_posterior, F_int_true)

            # --- 真实 Real SI-SNRi 计算 ---
            ests_dyn = outputs_dyn[0].detach().cpu().numpy().flatten()
            if np.max(np.abs(ests_dyn)) > 0:
                ests_dyn = ests_dyn / np.max(np.abs(ests_dyn)) * 0.9
            end_idx = min(len(ests_dyn), len(ref_1), len(mix_1))
            _, snri_real = cal_SISNRi(ests_dyn[:end_idx], ref_1[:end_idx],
                                      mix_1[:end_idx])

            # --- 理论上限 Oracle SI-SNRi 计算 ---
            ests_oracle = outputs_tgt[0].detach().cpu().numpy().flatten()
            if np.max(np.abs(ests_oracle)) > 0:
                ests_oracle = ests_oracle / np.max(np.abs(ests_oracle)) * 0.9
            _, snri_oracle = cal_SISNRi(ests_oracle[:end_idx], ref_1[:end_idx],
                                        mix_1[:end_idx])

            delta_snri = snri_oracle - snri_real

            plot_data_list.append({
                "Utterance": key[0],
                "Dist_Tgt": dist_tgt,
                "Dist_Int": dist_int,
                "Real_SISNRi": snri_real,
                "Oracle_SISNRi": snri_oracle,
                "Delta_SISNRi": delta_snri
            })

            logger.info(
                f"[{i+1}/{test_iter}] DT:{dist_tgt:.2f}|DI:{dist_int:.2f} | Real:{snri_real:+.1f}dB -> Oracle:{snri_oracle:+.1f}dB"
            )

    end = time.time()
    logger.info(f"Total Inference Time: {end - start:.1f}s")

    # ========================================================
    # 💾 数据保存
    # ========================================================
    df = pd.DataFrame(plot_data_list)
    csv_path = os.path.join(configs["exp_dir"],
                            "master_usef_oracle_dataframe.csv")
    df.to_csv(csv_path, index=False)

    # ========================================================
    # 🎨 绘图 1：原来的绝对误差距离 1x3 连体图 (保留)
    # ========================================================
    def categorize_snri(snri):
        if snri > 5.0: return "Safe (> 5dB)"
        elif snri > 0.0: return "Marginal (0~5dB)"
        else: return "Fatal Confusion (< 0dB)"

    df['State'] = df['Real_SISNRi'].apply(categorize_snri)

    sns.set_theme(style="whitegrid", font_scale=1.2)
    states_order = [
        "Safe (> 5dB)", "Marginal (0~5dB)", "Fatal Confusion (< 0dB)"
    ]
    palette = {
        "Safe (> 5dB)": "#2ca02c",
        "Marginal (0~5dB)": "#98df8a",
        "Fatal Confusion (< 0dB)": "#d62728"
    }
    markers = {
        "Safe (> 5dB)": "o",
        "Marginal (0~5dB)": "o",
        "Fatal Confusion (< 0dB)": "*"
    }

    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    limit_min = 0.0
    limit_max = max(df["Dist_Tgt"].max(), df["Dist_Int"].max()) * 1.05

    for i, state in enumerate(states_order):
        ax = axes1[i]
        subset = df[df["State"] == state]
        if len(subset) > 5:
            sns.kdeplot(data=subset,
                        x="Dist_Tgt",
                        y="Dist_Int",
                        levels=5,
                        color=palette[state],
                        alpha=0.3,
                        fill=True,
                        ax=ax)
        sns.scatterplot(data=subset,
                        x="Dist_Tgt",
                        y="Dist_Int",
                        color=palette[state],
                        marker=markers[state],
                        s=120 if state != "Fatal Confusion (< 0dB)" else 180,
                        alpha=0.8,
                        edgecolor="white" if state == "Safe (> 5dB)" else None,
                        ax=ax)
        ax.plot([limit_min, limit_max], [limit_min, limit_max],
                color='#c0392b',
                linestyle='--',
                linewidth=2,
                zorder=0)
        ax.set_xlim(limit_min, limit_max)
        ax.set_ylim(limit_min, limit_max)
        ax.set_title(f"{state}\n(N={len(subset)})",
                     fontweight="bold",
                     fontsize=15)
        ax.set_xlabel("Relative L2 Error to Oracle Target", fontweight="bold")
        if i == 0:
            ax.set_ylabel("Relative L2 Error to Oracle Interferer",
                          fontweight="bold")

    plt.suptitle("Target Confusion Decoupled by Absolute Energy Error",
                 fontweight="bold",
                 fontsize=18,
                 y=1.05)
    plt.tight_layout()
    fig1.savefig(os.path.join(configs["exp_dir"],
                              "plot_1_Usef_Distance_Split.png"),
                 dpi=300,
                 bbox_inches='tight')
    plt.close(fig1)

    # ========================================================
    # 🎨 绘图 2：🔥 新增 Usef 空间拯救热力图 (Rescue Heatmap)
    # ========================================================
    logger.info("🎨 正在生成 Usef 后验空间的拯救潜力分布图...")
    fig2, ax2 = plt.subplots(figsize=(10, 8))

    # 注意：这里的物理意义，越接近 0 越好。
    # 所以右下角 (Dist_Tgt 很大, Dist_Int 很小) 是彻底的认错人区域
    scatter = ax2.scatter(
        x=df['Dist_Tgt'],
        y=df['Dist_Int'],
        c=df['Delta_SISNRi'],  # 颜色映射为提升幅度
        cmap='plasma',  # 黑紫-红-黄-白 的醒目渐变
        s=70,
        alpha=0.85,
        edgecolors='w',
        linewidth=0.5)

    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('$\\Delta$ SI-SNRi (Oracle - Real) [dB]',
                   fontweight='bold',
                   fontsize=14)

    ax2.plot([limit_min, limit_max], [limit_min, limit_max],
             color='#c0392b',
             linestyle='--',
             linewidth=2.5,
             zorder=0,
             label='Decision Boundary ($y=x$)')

    # 添加文字引导：右下角是混淆区 (距离 Target 远，距离 Interferer 近)
    ax2.text(limit_max * 0.55,
             limit_max * 0.1,
             'Confusion Zone\n(High Rescue Potential)',
             color='#c0392b',
             fontweight='bold',
             fontsize=13,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # 左上角是安全区 (距离 Target 近，距离 Interferer 远)
    ax2.text(limit_max * 0.05,
             limit_max * 0.8,
             'Safe Zone\n(Low Rescue Potential)',
             color='#2ca02c',
             fontweight='bold',
             fontsize=13,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax2.set_xlim(limit_min, limit_max)
    ax2.set_ylim(limit_min, limit_max)
    ax2.set_xlabel('Usef Relative L2 Error to Target ($Dist\_Tgt$)',
                   fontweight='bold',
                   fontsize=14)
    ax2.set_ylabel('Usef Relative L2 Error to Interferer ($Dist\_Int$)',
                   fontweight='bold',
                   fontsize=14)
    ax2.set_title("Rescue Potential Heatmap in Dynamic Posterior Space",
                  fontweight='bold',
                  fontsize=16,
                  pad=15)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    save_path_heatmap = os.path.join(configs["exp_dir"],
                                     "plot_2_Rescue_Heatmap.png")
    fig2.savefig(save_path_heatmap, dpi=300, bbox_inches='tight')
    plt.close(fig2)

    logger.info(f"🎉 跑图全部完成！图1、图2均已保存。")


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
