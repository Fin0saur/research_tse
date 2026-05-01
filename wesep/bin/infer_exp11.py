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
    """
    计算推断特征与 Oracle 特征之间的相对 L2 距离。
    越接近 0 代表越完美拟合，距离越大代表特征错乱越严重。
    """
    # 直接在 GPU 上进行张量运算，速度极快
    diff = feat_est - feat_ref
    # 计算误差的 L2 范数
    error_norm = torch.norm(diff, p=2)
    # 计算参考特征的 L2 范数作为基底
    ref_norm = torch.norm(feat_ref, p=2)

    # 相对误差 = 绝对误差 / (参考能量 + 极小值防除零)
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

    logger = get_logger(configs["exp_dir"], "infer_usef_distance.log")
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

    # ========================================================
    # 🔪 Hook 注入：特征监听 (Feature Interception)
    # ========================================================
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

            # 提取 Ground Truth 干扰者
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
            # 🏃 截获 3 种 Usef 特征 (以 Mix 为底)
            # ==========================================
            # 1. Oracle Target Usef
            target_as_cue = [target_spk1.view(1, -1)]
            _ = model(mix_spk1, target_as_cue)
            F_tgt_true = usef_features['current'].clone()

            # 2. Oracle Interferer Usef
            interf_as_cue = [interf_spk1.view(1, -1)]
            _ = model(mix_spk1, interf_as_cue)
            F_int_true = usef_features['current'].clone()

            # 3. Dynamic Posterior Usef
            outputs_dyn = model(mix_spk1, cues_spk1)
            F_posterior = usef_features['current'].clone()

            # ==========================================
            # 🧮 计算物理度量：相对 L2 距离
            # ==========================================
            dist_tgt = calc_rel_distance(F_posterior, F_tgt_true)
            dist_int = calc_rel_distance(F_posterior, F_int_true)

            # ==========================================
            # 🎵 计算最终 SI-SNRi
            # ==========================================
            ests_dyn = outputs_dyn[0].detach().cpu().numpy().flatten()
            if np.max(np.abs(ests_dyn)) > 0:
                ests_dyn = ests_dyn / np.max(np.abs(ests_dyn)) * 0.9
            end_idx = min(len(ests_dyn), len(ref_1), len(mix_1))
            _, snri_value = cal_SISNRi(ests_dyn[:end_idx], ref_1[:end_idx],
                                       mix_1[:end_idx])

            plot_data_list.append({
                "Utterance": key[0],
                "Dist_Tgt": dist_tgt,
                "Dist_Int": dist_int,
                "Output_SISNRi": snri_value
            })

            # if (i + 1) % 50 == 0:
            logger.info(
                f"[{i+1}/{test_iter}] Dist_Tgt: {dist_tgt:.3f} | Dist_Int: {dist_int:.3f} | SI-SNRi: {snri_value:+.1f}dB"
            )

    end = time.time()
    logger.info(f"Total Inference Time: {end - start:.1f}s")

    # ========================================================
    # 🎨 核心绘图：绝对误差距离 1x3 连体图
    # ========================================================
    logger.info("🎨 正在生成 Usef L2 Distance 终极分析图...")
    df = pd.DataFrame(plot_data_list)
    csv_path = os.path.join(configs["exp_dir"],
                            "usef_distance_confusion_data.csv")
    df.to_csv(csv_path, index=False)

    def categorize_snri(snri):
        if snri > 5.0: return "Safe (> 5dB)"
        elif snri > 0.0: return "Marginal (0~5dB)"
        else: return "Fatal Confusion (< 0dB)"

    df['State'] = df['Output_SISNRi'].apply(categorize_snri)

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

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    # 距离最小为 0，最大值取数据中的最大距离，留一点余量
    limit_min = 0.0
    limit_max = max(df["Dist_Tgt"].max(), df["Dist_Int"].max()) * 1.05

    for i, state in enumerate(states_order):
        ax = axes[i]
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

        # y = x 虚线 (越界线：如果点跑到虚线下方，说明离干扰者比离目标还近！)
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
        else:
            ax.set_ylabel("")

    plt.suptitle("Target Confusion Decoupled by Absolute Energy Error",
                 fontweight="bold",
                 fontsize=18,
                 y=1.05)
    plt.tight_layout()

    save_path = os.path.join(configs["exp_dir"],
                             "plot_Usef_Distance_Split.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"🎉 物理距离计算完成！神图已保存至: {save_path}")


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
