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

    logger = get_logger(configs["exp_dir"], "infer_latent_interpolation.log")
    logger.info("Load checkpoint from {}".format(model_path))

    save_audio_dir = os.path.join(configs["exp_dir"], "audio_interpolation")
    if sign_save_wav and not os.path.exists(save_audio_dir):
        os.makedirs(save_audio_dir)

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
    # 🔪 Hook 注入：特征劫持与插值注入 (Feature Hijacking)
    # ========================================================
    usef_features = {}
    usef_features['mode'] = 'dynamic'

    target_module = model.module if hasattr(model, 'module') else model

    if hasattr(target_module, 'spk_configs'
               ) and target_module.spk_configs['features']['usef']['enabled']:
        original_usef_post = target_module.spk_ft.usef.post

        def hooked_usef_post(self, mix_repr, feat_repr):
            mode = usef_features.get('mode')

            if mode == 'interpolate':
                # 🚀 劫持！丢弃网络自己算的 feat_repr，强行注入我们在外部算好的插值特征
                injected_feat = usef_features['injected_feat']
                return original_usef_post(mix_repr, injected_feat)
            else:
                # 提取 Oracle 或 Baseline 时，记录当前特征并正常放行
                usef_features['current'] = feat_repr.detach().clone()
                return original_usef_post(mix_repr, feat_repr)

        target_module.spk_ft.usef.post = types.MethodType(
            hooked_usef_post, target_module.spk_ft.usef)
        logger.info("🔪 特征劫持 Hook 已注入，准备进行隐空间插值探测！")

    plot_data_list = []

    # 我们要测试的 Alpha 比例 (0.0=纯干扰，1.0=纯目标)
    alphas = [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60]

    # ========================================================
    # 🏃 宏观推断主循环
    # ========================================================
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):

            if i >= 100:
                break

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
            # 👼 第一步：提取隐空间两极的 Oracle 锚点
            # ==========================================
            usef_features['mode'] = 'extract_oracle'

            target_as_cue_spk1 = [target_spk1.view(1, -1)]
            _ = model(target_spk1, target_as_cue_spk1)
            F_tgt_true = usef_features['current'].clone()

            interf_as_cue_spk1 = [interf_spk1.view(1, -1)]
            _ = model(interf_spk1, interf_as_cue_spk1)
            F_int_true = usef_features['current'].clone()

            # --- 算能量权重 (为了让相似度计算更符合听觉物理) ---
            safe_target_for_stft = target_spk1.view(1, -1)
            tgt_spec = model.sep_model.stft(safe_target_for_stft)[-1]
            tgt_mag = torch.abs(tgt_spec)
            total_energy = tgt_mag.sum() + 1e-8
            weight_matrix = tgt_mag / total_energy

            # ==========================================
            # 🚀 第二步：Baseline 与 隐空间滑动插值
            # ==========================================
            log_metrics = {}

            # --- 1. Baseline ---
            usef_features['mode'] = 'dynamic'
            outputs_dyn = model(mix_spk1, cues_spk1)
            F_baseline = usef_features['current'].clone()

            # 计算 Baseline 的特征指标
            sim_tgt_base = (
                F.cosine_similarity(F_baseline, F_tgt_true, dim=1) *
                weight_matrix).sum().item()
            sim_int_base = (
                F.cosine_similarity(F_baseline, F_int_true, dim=1) *
                weight_matrix).sum().item()
            margin_base = sim_tgt_base - sim_int_base

            ests_dyn = outputs_dyn[0].detach().cpu().numpy().flatten()
            if np.max(np.abs(ests_dyn)) > 0:
                ests_dyn = ests_dyn / np.max(np.abs(ests_dyn)) * 0.9
            end_idx = min(len(ests_dyn), len(ref_1), len(mix_1))
            _, SNRi_dyn = cal_SISNRi(ests_dyn[:end_idx], ref_1[:end_idx],
                                     mix_1[:end_idx])

            log_metrics["Baseline"] = {
                "SNRi": SNRi_dyn,
                "Margin": margin_base,
                "Fid": sim_tgt_base,
                "Leak": sim_int_base
            }
            plot_data_list.append({
                "Utterance": key[0],
                "Alpha": "Baseline",
                "Target_Fidelity": sim_tgt_base,
                "Interferer_Leakage": sim_int_base,
                "Margin": margin_base,
                "Output_SISNRi": SNRi_dyn
            })

            # --- 2. 劫持插值 (Interpolation) ---
            usef_features['mode'] = 'interpolate'

            for alpha in alphas:
                # 核心数学操作：在隐空间画一条线并取点
                F_interp = alpha * F_tgt_true + (1.0 - alpha) * F_int_true

                # 计算这个强制插值特征的物理指标！
                sim_tgt = (F.cosine_similarity(F_interp, F_tgt_true, dim=1) *
                           weight_matrix).sum().item()
                sim_int = (F.cosine_similarity(F_interp, F_int_true, dim=1) *
                           weight_matrix).sum().item()
                margin = sim_tgt - sim_int

                # 装弹，准备注入
                usef_features['injected_feat'] = F_interp
                outputs = model(mix_spk1, cues_spk1)

                out_np = outputs[0].detach().cpu().numpy().flatten()
                if np.max(np.abs(out_np)) > 0:
                    out_np = out_np / np.max(np.abs(out_np)) * 0.9

                _, SNRi = cal_SISNRi(out_np[:end_idx], ref_1[:end_idx],
                                     mix_1[:end_idx])

                log_metrics[f"Alpha_{alpha:.2f}"] = {
                    "SNRi": SNRi,
                    "Margin": margin,
                    "Fid": sim_tgt,
                    "Leak": sim_int
                }
                plot_data_list.append({
                    "Utterance": key[0],
                    "Alpha": float(alpha),
                    "Target_Fidelity": sim_tgt,
                    "Interferer_Leakage": sim_int,
                    "Margin": margin,
                    "Output_SISNRi": SNRi
                })

            logger.info(
                f"\n{'='*60}\n🎯 样本 [{i+1}/{test_iter}] - ID: {key[0]}\n{'='*60}"
            )
            m_base = log_metrics['Baseline']
            logger.info(
                f" 🟢 [Baseline] Margin={m_base['Margin']:+.4f} (Fid={m_base['Fid']:.4f}, Leak={m_base['Leak']:.4f})  ==>  SI-SNRi: {m_base['SNRi']:+.2f} dB"
            )
            logger.info("-" * 60)
            for alpha in alphas:
                m = log_metrics[f"Alpha_{alpha:.2f}"]
                logger.info(
                    f" [a={alpha:.2f}] Margin={m['Margin']:+.4f} (Fid={m['Fid']:.4f}, Leak={m['Leak']:.4f})  ==>  SI-SNRi: {m['SNRi']:+.2f} dB"
                )

        end = time.time()

    logger.info(f"Total Time: {end - start:.1f}s")

    # ========================================================
    # 🧮 统计与可视化
    # ========================================================
    df = pd.DataFrame(plot_data_list)
    csv_path = os.path.join(configs["exp_dir"],
                            "latent_interpolation_results.csv")
    df.to_csv(csv_path, index=False)

    logger.info("=" * 50)
    logger.info("📊 正在计算隐空间插值平均指标 (Global Averages)...")
    avg_stats = df.groupby("Alpha")[[
        "Target_Fidelity", "Interferer_Leakage", "Margin", "Output_SISNRi"
    ]].mean()
    logger.info("\n" + avg_stats.to_string())

    # ========================================================
    # 🎨 画图：隐空间特征插值响应曲线 (极其震撼的双Y轴图)
    # ========================================================
    logger.info("🎨 正在生成隐空间插值轨迹图...")
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 过滤出纯数字的 Alpha 用于画线
    df_interp = df[df["Alpha"] != "Baseline"].copy()
    df_interp["Alpha"] = df_interp["Alpha"].astype(float)

    # 获取平均数据
    mean_data = df_interp.groupby("Alpha").mean().reset_index()
    baseline_snri = df[df["Alpha"] == "Baseline"]["Output_SISNRi"].mean()
    baseline_margin = df[df["Alpha"] == "Baseline"]["Margin"].mean()

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # 左Y轴：物理特征 Margin (红蓝冷暖色)
    color_margin = '#8e44ad'  # 紫色
    ax1.set_xlabel(
        r"Interpolation Ratio $\alpha$ (0.0=100% Interferer $\to$ 1.0=100% Target)",
        fontweight="bold")
    ax1.set_ylabel("Feature Margin (Target Fid - Interferer Leak)",
                   color=color_margin,
                   fontweight="bold")
    sns.lineplot(data=mean_data,
                 x="Alpha",
                 y="Margin",
                 marker="s",
                 markersize=10,
                 linewidth=3,
                 color=color_margin,
                 ax=ax1,
                 label="Latent Margin")
    ax1.tick_params(axis='y', labelcolor=color_margin)

    # 添加 Margin 的 0 线 (生死线)
    ax1.axhline(y=0,
                color='black',
                linestyle=":",
                linewidth=2,
                label="Confusion Boundary (Margin=0)")
    ax1.axhline(y=baseline_margin,
                color=color_margin,
                linestyle="--",
                alpha=0.5,
                linewidth=2,
                label=f"Baseline Margin ({baseline_margin:.4f})")

    # 右Y轴：模型表现 SI-SNRi (绿色)
    ax2 = ax1.twinx()
    color_snri = '#2ecc71'
    ax2.set_ylabel("Output SI-SNRi (dB)", color=color_snri, fontweight="bold")
    sns.lineplot(data=mean_data,
                 x="Alpha",
                 y="Output_SISNRi",
                 marker="o",
                 markersize=10,
                 linewidth=3,
                 color=color_snri,
                 ax=ax2,
                 label="Output SI-SNRi")
    ax2.tick_params(axis='y', labelcolor=color_snri)

    # 添加 Baseline SI-SNRi 参考线
    ax2.axhline(y=baseline_snri,
                color=color_snri,
                linestyle="--",
                alpha=0.5,
                linewidth=2,
                label=f"Baseline SI-SNRi ({baseline_snri:.2f}dB)")

    plt.title("Latent Manifold Probing: Margin vs. SI-SNRi Response",
              fontweight="bold",
              fontsize=15)

    # 合并两个轴的图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2,
               labels_1 + labels_2,
               loc="upper left",
               bbox_to_anchor=(1.1, 1))

    fig.savefig(os.path.join(configs["exp_dir"],
                             "plot_Latent_Trajectory_with_Margin.png"),
                dpi=300,
                bbox_inches='tight')
    logger.info("🎉 插值探测结束！请查看带 Margin 的终极轨迹图。")


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
