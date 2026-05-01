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

    logger = get_logger(configs["exp_dir"],
                        "infer_diagnostic_three_stages.log")
    logger.info("Load checkpoint from {}".format(model_path))

    save_audio_dir = os.path.join(configs["exp_dir"], "audio")
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
    # 🔪 Hook 注入：三级递进消融 (Zero-Inter, Post-Inter Avg, Dynamic)
    # ========================================================
    usef_features = {}
    usef_features['mode'] = 'dynamic'

    target_module = model.module if hasattr(model, 'module') else model

    if hasattr(target_module, 'spk_configs'
               ) and target_module.spk_configs['features']['usef']['enabled']:
        original_usef_compute = target_module.spk_ft.usef.compute
        original_usef_post = target_module.spk_ft.usef.post

        def hooked_usef_compute(self, enroll_repr, mix_repr):
            mode = usef_features.get('mode')

            if mode == 'static_zero':
                # Level 1: 绝对隔离 (Zero Interaction)
                enroll_only_out, _ = original_usef_compute(
                    enroll_repr, enroll_repr)
                static_emb = enroll_only_out.mean(dim=-1, keepdim=True)

                _, mix_only_out = original_usef_compute(mix_repr, mix_repr)
                T_m = mix_repr.shape[-1]
                static_enroll_expanded = static_emb.expand(
                    *static_emb.shape[:-1], T_m)

                return static_enroll_expanded, mix_only_out

            elif mode == 'static_inter':
                # Level 2: 允许融合但抹杀时间维度 (Post-Interaction Avg)
                enroll_usef, mix_usef = original_usef_compute(
                    enroll_repr, mix_repr)
                enroll_usef = enroll_usef.mean(
                    dim=-1, keepdim=True).expand_as(enroll_usef)
                return enroll_usef, mix_usef

            else:
                # Level 3: 完全体 (Dynamic)
                return original_usef_compute(enroll_repr, mix_repr)

        def hooked_usef_post(self, mix_repr, feat_repr):
            usef_features['current'] = feat_repr.detach().clone()
            return original_usef_post(mix_repr, feat_repr)

        target_module.spk_ft.usef.compute = types.MethodType(
            hooked_usef_compute, target_module.spk_ft.usef)
        target_module.spk_ft.usef.post = types.MethodType(
            hooked_usef_post, target_module.spk_ft.usef)
        logger.info("🔪 三级消融 Hook 已注入，准备起飞！")

    plot_data_list = []

    # ========================================================
    # 🏃 宏观推断主循环
    # ========================================================
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):

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

            # ==========================================
            # 👼 提取 Oracle 真理特征
            # ==========================================
            usef_features['mode'] = 'dynamic'

            target_as_cue_spk1 = [target_spk1.view(1, -1)]
            _ = model(target_spk1, target_as_cue_spk1)
            F_tgt_true = usef_features['current'].clone()

            interf_as_cue_spk1 = [interf_spk1.view(1, -1)]
            _ = model(interf_spk1, interf_as_cue_spk1)
            F_int_true = usef_features['current'].clone()

            # ==========================================
            # 🌟 修改 1：计算整帧的全局能量权重 (Frame-level Energy)
            # ==========================================
            safe_target_for_stft = target_spk1.view(1, -1)
            tgt_spec = model.sep_model.stft(safe_target_for_stft)[-1]
            tgt_mag = torch.abs(tgt_spec)

            # 确保 tgt_mag 是 3D 形状 (B, F, T)，以防 stft 返回 (F, T)
            if tgt_mag.dim() == 2:
                tgt_mag = tgt_mag.unsqueeze(0)

            # 【核心操作】：沿着频率轴 (dim=1) 求和，把每帧的所有频带能量合并
            # 这样 frame_energy 的形状就变成了 (B, T)
            frame_energy = tgt_mag.sum(dim=1)
            total_energy = frame_energy.sum() + 1e-8

            # 现在的权重矩阵只在时间维度上起作用
            frame_weight_matrix = frame_energy / total_energy

            ref_1 = target_spk1.detach().cpu().numpy().flatten()
            mix_1 = mix_spk1.detach().cpu().numpy().flatten()

            # ==========================================
            # 🚀 三轨推断循环
            # ==========================================
            modes = [("1_Static_Zero", "static_zero"),
                     ("2_Static_Inter", "static_inter"),
                     ("3_Dynamic", "dynamic")]

            log_metrics = {}

            for method_name, mode_flag in modes:
                usef_features['mode'] = mode_flag
                outputs = model(mix_spk1, cues_spk1)
                # ==========================================
                # 🌟 修改 2：展平特征，并进行去均值化 (Pearson相关系数)
                # ==========================================
                F_mix_current = usef_features['current'].clone()

                # 1. 动态适配：融合通道和频率维度
                if F_mix_current.dim() == 4:
                    B, C_dim, F_dim, T_dim = F_mix_current.shape
                    F_mix_flat = F_mix_current.view(B, C_dim * F_dim, T_dim)
                    F_tgt_flat = F_tgt_true.view(B, C_dim * F_dim, T_dim)
                    F_int_flat = F_int_true.view(B, C_dim * F_dim, T_dim)
                else:
                    F_mix_flat = F_mix_current
                    F_tgt_flat = F_tgt_true
                    F_int_flat = F_int_true

                # 2. 去均值化（沿着融合后的特征维度 dim=1）
                F_mix_centered = F_mix_flat - F_mix_flat.mean(dim=1,
                                                              keepdim=True)
                F_tgt_centered = F_tgt_flat - F_tgt_flat.mean(
                    dim=1, keepdim=True)  # <-- Target 必须去均值
                F_int_centered = F_int_flat - F_int_flat.mean(
                    dim=1, keepdim=True)  # <-- Interferer 也去均值

                # 3. 计算去均值后的余弦相似度 (等价于计算整帧的 Pearson 相关系数)
                sim_frame_tgt = F.cosine_similarity(F_mix_centered,
                                                    F_tgt_centered,
                                                    dim=1)
                sim_frame_int = F.cosine_similarity(F_mix_centered,
                                                    F_int_centered,
                                                    dim=1)

                # 4. 乘以帧级能量权重，聚合为标量
                sim_tgt = (sim_frame_tgt * frame_weight_matrix).sum().item()
                sim_int = (sim_frame_int * frame_weight_matrix).sum().item()
                margin = sim_tgt - sim_int

                out_np = outputs[0].detach().cpu().numpy() if isinstance(
                    outputs,
                    (list, tuple)) else outputs.detach().cpu().numpy()
                ests = out_np.flatten() if out_np.ndim >= 2 else out_np
                if np.max(np.abs(ests)) > 0:
                    ests = ests / np.max(np.abs(ests)) * 0.9
                end_idx = min(len(ests), len(ref_1), len(mix_1))
                _, SNRi = cal_SISNRi(ests[:end_idx], ref_1[:end_idx],
                                     mix_1[:end_idx])

                log_metrics[method_name] = {
                    "Fid": sim_tgt,
                    "Leak": sim_int,
                    "Margin": margin,
                    "SNRi": SNRi
                }

                plot_data_list.append({
                    "Utterance": key[0],
                    "Method": method_name,
                    "Target_Fidelity": sim_tgt,
                    "Interferer_Leakage": sim_int,
                    "Margin": margin,
                    "Output_SISNRi": SNRi
                })

            logger.info(f"--- Processed {i+1}/{test_iter} mixtures ---")
            logger.info(
                f"  [Zero ] Fid={log_metrics['1_Static_Zero']['Fid']:.4f} | Leak={log_metrics['1_Static_Zero']['Leak']:.4f} | Margin={log_metrics['1_Static_Zero']['Margin']:+.4f} | SNRi={log_metrics['1_Static_Zero']['SNRi']:+.2f}dB"
            )
            logger.info(
                f"  [Inter] Fid={log_metrics['2_Static_Inter']['Fid']:.4f} | Leak={log_metrics['2_Static_Inter']['Leak']:.4f} | Margin={log_metrics['2_Static_Inter']['Margin']:+.4f} | SNRi={log_metrics['2_Static_Inter']['SNRi']:+.2f}dB"
            )
            logger.info(
                f"  [Dyn  ] Fid={log_metrics['3_Dynamic']['Fid']:.4f} | Leak={log_metrics['3_Dynamic']['Leak']:.4f} | Margin={log_metrics['3_Dynamic']['Margin']:+.4f} | SNRi={log_metrics['3_Dynamic']['SNRi']:+.2f}dB"
            )

        end = time.time()

    if sign_save_wav:
        generate_enahnced_scp(os.path.abspath(save_audio_dir), extension="wav")

    logger.info(f"Total Time: {end - start:.1f}s")

    # ========================================================
    # 🧮 全局均值统计 & 困难/简单样本 分组深度解剖
    # ========================================================
    df = pd.DataFrame(plot_data_list)
    csv_path = os.path.join(configs["exp_dir"], "diagnostic_three_stages.csv")
    df.to_csv(csv_path, index=False)

    logger.info("=" * 50)
    logger.info("📊 [1] 正在计算全局平均指标 (Global Averages)...")
    avg_stats = df.groupby("Method")[[
        "Target_Fidelity", "Interferer_Leakage", "Margin", "Output_SISNRi"
    ]].mean()
    logger.info("\n" + avg_stats.to_string())

    # 挖掘巨大 Gap 样本 (对比 Level 3 和 Level 1)
    df_zero = df[df["Method"] == "1_Static_Zero"].set_index("Utterance")
    df_dyn = df[df["Method"] == "3_Dynamic"].set_index("Utterance")

    df_compare = df_dyn[["Output_SISNRi",
                         "Margin"]].join(df_zero[["Output_SISNRi", "Margin"]],
                                         lsuffix="_Dyn",
                                         rsuffix="_Zero")
    df_compare["SNRi_Gap_Dyn_vs_Zero"] = df_compare[
        "Output_SISNRi_Dyn"] - df_compare["Output_SISNRi_Zero"]

    gap_threshold = 3.0

    # 🔪 切分两大数据阵营的 Utterance ID
    hard_cases_utts = df_compare[
        df_compare["SNRi_Gap_Dyn_vs_Zero"] > gap_threshold].index
    easy_cases_utts = df_compare[
        df_compare["SNRi_Gap_Dyn_vs_Zero"] <= gap_threshold].index

    # 保存高 Gap 名单
    df_hard_cases = df_compare.loc[hard_cases_utts].sort_values(
        by="SNRi_Gap_Dyn_vs_Zero", ascending=False)
    hard_cases_path = os.path.join(configs["exp_dir"],
                                   "hard_cases_high_gap.csv")
    df_hard_cases.to_csv(hard_cases_path)

    logger.info("=" * 50)
    logger.info(
        f"🔥 [2] 发现 {len(hard_cases_utts)} 个产生了极端 Gap (> {gap_threshold}dB) 的生死局！名单已保存。"
    )

    # ========================================================
    # 🔬 终极解剖：高 Gap 组 vs 低 Gap 组 的内部特征演化
    # ========================================================
    logger.info("=" * 50)
    logger.info(
        f"🔬 [3] 深入解剖：【高 Gap 困难组】({len(hard_cases_utts)}个) vs 【低 Gap 顺风组】({len(easy_cases_utts)}个)"
    )

    df_high_gap = df[df["Utterance"].isin(hard_cases_utts)]
    df_low_gap = df[df["Utterance"].isin(easy_cases_utts)]

    if len(df_high_gap) > 0:
        logger.info("\n🚨 【高 Gap 困难组 (Dyn 提升 > 3dB)】指标演化：")
        logger.info("\n" + df_high_gap.groupby("Method")[[
            "Target_Fidelity", "Interferer_Leakage", "Margin", "Output_SISNRi"
        ]].mean().to_string())
    else:
        logger.info("\n🚨 未发现高 Gap 样本。")

    if len(df_low_gap) > 0:
        logger.info("\n🍵 【低 Gap 顺风组 (Dyn 提升 <= 3dB)】指标演化：")
        logger.info("\n" + df_low_gap.groupby("Method")[[
            "Target_Fidelity", "Interferer_Leakage", "Margin", "Output_SISNRi"
        ]].mean().to_string())
    logger.info("=" * 50)

    # ========================================================
    # 🎨 三重星云绘图模块 (补齐 Leakage!)
    # ========================================================
    logger.info("🎨 正在生成三级消融散点分析图...")
    sns.set_theme(style="whitegrid", font_scale=1.2)

    # 配色：Zero是红色(弱)，Inter是橙色(中)，Dynamic是蓝色(强)
    palette = {
        "1_Static_Zero": "#e74c3c",
        "2_Static_Inter": "#f39c12",
        "3_Dynamic": "#3498db"
    }

    # --- 图 1：Fidelity ---
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=df,
                    x="Target_Fidelity",
                    y="Output_SISNRi",
                    hue="Method",
                    palette=palette,
                    alpha=0.3,
                    s=20,
                    edgecolor=None,
                    ax=ax1)
    for method_name, color in palette.items():
        sns.regplot(data=df[df["Method"] == method_name],
                    x="Target_Fidelity",
                    y="Output_SISNRi",
                    scatter=False,
                    color=color,
                    line_kws={"linewidth": 3},
                    ax=ax1)
    ax1.set_title("Target Fidelity vs SI-SNRi (3-Stage Ablation)",
                  fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig1.savefig(os.path.join(configs["exp_dir"],
                              "plot_1_Fidelity_3stages.png"),
                 dpi=300,
                 bbox_inches='tight')

    # --- 图 2：Leakage (新增补齐) ---
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=df,
                    x="Interferer_Leakage",
                    y="Output_SISNRi",
                    hue="Method",
                    palette=palette,
                    alpha=0.3,
                    s=20,
                    edgecolor=None,
                    ax=ax2)
    for method_name, color in palette.items():
        sns.regplot(data=df[df["Method"] == method_name],
                    x="Interferer_Leakage",
                    y="Output_SISNRi",
                    scatter=False,
                    color=color,
                    line_kws={"linewidth": 3},
                    ax=ax2)
    ax2.set_title("Interferer Leakage vs SI-SNRi (3-Stage Ablation)",
                  fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig2.savefig(os.path.join(configs["exp_dir"],
                              "plot_2_Leakage_3stages.png"),
                 dpi=300,
                 bbox_inches='tight')

    # --- 图 3：Margin ---
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=df,
                    x="Margin",
                    y="Output_SISNRi",
                    hue="Method",
                    palette=palette,
                    alpha=0.3,
                    s=20,
                    edgecolor=None,
                    ax=ax3)
    for method_name, color in palette.items():
        sns.regplot(data=df[df["Method"] == method_name],
                    x="Margin",
                    y="Output_SISNRi",
                    scatter=False,
                    color=color,
                    line_kws={"linewidth": 3},
                    ax=ax3)
    ax3.axvline(x=0,
                color='black',
                linestyle=':',
                linewidth=2,
                label="Confusion Boundary (Margin=0)")
    ax3.set_title("Confusion Margin vs SI-SNRi (3-Stage Ablation)",
                  fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig3.savefig(os.path.join(configs["exp_dir"], "plot_3_Margin_3stages.png"),
                 dpi=300,
                 bbox_inches='tight')

    logger.info("🎉 全部流程圆满结束！所有数据和图片已落盘。")


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
