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


def infer(config="confs/conf.yaml", utt_id=None, **kwargs):
    """
    utt_id: 传入指定样本的 key (例如 "3536-8226-0012_overlap")，进行单样本微观解剖。
            如果不传，则进行整个测试集的宏观推断与画图统计。
    """
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
    if not os.path.exists(save_audio_dir) and (sign_save_wav
                                               or utt_id is not None):
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

    if utt_id is not None:
        logger.info(f"🔍 进入单样本微观解剖模式，正在寻找样本: {utt_id}")
    else:
        logger.info(f"🌍 进入全量宏观推断模式，总数: {test_iter}")

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
                enroll_only_out, _ = original_usef_compute(
                    enroll_repr, enroll_repr)
                static_emb = enroll_only_out.mean(dim=-1, keepdim=True)
                _, mix_only_out = original_usef_compute(mix_repr, mix_repr)
                T_m = mix_repr.shape[-1]
                static_enroll_expanded = static_emb.expand(
                    *static_emb.shape[:-1], T_m)
                return static_enroll_expanded, mix_only_out

            elif mode == 'static_inter':
                enroll_usef, mix_usef = original_usef_compute(
                    enroll_repr, mix_repr)
                enroll_usef = enroll_usef.mean(
                    dim=-1, keepdim=True).expand_as(enroll_usef)
                return enroll_usef, mix_usef

            else:
                return original_usef_compute(enroll_repr, mix_repr)

        def hooked_usef_post(self, mix_repr, feat_repr):
            usef_features['current'] = feat_repr.detach().clone()
            return original_usef_post(mix_repr, feat_repr)

        target_module.spk_ft.usef.compute = types.MethodType(
            hooked_usef_compute, target_module.spk_ft.usef)
        target_module.spk_ft.usef.post = types.MethodType(
            hooked_usef_post, target_module.spk_ft.usef)

    plot_data_list = []

    # ========================================================
    # 🏃 推断主循环
    # ========================================================
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            key = batch["key"]

            # 🎯 单样本模式拦截器
            if utt_id is not None and key[0] != utt_id:
                continue

            mix, cues, target = extract_model_inputs(batch, device)

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

            # 能量权重
            safe_target_for_stft = target_spk1.view(1, -1)
            tgt_spec = model.sep_model.stft(safe_target_for_stft)[-1]
            tgt_mag = torch.abs(tgt_spec)
            total_energy = tgt_mag.sum() + 1e-8
            weight_matrix = tgt_mag / total_energy

            # 提取一维音频数据
            ref_1 = target_spk1.detach().cpu().numpy().flatten()
            mix_1 = mix_spk1.detach().cpu().numpy().flatten()
            int_1 = interf_spk1.detach().cpu().numpy().flatten()
            enr_1 = cues_spk1[0].detach().cpu().numpy().flatten()

            # 🎧 如果是单样本模式，保存 4 个基础音轨
            if utt_id is not None:
                sf_sr = configs.get("dataset_args",
                                    {}).get("sample_rate", 16000)
                soundfile.write(
                    os.path.join(save_audio_dir, f"{key[0]}_00_Mix.wav"),
                    mix_1, sf_sr)
                soundfile.write(
                    os.path.join(save_audio_dir, f"{key[0]}_01_Enroll.wav"),
                    enr_1, sf_sr)
                soundfile.write(
                    os.path.join(save_audio_dir, f"{key[0]}_02_Target.wav"),
                    ref_1, sf_sr)
                soundfile.write(
                    os.path.join(save_audio_dir,
                                 f"{key[0]}_03_Interferer.wav"), int_1, sf_sr)

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
                F_mix_current = usef_features['current'].clone()

                sim_tgt = (
                    F.cosine_similarity(F_mix_current, F_tgt_true, dim=1) *
                    weight_matrix).sum().item()
                sim_int = (
                    F.cosine_similarity(F_mix_current, F_int_true, dim=1) *
                    weight_matrix).sum().item()
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

                # 🎧 单样本模式：保存各级别的增强音频
                if utt_id is not None:
                    soundfile.write(
                        os.path.join(save_audio_dir,
                                     f"{key[0]}_{method_name}.wav"),
                        ests[:end_idx], sf_sr)

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

            # 打印仪表盘
            if utt_id is not None:
                logger.info(f"🎯 样本锁定成功！ID: {key[0]}")
                logger.info(
                    f"  [Zero ] Fid={log_metrics['1_Static_Zero']['Fid']:.4f} | Leak={log_metrics['1_Static_Zero']['Leak']:.4f} | Margin={log_metrics['1_Static_Zero']['Margin']:+.4f} | SNRi={log_metrics['1_Static_Zero']['SNRi']:+.2f}dB"
                )
                logger.info(
                    f"  [Inter] Fid={log_metrics['2_Static_Inter']['Fid']:.4f} | Leak={log_metrics['2_Static_Inter']['Leak']:.4f} | Margin={log_metrics['2_Static_Inter']['Margin']:+.4f} | SNRi={log_metrics['2_Static_Inter']['SNRi']:+.2f}dB"
                )
                logger.info(
                    f"  [Dyn  ] Fid={log_metrics['3_Dynamic']['Fid']:.4f} | Leak={log_metrics['3_Dynamic']['Leak']:.4f} | Margin={log_metrics['3_Dynamic']['Margin']:+.4f} | SNRi={log_metrics['3_Dynamic']['SNRi']:+.2f}dB"
                )
                break  # 🛑 找到目标后立刻退出循环！
            elif (i + 1) % 100 == 0:
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

    if sign_save_wav and utt_id is None:
        generate_enahnced_scp(os.path.abspath(save_audio_dir), extension="wav")

    logger.info(f"Total Time: {end - start:.1f}s")

    # 🛑 如果是单样本分析模式，到此结束，不画图了。
    if utt_id is not None:
        logger.info(f"✅ 绝杀完毕！该样本的 7 个音频（4基础+3推断）已落盘至 {save_audio_dir}")
        return

    # ========================================================
    # 🧮 全局均值统计 & 困难/简单样本 分组深度解剖 (全量模式)
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

    df_zero = df[df["Method"] == "1_Static_Zero"].set_index("Utterance")
    df_dyn = df[df["Method"] == "3_Dynamic"].set_index("Utterance")

    df_compare = df_dyn[["Output_SISNRi",
                         "Margin"]].join(df_zero[["Output_SISNRi", "Margin"]],
                                         lsuffix="_Dyn",
                                         rsuffix="_Zero")
    df_compare["SNRi_Gap_Dyn_vs_Zero"] = df_compare[
        "Output_SISNRi_Dyn"] - df_compare["Output_SISNRi_Zero"]

    gap_threshold = 3.0
    hard_cases_utts = df_compare[
        df_compare["SNRi_Gap_Dyn_vs_Zero"] > gap_threshold].index
    easy_cases_utts = df_compare[
        df_compare["SNRi_Gap_Dyn_vs_Zero"] <= gap_threshold].index

    df_hard_cases = df_compare.loc[hard_cases_utts].sort_values(
        by="SNRi_Gap_Dyn_vs_Zero", ascending=False)
    hard_cases_path = os.path.join(configs["exp_dir"],
                                   "hard_cases_high_gap.csv")
    df_hard_cases.to_csv(hard_cases_path)

    logger.info("=" * 50)
    logger.info(
        f"🔥 [2] 发现 {len(hard_cases_utts)} 个产生了极端 Gap (> {gap_threshold}dB) 的生死局！名单已保存。"
    )

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
    # 🎨 三重星云绘图模块
    # ========================================================
    logger.info("🎨 正在生成三级消融散点分析图...")
    sns.set_theme(style="whitegrid", font_scale=1.2)
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

    # --- 图 2：Leakage ---
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
