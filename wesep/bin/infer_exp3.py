from __future__ import print_function

import os
import time
import types

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

# === 新增分析绘图依赖 ===
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


def infer(config="confs/conf.yaml", **kwargs):
    start = time.time()
    total_SISNR = 0
    total_SISNRi = 0
    total_cnt = 0
    accept_cnt = 0

    configs = parse_config_or_kwargs(config, **kwargs)
    sign_save_wav = configs.get("save_wav", True)

    rank = 0
    set_seed(configs["seed"] + rank)
    gpu = configs["gpus"]
    device = (torch.device("cuda:{}".format(gpu))
              if gpu >= 0 else torch.device("cpu"))

    sample_rate = configs.get("fs", None)
    if sample_rate is None or sample_rate == "16k":
        sample_rate = 16000
    else:
        sample_rate = 8000

    if 'spk_model_init' in configs['model_args']['tse_model']:
        configs['model_args']['tse_model']['spk_model_init'] = False
    model = get_model(configs["model"]["tse_model"])(
        configs["model_args"]["tse_model"])
    model_path = os.path.join(configs["checkpoint"])
    load_pretrained_model(model, model_path)

    logger = get_logger(configs["exp_dir"], "infer.log")
    logger.info("Load checkpoint from {}".format(model_path))
    save_audio_dir = os.path.join(configs["exp_dir"], "audio")
    if sign_save_wav:
        if not os.path.exists(save_audio_dir):
            try:
                os.makedirs(save_audio_dir)
                print(f"Directory {save_audio_dir} created successfully.")
            except OSError as e:
                print(f"Error creating directory {save_audio_dir}: {e}")
        else:
            print(f"Directory {save_audio_dir} already exists.")
    else:
        print("Do NOT save the results in wav.")

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
    logger.info("test number: {}".format(test_iter))

    # ========================================================
    # 🔪 Hook 注入：截获 USEF 后验特征
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
        logger.info("🔪 成功注入特征截获 Hook！")
    # ========================================================

    plot_data_list = []  # 收集绘图数据

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):

            mix, cues, target = extract_model_inputs(batch, device)
            spk = batch["spk"]
            key = batch["key"]

            mix_spk1 = mix[0:1, :, :]  # [1, 1, T]
            target_spk1 = target[0:1, :, :]  # [1, 1, T]
            interf_spk1 = target[1:2, :, :]  # [1, 1, T] 对 Spk1 来说，Spk2 就是干扰

            # 拆解 cues：只取属于 Spk1 的那一部分 cue
            cues_spk1 = [cues[0][0:1, :]] if cues is not None else None

            # ========================================================
            # 🏃 1. 跑现实：正常推断混合音频，获取 F_mix
            # ========================================================
            # 注意：这里送进去的都是 spk1 专属的 mix 和 cue
            outputs_mix = model(
                mix_spk1,
                cues_spk1) if cues_spk1 is not None else model(mix_spk1)
            F_mix_spk1 = usef_features['current'].clone(
            )  # 提取出来的形状应该是 [1, 128, F, T]

            # ========================================================
            # 👼 2. 跑真理 (Target)：提取完美的张三
            # ========================================================
            # 强行把 target 当成 cue，确保形状是 [1, T]
            target_as_cue_spk1 = [target_spk1.view(1, -1)]
            _ = model(target_spk1, target_as_cue_spk1)
            F_tgt_true = usef_features['current'].clone()  # [1, 128, F, T]

            # ========================================================
            # 👿 3. 跑假想敌 (Interferer)：提取完美的李四
            # ========================================================
            # 强行把 interferer 当成 cue，确保形状是 [1, T]
            interf_as_cue = [interf_spk1.view(1, -1)]
            _ = model(interf_spk1, interf_as_cue)
            F_int_true = usef_features['current'].clone()  # [1, 128, F, T]

            # ========================================================
            # 📐 4. 计算两大核心指标：保真度 (Fidelity) vs 干扰残留 (Leakage)
            # ========================================================
            # dim=1 是通道维度，求余弦相似度后在时间和频率 [F, T] 上求均值
            sim_target = F.cosine_similarity(F_mix_spk1, F_tgt_true,
                                             dim=1).mean().item()
            sim_interf = F.cosine_similarity(F_mix_spk1, F_int_true,
                                             dim=1).mean().item()

            # ========================================================
            # 📊 5. 计算 SI-SNR 并记录日志
            # ========================================================
            # 这里的 outputs_mix 是针对 spk1 的单人输出，剥离可能存在的 list
            outputs = outputs_mix[0] if isinstance(outputs_mix,
                                                   (list,
                                                    tuple)) else outputs_mix

            # 归一化处理
            if torch.min(outputs.max(dim=1).values) > 0:
                outputs = ((outputs /
                            abs(outputs).max(dim=1, keepdim=True)[0] *
                            0.9).cpu().numpy())
            else:
                outputs = outputs.cpu().numpy()

            # 暴力拉平，防止 cal_SISNRi 内部由于维度微小错位报 -160dB
            ests_1 = outputs.flatten()
            ref_1 = target_spk1.detach().cpu().numpy().flatten()
            mix_1 = mix_spk1.detach().cpu().numpy().flatten()

            end1 = min(len(ests_1), len(ref_1), len(mix_1))
            SISNR1, delta1 = cal_SISNRi(ests_1[:end1], ref_1[:end1],
                                        mix_1[:end1])

            # 记录数据 (去除了你原代码中重复 append 的一段)
            logger.info(
                f"Utt={key[0]} | Target Sim={sim_target:.4f} | Interferer Sim={sim_interf:.4f} | SI-SNR={SISNR1:.2f} dB"
            )

            plot_data_list.append({
                "Fidelity_to_Target": sim_target,
                "Leakage_to_Interferer": sim_interf,
                "Output_SISNR": SISNR1
            })

            total_SISNR += SISNR1
            total_SISNRi += delta1
            total_cnt += 1
            if delta1 > 1: accept_cnt += 1

        end = time.time()

    if sign_save_wav:
        generate_enahnced_scp(os.path.abspath(save_audio_dir), extension="wav")

    logger.info("Time Elapsed: {:.1f}s".format(end - start))
    logger.info("Average SI-SNR: {:.2f}".format(total_SISNR / total_cnt))

    # ========================================================
    # 🎨 绘制 Fidelity vs SI-SNR 分析图表
    # ========================================================
    logger.info("📊 跑分结束，正在生成散点图和箱线图...")
    df = pd.DataFrame(plot_data_list)
    df.to_csv(os.path.join(configs["exp_dir"], "fidelity_sisnr.csv"),
              index=False)

    sns.set_theme(style="whitegrid", font_scale=1.2)
    global_mean = df['Output_SISNR'].mean()

    # 1. 散点图
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 8))
    sns.regplot(x='Fidelity',
                y='Output_SISNR',
                data=df,
                ax=ax_scatter,
                scatter_kws={
                    'alpha': 0.4,
                    's': 30,
                    'color': '#9b59b6',
                    'edgecolor': 'none'
                },
                line_kws={
                    'color': 'crimson',
                    'linewidth': 3,
                    'label': 'Linear Trend'
                })
    ax_scatter.axhline(global_mean, color='grey', linestyle='--', linewidth=2)
    ax_scatter.set_title("Performance vs Posterior Fidelity (Oracle Truth)",
                         fontsize=16,
                         fontweight='bold')
    ax_scatter.set_xlabel("Fidelity (Cosine Similarity: Mix vs Oracle Target)",
                          fontsize=14)
    ax_scatter.set_ylabel("Absolute Output SI-SNR (dB)", fontsize=14)
    ax_scatter.legend()
    fig_scatter.tight_layout()
    fig_scatter.savefig(os.path.join(configs["exp_dir"],
                                     "scatter_fidelity.png"),
                        dpi=300)
    plt.close(fig_scatter)

    # 2. 箱线图
    fig_box, ax_box = plt.subplots(figsize=(10, 8))
    labels = [
        'Low Fidelity\n(Corrupted)', 'Medium Fidelity',
        'High Fidelity\n(Near Perfect)'
    ]
    try:
        df['Fidelity_Bin'] = pd.qcut(df['Fidelity'],
                                     q=3,
                                     labels=labels,
                                     duplicates='drop')
    except:
        df['Fidelity_Bin'] = pd.cut(df['Fidelity'], bins=3, labels=labels)

    palette = {
        "Low Fidelity\n(Corrupted)": "#A9C4EB",
        "Medium Fidelity": "#CCCCCC",
        "High Fidelity\n(Near Perfect)": "#F5BCA9"
    }
    sns.boxplot(x='Fidelity_Bin',
                y='Output_SISNR',
                data=df,
                ax=ax_box,
                palette=palette,
                width=0.4,
                flierprops={
                    "marker": "o",
                    "markerfacecolor": "none",
                    "markeredgecolor": "grey",
                    "alpha": 0.5
                })
    for line in ax_box.lines[4::6]:
        line.set_color('crimson')
        line.set_linewidth(3)

    ax_box.axhline(global_mean, color='grey', linestyle='--', linewidth=2)
    ax_box.set_title("Performance vs Posterior Fidelity (Boxplot)",
                     fontsize=16,
                     fontweight='bold')
    ax_box.set_xlabel("")
    ax_box.set_ylabel("Absolute Output SI-SNR (dB)", fontsize=14)
    fig_box.tight_layout()
    fig_box.savefig(os.path.join(configs["exp_dir"], "boxplot_fidelity.png"),
                    dpi=300)
    plt.close(fig_box)
    logger.info("🎉 绘图完成！图片已保存至 exp_dir 目录。")


def extract_model_inputs(batch, device):
    if "wav_mix" not in batch:
        raise RuntimeError("[executor] Missing required key: wav_mix")
    if "wav_target" not in batch:
        raise RuntimeError("[executor] Missing required key: wav_target")

    mix = batch["wav_mix"].float().to(device)
    target = batch["wav_target"].float().to(device)

    cues = []
    for k in list(AUX_KEY_MAP.values()):
        if k in batch and batch[k] is not None:
            cues.append(batch[k].float().to(device))

    if len(cues) == 0:
        cues = None

    return mix, cues, target


if __name__ == "__main__":
    fire.Fire(infer)
