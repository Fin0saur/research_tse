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

            mix_spk1 = mix[0:1, :, :]  # 提取 spk1 的混合音 [1, 1, T]
            target_spk1 = target[0:1, :, :] if target.dim() == 3 else target[
                0:1, 0:1, :]  # 提取纯净张三 [1, 1, T]

            # 提取原版属于张三的 cue (防爆关键！)
            original_cue_spk1 = [cues[0][0:1, :]] if cues is not None else None

            # ========================================================
            # 👼 步骤 1：获取绝对真理 F_true (上帝视角 Usef(target_spk1, target_spk1))
            # ========================================================
            target_as_cue_spk1 = [target_spk1.view(1, -1)]
            _ = model(target_spk1, target_as_cue_spk1)
            F_true_spk1 = usef_features['current'].clone(
            )  # 拿到属于张三的纯净后验 [1, 128, F, T]

            # ========================================================
            # ✂️ 步骤 2：构造属于张三的不同 Enroll (控制变量)
            # ========================================================
            fs = 16000
            enroll_variants = {
                "Original": original_cue_spk1,
                "Short_1s": [target_spk1.view(1, -1)[:, :fs * 1]],
                "Mid_2s": [target_spk1.view(1, -1)[:, -fs * 2:]],
                "Full_Oracle": target_as_cue_spk1
            }

            # ========================================================
            # 🏃 步骤 3：固定 Mix_Spk1，遍历 Enroll
            # ========================================================
            for condition_name, current_cue in enroll_variants.items():

                if current_cue is not None and current_cue[0].shape[-1] == 0:
                    continue

                # 1. 跑推断：确保送进去的只有 1 句话，1 个 cue
                outputs_mix = model(
                    mix_spk1, current_cue
                ) if current_cue is not None else model(mix_spk1)
                F_mix_spk1 = usef_features['current'].clone()

                # 2. 计算与真理的相似度 Sim (通道 dim=1)
                sim_score = F.cosine_similarity(F_mix_spk1, F_true_spk1,
                                                dim=1).mean().item()

                # 3. 极其严谨地提取一维波形
                # outputs_mix 的形状应该是 [1, T] 或者 [1, 1, T]
                outputs_np = outputs_mix.detach().cpu().numpy()
                if outputs_np.ndim >= 2:
                    ests_1 = outputs_np.flatten()
                else:
                    ests_1 = outputs_np

                ref_1 = target_spk1.detach().cpu().numpy().flatten()
                mix_1 = mix_spk1.detach().cpu().numpy().flatten()

                # 4. 归一化预测波形 (防止幅度极小导致 -160dB)
                if np.max(np.abs(ests_1)) > 0:
                    ests_1 = ests_1 / np.max(np.abs(ests_1)) * 0.9

                # 5. 计算 SI-SNR
                end1 = min(len(ests_1), len(ref_1), len(mix_1))
                SISNR1, _ = cal_SISNRi(ests_1[:end1], ref_1[:end1],
                                       mix_1[:end1])

                logger.info(
                    f"Utt={key[0]} | Cond={condition_name} | Sim={sim_score:.4f} | SI-SNR={SISNR1:.2f} dB"
                )

                plot_data_list.append({
                    "Utterance": key[0],
                    "Condition": condition_name,
                    "Fidelity": sim_score,
                    "Output_SISNR": SISNR1
                })

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
