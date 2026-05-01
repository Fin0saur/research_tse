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

            # --------------------------------------------------------
            # 🏃 1. 跑现实：正常推断混合音频，获取 F_mix
            # --------------------------------------------------------
            outputs_mix = model(mix, cues) if cues is not None else model(mix)

            # F_mix 形状为 [B, 128, F, T]，包含了现实中的后验特征
            F_mix = usef_features['current'].clone()

            # --------------------------------------------------------
            # 👼 2. 跑真理：极其暴力的 Oracle Self-Attention
            # --------------------------------------------------------
            # 解决形状诡计：强行把 target 变成和 mix 一模一样的形状 [B, 1, T]
            target_in = target.view(mix.shape)

            # 极其关键：把 target 伪装成 cues (enrollment)！
            # cues 通常是一个 list，里面装的是 [B, T] 的 tensor
            target_as_cue = [target.view(mix.shape[0], -1)]

            # 终极上帝视角：输入是目标，提示音也是目标本身！Usef(target, target)
            _ = model(target_in, target_as_cue)

            # 此时的特征，是绝对完美的自注意力动态轨迹！
            F_true = usef_features['current'].clone()

            # --------------------------------------------------------
            # 📐 3. 计算保真度相似度 (Fidelity)
            # --------------------------------------------------------
            # dim=1 是 128维的通道维度。计算结果为 [B, F, T]
            sim_batch = F.cosine_similarity(F_mix, F_true, dim=1)

            # 在时间和频率上求平均，得到每个 Batch 样本的最终保真度
            sim_scores = sim_batch.mean(dim=[-1, -2])

            sim_spk1 = sim_scores[0].item()
            # 如果是双人分离 (Batch=2)，则取第二个元素；否则取第一个
            sim_spk2 = sim_scores[1].item(
            ) if sim_scores.shape[0] > 1 else sim_spk1

            # --- 恢复原始脚本的处理和打分逻辑 ---
            outputs = outputs_mix
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            if torch.min(outputs.max(dim=1).values) > 0:
                outputs = ((outputs /
                            abs(outputs).max(dim=1, keepdim=True)[0] *
                            0.9).cpu().numpy())
            else:
                outputs = outputs.cpu().numpy()

            if sign_save_wav:
                file1 = os.path.join(
                    save_audio_dir,
                    f"Utt{total_cnt + 1}-{key[0]}-T{spk[0]}.wav")
                soundfile.write(file1, outputs[0], sample_rate)
                file2 = os.path.join(
                    save_audio_dir,
                    f"Utt{total_cnt + 1}-{key[1]}-T{spk[1]}.wav")
                soundfile.write(file2, outputs[1], sample_rate)

            ref = target.cpu().numpy()
            ests = outputs
            mix_np = mix.cpu().numpy()

            if mix_np.ndim == 3:
                mix_ref = mix_np[:, 0, :]
                mix_ref = np.expand_dims(mix_ref, axis=1)
            else:
                mix_ref = mix_np

            # --- 评估 Speaker 1 ---
            if ests[0].size != ref[0].size:
                end = min(ests[0].size, ref[0].size, mix_ref[0].size)
                ests_1 = ests[0][:end]
                ref_1 = ref[0][:end]
                mix_1 = mix_ref[0][:end]
                SISNR1, delta1 = cal_SISNRi(ests_1, ref_1, mix_1)
            else:
                SISNR1, delta1 = cal_SISNRi(ests[0], ref[0], mix_ref[0])

            logger.info(
                "Num={} | Utt={} | Target spk={} | SI-SNR={:.2f} | Sim={:.4f}".
                format(total_cnt + 1, key[0], spk[0], SISNR1, sim_spk1))
            total_SISNR += SISNR1
            total_SISNRi += delta1
            total_cnt += 1
            if delta1 > 1: accept_cnt += 1
            plot_data_list.append({
                "Fidelity": sim_spk1,
                "Output_SISNR": SISNR1
            })

            # --- 评估 Speaker 2 ---
            if ests[1].size != ref[1].size:
                end = min(ests[1].size, ref[1].size, mix_ref[1].size)
                ests_2 = ests[1][:end]
                ref_2 = ref[1][:end]
                mix_2 = mix_ref[1][:end]
                SISNR2, delta2 = cal_SISNRi(ests_2, ref_2, mix_2)
            else:
                SISNR2, delta2 = cal_SISNRi(ests[1], ref[1], mix_ref[1])

            logger.info(
                "Num={} | Utt={} | Target spk={} | SI-SNR={:.2f} | Sim={:.4f}".
                format(total_cnt + 1, key[1], spk[1], SISNR2, sim_spk2))
            total_SISNR += SISNR2
            total_SISNRi += delta2
            total_cnt += 1
            if delta2 > 1: accept_cnt += 1
            plot_data_list.append({
                "Fidelity": sim_spk2,
                "Output_SISNR": SISNR2
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
