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

# === 新增绘图依赖 ===
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
    sign_save_wav = configs.get("save_wav", False)  # 建议画图时关掉存波形，跑得快

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

    logger = get_logger(configs["exp_dir"], "infer_boxplot_0db.log")
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
    logger.info("test number: {}".format(test_iter))

    # ========================================================
    # 🔪 Hook 注入：截获纯粹的 Attention 特征
    # ========================================================
    usef_features = {}
    target_module = model.module if hasattr(model, 'module') else model

    if target_module.spk_configs['features']['usef']['enabled']:
        original_usef_post = target_module.spk_ft.usef.post

        def hooked_usef_post(self, mix_repr, feat_repr):
            usef_features['current'] = feat_repr.detach().clone()
            return original_usef_post(mix_repr, feat_repr)

        target_module.spk_ft.usef.post = types.MethodType(
            hooked_usef_post, target_module.spk_ft.usef)
        logger.info("🔪 Hook 注入成功！准备提取 Margin 特征...")
    else:
        logger.warning("USEF is not enabled.")

    plot_data_list = []
    logger.info("🏃 开始跑分并计算 Margin...")

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            mix, cues, target = extract_model_inputs(batch, device)

            # --- 准备真实信号 ---
            spk_idx = 0
            B, C_mix, T_mix = mix.shape
            if target.dim() == 4:
                oracle_target_wav = target[:, spk_idx, :, :]
            elif target.dim() == 3 and target.shape[1] == C_mix:
                oracle_target_wav = target
            else:
                oracle_target_wav = target[:, spk_idx, :].unsqueeze(1).expand(
                    B, C_mix, -1)

            min_T = min(oracle_target_wav.shape[-1], T_mix)
            oracle_target_wav = oracle_target_wav[..., :min_T].clone()
            interf_tensor = mix[..., :min_T] - oracle_target_wav

            # --- 1. 提取混合特征 F_mix ---
            outputs = model(mix[..., :min_T],
                            cues) if cues is not None else model(
                                mix[..., :min_T])
            if isinstance(outputs, (list, tuple)): outputs = outputs[0]
            F_mix = usef_features['current'][0]  # [128, F, T]

            # --- 2. 提取纯目标特征 F_tgt ---
            _ = model(oracle_target_wav, cues)
            F_tgt = usef_features['current'][0]

            # --- 3. 提取纯干扰特征 F_int ---
            _ = model(interf_tensor, cues)
            F_int = usef_features['current'][0]

            # --- 4. 计算 Margin (核心逻辑) ---
            # dim=0 表示在 128 维特征通道上计算余弦相似度，结果形状为 [F, T]
            sim_tgt = F.cosine_similarity(F_mix, F_tgt, dim=0)
            sim_int = F.cosine_similarity(F_mix, F_int, dim=0)

            # Margin = 靠近目标的程度 - 靠近干扰的程度
            margin_matrix = sim_tgt - sim_int
            margin_score = margin_matrix.mean().item()

            # --- 5. 计算 SI-SNR ---
            if torch.min(outputs.max(dim=1).values) > 0:
                outputs = ((outputs /
                            abs(outputs).max(dim=1, keepdim=True)[0] *
                            0.9).cpu().numpy())
            else:
                outputs = outputs.cpu().numpy()

            mix_np = mix.cpu().numpy()[0]
            mix_1d = mix_np[0] if mix_np.ndim == 2 else mix_np
            ref_np = target.cpu().numpy()[0]
            ref1_1d = ref_np[0, 0] if target.dim() == 4 else (
                ref_np[0] if target.dim() == 3 else ref_np)

            est1_1d = outputs[0].flatten()
            end1 = min(len(est1_1d), len(ref1_1d), len(mix_1d))
            SISNR1, delta1 = cal_SISNRi(est1_1d[:end1], ref1_1d[:end1],
                                        mix_1d[:end1])

            plot_data_list.append({
                "Margin": margin_score,
                "Output_SISNR": SISNR1
            })

            if (i + 1) % 100 == 0:
                logger.info(
                    f"Progress: {i+1}/{test_iter} | Margin: {margin_score:.6f} | SI-SNR: {SISNR1:.2f} dB"
                )

    # ========================================================
    # 🎨 数据处理与绘图 (散点图 + 箱线图)
    # ========================================================
    logger.info("📊 开始绘制 Margin 散点图与箱线图...")
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.DataFrame(plot_data_list)
    df.to_csv(os.path.join(configs["exp_dir"], "margin_sisnr_0db.csv"),
              index=False)

    sns.set_theme(style="whitegrid", font_scale=1.2)
    global_mean = df['Output_SISNR'].mean()

    # --------------------------------------------------------
    # 📈 图 1：散点图 (Scatter Plot)
    # --------------------------------------------------------
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 8))
    sns.regplot(x='Margin',
                y='Output_SISNR',
                data=df,
                ax=ax_scatter,
                scatter_kws={
                    'alpha': 0.3,
                    's': 40,
                    'color': '#2ECC71',
                    'edgecolor': 'none'
                },
                line_kws={
                    'color': 'crimson',
                    'linewidth': 3,
                    'label': 'Linear Fit'
                })
    ax_scatter.axhline(global_mean, color='grey', linestyle='--', linewidth=2)
    ax_scatter.set_title("Output SI-SNR vs Attention Margin",
                         fontsize=16,
                         fontweight='bold')
    ax_scatter.set_xlabel("Margin (Sim_tgt - Sim_int)", fontsize=14)
    ax_scatter.set_ylabel("Absolute Output SI-SNR (dB)", fontsize=14)
    fig_scatter.tight_layout()
    fig_scatter.savefig(os.path.join(configs["exp_dir"],
                                     "scatter_0db_margin.png"),
                        dpi=300)
    plt.close(fig_scatter)

    # --------------------------------------------------------
    # 📦 图 2：箱线图 (Box Plot)
    # --------------------------------------------------------
    fig_box, ax_box = plt.subplots(figsize=(10, 8))

    # 针对极其密集的分布，使用 qcut（加入 duplicates='drop' 防止边缘完全重叠报错）
    labels = [
        'Low Margin\n(Negative/Ambiguous)', 'Medium Margin',
        'High Margin\n(Target Biased)'
    ]
    try:
        df['Margin_Bin'] = pd.qcut(df['Margin'],
                                   q=3,
                                   labels=labels,
                                   duplicates='drop')
    except:
        # 如果退化得太厉害连分位数都切不开，就用平均切割
        df['Margin_Bin'] = pd.cut(df['Margin'], bins=3, labels=labels)

    palette = {
        "Low Margin\n(Negative/Ambiguous)": "#A9C4EB",
        "Medium Margin": "#CCCCCC",
        "High Margin\n(Target Biased)": "#F5BCA9"
    }

    sns.boxplot(x='Margin_Bin',
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
    ax_box.set_title("Output SI-SNR vs Attention Margin (Boxplot)",
                     fontsize=16,
                     fontweight='bold')
    ax_box.set_xlabel("")
    ax_box.set_ylabel("Absolute Output SI-SNR (dB)", fontsize=14)

    fig_box.tight_layout()
    fig_box.savefig(os.path.join(configs["exp_dir"], "boxplot_0db_margin.png"),
                    dpi=300)
    plt.close(fig_box)

    logger.info("🎉 Margin 绘图完成！")


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
