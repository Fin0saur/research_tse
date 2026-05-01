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
from scipy.stats import pearsonr

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


# =====================================================================
# 🪝 核心探针：动态劫持 USEF 提取 "能量侵占率 (Energy Alignment Ratio)"
# =====================================================================
def inject_joint_metrics_hook(model, global_metrics_dict):
    """
    寻找模型中的 USEF_attentionblock，截获输入(Mix)与输出(Posterior)计算能量比。
    """

    def find_usef_att(module):
        if hasattr(module, 'usef_att'): return module.usef_att
        for name, child in module.named_children():
            res = find_usef_att(child)
            if res is not None: return res
        return None

    attention_block = find_usef_att(model)
    if attention_block is None:
        raise RuntimeError("未在模型中找到 USEF_attentionblock！")

    # 替换原本的 forward 方法
    def hooked_forward(self, batch, aux):
        # batch: [B, C, T, Q] -> Mixture 特征
        # aux:   [B, C, T, Q] -> Enroll 特征
        B, _, old_T, old_Q = batch.shape
        aux_T = aux.shape[-2]

        Q = self["attn_norm_Q"](self["attn_conv_Q"](batch))
        K = self["attn_norm_K"](self["attn_conv_K"](aux))
        V = self["attn_norm_V"](self["attn_conv_V"](aux))

        Q = Q.view(-1, *Q.shape[2:])
        K = K.view(-1, *K.shape[2:])
        V = V.view(-1, *V.shape[2:])

        Q = Q.transpose(1, 2).flatten(start_dim=2)
        K = K.transpose(2, 3).contiguous().view([B * self.n_head, -1, aux_T])
        V = V.transpose(1, 2)

        old_shape = V.shape
        V = V.flatten(start_dim=2)
        emb_dim = Q.shape[-1]

        # Attention 计算
        attn_mat = torch.matmul(Q, K) / (emb_dim**0.5)
        attn_mat = F.softmax(attn_mat, dim=2)

        V = torch.matmul(attn_mat, V)
        V = V.reshape([old_shape[0], old_T, old_shape[-2], old_shape[-1]])
        V = V.transpose(1, 2)
        emb_dim = V.shape[1]

        # batch_out: [B, C, T, Q] -> Posterior 特征 (与 input batch 形状完全一致)
        batch_out = V.contiguous().view(
            [B, self.n_head * emb_dim, old_T, old_Q])
        batch_out = self["attn_concat_proj"](batch_out)

        # ----------------------------------------------------
        # 🌟 联合评估指标计算 (Joint Evaluation Metrics)
        # ----------------------------------------------------
        # 🎯 1. 能量侵占率 (Alignment Energy Ratio)
        # 计算 Posterior 在 Mixture 上的投影能量占比： sum(V' * X) / sum(X * X)
        dot_product = torch.sum(batch * batch_out).item()
        mix_energy = torch.sum(batch * batch).item() + 1e-8
        alignment_ratio = dot_product / mix_energy

        # 🎯 2. L2 范数比 (L2 Norm Ratio)
        # 计算绝对强度的变化
        norm_ratio = (torch.norm(batch_out) /
                      (torch.norm(batch) + 1e-8)).item()

        global_metrics_dict["Alignment_Ratio"] = alignment_ratio
        global_metrics_dict["L2_Norm_Ratio"] = norm_ratio

        return batch_out

    # 挂载 Hook
    attention_block.forward = types.MethodType(hooked_forward, attention_block)
    return True


# =====================================================================


def infer(config="confs/conf.yaml", **kwargs):
    start = time.time()
    configs = parse_config_or_kwargs(config, **kwargs)
    set_seed(configs["seed"])
    device = torch.device(
        f"cuda:{configs['gpus']}" if configs["gpus"] >= 0 else "cpu")

    if 'spk_model_init' in configs['model_args']['tse_model']:
        configs['model_args']['tse_model']['spk_model_init'] = False

    model = get_model(configs["model"]["tse_model"])(
        configs["model_args"]["tse_model"])
    load_pretrained_model(model, configs["checkpoint"])

    logger = get_logger(configs["exp_dir"], "infer_energy_ratio.log")
    logger.info(f"Load checkpoint from {configs['checkpoint']}")

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

    logger.info("🔪 正在注入 '后验-混合' 联合能量探针...")
    global_metrics = {}
    inject_joint_metrics_hook(model, global_metrics)
    logger.info("✅ 探针注入成功！")

    plot_data_list = []

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            # if i >= 1000: break # 正式跑解开注释

            mix, cues, target = extract_model_inputs(batch, device)
            key = batch["key"]

            if cues is None or cues[0].shape[-1] == 0:
                continue

            ref_1 = (target[0:1, :, :] if target.dim() == 3 else
                     target[0:1, 0:1, :]).detach().cpu().numpy().flatten()
            mix_1 = mix[0:1, :, :].detach().cpu().numpy().flatten()

            # 🚀 网络前向传播，探针自动截获特征并计算 Ratio
            outputs_dyn = model(mix[0:1, :, :], [cues[0][0:1, :]])

            # 取出计算好的指标
            align_ratio = global_metrics.get("Alignment_Ratio", 0.0)
            norm_ratio = global_metrics.get("L2_Norm_Ratio", 0.0)

            # 计算 SI-SNRi
            ests_dyn = outputs_dyn[0].detach().cpu().numpy().flatten()
            if np.max(np.abs(ests_dyn)) > 0:
                ests_dyn = ests_dyn / np.max(np.abs(ests_dyn)) * 0.9
            end_idx = min(len(ests_dyn), len(ref_1), len(mix_1))
            _, snri_value = cal_SISNRi(ests_dyn[:end_idx], ref_1[:end_idx],
                                       mix_1[:end_idx])

            plot_data_list.append({
                "Utterance": key[0],
                "Alignment_Ratio": align_ratio,
                "L2_Norm_Ratio": norm_ratio,
                "Output_SISNRi": snri_value
            })

            logger.info(
                f"[{i+1}] SNRi: {snri_value:+.1f}dB | Align_Ratio: {align_ratio:.4f} | L2_Ratio: {norm_ratio:.4f}"
            )

    logger.info(f"Inference Time: {time.time() - start:.1f}s")

    # ========================================================
    # 🧮 统计分析与双指标图表绘制
    # ========================================================
    df = pd.DataFrame(plot_data_list)
    df.to_csv(os.path.join(configs["exp_dir"], "joint_energy_metrics.csv"),
              index=False)

    # 计算皮尔逊相关系数
    r_align, p_align = pearsonr(df["Alignment_Ratio"], df["Output_SISNRi"])
    r_norm, p_norm = pearsonr(df["L2_Norm_Ratio"], df["Output_SISNRi"])

    logger.info("=" * 50)
    logger.info(f"📊 联合评估指标相关性分析 (Pearson R):")
    logger.info(f" 1. 能量侵占率 (Alignment Ratio):  R = {r_align:+.4f} (核心指标)")
    logger.info(f" 2. L2范数比 (L2 Norm Ratio):      R = {r_norm:+.4f}")
    logger.info("=" * 50)

    # 画图
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 图 1：能量侵占率 (点乘投影)
    sns.regplot(data=df,
                x="Alignment_Ratio",
                y="Output_SISNRi",
                ax=axes[0],
                scatter_kws={
                    'alpha': 0.5,
                    'color': '#d35400'
                },
                line_kws={
                    'color': '#e67e22',
                    'linewidth': 3
                })
    axes[0].set_title(
        f"Energy Alignment Ratio vs SI-SNRi\n(R = {r_align:.3f})",
        fontweight="bold")
    axes[0].set_xlabel(
        "Alignment Ratio $\Sigma(V' \cdot X) / \Sigma(X \cdot X)$")

    # 图 2：L2 范数比
    sns.regplot(data=df,
                x="L2_Norm_Ratio",
                y="Output_SISNRi",
                ax=axes[1],
                scatter_kws={
                    'alpha': 0.5,
                    'color': '#2980b9'
                },
                line_kws={
                    'color': '#3498db',
                    'linewidth': 3
                })
    axes[1].set_title(f"L2 Norm Ratio vs SI-SNRi\n(R = {r_norm:.3f})",
                      fontweight="bold")
    axes[1].set_xlabel("L2 Norm Ratio $||V'||_2 / ||X||_2$")

    plt.tight_layout()
    plt.savefig(os.path.join(configs["exp_dir"],
                             "plot_4_Joint_Energy_Metrics.png"),
                dpi=300,
                bbox_inches='tight')
    plt.close()

    logger.info("🎉 '后验-混合' 联合能量指标跑通！查看散点图确认异常值拉扯效果。")


def extract_model_inputs(batch, device):
    mix = batch["wav_mix"].float().to(device)
    target = batch["wav_target"].float().to(device)
    cues = [
        batch[k].float().to(device) for k in AUX_KEY_MAP.values()
        if k in batch and batch[k] is not None
    ]
    return mix, cues if len(cues) > 0 else None, target


if __name__ == "__main__":
    fire.Fire(infer)
