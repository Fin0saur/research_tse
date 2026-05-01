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

# === 新增数据分析和绘图依赖 ===
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
    sign_save_wav = configs.get("save_wav", False)  # 跑分析实验建议关掉保存wav，省硬盘和时间

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

    logger = get_logger(configs["exp_dir"], "infer_diagnostic.log")
    logger.info("Load checkpoint from {}".format(model_path))

    save_audio_dir = os.path.join(configs["exp_dir"], "audio")
    if sign_save_wav:
        if not os.path.exists(save_audio_dir):
            try:
                os.makedirs(save_audio_dir)
                logger.info(
                    f"Directory {save_audio_dir} created successfully.")
            except OSError as e:
                logger.error(f"Error creating directory {save_audio_dir}: {e}")
    else:
        logger.info("Do NOT save the results in wav to speed up analysis.")

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
    else:
        logger.warning("USEF is not enabled. Cannot extract features.")

    plot_data_list = []  # 用于收集画图的数据

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            mix, cues, target = extract_model_inputs(batch, device)
            key = batch["key"]

            # --- 极其重要：物理拆包，确保纯粹的单人环境 ---
            # DataLoader 的 mix 形状通常是 [2, 1, T] (针对 Libri2Mix 两人分离)
            # 我们强制剥离出 Speaker 1 的所有相关数据，防止跨 Batch 广播错误 (-160dB 惨案)
            mix_spk1 = mix[0:1, :, :]  # [1, 1, T]
            target_spk1 = target[0:1, :, :] if target.dim() == 3 else target[
                0:1, 0:1, :]  # [1, 1, T]
            interf_spk1 = target[1:2, :, :] if target.dim() == 3 else target[
                1:2, 0:1, :]  # [1, 1, T]

            cues_spk1 = [cues[0][0:1, :]] if cues is not None else None

            # ========================================================
            # 👼 步骤 1：获取“完美张三”和“完美李四”的真理特征
            # ========================================================
            target_as_cue_spk1 = [target_spk1.view(1, -1)]
            _ = model(target_spk1, target_as_cue_spk1)
            F_tgt_true = usef_features['current'].clone(
            )  # 张三真理 [1, 128, F, T]

            interf_as_cue_spk1 = [interf_spk1.view(1, -1)]
            _ = model(interf_spk1, interf_as_cue_spk1)
            F_int_true = usef_features['current'].clone(
            )  # 李四真理 [1, 128, F, T]

            # ========================================================
            # ⚖️ 步骤 2：生成目标音频的频域能量权重 (Soft Energy Weighting)
            # ========================================================
            # 解决 4D 报错：极其安全地拉平成 [1, Time] 喂给 STFT
            # 无论原来是 [1, 1, 1, T] 还是 [1, 1, T]，view(1, -1) 统统变成 [1, T]
            safe_target_for_stft = target_spk1.view(1, -1)

            # 使用模型自带的 STFT 提取张三纯净音频的声谱图
            tgt_spec = model.sep_model.stft(safe_target_for_stft)[-1]
            tgt_mag = torch.abs(tgt_spec)  # [1, F, T]

            # 计算全局能量占比矩阵
            total_energy = tgt_mag.sum() + 1e-8
            weight_matrix = tgt_mag / total_energy

            # ========================================================
            # ✂️ 步骤 3：构造不同的 Enroll (控制变量)
            # ========================================================
            fs = sample_rate
            enroll_variants = {
                "Original": cues_spk1,
                "Short_1s": [target_spk1.view(1, -1)[:, :fs * 1]],
                "Mid_2s": [target_spk1.view(1, -1)[:, -fs * 2:]],
                "Full_Oracle": target_as_cue_spk1
            }

            # ========================================================
            # 🏃 步骤 4：固定 Mix_Spk1，遍历不同的 Enroll 条件
            # ========================================================
            for condition_name, current_cue in enroll_variants.items():

                # 如果音频太短不够切，直接跳过这个条件
                if current_cue is not None and current_cue[0].shape[-1] == 0:
                    continue

                # --- a) 在当前 Enrollment 条件下进行推断 ---
                outputs_mix = model(
                    mix_spk1, current_cue
                ) if current_cue is not None else model(mix_spk1)
                F_mix_spk1 = usef_features['current'].clone()

                # --- b) 计算能量加权保真度 (Fidelity) 和 干扰残留 (Leakage) ---
                sim_matrix_tgt = F.cosine_similarity(F_mix_spk1,
                                                     F_tgt_true,
                                                     dim=1)
                sim_matrix_int = F.cosine_similarity(F_mix_spk1,
                                                     F_int_true,
                                                     dim=1)

                # 加权求和：能量越大的时频像素，其相似度得分在总分里的占比越高
                sim_target = (sim_matrix_tgt * weight_matrix).sum().item()
                sim_interf = (sim_matrix_int * weight_matrix).sum().item()

                # --- c) 提取一维预测波形与参考波形 ---
                outputs_np = outputs_mix.detach().cpu().numpy()
                ests_1 = outputs_np.flatten(
                ) if outputs_np.ndim >= 2 else outputs_np

                ref_1 = target_spk1.detach().cpu().numpy().flatten()
                mix_1 = mix_spk1.detach().cpu().numpy().flatten()

                # 极其严谨的归一化，防止极小幅度导致 SI-SNR 内部截断为 -160dB
                if np.max(np.abs(ests_1)) > 0:
                    ests_1 = ests_1 / np.max(np.abs(ests_1)) * 0.9

                # --- d) 计算 SI-SNR ---
                end1 = min(len(ests_1), len(ref_1), len(mix_1))
                SISNR1, delta1 = cal_SISNRi(ests_1[:end1], ref_1[:end1],
                                            mix_1[:end1])

                # --- e) 记录日志与数据 ---
                logger.info(
                    f"Utt={key[0]} | Cond={condition_name:12s} | Tgt_Fidelity={sim_target:.4f} | Int_Leakage={sim_interf:.4f} | SI-SNR={SISNR1:.2f} dB"
                )

                plot_data_list.append({
                    "Utterance": key[0],
                    "Condition": condition_name,
                    "Target_Fidelity": sim_target,
                    "Interferer_Leakage": sim_interf,
                    "Output_SISNR": SISNR1
                })

                # 只有 Original 条件下的成绩才计入全局平均 (为了和传统评估对齐)
                if condition_name == "Original":
                    total_SISNR += SISNR1
                    total_SISNRi += delta1
                    total_cnt += 1
                    if delta1 > 1: accept_cnt += 1

            # 进度提示
            if (i + 1) % 50 == 0:
                logger.info(f"--- Processed {i+1}/{test_iter} mixtures ---")

        end = time.time()

    if sign_save_wav:
        generate_enahnced_scp(os.path.abspath(save_audio_dir), extension="wav")

    logger.info("=" * 50)
    logger.info("Time Elapsed: {:.1f}s".format(end - start))
    logger.info("Average SI-SNR (Original Cond): {:.2f}".format(
        total_SISNR / total_cnt if total_cnt > 0 else 0))
    logger.info("Average SI-SNRi (Original Cond): {:.2f}".format(
        total_SISNRi / total_cnt if total_cnt > 0 else 0))
    logger.info("Acceptance rate (> 1 dB): {:.2f}%".format((
        accept_cnt / total_cnt * 100) if total_cnt > 0 else 0))
    logger.info("=" * 50)

    # ========================================================
    # 🎨 终极绘图模块：保存分析数据 (如果需要，你可以自己用 pandas 画轨迹图)
    # ========================================================
    logger.info("📊 跑分结束，正在保存诊断数据 csv...")
    df = pd.DataFrame(plot_data_list)
    csv_path = os.path.join(configs["exp_dir"],
                            "diagnostic_fidelity_leakage.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"🎉 诊断数据已保存至: {csv_path}")


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
