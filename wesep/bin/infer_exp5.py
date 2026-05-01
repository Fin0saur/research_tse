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

    logger = get_logger(configs["exp_dir"], "infer_diagnostic.log")
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

    # ========================================================
    # 🔪 Hook 注入：使用字典传址，支持一键退化为静态 Baseline
    # ========================================================
    usef_features = {}
    usef_features['mode'] = 'dynamic'  # 👈 初始状态设定为 dynamic！不再用 global

    target_module = model.module if hasattr(model, 'module') else model

    if hasattr(target_module, 'spk_configs'
               ) and target_module.spk_configs['features']['usef']['enabled']:
        original_usef_compute = target_module.spk_ft.usef.compute
        original_usef_post = target_module.spk_ft.usef.post

        def hooked_usef_compute(self, enroll_repr, mix_repr):
            enroll_usef, mix_usef = original_usef_compute(
                enroll_repr, mix_repr)

            # 💡 核心黑魔法：直接读取字典里的状态！
            if usef_features.get('mode') == 'static':
                # 在时间轴求均值，强行抹杀动态对齐能力
                enroll_usef = enroll_usef.mean(
                    dim=-1, keepdim=True).expand_as(enroll_usef)

            return enroll_usef, mix_usef

        def hooked_usef_post(self, mix_repr, feat_repr):
            usef_features['current'] = feat_repr.detach().clone()
            return original_usef_post(mix_repr, feat_repr)

        target_module.spk_ft.usef.compute = types.MethodType(
            hooked_usef_compute, target_module.spk_ft.usef)
        target_module.spk_ft.usef.post = types.MethodType(
            hooked_usef_post, target_module.spk_ft.usef)
        logger.info("🔪 成功注入特征截获 & 在线消融 Hook！")

    plot_data_list = []

    # ========================================================
    # 📚 加载武器库
    # ========================================================
    try:
        with open(
                "/home/yxy05/code/research_tse/examples/audio/librimix/data/clean/test/cues/audio.json",
                "r",
                encoding="utf-8") as f:
            spk_audio_dict = json.load(f)
        logger.info("已加载 audio.json 作为随机 Enrollment 武器库！")
    except Exception as e:
        logger.error(f"加载 audio.json 失败: {e}. 请检查文件路径！")
        return

    # ========================================================
    # 🏃 推断主循环
    # ========================================================
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            if i >= 10:
                logger.info("已完成 10 个 Mixture 的轰炸实验，停止推断。")
                break

            mix, cues, target = extract_model_inputs(batch, device)
            spk = batch["spk"]
            key = batch["key"]
            target_spk_id = str(spk[0])

            # 物理拆包
            mix_spk1 = mix[0:1, :, :]
            target_spk1 = target[0:1, :, :] if target.dim() == 3 else target[
                0:1, 0:1, :]
            interf_spk1 = target[1:2, :, :] if target.dim() == 3 else target[
                1:2, 0:1, :]

            target_as_cue_spk1 = [target_spk1.view(1, -1)]
            _ = model(target_spk1, target_as_cue_spk1)
            F_tgt_true = usef_features['current'].clone()

            interf_as_cue_spk1 = [interf_spk1.view(1, -1)]
            _ = model(interf_spk1, interf_as_cue_spk1)
            F_int_true = usef_features['current'].clone()

            # 生成能量权重
            safe_target_for_stft = target_spk1.view(1, -1)
            tgt_spec = model.sep_model.stft(safe_target_for_stft)[-1]
            tgt_mag = torch.abs(tgt_spec)
            total_energy = tgt_mag.sum() + 1e-8
            weight_matrix = tgt_mag / total_energy

            # ==========================================
            # 🎙️ 获取可用的随机音源
            # ==========================================
            available_utts = spk_audio_dict.get(target_spk_id, [])
            if len(available_utts) == 0:
                logger.warning(f"找不到说话人 {target_spk_id} 的音频，跳过。")
                continue

            num_samples = 50
            logger.info(f"🚀 轰炸 Mixture {i+1} (Utt={key[0]})... 50 次双轨推断开始！")

            for idx in range(num_samples):
                # 随机裁剪
                item = random.choice(available_utts)
                wav_path = item["path"]
                try:
                    enroll_wav, sr = soundfile.read(wav_path)
                    total_len = len(enroll_wav)
                    min_len = min(sr * 1, total_len)
                    if total_len > min_len:
                        crop_len = random.randint(min_len, total_len)
                        start_idx = random.randint(0, total_len - crop_len)
                        crop_wav = enroll_wav[start_idx:start_idx + crop_len]
                    else:
                        crop_wav = enroll_wav
                    current_cue = [
                        torch.tensor(crop_wav,
                                     dtype=torch.float32).to(device).view(
                                         1, -1)
                    ]
                except Exception as e:
                    logger.error(f"裁剪失败: {e}")
                    continue

                ref_1 = target_spk1.detach().cpu().numpy().flatten()
                mix_1 = mix_spk1.detach().cpu().numpy().flatten()

                # ==========================================
                # 🐌 模式 A：退化版 Static Baseline
                # ==========================================
                usef_features['mode'] = 'static'  # 👈 拨动开关为 static
                outputs_static = model(mix_spk1, current_cue)
                F_mix_stat = usef_features['current'].clone()

                sim_stat_tgt = (
                    F.cosine_similarity(F_mix_stat, F_tgt_true, dim=1) *
                    weight_matrix).sum().item()
                sim_stat_int = (
                    F.cosine_similarity(F_mix_stat, F_int_true, dim=1) *
                    weight_matrix).sum().item()

                out_np_stat = outputs_static[0].detach().cpu().numpy(
                ) if isinstance(
                    outputs_static,
                    (list, tuple)) else outputs_static.detach().cpu().numpy()
                ests_stat = out_np_stat.flatten(
                ) if out_np_stat.ndim >= 2 else out_np_stat
                if np.max(np.abs(ests_stat)) > 0:
                    ests_stat = ests_stat / np.max(np.abs(ests_stat)) * 0.9
                end_s = min(len(ests_stat), len(ref_1), len(mix_1))
                SISNR_static, _ = cal_SISNRi(ests_stat[:end_s], ref_1[:end_s],
                                             mix_1[:end_s])

                # ==========================================
                # 🚀 模式 B：完全体 Dynamic USEF
                # ==========================================
                usef_features['mode'] = 'dynamic'  # 👈 拨回开关为 dynamic
                outputs_dynamic = model(mix_spk1, current_cue)
                F_mix_dyn = usef_features['current'].clone()

                sim_dyn_tgt = (
                    F.cosine_similarity(F_mix_dyn, F_tgt_true, dim=1) *
                    weight_matrix).sum().item()
                sim_dyn_int = (
                    F.cosine_similarity(F_mix_dyn, F_int_true, dim=1) *
                    weight_matrix).sum().item()

                out_np_dyn = outputs_dynamic[0].detach().cpu().numpy(
                ) if isinstance(
                    outputs_dynamic,
                    (list, tuple)) else outputs_dynamic.detach().cpu().numpy()
                ests_dyn = out_np_dyn.flatten(
                ) if out_np_dyn.ndim >= 2 else out_np_dyn
                if np.max(np.abs(ests_dyn)) > 0:
                    ests_dyn = ests_dyn / np.max(np.abs(ests_dyn)) * 0.9
                end_d = min(len(ests_dyn), len(ref_1), len(mix_1))
                SISNR_dynamic, _ = cal_SISNRi(ests_dyn[:end_d], ref_1[:end_d],
                                              mix_1[:end_d])

                # 记录数据... (保持不变)
                # 📝 记录数据
                # ==========================================
                plot_data_list.append({
                    "Mixture_ID": f"Mix_{i+1}",
                    "Method": "Static Baseline",
                    "Target_Fidelity": sim_stat_tgt,
                    "Interferer_Leakage": sim_stat_int,
                    "Output_SISNR": SISNR_static
                })

                plot_data_list.append({
                    "Mixture_ID": f"Mix_{i+1}",
                    "Method": "Dynamic USEF",
                    "Target_Fidelity": sim_dyn_tgt,
                    "Interferer_Leakage": sim_dyn_int,
                    "Output_SISNR": SISNR_dynamic
                })

                logger.info(
                    f"  -> Sample {idx:02d} | Static SNR={SISNR_static:.1f} | Dynamic SNR={SISNR_dynamic:.1f}"
                )

        end = time.time()

    if sign_save_wav:
        generate_enahnced_scp(os.path.abspath(save_audio_dir), extension="wav")

    logger.info(f"Total Time: {end - start:.1f}s")

    # ========================================================
    # 🎨 终极自动分面绘图模块 (微距放大版 + 对比 Method)
    # ========================================================
    logger.info("📊 生成对比分析图...")
    df = pd.DataFrame(plot_data_list)
    df.to_csv(os.path.join(configs["exp_dir"], "diagnostic.csv"), index=False)

    df["Confusion_Margin"] = df["Target_Fidelity"] - df["Interferer_Leakage"]

    # 剔除严重提错人的情况 (Margin <= 0)
    df_normal = df[df["Confusion_Margin"] > 0].copy()

    sns.set_theme(style="whitegrid", font_scale=1.0)

    if len(df_normal) > 0:
        # 画两张图，用 hue="Method" 区分红蓝两军！
        g1 = sns.lmplot(
            data=df_normal,
            x="Target_Fidelity",
            y="Output_SISNR",
            col="Mixture_ID",
            col_wrap=5,
            hue="Method",  # 极其关键：用颜色区分静态和动态
            palette=["#e74c3c", "#3498db"],  # 静态红，动态蓝
            sharey=False,
            sharex=False,
            height=3.5,
            aspect=1.1,
            scatter_kws={
                "alpha": 0.5,
                "s": 30
            },
            line_kws={"linewidth": 2})
        g1.fig.suptitle("Target Fidelity vs Performance (Static vs Dynamic)",
                        y=1.05,
                        fontweight="bold")
        g1.savefig(os.path.join(configs["exp_dir"],
                                "plot_Fidelity_vs_SISNR_compare.png"),
                   dpi=300,
                   bbox_inches='tight')

        g2 = sns.lmplot(data=df_normal,
                        x="Confusion_Margin",
                        y="Output_SISNR",
                        col="Mixture_ID",
                        col_wrap=5,
                        hue="Method",
                        palette=["#e74c3c", "#3498db"],
                        sharey=False,
                        sharex=False,
                        height=3.5,
                        aspect=1.1,
                        scatter_kws={
                            "alpha": 0.5,
                            "s": 30
                        },
                        line_kws={"linewidth": 2})
        g2.fig.suptitle("Target Margin vs Performance (Static vs Dynamic)",
                        y=1.05,
                        fontweight="bold")
        g2.savefig(os.path.join(configs["exp_dir"],
                                "plot_Margin_vs_SISNR_compare.png"),
                   dpi=300,
                   bbox_inches='tight')

    logger.info("🎨 绘图完成！图片已保存至 exp_dir 目录。")


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
