from __future__ import print_function

import os
import time
import types
import json
import random
import glob

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
import pandas as pd
from wesep.models import get_model
from wesep.utils.checkpoint import load_pretrained_model
from wesep.utils.score import cal_SISNRi
from wesep.utils.utils import (
    get_logger,
    parse_config_or_kwargs,
    set_seed,
)
from wesep.utils.file_utils import load_yaml

from wesep.modules.speaker.encoder import Fbank_kaldi, SpeakerEncoder

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import warnings

warnings.filterwarnings('ignore')

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


# ========================================================
# 🚀 引擎 1：多卡并行特征提取
# ========================================================
def infer(config="confs/conf.yaml", num_shards=1, shard_id=0, **kwargs):
    """
    num_shards: 总共切分成几份 (通常等于你的 GPU 数量)
    shard_id: 当前脚本处理第几份 (0 到 num_shards-1)
    """
    start = time.time()
    configs = parse_config_or_kwargs(config, **kwargs)

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

    # 日志名加上 shard_id 防止多卡日志写入冲突
    logger = get_logger(configs["exp_dir"], f"infer_exp1_shard_{shard_id}.log")
    logger.info(
        f"🚀 [Shard {shard_id}/{num_shards}] 启动！Load checkpoint from {model_path}"
    )

    save_feat_dir = os.path.join(configs["exp_dir"], "exp1_tsne_features")
    os.makedirs(save_feat_dir, exist_ok=True)

    model = model.to(device)
    model.eval()

    try:
        spk_conf_all = configs["model_args"]["tse_model"]["speaker"]
        spk_model_conf = spk_conf_all.get("speaker_model", None)
        fbank_extractor = Fbank_kaldi(**spk_model_conf['fbank']).to(device)
        spk_extractor = SpeakerEncoder(
            spk_model_conf['speaker_encoder']).to(device)
        fbank_extractor.eval()
        spk_extractor.eval()
    except Exception as e:
        logger.error(f"初始化声纹探针失败: {e}")
        return

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

    hooked_features = {}
    target_module = model.module if hasattr(model, 'module') else model

    if hasattr(target_module, 'spk_configs'
               ) and target_module.spk_configs['features']['usef']['enabled']:
        original_usef_post = target_module.spk_ft.usef.post

        def hooked_usef_post(self, mix_repr, feat_repr):
            hooked_features['pmap'] = feat_repr.detach().clone()
            out_concat = original_usef_post(mix_repr, feat_repr)
            hooked_features['post_concat'] = out_concat.detach().clone()
            return out_concat

        target_module.spk_ft.usef.post = types.MethodType(
            hooked_usef_post, target_module.spk_ft.usef)

    try:
        with open(
                "/home/yxy05/code/research_tse/examples/audio/librimix/data/clean/test/cues/audio.json",
                "r",
                encoding="utf-8") as f:
            spk_audio_dict = json.load(f)
    except Exception as e:
        return

    def extract_three_stage_features(audio_tensor, mixture_tensor):
        fb = fbank_extractor(audio_tensor)
        emb = spk_extractor(fb)
        if isinstance(emb, (tuple, list)): emb = emb[-1]
        prior_v = emb.view(-1).detach().cpu().numpy()

        _ = model(mixture_tensor, [audio_tensor])

        pmap_t = hooked_features['pmap'].clone()
        if pmap_t.dim() > 2: pmap_t = pmap_t.mean(dim=-1)
        pmap_v = pmap_t.view(-1).cpu().numpy()

        post_t = hooked_features['post_concat'].clone()
        if post_t.dim() > 2: post_t = post_t.mean(dim=-1)
        post_v = post_t.view(-1).cpu().numpy()

        return prior_v, pmap_v, post_v

    num_samples_per_mix = 50
    # ✅ 修复方案：直接读取 test_data 列表的行数作为总数
    with open(configs["test_data"], "r", encoding="utf-8") as f:
        total_mix = sum(1 for _ in f)
    logger.info(
        f"🔥 开始全量刷库 (Total: {total_mix} Mixtures). 当前分片只处理 i % {num_shards} == {shard_id} 的样本。"
    )

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            # ✂️ 核心分片逻辑：只处理属于本 GPU 的任务！
            if i % num_shards != shard_id:
                continue

            mix, cues, target = extract_model_inputs(batch, device)
            spk = batch["spk"]
            key = batch["key"][0]
            target_spk_id = str(spk[0])

            mix_spk1 = mix[0:1, :, :]
            if target.shape[0] > 1:
                target_spk1 = target[
                    0:1, :, :] if target.dim() == 3 else target[0:1, 0:1, :]
                interf_spk1 = target[
                    1:2, :, :] if target.dim() == 3 else target[1:2, 0:1, :]
            else:
                target_spk1 = target[
                    0:1, :, :] if target.dim() == 3 else target[0:1, 0:1, :]
                interf_spk1 = mix_spk1 - target_spk1

            ref_1 = target_spk1.detach().cpu().numpy().flatten()
            mix_1 = mix_spk1.detach().cpu().numpy().flatten()

            available_utts = spk_audio_dict.get(target_spk_id, [])
            if len(available_utts) == 0: continue

            oracle_tgt_prior, oracle_tgt_pmap, oracle_tgt_post = extract_three_stage_features(
                target_spk1.view(1, -1), mix_spk1)
            oracle_int_prior, oracle_int_pmap, oracle_int_post = extract_three_stage_features(
                interf_spk1.view(1, -1), mix_spk1)

            out_oracle = model(
                mix_spk1,
                [target_spk1.view(1, -1)])[0].detach().cpu().numpy().flatten()
            if np.max(np.abs(out_oracle)) > 0:
                out_oracle = out_oracle / np.max(np.abs(out_oracle)) * 0.9
            end_s = min(len(out_oracle), len(ref_1), len(mix_1))
            oracle_snri, _ = cal_SISNRi(out_oracle[:end_s], ref_1[:end_s],
                                        mix_1[:end_s])

            mix_prior_list, mix_pmap_list, mix_post_list, mix_labels_list = [], [], [], []

            for idx in range(num_samples_per_mix):
                item = random.choice(available_utts)
                try:
                    enroll_wav, sr = soundfile.read(item["path"])
                    total_len = len(enroll_wav)
                    min_len = min(sr * 1, total_len)
                    if total_len > min_len:
                        crop_len = random.randint(min_len, total_len)
                        start_idx = random.randint(0, total_len - crop_len)
                        crop_wav = enroll_wav[start_idx:start_idx + crop_len]
                    else:
                        crop_wav = enroll_wav
                    current_cue = torch.tensor(
                        crop_wav, dtype=torch.float32).to(device).view(1, -1)
                except:
                    continue

                prior_v, pmap_v, post_v = extract_three_stage_features(
                    current_cue, mix_spk1)

                outputs_dynamic = model(mix_spk1, [current_cue])
                out_np_dyn = outputs_dynamic[0].detach().cpu().numpy().flatten(
                )
                if np.max(np.abs(out_np_dyn)) > 0:
                    out_np_dyn = out_np_dyn / np.max(np.abs(out_np_dyn)) * 0.9
                end_d = min(len(out_np_dyn), len(ref_1), len(mix_1))
                dyn_snri, _ = cal_SISNRi(out_np_dyn[:end_d], ref_1[:end_d],
                                         mix_1[:end_d])

                mix_prior_list.append(prior_v)
                mix_pmap_list.append(pmap_v)
                mix_post_list.append(post_v)
                mix_labels_list.append({
                    "Sample_Idx": idx,
                    "Target_Spk": target_spk_id,
                    "Dynamic_SISNRi": dyn_snri,
                    "Delta_SISNRi": oracle_snri - dyn_snri
                })

            mix_dir = os.path.join(save_feat_dir, f"mix_{i+1:04d}_{key}")
            os.makedirs(mix_dir, exist_ok=True)

            np.save(os.path.join(mix_dir, "prior_features.npy"),
                    np.array(mix_prior_list))
            np.save(os.path.join(mix_dir, "pmap_features.npy"),
                    np.array(mix_pmap_list))
            np.save(os.path.join(mix_dir, "post_concat_features.npy"),
                    np.array(mix_post_list))
            pd.DataFrame(mix_labels_list).to_csv(os.path.join(
                mix_dir, "labels.csv"),
                                                 index=False)
            np.savez(os.path.join(mix_dir, "oracle_anchors.npz"),
                     tgt_prior=oracle_tgt_prior,
                     tgt_pmap=oracle_tgt_pmap,
                     tgt_post=oracle_tgt_post,
                     int_prior=oracle_int_prior,
                     int_pmap=oracle_int_pmap,
                     int_post=oracle_int_post)

            logger.info(f"   [Shard {shard_id}] 完毕: {key}")

    end = time.time()
    logger.info(f"✅ [Shard {shard_id}] 任务完成！耗时: {end - start:.1f}s.")


# ========================================================
# 🎨 引擎 2：全自动挑选与绘图 (必须等所有提取跑完后单独调用)
# ========================================================
# ========================================================
# 🎨 引擎 2：全自动挑选与绘图 (移除了 Interferer 孤立点)
# ========================================================
def plot(config="confs/conf.yaml", **kwargs):
    configs = parse_config_or_kwargs(config, **kwargs)
    feat_base_dir = os.path.join(configs["exp_dir"], "exp1_tsne_features")
    output_dir = os.path.join(configs["exp_dir"],
                              "exp1_global_highlight_plots")
    os.makedirs(output_dir, exist_ok=True)

    mix_folders = sorted(glob.glob(os.path.join(feat_base_dir, "mix_*")))
    print(f"🔍 汇总完毕！共找到 {len(mix_folders)} 个 Mixture，开始自动化挑选黄金样本...")

    candidates = []
    for d in mix_folders:
        label_file = os.path.join(d, "labels.csv")
        if not os.path.exists(label_file): continue
        df = pd.read_csv(label_file)
        if len(df) < 10: continue

        n_fail = len(df[df['Delta_SISNRi'] > 5.0])
        n_safe = len(df[df['Delta_SISNRi'] < 2.0])
        target_spk = str(df['Target_Spk'].iloc[0])

        balance_score = min(n_fail, len(df) - n_fail)
        if n_fail > 5 and n_safe > 5:
            candidates.append({
                'mix_dir': d,
                'mix_name': os.path.basename(d),
                'target_spk': target_spk,
                'n_fail': n_fail,
                'n_safe': n_safe,
                'balance_score': balance_score
            })

    candidates.sort(key=lambda x: x['balance_score'], reverse=True)

    selected_mixes = []
    seen_spks = set()
    for cand in candidates:
        if cand['target_spk'] not in seen_spks:
            selected_mixes.append(cand)
            seen_spks.add(cand['target_spk'])
        if len(selected_mixes) == 10: break

    if len(selected_mixes) < 10:
        for cand in candidates:
            if cand not in selected_mixes: selected_mixes.append(cand)
            if len(selected_mixes) == 10: break

    if len(selected_mixes) == 0:
        print("❌ 没有找到任何既有成功又有失败的对抗样本！")
        return

    print(f"🎯 成功锁定 {len(selected_mixes)} 个黄金 Mixture！开始构建高维宇宙...")

    all_prior, all_pmap, all_post = [], [], []
    metadata = []

    def get_state(delta):
        if delta < 2.0: return "1. Safe (< 2dB)"
        elif delta <= 5.0: return "2. Marginal (2~5dB)"
        else: return "3. Failure (> 5dB)"

    palette = {
        "1. Safe (< 2dB)": "#2ca02c",
        "2. Marginal (2~5dB)": "#ff7f0e",
        "3. Failure (> 5dB)": "#d62728"
    }

    for cand in selected_mixes:
        mix_dir = cand['mix_dir']
        mix_name = cand['mix_name']
        prior_feats = np.load(os.path.join(mix_dir, "prior_features.npy"))
        pmap_feats = np.load(os.path.join(mix_dir, "pmap_features.npy"))
        post_feats = np.load(os.path.join(mix_dir, "post_concat_features.npy"))
        df_labels = pd.read_csv(os.path.join(mix_dir, "labels.csv"))
        anchors = np.load(os.path.join(mix_dir, "oracle_anchors.npz"))

        all_prior.append(prior_feats)
        all_pmap.append(pmap_feats)
        all_post.append(post_feats)
        for _, row in df_labels.iterrows():
            metadata.append({
                "Mix_Name": mix_name,
                "Type": "Sample",
                "State": get_state(row["Delta_SISNRi"])
            })

        # ✂️ 改动点 1：只提取和保存 Target 锚点，彻底抛弃 Interferer 锚点
        all_prior.append(anchors['tgt_prior'].reshape(1, -1))
        all_pmap.append(anchors['tgt_pmap'].reshape(1, -1))
        all_post.append(anchors['tgt_post'].reshape(1, -1))
        metadata.append({
            "Mix_Name": mix_name,
            "Type": "Anchor_Target",
            "State": "Anchor"
        })

    X_prior = normalize(np.vstack(all_prior), norm='l2', axis=1)
    X_pmap = np.vstack(all_pmap)
    X_post = np.vstack(all_post)
    df_meta = pd.DataFrame(metadata)

    print(f"⏳ 正在进行全局 t-SNE 降维 (已剔除孤立的干扰者，空间将更加舒展)...")
    perp = min(30, len(df_meta) // 4)
    tsne_args = dict(n_components=2,
                     perplexity=perp,
                     random_state=42,
                     init='pca',
                     learning_rate='auto')

    df_meta['Prior_X'], df_meta['Prior_Y'] = TSNE(
        **tsne_args).fit_transform(X_prior).T
    df_meta['Pmap_X'], df_meta['Pmap_Y'] = TSNE(
        **tsne_args).fit_transform(X_pmap).T
    df_meta['Post_X'], df_meta['Post_Y'] = TSNE(
        **tsne_args).fit_transform(X_post).T

    print("✅ 开始依次渲染高亮星系...")
    for cand in selected_mixes:
        target_mix = cand['mix_name']
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        df_bg = df_meta[df_meta['Mix_Name'] != target_mix]
        df_fg = df_meta[(df_meta['Mix_Name'] == target_mix)
                        & (df_meta['Type'] == 'Sample')]
        tgt_anchor = df_meta[(df_meta['Mix_Name'] == target_mix)
                             & (df_meta['Type'] == 'Anchor_Target')].iloc[0]

        spaces = [("1. Prior Space (Identity)", 'Prior_X', 'Prior_Y'),
                  ("2. Pmap Space (Energy Mask)", 'Pmap_X', 'Pmap_Y'),
                  ("3. Post-Concat Space (Decision)", 'Post_X', 'Post_Y')]

        for i, (title, cx, cy) in enumerate(spaces):
            ax = axes[i]

            # 画全局灰色背景星尘
            ax.scatter(df_bg[cx],
                       df_bg[cy],
                       c='#d3d3d3',
                       s=40,
                       alpha=0.4,
                       edgecolors='none',
                       zorder=1)

            # 画当前 Mixture 的样本点
            sns.scatterplot(data=df_fg,
                            x=cx,
                            y=cy,
                            hue='State',
                            palette=palette,
                            s=150,
                            alpha=0.9,
                            edgecolor='white',
                            ax=ax,
                            legend=(i == 0),
                            zorder=5)

            # 画唯一的 Oracle Target 绿星锚点
            ax.scatter(tgt_anchor[cx],
                       tgt_anchor[cy],
                       marker='*',
                       s=700,
                       color='#2ecc71',
                       edgecolor='black',
                       linewidth=1.5,
                       label='Oracle Target' if i == 0 else "",
                       zorder=10)

            # ✂️ 改动点 2：移除了这里画红叉的代码

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title(title, fontweight="bold", fontsize=14)

        axes[0].legend(loc='upper right',
                       bbox_to_anchor=(1.0, -0.05),
                       ncol=2,
                       fontsize=11)
        plt.suptitle(f"Global Context Highlight: {target_mix}",
                     fontweight="bold",
                     fontsize=16,
                     y=1.02)
        plt.tight_layout()

        fig.savefig(os.path.join(output_dir, f"highlight_{target_mix}.png"),
                    dpi=300,
                    bbox_inches='tight')
        plt.close(fig)

    print(f"🎉 惊天神图全部生成完毕！图库已保存至: {output_dir}")


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
    # 使用 Fire 暴露所有函数，可以通过命令行指定执行 infer 还是 plot
    fire.Fire()
