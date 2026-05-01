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
# 🚀 引擎 1：多卡并行特征提取 (支持保存伴随音频)
# ========================================================
def infer(config="confs/conf.yaml", num_shards=1, shard_id=0, **kwargs):
    start = time.time()
    configs = parse_config_or_kwargs(config, **kwargs)

    rank = 0
    set_seed(configs["seed"] + rank)
    gpu = configs["gpus"]
    device = (torch.device("cuda:{}".format(gpu))
              if gpu >= 0 else torch.device("cpu"))

    sign_save_wav = configs.get("save_wav", False)
    fs_raw = configs.get("fs", 16000)
    if isinstance(fs_raw, str):
        fs = int(fs_raw.replace("k", "")) * 1000
    else:
        fs = int(fs_raw)

    if 'spk_model_init' in configs['model_args']['tse_model']:
        configs['model_args']['tse_model']['spk_model_init'] = False

    model = get_model(configs["model"]["tse_model"])(
        configs["model_args"]["tse_model"])
    model_path = os.path.join(configs["checkpoint"])
    load_pretrained_model(model, model_path)

    logger = get_logger(configs["exp_dir"], f"infer_exp1_shard_{shard_id}.log")
    logger.info(
        f"🚀 [Shard {shard_id}/{num_shards}] 启动！Load checkpoint from {model_path}"
    )
    logger.info(f"🎧 是否保存伴随音频: {sign_save_wav} (采样率: {fs})")

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
    with open(configs["test_data"], "r", encoding="utf-8") as f:
        total_mix = sum(1 for _ in f)
    logger.info(f"🔥 开始全量刷库 (Total: {total_mix} Mixtures).")

    # ========================================================
    # 🔄 Resume 逻辑：跳过已处理的 mix 文件夹
    # ========================================================
    save_feat_dir = os.path.join(configs["exp_dir"], "exp1_tsne_features")
    os.makedirs(save_feat_dir, exist_ok=True)
    existing_mixes = set()
    if os.path.exists(save_feat_dir):
        for d in glob.glob(os.path.join(save_feat_dir, "mix_*")):
            basename = os.path.basename(d)
            try:
                # 文件夹名格式: mix_0992_xxx -> 提取 992
                idx = int(basename.split("_")[1])
                existing_mixes.add(idx)
            except (ValueError, IndexError):
                pass
    if existing_mixes:
        logger.info(f"📦 检测到 {len(existing_mixes)} 个已处理的 Mixture，将跳过这些样本...")
    # ========================================================

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            if i in existing_mixes:
                continue
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
            interf_1 = interf_spk1.detach().cpu().numpy().flatten()
            mix_1 = mix_spk1.detach().cpu().numpy().flatten()

            available_utts = spk_audio_dict.get(target_spk_id, [])
            if len(available_utts) == 0: continue

            # 提前建立文件夹，准备录音
            mix_dir = os.path.join(save_feat_dir, f"mix_{i+1:04d}_{key}")
            os.makedirs(mix_dir, exist_ok=True)

            audio_dir = os.path.join(mix_dir, "audios")
            if sign_save_wav:
                os.makedirs(audio_dir, exist_ok=True)
                # 🎙️ 保存“金标准”对比组
                end_s = min(len(ref_1), len(mix_1), len(interf_1))
                soundfile.write(os.path.join(mix_dir, "mix.wav"),
                                mix_1[:end_s], fs)
                soundfile.write(os.path.join(mix_dir, "tgt_true.wav"),
                                ref_1[:end_s], fs)
                soundfile.write(os.path.join(mix_dir, "int_true.wav"),
                                interf_1[:end_s], fs)

            oracle_tgt_prior, oracle_tgt_pmap, oracle_tgt_post = extract_three_stage_features(
                target_spk1.view(1, -1), mix_spk1)

            out_oracle = model(
                mix_spk1,
                [target_spk1.view(1, -1)])[0].detach().cpu().numpy().flatten()
            if np.max(np.abs(out_oracle)) > 0:
                out_oracle = out_oracle / np.max(np.abs(out_oracle)) * 0.9
            end_s = min(len(out_oracle), len(ref_1), len(mix_1), len(interf_1))
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

                end_d = min(len(out_np_dyn), len(ref_1), len(mix_1),
                            len(interf_1))
                dyn_snri, _ = cal_SISNRi(out_np_dyn[:end_d], ref_1[:end_d],
                                         mix_1[:end_d])
                dyn_snri_int, _ = cal_SISNRi(out_np_dyn[:end_d],
                                             interf_1[:end_d], mix_1[:end_d])

                # 🎙️ 极其硬核的音频保存逻辑：文件名挂载成绩和索引！
                if sign_save_wav:
                    soundfile.write(
                        os.path.join(audio_dir,
                                     f"est_{idx:02d}_snr_{dyn_snri:+.1f}.wav"),
                        out_np_dyn[:end_d], fs)
                    soundfile.write(
                        os.path.join(audio_dir, f"cue_{idx:02d}.wav"),
                        crop_wav, fs)

                mix_prior_list.append(prior_v)
                mix_pmap_list.append(pmap_v)
                mix_post_list.append(post_v)
                mix_labels_list.append({
                    "Sample_Idx": idx,
                    "Target_Spk": target_spk_id,
                    "Dynamic_SISNRi": dyn_snri,
                    "Interferer_SISNRi": dyn_snri_int,
                    "Delta_SISNRi": oracle_snri - dyn_snri
                })

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
                     tgt_post=oracle_tgt_post)

            if (i + 1) % 50 == 0:
                logger.info(f"   [Shard {shard_id}] 处理进度: {i+1}/{total_mix}")

    torch.cuda.empty_cache()
    import sys
    sys.exit(0)


# ========================================================
# 🎨 引擎 2：全局 t-SNE 绘图 (剔除干扰孤立点版)
# ========================================================
def plot(config="confs/conf.yaml", **kwargs):
    configs = parse_config_or_kwargs(config, **kwargs)
    feat_base_dir = os.path.join(configs["exp_dir"], "exp1_tsne_features")
    output_dir = os.path.join(configs["exp_dir"],
                              "exp1_global_highlight_plots")
    os.makedirs(output_dir, exist_ok=True)

    mix_folders = sorted(glob.glob(os.path.join(feat_base_dir, "mix_*")))
    print(f"🔍 汇总完毕！共找到 {len(mix_folders)} 个 Mixture，开始自动化挑选黄金样本...")

    candidates_full = []  # 四类全有
    candidates_partial = []  # 只有 Success + Wrong Person
    global_stats = {
        "1. Success": 0,
        "2. Not Separated": 0,
        "3. Wrong Person": 0,
        "4. Distortion": 0
    }

    for d in mix_folders:
        label_file = os.path.join(d, "labels.csv")
        if not os.path.exists(label_file): continue
        df = pd.read_csv(label_file)
        if len(df) < 10 or 'Interferer_SISNRi' not in df.columns: continue

        def get_state(dyn_snri, int_snri):
            if dyn_snri > 0 and int_snri < 0: return "1. Success"
            if dyn_snri > 0 and int_snri > 0: return "2. Not Separated"
            if dyn_snri < 0 and int_snri > 0: return "3. Wrong Person"
            return "4. Distortion"

        df['State'] = df.apply(
            lambda r: get_state(r['Dynamic_SISNRi'], r['Interferer_SISNRi']),
            axis=1)
        state_counts = df['State'].value_counts()
        for s, cnt in state_counts.items():
            global_stats[s] += cnt
        target_spk = str(df['Target_Spk'].iloc[0])
        entry = {
            'mix_dir': d,
            'mix_name': os.path.basename(d),
            'target_spk': target_spk,
            'state_counts': dict(state_counts)
        }

        all_states = {
            "1. Success", "2. Not Separated", "3. Wrong Person",
            "4. Distortion"
        }
        if all_states.issubset(set(state_counts.index)):
            candidates_full.append(entry)
        elif {"1. Success",
              "3. Wrong Person"}.issubset(set(state_counts.index)):
            candidates_partial.append(entry)

    total = sum(global_stats.values())
    print(f"\n🌍 全局样本统计 (共 {total} 个样本):")
    print(
        f"   1. Success:       {global_stats['1. Success']:>6} ({global_stats['1. Success']/total*100:>5.1f}%)"
    )
    print(
        f"   2. Not Separated: {global_stats['2. Not Separated']:>6} ({global_stats['2. Not Separated']/total*100:>5.1f}%)"
    )
    print(
        f"   3. Wrong Person:  {global_stats['3. Wrong Person']:>6} ({global_stats['3. Wrong Person']/total*100:>5.1f}%)"
    )
    print(
        f"   4. Distortion:    {global_stats['4. Distortion']:>6} ({global_stats['4. Distortion']/total*100:>5.1f}%)"
    )

    candidates_full.sort(key=lambda x: sum(x['state_counts'].values()),
                         reverse=True)
    candidates_partial.sort(key=lambda x: sum(x['state_counts'].values()),
                            reverse=True)

    print(f"\n📊 四类全的 Mixture ({len(candidates_full)} 个):")
    print(
        f"{'Mix Name':<30} {'1.Success':>8} {'2.NotSep':>8} {'3.WrongP':>8} {'4.Distort':>8}"
    )
    print("-" * 60)
    for c in candidates_full[:20]:
        sc = c['state_counts']
        print(
            f"{c['mix_name']:<30} {sc.get('1. Success',0):>8} {sc.get('2. Not Separated',0):>8} {sc.get('3. Wrong Person',0):>8} {sc.get('4. Distortion',0):>8}"
        )

    print(
        f"\n📊 Success + Wrong Person 的 Mixture ({len(candidates_partial)} 个):")
    print(
        f"{'Mix Name':<30} {'1.Success':>8} {'2.NotSep':>8} {'3.WrongP':>8} {'4.Distort':>8}"
    )
    print("-" * 60)
    for c in candidates_partial[:20]:
        sc = c['state_counts']
        print(
            f"{c['mix_name']:<30} {sc.get('1. Success',0):>8} {sc.get('2. Not Separated',0):>8} {sc.get('3. Wrong Person',0):>8} {sc.get('4. Distortion',0):>8}"
        )

    candidates = candidates_full + candidates_partial

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
        print("❌ 没有找到任何同时包含 Success 和 Wrong Person 的样本！")
        return

    print(f"🎯 成功锁定 {len(selected_mixes)} 个黄金 Mixture！开始构建高维宇宙...")

    all_prior, all_pmap, all_post = [], [], []
    metadata = []

    def get_state(dyn_snri, int_snri):
        if dyn_snri > 0 and int_snri < 0: return "1. Success"
        if dyn_snri > 0 and int_snri > 0: return "2. Not Separated"
        if dyn_snri < 0 and int_snri > 0: return "3. Wrong Person"
        return "4. Distortion"

    palette = {
        "1. Success": "#2ca02c",
        "2. Not Separated": "#ff7f0e",
        "3. Wrong Person": "#d62728",
        "4. Distortion": "#8e44ad",
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
                "Mix_Name":
                mix_name,
                "Type":
                "Sample",
                "State":
                get_state(row["Dynamic_SISNRi"], row["Interferer_SISNRi"]),
            })

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
            ax.scatter(df_bg[cx],
                       df_bg[cy],
                       c='#d3d3d3',
                       s=40,
                       alpha=0.4,
                       edgecolors='none',
                       zorder=1)
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
            ax.scatter(tgt_anchor[cx],
                       tgt_anchor[cy],
                       marker='*',
                       s=700,
                       color='#2ecc71',
                       edgecolor='black',
                       linewidth=1.5,
                       label='Oracle Target' if i == 0 else "",
                       zorder=10)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title(title, fontweight="bold", fontsize=14)

        axes[0].legend(loc='upper right',
                       bbox_to_anchor=(1.0, -0.05),
                       ncol=4,
                       fontsize=9)
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


# ========================================================
# 🚀 引擎 3：终极猜想验证 (SI-SNRi 象限解耦图)
# ========================================================
def verify(config="confs/conf.yaml", **kwargs):
    configs = parse_config_or_kwargs(config, **kwargs)
    feat_base_dir = os.path.join(configs["exp_dir"], "exp1_tsne_features")
    output_dir = os.path.join(configs["exp_dir"],
                              "exp1_hypothesis_verification")
    os.makedirs(output_dir, exist_ok=True)

    mix_folders = sorted(glob.glob(os.path.join(feat_base_dir, "mix_*")))
    print(f"🔍 汇总开始！准备验证【混淆 vs 畸变】猜想...")

    all_data = []

    def get_state(dyn_snri, int_snri):
        if dyn_snri > 0 and int_snri < 0: return "1. Success"
        if dyn_snri > 0 and int_snri > 0: return "2. Not Separated"
        if dyn_snri < 0 and int_snri > 0: return "3. Wrong Person"
        return "4. Distortion"

    for d in mix_folders:
        label_file = os.path.join(d, "labels.csv")
        if not os.path.exists(label_file): continue
        df = pd.read_csv(label_file)
        if len(df) < 10 or 'Interferer_SISNRi' not in df.columns: continue

        df['State'] = df.apply(
            lambda r: get_state(r['Dynamic_SISNRi'], r['Interferer_SISNRi']),
            axis=1)
        df['Mix_Name'] = os.path.basename(d)
        all_data.append(df)

    if len(all_data) == 0:
        print("❌ 未找到包含 Interferer_SISNRi 的数据，请先重新运行 infer！")
        return

    df_master = pd.concat(all_data, ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    palette = {
        "1. Success": "#2ca02c",
        "2. Not Separated": "#ff7f0e",
        "3. Wrong Person": "#d62728",
        "4. Distortion": "#8e44ad"
    }

    sns.scatterplot(data=df_master,
                    x='Interferer_SISNRi',
                    y='Dynamic_SISNRi',
                    hue='State',
                    palette=palette,
                    s=80,
                    alpha=0.7,
                    edgecolor='white',
                    ax=ax)

    ax.axhline(0, color='gray', linestyle='--', linewidth=1.5, zorder=0)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1.5, zorder=0)

    ax.text(-10,
            15,
            "Quadrant I: Clean Extraction\n(Successful)",
            color='#2ca02c',
            fontweight='bold',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax.text(10,
            -10,
            "Quadrant IV: Target Confusion\n(Extracted Interferer)",
            color='#c0392b',
            fontweight='bold',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax.text(-15,
            -10,
            "Quadrant III: Signal Distortion\n(Garbage/Silence)",
            color='#8e44ad',
            fontweight='bold',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax.set_xlabel(
        "Interferer SI-SNRi (dB) $\\rightarrow$ High means extracted wrong person",
        fontweight="bold",
        fontsize=13)
    ax.set_ylabel("Target SI-SNRi (dB) $\\rightarrow$ High means success",
                  fontweight="bold",
                  fontsize=13)
    ax.set_title(
        "Hypothesis Verification: Decoupling Confusion and Distortion",
        fontweight="bold",
        fontsize=16,
        pad=15)

    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "plot_SIR_SDR_Decoupling.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"🎉 猜想验证图谱生成完毕！请查看: {save_path}")


# ========================================================
# 🔧 辅助函数：提取模型输入
# ========================================================
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


# ========================================================
# 🎨 引擎 4：两类 t-SNE 降维可视化
#   Cat1 - 提错人（Dynamic_SISNRi < 0，干扰人被提取）
#   Cat2 - 音频损伤（Dynamic_SISNRi > 0 且 Delta_SISNRi > 5dB，目标正确提取但有明显损伤）
# ========================================================
def plot_two_category(config="confs/conf.yaml", **kwargs):
    """
    两类 t-SNE 降维可视化（全局一次性 t-SNE，与 plot 函数策略一致）：

    1. Category 1 - 提错人（Interferer Extracted）：
       Dynamic_SISNRi < 0 的样本标记为 Failure (Interferer) 红色，
       Dynamic_SISNRi >= 0 的样本按 Delta 分档（Safe/Marginal/Failure Distortion）

    2. Category 2 - 音频损伤（Audio Distortion）：
       Dynamic_SISNRi > 0 且 Delta_SISNRi > 5dB 的样本，
       按 Delta_SISNRi 分档（Safe/Marginal/Failure）
    """
    configs = parse_config_or_kwargs(config, **kwargs)
    feat_base_dir = os.path.join(configs["exp_dir"], "exp1_tsne_features")

    output_dir_cat1 = os.path.join(configs["exp_dir"],
                                   "exp1_tsne_cat1_interferer_extracted")
    output_dir_cat2 = os.path.join(configs["exp_dir"],
                                   "exp1_tsne_cat2_audio_distortion")
    os.makedirs(output_dir_cat1, exist_ok=True)
    os.makedirs(output_dir_cat2, exist_ok=True)

    mix_folders = sorted(glob.glob(os.path.join(feat_base_dir, "mix_*")))
    print(f"🔍 共找到 {len(mix_folders)} 个 Mixture，开始分类筛选...")

    # --------------------------------------------------
    # 扫描所有 mixture，按类别收集候选
    # --------------------------------------------------
    cat1_candidates = []
    cat2_candidates = []

    for d in mix_folders:
        csv_path = os.path.join(d, "labels.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if len(df) < 10 or 'Interferer_SISNRi' not in df.columns:
            continue

        mix_name = os.path.basename(d)
        target_spk = str(df['Target_Spk'].iloc[0])

        fail_df = df[df['Dynamic_SISNRi'] < 0]
        if len(fail_df) >= 5:
            n_success = len(df[df['Dynamic_SISNRi'] >= 0])
            balance_score = min(n_success, len(fail_df))
            cat1_candidates.append({
                'mix_dir':
                d,
                'mix_name':
                mix_name,
                'target_spk':
                target_spk,
                'n_fail':
                len(fail_df),
                'n_success':
                n_success,
                'balance_score':
                balance_score,
                'avg_fail_dyn':
                fail_df['Dynamic_SISNRi'].mean(),
                'df':
                df,
            })

        dist_df = df[(df['Dynamic_SISNRi'] > 0) & (df['Delta_SISNRi'] > 5)]
        if len(dist_df) >= 5:
            cat2_candidates.append({
                'mix_dir': d,
                'mix_name': mix_name,
                'target_spk': target_spk,
                'n_dist': len(dist_df),
                'avg_delta': dist_df['Delta_SISNRi'].mean(),
                'df': df,
            })

    print(
        f"\n📊 Category 1 (提错人, Dynamic_SISNRi < 0): {len(cat1_candidates)} 个候选"
    )
    print(
        f"📊 Category 2 (音频损伤, Dynamic_SISNRi > 0 & Delta > 5dB): {len(cat2_candidates)} 个候选"
    )

    # --------------------------------------------------
    # Speaker 多样性优先选取
    # --------------------------------------------------
    def select_mixes(candidates, target_count=10, key='size'):
        selected = []
        seen_spks = set()
        candidates_sorted = sorted(candidates,
                                   key=lambda x: x[key],
                                   reverse=True)
        for cand in candidates_sorted:
            if cand['target_spk'] not in seen_spks:
                selected.append(cand)
                seen_spks.add(cand['target_spk'])
            if len(selected) == target_count:
                break
        if len(selected) < target_count:
            for cand in candidates_sorted:
                if cand not in selected:
                    selected.append(cand)
                if len(selected) == target_count:
                    break
        return selected

    selected_cat1 = select_mixes(cat1_candidates, 10, key='balance_score')
    selected_cat2 = select_mixes(cat2_candidates, 10, key='n_dist')

    print(f"\n🎯 Category 1 选取 {len(selected_cat1)} 个 Mixture（平衡分数优先）：")
    for c in selected_cat1:
        print(
            f"   {c['mix_name']}: fail={c['n_fail']}, success={c['n_success']}, balance={c['balance_score']}, avg_fail_sisnr={c['avg_fail_dyn']:.2f}"
        )

    print(f"\n🎯 Category 2 选取 {len(selected_cat2)} 个 Mixture：")
    for c in selected_cat2:
        print(
            f"   {c['mix_name']}: {c['n_dist']} 个损伤样本, avg Delta={c['avg_delta']:.2f}"
        )

    # --------------------------------------------------
    # 标签映射函数
    # --------------------------------------------------
    def get_state_cat1(dyn_sisnr, delta):
        if dyn_sisnr < 0:
            return "Failure (Interferer)"
        elif delta < 2.0:
            return "Safe"
        elif delta <= 5.0:
            return "Marginal"
        else:
            return "Failure (Distortion)"

    def get_state_cat2(delta):
        if delta < 2.0:
            return "Safe (< 2dB)"
        elif delta <= 5.0:
            return "Marginal (2~5dB)"
        else:
            return "Failure (> 5dB)"

    palette_cat1 = {
        "Safe": "#2ca02c",
        "Marginal": "#ff7f0e",
        "Failure (Distortion)": "#e74c3c",
        "Failure (Interferer)": "#8b0000",
    }
    palette_cat2 = {
        "Safe (< 2dB)": "#2ca02c",
        "Marginal (2~5dB)": "#ff7f0e",
        "Failure (> 5dB)": "#d62728",
    }

    # --------------------------------------------------
    # 通用的"一次性全局 t-SNE + 绘图"函数
    # --------------------------------------------------
    def run_category(selected_mixes, output_dir, palette, get_state_fn,
                     title_prefix):
        """
        策略与 plot() 完全一致：
        1. 将所有 selected_mixes 的样本堆叠成一个大矩阵
        2. 全局只做一次 t-SNE（每个 space 各一次）
        3. 用 Mix_Name 列过滤前景/背景后逐个绘图
        """
        all_prior, all_pmap, all_post = [], [], []
        metadata = []

        for cand in selected_mixes:
            mix_dir = cand['mix_dir']
            mix_name = cand['mix_name']
            df = cand['df']
            anchors = np.load(os.path.join(mix_dir, "oracle_anchors.npz"))

            prior_feats = np.load(os.path.join(mix_dir, "prior_features.npy"))
            pmap_feats = np.load(os.path.join(mix_dir, "pmap_features.npy"))
            post_feats = np.load(
                os.path.join(mix_dir, "post_concat_features.npy"))

            all_prior.append(prior_feats)
            all_pmap.append(pmap_feats)
            all_post.append(post_feats)
            for _, row in df.iterrows():
                metadata.append({
                    "Mix_Name":
                    mix_name,
                    "Type":
                    "Sample",
                    "State":
                    get_state_fn(row["Dynamic_SISNRi"], row["Delta_SISNRi"]),
                })
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

        print(f"⏳ 全局 t-SNE 降维中（{len(df_meta)} 个点）...")
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

        print("✅ 开始依次渲染图像...")
        for cand in selected_mixes:
            target_mix = cand['mix_name']
            fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

            df_bg = df_meta[df_meta['Mix_Name'] != target_mix]
            df_fg = df_meta[(df_meta['Mix_Name'] == target_mix)
                            & (df_meta['Type'] == 'Sample')]
            tgt_anchor = df_meta[(df_meta['Mix_Name'] == target_mix) &
                                 (df_meta['Type'] == 'Anchor_Target')].iloc[0]

            spaces = [
                ("1. Prior Space (Speaker Identity)", 'Prior_X', 'Prior_Y'),
                ("2. Pmap Space (Energy Mask)", 'Pmap_X', 'Pmap_Y'),
                ("3. Post-Concat Space (Decision)", 'Post_X', 'Post_Y'),
            ]

            for i, (title, cx, cy) in enumerate(spaces):
                ax = axes[i]
                ax.scatter(df_bg[cx],
                           df_bg[cy],
                           c='#d3d3d3',
                           s=40,
                           alpha=0.4,
                           edgecolors='none',
                           zorder=1)
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
                ax.scatter(tgt_anchor[cx],
                           tgt_anchor[cy],
                           marker='*',
                           s=700,
                           color='#2ecc71',
                           edgecolor='black',
                           linewidth=1.5,
                           label='Oracle Target' if i == 0 else "",
                           zorder=10)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_title(title, fontweight="bold", fontsize=14)

            axes[0].legend(loc='upper right',
                           bbox_to_anchor=(1.0, -0.05),
                           ncol=2,
                           fontsize=10)
            plt.suptitle(f"{title_prefix}{target_mix}",
                         fontweight="bold",
                         fontsize=15,
                         y=1.02)
            plt.tight_layout()
            out_name = f"cat1_{target_mix}.png" if "cat1" in output_dir else f"cat2_{target_mix}.png"
            fig.savefig(os.path.join(output_dir, out_name),
                        dpi=300,
                        bbox_inches='tight')
            plt.close(fig)
            print(f"   已保存 {out_name}")

    # --------------------------------------------------
    # 执行 Cat1 和 Cat2
    # --------------------------------------------------
    print(f"\n⏳ 开始绘制 Category 1（提错人）...")
    run_category(selected_cat1, output_dir_cat1, palette_cat1,
                 lambda dyn, delta: get_state_cat1(dyn, delta),
                 "[Cat1] Interferer Extracted: ")
    print(f"✅ Category 1 绘图完毕，保存至: {output_dir_cat1}")

    print(f"\n⏳ 开始绘制 Category 2（音频损伤）...")
    run_category(selected_cat2, output_dir_cat2, palette_cat2,
                 lambda dyn, delta: get_state_cat2(delta),
                 "[Cat2] Audio Distortion: ")
    print(f"✅ Category 2 绘图完毕，保存至: {output_dir_cat2}")

    print(f"\n🎉 全部完成！")


if __name__ == "__main__":
    fire.Fire()
