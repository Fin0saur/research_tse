from __future__ import print_function

import glob
import json
import os
import types

import fire
import matplotlib
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F

from wesep.models import get_model
from wesep.modules.speaker.encoder import Fbank_kaldi, SpeakerEncoder
from wesep.utils.checkpoint import load_pretrained_model
from wesep.utils.score import cal_SISNRi
from wesep.utils.utils import get_logger, parse_config_or_kwargs, set_seed

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_gpu(gpu_value):
    if isinstance(gpu_value, int):
        return gpu_value
    if isinstance(gpu_value, str):
        s = gpu_value.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        s = s.split(",")[0].strip()
        return int(s)
    return 0


def _load_wav_mono(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav[:, 0]
    return torch.from_numpy(wav), sr


def _stft_mag(wav_1d, n_fft, hop):
    window = torch.hann_window(n_fft, device=wav_1d.device)
    spec = torch.stft(
        wav_1d,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    return torch.abs(spec)


def _plot_single_case(save_path, title, resp_map, mask_tar, mask_int):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(resp_map, origin="lower", aspect="auto", cmap="magma")
    axes[0].set_title("USEF Response |E|")
    axes[1].imshow(mask_tar, origin="lower", aspect="auto", cmap="Greens")
    axes[1].set_title("Target Mask")
    axes[2].imshow((mask_int * resp_map), origin="lower", aspect="auto", cmap="Reds")
    axes[2].set_title("Interf Mask * Response")
    fig.suptitle(title, fontsize=11)
    for ax in axes:
        ax.set_xlabel("Time")
        ax.set_ylabel("Freq")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _binary_auc(y_true, y_score):
    """Compute AUC from ranks (equivalent to Mann-Whitney U).
    Returns np.nan if only one class is present.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    mask = np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    if y_true.size == 0:
        return np.nan
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    # Stable sort for deterministic ranking
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)
    # Tie handling: assign average rank for ties
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j
    rank_sum_pos = ranks[y_true == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _extract_spk_embedding(wav_1d, fbank_extractor, spk_extractor, device):
    """wav_1d: [T] tensor -> embedding: [1, D] normalized"""
    x = wav_1d.to(device).unsqueeze(0)  # [1, T]
    fb = fbank_extractor(x)
    emb = spk_extractor(fb)
    if isinstance(emb, (tuple, list)):
        emb = emb[-1]
    emb = emb.reshape(emb.shape[0], -1)
    emb = F.normalize(emb, dim=-1)
    return emb


def _pmap_flat_norm(pmap_tensor):
    """pmap_tensor: [1, C, F, T] or [C, F, T] -> flattened l2-normalized [D]"""
    if pmap_tensor.dim() == 4:
        x = pmap_tensor[0]
    else:
        x = pmap_tensor
    x = x.reshape(-1)
    x = x / (torch.norm(x) + 1e-8)
    return x


def analyze(
    config="confs/conf.yaml",
    bias_jsonl_glob="data/clean/enroll_bias/test/bias_enroll20_scored.jsonl.part*",
    max_samples=400,
    conf_thresh=1.0,
    topk_vis=8,
    balance_by="json_label",
    neg_mode="wrong_speaker",
    **kwargs,
):
    configs = parse_config_or_kwargs(config, **kwargs)
    # fire may pass CLI values as strings; normalize types explicitly
    conf_thresh = float(conf_thresh)
    max_samples = int(max_samples)
    topk_vis = int(topk_vis)

    set_seed(configs["seed"])
    gpu = _parse_gpu(configs.get("gpus", 0))
    device = torch.device(f"cuda:{gpu}" if gpu >= 0 else "cpu")

    checkpoint = configs.get("checkpoint", None)
    if checkpoint is None:
        raise RuntimeError("checkpoint is required for analysis")

    model = get_model(configs["model"]["tse_model"])(configs["model_args"]["tse_model"])
    load_pretrained_model(model, checkpoint)
    model = model.to(device)
    model.eval()

    if not hasattr(model, "spk_ft") or not hasattr(model.spk_ft, "usef"):
        raise RuntimeError("Current model has no USEF frontend; cannot run this analysis.")

    # Speaker encoder branch for embedding-based similarity baseline
    try:
        spk_model_conf = configs["model_args"]["tse_model"]["speaker"]["speaker_model"]
        fbank_extractor = Fbank_kaldi(**spk_model_conf["fbank"]).to(device)
        spk_extractor = SpeakerEncoder(spk_model_conf["speaker_encoder"]).to(device)
        fbank_extractor.eval()
        spk_extractor.eval()
        for p in fbank_extractor.parameters():
            p.requires_grad = False
        for p in spk_extractor.parameters():
            p.requires_grad = False
        use_spk_baseline = True
    except Exception as ex:
        use_spk_baseline = False
        fbank_extractor = None
        spk_extractor = None

    exp_dir = configs["exp_dir"]
    out_dir = os.path.join(exp_dir, "exp18_target_confusion_usef")
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, "sample_vis")
    os.makedirs(vis_dir, exist_ok=True)

    logger = get_logger(exp_dir, "infer_exp18.log")
    logger.info("Start exp18 target confusion analysis.")
    logger.info(f"bias_jsonl_glob={bias_jsonl_glob}")
    logger.info(f"speaker_embedding_baseline_enabled={use_spk_baseline}")

    # ---- hooks: capture USEF feature map and attention matrix ----
    hook_cache = {}
    target_module = model.module if hasattr(model, "module") else model

    orig_usef_post = target_module.spk_ft.usef.post

    def hooked_usef_post(self, mix_repr, feat_repr):
        hook_cache["pmap"] = feat_repr.detach()
        return orig_usef_post(mix_repr, feat_repr)

    target_module.spk_ft.usef.post = types.MethodType(hooked_usef_post, target_module.spk_ft.usef)

    usef_att = target_module.spk_ft.usef.usef_att
    orig_att_forward = usef_att.forward

    def hooked_att_forward(self, batch, aux):
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

        attn_mat = torch.matmul(Q, K) / (emb_dim**0.5)
        attn_mat = F.softmax(attn_mat, dim=2)
        hook_cache["attn"] = attn_mat.detach()

        V = torch.matmul(attn_mat, V)
        V = V.reshape([old_shape[0], old_T, old_shape[-2], old_shape[-1]])
        V = V.transpose(1, 2)
        emb_dim = V.shape[1]
        batch = V.contiguous().view([B, self.n_head * emb_dim, old_T, old_Q])
        batch = self["attn_concat_proj"](batch)
        return batch

    usef_att.forward = types.MethodType(hooked_att_forward, usef_att)

    n_fft = int(configs["model_args"]["tse_model"]["separator"].get("win", 512))
    hop = int(configs["model_args"]["tse_model"]["separator"].get("stride", 128))

    jsonl_files = sorted(glob.glob(bias_jsonl_glob))
    if len(jsonl_files) == 0:
        raise RuntimeError(f"No jsonl files found by glob: {bias_jsonl_glob}")

    rows = []
    for jf in jsonl_files:
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if len(rows) == 0:
        raise RuntimeError("No rows loaded from bias jsonl.")

    logger.info(f"Loaded raw rows: {len(rows)}")

    # ---- candidate balanced sampling from scored bias list ----
    if balance_by not in ("json_label", "snri"):
        raise ValueError("balance_by must be 'json_label' or 'snri'")
    if neg_mode not in ("wrong_speaker",):
        raise ValueError("neg_mode must be 'wrong_speaker'")

    pos_rows = []
    neg_rows = []

    for row in rows:
        if balance_by == "json_label" and ("label" in row):
            y = int(row["label"])
        else:
            y = int(float(row.get("Dynamic_SISNRi", -999.0)) < conf_thresh)
        if y == 1:
            pos_rows.append(row)
        else:
            neg_rows.append(row)

    n_pos = len(pos_rows)
    n_neg = len(neg_rows)
    n_pair = min(n_pos, n_neg)
    if n_pair == 0:
        raise RuntimeError(f"Cannot balance samples: n_pos={n_pos}, n_neg={n_neg}")

    # NOTE:
    # keep candidate pool as large as possible (label-level balanced),
    # then do final balancing with strict negative definition later.

    rng = np.random.default_rng(configs["seed"])
    if n_pos > n_pair:
        pos_sel = [pos_rows[i] for i in rng.choice(n_pos, size=n_pair, replace=False)]
    else:
        pos_sel = pos_rows
    if n_neg > n_pair:
        neg_sel = [neg_rows[i] for i in rng.choice(n_neg, size=n_pair, replace=False)]
    else:
        neg_sel = neg_rows

    rows = pos_sel + neg_sel
    rng.shuffle(rows)

    logger.info(
        f"Candidate balanced sampling done: pos={len(pos_sel)}, neg={len(neg_sel)}, total={len(rows)}, "
        f"source_pos={n_pos}, source_neg={n_neg}, balance_by={balance_by}"
    )

    metrics = []
    plot_cache = []
    eps = 1e-8

    with torch.no_grad():
        for idx, row in enumerate(rows):
            try:
                mix_path = row["mix"]["default"][0]
                spk_list = row["spk"]
                src_map = row["src"]

                target_spk = str(spk_list[0])
                target_path = src_map[target_spk][0]
                cue_path = row["audio_spk1"]

                mix_wav, mix_sr = _load_wav_mono(mix_path)
                tgt_wav, tgt_sr = _load_wav_mono(target_path)
                cue_wav, cue_sr = _load_wav_mono(cue_path)
                if mix_sr != tgt_sr:
                    continue
                if cue_sr != mix_sr:
                    continue

                min_len = min(mix_wav.numel(), tgt_wav.numel())
                mix_wav = mix_wav[:min_len]
                tgt_wav = tgt_wav[:min_len]

                if len(spk_list) >= 2:
                    int_sum = torch.zeros_like(mix_wav)
                    for s in spk_list[1:]:
                        int_path = src_map[str(s)][0]
                        int_wav, int_sr = _load_wav_mono(int_path)
                        if int_sr != mix_sr:
                            continue
                        int_sum += int_wav[:min_len]
                    int_wav = int_sum
                else:
                    int_wav = mix_wav - tgt_wav

                mix_t = mix_wav.to(device).unsqueeze(0).unsqueeze(0)
                cue_t = cue_wav.to(device).unsqueeze(0).unsqueeze(0)
                out = model(mix_t, [cue_t])
                out_wav = out[0] if isinstance(out, (list, tuple)) else out
                out_np = out_wav.detach().cpu().numpy().flatten()

                ref_np = tgt_wav.cpu().numpy().flatten()
                int_np = int_wav.cpu().numpy().flatten()
                mix_np = mix_wav.cpu().numpy().flatten()
                end_s = min(len(out_np), len(ref_np), len(mix_np), len(int_np))
                dyn_snri, _ = cal_SISNRi(out_np[:end_s], ref_np[:end_s], mix_np[:end_s])
                int_snri, _ = cal_SISNRi(out_np[:end_s], int_np[:end_s], mix_np[:end_s])

                pmap = hook_cache.get("pmap", None)
                if pmap is None:
                    continue
                pmap = pmap[0]  # [C,F,T]
                resp_map = torch.norm(pmap, dim=0)  # [F,T]
                pmap_enroll_vec = _pmap_flat_norm(pmap)

                tgt_spec = _stft_mag(tgt_wav.to(device), n_fft, hop)
                int_spec = _stft_mag(int_wav.to(device), n_fft, hop)

                Fm = min(resp_map.shape[0], tgt_spec.shape[0], int_spec.shape[0])
                Tm = min(resp_map.shape[1], tgt_spec.shape[1], int_spec.shape[1])
                resp_map = resp_map[:Fm, :Tm]
                tgt_spec = tgt_spec[:Fm, :Tm]
                int_spec = int_spec[:Fm, :Tm]

                mask_tar = (tgt_spec >= int_spec).float()
                mask_int = (int_spec > tgt_spec).float()

                resp_tar = float((resp_map * mask_tar).sum().item() / (mask_tar.sum().item() + eps))
                resp_int = float((resp_map * mask_int).sum().item() / (mask_int.sum().item() + eps))
                sep_score = resp_tar - resp_int
                ratio_int_tar = resp_int / (resp_tar + eps)

                attn = hook_cache.get("attn", None)
                attn_entropy = np.nan
                attn_peak = np.nan
                attn_top12_margin = np.nan
                attn_diag_mass = np.nan
                attn_entropy_tar_q = np.nan
                attn_entropy_int_q = np.nan
                attn_peak_tar_q = np.nan
                attn_peak_int_q = np.nan
                if attn is not None:
                    p = attn.clamp_min(1e-9)
                    h = -(p * p.log()).sum(dim=-1)  # [B*H, T]
                    attn_entropy = float(h.mean().item())
                    peak = p.max(dim=-1).values  # [B*H, T]
                    attn_peak = float(peak.mean().item())

                    top2 = torch.topk(p, k=2, dim=-1).values
                    attn_top12_margin = float((top2[..., 0] - top2[..., 1]).mean().item())

                    attn_mean = p.mean(dim=0)  # [T, T]
                    diag_len = min(attn_mean.shape[-2], attn_mean.shape[-1])
                    if diag_len > 0:
                        diag_idx = torch.arange(diag_len, device=attn_mean.device)
                        attn_diag_mass = float(attn_mean[diag_idx, diag_idx].mean().item())

                    # query-level split by target/interference dominant frames
                    tgt_frame_e = tgt_spec.sum(dim=0)  # [T]
                    int_frame_e = int_spec.sum(dim=0)  # [T]
                    q_tar = (tgt_frame_e >= int_frame_e)
                    q_int = (int_frame_e > tgt_frame_e)

                    h_q = h.mean(dim=0)      # [T]
                    p_q = peak.mean(dim=0)   # [T]
                    Tq = min(h_q.shape[0], q_tar.shape[0])
                    q_tar = q_tar[:Tq]
                    q_int = q_int[:Tq]
                    h_q = h_q[:Tq]
                    p_q = p_q[:Tq]
                    if q_tar.any():
                        attn_entropy_tar_q = float(h_q[q_tar].mean().item())
                        attn_peak_tar_q = float(p_q[q_tar].mean().item())
                    if q_int.any():
                        attn_entropy_int_q = float(h_q[q_int].mean().item())
                        attn_peak_int_q = float(p_q[q_int].mean().item())

                row_label = int(row.get("label", 1 if dyn_snri < conf_thresh else 0))
                conf_by_snri = int(dyn_snri < conf_thresh)

                # Embedding baseline: delta_spk_sim = cos(enroll,target) - cos(enroll,interf)
                cos_enroll_target = np.nan
                cos_enroll_interf = np.nan
                delta_spk_sim = np.nan
                if use_spk_baseline:
                    try:
                        emb_enroll = _extract_spk_embedding(cue_wav, fbank_extractor, spk_extractor, device)
                        emb_target = _extract_spk_embedding(tgt_wav, fbank_extractor, spk_extractor, device)
                        emb_interf = _extract_spk_embedding(int_wav, fbank_extractor, spk_extractor, device)
                        cos_enroll_target = float((emb_enroll * emb_target).sum(dim=-1).mean().item())
                        cos_enroll_interf = float((emb_enroll * emb_interf).sum(dim=-1).mean().item())
                        delta_spk_sim = cos_enroll_target - cos_enroll_interf
                    except Exception:
                        pass

                # USEF-counterfactual baseline:
                # compare P(enroll) vs P(target-cue) / P(interf-cue) under the same mixture.
                usef_cos_target = np.nan
                usef_cos_interf = np.nan
                usef_delta_sim = np.nan
                try:
                    tgt_cue_t = tgt_wav.to(device).unsqueeze(0).unsqueeze(0)
                    _ = model(mix_t, [tgt_cue_t])
                    pmap_tgt = hook_cache.get("pmap", None)
                    if pmap_tgt is not None:
                        pmap_tgt_vec = _pmap_flat_norm(pmap_tgt[0])
                    else:
                        pmap_tgt_vec = None

                    int_cue_t = int_wav.to(device).unsqueeze(0).unsqueeze(0)
                    _ = model(mix_t, [int_cue_t])
                    pmap_int = hook_cache.get("pmap", None)
                    if pmap_int is not None:
                        pmap_int_vec = _pmap_flat_norm(pmap_int[0])
                    else:
                        pmap_int_vec = None

                    if pmap_tgt_vec is not None and pmap_int_vec is not None:
                        usef_cos_target = float((pmap_enroll_vec * pmap_tgt_vec).sum().item())
                        usef_cos_interf = float((pmap_enroll_vec * pmap_int_vec).sum().item())
                        usef_delta_sim = usef_cos_target - usef_cos_interf
                except Exception:
                    pass

                metrics.append(
                    {
                        "idx": idx,
                        "key": row.get("key", f"sample_{idx}"),
                        "target_spk": target_spk,
                        "dyn_snri": float(dyn_snri),
                        "target_sisnr": float(dyn_snri),
                        "interfer_sisnr": float(int_snri),
                        "label_jsonl": row_label,
                        "sampled_label": int(row.get("label", row_label)),
                        "label_snri": conf_by_snri,
                        "resp_tar": resp_tar,
                        "resp_int": resp_int,
                        "sep_score": sep_score,
                        "ratio_int_tar": ratio_int_tar,
                        "attn_entropy": attn_entropy,
                        "attn_peak": attn_peak,
                        "attn_top12_margin": attn_top12_margin,
                        "attn_diag_mass": attn_diag_mass,
                        "attn_entropy_tar_q": attn_entropy_tar_q,
                        "attn_entropy_int_q": attn_entropy_int_q,
                        "attn_peak_tar_q": attn_peak_tar_q,
                        "attn_peak_int_q": attn_peak_int_q,
                        "cos_enroll_target": cos_enroll_target,
                        "cos_enroll_interf": cos_enroll_interf,
                        "delta_spk_sim": delta_spk_sim,
                        "usef_cos_target": usef_cos_target,
                        "usef_cos_interf": usef_cos_interf,
                        "usef_delta_sim": usef_delta_sim,
                    }
                )
                plot_cache.append(
                    {
                        "idx": idx,
                        "key": row.get("key", f"sample_{idx}"),
                        "dyn_snri": float(dyn_snri),
                        "label": conf_by_snri,
                        "resp_map": resp_map.detach().cpu().numpy(),
                        "mask_tar": mask_tar.detach().cpu().numpy(),
                        "mask_int": mask_int.detach().cpu().numpy(),
                    }
                )
            except Exception as ex:
                logger.warning(f"Skip idx={idx}, err={ex}")
                continue

            if (idx + 1) % 50 == 0:
                logger.info(f"Processed {idx + 1}/{len(rows)}")

    if len(metrics) == 0:
        raise RuntimeError("No valid samples were analyzed.")

    df_all = pd.DataFrame(metrics)
    df_all.to_csv(os.path.join(out_dir, "metrics_all.csv"), index=False)

    # ---- final balanced set: clean positive vs wrong-speaker negative ----
    pos_df = df_all[(df_all["target_sisnr"] > 0) & (df_all["interfer_sisnr"] < 0)]
    neg_df = df_all[(df_all["target_sisnr"] < 0) & (df_all["interfer_sisnr"] > 0)]

    n_pos_final = len(pos_df)
    n_neg_final = len(neg_df)
    n_pair_final = min(n_pos_final, n_neg_final)
    if n_pair_final == 0:
        raise RuntimeError(
            f"Final balancing failed for wrong-speaker negatives: "
            f"n_pos_clean={n_pos_final}, n_neg_wrong_spk={n_neg_final}"
        )
    if max_samples and max_samples > 0:
        if max_samples < 2:
            raise ValueError("max_samples must be >=2 when positive for final balancing")
        n_pair_final = min(n_pair_final, max_samples // 2)

    rng_final = np.random.default_rng(configs["seed"] + 7)
    pos_idx = pos_df.index.to_numpy()
    neg_idx = neg_df.index.to_numpy()
    pos_take = rng_final.choice(pos_idx, size=n_pair_final, replace=False)
    neg_take = rng_final.choice(neg_idx, size=n_pair_final, replace=False)
    keep_idx = np.concatenate([pos_take, neg_take])
    rng_final.shuffle(keep_idx)

    df = df_all.loc[keep_idx].copy().reset_index(drop=True)
    df["analysis_label"] = ((df["target_sisnr"] < 0) & (df["interfer_sisnr"] > 0)).astype(int)
    logger.info(
        "Final balanced set (clean pos vs wrong-speaker neg): "
        f"pos={int((df['analysis_label'] == 0).sum())}, "
        f"neg={int((df['analysis_label'] == 1).sum())}, total={len(df)}"
    )
    df.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    # ---- AUC report: per-metric separability against analysis_label ----
    df["delta_attn_entropy_q"] = df["attn_entropy_tar_q"] - df["attn_entropy_int_q"]
    df["delta_attn_peak_q"] = df["attn_peak_tar_q"] - df["attn_peak_int_q"]

    auc_metrics = [
        "sep_score",
        "resp_tar",
        "resp_int",
        "ratio_int_tar",
        "attn_entropy",
        "attn_peak",
        "attn_top12_margin",
        "attn_diag_mass",
        "attn_entropy_tar_q",
        "attn_entropy_int_q",
        "attn_peak_tar_q",
        "attn_peak_int_q",
        "delta_attn_entropy_q",
        "delta_attn_peak_q",
        "cos_enroll_target",
        "cos_enroll_interf",
        "delta_spk_sim",
        "usef_cos_target",
        "usef_cos_interf",
        "usef_delta_sim",
    ]
    auc_rows = []
    y = df["analysis_label"].to_numpy()
    for m in auc_metrics:
        if m not in df.columns:
            continue
        x = df[m].to_numpy()
        auc = _binary_auc(y, x)
        auc_inv = _binary_auc(y, -x)
        best_auc = np.nanmax([auc, auc_inv]) if not (np.isnan(auc) and np.isnan(auc_inv)) else np.nan
        direction = "high->confusion" if auc >= auc_inv else "low->confusion"
        auc_rows.append(
            {
                "metric": m,
                "auc_high_is_confusion": auc,
                "auc_low_is_confusion": auc_inv,
                "auc_best": best_auc,
                "best_direction": direction,
            }
        )
    auc_df = pd.DataFrame(auc_rows).sort_values(by="auc_best", ascending=False)
    auc_df.to_csv(os.path.join(out_dir, "auc_report.csv"), index=False)

    # ---- global plots ----
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    colors = np.where(df["analysis_label"].values == 1, "#d62728", "#2ca02c")
    ax1.scatter(df["resp_tar"], df["resp_int"], c=colors, s=18, alpha=0.75)
    ax1.set_xlabel("Resp_tar (Target-mask weighted)")
    ax1.set_ylabel("Resp_int (Interf-mask weighted)")
    ax1.set_title("USEF Response Separation (red=confusion)")
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, "scatter_resp_tar_vs_resp_int.png"), dpi=260)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    ax2.scatter(df["sep_score"], df["dyn_snri"], c=colors, s=18, alpha=0.75)
    ax2.axvline(0.0, linestyle="--", color="gray", linewidth=1)
    ax2.axhline(conf_thresh, linestyle="--", color="gray", linewidth=1)
    ax2.set_xlabel("Sep_score = Resp_tar - Resp_int")
    ax2.set_ylabel("Dynamic SI-SNRi (dB)")
    ax2.set_title("Separation Score vs SI-SNRi")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "scatter_sep_vs_sisnri.png"), dpi=260)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    conf_vals = df[df["analysis_label"] == 1]["sep_score"].values
    ok_vals = df[df["analysis_label"] == 0]["sep_score"].values
    bins = 40
    ax3.hist(ok_vals, bins=bins, alpha=0.55, label="non-confusion", color="#2ca02c")
    ax3.hist(conf_vals, bins=bins, alpha=0.55, label="confusion", color="#d62728")
    ax3.set_xlabel("Sep_score")
    ax3.set_ylabel("Count")
    ax3.set_title("Sep_score Distribution")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, "hist_sep_score.png"), dpi=260)
    plt.close(fig3)

    # ---- sample visualizations ----
    keep_idx_set = set(df["idx"].astype(int).tolist())
    conf_idx_set = set(df[df["analysis_label"] == 1]["idx"].astype(int).tolist())
    ok_idx_set = set(df[df["analysis_label"] == 0]["idx"].astype(int).tolist())

    conf_cases = [x for x in plot_cache if int(x["idx"]) in keep_idx_set and int(x["idx"]) in conf_idx_set]
    ok_cases = [x for x in plot_cache if int(x["idx"]) in keep_idx_set and int(x["idx"]) in ok_idx_set]
    conf_cases = sorted(conf_cases, key=lambda x: x["dyn_snri"])[:topk_vis]
    ok_cases = sorted(ok_cases, key=lambda x: -x["dyn_snri"])[:topk_vis]

    for c in conf_cases:
        name = f"conf_idx{c['idx']:04d}_{c['key']}.png".replace("/", "_")
        title = f"CONFUSION | key={c['key']} | SI-SNRi={c['dyn_snri']:.2f} dB"
        _plot_single_case(
            os.path.join(vis_dir, name), title, c["resp_map"], c["mask_tar"], c["mask_int"]
        )

    for c in ok_cases:
        name = f"ok_idx{c['idx']:04d}_{c['key']}.png".replace("/", "_")
        title = f"NON-CONFUSION | key={c['key']} | SI-SNRi={c['dyn_snri']:.2f} dB"
        _plot_single_case(
            os.path.join(vis_dir, name), title, c["resp_map"], c["mask_tar"], c["mask_int"]
        )

    # ---- summary ----
    n_total = len(df)
    n_conf = int((df["analysis_label"] == 1).sum())
    n_ok = n_total - n_conf
    summary_lines = [
        f"total={n_total}",
        f"confusion={n_conf}",
        f"non_confusion={n_ok}",
        "negative_definition=wrong_speaker(target_sisnr<0 & interfer_sisnr>0)",
        f"mean_sep_confusion={df[df['analysis_label'] == 1]['sep_score'].mean():.6f}",
        f"mean_sep_non_confusion={df[df['analysis_label'] == 0]['sep_score'].mean():.6f}",
        f"mean_attn_entropy_confusion={df[df['analysis_label'] == 1]['attn_entropy'].mean():.6f}",
        f"mean_attn_entropy_non_confusion={df[df['analysis_label'] == 0]['attn_entropy'].mean():.6f}",
        f"mean_attn_peak_confusion={df[df['analysis_label'] == 1]['attn_peak'].mean():.6f}",
        f"mean_attn_peak_non_confusion={df[df['analysis_label'] == 0]['attn_peak'].mean():.6f}",
        f"mean_attn_margin_confusion={df[df['analysis_label'] == 1]['attn_top12_margin'].mean():.6f}",
        f"mean_attn_margin_non_confusion={df[df['analysis_label'] == 0]['attn_top12_margin'].mean():.6f}",
        f"mean_attn_diag_confusion={df[df['analysis_label'] == 1]['attn_diag_mass'].mean():.6f}",
        f"mean_attn_diag_non_confusion={df[df['analysis_label'] == 0]['attn_diag_mass'].mean():.6f}",
    ]
    if len(auc_df) > 0:
        summary_lines.append("top_auc_metrics:")
        for _, r in auc_df.head(5).iterrows():
            summary_lines.append(
                f"  {r['metric']}: auc_best={r['auc_best']:.4f}, direction={r['best_direction']}"
            )
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    logger.info("exp18 done.")
    logger.info(f"Output dir: {out_dir}")

    # restore hooks
    target_module.spk_ft.usef.post = orig_usef_post
    usef_att.forward = orig_att_forward


if __name__ == "__main__":
    fire.Fire({"analyze": analyze})
