from __future__ import print_function

import glob
import json
import os
import random
import types

import fire
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F

from wesep.models import get_model
from wesep.utils.checkpoint import load_pretrained_model
from wesep.utils.score import cal_SISNRi
from wesep.utils.utils import get_logger, parse_config_or_kwargs, set_seed


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


def _binary_auc(y_true, y_score):
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

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)

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


def _compute_attn_stats(attn, n_head):
    # attn: [B*H, Tq, Tk]
    p = attn.clamp_min(1e-9)
    h = -(p * p.log()).sum(dim=-1)  # [B*H, Tq]
    peak = p.max(dim=-1).values      # [B*H, Tq]
    k_top = min(2, int(p.shape[-1]))
    top2 = torch.topk(p, k=k_top, dim=-1).values

    attn_entropy = float(h.mean().item())
    attn_peak = float(peak.mean().item())
    if k_top == 2:
        margin = (top2[..., 0] - top2[..., 1])
    else:
        margin = top2[..., 0]
    attn_top12_margin = float(margin.mean().item())
    attn_top12_margin_std = float(margin.std(unbiased=False).item())

    attn_mean = p.mean(dim=0)  # [Tq, Tk]
    diag_len = min(attn_mean.shape[-2], attn_mean.shape[-1])
    if diag_len > 0:
        diag_idx = torch.arange(diag_len, device=attn_mean.device)
        attn_diag_mass = float(attn_mean[diag_idx, diag_idx].mean().item())
    else:
        attn_diag_mass = np.nan

    # old head disagreement proxy: variance across heads
    head_mean = p.mean(dim=0, keepdim=True)
    head_var = ((p - head_mean) ** 2).mean(dim=0)  # [Tq, Tk]
    attn_head_disagree = float(head_var.mean().item())

    # explicit head top1 consistency (high means more consistent matching across heads)
    attn_head_top1_consistency = np.nan
    if n_head is not None and int(n_head) > 0 and (p.shape[0] % int(n_head) == 0):
        B = p.shape[0] // int(n_head)
        argmax_idx = p.argmax(dim=-1).view(B, int(n_head), -1)  # [B, H, Tq]
        # for each (b, tq), compute dominant vote ratio among heads
        cons_vals = []
        for b in range(B):
            for t in range(argmax_idx.shape[-1]):
                vals = argmax_idx[b, :, t]
                uniq, cnt = torch.unique(vals, return_counts=True)
                cons_vals.append(float(cnt.max().item()) / float(int(n_head)))
        if len(cons_vals) > 0:
            attn_head_top1_consistency = float(np.mean(cons_vals))

    # effective rank of attention mean matrix (retrieval ambiguity proxy)
    attn_erank = np.nan
    attn_erank_norm = np.nan
    try:
        m = attn_mean.float()
        # speed guard: downsample large attention maps before SVD
        max_side = 256
        if m.shape[0] > max_side:
            idx0 = torch.linspace(0, m.shape[0] - 1, steps=max_side, device=m.device).long()
            m = m[idx0, :]
        if m.shape[1] > max_side:
            idx1 = torch.linspace(0, m.shape[1] - 1, steps=max_side, device=m.device).long()
            m = m[:, idx1]
        s = torch.linalg.svdvals(m)  # [min(Tq, Tk)]
        s_sum = float(s.sum().item())
        if s.numel() > 0 and s_sum > 0:
            ps = (s / s.sum()).clamp_min(1e-12)
            h_s = float((-(ps * ps.log()).sum()).item())
            erank = float(np.exp(h_s))
            attn_erank = erank
            attn_erank_norm = float(erank / float(s.numel()))
    except Exception:
        pass

    return {
        "attn_entropy": attn_entropy,
        "attn_peak": attn_peak,
        "attn_top12_margin": attn_top12_margin,
        "attn_top12_margin_std": attn_top12_margin_std,
        "attn_diag_mass": attn_diag_mass,
        "attn_head_disagree": attn_head_disagree,
        "attn_head_top1_consistency": attn_head_top1_consistency,
        "attn_erank": attn_erank,
        "attn_erank_norm": attn_erank_norm,
    }


def _random_crop_1d(wav_1d, ratio_low=0.6, ratio_high=0.95):
    T = wav_1d.numel()
    if T < 160:
        return wav_1d
    ratio = random.uniform(ratio_low, ratio_high)
    L = max(80, int(T * ratio))
    if L >= T:
        return wav_1d
    s = random.randint(0, T - L)
    return wav_1d[s:s + L]


def _forward_collect_attn(model, mix_t, cue_t, hook_cache, n_head):
    _ = model(mix_t, [cue_t])
    attn = hook_cache.get("attn", None)
    if attn is None:
        return None
    return _compute_attn_stats(attn, n_head=n_head)


def analyze(
    config="confs/conf.yaml",
    bias_jsonl_glob="data/clean/enroll_bias/test/bias_enroll20_scored.jsonl.part*",
    max_samples=200,
    conf_thresh=1.0,
    balance_by="json_label",
    num_stability_crops=3,
    num_robust_cues=3,
    **kwargs,
):
    configs = parse_config_or_kwargs(config, **kwargs)
    conf_thresh = float(conf_thresh)
    max_samples = int(max_samples)
    num_stability_crops = int(num_stability_crops)
    num_robust_cues = int(num_robust_cues)

    set_seed(configs["seed"])
    random.seed(configs["seed"])
    gpu = _parse_gpu(configs.get("gpus", 0))
    device = torch.device(f"cuda:{gpu}" if gpu >= 0 else "cpu")

    checkpoint = configs.get("checkpoint", None)
    if checkpoint is None:
        raise RuntimeError("checkpoint is required")

    model = get_model(configs["model"]["tse_model"])(configs["model_args"]["tse_model"])
    load_pretrained_model(model, checkpoint)
    model = model.to(device)
    model.eval()

    if not hasattr(model, "spk_ft") or not hasattr(model.spk_ft, "usef"):
        raise RuntimeError("Current model has no USEF frontend.")

    exp_dir = configs["exp_dir"]
    out_dir = os.path.join(exp_dir, "exp19_deployable_attn_auc")
    os.makedirs(out_dir, exist_ok=True)
    logger = get_logger(exp_dir, "infer_exp19.log")
    logger.info("Start exp19 deployment-feasible AUC analysis.")
    logger.info(f"bias_jsonl_glob={bias_jsonl_glob}")

    # hooks
    hook_cache = {}
    target_module = model.module if hasattr(model, "module") else model
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

        attn_mat = torch.matmul(Q, K) / (emb_dim ** 0.5)
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

    # load rows
    jsonl_files = sorted(glob.glob(bias_jsonl_glob))
    if len(jsonl_files) == 0:
        raise RuntimeError(f"No files found by glob: {bias_jsonl_glob}")
    rows = []
    for jf in jsonl_files:
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if len(rows) == 0:
        raise RuntimeError("No rows loaded.")

    # label-level balancing first
    pos_rows, neg_rows = [], []
    for row in rows:
        if balance_by == "json_label" and ("label" in row):
            y = int(row["label"])
        else:
            y = int(float(row.get("Dynamic_SISNRi", -999.0)) < conf_thresh)
        if y == 1:
            pos_rows.append(row)
        else:
            neg_rows.append(row)
    n_pair = min(len(pos_rows), len(neg_rows))
    if n_pair == 0:
        raise RuntimeError("Cannot balance candidates, no positive or negative rows.")
    if max_samples > 0:
        n_pair = min(n_pair, max_samples // 2)
    rng = np.random.default_rng(configs["seed"])
    pos_sel = [pos_rows[i] for i in rng.choice(len(pos_rows), size=n_pair, replace=False)]
    neg_sel = [neg_rows[i] for i in rng.choice(len(neg_rows), size=n_pair, replace=False)]
    rows = pos_sel + neg_sel
    rng.shuffle(rows)
    logger.info(f"Balanced candidate set: pos={len(pos_sel)}, neg={len(neg_sel)}, total={len(rows)}")

    metrics = []
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
                if mix_sr != tgt_sr or cue_sr != mix_sr:
                    continue

                min_len = min(mix_wav.numel(), tgt_wav.numel())
                mix_wav = mix_wav[:min_len]
                tgt_wav = tgt_wav[:min_len]
                if len(spk_list) >= 2:
                    int_sum = torch.zeros_like(mix_wav)
                    for s in spk_list[1:]:
                        int_path = src_map[str(s)][0]
                        int_wav, int_sr = _load_wav_mono(int_path)
                        if int_sr == mix_sr:
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
                end_s = min(len(out_np), len(ref_np), len(int_np), len(mix_np))
                target_sisnr, _ = cal_SISNRi(out_np[:end_s], ref_np[:end_s], mix_np[:end_s])
                interfer_sisnr, _ = cal_SISNRi(out_np[:end_s], int_np[:end_s], mix_np[:end_s])
                analysis_label = int((target_sisnr < 0) and (interfer_sisnr > 0))

                # branch 1: global
                g = _forward_collect_attn(model, mix_t, cue_t, hook_cache, n_head=usef_att.n_head)
                if g is None:
                    continue

                # branch 2: stability via cropped mixture
                stab_vals = []
                for _ in range(max(1, num_stability_crops)):
                    mix_crop = _random_crop_1d(mix_wav)
                    mix_crop_t = mix_crop.to(device).unsqueeze(0).unsqueeze(0)
                    s = _forward_collect_attn(model, mix_crop_t, cue_t, hook_cache, n_head=usef_att.n_head)
                    if s is not None:
                        stab_vals.append(s)
                # aggregate std over selected stats
                if len(stab_vals) >= 2:
                    arr_entropy = np.array([x["attn_entropy"] for x in stab_vals], dtype=float)
                    arr_peak = np.array([x["attn_peak"] for x in stab_vals], dtype=float)
                    arr_margin = np.array([x["attn_top12_margin"] for x in stab_vals], dtype=float)
                    arr_margin_std = np.array([x["attn_top12_margin_std"] for x in stab_vals], dtype=float)
                    arr_diag = np.array([x["attn_diag_mass"] for x in stab_vals], dtype=float)
                    arr_cons = np.array([x["attn_head_top1_consistency"] for x in stab_vals], dtype=float)
                    arr_erank = np.array([x["attn_erank_norm"] for x in stab_vals], dtype=float)
                    stability_entropy_std = float(np.std(arr_entropy))
                    stability_peak_std = float(np.std(arr_peak))
                    stability_margin_std = float(np.std(arr_margin))
                    stability_margin2_std = float(np.std(arr_margin_std))
                    stability_diag_std = float(np.std(arr_diag))
                    stability_consistency_std = float(np.std(arr_cons))
                    stability_erank_std = float(np.std(arr_erank))
                else:
                    stability_entropy_std = np.nan
                    stability_peak_std = np.nan
                    stability_margin_std = np.nan
                    stability_margin2_std = np.nan
                    stability_diag_std = np.nan
                    stability_consistency_std = np.nan
                    stability_erank_std = np.nan

                # branch 3: cue robustness via cue perturbation
                rob_vals = []
                for _ in range(max(1, num_robust_cues)):
                    cue_crop = _random_crop_1d(cue_wav, ratio_low=0.5, ratio_high=0.9)
                    if cue_crop.numel() > 200:
                        noise = 0.005 * torch.randn_like(cue_crop)
                        cue_crop = cue_crop + noise
                    cue_crop_t = cue_crop.to(device).unsqueeze(0).unsqueeze(0)
                    r = _forward_collect_attn(model, mix_t, cue_crop_t, hook_cache, n_head=usef_att.n_head)
                    if r is not None:
                        rob_vals.append(r)
                if len(rob_vals) >= 1:
                    arr_entropy = np.array([x["attn_entropy"] for x in rob_vals], dtype=float)
                    arr_peak = np.array([x["attn_peak"] for x in rob_vals], dtype=float)
                    arr_margin = np.array([x["attn_top12_margin"] for x in rob_vals], dtype=float)
                    arr_margin_std = np.array([x["attn_top12_margin_std"] for x in rob_vals], dtype=float)
                    arr_diag = np.array([x["attn_diag_mass"] for x in rob_vals], dtype=float)
                    arr_cons = np.array([x["attn_head_top1_consistency"] for x in rob_vals], dtype=float)
                    arr_erank = np.array([x["attn_erank_norm"] for x in rob_vals], dtype=float)
                    robust_entropy_shift = float(np.nanmean(np.abs(arr_entropy - g["attn_entropy"])))
                    robust_peak_shift = float(np.nanmean(np.abs(arr_peak - g["attn_peak"])))
                    robust_margin_shift = float(np.nanmean(np.abs(arr_margin - g["attn_top12_margin"])))
                    robust_margin2_shift = float(np.nanmean(np.abs(arr_margin_std - g["attn_top12_margin_std"])))
                    robust_diag_shift = float(np.nanmean(np.abs(arr_diag - g["attn_diag_mass"])))
                    robust_consistency_shift = float(np.nanmean(np.abs(arr_cons - g["attn_head_top1_consistency"])))
                    robust_erank_shift = float(np.nanmean(np.abs(arr_erank - g["attn_erank_norm"])))
                else:
                    robust_entropy_shift = np.nan
                    robust_peak_shift = np.nan
                    robust_margin_shift = np.nan
                    robust_margin2_shift = np.nan
                    robust_diag_shift = np.nan
                    robust_consistency_shift = np.nan
                    robust_erank_shift = np.nan

                metrics.append(
                    {
                        "idx": idx,
                        "key": row.get("key", f"sample_{idx}"),
                        "analysis_label": analysis_label,
                        "target_sisnr": float(target_sisnr),
                        "interfer_sisnr": float(interfer_sisnr),
                        # attn-global
                        "attn_entropy": g["attn_entropy"],
                        "attn_peak": g["attn_peak"],
                        "attn_top12_margin": g["attn_top12_margin"],
                        "attn_top12_margin_std": g["attn_top12_margin_std"],
                        "attn_diag_mass": g["attn_diag_mass"],
                        "attn_head_disagree": g["attn_head_disagree"],
                        "attn_head_top1_consistency": g["attn_head_top1_consistency"],
                        "attn_erank": g["attn_erank"],
                        "attn_erank_norm": g["attn_erank_norm"],
                        # attn-stability
                        "stability_entropy_std": stability_entropy_std,
                        "stability_peak_std": stability_peak_std,
                        "stability_margin_std": stability_margin_std,
                        "stability_margin2_std": stability_margin2_std,
                        "stability_diag_std": stability_diag_std,
                        "stability_consistency_std": stability_consistency_std,
                        "stability_erank_std": stability_erank_std,
                        # cue-robustness
                        "robust_entropy_shift": robust_entropy_shift,
                        "robust_peak_shift": robust_peak_shift,
                        "robust_margin_shift": robust_margin_shift,
                        "robust_margin2_shift": robust_margin2_shift,
                        "robust_diag_shift": robust_diag_shift,
                        "robust_consistency_shift": robust_consistency_shift,
                        "robust_erank_shift": robust_erank_shift,
                    }
                )

            except Exception as ex:
                logger.warning(f"Skip idx={idx}, err={ex}")
                continue

            if (idx + 1) % 50 == 0:
                logger.info(f"Processed {idx + 1}/{len(rows)}")

    if len(metrics) == 0:
        raise RuntimeError("No valid sample processed.")

    df = pd.DataFrame(metrics)
    # final strict balance on deploy target label
    pos_df = df[df["analysis_label"] == 1]
    neg_df = df[df["analysis_label"] == 0]
    n_pair_final = min(len(pos_df), len(neg_df))
    if n_pair_final == 0:
        raise RuntimeError("No final balanced set for analysis_label.")
    if max_samples > 0:
        n_pair_final = min(n_pair_final, max_samples // 2)
    rng2 = np.random.default_rng(configs["seed"] + 19)
    pos_idx = rng2.choice(pos_df.index.to_numpy(), size=n_pair_final, replace=False)
    neg_idx = rng2.choice(neg_df.index.to_numpy(), size=n_pair_final, replace=False)
    keep_idx = np.concatenate([pos_idx, neg_idx])
    rng2.shuffle(keep_idx)
    df = df.loc[keep_idx].reset_index(drop=True)
    df.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    # AUC report by branches
    branch_map = {
        "attn_global": [
            "attn_entropy",
            "attn_peak",
            "attn_top12_margin",
            "attn_top12_margin_std",
            "attn_diag_mass",
            "attn_head_disagree",
            "attn_head_top1_consistency",
            "attn_erank",
            "attn_erank_norm",
        ],
        "attn_stability": [
            "stability_entropy_std",
            "stability_peak_std",
            "stability_margin_std",
            "stability_margin2_std",
            "stability_diag_std",
            "stability_consistency_std",
            "stability_erank_std",
        ],
        "cue_robustness": [
            "robust_entropy_shift",
            "robust_peak_shift",
            "robust_margin_shift",
            "robust_margin2_shift",
            "robust_diag_shift",
            "robust_consistency_shift",
            "robust_erank_shift",
        ],
    }
    y = df["analysis_label"].to_numpy()
    auc_rows = []
    for branch, feats in branch_map.items():
        for feat in feats:
            if feat not in df.columns:
                continue
            x = df[feat].to_numpy()
            auc_h = _binary_auc(y, x)
            auc_l = _binary_auc(y, -x)
            if np.isnan(auc_h) and np.isnan(auc_l):
                best_auc = np.nan
                direction = "na"
            else:
                if auc_h >= auc_l:
                    best_auc = auc_h
                    direction = "high->confusion"
                else:
                    best_auc = auc_l
                    direction = "low->confusion"
            auc_rows.append(
                {
                    "branch": branch,
                    "metric": feat,
                    "auc_high_is_confusion": auc_h,
                    "auc_low_is_confusion": auc_l,
                    "auc_best": best_auc,
                    "best_direction": direction,
                }
            )
    auc_df = pd.DataFrame(auc_rows).sort_values(by="auc_best", ascending=False)
    auc_df.to_csv(os.path.join(out_dir, "auc_report.csv"), index=False)

    # summary
    lines = [
        f"total={len(df)}",
        f"confusion={int((df['analysis_label'] == 1).sum())}",
        f"non_confusion={int((df['analysis_label'] == 0).sum())}",
        f"num_stability_crops={num_stability_crops}",
        f"num_robust_cues={num_robust_cues}",
        "top_auc_metrics:",
    ]
    for _, r in auc_df.head(10).iterrows():
        lines.append(
            f"  [{r['branch']}] {r['metric']}: auc_best={r['auc_best']:.4f}, direction={r['best_direction']}"
        )
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("exp19 done.")
    logger.info(f"Output dir: {out_dir}")

    # restore
    usef_att.forward = orig_att_forward


if __name__ == "__main__":
    fire.Fire({"analyze": analyze})
