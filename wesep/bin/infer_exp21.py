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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

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


def _best_threshold(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    mask = np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    if y_true.size == 0:
        return np.nan, np.nan
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        y_hat = (y_score >= t).astype(int)
        f1 = f1_score(y_true, y_hat, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, float(best_f1)


class AttnTensorClassifier(nn.Module):
    """
    Input: [B, H, Tq, Tk] (resized to fixed T)
    Strategy: collapse Tk by learned weighted pooling + 2D CNN over (H, Tq)
    """

    def __init__(self, n_head=4, t_bins=96, hidden=128, dropout=0.2):
        super().__init__()
        self.n_head = n_head
        self.t_bins = t_bins

        self.conv = nn.Sequential(
            nn.Conv2d(n_head, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x: [B,H,T,T]
        z = self.conv(x)
        return self.head(z).squeeze(-1)


def _resize_attn(attn_bh, n_head, t_bins=96):
    """
    attn_bh: [B*H, Tq, Tk] -> [B,H,t_bins,t_bins]
    """
    bh, tq, tk = attn_bh.shape
    if bh % n_head != 0:
        return None
    bsz = bh // n_head
    x = attn_bh.view(bsz, n_head, tq, tk)
    x = F.interpolate(x, size=(t_bins, t_bins), mode="bilinear", align_corners=False)
    return x


def _hook_usef_attn(model):
    target_module = model.module if hasattr(model, "module") else model
    usef_att = target_module.spk_ft.usef.usef_att
    orig_att_forward = usef_att.forward
    hook_cache = {}

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
    return usef_att, orig_att_forward, hook_cache


def _load_rows(glob_pat):
    rows = []
    files = sorted(glob.glob(glob_pat))
    if len(files) == 0:
        raise RuntimeError(f"No files found by glob: {glob_pat}")
    for jf in files:
        with open(jf, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    rows.append(json.loads(ln))
    if len(rows) == 0:
        raise RuntimeError("No rows loaded.")
    return rows


def _balanced_rows(rows, conf_thresh, seed, max_samples):
    pos, neg = [], []
    for row in rows:
        y = int(row.get("label", int(float(row.get("Dynamic_SISNRi", -999.0)) < conf_thresh)))
        if y == 1:
            pos.append(row)
        else:
            neg.append(row)
    n = min(len(pos), len(neg))
    if n == 0:
        raise RuntimeError("Cannot balance: no positive or negative rows.")
    if max_samples > 0:
        n = min(n, max_samples // 2)
    rng = np.random.default_rng(seed)
    pos_sel = [pos[i] for i in rng.choice(len(pos), size=n, replace=False)]
    neg_sel = [neg[i] for i in rng.choice(len(neg), size=n, replace=False)]
    out = pos_sel + neg_sel
    rng.shuffle(out)
    return out


def collect(
    config="confs/conf.yaml",
    split="train",
    bias_jsonl_glob="data/clean/enroll_bias/train-100/bias_enroll20_scored.jsonl.part*",
    max_samples=4000,
    conf_thresh=1.0,
    t_bins=96,
    out_tag="exp21_early_usef_predictor",
    **kwargs,
):
    configs = parse_config_or_kwargs(config, **kwargs)
    set_seed(configs["seed"])
    random.seed(configs["seed"])

    gpu = _parse_gpu(configs.get("gpus", 0))
    device = torch.device(f"cuda:{gpu}" if gpu >= 0 else "cpu")

    model = get_model(configs["model"]["tse_model"])(configs["model_args"]["tse_model"])
    load_pretrained_model(model, configs["checkpoint"])
    model = model.to(device)
    model.eval()

    if not hasattr(model, "spk_ft") or not hasattr(model.spk_ft, "usef"):
        raise RuntimeError("Current model has no USEF frontend.")

    exp_dir = configs["exp_dir"]
    out_dir = os.path.join(exp_dir, out_tag)
    os.makedirs(out_dir, exist_ok=True)
    logger = get_logger(exp_dir, "infer_exp21.log")

    usef_att, orig_att_forward, hook_cache = _hook_usef_attn(model)

    rows = _load_rows(bias_jsonl_glob)
    rows = _balanced_rows(rows, conf_thresh=conf_thresh, seed=configs["seed"], max_samples=int(max_samples))
    logger.info(f"[exp21 collect] split={split}, balanced rows={len(rows)}")

    X, y, keys = [], [], []
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

                mix_t = mix_wav.to(device).unsqueeze(0).unsqueeze(0)
                cue_t = cue_wav.to(device).unsqueeze(0).unsqueeze(0)

                out = model(mix_t, [cue_t])
                out_wav = out[0] if isinstance(out, (list, tuple)) else out
                out_np = out_wav.detach().cpu().numpy().flatten()
                ref_np = tgt_wav.cpu().numpy().flatten()
                mix_np = mix_wav.cpu().numpy().flatten()
                end_s = min(len(out_np), len(ref_np), len(mix_np))
                dyn_snri, _ = cal_SISNRi(out_np[:end_s], ref_np[:end_s], mix_np[:end_s])
                label = int(dyn_snri < conf_thresh)

                attn = hook_cache.get("attn", None)
                if attn is None:
                    continue
                attn_resized = _resize_attn(attn, n_head=usef_att.n_head, t_bins=int(t_bins))
                if attn_resized is None:
                    continue

                X.append(attn_resized[0].cpu().numpy().astype(np.float32))  # [H,T,T]
                y.append(label)
                keys.append(row.get("key", f"sample_{idx}"))
            except Exception as ex:
                logger.warning(f"Skip idx={idx}, err={ex}")
                continue

            if (idx + 1) % 100 == 0:
                logger.info(f"collect processed {idx + 1}/{len(rows)}")

    usef_att.forward = orig_att_forward

    if len(X) == 0:
        raise RuntimeError("No valid sample collected.")

    X = np.stack(X, axis=0)  # [N,H,T,T]
    y = np.asarray(y, dtype=np.int64)

    out_npz = os.path.join(out_dir, f"{split}_attn_tensor.npz")
    np.savez_compressed(out_npz, X=X, y=y, keys=np.array(keys, dtype=object))

    logger.info(f"[exp21 collect] saved: {out_npz}, N={len(y)}, pos={(y==1).sum()}, neg={(y==0).sum()}")


def train(
    config="confs/conf.yaml",
    out_tag="exp21_early_usef_predictor",
    train_npz="",
    val_npz="",
    epochs=20,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    pos_weight=1.0,
    hidden=128,
    dropout=0.2,
    seed=42,
    **kwargs,
):
    configs = parse_config_or_kwargs(config, **kwargs)
    set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    gpu = _parse_gpu(configs.get("gpus", 0))
    device = torch.device(f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu")

    exp_dir = configs["exp_dir"]
    out_dir = os.path.join(exp_dir, out_tag)
    os.makedirs(out_dir, exist_ok=True)
    logger = get_logger(exp_dir, "infer_exp21.log")

    if train_npz == "":
        train_npz = os.path.join(out_dir, "train_attn_tensor.npz")
    if val_npz == "":
        val_npz = os.path.join(out_dir, "dev_attn_tensor.npz")

    tr = np.load(train_npz, allow_pickle=True)
    va = np.load(val_npz, allow_pickle=True)

    Xtr, ytr = tr["X"], tr["y"]
    Xva, yva = va["X"], va["y"]

    n_head = int(Xtr.shape[1])
    t_bins = int(Xtr.shape[2])

    model = AttnTensorClassifier(n_head=n_head, t_bins=t_bins, hidden=int(hidden), dropout=float(dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)], device=device))

    tr_idx = np.arange(len(ytr))
    best_auc = -1.0
    best_state = None

    for ep in range(1, int(epochs) + 1):
        np.random.shuffle(tr_idx)
        model.train()
        losses = []

        for s in range(0, len(tr_idx), int(batch_size)):
            ids = tr_idx[s:s + int(batch_size)]
            xb = torch.from_numpy(Xtr[ids]).to(device)
            yb = torch.from_numpy(ytr[ids].astype(np.float32)).to(device)

            logits = model(xb)
            loss = crit(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        # val
        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(Xva).to(device)
            lv = model(xv)
            pv = torch.sigmoid(lv).cpu().numpy()
            yv = yva.astype(int)

            try:
                auc = float(roc_auc_score(yv, pv))
            except Exception:
                auc = np.nan
            try:
                ap = float(average_precision_score(yv, pv))
            except Exception:
                ap = np.nan
            th, best_f1 = _best_threshold(yv, pv)

        logger.info(
            f"[exp21 train] epoch={ep} train_loss={np.mean(losses):.4f} "
            f"val_auc={auc:.4f} val_ap={ap:.4f} val_best_f1={best_f1:.4f} best_th={th:.2f}"
        )

        if np.isfinite(auc) and auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "epoch": ep,
                    "val_auc": auc,
                    "val_ap": ap,
                    "val_best_f1": best_f1,
                    "best_threshold": th,
                    "model_args": {
                        "n_head": n_head,
                        "t_bins": t_bins,
                        "hidden": int(hidden),
                        "dropout": float(dropout),
                    },
                    "state_dict": model.state_dict(),
                },
                os.path.join(out_dir, "best_model.pt"),
            )

    # final eval dump
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        xv = torch.from_numpy(Xva).to(device)
        pv = torch.sigmoid(model(xv)).cpu().numpy()
    yv = yva.astype(int)
    th, best_f1 = _best_threshold(yv, pv)
    auc = float(roc_auc_score(yv, pv)) if len(np.unique(yv)) == 2 else np.nan
    ap = float(average_precision_score(yv, pv)) if len(np.unique(yv)) == 2 else np.nan

    out_csv = os.path.join(out_dir, "val_predictions.csv")
    pd.DataFrame({"y": yv, "score": pv, "pred": (pv >= th).astype(int)}).to_csv(out_csv, index=False)

    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"best_val_auc={auc:.4f}\n")
        f.write(f"best_val_ap={ap:.4f}\n")
        f.write(f"best_threshold={th:.4f}\n")
        f.write(f"best_val_f1={best_f1:.4f}\n")
        f.write(f"train_npz={train_npz}\n")
        f.write(f"val_npz={val_npz}\n")

    logger.info(f"[exp21 train] done. auc={auc:.4f}, ap={ap:.4f}, th={th:.3f}, f1={best_f1:.4f}")


if __name__ == "__main__":
    fire.Fire({"collect": collect, "train": train})
