from __future__ import print_function

import json
import os
import time
import types

import fire
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from wesep.dataset.collate import AUX_KEY_MAP, BASE_COLLECT_KEYS, build_collect_keys, tse_collate_fn
from wesep.dataset.dataset import Dataset
from wesep.models import get_model
from wesep.utils.checkpoint import load_pretrained_model
from wesep.utils.file_utils import load_yaml
from wesep.utils.losses import parse_loss
from wesep.utils.score import cal_SISNRi
from wesep.utils.utils import get_logger, parse_config_or_kwargs, set_seed


class DeployableRiskHead(nn.Module):

    def __init__(self, in_dim=8, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _parse_gpu(gpu_value):
    if isinstance(gpu_value, int):
        return gpu_value
    if isinstance(gpu_value, (list, tuple)):
        return int(gpu_value[0])
    if isinstance(gpu_value, str):
        s = gpu_value.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        s = s.split(",")[0].strip()
        return int(s)
    return 0


def _count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _build_dataloader(configs, split):
    data_key = f"{split}_data"
    cues_key = f"{split}_cues"

    dataset = Dataset(
        configs["data_type"],
        configs[data_key],
        configs["dataset_args"],
        state=split,
        repeat_dataset=(split == "train"),
        cues_yaml=configs.get(cues_key, None),
    )

    collect_keys = build_collect_keys(
        load_yaml(configs[cues_key]),
        configs["dataset_args"],
        BASE_COLLECT_KEYS,
    )

    dl_args = dict(configs["dataloader_args"])
    if split != "train":
        dl_args["shuffle"] = False
        dl_args["drop_last"] = False

    dataloader = DataLoader(
        dataset,
        collate_fn=lambda batch: tse_collate_fn(batch, collect_keys),
        **dl_args,
    )

    return dataset, dataloader


def _extract_inputs(batch, device):
    mix = batch["wav_mix"].float().to(device)
    target = batch["wav_target"].float().to(device)

    cues = []
    for k in list(AUX_KEY_MAP.values()):
        if k in batch and batch[k] is not None:
            cues.append(batch[k].float().to(device))
    if len(cues) == 0:
        cues = None

    return mix, cues, target


def _compute_confusion_labels(est_wav, target_wav, mix_wav, conf_thresh=1.0):
    est = est_wav.detach().cpu().float().numpy()
    tgt = target_wav.detach().cpu().float().numpy()
    mix = mix_wav.detach().cpu().float().numpy()

    if est.ndim == 3:
        est = est[:, 0, :]
    if tgt.ndim == 3:
        tgt = tgt[:, 0, :]
    if mix.ndim == 3:
        mix = mix[:, 0, :]

    labels = []
    for b in range(est.shape[0]):
        end = min(est[b].shape[-1], tgt[b].shape[-1], mix[b].shape[-1])
        if end < 32:
            labels.append(1)
            continue
        _, sisnri = cal_SISNRi(est[b][:end], tgt[b][:end], mix[b][:end])
        labels.append(1 if float(sisnri) < conf_thresh else 0)

    return torch.tensor(labels, dtype=torch.float32, device=est_wav.device)


def _extract_deployable_attn_features(attn_mat):
    if attn_mat is None:
        return None

    p = attn_mat.clamp_min(1e-9)
    tk = p.shape[-1]

    entropy_q = -(p * p.log()).sum(dim=-1)
    peak_q = p.max(dim=-1).values

    top2 = p.topk(k=min(2, tk), dim=-1).values
    margin_q = top2[..., 0] - top2[..., 1] if top2.shape[-1] == 2 else top2[..., 0]

    concentration_q = (p * p).sum(dim=-1)
    key_usage = p.mean(dim=1)
    key_usage_entropy = -(key_usage * key_usage.clamp_min(1e-9).log()).sum(dim=-1)

    return torch.stack(
        [
            entropy_q.mean(dim=-1),
            entropy_q.std(dim=-1, unbiased=False),
            peak_q.mean(dim=-1),
            peak_q.std(dim=-1, unbiased=False),
            margin_q.mean(dim=-1),
            margin_q.std(dim=-1, unbiased=False),
            concentration_q.mean(dim=-1),
            key_usage_entropy,
        ],
        dim=-1,
    )


def _set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad = flag


def _unfreeze_by_scope(model, scope):
    _set_requires_grad(model, False)

    if scope == "all":
        _set_requires_grad(model, True)
        return

    if hasattr(model, "spk_ft") and hasattr(model.spk_ft, "usef"):
        _set_requires_grad(model.spk_ft.usef, True)

    if scope == "usef_sep":
        if hasattr(model, "sep_model") and hasattr(model.sep_model, "separator"):
            _set_requires_grad(model.sep_model.separator, True)


def _compute_auc_ap(labels, probs):
    try:
        auc = float(roc_auc_score(labels, probs))
    except Exception:
        auc = float("nan")
    try:
        ap = float(average_precision_score(labels, probs))
    except Exception:
        ap = float("nan")
    return auc, ap


def _to_wave_3d(x):
    if x is None:
        return None
    if x.ndim == 2:
        return x.unsqueeze(1)
    if x.ndim == 3:
        return x
    raise RuntimeError(f"Unsupported waveform ndim={x.ndim}, expected 2 or 3")


def _mean_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.mean() if x.ndim > 0 else x
    return x


def _compute_batch_sisnri(est_wav, target_wav, mix_wav):
    est = est_wav.detach().cpu().float().numpy()
    tgt = target_wav.detach().cpu().float().numpy()
    mix = mix_wav.detach().cpu().float().numpy()

    if est.ndim == 3:
        est = est[:, 0, :]
    if tgt.ndim == 3:
        tgt = tgt[:, 0, :]
    if mix.ndim == 3:
        mix = mix[:, 0, :]

    vals = []
    for b in range(est.shape[0]):
        end = min(est[b].shape[-1], tgt[b].shape[-1], mix[b].shape[-1])
        if end < 32:
            continue
        _, sisnri = cal_SISNRi(est[b][:end], tgt[b][:end], mix[b][:end])
        vals.append(float(sisnri))
    return vals


def _run_epoch(
    model,
    risk_head,
    dataloader,
    device,
    usef_att,
    hook_cache,
    criterion,
    sep_criterion,
    lambda_risk,
    lambda_sep,
    conf_thresh,
    epoch_iter,
    train_mode,
    model_optimizer,
    head_optimizer,
    scaler,
    accum_steps,
    use_amp,
    log_interval,
    logger,
    tag,
):
    if train_mode:
        model.train()
        risk_head.train()
    else:
        model.eval()
        risk_head.eval()

    losses = []
    all_labels = []
    all_probs = []
    losses_sep = []
    losses_risk = []
    all_sisnri = []
    processed_steps = 0

    if train_mode:
        if model_optimizer is not None:
            model_optimizer.zero_grad(set_to_none=True)
        head_optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(dataloader, start=1):
        processed_steps = step
        mix, cues, target = _extract_inputs(batch, device)

        grad_ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with grad_ctx:
            amp_ctx = torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda"))
            with amp_ctx:
                out = model(mix) if cues is None else model(mix, cues)
                if isinstance(out, (tuple, list)):
                    out = out[0]

                attn = hook_cache.get("attn", None)
                feat = _extract_deployable_attn_features(attn)
                if feat is None:
                    continue

                bsz = mix.shape[0]
                n_head = int(usef_att.n_head)
                feat = feat.view(bsz, n_head, -1).mean(dim=1)

                labels = _compute_confusion_labels(out, target, mix, conf_thresh=conf_thresh)
                if labels.numel() != feat.shape[0]:
                    continue

                logits = risk_head(feat)
                loss_risk = criterion(logits, labels)
                loss_sep = 0.0
                if sep_criterion is not None and float(lambda_sep) > 0:
                    est_3d = _to_wave_3d(out)
                    tgt_3d = _to_wave_3d(target)
                    loss_sep = _mean_tensor(sep_criterion(est_3d, tgt_3d))
                if not isinstance(loss_sep, torch.Tensor):
                    loss_sep = torch.tensor(0.0, device=logits.device)

                loss_full = float(lambda_risk) * loss_risk + float(lambda_sep) * loss_sep
                loss = loss_full / max(1, int(accum_steps))

            if train_mode:
                scaler.scale(loss).backward()
                if step % int(accum_steps) == 0:
                    if model_optimizer is not None:
                        scaler.step(model_optimizer)
                    scaler.step(head_optimizer)
                    scaler.update()
                    if model_optimizer is not None:
                        model_optimizer.zero_grad(set_to_none=True)
                    head_optimizer.zero_grad(set_to_none=True)

        losses.append(float(loss_full.detach().item()))
        losses_risk.append(float(loss_risk.detach().item()))
        losses_sep.append(float(loss_sep.detach().item()))
        probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
        all_probs.extend(probs)
        all_labels.extend(labels.detach().cpu().numpy().astype(np.int32).tolist())
        all_sisnri.extend(_compute_batch_sisnri(out, target, mix))

        if train_mode and step % int(log_interval) == 0:
            logger.info(
                f"{tag} Step {step}/{epoch_iter} | loss={np.mean(losses):.4f} | "
                f"mem_alloc={torch.cuda.memory_allocated(device)/(1024**3):.2f}GB"
                if device.type == "cuda"
                else f"{tag} Step {step}/{epoch_iter} | loss={np.mean(losses):.4f}"
            )

        if step >= epoch_iter:
            break

    if train_mode and processed_steps > 0 and (processed_steps % int(accum_steps) != 0):
        if model_optimizer is not None:
            scaler.step(model_optimizer)
        scaler.step(head_optimizer)
        scaler.update()
        if model_optimizer is not None:
            model_optimizer.zero_grad(set_to_none=True)
        head_optimizer.zero_grad(set_to_none=True)

    avg_loss = float(np.mean(losses)) if losses else float("nan")
    avg_loss_risk = float(np.mean(losses_risk)) if losses_risk else float("nan")
    avg_loss_sep = float(np.mean(losses_sep)) if losses_sep else float("nan")
    avg_sisnri = float(np.mean(all_sisnri)) if all_sisnri else float("nan")
    auc, ap = _compute_auc_ap(all_labels, all_probs)
    return avg_loss, avg_loss_risk, avg_loss_sep, avg_sisnri, auc, ap


def train(
    config="confs/conf.yaml",
    out_tag="exp20_deployable_risk_head",
    conf_thresh=1.0,
    epochs=8,
    lr=1e-3,
    hidden_dim=64,
    dropout=0.2,
    pos_weight=1.0,
    max_train_steps_per_epoch=0,
    max_val_steps=0,
    log_interval=50,
    freeze_tse=True,
    fine_tune_epochs=0,
    fine_tune_scope="usef",
    phase1_warmup_scope="",
    phase1_warmup_epochs=0,
    ft_lr=3e-5,
    warmstart_head_from_ft=True,
    fine_tune_batch_size=1,
    accum_steps=2,
    use_amp=True,
    empty_cache_interval=100,
    sep_loss_name="SISDR",
    lambda_sep=1.0,
    lambda_risk=0.2,
    **kwargs,
):
    start_time = time.time()
    configs = parse_config_or_kwargs(config, **kwargs)
    set_seed(configs["seed"])

    gpu = _parse_gpu(configs.get("gpus", 0))
    device = torch.device(f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu")

    exp_dir = configs["exp_dir"]
    out_dir = os.path.join(exp_dir, out_tag)
    os.makedirs(out_dir, exist_ok=True)

    logger = get_logger(configs["exp_dir"], "train_exp20_risk_head.log")
    logger.info("Start exp20 deployable risk-head training (2-stage, memory-safe).")
    logger.info(f"Output dir: {out_dir}")

    if "spk_model_init" in configs["model_args"]["tse_model"]:
        configs["model_args"]["tse_model"]["spk_model_init"] = False

    model = get_model(configs["model"]["tse_model"])(configs["model_args"]["tse_model"])
    checkpoint = configs.get("checkpoint", None)
    if checkpoint:
        load_pretrained_model(model, checkpoint)
        logger.info(f"Loaded checkpoint: {checkpoint}")

    model = model.to(device)

    target_module = model.module if hasattr(model, "module") else model
    if not hasattr(target_module, "spk_ft") or not hasattr(target_module.spk_ft, "usef"):
        raise RuntimeError("USEF frontend not enabled; cannot train deployable attention risk head.")

    hook_cache = {"detach_attn": True}
    usef_att = target_module.spk_ft.usef.usef_att
    original_att_forward = usef_att.forward

    def hooked_att_forward(self, batch, aux):
        B, _, old_t, old_q = batch.shape
        aux_t = aux.shape[-2]

        q = self["attn_norm_Q"](self["attn_conv_Q"](batch))
        k = self["attn_norm_K"](self["attn_conv_K"](aux))
        v = self["attn_norm_V"](self["attn_conv_V"](aux))

        q = q.view(-1, *q.shape[2:])
        k = k.view(-1, *k.shape[2:])
        v = v.view(-1, *v.shape[2:])

        q = q.transpose(1, 2).flatten(start_dim=2)
        k = k.transpose(2, 3).contiguous().view([B * self.n_head, -1, aux_t])

        v = v.transpose(1, 2)
        old_shape = v.shape
        v = v.flatten(start_dim=2)

        emb_dim = q.shape[-1]
        attn_mat = torch.matmul(q, k) / (emb_dim**0.5)
        attn_mat = torch.softmax(attn_mat, dim=2)
        hook_cache["attn"] = attn_mat.detach() if hook_cache.get("detach_attn", True) else attn_mat

        v = torch.matmul(attn_mat, v)
        v = v.reshape([old_shape[0], old_t, old_shape[-2], old_shape[-1]])
        v = v.transpose(1, 2)

        emb_dim = v.shape[1]
        batch = v.contiguous().view([B, self.n_head * emb_dim, old_t, old_q])
        return self["attn_concat_proj"](batch)

    usef_att.forward = types.MethodType(hooked_att_forward, usef_att)

    try:
        train_dataset, train_loader = _build_dataloader(configs, "train")
        _, val_loader = _build_dataloader(configs, "val")

        # force micro-batch for phase1 if requested, but keep dataloader parallelism
        if int(fine_tune_batch_size) > 0:
            train_dl_args = dict(configs["dataloader_args"])
            train_dl_args["batch_size"] = int(fine_tune_batch_size)
            val_dl_args = dict(configs["dataloader_args"])
            val_dl_args["batch_size"] = int(fine_tune_batch_size)
            val_dl_args["shuffle"] = False
            val_dl_args["drop_last"] = False

            train_loader = DataLoader(
                train_loader.dataset,
                collate_fn=train_loader.collate_fn,
                **train_dl_args,
            )
            val_loader = DataLoader(
                val_loader.dataset,
                collate_fn=val_loader.collate_fn,
                **val_dl_args,
            )

        batch_size = int(getattr(train_loader, "batch_size", 1) or 1)
        sample_num_per_epoch = int(configs["dataset_args"].get("sample_num_per_epoch", 0))
        if sample_num_per_epoch > 0:
            epoch_iter = max(1, sample_num_per_epoch // batch_size)
        else:
            epoch_iter = max(1, _count_lines(configs["train_samples"]) // batch_size)
        val_iter = max(1, _count_lines(configs["val_samples"]) // batch_size)

        if max_train_steps_per_epoch > 0:
            epoch_iter = min(epoch_iter, int(max_train_steps_per_epoch))
        if max_val_steps > 0:
            val_iter = min(val_iter, int(max_val_steps))

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
        sep_criterion = parse_loss(sep_loss_name)[0].to(device)
        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
        history = []

        logger.info(
            f"epoch_iter={epoch_iter}, val_iter={val_iter}, batch_size={batch_size}, accum_steps={accum_steps}, use_amp={use_amp}"
        )
        logger.info(
            f"fine_tune_epochs={fine_tune_epochs}, scope={fine_tune_scope}, "
            f"warmup_scope={phase1_warmup_scope}, warmup_epochs={phase1_warmup_epochs}, ft_lr={ft_lr}"
        )
        logger.info(
            f"joint loss: sep={sep_loss_name}, lambda_sep={lambda_sep}, lambda_risk={lambda_risk}"
        )
        logger.info(f"head_epochs={epochs}, head_lr={lr}")

        ft_head_state = None
        if int(fine_tune_epochs) > 0:
            warmup_epochs = int(phase1_warmup_epochs)
            main_ft_epochs = int(fine_tune_epochs)
            model_trainable = [p for p in model.parameters() if p.requires_grad]

            phase1_head = DeployableRiskHead(in_dim=8, hidden_dim=hidden_dim, dropout=dropout).to(device)
            hook_cache["detach_attn"] = False

            best_ft_auc = -1.0
            for ep in range(1, warmup_epochs + main_ft_epochs + 1):
                if warmup_epochs > 0 and ep <= warmup_epochs and phase1_warmup_scope:
                    cur_scope = phase1_warmup_scope
                else:
                    cur_scope = fine_tune_scope
                _unfreeze_by_scope(target_module, cur_scope)
                model_trainable = [p for p in model.parameters() if p.requires_grad]
                model_optimizer = torch.optim.Adam(model_trainable, lr=ft_lr)
                head_optimizer = torch.optim.Adam(phase1_head.parameters(), lr=lr)
                logger.info(
                    f"[Phase1] epoch={ep} scope={cur_scope} "
                    f"trainable_backbone_params={sum(p.numel() for p in model_trainable)}"
                )

                if hasattr(train_dataset, "set_epoch"):
                    train_dataset.set_epoch(ep)

                tr_loss, tr_loss_risk, tr_loss_sep, tr_sisnri, tr_auc, tr_ap = _run_epoch(
                    model,
                    phase1_head,
                    train_loader,
                    device,
                    usef_att,
                    hook_cache,
                    criterion,
                    sep_criterion,
                    lambda_risk,
                    lambda_sep,
                    conf_thresh,
                    epoch_iter,
                    True,
                    model_optimizer,
                    head_optimizer,
                    scaler,
                    accum_steps,
                    use_amp,
                    log_interval,
                    logger,
                    f"[Phase1][Train][E{ep}]",
                )

                va_loss, va_loss_risk, va_loss_sep, va_sisnri, va_auc, va_ap = _run_epoch(
                    model,
                    phase1_head,
                    val_loader,
                    device,
                    usef_att,
                    hook_cache,
                    criterion,
                    sep_criterion,
                    lambda_risk,
                    lambda_sep,
                    conf_thresh,
                    val_iter,
                    False,
                    None,
                    None,
                    scaler,
                    accum_steps,
                    use_amp,
                    log_interval,
                    logger,
                    f"[Phase1][Val][E{ep}]",
                )

                logger.info(
                    f"[Phase1] Epoch {ep} | "
                    f"train(total={tr_loss:.4f}, risk={tr_loss_risk:.4f}, sep={tr_loss_sep:.4f}, sisnri={tr_sisnri:.3f}, auc={tr_auc:.4f}, ap={tr_ap:.4f}) "
                    f"| val(total={va_loss:.4f}, risk={va_loss_risk:.4f}, sep={va_loss_sep:.4f}, sisnri={va_sisnri:.3f}, auc={va_auc:.4f}, ap={va_ap:.4f})"
                )
                history.append(
                    {
                        "phase": "phase1",
                        "epoch": ep,
                        "train_loss": tr_loss,
                        "train_risk_loss": tr_loss_risk,
                        "train_sep_loss": tr_loss_sep,
                        "train_sisnri": tr_sisnri,
                        "train_auc": tr_auc,
                        "train_ap": tr_ap,
                        "val_loss": va_loss,
                        "val_risk_loss": va_loss_risk,
                        "val_sep_loss": va_loss_sep,
                        "val_sisnri": va_sisnri,
                        "val_auc": va_auc,
                        "val_ap": va_ap,
                        "phase1_scope": cur_scope,
                    }
                )

                if not np.isnan(va_auc) and va_auc > best_ft_auc:
                    best_ft_auc = va_auc
                    ft_head_state = {k: v.detach().cpu() for k, v in phase1_head.state_dict().items()}
                    torch.save(
                        {
                            "phase": "phase1",
                            "epoch": ep,
                            "val_auc": va_auc,
                            "model_state": model.state_dict(),
                            "head_state": phase1_head.state_dict(),
                        },
                        os.path.join(out_dir, "best_phase1_finetune.pt"),
                    )

                if device.type == "cuda" and int(empty_cache_interval) > 0 and ep % int(empty_cache_interval) == 0:
                    torch.cuda.empty_cache()

            torch.save(model.state_dict(), os.path.join(out_dir, "phase1_finetuned_tse.pt"))
            logger.info("[Phase1] Saved finetuned backbone to phase1_finetuned_tse.pt")

        _set_requires_grad(model, False)
        if freeze_tse:
            model.eval()

        hook_cache["detach_attn"] = True

        risk_head = DeployableRiskHead(in_dim=8, hidden_dim=hidden_dim, dropout=dropout).to(device)
        if warmstart_head_from_ft and ft_head_state is not None:
            risk_head.load_state_dict(ft_head_state, strict=True)
            logger.info("[Phase2] Warm-start risk head from phase1 best head.")

        head_optimizer = torch.optim.Adam(risk_head.parameters(), lr=lr)

        best_auc = -1.0
        best_epoch = -1
        for ep in range(1, int(epochs) + 1):
            if hasattr(train_dataset, "set_epoch"):
                train_dataset.set_epoch(int(fine_tune_epochs) + ep)

            tr_loss, tr_loss_risk, tr_loss_sep, tr_sisnri, tr_auc, tr_ap = _run_epoch(
                model,
                risk_head,
                train_loader,
                device,
                usef_att,
                hook_cache,
                criterion,
                sep_criterion,
                lambda_risk,
                0.0,
                conf_thresh,
                epoch_iter,
                True,
                None,
                head_optimizer,
                scaler,
                accum_steps,
                use_amp,
                log_interval,
                logger,
                f"[Phase2][Train][E{ep}]",
            )

            va_loss, va_loss_risk, va_loss_sep, va_sisnri, va_auc, va_ap = _run_epoch(
                model,
                risk_head,
                val_loader,
                device,
                usef_att,
                hook_cache,
                criterion,
                sep_criterion,
                lambda_risk,
                0.0,
                conf_thresh,
                val_iter,
                False,
                None,
                None,
                scaler,
                accum_steps,
                use_amp,
                log_interval,
                logger,
                f"[Phase2][Val][E{ep}]",
            )

            logger.info(
                f"[Phase2] Epoch {ep} | "
                f"train(total={tr_loss:.4f}, risk={tr_loss_risk:.4f}, sisnri={tr_sisnri:.3f}, auc={tr_auc:.4f}, ap={tr_ap:.4f}) "
                f"| val(total={va_loss:.4f}, risk={va_loss_risk:.4f}, sisnri={va_sisnri:.3f}, auc={va_auc:.4f}, ap={va_ap:.4f})"
            )
            history.append(
                {
                    "phase": "phase2",
                    "epoch": ep,
                    "train_loss": tr_loss,
                    "train_risk_loss": tr_loss_risk,
                    "train_sep_loss": tr_loss_sep,
                    "train_sisnri": tr_sisnri,
                    "train_auc": tr_auc,
                    "train_ap": tr_ap,
                    "val_loss": va_loss,
                    "val_risk_loss": va_loss_risk,
                    "val_sep_loss": va_loss_sep,
                    "val_sisnri": va_sisnri,
                    "val_auc": va_auc,
                    "val_ap": va_ap,
                }
            )

            if not np.isnan(va_auc) and va_auc > best_auc:
                best_auc = va_auc
                best_epoch = ep
                torch.save(
                    {
                        "phase": "phase2",
                        "epoch": ep,
                        "val_auc": va_auc,
                        "risk_head_state": risk_head.state_dict(),
                        "risk_head_args": {
                            "in_dim": 8,
                            "hidden_dim": hidden_dim,
                            "dropout": dropout,
                        },
                        "train_args": {
                            "conf_thresh": conf_thresh,
                            "head_lr": lr,
                            "pos_weight": pos_weight,
                            "checkpoint": checkpoint,
                            "fine_tune_epochs": fine_tune_epochs,
                            "fine_tune_scope": fine_tune_scope,
                            "phase1_warmup_scope": phase1_warmup_scope,
                            "phase1_warmup_epochs": phase1_warmup_epochs,
                            "ft_lr": ft_lr,
                            "warmstart_head_from_ft": warmstart_head_from_ft,
                            "fine_tune_batch_size": fine_tune_batch_size,
                            "accum_steps": accum_steps,
                            "use_amp": use_amp,
                            "sep_loss_name": sep_loss_name,
                            "lambda_sep": lambda_sep,
                            "lambda_risk": lambda_risk,
                        },
                    },
                    os.path.join(out_dir, "best_risk_head.pt"),
                )

            if device.type == "cuda" and int(empty_cache_interval) > 0 and ep % int(empty_cache_interval) == 0:
                torch.cuda.empty_cache()

        with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        summary = {
            "best_val_auc": best_auc,
            "best_phase2_epoch": best_epoch,
            "phase1_epochs": int(fine_tune_epochs),
            "phase2_epochs": int(epochs),
            "time_sec": float(time.time() - start_time),
            "out_dir": out_dir,
        }
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Training done. best_phase2_val_auc={best_auc:.4f} @ epoch {best_epoch}")
        logger.info(f"Total elapsed: {summary['time_sec']:.1f}s")

    finally:
        usef_att.forward = original_att_forward


if __name__ == "__main__":
    fire.Fire(train)
