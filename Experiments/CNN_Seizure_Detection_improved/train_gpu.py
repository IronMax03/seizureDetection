#!/usr/bin/env python3
"""
Seizure detection on CHB-MIT: FiLM-CNN conditioned on per-channel entropy features.
GPU-resident, memory-aware version of Notebook.ipynb.

Pipeline
  1. EEG   : X.npy of each patient is read memory-mapped, converted to uV float16 and
             cached to disk (cache/X_uV_f16_<patients>.npy). Peak RAM ~ one chunk.
  2. ApEn  : utils.APEN on every (window, channel), computed in parallel, cached to disk
             (cache/APEN_<patients>.npy). Computed once, reused on every later run.
  3. Train : whole dataset on the GPU as float16 (~4.8 GB for 410k windows), batches are
             sliced on the GPU (no DataLoader), class-weighted loss, AdamW, grad clipping,
             best-epoch checkpoint by validation AUPRC.
  4. Output: results/<experiment>/  -> summary.json, history.json, summary.png, fold*.pt

Usage
  python train_gpu.py                                   # apen + control, 10-fold window-level CV
  python train_gpu.py --exp apen control denoised
  python train_gpu.py --split patient                   # leave-one-patient-out (no leakage)
  python train_gpu.py --epochs 30 --batch-size 256 --lr 3e-4 --power 0.5

Note: the best epoch is chosen on the same fold it is reported on, which is slightly
optimistic. The last-epoch numbers are reported too.
"""
import argparse
import copy
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CHANNELS = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8-0", "P8-O2",
            "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8-1"]

CHUNK = 20_000          # windows per chunk for disk / GPU transfers (~230 MB float64)
F16_MAX = 65_000.0      # float16 max is 65504; uV values beyond this are clipped


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# =============================================================================
# 1. Data: memory-mapped loading + on-disk caches
# =============================================================================
def patient_paths(data_dir, patients):
    return [(p, Path(data_dir) / p / "X.npy", Path(data_dir) / p / "y.npy") for p in patients]


def _to_uv_f16(block, scale=1e6):
    uv = np.asarray(block, dtype=np.float32) * scale            # volts -> uV BEFORE float16
    n_clip = int(np.count_nonzero(np.abs(uv) > F16_MAX))
    return np.clip(uv, -F16_MAX, F16_MAX).astype(np.float16), n_clip


def build_eeg_cache(data_dir, patients, cache_dir, transform=None, tag="raw"):
    """Write all patients' windows into one uV float16 .npy on disk.
    Returns (X memmap (N,C,T) f16, y (N,) int64, groups (N,) int16 patient index)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = "_".join(patients)
    x_path = cache_dir / f"X_uV_f16_{tag}_{key}.npy"
    y_path = cache_dir / f"y_{key}.npy"
    g_path = cache_dir / f"groups_{key}.npy"

    files = patient_paths(data_dir, patients)

    if not (y_path.exists() and g_path.exists()):
        ys, gs = [], []
        for k, (p, _, yf) in enumerate(files):
            yk = np.load(yf).astype(np.int64)
            ys.append(yk)
            gs.append(np.full(len(yk), k, dtype=np.int16))
        np.save(y_path, np.concatenate(ys))
        np.save(g_path, np.concatenate(gs))

    if not x_path.exists():
        shapes = [np.load(xf, mmap_mode="r").shape for _, xf, _ in files]
        N = sum(s[0] for s in shapes)
        C, T = shapes[0][1:]
        assert all(s[1:] == (C, T) for s in shapes), f"inconsistent window shapes: {shapes}"
        log(f"building EEG cache [{tag}] {N} x {C} x {T} -> {x_path.name} "
            f"({N * C * T * 2 / 1e9:.1f} GB)")
        tmp = x_path.with_name(x_path.stem + ".tmp.npy")
        X = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(N, C, T))
        i, clipped = 0, 0
        for (p, xf, _), shp in zip(files, shapes):
            src = np.load(xf, mmap_mode="r")
            for s in range(0, len(src), CHUNK):
                blk = np.asarray(src[s:s + CHUNK], dtype=np.float64)
                if transform is not None:
                    out = transform(blk)
                    assert out.shape == blk.shape, \
                        f"transform changed shape {blk.shape} -> {out.shape}"
                    blk = out
                X[i + s:i + s + len(blk)], n_clip = _to_uv_f16(blk)
                clipped += n_clip
            i += len(src)
            log(f"  {p}: {len(src)} windows")
        X.flush()
        del X
        os.replace(tmp, x_path)
        if clipped:
            log(f"  warning: {clipped} samples exceeded +/-{F16_MAX:.0f} uV and were clipped")

    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path)
    groups = np.load(g_path)
    assert len(X) == len(y) == len(groups)
    return X, y, groups


# ---- ApEn features: parallel, chunked, cached -------------------------------
_SRC = None


def _apen_init(x_paths):
    global _SRC
    _SRC = [np.load(p, mmap_mode="r") for p in x_paths]


def _apen_job(job):
    from utils import APEN                       # your implementation, unchanged
    k, start, stop = job
    blk = np.asarray(_SRC[k][start:stop], dtype=np.float64)   # same input as the notebook
    out = np.empty(blk.shape[:2], dtype=np.float32)
    for i in range(blk.shape[0]):
        for c in range(blk.shape[1]):
            out[i, c] = APEN(blk[i, c])
    return k, start, out


def build_apen_cache(data_dir, patients, cache_dir, workers):
    cache_dir = Path(cache_dir)
    path = cache_dir / f"APEN_{'_'.join(patients)}.npy"
    if path.exists():
        return np.load(path)

    files = patient_paths(data_dir, patients)
    x_paths = [str(xf) for _, xf, _ in files]
    lens = [np.load(p, mmap_mode="r").shape for p in x_paths]
    offsets = np.concatenate([[0], np.cumsum([s[0] for s in lens])])
    N, C = int(offsets[-1]), lens[0][1]
    feats = np.full((N, C), np.nan, dtype=np.float32)          # 38 MB for 410k x 23

    step = 2_000
    jobs = [(k, s, min(s + step, lens[k][0]))
            for k in range(len(x_paths)) for s in range(0, lens[k][0], step)]
    log(f"computing ApEn for {N} windows x {C} channels on {workers} processes "
        f"(cached afterwards in {path.name})")
    t0, done = time.time(), 0
    with Pool(workers, initializer=_apen_init, initargs=(x_paths,)) as pool:
        for k, start, out in pool.imap_unordered(_apen_job, jobs):
            feats[offsets[k] + start: offsets[k] + start + len(out)] = out
            done += len(out)
            if done % (step * 20) < step or done == N:
                el = time.time() - t0
                log(f"  ApEn {done}/{N} ({100 * done / N:.0f}%) "
                    f"| eta {el / done * (N - done) / 60:.1f} min")
    bad = ~np.isfinite(feats)
    if bad.any():
        log(f"  {bad.sum()} non-finite ApEn values -> replaced by column median")
        med = np.nanmedian(np.where(bad, np.nan, feats), axis=0)
        feats[bad] = np.take(med, np.where(bad)[1])
    np.save(path, feats)
    return feats


def _atar_job(job):
    from denois import atar                      # your implementation, unchanged (1-D input)
    k, start, stop = job
    # ATAR's thresholds (k1=10, k2=100) are absolute amplitudes meant for uV:
    # on volts (~1e-5) nothing would ever exceed them and ATAR would do nothing.
    blk = np.asarray(_SRC[k][start:stop], dtype=np.float64) * 1e6  # uV, (n, C, T)
    out = np.empty_like(blk)
    for i in range(blk.shape[0]):
        for c in range(blk.shape[1]):
            r = np.asarray(atar(blk[i, c]), dtype=np.float64)
            if r.shape != blk[i, c].shape:
                raise ValueError(f"atar returned shape {r.shape} for input {blk[i, c].shape}")
            out[i, c] = r
    uv, n_clip = _to_uv_f16(out, scale=1.0)                      # already uV
    return k, start, uv, n_clip


def build_denoised_cache(data_dir, patients, cache_dir, workers):
    """ATAR-denoise every (window, channel) in parallel; cache as uV float16 on disk."""
    cache_dir = Path(cache_dir)
    key = "_".join(patients)
    path = cache_dir / f"X_uV_f16_atar_{key}.npy"
    if not path.exists():
        files = patient_paths(data_dir, patients)
        x_paths = [str(xf) for _, xf, _ in files]
        shapes = [np.load(p, mmap_mode="r").shape for p in x_paths]
        offsets = np.concatenate([[0], np.cumsum([s[0] for s in shapes])])
        N, (C, T) = int(offsets[-1]), shapes[0][1:]
        log(f"ATAR-denoising {N} windows x {C} channels on {workers} processes "
            f"-> {path.name} ({N * C * T * 2 / 1e9:.1f} GB, cached afterwards)")
        tmp = path.with_name(path.stem + ".tmp.npy")
        X = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(N, C, T))
        step = 1_000
        jobs = [(k, s, min(s + step, shapes[k][0]))
                for k in range(len(x_paths)) for s in range(0, shapes[k][0], step)]
        t0, done, clipped = time.time(), 0, 0
        with Pool(workers, initializer=_apen_init, initargs=(x_paths,)) as pool:
            for k, start, uv, n_clip in pool.imap_unordered(_atar_job, jobs):
                X[offsets[k] + start: offsets[k] + start + len(uv)] = uv
                done += len(uv)
                clipped += n_clip
                if done % (step * 40) < step or done == N:
                    el = time.time() - t0
                    log(f"  ATAR {done}/{N} ({100 * done / N:.0f}%) "
                        f"| eta {el / done * (N - done) / 60:.1f} min")
        X.flush()
        del X
        os.replace(tmp, path)
        if clipped:
            log(f"  warning: {clipped} samples exceeded +/-{F16_MAX:.0f} uV and were clipped")
    return np.load(path, mmap_mode="r")


# =============================================================================
# 2. Model
# =============================================================================
class FiLMBlock(nn.Module):
    """Conv -> BatchNorm -> LeakyReLU -> FiLM (gamma*h + beta) -> MaxPool."""
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=1)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.LeakyReLU(0.01)
        self.pool = nn.MaxPool1d(2, stride=2)
        self.out_ch = out_ch

    def forward(self, x, gamma, beta):
        h = self.act(self.bn(self.conv(x)))
        h = gamma.unsqueeze(-1) * h + beta.unsqueeze(-1)
        return self.pool(h)


class CNN(nn.Module):
    def __init__(self, extra_features=0, n_classes=2, n_eeg_ch=23, length=256):
        super().__init__()
        self.channels = [18, 15, 10, 10, 10]
        kernels = [6, 5, 4, 4, 4]
        in_chs = [n_eeg_ch] + self.channels[:-1]
        self.blocks = nn.ModuleList(
            [FiLMBlock(ic, oc, k) for ic, oc, k in zip(in_chs, self.channels, kernels)])

        self.no_features = extra_features == 0
        gen_in = extra_features if extra_features > 0 else 1
        self.film_generator = nn.Sequential(
            nn.Linear(gen_in, 32), nn.LeakyReLU(0.01),
            nn.Linear(32, 2 * sum(self.channels)),
        )
        nn.init.zeros_(self.film_generator[-1].weight)      # identity FiLM at init
        nn.init.zeros_(self.film_generator[-1].bias)

        L = length
        for k in kernels:
            L = (L - k + 1) // 2
        assert L > 0, f"window length {length} too short for this network"
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(self.channels[-1] * L, 50), nn.LeakyReLU(0.01),   # 40 for L=256
            nn.Linear(50, 20), nn.LeakyReLU(0.01),
            nn.Linear(20, n_classes),
        )

    def forward(self, x_serie, x_features=None):
        if self.no_features:
            cond = torch.ones(x_serie.size(0), 1, device=x_serie.device, dtype=x_serie.dtype)
        else:
            cond = x_features
        film = self.film_generator(cond)
        idx, z = 0, x_serie
        for blk in self.blocks:
            c = blk.out_ch
            gamma = 1.0 + film[:, idx:idx + c]; idx += c
            beta = film[:, idx:idx + c];        idx += c
            z = blk(z, gamma, beta)
        return self.classifier(self.flatten(z))


# =============================================================================
# 3. Training
# =============================================================================
def pick_device():
    if not torch.cuda.is_available():
        log("CUDA not available -> running on CPU (slow)")
        return torch.device("cpu")
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    archs = torch.cuda.get_arch_list()
    # sm_XY binaries run on any GPU of the same major version with minor >= Y
    # (e.g. sm_60 kernels run on a sm_61 Titan Xp); compute_XY (PTX) can be JIT-compiled.
    compatible = any(
        (a.startswith("sm_") and int(a[3:-1]) == major and int(a[-1]) <= minor) or
        (a.startswith("compute_") and int(a[8:]) <= major * 10 + minor)
        for a in archs if a[-1].isdigit())
    try:                                             # the real test: run a kernel
        torch.nn.Conv1d(23, 4, 3).cuda()(torch.randn(2, 23, 16, device="cuda"))
        torch.cuda.synchronize()
        ran = True
    except Exception as e:                           # noqa: BLE001
        ran, err = False, e
    if not ran:
        raise SystemExit(
            f"PyTorch {torch.__version__} cannot run kernels on {torch.cuda.get_device_name(0)} "
            f"({arch}).\n  built for: {archs}\n  error: {err}")
    if not compatible:
        log(f"warning: {arch} not in {archs}, but a test kernel ran; continuing")
    free, total = torch.cuda.mem_get_info()
    log(f"GPU: {torch.cuda.get_device_name(0)} ({arch}) | "
        f"free {free / 2**30:.1f} / {total / 2**30:.1f} GB")
    torch.backends.cudnn.benchmark = True
    return torch.device("cuda")


def to_device_chunked(X, device):
    """Copy an (N, C, T) float16 memmap to the device chunk by chunk."""
    need = X.size * 2
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info()
        if need > free - 1.5 * 2**30:
            raise SystemExit(f"dataset needs {need / 2**30:.1f} GB but only "
                             f"{free / 2**30:.1f} GB VRAM is free; use fewer patients")
    out = torch.empty(X.shape, dtype=torch.float16, device=device)
    for s in range(0, len(X), CHUNK):
        out[s:s + CHUNK] = torch.from_numpy(np.array(X[s:s + CHUNK])).to(device)
    return out


@torch.no_grad()
def evaluate(model, x_dev, f_all, y_dev, idx, class_w, pos_class, bs):
    model.eval()
    loss_sum = torch.zeros((), device=x_dev.device)
    w_sum = torch.zeros((), device=x_dev.device)
    outs = []
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        out = model(x_dev[b].float(), None if f_all is None else f_all[b])
        yb = y_dev[b]
        loss_sum += F.cross_entropy(out, yb, weight=class_w, reduction="sum")
        w_sum += class_w[yb].sum()
        outs.append(out)
    out = torch.cat(outs)
    preds = out.argmax(1).cpu().numpy()
    probs = out.softmax(1)[:, pos_class].float().cpu().numpy()
    labels = y_dev[idx].cpu().numpy()
    return (loss_sum / w_sum).item(), preds, labels, probs


def fold_metrics(labels, preds, probs, pos_class):
    has_pos = bool((labels == pos_class).any())
    return {
        "acc": float((preds == labels).mean()),
        "bacc": float(balanced_accuracy_score(labels, preds)),
        "auprc": float(average_precision_score(labels == pos_class, probs)) if has_pos else float("nan"),
    }


def make_splits(y, groups, split, n_splits, seed):
    if split == "patient":
        return list(LeaveOneGroupOut().split(np.zeros(len(y)), y, groups))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def train_cv(name, x_dev, feats, y, groups, args, device, out_dir, class_names):
    """Cross-validated training. feats=None -> no conditioning (control)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_dev = torch.as_tensor(y, dtype=torch.long, device=device)
    f_dev = None if feats is None else torch.as_tensor(feats, dtype=torch.float32, device=device)
    n_feat = 0 if feats is None else feats.shape[1]
    K = int(y.max()) + 1
    pos = K - 1
    splits = make_splits(y, groups, args.split, args.folds, args.seed)
    n_folds = len(splits)

    histories, best_rows, last_rows = [], [], []
    cm_best = np.zeros((K, K), dtype=np.int64)

    for fold, (tr, va) in enumerate(splits):
        t_fold = time.time()
        tr_t = torch.as_tensor(tr, device=device)
        va_t = torch.as_tensor(va, device=device)

        counts = torch.bincount(y_dev[tr_t], minlength=K).float()
        class_w = (counts.sum() / (K * counts.clamp(min=1))) ** args.power

        if f_dev is not None:
            mu = f_dev[tr_t].mean(0, keepdim=True)
            sd = f_dev[tr_t].std(0, keepdim=True) + 1e-8
            f_all = (f_dev - mu) / sd
        else:
            f_all = None

        torch.manual_seed(args.seed + fold)           # same init across experiments
        model = CNN(extra_features=n_feat, n_classes=K, n_eeg_ch=x_dev.shape[1],
                    length=x_dev.shape[2]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        log(f"[{name}] fold {fold + 1}/{n_folds} | train {len(tr)} (pos {int(counts[pos])}) "
            f"| val {len(va)} | class w {[round(float(w), 2) for w in class_w]}")

        hist = {k: [] for k in ("train_loss", "val_loss", "val_acc", "val_bacc", "val_auprc")}
        best = {"auprc": -1.0, "epoch": -1}
        bs = args.batch_size

        for epoch in range(args.epochs):
            t_ep = time.time()
            model.train()
            perm = tr_t[torch.randperm(len(tr_t), device=device)]
            n_batches = len(perm) // bs                      # drop_last: BatchNorm-safe
            run = torch.zeros((), device=device)
            for b in range(n_batches):
                idx = perm[b * bs:(b + 1) * bs]
                xs = x_dev[idx].float()
                xf = None if f_all is None else f_all[idx]
                loss = F.cross_entropy(model(xs, xf), y_dev[idx], weight=class_w)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
                run += loss.detach()                         # no per-step GPU sync
            hist["train_loss"].append((run / max(n_batches, 1)).item())

            vloss, preds, labels, probs = evaluate(model, x_dev, f_all, y_dev, va_t,
                                                   class_w, pos, args.eval_batch_size)
            m = fold_metrics(labels, preds, probs, pos)
            hist["val_loss"].append(vloss)
            hist["val_acc"].append(m["acc"])
            hist["val_bacc"].append(m["bacc"])
            hist["val_auprc"].append(m["auprc"])

            score = m["auprc"] if np.isfinite(m["auprc"]) else m["bacc"]
            if score > best["auprc"]:
                best = {"auprc": score, "epoch": epoch, "preds": preds, "labels": labels,
                        "state": copy.deepcopy(model.state_dict())}

            log(f"    epoch {epoch:2d} | train {hist['train_loss'][-1]:.4f} | val {vloss:.4f} "
                f"| acc {m['acc']:.3f} | bal acc {m['bacc']:.3f} | AUPRC {m['auprc']:.3f} "
                f"| {time.time() - t_ep:.1f}s")

            if args.patience and epoch - best["epoch"] >= args.patience:
                log(f"    early stop (no AUPRC gain for {args.patience} epochs)")
                break

        torch.save(best["state"], out_dir / f"fold{fold + 1}.pt")
        cm_best += confusion_matrix(best["labels"], best["preds"], labels=list(range(K)))
        e = best["epoch"]
        best_rows.append({"fold": fold + 1, "epoch": e, "bacc": hist["val_bacc"][e],
                          "auprc": hist["val_auprc"][e]})
        last_rows.append({"fold": fold + 1, "bacc": hist["val_bacc"][-1],
                          "auprc": hist["val_auprc"][-1]})
        histories.append(hist)
        log(f"[{name}] fold {fold + 1}/{n_folds} done in {(time.time() - t_fold) / 60:.1f} min "
            f"| best epoch {e}: bal acc {hist['val_bacc'][e]:.4f}, AUPRC {hist['val_auprc'][e]:.4f}")

        del model, opt, best
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = summarize(name, best_rows, last_rows, cm_best, pos, args)
    (out_dir / "history.json").write_text(json.dumps(histories))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot(name, histories, cm_best, class_names, pos, out_dir / "summary.png")
    return summary


def summarize(name, best_rows, last_rows, cm, pos, args):
    def ms(rows, k):
        v = np.array([r[k] for r in rows], dtype=float)
        return float(np.nanmean(v)), float(np.nanstd(v))

    tn, fp = int(cm[0, 0]), int(cm[0, 1:].sum())
    fn, tp = int(cm[pos, :pos].sum()), int(cm[pos, pos])
    win_sec = args.window_len / args.fs
    hours = cm.sum() * win_sec / 3600
    s = {
        "experiment": name,
        "best_epoch": {"bacc": ms(best_rows, "bacc"), "auprc": ms(best_rows, "auprc")},
        "last_epoch": {"bacc": ms(last_rows, "bacc"), "auprc": ms(last_rows, "auprc")},
        "pooled_best": {
            "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "precision": tp / max(tp + fp, 1),
            "fp_per_hour": fp / hours if hours else float("nan"),
            "confusion_matrix": cm.tolist(),
        },
        "folds_best": best_rows,
        "config": vars(args),
    }
    b, l, p = s["best_epoch"], s["last_epoch"], s["pooled_best"]
    log(f"[{name}] CV ({len(best_rows)} folds)\n"
        f"    best epoch : bal acc {b['bacc'][0]:.4f} +/- {b['bacc'][1]:.4f} "
        f"| AUPRC {b['auprc'][0]:.4f} +/- {b['auprc'][1]:.4f}\n"
        f"    last epoch : bal acc {l['bacc'][0]:.4f} +/- {l['bacc'][1]:.4f} "
        f"| AUPRC {l['auprc'][0]:.4f} +/- {l['auprc'][1]:.4f}\n"
        f"    pooled     : sens {p['sensitivity']:.3f} | spec {p['specificity']:.4f} "
        f"| precision {p['precision']:.3f} | FP/h {p['fp_per_hour']:.1f} "
        f"(assumes non-overlapping {win_sec:g}s windows)")
    return s


def plot(name, histories, cm, class_names, pos, path):
    max_ep = max(len(h["train_loss"]) for h in histories)

    def stack(key):  # pad early-stopped folds with NaN
        return np.array([h[key] + [np.nan] * (max_ep - len(h[key])) for h in histories], float)

    ep = np.arange(1, max_ep + 1)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))

    def band(a, axis, label, color):
        m, s = np.nanmean(a, 0), np.nanstd(a, 0)
        axis.plot(ep, m, label=label, color=color)
        axis.fill_between(ep, m - s, m + s, alpha=0.2, color=color)

    band(stack("train_loss"), ax[0], "train loss", "C0")
    band(stack("val_loss"), ax[0], "val loss", "C1")
    ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].set_title(f"{name}: weighted loss")
    band(stack("val_bacc"), ax[1], "val balanced acc", "C2")
    band(stack("val_auprc"), ax[1], f"val AUPRC ({class_names[pos]})", "C3")
    ax[1].set_xlabel("epoch"); ax[1].set_ylim(0, 1); ax[1].legend()
    ax[1].set_title("validation metrics (mean +/- std over folds)")

    K = len(class_names)
    im = ax[2].imshow(cm, cmap="Blues")
    ax[2].set_title("confusion matrix (all folds, best epoch)")
    ax[2].set_xlabel("predicted"); ax[2].set_ylabel("true")
    ax[2].set_xticks(range(K)); ax[2].set_xticklabels(class_names, rotation=45, ha="right")
    ax[2].set_yticks(range(K)); ax[2].set_yticklabels(class_names)
    for i in range(K):
        for j in range(K):
            ax[2].text(j, i, cm[i, j], ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# =============================================================================
# 4. Main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="../CHB-MIT_Formating/npy_data")
    p.add_argument("--patients", nargs="+", default=None,
                   help="e.g. chb01 chb02; default: every folder in --data-dir with X.npy and y.npy")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--exp", nargs="+", default=["apen", "control"],
                   choices=["apen", "control", "denoised"])
    p.add_argument("--split", default="window", choices=["window", "patient"],
                   help="window: StratifiedKFold over windows (as in the notebook, optimistic); "
                        "patient: leave-one-patient-out")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=1.0, help="gradient-norm clipping")
    p.add_argument("--power", type=float, default=0.5,
                   help="class-weight exponent: 1 = inverse frequency, 0.5 = sqrt, 0 = none")
    p.add_argument("--patience", type=int, default=0, help="early stopping on AUPRC (0 = off)")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="processes for ApEn")
    p.add_argument("--fs", type=float, default=256.0, help="sampling rate, for FP/h")
    p.add_argument("--class-names", nargs="+", default=["normal", "seizure"])
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    if args.patients is None:
        args.patients = sorted(d.name for d in data_dir.iterdir()
                               if (d / "X.npy").exists() and (d / "y.npy").exists()) \
            if data_dir.is_dir() else []
    missing = [p for p in args.patients
               if not ((data_dir / p / "X.npy").exists() and (data_dir / p / "y.npy").exists())]
    if missing or not args.patients:
        raise SystemExit(f"no X.npy/y.npy for {missing or 'any patient'} in {data_dir.resolve()}")
    log(f"patients: {args.patients}")
    device = pick_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, y, groups = build_eeg_cache(args.data_dir, args.patients, args.cache_dir)
    args.window_len = int(X.shape[2])
    log(f"data: X {X.shape} float16 uV | seizure windows {int((y == y.max()).sum())} "
        f"({100 * (y == y.max()).mean():.2f}%) | patients {args.patients}")
    if len(args.class_names) != int(y.max()) + 1:
        args.class_names = [str(i) for i in range(int(y.max()) + 1)]

    apen = None
    if "apen" in args.exp:
        apen = build_apen_cache(args.data_dir, args.patients, args.cache_dir, args.workers)

    results = []
    x_dev = None
    for exp in args.exp:
        if exp == "denoised":
            x_dev = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
            Xd = build_denoised_cache(args.data_dir, args.patients, args.cache_dir, args.workers)
            assert Xd.shape == X.shape
            x_exp = to_device_chunked(Xd, device)
            feats = apen if apen is not None else build_apen_cache(
                args.data_dir, args.patients, args.cache_dir, args.workers)
        else:
            if x_dev is None:
                x_dev = to_device_chunked(X, device)
            x_exp = x_dev
            feats = apen if exp == "apen" else None

        if device.type == "cuda":
            log(f"VRAM in use: {torch.cuda.memory_allocated() / 2**30:.1f} GB")
        results.append(train_cv(exp, x_exp, feats, y, groups, args, device,
                                Path(args.out_dir) / exp, args.class_names))
        if exp == "denoised":
            del x_exp
            if device.type == "cuda":
                torch.cuda.empty_cache()

    log("=" * 78)
    log(f"{'experiment':<10} | {'bal acc (best)':>16} | {'AUPRC (best)':>16} | "
        f"{'sens':>5} | {'spec':>6} | {'FP/h':>6}")
    for r in results:
        b, p = r["best_epoch"], r["pooled_best"]
        log(f"{r['experiment']:<10} | {b['bacc'][0]:.3f} +/- {b['bacc'][1]:.3f} | "
            f"{b['auprc'][0]:.3f} +/- {b['auprc'][1]:.3f} | {p['sensitivity']:.3f} | "
            f"{p['specificity']:.4f} | {p['fp_per_hour']:6.1f}")


if __name__ == "__main__":
    main()