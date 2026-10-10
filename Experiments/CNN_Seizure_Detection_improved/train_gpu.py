#!/usr/bin/env python3
"""
Seizure detection on CHB-MIT – experiment-driven training script.

Pipeline
  0. Check  : every X.npy validated against its header (truncated files = clear error).
  1. EEG    : X.npy per patient -> float16 uV cache (cache/raw/<p>.npy or cache/atar/<p>.npy).
  2. ApEn   : approximate entropy per (window, channel), cached per patient (cache/apen/<p>.npy).
  3. Tune   : optional quick random search over lr/wd/batch_size (--tune flag).
  4. Train  : patient-stratified k-fold CV.  Data lives on GPU when it fits, else streamed from
              pinned RAM.  Class-weighted loss, AdamW, grad clipping, best-epoch checkpoint.
  5. Output : results/<exp_name>/  -> summary.json, history.json, summary.png, fold*.pt

Config system (each layer overrides the previous):
  configs/default_training_config.yaml   <- base hyperparams
  configs/experiments/<name>.yaml        <- arch + data choices + optional hp overrides + tune block
  CLI flags                              <- one-off overrides

Usage
  python train_gpu.py --exp ficnn_apen
  python train_gpu.py --exp ficnn_apen --tune
  python train_gpu.py --exp ficnn_apen --lr 1e-3 --epochs 50
  python train_gpu.py --list-experiments

Adding a new model
  1. Define it in Architectures.py with signature (extra_features, n_classes).
  2. Add one line to MODEL_REGISTRY below.
  3. Create configs/experiments/<your_exp>.yaml pointing to it.
"""
import argparse
import copy
import json
import os
import random
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Model registry – add new architectures here (one line each).
# ---------------------------------------------------------------------------
from Architectures import FiCNN, FiCRNN  # noqa: E402

MODEL_REGISTRY = {
    "ficnn":  FiCNN,
    "ficrnn": FiCRNN,
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = Path(__file__).parent
_CONFIG_DIR  = _SCRIPT_DIR / "configs"
_EXP_DIR     = _CONFIG_DIR / "experiments"
_DEFAULT_CFG = _CONFIG_DIR / "default_training_config.yaml"

# ---------------------------------------------------------------------------
# Tune search-space defaults (used when the experiment yaml has no tune block)
# ---------------------------------------------------------------------------
_DEFAULT_TUNE = {
    "n_trials": 8,
    "n_folds":  1,
    "n_epochs": 10,
    "search": {
        "lr":           [1e-4, 5e-4, 1e-3, 3e-3],
        "weight_decay": [0.0,  1e-4, 1e-3],
        "batch_size":   [128,  256,  512],
    },
}

CHUNK   = 20_000
F16_MAX = 65_000.0


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# =============================================================================
# 1. Data helpers
# =============================================================================
def npy_info(path):
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        readers = {(1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0}
        if version in readers:
            shape, _, dtype = readers[version](f)
        else:
            shape, _, dtype = np.lib.format._read_array_header(f, version)  # noqa: SLF001
        offset = f.tell()
    expected = offset + int(np.prod(shape)) * dtype.itemsize
    return shape, dtype, expected, os.path.getsize(path)


def check_input_npy(path):
    shape, dtype, expected, actual = npy_info(path)
    if dtype.hasobject:
        raise SystemExit(f"{path}: object array, cannot be memory-mapped")
    if actual < expected:
        raise SystemExit(
            f"{path} is truncated: {actual / 1e9:.2f} GB on disk but header describes "
            f"{expected / 1e9:.2f} GB (shape {shape}, {dtype}).\nRe-copy or re-export.")
    return shape


def cache_ok(cache, src):
    if not cache.exists() or cache.stat().st_mtime < Path(src).stat().st_mtime:
        return False
    _, _, expected, actual = npy_info(cache)
    return actual >= expected


def ensure_space(directory, need_bytes, what):
    free = shutil.disk_usage(directory).free
    if need_bytes + 2 * 2**30 > free:
        raise SystemExit(
            f"not enough disk space for {what}: needs {need_bytes / 2**30:.1f} GB, "
            f"only {free / 2**30:.1f} GB free in {Path(directory).resolve()}.")


def _to_uv_f16(block, scale=1e6):
    uv = np.asarray(block, dtype=np.float32) * scale
    bad = ~np.isfinite(uv)
    if bad.any():
        uv[bad] = 0.0
    n_clip = int(np.count_nonzero(np.abs(uv) > F16_MAX)) + int(bad.sum())
    return np.clip(uv, -F16_MAX, F16_MAX).astype(np.float16), n_clip


_SRC = None


def _worker_init(x_paths):
    global _SRC
    _SRC = [np.load(p, mmap_mode="r") for p in x_paths]


def _apen_job(job):
    from utils import APEN
    k, start, stop = job
    blk = np.asarray(_SRC[k][start:stop], dtype=np.float64)
    out = np.empty(blk.shape[:2], dtype=np.float32)
    for i in range(blk.shape[0]):
        for c in range(blk.shape[1]):
            out[i, c] = APEN(blk[i, c])
    return k, start, out, 0


def _atar_job(job):
    from denois import atar
    k, start, stop = job
    blk = np.asarray(_SRC[k][start:stop], dtype=np.float64) * 1e6
    out = np.empty_like(blk)
    for i in range(blk.shape[0]):
        for c in range(blk.shape[1]):
            r = np.asarray(atar(blk[i, c]), dtype=np.float64)
            if r.shape != blk[i, c].shape:
                raise ValueError(f"atar returned shape {r.shape} for input {blk[i, c].shape}")
            out[i, c] = r
    uv, n_clip = _to_uv_f16(out, scale=1.0)
    return k, start, uv, n_clip


def build_patient_cache(kind, patient, src, cache_dir, workers):
    out_dir = Path(cache_dir) / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{patient}.npy"
    if cache_ok(path, src):
        return np.load(path, mmap_mode="r")

    N, C, T = check_input_npy(src)
    dtype, shape = (np.float32, (N, C)) if kind == "apen" else (np.float16, (N, C, T))
    need = int(np.prod(shape)) * np.dtype(dtype).itemsize
    ensure_space(out_dir, need, f"the {kind} cache of {patient}")
    log(f"building {kind} cache for {patient}: {N} windows -> {path} ({need / 2**30:.2f} GB)")

    tmp = path.with_name(path.stem + ".tmp.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype, shape=shape)
    t0, clipped = time.time(), 0
    if kind == "raw":
        src_mm = np.load(src, mmap_mode="r")
        for s in range(0, N, CHUNK):
            out[s:s + CHUNK], n = _to_uv_f16(src_mm[s:s + CHUNK])
            clipped += n
    else:
        job_fn, step = (_apen_job, 2_000) if kind == "apen" else (_atar_job, 1_000)
        jobs = [(0, s, min(s + step, N)) for s in range(0, N, step)]
        done, next_log = 0, 0.1
        with Pool(workers, initializer=_worker_init, initargs=([str(src)],)) as pool:
            for _, start, arr, n in pool.imap_unordered(job_fn, jobs):
                out[start:start + len(arr)] = arr
                clipped += n
                done += len(arr)
                if done / N >= next_log or done == N:
                    el = time.time() - t0
                    log(f"  {kind} {patient}: {100 * done / N:.0f}% "
                        f"| eta {el / done * (N - done) / 60:.1f} min")
                    next_log += 0.1
        if kind == "apen":
            bad = ~np.isfinite(out)
            if bad.any():
                log(f"  {int(bad.sum())} non-finite ApEn values -> replaced by channel median")
                med = np.nanmedian(np.where(bad, np.nan, out), axis=0)
                out[bad] = np.take(med, np.where(bad)[1])
    out.flush()
    del out
    os.replace(tmp, path)
    if clipped:
        log(f"  warning: {clipped} samples were NaN/inf or exceeded +/-{F16_MAX:.0f} uV (clipped)")
    log(f"  {patient} {kind} done in {(time.time() - t0) / 60:.1f} min")
    return np.load(path, mmap_mode="r")


def load_labels(data_dir, patients, n_windows):
    ys, gs = [], []
    for k, (p, n) in enumerate(zip(patients, n_windows)):
        yk = np.load(Path(data_dir) / p / "y.npy").astype(np.int64).ravel()
        if len(yk) != n:
            raise SystemExit(f"{p}: y.npy has {len(yk)} labels but X.npy has {n} windows")
        ys.append(yk)
        gs.append(np.full(n, k, dtype=np.int16))
    return np.concatenate(ys), np.concatenate(gs)


def ram_available():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


# =============================================================================
# 2. WindowStore
# =============================================================================
class WindowStore:
    """All windows as float16 on GPU (fastest) or in pinned CPU RAM (fallback)."""

    def __init__(self, parts, device, name, mode="auto", max_batch=4096):
        N = sum(len(p) for p in parts)
        C, T = parts[0].shape[1:]
        self.shape, self.device = (N, C, T), device
        need = N * C * T * 2
        cuda = device.type == "cuda"
        if not cuda or mode == "ram":
            self.on_gpu = False
        elif mode == "gpu":
            self.on_gpu = True
        else:
            free, _ = torch.cuda.mem_get_info()
            self.on_gpu = need < free - 1.5 * 2**30
            if not self.on_gpu:
                log(f"{name}: {need / 2**30:.1f} GB does not fit in VRAM -> kept in RAM")
        if not self.on_gpu:
            avail = ram_available()
            if avail is not None and need > avail - 2 * 2**30:
                raise SystemExit(f"{name}: needs {need / 2**30:.1f} GB but only "
                                 f"{avail / 2**30:.1f} GB RAM available; use fewer patients")
        self.streaming = cuda and not self.on_gpu
        self.index_device = device if self.on_gpu else torch.device("cpu")

        pin = False
        if self.on_gpu:
            self.x = torch.empty(self.shape, dtype=torch.float16, device=device)
        else:
            if self.streaming:
                try:
                    self.x = torch.empty(self.shape, dtype=torch.float16, pin_memory=True)
                    pin = True
                except RuntimeError:
                    log(f"{name}: could not pin {need / 2**30:.1f} GB, using pageable RAM")
            if not pin:
                self.x = torch.empty(self.shape, dtype=torch.float16)
        i, n_bad = 0, 0
        for p in parts:
            for s in range(0, len(p), CHUNK):
                arr = np.array(p[s:s + CHUNK])
                bad = ~np.isfinite(arr)
                if bad.any():
                    n_bad += int(bad.sum())
                    arr[bad] = 0
                self.x[i + s:i + s + len(arr)] = torch.from_numpy(arr).to(self.x.device)
            i += len(p)
        if n_bad:
            log(f"{name}: warning: {n_bad} NaN/inf samples set to 0")

        if self.streaming:
            self.buf = [torch.empty((max_batch, C, T), dtype=torch.float16, pin_memory=True)
                        for _ in range(2)]
            self.events = [None, None]
            self.k = 0
        log(f"{name}: {N} windows ({need / 2**30:.1f} GB) in "
            f"{'VRAM' if self.on_gpu else ('pinned RAM, streamed' if pin else 'RAM')}")

    def get(self, idx):
        if not self.streaming:
            return self.x[idx].float()
        n = len(idx)
        if n > self.buf[0].shape[0]:
            self.buf = [torch.empty((n,) + self.shape[1:], dtype=torch.float16, pin_memory=True)
                        for _ in range(2)]
            self.events = [None, None]
        k, self.k = self.k, self.k ^ 1
        if self.events[k] is not None:
            self.events[k].synchronize()
        buf = self.buf[k][:n]
        torch.index_select(self.x, 0, idx, out=buf)
        out = buf.to(self.device, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self.events[k] = ev
        return out.float()


# =============================================================================
# 3. Training core
# =============================================================================
def pick_device():
    if not torch.cuda.is_available():
        log("CUDA not available -> running on CPU (slow)")
        return torch.device("cpu")
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    archs = torch.cuda.get_arch_list()
    compatible = any(
        (a.startswith("sm_") and int(a[3:-1]) == major and int(a[-1]) <= minor) or
        (a.startswith("compute_") and int(a[8:]) <= major * 10 + minor)
        for a in archs if a[-1].isdigit())
    try:
        torch.nn.Conv1d(23, 4, 3).cuda()(torch.randn(2, 23, 16, device="cuda"))
        torch.cuda.synchronize()
        ran = True
    except Exception as e:  # noqa: BLE001
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


@torch.no_grad()
def evaluate(model, store, f_all, y_dev, idx, class_w, pos_class, bs):
    model.eval()
    loss_sum = torch.zeros((), device=y_dev.device)
    w_sum    = torch.zeros((), device=y_dev.device)
    outs = []
    dev = y_dev.device
    for s in range(0, len(idx), bs):
        b  = idx[s:s + bs]
        bd = b.to(dev, non_blocking=True)
        out = model(store.get(b), None if f_all is None else f_all[bd])
        yb = y_dev[bd]
        loss_sum += F.cross_entropy(out, yb, weight=class_w, reduction="sum")
        w_sum    += class_w[yb].sum()
        outs.append(out)
    out    = torch.cat(outs)
    preds  = out.argmax(1).cpu().numpy()
    probs  = out.softmax(1)[:, pos_class].float().cpu().numpy()
    labels = y_dev[idx.to(dev)].cpu().numpy()
    return (loss_sum / w_sum).item(), preds, labels, probs


def fold_metrics(labels, preds, probs, pos_class):
    if not np.isfinite(probs).all():
        log(f"    warning: {int((~np.isfinite(probs)).sum())} non-finite predictions -> NaN metrics")
        return {"acc": float("nan"), "bacc": float("nan"), "auprc": float("nan")}
    has_pos = bool((labels == pos_class).any())
    return {
        "acc":   float((preds == labels).mean()),
        "bacc":  float(balanced_accuracy_score(labels, preds)),
        "auprc": float(average_precision_score(labels == pos_class, probs)) if has_pos else float("nan"),
    }


def make_splits(y, groups, folds, seed):
    """Always patient-stratified – no window leakage across folds."""
    n_patients = len(np.unique(groups))
    if folds >= n_patients:
        log(f"warning: folds ({folds}) >= patients ({n_patients}); "
            f"falling back to leave-one-patient-out ({n_patients} folds)")
        return list(LeaveOneGroupOut().split(np.zeros(len(y)), y, groups))
    sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(y)), y, groups))


def _run_one_fold(ModelCls, cfg, store, f_all, y_dev, tr, va, device, n_feat, K, pos,
                  n_epochs, tag):
    """Train for n_epochs on one fold; return best val AUPRC."""
    tr_t = torch.as_tensor(tr, device=store.index_device)
    va_t = torch.as_tensor(va, device=store.index_device)
    tr_d = torch.as_tensor(tr, device=device)

    counts  = torch.bincount(y_dev[tr_d], minlength=K).float()
    class_w = (counts.sum() / (K * counts.clamp(min=1))) ** cfg["power"]

    if f_all is not None:
        mu    = f_all[tr_d].mean(0, keepdim=True)
        sd    = f_all[tr_d].std(0, keepdim=True) + 1e-8
        f_norm = (f_all - mu) / sd
    else:
        f_norm = None

    model = ModelCls(extra_features=n_feat, n_classes=K).to(device)
    opt   = torch.optim.AdamW(model.parameters(),
                              lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    bs        = cfg["batch_size"]
    best_auprc = -1.0

    for epoch in range(n_epochs):
        model.train()
        perm      = tr_t[torch.randperm(len(tr_t), device=tr_t.device)]
        n_batches = len(perm) // bs
        for b in range(n_batches):
            idx   = perm[b * bs:(b + 1) * bs]
            idx_d = idx.to(device, non_blocking=True)
            xs    = store.get(idx)
            xf    = None if f_norm is None else f_norm[idx_d]
            loss  = F.cross_entropy(model(xs, xf), y_dev[idx_d], weight=class_w)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()

        _, preds, labels, probs = evaluate(
            model, store, f_norm, y_dev, va_t, class_w, pos, cfg["eval_batch_size"])
        m = fold_metrics(labels, preds, probs, pos)
        score = next((v for v in (m["auprc"], m["bacc"]) if np.isfinite(v)), -np.inf)
        if score > best_auprc:
            best_auprc = score

    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_auprc


# =============================================================================
# 4. Quick hyperparameter search
# =============================================================================
def run_tune(cfg_dict, tune_cfg, ModelCls, store, feats, y, groups, device, out_dir):
    """
    Random search over lr / weight_decay / batch_size.
    Runs n_trials × n_folds × n_epochs, picks the best combo by val AUPRC.
    Updates cfg_dict in-place with the winning params and saves tune_results.json.
    """
    search   = tune_cfg["search"]
    n_trials = tune_cfg["n_trials"]
    n_folds  = tune_cfg["n_folds"]
    n_epochs = tune_cfg["n_epochs"]

    K   = int(y.max()) + 1
    pos = K - 1
    y_dev = torch.as_tensor(y, dtype=torch.long, device=device)
    f_dev = None if feats is None else torch.as_tensor(feats, dtype=torch.float32, device=device)
    n_feat = 0 if feats is None else feats.shape[1]

    all_splits = make_splits(y, groups, max(cfg_dict["folds"], n_folds + 1), cfg_dict["seed"])
    # use the first n_folds splits for speed
    tune_splits = all_splits[:n_folds]

    rng     = random.Random(cfg_dict["seed"])
    results = []

    log(f"[tune] {n_trials} trials × {n_folds} fold(s) × {n_epochs} epochs")
    log(f"[tune] search space: { {k: v for k, v in search.items()} }")

    for trial in range(n_trials):
        trial_cfg = dict(cfg_dict)  # shallow copy, scalars only
        for param, choices in search.items():
            trial_cfg[param] = rng.choice(choices)

        tag = (f"lr={trial_cfg['lr']:.0e}  wd={trial_cfg['weight_decay']:.0e}"
               f"  bs={trial_cfg['batch_size']}")
        scores = []
        for tr, va in tune_splits:
            s = _run_one_fold(ModelCls, trial_cfg, store, f_dev, y_dev,
                              tr, va, device, n_feat, K, pos, n_epochs, tag)
            scores.append(s)
        mean_auprc = float(np.mean(scores))
        results.append({"trial": trial + 1, "params": {p: trial_cfg[p] for p in search},
                        "auprc": mean_auprc})
        log(f"  trial {trial + 1:2d}/{n_trials} | {tag} | AUPRC {mean_auprc:.4f}")

    best = max(results, key=lambda r: r["auprc"])
    log(f"[tune] best trial {best['trial']}: {best['params']}  AUPRC {best['auprc']:.4f}")

    # inject winning params into the live config
    for param, val in best["params"].items():
        cfg_dict[param] = val

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tune_results.json").write_text(json.dumps(
        {"best": best, "all_trials": results}, indent=2))
    return best


# =============================================================================
# 5. Full cross-validated training
# =============================================================================
def train_cv(cfg_dict, ModelCls, store, feats, y, groups, device, out_dir, run_name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = cfg_dict["class_names"]
    y_dev  = torch.as_tensor(y, dtype=torch.long, device=device)
    f_dev  = None if feats is None else torch.as_tensor(feats, dtype=torch.float32, device=device)
    n_feat = 0 if feats is None else feats.shape[1]
    K      = int(y.max()) + 1
    pos    = K - 1

    splits  = make_splits(y, groups, cfg_dict["folds"], cfg_dict["seed"])
    n_folds = len(splits)

    histories, best_rows, last_rows = [], [], []
    cm_best = np.zeros((K, K), dtype=np.int64)

    for fold, (tr, va) in enumerate(splits):
        t_fold = time.time()
        tr_t = torch.as_tensor(tr, device=store.index_device)
        va_t = torch.as_tensor(va, device=store.index_device)
        tr_d = torch.as_tensor(tr, device=device)

        counts  = torch.bincount(y_dev[tr_d], minlength=K).float()
        class_w = (counts.sum() / (K * counts.clamp(min=1))) ** cfg_dict["power"]

        if f_dev is not None:
            mu    = f_dev[tr_d].mean(0, keepdim=True)
            sd    = f_dev[tr_d].std(0, keepdim=True) + 1e-8
            f_all = (f_dev - mu) / sd
        else:
            f_all = None

        torch.manual_seed(cfg_dict["seed"] + fold)
        model = ModelCls(extra_features=n_feat, n_classes=K).to(device)
        opt   = torch.optim.AdamW(model.parameters(),
                                  lr=cfg_dict["lr"], weight_decay=cfg_dict["weight_decay"])

        log(f"[{run_name}] fold {fold + 1}/{n_folds} | train {len(tr)} "
            f"(pos {int(counts[pos])}) | val {len(va)} "
            f"| class w {[round(float(w), 2) for w in class_w]}")

        hist = {k: [] for k in ("train_loss", "val_loss", "val_acc", "val_bacc", "val_auprc")}
        best = {"auprc": -1.0, "epoch": -1}
        bs   = cfg_dict["batch_size"]

        for epoch in range(cfg_dict["epochs"]):
            t_ep = time.time()
            model.train()
            perm      = tr_t[torch.randperm(len(tr_t), device=tr_t.device)]
            n_batches = len(perm) // bs
            run       = torch.zeros((), device=device)
            n_skipped = 0
            for b in range(n_batches):
                idx   = perm[b * bs:(b + 1) * bs]
                idx_d = idx.to(device, non_blocking=True)
                xs    = store.get(idx)
                xf    = None if f_all is None else f_all[idx_d]
                loss  = F.cross_entropy(model(xs, xf), y_dev[idx_d], weight=class_w)
                if not torch.isfinite(loss):
                    n_skipped += 1
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_dict["clip"])
                opt.step()
                run += loss.detach()
            hist["train_loss"].append((run / max(n_batches - n_skipped, 1)).item())
            if n_skipped:
                log(f"    warning: skipped {n_skipped} batches with non-finite loss")

            vloss, preds, labels, probs = evaluate(
                model, store, f_all, y_dev, va_t, class_w, pos, cfg_dict["eval_batch_size"])
            m = fold_metrics(labels, preds, probs, pos)
            hist["val_loss"].append(vloss)
            hist["val_acc"].append(m["acc"])
            hist["val_bacc"].append(m["bacc"])
            hist["val_auprc"].append(m["auprc"])

            score = next((v for v in (m["auprc"], m["bacc"]) if np.isfinite(v)), -np.inf)
            if best["epoch"] < 0 or score > best["auprc"]:
                best = {"auprc": score, "epoch": epoch, "preds": preds, "labels": labels,
                        "state": copy.deepcopy(model.state_dict())}

            log(f"    epoch {epoch:2d} | train {hist['train_loss'][-1]:.4f} "
                f"| val {vloss:.4f} | acc {m['acc']:.3f} "
                f"| bal acc {m['bacc']:.3f} | AUPRC {m['auprc']:.3f} "
                f"| {time.time() - t_ep:.1f}s")

            if cfg_dict["patience"] and epoch - best["epoch"] >= cfg_dict["patience"]:
                log(f"    early stop (no AUPRC gain for {cfg_dict['patience']} epochs)")
                break

        torch.save(best["state"], out_dir / f"fold{fold + 1}.pt")
        cm_best += confusion_matrix(best["labels"], best["preds"], labels=list(range(K)))
        e = best["epoch"]
        best_rows.append({"fold": fold + 1, "epoch": e,
                          "bacc": hist["val_bacc"][e], "auprc": hist["val_auprc"][e]})
        last_rows.append({"fold": fold + 1,
                          "bacc": hist["val_bacc"][-1], "auprc": hist["val_auprc"][-1]})
        histories.append(hist)
        log(f"[{run_name}] fold {fold + 1}/{n_folds} done in "
            f"{(time.time() - t_fold) / 60:.1f} min "
            f"| best epoch {e}: bal acc {hist['val_bacc'][e]:.4f}, "
            f"AUPRC {hist['val_auprc'][e]:.4f}")

        del model, opt, best
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = _summarize(run_name, best_rows, last_rows, cm_best, pos, cfg_dict)
    (out_dir / "history.json").write_text(json.dumps(histories))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _plot(run_name, histories, cm_best, class_names, pos, out_dir / "summary.png")
    return summary


def _summarize(name, best_rows, last_rows, cm, pos, cfg_dict):
    def ms(rows, k):
        v = np.array([r[k] for r in rows], dtype=float)
        return float(np.nanmean(v)), float(np.nanstd(v))

    tn, fp  = int(cm[0, 0]), int(cm[0, 1:].sum())
    fn, tp  = int(cm[pos, :pos].sum()), int(cm[pos, pos])
    win_sec = cfg_dict["window_len"] / cfg_dict["fs"]
    hours   = cm.sum() * win_sec / 3600
    s = {
        "experiment": name,
        "best_epoch": {"bacc": ms(best_rows, "bacc"), "auprc": ms(best_rows, "auprc")},
        "last_epoch": {"bacc": ms(last_rows, "bacc"), "auprc": ms(last_rows, "auprc")},
        "pooled_best": {
            "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "precision":   tp / max(tp + fp, 1),
            "fp_per_hour": fp / hours if hours else float("nan"),
            "confusion_matrix": cm.tolist(),
        },
        "folds_best": best_rows,
        "config": {k: v for k, v in cfg_dict.items() if not k.startswith("_")},
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


def _plot(name, histories, cm, class_names, pos, path):
    max_ep = max(len(h["train_loss"]) for h in histories)

    def stack(key):
        return np.array([h[key] + [np.nan] * (max_ep - len(h[key])) for h in histories], float)

    ep  = np.arange(1, max_ep + 1)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))

    def band(a, axis, label, color):
        m, s = np.nanmean(a, 0), np.nanstd(a, 0)
        axis.plot(ep, m, label=label, color=color)
        axis.fill_between(ep, m - s, m + s, alpha=0.2, color=color)

    band(stack("train_loss"), ax[0], "train loss", "C0")
    band(stack("val_loss"),   ax[0], "val loss",   "C1")
    ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].set_title(f"{name}: weighted loss")
    band(stack("val_bacc"),  ax[1], "val balanced acc",               "C2")
    band(stack("val_auprc"), ax[1], f"val AUPRC ({class_names[pos]})", "C3")
    ax[1].set_xlabel("epoch"); ax[1].set_ylim(0, 1); ax[1].legend()
    ax[1].set_title("validation metrics (mean +/- std over folds)")

    K  = len(class_names)
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
# 6. Config loading
# =============================================================================
def _available_experiments():
    return sorted(p.stem for p in _EXP_DIR.glob("*.yaml")) if _EXP_DIR.exists() else []


def load_cfg(exp_name, cli_overrides: dict) -> dict:
    """
    Build a flat config dict:
      default_training_config.yaml
        <- experiments/<exp_name>.yaml  (model/preprocess/features + optional overrides)
          <- cli_overrides              (anything explicitly passed on the command line)
    The 'tune' block is extracted and returned separately.
    """
    # 1. base
    if not _DEFAULT_CFG.exists():
        raise SystemExit(f"Base config not found: {_DEFAULT_CFG}")
    with _DEFAULT_CFG.open() as f:
        cfg = yaml.safe_load(f) or {}

    # 2. experiment yaml
    exp_path = _EXP_DIR / f"{exp_name}.yaml"
    if not exp_path.exists():
        available = _available_experiments()
        raise SystemExit(
            f"Experiment '{exp_name}' not found.\n"
            f"Available: {available or '(none – create a yaml in configs/experiments/)'}\n"
            f"Expected path: {exp_path}")
    with exp_path.open() as f:
        exp = yaml.safe_load(f) or {}

    tune_cfg = {**_DEFAULT_TUNE, **exp.pop("tune", {})}
    # merge search sub-dict if the experiment yaml partially overrides it
    if "search" in exp.get("tune", {}):
        tune_cfg["search"] = {**_DEFAULT_TUNE["search"], **exp["tune"]["search"]}

    cfg.update(exp)

    # 3. CLI overrides (None = not provided, skip)
    for k, v in cli_overrides.items():
        if v is not None:
            cfg[k] = v

    # workers default
    if cfg.get("workers") is None:
        cfg["workers"] = max(1, (os.cpu_count() or 2) - 1)

    # required fields
    for required in ("model", "preprocess", "features"):
        if required not in cfg:
            raise SystemExit(f"Experiment yaml '{exp_name}' must define '{required}'.")

    return cfg, tune_cfg


# =============================================================================
# 7. CLI + main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--exp", default=None,
                   help="experiment name (stem of a yaml in configs/experiments/); "
                        "use --list-experiments to see what's available")
    p.add_argument("--list-experiments", action="store_true",
                   help="print available experiment names and exit")
    p.add_argument("--tune", action="store_true",
                   help="run a quick random search before full CV")

    # paths – can also be set in the experiment yaml; CLI takes precedence
    p.add_argument("--data-dir",  default=None,
                   help="path to npy_data/ folder (default: ../CHB-MIT_Formating/npy_data, "
                        "or data_dir in the experiment yaml)")
    p.add_argument("--cache-dir", default=None,
                   help="cache directory (default: cache, or cache_dir in the experiment yaml)")
    p.add_argument("--out-dir",   default=None,
                   help="results root (default: results, or out_dir in the experiment yaml)")
    p.add_argument("--patients",  nargs="+", default=None)

    # hyperparams – all optional, override the yaml stack
    p.add_argument("--folds",           type=int)
    p.add_argument("--epochs",          type=int)
    p.add_argument("--batch-size",      type=int,   dest="batch_size")
    p.add_argument("--eval-batch-size", type=int,   dest="eval_batch_size")
    p.add_argument("--lr",              type=float)
    p.add_argument("--weight-decay",    type=float, dest="weight_decay")
    p.add_argument("--clip",            type=float)
    p.add_argument("--power",           type=float)
    p.add_argument("--patience",        type=int)
    p.add_argument("--seed",            type=int)
    p.add_argument("--workers",         type=int)
    p.add_argument("--fs",              type=float)
    p.add_argument("--data-on",         dest="data_on", choices=["auto", "gpu", "ram"])
    p.add_argument("--class-names",     nargs="+",  dest="class_names")

    return p.parse_args()


def main():
    args = parse_args()

    if args.list_experiments:
        exps = _available_experiments()
        print("Available experiments:")
        for e in exps:
            print(f"  {e}")
        return

    if args.exp is None:
        exps = _available_experiments()
        raise SystemExit(
            "Specify an experiment with --exp <name>.\n"
            f"Available: {exps or '(none)'}\n"
            "Use --list-experiments for details.")

    cli_overrides = {
        "folds": args.folds, "epochs": args.epochs, "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size, "lr": args.lr,
        "weight_decay": args.weight_decay, "clip": args.clip, "power": args.power,
        "patience": args.patience, "seed": args.seed, "workers": args.workers,
        "fs": args.fs, "data_on": args.data_on, "class_names": args.class_names,
        "data_dir": args.data_dir, "cache_dir": args.cache_dir, "out_dir": args.out_dir,
    }
    cfg, tune_cfg = load_cfg(args.exp, cli_overrides)

    ModelCls = MODEL_REGISTRY.get(cfg["model"])
    if ModelCls is None:
        raise SystemExit(f"Unknown model '{cfg['model']}'. "
                         f"Available: {list(MODEL_REGISTRY)}")

    data_dir  = Path(cfg.get("data_dir")  or "../CHB-MIT_Formating/npy_data")
    cache_dir = Path(cfg.get("cache_dir") or "cache")
    out_dir   = Path(cfg.get("out_dir")   or "results") / args.exp

    patients = args.patients
    if patients is None:
        patients = sorted(
            d.name for d in data_dir.iterdir()
            if (d / "X.npy").exists() and (d / "y.npy").exists()
        ) if data_dir.is_dir() else []
    missing = [p for p in patients
               if not ((data_dir / p / "X.npy").exists() and (data_dir / p / "y.npy").exists())]
    if missing or not patients:
        raise SystemExit(f"no X.npy/y.npy for {missing or 'any patient'} in {data_dir.resolve()}")

    log(f"experiment : {args.exp}")
    log(f"model      : {cfg['model']}  |  preprocess: {cfg['preprocess']}"
        f"  |  features: {cfg['features']}")
    log(f"patients   : {patients}")

    device = pick_device()
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    srcs   = [data_dir / p / "X.npy" for p in patients]
    shapes = [check_input_npy(s) for s in srcs]
    if len({s[1:] for s in shapes}) != 1:
        raise SystemExit(f"patients have different window shapes: "
                         f"{dict(zip(patients, shapes))}")

    y, groups = load_labels(data_dir, patients, [s[0] for s in shapes])
    cfg["window_len"] = int(shapes[0][2])

    log(f"data: {len(y)} windows | seizure {int((y == y.max()).sum())} "
        f"({100 * (y == y.max()).mean():.2f}%) | "
        f"{len(y) * shapes[0][1] * shapes[0][2] * 2 / 2**30:.1f} GB as float16")

    if len(cfg["class_names"]) != int(y.max()) + 1:
        cfg["class_names"] = [str(i) for i in range(int(y.max()) + 1)]

    def parts(kind):
        return [build_patient_cache(kind, p, s, cache_dir, cfg["workers"])
                for p, s in zip(patients, srcs)]

    eeg_parts = parts(cfg["preprocess"])
    apen      = np.concatenate([np.asarray(a) for a in parts("apen")]) \
                if cfg["features"] == "apen" else None

    store = WindowStore(eeg_parts, device, f"{cfg['preprocess']} EEG",
                        cfg["data_on"], cfg["eval_batch_size"])

    if device.type == "cuda":
        log(f"VRAM in use: {torch.cuda.memory_allocated() / 2**30:.1f} GB")

    # optional quick random search
    if args.tune:
        run_tune(cfg, tune_cfg, ModelCls, store, apen, y, groups, device, out_dir)
        log(f"[tune] proceeding to full CV with: "
            f"lr={cfg['lr']}  wd={cfg['weight_decay']}  bs={cfg['batch_size']}")

    result = train_cv(cfg, ModelCls, store, apen, y, groups, device, out_dir, args.exp)

    b, p2 = result["best_epoch"], result["pooled_best"]
    log("=" * 78)
    log(f"{args.exp} | bal acc {b['bacc'][0]:.3f} +/- {b['bacc'][1]:.3f} "
        f"| AUPRC {b['auprc'][0]:.3f} +/- {b['auprc'][1]:.3f} "
        f"| sens {p2['sensitivity']:.3f} | spec {p2['specificity']:.4f} "
        f"| FP/h {p2['fp_per_hour']:.1f}")


if __name__ == "__main__":
    main()
