import json
import sys
import time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "CNN_Seizure_Detection_improved"))  # train_gpu lives there
from train_gpu import *
from train_gpu import (_make_batch_idx, _make_eval_idx, _forward_and_loss,  # import * skips _ names
                       _window_labels, _to_uv_f16)
from Architecture import *
import optuna
from optuna.visualization import plot_parallel_coordinate, plot_param_importances


MODEL_REGISTRY = {
    "ficnn":      FiCNN,
    "ficnn_reg":  FiCNN_reg,
    "ficrnn":     FiCRNN,
    "ficrnn_gru": FiCRNN_GRU,
    "ficrnn_gru_opt": FiCRNN_GRU_OPT
}


def save_optuna_plots(study, output_dir):
    """Save high-level Optuna parameter-effect plots as interactive HTML files."""
    completed = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if not completed:
        raise ValueError("Cannot create Optuna plots without completed trials")

    excluded_prefixes = ("kernel_size_", "channels_")
    plot_params = sorted({
        parameter
        for trial in completed
        for parameter in trial.params
        if not parameter.startswith(excluded_prefixes)
    })
    if not plot_params:
        raise ValueError("No high-level parameters are available for Optuna plots")

    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "param_importances": plot_param_importances(study, params=plot_params),
        "parallel_coordinate": plot_parallel_coordinate(study, params=plot_params),
    }
    paths = {}
    for name, figure in figures.items():
        path = output_dir / f"{name}.html"
        figure.write_html(str(path), include_plotlyjs=True)
        paths[name] = str(path)
    return paths


def sample_architecture(trial, config):
    """Sample the FiCRNN_GRU_OPT kwargs. Every parameter has a unique name and a
    fixed distribution; the constraints (kernels stay/shrink, channels stay/grow)
    hold by construction."""
    window_length = 256
    kernel_min = config["kernel_sizes"]["MIN"]
    kernel_max = config["kernel_sizes"]["MAX"]

    def output_length(length, kernel):
        return (length - kernel + 1) // 2

    max_layers = 0
    length = window_length
    while output_length(length, kernel_min) > 0:
        length = output_length(length, kernel_min)
        max_layers += 1

    configured_min = config["film_layers"]["MIN"]
    effective_min = min(configured_min, max_layers)
    if effective_min != configured_min:
        log(f"film_layers minimum {configured_min} reduced to {effective_min} "
            f"for window length {window_length}")
    if effective_min < 1:
        raise ValueError("No valid FiLM layer can fit the configured window length")

    N = trial.suggest_int(
        "film_layers",
        effective_min,
        min(config["film_layers"]["MAX"], max_layers),
    )

    kernels, channels = [], []
    previous_kernel_max = kernel_max
    # the input EEG channels count as layer -1 of the "channels stay/grow" rule,
    # so the first FiLM layer may not compress below the number of input channels
    channel_max = config["n_chanels"]["MAX"]
    previous_channel_min = max(config["n_chanels"]["MIN"], config["n_channels"])
    if previous_channel_min > channel_max:
        raise ValueError(f"n_chanels.MAX ({channel_max}) must be >= the number of input "
                         f"channels ({config['n_channels']})")
    length = window_length
    for i in range(N):
        feasible_kernels = []
        for kernel in range(kernel_min, previous_kernel_max + 1):
            candidate_length = output_length(length, kernel)
            for _ in range(N - i - 1):
                candidate_length = output_length(candidate_length, kernel)
            if candidate_length > 0:
                feasible_kernels.append(kernel)
        if not feasible_kernels:
            raise ValueError(f"No valid kernel size remains at FiLM layer {i}")

        kernel = trial.suggest_int(
            f"kernel_size_{i}",
            feasible_kernels[0],
            feasible_kernels[-1],
        )
        length = output_length(length, kernel)
        kernels.append(kernel)
        channels.append(trial.suggest_int(
            f"channels_{i}",
            previous_channel_min,
            channel_max,
        ))
        previous_kernel_max = kernel
        previous_channel_min = channels[-1]

    trial.set_user_attr("resolved_kernels", kernels)
    return dict(
        kernels=kernels,
        channels=channels,
        film_generator_layers=trial.suggest_int("film_generator_layers",
                                                config["film_generator_layers"]["MIN"],
                                                config["film_generator_layers"]["MAX"]),
        classifier_hiden=trial.suggest_int("classifier_hiden",
                                           config["classifier_hiden"]["MIN"],
                                           config["classifier_hiden"]["MAX"]),
        dropouts_film_cnn=0.0,
        dropouts_classifier=trial.suggest_categorical("dropout_classifier", config["dropouts"]),
        dropout_film_generator=0.0,
    )

def optimize(config, x, y, features):
    """x, y: per-patient lists (x[i] (n_i, C, T), y[i] (n_i,)); patient id = list index.
    Patient-wise k-fold, one model per fold, trained one epoch per kfold_epoch call."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    def patient_splits(x, y):
        """Per-patient lists -> window store, labels on device, patient-wise (tr, va) folds."""
        groups = np.repeat(np.arange(len(x)), [len(p) for p in x])
        y = np.concatenate(y)
        return (WindowStore(x, device, "kfold"),
                torch.as_tensor(y, dtype=torch.long, device=device),
                groups,
                make_splits(y, groups, config["n_folds"], config["seed"]))

    def kfold_epoch(trial, epoch, models, opts, splits, fold_data, groups, x, y, seq_len):
        """One epoch on each fold's own model (updated in place); reports mean val AUPRC."""
        K, scores = int(y.max()) + 1, []
        # batch_windows is a window budget: memory per step stays constant whatever seq_len is
        seqs_per_batch = max(1, config["batch_windows"] // seq_len)
        for (tr, va), (class_w, f_norm, va_idx), model, opt in zip(splits, fold_data, models, opts):
            model.train()
            for idx in _make_batch_idx(tr, groups, seq_len, seqs_per_batch,
                                       config["seed"] + epoch, x.index_device):
                opt.zero_grad(set_to_none=True)
                loss, _, _ = _forward_and_loss(model, x, f_norm, y, idx, class_w, seq_len, device)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip"])
                opt.step()
            _, preds, labels, probs = evaluate(model, x, f_norm, y, va_idx,
                                               class_w, K - 1, config["eval_batch_size"], seq_len, device)
            scores.append(fold_metrics(labels, preds, probs, K - 1)["auprc"])
        if not np.isfinite(scores).any():
            raise optuna.TrialPruned(f"no finite validation AUPRC at epoch {epoch}")
        score = float(np.nanmean(scores))
        trial.report(score, epoch)
        if trial.should_prune():
            log(f"  trial {trial.number} epoch {epoch}: pruned at AUPRC {score:.4f}")
            raise optuna.TrialPruned()
        return score

    store, y_dev, groups, splits = patient_splits(x, y)
    K = int(y_dev.max()) + 1
    f_all = torch.as_tensor(np.concatenate(features), dtype=torch.float32, device=device)
    n_features = f_all.shape[1]
    seq_cfg = config.get("sequence_len", 32)
    if not isinstance(seq_cfg, dict):          # plain int = fixed sequence length
        seq_cfg = {"MIN": seq_cfg, "MAX": seq_cfg}
    seq_min, seq_max = int(seq_cfg["MIN"]), int(seq_cfg["MAX"])
    if not 1 <= seq_min <= seq_max:
        raise ValueError(f"sequence_len needs 1 <= MIN <= MAX, got MIN={seq_min}, MAX={seq_max}")

    # per-fold class weights and train-normalised features are trial-independent:
    # compute once instead of every fold of every epoch of every trial
    fold_stats = []
    for tr, va in splits:
        tr_dev = torch.as_tensor(tr, device=device)
        counts = torch.bincount(y_dev[tr_dev], minlength=K).float()
        class_w = (counts.sum() / (K * counts.clamp(min=1))) ** config["power"]
        f_mu = f_all[tr_dev].mean(0, keepdim=True)
        f_sd = f_all[tr_dev].std(0, keepdim=True) + 1e-8
        fold_stats.append((class_w, (f_all - f_mu) / f_sd))

    # validation indices depend on the sampled sequence length: build once per length
    eval_idx_cache = {}
    def fold_data_for(seq_len):
        if seq_len not in eval_idx_cache:
            eval_idx_cache[seq_len] = [_make_eval_idx(va, groups, seq_len, store.index_device)
                                       for _, va in splits]
        return [(class_w, f_norm, va_idx)
                for (class_w, f_norm), va_idx in zip(fold_stats, eval_idx_cache[seq_len])]

    class TrialOutOfMemory(Exception):
        """Trial needed more GPU memory than available; recorded as FAIL, study continues."""

    def Objective(trial):
        oom = None
        try:
            return run_trial(trial)
        except torch.cuda.OutOfMemoryError as e:
            oom = str(e).split(".")[0]
        # outside the except block the traceback (and with it run_trial's models, optimiser
        # state and activations) is released, so the cached GPU memory can really be freed
        if device.type == "cuda":
            torch.cuda.empty_cache()
        log(f"trial {trial.number}: out of GPU memory ({oom}), skipped; params {trial.params}")
        raise TrialOutOfMemory(oom)

    def run_trial(trial):
        trial_start = time.perf_counter()
        lr = trial.suggest_float("learning_rate", float(config["lr"]["MIN"]),
                                 float(config["lr"]["MAX"]), log=True)
        # log scale: 1->2 matters as much as 32->64; sequence_len=1 = window-independent model
        seq_len = trial.suggest_int("sequence_len", seq_min, seq_max, log=True)
        fold_data = fold_data_for(seq_len)
        arch = sample_architecture(trial, config)
        models = [FiCRNN_GRU_OPT(extra_features=n_features, n_classes=K,
                                 n_channels=config["n_channels"], **arch).to(device)
                  for _ in splits]
        opts = [torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=config["weight_decay"]) for m in models]
        best = -np.inf
        for epoch in range(config["epochs"]):
            score = kfold_epoch(trial, epoch, models, opts, splits, fold_data, groups,
                                store, y_dev, seq_len)
            best = max(best, score)
            log(f"  trial {trial.number} epoch {epoch}: AUPRC {score:.4f} (best {best:.4f})")

        film = ", ".join(f"{kernel}x{channel}" for kernel, channel in zip(arch["kernels"], arch["channels"]))
        elapsed = time.perf_counter() - trial_start
        log(f"trial {trial.number}: AUPRC {best:.4f} | time {elapsed:.2f}s | seq_len {seq_len} | "
            f"arch FiLM[{film}] -> GRU[{arch['classifier_hiden']}] -> "
            f"FiLM generator[{arch['film_generator_layers']}]")
        return best

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config["seed"]),
                                pruner=optuna.pruners.HyperbandPruner(min_resource=3, max_resource=config["epochs"]))
    study.optimize(Objective, n_trials=config["n_trials"],
                   catch=(AssertionError, TrialOutOfMemory))  # skip too-deep or too-large archs
    all_trials = [
        {
            "trial": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
        }
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    all_trials.sort(key=lambda trial: trial["value"], reverse=True)
    results = {
        "best": {
            "trial": study.best_trial.number,
            "value": study.best_value,
            "params": study.best_params,
        },
        "all_trials": all_trials,
    }
    output_path = HERE / "architecture_optimizer_results.json"
    try:
        results["plots"] = save_optuna_plots(study, HERE)
    except Exception as e:  # noqa: BLE001 - plots are optional, never lose the results over them
        log(f"warning: could not create Optuna plots: {e}")
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    log(json.dumps(results, indent=2))
    log(f"Results written to {output_path}")
    return study


def load_data(config):
    """Load per-patient EEG windows, labels, and ApEn conditioning features."""
    data_dir = HERE.parent / "CHB-MIT_Formating" / "npy_data"
    cache_dir = HERE.parent / "CNN_Seizure_Detection_improved" / "cache" / "apen"
    x, y = [], []
    features = []
    for p in config["patients"]:
        # shortcut: whole patient in RAM (float32 peak during scaling), stream via mmap chunks if OOM
        xk = _to_uv_f16(np.load(data_dir / p / "X.npy", mmap_mode="r"))[0]
        y.append(_window_labels(np.load(data_dir / p / "y.npy"), len(xk), p))
        x.append(xk)
        feature_path = cache_dir / f"{p}.npy"
        if not feature_path.exists():
            raise FileNotFoundError(
                f"ApEn cache not found at {feature_path}; run the ApEn preprocessing first")
        apen = np.asarray(np.load(feature_path, mmap_mode="r"), dtype=np.float32)
        if len(apen) != len(xk):
            raise ValueError(f"ApEn feature count ({len(apen)}) does not match EEG windows "
                             f"({len(xk)}) for {p}")
        features.append(apen.reshape(len(apen), -1))

    log (f"loaded {len(x)} patients, {sum(len(p) for p in x)} windows")
    log(f"loaded ApEn features with shape {features[0].shape[1:]}")
    return x, y, features


def main():
    import yaml
    with open(HERE / "Architecture_Optimizer_config.yaml") as f:
        config = yaml.safe_load(f)
    log(f"Config loaded: {config}")
    x, y, features = load_data(config)
    config["n_channels"] = int(x[0].shape[1])
    config["window_len"] = int(x[0].shape[2])
    optimize(config, x, y, features)
    log("Optimization finished")


if __name__ == "__main__":
    main()
