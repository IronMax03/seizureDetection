import json
import sys
import time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "CNN_Seizure_Detection_improved"))  # train_gpu lives there
from train_gpu import *
from train_gpu import _make_batch_idx, _forward_and_loss, _window_labels, _to_uv_f16  # import * skips _ names
from Architecture import *
import optuna


MODEL_REGISTRY = {
    "ficnn":      FiCNN,
    "ficnn_reg":  FiCNN_reg,
    "ficrnn":     FiCRNN,
    "ficrnn_gru": FiCRNN_GRU,
    "ficrnn_gru_opt": FiCRNN_GRU_OPT
}

def sample_architecture(trial, config):
    """Sample the FiCRNN_GRU_OPT kwargs. Every parameter has a unique name and a
    fixed distribution; the constraints (kernels stay/shrink, channels stay/grow)
    hold by construction."""
    N = trial.suggest_int("film_layers", config["film_layers"]["MIN"], config["film_layers"]["MAX"])

    kernels, channels = [], []
    k_max, c_min = config["kernel_sizes"]["MAX"], config["n_chanels"]["MIN"]
    for i in range(N):
        kernels.append(trial.suggest_int(f"kernel_size_{i}", config["kernel_sizes"]["MIN"], k_max))
        channels.append(trial.suggest_int(f"channels_{i}", c_min, config["n_chanels"]["MAX"]))
        k_max, c_min = kernels[-1], channels[-1]

    return dict(
        kernels=kernels,
        channels=channels,
        film_generator_layers=trial.suggest_int("film_generator_layers",
                                                config["film_generator_layers"]["MIN"],
                                                config["film_generator_layers"]["MAX"]),
        classifier_hiden=trial.suggest_int("classifier_hiden",
                                           config["classifier_hiden"]["MIN"],
                                           config["classifier_hiden"]["MAX"]),
        dropouts_film_cnn=trial.suggest_categorical("dropout_film_cnn", config["dropouts"]),
        dropouts_classifier=trial.suggest_categorical("dropout_classifier", config["dropouts"]),
        dropout_film_generator=trial.suggest_categorical("dropout_film_generator", config["dropouts"]),
    )

def optimize(config, x, y):
    """x, y: per-patient lists (x[i] (n_i, C, T), y[i] (n_i,)); patient id = list index.
    Patient-wise k-fold, one model per fold, trained one epoch per kfold_epoch call."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    def patient_splits(x, y):
        """Per-patient lists -> window store, labels on device, patient-wise (tr, va) folds."""
        groups = np.repeat(np.arange(len(x)), [len(p) for p in x])
        y = np.concatenate(y)
        return (WindowStore(x, device, "kfold"),
                torch.as_tensor(y, dtype=torch.long, device=device),
                make_splits(y, groups, config["n_folds"], config["seed"]))

    def kfold_epoch(trial, epoch, models, opts, splits, x, y):
        """One epoch on each fold's own model (updated in place); reports mean val AUPRC."""
        K, scores = int(y.max()) + 1, []
        for (tr, va), model, opt in zip(splits, models, opts):
            counts  = torch.bincount(y[torch.as_tensor(tr, device=device)], minlength=K).float()
            class_w = (counts.sum() / (K * counts.clamp(min=1))) ** config["power"]
            model.train()
            for idx in _make_batch_idx(tr, None, 1, config["batch_size"], config["seed"] + epoch, x.index_device):
                opt.zero_grad(set_to_none=True)
                loss, _, _ = _forward_and_loss(model, x, None, y, idx, class_w, 1, device)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip"])
                opt.step()
            _, preds, labels, probs = evaluate(model, x, None, y, torch.as_tensor(va, device=x.index_device),
                                               class_w, K - 1, config["eval_batch_size"], 1, device)
            scores.append(fold_metrics(labels, preds, probs, K - 1)["auprc"])
        score = float(np.mean(scores))
        trial.report(score, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return score

    store, y_dev, splits = patient_splits(x, y)
    K = int(y_dev.max()) + 1

    def Objective(trial):
        trial_start = time.perf_counter()
        lr = trial.suggest_float("learning_rate", config["lr"]["MIN"], config["lr"]["MAX"])
        arch = sample_architecture(trial, config)
        models = [FiCRNN_GRU_OPT(n_classes=K, n_channels=config["n_channels"], **arch).to(device) for _ in splits]
        opts = [torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=config["weight_decay"]) for m in models]
        best = -np.inf
        for epoch in range(config["epochs"]):
            best = max(best, kfold_epoch(trial, epoch, models, opts, splits, store, y_dev))
            log(f"  trial {trial.number} epoch {epoch}: AUPRC {best:.4f}")

        film = ", ".join(f"{kernel}x{channel}" for kernel, channel in zip(arch["kernels"], arch["channels"]))
        elapsed = time.perf_counter() - trial_start
        log(f"trial {trial.number}: AUPRC {best:.4f} | time {elapsed:.2f}s | "
            f"arch FiLM[{film}] -> GRU[{arch['classifier_hiden']}] -> "
            f"FiLM generator[{arch['film_generator_layers']}]")
        return best

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config["seed"]),
                                pruner=optuna.pruners.HyperbandPruner(min_resource=3, max_resource=config["epochs"]))
    study.optimize(Objective, n_trials=config["n_trials"],
                   catch=(AssertionError,))  # too-deep archs fail the model's window-length assert; skip them
    all_trials = [
        {
            "trial": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "params": trial.params,
        }
        for trial in study.trials
    ]
    log(json.dumps({
        "best": {
            "trial": study.best_trial.number,
            "value": study.best_value,
            "params": study.best_params,
        },
        "last_trial": study.trials[-1].number,
        "all_trials": all_trials,
    }, indent=2))
    return study


def load_data(config):
    """Per-patient lists from CHB-MIT_Formating/npy_data: x[i] (n_i, C, T) uV float16, y[i] (n_i,) labels."""
    data_dir = HERE.parent / "CHB-MIT_Formating" / "npy_data"
    x, y = [], []
    for p in config["patients"]:
        # shortcut: whole patient in RAM (float32 peak during scaling), stream via mmap chunks if OOM
        xk = _to_uv_f16(np.load(data_dir / p / "X.npy", mmap_mode="r"))[0]
        y.append(_window_labels(np.load(data_dir / p / "y.npy"), len(xk), p))
        x.append(xk)

    log (f"loaded {len(x)} patients, {sum(len(p) for p in x)} windows")
    return x, y


def main():
    import yaml
    with open(HERE / "Architecture_Optimizer_config.yaml") as f:
        config = yaml.safe_load(f)
    log(f"Config loaded: {config}")
    x, y = load_data(config)
    config["n_channels"] = int(x[0].shape[1])
    config["window_len"] = int(x[0].shape[2])
    optimize(config, x, y)
    log("Optimization finished")


if __name__ == "__main__":
    main()
