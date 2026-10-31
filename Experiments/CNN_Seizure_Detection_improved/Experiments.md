# Quick start

```bash
pip install optuna  # only needed for --tune

# list available experiments
python train_gpu.py --list-experiments

# run an experiment with default hyperparams
python train_gpu.py --exp ficnn_apen

# run several experiments one after the other
python train_gpu.py --exp ficnn_apen ficrnn_apen ficrnn_gru_apen_1ch

# Optuna successive-halving search first, then full CV
python train_gpu.py --exp ficnn_apen --tune

# override any hyperparam on the fly
python train_gpu.py --exp ficnn_apen --lr 1e-3 --epochs 50 --folds 5

# train on a subset of patients only
python train_gpu.py --exp ficnn_apen --patients chb01 chb02
```

Results land in `results/<exp_name>/` — one subfolder per experiment run.


# CLI reference

**Run control**

| Flag | Description |
|---|---|
| `--exp NAME [NAME …]` | One or more experiment names (stems of yamls in `configs/experiments/`), run one after the other |
| `--list-experiments` | Print the available experiments and exit |
| `--tune` | Run an Optuna successive-halving search before full CV (see [Hyperparameter tuning](#hyperparameter-tuning---tune)) |

**Data / paths**

| Flag | Description |
|---|---|
| `--data-dir PATH` | Folder containing per-patient `X.npy`/`y.npy` (yaml key `data_dir`) |
| `--cache-dir PATH` | Directory for preprocessed caches (yaml key `cache_dir`) |
| `--out-dir PATH` | Root directory for run outputs (yaml key `out_dir`) |
| `--patients ID [ID …]` | Patient folder names, e.g. `chb01 chb02`. Default: every folder with `X.npy`/`y.npy`. **Needs ≥ 2 patients**: folds split by patient, so a single patient raises a `LeaveOneGroupOut` ValueError |

**Hyperparameter overrides** (each overrides the yaml key of the same meaning, see [Config system](#config-system))

| Flag | Yaml key |
|---|---|
| `--folds INT` | `folds` |
| `--epochs INT` | `epochs` |
| `--batch-size INT` | `batch_size` |
| `--eval-batch-size INT` | `eval_batch_size` |
| `--lr FLOAT` | `lr` |
| `--weight-decay FLOAT` | `weight_decay` |
| `--clip FLOAT` | `clip` |
| `--power FLOAT` | `power` |
| `--patience INT` | `patience` |
| `--seed INT` | `seed` |
| `--workers INT` | `workers` |
| `--fs FLOAT` | `fs` |
| `--data-on {auto,gpu,ram}` | `data_on` |
| `--class-names NAME [NAME …]` | `class_names` |
| `--sequence-len INT` | `sequence_len` |

Precedence: CLI flag > experiment yaml > base yaml. With `--tune`, each outer fold's winning `search` values replace these values for that fold.


# Hyperparameter tuning (`--tune`)

`--tune` tries several hyperparameter combinations from `tune.search` using **successive halving**: every combination starts training, and the weaker ones are stopped early so the training time goes to the most promising ones. The search runs inside every outer CV fold, on that fold's training patients only (nested CV), and the fold is then trained with its own best combination.

**`tune:` keys** (in the experiment yaml):

| Key | Default | Description |
|---|---|---|
| `n_trials` | 8 | Total trials, pruned ones included. 20–30 recommended |
| `n_folds` | 1 | Patient folds per trial |
| `n_epochs` | 10 | Maximum epochs per fold per trial |
| `search` | `lr`, `weight_decay`, `batch_size` | Any base-config key mapped to a list of choices or to a range `{low, high, log}`, except `sequence_len`, `folds`, `seed` |

Gotchas:

- A `search` block in the yaml **replaces** the default search space, it is not merged. List every key you want searched.
- `search` must sit directly under `tune:`. A nested `tune: tune: search:` is silently ignored and the default search space is used.
- The first trial is never pruned.
- Nothing prints during a trial, only one line per finished or pruned trial:
  ```
  [tune] 20 trials × 1 fold(s) × ≤10 epochs  (successive halving, sequence_len=1)
    trial  1/20 | lr=3e-04  wd=1e-04  bs=256 | AUPRC 0.4123
    trial  2/20 | lr=3e-03  wd=0e+00  bs=512 | pruned
  ```
- Pruner settings are Optuna defaults and cannot be set from the config.


# Setting up a new experiment

## 1. Define your model in `Architectures.py`

The required signature is `(extra_features, n_classes, n_channels)`. `n_channels` is the number of EEG input channels (1 with `single_chanel_only: true`):

```python
class MyModel(nn.Module):
    def __init__(self, extra_features=0, n_classes=2, n_channels=23):
        ...

    def forward(self, x_serie, x_features):
        # x_serie:   (N, C, T) float32
        # x_features: (N, extra_features) float32, or None if extra_features=0
        ...
```

## 2. Register it in `train_gpu.py`

Find the `MODEL_REGISTRY` dict near the top and add one line:

```python
MODEL_REGISTRY = {
    "ficnn":      FiCNN,
    "ficnn_reg":  FiCNN_reg,
    "ficrnn":     FiCRNN,
    "ficrnn_gru": FiCRNN_GRU,
    "mymodel":    MyModel,   # ← add this
}
```

Also add the import at the top:

```python
from Architectures import FiCNN, FiCNN_reg, FiCRNN, FiCRNN_GRU, MyModel
```

## 3. Create an experiment yaml

Create `configs/experiments/mymodel_apen.yaml`:

```yaml
# required: what to train
model: mymodel
preprocess: raw      # raw | atar  (which EEG cache to use)
features: apen       # apen | none (conditioning signal; none = control)

# optional: dataset paths (override the CLI defaults for this experiment)
data_dir: /path/to/my/npy_data
cache_dir: /path/to/cache
out_dir: /path/to/results

# optional: train on the first EEG channel only (default false = all channels)
single_chanel_only: true

# optional: override base hyperparams for this experiment
epochs: 40
folds: 5

# optional: hyperparameter search config (only used with --tune)
tune:
  n_trials: 20       # total trials (pruned ones included)
  n_folds: 1         # folds per trial (1 = fast signal)
  n_epochs: 10       # max epochs per trial
  search:
    lr: [1.0e-4, 5.0e-4, 1.0e-3, 3.0e-3]
    weight_decay: [0.0, 1.0e-4, 1.0e-3]
    batch_size: [128, 256, 512]
```

## 4. Run it

```bash
# quick sanity check — tune finds decent hyperparams, then trains fully
python train_gpu.py --exp mymodel_apen --tune

# or skip tune if you already know good hyperparams
python train_gpu.py --exp mymodel_apen
```


# Config system

Parameters are resolved in this order (later overrides earlier):

```
configs/default_training_config.yaml      ← base hyperparams (lr, wd, epochs, …)
  └── configs/experiments/<name>.yaml     ← experiment-specific overrides + tune block
        └── CLI flags                     ← one-off overrides (--lr, --epochs, …)
```

**`default_training_config.yaml` reference:**

| Key | Default | Description |
|---|---|---|
| `folds` | 10 | Patient-stratified k-folds (`StratifiedGroupKFold`) |
| `epochs` | 30 | Training epochs per fold |
| `batch_size` | 256 | Training batch size |
| `eval_batch_size` | 4096 | Evaluation batch size |
| `lr` | 3e-4 | AdamW learning rate |
| `weight_decay` | 1e-4 | AdamW weight decay |
| `clip` | 1.0 | Gradient norm clipping |
| `power` | 0.5 | Class-weight exponent (0=none, 0.5=sqrt, 1=inv-freq) |
| `patience` | 0 | Early stopping patience in epochs (0 = disabled) |
| `data_on` | auto | Where EEG windows live: `auto` / `gpu` / `ram` |
| `seed` | 123 | Random seed |
| `workers` | cpu_count−1 | Parallel workers for cache building |
| `single_chanel_only` | `false` | `true` = train on the first EEG channel of each patient only (patients may then have different channel counts); `false` = all channels |
| `sequence_len` | 1 | Windows per RNN sequence; >1 for FiCRNN, 1 = window-independent (FiCNN) |
| `fs` | 256.0 | Sampling rate Hz (used for FP/hour reporting only) |
| `data_fraction` | `1.0` | Fraction of windows to keep per patient (stratified by class). Useful values: `0.75`, `0.5`, `0.25`, `0.1` |
| `data_dir` | `../CHB-MIT_Formating/npy_data` | Path to the folder containing per-patient `X.npy`/`y.npy` |
| `cache_dir` | `cache` | Directory for preprocessed caches |
| `out_dir` | `results` | Root directory for run outputs |

Path keys (`data_dir`, `cache_dir`, `out_dir`) are not in the base config by default — set them only in experiment yamls where you need to override the defaults, or pass `--data-dir` / `--cache-dir` / `--out-dir` on the CLI.


# Output files

After a run, `results/<exp_name>/` contains:

| File | Content |
|---|---|
| `summary.json` | Mean ± std metrics across folds (best and last epoch), pooled confusion matrix |
| `history.json` | Per-fold, per-epoch loss and metrics |
| `summary.png` | Loss curves, validation metrics, confusion matrix |
| `fold1.pt … foldN.pt` | Best-epoch model weights per fold |
| `tune_fold<k>.json` | (only with `--tune`, one per outer fold, nested CV) All trials (params, state `COMPLETE`/`PRUNED`, best score reached) + winning params |
