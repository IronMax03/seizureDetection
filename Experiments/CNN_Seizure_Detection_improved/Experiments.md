# Quick start

```bash
# list available experiments
python train_gpu.py --list-experiments

# run an experiment with default hyperparams
python train_gpu.py --exp ficnn_apen

# run several experiments one after the other
python train_gpu.py --exp ficnn_apen ficrnn_apen ficrnn_gru_apen_1ch

# run a quick hyperparameter search first, then full CV
python train_gpu.py --exp ficnn_apen --tune

# override any hyperparam on the fly
python train_gpu.py --exp ficnn_apen --lr 1e-3 --epochs 50 --folds 5
```

Results land in `results/<exp_name>/` — one subfolder per experiment run.

---

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
    "ficnn":   FiCNN,
    "ficrnn":  FiCRNN,
    "mymodel": MyModel,   # ← add this
}
```

Also add the import at the top:

```python
from Architectures import FiCNN, FiCRNN, MyModel
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
  n_trials: 8        # random configs to evaluate
  n_folds: 1         # folds per trial (1 = fast signal)
  n_epochs: 10       # epochs per trial
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

---

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
| `fs` | 256.0 | Sampling rate Hz (used for FP/hour reporting only) |
| `data_fraction` | `1.0` | Fraction of windows to keep per patient (stratified by class). Useful values: `0.75`, `0.5`, `0.25`, `0.1` |
| `data_dir` | `../CHB-MIT_Formating/npy_data` | Path to the folder containing per-patient `X.npy`/`y.npy` |
| `cache_dir` | `cache` | Directory for preprocessed caches |
| `out_dir` | `results` | Root directory for run outputs |

Path keys (`data_dir`, `cache_dir`, `out_dir`) are not in the base config by default — set them only in experiment yamls where you need to override the defaults, or pass `--data-dir` / `--cache-dir` / `--out-dir` on the CLI.

---

# Output files

After a run, `results/<exp_name>/` contains:

| File | Content |
|---|---|
| `summary.json` | Mean ± std metrics across folds (best and last epoch), pooled confusion matrix |
| `history.json` | Per-fold, per-epoch loss and metrics |
| `summary.png` | Loss curves, validation metrics, confusion matrix |
| `fold1.pt … foldN.pt` | Best-epoch model weights per fold |
| `tune_results.json` | (only with `--tune`) All trial results + winning params |

---
