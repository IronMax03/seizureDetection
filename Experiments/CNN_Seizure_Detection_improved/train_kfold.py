import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, average_precision_score
import matplotlib.pyplot as plt
import numpy as np


class FiLMBlock(nn.Module):
    """Conv -> BatchNorm -> LeakyReLU -> FiLM modulation -> MaxPool.
    FiLM: h = gamma * h + beta, with gamma/beta per-channel, broadcast over time."""
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=1)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.LeakyReLU(0.01)
        self.pool = nn.MaxPool1d(2, stride=2)
        self.out_ch = out_ch

    def forward(self, x, gamma, beta):
        h = self.act(self.bn(self.conv(x)))
        # gamma, beta: (N, out_ch) -> (N, out_ch, 1) to broadcast over length
        h = gamma.unsqueeze(-1) * h + beta.unsqueeze(-1)
        return self.pool(h)


class CNN(nn.Module):
    def __init__(self, extra_features=0, n_classes=2):
        super().__init__()
        self.channels = [18, 15, 10, 10, 10]
        kernels = [6, 5, 4, 4, 4]
        in_chs = [23, 18, 15, 10, 10]

        self.blocks = nn.ModuleList([
            FiLMBlock(ic, oc, k) for ic, oc, k in zip(in_chs, self.channels, kernels)
        ])

        # FiLM generator: features -> (gamma, beta) for every channel of every block.
        total_ch = sum(self.channels)
        self.film_dim = 2 * total_ch
        # if there are no extra features, fall back to a learnable constant input
        self.no_features = (extra_features == 0)
        gen_in = extra_features if extra_features > 0 else 1
        self.film_generator = nn.Sequential(
            nn.Linear(gen_in, 32), nn.LeakyReLU(0.01),
            nn.Linear(32, self.film_dim),
        )
        # init the final layer so gamma starts at 1 and beta at 0 (identity FiLM)
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)

        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(40, 50), nn.LeakyReLU(0.01),   # 10 ch x 4 samples for L=256
            nn.Linear(50, 20), nn.LeakyReLU(0.01),
            nn.Linear(20, n_classes),
        )

    def forward(self, x_serie, x_features):
        N = x_serie.size(0)
        if self.no_features:
            cond = torch.ones(N, 1, device=x_serie.device, dtype=x_serie.dtype)
        else:
            cond = x_features

        film = self.film_generator(cond)          # (N, film_dim)
        idx = 0
        z = x_serie
        for blk in self.blocks:
            c = blk.out_ch
            gamma = 1.0 + film[:, idx:idx + c];        idx += c
            beta = film[:, idx:idx + c];               idx += c
            z = blk(z, gamma, beta)

        z = self.flatten(z)
        return self.classifier(z)


def train_model(x_serie, x_features, y,
                epochs=30, batch_size=32, lr=1e-3,
                n_splits=10, random_seed=123, device=None,
                class_weight_power=1.0,
                class_names=None,
                groups=None):
    """
    x_serie:            (N, 23, L) float tensor
    x_features:         (N, F)     float tensor (F=0 -> FiLM uses a constant, no conditioning)
    y:                  (N,)       long tensor, class indices 0..K-1 (last class = seizure/ictal)
    class_weight_power: 1.0 = inverse-frequency weights, 0.5 = sqrt (softer), 0 = unweighted
    class_names:        optional list of K names for the confusion matrix
    groups:             optional (N,) patient/recording IDs -> StratifiedGroupKFold (no leakage)
    Returns (fold_histories, fold_scores, model), fold_scores = final balanced accuracy per fold.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(random_seed)

    x_serie = torch.as_tensor(x_serie, dtype=torch.float32)
    x_features = torch.as_tensor(x_features, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.long)

    if x_serie.dim() == 2:
        x_serie = x_serie.unsqueeze(1)
    if x_features.dim() == 1:
        x_features = x_features.unsqueeze(1)
    assert x_serie.dim() == 3 and x_serie.shape[1] == 23, \
        f"x_serie must be (N, 23, L), got {tuple(x_serie.shape)}"
    assert x_serie.shape[0] == x_features.shape[0] == y.shape[0], \
        f"batch mismatch: {x_serie.shape[0]}, {x_features.shape[0]}, {y.shape[0]}"

    x_features = torch.nan_to_num(x_features, nan=0.0, posinf=0.0, neginf=0.0)
    n_feat = x_features.shape[1]

    n_classes = int(y.max().item()) + 1
    class_names = class_names or [str(i) for i in range(n_classes)]
    assert len(class_names) == n_classes, \
        f"{len(class_names)} class names for {n_classes} classes"
    pos_class = n_classes - 1                      # seizure / ictal = last class

    if groups is not None:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        splits = splitter.split(np.zeros(len(y)), y.numpy(), groups=np.asarray(groups))
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        splits = splitter.split(np.zeros(len(y)), y.numpy())

    fold_histories = []
    fold_scores = []
    cm_total = np.zeros((n_classes, n_classes), dtype=int)

    for fold, (tr, va) in enumerate(splits):
        tr = torch.as_tensor(tr)
        va = torch.as_tensor(va)

        # class weights from this fold's TRAIN labels
        counts = torch.bincount(y[tr], minlength=n_classes).float()
        class_w = (counts.sum() / (n_classes * counts.clamp(min=1))) ** class_weight_power
        class_w = class_w.to(device)
        print(f"fold {fold+1:2d}/{n_splits} | train counts {counts.long().tolist()} "
              f"| class weights {[round(float(w), 2) for w in class_w.cpu()]}")

        # standardize features using TRAIN stats only (skip if no features)
        if n_feat > 0:
            f_mean = x_features[tr].mean(dim=0, keepdim=True)
            f_std = x_features[tr].std(dim=0, keepdim=True) + 1e-8
            xf_all = (x_features - f_mean) / f_std
        else:
            xf_all = x_features

        train_ds = TensorDataset(x_serie[tr], xf_all[tr], y[tr])
        val_ds = TensorDataset(x_serie[va], xf_all[va], y[va])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = CNN(extra_features=n_feat, n_classes=n_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        hist = {"train_loss": [], "val_loss": [], "val_acc": [], "val_bacc": [], "val_auprc": []}

        for epoch in range(epochs):
            # ---- train ----
            model.train()
            running, n_seen = 0.0, 0
            for xs, xf, yb in train_loader:
                xs, xf, yb = xs.to(device), xf.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(xs, xf), yb, weight=class_w)
                loss.backward()
                optimizer.step()
                running += loss.item() * xs.size(0)
                n_seen += xs.size(0)
            hist["train_loss"].append(running / max(n_seen, 1))

            # ---- validate ----
            model.eval()
            running, w_sum = 0.0, 0.0
            preds_all, labels_all, probs_all = [], [], []
            with torch.no_grad():
                for xs, xf, yb in val_loader:
                    xs, xf, yb = xs.to(device), xf.to(device), yb.to(device)
                    out = model(xs, xf)
                    running += F.cross_entropy(out, yb, weight=class_w, reduction="sum").item()
                    w_sum += class_w[yb].sum().item()
                    preds_all.append(out.argmax(1).cpu())
                    labels_all.append(yb.cpu())
                    probs_all.append(out.softmax(1)[:, pos_class].cpu())

            preds = torch.cat(preds_all).numpy()
            labels = torch.cat(labels_all).numpy()
            probs = torch.cat(probs_all).numpy()

            hist["val_loss"].append(running / max(w_sum, 1e-8))
            hist["val_acc"].append(float((preds == labels).mean()))
            hist["val_bacc"].append(balanced_accuracy_score(labels, preds))
            has_pos = (labels == pos_class).any()
            hist["val_auprc"].append(
                average_precision_score(labels == pos_class, probs) if has_pos else np.nan
            )

            print(f"    epoch {epoch:2d} | train {hist['train_loss'][-1]:.4f} | "
                  f"val {hist['val_loss'][-1]:.4f} | acc {hist['val_acc'][-1]:.3f} | "
                  f"bal acc {hist['val_bacc'][-1]:.3f} | AUPRC {hist['val_auprc'][-1]:.3f}")

        # confusion matrix from the last epoch of this fold
        cm_total += confusion_matrix(labels, preds, labels=list(range(n_classes)))

        fold_histories.append(hist)
        fold_scores.append(hist["val_bacc"][-1])
        print(f"fold {fold+1:2d}/{n_splits} | final bal acc {hist['val_bacc'][-1]:.4f} "
              f"| AUPRC {hist['val_auprc'][-1]:.4f}")

    # ---- plots ----
    tl = np.array([h["train_loss"] for h in fold_histories])
    vl = np.array([h["val_loss"] for h in fold_histories])
    vbacc = np.array([h["val_bacc"] for h in fold_histories])
    vauprc = np.array([h["val_auprc"] for h in fold_histories], dtype=float)
    ep = np.arange(1, epochs + 1)

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))

    def band(a_, axis, label, color):
        m, s = np.nanmean(a_, 0), np.nanstd(a_, 0)
        axis.plot(ep, m, label=label, color=color)
        axis.fill_between(ep, m - s, m + s, alpha=0.2, color=color)

    band(tl, ax[0], "train loss", "C0")
    band(vl, ax[0], "val loss", "C1")
    ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[0].set_title(f"weighted loss (mean +/- std, {n_splits} folds)")

    band(vbacc, ax[1], "val balanced acc", "C2")
    band(vauprc, ax[1], f"val AUPRC ({class_names[pos_class]})", "C3")
    ax[1].set_xlabel("epoch"); ax[1].set_ylim(0, 1); ax[1].legend()
    ax[1].set_title("validation metrics")

    im = ax[2].imshow(cm_total, cmap="Blues")
    ax[2].set_title("confusion matrix (all folds, last epoch)")
    ax[2].set_xlabel("predicted"); ax[2].set_ylabel("true")
    ax[2].set_xticks(range(n_classes)); ax[2].set_xticklabels(class_names, rotation=45, ha="right")
    ax[2].set_yticks(range(n_classes)); ax[2].set_yticklabels(class_names)
    for i in range(n_classes):
        for j in range(n_classes):
            ax[2].text(j, i, cm_total[i, j], ha="center", va="center",
                       color="white" if cm_total[i, j] > cm_total.max() / 2 else "black")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    plt.tight_layout(); plt.show()

    final_auprc = vauprc[:, -1]
    print(f"\nCV over {n_splits} folds | balanced acc {np.mean(fold_scores):.4f} "
          f"+/- {np.std(fold_scores):.4f} | AUPRC {np.nanmean(final_auprc):.4f} "
          f"+/- {np.nanstd(final_auprc):.4f}")

    return fold_histories, fold_scores, model