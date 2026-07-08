"""
train_kfold.py
==============

Replicates the ten-fold cross-validation training / evaluation setup described in

    U. R. Acharya, S. L. Oh, Y. Hagiwara, J. H. Tan, H. Adeli (2018),
    "Deep convolutional neural network for the automated detection and diagnosis
    of seizure using EEG signals", Computers in Biology and Medicine.

Quick start (from a Jupyter notebook)
-------------------------------------
    from train_kfold import main, plot_learning_curves, plot_confusion_matrix

    results = main(X, y)                 # runs 10-fold CV, returns a dict
    plot_learning_curves(results)        # train/val loss + accuracy vs epoch
    plot_confusion_matrix(results)       # aggregated 3x3 confusion matrix

    # Faster smoke run while wiring things up:
    # results = main(X, y, n_folds=2, epochs=10)

What the paper specifies (and how it is implemented here)
---------------------------------------------------------
* 13-layer 1-D CNN (5 conv + 5 max-pool + 3 fully-connected).  ->  Arch.Model
* Ten-fold cross-validation: 9/10 train, 1/10 test, rotated ten times; the
  reported metrics are averaged / pooled over the ten folds.
* Data allocation per fold (Figure 5 of the paper):

      All EEG segments
        |-- 90% training ---+-- 70% train
        |                   +-- 30% validation   (checked every epoch)
        +-- 10% test

* Optimiser: back-propagation (SGD) with lr x = 1e-3, momentum m = 0.3,
  regularization lambda = 0.7 (L2 weight decay, eq. 1 of the paper).
* Batch size = 3, 150 epochs, softmax + cross-entropy loss.

Input
-----
`main(X, y)` expects
    X : float array (N, L) -- one EEG segment per row, ALREADY per-segment
                              z-score normalised (as in the preprocessing nb).
    y : int array   (N,)   -- class index (0 = normal, 1 = interictal/preictal,
                              2 = ictal/seizure).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split

from Arch import Model


# --------------------------------------------------------------------------- #
# Hyper-parameters taken directly from the paper
# --------------------------------------------------------------------------- #
N_FOLDS       = 10        # ten-fold cross-validation
EPOCHS        = 150       # "A total of 150 epochs of training were run"
BATCH_SIZE    = 3         # "a batch size of 3 is employed"
LEARNING_RATE = 1e-3      # keep the paper's LR...
MOMENTUM      = 0.9       # ...but 0.3 is too weak; 0.9 drives the small conv-layer grads
WEIGHT_DECAY  = 1e-4      # NOT 0.7 — that maps to a ~1000x-too-strong L2 penalty in PyTorch
VAL_FRACTION  = 0.30      # 30% of the 90% training portion -> validation
NORMAL_CLASS  = 0         # index of the "normal" (negative) class
RANDOM_SEED   = 123

CLASS_NAMES = ["Normal", "Interictal", "Ictal"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_loader(X, y, batch_size, shuffle):
    """Build a DataLoader of (N, 1, L) float tensors + int64 labels."""
    xt = torch.as_tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, L)
    yt = torch.as_tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def _evaluate(model, loader, device, n_classes, criterion=None):
    """Return (confusion_matrix, accuracy, mean_loss) over a loader."""
    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    loss_sum, n_seen = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        if criterion is not None:
            loss_sum += criterion(logits, yb).item() * len(yb)
            n_seen += len(yb)
        preds = logits.argmax(dim=1).cpu().numpy()
        for t, p in zip(yb.cpu().numpy(), preds):
            cm[t, p] += 1
    acc = np.trace(cm) / cm.sum() if cm.sum() else 0.0
    mean_loss = loss_sum / n_seen if n_seen else float("nan")
    return cm, acc, mean_loss


def _train_one_fold(X_tr, y_tr, X_val, y_val, device, n_classes,
                    epochs, batch_size, verbose_every=25):
    """Train a fresh model on the 70% split, keeping best-validation weights.

    Returns (model, history) where history has per-epoch lists:
        train_loss, train_acc, val_loss, val_acc
    """
    model = Model(num_features=X_tr.shape[1], random_seed=RANDOM_SEED).to(device)
    optimiser = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE, momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,       # L2 term = lambda in the paper's eq. (1)
    )
    criterion = nn.CrossEntropyLoss()    # softmax (Layer 13) + cross-entropy

    train_loader = _make_loader(X_tr, y_tr, batch_size, shuffle=True)
    val_loader   = _make_loader(X_val, y_val, batch_size, shuffle=False)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimiser.step()

            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)

        train_loss = loss_sum / total
        train_acc = correct / total
        _, val_acc, val_loss = _evaluate(model, val_loader, device, n_classes, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc >= best_val_acc:                     # keep best-generalising weights
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose_every and (epoch % verbose_every == 0 or epoch == 1):
            print(f"      epoch {epoch:3d}/{epochs}  "
                  f"train_loss={train_loss:6.3f} train_acc={train_acc:5.3f}  "
                  f"val_loss={val_loss:6.3f} val_acc={val_acc:5.3f}")

    model.load_state_dict(best_state)
    return model, history


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _metrics_from_confusion(cm, negative_class=NORMAL_CLASS):
    """Compute the metrics reported in the paper from an aggregated 3x3 matrix.

    Rows = true class, columns = predicted class.
    """
    cm = np.asarray(cm, dtype=np.float64)
    n_classes = cm.shape[0]
    total = cm.sum()
    overall_acc = np.trace(cm) / total if total else 0.0

    per_class = []                                       # one-vs-rest
    for c in range(n_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        per_class.append({
            "accuracy":    (tp + tn) / total if total else 0.0,
            "ppv":         tp / (tp + fp) if (tp + fp) else 0.0,
            "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
            "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        })

    pos = [c for c in range(n_classes) if c != negative_class]  # normal vs abnormal
    neg = negative_class
    tp = cm[np.ix_(pos, pos)].sum()
    fn = cm[np.ix_(pos, [neg])].sum()
    fp = cm[np.ix_([neg], pos)].sum()
    tn = cm[neg, neg]
    binary = {
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "ppv":         tp / (tp + fp) if (tp + fp) else 0.0,
        "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
    }
    return overall_acc, per_class, binary


# --------------------------------------------------------------------------- #
# Plotting (call these from a notebook cell)
# --------------------------------------------------------------------------- #
def plot_learning_curves(results, save_path=None, figsize=(11, 4.2)):
    """Plot loss and accuracy vs epoch, averaged over folds (mean +/- std band).

    Parameters
    ----------
    results   : dict returned by ``main`` (must contain "history"), or a list of
                per-fold history dicts.
    save_path : optional path to also save the figure (e.g. "curves.png").
    """
    import matplotlib.pyplot as plt

    histories = results["history"] if isinstance(results, dict) else results
    if not histories:
        raise ValueError("No training history found to plot.")

    def stack(key):
        return np.array([h[key] for h in histories], dtype=float)  # (folds, epochs)

    epochs = np.arange(1, stack("train_loss").shape[1] + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=figsize)

    for ax, (tr_key, va_key, ylabel, title) in zip(
        (ax_loss, ax_acc),
        [("train_loss", "val_loss", "Loss", "Loss vs epoch"),
         ("train_acc", "val_acc", "Accuracy", "Accuracy vs epoch")],
    ):
        for key, label, colour in ((tr_key, "train", "#1f77b4"),
                                   (va_key, "validation", "#d62728")):
            data = stack(key)
            mean, std = data.mean(0), data.std(0)
            ax.plot(epochs, mean, color=colour, label=label)
            if len(histories) > 1:
                ax.fill_between(epochs, mean - std, mean + std, color=colour, alpha=0.15)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    n = len(histories)
    fig.suptitle(f"Learning curves (mean{' +/- std' if n > 1 else ''} over {n} fold(s))")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved learning curves to {save_path}")
    plt.show()
    return fig


def plot_confusion_matrix(results, class_names=None, normalize=False,
                          save_path=None, figsize=(5.4, 4.8)):
    """Heat-map of the aggregated confusion matrix.

    Parameters
    ----------
    results     : dict returned by ``main`` (uses "confusion_matrix"), or a raw
                  2-D array / list.
    class_names : labels for the axes (defaults to Normal / Interictal / Ictal).
    normalize   : if True, show row-normalised proportions instead of counts.
    """
    import matplotlib.pyplot as plt

    cm = results["confusion_matrix"] if isinstance(results, dict) else results
    cm = np.asarray(cm)
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = (CLASS_NAMES[:n_classes]
                       if n_classes <= len(CLASS_NAMES)
                       else [str(i) for i in range(n_classes)])

    disp = cm.astype(float)
    if normalize:
        row_sums = disp.sum(axis=1, keepdims=True)
        disp = np.divide(disp, row_sums, out=np.zeros_like(disp), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(disp, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n_classes), labels=class_names, rotation=45, ha="right")
    ax.set_yticks(range(n_classes), labels=class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix" + (" (row-normalised)" if normalize else ""))

    thresh = disp.max() / 2.0 if disp.size else 0
    for i in range(n_classes):
        for j in range(n_classes):
            txt = f"{disp[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if disp[i, j] > thresh else "black")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")
    plt.show()
    return fig


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main(X, y,
         n_folds=N_FOLDS,
         epochs=EPOCHS,
         batch_size=BATCH_SIZE,
         val_fraction=VAL_FRACTION,
         negative_class=NORMAL_CLASS,
         random_seed=RANDOM_SEED,
         device=None,
         verbose_every=25):
    """Run ten-fold cross-validation; print and return the averaged results.

    Returns a dict with keys:
        confusion_matrix, overall_accuracy, fold_accuracies,
        per_class, binary, history (list of per-fold epoch histories).
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    n_classes = int(y.max()) + 1

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | samples: {len(X)} | length: {X.shape[1]} | "
          f"classes: {n_classes}")

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    total_cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    fold_accuracies, histories = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train90, y_train90 = X[train_idx], y[train_idx]
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train90, y_train90, test_size=val_fraction,
            stratify=y_train90, random_state=random_seed,
        )
        X_test, y_test = X[test_idx], y[test_idx]

        print(f"\nFold {fold}/{n_folds}  "
              f"(train={len(X_tr)}, val={len(X_val)}, test={len(X_test)})")

        model, history = _train_one_fold(
            X_tr, y_tr, X_val, y_val, device, n_classes,
            epochs=epochs, batch_size=batch_size, verbose_every=verbose_every,
        )
        histories.append(history)

        test_loader = _make_loader(X_test, y_test, batch_size, shuffle=False)
        fold_cm, fold_acc, _ = _evaluate(model, test_loader, device, n_classes)
        total_cm += fold_cm
        fold_accuracies.append(fold_acc)
        print(f"   -> fold test accuracy: {fold_acc*100:5.2f}%")

    overall_acc, per_class, binary = _metrics_from_confusion(total_cm, negative_class)

    print("\n" + "=" * 62)
    print("TEN-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 62)
    print(f"Mean fold accuracy : {np.mean(fold_accuracies)*100:5.2f}% "
          f"(+/- {np.std(fold_accuracies)*100:4.2f})")
    print(f"Pooled 3-class acc : {overall_acc*100:5.2f}%")

    print("\nAggregated confusion matrix (rows = true, cols = predicted):")
    print("            " + "".join(f"pred{c:<6d}" for c in range(n_classes)))
    for r in range(n_classes):
        print(f"   true{r:<3d}" + "".join(f"{total_cm[r, c]:>10d}"
                                          for c in range(n_classes)))

    print("\nPer-class (one-vs-rest):")
    print("   class   acc(%)   ppv(%)   sens(%)   spec(%)")
    for c, m in enumerate(per_class):
        print(f"     {c}    {m['accuracy']*100:6.2f}  {m['ppv']*100:6.2f}   "
              f"{m['sensitivity']*100:6.2f}    {m['specificity']*100:6.2f}")

    print("\nBinary (normal vs. abnormal) -- paper's Table 3 convention:")
    print(f"   tp={binary['tp']}  tn={binary['tn']}  "
          f"fp={binary['fp']}  fn={binary['fn']}")
    print(f"   accuracy    : {overall_acc*100:5.2f}%   (3-class overall)")
    print(f"   ppv         : {binary['ppv']*100:5.2f}%")
    print(f"   sensitivity : {binary['sensitivity']*100:5.2f}%")
    print(f"   specificity : {binary['specificity']*100:5.2f}%")
    print("=" * 62)

    return {
        "confusion_matrix": total_cm,
        "overall_accuracy": overall_acc,
        "fold_accuracies": fold_accuracies,
        "per_class": per_class,
        "binary": binary,
        "history": histories,
    }


if __name__ == "__main__":
    # Tiny synthetic smoke test so the file can be run standalone.
    rng = np.random.default_rng(0)
    N, L = 60, 4097
    Xd = rng.standard_normal((N, L)).astype(np.float32)
    Xd = (Xd - Xd.mean(1, keepdims=True)) / (Xd.std(1, keepdims=True) + 1e-8)
    yd = rng.integers(0, 3, size=N)
    res = main(Xd, yd, n_folds=2, epochs=3, verbose_every=0)
    plot_learning_curves(res, save_path="smoke_curves.png")
    plot_confusion_matrix(res, save_path="smoke_cm.png")