from pathlib import Path
from typing import Dict, List
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_folds(summaryPath: Path) -> List[Dict]:
    with open(summaryPath, "r") as file:
        data = json.load(file)

    # One row per fold: the scores of that fold's best epoch
    return [{"Name": data["experiment"], "Fold": f["fold"], "bacc": f["bacc"], "auprc": f["auprc"]}
            for f in data["folds_best"]]


def load_sensitivity(summaryPath: Path) -> List[Dict]:
    with open(summaryPath, "r") as file:
        data = json.load(file)

    folds = data["folds_best"]
    if all("sensitivity" in f for f in folds):
        # One value per fold, so the plot gets a mean and a std error bar
        return [{"Name": data["experiment"], "Metric": "sensitivity", "Score": f["sensitivity"]}
                for f in folds]

    # Older runs only saved sensitivity pooled over all folds: a single value, no error bar
    return [{"Name": data["experiment"], "Metric": "sensitivity",
             "Score": data["pooled_best"]["sensitivity"]}]


root = Path("results")
rows = []
sensRows = []

print("\033[1m" + "file found:" + "\033[0m")
for path in root.rglob("summary.json"):
    print("  ", path.relative_to(root))
    rows.extend(load_folds(path))
    sensRows.extend(load_sensitivity(path))

if not rows:
    raise SystemExit("No summary.json files found.")

# Seaborn wants "long" data for grouped bars: one row per (Name, fold, Metric, Score)
folds = pd.DataFrame(rows)
df = folds.melt(id_vars=["Name", "Fold"], value_vars=["bacc", "auprc"],
                var_name="Metric", value_name="Score")
nExp = df["Name"].nunique()

# Experiment order for both plots: highest mean auprc first
meanOrder = folds.groupby("Name")["auprc"].mean().sort_values(ascending=False).index

# Three plots stacked vertically; same experiment order, so only the bottom one shows the names
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(max(8, nExp * 1.7), 10), sharex=True,
                                    layout="constrained")

# Mean over folds of each fold's best epoch, error bar = std over folds
sns.barplot(data=pd.concat([df, pd.DataFrame(sensRows)]), x="Name", y="Score", hue="Metric",
            order=meanOrder, errorbar="sd", ax=ax1)
ax1.set_title("Average of best epochs over folds (± std)")

# One bar per fold: score of that fold's best epoch
for ax, metric, title in ((ax2, "bacc", "Balanced accuracy"), (ax3, "auprc", "AUPRC")):
    sns.barplot(data=folds, x="Name", y=metric, hue="Fold", order=meanOrder, errorbar=None,
                legend="full", ax=ax)
    ax.set_title(f"{title} per fold (best epoch)")
    ax.set_ylim(0, 1.01)

for ax in (ax1, ax2, ax3):
    ax.set_xlabel("")
    ax.legend(title=ax.get_legend().get_title().get_text() if ax is not ax1 else "",
              loc="upper left", bbox_to_anchor=(1, 1))  # outside, so no bar is hidden

# Value labels only on the top plot: the per-fold ones have too many bars to read them
# Written inside the bar, near its base, so they never hit the error bars or the title
for container in ax1.containers:
    for bar in container:
        ax1.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width() / 2, 0),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                     fontsize=8, rotation=90, color="white")
ax1.set_ylim(0, 1.01)

# Anchor rotated names at their right end so long neighbours don't overlap
plt.setp(ax3.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

plt.savefig("results/experiments_comparison.png")
plt.show()
