"""
visualizer.py
-------------
Generates publication-quality plots:
  - ROC curves (all models on one axes)
  - Confusion matrix heatmap
  - Feature importance bar chart
  - Model comparison bar chart
  - Cross-validation comparison
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# ── Style ──────────────────────────────────────────────────────────────────────

PALETTE = {
    "SVM":                 "#185FA5",
    "Logistic Regression": "#0F6E56",
    "Random Forest":       "#534AB7",
    "XGBoost":             "#BA7517",
}

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})


def _save(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


# ── ROC Curves ────────────────────────────────────────────────────────────────

def plot_roc_curves(eval_results, dataset_name, output_dir="results"):
    """Plot all four ROC curves on a single axes."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.50)")

    for name, metrics in eval_results.items():
        auc = metrics["auc"]
        ax.plot(
            metrics["fpr"],
            metrics["tpr"],
            color=PALETTE.get(name, "#888"),
            lw=2,
            label=f"{name} (AUC={auc:.3f})",
        )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {dataset_name}")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    return _save(fig, output_dir, f"roc_{dataset_name.lower().replace(' ', '_')}.png")


# ── Confusion Matrices ────────────────────────────────────────────────────────

def plot_confusion_matrices(eval_results, dataset_name,
                             class_labels=("Negative", "Positive"),
                             output_dir="results"):
    """Plot a 2×2 grid of confusion matrices, one per model."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"Confusion Matrices — {dataset_name}", fontsize=13, y=1.01)

    for ax, (name, metrics) in zip(axes.flat, eval_results.items()):
        cm = metrics["confusion_matrix"]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}\nAcc={metrics['accuracy']}%", fontsize=10)

    fig.tight_layout()
    return _save(fig, output_dir, f"cm_{dataset_name.lower().replace(' ', '_')}.png")


# ── Feature Importance ────────────────────────────────────────────────────────

def plot_feature_importance(importance_pairs, model_name, dataset_name,
                             top_n=10, output_dir="results"):
    """
    Horizontal bar chart of feature importances.

    importance_pairs: list of (feature_name, importance) sorted desc
    """
    if importance_pairs is None:
        return None

    pairs = importance_pairs[:top_n]
    features, values = zip(*pairs)

    fig, ax = plt.subplots(figsize=(7, max(3, len(features) * 0.45)))
    y_pos = np.arange(len(features))

    ax.barh(y_pos, values, color=PALETTE.get(model_name, "#888"), alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {model_name} on {dataset_name}")

    for i, v in enumerate(values):
        ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=8)

    fig.tight_layout()
    fname = f"fi_{model_name.lower().replace(' ', '_')}_{dataset_name.lower().replace(' ', '_')}.png"
    return _save(fig, output_dir, fname)


# ── Model Comparison Bar Chart ────────────────────────────────────────────────

def plot_model_comparison(eval_results, dataset_name, output_dir="results"):
    """
    Grouped bar chart: Accuracy, Precision, Recall, F1, AUC×100 per model.
    """
    metrics_keys = ["accuracy", "precision", "recall", "f1", "auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC×100"]
    model_names = list(eval_results.keys())

    x = np.arange(len(metrics_keys))
    n = len(model_names)
    width = 0.18
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, name in enumerate(model_names):
        m = eval_results[name]
        vals = [
            m["accuracy"], m["precision"], m["recall"], m["f1"],
            round(m["auc"] * 100, 2),
        ]
        bars = ax.bar(
            x + offsets[i], vals,
            width, label=name,
            color=PALETTE.get(name, "#888"), alpha=0.85,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=7, rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score (%)")
    ax.set_ylim([50, 105])
    ax.set_title(f"Model Comparison — {dataset_name}")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    return _save(fig, output_dir, f"comparison_{dataset_name.lower().replace(' ', '_')}.png")


# ── CV Comparison ─────────────────────────────────────────────────────────────

def plot_cv_comparison(cv_results, dataset_name, output_dir="results"):
    """
    Bar chart with error bars showing CV accuracy ± std for all models.
    """
    names  = list(cv_results.keys())
    means  = [cv_results[n]["accuracy"]["mean"] for n in names]
    stds   = [cv_results[n]["accuracy"]["std"]  for n in names]
    colors = [PALETTE.get(n, "#888") for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, means, yerr=stds, color=colors, alpha=0.85,
                  capsize=5, error_kw={"elinewidth": 1.5})

    for bar, m, s in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s + 0.5,
            f"{m:.1f}%",
            ha="center", fontsize=9,
        )

    ax.set_ylabel("CV Accuracy (%)")
    ax.set_ylim([max(0, min(means) - 10), min(105, max(means) + 10)])
    ax.set_title(f"5-Fold CV Accuracy — {dataset_name}")
    fig.tight_layout()

    return _save(fig, output_dir, f"cv_{dataset_name.lower().replace(' ', '_')}.png")


# ── Full dashboard (one figure per dataset) ───────────────────────────────────

def plot_dashboard(eval_results, dataset_name, output_dir="results"):
    """
    Compact 2×2 summary dashboard:
      [ROC curves]      [Model comparison bars]
      [Best CM]         [Feature importance of best model (if available)]
    """
    best_name = max(eval_results, key=lambda n: eval_results[n]["accuracy"])
    best = eval_results[best_name]

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # — ROC curves
    ax_roc = fig.add_subplot(gs[0, 0])
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
    for name, metrics in eval_results.items():
        ax_roc.plot(metrics["fpr"], metrics["tpr"],
                    color=PALETTE.get(name, "#888"), lw=2,
                    label=f"{name} ({metrics['auc']:.3f})")
    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
    ax_roc.set_title("ROC Curves"); ax_roc.legend(fontsize=8)

    # — Model comparison
    ax_cmp = fig.add_subplot(gs[0, 1])
    metrics_k = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metrics_k))
    n = len(eval_results)
    width = 0.18
    offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width
    for i, (name, m) in enumerate(eval_results.items()):
        vals = [m["accuracy"], m["precision"], m["recall"], m["f1"]]
        ax_cmp.bar(x + offsets[i], vals, width,
                   label=name, color=PALETTE.get(name, "#888"), alpha=0.85)
    ax_cmp.set_xticks(x)
    ax_cmp.set_xticklabels(["Acc", "Prec", "Rec", "F1"])
    ax_cmp.set_ylim([50, 105]); ax_cmp.set_ylabel("Score (%)")
    ax_cmp.set_title("Metric Comparison"); ax_cmp.legend(fontsize=8)

    # — Best model confusion matrix
    ax_cm = fig.add_subplot(gs[1, 0])
    sns.heatmap(best["confusion_matrix"], annot=True, fmt="d",
                cmap="Blues", ax=ax_cm, cbar=False)
    ax_cm.set_title(f"Confusion Matrix — {best_name}")
    ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")

    # — Dataset info text
    ax_txt = fig.add_subplot(gs[1, 1])
    ax_txt.axis("off")
    summary = "\n".join([
        f"Dataset: {dataset_name}",
        "",
        f"{'Model':<22} {'Acc':>7} {'AUC':>7}",
        "─" * 38,
    ] + [
        f"{n:<22} {m['accuracy']:>6.1f}% {m['auc']:>7.3f}"
        for n, m in sorted(eval_results.items(),
                           key=lambda x: x[1]["accuracy"], reverse=True)
    ])
    ax_txt.text(0.05, 0.95, summary, transform=ax_txt.transAxes,
                va="top", fontsize=9.5, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", alpha=0.8))

    fig.suptitle(f"Disease Prediction Dashboard — {dataset_name}", fontsize=14, y=1.01)
    return _save(fig, output_dir, f"dashboard_{dataset_name.lower().replace(' ', '_')}.png")
