"""Generate result plots for MRKR training experiments.

Reads test_metrics.json and history.json from each model run
and produces PNG figures ready for the report.

Outputs (saved to --output_dir):
  01_confusion_matrices.png   -- 3-panel confusion matrix
  02_per_grade_f1.png         -- per-grade F1 grouped bar chart
  03_grade1_comparison.png    -- grade-1 metrics comparison
  04_training_curves.png      -- loss and balanced accuracy curves
  05_model_summary.png        -- overall metrics comparison bar chart

Usage:
  python mrkr_plot_results.py \
      --runs_dir   /home/tm922/mrkr_klg/runs \
      --output_dir /home/tm922/mrkr_klg/results
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
DPI = 150

MODELS = ["resnet50", "densenet121", "efficientnet_b0"]
LABELS = {
    "resnet50":        "ResNet50",
    "densenet121":     "DenseNet121",
    "efficientnet_b0": "EfficientNet-B0",
}
COLORS = ["#4878d0", "#ee854a", "#6acc65"]
KL_LABELS = ["KL 0", "KL 1", "KL 2", "KL 3", "KL 4"]


def load_metrics(runs_dir, model):
    with open(os.path.join(runs_dir, model, "test_metrics.json")) as f:
        return json.load(f)


def load_history(runs_dir, model):
    with open(os.path.join(runs_dir, model, "history.json")) as f:
        data = json.load(f)
    return pd.DataFrame(data["history"])


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {os.path.basename(path)}")


# ── 1. Confusion matrices ─────────────────────────────────────────────────────

def plot_confusion_matrices(runs_dir, output_dir):
    print("\n[1] Confusion matrices")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, model, color in zip(axes, MODELS, COLORS):
        m  = load_metrics(runs_dir, model)
        cm = np.array(m["confusion_matrix"])

        # Normalise by row (true label)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(
            cm_norm, annot=cm, fmt="d", ax=ax,
            cmap="Blues", cbar=True,
            xticklabels=KL_LABELS,
            yticklabels=KL_LABELS,
            annot_kws={"size": 9},
        )
        ax.set_title(f"{LABELS[model]}\n"
                     f"Balanced Acc: {m['balanced_accuracy']:.3f}",
                     fontsize=11)
        ax.set_xlabel("Predicted Grade")
        ax.set_ylabel("True Grade")

    fig.suptitle("Confusion Matrices — MRKR Test Set",
                 fontsize=13, fontweight="bold", y=1.02)
    save_fig(fig, os.path.join(output_dir, "01_confusion_matrices.png"))


# ── 2. Per-grade F1 bar chart ─────────────────────────────────────────────────

def plot_per_grade_f1(runs_dir, output_dir):
    print("\n[2] Per-grade F1 comparison")

    grades = ["0", "1", "2", "3", "4"]
    x      = np.arange(len(grades))
    width  = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (model, color) in enumerate(zip(MODELS, COLORS)):
        m  = load_metrics(runs_dir, model)
        cr = m["classification_report"]
        f1_scores = [cr[g]["f1-score"] for g in grades]
        bars = ax.bar(x + i * width, f1_scores, width,
                      label=LABELS[model], color=color,
                      edgecolor="white")
        for bar, val in zip(bars, f1_scores):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=8)

    ax.set_xlabel("KL Grade")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Grade F1 Score — All Three Models\n(MRKR Test Set)",
                 fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(KL_LABELS)
    ax.set_ylim(0, 0.95)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.4,
               label="F1 = 0.5 reference")
    ax.legend()

    save_fig(fig, os.path.join(output_dir, "02_per_grade_f1.png"))


# ── 3. Grade-1 metrics comparison ────────────────────────────────────────────

def plot_grade1_comparison(runs_dir, output_dir):
    print("\n[3] Grade-1 comparison")

    metrics_to_plot = ["precision", "recall", "f1"]
    metric_labels   = ["Precision", "Recall", "F1"]
    x     = np.arange(len(metrics_to_plot))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (model, color) in enumerate(zip(MODELS, COLORS)):
        m  = load_metrics(runs_dir, model)
        g1 = m["grade1_one_vs_rest"]
        values = [g1["precision"], g1["recall"], g1["f1"]]
        bars = ax.bar(x + i * width, values, width,
                      label=LABELS[model], color=color,
                      edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=9)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("KL Grade 1 Performance — One-vs-Rest\n"
                 "(MRKR Test Set — clinically most important grade)",
                 fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 0.75)
    ax.axhline(y=0.64, color="red", linestyle="--", alpha=0.5,
               label="Literature benchmark (Zhao et al. 2024)")
    ax.legend()

    save_fig(fig, os.path.join(output_dir, "03_grade1_comparison.png"))


# ── 4. Training curves ────────────────────────────────────────────────────────

def plot_training_curves(runs_dir, output_dir):
    print("\n[4] Training curves")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model, color in zip(MODELS, COLORS):
        hist = load_history(runs_dir, model)
        epochs = hist["epoch"]

        axes[0].plot(epochs, hist["train_loss"],
                     color=color, linestyle="-",
                     label=f"{LABELS[model]} train", alpha=0.8)
        axes[0].plot(epochs, hist["val_loss"],
                     color=color, linestyle="--",
                     label=f"{LABELS[model]} val", alpha=0.6)

        axes[1].plot(epochs, hist["val_balanced_acc"],
                     color=color, linestyle="-",
                     label=LABELS[model])

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss", fontweight="bold")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Balanced Accuracy")
    axes[1].set_title("Validation Balanced Accuracy", fontweight="bold")
    axes[1].legend()

    fig.suptitle("Training History — All Three Models",
                 fontsize=13, fontweight="bold")
    save_fig(fig, os.path.join(output_dir, "04_training_curves.png"))


# ── 5. Overall model summary ──────────────────────────────────────────────────

def plot_model_summary(runs_dir, output_dir):
    print("\n[5] Overall model summary")

    metric_keys   = ["balanced_accuracy", "macro_f1", "mae_grade"]
    metric_labels = ["Balanced Accuracy", "Macro F1", "MAE Grade"]
    x     = np.arange(len(metric_keys))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left — balanced acc, macro f1
    for i, (model, color) in enumerate(zip(MODELS, COLORS)):
        m = load_metrics(runs_dir, model)
        values = [m["balanced_accuracy"], m["macro_f1"]]
        bars = axes[0].bar(np.arange(2) + i * width, values, width,
                           label=LABELS[model], color=color,
                           edgecolor="white")
        for bar, val in zip(bars, values):
            axes[0].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.003,
                         f"{val:.3f}", ha="center", va="bottom",
                         fontsize=9)

    axes[0].set_xticks(np.arange(2) + width)
    axes[0].set_xticklabels(["Balanced\nAccuracy", "Macro F1"])
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 0.75)
    axes[0].set_title("Overall Performance", fontweight="bold")
    axes[0].legend()

    # Right — grade 1 recall
    recalls = []
    for model in MODELS:
        m = load_metrics(runs_dir, model)
        recalls.append(m["grade1_one_vs_rest"]["recall"])

    bars = axes[1].bar(
        [LABELS[m] for m in MODELS], recalls,
        color=COLORS, edgecolor="white"
    )
    for bar, val in zip(bars, recalls):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")

    axes[1].axhline(y=0.64, color="red", linestyle="--", alpha=0.5,
                    label="Literature KL1 sensitivity\n(Zhao et al. 2024)")
    axes[1].set_ylabel("Recall")
    axes[1].set_ylim(0, 0.75)
    axes[1].set_title("Grade-1 Recall — Key Clinical Metric",
                       fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].tick_params(axis="x", rotation=10)

    fig.suptitle("Model Comparison Summary — MRKR Test Set",
                 fontsize=13, fontweight="bold")
    save_fig(fig, os.path.join(output_dir, "05_model_summary.png"))


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir",   type=str,
                   default="/home/tm922/mrkr_klg/runs")
    p.add_argument("--output_dir", type=str,
                   default="/home/tm922/mrkr_klg/results")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("MRKR — Results Plotting")
    print("=" * 60)
    print(f"  Runs dir   : {args.runs_dir}")
    print(f"  Output dir : {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    plot_confusion_matrices(args.runs_dir, args.output_dir)
    plot_per_grade_f1(args.runs_dir, args.output_dir)
    plot_grade1_comparison(args.runs_dir, args.output_dir)
    plot_training_curves(args.runs_dir, args.output_dir)
    plot_model_summary(args.runs_dir, args.output_dir)

    print("\n" + "=" * 60)
    print(f"All plots saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
