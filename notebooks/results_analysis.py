"""
02_results_analysis.py
Results analysis and figure generation for the 2x2 KL grading experiment.

Generates all figures and tables needed for the report:
    01_learning_curves.png        -- train/val loss per epoch per config
    02_per_grade_recall.png       -- per-grade recall comparison bar chart
    03_kl1_recall_comparison.png  -- KL1 recall CE vs CORAL with seed spread
    04_qwk_comparison.png         -- QWK comparison across configurations
    05_confusion_matrices.png     -- confusion matrices for best seed per config
    06_metrics_summary.png        -- summary table as figure
    07_localisation_scores.png    -- Grad-CAM localisation score comparison

Usage
-----
    python notebooks/02_results_analysis.py \
        --results_dir /home/tm922/knee-oa-kl-grading/results \
        --output_dir  /home/tm922/knee-oa-kl-grading/results/figures
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats

# ── Style ─────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":      150,
    "savefig.dpi":     150,
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "figure.facecolor": "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

DPI     = 150
GRADES  = [0, 1, 2, 3, 4]
SEEDS   = [42, 123, 456]

# Configuration display names and colours
CONFIGS = {
    "resnet50_ce":       {"label": "ResNet50\nCE",       "colour": "#4878d0"},
    "resnet50_coral":    {"label": "ResNet50\nCORAL",    "colour": "#ee854a"},
    "efficientnet_ce":   {"label": "EffNet-B0\nCE",      "colour": "#6acc65"},
    "efficientnet_coral":{"label": "EffNet-B0\nCORAL",   "colour": "#d65f5f"},
}

KL_GRADE_NAMES = ["Normal\n(KL0)", "Doubtful\n(KL1)", "Mild\n(KL2)",
                   "Moderate\n(KL3)", "Severe\n(KL4)"]


# ─────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────

def load_eval_results(metrics_dir: Path) -> dict:
    """Load all evaluation JSON files. Returns dict keyed by run_name."""
    results = {}
    for f in sorted(metrics_dir.glob("*_eval.json")):
        with open(f) as fp:
            r = json.load(fp)
        results[r["run_name"]] = r
    return results


def load_history(metrics_dir: Path, run_name: str) -> dict:
    """Load training history JSON for one run."""
    path = metrics_dir / f"{run_name}_history.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_config_seeds(results: dict, arch: str, loss: str) -> list:
    """Get all seed results for one configuration."""
    runs = []
    for seed in SEEDS:
        key = f"{arch}_{loss}_seed{seed}"
        if key in results:
            runs.append(results[key])
    return runs


def get_best_seed(results: dict, arch: str, loss: str) -> dict:
    """Get the best seed result (lowest val loss) for one configuration."""
    runs = get_config_seeds(results, arch, loss)
    if not runs:
        return None
    return min(runs, key=lambda r: r["best_val_loss"])


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path.name}")


# ─────────────────────────────────────────────────────────
# Figure 1 — Learning curves
# ─────────────────────────────────────────────────────────

def plot_learning_curves(metrics_dir: Path, out: Path) -> None:
    print("\n[1/7] Learning curves")

    configs_list = [
        ("resnet50",     "ce"),
        ("resnet50",     "coral"),
        ("efficientnet", "ce"),
        ("efficientnet", "coral"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Training and Validation Loss per Epoch\n"
        "Best seed per configuration (lowest validation loss)",
        fontsize=13, y=1.01
    )

    for ax, (arch, loss) in zip(axes.flat, configs_list):
        config_name = f"{arch}_{loss}"
        colour      = CONFIGS[config_name]["colour"]
        label       = CONFIGS[config_name]["label"].replace("\n", " ")

        # Find best seed
        best_seed = None
        best_loss = float("inf")
        for seed in SEEDS:
            run_name = f"{arch}_{loss}_seed{seed}"
            config_path = metrics_dir / f"{run_name}_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                if cfg.get("best_val_loss", float("inf")) < best_loss:
                    best_loss = cfg["best_val_loss"]
                    best_seed = seed

        if best_seed is None:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(label)
            continue

        run_name = f"{arch}_{loss}_seed{best_seed}"
        history  = load_history(metrics_dir, run_name)

        if history is None:
            ax.text(0.5, 0.5, "History not found",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_title(label)
            continue

        epochs     = range(1, len(history["train_loss"]) + 1)
        train_loss = history["train_loss"]
        val_loss   = history["val_loss"]
        best_val_epoch = val_loss.index(min(val_loss)) + 1

        ax.plot(epochs, train_loss, colour, linewidth=2,
                label="Train loss")
        ax.plot(epochs, val_loss, colour, linewidth=2,
                linestyle="--", alpha=0.7, label="Val loss")
        ax.axvline(best_val_epoch, color="red", linestyle=":",
                   linewidth=1.5,
                   label=f"Best val epoch {best_val_epoch}")
        ax.set_title(f"{label}  (seed={best_seed})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)

    plt.tight_layout()
    save(fig, out / "01_learning_curves.png")


# ─────────────────────────────────────────────────────────
# Figure 2 — Per-grade recall comparison
# ─────────────────────────────────────────────────────────

def plot_per_grade_recall(results: dict, out: Path) -> None:
    print("\n[2/7] Per-grade recall comparison")

    configs_list = [
        ("resnet50",     "ce"),
        ("resnet50",     "coral"),
        ("efficientnet", "ce"),
        ("efficientnet", "coral"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)
    fig.suptitle(
        "Per-Grade Recall Across Configurations\n"
        "Mean ± SD across 3 seeds  |  KL1 highlighted as primary outcome",
        fontsize=13, y=1.02
    )

    for ax, (arch, loss) in zip(axes, configs_list):
        config_name = f"{arch}_{loss}"
        colour      = CONFIGS[config_name]["colour"]
        label       = CONFIGS[config_name]["label"].replace("\n", " ")
        runs        = get_config_seeds(results, arch, loss)

        if not runs:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        grade_recalls = {g: [] for g in GRADES}
        for r in runs:
            for g in GRADES:
                grade_recalls[g].append(r[f"recall_kl{g}"])

        means = [np.mean(grade_recalls[g]) for g in GRADES]
        stds  = [np.std(grade_recalls[g])  for g in GRADES]

        bar_colours = ["#d65f5f" if g == 1 else colour for g in GRADES]

        bars = ax.bar(KL_GRADE_NAMES, means, color=bar_colours,
                      edgecolor="white", yerr=stds, capsize=4)

        kl1_bar = bars[1]
        ax.text(
            kl1_bar.get_x() + kl1_bar.get_width() / 2,
            kl1_bar.get_height() + stds[1] + 0.02,
            f"{means[1]:.3f}",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#d65f5f"
        )

        ax.set_title(label)
        ax.set_ylabel("Recall" if ax == axes[0] else "")
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis="x", labelsize=8)
        ax.axhline(0.64, color="black", linestyle="--",
                   linewidth=1, alpha=0.5,
                   label="Meta-analytic KL1\nbenchmark (0.64)")
        if ax == axes[-1]:
            ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    save(fig, out / "02_per_grade_recall.png")


# ─────────────────────────────────────────────────────────
# Figure 3 — KL1 recall comparison CE vs CORAL
# ─────────────────────────────────────────────────────────

def plot_kl1_comparison(results: dict, out: Path) -> None:
    print("\n[3/7] KL1 recall CE vs CORAL comparison")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(
        "KL1 Recall: Categorical CE vs Ordinal CORAL\n"
        "Each point = one seed  |  Bar = mean across 3 seeds",
        fontsize=13, y=1.02
    )

    for ax, arch in zip(axes, ["resnet50", "efficientnet"]):
        arch_label = "ResNet50" if arch == "resnet50" else "EfficientNet-B0"

        ce_recalls    = [r["kl1_recall"] for r in
                         get_config_seeds(results, arch, "ce")]
        coral_recalls = [r["kl1_recall"] for r in
                         get_config_seeds(results, arch, "coral")]

        if not ce_recalls or not coral_recalls:
            continue

        ce_mean    = np.mean(ce_recalls)
        coral_mean = np.mean(coral_recalls)
        ce_std     = np.std(ce_recalls)
        coral_std  = np.std(coral_recalls)

        x       = [0, 1]
        means   = [ce_mean, coral_mean]
        stds    = [ce_std,  coral_std]
        colours = [CONFIGS[f"{arch}_ce"]["colour"],
                   CONFIGS[f"{arch}_coral"]["colour"]]

        bars = ax.bar(x, means, color=colours, edgecolor="white",
                      width=0.5, yerr=stds, capsize=6)

        for xi, recalls in zip(x, [ce_recalls, coral_recalls]):
            ax.scatter([xi] * len(recalls), recalls,
                       color="black", zorder=5, s=40, alpha=0.7)

        for bar, mean, std in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.01,
                f"{mean:.3f}",
                ha="center", va="bottom",
                fontsize=11, fontweight="bold"
            )

        if len(ce_recalls) >= 3 and len(coral_recalls) >= 3:
            stat, pval = stats.wilcoxon(ce_recalls, coral_recalls)
            sig = "p<0.05 *" if pval < 0.05 else f"p={pval:.3f}"
            ax.text(0.5, 0.95, f"Wilcoxon: {sig}",
                    transform=ax.transAxes,
                    ha="center", va="top", fontsize=10,
                    color="red" if pval < 0.05 else "black")

        ax.axhline(0.64, color="black", linestyle="--",
                   linewidth=1.5, alpha=0.6,
                   label="Meta-analytic benchmark (0.64)")
        ax.set_xticks(x)
        ax.set_xticklabels(["CE\n(Categorical)", "CORAL\n(Ordinal)"],
                            fontsize=11)
        ax.set_ylabel("KL1 Recall")
        ax.set_ylim(0, 0.85)
        ax.set_title(arch_label)
        ax.legend(fontsize=8)

    plt.tight_layout()
    save(fig, out / "03_kl1_recall_comparison.png")


# ─────────────────────────────────────────────────────────
# Figure 4 — QWK comparison
# ─────────────────────────────────────────────────────────

def plot_qwk_comparison(results: dict, out: Path) -> None:
    print("\n[4/7] QWK comparison")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(
        "Quadratic Weighted Kappa (QWK): CE vs CORAL\n"
        "Each point = one seed  |  Bar = mean across 3 seeds",
        fontsize=13, y=1.02
    )

    for ax, arch in zip(axes, ["resnet50", "efficientnet"]):
        arch_label = "ResNet50" if arch == "resnet50" else "EfficientNet-B0"

        ce_qwk    = [r["qwk"] for r in get_config_seeds(results, arch, "ce")]
        coral_qwk = [r["qwk"] for r in get_config_seeds(results, arch, "coral")]

        if not ce_qwk or not coral_qwk:
            continue

        x       = [0, 1]
        means   = [np.mean(ce_qwk), np.mean(coral_qwk)]
        stds    = [np.std(ce_qwk),  np.std(coral_qwk)]
        colours = [CONFIGS[f"{arch}_ce"]["colour"],
                   CONFIGS[f"{arch}_coral"]["colour"]]

        bars = ax.bar(x, means, color=colours, edgecolor="white",
                      width=0.5, yerr=stds, capsize=6)

        for xi, qwks in zip(x, [ce_qwk, coral_qwk]):
            ax.scatter([xi] * len(qwks), qwks,
                       color="black", zorder=5, s=40, alpha=0.7)

        for bar, mean, std in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.002,
                f"{mean:.4f}",
                ha="center", va="bottom",
                fontsize=11, fontweight="bold"
            )

        if len(ce_qwk) >= 3 and len(coral_qwk) >= 3:
            stat, pval = stats.wilcoxon(ce_qwk, coral_qwk)
            sig = "p<0.05 *" if pval < 0.05 else f"p={pval:.3f}"
            ax.text(0.5, 0.95, f"Wilcoxon: {sig}",
                    transform=ax.transAxes,
                    ha="center", va="top", fontsize=10,
                    color="red" if pval < 0.05 else "black")

        ax.set_xticks(x)
        ax.set_xticklabels(["CE\n(Categorical)", "CORAL\n(Ordinal)"],
                            fontsize=11)
        ax.set_ylabel("QWK")
        ax.set_ylim(0.70, 0.87)
        ax.set_title(arch_label)
        ax.axhline(0.81, color="green", linestyle=":",
                   linewidth=1, alpha=0.5, label="Almost perfect (0.81)")
        ax.axhline(0.61, color="orange", linestyle=":",
                   linewidth=1, alpha=0.5, label="Substantial (0.61)")
        ax.legend(fontsize=7)

    plt.tight_layout()
    save(fig, out / "04_qwk_comparison.png")


# ─────────────────────────────────────────────────────────
# Figure 5 — Confusion matrices
# ─────────────────────────────────────────────────────────

def plot_confusion_matrices(results: dict, out: Path) -> None:
    print("\n[5/7] Confusion matrices")

    configs_list = [
        ("resnet50",     "ce"),
        ("resnet50",     "coral"),
        ("efficientnet", "ce"),
        ("efficientnet", "coral"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        "Confusion Matrices — Best Seed per Configuration (Test Set)\n"
        "Rows = True KL Grade  |  Columns = Predicted KL Grade",
        fontsize=13, y=1.01
    )

    grade_labels = ["KL0", "KL1", "KL2", "KL3", "KL4"]

    for ax, (arch, loss) in zip(axes.flat, configs_list):
        config_name = f"{arch}_{loss}"
        label       = CONFIGS[config_name]["label"].replace("\n", " ")
        best        = get_best_seed(results, arch, loss)

        if best is None:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        cm      = np.array(best["confusion_matrix"])
        cm_norm = cm.astype(float)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm_norm / row_sums

        im = ax.imshow(cm_norm, interpolation="nearest",
                       cmap="Blues", vmin=0, vmax=1)

        for i in range(len(GRADES)):
            for j in range(len(GRADES)):
                count  = cm[i, j]
                pct    = cm_norm[i, j]
                colour = "white" if pct > 0.5 else "black"
                ax.text(j, i, f"{count}\n({pct:.0%})",
                        ha="center", va="center",
                        fontsize=8, color=colour)

        ax.set_xticks(range(len(GRADES)))
        ax.set_yticks(range(len(GRADES)))
        ax.set_xticklabels(grade_labels)
        ax.set_yticklabels(grade_labels)
        ax.set_xlabel("Predicted Grade")
        ax.set_ylabel("True Grade")
        ax.set_title(
            f"{label}  (seed={best['seed']})\n"
            f"QWK={best['qwk']:.4f}  KL1 Recall={best['kl1_recall']:.3f}"
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Recall (row-normalised)")

    plt.tight_layout()
    save(fig, out / "05_confusion_matrices.png")


# ─────────────────────────────────────────────────────────
# Figure 6 — Metrics summary table
# ─────────────────────────────────────────────────────────

def plot_metrics_summary(results: dict, out: Path) -> None:
    print("\n[6/7] Metrics summary table")

    configs_list = [
        ("resnet50",     "ce"),
        ("resnet50",     "coral"),
        ("efficientnet", "ce"),
        ("efficientnet", "coral"),
    ]

    metrics       = ["accuracy", "f1_weighted", "qwk", "macro_auc",
                     "kl1_recall", "auc_kl1"]
    metric_labels = ["Accuracy", "Weighted\nF1", "QWK",
                     "Macro\nAUC", "KL1\nRecall", "KL1\nAUC"]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(
        "Metrics Summary — Mean ± SD Across 3 Seeds\n"
        "Primary outcome: KL1 Recall  |  Field standard: QWK",
        fontsize=13
    )

    n_configs = len(configs_list)
    n_metrics = len(metrics)
    x         = np.arange(n_metrics)
    width     = 0.18

    for i, (arch, loss) in enumerate(configs_list):
        config_name = f"{arch}_{loss}"
        colour      = CONFIGS[config_name]["colour"]
        label       = CONFIGS[config_name]["label"].replace("\n", " ")
        runs        = get_config_seeds(results, arch, loss)

        if not runs:
            continue

        means = []
        stds  = []
        for m in metrics:
            vals = [r[m] for r in runs if r.get(m) is not None]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals)   if vals else 0)

        offset = (i - n_configs / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=label,
               color=colour, edgecolor="white",
               yerr=stds, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=9)
    ax.axvspan(3.5, 4.5, alpha=0.05, color="red")

    plt.tight_layout()
    save(fig, out / "06_metrics_summary.png")


# ─────────────────────────────────────────────────────────
# Figure 7 — Grad-CAM localisation scores
# ─────────────────────────────────────────────────────────

def plot_localisation_scores(gradcam_dir: Path, out: Path) -> None:
    print("\n[7/7] Grad-CAM localisation scores")

    scores_path = gradcam_dir / "audit_scores.json"
    if not scores_path.exists():
        print("  [SKIP] audit_scores.json not found")
        return

    with open(scores_path) as f:
        audit = json.load(f)

    cases   = audit["cases"]
    configs = list(cases[0]["localisation_scores"].keys()) if cases else []

    if not configs:
        print("  [SKIP] No cases found")
        return

    config_scores = {c: [] for c in configs}
    for case in cases:
        for c, score in case["localisation_scores"].items():
            config_scores[c].append(score)

    fig, ax = plt.subplots(figsize=(12, 6))

    x       = np.arange(len(configs))
    means   = [np.mean(config_scores[c]) for c in configs]
    stds    = [np.std(config_scores[c])  for c in configs]
    colours = [CONFIGS.get(c, {}).get("colour", "#aec6e8") for c in configs]
    labels  = [CONFIGS.get(c, {}).get("label", c).replace("\n", " ")
               for c in configs]

    bars = ax.bar(x, means, color=colours, edgecolor="white",
                  yerr=stds, capsize=5)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.01,
            f"{mean:.3f}",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold"
        )

    for i, c in enumerate(configs):
        ax.scatter([i] * len(config_scores[c]), config_scores[c],
                   color="black", zorder=5, s=20, alpha=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(
        f"Localisation Score\n(% activation in central "
        f"{int(audit['centre_crop_ratio']*100)}% crop)"
    )
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Grad-CAM Attention Localisation — KL1 Misclassification Cases\n"
        f"({audit['n_audit_cases']} cases misclassified by all 4 models)\n"
        f"Higher = more activation on joint space region"
    )

    plt.tight_layout()
    save(fig, out / "07_localisation_scores.png")


# ─────────────────────────────────────────────────────────
# Print summary statistics
# ─────────────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"\n  {'Configuration':<25} {'Acc':>7} {'F1w':>7} "
          f"{'QWK':>7} {'AUC':>7} {'KL1Rec':>8}")
    print(f"  {'-'*60}")

    for arch in ["resnet50", "efficientnet"]:
        for loss in ["ce", "coral"]:
            runs = get_config_seeds(results, arch, loss)
            if not runs:
                continue
            acc_mean = np.mean([r.get("accuracy",    0) for r in runs])
            f1_mean  = np.mean([r.get("f1_weighted", 0) for r in runs])
            qwk_mean = np.mean([r["qwk"]               for r in runs])
            auc_mean = np.mean([r["macro_auc"] for r in runs
                                if r.get("macro_auc") is not None])
            kl1_mean = np.mean([r["kl1_recall"]        for r in runs])
            kl1_sd   = np.std( [r["kl1_recall"]        for r in runs])
            name     = f"{arch}_{loss}"
            print(f"  {name:<25} "
                  f"{acc_mean:.3f}  "
                  f"{f1_mean:.3f}  "
                  f"{qwk_mean:.4f}  "
                  f"{auc_mean:.4f}  "
                  f"{kl1_mean:.4f}±{kl1_sd:.4f}")

    print(f"\n  Meta-analytic KL1 benchmark (Zhao et al. 2024): 0.64")
    print("=" * 65)


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Results analysis and figure generation"
    )
    p.add_argument("--results_dir", type=str, required=True)
    p.add_argument("--output_dir",  type=str, default="results/figures")
    return p.parse_args()


def main():
    args        = parse_args()
    results_dir = Path(args.results_dir)
    metrics_dir = results_dir / "metrics"
    gradcam_dir = results_dir / "gradcam"
    out_dir     = Path(args.output_dir)

    print("=" * 60)
    print("  02_results_analysis.py")
    print("=" * 60)
    print(f"  Results dir : {results_dir}")
    print(f"  Output dir  : {out_dir}")

    if not metrics_dir.exists():
        print(f"\n  ERROR: metrics dir not found: {metrics_dir}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n  Loading evaluation results...")
    results = load_eval_results(metrics_dir)
    print(f"  Loaded {len(results)} evaluation files")

    if len(results) == 0:
        print("  ERROR: No evaluation files found.")
        sys.exit(1)

    print_summary(results)

    plot_learning_curves(metrics_dir, out_dir)
    plot_per_grade_recall(results, out_dir)
    plot_kl1_comparison(results, out_dir)
    plot_qwk_comparison(results, out_dir)
    plot_confusion_matrices(results, out_dir)
    plot_metrics_summary(results, out_dir)
    plot_localisation_scores(gradcam_dir, out_dir)

    print("\n" + "=" * 60)
    print(f"  All figures saved to: {out_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
