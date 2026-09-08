#!/usr/bin/env python3
"""Regenerate every CSV-derived figure from the master benchmark CSV.

Figures 04 and 05 summarise rescue deltas, so both changed when the corrected
per-severity Wiener rows were merged. The original generator for the current
figure set is not in the repository (the only historical one,
notebooks/10_analysis_and_figures.py, reads a different, long-gone set of Kaggle
CSVs), so this script replaces it for the two data-derived rescue figures.

Scope is MVTec-AD, matching the published figures -- verified by reproducing
unchanged (non-Wiener) cells of the previous renders exactly.

Usage:
    python scripts/analysis/build_figures.py [--csv PATH] [--out DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PAIR_KEY = ["dataset", "model", "category", "seed", "ctype", "severity", "training"]
SEV_ORDER = ["mild", "moderate", "severe"]
ROW_ORDER = [
    ("low_light", "CLAHE"), ("low_light", "Retinex"),
    ("gaussian_blur", "Wiener"), ("motion_blur", "Wiener (Motion PSF)"),
    ("sensor_noise", "NLM Denoise"), ("fog_haze", "Dehaze (Dark Channel)"),
]
COLUMNS = [("PatchCore", "clean", "PC\nClean"), ("PatchCore", "augmented", "PC\nAug"),
           ("PaDiM", "clean", "PD\nClean"), ("PaDiM", "augmented", "PD\nAug")]
BAR_CONDITIONS = [("PatchCore", "clean", "PatchCore\nClean", "#2f5f8a"),
                  ("PatchCore", "augmented", "PatchCore\nAugmented", "#3d9ad1"),
                  ("PaDiM", "clean", "PaDiM\nClean", "#a33327"),
                  ("PaDiM", "augmented", "PaDiM\nAugmented", "#e8553f")]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "grid.alpha": 0.3, "grid.linestyle": "--", "figure.dpi": 150,
})


def rescue_deltas(csv: str) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df = df[df.dataset == "MVTec-AD"]
    resc = df[df.phase == "rescue"]
    deg = df[df.phase == "degradation"][PAIR_KEY + ["image_AUROC"]].rename(
        columns={"image_AUROC": "deg"})
    m = resc.merge(deg, on=PAIR_KEY)
    m["delta"] = m.image_AUROC - m.deg
    return m


def figure_04(m: pd.DataFrame, out: Path) -> None:
    g = m.groupby(["model", "training", "ctype", "rescue", "severity"])["delta"].mean()
    labels, matrix = [], []
    for ctype, rescue in ROW_ORDER:
        for sev in SEV_ORDER:
            labels.append(f"{ctype.replace('_', ' ')}\n{rescue}\n{sev}")
            matrix.append([g.get((mo, tr, ctype, rescue, sev), float("nan"))
                           for mo, tr, _ in COLUMNS])
    data = pd.DataFrame(matrix, index=labels, columns=[c[2] for c in COLUMNS])

    fig, ax = plt.subplots(figsize=(9, 16))
    vmax = 0.1
    vmin = min(-0.45, float(data.min().min()) - 0.01)
    im = ax.imshow(data.values, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(data.columns)), data.columns)
    ax.set_yticks(range(len(data.index)), data.index, fontsize=9)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.values[i, j]
            if pd.isna(v):
                continue
            # white text on the darkest cells only
            colour = "white" if v < vmin + 0.12 else "black"
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", color=colour, fontsize=9)
    ax.set_title("Rescue Preprocessing Delta — All 4 Conditions\n"
                 "(Wiener rows use per-severity oracle PSFs)",
                 fontsize=15, fontweight="bold", pad=14)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label("AUROC Δ (rescue − degraded)", fontsize=12)
    ax.set_xticks([x - 0.5 for x in range(1, data.shape[1])], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, data.shape[0])], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    fig.savefig(out / "04_rescue_delta_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote 04_rescue_delta_heatmap.png")


def figure_05(m: pd.DataFrame, out: Path) -> None:
    rates, means, names, colours = [], [], [], []
    for model, training, label, colour in BAR_CONDITIONS:
        g = m[(m.model == model) & (m.training == training)]
        rates.append((g.delta > 0).mean() * 100)
        means.append(g.delta.mean())
        names.append(label)
        colours.append(colour)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    bars = ax.bar(names, rates, color=colours, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}%", ha="center", fontsize=11)
    ax.set_ylim(0, 50)
    ax.set_ylabel("% of rescue instances where rescue AUROC > deg AUROC")
    ax.set_title("Rescue Success Rate", fontweight="bold")
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    ax = axes[1]
    bars = ax.bar(names, means, color=colours, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v - 0.004, f"{v:.4f}", ha="center",
                va="top", fontsize=11)
    ax.set_ylabel("Mean AUROC Δ (rescue − degraded)")
    ax.set_title("Mean Rescue Delta", fontweight="bold")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_ylim(min(means) * 1.25, 0)

    fig.suptitle("Rescue Preprocessing Effectiveness Across All 4 Conditions",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "05_rescue_success_rates.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote 05_rescue_success_rates.png")
    for n, r, v in zip(names, rates, means):
        print(f"   {n.replace(chr(10), ' '):22s} rate={r:5.1f}%  mean={v:+.4f}")


def figure_01(df: pd.DataFrame, out: Path) -> None:
    """Clean baseline AUROC per model and training regime."""
    b = df[(df.dataset == "MVTec-AD") & (df.phase == "baseline")]
    g = b.groupby(["model", "training"])["image_AUROC"].agg(["mean", "std"])
    labels, means, errs, colours = [], [], [], []
    for model, training, label, colour in BAR_CONDITIONS:
        labels.append(label)
        means.append(g.loc[(model, training), "mean"])
        errs.append(g.loc[(model, training), "std"])
        colours.append(colour)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(labels, means, yerr=errs, capsize=5, color=colours,
                  edgecolor="black", linewidth=0.6)
    for b_, v in zip(bars, means):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Clean-image AUROC")
    ax.set_title("Clean Baseline AUROC (MVTec-AD, 15 categories x 3 seeds)", fontweight="bold")
    ax.grid(axis="y"); ax.set_axisbelow(True)
    fig.savefig(out / "01_baseline_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote 01_baseline_comparison.png")


def figure_02(df: pd.DataFrame, out: Path) -> None:
    """Degradation AUROC vs severity, one panel per corruption type."""
    d = df[(df.dataset == "MVTec-AD") & (df.phase == "degradation") & (df.rescue == "none")]
    ctypes = sorted(d.ctype.unique())
    fig, axes = plt.subplots(1, len(ctypes), figsize=(4 * len(ctypes), 4.5), sharey=True)
    for ax, ctype in zip(axes, ctypes):
        sub = d[d.ctype == ctype]
        for model, training, label, colour in BAR_CONDITIONS:
            g = sub[(sub.model == model) & (sub.training == training)]
            y = [g[g.severity == s]["image_AUROC"].mean() for s in SEV_ORDER]
            ax.plot(SEV_ORDER, y, marker="o", color=colour, label=label.replace("\n", " "),
                    linestyle="--" if training == "augmented" else "-")
        ax.set_title(ctype.replace("_", " "), fontweight="bold")
        ax.grid(True); ax.set_axisbelow(True); ax.set_ylim(0.45, 1.0)
    axes[0].set_ylabel("Mean degradation AUROC")
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Degradation AUROC vs Severity — All 4 Conditions",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "02_degradation_curves_4way.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote 02_degradation_curves_4way.png")


def figure_03(df: pd.DataFrame, out: Path) -> None:
    """Augmentation gain per corruption type and severity."""
    d = df[(df.dataset == "MVTec-AD") & (df.phase == "degradation") & (df.rescue == "none")]
    piv = d.pivot_table(index=["ctype", "severity"], columns=["model", "training"],
                        values="image_AUROC", aggfunc="mean")
    rows, labels = [], []
    for ctype in sorted(d.ctype.unique()):
        for sev in SEV_ORDER:
            labels.append(f"{ctype.replace('_', ' ')}\n{sev}")
            rows.append([piv.loc[(ctype, sev), (m, "augmented")] - piv.loc[(ctype, sev), (m, "clean")]
                         for m in ["PatchCore", "PaDiM"]])
    data = np.array(rows) * 100
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(x - 0.2, data[:, 0], 0.4, label="PatchCore", color="#3d9ad1", edgecolor="black", linewidth=0.5)
    ax.bar(x + 0.2, data[:, 1], 0.4, label="PaDiM", color="#e8553f", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x, labels, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Augmentation gain (pp)")
    ax.set_title("Augmented Training Gain by Corruption and Severity", fontweight="bold")
    ax.legend(); ax.grid(axis="y"); ax.set_axisbelow(True)
    fig.savefig(out / "03_augmented_training_gain.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote 03_augmented_training_gain.png")


def figure_06(df: pd.DataFrame, out: Path) -> None:
    """Mean degradation AUROC vs severity, one panel per model."""
    d = df[(df.dataset == "MVTec-AD") & (df.phase == "degradation") & (df.rescue == "none")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, model in zip(axes, ["PatchCore", "PaDiM"]):
        for m, training, label, colour in BAR_CONDITIONS:
            if m != model:
                continue
            g = d[(d.model == model) & (d.training == training)]
            y = [g[g.severity == s]["image_AUROC"].mean() for s in SEV_ORDER]
            ax.plot(SEV_ORDER, y, marker="o", markersize=9, linewidth=2.5, color=colour,
                    linestyle="--" if training == "augmented" else "-",
                    label=f"{model} — {training.capitalize()}")
        ax.set_title(f"{model} — Mean Deg. AUROC vs Severity", fontweight="bold")
        ax.set_ylim(0.5, 1.0); ax.grid(True); ax.set_axisbelow(True); ax.legend()
    axes[0].set_ylabel("Mean AUROC (all ctypes)")
    fig.tight_layout()
    fig.savefig(out / "06_severity_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote 06_severity_comparison.png")



def figure_11(csv: str, out: Path) -> None:
    """Generalization controls: how much of the augmentation gain survives when
    the tested condition is withheld from training."""
    runs = [("results/leave_one_corruption_out.txt", "Leave-one-corruption-out"),
            ("results/severity_holdout.txt", "Severity holdout")]
    runs = [(p_, n) for p_, n in runs if Path(p_).exists()]
    if not runs:
        print("skipped 11 (no control runs found)")
        return
    master = pd.read_csv(csv)
    fig, axes = plt.subplots(1, len(runs), figsize=(7 * len(runs), 5.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (path, name) in zip(axes, runs):
        run = pd.read_csv(path)
        key = ["model", "category", "ctype", "severity"]
        base = master[(master.dataset == "MVTec-AD") & (master.phase == "degradation")
                      & (master.rescue == "none") & (master.seed.isin(run.seed.unique()))
                      & (master.category.isin(run.category.unique()))]
        m = run[key + ["image_AUROC"]].rename(columns={"image_AUROC": "holdout"})
        for tr, lab in [("clean", "clean"), ("augmented", "matched")]:
            m = m.merge(base[base.training == tr][key + ["image_AUROC"]].rename(
                columns={"image_AUROC": lab}), on=key)
        models = ["PatchCore", "PaDiM"]
        matched = [(m[m.model == k].matched - m[m.model == k]["clean"]).mean() * 100 for k in models]
        held = [(m[m.model == k].holdout - m[m.model == k]["clean"]).mean() * 100 for k in models]
        x = np.arange(len(models))
        ax.bar(x - 0.2, matched, 0.4, label="Matched-oracle gain",
               color="#9ec9e2", edgecolor="black", linewidth=0.6)
        ax.bar(x + 0.2, held, 0.4, label="Held-out gain",
               color="#2f5f8a", edgecolor="black", linewidth=0.6)
        for xi, v in zip(x - 0.2, matched):
            ax.text(xi, v + 0.25, f"{v:+.1f}", ha="center", fontsize=10)
        for xi, v in zip(x + 0.2, held):
            ax.text(xi, v + 0.25 if v >= 0 else v - 0.75, f"{v:+.1f}", ha="center", fontsize=10)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, models)
        ax.set_title(name, fontweight="bold")
        ax.grid(axis="y"); ax.set_axisbelow(True)
    axes[0].set_ylabel("Augmentation gain vs clean training (pp)")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("How Much of the Augmentation Gain Survives When the Tested "
                 "Condition Is Withheld", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "11_generalization_controls.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote 11_generalization_controls.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/benchmark_master_combined.csv")
    ap.add_argument("--out", default="results/analysis")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    figure_01(df, out)
    figure_02(df, out)
    figure_03(df, out)
    figure_06(df, out)
    m = rescue_deltas(args.csv)
    figure_04(m, out)
    figure_05(m, out)
    figure_11(args.csv, out)


if __name__ == "__main__":
    main()
