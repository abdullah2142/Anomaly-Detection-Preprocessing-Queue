#!/usr/bin/env python3
"""Regenerate the CSV-derived rescue figures from the master benchmark CSV.

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/benchmark_master_combined.csv")
    ap.add_argument("--out", default="results/analysis")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m = rescue_deltas(args.csv)
    figure_04(m, out)
    figure_05(m, out)


if __name__ == "__main__":
    main()
