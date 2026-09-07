#!/usr/bin/env python3
"""Regenerate docs/per_corruption_tables.md from the master benchmark CSV.

Every number in that file is a mean over (category, seed) for one
(model, training, corruption, severity[, rescue]) cell. Deriving it by script
rather than by hand removes a whole class of staleness -- the kind that left
to_do.md quoting VisA rescue deltas that no longer matched the data.

Usage:
    python scripts/analysis/build_per_corruption_tables.py [--csv PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PAIR_KEY = ["dataset", "model", "category", "seed", "ctype", "severity", "training"]


def gains_table(df: pd.DataFrame, dataset: str) -> str:
    deg = df[(df.phase == "degradation") & (df.rescue == "none") & (df.dataset == dataset)]
    piv = deg.pivot_table(index=["model", "ctype", "severity"],
                          columns="training", values="image_AUROC", aggfunc="mean")
    sev_order = {"mild": 0, "moderate": 1, "severe": 2}
    piv = piv.reset_index()
    piv = piv.sort_values(["model", "ctype", "severity"],
                          key=lambda c: c.map(sev_order) if c.name == "severity" else c)
    lines = ["| Model | Corruption | Severity | Clean Train AUROC | Aug Train AUROC | Delta (Aug - Clean) |",
             "|---|---|---|---|---|---|"]
    for _, r in piv.iterrows():
        if pd.isna(r.get("clean")) or pd.isna(r.get("augmented")):
            continue
        d = r["augmented"] - r["clean"]
        lines.append(f"| {r['model']} | {r['ctype']} | {r['severity']} | "
                     f"{r['clean']:.4f} | {r['augmented']:.4f} | {d:+.4f} |")
    return "\n".join(lines)


def rescue_table(df: pd.DataFrame, dataset: str) -> str:
    sub = df[df.dataset == dataset]
    resc = sub[sub.phase == "rescue"]
    deg = sub[sub.phase == "degradation"][PAIR_KEY + ["image_AUROC"]].rename(
        columns={"image_AUROC": "deg"})
    m = resc.merge(deg, on=PAIR_KEY)
    piv = m.groupby(["model", "training", "ctype", "severity", "rescue"])[
        ["deg", "image_AUROC"]].mean().reset_index()
    sev_order = {"mild": 0, "moderate": 1, "severe": 2}
    piv = piv.sort_values(["model", "training", "ctype", "severity", "rescue"],
                          key=lambda c: c.map(sev_order) if c.name == "severity" else c)
    lines = ["| Model | Training | Corruption | Severity | Rescue Method | Degraded AUROC | Rescued AUROC | Rescue Delta |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in piv.iterrows():
        d = r["image_AUROC"] - r["deg"]
        lines.append(f"| {r['model']} | {r['training']} | {r['ctype']} | {r['severity']} | "
                     f"{r['rescue']} | {r['deg']:.4f} | {r['image_AUROC']:.4f} | {d:+.4f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/benchmark_master_combined.csv")
    ap.add_argument("--out", default="docs/per_corruption_tables.md")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    parts = [
        "# Per-Corruption Type Breakdown Tables",
        "",
        "This file is generated. Do not edit by hand -- rerun",
        "`python scripts/analysis/build_per_corruption_tables.py` after any change to",
        "the benchmark CSV.",
        "",
        "Wiener rescue rows use per-severity PSF parameters matched to the corruption",
        "that generated the blur.",
        "",
        "## 1. Augmented Training Gains by Corruption Type",
        "",
    ]
    for ds in ["MVTec-AD", "VisA"]:
        parts += [f"### Dataset: {ds}", "", gains_table(df, ds), ""]
    parts += ["## 2. Rescue Deltas by Corruption Type", ""]
    for ds in ["MVTec-AD", "VisA"]:
        parts += [f"### Dataset: {ds}", "", rescue_table(df, ds), ""]

    Path(args.out).write_text("\n".join(parts))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
