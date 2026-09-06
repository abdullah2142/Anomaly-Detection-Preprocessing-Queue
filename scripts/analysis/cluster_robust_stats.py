#!/usr/bin/env python3
"""Cluster-robust re-analysis of the benchmark master CSV.

The paired tests in notebooks/10_wilcoxon_testing.ipynb treat every
(category, seed, corruption, severity) cell as an independent sample. They are
not independent: each category is resampled up to 45 times. This script
aggregates to the category level -- the unit that is genuinely independent --
and runs an exact two-sided paired permutation (sign-flip) test on the category
means. It also reports the effective N after dropping zero differences, which
scipy.stats.wilcoxon discards silently, and the share of measurements sitting on
the AUROC = 0.5 floor.

Point estimates are unchanged by clustering; only the p-values move.

Usage:
    python scripts/analysis/cluster_robust_stats.py [--csv PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["dataset", "model", "category", "seed", "ctype", "severity"]


def exact_signflip(values: np.ndarray) -> tuple[float, float, int]:
    """Exact two-sided paired permutation test on cluster means.

    Enumerates all 2^n sign assignments, so it is exact rather than asymptotic.
    Returns (mean, p_value, n_clusters). With n clusters the smallest attainable
    p is 2 / 2**n -- with 4 clusters that floor is 0.125, so no result on four
    categories can reach conventional significance.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    observed = abs(x.mean())
    hits = sum(
        1
        for signs in itertools.product([1, -1], repeat=n)
        if abs((x * np.array(signs)).mean()) >= observed - 1e-12
    )
    return x.mean(), hits / 2 ** n, n


def augmentation_gains(df: pd.DataFrame) -> pd.DataFrame:
    """Augmented-trained minus clean-trained AUROC on the same degraded cell."""
    deg = df[(df.phase == "degradation") & (df.rescue == "none")]
    merged = deg[deg.training == "clean"][KEY + ["image_AUROC"]].merge(
        deg[deg.training == "augmented"][KEY + ["image_AUROC"]],
        on=KEY,
        suffixes=("_clean", "_aug"),
    )
    merged["delta"] = (merged.image_AUROC_aug - merged.image_AUROC_clean) * 100
    return summarise(merged, ["dataset", "model"])


def rescue_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Rescued minus degraded AUROC on the same cell, all methods pooled."""
    merged = paired_rescue(df)
    return summarise(merged, ["dataset", "model", "training"])


def paired_rescue(df: pd.DataFrame) -> pd.DataFrame:
    rescue = df[df.phase == "rescue"]
    degraded = df[df.phase == "degradation"][KEY + ["training", "image_AUROC"]]
    merged = rescue.merge(
        degraded.rename(columns={"image_AUROC": "degraded_AUROC"}),
        on=KEY + ["training"],
    )
    merged["delta"] = (merged.image_AUROC - merged.degraded_AUROC) * 100
    return merged


def summarise(merged: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in merged.groupby(group_cols):
        per_category = group.groupby("category")["delta"].mean()
        mean, p_value, n_clusters = exact_signflip(per_category.values)
        zeros = int((group.delta == 0).sum())
        rows.append(
            dict(
                zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            | {
                "mean_pp": round(mean, 2),
                "naive_N": len(group),
                "zero_diff": zeros,
                "effective_N": len(group) - zeros,
                "categories": n_clusters,
                "clustered_p": p_value,
                "p_floor": 2 / 2 ** n_clusters,
            }
        )
    return pd.DataFrame(rows)


def floor_report(df: pd.DataFrame) -> dict:
    paired = paired_rescue(df)
    both = int(((paired.degraded_AUROC == 0.5) & (paired.image_AUROC == 0.5)).sum())
    return {
        "rows_at_floor": int((df.image_AUROC == 0.5).sum()),
        "rows_total": len(df),
        "rescue_at_floor": int((paired.image_AUROC == 0.5).sum()),
        "degraded_at_floor": int((paired.degraded_AUROC == 0.5).sum()),
        "paired_rows": len(paired),
        "uninformative_pairs": both,
    }


def fmt_p(p: float, floor: float) -> str:
    """Format a p-value, flagging results pinned to the permutation floor."""
    text = f"{p:.5f}" if p >= 1e-4 else f"{p:.2e}"
    return f"{text} (at floor)" if p <= floor + 1e-12 else text


PREAMBLE = """# Statistical Validation (Cluster-Robust)

This file is generated. Do not edit by hand -- rerun
`python scripts/analysis/cluster_robust_stats.py --out docs/statistical_validation.md`
after any change to the benchmark CSV.

## Why these numbers differ from notebooks/10_wilcoxon_testing.ipynb

Notebook 10 treats every (category, seed, corruption, severity) cell as an
independent paired sample. Those cells are not independent: each category
contributes up to 45 of them, so the tests are pseudoreplicated and their
p-values are inflated by many orders of magnitude.

The tables below aggregate each condition to per-category means -- the level at
which observations are genuinely independent -- and apply an exact two-sided
paired permutation (sign-flip) test over all 2^n sign assignments. **The point
estimates are identical to the pseudoreplicated analysis; only the p-values
move.** The findings are unchanged in direction and magnitude.

With n clusters the smallest attainable two-sided p is 2 / 2^n. For the four
VisA augmented categories that floor is 0.125, so no result computed on them can
reach conventional significance regardless of effect size; those rows are
reported as descriptive evidence, not as significance tests.

"""


def to_markdown(aug: pd.DataFrame, resc: pd.DataFrame, floor: dict) -> str:
    lines = [
        "### Augmentation gain (augmented-trained - clean-trained, same degraded cell)",
        "",
        "| Dataset | Model | Mean gain | Categories | Naive N | Clustered exact p |",
        "|---|---|---|---:|---:|---:|",
    ]
    for _, r in aug.sort_values(["dataset", "model"]).iterrows():
        lines.append(
            f"| {r.dataset} | {r.model} | {r.mean_pp:+.2f} pp | {r.categories} | "
            f"{r.naive_N} | {fmt_p(r.clustered_p, r.p_floor)} |"
        )
    lines += [
        "",
        "### Rescue delta (rescued - degraded, all methods pooled)",
        "",
        "| Dataset | Model | Training | Mean delta | Categories | Naive N | Effective N | Clustered exact p |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for _, r in resc.sort_values(["dataset", "model", "training"]).iterrows():
        lines.append(
            f"| {r.dataset} | {r.model} | {r.training} | {r.mean_pp:+.2f} pp | "
            f"{r.categories} | {r.naive_N} | {r.effective_N} | {fmt_p(r.clustered_p, r.p_floor)} |"
        )
    lines += [
        "",
        "### AUROC floor saturation",
        "",
        f"- Rows at exactly 0.5: **{floor['rows_at_floor']} / {floor['rows_total']}** "
        f"({floor['rows_at_floor'] / floor['rows_total'] * 100:.1f}%)",
        f"- Rescue values at the floor: {floor['rescue_at_floor']} / {floor['paired_rows']} "
        f"({floor['rescue_at_floor'] / floor['paired_rows'] * 100:.1f}%)",
        f"- Degraded values at the floor: {floor['degraded_at_floor']} / {floor['paired_rows']} "
        f"({floor['degraded_at_floor'] / floor['paired_rows'] * 100:.1f}%)",
        f"- Pairs where both sides are 0.5 (zero difference, dropped by the signed-rank test): "
        f"**{floor['uninformative_pairs']}**",
        "",
        "The signed-rank test drops zero differences silently, so the effective N",
        "column above -- not the naive N -- is the sample size those tests actually",
        "used. The floor share also means severity means are censored from below:",
        "the true collapse under severe corruption is worse than the reported AUROC.",
    ]
    return PREAMBLE + "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/benchmark_master_combined.csv")
    parser.add_argument("--out", default=None, help="write the markdown tables here")
    parser.add_argument(
        "--wiener",
        default="hardcoded severe-tier PSF (pre-rerun)",
        help="provenance note for the Wiener rescue rows in this CSV",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    aug = augmentation_gains(df)
    resc = rescue_deltas(df)
    floor = floor_report(df)
    markdown = to_markdown(aug, resc, floor)
    markdown += (
        "\n\n---\n\n"
        f"Source: `{args.csv}` ({len(df)} rows, "
        f"{df.dataset.nunique()} datasets, {df.category.nunique()} categories). "
        f"Wiener rescue rows: **{args.wiener}**.\n"
    )
    print(markdown)
    if args.out:
        Path(args.out).write_text(markdown)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
