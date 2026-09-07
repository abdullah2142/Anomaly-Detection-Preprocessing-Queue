#!/usr/bin/env python3
"""Analyse the generalization-control runs (severity holdout, leave-one-corruption-out).

Both experiments ask whether the published augmentation gain reflects genuine
robustness or a corruption-matched oracle. Each run supplies only the new
augmented arm; the clean-trained and matched-augmented comparison rows come from
the master benchmark CSV, restricted to the same categories, corruptions,
severities and seed.

Three quantities per model:
  gain vs clean     -- did augmentation help at all on the held-out condition?
  gain vs matched   -- what did holding the condition out cost?
  survival          -- gain vs clean, as a share of the matched-oracle gain.

Significance is an exact two-sided paired permutation test over per-category
means, matching scripts/analysis/cluster_robust_stats.py.

Usage:
    python scripts/analysis/analyze_generalization.py results/severity_holdout.txt
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

# severity must be part of the join key: the leave-one-corruption-out run carries
# three severities per (model, category, corruption), so joining without it would
# cross-join each run row against all three master rows.
KEY = ["model", "category", "ctype", "severity"]


def exact_signflip(values) -> tuple[float, float, int]:
    x = np.asarray(values, dtype=float)
    n = len(x)
    obs = abs(x.mean())
    hits = sum(1 for s in itertools.product([1, -1], repeat=n)
               if abs((x * np.array(s)).mean()) >= obs - 1e-12)
    return x.mean(), hits / 2 ** n, n


def load(run_path: str, master_path: str) -> pd.DataFrame:
    run = pd.read_csv(run_path)
    master = pd.read_csv(master_path)
    seeds = sorted(run.seed.unique())
    sevs = sorted(run.severity.unique())
    ctypes = sorted(run.ctype.unique())
    cats = sorted(run.category.unique())

    base = master[(master.dataset == "MVTec-AD") & (master.phase == "degradation")
                  & (master.rescue == "none") & (master.seed.isin(seeds))
                  & (master.severity.isin(sevs)) & (master.ctype.isin(ctypes))
                  & (master.category.isin(cats))]
    merged = run[KEY + ["image_AUROC"]].rename(columns={"image_AUROC": "holdout"})
    for training, label in [("clean", "clean"), ("augmented", "matched")]:
        side = base[base.training == training][KEY + ["image_AUROC"]].rename(
            columns={"image_AUROC": label})
        merged = merged.merge(side, on=KEY, how="left")
    assert len(merged) == len(run), (
        f"join changed row count: {len(run)} run rows -> {len(merged)}; "
        "the key is not unique on one side")
    missing = merged[["clean", "matched"]].isna().sum().sum()
    if missing:
        print(f"warning: {missing} comparison values missing from the master CSV")
    return merged


def report(m: pd.DataFrame) -> None:
    m = m.dropna(subset=["clean", "matched"]).copy()
    m["gain_vs_clean"] = (m.holdout - m["clean"]) * 100
    m["gain_matched"] = (m.matched - m["clean"]) * 100
    m["cost_of_holdout"] = (m.holdout - m.matched) * 100

    print(f"{len(m)} paired rows | {m.category.nunique()} categories | "
          f"{m.ctype.nunique()} corruptions\n")
    for model, g in m.groupby("model"):
        per_cat = g.groupby("category")
        gain, p_gain, n = exact_signflip(per_cat["gain_vs_clean"].mean().values)
        matched, p_matched, _ = exact_signflip(per_cat["gain_matched"].mean().values)
        cost, p_cost, _ = exact_signflip(per_cat["cost_of_holdout"].mean().values)
        share = 100 * gain / matched if matched else float("nan")
        print(f"{model}  ({n} categories, permutation floor p = {2/2**n:.4f})")
        print(f"   matched-oracle gain : {matched:+6.2f} pp   p = {p_matched:.4f}")
        print(f"   held-out gain       : {gain:+6.2f} pp   p = {p_gain:.4f}")
        print(f"   cost of holding out : {cost:+6.2f} pp   p = {p_cost:.4f}")
        print(f"   share of gain that survives the holdout: {share:.0f}%\n")

    print("Per-corruption held-out gain (models pooled):")
    for ctype, g in m.groupby("ctype"):
        mean, p, _ = exact_signflip(g.groupby("category")["gain_vs_clean"].mean().values)
        print(f"   {ctype:15s} {mean:+6.2f} pp   p = {p:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", help="CSV produced by a generalization-control notebook")
    ap.add_argument("--master", default="data/benchmark_master_combined.csv")
    args = ap.parse_args()
    print(f"=== {Path(args.run).name} ===\n")
    report(load(args.run, args.master))


if __name__ == "__main__":
    main()
