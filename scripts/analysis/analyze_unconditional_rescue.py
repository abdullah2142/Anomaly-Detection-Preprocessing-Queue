#!/usr/bin/env python3
"""Analyse the unconditional / misidentified rescue run.

The benchmark applies each rescue only to the corruption it targets, selected
using that corruption's ground-truth identity. Deployment offers neither
guarantee. This run measures both missing cases, and both comparison baselines
already exist in the master CSV:

  Part A -- rescue on undegraded images, compared against the clean baseline
            AUROC for the same (model, category, seed).
  Part B -- rescue on the wrong corruption, compared against the degradation
            AUROC for the same (model, category, seed, corruption, severity),
            and against the matched rescue for that cell.

Significance is an exact two-sided paired permutation test over per-category
means, matching the other analysis scripts.

Usage:
    python scripts/analysis/analyze_unconditional_rescue.py results/unconditional_rescue.txt
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


def exact_signflip(values) -> tuple[float, float, int]:
    x = np.asarray(values, dtype=float)
    n = len(x)
    obs = abs(x.mean())
    hits = sum(1 for s in itertools.product([1, -1], repeat=n)
               if abs((x * np.array(s)).mean()) >= obs - 1e-12)
    return x.mean(), hits / 2 ** n, n


def part_a(run: pd.DataFrame, master: pd.DataFrame) -> None:
    """Cost of preprocessing an image that did not need it."""
    a = run[run.experiment == "rescue_on_clean"]
    if a.empty:
        return
    base = master[(master.phase == "baseline") & (master.dataset == "MVTec-AD")
                  & (master.training == "clean") & (master.seed.isin(a.seed.unique()))
                  & (master.category.isin(a.category.unique()))]
    key = ["model", "category", "seed"]
    m = a.merge(base[key + ["image_AUROC"]].rename(columns={"image_AUROC": "clean_baseline"}),
                on=key, how="left")
    assert len(m) == len(a), "baseline join is not 1:1"
    m["delta"] = (m.image_AUROC - m.clean_baseline) * 100

    print("=== Part A: rescue applied to UNDEGRADED images ===")
    print("(delta vs the clean baseline; negative means preprocessing cost you accuracy)\n")
    print(f"{'Model':11s} {'Rescue':24s} {'Delta':>9s} {'p':>9s}")
    for (model, rescue), g in m.groupby(["model", "rescue"]):
        mean, p, n = exact_signflip(g.groupby("category")["image_AUROC"].mean().values
                                    - g.groupby("category")["clean_baseline"].mean().values)
        print(f"{model:11s} {rescue:24s} {mean*100:+8.2f} pp {p:8.4f}")
    print()
    for model, g in m.groupby("model"):
        mean, p, n = exact_signflip(g.groupby("category")["delta"].mean().values)
        print(f"  {model:11s} pooled over all six rescues: {mean:+.2f} pp (p = {p:.4f}, {n} categories)")


def part_b(run: pd.DataFrame, master: pd.DataFrame) -> None:
    """Cost of applying the rescue for the wrong corruption."""
    b = run[run.experiment == "rescue_mismatched"]
    if b.empty:
        return
    key = ["model", "category", "seed", "ctype", "severity"]
    deg = master[(master.phase == "degradation") & (master.rescue == "none")
                 & (master.dataset == "MVTec-AD") & (master.training == "clean")]
    matched = master[(master.phase == "rescue") & (master.dataset == "MVTec-AD")
                     & (master.training == "clean")]
    m = b.merge(deg[key + ["image_AUROC"]].rename(columns={"image_AUROC": "degraded"}),
                on=key, how="left")
    matched_mean = matched.groupby(key)["image_AUROC"].mean().rename("matched_rescue")
    m = m.merge(matched_mean, on=key, how="left")
    assert len(m) == len(b), "degradation join is not 1:1"
    m["vs_degraded"] = (m.image_AUROC - m.degraded) * 100
    m["vs_matched"] = (m.image_AUROC - m.matched_rescue) * 100

    print("\n=== Part B: rescue applied to the WRONG corruption ===")
    print("(vs_degraded: cost against doing nothing. vs_matched: cost of misidentifying)\n")
    print(f"{'Corruption':15s} {'Wrong rescue':24s} {'vs degraded':>12s} {'p':>8s} {'vs matched':>12s}")
    for (ctype, rescue), g in m.groupby(["ctype", "rescue"]):
        d, pd_, _ = exact_signflip(g.groupby("category")["vs_degraded"].mean().values)
        v, _, _ = exact_signflip(g.groupby("category")["vs_matched"].mean().dropna().values) \
            if g.vs_matched.notna().any() else (float("nan"), 1.0, 0)
        print(f"{ctype:15s} {rescue:24s} {d:+11.2f} pp {pd_:7.4f} {v:+11.2f} pp")
    print()
    for model, g in m.groupby("model"):
        mean, p, n = exact_signflip(g.groupby("category")["vs_degraded"].mean().values)
        print(f"  {model:11s} pooled misidentified rescue vs doing nothing: "
              f"{mean:+.2f} pp (p = {p:.4f}, {n} categories)")


def resolve_run(path: str) -> str:
    """Accept either extension. Notebooks write .csv; downloads often arrive .txt."""
    p = Path(path)
    if p.exists():
        return str(p)
    for alt in (p.with_suffix(".csv"), p.with_suffix(".txt")):
        if alt.exists():
            print(f"note: using {alt} for {path}")
            return str(alt)
    raise SystemExit(f"not found: {path} (tried .csv and .txt)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run")
    ap.add_argument("--master", default="data/benchmark_master_combined.csv")
    args = ap.parse_args()
    run = pd.read_csv(resolve_run(args.run))
    master = pd.read_csv(args.master)
    print(f"=== {Path(resolve_run(args.run)).name} ===")
    print(f"{len(run)} rows | {run.category.nunique()} categories | "
          f"{sorted(run.model.unique())} | seed {sorted(run.seed.unique())}\n")
    part_a(run, master)
    part_b(run, master)


if __name__ == "__main__":
    main()
