#!/usr/bin/env python3
"""Test the preprocessing fallacy directly from anomaly-score magnitudes.

The paper claims restoration does not return a corrupted image to the clean
distribution but creates a third distribution, further from normal than the
corruption was. AUROC cannot test this: it measures only the ranking of scores.
The anomaly score itself is a distance from normal -- nearest-neighbour distance
to the coreset for PatchCore, Mahalanobis distance for PaDiM -- so its magnitude
can.

The comparison is restricted to NORMAL test images. They contain no defect, so a
rise in their score is the model reacting to something that is not a defect. The
fallacy predicts:

    score(clean)  <  score(degraded)  <  score(rescued)

The third inequality is the one the paper asserts and has never measured.

SCORE COMPARABILITY. Scores are only comparable within one trained model, so all
comparisons are within a (category, model) unit and reported as per-unit
differences. If anomalib min-max normalised each run separately the magnitudes
would be meaningless; this script detects that failure mode and refuses to draw a
conclusion.

Usage:
    python scripts/analysis/analyze_feature_space.py results/feature_space_probe.txt
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

UNIT = ["category", "model"]


def exact_signflip(values) -> tuple[float, float, int]:
    x = np.asarray(values, dtype=float)
    n = len(x)
    obs = abs(x.mean())
    hits = sum(1 for s in itertools.product([1, -1], repeat=n)
               if abs((x * np.array(s)).mean()) >= obs - 1e-12)
    return x.mean(), hits / 2 ** n, n


def normalisation_guard(df: pd.DataFrame) -> bool:
    """True if scores look safe to compare across conditions.

    Per-run min-max normalisation would pin every condition to [0, 1]. If nearly
    every condition does that, magnitudes were rescaled per run and cannot be
    compared.
    """
    spans = df.groupby(UNIT + ["condition", "ctype", "severity", "rescue"])["anomaly_score"] \
              .agg(["min", "max"])
    pinned = ((spans["min"].abs() < 1e-6) & ((spans["max"] - 1).abs() < 1e-6))
    share = pinned.mean()
    print(f"Normalisation check: {share*100:.1f}% of conditions span exactly [0, 1] "
          f"({int(pinned.sum())}/{len(pinned)})")
    if share > 0.9:
        print("  REFUSING TO CONCLUDE -- scores were renormalised per run, so their")
        print("  magnitudes are not comparable across conditions. Re-run with score")
        print("  normalisation disabled (see make_score_engine in the notebook).")
        return False
    if share > 0.1:
        print("  WARNING: some conditions are pinned to [0, 1]; interpret with care.")
    else:
        print("  OK -- scores retain their raw scale.")
    return True


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
    args = ap.parse_args()
    df = pd.read_csv(resolve_run(args.run))

    print(f"=== {Path(resolve_run(args.run)).name} ===")
    print(f"{len(df)} score rows | {df.category.nunique()} categories | "
          f"{sorted(df.model.unique())}\n")

    if not normalisation_guard(df):
        return
    print()

    normal = df[df.is_anomalous == 0]
    print(f"Restricting to normal test images: {len(normal)} of {len(df)} rows\n")

    # mean score per unit and condition
    per_unit = normal.groupby(UNIT + ["condition"])["anomaly_score"].mean().unstack()
    missing = [c for c in ["clean", "degraded", "rescued"] if c not in per_unit.columns]
    if missing:
        print(f"missing conditions: {missing}")
        return

    print("Mean anomaly score on NORMAL images, by condition:")
    print(per_unit.round(4).to_string())

    print("\n=== the fallacy's three predictions ===")
    tests = [("degraded > clean   (corruption moves images away from normal)", "degraded", "clean"),
             ("rescued  > clean   (rescued images are not back home)", "rescued", "clean"),
             ("rescued  > degraded (THE CLAIM: restoration moves them FURTHER)", "rescued", "degraded")]
    for label, a, b in tests:
        diff = (per_unit[a] - per_unit[b]).dropna()
        mean, p, n = exact_signflip(diff.values)
        verdict = "SUPPORTED" if mean > 0 and p < 0.05 else (
            "not supported" if mean <= 0 else "direction right, not significant")
        print(f"  {label}")
        print(f"      mean difference {mean:+.4f}   p = {p:.4f}   "
              f"({n} units, {int((diff > 0).sum())}/{n} positive)   -> {verdict}")

    print("\n=== per rescue method: rescued − degraded on normal images ===")
    resc = normal[normal.condition == "rescued"].groupby(UNIT + ["ctype", "rescue"])["anomaly_score"].mean()
    deg = normal[normal.condition == "degraded"].groupby(UNIT + ["ctype"])["anomaly_score"].mean()
    rows = []
    for (cat, model, ctype, rescue), score in resc.items():
        rows.append({"category": cat, "model": model, "ctype": ctype,
                     "rescue": rescue, "diff": score - deg.loc[(cat, model, ctype)]})
    r = pd.DataFrame(rows)
    for rescue, g in r.groupby("rescue"):
        mean, p, n = exact_signflip(g.groupby("category")["diff"].mean().values)
        print(f"  {rescue:24s} {mean:+.4f}   p = {p:.4f}")


if __name__ == "__main__":
    main()
