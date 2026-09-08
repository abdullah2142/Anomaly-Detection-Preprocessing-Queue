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


def summarise(m: pd.DataFrame) -> dict:
    m = m.dropna(subset=["clean", "matched"]).copy()
    m["gain_vs_clean"] = (m.holdout - m["clean"]) * 100
    m["gain_matched"] = (m.matched - m["clean"]) * 100
    m["cost_of_holdout"] = (m.holdout - m.matched) * 100
    out = {"rows": len(m), "categories": m.category.nunique(), "models": {}, "by_ctype": {},
           "by_severity": {}}
    for model, g in m.groupby("model"):
        per_cat = g.groupby("category")
        gain, p_gain, n = exact_signflip(per_cat["gain_vs_clean"].mean().values)
        matched, p_matched, _ = exact_signflip(per_cat["gain_matched"].mean().values)
        cost, p_cost, _ = exact_signflip(per_cat["cost_of_holdout"].mean().values)
        per = per_cat["gain_vs_clean"].mean()
        out["models"][model] = dict(
            matched=matched, p_matched=p_matched, gain=gain, p_gain=p_gain,
            cost=cost, p_cost=p_cost, n=n, floor=2 / 2 ** n,
            share=100 * gain / matched if matched else float("nan"),
            positive=int((per > 0).sum()))
    for ctype, g in m.groupby("ctype"):
        mean, p, _ = exact_signflip(g.groupby("category")["gain_vs_clean"].mean().values)
        out["by_ctype"][ctype] = (mean, p)
    if m.severity.nunique() > 1:
        for sev in ["mild", "moderate", "severe"]:
            g = m[m.severity == sev]
            if g.empty:
                continue
            mean, p, _ = exact_signflip(g.groupby("category")["gain_vs_clean"].mean().values)
            out["by_severity"][sev] = (mean, p)
    return out


def fmt_p(p: float, floor: float) -> str:
    return f"{p:.4f}" + (", at floor" if p <= floor + 1e-12 else "")


def render(name: str, blurb: str, s: dict) -> str:
    lines = [f"### {name}", "", blurb, "",
             f"{s['rows']} paired rows over {s['categories']} categories.", "",
             "| Model | Matched-oracle gain | Held-out gain | Share surviving | Cost of holding out |",
             "|---|---|---|---:|---|"]
    for model, d in sorted(s["models"].items()):
        share_cell = "none" if d["gain"] <= 0 else f"{d['share']:.0f}%"
        lines.append(
            f"| {model} | {d['matched']:+.2f} pp (p = {fmt_p(d['p_matched'], d['floor'])}) "
            f"| **{d['gain']:+.2f} pp** (p = {fmt_p(d['p_gain'], d['floor'])}) "
            f"| {share_cell} "
            f"| {d['cost']:+.2f} pp (p = {fmt_p(d['p_cost'], d['floor'])}) |")
    lines += ["", "Held-out gain per corruption type (models pooled):", "",
              "| Corruption | Gain | p |", "|---|---|---:|"]
    for ctype, (mean, p) in sorted(s["by_ctype"].items()):
        lines.append(f"| {ctype} | {mean:+.2f} pp | {p:.4f} |")
    if s["by_severity"]:
        lines += ["", "Held-out gain per severity (models pooled):", "",
                  "| Severity | Gain | p |", "|---|---|---:|"]
        for sev, (mean, p) in s["by_severity"].items():
            lines.append(f"| {sev} | {mean:+.2f} pp | {p:.4f} |")
    return "\n".join(lines)


def report(m: pd.DataFrame) -> None:
    s = summarise(m)
    for model, d in sorted(s["models"].items()):
        print(f"{model}  ({d['n']} categories, permutation floor p = {d['floor']:.4f})")
        print(f"   matched-oracle gain : {d['matched']:+6.2f} pp   p = {d['p_matched']:.4f}")
        print(f"   held-out gain       : {d['gain']:+6.2f} pp   p = {d['p_gain']:.4f}")
        print(f"   cost of holding out : {d['cost']:+6.2f} pp   p = {d['p_cost']:.4f}")
        print(f"   share surviving: {d['share']:.0f}%   positive in {d['positive']}/{d['n']} categories\n")
    print("Per-corruption held-out gain (models pooled):")
    for ctype, (mean, p) in sorted(s["by_ctype"].items()):
        print(f"   {ctype:15s} {mean:+6.2f} pp   p = {p:.4f}")


PREAMBLE = """# Generalization Controls

This file is generated. Do not edit by hand -- rerun
`python scripts/analysis/analyze_generalization.py --doc docs/generalization_controls.md`
after any change to the control runs or the benchmark CSV.

The headline augmentation gains (+12.3 pp PatchCore, +10.1 pp PaDiM) are a
**corruption-matched oracle bound**: training draws from the same five corruption
types and three severities used at test time, so the benchmark alone cannot
separate genuine robustness from having trained on the test distribution. These
two controls separate them.

Each control trains the augmented arm with one condition withheld and tests only
on that withheld condition. The clean-trained and matched-augmented comparison
rows come from the master CSV, restricted to the same categories, corruptions,
severities and seed. Significance is an exact two-sided paired permutation test
over per-category means; with eight categories the smallest attainable p is
0.0078.

**Share surviving** is the held-out gain as a percentage of the matched-oracle
gain -- how much of the published benefit remains when the model has not been
trained on what it is tested against.

"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", nargs="*", help="CSV(s) produced by a control notebook")
    ap.add_argument("--master", default="data/benchmark_master_combined.csv")
    ap.add_argument("--doc", default=None, help="write the markdown summary here")
    args = ap.parse_args()

    runs = args.run or ["results/leave_one_corruption_out.txt", "results/severity_holdout.txt"]
    meta = {
        "leave_one_corruption_out": (
            "Leave-One-Corruption-Out",
            "Trained on four corruption types, tested only on the withheld fifth, "
            "rotating through all five. Answers whether augmentation transfers to an "
            "unseen *kind* of degradation."),
        "severity_holdout": (
            "Severity Holdout",
            "Trained on mild and moderate only, tested on severe. Answers whether "
            "augmentation transfers to degradation *worse* than it trained for."),
    }
    sections = []
    for run in runs:
        stem = Path(run).stem
        name, blurb = meta.get(stem, (stem, ""))
        m = load(run, args.master)
        if args.doc:
            sections.append(render(name, blurb, summarise(m)))
        else:
            print(f"=== {Path(run).name} ===\n")
            report(m)
            print()
    if args.doc:
        Path(args.doc).write_text(PREAMBLE + "\n\n".join(sections) + "\n")
        print(f"wrote {args.doc}")


if __name__ == "__main__":
    main()
