#!/usr/bin/env python3
"""Merge the corrected Wiener rescue rows into the master benchmark CSV.

The published mild/moderate Wiener rows were produced with the severe-tier PSF
hardcoded across all severities, so they measure kernel misspecification rather
than oracle deconvolution. scripts/wiener_reruns/ retrains each model identically
and re-executes only those rescue inferences with per-severity PSFs. This script
substitutes the corrected AUROC values in place.

The rerun CSVs carry neither a `training` nor (for MVTec) a `dataset` column, so
clean and augmented rows for the same (model, category, seed, corruption,
severity, rescue) are indistinguishable inside a naive concatenation -- 456 of
the 1104 rows collide. Both labels are therefore assigned from the source
filename, and the join is asserted to be exactly 1:1 before anything is written.

Usage:
    python scripts/analysis/merge_wiener_reruns.py [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import pandas as pd

MASTER = Path("data/benchmark_master_combined.csv")
RERUN_DIR = Path("results/rerun")

# source stem -> (model, dataset, training)
SOURCES = {
    "rerun_01_mvtec_patchcore_clean": ("PatchCore", "MVTec-AD", "clean"),
    "rerun_02_mvtec_patchcore_aug":   ("PatchCore", "MVTec-AD", "augmented"),
    "rerun_03_mvtec_padim_clean":     ("PaDiM",     "MVTec-AD", "clean"),
    "rerun_04_mvtec_padim_aug":       ("PaDiM",     "MVTec-AD", "augmented"),
    "rerun_05_visa_patchcore_clean":  ("PatchCore", "VisA",     "clean"),
    "rerun_06_visa_patchcore_aug":    ("PatchCore", "VisA",     "augmented"),
    "rerun_07_visa_padim_clean":      ("PaDiM",     "VisA",     "clean"),
    "rerun_08_visa_padim_aug":        ("PaDiM",     "VisA",     "augmented"),
}

KEY = ["model", "dataset", "category", "seed", "ctype", "severity", "rescue", "training"]
EXPECTED_ROWS_PER_PAIR = 4


def load_reruns() -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(str(RERUN_DIR / "*"))):
        stem = os.path.basename(path).split(".")[0]
        if stem not in SOURCES:
            raise SystemExit(f"Unrecognised rerun file: {path}")
        model, dataset, training = SOURCES[stem]
        df = pd.read_csv(path)

        # the file must actually contain what its name claims
        assert set(df.model.unique()) == {model}, f"{stem}: model mismatch"
        if "dataset" in df.columns:
            assert set(df.dataset.unique()) == {dataset}, f"{stem}: dataset mismatch"
        assert set(df.phase.unique()) == {"rescue"}, f"{stem}: non-rescue rows present"
        assert set(df.ctype.unique()) <= {"gaussian_blur", "motion_blur"}, f"{stem}: unexpected ctype"
        assert set(df.severity.unique()) <= {"mild", "moderate"}, f"{stem}: unexpected severity"
        assert df.rescue.str.startswith("Wiener").all(), f"{stem}: non-Wiener rescue"
        assert df.image_AUROC.notna().all(), f"{stem}: null AUROC"

        counts = df.groupby(["category", "seed"]).size()
        bad = counts[counts != EXPECTED_ROWS_PER_PAIR]
        assert bad.empty, f"{stem}: incomplete pairs\n{bad}"

        df["dataset"] = dataset
        df["training"] = training
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    assert not out.duplicated(subset=KEY).any(), "duplicate keys across rerun files"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    new = load_reruns()
    master = pd.read_csv(MASTER)
    before = len(master)

    target = (
        (master.phase == "rescue")
        & (master.rescue.str.startswith("Wiener"))
        & (master.severity.isin(["mild", "moderate"]))
    )
    print(f"rerun rows: {len(new)} | master rows to replace: {int(target.sum())}")
    assert int(target.sum()) == len(new), "row-count mismatch between master and reruns"

    joined = (
        master[target].reset_index()
        .merge(new[KEY + ["image_AUROC"]], on=KEY, how="outer",
               suffixes=("_old", "_new"), indicator=True)
    )
    counts = joined._merge.value_counts().to_dict()
    print(f"join: {counts}")
    assert counts.get("left_only", 0) == 0 and counts.get("right_only", 0) == 0, \
        "join is not 1:1 -- refusing to write"

    joined = joined.set_index("index")
    changed = int((joined.image_AUROC_old != joined.image_AUROC_new).sum())
    delta = (joined.image_AUROC_new - joined.image_AUROC_old).mean()
    print(f"values changed: {changed}/{len(joined)} | mean change: {delta:+.4f}")

    master.loc[joined.index, "image_AUROC"] = joined["image_AUROC_new"]

    assert len(master) == before, "row count changed"
    assert master.image_AUROC.notna().all(), "nulls introduced"
    assert not master.duplicated(
        subset=["model", "dataset", "category", "seed", "phase",
                "ctype", "severity", "rescue", "training"]).any(), "duplicate keys introduced"

    if args.dry_run:
        print("dry run -- master not written")
        return
    master.to_csv(MASTER, index=False)
    print(f"wrote {MASTER} ({len(master)} rows)")


if __name__ == "__main__":
    main()
