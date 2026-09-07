# Generalization Controls — Kaggle Execution Guide

Two experiments that answer the strongest objection to the augmentation finding:
training draws from the **same** 5 corruption types × 3 severities used at test
time, so the reported +12.3 / +10.1 pp gains are a *corruption-matched oracle
upper bound* (see `docs/paper_limitations.md`, Limitation 11). A reviewer will
ask whether the models generalised or simply memorised the test distribution.

| Notebook | Trains on | Tests on | Question answered |
|---|---|---|---|
| `leave_one_corruption_out.ipynb` | 4 corruption types | the held-out 5th | Does it transfer to an unseen *kind* of degradation? |
| `severity_holdout.ipynb` | mild + moderate | severe | Does it transfer to degradation *worse* than it trained for? |

## Design

- **Dataset**: MVTec-AD only.
- **Categories** (8, spanning both families): `carpet, grid, leather, bottle,
  cable, capsule, hazelnut, screw`. Eight clusters puts the exact permutation
  test's floor at 2/2⁸ = 0.0078, so a clean result can reach significance —
  five categories could not (floor 0.0625).
- **Seed**: 42 only. Categories, not seeds, are the unit of inference.
- **Models**: PatchCore and PaDiM, identical hyperparameters to the published arms.
- **Rows**: degradation only. No rescue phase — that question is already answered
  across 4,968 rows.
- **Comparison arm**: none needed. The clean-trained rows for every
  (category, corruption, severity) already exist in
  `data/benchmark_master_combined.csv`.

The corruption functions, dataset wrapper and engine setup are lifted verbatim
from `notebooks/02_mvtec_patchcore_augmented.ipynb`, including the fixed
`seed=42` inside the augmentation call. That seed is a known defect
(Limitation 15), but the published arm carries it too — holding it constant
keeps the two arms comparable. The **only** difference from the published
augmented arm is which (corruption, severity) pairs the training data may draw
from.

## Cost

Estimates extrapolated from the observed MVTec reruns (~6 min per
prep + fit + ~9 test passes); treat as ±50%.

| | Trainings | Test passes | Est. GPU |
|---|---:|---:|---:|
| `leave_one_corruption_out` | 80 | 240 | ~4.5 h |
| `severity_holdout` | 16 | 80 | ~1.2 h |
| **Total** | 96 | 320 | **~5.5–6 h** |

Both models train on the same augmented dataset before it is deleted, so the
second model costs no extra data preparation.

Each notebook fits in one Kaggle session with the 11.5 h graceful timeout as
backstop. Resume is row-count based: a unit is complete only at
`ROWS_PER_UNIT` rows (6 per category × held-out type for LOCO, 10 per category
for the severity holdout); partial units are discarded and re-run.

## Running

1. Upload the notebook to Kaggle, attach `ipythonx/mvtec-ad` and
   `experiment_config.json`, enable the **T4 GPU** accelerator.
2. Run all. Results land in `results/<experiment>.csv`.
3. To resume, attach the previous session's CSV output and run again.

## Analysing

Both write the master CSV schema plus `experiment` and `held_out` columns, with
`training` set to `augmented_loco` / `augmented_sevholdout`. Pair each row
against the clean-trained row for the same
`(model, category, seed, ctype, severity)` in the master CSV, then cluster by
category:

```bash
python scripts/analysis/cluster_robust_stats.py --csv <merged.csv>
```

**Reading the result.** If the gain against clean training survives on the
held-out corruption, augmentation generalises and the matched-oracle ceiling
becomes a floor. If it collapses, the honest finding is that augmentation works
only for degradations you can characterise in advance — which is still a useful
and publishable deployment claim, just a narrower one. Either outcome is worth
reporting; do not run this expecting only the favourable one.

## Regenerating the notebooks

```bash
python scripts/generalization/build_notebooks.py
```

Edit `CATEGORIES`, `SEED` or the loop bodies in that script rather than the
generated `.ipynb` files.
