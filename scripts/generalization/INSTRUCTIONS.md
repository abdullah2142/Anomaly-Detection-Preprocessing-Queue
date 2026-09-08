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
| `unconditional_rescue.ipynb` | clean data (no augmentation) | undegraded images, and mismatched corruptions | What does preprocessing cost when it wasn't needed, or when the degradation is misidentified? |
| `feature_space_probe.ipynb` | clean data (no augmentation) | clean, corrupted and rescued images | Where do rescued images actually land relative to normal? Direct evidence for the preprocessing fallacy. |

The third notebook addresses a different gap. The benchmark applies each rescue
only to the corruption it targets, chosen using that corruption's ground-truth
identity (Limitations 13 and 14). Deployment has neither guarantee: it
preprocesses a whole stream, most of which may be undegraded, and its detector
can misclassify. Part A applies all six rescues to clean images; Part B applies
five plausible *wrong* rescues to corrupted images. Both comparison baselines --
the clean baseline AUROC and the degradation AUROC -- already exist in the master
CSV, so only the new arm runs.

This matters most for the paper's one positive recommendation, "CLAHE is
conditionally safe for low-light PatchCore", which currently assumes a
low-light detector gates it. Analyse with
`scripts/analysis/analyze_unconditional_rescue.py`.

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
| `unconditional_rescue` | 16 | 336 | ~3 h |
| `feature_space_probe` | 10 | 120 | ~1.2 h |
| **Total** | 122 | 776 | **~10 h** |

`unconditional_rescue` emits 21 rows per (category, model): 6 rescues on clean
images, plus 5 confusions x 3 severities.

Both models train on the same augmented dataset before it is deleted, so the
second model costs no extra data preparation.

Each notebook fits in one Kaggle session with the 11.5 h graceful timeout as
backstop. Resume is row-count based: a unit is complete only at
`ROWS_PER_UNIT` rows (6 per category x held-out type for LOCO, 10 per category
for the severity holdout, 21 per category x model for unconditional rescue);
partial units are discarded and re-run.

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

## `feature_space_probe` — reading the result

The paper names its contribution the *preprocessing fallacy* and defines it as a
mechanism: restoration does not return a corrupted image to the clean
distribution, it creates a **third distribution** further from normal than the
corruption was. AUROC cannot test that — it measures only the ranking of scores,
never how far anything sits from normal. The distance is already computed (it *is*
the anomaly score) and the benchmark discards it, keeping only `image_AUROC`.

This notebook records the raw per-image score for clean, corrupted and
corrupted-then-rescued images, on 5 categories × 2 models, at moderate severity.
Analysis restricts to **normal** test images — they contain no defect, so a rise
in their score is the model reacting to something that is not a defect.

```bash
python scripts/analysis/analyze_feature_space.py results/feature_space_probe.txt
```

It reports the three predictions in order. The third is the one the paper asserts
and has never measured:

    score(clean)  <  score(degraded)  <  score(rescued)

**If the third inequality holds**, the fallacy is demonstrated rather than
inferred, and the abstract's mechanism claim is earned. **If it does not**, the
honest response is to soften that claim to a hypothesis — the AUROC results stand
either way, since they never depended on the mechanism being right.

⚠️ **One way this run can be invalid.** Anomalib may min-max normalise scores. If
it renormalises each run separately, every condition gets pinned to [0, 1] and the
magnitudes become meaningless. The notebook disables normalisation where the
installed API allows, and the analysis detects the failure mode and refuses to
conclude rather than reporting a rescaled artefact. If it refuses, fix the engine
configuration and re-run — do not interpret the numbers.
