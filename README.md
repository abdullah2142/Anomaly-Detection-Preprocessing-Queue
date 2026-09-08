# Robust Industrial Anomaly Detection Under Adverse Imaging Conditions

A rigorous 4-way comparative benchmark evaluating **test-time rescue preprocessing** vs. **training-time data augmentation** for anomaly detection robustness on MVTec-AD and VisA.

## Status

- **MVTec-AD**: ✅ Complete — 6,120 benchmark rows (4 conditions × 2 models × 15 categories × 3 seeds)
- **VisA (clean training)**: ✅ Complete — 12 categories, 2 models, 3 seeds
- **VisA (augmented training)**: ✅ Complete — 4 categories (candle, cashew, pcb1, pipe_fryum), 2 models, 3 seeds
- **Statistical validation**: ✅ Category-clustered permutation tests — see [`docs/statistical_validation.md`](docs/statistical_validation.md)
- **Generalization controls**: ✅ Complete — leave-one-corruption-out and severity holdout, see [`docs/generalization_controls.md`](docs/generalization_controls.md)
- **Wiener PSF rerun**: ✅ Complete — all 1,104 mild/moderate Wiener rows regenerated with per-severity PSFs and merged ([`scripts/wiener_reruns/`](scripts/wiener_reruns/), [`scripts/analysis/merge_wiener_reruns.py`](scripts/analysis/merge_wiener_reruns.py))
- **Total benchmark database**: 9,384 rows in `data/benchmark_master_combined.csv`

---

## Key Results

| Condition | Mean Deg. AUROC | Rescue Success Rate | Mean Rescue Δ |
|---|---|---|---|
| PatchCore — Clean training | 0.7356 | 29.8% | −0.0394 |
| PatchCore — Augmented training | **0.8588** (+12.3 pp) | 24.0% | −0.1179 |
| PaDiM — Clean training | 0.6390 | 38.4% | −0.0275 |
| PaDiM — Augmented training | **0.7400** (+10.1 pp) | 21.1% | −0.0955 |

**Core findings:**
1. **Augmented training significantly improves robustness** (+12.3 pp PatchCore, +10.1 pp PaDiM on MVTec-AD, category-clustered p ≤ 1.2×10⁻⁴) — but the benefit is **part robustness, part distribution matching**. Against corruptions and severities never seen in training, PatchCore keeps ~⅓ of its gain; PaDiM keeps none. See Generalization Controls below.
2. **Rescue preprocessing is net-harmful** in all 4 conditions — the *preprocessing fallacy* — even when the corruption type, severity, and restoration parameters are all known exactly (no degradation-detection step is included). Probing the detectors' own distance-from-normality scores shows restoration **fails to return images to the clean distribution**, but does *not* displace them further than the corruption did — see [`docs/feature_space_evidence.md`](docs/feature_space_evidence.md).
3. **Wiener deconvolution is the most harmful rescue** at every severity, even with an oracle PSF matched to the blur that generated the image: −5.7 pp (mild), −15.1 pp (moderate), −12.5 pp (severe) for Gaussian blur. Misspecifying the kernel roughly 5× makes it far worse (−32.2 pp at mild), quantifying the cost of blind deconvolution.

---

## Experiment Design

### Models
- **PatchCore**: `backbone=wide_resnet50_2`, `num_neighbors=9`, `max_epochs=1`
- **PaDiM**: `backbone=wide_resnet50_2`, `layers=[layer1,layer2,layer3]`, `n_features=100`, `max_epochs=1`

### Corruption Types (5 × 3 = 15 conditions)

Parameters match `data/experiment_config.json` — the authoritative source.

| Type | Mild | Moderate | Severe |
|---|---|---|---|
| Low Light | γ=0.50 | γ=0.35 | γ=0.20 |
| Gaussian Blur | σ=5, k=31 | σ=15, k=101 | σ=25, k=281 |
| Motion Blur | k=31 | k=81 | k=151 |
| Sensor Noise | Gaussian var=0.05 + 5% S&P | Gaussian var=0.18 + 5% S&P | Gaussian var=0.35 + 5% S&P |
| Fog/Haze | coef 0.20–0.35 | coef 0.45–0.65 | coef 0.75–0.90 |

> **Note on Sensor Noise**: `apply_sensor_noise` adds 5% salt-and-pepper impulse noise on top of the Gaussian noise at all severities. This is undocumented in early versions of this README. NLM denoising is suboptimal for the impulse component; a median filter would be more appropriate.

### Rescue Methods (6 streams × 3 severities = 18 conditions)
| Corruption | Rescue |
|---|---|
| Low-light | CLAHE (clip=3.0, tile=8×8) + Retinex (SSR, σ=30) |
| Gaussian blur | Wiener deconvolution (PSF matched to corruption severity) |
| Motion blur | Wiener deconvolution (motion PSF matched to corruption severity) |
| Sensor noise | Non-Local Means (NLM) |
| Fog/haze | Dark Channel Prior dehaze (ω=0.95, patch=15) |

### 4 Training Conditions
| Condition | Training data | Description |
|---|---|---|
| PatchCore — Clean | Standard train split | No augmentation |
| PatchCore — Augmented | 50% randomly corrupted train images | `prepare_augmented_train_data()` |
| PaDiM — Clean | Standard train split | No augmentation |
| PaDiM — Augmented | 50% randomly corrupted train images | `prepare_augmented_train_data()` |

### Scope
- 15 MVTec-AD categories × 3 seeds = 45 pairs per condition
- Per pair: 1 baseline + 15 degradation + 18 rescue = **34 rows**
- Per condition: **1,530 rows**
- **MVTec-AD total: 6,120 rows across 4 conditions**
- **VisA total: 3,264 rows (clean: 12 categories; augmented: 4 categories)**

---

## Statistical Validation

Significance is reported **clustered by category**, the level at which observations
are independent. `notebooks/10_wilcoxon_testing.ipynb` treats each
(category, seed, corruption, severity) cell as independent, which is
pseudoreplicated — 15 categories resampled 45 times each — and inflates its
p-values by many orders of magnitude. Point estimates are identical either way.
Authoritative numbers: [`docs/statistical_validation.md`](docs/statistical_validation.md),
regenerated by `scripts/analysis/cluster_robust_stats.py`.

### Augmentation Gains (H₁: Clean < Augmented)
* **PatchCore** (MVTec-AD): Mean Gain = **+12.3 pp** (15 categories, exact clustered p = 1.2×10⁻⁴)
* **PaDiM** (MVTec-AD): Mean Gain = **+10.1 pp** (15 categories, exact clustered p = 6.1×10⁻⁵)
* **PatchCore** (VisA): Mean Gain = **+17.8 pp** — descriptive only (4 categories)
* **PaDiM** (VisA): Mean Gain = **+13.9 pp** — descriptive only (4 categories)

> The VisA augmented arm covers 4 categories. The smallest two-sided p an exact
> clustered test can return on 4 clusters is 0.125, so these gains are reported as
> descriptive evidence of consistency with MVTec-AD, **not** as significance tests.

### Rescue Deltas (category-clustered, two-sided)
All four MVTec-AD conditions are significantly negative: −2.75 to −11.79 pp,
p = 6.1×10⁻⁵ to 2.1×10⁻³.

On VisA, PatchCore clean-trained holds (−3.73 pp, p = 4.9×10⁻⁴), but **PaDiM
clean-trained is −0.61 pp at p = 0.375 — not distinguishable from zero** once the
Wiener PSF is corrected. The preprocessing fallacy should not be claimed for that
condition. VisA augmented (4 categories) remains descriptive only.

### Per-Method Breakdown
All figures below use per-severity oracle PSFs for Wiener.

* **Wiener (Gaussian)**: most harmful overall, −11.07 pp pooled.
* **Wiener (Motion PSF)**: −10.59 pp pooled.
* **Retinex**: consistently harmful, −9.52 pp pooled.
* **NLM Denoise** (−2.65 pp) and **DCP Dehaze** (−2.36 pp): mild harm.
* **CLAHE**: least harmful at −1.61 pp pooled, and near-neutral for PatchCore
  under low-light specifically (−0.16 pp clean, −0.38 pp augmented) — the only
  conditionally safe method.

---

## Generalization Controls

Augmented training draws from the same 5 corruption types × 3 severities used at
test time, so the headline gains are a **corruption-matched oracle bound**. Two
controls withhold one condition from the augmented arm and test only on it
(8 MVTec-AD categories, seed 42). Full tables:
[`docs/generalization_controls.md`](docs/generalization_controls.md).

| Control | Model | Matched gain | Held-out gain | Survives |
|---|---|---|---|---:|
| Leave-one-corruption-out | PatchCore | +10.66 pp | **+3.58 pp** (p = 0.031) | 34% |
| Leave-one-corruption-out | PaDiM | +8.02 pp | −0.22 pp (p = 0.828) | 0% |
| Severity holdout | PatchCore | +13.21 pp | **+4.58 pp** (p = 0.008) | 35% |
| Severity holdout | PaDiM | +8.83 pp | +1.02 pp (p = 0.461) | 12% |

**PatchCore generalises partially; PaDiM does not.** Both controls agree despite
withholding along different axes. Withholding a condition costs −7.1 to −8.6 pp
in every case, and transfer decays with severity (+2.88 pp mild → +0.53 pp
severe).

**Deployment consequence**: augment with the corruptions and severities you
actually expect. Extrapolating beyond them buys little for PatchCore and nothing
measurable for PaDiM.

---

## Visualizations

All generated plots are saved in `results/analysis/`:

| Figure | Description |
|---|---|
| `01_baseline_comparison.png` | Category-wise baseline AUROC (PatchCore vs PaDiM) |
| `02_degradation_curves_4way.png` | AUROC degradation trajectories across severity levels |
| `03_augmented_training_gain.png` | Augmented training gains by corruption type |
| `04_rescue_delta_heatmap.png` | Rescue AUROC delta heatmap |
| `05_rescue_success_rates.png` | Rescue success rates across conditions |
| `06_severity_comparison.png` | Severity-stratified AUROC comparison |
| `07_wilcoxon_augmentation_gains.png` | Wilcoxon test: augmentation gains |
| `08_wilcoxon_rescue_deltas.png` | Wilcoxon test: rescue harm |
| `09_wilcoxon_per_method_heatmap.png` | Per-method Wilcoxon FDR-corrected results (cell-level; see `docs/statistical_validation.md` for clustered inference) |
| `11_generalization_controls.png` | How much of the augmentation gain survives when the tested condition is withheld |
| `10_experimental_pipeline.md` | Mermaid architectural diagram |

---

## Repository Structure

```
├── notebooks/               # Training & testing notebooks (01-08) + analysis (09-10)
│   ├── 01_mvtec_patchcore_clean.ipynb
│   ├── 02_mvtec_patchcore_augmented.ipynb
│   ├── 03_mvtec_padim_clean.ipynb
│   ├── 04_mvtec_padim_augmented.ipynb
│   ├── 05_visa_patchcore_clean.ipynb
│   ├── 06_visa_patchcore_augmented.ipynb
│   ├── 07_visa_padim_clean.ipynb
│   ├── 08_visa_padim_augmented.ipynb
│   ├── 09_severity_calibration.ipynb
│   └── 10_wilcoxon_testing.ipynb
├── docs/                    # Research documentation
├── data/                    # Master CSV + experiment config
│   ├── benchmark_master_combined.csv
│   └── experiment_config.json
├── results/analysis/        # Generated figures
├── scripts/                 # Utility scripts
└── requirements.txt         # Pinned dependencies
```

---

## Known Limitations

1. **Sensor noise includes undocumented salt-and-pepper**: 5% S&P impulse noise is applied on top of Gaussian noise. NLM is suboptimal for impulse noise.
2. **Train/test resolution mismatch**: Training augmentation is applied at raw image resolution (700–1024 px for MVTec, ~1500 px for VisA), then resized to 256×256. Test corruption is applied post-resize at 256×256. Same nominal parameters produce different effective severity.
3. **AUROC floor saturation**: 23% of rows are exactly 0.5000 (fully tied anomaly scores), concentrated at severe corruption. Reported means are censored at this floor.
4. **Matched corruption distributions (quantified)**: Training augmentation uses the same 5 types × 3 severities as the test set, so +12.3 pp is a corruption-matched upper bound. Two withholding controls measure how much survives: PatchCore ~34–35%, PaDiM 0–12%. Run on 8 MVTec-AD categories at one seed; not run on VisA.
5. **Fog non-reproducibility**: `A.RandomFog` is not seeded in the original experiment runs (fixed in current codebase).
6. **Pseudoreplication in notebook 10**: the shipped Wilcoxon notebook treats non-independent cells as independent. Use `scripts/analysis/cluster_robust_stats.py` for the corrected inference.
7. **Oracle corruption identification**: rescue methods are selected by the corruption's ground-truth type and severity. There is no degradation-detection or classification step, so the reported rescue deltas exclude identification error and represent the best case for rescue.
8. **Rescue never applied to mismatched or clean inputs**: each method only ever sees the corruption it targets. Unconditional deployment preprocessing is unmeasured.
9. **Single noise realization in augmentation**: augmented training images share one fixed noise/fog realization per corruption type (fixed seed), while test-time corruption varies per image. Biases measured augmentation gains downward.
10. **MVTec augmented data is seed-invariant**: all three seeds train on byte-identical augmented images (VisA does not share this).
11. **Wiener PSF misspecification (resolved)**: the originally published mild/moderate Wiener rows used the severe-tier kernel. All 1,104 affected rows (22.2% of rescue rows) have been regenerated with per-severity PSFs and merged. The misspecified values are retained in git history and are reported as a deliberate PSF-sensitivity comparison.

---

## Data Provenance

`data/benchmark_master_combined.csv` (9,384 rows) is assembled from several runs.
The committed notebooks are consolidated versions; some rows were produced by
earlier split shards that are recoverable from git history but not from the
current `notebooks/` directory alone.

| Rows | Produced by | Note |
|---|---|---|
| MVTec-AD, 6,120 | `notebooks/01`–`04` (consolidated) | The original runs used per-severity shards plus dedicated rescue runners — `01-baseline-patchcore-mod-mild.ipynb`, `02-padim-baseline-mild-moderate.ipynb`, `run_patchcore_rescue.py`, `run_padim_rescue.py` — added in commit `21ce8ef` (2026-05-30) and since removed. |
| VisA, 3,264 | `notebooks/05`–`08` | Clean training covers 12 categories; augmented training covers 4. |
| Wiener rescue rows at mild/moderate, 1,104 | `scripts/wiener_reruns/rerun_01`–`08` | Regenerated with per-severity PSFs and merged by `scripts/analysis/merge_wiener_reruns.py`, replacing values produced with a hardcoded severe-tier kernel. |

The generalization controls are **not** part of the master CSV. They live in
`results/` as separate files with their own `experiment` column, because they use
a different training regime (one condition withheld) and would not be comparable
if pooled:

| File | Produced by |
|---|---|
| `results/leave_one_corruption_out.txt` | `scripts/generalization/leave_one_corruption_out.ipynb` |
| `results/severity_holdout.txt` | `scripts/generalization/severity_holdout.ipynb` |

Corruption parameters have been stable since commit `21ce8ef` (2026-05-30). The
only later change to `experiment_config.json` (`74b1be7`, 2026-09-04) documented
the salt-and-pepper ratio the code already applied; it altered no pixel. Results
and figures predating that commit remain valid.

---

## Datasets
- **MVTec-AD**: 15 categories (Kaggle: `ipythonx/mvtec-ad`)
- **VisA**: 12 categories (Kaggle: `ess1004/visa-anomaly-detection`) — Clean training: 12 categories; Augmented training: 4 categories (candle, cashew, pcb1, pipe_fryum)

---

## Running on Kaggle

Upload notebooks 01-08 as Kaggle notebooks with the MVTec-AD / VisA datasets attached. Each notebook contains all corruption, rescue, and augmentation functions inline. Resume logic reads existing CSV and skips completed `(category, seed)` pairs. Results are saved after every `engine.test()` call.

**Session estimates**: ~6-8 MVTec-AD categories per 12-hour session; ~8-10 VisA categories per session.
