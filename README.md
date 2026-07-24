# Robust Industrial Anomaly Detection Under Adverse Imaging Conditions

A rigorous 4-way comparative benchmark evaluating **test-time rescue preprocessing** vs. **training-time data augmentation** for anomaly detection robustness on MVTec-AD.

## Status: ✅ Statistical Significance & Analysis Complete

**6,120 benchmark rows collected, verified, and statistically validated via Wilcoxon signed-rank tests (with Benjamini-Hochberg FDR correction).** All 9 analysis figures (6 descriptive, 3 statistical) are generated and saved in `results/analysis/`.

---

## Key Results (Headline Numbers)

| Condition | Mean Deg. AUROC | Rescue Success Rate | Mean Rescue Δ |
|---|---|---|---|
| PatchCore — Clean training | 0.7356 | 25.8% | −0.0717 |
| PatchCore — Augmented training | **0.8588** (+12.3 pp) | 22.7% | −0.1490 |
| PaDiM — Clean training | 0.6390 | 35.1% | −0.0456 |
| PaDiM — Augmented training | **0.7400** (+10.1 pp) | 19.4% | −0.1134 |

**Core finding**: Three phenomena are established with statistical significance:
1. **Augmented training significantly improves robustness** (+12.3 pp PatchCore, +10.1 pp PaDiM, p < 0.001) by expanding the learned normality manifold into corruption-domain feature space.
2. **Rescue preprocessing is net-harmful** in all 4 conditions — the *preprocessing fallacy* — because restoration algorithms create a third distribution (distinct from both clean training and corrupted images) whose artifacts anomaly detectors score as defects.
3. **Harm is method-specific**: Wiener deconvolution (−14 to −37 pp via Gibbs spectral ringing) is catastrophic; CLAHE is the only conditionally safe method due to its spatial frequency-preserving contrast adjustment.

---

## Experiment Design

### Models
- **PatchCore**: `backbone=wide_resnet50_2`, `num_neighbors=9`, `max_epochs=1`
- **PaDiM**: `backbone=wide_resnet50_2`, `layers=[layer1,layer2,layer3]`, `n_features=100`, `max_epochs=1`

### 4 Training Conditions
| Condition | Training data | Description |
|---|---|---|
| PatchCore — Clean | Standard MVTec-AD train split | No augmentation |
| PatchCore — Augmented | 50% randomly corrupted train images | `prepare_augmented_train_data()` |
| PaDiM — Clean | Standard MVTec-AD train split | No augmentation |
| PaDiM — Augmented | 50% randomly corrupted train images | `prepare_augmented_train_data()` |

### Corruption Types (5 × 3 = 15 conditions)
| Type | Mild | Moderate | Severe |
|---|---|---|---|
| Low Light | γ=0.65 | γ=0.35 | γ=0.15 |
| Gaussian Blur | σ=1, k=5 | σ=3, k=15 | σ=5, k=25 |
| Motion Blur | k=7 | k=15 | k=25 |
| Sensor Noise | var=0.005 | var=0.02 | var=0.05 |
| Fog/Haze | coef 0.2–0.4 | coef 0.4–0.6 | coef 0.6–0.8 |

### Rescue Methods (6 streams × 3 severities = 18 conditions)
| Corruption | Rescue |
|---|---|
| Low-light | CLAHE + Retinex (2 methods) |
| Gaussian blur | Wiener deconvolution (Gaussian PSF) |
| Motion blur | Wiener deconvolution (Motion PSF) |
| Sensor noise | Non-Local Means (NLM) |
| Fog/haze | Dark Channel Prior dehaze |

### Scope
- 15 categories × 3 seeds = 45 pairs per condition
- Per pair: 1 baseline + 15 degradation + 18 rescue = **34 rows**
- Per condition: **1,530 rows**
- **Total: 6,120 rows across 4 conditions**

---

## Running on Kaggle

### For clean training runs (already complete):
Use `patchcore_run2_severe_complete.ipynb` / `padim_run1_severe_complete.ipynb` as reference.

### For augmented training runs (already complete):
Upload `run_patchcore_augmented.py` / `run_padim_augmented.py` as notebook source.
Input: previous run's CSV from `patchcore_augmented.csv` / `padim_augmented.csv`.

### Session management:
- Resume logic reads existing CSV and skips completed (category, seed) pairs
- Results saved after **every** `engine.test()` call
- GPU memory freed after each pair with `del model; torch.cuda.empty_cache()`
- 12-hour session limit: expect ~6–8 categories per session for augmented runs

---

## What's Left Before Paper Writing

| Task | Priority | Notes |
|---|---|---|
| Statistical significance tests | **✅ Complete** | Paired Wilcoxon signed-rank tests run and plotted in `run_wilcoxon_tests.ipynb` |
| Figure polish | **High** | Colorblind palette, LaTeX axis labels if CVPR/IEEE |
| VisA generalization | **Medium** | Adds cross-dataset credibility; needed for top-venue papers |
| Write paper | — | Full structure in `benchmark_report.md` |

---

## Statistical Significance (Wilcoxon Signed-Rank Tests)

We performed paired Wilcoxon signed-rank tests (non-parametric, paired) on the unified dataset of 6,120 rows, with **Benjamini-Hochberg FDR correction** applied to the 24 per-method comparisons in Test 3. Results are fully documented in [run_wilcoxon_tests.ipynb](run_wilcoxon_tests.ipynb).

### 1. Augmentation Gains (H₁: Clean < Augmented)
Tests whether training-time corruption augmentation yields statistically significant improvements under test-time degradation.
* **PatchCore** (N = 675 pairs): Mean Gain = **+12.32 pp** ($p = 3.25 \times 10^{-69}$, ***)
* **PaDiM** (N = 675 pairs): Mean Gain = **+10.09 pp** ($p = 5.32 \times 10^{-75}$, ***)

**Mechanism**: Augmented training expands the normality manifold to encompass corruption-domain features, reducing nearest-neighbour distance inflation (PatchCore) and Mahalanobis distance inflation (PaDiM) for corrupted-but-normal inputs.

*Visualized in `results/analysis/07_wilcoxon_augmentation_gains.png`.*

### 2. Rescue Preprocessing Effectiveness (Two-sided Wilcoxon)
Tests whether test-time rescue preprocessing changes AUROC relative to degraded-only baselines.
* **PaDiM (Clean)**: Mean Delta = **−4.56 pp** ($p < 0.001$, significantly harmful)
* **PaDiM (Augmented)**: Mean Delta = **−11.34 pp** ($p < 0.001$, significantly harmful)
* **PatchCore (Clean)**: Mean Delta = **−7.17 pp** ($p < 0.001$, significantly harmful)
* **PatchCore (Augmented)**: Mean Delta = **−14.90 pp** ($p < 0.001$, significantly harmful)

**Mechanism (The Preprocessing Fallacy)**: Restoration algorithms create a *third distribution* — distinct from both the clean training set and naturally corrupted images — containing restoration-specific artifacts (Gibbs ringing, illumination maps, over-smoothing). Anomaly detectors score distance from the learned normality distribution; restored images fall further from it than the raw corrupted inputs.

*Visualized in `results/analysis/08_wilcoxon_rescue_deltas.png`.*

### 3. Per-Method Breakdown (FDR-corrected)
When analyzing individual rescue methods, tested with BH-FDR correction across 24 comparisons:
* **Wiener Deconvolution** is **catastrophically harmful** (q < 0.001), dropping AUROC by **−14 to −37 pp**. Mechanism: Wiener inversion amplifies noise into Gibbs spectral ringing at every edge, creating patch embeddings orthogonal to the entire clean coreset.
* **Retinex** is consistently harmful (q < 0.001, −3 to −11 pp) via colour channel distortion and high-frequency compression.
* **CLAHE** is the only conditionally safe method: near-neutral for PatchCore (clean/augmented: ~−0.2 pp, n.s. after FDR). Mechanism: CLAHE adjusts local luminance only, preserving spatial frequencies that deep feature extractors rely on. Exception: PaDiM (augmented) −5.5 pp (q < 0.001) due to spatially non-uniform contrast falling outside uniform brightness augmentation coverage.
* **NLM Denoise** and **Dehaze (Dark Channel)**: mild but significant harm (−1 to −5 pp, q < 0.05).

*Visualized in `results/analysis/09_wilcoxon_per_method_heatmap.png`.*

---

## Visualizations & Figures

All generated plots are saved in `results/analysis/` and categorized as follows:

### Descriptive Analysis Figures
1. **[01_baseline_comparison.png](file:///c:/Users/User/Downloads/Anomaly-Detection-main/Anomaly-Detection-main/results/analysis/01_baseline_comparison.png)**: Category-wise baseline AUROC comparison (PatchCore vs. PaDiM) under clean training and clean testing, establishing initial model performance.
2. **[02_degradation_curves_4way.png](file:///c:/Users/User/Downloads/Anomaly-Detection-main/Anomaly-Detection-main/results/analysis/02_degradation_curves_4way.png)**: Trajectories of AUROC degradation across three severity levels (mild, moderate, severe) for five physical corruptions across all four training conditions.
3. **[03_rescue_heatmaps.png](file:///c:/Users/User/Downloads/Anomaly-Detection-main/Anomaly-Detection-main/results/analysis/03_rescue_heatmaps.png)**: Detailed heatmaps representing the absolute change in AUROC ($\Delta$) after applying specific classical rescue methods.
4. **[04_robustness_gains_summary.png](file:///c:/Users/User/Downloads/Anomaly-Detection-main/Anomaly-Detection-main/results/analysis/04_robustness_gains_summary.png)**: Summary bar chart highlighting the average robustness gains of training-time augmentation over clean training across corruption types.
5. **[05_relative_harm_scatter.png](file:///c:/Users/User/Downloads/Anomaly-Detection-main/Anomaly-Detection-main/results/analysis/05_relative_harm_scatter.png)**: Scatter plot of degraded AUROC vs. rescued AUROC, visually illustrating the "Preprocessing Fallacy" (most rescue methods fall below the $y=x$ parity line).
6. **[06_category_sensitivity.png](file:///c:/Users/User/Downloads/Anomaly-Detection-main/Anomaly-Detection-main/results/analysis/06_category_sensitivity.png)**: Heatmap showing how sensitive different object/texture categories are to corruptions under clean vs. augmented training.

### Statistical Wilcoxon Figures
7. **[07_wilcoxon_augmentation_gains.png](results/analysis/07_wilcoxon_augmentation_gains.png)**: Bar chart showing average robustness gains from augmented training for PatchCore and PaDiM, annotated with W-statistics, one-sided Wilcoxon significance (*** p < 0.001), and absolute baseline AUROC for context.
8. **[08_wilcoxon_rescue_deltas.png](results/analysis/08_wilcoxon_rescue_deltas.png)**: Bar chart showing the overall impact of classical rescue preprocessing, proving it is significantly harmful in all four conditions (*** two-sided p < 0.001), with per-bar baseline AUROC callouts. Note: aggregate across all 6 rescue methods; Wiener dominates the negative effect.
9. **[09_wilcoxon_per_method_heatmap.png](results/analysis/09_wilcoxon_per_method_heatmap.png)**: Heatmap of BH-FDR corrected Wilcoxon results per rescue method across all training conditions and models. n.s. shown explicitly. CLAHE is near-neutral for PatchCore; Wiener is catastrophic.

---

## Dataset
- **MVTec-AD**: 15 categories (Kaggle: `ipythonx/mvtec-ad`)
- **VisA** (optional): 12 categories (Kaggle: `marquis03/visa-dataset`)
