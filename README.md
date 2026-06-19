# Robust Industrial Anomaly Detection Under Adverse Imaging Conditions

A rigorous 4-way comparative benchmark evaluating **test-time rescue preprocessing** vs. **training-time data augmentation** for anomaly detection robustness on MVTec-AD.

## Status: ✅ Data Collection Complete

**6,120 benchmark rows collected and verified.** All analysis figures generated.

---

## Key Results (Headline Numbers)

| Condition | Mean Deg. AUROC | Rescue Success Rate | Mean Rescue Δ |
|---|---|---|---|
| PatchCore — Clean training | 0.7356 | 25.8% | −0.0717 |
| PatchCore — Augmented training | **0.8588** (+12.3 pp) | 22.7% | −0.1490 |
| PaDiM — Clean training | 0.6390 | 35.1% | −0.0456 |
| PaDiM — Augmented training | **0.7400** (+10.1 pp) | 19.4% | −0.1134 |

**Core finding**: Augmented training significantly improves corruption robustness (up to +24.6 pp for PatchCore/Gaussian blur/severe). Rescue preprocessing is net-harmful in all four conditions — and is *more* harmful when applied to augmented-trained models.

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
| Statistical significance tests | **High** | Wilcoxon signed-rank on rescue deltas and aug-training gains |
| Figure polish | **High** | Colorblind palette, LaTeX axis labels if CVPR/IEEE |
| VisA generalization | **Medium** | Adds cross-dataset credibility; needed for top-venue papers |
| Write paper | — | Full structure in `benchmark_report.md` |

---

## Dataset
- **MVTec-AD**: 15 categories (Kaggle: `ipythonx/mvtec-ad`)
- **VisA** (optional): 12 categories (Kaggle: `marquis03/visa-dataset`)
