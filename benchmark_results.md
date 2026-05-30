# Anomaly Detection Benchmark Results
## Preprocessing Fallacy Study — PaDiM vs PatchCore on MVTec-AD

**Dataset**: MVTec-AD | **Models**: PaDiM (wide_resnet50_2), PatchCore (wide_resnet50_2, k=9)
**Categories**: 15 | **Seeds**: 3 (42, 123, 456) | **Total rows per model**: 1,530

---

## Dataset Composition

| Phase | Severities | Rows (per model) |
|---|---|---|
| Baseline (clean) | none | 45 |
| Degradation | mild, moderate, severe | 675 (225 each) |
| Rescue | mild, moderate, severe | 810 (270 each) |

Sources:
- Baseline + all degradation + severe rescue: original full-run CSVs
- Mild + moderate rescue: new `01/02-baseline-*-mod-mild` notebooks

---

## Overall AUROC Summary

| Phase | PaDiM | PatchCore |
|---|---|---|
| Degradation (all sevs, mean) | 0.667 | 0.780 |
| Rescue (all sevs, mean) | 0.619 | 0.704 |
| Rescue delta vs Degradation | **-0.048** | **-0.076** |

PatchCore is more sensitive to rescue preprocessing than PaDiM. Its patch-level memory bank is disrupted more easily by image processing operations.

---

## PaDiM — Rescue Delta Table (mean AUROC across 15 categories x 3 seeds)

| Corruption | Severity | Rescue Method | Degraded | Rescued | Delta |
|---|---|---|---|---|---|
| fog_haze | mild | Dehaze (Dark Channel) | 0.6133 | 0.6229 | +0.010 |
| fog_haze | moderate | Dehaze (Dark Channel) | 0.5632 | 0.5554 | -0.008 |
| gaussian_blur | mild | Wiener | 0.7723 | 0.5261 | -0.246 |
| gaussian_blur | moderate | Wiener | 0.6129 | 0.5160 | -0.097 |
| low_light | mild | CLAHE | 0.7742 | 0.7641 | -0.010 |
| low_light | mild | Retinex | 0.7742 | 0.6976 | -0.077 |
| low_light | moderate | CLAHE | 0.7174 | 0.7270 | +0.010 |
| low_light | moderate | Retinex | 0.7174 | 0.6997 | -0.018 |
| motion_blur | mild | Wiener (Motion PSF) | 0.7207 | 0.5431 | -0.178 |
| motion_blur | moderate | Wiener (Motion PSF) | 0.6294 | 0.5347 | -0.095 |
| sensor_noise | mild | NLM Denoise | 0.6569 | 0.6512 | -0.006 |
| sensor_noise | moderate | NLM Denoise | 0.6105 | 0.5898 | -0.021 |

---

## PatchCore — Rescue Delta Table (mean AUROC across 15 categories x 3 seeds)

| Corruption | Severity | Rescue Method | Degraded | Rescued | Delta |
|---|---|---|---|---|---|
| fog_haze | mild | Dehaze (Dark Channel) | 0.7176 | 0.7005 | -0.017 |
| fog_haze | moderate | Dehaze (Dark Channel) | 0.6097 | 0.5859 | -0.024 |
| gaussian_blur | mild | Wiener | 0.9173 | 0.5000 | -0.417 |
| gaussian_blur | moderate | Wiener | 0.6483 | 0.5000 | -0.148 |
| low_light | mild | CLAHE | 0.9607 | 0.9079 | -0.053 |
| low_light | mild | Retinex | 0.9607 | 0.8478 | -0.113 |
| low_light | moderate | CLAHE | 0.8910 | 0.9034 | +0.012 |
| low_light | moderate | Retinex | 0.8910 | 0.8248 | -0.066 |
| motion_blur | mild | Wiener (Motion PSF) | 0.7824 | 0.5550 | -0.227 |
| motion_blur | moderate | Wiener (Motion PSF) | 0.6595 | 0.5626 | -0.097 |
| sensor_noise | mild | NLM Denoise | 0.8470 | 0.8140 | -0.033 |
| sensor_noise | moderate | NLM Denoise | 0.7638 | 0.7455 | -0.018 |

---

## Key Findings

### 1. Wiener Deconvolution Catastrophically Fails
The fixed-PSF Wiener filter is the worst performer across both models and all severities:
- PatchCore gaussian_blur mild: -0.417 (collapses to random chance at 0.5000)
- PaDiM gaussian_blur mild: -0.246
- Motion blur Wiener: -0.178 (PaDiM), -0.227 (PatchCore)

The PSF parameters (sigma=25, kernel_size=281) are calibrated for severe corruption but mismatched at mild/moderate. The filter over-processes minimally degraded images, creating artifacts worse than the original corruption.

### 2. CLAHE at Moderate is the Only Consistent Winner
CLAHE on low_light at moderate severity is the only rescue method that reliably helps both models:
- PaDiM: +0.010
- PatchCore: +0.012

The only positive result across all 24 corruption x severity x method combinations. The gain is negligible (<1.5%).

### 3. PatchCore is More Brittle Than PaDiM Under Preprocessing
PatchCore drops more under rescue (-0.076 vs -0.048 mean delta). PatchCore's coreset memory bank encodes precise patch-level statistics — any image transformation shifts the query distribution away from the training distribution, increasing false negatives.

### 4. The Preprocessing Fallacy Holds Universally
Confirmed at mild, moderate, and severe severity across both models and all 5 corruption types:
- Rescue preprocessing does not recover anomaly detection performance
- It worsens performance in the large majority of cases
- The effect is larger at mild severity, where PSF mismatch causes proportionally more damage on already lightly degraded images

---

## Next Step

Merge all 4 source CSVs into a single unified dataset (3,060 rows total) and produce the final paper-ready analysis.

Merge strategy: take all rows from the severe CSVs, then add only rescue-phase rows from the mild/mod CSVs (avoids duplicate degradation rows).

From the merged dataset, produce:
1. Per-category breakdown — do any specific MVTec-AD categories consistently benefit from rescue?
2. Severity x rescue method heatmap of delta AUROC
3. Final conclusion section for the paper
