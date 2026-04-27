# Robust Industrial Anomaly Detection Under Adverse Imaging Conditions

Systematic benchmarking of anomaly detection models (PatchCore, PaDiM) under
real-world image degradations (low-light, blur, noise, fog) and evaluation of
classical preprocessing methods as mitigation strategies.

## Project Structure

```
anomaly-detection/
├── experiment_config.json          # Locked corruption parameters & seeds
├── anomaly_detection_briefing.md   # Research requirements document
├── notebooks/
│   ├── shared_utils.py             # Shared corruption/preprocessing functions
│   ├── 00_setup_and_verify.py      # NB-0: Environment setup
│   ├── 01_baselines_patchcore.py   # NB-1: PatchCore clean baselines
│   ├── 02_baselines_padim.py       # NB-2: PaDiM clean baselines
│   ├── 03_severity_calibration.py  # NB-3: Visual severity verification
│   ├── 04_degradation_patchcore.py # NB-4: PatchCore degradation curves
│   ├── 05_degradation_padim.py     # NB-5: PaDiM degradation curves
│   ├── 06a_preprocessing_patchcore_blur_light.py  # NB-6a: Preprocessing (blur/light)
│   ├── 06b_preprocessing_patchcore_noise_fog.py   # NB-6b: Preprocessing (noise/fog)
│   ├── 07a_preprocessing_padim_blur_light.py      # NB-7a: PaDiM preprocessing (blur/light)
│   ├── 07b_preprocessing_padim_noise_fog.py       # NB-7b: PaDiM preprocessing (noise/fog)
│   ├── 08_training_corruption.py   # NB-8: Training-set corruption experiment
│   ├── 09_visa_generalization.py   # NB-9: VisA generalization (stretch)
│   └── 10_analysis_and_figures.py  # NB-10: Final analysis & figures
```

## Experiment Pipeline

```
NB-0 (setup) ─┬─► NB-1 (PatchCore) ──► NB-4 (degradation) ──► NB-6a/6b (preprocessing)
               ├─► NB-2 (PaDiM) ──────► NB-5 (degradation) ──► NB-7a/7b (preprocessing)
               ├─► NB-3 (calibration)
               └─► NB-8 (training corruption)
                                                                         ↓
                                                               NB-10 (analysis)
```

## Key Design Decisions

- **On-the-fly corruption**: Applied during inference, not saved to disk
- **Kaggle-native**: All notebooks designed for Kaggle's GPU environment
- **Session-safe**: Intermediate results saved after each category run
- **Error-resilient**: try/except around each experiment to prevent total failure

## Running on Kaggle

1. Upload each `.py` file as a Kaggle notebook (paste cell by cell)
2. Add `ipythonx/mvtec-ad` as input dataset
3. Chain notebooks: each notebook's output → next notebook's input
4. NB-1, NB-2, NB-3 can run in **parallel** (independent)
5. NB-4/5 depend on NB-1/2 respectively
6. NB-10 requires all prior outputs

## Corruption Types

| Type | Mild | Moderate | Severe |
|------|------|----------|--------|
| Low Light | γ=0.5 | γ=0.35 | γ=0.2 |
| Gaussian Blur | σ=1.0, k=7 | σ=2.0, k=11 | σ=4.0, k=23 |
| Motion Blur | k=7 | k=15 | k=25 |
| Sensor Noise | σ²=0.02 | σ²=0.05 | σ²=0.10 |
| Fog/Haze | 0.2–0.35 | 0.45–0.65 | 0.75–0.95 |
| Combined | Low-light (γ=0.35) + Noise (σ²=0.05) |

## Preprocessing Methods

| Corruption | Method 1 | Method 2 | Baseline |
|-----------|----------|----------|----------|
| Low Light | CLAHE | — | HistMatch |
| Gaussian Blur | Wiener Deconv | — | HistMatch |
| Motion Blur | Unsharp Mask | — | HistMatch |
| Sensor Noise | NLM Denoise | (BM3D optional) | HistMatch |
| Fog/Haze | Dark Channel Prior | — | HistMatch |

## Requirements

Kaggle pre-installs most dependencies. Additional:
- `anomalib` (installed with `--no-deps` to protect CUDA)
- `lightning`, `albumentationsx`, `jsonargparse`, `docstring_parser`, `rich`
- Optional: `bm3d` for BM3D denoising

## Dataset

- **MVTec-AD**: 15 categories, 5354 images (Kaggle: `ipythonx/mvtec-ad`)
- **VisA** (stretch): 12 categories (Kaggle: search "visa anomaly")
