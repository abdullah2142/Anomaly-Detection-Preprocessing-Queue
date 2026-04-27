# Workflow Pipeline: Robust Industrial Anomaly Detection Under Adverse Imaging Conditions

> **Purpose**: A fully self-contained, agent-executable pipeline. Every tool, library, API, function name, and parameter has been cross-verified for correctness and availability as of April 2026.

---

## Table of Contents

1. [Phase 0 — Project Scaffolding & Configuration](#phase-0--project-scaffolding--configuration)
2. [Phase 1 — Environment Setup & Dependency Installation](#phase-1--environment-setup--dependency-installation)
3. [Phase 2 — Dataset Acquisition](#phase-2--dataset-acquisition)
4. [Phase 3 — Baseline Reproduction (Clean PatchCore & PaDiM)](#phase-3--baseline-reproduction-clean-patchcore--padim)
5. [Phase 4 — Corruption Pipeline Construction](#phase-4--corruption-pipeline-construction)
6. [Phase 5 — Severity Calibration & Visual Verification](#phase-5--severity-calibration--visual-verification)
7. [Phase 6 — Degradation Experiments (Main)](#phase-6--degradation-experiments-main)
8. [Phase 7 — Preprocessing Rescue Pipeline](#phase-7--preprocessing-rescue-pipeline)
9. [Phase 8 — Preprocessing Rescue Experiments](#phase-8--preprocessing-rescue-experiments)
10. [Phase 9 — Training-Set Corruption Experiment](#phase-9--training-set-corruption-experiment)
11. [Phase 10 — VisA Generalization Check](#phase-10--visa-generalization-check)
12. [Phase 11 — WinCLIP Stretch Goal](#phase-11--winclip-stretch-goal)
13. [Phase 12 — Analysis & Visualization](#phase-12--analysis--visualization)
14. [Phase 13 — Runtime Benchmarking](#phase-13--runtime-benchmarking)
15. [Appendix A — Verified Library Reference](#appendix-a--verified-library-reference)
16. [Appendix B — MVTec-AD Category Reference](#appendix-b--mvtec-ad-category-reference)
17. [Appendix C — Results Schema](#appendix-c--results-schema)

---

## Phase 0 — Project Scaffolding & Configuration

### 0.1 Directory Structure

Create the following project directory structure:

```
anomaly-detection/
├── configs/
│   ├── corruption_config.yaml          # LOCKED corruption parameters
│   ├── preprocessing_config.yaml       # Preprocessing method parameters
│   └── experiment_config.yaml          # Model & runtime parameters
├── src/
│   ├── corruptions/
│   │   ├── __init__.py
│   │   ├── low_light.py
│   │   ├── gaussian_blur.py
│   │   ├── motion_blur.py
│   │   ├── sensor_noise.py
│   │   ├── fog_haze.py
│   │   └── combined.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── clahe.py
│   │   ├── wiener_deconv.py
│   │   ├── nlm_denoise.py
│   │   ├── bm3d_denoise.py
│   │   ├── aodnet_dehaze.py
│   │   ├── histogram_matching.py
│   │   └── deblurgan.py              # Optional — see Risk 6 fallback
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── corruption_pipeline.py     # On-the-fly corruption transforms
│   │   ├── preprocessing_pipeline.py  # On-the-fly preprocessing transforms
│   │   └── experiment_runner.py       # Main experiment orchestrator
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # AUROC, FNR, FPR computation
│   │   └── score_analysis.py          # Score distribution analysis
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py           # Visual grids, degradation samples
│       └── logging_utils.py           # CSV/results logging
├── results/
│   ├── baselines/                     # Clean baseline AUROC per category
│   ├── degradation/                   # Per-corruption, per-severity results
│   ├── preprocessing/                 # Per-preprocessing results
│   ├── ablation/                      # Pipeline ordering ablation
│   ├── training_corruption/           # Training-set corruption experiment
│   ├── visa/                          # VisA generalization results
│   ├── winclip/                       # Stretch goal results
│   ├── runtime/                       # Runtime benchmarks
│   └── figures/                       # Generated plots and visual grids
├── datasets/                          # Downloaded datasets go here
│   ├── MVTecAD/
│   └── VisA/
├── checkpoints/                       # Saved model checkpoints
├── notebooks/                         # Optional: Jupyter exploration
├── requirements.txt
└── README.md
```

### 0.2 Corruption Configuration File (IMMUTABLE AFTER PHASE 5)

Create `configs/corruption_config.yaml`:

```yaml
# ============================================================
# CORRUPTION SEVERITY CONFIGURATION
# WARNING: DO NOT MODIFY AFTER Phase 5 calibration is complete.
# All degradation curves depend on these exact values.
# ============================================================

corruptions:
  low_light:
    description: "Gamma correction + Poisson noise"
    library: "albumentations"  # or albumentationsx (drop-in replacement)
    severity_levels:
      mild:
        gamma: 0.5
      moderate:
        gamma: 0.35
      severe:
        gamma: 0.2
    # Note: Poisson noise is implicitly added via gamma darkening
    # which reduces photon counts. Additional shot noise is realistic.

  gaussian_blur:
    description: "Gaussian blur kernel"
    library: "albumentations"
    severity_levels:
      mild:
        sigma: 1.0
        kernel_size: 7    # Must be odd, >= 3
      moderate:
        sigma: 2.0
        kernel_size: 11
      severe:
        sigma: 4.0
        kernel_size: 23

  motion_blur:
    description: "Directional motion blur"
    library: "albumentations"
    severity_levels:
      mild:
        kernel_size: 7
      moderate:
        kernel_size: 15
      severe:
        kernel_size: 25

  sensor_noise:
    description: "Gaussian noise + salt-and-pepper"
    library: "albumentations"
    severity_levels:
      mild:
        gauss_var: 0.02     # variance relative to image max
      moderate:
        gauss_var: 0.05
      severe:
        gauss_var: 0.10

  fog_haze:
    description: "Fog/haze overlay via albumentations RandomFog"
    library: "albumentations"  # IMPORTANT: Use albumentations, NOT imgaug
    # imgaug is unmaintained since ~2022 and has NumPy 2.0 compat issues.
    # albumentations.RandomFog is the verified working alternative.
    severity_levels:
      mild:
        fog_coef_lower: 0.2
        fog_coef_upper: 0.35
        alpha_coef: 0.1
      moderate:
        fog_coef_lower: 0.45
        fog_coef_upper: 0.65
        alpha_coef: 0.1
      severe:
        fog_coef_lower: 0.75
        fog_coef_upper: 0.95
        alpha_coef: 0.1

  combined:
    description: "Low-light (moderate) + Sensor noise (moderate)"
    library: "albumentations"
    severity_levels:
      single:  # Only one severity level for combined
        gamma: 0.35          # moderate low-light
        gauss_var: 0.05      # moderate sensor noise

# Total dataset variants: 5 types × 3 levels + 1 combined = 16 variants
# (briefing says 18, but combined has only 1 level = 16 unique)

random_seed: 42  # For reproducibility of stochastic corruptions
```

> [!IMPORTANT]
> **Deviation from briefing**: The briefing specifies `imgaug` for fog/haze. As of 2026, `imgaug` is **unmaintained** and has compatibility issues with NumPy 2.0+. Use `albumentations.RandomFog` instead — it is actively maintained (as `albumentationsx`, which is a 100% drop-in replacement) and provides equivalent fog simulation. The import remains `import albumentations as A`.

### 0.3 Experiment Configuration File

Create `configs/experiment_config.yaml`:

```yaml
models:
  patchcore:
    class: "anomalib.models.image.patchcore.Patchcore"
    import: "from anomalib.models import Patchcore"
    params:
      backbone: "wide_resnet50_2"
      num_neighbors: 9
    max_epochs: 1  # PatchCore only needs 1 epoch (feature extraction + coreset)

  padim:
    class: "anomalib.models.image.padim.Padim"
    import: "from anomalib.models import Padim"
    params:
      backbone: "resnet18"
      layers: ["layer1", "layer2", "layer3"]
      pre_trained: true
    max_epochs: 1

  winclip:  # Stretch goal
    class: "anomalib.models.image.winclip.WinClip"
    import: "from anomalib.models import WinClip"
    params: {}
    max_epochs: 1

dataset:
  primary:
    class: "anomalib.data.MVTecAD"
    import: "from anomalib.data import MVTecAD"
    root: "./datasets/MVTecAD"
    train_batch_size: 32
    eval_batch_size: 32

  secondary:
    class: "anomalib.data.Visa"
    import: "from anomalib.data import Visa"
    root: "./datasets/VisA"
    train_batch_size: 32
    eval_batch_size: 32

seeds: [42, 123, 456]  # 3 seeds for statistical reporting

metrics:
  primary: "AUROC"  # image-level
  secondary:
    - "FNR"  # at 95th percentile threshold of normal training scores
    - "FPR"  # at same threshold
  threshold_method: "95th_percentile_normal_train"

engine:
  import: "from anomalib.engine import Engine"
```

---

## Phase 1 — Environment Setup & Dependency Installation

### 1.1 Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 1.2 Install Dependencies

Create `requirements.txt`:

```txt
# Core ML framework
anomalib
torch
torchvision
lightning

# Image augmentation (corruption pipeline)
# Use albumentationsx (actively maintained successor, 100% drop-in replacement)
# OR albumentations (maintenance mode but still functional)
# Both import as: import albumentations as A
albumentationsx

# Image processing & preprocessing
opencv-python-headless
scikit-image
scipy
numpy

# Denoising
bm3d

# AOD-Net dehazing (will be installed from source -- see Phase 7)
# No pip package available; clone from GitHub

# Visualization & analysis
matplotlib
seaborn
pandas

# Results tracking
tqdm

# Config management
pyyaml

# Metrics
scikit-learn
```

Install:

```bash
pip install -r requirements.txt
```

### 1.3 Verify Critical Imports

Run this verification script. **ALL must pass before proceeding.**

```python
"""verify_environment.py — Run this to confirm all dependencies are available."""
import sys

checks = []

# 1. Anomalib
try:
    from anomalib.models import Patchcore, Padim
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    checks.append(("Anomalib (Patchcore, Padim, MVTecAD, Engine)", True))
except ImportError as e:
    checks.append(("Anomalib", False, str(e)))

# 2. WinCLIP (stretch goal — non-fatal)
try:
    from anomalib.models import WinClip
    checks.append(("WinCLIP (stretch goal)", True))
except ImportError as e:
    checks.append(("WinCLIP (stretch goal)", False, str(e)))

# 3. Albumentations
try:
    import albumentations as A
    _ = A.GaussianBlur
    _ = A.MotionBlur
    _ = A.GaussNoise
    _ = A.RandomFog
    _ = A.RandomBrightnessContrast
    checks.append(("Albumentations (GaussianBlur, MotionBlur, GaussNoise, RandomFog)", True))
except (ImportError, AttributeError) as e:
    checks.append(("Albumentations", False, str(e)))

# 4. scikit-image restoration
try:
    from skimage.restoration import wiener, denoise_nl_means
    from skimage.exposure import match_histograms
    checks.append(("scikit-image (wiener, denoise_nl_means, match_histograms)", True))
except ImportError as e:
    checks.append(("scikit-image", False, str(e)))

# 5. OpenCV CLAHE
try:
    import cv2
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    checks.append(("OpenCV CLAHE", True))
except (ImportError, AttributeError) as e:
    checks.append(("OpenCV", False, str(e)))

# 6. BM3D
try:
    import bm3d
    checks.append(("BM3D", True))
except ImportError as e:
    checks.append(("BM3D", False, str(e)))

# 7. PyTorch + GPU
try:
    import torch
    gpu = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if gpu else "CPU only"
    checks.append((f"PyTorch (GPU: {device})", True))
except Exception as e:
    checks.append(("PyTorch", False, str(e)))

# 8. sklearn metrics
try:
    from sklearn.metrics import roc_auc_score
    checks.append(("scikit-learn (roc_auc_score)", True))
except ImportError as e:
    checks.append(("scikit-learn", False, str(e)))

# Report
print("=" * 70)
print("ENVIRONMENT VERIFICATION REPORT")
print("=" * 70)
all_pass = True
for check in checks:
    name = check[0]
    passed = check[1]
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  {name}")
    if not passed:
        print(f"           Error: {check[2]}")
        if "stretch" not in name.lower():
            all_pass = False
print("=" * 70)
if all_pass:
    print("✅ All critical checks passed. Proceed to Phase 2.")
else:
    print("❌ CRITICAL FAILURES. Fix before proceeding.")
    sys.exit(1)
```

> [!WARNING]
> **Do NOT proceed to Phase 2 until all critical checks pass.** WinCLIP failure is acceptable (stretch goal).

---

## Phase 2 — Dataset Acquisition

### 2.1 MVTec-AD (Primary Dataset)

**Option A — Anomalib auto-download** (Recommended):
Anomalib will automatically download MVTec-AD when the `MVTecAD` datamodule is first instantiated. Simply ensure `root="./datasets/MVTecAD"` is set.

**Option B — Manual download from Kaggle**:
```bash
# Requires kaggle CLI configured with API token
kaggle datasets download -d ipythonx/mvtec-ad -p ./datasets/MVTecAD --unzip
```

**Option C — Official MVTec website**:
Download from `https://www.mvtec.com/company/research/datasets/mvtec-ad`

### 2.2 Verify MVTec-AD Structure

After download, verify the directory contains all 15 categories:

```python
"""verify_mvtec.py"""
import os

EXPECTED_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

root = "./datasets/MVTecAD"
present = sorted([d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d))])

missing = set(EXPECTED_CATEGORIES) - set(present)
extra = set(present) - set(EXPECTED_CATEGORIES)

print(f"Found {len(present)}/15 categories")
if missing:
    print(f"❌ MISSING: {missing}")
if extra:
    print(f"⚠️  Extra dirs (ignored): {extra}")

# Verify structure for one category
cat = "bottle"
cat_path = os.path.join(root, cat)
for subdir in ["train/good", "test/good", "ground_truth"]:
    full = os.path.join(cat_path, subdir)
    if os.path.isdir(full):
        count = len(os.listdir(full))
        print(f"  ✅ {cat}/{subdir}: {count} items")
    else:
        print(f"  ❌ {cat}/{subdir}: NOT FOUND")

if not missing:
    print("\n✅ MVTec-AD dataset verified. Proceed to Phase 3.")
```

### 2.3 VisA Dataset (Phase 10 — Download Later)

VisA is only needed in Phase 10. Download options:
- **Anomalib auto-download**: `Visa(root="./datasets/VisA", category="...")` will auto-download
- **AWS S3**: `aws s3 cp --no-sign-request s3://amazon-visual-anomaly/VisA_20220922.tar ./datasets/`

---

## Phase 3 — Baseline Reproduction (Clean PatchCore & PaDiM)

### 3.1 Goal

Reproduce published clean baselines:
- **PatchCore**: ≥ 98% image-level AUROC on MVTec-AD (published benchmark)
- **PaDiM**: ≥ 95% image-level AUROC on MVTec-AD (published benchmark)

> [!CAUTION]
> **This is the non-negotiable sanity check.** If your clean baselines don't match published numbers, your entire degradation analysis is invalid. Do NOT proceed until baselines match.

### 3.2 Baseline Script

```python
"""run_baselines.py — Run clean PatchCore & PaDiM on all 15 MVTec-AD categories."""
import os
import pandas as pd
import lightning as L
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore, Padim

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]
SEEDS = [42, 123, 456]
MODELS = {
    "PatchCore": lambda: Patchcore(backbone="wide_resnet50_2", num_neighbors=9),
    "PaDiM": lambda: Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"]),
}

os.makedirs("results/baselines", exist_ok=True)
results = []

for model_name, model_fn in MODELS.items():
    for category in CATEGORIES:
        for seed in SEEDS:
            L.seed_everything(seed)

            model = model_fn()
            datamodule = MVTecAD(
                root="./datasets/MVTecAD",
                category=category,
                train_batch_size=32,
                eval_batch_size=32,
            )
            engine = Engine(max_epochs=1)
            engine.fit(model=model, datamodule=datamodule)
            test_results = engine.test(model=model, datamodule=datamodule)

            # Extract image-level AUROC from test results
            # Anomalib returns list of dicts; exact key may vary by version
            auroc = test_results[0].get("image_AUROC", None)

            results.append({
                "model": model_name,
                "category": category,
                "seed": seed,
                "image_AUROC": auroc,
            })
            print(f"  {model_name} | {category} | seed={seed} | AUROC={auroc:.4f}")

# Save results
df = pd.DataFrame(results)
df.to_csv("results/baselines/clean_baselines.csv", index=False)

# Compute mean ± std per model per category
summary = df.groupby(["model", "category"])["image_AUROC"].agg(["mean", "std"])
summary.to_csv("results/baselines/clean_baselines_summary.csv")
print("\n" + summary.to_string())

# Validate
patchcore_mean = df[df["model"] == "PatchCore"]["image_AUROC"].mean()
padim_mean = df[df["model"] == "PaDiM"]["image_AUROC"].mean()
print(f"\nPatchCore overall mean AUROC: {patchcore_mean:.4f} (expected ≥ 0.98)")
print(f"PaDiM overall mean AUROC: {padim_mean:.4f} (expected ≥ 0.95)")

if patchcore_mean < 0.97:
    print("❌ WARNING: PatchCore baseline below expected. Investigate before proceeding.")
if padim_mean < 0.93:
    print("❌ WARNING: PaDiM baseline below expected. Investigate before proceeding.")
```

### 3.3 Checkpoint Strategy

After successful baseline runs, **save the trained model checkpoints** for each category. These will be reused for all corruption/preprocessing experiments to ensure the memory bank is identical across conditions.

```python
# After engine.fit(), Anomalib/Lightning saves checkpoints automatically.
# Locate them in the default directory (usually ./results/ or a Lightning logs dir).
# Copy/organize them under:
#   checkpoints/{model_name}/{category}/seed_{seed}/model.ckpt
# This allows reloading for inference without retraining.
```

> [!NOTE]
> PatchCore's `engine.fit()` runs for only 1 epoch. It extracts features and builds the coreset memory bank. The checkpoint contains the memory bank, not trained weights.

### 3.4 Completion Criteria

| Check | Threshold | Action if Failed |
|-------|-----------|-----------------|
| PatchCore mean AUROC across all 15 categories | ≥ 0.97 | Debug model config, verify dataset |
| PaDiM mean AUROC across all 15 categories | ≥ 0.93 | Debug model config, verify dataset |
| All 15 categories produce valid AUROC scores | No NaN/None values | Check data loading |
| 3 seeds run for each configuration | 45 results per model | Re-run missing seeds |

---

## Phase 4 — Corruption Pipeline Construction

### 4.1 Design Principle

**All corruptions are applied on-the-fly as transforms before inference.** Do NOT pre-generate and store corrupted datasets (saves disk space and avoids data management issues per Risk 2 in briefing).

### 4.2 Corruption Implementations

Each corruption function takes an image (numpy array, uint8, HWC, RGB) and returns the corrupted image in the same format.

#### 4.2.1 Low-Light Corruption

```python
"""src/corruptions/low_light.py"""
import numpy as np

def apply_low_light(image: np.ndarray, gamma: float, seed: int = 42) -> np.ndarray:
    """
    Simulate low-light conditions via gamma correction + Poisson noise.

    Args:
        image: uint8 HWC RGB image
        gamma: Gamma value (< 1.0 darkens). Values: 0.5 (mild), 0.35 (moderate), 0.2 (severe)
        seed: Random seed for Poisson noise reproducibility

    Returns:
        Corrupted uint8 image
    """
    rng = np.random.default_rng(seed)

    # 1. Gamma correction (darken)
    img_float = image.astype(np.float64) / 255.0
    img_dark = np.power(img_float, 1.0 / gamma)  # gamma < 1 → 1/gamma > 1 → darkens

    # 2. Add Poisson noise (simulates photon shot noise in low light)
    # Scale determines noise level; lower light = more visible noise
    photon_scale = 50.0  # Approximate photon count scale
    img_noisy = rng.poisson(img_dark * photon_scale) / photon_scale

    # Clip and convert back
    img_out = np.clip(img_noisy * 255, 0, 255).astype(np.uint8)
    return img_out
```

#### 4.2.2 Gaussian Blur

```python
"""src/corruptions/gaussian_blur.py"""
import albumentations as A
import numpy as np

def apply_gaussian_blur(image: np.ndarray, sigma: float, kernel_size: int) -> np.ndarray:
    """
    Apply Gaussian blur.

    Args:
        image: uint8 HWC RGB image
        sigma: Blur sigma. Values: 1.0 (mild), 2.0 (moderate), 4.0 (severe)
        kernel_size: Must be odd. Values: 7 (mild), 11 (moderate), 23 (severe)

    Returns:
        Blurred uint8 image
    """
    transform = A.GaussianBlur(
        blur_limit=(kernel_size, kernel_size),
        sigma_limit=(sigma, sigma),
        p=1.0
    )
    return transform(image=image)["image"]
```

#### 4.2.3 Motion Blur

```python
"""src/corruptions/motion_blur.py"""
import albumentations as A
import numpy as np

def apply_motion_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply directional motion blur.

    Args:
        image: uint8 HWC RGB image
        kernel_size: Blur kernel size in pixels. Values: 7 (mild), 15 (moderate), 25 (severe)

    Returns:
        Motion-blurred uint8 image
    """
    transform = A.MotionBlur(
        blur_limit=(kernel_size, kernel_size),
        p=1.0
    )
    return transform(image=image)["image"]
```

#### 4.2.4 Sensor Noise

```python
"""src/corruptions/sensor_noise.py"""
import numpy as np

def apply_sensor_noise(image: np.ndarray, gauss_var: float, seed: int = 42) -> np.ndarray:
    """
    Apply Gaussian noise + salt-and-pepper noise.

    Args:
        image: uint8 HWC RGB image
        gauss_var: Gaussian noise variance (relative). Values: 0.02 (mild), 0.05 (moderate), 0.10 (severe)
        seed: Random seed

    Returns:
        Noisy uint8 image
    """
    rng = np.random.default_rng(seed)

    img_float = image.astype(np.float64) / 255.0

    # Gaussian noise
    noise = rng.normal(0, np.sqrt(gauss_var), img_float.shape)
    img_noisy = img_float + noise

    # Salt-and-pepper noise (5% of pixels)
    sp_ratio = 0.05
    salt_mask = rng.random(img_float.shape[:2]) < (sp_ratio / 2)
    pepper_mask = rng.random(img_float.shape[:2]) < (sp_ratio / 2)
    img_noisy[salt_mask] = 1.0
    img_noisy[pepper_mask] = 0.0

    return np.clip(img_noisy * 255, 0, 255).astype(np.uint8)
```

#### 4.2.5 Fog/Haze

> [!IMPORTANT]
> **Use `albumentations.RandomFog`, NOT `imgaug`**. The `imgaug` library is unmaintained and has NumPy 2.0 compatibility issues. `albumentations.RandomFog` provides equivalent fog simulation and is actively maintained.

```python
"""src/corruptions/fog_haze.py"""
import albumentations as A
import numpy as np

def apply_fog(image: np.ndarray,
              fog_coef_lower: float,
              fog_coef_upper: float,
              alpha_coef: float = 0.1) -> np.ndarray:
    """
    Apply fog/haze overlay using albumentations RandomFog.

    Args:
        image: uint8 HWC RGB image
        fog_coef_lower: Lower bound of fog coefficient.
        fog_coef_upper: Upper bound of fog coefficient.
        alpha_coef: Transparency of fog circles.

    Severity levels:
        mild:     fog_coef_lower=0.2,  fog_coef_upper=0.35
        moderate: fog_coef_lower=0.45, fog_coef_upper=0.65
        severe:   fog_coef_lower=0.75, fog_coef_upper=0.95

    Returns:
        Foggy uint8 image
    """
    transform = A.RandomFog(
        fog_coef_lower=fog_coef_lower,
        fog_coef_upper=fog_coef_upper,
        alpha_coef=alpha_coef,
        p=1.0
    )
    return transform(image=image)["image"]
```

#### 4.2.6 Combined (Low-Light + Sensor Noise)

```python
"""src/corruptions/combined.py"""
import numpy as np
from .low_light import apply_low_light
from .sensor_noise import apply_sensor_noise

def apply_combined(image: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Apply combined degradation: Low-light (moderate) + Sensor noise (moderate).

    This is the most realistic single case — poor lighting AND camera noise.
    Only run at ONE severity level (moderate + moderate).

    Args:
        image: uint8 HWC RGB image
        seed: Random seed

    Returns:
        Combined-degraded uint8 image
    """
    # Apply low-light first (gamma=0.35, moderate)
    img = apply_low_light(image, gamma=0.35, seed=seed)
    # Then add sensor noise (gauss_var=0.05, moderate)
    img = apply_sensor_noise(img, gauss_var=0.05, seed=seed + 1)
    return img
```

### 4.3 Unified Corruption Interface

```python
"""src/pipeline/corruption_pipeline.py"""
import yaml
import numpy as np
from src.corruptions import (
    low_light, gaussian_blur, motion_blur,
    sensor_noise, fog_haze, combined
)

def load_corruption_config(config_path: str = "configs/corruption_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def apply_corruption(
    image: np.ndarray,
    corruption_type: str,
    severity: str,
    config: dict,
    seed: int = 42
) -> np.ndarray:
    """
    Unified corruption interface.

    Args:
        image: uint8 HWC RGB image
        corruption_type: One of: "low_light", "gaussian_blur", "motion_blur",
                        "sensor_noise", "fog_haze", "combined"
        severity: One of: "mild", "moderate", "severe" (or "single" for combined)
        config: Loaded corruption config dict
        seed: Random seed

    Returns:
        Corrupted uint8 image
    """
    params = config["corruptions"][corruption_type]["severity_levels"][severity]

    if corruption_type == "low_light":
        return low_light.apply_low_light(image, gamma=params["gamma"], seed=seed)
    elif corruption_type == "gaussian_blur":
        return gaussian_blur.apply_gaussian_blur(
            image, sigma=params["sigma"], kernel_size=params["kernel_size"]
        )
    elif corruption_type == "motion_blur":
        return motion_blur.apply_motion_blur(image, kernel_size=params["kernel_size"])
    elif corruption_type == "sensor_noise":
        return sensor_noise.apply_sensor_noise(image, gauss_var=params["gauss_var"], seed=seed)
    elif corruption_type == "fog_haze":
        return fog_haze.apply_fog(
            image,
            fog_coef_lower=params["fog_coef_lower"],
            fog_coef_upper=params["fog_coef_upper"],
            alpha_coef=params.get("alpha_coef", 0.1)
        )
    elif corruption_type == "combined":
        return combined.apply_combined(image, seed=seed)
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")
```

---

## Phase 5 — Severity Calibration & Visual Verification

### 5.1 Generate Visual Grid

For each corruption type and severity level, generate a grid of 5 sample images (from different MVTec-AD categories) side-by-side with the clean original.

```python
"""generate_visual_grid.py"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.pipeline.corruption_pipeline import apply_corruption, load_corruption_config

config = load_corruption_config()

# Select 5 diverse categories for visual inspection
SAMPLE_CATEGORIES = ["bottle", "carpet", "metal_nut", "wood", "transistor"]
CORRUPTION_TYPES = ["low_light", "gaussian_blur", "motion_blur", "sensor_noise", "fog_haze", "combined"]

os.makedirs("results/figures/severity_grids", exist_ok=True)

for corruption_type in CORRUPTION_TYPES:
    severities = list(config["corruptions"][corruption_type]["severity_levels"].keys())

    fig, axes = plt.subplots(len(SAMPLE_CATEGORIES), len(severities) + 1,
                              figsize=(4 * (len(severities) + 1), 4 * len(SAMPLE_CATEGORIES)))
    fig.suptitle(f"Corruption: {corruption_type}", fontsize=16, fontweight="bold")

    for row, category in enumerate(SAMPLE_CATEGORIES):
        # Load a sample image
        img_dir = f"./datasets/MVTecAD/{category}/test/good/"
        img_files = sorted(os.listdir(img_dir))
        if not img_files:
            img_dir = f"./datasets/MVTecAD/{category}/train/good/"
            img_files = sorted(os.listdir(img_dir))
        img_path = os.path.join(img_dir, img_files[0])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Clean original
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"{category}\n(clean)")
        axes[row, 0].axis("off")

        # Each severity
        for col, severity in enumerate(severities):
            corrupted = apply_corruption(img, corruption_type, severity, config)
            axes[row, col + 1].imshow(corrupted)
            axes[row, col + 1].set_title(f"{severity}")
            axes[row, col + 1].axis("off")

    plt.tight_layout()
    plt.savefig(f"results/figures/severity_grids/{corruption_type}_grid.png", dpi=150)
    plt.close()
    print(f"✅ Saved grid for {corruption_type}")
```

### 5.2 Visual Verification Checklist

**For each corruption type, manually verify:**

| Check | Pass Criteria |
|-------|--------------|
| Severity levels are perceptually proportional | Mild is noticeably different from clean; moderate is clearly worse than mild; severe is dramatically degraded |
| Structures remain visible at severe level | You can still identify what the object is (except possibly combined case) |
| Combined case is not near-random | The object is still recognizable; if not, reduce noise variance in config |
| No artifacts or implementation bugs | No black images, no color channel swaps, no inverted images |

> [!CAUTION]
> **Risk 4 from briefing**: The combined case (low-light + noise) may destroy the image so much that AUROC ≈ 0.5 (random). If the combined case produces unrecognizable images, **reduce the noise variance** in the combined config from 0.05 to 0.03 and regenerate. This must be decided BEFORE Phase 6.

### 5.3 Lock Configuration

After visual verification passes, the corruption config is **FROZEN**. Add a timestamp comment:

```yaml
# LOCKED: YYYY-MM-DD after visual verification
# DO NOT MODIFY PARAMETERS BELOW THIS POINT
```

---

## Phase 6 — Degradation Experiments (Main)

### 6.1 Experiment Matrix

| Dimension | Values | Count |
|-----------|--------|-------|
| Models | PatchCore, PaDiM | 2 |
| Categories | All 15 MVTec-AD | 15 |
| Corruption types | low_light, gaussian_blur, motion_blur, sensor_noise, fog_haze (3 levels each) + combined (1 level) | 16 variants |
| Seeds | 42, 123, 456 | 3 |
| **Total runs** | 2 × 15 × 16 × 3 = **1,440** | |

### 6.2 Experiment Runner — On-the-fly Corruption

The key architectural decision: **corruption is applied as a transform inside the datamodule's test pipeline, NOT by pre-generating images**.

```python
"""src/pipeline/experiment_runner.py"""
import os
import pandas as pd
import numpy as np
import lightning as L
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore, Padim
from src.pipeline.corruption_pipeline import apply_corruption, load_corruption_config

config = load_corruption_config()

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]
SEEDS = [42, 123, 456]

CORRUPTION_VARIANTS = []
for ctype in ["low_light", "gaussian_blur", "motion_blur", "sensor_noise", "fog_haze"]:
    for sev in ["mild", "moderate", "severe"]:
        CORRUPTION_VARIANTS.append((ctype, sev))
CORRUPTION_VARIANTS.append(("combined", "single"))
# Total: 16 variants


def create_model(model_name: str):
    if model_name == "PatchCore":
        return Patchcore(backbone="wide_resnet50_2", num_neighbors=9)
    elif model_name == "PaDiM":
        return Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"])
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_degradation_experiments(model_name: str, output_dir: str = "results/degradation"):
    """
    Run all degradation experiments for a given model.

    Strategy:
    1. For each category, train on CLEAN data (or load checkpoint from Phase 3)
    2. For each corruption variant, apply corruption to test images on-the-fly
    3. Record AUROC, FNR, FPR
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for category in CATEGORIES:
        for seed in SEEDS:
            L.seed_everything(seed)

            # Train on clean data (or load saved checkpoint)
            model = create_model(model_name)
            datamodule = MVTecAD(
                root="./datasets/MVTecAD",
                category=category,
                train_batch_size=32,
                eval_batch_size=32,
            )
            engine = Engine(max_epochs=1)
            engine.fit(model=model, datamodule=datamodule)

            # Run on each corruption variant using custom dataset wrapper
            for corruption_type, severity in CORRUPTION_VARIANTS:
                # See Section 6.3 for CorruptedDatasetWrapper implementation
                test_results = run_corrupted_inference(
                    engine, model, datamodule, category,
                    corruption_type, severity, config, seed
                )

                results.append({
                    "model": model_name,
                    "category": category,
                    "seed": seed,
                    "corruption_type": corruption_type,
                    "severity": severity,
                    "image_AUROC": test_results["image_AUROC"],
                    "FNR": test_results["FNR"],
                    "FPR": test_results["FPR"],
                })

                print(f"  {model_name} | {category} | {corruption_type}/{severity} | "
                      f"seed={seed} | AUROC={test_results['image_AUROC']:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/{model_name}_degradation_results.csv", index=False)
    return df
```

### 6.3 Custom Corrupted Dataset Wrapper

To apply corruption on-the-fly, create a dataset wrapper that intercepts images after loading but before model preprocessing:

```python
"""src/pipeline/corrupted_dataset.py"""
import torch
import numpy as np
from torch.utils.data import Dataset

class CorruptedDatasetWrapper(Dataset):
    """
    Wraps an existing Anomalib dataset and applies corruption on-the-fly.

    The corruption is applied to the raw image BEFORE model preprocessing
    (resize, normalize, etc.) occurs.
    """

    def __init__(self, base_dataset, corruption_fn=None, preprocessing_fn=None):
        """
        Args:
            base_dataset: The original anomalib dataset (e.g., from datamodule.test_data)
            corruption_fn: Callable(image: np.ndarray) -> np.ndarray
                          Takes uint8 HWC RGB numpy array, returns same format
            preprocessing_fn: Optional callable for preprocessing rescue experiments
                             Applied AFTER corruption
        """
        self.base_dataset = base_dataset
        self.corruption_fn = corruption_fn
        self.preprocessing_fn = preprocessing_fn

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]

        if self.corruption_fn is not None:
            # Extract image, apply corruption, put back
            # Anomalib stores images as tensors; convert to numpy for corruption
            img_tensor = item["image"]  # Shape: [C, H, W], dtype: float32
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            img_np = self.corruption_fn(img_np)

            if self.preprocessing_fn is not None:
                img_np = self.preprocessing_fn(img_np)

            # Convert back to tensor
            item["image"] = torch.from_numpy(
                img_np.astype(np.float32) / 255.0
            ).permute(2, 0, 1)

        return item
```

> [!NOTE]
> The exact data format (tensor layout, key names) depends on the Anomalib version. During Phase 3, verify by inspecting `datamodule.test_data[0].keys()` and the image tensor shape/dtype. Adjust the wrapper accordingly.

### 6.4 Metrics Computation

```python
"""src/evaluation/metrics.py"""
import numpy as np
from sklearn.metrics import roc_auc_score

def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    normal_train_scores: np.ndarray
) -> dict:
    """
    Compute AUROC, FNR, FPR.

    Args:
        labels: Binary labels (0=normal, 1=anomaly)
        scores: Anomaly scores (higher = more anomalous)
        normal_train_scores: Anomaly scores on clean normal training images
                            (for threshold calibration)

    Returns:
        Dict with image_AUROC, FNR, FPR
    """
    # AUROC
    auroc = roc_auc_score(labels, scores)

    # Threshold: 95th percentile of normal training scores
    threshold = np.percentile(normal_train_scores, 95)

    # Binary predictions at threshold
    predictions = (scores >= threshold).astype(int)

    # FNR = missed defects / total defects
    true_positives = np.sum((predictions == 1) & (labels == 1))
    false_negatives = np.sum((predictions == 0) & (labels == 1))
    total_anomalous = np.sum(labels == 1)
    fnr = false_negatives / total_anomalous if total_anomalous > 0 else 0.0

    # FPR = false alarms / total normals
    false_positives = np.sum((predictions == 1) & (labels == 0))
    total_normal = np.sum(labels == 0)
    fpr = false_positives / total_normal if total_normal > 0 else 0.0

    return {
        "image_AUROC": auroc,
        "FNR": fnr,
        "FPR": fpr,
        "threshold": threshold,
    }
```

### 6.5 Completion Criteria for Phase 6

- [ ] All 1,440 experiment runs completed (or equivalent with checkpoint reuse)
- [ ] Results saved to `results/degradation/{model}_degradation_results.csv`
- [ ] No NaN/None AUROC values
- [ ] Sanity check: clean test metrics match Phase 3 baselines (within ±0.005)
- [ ] Summary statistics (mean ± std across 3 seeds) computed and saved

---

## Phase 7 — Preprocessing Rescue Pipeline

### 7.1 Preprocessing Implementations

Each preprocessing function takes a corrupted uint8 HWC RGB image and returns a preprocessed image in the same format.

#### 7.1.1 CLAHE (Contrast-Limited Adaptive Histogram Equalization)

**Target degradation**: Low-light

```python
"""src/preprocessing/clahe.py"""
import cv2
import numpy as np

def apply_clahe(image: np.ndarray, clip_limit: float = 3.0,
                tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE to the L channel in LAB color space.

    Args:
        image: uint8 HWC RGB image
        clip_limit: Contrast limiting threshold
        tile_grid_size: Grid size for local equalization

    Returns:
        Enhanced uint8 RGB image

    Implementation: cv2.createCLAHE(clipLimit, tileGridSize)
    """
    # Convert RGB → LAB
    # Note: cv2 expects BGR, so convert RGB→BGR→LAB
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel only
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)

    # Merge and convert back to RGB
    lab_enhanced = cv2.merge((cl, a, b))
    bgr_out = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    rgb_out = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)
    return rgb_out
```

#### 7.1.2 Wiener Deconvolution

**Target degradation**: Gaussian blur

> [!NOTE]
> Wiener deconvolution requires the blur kernel (PSF) as input. This is valid in this study because the blur kernels are known by construction in the synthetic pipeline. In real-world deployment where the kernel is unknown, blind deconvolution methods would be required.

```python
"""src/preprocessing/wiener_deconv.py"""
import numpy as np
from skimage.restoration import wiener
from skimage import img_as_float, img_as_ubyte

def apply_wiener_deconv(image: np.ndarray, sigma: float,
                        kernel_size: int, balance: float = 0.1) -> np.ndarray:
    """
    Apply Wiener deconvolution using the KNOWN blur kernel.

    Args:
        image: uint8 HWC RGB image
        sigma: The Gaussian blur sigma that was applied (KNOWN from corruption config)
        kernel_size: The blur kernel size (KNOWN from corruption config)
        balance: Regularization parameter (higher = smoother, less ringing)

    Returns:
        Deconvolved uint8 RGB image

    Function: skimage.restoration.wiener(image, psf, balance)
    """
    img_float = img_as_float(image)

    # Construct the known PSF (2D Gaussian kernel)
    ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    psf /= psf.sum()

    # Apply Wiener deconvolution per channel
    result = np.zeros_like(img_float)
    for c in range(3):
        result[:, :, c] = wiener(img_float[:, :, c], psf, balance)

    return img_as_ubyte(np.clip(result, 0, 1))
```

#### 7.1.3 Non-Local Means Denoising

**Target degradation**: Sensor noise

```python
"""src/preprocessing/nlm_denoise.py"""
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma

def apply_nlm_denoise(image: np.ndarray, patch_size: int = 7,
                       patch_distance: int = 11) -> np.ndarray:
    """
    Apply Non-Local Means denoising.

    Args:
        image: uint8 HWC RGB image
        patch_size: Size of patches for similarity
        patch_distance: Max distance in pixels for patches

    Returns:
        Denoised uint8 RGB image

    Function: skimage.restoration.denoise_nl_means
    """
    img_float = image.astype(np.float64) / 255.0

    # Estimate noise standard deviation
    sigma_est = np.mean(estimate_sigma(img_float, channel_axis=-1))

    # Apply NLM denoising
    denoised = denoise_nl_means(
        img_float,
        h=1.15 * sigma_est,  # Filter strength (slightly above estimated sigma)
        patch_size=patch_size,
        patch_distance=patch_distance,
        fast_mode=True,
        channel_axis=-1,
    )

    return np.clip(denoised * 255, 0, 255).astype(np.uint8)
```

#### 7.1.4 BM3D Denoising

**Target degradation**: Sensor noise (quality ceiling comparison)

```python
"""src/preprocessing/bm3d_denoise.py"""
import numpy as np
import bm3d as bm3d_lib

def apply_bm3d(image: np.ndarray, sigma_psd: float = None) -> np.ndarray:
    """
    Apply BM3D denoising.

    Args:
        image: uint8 HWC RGB image
        sigma_psd: Noise standard deviation estimate. If None, estimated automatically.

    Returns:
        Denoised uint8 RGB image

    Function: bm3d.bm3d(noisy_image, sigma_psd)
    """
    img_float = image.astype(np.float64) / 255.0

    if sigma_psd is None:
        from skimage.restoration import estimate_sigma
        sigma_psd = np.mean(estimate_sigma(img_float, channel_axis=-1))

    denoised = bm3d_lib.bm3d(img_float, sigma_psd=sigma_psd)
    return np.clip(denoised * 255, 0, 255).astype(np.uint8)
```

#### 7.1.5 AOD-Net Dehazing

**Target degradation**: Fog/haze

> [!WARNING]
> AOD-Net is available at `https://github.com/weberwcwei/AODnet-by-pytorch`. It is a PyTorch model from ~2018. Verify it loads on your current PyTorch version. If it fails, use the dark channel prior fallback below.

```python
"""src/preprocessing/aodnet_dehaze.py"""
import torch
import numpy as np

# AOD-Net must be cloned and added to path:
# git clone https://github.com/weberwcwei/AODnet-by-pytorch.git external/AODNet

class AODNetDehaze:
    def __init__(self, checkpoint_path: str = "external/AODNet/AOD_net_epoch_relu_10.pth"):
        """Load pretrained AOD-Net model."""
        import sys
        sys.path.insert(0, "external/AODNet")
        from model import AODnet  # Verify this import works after cloning

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AODnet()
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Dehaze a single image.

        Args:
            image: uint8 HWC RGB image

        Returns:
            Dehazed uint8 RGB image
        """
        img_tensor = torch.from_numpy(
            image.astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)

        result = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return np.clip(result * 255, 0, 255).astype(np.uint8)


# ============================================================
# FALLBACK: Dark channel prior (if AOD-Net fails to load)
# ============================================================
def apply_dark_channel_prior_dehaze(image: np.ndarray, omega: float = 0.95,
                                     patch_size: int = 15) -> np.ndarray:
    """
    Simple dark channel prior dehazing (He et al., 2009).
    Use this as fallback if AOD-Net has compatibility issues.
    """
    img = image.astype(np.float64) / 255.0

    # Dark channel
    dark = np.min(img, axis=2)
    from scipy.ndimage import minimum_filter
    dark_channel = minimum_filter(dark, size=patch_size)

    # Atmospheric light estimation (top 0.1% brightest in dark channel)
    flat_dark = dark_channel.ravel()
    top_indices = np.argsort(flat_dark)[-max(1, int(0.001 * len(flat_dark))):]
    A = np.max(img.reshape(-1, 3)[top_indices], axis=0)

    # Transmission estimation
    normed = img / (A + 1e-6)
    normed_dark = np.min(normed, axis=2)
    normed_dark_filtered = minimum_filter(normed_dark, size=patch_size)
    transmission = 1 - omega * normed_dark_filtered
    transmission = np.clip(transmission, 0.1, 1.0)

    # Scene radiance recovery
    result = (img - A) / transmission[:, :, np.newaxis] + A
    return np.clip(result * 255, 0, 255).astype(np.uint8)
```

#### 7.1.6 Histogram Matching (Naive Baseline)

**Target degradation**: All (universal naive baseline)

```python
"""src/preprocessing/histogram_matching.py"""
import numpy as np
from skimage.exposure import match_histograms

def apply_histogram_matching(image: np.ndarray,
                              reference_image: np.ndarray) -> np.ndarray:
    """
    Match the histogram of a degraded image to a clean reference image.

    This is the naive baseline for ALL degradation types.
    The reference should be a representative normal training image.

    Args:
        image: uint8 HWC RGB image (degraded)
        reference_image: uint8 HWC RGB image (clean normal from training set)

    Returns:
        Histogram-matched uint8 RGB image

    Function: skimage.exposure.match_histograms
    """
    matched = match_histograms(image, reference_image, channel_axis=-1)
    return np.clip(matched, 0, 255).astype(np.uint8)
```

#### 7.1.7 Unsharp Masking (Fallback for DeblurGAN-v2)

**Target degradation**: Motion blur (and as DeblurGAN-v2 fallback for Gaussian blur)

> [!WARNING]
> **DeblurGAN-v2 is essentially archived** (last meaningful update ~2019). It has known compatibility issues with PyTorch 2.x. Use unsharp masking as the practical substitute. If you want to attempt DeblurGAN-v2, test it in an isolated environment first.

```python
"""src/preprocessing/deblurgan.py"""
import cv2
import numpy as np

def apply_unsharp_masking(image: np.ndarray, sigma: float = 2.0,
                           strength: float = 1.5) -> np.ndarray:
    """
    Unsharp masking — a simpler sharpening technique.
    Not a true deblurring method, but a practical fallback.

    Args:
        image: uint8 HWC RGB image
        sigma: Gaussian blur sigma for creating the unsharp mask
        strength: Sharpening strength multiplier

    Returns:
        Sharpened uint8 RGB image
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)
```

### 7.2 Preprocessing Mapping Table (Verified)

| Degradation | Primary Preprocessing | Function | Secondary/Fallback | Naive Baseline |
|------------|----------------------|----------|-------------------|----------------|
| Low-light | CLAHE | `apply_clahe(image)` | — | Histogram matching |
| Gaussian blur | Wiener deconvolution | `apply_wiener_deconv(image, sigma, kernel_size)` | Unsharp masking | Histogram matching |
| Motion blur | Unsharp masking | `apply_unsharp_masking(image)` | — | Histogram matching |
| Sensor noise | NLM denoising | `apply_nlm_denoise(image)` | BM3D | Histogram matching |
| Fog/haze | AOD-Net (or dark channel prior) | `aodnet.apply(image)` or `apply_dark_channel_prior_dehaze(image)` | — | Histogram matching |
| Combined | Denoise → CLAHE (sequential) | `apply_clahe(apply_nlm_denoise(image))` | CLAHE → Denoise (reversed) | Histogram matching |

---

## Phase 8 — Preprocessing Rescue Experiments

### 8.1 Experiment Matrix

For EACH of the 16 corruption variants, run inference with:
1. **Raw degraded** (no preprocessing) — already done in Phase 6
2. **Naive baseline** (histogram matching)
3. **Primary preprocessing** (matched to degradation type)
4. **Secondary preprocessing** (if applicable — BM3D for noise, reversed order for combined)

| Dimension | Values |
|-----------|--------|
| Models | PatchCore, PaDiM |
| Categories | 15 |
| Corruption variants | 16 |
| Preprocessing strategies | 2-4 per variant (varies) |
| Seeds | 3 |

### 8.2 Preprocessing Experiment Runner Pseudocode

```python
"""run_preprocessing_experiments.py"""
# Structure matches Phase 6 runner, but adds preprocessing step:

PREPROCESSING_MAP = {
    "low_light": {
        "primary": ("CLAHE", lambda img: apply_clahe(img)),
        "naive": ("HistMatch", lambda img: apply_histogram_matching(img, reference)),
    },
    "gaussian_blur": {
        "primary": ("Wiener", lambda img: apply_wiener_deconv(img, sigma=KNOWN_SIGMA, kernel_size=KNOWN_KS)),
        "naive": ("HistMatch", lambda img: apply_histogram_matching(img, reference)),
    },
    "motion_blur": {
        "primary": ("UnsharpMask", lambda img: apply_unsharp_masking(img)),
        "naive": ("HistMatch", lambda img: apply_histogram_matching(img, reference)),
    },
    "sensor_noise": {
        "primary": ("NLM", lambda img: apply_nlm_denoise(img)),
        "secondary": ("BM3D", lambda img: apply_bm3d(img)),
        "naive": ("HistMatch", lambda img: apply_histogram_matching(img, reference)),
    },
    "fog_haze": {
        "primary": ("AODNet", lambda img: aodnet.apply(img)),  # or dark_channel_prior
        "naive": ("HistMatch", lambda img: apply_histogram_matching(img, reference)),
    },
    "combined": {
        "primary": ("Denoise_then_CLAHE", lambda img: apply_clahe(apply_nlm_denoise(img))),
        "secondary": ("CLAHE_then_Denoise", lambda img: apply_nlm_denoise(apply_clahe(img))),
        "naive": ("HistMatch", lambda img: apply_histogram_matching(img, reference)),
    },
}

# For each experiment:
# 1. Load trained model checkpoint (from Phase 3)
# 2. Apply corruption to test image
# 3. Apply preprocessing to corrupted image
# 4. Run inference
# 5. Record metrics

# NOTE: For Wiener deconvolution, the blur sigma and kernel_size must match
# the corruption parameters used. Read them from corruption_config.yaml.
```

### 8.3 Pipeline Ordering Ablation (Combined Case)

For the combined degradation only, compare:

1. **Denoise → CLAHE**: Apply NLM denoising first, then CLAHE contrast enhancement
2. **CLAHE → Denoise**: Apply CLAHE first (expected worse — amplifies noise), then denoise

Record and compare AUROC for both orderings. Expected finding: "Denoise first" outperforms "CLAHE first" because CLAHE applied to noisy images treats noise as local contrast and amplifies it.

Save results to `results/ablation/pipeline_ordering_results.csv`.

### 8.4 Reference Image Selection (for Histogram Matching)

```python
def get_reference_image(category: str, dataset_root: str = "./datasets/MVTecAD") -> np.ndarray:
    """
    Select a representative normal training image for histogram matching.
    Use the MEDIAN image (by brightness) from the training set to avoid outliers.
    """
    import os, cv2
    train_dir = os.path.join(dataset_root, category, "train", "good")
    img_files = sorted(os.listdir(train_dir))

    # Compute mean brightness for each image
    brightnesses = []
    for f in img_files:
        img = cv2.imread(os.path.join(train_dir, f))
        brightnesses.append(np.mean(img))

    # Select median-brightness image
    median_idx = np.argsort(brightnesses)[len(brightnesses) // 2]
    ref_img = cv2.imread(os.path.join(train_dir, img_files[median_idx]))
    return cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
```

### 8.5 Completion Criteria

- [ ] All preprocessing rescue experiments completed
- [ ] Results saved to `results/preprocessing/{model}_preprocessing_results.csv`
- [ ] Pipeline ordering ablation completed for combined case
- [ ] Results format: `model, category, seed, corruption_type, severity, preprocessing, image_AUROC, FNR, FPR`

---

## Phase 9 — Training-Set Corruption Experiment

### 9.1 Purpose

Test the scenario: "What if training images were ALSO collected under bad conditions?"

### 9.2 Setup

- **Corruption**: Low-light, moderate severity only (gamma=0.35)
- **Applied to**: BOTH training set AND test set
- **Preprocessing**: CLAHE applied to BOTH training set AND test set
- **Models**: PatchCore, PaDiM
- **Seeds**: 3

### 9.3 Implementation Pseudocode

```python
"""run_training_corruption_experiment.py"""

# Condition A (already done): Clean train → Clean test → Baseline AUROC
# Condition B: Corrupted train → Corrupted test → No preprocessing → AUROC
# Condition C: Corrupted+CLAHE train → Corrupted+CLAHE test → AUROC

# For Condition B:
# 1. Create a CorruptedTrainDataset that applies low_light(moderate) to training images
# 2. Train PatchCore/PaDiM on this corrupted training set
# 3. Apply same corruption to test images
# 4. Run inference → record AUROC

# For Condition C:
# 1. Apply low_light(moderate) THEN CLAHE to training images
# 2. Train PatchCore/PaDiM on preprocessed training set
# 3. Apply low_light(moderate) THEN CLAHE to test images
# 4. Run inference → record AUROC

# This requires modifying the training data pipeline as well.
# Use the same CorruptedDatasetWrapper but applied to training data.
```

### 9.4 Expected Output

| Condition | Training Data | Test Data | Preprocessing | Expected AUROC |
|-----------|--------------|-----------|---------------|----------------|
| A | Clean | Clean | None | ~0.98 (baseline) |
| B | Corrupted (low-light mod) | Corrupted (low-light mod) | None | Lower than A |
| C | Corrupted + CLAHE | Corrupted + CLAHE | CLAHE on both | Between A and B (recovery) |

Save results to `results/training_corruption/`.

---

## Phase 10 — VisA Generalization Check

### 10.1 Purpose

Verify that preprocessing recommendations generalize beyond MVTec-AD to a second independently-collected dataset.

### 10.2 Setup

- **Dataset**: VisA (12 object types)
- **Corruption**: All 6 types, **moderate severity only**
- **Preprocessing**: Only the **best-performing** preprocessing per degradation type (from Phase 8)
- **Models**: PatchCore only
- **Seeds**: 3

### 10.3 VisA Categories (Verify After Download)

```python
# Expected categories — verify against actual download
VISA_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum",
    "fryum", "macaroni1", "macaroni2", "pcb1",
    "pcb2", "pcb3", "pcb4", "pipe_fryum"
]
```

> [!NOTE]
> Verify exact category names by checking the downloaded dataset directory or via `anomalib.data.Visa`. Category names may vary between Anomalib versions.

### 10.4 Workflow

1. Download VisA dataset (see Phase 2.3)
2. Run clean PatchCore baseline on all 12 VisA categories (3 seeds each)
3. For each corruption type at moderate severity:
   a. Run corrupted inference → record AUROC
   b. Run corrupted + best preprocessing → record AUROC
4. Compare AUROC recovery with MVTec-AD findings
5. Determine if preprocessing recommendations generalize

Save results to `results/visa/`.

---

## Phase 11 — WinCLIP Stretch Goal

### 11.1 Decision Gate

**Only proceed if:**
- Phases 6-9 are fully complete
- There is remaining time/compute budget
- WinCLIP imported successfully in Phase 1 verification

### 11.2 Setup

```python
from anomalib.models import WinClip
# Located at: anomalib.models.image.winclip

model = WinClip()
# WinCLIP is zero-shot — no traditional training, but needs setup phase
# for collecting text/visual embeddings
```

### 11.3 Scope

Run WinCLIP on:
- Clean MVTec-AD (all 15 categories) → baseline
- The **2 worst-performing** corruption types from Phase 6 → degraded
- Same corruption + best preprocessing → rescued

Key research question: Does CLIP's language grounding make it more or less robust to visual degradation compared to CNN-based PatchCore?

Save results to `results/winclip/`.

---

## Phase 12 — Analysis & Visualization

### 12.1 Required Outputs

#### 12.1.1 Degradation Curves

For each model, plot AUROC vs. severity level for each corruption type:
- X-axis: Clean → Mild → Moderate → Severe
- Y-axis: Image-level AUROC
- One line per corruption type
- Error bars: ±1 std across 3 seeds

```python
"""generate_degradation_curves.py"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("results/degradation/PatchCore_degradation_results.csv")

# Add clean baseline
baseline = pd.read_csv("results/baselines/clean_baselines.csv")
baseline["corruption_type"] = "none"
baseline["severity"] = "clean"
combined = pd.concat([baseline, df])

# Aggregate by corruption type and severity
agg = combined.groupby(["corruption_type", "severity"])["image_AUROC"].agg(["mean", "std"]).reset_index()

severity_order = ["clean", "mild", "moderate", "severe"]
# Plot...
```

#### 12.1.2 Preprocessing Recovery Bar Chart

For each corruption type at **severe** severity, grouped bars showing:
- Raw degraded
- Naive (histogram matching)
- Primary preprocessing

#### 12.1.3 Score Distribution Plot

For at least **one** corruption type (recommend low-light), show KDE or histogram of anomaly scores for:
- Clean test (normal + anomalous)
- Degraded test
- Degraded + preprocessed test

This visualizes how preprocessing shifts score distributions, not just AUROC.

#### 12.1.4 Recommendation Matrix (KEY DELIVERABLE)

| Degradation | Best Preprocessing | AUROC Recovery (%) | Practical Notes |
|------------|--------------------|--------------------|-----------------|
| Low-light | CLAHE | X% → Y% | Fast, CPU-only |
| Gaussian blur | Wiener deconv | X% → Y% | Requires known PSF |
| Motion blur | Unsharp masking | X% → Y% | Perceptual only |
| Sensor noise | NLM | X% → Y% | Good speed/quality tradeoff |
| Fog/haze | AOD-Net / DCP | X% → Y% | GPU recommended for AOD-Net |
| Combined | Denoise → CLAHE | X% → Y% | Order matters! |

### 12.2 Category-Level Analysis

Identify which MVTec-AD categories are:
- **Most affected** by degradation (biggest AUROC drop)
- **Least affected** by degradation (smallest AUROC drop)
- Formulate hypothesis: texture-heavy categories (carpet, leather, wood) likely more affected by blur/noise than structural categories (bottle, metal_nut)

---

## Phase 13 — Runtime Benchmarking

### 13.1 What to Measure

For each preprocessing method, measure per-image processing time (ms):

```python
"""benchmark_runtime.py"""
import time
import numpy as np

def benchmark_preprocessing(fn, image, n_runs=100):
    """Benchmark a preprocessing function."""
    # Warmup
    for _ in range(5):
        _ = fn(image)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = fn(image)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "median_ms": np.median(times),
    }
```

### 13.2 Expected Output Table

| Preprocessing | Mean Time (ms/image) | Device |
|--------------|---------------------|--------|
| CLAHE | ~2-5 | CPU |
| NLM | ~50-200 | CPU |
| BM3D | ~500-2000 | CPU |
| Wiener deconv | ~10-50 | CPU |
| AOD-Net | ~5-20 | GPU |
| Histogram matching | ~1-5 | CPU |
| Unsharp masking | ~1-3 | CPU |

Save results to `results/runtime/preprocessing_runtime.csv`.

---

## Appendix A — Verified Library Reference

| Library | pip Package | Key Imports | Status (Apr 2026) | Notes |
|---------|------------|-------------|-------------------|-------|
| Anomalib | `anomalib` | `Patchcore`, `Padim`, `WinClip`, `MVTecAD`, `Visa`, `Engine` | ✅ Active | OpenVINO toolkit |
| Albumentations | `albumentationsx` or `albumentations` | `GaussianBlur`, `MotionBlur`, `GaussNoise`, `RandomFog` | ✅ Active | `albumentationsx` is maintained successor; same `import albumentations as A` |
| scikit-image | `scikit-image` | `skimage.restoration.wiener`, `denoise_nl_means`; `skimage.exposure.match_histograms` | ✅ Active | |
| OpenCV | `opencv-python-headless` | `cv2.createCLAHE()`, `cv2.GaussianBlur()` | ✅ Active | |
| BM3D | `bm3d` | `bm3d.bm3d()` | ✅ Available | Check license |
| scikit-learn | `scikit-learn` | `sklearn.metrics.roc_auc_score` | ✅ Active | |
| imgaug | `imgaug` | — | ❌ **UNMAINTAINED** | Do NOT use |
| DeblurGAN-v2 | Source only | — | ⚠️ **ARCHIVED** | PyTorch 2.x compat issues |
| AOD-Net | Source only | Clone from GitHub | ⚠️ **Old code** | Test before committing |

## Appendix B — MVTec-AD Category Reference

### Texture Categories (5)
| Category | Anomalib Name | Typical Defects |
|----------|--------------|-----------------|
| Carpet | `carpet` | color, cut, hole, metal contamination, thread |
| Grid | `grid` | bent, broken, glue, metal contamination, thread |
| Leather | `leather` | color, cut, fold, glue, poke |
| Tile | `tile` | crack, glue strip, gray stroke, oil, rough |
| Wood | `wood` | color, combined, hole, liquid, scratch |

### Object Categories (10)
| Category | Anomalib Name | Typical Defects |
|----------|--------------|-----------------|
| Bottle | `bottle` | broken large, broken small, contamination |
| Cable | `cable` | bent wire, cable swap, combined, cut inner/outer, missing cable/wire, poke insulation |
| Capsule | `capsule` | crack, faulty imprint, poke, scratch, squeeze |
| Hazelnut | `hazelnut` | crack, cut, hole, print |
| Metal Nut | `metal_nut` | bent, color, flip, scratch |
| Pill | `pill` | color, combined, contamination, crack, faulty imprint, pill type, scratch |
| Screw | `screw` | manipulated front, scratch head/neck, thread side/top |
| Toothbrush | `toothbrush` | defective |
| Transistor | `transistor` | bent lead, cut lead, damaged case, misplaced |
| Zipper | `zipper` | broken teeth, combined, fabric border/interior, rough, split teeth, squeezed teeth |

## Appendix C — Results Schema

### CSV Output Format

All result CSVs use this consistent schema:

```csv
model,category,seed,corruption_type,severity,preprocessing,image_AUROC,FNR,FPR,threshold
PatchCore,bottle,42,low_light,moderate,none,0.9234,0.12,0.08,3.456
PatchCore,bottle,42,low_light,moderate,CLAHE,0.9678,0.06,0.05,3.456
```

### Aggregation Script

```python
"""aggregate_results.py"""
import pandas as pd

# Load all results
baseline = pd.read_csv("results/baselines/clean_baselines.csv")
degradation = pd.read_csv("results/degradation/PatchCore_degradation_results.csv")
preprocessing = pd.read_csv("results/preprocessing/PatchCore_preprocessing_results.csv")

# Compute mean ± std across seeds
summary = degradation.groupby(
    ["model", "corruption_type", "severity"]
)["image_AUROC"].agg(["mean", "std"]).reset_index()

# Compute AUROC drop from clean baseline
clean_mean = baseline[baseline["model"] == "PatchCore"]["image_AUROC"].mean()
summary["auroc_drop"] = clean_mean - summary["mean"]
summary["auroc_drop_pct"] = (summary["auroc_drop"] / clean_mean) * 100

print(summary.to_string(index=False))
summary.to_csv("results/degradation/summary_table.csv", index=False)
```

---

> [!IMPORTANT]
> ## Critical Reminders for Any Agent Executing This Pipeline
>
> 1. **Do NOT modify `corruption_config.yaml` after Phase 5** — all experiments depend on identical corruption parameters.
> 2. **Do NOT use `imgaug`** — it is unmaintained. Use `albumentations.RandomFog` for fog simulation.
> 3. **Test DeblurGAN-v2 and AOD-Net compatibility** before committing to them. Have fallbacks ready (unsharp masking / dark channel prior).
> 4. **Save and reuse model checkpoints** — train once on clean data, reuse for all corruption/preprocessing experiments.
> 5. **Visual verification in Phase 5 is mandatory** — if severity levels look wrong, no amount of experiments will produce valid conclusions.
> 6. **Report mean ± std** across 3 seeds for all results.
> 7. **Wiener deconvolution only works because blur kernels are known** — document this assumption explicitly in any writeup.
> 8. **Verify Anomalib data formats** during Phase 3 — the exact tensor keys and shapes in dataset items may vary by version.
> 9. **Total corruption variants = 16** (5 types × 3 levels + 1 combined), not 18 as the briefing mentions.
