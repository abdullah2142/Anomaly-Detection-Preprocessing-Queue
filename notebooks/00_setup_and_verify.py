# %% [markdown]
# # NB-0: Setup & Environment Verification
# **GPU**: Off | **Internet**: On | **Time**: ~10 min
#
# **Input Data Required**: Add `ipythonx/mvtec-ad` dataset from Kaggle Datasets
#
# This notebook:
# 1. Installs missing packages
# 2. Verifies all imports work
# 3. Locates MVTec-AD dataset path
# 4. Saves shared experiment config
#
# **After running**: Click "Save & Run All" → output becomes input for all subsequent notebooks.

# %% [markdown]
# ## Cell 1: Install Missing Packages

# %%
# Kaggle pre-installed: torch, torchvision, numpy, scipy, sklearn, skimage, cv2, matplotlib, pandas, seaborn
# Install anomalib WITHOUT deps to protect Kaggle's CUDA stack:
!pip install -q anomalib --no-deps
!pip install -q lightning albumentationsx jsonargparse docstring_parser rich
!pip install -q bm3d pyyaml

# %% [markdown]
# ## Cell 2: Verify Environment

# %%
import sys

checks = []

# 1. Anomalib core
try:
    from anomalib.models import Patchcore, Padim
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    checks.append(("Anomalib (Patchcore, Padim, MVTecAD, Engine)", True, ""))
except ImportError as e:
    checks.append(("Anomalib", False, str(e)))

# 2. WinCLIP (stretch goal — non-fatal)
try:
    from anomalib.models import WinClip
    checks.append(("WinCLIP (stretch goal)", True, ""))
except ImportError as e:
    checks.append(("WinCLIP (stretch goal — non-fatal)", False, str(e)))

# 3. Albumentations with required transforms
try:
    import albumentations as A
    _ = A.GaussianBlur; _ = A.MotionBlur; _ = A.GaussNoise; _ = A.RandomFog
    checks.append(("Albumentations (GaussianBlur, MotionBlur, GaussNoise, RandomFog)", True, ""))
except (ImportError, AttributeError) as e:
    checks.append(("Albumentations", False, str(e)))

# 4. scikit-image restoration
try:
    from skimage.restoration import wiener, denoise_nl_means
    from skimage.exposure import match_histograms
    checks.append(("scikit-image (wiener, denoise_nl_means, match_histograms)", True, ""))
except ImportError as e:
    checks.append(("scikit-image", False, str(e)))

# 5. OpenCV CLAHE
try:
    import cv2
    clahe_test = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    checks.append(("OpenCV CLAHE (cv2.createCLAHE)", True, ""))
except (ImportError, AttributeError) as e:
    checks.append(("OpenCV", False, str(e)))

# 6. BM3D
try:
    import bm3d
    checks.append(("BM3D", True, ""))
except ImportError as e:
    checks.append(("BM3D", False, str(e)))

# 7. PyTorch + GPU
try:
    import torch
    gpu = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if gpu else "CPU only"
    checks.append((f"PyTorch (GPU: {device})", True, ""))
except Exception as e:
    checks.append(("PyTorch", False, str(e)))

# 8. sklearn
try:
    from sklearn.metrics import roc_auc_score
    checks.append(("scikit-learn (roc_auc_score)", True, ""))
except ImportError as e:
    checks.append(("scikit-learn", False, str(e)))

# Report
print("=" * 70)
print("ENVIRONMENT VERIFICATION REPORT")
print("=" * 70)
all_pass = True
for name, passed, err in checks:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  {name}")
    if not passed:
        print(f"           Error: {err}")
        if "stretch" not in name.lower():
            all_pass = False
print("=" * 70)
if all_pass:
    print("✅ All critical checks passed. Proceed.")
else:
    print("❌ CRITICAL FAILURES. Fix before proceeding.")

# %% [markdown]
# ## Cell 3: Locate MVTec-AD Dataset

# %%
import os

EXPECTED_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

# Kaggle mounts datasets at /kaggle/input/<dataset-slug>/
# Try common paths
CANDIDATE_ROOTS = [
    "/kaggle/input/mvtec-ad",
    "/kaggle/input/mvtec-anomaly-detection",
    "/kaggle/input/mvtecad",
]

MVTEC_ROOT = None

for root in CANDIDATE_ROOTS:
    if not os.path.isdir(root):
        continue
    # Check if categories are directly here
    dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if set(EXPECTED_CATEGORIES).issubset(set(dirs)):
        MVTEC_ROOT = root
        break
    # Check one level deeper
    for sub in dirs:
        deeper = os.path.join(root, sub)
        if os.path.isdir(deeper):
            deeper_dirs = [d for d in os.listdir(deeper) if os.path.isdir(os.path.join(deeper, d))]
            if set(EXPECTED_CATEGORIES).issubset(set(deeper_dirs)):
                MVTEC_ROOT = deeper
                break
    if MVTEC_ROOT:
        break

if MVTEC_ROOT is None:
    # Fallback: list everything in /kaggle/input for debugging
    print("❌ Could not auto-detect MVTec-AD root. Listing /kaggle/input/:")
    for item in os.listdir("/kaggle/input"):
        sub_path = os.path.join("/kaggle/input", item)
        if os.path.isdir(sub_path):
            children = os.listdir(sub_path)[:10]
            print(f"  📁 {item}/ → {children}")
    print("\n⚠️  Manually set MVTEC_ROOT in the next cell!")
else:
    print(f"✅ Found MVTec-AD at: {MVTEC_ROOT}")
    # Verify all categories
    found = sorted([d for d in os.listdir(MVTEC_ROOT)
                    if os.path.isdir(os.path.join(MVTEC_ROOT, d))])
    print(f"   Categories ({len(found)}/15): {found}")

    # Verify structure for one category
    cat = "bottle"
    for sub in ["train/good", "test/good", "ground_truth"]:
        full = os.path.join(MVTEC_ROOT, cat, sub)
        if os.path.isdir(full):
            count = len(os.listdir(full))
            print(f"   ✅ {cat}/{sub}: {count} items")
        else:
            print(f"   ❌ {cat}/{sub}: NOT FOUND")

# %% [markdown]
# ## Cell 4: Manual Override (if auto-detect failed)

# %%
# Uncomment and set manually if Cell 3 could not auto-detect:
# MVTEC_ROOT = "/kaggle/input/YOUR-DATASET-SLUG/path/to/categories"

assert MVTEC_ROOT is not None, "MVTEC_ROOT must be set before proceeding!"
print(f"Using MVTEC_ROOT = '{MVTEC_ROOT}'")

# %% [markdown]
# ## Cell 5: Save Experiment Config

# %%
import json

config = {
    "mvtec_root": MVTEC_ROOT,
    "categories": EXPECTED_CATEGORIES,
    "seeds": [42, 123, 456],
    "corruptions": {
        "low_light": {
            "mild": {"gamma": 0.5},
            "moderate": {"gamma": 0.35},
            "severe": {"gamma": 0.2}
        },
        "gaussian_blur": {
            "mild": {"sigma": 1.0, "kernel_size": 7},
            "moderate": {"sigma": 2.0, "kernel_size": 11},
            "severe": {"sigma": 4.0, "kernel_size": 23}
        },
        "motion_blur": {
            "mild": {"kernel_size": 7},
            "moderate": {"kernel_size": 15},
            "severe": {"kernel_size": 25}
        },
        "sensor_noise": {
            "mild": {"gauss_var": 0.02},
            "moderate": {"gauss_var": 0.05},
            "severe": {"gauss_var": 0.10}
        },
        "fog_haze": {
            "mild": {"fog_coef_lower": 0.2, "fog_coef_upper": 0.35, "alpha_coef": 0.1},
            "moderate": {"fog_coef_lower": 0.45, "fog_coef_upper": 0.65, "alpha_coef": 0.1},
            "severe": {"fog_coef_lower": 0.75, "fog_coef_upper": 0.95, "alpha_coef": 0.1}
        },
        "combined": {
            "single": {"gamma": 0.35, "gauss_var": 0.05}
        }
    }
}

os.makedirs("/kaggle/working", exist_ok=True)
with open("/kaggle/working/experiment_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("✅ Config saved to /kaggle/working/experiment_config.json")
print(f"   Total corruption variants: 5×3 + 1 = 16")
print(f"   Seeds: {config['seeds']}")
print(f"   Categories: {len(config['categories'])}")
print()
print("⚠️  IMPORTANT: After NB-3 severity calibration approval,")
print("   corruption parameters are FROZEN. No changes allowed.")
print()
print("📋 NEXT STEPS:")
print("   1. Click 'Save & Run All' to commit this notebook's output")
print("   2. Proceed to NB-1 (PatchCore Baselines) — add this notebook as input data")
