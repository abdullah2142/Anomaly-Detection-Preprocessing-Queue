# %% [markdown]
# # NB-4: Degradation Experiments — PatchCore
# **GPU**: ON | **Internet**: On | **Time**: ~10-12 hrs (fits in one session)
#
# **Input Data Required**:
# - MVTec-AD dataset
# - NB-0 output (`experiment_config.json`)
# - NB-1 output (PatchCore checkpoints + baselines)
#
# This notebook runs PatchCore inference on all 16 corruption variants
# (5 types × 3 severities + 1 combined) across all 15 categories and 3 seeds.
#
# **Key design**: Model is trained ONCE per (category, seed) on clean data,
# then tested on all 16 corruption variants — corruption is applied on-the-fly.

# %% [markdown]
# ## Cell 1: Install

# %%
# (constraints replaced by --no-deps below)
!pip install -q anomalib --no-deps
!pip install -q lightning albumentationsx jsonargparse docstring_parser rich

# %% [markdown]
# ## Cell 2: Imports & Config

# %%
import json, os, time, glob, shutil
import numpy as np
import pandas as pd
import torch
import lightning as L
import cv2
import albumentations as A
from pathlib import Path
from sklearn.metrics import roc_auc_score
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

# Load config
# ⚠️ ADJUST these paths to match your Kaggle notebook output slugs
CONFIG_PATH = "/kaggle/input/00-setup-and-verify/experiment_config.json"
BASELINE_PATH = "/kaggle/input/01-baselines-patchcore/baselines/patchcore_baselines.csv"

with open(CONFIG_PATH) as f:
    config = json.load(f)

MVTEC_ROOT = config["mvtec_root"]
CATEGORIES = config["categories"]
SEEDS = config["seeds"]

print(f"MVTec root: {MVTEC_ROOT}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## Cell 3: Corruption Functions

# %%
def apply_low_light(image, gamma, seed=42):
    rng = np.random.default_rng(seed)
    img_float = image.astype(np.float64) / 255.0
    img_dark = np.power(img_float, 1.0 / gamma)
    photon_scale = 50.0
    img_noisy = rng.poisson(np.clip(img_dark * photon_scale, 0, None)) / photon_scale
    return np.clip(img_noisy * 255, 0, 255).astype(np.uint8)

def apply_gaussian_blur(image, sigma, kernel_size):
    t = A.GaussianBlur(blur_limit=(kernel_size, kernel_size), sigma_limit=(sigma, sigma), p=1.0)
    return t(image=image)["image"]

def apply_motion_blur(image, kernel_size):
    t = A.MotionBlur(blur_limit=(kernel_size, kernel_size), p=1.0)
    return t(image=image)["image"]

def apply_sensor_noise(image, gauss_var, seed=42):
    rng = np.random.default_rng(seed)
    img_float = image.astype(np.float64) / 255.0
    noise = rng.normal(0, np.sqrt(gauss_var), img_float.shape)
    img_noisy = img_float + noise
    sp_ratio = 0.05
    salt = rng.random(img_float.shape[:2]) < (sp_ratio / 2)
    pepper = rng.random(img_float.shape[:2]) < (sp_ratio / 2)
    img_noisy[salt] = 1.0
    img_noisy[pepper] = 0.0
    return np.clip(img_noisy * 255, 0, 255).astype(np.uint8)

def apply_fog(image, fog_coef_lower, fog_coef_upper, alpha_coef=0.1):
    t = A.RandomFog(fog_coef_lower=fog_coef_lower, fog_coef_upper=fog_coef_upper,
                    alpha_coef=alpha_coef, p=1.0)
    return t(image=image)["image"]

def apply_combined(image, seed=42):
    img = apply_low_light(image, gamma=0.35, seed=seed)
    return apply_sensor_noise(img, gauss_var=0.05, seed=seed+1)

def apply_corruption(image, ctype, severity, config, seed=42):
    params = config["corruptions"][ctype][severity]
    if ctype == "low_light": return apply_low_light(image, params["gamma"], seed)
    elif ctype == "gaussian_blur": return apply_gaussian_blur(image, params["sigma"], params["kernel_size"])
    elif ctype == "motion_blur": return apply_motion_blur(image, params["kernel_size"])
    elif ctype == "sensor_noise": return apply_sensor_noise(image, params["gauss_var"], seed)
    elif ctype == "fog_haze": return apply_fog(image, params["fog_coef_lower"], params["fog_coef_upper"], params.get("alpha_coef", 0.1))
    elif ctype == "combined": return apply_combined(image, seed)
    else: raise ValueError(f"Unknown: {ctype}")

# Build variant list
VARIANTS = []
for ctype in ["low_light", "gaussian_blur", "motion_blur", "sensor_noise", "fog_haze"]:
    for sev in ["mild", "moderate", "severe"]:
        VARIANTS.append((ctype, sev))
VARIANTS.append(("combined", "single"))
print(f"Total corruption variants: {len(VARIANTS)}")  # 16

# %% [markdown]
# ## Cell 4: Helper — Load test images manually & run inference

# %%
def load_test_images(category, mvtec_root):
    """Load all test images + labels from MVTec-AD category.

    Returns:
        images: list of uint8 HWC RGB numpy arrays
        labels: list of int (0=normal, 1=anomalous)
        paths: list of file paths
    """
    test_dir = os.path.join(mvtec_root, category, "test")
    images, labels, paths = [], [], []

    for defect_type in sorted(os.listdir(test_dir)):
        defect_dir = os.path.join(test_dir, defect_type)
        if not os.path.isdir(defect_dir):
            continue

        label = 0 if defect_type == "good" else 1

        for img_file in sorted(os.listdir(defect_dir)):
            if not img_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            img_path = os.path.join(defect_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(label)
            paths.append(img_path)

    return images, labels, paths


def get_anomaly_scores(model, images, device="cuda", target_size=(256, 256)):
    """Run model inference on a list of images, return anomaly scores.

    Args:
        model: trained Anomalib model
        images: list of uint8 HWC RGB numpy arrays
        device: 'cuda' or 'cpu'
        target_size: resize images to this size (H, W)

    Returns:
        scores: numpy array of image-level anomaly scores
    """
    model.eval()
    model.to(device)
    scores = []

    with torch.no_grad():
        for img in images:
            # Resize to model's expected input size
            img_resized = cv2.resize(img, target_size)
            # Convert to float tensor [C, H, W], normalized to [0, 1]
            img_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            img_tensor = img_tensor.to(device)

            # Model forward pass — output depends on Anomalib version
            output = model(img_tensor)

            # Extract image-level anomaly score
            # Anomalib models return different formats; try common patterns
            if hasattr(output, 'pred_score'):
                score = output.pred_score.cpu().item()
            elif hasattr(output, 'anomaly_map'):
                # Use max of anomaly map as image-level score
                score = output.anomaly_map.max().cpu().item()
            elif isinstance(output, dict):
                score = output.get("pred_score", output.get("anomaly_map", torch.tensor(0)).max()).cpu().item()
            elif isinstance(output, torch.Tensor):
                score = output.max().cpu().item()
            else:
                raise ValueError(f"Unknown output type: {type(output)}")

            scores.append(score)

    return np.array(scores)


def compute_auroc(labels, scores):
    """Compute image-level AUROC. Returns NaN if only one class present."""
    labels = np.array(labels)
    scores = np.array(scores)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, scores)

# %% [markdown]
# ## Cell 5: Discover input image size from Anomalib

# %%
# Determine the correct input size by looking at the datamodule
test_dm = MVTecAD(root=MVTEC_ROOT, category="bottle", train_batch_size=1, eval_batch_size=1)
test_dm.setup("fit")

# Get image size from a training sample
import dataclasses
train_item = test_dm.train_data[0]
if dataclasses.is_dataclass(train_item):
    img_field = getattr(train_item, "image", None)
    if img_field is not None and hasattr(img_field, 'shape'):
        INPUT_SIZE = (img_field.shape[-2], img_field.shape[-1])  # (H, W)
    else:
        INPUT_SIZE = (256, 256)
else:
    INPUT_SIZE = (256, 256)

print(f"Model input size: {INPUT_SIZE}")
del test_dm

# %% [markdown]
# ## Cell 6: Run Degradation Experiments

# %%
os.makedirs("/kaggle/working/degradation", exist_ok=True)

results = []
start_time = time.time()
total_combos = len(CATEGORIES) * len(SEEDS)
combo_count = 0

for category in CATEGORIES:
    for seed in SEEDS:
        combo_count += 1
        L.seed_everything(seed)

        print(f"\n{'='*70}")
        print(f"[{combo_count}/{total_combos}] {category} | seed={seed}")
        print(f"{'='*70}")
        combo_start = time.time()

        # === TRAIN MODEL ON CLEAN DATA ===
        model = Patchcore(backbone="wide_resnet50_2", num_neighbors=9)
        datamodule = MVTecAD(
            root=MVTEC_ROOT, category=category,
            train_batch_size=32, eval_batch_size=32,
        )
        engine = Engine(max_epochs=1)
        engine.fit(model=model, datamodule=datamodule)

        # === LOAD TEST IMAGES ===
        test_images, test_labels, test_paths = load_test_images(category, MVTEC_ROOT)
        print(f"  Test set: {len(test_images)} images ({sum(test_labels)} anomalous)")

        # === CLEAN BASELINE (for comparison) ===
        clean_scores = get_anomaly_scores(model, test_images, target_size=INPUT_SIZE)
        clean_auroc = compute_auroc(test_labels, clean_scores)
        results.append({
            "model": "PatchCore", "category": category, "seed": seed,
            "corruption_type": "clean", "severity": "none",
            "image_AUROC": clean_auroc,
        })
        print(f"  Clean AUROC: {clean_auroc:.4f}")

        # === RUN EACH CORRUPTION VARIANT ===
        for ctype, severity in VARIANTS:
            # Apply corruption to all test images
            corrupted_images = []
            for i, img in enumerate(test_images):
                corrupted = apply_corruption(img, ctype, severity, config, seed=seed+i)
                corrupted_images.append(corrupted)

            # Get anomaly scores on corrupted images
            corrupt_scores = get_anomaly_scores(model, corrupted_images, target_size=INPUT_SIZE)
            auroc = compute_auroc(test_labels, corrupt_scores)

            results.append({
                "model": "PatchCore", "category": category, "seed": seed,
                "corruption_type": ctype, "severity": severity,
                "image_AUROC": auroc,
            })
            print(f"  {ctype}/{severity}: AUROC={auroc:.4f}")

        # Cleanup to free GPU memory
        del model, engine
        torch.cuda.empty_cache()

        combo_time = time.time() - combo_start
        elapsed = time.time() - start_time
        eta = (elapsed / combo_count) * (total_combos - combo_count)
        print(f"  Time: {combo_time/60:.1f} min | Elapsed: {elapsed/3600:.1f}h | ETA: {eta/3600:.1f}h")

        # Save intermediate results (in case session times out)
        df_interim = pd.DataFrame(results)
        df_interim.to_csv("/kaggle/working/degradation/patchcore_degradation_WIP.csv", index=False)

# %% [markdown]
# ## Cell 7: Save Final Results & Summary

# %%
df = pd.DataFrame(results)
df.to_csv("/kaggle/working/degradation/patchcore_degradation.csv", index=False)

# Summary: mean AUROC per corruption/severity
pivot = df.groupby(["corruption_type", "severity"])["image_AUROC"].agg(["mean", "std"]).round(4)
print("\n" + "="*70)
print("PatchCore Degradation Summary (mean ± std across categories & seeds)")
print("="*70)
print(pivot.to_string())

# AUROC drop from clean baseline
clean_mean = df[df["corruption_type"] == "clean"]["image_AUROC"].mean()
print(f"\nClean baseline mean: {clean_mean:.4f}")
for (ctype, sev), row in pivot.iterrows():
    if ctype != "clean":
        drop = clean_mean - row["mean"]
        print(f"  {ctype}/{sev}: AUROC={row['mean']:.4f} (drop={drop:+.4f})")

total_time = time.time() - start_time
print(f"\nTotal wall time: {total_time/3600:.2f} hours")
print(f"Results saved to /kaggle/working/degradation/patchcore_degradation.csv")
print("\n📋 NEXT: Save & Run All → proceed to NB-5 (PaDiM degradation)")
