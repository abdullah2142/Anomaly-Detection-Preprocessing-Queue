# %% [markdown]
# # NB-5: Degradation Experiments — PaDiM
# **GPU**: ON | **Internet**: On | **Time**: ~8-10 hrs
#
# **Input Data Required**:
# - MVTec-AD dataset
# - NB-0 output (`experiment_config.json`)
# - NB-2 output (PaDiM checkpoints + baselines)
#
# Identical structure to NB-4 but uses PaDiM model.

# %% [markdown]
# ## Cell 1: Install

# %%
# (constraints replaced by --no-deps below)
!pip install -q anomalib --no-deps
!pip install -q lightning albumentationsx jsonargparse docstring_parser rich

# %% [markdown]
# ## Cell 2: Imports & Config

# %%
import json, os, time
import numpy as np
import pandas as pd
import torch
import lightning as L
import cv2
import albumentations as A
from sklearn.metrics import roc_auc_score
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Padim

CONFIG_PATH = "/kaggle/input/00-setup-and-verify/experiment_config.json"  # ADJUST
with open(CONFIG_PATH) as f:
    config = json.load(f)

MVTEC_ROOT = config["mvtec_root"]
CATEGORIES = config["categories"]
SEEDS = config["seeds"]

print(f"MVTec root: {MVTEC_ROOT}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## Cell 3: Corruption Functions (same as NB-4)

# %%
def apply_low_light(image, gamma, seed=42):
    rng = np.random.default_rng(seed)
    img_float = image.astype(np.float64) / 255.0
    img_dark = np.power(img_float, 1.0 / gamma)
    img_noisy = rng.poisson(np.clip(img_dark * 50.0, 0, None)) / 50.0
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

VARIANTS = []
for ctype in ["low_light", "gaussian_blur", "motion_blur", "sensor_noise", "fog_haze"]:
    for sev in ["mild", "moderate", "severe"]:
        VARIANTS.append((ctype, sev))
VARIANTS.append(("combined", "single"))
print(f"Total corruption variants: {len(VARIANTS)}")

# %% [markdown]
# ## Cell 4: Helpers (same as NB-4)

# %%
def load_test_images(category, mvtec_root):
    test_dir = os.path.join(mvtec_root, category, "test")
    images, labels, paths = [], [], []
    for defect_type in sorted(os.listdir(test_dir)):
        defect_dir = os.path.join(test_dir, defect_type)
        if not os.path.isdir(defect_dir): continue
        label = 0 if defect_type == "good" else 1
        for img_file in sorted(os.listdir(defect_dir)):
            if not img_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")): continue
            img_path = os.path.join(defect_dir, img_file)
            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(label)
            paths.append(img_path)
    return images, labels, paths

def get_anomaly_scores(model, images, device="cuda", target_size=(256, 256)):
    model.eval()
    model.to(device)
    scores = []
    with torch.no_grad():
        for img in images:
            img_resized = cv2.resize(img, target_size)
            img_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            output = model(img_tensor)
            if hasattr(output, 'pred_score'):
                score = output.pred_score.cpu().item()
            elif hasattr(output, 'anomaly_map'):
                score = output.anomaly_map.max().cpu().item()
            elif isinstance(output, torch.Tensor):
                score = output.max().cpu().item()
            else:
                raise ValueError(f"Unknown output type: {type(output)}")
            scores.append(score)
    return np.array(scores)

def compute_auroc(labels, scores):
    labels, scores = np.array(labels), np.array(scores)
    if len(np.unique(labels)) < 2: return float("nan")
    return roc_auc_score(labels, scores)

# Determine input size
test_dm = MVTecAD(root=MVTEC_ROOT, category="bottle", train_batch_size=1, eval_batch_size=1)
test_dm.setup("fit")
import dataclasses
train_item = test_dm.train_data[0]
if dataclasses.is_dataclass(train_item):
    img_field = getattr(train_item, "image", None)
    INPUT_SIZE = (img_field.shape[-2], img_field.shape[-1]) if img_field is not None and hasattr(img_field, 'shape') else (256, 256)
else:
    INPUT_SIZE = (256, 256)
print(f"Input size: {INPUT_SIZE}")
del test_dm

# %% [markdown]
# ## Cell 5: Run Degradation Experiments

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
        print(f"[{combo_count}/{total_combos}] PaDiM | {category} | seed={seed}")
        combo_start = time.time()

        model = Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"])
        datamodule = MVTecAD(root=MVTEC_ROOT, category=category,
                             train_batch_size=32, eval_batch_size=32)
        engine = Engine(max_epochs=1)
        engine.fit(model=model, datamodule=datamodule)

        test_images, test_labels, _ = load_test_images(category, MVTEC_ROOT)
        print(f"  Test set: {len(test_images)} images ({sum(test_labels)} anomalous)")

        # Clean baseline
        clean_scores = get_anomaly_scores(model, test_images, target_size=INPUT_SIZE)
        clean_auroc = compute_auroc(test_labels, clean_scores)
        results.append({"model": "PaDiM", "category": category, "seed": seed,
                        "corruption_type": "clean", "severity": "none", "image_AUROC": clean_auroc})
        print(f"  Clean: {clean_auroc:.4f}")

        for ctype, severity in VARIANTS:
            corrupted = [apply_corruption(img, ctype, severity, config, seed=seed+i)
                         for i, img in enumerate(test_images)]
            scores = get_anomaly_scores(model, corrupted, target_size=INPUT_SIZE)
            auroc = compute_auroc(test_labels, scores)
            results.append({"model": "PaDiM", "category": category, "seed": seed,
                            "corruption_type": ctype, "severity": severity, "image_AUROC": auroc})
            print(f"  {ctype}/{severity}: {auroc:.4f}")

        del model, engine
        torch.cuda.empty_cache()
        combo_time = time.time() - combo_start
        elapsed = time.time() - start_time
        eta = (elapsed / combo_count) * (total_combos - combo_count)
        print(f"  Time: {combo_time/60:.1f}m | Elapsed: {elapsed/3600:.1f}h | ETA: {eta/3600:.1f}h")
        pd.DataFrame(results).to_csv("/kaggle/working/degradation/padim_degradation_WIP.csv", index=False)

# %% [markdown]
# ## Cell 6: Save & Summary

# %%
df = pd.DataFrame(results)
df.to_csv("/kaggle/working/degradation/padim_degradation.csv", index=False)

pivot = df.groupby(["corruption_type", "severity"])["image_AUROC"].agg(["mean", "std"]).round(4)
print("\nPaDiM Degradation Summary")
print("="*70)
print(pivot.to_string())

clean_mean = df[df["corruption_type"] == "clean"]["image_AUROC"].mean()
print(f"\nClean baseline: {clean_mean:.4f}")
for (ctype, sev), row in pivot.iterrows():
    if ctype != "clean":
        print(f"  {ctype}/{sev}: {row['mean']:.4f} (drop={clean_mean - row['mean']:+.4f})")

print(f"\nWall time: {(time.time()-start_time)/3600:.2f}h")
print("📋 NEXT: Save & Run All → NB-6 (Preprocessing)")
