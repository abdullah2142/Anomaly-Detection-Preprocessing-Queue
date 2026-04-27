# %% [markdown]
# # NB-6a: Preprocessing Rescue — PatchCore (Part 1: Low-light, Blur)
# **GPU**: ON | **Internet**: On | **Time**: ~9-11 hrs
#
# **Input Data Required**:
# - MVTec-AD dataset
# - NB-0 output (`experiment_config.json`)
# - NB-4 output (degradation results for reference)
#
# This notebook covers preprocessing rescue for:
# - low_light (3 severities × 2 methods: CLAHE, HistMatch)
# - gaussian_blur (3 severities × 2 methods: Wiener, HistMatch)
# - motion_blur (3 severities × 2 methods: UnsharpMask, HistMatch)
# Total: 18 preprocessing variants across 15 categories × 3 seeds

# %% [markdown]
# ## Cell 1: Install

# %%
# (constraints replaced by --no-deps below)
!pip install -q anomalib --no-deps
!pip install -q lightning albumentationsx jsonargparse docstring_parser rich

# %% [markdown]
# ## Cell 2: Imports

# %%
import json, os, time
import numpy as np
import pandas as pd
import torch
import lightning as L
import cv2
import albumentations as A
from sklearn.metrics import roc_auc_score
from skimage.restoration import wiener
from skimage import img_as_float, img_as_ubyte
from skimage.exposure import match_histograms
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

CONFIG_PATH = "/kaggle/input/00-setup-and-verify/experiment_config.json"  # ADJUST
with open(CONFIG_PATH) as f:
    config = json.load(f)

MVTEC_ROOT = config["mvtec_root"]
CATEGORIES = config["categories"]
SEEDS = config["seeds"]
print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## Cell 3: All Functions (corruption + preprocessing + helpers)

# %%
# === CORRUPTION FUNCTIONS ===
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

def apply_corruption(image, ctype, severity, config, seed=42):
    params = config["corruptions"][ctype][severity]
    if ctype == "low_light": return apply_low_light(image, params["gamma"], seed)
    elif ctype == "gaussian_blur": return apply_gaussian_blur(image, params["sigma"], params["kernel_size"])
    elif ctype == "motion_blur": return apply_motion_blur(image, params["kernel_size"])
    else: raise ValueError(f"Not handled in this notebook: {ctype}")

# === PREPROCESSING FUNCTIONS ===
def apply_clahe(image, clip_limit=3.0, tile_grid_size=(8, 8)):
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    lab_out = cv2.merge((cl, a, b))
    bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)

def apply_wiener_deconv(image, sigma, kernel_size, balance=0.1):
    img_float = img_as_float(image)
    ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    psf /= psf.sum()
    result = np.zeros_like(img_float)
    for c in range(3):
        result[:, :, c] = wiener(img_float[:, :, c], psf, balance)
    return img_as_ubyte(np.clip(result, 0, 1))

def apply_unsharp_masking(image, sigma=2.0, strength=1.5):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def apply_histogram_matching(image, reference_image):
    matched = match_histograms(image, reference_image, channel_axis=-1)
    return np.clip(matched, 0, 255).astype(np.uint8)

def get_reference_image(category, mvtec_root):
    train_dir = os.path.join(mvtec_root, category, "train", "good")
    files = sorted(os.listdir(train_dir))
    brightnesses = [np.mean(cv2.imread(os.path.join(train_dir, f))) for f in files]
    median_idx = np.argsort(brightnesses)[len(brightnesses) // 2]
    ref = cv2.imread(os.path.join(train_dir, files[median_idx]))
    return cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

# === HELPERS ===
def load_test_images(category, mvtec_root):
    test_dir = os.path.join(mvtec_root, category, "test")
    images, labels, paths = [], [], []
    for defect_type in sorted(os.listdir(test_dir)):
        defect_dir = os.path.join(test_dir, defect_type)
        if not os.path.isdir(defect_dir): continue
        label = 0 if defect_type == "good" else 1
        for img_file in sorted(os.listdir(defect_dir)):
            if not img_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")): continue
            img = cv2.imread(os.path.join(defect_dir, img_file))
            if img is None: continue
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            labels.append(label)
            paths.append(os.path.join(defect_dir, img_file))
    return images, labels, paths

def get_anomaly_scores(model, images, device="cuda", target_size=(256, 256)):
    model.eval(); model.to(device)
    scores = []
    with torch.no_grad():
        for img in images:
            img_r = cv2.resize(img, target_size)
            t = torch.from_numpy(img_r.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            out = model(t)
            if hasattr(out, 'pred_score'): scores.append(out.pred_score.cpu().item())
            elif hasattr(out, 'anomaly_map'): scores.append(out.anomaly_map.max().cpu().item())
            else: scores.append(out.max().cpu().item() if isinstance(out, torch.Tensor) else 0.0)
    return np.array(scores)

def compute_auroc(labels, scores):
    labels, scores = np.array(labels), np.array(scores)
    return roc_auc_score(labels, scores) if len(np.unique(labels)) >= 2 else float("nan")

# Determine input size
_dm = MVTecAD(root=MVTEC_ROOT, category="bottle", train_batch_size=1, eval_batch_size=1)
_dm.setup("fit")
import dataclasses
_item = _dm.train_data[0]
INPUT_SIZE = (getattr(_item, "image").shape[-2], getattr(_item, "image").shape[-1]) if dataclasses.is_dataclass(_item) and hasattr(getattr(_item, "image", None), 'shape') else (256, 256)
del _dm
print(f"Input size: {INPUT_SIZE}")

# %% [markdown]
# ## Cell 4: Define Experiment Variants for This Notebook

# %%
# This notebook handles: low_light, gaussian_blur, motion_blur
EXPERIMENT_VARIANTS = []

for sev in ["mild", "moderate", "severe"]:
    # Low-light → CLAHE, HistMatch
    EXPERIMENT_VARIANTS.append(("low_light", sev, "CLAHE"))
    EXPERIMENT_VARIANTS.append(("low_light", sev, "HistMatch"))
    # Gaussian blur → Wiener, HistMatch
    EXPERIMENT_VARIANTS.append(("gaussian_blur", sev, "Wiener"))
    EXPERIMENT_VARIANTS.append(("gaussian_blur", sev, "HistMatch"))
    # Motion blur → UnsharpMask, HistMatch
    EXPERIMENT_VARIANTS.append(("motion_blur", sev, "UnsharpMask"))
    EXPERIMENT_VARIANTS.append(("motion_blur", sev, "HistMatch"))

print(f"Total variants: {len(EXPERIMENT_VARIANTS)}")  # 18
for v in EXPERIMENT_VARIANTS:
    print(f"  {v[0]}/{v[1]} → {v[2]}")

# %% [markdown]
# ## Cell 5: Run Preprocessing Experiments

# %%
os.makedirs("/kaggle/working/preprocessing", exist_ok=True)
results = []
start_time = time.time()
total_combos = len(CATEGORIES) * len(SEEDS)
combo_count = 0

# Cache reference images for histogram matching
ref_images = {cat: get_reference_image(cat, MVTEC_ROOT) for cat in CATEGORIES}

for category in CATEGORIES:
    for seed in SEEDS:
        combo_count += 1
        L.seed_everything(seed)
        print(f"\n{'='*70}")
        print(f"[{combo_count}/{total_combos}] PatchCore | {category} | seed={seed}")
        combo_start = time.time()

        # Train model on clean data
        model = Patchcore(backbone="wide_resnet50_2", num_neighbors=9)
        datamodule = MVTecAD(root=MVTEC_ROOT, category=category,
                             train_batch_size=32, eval_batch_size=32)
        engine = Engine(max_epochs=1)
        engine.fit(model=model, datamodule=datamodule)

        test_images, test_labels, _ = load_test_images(category, MVTEC_ROOT)
        ref_img = ref_images[category]

        for ctype, sev, preprocess_name in EXPERIMENT_VARIANTS:
            # Step 1: Corrupt
            corrupted = [apply_corruption(img, ctype, sev, config, seed=seed+i)
                         for i, img in enumerate(test_images)]

            # Step 2: Preprocess
            if preprocess_name == "CLAHE":
                processed = [apply_clahe(img) for img in corrupted]
            elif preprocess_name == "Wiener":
                s = config["corruptions"]["gaussian_blur"][sev]["sigma"]
                ks = config["corruptions"]["gaussian_blur"][sev]["kernel_size"]
                processed = [apply_wiener_deconv(img, s, ks) for img in corrupted]
            elif preprocess_name == "UnsharpMask":
                processed = [apply_unsharp_masking(img) for img in corrupted]
            elif preprocess_name == "HistMatch":
                processed = [apply_histogram_matching(img, ref_img) for img in corrupted]
            else:
                raise ValueError(f"Unknown: {preprocess_name}")

            # Step 3: Inference
            scores = get_anomaly_scores(model, processed, target_size=INPUT_SIZE)
            auroc = compute_auroc(test_labels, scores)

            results.append({
                "model": "PatchCore", "category": category, "seed": seed,
                "corruption_type": ctype, "severity": sev,
                "preprocessing": preprocess_name, "image_AUROC": auroc,
            })
            print(f"  {ctype}/{sev}→{preprocess_name}: {auroc:.4f}")

        del model, engine
        torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        eta = (elapsed / combo_count) * (total_combos - combo_count)
        print(f"  Elapsed: {elapsed/3600:.1f}h | ETA: {eta/3600:.1f}h")
        pd.DataFrame(results).to_csv(
            "/kaggle/working/preprocessing/patchcore_preprocess_blur_light_WIP.csv", index=False)

# %% [markdown]
# ## Cell 6: Save Final Results

# %%
df = pd.DataFrame(results)
df.to_csv("/kaggle/working/preprocessing/patchcore_preprocess_blur_light.csv", index=False)

pivot = df.groupby(["corruption_type", "severity", "preprocessing"])["image_AUROC"].agg(["mean", "std"]).round(4)
print("\nPreprocessing Rescue Results (PatchCore — blur/light)")
print("="*70)
print(pivot.to_string())
print(f"\nWall time: {(time.time()-start_time)/3600:.2f}h")
print("📋 NEXT: Save & Run All → NB-6b (noise/fog/combined)")
