# %% [markdown]
# # NB-7b: Preprocessing Rescue — PaDiM (Part 2: Noise, Fog, Combined)
# **GPU**: ON | **Internet**: On | **Time**: ~7-10 hrs
#
# **Input Data Required**:
# - MVTec-AD dataset
# - NB-0 output (`experiment_config.json`)
#
# Same structure as NB-6b but uses PaDiM model.

# %% [markdown]
# ## Cell 1: Install

# %%
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
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.exposure import match_histograms
from scipy.ndimage import minimum_filter
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Padim

CONFIG_PATH = "/kaggle/input/00-setup-and-verify/experiment_config.json"  # ADJUST
with open(CONFIG_PATH) as f:
    config = json.load(f)

MVTEC_ROOT = config["mvtec_root"]
CATEGORIES = config["categories"]
SEEDS = config["seeds"]
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# %% [markdown]
# ## Cell 3: Functions

# %%
def apply_low_light(image, gamma, seed=42):
    rng = np.random.default_rng(seed)
    img_float = image.astype(np.float64) / 255.0
    img_dark = np.power(img_float, 1.0 / gamma)
    img_noisy = rng.poisson(np.clip(img_dark * 50.0, 0, None)) / 50.0
    return np.clip(img_noisy * 255, 0, 255).astype(np.uint8)

def apply_sensor_noise(image, gauss_var, seed=42):
    rng = np.random.default_rng(seed)
    img_float = image.astype(np.float64) / 255.0
    noise = rng.normal(0, np.sqrt(gauss_var), img_float.shape)
    img_noisy = img_float + noise
    sp_ratio = 0.05
    salt = rng.random(img_float.shape[:2]) < (sp_ratio / 2)
    pepper = rng.random(img_float.shape[:2]) < (sp_ratio / 2)
    img_noisy[salt] = 1.0; img_noisy[pepper] = 0.0
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
    if ctype == "sensor_noise": return apply_sensor_noise(image, params["gauss_var"], seed)
    elif ctype == "fog_haze": return apply_fog(image, params["fog_coef_lower"], params["fog_coef_upper"], params.get("alpha_coef", 0.1))
    elif ctype == "combined": return apply_combined(image, seed)
    else: raise ValueError(f"Not handled: {ctype}")

def apply_clahe(image, clip_limit=3.0, tile_grid_size=(8, 8)):
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)

def apply_nlm_denoise(image, patch_size=7, patch_distance=11):
    img_float = image.astype(np.float64) / 255.0
    sigma_est = np.mean(estimate_sigma(img_float, channel_axis=-1))
    denoised = denoise_nl_means(img_float, h=1.15 * sigma_est, patch_size=patch_size,
                                 patch_distance=patch_distance, fast_mode=True, channel_axis=-1)
    return np.clip(denoised * 255, 0, 255).astype(np.uint8)

def apply_dark_channel_prior_dehaze(image, omega=0.95, patch_size=15):
    img = image.astype(np.float64) / 255.0
    dark = np.min(img, axis=2)
    dark_ch = minimum_filter(dark, size=patch_size)
    flat = dark_ch.ravel()
    top_idx = np.argsort(flat)[-max(1, int(0.001 * len(flat))):]
    A_light = np.max(img.reshape(-1, 3)[top_idx], axis=0)
    normed = img / (A_light + 1e-6)
    normed_dark = minimum_filter(np.min(normed, axis=2), size=patch_size)
    transmission = np.clip(1 - omega * normed_dark, 0.1, 1.0)
    result = (img - A_light) / transmission[:, :, np.newaxis] + A_light
    return np.clip(result * 255, 0, 255).astype(np.uint8)

def apply_histogram_matching(image, reference_image):
    matched = match_histograms(image, reference_image, channel_axis=-1)
    return np.clip(matched, 0, 255).astype(np.uint8)

def get_reference_image(category, mvtec_root):
    train_dir = os.path.join(mvtec_root, category, "train", "good")
    files = sorted(os.listdir(train_dir))
    brightnesses = [np.mean(cv2.imread(os.path.join(train_dir, f))) for f in files]
    median_idx = np.argsort(brightnesses)[len(brightnesses) // 2]
    return cv2.cvtColor(cv2.imread(os.path.join(train_dir, files[median_idx])), cv2.COLOR_BGR2RGB)

def load_test_images(category, mvtec_root):
    test_dir = os.path.join(mvtec_root, category, "test")
    images, labels = [], []
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
    return images, labels

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

_dm = MVTecAD(root=MVTEC_ROOT, category="bottle", train_batch_size=1, eval_batch_size=1)
_dm.setup("fit")
import dataclasses
_item = _dm.train_data[0]
INPUT_SIZE = (getattr(_item, "image").shape[-2], getattr(_item, "image").shape[-1]) if dataclasses.is_dataclass(_item) and hasattr(getattr(_item, "image", None), 'shape') else (256, 256)
del _dm
print(f"Input size: {INPUT_SIZE}")

# %% [markdown]
# ## Cell 4: Experiment Variants

# %%
EXPERIMENT_VARIANTS = []
for sev in ["mild", "moderate", "severe"]:
    EXPERIMENT_VARIANTS.append(("sensor_noise", sev, "NLM"))
    EXPERIMENT_VARIANTS.append(("sensor_noise", sev, "HistMatch"))
    EXPERIMENT_VARIANTS.append(("fog_haze", sev, "DarkChannelPrior"))
    EXPERIMENT_VARIANTS.append(("fog_haze", sev, "HistMatch"))

EXPERIMENT_VARIANTS.append(("combined", "single", "Denoise_then_CLAHE"))
EXPERIMENT_VARIANTS.append(("combined", "single", "CLAHE_then_Denoise"))
EXPERIMENT_VARIANTS.append(("combined", "single", "HistMatch"))

print(f"Total variants: {len(EXPERIMENT_VARIANTS)}")

# %% [markdown]
# ## Cell 5: Run Experiments

# %%
os.makedirs("/kaggle/working/preprocessing", exist_ok=True)
results = []
start_time = time.time()
total_combos = len(CATEGORIES) * len(SEEDS)
combo_count = 0
ref_images = {cat: get_reference_image(cat, MVTEC_ROOT) for cat in CATEGORIES}

for category in CATEGORIES:
    for seed in SEEDS:
        combo_count += 1
        L.seed_everything(seed)
        print(f"\n{'='*70}")
        print(f"[{combo_count}/{total_combos}] PaDiM | {category} | seed={seed}")

        try:
            model = Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"])
            datamodule = MVTecAD(root=MVTEC_ROOT, category=category,
                                 train_batch_size=32, eval_batch_size=32)
            engine = Engine(max_epochs=1)
            engine.fit(model=model, datamodule=datamodule)

            test_images, test_labels = load_test_images(category, MVTEC_ROOT)
            ref_img = ref_images[category]

            for ctype, sev, preprocess_name in EXPERIMENT_VARIANTS:
                corrupted = [apply_corruption(img, ctype, sev, config, seed=seed+i)
                             for i, img in enumerate(test_images)]

                if preprocess_name == "NLM":
                    processed = [apply_nlm_denoise(img) for img in corrupted]
                elif preprocess_name == "DarkChannelPrior":
                    processed = [apply_dark_channel_prior_dehaze(img) for img in corrupted]
                elif preprocess_name == "HistMatch":
                    processed = [apply_histogram_matching(img, ref_img) for img in corrupted]
                elif preprocess_name == "Denoise_then_CLAHE":
                    processed = [apply_clahe(apply_nlm_denoise(img)) for img in corrupted]
                elif preprocess_name == "CLAHE_then_Denoise":
                    processed = [apply_nlm_denoise(apply_clahe(img)) for img in corrupted]
                else:
                    raise ValueError(f"Unknown: {preprocess_name}")

                scores = get_anomaly_scores(model, processed, target_size=INPUT_SIZE)
                auroc = compute_auroc(test_labels, scores)
                results.append({"model": "PaDiM", "category": category, "seed": seed,
                                "corruption_type": ctype, "severity": sev,
                                "preprocessing": preprocess_name, "image_AUROC": auroc})
                print(f"  {ctype}/{sev}→{preprocess_name}: {auroc:.4f}")

        except Exception as e:
            print(f"  ❌ FAILED: {e}")

        try: del model, engine
        except: pass
        torch.cuda.empty_cache()
        import gc; gc.collect()

        elapsed = time.time() - start_time
        eta = (elapsed / combo_count) * (total_combos - combo_count)
        print(f"  Elapsed: {elapsed/3600:.1f}h | ETA: {eta/3600:.1f}h")
        pd.DataFrame(results).to_csv(
            "/kaggle/working/preprocessing/padim_preprocess_noise_fog_WIP.csv", index=False)

# %% [markdown]
# ## Cell 6: Save

# %%
df = pd.DataFrame(results)
df.to_csv("/kaggle/working/preprocessing/padim_preprocess_noise_fog.csv", index=False)
pivot = df.groupby(["corruption_type", "severity", "preprocessing"])["image_AUROC"].agg(["mean", "std"]).round(4)
print("\nPaDiM Preprocessing Results (noise/fog/combined)")
print("="*70)
print(pivot.to_string())
print(f"\nWall time: {(time.time()-start_time)/3600:.2f}h")
