# %% [markdown]
# # NB-9: VisA Generalization (Stretch Goal)
# **GPU**: ON | **Internet**: On | **Time**: ~4-6 hrs
#
# **Input Data Required**:
# - VisA dataset (add from Kaggle Datasets — search "visa anomaly")
# - NB-0 output (`experiment_config.json`)
#
# **Purpose**: Verify that findings from MVTec-AD generalize to a different
# industrial anomaly dataset. Uses a subset of VisA categories with
# the best-performing corruption/preprocessing pairs from the main experiments.

# %% [markdown]
# ## Cell 1: Install

# %%
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
from skimage.exposure import match_histograms
from anomalib.engine import Engine
from anomalib.models import Patchcore

CONFIG_PATH = "/kaggle/input/00-setup-and-verify/experiment_config.json"  # ADJUST
with open(CONFIG_PATH) as f:
    config = json.load(f)

SEEDS = config["seeds"]

# VisA categories — adjust based on what's available in the dataset
# Common VisA categories: candle, capsules, cashew, chewinggum, fryum,
# macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum
VISA_CATEGORIES = ["candle", "capsules", "cashew", "chewinggum", "fryum", "pcb1"]

# ⚠️ ADJUST this to match your VisA dataset path on Kaggle
VISA_ROOT = "/kaggle/input/visa-anomaly-detection"

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Discover actual path
if os.path.isdir(VISA_ROOT):
    print(f"VisA root: {VISA_ROOT}")
    print(f"Contents: {os.listdir(VISA_ROOT)[:10]}")
else:
    print(f"⚠️ VisA not found at {VISA_ROOT}")
    print("Available datasets:")
    for d in os.listdir("/kaggle/input"):
        print(f"  /kaggle/input/{d}/")

# %% [markdown]
# ## Cell 3: Functions

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

def apply_clahe(image, clip_limit=3.0, tile_grid_size=(8, 8)):
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)

def apply_histogram_matching(image, reference_image):
    matched = match_histograms(image, reference_image, channel_axis=-1)
    return np.clip(matched, 0, 255).astype(np.uint8)

def load_test_images(category, visa_root):
    """Load VisA test images. Adjust paths based on VisA directory structure."""
    test_dir = os.path.join(visa_root, category, "test")
    if not os.path.isdir(test_dir):
        # Try alternative VisA structures
        for alt in [os.path.join(visa_root, category, "Data", "Images", "Anomaly"),
                    os.path.join(visa_root, category)]:
            if os.path.isdir(alt):
                test_dir = alt
                break

    images, labels = [], []
    for defect_type in sorted(os.listdir(test_dir)):
        defect_dir = os.path.join(test_dir, defect_type)
        if not os.path.isdir(defect_dir):
            # Might be a flat directory structure
            if defect_type.lower().endswith((".png", ".jpg", ".jpeg")):
                img = cv2.imread(os.path.join(test_dir, defect_type))
                if img is not None:
                    images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    labels.append(1)  # Assume anomalous in test
            continue

        label = 0 if defect_type.lower() in ("good", "normal") else 1
        for img_file in sorted(os.listdir(defect_dir)):
            if not img_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")): continue
            img = cv2.imread(os.path.join(defect_dir, img_file))
            if img is None: continue
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            labels.append(label)

    return images, labels

def get_reference_image(category, visa_root):
    train_dir = os.path.join(visa_root, category, "train", "good")
    if not os.path.isdir(train_dir):
        train_dir = os.path.join(visa_root, category, "train")
    files = sorted([f for f in os.listdir(train_dir) if f.lower().endswith((".png", ".jpg"))])
    if not files:
        return np.ones((256, 256, 3), dtype=np.uint8) * 128
    mid = len(files) // 2
    return cv2.cvtColor(cv2.imread(os.path.join(train_dir, files[mid])), cv2.COLOR_BGR2RGB)

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

# %% [markdown]
# ## Cell 4: Run Generalization Experiments
# Test top corruption/preprocessing pairs from MVTec results on VisA.

# %%
# Key experiments to replicate on VisA (best pairs from MVTec)
EXPERIMENTS = [
    # (corruption_type, severity, preprocessing_or_None)
    ("clean", "none", None),
    ("low_light", "severe", None),
    ("low_light", "severe", "CLAHE"),
    ("gaussian_blur", "severe", None),
    ("sensor_noise", "severe", None),
]

os.makedirs("/kaggle/working/visa", exist_ok=True)
results = []
start_time = time.time()

for category in VISA_CATEGORIES:
    for seed in SEEDS[:1]:  # Single seed for generalization (save time)
        L.seed_everything(seed)
        print(f"\n{'='*70}")
        print(f"PatchCore | VisA/{category} | seed={seed}")

        try:
            # Try Anomalib's built-in VisA support first
            try:
                from anomalib.data import Visa
                datamodule = Visa(root=VISA_ROOT, category=category,
                                  train_batch_size=32, eval_batch_size=32)
            except (ImportError, Exception):
                # Fall back to MVTecAD-style loading if VisA has compatible structure
                datamodule = MVTecAD(root=VISA_ROOT, category=category,
                                     train_batch_size=32, eval_batch_size=32)

            model = Patchcore(backbone="wide_resnet50_2", num_neighbors=9)
            engine = Engine(max_epochs=1)
            engine.fit(model=model, datamodule=datamodule)

            test_images, test_labels = load_test_images(category, VISA_ROOT)
            ref_img = get_reference_image(category, VISA_ROOT)
            print(f"  Test set: {len(test_images)} images ({sum(test_labels)} anomalous)")

            for ctype, sev, preprocess in EXPERIMENTS:
                if ctype == "clean":
                    input_images = test_images
                else:
                    params = config["corruptions"][ctype][sev]
                    if ctype == "low_light":
                        input_images = [apply_low_light(img, params["gamma"], seed+i)
                                       for i, img in enumerate(test_images)]
                    elif ctype == "gaussian_blur":
                        input_images = [apply_gaussian_blur(img, params["sigma"], params["kernel_size"])
                                       for img in test_images]
                    elif ctype == "sensor_noise":
                        input_images = [apply_sensor_noise(img, params["gauss_var"], seed+i)
                                       for i, img in enumerate(test_images)]

                if preprocess == "CLAHE":
                    input_images = [apply_clahe(img) for img in input_images]

                scores = get_anomaly_scores(model, input_images, target_size=(256, 256))
                auroc = compute_auroc(test_labels, scores)

                results.append({
                    "dataset": "VisA", "model": "PatchCore",
                    "category": category, "seed": seed,
                    "corruption_type": ctype, "severity": sev,
                    "preprocessing": preprocess or "none",
                    "image_AUROC": auroc,
                })
                print(f"  {ctype}/{sev} → {preprocess or 'none'}: {auroc:.4f}")

        except Exception as e:
            print(f"  ❌ FAILED: {e}")

        try: del model, engine
        except: pass
        torch.cuda.empty_cache()

# %% [markdown]
# ## Cell 5: Save & Compare

# %%
df = pd.DataFrame(results)
df.to_csv("/kaggle/working/visa/visa_generalization.csv", index=False)

pivot = df.groupby(["corruption_type", "severity", "preprocessing"])["image_AUROC"].agg(["mean", "std"]).round(4)
print("\nVisA Generalization Results")
print("="*70)
print(pivot.to_string())

print(f"\nWall time: {(time.time()-start_time)/3600:.2f}h")
print("\n✅ VisA generalization complete!")
print("📋 Add this output as input to NB-10 for final analysis.")
