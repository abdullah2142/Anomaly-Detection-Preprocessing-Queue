# %% [markdown]
# # NB-8: Training-Set Corruption Experiment
# **GPU**: ON | **Internet**: On | **Time**: ~6-8 hrs
#
# **Input Data Required**:
# - MVTec-AD dataset
# - NB-0 output (`experiment_config.json`)
#
# **Purpose**: Answer "Does preprocessing help when training data is ALSO corrupted?"
#
# Experiment uses low-light (moderate, gamma=0.35) only:
# - **Condition A**: Clean train → Clean test (baseline, from NB-1/NB-2)
# - **Condition B**: Corrupted train → Corrupted test (no preprocessing)
# - **Condition C**: Corrupted+CLAHE train → Corrupted+CLAHE test

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
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore, Padim

CONFIG_PATH = "/kaggle/input/00-setup-and-verify/experiment_config.json"  # ADJUST
with open(CONFIG_PATH) as f:
    config = json.load(f)

MVTEC_ROOT = config["mvtec_root"]
CATEGORIES = config["categories"]
SEEDS = config["seeds"]
print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## Cell 3: Corruption + Preprocessing Functions

# %%
def apply_low_light(image, gamma=0.35, seed=42):
    rng = np.random.default_rng(seed)
    img_float = image.astype(np.float64) / 255.0
    img_dark = np.power(img_float, 1.0 / gamma)
    img_noisy = rng.poisson(np.clip(img_dark * 50.0, 0, None)) / 50.0
    return np.clip(img_noisy * 255, 0, 255).astype(np.uint8)

def apply_clahe(image, clip_limit=3.0, tile_grid_size=(8, 8)):
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)

# %% [markdown]
# ## Cell 4: Helpers

# %%
def load_images_from_dir(directory):
    """Load all images from a directory, return list of uint8 RGB arrays."""
    images, paths = [], []
    for f in sorted(os.listdir(directory)):
        if not f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")): continue
        img = cv2.imread(os.path.join(directory, f))
        if img is None: continue
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        paths.append(os.path.join(directory, f))
    return images, paths

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

# Input size
_dm = MVTecAD(root=MVTEC_ROOT, category="bottle", train_batch_size=1, eval_batch_size=1)
_dm.setup("fit")
import dataclasses
_item = _dm.train_data[0]
INPUT_SIZE = (getattr(_item, "image").shape[-2], getattr(_item, "image").shape[-1]) if dataclasses.is_dataclass(_item) and hasattr(getattr(_item, "image", None), 'shape') else (256, 256)
del _dm
print(f"Input size: {INPUT_SIZE}")

# %% [markdown]
# ## Cell 5: Save corrupted training images to /kaggle/tmp

# %%
# We need to create corrupted versions of training data for Conditions B and C.
# Save to /kaggle/tmp (60GB scratch, not persisted but fine for one session).

import shutil

for category in CATEGORIES:
    train_dir = os.path.join(MVTEC_ROOT, category, "train", "good")
    images, paths = load_images_from_dir(train_dir)

    for condition in ["B", "C"]:
        out_dir = f"/kaggle/tmp/train_corrupted/{condition}/{category}/train/good"
        os.makedirs(out_dir, exist_ok=True)

        for i, (img, path) in enumerate(zip(images, paths)):
            # Condition B: corrupt only
            corrupted = apply_low_light(img, gamma=0.35, seed=42+i)

            if condition == "B":
                out_img = corrupted
            else:  # Condition C: corrupt + CLAHE
                out_img = apply_clahe(corrupted)

            fname = os.path.basename(path)
            cv2.imwrite(os.path.join(out_dir, fname),
                       cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

    # Also copy test directory structure for anomalib datamodule
    for condition in ["B", "C"]:
        test_src = os.path.join(MVTEC_ROOT, category, "test")
        test_dst = f"/kaggle/tmp/train_corrupted/{condition}/{category}/test"
        gt_src = os.path.join(MVTEC_ROOT, category, "ground_truth")
        gt_dst = f"/kaggle/tmp/train_corrupted/{condition}/{category}/ground_truth"

        if not os.path.exists(test_dst):
            shutil.copytree(test_src, test_dst)
        if os.path.exists(gt_src) and not os.path.exists(gt_dst):
            shutil.copytree(gt_src, gt_dst)

    print(f"✅ {category}: corrupted training data saved")

print(f"\nTotal tmp usage: {sum(os.path.getsize(os.path.join(dp,f)) for dp,_,fn in os.walk('/kaggle/tmp/train_corrupted') for f in fn)/1e9:.2f} GB")

# %% [markdown]
# ## Cell 6: Run Experiments

# %%
os.makedirs("/kaggle/working/training_corruption", exist_ok=True)
MODELS = {
    "PatchCore": lambda: Patchcore(backbone="wide_resnet50_2", num_neighbors=9),
    "PaDiM": lambda: Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"]),
}
CONDITIONS = {
    "B": {"train_root_template": "/kaggle/tmp/train_corrupted/B/{category}",
           "test_transform": "corrupt_only"},
    "C": {"train_root_template": "/kaggle/tmp/train_corrupted/C/{category}",
           "test_transform": "corrupt_and_clahe"},
}

results = []
start_time = time.time()

for model_name, model_fn in MODELS.items():
    for category in CATEGORIES:
        for seed in SEEDS:
            L.seed_everything(seed)
            test_images, test_labels = load_test_images(category, MVTEC_ROOT)

            for cond_name, cond_cfg in CONDITIONS.items():
                print(f"  {model_name} | {category} | seed={seed} | Cond {cond_name}")

                # Train on corrupted data
                corrupted_root = cond_cfg["train_root_template"].format(category=category)
                model = model_fn()

                # Use Anomalib datamodule pointing to corrupted training dir
                # We need the parent dir that contains the category folder
                corrupted_parent = os.path.dirname(corrupted_root)
                datamodule = MVTecAD(
                    root=corrupted_parent,
                    category=category,
                    train_batch_size=32, eval_batch_size=32,
                )
                engine = Engine(max_epochs=1)
                engine.fit(model=model, datamodule=datamodule)

                # Test on corrupted (+ optionally preprocessed) test data
                if cond_name == "B":
                    test_input = [apply_low_light(img, gamma=0.35, seed=42+i)
                                  for i, img in enumerate(test_images)]
                else:  # C
                    test_input = [apply_clahe(apply_low_light(img, gamma=0.35, seed=42+i))
                                  for i, img in enumerate(test_images)]

                scores = get_anomaly_scores(model, test_input, target_size=INPUT_SIZE)
                auroc = compute_auroc(test_labels, scores)

                results.append({
                    "model": model_name, "category": category, "seed": seed,
                    "condition": cond_name, "image_AUROC": auroc,
                })
                print(f"    AUROC={auroc:.4f}")

                del model, engine; torch.cuda.empty_cache()

            # Save WIP
            pd.DataFrame(results).to_csv(
                "/kaggle/working/training_corruption/training_corruption_WIP.csv", index=False)

# %% [markdown]
# ## Cell 7: Save & Summary

# %%
df = pd.DataFrame(results)
df.to_csv("/kaggle/working/training_corruption/training_corruption.csv", index=False)

pivot = df.groupby(["model", "condition"])["image_AUROC"].agg(["mean", "std"]).round(4)
print("\nTraining Corruption Results")
print("="*60)
print(pivot.to_string())
print(f"\nWall time: {(time.time()-start_time)/3600:.2f}h")

print("\nInterpretation:")
print("  Condition B (corrupt train + corrupt test, no preprocessing):")
print("    → Shows model adapts to corruption during training")
print("  Condition C (corrupt+CLAHE train + corrupt+CLAHE test):")
print("    → Shows if CLAHE helps even when model trains on corrupted data")
