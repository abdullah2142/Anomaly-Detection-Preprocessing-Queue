import os
import sys

# ---------------------------------------------------------------------------
# Fix: Disable rich/tqdm progress bars that cause RecursionError on Kaggle.
# rich's console proxy enters an infinite loop on Jupyter output streams.
# Reference: https://github.com/openvinotoolkit/anomalib/issues
# ---------------------------------------------------------------------------
os.environ["ANOMALIB_USE_RICH"] = "0"
os.environ["RICH_NO_THEME"] = "1"
sys.setrecursionlimit(5000)  # Guard against any residual recursion

import time
import json
import gc
import numpy as np
import pandas as pd
import cv2
import albumentations as A
import torch
from torch.utils.data import Dataset
from PIL import Image

# ---------------------------------------------------------------------------
# 0. Global Setup & Timeout Logic
# ---------------------------------------------------------------------------
START_TIME = time.time()
TIMEOUT_SECONDS = 11.5 * 3600  # 11.5 hours
OUTPUT_FILE = "results/padim_final.csv"
PARTIAL_FILE = "results/padim_partial.csv"

print(f"Script started at {time.ctime(START_TIME)}")
print(f"Graceful timeout set to {TIMEOUT_SECONDS / 3600:.1f} hours.")

def check_timeout():
    elapsed = time.time() - START_TIME
    if elapsed > TIMEOUT_SECONDS:
        print(f"\n" + "!"*60)
        print(f"TIMEOUT REACHED ({elapsed/3600:.1f}h). Exiting gracefully.")
        print("!"*60)
        save_results()
        sys.exit(0)

def save_results():
    if 'all_results' in globals() and all_results:
        os.makedirs("results", exist_ok=True)
        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT_FILE, index=False)
        df.to_csv(PARTIAL_FILE, index=False)

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 1. Dependency Check
# ---------------------------------------------------------------------------
def ensure_dependencies():
    import subprocess
    packages = ["anomalib", "lightning", "albumentationsx", "scikit-image", "opencv-python-headless"]
    for package in packages:
        try:
            check_name = "cv2" if package == "opencv-python-headless" else (package.replace("-", "_") if package != "albumentationsx" else "albumentations")
            __import__(check_name)
        except ImportError:
            print(f"Installing missing dependency: {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

ensure_dependencies()

# Now safe to import
import lightning as L
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Padim

# ---------------------------------------------------------------------------
# 2. Embedded Corruption & Rescue Functions
# ---------------------------------------------------------------------------

def apply_low_light(image, gamma, seed=42):
    rng = np.random.default_rng(seed)
    img_float = image.astype(np.float64) / 255.0
    img_dark = np.power(img_float, 1.0 / gamma)
    photon_scale = 50.0
    img_noisy = rng.poisson(np.clip(img_dark * photon_scale, 0, None)) / photon_scale
    return np.clip(img_noisy * 255, 0, 255).astype(np.uint8)

def apply_gaussian_blur(image, sigma, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)

def apply_motion_blur(image, kernel_size):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    return cv2.filter2D(image, -1, kernel)

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
    t = A.RandomFog(fog_coef_range=(fog_coef_lower, fog_coef_upper),
                    alpha_coef=alpha_coef, p=1.0)
    return t(image=image)["image"]

def apply_corruption(image, ctype, severity, config, seed=42):
    params = config["corruptions"][ctype][severity]
    if ctype == "low_light": return apply_low_light(image, params["gamma"], seed)
    elif ctype == "gaussian_blur": return apply_gaussian_blur(image, params["sigma"], params["kernel_size"])
    elif ctype == "motion_blur": return apply_motion_blur(image, params["kernel_size"])
    elif ctype == "sensor_noise": return apply_sensor_noise(image, params["gauss_var"], seed)
    elif ctype == "fog_haze": return apply_fog(image, params["fog_coef_lower"], params["fog_coef_upper"], params.get("alpha_coef", 0.1))
    else: raise ValueError(f"Unknown corruption type: {ctype}")

def apply_clahe(image: np.ndarray, clip_limit: float = 3.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    lab_enhanced = cv2.merge((cl, a, b))
    bgr_out = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)

def apply_wiener_deconv(image: np.ndarray, sigma: float, kernel_size: int, balance: float = 0.1) -> np.ndarray:
    from skimage.restoration import wiener
    from skimage import img_as_float, img_as_ubyte
    img_float = img_as_float(image)
    ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    psf /= psf.sum()
    result = np.zeros_like(img_float)
    for c in range(3):
        result[:, :, c] = wiener(img_float[:, :, c], psf, balance)
    return img_as_ubyte(np.clip(result, 0, 1))

def apply_motion_wiener_deconv(image: np.ndarray, kernel_size: int, balance: float = 0.1) -> np.ndarray:
    from skimage.restoration import wiener
    from skimage import img_as_float, img_as_ubyte
    img_float = img_as_float(image)
    psf = np.zeros((kernel_size, kernel_size))
    psf[kernel_size // 2, :] = 1.0 / kernel_size
    result = np.zeros_like(img_float)
    for c in range(3):
        result[:, :, c] = wiener(img_float[:, :, c], psf, balance)
    return img_as_ubyte(np.clip(result, 0, 1))

def apply_nlm_denoise(image: np.ndarray, patch_size: int = 7, patch_distance: int = 11) -> np.ndarray:
    from skimage.restoration import denoise_nl_means, estimate_sigma
    img_float = image.astype(np.float64) / 255.0
    sigma_est = np.mean(estimate_sigma(img_float, channel_axis=-1))
    denoised = denoise_nl_means(img_float, h=1.15 * sigma_est, patch_size=patch_size, patch_distance=patch_distance, fast_mode=True, channel_axis=-1)
    return np.clip(denoised * 255, 0, 255).astype(np.uint8)

def apply_retinex(image: np.ndarray) -> np.ndarray:
    img_float = image.astype(np.float64) + 1.0
    blurred = cv2.GaussianBlur(img_float, (0, 0), 30)
    log_retinex = np.log10(img_float) - np.log10(blurred + 1.0)
    for i in range(3):
        log_retinex[:,:,i] = (log_retinex[:,:,i] - np.min(log_retinex[:,:,i])) / (np.max(log_retinex[:,:,i]) - np.min(log_retinex[:,:,i])) * 255
    return log_retinex.astype(np.uint8)

def apply_dark_channel_prior_dehaze(image: np.ndarray, omega: float = 0.95, patch_size: int = 15) -> np.ndarray:
    img = image.astype(np.float64) / 255.0
    dark = np.min(img, axis=2)
    from scipy.ndimage import minimum_filter
    dark_channel = minimum_filter(dark, size=patch_size)
    flat_dark = dark_channel.ravel()
    top_indices = np.argsort(flat_dark)[-max(1, int(0.001 * len(flat_dark))):]
    A = np.max(img.reshape(-1, 3)[top_indices], axis=0)
    normed = img / (A + 1e-6)
    normed_dark = np.min(normed, axis=2)
    normed_dark_filtered = minimum_filter(normed_dark, size=patch_size)
    transmission = np.clip(1 - omega * normed_dark_filtered, 0.1, 1.0)
    result = (img - A) / transmission[:, :, np.newaxis] + A
    return np.clip(result * 255, 0, 255).astype(np.uint8)

RESCUE_MAP = {
    "low_light": [("CLAHE", apply_clahe), ("Retinex", apply_retinex)],
    "gaussian_blur": [("Wiener", lambda img: apply_wiener_deconv(img, sigma=25.0, kernel_size=281))],
    "motion_blur": [("Wiener (Motion PSF)", lambda img: apply_motion_wiener_deconv(img, kernel_size=151))],
    "sensor_noise": [("NLM Denoise", apply_nlm_denoise)],
    "fog_haze": [("Dehaze (Dark Channel)", apply_dark_channel_prior_dehaze)]
}

# ---------------------------------------------------------------------------
# 3. Load configuration
# ---------------------------------------------------------------------------
CONFIG_PATH = "/kaggle/input/notebooks/hasanmahmudabdullah/03-severity-calibration/experiment_config.json"
with open(CONFIG_PATH) as f: config = json.load(f)

MVTEC_ROOT = config["mvtec_root"]
CATEGORIES = config["categories"]
SEEDS = config["seeds"]

# ---------------------------------------------------------------------------
# 4. Corrupted Dataset Wrapper
# ---------------------------------------------------------------------------
class CorruptedDatasetWrapper(Dataset):
    def __init__(self, base_dataset, ctype, severity, config, rescue_func=None):
        self.base_dataset = base_dataset
        self.ctype = ctype
        self.severity = severity
        self.config = config
        self.rescue_func = rescue_func
    def __len__(self): return len(self.base_dataset)
    def __getattr__(self, name):
        # Transparently proxy any attribute Anomalib expects (collate_fn, transform, etc.)
        # to the underlying base dataset. This prevents AttributeError on Anomalib internals.
        return getattr(self.base_dataset, name)
    def __getitem__(self, idx):
        import dataclasses
        item = self.base_dataset[idx]

        # Anomalib v1.x returns an ImageItem dataclass, not a dict.
        if dataclasses.is_dataclass(item):
            image = item.image
        else:
            image = item["image"]

        if isinstance(image, torch.Tensor):
            img_np = image.permute(1, 2, 0).cpu().numpy()
            if img_np.max() <= 1.0: img_np = (img_np * 255).astype(np.uint8)
            else: img_np = img_np.astype(np.uint8)
        else:
            img_np = np.array(image).astype(np.uint8)

        corrupted = apply_corruption(img_np, self.ctype, self.severity, self.config, seed=42+idx)
        final_img = self.rescue_func(corrupted) if self.rescue_func else corrupted
        final_tensor = torch.from_numpy(final_img).permute(2, 0, 1).float() / 255.0

        if dataclasses.is_dataclass(item):
            return dataclasses.replace(item, image=final_tensor)
        else:
            item["image"] = final_tensor
            return item

# ---------------------------------------------------------------------------
# 5. Resume Logic
# ---------------------------------------------------------------------------
completed_keys = set()
all_results = []

class DisableCheckpointing(L.Callback):
    """Strips any ModelCheckpoint callbacks before training starts."""
    def setup(self, trainer, pl_module, stage):
        from lightning.pytorch.callbacks import ModelCheckpoint
        trainer.callbacks = [
            cb for cb in trainer.callbacks
            if not isinstance(cb, ModelCheckpoint)
        ]

def make_engine():
    """Engine configured to avoid:
    1. RecursionError from rich/tqdm on Kaggle (enable_progress_bar=False)
    2. ModelCheckpoint contradiction error (DisableCheckpointing callback)
    3. Heatmap spam (default_root_dir=/tmp)
    """
    return Engine(
        max_epochs=1,
        accelerator="auto",
        devices=1,
        default_root_dir="/tmp/anomalib",
        enable_progress_bar=False,
        callbacks=[DisableCheckpointing()]
    )

def safe_auroc(result_dict):
    """Extract AUROC from Anomalib result dict. Logs a warning if key not found."""
    for key in ["image_AUROC", "image_auroc", "auroc", "AUROC", "test_image_AUROC"]:
        if key in result_dict:
            return result_dict[key]
    print(f"  ⚠️  WARNING: AUROC key not found in results. Available keys: {list(result_dict.keys())}")
    return None  # None instead of 0 so it's visible in CSV
possible_resume_paths = [OUTPUT_FILE, PARTIAL_FILE, "/kaggle/input/anomaly-detection-results/padim_final.csv"]
for p in possible_resume_paths:
    if os.path.exists(p):
        try:
            old_df = pd.read_csv(p)
            all_results = old_df.to_dict('records')
            for _, row in old_df.iterrows(): completed_keys.add((row['category'], int(row['seed'])))
            print(f"Loaded {len(completed_keys)} completed category/seed pairs.")
            break
        except Exception as e: print(f"Could not load existing results: {e}")

# ---------------------------------------------------------------------------
# 6. Main Loop
# ---------------------------------------------------------------------------
os.makedirs("results", exist_ok=True)

for category in CATEGORIES:
    for seed in SEEDS:
        check_timeout()
        if (category, int(seed)) in completed_keys:
            print(f"⏩ Skipping {category} (Seed {seed})")
            continue
            
        print(f"\n" + "="*60 + f"\nCATEGORY: {category.upper()} | SEED: {seed}\n" + "="*60)
        L.seed_everything(seed)
        
        # Create engine without ModelCheckpoint callbacks
        engine = make_engine()
        model = Padim(backbone="wide_resnet50_2", layers=["layer1", "layer2", "layer3"], n_features=100)
        datamodule = MVTecAD(root=MVTEC_ROOT, category=category, train_batch_size=32, eval_batch_size=32)
        
        try:
            print(f"[PHASE 1] Training Baseline...")
            engine.fit(model=model, datamodule=datamodule)
            res = engine.test(model=model, datamodule=datamodule)
            clean_auroc = safe_auroc(res[0])
            all_results.append({"model": "PaDiM", "category": category, "seed": seed, "phase": "baseline", "ctype": "clean", "severity": "none", "rescue": "none", "image_AUROC": clean_auroc})
            save_results()
            
            for ctype, severities in config["corruptions"].items():
                for sev in severities.keys():
                    print(f"[PHASE 2] Degradation: {ctype} ({sev})")
                    
                    # Get a clean base test dataset from a fresh datamodule
                    dm_base = MVTecAD(root=MVTEC_ROOT, category=category, train_batch_size=32, eval_batch_size=32)
                    dm_base.setup(stage="test")
                    clean_base = dm_base.test_data
                    
                    # Build corrupted DataLoader directly — bypasses Anomalib's internal
                    # caching which ignores datamodule.test_data reassignment
                    from torch.utils.data import DataLoader
                    deg_loader = DataLoader(
                        CorruptedDatasetWrapper(clean_base, ctype, sev, config),
                        batch_size=32, num_workers=2, collate_fn=clean_base.collate_fn
                    )
                    res = engine.test(model=model, dataloaders=deg_loader)
                    deg_auroc = safe_auroc(res[0])
                    all_results.append({"model": "PaDiM", "category": category, "seed": seed, "phase": "degradation", "ctype": ctype, "severity": sev, "rescue": "none", "image_AUROC": deg_auroc})
                    save_results()

                    if ctype in RESCUE_MAP and sev == "severe":
                        for r_name, r_func in RESCUE_MAP[ctype]:
                            print(f"[PHASE 3] Rescue: {ctype} ({sev}) + {r_name}")
                            dm_r = MVTecAD(root=MVTEC_ROOT, category=category, train_batch_size=32, eval_batch_size=32)
                            dm_r.setup(stage="test")
                            res_loader = DataLoader(
                                CorruptedDatasetWrapper(dm_r.test_data, ctype, sev, config, rescue_func=r_func),
                                batch_size=32, num_workers=2, collate_fn=dm_r.test_data.collate_fn
                            )
                            res = engine.test(model=model, dataloaders=res_loader)
                            res_auroc = safe_auroc(res[0])
                            all_results.append({"model": "PaDiM", "category": category, "seed": seed, "phase": "rescue", "ctype": ctype, "severity": sev, "rescue": r_name, "image_AUROC": res_auroc})
                            save_results()
        except Exception as e:
            print(f"❌ Error in {category}/{seed}: {e}")
        
        del model
        del engine
        cleanup_memory()

save_results()
print("\n" + "="*60 + "\nEXPERIMENT COMPLETE\n" + "="*60)
