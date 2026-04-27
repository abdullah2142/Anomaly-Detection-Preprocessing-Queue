# %% [markdown]
# # NB-3: Severity Calibration & Visual Verification
# **GPU**: Off | **Internet**: On (for pip) | **Time**: ~15 min
#
# **Input Data Required**:
# - MVTec-AD dataset
# - NB-0 output (for config)
#
# This notebook generates visual grids showing all corruption types
# at all severity levels. **YOU must visually inspect and approve** before
# proceeding to experiments.
#
# ⚠️ After approval, corruption parameters are **FROZEN**.

# %% [markdown]
# ## Cell 1: Setup

# %%
!pip install -q albumentationsx

import json, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as A

CONFIG_PATH = "/kaggle/input/00-setup-and-verify/experiment_config.json"  # ADJUST
with open(CONFIG_PATH) as f:
    config = json.load(f)

MVTEC_ROOT = config["mvtec_root"]

# %% [markdown]
# ## Cell 2: Define Corruption Functions

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

# %% [markdown]
# ## Cell 3: Generate Visual Grids

# %%
SAMPLE_CATS = ["bottle", "carpet", "metal_nut", "wood", "transistor"]
CTYPES = ["low_light", "gaussian_blur", "motion_blur", "sensor_noise", "fog_haze", "combined"]

os.makedirs("/kaggle/working/severity_grids", exist_ok=True)

for ctype in CTYPES:
    severities = list(config["corruptions"][ctype].keys())
    n_cols = len(severities) + 1

    fig, axes = plt.subplots(len(SAMPLE_CATS), n_cols,
                              figsize=(4*n_cols, 4*len(SAMPLE_CATS)))
    fig.suptitle(f"Corruption: {ctype}", fontsize=18, fontweight="bold", y=1.01)

    for row, cat in enumerate(SAMPLE_CATS):
        # Load sample image
        test_dir = os.path.join(MVTEC_ROOT, cat, "test", "good")
        if not os.path.isdir(test_dir):
            test_dir = os.path.join(MVTEC_ROOT, cat, "train", "good")
        img_file = sorted(os.listdir(test_dir))[0]
        img = cv2.imread(os.path.join(test_dir, img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Clean
        ax = axes[row, 0] if len(SAMPLE_CATS) > 1 else axes[0]
        ax.imshow(img)
        ax.set_title(f"{cat}\n(clean)", fontsize=10)
        ax.axis("off")

        # Each severity
        for col, sev in enumerate(severities):
            corrupted = apply_corruption(img, ctype, sev, config)
            ax = axes[row, col+1] if len(SAMPLE_CATS) > 1 else axes[col+1]
            ax.imshow(corrupted)
            ax.set_title(f"{sev}", fontsize=10)
            ax.axis("off")

    plt.tight_layout()
    save_path = f"/kaggle/working/severity_grids/{ctype}_grid.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"✅ Saved: {save_path}")

# %% [markdown]
# ## Cell 4: Verification Checklist
# **🧑 HUMAN REVIEW REQUIRED**

# %%
print("""
╔══════════════════════════════════════════════════════════════════╗
║                    HUMAN VERIFICATION REQUIRED                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  For EACH corruption type grid above, verify:                    ║
║                                                                  ║
║  ✅ 1. Severity levels look perceptually proportional            ║
║        (mild < moderate < severe, clear visual progression)      ║
║                                                                  ║
║  ✅ 2. Objects remain RECOGNIZABLE at severe level               ║
║                                                                  ║
║  ⚠️  3. COMBINED case specifically:                              ║
║        - Is the object still identifiable?                       ║
║        - If NOT → reduce gauss_var from 0.05 to 0.03            ║
║                                                                  ║
║  ✅ 4. No implementation bugs                                    ║
║        (no black images, no color channel swaps)                 ║
║                                                                  ║
║  ══════════════════════════════════════════════════════════════   ║
║  After you approve:                                              ║
║  → Corruption parameters are LOCKED                              ║
║  → Click "Save & Run All" to commit                              ║
║  → Proceed to NB-4 (Degradation experiments)                     ║
╚══════════════════════════════════════════════════════════════════╝
""")
