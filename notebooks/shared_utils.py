"""
Shared utility functions for the Anomaly Detection pipeline.
Copy this file's contents into Cell 2 of each Kaggle notebook,
or upload as a Kaggle dataset utility.

All corruptions: input/output = uint8 HWC RGB numpy array
All preprocessing: input/output = uint8 HWC RGB numpy array
"""

import numpy as np
import cv2
import albumentations as A
from skimage.restoration import wiener, denoise_nl_means, estimate_sigma
from skimage.exposure import match_histograms
from skimage import img_as_float, img_as_ubyte
from scipy.ndimage import minimum_filter


# ============================================================
# CORRUPTION FUNCTIONS
# ============================================================

def apply_low_light(image: np.ndarray, gamma: float, seed: int = 42) -> np.ndarray:
    """Simulate low-light via gamma correction + Poisson noise."""
    rng = np.random.default_rng(seed)
    img_float = image.astype(np.float64) / 255.0
    img_dark = np.power(img_float, 1.0 / gamma)
    photon_scale = 50.0
    img_noisy = rng.poisson(np.clip(img_dark * photon_scale, 0, None)) / photon_scale
    return np.clip(img_noisy * 255, 0, 255).astype(np.uint8)


def apply_gaussian_blur(image: np.ndarray, sigma: float, kernel_size: int) -> np.ndarray:
    """Apply Gaussian blur via albumentations."""
    t = A.GaussianBlur(blur_limit=(kernel_size, kernel_size),
                       sigma_limit=(sigma, sigma), p=1.0)
    return t(image=image)["image"]


def apply_motion_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply directional motion blur via albumentations."""
    t = A.MotionBlur(blur_limit=(kernel_size, kernel_size), p=1.0)
    return t(image=image)["image"]


def apply_sensor_noise(image: np.ndarray, gauss_var: float, seed: int = 42) -> np.ndarray:
    """Apply Gaussian noise + salt-and-pepper noise."""
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


def apply_fog(image: np.ndarray, fog_coef_lower: float,
              fog_coef_upper: float, alpha_coef: float = 0.1) -> np.ndarray:
    """Apply fog/haze overlay via albumentations RandomFog (NOT imgaug)."""
    t = A.RandomFog(fog_coef_lower=fog_coef_lower,
                    fog_coef_upper=fog_coef_upper,
                    alpha_coef=alpha_coef, p=1.0)
    return t(image=image)["image"]


def apply_combined(image: np.ndarray, seed: int = 42) -> np.ndarray:
    """Combined: Low-light (moderate) + Sensor noise (moderate)."""
    img = apply_low_light(image, gamma=0.35, seed=seed)
    return apply_sensor_noise(img, gauss_var=0.05, seed=seed + 1)


def apply_corruption(image: np.ndarray, ctype: str, severity: str,
                     config: dict, seed: int = 42) -> np.ndarray:
    """Unified corruption interface. Dispatches to specific function."""
    params = config["corruptions"][ctype][severity]
    if ctype == "low_light":
        return apply_low_light(image, params["gamma"], seed)
    elif ctype == "gaussian_blur":
        return apply_gaussian_blur(image, params["sigma"], params["kernel_size"])
    elif ctype == "motion_blur":
        return apply_motion_blur(image, params["kernel_size"])
    elif ctype == "sensor_noise":
        return apply_sensor_noise(image, params["gauss_var"], seed)
    elif ctype == "fog_haze":
        return apply_fog(image, params["fog_coef_lower"],
                         params["fog_coef_upper"], params.get("alpha_coef", 0.1))
    elif ctype == "combined":
        return apply_combined(image, seed)
    else:
        raise ValueError(f"Unknown corruption type: {ctype}")


# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def apply_clahe(image: np.ndarray, clip_limit: float = 3.0,
                tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """CLAHE on L channel in LAB color space. Target: low-light."""
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    lab_out = cv2.merge((cl, a, b))
    bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)


def apply_wiener_deconv(image: np.ndarray, sigma: float,
                        kernel_size: int, balance: float = 0.1) -> np.ndarray:
    """Wiener deconvolution with KNOWN PSF. Target: Gaussian blur.
    Only valid because blur kernel is known by synthetic construction."""
    img_float = img_as_float(image)
    ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    psf /= psf.sum()
    result = np.zeros_like(img_float)
    for c in range(3):
        result[:, :, c] = wiener(img_float[:, :, c], psf, balance)
    return img_as_ubyte(np.clip(result, 0, 1))


def apply_unsharp_masking(image: np.ndarray, sigma: float = 2.0,
                          strength: float = 1.5) -> np.ndarray:
    """Unsharp masking. Target: motion blur (DeblurGAN-v2 fallback)."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_nlm_denoise(image: np.ndarray, patch_size: int = 7,
                      patch_distance: int = 11) -> np.ndarray:
    """Non-Local Means denoising. Target: sensor noise."""
    img_float = image.astype(np.float64) / 255.0
    sigma_est = np.mean(estimate_sigma(img_float, channel_axis=-1))
    denoised = denoise_nl_means(
        img_float, h=1.15 * sigma_est,
        patch_size=patch_size, patch_distance=patch_distance,
        fast_mode=True, channel_axis=-1)
    return np.clip(denoised * 255, 0, 255).astype(np.uint8)


def apply_bm3d_denoise(image: np.ndarray, sigma_psd: float = None) -> np.ndarray:
    """BM3D denoising. Target: sensor noise (quality ceiling comparison).
    WARNING: Slow (~2s per image). Consider running only at severe level."""
    import bm3d as bm3d_lib
    img_float = image.astype(np.float64) / 255.0
    if sigma_psd is None:
        sigma_psd = np.mean(estimate_sigma(img_float, channel_axis=-1))
    denoised = bm3d_lib.bm3d(img_float, sigma_psd=sigma_psd)
    return np.clip(denoised * 255, 0, 255).astype(np.uint8)


def apply_dark_channel_prior_dehaze(image: np.ndarray, omega: float = 0.95,
                                    patch_size: int = 15) -> np.ndarray:
    """Dark channel prior dehazing (He et al., 2009). Target: fog/haze.
    Dependency-free alternative to AOD-Net."""
    img = image.astype(np.float64) / 255.0
    dark = np.min(img, axis=2)
    dark_ch = minimum_filter(dark, size=patch_size)
    flat = dark_ch.ravel()
    top_idx = np.argsort(flat)[-max(1, int(0.001 * len(flat))):]
    atm_light = np.max(img.reshape(-1, 3)[top_idx], axis=0)
    normed = img / (atm_light + 1e-6)
    normed_dark = minimum_filter(np.min(normed, axis=2), size=patch_size)
    transmission = np.clip(1 - omega * normed_dark, 0.1, 1.0)
    result = (img - atm_light) / transmission[:, :, np.newaxis] + atm_light
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def apply_histogram_matching(image: np.ndarray,
                             reference_image: np.ndarray) -> np.ndarray:
    """Naive baseline for ALL degradation types."""
    matched = match_histograms(image, reference_image, channel_axis=-1)
    return np.clip(matched, 0, 255).astype(np.uint8)


def get_reference_image(category: str, mvtec_root: str) -> np.ndarray:
    """Get median-brightness training image for histogram matching reference."""
    import os
    train_dir = os.path.join(mvtec_root, category, "train", "good")
    files = sorted(os.listdir(train_dir))
    brightnesses = []
    for f in files:
        img = cv2.imread(os.path.join(train_dir, f))
        brightnesses.append(np.mean(img))
    median_idx = np.argsort(brightnesses)[len(brightnesses) // 2]
    ref = cv2.imread(os.path.join(train_dir, files[median_idx]))
    return cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(labels: np.ndarray, scores: np.ndarray,
                    normal_train_scores: np.ndarray) -> dict:
    """Compute AUROC, FNR, FPR at 95th percentile threshold."""
    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(labels, scores)
    threshold = np.percentile(normal_train_scores, 95)
    preds = (scores >= threshold).astype(int)
    total_anom = np.sum(labels == 1)
    total_norm = np.sum(labels == 0)
    fnr = np.sum((preds == 0) & (labels == 1)) / total_anom if total_anom > 0 else 0.0
    fpr = np.sum((preds == 1) & (labels == 0)) / total_norm if total_norm > 0 else 0.0
    return {"image_AUROC": auroc, "FNR": fnr, "FPR": fpr, "threshold": threshold}


# ============================================================
# CORRUPTION VARIANT LIST
# ============================================================

def get_corruption_variants():
    """Return list of (corruption_type, severity) tuples for all 16 variants."""
    variants = []
    for ctype in ["low_light", "gaussian_blur", "motion_blur", "sensor_noise", "fog_haze"]:
        for sev in ["mild", "moderate", "severe"]:
            variants.append((ctype, sev))
    variants.append(("combined", "single"))
    return variants  # 16 total


# ============================================================
# PREPROCESSING MAP BUILDER
# ============================================================

def get_preprocessing_map(config: dict, mvtec_root: str):
    """
    Returns dict mapping corruption_type -> {preprocess_name: callable}.
    Each callable takes (image, category, severity) and returns preprocessed image.
    """
    return {
        "low_light": {
            "CLAHE": lambda img, cat, sev: apply_clahe(img),
            "HistMatch": lambda img, cat, sev: apply_histogram_matching(
                img, get_reference_image(cat, mvtec_root)),
        },
        "gaussian_blur": {
            "Wiener": lambda img, cat, sev: apply_wiener_deconv(
                img,
                config["corruptions"]["gaussian_blur"][sev]["sigma"],
                config["corruptions"]["gaussian_blur"][sev]["kernel_size"]
            ),
            "HistMatch": lambda img, cat, sev: apply_histogram_matching(
                img, get_reference_image(cat, mvtec_root)),
        },
        "motion_blur": {
            "UnsharpMask": lambda img, cat, sev: apply_unsharp_masking(img),
            "HistMatch": lambda img, cat, sev: apply_histogram_matching(
                img, get_reference_image(cat, mvtec_root)),
        },
        "sensor_noise": {
            "NLM": lambda img, cat, sev: apply_nlm_denoise(img),
            "BM3D": lambda img, cat, sev: apply_bm3d_denoise(img),
            "HistMatch": lambda img, cat, sev: apply_histogram_matching(
                img, get_reference_image(cat, mvtec_root)),
        },
        "fog_haze": {
            "DarkChannelPrior": lambda img, cat, sev: apply_dark_channel_prior_dehaze(img),
            "HistMatch": lambda img, cat, sev: apply_histogram_matching(
                img, get_reference_image(cat, mvtec_root)),
        },
        "combined": {
            "Denoise_then_CLAHE": lambda img, cat, sev: apply_clahe(apply_nlm_denoise(img)),
            "CLAHE_then_Denoise": lambda img, cat, sev: apply_nlm_denoise(apply_clahe(img)),
            "HistMatch": lambda img, cat, sev: apply_histogram_matching(
                img, get_reference_image(cat, mvtec_root)),
        },
    }
