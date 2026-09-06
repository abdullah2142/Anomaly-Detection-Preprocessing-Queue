# Limitations

## 1. Synthetic Corruption Gap

All five corruption types are synthetically generated using parameterized functions applied to clean MVTec-AD images. Real-world imaging degradation is more complex: manufacturing lighting variation involves spatial non-uniformity, reflections, and spectral shifts that a simple gamma-darkening model does not capture. Real sensor noise profiles are camera-specific and correlated across channels. Real fog and haze involve depth-dependent scattering that a uniform coefficient model approximates poorly. **Implication:** results represent a controlled lower bound on degradation diversity. Models may perform differently under real-world corruption even at matching perceptual severity.

## 2. Wiener PSF Specification

Wiener deconvolution uses the exact PSF used to generate the blur only at **severe** severity. At mild and moderate severities, the PSF parameters are matched to the corruption severity in the corrected codebase, but original published results used hardcoded severe-tier parameters at all severities. Real-world blur kernels are camera-, motion-, and scene-dependent and must be estimated blindly. Our severe-tier Wiener results (oracle PSF, mean Δ ≈ −12.5 pp) therefore represent an **upper bound** on deconvolution performance; blind deconvolution on unknown kernels is expected to be substantially worse.

## 3. Limited Model Diversity

Results are reported for two feature-embedding anomaly detectors sharing the same WideResNet-50-2 backbone. The findings may not generalize to:
- **Reconstruction-based methods** (autoencoders, diffusion models): score by pixel/patch reconstruction error and may respond differently to preprocessing that improves perceptual quality even if it introduces artifacts.
- **Vision-language models** (WinCLIP, AnomalyCLIP): use semantic embeddings which may be more robust to low-level corruption but less interpretable under augmented training.
- **Different backbones**: ResNet-18 or ViT-based extractors may produce different corruption sensitivity profiles.

## 4. Single Augmentation Probability

Augmented training uses aug_prob=0.50. This is a single operating point. The trade-off curve across aug_prob ∈ {0.1, 0.25, 0.5, 0.75, 1.0} is not measured. Lower values may recover more clean-image AUROC with comparable robustness gains; higher values may further improve robustness at greater clean-image cost.

## 5. Restricted Preprocessing Search Space

Only classical, parameter-light restoration methods are evaluated. Modern deep restoration (diffusion-based deblurring, transformer denoising, neural dehazing) is not included. These methods may produce feature-compatible restored images, but including them raises model-size, latency, and deployment complexity questions that are themselves open research problems for real-time inspection.

## 6. Dataset Diversity

While the primary quantitative results are derived from the 15 categories of MVTec-AD, we conducted a cross-dataset validation on a 4-category subset of the VisA dataset (candle, cashew, pcb1, pipe_fryum) to verify generalizability. The core phenomena identified on MVTec-AD hold on VisA. Under category-clustered testing, the clean-trained VisA rescue harm is significant across its 12 categories (PatchCore $p = 4.9\times10^{-4}$, PaDiM $p = 2.9\times10^{-3}$). The augmented VisA arm spans only 4 categories, where the smallest attainable two-sided p is 0.125; its gains (+17.8 / +13.9 pp) are therefore reported as descriptive corroboration of the MVTec-AD result rather than as independent significance.

## 7. Threshold-Free Metric Only

Only image-level AUROC is reported. AUROC is threshold-free and insensitive to operating point. Industrial deployment requires a detection threshold, and preprocessing can shift score distributions such that a previously calibrated threshold produces more false alarms even if AUROC is unchanged. FNR/FPR at a fixed percentile threshold are left as future work.

## 8. Computational Environment

Experiments were executed on Kaggle T4/P100 GPUs (16 GB VRAM) under a 12-hour session limit, which introduced specific practical constraints. For instance, PaDiM's memory footprint required explicitly capping the feature dimension (`n_features=100`) to prevent Out-Of-Memory (OOM) errors during covariance estimation across augmented datasets. This represents a realistic deployment constraint for high-resolution industrial inspection on edge devices or standard GPUs. Results on larger hardware with unconstrained memory or larger batch sizes may differ marginally.

## 9. AUROC Floor Saturation

23.1% of all benchmark rows (2,169 / 9,384) exhibit AUROC = 0.5000 exactly, indicating fully tied anomaly scores with zero ranking ability. This is concentrated at severe corruption: 52.2% of severe Gaussian blur rows hit the floor. Reported mean AUROC values are therefore censored at a floor of 0.5 and may understate the true severity of model collapse under extreme corruption. Wilcoxon tests on rescue deltas also have reduced effective N where both degraded and rescued AUROC are 0.5 (zero differences are silently dropped).

## 10. Train/Test Resolution Mismatch

Training-time augmentation (`prepare_augmented_train_data`) applies corruption to raw-resolution images (700–1024 px for MVTec-AD, ~1500 px for VisA) which are subsequently resized to 256×256 by the datamodule. Test-time corruption (`CorruptedDatasetWrapper`) applies corruption to images that have already been resized to 256×256. The same nominal corruption parameters therefore produce substantially different effective severity: e.g., σ=25 Gaussian blur at 1024 px downsampled to 256 px is effectively ~σ=6. Augmented models were trained on milder effective corruption than they are tested on, which means the reported robustness gains may partially reflect generalization beyond the training distribution rather than exact distribution matching.

## 11. Matched Corruption Distributions

Training augmentation draws from the same 5 corruption types × 3 severity levels used for test-time evaluation. The reported augmentation gains (+12.3 pp PatchCore, +10.1 pp PaDiM) therefore represent a **corruption-matched oracle upper bound** on augmentation effectiveness. A leave-one-corruption-out cross-validation would provide stronger evidence of generalization but is left as future work.

## 12. Undocumented Salt-and-Pepper Component in Sensor Noise

The `apply_sensor_noise` function adds 5% salt-and-pepper impulse noise on top of the Gaussian noise specified in `experiment_config.json`. This impulse component is not reflected in the configuration file's `gauss_var` parameter. NLM denoising is suboptimal for impulse noise (median filtering is standard), which means the modest harm of NLM rescue (−1 to −3 pp) may partly reflect a rescue-method mismatch rather than a fundamental incompatibility between denoising and anomaly detection.
