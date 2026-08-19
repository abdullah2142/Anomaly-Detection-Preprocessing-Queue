# Limitations

## 1. Synthetic Corruption Gap

All five corruption types are synthetically generated using parameterized functions applied to clean MVTec-AD images. Real-world imaging degradation is more complex: manufacturing lighting variation involves spatial non-uniformity, reflections, and spectral shifts that a simple gamma-darkening model does not capture. Real sensor noise profiles are camera-specific and correlated across channels. Real fog and haze involve depth-dependent scattering that a uniform coefficient model approximates poorly. **Implication:** results represent a controlled lower bound on degradation diversity. Models may perform differently under real-world corruption even at matching perceptual severity.

## 2. Known-Kernel Assumption for Wiener Deconvolution

Wiener deconvolution in this study uses the exact PSF used to generate the blur — a condition that does not hold in practice. Real-world blur kernels are camera-, motion-, and scene-dependent and must be estimated blindly. Our Wiener results therefore represent an **upper bound** on deconvolution performance; blind deconvolution on unknown kernels is expected to be substantially worse. Despite this favorable condition, Wiener deconvolution is still catastrophically harmful (mean Δ = −0.12 to −0.34 AUROC), strengthening the anti-deconvolution conclusion.

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

While the primary quantitative results are derived from the 15 categories of MVTec-AD, we conducted a cross-dataset validation on a 4-category subset of the VisA dataset (candle, cashew, pcb1, pipe_fryum) to verify generalizability. Wilcoxon signed-rank tests confirm that the core phenomena identified on MVTec-AD hold true on VisA with high statistical significance ($p < 0.001$): augmented training significantly improves robustness, and test-time rescue preprocessing remains net-harmful.

## 7. Threshold-Free Metric Only

Only image-level AUROC is reported. AUROC is threshold-free and insensitive to operating point. Industrial deployment requires a detection threshold, and preprocessing can shift score distributions such that a previously calibrated threshold produces more false alarms even if AUROC is unchanged. FNR/FPR at a fixed percentile threshold are left as future work.

## 8. Computational Environment

Experiments were executed on Kaggle T4/P100 GPUs (16 GB VRAM) under a 12-hour session limit, which introduced specific practical constraints. For instance, PaDiM's memory footprint required explicitly capping the feature dimension (`n_features=100`) to prevent Out-Of-Memory (OOM) errors during covariance estimation across augmented datasets. This represents a realistic deployment constraint for high-resolution industrial inspection on edge devices or standard GPUs. Results on larger hardware with unconstrained memory or larger batch sizes may differ marginally.
