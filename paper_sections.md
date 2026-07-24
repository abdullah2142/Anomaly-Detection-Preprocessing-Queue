# Paper Section Drafts — Robust IAD Under Adverse Imaging Conditions

> Use these as first drafts. Numbers are exact from the benchmark. Citations marked [REF:X] need DOIs inserted.
> Mechanistic explanations updated from Wilcoxon analysis (July 2026).

---

## 1. ABSTRACT (≤250 words)

State-of-the-art unsupervised anomaly detection models achieve near-perfect AUROC on standard benchmarks yet are evaluated exclusively under controlled studio conditions. In industrial deployment, cameras routinely capture images degraded by low illumination, motion and defocus blur, sensor noise, and environmental haze. Two intuitive mitigations exist: apply classical image restoration as a preprocessing step before inference (test-time rescue), or train models on corrupted images to build robustness directly (training-time augmentation). We conduct the first systematic 4-way comparison of these strategies across two leading feature-embedding anomaly detectors — PatchCore and PaDiM — on the full MVTec-AD benchmark (15 categories, 3 seeds, 5 corruption types, 3 severity levels, 6 rescue methods), yielding 6,120 controlled measurements. Our results reveal three findings. First, augmented training consistently improves robustness (+12.3 pp PatchCore, +10.1 pp PaDiM, both p < 0.001) by expanding the learned normality manifold to encompass corruption-domain features. Second, classical rescue preprocessing is net-harmful in all four experimental conditions (range: −4.6 to −14.9 pp, all p < 0.001) — a phenomenon we term the *preprocessing fallacy*: restoration algorithms produce a third distribution distinct from both the clean training set and naturally corrupted images, causing anomaly detectors to flag restoration artifacts as defects. Third, Wiener deconvolution collapses AUROC to near-random chance (−14 to −37 pp) by introducing spectral ringing that saturates the patch-level anomaly score, while CLAHE is the only conditionally safe method. These findings provide actionable deployment guidance: use augmented training; avoid test-time restoration except CLAHE for low-light PatchCore.

---

## 2. INTRODUCTION (~600–800 words)

### Draft

Industrial visual inspection has undergone a dramatic transition over the past five years. Unsupervised anomaly detection models — trained exclusively on defect-free images — now achieve image-level AUROC above 98% on MVTec-AD [REF:Bergmann2019], the field's primary benchmark. This performance has been interpreted as a signal that the core detection problem is largely solved, enabling rapid industrial adoption.

This interpretation rests on a critical assumption: that test-time images will resemble the controlled studio conditions under which benchmark images were captured. In practice, manufacturing environments violate this assumption routinely. Conveyor lines produce motion blur at high throughput speeds. Poorly maintained lighting fixtures cause low-illumination degradation that is uneven across categories and shifts over time. Dust, fog, or airborne chemical aerosols scatter light before it reaches the camera sensor. Sensor heating at high frame rates amplifies read noise. A recent large-scale industry evaluation of 11 state-of-the-art models across 9 datasets found that models with 99.9% MVTec-AD AUROC exhibit significant performance degradation on real-world factory data [REF:Baitieva2025]. Adverse imaging conditions are explicitly listed as an unresolved open challenge for industrial deployment [REF:IADChallenges2025].

Two engineering responses are available to a practitioner deploying anomaly detection under degraded imaging conditions. The first — *test-time rescue preprocessing* — applies classical image restoration (contrast enhancement, deconvolution, denoising, dehazing) to the camera output before passing it to the anomaly detector. This approach is model-agnostic, requires no retraining, and draws on a mature signal processing literature. The second — *training-time data augmentation* — exposes the model to synthetic corruptions during the training phase, with the expectation that the learned normal representation will be more tolerant of corruption at test time. Both strategies appear intuitively reasonable, and both are deployed in practice without systematic evaluation of their actual benefit.

The literature offers limited guidance. The closest prior work — a study of paired well-lit and low-light images [REF:LowLightJCDE2025] — covers only illumination variation for a single condition type and proposes a novel architecture rather than evaluating existing deployed models. The large-scale corruption study of [REF:Baitieva2025] tests synthetic corruptions as a secondary experiment but does not evaluate preprocessing rescue or training-time augmentation. No published work delivers a systematic multi-corruption, multi-severity benchmark with a controlled preprocessing rescue analysis.

This paper fills that gap with a controlled 4-way study. We evaluate PatchCore [REF:Roth2022] and PaDiM [REF:Defard2021] — the two most widely used open-source feature-embedding detectors — under clean training, augmented training, and six rescue preprocessing methods across the complete MVTec-AD benchmark. Our experimental design produces 6,120 controlled AUROC measurements, enabling direct comparison of all strategy combinations at each corruption type and severity level.

**Our contributions are:**
1. The first large-scale, multi-corruption, multi-severity rescue preprocessing benchmark for industrial anomaly detection, covering 6,120 measurements across 4 experimental conditions.
2. Empirical evidence that rescue preprocessing is net-harmful for feature-embedding anomaly detectors regardless of training regime, with harmful rates of 65–81% across all conditions — the *preprocessing fallacy*.
3. Empirical evidence that training-time augmentation substantially improves corruption robustness (+10–12 pp mean AUROC) at negligible clean-image cost (−0.5 to −2.5 pp).
4. The counter-intuitive finding that augmented training increases rescue harm, not reduces it, providing a mechanistic explanation grounded in feature distribution alignment.
5. A practical deployment recommendation table for engineers selecting imaging pipelines under constrained budgets.

---

## 3. RELATED WORK (~500 words)

### 3.1 Industrial Anomaly Detection

Feature-embedding methods dominate the current landscape. PatchCore [REF:Roth2022] constructs a greedy coreset of mid-level patch features (WideResNet-50, layers 2–3) and scores test patches by nearest-neighbor distance to the coreset. PaDiM [REF:Defard2021] fits a multivariate Gaussian distribution at each spatial patch position across the training set. Both achieve near-perfect AUROC on MVTec-AD under standard conditions. Recent surveys [REF:LiSurvey2025, REF:RealWorldSurvey2025] provide comprehensive taxonomies of reconstruction-based, embedding-based, and vision-language approaches; this work focuses on the embedding-based family as the deployed industrial standard.

### 3.2 Robustness of Anomaly Detectors

The robustness of anomaly detection models under distribution shift has received growing attention. [REF:Baitieva2025] evaluate 11 models across 9 datasets and demonstrate that MVTec-AD AUROC does not predict real-world performance. [REF:IADChallenges2025] enumerate adverse imaging conditions as an explicitly open challenge. [REF:LowLightJCDE2025] study low-light specifically but for a single condition type. No prior work provides a comprehensive multi-type, multi-severity, multi-model benchmark with controlled rescue preprocessing.

### 3.3 Classical Image Restoration for Computer Vision

Classical restoration methods — CLAHE [REF:Pizer1987, REF:Zuiderveld1994], Retinex [REF:Land1971, REF:Jobson1997], Wiener deconvolution [REF:Wiener1949], Non-Local Means [REF:Buades2005], and Dark Channel Prior dehazing [REF:He2009] — were designed to maximize perceptual quality for human observers. Their compatibility with deep feature extractors is less studied. Prior work in object recognition [REF:TendencyToOverfit] shows preprocessing can harm CNN performance by introducing distribution-shift artifacts. We extend this finding to the anomaly detection domain where the effect is more severe: unlike discriminative networks, anomaly detectors flag any out-of-distribution signal, making them maximally sensitive to preprocessing-induced artifacts.

### 3.4 Training-Time Augmentation for Robustness

Data augmentation for robustness under corruption is well-studied in classification [REF:AugMix, REF:DeepAugment]. Its application to anomaly detection is constrained by the one-class learning paradigm: augmenting normal training images risks shifting the normal distribution, potentially harming clean-image detection. Our results quantify this trade-off: augmented training costs 0.5–2.5 pp clean-image AUROC while gaining 10–12 pp robustness under corruption.

---

## 4. METHOD (~700 words)

### 4.1 Models

We evaluate two feature-embedding anomaly detectors available through Anomalib [REF:Anomalib]:

**PatchCore** uses a WideResNet-50-2 backbone with features extracted from layers 2 and 3. A greedy coreset (10% subsampling) reduces the memory bank to manageable size. Anomaly scores are computed as the distance to the nearest `k=9` coreset neighbors.

**PaDiM** uses a WideResNet-50-2 backbone with features from layers 1, 2, and 3 (reduced to 100 dimensions via random projection). A multivariate Gaussian is fit per spatial patch position across training images. Both models train for 1 epoch (`max_epochs=1`); no gradient updates occur — PatchCore constructs the coreset and PaDiM fits the Gaussians from a single pass.

### 4.2 Dataset

We use MVTec-AD [REF:Bergmann2019] (15 industrial categories, 4,096 training images, 1,725 test images with ground-truth anomaly labels). MVTec-AD was captured under controlled studio conditions, providing a clean baseline against which synthetic corruptions can be precisely attributed. All experiments use 3 random seeds (42, 123, 456) for training; results are reported as means across seeds.

### 4.3 Corruption Protocol

Five corruption types are applied on-the-fly via a `CorruptedDatasetWrapper` at three severity levels:

| Corruption | Mild | Moderate | Severe |
|---|---|---|---|
| Low-light | γ=0.65 | γ=0.35 | γ=0.15 |
| Gaussian blur | σ=1, k=5 | σ=3, k=15 | σ=5, k=25 |
| Motion blur | k=7 | k=15 | k=25 |
| Sensor noise | var=0.005 | var=0.02 | var=0.05 |
| Fog/haze | coef∈[0.2,0.4] | coef∈[0.4,0.6] | coef∈[0.6,0.8] |

Stochastic corruptions (noise, fog) use per-sample seeds derived from image index to ensure reproducibility. Corruptions are applied only to test images; training images use the clean split unless stated otherwise.

### 4.4 Rescue Preprocessing

Six rescue methods are evaluated at all three severity levels:

| Corruption | Method | Parameters |
|---|---|---|
| Low-light | CLAHE | clip=3.0, tile=8×8 |
| Low-light | Retinex (SSR) | σ=30 |
| Gaussian blur | Wiener (Gaussian PSF) | σ matched to corruption |
| Motion blur | Wiener (Motion PSF) | k matched to corruption |
| Sensor noise | Non-Local Means | patch=7, dist=11 |
| Fog/haze | Dark Channel Prior | ω=0.95, patch=15 |

Each rescue method is applied to already-corrupted images before passing them to the frozen detector. No model parameters are modified.

> **Note on Wiener applicability:** In this study, blur kernels are known by construction, making Wiener deconvolution tractable. In real-world deployment with unknown kernels, blind deconvolution would be required. Results for Wiener therefore represent an **upper bound** on deconvolution performance.

### 4.5 Augmented Training

For the augmented training condition, training images are pre-processed by `prepare_augmented_train_data(aug_prob=0.50)`: each training image is independently corrupted with probability 0.50, with corruption type and severity drawn uniformly at random from all 15 (type, severity) combinations. Corrupted images are saved to disk before training begins. The MVTecAD datamodule is then pointed at this pre-generated dataset; all other training parameters are identical to the clean condition.

### 4.6 Evaluation Protocol and Row Structure

Each (category, seed, training condition) triple produces 34 rows:
- 1 baseline row: AUROC on clean test images post-training
- 15 degradation rows: AUROC on corrupted test images (5 types × 3 severities)
- 18 rescue rows: AUROC on corrupted-then-restored test images (6 methods × 3 severities)

The primary metric is **image-level AUROC**, consistent with all published baselines. 6,120 total measurements (2 models × 2 training conditions × 45 pairs × 34 rows).

---

## 5. RESULTS (~900 words)

### 5.1 Baseline Reproduction

Clean-trained model AUROC on clean test images matches published literature closely:

| Model | This Work | Published |
|---|---|---|
| PatchCore | 0.9822 ± 0.0235 | 0.981 [REF:Roth2022] |
| PaDiM | 0.9226 ± 0.0818 | 0.918 [REF:Defard2021] |

Augmented training costs −0.5 pp (PatchCore: 0.9773) and −2.5 pp (PaDiM: 0.8979) on clean-image AUROC.

### 5.2 Effect of Corruption on Clean-Trained Models

Table X reports mean degradation AUROC across 15 categories and 3 seeds for clean-trained models. Key observations:

- **PatchCore is more vulnerable at mild severity** for motion blur (0.782) and Gaussian blur (0.917→0.648 at moderate), while PaDiM holds higher AUROC at mild blur.
- **Both models collapse at severe corruption**: fog/haze severe reduces PatchCore to 0.562 and PaDiM to 0.521; Gaussian blur severe reduces both to ≈0.58.
- **Sensor noise is the most tolerable corruption**: PatchCore retains 0.847 mean AUROC at mild severity.

### 5.3 Augmented Training Improves Robustness

Augmented training improves degradation AUROC across every corruption type and severity level without exception (Wilcoxon one-sided, PatchCore: +12.3 pp, p < 0.001; PaDiM: +10.1 pp, p < 0.001). Table Y reports the per-condition deltas:

**Mechanism**: Augmented training narrows the covariate shift between clean training images and corrupted test images by expanding the support of the learned normality model. For PatchCore, the coreset of normal-class patch embeddings grows to include representations of clean-but-degraded textures, reducing the nearest-neighbour anomaly score inflation caused by corruption. For PaDiM, the class-conditional Gaussian fitted to multi-scale feature activations acquires greater covariance along corruption-induced directions, reducing the Mahalanobis distance for corrupted-but-normal inputs. Crucially, the gain is consistent across all five corruption types and three severity levels, indicating that augmented training is not tailored to any specific degradation but generalises as a general robustness mechanism.

| Corruption | Severity | PatchCore Δ | PaDiM Δ |
|---|---|---|---|
| Gaussian blur | Severe | **+0.246** | +0.129 |
| Motion blur | Severe | **+0.226** | +0.170 |
| Motion blur | Moderate | +0.216 | +0.141 |
| Fog/haze | Moderate | +0.156 | +0.068 |
| Low-light | Severe | +0.091 | +0.137 |
| Sensor noise | Severe | +0.118 | +0.075 |

PatchCore benefits more than PaDiM for blur corruptions (+0.21–0.25 pp vs +0.13–0.17 pp at moderate/severe), likely because PatchCore's nearest-neighbor coreset matching is strongly sensitive to high-frequency feature collapse under blur, and augmented training re-calibrates the coreset to include blur-domain features. PaDiM's per-position Gaussian is inherently smoother and more tolerant of mild distribution shifts, explaining its smaller but consistent gains.

### 5.4 Rescue Preprocessing Is Net-Harmful (The Preprocessing Fallacy)

Table Z reports rescue success rates and mean AUROC deltas across all 810 rescue instances per condition. In all four conditions, mean rescue delta is significantly negative (Wilcoxon two-sided, all p < 0.001):

| Condition | Beneficial | Success Rate | Mean Δ |
|---|---|---|---|
| PaDiM — Clean | 284 / 810 | 35.1% | −0.046 |
| PatchCore — Clean | 209 / 810 | 25.8% | −0.072 |
| PatchCore — Augmented | 184 / 810 | 22.7% | −0.149 |
| PaDiM — Augmented | 157 / 810 | 19.4% | −0.113 |

**Mechanism — The Preprocessing Fallacy**: Restoration algorithms do not return corrupted images to the clean training distribution. They create a *third distribution* — distinct from both clean training images and naturally corrupted images — that contains restoration-specific artifacts. Anomaly detectors, which score images by distance from their learned normality distribution, flag these artifacts as defects. The result is elevated anomaly scores even in normal image regions, yielding worse performance than the degraded-but-unprocessed input.

**Wiener deconvolution** is the most harmful rescue (FDR-corrected Wilcoxon, all p < 0.001): mean AUROC delta ranges from −0.141 (PaDiM clean) to −0.370 (PatchCore augmented). Wiener filters invert the blur kernel in the frequency domain, which in low-SNR conditions amplifies noise into structured, high-frequency ringing artefacts (the Gibbs phenomenon) at every edge transition. PatchCore, which scores anomalies via nearest-neighbour distance in a patch embedding space built from clean images, generates embedding vectors for these ringing artefacts that are orthogonal to the entire normal coreset, producing catastrophically elevated anomaly scores across normal image regions.

**CLAHE** on low-light is the only rescue with a near-neutral profile (PatchCore clean/augmented: −0.2 to −0.4 pp, not significant after FDR correction). CLAHE's adaptive histogram equalisation adjusts only local luminance distributions without modifying spatial frequency content or inter-channel colour relationships. Deep convolutional feature extractors exhibit natural contrast invariance in intermediate and late layers, meaning CLAHE-processed images occupy a region of feature space close to the clean training distribution. The exception is PaDiM (augmented) at −5.5 pp (p < 0.001 after FDR): augmented training incorporates uniform global brightness jitter, but CLAHE applies spatially *non-uniform* local contrast enhancement most aggressively in low-luminance regions — a pattern outside the augmentation coverage, explaining the localised degradation for this specific condition.

**Dark Channel Prior dehazing** provides marginal and inconsistent benefit: PatchCore clean fog/haze severe +0.011 pp; all other conditions negative.

### 5.5 Augmented Training Amplifies Rescue Harm

A key finding is that combining augmented training with rescue preprocessing produces the worst outcomes. Mean rescue delta worsens from −0.046 to −0.113 for PaDiM and from −0.072 to −0.149 for PatchCore. This is counterintuitive: one might expect a more corruption-robust model to be more tolerant of preprocessing. The opposite is observed.

**Mechanism**: Augmented training recalibrates the model's feature distribution to include corruption-domain patterns (blurred textures, noise-altered gradients, fog-attenuated contrast). When rescue preprocessing is then applied, it removes exactly those patterns — but imperfectly, introducing PSF-mismatch artifacts or over-smoothing. The result is a test image that is out-of-distribution *relative to the augmented model's learned normal space*, triggering higher anomaly scores.

---

## 6. DISCUSSION AND CONCLUSIONS (~500 words)

### 6.1 The Preprocessing Fallacy in Feature-Embedding Anomaly Detection

Classical image restoration was designed to optimize perceptual quality for human observers. Human visual perception integrates global semantic context, is tolerant of high-frequency artifacts, and actively suppresses ringing as long as edges are visible. Feature-embedding anomaly detectors do the opposite: they are highly sensitive to local patch-level texture statistics and flag any deviation from the distribution of normal patches.

This asymmetry is formalised in the *preprocessing fallacy*: restoration algorithms do not map corrupted images back into the clean training distribution. They create a third distribution — containing restoration-specific artifacts (ringing, illumination maps, smoothing boundaries) — that is further from the learned normality model than the original corruption was. This explains not only why all rescue methods are net-harmful, but also why Wiener deconvolution is catastrophically harmful: PSF-mismatch ringing saturates the patch-embedding anomaly score globally, while perceptual quality metrics (SSIM, PSNR) would rate the same image as improved. Standard image processing pipelines evaluated by perceptual quality metrics cannot be assumed safe for anomaly detection deployment.

### 6.2 Augmented Training as the Practical Solution

Augmented training is the clear recommendation. At 50% corruption probability during training, with random type and severity selection, both PatchCore and PaDiM achieve 10–12 pp mean robustness improvement at a cost of 0.5–2.5 pp on clean-image performance. This trade-off is acceptable in virtually any industrial deployment where some proportion of images will be captured under degraded conditions.

The augmentation strategy is model-agnostic and requires no changes to inference infrastructure. The implementation overhead is a single pre-processing step (`prepare_augmented_train_data`) that runs once before training.

### 6.3 Deployment Recommendations

| Condition | Recommendation |
|---|---|
| Low-light, severe | Augmented training + CLAHE (PatchCore only) |
| Blur (any type/severity) | Augmented training only. No deconvolution. |
| Sensor noise | Augmented training only. NLM denoising neutral-to-harmful. |
| Fog/haze | Augmented training only. DCP marginally positive but inconsistent. |
| Clean images | Clean training. Augmented training safe (−0.5 pp PatchCore). |

### 6.4 Limitations and Future Work

**Synthetic-to-real gap**: Corruptions are synthetically generated and may not fully capture real-world imaging variation. **Limited model scope**: Results apply to feature-embedding detectors; reconstruction-based methods and vision-language models (WinCLIP, AnomalyGPT) may show different patterns. **Aug. probability not ablated**: aug_prob=0.50 is a single operating point; the trade-off curve across probabilities is left for future work. **Blind deconvolution**: Known-kernel Wiener results are an upper bound; real-world unknown-kernel deconvolution is expected to be worse. **Cross-dataset generalization**: MVTec-AD results should be validated on VisA to confirm generalizability.

### 6.5 Conclusions

We present the first systematic 4-way benchmark of rescue preprocessing and training-time augmentation for robust industrial anomaly detection. Three phenomena are established with statistical significance:

1. **Augmented training significantly improves corruption robustness** (+12.3 pp PatchCore, +10.1 pp PaDiM, Wilcoxon p < 0.001) by expanding the learned normality manifold into corruption-domain feature space, with negligible clean-image cost (−0.5 to −2.5 pp).

2. **Test-time rescue preprocessing is net-harmful** in all 4 conditions (−4.6 to −14.9 pp, all p < 0.001) — the *preprocessing fallacy* — because restoration algorithms produce a third distribution containing novel artifacts that anomaly detectors score as defects.

3. **Rescue harm is method-specific and severe**: Wiener deconvolution (−14 to −37 pp via Gibbs ringing) and Retinex (−3 to −11 pp via colour distortion) are consistently harmful; CLAHE is the only conditionally safe method for PatchCore under low-light due to its spatial frequency-preserving contrast adjustment.

Engineers deploying anomaly detection under adverse imaging conditions should adopt augmented training as the primary robustness strategy and avoid test-time restoration pipelines, with the sole exception of CLAHE for PatchCore under low-light conditions.

---

## 7. APPENDIX

### A. Per-Category Baseline AUROC

*(Full table in `benchmark_report.md` §Table 1)*

### B. Full Rescue Delta Table

*(Full table in `benchmark_report.md` §Tables 5a–5b)*

### C. Implementation Details

All experiments use Anomalib v1.x with PyTorch on Kaggle GPU (NVIDIA T4/P100, 16 GB VRAM). `ANOMALIB_USE_RICH=0` and `sys.setrecursionlimit(5000)` required for stable Kaggle execution. Results saved incrementally after every `engine.test()` call. GPU memory freed after each (category, seed) pair via `del model; torch.cuda.empty_cache()`.

Full source code and benchmark CSVs available at: [repository link TBD]

---

## CITATION PLACEHOLDERS

| Tag | Paper | Notes |
|---|---|---|
| REF:Bergmann2019 | MVTec AD — CVPR 2019 | Primary dataset |
| REF:Roth2022 | PatchCore — CVPR 2022 | Primary model |
| REF:Defard2021 | PaDiM — ICPR 2021 | Secondary model |
| REF:Baitieva2025 | arXiv 2503.23451 — Valeo/Intel | Gap evidence |
| REF:IADChallenges2025 | arXiv 2501.11310 | Gap evidence |
| REF:LowLightJCDE2025 | Oxford JCDE May 2025 | Prior narrow work |
| REF:LiSurvey2025 | arXiv 2503.13195 | Methods survey |
| REF:RealWorldSurvey2025 | arXiv 2507.13378 | Recent survey |
| REF:Anomalib | OpenVINO Anomalib — GitHub | Framework |
| REF:Pizer1987 | Pizer et al. 1987 | CLAHE seminal |
| REF:Zuiderveld1994 | Zuiderveld 1994 | CLAHE implementation |
| REF:Land1971 | Land & McCann 1971 | Retinex theory |
| REF:Jobson1997 | Jobson et al. 1997 | SSR Retinex |
| REF:Wiener1949 | Wiener 1949 | Wiener filter |
| REF:Buades2005 | Buades et al. CVPR 2005 | NLM denoising |
| REF:He2009 | He et al. CVPR 2009 / TPAMI 2011 | Dark Channel Prior |
