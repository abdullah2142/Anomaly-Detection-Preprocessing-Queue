# Literature Review
## Robust Industrial Anomaly Detection Under Adverse Imaging Conditions

*Complete reference compendium for Related Work section, citations, and background framing.*

---

## PART I: ANOMALY DETECTION — CORE METHODS

### 1.1 Foundational Datasets

| Paper | Venue | Key Contribution | Notes for Your Paper |
|:---|:---|:---|:---|
| Bergmann et al., **MVTec AD: A Comprehensive Real-World Dataset and Benchmark for the Detection of Industrial Anomalies** | CVPR 2019 / IJCV 2021 | 15-category industrial anomaly dataset; studio-controlled captures with pixel-level masks | The dataset you benchmark on. Cite when describing MVTec-AD. The studio-controlled capture is the feature you exploit. |
| Bergmann et al., **The MVTec 3D-AD Dataset for Unsupervised 3D Anomaly Detection and Localization** | VISAPP 2022 | 3D point cloud extension of MVTec | Out of scope but cite as "future work — 3D degradation" |
| Bergmann et al., **Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection and Localization (MVTec LOCO)** | IJCV 2022 | Logical vs. structural anomaly split | Cite as rejected primary dataset (logical anomalies are orthogonal to low-level corruption) |
| Zou et al., **SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation (VisA)** | ECCV 2022 | 12-class, 10,821-image industrial benchmark; harder than MVTec | Cite when introducing VisA as your generalization check dataset |
| Mishra et al., **BTAD: A Large-Scale Dataset for Real-World Blurry Text Anomaly Detection** | CVPR Workshop 2021 | Real-world capture variation baked in | Cite as rejected dataset — uncontrolled baseline confounds corruption attribution |

---

### 1.2 Feature Embedding / Memory Bank Methods

These are the method family your paper studies (PatchCore, PaDiM).

| Paper | Venue | Key Contribution | Notes for Your Paper |
|:---|:---|:---|:---|
| Roth et al., **Towards Total Recall in Industrial Anomaly Detection (PatchCore)** | CVPR 2022 | WideResNet50 mid-level patch features + greedy coreset memory bank; SOTA on MVTec at time of publication | **Primary model.** Must cite. Use Table 2 (per-category AUROCs) as your baseline comparison target. |
| Defard et al., **PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization** | ICPR 2021 | Per-patch multivariate Gaussian; ResNet/EfficientNet backbone | **Secondary model.** Must cite for methodology. |
| Cohen & Hoshen, **Sub-Image Anomaly Detection with Deep Pyramid Correspondences (SPADE)** | arXiv 2020 | KNN in feature space + pyramid pooling; early memory bank method | Cite as PatchCore predecessor in Related Work |
| Lee et al., **CFA: Coupled-hypersphere-based Feature Adaptation for Target-Oriented Anomaly Localization** | IEEE Access 2022 | Hypersphere memory bank with domain-adapted features | Related method in the memory bank family |
| Liu et al., **SimpleNet: A Simple Network for Image Anomaly Detection and Localization** | CVPR 2023 | Feature adapter + Gaussian noise in feature space; 99.6% AUROC; fast inference | Cite as evidence of embedding methods' near-saturation on clean MVTec |
| Bae et al., **PNI: Industrial Anomaly Detection using Position and Neighborhood Information** | ICCV 2023 | Spatial position-aware neighborhood features | Recent embedding variant; cite in methods landscape |

---

### 1.3 Knowledge Distillation / Teacher-Student Methods

| Paper | Venue | Key Contribution | Notes for Your Paper |
|:---|:---|:---|:---|
| Wang et al., **Student-Teacher Feature Pyramid Matching for Anomaly Detection** | BMVC 2021 | Multi-scale teacher-student feature discrepancy | Classic KD baseline |
| Deng & Li, **Anomaly Detection via Reverse Distillation from One-Class Embedding (RD4AD)** | CVPR 2022 | Reverse distillation — student decoder reconstructs teacher features; anomalies fail to reconstruct | Widely cited KD method; contrast with memory bank in Related Work |
| Batzner et al., **EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies** | WACV 2024 | Lightweight student-teacher + autoencoder; <2ms inference | Cite as evidence of efficiency focus in SOTA; contrasts with your deployment-oriented preprocessing goal |

---

### 1.4 Reconstruction-Based Methods

| Paper | Venue | Key Contribution | Notes for Your Paper |
|:---|:---|:---|:---|
| Bergmann et al., **Improving Unsupervised Defect Segmentation by Applying Structural Similarity to Autoencoders** | VISAPP 2019 | SSIM loss for autoencoder-based anomaly detection | Early reconstruction baseline |
| Zavrtanik et al., **Reconstruction by Inpainting for Visual Anomaly Detection (RIAD)** | Pattern Recognition 2021 | Inpainting-based reconstruction; different from memory bank | Cite to show reconstruction family exists and is sensitive to texture degradation (contrasting with your methods) |
| Zavrtanik et al., **DSR: A Dual Subspace Re-projection Network for Surface Anomaly Detection** | ECCV 2022 | Feature-space anomaly synthesis + dual decoder reconstruction | Modern reconstruction variant |
| Gudovskiy et al., **CFLOW-AD: Real-Time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows** | WACV 2022 | Normalizing flow density estimation | Cite as non-embedding alternative in methods taxonomy |

---

### 1.5 Vision-Language / Foundation Model Methods

| Paper | Venue | Key Contribution | Notes for Your Paper |
|:---|:---|:---|:---|
| Jeong et al., **WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation** | CVPR 2023 | CLIP-based window scoring with compositional text prompts; zero-shot anomaly detection | Cite as your stretch goal / future work. "VLM robustness under corruption is an open question" |
| Gu et al., **AnomalyGPT: Detecting Industrial Anomalies Using Large Vision-Language Models** | AAAI 2024 | LLM-based anomaly reasoning and explanation | Positions your work on the practical/deployment end of the spectrum |
| Zhou et al., **AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection** | ICLR 2024 | Learnable prompts for CLIP-based zero-shot detection | Related to WinCLIP; cite if discussing VLM family broadly |
| Li et al., **MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled Images** | ICLR 2024 | Zero-shot using mutual comparison between unlabeled test images | Emerging direction; cite briefly |

---

## PART II: ROBUSTNESS, CORRUPTION, AND BENCHMARKING

### 2.1 Corruption Robustness — General Computer Vision

| Paper | Venue | Key Contribution | Notes for Your Paper |
|:---|:---|:---|:---|
| Hendrycks & Dietterich, **Benchmarking Neural Network Robustness to Common Corruptions and Perturbations (ImageNet-C)** | ICLR 2019 | 15 corruption types × 5 severities on ImageNet; mCE metric | **Methodological ancestor of your work.** Cite to show the corruption benchmark paradigm exists in classification and you are bringing it to anomaly detection |
| Michaelis et al., **Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming** | arXiv 2019 | Extends ImageNet-C idea to object detection | Cite to show the paradigm generalises across vision tasks — anomaly detection is a gap |
| Hendrycks et al., **The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization** | ICCV 2021 | Multiple robustness dimensions beyond corruptions | Good conceptual framing for your Introduction |

---

### 2.2 Corruption Robustness — Anomaly Detection Specific

| Paper | Venue | Key Contribution | Notes for Your Paper |
|:---|:---|:---|:---|
| **MVTec-C / Robustness of UAD Models** (USTC group) | arXiv/ECCV 2023 | 8 corruption types × 5 severities on MVTec; representation similarity and KD methods most robust; Feature Alignment Module (FAM) | **Most directly related work.** Cite explicitly and differentiate: they diagnose robustness without preprocessing rescue analysis; your work adds the rescue layer. |
| Baitieva et al., **Beyond Academic Benchmarks: Real-World Performance Evaluation of SOTA Anomaly Detection Methods** | arXiv 2503.23451, Mar 2025 | 11 SOTA models × 9 datasets; documents academia-industry AUROC collapse; tests Gaussian noise/blur/shadowing | **Primary gap evidence paper.** Must cite in Introduction. Note: they include synthetic corruptions but no preprocessing rescue. |
| *Anomaly Detection for Industrial Applications: Challenges & Future Directions* | arXiv 2501.11310, Jan 2025 | Lists adverse imaging as explicitly open challenge | Good for framing "the gap has been acknowledged" |
| *Unsupervised IAD Using Paired Well-lit and Low-light Images* | Oxford JCDE, May 2025 | Paired well-lit/low-light images; proposes new architecture for illumination robustness | Only covers one condition type; proposes new model rather than studying existing ones. Cite as "closest related work — narrow scope" |

---

### 2.3 Real-World Industrial Deployment Challenges

| Paper | Venue | Key Contribution | Notes for Your Paper |
|:---|:---|:---|:---|
| Li et al., **A Comprehensive Survey for Real-World Industrial Defect Detection** | arXiv 2507.13378, Jul 2025 | Bridges academic benchmarks and real-world deployment challenges across multiple industries | Cite in Introduction and Related Work to contextualize deployment difficulty |
| Yang et al., **A Survey on Visual Anomaly Detection: Challenge, Approach, and Prospect** | arXiv 2024 | Reviews data scarcity, modality diversity, anomaly hierarchy challenges | Good background survey for anomaly detection challenges section |
| Bonfiglioli et al., **Datasets for Defect Detection: A Systematic Review** | IEEE Trans. on Industrial Informatics, 2022 | Systematic review of industrial inspection datasets | Broader context for why MVTec is the standard |

---

## PART III: IMAGE RESTORATION AND PREPROCESSING

### 3.1 Low-Light Enhancement

| Paper | Venue | Key Contribution | Citation Purpose |
|:---|:---|:---|:---|
| Land & McCann, **Lightness and Retinex Theory** | Journal of the Optical Society of America, 1971 | Original Retinex theory: lightness perception is based on local ratios, not absolute luminance | Foundational theory behind Retinex rescue |
| Jobson et al., **A Multiscale Retinex for Bridging the Gap Between Color Images and the Human Observation of Scenes (SSR/MSR)** | IEEE TIP, 1997 | Single-Scale and Multi-Scale Retinex for illumination normalization | Cite as the Retinex implementation basis |
| Zuiderveld, **Contrast Limited Adaptive Histogram Equalization (CLAHE)** | Graphics Gems IV, 1994 | Local contrast enhancement with clipping to prevent noise amplification | Foundational reference for CLAHE rescue |
| Guo et al., **LIME: Low-Light Image Enhancement via Illumination Map Estimation** | IEEE TIP, 2017 | Structure-aware illumination map estimation; one of the most cited low-light methods | Cite to show classical LLIE alternatives beyond CLAHE |
| Wei et al., **Deep Retinex Decomposition for Low-Light Enhancement (RetinexNet)** | BMVC 2018 | Deep learning Retinex for paired training; noise-aware decomposition | Modern learned alternative to classical Retinex |
| Li et al., **A Survey of Deep Learning-Based Low-Light Image Enhancement** | MDPI Sensors, 2023 | Comprehensive review of LLIE methods | Cite as survey context when discussing rescue strategy selection rationale |

---

### 3.2 Deblurring

| Paper | Venue | Key Contribution | Citation Purpose |
|:---|:---|:---|:---|
| Wiener, **Extrapolation, Interpolation, and Smoothing of Stationary Time Series** | MIT Press, 1949 | Original Wiener filter theory | Foundational citation for Wiener deconvolution rescue |
| Gonzalez & Woods, **Digital Image Processing** (3rd ed.) | Pearson, 2008 | Chapter 5: image restoration — Wiener filter derivation, PSF theory | Standard textbook reference for Wiener and PSF formulation |
| Kupyn et al., **DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks** | CVPR 2018 | GAN-based blind deblurring | Cite as a learned alternative excluded from study (environment compatibility issues) |
| Kupyn et al., **DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better** | ICCV 2019 | Improved GAN-based deblurring; real-time capable | Cite as excluded method with rationale (PyTorch compatibility) |
| Zamir et al., **Restormer: Efficient Transformer for High-Resolution Image Restoration** | CVPR 2022 | Transformer-based multi-task restoration (deblur, denoise, dehaze) | Cite as modern learned alternative; excluded as out-of-scope for this work |

---

### 3.3 Denoising

| Paper | Venue | Key Contribution | Citation Purpose |
|:---|:---|:---|:---|
| Buades et al., **A Non-Local Algorithm for Image Denoising (NLM)** | CVPR 2005 | Non-local means: weight patches by similarity across whole image | **Must cite** for NLM rescue method |
| Dabov et al., **Image Denoising by Sparse 3D Transform-Domain Collaborative Filtering (BM3D)** | IEEE TIP, 2007 | Block-matching 3D collaborative filtering; long-time SOTA | Cite as excluded method (runtime/benefit ratio) — shows you considered it |
| Lehtinen et al., **Noise2Noise: Learning Image Restoration without Clean Data** | ICML 2018 | Self-supervised denoising without clean reference | Cite as learned alternative; out of scope for this work |
| Zhang et al., **Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising (DnCNN)** | IEEE TIP, 2017 | CNN residual learning for blind Gaussian denoising | Frequently cited as bridge between classical and learned denoising |

---

### 3.4 Dehazing / Fog Removal

| Paper | Venue | Key Contribution | Citation Purpose |
|:---|:---|:---|:---|
| He et al., **Single Image Haze Removal Using Dark Channel Prior** | CVPR 2009 / IEEE TPAMI 2011 | Dark Channel Prior (DCP): haze-free patches have near-zero minimum intensity in some channel | **Must cite** for DCP dehaze rescue |
| Fattal, **Dehazing Using Color-Lines** | ACM TOG, 2014 | Color-line dehazing; alternative to DCP | Brief cite as DCP alternative |
| Li et al., **AOD-Net: All-in-One Dehazing Network** | ICCV 2017 | End-to-end learned dehazing with atmospheric scattering integration | Cite as learned alternative; excluded from study |
| Qin et al., **FFA-Net: Feature Fusion Attention Network for Single Image Dehazing** | AAAI 2020 | Attention-based dehazing; near-SOTA on RESIDE | Cite to frame DCP as classical baseline vs. learned SOTA |

---

### 3.5 Image Quality Assessment

| Paper | Venue | Key Contribution | Citation Purpose |
|:---|:---|:---|:---|
| Wang et al., **Image Quality Assessment: From Error Visibility to Structural Similarity (SSIM)** | IEEE TIP, 2004 | SSIM metric | Useful if you report image quality alongside AUROC |
| Mittal et al., **Making a 'Completely Blind' Image Quality Analyzer (BRISQUE)** | IEEE Signal Processing Letters, 2012 | No-reference IQA | Could cite if including perceptual quality measurements |

---

## PART IV: BROADER CONTEXT

### 4.1 Surveys on Anomaly Detection

| Paper | Venue | Key Contribution | Notes |
|:---|:---|:---|:---|
| Pang et al., **Deep Learning for Anomaly Detection: A Review** | ACM CSUR, 2021 | Classic comprehensive survey; classification, sequence, graph anomalies | Broad background; use for general anomaly detection framing in Introduction |
| Zhao et al., **Deep Industrial Image Anomaly Detection: A Survey** | Machine Intelligence Research, 2024 | Deep learning IAD taxonomy: reconstruction, embedding, self-supervised, foundation models | Best recent comprehensive IAD survey; cite frequently in Related Work |
| Xie et al., **A Survey on Visual Anomaly Detection: Challenge, Approach, and Prospect** | arXiv 2024 | Data scarcity, modality diversity, anomaly hierarchy | Cite in Related Work for challenge framing |
| Li et al., **Deep Learning Advancements in AD: A Comprehensive Survey** | arXiv 2503.13195, Mar 2025 | Most recent comprehensive methods taxonomy | Cite for recency; shows field is still rapidly evolving |
| *A Survey on Industrial Anomalies Synthesis* | arXiv 2025 | First dedicated survey on anomaly synthesis (hand-crafted, generative, VLM-based) | Cite in context of data augmentation and synthetic corruption design |

---

### 4.2 Anomaly Detection Toolkits

| Resource | Notes |
|:---|:---|
| Batzner et al., **Anomalib: A Deep Learning Library for Anomaly Detection in Images** | arXiv 2022. The framework used in this study. Must cite. |

---

### 4.3 Pre-Trained Backbone Literature

| Paper | Venue | Notes |
|:---|:---|:---|
| He et al., **Deep Residual Learning for Image Recognition (ResNet)** | CVPR 2016 | Backbone for PaDiM. Must cite. |
| Zagoruyko & Komodakis, **Wide Residual Networks (WideResNet)** | BMVC 2016 | Backbone for PatchCore. Must cite. |
| Deng et al., **ImageNet: A Large-Scale Hierarchical Image Database** | CVPR 2009 | Pre-training dataset for both backbones. Cite when describing "ImageNet-pretrained features." |

---

## PART V: CITATION PRIORITY GUIDE

Use this when writing to avoid over-citing.

### Must Cite (paper cannot be published without these)
1. Roth et al. 2022 — PatchCore
2. Defard et al. 2021 — PaDiM
3. Bergmann et al. 2019/2021 — MVTec-AD
4. Zou et al. 2022 — VisA
5. Baitieva et al. 2025 (2503.23451) — gap evidence
6. Buades et al. 2005 — NLM
7. He et al. 2009/2011 — DCP
8. Wiener 1949 — Wiener filter
9. Pizer/Zuiderveld 1987/1994 — CLAHE
10. Hendrycks & Dietterich 2019 — ImageNet-C (corruption benchmark precedent)
11. MVTec-C robustness paper (USTC) — most related existing work

### Should Cite (strengthens the paper)
12. Deng & Li 2022 — RD4AD (KD contrast)
13. Jeong et al. 2023 — WinCLIP (future work)
14. Li et al. 2023 — LLIE survey (rescue selection rationale)
15. He et al. 2016 — ResNet backbone
16. Zagoruyko 2016 — WideResNet backbone
17. Batzner et al. 2022 — Anomalib framework
18. Land & McCann 1971 / Jobson 1997 — Retinex theory
19. Oxford JCDE May 2025 — closest related work

### Cite if Space Permits
20. SimpleNet 2023 — near-saturation on clean MVTec
21. EfficientAD 2024 — efficiency angle
22. Deng et al. 2009 — ImageNet
23. Zhao et al. 2024 — IAD survey
24. Pang et al. 2021 — general AD survey
25. Kupyn et al. 2019 — DeblurGAN-v2 (excluded method)
26. Dabov et al. 2007 — BM3D (excluded method)

---

## PART VI: POSITIONING YOUR PAPER IN ONE PARAGRAPH

> *"Prior work on anomaly detection robustness has largely focused on clean, controlled benchmarks [Bergmann et al. 2019, Roth et al. 2022]. While Hendrycks & Dietterich [2019] established the corruption benchmark paradigm for image classification, and Baitieva et al. [2025] documented an academia-industry performance gap across 11 detectors, no systematic study has paired multi-condition corruption analysis with a preprocessing rescue evaluation for industrial anomaly detection. The closest related work is MVTec-C [USTC, 2023], which evaluates 8 corruption types on MVTec but does not study mitigation strategies. Recent work on low-light adaptation [Oxford JCDE, 2025] addresses one condition type but requires architectural modification. Our work differs on three axes: (1) we evaluate 5 physically motivated corruption types at 3 severity levels; (2) we systematically evaluate classical preprocessing as a deployment-ready mitigation strategy without modifying the detector; and (3) we evaluate two representative embedding-based models (PatchCore, PaDiM) to support family-level conclusions."*

---

*Last updated: May 2026*
