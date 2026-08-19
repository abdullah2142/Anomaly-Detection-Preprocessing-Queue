# Related Work

> Format: Paper | Venue | Role | Where to cite

---

## 1. Industrial Anomaly Detection Methods

These establish the field context and position PatchCore/PaDiM as the deployed standard.

### Primary Models

**Roth et al., *Towards Total Recall in Industrial Anomaly Detection (PatchCore)***
CVPR 2022
→ Primary model in this study. Achieves 98.1% mean image AUROC on MVTec-AD.
→ Cite in: Introduction (baseline performance claim), Method §4.1, Results §R1 (published numbers).

**Defard et al., *PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization***
ICPR 2021
→ Secondary model. Per-position Gaussian modeling. 93.1% mean MVTec-AD AUROC.
→ Cite in: Method §4.1, Results §R1.

### Benchmark Dataset

**Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection***
CVPR 2019
→ Primary dataset. Studio-controlled conditions. 15 industrial categories.
→ Cite in: Introduction (controlled baseline argument), Method §4.2, Results §R1.

### Surveys (for methods taxonomy in Related Work)

**Li et al., *Deep Learning Advancements in Anomaly Detection: A Comprehensive Survey***
arXiv 2503.13195, Mar 2025
→ Most current taxonomy: reconstruction-based, embedding-based, VLM approaches.
→ Cite in: Related Work §1 opening paragraph to establish breadth.

**[Comprehensive Survey: Real-World Industrial Defect Detection]**
arXiv 2507.13378, Jul 2025
→ Most recent survey. Cite for recency signal and to show awareness of latest work.
→ Cite in: Related Work §1 closing sentence.

### Adjacent Methods

**Jeong et al., *WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation***
CVPR 2023
→ Vision-language anomaly detection. Positions this study on the feature-embedding/CNN end.
→ Cite in: Related Work §1 as contrast to feature-embedding methods.

**Gu et al., *AnomalyGPT: Detecting Industrial Anomalies using Large Vision-Language Models***
AAAI 2024
→ LLM-based. Positions this work as the practical/deployment-oriented alternative.
→ Cite in: Related Work §1 final sentence (LLM methods not yet deployment-ready at scale).

---

## 2. The Research Gap — Robustness Under Deployment Conditions

These three papers form the core gap argument in the Introduction.

**Baitieva et al., *Beyond Academic Benchmarks: Towards Real-World Industrial Anomaly Detection***
arXiv 2503.23451, Mar 2025 (Valeo / Intel)
→ **Primary gap evidence.** 11 SOTA models × 9 datasets. Models with 99.9% MVTec AUROC collapse on factory data. Synthetic corruptions tested as secondary experiment but preprocessing rescue and augmented training not evaluated.
→ Cite in: Introduction (gap evidence, first paragraph), Related Work §2, Discussion (comparison).
→ Key sentence: *"Popular datasets are produced in controlled lab environments with artificially created defects, unable to capture the diversity of real production conditions."*

**[Anomaly Detection for Industrial Applications: Challenges and Future Directions]**
arXiv 2501.11310, Jan 2025
→ Adverse imaging conditions explicitly listed as an open challenge.
→ Cite in: Introduction paragraph 2 (problem framing), Related Work §2.

**[Unsupervised Industrial Anomaly Detection Using Paired Well-Lit and Low-Light Images]**
Oxford JCDE, May 2025
→ Closest prior work. Covers illumination variation for one condition type, proposes a new architecture rather than evaluating existing ones.
→ Cite in: Introduction (narrowness of prior work), Related Work §2.
→ Key distinction: single condition type, new architecture vs. existing detectors, no rescue analysis.

### Completing the 3-Paper Gap Argument

Use in this order in the Introduction:
1. **Problem exists** (Baitieva 2025): models collapse on real-world data.
2. **Benchmark does not capture it** (Bergmann 2019): MVTec-AD is studio-controlled by design.
3. **One narrow prior attempt is insufficient** (JCDE 2025): single condition, proposes new architecture.

Therefore: a systematic multi-corruption, multi-model, preprocessing rescue study is needed.

---

## 3. Adverse Imaging Conditions in Computer Vision

These papers establish that the problem of imaging degradation is real and broadly studied.

**[Sequential PatchCore / Real-World Surface Anomaly Study]** (Mao et al.)
ECCV 2024 Workshop
→ Adjacent robustness work from a different angle (surface impurities in real factory footage).
→ Cite in: Related Work §2 as real-world deployment challenge evidence.

**[AeBAD / Domain Generalization for Anomaly Detection]**
→ If found: studies domain shift in anomaly detection. Different angle (domain shift vs. corruption).
→ Cite if relevant to distinguish this work's focus on physical corruption vs. dataset domain shift.

---

## 4. Classical Image Restoration — Rescue Methods

Cite these in Method §4.4 when introducing each rescue technique.

**Pizer et al. (1987) + Zuiderveld (1994)**
→ CLAHE seminal references.
→ *"Adaptive histogram equalization and its variations"* (Pizer), *"Contrast limited adaptive histogram equalization"* (Zuiderveld).

**Land & McCann (1971) + Jobson et al. (1997)**
→ Retinex theory and Single-Scale Retinex implementation.
→ Cite for the illumination normalization principle underlying Retinex rescue.

**Wiener (1949)**
→ Wiener filter theory.
→ Cite for deconvolution formulation. Implementation via `skimage.restoration.wiener`.

**Buades et al., *A Non-Local Algorithm for Image Denoising***
CVPR 2005
→ NLM denoising seminal paper.

**He et al., *Single Image Haze Removal Using Dark Channel Prior***
CVPR 2009 / TPAMI 2011
→ Dark Channel Prior dehazing. Both versions exist; TPAMI is more complete.

**Dabov et al., *Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering (BM3D)***
IEEE TIP 2007
→ Cite as the excluded method: "BM3D was excluded due to prohibitive runtime with marginal benefit over NLM on MVTec image sizes."

---

## 5. Training-Time Augmentation for Robustness

**Hendrycks et al., *AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty***
ICLR 2020
→ Augmentation for corruption robustness in classification. Conceptually related but classification context.
→ Cite in Related Work §4 as prior augmentation-for-robustness work, then distinguish: anomaly detection is one-class and augmenting normals risks shifting the learned distribution.

**Geirhos et al., *Shortcut Learning in Deep Neural Networks***
Nature Machine Intelligence 2020
→ CNN feature sensitivity to texture vs. shape. Background for explaining why preprocessing artifacts harm feature-embedding detectors.
→ Optional cite in Discussion §6.1 (mechanistic explanation).

---

## Suggested Related Work Section Structure

```
§ Related Work

  1. Industrial Anomaly Detection
     - Feature embedding methods (PatchCore, PaDiM)
     - Reconstruction-based methods (brief)
     - Vision-language approaches (brief)
     - [Li 2025, RealWorld 2025, Roth 2022, Defard 2021, WinCLIP, AnomalyGPT]

  2. Robustness and Deployment Challenges
     - Academia-industry gap [Baitieva 2025, IADChallenges 2025]
     - Prior corruption studies (narrow) [JCDE 2025, Baitieva secondary]
     - What this work adds (systematic + rescue + augmentation)

  3. Classical Image Restoration for Computer Vision
     - CLAHE, Retinex, Wiener, NLM, DCP (cite seminal papers)
     - Prior evidence that restoration can harm CNN recognition (optional)

  4. Training-Time Augmentation
     - AugMix and related [Hendrycks 2020]
     - One-class learning constraint: augmenting normals shifts normal distribution
     - This work quantifies the trade-off for the first time in anomaly detection
```
