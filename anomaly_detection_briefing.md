Research Briefing

**Robust Industrial Anomaly Detection Under Adverse Imaging Conditions**

*A Systematic Corruption Benchmark and Preprocessing Rescue Analysis*

# **1\. The State of Industrial Anomaly Detection**

The field has converged around a clean paradigm over the last 3-4 years. Nearly all competitive methods are unsupervised — you train only on normal images, build a representation of what 'good' looks like, and flag anything that deviates. Broadly, these methods divide into two families: reconstruction-based approaches (autoencoders, GANs, transformers, diffusion models) and feature-embedding-based approaches, which use memory banks of normal feature representations.

## **1.1 The Dominant Baseline: PatchCore**

The primary model for this project is PatchCore. It extracts feature maps from an ImageNet-pretrained convolutional network (WideResNet50), using mid-level feature hierarchies (layers 2 and 3\) rather than the deepest layers, which are known to be strongly biased toward semantic object categories. Each spatial location in a feature map corresponds to a patch-level representation, and during inference the distance to the nearest nominal neighbor becomes the anomaly score. On the standard MVTec-AD benchmark it consistently scores above 98% AUROC — seemingly solved. The problem is that benchmark is deceptively clean.

## **1.2 The Broader Methods Landscape**

Understanding the full taxonomy matters for positioning this project:

* **Reconstruction-based methods** (autoencoders, GANs, diffusion models): flag high reconstruction error. Sensitive to texture degradation.

* **Feature embedding methods** (PatchCore, PaDiM): memory bank \+ nearest-neighbor distance. The dominant paradigm and focus of this project.

* **Vision-language / foundation models** (WinCLIP, DINOv2): zero/few-shot detection via text prompts or self-supervised features. Emerging direction with unknown robustness under degradation.

* **LLM-based reasoning** (AnomalyGPT): anomaly comprehension and explanation via large vision-language models. Theoretically interesting but not deployment-ready at scale.

| *Novelty framing: This project is not proposing a new anomaly detection algorithm. While degradation effects on anomaly detectors have been noted as a concern in recent literature, no published work provides a systematic multi-condition, multi-severity benchmark combined with a preprocessing rescue analysis that produces deployment-ready recommendations. This project fills that specific gap..* |
| :---- |

# **2\. The Research Gap**

## **2.1 The Academia-Industry Disconnect**

The March 2025 Valeo/Intel paper (arXiv 2503.23451) provides the strongest independent evidence for the gap. Their analysis across 11 SOTA models and 9 datasets reveals that models with 99.9% image-level AUROC on MVTec-AD show significant degradation on real-world data. Popular datasets are produced in controlled lab environments with artificially created defects, unable to capture the diversity of real production conditions. New methods often fail in production settings, showing significant performance degradation or requiring impractical computational resources.

## **2.2 The Adverse Condition Angle**

Embedding-based methods may struggle to adapt to low-light or variable lighting conditions that are common in manufacturing environments, potentially leading to misdetections or missed defects. This is an acknowledged problem with almost no systematic study behind it.

The one paper that directly tackles it — a May 2025 Oxford JCDE paper on paired low-light images — only covers illumination variation for one condition type and proposes a specific new architecture. This project is different and broader: systematic corruption benchmarking across multiple degradation types, combined with a preprocessing rescue analysis, using existing open-source detectors unchanged.

| IMPORTANT: The Valeo/Intel paper (2503.23451) does test synthetic corruptions (Gaussian noise, blur, shadowing) as a secondary experiment. This means the degradation diagnosis half of the project has been partially done. The genuine novelty of this project therefore sits primarily in the PREPROCESSING RESCUE ANALYSIS — the practical recommendation table that answers 'which preprocessing pipeline should a deploying engineer use?' That question has not been answered anywhere in the literature. |
| :---- |

## **2.3 Your Specific Research Contribution**

The contribution can be framed cleanly as two parts:

1. **Corruption benchmark:** A systematic evaluation of how image degradation (low-light, Gaussian blur, sensor noise, fog) affects state-of-the-art unsupervised anomaly detectors across multiple severity levels, applied to a clean controlled dataset.

2. **Preprocessing rescue analysis:** A comparative study of classical and learned preprocessing techniques as mitigation strategies, producing the first deployment-oriented recommendation: which preprocessing pipeline to use, for which degradation type, and at what cost to performance.

# **3\. Datasets**

## **3.1 Primary Dataset: MVTec-AD**

MVTec-AD (15 categories, freely available on Kaggle and the MVTec website) is the correct primary dataset for this project. The reasoning is experimental, not merely practical:

This project's core logic is: take a pristine starting point, introduce controlled adverse conditions, and measure the effect. For that to work, the baseline must be as clean as possible. MVTec-AD was captured in ideal studio conditions with consistent lighting and fixed framing. That cleanliness is the feature, not a limitation — you are the one introducing the adverse conditions, synthetically and deliberately.

* **Controlled baseline:** Studio-captured, consistent lighting, fixed framing. You control every variable.

* **Published calibration numbers:** Clean PatchCore hits 98%+ AUROC on all 15 categories. When your reproduction matches, your setup is verified.

* **Anomalib native support:** Zero configuration, loads out of the box, documented thoroughly.

* **Community familiarity:** Reviewers know this dataset intuitively. Your degradation curves are immediately interpretable.

## **3.2 Why Other Datasets Are Unsuitable as Primary**

MVTec-AD 2, BTAD, AeBAD, and VAD were all considered and rejected as primary datasets for a shared reason: they already contain real-world capture variation baked into the images. Starting from a noisy or variably-lit baseline makes it impossible to cleanly attribute AUROC drops to your synthetic corruptions. You cannot say 'the model degrades by X% under low-light conditions' if the dataset already has uncontrolled lighting variation in it.

MVTec LOCO was also considered and rejected. LOCO's unique feature — logical anomaly detection (wrong component placement, structural inconsistencies) — is largely orthogonal to the experimental variable in this project. Image-level corruption primarily degrades local texture discrimination, which affects structural anomalies. Logical anomalies are detected through global spatial reasoning and are less sensitive to low-level image quality. PatchCore also performs notably worse on LOCO under clean conditions, making degradation curves harder to interpret.

## **3.3 Secondary Dataset: VisA (Validation Only)**

VisA (12 object types, freely available, studio-captured conditions) is added as a late-stage generalization check — not as a parallel experiment stream. After completing the full corruption and preprocessing analysis on MVTec-AD, the best-performing preprocessing pipeline for each degradation type is run on VisA to verify that findings generalize.

This addition costs approximately 3-4 days of compute and produces a two-paragraph discussion section that protects against the most common reviewer objection: 'Your findings may be specific to MVTec's object types.' The conclusion becomes: 'Our preprocessing recommendations improve performance under synthetic corruption and generalize to a second independently-collected dataset.'

| *Dataset access: MVTec-AD is available at kaggle.com/datasets/ipythonx/mvtec-ad and mvtec.com/research-teaching/datasets/mvtec-ad. VisA is available at paperswithcode.com/dataset/visa.* |
| :---- |

# **4\. Models**

## **4.1 Why PatchCore as Primary**

PatchCore is the right anchor because it is the model most likely deployed in actual factories today. It has exhaustively documented clean-image baselines, trivial setup through Anomalib, and its architecture (mid-level ResNet features \+ nearest-neighbor memory bank) makes it maximally interpretable when things go wrong under corruption. When a blurred image confuses PatchCore, you can reason about exactly why: the blur degrades the texture edge information that patch features rely on.

## **4.2 Secondary Model: PaDiM**

PaDiM is added as a second embedding-based model. The practical reasons: it is already in Anomalib with zero additional setup cost, it uses the same feature extraction family as PatchCore (pretrained CNN features), and your corruption scripts and preprocessing pipelines apply identically. Running PaDiM alongside PatchCore allows you to make statements about the embedding-based family of methods, not just a single implementation. Reviewers will not be able to dismiss findings as 'PatchCore-specific.'

## **4.3 Stretch Goal: WinCLIP**

WinCLIP (vision-language, zero-shot detection via text prompts) is added as a stretch goal if weeks 3-4 remain on schedule. The scientific payoff is high: showing that a CLIP-based detector degrades differently than CNN-based embedding methods under the same corruptions is a genuinely novel finding. CLIP's language grounding may make it more or less robust to visual degradation — nobody has published this comparison. WinCLIP is also available through Anomalib, keeping the setup overhead low.

| *If WinCLIP is added, the paper title shifts to something like: 'How Robust Is Unsupervised Industrial Anomaly Detection? A Systematic Study Comparing CNN-Embedding and Vision-Language Methods Under Adverse Imaging Conditions.' That is a stronger paper.* |
| :---- |

# **5\. Corruption Benchmark Design**

## **5.1 Corruption Types and Severity Levels**

Five corruption types are implemented across three severity levels each, producing 15 degraded dataset variants from the clean MVTec-AD baseline. Severity levels are defined parametrically (specific parameter values) but should also be validated perceptually before committing — the jump between levels should feel proportional to a human observer, not just numerically even.

| Corruption Type | Implementation | Severity Levels | Library |
| ----- | ----- | ----- | ----- |
| Low-light | Gamma correction \+ Poisson noise | γ \= 0.5 (mild), 0.35 (moderate), 0.2 (severe) | albumentations |
| Gaussian blur | Gaussian blur kernel | σ \= 1 (mild), 2 (moderate), 4 (severe) | albumentations |
| Motion blur | Directional motion blur | kernel \= 7, 15, 25 px | albumentations |
| Sensor noise | Gaussian noise \+ salt-and-pepper | σ \= 0.02, 0.05, 0.10 | albumentations |
| Fog/haze | Atmospheric scattering overlay | density \= 0.3, 0.6, 0.9 | imgaug |
| Combined (hardest) | Low-light \+ Gaussian noise | moderate \+ moderate levels | albumentations |

| Severity level decision (Week 1, non-negotiable): Commit to exact parameter values before running any experiments. If you change severity definitions midway, your degradation curves become incomparable across runs. Define them in a config file, version-control it, and never change it after week 2\. |
| :---- |

## **5.2 What the Combined Case Tests**

The combined low-light \+ noise case is the most practically important. In a real factory in Bangladesh, you rarely face one degradation in isolation — you get poor lighting AND camera sensor noise simultaneously. This variant also stress-tests the preprocessing pipeline ordering question (Section 6.4), because applying CLAHE to a noisy dark image amplifies noise before improving contrast. The combined case is run at a single severity level (moderate \+ moderate) to keep the experiment tractable.

# **6\. Preprocessing Pipeline**

## **6.1 Design Philosophy**

All preprocessing is applied before the anomaly detector sees the image. PatchCore, PaDiM, and WinCLIP are not modified in any way. This is intentional — the research question is whether off-the-shelf preprocessing can rescue a deployed system, not whether a new detection architecture is needed.

## **6.2 Per-Degradation Preprocessing Stack**

| Degradation | Primary Technique | Alternative/Learned | Notes |
| ----- | ----- | ----- | ----- |
| Low-light | CLAHE (contrast-limited AHE) | Histogram matching (naive baseline) | Retinex dropped — redundant with CLAHE, heavier to tune |
| Gaussian blur | Wiener deconvolution | DeblurGAN-v2 (pretrained) | Unsharp masking demoted to secondary — perceptual trick, not true restoration |
| Motion blur | Blind deconvolution | DeblurGAN-v2 (same model handles both) | DeblurGAN-v2 handles Gaussian and motion blur in one model |
| Sensor noise | Non-local Means denoising | NLM | NLM is the practical choice; BM3D as quality ceiling comparison |
| Fog/haze | AOD-Net (lightweight learned dehaze) | — | Pretrained, no fine-tuning required |
| Combined | Denoise → CLAHE (sequential) | CLAHE → Denoise (reversed order) | Ordering ablation — see Section 6.4 |

## **6.3 Naive Baseline: Histogram Matching**

Histogram matching is added as a universal naive baseline across all degradation types. It normalizes the global intensity distribution of a test image to match a reference normal image from the training set. It is trivially implemented (one scipy function call), requires no training, and probably will not outperform CLAHE or NLM. Its value is as a reference tier — it lets you show a three-level table: raw degraded / naive preprocessing / principled preprocessing. That structure makes your results much more informative and gives reviewers an intuitive anchor.

## **6.4 Pipeline Ordering Ablation (Combined Case)**

For the combined low-light \+ noise case, two pipeline orderings are tested:

* **Denoise first, then CLAHE:** Remove noise before contrast enhancement. This is the theoretically correct order — CLAHE applied to a noisy image treats noise as local contrast and amplifies it.

* **CLAHE first, then denoise:** The naive order. Expected to perform worse on combined corruptions. Including it makes the recommendation more concrete: 'always denoise before enhancing contrast.'

This ablation costs almost nothing to run (same images, reversed pipeline) and produces a practically useful finding about pipeline design. It can be reported in a single paragraph and one additional table row.

## **6.5 What Is Not Included and Why**

* **Retinex:** Dropped in favor of CLAHE. Both target illumination normalization; CLAHE is faster, better documented, and easier to tune. Including both would muddy the comparison without meaningful benefit.

* **Unsharp masking:** Demoted from primary to secondary. It is a perceptual sharpening trick, not a true deblurring technique. Wiener deconvolution and DeblurGAN-v2 are more principled choices for the blur track.

* **DINOv2 backbone swap:** Excluded from the core plan. Swapping PatchCore's backbone changes the model, not just the preprocessing. This conflates two experimental variables. It can be noted as future work.

# **7\. Experimental Design and Rigor**

## **7.1 Metrics**

AUROC is the primary metric, consistent with all published baselines. However, two additional measurement layers are added:

* **False negative rate (FNR) vs false positive rate (FPR) separately:** AUROC is threshold-free and hides the direction of failure. In industrial deployment, FNR (defect missed, ships to customer) and FPR (false alarm, stops the line) have very different costs. Reporting both separately makes findings practically actionable. This requires choosing an operating threshold — use the standard practice of setting threshold at 95th percentile of normal training scores.

* **Score distribution analysis:** Preprocessing changes anomaly score distributions, not just AUROC. A preprocessing step can improve AUROC but shift the score distribution so that a previously calibrated threshold now produces many false positives. At least one degradation type should include a score distribution visualization (histogram or KDE plot) for clean / degraded / degraded+preprocessing conditions.

## **7.2 Statistical Reporting**

PatchCore includes a coreset subsampling step with random seed dependence. All experiments are run with 3 different random seeds, and results are reported as mean ± standard deviation. This is low effort (3x compute) and makes findings defensible against the obvious reviewer question: 'Are these differences statistically meaningful or just noise?'

## **7.3 Training Set Corruption Experiment**

The main experiment corrupts only the test set, keeping the training set (memory bank) clean. This is a valid starting point but misses a more realistic deployment scenario: a factory where images were collected under the same bad conditions that now affect inference.

A secondary experiment is run for one degradation type (low-light, as the most common real-world case) where both the training set and test set are corrupted at the moderate severity level, and preprocessing is applied to both. This addresses the question: 'If I deployed this system in a dark factory and collected my training images there too, does preprocessing still help?' Even a single degradation type done this way is enough to include in the discussion and preempts a likely reviewer objection.

# **8\. Revised 8-Week Execution Plan**

## **Week 1 — Environment, Baseline, and Severity Calibration**

Both teammates set up Anomalib. Download MVTec-AD from Kaggle. Run vanilla PatchCore and PaDiM on all 15 categories. Confirm PatchCore matches published 98%+ AUROC — this is the non-negotiable sanity check. Simultaneously, define and lock all corruption severity levels in a versioned config file. Do a quick visual inspection of 5 sample images per corruption type per severity level to confirm the levels feel proportionally spaced. Do not proceed to Week 2 until the config is finalized.

## **Week 2 — Corruption Pipeline and Dataset Generation**

Use an AI agent to write all corruption scripts. Generate the full degraded dataset as Kaggle output datasets so both teammates can import it and it is only generated once. Target: 5 corruption types (+ combined) × 3 severity levels \= 18 degraded dataset variants. Validate generated images visually before moving on. Store all variants with consistent naming convention keyed to the config file.

## **Weeks 3-4 — Main Degradation Experiments**

Split the 15 MVTec-AD categories and 18 degraded variants between the two teammates. Every result (AUROC, FNR, FPR per category per variant) goes into a shared spreadsheet immediately. By end of week 4, the full degradation curve exists — the central quantitative finding. Run with 3 seeds and record mean ± std. If WinCLIP is on track to be added, set it up during week 4 and run it on clean \+ the two worst-performing corruption types as a feasibility check.

## **Weeks 5-6 — Preprocessing Rescue Experiments**

Apply the full preprocessing stack (CLAHE, Wiener/DeblurGAN-v2, NLM/BM3D, AOD-Net, histogram matching as naive baseline) before PatchCore and PaDiM inference. Run on all degradation types, not just the worst-performing ones — this produces a complete recommendation matrix. Run the pipeline ordering ablation (denoise→CLAHE vs CLAHE→denoise) on the combined corruption case. Run the training-set corruption secondary experiment for low-light at moderate severity.

## **Week 7 — Analysis and VisA Generalization Check**

Run the top-performing preprocessing pipeline for each degradation type on VisA. This is the generalization validation. Simultaneously: identify which of the 15 MVTec-AD categories are most and least affected by each degradation type, and formulate a qualitative hypothesis for why (texture-heavy categories vs. structural categories). Generate score distribution plots for at least one degradation type. Write the qualitative analysis section — this is where human interpretation matters and AI agents help less.

## **Week 8 — Write-Up**

Both teammates write assigned sections. The paper structure should be: Abstract / Introduction (gap framing, three-paper argument) / Related Work / Experimental Setup / Results: Degradation Curves / Results: Preprocessing Recovery / Results: Failure Mode Analysis / Discussion (practical recommendations, limitations, future work) / Conclusion. AI agents assist with formatting, figure captions, and proofreading.

| *If WinCLIP was added: include a dedicated subsection in Results comparing CNN-embedding vs. vision-language robustness profiles. This is the strongest element of the paper if the data supports a clear finding.* |
| :---- |

# **9\. Related Work**

## **9.1 Papers That Establish the Research Gap**

These papers are the evidence base for the gap slide in your proposal presentation. Use them to argue the problem exists, not to describe the methods landscape.

| Paper | Venue | Role in Your Argument |
| ----- | ----- | ----- |
| Beyond Academic Benchmarks (Baitieva et al.) | arXiv 2503.23451, Mar 2025 | Primary gap evidence: 11 models, 9 datasets, documents the academia-industry AUROC collapse |
| Anomaly Detection for Industrial Applications: Challenges & Future Directions | arXiv 2501.11310, Jan 2025 | Lists adverse imaging conditions as explicitly open challenge |
| Unsupervised IAD Using Paired Well-lit and Low-light Images | Oxford JCDE, May 2025 | Confirms low-light is real problem; shows existing work is narrow (one condition, new architecture) |
| MVTec AD (Bergmann et al.) | CVPR 2019 | Establishes that the benchmark was designed under controlled conditions — the controlled baseline you exploit |
| PatchCore (Roth et al.) | CVPR 2022 | The model you are stress-testing; documented exclusively on clean studio images |

## **9.2 Methods Landscape Papers**

These go on a separate 'related methods' slide. They are not gap evidence — they show you know the broader field.

| Paper | Venue | Relevance |
| ----- | ----- | ----- |
| Deep Learning Advancements in AD: A Comprehensive Survey | arXiv 2503.13195, Mar 2025 | Methods taxonomy reference |
| WinCLIP (Jeong et al.) | arXiv 2303.14814, 2023/2024 | Vision-language baseline; potential third model |
| AnomalyGPT (Gu et al.) | arXiv 2308.15366, 2024 | LLM-based AD direction; positions your project on the practical end of the spectrum |
| Sequential PatchCore (Mao et al.) | ECCV 2024 Workshop | Adjacent real-world robustness work; surface impurities angle |
| FR-PatchCore | PMC 2024 | PatchCore generalization extensions |
| Comprehensive Survey: Real-World Industrial Defect Detection | arXiv 2507.13378, Jul 2025 | Most recent survey; cite for completeness |

| Do NOT use WinCLIP, AnomalyGPT, or PaDiM as background for your gap argument. They are methods papers proposing solutions to detection accuracy problems — not evidence that adverse conditions are understudied. Keep them strictly on the 'methods landscape' slide. |
| :---- |

## **9.3 Proposal Presentation: Three-Paper Gap Argument**

The gap can be established cleanly with three papers in sequence:

1. **Problem exists** (Valeo/Intel 2503.23451): models with 99.9% MVTec AUROC collapse on real-world data.

2. **Benchmark does not capture it** (MVTec-AD CVPR 2019): the dominant benchmark was designed under controlled conditions.

3. **One narrow attempt exists but is insufficient** (Oxford JCDE May 2025): only covers illumination for one condition, proposes new architecture rather than studying existing ones.

Therefore: a systematic study across multiple degradation types with preprocessing rescue analysis is needed. That is a three-paper argument for the gap, which is clean and defensible in front of any committee.

# **10\. Key Resources**

| Resource | Type | Link / Notes |
| ----- | ----- | ----- |
| Anomalib (OpenVINO) | GitHub | github.com/openvinotoolkit/anomalib — PatchCore, PaDiM, WinCLIP all included |
| MVTec-AD | Dataset | kaggle.com/datasets/ipythonx/mvtec-ad and mvtec.com/research-teaching/datasets/mvtec-ad |
| VisA | Dataset | paperswithcode.com/dataset/visa — for Week 7 generalization check |
| Awesome Industrial AD | GitHub tracker | github.com/M-3LAB/awesome-industrial-anomaly-detection — live paper tracker |
| PatchCore \+ CLIP implementation | GitHub | github.com/LuigiFederico/PatchCore-for-Industrial-Anomaly-Detection |
| DeblurGAN-v2 | GitHub | github.com/VITA-Group/DeblurGANv2 — pretrained model, handles Gaussian and motion blur |
| albumentations | Python library | pip install albumentations — primary corruption library |
| scikit-image | Python library | Wiener deconvolution, NLM denoising, histogram matching all built-in |
| AOD-Net | Pretrained model | https://github.com/weberwcwei/AODnet-by-pytorch |

# **11\. Scope Boundaries**

## **What This Project Does**

* Corrupts the MVTec-AD test set (and one training-set variant) with controlled synthetic degradations

* Runs PatchCore and PaDiM (+ optionally WinCLIP) on clean and all degraded variants

* Applies and evaluates classical and learned preprocessing as mitigation

* Reports AUROC, FNR, FPR per degradation type, severity level, and preprocessing strategy

* Validates top preprocessing recommendations on VisA

* Produces a practical deployment recommendation table

## **What This Project Does Not Do**

* Design or modify any anomaly detection algorithm

* Collect real-world factory images

* Train any model from scratch

* Swap PatchCore's backbone (conflates experimental variables)

* Use MVTec LOCO, MVTec AD 2, BTAD, or AeBAD as primary datasets

# **12\. Risks and Recommendations**

**Risk 1**: The "Combined Case" Complexity

Evaluation: Combining Low-light \+ Sensor Noise (Section 5.2) can lead to unexpected artifacts that PatchCore might find "anomalous" simply because the preprocessing didn't clean it perfectly.

**Recommendation**: Use the AI agents to generate a "Visual Grid" early (Week 2). If the preprocessed images look "off" to the human eye, the model will likely fail.

**Risk 2**: Dataset Size

Evaluation: Running 18 variants of the full MVTec-AD dataset will consume significant disk space on Kaggle.

**Recommendation**: Do not store the corrupted images. Write a "Transform Pipeline" that applies the corruption and preprocessing on-the-fly during inference. This saves hours of uploading/downloading data.

**Risk 3**: Metric Overload

Evaluation: Tracking AUROC, FNR, and FPR across 15 categories and 18 variants is a lot of data (400+ data points).

**Recommendation**: Use a shared Google Sheet or a tool like Weights & Biases (W\&B) from Day 1 to track results. If you wait until Week 8 to "find the trend," you will be overwhelmed

**Risk 4**: Low-light \+ Gaussian noise.

**The Issue:** Physically, low-light imaging *is* noise. When a sensor operates in low light, it suffers from **Poisson (shot) noise** due to low photon counts.

**The Risk:** By adding "Gaussian noise" on top of a "low-light" simulation that already includes Poisson noise, you might be creating a dataset so destroyed that no preprocessing can rescue it. This makes the "Ordering Ablation" (Section 6.4) less useful if the starting AUROC is already near-random (0.5). You should visually verify in Week 1 that the "Combined Case" still contains visible structures.

**Risk 5** — Wiener deconvolution requires kernel knowledge Section 6.2 lists Wiener deconvolution as the primary technique for Gaussian blur mitigation. Wiener deconvolution in scikit-image requires you to supply the blur kernel as an input parameter. In a synthetic corruption setting this is fine — you know exactly which kernel you used — but the document should make this explicit. If you ever want to generalize findings to real-world blur where the kernel is unknown, Wiener breaks down. 

**Solution:** The document should add one sentence: "Wiener deconvolution is applicable in this study because the blur kernels are known by construction in the synthetic corruption pipeline. In real-world deployment scenarios where the blur kernel is unknown, Wiener filtering may not be directly applicable; blind deconvolution methods would be required and are left as future work.”

**Risk 6:** DeblurGAN-v2 runtime on Kaggle Section 6.2 proposes DeblurGAN-v2 as a secondary blur mitigation technique. DeblurGAN-v2 is a learned model — inference is fast per image on a GPU, but it requires loading a pretrained model (\~140MB checkpoint) and running it on the full test set of MVTec-AD. The concern is not speed per image but setup friction: the VITA-Group GitHub repo has some version compatibility issues with newer PyTorch builds.

**Solution:** Verify it imports cleanly on a Kaggle T4 before committing to it in week 6\. If it fails to load, Wiener deconvolution \+ unsharp masking is an adequate substitute that uses only scikit-image. 

**Risk 7:** No Runtime Analysis

**Solution:** Add Runtime Analysis 

# **13\. Experimental Constraints & Assumptions**

This study evaluates anomaly detection robustness under controlled synthetic degradations. All corruptions are applied programmatically to ensure precise control over type and severity; however, such transformations may not fully capture real-world imaging artifacts, and results should be interpreted accordingly.

Certain preprocessing methods assume access to degradation parameters. In particular, Wiener deconvolution is applied with known blur kernels, which are available by construction in the synthetic setting. In practical deployments where degradation parameters are unknown, such methods may require replacement with blind alternatives.

All preprocessing is applied as a front-end transformation without modifying or retraining the underlying anomaly detection models. This isolates preprocessing effects and reflects deployment scenarios where model retraining is infeasible.

Experiments are conducted in a constrained environment (single GPU). Methods with high computational or integration overhead (e.g., BM3D, DeblurGAN-v2) are included selectively, and classical alternatives are used where necessary to ensure reproducibility.

Finally, while experiments are conducted primarily on MVTec-AD to maintain a controlled baseline, a secondary evaluation on VisA is used to assess generalization beyond a single dataset.

## **Limitations**

While this study provides a systematic evaluation of robustness under controlled degradations, several limitations should be acknowledged.

**Synthetic-to-real gap.**  
 All corruptions are synthetically generated and may not fully reflect real-world imaging conditions such as complex lighting variations, sensor-specific noise, or environmental artifacts. Although trends observed under controlled degradations are informative, absolute performance values may differ in deployment settings.

**Assumption of known degradation parameters.**  
 Some preprocessing techniques, such as Wiener deconvolution, rely on known degradation parameters (e.g., blur kernels). This assumption holds in the synthetic setting but limits direct applicability in real-world scenarios where such parameters are unknown.

**Limited model diversity.**  
 The primary evaluation focuses on embedding-based methods (PatchCore and PaDiM). While these are representative of current industrial practice, the findings may not fully generalize to fundamentally different paradigms (e.g., reconstruction-based or vision-language models). Optional inclusion of WinCLIP partially mitigates this but is not exhaustive.

**Restricted preprocessing search space.**  
 The study evaluates a targeted set of classical and lightweight learned preprocessing techniques. While these are chosen for practicality and interpretability, they do not cover the full space of modern restoration methods (e.g., diffusion-based or transformer-based approaches).

**Computational constraints.**  
 Experiments are conducted under limited computational resources (single GPU), which restricts the inclusion of more computationally intensive methods and large-scale hyperparameter exploration. Runtime measurements are therefore indicative rather than exhaustive.

**Dataset scope.**  
 The primary dataset (MVTec-AD) consists of clean, controlled images, which is necessary for isolating corruption effects but limits ecological validity. A secondary evaluation on VisA is included to partially address generalization, though broader validation across additional real-world datasets remains future work.

*feasibility: all components are open-source  •  datasets are freely available  •  runs on a single GPU  •  no new training required*

