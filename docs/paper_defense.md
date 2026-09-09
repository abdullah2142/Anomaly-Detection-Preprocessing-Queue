# Paper Defense Preparation

## How to Use This Document

Each entry is a question a reviewer or committee member is likely to ask. Answer
the core question in one or two sentences, then justify with numbers. Keep spoken
answers under 90 seconds.

Every figure quoted here is generated, not typed. The authoritative sources are
[`statistical_validation.md`](statistical_validation.md),
[`per_corruption_tables.md`](per_corruption_tables.md),
[`generalization_controls.md`](generalization_controls.md),
[`feature_space_evidence.md`](feature_space_evidence.md) and
[`unconditional_rescue.md`](unconditional_rescue.md). If a number here disagrees
with one of those, the generated file is right and this document is stale.

**Section 1 contains the questions most likely to sink the paper.** Rehearse those
first. They concern errors we found and corrected in our own work; a reviewer who
finds them before we disclose them will assume there are more.

---

## 1. The Hard Questions — Errors We Found in Our Own Work

**Q: You published Wiener numbers that were wrong. Why should we trust anything else in this paper?**

A: Because we found it ourselves, fixed it, and can show exactly what it did and did not touch. The rescue map hardcoded the severe-tier point-spread function ($\sigma=25$, $k=281$; motion $k=151$) across all severities, so mild and moderate Wiener rows measured kernel *misspecification* rather than oracle deconvolution. The defect entered on a specific date (commit `21ce8ef`, 2026-05-30) when a working per-severity implementation was replaced during a rewrite; it is recoverable from git history.

We re-ran all 1,104 affected rows with per-severity PSFs and merged them under a script that asserts a 1:1 join before writing (`scripts/analysis/merge_wiener_reruns.py`). 872 of 1,104 values changed. The correction moved pooled rescue harm by +1.4 to +3.2 pp per condition and did not change the direction of any MVTec-AD finding — with one exception we report rather than bury: clean-trained VisA PaDiM moved from −2.74 pp to −0.61 pp and is no longer distinguishable from zero.

The scope of the bug was bounded and verifiable: it touched only Wiener rows at mild and moderate severity, 22.2% of rescue rows and 11.8% of the database. Every other row is untouched, and the degradation rows the reruns re-derived reproduce the originals.

---

**Q: Your original significance tests were wrong by orders of magnitude. What else is wrong?**

A: The original tests treated every (category, seed, corruption, severity) cell as an independent paired sample, N = 675. They are not independent — each category contributes up to 45 cells. Clustering to the 15 genuinely independent categories and running an exact permutation test moves the augmentation-gain p-value from $3.25\times10^{-69}$ to $1.2\times10^{-4}$.

The important part: **not one point estimate moved.** +12.3 pp is +12.3 pp under either analysis. Pseudoreplication inflates confidence, not effect size, so the correction cost us an overstated p-value and nothing else. Every finding that was significant at the cell level remains significant when clustered, on MVTec-AD at $p \le 2.1\times10^{-3}$.

What it did change is which claims we are allowed to make about VisA. The augmented VisA arm covers 4 categories, where the smallest attainable two-sided p is $2/2^4 = 0.125$. No result on that arm can reach conventional significance regardless of effect size, so we now report those gains descriptively. We previously claimed $p < 10^{-27}$ for them, which was unsupportable.

---

**Q: You named your contribution the "preprocessing fallacy" and defined it as a mechanism. Did you demonstrate that mechanism?**

A: Half of it, and we corrected the other half rather than defending it. AUROC cannot test a claim about where images sit in feature space — it measures only the ranking of scores. The anomaly score *is* a distance from learned normality (nearest-neighbour distance to the coreset for PatchCore, Mahalanobis distance for PaDiM), and our pipeline discarded it, keeping only `image_AUROC`.

We recorded the raw score for normal test images under clean, corrupted and restored inputs (5 categories × 2 models, moderate severity). Restored images sit +0.396 from the clean baseline ($p = 0.002$, 10/10 units) — restoration does **not** return them to the clean distribution, which is the deployment-relevant half and is now measured rather than asserted. But they are not measurably *further* from normal than the corrupted images they came from (−0.002, $p = 0.914$, 5/10 units). We had asserted that stronger form in the abstract; it is now removed.

This is not a clipping artefact. 46% of scores sit at the normaliser's ceiling, and both ceiling-robust checks agree with the main test: the share of images pinned at 1.0 ($p = 0.97$) and the subset of units with under 20% clipping ($p = 0.63$).

One thing remains genuinely open, and we say so: distance is not direction. A scalar score cannot rule out restored images occupying a *distinct* region of feature space at the same radius. Testing that needs embedding geometry, not the score.

---

**Q: There is one condition where your central claim fails. Why is it still a general finding?**

A: Clean-trained VisA PaDiM sits at −0.61 pp, $p = 0.375$ — indistinguishable from neutral. We report it in the abstract-level claims, not in a footnote, and we do not claim the fallacy universally.

The finding is general over the space we can defend: all four MVTec-AD conditions ($p \le 2.1\times10^{-3}$, 15 categories) and clean-trained VisA PatchCore (−3.73 pp, $p = 4.9\times10^{-4}$, 12 categories). That is two datasets, two architectures, three seeds. The exception is a single model-dataset-training cell, and it became an exception only after we corrected the Wiener PSF — before the correction it looked significant, and reporting it as such would have been reporting an artefact.

---

**Q: Your effect sizes are unusually large. Is something broken in your pipeline?**

A: The clean baselines rule that out. PatchCore reaches 0.982 and PaDiM 0.916 image-level AUROC on MVTec-AD, both matching published values for these architectures on this dataset. A pipeline with a systematic defect would not reproduce the literature on the clean condition and then diverge only under corruption.

The corruptions are genuinely severe by design: severe Gaussian blur is $\sigma=25$ with a 281×281 kernel on a 256×256 image, which is close to convolving the image into a constant field. We report that 52.2% of severe Gaussian rows land at exactly AUROC 0.5 for precisely this reason.

---

## 2. On Metrics and Measurement

**Q: 17% of your rows are exactly 0.5. Isn't your metric saturated to the point of meaninglessness?**

A: 1,612 of 9,384 rows sit at exactly 0.5, concentrated at severe corruption (52.2% of severe Gaussian blur). This is a real censoring effect and we report it rather than smoothing over it. Three consequences, all disclosed:

Reported severe-tier means are censored at a floor, so the true collapse is worse than the number suggests — the bias runs against the corruption being *less* damaging than we claim, not more. Second, `scipy.stats.wilcoxon` silently discards zero differences, so effective N is 645–806 rather than the nominal 810, and we report the effective figure. Third, rescue deltas at severe are compressed toward zero because a score already at the floor cannot fall further; this is why we attribute the apparent decline in rescue harm at high severity to censoring rather than to restoration becoming safer.

The findings do not rest on the censored cells. Mild and moderate rows are largely unaffected, and the clustered tests aggregate to category means where the floor is diluted.

---

**Q: Why image-level AUROC only? No pixel-level metrics, no PRO?**

A: The research question is whether preprocessing helps or harms detection, and image-level AUROC is the decision-relevant quantity for the deployment scenario we study: does the system flag this part or not. Pixel-level metrics would characterise *where* the model's attention moves under restoration, which is a mechanistic question our feature-space probe addresses more directly.

AUROC is also threshold-free, which matters here: any threshold-dependent metric would require a threshold selection policy, and that policy would interact with the corruption in ways that confound the comparison. Adding pixel-level and PRO metrics is a straightforward extension, and we note it as such.

---

**Q: Corruptions are synthetic. Why should any of this transfer to a real factory?**

A: It transfers as a lower bound on the problem, not as a simulation of it. Synthetic corruption is applied uniformly with known parameters; real degradation is spatially non-uniform, camera-specific and compound. Our numbers therefore describe the *easy* case for restoration — known corruption type, known severity, known parameters, uniform application.

Every one of those advantages is one a real system does not have, and we now quantify what losing them costs. Applying the correct rescue to a genuinely degraded image costs −3.35 pp; applying it to an image that needed no treatment costs −14.7 to −15.1 pp; applying the wrong one costs a further −5.2 to −7.6 pp. Real deployment sits in the harder region, so the synthetic setting understates the risk rather than manufacturing it.

---

## 3. On Experimental Design

**Q: Your augmentation draws from the same corruption types and severities you test on. Haven't you just trained on the test distribution?**

A: Partly, and we measured exactly how much. Two withholding controls train the augmented arm with one condition removed and test only on that removed condition, over 8 MVTec-AD categories. Leave-one-corruption-out withholds a corruption type; the severity holdout trains on mild and moderate and tests on severe.

The answer differs by architecture. PatchCore retains 34% (+3.58 pp of +10.66 pp, $p = 0.031$) and 35% (+4.58 pp of +13.21 pp, $p = 0.008$) of its matched gain — significant in both, positive in 6 of 8 categories. PaDiM retains nothing distinguishable from zero: −0.22 pp ($p = 0.828$) against an unseen corruption type, +1.02 pp ($p = 0.461$) against an unseen severity.

So the augmentation benefit is part genuine robustness and part distribution matching, and the split depends on the architecture. We think that is more useful than a flat answer: PatchCore buys margin against degradations you did not anticipate; PaDiM must be augmented with the corruptions you actually expect. The controls use a single seed and 8 of 15 categories, chosen so the exact permutation test can reach significance, and were not repeated on VisA — we state the result at the power available.

---

**Q: How does your pipeline know which rescue method to apply to a given test image?**

A: It does not — we tell it. The rescue is selected by the corruption's ground-truth type and the Wiener PSF by its ground-truth severity; nothing inspects the image. This is deliberate, and it makes the finding stronger rather than weaker. A deployed system needs four steps we supply for free: detect that an image is degraded, classify which degradation, estimate the severity, then estimate the restoration parameters.

We have since measured what losing the last two costs. Misidentifying the degradation costs −5.17 pp (PaDiM) and −7.57 pp (PatchCore) against doing nothing, concentrated in blur-for-blur confusion (Gaussian treated as motion: −15.84 pp). Applying restoration to an undegraded image — what unconditional preprocessing does to most of a real stream — costs −14.66 pp and −15.12 pp, over four times the penalty of correctly matched restoration.

The correct reading is therefore: rescue preprocessing is net-harmful *even under oracle knowledge of corruption type, severity and parameters*, and any real pipeline must absorb identification error on top of that. Our negative numbers are an upper bound on real-world rescue performance.

---

**Q: Your augmented training data is identical across all three seeds. Doesn't that invalidate your variance estimates?**

A: On MVTec-AD it is, and we disclose it. `prepare_augmented_train_data()` is invoked once at module scope with a fixed RNG seed and skips files already written, so all three seeds train on byte-identical augmented images; seed variation captures model initialisation and coreset sampling but not augmentation sampling. VisA does not share this defect — it passes the run seed through.

It does not affect the inference, because we cluster by category rather than by seed: categories are the independent unit, and the exact permutation test operates on category means. What it does affect is per-cell standard deviations on that arm, which understate the augmentation procedure's true variance. We say so rather than reporting those as if they were full procedural variance.

A related defect, also disclosed: the augmentation calls the corruption function with a fixed per-image seed, so every augmented image of a given stochastic corruption carries the identical noise field. This reduces training diversity relative to the test set, which biases the measured augmentation gain **downward** — it makes +12.3 pp conservative, not inflated.

---

**Q: Isn't the high rate of identical AUROC values across seeds evidence your seeds do nothing?**

A: That was our initial reading too, and it is wrong. Up to 50% of cells are identical across all three seeds for VisA PatchCore clean-trained, which looks alarming. Excluding cells pinned at the AUROC floor of 0.5, the identical rate falls to 0.0–3.7% in every condition. The apparent seed-invariance is floor saturation — two runs agreeing that a collapsed cell is 0.5 — not a seeding failure. The seeds do vary the models.

---

**Q: Your training augmentation is applied at full resolution but tested at 256 px. Isn't that a confound?**

A: It is, and it is disclosed. Training augmentation runs on raw files (700–1024 px for MVTec, ~1500 px for VisA) which are then resized to 256; test corruption is applied post-resize at 256. The same nominal parameters therefore produce different effective severity — a $\sigma=25$ blur at 1024 px downsampled to 256 is roughly $\sigma=6$.

The direction matters: the augmented models trained on *milder* effective corruption than they were tested against. That biases the augmentation gain downward, so the confound works against our own claim rather than for it. It also means the "corruption-matched oracle" framing is, if anything, an overstatement of how well matched the two distributions are. Fixing it would require re-running the entire augmented arm, and we judged the disclosed conservative bias preferable to a re-run that could only increase our reported gains.

---

**Q: Why only PatchCore and PaDiM? Why not WinCLIP or a reconstruction-based method?**

A: PatchCore and PaDiM are the deployed industrial standard and cover two distinct feature-aggregation strategies — nearest-neighbour coreset versus per-position Gaussian. That contrast turned out to be load-bearing: the two architectures behave *differently* in our generalization controls, with PatchCore transferring to unseen corruptions and PaDiM not. A single-architecture study would have produced a false general claim.

Reconstruction-based methods and vision-language models would each require their own mechanistic argument and would dilute the controlled comparison. We claim results for the feature-embedding family and note the extension explicitly.

---

**Q: Why MVTec-AD, a lab dataset? Doesn't that undermine ecological validity?**

A: The controlled baseline is what makes the attribution possible. A dataset with pre-existing real-world variation would conflate our synthetic corruption with that variation, making the AUROC drop unattributable. Starting from a near-perfect clean baseline lets us measure the marginal effect of each corruption exactly.

We then test whether the conclusions survive a second domain: VisA, 12 categories including PCBs, which is structurally very different. The preprocessing fallacy replicates there for PatchCore ($p = 4.9\times10^{-4}$) and fails for PaDiM ($p = 0.375$), and we report both.

---

**Q: You use 3 seeds and one augmentation probability. Is that enough?**

A: For the seeds, yes, because seeds are not our unit of inference. Significance is computed over 15 category means with an exact permutation test; the seeds contribute to each category's mean rather than serving as independent samples. Effect sizes of 10–12 pp against a floor of $p = 6.1\times10^{-5}$ leave ample margin.

For `aug_prob = 0.50`, it is a single operating point and we declare it as a limitation. Ablating it would require retraining the entire augmented arm three more times for what is an engineering optimisation rather than a scientific claim: the finding is that augmented training works and that its benefit is partly corruption-matched, neither of which turns on the exact probability.

---

## 4. On the Findings Themselves

**Q: Rescue helps in 38% of PaDiM cases. Isn't that worth having?**

A: Only if you can predict which 38%, and you cannot. The mean delta in that best case is still −0.028, so the expected value of applying rescue is a net loss even there. The beneficial cells are not systematic across category, corruption type or severity, which means there is no rule a practitioner could follow to capture them.

The one exception is genuine and we state it: CLAHE on PatchCore under low-light at severe corruption is consistently positive at +3.54 pp. The same method on PaDiM costs −0.71 pp. A practitioner cannot apply CLAHE safely without knowing which architecture they are running, which is a narrower recommendation than "CLAHE helps".

---

**Q: Your one positive recommendation assumes a detector that tells you the image is dark. Does it survive without one?**

A: We tested exactly that. Applied to undegraded images with no gating detector, CLAHE costs PatchCore −0.95 pp ($p = 0.2031$) — indistinguishable from zero, so the recommendation stands unconditionally for that architecture. It costs PaDiM −4.92 pp ($p = 0.0156$), so we confine the recommendation to PatchCore.

NLM denoising is likewise near-harmless applied blind (−0.09 pp). Every other method is significantly harmful on images that needed no treatment, with Wiener on an unblurred image at −40.76 pp pooled.

---

**Q: Wiener fails even with the exact kernel. What does that tell us?**

A: That the damage is intrinsic to deconvolution, not to getting the kernel wrong. With a per-severity oracle PSF, Wiener still costs −5.7 pp (mild), −15.1 pp (moderate) and −12.5 pp (severe) on Gaussian blur. Deconvolution amplifies high-frequency content into structured ringing at every edge; PatchCore scores by nearest-neighbour distance in a patch embedding space built from clean images, and ringing produces embeddings unlike anything in the coreset.

The misspecified run is now a deliberate sensitivity comparison rather than an embarrassment: a kernel 5× too wide deepens the mild-severity penalty from −5.7 pp to −32.2 pp. That prices the kernel-estimation error blind deconvolution must incur, and it is a stronger statement than the oracle result alone.

---

**Q: Why does augmented training make rescue *worse*?**

A: Augmented training expands the learned normal distribution to include corruption-domain patterns. Restoration then removes those patterns imperfectly, producing an image that matches neither the clean nor the corrupted manifold the model has learned. Rescue harm is consequently proportional to how well the model has learned to handle raw corruption: PaDiM worsens from −0.028 to −0.096, PatchCore from −0.039 to −0.118.

This also yields a clean deployment rule — augmentation and rescue are substitutes, not complements. Choosing both is worse than choosing either.

---

**Q: At severe corruption your models are near chance. Aren't they simply useless there, making the rescue comparison moot?**

A: Near-chance performance is the finding, not a flaw in the comparison. It establishes that these detectors fail outright under conditions a factory floor can produce, which is the motivation for the paper. The rescue comparison remains meaningful because both arms face the identical input: the question is whether restoration recovers any of the lost performance, and the answer is that it does not, and typically costs more.

We are careful not to over-read the severe tier, because scores there are censored at the AUROC floor. That is why the headline claims rest on the pooled and per-severity results, and why we attribute the flattening of rescue harm at severe to censoring rather than to safety.

---

## 5. On Positioning and Practice

**Q: Didn't Baitieva et al. (2025) already study corruption effects on anomaly detectors?**

A: They documented an academia–industry performance gap across 11 detectors and 9 datasets, establishing that the problem exists. They did not evaluate mitigation. Our contribution is the controlled comparison of the two candidate mitigations — test-time restoration versus training-time augmentation — under matched conditions, plus the negative result on restoration. The closest corruption benchmark, MVTec-C, evaluates 8 corruption types without studying any mitigation strategy.

---

**Q: What is the practical takeaway for an engineer deploying anomaly detection?**

A: Three rules, in order of confidence.

Augment your training data with the degradations you actually expect. The gain is +10 to +12 pp, and if you use PatchCore roughly a third of it extends to degradations you did not anticipate. With PaDiM, none of it does, so your augmentation set must cover your real failure modes.

Do not apply restoration preprocessing before an anomaly detector. It is net-harmful in every condition we measured, even with perfect knowledge of what went wrong and how to fix it.

If you must preprocess — because the same stream feeds a human inspector, say — only CLAHE and NLM are safe to apply blind, and only for PatchCore. Wiener deconvolution on an image that did not need it costs 40 points of AUROC.

---

**Q: This contradicts standard engineering practice. Why would practitioners believe it?**

A: Because the result is counterintuitive in a specific, checkable way: restoration improves images by every perceptual metric while degrading anomaly detection. An engineer can verify our central claim cheaply in their own pipeline by measuring detection AUROC with and without their existing preprocessing step, on their own data, without retraining anything.

The mechanism also gives a reason to expect it rather than a bare correlation: these detectors score by distance from a learned distribution of clean images, and restoration produces images that are not in that distribution — we measured that restored images remain as far from learned normality as the damaged ones. Optimising for human perception and optimising for distributional proximity are different objectives, and this paper quantifies the gap between them.

---

**Q: Can we reproduce this?**

A: Every number in the paper regenerates from the committed CSV by script:
`cluster_robust_stats.py` (significance), `build_per_corruption_tables.py`
(per-condition tables), `analyze_generalization.py` (withholding controls),
`analyze_feature_space.py` (mechanism probe), `analyze_unconditional_rescue.py`
(deployment controls) and `build_figures.py` (all data-derived figures). Each
writes a generated Markdown file that is byte-stable on re-run.

The experiment notebooks are Kaggle-ready with resume logic, and the README
records which shard produced which rows, including that the committed clean
notebooks are consolidations of split shards recoverable from git history. Known
gaps are declared: `requirements.txt` pins versions but the original runs
straddled two anomalib APIs, and the raw datasets are external.
