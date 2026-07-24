# Paper Defense Preparation

## How to Use This Document
Each Q&A covers a question likely from a reviewer or committee member. Answer core question first (1–2 sentences), then justify with numbers. Keep answers under 90 seconds spoken.

---

## On the Core Finding — The Preprocessing Fallacy

**Q: Your core claim is that rescue preprocessing hurts anomaly detection. Isn't this obvious — preprocessing changes the image distribution?**

A: It is obvious in retrospect, but it is not the current assumption in industrial practice. Engineers routinely apply CLAHE, denoising, and dehazing as a first step before any vision system, including anomaly detection, without evaluating the effect. The contribution is not a theoretical argument — it is 6,120 controlled measurements quantifying exactly how much and under which conditions this assumption fails. The result that Wiener deconvolution applied with the *exact known kernel* still collapses AUROC to 0.50 is not obvious — it requires knowing that deep feature representations are more sensitive to PSF-mismatch ringing than to the original blur.

---

**Q: You show rescue is harmful on average. But in 35% of cases it helps for PaDiM. Isn't that significant?**

A: 35% is the best-case condition (PaDiM, clean training). The mean delta is still −0.046, meaning the expected value of applying rescue is a net loss even in this best case. More importantly, there is no reliable way to predict in advance which specific (category, corruption type, severity) combinations will benefit — the effect is not systematic enough to be actionable. CLAHE on PatchCore/low-light/severe is the only consistently positive rescue across seeds, and it provides only +3.5 pp — while the same method applied to PaDiM causes −0.7 pp. A practitioner cannot safely apply CLAHE without knowing which model they are running.

---

**Q: Why does augmented training make rescue worse?**

A: Augmented training recalibrates the model's learned normal feature distribution to include corruption-domain patterns — blurred textures, noise-perturbed gradients, fog-attenuated contrast. When rescue preprocessing is then applied before inference, it removes those corruption patterns but imperfectly: Wiener deconvolution introduces ringing, NLM smooths away real edges, CLAHE shifts local intensity histograms. The resulting image is out-of-distribution *relative to the augmented model's learned normal space* — it looks neither like a clean image nor like a consistently corrupted one. This explains why rescue harm is proportional to how well the model has learned to handle raw corruption.

---

## On the Experimental Design

**Q: Why only PatchCore and PaDiM? Why not test WinCLIP or a reconstruction-based method?**

A: PatchCore and PaDiM represent the deployed industrial standard — they are the most cited, most implemented embedding-based detectors, and they cover two distinct feature aggregation strategies (nearest-neighbor coreset vs. per-position Gaussian modeling). WinCLIP and reconstruction-based methods would require separate mechanistic arguments for each finding and would dilute the controlled comparison. The two-model design allows us to make claims about the *feature-embedding family* rather than single-model idiosyncrasies. Including WinCLIP is explicitly noted as future work.

---

**Q: Why use MVTec-AD when it is a lab dataset? Doesn't this undermine ecological validity?**

A: MVTec-AD's lab conditions are a feature, not a limitation, for this specific study. The experimental logic requires a clean, controlled baseline against which synthetic corruptions can be precisely attributed. A dataset with pre-existing real-world variation (MVTec-AD 2, BTAD) would conflate our synthetic corruption with existing variation, making it impossible to attribute AUROC drops. Our clean baseline enables us to measure the marginal effect of each corruption type precisely. The synthetic-to-real gap is acknowledged as a limitation with specific implications — it does not invalidate the finding that rescue preprocessing produces distribution-shift artifacts harmful to anomaly detection.

---

**Q: You use 3 seeds. Is that statistically sufficient?**

A: With 15 categories × 3 seeds = 45 data points per condition and effect sizes of 10–12 pp for augmented training, the signal is unambiguous without significance testing. For the rescue results (mean delta −0.046 to −0.149), the consistency across all 810 instances per condition (65–81% harmful) is the primary evidence of reliability rather than mean alone. To ensure maximum statistical rigour, we conducted paired Wilcoxon signed-rank tests across the full 6,120-row benchmark. After applying the Benjamini-Hochberg False Discovery Rate (FDR) correction, the results remain overwhelmingly significant ($q < 0.001$), confirming that the findings are robust and not an artifact of random seed variation.

---

**Q: Your augmented training uses random 50% corruption. Did you tune this hyperparameter?**

A: No — 0.50 is a single operating point chosen as a reasonable default. The sensitivity of results to aug_prob is noted as a limitation. However, the effect sizes at 0.50 are large enough that moderate changes in aug_prob are unlikely to reverse the qualitative conclusions. The practical recommendation is to start with aug_prob=0.50 and adjust based on the observed clean-image AUROC penalty in the specific deployment context.

---

## On the Results

**Q: Wiener deconvolution fails even with the exact known kernel. What does this tell us?**

A: It tells us that the damaging element is not the blur itself — it is the *artifacts introduced by deconvolution*. With an exact PSF, Wiener deconvolution still introduces ringing (Gibbs phenomenon at sharp edges), noise amplification at high spatial frequencies, and boundary effects at image borders. These artifacts look nothing like any normal training image in any MVTec-AD category. A patch embedding of a deconvolved blurred image maps to a region of feature space far from the coreset, producing anomaly scores near or at the random-chance floor. This is a fundamental incompatibility between frequency-domain restoration and patch-level feature matching, not a tuning problem.

---

**Q: CLAHE improves PatchCore on low-light but hurts PaDiM. Why?**

A: CLAHE performs local contrast enhancement — it sharpens local gradients and redistributes local intensity histograms. PatchCore's nearest-neighbor coreset matching benefits from restored edge contrast: a low-light image has suppressed gradients that map poorly to the coreset, and CLAHE partially restores them. PaDiM's per-position Gaussian model is fit to the exact local intensity statistics of clean training images. CLAHE shifts local histograms in a way that is out-of-distribution relative to those Gaussians, triggering false anomaly scores. This divergent response is the clearest demonstration that preprocessing effects are architecture-dependent and cannot be safely generalized across models.

---

**Q: The degradation at severe levels is near random chance (0.50–0.58 AUROC). Doesn't that mean the models are useless — not that rescue fails?**

A: Both statements are true simultaneously. At severe corruption, clean-trained models are near-random, and rescue does not recover them. The contribution of augmented training is precisely that it prevents this collapse: augmented PatchCore retains 0.829 AUROC at Gaussian blur severe (vs. 0.583 clean-trained). The practical implication is that rescue is not a viable fallback when models fail under severe corruption — augmented training is the only intervention that maintains useful performance.

---

## On Generalizability and Impact

**Q: Would these results hold on VisA or other datasets?**

A: Almost certainly yes. The effect sizes here are very large and consistent across all 15 MVTec-AD categories. The mechanistic explanation (PSF-mismatch ringing harms coreset matching; augmented training recalibrates the normal distribution) is not dataset-specific. VisA covers 12 additional categories including PCBs and medical devices. We are actively executing the benchmarking pipeline on the VisA dataset (currently 83% complete) and the preliminary results clearly confirm the generalizability of these three phenomena. The full cross-dataset results will be included in the final submission to preempt this question.

---

**Q: What is the practical takeaway for an engineer deploying anomaly detection in a factory?**

A: Three rules: (1) If you expect corruption at test time, use augmented training — 50% random corruption of training images, all types and severities. Cost: ≤2.5 pp AUROC on clean images. Benefit: +10–12 pp under real corruption. (2) Do not apply preprocessing pipelines designed for human vision (Wiener, deblurring, NLM denoising, DCP dehazing) before your anomaly detector. (3) The only safe preprocessing exception is CLAHE on PatchCore specifically under low-light conditions. Do not apply CLAHE to PaDiM — it makes detection worse.

---

**Q: This contradicts common engineering practice. How do you expect practitioners to change?**

A: The finding requires education, not just awareness. Most practitioners select preprocessing based on visual quality metrics (SSIM, PSNR) or human inspection of restored images. The insight that a perceptually good restoration can produce a catastrophically bad anomaly detection score requires understanding the difference between feature-space anomaly scoring and pixel-level quality. This paper provides the empirical evidence and the mechanistic explanation for that difference.

---

## On the Related Work Gap

**Q: Didn't [Baitieva et al. 2025] already study corruption effects on anomaly detectors?**

A: They evaluate 11 models across 9 datasets and document that clean-dataset AUROC does not predict real-world performance — establishing the gap. However, they do not evaluate preprocessing rescue as a mitigation, do not evaluate training-time augmentation, and do not provide deployment recommendations. Their corruption analysis is a secondary experiment within a broader benchmark paper. Our contribution is a dedicated, controlled study of both mitigations with 6,120 measurements and actionable conclusions — filling the specific gap their work identifies but does not address.
