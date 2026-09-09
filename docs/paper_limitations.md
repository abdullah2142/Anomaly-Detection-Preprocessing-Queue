# Limitations

## 1. Synthetic Corruption Gap

All five corruption types are synthetically generated using parameterized functions applied to clean MVTec-AD images. Real-world imaging degradation is more complex: manufacturing lighting variation involves spatial non-uniformity, reflections, and spectral shifts that a simple gamma-darkening model does not capture. Real sensor noise profiles are camera-specific and correlated across channels. Real fog and haze involve depth-dependent scattering that a uniform coefficient model approximates poorly. **Implication:** results represent a controlled lower bound on degradation diversity. Models may perform differently under real-world corruption even at matching perceptual severity.

## 2. Wiener PSF Specification

Wiener deconvolution uses the exact PSF that generated the blur at **every**
severity. The originally published mild and moderate rows were produced with the
severe-tier kernel hardcoded across all severities; all 1,104 affected rows were
regenerated with per-severity parameters and merged (see
`scripts/wiener_reruns/` and `scripts/analysis/merge_wiener_reruns.py`).

Real-world blur kernels are camera-, motion-, and scene-dependent and must be
estimated blindly. Our Wiener results are therefore an **upper bound** on
deconvolution performance: an oracle PSF still costs −5.7 pp (mild), −15.1 pp
(moderate) and −12.5 pp (severe) for Gaussian blur.

The superseded misspecified-kernel run is retained in git history and reported as
a deliberate sensitivity comparison: a kernel 5× too wide deepens the mild-severity
penalty from −5.7 pp to −32.2 pp. This quantifies the additional cost that blind
deconvolution — which must estimate the kernel — has to pay, and is a stronger
statement than the oracle result alone.

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

17.2% of all benchmark rows (1,612 / 9,384) exhibit AUROC = 0.5000 exactly, indicating fully tied anomaly scores with zero ranking ability. This fell from 23.1% (2,169 rows) when the corrected per-severity Wiener rows were merged: the misspecified severe-tier kernel had been collapsing 557 cells that the matched kernel leaves informative, which is itself evidence of how much of the original Wiener harm was kernel error rather than deconvolution. This is concentrated at severe corruption: 52.2% of severe Gaussian blur rows hit the floor. Reported mean AUROC values are therefore censored at a floor of 0.5 and may understate the true severity of model collapse under extreme corruption. Wilcoxon tests on rescue deltas also have reduced effective N where both degraded and rescued AUROC are 0.5 (zero differences are silently dropped): 489 such pairs remain, giving effective N of 645–806 against a nominal 810.

A second consequence: rescue *deltas* are censored at severe severity. Because
33% of severe degraded baselines already sit at 0.5, a rescue applied to them
cannot push the score lower, compressing the measured delta toward zero. Apparent
declines in rescue harm as severity rises — visible for motion-blur Wiener
(−12.2 pp mild, −10.6 pp moderate, −8.9 pp severe) — are therefore not evidence
that restoration becomes safer under worse degradation. They should not be read
as a severity trend.

## 10. Train/Test Resolution Mismatch

Training-time augmentation (`prepare_augmented_train_data`) applies corruption to raw-resolution images (700–1024 px for MVTec-AD, ~1500 px for VisA) which are subsequently resized to 256×256 by the datamodule. Test-time corruption (`CorruptedDatasetWrapper`) applies corruption to images that have already been resized to 256×256. The same nominal corruption parameters therefore produce substantially different effective severity: e.g., σ=25 Gaussian blur at 1024 px downsampled to 256 px is effectively ~σ=6. Augmented models were trained on milder effective corruption than they are tested on, which means the reported robustness gains may partially reflect generalization beyond the training distribution rather than exact distribution matching.

## 11. Matched Corruption Distributions (quantified by two controls)

Training augmentation draws from the same 5 corruption types × 3 severity levels
used for test-time evaluation, so the headline gains (+12.3 pp PatchCore,
+10.1 pp PaDiM) are a **corruption-matched oracle bound**. The benchmark alone
cannot separate genuine robustness from having trained on the test distribution.

Two controls separate them. Each withholds one condition from the augmented arm
and tests only on that withheld condition, over 8 MVTec-AD categories at seed 42,
with clean-trained and matched-augmented comparisons drawn from the same
categories and cells (full tables: [`generalization_controls.md`](generalization_controls.md)).

| Control | Withheld | PatchCore survives | PaDiM survives |
|---|---|---|---|
| Leave-one-corruption-out | one corruption type, rotating through all five | +3.58 pp of +10.66 pp (**34%**, p = 0.0312) | −0.22 pp of +8.02 pp (**0%**, p = 0.8281) |
| Severity holdout | the severe tier (trained mild + moderate) | +4.58 pp of +13.21 pp (**35%**, p = 0.0078) | +1.02 pp of +8.83 pp (**12%**, p = 0.4609) |

**Implication — the gain is part robustness, part distribution matching, and the
split depends on the detector.** PatchCore retains roughly a third of its benefit
against corruption types and severities it never saw, significantly in both
controls and consistently across categories (positive in 6 of 8). PaDiM retains
none against an unseen corruption type and nothing distinguishable from zero
against an unseen severity. Withholding a condition costs −7.1 to −8.6 pp in
every case, significant throughout.

Transfer also decays with severity (+2.88 pp mild, +1.63 pp moderate, +0.53 pp
severe, models pooled), so what does generalise generalises best where the
degradation is mildest.

Two caveats on scope. Both controls use a single seed and 8 of the 15 MVTec-AD
categories, chosen so that the exact permutation test's floor (2/2⁸ = 0.0078)
permits significance; they are not run on VisA. And the per-corruption
breakdowns rest on 8 categories each, so only the pooled and per-model figures
should be treated as well-powered.

**Deployment consequence:** augment with the corruptions and severities you
actually expect. Extrapolation beyond the augmented distribution buys little, and
for PaDiM it buys nothing.

## 12. Undocumented Salt-and-Pepper Component in Sensor Noise

The `apply_sensor_noise` function adds 5% salt-and-pepper impulse noise on top of the Gaussian noise specified in `experiment_config.json`. This impulse component is not reflected in the configuration file's `gauss_var` parameter. NLM denoising is suboptimal for impulse noise (median filtering is standard), which means the modest harm of NLM rescue (−1 to −3 pp) may partly reflect a rescue-method mismatch rather than a fundamental incompatibility between denoising and anomaly detection.

## 13. Oracle Corruption Identification

Rescue methods are selected by the corruption's ground-truth identity. In the
evaluation loop the rescue is looked up by corruption type, and the Wiener PSF by
severity:

```python
for ctype, severities in config["corruptions"].items():
    for sev in severities.keys():
        for r_name, r_func in get_rescue_map(sev, config)[ctype]:
```

No component inspects the image to decide whether it is degraded, which
degradation it carries, or how severe that degradation is. A deployed system
would require four inference steps our benchmark supplies for free: detect that
an image is degraded, classify the degradation type, estimate its severity, and
estimate the restoration parameters. Each step can fail, and a misclassification
applies a restoration matched to the wrong corruption.

**Implication:** this strengthens rather than weakens the central finding. Rescue
preprocessing is net-harmful *under oracle knowledge of corruption type, severity,
and restoration parameters* — the most favourable condition that can be
constructed.

**Now measured (§5.7).** The cost of identification error is no longer a
conjecture: applying the rescue for the wrong corruption costs −5.17 pp (PaDiM)
and −7.57 pp (PatchCore) against doing nothing, both $p = 0.0078$, and a further
−2.0 to −5.7 pp beyond the correct rescue. The damage concentrates in blur-for-blur
confusion (−15.84 pp and −11.48 pp), which is where a classifier is most likely to
err. Real-world rescue is therefore bounded above by our already-negative oracle
results, by a quantified margin. See [`unconditional_rescue.md`](unconditional_rescue.md).

## 14. Rescue Applied Only to Matched, Known-Degraded Inputs

Each rescue method is only ever applied to the corruption it targets: CLAHE and
Retinex to low-light images, Wiener to blurred images, NLM to noisy images, DCP
to foggy images. Two cases are therefore unmeasured:

1. **Mismatched application** — e.g. dehazing a motion-blurred image. A
   deployment that misclassifies the degradation lands here.
2. **Application to undegraded images** — no clean image in the benchmark is ever
   passed through a rescue function.

Case 2 matters for deployment advice. Preprocessing is normally applied
unconditionally to an entire image stream, most of which may be undegraded. Our
recommendation that CLAHE is conditionally safe for low-light PatchCore
implicitly assumes a low-light detector gates it.

**Implication:** the reported rescue deltas describe the best case for rescue —
correct method, correct target.

**Now measured (§5.7).** Case 2 is quantified and it is the larger hazard.
Applying the six rescues to undegraded images costs −14.66 pp (PaDiM) and
−15.12 pp (PatchCore) pooled, both $p = 0.0078$ — more than four times the
−3.35 pp penalty of matched restoration on genuinely degraded input. Wiener on an
unblurred image is catastrophic (−40.76 pp pooled). Unnecessary preprocessing, not
mistaken preprocessing, is the dominant deployment risk.

The recommendation this limitation threatened survives, but only where it was
made: CLAHE on undegraded images costs PatchCore −0.95 pp ($p = 0.2031$),
indistinguishable from zero, so it is safe to apply without a gating detector. The
same method costs PaDiM −4.92 pp ($p = 0.0156$), so the recommendation does not
generalise across architectures. NLM is likewise near-harmless (−0.09 pp). Case 1
(mismatched application) is covered under Limitation 13. Both rest on 8 categories
at a single seed. See [`unconditional_rescue.md`](unconditional_rescue.md).

## 15. Single Noise Realization in Training Augmentation

The augmentation pipeline calls `apply_corruption(...)` with a fixed seed for
every image (`seed=42` on MVTec-AD; `seed=rng_seed` on VisA). Because the noise
functions draw from `np.random.default_rng(seed)` at the image's shape, every
augmented image of a given corruption type within a category receives the
**identical** noise field or fog pattern. Test-time corruption, by contrast, uses
`seed=42+idx`, which varies per image.

Blur corruptions are deterministic and unaffected; this applies to
`sensor_noise`, `fog_haze`, and partially to `low_light`.

**Implication:** the augmented models saw one fixed realization of each stochastic
corruption rather than a distribution, so they had less variation to generalise
over than the test set presents. This biases the measured augmentation gain
**downward**, making the reported +12.3 / +10.1 pp conservative with respect to
this specific defect.

## 16. Augmented Training Data Is Seed-Invariant on MVTec-AD

On MVTec-AD, `prepare_augmented_train_data()` is invoked once at module scope
with the default `rng_seed=0`, and skips any file already written. All three
seeds therefore train on byte-identical augmented images; seed variation captures
only model initialisation and coreset sampling, not augmentation sampling. VisA
does not share this defect — it passes `rng_seed=seed` per run.

**Implication:** reported seed-to-seed variance on the MVTec-AD augmented arm
understates the true variance of the augmentation procedure. Inference in this
paper clusters by category rather than by seed, so significance is unaffected,
but per-cell standard deviations on that arm should not be read as full
procedural variance.

> **Not a limitation:** the high rate of AUROC values identical across all three
> seeds (up to 50% on VisA PatchCore clean-trained) is a consequence of floor
> saturation, not of seed invariance. Excluding cells pinned at AUROC = 0.5,
> the identical-across-seeds rate falls to 0.0–3.7% in every condition. See
> Limitation 9.

## 17. The Fallacy's Mechanism Is Established Only as Failure-to-Return

The *preprocessing fallacy* is measured on the detectors' own distance-from-normality
scores rather than inferred from AUROC ([`feature_space_evidence.md`](feature_space_evidence.md)).
That measurement supports the claim that restoration does not return images to the
clean distribution: on normal test images, restored inputs sit +0.396 from the
clean baseline ($p = 0.002$, 10/10 units), statistically indistinguishable from the
corrupted inputs they came from.

It does **not** support the stronger form we originally asserted — that restoration
lands images *further* from normal, in a third distribution of restoration
artifacts. That difference is −0.002 ($p = 0.914$, 5/10 units), and it is not a
clipping artefact: the ceiling-share test ($p = 0.97$) and the low-clipping subset
($p = 0.63$) agree. The claim has been withdrawn from the abstract, §5.4 and §6.1.

Three scope limits on the probe itself. It covers 5 categories at one seed and one
severity (moderate). Scores were subject to the library's normalisation, which
clipped 46% of them at 1.0 — the ceiling-robust checks agree with the main result,
but an unclipped rerun would sharpen the estimate. And most importantly, **distance
is not direction**: a scalar score cannot distinguish "the same distance from
normal" from "a distinct region of feature space at the same radius". Testing
distinctness requires embedding geometry, and remains future work.

**Implication:** the deployment-relevant half of the fallacy is demonstrated —
restoration does not undo the distribution shift, so preprocessing cannot be
assumed to recover clean-image performance. The mechanistic story about *why* the
AUROC penalty arises is not settled by this evidence, and the paper no longer
claims otherwise.
