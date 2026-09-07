# Response to Professor's Review & Project To-Do List

## Part 1: Strategic Responses to Professor's Critiques

This document outlines the defense strategy against the six methodological critiques raised by the professor. It maps out which points are already completed, which are actionable next steps, and which we will strategically defend against based on theoretical bounds and computational limits.

### 🟢 Group A: Already Completed (Points 1 & 6)

**1. Statistical Significance (Paired Tests & Confidence)**
* **The Critique:** Relying on average AUROC values is insufficient; paired statistical tests (Augmented vs Clean, Rescued vs Degraded) are required.
* **Our Response:** We agree, and have gone further. Paired tests are computed on exact paired deltas, and significance is reported **clustered by category** — the level at which observations are independent — using an exact two-sided permutation test over category means. Our earlier cell-level Wilcoxon tests were pseudoreplicated (15 categories resampled 45 times each), which inflated the p-values by roughly 65 orders of magnitude without changing a single point estimate. The corrected inference is in [`statistical_validation.md`](statistical_validation.md): augmented-training gains on MVTec-AD hold at $p = 6.1\times10^{-5}$ (PaDiM) and $p = 1.2\times10^{-4}$ (PatchCore), and all four MVTec-AD rescue conditions remain significantly negative ($p \le 2.4\times10^{-4}$).

**6. Validation on a Second Dataset**
* **The Critique:** Experiments are limited to MVTec-AD; validation on a second dataset is required.
* **Our Response:** Complete. We executed the entire pipeline on the VisA dataset (12 distinct categories, including PCBs and medical capsules). The full VisA benchmark (3,264 rows) validates the MVTec-AD findings: test-time rescue preprocessing remains significantly net-harmful for PatchCore ($\Delta = -0.0373$, 12 categories, clustered $p = 4.9\times10^{-4}$). For PaDiM it is $\Delta = -0.0061$ at $p = 0.375$ — not distinguishable from zero once the Wiener PSF is corrected, and we do not claim the fallacy for that condition. The augmented VisA arm covers 4 categories and is reported as descriptive support only — with 4 clusters the smallest attainable two-sided p is 0.125.

### 🔴 Group B: Deferred & Defended (Points 2, 3, 4, & 5)

*Note: Executing these critiques would either be computationally prohibitive (hundreds of Kaggle GPU hours) or outside the scope of our primary research question. We will defend against them directly in the manuscript's limitations section.*

**5. Feature-Space Evidence (Artifact Proof)** — *DEFERRED to future work*
* **The Critique:** Support the "Preprocessing Fallacy" claim with feature-space evidence (e.g., measuring how far extracted features are from the normal distribution).
* **Our Defense:** The effect is established, though not at the significance we previously claimed. Under correct category-clustered inference the rescue harm holds at $p \le 2.1\times10^{-3}$ across all four MVTec-AD conditions and for clean-trained VisA PatchCore ($p = 4.9\times10^{-4}$); clean-trained VisA PaDiM is the one exception at $p = 0.375$. That establishes *that* rescue methods harm performance. Extracting high-dimensional embeddings to compute Mahalanobis distances or plot t-SNE clusters would explain *why* (distributional shift). Because that requires custom feature-extraction hooks and thousands of regenerated inferences, it is deferred to future work and declared as such in the limitations.

**2. Augmentation Probability Ablation (0.25, 0.50, 0.75, 1.00)**
* **Our Defense:** Ablating probabilities would require retraining and re-inferencing the entire dataset 3 more times (~18,000 additional inferences). The Kaggle compute cost is prohibitive. More importantly, our current operating point ($p=0.50$) achieved up to +24 percentage points of AUROC gain. The scientific claim that "augmented training works" is fully proven; finding the mathematically optimal $p$ is an engineering optimization detail, not a scientific necessity.

**3. Unseen-Corruption Experiment (Train on A, Test on B)**
* **Our Position (revised):** This is the one control that separates "augmentation works" from "we trained on the test distribution," and it should not be waved off. Our augmentation draws from the *same* 5 types × 3 severities used at test time, so the reported +12.3 pp / +10.1 pp gains are explicitly a **corruption-matched oracle upper bound**, declared as such in the limitations. Our scope claim stands — we evaluate targeted interventions for known imaging problems, where the corruption is characterised in advance — but the honest framing is that generalisation to unseen corruptions is *untested*, not out of scope.
* **Bounded control (recommended):** a leave-one-corruption-out run does not require repeating the whole benchmark. Holding out one corruption type, training the augmented arm on the remaining four, and testing only on the held-out type over a 5-category subset at 1 seed is roughly 1/9 of the augmented arm's cost and would convert an admitted weakness into a positive result.

**4. Rescue Parameter Tuning**
* **Our Defense (corrected):** The oracle-PSF argument holds **only at severe severity**, and we have corrected the record. The published mild and moderate Wiener rows were produced with the severe-tier kernel ($\sigma=25$, $k=281$; motion $k=151$) hardcoded across all severities, so they measure kernel *misspecification*, not oracle deconvolution. At severe corruption, where the kernel was correct, Wiener still costs ~−12.5 pp — an exact-PSF filter failing on spectral ringing. That result carries the argument: if the oracle filter fails, a blind filter performs no better. The mild/moderate cells are being rerun with per-severity PSFs ([`scripts/wiener_reruns/`](../scripts/wiener_reruns/)); until they land, only the severe tier may be described as oracle.

---

## Part 2: Current Inventory of Completed Runs

### MVTec-AD (Complete ✅)
| Notebook (current) | Model | Training | Status |
|---|---|---|---|
| `01_mvtec_patchcore_clean.ipynb` | PatchCore | Clean | ✅ Complete (15 categories × 3 seeds = 1,530 rows) |
| `02_mvtec_patchcore_augmented.ipynb` | PatchCore | Augmented | ✅ Complete (15 categories × 3 seeds = 1,530 rows) |
| `03_mvtec_padim_clean.ipynb` | PaDiM | Clean | ✅ Complete (15 categories × 3 seeds = 1,530 rows) |
| `04_mvtec_padim_augmented.ipynb` | PaDiM | Augmented | ✅ Complete (15 categories × 3 seeds = 1,530 rows) |
| **Total:** | | | **6,120 rows** |

> **Provenance.** The committed clean-training notebooks are consolidated versions.
> The rows themselves were produced on Kaggle by split shards — one per severity
> band — plus dedicated rescue runners, all of which are recoverable from git
> history (`01-baseline-patchcore-mod-mild.ipynb`,
> `02-padim-baseline-mild-moderate.ipynb`, `run_patchcore_rescue.py`,
> `run_padim_rescue.py`; added in `21ce8ef`, 2026-05-30). A reader cannot
> reconstruct the full original run from the current `notebooks/` directory alone.
> The Wiener reruns in `scripts/wiener_reruns/` are self-contained and do not have
> this problem.

### VisA Dataset
| Notebook | Model | Training | Status |
|---|---|---|---|
| `05_visa_patchcore_clean.ipynb` | PatchCore | Clean | ✅ Complete (12 categories × 3 seeds) |
| `07_visa_padim_clean.ipynb` | PaDiM | Clean | ✅ Complete (12 categories × 3 seeds) |
| `06_visa_patchcore_augmented.ipynb` | PatchCore | Augmented | ✅ Complete (4 categories × 3 seeds) |
| `08_visa_padim_augmented.ipynb` | PaDiM | Augmented | ✅ Complete (4 categories × 3 seeds) |
| **Total:** | | | **3,264 rows** |

All VisA results are merged into `data/benchmark_master_combined.csv` (9,384 rows:
6,120 MVTec-AD + 3,264 VisA). Every (category, seed) pair carries 34 rows; no
duplicates, no nulls.

---

## Part 3: Final Action Plan

### Priority 1: Wiener PSF rerun ✅ DONE
All 1,104 mild/moderate Wiener rescue rows (22.2% of rescue rows) were
regenerated with per-severity PSFs by the eight notebooks in
[`scripts/wiener_reruns/`](../scripts/wiener_reruns/) and merged into the master
CSV by [`merge_wiener_reruns.py`](../scripts/analysis/merge_wiener_reruns.py),
which asserts a 1:1 join before writing.

- [x] Execute reruns 01–08 on Kaggle
- [x] Merge corrected rescue rows into the master CSV (872 of 1,104 values changed)
- [x] Rescue cells pinned at the AUROC floor fell from 795 to 238

### Priority 2: Re-run the statistics after the merge ✅ DONE
- [x] Regenerated [`statistical_validation.md`](statistical_validation.md)
- [x] Regenerated [`per_corruption_tables.md`](per_corruption_tables.md) by script
- [x] Updated rescue figures in `README.md`, `benchmark_report.md`,
      `benchmark_results.md`, `paper_sections.md`
- [ ] Re-run `notebooks/10_wilcoxon_testing.ipynb` to regenerate figures 04, 05, 07–09

**Outcome.** Pooled rescue harm softened by 1.4–3.2 pp per condition, as
projected. All four MVTec-AD conditions remain significantly negative
(−2.75 to −11.79 pp, $p \le 2.1\times10^{-3}$), as does clean-trained VisA
PatchCore (−3.73 pp, $p = 4.9\times10^{-4}$). **Clean-trained VisA PaDiM moved to
−0.61 pp at $p = 0.375$ and is no longer distinguishable from zero** — the one
condition where the preprocessing fallacy does not hold, and it must not be
claimed there.

### Priority 3: Statistical hygiene ✅ DONE
- [x] **Cluster inference by category** — `scripts/analysis/cluster_robust_stats.py`, results in [`statistical_validation.md`](statistical_validation.md)
- [x] **Report effective N** after zero-difference dropping (645–806, not 810)
- [x] **Report AUROC floor saturation** (23.1% of rows; 545 uninformative pairs)
- [x] **Stop claiming significance for the 4-category VisA augmented arm** (p floor = 0.125)

### Priority 4: Paper Writing
- [ ] **Write the paper** using the corrected statistical results
- [ ] **Frame VisA augmented as a representative subset**: "Augmented training was validated on a representative subset of 4 VisA categories spanning diverse product types" — descriptive, not significance-tested
- [ ] **Limitations section**: subset evaluation on VisA augmented, single aug_prob operating point, no unseen-corruption test, train/test resolution mismatch, AUROC floor censoring
- [ ] **Provenance note**: state which notebook shard produced which CSV rows, so a reader can reconstruct the run

### Optional: bounded leave-one-corruption-out control
Holding out one corruption type, training the augmented arm on the remaining four
and testing only on the held-out type — over a 5-category subset at 1 seed — is
roughly 1/9 of the augmented arm's cost and would convert the "corruption-matched
oracle" limitation into a positive generalisation result.

### Kaggle Execution Strategy (Wiener reruns)
- **GPU required**: the reruns retrain each model before re-executing the rescue inferences
- **Session plan and time estimates**: `scripts/wiener_reruns/INSTRUCTIONS.md`
- Resume logic keys on `(category, seed)`; notebooks 06 and 08 additionally resume at row level
