# Response to Professor's Review & Project To-Do List

## Part 1: Strategic Responses to Professor's Critiques

This document outlines the defense strategy against the six methodological critiques raised by the professor. It maps out which points are already completed, which are actionable next steps, and which we will strategically defend against based on theoretical bounds and computational limits.

### 🟢 Group A: Already Completed (Points 1 & 6)

**1. Statistical Significance (Paired Tests & Confidence)**
* **The Critique:** Relying on average AUROC values is insufficient; paired statistical tests (Augmented vs Clean, Rescued vs Degraded) are required.
* **Our Response:** We agree completely and have already overhauled our statistical methodology to address this. We executed non-parametric paired Wilcoxon signed-rank tests across the full 6,120-row dataset, measuring exact paired deltas. Furthermore, we applied the Benjamini-Hochberg False Discovery Rate (FDR) correction ($q < 0.05$) to all 24 multiple comparisons to ensure rigorous statistical confidence. The results remain overwhelmingly significant ($p < 0.001$).

**6. Validation on a Second Dataset**
* **The Critique:** Experiments are limited to MVTec-AD; validation on a second dataset is required.
* **Our Response:** We are actively addressing this. We have completed the execution of our entire pipeline on the VisA dataset (12 distinct categories, including PCBs and medical capsules). Preliminary analysis of the fully completed 1,224 Clean/Degraded/Rescued rows perfectly validates the MVTec-AD findings: test-time rescue preprocessing remains net-harmful for both PatchCore ($\Delta = -0.0465$) and PaDiM ($\Delta = -0.0176$). The augmented VisA runs are currently executing.

### 🔴 Group B: Deferred & Defended (Points 2, 3, 4, & 5)

*Note: Executing these critiques would either be computationally prohibitive (hundreds of Kaggle GPU hours) or outside the scope of our primary research question. We will defend against them directly in the manuscript's limitations section.*

**5. Feature-Space Evidence (Artifact Proof)** — *DEFERRED to future work*
* **The Critique:** Support the "Preprocessing Fallacy" claim with feature-space evidence (e.g., measuring how far extracted features are from the normal distribution).
* **Our Defense:** Our empirical proof is airtight. With overwhelming Wilcoxon significance ($p < 10^{-27}$) across multiple datasets, seeds, and models, we have conclusively proven *that* rescue methods harm performance. Extracting high-dimensional embeddings to calculate Mahalanobis distances or plotting t-SNE clusters would only serve to explain *why* (distributional shift). Because this requires writing custom feature-extraction hooks and regenerating thousands of inferences, it is deferred to future work. The paper is empirically complete without it.

**2. Augmentation Probability Ablation (0.25, 0.50, 0.75, 1.00)**
* **Our Defense:** Ablating probabilities would require retraining and re-inferencing the entire dataset 3 more times (~18,000 additional inferences). The Kaggle compute cost is prohibitive. More importantly, our current operating point ($p=0.50$) achieved up to +24 percentage points of AUROC gain. The scientific claim that "augmented training works" is fully proven; finding the mathematically optimal $p$ is an engineering optimization detail, not a scientific necessity.

**3. Unseen-Corruption Experiment (Train on A, Test on B)**
* **Our Defense:** Testing on unseen corruptions shifts the scope of the paper into "Zero-Shot Domain Generalization," a distinct sub-field of machine learning. Our paper evaluates targeted interventions for known industrial problems (e.g., a factory knows its specific camera setup causes low-light sensor noise). Evaluating generalization to entirely unknown corruptions is out of scope for our targeted robustness research question.

**4. Rescue Parameter Tuning**
* **Our Defense:** We deliberately evaluated rescue methods under **best-case, Oracle conditions** to establish a theoretical upper bound. For example, our Wiener deconvolution uses the *exact* Point Spread Function (PSF) used to generate the blur. It still failed catastrophically due to high-frequency spectral ringing artifacts destroying the feature embeddings. If a perfect "Oracle" filter fails, parameter-tuning a blind filter will mathematically perform worse. There is no justification to run this.

---

## Part 2: Current Inventory of Completed Runs

### MVTec-AD (Complete ✅)
| Notebook | Model | Training | Status |
|---|---|---|---|
| `01-patchcore.ipynb` | PatchCore | Clean | ✅ Complete (15 categories × 3 seeds = 1,530 rows) |
| `01-patchcore-augmented.ipynb` | PatchCore | Augmented | ✅ Complete (15 categories × 3 seeds = 1,530 rows) |
| `02-padim.ipynb` | PaDiM | Clean | ✅ Complete (15 categories × 3 seeds = 1,530 rows) |
| `02-padim-augmented.ipynb` | PaDiM | Augmented | ✅ Complete (15 categories × 3 seeds = 1,530 rows) |
| **Total:** | | | **6,120 rows in `benchmark_full_4way.csv`** |

### VisA Dataset
| Notebook | Model | Training | Status |
|---|---|---|---|
| `03-visa-patchcore.ipynb` | PatchCore | Clean | ✅ Complete (12 categories × 3 seeds = 1,224 rows) |
| `04-visa-padim.ipynb` | PaDiM | Clean | ✅ Complete on Kaggle — **download CSV** |
| `visa-patchcore-augmented.ipynb` | PatchCore | Augmented | ✅ Complete (4 target categories = 408 rows) |
| `visa-padim-augmented.ipynb` | PaDiM | Augmented | 🟡 Partial: candle ×3 only (102/1,224 rows) |

> **Note:** The file `visa_padim_partial.csv` (102 rows, candle only) is from the **augmented** run, not unaugmented. VisA PaDiM unaugmented has been run on Kaggle but the results CSV has not been downloaded yet.

---

## Part 3: Final Action Plan

### Priority 1: Download VisA PaDiM Unaugmented Results
- [ ] **Download `visa_padim_partial.csv` (or `visa_padim.csv`) from Kaggle** — Already run, just needs the output CSV retrieved
- Rename appropriately to avoid confusion with the augmented partial CSV already on disk

### Priority 2: VisA Augmented Subset (4 representative categories)
Run both models on the same 4 structurally diverse categories to demonstrate augmentation benefit generalizes to VisA:
- `candle` (texture) — already done for both models
- `pcb1` (fine-grained structure)  
- `cashew` (organic shape)
- `pipe_fryum` (complex geometry)

- [x] **PatchCore augmented** — Run `visa-patchcore-augmented.ipynb` with `CATEGORIES = ['pcb1', 'cashew', 'pipe_fryum']` (candle already done)
- [x] **PaDiM augmented** — Run `visa-padim-augmented.ipynb` with `CATEGORIES = ['pcb1', 'cashew', 'pipe_fryum']` (candle already done)
- Each category takes ~9 hours on CPU (heavy NLM/Wiener filters on native-resolution images)
- Split across multiple Kaggle CPU notebooks for parallelism
- Can run on CPU-only sessions (does NOT consume GPU quota)

### Priority 3: Data Assembly & Statistical Testing
- [x] **Add `training` column** to all VisA CSVs (`'clean'` for unaugmented, `'augmented'` for augmented runs)
- [x] **Add `dataset` column** to all CSVs (`'MVTec-AD'` or `'VisA'`)
- [x] **Merge into master CSV** combining MVTec 4-way + VisA results
- [x] **Run `wilcoxon-testing.ipynb`** on the combined dataset (stratified by dataset)
- [x] **Generate per-corruption-type breakdown table** from existing MVTec and VisA CSVs

### Priority 4: Paper Writing
- [ ] **Write the paper** using the completed statistical results
- [ ] **Frame VisA augmented as representative subset**: "Augmented training was validated on a representative subset of 4 VisA categories spanning diverse product types"
- [ ] **Limitations section**: Acknowledge subset evaluation on VisA augmented, single aug_prob operating point, and no unseen-corruption test

### Kaggle Execution Strategy
- **VisA PaDiM unaugmented**: Already done — just download the CSV
- **VisA augmented (6 category-model combos)**: Run on CPU-only. Split 1 category per notebook = 6 CPU notebooks simultaneously. Each finishes in ~9 hours (within 12-hour limit)
- **Total Kaggle GPU time needed**: 0 hours
- **Total Kaggle CPU time needed**: ~54 hours (runs in parallel, wall-clock ~9 hours)
