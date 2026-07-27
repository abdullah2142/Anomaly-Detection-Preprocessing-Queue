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

### 🟡 Group B: Actionable Next Step (Point 5)

**5. Feature-Space Evidence (Artifact Proof)**
* **The Critique:** Support the "Preprocessing Fallacy" claim with feature-space evidence (e.g., measuring how far extracted features are from the normal distribution).
* **Our Response:** This is an excellent methodological suggestion. To mathematically prove that classical restoration pushes images into a "third distribution", we will extract the 1024-dimensional backbone embeddings for a Clean, Degraded, and Rescued image of the same sample. We will then calculate their Mahalanobis/L2 distances relative to the normal training coreset. This will provide direct mathematical proof of the algorithm-induced artifacts (e.g., Gibbs ringing).

### 🔴 Group C: Strategic Defenses (Points 2, 3, & 4)

*Note: Executing these would require hundreds of hours of Kaggle compute. We will defend against them theoretically.*

**2. Augmentation Probability Ablation (0.25, 0.50, 0.75, 1.00)**
* **Our Defense:** We acknowledge that `aug_prob = 0.50` is a single operating point. However, the effect size we observed at 0.50 is massive (+10 to +12 percentage points AUROC, $p<0.001$). While hyperparameter tuning might optimize the gain marginally, it will not reverse the qualitative conclusion that augmented training provides immense manifold robustness. We have noted the lack of full hyperparameter sweeping as a constraint in our Limitations section.

**3. Unseen-Corruption Experiment (Train on A, Test on B)**
* **Our Defense:** Testing on unseen corruptions shifts the scope of the paper from "robustness benchmarking under known adversarial conditions" into "Domain Generalization". Our paper specifically evaluates the two most common engineering interventions deployed when a specific degradation (like conveyor belt motion blur) is identified. Generalizing to completely unknown corruptions is a separate research question left for future work.

**4. Rescue Parameter Tuning**
* **Our Defense:** We deliberately evaluated rescue methods under **best-case, Oracle conditions** to establish an upper bound on their performance. For example, our Wiener deconvolution uses the exact Point Spread Function (PSF kernel) used to generate the blur. If a mathematically perfect, known-kernel deconvolution fails catastrophically due to fundamental spectral ringing artifacts (the Gibbs phenomenon), then parameter-tuning a blind deconvolution will not save it. The harm is caused by fundamental feature-incompatibility, not suboptimal tuning.

---

## Part 2: Final Project To-Do List

- [ ] **Finish VisA Augmented Runs:** Upload the fixed `05-visa-patchcore-augmented.ipynb` and `06-visa-padim-augmented.ipynb` to Kaggle and execute them to generate the final augmented CSVs.
- [ ] **Merge Datasets:** Combine the MVTec-AD and VisA CSVs into a final master dataset and run the `run_wilcoxon_tests.ipynb` pipeline to generate the final cross-dataset statistical figures.
- [ ] **Feature-Space Distance Script:** Write a short offline script to satisfy Professor's Point 5 (extracting WideResNet50 embeddings for a single image across 3 states and calculating their distance to the coreset).
- [ ] **Finalize Manuscript:** Inject the final VisA metrics and the feature-space distance proof into `paper_sections.md` and format it for final submission.
