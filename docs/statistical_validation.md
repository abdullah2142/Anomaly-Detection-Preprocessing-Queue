# Statistical Validation (Cluster-Robust)

This file is generated. Do not edit by hand -- rerun
`python scripts/analysis/cluster_robust_stats.py --out docs/statistical_validation.md`
after any change to the benchmark CSV.

## Why these numbers differ from notebooks/10_wilcoxon_testing.ipynb

Notebook 10 treats every (category, seed, corruption, severity) cell as an
independent paired sample. Those cells are not independent: each category
contributes up to 45 of them, so the tests are pseudoreplicated and their
p-values are inflated by many orders of magnitude.

The tables below aggregate each condition to per-category means -- the level at
which observations are genuinely independent -- and apply an exact two-sided
paired permutation (sign-flip) test over all 2^n sign assignments. **The point
estimates are identical to the pseudoreplicated analysis; only the p-values
move.** The findings are unchanged in direction and magnitude.

With n clusters the smallest attainable two-sided p is 2 / 2^n. For the four
VisA augmented categories that floor is 0.125, so no result computed on them can
reach conventional significance regardless of effect size; those rows are
reported as descriptive evidence, not as significance tests.

### Augmentation gain (augmented-trained - clean-trained, same degraded cell)

| Dataset | Model | Mean gain | Categories | Naive N | Clustered exact p |
|---|---|---|---:|---:|---:|
| MVTec-AD | PaDiM | +10.09 pp | 15 | 675 | 6.10e-05 (at floor) |
| MVTec-AD | PatchCore | +12.32 pp | 15 | 675 | 0.00012 |
| VisA | PaDiM | +13.88 pp | 4 | 180 | 0.12500 (at floor) |
| VisA | PatchCore | +17.80 pp | 4 | 180 | 0.12500 (at floor) |

### Rescue delta (rescued - degraded, all methods pooled)

| Dataset | Model | Training | Mean delta | Categories | Naive N | Effective N | Clustered exact p |
|---|---|---|---|---:|---:|---:|---:|
| MVTec-AD | PaDiM | augmented | -9.55 pp | 15 | 810 | 806 | 0.00012 |
| MVTec-AD | PaDiM | clean | -2.75 pp | 15 | 810 | 732 | 0.00214 |
| MVTec-AD | PatchCore | augmented | -11.79 pp | 15 | 810 | 800 | 6.10e-05 (at floor) |
| MVTec-AD | PatchCore | clean | -3.94 pp | 15 | 810 | 661 | 0.00110 |
| VisA | PaDiM | augmented | -11.00 pp | 4 | 216 | 216 | 0.12500 (at floor) |
| VisA | PaDiM | clean | -0.61 pp | 12 | 648 | 587 | 0.37500 |
| VisA | PatchCore | augmented | -15.75 pp | 4 | 216 | 216 | 0.12500 (at floor) |
| VisA | PatchCore | clean | -3.73 pp | 12 | 648 | 437 | 0.00049 (at floor) |

### AUROC floor saturation

- Rows at exactly 0.5: **1612 / 9384** (17.2%)
- Rescue values at the floor: 967 / 4968 (19.5%)
- Degraded values at the floor: 687 / 4968 (13.8%)
- Pairs where both sides are 0.5 (zero difference, dropped by the signed-rank test): **489**

The signed-rank test drops zero differences silently, so the effective N
column above -- not the naive N -- is the sample size those tests actually
used. The floor share also means severity means are censored from below:
the true collapse under severe corruption is worse than the reported AUROC.

---

Source: `data/benchmark_master_combined.csv` (9384 rows, 2 datasets, 27 categories). Wiener rescue rows: **per-severity PSF (corrected)**.
