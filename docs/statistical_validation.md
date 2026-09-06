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
| MVTec-AD | PaDiM | augmented | -11.34 pp | 15 | 810 | 806 | 6.10e-05 (at floor) |
| MVTec-AD | PaDiM | clean | -4.56 pp | 15 | 810 | 726 | 0.00024 |
| MVTec-AD | PatchCore | augmented | -14.90 pp | 15 | 810 | 803 | 6.10e-05 (at floor) |
| MVTec-AD | PatchCore | clean | -7.17 pp | 15 | 810 | 645 | 6.10e-05 (at floor) |
| VisA | PaDiM | augmented | -12.37 pp | 4 | 216 | 216 | 0.12500 (at floor) |
| VisA | PaDiM | clean | -2.74 pp | 12 | 648 | 576 | 0.00293 |
| VisA | PatchCore | augmented | -18.46 pp | 4 | 216 | 216 | 0.12500 (at floor) |
| VisA | PatchCore | clean | -6.42 pp | 12 | 648 | 416 | 0.00049 (at floor) |

### AUROC floor saturation

- Rows at exactly 0.5: **2169 / 9384** (23.1%)
- Rescue values at the floor: 1524 / 4968 (30.7%)
- Degraded values at the floor: 687 / 4968 (13.8%)
- Pairs where both sides are 0.5 (zero difference, dropped by the signed-rank test): **545**

The signed-rank test drops zero differences silently, so the effective N
column above -- not the naive N -- is the sample size those tests actually
used. The floor share also means severity means are censored from below:
the true collapse under severe corruption is worse than the reported AUROC.

---

Source: `data/benchmark_master_combined.csv` (9384 rows, 2 datasets, 27 categories). Wiener rescue rows: **hardcoded severe-tier PSF (pre-rerun)**.
