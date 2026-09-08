# Generalization Controls

This file is generated. Do not edit by hand -- rerun
`python scripts/analysis/analyze_generalization.py --doc docs/generalization_controls.md`
after any change to the control runs or the benchmark CSV.

The headline augmentation gains (+12.3 pp PatchCore, +10.1 pp PaDiM) are a
**corruption-matched oracle bound**: training draws from the same five corruption
types and three severities used at test time, so the benchmark alone cannot
separate genuine robustness from having trained on the test distribution. These
two controls separate them.

Each control trains the augmented arm with one condition withheld and tests only
on that withheld condition. The clean-trained and matched-augmented comparison
rows come from the master CSV, restricted to the same categories, corruptions,
severities and seed. Significance is an exact two-sided paired permutation test
over per-category means; with eight categories the smallest attainable p is
0.0078.

**Share surviving** is the held-out gain as a percentage of the matched-oracle
gain -- how much of the published benefit remains when the model has not been
trained on what it is tested against.

### Leave-One-Corruption-Out

Trained on four corruption types, tested only on the withheld fifth, rotating through all five. Answers whether augmentation transfers to an unseen *kind* of degradation.

240 paired rows over 8 categories.

| Model | Matched-oracle gain | Held-out gain | Share surviving | Cost of holding out |
|---|---|---|---:|---|
| PaDiM | +8.02 pp (p = 0.0078, at floor) | **-0.22 pp** (p = 0.8281) | none | -8.23 pp (p = 0.0078, at floor) |
| PatchCore | +10.66 pp (p = 0.0156) | **+3.58 pp** (p = 0.0312) | 34% | -7.08 pp (p = 0.0156) |

Held-out gain per corruption type (models pooled):

| Corruption | Gain | p |
|---|---|---:|
| fog_haze | +1.18 pp | 0.3359 |
| gaussian_blur | +1.61 pp | 0.2734 |
| low_light | +3.12 pp | 0.0703 |
| motion_blur | +0.53 pp | 0.7031 |
| sensor_noise | +1.95 pp | 0.3906 |

Held-out gain per severity (models pooled):

| Severity | Gain | p |
|---|---|---:|
| mild | +2.88 pp | 0.0938 |
| moderate | +1.63 pp | 0.2656 |
| severe | +0.53 pp | 0.5000 |

### Severity Holdout

Trained on mild and moderate only, tested on severe. Answers whether augmentation transfers to degradation *worse* than it trained for.

80 paired rows over 8 categories.

| Model | Matched-oracle gain | Held-out gain | Share surviving | Cost of holding out |
|---|---|---|---:|---|
| PaDiM | +8.83 pp (p = 0.0078, at floor) | **+1.02 pp** (p = 0.4609) | 12% | -7.82 pp (p = 0.0234) |
| PatchCore | +13.21 pp (p = 0.0078, at floor) | **+4.58 pp** (p = 0.0078, at floor) | 35% | -8.63 pp (p = 0.0156) |

Held-out gain per corruption type (models pooled):

| Corruption | Gain | p |
|---|---|---:|
| fog_haze | +1.57 pp | 0.1562 |
| gaussian_blur | +3.94 pp | 0.1094 |
| low_light | +3.64 pp | 0.0078 |
| motion_blur | +4.28 pp | 0.0234 |
| sensor_noise | +0.57 pp | 0.8281 |
