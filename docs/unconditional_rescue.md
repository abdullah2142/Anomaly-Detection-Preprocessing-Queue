# Unconditional and Misidentified Rescue

This file is generated. Do not edit by hand -- rerun
`python scripts/analysis/analyze_unconditional_rescue.py results/unconditional_rescue.csv --doc docs/unconditional_rescue.md`.

The benchmark applies each rescue only to the corruption it targets,
selected using that corruption's ground-truth identity. Deployment offers
neither guarantee: it preprocesses a whole stream, most of which may be
undegraded, and its detector can misclassify. Part A measures the cost of
preprocessing an image that needed no treatment; Part B the cost of
applying the rescue for the wrong corruption. Both comparison baselines --
the clean baseline AUROC and the degradation AUROC -- come from the master
CSV. Significance is an exact paired permutation test over per-category
means; with eight categories the smallest attainable p is 0.0078.

```
=== unconditional_rescue.csv ===
336 rows | 8 categories | ['PaDiM', 'PatchCore'] | seed [np.int64(42)]

=== Part A: rescue applied to UNDEGRADED images ===
(delta vs the clean baseline; negative means preprocessing cost you accuracy)

Model       Rescue                       Delta         p
PaDiM       CLAHE                       -4.92 pp   0.0156
PaDiM       Dehaze (Dark Channel)      -10.86 pp   0.0234
PaDiM       NLM Denoise                 -0.06 pp   0.8125
PaDiM       Retinex                    -16.50 pp   0.0078
PaDiM       Wiener                     -20.35 pp   0.0078
PaDiM       Wiener (Motion PSF)        -35.27 pp   0.0078
PatchCore   CLAHE                       -0.95 pp   0.2031
PatchCore   Dehaze (Dark Channel)      -12.86 pp   0.0625
PatchCore   NLM Denoise                 -0.13 pp   0.2500
PatchCore   Retinex                    -10.80 pp   0.0312
PatchCore   Wiener                     -19.69 pp   0.0078
PatchCore   Wiener (Motion PSF)        -46.25 pp   0.0078

Pooled across both models, per rescue (which methods are safe to apply blind):
  CLAHE                       -2.93 pp   p = 0.0156
  Dehaze (Dark Channel)      -11.86 pp   p = 0.0156
  NLM Denoise                 -0.09 pp   p = 0.3750
  Retinex                    -13.65 pp   p = 0.0078
  Wiener                     -20.02 pp   p = 0.0078
  Wiener (Motion PSF)        -40.76 pp   p = 0.0078

  PaDiM       pooled over all six rescues: -14.66 pp (p = 0.0078, 8 categories)
  PatchCore   pooled over all six rescues: -15.12 pp (p = 0.0078, 8 categories)

=== Part B: rescue applied to the WRONG corruption ===
(vs_degraded: cost against doing nothing. vs_matched: cost of misidentifying)

Corruption      Wrong rescue              vs degraded        p   vs matched
fog_haze        CLAHE                          -0.19 pp  0.9375       +1.59 pp
gaussian_blur   Wiener (Motion PSF)           -15.84 pp  0.0078       -5.71 pp
low_light       NLM Denoise                    -4.39 pp  0.0234       -2.00 pp
motion_blur     Wiener                        -11.48 pp  0.0078       -2.90 pp
sensor_noise    Dehaze (Dark Channel)          +0.04 pp  0.3750       -2.01 pp

  PaDiM       pooled misidentified rescue vs doing nothing: -5.17 pp (p = 0.0078, 8 categories)
  PatchCore   pooled misidentified rescue vs doing nothing: -7.57 pp (p = 0.0078, 8 categories)
```
