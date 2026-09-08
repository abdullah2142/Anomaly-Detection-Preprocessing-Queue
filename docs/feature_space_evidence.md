# Feature-Space Evidence for the Preprocessing Fallacy

This file is generated. Do not edit by hand -- rerun
`python scripts/analysis/analyze_feature_space.py results/feature_space_probe.txt --doc docs/feature_space_evidence.md`.

The paper's named contribution is a mechanism: restoration does not return a
corrupted image to the clean distribution but creates a third distribution,
further from normal than the corruption was. AUROC cannot test that -- it
measures only the ranking of scores, never how far anything sits from normal.
The anomaly score IS that distance, so its magnitude can.

Comparisons are restricted to normal test images and made within a
(category, model) unit, since scores are only comparable within one trained
model. Significance is an exact paired permutation test over per-unit
differences.

```
=== feature_space_probe.txt ===
14880 score rows | 5 categories | ['PaDiM', 'PatchCore']

Normalisation check: 0.0% of conditions span exactly [0, 1] (0/120)
  OK -- scores retain their raw scale.
Ceiling check: share of images with score exactly 1.0, by condition:
  clean       11.0%
  degraded    46.6%
  rescued     44.3%
  WARNING: some scores are clipped at 1.0. Any rescued-minus-degraded
  difference is a LOWER BOUND -- clipping can only shrink it.

Restricting to normal test images: 4488 of 14880 rows

Mean anomaly score on NORMAL images, by condition:
condition            clean  degraded  rescued
category model                               
bottle   PaDiM      0.4370    0.8458   0.9603
         PatchCore  0.2544    0.8102   0.8301
cable    PaDiM      0.5218    0.7118   0.7358
         PatchCore  0.4259    0.8261   0.7973
carpet   PaDiM      0.4105    0.6510   0.5999
         PatchCore  0.3909    0.7966   0.7706
hazelnut PaDiM      0.4509    0.9700   0.8534
         PatchCore  0.3033    0.6734   0.7297
screw    PaDiM      0.6221    0.9994   0.9710
         PatchCore  0.4662    0.9765   0.9932

=== the fallacy's three predictions ===
  degraded > clean   (corruption moves images away from normal)
      mean difference +0.3978   p = 0.0020   (10 units, 10/10 positive)   -> SUPPORTED
  rescued  > clean   (rescued images are not back home)
      mean difference +0.3958   p = 0.0020   (10 units, 10/10 positive)   -> SUPPORTED
  rescued  > degraded (THE CLAIM: restoration moves them FURTHER)
      mean difference -0.0020   p = 0.9141   (10 units, 5/10 positive)   -> not supported

=== ceiling-robust checks on the same claim ===
  share of normal images pinned at the ceiling, rescued − degraded:
      +0.19 pp   p = 0.9727   (6/10 positive)
      (restoration pushing MORE images to the ceiling would support the
       claim even where the means are censored)
  restricted to units with <20% clipping on both sides: 4 of 10 units
      rescued − degraded: +0.0123   p = 0.6250   (3/4 positive)

=== discriminability: mean score(anomalous) − mean score(normal) ===
(what AUROC depends on -- can the model still tell them apart?)
  degraded − clean     -0.2584   p = 0.0020   (10/10 shrank)
  rescued  − degraded  -0.0010   p = 0.9082   (6/10 shrank)

=== per rescue method: rescued − degraded on normal images ===
  CLAHE                    -0.0258   p = 0.7500
  Dehaze (Dark Channel)    +0.0206   p = 0.2500
  NLM Denoise              -0.0214   p = 0.6250
  Retinex                  +0.0130   p = 0.6250
  Wiener                   +0.0886   p = 0.2500
  Wiener (Motion PSF)      +0.0285   p = 0.2500
```
