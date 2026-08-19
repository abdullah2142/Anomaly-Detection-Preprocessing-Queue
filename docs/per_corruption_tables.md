# Per-Corruption Type Breakdown Tables

## 1. Augmented Training Gains by Corruption Type

### Dataset: MVTec-AD

| Model | Corruption | Severity | Clean Train AUROC | Aug Train AUROC | Delta (Aug - Clean) |
|---|---|---|---|---|---|
| PaDiM | fog_haze | mild | 0.6243 | 0.7238 | +0.0994 |
| PaDiM | fog_haze | moderate | 0.5539 | 0.6218 | +0.0679 |
| PaDiM | fog_haze | severe | 0.5214 | 0.5592 | +0.0378 |
| PaDiM | gaussian_blur | mild | 0.7723 | 0.8478 | +0.0755 |
| PaDiM | gaussian_blur | moderate | 0.6129 | 0.7475 | +0.1346 |
| PaDiM | gaussian_blur | severe | 0.5813 | 0.7107 | +0.1294 |
| PaDiM | low_light | mild | 0.7742 | 0.8545 | +0.0803 |
| PaDiM | low_light | moderate | 0.7174 | 0.8400 | +0.1227 |
| PaDiM | low_light | severe | 0.6640 | 0.8012 | +0.1372 |
| PaDiM | motion_blur | mild | 0.7207 | 0.8163 | +0.0955 |
| PaDiM | motion_blur | moderate | 0.6294 | 0.7709 | +0.1415 |
| PaDiM | motion_blur | severe | 0.5681 | 0.7377 | +0.1697 |
| PaDiM | sensor_noise | mild | 0.6569 | 0.7225 | +0.0655 |
| PaDiM | sensor_noise | moderate | 0.6105 | 0.6925 | +0.0821 |
| PaDiM | sensor_noise | severe | 0.5784 | 0.6535 | +0.0751 |
| PatchCore | fog_haze | mild | 0.7098 | 0.8491 | +0.1393 |
| PatchCore | fog_haze | moderate | 0.6161 | 0.7717 | +0.1555 |
| PatchCore | fog_haze | severe | 0.5623 | 0.6874 | +0.1251 |
| PatchCore | gaussian_blur | mild | 0.9173 | 0.9274 | +0.0101 |
| PatchCore | gaussian_blur | moderate | 0.6483 | 0.8588 | +0.2105 |
| PatchCore | gaussian_blur | severe | 0.5831 | 0.8292 | +0.2461 |
| PatchCore | low_light | mild | 0.9607 | 0.9548 | -0.0059 |
| PatchCore | low_light | moderate | 0.8910 | 0.9356 | +0.0446 |
| PatchCore | low_light | severe | 0.7840 | 0.8746 | +0.0905 |
| PatchCore | motion_blur | mild | 0.7824 | 0.9159 | +0.1335 |
| PatchCore | motion_blur | moderate | 0.6595 | 0.8760 | +0.2165 |
| PatchCore | motion_blur | severe | 0.6127 | 0.8389 | +0.2262 |
| PatchCore | sensor_noise | mild | 0.8470 | 0.8942 | +0.0472 |
| PatchCore | sensor_noise | moderate | 0.7638 | 0.8535 | +0.0898 |
| PatchCore | sensor_noise | severe | 0.6966 | 0.8149 | +0.1183 |

### Dataset: VisA

| Model | Corruption | Severity | Clean Train AUROC | Aug Train AUROC | Delta (Aug - Clean) |
|---|---|---|---|---|---|
| PaDiM | fog_haze | mild | 0.5650 | 0.6405 | +0.0755 |
| PaDiM | fog_haze | moderate | 0.5245 | 0.5796 | +0.0551 |
| PaDiM | fog_haze | severe | 0.5009 | 0.5324 | +0.0315 |
| PaDiM | gaussian_blur | mild | 0.6778 | 0.7753 | +0.0974 |
| PaDiM | gaussian_blur | moderate | 0.5367 | 0.6316 | +0.0949 |
| PaDiM | gaussian_blur | severe | 0.5094 | 0.6401 | +0.1308 |
| PaDiM | low_light | mild | 0.7112 | 0.8755 | +0.1642 |
| PaDiM | low_light | moderate | 0.6481 | 0.8695 | +0.2215 |
| PaDiM | low_light | severe | 0.5436 | 0.8171 | +0.2735 |
| PaDiM | motion_blur | mild | 0.6529 | 0.7635 | +0.1106 |
| PaDiM | motion_blur | moderate | 0.5568 | 0.7137 | +0.1569 |
| PaDiM | motion_blur | severe | 0.5373 | 0.6979 | +0.1606 |
| PaDiM | sensor_noise | mild | 0.6011 | 0.6754 | +0.0743 |
| PaDiM | sensor_noise | moderate | 0.5496 | 0.6330 | +0.0834 |
| PaDiM | sensor_noise | severe | 0.5219 | 0.5936 | +0.0717 |
| PatchCore | fog_haze | mild | 0.6104 | 0.7650 | +0.1546 |
| PatchCore | fog_haze | moderate | 0.5024 | 0.6458 | +0.1434 |
| PatchCore | fog_haze | severe | 0.5002 | 0.5759 | +0.0757 |
| PatchCore | gaussian_blur | mild | 0.8277 | 0.9242 | +0.0966 |
| PatchCore | gaussian_blur | moderate | 0.5392 | 0.7459 | +0.2067 |
| PatchCore | gaussian_blur | severe | 0.5000 | 0.7377 | +0.2377 |
| PatchCore | low_light | mild | 0.8668 | 0.9723 | +0.1056 |
| PatchCore | low_light | moderate | 0.7754 | 0.9640 | +0.1885 |
| PatchCore | low_light | severe | 0.5646 | 0.9648 | +0.4002 |
| PatchCore | motion_blur | mild | 0.7366 | 0.9116 | +0.1750 |
| PatchCore | motion_blur | moderate | 0.5047 | 0.8298 | +0.3251 |
| PatchCore | motion_blur | severe | 0.5000 | 0.8328 | +0.3328 |
| PatchCore | sensor_noise | mild | 0.7495 | 0.8987 | +0.1492 |
| PatchCore | sensor_noise | moderate | 0.6645 | 0.8382 | +0.1737 |
| PatchCore | sensor_noise | severe | 0.5939 | 0.7617 | +0.1677 |

## 2. Rescue Deltas by Corruption Type

### Dataset: MVTec-AD

| Model | Training | Corruption | Severity | Rescue Method | Degraded AUROC | Rescued AUROC | Rescue Delta |
|---|---|---|---|---|---|---|---|
| PaDiM | augmented | fog_haze | mild | Dehaze (Dark Channel) | 0.7238 | 0.6839 | -0.0399 |
| PaDiM | augmented | fog_haze | moderate | Dehaze (Dark Channel) | 0.6218 | 0.5828 | -0.0389 |
| PaDiM | augmented | fog_haze | severe | Dehaze (Dark Channel) | 0.5592 | 0.5577 | -0.0015 |
| PaDiM | augmented | gaussian_blur | mild | Wiener | 0.8478 | 0.5425 | -0.3053 |
| PaDiM | augmented | gaussian_blur | moderate | Wiener | 0.7475 | 0.5471 | -0.2005 |
| PaDiM | augmented | gaussian_blur | severe | Wiener | 0.7107 | 0.5161 | -0.1946 |
| PaDiM | augmented | low_light | mild | CLAHE | 0.8545 | 0.7894 | -0.0651 |
| PaDiM | augmented | low_light | mild | Retinex | 0.8545 | 0.7221 | -0.1324 |
| PaDiM | augmented | low_light | moderate | CLAHE | 0.8400 | 0.7678 | -0.0723 |
| PaDiM | augmented | low_light | moderate | Retinex | 0.8400 | 0.7331 | -0.1069 |
| PaDiM | augmented | low_light | severe | CLAHE | 0.8012 | 0.7742 | -0.0270 |
| PaDiM | augmented | low_light | severe | Retinex | 0.8012 | 0.7166 | -0.0846 |
| PaDiM | augmented | motion_blur | mild | Wiener (Motion PSF) | 0.8163 | 0.5481 | -0.2682 |
| PaDiM | augmented | motion_blur | moderate | Wiener (Motion PSF) | 0.7709 | 0.5612 | -0.2097 |
| PaDiM | augmented | motion_blur | severe | Wiener (Motion PSF) | 0.7377 | 0.5798 | -0.1580 |
| PaDiM | augmented | sensor_noise | mild | NLM Denoise | 0.7225 | 0.7089 | -0.0136 |
| PaDiM | augmented | sensor_noise | moderate | NLM Denoise | 0.6925 | 0.6299 | -0.0627 |
| PaDiM | augmented | sensor_noise | severe | NLM Denoise | 0.6535 | 0.5941 | -0.0593 |
| PaDiM | clean | fog_haze | mild | Dehaze (Dark Channel) | 0.6243 | 0.6229 | -0.0015 |
| PaDiM | clean | fog_haze | moderate | Dehaze (Dark Channel) | 0.5539 | 0.5554 | +0.0015 |
| PaDiM | clean | fog_haze | severe | Dehaze (Dark Channel) | 0.5214 | 0.5320 | +0.0106 |
| PaDiM | clean | gaussian_blur | mild | Wiener | 0.7723 | 0.5261 | -0.2462 |
| PaDiM | clean | gaussian_blur | moderate | Wiener | 0.6129 | 0.5160 | -0.0969 |
| PaDiM | clean | gaussian_blur | severe | Wiener | 0.5813 | 0.5011 | -0.0802 |
| PaDiM | clean | low_light | mild | CLAHE | 0.7742 | 0.7641 | -0.0101 |
| PaDiM | clean | low_light | mild | Retinex | 0.7742 | 0.6976 | -0.0766 |
| PaDiM | clean | low_light | moderate | CLAHE | 0.7174 | 0.7270 | +0.0097 |
| PaDiM | clean | low_light | moderate | Retinex | 0.7174 | 0.6997 | -0.0177 |
| PaDiM | clean | low_light | severe | CLAHE | 0.6640 | 0.6569 | -0.0071 |
| PaDiM | clean | low_light | severe | Retinex | 0.6640 | 0.6758 | +0.0118 |
| PaDiM | clean | motion_blur | mild | Wiener (Motion PSF) | 0.7207 | 0.5431 | -0.1777 |
| PaDiM | clean | motion_blur | moderate | Wiener (Motion PSF) | 0.6294 | 0.5347 | -0.0947 |
| PaDiM | clean | motion_blur | severe | Wiener (Motion PSF) | 0.5681 | 0.5594 | -0.0087 |
| PaDiM | clean | sensor_noise | mild | NLM Denoise | 0.6569 | 0.6512 | -0.0057 |
| PaDiM | clean | sensor_noise | moderate | NLM Denoise | 0.6105 | 0.5898 | -0.0207 |
| PaDiM | clean | sensor_noise | severe | NLM Denoise | 0.5784 | 0.5678 | -0.0106 |
| PatchCore | augmented | fog_haze | mild | Dehaze (Dark Channel) | 0.8491 | 0.7996 | -0.0495 |
| PatchCore | augmented | fog_haze | moderate | Dehaze (Dark Channel) | 0.7717 | 0.6607 | -0.1110 |
| PatchCore | augmented | fog_haze | severe | Dehaze (Dark Channel) | 0.6874 | 0.6134 | -0.0740 |
| PatchCore | augmented | gaussian_blur | mild | Wiener | 0.9274 | 0.5000 | -0.4274 |
| PatchCore | augmented | gaussian_blur | moderate | Wiener | 0.8588 | 0.5027 | -0.3562 |
| PatchCore | augmented | gaussian_blur | severe | Wiener | 0.8292 | 0.5022 | -0.3270 |
| PatchCore | augmented | low_light | mild | CLAHE | 0.9548 | 0.9453 | -0.0095 |
| PatchCore | augmented | low_light | mild | Retinex | 0.9548 | 0.8416 | -0.1131 |
| PatchCore | augmented | low_light | moderate | CLAHE | 0.9356 | 0.9233 | -0.0123 |
| PatchCore | augmented | low_light | moderate | Retinex | 0.9356 | 0.8214 | -0.1142 |
| PatchCore | augmented | low_light | severe | CLAHE | 0.8746 | 0.8850 | +0.0105 |
| PatchCore | augmented | low_light | severe | Retinex | 0.8746 | 0.7853 | -0.0893 |
| PatchCore | augmented | motion_blur | mild | Wiener (Motion PSF) | 0.9159 | 0.5528 | -0.3631 |
| PatchCore | augmented | motion_blur | moderate | Wiener (Motion PSF) | 0.8760 | 0.5610 | -0.3150 |
| PatchCore | augmented | motion_blur | severe | Wiener (Motion PSF) | 0.8389 | 0.5930 | -0.2459 |
| PatchCore | augmented | sensor_noise | mild | NLM Denoise | 0.8942 | 0.8730 | -0.0211 |
| PatchCore | augmented | sensor_noise | moderate | NLM Denoise | 0.8535 | 0.8199 | -0.0337 |
| PatchCore | augmented | sensor_noise | severe | NLM Denoise | 0.8149 | 0.7843 | -0.0306 |
| PatchCore | clean | fog_haze | mild | Dehaze (Dark Channel) | 0.7098 | 0.7005 | -0.0093 |
| PatchCore | clean | fog_haze | moderate | Dehaze (Dark Channel) | 0.6161 | 0.5859 | -0.0303 |
| PatchCore | clean | fog_haze | severe | Dehaze (Dark Channel) | 0.5623 | 0.5729 | +0.0107 |
| PatchCore | clean | gaussian_blur | mild | Wiener | 0.9173 | 0.5000 | -0.4173 |
| PatchCore | clean | gaussian_blur | moderate | Wiener | 0.6483 | 0.5000 | -0.1483 |
| PatchCore | clean | gaussian_blur | severe | Wiener | 0.5831 | 0.5157 | -0.0674 |
| PatchCore | clean | low_light | mild | CLAHE | 0.9607 | 0.9079 | -0.0527 |
| PatchCore | clean | low_light | mild | Retinex | 0.9607 | 0.8478 | -0.1129 |
| PatchCore | clean | low_light | moderate | CLAHE | 0.8910 | 0.9034 | +0.0124 |
| PatchCore | clean | low_light | moderate | Retinex | 0.8910 | 0.8248 | -0.0662 |
| PatchCore | clean | low_light | severe | CLAHE | 0.7840 | 0.8195 | +0.0354 |
| PatchCore | clean | low_light | severe | Retinex | 0.7840 | 0.7456 | -0.0384 |
| PatchCore | clean | motion_blur | mild | Wiener (Motion PSF) | 0.7824 | 0.5550 | -0.2274 |
| PatchCore | clean | motion_blur | moderate | Wiener (Motion PSF) | 0.6595 | 0.5626 | -0.0969 |
| PatchCore | clean | motion_blur | severe | Wiener (Motion PSF) | 0.6127 | 0.5977 | -0.0150 |
| PatchCore | clean | sensor_noise | mild | NLM Denoise | 0.8470 | 0.8140 | -0.0330 |
| PatchCore | clean | sensor_noise | moderate | NLM Denoise | 0.7638 | 0.7455 | -0.0183 |
| PatchCore | clean | sensor_noise | severe | NLM Denoise | 0.6966 | 0.6812 | -0.0154 |

### Dataset: VisA

| Model | Training | Corruption | Severity | Rescue Method | Degraded AUROC | Rescued AUROC | Rescue Delta |
|---|---|---|---|---|---|---|---|
| PaDiM | augmented | fog_haze | mild | Dehaze (Dark Channel) | 0.6405 | 0.5573 | -0.0832 |
| PaDiM | augmented | fog_haze | moderate | Dehaze (Dark Channel) | 0.5796 | 0.5402 | -0.0394 |
| PaDiM | augmented | fog_haze | severe | Dehaze (Dark Channel) | 0.5324 | 0.5097 | -0.0226 |
| PaDiM | augmented | gaussian_blur | mild | Wiener | 0.7753 | 0.5222 | -0.2530 |
| PaDiM | augmented | gaussian_blur | moderate | Wiener | 0.6316 | 0.5202 | -0.1114 |
| PaDiM | augmented | gaussian_blur | severe | Wiener | 0.6401 | 0.5012 | -0.1389 |
| PaDiM | augmented | low_light | mild | CLAHE | 0.8755 | 0.7728 | -0.1027 |
| PaDiM | augmented | low_light | mild | Retinex | 0.8755 | 0.6516 | -0.2239 |
| PaDiM | augmented | low_light | moderate | CLAHE | 0.8695 | 0.8064 | -0.0631 |
| PaDiM | augmented | low_light | moderate | Retinex | 0.8695 | 0.5850 | -0.2845 |
| PaDiM | augmented | low_light | severe | CLAHE | 0.8171 | 0.7919 | -0.0253 |
| PaDiM | augmented | low_light | severe | Retinex | 0.8171 | 0.6339 | -0.1832 |
| PaDiM | augmented | motion_blur | mild | Wiener (Motion PSF) | 0.7635 | 0.5142 | -0.2493 |
| PaDiM | augmented | motion_blur | moderate | Wiener (Motion PSF) | 0.7137 | 0.5183 | -0.1954 |
| PaDiM | augmented | motion_blur | severe | Wiener (Motion PSF) | 0.6979 | 0.5331 | -0.1648 |
| PaDiM | augmented | sensor_noise | mild | NLM Denoise | 0.6754 | 0.6389 | -0.0365 |
| PaDiM | augmented | sensor_noise | moderate | NLM Denoise | 0.6330 | 0.5917 | -0.0413 |
| PaDiM | augmented | sensor_noise | severe | NLM Denoise | 0.5936 | 0.5850 | -0.0086 |
| PaDiM | clean | fog_haze | mild | Dehaze (Dark Channel) | 0.5650 | 0.5515 | -0.0135 |
| PaDiM | clean | fog_haze | moderate | Dehaze (Dark Channel) | 0.5245 | 0.5342 | +0.0097 |
| PaDiM | clean | fog_haze | severe | Dehaze (Dark Channel) | 0.5009 | 0.5224 | +0.0215 |
| PaDiM | clean | gaussian_blur | mild | Wiener | 0.6778 | 0.5069 | -0.1710 |
| PaDiM | clean | gaussian_blur | moderate | Wiener | 0.5367 | 0.5132 | -0.0235 |
| PaDiM | clean | gaussian_blur | severe | Wiener | 0.5094 | 0.5161 | +0.0068 |
| PaDiM | clean | low_light | mild | CLAHE | 0.7112 | 0.6754 | -0.0358 |
| PaDiM | clean | low_light | mild | Retinex | 0.7112 | 0.6081 | -0.1031 |
| PaDiM | clean | low_light | moderate | CLAHE | 0.6481 | 0.6556 | +0.0075 |
| PaDiM | clean | low_light | moderate | Retinex | 0.6481 | 0.5842 | -0.0639 |
| PaDiM | clean | low_light | severe | CLAHE | 0.5436 | 0.5821 | +0.0384 |
| PaDiM | clean | low_light | severe | Retinex | 0.5436 | 0.5416 | -0.0020 |
| PaDiM | clean | motion_blur | mild | Wiener (Motion PSF) | 0.6529 | 0.5019 | -0.1510 |
| PaDiM | clean | motion_blur | moderate | Wiener (Motion PSF) | 0.5568 | 0.5007 | -0.0561 |
| PaDiM | clean | motion_blur | severe | Wiener (Motion PSF) | 0.5373 | 0.5350 | -0.0024 |
| PaDiM | clean | sensor_noise | mild | NLM Denoise | 0.6011 | 0.5748 | -0.0263 |
| PaDiM | clean | sensor_noise | moderate | NLM Denoise | 0.5496 | 0.5773 | +0.0277 |
| PaDiM | clean | sensor_noise | severe | NLM Denoise | 0.5219 | 0.5660 | +0.0441 |
| PatchCore | augmented | fog_haze | mild | Dehaze (Dark Channel) | 0.7650 | 0.7040 | -0.0610 |
| PatchCore | augmented | fog_haze | moderate | Dehaze (Dark Channel) | 0.6458 | 0.5439 | -0.1018 |
| PatchCore | augmented | fog_haze | severe | Dehaze (Dark Channel) | 0.5759 | 0.5165 | -0.0594 |
| PatchCore | augmented | gaussian_blur | mild | Wiener | 0.9242 | 0.5000 | -0.4242 |
| PatchCore | augmented | gaussian_blur | moderate | Wiener | 0.7459 | 0.5000 | -0.2459 |
| PatchCore | augmented | gaussian_blur | severe | Wiener | 0.7377 | 0.5000 | -0.2377 |
| PatchCore | augmented | low_light | mild | CLAHE | 0.9723 | 0.9027 | -0.0696 |
| PatchCore | augmented | low_light | mild | Retinex | 0.9723 | 0.7327 | -0.2396 |
| PatchCore | augmented | low_light | moderate | CLAHE | 0.9640 | 0.9254 | -0.0386 |
| PatchCore | augmented | low_light | moderate | Retinex | 0.9640 | 0.6403 | -0.3237 |
| PatchCore | augmented | low_light | severe | CLAHE | 0.9648 | 0.9370 | -0.0278 |
| PatchCore | augmented | low_light | severe | Retinex | 0.9648 | 0.6091 | -0.3557 |
| PatchCore | augmented | motion_blur | mild | Wiener (Motion PSF) | 0.9116 | 0.5000 | -0.4116 |
| PatchCore | augmented | motion_blur | moderate | Wiener (Motion PSF) | 0.8298 | 0.5000 | -0.3298 |
| PatchCore | augmented | motion_blur | severe | Wiener (Motion PSF) | 0.8328 | 0.5521 | -0.2807 |
| PatchCore | augmented | sensor_noise | mild | NLM Denoise | 0.8987 | 0.7992 | -0.0995 |
| PatchCore | augmented | sensor_noise | moderate | NLM Denoise | 0.8382 | 0.8162 | -0.0220 |
| PatchCore | augmented | sensor_noise | severe | NLM Denoise | 0.7617 | 0.7674 | +0.0057 |
| PatchCore | clean | fog_haze | mild | Dehaze (Dark Channel) | 0.6104 | 0.5849 | -0.0255 |
| PatchCore | clean | fog_haze | moderate | Dehaze (Dark Channel) | 0.5024 | 0.5040 | +0.0016 |
| PatchCore | clean | fog_haze | severe | Dehaze (Dark Channel) | 0.5002 | 0.5013 | +0.0010 |
| PatchCore | clean | gaussian_blur | mild | Wiener | 0.8277 | 0.5000 | -0.3277 |
| PatchCore | clean | gaussian_blur | moderate | Wiener | 0.5392 | 0.5000 | -0.0392 |
| PatchCore | clean | gaussian_blur | severe | Wiener | 0.5000 | 0.5000 | +0.0000 |
| PatchCore | clean | low_light | mild | CLAHE | 0.8668 | 0.7756 | -0.0912 |
| PatchCore | clean | low_light | mild | Retinex | 0.8668 | 0.7230 | -0.1438 |
| PatchCore | clean | low_light | moderate | CLAHE | 0.7754 | 0.7595 | -0.0159 |
| PatchCore | clean | low_light | moderate | Retinex | 0.7754 | 0.6496 | -0.1258 |
| PatchCore | clean | low_light | severe | CLAHE | 0.5646 | 0.6352 | +0.0706 |
| PatchCore | clean | low_light | severe | Retinex | 0.5646 | 0.5267 | -0.0379 |
| PatchCore | clean | motion_blur | mild | Wiener (Motion PSF) | 0.7366 | 0.5000 | -0.2366 |
| PatchCore | clean | motion_blur | moderate | Wiener (Motion PSF) | 0.5047 | 0.5000 | -0.0047 |
| PatchCore | clean | motion_blur | severe | Wiener (Motion PSF) | 0.5000 | 0.5000 | -0.0000 |
| PatchCore | clean | sensor_noise | mild | NLM Denoise | 0.7495 | 0.6723 | -0.0772 |
| PatchCore | clean | sensor_noise | moderate | NLM Denoise | 0.6645 | 0.6236 | -0.0408 |
| PatchCore | clean | sensor_noise | severe | NLM Denoise | 0.5939 | 0.5306 | -0.0633 |

