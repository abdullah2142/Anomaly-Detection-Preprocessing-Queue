# Wiener PSF Rerun — Kaggle Execution Guide

## What These Notebooks Do

The original experiment notebooks hardcoded the Wiener deconvolution PSF at the
**severe** tier (`σ=25, k=281` for Gaussian; `k=151` for motion) for all severity
levels. This means mild and moderate images were deconvolved with a grossly
mismatched kernel (5× too wide), inflating the reported Wiener harm.

These 8 rerun notebooks **retrain each model identically** then execute **only**
the 4 Wiener rescue inferences per (category, seed) pair with the **correct,
per-severity PSF parameters**:

| Corruption | Severity | Wiener PSF |
|---|---|---|
| Gaussian Blur | Mild | σ=5, k=31 |
| Gaussian Blur | Moderate | σ=15, k=101 |
| Motion Blur | Mild | k=31 |
| Motion Blur | Moderate | k=81 |

Severe rows are NOT re-run — they were already oracle-correct.

---

## Prerequisites

### Datasets (Kaggle)
- **MVTec-AD** (notebooks 01–04): [`ipythonx/mvtec-ad`](https://www.kaggle.com/datasets/ipythonx/mvtec-ad)
- **VisA** (notebooks 05–08): [`ess1004/visa-anomaly-detection`](https://www.kaggle.com/datasets/ess1004/visa-anomaly-detection)

### Configuration File
Each notebook loads corruption parameters from `data/experiment_config.json`.
Upload this file as a Kaggle dataset or include it in your notebook's input:
- **GitHub**: `https://github.com/<your-username>/Anomaly-Detection-Preprocessing-Queue/blob/main/data/experiment_config.json`

### Hardware
- **Accelerator**: T4 GPU (required — these notebooks use PyTorch + anomalib)
- **Persistence**: Enable "Save Version" to retain output CSVs

---

## Notebook → Session Mapping

You can run these in **3 Kaggle sessions** (or 4 if you prefer smaller batches):

### Session 1 — MVTec-AD Clean (both models)
| Notebook | Model | Training | Categories | Est. Time |
|---|---|---|---|---|
| `rerun_01_mvtec_patchcore_clean.ipynb` | PatchCore | Clean | 15 × 3 seeds | ~2 hrs |
| `rerun_03_mvtec_padim_clean.ipynb` | PaDiM | Clean | 15 × 3 seeds | ~2 hrs |
| **Total** | | | | **~4 hrs** |

### Session 2 — MVTec-AD Augmented (both models)
| Notebook | Model | Training | Categories | Est. Time |
|---|---|---|---|---|
| `rerun_02_mvtec_patchcore_aug.ipynb` | PatchCore | Augmented | 15 × 3 seeds | ~2.5 hrs |
| `rerun_04_mvtec_padim_aug.ipynb` | PaDiM | Augmented | 15 × 3 seeds | ~2.5 hrs |
| **Total** | | | | **~5 hrs** |

### Session 3 — VisA augmented (fits comfortably in one session)
| Notebook | Model | Training | Pairs | Observed Time |
|---|---|---|---|---|
| `rerun_06_visa_patchcore_aug.ipynb` | PatchCore | Augmented | 4 cats × 3 seeds = 12 | ~4–5 hrs |
| `rerun_08_visa_padim_aug.ipynb` | PaDiM | Augmented | 4 cats × 3 seeds = 12 | ~4–5 hrs |

### Sessions 4–7 — VisA clean (⚠️ split these; they exceed 12 hours)
`rerun_05` and `rerun_07` cover **12** VisA categories, three times the work of the
augmented arm, so a full pass runs ~12–15 hrs against Kaggle's 12-hour limit.
Split each into two sessions by editing `CATEGORIES` in the config cell:

```python
# session A
CATEGORIES = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1"]
# session B
CATEGORIES = ["macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]
```

Resume works across sessions: attach the previous session's output as a Kaggle
dataset and the notebook auto-restores from any `*partial*.csv` under
`/kaggle/input/`. A `(category, seed)` pair counts as complete at
`EXPECTED_ROWS_PER_PAIR = 4` rows — the 2 corruptions × 2 severities × 1 Wiener
rescue this rerun emits. Do not restore that constant to the original
benchmark's 34; at 34 no pair ever completes and finished rows are dropped
rather than resumed.

---

## Step-by-Step Instructions

### 1. Upload Notebook to Kaggle
1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. Click **File → Upload Notebook** and select the `.ipynb` file
3. Or import directly from your GitHub repository

### 2. Attach Dataset
1. In the right sidebar, click **+ Add Data**
2. Search for and add:
   - `ipythonx/mvtec-ad` (for MVTec notebooks 01–04)
   - `ess1004/visa-anomaly-detection` (for VisA notebooks 05–08)
3. Also add `experiment_config.json` if it's hosted as a separate Kaggle dataset

### 3. Configure Environment
1. **Accelerator**: Set to **GPU T4 x2** (or T4 x1 — either works)
2. **Internet**: Enable internet access (required for `pip install` of dependencies)
3. **Persistence**: Ensure "Save Version" output will be available after the run

### 4. Verify Paths
The notebooks expect:
- **MVTec-AD**: `/kaggle/input/mvtec-ad/` (or similar — check your dataset mount path)
- **VisA**: `/kaggle/input/visa-anomaly-detection/` (or similar)
- **Config**: The config JSON path — update `CONFIG_FILE` if your mount path differs

### 5. Run All Cells
1. Click **Run All** or `Shift+Enter` through each cell
2. The notebook will:
   - Install dependencies (anomalib, lightning, etc.)
   - Load the experiment config
   - For each (category, seed):
     - Train the model from scratch (identical to original)
     - Run 4 Wiener rescue inferences (Gaussian mild/mod + Motion mild/mod)
     - Append results to the output CSV
   - Resume automatically if the session restarts (uses partial CSV)

### 6. Save and Download
1. Once complete, click **Save Version** → **Save & Run All (Commit)**
2. After the committed version finishes, go to **Output** tab
3. Download the output CSV:
   - `results/rerun_01_mvtec_patchcore_clean.csv` (etc.)

### 7. Collect All 8 CSVs
Download all output CSVs to your local `data/` directory:
```
data/rerun_01_mvtec_patchcore_clean.csv
data/rerun_02_mvtec_patchcore_aug.csv
data/rerun_03_mvtec_padim_clean.csv
data/rerun_04_mvtec_padim_aug.csv
data/rerun_05_visa_patchcore_clean.csv
data/rerun_06_visa_patchcore_aug.csv
data/rerun_07_visa_padim_clean.csv
data/rerun_08_visa_padim_aug.csv
```

### 8. Merge Into Master CSV
Once you have all 8 files locally, run the merge script (which will be provided
after you confirm the reruns are complete). It will:
1. Drop the 1,104 old Wiener mild/moderate rows from `benchmark_master_combined.csv`
2. Insert the 1,104 corrected rows
3. Re-validate the merged CSV

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: anomalib` | Enable internet access in notebook settings |
| OOM on PaDiM augmented | Already handled: `n_features=100` caps memory |
| Session timeout (12h limit) | Resume-safe: re-run the same notebook and it skips completed (cat, seed) pairs |
| Config file not found | Update the `CONFIG_FILE` path to match your Kaggle dataset mount |
| Wrong dataset path | Check `/kaggle/input/` for the exact dataset directory name |
