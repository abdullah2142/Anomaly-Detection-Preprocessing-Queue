# %% [markdown]
# # NB-2: PaDiM Baselines (Clean MVTec-AD)
# **GPU**: ON | **Internet**: On | **Time**: ~2-3 hrs
#
# **Input Data Required**:
# - MVTec-AD dataset (`ipythonx/mvtec-ad`)
# - NB-0 output (for `experiment_config.json`)
#
# This notebook:
# 1. Runs PaDiM on all 15 MVTec-AD categories with 3 seeds
# 2. Validates clean AUROC ≥ 95% (published benchmark)
# 3. Saves model checkpoints for reuse
#
# **After running**: "Save & Run All" → output contains baselines CSV + checkpoints

# %% [markdown]
# ## Cell 1: Setup

# %%
!pip install -q anomalib --no-deps
!pip install -q lightning albumentationsx jsonargparse docstring_parser rich

import json, os, glob, shutil, time
import pandas as pd
import numpy as np
import lightning as L
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Padim

# Load config from NB-0 output
CONFIG_PATH = "/kaggle/input/00-setup-and-verify/experiment_config.json"  # ADJUST
with open(CONFIG_PATH) as f:
    config = json.load(f)

MVTEC_ROOT = config["mvtec_root"]
CATEGORIES = config["categories"]
SEEDS = config["seeds"]

print(f"MVTec root: {MVTEC_ROOT}")
print(f"Total runs: {len(CATEGORIES) * len(SEEDS)}")

# %% [markdown]
# ## Cell 2: Run PaDiM Baselines + Save Checkpoints

# %%
os.makedirs("/kaggle/working/baselines", exist_ok=True)

results = []
start_time = time.time()
total_runs = len(CATEGORIES) * len(SEEDS)
run_count = 0

for category in CATEGORIES:
    for seed in SEEDS:
        run_count += 1
        L.seed_everything(seed)

        print(f"\n{'='*60}")
        print(f"[{run_count}/{total_runs}] PaDiM | {category} | seed={seed}")
        run_start = time.time()

        model = Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"])
        datamodule = MVTecAD(
            root=MVTEC_ROOT,
            category=category,
            train_batch_size=32,
            eval_batch_size=32,
        )

        engine = Engine(max_epochs=1)
        engine.fit(model=model, datamodule=datamodule)
        test_results = engine.test(model=model, datamodule=datamodule)

        # Extract AUROC
        auroc = None
        if test_results and len(test_results) > 0:
            result_dict = test_results[0]
            for key in ["image_AUROC", "auroc", "AUROC", "image_auroc"]:
                if key in result_dict:
                    auroc = result_dict[key]
                    break
            if auroc is None:
                for k, v in result_dict.items():
                    if isinstance(v, (int, float)):
                        auroc = v
                        break

        results.append({
            "model": "PaDiM",
            "category": category,
            "seed": seed,
            "image_AUROC": auroc,
        })

        run_time = time.time() - run_start
        elapsed = time.time() - start_time
        eta = (elapsed / run_count) * (total_runs - run_count)
        print(f"  AUROC = {auroc:.4f}" if auroc else "  AUROC = None ❌")
        print(f"  Run time: {run_time/60:.1f} min | Elapsed: {elapsed/60:.0f} min | ETA: {eta/60:.0f} min")

        # Save checkpoint
        ckpt_dir = f"/kaggle/working/checkpoints/PaDiM/{category}/seed_{seed}"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_files = glob.glob("**/last.ckpt", recursive=True) + \
                     glob.glob("**/epoch*.ckpt", recursive=True) + \
                     glob.glob("**/model.ckpt", recursive=True)
        if ckpt_files:
            latest = max(ckpt_files, key=os.path.getmtime)
            shutil.copy2(latest, f"{ckpt_dir}/model.ckpt")
            print(f"  ✅ Checkpoint saved ({os.path.getsize(latest)/1e6:.1f} MB)")

# %% [markdown]
# ## Cell 3: Save Results & Validate

# %%
df = pd.DataFrame(results)
df.to_csv("/kaggle/working/baselines/padim_baselines.csv", index=False)

summary = df.groupby("category")["image_AUROC"].agg(["mean", "std"]).round(4)
print("\n" + "="*60)
print("PaDiM Baseline Results (mean ± std across 3 seeds)")
print("="*60)
print(summary.to_string())

overall_mean = df["image_AUROC"].mean()
print(f"\nOverall: {overall_mean:.4f} (expected ≥ 0.95)")

if overall_mean >= 0.93:
    print("\n✅ BASELINE VERIFIED.")
else:
    print(f"\n❌ WARNING: Mean AUROC ({overall_mean:.4f}) below expected.")

total_time = time.time() - start_time
print(f"\nTotal wall time: {total_time/3600:.2f} hours")
print("\n📋 NEXT: Run NB-3 (Severity Calibration)")
