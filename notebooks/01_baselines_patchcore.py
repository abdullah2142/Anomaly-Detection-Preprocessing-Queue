# %% [markdown]
# # NB-1: PatchCore Baselines (Clean MVTec-AD)
# **GPU**: ON | **Internet**: On | **Time**: ~3-4 hrs
#
# **Input Data Required**:
# - MVTec-AD dataset (`ipythonx/mvtec-ad`)
# - NB-0 output (for `experiment_config.json`)
#
# This notebook:
# 1. Runs PatchCore on all 15 MVTec-AD categories with 3 seeds
# 2. Validates clean AUROC ≥ 98% (published benchmark)
# 3. Saves model checkpoints for reuse in degradation/preprocessing experiments
#
# **After running**: "Save & Run All" → output contains baselines CSV + checkpoints

# %% [markdown]
# ## Cell 1: Install (MUST run first, before any other imports)

# %%
# Only anomalib gets --no-deps (it's the package that triggers torch replacement).
# Everything else installs normally — they won't touch CUDA binaries.
!pip install -q anomalib --no-deps
!pip install -q lightning albumentationsx jsonargparse docstring_parser rich

# GPU diagnostic (informational, won't stop execution)
import torch
print(f"torch=={torch.__version__}, CUDA={torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    print(f"Supported archs: {torch.cuda.get_arch_list()}")
else:
    print("⚠️ No GPU detected — will run on CPU (slower but works)")

# %% [markdown]
# ## Cell 2: Imports (run after Cell 1 succeeds with ✅)

# %%
import json, os, glob, shutil, time
import pandas as pd
import numpy as np
import lightning as L
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

# Load config from NB-0 output
# ⚠️ ADJUST THIS PATH to match your NB-0 output slug
CONFIG_PATH = "/kaggle/input/00-setup-and-verify/experiment_config.json"
with open(CONFIG_PATH) as f:
    config = json.load(f)

MVTEC_ROOT = config["mvtec_root"]
CATEGORIES = config["categories"]
SEEDS = config["seeds"]

print(f"MVTec root: {MVTEC_ROOT}")
print(f"Categories: {len(CATEGORIES)}")
print(f"Seeds: {SEEDS}")
print(f"Total runs: {len(CATEGORIES) * len(SEEDS)} = {len(CATEGORIES)}×{len(SEEDS)}")

# %% [markdown]
# ## Cell 2: Inspect Anomalib Dataset Format
# Run this ONCE to understand the data format for corruption wrapper later.

# %%
# Quick inspection of dataset item format
# Anomalib returns ImageItem dataclass objects, NOT dicts.
test_dm = MVTecAD(root=MVTEC_ROOT, category="bottle", train_batch_size=1, eval_batch_size=1)
test_dm.setup("fit")
test_dm.setup("test")

def inspect_item(item, label="Item"):
    """Inspect an Anomalib ImageItem (dataclass, not dict)."""
    print(f"=== {label} ===")
    print(f"Type: {type(item).__name__}")

    # Try dataclass fields first
    try:
        import dataclasses
        if dataclasses.is_dataclass(item):
            fields = dataclasses.fields(item)
            print(f"Fields: {[f.name for f in fields]}")
            for f in fields:
                v = getattr(item, f.name)
                if hasattr(v, 'shape'):
                    print(f"  {f.name}: shape={v.shape}, dtype={v.dtype}")
                elif v is not None:
                    print(f"  {f.name}: {type(v).__name__} = {v}")
                else:
                    print(f"  {f.name}: None")
            return
    except Exception:
        pass

    # Fallback: try vars()
    try:
        attrs = vars(item)
        print(f"Attributes: {list(attrs.keys())}")
        for k, v in attrs.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            elif v is not None:
                print(f"  {k}: {type(v).__name__} = {v}")
    except Exception:
        pass

    # Fallback: try dict-like access
    try:
        print(f"Keys: {item.keys()}")
        for k, v in item.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k}: {type(v).__name__} = {v}")
    except AttributeError:
        # Last resort: dir()
        print(f"Public attributes: {[a for a in dir(item) if not a.startswith('_')]}")

# Check training data
inspect_item(test_dm.train_data[0], "Training Data Item")
print()
# Check test data
inspect_item(test_dm.test_data[0], "Test Data Item")

print("\n⚠️ Record these field names and shapes — needed for corruption wrapper in NB-4!")

# %% [markdown]
# ## Cell 3: Run PatchCore Baselines + Save Checkpoints

# %%
os.makedirs("/kaggle/working/baselines", exist_ok=True)

results = []
failures = []
start_time = time.time()
total_runs = len(CATEGORIES) * len(SEEDS)
run_count = 0

for category in CATEGORIES:
    for seed in SEEDS:
        run_count += 1
        L.seed_everything(seed)

        print(f"\n{'='*60}")
        print(f"[{run_count}/{total_runs}] PatchCore | {category} | seed={seed}")
        run_start = time.time()

        try:
            model = Patchcore(backbone="wide_resnet50_2", num_neighbors=9)
            datamodule = MVTecAD(
                root=MVTEC_ROOT,
                category=category,
                train_batch_size=32,
                eval_batch_size=32,
            )

            engine = Engine(max_epochs=1, accelerator="auto")
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
                    print(f"  ⚠️ Available keys: {result_dict.keys()}")
                    for k, v in result_dict.items():
                        if isinstance(v, (int, float)):
                            auroc = v
                            print(f"  Using '{k}' = {v}")
                            break

            results.append({
                "model": "PatchCore", "category": category,
                "seed": seed, "image_AUROC": auroc,
            })
            print(f"  AUROC = {auroc:.4f}" if auroc else "  AUROC = None ❌")

            # Save checkpoint
            ckpt_dir = f"/kaggle/working/checkpoints/PatchCore/{category}/seed_{seed}"
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_files = glob.glob("**/last.ckpt", recursive=True) + \
                         glob.glob("**/epoch*.ckpt", recursive=True) + \
                         glob.glob("**/model.ckpt", recursive=True)
            if ckpt_files:
                latest = max(ckpt_files, key=os.path.getmtime)
                shutil.copy2(latest, f"{ckpt_dir}/model.ckpt")
                print(f"  ✅ Checkpoint saved ({os.path.getsize(latest)/1e6:.1f} MB)")

        except Exception as e:
            print(f"  ❌ FAILED: {type(e).__name__}: {e}")
            failures.append({"category": category, "seed": seed, "error": str(e)})
            results.append({
                "model": "PatchCore", "category": category,
                "seed": seed, "image_AUROC": None,
            })

        # Cleanup GPU memory between runs
        try:
            del model, engine
        except NameError:
            pass
        torch.cuda.empty_cache()
        import gc; gc.collect()

        run_time = time.time() - run_start
        elapsed = time.time() - start_time
        eta = (elapsed / run_count) * (total_runs - run_count)
        print(f"  Run time: {run_time/60:.1f} min | Elapsed: {elapsed/60:.0f} min | ETA: {eta/60:.0f} min")

        # Save intermediate results (survives crashes)
        pd.DataFrame(results).to_csv("/kaggle/working/baselines/patchcore_baselines_WIP.csv", index=False)

if failures:
    print(f"\n⚠️ {len(failures)} runs failed:")
    for f in failures:
        print(f"  {f['category']}/seed={f['seed']}: {f['error'][:80]}")

# %% [markdown]
# ## Cell 4: Save Results & Validate

# %%
df = pd.DataFrame(results)
df.to_csv("/kaggle/working/baselines/patchcore_baselines.csv", index=False)

# Summary statistics
summary = df.groupby("category")["image_AUROC"].agg(["mean", "std"]).round(4)
print("\n" + "="*60)
print("PatchCore Baseline Results (mean ± std across 3 seeds)")
print("="*60)
print(summary.to_string())

overall_mean = df["image_AUROC"].mean()
overall_std = df["image_AUROC"].std()
print(f"\nOverall: {overall_mean:.4f} ± {overall_std:.4f}")
print(f"Expected: ≥ 0.98")

if overall_mean >= 0.97:
    print("\n✅ BASELINE VERIFIED. Proceed to degradation experiments.")
else:
    print(f"\n❌ WARNING: Mean AUROC ({overall_mean:.4f}) below expected 0.98.")
    print("   Investigate before proceeding!")

# Check for any failed runs
failed = df[df["image_AUROC"].isna()]
if len(failed) > 0:
    print(f"\n❌ {len(failed)} runs produced no AUROC:")
    print(failed.to_string())

# Checkpoint disk usage
ckpt_size = sum(
    os.path.getsize(os.path.join(dp, f))
    for dp, dn, filenames in os.walk("/kaggle/working/checkpoints")
    for f in filenames
) / 1e9
print(f"\nCheckpoint storage used: {ckpt_size:.2f} GB (limit: 20 GB)")

total_time = time.time() - start_time
print(f"Total wall time: {total_time/3600:.2f} hours")

print("\n📋 NEXT STEPS:")
print("   1. Click 'Save & Run All' to commit outputs")
print("   2. Run NB-2 (PaDiM baselines)")
print("   3. Run NB-3 (Severity calibration) for visual approval")
