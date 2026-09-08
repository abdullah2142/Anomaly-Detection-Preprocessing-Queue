#!/usr/bin/env python3
"""Generate the two generalization-control notebooks from notebooks/02.

Both experiments answer the objection that the reported augmentation gains are a
corruption-matched oracle bound: training draws from the same 5 corruption types
and 3 severities used at test time, so the models may simply have memorised the
test distribution.

  A. leave_one_corruption_out -- train the augmented arm on 4 corruption types,
     test only on the 5th, which the model has never seen. Rotates through all
     five, so every corruption serves as the held-out one exactly once.

  B. severity_holdout -- train on mild + moderate only, test on severe. Answers
     the complementary question: does augmentation transfer to degradation worse
     than it trained for?

Both reuse notebook 02's corruption functions, dataset wrapper and engine setup
byte-for-byte, so the only difference from the published augmented arm is which
(corruption, severity) pairs the training data may draw from. That includes
keeping `seed=42` inside the augmentation call: it is a known defect
(Limitation 15) but the published arm has it too, and holding it constant keeps
the two arms comparable.

The clean-trained comparison rows already exist in the master CSV -- these runs
only produce the new augmented arm.

Usage:  python scripts/generalization/build_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

SOURCE = Path("notebooks/02_mvtec_patchcore_augmented.ipynb")
OUT_DIR = Path("scripts/generalization")

# 8 categories spanning both MVTec-AD families. Eight clusters puts the exact
# permutation test's floor at 2/2**8 = 0.0078, so a clean result can actually
# reach significance -- five categories could not (floor 0.0625).
CATEGORIES = ["carpet", "grid", "leather", "bottle", "cable", "capsule", "hazelnut", "screw"]
SEED = 42
ALL_CTYPES = ["low_light", "gaussian_blur", "motion_blur", "sensor_noise", "fog_haze"]
ALL_SEVS = ["mild", "moderate", "severe"]


def source_prelude() -> str:
    """Everything from notebook 02 up to its augmentation prep, plus the engine
    helpers: imports, timeout guard, corruption functions, dataset wrapper."""
    text = "".join(json.load(SOURCE.open())["cells"][0]["source"])
    lines = text.split("\n")
    stop = next(i for i, l in enumerate(lines) if l.startswith("# 4b."))
    head = "\n".join(lines[:stop])

    start = next(i for i, l in enumerate(lines) if l.startswith("class DisableCheckpointing"))
    # Stop before notebook 02's own resume block: it keys on (category, seed),
    # accepts any row count, and reads a hardcoded path to the ORIGINAL benchmark
    # CSV. These experiments define their own resume below.
    end = next(i for i, l in enumerate(lines)
               if i > start and ("possible_resume_paths" in l
                                 or "completed_keys" in l
                                 or l.startswith("# 6. Main Loop")))
    engine = "\n".join(lines[start:end]).rstrip()
    engine = engine.rstrip("- \n#")
    return head + "\n" + engine + "\n"


COMMON = '''
# ---------------------------------------------------------------------------
# Generalization control -- experiment configuration
# ---------------------------------------------------------------------------
from anomalib.models import Patchcore, Padim

CATEGORIES = {categories!r}
SEED = {seed}
MODELS = ["PatchCore", "PaDiM"]
ALL_CTYPES = {all_ctypes!r}
ALL_SEVS = {all_sevs!r}

OUTPUT_FILE = "results/{slug}.csv"
PARTIAL_FILE = "results/{slug}_partial.csv"
AUG_ROOT_BASE = "/kaggle/tmp/aug_{slug}"

os.makedirs("results", exist_ok=True)


def make_model(name):
    """Identical hyperparameters to the published benchmark arms."""
    if name == "PatchCore":
        return Patchcore(backbone="wide_resnet50_2", num_neighbors=9)
    return Padim(backbone="wide_resnet50_2",
                 layers=["layer1", "layer2", "layer3"], n_features=100)


def prepare_augmented_category(category, dst_root, aug_ctypes, aug_sevs,
                               aug_prob=0.5, rng_seed=0):
    """Write an augmented copy of one category, drawing only from aug_ctypes/aug_sevs.

    Deliberately identical to notebook 02's prepare_augmented_train_data() except
    for the restricted draw pools and the per-category destination -- including
    the fixed seed=42 passed to apply_corruption, so this arm carries the same
    known noise-realization defect as the published arm and stays comparable.
    """
    random.seed(rng_seed)
    src_train = os.path.join(MVTEC_ROOT, category, "train", "good")
    dst_train = os.path.join(dst_root, category, "train", "good")
    os.makedirs(dst_train, exist_ok=True)
    n_aug = 0
    for fname in sorted(os.listdir(src_train)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        dst_path = os.path.join(dst_train, fname)
        if os.path.exists(dst_path):
            continue
        img_bgr = cv2.imread(os.path.join(src_train, fname))
        if img_bgr is None:
            continue
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if random.random() < aug_prob:
            ctype = random.choice(aug_ctypes)
            sev = random.choice(aug_sevs)
            img = apply_corruption(img, ctype, sev, config, seed=42)
            n_aug += 1
        cv2.imwrite(dst_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    for subdir in ["test", "ground_truth"]:
        src = os.path.join(MVTEC_ROOT, category, subdir)
        dst = os.path.join(dst_root, category, subdir)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copytree(src, dst)
    print(f"    augmented {{n_aug}} training images from {{aug_ctypes}} x {{aug_sevs}}")


def evaluate(engine, model, base_test_data, ctype, sev):
    """One degradation inference pass on a held-out (corruption, severity)."""
    from torch.utils.data import DataLoader
    loader = DataLoader(
        CorruptedDatasetWrapper(base_test_data, ctype, sev, config),
        batch_size=32, num_workers=4, collate_fn=base_test_data.collate_fn)
    return safe_auroc(engine.test(model=model, dataloaders=loader)[0])


# ---------------------------------------------------------------------------
# Resume: a unit counts as complete only at its full row count.
# ---------------------------------------------------------------------------
completed_units = set()
all_results = []

for p in [OUTPUT_FILE, PARTIAL_FILE]:
    if os.path.exists(p):
        try:
            old_df = pd.read_csv(p)
            counts = old_df.groupby({unit_cols!r}).size()
            valid = [k for k, n in counts.items() if n == ROWS_PER_UNIT]
            for k in valid:
                completed_units.add(k if isinstance(k, tuple) else (k,))
            for k, n in counts.items():
                if n != ROWS_PER_UNIT:
                    print(f"Discarding partial unit {{k}} with {{n}}/{{ROWS_PER_UNIT}} rows.")
            keep = old_df[old_df.apply(
                lambda r: tuple(r[c] for c in {unit_cols!r}) in set(
                    k if isinstance(k, tuple) else (k,) for k in valid), axis=1)]
            all_results = keep.to_dict("records")
            print(f"Resumed {{len(completed_units)}} completed units from {{p}}.")
            break
        except Exception as e:
            print(f"Could not load existing results: {{e}}")
'''


LOCO_LOOP = '''
# ---------------------------------------------------------------------------
# Experiment A: leave one corruption out
#
# For each held-out corruption type, train on the other four and test ONLY on
# the held-out type. The clean-trained comparison rows for the same
# (category, ctype, severity) already exist in benchmark_master_combined.csv.
# ---------------------------------------------------------------------------
print(f"\\nGPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Categories: {CATEGORIES}\\nHeld-out types: {ALL_CTYPES}\\nSeed: {SEED}")

for category in CATEGORIES:
    # The test set is the pristine MVTec-AD category; build it once per category.
    dm_test = MVTecAD(root=MVTEC_ROOT, category=category,
                      train_batch_size=32, eval_batch_size=32)
    dm_test.setup(stage="test")
    base_test_data = dm_test.test_data

    for held_out in ALL_CTYPES:
        check_timeout()
        if (category, held_out) in completed_units:
            print(f"Skipping {category} / held-out {held_out}")
            continue

        print(f"\\n{'='*60}\\n{category.upper()} | held out: {held_out}\\n{'='*60}")
        aug_ctypes = [c for c in ALL_CTYPES if c != held_out]
        dst_root = os.path.join(AUG_ROOT_BASE, held_out)
        try:
            prepare_augmented_category(category, dst_root, aug_ctypes, ALL_SEVS,
                                       aug_prob=0.5, rng_seed=SEED)
            for model_name in MODELS:
                L.seed_everything(SEED)
                engine = make_engine()
                model = make_model(model_name)
                datamodule = MVTecAD(root=dst_root, category=category,
                                     train_batch_size=32, eval_batch_size=32)
                print(f"[TRAIN] {model_name} without {held_out}")
                engine.fit(model=model, datamodule=datamodule)

                for sev in ALL_SEVS:
                    auroc = evaluate(engine, model, base_test_data, held_out, sev)
                    all_results.append({
                        "model": model_name, "dataset": "MVTec-AD", "category": category,
                        "seed": SEED, "phase": "degradation", "ctype": held_out,
                        "severity": sev, "rescue": "none", "image_AUROC": auroc,
                        "training": "augmented_loco",
                        "experiment": "leave_one_corruption_out", "held_out": held_out})
                    save_results()
                del model, engine
                cleanup_memory()
            completed_units.add((category, held_out))
        except Exception as e:
            print(f"Error in {category}/{held_out}: {e}")
            save_results()
            raise
        finally:
            shutil.rmtree(os.path.join(dst_root, category), ignore_errors=True)

save_results()
print(f"\\n{'='*60}\\nLEAVE-ONE-CORRUPTION-OUT COMPLETE - {len(all_results)} rows\\n{'='*60}")
'''


SEV_LOOP = '''
# ---------------------------------------------------------------------------
# Experiment B: severity holdout
#
# Train on mild + moderate only (all five corruption types), test on severe.
# The clean-trained comparison rows for severe already exist in
# benchmark_master_combined.csv.
# ---------------------------------------------------------------------------
TRAIN_SEVS = ["mild", "moderate"]
TEST_SEV = "severe"

print(f"\\nGPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Categories: {CATEGORIES}\\nTrain severities: {TRAIN_SEVS} -> test: {TEST_SEV}")

for category in CATEGORIES:
    check_timeout()
    if (category,) in completed_units:
        print(f"Skipping {category}")
        continue

    print(f"\\n{'='*60}\\n{category.upper()} | train {TRAIN_SEVS} -> test {TEST_SEV}\\n{'='*60}")
    dm_test = MVTecAD(root=MVTEC_ROOT, category=category,
                      train_batch_size=32, eval_batch_size=32)
    dm_test.setup(stage="test")
    base_test_data = dm_test.test_data

    dst_root = os.path.join(AUG_ROOT_BASE, "mild_moderate")
    try:
        prepare_augmented_category(category, dst_root, ALL_CTYPES, TRAIN_SEVS,
                                   aug_prob=0.5, rng_seed=SEED)
        for model_name in MODELS:
            L.seed_everything(SEED)
            engine = make_engine()
            model = make_model(model_name)
            datamodule = MVTecAD(root=dst_root, category=category,
                                 train_batch_size=32, eval_batch_size=32)
            print(f"[TRAIN] {model_name} on {TRAIN_SEVS} only")
            engine.fit(model=model, datamodule=datamodule)

            for ctype in ALL_CTYPES:
                auroc = evaluate(engine, model, base_test_data, ctype, TEST_SEV)
                all_results.append({
                    "model": model_name, "dataset": "MVTec-AD", "category": category,
                    "seed": SEED, "phase": "degradation", "ctype": ctype,
                    "severity": TEST_SEV, "rescue": "none", "image_AUROC": auroc,
                    "training": "augmented_sevholdout",
                    "experiment": "severity_holdout", "held_out": TEST_SEV})
                save_results()
            del model, engine
            cleanup_memory()
        completed_units.add((category,))
    except Exception as e:
        print(f"Error in {category}: {e}")
        save_results()
        raise
    finally:
        shutil.rmtree(os.path.join(dst_root, category), ignore_errors=True)

save_results()
print(f"\\n{'='*60}\\nSEVERITY HOLDOUT COMPLETE - {len(all_results)} rows\\n{'='*60}")
'''


UNCONDITIONAL_LOOP = '''
# ---------------------------------------------------------------------------
# Experiment C: unconditional and misidentified rescue
#
# The benchmark only ever applies a rescue to the corruption it targets, chosen
# using that corruption's ground-truth identity. A deployed system has neither
# guarantee: it preprocesses a whole stream, most of which may be undegraded, and
# it can misclassify what degradation it is looking at.
#
#   Part A -- rescue applied to CLEAN images. Cost of unnecessary preprocessing.
#             The comparison is the clean baseline AUROC already in the master CSV.
#             Wiener uses mild-tier PSFs: the kernel a detector would pick if it
#             wrongly flagged mild blur on an undegraded frame.
#
#   Part B -- rescue applied to the WRONG corruption. Cost of misidentification.
#             The comparison is the degradation AUROC already in the master CSV.
#             Confusions are plausible detector errors, not an exhaustive cross.
#
# Clean-trained models only: that is the deployment-relevant arm, and the one the
# paper's single positive recommendation (CLAHE for low-light PatchCore) is about.
# ---------------------------------------------------------------------------
from torch.utils.data import DataLoader

RESCUE_ORDER = ["low_light", "gaussian_blur", "motion_blur", "sensor_noise", "fog_haze"]

# Plausible confusions a degradation classifier could make.
CONFUSIONS = [
    ("fog_haze",      "CLAHE"),                  # haze read as low light
    ("motion_blur",   "Wiener"),                 # motion read as defocus
    ("gaussian_blur", "Wiener (Motion PSF)"),    # defocus read as motion
    ("sensor_noise",  "Dehaze (Dark Channel)"),  # noise read as haze
    ("low_light",     "NLM Denoise"),            # low light read as noise
]


def all_rescue_methods(severity, config):
    """The six benchmark rescues as a flat list, PSFs taken at `severity`."""
    m = get_rescue_map(severity, config)
    out = []
    for ctype in RESCUE_ORDER:
        out.extend(m[ctype])
    return out


class RescueOnlyWrapper(Dataset):
    """Applies a rescue to the UNCORRUPTED image. Mirrors CorruptedDatasetWrapper
    exactly except that no corruption is applied first."""

    def __init__(self, base_dataset, rescue_func):
        self.base_dataset = base_dataset
        self.rescue_func = rescue_func

    def __len__(self):
        return len(self.base_dataset)

    def __getattr__(self, name):
        return getattr(self.base_dataset, name)

    def __getitem__(self, idx):
        import dataclasses
        item = self.base_dataset[idx]
        image = item.image if dataclasses.is_dataclass(item) else item["image"]
        if isinstance(image, torch.Tensor):
            img_np = image.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
        else:
            img_np = np.array(image).astype(np.uint8)
        final = self.rescue_func(img_np)
        tensor = torch.from_numpy(final).permute(2, 0, 1).float() / 255.0
        if dataclasses.is_dataclass(item):
            return dataclasses.replace(item, image=tensor)
        item["image"] = tensor
        return item


def record(model_name, ctype, severity, rescue, auroc, experiment, applied_to):
    all_results.append({
        "model": model_name, "dataset": "MVTec-AD", "category": category,
        "seed": SEED, "phase": "rescue", "ctype": ctype, "severity": severity,
        "rescue": rescue, "image_AUROC": auroc, "training": "clean",
        "experiment": experiment, "applied_to": applied_to})
    save_results()


print(f"\\nGPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Categories: {CATEGORIES}\\nSeed: {SEED}")

for category in CATEGORIES:
    for model_name in MODELS:
        check_timeout()
        if (category, model_name) in completed_units:
            print(f"Skipping {category} / {model_name}")
            continue

        print(f"\\n{'='*60}\\n{category.upper()} | {model_name}\\n{'='*60}")
        try:
            L.seed_everything(SEED)
            engine = make_engine()
            model = make_model(model_name)
            datamodule = MVTecAD(root=MVTEC_ROOT, category=category,
                                 train_batch_size=32, eval_batch_size=32)
            print(f"[TRAIN] {model_name} on clean data")
            engine.fit(model=model, datamodule=datamodule)

            dm_test = MVTecAD(root=MVTEC_ROOT, category=category,
                              train_batch_size=32, eval_batch_size=32)
            dm_test.setup(stage="test")
            base_test_data = dm_test.test_data

            # --- Part A: rescue on undegraded images -------------------------
            for r_name, r_func in all_rescue_methods("mild", config):
                print(f"[A] clean + {r_name}")
                loader = DataLoader(RescueOnlyWrapper(base_test_data, r_func),
                                    batch_size=32, num_workers=4,
                                    collate_fn=base_test_data.collate_fn)
                auroc = safe_auroc(engine.test(model=model, dataloaders=loader)[0])
                record(model_name, "clean", "none", r_name, auroc,
                       "rescue_on_clean", "undegraded")

            # --- Part B: rescue on the wrong corruption ----------------------
            for actual_ctype, wrong_name in CONFUSIONS:
                for sev in ALL_SEVS:
                    methods = dict(all_rescue_methods(sev, config))
                    r_func = methods[wrong_name]
                    print(f"[B] {actual_ctype} ({sev}) + {wrong_name}  <- misidentified")
                    loader = DataLoader(
                        CorruptedDatasetWrapper(base_test_data, actual_ctype, sev,
                                                config, rescue_func=r_func),
                        batch_size=32, num_workers=4,
                        collate_fn=base_test_data.collate_fn)
                    auroc = safe_auroc(engine.test(model=model, dataloaders=loader)[0])
                    record(model_name, actual_ctype, sev, wrong_name, auroc,
                           "rescue_mismatched", "wrong_corruption")

            del model, engine
            cleanup_memory()
            completed_units.add((category, model_name))
        except Exception as e:
            print(f"Error in {category}/{model_name}: {e}")
            save_results()
            raise

save_results()
print(f"\\n{'='*60}\\nUNCONDITIONAL RESCUE COMPLETE - {len(all_results)} rows\\n{'='*60}")
'''


def build(slug, unit_cols, rows_per_unit, loop, title, blurb):
    prelude = source_prelude()
    common = COMMON.format(categories=CATEGORIES, seed=SEED, slug=slug,
                           all_ctypes=ALL_CTYPES, all_sevs=ALL_SEVS,
                           unit_cols=unit_cols)
    body = f"ROWS_PER_UNIT = {rows_per_unit}\n" + common + loop
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [f"# {title}\n", "\n", blurb]},
            {"cell_type": "code", "execution_count": None, "metadata": {},
             "outputs": [], "source": (prelude + body).split("\n")},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    # nbformat wants each source line to keep its newline
    src = nb["cells"][1]["source"]
    nb["cells"][1]["source"] = [l + "\n" for l in src[:-1]] + [src[-1]]
    path = OUT_DIR / f"{slug}.ipynb"
    with path.open("w") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    build("leave_one_corruption_out", ["category", "held_out"], 6, LOCO_LOOP,
          "Leave-One-Corruption-Out Generalization Control",
          "Trains the augmented arm on 4 of the 5 corruption types and tests only on the "
          "held-out 5th, rotating through all five. Answers whether the published "
          "augmentation gain reflects genuine robustness or a corruption-matched oracle. "
          "Clean-trained comparison rows already exist in the master CSV.\n")
    build("unconditional_rescue", ["category", "model"], 21, UNCONDITIONAL_LOOP,
          "Unconditional and Misidentified Rescue",
          "Applies each rescue to undegraded images (cost of unnecessary preprocessing) "
          "and to the wrong corruption (cost of misidentification). The benchmark only "
          "ever applies a rescue to its matched corruption, selected by ground-truth "
          "identity; neither guarantee holds in deployment. Clean-trained models only. "
          "Comparison rows already exist in the master CSV.\n")
    build("severity_holdout", ["category"], 10, SEV_LOOP,
          "Severity-Holdout Generalization Control",
          "Trains the augmented arm on mild and moderate corruption only, then tests on "
          "severe. Answers whether augmentation transfers to degradation worse than it "
          "trained for. Clean-trained comparison rows already exist in the master CSV.\n")
