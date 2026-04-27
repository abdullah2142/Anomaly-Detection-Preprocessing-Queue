# %% [markdown]
# # NB-10: Analysis & Figures
# **GPU**: Off | **Internet**: Off | **Time**: ~30 min
#
# **Input Data Required** (add ALL as input data sources):
# - NB-1 output (PatchCore baselines)
# - NB-2 output (PaDiM baselines)
# - NB-4 output (PatchCore degradation)
# - NB-5 output (PaDiM degradation)
# - NB-6a output (PatchCore preprocessing — blur/light)
# - NB-6b output (PatchCore preprocessing — noise/fog/combined)
# - NB-8 output (Training corruption)
#
# Generates all analysis figures and the recommendation matrix.

# %% [markdown]
# ## Cell 1: Load All Results

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams.update({
    'figure.figsize': (12, 7),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

os.makedirs("/kaggle/working/figures", exist_ok=True)

# ⚠️ ADJUST these paths to match your notebook output slugs
# Pattern: /kaggle/input/<notebook-slug>/<path-within-output>

# Degradation results
pc_deg = pd.read_csv("/kaggle/input/04-degradation-patchcore/degradation/patchcore_degradation.csv")
padim_deg = pd.read_csv("/kaggle/input/05-degradation-padim/degradation/padim_degradation.csv")
degradation = pd.concat([pc_deg, padim_deg], ignore_index=True)

# Preprocessing results
pc_pre1 = pd.read_csv("/kaggle/input/06a-preprocessing-patchcore-blur-light/preprocessing/patchcore_preprocess_blur_light.csv")
pc_pre2 = pd.read_csv("/kaggle/input/06b-preprocessing-patchcore-noise-fog/preprocessing/patchcore_preprocess_noise_fog.csv")
preprocessing = pd.concat([pc_pre1, pc_pre2], ignore_index=True)

# Training corruption
train_corr = pd.read_csv("/kaggle/input/08-training-corruption/training_corruption/training_corruption.csv")

print(f"Degradation: {len(degradation)} rows")
print(f"Preprocessing: {len(preprocessing)} rows")
print(f"Training corruption: {len(train_corr)} rows")

# %% [markdown]
# ## Cell 2: Figure 1 — Degradation Curves (AUROC vs Severity)

# %%
SEVERITY_ORDER = ["none", "mild", "moderate", "severe"]
CTYPES = ["low_light", "gaussian_blur", "motion_blur", "sensor_noise", "fog_haze"]
COLORS = {"low_light": "#FF6B35", "gaussian_blur": "#3498DB", "motion_blur": "#9B59B6",
          "sensor_noise": "#E74C3C", "fog_haze": "#1ABC9C"}

for model_name in ["PatchCore", "PaDiM"]:
    fig, ax = plt.subplots(figsize=(10, 6))
    model_data = degradation[degradation["model"] == model_name]

    for ctype in CTYPES:
        means, stds = [], []
        for sev in SEVERITY_ORDER:
            if sev == "none":
                subset = model_data[model_data["corruption_type"] == "clean"]
            else:
                subset = model_data[(model_data["corruption_type"] == ctype) &
                                    (model_data["severity"] == sev)]
            means.append(subset["image_AUROC"].mean())
            stds.append(subset["image_AUROC"].std())

        ax.errorbar(range(4), means, yerr=stds, marker='o', linewidth=2,
                    label=ctype.replace("_", " ").title(), color=COLORS[ctype],
                    capsize=4, capthick=1.5)

    ax.set_xticks(range(4))
    ax.set_xticklabels(["Clean", "Mild", "Moderate", "Severe"])
    ax.set_xlabel("Severity Level")
    ax.set_ylabel("Image-Level AUROC")
    ax.set_title(f"{model_name} — AUROC Degradation Curves")
    ax.legend(loc="lower left")
    ax.set_ylim([0.4, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"/kaggle/working/figures/degradation_curves_{model_name.lower()}.png", dpi=200)
    plt.show()

# %% [markdown]
# ## Cell 3: Figure 2 — Preprocessing Recovery Bar Chart (Severe)

# %%
# Compare: degraded (no preprocess) vs HistMatch vs best preprocessing at SEVERE level
PREPROCESS_NAMES = {
    "low_light": "CLAHE",
    "gaussian_blur": "Wiener",
    "motion_blur": "UnsharpMask",
    "sensor_noise": "NLM",
    "fog_haze": "DarkChannelPrior",
}

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(CTYPES))
width = 0.25

# Get clean baseline
pc_clean = degradation[(degradation["model"] == "PatchCore") &
                        (degradation["corruption_type"] == "clean")]["image_AUROC"].mean()

degraded_means, hist_means, best_means = [], [], []
degraded_stds, hist_stds, best_stds = [], [], []

for ctype in CTYPES:
    # Degraded (severe, no preprocessing) — from degradation results
    deg = degradation[(degradation["model"] == "PatchCore") &
                      (degradation["corruption_type"] == ctype) &
                      (degradation["severity"] == "severe")]
    degraded_means.append(deg["image_AUROC"].mean())
    degraded_stds.append(deg["image_AUROC"].std())

    # HistMatch (severe)
    hist = preprocessing[(preprocessing["corruption_type"] == ctype) &
                         (preprocessing["severity"] == "severe") &
                         (preprocessing["preprocessing"] == "HistMatch")]
    hist_means.append(hist["image_AUROC"].mean() if len(hist) > 0 else np.nan)
    hist_stds.append(hist["image_AUROC"].std() if len(hist) > 0 else 0)

    # Best preprocessing (severe)
    best_name = PREPROCESS_NAMES[ctype]
    best = preprocessing[(preprocessing["corruption_type"] == ctype) &
                         (preprocessing["severity"] == "severe") &
                         (preprocessing["preprocessing"] == best_name)]
    best_means.append(best["image_AUROC"].mean() if len(best) > 0 else np.nan)
    best_stds.append(best["image_AUROC"].std() if len(best) > 0 else 0)

bars1 = ax.bar(x - width, degraded_means, width, yerr=degraded_stds,
               label="Degraded (no preprocess)", color="#E74C3C", alpha=0.8, capsize=3)
bars2 = ax.bar(x, hist_means, width, yerr=hist_stds,
               label="HistMatch (naive baseline)", color="#F39C12", alpha=0.8, capsize=3)
bars3 = ax.bar(x + width, best_means, width, yerr=best_stds,
               label="Best preprocessing", color="#27AE60", alpha=0.8, capsize=3)

ax.axhline(y=pc_clean, color='blue', linestyle='--', alpha=0.5, label=f'Clean baseline ({pc_clean:.3f})')
ax.set_xticks(x)
ax.set_xticklabels([c.replace("_", "\n") for c in CTYPES])
ax.set_ylabel("Image-Level AUROC")
ax.set_title("PatchCore — Preprocessing Recovery at Severe Corruption")
ax.legend()
ax.set_ylim([0.4, 1.05])
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("/kaggle/working/figures/preprocessing_recovery_severe.png", dpi=200)
plt.show()

# %% [markdown]
# ## Cell 4: Figure 3 — Category Heatmap

# %%
for model_name in ["PatchCore"]:
    model_deg = degradation[degradation["model"] == model_name]

    # Pivot: categories × corruption_type (at severe)
    severe = model_deg[model_deg["severity"] == "severe"]
    pivot = severe.groupby(["category", "corruption_type"])["image_AUROC"].mean().unstack()

    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.5, vmax=1.0,
                ax=ax, linewidths=0.5)
    ax.set_title(f"{model_name} — AUROC by Category × Corruption (Severe)")
    ax.set_xlabel("Corruption Type")
    ax.set_ylabel("Category")
    plt.tight_layout()
    plt.savefig(f"/kaggle/working/figures/category_heatmap_{model_name.lower()}.png", dpi=200)
    plt.show()

# %% [markdown]
# ## Cell 5: Figure 4 — Training Corruption Comparison

# %%
fig, ax = plt.subplots(figsize=(8, 5))

# Add Condition A (clean baseline) from degradation data
cond_a = degradation[(degradation["corruption_type"] == "clean")]

for model_name in ["PatchCore", "PaDiM"]:
    clean = cond_a[cond_a["model"] == model_name]["image_AUROC"].mean()
    cond_b = train_corr[(train_corr["model"] == model_name) &
                        (train_corr["condition"] == "B")]["image_AUROC"].mean()
    cond_c = train_corr[(train_corr["model"] == model_name) &
                        (train_corr["condition"] == "C")]["image_AUROC"].mean()

    x_pos = [0, 1, 2] if model_name == "PatchCore" else [0.3, 1.3, 2.3]
    color = "#3498DB" if model_name == "PatchCore" else "#E74C3C"
    ax.bar(x_pos, [clean, cond_b, cond_c], 0.25, label=model_name, color=color, alpha=0.8)

ax.set_xticks([0.15, 1.15, 2.15])
ax.set_xticklabels(["A: Clean→Clean", "B: Corrupt→Corrupt", "C: Corrupt+CLAHE\n→Corrupt+CLAHE"])
ax.set_ylabel("Image-Level AUROC")
ax.set_title("Training-Set Corruption Experiment (Low-light Moderate)")
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("/kaggle/working/figures/training_corruption.png", dpi=200)
plt.show()

# %% [markdown]
# ## Cell 6: Recommendation Matrix (KEY DELIVERABLE)

# %%
print("\n" + "="*90)
print("RECOMMENDATION MATRIX — Preprocessing for Industrial Anomaly Detection")
print("="*90)

recs = []
for ctype in CTYPES:
    best_name = PREPROCESS_NAMES[ctype]

    # Degraded AUROC (severe, no preprocessing)
    deg_auroc = degradation[(degradation["model"] == "PatchCore") &
                            (degradation["corruption_type"] == ctype) &
                            (degradation["severity"] == "severe")]["image_AUROC"].mean()

    # Rescued AUROC (severe, best preprocessing)
    best = preprocessing[(preprocessing["corruption_type"] == ctype) &
                         (preprocessing["severity"] == "severe") &
                         (preprocessing["preprocessing"] == best_name)]
    rescued_auroc = best["image_AUROC"].mean() if len(best) > 0 else np.nan
    recovery = rescued_auroc - deg_auroc if not np.isnan(rescued_auroc) else np.nan

    recs.append({
        "Degradation": ctype.replace("_", " ").title(),
        "Best Preprocessing": best_name,
        "AUROC (degraded)": f"{deg_auroc:.4f}",
        "AUROC (rescued)": f"{rescued_auroc:.4f}" if not np.isnan(rescued_auroc) else "N/A",
        "Recovery": f"+{recovery:.4f}" if not np.isnan(recovery) else "N/A",
    })

rec_df = pd.DataFrame(recs)
print(rec_df.to_string(index=False))
rec_df.to_csv("/kaggle/working/figures/recommendation_matrix.csv", index=False)

# %% [markdown]
# ## Cell 7: Pipeline Ordering Ablation

# %%
combined = preprocessing[preprocessing["corruption_type"] == "combined"]
if len(combined) > 0:
    ablation = combined.groupby("preprocessing")["image_AUROC"].agg(["mean", "std"]).round(4)
    print("\nPipeline Ordering Ablation (Combined: Low-light + Noise)")
    print("="*60)
    print(ablation.to_string())

    winner = ablation["mean"].idxmax()
    print(f"\n✅ Best ordering: {winner} (AUROC={ablation.loc[winner, 'mean']:.4f})")
else:
    print("⚠️ No combined preprocessing results found")

# %% [markdown]
# ## Cell 8: Most/Least Affected Categories

# %%
print("\n" + "="*60)
print("CATEGORY-LEVEL ANALYSIS")
print("="*60)

for ctype in CTYPES:
    severe = degradation[(degradation["model"] == "PatchCore") &
                         (degradation["corruption_type"] == ctype) &
                         (degradation["severity"] == "severe")]
    cat_means = severe.groupby("category")["image_AUROC"].mean().sort_values()

    print(f"\n{ctype.replace('_', ' ').title()} (severe):")
    print(f"  Most affected:  {cat_means.index[0]} (AUROC={cat_means.iloc[0]:.4f})")
    print(f"  Least affected: {cat_means.index[-1]} (AUROC={cat_means.iloc[-1]:.4f})")

# %% [markdown]
# ## Cell 9: Save All Figures List

# %%
print("\n📁 Generated figures:")
for f in sorted(os.listdir("/kaggle/working/figures")):
    size = os.path.getsize(f"/kaggle/working/figures/{f}")
    print(f"  {f} ({size/1024:.0f} KB)")

print("\n✅ Analysis complete!")
print("📋 Key deliverables in /kaggle/working/figures/:")
print("   - degradation_curves_patchcore.png")
print("   - degradation_curves_padim.png")
print("   - preprocessing_recovery_severe.png")
print("   - category_heatmap_patchcore.png")
print("   - training_corruption.png")
print("   - recommendation_matrix.csv")
