import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from crtical_diagram import draw_cd_diagram
folders = ["Rec-PCA", "AutoEncoder", "Autoformer", "DLinear", "FEDformer", "TimesNet", "Transformer"]
seeds = range(3, 8, 1)

execution_times = {folder: [] for folder in folders}
auc_pr_scores = {folder: [] for folder in folders}
auc_pr_mean_per_dataset = {folder: [] for folder in folders}

# =========================
# Load execution time + AUC-PR
# =========================
for folder in folders:
    all_data = []
    for seed in seeds:
        filename = f"{folder}/seed{seed}_overlapping.csv"
        df = pd.read_csv(filename)
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    mean_df = combined_df.groupby('filename').mean(numeric_only=True).reset_index()
    std_df = combined_df.groupby('filename').std(numeric_only=True).reset_index()

    print(f"Results for {folder}:")
    print(f"Average {mean_df.mean(numeric_only=True)}")
    print(f"Standard Deviation {std_df.mean(numeric_only=True)}")

    execution_times[folder] = mean_df['execution_time_seconds'].tolist()
    auc_pr_scores[folder] = mean_df['AUC-PR'].tolist()
    auc_pr_mean_per_dataset[folder] = mean_df.groupby('filename')['AUC-PR'].mean().tolist()



# =========================
# Plot settings
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300
})

# =========================
# FIXED COLOR MAP (consistent across all plots)
# =========================

# HEX color map (as strings)
hex_colors = [
    "#F94144",
    "#F3722C",
    "#F8961E",
    "#F9C74F",
    "#90BE6D",
    "#43AA8B",
    "#577590",
]

# Create a simple color mapping for models
model_colors = {folder: hex_colors[i % len(hex_colors)] for i, folder in enumerate(folders)}

# =========================
# Sort by execution time (used for first plot only)
# =========================
folders_sorted_time = sorted(
    folders,
    key=lambda f: sum(execution_times[f]) / len(execution_times[f]) if execution_times[f] else float('inf')
)

# =========================
# Plot 1: Execution Time
# =========================
fig, ax = plt.subplots(figsize=(8, 3.5))

bp = ax.boxplot(
    [execution_times[f] for f in folders_sorted_time],
    tick_labels=folders_sorted_time,
    patch_artist=True,
    showfliers=False,
    widths=0.6,
    medianprops=dict(color='black', linewidth=2),
    boxprops=dict(linewidth=1.5),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5)
)

for patch, folder in zip(bp['boxes'], folders_sorted_time):
    patch.set_facecolor(model_colors[folder])
    patch.set_alpha(0.90)


ax.set_yscale('log')
ax.set_ylim(0.01, 1000)

ax.set_ylabel('Execution Time (s)')
ax.set_xlabel('Models')

ax.grid(which='major', axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
ax.yaxis.set_minor_locator(mticker.NullLocator())

plt.xticks(rotation=15)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("execution_time_boxplot.pdf", bbox_inches='tight')

# =========================
# AUC-PR variability metrics
# =========================
auc_pr_std = {folder: [] for folder in folders}
auc_pr_range = {folder: [] for folder in folders}

for folder in folders:
    all_data = []
    for seed in seeds:
        filename = f"{folder}/seed{seed}_overlapping.csv"
        df = pd.read_csv(filename)
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)

    std_df = combined_df.groupby('filename').std(numeric_only=True).reset_index()
    auc_pr_std[folder] = std_df['AUC-PR'].tolist()

    range_df = combined_df.groupby('filename')['AUC-PR'].apply(lambda x: x.max() - x.min()).reset_index()
    range_df.columns = ['filename', 'AUC-PR']
    auc_pr_range[folder] = range_df['AUC-PR'].tolist()

# =========================
# Sort by AUC-PR std (used ONLY for second plots)
# =========================
folders_sorted_std = sorted(folders, key=lambda f: np.median(auc_pr_std[f]))

# =========================
# Plot 2: AUC-PR STD
# =========================
fig, ax1 = plt.subplots(figsize=(8, 3.5))

bp1 = ax1.boxplot(
    [auc_pr_std[f] for f in folders_sorted_std],
    tick_labels=folders_sorted_std,
    patch_artist=True,
    showfliers=False,
    widths=0.6,
    medianprops=dict(color='black', linewidth=2),
    boxprops=dict(linewidth=1.5),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5)
)

for patch, folder in zip(bp1['boxes'], folders_sorted_std):
    patch.set_facecolor(model_colors[folder])
    patch.set_alpha(0.90)

ax1.set_ylabel('Std of AUC-PR')
ax1.set_xlabel('Models')
ax1.grid(True, which="major", ls="--", linewidth=0.5, alpha=0.7)
ax1.tick_params(axis='x', rotation=15)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("auc_pr_std_boxplot.pdf", bbox_inches='tight')
plt.show()

# =========================
# Plot 3: AUC-PR Range
# =========================
fig, ax2 = plt.subplots(figsize=(8, 3.5))

bp2 = ax2.boxplot(
    [auc_pr_range[f] for f in folders_sorted_std],
    tick_labels=folders_sorted_std,
    patch_artist=True,
    showfliers=False,
    widths=0.6,
    medianprops=dict(color='black', linewidth=2),
    boxprops=dict(linewidth=1.5),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5)
)

for patch, folder in zip(bp2['boxes'], folders_sorted_std):
    patch.set_facecolor(model_colors[folder])
    patch.set_alpha(0.90)


ax2.set_ylabel('Range of AUC-PR')
ax2.set_xlabel('Models')
ax2.grid(True, which="major", ls="--", linewidth=0.5, alpha=0.7)
ax2.tick_params(axis='x', rotation=15)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("auc_pr_range_boxplot.pdf", bbox_inches='tight')
plt.show()




def build_df_perf_from_auc_pr(folders, auc_pr_scores, metric_col_name="accuracy"):
    """
    Builds the df_perf DataFrame expected by result/statistical_test/main.py:
      - classifier_name
      - dataset_name
      - accuracy (or another name, but main.py uses 'accuracy' so keep it)
    """
    # Convert to arrays + check lengths
    arrays = {f: np.asarray(auc_pr_scores[f], dtype=float) for f in folders}
    lengths = {f: len(arr) for f, arr in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"All methods must have the same number of paired samples. Got lengths: {lengths}")

    n = next(iter(lengths.values()))
    dataset_names = [f"ds_{i:04d}" for i in range(n)]  # or replace with your real dataset IDs

    rows = []
    for f in folders:
        for ds, val in zip(dataset_names, arrays[f]):
            rows.append(
                {
                    "classifier_name": f,
                    "dataset_name": ds,
                    metric_col_name: float(val),
                }
            )
    return pd.DataFrame(rows)


df_perf = build_df_perf_from_auc_pr(folders, auc_pr_scores, metric_col_name="accuracy")
draw_cd_diagram(df_perf, name="critical_diagram_auc_pr_5_evalset.pdf", alpha=0.05)

# Build df_perf in the format expected by main.py using average results per dataset
df_perf = build_df_perf_from_auc_pr(folders, auc_pr_mean_per_dataset, metric_col_name="accuracy")
draw_cd_diagram(df_perf, name="critical_diagram_auc_pr_5_dataset.pdf", alpha=0.05)