import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

models = ["AutoEncoder", "Autoformer", "DLinear", "FEDformer", "TimesNet", "Transformer"]
file_names = [f"FHP_{name}" for name in models]

auc_pr_scores = {model: [] for model in models}

for model, file_name in zip(models,file_names):
    print(f"Loading AUC-PR scores for {model} from {file_name}.csv")
    df = pd.read_csv(f"{file_name}.csv")
    df = df[df['params_strategy'] == 'overlapping']
    df = df.dropna(subset=['value'])
    auc_pr_scores[model] = df['value'].tolist()
    print(f"Loaded {len(auc_pr_scores[model])} AUC-PR scores for {model}. Sample: {auc_pr_scores[model][:5]}")

# Sort models by median AUC-PR (descending)
model_medians = {m: (np.median(auc_pr_scores[m]) if len(auc_pr_scores[m])>0 else -np.inf) for m in models}


plt.figure(figsize=(10, 6))
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300
})
cmap = plt.get_cmap("tab10", len(models)+1)
model_colors = {model: cmap(i+1) for i, model in enumerate(models)}   
models = sorted(models, key=lambda m: model_medians[m], reverse=True)

plt.figure(figsize=(8, 3.5))

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300
})


bp = plt.boxplot(
    [auc_pr_scores[model] for model in models],
    labels=models,
    patch_artist=True,
    showfliers=False,
    widths=0.6,
    medianprops=dict(color='black', linewidth=2),
    boxprops=dict(linewidth=1.5),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5)
)

# ==========================================
# Apply consistent colors
# ==========================================
for patch, model in zip(bp['boxes'], models):
    patch.set_facecolor(model_colors[model])

plt.ylabel('AUC-PR')
plt.xlabel('Models')

# Optional for TKDE:
# titles are often omitted in final figures
# plt.title('AUC-PR Scores by Model')

plt.grid(
    True,
    which="major",
    axis='y',
    linestyle="--",
    linewidth=0.5,
    alpha=0.7
)

plt.xticks(rotation=15)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(
    'auc_pr_comparison.pdf',
    bbox_inches='tight'
)

# plt.show()
plt.close()


df_robust = pd.read_csv("robust_AutoEncoder.csv")
values = df_robust['value'].dropna().tolist()

count_above_037 = sum(v > 0.37 for v in values)
count_below_037 = sum(v < 0.37 for v in values)
count_equal_037 = sum(v == 0.37 for v in values)
print(f"Count of AUC-PR > 0.37: {count_above_037}")
print(f"Count of AUC-PR < 0.37: {count_below_037}")
print(f"Count of AUC-PR = 0.37: {count_equal_037}")

from matplotlib.ticker import FormatStrFormatter
fig, ax = plt.subplots(figsize=(8, 3.5))

# Histogram
ax.hist(
    values,
    bins=54,
    color='steelblue',
    edgecolor='white',
    linewidth=0.8,
    alpha=0.9,

)

# Reference methods
ax.axvline(
    0.47,
    color='forestgreen',
    linestyle='--',
    linewidth=2,
    label='Best AE'
)

ax.axvline(
    0.37,
    color='firebrick',
    linestyle='-.',
    linewidth=2,
    label='Sub-PCA'
)

# Labels and limits
ax.set_xlabel('AUC-PR Score', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_xlim(0.3, 0.5)

# Round x-axis labels to 10^-2
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# Clean TKDE-style formatting
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# Legend
ax.legend(frameon=False)

plt.tight_layout()
plt.savefig('auc_pr_distribution_ae.pdf', dpi=300, bbox_inches='tight')
plt.close()