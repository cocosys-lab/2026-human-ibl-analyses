"""

Figure 1b,c: psychometric curves and easy-trial accuracy by instruction.

Panel b compares instructed and not-instructed participants.
Panel c shows each participant's accuracy on the easy trials.

"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from utils.plot_config import COLORS, apply_style
import utils.form_psychometrics as P
import utils.transform_data as TD

apply_style()

DATA_PATH = REPO_ROOT / "data" / "processed_data_2026.csv"
FIGURE_PATH = REPO_ROOT / "figures"
FIGURE_PATH.mkdir(parents=True, exist_ok=True)

GROUP_ORDER = [1, 0]
GROUP_LABELS = {1: "Instructed", 0: "Not instructed"}
GROUP_COLORS = {
    1: COLORS["instructions"],
    0: COLORS["no_instructions"],
}

# In processed_data.csv, stimContrast == 1 denotes the highest contrast.
# stimContrast of 0.5 or 1.0.
EASY_STIM_CONTRASTS = (0.5, 1.0)

# %% Apply the same trial selection as Figure 2.
data = pd.read_csv(DATA_PATH, dtype={"subject": str})
data = TD.preprocess_trials(data)
data = data[
    data["instructions"].isin(GROUP_ORDER)
    & data["choice"].isin([-1, 1])
].copy()

if data.empty:
    raise ValueError("No responded trials in either instruction group.")

# %% Subject-level accuracy on easy trials.
is_easy = (np.isclose(data["stimContrast"], EASY_STIM_CONTRASTS[0])| np.isclose(data["stimContrast"], EASY_STIM_CONTRASTS[1]))
easy_trials = data[is_easy & data["feedbackType"].isin([-1, 1])].copy()
easy_trials["correct"] = (easy_trials["feedbackType"] == 1).astype(int)

subject_accuracy = (
    easy_trials.groupby(["subject", "instructions"], as_index=False)
    .agg(accuracy=("correct", "mean"))
)
subject_accuracy["accuracy"] *= 100

if subject_accuracy.empty:
    raise ValueError("No responded easy trials with valid feedback.")

# %% Figure 1b–c.
fig, axes = plt.subplots(
    1, 2, figsize=(6.5, 3.6), gridspec_kw={"width_ratios": [1.2, 0.8]}
)

P.plot_two_curves_on_ax(
    axes[0],
    data,
    group_col="instructions",
    group_values=GROUP_ORDER,
    palette=GROUP_COLORS,
    labels=GROUP_LABELS,
    title="Psychometric curves",
    show_ylabel=True,
    show_legend=True,
    add_subj_number=False,
    scatter=True,
)

axes[0].set_ylabel("P(right) (%)")

rng = np.random.default_rng(42)
for x, group in enumerate(GROUP_ORDER):
    values = subject_accuracy.loc[
        subject_accuracy["instructions"] == group, "accuracy"
    ].dropna()
    if values.empty:
        continue

    n = len(values)
    ci95 = t.ppf(0.975, df=n - 1) * values.sem() if n >= 2 else None
    axes[1].bar(
        x,
        values.mean(),
        yerr=ci95,
        width=0.6,
        color=GROUP_COLORS[group],
        alpha=0.55,
        edgecolor=GROUP_COLORS[group],
        capsize=4,
    )

    jitter = rng.uniform(-0.12, 0.12, size=len(values))
    axes[1].scatter(
        x + jitter,
        values,
        color=GROUP_COLORS[group],
        s=18,
        edgecolor="white",
        linewidth=0.3,
        alpha=0.75,
        zorder=3,
    )

axes[1].set_xticks(range(len(GROUP_ORDER)), [GROUP_LABELS[g] for g in GROUP_ORDER])
axes[1].set_xlabel("")
axes[1].set_ylabel("Accuracy on easy trials (%)")
axes[1].set_ylim(0, 105)
axes[1].set_title("Easy-trial accuracy")

sns.despine(fig=fig)
fig.tight_layout()
fig.subplots_adjust(wspace=0.45)

for extension in ("png", "svg"):
    fig.savefig(
        FIGURE_PATH / f"figure1_psychometric_and_easy_accuracy.{extension}",
        dpi=300,
        bbox_inches="tight",
    )

plt.close(fig)

print("Subjects in psychometric panel:", data["subject"].nunique())
print(
    "Subjects with easy-trial accuracy by group:",
    subject_accuracy.groupby("instructions")["subject"].nunique().to_dict(),
)
#%%