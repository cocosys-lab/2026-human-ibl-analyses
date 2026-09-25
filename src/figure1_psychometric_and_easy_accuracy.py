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
from scipy.stats import t, ttest_ind

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
    title="",#Psychometric curves
    show_ylabel=True,
    show_legend=True,
    add_subj_number=False,
    scatter=False,
)
grouped_df = data.groupby(['subject','signed_contrast', 'instructions']).agg({'choice_right':'mean'}).reset_index()
grouped_df['choice_right'] = grouped_df['choice_right']*100
sns.lineplot(data=grouped_df[grouped_df.instructions==1], x='signed_contrast', y='choice_right',
                ax=axes[0], color=COLORS['instructions'], legend=False, errorbar='ci',
                err_style='bars', linewidth=0, marker='o', markersize=5)
sns.lineplot(data=grouped_df[grouped_df.instructions==0], x='signed_contrast', y='choice_right',
                ax=axes[0], color=COLORS['no_instructions'], legend=False, errorbar='ci',
                err_style='bars', linewidth=0, marker='o', markersize=5)

axes[0].set_ylabel("P(right) (%)")

rng = np.random.default_rng(42)

data_high_contrast = data[np.abs(data.signed_contrast)>=50]
data_high_contrast['correct'] = (data['feedbackType']==1).astype(int)
data_high = data_high_contrast.groupby(['subject','instructions'])['correct'].mean().reset_index()
data_high['correct'] = data_high['correct']*100
sns.stripplot(data=data_high, x='instructions', y='correct',
              hue='instructions', ax=axes[1], jitter=True, edgecolor='white', linewidth=0.5,
              palette=[COLORS['no_instructions'],COLORS['instructions']],
              zorder=1)
sns.boxplot(data=data_high, x='instructions',y='correct',
            hue='instructions',ax=axes[1], saturation=0.5,
            palette=[COLORS['no_instructions'],COLORS['instructions']],
            zorder=0)
ttest_result = ttest_ind(data_high.loc[data_high.instructions==0,'correct'], data_high.loc[data_high.instructions==1,'correct'],
                         alternative='two-sided')
annotation = '***' if ttest_result.pvalue < 0.001 else ('**' if ttest_result.pvalue < 0.01 else ('*'if ttest_result.pvalue < 0.05 else 'n.s.') )
axes[1].text(0.5, 1.05, annotation, transform=axes[1].transAxes, ha='center', va='top')
axes[1].plot([0.25, 0.75], [1, 1], transform=axes[1].transAxes, color='k', lw=1)
# axes[1].legend(['Not instructed','Instructed'],title='')
axes[1].get_legend().remove()
axes[1].set_ylabel('Accuracy on easy trials (%)')
axes[1].set_xlabel('Instruction group')
axes[1].set_xticklabels(['Not instructed','Instructed'])

# sns.despine(fig=fig)

# axes[1].set_xticks(range(len(GROUP_ORDER)), [GROUP_LABELS[g] for g in GROUP_ORDER])
# axes[1].set_xlabel("")
# axes[1].set_ylabel("Accuracy on easy trials (%)")
# axes[1].set_ylim(0, 105)
# axes[1].set_title("Easy-trial accuracy")

sns.despine(fig=fig)
fig.tight_layout()
fig.subplots_adjust(wspace=0.45)

for extension in ("png", "svg"):
    fig.savefig(
        FIGURE_PATH / f"figure1_psychometric_and_easy_accuracy.{extension}",
        dpi=300,
        bbox_inches="tight",
    )

# plt.close(fig)

print("Subjects in psychometric panel:", data["subject"].nunique())
print(
    "Subjects with easy-trial accuracy by group:",
    subject_accuracy.groupby("instructions")["subject"].nunique().to_dict(),
)
#%%