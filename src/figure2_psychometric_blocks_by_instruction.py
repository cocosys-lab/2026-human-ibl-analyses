"""
fenying, September 2026
Plot block-conditioned psychometric curves by instruction group.

3 Panels:
1. Participants with instructions: psychometric curves for Left/Right blocks.
2. Participants without instructions: psychometric curves for Left/Right blocks.
3. Subject-level change in P(right) at 0% contrast between Right/Left blocks.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psychofit as psy
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from utils.plot_config import COLORS, apply_style
import utils.form_psychometrics as P
import utils.transform_data as TD

apply_style()

DATA_PATH = REPO_ROOT.parent / "2026-human-ibl-analyses" / "data" / "processed_data.csv"
FIGURE_PATH = REPO_ROOT / "figures"

ADD_INSETS = False
# raw_zero: observed P(right) from actual 0%-contrast trials.
# fitted_zero: P(right) predicted at 0% by each subject/block curve.
DELTA_METHOD = "raw_zero"
MIN_UNIQUE_CONTRASTS_FOR_FIT = 4

# %% Load and preprocess data 
data = pd.read_csv(DATA_PATH)
data = TD.preprocess_trials(data)
data = data[data["block"].isin(["Left", "Right"])].copy()

# %% Plot psychometric curves and the subject-level block effect
fig, axes = plt.subplots(1, 3, figsize=(9, 4),
    gridspec_kw={"width_ratios": [1, 1, 0.8]})

block_values = ["Left", "Right"]
block_palette = {
    "Left": COLORS["left_block"],
    "Right": COLORS["right_block"],
}
block_labels = {
    "Left": "Left blocks",
    "Right": "Right blocks",
}

instruction_panels = [
    (1, "Instructions"),
    (0, "No instructions"),
]

for ax, (instruction_value, title) in zip(axes[:2], instruction_panels):
    panel_data = data[data["instructions"] == instruction_value]

    if ADD_INSETS:
        P.plot_panel_with_inset(
            ax,
            panel_data,
            group_col="block",
            group_values=block_values,
            palette=block_palette,
            labels=block_labels,
            title=title,
            scatter=True,
        )
    else:
        P.plot_two_curves_on_ax(
            ax,
            panel_data,
            group_col="block",
            group_values=block_values,
            palette=block_palette,
            labels=block_labels,
            title=title,
            show_ylabel=False,
            show_legend=False,
            add_subj_number=False,
            scatter=True,
        )

axes[0].set_ylabel("P(right) (%)")
axes[0].legend(frameon=False, loc="upper left")

# %% option 1: observed subject-level delta P(right) at 0% contrast
zero_contrast = data[np.isclose(data["signed_contrast"], 0)].copy()

raw_block_choice = (
    zero_contrast.groupby(["subject", "instructions", "block"])
    .agg(p_right=("choice_right", "mean"))
    .reset_index()
)

raw_block_choice = raw_block_choice.pivot_table(
    index=["subject", "instructions"],
    columns="block",
    values="p_right",
).reindex(columns=block_values)

raw_paired = raw_block_choice.dropna(subset=block_values).copy()
raw_paired["delta_right"] = (
    raw_paired["Right"] - raw_paired["Left"]
) * 100
raw_delta = raw_paired.reset_index()


# %% option 2: fit each curve and predict P(right) at 0%
fitted_records = []
fit_failures = []

for (subject, instructions, block), subject_block_data in data.groupby(
    ["subject", "instructions", "block"],
    sort=False,
):
    if subject_block_data["signed_contrast"].nunique() < MIN_UNIQUE_CONTRASTS_FOR_FIT:
        continue

    try:
        pars, _, _ = P.fit_psychometric(subject_block_data)
        p_right_zero = float(psy.erf_psycho_2gammas(pars, np.array([0.0]))[0] * 100)
        fitted_records.append(
            {
                "subject": subject,
                "instructions": instructions,
                "block": block,
                "p_right": p_right_zero,
            }
        )
    except Exception as error:
        fit_failures.append((subject, instructions, block, str(error)))

fitted_long = pd.DataFrame(
    fitted_records,
    columns=["subject", "instructions", "block", "p_right"],
)

if fitted_long.empty:
    fitted_delta = pd.DataFrame(
        columns=["subject", "instructions", "Left", "Right", "delta_right"]
    )
else:
    fitted_block_choice = fitted_long.pivot_table(
        index=["subject", "instructions"],
        columns="block",
        values="p_right",
    ).reindex(columns=block_values)

    fitted_paired = fitted_block_choice.dropna(subset=block_values).copy()
    fitted_paired["delta_right"] = (
        fitted_paired["Right"] - fitted_paired["Left"]
    )
    fitted_delta = fitted_paired.reset_index()


#%% which one to show in panel 3.
if DELTA_METHOD == "raw_zero":
    delta_data = raw_delta
    delta_method_label = "Observed"
elif DELTA_METHOD == "fitted_zero":
    delta_data = fitted_delta
    delta_method_label = "Fitted"


if delta_data.empty:
    raise ValueError(
        f"No paired subject-level estimates are available for {DELTA_METHOD}. "
        "See the printed coverage table."
    )

instruction_order = [1, 0]
instruction_labels = ["Instructions", "No instructions"]
instruction_colors = [
    COLORS["instructions"],
    COLORS["no_instructions"],
]

rng = np.random.default_rng(42)

for x, (instruction_value, color) in enumerate(
    zip(instruction_order, instruction_colors)
):
    values = delta_data.loc[
        delta_data["instructions"] == instruction_value,
        "delta_right",
    ].dropna()

    if values.empty:
        continue

    axes[2].bar(
        x,
        values.mean(),
        yerr=values.sem(),
        width=0.6,
        color=color,
        alpha=0.55,
        edgecolor=color,
        capsize=4,
    )

    jitter = rng.uniform(-0.12, 0.12, size=len(values))
    axes[2].scatter(
        x + jitter,
        values,
        s=18,
        color=color,
        edgecolor="white",
        linewidth=0.3,
        alpha=0.75,
        zorder=3,
    )

axes[2].axhline(0, color="0.5", linewidth=1, linestyle="--")
axes[2].set_xticks([0, 1], instruction_labels)
axes[2].set_ylabel(
    f"{delta_method_label} ΔP(right) at 0% contrast\n"
    "(Right block - Left block)"
)
axes[2].set_title("Block-induced choice bias")

sns.despine(fig=fig)
fig.tight_layout()

for extension in ["png", "svg"]:
    if ADD_INSETS:
        fig.savefig(
            FIGURE_PATH / f"figure2_psychometric_blocks_by_instruction_{DELTA_METHOD}_insets.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    else:
        fig.savefig(
            FIGURE_PATH / f"figure2_psychometric_blocks_by_instruction_{DELTA_METHOD}.{extension}",
            dpi=300,
            bbox_inches="tight",
        )

plt.close(fig)
#%%