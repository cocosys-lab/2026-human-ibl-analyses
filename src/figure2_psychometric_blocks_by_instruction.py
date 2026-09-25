"""
fenying, September 2026
Plot block-conditioned psychometric curves by instruction group.

3 Panels:
1. Participants with instructions: psychometric curves for Left/Right blocks.
2. Participants without instructions: psychometric curves for Left/Right blocks.
3. Subject-level change in P(right) at 0% contrast between Right/Left blocks.
"""
#%%
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psychofit as psy
import seaborn as sns
from scipy.stats import ttest_rel

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from utils.plot_config import COLORS, apply_style
import utils.form_psychometrics as P
import utils.transform_data as TD

apply_style()

DATA_PATH = REPO_ROOT / "data" / "processed_data.csv"
FIGURE_PATH = REPO_ROOT / "figures"

ADD_INSETS = False
# raw_zero: observed P(right) from actual 0%-contrast trials.
# fitted_zero: P(right) predicted at 0% by each subject/block curve.
DELTA_METHOD = "raw_zero"
MIN_UNIQUE_CONTRASTS_FOR_FIT = 4


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
    (1, "Instructed"),
    (0, "Not Instructed"),
]

instruction_order = [1, 0]
instruction_labels = ["Instructed", "Not Instructed"]
instruction_colors = [
    COLORS["instructions"],
    COLORS["no_instructions"],
]


# %% Load and preprocess data
data = pd.read_csv(DATA_PATH)
data = TD.preprocess_trials(data)
data = data[data["choice"].isin([-1, 1])].copy() #TODO: check and discuss
data = data[data["block"].isin(["Left", "Right"])].copy()

#%% fit psychometrics per subject and block
subjects = data['subject'].unique()
psychometric_params = []
blocks = ['Left', 'Right']
for subject in subjects:
    subdat = data[data['subject'] == subject]
    pars, xx, yy = P.fit_psychometric(subdat)
    sub_dict = {
        'subject': subject,
        'instructions': subdat['instructions'].iloc[0],
        'bias': pars[0],
        'slope': pars[1],
        'lapse_low': pars[2],
        'lapse_high': pars[3]
    }
    for block in blocks:
        block_df = data[(data['subject'] == subject) & (data['block'] == block)]
        if len(block_df) > 0:
            pars, xx, yy = P.fit_psychometric(block_df)
            sub_dict.update({
                f'bias_{block}': pars[0],
                f'slope_{block}': pars[1],
                f'lapse_low_{block}': pars[2],
                f'lapse_high_{block}': pars[3]
            })
    psychometric_params.append(sub_dict)
       
#%% merge the fits
psychometric_df = pd.DataFrame(psychometric_params)

#%% get relevant quantities for plotting
psychometric_df['Absolute bias'] = np.abs(psychometric_df['bias'])
psychometric_df['Absolute_bias_Right'] = np.abs(psychometric_df['bias_Right'])
psychometric_df['Absolute_bias_Left'] = np.abs(psychometric_df['bias_Left'])
psychometric_df['Mean lapse'] = (psychometric_df['lapse_low'] + psychometric_df['lapse_high']) / 2
psychometric_df['Mean_lapse_Right'] = (psychometric_df['lapse_low_Right'] + psychometric_df['lapse_high_Right']) / 2
psychometric_df['Mean_lapse_Left'] = (psychometric_df['lapse_low_Left'] + psychometric_df['lapse_high_Left']) / 2
psychometric_df[r'$\Delta$ Absolute bias'] = psychometric_df['Absolute_bias_Right'] - psychometric_df['Absolute_bias_Left']
psychometric_df[r'$\Delta$ Mean lapse'] = psychometric_df['Mean_lapse_Right'] - psychometric_df['Mean_lapse_Left']
psychometric_df[r'$\Delta$ Slope'] = psychometric_df['slope_Right'] - psychometric_df['slope_Left']
psychometric_df['Slope'] = psychometric_df['slope']

#and transform to longform for plotting
longform = pd.melt(psychometric_df, 
                   id_vars=['subject', 'instructions'],
                   value_vars=['Absolute_bias_Left', 'Absolute_bias_Right', 
                               'Mean_lapse_Left', 'Mean_lapse_Right',
                               'lapse_low_Left', 'lapse_low_Right',
                               'lapse_high_Left', 'lapse_high_Right',
                                'slope_Left', 'slope_Right'], #'bias_Left', 'bias_Right',  
                   var_name='parameter', value_name='value')
#rename the parameter column such that we have 'parameter name' and 'block' columns
longform['block'] = longform['parameter'].apply(lambda x: x.split('_')[-1])
longform['parameter'] = longform['parameter'].apply(lambda x: '_'.join(x.split('_')[:-1]))

#%% compute what goes in the third panel before plotting
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
if DELTA_METHOD == "fitted_zero":
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

# %% Plot psychometric curves and the subject-level block effect
fig, axes = plt.subplots(1, 3, figsize=(12,4),gridspec_kw={"width_ratios": [1, 1, 0.8]})#

for ax, (instruction_value, title) in zip(axes[:2], instruction_panels):
    panel_data = data[data["instructions"] == instruction_value]
    title_color = instruction_colors[instruction_order.index(instruction_value)]
    if ADD_INSETS:
        P.plot_panel_with_inset(
            ax,
            panel_data,
            group_col="block",
            group_values=block_values,
            palette=block_palette,
            labels=block_labels,
            title=title,
            scatter=False,
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
            title_color=title_color,
            show_ylabel=False,
            show_legend=False,
            add_subj_number=False,
            scatter=False,
        )
    left_group = panel_data[panel_data["block"] == "Left"].groupby(["subject","signed_contrast"])["choice_right"].mean().reset_index()
    right_group = panel_data[panel_data["block"] == "Right"].groupby(["subject","signed_contrast"])["choice_right"].mean().reset_index()
    left_group['choice_right'] = left_group['choice_right']*100
    right_group['choice_right'] = right_group['choice_right']*100
    sns.lineplot(data=left_group, x='signed_contrast', y='choice_right',
                 ax=ax, color=block_palette['Left'], legend=False, errorbar='ci',
                 linewidth=0, err_style='bars', marker='o', markersize=5)
    sns.lineplot(data=right_group, x='signed_contrast', y='choice_right',
                 ax=ax, color=block_palette['Right'], legend=False, errorbar='ci',
                 linewidth=0, err_style='bars', marker='o', markersize=5)


axes[0].set_ylabel("P(right) (%)")
axes[1].set_ylabel("")
axes[0].legend(frameon=False, loc="upper left")


rng = np.random.default_rng(42)
sns.stripplot(
    data=delta_data, x='instructions', y='delta_right', hue='instructions',
    ax=axes[2], palette=instruction_colors, hue_order=instruction_order,
    jitter=True, edgecolor='lightgrey', linewidth=0.5, size=6,
    zorder=1, dodge=False)

sns.boxplot(
    data=delta_data,
    x="instructions",
    y="delta_right",legend=False,
    hue="instructions", hue_order=instruction_order,
    palette=instruction_colors, zorder=0,
    ax=axes[2], saturation=0.7, dodge=False,
)
axes[2].legend(['Not instructed', 'Instructed'], frameon=False, loc='upper left', bbox_to_anchor=(1, 0.7))
axes[2].axhline(0, color="0.5", linewidth=1, linestyle="--")
axes[2].set_xticks([0, 1], ['Not instructed', 'Instructed'])
axes[2].set_ylabel(
    f"{delta_method_label} ΔP(right) at 0% contrast\n"
    #"(Right block - Left block)"
)
axes[2].set_title("Block-induced choice bias")
axes[2].set_xlabel('Instruction group')

sns.despine(fig=fig)
fig.tight_layout()
for extension in ["png", "svg"]:
    if ADD_INSETS:
        fig.savefig(
            FIGURE_PATH / f"figure2_psychometric_blocks_by_instruction_{DELTA_METHOD}_insets.{extension}",
            dpi=300,
            bbox_inches="tight",)
    else:
        fig.savefig(
            FIGURE_PATH / f"figure2_psychometric_blocks_by_instruction_{DELTA_METHOD}.{extension}",
            dpi=300,
            bbox_inches="tight",)
        
#%% and then parameter comparison at the bottom
fig, ax = plt.subplots(1,4, figsize=(12,3)) #(20,5)
param_names = ['Absolute_bias', 'lapse_low', 'lapse_high', 'slope']
param_labels = ['Absolute bias', 'Lapse low', 'Lapse high', 'Slope']
for i,param in enumerate(param_names):
    g=sns.stripplot(x='instructions', y='value', hue='block', data=longform[longform['parameter']==param],
                  ax=ax[i], palette=[COLORS['left_block'], COLORS['right_block']],
                  hue_order = ['Left', 'Right'],legend=False, jitter=True, edgecolor='lightgrey', linewidth=0.5, size=6,
                  zorder=1, dodge=True)
    all_x_values = [path.get_offsets()[:, 0] for path in g.collections]
    all_y_values = [path.get_offsets()[:, 1] for path in g.collections]
    for ix in range(len(all_y_values)//2):
        ax[i].plot(all_x_values[2*ix:2*ix+2],all_y_values[2*ix:2*ix+2],
                linewidth=.5,color='grey',alpha=0.5)
    sns.boxplot(x='instructions', y='value', hue='block', data=longform[longform['parameter']==param],
                ax=ax[i], palette=[COLORS['left_block'], COLORS['right_block']],
                hue_order = ['Left', 'Right'],legend=False, saturation=0.7, zorder=0)
    ax[i].set_ylabel(param_labels[i])
    ax[i].set_xticklabels(['Not instructed', 'Instructed'])
    #then, for each instruction group, perform a t-test between left and right blocks
    for j, instr in enumerate([1,0]):
        left_values = longform[(longform['parameter']==param) & (longform['instructions']==instr) & (longform['block']=='Left')]['value']
        right_values = longform[(longform['parameter']==param) & (longform['instructions']==instr) & (longform['block']=='Right')]['value']
        test_res = ttest_rel(left_values, right_values, alternative='two-sided')
        annotation = '***' if test_res.pvalue < 0.001 else ('**' if test_res.pvalue < 0.01 else ('*'if test_res.pvalue < 0.05 else 'n.s.') )
        ax[i].text(j*0.5+.25, 0.95, annotation, transform=ax[i].transAxes, ha='center', va='top')
        ax[i].plot([j*0.5+0.15, j*0.5+0.35], [0.9, 0.9], transform=ax[i].transAxes, color='k', lw=1)
        ax[i].set_xlabel('')
ax[3].legend(['Left blocks', 'Right blocks'], loc='upper left', frameon=False, bbox_to_anchor=(1, 0.7))
sns.despine(fig=fig)
fig.tight_layout()
for extension in ["png", "svg"]:
    fig.savefig(
        FIGURE_PATH / f"figure2_psychometric_parameter_comparison.{extension}",
        dpi=300,
        bbox_inches="tight",)

# %%
