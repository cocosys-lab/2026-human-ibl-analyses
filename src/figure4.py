import sys
sys.path.append("..")
from utils.plot_config import apply_style, COLORS
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import utils.transform_data as TD
from matplotlib.gridspec import GridSpec
from utils.plotting_funs import plot_learning_curve, plot_median_rt, plot_var_rt
import utils.form_psychometrics as P

apply_style()

#%% import data
data = pd.read_csv("../../hivemind2025/data/processed_data.csv")
data = TD.preprocess_trials(data)
all_trials = pd.read_csv('../../hivemind2025/data/human_trials.csv')
data['rt'] = all_trials['response_times_from_stim']

#%% create figure with proper dimensions
windows = [[0,200], [200,400], [400,600]]
fig = plt.figure(figsize=(10, 12))

# Outer grid: just splits the figure into a top region and a bottom region
outer = GridSpec(2, 1, figure=fig, height_ratios=[1.4, 3], hspace=0.30)

# Top region gets its own 1x2 subgrid, independent column count
top_gs = outer[0].subgridspec(1, 2, wspace=0.3)
axes_top = [fig.add_subplot(top_gs[0, i]) for i in range(2)]

# Bottom region gets its own 3x3 subgrid
bottom_gs = outer[1].subgridspec(3, 3, hspace=0.45, wspace=0.4)
axes_grid = [fig.add_subplot(bottom_gs[r, c]) for r in range(3) for c in range(3)]


# on to the plotting
#top row is learning curve and evolution of normalized rt
data['rt_normalized'] = data.groupby('subject')['rt'].transform(lambda x: (x - x.min()) / (x.max()-x.min()))
data = TD.transform_contrast_to_ix(data) #add contrast index for plotting
plot_learning_curve(data[data.instructions==1], ax=axes_top[0], 
                    color=COLORS['instructions'], 
                    label='Instructed', errorbar='ci',
                    x='trial', y='correct',
                    only_easy=False, smooth=True, smooth_window=50)
plot_learning_curve(data[data.instructions==0], ax=axes_top[0],
                    color=COLORS['no_instructions'], 
                    label='Not Instructed', errorbar='ci',
                    x='trial', y='correct',
                    only_easy=False, smooth=True, smooth_window=50)

plot_learning_curve(data[data.instructions==1], ax=axes_top[1],
                    color=COLORS['instructions'],
                    label='Instructed', errorbar='ci',
                    x='trial', y='rt',
                    only_easy=False, smooth=True, smooth_window=50)
plot_learning_curve(data[data.instructions==0], ax=axes_top[1],
                    color=COLORS['no_instructions'],
                    label='Not Instructed', errorbar='ci',
                    x='trial', y='rt',
                    only_easy=False, smooth=True, smooth_window=50)
axes_top[1].set_ylabel('Response time (s)')
for w in windows:
    trials_df = data[(data.trial>=w[0]) & (data.trial<w[1])]
    P.plot_panel_with_inset(
        axes_grid[windows.index(w)], trials_df,
        group_col="instructions", group_values=[1, 0],
        palette={1: "mediumturquoise", 0: "palevioletred"},
        labels={1: "Instructed", 0: "Not Instructed"},
        title=f"Trials {w[0]}-{w[1]}",scatter=False
    )
    grouped_df = trials_df.groupby(['subject','signed_contrast', 'instructions']).agg({'choice_right':'mean'}).reset_index()
    grouped_df['choice_right'] = grouped_df['choice_right']*100
    sns.lineplot(data=grouped_df[grouped_df.instructions==1], x='signed_contrast', y='choice_right',
                 ax=axes_grid[windows.index(w)], color='mediumturquoise', label='Instructed', errorbar='ci',
                 err_style='bars', linewidth=0, marker='o', markersize=5)
    sns.lineplot(data=grouped_df[grouped_df.instructions==0], x='signed_contrast', y='choice_right',
                 ax=axes_grid[windows.index(w)], color='palevioletred', label='Non-Instructed', errorbar='ci',
                 err_style='bars', linewidth=0, marker='o', markersize=5)
    axes_grid[windows.index(w)].set_ylabel("P(right) (%)")
    axes_grid[windows.index(w)].legend()
    
    plot_median_rt(trials_df[trials_df.instructions==1],ax=axes_grid[3+windows.index(w)], 
                   color='mediumturquoise', label='Instructed', errorbar='ci')
    plot_median_rt(trials_df[trials_df.instructions==0],ax=axes_grid[3+windows.index(w)], 
                   color='palevioletred', label='Not Instructed', errorbar='ci')
    axes_grid[3+windows.index(w)].set_ylabel("Median RT (s)")
    axes_grid[3+windows.index(w)].legend()
    
    plot_var_rt(trials_df[trials_df.instructions==1],ax=axes_grid[6+windows.index(w)], 
                color='mediumturquoise', label='Instructed', errorbar='ci')
    plot_var_rt(trials_df[trials_df.instructions==0],ax=axes_grid[6+windows.index(w)], 
                color='palevioletred', label='Not Instructed', errorbar='ci')
    axes_grid[6+windows.index(w)].set_ylabel("Variance RT (s^2)")
    axes_grid[6+windows.index(w)].legend()
    
sns.despine(fig=fig)
for x in axes_grid:
    x.get_legend().remove()
axes_top[0].get_legend().remove()
#bottom row is psychometrics, median rt and variance of rt at different time points
fig.savefig('../figures/figure4_all_trials.png', dpi=300, bbox_inches='tight')
fig.savefig('../figures/figure4_all_trials.svg', dpi=300, bbox_inches='tight')
# %%
