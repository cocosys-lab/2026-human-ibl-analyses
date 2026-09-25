#%% imports

import sys
import ast
sys.path.append("..")
from utils.plot_config import apply_style, COLORS, CONTRAST_PALETTE, EXAMPLE_SESSIONS
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import re

import seaborn as sns
import pandas as pd
import numpy as np
import utils.transform_data as TD
from matplotlib.gridspec import GridSpec
from utils.plotting_funs import plot_median_rt, plot_var_rt, plot_mean_cursor_trajectories, plot_all_session_trajectories
import utils.form_psychometrics as P

apply_style()

#%% import data

data = pd.read_csv('../data/processed_data_2026.csv', converters={'cursorTime': ast.literal_eval, 
                                                             'cursorPosition': ast.literal_eval})

data = TD.preprocess_trials(data)
data = TD.transform_contrast_to_ix(data) #add contrast index for plotting

contrast_level = None #1. # or None for all contrasts in wiggle plot

#%% prepare cursor data

# rereference cursor position and time within trial
data['cursorPosition_fromcentre'] = [np.array(row['cursorPosition']) - row['cursorPosition'][0] for i,row in data.iterrows()]
data['cursorTime_fromstim'] = [np.array(row['cursorTime']) - row['cursorTime'][0] for i,row in data.iterrows()]

# make some extra time columns for plotting
refresh_rate = 75
max_rt = data['rt'].max()
data['timeToResp'] = [np.arange(0, max_rt+0.1, 1/refresh_rate) for row in data.iterrows()] # fill in the total time to response from stim (slightly longer than necessary to make sure all samples have timestamps)
data['timeFromResp'] = [np.arange(-max_rt-0.1, 0, 1/refresh_rate) for row in data.iterrows()] # same but with response at time 0s and stim at time -10s

# make cursor column equal lengths for averaging
def _traj_fill_in_nans(trajectory, max_rt, refresh_rate, direction='from_stim'):
    x_time = np.arange(0, max_rt+0.1, 1/refresh_rate)
    traj_nans = np.ones_like(x_time) * np.nan
    if direction=='from_stim':
        traj_nans[0:len(trajectory)] = trajectory
    elif direction=='from_resp':
        traj_nans[-len(trajectory):] = trajectory
    else:
        raise(ValueError)
    return traj_nans

data['cursorPositionNan'] = [_traj_fill_in_nans(row[1]['cursorPosition_fromcentre'], max_rt, refresh_rate, direction='from_stim') for row in data.iterrows()]
data['nanCursorPosition'] = [_traj_fill_in_nans(row[1]['cursorPosition_fromcentre'], max_rt, refresh_rate, direction='from_resp') for row in data.iterrows()]

#%% select example sessions

ex_session_ins = data[data['subject']==EXAMPLE_SESSIONS['instructions']]
ex_session_no = data[data['subject']==EXAMPLE_SESSIONS['no_instructions']]

ex_data_dict = {}
ex_data_dict['Instructed'] = {'data':ex_session_ins, 'col':COLORS['instructions']}
ex_data_dict['Not instructed'] = {'data':ex_session_no, 'col':COLORS['no_instructions']}

#%% create figure

fig_layout = '''
            AA
            BB
            CD
            EF
            GH
            '''

fig, ax = plt.subplot_mosaic(fig_layout, figsize=(10,12), height_ratios=[0.125, 0.125, 0.25, 0.25, 0.25])

### EXAMPLE SESSIONS - ROLLING RT
ylims = (0.2, data['rt'].max()+0.1)
for key, a in zip(ex_data_dict, [ax['A'], ax['B']]):
    d = ex_data_dict[key]['data']

    # rt scatterplot
    sns.scatterplot(data=d, ax=a, 
        x='trial', y='rt', 
        style='feedbackType', hue='feedbackType',
        palette={1.:COLORS['correct'], -1.:COLORS['incorrect']}, 
        markers={1.:'o', -1.:'X'}, s=10, edgecolors='face',
        alpha=.5, legend=True)

    # running median overlaid
    sns.lineplot(
        data=d[['trial', 'rt']].rolling(10).median(), 
        ax=a,
        x='trial', y='rt', color='black', errorbar=None, label='rolling median RT')

    a.set(xlabel="Trial number", ylabel="Response time (s)", ylim=[0.01, 10])
    a.set_yscale("log")
    a.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y,pos:
        ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    if key == 'Instructed':
        key2 = 'instructions'
    else:
        key2 = 'no_instructions'
    a.set_title(f'Participant {EXAMPLE_SESSIONS[key2]}: {key}', color=ex_data_dict[key]['col'])
    a.set_ylim(ylims)
handles, labels = ax['A'].get_legend_handles_labels()
# legend_a = 
legend_a = ax['B'].legend(handles, ['Incorrect', 'Correct', 'rolling median RT'], 
                          bbox_to_anchor=[1, 1], loc='center left', frameon=False)
ax['A'].get_legend().remove()

### MOUSE WIGGLES
# whole session of cursor movements
for key, a in zip(ex_data_dict, [ax['C'], ax['D']]):
    plot_all_session_trajectories(ex_data_dict[key]['data'], (COLORS['cursor_early_trial'], COLORS['cursor_late_trial']), ax=a, contrast_level=contrast_level)
    if key == 'Instructed':
        key2 = 'instructions'
    else:
        key2 = 'no_instructions'
    a.set_title(f'Participant {EXAMPLE_SESSIONS[key2]}: {key}', color=ex_data_dict[key]['col'])
    if contrast_level:
        a.set_title(f'Participant {EXAMPLE_SESSIONS[key2]}: {key}\nEasy trials', color=ex_data_dict[key]['col'])
    else:
        a.set_title(f'Participant {EXAMPLE_SESSIONS[key2]}: {key}', color=ex_data_dict[key]['col'])
ax['C'].get_legend().remove()
ax['D'].legend(loc='center left', bbox_to_anchor=[1.02, 0.5], frameon=False)

# from stimulus
data_instructions = data[data['instructions']==1]
data_no_instructions = data[data['instructions']==0]

plot_mean_cursor_trajectories(data_instructions, 'cursorPositionNan', 'timeToResp', ax['E'], CONTRAST_PALETTE, (0,2), 'Time from stimulus onset (s)')
plot_mean_cursor_trajectories(data_no_instructions, 'cursorPositionNan', 'timeToResp', ax['F'], CONTRAST_PALETTE, (0,2), 'Time from stimulus onset (s)')
ax['E'].get_legend().remove()
ax['F'].legend(loc='center left', bbox_to_anchor=[1.02, 0.5], frameon=False, title='Contrast (%)')
ax['E'].set_title(f'All instructed participants', color=COLORS['instructions'])
ax['F'].set_title(f'All not instructed participants', color=COLORS['no_instructions'])

# # from response
# plot_mean_cursor_trajectories(data_instructions, 'nanCursorPosition', 'timeFromResp', ax['I'], COLORS['instructions'], (-0.5,0), 'Time from response (s)')
# plot_mean_cursor_trajectories(data_no_instructions, 'nanCursorPosition', 'timeFromResp', ax['J'], COLORS['no_instructions'], (-0.5,0), 'Time from response (s)')
# ax['I'].get_legend().remove()
# ax['J'].get_legend().remove()

### MEDIAN RT AND RT VARIANCE
# check functions in utils
plot_median_rt(data[data['instructions']==1], ax['G'], label='Instructed', color=COLORS['instructions'])
plot_median_rt(data[data['instructions']==0], ax['G'], label='Not instructed', color=COLORS['no_instructions'])
plot_var_rt(data[data['instructions']==1], ax['H'], label='Instructed', color=COLORS['instructions'])
plot_var_rt(data[data['instructions']==0], ax['H'], label='Not instructed', color=COLORS['no_instructions'])
ax['G'].get_legend().remove()
ax['H'].legend(loc='center left', bbox_to_anchor=[1.02, 0.5], frameon=False)
fig.tight_layout()

# make only the E/F row narrower to leave space for the legend on the right
# _e_pos = ax['E'].get_position()
# _f_pos = ax['F'].get_position()
# _gap = _f_pos.x0 - _e_pos.x1
# _shrink = 0.1
# _scale = ((_e_pos.width + _f_pos.width) - _shrink) / (_e_pos.width + _f_pos.width)
# ax['E'].set_position([_e_pos.x0, _e_pos.y0, _e_pos.width * _scale, _e_pos.height])
# ax['F'].set_position([_e_pos.x0 + (_e_pos.width * _scale) + _gap, _f_pos.y0, _f_pos.width * _scale, _f_pos.height])

sns.despine(fig=fig)

if contrast_level:
    fig.savefig(f'../figures/figure3_highcon.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'../figures/figure3_highcon.svg', dpi=300, bbox_inches='tight')
else:
    fig.savefig('../figures/figure3_alltrials.png', dpi=300, bbox_inches='tight')
    fig.savefig('../figures/figure3_alltrials.svg', dpi=300, bbox_inches='tight')

# %%
