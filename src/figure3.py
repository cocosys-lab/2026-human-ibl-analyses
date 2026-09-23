#%% imports

import sys
import ast
sys.path.append("..")
from utils.plot_config import apply_style, COLORS, EXAMPLE_SESSIONS
import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
import pandas as pd
import numpy as np
import utils.transform_data as TD
from matplotlib.gridspec import GridSpec
from utils.plotting_funs import plot_median_rt, plot_var_rt, plot_mean_cursor_trajectories
import utils.form_psychometrics as P

apply_style()

#%% import data
data = pd.read_csv("../../hivemind2025/data/processed_data.csv")
data = TD.preprocess_trials(data)
all_trials = pd.read_csv('../../hivemind2025/data/human_trials.csv')
data['rt'] = all_trials['response_times_from_stim']

wiggle_data = pd.read_csv("../../misc-scripts/data/processed_data.csv") # FIXME: need to get this from the data not from a separate file; don't delete wiggle column in prepro
wiggles_as_array = [np.array(ast.literal_eval(row[1]['stimTrajectory'])) for row in wiggle_data.iterrows()]
wiggles_from_zero = [wiggle - wiggle[0] for wiggle in wiggles_as_array]
data['cursorPosition'] = wiggles_from_zero

#%% prepare cursor data

# screen refresh rate is 75Hz
refresh_rate = 75 # TODO: get this from the data somehow
max_rt = data['rt'].max()
data['cursorTime'] = [np.arange(0, max_rt, 1/refresh_rate)[:len(row[1]['cursorPosition'])] for row in data.iterrows()] # FIXME: this is maybe hacky
data['timeToResp'] = [np.arange(0, max_rt+0.1, 1/refresh_rate) for row in data.iterrows()]
data['timeFromResp'] = [np.arange(-max_rt-0.1, 0, 1/refresh_rate) for row in data.iterrows()]

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

data['cursorPositionNan'] = [_traj_fill_in_nans(row[1]['cursorPosition'], max_rt, refresh_rate, direction='from_stim') for row in data.iterrows()]
data['nanCursorPosition'] = [_traj_fill_in_nans(row[1]['cursorPosition'], max_rt, refresh_rate, direction='from_resp') for row in data.iterrows()]

data = TD.transform_contrast_to_ix(data) #add contrast index for plotting

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
    a.set_title(f'Example session: {key}', color=ex_data_dict[key]['col'])
    a.set_ylim(ylims)
handles, labels = ax['A'].get_legend_handles_labels()
ax['A'].legend(handles, ['incorrect', 'correct', 'rolling median RT'])
ax['B'].get_legend().remove()

### MEDIAN RT AND RT VARIANCE
# check functions in utils
plot_median_rt(data[data['instructions']==1], ax['C'], label='Instructed', color=COLORS['instructions'])
plot_median_rt(data[data['instructions']==0], ax['C'], label='Non-instructed', color=COLORS['no_instructions'])
plot_var_rt(data[data['instructions']==1], ax['D'], label='Instructed', color=COLORS['instructions'])
plot_var_rt(data[data['instructions']==0], ax['D'], label='Not instructed', color=COLORS['no_instructions'])
ax['D'].get_legend().remove()

### MOUSE WIGGLES
data_instructions = data[data['instructions']==1]
data_no_instructions = data[data['instructions']==0]

plot_mean_cursor_trajectories(data_instructions, 'cursorPositionNan', 'timeToResp', ax['E'], COLORS['instructions'], (0,3), 'Time from stimulus onset (s)')
plot_mean_cursor_trajectories(data_no_instructions, 'cursorPositionNan', 'timeToResp', ax['F'], COLORS['no_instructions'], (0,3), 'Time from stimulus onset (s)')
ax['F'].get_legend().remove()

plot_mean_cursor_trajectories(data_instructions, 'nanCursorPosition', 'timeFromResp', ax['G'], COLORS['instructions'], (-3,0), 'Time from response (s)')
plot_mean_cursor_trajectories(data_no_instructions, 'nanCursorPosition', 'timeFromResp', ax['H'], COLORS['no_instructions'], (-3,0), 'Time from response (s)')
ax['G'].get_legend().remove()
ax['H'].get_legend().remove()

fig.tight_layout()
sns.despine(fig=fig)

fig.savefig('../figures/figure3.png', dpi=300, bbox_inches='tight')
fig.savefig('../figures/figure3.svg', dpi=300, bbox_inches='tight')

# %%
