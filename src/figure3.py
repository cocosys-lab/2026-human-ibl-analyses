#%% imports

import sys
import ast
sys.path.append("..")
from utils.plot_config import apply_style, COLORS
import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
import pandas as pd
import numpy as np
import utils.transform_data as TD
from matplotlib.gridspec import GridSpec
from utils.plotting_funs import plot_median_rt, plot_var_rt
import utils.form_psychometrics as P

apply_style()

#%% import data
data = pd.read_csv("../../hivemind2025/data/processed_data.csv")
data = TD.preprocess_trials(data)
all_trials = pd.read_csv('../../hivemind2025/data/human_trials.csv')
data['rt'] = all_trials['response_times_from_stim']

wiggle_data = pd.read_csv("../../misc-scripts/data/processed_data.csv")
wiggles_as_array = [np.array(ast.literal_eval(row[1]['stimTrajectory'])) for row in wiggle_data.iterrows()]
data['cursorPosition'] = wiggles_as_array

# screen refresh rate is 75Hz
refresh_rate = 75 # TODO: get this from the data somehow
max_rt = data['rt'].max()
data['cursorTime'] = [np.arange(0, max_rt, 1/refresh_rate)[:len(row[1]['cursorPosition'])] for row in data.iterrows()] # FIXME: this is maybe hacky
data['timeToResp'] = [np.arange(0, max_rt+0.1, 1/refresh_rate) for row in data.iterrows()]
data['timeFromResp'] = [np.arange(0, -max_rt-0.1,-1/refresh_rate) for row in data.iterrows()]

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
ex_session_num_no = '010'
ex_session_num_ins = '094'

ex_session_no = data[data['subject']==ex_session_num_no]
ex_session_ins = data[data['subject']==ex_session_num_ins]

ex_data_dict = {}
ex_data_dict['Instructed'] = {'data':ex_session_no, 'col':COLORS['instructions']}
ex_data_dict['Not instructed'] = {'data':ex_session_ins, 'col':COLORS['no_instructions']}

#%% create figure with proper dimensions

fig_layout = '''
            AA
            BB
            CD
            EF
            GH
            '''

fig, ax = plt.subplot_mosaic(fig_layout, figsize=(10,12), height_ratios=[0.125, 0.125, 0.25, 0.25, 0.25])

### EXAMPLE SESSIONS - ROLLING RT
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
    a.set_ylim(d['rt'].min()-0.01, d['rt'].max()+0.1)
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
# what is the response threshold? 618
# # ax['E'] - early and late wiggles from example sessions, with dotted line at threshold 'threshold left/right response'

# x_time = np.arange(0, data['rt'].max(), 1/75)

# for key in ex_data_dict:
#     d = ex_data_dict[key]['data']

#     for choice in [1, -1]:
#         early_wiggle_trial = d[(d['choice']==choice)&(d['feedbackType']==1)&(d['stimContrast']==1)].iloc[0]# early trial: left choice - correct - high contrast
#         early_wiggle = early_wiggle_trial['cursorPosition'] 
#         early_wiggle -= early_wiggle[0]
#         early_trial_n = early_wiggle_trial['trial']

#         late_wiggle_trial = d[(d['choice']==choice)&(d['feedbackType']==1)&(d['stimContrast']==1)].iloc[-10] # late trial: left choice - correct - high contrast
#         late_wiggle = late_wiggle_trial['cursorPosition']
#         late_wiggle -= late_wiggle[0]
#         late_trial_n = d[(d['choice']==1)&(d['feedbackType']==1)].iloc[-10]['trial']

#         ax['E'].plot(x_time[:len(early_wiggle)], early_wiggle, color=ex_data_dict[key]['col'], alpha=0.5, linewidth=2, label=f'trial {early_trial_n}')
#         ax['E'].plot(x_time[:len(late_wiggle)], late_wiggle, color=ex_data_dict[key]['col'], alpha=1, linewidth=2, label=f'trial {late_trial_n}')

# ax['E'].axhline(618, 0, x_time[-1], linestyle='--', color='grey', label='threshold left response')
# ax['E'].axhline(-618, 0, x_time[-1], linestyle='--', color='grey', label='threshold right response')
# ax['E'].set_xlabel('time from stimulus onset (s)')
# ax['E'].set_ylabel('cursor position (pix)')
# ax['E'].set_title('Cursor movements on easy, correct trials')
# # TODO: add legend for early/late

# # ax['F'] - average wiggles for ins and no, left/right
# # for now, I am plotting all wiggles in example session
# for row in ex_session_no.iterrows():
#     ax['F'].plot(row[1]['cursorTime'], row[1]['cursorPosition']-row[1]['cursorPosition'][0], color=COLORS['no_instructions'], alpha=0.1)

# for row in ex_session_ins.iterrows():
#     ax['F'].plot(row[1]['cursorTime'], row[1]['cursorPosition']-row[1]['cursorPosition'][0], color=COLORS['instructions'], alpha=0.1)

# ax['F'].axhline(618, 0, x_time[-1], linestyle='--', color='grey', label='threshold left response')
# ax['F'].axhline(-618, 0, x_time[-1], linestyle='--', color='grey', label='threshold right response')
# ax['F'].set_xlabel('time from stimulus onset (s)')
# ax['F'].set_ylabel('cursor position (pix)')

# cursor_data = np.vstack(data['cursorPositionNan'].values)
# cursor_data -= np.ones_like(cursor_data) * cursor_data[0,0]

# data_no_instructions = data[data['instructions']==0]

def plot_mean_cursor_trajectories_from_stim(data, cursor_col, time_col, ax, color, time_window, xlabel):
    for contrast in np.sort(data['stimContrast'].unique()):
        for choice in [1,-1]:
            selected_data = data[(data['stimContrast']==contrast)&(data['choice']==choice)]
            cursor_data = np.vstack(selected_data[cursor_col].values)
            cursor_data -= np.ones_like(cursor_data) * cursor_data[0,0]

            mean_cursor = np.nanmean(cursor_data, axis=0)
            if choice == 1:
                ax.plot(data[time_col].iloc[0], mean_cursor, color=color, alpha=np.min([contrast+0.3, 1]), label=f'{contrast:.2f}')
            else:
                ax.plot(data[time_col].iloc[0], mean_cursor, color=color, alpha=np.min([contrast+0.3, 1]), label=None)
    ax.legend(title='Stimulus contrast')
    ax.set_xlim(time_window)
    ax.set_ylim(-600, 600)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cursor position (pix)')

data_instructions = data[data['instructions']==1]
data_no_instructions = data[data['instructions']==0]

plot_mean_cursor_trajectories_from_stim(data_instructions, 'cursorPositionNan', 'timeToResp', ax['E'], COLORS['instructions'], (0,3), 'Time from stimulus onset (s)')
plot_mean_cursor_trajectories_from_stim(data_no_instructions, 'cursorPositionNan', 'timeToResp', ax['F'], COLORS['no_instructions'], (0,3), 'Time from stimulus onset (s)')

plot_mean_cursor_trajectories_from_stim(data_instructions, 'nanCursorPosition', 'timeFromResp', ax['G'], COLORS['instructions'], (-3,0), 'Time from response (s)')
plot_mean_cursor_trajectories_from_stim(data_no_instructions, 'nanCursorPosition', 'timeFromResp', ax['H'], COLORS['no_instructions'], (-3,0), 'Time from response (s)')

fig.tight_layout()
sns.despine(fig=fig)

fig.savefig('../figures/figure3.png', dpi=300, bbox_inches='tight')
fig.savefig('../figures/figure3.svg', dpi=300, bbox_inches='tight')

# %%
