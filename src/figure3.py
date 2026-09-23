#%% imports

import sys
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

data = TD.transform_contrast_to_ix(data) #add contrast index for plotting

#%% select example sessions
ex_session_num_no = '096'
ex_session_num_ins = '094'

ex_session_no = data[data['subject']==ex_session_num_no]
ex_session_ins = data[data['subject']==ex_session_num_ins]

ex_data_dict = {}
ex_data_dict['Instructed'] = {'data':ex_session_no, 'col':COLORS['instructions']}
ex_data_dict['Non-instructed'] = {'data':ex_session_ins, 'col':COLORS['no_instructions']}

#%% create figure with proper dimensions

fig_layout = '''
            AA
            BB
            CD
            EF
            '''

fig, ax = plt.subplot_mosaic(fig_layout, figsize=(10,12), height_ratios=[0.2, 0.2, 0.4, 0.4])

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

    a.set(xlabel="Trial number", ylabel="RT (s)", ylim=[0.01, 10])
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
plot_var_rt(data[data['instructions']==0], ax['D'], label='Non-instructed', color=COLORS['no_instructions'])
ax['D'].get_legend().remove()

### MOUSE WIGGLES
# ax['E'] - early and late wiggles from example sessions, with dotted line at threshold 'threshold left/right response'
# ax['F'] - average wiggles for ins and no, left/right

fig.tight_layout()
sns.despine(fig=fig)

fig.savefig('../figures/figure3.png', dpi=300, bbox_inches='tight')
fig.savefig('../figures/figure3.svg', dpi=300, bbox_inches='tight')

# %%
