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

#%% select example sessions
ex_session_num_no = '096'
ex_session_num_ins = '094'

ex_session_no = data[data['subject']==ex_session_num_no]
ex_session_ins = data[data['subject']==ex_session_num_ins]

data_dict = {'Instructions':{}, 'No Instructions':{}}
data_dict['Instructions'] = {'data':ex_session_no, 'col':COLORS['instructions']}
data_dict['No Instructions'] = {'data':ex_session_ins, 'col':COLORS['no_instructions']}

#%% create figure with proper dimensions

fig_layout = '''
            AA
            BB
            CD
            EF
            '''

fig, ax = plt.subplot_mosaic(fig_layout, figsize=(10,12), height_ratios=[0.2, 0.2, 0.4, 0.4])

### EXAMPLE SESSIONS - ROLLING RT
for key, a in zip(data_dict, [ax['A'], ax['B']]):
    d = data_dict[key]['data']

    # rt scatterplot
    sns.scatterplot(data=d, ax=a, 
        x='trial', y='rt', 
        style='feedbackType', hue='feedbackType',
        palette={1.:COLORS['correct'], -1.:COLORS['incorrect']}, 
        markers={1.:'o', -1.:'X'}, s=10, edgecolors='face',
        alpha=.5, legend=False)

    # running median overlaid
    sns.lineplot(
        data=d[['trial', 'rt']].rolling(10).median(), 
        ax=a,
        x='trial', y='rt', color='black', errorbar=None)

    a.set(xlabel="Trial number", ylabel="RT (s)", ylim=[0.01, 10])
    a.set_yscale("log")
    a.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y,pos:
        ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    a.set_title(f'Example Session: {key}', color=data_dict[key]['col'])


### MEDIAN RT AND RT VARIANCE
# check functions in utils


### MOUSE WIGGLES
# ax['E'] - early and late wiggles from example sessions, with dotted line at threshold 'threshold left/right response'
# ax['F'] - average wiggles for ins and no, left/right