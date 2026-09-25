import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from utils.plot_config import apply_style, COLORS
from utils.form_psychometrics import fit_psychometric #, plot_psychometric_fit
import utils.transform_data as TD
from scipy.stats import ttest_ind, ttest_rel
import seaborn as sns

apply_style()

#%% import data
data = pd.read_csv("../../hivemind2025/data/processed_data.csv")
data = TD.preprocess_trials(data)
all_trials = pd.read_csv('../../hivemind2025/data/human_trials.csv')
data['rt'] = all_trials['response_times_from_stim']

#%% fit psychometric curves for each subject and each block type
subjects = data['subject'].unique()
psychometric_params = []
blocks = ['Left', 'Right']
for subject in subjects:
    subdat = data[data['subject'] == subject]
    pars, xx, yy = fit_psychometric(subdat)
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
            pars, xx, yy = fit_psychometric(block_df)
            sub_dict.update({
                f'bias_{block}': pars[0],
                f'slope_{block}': pars[1],
                f'lapse_low_{block}': pars[2],
                f'lapse_high_{block}': pars[3]
            })
            # psychometric_params.append({
            #     'subject': subject,
            #     'instructions': block_df['instructions'].iloc[0],
            #     f'bias_{block}': pars[0],
            #     f'slope_{block}': pars[1],
            #     f'lapse_low_{block}': pars[2],
            #     f'lapse_high_{block}': pars[3]
            # })
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
psychometric_df[r'$\Delta$ Absolute bias'] = psychometric_df['Abolute_bias_Right'] - psychometric_df['Absolute_bias_Left']
psychometric_df[r'$\Delta$ Mean lapse'] = psychometric_df['Mean_lapse_Right'] - psychometric_df['Mean_lapse_Left']
psychometric_df[r'$\Delta$ Slope'] = psychometric_df['slope_Right'] - psychometric_df['slope_Left']
psychometric_df['Slope'] = psychometric_df['slope']

#%% plot differences in parameters between instructed and non-instructed subjects
instructed = psychometric_df[psychometric_df['instructions'] == 0]
non_instructed = psychometric_df[psychometric_df['instructions'] == 1]
fig, ax = plt.subplots(1,3, figsize=(15,5))

for i,param in enumerate(['Absolute bias', 'Slope', 'Mean lapse']):
    sns.stripplot(x='instructions', y=param, data=psychometric_df, 
                  ax=ax[i], palette=[COLORS['no_instructions'], COLORS['instructions']],
                  hue='instructions',  jitter=True, edgecolor='lightgray', linewidth=0.5, size=6,
                  zorder=1)
    sns.boxplot(x='instructions', y=param, data=psychometric_df, 
                ax=ax[i], palette=[COLORS['no_instructions'], COLORS['instructions']],
                hue='instructions',legend=False, saturation=0.7,
                zorder=0)
    test_res = ttest_ind(instructed[param], non_instructed[param], 
                         equal_var=False, nan_policy='omit')
    # pstr = f"p = {test_res.pvalue:.3f}" if test_res.pvalue >= 0.001 else "p < 0.001"
    # ax[i].set_title(f"{param} ({pstr})")
    annotation = '***' if test_res.pvalue < 0.001 else ('**' if test_res.pvalue < 0.01 else ('*'if test_res.pvalue < 0.05 else 'n.s.') )
    ax[i].text(0.5, 0.95, annotation, transform=ax[i].transAxes, ha='center', va='top')
    ax[i].plot([0.3, 0.7], [0.9, 0.9], transform=ax[i].transAxes, color='k', lw=1)
    ax[i].get_legend().remove()
    ax[i].set_xticklabels(['Not instructed', 'Instructed'])
    ax[i].set_xlabel('')
    
ax[2].legend(['Not instructed', 'Instructed'], loc='best')
sns.despine(fig=fig)
# %% then another figure for the differences between left and right blocks, between groups
fig, ax = plt.subplots(1,3, figsize=(15,5))
for i,param in enumerate([r'$\Delta$ Absolute bias', r'$\Delta$ Mean lapse', r'$\Delta$ Slope']):
    sns.stripplot(x='instructions', y=param, data=psychometric_df, 
                  ax=ax[i], palette=[COLORS['no_instructions'], COLORS['instructions']],
                  hue='instructions', jitter=True, edgecolor='lightgray', linewidth=0.5, size=6,
                  zorder=1)
    sns.boxplot(x='instructions', y=param, data=psychometric_df, 
                ax=ax[i], palette=[COLORS['no_instructions'], COLORS['instructions']],
                hue='instructions', hue_order = [0,1],legend=False, saturation=0.7,
                zorder=0)
    test_res = ttest_ind(instructed[param], non_instructed[param], 
                         equal_var=False, nan_policy='omit')
    # pstr = f"p = {test_res.pvalue:.3f}" if test_res.pvalue >= 0.001 else "p < 0.001"
    # ax[i].set_title(f"{param} ({pstr})")
    annotation = '***' if test_res.pvalue < 0.001 else ('**' if test_res.pvalue < 0.01 else ('*'if test_res.pvalue < 0.05 else 'n.s.') )
    ax[i].text(0.5, 0.95, annotation, transform=ax[i].transAxes, ha='center', va='top')
    ax[i].plot([0.3, 0.7], [0.9, 0.9], transform=ax[i].transAxes, color='k', lw=1)
    ax[i].get_legend().remove()
    ax[i].set_xticklabels(['Not instructed', 'Instructed'])
    ax[i].set_xlabel('')
ax[2].legend(['Not instructed', 'Instructed'], loc='best')
sns.despine(fig=fig)

# %% then, get the statistical difference within instruction group between left and right
#first, rotate the dataframe to have one row per subject and block type
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
#then plot
fig, ax = plt.subplots(1,4, figsize=(20,5))
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
ax[3].legend(['Left block', 'Right block'], loc='best')
sns.despine(fig=fig)

# %% also make the t-tests for accuracy on high-contrast trials
data_high_contrast = data[np.abs(data.signed_contrast)>=50]
data_high_contrast['correct'] = (data['feedbackType']==1).astype(int)
data_high = data_high_contrast.groupby(['subject','instructions'])['correct'].mean().reset_index()
data_high['correct'] = data_high['correct']*100
fig, ax = plt.subplots(figsize=(5,5))
sns.stripplot(data=data_high, x='instructions', y='correct',
              hue='instructions', ax=ax, jitter=True, edgecolor='white', linewidth=0.5,
              palette=[COLORS['no_instructions'],COLORS['instructions']],
              zorder=1)
sns.boxplot(data=data_high, x='instructions',y='correct',
            hue='instructions',ax=ax, saturation=0.5,
            palette=[COLORS['no_instructions'],COLORS['instructions']],
            zorder=0)
ttest_result = ttest_ind(data_high.loc[data_high.instructions==0,'correct'], data_high.loc[data_high.instructions==1,'correct'],
                         alternative='two-sided')
annotation = '***' if ttest_result.pvalue < 0.001 else ('**' if ttest_result.pvalue < 0.01 else ('*'if ttest_result.pvalue < 0.05 else 'n.s.') )
ax.text(0.5, 1.05, annotation, transform=ax.transAxes, ha='center', va='top')
ax.plot([0.25, 0.75], [1, 1], transform=ax.transAxes, color='k', lw=1)
ax.legend(['Not instructed','Instructed'],title='')
ax.set_ylabel('P(correct) (%)')
ax.set_xlabel('')
ax.set_xticklabels(['Not instructed','Instructed'])

sns.despine(fig=fig)



# %%
