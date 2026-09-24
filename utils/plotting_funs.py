import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_median_rt(data, ax=None, color=None, label=None,
                   x='contrast_ix', y='rt', groupby=['subject','contrast_ix']):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))
    if color is None:
        color = 'mediumturquoise'
    if label is None:
        label = 'All subjects'
    dat = data.groupby(['subject','contrast_ix']).agg({y: 'median'}).reset_index()
    sns.lineplot(data=dat,x=x,y=y,ax=ax,errorbar='se',
                 color=color, label=label)
    unique_contrast_ixs = np.unique(data.contrast_ix)
    unique_signed_contrasts = np.unique(data.signed_contrast)
    ax.set_xticks([np.min(unique_contrast_ixs),np.median(unique_contrast_ixs), np.max(unique_contrast_ixs)], 
                    [np.min(unique_signed_contrasts), 0, np.max(unique_signed_contrasts)])
    ax.set_xlabel('Signed contrast (%)')
    ax.set_ylabel('Median response time (s)')
    
def plot_var_rt(data, ax=None, color=None, label=None,
                x='contrast_ix', y='rt', groupby=['subject','contrast_ix']):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))
    if color is None:
        color = 'mediumturquoise'
    if label is None:
        label = 'All subjects'
    dat = data.groupby(['subject','contrast_ix']).agg({y: 'var'}).reset_index()
    sns.lineplot(data=dat,x=x,y=y,ax=ax,errorbar='se',
                 color=color, label=label)
    unique_contrast_ixs = np.unique(data.contrast_ix)
    unique_signed_contrasts = np.unique(data.signed_contrast)
    ax.set_xticks([np.min(unique_contrast_ixs), np.median(unique_contrast_ixs), np.max(unique_contrast_ixs)], 
                    [np.min(unique_signed_contrasts), 0, np.max(unique_signed_contrasts)])
    ax.set_xlabel('Signed contrast (%)')
    ax.set_ylabel('Variance of response time (s^2)')
        
def plot_learning_curve(data, ax=None, color=None, label=None,
                         x='trial', y='correct', groupby=['subject', 'trial'],
                         x_max=600, only_easy=False,
                         smooth=False, smooth_window=10):
    data = data[data[x] < x_max].copy()
    if only_easy:
        data = data[(data['signed_contrast'] >= 50) | (data['signed_contrast'] <= -50)]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    if color is None:
        color = 'mediumturquoise'
    if label is None:
        label = 'All subjects'
 
    data['correct'] = (data['feedbackType'] > 0) + 0
 
    if smooth:
        # Sort so each subject's rolling window walks forward in trial order
        data = data.sort_values(['subject', x])
        data[y] = (
            data.groupby('subject')[y]
            .transform(lambda s: s.rolling(smooth_window, min_periods=1, center=True).mean())
        )
 
    sns.lineplot(data=data, x=x, y=y, ax=ax, errorbar='se',
                 color=color, label=label)
    ax.set_xlabel('Trial number')
    if y == 'correct':
        ax.set_ylabel('P(correct choice)')
    elif y == 'rt_normalized':
        ax.set_ylabel('Normalized response time')
    return ax


def plot_mean_cursor_trajectories(data, cursor_col, time_col, ax, color, time_window, xlabel):
    for contrast in np.sort(data['stimContrast'].unique()):
        for choice in [1,-1]:
            selected_data = data[(data['stimContrast']==contrast)&(data['choice']==choice)]
            if selected_data.empty:
                continue
            cursor_data = np.vstack(selected_data[cursor_col].values)
            if choice == 1:
                ax.plot(data[time_col].iloc[0], mean_cursor, color=color, alpha=np.min([contrast+0.3, 1]), label=f'{contrast:.2f}')
            else:
                ax.plot(data[time_col].iloc[0], mean_cursor, color=color, alpha=np.min([contrast+0.3, 1]), label=None)
    ax.legend(title='Stimulus contrast')
    ax.set_xlim(time_window)
    ax.set_ylim(-600, 600)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cursor position (pix)')
    return ax