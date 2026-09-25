import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_median_rt(data, ax=None, color=None, label=None,
                   x='contrast_ix', y='rt', groupby=['subject','contrast_ix'],
                   errorbar='se'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))
    if color is None:
        color = 'mediumturquoise'
    if label is None:
        label = 'All subjects'
    dat = data.groupby(['subject','contrast_ix']).agg({y: 'median'}).reset_index()
    sns.lineplot(data=dat,x=x,y=y,ax=ax,errorbar=errorbar,
                 color=color, label=label)
    unique_contrast_ixs = np.unique(data.contrast_ix)
    unique_signed_contrasts = np.unique(data.signed_contrast)
    ax.set_xticks([np.min(unique_contrast_ixs),np.median(unique_contrast_ixs), np.max(unique_contrast_ixs)], 
                    [np.min(unique_signed_contrasts), 0, np.max(unique_signed_contrasts)])
    ax.set_xlabel('Signed contrast (%)')
    ax.set_ylabel('Median response time (s)')
    
def plot_var_rt(data, ax=None, color=None, label=None,
                x='contrast_ix', y='rt', groupby=['subject','contrast_ix'],
                errorbar='se'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))
    if color is None:
        color = 'mediumturquoise'
    if label is None:
        label = 'All subjects'
    dat = data.groupby(['subject','contrast_ix']).agg({y: 'var'}).reset_index()
    sns.lineplot(data=dat,x=x,y=y,ax=ax,errorbar=errorbar,
                 color=color, label=label)
    unique_contrast_ixs = np.unique(data.contrast_ix)
    unique_signed_contrasts = np.unique(data.signed_contrast)
    ax.set_xticks([np.min(unique_contrast_ixs), np.median(unique_contrast_ixs), np.max(unique_contrast_ixs)], 
                    [np.min(unique_signed_contrasts), 0, np.max(unique_signed_contrasts)])
    ax.set_xlabel('Signed contrast (%)')
    ax.set_ylabel('Variance of response time (s^2)')
        
def plot_learning_curve(data, ax=None, color=None, label=None,
                         x='trial', y='correct', groupby=['subject', 'trial'],
                         x_max=600, only_easy=False, errorbar='se',
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
 
    sns.lineplot(data=data, x=x, y=y, ax=ax, errorbar=errorbar,
                 color=color, label=label)
    ax.set_xlabel('Trial number')
    if y == 'correct':
        ax.set_ylabel('P(correct choice)')
    elif y == 'rt_normalized':
        ax.set_ylabel('Normalized response time')
    return ax


def plot_mean_cursor_trajectories(data, cursor_col, time_col, ax, palette, time_window, xlabel):
    for i, contrast in enumerate(np.sort(data['stimContrast'].unique())):
        for choice in [1,-1]:
            selected_data = data[(data['stimContrast']==contrast)&(data['choice']==choice)]
            if selected_data.empty:
                print(f'no trials for choice {choice} and contrast {contrast:.2f}')
                continue
            cursor_data = np.vstack(selected_data[cursor_col].values)
            mean_cursor = np.nanmean(cursor_data, axis=0)
            if choice == 1:
                ax.plot(data[time_col].iloc[0], mean_cursor, color=palette[i], alpha=np.min([contrast+0.3, 1]), label=f'{contrast*100:.0f}')
            else:
                ax.plot(data[time_col].iloc[0], mean_cursor, color=palette[i], alpha=np.min([contrast+0.3, 1]), label=None)
    ax.legend(title='Stimulus contrast (%)')
    ax.set_xlim(time_window)
    ax.set_ylim(-600, 600)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cursor position (pix)')
    return ax


def plot_all_session_trajectories(data, color_first_last, ax, highlight_first_last=True, contrast_level=None):

    if contrast_level:
        data = data[data['stimContrast']==contrast_level]
        if len(data) == 0:
            raise(ValueError(f'No trials available at contrast {contrast_level}'))

    palette = sns.color_palette(f"blend:{color_first_last[0]},{color_first_last[1]}", n_colors=len(data))

    for i, index_row in enumerate(data.iterrows()):
        row_i, row = index_row
        if len(row['cursorTime']) != len(row['cursorPosition']):
            print(f'cursor data and time not aligned on row {row_i}')
            continue
        ax.plot(row['cursorTime'], row['cursorPosition']-row['cursorPosition'][0], color=palette[i], alpha=0.2, label=None)
    if highlight_first_last:
        first_trial = data.iloc[0]
        ax.plot(first_trial['cursorTime'], first_trial['cursorPosition'], color=palette[0], linewidth=3, label='first trial')
        last_trial = data.iloc[-1]
        ax.plot(last_trial['cursorTime'], last_trial['cursorPosition'], color=palette[-1], linewidth=3, label='last trial')
        ax.legend()
    ax.axhline(-600, 0, 3, linestyle='--', color='lightgrey', label=None)
    ax.axhline(600, 0, 3, linestyle='--', color='lightgrey', label=None)
    ax.set_xlim(0,3)
    ax.set_ylim(-630,630)
    ax.set_xlabel('Time from stimulus onset (s)')
    ax.set_ylabel('Cursor position (pix)')

    return ax