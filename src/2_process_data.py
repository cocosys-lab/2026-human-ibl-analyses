"""
Input: human_trials_release.csv
Output: processed_data_2026.csv

"""
#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

data_path = Path(__file__).resolve().parents[1] / 'data'
data_path.mkdir(parents=True, exist_ok=True)

trials_df = pd.read_csv(data_path / 'human_trials_release.csv')

processed_df = trials_df.copy()

# get sub ids and make sure they are in correct format
processed_df['subject'] = processed_df['subject'].astype(str)
processed_df['subject'] = processed_df['subject'].str.zfill(3)
sub_ids = processed_df['subject'].unique()

# recode choices
processed_df['choice'] = processed_df['choice'].map({0:1, 1:-1, np.nan: 0, pd.NA: 0})
assert processed_df['choice'].isna().sum() == 0

# rescale stimulus contrast between -1 and 1
processed_df['stimSide'] = processed_df['eccentricity'].replace({-15:1, 15:-1}) # this matches the direction of the choice coding

scaler = MinMaxScaler(feature_range=(0, 1))
processed_df['contLeft'] = scaler.fit_transform(processed_df[['contrastLeft']]).ravel().round(2)
processed_df['contRight'] = scaler.fit_transform(processed_df[['contrastRight']]).ravel().round(2)

# compute some columns
processed_df['stimContrast'] = processed_df['contLeft'] + processed_df['contRight']
processed_df['sideContrast'] = processed_df['contLeft'] - processed_df['contRight'] # this matches the direction of the choice coding

# replace with nans when both sides are 0
# processed_df.loc[(processed_df['contrastLeft']==0.5)&(processed_df['contrastRight']==0.5), 'contLeft'] = np.nan
# processed_df.loc[(processed_df['contrastLeft']==0.5)&(processed_df['contrastRight']==0.5), 'contRight'] = np.nan

# get the stimulus trajectory and convert to float
# N.B. The cursor position could be recorded for up to 506ms after the response window time out at 10s. This was always registered as a no-response and feedback was given accordingly, even if the cursor crossed the response threshold in the extra samples. We therefore remove the extra samples here.
processed_df['cursorPosition'] = None
processed_df['cursorTime'] = None
not_same_len = []
discrepancy = []
for idx in processed_df.index:
    mouse_str = processed_df.loc[idx, 'dot_trajectory'][1:-1]
    mouse_float = np.array([float(n) for n in mouse_str.split(",")])
    processed_df.at[idx, 'cursorPosition'] = mouse_float
    time_str = processed_df.loc[idx, 'mouse.time'][1:-1]
    time_float = np.array([float(n) for n in time_str.split(",")])
    processed_df.at[idx, 'cursorTime'] = time_float
    if len(mouse_float) > len(time_float):
        processed_df.at[idx, 'response_time_from_stim'] = np.nan
        processed_df.at[idx, 'cursorPosition'] = mouse_float[:len(time_float)]

# only keep columns we need to avoid confusion
processed_df = processed_df[['subject', 'session', 'trial', 'probabilityLeft', 'contLeft', 'contRight', 'choice', 'firstMovement_times_from_stim', 'response_times_from_stim', 'feedbackType', 'stimContrast', 'sideContrast', 'stimSide', 'instructions', 'cursorPosition', 'cursorTime', 'dot.started']]
processed_df.rename(columns={'contLeft':'contrastLeft', 'contRight':'contrastRight'}, inplace=True)

# save csv
processed_df.to_csv(data_path / 'processed_data_2026.csv', index=False)
