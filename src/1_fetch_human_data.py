import pandas as pd
import numpy as np
import ast

from pathlib import Path
from tqdm import tqdm


def get_info_value(session_info, name):
    values = session_info.loc[session_info['name'] == name, 'value']
    return values.iloc[0] if len(values) else pd.NA


def get_selection(value):
    if pd.isna(value):
        return pd.NA
    selections = ast.literal_eval(value)
    return selections[-1] if selections else pd.NA


def load_human_session(human_data_dir: Path, subj: Path):
    session_info = pd.read_csv(human_data_dir / subj / 'alf' / 'session_info.csv')
    data = pd.read_csv(human_data_dir / subj / 'alf' / 'trials_table.csv')
    data = data.drop(columns=[c for c in data.columns if 'started' in c and c != 'dot.started'])
    data = data[data.columns.drop(list(data.filter(regex='stopped')))]
    # data = data[data.columns.drop(list(data.filter(regex='mouse')))]
    data = data[data.columns.drop(list(data.filter(regex='Unnamed')))]
    data.rename(columns={'response_time':'response_times_from_stim', 'signed_contrast':'sideContrast', 'firstMovement_time':'firstMovement_times_from_stim'}, inplace=True)
    data['subject'] = subj.parts[-1]
    data['session'] = data['session'].astype(str)
    if session_info[session_info['name']=='instructions']['value'].isna().all():
        data['instructions'] = pd.NA
    else:
        data['instructions'] = int(float(session_info[session_info['name']=='instructions']['value'].values[0]))

    participant = {
        'subject': subj.name,
        'session_start': get_info_value(session_info, 'expStart'),
        'gender': get_selection(get_info_value(session_info, 'mouse_5.clicked_name')),
        'age': get_info_value(session_info, 'age_slider.response'),
        'handedness': get_selection(get_info_value(session_info, 'mouse_6.clicked_name')),
        'textbox1_text': get_info_value(session_info, 'textbox1.text'),
        'textbox2_text': get_info_value(session_info, 'textbox2.text'),
        'textbox3_text': get_info_value(session_info, 'textbox3.text'),
        'textbox4_text': get_info_value(session_info, 'textbox4.text'),
        'textbox5_text': get_info_value(session_info, 'textbox5.text'),
    }
    return data, participant


def load_human_trials(human_data_dir:Path):

    subjects = sorted(list(human_data_dir.iterdir()))

    session_trials = []
    participants = []
    for subj in tqdm(subjects):
        try:
            data, participant = load_human_session(human_data_dir, subj)
            if data is not None:
                session_trials.append(data)
                participants.append(participant)
                
        except  Exception as e:
            print("skipped file with error", subj, e)

    human_trials_df = pd.concat(session_trials)
    human_trials_df.set_index(['subject', 'session', 'trial'], inplace=True)
    human_trials_df.sort_index(level=['subject','session','trial'], ascending=[1,1,1], inplace=True)

    human_trials_df = human_trials_df.drop(columns=['date', 'expName', 'psychopyVersion', 'expStart',])

    return human_trials_df, pd.DataFrame(participants)

human_data_dir = Path(r'D:\winshare\workgroups\FSW\VISUAL-DECISIONS\subjects_2024')

trials_df, participants_df = load_human_trials(human_data_dir)
### error for subjects - no session info:
# 16 - experiment was force quit so no session_info
# 19 - experiment was force quit so no session_info
# 84 - participant reran as 85 (no data in 84) 
# 90 - not a data file

bad_subjects = ['014', '082A'] # incomplete sessions

trials_df.drop(bad_subjects, level=0, inplace=True)

pilots = ['001', '002', '003', '004', '005']
trials_df.drop(pilots, level=0, inplace=True)

trial_counts = trials_df.groupby(level='subject').size()
short_subjects = trial_counts[trial_counts < 600].index.tolist()
print('Excluded subjects with fewer than 600 trials:', short_subjects)
trials_df.drop(short_subjects, level=0, inplace=True)

excluded = bad_subjects + pilots + short_subjects
participants_df = participants_df[~participants_df['subject'].isin(excluded)].copy()
participants_df.sort_values('subject', inplace=True)

# save data in csv
Path('./data').mkdir(parents=True, exist_ok=True)
trials_df.to_csv('./data/human_trials.csv')
participants_df.to_csv('./data/participants.csv', index=False)
print(f'Saved {trials_df.shape[0]} trials from {participants_df.shape[0]} participants')
