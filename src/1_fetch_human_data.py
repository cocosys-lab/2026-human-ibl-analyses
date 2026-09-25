#%%
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

    subject_id = subj.name
    if subject_id == '041A':
        subject_id = '041'
    elif subject_id == '041B':
        subject_id = '042'
    elif subject_id == '082A':
        subject_id = '082'

    data['subject'] = subject_id
    data['session'] = data['session'].astype(str)
    if session_info[session_info['name']=='instructions']['value'].isna().all():
        data['instructions'] = pd.NA
    else:
        data['instructions'] = int(float(session_info[session_info['name']=='instructions']['value'].values[0]))

    participant = {
        'subject': subject_id,
        'session_start': get_info_value(session_info, 'expStart'),
        'gender': get_selection(get_info_value(session_info, 'mouse_5.clicked_name')),
        'age': get_info_value(session_info, 'age_slider.response'),
        'handedness': get_selection(get_info_value(session_info, 'mouse_6.clicked_name')),
        'textbox1_text': get_info_value(session_info, 'textbox1.text'),
        'textbox2_text': get_info_value(session_info, 'textbox2.text'),
        'textbox3_text': get_info_value(session_info, 'textbox3.text'),
        'textbox4_text': get_info_value(session_info, 'textbox4.text'),
        'note': '',
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

human_data_dir = Path('J:\Workgroups\FSW\VISUAL-DECISIONS\subjects_2024')

trials_df, participants_df = load_human_trials(human_data_dir)
### error for 4 folders - no session info:
# 16 - experiment was force quit so no session_info
# 19 - experiment was force quit so no session_info
# 84 - participant reran as 85 (no data in 84) 
# 90 - not a data file

incomplete_subjects = ['014', '082', '040', '071', '083']
trials_df.drop(incomplete_subjects, level=0, inplace=True)

pilots = ['001', '002', '003', '004', '005']
trials_df.drop(pilots, level=0, inplace=True)

# 087 is the same person as 066; retain their first participation (066).
repeat_participation_subjects  = ['087']
trials_df.drop(repeat_participation_subjects, level=0, inplace=True)

# Keep the first 600 trials from subject 006.
trials_df = trials_df[
    (trials_df.index.get_level_values('subject') != '006')
    | (trials_df.index.get_level_values('trial') < 600)
]
participants_df.loc[
    participants_df['subject'] == '006', 'note'
] = 'Session had no 600-trial limit and was manually stopped; only trials 0-599 are included in the trial file.'


excluded = incomplete_subjects + pilots + repeat_participation_subjects
participants_df = participants_df[~participants_df['subject'].isin(excluded)].copy()
participants_df.sort_values('subject', inplace=True)

# save data in csv
data_path = Path(__file__).resolve().parents[1] / 'data'
data_path.mkdir(parents=True, exist_ok=True)

trials_df.to_csv(data_path / 'human_trials_release.csv')
participants_df.to_csv(data_path / 'participants_info.csv', index=False)
print(f'Saved {trials_df.shape[0]} trials from {participants_df.shape[0]} participants')
