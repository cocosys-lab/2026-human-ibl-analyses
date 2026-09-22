'''Transform data for plotting, creates the adequate columns'''
import pandas as pd
import numpy as np

def preprocess_trials(df, remove_pilots=False):
    """
    Add block, phase, signed_contrast, choice_right.
    This preprocessing is shared by all plots.
    """
    df = df.copy()
    if remove_pilots:
        pilots = ['001','002','003','004','005']
        df = df[df.subject.isin(pilots) == False]
    df.loc[df.subject=='041A','subject'] = '041'
    df.loc[df.subject=='041B','subject'] = '042'
    # block: keep only Left / Right blocks (drop unbiased)
    df["block"] = df["probabilityLeft"].apply(
        lambda x: "Left" if x > 0.5 else ("Right" if x < 0.5 else "Unbiased")
    )
    # phase by trial index
    df["phase"] = df["trial"].apply(
        lambda x: "late" if x > 345 else ("early" if x > 90 else "first90")
    )
    # choice_right: choice == -1 means right
    df["choice_right"] = (df["choice"].astype(int) == -1).astype(int)
    # contrast: [-0.2, 0.2] -> [-100, 100]; flip sign to match choice coding
    df["signed_contrast"] = df["sideContrast"].astype(float) * 100 * (-1)
    #remove null contrast trials
    df = df[~np.isnan(df.signed_contrast)]
    #compute trial since block change
    change_points = df["probabilityLeft"].ne(df["probabilityLeft"].shift())#.cumsum()
    group_id = change_points.cumsum()
    up = df.groupby(group_id).cumcount()
    run_len = df.groupby(group_id)["probabilityLeft"].transform("size")
    # countdown until the next change
    countdown = up - run_len
    # we want final 5 trials before a switch as a countdown
    switch_soon = (countdown<0) & (countdown>=-5)
    df['Trials_since_change'] = np.where(switch_soon, countdown, up)
    return df

def transform_contrast_to_ix(data):
    # transform contrast to index for plotting
    data['contrast_ix'] = np.nan
    for i, c in enumerate(np.sort(data.signed_contrast.unique())):
        data.loc[data.signed_contrast==c,'contrast_ix'] = i
    return data