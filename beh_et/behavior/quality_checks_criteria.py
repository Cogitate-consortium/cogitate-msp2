import os
import data_reader
import pandas as pd

"""
This sets the criteria for the quality checks on subject data of experiment 2. 
It includes:
- behavioral screening criteria
- full run criteria : AT and dAT conditions for deeming subject data as valid

@author: RonyHirsch
"""

# modules
MEG = 'MEG'
ECOG = 'ECOG'
LAB = 'Lab'
ECoG = "ECoG"

METHOD_HPC = {'SA': MEG, 'SB': MEG, 'SC': data_reader.FMRI, 'SD': data_reader.FMRI,
              'SE': ECoG, 'SF': ECoG, 'SG': ECoG, 'SX': ECoG, 'SZ': MEG}  # This is used when saving to the HPC derivatives folder

# Behavioral screening exclusion criteria
SCRTPMIN = 0.125  # minimum true positive rate: 12.5%
SCRTPMAX = 0.8  # maximum true positive rate: 80%
SCRFAMAX = 0.4  # maximum false positive rate: 40%
SCREEN_OK = 'SCREEN_OK?'

# GAME levels (dAT)
FFAMAX_G = {data_reader.FMRI: 0.2, data_reader.MEEG: 0.2}  # maximum false alarm rate
DAT_OK = 'dAT_OK?'

# LOCALIZER/REPLAY levels (AT)
WRONG_LEVEL_INCLUSION_MAX = 2  # the highest number of wrong (discarded) levels a subject can have
FFAMAX_L = {data_reader.FMRI: 0.2, data_reader.MEEG: 0.2}  # maximum false alarm rate
FTPMIN_L = {data_reader.FMRI: 0.7, data_reader.MEEG: 0.7}  # minimum true positive rate
AT_OK = 'AT_OK?'

# Valid
VALID = "Is_Valid?"

# ANALYSIS TRIAL THRESHOLD
ANALYSIS_THRESHOLD = 20  # this is the MINIMUM NUMBER OF TRIALS that are required for ANY analysis for both fMRI and MEG
OPTIMIZATION_AMT = 5  # this is the number of participants each lab needs for OPTIMIZING the analysis code PRIOR to the actual analysis step


# Screening conditions
def screening_tp_criteria(ratio):
    if ratio is None:
        return None
    if SCRTPMIN <= ratio <= SCRTPMAX:
        return True
    return False


def screening_fa_criteria(ratio):
    if ratio is None:
        return None
    if ratio <= SCRFAMAX:
        return True
    return False


# dAT conditions
def full_fp_criteria_game(ratio, method):
    if ratio <= FFAMAX_G[method]:
        return True
    return False


# AT conditions
# first type of inclusion criterion for TP during REPLAY
def full_tp_criteria_localizer(ratio, method, thresh_dict=FTPMIN_L):
    if thresh_dict[method] <= ratio:
        return True
    return False


def full_fa_criteria_localizer(ratio, method):
    if ratio <= FFAMAX_L[method]:
        return True
    return False


def check_screening(sub_data: pd.Series):
    scr_tp = screening_tp_criteria(sub_data['SCREEN_hitrate'])
    scr_fa = screening_fa_criteria(sub_data['SCREEN_farate'])
    if scr_tp is None and scr_fa is None:
        return None
    if scr_tp and scr_fa:
        return True
    return False


def check_game(sub_data: pd.Series, method):
    """
    According to the threshold defined by the DMT, a subject's game (dAT condition) is valid if and only if it contains
    no more than X% false-alarm rate. (checked by full_fp_criteria_game)
    Return True if game is valid, False otherwise
    """

    fa_col = 'GAME_farate'

    res = full_fp_criteria_game(sub_data[fa_col], method)

    return res


def levels_discarded_localizer(sub_data, col_list):
    sub_discarded_total = 0
    for c in col_list:
        sub_discarded_total += sub_data[c]
    if sub_discarded_total > WRONG_LEVEL_INCLUSION_MAX:
        return False
    return True


def check_replay(sub_data: pd.Series, method):
    """
    According to DMT criteria, a replay (AT condition) is valid if and only if it has
    - less than X invalid replay levels (levels_discarded_localizer)
    - more than Y % hit rate (full_tp_criteria_localizer)
    - less than Z% false-alarm rate (full_fa_criteria_localizer)
    :param sub_data:
    :param method:
    :return:
    """
    # DISCARDED REPLAY LEVELS SUM
    loc_discarded = levels_discarded_localizer(sub_data, ['LOC_wrongButtonLevels', 'LOC_wrongTargetLevels'])
    # hits
    loc_tp = full_tp_criteria_localizer(sub_data['LOC_hitrate'], method)
    # false alarms
    loc_fa = full_fa_criteria_localizer(sub_data['LOC_farate_wfilllers'], method)

    if loc_tp and loc_fa and loc_discarded:
        return True
    return False


def check_data_table(data):
    """
    :param data:
    :return:
    """
    data_check = data.copy()
    data_check[SCREEN_OK] = None
    data_check[DAT_OK] = None
    data_check[AT_OK] = None
    data_check[VALID] = None
    for ind, row in data_check.iterrows():
        method = row[data_reader.MODALITY]
        data_check.at[ind, SCREEN_OK] = check_screening(row)
        data_check.at[ind, DAT_OK] = check_game(row, method)
        data_check.at[ind, AT_OK] = check_replay(row, method)
        data_check.at[ind, VALID] = data_check.at[ind, DAT_OK] & data_check.at[ind, AT_OK]
    return data_check


def decoding_lists(valid_qc_table):
    """
    This is the manager of the subject lists for the decoding/RSA analysis
    :param valid_qc_table: the table containing all valid subjects (i.e., those who passed the behavioral QC), to be
    analyzed.
    :param save_path: path to save the data to
    """

    """
    Predictions:
    https://docs.google.com/spreadsheets/d/1yt07HpNsbElpXVV5S-EFU3cZBo-NXPF7hp-SQFxlb0o/edit?usp=sharing
    
    Lists: 
    (1) highest min(seen face, seen object, unseen face, unseen object) in dAT  # for within-subject analysis 
    # for between-subjects
    highest min(seen face, seen object) in dAT
    highest min(unseen face, unseen object) in dAT
    
    (2) highest min(seen left, seen right, unseen left, unseen right) in dAT  # for within-subject analysis 
    # for between-subjects
    highest min(seen left, seen right) in dAT
    highest min(unseen left, unseen right) in dAT
    
    """

    ### --------- (1) ---------

    valid_qc_table.loc[:, "DECODING_min_sf_so_uf_uo"] = valid_qc_table.apply(lambda row: min(row["GAME_hits_face"], row["GAME_hits_obj"],  row["GAME_misses_face"], row["GAME_misses_obj"]), axis=1)
    valid_qc_table.loc[:, "DECODING_RANK_min_sf_so_uf_uo"] = valid_qc_table["DECODING_min_sf_so_uf_uo"].rank(ascending=False)  # meaning, ranking is in DESCENDING order

    valid_qc_table.loc[:, "DECODING_min_sf_so"] = valid_qc_table.apply(lambda row: min(row["GAME_hits_face"], row["GAME_hits_obj"]), axis=1)
    valid_qc_table.loc[:, "DECODING_RANK_min_sf_so"] = valid_qc_table["DECODING_min_sf_so"].rank(ascending=False)

    # deprecated
    #valid_qc_table.loc[:, "DECODING_min_uf_uo"] = valid_qc_table.apply(lambda row: min(row["GAME_misses_face"], row["GAME_misses_obj"]), axis=1)
    #valid_qc_table.loc[:, "DECODING_RANK_min_uf_uo"] = valid_qc_table["DECODING_min_uf_uo"].rank(ascending=False)


    ### --------- (2) ---------

    valid_qc_table.loc[:, "DECODING_min_sl_sr_ul_ur"] = valid_qc_table.apply(lambda row: min(row["GAME_hits_left"], row["GAME_hits_right"], row["GAME_misses_left"], row["GAME_misses_right"]), axis=1)
    valid_qc_table.loc[:, "DECODING_RANK_min_sl_sr_ul_ur"] = valid_qc_table["DECODING_min_sl_sr_ul_ur"].rank(ascending=False)

    valid_qc_table.loc[:, "DECODING_min_sl_sr"] = valid_qc_table.apply(lambda row: min(row["GAME_hits_left"], row["GAME_hits_right"]), axis=1)
    valid_qc_table.loc[:, "DECODING_RANK_min_sl_sr"] = valid_qc_table["DECODING_min_sl_sr"].rank(ascending=False)

    # deprecated
    #valid_qc_table.loc[:, "DECODING_min_ul_ur"] = valid_qc_table.apply(lambda row: min(row["GAME_misses_left"], row["GAME_misses_right"]), axis=1)
    #valid_qc_table.loc[:, "DECODING_RANK_min_ul_ur"] = valid_qc_table["DECODING_min_ul_ur"].rank(ascending=False)

    return valid_qc_table


def activation_lists(valid_qc_table):
    """
    This is the manager of the subject lists for the analysis of levels of activation and temporal dynamics
    :param valid_qc_table: the table containing all valid subjects (i.e., those who passed the behavioral QC), to be
    analyzed.
    :param save_path: path to save the data to
    """

    """
    Predictions:
    https://docs.google.com/spreadsheets/d/1yt07HpNsbElpXVV5S-EFU3cZBo-NXPF7hp-SQFxlb0o/edit?usp=sharing
    
    Lists: 
   (1) highest min(seen face, unseen face) in dAT
   
   (2) highest min(seen object, unseen object) in dAT
    """

    ### --------- (1) ---------

    valid_qc_table.loc[:, "ACTIVATION_min_sf_uf"] = valid_qc_table.apply(lambda row: min(row["GAME_hits_face"], row["GAME_misses_face"]), axis=1)
    valid_qc_table.loc[:, "ACTIVATION_RANK_min_sf_uf"] = valid_qc_table["ACTIVATION_min_sf_uf"].rank(ascending=False)

    ### --------- (2) ---------

    valid_qc_table.loc[:, "ACTIVATION_min_so_uo"] = valid_qc_table.apply(lambda row: min(row["GAME_hits_obj"], row["GAME_misses_obj"]), axis=1)
    valid_qc_table.loc[:, "ACTIVATION_RANK_min_so_uo"] = valid_qc_table["ACTIVATION_min_so_uo"].rank(ascending=False)

    return valid_qc_table


def synchrony_lists(valid_qc_table):
    """
    This is the manager of the subject lists for the synchronization analysis
    :param valid_qc_table: the table containing all valid subjects (i.e., those who passed the behavioral QC), to be
    analyzed.
    :param save_path: path to save the data to
    """

    """
    Predictions:
    https://docs.google.com/spreadsheets/d/1yt07HpNsbElpXVV5S-EFU3cZBo-NXPF7hp-SQFxlb0o/edit?usp=sharing
    
    Lists: 
    (1) highest min(seen) in dAT
    
    """

    ### --------- (1) ---------
    valid_qc_table.loc[:, "SYNCHRONY_min_seen"] = valid_qc_table["GAME_hits"]   # this is a duplication for comprehension purposes
    valid_qc_table.loc[:, "SYNCHRONY_RANK_min_seen"] = valid_qc_table["GAME_hits"].rank(ascending=False)

    return valid_qc_table


def baseline_lists(valid_qc_table):
    """
    This is the manager of the subject lists for the baseline analysis
    :param valid_qc_table: the table containing all valid subjects (i.e., those who passed the behavioral QC), to be
    analyzed.
    :param save_path: path to save the data to
    """

    """
    
    Predictions:
    https://docs.google.com/spreadsheets/d/1yt07HpNsbElpXVV5S-EFU3cZBo-NXPF7hp-SQFxlb0o/edit?usp=sharing
    
    Lists: 
    (1) highest min(seen, unseen) in dAT
    
    """

    valid_qc_table.loc[:, "BASELINE_min_seen_unseen"] = valid_qc_table.apply(lambda row: min(row["GAME_hits"], row["GAME_misses"]), axis=1)
    valid_qc_table.loc[:, "BASELINE_RANK_min_seen_unseen"] = valid_qc_table["BASELINE_min_seen_unseen"].rank(ascending=False)

    return valid_qc_table


def rate_subjects(meg_df, fmri_df, save_path):
    """
    Rate VALID subject dataframes according to the different analyses criteria.
    NOTE: we assume that all the subjects in both dfs are VALID, meaning their "VALID" column is True (see check_data_table)
    ***Based on decisions made 23-01-19 by the DMT ***
    :param meg_df: df of all VALID MEEG subjects
    :param fmri_df: df of all VALID fMRI subjects
    :param save_path: path to save the result rating dfs to
    """
    modality_tables = {"meg": meg_df, "fmri": fmri_df}
    modality_names = {"meg": data_reader.MEG, "fmri": data_reader.FMRI}
    modality_result = {data_reader.MEG: None, data_reader.FMRI: None}

    for mod in modality_tables:
        df = modality_tables[mod]
        # lists
        df = decoding_lists(df)
        df = activation_lists(df)
        df = synchrony_lists(df)
        df = baseline_lists(df)

        df.to_csv(os.path.join(save_path, f"{mod}_quality_checks_ranked.csv"))
        modality_result[modality_names[mod]] = df

    return modality_result
