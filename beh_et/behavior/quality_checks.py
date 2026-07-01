import os
import pickle
# ENVIRONMENT variable
os.environ["OUTDATED_IGNORE"] = "1"
import pandas as pd
import numpy as np
import data_reader
import quality_checks_criteria
import boxplotter
import lineplotter
import data_saver
import stats
import response_type_analysis
import diff_perf_analysis
import general_analysis
import vg_features_analysis
import itertools
import seaborn as sns

"""
This module performs quality checks on subject data of experiment 2. 

### LAST UPDATED: 2022-02-28 : All thresholds and conditions are based on DMT decisions from said date. ###

The quality-check module includes: 
- Checks of the behavioral screening session data (i.e., 'SCREEN') : to determine whether subjects should be summoned 
to a full session of the experiment, based on their behavioral data in the screening.
- Checks of the experimental data : to determine which subjects will be excluded from the general analyses of exp.2.
The module outputs a table in which each row is a subject and each column is a statistic of this subject's behavior 
in the screening and/or in the game. Subjects who don't have full game data are analyzed anyway, the game-analysis
columns will remain empty. 
Based on the thresholds that exist in the "quality_checks_criteria" module, the last 3 columns of the output table 
indicate whether the subject's screening data is valid, whether the subject's dAT data is valid, and whether the 
subject's AT data is valid. Of course, empty/missing data will lead to the validity being false.
Notes: 
The data is uploaded using the data_reader module, 
The criteria for those checks is imported from the quality_checks_criteria module, 
The hit and fa rate is plotted using the boxplotter module.

@author: RonyHirsch
"""


# stimulus types
FACE = 'Face'
OBJ = 'Object'
NON = 'None'

# responseEvaluation types
EVAL = 'responseEvaluation'
TRUEPOSITIVE = 'TruePositive'
FALSEPOSITIVE = 'FalsePositive'
TRUENEGATIVE = 'TrueNegative'
FALSENEGATIVE = 'FalseNegative'
IRREL = 'OutsideWindowPress'
IRREL_ADDITION = 'AdditionalPress'

# additional 'response' column types
RESP_COL = 'response'
AFTERW = 'AfterWindowPress'

# level column and other columns
LEVEL = "activeLevelID"
REAL_LEVEL = "currentLevelID"
LEVEL_MAX_PROBES = 13  # maximum number of probes in game level
DT_SINCE_LAST = "dTSinceLast_NoPauses"
STIM_TYPE = "stimulusType"
STIM_LOC = "stimulusLocation"
TYPE_ = "type"
TYPE_STIM_LOC = "LOCALIZER_STIMULUS"
STIM_ONSET_TS = "stimulusOnsetTS"
RESP_TS = "responseTS"
TL = "TopLeft"
TR = "TopRight"
BL = "BottomLeft"
BR = "BottomRight"
NEG = 'Negative'
LEFT = "Left"
RIGHT = "Right"
BOTTOM = "Bottom"
TOP = "Top"
DIFF = "difficulty"
LOCATIONS = [TL, TR, BL, BR]
SPECIAL_LOCS = ["Top", "Bottom", "Left", "Right"]
LOCATION_NAMES = [item for sublist in [[f'hits_{loc}', f'hitrate_{loc}', f'misses_{loc}', f'missrate_{loc}'] for loc in (LOCATIONS + SPECIAL_LOCS)] for item in sublist]
FLIP_DICT = {FALSENEGATIVE: TRUEPOSITIVE, TRUENEGATIVE: FALSEPOSITIVE,
             TRUEPOSITIVE: TRUEPOSITIVE, FALSEPOSITIVE: FALSEPOSITIVE}

# replay targets df column names
LEVEL_ID = "LevelToReplayID_1Based"
TARGET_TYPE = "TargetType"
QID = 'questionID'

# replay recognition of faulty levels
WRONG_BUTTON_SUSPECT = {'hitrate': 0, 'farate': 0}
WRONG_TARGET_SUSPECT = {'missrate': 0.6, 'farate_noblanks': 0.6}  # NOTE THAT THIS IS THE FALSE ALARM RATE W/O BLANKS - MEANING ONLY RESPONSES TO WRONG STIMULI

# different cases for replay level count and calculations
# orig = take as-is, after_window = include after window presses (see calculation for more),
# wrong_button = remove replay levels with WRONG_BUTTON_SUSPECT, wrong_instructions = same for WRONG_TARGET_SUSPECT
REPLAY_CASES = {"orig": 0, "after_window": 1, "wrong_button": 2, "wrong_instructions": 3, "combo": 4}
REPLAY_LEVELS = [l for l in range(100, 108)]

# Config file parameters
INPUT = "Input"
CONFIG_INPUT_SERIAL_PORT = f"{INPUT}SerialPort_ResponseBox_TwoHands"
CONFIG_INPUT_KEY_CODE = f"{INPUT}KeyCode_ResponseBox_TwoHands"
CONFIG_USE_PROBE_FOR_REPLAY = "useProbeYesForReplayLevelReporting"
PROBE_YES = "_Probe_Yes"
PROBE_NO = "_Probe_No"
KEY_CODES = "_keyCodes"
KEY_FACES = "_Report_Face"
KEY_OBJECTS = "_Report_Object"
MIDWAY_FLIP = "doFlipMidWay"
pROBE_YES = "_probeYes"
pROBE_NO = "_probeNo"
START_FLIPPED_STRATEGY = "startFlippedStrategy"
START_FLIPPED_FOR_ODDS = 3

# data table column names
ID_NAMES = ['Lab', 'Subject']
SCREEN_NAMES = ['Is_SCREEN', 'ScreeningID', 'SCREEN_probes', 'SCREEN_probes_face', 'SCREEN_probes_obj',
                'SCREEN_probes_blank', 'SCREEN_hitrate', 'SCREEN_farate']
GAME_NAMES = ['FullRunID', 'GAME_probes', 'GAME_probes_face', 'GAME_probes_obj', 'GAME_probes_blank', 'GAME_hits',
              'GAME_hitrate', 'GAME_misses', 'GAME_missrate', 'GAME_crrate', 'GAME_fas', 'GAME_farate', 'GAME_dprime',
              'GAME_hit_rt_mean', 'GAME_hit_rt_std', 'GAME_miss_rt_mean', 'GAME_miss_rt_std', 'GAME_fa_rt_mean', 'GAME_fa_rt_std', 'GAME_hit_difficulty', 'GAME_miss_difficulty',
              'GAME_hits_face', 'GAME_hitrate_face', 'GAME_misses_face', 'GAME_missrate_face', 'GAME_farate_face',
              'GAME_hits_obj', 'GAME_hitrate_obj', 'GAME_misses_obj', 'GAME_missrate_obj', 'GAME_farate_obj', 'GAME_farate_blank',
              'GAME_hit_face_rt_mean', 'GAME_hit_face_rt_std', 'GAME_miss_face_rt_mean', 'GAME_miss_face_rt_std',
              'GAME_hit_obj_rt_mean', 'GAME_hit_obj_rt_std', 'GAME_miss_obj_rt_mean', 'GAME_miss_obj_rt_std'] + ['GAME_' + n for n in LOCATION_NAMES]
PREP_LEVEL_NAMES = ['PREP_probes', 'PREP_probes_face', 'PREP_probes_obj', 'PREP_probes_blank', 'PREP_hits',
                    'PREP_hitrate', 'PREP_misses', 'PREP_missrate', 'PREP_crrate', 'PREP_farate', 'PREP_dprime']
FACE_OBJ_REPLAY_LIST = ['LOC_face_hits', 'LOC_face_hit_rate', 'LOC_face_miss', 'LOC_face_miss_rate', 'LOC_face_farate',
                        'LOC_obj_hits', 'LOC_obj_hits_rate', 'LOC_obj_miss', 'LOC_obj_miss_rate', 'LOC_obj_farate', 'LOC_blank_farate']
REPLAY_NAMES = ['LOC_total', 'LOC_fillers', 'LOC_stim(F/O/B)', 'LOC_target_stim(F/O)', 'LOC_nontarget_stim(F/O)',
                'LOC_nontarget_total', 'LOC_hits', 'LOC_hitrate', 'LOC_hit_rt_mean', 'LOC_hit_rt_std',
                'LOC_fas', 'LOC_fas_nofillers', 'LOC_farate_wfilllers', 'LOC_farate_nofillers_wblanks',
                'LOC_farate_onlyNonTargetStim', 'LOC_fa_rt_mean', 'LOC_fa_rt_std',
                'LOC_face_w_fillers_farate', 'LOC_face_w_fillers_fa_count', 'LOC_obj_w_fillers_farate', 'LOC_obj_w_fillers_fa_count',
                'LOC_dprime'] + FACE_OBJ_REPLAY_LIST + ['LOC_' + n for n in LOCATION_NAMES] + [f"LOC_{n}_wo_fillers_farate" for n in SPECIAL_LOCS] +\
               [f"LOC_{n}_wo_fillers_fa_count" for n in SPECIAL_LOCS] + ['LOC_afterWindowPresses', 'LOC_wrongButtonLevels', 'LOC_wrongTargetLevels']

NAMES = [data_reader.MODALITY] + ID_NAMES + SCREEN_NAMES + GAME_NAMES + PREP_LEVEL_NAMES + REPLAY_NAMES
EXTRA_ID_NAMES = [REAL_LEVEL, LEVEL]
EXTRA_REPLAY_NAMES = ['probes', 'probes_face_target', 'probes_obj_target', 'probes_non_target', 'hits', 'hitrate',
                       'misses', 'missrate', 'crrate', 'farate', 'farate_noblanks', 'hitrate_face', 'missrate_face', 'hitrate_obj',
                      'missrate_obj']

EXTRA_NAMES = ID_NAMES + EXTRA_ID_NAMES + EXTRA_REPLAY_NAMES
REPLAY_INFO_NAMES = ID_NAMES + EXTRA_REPLAY_NAMES

# names for saving
FILENAME = "quality_checks.csv"
EXTRA_FILENAME_REPLAY = "exp2_replay_extra_info.csv"

LAB_HUE_DICT = {"SA": "orange", "SB": "tomato", "SC": "darkblue", "SD": "mediumpurple", "SE": "teal",
                "SF": "mediumturquoise", "SG": "lightgreen", "SZ": "tab:Blue"}
LAB_NAME_DICT = {"SA": "Birmingham", "SB": "PKU", "SC": "Donders", "SD": "Yale", "SE": "Harvard", "SF": "NYU",
                 "SG": "Madison", "SZ": "TAU"}
LAB_NAME_HUE_DICT = {"Birmingham": "orange", "PKU": "tomato", "Donders": "darkblue", "Yale": "mediumpurple",
                     "Harvard": "teal", "NYU": "mediumturquoise", "Madison": "lightgreen", "TAU": "tab:Blue"}
#CHOSEN_10 = ["SA111", "SA148", "SB040", "SB069", "SB081", "SC109", "SC143", "SC160", "SD107", "SD165"]


def get_chosen_ten():
    mod_chosen_list = list()
    for modality in [data_reader.MEG, data_reader.FMRI]:
        mod_qc_folder = data_saver.create_derivative_qc_hpc(modality, root_folder=data_reader.COGITATE_PATH)
        mod_chosen_10 = pd.read_csv(os.path.join(mod_qc_folder, "ses-v2-optimization-subs.csv"))
        mod_chosen_list.extend(list(mod_chosen_10["sub_code"]))
    return mod_chosen_list


CHOSEN_10 = get_chosen_ten()
INDEPENDENT = "Independent"


def probe_data_go_nogo(probe_data, replay_targets):
    stim_present_list = list()
    stim_absent_list = list()
    for ind, row in replay_targets.iterrows():
        active_level = row[LEVEL_ID]
        active_target = row[TARGET_TYPE]
        present_level_data = probe_data[(probe_data[LEVEL] == active_level) & (probe_data[STIM_TYPE] == active_target)]
        stim_present_list.append(present_level_data)
        """
        Stimulus absent in REPLAY (localizer) levels:
        - 1: stimulus from the wrong category (stimulusID=int, stimulusType="Face"/"Object")
        - 2: blank stimulus                   (stimulusID=int, stimulusType="None")
        - 3: filler (non-stimulus)            (stimulusID=nan, stimulusType=nan)
        """
        absent_level_data = probe_data[(probe_data[LEVEL] == active_level) & (probe_data[STIM_TYPE] != active_target)]
        stim_absent_list.append(absent_level_data)
    stim_present = pd.concat(stim_present_list)
    stim_absent = pd.concat(stim_absent_list)
    # these lines are to make sure we did not miss anything and that stim_present + stim_absent include all probe_data
    processed_df = pd.concat([stim_present, stim_absent])
    difference_df = probe_data.merge(processed_df, indicator=True, how='left').loc[lambda x: x['_merge'] != 'both']
    if not difference_df.empty:
        print('probe_data_go_nogo: stim_present + stim_absent != probe_data')
    non_target_stim_only = stim_absent[(stim_absent[STIM_TYPE].notnull()) & (stim_absent[STIM_TYPE] != NON)]  # these are all the non-target stimuli, w/o blanks
    return stim_present, stim_absent, non_target_stim_only


def sess_probe_data(probe_data, sess_name, screen, replay, replay_targets=None, prep=False):
    """
    The actual analysis function, that received the relevant dataframes for a single session, and the information about
    what is the type of this data (dAT, AT, SCREEN). Based on that the function extracts the relevant statistics,
    and analyzes the data based on the relevant thresholds from quality_checks_criteria.
    :param probe_data: the dataframe containing the subject's answers to probes
    :param sess_name: name of the session
    :param screen: whether it's a behavioral screening session (True) or a full game experimental data (False)
    :param replay: whether it's a replay (AT) information (True) or a game (dAT) information (False).
    :return:
    """
    if replay:
        stim_present, stim_absent, stim_nontarget = probe_data_go_nogo(probe_data, replay_targets)
    if not replay:
        stim_present = probe_data[~probe_data[STIM_TYPE].str.contains(NON)]
        stim_absent = probe_data[probe_data[STIM_TYPE].str.contains(NON)]

    # probe analysis
    num_of_probes = int(probe_data.shape[0])  # total number of all probes
    num_of_present = int(stim_present.shape[0])
    num_of_absent = int(stim_absent.shape[0])

    if replay:
        num_of_stim = int(probe_data[probe_data[TYPE_] == TYPE_STIM_LOC].shape[0])
        num_of_fillers = int(probe_data[probe_data[TYPE_] != TYPE_STIM_LOC].shape[0])
        num_of_nontarget = int(stim_nontarget.shape[0])

    loc_present = dict()
    for loc in LOCATIONS:
        num_of_present_in_loc = stim_present[stim_present[STIM_LOC] == loc]
        loc_present[loc] = num_of_present_in_loc

    for loc in SPECIAL_LOCS:
        num_of_present_in_loc = stim_present[stim_present[STIM_LOC].str.contains(loc)]
        loc_present[loc] = num_of_present_in_loc

    # HITS AND MISSES : stimulus is there, subject sees it (TruePositive) or not (FalseNegative)
    hits = stim_present[stim_present[EVAL].str.contains(TRUEPOSITIVE)]
    misses = stim_present[stim_present[EVAL].str.contains(FALSENEGATIVE)]
    # FAs AND CRs : stimulus is not there, subject either saw nothing (TrueNegative) or thought they saw something (
    # FalsePositive)
    crs = stim_absent[(stim_absent[EVAL].notnull()) & (stim_absent[EVAL].str.contains(TRUENEGATIVE))]
    fas = stim_absent[(stim_absent[EVAL].notnull()) & (stim_absent[EVAL].str.contains(FALSEPOSITIVE))]
    if replay:
        fas_nontarget = stim_nontarget[stim_nontarget[EVAL].str.contains(FALSEPOSITIVE)]
    # calculate rates
    hit_rate = hits.shape[0] / num_of_present if num_of_present != 0 else 0
    miss_rate = misses.shape[0] / num_of_present if num_of_present != 0 else 0
    cr_rate = crs.shape[0] / num_of_absent if num_of_absent != 0 else 0
    fa_rate = fas.shape[0] / num_of_absent if num_of_absent != 0 else 0
    crs_no_filler = stim_absent[(stim_absent[EVAL].notnull()) & (stim_absent[EVAL].str.contains(TRUENEGATIVE)) & (~stim_absent[STIM_LOC].isna())]
    fas_no_filler = stim_absent[(stim_absent[EVAL].notnull()) & (stim_absent[EVAL].str.contains(FALSEPOSITIVE)) & (~stim_absent[STIM_LOC].isna())]

    if replay:
        fa_nontarget_rate = fas_nontarget.shape[0] / num_of_nontarget if num_of_nontarget != 0 else 0
        stim_absent_no_fillers = stim_absent[~stim_absent[STIM_LOC].isna()]
        fa_wblanks_nfillers = fas_no_filler.shape[0] / stim_absent_no_fillers.shape[0] if stim_absent_no_fillers.shape[0] != 0 else 0

    # calculate reaction times
    hit_rt_mean = hits["responseDT"].astype(float).mean()
    hit_rt_std = hits["responseDT"].astype(float).std()
    miss_rt_mean = misses["responseDT"].astype(float).mean()
    miss_rt_std = misses["responseDT"].astype(float).std()
    fa_rt_mean = fas["responseDT"].astype(float).mean()
    fa_rt_std = fas["responseDT"].astype(float).std()
    hit_difficulty = hits["difficulty"].astype(float).mean()
    miss_difficulty = misses["difficulty"].astype(float).mean()

    # calculate d prime
    if replay:
        # In replay, dprime shouldnt be with fillers
        dprime = stats.SDT(hits=hits.shape[0], misses=misses.shape[0], fas=fas_no_filler.shape[0], crs=crs_no_filler.shape[0])['d'].iloc[0]
    else:
        dprime = stats.SDT(hits=hits.shape[0], misses=misses.shape[0], fas=fas.shape[0], crs=crs.shape[0])['d'].iloc[0]

    # face/object breakdown
    if replay:
        # for replay, calculate FAs WITH fillers
        face_absent_w_fillers = stim_absent[stim_absent[TARGET_TYPE] == FACE]  # W fillers
        fas_face = face_absent_w_fillers[(face_absent_w_fillers[EVAL].notnull()) & (face_absent_w_fillers[EVAL].str.contains(FALSEPOSITIVE))]
        num_of_face_absent = int(face_absent_w_fillers.shape[0])
        face_w_fillers_fa_rate = fas_face.shape[0] / num_of_face_absent if num_of_face_absent != 0 else 0
        face_w_fillers_fa_count = fas_face.shape[0]

        object_absent_w_fillers = stim_absent[stim_absent[TARGET_TYPE] == OBJ]  # W fillers
        num_of_object_absent = int(object_absent_w_fillers.shape[0])
        fas_object = object_absent_w_fillers[(object_absent_w_fillers[EVAL].notnull()) & (object_absent_w_fillers[EVAL].str.contains(FALSEPOSITIVE))]
        obj_w_fillers_fa_rate = fas_object.shape[0] / num_of_object_absent if num_of_object_absent != 0 else 0
        object_w_fillers_fa_count = fas_object.shape[0]

        """
        We also want to analyze the FAr in the replay per location. As fillers do not have a location (4 fillers 
        at all locations), we filter them out and are left only with non-target stimuli (f/o) and blank stimuli.
        """
        special_loc_farate=[]
        special_loc_facount = []
        for loc in SPECIAL_LOCS:
            stim_absent_no_fillers = stim_absent[~stim_absent[STIM_LOC].isna()]
            absent_loc = stim_absent_no_fillers[stim_absent_no_fillers[STIM_LOC].str.contains(loc)]  # W/O fillers
            fas_loc = absent_loc[(absent_loc[EVAL].notnull()) & (absent_loc[EVAL].str.contains(FALSEPOSITIVE))]
            num_of_loc_absent = int(absent_loc.shape[0])
            loc_fa_rate = fas_loc.shape[0] / num_of_loc_absent if num_of_loc_absent != 0 else 0
            loc_fa_count = fas_loc.shape[0]
            special_loc_farate.append(loc_fa_rate)
            special_loc_facount.append(loc_fa_count)

    face_present = stim_present[stim_present[STIM_TYPE] == FACE]
    face_absent = stim_absent[stim_absent[STIM_TYPE] == FACE] # W/O fillers
    num_of_face_absent = int(face_absent.shape[0])
    fas_face = face_absent[(face_absent[EVAL].notnull()) & (face_absent[EVAL].str.contains(FALSEPOSITIVE))]
    blank_absent = stim_absent[(stim_absent[STIM_TYPE] != FACE) & (stim_absent[STIM_TYPE] != OBJ) & (~stim_absent[STIM_TYPE].isna())] # W/O fillers
    num_of_blank_absent = int(blank_absent.shape[0])
    fas_blank = blank_absent[(blank_absent[EVAL].notnull()) & (blank_absent[EVAL].str.contains(FALSEPOSITIVE))]
    object_present = stim_present[stim_present[STIM_TYPE] == OBJ]
    object_absent = stim_absent[stim_absent[STIM_TYPE] == OBJ] # W/O fillers
    num_of_object_absent = int(object_absent.shape[0])
    fas_object = object_absent[(object_absent[EVAL].notnull()) & (object_absent[EVAL].str.contains(FALSEPOSITIVE))]
    # HITS AND MISSES PER TYPE
    face_hits = face_present[face_present[EVAL] == TRUEPOSITIVE]
    face_misses = face_present[face_present[EVAL] == FALSENEGATIVE]
    obj_hits = object_present[object_present[EVAL] == TRUEPOSITIVE]
    obj_misses = object_present[object_present[EVAL] == FALSENEGATIVE]
    # calculate rates
    face_hit_rate = face_hits.shape[0] / face_present.shape[0] if face_present.shape[0] != 0 else 0
    face_miss_rate = face_misses.shape[0] / face_present.shape[0] if face_present.shape[0] != 0 else 0
    face_fa_rate = fas_face.shape[0] / num_of_face_absent if num_of_face_absent != 0 else 0
    obj_hit_rate = obj_hits.shape[0] / object_present.shape[0] if object_present.shape[0] != 0 else 0
    obj_miss_rate = obj_misses.shape[0] / object_present.shape[0] if object_present.shape[0] != 0 else 0
    obj_fa_rate = fas_object.shape[0] / num_of_object_absent if num_of_object_absent != 0 else 0
    blank_fa_rate = fas_blank.shape[0] / num_of_blank_absent if num_of_blank_absent != 0 else 0
    # calculate reaction times
    hit_face_rt_mean = face_hits["responseDT"].astype(float).mean()
    hit_face_rt_std = face_hits["responseDT"].astype(float).std()
    miss_face_rt_mean = face_misses["responseDT"].astype(float).mean()
    miss_face_rt_std = face_misses["responseDT"].astype(float).std()
    hit_obj_rt_mean = obj_hits["responseDT"].astype(float).mean()
    hit_obj_rt_std = obj_hits["responseDT"].astype(float).std()
    miss_obj_rt_mean = obj_misses["responseDT"].astype(float).mean()
    miss_obj_rt_std = obj_misses["responseDT"].astype(float).std()

    # location breakdown
    # HITS AND MISSES PER LOC
    loc_hit_dict = dict()
    loc_hitrate_dict = dict()
    for loc in (LOCATIONS + SPECIAL_LOCS):
        present = loc_present[loc]
        loc_hits = present[present[EVAL] == TRUEPOSITIVE]
        loc_hit_rate = loc_hits.shape[0] / present.shape[0] if present.shape[0] != 0 else 0
        loc_hit_dict[loc] = loc_hits.shape[0]
        loc_hitrate_dict[loc] = loc_hit_rate
    loc_miss_dict = dict()
    loc_missrate_dict = dict()
    for loc in (LOCATIONS + SPECIAL_LOCS):
        present = loc_present[loc]
        loc_miss = present[present[EVAL] == FALSENEGATIVE]
        loc_miss_rate = loc_miss.shape[0] / present.shape[0] if present.shape[0] != 0 else 0
        loc_miss_dict[loc] = loc_miss.shape[0]
        loc_missrate_dict[loc] = loc_miss_rate

    loc_list = list()
    for loc in (LOCATIONS + SPECIAL_LOCS):
        loc_stats = [loc_hit_dict[loc], loc_hitrate_dict[loc], loc_miss_dict[loc], loc_missrate_dict[loc]]
        loc_list.extend(loc_stats)

    type_list = [face_hits.shape[0], face_hit_rate, face_misses.shape[0], face_miss_rate, face_fa_rate,
                 obj_hits.shape[0], obj_hit_rate, obj_misses.shape[0], obj_miss_rate, obj_fa_rate, blank_fa_rate]

    # if full game, we want more types of analyses. For pre-screening we only need %hits and %false alarms
    if screen:
        probe_analysis_list = [screen, sess_name, num_of_probes, int(face_present.shape[0]),
                               int(object_present.shape[0]), num_of_absent, hit_rate, fa_rate]
    elif replay:
        probe_analysis_list = [num_of_probes, num_of_fillers, num_of_stim, num_of_present, num_of_nontarget,
                               num_of_absent, int(hits.shape[0]), hit_rate, hit_rt_mean, hit_rt_std, int(fas.shape[0]), int(fas_no_filler.shape[0]), fa_rate, fa_wblanks_nfillers,
                               fa_nontarget_rate, fa_rt_mean, fa_rt_std, face_w_fillers_fa_rate, face_w_fillers_fa_count, obj_w_fillers_fa_rate,
                               object_w_fillers_fa_count, dprime] + type_list + loc_list + special_loc_farate + special_loc_facount
    if not screen and not replay:
        if prep:
            probe_analysis_list = [num_of_probes, int(face_present.shape[0]), int(object_present.shape[0]),
                                   num_of_absent, int(hits.shape[0]), hit_rate, int(misses.shape[0]), miss_rate,
                                   cr_rate, fa_rate, dprime]
        else:
            probe_analysis_list = [sess_name, num_of_probes, int(face_present.shape[0]), int(object_present.shape[0]),
                                   num_of_absent, int(hits.shape[0]), hit_rate, int(misses.shape[0]), miss_rate,
                                   cr_rate, int(fas.shape[0]), fa_rate, dprime, hit_rt_mean, hit_rt_std, miss_rt_mean, miss_rt_std,
                                   fa_rt_mean, fa_rt_std,
                                   hit_difficulty, miss_difficulty, int(face_hits.shape[0]), face_hit_rate,
                                   int(face_misses.shape[0]), face_miss_rate, face_fa_rate, int(obj_hits.shape[0]), obj_hit_rate,
                                   int(obj_misses.shape[0]), obj_miss_rate, obj_fa_rate, blank_fa_rate,
                                   hit_face_rt_mean, hit_face_rt_std, miss_face_rt_mean, miss_face_rt_std,
                                   hit_obj_rt_mean, hit_obj_rt_std, miss_obj_rt_mean, miss_obj_rt_std] + loc_list
    return probe_analysis_list


def remove_aborted_levels(probe_data):
    """
    *** DEPRECATED: DMT DECISION 2022-10 - KEEP TRIALS OF PARTIAL RUNS AND LEVELS - THIS CODE SHOULD NOT BE CALLED ***
    Sometimes due to an issue while running the VG, a level was aborted and re-started. This method drops the aborted
    instance of a level, such that only the completed level is counted in following calculationg.
    :param probe_data: df of the game's probe data
    :return: the same df, w/o the dropped probes from the aborted level
    """
    level_list = probe_data[REAL_LEVEL].unique().tolist()
    for level in level_list:
        if level != 0:  # not practice
            level_data = probe_data[probe_data[REAL_LEVEL] == level]
            if level_data.shape[0] > LEVEL_MAX_PROBES:  # this level was aborted and restarted
                first_probe_in_level = level_data[level_data[DT_SINCE_LAST].isna()]
                if first_probe_in_level.shape[0] > 1:  # Yup, this level was played more than once (aborted and restarted)
                    if first_probe_in_level.shape[0] == 2:  # this only happened once
                        print(f"Level {level} was aborted and restarted. Removing aborted part")
                        indices = first_probe_in_level.index.tolist()
                        indice_range = [x for x in range(indices[0], indices[1])]
                        probe_data = probe_data.drop(indice_range)  # drop the aborted level
                    else:
                        c = 3  # stop in case a level was aborted and restarted more than once
    return probe_data


def sess_probe_data_game(sess: data_reader.Session, screen):
    """
    Analyze the behavioral data (answers to probes) during game (dAT), thresholds for exclusion depend on whether
    this is a behavioral screening game or a game-part of the full experimental session.
    :param sess: instance of class data_reader.Session, a single session belonging to a single subject
    :param screen: whether this session is a screening session or not. If False, this is a full-game session.
    :return:
    """
    probe_data = sess.SessDetails.ProbeDet  # probe data of this session
    # delete ABORTED levels from the session's probe data (2022-09-29: SD101, SD193) : DEPRECATED!
    #probe_data_full = remove_aborted_levels(probe_data)
    if -1 in probe_data['activeWorldID'].unique():
        probe_data = probe_data[probe_data['activeWorldID'] != -1]  # exclude practice levels from analysis
    result = sess_probe_data(probe_data, sess.name, screen, replay=False)

    if screen is False:
        prep_data_in_full = probe_data[probe_data['activeWorldID'] == -1]  # practice level in the full session
        prep_res = sess_probe_data(prep_data_in_full, sess.name, screen, replay=False, prep=True)
        result = result + prep_res

    return result


def replay_probe_data_df(localizer_data, replay_targets, sub_id):
    # remove the OutOfWindowPress lines, as they are no longer needed (see sess_probe_data_replay)
    localizer_data = localizer_data[localizer_data[EVAL] != IRREL]
    # calculate replay level probe data information
    data_levels = split_to_levels(localizer_data)
    level_data_list = list()
    localizer_targets = replay_targets
    for level in data_levels:
        level_data = calc_probe_data(probe_data=data_levels[level], probe_targets=localizer_targets)
        full_level_data = sub_id + level_data
        level_data_list.append(full_level_data)
    stat_df = pd.DataFrame(level_data_list, columns=EXTRA_NAMES)
    return stat_df


def after_window_presses_classifier(localizer_data_with_fillers):
    """
        Logic: take AfterWindowPresses into account: The response window is from 300ms after stimulus onset to 1300ms.
        Any response after that and before the next stimulus onset is an AfterWindowPress (response column; in the
        responseEvaluation column it is an OutOfWindow). In cases 1/4, we want to incorporate AfterWindowPresses,
        dividing them to 2 types:
        - Type 1: an AfterWindowPress that was made before the first filler appeared. In this case, we assume subjects
        responded to the stimulus but were too slow, so if the responseEvaluation to that stimulus is "_Negative" we
        will flip it to "_Positive" (either true or false).
        - Type 2: an AfterWindowPress that was made AFTER the first filler appeared. In this case, we assume the
        subjects responded to the filler, which will add a FalsePositive response to the filler.
    """
    after_window_presses = 0
    localizer_data = localizer_data_with_fillers
    after_window_data = localizer_data[localizer_data[RESP_COL] == AFTERW]
    just_stimuli = localizer_data[localizer_data[TYPE_] == TYPE_STIM_LOC]
    just_fillers = localizer_data[(localizer_data[TYPE_] != TYPE_STIM_LOC) & (localizer_data[RESP_COL].isnull())]
    if not after_window_data.empty:  # if there are any afterWindow presses
        after_window_presses = after_window_data.shape[0]
        level_begin_with_afterwindow = False
        for ind, row in after_window_data.iterrows():
            if ind == 0:  # then there was some sort of bug in the beginning of the level, making it -start- with an afterWindow press
                level_begin_with_afterwindow = True
                continue
            if ind == 1 and level_begin_with_afterwindow:
                continue
            prev_stim_time = row[STIM_ONSET_TS]  # the stimulus we refer to
            resp_time = row[RESP_TS]  # the afterWindow response
            # find the next stimulus onset, to get a range of fillers between the two
            curr_stim_ind = just_stimuli.index[just_stimuli[STIM_ONSET_TS] == prev_stim_time][0]
            next_stim_ind = just_stimuli.index.tolist().index(curr_stim_ind) + 1
            next_stim_row = just_stimuli.iloc[next_stim_ind]
            next_stim_time = next_stim_row[STIM_ONSET_TS]  # next stimulus onset
            # find all fillers in that range (between current and next stimulus onsets)
            fillers = just_fillers[
                (just_fillers[STIM_ONSET_TS] >= prev_stim_time) & (just_fillers[STIM_ONSET_TS] <= next_stim_time)]
            first_filler_time = fillers.iloc[0, fillers.columns.get_loc(STIM_ONSET_TS)]
            # now we have everything and ready to check the late response:
            if resp_time <= first_filler_time:  # it's interpreted as a late response to the STIMULUS
                localizer_data.at[curr_stim_ind, RESP_COL] = AFTERW
                localizer_data.at[curr_stim_ind, EVAL] = FLIP_DICT[localizer_data.loc[curr_stim_ind, EVAL]]
            else:  # it's interpreted as a response to the fillers, which is always wrongful
                filler_times = fillers[STIM_ONSET_TS].tolist()  # find out which filler
                for i in range(len(filler_times) - 1):
                    if filler_times[i] <= resp_time <= filler_times[i + 1]:  # filler i is the one the resp belongs to
                        filler_ind = fillers.index[fillers[STIM_ONSET_TS] == filler_times[i]][0]
                        if (localizer_data.loc[filler_ind, EVAL] != TRUEPOSITIVE) and (
                                localizer_data.loc[filler_ind, EVAL] != FALSEPOSITIVE):
                            localizer_data.at[filler_ind, EVAL] = FLIP_DICT[localizer_data.loc[filler_ind, EVAL]]
                if filler_times[-1] <= resp_time:  # maybe the response is after the last filler
                    filler_ind = fillers.index[fillers[STIM_ONSET_TS] == filler_times[-1]][0]
                    if (localizer_data.loc[filler_ind, EVAL] != TRUEPOSITIVE) and (
                            localizer_data.loc[filler_ind, EVAL] != FALSEPOSITIVE):
                        localizer_data.at[filler_ind, EVAL] = FLIP_DICT[localizer_data.loc[filler_ind, EVAL]]
    else:  # we are not ignoring AfterWindowPresses but there aren't any
        pass

    return localizer_data, after_window_presses


def calc_sub_num_wrong_presses(sess, zero_levels, buttonA, buttonB):
    """
    This method gets all replay levels, and a list of levels that were excluded BECAUSE they had WRONG_BUTTON_SUSPECT.
    In the replay there are two possible response buttons (1 response button for the first 4 levels, and another one
    for the last 4 levels, see "doFlipMidWay" in https://twcf-arc.slab.com/posts/the-config-file-everything-b1xt0jhf)
    Then, for each zero level, the method checks if the OTHER button was pressed, and if it was indeed pressed -
    whether it was pressed in response for that level's target.
    This was meant to understand if participants' "no response" levels are actually levels where they simply pressed
    the wrong button, but kept on doing the task.
    """
    midlevel = int(data_reader.REPLAY_LEVEL_ORDER[int(len(data_reader.REPLAY_LEVEL_ORDER)/2)])  # the level where the response button is flipped
    level_presses = list()  # all presses (on one of the two possible buttons during the level)
    correct_level_presses = list()  # all RELEVANT presses (presses WITHIN THE STIMULUS RESPONSE WINDOW, for the TARGET stimulus type)
    for level in zero_levels:
        if level >= midlevel:
            relevant_button = buttonA
        else:
            relevant_button = buttonB
        relevant_level_data = sess.ReplayRaw[str(level)]  # get the RAW replay logfile (heavy)
        # stimulus onset
        stim_start_rows = list(relevant_level_data[relevant_level_data[0].str.contains('SHOWING_STIMULUS') & relevant_level_data[0].str.contains('Stimulus Manager')].index)
        # stimulus RESPONSE WINDOW is over
        stim_end_rows = list(relevant_level_data[relevant_level_data[0].str.contains('ReplayWindowOver') & relevant_level_data[0].str.contains('TRIGGER_MANAGER_CENTRAL')].index)
        if len(stim_start_rows) != len(stim_end_rows):  # each onset must have a corresponding response window message
            print("Lengths of stimulus starts and end do not match")
            return
        replay_target = sess.replay_targets.iloc[level - 100, 1]  # what is the target stimulus of that level
        total_num_presses = 0
        total_correct_presses = 0
        for ind in range(len(stim_start_rows)):
            stim_start_ind = stim_start_rows[ind]
            stim_end_ind = stim_end_rows[ind]
            relevant_stim_data = relevant_level_data.iloc[stim_start_ind: stim_end_ind + 1]  # the response window for a given stimulus
            for button in relevant_button:
                if type(button) != str:
                    button = chr(button)
                    if button.isdigit():
                        button = f"Alpha{button}"  # RONY CHECK MAPPING ACROSS LABS
                stim_presses = relevant_stim_data[relevant_stim_data[0].str.contains('KEY_DOWN') & relevant_stim_data[0].str.contains(button)].shape[0]
                total_num_presses += stim_presses  # add a press
                if replay_target in relevant_stim_data.loc[stim_start_ind, 0]:
                    total_correct_presses += stim_presses  # this was a TARGET RELEVANT press
        level_presses.append(total_num_presses)
        correct_level_presses.append(total_correct_presses)
    return correct_level_presses, level_presses


def calculate_replay_responses(sess: data_reader.Session, sub_id, extra=False):
    """
    Given a session and subject id, calculate the behavioral data responses (answers to probes) during replay (AT).
    :param sub_id: subject id (code)
    :param sess: instance of class data_reader.Session, a single session belonging to a single subject.
    :param extra: whether or not to ouput a table named stat_df, which is a breakdown of responses level by level.
    :return:
    """
    # Counters for any manipulation on raw: afterWindow presses are counted below
    wrong_button_levels = 0
    wrong_target_levels = 0

    localizer_data = sess.SessDetails.LocalizerDetWithFillers  # probe data of replay levels are found in the localizer details

    # first, take out "additional presses": double-presses after the first press was already registered
    localizer_data = localizer_data[localizer_data[EVAL] != IRREL_ADDITION].reset_index(drop=True, inplace=False)

    """
    The active level ID (the game level being re-played in the localizer) should be selected once per replay level 
    (i.e., no game level repeates twice as a replay level). However, subject SB036 had so many interrupts (3!), that
    something went wrong with logging the active level ID. This is a correction made to deal with this subject. 
    """
    activeLevels = localizer_data.groupby([LEVEL])[REAL_LEVEL].nunique()
    has_more_than_one = any(x > 1 for x in activeLevels.values)
    if has_more_than_one:
        print(f"{sub_id} has bad replay targets, please check manually")
        if sub_id[0] != "SB" or sub_id[1] != "036":  # to understand if there are more like this - this didn't happen
            raise Exception("Need to look manually")
        # A MANUAL correction ONLY for SB036
        localizer_data.loc[localizer_data[REAL_LEVEL] == 100, LEVEL] = 1
        localizer_data.loc[localizer_data[REAL_LEVEL] == 101, LEVEL] = 2

    localizer_data, after_window_presses = after_window_presses_classifier(localizer_data)  # flip "afterwindow presses" depending on their time (see method)

    # now, calculate the per-level statistics of the replay
    stat_df = replay_probe_data_df(localizer_data, sess.replay_targets, sub_id)
    stat_df.loc[:, 'IS_LEVEL_ELIMINATED'] = 0  # below we will remove levels based on conditions; this column indicates per level if it was eliminated or not

    # create a copy of localizer data where nothing is removed - responses were flipped above (afterWindow), and removed levels are marked (not deleted)
    corrected_localizer_data = localizer_data.copy()
    corrected_localizer_data.loc[:, 'IS_LEVEL_ELIMINATED'] = 0

    """wrong_button: sub might have pressed the wrong button for that level by looking if a response was not made
    recognize by hit rate + false alarm rate combo indicator.
    - COUNT THE NUMBER OF SUCH LEVELS
    - *REMOVE* those levels from localizer data so they are not counted in ANY future replay calculation"""
    dkeys = list(WRONG_BUTTON_SUSPECT.keys())
    wb_df = stat_df[(stat_df[dkeys[0]] == WRONG_BUTTON_SUSPECT[dkeys[0]]) &
                    (stat_df[dkeys[1]] == WRONG_BUTTON_SUSPECT[dkeys[1]])]
    if not wb_df.empty:  # we have such levels, REMOVE THEM FROM QC so they won't count
        sess.ReplayRaw = sess.get_replay_raw(sess.path)
        levels = list(wb_df[REAL_LEVEL])
        wrong_button_levels = len(levels)
        localizer_data = localizer_data[~localizer_data[REAL_LEVEL].isin(levels)]  # REMOVE FROM DATA
        stat_df.loc[stat_df[REAL_LEVEL].isin(levels), 'IS_LEVEL_ELIMINATED'] = 1  # update the elimination in stat_df
        stat_df.loc[stat_df[REAL_LEVEL].isin(levels), 'ELIMINATION_REASON1'] = "WRONG_BUTTON"  # update the elimination in stat_df
        corrected_localizer_data.loc[corrected_localizer_data[REAL_LEVEL].isin(levels), 'IS_LEVEL_ELIMINATED'] = 1
        corrected_localizer_data.loc[corrected_localizer_data[REAL_LEVEL].isin(levels), 'ELIMINATION_REASON1'] = "WRONG_BUTTON"


        """
        The Config file (in the subject struct) contains information about the response button mappings in the VG. 
        In all Configs, you can find all options - therefore, we need to identify the set of response configurations
        that matches each lab. 
        NOTABLY, ALL LABS but Donders (SC) had a USB-port response box; Donders had a serial port one. 
        This is documented here: https://twcf-arc.slab.com/posts/video-game-lab-specific-input-connections-nmb0f65d
        And the institutional abbreviations are here: https://twcf-arc.slab.com/posts/institutional-abbreviations-rsi4obcd
        """

        """
        MIDWAY_FLIP is a config bool parameter where if true, then the button for response in the replay changes 
        after 4 levels (from whatever was set in CONFIG_USE_PROBE_FOR_REPLAY). 
        If it's false, then it's the same button throughout the entire replay
        This can be found here: https://twcf-arc.slab.com/posts/the-config-file-everything-b1xt0jhf
        including the site table and their values here
        """
        has_flip = sess.config[INPUT][MIDWAY_FLIP]
        if has_flip:
            """
            According to https://twcf-arc.slab.com/posts/video-game-lab-specific-input-connections-nmb0f65d, 
            SC participants (Donders) have their response input configured differently. 
            Therefore, we need to have differential codes for serial input (SC) and everything else.
            SC189 replay level 0 raw data was such that the Donders-specific configuration did not pick up responses
            that the other configuration did. Therefore the condition below is ALWAYS TRUE.
            """
            if sub_id[0] != "SC" or True:  # the right config response mapping is the usb port
                config_section = CONFIG_INPUT_KEY_CODE
                relevant_config = sess.config[config_section]
                """
                The CONFIG_USE_PROBE_FOR_REPLAY parameter is a boolean; if it's set on "true", then the game's "yes" button 
                is the report button in the replay (go-nogo). If it's no, then it's another button; therefore, we should make
                sure to get the right response buttons. 
                This is documented here: https://twcf-arc.slab.com/posts/the-config-file-everything-b1xt0jhf
                """
                is_probes = relevant_config[CONFIG_USE_PROBE_FOR_REPLAY]
                if is_probes:
                    buttonA = relevant_config[PROBE_YES][KEY_CODES]
                    buttonB = relevant_config[PROBE_NO][KEY_CODES]
                else:
                    buttonA = relevant_config[KEY_FACES][KEY_CODES]
                    buttonB = relevant_config[KEY_OBJECTS][KEY_CODES]

            else:  # the right config response mapping is the serial port
                config_section = CONFIG_INPUT_SERIAL_PORT
                relevant_config = sess.config[config_section]

                is_probes = relevant_config[CONFIG_USE_PROBE_FOR_REPLAY]
                if is_probes:
                    buttonA = relevant_config[pROBE_YES]["highToLow"]
                    buttonB = relevant_config[pROBE_NO]["highToLow"]
                else:
                    buttonA = relevant_config[KEY_FACES][KEY_CODES]
                    buttonB = relevant_config[KEY_OBJECTS][KEY_CODES]

            """
            Replay levels' response button is a single button, which either switches after 4 levels, or stays the same
            depending on startFlippedStrategy (https://twcf-arc.slab.com/posts/the-config-file-everything-b1xt0jhf). 
            IN COGITATE it's flipping after 4 levels, always. To counterbalance across subjects, the even subjects
            start with "ProbeYes" as the response button (and then it flips to "ProbeNo"), while odds do the opposite. 
            """
            flipped_strategy = sess.config[INPUT][START_FLIPPED_STRATEGY]
            if flipped_strategy != 3:  # midway flip, always
                raise Exception("Unexpected Flipped Strategy")
            should_flip = (int(sub_id[1]) % 2) != 0
            if should_flip:
                tempButton = buttonB
                buttonB = buttonA
                buttonA = tempButton

            correct_level_presses, level_presses = calc_sub_num_wrong_presses(sess, levels, buttonA, buttonB)
            stat_df.loc[stat_df[REAL_LEVEL].isin(levels), 'CORRECT_WRONG_PRESSES'] = correct_level_presses
            stat_df.loc[stat_df[REAL_LEVEL].isin(levels), 'TOTAL_WRONG_PRESSES'] = level_presses

    """wrong_instructions: sub might have pressed responding to the wrong target for that level
    recognize by false alarm rate + miss rate
    - COUNT THE NUMBER OF SUCH LEVELS
    - *REMOVE* those levels from localizer data so they are not counted in ANY future replay calculation"""
    dkeys = list(WRONG_TARGET_SUSPECT.keys())
    wb_df = stat_df[(stat_df[dkeys[0]] >= WRONG_TARGET_SUSPECT[dkeys[0]]) &
                    (stat_df[dkeys[1]] >= WRONG_TARGET_SUSPECT[dkeys[1]])]
    if not wb_df.empty:  # we have such levels, REMOVE THEM FROM QC so they won't count
        levels = list(wb_df[REAL_LEVEL])
        wrong_target_levels = len(levels)
        localizer_data = localizer_data[~localizer_data[REAL_LEVEL].isin(levels)]  # REMOVE FROM DATA
        stat_df.loc[stat_df[REAL_LEVEL].isin(levels), 'IS_LEVEL_ELIMINATED'] = 1  # update the elimination in stat_df
        stat_df.loc[stat_df[REAL_LEVEL].isin(levels), 'ELIMINATION_REASON2'] = "WRONG_TARGET"  # update the elimination in stat_df
        corrected_localizer_data.loc[corrected_localizer_data[REAL_LEVEL].isin(levels), 'IS_LEVEL_ELIMINATED'] = 1
        corrected_localizer_data.loc[corrected_localizer_data[REAL_LEVEL].isin(levels), 'ELIMINATION_REASON2'] = "WRONG_TARGET"

    # remove the OutOfWindowPress lines, as they are no longer needed
    localizer_data = localizer_data[localizer_data[EVAL] != IRREL]
    corrected_localizer_data = corrected_localizer_data[corrected_localizer_data[EVAL] != IRREL]

    if extra:
        return localizer_data, after_window_presses, wrong_button_levels, wrong_target_levels, stat_df, corrected_localizer_data
    else:
        return localizer_data, after_window_presses, wrong_button_levels, wrong_target_levels, corrected_localizer_data


def sess_probe_data_replay(sess: data_reader.Session, sub_id):
    if sess.SessDetails.LocalizerDet.empty:  # no replay levels were played at all
        result = [np.nan] * 39
        return result, pd.DataFrame()
    localizer_data, after_window_presses, wrong_button_levels, wrong_target_levels, stat_df, corrected_localizer_data = calculate_replay_responses(sess, sub_id, extra=True)
    # update localizer_data with a column denoting the target stimulus of that localizer level
    target_mapping = sess.replay_targets.set_index(data_reader.LOC_LEVEL_TITLE).T.to_dict('records')[0]
    localizer_data[TARGET_TYPE] = localizer_data[LEVEL].map(target_mapping)
    corrected_localizer_data[TARGET_TYPE] = corrected_localizer_data[LEVEL].map(target_mapping)
    # update LocalizerDetCalculated and LocalizerDetFullCorrected: in BOTH of them, AfterWindowPresses flipped the responses.
    sess.SessDetails.LocalizerDetCalculated = localizer_data  # THIS IS ONE WHERE ERRONEOUS LEVELS WERE *REMOVED*
    sess.SessDetails.LocalizerDetFullCorrected = corrected_localizer_data  # THIS IS ONE WHERE ERRONOUS LEVELS WERE *MARKED*
    # now get the target of the replay levels (go/no-go target)
    localizer_targets = sess.replay_targets
    result = sess_probe_data(localizer_data, sess.name, screen=False, replay=True, replay_targets=localizer_targets)
    # add the 3 columns about the additional manipulations on raw
    result.extend([after_window_presses, wrong_button_levels, wrong_target_levels])
    return result, stat_df


def split_to_levels(response_data):
    # remove the OutOfWindowPress lines, as they are no longer needed
    response_data = response_data[response_data[EVAL] != IRREL]
    all_levels = response_data[REAL_LEVEL].unique()
    levels_dict = {l: None for l in all_levels}
    for level in all_levels:
        levels_dict[level] = response_data[response_data[REAL_LEVEL] == level]
    return levels_dict


def calc_probe_data(probe_data, probe_targets):

    curr_level = list(probe_data[REAL_LEVEL])[0]
    active_level = list(probe_data[LEVEL])[0]

    stim_present, stim_absent, stim_nontarget = probe_data_go_nogo(probe_data, probe_targets)
    num_of_probes = int(probe_data.shape[0])  # total number of all probes
    num_of_present = int(stim_present.shape[0])
    num_of_absent = int(stim_absent.shape[0])
    num_of_nontarget = int(stim_nontarget.shape[0])

    # HITS AND MISSES : stimulus is there, subject sees it (TruePositive) or not (FalseNegative)
    hits = stim_present[stim_present[EVAL].str.contains(TRUEPOSITIVE)]
    misses = stim_present[stim_present[EVAL].str.contains(FALSENEGATIVE)]
    # FAs AND CRs : stimulus is not there, subject either saw nothing (TrueNegative) or thought they saw something (
    # FalsePositive)
    crs = stim_absent[(stim_absent[EVAL].notnull()) & (stim_absent[EVAL].str.contains(TRUENEGATIVE))]
    fas = stim_absent[(stim_absent[EVAL].notnull()) & (stim_absent[EVAL].str.contains(FALSEPOSITIVE))]
    fas_nontarget = stim_nontarget[stim_nontarget[EVAL].str.contains(FALSEPOSITIVE)]
    # calculate rates
    hit_rate = hits.shape[0] / num_of_present if num_of_present != 0 else 0
    miss_rate = misses.shape[0] / num_of_present if num_of_present != 0 else 0
    cr_rate = crs.shape[0] / num_of_absent if num_of_absent != 0 else 0
    fa_rate = fas.shape[0] / num_of_absent if num_of_absent != 0 else 0
    fa_nontarget_rate = fas_nontarget.shape[0] / num_of_nontarget if num_of_nontarget != 0 else 0

    # face/object breakdown
    face_present = stim_present[stim_present[STIM_TYPE] == FACE]
    object_present = stim_present[stim_present[STIM_TYPE] == OBJ]
    # HITS AND MISSES PER TYPE
    face_hits = face_present[face_present[EVAL] == TRUEPOSITIVE]
    face_misses = face_present[face_present[EVAL] == FALSENEGATIVE]
    obj_hits = object_present[object_present[EVAL] == TRUEPOSITIVE]
    obj_misses = object_present[object_present[EVAL] == FALSENEGATIVE]
    # calculate rates
    face_hit_rate = face_hits.shape[0] / face_present.shape[0] if face_present.shape[0] != 0 else 0
    face_miss_rate = face_misses.shape[0] / face_present.shape[0] if face_present.shape[0] != 0 else 0
    obj_hit_rate = obj_hits.shape[0] / object_present.shape[0] if object_present.shape[0] != 0 else 0
    obj_miss_rate = obj_misses.shape[0] / object_present.shape[0] if object_present.shape[0] != 0 else 0

    res_list = [curr_level, active_level, num_of_probes, int(face_present.shape[0]), int(object_present.shape[0]),
                num_of_absent, int(hits.shape[0]), hit_rate, int(misses.shape[0]), miss_rate, cr_rate, fa_rate,
                fa_nontarget_rate, face_hit_rate, face_miss_rate, obj_hit_rate, obj_miss_rate]
    return res_list


def extract_subject_data(sub: data_reader.Subject):
    """
    This function receives an instance of class Subject. It checks which session this subject has, and calls
    the relevant QC functions to check that data.
    For the replay condition it calles "sess_probe_data_replay". NOTE: this method:
    - flips "afterWindowPresses" based on the time in which they occurred (turns "negatives" into "positives")
    - counts how many wrong button levels there were
    - counts how many empty response levels there were
    :param sub: instance of Class data_reader.Subject
    :return: it returns a list containing all the data stats of this subject (to be concatenated to the final table)
    """
    data = [sub.mod, sub.lab, sub.id]
    screen = 0
    if hasattr(sub, 'prescreen'):
        screen = 1
        screen_data = sess_probe_data_game(sub.prescreen, screen=True)
        data.extend(screen_data)

    if hasattr(sub, 'full'):
        if screen == 0:
            data.extend([False])
            empty = [None] * (len(SCREEN_NAMES) - 1)
            data.extend(empty)
        full_data_game = sess_probe_data_game(sub.full, screen=False)
        data.extend(full_data_game)
        full_data_replay, sub_stat_df = sess_probe_data_replay(sub.full, sub_id=[sub.lab, sub.id])
        data.extend(full_data_replay)

    if not hasattr(sub, 'full'):
        game_cols = len(GAME_NAMES)
        data.extend([None for x in range(game_cols)])
    return data, sub_stat_df

def practice_trials_num(subs):
    sub_name_list = []
    sub_p_list = []
    for subject in subs:
        print(f"----- Subject {subject} -----")
        if not hasattr(subs[subject], 'full'):
            print(f"Subject {subject} has no full v2 session. Skipping!")
            continue
        stim_det = subs[subject].full.SessDetails.StimulusDet
        num_of_p_trias = stim_det[(stim_det[REAL_LEVEL] == 0) & (stim_det['activeWorldID'] == -1)].shape[0]
        sub_name_list.append(subject)
        sub_p_list.append(num_of_p_trias)
    x = 5

def extract_data(root_folder=data_reader.COGITATE_PATH):
    """
    Go over every folder within the data_path folder, and create a list of subjects to be extracted. After they are
    extracted, create the results table (the final table to be saved, containing all subjects).
    **NOTE**: all stats rely on subject behavior AFTER correcting for afterWindowPresses, and IGNORING wrong button /
    wrong target levels (see replay methods for more).
    :return:
    result_df = df where row=subject, cols=stats on subject behavior in the VG. THIS IS WHAT THE BEH QC IS CHECKING
    replay_level_stats = df specific to replay levels, with replay level breakdown.
    subjects = dict, where key=sub, val=a Subject instance where all the data of exp.2 sessions is.
    THIS IS WHAT WILL BE USED FOR ANALYSIS
    """
    # Check every subject folder to be valid, then create an instance of the class Subject for each valid subject,
    # containing all the session and info
    subjects = data_reader.data_reader_hpc(root_folder)  # create Subject struct for all subjects, and put in dict

    practice_trials_num(subjects)
    sub_data_list = []
    sub_replay_level_df_list = []
    for subject in subjects:
        print(f"----- Subject {subject} -----")
        if not hasattr(subjects[subject], 'full'):
            print(f"Subject {subject} has no full v2 session. Skipping!")
            continue
        sub_data, sub_stat_df = extract_subject_data(subjects[subject])  # from class Subject, extract the relevant data
        sub_data_list.append(sub_data)
        sub_replay_level_df_list.append(sub_stat_df)

    result_df = pd.DataFrame(sub_data_list, columns=NAMES)
    replay_level_stats = pd.concat(sub_replay_level_df_list)
    return result_df, replay_level_stats, subjects


""" DEPRECATED
def check_prescreen_game_corr(summary_table, save_path, save_prefix=""):
    successful_prescreening = summary_table[summary_table['SCREEN_OK?'] == True]  # only SUCCESSFUL prescreening
    corr_names = ["Hit Rate", "FA Rate"]
    test_cols = ["SCREEN_hitrate", "SCREEN_farate"]
    retest_cols = ["GAME_hitrate", "GAME_farate"]
    result = {"Hit Rate": None, "FA Rate": None}
    if not successful_prescreening.empty:
        for i in range(len(test_cols)):
            corr_name = corr_names[i]
            test = test_cols[i]
            retest = retest_cols[i]
            result[corr_name] = stats.corr_test(corr_name, successful_prescreening, test, retest, corr_method='pearson')
            if result[corr_name] is not None:
                result[corr_name] = result[corr_name].round(decimals=3)  # round numbers to 3 digits after decimal pt
                # corr data
                result[corr_name].to_csv(os.path.join(save_path, f"{save_prefix}{corr_name.replace(' ', '')}_corr_presrcreen_game.csv"))
                successful_prescreening.loc[:, 'Lab'] = successful_prescreening.replace(LAB_NAME_DICT)
                lineplotter.plot_scatter(title=corr_name + " Correlation Between Prescreening and Game (dAT)",
                                         data=successful_prescreening, x_col=test, y_col=retest,
                                         hue='Lab', hue_order=list(LAB_NAME_HUE_DICT.keys()),
                                         palette=list(LAB_NAME_HUE_DICT.values()), line=True,
                                         x_name="Successful " + test.lower().replace("_", " ").title(),
                                         y_name=retest.lower().replace("_", " ").title(), save=True,
                                         save_name=f"{save_prefix}{corr_name}_corr",
                                         save_path=save_path)
    return result

"""


def create_optimization_table(mod_dict):
    analysis_col_names = ["DECODING", "ACTIVATION", "SYNCHRONY", "BASELINE"]
    sub_identifier_col_names = ["sub_code", "modality", "Lab"]
    optimization_dict = dict()

    for modal in list(mod_dict.keys()):
        df = mod_dict[modal]
        analysis_cols = [col for col in df.columns if col.startswith(tuple(analysis_col_names)) and "RANK" not in col]
        df.reset_index(drop=False, inplace=True)
        optimization_df = df.loc[:, sub_identifier_col_names]
        optimization_df.loc[:, analysis_cols] = False
        for analysis in analysis_cols:
            df_subthresh = df[df[analysis] < quality_checks_criteria.ANALYSIS_THRESHOLD]  # not enough trials to be included in the analysis
            df_topN = df_subthresh.nlargest(quality_checks_criteria.OPTIMIZATION_AMT, analysis)  # these are the TOP N out of those who don't have enough trials for analysis
            # now add "True" to this analysis column for each of the top N, to include them in the optimization phase
            for ind, row in df_topN.iterrows():
                optimization_df.loc[optimization_df["sub_code"] == row["sub_code"], analysis] = True

        # now, if a subject doesn't have any "True" in any column - they DO NOT BELONG to the optimization phase.
        mask = optimization_df[analysis_cols].ne(0).any(axis=1)  # "1" only if has at least 1 True
        result_df = optimization_df.loc[mask]

        # save to modality's folder
        mod_qc_folder = data_saver.create_derivative_qc_hpc(modal, root_folder=data_reader.COGITATE_PATH)
        result_df.to_csv(os.path.join(mod_qc_folder, f"ses-v2-optimization-subs-raw.csv"), index=False)

        """
        2023-07-14 
        The following update was added once a DMT decision was reached about having THE SAME 5 PARTICIPANTS used for
        optimization of ALL ANALYSES. Therefore, we will now select the "top 5" participants in the optimization table:
        those who are "TRUE" in the highest number of columns. Then, to avoid confusion for the teams - we will 
        LEAVE JUST THE SELECTED 5 in the optimization file. 
        """

        # add a column to result_df which counts the number of TRUE cells in each row
        result_df["sum_true"] = result_df.iloc[:, 3:].sum(axis=1)  # the first 3 columns are ID columns
        # sort the df based on the summing column, in descending order
        new_result_df = result_df.sort_values("sum_true", ascending=False, inplace=False)
        five_result_df = new_result_df.head(5)
        optimization_ids = five_result_df["sub_code"].tolist()

        # save to modality's folder
        mod_qc_folder = data_saver.create_derivative_qc_hpc(modal, root_folder=data_reader.COGITATE_PATH)
        five_result_df.to_csv(os.path.join(mod_qc_folder, f"ses-v2-optimization-subs.csv"), index=False)
        optimization_dict[modal] = optimization_ids

    return optimization_dict


def create_analysis_table(mod_dict, optimization_lists):
    analysis_col_names = ["DECODING", "ACTIVATION", "SYNCHRONY", "BASELINE"]
    sub_identifier_col_names = ["sub_code", "modality", "Lab"]

    for modal in list(mod_dict.keys()):
        df = mod_dict[modal]
        analysis_cols = [col for col in df.columns if col.startswith(tuple(analysis_col_names)) and "RANK" not in col]
        df.reset_index(drop=False, inplace=True)
        analysis_df = df.loc[:, sub_identifier_col_names]
        analysis_df.loc[:, analysis_cols] = False
        for analysis in analysis_cols:
            df_subthresh = df[df[analysis] >= quality_checks_criteria.ANALYSIS_THRESHOLD]  # ENOUGH trials to be included in the analysis
            # now add "True" to this analysis column for each of the top N, to include them in the optimization phase
            for ind, row in df_subthresh.iterrows():
                analysis_df.loc[analysis_df["sub_code"] == row["sub_code"], analysis] = True

        # now, if a subject doesn't have any "True" in any column - they DO NOT BELONG to the analysis phase.
        mask = analysis_df[analysis_cols].ne(0).any(axis=1)  # "1" only if has at least 1 True
        result_df = analysis_df.loc[mask]

        # save a table with all the valid subjects that do not have enough trials for ANY of the analyses
        mod_qc_folder = data_saver.create_derivative_qc_hpc(modal, root_folder=data_reader.COGITATE_PATH)
        residual_df = analysis_df.loc[~mask]
        residual_df.to_csv(os.path.join(mod_qc_folder, f"ses-v2-residual-subs.csv"), index=False)

        """
        2023-07-14
        Now (yes, AFTER the residual file is created, to make sure no subject is "lost"), NULLIFY THE OPTIMIZATION 
        SUBJECTS received as input
        """
        optimization_list = optimization_lists[modal]  # the modality-specific optimization list
        for ind, row in result_df.iterrows():
            if row["sub_code"] in optimization_list:  # if this subject is in the OPTIMIZATION list, it CANNOT be analyzed
                result_df.loc[ind, result_df.columns[3:]] = False

        # save to modality's folder
        result_df.to_csv(os.path.join(mod_qc_folder, f"ses-v2-analysis-subs.csv"), index=False)

    return


def create_subject_lists(qc_table, save_path):
    """
    For each of the pre-registered analyses, participants who already passed the criteria for valid game and replay
    will be sorted. As each pre-registered analysis includes different comparisons, participants are sorted based on
    the number of trials they have. This method prepares the relevant data columns and sends the MEG and fMRI tables
    separately to the quality checks module, to rate the subjects
    :param qc_table: the talbe that is a result of the behavioral QC
    :param save_path: path to save the data to
    """
    # ---------- STEP 1: filter out invalid subjects ----------
    valid_qc_table = qc_table[qc_table[quality_checks_criteria.VALID] == True]
    print(f"{valid_qc_table.shape[0]} out of {qc_table.shape[0]} are valid for analysis: ")
    for lab in valid_qc_table[quality_checks_criteria.LAB].unique():
        print(f"{lab}: {valid_qc_table[valid_qc_table[quality_checks_criteria.LAB] == lab].shape[0]} out of {qc_table[qc_table[quality_checks_criteria.LAB] == lab].shape[0]}")

    # ---------- STEP 2: create location columns (L/R) for future sorting and filtering ----------
    # dAT (game) hits
    valid_qc_table["GAME_hits_left"] = valid_qc_table["GAME_hits_TopLeft"] + valid_qc_table["GAME_hits_BottomLeft"]
    valid_qc_table["GAME_hits_right"] = valid_qc_table["GAME_hits_TopRight"] + valid_qc_table["GAME_hits_BottomRight"]
    # dAT misses
    valid_qc_table["GAME_misses_left"] = valid_qc_table["GAME_misses_TopLeft"] + valid_qc_table["GAME_misses_BottomLeft"]
    valid_qc_table["GAME_misses_right"] = valid_qc_table["GAME_misses_TopRight"] + valid_qc_table["GAME_misses_BottomRight"]

    # AT (localizer, replay) hits
    valid_qc_table["LOC_hits_left"] = valid_qc_table["LOC_hits_TopLeft"] + valid_qc_table["LOC_hits_BottomLeft"]
    valid_qc_table["LOC_hits_right"] = valid_qc_table["LOC_hits_TopRight"] + valid_qc_table["LOC_hits_BottomRight"]

    # ---------- STEP 3: rate subjects in each modality according to the different analyses requirements ----------
    meg_table = valid_qc_table[valid_qc_table[data_reader.MODALITY] == data_reader.MEEG]
    fmri_table = valid_qc_table[valid_qc_table[data_reader.MODALITY] == data_reader.FMRI]
    mod_result = quality_checks_criteria.rate_subjects(meg_df=meg_table, fmri_df=fmri_table, save_path=save_path)

    # ---------- STEP 4: Optimization table ----------
    optimization_id_dict = create_optimization_table(mod_dict=mod_result)

    # ---------- STEP 5: Analysis table ----------
    create_analysis_table(mod_dict=mod_result, optimization_lists=optimization_id_dict)

    return


def split_replay_data_to_mods(root_folder=data_reader.COGITATE_PATH):
    qc_res_path = data_saver.create_hpc_quality_checks(root_folder)  # the result folder on the HPC (DMT QC folder)
    replay_level_stats_table = pd.read_csv(os.path.join(qc_res_path, EXTRA_FILENAME_REPLAY))

    qc_table = pd.read_csv(os.path.join(qc_res_path, FILENAME))  # save
    valid_subs = qc_table[qc_table[quality_checks_criteria.VALID] == True]["sub_code"]  # take all valid subjects

    #replay_level_stats_table.loc[:, "sub_code"] = replay_level_stats_table["Lab"] + replay_level_stats_table["Subject"].astype(str)
    replay_level_stats_table.loc[:, "sub_code"] = replay_level_stats_table.apply(lambda x: f"{x['Lab']}{x['Subject']:03}", axis=1)
    replay_level_stats_table = replay_level_stats_table[replay_level_stats_table["sub_code"].isin(valid_subs)]
    replay_level_stats_table.loc[:, data_reader.MODALITY] = replay_level_stats_table['Lab'].apply(lambda x: data_reader.METHOD[x])
    modality_names = {data_reader.MEEG: data_reader.MEG, data_reader.FMRI: data_reader.FMRI}
    for modality in modality_names:
        mod_qc = data_saver.create_derivative_qc_hpc(modality_names[modality], root_folder=data_reader.COGITATE_PATH)
        mod_data = replay_level_stats_table.loc[replay_level_stats_table[data_reader.MODALITY] == modality, :]
        mod_data.to_csv(os.path.join(mod_qc, f"ses-v2-included-subs-replay.csv"), index=True)
    return


def check_data(root_folder=data_reader.COGITATE_PATH, prescreen_info=False, is_tau=False):
    """
    This function triggers all the quality checks that will be done on the data.
    :param prescreen_info: if True, provide information about correlations between the prescreening session
    and the experimental one
    :param root_folder: the folder on the HPC that is root, underwhich the paths of the raw and saved data are
    :return: a table where each row = subject, and each column contains information about the behavior of the subject
    in the experiment (game and replay), as well as the results of the behavioral QC done on it.
    """

    qc_res_path = data_saver.create_hpc_quality_checks(root_folder)  # the result folder on the HPC (DMT QC folder)
    # STEP 1: load and save all subject data
    data_table, replay_level_stats_table, subject_dict = extract_data(root_folder=root_folder)
    # Save the subject struct to a pickle
    file_name = f'subject_beh-06-03-26.pickle'
    fl = open(os.path.join(qc_res_path, file_name), 'wb')  # 'ab' apppends new info to the existing file; 'wb' overwrites the entire file
    pickle.dump(subject_dict, fl)
    fl.close()
    # Save the replay level breakdown
    replay_level_stats_table.to_csv(os.path.join(qc_res_path, EXTRA_FILENAME_REPLAY))
    return #TODO: REMOVE THIS LINE

    # STEP 2: BEHAVIORAL QC (on data_table)
    data_table_res = quality_checks_criteria.check_data_table(data_table)
    # make the index of the dataframe the subject codes
    data_table_res["sub_code"] = data_table_res[ID_NAMES[0]] + data_table_res[ID_NAMES[1]]
    data_table_res.set_index("sub_code", inplace=True)

    """
    ** DEPRECATED ** 
    if prescreen_info:  # Optional step: information about correlation between prescreening and game
        boxplotter.plot(data=data_table_res, data_col_order=['SCREEN_hitrate', 'SCREEN_farate'],
                        data_name_order={'SCREEN_hitrate': 'True Positive', 'SCREEN_farate': 'False Positive'},
                        scatter=True, sub_line=True, plot_title="Screening Performance", plot_x_label="Response Type",
                        plot_y_label="Response Type Rate (Percentage)", save_plot=True, save_path=qc_res_path)
        check_prescreen_game_corr(data_table_res, save_path=qc_res_path)
    """

    # STEP 3: SAVE BEH QC per subject in subject derivatives as csv
    for subcode, data in data_table_res.iterrows():
        save_path_tmp = data_saver.create_sub_qcs_hpc(subcode, quality_checks_criteria.METHOD_HPC[subcode[:2]], root_folder)
        data.to_csv(os.path.join(save_path_tmp, f"sub-{subcode}_ses-v2_beh-qc-result.csv"))

    # STEP 3a: IN the derivatives, save all subjects who need to be completely excluded from any analysis (invalid)
    """
    2023-07-14: Manual exclusion of participants who did not pass the 3rd level QCs [reported by Urszula]
    2023-07-17: A joint DMT & fMRI decision to exclude SD142 as well
    2023-10-14: MEG exclusion addition
    2023-11-13: MEG "SB110", "SB999"
    2024-01-31: MEG SB999 NOT EXCLUDED
    2024-02-06: ET invalid - exclude "SD176", "SD156", "SD201", "SA151" (put it in the 3rd level)
    2024-05-23: Added SD116 to third level exclusion
    """
    third_level_exclusion = ["SC101", "SC126", "SD116", "SC130", "SC169", "SD111", "SD122", "SD142", "SD187", "SB038", "SA110"]  #, "SB999" , "SD176", "SD156", "SD201", "SA151"
    for ind, row in data_table_res.iterrows():
        sub_code = row["Lab"] + row["Subject"]
        if sub_code in third_level_exclusion:  # if subject is 3rd level failing
            data_table_res.loc[ind, quality_checks_criteria.VALID] = False  # make them invalid, no matter what happened in 2nd

    # save the QC data table
    data_saver.safe_save(qc_res_path, FILENAME)  # check if there's an existing table with that name and alert if so
    data_table_res.to_csv(os.path.join(qc_res_path, FILENAME))  # save
    invalid_subs = data_table_res[data_table_res[quality_checks_criteria.VALID] != True]  # take all invalid subjects
    modality_names = {data_reader.MEEG: data_reader.MEG, data_reader.FMRI: data_reader.FMRI}
    for modality in modality_names:
        mod_qc = data_saver.create_derivative_qc_hpc(modality_names[modality], root_folder=data_reader.COGITATE_PATH)
        mod_data = invalid_subs.loc[invalid_subs[data_reader.MODALITY] == modality, quality_checks_criteria.VALID]
        mod_data.to_csv(os.path.join(mod_qc, f"ses-v2-excluded-subs.csv"), index=True)

    # STEP 4: RATE SUBJECTS FOR NEURAL DATA ANALYSIS
    if not is_tau:
        create_subject_lists(data_table_res, qc_res_path)

    return data_table_res


def calculate_difficulty_stats(subs_data, only_valid, save_path):
    all_subs_df_list = []

    for k in subs_data.keys():
        if k not in list(only_valid["sub_code"]):
            continue
        if k in CHOSEN_10:
            continue

        # First, we do this for GAME
        game_data = subs_data[k].full.SessDetails.ProbeDet[subs_data[k].full.SessDetails.ProbeDet['activeWorldID'] != -1]
        game_data.loc[:, "sub"] = k
        game_data.loc[:, "mod"] = subs_data[k].mod
        game_data.loc[:, "condition"] = "GAME"
        stim_present = game_data[~game_data[STIM_TYPE].str.contains(NON)]
        stim_present = stim_present[(stim_present[EVAL].str.contains(TRUEPOSITIVE)) | (stim_present[EVAL].str.contains(FALSENEGATIVE))]
        stim_present = stim_present.loc[:, ["sub", "mod", EVAL, "difficulty"]]
        all_subs_df_list.append(stim_present)

    all_df = pd.concat(all_subs_df_list)
    per_sub_avg = all_df.groupby(["sub", "mod", EVAL]).mean().reset_index()

    all_stats = list()
    for group in [["mod", EVAL], [EVAL], ["mod"]]:
        per_group_avg = per_sub_avg.groupby(group).mean().reset_index()
        per_group_std = per_sub_avg.groupby(group).std().reset_index()
        per_group_avg["difficulty_std"] = per_group_std["difficulty"]
        all_stats.append(per_group_avg)
    total_avg = pd.concat(all_stats)

    total_avg.to_csv(os.path.join(save_path, f"vg_difficulty_stats.csv"), index=False)
    return


def calculate_fa_wfillers_stats(subs_data, only_valid, save_path):
    all_subs_df_list = []

    for k in subs_data.keys():
        if k not in list(only_valid["sub_code"]):
            continue
        if k in CHOSEN_10:
            continue

        # REPLAY ONLY, REMOVE FILLERS!
        replay_data = subs_data[k].full.SessDetails.LocalizerDetFullCorrected
        replay_data.loc[:, "sub"] = k
        replay_data.loc[:, "mod"] = subs_data[k].mod
        replay_data.loc[:, "condition"] = "REPLAY"
        stim_present, stim_absent, stim_nontarget = probe_data_go_nogo(replay_data, subs_data[k].full.replay_targets)
        stim_absent = stim_absent.loc[stim_absent["IS_LEVEL_ELIMINATED"] == 0, :]
        stim_absent.loc[:, "fa"] = False
        stim_absent.loc[(stim_absent[EVAL].notnull()) & (stim_absent[EVAL].str.contains(FALSEPOSITIVE)), "fa"] = True
        stim_absent.loc[:, TARGET_TYPE] = stim_absent[TARGET_TYPE]
        stim_absent = stim_absent.loc[:, ["sub", "mod", "condition", "fa", TARGET_TYPE]]
        all_subs_df_list.append(stim_absent)

    all_df = pd.concat(all_subs_df_list)
    all_df.to_csv(os.path.join(save_path, f"fa_categoryirrelevant_wfillers_giant_replay.csv"), index=False)
    list_to_combo_from = ["mod", TARGET_TYPE]
    combos = []
    for r in range(1, len(list_to_combo_from) + 1):
        combos.append(itertools.combinations(list_to_combo_from, r))

    stats_df_list = []
    df_to_mean_list = []
    for combo in combos:
        for x in combo:
            if "mod" not in list(x):
                df_to_mean = all_df.groupby(list(x) + ["sub", "mod"]).mean().reset_index()
            else:
                df_to_mean = all_df.groupby(list(x) + ["sub"]).mean().reset_index()
            mean_df = df_to_mean.groupby(list(x)).mean().reset_index()
            std_df = df_to_mean.groupby(list(x)).std().reset_index()
            mean_df.loc[:, "farate_std"] = std_df["fa"]
            mean_df.rename(columns={"hit": "farate_mean"}, inplace=True)
            stats_df_list.append(mean_df)
            df_to_mean_list.append(df_to_mean)

    df_to_mean = pd.concat(df_to_mean_list)
    df_to_mean.to_csv(os.path.join(save_path, f"fa_categoryirrelevant_wfillers_giant_replay_sub_stats.csv"), index=False)
    stat_df = pd.concat(stats_df_list)
    stat_df.to_csv(os.path.join(save_path, f"fa_categoryirrelevant_wfillers_giant_replay_stats.csv"), index=False)
    return


def calculate_hitrate_everything(subs_data, only_valid, modality, save, save_path):
    all_subs_df_list = []

    for k in subs_data.keys():
        if k not in list(only_valid["sub_code"]):
            continue
        if k in CHOSEN_10:
            continue
        if modality != "both" and subs_data[k].mod != modality:
            continue

        # First, we do this for GAME
        game_data = subs_data[k].full.SessDetails.ProbeDet[subs_data[k].full.SessDetails.ProbeDet['activeWorldID'] != -1]
        game_data.loc[:, "sub"] = k
        game_data.loc[:, "mod"] = subs_data[k].mod
        game_data.loc[:, "condition"] = "GAME"
        stim_present = game_data[~game_data[STIM_TYPE].str.contains(NON)]
        stim_present.loc[:, "hit"] = False
        stim_present.loc[stim_present[EVAL].str.contains(TRUEPOSITIVE), "hit"] = True
        stim_present.loc[stim_present[STIM_LOC].str.contains(LEFT), "horizontal"] = LEFT
        stim_present.loc[stim_present[STIM_LOC].str.contains(RIGHT), "horizontal"] = RIGHT
        stim_present.loc[stim_present[STIM_LOC].str.contains(TOP), "vertical"] = TOP
        stim_present.loc[stim_present[STIM_LOC].str.contains(BOTTOM), "vertical"] = BOTTOM
        stim_present.loc[:, "stim_category"] = stim_present[STIM_TYPE]
        stim_present = stim_present.loc[:, ["sub", "mod", "condition", "hit", "stim_category", "horizontal", "vertical"]]
        all_subs_df_list.append(stim_present)

        replay_data = subs_data[k].full.SessDetails.LocalizerDetFullCorrected[subs_data[k].full.SessDetails.LocalizerDetFullCorrected["type"] != 'LOCALIZER_FILLER']
        replay_data.loc[:, "sub"] = k
        replay_data.loc[:, "mod"] = subs_data[k].mod
        replay_data.loc[:, "condition"] = "REPLAY"
        stim_present, stim_absent, stim_nontarget = probe_data_go_nogo(replay_data, subs_data[k].full.replay_targets)
        stim_present = stim_present.loc[stim_present["IS_LEVEL_ELIMINATED"] == 0, :]
        stim_present.loc[:, "hit"] = False
        stim_present.loc[stim_present[EVAL].str.contains(TRUEPOSITIVE), "hit"] = True
        stim_present.loc[stim_present[STIM_LOC].str.contains(LEFT), "horizontal"] = LEFT
        stim_present.loc[stim_present[STIM_LOC].str.contains(RIGHT), "horizontal"] = RIGHT
        stim_present.loc[stim_present[STIM_LOC].str.contains(TOP), "vertical"] = TOP
        stim_present.loc[stim_present[STIM_LOC].str.contains(BOTTOM), "vertical"] = BOTTOM
        stim_present.loc[:, "stim_category"] = stim_present[TARGET_TYPE]
        stim_present = stim_present.loc[:, ["sub", "mod", "condition", "hit", "stim_category", "horizontal", "vertical"]]
        all_subs_df_list.append(stim_present)

    all_df = pd.concat(all_subs_df_list)
    all_df.to_csv(os.path.join(save_path, f"hitrate_giant.csv"), index=False)
    list_to_combo_from = ["mod", "condition", "stim_category", "horizontal", "vertical"]
    combos = []
    for r in range(1, len(list_to_combo_from) + 1):
        combos.append(itertools.combinations(list_to_combo_from, r))

    stats_df_list = []
    df_to_mean_list = []
    for combo in combos:
        for x in combo:
            if "mod" not in list(x):
                df_to_mean = all_df.groupby(list(x)+["sub", "mod"]).mean().reset_index()
            else:
                df_to_mean = all_df.groupby(list(x) + ["sub"]).mean().reset_index()
            mean_df = df_to_mean.groupby(list(x)).mean().reset_index()
            std_df = df_to_mean.groupby(list(x)).std().reset_index()
            mean_df.loc[:, "hitrate_std"] = std_df["hit"]
            mean_df.rename(columns={"hit": "hitrate_mean"}, inplace=True)
            stats_df_list.append(mean_df)
            df_to_mean_list.append(df_to_mean)

    df_to_mean = pd.concat(df_to_mean_list)
    df_to_mean.to_csv(os.path.join(save_path, f"hitrate_sub_stats.csv"), index=False)
    stat_df = pd.concat(stats_df_list)
    stat_df.to_csv(os.path.join(save_path, f"hitrate_stats.csv"), index=False)
    return


def calculate_fa_everything(subs_data, only_valid, modality, save, save_path):
    all_subs_df_list = []

    for k in subs_data.keys():
        if k not in list(only_valid["sub_code"]):
            continue
        if k in CHOSEN_10:
            continue
        if modality != "both" and subs_data[k].mod != modality:
            continue

        # REPLAY ONLY, REMOVE FILLERS!
        replay_data = subs_data[k].full.SessDetails.LocalizerDetFullCorrected[subs_data[k].full.SessDetails.LocalizerDetFullCorrected["type"] != 'LOCALIZER_FILLER']
        replay_data.loc[:, "sub"] = k
        replay_data.loc[:, "mod"] = subs_data[k].mod
        replay_data.loc[:, "condition"] = "REPLAY"
        stim_present, stim_absent, stim_nontarget = probe_data_go_nogo(replay_data, subs_data[k].full.replay_targets)
        stim_absent = stim_absent.loc[stim_absent["IS_LEVEL_ELIMINATED"] == 0, :]
        stim_absent.loc[:, "fa"] = False
        stim_absent.loc[(stim_absent[EVAL].notnull()) & (stim_absent[EVAL].str.contains(FALSEPOSITIVE)), "fa"] = True
        stim_absent.loc[stim_absent[STIM_LOC].str.contains(LEFT), "horizontal"] = LEFT
        stim_absent.loc[stim_absent[STIM_LOC].str.contains(RIGHT), "horizontal"] = RIGHT
        stim_absent.loc[stim_absent[STIM_LOC].str.contains(TOP), "vertical"] = TOP
        stim_absent.loc[stim_absent[STIM_LOC].str.contains(BOTTOM), "vertical"] = BOTTOM
        stim_absent.loc[:, TARGET_TYPE] = stim_absent[TARGET_TYPE]
        stim_absent = stim_absent.loc[:, ["sub", "mod", "condition", "fa", TARGET_TYPE, "horizontal", "vertical"]]
        all_subs_df_list.append(stim_absent)

    all_df = pd.concat(all_subs_df_list)
    all_df.to_csv(os.path.join(save_path, f"fa_categoryirrelevant_giant_replay.csv"), index=False)
    list_to_combo_from = ["mod", TARGET_TYPE, "horizontal", "vertical"]
    combos = []
    for r in range(1, len(list_to_combo_from) + 1):
        combos.append(itertools.combinations(list_to_combo_from, r))

    stats_df_list = []
    df_to_mean_list = []
    for combo in combos:
        for x in combo:
            if "mod" not in list(x):
                df_to_mean = all_df.groupby(list(x)+["sub", "mod"]).mean().reset_index()
            else:
                df_to_mean = all_df.groupby(list(x) + ["sub"]).mean().reset_index()
            mean_df = df_to_mean.groupby(list(x)).mean().reset_index()
            std_df = df_to_mean.groupby(list(x)).std().reset_index()
            mean_df.loc[:, "farate_std"] = std_df["fa"]
            mean_df.rename(columns={"hit": "farate_mean"}, inplace=True)
            stats_df_list.append(mean_df)
            df_to_mean_list.append(df_to_mean)

    df_to_mean = pd.concat(df_to_mean_list)
    df_to_mean.to_csv(os.path.join(save_path, f"fa_categoryirrelevant_giant_replay_sub_stats.csv"), index=False)
    stat_df = pd.concat(stats_df_list)
    stat_df.to_csv(os.path.join(save_path, f"fa_categoryirrelevant_giant_replay_stats.csv"), index=False)

    """
    FAs in the game are "Yes" response when no stim was presented.
    FA in the replay is a response only to the non-target stimulus. 
    """
    all_subs_df_list = []

    for k in subs_data.keys():
        if k not in list(only_valid["sub_code"]):
            continue
        if k in CHOSEN_10:
            continue
        if modality != "both" and subs_data[k].mod != modality:
            continue

        # REPLAY ONLY, REMOVE FILLERS!
        replay_data = subs_data[k].full.SessDetails.LocalizerDetFullCorrected[
            subs_data[k].full.SessDetails.LocalizerDetFullCorrected["type"] != 'LOCALIZER_FILLER']
        replay_data.loc[:, "sub"] = k
        replay_data.loc[:, "mod"] = subs_data[k].mod
        stim_present, stim_absent, stim_nontarget = probe_data_go_nogo(replay_data, subs_data[k].full.replay_targets)
        stim_absent = stim_absent.loc[stim_absent["IS_LEVEL_ELIMINATED"] == 0, :]
        stim_absent.loc[:, "fa"] = False
        stim_absent.loc[(stim_absent[EVAL].notnull()) & (stim_absent[EVAL].str.contains(FALSEPOSITIVE)), "fa"] = True
        stim_absent.loc[stim_absent[STIM_LOC].str.contains(LEFT), "horizontal"] = LEFT
        stim_absent.loc[stim_absent[STIM_LOC].str.contains(RIGHT), "horizontal"] = RIGHT
        stim_absent.loc[stim_absent[STIM_LOC].str.contains(TOP), "vertical"] = TOP
        stim_absent.loc[stim_absent[STIM_LOC].str.contains(BOTTOM), "vertical"] = BOTTOM
        stim_absent.loc[:, "stim_category"] = stim_absent[STIM_TYPE]
        stim_absent = stim_absent.loc[:, ["sub", "mod", "fa", "stim_category", "horizontal", "vertical"]]
        all_subs_df_list.append(stim_absent)

    all_df = pd.concat(all_subs_df_list)
    all_df.to_csv(os.path.join(save_path, f"fa_giant_replay_stimCat.csv"), index=False)
    replay_fas = all_df

    for k in subs_data.keys():
        if k not in list(only_valid["sub_code"]):
            continue
        if k in CHOSEN_10:
            continue
        if modality != "both" and subs_data[k].mod != modality:
            continue

        # First, we do this for GAME
        game_data = subs_data[k].full.SessDetails.ProbeDet[
            subs_data[k].full.SessDetails.ProbeDet['activeWorldID'] != -1]
        game_data.loc[:, "sub"] = k
        game_data.loc[:, "mod"] = subs_data[k].mod

        stim_absent = game_data[game_data[STIM_TYPE].str.contains(NON)]
        stim_absent.loc[:, "fa"] = False
        stim_absent.loc[(stim_absent[EVAL].notnull()) & (stim_absent[EVAL].str.contains(FALSEPOSITIVE)), "fa"] = True
        stim_absent.loc[stim_absent[STIM_LOC].str.contains(LEFT), "horizontal"] = LEFT
        stim_absent.loc[stim_absent[STIM_LOC].str.contains(RIGHT), "horizontal"] = RIGHT
        stim_absent.loc[stim_absent[STIM_LOC].str.contains(TOP), "vertical"] = TOP
        stim_absent.loc[stim_absent[STIM_LOC].str.contains(BOTTOM), "vertical"] = BOTTOM
        stim_absent = stim_absent.loc[:, ["sub", "mod", "fa", "horizontal", "vertical"]]
        all_subs_df_list.append(stim_absent)

    all_df = pd.concat(all_subs_df_list)
    all_df.to_csv(os.path.join(save_path, f"fa_giant_game.csv"), index=False)
    game_fas = all_df

    game_fas.loc[:, "category"] = "GAME"
    replay_fas.loc[:, "category"] = "REPLAY"
    game_means = game_fas.groupby(["sub"]).mean().reset_index()
    replay_means = replay_fas.groupby(["sub"]).mean().reset_index()
    game_means.loc[:, "fa_stim_replay"] = replay_means["fa"]
    game_means.rename(columns={"fa": "fa_game"}, inplace=True)
    game_means.to_csv(os.path.join(save_path, f"fa_overall_game_v_replay.csv"), index=False)

    all_subs_df_list = []

    for k in subs_data.keys():
        if k not in list(only_valid["sub_code"]):
            continue
        if k in CHOSEN_10:
            continue
        if modality != "both" and subs_data[k].mod != modality:
            continue

        # REPLAY, THIS TIME ONLY FILLERS!
        replay_data = subs_data[k].full.SessDetails.LocalizerDetFullCorrected
        replay_data.loc[:, "sub"] = k
        replay_data.loc[:, "mod"] = subs_data[k].mod
        replay_data.loc[:, "condition"] = "REPLAY"
        stim_present, stim_absent, stim_nontarget = probe_data_go_nogo(replay_data, subs_data[k].full.replay_targets)
        stim_absent = pd.concat([stim_absent, stim_nontarget])
        stim_absent = stim_absent[stim_absent["type"] == 'LOCALIZER_FILLER']
        stim_absent = stim_absent.loc[stim_absent["IS_LEVEL_ELIMINATED"] == 0, :]
        stim_absent.loc[:, "fa"] = False
        stim_absent.loc[:, TARGET_TYPE] = stim_absent[TARGET_TYPE]
        stim_absent.loc[(stim_absent[EVAL].notnull()) & (stim_absent[EVAL].str.contains(FALSEPOSITIVE)), "fa"] = True
        stim_absent = stim_absent.loc[:, ["sub", "mod", "fa", TARGET_TYPE]]
        all_subs_df_list.append(stim_absent)

    all_df = pd.concat(all_subs_df_list)
    all_df.to_csv(os.path.join(save_path, f"fa_giant_replay_OnlyFillers.csv"), index=False)
    return


def get_tables_for_further_analysis(root_folder, is_tau=False):
    qc_res_path = data_saver.create_hpc_quality_checks(root_folder)  # the result folder on the HPC (DMT QC folder)
    data = pd.read_csv(os.path.join(qc_res_path, FILENAME))
    data = data[data[quality_checks_criteria.VALID] == True]  # filter valid subs only

    """
    # TODO: REMOVE THIS START
    # Part5: plot moving average and difficulty - VALID ONLY
    file_name = f'subject_beh.pickle'
    fl = open(os.path.join(qc_res_path, file_name), 'rb')
    subs_data = pickle.load(fl)
    fl.close()
    subs_list = list()
    means_list = list()
    stds_list = list()
    for k in subs_data.keys():
        if k not in list(data["sub_code"]):
            continue
        replay_data = subs_data[k].full.SessDetails.LocalizerDetFullCorrected[subs_data[k].full.SessDetails.LocalizerDetFullCorrected["type"] != 'LOCALIZER_FILLER']
        subs_list.append(k)
        means_list.append(replay_data[DT_SINCE_LAST].mean())
        stds_list.append(replay_data[DT_SINCE_LAST].std())
    dt_df = pd.DataFrame({"sub": subs_list, "mean": means_list, "std": stds_list})
    dt_df.to_csv(os.path.join(qc_res_path, f"dtReplaySubs.csv"), index=False)
    # TODO: REMOVE THIS END
    """
    # Part0: output general analysis stats of all valid subs
    """
    general_data_df = data[["sub_code", data_reader.MODALITY, "GAME_hitrate", "GAME_missrate", "GAME_farate", "GAME_hit_rt_mean",
                            "GAME_hit_rt_std", "GAME_miss_rt_mean", "GAME_miss_rt_std" , "GAME_hit_difficulty", "GAME_miss_difficulty",
                            "LOC_hitrate", "LOC_farate", "LOC_farate_onlyNonTargetStim"]]

    general_data_df.to_csv(os.path.join(qc_res_path, f"quality_checks_general_stats.csv"), index=False)
    general_data_df = general_data_df[~general_data_df["sub_code"].isin(CHOSEN_10)]
    general_data_df.to_csv(os.path.join(qc_res_path, f"quality_checks_general_stats_no_chosen10.csv"), index=False)
    """

    general_data_df = data[~data["sub_code"].isin(CHOSEN_10)]
    means_both_df = pd.DataFrame(general_data_df.mean()).T
    std_both_df = pd.DataFrame(general_data_df.std()).T
    means_mod_df = general_data_df.groupby([data_reader.MODALITY]).mean().reset_index()
    stds_mod_df = general_data_df.groupby([data_reader.MODALITY]).std().reset_index()
    means_mod_df.loc[:, "What"] = "Means"
    stds_mod_df.loc[:, "What"] = "STDs"
    means_both_df.loc[:, "What"] = "Means"
    std_both_df.loc[:, "What"] = "STDs"
    general_data_df_stats = pd.concat([means_mod_df, stds_mod_df, means_both_df, std_both_df])
    general_data_df_stats.to_csv(os.path.join(qc_res_path, f"quality_checks_general_stats_means_no_chosen10.csv"), index=False)
    return
    if is_tau:
        modality_list = [data_reader.MEEG]
    else:
        modality_list = ["both", data_reader.FMRI, data_reader.MEEG]

    # Part1: hits_per_cond_category.csv
    hitrate_cols = ["GAME_hitrate_face", "GAME_hitrate_obj", "LOC_face_hit_rate", "LOC_obj_hits_rate"]
    df_list = list()
    for i in range(len(hitrate_cols)):
        hitrate_col = hitrate_cols[i]
        data_subdf = data[["sub_code", data_reader.MODALITY, hitrate_col]]
        data_subdf.rename(columns={hitrate_col: "hitrate"}, inplace=True)
        obj_type = "obj" if "obj" in hitrate_col else "face"
        condition = "GAME" if "GAME" in hitrate_col else "REPLAY"
        data_subdf["stim_category"] = obj_type
        data_subdf["condition"] = condition
        df_list.append(data_subdf)
    result_df = pd.concat(df_list)
    result_df.to_csv(os.path.join(qc_res_path, f"hits_per_cond_category.csv"), index=False)

    # Without a variable
    for name in ["all", "chosen10"]:
        all_dfs = []
        for indep in [data_reader.MODALITY, "stim_category", "condition"]:
            if name == "all":
                relevant_data = result_df[~result_df["sub_code"].isin(CHOSEN_10)]
            else:
                relevant_data = result_df[result_df["sub_code"].isin(CHOSEN_10)]
            means_df = relevant_data.groupby([indep]).mean().reset_index()
            means_df = means_df[[indep, "hitrate"]]
            means_df.rename(columns={indep: INDEPENDENT, "hitrate": "hitrate_mean"}, inplace=True)
            all_dfs.append(means_df)

            means_df = relevant_data.groupby([indep]).std().reset_index()
            means_df = means_df[[indep, "hitrate"]]
            means_df.rename(columns={indep: INDEPENDENT, "hitrate": "hitrate_std"},
                            inplace=True)
            all_dfs.append(means_df)

        for combo in [[data_reader.MODALITY, "stim_category"], ["stim_category", "condition"], [data_reader.MODALITY, "condition"], [data_reader.MODALITY, "stim_category", "condition"]]:
            if name == "all":
                relevant_data = result_df[~result_df["sub_code"].isin(CHOSEN_10)]
            else:
                relevant_data = result_df[result_df["sub_code"].isin(CHOSEN_10)]

            means_df = relevant_data.groupby(combo).mean().reset_index()
            if len(combo) == 2:
                means_df.loc[:, INDEPENDENT] = means_df[combo[0]] + " " + means_df[combo[1]]
            else:
                means_df.loc[:, INDEPENDENT] = means_df[combo[0]] + " " + means_df[combo[1]] + " " + means_df[combo[2]]
            means_df = means_df[[INDEPENDENT, "hitrate"]]
            means_df.rename(columns={"hitrate": "hitrate_mean"},
                            inplace=True)
            all_dfs.append(means_df)

            means_df = relevant_data.groupby(combo).std().reset_index()
            if len(combo) == 2:
                means_df.loc[:, INDEPENDENT] = means_df[combo[0]] + " " + means_df[combo[1]]
            else:
                means_df.loc[:, INDEPENDENT] = means_df[combo[0]] + " " + means_df[combo[1]] + " " + means_df[combo[2]]
            means_df = means_df[[INDEPENDENT, "hitrate"]]
            means_df.rename(columns={"hitrate": "hitrate_std"},
                            inplace=True)
            all_dfs.append(means_df)

        merged_df = pd.concat(all_dfs)
        merged_df.to_csv(os.path.join(qc_res_path, f"hits_per_cond_category_{name}_descriptives.csv"), index=False)

    # Part2: hits_per_cond_top_bottom.csv
    hitrate_cols = ["GAME_hitrate_Top", "GAME_hitrate_Bottom", "LOC_hitrate_Top", "LOC_hitrate_Bottom"]
    df_list = list()
    for i in range(len(hitrate_cols)):
        hitrate_col = hitrate_cols[i]
        data_subdf = data[["sub_code", data_reader.MODALITY, hitrate_col]]
        data_subdf.rename(columns={hitrate_col: "hitrate"}, inplace=True)
        obj_type = "Top" if "Top" in hitrate_col else "Bottom"
        condition = "GAME" if "GAME" in hitrate_col else "REPLAY"
        data_subdf["location"] = obj_type
        data_subdf["condition"] = condition
        df_list.append(data_subdf)
    result_df = pd.concat(df_list)
    result_df.to_csv(os.path.join(qc_res_path, f"hits_per_cond_top_bottom.csv"), index=False)

    # Without a variable
    for name in ["all", "chosen10"]:
        all_dfs = []
        for indep in [data_reader.MODALITY, "location", "condition"]:
            if name == "all":
                relevant_data = result_df[~result_df["sub_code"].isin(CHOSEN_10)]
            else:
                relevant_data = result_df[result_df["sub_code"].isin(CHOSEN_10)]
            means_df = relevant_data.groupby([indep]).mean().reset_index()
            means_df = means_df[[indep, "hitrate"]]
            means_df.rename(columns={indep: INDEPENDENT, "hitrate": "hitrate_mean"}, inplace=True)
            all_dfs.append(means_df)

            means_df = relevant_data.groupby([indep]).std().reset_index()
            means_df = means_df[[indep, "hitrate"]]
            means_df.rename(columns={indep: INDEPENDENT, "hitrate": "hitrate_std"},
                            inplace=True)
            all_dfs.append(means_df)

        for combo in [[data_reader.MODALITY, "location"], ["location", "condition"], [data_reader.MODALITY, "condition"], [data_reader.MODALITY, "location", "condition"]]:
            if name == "all":
                relevant_data = result_df[~result_df["sub_code"].isin(CHOSEN_10)]
            else:
                relevant_data = result_df[result_df["sub_code"].isin(CHOSEN_10)]

            means_df = relevant_data.groupby(combo).mean().reset_index()
            if len(combo) == 2:
                means_df.loc[:, INDEPENDENT] = means_df[combo[0]] + " " + means_df[combo[1]]
            else:
                means_df.loc[:, INDEPENDENT] = means_df[combo[0]] + " " + means_df[combo[1]] + " " + means_df[combo[2]]
            means_df = means_df[[INDEPENDENT, "hitrate"]]
            means_df.rename(columns={"hitrate": "hitrate_mean"},
                            inplace=True)
            all_dfs.append(means_df)

            means_df = relevant_data.groupby(combo).std().reset_index()
            if len(combo) == 2:
                means_df.loc[:, INDEPENDENT] = means_df[combo[0]] + " " + means_df[combo[1]]
            else:
                means_df.loc[:, INDEPENDENT] = means_df[combo[0]] + " " + means_df[combo[1]] + " " + means_df[combo[2]]
            means_df = means_df[[INDEPENDENT, "hitrate"]]
            means_df.rename(columns={"hitrate": "hitrate_std"},
                            inplace=True)
            all_dfs.append(means_df)

        merged_df = pd.concat(all_dfs)
        merged_df.to_csv(os.path.join(qc_res_path, f"hits_per_cond_top_bottom_{name}_descriptives.csv"), index=False)

    # Part3: hits_per_cond_left_right.csv
    hitrate_cols = ["GAME_hitrate_Left", "GAME_hitrate_Right", "LOC_hitrate_Left", "LOC_hitrate_Right"]
    df_list = list()
    for i in range(len(hitrate_cols)):
        hitrate_col = hitrate_cols[i]
        data_subdf = data[["sub_code", data_reader.MODALITY, hitrate_col]]
        data_subdf.rename(columns={hitrate_col: "hitrate"}, inplace=True)
        obj_type = "Left" if "Left" in hitrate_col else "Right"
        condition = "GAME" if "GAME" in hitrate_col else "REPLAY"
        data_subdf["location"] = obj_type
        data_subdf["condition"] = condition
        df_list.append(data_subdf)
    result_df = pd.concat(df_list)
    result_df.to_csv(os.path.join(qc_res_path, f"hits_per_cond_left_right.csv"), index=False)

    # Without a variable
    for name in ["all", "chosen10"]:
        all_dfs = []
        for indep in [data_reader.MODALITY, "location", "condition"]:
            if name == "all":
                relevant_data = result_df[~result_df["sub_code"].isin(CHOSEN_10)]
            else:
                relevant_data = result_df[result_df["sub_code"].isin(CHOSEN_10)]
            means_df = relevant_data.groupby([indep]).mean().reset_index()
            means_df = means_df[[indep, "hitrate"]]
            means_df.rename(columns={indep: INDEPENDENT, "hitrate": "hitrate_mean"}, inplace=True)
            all_dfs.append(means_df)

            means_df = relevant_data.groupby([indep]).std().reset_index()
            means_df = means_df[[indep, "hitrate"]]
            means_df.rename(columns={indep: INDEPENDENT, "hitrate": "hitrate_std"},
                            inplace=True)
            all_dfs.append(means_df)

        for combo in [[data_reader.MODALITY, "location"], ["location", "condition"], [data_reader.MODALITY, "condition"], [data_reader.MODALITY, "location", "condition"]]:
            if name == "all":
                relevant_data = result_df[~result_df["sub_code"].isin(CHOSEN_10)]
            else:
                relevant_data = result_df[result_df["sub_code"].isin(CHOSEN_10)]

            means_df = relevant_data.groupby(combo).mean().reset_index()
            if len(combo) == 2:
                means_df.loc[:, INDEPENDENT] = means_df[combo[0]] + " " + means_df[combo[1]]
            else:
                means_df.loc[:, INDEPENDENT] = means_df[combo[0]] + " " + means_df[combo[1]] + " " + means_df[combo[2]]
            means_df = means_df[[INDEPENDENT, "hitrate"]]
            means_df.rename(columns={"hitrate": "hitrate_mean"},
                            inplace=True)
            all_dfs.append(means_df)

            means_df = relevant_data.groupby(combo).std().reset_index()
            if len(combo) == 2:
                means_df.loc[:, INDEPENDENT] = means_df[combo[0]] + " " + means_df[combo[1]]
            else:
                means_df.loc[:, INDEPENDENT] = means_df[combo[0]] + " " + means_df[combo[1]] + " " + means_df[combo[2]]
            means_df = means_df[[INDEPENDENT, "hitrate"]]
            means_df.rename(columns={"hitrate": "hitrate_std"},
                            inplace=True)
            all_dfs.append(means_df)

        merged_df = pd.concat(all_dfs)
        merged_df.to_csv(os.path.join(qc_res_path, f"hits_per_cond_left_right_{name}_descriptives.csv"), index=False)

    """
        # Part1: hits_fas_per_cond_category.csv
    farate_cols = ["GAME_farate_face", "GAME_farate_obj", "LOC_face_farate", "LOC_obj_farate"]
    hitrate_cols = ["GAME_hitrate_face", "GAME_hitrate_obj", "LOC_face_hit_rate", "LOC_obj_hits_rate"]
    df_list = list()
    for i in range(len(hitrate_cols)):
        hitrate_col = hitrate_cols[i]
        farate_col = farate_cols[i]
        data_subdf = data[["sub_code", data_reader.MODALITY, hitrate_col, farate_col]]
        data_subdf.rename(columns={hitrate_col: "hitrate", farate_col: "farate"}, inplace=True)
        obj_type = "obj" if "obj" in hitrate_col else "face"
        condition = "GAME" if "GAME" in hitrate_col else "REPLAY"
        data_subdf["stim_category"] = obj_type
        data_subdf["condition"] = condition
        df_list.append(data_subdf)
    result_df = pd.concat(df_list)
    result_df.to_csv(os.path.join(qc_res_path, f"hits_fas_per_cond_category.csv"), index=False)

    # Without a variable
    for name in ["all" ,"chosen10"]:
        all_dfs = []
        for indep in [data_reader.MODALITY, "stim_category", "condition"]:
            if name == "all":
                relevant_data = result_df[~result_df["sub_code"].isin(CHOSEN_10)]
            else:
                relevant_data = result_df[result_df["sub_code"].isin(CHOSEN_10)]
            means_df = relevant_data.groupby([indep]).mean().reset_index()
            means_df = means_df[[indep, "hitrate", "farate"]]
            means_df.rename(columns={indep: INDEPENDENT, "hitrate": "hitrate_mean", "farate": "farate_mean"}, inplace=True)
            all_dfs.append(means_df)

            means_df = relevant_data.groupby([indep]).std().reset_index()
            means_df = means_df[[indep, "hitrate", "farate"]]
            means_df.rename(columns={indep: INDEPENDENT, "hitrate": "hitrate_std", "farate": "farate_std"},
                            inplace=True)
            all_dfs.append(means_df)

        merged_df = pd.concat(all_dfs)
        merged_df.to_csv(os.path.join(qc_res_path, f"hits_fas_per_cond_category_{name}_descriptives.csv"), index=False)

    DEPRECATED
    # Part2: fas_in_blanks_per_cond_category.csv
    farate_cols = ["GAME_farate_blank", "LOC_blank_farate"]
    df_list = list()
    for farate_col in farate_cols:
        data_subdf = data[["sub_code", data_reader.MODALITY, farate_col]]
        data_subdf.rename(columns={farate_col: "blank_fa_rate"}, inplace=True)
        condition = "GAME" if "GAME" in farate_col else "REPLAY"
        data_subdf["condition"] = condition
        df_list.append(data_subdf)
    result_df = pd.concat(df_list)
    result_df.to_csv(os.path.join(qc_res_path, f"fas_in_blanks_per_cond_category.csv"), index=False)
    
    # Without a variable
    for name in ["all" ,"chosen10"]:
        all_dfs = []
        for indep in [data_reader.MODALITY, "condition"]:
            if name == "all":
                relevant_data = result_df[~result_df["sub_code"].isin(CHOSEN_10)]
            else:
                relevant_data = result_df[result_df["sub_code"].isin(CHOSEN_10)]
            means_df = relevant_data.groupby([indep]).mean().reset_index()
            means_df = means_df[[indep, "blank_fa_rate"]]
            means_df.rename(columns={indep: INDEPENDENT, "blank_fa_rate": "blank_fa_rate_mean"},
                            inplace=True)
            all_dfs.append(means_df)

            means_df = relevant_data.groupby([indep]).std().reset_index()
            means_df = means_df[[indep, "blank_fa_rate"]]
            means_df.rename(columns={indep: INDEPENDENT, "blank_fa_rate": "blank_fa_rate_std"},
                            inplace=True)
            all_dfs.append(means_df)

        merged_df = pd.concat(all_dfs)
        merged_df.to_csv(os.path.join(qc_res_path, f"fas_in_blanks_per_cond_category_{name}_descriptives.csv"), index=False)
    """

    # Part3: vg_difficulty_per_vis.csv
    vgdiff_cols = ["GAME_hit_difficulty", "GAME_miss_difficulty"]
    df_list = list()
    for vgdiff_col in vgdiff_cols:
        data_subdf = data[["sub_code", data_reader.MODALITY, vgdiff_col]]
        data_subdf["visibility"] = "seen" if "hit" in vgdiff_col else "unseen"
        data_subdf.rename(columns={vgdiff_col: "vgdiff"}, inplace=True)
        df_list.append(data_subdf)
    result_df = pd.concat(df_list)
    result_df.to_csv(os.path.join(qc_res_path, f"vg_difficulty_per_vis.csv"), index=False)

    # Without a variable
    for name in ["all", "chosen10"]:
        all_dfs = []
        for indep in [data_reader.MODALITY, "visibility"]:
            if name == "all":
                relevant_data = result_df[~result_df["sub_code"].isin(CHOSEN_10)]
            else:
                relevant_data = result_df[result_df["sub_code"].isin(CHOSEN_10)]
            means_df = relevant_data.groupby([indep]).mean().reset_index()
            means_df = means_df[[indep, "vgdiff"]]
            means_df.rename(columns={indep: INDEPENDENT, "vgdiff": "vgdiff_mean"},
                            inplace=True)
            all_dfs.append(means_df)

            means_df = relevant_data.groupby([indep]).std().reset_index()
            means_df = means_df[[indep, "vgdiff"]]
            means_df.rename(columns={indep: INDEPENDENT, "vgdiff": "vgdiff_std"},
                            inplace=True)
            all_dfs.append(means_df)

        if name == "all":
            relevant_data = result_df[~result_df["sub_code"].isin(CHOSEN_10)]
        else:
            relevant_data = result_df[result_df["sub_code"].isin(CHOSEN_10)]

        means_df = relevant_data.groupby([data_reader.MODALITY, "visibility"]).mean().reset_index()
        means_df.loc[:, INDEPENDENT] = means_df[data_reader.MODALITY] + " " + means_df["visibility"]
        means_df = means_df[[INDEPENDENT, "vgdiff"]]
        means_df.rename(columns={"vgdiff": "vgdiff_mean"},
                        inplace=True)
        all_dfs.append(means_df)

        means_df = relevant_data.groupby([data_reader.MODALITY, "visibility"]).std().reset_index()
        means_df.loc[:, INDEPENDENT] = means_df[data_reader.MODALITY] + " " + means_df["visibility"]
        means_df = means_df[[INDEPENDENT, "vgdiff"]]
        means_df.rename(columns={"vgdiff": "vgdiff_std"},
                        inplace=True)
        all_dfs.append(means_df)

        merged_df = pd.concat(all_dfs)
        merged_df.to_csv(os.path.join(qc_res_path, f"vg_difficulty_per_vis_{name}_descriptives.csv"),
                         index=False)

    # Part4: replay_fa_wfillers_per_cat.csv
    fa_cols = ["LOC_face_w_fillers_farate", "LOC_obj_w_fillers_farate"]
    fa_counts = ["LOC_face_w_fillers_fa_count", "LOC_obj_w_fillers_fa_count"]
    df_list = list()
    for i in range(len(fa_cols)):
        # This is TARGET TYPE (in category)
        farate_col = fa_cols[i]
        data_subdf = data[["sub_code", data_reader.MODALITY, farate_col, fa_counts[i]]]
        data_subdf.rename(columns={farate_col: "replay_farate_wfillers", fa_counts[i]: "replay_fa_count"}, inplace=True)
        obj_type = "obj" if "obj" in farate_col else "face"
        data_subdf["category"] = obj_type
        df_list.append(data_subdf)
    data_subdf = pd.concat(df_list)
    data_subdf.to_csv(os.path.join(qc_res_path, f"replay_fa_wfillers_per_cat.csv"), index=False)

    # PLOT thingy
    for name in ["all", "chosen10"]:
        if name == "all":
            relevant_data = data[~data["sub_code"].isin(CHOSEN_10)]
        else:
            relevant_data = data[data["sub_code"].isin(CHOSEN_10)]

        colors = ["#003544", "#ad501d"]  # Face, Object
        boxplotter.plot(data=relevant_data,
                        data_col_order=['LOC_face_w_fillers_farate', 'LOC_obj_w_fillers_farate'],
                        data_name_order={'LOC_face_w_fillers_farate': "Face",
                                         'LOC_obj_w_fillers_farate': "Object"},
                        scatter=False, raincloud=True, sub_line=[[0, 1]],
                        color_list=colors,
                        plot_title=f"False Alarm Rate per target", plot_x_label="Target",
                        plot_y_label=f"False Alarm Rate", save_plot=True, save_path=qc_res_path,
                        save_name=f"replay_fa_fillers_per_cat_{name}",
                        sub_folder="./", custom_ylim=0.1, skip=0.02)

    # Without a variable
    for name in ["all", "chosen10"]:
        all_dfs = []
        for indep in [data_reader.MODALITY, "category"]:
            if name == "all":
                relevant_data = data_subdf[~data_subdf["sub_code"].isin(CHOSEN_10)]
            else:
                relevant_data = data_subdf[data_subdf["sub_code"].isin(CHOSEN_10)]
            means_df = relevant_data.groupby([indep]).mean().reset_index()
            means_df = means_df[[indep, "replay_farate_wfillers", "replay_fa_count"]]
            means_df.rename(columns={indep: INDEPENDENT, "replay_farate_wfillers": "replay_farate_wfillers_mean", "replay_fa_count": "replay_fa_count_mean"},
                            inplace=True)
            all_dfs.append(means_df)

            means_df = relevant_data.groupby([indep]).std().reset_index()
            means_df = means_df[[indep, "replay_farate_wfillers", "replay_fa_count"]]
            means_df.rename(columns={indep: INDEPENDENT, "replay_farate_wfillers": "replay_farate_wfillers_std",
                                     "replay_fa_count": "replay_fa_count_std"},
                            inplace=True)
            all_dfs.append(means_df)

        for combo in [[data_reader.MODALITY, "category"]]:
            if name == "all":
                relevant_data = data_subdf[~data_subdf["sub_code"].isin(CHOSEN_10)]
            else:
                relevant_data = data_subdf[data_subdf["sub_code"].isin(CHOSEN_10)]

            means_df = relevant_data.groupby(combo).mean().reset_index()
            if len(combo) == 2:
                means_df.loc[:, INDEPENDENT] = means_df[combo[0]] + " " + means_df[combo[1]]
            else:
                means_df.loc[:, INDEPENDENT] = means_df[combo[0]] + " " + means_df[combo[1]] + " " + means_df[combo[2]]
            means_df = means_df[[INDEPENDENT, "replay_farate_wfillers", "replay_fa_count"]]
            means_df.rename(columns={"replay_farate_wfillers": "replay_farate_wfillers_mean",
                                     "replay_fa_count": "replay_fa_count_mean"},
                            inplace=True)
            all_dfs.append(means_df)

            means_df = relevant_data.groupby(combo).std().reset_index()
            if len(combo) == 2:
                means_df.loc[:, INDEPENDENT] = means_df[combo[0]] + " " + means_df[combo[1]]
            else:
                means_df.loc[:, INDEPENDENT] = means_df[combo[0]] + " " + means_df[combo[1]] + " " + means_df[combo[2]]
            means_df = means_df[[INDEPENDENT, "replay_farate_wfillers", "replay_fa_count"]]
            means_df.rename(columns={"replay_farate_wfillers": "replay_farate_wfillers_std",
                                     "replay_fa_count": "replay_fa_count_std"},inplace=True)
            all_dfs.append(means_df)

        merged_df = pd.concat(all_dfs)
        merged_df.to_csv(os.path.join(qc_res_path, f"replay_fa_wfillers_per_cat_{name}_descriptives.csv"),
                         index=False)

    # Part4.1: replay_fa_wofillers_per_left_right.csv
    fa_cols = ["LOC_Left_wo_fillers_farate", "LOC_Right_wo_fillers_farate"]
    fa_counts = ["LOC_Left_wo_fillers_fa_count", "LOC_Right_wo_fillers_fa_count"]
    df_list = list()
    for i in range(len(fa_cols)):
        # This is TARGET TYPE (in category)
        farate_col = fa_cols[i]
        data_subdf = data[["sub_code", data_reader.MODALITY, farate_col, fa_counts[i]]]
        data_subdf.rename(columns={farate_col: "replay_farate_wofillers", fa_counts[i]: "replay_fa_count"},
                          inplace=True)
        obj_type = "Left" if "Left" in farate_col else "Right"
        data_subdf["location"] = obj_type
        df_list.append(data_subdf)
    data_subdf = pd.concat(df_list)
    data_subdf.to_csv(os.path.join(qc_res_path, f"replay_fa_wofillers_per_left_right.csv"), index=False)

    # PLOT thingy
    for name in ["all", "chosen10"]:
        if name == "all":
            relevant_data = data[~data["sub_code"].isin(CHOSEN_10)]
        else:
            relevant_data = data[data["sub_code"].isin(CHOSEN_10)]

        colors = ["#003544", "#ad501d"]  # Face, Object
        boxplotter.plot(data=relevant_data,
                        data_col_order=['LOC_Left_wo_fillers_farate', 'LOC_Right_wo_fillers_farate'],
                        data_name_order={'LOC_Left_wo_fillers_farate': "Left",
                                         'LOC_Right_wo_fillers_farate': "Right"},
                        scatter=False, raincloud=True, sub_line=[[0, 1]],
                        color_list=colors,
                        plot_title=f"False Alarm Rate per target", plot_x_label="Target",
                        plot_y_label=f"False Alarm Rate", save_plot=True, save_path=qc_res_path,
                        save_name=f"replay_fa_wo_fillers_per_left_right_{name}",
                        sub_folder="./", custom_ylim=0.1, skip=0.02)

    # Without a variable
    for name in ["all", "chosen10"]:
        all_dfs = []
        for indep in [data_reader.MODALITY, "location"]:
            if name == "all":
                relevant_data = data_subdf[~data_subdf["sub_code"].isin(CHOSEN_10)]
            else:
                relevant_data = data_subdf[data_subdf["sub_code"].isin(CHOSEN_10)]
            means_df = relevant_data.groupby([indep]).mean().reset_index()
            means_df = means_df[[indep, "replay_farate_wofillers", "replay_fa_count"]]
            means_df.rename(columns={indep: INDEPENDENT, "replay_farate_wofillers": "replay_farate_wofillers_mean",
                                     "replay_fa_count": "replay_fa_count_mean"},
                            inplace=True)
            all_dfs.append(means_df)

            means_df = relevant_data.groupby([indep]).std().reset_index()
            means_df = means_df[[indep, "replay_farate_wofillers", "replay_fa_count"]]
            means_df.rename(columns={indep: INDEPENDENT, "replay_farate_wofillers": "replay_farate_wofillers_std",
                                     "replay_fa_count": "replay_fa_count_std"},
                            inplace=True)
            all_dfs.append(means_df)

        for combo in [[data_reader.MODALITY, "location"]]:
            if name == "all":
                relevant_data = data_subdf[~data_subdf["sub_code"].isin(CHOSEN_10)]
            else:
                relevant_data = data_subdf[data_subdf["sub_code"].isin(CHOSEN_10)]

            means_df = relevant_data.groupby(combo).mean().reset_index()
            if len(combo) == 2:
                means_df.loc[:, INDEPENDENT] = means_df[combo[0]] + " " + means_df[combo[1]]
            else:
                means_df.loc[:, INDEPENDENT] = means_df[combo[0]] + " " + means_df[combo[1]] + " " + means_df[
                    combo[2]]
            means_df = means_df[[INDEPENDENT, "replay_farate_wofillers", "replay_fa_count"]]
            means_df.rename(columns={"replay_farate_wofillers": "replay_farate_wofillers_mean",
                                     "replay_fa_count": "replay_fa_count_mean"},
                            inplace=True)
            all_dfs.append(means_df)

            means_df = relevant_data.groupby(combo).std().reset_index()
            if len(combo) == 2:
                means_df.loc[:, INDEPENDENT] = means_df[combo[0]] + " " + means_df[combo[1]]
            else:
                means_df.loc[:, INDEPENDENT] = means_df[combo[0]] + " " + means_df[combo[1]] + " " + means_df[
                    combo[2]]
            means_df = means_df[[INDEPENDENT, "replay_farate_wofillers", "replay_fa_count"]]
            means_df.rename(columns={"replay_farate_wofillers": "replay_farate_wofillers_std",
                                     "replay_fa_count": "replay_fa_count_std"}, inplace=True)
            all_dfs.append(means_df)

        merged_df = pd.concat(all_dfs)
        merged_df.to_csv(os.path.join(qc_res_path, f"replay_fa_wofillers_per_left_right_{name}_descriptives.csv"),
                         index=False)

    # Part4.2: replay_fa_wofillers_per_top_bottom.csv
    fa_cols = ["LOC_Top_wo_fillers_farate", "LOC_Bottom_wo_fillers_farate"]
    fa_counts = ["LOC_Top_wo_fillers_fa_count", "LOC_Bottom_wo_fillers_fa_count"]
    df_list = list()
    for i in range(len(fa_cols)):
        # This is TARGET TYPE (in category)
        farate_col = fa_cols[i]
        data_subdf = data[["sub_code", data_reader.MODALITY, farate_col, fa_counts[i]]]
        data_subdf.rename(columns={farate_col: "replay_farate_wofillers", fa_counts[i]: "replay_fa_count"},
                          inplace=True)
        obj_type = "Top" if "Top" in farate_col else "Bottom"
        data_subdf["location"] = obj_type
        df_list.append(data_subdf)
    data_subdf = pd.concat(df_list)
    data_subdf.to_csv(os.path.join(qc_res_path, f"replay_fa_wofillers_per_top_bottom.csv"), index=False)

    # PLOT thingy
    for name in ["all", "chosen10"]:
        if name == "all":
            relevant_data = data[~data["sub_code"].isin(CHOSEN_10)]
        else:
            relevant_data = data[data["sub_code"].isin(CHOSEN_10)]

        colors = ["#003544", "#ad501d"]  # Face, Object
        boxplotter.plot(data=relevant_data,
                        data_col_order=['LOC_Top_wo_fillers_farate', 'LOC_Bottom_wo_fillers_farate'],
                        data_name_order={'LOC_Top_wo_fillers_farate': "Top",
                                         'LOC_Bottom_wo_fillers_farate': "Bottom"},
                        scatter=False, raincloud=True, sub_line=[[0, 1]],
                        color_list=colors,
                        plot_title=f"False Alarm Rate per target", plot_x_label="Target",
                        plot_y_label=f"False Alarm Rate", save_plot=True, save_path=qc_res_path,
                        save_name=f"replay_fa_wo_fillers_per_top_bottom_{name}",
                        sub_folder="./", custom_ylim=0.1, skip=0.02)

    # Without a variable
    for name in ["all", "chosen10"]:
        all_dfs = []
        for indep in [data_reader.MODALITY, "location"]:
            if name == "all":
                relevant_data = data_subdf[~data_subdf["sub_code"].isin(CHOSEN_10)]
            else:
                relevant_data = data_subdf[data_subdf["sub_code"].isin(CHOSEN_10)]
            means_df = relevant_data.groupby([indep]).mean().reset_index()
            means_df = means_df[[indep, "replay_farate_wofillers", "replay_fa_count"]]
            means_df.rename(columns={indep: INDEPENDENT, "replay_farate_wofillers": "replay_farate_wofillers_mean",
                                     "replay_fa_count": "replay_fa_count_mean"},
                            inplace=True)
            all_dfs.append(means_df)

            means_df = relevant_data.groupby([indep]).std().reset_index()
            means_df = means_df[[indep, "replay_farate_wofillers", "replay_fa_count"]]
            means_df.rename(columns={indep: INDEPENDENT, "replay_farate_wofillers": "replay_farate_wofillers_std",
                                     "replay_fa_count": "replay_fa_count_std"},
                            inplace=True)
            all_dfs.append(means_df)

        for combo in [[data_reader.MODALITY, "location"]]:
            if name == "all":
                relevant_data = data_subdf[~data_subdf["sub_code"].isin(CHOSEN_10)]
            else:
                relevant_data = data_subdf[data_subdf["sub_code"].isin(CHOSEN_10)]

            means_df = relevant_data.groupby(combo).mean().reset_index()
            if len(combo) == 2:
                means_df.loc[:, INDEPENDENT] = means_df[combo[0]] + " " + means_df[combo[1]]
            else:
                means_df.loc[:, INDEPENDENT] = means_df[combo[0]] + " " + means_df[combo[1]] + " " + means_df[
                    combo[2]]
            means_df = means_df[[INDEPENDENT, "replay_farate_wofillers", "replay_fa_count"]]
            means_df.rename(columns={"replay_farate_wofillers": "replay_farate_wofillers_mean",
                                     "replay_fa_count": "replay_fa_count_mean"},
                            inplace=True)
            all_dfs.append(means_df)

            means_df = relevant_data.groupby(combo).std().reset_index()
            if len(combo) == 2:
                means_df.loc[:, INDEPENDENT] = means_df[combo[0]] + " " + means_df[combo[1]]
            else:
                means_df.loc[:, INDEPENDENT] = means_df[combo[0]] + " " + means_df[combo[1]] + " " + means_df[
                    combo[2]]
            means_df = means_df[[INDEPENDENT, "replay_farate_wofillers", "replay_fa_count"]]
            means_df.rename(columns={"replay_farate_wofillers": "replay_farate_wofillers_std",
                                     "replay_fa_count": "replay_fa_count_std"}, inplace=True)
            all_dfs.append(means_df)

        merged_df = pd.concat(all_dfs)
        merged_df.to_csv(os.path.join(qc_res_path, f"replay_fa_wofillers_per_top_bottom_{name}_descriptives.csv"),
                         index=False)

    # Part5: plot moving average and difficulty - VALID ONLY
    file_name = f'subject_beh.pickle'
    fl = open(os.path.join(qc_res_path, file_name), 'rb')
    subs_data = pickle.load(fl)
    fl.close()

    for modality in modality_list:
        my_dict = dict()
        # select ALL BUT CHOSEN10
        for k in subs_data.keys():
            if k not in list(data["sub_code"]):
                continue
            if k in CHOSEN_10:
                continue
            if modality != "both" and subs_data[k].mod != modality:
                continue
            my_dict[k] = dict()
            my_dict[k]["GAME"] = subs_data[k].full.SessDetails.ProbeDet[subs_data[k].full.SessDetails.ProbeDet['activeWorldID'] != -1]
            my_dict[k]["REPLAY"] = subs_data[k].full.SessDetails.LocalizerDetFullCorrected[subs_data[k].full.SessDetails.LocalizerDetFullCorrected["type"] != 'LOCALIZER_FILLER']
            my_dict[k]["REPLAY"][general_analysis.REPLAY_TARGET] = my_dict[k]["REPLAY"].apply(
                lambda row: general_analysis.match_replay_level_to_target(row, subs_data[k].full.replay_targets), axis=1)
        calculate_hitrate_everything(subs_data, data, modality, save=True, save_path=os.path.join(qc_res_path, "all_valid", modality))
        calculate_fa_everything(subs_data, data, modality, save=True, save_path=os.path.join(qc_res_path, "all_valid", modality))
        # ANALYSIS OF DIFFICULTY/PERFORMANCE ACROSS THE GAME
        diff_perf_analysis.analyze_diff_perf(my_dict, save=True, save_path=os.path.join(qc_res_path, "all_valid", modality))
        # ANALYSIS OF RESPONSE TYPES ACROSS
        response_type_analysis.analyze_response_types(my_dict, save=True, plot=True, save_path=os.path.join(qc_res_path, "all_valid", modality), seperate_mods=(modality=="both"))
        # Moving avg analysis
        response_type_analysis.mov_avg_stats(my_dict, save=True, save_path=os.path.join(qc_res_path, "all_valid", modality, "BEH_analysis"),
                                             plot=True, colors={"GAME": 'orange', "REPLAY": 'tab:green'})

    # Part6: plot moving average and difficulty - CHOSEN 10
    if not is_tau:
        for modality in modality_list:
            my_dict = dict()
            for k in CHOSEN_10:
                if modality != "both" and subs_data[k].mod != modality:
                    continue
                my_dict[k] = dict()
                my_dict[k]["GAME"] = subs_data[k].full.SessDetails.ProbeDet[subs_data[k].full.SessDetails.ProbeDet['activeWorldID'] != -1]
                my_dict[k]["REPLAY"] = subs_data[k].full.SessDetails.LocalizerDetFullCorrected[subs_data[k].full.SessDetails.LocalizerDetFullCorrected["type"] != 'LOCALIZER_FILLER']
                my_dict[k]["REPLAY"][general_analysis.REPLAY_TARGET] = my_dict[k]["REPLAY"].apply(
                    lambda row: general_analysis.match_replay_level_to_target(row, subs_data[k].full.replay_targets), axis=1)
            # ANALYSIS OF DIFFICULTY/PERFORMANCE ACROSS THE GAME
            diff_perf_analysis.analyze_diff_perf(my_dict, save=True, save_path=os.path.join(qc_res_path, "chosen_10", modality))
            # ANALYSIS OF RESPONSE TYPES ACROSS
            response_type_analysis.analyze_response_types(my_dict, save=True, plot=True, save_path=os.path.join(qc_res_path, "chosen_10", modality))

            response_type_analysis.mov_avg_stats(my_dict, save=True, save_path=os.path.join(qc_res_path, "chosen_10", modality, "BEH_analysis"),
                                                 plot=True, colors={"GAME": 'orange', "REPLAY": 'tab:green'})

    # Part7: plot moving average and difficulty - INVALID ONLY
    my_dict = dict()
    for k in subs_data.keys():
        if k in list(data["sub_code"]):
            continue
        if subs_data[k].full.SessDetails.LocalizerDetFullCorrected is None:
                continue

        my_dict[k] = dict()
        my_dict[k]["GAME"] = subs_data[k].full.SessDetails.ProbeDet[subs_data[k].full.SessDetails.ProbeDet['activeWorldID'] != -1]
        my_dict[k]["REPLAY"] = subs_data[k].full.SessDetails.LocalizerDetFullCorrected[subs_data[k].full.SessDetails.LocalizerDetFullCorrected["type"] != 'LOCALIZER_FILLER']
    # ANALYSIS OF DIFFICULTY/PERFORMANCE ACROSS THE GAME
    diff_perf_analysis.analyze_diff_perf(my_dict, save=True, save_path=os.path.join(qc_res_path, "all_invalid", "both"))

    response_type_analysis.mov_avg_stats(my_dict, save=True, save_path=os.path.join(qc_res_path, "all_invalid", "both", "BEH_analysis"),
                                         plot=True, colors={"GAME": 'orange', "REPLAY": 'tab:green'})
    return


def ml_analysis(root_folder, is_tau=False):
    qc_res_path = data_saver.create_hpc_quality_checks(root_folder)  # the result folder on the HPC (DMT QC folder)
    data = pd.read_csv(os.path.join(qc_res_path, FILENAME))
    data = data[data[quality_checks_criteria.VALID] == True]  # filter valid subs only
    file_name = f'subject_beh.pickle'
    fl = open(os.path.join(qc_res_path, file_name), 'rb')
    subs_data = pickle.load(fl)
    fl.close()

    # select ALL BUT CHOSEN10
    for mod in [data_reader.FMRI, data_reader.MEEG]:
        my_dict = dict()
        for k in subs_data.keys():
            if k not in list(data["sub_code"]):
                continue
            if not is_tau:
                if k in CHOSEN_10:
                    continue
            if mod != "both" and subs_data[k].mod != mod:
                continue
            my_dict[k] = subs_data[k]
        # Part8 - run Machine Learning Models
        vg_features_analysis.analyze_vg_effects_on_vis(my_dict, save_path=qc_res_path, kernel='linear', name=mod, normalize=True, alt=True)
    #vg_features_analysis.analyze_vg_effects_on_vis(my_dict, save_path=qc_res_path, kernel='rbf')


def additional_stats(root_folder):
    qc_res_path = data_saver.create_hpc_quality_checks(root_folder)  # the result folder on the HPC (DMT QC folder)
    file_name = f'subject_beh.pickle'
    fl = open(os.path.join(qc_res_path, file_name), 'rb')
    subs_data = pickle.load(fl)
    fl.close()

    data = pd.read_csv(os.path.join(qc_res_path, FILENAME))
    valid_data = data[data[quality_checks_criteria.VALID] == True]  # filter valid subs only
    valid_list = list(valid_data["sub_code"])

    # Part 1 : go over ALL OF THE SUBJECTS (not only VALID), and get RTs of everyone.
    all_rts_list = list()
    for k in subs_data:
        # Start with GAME
        trials = subs_data[k].full.SessDetails.ProbeDet[subs_data[k].full.SessDetails.ProbeDet['activeWorldID'] != -1]
        rt_trials = trials[~trials["responseDT"].isna()]
        rt_trials.loc[:, "sub"] = k
        rt_trials.loc[:, "mod"] = subs_data[k].mod
        rt_trials.loc[:, "lab"] = subs_data[k].lab
        rt_trials.loc[:, "condition"] = "GAME"
        rt_trials.loc[:, quality_checks_criteria.VALID] = k in valid_list
        all_rts_list.append(rt_trials)  # filtered table (row per trial, all relevant trials)

        # Take Replay as well, drop eliminated levels
        # Some participants dont have replay, keep them in as well
        if subs_data[k].full.SessDetails.LocalizerDetFullCorrected is None:
            continue

        trials = subs_data[k].full.SessDetails.LocalizerDetFullCorrected[subs_data[k].full.SessDetails.LocalizerDetFullCorrected["IS_LEVEL_ELIMINATED"] == 0]
        rt_trials = trials[~trials["responseDT"].isna()]
        rt_trials.loc[:, "sub"] = k
        rt_trials.loc[:, "mod"] = subs_data[k].mod
        rt_trials.loc[:, "lab"] = subs_data[k].lab
        rt_trials.loc[:, "condition"] = "REPLAY"
        rt_trials.loc[:, quality_checks_criteria.VALID] = k in valid_list
        all_rts_list.append(rt_trials)

    all_subs_df = pd.concat(all_rts_list)
    subs_means_together = all_subs_df.groupby(["sub", "mod", "lab", quality_checks_criteria.VALID]).mean().reset_index()
    subs_means_together = subs_means_together[["sub", "mod", "lab", "responseDT", quality_checks_criteria.VALID]]
    subs_means_sep = all_subs_df.groupby(["sub", "mod", "lab", "condition", quality_checks_criteria.VALID]).mean().reset_index()
    subs_means_sep = subs_means_sep[["sub", "mod", "lab", "condition", "responseDT", quality_checks_criteria.VALID]]

    pivoted_conditions = subs_means_sep.pivot(
        index=["sub", "mod", "lab", quality_checks_criteria.VALID],
        columns="condition",
        values="responseDT").reset_index()

    # Merge the pivoted DataFrame back into subs_means_together
    subs_means_together = subs_means_together.merge(pivoted_conditions, on=["sub", "mod", "lab", quality_checks_criteria.VALID], how="left")
    subs_means_together.to_csv(os.path.join(qc_res_path, f"all_subs_mean_rts.csv"), index=False)

    # Now, calculate per visibility, only for GAME
    all_subs_df = all_subs_df[all_subs_df["condition"] == "GAME"]
    subs_means_together = all_subs_df.groupby(["sub", "mod", "lab", quality_checks_criteria.VALID]).mean().reset_index()
    subs_means_together = subs_means_together[["sub", "mod", "lab", "responseDT", quality_checks_criteria.VALID]]
    subs_means_sep = all_subs_df.groupby(["sub", "mod", "lab", quality_checks_criteria.VALID, EVAL]).mean().reset_index()
    subs_means_sep = subs_means_sep[["sub", "mod", "lab", "responseDT", quality_checks_criteria.VALID, EVAL]]

    pivoted_conditions = subs_means_sep.pivot(
        index=["sub", "mod", "lab", quality_checks_criteria.VALID],
        columns=EVAL,
        values="responseDT").reset_index()

    # Merge the pivoted DataFrame back into subs_means_together
    subs_means_together = subs_means_together.merge(pivoted_conditions, on=["sub", "mod", "lab", quality_checks_criteria.VALID], how="left")
    subs_means_together.to_csv(os.path.join(qc_res_path, f"all_game_subs_mean_rts_per_vis.csv"), index=False)

    for k in subs_data:
        # Start with GAME
        trials = subs_data[k].full.SessDetails.ProbeDet[subs_data[k].full.SessDetails.ProbeDet['activeWorldID'] != -1]
        rt_trials = trials[~trials[DIFF].isna()]
        rt_trials.loc[:, "sub"] = k
        rt_trials.loc[:, "mod"] = subs_data[k].mod
        rt_trials.loc[:, "lab"] = subs_data[k].lab
        rt_trials.loc[:, "condition"] = "GAME"
        rt_trials.loc[:, quality_checks_criteria.VALID] = k in valid_list
        all_rts_list.append(rt_trials)  # filtered table (row per trial, all relevant trials)

        # Take Replay as well, drop eliminated levels
        # Some participants dont have replay, keep them in as well
        if subs_data[k].full.SessDetails.LocalizerDetFullCorrected is None:
            continue

        trials = subs_data[k].full.SessDetails.LocalizerDetFullCorrected[subs_data[k].full.SessDetails.LocalizerDetFullCorrected["IS_LEVEL_ELIMINATED"] == 0]
        rt_trials = trials[~trials[DIFF].isna()]
        rt_trials.loc[:, "sub"] = k
        rt_trials.loc[:, "mod"] = subs_data[k].mod
        rt_trials.loc[:, "lab"] = subs_data[k].lab
        rt_trials.loc[:, "condition"] = "REPLAY"
        rt_trials.loc[:, quality_checks_criteria.VALID] = k in valid_list
        all_rts_list.append(rt_trials)

    all_subs_df = pd.concat(all_rts_list)
    subs_means_together = all_subs_df.groupby(["sub", "mod", "lab", quality_checks_criteria.VALID]).mean().reset_index()
    subs_means_together = subs_means_together[["sub", "mod", "lab", quality_checks_criteria.VALID, DIFF]]
    subs_means_sep = all_subs_df.groupby(["sub", "mod", "lab", quality_checks_criteria.VALID, EVAL]).mean().reset_index()
    subs_means_sep = subs_means_sep[["sub", "mod", "lab", DIFF, quality_checks_criteria.VALID, EVAL]]

    pivoted_conditions = subs_means_sep.pivot(
        index=["sub", "mod", "lab", quality_checks_criteria.VALID],
        columns=EVAL,
        values=DIFF).reset_index()

    # Merge the pivoted DataFrame back into subs_means_together
    subs_means_together = subs_means_together.merge(pivoted_conditions, on=["sub", "mod", "lab", quality_checks_criteria.VALID], how="left")
    subs_means_together.to_csv(os.path.join(qc_res_path, f"all_game_subs_mean_difficulty_per_vis.csv"), index=False)

    # Part 2 - accuracy
    # Accuracy is (TP + TN) / (TP + TN + FP + FN)
    # Because we don have TN directly, we remove all of the false positives (false alarms) from the actual no stimulus
    data["accuracy_game"] = (data["GAME_hits"] + data["GAME_probes_blank"] - data["GAME_fas"]) / (data["GAME_probes"])
    data["accuracy_replay"] = (data["LOC_hits"] + data["LOC_stim(F/O/B)"] - data["LOC_target_stim(F/O)"] - data["LOC_fas_nofillers"]) / (data["LOC_stim(F/O/B)"])
    data["accuracy_overall"] = (data["GAME_hits"] + data["GAME_probes_blank"] - data["GAME_fas"] + data["LOC_hits"] + data["LOC_stim(F/O/B)"] - data["LOC_target_stim(F/O)"] - data["LOC_fas_nofillers"]) / (data["LOC_stim(F/O/B)"] + data["GAME_probes"])
    data.to_csv(os.path.join(qc_res_path, f"quality_checks_w_accuracy.csv"), index=False)
    return


def debugging_session(root_folder):
    qc_res_path = data_saver.create_hpc_quality_checks(root_folder)  # the result folder on the HPC (DMT QC folder)
    # Part5: plot moving average and difficulty - VALID ONLY
    file_name = f'subject_beh.pickle'
    fl = open(os.path.join(qc_res_path, file_name), 'rb')
    subs_data = pickle.load(fl)
    fl.close()

    data = pd.read_csv(os.path.join(qc_res_path, FILENAME))
    data = data[data[quality_checks_criteria.VALID] == True]  # filter valid subs only
    #calculate_fa_everything(subs_data, data, "both", save=True, save_path=qc_res_path)
    #calculate_fa_wfillers_stats(subs_data, data, qc_res_path)

    #calculate_difficulty_stats(subs_data, data, qc_res_path)
    #calculate_fa_wfillers_stats(subs_data, data, qc_res_path)

    modality_list = ["both", data_reader.FMRI, data_reader.MEEG]

    all_subs_df_list = []

    for modality in modality_list:
        my_dict = dict()
        # select ALL BUT CHOSEN10
        for k in subs_data.keys():
            if k not in list(data["sub_code"]):
                continue
            if k in CHOSEN_10:
                continue
            if modality != "both" and subs_data[k].mod != modality:
                continue
            my_dict[k] = dict()
            my_dict[k]["GAME"] = subs_data[k].full.SessDetails.ProbeDet[
                subs_data[k].full.SessDetails.ProbeDet['activeWorldID'] != -1]
            my_dict[k]["REPLAY"] = subs_data[k].full.SessDetails.LocalizerDetFullCorrected[
                subs_data[k].full.SessDetails.LocalizerDetFullCorrected["type"] != 'LOCALIZER_FILLER']
            my_dict[k]["REPLAY"][general_analysis.REPLAY_TARGET] = my_dict[k]["REPLAY"].apply(
                lambda row: general_analysis.match_replay_level_to_target(row, subs_data[k].full.replay_targets),
                axis=1)
            my_dict[k]["REPLAY_TARGETS"] = subs_data[k].full.replay_targets

        # ANALYSIS OF RESPONSE TYPES ACROSS
        #response_type_analysis.analyze_response_types(my_dict, save=True, plot=True,
        #                                              save_path=os.path.join(qc_res_path, "all_valid", modality), seperate_mods=False)
        diff_perf_analysis.analyze_diff_perf(my_dict, save=True, save_path=os.path.join(qc_res_path, "all_valid", modality), unseen_only=True)
        diff_perf_analysis.analyze_diff_perf(my_dict, save=True, save_path=os.path.join(qc_res_path, "all_valid", modality), seen_only=True)
        diff_perf_analysis.analyze_diff_perf(my_dict, save=True, save_path=os.path.join(qc_res_path, "all_valid", modality))

    exit()
    for k in subs_data.keys():
        if k not in list(data["sub_code"]):
            continue
        if k in CHOSEN_10:
            continue

        # REPLAY, THIS TIME ONLY FILLERS!
        replay_data = subs_data[k].full.SessDetails.LocalizerDetFullCorrected
        replay_data.loc[:, "sub"] = k
        replay_data.loc[:, "mod"] = subs_data[k].mod
        replay_data.loc[:, "condition"] = "REPLAY"
        stim_present, stim_absent, stim_nontarget = probe_data_go_nogo(replay_data, subs_data[k].full.replay_targets)
        stim_absent = pd.concat([stim_absent, stim_nontarget])
        stim_absent = stim_absent[stim_absent["type"] == 'LOCALIZER_FILLER']
        stim_absent = stim_absent.loc[stim_absent["IS_LEVEL_ELIMINATED"] == 0, :]
        stim_absent.loc[:, "fa"] = False
        stim_absent.loc[:, TARGET_TYPE] = stim_absent[TARGET_TYPE]
        stim_absent.loc[(stim_absent[EVAL].notnull()) & (stim_absent[EVAL].str.contains(FALSEPOSITIVE)), "fa"] = True
        stim_absent = stim_absent.loc[:, ["sub", "mod", "fa", TARGET_TYPE]]
        all_subs_df_list.append(stim_absent)

    all_df = pd.concat(all_subs_df_list)
    means = all_df.groupby(["sub", "TargetType"]).mean().reset_index()
    stds = all_df.groupby(["sub", "TargetType"]).std().reset_index()
    means["std"] = stds["fa"]
    double_means = means.groupby(["TargetType"]).mean().reset_index()
    stds = means.groupby(["TargetType"]).std().reset_index()
    double_means["std"] = stds["fa"]
    total_thing = all_df.groupby(["sub"]).mean()
    total_mean = total_thing["fa"].mean()
    total_std = total_thing["fa"].std()
    #total_mean["std"] = total_std
    double_means.loc[len(double_means.index)] = ['', total_mean, total_std]
    all_df.to_csv(os.path.join(qc_res_path, "all_valid", f"fa_giant_replay_OnlyFillers.csv"), index=False)
    means.to_csv(os.path.join(qc_res_path, "all_valid", f"fa_giant_replay_OnlyFillers_sub_stats.csv"), index=False)
    double_means.to_csv(os.path.join(qc_res_path, "all_valid", f"fa_giant_replay_OnlyFillers_stats.csv"), index=False)


def diff_perf_additional_plot(root_folder):
    qc_res_path = data_saver.create_hpc_quality_checks(root_folder)

    seen_names = ["seen", "unseen"]
    modality_list = [data_reader.FMRI, data_reader.MEEG]

    for modality in modality_list:
        data_files = []
        for seen in seen_names:
            file_names = [f"diff_perf_across_game_diff_{seen}.csv", f"diff_perf_across_game_diff_sma_13_{seen}.csv", f"diff_perf_across_game_perf_{seen}.csv", f"diff_perf_across_game_perf_sma_13_{seen}.csv"]
            for file_n in file_names:
                data_f = pd.read_csv(os.path.join(qc_res_path, "all_valid", modality, "BEH_analysis", diff_perf_analysis.DIFF_V_PERF, file_n))
                data_files.append(data_f)

        lineplotter.plot_avg_line(title="Moving Average Difficulty and Performance Across Game",
                                  trial_df_list=data_files, avg_col_list=[diff_perf_analysis.AVG for _ in range(8)],
                                  se_col_list=[diff_perf_analysis.SE if i % 2 == 0 else None for i in range(8)],
                                  label_list=["Average Difficulty Seen", "SMA Difficulty Seen", "Average Performance Seen", "SMA Performance Seen", "Average Difficulty Unseen", "SMA Difficulty Unseen", "Average Performance Unseen", "SMA Performance Unseen"],
                                  color_list=["#a2273e", "#821f32", "#415c75", "#32475b", "#ef738a", "#ea4b69", "#91a3b4", "#6d859b"],
                                  y_name=f"Moving Average", save=True, save_name=f"diff_perf_across_game_seen_v_unseen_modalities",
                                  save_path=os.path.join(qc_res_path, "all_valid", modality, "BEH_analysis"), sub_folder="performance_v_difficulty_extra")

    return


if __name__ == "__main__":
    check_data(root_folder=r"/mnt/beegfs/XNAT/COGITATE")
    exit()
    #additional_stats(root_folder=r"/mnt/beegfs/XNAT/COGITATE")
    diff_perf_additional_plot(root_folder=r"/mnt/beegfs/XNAT/COGITATE")
    #debugging_session(root_folder=r"/mnt/beegfs/XNAT/COGITATE")
    #ml_analysis(root_folder=r"/mnt/beegfs/XNAT/COGITATE")
    exit()
    get_tables_for_further_analysis(root_folder=r"/mnt/beegfs/XNAT/COGITATE", is_tau=False)

    debugging_session(root_folder=r"/mnt/beegfs/XNAT/COGITATE")
    #additional_stats(root_folder=r"/mnt/beegfs/XNAT/COGITATE")
    #check_data(root_folder=r"/mnt/beegfs/XNAT/COGITATE/QC/v2/TAU", is_tau=True)
    #debugging_session(root_folder=r"/mnt/beegfs/XNAT/COGITATE/QC/v2/TAU")
    #get_tables_for_further_analysis(root_folder=r"/mnt/beegfs/XNAT/COGITATE/QC/v2/TAU", is_tau=True)
    #ml_analysis(root_folder=r"/mnt/beegfs/XNAT/COGITATE/QC/v2/TAU", is_tau=True)
    exit()
    #qc_res_path = data_saver.create_hpc_quality_checks(r"/mnt/beegfs/XNAT/COGITATE")
    #data_table_res = pd.read_csv(os.path.join(qc_res_path, FILENAME))  # save
    #create_subject_lists(data_table_res, qc_res_path)
    debugging_session(root_folder=r"/mnt/beegfs/XNAT/COGITATE")
    exit()
    check_data(root_folder=r"/mnt/beegfs/XNAT/COGITATE")
    split_replay_data_to_mods(root_folder=r"/mnt/beegfs/XNAT/COGITATE")
    get_tables_for_further_analysis(root_folder=r"/mnt/beegfs/XNAT/COGITATE", is_tau=False)
    ml_analysis(root_folder=r"/mnt/beegfs/XNAT/COGITATE")
    #debugging_session(root_folder=r"/mnt/beegfs/XNAT/COGITATE")
