import pickle
import numpy as np
import os
import pandas as pd
import gc
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import math
import DataParser
import ET_param_manager
import ET_qc_manager
import QualityChecker
import ET_data_extraction
import itertools
import plotter
import pycircstat
import astropy.stats
import scipy
from functools import reduce

SUBJECT_NAME = "SubjectName"
PARAMS = "params"
TRIAL_INFO = "trial_info"
ET_DATA_DICT = "et_data_dict"
TRIAL_NUMBER = "trialNumber"
TIME_IN_EPOCH = "timeInEpoch"
SUBJECT = "sub"
MODALITY = "mod"
LAB = "Lab"
VIS = "visibility"
INDEPENDENT = "Independent"
BIN_PROPORTION = "bin_proportion"
RANGE_START_RAD = "range_starts_rad"
TARGET = "is_target"
REF_ANGLE_RADIUS_SMALL = 1.5
REF_ANGLE_RADIUS_BIG = 3
FIXATION_DENSITY_XBINS = 96  # number of bins on the X axis to be calculated (width)
FIXATION_DENSITY_YBINS = 54  # number of bins on the Y axis to be calculated (height)
LEFT = "Left"
RIGHT = "Right"
TOP = "Top"
BOTTOM = "Bottom"
LOCATION = "location"
HORIZONTAL = "horizontal"
VERTICAL = "vertical"
TASK_RELEVANT = "task_relevant"

# fonts
TITLE_SIZE = 20 + 5
AXIS_SIZE = 19 + 5
TICK_SIZE = 17 + 3
LABEL_PAD = 8

STIMULUS_RELEVANCE_MAP = {"Face": "#003544", "Object": "#ad501d",
                          "Blank": (0.5, 0.5, 0.0)}
VIS_MAP = {"FalseNegative": (0.8352941176470589, 0.3686274509803922, 0.0), "TruePositive": (0.5450980392156862, 0.16862745098039217, 0.8862745098039215)}
STIMULUS_RELEVANCE_NAME_MAP = {"Face": "Face", "Object": "Object", "Blank": "Blank"}
LOCATION_NAME_MAP = {"Left": "Left", "Right": "Right", "Top": "Top", "Bottom": "Bottom"}
LOCATION_MAP = {"Left": "#003544", "Right": "#ad501d", "Top": "#003544", "Bottom": "#ad501d"}
VISIBILITY_NAME_MAP = {"FalseNegative": "Unseen", "TruePositive": "Seen", "All": "All"}
VISIBILITY_MAP = {"FalseNegative": "#578485", "TruePositive": "#933E3D",
                  "All": (0.8352941176470589, 0.3686274509803922, 0.0)}
TASK_RELEVANT_NAME_MAP = {True: "AT Task Relevant", False: "AT Task Irrelevant", "Seen": "dAT seen"}
TASK_RELEVANT_MAP = {True: "#578485", False: "#933E3D", "Seen": (0.8352941176470589, 0.3686274509803922, 0.0)}

VISIBILITY_MAP_GAME = {"FalseNegative": "#AE6BEB", "TruePositive": "#611E9E"}
VISIBILITY_MAP_REPLAY = {"FalseNegative": "#E28E4D", "TruePositive": "#AA4B00"}
MODALITY_MAP = {ET_qc_manager.FMRI: (0.8352941176470589, 0.3686274509803922, 0.0), ET_qc_manager.MEG: (0.5450980392156862, 0.16862745098039217, 0.8862745098039215)}
MODALITY_NAME_MAP = {ET_qc_manager.FMRI: "FMRI", ET_qc_manager.MEG: "MEEG"}
SACC_VISIBILITY_NAME_MAP = {"FalseNegative_Saccade": "Unseen Saccade", "TruePositive_Saccade": "Seen Saccade",
                            "FalseNegative_Microsaccade": "Unseen Micro Saccade", "TruePositive_Microsaccade": "Seen Micro Saccade"}
SACC_VISIBILITY_MAP = {"FalseNegative_Saccade": (0.8352941176470589, 0.3686274509803922, 0.0), "TruePositive_Saccade": (0.5450980392156862, 0.16862745098039217, 0.8862745098039215),
                            "FalseNegative_Microsaccade": (0.5, 0.5, 0.0), "TruePositive_Microsaccade": (0.25, 0.25, 0.0)}
SACC_VISIBILITY_MOD_NAME_MAP = {f"FalseNegative_{ET_qc_manager.FMRI}": "Unseen fMRI", f"TruePositive_{ET_qc_manager.FMRI}": "Seen fMRI",
                            f"FalseNegative_{ET_qc_manager.MEG}": "Unseen MEG", f"TruePositive_{ET_qc_manager.MEG}": "Seen MEG"}
SACC_VISIBILITY_GAME_MAP = {"FalseNegative_False": "#74b9ba", "TruePositive_False": "#188a8c",
                            "FalseNegative_True": "#e9b86a", "TruePositive_True": "#DB8906"}
SACC_VISIBILITY_GAME_NAME_MAP = {"FalseNegative_False": "Game Unseen", "TruePositive_False": "Game Seen",
                            "FalseNegative_True": "Replay Unseen", "TruePositive_True": "Replay Seen"}
SACC_VISIBILITY_MOD_MAP = {f"FalseNegative_{ET_qc_manager.FMRI}": (0.8352941176470589, 0.3686274509803922, 0.0), f"TruePositive_{ET_qc_manager.FMRI}": (0.5450980392156862, 0.16862745098039217, 0.8862745098039215),
                            f"FalseNegative_{ET_qc_manager.MEG}": (0.5, 0.5, 0.0), f"TruePositive_{ET_qc_manager.MEG}": (0.25, 0.25, 0.0)}

SACC_VISIBILITY_OBJECTS_MAP = {"FalseNegative_Faces": "#003544", "TruePositive_Faces": "#397384",
                               "FalseNegative_Objects": "#ad501d", "TruePositive_Objects": "#610f00"}
SACC_VISIBILITY_OBJECTS_NAME_MAP = {"FalseNegative_Faces": "Unseen Face", "TruePositive_Faces": "Seen Face",
                            "FalseNegative_Objects": "Unseen Object", "TruePositive_Objects": "Seen Object"}
SACC_VISIBILITY_LOCATION_MAP = {"FalseNegative_Left": "#003544", "TruePositive_Left": "#397384",
                               "FalseNegative_Right": "#ad501d", "TruePositive_Right": "#610f00",
                                "FalseNegative_Top": "#003544", "TruePositive_Top": "#397384",
                                "FalseNegative_Bottom": "#ad501d", "TruePositive_Bottom": "#610f00"}
SACC_VISIBILITY_LOCATION_NAME_MAP = {"FalseNegative_Left": "Unseen Left", "TruePositive_Left": "Seen Left",
                            "FalseNegative_Right": "Unseen Right", "TruePositive_Right": "Seen Right",
                                     "FalseNegative_Top": "Unseen Top", "TruePositive_Top": "Seen Top",
                                     "FalseNegative_Bottom": "Unseen Bottom", "TruePositive_Bottom": "Seen Bottom"}
SACC_STIMULUS_NAME_MAP = {"Face_Saccade": "Face Saccade", "Object_Saccade": "Object Saccade",
                          "Face_Microsaccade": "Face Micro Saccade", "Object_Microsaccade": "Object Micro Saccade"}
SACC_STIMULUS_MAP = {"Face_Saccade": (0.8352941176470589, 0.3686274509803922, 0.0), "Object_Saccade": (0.5450980392156862, 0.16862745098039217, 0.8862745098039215),
                          "Face_Microsaccade": (0.5, 0.5, 0.0), "Object_Microsaccade": (0.25, 0.25, 0.0)}
VIS_SPEED_MAP = {"FalseNegativeFast": (0.8352941176470589, 0.3686274509803922, 0.0), "TruePositiveFast": (0.5450980392156862, 0.16862745098039217, 0.8862745098039215),
                 "FalseNegativeSlow": (0.3,0.3, 0.3), "TruePositiveSlow": (0.7,0.7,0.0)}
TARGET_MAP = {True: (0.8352941176470589, 0.3686274509803922, 0.0), False: (0.5450980392156862, 0.16862745098039217, 0.8862745098039215)}
TARGET_NAME_MAP = {True: "Target", False: "Non target"}
STIM_LOCS_ANGLES = {"TopRight": 45, "TopLeft": 135, "BottomLeft": 225, "BottomRight": 315}


def get_chosen_ten():
    mod_chosen_list = list()
    for modality in ["fMRI", "MEG"]:
        mod_chosen_10 = pd.read_csv(os.path.join(f"/mnt/beegfs/XNAT/COGITATE/{modality}/phase_2/processed/bids/derivatives/qcs", "ses-v2-optimization-subs.csv"))
        mod_chosen_list.extend(list(mod_chosen_10["sub_code"]))
    return mod_chosen_list


CHOSEN_10 = get_chosen_ten()

#CHOSEN_10 = ["SA111", "SA148", "SB040", "SB069", "SB081", "SC109", "SC143", "SC160", "SD107", "SD165"]


def get_all_subs_giant_df(subs_dict, save_path, rerun_analysis_windows=False, load=False):
    if not load:
        all_df_list = list()
        for mod in subs_dict:
            print(f"Loading {mod}")
            i = 0
            for sub in subs_dict[mod]:
                fl = open(sub, 'rb')
                sub_data = pickle.load(fl)
                fl.close()
                trial_info = sub_data[TRIAL_INFO]
                sub_code = sub_data[PARAMS][SUBJECT_NAME]
                if rerun_analysis_windows:
                    trial_info = ET_data_extraction.analysis_windows(trial_info, sub_data[ET_DATA_DICT], sub_data[PARAMS], is_tobii=False)
                trial_info[SUBJECT_NAME] = sub_code
                trial_info[MODALITY] = mod
                all_df_list.append(trial_info)
                #if i == 5:
                #    break
                i += 1
        all_df = pd.concat(all_df_list)
        all_df.to_csv(os.path.join(save_path, f"everything.csv"), index=False)
    else:
        all_df = pd.read_csv(os.path.join(save_path, f"everything.csv"))
    return all_df


def pre_stim_fixation_analysis(all_subs_df, save_path):
    relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == False) & (~all_subs_df["PreStimStimDistDegs"].isna())]
    means_df = relevant_trials.groupby([SUBJECT_NAME, MODALITY, DataParser.VIS]).mean().reset_index()
    means_df = means_df[[SUBJECT_NAME, MODALITY, DataParser.VIS, "PreStimStimDistDegs"]]
    seen_df = means_df[means_df[DataParser.VIS] == "TruePositive"]
    seen_df = seen_df[[SUBJECT_NAME, "PreStimStimDistDegs", MODALITY]]
    seen_df.rename(columns={SUBJECT_NAME: "sub_code", "PreStimStimDistDegs": "vg_fixdiststim_seen"}, inplace=True)
    unseen_df = means_df[means_df[DataParser.VIS] == "FalseNegative"]
    unseen_df = unseen_df[[SUBJECT_NAME, "PreStimStimDistDegs", MODALITY]]
    unseen_df.rename(columns={SUBJECT_NAME: "sub_code", "PreStimStimDistDegs": "vg_fixdiststim_unseen"}, inplace=True)
    merged_df = pd.merge(seen_df, unseen_df, how="inner", on=["sub_code", MODALITY])
    partial_df = pd.merge(seen_df, unseen_df, how="outer", on="sub_code")
    partial_set = set(partial_df["sub_code"])
    # format to R format
    df_list = list()
    seen_df = merged_df[["sub_code", "vg_fixdiststim_seen", MODALITY]]
    seen_df[VIS] = "seen"
    seen_df.rename(columns={"vg_fixdiststim_seen": "vg_fixdiststim"}, inplace=True)
    df_list.append(seen_df)
    unseen_df = merged_df[["sub_code", "vg_fixdiststim_unseen", MODALITY]]
    unseen_df[VIS] = "unseen"
    unseen_df.rename(columns={"vg_fixdiststim_unseen": "vg_fixdiststim"}, inplace=True)
    df_list.append(unseen_df)
    merged_df = pd.concat(df_list)
    merged_df.to_csv(os.path.join(save_path, f"vg_prestim_fixdist_per_vis_all.csv"), index=False)
    all_missing_subs = list(set(all_subs_df[SUBJECT_NAME]) - set(merged_df["sub_code"]))
    missing_reason = ["Partial Data" if x in partial_set else "No Data" for x in all_missing_subs]
    missing_subs = pd.DataFrame({"subCode": all_missing_subs, "reason": missing_reason})
    missing_subs.to_csv(os.path.join(save_path, f"vg_prestim_fixdist_per_vis_missing_subs.csv"), index=False)

    # Without a variable
    for name in ["nochosen10", "chosen10"]:
        relevant_data = relevant_trials[(relevant_trials[DataParser.VIS] == "TruePositive") | (relevant_trials[DataParser.VIS] == "FalseNegative")]
        relevant_data = relevant_data[[SUBJECT_NAME, MODALITY, DataParser.VIS, "PreStimStimDistDegs"]]
        if name == "nochosen10":
            relevant_data = relevant_data[~relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = relevant_data[relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]

        list_to_combo_from = [MODALITY, DataParser.VIS]
        combos = []
        for r in range(1, len(list_to_combo_from) + 1):
            combos.append(itertools.combinations(list_to_combo_from, r))

        stats_df_list = []
        for combo in combos:
            for x in combo:
                df_to_mean = relevant_data.groupby(list(x) + [SUBJECT_NAME]).mean().reset_index()
                mean_df = df_to_mean.groupby(list(x)).mean().reset_index()
                std_df = df_to_mean.groupby(list(x)).std().reset_index()
                mean_df.loc[:, "PreStimStimDistDegs_std"] = std_df["PreStimStimDistDegs"]
                mean_df.rename(columns={"PreStimStimDistDegs": "vg_fixdiststim_mean", "PreStimStimDistDegs_std": "vg_fixdiststim_std"}, inplace=True)
                stats_df_list.append(mean_df)
        merged_df = pd.concat(stats_df_list)
        merged_df.to_csv(os.path.join(save_path, f"vg_prestim_fixdist_per_vis_{name}_descriptives.csv"), index=False)

    return


def pre_stim_blink_num_analysis(all_subs_df, save_path):
    relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == False)]
    # because if there are 0 saccades in pre stim we want to mean WITH it, fill na with zeros
    relevant_trials = relevant_trials.fillna(0)
    means_df = relevant_trials.groupby([SUBJECT_NAME, MODALITY, DataParser.VIS]).mean().reset_index()
    means_df = means_df[[SUBJECT_NAME, DataParser.VIS, "PreStimNumBlinks", MODALITY]]
    seen_df = means_df[means_df[DataParser.VIS] == "TruePositive"]
    seen_df = seen_df[[SUBJECT_NAME, "PreStimNumBlinks", MODALITY]]
    seen_df.rename(columns={SUBJECT_NAME: "sub_code", "PreStimNumBlinks": "vg_numblinks_seen"}, inplace=True)
    unseen_df = means_df[means_df[DataParser.VIS] == "FalseNegative"]
    unseen_df = unseen_df[[SUBJECT_NAME, "PreStimNumBlinks", MODALITY]]
    unseen_df.rename(columns={SUBJECT_NAME: "sub_code", "PreStimNumBlinks": "vg_numblinks_unseen"}, inplace=True)
    merged_df = pd.merge(seen_df, unseen_df, how="inner", on=["sub_code", MODALITY])
    partial_df = pd.merge(seen_df, unseen_df, how="outer", on="sub_code")
    partial_set = set(partial_df["sub_code"])

    # format to R format
    df_list = list()
    seen_df = merged_df[["sub_code", "vg_numblinks_seen", MODALITY]]
    seen_df[VIS] = "seen"
    seen_df.rename(columns={"vg_numblinks_seen": "vg_numblinks"}, inplace=True)
    df_list.append(seen_df)

    unseen_df = merged_df[["sub_code", "vg_numblinks_unseen", MODALITY]]
    unseen_df[VIS] = "unseen"
    unseen_df.rename(columns={"vg_numblinks_unseen": "vg_numblinks"}, inplace=True)
    df_list.append(unseen_df)
    merged_df = pd.concat(df_list)
    merged_df.to_csv(os.path.join(save_path, f"vg_prestim_blinks_per_vis_all.csv"), index=False)
    all_missing_subs = list(set(all_subs_df[SUBJECT_NAME]) - set(merged_df["sub_code"]))
    missing_reason = ["Partial Data" if x in partial_set else "No Data" for x in all_missing_subs]
    missing_subs = pd.DataFrame({"subCode": all_missing_subs, "reason": missing_reason})
    missing_subs.to_csv(os.path.join(save_path, f"vg_prestim_blinks_per_vis_missing_subs.csv"), index=False)

    # Without a variable
    for name in ["nochosen10", "chosen10"]:
        all_dfs = []
        relevant_data = relevant_trials[(relevant_trials[DataParser.VIS] == "TruePositive") | (relevant_trials[DataParser.VIS] == "FalseNegative")]
        if name == "nochosen10":
            relevant_data = relevant_data[~relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = relevant_data[relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]
        relevant_data = relevant_data[[SUBJECT_NAME, MODALITY, DataParser.VIS, "PreStimNumBlinks"]]

        list_to_combo_from = [MODALITY, DataParser.VIS]
        combos = []
        for r in range(1, len(list_to_combo_from) + 1):
            combos.append(itertools.combinations(list_to_combo_from, r))

        stats_df_list = []
        for combo in combos:
            for x in combo:
                df_to_mean = relevant_data.groupby(list(x) + [SUBJECT_NAME]).mean().reset_index()
                mean_df = df_to_mean.groupby(list(x)).mean().reset_index()
                std_df = df_to_mean.groupby(list(x)).std().reset_index()
                mean_df.loc[:, "PreStimNumBlinks_std"] = std_df["PreStimNumBlinks"]
                mean_df.rename(columns={"PreStimNumBlinks": "vg_numblinks_mean", "PreStimNumBlinks_std": "vg_numblinks_std"}, inplace=True)
                stats_df_list.append(mean_df)
        merged_df = pd.concat(stats_df_list)
        merged_df.to_csv(os.path.join(save_path, f"vg_prestim_blink_per_vis_{name}_descriptives.csv"), index=False)
    return


def pre_stim_saccade_num_analysis(all_subs_df, save_path, w_micro=True):
    if w_micro:
        w_micro_str = "withmicro"
    else:
        w_micro_str = "womicro"

    relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == False)]
    # because if there are 0 saccades in pre stim we want to mean WITH it, fill na with zeros
    relevant_trials = relevant_trials.fillna(0)
    means_df = relevant_trials.groupby([SUBJECT_NAME, MODALITY, DataParser.VIS]).mean().reset_index()
    means_df = means_df[[SUBJECT_NAME, DataParser.VIS, "PreStimMicroNumSaccs", "PreStimNumSaccs", MODALITY]]
    seen_df = means_df[means_df[DataParser.VIS] == "TruePositive"]
    seen_df = seen_df[[SUBJECT_NAME, "PreStimMicroNumSaccs", "PreStimNumSaccs", MODALITY]]
    seen_df.rename(columns={SUBJECT_NAME: "sub_code", "PreStimMicroNumSaccs": "vg_nummicrosaccs_seen",
                            "PreStimNumSaccs": "vg_numsaccs_seen"}, inplace=True)
    unseen_df = means_df[means_df[DataParser.VIS] == "FalseNegative"]
    unseen_df = unseen_df[[SUBJECT_NAME, "PreStimMicroNumSaccs", "PreStimNumSaccs", MODALITY]]
    unseen_df.rename(columns={SUBJECT_NAME: "sub_code", "PreStimMicroNumSaccs": "vg_nummicrosaccs_unseen",
                              "PreStimNumSaccs": "vg_numsaccs_unseen"}, inplace=True)
    merged_df = pd.merge(seen_df, unseen_df, how="inner", on=["sub_code", MODALITY])
    partial_df = pd.merge(seen_df, unseen_df, how="outer", on="sub_code")
    partial_set = set(partial_df["sub_code"])

    # format to R format
    # IMPORTANT - WE COLLAPSE MICROSACCADES AND SACCADES TOGETHER
    df_list = list()
    seen_df = merged_df[["sub_code", "vg_nummicrosaccs_seen", "vg_numsaccs_seen", MODALITY]]
    if w_micro:
        seen_df["vg_numsaccs"] = seen_df["vg_nummicrosaccs_seen"] + seen_df["vg_numsaccs_seen"]
    else:
        seen_df["vg_numsaccs"] = seen_df["vg_numsaccs_seen"]
    seen_df[VIS] = "seen"
    seen_df.drop(["vg_nummicrosaccs_seen", "vg_numsaccs_seen"], errors="ignore", axis=1, inplace=True)
    df_list.append(seen_df)

    unseen_df = merged_df[["sub_code", "vg_nummicrosaccs_unseen", "vg_numsaccs_unseen", MODALITY]]
    if w_micro:
        unseen_df["vg_numsaccs"] = unseen_df["vg_nummicrosaccs_unseen"] + unseen_df["vg_numsaccs_unseen"]
    else:
        unseen_df["vg_numsaccs"] = unseen_df["vg_numsaccs_unseen"]
    unseen_df[VIS] = "unseen"
    unseen_df.drop(["vg_nummicrosaccs_unseen", "vg_numsaccs_unseen"], errors="ignore", axis=1, inplace=True)
    df_list.append(unseen_df)
    merged_df = pd.concat(df_list)
    merged_df.to_csv(os.path.join(save_path, f"vg_prestim_sacc_per_vis_all_{w_micro_str}.csv"), index=False)
    all_missing_subs = list(set(all_subs_df[SUBJECT_NAME]) - set(merged_df["sub_code"]))
    missing_reason = ["Partial Data" if x in partial_set else "No Data" for x in all_missing_subs]
    missing_subs = pd.DataFrame({"subCode": all_missing_subs, "reason": missing_reason})
    missing_subs.to_csv(os.path.join(save_path, f"vg_prestim_sacc_per_vis_missing_subs_{w_micro_str}.csv"), index=False)

    # Without a variable
    for name in ["nochosen10", "chosen10"]:
        relevant_data = relevant_trials[(relevant_trials[DataParser.VIS] == "TruePositive") | (relevant_trials[DataParser.VIS] == "FalseNegative")]
        if name == "nochosen10":
            relevant_data = relevant_data[~relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = relevant_data[relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]

        if w_micro:
            relevant_data["vg_numsaccs"] = relevant_data["PreStimMicroNumSaccs"] + relevant_data["PreStimNumSaccs"]
        else:
            relevant_data["vg_numsaccs"] = relevant_data["PreStimNumSaccs"]
        relevant_data = relevant_data[[SUBJECT_NAME, MODALITY, DataParser.VIS, "vg_numsaccs"]]

        list_to_combo_from = [MODALITY, DataParser.VIS]
        combos = []
        for r in range(1, len(list_to_combo_from) + 1):
            combos.append(itertools.combinations(list_to_combo_from, r))

        stats_df_list = []
        for combo in combos:
            for x in combo:
                df_to_mean = relevant_data.groupby(list(x) + [SUBJECT_NAME]).mean().reset_index()
                mean_df = df_to_mean.groupby(list(x)).mean().reset_index()
                std_df = df_to_mean.groupby(list(x)).std().reset_index()
                mean_df.loc[:, "vg_numsaccs_std"] = std_df["vg_numsaccs"]
                stats_df_list.append(mean_df)

        merged_df = pd.concat(stats_df_list)
        merged_df.to_csv(os.path.join(save_path, f"vg_prestim_sacc_per_vis_{name}_{w_micro_str}_descriptives.csv"), index=False)
    return


def poststim_first_fix_analysis_left_right(all_subs_df, save_path, left_right=True):
    relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == False)]
    # We want only fixations in the timewindow [0, 500], since we have the time postonset,
    # take only those with this value smaller than 500
    relevant_trials = relevant_trials[(relevant_trials["FirstFixTimePostOnset"] <= 500)]

    if left_right:
        savename = "left_right"
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(RIGHT), "location"] = RIGHT
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(LEFT), "location"] = LEFT
        locations = [LEFT, RIGHT]
    else:
        savename = "top_bottom"
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(TOP), "location"] = TOP
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(BOTTOM), "location"] = BOTTOM
        locations = [TOP, BOTTOM]

    means_df = relevant_trials.groupby([SUBJECT_NAME, DataParser.VIS, "location", MODALITY]).mean().reset_index()
    means_df = means_df[[SUBJECT_NAME, DataParser.VIS, "FirstFixDur", "location", "FirstFixTimePostOnset", MODALITY]]
    df_list = []
    for vis in ["TruePositive", "FalseNegative"]:
        for location in locations:
            vis_type_df = means_df[(means_df[DataParser.VIS] == vis) & (means_df["location"] == location)]
            vis_str = "seen" if "True" in vis else "unseen"
            location_str = location.lower()
            vis_type_df = vis_type_df[[SUBJECT_NAME, "FirstFixDur", MODALITY, "FirstFixTimePostOnset"]]
            vis_type_df.rename(
                columns={SUBJECT_NAME: "sub_code", "FirstFixDur": f"vg_fixdur_{vis_str}_{location_str}",
                         "FirstFixTimePostOnset": f"vg_fixonset_{vis_str}_{location_str}"},
                inplace=True)
            df_list.append(vis_type_df)
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='inner'), df_list)
    partial_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='outer'), df_list)
    partial_set = set(partial_df["sub_code"])
    # format to R format
    df_list = list()

    for vis in ["TruePositive", "FalseNegative"]:
        for location in locations:
            location_str = location.lower()
            vis_str = "seen" if "True" in vis else "unseen"
            relevant_col1 = f"vg_fixdur_{vis_str}_{location_str}"
            relevant_col2 = f"vg_fixonset_{vis_str}_{location_str}"
            temp_df = merged_df[["sub_code", relevant_col1, relevant_col2, MODALITY]]
            temp_df[VIS] = vis_str
            temp_df["location"] = location_str
            temp_df.rename(columns={relevant_col1: "vg_fixdur", relevant_col2: "vg_fixonset"}, inplace=True)
            df_list.append(temp_df)

    merged_df = pd.concat(df_list)
    merged_df.to_csv(os.path.join(save_path, f"vg_poststim_fixdur_{savename}_per_vis.csv"), index=False)
    all_missing_subs = list(set(all_subs_df[SUBJECT_NAME]) - set(merged_df["sub_code"]))
    missing_reason = ["Partial Data" if x in partial_set else "No Data" for x in all_missing_subs]
    missing_subs = pd.DataFrame({"subCode": all_missing_subs, "reason": missing_reason})
    missing_subs.to_csv(os.path.join(save_path, f"vg_poststim_fixdur_{savename}_per_vis_missing_subs.csv"), index=False)

    # Without a variable
    for name in ["all", "chosen10"]:
        all_dfs = []
        relevant_data = relevant_trials[(relevant_trials[DataParser.VIS] == "TruePositive") | (relevant_trials[DataParser.VIS] == "FalseNegative")]
        if name == "all":
            relevant_data = relevant_data[~relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = relevant_data[relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]

        #relevant_data = relevant_data[((relevant_data["stimulusType"] == "Face") | (relevant_data["stimulusType"] == "Object"))]
        means_df = relevant_data.groupby([SUBJECT_NAME, MODALITY]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, MODALITY, "FirstFixDur", "FirstFixTimePostOnset"]]
        means_df2 = means_df.groupby([MODALITY]).mean().reset_index()
        means_df2.rename(columns={MODALITY: INDEPENDENT, "FirstFixDur": "vg_fixdur_mean", "FirstFixTimePostOnset": "vg_fixonset_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([MODALITY]).std().reset_index()
        std_df.rename(columns={MODALITY: INDEPENDENT, "FirstFixDur": "vg_fixdur_std", "FirstFixTimePostOnset": "vg_fixonset_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, DataParser.VIS]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, DataParser.VIS, "FirstFixDur", "FirstFixTimePostOnset"]]
        means_df2 = means_df.groupby([DataParser.VIS]).mean().reset_index()
        means_df2.rename(columns={DataParser.VIS: INDEPENDENT, "FirstFixDur": "vg_fixdur_mean", "FirstFixTimePostOnset": "vg_fixonset_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([DataParser.VIS]).std().reset_index()
        std_df.rename(columns={DataParser.VIS: INDEPENDENT, "FirstFixDur": "vg_fixdur_std", "FirstFixTimePostOnset": "vg_fixonset_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, "location"]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, "location", "FirstFixDur", "FirstFixTimePostOnset"]]
        means_df2 = means_df.groupby(["location"]).mean().reset_index()
        means_df2.rename(columns={"location": INDEPENDENT, "FirstFixDur": "vg_fixdur_mean", "FirstFixTimePostOnset": "vg_fixonset_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby(["location"]).std().reset_index()
        std_df.rename(columns={"location": INDEPENDENT, "FirstFixDur": "vg_fixdur_std", "FirstFixTimePostOnset": "vg_fixonset_std"}, inplace=True)
        all_dfs.append(std_df)

        merged_df = pd.concat(all_dfs)
        merged_df.to_csv(os.path.join(save_path, f"vg_poststim_fixdur_{savename}_per_vis_{name}_descriptives.csv"), index=False)
    return


def poststim_first_fix_analysis_left_right_replay(all_subs_df, save_path, left_right=True):
    relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == True)]

    relevant_trials.loc[:, TARGET] = relevant_trials[QualityChecker.STIM_TYPE] == relevant_trials["TargetType"]

    if left_right:
        savename = "left_right"
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(RIGHT), "location"] = RIGHT
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(LEFT), "location"] = LEFT
        locations = [LEFT, RIGHT]
    else:
        savename = "top_bottom"
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(TOP), "location"] = TOP
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(BOTTOM), "location"] = BOTTOM
        locations = [TOP, BOTTOM]

    # We want only fixations in the timewindow [0, 500], since we have the time postonset,
    # take only those with this value smaller than 500
    relevant_trials = relevant_trials[(relevant_trials["FirstFixTimePostOnset"] <= 500)]
    means_df = relevant_trials.groupby([SUBJECT_NAME, TARGET, "location", MODALITY]).mean().reset_index()
    means_df = means_df[[SUBJECT_NAME, TARGET, "FirstFixDur", "location", "FirstFixTimePostOnset", MODALITY]]
    df_list = []
    for vis in [True, False]:
        for stim_type in locations:
            vis_type_df = means_df[(means_df[TARGET] == vis) & (means_df["location"] == stim_type)]
            vis_str = "target" if vis==True else "non_target"
            stim_str = stim_type.lower()
            vis_type_df = vis_type_df[[SUBJECT_NAME, "FirstFixDur", MODALITY, "FirstFixTimePostOnset"]]
            vis_type_df.rename(
                columns={SUBJECT_NAME: "sub_code", "FirstFixDur": f"vg_fixdur_{vis_str}_{stim_str}",
                         "FirstFixTimePostOnset": f"vg_fixonset_{vis_str}_{stim_str}"},
                inplace=True)
            df_list.append(vis_type_df)
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='inner'), df_list)
    partial_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='outer'), df_list)
    partial_set = set(partial_df["sub_code"])
    # format to R format
    df_list = list()

    for vis in [True, False]:
        for stim_type in locations:
            stim_str = stim_type.lower()
            vis_str = "target" if vis==True else "non_target"
            relevant_col1 = f"vg_fixdur_{vis_str}_{stim_str}"
            relevant_col2 = f"vg_fixonset_{vis_str}_{stim_str}"
            temp_df = merged_df[["sub_code", relevant_col1, relevant_col2, MODALITY]]
            temp_df[TARGET] = vis_str
            temp_df["location"] = stim_str
            temp_df.rename(columns={relevant_col1: "vg_fixdur", relevant_col2: "vg_fixonset"}, inplace=True)
            df_list.append(temp_df)

    merged_df = pd.concat(df_list)
    merged_df.to_csv(os.path.join(save_path, f"vg_poststim_fixdur_{savename}_per_vis_replay.csv"), index=False)
    all_missing_subs = list(set(all_subs_df[SUBJECT_NAME]) - set(merged_df["sub_code"]))
    missing_reason = ["Partial Data" if x in partial_set else "No Data" for x in all_missing_subs]
    missing_subs = pd.DataFrame({"subCode": all_missing_subs, "reason": missing_reason})
    missing_subs.to_csv(os.path.join(save_path, f"vg_poststim_fixdur_{savename}_per_vis_missing_subs_replay.csv"), index=False)

    # Without a variable
    for name in ["all", "chosen10"]:
        all_dfs = []
        relevant_data = relevant_trials
        if name == "all":
            relevant_data = relevant_data[~relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = relevant_data[relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]

        relevant_data = relevant_data[((relevant_data["stimulusType"] == "Face") | (relevant_data["stimulusType"] == "Object"))]
        means_df = relevant_data.groupby([SUBJECT_NAME, MODALITY]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, MODALITY, "FirstFixDur", "FirstFixTimePostOnset"]]
        means_df2 = means_df.groupby([MODALITY]).mean().reset_index()
        means_df2.rename(columns={MODALITY: INDEPENDENT, "FirstFixDur": "vg_fixdur_mean", "FirstFixTimePostOnset": "vg_fixonset_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([MODALITY]).std().reset_index()
        std_df.rename(columns={MODALITY: INDEPENDENT, "FirstFixDur": "vg_fixdur_std", "FirstFixTimePostOnset": "vg_fixonset_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, TARGET]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, TARGET, "FirstFixDur", "FirstFixTimePostOnset"]]
        means_df2 = means_df.groupby([TARGET]).mean().reset_index()
        means_df2.rename(columns={TARGET: INDEPENDENT, "FirstFixDur": "vg_fixdur_mean", "FirstFixTimePostOnset": "vg_fixonset_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([TARGET]).std().reset_index()
        std_df.rename(columns={TARGET: INDEPENDENT, "FirstFixDur": "vg_fixdur_std", "FirstFixTimePostOnset": "vg_fixonset_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, "location"]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, "location", "FirstFixDur", "FirstFixTimePostOnset"]]
        means_df2 = means_df.groupby(["location"]).mean().reset_index()
        means_df2.rename(columns={"location": INDEPENDENT, "FirstFixDur": "vg_fixdur_mean", "FirstFixTimePostOnset": "vg_fixonset_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby(["location"]).std().reset_index()
        std_df.rename(columns={"location": INDEPENDENT, "FirstFixDur": "vg_fixdur_std", "FirstFixTimePostOnset": "vg_fixonset_std"}, inplace=True)
        all_dfs.append(std_df)

        merged_df = pd.concat(all_dfs)
        merged_df.to_csv(os.path.join(save_path, f"vg_poststim_fixdur_{savename}_per_vis_{name}_descriptives_replay.csv"), index=False)
    return


def poststim_first_fix_analysis_new(all_subs_df, save_path, is_250_500=False):
    if is_250_500:
        first_fix_col = "FirstFixTimePost250"
        first_fix_dur_col = "FirstFixDur250"
    else:
        first_fix_col = "FirstFixTimePostOnset"
        first_fix_dur_col = "FirstFixDur"

    relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == False)]
    # We want only fixations in the timewindow [0, 500], since we have the time postonset,
    # take only those with this value smaller than 500
    relevant_trials = relevant_trials[(relevant_trials[first_fix_col] <= 500)]
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(LEFT), "horizontal"] = LEFT
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(RIGHT), "horizontal"] = RIGHT
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(TOP), "vertical"] = TOP
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(BOTTOM), "vertical"] = BOTTOM

    means_df = relevant_trials.groupby([SUBJECT_NAME, DataParser.VIS, "stimulusType", MODALITY, "horizontal", "vertical"]).mean().reset_index()
    means_df = means_df[[SUBJECT_NAME, DataParser.VIS, first_fix_dur_col, "stimulusType", first_fix_col, MODALITY, "horizontal", "vertical"]]
    means_df = means_df[(means_df[DataParser.VIS] == "TruePositive") | (means_df[DataParser.VIS] == "FalseNegative")]
    means_df.rename(
        columns={SUBJECT_NAME: "sub_code", first_fix_dur_col: f"vg_fixdur", DataParser.VIS: VIS,
                 first_fix_col: f"vg_fixonset", "stimulusType": "category"},
        inplace=True)
    means_df[VIS].replace({"TruePositive": "seen", "FalseNegative": "unseen"}, inplace=True)
    means_df.to_csv(os.path.join(save_path, f"vg_poststim_fixdur_{is_250_500}_per_vis_all.csv"), index=False)

    relevant_trials = relevant_trials[[SUBJECT_NAME, DataParser.VIS, first_fix_dur_col, "stimulusType", first_fix_col, MODALITY, "horizontal", "vertical"]]

    # Without a variable
    for name in ["nochosen10", "chosen10"]:
        relevant_data = relevant_trials[(relevant_trials[DataParser.VIS] == "TruePositive") | (relevant_trials[DataParser.VIS] == "FalseNegative")]
        if name == "nochosen10":
            relevant_data = relevant_data[~relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = relevant_data[relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]

        list_to_combo_from = [MODALITY, "stimulusType", DataParser.VIS, "horizontal", "vertical"]
        combos = []
        for r in range(1, len(list_to_combo_from) + 1):
            combos.append(itertools.combinations(list_to_combo_from, r))

        stats_df_list = []
        df_to_mean_list = []
        for combo in combos:
            for x in combo:
                if "mod" not in list(x):
                    df_to_mean = relevant_data.groupby(list(x)+[SUBJECT_NAME, MODALITY]).mean().reset_index()
                else:
                    df_to_mean = relevant_data.groupby(list(x) + [SUBJECT_NAME]).mean().reset_index()
                mean_df = df_to_mean.groupby(list(x)).mean().reset_index()
                std_df = df_to_mean.groupby(list(x)).std().reset_index()
                for col in [first_fix_dur_col, first_fix_col]:
                    mean_df.loc[:, f"{col}_std"] = std_df[col]
                    mean_df.rename(columns={col: f"{col}_mean"}, inplace=True)
                stats_df_list.append(mean_df)
                df_to_mean_list.append(df_to_mean)

        df_to_mean = pd.concat(df_to_mean_list)
        df_to_mean.to_csv(os.path.join(save_path, f"vg_poststim_fixdur_per_vis_{name}_{is_250_500}_sub_stats.csv"), index=False)
        stat_df = pd.concat(stats_df_list)
        stat_df.to_csv(os.path.join(save_path, f"vg_poststim_fixdur_per_vis_{name}_{is_250_500}_descriptives.csv"), index=False)

    # NEW PLOT
    relevant_data = relevant_trials[(relevant_trials[DataParser.VIS] == "TruePositive") | (relevant_trials[DataParser.VIS] == "FalseNegative")]
    relevant_data = relevant_data[~relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]

    df_to_mean = relevant_data.groupby([SUBJECT_NAME, DataParser.VIS]).mean().reset_index()
    seen = df_to_mean[df_to_mean[DataParser.VIS]=="TruePositive"]
    unseen = df_to_mean[df_to_mean[DataParser.VIS]=="FalseNegative"]
    merged_df = pd.merge(seen, unseen, on=SUBJECT_NAME, suffixes=("_seen", "_unseen"))
    merged_df = merged_df.loc[:, [SUBJECT_NAME, f"{first_fix_dur_col}_seen", f"{first_fix_dur_col}_unseen"]]
    colors = ["#1F7A8C", "#90141E"]
    plotter.make_it_rain(data=merged_df, cols=[f"{first_fix_dur_col}_seen", f"{first_fix_dur_col}_unseen"], col_names=["Seen", "Unseen"],
                         color_list=colors, plot_title="Fixation Duration Per Visibility In Game", plot_x_label="Visibility", plot_y_label="Fixation Duration (ms)",
                         sub_line=[list(range(len(colors)))], save_plot=True,
                         save_path=save_path, save_folder="./", horizontal_lines=None,
                         save_name=f"vg_poststim_fixdur_{is_250_500}_per_vis", custom_ylim=1500, skip=100)
    """
        colors = ["#1F7A8C", "#90141E"]  # seen, unseen
    boxplotter.plot(data=summary_table, data_col_order=[prefix + 'avg_tp', prefix + 'avg_fn'],
                    data_name_order={prefix + 'avg_tp': SEEN, prefix + 'avg_fn': UNSEEN}, raincloud=True,
                    sub_line=[list(range(len(colors)))], plot_title=f"Average {title} in Seen vs Unseen Stimuli",
                    plot_x_label="Response Type", plot_y_label=f"Average {title}", colors_per_lab=0, color_list=colors,
                    save_plot=True, save_path=save_path, save_name=avg_save_name,
                    sub_folder=os.path.join(title, PER_RESP))

    """
    return


def poststim_first_fix_analysis_game_vs_replay(all_subs_df, save_path, is_250_500=False):
    if is_250_500:
        first_fix_col = "FirstFixTimePost250"
    else:
        first_fix_col = "FirstFixTimePostOnset"
    game_relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == False)]
    game_relevant_trials = game_relevant_trials[(game_relevant_trials[DataParser.VIS] == "TruePositive")]

    # We want only fixations in the timewindow [0, 500], since we have the time postonset,
    # take only those with this value smaller than 500
    game_relevant_trials = game_relevant_trials[(game_relevant_trials[first_fix_col] <= 500)]
    game_relevant_trials.loc[game_relevant_trials[DataParser.STIM_LOC].str.contains(LEFT), "horizontal"] = LEFT
    game_relevant_trials.loc[game_relevant_trials[DataParser.STIM_LOC].str.contains(RIGHT), "horizontal"] = RIGHT
    game_relevant_trials.loc[game_relevant_trials[DataParser.STIM_LOC].str.contains(TOP), "vertical"] = TOP
    game_relevant_trials.loc[game_relevant_trials[DataParser.STIM_LOC].str.contains(BOTTOM), "vertical"] = BOTTOM

    game_relevant_trials["condition"] = "Game Seen"

    game_means_df = game_relevant_trials.groupby([SUBJECT_NAME, MODALITY, "horizontal", "vertical"]).mean().reset_index()
    game_means_df = game_means_df[[SUBJECT_NAME, "FirstFixDur", first_fix_col, MODALITY, "horizontal", "vertical"]]
    game_means_df["condition"] = "Game Seen"

    replay_relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == True)]

    replay_relevant_trials.loc[:, TARGET] = replay_relevant_trials[QualityChecker.STIM_TYPE] == replay_relevant_trials["TargetType"]
    # We want only fixations in the timewindow [0, 500], since we have the time postonset,
    # take only those with this value smaller than 500
    replay_relevant_trials = replay_relevant_trials[replay_relevant_trials[first_fix_col] <= 500]
    replay_relevant_trials.loc[replay_relevant_trials[DataParser.STIM_LOC].str.contains(LEFT), "horizontal"] = LEFT
    replay_relevant_trials.loc[replay_relevant_trials[DataParser.STIM_LOC].str.contains(RIGHT), "horizontal"] = RIGHT
    replay_relevant_trials.loc[replay_relevant_trials[DataParser.STIM_LOC].str.contains(TOP), "vertical"] = TOP
    replay_relevant_trials.loc[replay_relevant_trials[DataParser.STIM_LOC].str.contains(BOTTOM), "vertical"] = BOTTOM

    replay_relevant_trials = replay_relevant_trials[((replay_relevant_trials["stimulusType"] == "Face") | (replay_relevant_trials["stimulusType"] == "Object"))]
    replay_relevant_target_trials = replay_relevant_trials[replay_relevant_trials[TARGET] == True]
    replay_relevant_target_trials["condition"] = "Replay Target"
    replay_relevant_nontarget_trials = replay_relevant_trials[replay_relevant_trials[TARGET] == False]
    replay_relevant_nontarget_trials["condition"] = "Replay Non Target"
    all_relevant_trials = pd.concat([game_relevant_trials, replay_relevant_target_trials, replay_relevant_nontarget_trials])
    all_relevant_trials = all_relevant_trials[[SUBJECT_NAME, "condition", "FirstFixDur", MODALITY, "horizontal", "vertical"]]

    replay_means_df = replay_relevant_trials.groupby([SUBJECT_NAME, TARGET, MODALITY, "horizontal", "vertical"]).mean().reset_index()
    replay_means_df = replay_means_df[[SUBJECT_NAME, TARGET, "FirstFixDur", first_fix_col, MODALITY, "horizontal", "vertical"]]

    replay_means_df_target = replay_means_df[replay_means_df[TARGET] == True]
    replay_means_df_target["condition"] = "Replay Target"
    replay_means_df_target = replay_means_df_target[[SUBJECT_NAME, "condition", "FirstFixDur", first_fix_col, MODALITY, "horizontal", "vertical"]]

    replay_means_df_nontarget = replay_means_df[replay_means_df[TARGET] == False]
    replay_means_df_nontarget["condition"] = "Replay Non Target"
    replay_means_df_nontarget = replay_means_df_nontarget[[SUBJECT_NAME, "condition", "FirstFixDur", first_fix_col, MODALITY, "horizontal", "vertical"]]

    data_for_model = pd.concat([game_means_df, replay_means_df_target, replay_means_df_nontarget])
    data_for_model.to_csv(os.path.join(save_path, f"vg_v_replay_poststim_fixdur_per_loc_all.csv"), index=False)

    # Without a variable
    for name in ["nochosen10", "chosen10"]:
        if name == "nochosen10":
            relevant_data = all_relevant_trials[~all_relevant_trials[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = all_relevant_trials[all_relevant_trials[SUBJECT_NAME].isin(CHOSEN_10)]

        list_to_combo_from = ["condition", MODALITY, "horizontal", "vertical"]
        combos = []
        for r in range(1, len(list_to_combo_from) + 1):
            combos.append(itertools.combinations(list_to_combo_from, r))

        stats_df_list = []
        df_to_mean_list = []
        for combo in combos:
            for x in combo:
                if MODALITY not in list(x):
                    df_to_mean = relevant_data.groupby(list(x)+[SUBJECT_NAME, MODALITY]).mean().reset_index()
                else:
                    df_to_mean = relevant_data.groupby(list(x) + [SUBJECT_NAME]).mean().reset_index()
                mean_df = df_to_mean.groupby(list(x)).mean().reset_index()
                std_df = df_to_mean.groupby(list(x)).std().reset_index()
                for col in ["FirstFixDur"]:
                    mean_df.loc[:, f"{col}_std"] = std_df[col]
                    mean_df.rename(columns={col: f"{col}_mean"}, inplace=True)
                stats_df_list.append(mean_df)
                df_to_mean_list.append(df_to_mean)

        stat_df = pd.concat(stats_df_list)
        stat_df.to_csv(os.path.join(save_path, f"vg_v_replay_poststim_fixdur_per_loc_{name}_descriptives.csv"), index=False)
    return


def poststim_saccade_num_analysis_game_vs_replay(all_subs_df, save_path):
    game_relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == False)]
    game_relevant_trials = game_relevant_trials.fillna(0)
    game_relevant_trials["StimDurationAllNumSaccs"] = game_relevant_trials["StimDurationMicroNumSaccs"] + game_relevant_trials["StimDurationNumSaccs"]

    game_relevant_trials = game_relevant_trials[(game_relevant_trials[DataParser.VIS] == "TruePositive")]

    game_relevant_trials.loc[game_relevant_trials[DataParser.STIM_LOC].str.contains(LEFT), "horizontal"] = LEFT
    game_relevant_trials.loc[game_relevant_trials[DataParser.STIM_LOC].str.contains(RIGHT), "horizontal"] = RIGHT
    game_relevant_trials.loc[game_relevant_trials[DataParser.STIM_LOC].str.contains(TOP), "vertical"] = TOP
    game_relevant_trials.loc[game_relevant_trials[DataParser.STIM_LOC].str.contains(BOTTOM), "vertical"] = BOTTOM

    game_relevant_trials["condition"] = "Game Seen"

    game_means_df = game_relevant_trials.groupby([SUBJECT_NAME, MODALITY, "horizontal", "vertical"]).mean().reset_index()
    game_means_df = game_means_df[[SUBJECT_NAME, "StimDurationAllNumSaccs", MODALITY, "horizontal", "vertical"]]
    game_means_df["condition"] = "Game Seen"

    replay_relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == True)]
    replay_relevant_trials = replay_relevant_trials.fillna(0)
    replay_relevant_trials["StimDurationAllNumSaccs"] = replay_relevant_trials["StimDurationMicroNumSaccs"] + replay_relevant_trials["StimDurationNumSaccs"]

    replay_relevant_trials.loc[:, TARGET] = replay_relevant_trials[QualityChecker.STIM_TYPE] == replay_relevant_trials["TargetType"]
    replay_relevant_trials.loc[replay_relevant_trials[DataParser.STIM_LOC].str.contains(LEFT), "horizontal"] = LEFT
    replay_relevant_trials.loc[replay_relevant_trials[DataParser.STIM_LOC].str.contains(RIGHT), "horizontal"] = RIGHT
    replay_relevant_trials.loc[replay_relevant_trials[DataParser.STIM_LOC].str.contains(TOP), "vertical"] = TOP
    replay_relevant_trials.loc[replay_relevant_trials[DataParser.STIM_LOC].str.contains(BOTTOM), "vertical"] = BOTTOM

    replay_relevant_trials = replay_relevant_trials[((replay_relevant_trials["stimulusType"] == "Face") | (replay_relevant_trials["stimulusType"] == "Object"))]
    replay_relevant_target_trials = replay_relevant_trials[replay_relevant_trials[TARGET] == True]
    replay_relevant_target_trials["condition"] = "Replay Target"
    replay_relevant_nontarget_trials = replay_relevant_trials[replay_relevant_trials[TARGET] == False]
    replay_relevant_nontarget_trials["condition"] = "Replay Non Target"
    all_relevant_trials = pd.concat([game_relevant_trials, replay_relevant_target_trials, replay_relevant_nontarget_trials])
    all_relevant_trials = all_relevant_trials[[SUBJECT_NAME, "condition", "StimDurationAllNumSaccs", MODALITY, "horizontal", "vertical"]]

    replay_means_df = replay_relevant_trials.groupby([SUBJECT_NAME, TARGET, MODALITY, "horizontal", "vertical"]).mean().reset_index()
    replay_means_df = replay_means_df[[SUBJECT_NAME, TARGET, "StimDurationAllNumSaccs", MODALITY, "horizontal", "vertical"]]

    replay_means_df_target = replay_means_df[replay_means_df[TARGET] == True]
    replay_means_df_target["condition"] = "Replay Target"
    replay_means_df_target = replay_means_df_target[[SUBJECT_NAME, "condition", "StimDurationAllNumSaccs", MODALITY, "horizontal", "vertical"]]

    replay_means_df_nontarget = replay_means_df[replay_means_df[TARGET] == False]
    replay_means_df_nontarget["condition"] = "Replay Non Target"
    replay_means_df_nontarget = replay_means_df_nontarget[[SUBJECT_NAME, "condition", "StimDurationAllNumSaccs", MODALITY, "horizontal", "vertical"]]

    data_for_model = pd.concat([game_means_df, replay_means_df_target, replay_means_df_nontarget])
    data_for_model.to_csv(os.path.join(save_path, f"vg_v_replay_poststim_numsaccs_per_loc_withmicro_all.csv"), index=False)

    # Without a variable
    for name in ["nochosen10", "chosen10"]:
        if name == "nochosen10":
            relevant_data = all_relevant_trials[~all_relevant_trials[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = all_relevant_trials[all_relevant_trials[SUBJECT_NAME].isin(CHOSEN_10)]

        list_to_combo_from = ["condition", MODALITY, "horizontal", "vertical"]
        combos = []
        for r in range(1, len(list_to_combo_from) + 1):
            combos.append(itertools.combinations(list_to_combo_from, r))

        stats_df_list = []
        df_to_mean_list = []
        for combo in combos:
            for x in combo:
                if MODALITY not in list(x):
                    df_to_mean = relevant_data.groupby(list(x)+[SUBJECT_NAME, MODALITY]).mean().reset_index()
                else:
                    df_to_mean = relevant_data.groupby(list(x) + [SUBJECT_NAME]).mean().reset_index()
                mean_df = df_to_mean.groupby(list(x)).mean().reset_index()
                std_df = df_to_mean.groupby(list(x)).std().reset_index()
                for col in ["StimDurationAllNumSaccs"]:
                    mean_df.loc[:, f"{col}_std"] = std_df[col]
                    mean_df.rename(columns={col: f"{col}_mean"}, inplace=True)
                stats_df_list.append(mean_df)
                df_to_mean_list.append(df_to_mean)

        stat_df = pd.concat(stats_df_list)
        stat_df.to_csv(os.path.join(save_path, f"vg_v_replay_poststim_numsaccs_per_loc_{name}_withmicro_descriptives.csv"), index=False)
    return


def poststim_blink_num_analysis_game_vs_replay(all_subs_df, save_path):
    game_relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == False)]
    game_relevant_trials = game_relevant_trials.fillna(0)

    game_relevant_trials = game_relevant_trials[(game_relevant_trials[DataParser.VIS] == "TruePositive")]

    game_relevant_trials.loc[game_relevant_trials[DataParser.STIM_LOC].str.contains(LEFT), "horizontal"] = LEFT
    game_relevant_trials.loc[game_relevant_trials[DataParser.STIM_LOC].str.contains(RIGHT), "horizontal"] = RIGHT
    game_relevant_trials.loc[game_relevant_trials[DataParser.STIM_LOC].str.contains(TOP), "vertical"] = TOP
    game_relevant_trials.loc[game_relevant_trials[DataParser.STIM_LOC].str.contains(BOTTOM), "vertical"] = BOTTOM

    game_relevant_trials["condition"] = "Game Seen"

    game_means_df = game_relevant_trials.groupby([SUBJECT_NAME, MODALITY, "horizontal", "vertical"]).mean().reset_index()
    game_means_df = game_means_df[[SUBJECT_NAME, "StimDurationNumBlinks", MODALITY, "horizontal", "vertical"]]
    game_means_df["condition"] = "Game Seen"

    replay_relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == True)]
    replay_relevant_trials = replay_relevant_trials.fillna(0)

    replay_relevant_trials.loc[:, TARGET] = replay_relevant_trials[QualityChecker.STIM_TYPE] == replay_relevant_trials["TargetType"]
    replay_relevant_trials.loc[replay_relevant_trials[DataParser.STIM_LOC].str.contains(LEFT), "horizontal"] = LEFT
    replay_relevant_trials.loc[replay_relevant_trials[DataParser.STIM_LOC].str.contains(RIGHT), "horizontal"] = RIGHT
    replay_relevant_trials.loc[replay_relevant_trials[DataParser.STIM_LOC].str.contains(TOP), "vertical"] = TOP
    replay_relevant_trials.loc[replay_relevant_trials[DataParser.STIM_LOC].str.contains(BOTTOM), "vertical"] = BOTTOM

    replay_relevant_trials = replay_relevant_trials[((replay_relevant_trials["stimulusType"] == "Face") | (replay_relevant_trials["stimulusType"] == "Object"))]
    replay_relevant_target_trials = replay_relevant_trials[replay_relevant_trials[TARGET] == True]
    replay_relevant_target_trials["condition"] = "Replay Target"
    replay_relevant_nontarget_trials = replay_relevant_trials[replay_relevant_trials[TARGET] == False]
    replay_relevant_nontarget_trials["condition"] = "Replay Non Target"
    all_relevant_trials = pd.concat([game_relevant_trials, replay_relevant_target_trials, replay_relevant_nontarget_trials])
    all_relevant_trials = all_relevant_trials[[SUBJECT_NAME, "condition", "StimDurationNumBlinks", MODALITY, "horizontal", "vertical"]]

    replay_means_df = replay_relevant_trials.groupby([SUBJECT_NAME, TARGET, MODALITY, "horizontal", "vertical"]).mean().reset_index()
    replay_means_df = replay_means_df[[SUBJECT_NAME, TARGET, "StimDurationNumBlinks", MODALITY, "horizontal", "vertical"]]

    replay_means_df_target = replay_means_df[replay_means_df[TARGET] == True]
    replay_means_df_target["condition"] = "Replay Target"
    replay_means_df_target = replay_means_df_target[[SUBJECT_NAME, "condition", "StimDurationNumBlinks", MODALITY, "horizontal", "vertical"]]

    replay_means_df_nontarget = replay_means_df[replay_means_df[TARGET] == False]
    replay_means_df_nontarget["condition"] = "Replay Non Target"
    replay_means_df_nontarget = replay_means_df_nontarget[[SUBJECT_NAME, "condition", "StimDurationNumBlinks", MODALITY, "horizontal", "vertical"]]

    data_for_model = pd.concat([game_means_df, replay_means_df_target, replay_means_df_nontarget])
    data_for_model.to_csv(os.path.join(save_path, f"vg_v_replay_poststim_numblinks_per_loc_all.csv"), index=False)

    # Without a variable
    for name in ["nochosen10", "chosen10"]:
        if name == "nochosen10":
            relevant_data = all_relevant_trials[~all_relevant_trials[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = all_relevant_trials[all_relevant_trials[SUBJECT_NAME].isin(CHOSEN_10)]

        list_to_combo_from = ["condition", MODALITY, "horizontal", "vertical"]
        combos = []
        for r in range(1, len(list_to_combo_from) + 1):
            combos.append(itertools.combinations(list_to_combo_from, r))

        stats_df_list = []
        df_to_mean_list = []
        for combo in combos:
            for x in combo:
                if MODALITY not in list(x):
                    df_to_mean = relevant_data.groupby(list(x)+[SUBJECT_NAME, MODALITY]).mean().reset_index()
                else:
                    df_to_mean = relevant_data.groupby(list(x) + [SUBJECT_NAME]).mean().reset_index()
                mean_df = df_to_mean.groupby(list(x)).mean().reset_index()
                std_df = df_to_mean.groupby(list(x)).std().reset_index()
                for col in ["StimDurationNumBlinks"]:
                    mean_df.loc[:, f"{col}_std"] = std_df[col]
                    mean_df.rename(columns={col: f"{col}_mean"}, inplace=True)
                stats_df_list.append(mean_df)
                df_to_mean_list.append(df_to_mean)

        stat_df = pd.concat(stats_df_list)
        stat_df.to_csv(os.path.join(save_path, f"vg_v_replay_poststim_numblinks_per_loc_{name}_descriptives.csv"), index=False)
    return


def poststim_first_fix_analysis(all_subs_df, save_path):
    relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == False)]
    # We want only fixations in the timewindow [0, 500], since we have the time postonset,
    # take only those with this value smaller than 500
    relevant_trials = relevant_trials[(relevant_trials["FirstFixTimePostOnset"] <= 500)]
    means_df = relevant_trials.groupby([SUBJECT_NAME, DataParser.VIS, "stimulusType", MODALITY]).mean().reset_index()
    means_df = means_df[[SUBJECT_NAME, DataParser.VIS, "FirstFixDur", "stimulusType", "FirstFixTimePostOnset", MODALITY]]
    df_list = []
    for vis in ["TruePositive", "FalseNegative"]:
        for stim_type in ["Face", "Object"]:
            vis_type_df = means_df[(means_df[DataParser.VIS] == vis) & (means_df["stimulusType"] == stim_type)]
            vis_str = "seen" if "True" in vis else "unseen"
            stim_str = stim_type.lower()
            vis_type_df = vis_type_df[[SUBJECT_NAME, "FirstFixDur", MODALITY, "FirstFixTimePostOnset"]]
            vis_type_df.rename(
                columns={SUBJECT_NAME: "sub_code", "FirstFixDur": f"vg_fixdur_{vis_str}_{stim_str}",
                         "FirstFixTimePostOnset": f"vg_fixonset_{vis_str}_{stim_str}"},
                inplace=True)
            df_list.append(vis_type_df)
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='inner'), df_list)
    partial_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='outer'), df_list)
    partial_set = set(partial_df["sub_code"])
    # format to R format
    df_list = list()

    for vis in ["TruePositive", "FalseNegative"]:
        for stim_type in ["Face", "Object"]:
            stim_str = stim_type.lower()
            vis_str = "seen" if "True" in vis else "unseen"
            relevant_col1 = f"vg_fixdur_{vis_str}_{stim_str}"
            relevant_col2 = f"vg_fixonset_{vis_str}_{stim_str}"
            temp_df = merged_df[["sub_code", relevant_col1, relevant_col2, MODALITY]]
            temp_df[VIS] = vis_str
            temp_df["category"] = stim_str
            temp_df.rename(columns={relevant_col1: "vg_fixdur", relevant_col2: "vg_fixonset"}, inplace=True)
            df_list.append(temp_df)

    merged_df = pd.concat(df_list)
    merged_df.to_csv(os.path.join(save_path, f"vg_poststim_fixdur_per_vis.csv"), index=False)
    all_missing_subs = list(set(all_subs_df[SUBJECT_NAME]) - set(merged_df["sub_code"]))
    missing_reason = ["Partial Data" if x in partial_set else "No Data" for x in all_missing_subs]
    missing_subs = pd.DataFrame({"subCode": all_missing_subs, "reason": missing_reason})
    missing_subs.to_csv(os.path.join(save_path, f"vg_poststim_fixdur_per_vis_missing_subs.csv"), index=False)

    # Without a variable
    for name in ["all", "chosen10"]:
        all_dfs = []
        relevant_data = relevant_trials[(relevant_trials[DataParser.VIS] == "TruePositive") | (relevant_trials[DataParser.VIS] == "FalseNegative")]
        if name == "all":
            relevant_data = relevant_data[~relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = relevant_data[relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]

        relevant_data = relevant_data[((relevant_data["stimulusType"] == "Face") | (relevant_data["stimulusType"] == "Object"))]
        means_df = relevant_data.groupby([SUBJECT_NAME, MODALITY]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, MODALITY, "FirstFixDur", "FirstFixTimePostOnset"]]
        means_df2 = means_df.groupby([MODALITY]).mean().reset_index()
        means_df2.rename(columns={MODALITY: INDEPENDENT, "FirstFixDur": "vg_fixdur_mean", "FirstFixTimePostOnset": "vg_fixonset_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([MODALITY]).std().reset_index()
        std_df.rename(columns={MODALITY: INDEPENDENT, "FirstFixDur": "vg_fixdur_std", "FirstFixTimePostOnset": "vg_fixonset_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, DataParser.VIS]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, DataParser.VIS, "FirstFixDur", "FirstFixTimePostOnset"]]
        means_df2 = means_df.groupby([DataParser.VIS]).mean().reset_index()
        means_df2.rename(columns={DataParser.VIS: INDEPENDENT, "FirstFixDur": "vg_fixdur_mean", "FirstFixTimePostOnset": "vg_fixonset_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([DataParser.VIS]).std().reset_index()
        std_df.rename(columns={DataParser.VIS: INDEPENDENT, "FirstFixDur": "vg_fixdur_std", "FirstFixTimePostOnset": "vg_fixonset_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, "stimulusType"]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, "stimulusType", "FirstFixDur", "FirstFixTimePostOnset"]]
        means_df2 = means_df.groupby(["stimulusType"]).mean().reset_index()
        means_df2.rename(columns={"stimulusType": INDEPENDENT, "FirstFixDur": "vg_fixdur_mean", "FirstFixTimePostOnset": "vg_fixonset_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby(["stimulusType"]).std().reset_index()
        std_df.rename(columns={"stimulusType": INDEPENDENT, "FirstFixDur": "vg_fixdur_std", "FirstFixTimePostOnset": "vg_fixonset_std"}, inplace=True)
        all_dfs.append(std_df)

        merged_df = pd.concat(all_dfs)
        merged_df.to_csv(os.path.join(save_path, f"vg_poststim_fixdur_per_vis_{name}_descriptives.csv"), index=False)
    return


def poststim_first_fix_analysis_replay_new(all_subs_df, save_path):
    relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == True)]

    relevant_trials.loc[:, TARGET] = relevant_trials[QualityChecker.STIM_TYPE] == relevant_trials["TargetType"]
    # We want only fixations in the timewindow [0, 500], since we have the time postonset,
    # take only those with this value smaller than 500
    relevant_trials = relevant_trials[(relevant_trials["FirstFixTimePostOnset"] <= 500)]
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(LEFT), "horizontal"] = LEFT
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(RIGHT), "horizontal"] = RIGHT
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(TOP), "vertical"] = TOP
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(BOTTOM), "vertical"] = BOTTOM

    relevant_trials = relevant_trials[((relevant_trials["stimulusType"] == "Face") | (relevant_trials["stimulusType"] == "Object"))]

    means_df = relevant_trials.groupby([SUBJECT_NAME, TARGET, "stimulusType", MODALITY, "horizontal", "vertical"]).mean().reset_index()
    means_df = means_df[[SUBJECT_NAME, TARGET, "FirstFixDur", "stimulusType", "FirstFixTimePostOnset", MODALITY, "horizontal", "vertical"]]

    means_df.rename(
        columns={SUBJECT_NAME: "sub_code", "FirstFixDur": f"vg_fixdur",
                 "FirstFixTimePostOnset": f"vg_fixonset", "stimulusType": "category"},
        inplace=True)
    means_df[TARGET].replace({True: "target", False: "non_target"}, inplace=True)
    means_df.to_csv(os.path.join(save_path, f"vg_poststim_fixdur_per_vis_replay.csv"), index=False)

    relevant_trials = relevant_trials[[SUBJECT_NAME, TARGET, "FirstFixDur", "stimulusType", "FirstFixTimePostOnset", MODALITY, "horizontal", "vertical"]]
    relevant_trials[TARGET].replace({True: "target", False: "non_target"}, inplace=True)
    # Without a variable
    for name in ["all", "chosen10"]:
        if name == "all":
            relevant_data = relevant_trials[~relevant_trials[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = relevant_trials[relevant_trials[SUBJECT_NAME].isin(CHOSEN_10)]

        list_to_combo_from = [MODALITY, "stimulusType", TARGET, "horizontal", "vertical"]
        combos = []
        for r in range(1, len(list_to_combo_from) + 1):
            combos.append(itertools.combinations(list_to_combo_from, r))

        stats_df_list = []
        df_to_mean_list = []
        for combo in combos:
            for x in combo:
                if "mod" not in list(x):
                    df_to_mean = relevant_data.groupby(list(x)+[SUBJECT_NAME, MODALITY]).mean().reset_index()
                else:
                    df_to_mean = relevant_data.groupby(list(x) + [SUBJECT_NAME]).mean().reset_index()
                mean_df = df_to_mean.groupby(list(x)).mean().reset_index()
                std_df = df_to_mean.groupby(list(x)).std().reset_index()
                for col in ["FirstFixDur", "FirstFixTimePostOnset"]:
                    mean_df.loc[:, f"{col}_std"] = std_df[col]
                    mean_df.rename(columns={col: f"{col}_mean"}, inplace=True)
                stats_df_list.append(mean_df)
                df_to_mean_list.append(df_to_mean)

        df_to_mean = pd.concat(df_to_mean_list)
        df_to_mean.to_csv(os.path.join(save_path, f"vg_poststim_fixdur_per_vis_{name}_sub_stats_replay.csv"), index=False)
        stat_df = pd.concat(stats_df_list)
        stat_df.to_csv(os.path.join(save_path, f"vg_poststim_fixdur_per_vis_{name}_descriptives_replay.csv"), index=False)
    return


def poststim_first_fix_analysis_replay(all_subs_df, save_path):
    relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == True)]

    relevant_trials.loc[:, TARGET] = relevant_trials[QualityChecker.STIM_TYPE] == relevant_trials["TargetType"]
    # We want only fixations in the timewindow [0, 500], since we have the time postonset,
    # take only those with this value smaller than 500
    relevant_trials = relevant_trials[(relevant_trials["FirstFixTimePostOnset"] <= 500)]
    means_df = relevant_trials.groupby([SUBJECT_NAME, TARGET, "stimulusType", MODALITY]).mean().reset_index()
    means_df = means_df[[SUBJECT_NAME, TARGET, "FirstFixDur", "stimulusType", "FirstFixTimePostOnset", MODALITY]]
    df_list = []
    for vis in [True, False]:
        for stim_type in ["Face", "Object"]:
            vis_type_df = means_df[(means_df[TARGET] == vis) & (means_df["stimulusType"] == stim_type)]
            vis_str = "target" if vis==True else "non_target"
            stim_str = stim_type.lower()
            vis_type_df = vis_type_df[[SUBJECT_NAME, "FirstFixDur", MODALITY, "FirstFixTimePostOnset"]]
            vis_type_df.rename(
                columns={SUBJECT_NAME: "sub_code", "FirstFixDur": f"vg_fixdur_{vis_str}_{stim_str}",
                         "FirstFixTimePostOnset": f"vg_fixonset_{vis_str}_{stim_str}"},
                inplace=True)
            df_list.append(vis_type_df)
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='inner'), df_list)
    partial_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='outer'), df_list)
    partial_set = set(partial_df["sub_code"])
    # format to R format
    df_list = list()

    for vis in [True, False]:
        for stim_type in ["Face", "Object"]:
            stim_str = stim_type.lower()
            vis_str = "target" if vis==True else "non_target"
            relevant_col1 = f"vg_fixdur_{vis_str}_{stim_str}"
            relevant_col2 = f"vg_fixonset_{vis_str}_{stim_str}"
            temp_df = merged_df[["sub_code", relevant_col1, relevant_col2, MODALITY]]
            temp_df[TARGET] = vis_str
            temp_df["category"] = stim_str
            temp_df.rename(columns={relevant_col1: "vg_fixdur", relevant_col2: "vg_fixonset"}, inplace=True)
            df_list.append(temp_df)

    merged_df = pd.concat(df_list)
    merged_df.to_csv(os.path.join(save_path, f"vg_poststim_fixdur_per_vis_replay.csv"), index=False)
    all_missing_subs = list(set(all_subs_df[SUBJECT_NAME]) - set(merged_df["sub_code"]))
    missing_reason = ["Partial Data" if x in partial_set else "No Data" for x in all_missing_subs]
    missing_subs = pd.DataFrame({"subCode": all_missing_subs, "reason": missing_reason})
    missing_subs.to_csv(os.path.join(save_path, f"vg_poststim_fixdur_per_vis_missing_subs_replay.csv"), index=False)

    # Without a variable
    for name in ["all", "chosen10"]:
        all_dfs = []
        relevant_data = relevant_trials
        if name == "all":
            relevant_data = relevant_data[~relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = relevant_data[relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]

        relevant_data = relevant_data[((relevant_data["stimulusType"] == "Face") | (relevant_data["stimulusType"] == "Object"))]
        means_df = relevant_data.groupby([SUBJECT_NAME, MODALITY]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, MODALITY, "FirstFixDur", "FirstFixTimePostOnset"]]
        means_df2 = means_df.groupby([MODALITY]).mean().reset_index()
        means_df2.rename(columns={MODALITY: INDEPENDENT, "FirstFixDur": "vg_fixdur_mean", "FirstFixTimePostOnset": "vg_fixonset_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([MODALITY]).std().reset_index()
        std_df.rename(columns={MODALITY: INDEPENDENT, "FirstFixDur": "vg_fixdur_std", "FirstFixTimePostOnset": "vg_fixonset_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, TARGET]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, TARGET, "FirstFixDur", "FirstFixTimePostOnset"]]
        means_df2 = means_df.groupby([TARGET]).mean().reset_index()
        means_df2.rename(columns={TARGET: INDEPENDENT, "FirstFixDur": "vg_fixdur_mean", "FirstFixTimePostOnset": "vg_fixonset_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([TARGET]).std().reset_index()
        std_df.rename(columns={TARGET: INDEPENDENT, "FirstFixDur": "vg_fixdur_std", "FirstFixTimePostOnset": "vg_fixonset_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, "stimulusType"]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, "stimulusType", "FirstFixDur", "FirstFixTimePostOnset"]]
        means_df2 = means_df.groupby(["stimulusType"]).mean().reset_index()
        means_df2.rename(columns={"stimulusType": INDEPENDENT, "FirstFixDur": "vg_fixdur_mean", "FirstFixTimePostOnset": "vg_fixonset_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby(["stimulusType"]).std().reset_index()
        std_df.rename(columns={"stimulusType": INDEPENDENT, "FirstFixDur": "vg_fixdur_std", "FirstFixTimePostOnset": "vg_fixonset_std"}, inplace=True)
        all_dfs.append(std_df)

        merged_df = pd.concat(all_dfs)
        merged_df.to_csv(os.path.join(save_path, f"vg_poststim_fixdur_per_vis_{name}_descriptives_replay.csv"), index=False)
    return


def stim_duration_saccade_num_analysis_new(all_subs_df, save_path, w_micro=True, is_250_500=False):
    if w_micro:
        w_micro_str = "withmicro"
    else:
        w_micro_str = "womicro"

    time_str = "StimDuration"

    if is_250_500:
        w_micro_str += "_250_500"
        time_str = "P250P500"

    relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == False)]

    # because if there are 0 saccades in pre stim we want to mean WITH it, fill na with zeros
    relevant_trials = relevant_trials.fillna(0)
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(LEFT), "horizontal"] = LEFT
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(RIGHT), "horizontal"] = RIGHT
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(TOP), "vertical"] = TOP
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(BOTTOM), "vertical"] = BOTTOM
    relevant_trials = relevant_trials[(relevant_trials[DataParser.VIS] == "TruePositive") | (relevant_trials[DataParser.VIS] == "FalseNegative")]
    if w_micro:
        relevant_trials["StimDurationAllNumSaccs"] = relevant_trials[f"{time_str}MicroNumSaccs"] + relevant_trials[f"{time_str}NumSaccs"]
    else:
        relevant_trials["StimDurationAllNumSaccs"] = relevant_trials[f"{time_str}NumSaccs"]

    means_df = relevant_trials.groupby([SUBJECT_NAME, DataParser.VIS, "stimulusType", MODALITY, "horizontal", "vertical"]).mean().reset_index()
    means_df = means_df[[SUBJECT_NAME, DataParser.VIS, "stimulusType", "StimDurationAllNumSaccs", MODALITY, "horizontal", "vertical"]]
    means_df.rename(
        columns={SUBJECT_NAME: "sub_code", DataParser.VIS: VIS,
                 "StimDurationAllNumSaccs": f"vg_numsaccs", "stimulusType": "category"},
        inplace=True)
    means_df[VIS].replace({"TruePositive": "seen", "FalseNegative": "unseen"}, inplace=True)
    means_df.to_csv(os.path.join(save_path, f"vg_poststim_sacc_per_vis_{w_micro_str}_all.csv"), index=False)

    relevant_trials = relevant_trials[[SUBJECT_NAME, DataParser.VIS, "stimulusType", "StimDurationAllNumSaccs", MODALITY, "horizontal", "vertical"]]

    # Without a variable
    for name in ["nochosen10", "chosen10"]:
        relevant_data = relevant_trials[(relevant_trials[DataParser.VIS] == "TruePositive") | (relevant_trials[DataParser.VIS] == "FalseNegative")]
        if name == "nochosen10":
            relevant_data = relevant_data[~relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = relevant_data[relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]

        list_to_combo_from = [MODALITY, "stimulusType", DataParser.VIS, "horizontal", "vertical"]
        combos = []
        for r in range(1, len(list_to_combo_from) + 1):
            combos.append(itertools.combinations(list_to_combo_from, r))

        stats_df_list = []
        df_to_mean_list = []
        for combo in combos:
            for x in combo:
                if "mod" not in list(x):
                    df_to_mean = relevant_data.groupby(list(x)+[SUBJECT_NAME, MODALITY]).mean().reset_index()
                else:
                    df_to_mean = relevant_data.groupby(list(x) + [SUBJECT_NAME]).mean().reset_index()
                mean_df = df_to_mean.groupby(list(x)).mean().reset_index()
                std_df = df_to_mean.groupby(list(x)).std().reset_index()
                for col in ["StimDurationAllNumSaccs"]:
                    mean_df.loc[:, f"{col}_std"] = std_df[col]
                    mean_df.rename(columns={col: f"{col}_mean"}, inplace=True)
                stats_df_list.append(mean_df)
                df_to_mean_list.append(df_to_mean)

        df_to_mean = pd.concat(df_to_mean_list)
        df_to_mean.to_csv(os.path.join(save_path, f"vg_poststim_sacc_per_vis_{name}_{w_micro_str}_sub_stats.csv"), index=False)
        stat_df = pd.concat(stats_df_list)
        stat_df.to_csv(os.path.join(save_path, f"vg_poststim_sacc_per_vis_{name}_{w_micro_str}_descriptives.csv"), index=False)

    # NEW PLOT
    relevant_data = relevant_trials[
        (relevant_trials[DataParser.VIS] == "TruePositive") | (relevant_trials[DataParser.VIS] == "FalseNegative")]
    relevant_data = relevant_data[~relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]

    for modality in [ET_qc_manager.FMRI, ET_qc_manager.MEG]:
        relevant_data2 = relevant_data[relevant_data[MODALITY] == modality]
        df_to_mean = relevant_data2.groupby([SUBJECT_NAME, DataParser.VIS]).mean().reset_index()
        seen = df_to_mean[df_to_mean[DataParser.VIS] == "TruePositive"]
        unseen = df_to_mean[df_to_mean[DataParser.VIS] == "FalseNegative"]
        merged_df = pd.merge(seen, unseen, on=SUBJECT_NAME, suffixes=("_seen", "_unseen"))
        merged_df = merged_df.loc[:, [SUBJECT_NAME, "StimDurationAllNumSaccs_seen", "StimDurationAllNumSaccs_unseen"]]
        colors = ["#1F7A8C", "#90141E"]
        plotter.make_it_rain(data=merged_df, cols=["StimDurationAllNumSaccs_seen", "StimDurationAllNumSaccs_unseen"],
                             col_names=["seen", "unseen"],
                             color_list=colors, plot_title="Saccade Number Per Visibility In Game",
                             plot_x_label="Visibility", plot_y_label="Saccade Number",
                             sub_line=[list(range(len(colors)))], save_plot=True,
                             save_path=save_path, save_folder="./", horizontal_lines=None,
                             save_name=f"vg_poststim_sacc_per_vis_{modality}_{w_micro_str}", custom_ylim=5, skip=1)

    df_to_mean = relevant_data.groupby([SUBJECT_NAME, DataParser.VIS]).mean().reset_index()
    seen = df_to_mean[df_to_mean[DataParser.VIS] == "TruePositive"]
    unseen = df_to_mean[df_to_mean[DataParser.VIS] == "FalseNegative"]
    merged_df = pd.merge(seen, unseen, on=SUBJECT_NAME, suffixes=("_seen", "_unseen"))
    merged_df = merged_df.loc[:, [SUBJECT_NAME, "StimDurationAllNumSaccs_seen", "StimDurationAllNumSaccs_unseen"]]
    colors = ["#1F7A8C", "#90141E"]
    plotter.make_it_rain(data=merged_df, cols=["StimDurationAllNumSaccs_seen", "StimDurationAllNumSaccs_unseen"],
                         col_names=["seen", "unseen"],
                         color_list=colors, plot_title="Saccade Number Per Visibility In Game",
                         plot_x_label="Visibility", plot_y_label="Saccade Number",
                         sub_line=[list(range(len(colors)))], save_plot=True,
                         save_path=save_path, save_folder="./", horizontal_lines=None,
                         save_name=f"vg_poststim_sacc_per_vis_{w_micro_str}", custom_ylim=5, skip=1)
    return


def stim_duration_saccade_num_analysis(all_subs_df, save_path):
    relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == False)]

    # because if there are 0 saccades in pre stim we want to mean WITH it, fill na with zeros
    relevant_trials = relevant_trials.fillna(0)

    means_df = relevant_trials.groupby([SUBJECT_NAME, DataParser.VIS, "stimulusType", MODALITY]).mean().reset_index()
    means_df = means_df[[SUBJECT_NAME, DataParser.VIS, "stimulusType", "StimDurationMicroNumSaccs", "StimDurationNumSaccs", MODALITY]]
    df_list = []
    for vis in ["TruePositive", "FalseNegative"]:
        for stim_type in ["Face", "Object"]:
            vis_type_df = means_df[(means_df[DataParser.VIS] == vis) & (means_df["stimulusType"] == stim_type)]
            vis_str = "seen" if "True" in vis else "unseen"
            stim_str = stim_type.lower()
            vis_type_df["StimDurationAllNumSaccs"] = vis_type_df["StimDurationMicroNumSaccs"] + vis_type_df["StimDurationNumSaccs"]
            vis_type_df = vis_type_df[[SUBJECT_NAME, "StimDurationAllNumSaccs", MODALITY]]

            vis_type_df.rename(
                columns={SUBJECT_NAME: "sub_code", "StimDurationAllNumSaccs": f"vg_numsaccs_{vis_str}_{stim_str}"},
                inplace=True)
            df_list.append(vis_type_df)
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='inner'), df_list)
    partial_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='outer'), df_list)
    partial_set = set(partial_df["sub_code"])

    # format to R format
    df_list = list()

    for vis in ["TruePositive", "FalseNegative"]:
        for stim_type in ["Face", "Object"]:
            stim_str = stim_type.lower()
            vis_str = "seen" if "True" in vis else "unseen"
            relevant_col = f"vg_numsaccs_{vis_str}_{stim_str}"
            temp_df = merged_df[["sub_code", relevant_col, MODALITY]]
            temp_df[VIS] = vis_str
            temp_df["category"] = stim_str
            temp_df.rename(columns={relevant_col: "vg_numsaccs"}, inplace=True)
            df_list.append(temp_df)

    merged_df = pd.concat(df_list)
    merged_df.to_csv(os.path.join(save_path, f"vg_poststim_sacc_per_vis.csv"), index=False)
    all_missing_subs = list(set(all_subs_df[SUBJECT_NAME]) - set(merged_df["sub_code"]))
    missing_reason = ["Partial Data" if x in partial_set else "No Data" for x in all_missing_subs]
    missing_subs = pd.DataFrame({"subCode": all_missing_subs, "reason": missing_reason})
    missing_subs.to_csv(os.path.join(save_path, f"vg_poststim_sacc_per_vis_missing_subs.csv"), index=False)

    # Without a variable
    for name in ["all", "chosen10"]:
        all_dfs = []
        relevant_data = relevant_trials[
            (relevant_trials[DataParser.VIS] == "TruePositive") | (relevant_trials[DataParser.VIS] == "FalseNegative")]
        if name == "all":
            relevant_data = relevant_data[~relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = relevant_data[relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]

        relevant_data = relevant_data[
            ((relevant_data[DataParser.VIS] == "TruePositive") | (relevant_data[DataParser.VIS] == "FalseNegative")) &
            ((relevant_data["stimulusType"] == "Face") | (relevant_data["stimulusType"] == "Object"))]

        relevant_data["StimDurationAllNumSaccs"] = relevant_data["StimDurationMicroNumSaccs"] + relevant_data["StimDurationNumSaccs"]
        means_df = relevant_data.groupby([SUBJECT_NAME, MODALITY]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, MODALITY, "StimDurationAllNumSaccs"]]
        means_df2 = means_df.groupby([MODALITY]).mean().reset_index()
        means_df2.rename(columns={MODALITY: INDEPENDENT, "StimDurationAllNumSaccs": "vg_numsaccs_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([MODALITY]).std().reset_index()
        std_df.rename(columns={MODALITY: INDEPENDENT, "StimDurationAllNumSaccs": "vg_numsaccs_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, DataParser.VIS]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, DataParser.VIS, "StimDurationAllNumSaccs"]]
        means_df2 = means_df.groupby([DataParser.VIS]).mean().reset_index()
        means_df2.rename(columns={DataParser.VIS: INDEPENDENT, "StimDurationAllNumSaccs": "vg_numsaccs_mean"},
                         inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([DataParser.VIS]).std().reset_index()
        std_df.rename(columns={DataParser.VIS: INDEPENDENT, "StimDurationAllNumSaccs": "vg_numsaccs_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, "stimulusType"]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, "stimulusType", "StimDurationAllNumSaccs"]]
        means_df2 = means_df.groupby(["stimulusType"]).mean().reset_index()
        means_df2.rename(columns={"stimulusType": INDEPENDENT, "StimDurationAllNumSaccs": "vg_numsaccs_mean"},
                         inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby(["stimulusType"]).std().reset_index()
        std_df.rename(columns={"stimulusType": INDEPENDENT, "StimDurationAllNumSaccs": "vg_numsaccs_std"}, inplace=True)
        all_dfs.append(std_df)

        merged_df = pd.concat(all_dfs)
        merged_df.to_csv(os.path.join(save_path, f"vg_poststim_sacc_per_vis_{name}_descriptives.csv"), index=False)
    return


def stim_duration_saccade_num_analysis_left_right(all_subs_df, save_path, left_right=True):
    relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == False)]

    if left_right:
        savename = "left_right"
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(RIGHT), "location"] = RIGHT
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(LEFT), "location"] = LEFT
        locations = [LEFT, RIGHT]
    else:
        savename = "top_bottom"
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(TOP), "location"] = TOP
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(BOTTOM), "location"] = BOTTOM
        locations = [TOP, BOTTOM]

    # because if there are 0 saccades in pre stim we want to mean WITH it, fill na with zeros
    relevant_trials = relevant_trials.fillna(0)

    means_df = relevant_trials.groupby([SUBJECT_NAME, DataParser.VIS, "location", MODALITY]).mean().reset_index()
    means_df = means_df[[SUBJECT_NAME, DataParser.VIS, "location", "StimDurationMicroNumSaccs", "StimDurationNumSaccs", MODALITY]]
    df_list = []
    for vis in ["TruePositive", "FalseNegative"]:
        for stim_type in locations:
            vis_type_df = means_df[(means_df[DataParser.VIS] == vis) & (means_df["location"] == stim_type)]
            vis_str = "seen" if "True" in vis else "unseen"
            stim_str = stim_type.lower()
            vis_type_df["StimDurationAllNumSaccs"] = vis_type_df["StimDurationMicroNumSaccs"] + vis_type_df["StimDurationNumSaccs"]
            vis_type_df = vis_type_df[[SUBJECT_NAME, "StimDurationAllNumSaccs", MODALITY]]

            vis_type_df.rename(
                columns={SUBJECT_NAME: "sub_code", "StimDurationAllNumSaccs": f"vg_numsaccs_{vis_str}_{stim_str}"},
                inplace=True)
            df_list.append(vis_type_df)
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='inner'), df_list)
    partial_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='outer'), df_list)
    partial_set = set(partial_df["sub_code"])

    # format to R format
    df_list = list()

    for vis in ["TruePositive", "FalseNegative"]:
        for stim_type in locations:
            stim_str = stim_type.lower()
            vis_str = "seen" if "True" in vis else "unseen"
            relevant_col = f"vg_numsaccs_{vis_str}_{stim_str}"
            temp_df = merged_df[["sub_code", relevant_col, MODALITY]]
            temp_df[VIS] = vis_str
            temp_df["category"] = stim_str
            temp_df.rename(columns={relevant_col: "vg_numsaccs"}, inplace=True)
            df_list.append(temp_df)

    merged_df = pd.concat(df_list)
    merged_df.to_csv(os.path.join(save_path, f"vg_poststim_sacc_per_vis_{savename}.csv"), index=False)
    all_missing_subs = list(set(all_subs_df[SUBJECT_NAME]) - set(merged_df["sub_code"]))
    missing_reason = ["Partial Data" if x in partial_set else "No Data" for x in all_missing_subs]
    missing_subs = pd.DataFrame({"subCode": all_missing_subs, "reason": missing_reason})
    missing_subs.to_csv(os.path.join(save_path, f"vg_poststim_sacc_per_vis_{savename}_missing_subs.csv"), index=False)

    # Without a variable
    for name in ["all", "chosen10"]:
        all_dfs = []
        relevant_data = relevant_trials[
            (relevant_trials[DataParser.VIS] == "TruePositive") | (relevant_trials[DataParser.VIS] == "FalseNegative")]
        if name == "all":
            relevant_data = relevant_data[~relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = relevant_data[relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]

        relevant_data = relevant_data[
            ((relevant_data[DataParser.VIS] == "TruePositive") | (relevant_data[DataParser.VIS] == "FalseNegative")) &
            ((relevant_data["stimulusType"] == "Face") | (relevant_data["stimulusType"] == "Object"))]

        relevant_data["StimDurationAllNumSaccs"] = relevant_data["StimDurationMicroNumSaccs"] + relevant_data["StimDurationNumSaccs"]
        means_df = relevant_data.groupby([SUBJECT_NAME, MODALITY]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, MODALITY, "StimDurationAllNumSaccs"]]
        means_df2 = means_df.groupby([MODALITY]).mean().reset_index()
        means_df2.rename(columns={MODALITY: INDEPENDENT, "StimDurationAllNumSaccs": "vg_numsaccs_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([MODALITY]).std().reset_index()
        std_df.rename(columns={MODALITY: INDEPENDENT, "StimDurationAllNumSaccs": "vg_numsaccs_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, DataParser.VIS]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, DataParser.VIS, "StimDurationAllNumSaccs"]]
        means_df2 = means_df.groupby([DataParser.VIS]).mean().reset_index()
        means_df2.rename(columns={DataParser.VIS: INDEPENDENT, "StimDurationAllNumSaccs": "vg_numsaccs_mean"},
                         inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([DataParser.VIS]).std().reset_index()
        std_df.rename(columns={DataParser.VIS: INDEPENDENT, "StimDurationAllNumSaccs": "vg_numsaccs_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, "location"]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, "location", "StimDurationAllNumSaccs"]]
        means_df2 = means_df.groupby(["location"]).mean().reset_index()
        means_df2.rename(columns={"location": INDEPENDENT, "StimDurationAllNumSaccs": "vg_numsaccs_mean"},
                         inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby(["location"]).std().reset_index()
        std_df.rename(columns={"location": INDEPENDENT, "StimDurationAllNumSaccs": "vg_numsaccs_std"}, inplace=True)
        all_dfs.append(std_df)

        merged_df = pd.concat(all_dfs)
        merged_df.to_csv(os.path.join(save_path, f"vg_poststim_sacc_per_vis_{savename}_{name}_descriptives.csv"), index=False)
    return


def stim_duration_saccade_num_analysis_replay_new(all_subs_df, save_path):
    relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == True)]
    relevant_trials.loc[:, TARGET] = relevant_trials[QualityChecker.STIM_TYPE] == relevant_trials["TargetType"]

    # because if there are 0 saccades in pre stim we want to mean WITH it, fill na with zeros
    relevant_trials = relevant_trials.fillna(0)
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(LEFT), "horizontal"] = LEFT
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(RIGHT), "horizontal"] = RIGHT
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(TOP), "vertical"] = TOP
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(BOTTOM), "vertical"] = BOTTOM

    relevant_trials = relevant_trials[((relevant_trials["stimulusType"] == "Face") | (relevant_trials["stimulusType"] == "Object"))]
    relevant_trials["StimDurationAllNumSaccs"] = relevant_trials["StimDurationMicroNumSaccs"] + relevant_trials["StimDurationNumSaccs"]

    means_df = relevant_trials.groupby([SUBJECT_NAME, TARGET, "stimulusType", MODALITY, "horizontal", "vertical"]).mean().reset_index()
    means_df = means_df[[SUBJECT_NAME, TARGET, "stimulusType", "StimDurationAllNumSaccs", MODALITY, "horizontal", "vertical"]]

    means_df.rename(
        columns={SUBJECT_NAME: "sub_code",
                 "StimDurationAllNumSaccs": f"vg_numsaccs", "stimulusType": "category"},
        inplace=True)
    means_df[TARGET].replace({True: "target", False: "non_target"}, inplace=True)
    means_df.to_csv(os.path.join(save_path, f"replay_poststim_sacc_per_vis.csv"), index=False)

    relevant_trials = relevant_trials[[SUBJECT_NAME, TARGET, "stimulusType", "StimDurationAllNumSaccs", MODALITY, "horizontal", "vertical"]]
    relevant_trials[TARGET].replace({True: "target", False: "non_target"}, inplace=True)
    # Without a variable
    for name in ["all", "chosen10"]:
        if name == "all":
            relevant_data = relevant_trials[~relevant_trials[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = relevant_trials[relevant_trials[SUBJECT_NAME].isin(CHOSEN_10)]

        list_to_combo_from = [MODALITY, "stimulusType", TARGET, "horizontal", "vertical"]
        combos = []
        for r in range(1, len(list_to_combo_from) + 1):
            combos.append(itertools.combinations(list_to_combo_from, r))

        stats_df_list = []
        df_to_mean_list = []
        for combo in combos:
            for x in combo:
                if "mod" not in list(x):
                    df_to_mean = relevant_data.groupby(list(x) + [SUBJECT_NAME, MODALITY]).mean().reset_index()
                else:
                    df_to_mean = relevant_data.groupby(list(x) + [SUBJECT_NAME]).mean().reset_index()
                mean_df = df_to_mean.groupby(list(x)).mean().reset_index()
                std_df = df_to_mean.groupby(list(x)).std().reset_index()
                for col in ["StimDurationAllNumSaccs"]:
                    mean_df.loc[:, f"{col}_std"] = std_df[col]
                    mean_df.rename(columns={col: f"{col}_mean"}, inplace=True)
                stats_df_list.append(mean_df)
                df_to_mean_list.append(df_to_mean)

        df_to_mean = pd.concat(df_to_mean_list)
        df_to_mean.to_csv(os.path.join(save_path, f"replay_poststim_sacc_per_vis_{name}_sub_stats.csv"),
                          index=False)
        stat_df = pd.concat(stats_df_list)
        stat_df.to_csv(os.path.join(save_path, f"replay_poststim_sacc_per_vis_{name}_descriptives.csv"),
                       index=False)

    # NEW PLOT
    relevant_data = relevant_trials[~relevant_trials[SUBJECT_NAME].isin(CHOSEN_10)]

    for modality in [ET_qc_manager.FMRI, ET_qc_manager.MEG]:
        relevant_data2 = relevant_data[relevant_data[MODALITY] == modality]
        df_to_mean = relevant_data2.groupby([SUBJECT_NAME, TARGET]).mean().reset_index()

        target = df_to_mean[df_to_mean[TARGET] == "target"]
        non_target = df_to_mean[df_to_mean[TARGET] == "non_target"]
        merged_df = pd.merge(target, non_target, on=SUBJECT_NAME, suffixes=("_seen", "_unseen"))
        merged_df = merged_df.loc[:, [SUBJECT_NAME, "StimDurationAllNumSaccs_seen", "StimDurationAllNumSaccs_unseen"]]
        colors = ["#1F7A8C", "#90141E"]
        plotter.make_it_rain(data=merged_df, cols=["StimDurationAllNumSaccs_seen", "StimDurationAllNumSaccs_unseen"], col_names=["Target", "Non Target"],
                             color_list=colors, plot_title="Saccade Number Per Target In Replay", plot_x_label="Target", plot_y_label="Saccade Number",
                             sub_line=[list(range(len(colors)))], save_plot=True,
                             save_path=save_path, save_folder="./", horizontal_lines=None,
                             save_name=f"replay_poststim_sacc_per_target_{modality}", custom_ylim=8, skip=1)

        df_to_mean = relevant_data2.groupby([SUBJECT_NAME, "vertical"]).mean().reset_index()

        target = df_to_mean[df_to_mean["vertical"] == TOP]
        non_target = df_to_mean[df_to_mean["vertical"] == BOTTOM]
        merged_df = pd.merge(target, non_target, on=SUBJECT_NAME, suffixes=("_seen", "_unseen"))
        merged_df = merged_df.loc[:, [SUBJECT_NAME, "StimDurationAllNumSaccs_seen", "StimDurationAllNumSaccs_unseen"]]
        colors = ["#1F7A8C", "#90141E"]
        plotter.make_it_rain(data=merged_df, cols=["StimDurationAllNumSaccs_seen", "StimDurationAllNumSaccs_unseen"], col_names=[TOP, BOTTOM],
                             color_list=colors, plot_title="Saccade Number Per Vertical Location In Replay", plot_x_label="Vertical Location", plot_y_label="Saccade Number",
                             sub_line=[list(range(len(colors)))], save_plot=True,
                             save_path=save_path, save_folder="./", horizontal_lines=None,
                             save_name=f"replay_poststim_sacc_per_vertical_{modality}", custom_ylim=8, skip=1)

    return


def stim_duration_saccade_num_analysis_replay(all_subs_df, save_path):
    relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == True)]
    relevant_trials.loc[:, TARGET] = relevant_trials[QualityChecker.STIM_TYPE] == relevant_trials["TargetType"]

    # because if there are 0 saccades in pre stim we want to mean WITH it, fill na with zeros
    relevant_trials = relevant_trials.fillna(0)

    means_df = relevant_trials.groupby([SUBJECT_NAME, TARGET, "stimulusType", MODALITY]).mean().reset_index()
    means_df = means_df[[SUBJECT_NAME, TARGET, "stimulusType", "StimDurationMicroNumSaccs", "StimDurationNumSaccs", MODALITY]]
    df_list = []
    for vis in [True, False]:
        for stim_type in ["Face", "Object"]:
            vis_type_df = means_df[(means_df[TARGET] == vis) & (means_df["stimulusType"] == stim_type)]
            vis_str = "target" if vis==True else "non_target"
            stim_str = stim_type.lower()
            vis_type_df["StimDurationAllNumSaccs"] = vis_type_df["StimDurationMicroNumSaccs"] + vis_type_df["StimDurationNumSaccs"]
            vis_type_df = vis_type_df[[SUBJECT_NAME, "StimDurationAllNumSaccs", MODALITY]]

            vis_type_df.rename(
                columns={SUBJECT_NAME: "sub_code", "StimDurationAllNumSaccs": f"vg_numsaccs_{vis_str}_{stim_str}"},
                inplace=True)
            df_list.append(vis_type_df)
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='inner'), df_list)
    partial_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='outer'), df_list)
    partial_set = set(partial_df["sub_code"])

    # format to R format
    df_list = list()

    for vis in [True, False]:
        for stim_type in ["Face", "Object"]:
            stim_str = stim_type.lower()
            vis_str = "target" if vis==True else "non_target"
            relevant_col = f"vg_numsaccs_{vis_str}_{stim_str}"
            temp_df = merged_df[["sub_code", relevant_col, MODALITY]]
            temp_df[TARGET] = vis_str
            temp_df["category"] = stim_str
            temp_df.rename(columns={relevant_col: "vg_numsaccs"}, inplace=True)
            df_list.append(temp_df)

    merged_df = pd.concat(df_list)
    merged_df.to_csv(os.path.join(save_path, f"vg_poststim_sacc_per_vis_replay.csv"), index=False)
    all_missing_subs = list(set(all_subs_df[SUBJECT_NAME]) - set(merged_df["sub_code"]))
    missing_reason = ["Partial Data" if x in partial_set else "No Data" for x in all_missing_subs]
    missing_subs = pd.DataFrame({"subCode": all_missing_subs, "reason": missing_reason})
    missing_subs.to_csv(os.path.join(save_path, f"vg_poststim_sacc_per_vis_missing_subs_replay.csv"), index=False)

    # Without a variable
    for name in ["all", "chosen10"]:
        all_dfs = []
        relevant_data = relevant_trials
        if name == "all":
            relevant_data = relevant_data[~relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = relevant_data[relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]

        relevant_data = relevant_data[((relevant_data["stimulusType"] == "Face") | (relevant_data["stimulusType"] == "Object"))]

        relevant_data["StimDurationAllNumSaccs"] = relevant_data["StimDurationMicroNumSaccs"] + relevant_data["StimDurationNumSaccs"]
        means_df = relevant_data.groupby([SUBJECT_NAME, MODALITY]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, MODALITY, "StimDurationAllNumSaccs"]]
        means_df2 = means_df.groupby([MODALITY]).mean().reset_index()
        means_df2.rename(columns={MODALITY: INDEPENDENT, "StimDurationAllNumSaccs": "vg_numsaccs_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([MODALITY]).std().reset_index()
        std_df.rename(columns={MODALITY: INDEPENDENT, "StimDurationAllNumSaccs": "vg_numsaccs_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, TARGET]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, TARGET, "StimDurationAllNumSaccs"]]
        means_df2 = means_df.groupby([TARGET]).mean().reset_index()
        means_df2.rename(columns={TARGET: INDEPENDENT, "StimDurationAllNumSaccs": "vg_numsaccs_mean"},
                         inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([TARGET]).std().reset_index()
        std_df.rename(columns={TARGET: INDEPENDENT, "StimDurationAllNumSaccs": "vg_numsaccs_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, "stimulusType"]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, "stimulusType", "StimDurationAllNumSaccs"]]
        means_df2 = means_df.groupby(["stimulusType"]).mean().reset_index()
        means_df2.rename(columns={"stimulusType": INDEPENDENT, "StimDurationAllNumSaccs": "vg_numsaccs_mean"},
                         inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby(["stimulusType"]).std().reset_index()
        std_df.rename(columns={"stimulusType": INDEPENDENT, "StimDurationAllNumSaccs": "vg_numsaccs_std"}, inplace=True)
        all_dfs.append(std_df)

        merged_df = pd.concat(all_dfs)
        merged_df.to_csv(os.path.join(save_path, f"vg_poststim_sacc_per_vis_{name}_descriptives_replay.csv"), index=False)
    return


def stim_duration_saccade_num_analysis_left_right_replay(all_subs_df, save_path, left_right=True):
    relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == True)]
    relevant_trials.loc[:, TARGET] = relevant_trials[QualityChecker.STIM_TYPE] == relevant_trials["TargetType"]

    if left_right:
        savename = "left_right"
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(RIGHT), "location"] = RIGHT
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(LEFT), "location"] = LEFT
        locations = [LEFT, RIGHT]
    else:
        savename = "top_bottom"
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(TOP), "location"] = TOP
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(BOTTOM), "location"] = BOTTOM
        locations = [TOP, BOTTOM]

    # because if there are 0 saccades in pre stim we want to mean WITH it, fill na with zeros
    relevant_trials = relevant_trials.fillna(0)

    means_df = relevant_trials.groupby([SUBJECT_NAME, TARGET, "location", MODALITY]).mean().reset_index()
    means_df = means_df[[SUBJECT_NAME, TARGET, "location", "StimDurationMicroNumSaccs", "StimDurationNumSaccs", MODALITY]]
    df_list = []
    for vis in [True, False]:
        for stim_type in locations:
            vis_type_df = means_df[(means_df[TARGET] == vis) & (means_df["location"] == stim_type)]
            vis_str = "target" if vis==True else "non_target"
            stim_str = stim_type.lower()
            vis_type_df["StimDurationAllNumSaccs"] = vis_type_df["StimDurationMicroNumSaccs"] + vis_type_df["StimDurationNumSaccs"]
            vis_type_df = vis_type_df[[SUBJECT_NAME, "StimDurationAllNumSaccs", MODALITY]]

            vis_type_df.rename(
                columns={SUBJECT_NAME: "sub_code", "StimDurationAllNumSaccs": f"vg_numsaccs_{vis_str}_{stim_str}"},
                inplace=True)
            df_list.append(vis_type_df)
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='inner'), df_list)
    partial_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='outer'), df_list)
    partial_set = set(partial_df["sub_code"])

    # format to R format
    df_list = list()

    for vis in [True, False]:
        for stim_type in locations:
            stim_str = stim_type.lower()
            vis_str = "target" if vis==True else "non_target"
            relevant_col = f"vg_numsaccs_{vis_str}_{stim_str}"
            temp_df = merged_df[["sub_code", relevant_col, MODALITY]]
            temp_df[TARGET] = vis_str
            temp_df["location"] = stim_str
            temp_df.rename(columns={relevant_col: "vg_numsaccs"}, inplace=True)
            df_list.append(temp_df)

    merged_df = pd.concat(df_list)
    merged_df.to_csv(os.path.join(save_path, f"vg_poststim_sacc_per_vis_{savename}_replay.csv"), index=False)
    all_missing_subs = list(set(all_subs_df[SUBJECT_NAME]) - set(merged_df["sub_code"]))
    missing_reason = ["Partial Data" if x in partial_set else "No Data" for x in all_missing_subs]
    missing_subs = pd.DataFrame({"subCode": all_missing_subs, "reason": missing_reason})
    missing_subs.to_csv(os.path.join(save_path, f"vg_poststim_sacc_per_vis_{savename}_missing_subs_replay.csv"), index=False)

    # Without a variable
    for name in ["all", "chosen10"]:
        all_dfs = []
        relevant_data = relevant_trials
        if name == "all":
            relevant_data = relevant_data[~relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = relevant_data[relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]

        relevant_data = relevant_data[((relevant_data["stimulusType"] == "Face") | (relevant_data["stimulusType"] == "Object"))]

        relevant_data["StimDurationAllNumSaccs"] = relevant_data["StimDurationMicroNumSaccs"] + relevant_data["StimDurationNumSaccs"]
        means_df = relevant_data.groupby([SUBJECT_NAME, MODALITY]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, MODALITY, "StimDurationAllNumSaccs"]]
        means_df2 = means_df.groupby([MODALITY]).mean().reset_index()
        means_df2.rename(columns={MODALITY: INDEPENDENT, "StimDurationAllNumSaccs": "vg_numsaccs_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([MODALITY]).std().reset_index()
        std_df.rename(columns={MODALITY: INDEPENDENT, "StimDurationAllNumSaccs": "vg_numsaccs_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, TARGET]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, TARGET, "StimDurationAllNumSaccs"]]
        means_df2 = means_df.groupby([TARGET]).mean().reset_index()
        means_df2.rename(columns={TARGET: INDEPENDENT, "StimDurationAllNumSaccs": "vg_numsaccs_mean"},
                         inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([TARGET]).std().reset_index()
        std_df.rename(columns={TARGET: INDEPENDENT, "StimDurationAllNumSaccs": "vg_numsaccs_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, "location"]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, "location", "StimDurationAllNumSaccs"]]
        means_df2 = means_df.groupby(["location"]).mean().reset_index()
        means_df2.rename(columns={"location": INDEPENDENT, "StimDurationAllNumSaccs": "vg_numsaccs_mean"},
                         inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby(["location"]).std().reset_index()
        std_df.rename(columns={"location": INDEPENDENT, "StimDurationAllNumSaccs": "vg_numsaccs_std"}, inplace=True)
        all_dfs.append(std_df)

        merged_df = pd.concat(all_dfs)
        merged_df.to_csv(os.path.join(save_path, f"vg_poststim_sacc_per_vis_{savename}_{name}_descriptives_replay.csv"), index=False)
    return


def poststim_blink_num_analysis_left_right(all_subs_df, save_path, left_right=False):
    relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == False)]

    if left_right:
        savename = "left_right"
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(RIGHT), "location"] = RIGHT
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(LEFT), "location"] = LEFT
        locations = [LEFT, RIGHT]
    else:
        savename = "top_bottom"
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(TOP), "location"] = TOP
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(BOTTOM), "location"] = BOTTOM
        locations = [TOP, BOTTOM]

    # because if there are 0 saccades in pre stim we want to mean WITH it, fill na with zeros
    relevant_trials = relevant_trials.fillna(0)
    means_df = relevant_trials.groupby([SUBJECT_NAME, DataParser.VIS, "location", MODALITY]).mean().reset_index()
    means_df = means_df[[SUBJECT_NAME, DataParser.VIS, "location", "StimDurationNumBlinks", MODALITY]]
    df_list = []
    for vis in ["TruePositive", "FalseNegative"]:
        for stim_type in locations:
            vis_type_df = means_df[(means_df[DataParser.VIS] == vis) & (means_df["location"] == stim_type)]
            vis_str = "seen" if "True" in vis else "unseen"
            stim_str = stim_type.lower()
            vis_type_df = vis_type_df[[SUBJECT_NAME, "StimDurationNumBlinks", MODALITY]]
            vis_type_df.rename(
                columns={SUBJECT_NAME: "sub_code", "StimDurationNumBlinks": f"vg_numblinks_{vis_str}_{stim_str}"},
                inplace=True)
            df_list.append(vis_type_df)
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='inner'), df_list)
    partial_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='outer'), df_list)
    partial_set = set(partial_df["sub_code"])

    # format to R format
    df_list = list()

    for vis in ["TruePositive", "FalseNegative"]:
        for stim_type in locations:
            stim_str = stim_type.lower()
            vis_str = "seen" if "True" in vis else "unseen"
            relevant_col = f"vg_numblinks_{vis_str}_{stim_str}"
            temp_df = merged_df[["sub_code", relevant_col, MODALITY]]
            temp_df[VIS] = vis_str
            temp_df["location"] = stim_str
            temp_df.rename(columns={relevant_col: "vg_numblinks"}, inplace=True)
            df_list.append(temp_df)

    merged_df = pd.concat(df_list)
    merged_df.to_csv(os.path.join(save_path, f"vg_poststim_blink_{savename}_per_vis.csv"), index=False)
    all_missing_subs = list(set(all_subs_df[SUBJECT_NAME]) - set(merged_df["sub_code"]))
    missing_reason = ["Partial Data" if x in partial_set else "No Data" for x in all_missing_subs]
    missing_subs = pd.DataFrame({"subCode": all_missing_subs, "reason": missing_reason})
    missing_subs.to_csv(os.path.join(save_path, f"vg_poststim_blink_{savename}_per_vis_missing_subs.csv"), index=False)

    # Without a variable
    for name in ["all", "chosen10"]:
        all_dfs = []
        relevant_data = relevant_trials[
            (relevant_trials[DataParser.VIS] == "TruePositive") | (relevant_trials[DataParser.VIS] == "FalseNegative")]
        if name == "all":
            relevant_data = relevant_data[~relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = relevant_data[relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]

        means_df = relevant_data.groupby([SUBJECT_NAME, MODALITY]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, MODALITY, "StimDurationNumBlinks"]]
        means_df2 = means_df.groupby([MODALITY]).mean().reset_index()
        means_df2.rename(columns={MODALITY: INDEPENDENT, "StimDurationNumBlinks": "vg_numblinks_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([MODALITY]).std().reset_index()
        std_df.rename(columns={MODALITY: INDEPENDENT, "StimDurationNumBlinks": "vg_numblinks_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, DataParser.VIS]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, DataParser.VIS, "StimDurationNumBlinks"]]
        means_df2 = means_df.groupby([DataParser.VIS]).mean().reset_index()
        means_df2.rename(columns={DataParser.VIS: INDEPENDENT, "StimDurationNumBlinks": "vg_numblinks_mean"},
                         inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([DataParser.VIS]).std().reset_index()
        std_df.rename(columns={DataParser.VIS: INDEPENDENT, "StimDurationNumBlinks": "vg_numblinks_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, "location"]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, "location", "StimDurationNumBlinks"]]
        means_df2 = means_df.groupby(["location"]).mean().reset_index()
        means_df2.rename(columns={"location": INDEPENDENT, "StimDurationNumBlinks": "vg_numblinks_mean"},
                         inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby(["location"]).std().reset_index()
        std_df.rename(columns={"location": INDEPENDENT, "StimDurationNumBlinks": "vg_numblinks_std"}, inplace=True)
        all_dfs.append(std_df)

        merged_df = pd.concat(all_dfs)
        merged_df.to_csv(os.path.join(save_path, f"vg_poststim_blink_{savename}_per_vis_{name}_descriptives.csv"), index=False)
    return


def poststim_blink_num_analysis_replay_left_right(all_subs_df, save_path, left_right=True):
    relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == True)]

    if left_right:
        savename = "left_right"
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(RIGHT), "location"] = RIGHT
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(LEFT), "location"] = LEFT
        locations = [LEFT, RIGHT]
    else:
        savename = "top_bottom"
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(TOP), "location"] = TOP
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(BOTTOM), "location"] = BOTTOM
        locations = [TOP, BOTTOM]

    relevant_trials.loc[:, TARGET] = relevant_trials[QualityChecker.STIM_TYPE] == relevant_trials["TargetType"]

    # because if there are 0 saccades in pre stim we want to mean WITH it, fill na with zeros
    relevant_trials = relevant_trials.fillna(0)
    means_df = relevant_trials.groupby([SUBJECT_NAME, TARGET, "location", MODALITY]).mean().reset_index()
    means_df = means_df[[SUBJECT_NAME, TARGET, "location", "StimDurationNumBlinks", MODALITY]]
    df_list = []
    for vis in [True, False]:
        for stim_type in locations:
            vis_type_df = means_df[(means_df[TARGET] == vis) & (means_df["location"] == stim_type)]
            vis_str = "target" if vis==True else "non_target"
            stim_str = stim_type.lower()
            vis_type_df = vis_type_df[[SUBJECT_NAME, "StimDurationNumBlinks", MODALITY]]
            vis_type_df.rename(
                columns={SUBJECT_NAME: "sub_code", "StimDurationNumBlinks": f"vg_numblinks_{vis_str}_{stim_str}"},
                inplace=True)
            df_list.append(vis_type_df)
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='inner'), df_list)
    partial_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='outer'), df_list)
    partial_set = set(partial_df["sub_code"])

    # format to R format
    df_list = list()

    for vis in [True, False]:
        for stim_type in locations:
            stim_str = stim_type.lower()
            vis_str = "target" if vis==True else "non_target"
            relevant_col = f"vg_numblinks_{vis_str}_{stim_str}"
            temp_df = merged_df[["sub_code", relevant_col, MODALITY]]
            temp_df[TARGET] = vis_str
            temp_df["location"] = stim_str
            temp_df.rename(columns={relevant_col: "vg_numblinks"}, inplace=True)
            df_list.append(temp_df)

    merged_df = pd.concat(df_list)
    merged_df.to_csv(os.path.join(save_path, f"vg_poststim_blink_{savename}_per_vis_replay.csv"), index=False)
    all_missing_subs = list(set(all_subs_df[SUBJECT_NAME]) - set(merged_df["sub_code"]))
    missing_reason = ["Partial Data" if x in partial_set else "No Data" for x in all_missing_subs]
    missing_subs = pd.DataFrame({"subCode": all_missing_subs, "reason": missing_reason})
    missing_subs.to_csv(os.path.join(save_path, f"vg_poststim_blink_{savename}_per_vis_missing_subs_replay.csv"), index=False)

    # Without a variable
    for name in ["all", "chosen10"]:
        all_dfs = []
        relevant_data = relevant_trials
        if name == "all":
            relevant_data = relevant_data[~relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = relevant_data[relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]

        relevant_data = relevant_data[((relevant_data["stimulusType"] == "Face") | (relevant_data["stimulusType"] == "Object"))]
        means_df = relevant_data.groupby([SUBJECT_NAME, MODALITY]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, MODALITY, "StimDurationNumBlinks"]]
        means_df2 = means_df.groupby([MODALITY]).mean().reset_index()
        means_df2.rename(columns={MODALITY: INDEPENDENT, "StimDurationNumBlinks": "vg_numblinks_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([MODALITY]).std().reset_index()
        std_df.rename(columns={MODALITY: INDEPENDENT, "StimDurationNumBlinks": "vg_numblinks_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, TARGET]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, TARGET, "StimDurationNumBlinks"]]
        means_df2 = means_df.groupby([TARGET]).mean().reset_index()
        means_df2.rename(columns={TARGET: INDEPENDENT, "StimDurationNumBlinks": "vg_numblinks_mean"},
                         inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([TARGET]).std().reset_index()
        std_df.rename(columns={TARGET: INDEPENDENT, "StimDurationNumBlinks": "vg_numblinks_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, "location"]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, "location", "StimDurationNumBlinks"]]
        means_df2 = means_df.groupby(["location"]).mean().reset_index()
        means_df2.rename(columns={"location": INDEPENDENT, "StimDurationNumBlinks": "vg_numblinks_mean"},
                         inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby(["location"]).std().reset_index()
        std_df.rename(columns={"location": INDEPENDENT, "StimDurationNumBlinks": "vg_numblinks_std"}, inplace=True)
        all_dfs.append(std_df)

        merged_df = pd.concat(all_dfs)
        merged_df.to_csv(os.path.join(save_path, f"vg_poststim_blink_{savename}_per_vis_{name}_descriptives_replay.csv"), index=False)
    return


def poststim_blink_num_analysis_new(all_subs_df, save_path, is_250_500=False):
    if is_250_500:
        num_blinks_col = "P250P500NumBlinks"
    else:
        num_blinks_col = "StimDurationNumBlinks"

    relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == False)]

    # We want only fixations in the timewindow [0, 500], since we have the time postonset,
    # take only those with this value smaller than 500
    relevant_trials = relevant_trials.fillna(0)
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(LEFT), "horizontal"] = LEFT
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(RIGHT), "horizontal"] = RIGHT
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(TOP), "vertical"] = TOP
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(BOTTOM), "vertical"] = BOTTOM
    relevant_trials = relevant_trials[(relevant_trials[DataParser.VIS] == "TruePositive") | (relevant_trials[DataParser.VIS] == "FalseNegative")]

    means_df = relevant_trials.groupby([SUBJECT_NAME, DataParser.VIS, "stimulusType", MODALITY, "horizontal", "vertical"]).mean().reset_index()
    means_df = means_df[[SUBJECT_NAME, DataParser.VIS, "stimulusType", num_blinks_col, MODALITY, "horizontal", "vertical"]]
    means_df.rename(
        columns={SUBJECT_NAME: "sub_code", DataParser.VIS: VIS,
                 num_blinks_col: f"vg_numblinks", "stimulusType": "category"},
        inplace=True)
    means_df[VIS].replace({"TruePositive": "seen", "FalseNegative": "unseen"}, inplace=True)
    means_df.to_csv(os.path.join(save_path, f"vg_poststim_blink_{is_250_500}_per_vis_all.csv"), index=False)

    relevant_trials = relevant_trials[[SUBJECT_NAME, DataParser.VIS, "stimulusType", num_blinks_col, MODALITY, "horizontal", "vertical"]]

    # Without a variable
    for name in ["nochosen10", "chosen10"]:
        relevant_data = relevant_trials[(relevant_trials[DataParser.VIS] == "TruePositive") | (relevant_trials[DataParser.VIS] == "FalseNegative")]
        if name == "nochosen10":
            relevant_data = relevant_data[~relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = relevant_data[relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]

        list_to_combo_from = [MODALITY, "stimulusType", DataParser.VIS, "horizontal", "vertical"]
        combos = []
        for r in range(1, len(list_to_combo_from) + 1):
            combos.append(itertools.combinations(list_to_combo_from, r))

        stats_df_list = []
        df_to_mean_list = []
        for combo in combos:
            for x in combo:
                if "mod" not in list(x):
                    df_to_mean = relevant_data.groupby(list(x)+[SUBJECT_NAME, MODALITY]).mean().reset_index()
                else:
                    df_to_mean = relevant_data.groupby(list(x) + [SUBJECT_NAME]).mean().reset_index()
                mean_df = df_to_mean.groupby(list(x)).mean().reset_index()
                std_df = df_to_mean.groupby(list(x)).std().reset_index()
                for col in [num_blinks_col]:
                    mean_df.loc[:, f"{col}_std"] = std_df[col]
                    mean_df.rename(columns={col: f"{col}_mean"}, inplace=True)
                stats_df_list.append(mean_df)
                df_to_mean_list.append(df_to_mean)

        df_to_mean = pd.concat(df_to_mean_list)
        df_to_mean.to_csv(os.path.join(save_path, f"vg_poststim_blink_{is_250_500}_per_vis_{name}_sub_stats.csv"), index=False)
        stat_df = pd.concat(stats_df_list)
        stat_df.to_csv(os.path.join(save_path, f"vg_poststim_blink_{is_250_500}_per_vis_{name}_descriptives.csv"), index=False)
    return


def poststim_blink_num_analysis(all_subs_df, save_path):
    relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == False)]
    # because if there are 0 saccades in pre stim we want to mean WITH it, fill na with zeros
    relevant_trials = relevant_trials.fillna(0)
    means_df = relevant_trials.groupby([SUBJECT_NAME, DataParser.VIS, "stimulusType", MODALITY]).mean().reset_index()
    means_df = means_df[[SUBJECT_NAME, DataParser.VIS, "stimulusType", "StimDurationNumBlinks", MODALITY]]
    df_list = []
    for vis in ["TruePositive", "FalseNegative"]:
        for stim_type in ["Face", "Object"]:
            vis_type_df = means_df[(means_df[DataParser.VIS] == vis) & (means_df["stimulusType"] == stim_type)]
            vis_str = "seen" if "True" in vis else "unseen"
            stim_str = stim_type.lower()
            vis_type_df = vis_type_df[[SUBJECT_NAME, "StimDurationNumBlinks", MODALITY]]
            vis_type_df.rename(
                columns={SUBJECT_NAME: "sub_code", "StimDurationNumBlinks": f"vg_numblinks_{vis_str}_{stim_str}"},
                inplace=True)
            df_list.append(vis_type_df)
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='inner'), df_list)
    partial_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='outer'), df_list)
    partial_set = set(partial_df["sub_code"])

    # format to R format
    df_list = list()

    for vis in ["TruePositive", "FalseNegative"]:
        for stim_type in ["Face", "Object"]:
            stim_str = stim_type.lower()
            vis_str = "seen" if "True" in vis else "unseen"
            relevant_col = f"vg_numblinks_{vis_str}_{stim_str}"
            temp_df = merged_df[["sub_code", relevant_col, MODALITY]]
            temp_df[VIS] = vis_str
            temp_df["category"] = stim_str
            temp_df.rename(columns={relevant_col: "vg_numblinks"}, inplace=True)
            df_list.append(temp_df)

    merged_df = pd.concat(df_list)
    merged_df.to_csv(os.path.join(save_path, f"vg_poststim_blink_per_vis.csv"), index=False)
    all_missing_subs = list(set(all_subs_df[SUBJECT_NAME]) - set(merged_df["sub_code"]))
    missing_reason = ["Partial Data" if x in partial_set else "No Data" for x in all_missing_subs]
    missing_subs = pd.DataFrame({"subCode": all_missing_subs, "reason": missing_reason})
    missing_subs.to_csv(os.path.join(save_path, f"vg_poststim_blink_per_vis_missing_subs.csv"), index=False)

    # Without a variable
    for name in ["all", "chosen10"]:
        all_dfs = []
        relevant_data = relevant_trials[
            (relevant_trials[DataParser.VIS] == "TruePositive") | (relevant_trials[DataParser.VIS] == "FalseNegative")]
        if name == "all":
            relevant_data = relevant_data[~relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = relevant_data[relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]

        relevant_data = relevant_data[
            ((relevant_data[DataParser.VIS] == "TruePositive") | (relevant_data[DataParser.VIS] == "FalseNegative")) &
            ((relevant_data["stimulusType"] == "Face") | (relevant_data["stimulusType"] == "Object"))]
        means_df = relevant_data.groupby([SUBJECT_NAME, MODALITY]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, MODALITY, "StimDurationNumBlinks"]]
        means_df2 = means_df.groupby([MODALITY]).mean().reset_index()
        means_df2.rename(columns={MODALITY: INDEPENDENT, "StimDurationNumBlinks": "vg_numblinks_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([MODALITY]).std().reset_index()
        std_df.rename(columns={MODALITY: INDEPENDENT, "StimDurationNumBlinks": "vg_numblinks_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, DataParser.VIS]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, DataParser.VIS, "StimDurationNumBlinks"]]
        means_df2 = means_df.groupby([DataParser.VIS]).mean().reset_index()
        means_df2.rename(columns={DataParser.VIS: INDEPENDENT, "StimDurationNumBlinks": "vg_numblinks_mean"},
                         inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([DataParser.VIS]).std().reset_index()
        std_df.rename(columns={DataParser.VIS: INDEPENDENT, "StimDurationNumBlinks": "vg_numblinks_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, "stimulusType"]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, "stimulusType", "StimDurationNumBlinks"]]
        means_df2 = means_df.groupby(["stimulusType"]).mean().reset_index()
        means_df2.rename(columns={"stimulusType": INDEPENDENT, "StimDurationNumBlinks": "vg_numblinks_mean"},
                         inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby(["stimulusType"]).std().reset_index()
        std_df.rename(columns={"stimulusType": INDEPENDENT, "StimDurationNumBlinks": "vg_numblinks_std"}, inplace=True)
        all_dfs.append(std_df)

        merged_df = pd.concat(all_dfs)
        merged_df.to_csv(os.path.join(save_path, f"vg_poststim_blink_per_vis_{name}_descriptives.csv"), index=False)
    return


def poststim_blink_num_analysis_replay_new(all_subs_df, save_path):
    relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == True)]
    relevant_trials.loc[:, TARGET] = relevant_trials[QualityChecker.STIM_TYPE] == relevant_trials["TargetType"]

    # because if there are 0 saccades in pre stim we want to mean WITH it, fill na with zeros
    relevant_trials = relevant_trials.fillna(0)
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(LEFT), "horizontal"] = LEFT
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(RIGHT), "horizontal"] = RIGHT
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(TOP), "vertical"] = TOP
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(BOTTOM), "vertical"] = BOTTOM

    relevant_trials = relevant_trials[((relevant_trials["stimulusType"] == "Face") | (relevant_trials["stimulusType"] == "Object"))]

    means_df = relevant_trials.groupby([SUBJECT_NAME, TARGET, "stimulusType", MODALITY, "horizontal", "vertical"]).mean().reset_index()
    means_df = means_df[[SUBJECT_NAME, TARGET, "stimulusType", "StimDurationNumBlinks", MODALITY, "horizontal", "vertical"]]

    means_df.rename(
        columns={SUBJECT_NAME: "sub_code",
                 "StimDurationNumBlinks": f"vg_numblinks", "stimulusType": "category"},
        inplace=True)
    means_df[TARGET].replace({True: "target", False: "non_target"}, inplace=True)
    means_df.to_csv(os.path.join(save_path, f"vg_poststim_blink_per_vis_replay.csv"), index=False)

    relevant_trials = relevant_trials[[SUBJECT_NAME, TARGET, "stimulusType", "StimDurationNumBlinks", MODALITY, "horizontal", "vertical"]]
    relevant_trials[TARGET].replace({True: "target", False: "non_target"}, inplace=True)
    # Without a variable
    for name in ["all", "chosen10"]:
        if name == "all":
            relevant_data = relevant_trials[~relevant_trials[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = relevant_trials[relevant_trials[SUBJECT_NAME].isin(CHOSEN_10)]

        list_to_combo_from = [MODALITY, "stimulusType", TARGET, "horizontal", "vertical"]
        combos = []
        for r in range(1, len(list_to_combo_from) + 1):
            combos.append(itertools.combinations(list_to_combo_from, r))

        stats_df_list = []
        df_to_mean_list = []
        for combo in combos:
            for x in combo:
                if "mod" not in list(x):
                    df_to_mean = relevant_data.groupby(list(x)+[SUBJECT_NAME, MODALITY]).mean().reset_index()
                else:
                    df_to_mean = relevant_data.groupby(list(x) + [SUBJECT_NAME]).mean().reset_index()
                mean_df = df_to_mean.groupby(list(x)).mean().reset_index()
                std_df = df_to_mean.groupby(list(x)).std().reset_index()
                for col in ["StimDurationNumBlinks"]:
                    mean_df.loc[:, f"{col}_std"] = std_df[col]
                    mean_df.rename(columns={col: f"{col}_mean"}, inplace=True)
                stats_df_list.append(mean_df)
                df_to_mean_list.append(df_to_mean)

        df_to_mean = pd.concat(df_to_mean_list)
        df_to_mean.to_csv(os.path.join(save_path, f"vg_poststim_blink_per_vis_{name}_sub_stats_replay.csv"), index=False)
        stat_df = pd.concat(stats_df_list)
        stat_df.to_csv(os.path.join(save_path, f"vg_poststim_blink_per_vis_{name}_descriptives_replay.csv"), index=False)
    return


def poststim_blink_num_analysis_replay(all_subs_df, save_path):
    relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == True)]
    relevant_trials.loc[:, TARGET] = relevant_trials[QualityChecker.STIM_TYPE] == relevant_trials["TargetType"]

    # because if there are 0 saccades in pre stim we want to mean WITH it, fill na with zeros
    relevant_trials = relevant_trials.fillna(0)
    means_df = relevant_trials.groupby([SUBJECT_NAME, TARGET, "stimulusType", MODALITY]).mean().reset_index()
    means_df = means_df[[SUBJECT_NAME, TARGET, "stimulusType", "StimDurationNumBlinks", MODALITY]]
    df_list = []
    for vis in [True, False]:
        for stim_type in ["Face", "Object"]:
            vis_type_df = means_df[(means_df[TARGET] == vis) & (means_df["stimulusType"] == stim_type)]
            vis_str = "target" if vis==True else "non_target"
            stim_str = stim_type.lower()
            vis_type_df = vis_type_df[[SUBJECT_NAME, "StimDurationNumBlinks", MODALITY]]
            vis_type_df.rename(
                columns={SUBJECT_NAME: "sub_code", "StimDurationNumBlinks": f"vg_numblinks_{vis_str}_{stim_str}"},
                inplace=True)
            df_list.append(vis_type_df)
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='inner'), df_list)
    partial_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='outer'), df_list)
    partial_set = set(partial_df["sub_code"])

    # format to R format
    df_list = list()

    for vis in [True, False]:
        for stim_type in ["Face", "Object"]:
            stim_str = stim_type.lower()
            vis_str = "target" if vis==True else "non_target"
            relevant_col = f"vg_numblinks_{vis_str}_{stim_str}"
            temp_df = merged_df[["sub_code", relevant_col, MODALITY]]
            temp_df[TARGET] = vis_str
            temp_df["category"] = stim_str
            temp_df.rename(columns={relevant_col: "vg_numblinks"}, inplace=True)
            df_list.append(temp_df)

    merged_df = pd.concat(df_list)
    merged_df.to_csv(os.path.join(save_path, f"vg_poststim_blink_per_vis_replay.csv"), index=False)
    all_missing_subs = list(set(all_subs_df[SUBJECT_NAME]) - set(merged_df["sub_code"]))
    missing_reason = ["Partial Data" if x in partial_set else "No Data" for x in all_missing_subs]
    missing_subs = pd.DataFrame({"subCode": all_missing_subs, "reason": missing_reason})
    missing_subs.to_csv(os.path.join(save_path, f"vg_poststim_blink_per_vis_missing_subs_replay.csv"), index=False)

    # Without a variable
    for name in ["all", "chosen10"]:
        all_dfs = []
        relevant_data = relevant_trials
        if name == "all":
            relevant_data = relevant_data[~relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = relevant_data[relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]

        relevant_data = relevant_data[((relevant_data["stimulusType"] == "Face") | (relevant_data["stimulusType"] == "Object"))]
        means_df = relevant_data.groupby([SUBJECT_NAME, MODALITY]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, MODALITY, "StimDurationNumBlinks"]]
        means_df2 = means_df.groupby([MODALITY]).mean().reset_index()
        means_df2.rename(columns={MODALITY: INDEPENDENT, "StimDurationNumBlinks": "vg_numblinks_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([MODALITY]).std().reset_index()
        std_df.rename(columns={MODALITY: INDEPENDENT, "StimDurationNumBlinks": "vg_numblinks_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, TARGET]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, TARGET, "StimDurationNumBlinks"]]
        means_df2 = means_df.groupby([TARGET]).mean().reset_index()
        means_df2.rename(columns={TARGET: INDEPENDENT, "StimDurationNumBlinks": "vg_numblinks_mean"},
                         inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([TARGET]).std().reset_index()
        std_df.rename(columns={TARGET: INDEPENDENT, "StimDurationNumBlinks": "vg_numblinks_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, "stimulusType"]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, "stimulusType", "StimDurationNumBlinks"]]
        means_df2 = means_df.groupby(["stimulusType"]).mean().reset_index()
        means_df2.rename(columns={"stimulusType": INDEPENDENT, "StimDurationNumBlinks": "vg_numblinks_mean"},
                         inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby(["stimulusType"]).std().reset_index()
        std_df.rename(columns={"stimulusType": INDEPENDENT, "StimDurationNumBlinks": "vg_numblinks_std"}, inplace=True)
        all_dfs.append(std_df)

        merged_df = pd.concat(all_dfs)
        merged_df.to_csv(os.path.join(save_path, f"vg_poststim_blink_per_vis_{name}_descriptives_replay.csv"), index=False)
    return


def poststim_first_sacc_analysis_new(all_subs_df, save_path, w_micro=True, is_250_500=False):
    if w_micro:
        w_micro_str = "withmicro"
        sacc_time = "FirstbothSaccTimePostOnset"
    else:
        w_micro_str = "womicro"
        sacc_time = "FirstSaccTimePostOnset"

    if is_250_500:
        w_micro_str += "_250_500"
        sacc_time += "250"

    relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == False)]

    # We want only fixations in the timewindow [0, 500], since we have the time postonset,
    # take only those with this value smaller than 500
    relevant_trials = relevant_trials[(relevant_trials[sacc_time] <= 500)]
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(LEFT), "horizontal"] = LEFT
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(RIGHT), "horizontal"] = RIGHT
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(TOP), "vertical"] = TOP
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(BOTTOM), "vertical"] = BOTTOM
    relevant_trials = relevant_trials[(relevant_trials[DataParser.VIS] == "TruePositive") | (relevant_trials[DataParser.VIS] == "FalseNegative")]

    means_df = relevant_trials.groupby([SUBJECT_NAME, DataParser.VIS, "stimulusType", MODALITY, "horizontal", "vertical"]).mean().reset_index()
    means_df = means_df[[SUBJECT_NAME, DataParser.VIS, "stimulusType", sacc_time, MODALITY, "horizontal", "vertical"]]
    means_df.rename(
        columns={SUBJECT_NAME: "sub_code", DataParser.VIS: VIS,
                 sacc_time: f"vg_sacconset", "stimulusType": "category"},
        inplace=True)
    means_df[VIS].replace({"TruePositive": "seen", "FalseNegative": "unseen"}, inplace=True)
    means_df.to_csv(os.path.join(save_path, f"vg_poststim_sacc_onset_per_vis_all_{w_micro_str}.csv"), index=False)

    relevant_trials = relevant_trials[[SUBJECT_NAME, DataParser.VIS, "stimulusType", sacc_time, MODALITY, "horizontal", "vertical"]]

    # Without a variable
    for name in ["nochosen10", "chosen10"]:
        relevant_data = relevant_trials[(relevant_trials[DataParser.VIS] == "TruePositive") | (relevant_trials[DataParser.VIS] == "FalseNegative")]
        if name == "nochosen10":
            relevant_data = relevant_data[~relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = relevant_data[relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]

        list_to_combo_from = [MODALITY, "stimulusType", DataParser.VIS, "horizontal", "vertical"]
        combos = []
        for r in range(1, len(list_to_combo_from) + 1):
            combos.append(itertools.combinations(list_to_combo_from, r))

        stats_df_list = []
        df_to_mean_list = []
        for combo in combos:
            for x in combo:
                if "mod" not in list(x):
                    df_to_mean = relevant_data.groupby(list(x)+[SUBJECT_NAME, MODALITY]).mean().reset_index()
                else:
                    df_to_mean = relevant_data.groupby(list(x) + [SUBJECT_NAME]).mean().reset_index()
                mean_df = df_to_mean.groupby(list(x)).mean().reset_index()
                std_df = df_to_mean.groupby(list(x)).std().reset_index()
                for col in [sacc_time]:
                    mean_df.loc[:, f"{col}_std"] = std_df[col]
                    mean_df.rename(columns={col: f"{col}_mean"}, inplace=True)
                stats_df_list.append(mean_df)
                df_to_mean_list.append(df_to_mean)

        df_to_mean = pd.concat(df_to_mean_list)
        df_to_mean.to_csv(os.path.join(save_path, f"vg_poststim_sacc_onset_per_vis_{name}_{w_micro_str}_sub_stats.csv"), index=False)
        stat_df = pd.concat(stats_df_list)
        stat_df.to_csv(os.path.join(save_path, f"vg_poststim_sacc_onset_per_vis_{name}_{w_micro_str}_descriptives.csv"), index=False)

    # NEW PLOT
    relevant_data = relevant_trials[(relevant_trials[DataParser.VIS] == "TruePositive") | (relevant_trials[DataParser.VIS] == "FalseNegative")]
    relevant_data = relevant_data[~relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]

    df_to_mean = relevant_data.groupby([SUBJECT_NAME, DataParser.VIS]).mean().reset_index()
    seen = df_to_mean[df_to_mean[DataParser.VIS]=="TruePositive"]
    unseen = df_to_mean[df_to_mean[DataParser.VIS]=="FalseNegative"]
    merged_df = pd.merge(seen, unseen, on=SUBJECT_NAME, suffixes=("_seen", "_unseen"))
    merged_df = merged_df.loc[:, [SUBJECT_NAME, f"{sacc_time}_seen", f"{sacc_time}_unseen"]]
    colors = ["#1F7A8C", "#90141E"]
    plotter.make_it_rain(data=merged_df, cols=[f"{sacc_time}_seen", f"{sacc_time}_unseen"], col_names=["seen", "unseen"],
                         color_list=colors, plot_title="Saccade Onset Per Visibility In Game", plot_x_label="Visibility", plot_y_label="Saccade Onset (ms)",
                         sub_line=[list(range(len(colors)))], save_plot=True,
                         save_path=save_path, save_folder="./", horizontal_lines=None,
                         save_name=f"vg_poststim_sacc_onset_per_vis_{w_micro_str}", custom_ylim=500, skip=50)

    return


def poststim_first_sacc_analysis(all_subs_df, save_path):
    relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == False)]

    # We want only fixations in the timewindow [0, 500], since we have the time postonset,
    # take only those with this value smaller than 500
    relevant_trials = relevant_trials[(relevant_trials["FirstSaccTimePostOnset"] <= 500)]
    means_df = relevant_trials.groupby([SUBJECT_NAME, DataParser.VIS, "stimulusType", MODALITY]).mean().reset_index()
    means_df = means_df[[SUBJECT_NAME, DataParser.VIS, "stimulusType", "FirstSaccTimePostOnset", MODALITY]]
    df_list = []
    for vis in ["TruePositive", "FalseNegative"]:
        for stim_type in ["Face", "Object"]:
            vis_type_df = means_df[(means_df[DataParser.VIS] == vis) & (means_df["stimulusType"] == stim_type)]
            vis_str = "seen" if "True" in vis else "unseen"
            stim_str = stim_type.lower()
            vis_type_df = vis_type_df[[SUBJECT_NAME, "FirstSaccTimePostOnset", MODALITY]]
            vis_type_df.rename(columns={SUBJECT_NAME: "sub_code", "FirstSaccTimePostOnset": f"vg_sacconset_{vis_str}_{stim_str}"}, inplace=True)
            df_list.append(vis_type_df)
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='inner'), df_list)
    partial_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='outer'), df_list)
    partial_set = set(partial_df["sub_code"])

    # format to R format
    df_list = list()

    for vis in ["TruePositive", "FalseNegative"]:
        for stim_type in ["Face", "Object"]:
            stim_str = stim_type.lower()
            vis_str = "seen" if "True" in vis else "unseen"
            relevant_col = f"vg_sacconset_{vis_str}_{stim_str}"
            temp_df = merged_df[["sub_code", relevant_col, MODALITY]]
            temp_df[VIS] = vis_str
            temp_df["category"] = stim_str
            temp_df.rename(columns={relevant_col: "vg_sacconset"}, inplace=True)
            df_list.append(temp_df)

    merged_df = pd.concat(df_list)
    merged_df.to_csv(os.path.join(save_path, f"vg_poststim_sacc_onset_per_vis.csv"), index=False)
    all_missing_subs = list(set(all_subs_df[SUBJECT_NAME]) - set(merged_df["sub_code"]))
    missing_reason = ["Partial Data" if x in partial_set else "No Data" for x in all_missing_subs]
    missing_subs = pd.DataFrame({"subCode": all_missing_subs, "reason": missing_reason})
    missing_subs.to_csv(os.path.join(save_path, f"vg_poststim_sacc_onset_per_vis_missing_subs.csv"), index=False)

    # Without a variable
    for name in ["all", "chosen10"]:
        all_dfs = []
        relevant_data = relevant_trials[(relevant_trials[DataParser.VIS] == "TruePositive") | (relevant_trials[DataParser.VIS] == "FalseNegative")]
        if name == "all":
            relevant_data = relevant_data[~relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = relevant_data[relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]

        relevant_data = relevant_data[
            ((relevant_data[DataParser.VIS] == "TruePositive") | (relevant_data[DataParser.VIS] == "FalseNegative")) &
            ((relevant_data["stimulusType"] == "Face") | (relevant_data["stimulusType"] == "Object"))]
        means_df = relevant_data.groupby([SUBJECT_NAME, MODALITY]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, MODALITY, "FirstSaccTimePostOnset"]]
        means_df2 = means_df.groupby([MODALITY]).mean().reset_index()
        means_df2.rename(columns={MODALITY: INDEPENDENT, "FirstSaccTimePostOnset": "vg_sacconset_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([MODALITY]).std().reset_index()
        std_df.rename(columns={MODALITY: INDEPENDENT, "FirstSaccTimePostOnset": "vg_sacconset_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, DataParser.VIS]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, DataParser.VIS, "FirstSaccTimePostOnset"]]
        means_df2 = means_df.groupby([DataParser.VIS]).mean().reset_index()
        means_df2.rename(columns={DataParser.VIS: INDEPENDENT, "FirstSaccTimePostOnset": "vg_sacconset_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([DataParser.VIS]).std().reset_index()
        std_df.rename(columns={DataParser.VIS: INDEPENDENT, "FirstSaccTimePostOnset": "vg_sacconset_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, "stimulusType"]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, "stimulusType", "FirstSaccTimePostOnset"]]
        means_df2 = means_df.groupby(["stimulusType"]).mean().reset_index()
        means_df2.rename(columns={"stimulusType": INDEPENDENT, "FirstSaccTimePostOnset": "vg_sacconset_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby(["stimulusType"]).std().reset_index()
        std_df.rename(columns={"stimulusType": INDEPENDENT, "FirstSaccTimePostOnset": "vg_sacconset_std"}, inplace=True)
        all_dfs.append(std_df)

        merged_df = pd.concat(all_dfs)
        merged_df.to_csv(os.path.join(save_path, f"vg_poststim_sacc_onset_per_vis_{name}_descriptives.csv"), index=False)
    return


def poststim_first_sacc_analysis_left_right(all_subs_df, save_path, left_right=True):
    relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == False)]

    if left_right:
        savename = "left_right"
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(RIGHT), "location"] = RIGHT
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(LEFT), "location"] = LEFT
        locations = [LEFT, RIGHT]
    else:
        savename = "top_bottom"
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(TOP), "location"] = TOP
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(BOTTOM), "location"] = BOTTOM
        locations = [TOP, BOTTOM]

    # We want only fixations in the timewindow [0, 500], since we have the time postonset,
    # take only those with this value smaller than 500
    relevant_trials = relevant_trials[(relevant_trials["FirstSaccTimePostOnset"] <= 500)]
    means_df = relevant_trials.groupby([SUBJECT_NAME, DataParser.VIS, "location", MODALITY]).mean().reset_index()
    means_df = means_df[[SUBJECT_NAME, DataParser.VIS, "location", "FirstSaccTimePostOnset", MODALITY]]
    df_list = []
    for vis in ["TruePositive", "FalseNegative"]:
        for stim_type in locations:
            vis_type_df = means_df[(means_df[DataParser.VIS] == vis) & (means_df["location"] == stim_type)]
            vis_str = "seen" if "True" in vis else "unseen"
            stim_str = stim_type.lower()
            vis_type_df = vis_type_df[[SUBJECT_NAME, "FirstSaccTimePostOnset", MODALITY]]
            vis_type_df.rename(columns={SUBJECT_NAME: "sub_code", "FirstSaccTimePostOnset": f"vg_sacconset_{vis_str}_{stim_str}"}, inplace=True)
            df_list.append(vis_type_df)
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='inner'), df_list)
    partial_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='outer'), df_list)
    partial_set = set(partial_df["sub_code"])

    # format to R format
    df_list = list()

    for vis in ["TruePositive", "FalseNegative"]:
        for stim_type in locations:
            stim_str = stim_type.lower()
            vis_str = "seen" if "True" in vis else "unseen"
            relevant_col = f"vg_sacconset_{vis_str}_{stim_str}"
            temp_df = merged_df[["sub_code", relevant_col, MODALITY]]
            temp_df[VIS] = vis_str
            temp_df["location"] = stim_str
            temp_df.rename(columns={relevant_col: "vg_sacconset"}, inplace=True)
            df_list.append(temp_df)

    merged_df = pd.concat(df_list)
    merged_df.to_csv(os.path.join(save_path, f"vg_poststim_sacc_onset_per_vis_{savename}.csv"), index=False)
    all_missing_subs = list(set(all_subs_df[SUBJECT_NAME]) - set(merged_df["sub_code"]))
    missing_reason = ["Partial Data" if x in partial_set else "No Data" for x in all_missing_subs]
    missing_subs = pd.DataFrame({"subCode": all_missing_subs, "reason": missing_reason})
    missing_subs.to_csv(os.path.join(save_path, f"vg_poststim_sacc_onset_per_vis_{savename}_missing_subs.csv"), index=False)

    # Without a variable
    for name in ["all", "chosen10"]:
        all_dfs = []
        relevant_data = relevant_trials[(relevant_trials[DataParser.VIS] == "TruePositive") | (relevant_trials[DataParser.VIS] == "FalseNegative")]
        if name == "all":
            relevant_data = relevant_data[~relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = relevant_data[relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]

        means_df = relevant_data.groupby([SUBJECT_NAME, MODALITY]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, MODALITY, "FirstSaccTimePostOnset"]]
        means_df2 = means_df.groupby([MODALITY]).mean().reset_index()
        means_df2.rename(columns={MODALITY: INDEPENDENT, "FirstSaccTimePostOnset": "vg_sacconset_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([MODALITY]).std().reset_index()
        std_df.rename(columns={MODALITY: INDEPENDENT, "FirstSaccTimePostOnset": "vg_sacconset_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, DataParser.VIS]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, DataParser.VIS, "FirstSaccTimePostOnset"]]
        means_df2 = means_df.groupby([DataParser.VIS]).mean().reset_index()
        means_df2.rename(columns={DataParser.VIS: INDEPENDENT, "FirstSaccTimePostOnset": "vg_sacconset_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([DataParser.VIS]).std().reset_index()
        std_df.rename(columns={DataParser.VIS: INDEPENDENT, "FirstSaccTimePostOnset": "vg_sacconset_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, "location"]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, "location", "FirstSaccTimePostOnset"]]
        means_df2 = means_df.groupby(["location"]).mean().reset_index()
        means_df2.rename(columns={"location": INDEPENDENT, "FirstSaccTimePostOnset": "vg_sacconset_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby(["location"]).std().reset_index()
        std_df.rename(columns={"location": INDEPENDENT, "FirstSaccTimePostOnset": "vg_sacconset_std"}, inplace=True)
        all_dfs.append(std_df)

        merged_df = pd.concat(all_dfs)
        merged_df.to_csv(os.path.join(save_path, f"vg_poststim_sacc_onset_per_vis_{savename}_{name}_descriptives.csv"), index=False)
    return


def poststim_first_sacc_analysis_left_right_replay(all_subs_df, save_path, left_right=True):
    relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == True)]

    if left_right:
        savename = "left_right"
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(RIGHT), "location"] = RIGHT
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(LEFT), "location"] = LEFT
        locations = [LEFT, RIGHT]
    else:
        savename = "top_bottom"
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(TOP), "location"] = TOP
        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(BOTTOM), "location"] = BOTTOM
        locations = [TOP, BOTTOM]

    relevant_trials.loc[:, TARGET] = relevant_trials[QualityChecker.STIM_TYPE] == relevant_trials["TargetType"]

    # We want only fixations in the timewindow [0, 500], since we have the time postonset,
    # take only those with this value smaller than 500
    relevant_trials = relevant_trials[(relevant_trials["FirstSaccTimePostOnset"] <= 500)]
    means_df = relevant_trials.groupby([SUBJECT_NAME, TARGET, "location", MODALITY]).mean().reset_index()
    means_df = means_df[[SUBJECT_NAME, TARGET, "location", "FirstSaccTimePostOnset", MODALITY]]
    df_list = []
    for vis in [True, False]:
        for stim_type in locations:
            vis_type_df = means_df[(means_df[TARGET] == vis) & (means_df["location"] == stim_type)]
            vis_str = "target" if vis==True else "non_target"
            stim_str = stim_type.lower()
            vis_type_df = vis_type_df[[SUBJECT_NAME, "FirstSaccTimePostOnset", MODALITY]]
            vis_type_df.rename(columns={SUBJECT_NAME: "sub_code", "FirstSaccTimePostOnset": f"vg_sacconset_{vis_str}_{stim_str}"}, inplace=True)
            df_list.append(vis_type_df)
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='inner'), df_list)
    partial_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='outer'), df_list)
    partial_set = set(partial_df["sub_code"])

    # format to R format
    df_list = list()

    for vis in [True, False]:
        for stim_type in locations:
            stim_str = stim_type.lower()
            vis_str = "target" if vis==True else "non_target"
            relevant_col = f"vg_sacconset_{vis_str}_{stim_str}"
            temp_df = merged_df[["sub_code", relevant_col, MODALITY]]
            temp_df[TARGET] = vis_str
            temp_df["location"] = stim_str
            temp_df.rename(columns={relevant_col: "vg_sacconset"}, inplace=True)
            df_list.append(temp_df)

    merged_df = pd.concat(df_list)
    merged_df.to_csv(os.path.join(save_path, f"vg_poststim_sacc_onset_per_vis_{savename}_replay.csv"), index=False)
    all_missing_subs = list(set(all_subs_df[SUBJECT_NAME]) - set(merged_df["sub_code"]))
    missing_reason = ["Partial Data" if x in partial_set else "No Data" for x in all_missing_subs]
    missing_subs = pd.DataFrame({"subCode": all_missing_subs, "reason": missing_reason})
    missing_subs.to_csv(os.path.join(save_path, f"vg_poststim_sacc_onset_per_vis_{savename}_missing_subs_replay.csv"), index=False)

    # Without a variable
    for name in ["all", "chosen10"]:
        all_dfs = []
        relevant_data = relevant_trials
        if name == "all":
            relevant_data = relevant_data[~relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = relevant_data[relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]

        # TODO: - ask about replay stimulusType filter in replace leftright
        relevant_data = relevant_data[((relevant_data["stimulusType"] == "Face") | (relevant_data["stimulusType"] == "Object"))]
        means_df = relevant_data.groupby([SUBJECT_NAME, MODALITY]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, MODALITY, "FirstSaccTimePostOnset"]]
        means_df2 = means_df.groupby([MODALITY]).mean().reset_index()
        means_df2.rename(columns={MODALITY: INDEPENDENT, "FirstSaccTimePostOnset": "vg_sacconset_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([MODALITY]).std().reset_index()
        std_df.rename(columns={MODALITY: INDEPENDENT, "FirstSaccTimePostOnset": "vg_sacconset_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, TARGET]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, TARGET, "FirstSaccTimePostOnset"]]
        means_df2 = means_df.groupby([TARGET]).mean().reset_index()
        means_df2.rename(columns={TARGET: INDEPENDENT, "FirstSaccTimePostOnset": "vg_sacconset_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([TARGET]).std().reset_index()
        std_df.rename(columns={TARGET: INDEPENDENT, "FirstSaccTimePostOnset": "vg_sacconset_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, "location"]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, "location", "FirstSaccTimePostOnset"]]
        means_df2 = means_df.groupby(["location"]).mean().reset_index()
        means_df2.rename(columns={"location": INDEPENDENT, "FirstSaccTimePostOnset": "vg_sacconset_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby(["location"]).std().reset_index()
        std_df.rename(columns={"location": INDEPENDENT, "FirstSaccTimePostOnset": "vg_sacconset_std"}, inplace=True)
        all_dfs.append(std_df)

        merged_df = pd.concat(all_dfs)
        merged_df.to_csv(os.path.join(save_path, f"vg_poststim_sacc_onset_per_vis_{savename}_{name}_descriptives_replay.csv"), index=False)
    return


def poststim_first_sacc_analysis_replay_new(all_subs_df, save_path):
    relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == True)]
    relevant_trials.loc[:, TARGET] = relevant_trials[QualityChecker.STIM_TYPE] == relevant_trials["TargetType"]

    # We want only fixations in the timewindow [0, 500], since we have the time postonset,
    # take only those with this value smaller than 500
    relevant_trials = relevant_trials[(relevant_trials["FirstSaccTimePostOnset"] <= 500)]
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(LEFT), "horizontal"] = LEFT
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(RIGHT), "horizontal"] = RIGHT
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(TOP), "vertical"] = TOP
    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(BOTTOM), "vertical"] = BOTTOM

    relevant_trials = relevant_trials[((relevant_trials["stimulusType"] == "Face") | (relevant_trials["stimulusType"] == "Object"))]

    means_df = relevant_trials.groupby([SUBJECT_NAME, TARGET, "stimulusType", MODALITY, "horizontal", "vertical"]).mean().reset_index()
    means_df = means_df[[SUBJECT_NAME, TARGET, "stimulusType", "FirstSaccTimePostOnset", MODALITY, "horizontal", "vertical"]]

    means_df.rename(
        columns={SUBJECT_NAME: "sub_code",
                 "FirstSaccTimePostOnset": f"vg_sacconset", "stimulusType": "category"},
        inplace=True)
    means_df[TARGET].replace({True: "target", False: "non_target"}, inplace=True)
    means_df.to_csv(os.path.join(save_path, f"replay_poststim_sacc_onset_per_vis.csv"), index=False)

    relevant_trials = relevant_trials[[SUBJECT_NAME, TARGET, "stimulusType", "FirstSaccTimePostOnset", MODALITY, "horizontal", "vertical"]]
    relevant_trials[TARGET].replace({True: "target", False: "non_target"}, inplace=True)
    # Without a variable
    for name in ["all", "chosen10"]:
        if name == "all":
            relevant_data = relevant_trials[~relevant_trials[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = relevant_trials[relevant_trials[SUBJECT_NAME].isin(CHOSEN_10)]

        list_to_combo_from = [MODALITY, "stimulusType", TARGET, "horizontal", "vertical"]
        combos = []
        for r in range(1, len(list_to_combo_from) + 1):
            combos.append(itertools.combinations(list_to_combo_from, r))

        stats_df_list = []
        df_to_mean_list = []
        for combo in combos:
            for x in combo:
                if "mod" not in list(x):
                    df_to_mean = relevant_data.groupby(list(x)+[SUBJECT_NAME, MODALITY]).mean().reset_index()
                else:
                    df_to_mean = relevant_data.groupby(list(x) + [SUBJECT_NAME]).mean().reset_index()
                mean_df = df_to_mean.groupby(list(x)).mean().reset_index()
                std_df = df_to_mean.groupby(list(x)).std().reset_index()
                for col in ["FirstSaccTimePostOnset"]:
                    mean_df.loc[:, f"{col}_std"] = std_df[col]
                    mean_df.rename(columns={col: f"{col}_mean"}, inplace=True)
                stats_df_list.append(mean_df)
                df_to_mean_list.append(df_to_mean)

        df_to_mean = pd.concat(df_to_mean_list)
        df_to_mean.to_csv(os.path.join(save_path, f"replay_poststim_sacc_onset_per_vis_{name}_sub_stats_replay.csv"), index=False)
        stat_df = pd.concat(stats_df_list)
        stat_df.to_csv(os.path.join(save_path, f"replay_poststim_sacc_onset_per_vis_{name}_descriptives_replay.csv"), index=False)

    # NEW PLOT
    relevant_data = relevant_trials[~relevant_trials[SUBJECT_NAME].isin(CHOSEN_10)]
    df_to_mean = relevant_data.groupby([SUBJECT_NAME, "vertical"]).mean().reset_index()

    target = df_to_mean[df_to_mean["vertical"] == TOP]
    non_target = df_to_mean[df_to_mean["vertical"] == BOTTOM]
    merged_df = pd.merge(target, non_target, on=SUBJECT_NAME, suffixes=("_seen", "_unseen"))
    merged_df = merged_df.loc[:, [SUBJECT_NAME, "FirstSaccTimePostOnset_seen", "FirstSaccTimePostOnset_unseen"]]
    colors = ["#1F7A8C", "#90141E"]
    plotter.make_it_rain(data=merged_df, cols=["FirstSaccTimePostOnset_seen", "FirstSaccTimePostOnset_unseen"],
                         col_names=[TOP, BOTTOM],
                         color_list=colors, plot_title="First Saccade Onset Per Vertical Location In Replay",
                         plot_x_label="Vertical Location", plot_y_label="Onset time (ms)",
                         sub_line=[list(range(len(colors)))], save_plot=True,
                         save_path=save_path, save_folder="./", horizontal_lines=None,
                         save_name=f"replay_poststim_sacc_onset_per_vertical", custom_ylim=500, skip=100)

    return


def poststim_first_sacc_analysis_replay(all_subs_df, save_path):
    relevant_trials = all_subs_df[(all_subs_df[DataParser.REPLAY] == True)]
    relevant_trials.loc[:, TARGET] = relevant_trials[QualityChecker.STIM_TYPE] == relevant_trials["TargetType"]

    # We want only fixations in the timewindow [0, 500], since we have the time postonset,
    # take only those with this value smaller than 500
    relevant_trials = relevant_trials[(relevant_trials["FirstSaccTimePostOnset"] <= 500)]
    means_df = relevant_trials.groupby([SUBJECT_NAME, TARGET, "stimulusType", MODALITY]).mean().reset_index()
    means_df = means_df[[SUBJECT_NAME, TARGET, "stimulusType", "FirstSaccTimePostOnset", MODALITY]]
    df_list = []
    for vis in [True, False]:
        for stim_type in ["Face", "Object"]:
            vis_type_df = means_df[(means_df[TARGET] == vis) & (means_df["stimulusType"] == stim_type)]
            vis_str = "target" if vis==True else "non_target"
            stim_str = stim_type.lower()
            vis_type_df = vis_type_df[[SUBJECT_NAME, "FirstSaccTimePostOnset", MODALITY]]
            vis_type_df.rename(columns={SUBJECT_NAME: "sub_code", "FirstSaccTimePostOnset": f"vg_sacconset_{vis_str}_{stim_str}"}, inplace=True)
            df_list.append(vis_type_df)
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='inner'), df_list)
    partial_df = reduce(lambda left, right: pd.merge(left, right, on=['sub_code', MODALITY], how='outer'), df_list)
    partial_set = set(partial_df["sub_code"])

    # format to R format
    df_list = list()

    for vis in [True, False]:
        for stim_type in ["Face", "Object"]:
            stim_str = stim_type.lower()
            vis_str = "target" if vis==True else "non_target"
            relevant_col = f"vg_sacconset_{vis_str}_{stim_str}"
            temp_df = merged_df[["sub_code", relevant_col, MODALITY]]
            temp_df[TARGET] = vis_str
            temp_df["category"] = stim_str
            temp_df.rename(columns={relevant_col: "vg_sacconset"}, inplace=True)
            df_list.append(temp_df)

    merged_df = pd.concat(df_list)
    merged_df.to_csv(os.path.join(save_path, f"vg_poststim_sacc_onset_per_vis_replay.csv"), index=False)
    all_missing_subs = list(set(all_subs_df[SUBJECT_NAME]) - set(merged_df["sub_code"]))
    missing_reason = ["Partial Data" if x in partial_set else "No Data" for x in all_missing_subs]
    missing_subs = pd.DataFrame({"subCode": all_missing_subs, "reason": missing_reason})
    missing_subs.to_csv(os.path.join(save_path, f"vg_poststim_sacc_onset_per_vis_missing_subs_replay.csv"), index=False)

    # Without a variable
    for name in ["all", "chosen10"]:
        all_dfs = []
        relevant_data = relevant_trials
        if name == "all":
            relevant_data = relevant_data[~relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]
        else:
            relevant_data = relevant_data[relevant_data[SUBJECT_NAME].isin(CHOSEN_10)]

        relevant_data = relevant_data[((relevant_data["stimulusType"] == "Face") | (relevant_data["stimulusType"] == "Object"))]
        means_df = relevant_data.groupby([SUBJECT_NAME, MODALITY]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, MODALITY, "FirstSaccTimePostOnset"]]
        means_df2 = means_df.groupby([MODALITY]).mean().reset_index()
        means_df2.rename(columns={MODALITY: INDEPENDENT, "FirstSaccTimePostOnset": "vg_sacconset_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([MODALITY]).std().reset_index()
        std_df.rename(columns={MODALITY: INDEPENDENT, "FirstSaccTimePostOnset": "vg_sacconset_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, TARGET]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, TARGET, "FirstSaccTimePostOnset"]]
        means_df2 = means_df.groupby([TARGET]).mean().reset_index()
        means_df2.rename(columns={TARGET: INDEPENDENT, "FirstSaccTimePostOnset": "vg_sacconset_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby([TARGET]).std().reset_index()
        std_df.rename(columns={TARGET: INDEPENDENT, "FirstSaccTimePostOnset": "vg_sacconset_std"}, inplace=True)
        all_dfs.append(std_df)

        means_df = relevant_data.groupby([SUBJECT_NAME, "stimulusType"]).mean().reset_index()
        means_df = means_df[[SUBJECT_NAME, "stimulusType", "FirstSaccTimePostOnset"]]
        means_df2 = means_df.groupby(["stimulusType"]).mean().reset_index()
        means_df2.rename(columns={"stimulusType": INDEPENDENT, "FirstSaccTimePostOnset": "vg_sacconset_mean"}, inplace=True)
        all_dfs.append(means_df2)

        std_df = means_df.groupby(["stimulusType"]).std().reset_index()
        std_df.rename(columns={"stimulusType": INDEPENDENT, "FirstSaccTimePostOnset": "vg_sacconset_std"}, inplace=True)
        all_dfs.append(std_df)

        merged_df = pd.concat(all_dfs)
        merged_df.to_csv(os.path.join(save_path, f"vg_poststim_sacc_onset_per_vis_{name}_descriptives_replay.csv"), index=False)
    return


def calculate_sample_time_in_trial(relevant_trials, trial_samps, additional_cols):
    for index, trial in relevant_trials.iterrows():  # for each sample, calculate its time relative to the epoch (trial) start
        # The EPOCH starts *before* stimulus onset https://osf.io/gm3vd
        trial_samps.loc[trial_samps[DataParser.TRIAL] == trial[TRIAL_NUMBER], TIME_IN_EPOCH] = trial_samps[DataParser.T_SAMPLE] - trial["EpochWindowStart"] - ET_param_manager.EPOCH_START
        for col in additional_cols:
            trial_samps.loc[trial_samps[DataParser.TRIAL] == trial[TRIAL_NUMBER], col] = trial[col]

    """
    That subject (who was found to be a bit erroneous already at the prepro stage, but fixed - see comments there) 
    for some reason has excessive samples (i.e., more samples than defined in an epoch - exceeds the epoch end time 
    by 4 samples). 
    In order to ensure that time-based plots and stats begin at epoch start and end at epoch end, we trim:
    """
    trial_samps = trial_samps[trial_samps[TIME_IN_EPOCH] <= ET_param_manager.EPOCH_END]
    return trial_samps


def calculate_within_ci(df, groupby_cols, hue_col, y_col):
    """
    Based of https://github.com/Cogitate-consortium/plotting_uniformization/blob/meg_oscar/MEG_activation/plot_spectral_activation.py,
    mean_ci_group function
    """
    for hue_value in df[hue_col].unique():
        df_hue = df.loc[df[hue_col] == hue_value, :]
        means_without_subs = df_hue.groupby(groupby_cols).mean().reset_index()
        means_with_subs = df_hue.groupby(groupby_cols + [SUBJECT]).mean().reset_index()
        for sub in df_hue[SUBJECT].unique():
            sub_y_col_mean = means_with_subs.loc[means_with_subs[SUBJECT] == sub, y_col]
            sbj_mean = sub_y_col_mean.mean()
            group_mean = means_without_subs[y_col].mean()
            within_y_col = sub_y_col_mean - sbj_mean + group_mean
            df_hue.loc[df_hue[SUBJECT] == sub, "y_col_tag"] = list(within_y_col)
        df.loc[df[hue_col] == hue_value, "y_col_tag"] = list(df_hue["y_col_tag"])

    df_ci = df.groupby(groupby_cols)["y_col_tag"].sem() * 1.96

    return df_ci


def line_plotter_ci(df, x_col, x_col_name, y_col, y_col_name, hue_col, hue_col_name, ci_col, title, save_path, save_name,
                    hue_names_map=None, palette=None, y_max=None, y_min=None, black_vertical_x=None, gray_vertical_x=None, y_ticks=None,
                    black_horizontal_y=None, x_ticks=None):
    plt.gcf()
    plt.figure()
    sns.reset_orig()

    hue_groups = df[hue_col].unique()
    for group in hue_groups:
        df_group = df[df[hue_col] == group]
        color = palette[group]
        plt.plot(df_group[x_col], df_group[y_col], color=color)
        plt.fill_between(df_group[x_col], df_group[y_col]-df_group[ci_col], df_group[y_col]+df_group[ci_col],
                         alpha=0.2, edgecolor=color, facecolor=color)
    # add vertical lines
    if not(black_vertical_x is None):
        if y_max is not None and y_min is not None and not np.isnan(y_max) and not np.isnan(y_min):
            mi = y_min
            ma = y_max
        else:
            mi = df_group[y_col].min()
            ma = df_group[y_col].max()
        plt.vlines(x=black_vertical_x, ymin=mi, ymax=ma, colors='black', ls='--', lw=1)
    if not(gray_vertical_x is None):
        if y_max is not None and y_min is not None and not np.isnan(y_max) and not np.isnan(y_min):
            mi = y_min
            ma = y_max
        else:
            mi = df_group[y_col].min()
            ma = df_group[y_col].max()
        plt.vlines(x=gray_vertical_x, ymin=mi, ymax=ma, colors='gray', ls='--', lw=1)

    # Add hozintal lines
    if not(black_horizontal_y is None):
        mi = df_group[x_col].min()
        ma = df_group[x_col].max()
        plt.hlines(y=black_horizontal_y, xmin=mi, xmax=ma, colors='black', lw=1)

    plt.title(f"{title}", fontsize=TITLE_SIZE, pad=LABEL_PAD + 5)
    if x_ticks is not None:
        plt.xticks(x_ticks, fontsize=TICK_SIZE)
    else:
        plt.xticks(fontsize=TICK_SIZE)

    if y_max is not None and y_min is not None and not np.isnan(y_max) and not np.isnan(y_min):
        #plt.ylim(0.0,  0.5)
        plt.ylim(y_min, y_max)

    if y_ticks is not None:
        plt.yticks(y_ticks, fontsize=TICK_SIZE)
    else:
        plt.yticks(y_ticks, fontsize=TICK_SIZE)

    plt.locator_params(axis='y', nbins=6)
    plt.locator_params(axis='x', nbins=18)
    plt.xlabel(x_col_name, fontsize=AXIS_SIZE, labelpad=LABEL_PAD)
    plt.ylabel(y_col_name, fontsize=AXIS_SIZE, labelpad=LABEL_PAD)

    markers = [plt.Line2D([0, 0], [0, 0], color=palette[label], marker='o', linestyle='') for label in palette]
    new_labels = [hue_names_map[label] for label in palette]
    legend = plt.legend(markers, new_labels, title=hue_col_name, markerscale=1, fontsize=TICK_SIZE - 2)
    plt.setp(legend.get_title(), fontsize=TICK_SIZE - 2)

    # save
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 12)
    plt.savefig(os.path.join(save_path, f"{save_name}.png"), dpi=1000)
    plt.savefig(os.path.join(save_path, f"{save_name}.svg"), format="svg", dpi=1000)

    # now free some memory
    del figure
    plt.close()
    gc.collect()
    return


def line_plot_per_mod(data, x_col, x_col_name, y_col, y_col_name, hue_col, hue_col_name, title_name,
                      save_path, save_name, hue_names_map=None, palette=None, global_y_min=None, epsilon=None,
                      global_y_max=True, black_vertical_x=None, gray_vertical_x=None, y_lim_dict=None, y_ticks=None, same_y_across_mods=False,
                      x_ticks=None, black_horizontal_y=None):
    modality_list = list(data[MODALITY].unique())

    if same_y_across_mods:
        reps = 2
    else:
        reps = 1

    y_max = None
    y_min = None
    orig_data = data.copy().reset_index()

    for _ in range(reps):
        for mod in modality_list:
            data = orig_data
            df = data[data[MODALITY] == mod]
            df_ci = calculate_within_ci(df, [x_col, hue_col], hue_col, y_col)
            df_mean = df.groupby([x_col, hue_col]).mean().reset_index()
            df_mean[f"{y_col}_CI_SE"] = list(df_ci)

            if y_lim_dict is None:
                if y_max is None:
                    y_max = max(df_mean[y_col] + df_mean[f"{y_col}_CI_SE"])
                    y_min = min(df_mean[y_col] - df_mean[f"{y_col}_CI_SE"])
                    y_ticks_tag = None
                else:
                    y_max2 = max(df_mean[y_col] + df_mean[f"{y_col}_CI_SE"])
                    y_min2 = min(df_mean[y_col] - df_mean[f"{y_col}_CI_SE"])
                    y_max = max(y_max2, y_max)
                    y_min = min(y_min2, y_min)
                    y_ticks_tag = None
            else:
                y_min, y_max = y_lim_dict[mod]
                y_ticks_tag = y_ticks[mod]

            line_plotter_ci(df=df_mean, x_col=x_col, x_col_name=x_col_name, y_col=y_col, y_col_name=y_col_name,
                            hue_col=hue_col, hue_col_name=hue_col_name, ci_col=f"{y_col}_CI_SE",
                            title=f"{title_name} {mod}", save_path=save_path, save_name=f"{save_name}_{mod}",
                            hue_names_map=hue_names_map, palette=palette, y_max=y_max, y_min=y_min,
                            black_vertical_x=black_vertical_x, gray_vertical_x=gray_vertical_x, y_ticks=y_ticks_tag,
                            x_ticks=x_ticks, black_horizontal_y=black_horizontal_y)

            df.to_csv(os.path.join(save_path, f"{save_name}_{mod}.csv"), index=False)


def blink_plot_trial_vis(subs_dict, save_path, load=False, window_width=100):
    """
    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    WE TAKE ONLY GAME TRIALS!!!
    """
    if not load:
        everything_df_list = []
        for vis in ["FalseNegative", "TruePositive"]:
            all_dfs_list = []
            for modality in subs_dict.keys():
                only_mod_dfs_list = []
                for sub_data_path in subs_dict[modality]:
                    print(sub_data_path)
                    fl = open(sub_data_path, 'rb')
                    sub_data = pickle.load(fl)
                    fl.close()
                    analyzed_eye = sub_data[PARAMS][DataParser.EYE]
                    trial_samples = sub_data[ET_DATA_DICT][DataParser.DF_SAMPLES]
                    trial_samples[f'{analyzed_eye}{DataParser.HERSHMAN}'].fillna(0,
                                                                                 inplace=True)  # Replace non blinks with 0, so the mean will actually be PROPORTION
                    relevant_trials = sub_data[TRIAL_INFO]
                    relevant_trials = relevant_trials.loc[(relevant_trials[DataParser.IS_LEVEL_ELIMINATED] == False) & (relevant_trials[DataParser.VIS] == vis) &
                                                          (relevant_trials[DataParser.REPLAY] == False) & (relevant_trials[ET_data_extraction.IS_PROBED] == True), :]
                    if relevant_trials.empty:
                        continue

                    # For Z score, note that we calculate the zscore according to the GAME BASELINE, similar to GAME
                    trial_samples_zscore = trial_samples.loc[trial_samples[DataParser.TRIAL].isin(
                        list(relevant_trials[TRIAL_NUMBER])), :]
                    trial_samples_zscore = calculate_sample_time_in_trial(relevant_trials, trial_samples_zscore, [])

                    trial_samples_zscore = trial_samples_zscore.loc[trial_samples_zscore[DataParser.TRIAL] != -1, :]  # ones that are within a TRIAL epoch (not between trials)
                    # collapse across stimulus duration groups, at each timepoint (from epoch start to end) and COUNT SACCADES
                    trial_means = trial_samples_zscore.groupby([TIME_IN_EPOCH]).mean(numeric_only=True).reset_index()
                    sub_mean = trial_means.loc[(trial_means[TIME_IN_EPOCH] >= -1000) & (trial_means[TIME_IN_EPOCH] <= -500), f"{analyzed_eye}{DataParser.HERSHMAN}"].mean()
                    sub_std = trial_means.loc[(trial_means[TIME_IN_EPOCH] >= -1000) & (trial_means[TIME_IN_EPOCH] <= -500), f"{analyzed_eye}{DataParser.HERSHMAN}"].std()

                    sub_mean1000 = trial_means.loc[(trial_means[TIME_IN_EPOCH] >= -1000) & (
                                trial_means[TIME_IN_EPOCH] <= -950), f"{analyzed_eye}{DataParser.HERSHMAN}"].mean()
                    sub_std1000 = trial_means.loc[(trial_means[TIME_IN_EPOCH] >= -1000) & (
                                trial_means[TIME_IN_EPOCH] <= -950), f"{analyzed_eye}{DataParser.HERSHMAN}"].std()

                    sub_mean950 = trial_means.loc[(trial_means[TIME_IN_EPOCH] >= -950) & (
                                trial_means[TIME_IN_EPOCH] <= -500), f"{analyzed_eye}{DataParser.HERSHMAN}"].mean()
                    sub_std950 = trial_means.loc[(trial_means[TIME_IN_EPOCH] >= -950) & (
                                trial_means[TIME_IN_EPOCH] <= -500), f"{analyzed_eye}{DataParser.HERSHMAN}"].std()
                    # End of zscore floofy

                    trial_samples = trial_samples.loc[trial_samples[DataParser.TRIAL].isin(list(relevant_trials[TRIAL_NUMBER])), :]
                    trial_samples = calculate_sample_time_in_trial(relevant_trials, trial_samples, [])

                    trial_samples = trial_samples.loc[trial_samples[DataParser.TRIAL] != -1, :]  # ones that are within a TRIAL epoch (not between trials)
                    # collapse across stimulus duration groups, at each timepoint (from epoch start to end) and COUNT BLINKS
                    trial_means = trial_samples.groupby([TIME_IN_EPOCH]).mean(numeric_only=True).reset_index()
                    trial_means = trial_means.loc[:, [TIME_IN_EPOCH, f"{analyzed_eye}{DataParser.HERSHMAN}"]]
                    trial_means.loc[:, SUBJECT] = sub_data[PARAMS]['SubjectName']
                    trial_means.loc[:, DataParser.VIS] = vis
                    trial_means.loc[:, MODALITY] = modality
                    trial_means.rename(columns={f"{analyzed_eye}{DataParser.HERSHMAN}": f"{DataParser.HERSHMAN}"}, inplace=True)
                    if sub_mean == 0 and sub_std == 0:
                        trial_means.loc[:, f"{DataParser.HERSHMAN}_zscore"] = 0
                    else:
                        trial_means.loc[:, f"{DataParser.HERSHMAN}_zscore"] = (trial_means[DataParser.HERSHMAN] - sub_mean) / sub_std

                    only_mod_dfs_list.append(trial_means)
                # summarize this modality - average across all subjects' averages (and calculate std)
                mod_mean_df = pd.concat(only_mod_dfs_list)
                all_dfs_list.append(mod_mean_df)

            # Now, we do the all subs plot
            all_mean_df = pd.concat(all_dfs_list)
            everything_df_list.append(all_mean_df)

        # Now, we do the all subs plot
        all_mean_df = pd.concat(everything_df_list)
        all_mean_df.to_csv(os.path.join(save_path, f"blink_plot_vis_data.csv"), index=False)
    else:
        all_mean_df = pd.read_csv(os.path.join(save_path, f"blink_plot_vis_data.csv"))

    # as we have samples where only one subject had a blink, this skewes the plot. Therefore, create a version w/o it:
    # select EVERYONE BUT CHOSEN 10
    df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]

    count_samps = df_cat.groupby([TIME_IN_EPOCH, DataParser.VIS, MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                           (df_cat.Visibility.isin(count_samps[DataParser.VIS])) &
                           (df_cat["mod"].isin(count_samps["mod"])))]

    df_crazy = df_cat[(df_cat[f"{DataParser.HERSHMAN}_zscore"] > 1000000) | (df_cat[f"{DataParser.HERSHMAN}_zscore"] < -1000000)]
    df_crazy.to_csv(os.path.join(save_path, f"blink_plot_vis_data_crazy_zscores.csv"), index=False)


    # SMOOTHING - old
    """
    data = np.pad(list(df_cat[DataParser.HERSHMAN]), int(window_width / 2), mode='edge')
    cumsum_vec = np.cumsum(data)
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    df_cat[DataParser.HERSHMAN] = ma_vec

    data = np.pad(list(df_cat[f"{DataParser.HERSHMAN}_zscore"]), int(window_width / 2), mode='edge')
    cumsum_vec = np.cumsum(data)
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    df_cat[f"{DataParser.HERSHMAN}_zscore"] = ma_vec
    """
    df_cat = df_cat.sort_values(by=["sub", DataParser.VIS, "timeInEpoch"])
    df_cat[DataParser.HERSHMAN] = (
        df_cat.groupby(["sub", DataParser.VIS])[DataParser.HERSHMAN]
            .transform(lambda x: x.rolling(window=window_width, center=True, min_periods=1).mean())
    )
    df_cat[f"{DataParser.HERSHMAN}_zscore"] = (
        df_cat.groupby(["sub", DataParser.VIS])[f"{DataParser.HERSHMAN}_zscore"]
            .transform(lambda x: x.rolling(window=window_width, center=True, min_periods=1).mean())
    )

    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.HERSHMAN, y_col_name="Blink rate in dAT",
                      hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      title_name="Average blink rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"vg_blinkrate_vis_all",
                      hue_names_map=VISIBILITY_NAME_MAP, palette=VISIBILITY_MAP,
                      y_lim_dict={ET_qc_manager.FMRI: (0, 0.55), ET_qc_manager.MEG: (0, 0.55)},
                      y_ticks={ET_qc_manager.FMRI: [0.1, 0.2, 0.3, 0.4, 0.5],
                               ET_qc_manager.MEG: [0.1, 0.2, 0.3, 0.4, 0.5]})

    df_cat = df_cat.loc[(-1000 <= df_cat[TIME_IN_EPOCH]) & (df_cat[TIME_IN_EPOCH] <= 500)]
    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.HERSHMAN, y_col_name="Blink rate in dAT",
                      hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      title_name="Average blink rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"vg_blinkrate_vis_all_zoom",
                      hue_names_map=VISIBILITY_NAME_MAP, palette=VISIBILITY_MAP,
                      y_lim_dict={ET_qc_manager.FMRI: (0.02, 0.07), ET_qc_manager.MEG: (0.02, 0.07)},
                      y_ticks={ET_qc_manager.FMRI: [0.03, 0.04, 0.05, 0.06],
                               ET_qc_manager.MEG: [0.03, 0.04, 0.05, 0.06]},
                      black_vertical_x=0, gray_vertical_x=250)

    df_cat = df_cat.loc[(-1000 <= df_cat[TIME_IN_EPOCH]) & (df_cat[TIME_IN_EPOCH] <= 500)]
    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=f"{DataParser.HERSHMAN}_zscore", y_col_name="Blink rate in dAT (zscored)",
                      hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      title_name="Average blink rate in Trial (z-scored)", global_y_max=False,
                      save_path=save_path, save_name=f"vg_blinkrate_vis_all_zoom_zscore",
                      hue_names_map=VISIBILITY_NAME_MAP, palette=VISIBILITY_MAP,
                      black_vertical_x=0, gray_vertical_x=250,
                      y_lim_dict={ET_qc_manager.FMRI: (-1, 1.5), ET_qc_manager.MEG: (-1, 1.5)},
                      y_ticks={ET_qc_manager.FMRI: [-1, -0.5, 0, 0.5, 1],
                               ET_qc_manager.MEG: [-1, -0.5, 0, 0.5, 1]},
                      x_ticks=[-1000, -750, -500, -250, 0, 250, 500],
                      black_horizontal_y=0)

    df_cat = df_cat.loc[(-250 <= df_cat[TIME_IN_EPOCH]) & (df_cat[TIME_IN_EPOCH] <= 750)]
    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.HERSHMAN, y_col_name="Blink rate in dAT",
                      hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      title_name="Average blink rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"vg_blinkrate_vis_all_zoom_250",
                      hue_names_map=VISIBILITY_NAME_MAP, palette=VISIBILITY_MAP,
                      y_lim_dict={ET_qc_manager.FMRI: (0.02, 0.07), ET_qc_manager.MEG: (0.02, 0.07)},
                      y_ticks={ET_qc_manager.FMRI: [0.03, 0.04, 0.05, 0.06],
                               ET_qc_manager.MEG: [0.03, 0.04, 0.05, 0.06]},
                      black_vertical_x=0, gray_vertical_x=250)

    """
    # Chosen 10
    df_cat = all_mean_df[all_mean_df[SUBJECT].isin(CHOSEN_10)]
    count_samps = df_cat.groupby([TIME_IN_EPOCH, DataParser.VIS, MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    mod_df = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                           (df_cat.Visibility.isin(count_samps[DataParser.VIS])) &
                           (df_cat["mod"].isin(count_samps["mod"])))]
    line_plot_per_mod(data=mod_df, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.HERSHMAN, y_col_name="Blink rate in dAT",
                      hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      title_name="Average blink rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"vg_blinkrate_vis_chosen10",
                      hue_names_map=VISIBILITY_NAME_MAP, palette=VISIBILITY_MAP)

    df_cat[MODALITY] = "all"

    count_samps = df_cat.groupby([TIME_IN_EPOCH, DataParser.VIS, MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                           (df_cat.Visibility.isin(count_samps[DataParser.VIS])) &
                           (df_cat["mod"].isin(count_samps["mod"])))]
    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.HERSHMAN, y_col_name="Blink rate in dAT",
                      hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      title_name="Average blink rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"vg_blinkrate_vis_chosen10",
                      hue_names_map=VISIBILITY_NAME_MAP, palette=VISIBILITY_MAP)
    """
    return


def blink_plot_trial_relevance_replay(subs_dict, save_path, load=False, window_width=100):
    """
    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    WE TAKE ONLY GAME TRIALS!!!
    """
    if not load:
        everything_df_list = []
        for task_relevant in [True, False]:
            all_dfs_list = []
            for modality in subs_dict.keys():
                only_mod_dfs_list = []
                for sub_data_path in subs_dict[modality]:
                    print(sub_data_path)
                    fl = open(sub_data_path, 'rb')
                    sub_data = pickle.load(fl)
                    fl.close()
                    analyzed_eye = sub_data[PARAMS][DataParser.EYE]
                    trial_samples = sub_data[ET_DATA_DICT][DataParser.DF_SAMPLES]
                    trial_samples[f'{analyzed_eye}{DataParser.HERSHMAN}'].fillna(0,
                                                                                 inplace=True)  # Replace non blinks with 0, so the mean will actually be PROPORTION
                    relevant_trials = sub_data[TRIAL_INFO]
                    relevant_trials.loc[:, TARGET] = relevant_trials[QualityChecker.STIM_TYPE] == relevant_trials["TargetType"]
                    relevant_trials = relevant_trials.loc[(relevant_trials[DataParser.IS_LEVEL_ELIMINATED] == False) &
                                                          (relevant_trials[DataParser.REPLAY] == True) & (relevant_trials[ET_data_extraction.IS_PROBED] == True), :]

                    if task_relevant == True:
                        relevant_trials = relevant_trials.loc[~relevant_trials[QualityChecker.STIM_TYPE].str.contains("None")]
                    else:
                        relevant_trials = relevant_trials.loc[relevant_trials[QualityChecker.STIM_TYPE].str.contains("None")]

                    if relevant_trials.empty:
                        continue

                    # For Z score, note that we calculate the zscore according to the GAME BASELINE, similar to GAME
                    trial_samples_zscore = trial_samples.loc[trial_samples[DataParser.TRIAL].isin(
                        list(relevant_trials[TRIAL_NUMBER])), :]
                    trial_samples_zscore = calculate_sample_time_in_trial(relevant_trials, trial_samples_zscore, [])

                    trial_samples_zscore = trial_samples_zscore.loc[trial_samples_zscore[DataParser.TRIAL] != -1, :]  # ones that are within a TRIAL epoch (not between trials)
                    # collapse across stimulus duration groups, at each timepoint (from epoch start to end) and COUNT SACCADES
                    trial_means = trial_samples_zscore.groupby([TIME_IN_EPOCH]).mean(numeric_only=True).reset_index()
                    sub_mean = trial_means.loc[(trial_means[TIME_IN_EPOCH] >= -1000) & (trial_means[TIME_IN_EPOCH] <= -500), f"{analyzed_eye}{DataParser.HERSHMAN}"].mean()
                    sub_std = trial_means.loc[(trial_means[TIME_IN_EPOCH] >= -1000) & (trial_means[TIME_IN_EPOCH] <= -500), f"{analyzed_eye}{DataParser.HERSHMAN}"].std()
                    # End of zscore floofy

                    trial_samples = trial_samples.loc[trial_samples[DataParser.TRIAL].isin(list(relevant_trials[TRIAL_NUMBER])), :]
                    trial_samples = calculate_sample_time_in_trial(relevant_trials, trial_samples, [])

                    trial_samples = trial_samples.loc[trial_samples[DataParser.TRIAL] != -1, :]  # ones that are within a TRIAL epoch (not between trials)
                    # collapse across stimulus duration groups, at each timepoint (from epoch start to end) and COUNT BLINKS
                    trial_means = trial_samples.groupby([TIME_IN_EPOCH]).mean(numeric_only=True).reset_index()
                    trial_means = trial_means.loc[:, [TIME_IN_EPOCH, f"{analyzed_eye}{DataParser.HERSHMAN}"]]
                    trial_means.loc[:, SUBJECT] = sub_data[PARAMS]['SubjectName']
                    trial_means.loc[:, "task_relevant"] = task_relevant
                    trial_means.loc[:, MODALITY] = modality
                    trial_means.rename(columns={f"{analyzed_eye}{DataParser.HERSHMAN}": f"{DataParser.HERSHMAN}"}, inplace=True)
                    if sub_mean == 0 and sub_std == 0:
                        trial_means.loc[:, f"{DataParser.HERSHMAN}_zscore"] = 0
                    else:
                        trial_means.loc[:, f"{DataParser.HERSHMAN}_zscore"] = (trial_means[DataParser.HERSHMAN] - sub_mean) / sub_std

                    only_mod_dfs_list.append(trial_means)
                # summarize this modality - average across all subjects' averages (and calculate std)
                mod_mean_df = pd.concat(only_mod_dfs_list)
                all_dfs_list.append(mod_mean_df)

            # Now, we do the all subs plot
            all_mean_df = pd.concat(all_dfs_list)
            everything_df_list.append(all_mean_df)

        # Now, we do the all subs plot
        all_mean_df = pd.concat(everything_df_list)
        all_mean_df.to_csv(os.path.join(save_path, f"blink_plot_relevance_replay_data.csv"), index=False)
    else:
        all_mean_df = pd.read_csv(os.path.join(save_path, f"blink_plot_relevance_replay_data.csv"))

        # as we have samples where only one subject had a blink, this skewes the plot. Therefore, create a version w/o it:
        # select EVERYONE BUT CHOSEN 10
    df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]

    count_samps = df_cat.groupby([TIME_IN_EPOCH, TASK_RELEVANT, MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                           (df_cat.task_relevant.isin(count_samps[TASK_RELEVANT])) &
                           (df_cat["mod"].isin(count_samps["mod"])))]

    df_cat = df_cat.sort_values(by=["sub", TASK_RELEVANT, "timeInEpoch"])
    df_cat[DataParser.HERSHMAN] = (
        df_cat.groupby(["sub", TASK_RELEVANT])[DataParser.HERSHMAN]
            .transform(lambda x: x.rolling(window=window_width, center=True, min_periods=1).mean())
    )
    df_cat[f"{DataParser.HERSHMAN}_zscore"] = (
        df_cat.groupby(["sub", TASK_RELEVANT])[f"{DataParser.HERSHMAN}_zscore"]
            .transform(lambda x: x.rolling(window=window_width, center=True, min_periods=1).mean())
    )
    replay_df_cat = df_cat
    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.HERSHMAN, y_col_name="Blink rate in Replay",
                      hue_col=TASK_RELEVANT, hue_col_name=TASK_RELEVANT,
                      title_name="Average blink rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"replay_blinkrate_task_nochosen10",
                      hue_names_map=TASK_RELEVANT_NAME_MAP, palette=TASK_RELEVANT_MAP,
                      same_y_across_mods=True)

    df_cat = df_cat.loc[(-1000 <= df_cat[TIME_IN_EPOCH]) & (df_cat[TIME_IN_EPOCH] <= 500)]
    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.HERSHMAN, y_col_name="Blink rate in Replay",
                      hue_col=TASK_RELEVANT, hue_col_name=TASK_RELEVANT,
                      title_name="Average blink rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"replay_blinkrate_task_nochosen10_zoom",
                      hue_names_map=TASK_RELEVANT_NAME_MAP, palette=TASK_RELEVANT_MAP,
                      black_vertical_x=0, gray_vertical_x=250, same_y_across_mods=True)

    df_cat = df_cat.loc[(-1000 <= df_cat[TIME_IN_EPOCH]) & (df_cat[TIME_IN_EPOCH] <= 500)]
    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=f"{DataParser.HERSHMAN}_zscore", y_col_name="Blink rate in Replay (z-scored)",
                      hue_col=TASK_RELEVANT, hue_col_name=TASK_RELEVANT,
                      title_name="Average blink rate in Trial (z-scored)", global_y_max=False,
                      save_path=save_path, save_name=f"replay_blinkrate_task_nochosen10_zscore_zoom",
                      hue_names_map=TASK_RELEVANT_NAME_MAP, palette=TASK_RELEVANT_MAP,
                      black_vertical_x=0, gray_vertical_x=250, same_y_across_mods=True)

    df_cat = df_cat.loc[(-250 <= df_cat[TIME_IN_EPOCH]) & (df_cat[TIME_IN_EPOCH] <= 750)]
    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.HERSHMAN, y_col_name="Blink rate in Replay",
                      hue_col=TASK_RELEVANT, hue_col_name=TASK_RELEVANT,
                      title_name="Average blink rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"replay_blinkrate_task_nochosen10_zoom_250",
                      hue_names_map=TASK_RELEVANT_NAME_MAP, palette=TASK_RELEVANT_MAP,
                      black_vertical_x=0, gray_vertical_x=250, same_y_across_mods=True)

    game_df = pd.read_csv(os.path.join(save_path, f"blink_plot_vis_data.csv"))
    game_df[TASK_RELEVANT] = "Seen"
    game_df = game_df[game_df[DataParser.VIS] == "TruePositive"]
    all_mean_df = game_df
    df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]

    count_samps = df_cat.groupby([TIME_IN_EPOCH, TASK_RELEVANT, MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                           (df_cat.task_relevant.isin(count_samps[TASK_RELEVANT])) &
                           (df_cat["mod"].isin(count_samps["mod"])))]

    df_cat = df_cat.sort_values(by=["sub", TASK_RELEVANT, "timeInEpoch"])
    df_cat[DataParser.HERSHMAN] = (
        df_cat.groupby(["sub", TASK_RELEVANT])[DataParser.HERSHMAN]
            .transform(lambda x: x.rolling(window=window_width, center=True, min_periods=1).mean())
    )
    df_cat[f"{DataParser.HERSHMAN}_zscore"] = (
        df_cat.groupby(["sub", TASK_RELEVANT])[f"{DataParser.HERSHMAN}_zscore"]
            .transform(lambda x: x.rolling(window=window_width, center=True, min_periods=1).mean())
    )
    df_cat_both = pd.concat([df_cat, replay_df_cat])

    line_plot_per_mod(data=df_cat_both, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.HERSHMAN, y_col_name="Blink rate in dAT vs AT",
                      hue_col=TASK_RELEVANT, hue_col_name=TASK_RELEVANT,
                      title_name="Average blink rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"replay_v_game_blinkrate_nochosen10",
                      hue_names_map=TASK_RELEVANT_NAME_MAP, palette=TASK_RELEVANT_MAP,
                      same_y_across_mods=True)

    df_cat_both = df_cat_both.loc[(-1000 <= df_cat_both[TIME_IN_EPOCH]) & (df_cat_both[TIME_IN_EPOCH] <= 500)]
    line_plot_per_mod(data=df_cat_both, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.HERSHMAN, y_col_name="Blink rate in dAT vs AT",
                      hue_col=TASK_RELEVANT, hue_col_name=TASK_RELEVANT,
                      title_name="Average blink rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"replay_v_game_blinkrate_nochosen10_zoom",
                      hue_names_map=TASK_RELEVANT_NAME_MAP, palette=TASK_RELEVANT_MAP,
                      black_vertical_x=0, gray_vertical_x=250, same_y_across_mods=True)

    df_cat_both = df_cat_both.loc[(-1000 <= df_cat_both[TIME_IN_EPOCH]) & (df_cat_both[TIME_IN_EPOCH] <= 500)]
    line_plot_per_mod(data=df_cat_both, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=f"{DataParser.HERSHMAN}_zscore", y_col_name="Blink rate in dAT vs AT (z-scored)",
                      hue_col=TASK_RELEVANT, hue_col_name=TASK_RELEVANT,
                      title_name="Average blink rate in Trial (z-scored)", global_y_max=False,
                      save_path=save_path, save_name=f"replay_v_game_blinkrate_nochosen10_zscore_zoom",
                      hue_names_map=TASK_RELEVANT_NAME_MAP, palette=TASK_RELEVANT_MAP,
                      black_vertical_x=0, gray_vertical_x=250,
                      y_lim_dict = {ET_qc_manager.FMRI: (-3, 2.1), ET_qc_manager.MEG: (-3, 2.1)},
                      y_ticks = {ET_qc_manager.FMRI: [-2, -1, 0, 1, 2],
                                 ET_qc_manager.MEG: [-2, -1, 0, 1, 2]},
                      x_ticks=[-1000, -750, -500, -250, 0, 250, 500],
                      black_horizontal_y=0)

    df_cat_both = df_cat_both.loc[(-250 <= df_cat_both[TIME_IN_EPOCH]) & (df_cat_both[TIME_IN_EPOCH] <= 750)]
    line_plot_per_mod(data=df_cat_both, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.HERSHMAN, y_col_name="Blink rate in dAT vs AT",
                      hue_col=TASK_RELEVANT, hue_col_name=TASK_RELEVANT,
                      title_name="Average blink rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"replay_v_game_blinkrate_zoom_250",
                      hue_names_map=TASK_RELEVANT_NAME_MAP, palette=TASK_RELEVANT_MAP,
                      black_vertical_x=0, gray_vertical_x=250, same_y_across_mods=True)
    return


def blink_plot_trial_replay(subs_dict, save_path, load=False):
    """
    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    WE TAKE ONLY GAME TRIALS!!!
    """
    if not load:
        everything_df_list = []
        for target in [True, False]:
            all_dfs_list = []
            for modality in subs_dict.keys():
                only_mod_dfs_list = []
                for sub_data_path in subs_dict[modality]:
                    print(sub_data_path)
                    fl = open(sub_data_path, 'rb')
                    sub_data = pickle.load(fl)
                    fl.close()
                    analyzed_eye = sub_data[PARAMS][DataParser.EYE]
                    trial_samples = sub_data[ET_DATA_DICT][DataParser.DF_SAMPLES]
                    trial_samples[f'{analyzed_eye}{DataParser.HERSHMAN}'].fillna(0,
                                                                                 inplace=True)  # Replace non blinks with 0, so the mean will actually be PROPORTION
                    relevant_trials = sub_data[TRIAL_INFO]
                    relevant_trials.loc[:, TARGET] = relevant_trials[QualityChecker.STIM_TYPE] == relevant_trials["TargetType"]
                    relevant_trials = relevant_trials.loc[(relevant_trials[DataParser.IS_LEVEL_ELIMINATED] == False) & (relevant_trials[TARGET] == target) &
                                                          (relevant_trials[DataParser.REPLAY] == True) & (relevant_trials[ET_data_extraction.IS_PROBED] == True), :]
                    if relevant_trials.empty:
                        continue
                    trial_samples = trial_samples.loc[trial_samples[DataParser.TRIAL].isin(list(relevant_trials[TRIAL_NUMBER])), :]
                    trial_samples = calculate_sample_time_in_trial(relevant_trials, trial_samples, [])

                    trial_samples = trial_samples.loc[trial_samples[DataParser.TRIAL] != -1, :]  # ones that are within a TRIAL epoch (not between trials)
                    # collapse across stimulus duration groups, at each timepoint (from epoch start to end) and COUNT BLINKS
                    trial_means = trial_samples.groupby([TIME_IN_EPOCH]).mean(numeric_only=True).reset_index()
                    trial_means = trial_means.loc[:, [TIME_IN_EPOCH, f"{analyzed_eye}{DataParser.HERSHMAN}"]]
                    trial_means.loc[:, SUBJECT] = sub_data[PARAMS]['SubjectName']
                    trial_means.loc[:, TARGET] = target
                    trial_means.loc[:, MODALITY] = modality
                    trial_means.rename(columns={f"{analyzed_eye}{DataParser.HERSHMAN}": f"{DataParser.HERSHMAN}"}, inplace=True)
                    only_mod_dfs_list.append(trial_means)
                # summarize this modality - average across all subjects' averages (and calculate std)
                mod_mean_df = pd.concat(only_mod_dfs_list)
                all_dfs_list.append(mod_mean_df)

            # Now, we do the all subs plot
            all_mean_df = pd.concat(all_dfs_list)
            everything_df_list.append(all_mean_df)

        # Now, we do the all subs plot
        all_mean_df = pd.concat(everything_df_list)
        all_mean_df.to_csv(os.path.join(save_path, f"blink_plot_target_replay_data.csv"), index=False)
    else:
        all_mean_df = pd.read_csv(os.path.join(save_path, f"blink_plot_target_replay_data.csv"))

    # as we have samples where only one subject had a blink, this skewes the plot. Therefore, create a version w/o it:
    # select EVERYONE BUT CHOSEN 10
    df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]

    count_samps = df_cat.groupby([TIME_IN_EPOCH, TARGET, MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                           (df_cat.is_target.isin(count_samps[TARGET])) &
                           (df_cat["mod"].isin(count_samps["mod"])))]
    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.HERSHMAN, y_col_name="Blink rate in replay",
                      hue_col=TARGET, hue_col_name=TARGET,
                      title_name="Average blink rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"vg_blinkrate_target_replay_all",
                      hue_names_map=TARGET_NAME_MAP, palette=TARGET_MAP)

    return


def blink_plot_vis(subs_dict, save_path, only_objects=False, only_blanks=False, only_faces=False, only_left=False,
                   only_right=False, only_top=False, only_bottom=False, probed=True, load=False):
    """
    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    WE TAKE ONLY GAME TRIALS!!!
    """
    save_name = f"blink_vis_plot_data_{only_faces}_{only_objects}_{only_blanks}_{only_left}_{only_right}_{only_top}_{only_bottom}_{probed}"
    if probed:
        vis_list = ["TruePositive", "FalseNegative"]
    else:
        vis_list = ["Any"]

    if not load:
        everything_df_list = []
        for vis in vis_list:
            all_dfs_list = []
            for modality in subs_dict.keys():
                only_mod_dfs_list = []
                for sub_data_path in subs_dict[modality]:
                    print(sub_data_path)
                    fl = open(sub_data_path, 'rb')
                    sub_data = pickle.load(fl)
                    fl.close()
                    analyzed_eye = sub_data[PARAMS][DataParser.EYE]
                    trial_samples = sub_data[ET_DATA_DICT][DataParser.DF_SAMPLES]
                    trial_samples[f'{analyzed_eye}{DataParser.HERSHMAN}'].fillna(0,
                                                                                 inplace=True)  # Replace non blinks with 0, so the mean will actually be PROPORTION
                    relevant_trials = sub_data[TRIAL_INFO]
                    if only_left or only_right:
                        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(RIGHT), "location"] = RIGHT
                        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(LEFT), "location"] = LEFT
                    else:
                        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(TOP), "location"] = TOP
                        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(BOTTOM), "location"] = BOTTOM

                    if only_faces:
                        relevant_trials = relevant_trials.loc[relevant_trials["stimulusType"] == "Face", :]
                    if only_objects:
                        relevant_trials = relevant_trials.loc[relevant_trials["stimulusType"] == "Object", :]
                    if only_blanks:
                        relevant_trials = relevant_trials.loc[relevant_trials["stimulusType"] == "None", :]
                        if vis == "TruePositive":
                            vis = "FalsePositive"
                        if vis == "FalseNegative":
                            vis = "TrueNegative"
                    if only_left:
                        relevant_trials = relevant_trials.loc[relevant_trials["location"] == LEFT, :]
                    if only_right:
                        relevant_trials = relevant_trials.loc[relevant_trials["location"] == RIGHT, :]
                    if only_top:
                        relevant_trials = relevant_trials.loc[relevant_trials["location"] == TOP, :]
                    if only_bottom:
                        relevant_trials = relevant_trials.loc[relevant_trials["location"] == BOTTOM, :]

                    if vis == "Any":
                        relevant_trials = relevant_trials.loc[(relevant_trials[DataParser.IS_LEVEL_ELIMINATED] == False) &
                                                              (relevant_trials[DataParser.REPLAY] == False) & (relevant_trials[ET_data_extraction.IS_PROBED] == probed), :]
                    else:
                        relevant_trials = relevant_trials.loc[(relevant_trials[DataParser.IS_LEVEL_ELIMINATED] == False) & (relevant_trials[DataParser.VIS] == vis) &
                                                              (relevant_trials[DataParser.REPLAY] == False) & (relevant_trials[ET_data_extraction.IS_PROBED] == probed), :]

                    if relevant_trials.empty:
                        continue
                    trial_samples = trial_samples.loc[
                                    trial_samples[DataParser.TRIAL].isin(list(relevant_trials[TRIAL_NUMBER])), :]
                    trial_samples = calculate_sample_time_in_trial(relevant_trials, trial_samples, [])

                    trial_samples = trial_samples.loc[trial_samples[DataParser.TRIAL] != -1,
                                    :]  # ones that are within a TRIAL epoch (not between trials)
                    # collapse across stimulus duration groups, at each timepoint (from epoch start to end) and COUNT BLINKS
                    trial_means = trial_samples.groupby([TIME_IN_EPOCH]).mean(numeric_only=True).reset_index()
                    trial_means = trial_means.loc[:, [TIME_IN_EPOCH, f"{analyzed_eye}{DataParser.HERSHMAN}"]]
                    trial_means.loc[:, SUBJECT] = sub_data[PARAMS]['SubjectName']
                    trial_means.loc[:, DataParser.VIS] = vis
                    trial_means.loc[:, MODALITY] = modality
                    trial_means.rename(columns={f"{analyzed_eye}{DataParser.HERSHMAN}": f"{DataParser.HERSHMAN}"},
                                       inplace=True)
                    only_mod_dfs_list.append(trial_means)
                # summarize this modality - average across all subjects' averages (and calculate std)
                mod_mean_df = pd.concat(only_mod_dfs_list)
                all_dfs_list.append(mod_mean_df)

            # Now, we do the all subs plot
            all_mean_df = pd.concat(all_dfs_list)
            everything_df_list.append(all_mean_df)

        # Now, we do the all subs plot
        all_mean_df = pd.concat(everything_df_list)
        all_mean_df.to_csv(os.path.join(save_path, save_name), index=False)
    else:
        all_mean_df = pd.read_csv(os.path.join(save_path, save_name))
    return all_mean_df


def blink_plot_trial_unprobe_vis(subs_dict, save_path, load=False):
    save_name = "vg_blinkrate_vis_unprobed"

    sacc_df_blanks = blink_plot_vis(subs_dict, save_path, only_blanks=True, probed=False, load=load)
    sacc_df_objs = blink_plot_vis(subs_dict, save_path, only_objects=True, probed=False, load=load)
    sacc_df_faces = blink_plot_vis(subs_dict, save_path, only_faces=True, probed=False, load=load)

    sacc_df_blanks.loc[:, "target"] = "Blank"
    sacc_df_objs.loc[:, "target"] = "Object"
    sacc_df_faces.loc[:, "target"] = "Face"
    all_mean_df = pd.concat([sacc_df_blanks, sacc_df_objs, sacc_df_faces])

    df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]
    count_samps = df_cat.groupby([TIME_IN_EPOCH, "target", MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                           (df_cat["mod"].isin(count_samps["mod"])))]

    y_lim_dict = {"fMRI": (0, 0.5), "MEG": (0, 0.5)}
    y_ticks = {"fMRI": None, "MEG": None}
    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.HERSHMAN, y_col_name="Blink rate in dAT",
                      hue_col="target", hue_col_name="target",
                      title_name="Average blink rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_all",
                      y_lim_dict=y_lim_dict, y_ticks=y_ticks,
                      hue_names_map=STIMULUS_RELEVANCE_NAME_MAP, palette=STIMULUS_RELEVANCE_MAP)
    return


def blink_plot_trial_unprobe_vis_location(subs_dict, save_path, load=False, left_right=True):

    if left_right:
        sacc_df_left = blink_plot_vis(subs_dict, save_path, only_left=True, probed=False, load=load)
        sacc_df_right = blink_plot_vis(subs_dict, save_path, only_right=True, probed=False, load=load)
        save_name = "vg_blinkrate_vis_location_left_right_unprobed"
    else:
        sacc_df_left = blink_plot_vis(subs_dict, save_path, only_top=True, probed=False, load=load)
        sacc_df_right = blink_plot_vis(subs_dict, save_path, only_bottom=True, probed=False, load=load)
        save_name = "vg_blinkrate_vis_location_top_bottom_unprobed"

    if left_right:
        sacc_df_left.loc[:, "location"] = LEFT
        sacc_df_right.loc[:, "location"] = RIGHT
    else:
        sacc_df_left.loc[:, "location"] = TOP
        sacc_df_right.loc[:, "location"] = BOTTOM

    all_mean_df = pd.concat([sacc_df_left, sacc_df_right])

    df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]
    count_samps = df_cat.groupby([TIME_IN_EPOCH, "location", MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                           (df_cat["mod"].isin(count_samps["mod"])))]

    y_lim_dict = {"fMRI": (0, 0.5), "MEG": (0, 0.5)}
    y_ticks = {"fMRI": None, "MEG": None}
    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.HERSHMAN, y_col_name="Blink rate in dAT",
                      hue_col="location", hue_col_name="location",
                      title_name="Average blink rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_all",
                      y_lim_dict=y_lim_dict, y_ticks=y_ticks,
                      hue_names_map=LOCATION_NAME_MAP, palette=LOCATION_MAP)
    return


def blink_plot_trial_probe_vis(subs_dict, save_path, load=False):
    save_name = "vg_blinkrate_vis"

    sacc_df_blanks = blink_plot_vis(subs_dict, save_path, only_blanks=True, probed=True, load=load)
    sacc_df_objs = blink_plot_vis(subs_dict, save_path, only_objects=True, probed=True, load=load)
    sacc_df_faces = blink_plot_vis(subs_dict, save_path, only_faces=True, probed=True, load=load)

    for obj_type in ["faces", "objects", "blanks"]:
        if obj_type == "faces":
            all_mean_df = sacc_df_faces
        elif obj_type == "objects":
            all_mean_df = sacc_df_objs
        else:
            all_mean_df = sacc_df_blanks
            all_mean_df.loc[all_mean_df[DataParser.VIS] == "FalsePositive", DataParser.VIS] = "TruePositive"
            all_mean_df.loc[all_mean_df[DataParser.VIS] == "TrueNegative", DataParser.VIS] = "FalseNegative"

        df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]
        count_samps = df_cat.groupby([TIME_IN_EPOCH, DataParser.VIS, MODALITY]).count().reset_index()
        count_samps = count_samps[count_samps["sub"] == 1]
        df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                               (df_cat.Visibility.isin(count_samps[DataParser.VIS])) &
                               (df_cat["mod"].isin(count_samps["mod"])))]

        line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                          y_col=DataParser.HERSHMAN, y_col_name="Blink rate in dAT",
                          hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                          title_name="Average blink rate in Trial", global_y_max=False,
                          save_path=save_path, save_name=f"{save_name}_{obj_type}",
                          hue_names_map=VISIBILITY_NAME_MAP, palette=VISIBILITY_MAP)

    sacc_df_objs[QualityChecker.STIM_TYPE] = "Object"
    sacc_df_faces[QualityChecker.STIM_TYPE] = "Face"
    for vis in ["TruePositive", "FalseNegative"]:
        sacc_df_objs[QualityChecker.STIM_TYPE] = "Object"
        sacc_df_faces[QualityChecker.STIM_TYPE] = "Face"

        all_mean_df = pd.concat([sacc_df_objs, sacc_df_faces])
        all_mean_df = all_mean_df[all_mean_df[DataParser.VIS] == vis]
        df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]
        count_samps = df_cat.groupby([TIME_IN_EPOCH, QualityChecker.STIM_TYPE, MODALITY]).count().reset_index()
        count_samps = count_samps[count_samps["sub"] == 1]
        df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                               (df_cat.stimulusType.isin(count_samps[QualityChecker.STIM_TYPE])) &
                               (df_cat["mod"].isin(count_samps["mod"])))]

        line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                          y_col=DataParser.HERSHMAN, y_col_name="Blink rate in dAT",
                          hue_col=QualityChecker.STIM_TYPE, hue_col_name=QualityChecker.STIM_TYPE,
                          title_name="Average blink rate in Trial", global_y_max=False,
                          save_path=save_path, save_name=f"{save_name}_{vis}_all",
                          hue_names_map=STIMULUS_RELEVANCE_NAME_MAP, palette=STIMULUS_RELEVANCE_MAP)

    return


def blink_plot_trial_probe_vis_location(subs_dict, save_path, load=False, left_right=True):

    if left_right:
        sacc_df_left = blink_plot_vis(subs_dict, save_path, only_left=True, probed=True, load=load)
        sacc_df_right = blink_plot_vis(subs_dict, save_path, only_right=True, probed=True, load=load)
        locations = [LEFT, RIGHT]
        save_name = "vg_blinkrate_vis_location_left_right"
    else:
        sacc_df_left = blink_plot_vis(subs_dict, save_path, only_top=True, probed=True, load=load)
        sacc_df_right = blink_plot_vis(subs_dict, save_path, only_bottom=True, probed=True, load=load)
        locations = [TOP, BOTTOM]
        save_name = "vg_blinkrate_vis_location_top_bottom"

    for loc in locations:
        if loc == LEFT or loc == TOP:
            all_mean_df = sacc_df_left
        elif loc == RIGHT or loc == BOTTOM:
            all_mean_df = sacc_df_right

        df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]
        count_samps = df_cat.groupby([TIME_IN_EPOCH, DataParser.VIS, MODALITY]).count().reset_index()
        count_samps = count_samps[count_samps["sub"] == 1]
        df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                               (df_cat.Visibility.isin(count_samps[DataParser.VIS])) &
                               (df_cat["mod"].isin(count_samps["mod"])))]

        line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                          y_col=DataParser.HERSHMAN, y_col_name="Blink rate in dAT",
                          hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                          title_name="Average blink rate in Trial", global_y_max=False,
                          save_path=save_path, save_name=f"{save_name}_{loc}",
                          hue_names_map=VISIBILITY_NAME_MAP, palette=VISIBILITY_MAP)

    if left_right:
        sacc_df_left["location"] = LEFT
        sacc_df_right["location"] = RIGHT
    else:
        sacc_df_left["location"] = TOP
        sacc_df_right["location"] = BOTTOM

    for vis in ["TruePositive", "FalseNegative"]:
        all_mean_df = pd.concat([sacc_df_left, sacc_df_right])
        all_mean_df = all_mean_df[all_mean_df[DataParser.VIS] == vis]
        df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]
        count_samps = df_cat.groupby([TIME_IN_EPOCH, "location", MODALITY]).count().reset_index()
        count_samps = count_samps[count_samps["sub"] == 1]
        df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                               (df_cat.location.isin(count_samps["location"])) &
                               (df_cat["mod"].isin(count_samps["mod"])))]

        line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                          y_col=DataParser.HERSHMAN, y_col_name="Blink rate in dAT",
                          hue_col="location", hue_col_name="location",
                          title_name="Average blink rate in Trial", global_y_max=False,
                          save_path=save_path, save_name=f"{save_name}_{vis}_all",
                          hue_names_map=LOCATION_NAME_MAP, palette=LOCATION_MAP)

    return


def unprobed_to_trials(df_samples, unprobed_data, params):
    # get the inds of the eye tracking data samples (DF_SAMPLES) that match stimulus onset
    probed_stim_onsets = np.array(unprobed_data[DataParser.ONSET])  # remember, these onsets are derived from eyelink trigger messages
    probed_stim_onsets_sample_inds = np.array([np.where(df_samples[DataParser.T_SAMPLE] == onset)[0][0] for onset in probed_stim_onsets])

    # get the inds of the eye tracking data samples that match different interesting events (e.g.epoch beginning)
    # and then add to trial_info the timestamps of the samples that match these events for each trial
    interesting_events = {DataParser.EPOCH + DataParser.WINDOW_START: ET_param_manager.EPOCH_START,  # EPOCH start (with respect to stim ONSET)
                         DataParser.EPOCH + DataParser.WINDOW_END: ET_param_manager.EPOCH_END,  # EPOCH end (with respect to stim ONSET)
                         DataParser.PRE_STIM_DUR + DataParser.WINDOW_START: ET_param_manager.PRE_STIM_DUR,  # pre-stimulus start (end=onset)
                         DataParser.STIM_DUR + DataParser.WINDOW_END: ET_param_manager.STIM_DUR}  # stim duration (start=onset)

    for event in interesting_events:
        event_time = interesting_events[event] / 1000  # div by 1000 to turn MILLISECONDS TO SECONDS
        rel = event_time * params[DataParser.SAMPLING_FREQ] if "End" in event else event_time * params[DataParser.SAMPLING_FREQ] * (-1)
        event_sample_inds = list(np.ceil(probed_stim_onsets_sample_inds + rel).astype(int))
        event_samples = np.array(df_samples.iloc[event_sample_inds, 0])
        unprobed_data[event] = event_samples

    # prepare a column in all ET dataframes to contain the trial number of samples within each epoch
    time_windows = {DataParser.TRIAL: (DataParser.EPOCH + DataParser.WINDOW_START, DataParser.EPOCH + DataParser.WINDOW_END)}

    for window in time_windows:
        df_samples[window] = -1

    for index, trial in unprobed_data.iterrows():  # iterate trials
        for window in time_windows:
            window_start = trial[time_windows[window][0]]
            window_end = trial[time_windows[window][1]]
            df_samples.loc[(df_samples[DataParser.T_SAMPLE] <= window_end) & (df_samples[DataParser.T_SAMPLE] >= window_start), window] = trial["trialNumber"]
            df_samples.loc[(df_samples[DataParser.T_SAMPLE] <= window_end) & (df_samples[DataParser.T_SAMPLE] >= window_start), DataParser.IS_LEVEL_ELIMINATED] = trial[DataParser.IS_LEVEL_ELIMINATED]

    return df_samples


def saccade_plot_trial_unprobed_visibility(subs_dict, save_path, only_micro=False, only_saccades=False, load=False):
    """
    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    WE TAKE ONLY GAME TRIALS!!!
    """
    if only_saccades:
        save_name = "vg_saccrate_vis_unprobed_sacc_only"
        data_save_name = "saccade_vis_unprobed_plot_data_sacc_only"
    elif only_micro:
        save_name = "vg_saccrate_vis_unprobed_micro_only"
        data_save_name = "saccade_vis_unprobed_plot_data_micro_only"
    else:
        save_name = "vg_saccrate_vis_unprobed"
        data_save_name = "saccade_vis_unprobed_plot_data"

    if not load:
        everything_df_list = []
        all_dfs_list = []
        my_cnt = 0
        for modality in subs_dict.keys():
            only_mod_dfs_list = []
            for sub_data_path in subs_dict[modality]:
                print(sub_data_path)
                fl = open(sub_data_path, 'rb')
                sub_data = pickle.load(fl)
                fl.close()
                sub_name = sub_data[PARAMS][SUBJECT_NAME]
                if sub_name in ["SA144", "SD193"]:
                    continue
                print(my_cnt)
                my_cnt += 1
                unprobed_csv = [f for f in os.listdir(sub_data_path[:sub_data_path.rfind('/')]) if "trial_data_unprobed_full" in f][0]
                unprobed_df = pd.read_csv(os.path.join(sub_data_path[:sub_data_path.rfind('/')], unprobed_csv))
                trial_samples = sub_data[ET_DATA_DICT][DataParser.DF_SAMPLES]
                trial_samples = unprobed_to_trials(trial_samples, unprobed_df, sub_data[PARAMS])

                if only_saccades:
                    trial_samples.loc[trial_samples["microsaccade"] == True, f'{DataParser.REAL_SACC}'] = False
                elif only_micro:
                    trial_samples.loc[trial_samples["microsaccade"] == False, f'{DataParser.REAL_SACC}'] = False
                trial_samples[f'{DataParser.REAL_SACC}'].fillna(0, inplace=True)  # Replace non saccades with 0, so the mean will actually be PROPORTION

                relevant_trials = unprobed_df
                relevant_trials = relevant_trials.loc[(relevant_trials[DataParser.IS_LEVEL_ELIMINATED] == False), :]
                if relevant_trials.empty:
                    continue
                trial_samples = trial_samples.loc[trial_samples[DataParser.TRIAL].isin(list(relevant_trials[TRIAL_NUMBER])), :]
                trial_samples = calculate_sample_time_in_trial(relevant_trials, trial_samples, [])

                trial_samples = trial_samples.loc[trial_samples[DataParser.TRIAL] != -1, :]  # ones that are within a TRIAL epoch (not between trials)
                # collapse across stimulus duration groups, at each timepoint (from epoch start to end) and COUNT SACCADES
                trial_means = trial_samples.groupby([TIME_IN_EPOCH]).mean(numeric_only=True).reset_index()
                trial_means = trial_means.loc[:, [TIME_IN_EPOCH, f"{DataParser.REAL_SACC}"]]
                trial_means.loc[:, SUBJECT] = sub_data[PARAMS]['SubjectName']
                trial_means.loc[:, LAB] = sub_data[PARAMS]['SubjectName'][:2]
                trial_means.loc[:, DataParser.VIS] = "All"
                trial_means.loc[:, MODALITY] = modality
                only_mod_dfs_list.append(trial_means)
            # summarize this modality - average across all subjects' averages (and calculate std)
            mod_mean_df = pd.concat(only_mod_dfs_list)
            all_dfs_list.append(mod_mean_df)

        # Now, we do the all subs plot
        all_mean_df = pd.concat(all_dfs_list)
        everything_df_list.append(all_mean_df)

        # Now, we do the all subs plot
        all_mean_df = pd.concat(everything_df_list)
        all_mean_df.to_csv(os.path.join(save_path, f"{data_save_name}.csv"), index=False)
    else:
        all_mean_df = pd.read_csv(os.path.join(save_path, f"{data_save_name}.csv"))

    # as we have samples where only one subject had a saccade, this skewes the plot. Therefore, create a version w/o it:
    # select all BUT CHOSEN 10
    df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]

    for lab in list(df_cat[LAB].unique()):
        df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]
        df_cat = df_cat[df_cat[LAB] == lab]
        count_samps = df_cat.groupby([TIME_IN_EPOCH, DataParser.VIS, MODALITY]).count().reset_index()
        count_samps = count_samps[count_samps["sub"] == 1]
        df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                               (df_cat.Visibility.isin(count_samps[DataParser.VIS])) &
                               (df_cat["mod"].isin(count_samps["mod"])))]
        line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                          y_col=DataParser.REAL_SACC, y_col_name="Saccade rate in dAT",
                          hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                          title_name="Average saccade rate in Trial", global_y_max=False,
                          save_path=save_path, save_name=f"{save_name}_{lab}_all",
                          hue_names_map=VISIBILITY_NAME_MAP, palette=VISIBILITY_MAP)

    df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]
    count_samps = df_cat.groupby([TIME_IN_EPOCH, DataParser.VIS, MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                           (df_cat.Visibility.isin(count_samps[DataParser.VIS])) &
                           (df_cat["mod"].isin(count_samps["mod"])))]
    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.REAL_SACC, y_col_name="Saccade rate in dAT",
                      hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      title_name="Average saccade rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_all",
                      hue_names_map=VISIBILITY_NAME_MAP, palette=VISIBILITY_MAP)

    # Chosen 10
    df_cat = all_mean_df[all_mean_df[SUBJECT].isin(CHOSEN_10)]
    df_cat[MODALITY] = "all"

    count_samps = df_cat.groupby([TIME_IN_EPOCH, DataParser.VIS, MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                           (df_cat.Visibility.isin(count_samps[DataParser.VIS])) &
                           (df_cat["mod"].isin(count_samps["mod"])))]
    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.REAL_SACC, y_col_name="Saccade rate in dAT",
                      hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      title_name="Average saccade rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_chosen10",
                      hue_names_map=VISIBILITY_NAME_MAP, palette=VISIBILITY_MAP)
    return


def saccade_plot_trial_visibility(subs_dict, save_path, only_micro=False, only_saccades=False, only_faces=False,
                                  only_objects=False, only_blanks=False, probed=True, only_right=False, only_left=False,
                                  only_top=False, only_bottom=False, load=False):
    """
    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    WE TAKE ONLY GAME TRIALS!!!
    """
    if only_saccades:
        save_name = "vg_saccrate_vis_sacc_only"
        data_save_name = f"saccade_vis_plot_data_sacc_only_{only_faces}_{only_objects}_{only_blanks}_{probed}_{only_left}_{only_right}_{only_top}_{only_bottom}"
    elif only_micro:
        save_name = "vg_saccrate_vis_micro_only"
        data_save_name = f"saccade_vis_plot_data_micro_only_{only_faces}_{only_objects}_{only_blanks}_{probed}_{only_left}_{only_right}_{only_top}_{only_bottom}"
    else:
        save_name = "vg_saccrate_vis"
        data_save_name = f"saccade_vis_plot_data_{only_faces}_{only_objects}_{only_blanks}_{probed}_{only_left}_{only_right}_{only_top}_{only_bottom}"

    if probed:
        vis_list = ["TruePositive", "FalseNegative"]
    else:
        vis_list = ["Any"]

    if not load:
        everything_df_list = []
        for vis in vis_list:
            all_dfs_list = []
            for modality in subs_dict.keys():
                only_mod_dfs_list = []
                for sub_data_path in subs_dict[modality]:
                    print(sub_data_path)
                    fl = open(sub_data_path, 'rb')
                    sub_data = pickle.load(fl)
                    fl.close()
                    trial_samples = sub_data[ET_DATA_DICT][DataParser.DF_SAMPLES]
                    if only_saccades:
                        trial_samples.loc[trial_samples["microsaccade"] == True, f'{DataParser.REAL_SACC}'] = False
                    elif only_micro:
                        trial_samples.loc[trial_samples["microsaccade"] == False, f'{DataParser.REAL_SACC}'] = False
                    trial_samples[f'{DataParser.REAL_SACC}'].fillna(0, inplace=True)  # Replace non saccades with 0, so the mean will actually be PROPORTION

                    relevant_trials = sub_data[TRIAL_INFO]

                    # For Z score floofy
                    """
                    DEPRECATED z-score calculation, now we calculate BL correction seperately per condition
                    
                    relevant_trials_zscore = relevant_trials.loc[(relevant_trials[DataParser.IS_LEVEL_ELIMINATED] == False) & (relevant_trials[DataParser.REPLAY] == False), :]
                    trial_samples_zscore = trial_samples.loc[trial_samples[DataParser.TRIAL].isin(list(relevant_trials_zscore[TRIAL_NUMBER])), :]
                    trial_samples_zscore = calculate_sample_time_in_trial(relevant_trials_zscore, trial_samples_zscore, [])
                    
                    trial_samples_zscore = trial_samples_zscore.loc[trial_samples_zscore[DataParser.TRIAL] != -1, :]  # ones that are within a TRIAL epoch (not between trials)
                    # collapse across stimulus duration groups, at each timepoint (from epoch start to end) and COUNT SACCADES
                    trial_means = trial_samples_zscore.groupby([TIME_IN_EPOCH]).mean(numeric_only=True).reset_index()
                    sub_mean = trial_means.loc[(trial_means[TIME_IN_EPOCH] >= -1000) & (trial_means[TIME_IN_EPOCH] <= -500), DataParser.REAL_SACC].mean()
                    sub_std = trial_means.loc[(trial_means[TIME_IN_EPOCH] >= -1000) & (trial_means[TIME_IN_EPOCH] <= -500), DataParser.REAL_SACC].std()
                    """
                    # End of zscore floofy

                    if only_right or only_left:
                        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(RIGHT), "location"] = RIGHT
                        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(LEFT), "location"] = LEFT
                    else:
                        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(TOP), "location"] = TOP
                        relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(BOTTOM), "location"] = BOTTOM

                    if only_right:
                        relevant_trials = relevant_trials.loc[relevant_trials["location"] == RIGHT, :]
                    if only_left:
                        relevant_trials = relevant_trials.loc[relevant_trials["location"] == LEFT, :]
                    if only_top:
                        relevant_trials = relevant_trials.loc[relevant_trials["location"] == TOP, :]
                    if only_bottom:
                        relevant_trials = relevant_trials.loc[relevant_trials["location"] == BOTTOM, :]
                    if only_faces:
                        relevant_trials = relevant_trials.loc[relevant_trials["stimulusType"] == "Face", :]
                    if only_objects:
                        relevant_trials = relevant_trials.loc[relevant_trials["stimulusType"] == "Object", :]
                    if only_blanks:
                        relevant_trials = relevant_trials.loc[relevant_trials["stimulusType"] == "None", :]
                        if vis == "TruePositive":
                            vis = "FalsePositive"
                        if vis == "FalseNegative":
                            vis = "TrueNegative"

                    if vis == "Any":
                        relevant_trials = relevant_trials.loc[(relevant_trials[DataParser.IS_LEVEL_ELIMINATED] == False) &
                                                              (relevant_trials[DataParser.REPLAY] == False) & (relevant_trials[ET_data_extraction.IS_PROBED] == probed), :]
                    else:
                        relevant_trials = relevant_trials.loc[(relevant_trials[DataParser.IS_LEVEL_ELIMINATED] == False) & (relevant_trials[DataParser.VIS] == vis) &
                                                              (relevant_trials[DataParser.REPLAY] == False) & (relevant_trials[ET_data_extraction.IS_PROBED] == probed), :]

                    if relevant_trials.empty:
                        continue

                    relevant_trials_zscore = relevant_trials.loc[(relevant_trials[DataParser.IS_LEVEL_ELIMINATED] == False) & (relevant_trials[DataParser.REPLAY] == False), :]
                    trial_samples_zscore = trial_samples.loc[trial_samples[DataParser.TRIAL].isin(list(relevant_trials_zscore[TRIAL_NUMBER])), :]
                    trial_samples_zscore = calculate_sample_time_in_trial(relevant_trials_zscore, trial_samples_zscore, [])

                    trial_samples_zscore = trial_samples_zscore.loc[trial_samples_zscore[DataParser.TRIAL] != -1, :]  # ones that are within a TRIAL epoch (not between trials)
                    # collapse across stimulus duration groups, at each timepoint (from epoch start to end) and COUNT SACCADES
                    trial_means = trial_samples_zscore.groupby([TIME_IN_EPOCH]).mean(numeric_only=True).reset_index()
                    sub_mean = trial_means.loc[(trial_means[TIME_IN_EPOCH] >= -1000) & (trial_means[TIME_IN_EPOCH] <= -500), DataParser.REAL_SACC].mean()
                    sub_std = trial_means.loc[(trial_means[TIME_IN_EPOCH] >= -1000) & (trial_means[TIME_IN_EPOCH] <= -500), DataParser.REAL_SACC].std()

                    trial_samples = trial_samples.loc[trial_samples[DataParser.TRIAL].isin(list(relevant_trials[TRIAL_NUMBER])), :]
                    trial_samples = calculate_sample_time_in_trial(relevant_trials, trial_samples, [])

                    trial_samples = trial_samples.loc[trial_samples[DataParser.TRIAL] != -1, :]  # ones that are within a TRIAL epoch (not between trials)
                    # collapse across stimulus duration groups, at each timepoint (from epoch start to end) and COUNT SACCADES
                    trial_means = trial_samples.groupby([TIME_IN_EPOCH]).mean(numeric_only=True).reset_index()
                    trial_means = trial_means.loc[:, [TIME_IN_EPOCH, f"{DataParser.REAL_SACC}"]]
                    trial_means.loc[:, SUBJECT] = sub_data[PARAMS]['SubjectName']
                    trial_means.loc[:, LAB] = sub_data[PARAMS]['SubjectName'][:2]
                    trial_means.loc[:, DataParser.VIS] = vis
                    trial_means.loc[:, MODALITY] = modality
                    if sub_mean == 0 and sub_std == 0:
                        trial_means.loc[:, f"{DataParser.REAL_SACC}_zscore"] = 0
                    else:
                        trial_means.loc[:, f"{DataParser.REAL_SACC}_zscore"] = (trial_means[DataParser.REAL_SACC] - sub_mean) / sub_std
                    only_mod_dfs_list.append(trial_means)
                # summarize this modality - average across all subjects' averages (and calculate std)
                mod_mean_df = pd.concat(only_mod_dfs_list)
                all_dfs_list.append(mod_mean_df)

            # Now, we do the all subs plot
            all_mean_df = pd.concat(all_dfs_list)
            everything_df_list.append(all_mean_df)

        # Now, we do the all subs plot
        all_mean_df = pd.concat(everything_df_list)
        all_mean_df.to_csv(os.path.join(save_path, f"{data_save_name}.csv"), index=False)
    else:
        all_mean_df = pd.read_csv(os.path.join(save_path, f"{data_save_name}.csv"))
    return all_mean_df


def saccade_plot_replay_trial_task(subs_dict, save_path, only_micro=False, only_saccades=False, load=False):
    """
    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    WE TAKE ONLY REPLAY TRIALS!!!
    """
    if only_saccades:
        data_save_name = f"saccade_replay_task_plot_data_sacc_only"
    elif only_micro:
        data_save_name = f"saccade_replay_task_plot_data_micro_only"
    else:
        data_save_name = f"saccade_replay_task_plot_data"

    if not load:
        everything_df_list = []
        for task_relevant in [True, False]:
            all_dfs_list = []
            for modality in subs_dict.keys():
                only_mod_dfs_list = []
                for sub_data_path in subs_dict[modality]:
                    print(sub_data_path)
                    fl = open(sub_data_path, 'rb')
                    sub_data = pickle.load(fl)
                    fl.close()
                    trial_samples = sub_data[ET_DATA_DICT][DataParser.DF_SAMPLES]
                    if only_saccades:
                        trial_samples.loc[trial_samples["microsaccade"] == True, f'{DataParser.REAL_SACC}'] = False
                    elif only_micro:
                        trial_samples.loc[trial_samples["microsaccade"] == False, f'{DataParser.REAL_SACC}'] = False
                    trial_samples[f'{DataParser.REAL_SACC}'].fillna(0, inplace=True)  # Replace non saccades with 0, so the mean will actually be PROPORTION

                    relevant_trials = sub_data[TRIAL_INFO]
                    relevant_trials = relevant_trials.loc[(relevant_trials[DataParser.IS_LEVEL_ELIMINATED] == False) &
                                                          (relevant_trials[DataParser.REPLAY] == True) &
                                                          (relevant_trials[ET_data_extraction.IS_PROBED] == True), :]

                    if task_relevant == True:
                        relevant_trials = relevant_trials.loc[~relevant_trials[QualityChecker.STIM_TYPE].str.contains("None")]
                    else:
                        relevant_trials = relevant_trials.loc[relevant_trials[QualityChecker.STIM_TYPE].str.contains("None")]

                    if relevant_trials.empty:
                        continue

                    # For Z score, note that we calculate the zscore according to the GAME BASELINE, similar to GAME
                    trial_samples_zscore = trial_samples.loc[trial_samples[DataParser.TRIAL].isin(list(relevant_trials[TRIAL_NUMBER])), :]
                    trial_samples_zscore = calculate_sample_time_in_trial(relevant_trials, trial_samples_zscore, [])

                    trial_samples_zscore = trial_samples_zscore.loc[trial_samples_zscore[DataParser.TRIAL] != -1, :]  # ones that are within a TRIAL epoch (not between trials)
                    # collapse across stimulus duration groups, at each timepoint (from epoch start to end) and COUNT SACCADES
                    trial_means = trial_samples_zscore.groupby([TIME_IN_EPOCH]).mean(numeric_only=True).reset_index()
                    sub_mean = trial_means.loc[(trial_means[TIME_IN_EPOCH] >= -1000) & (trial_means[TIME_IN_EPOCH] <= -500), DataParser.REAL_SACC].mean()
                    sub_std = trial_means.loc[(trial_means[TIME_IN_EPOCH] >= -1000) & (trial_means[TIME_IN_EPOCH] <= -500), DataParser.REAL_SACC].std()
                    # End of zscore floofy

                    trial_samples = trial_samples.loc[trial_samples[DataParser.TRIAL].isin(list(relevant_trials[TRIAL_NUMBER])), :]
                    trial_samples = calculate_sample_time_in_trial(relevant_trials, trial_samples, [])

                    trial_samples = trial_samples.loc[trial_samples[DataParser.TRIAL] != -1, :]  # ones that are within a TRIAL epoch (not between trials)
                    # collapse across stimulus duration groups, at each timepoint (from epoch start to end) and COUNT SACCADES
                    trial_means = trial_samples.groupby([TIME_IN_EPOCH]).mean(numeric_only=True).reset_index()
                    trial_means = trial_means.loc[:, [TIME_IN_EPOCH, f"{DataParser.REAL_SACC}"]]
                    trial_means.loc[:, SUBJECT] = sub_data[PARAMS]['SubjectName']
                    trial_means.loc[:, LAB] = sub_data[PARAMS]['SubjectName'][:2]
                    trial_means.loc[:, "task_relevant"] = task_relevant
                    trial_means.loc[:, MODALITY] = modality
                    if sub_mean == 0 and sub_std == 0:
                        trial_means.loc[:, f"{DataParser.REAL_SACC}_zscore"] = 0
                    else:
                        trial_means.loc[:, f"{DataParser.REAL_SACC}_zscore"] = (trial_means[DataParser.REAL_SACC] - sub_mean) / sub_std
                    only_mod_dfs_list.append(trial_means)
                # summarize this modality - average across all subjects' averages (and calculate std)
                mod_mean_df = pd.concat(only_mod_dfs_list)
                all_dfs_list.append(mod_mean_df)

            # Now, we do the all subs plot
            all_mean_df = pd.concat(all_dfs_list)
            everything_df_list.append(all_mean_df)

        # Now, we do the all subs plot
        all_mean_df = pd.concat(everything_df_list)
        all_mean_df.to_csv(os.path.join(save_path, f"{data_save_name}.csv"), index=False)
    else:
        all_mean_df = pd.read_csv(os.path.join(save_path, f"{data_save_name}.csv"))
    return all_mean_df


def saccade_plot_replay_target(subs_dict, save_path, load=False, window_width=100):
    """
    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    WE TAKE ONLY GAME TRIALS!!!
    """

    save_name = "replay_saccrate_target"
    data_save_name = f"replay_saccrate_target_data"

    if not load:
        everything_df_list = []
        for target in [True, False]:
            all_dfs_list = []
            for modality in subs_dict.keys():
                only_mod_dfs_list = []
                for sub_data_path in subs_dict[modality]:
                    print(sub_data_path)
                    fl = open(sub_data_path, 'rb')
                    sub_data = pickle.load(fl)
                    fl.close()
                    trial_samples = sub_data[ET_DATA_DICT][DataParser.DF_SAMPLES]
                    trial_samples[f'{DataParser.REAL_SACC}'].fillna(0, inplace=True)  # Replace non saccades with 0, so the mean will actually be PROPORTION

                    relevant_trials = sub_data[TRIAL_INFO]
                    relevant_trials.loc[:, TARGET] = relevant_trials[QualityChecker.STIM_TYPE] == relevant_trials["TargetType"]
                    relevant_trials = relevant_trials.loc[(relevant_trials[DataParser.IS_LEVEL_ELIMINATED] == False) & (relevant_trials[TARGET] == target) &
                                      (relevant_trials[DataParser.REPLAY] == True) & (relevant_trials[ET_data_extraction.IS_PROBED] == True), :]

                    if relevant_trials.empty:
                        continue
                    trial_samples = trial_samples.loc[
                                    trial_samples[DataParser.TRIAL].isin(list(relevant_trials[TRIAL_NUMBER])), :]
                    trial_samples = calculate_sample_time_in_trial(relevant_trials, trial_samples, [])

                    trial_samples = trial_samples.loc[trial_samples[DataParser.TRIAL] != -1,
                                    :]  # ones that are within a TRIAL epoch (not between trials)
                    # collapse across stimulus duration groups, at each timepoint (from epoch start to end) and COUNT SACCADES
                    trial_means = trial_samples.groupby([TIME_IN_EPOCH]).mean(numeric_only=True).reset_index()
                    trial_means = trial_means.loc[:, [TIME_IN_EPOCH, f"{DataParser.REAL_SACC}"]]
                    trial_means.loc[:, SUBJECT] = sub_data[PARAMS]['SubjectName']
                    trial_means.loc[:, LAB] = sub_data[PARAMS]['SubjectName'][:2]
                    trial_means.loc[:, TARGET] = target
                    trial_means.loc[:, MODALITY] = modality
                    only_mod_dfs_list.append(trial_means)
                # summarize this modality - average across all subjects' averages (and calculate std)
                mod_mean_df = pd.concat(only_mod_dfs_list)
                all_dfs_list.append(mod_mean_df)

            # Now, we do the all subs plot
            all_mean_df = pd.concat(all_dfs_list)
            everything_df_list.append(all_mean_df)

        # Now, we do the all subs plot
        all_mean_df = pd.concat(everything_df_list)
        all_mean_df.to_csv(os.path.join(save_path, f"{data_save_name}.csv"), index=False)
    else:
        all_mean_df = pd.read_csv(os.path.join(save_path, f"{data_save_name}.csv"))

    df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]
    count_samps = df_cat.groupby([TIME_IN_EPOCH, TARGET, MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                           (df_cat.is_target.isin(count_samps[TARGET])) &
                           (df_cat["mod"].isin(count_samps["mod"])))]

    # SMOOTHING - old
    """
    data = np.pad(list(df_cat[DataParser.REAL_SACC]), int(window_width / 2), mode='edge')
    cumsum_vec = np.cumsum(data)
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    df_cat[DataParser.REAL_SACC] = ma_vec
    """
    df_cat = df_cat.sort_values(by=["sub", TARGET, "timeInEpoch"])
    df_cat[DataParser.REAL_SACC] = (
        df_cat.groupby(["sub", TARGET])[DataParser.REAL_SACC]
            .transform(lambda x: x.rolling(window=window_width, center=True, min_periods=1).mean())
    )

    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.REAL_SACC, y_col_name="Saccade rate in replay",
                      hue_col=TARGET, hue_col_name=TARGET,
                      title_name="Average saccade rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_all",
                      hue_names_map=TARGET_NAME_MAP, palette=TARGET_MAP)

    df_cat = df_cat.loc[(-500 <= df_cat[TIME_IN_EPOCH]) & (df_cat[TIME_IN_EPOCH] <= 500)]
    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.REAL_SACC, y_col_name="Saccade rate in replay",
                      hue_col=TARGET, hue_col_name=TARGET,
                      title_name="Average saccade rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_all_zoom",
                      hue_names_map=TARGET_NAME_MAP, palette=TARGET_MAP)
    return


def saccade_trio(subs_dict, save_path, load=False, window_width=100):
    save_name = "replay_saccrate"

    sacc_replay_df = saccade_plot_replay_trial_task(subs_dict, save_path, load=load)
    all_mean_df = sacc_replay_df
    df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]
    count_samps = df_cat.groupby([TIME_IN_EPOCH, TASK_RELEVANT, MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                           (df_cat.task_relevant.isin(count_samps[TASK_RELEVANT])) &
                           (df_cat["mod"].isin(count_samps["mod"])))]

    # SMOOTHING
    """
    data = np.pad(list(df_cat[DataParser.REAL_SACC]), int(window_width / 2), mode='edge')
    cumsum_vec = np.cumsum(data)
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    df_cat[DataParser.REAL_SACC] = ma_vec

    data = np.pad(list(df_cat[f"{DataParser.REAL_SACC}_zscore"]), int(window_width / 2), mode='edge')
    cumsum_vec = np.cumsum(data)
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    df_cat[f"{DataParser.REAL_SACC}_zscore"] = ma_vec
    """
    df_cat = df_cat.sort_values(by=["sub", TASK_RELEVANT, "timeInEpoch"])
    df_cat[DataParser.REAL_SACC] = (
        df_cat.groupby(["sub", TASK_RELEVANT])[DataParser.REAL_SACC]
            .transform(lambda x: x.rolling(window=window_width, center=True, min_periods=1).mean())
    )
    df_cat[f"{DataParser.REAL_SACC}_zscore"] = (
        df_cat.groupby(["sub", TASK_RELEVANT])[f"{DataParser.REAL_SACC}_zscore"]
            .transform(lambda x: x.rolling(window=window_width, center=True, min_periods=1).mean())
    )

    df_cat_replay = df_cat

    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.REAL_SACC, y_col_name="Saccade rate in AT",
                      hue_col=TASK_RELEVANT, hue_col_name=TASK_RELEVANT,
                      title_name="Average saccade rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_withmicro_nochosen10",
                      hue_names_map=TASK_RELEVANT_NAME_MAP, palette=TASK_RELEVANT_MAP,
                      same_y_across_mods=True)

    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=f"{DataParser.REAL_SACC}_zscore", y_col_name="Saccade Number",
                      hue_col=TASK_RELEVANT, hue_col_name=TASK_RELEVANT,
                      title_name="Average saccade rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_withmicro_nochosen10_zscore",
                      hue_names_map=TASK_RELEVANT_NAME_MAP, palette=TASK_RELEVANT_MAP, same_y_across_mods=True)

    df_cat = df_cat.loc[(-500 <= df_cat[TIME_IN_EPOCH]) & (df_cat[TIME_IN_EPOCH] <= 500)]
    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.REAL_SACC, y_col_name="Saccade Rate",
                      hue_col=TASK_RELEVANT, hue_col_name=TASK_RELEVANT,
                      title_name="Average saccade rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_withmicro_nochosen10_zoom",
                      hue_names_map=TASK_RELEVANT_NAME_MAP, palette=TASK_RELEVANT_MAP, same_y_across_mods=True,
                      black_vertical_x=0, gray_vertical_x=250)

    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=f"{DataParser.REAL_SACC}_zscore", y_col_name="Saccade Rate (z-scored)",
                      hue_col=TASK_RELEVANT, hue_col_name=TASK_RELEVANT,
                      title_name="Average saccade rate in Trial (z-scored)", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_withmicro_nochosen10_zscore_zoom",
                      hue_names_map=TASK_RELEVANT_NAME_MAP, palette=TASK_RELEVANT_MAP, same_y_across_mods=True,
                      black_vertical_x=0, gray_vertical_x=250,
                      x_ticks=[-1000, -750, -500, -250, 0, 250, 500],
                      black_horizontal_y=0)

    sacc_game_df = saccade_plot_trial_visibility(subs_dict, save_path, only_saccades=False, only_micro=False,
                                                 only_faces=False,
                                                 only_objects=False, only_blanks=False, probed=True, load=load)
    sacc_game_df[TASK_RELEVANT] = "Seen"
    sacc_game_df = sacc_game_df[sacc_game_df[DataParser.VIS] == "TruePositive"]
    all_mean_df = sacc_game_df
    df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]
    count_samps = df_cat.groupby([TIME_IN_EPOCH, TASK_RELEVANT, MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                           (df_cat.task_relevant.isin(count_samps[TASK_RELEVANT])) &
                           (df_cat["mod"].isin(count_samps["mod"])))]

    # SMOOTHING
    """
    data = np.pad(list(df_cat[DataParser.REAL_SACC]), int(window_width / 2), mode='edge')
    cumsum_vec = np.cumsum(data)
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    df_cat[DataParser.REAL_SACC] = ma_vec

    data = np.pad(list(df_cat[f"{DataParser.REAL_SACC}_zscore"]), int(window_width / 2), mode='edge')
    cumsum_vec = np.cumsum(data)
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    df_cat[f"{DataParser.REAL_SACC}_zscore"] = ma_vec
    """
    df_cat = df_cat.sort_values(by=["sub", TASK_RELEVANT, "timeInEpoch"])
    df_cat[DataParser.REAL_SACC] = (
        df_cat.groupby(["sub", TASK_RELEVANT])[DataParser.REAL_SACC]
            .transform(lambda x: x.rolling(window=window_width, center=True, min_periods=1).mean())
    )
    df_cat[f"{DataParser.REAL_SACC}_zscore"] = (
        df_cat.groupby(["sub", TASK_RELEVANT])[f"{DataParser.REAL_SACC}_zscore"]
            .transform(lambda x: x.rolling(window=window_width, center=True, min_periods=1).mean())
    )

    save_name = "replay_v_game_saccrate"
    both_df = pd.concat([df_cat, df_cat_replay])
    line_plot_per_mod(data=both_df, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.REAL_SACC, y_col_name="Saccade rate in AT vs dAT",
                      hue_col=TASK_RELEVANT, hue_col_name=TASK_RELEVANT,
                      title_name="Average saccade rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_withmicro_nochosen10",
                      hue_names_map=TASK_RELEVANT_NAME_MAP, palette=TASK_RELEVANT_MAP,
                      same_y_across_mods=True)

    line_plot_per_mod(data=both_df, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=f"{DataParser.REAL_SACC}_zscore", y_col_name="Saccade rate (z-scored)",
                      hue_col=TASK_RELEVANT, hue_col_name=TASK_RELEVANT,
                      title_name="Average saccade rate in Trial (z-scored)", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_withmicro_nochosen10_zscore",
                      hue_names_map=TASK_RELEVANT_NAME_MAP, palette=TASK_RELEVANT_MAP, same_y_across_mods=True)

    both_df = both_df.loc[(-1000 <= both_df[TIME_IN_EPOCH]) & (both_df[TIME_IN_EPOCH] <= 500)]
    line_plot_per_mod(data=both_df, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.REAL_SACC, y_col_name="Saccade rate",
                      hue_col=TASK_RELEVANT, hue_col_name=TASK_RELEVANT,
                      title_name="Average saccade rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_withmicro_nochosen10_zoom",
                      hue_names_map=TASK_RELEVANT_NAME_MAP, palette=TASK_RELEVANT_MAP, same_y_across_mods=True,
                      black_vertical_x=0, gray_vertical_x=250)

    line_plot_per_mod(data=both_df, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=f"{DataParser.REAL_SACC}_zscore", y_col_name="Saccade Rate (z-scored)",
                      hue_col=TASK_RELEVANT, hue_col_name=TASK_RELEVANT,
                      title_name="Average saccade rate in Trial (z-scored)", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_withmicro_nochosen10_zscore_zoom",
                      hue_names_map=TASK_RELEVANT_NAME_MAP, palette=TASK_RELEVANT_MAP,
                      black_vertical_x=0, gray_vertical_x=250,
                      y_lim_dict={ET_qc_manager.FMRI: (-1, 4), ET_qc_manager.MEG: (-1, 4)},
                      y_ticks={ET_qc_manager.FMRI: [-1, 0, 1, 2, 3],
                               ET_qc_manager.MEG: [-1, 0, 1, 2, 3]},
                      x_ticks=[-1000, -750, -500, -250, 0, 250, 500],
                      black_horizontal_y=0)
    return


def saccade_plot_trial_visibility_no_split(subs_dict, save_path, load=False, window_width=100, w_micro=True):
    if w_micro:
        w_micro_str = "withmicro"
    else:
        w_micro_str = "womicro"
    save_name = "vg_saccrate_vis"

    if w_micro:
        sacc_df = saccade_plot_trial_visibility(subs_dict, save_path, only_saccades=False, only_micro=False, only_faces=False,
                                                only_objects=False, only_blanks=False, probed=True, load=load)
    else:
        sacc_df = saccade_plot_trial_visibility(subs_dict, save_path, only_saccades=True, only_micro=False,
                                                only_faces=False, only_objects=False, only_blanks=False, probed=True, load=load)

    #sacc_df[DataParser.VIS] = sacc_df[DataParser.VIS] + "_Saccade"
    all_mean_df = sacc_df

    df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]
    count_samps = df_cat.groupby([TIME_IN_EPOCH, DataParser.VIS, MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                           (df_cat.Visibility.isin(count_samps[DataParser.VIS])) &
                           (df_cat["mod"].isin(count_samps["mod"])))]

    # SMOOTHING
    """
    data = np.pad(list(df_cat[DataParser.REAL_SACC]), int(window_width / 2), mode='edge')
    cumsum_vec = np.cumsum(data)
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    df_cat[DataParser.REAL_SACC] = ma_vec

    data = np.pad(list(df_cat[f"{DataParser.REAL_SACC}_zscore"]), int(window_width / 2), mode='edge')
    cumsum_vec = np.cumsum(data)
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    df_cat[f"{DataParser.REAL_SACC}_zscore"] = ma_vec
    """
    df_cat = df_cat.sort_values(by=["sub", DataParser.VIS, "timeInEpoch"])
    df_cat[DataParser.REAL_SACC] = (
        df_cat.groupby(["sub", DataParser.VIS])[DataParser.REAL_SACC]
            .transform(lambda x: x.rolling(window=window_width, center=True, min_periods=1).mean())
    )
    df_cat[f"{DataParser.REAL_SACC}_zscore"] = (
        df_cat.groupby(["sub", DataParser.VIS])[f"{DataParser.REAL_SACC}_zscore"]
            .transform(lambda x: x.rolling(window=window_width, center=True, min_periods=1).mean())
    )

    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.REAL_SACC, y_col_name="Saccade rate in dAT",
                      hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      title_name="Average saccade rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_{w_micro_str}_nochosen10",
                      hue_names_map=VISIBILITY_NAME_MAP, palette=VISIBILITY_MAP,
                      y_lim_dict = {ET_qc_manager.FMRI: (0,0.06), ET_qc_manager.MEG: (0, 0.06)},
                      y_ticks = {ET_qc_manager.FMRI: [0.01, 0.02, 0.03, 0.04, 0.05], ET_qc_manager.MEG:  [0.01, 0.02, 0.03, 0.04, 0.05]})

    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=f"{DataParser.REAL_SACC}_zscore", y_col_name="Saccade Rate (z-scored)",
                      hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      title_name="Average saccade rate in Trial (z-scored)", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_{w_micro_str}_nochosen10_zscore",
                      hue_names_map=VISIBILITY_NAME_MAP, palette=VISIBILITY_MAP, same_y_across_mods=True)

    df_cat = df_cat.loc[(-1000 <= df_cat[TIME_IN_EPOCH]) & (df_cat[TIME_IN_EPOCH] <= 500)]
    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.REAL_SACC, y_col_name="Saccade Rate (z-scored)",
                      hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      title_name="Average saccade rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_{w_micro_str}_nochosen10_zoom",
                      hue_names_map=VISIBILITY_NAME_MAP, palette=VISIBILITY_MAP, same_y_across_mods=True,
                      black_vertical_x=0, gray_vertical_x=250)

    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=f"{DataParser.REAL_SACC}_zscore", y_col_name="Saccade Rate (z-scored)",
                      hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      title_name="Average saccade rate in Trial (z-scored)", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_{w_micro_str}_nochosen10_zscore_zoom",
                      hue_names_map=VISIBILITY_NAME_MAP, palette=VISIBILITY_MAP, same_y_across_mods=True,
                      black_vertical_x=0, gray_vertical_x=250,
                      y_lim_dict = {ET_qc_manager.FMRI: (-1, 1), ET_qc_manager.MEG: (-1, 1)},
                      y_ticks = {ET_qc_manager.FMRI: [-1, -0.5, 0, 0.5, 1],
                                 ET_qc_manager.MEG: [-1, -0.5, 0, 0.5, 1]},
                      x_ticks=[-1000, -750, -500, -250, 0, 250, 500],
                      black_horizontal_y=0)

    """
    # Chosen 10
    df_cat = all_mean_df[all_mean_df[SUBJECT].isin(CHOSEN_10)]

    count_samps = df_cat.groupby([TIME_IN_EPOCH, DataParser.VIS, MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    df_mod = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                           (df_cat.Visibility.isin(count_samps[DataParser.VIS])) &
                           (df_cat["mod"].isin(count_samps["mod"])))]

    line_plot_per_mod(data=df_mod, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.REAL_SACC, y_col_name="Saccade rate in dAT",
                      hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      title_name="Average saccade rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_chosen10",
                      hue_names_map=SACC_VISIBILITY_NAME_MAP, palette=SACC_VISIBILITY_MAP)

    df_cat[MODALITY] = "all"

    count_samps = df_cat.groupby([TIME_IN_EPOCH, DataParser.VIS, MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                           (df_cat.Visibility.isin(count_samps[DataParser.VIS])) &
                           (df_cat["mod"].isin(count_samps["mod"])))]
    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.REAL_SACC, y_col_name="Saccade rate in dAT",
                      hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      title_name="Average saccade rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_chosen10",
                      hue_names_map=SACC_VISIBILITY_NAME_MAP, palette=SACC_VISIBILITY_MAP)
    """
    return


def saccade_plot_trial_visibility_no_split_objects(subs_dict, save_path, load=False, window_width=100):
    save_name = "vg_saccrate_vis_target"

    sacc_df_blanks = saccade_plot_trial_visibility(subs_dict, save_path, only_saccades=False, only_micro=False, only_blanks=True, probed=True, load=load)
    sacc_df_objs = saccade_plot_trial_visibility(subs_dict, save_path, only_saccades=False, only_micro=False, only_objects=True, probed=True, load=load)
    sacc_df_faces = saccade_plot_trial_visibility(subs_dict, save_path, only_saccades=False, only_micro=False, only_faces=True, probed=True, load=load)

    for obj_type in ["faces", "objects", "blanks"]:
        if obj_type == "faces":
            all_mean_df = sacc_df_faces
        elif obj_type == "objects":
            all_mean_df = sacc_df_objs
        else:
            all_mean_df = sacc_df_blanks
            all_mean_df.loc[all_mean_df[DataParser.VIS] == "FalsePositive", DataParser.VIS] = "TruePositive"
            all_mean_df.loc[all_mean_df[DataParser.VIS] == "TrueNegative", DataParser.VIS] = "FalseNegative"

        df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]
        count_samps = df_cat.groupby([TIME_IN_EPOCH, DataParser.VIS, MODALITY]).count().reset_index()
        count_samps = count_samps[count_samps["sub"] == 1]
        df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                               (df_cat.Visibility.isin(count_samps[DataParser.VIS])) &
                               (df_cat["mod"].isin(count_samps["mod"])))]

        # SMOOTHING
        """
        data = np.pad(list(df_cat[DataParser.REAL_SACC]), int(window_width / 2), mode='edge')
        cumsum_vec = np.cumsum(data)
        ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
        df_cat[DataParser.REAL_SACC] = ma_vec

        data = np.pad(list(df_cat[f"{DataParser.REAL_SACC}_zscore"]), int(window_width / 2), mode='edge')
        cumsum_vec = np.cumsum(data)
        ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
        df_cat[f"{DataParser.REAL_SACC}_zscore"] = ma_vec
        """
        df_cat = df_cat.sort_values(by=["sub", DataParser.VIS, "timeInEpoch"])
        df_cat[DataParser.REAL_SACC] = (
            df_cat.groupby(["sub", DataParser.VIS])[DataParser.REAL_SACC]
                .transform(lambda x: x.rolling(window=window_width, center=True, min_periods=1).mean())
        )
        df_cat[f"{DataParser.REAL_SACC}_zscore"] = (
            df_cat.groupby(["sub", DataParser.VIS])[f"{DataParser.REAL_SACC}_zscore"]
                .transform(lambda x: x.rolling(window=window_width, center=True, min_periods=1).mean())
        )

        line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                          y_col=DataParser.REAL_SACC, y_col_name="Saccade rate in dAT",
                          hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                          title_name="Average saccade rate in Trial", global_y_max=False,
                          save_path=save_path, save_name=f"{save_name}_{obj_type}_all",
                          hue_names_map=VISIBILITY_NAME_MAP, palette=VISIBILITY_MAP)
        line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                          y_col=f"{DataParser.REAL_SACC}_zscore", y_col_name="Saccade rate in dAT",
                          hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                          title_name="Average saccade rate in Trial", global_y_max=False,
                          save_path=save_path, save_name=f"{save_name}_{obj_type}_all_zscore",
                          hue_names_map=VISIBILITY_NAME_MAP, palette=VISIBILITY_MAP)

    sacc_df_objs[QualityChecker.STIM_TYPE] = "Object"
    sacc_df_faces[QualityChecker.STIM_TYPE] = "Face"
    for vis in ["TruePositive", "FalseNegative"]:
        sacc_df_objs[QualityChecker.STIM_TYPE] = "Object"
        sacc_df_faces[QualityChecker.STIM_TYPE] = "Face"

        all_mean_df = pd.concat([sacc_df_objs, sacc_df_faces])
        all_mean_df = all_mean_df[all_mean_df[DataParser.VIS] == vis]
        df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]
        count_samps = df_cat.groupby([TIME_IN_EPOCH, QualityChecker.STIM_TYPE, MODALITY]).count().reset_index()
        count_samps = count_samps[count_samps["sub"] == 1]
        df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                               (df_cat.stimulusType.isin(count_samps[QualityChecker.STIM_TYPE])) &
                               (df_cat["mod"].isin(count_samps["mod"])))]

        # SMOOTHING
        data = np.pad(list(df_cat[DataParser.REAL_SACC]), int(window_width / 2), mode='edge')
        cumsum_vec = np.cumsum(data)
        ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
        df_cat[DataParser.REAL_SACC] = ma_vec

        data = np.pad(list(df_cat[f"{DataParser.REAL_SACC}_zscore"]), int(window_width / 2), mode='edge')
        cumsum_vec = np.cumsum(data)
        ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
        df_cat[f"{DataParser.REAL_SACC}_zscore"] = ma_vec

        line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                          y_col=DataParser.REAL_SACC, y_col_name="Saccade rate in dAT",
                          hue_col=QualityChecker.STIM_TYPE, hue_col_name=QualityChecker.STIM_TYPE,
                          title_name="Average saccade rate in Trial", global_y_max=False,
                          save_path=save_path, save_name=f"{save_name}_{vis}_all",
                          hue_names_map=STIMULUS_RELEVANCE_NAME_MAP, palette=STIMULUS_RELEVANCE_MAP)
        line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                          y_col=f"{DataParser.REAL_SACC}_zscore", y_col_name="Saccade rate in dAT",
                          hue_col=QualityChecker.STIM_TYPE, hue_col_name=QualityChecker.STIM_TYPE,
                          title_name="Average saccade rate in Trial", global_y_max=False,
                          save_path=save_path, save_name=f"{save_name}_{vis}_all_zscore",
                          hue_names_map=STIMULUS_RELEVANCE_NAME_MAP, palette=STIMULUS_RELEVANCE_MAP)

    """
    sacc_df_objs[DataParser.VIS] = sacc_df_objs[DataParser.VIS] + "_Objects"
    sacc_df_faces[DataParser.VIS] = sacc_df_faces[DataParser.VIS] + "_Faces"
    all_mean_df = pd.concat([sacc_df_objs, sacc_df_faces])

    df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]
    count_samps = df_cat.groupby([TIME_IN_EPOCH, DataParser.VIS, MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                           (df_cat.Visibility.isin(count_samps[DataParser.VIS])) &
                           (df_cat["mod"].isin(count_samps["mod"])))]

    # SMOOTHING TODO: CHECK PARAMS
    data = np.pad(list(df_cat[DataParser.REAL_SACC]), int(window_width / 2), mode='edge')
    cumsum_vec = np.cumsum(data)
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    df_cat[DataParser.REAL_SACC] = ma_vec

    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.REAL_SACC, y_col_name="Saccade rate in dAT",
                      hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      title_name="Average saccade rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_all",
                      hue_names_map=SACC_VISIBILITY_OBJECTS_NAME_MAP, palette=SACC_VISIBILITY_OBJECTS_MAP)

    # Chosen 10
    df_cat = all_mean_df[all_mean_df[SUBJECT].isin(CHOSEN_10)]

    count_samps = df_cat.groupby([TIME_IN_EPOCH, DataParser.VIS, MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    df_mod = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                           (df_cat.Visibility.isin(count_samps[DataParser.VIS])) &
                           (df_cat["mod"].isin(count_samps["mod"])))]

    line_plot_per_mod(data=df_mod, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.REAL_SACC, y_col_name="Saccade rate in dAT",
                      hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      title_name="Average saccade rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_chosen10",
                      hue_names_map=SACC_VISIBILITY_OBJECTS_NAME_MAP, palette=SACC_VISIBILITY_OBJECTS_MAP)

    df_cat[MODALITY] = "all"

    count_samps = df_cat.groupby([TIME_IN_EPOCH, DataParser.VIS, MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                           (df_cat.Visibility.isin(count_samps[DataParser.VIS])) &
                           (df_cat["mod"].isin(count_samps["mod"])))]
    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.REAL_SACC, y_col_name="Saccade rate in dAT",
                      hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      title_name="Average saccade rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_chosen10",
                      hue_names_map=SACC_VISIBILITY_OBJECTS_NAME_MAP, palette=SACC_VISIBILITY_OBJECTS_MAP)
    """

    return


def saccade_plot_trial_visibility_no_split_locations(subs_dict, save_path, load=False, window_width=100, left_right=True):

    if left_right:
        sacc_df_left = saccade_plot_trial_visibility(subs_dict, save_path, only_saccades=False, only_micro=False, only_blanks=False, probed=True, only_left=True, load=load)
        sacc_df_right = saccade_plot_trial_visibility(subs_dict, save_path, only_saccades=False, only_micro=False, only_blanks=False, probed=True, only_right=True, load=load)
        locations = [LEFT, RIGHT]
        save_name = "vg_saccrate_vis_location_left_right"
    else:
        sacc_df_left = saccade_plot_trial_visibility(subs_dict, save_path, only_saccades=False, only_micro=False, only_blanks=False, probed=True, only_top=True, load=load)
        sacc_df_right = saccade_plot_trial_visibility(subs_dict, save_path, only_saccades=False, only_micro=False, only_blanks=False, probed=True, only_bottom=True, load=load)
        locations = [TOP, BOTTOM]
        save_name = "vg_saccrate_vis_location_top_bottom"

    for loc in locations:
        if loc == LEFT or loc == TOP:
            all_mean_df = sacc_df_left
        elif loc == RIGHT or loc == BOTTOM:
            all_mean_df = sacc_df_right

        df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]
        count_samps = df_cat.groupby([TIME_IN_EPOCH, DataParser.VIS, MODALITY]).count().reset_index()
        count_samps = count_samps[count_samps["sub"] == 1]
        df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                               (df_cat.Visibility.isin(count_samps[DataParser.VIS])) &
                               (df_cat["mod"].isin(count_samps["mod"])))]

        # SMOOTHING
        data = np.pad(list(df_cat[DataParser.REAL_SACC]), int(window_width / 2), mode='edge')
        cumsum_vec = np.cumsum(data)
        ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
        df_cat[DataParser.REAL_SACC] = ma_vec

        data = np.pad(list(df_cat[f"{DataParser.REAL_SACC}_zscore"]), int(window_width / 2), mode='edge')
        cumsum_vec = np.cumsum(data)
        ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
        df_cat[f"{DataParser.REAL_SACC}_zscore"] = ma_vec

        line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                          y_col=DataParser.REAL_SACC, y_col_name="Saccade rate in dAT",
                          hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                          title_name="Average saccade rate in Trial", global_y_max=False,
                          save_path=save_path, save_name=f"{save_name}_{loc}_all",
                          hue_names_map=VISIBILITY_NAME_MAP, palette=VISIBILITY_MAP)
        line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                          y_col=f"{DataParser.REAL_SACC}_zscore", y_col_name="Saccade rate in dAT",
                          hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                          title_name="Average saccade rate in Trial", global_y_max=False,
                          save_path=save_path, save_name=f"{save_name}_{loc}_all_zscore",
                          hue_names_map=VISIBILITY_NAME_MAP, palette=VISIBILITY_MAP)

    if left_right:
        sacc_df_left["location"] = LEFT
        sacc_df_right["location"] = RIGHT
    else:
        sacc_df_left["location"] = TOP
        sacc_df_right["location"] = BOTTOM

    for vis in ["TruePositive", "FalseNegative"]:
        all_mean_df = pd.concat([sacc_df_left, sacc_df_right])
        all_mean_df = all_mean_df[all_mean_df[DataParser.VIS] == vis]
        df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]
        count_samps = df_cat.groupby([TIME_IN_EPOCH, "location", MODALITY]).count().reset_index()
        count_samps = count_samps[count_samps["sub"] == 1]
        df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                               (df_cat.location.isin(count_samps["location"])) &
                               (df_cat["mod"].isin(count_samps["mod"])))]

        # SMOOTHING
        data = np.pad(list(df_cat[DataParser.REAL_SACC]), int(window_width / 2), mode='edge')
        cumsum_vec = np.cumsum(data)
        ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
        df_cat[DataParser.REAL_SACC] = ma_vec

        data = np.pad(list(df_cat[f"{DataParser.REAL_SACC}_zscore"]), int(window_width / 2), mode='edge')
        cumsum_vec = np.cumsum(data)
        ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
        df_cat[f"{DataParser.REAL_SACC}_zscore"] = ma_vec

        line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                          y_col=DataParser.REAL_SACC, y_col_name="Saccade rate in dAT",
                          hue_col="location", hue_col_name="location",
                          title_name="Average saccade rate in Trial", global_y_max=False,
                          save_path=save_path, save_name=f"{save_name}_{vis}_all",
                          hue_names_map=LOCATION_NAME_MAP, palette=LOCATION_MAP)
        line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                          y_col=f"{DataParser.REAL_SACC}_zscore", y_col_name="Saccade rate in dAT",
                          hue_col="location", hue_col_name="location",
                          title_name="Average saccade rate in Trial", global_y_max=False,
                          save_path=save_path, save_name=f"{save_name}_{vis}_all_zscore",
                          hue_names_map=LOCATION_NAME_MAP, palette=LOCATION_MAP)
    return


def saccade_plot_trial_visibility_no_split_locations_unprobed(subs_dict, save_path, load=False, window_width=100, left_right=True):

    if left_right:
        sacc_df_left = saccade_plot_trial_visibility(subs_dict, save_path, only_saccades=False, only_micro=False, only_blanks=False, only_left=True, probed=False, load=load)
        sacc_df_right = saccade_plot_trial_visibility(subs_dict, save_path, only_saccades=False, only_micro=False, only_objects=False, only_right=True, probed=False, load=load)
        locations = [LEFT, RIGHT]
        save_name = "vg_saccrate_vis_location_unprobed_left_right"
    else:
        sacc_df_left = saccade_plot_trial_visibility(subs_dict, save_path, only_saccades=False, only_micro=False, only_blanks=False, only_top=True, probed=False, load=load)
        sacc_df_right = saccade_plot_trial_visibility(subs_dict, save_path, only_saccades=False, only_micro=False, only_objects=False, only_bottom=True, probed=False, load=load)
        locations = [TOP, BOTTOM]
        save_name = "vg_saccrate_vis_location_unprobed_top_bottom"

    if left_right:
        sacc_df_left.loc[:, "location"] = LEFT
        sacc_df_right.loc[:, "location"] = RIGHT
    else:
        sacc_df_left.loc[:, "location"] = TOP
        sacc_df_right.loc[:, "location"] = BOTTOM

    all_mean_df = pd.concat([sacc_df_left, sacc_df_right])

    df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]
    count_samps = df_cat.groupby([TIME_IN_EPOCH, "location", MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                           (df_cat.location.isin(count_samps["location"])) &
                           (df_cat["mod"].isin(count_samps["mod"])))]

    # SMOOTHING
    data = np.pad(list(df_cat[DataParser.REAL_SACC]), int(window_width / 2), mode='edge')
    cumsum_vec = np.cumsum(data)
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    df_cat[DataParser.REAL_SACC] = ma_vec

    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.REAL_SACC, y_col_name="Saccade rate in dAT",
                      hue_col="location", hue_col_name="location",
                      title_name="Average saccade rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_all",
                      hue_names_map=LOCATION_NAME_MAP, palette=LOCATION_MAP)

    # SMOOTHING
    data = np.pad(list(df_cat[f"{DataParser.REAL_SACC}_zscore"]), int(window_width / 2), mode='edge')
    cumsum_vec = np.cumsum(data)
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    df_cat[f"{DataParser.REAL_SACC}_zscore"] = ma_vec

    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=f"{DataParser.REAL_SACC}_zscore", y_col_name="Saccade rate in dAT",
                      hue_col="location", hue_col_name="location",
                      title_name="Average saccade rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_all_zscore",
                      hue_names_map=LOCATION_NAME_MAP, palette=LOCATION_MAP)
    return


def saccade_plot_trial_visibility_no_split_objects_unprobed(subs_dict, save_path, load=False, window_width=100):
    save_name = "vg_saccrate_vis_target_unprobed"

    sacc_df_blanks = saccade_plot_trial_visibility(subs_dict, save_path, only_saccades=False, only_micro=False, only_blanks=True, probed=False, load=load)
    sacc_df_objs = saccade_plot_trial_visibility(subs_dict, save_path, only_saccades=False, only_micro=False, only_objects=True, probed=False, load=load)
    sacc_df_faces = saccade_plot_trial_visibility(subs_dict, save_path, only_saccades=False, only_micro=False, only_faces=True, probed=False, load=load)

    sacc_df_blanks.loc[:, "target"] = "Blank"
    sacc_df_objs.loc[:, "target"] = "Object"
    sacc_df_faces.loc[:, "target"] = "Face"
    all_mean_df = pd.concat([sacc_df_blanks, sacc_df_objs, sacc_df_faces])

    df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]
    count_samps = df_cat.groupby([TIME_IN_EPOCH, "target", MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                           (df_cat.Visibility.isin(count_samps[DataParser.VIS])) &
                           (df_cat["mod"].isin(count_samps["mod"])))]

    # SMOOTHING
    data = np.pad(list(df_cat[DataParser.REAL_SACC]), int(window_width / 2), mode='edge')
    cumsum_vec = np.cumsum(data)
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    df_cat[DataParser.REAL_SACC] = ma_vec

    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.REAL_SACC, y_col_name="Saccade rate in dAT",
                      hue_col="target", hue_col_name="target",
                      title_name="Average saccade rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_all",
                      hue_names_map=STIMULUS_RELEVANCE_NAME_MAP, palette=STIMULUS_RELEVANCE_MAP)

    # SMOOTHING
    data = np.pad(list(df_cat[f"{DataParser.REAL_SACC}_zscore"]), int(window_width / 2), mode='edge')
    cumsum_vec = np.cumsum(data)
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    df_cat[f"{DataParser.REAL_SACC}_zscore"] = ma_vec

    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=f"{DataParser.REAL_SACC}_zscore", y_col_name="Saccade rate in dAT",
                      hue_col="target", hue_col_name="target",
                      title_name="Average saccade rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_all_zscore",
                      hue_names_map=STIMULUS_RELEVANCE_NAME_MAP, palette=STIMULUS_RELEVANCE_MAP)
    return


def saccade_plot_trial_visibility_all(subs_dict, save_path, load=False, window_width=100):
    save_name = "vg_saccrate_vis_split"
    sacc_df = saccade_plot_trial_visibility(subs_dict, save_path, only_saccades=True, load=load)
    mic_sacc_df = saccade_plot_trial_visibility(subs_dict, save_path, only_micro=True, load=load)

    sacc_df[DataParser.VIS] = sacc_df[DataParser.VIS] + "_Saccade"
    mic_sacc_df[DataParser.VIS] = mic_sacc_df[DataParser.VIS] + "_Microsaccade"
    all_mean_df = pd.concat([sacc_df, mic_sacc_df])
    """
    # PER LAB - NEEDED?
    
    # as we have samples where only one subject had a saccade, this skewes the plot. Therefore, create a version w/o it:
    # select all BUT CHOSEN 10
    df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]

    for lab in list(df_cat[LAB].unique()):
    df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]
    df_cat = df_cat[df_cat[LAB] == lab]
    count_samps = df_cat.groupby([TIME_IN_EPOCH, DataParser.VIS, MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                           (df_cat.Visibility.isin(count_samps[DataParser.VIS])) &
                           (df_cat["mod"].isin(count_samps["mod"])))]
    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.REAL_SACC, y_col_name="Saccade rate in dAT",
                      hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      title_name="Average saccade rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_{lab}_all",
                      hue_names_map=VISIBILITY_NAME_MAP, palette=VISIBILITY_MAP)

    """

    df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]
    count_samps = df_cat.groupby([TIME_IN_EPOCH, DataParser.VIS, MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                           (df_cat.Visibility.isin(count_samps[DataParser.VIS])) &
                           (df_cat["mod"].isin(count_samps["mod"])))]

    # SMOOTHING
    data = np.pad(list(df_cat[DataParser.REAL_SACC]), int(window_width / 2), mode='edge')
    cumsum_vec = np.cumsum(data)
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    df_cat[DataParser.REAL_SACC] = ma_vec

    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.REAL_SACC, y_col_name="Saccade rate in dAT",
                      hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      title_name="Average saccade rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_all",
                      hue_names_map=SACC_VISIBILITY_NAME_MAP, palette=SACC_VISIBILITY_MAP)

    # Chosen 10
    df_cat = all_mean_df[all_mean_df[SUBJECT].isin(CHOSEN_10)]

    count_samps = df_cat.groupby([TIME_IN_EPOCH, DataParser.VIS, MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    df_mod = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                           (df_cat.Visibility.isin(count_samps[DataParser.VIS])) &
                           (df_cat["mod"].isin(count_samps["mod"])))]

    # SMOOTHING
    data = np.pad(list(df_mod[DataParser.REAL_SACC]), int(window_width / 2), mode='edge')
    cumsum_vec = np.cumsum(data)
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    df_mod[DataParser.REAL_SACC] = ma_vec

    line_plot_per_mod(data=df_mod, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.REAL_SACC, y_col_name="Saccade rate in dAT",
                      hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      title_name="Average saccade rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_chosen10",
                      hue_names_map=SACC_VISIBILITY_NAME_MAP, palette=SACC_VISIBILITY_MAP)

    df_cat[MODALITY] = "all"

    count_samps = df_cat.groupby([TIME_IN_EPOCH, DataParser.VIS, MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                           (df_cat.Visibility.isin(count_samps[DataParser.VIS])) &
                           (df_cat["mod"].isin(count_samps["mod"])))]
    # SMOOTHING
    data = np.pad(list(df_cat[DataParser.REAL_SACC]), int(window_width / 2), mode='edge')
    cumsum_vec = np.cumsum(data)
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    df_cat[DataParser.REAL_SACC] = ma_vec

    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.REAL_SACC, y_col_name="Saccade rate in dAT",
                      hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      title_name="Average saccade rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"{save_name}_chosen10",
                      hue_names_map=SACC_VISIBILITY_NAME_MAP, palette=SACC_VISIBILITY_MAP)

    return


def saccade_plot_trial_blank_visibility(subs_dict, save_path, load=False):
    """
    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    WE TAKE ONLY GAME TRIALS!!!
    """

    if not load:
        everything_df_list = []
        for vis in ["FalsePositive", "TrueNegative"]:
            all_dfs_list = []
            for modality in subs_dict.keys():
                only_mod_dfs_list = []
                for sub_data_path in subs_dict[modality]:
                    print(sub_data_path)
                    fl = open(sub_data_path, 'rb')
                    sub_data = pickle.load(fl)
                    fl.close()
                    trial_samples = sub_data[ET_DATA_DICT][DataParser.DF_SAMPLES]
                    trial_samples[f'{DataParser.REAL_SACC}'].fillna(0, inplace=True)  # Replace non saccades with 0, so the mean will actually be PROPORTION

                    relevant_trials = sub_data[TRIAL_INFO]
                    relevant_trials = relevant_trials.loc[(relevant_trials[DataParser.IS_LEVEL_ELIMINATED] == False) & (relevant_trials[DataParser.VIS] == vis) & (relevant_trials[DataParser.REPLAY] == False), :]
                    if relevant_trials.empty:
                        continue
                    trial_samples = trial_samples.loc[trial_samples[DataParser.TRIAL].isin(list(relevant_trials[TRIAL_NUMBER])), :]
                    trial_samples = calculate_sample_time_in_trial(relevant_trials, trial_samples, [])

                    trial_samples = trial_samples.loc[trial_samples[DataParser.TRIAL] != -1, :]  # ones that are within a TRIAL epoch (not between trials)
                    # collapse across stimulus duration groups, at each timepoint (from epoch start to end) and COUNT SACCADES
                    trial_means = trial_samples.groupby([TIME_IN_EPOCH]).mean(numeric_only=True).reset_index()
                    trial_means = trial_means.loc[:, [TIME_IN_EPOCH, f"{DataParser.REAL_SACC}"]]
                    trial_means.loc[:, SUBJECT] = sub_data[PARAMS]['SubjectName']
                    trial_means.loc[:, LAB] = sub_data[PARAMS]['SubjectName'][:2]
                    if vis == "FalsePositive":
                        trial_means.loc[:, DataParser.VIS] = "TruePositive"
                    else:
                        trial_means.loc[:, DataParser.VIS] = "FalseNegative"
                    trial_means.loc[:, MODALITY] = modality
                    only_mod_dfs_list.append(trial_means)
                # summarize this modality - average across all subjects' averages (and calculate std)
                mod_mean_df = pd.concat(only_mod_dfs_list)
                all_dfs_list.append(mod_mean_df)

            # Now, we do the all subs plot
            all_mean_df = pd.concat(all_dfs_list)
            everything_df_list.append(all_mean_df)

        # Now, we do the all subs plot
        all_mean_df = pd.concat(everything_df_list)
        all_mean_df.to_csv(os.path.join(save_path, f"saccade_vis_blank_plot_data.csv"), index=False)
    else:
        all_mean_df = pd.read_csv(os.path.join(save_path, f"saccade_vis_blank_plot_data.csv"))

    # as we have samples where only one subject had a saccade, this skewes the plot. Therefore, create a version w/o it:
    # select all BUT CHOSEN 10
    df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]

    for lab in list(df_cat[LAB].unique()):
        df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]
        df_cat = df_cat[df_cat[LAB] == lab]
        count_samps = df_cat.groupby([TIME_IN_EPOCH, DataParser.VIS, MODALITY]).count().reset_index()
        count_samps = count_samps[count_samps["sub"] == 1]
        df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                               (df_cat.Visibility.isin(count_samps[DataParser.VIS])) &
                               (df_cat["mod"].isin(count_samps["mod"])))]
        line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                          y_col=DataParser.REAL_SACC, y_col_name="Saccade rate in dAT",
                          hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                          title_name="Average saccade rate in Trial", global_y_max=False,
                          save_path=save_path, save_name=f"vg_saccrate_vis_blank_{lab}_all",
                          hue_names_map=VISIBILITY_NAME_MAP, palette=VISIBILITY_MAP)

    df_cat = all_mean_df[~all_mean_df[SUBJECT].isin(CHOSEN_10)]
    count_samps = df_cat.groupby([TIME_IN_EPOCH, DataParser.VIS, MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                           (df_cat.Visibility.isin(count_samps[DataParser.VIS])) &
                           (df_cat["mod"].isin(count_samps["mod"])))]
    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.REAL_SACC, y_col_name="Saccade rate in dAT",
                      hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      title_name="Average saccade rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"vg_saccrate_vis_blank_all",
                      hue_names_map=VISIBILITY_NAME_MAP, palette=VISIBILITY_MAP)

    # Chosen 10
    df_cat = all_mean_df[all_mean_df[SUBJECT].isin(CHOSEN_10)]
    df_cat[MODALITY] = "all"

    count_samps = df_cat.groupby([TIME_IN_EPOCH, DataParser.VIS, MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                           (df_cat.Visibility.isin(count_samps[DataParser.VIS])) &
                           (df_cat["mod"].isin(count_samps["mod"])))]
    line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.REAL_SACC, y_col_name="Saccade rate in dAT",
                      hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      title_name="Average saccade rate in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"vg_saccrate_vis_blank_chosen10",
                      hue_names_map=VISIBILITY_NAME_MAP, palette=VISIBILITY_MAP)
    return


def saccade_trial_replay_target_data(subs_dict, save_path, microsaccade=False, load=False):
    """
    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    WE TAKE ONLY REPLAY TRIALS!!!
    """

    if microsaccade:
        f_name = "microsacc"
    else:
        f_name = "saccade"

    if not load:
        everything_df_list = []
        for category in ET_param_manager.STIM_TYPES:
            all_dfs_list = []
            for modality in subs_dict.keys():
                only_mod_dfs_list = []
                for sub_data_path in subs_dict[modality]:
                    print(sub_data_path)
                    fl = open(sub_data_path, 'rb')
                    sub_data = pickle.load(fl)
                    fl.close()
                    analyzed_eye = sub_data[PARAMS][DataParser.EYE]
                    trial_samples = sub_data[ET_DATA_DICT][DataParser.DF_SAMPLES]
                    if not microsaccade:
                        trial_samples.loc[trial_samples["microsaccade"] == True, f'{DataParser.REAL_SACC}'] = False
                    else:
                        trial_samples.loc[trial_samples["microsaccade"] == False, f'{DataParser.REAL_SACC}'] = False

                    trial_samples[f'{DataParser.REAL_SACC}'].fillna(0, inplace=True)  # Replace non saccades with 0, so the mean will actually be PROPORTION

                    relevant_trials = sub_data[TRIAL_INFO]
                    relevant_trials = relevant_trials.loc[(relevant_trials[DataParser.IS_LEVEL_ELIMINATED] == False) & (relevant_trials[QualityChecker.STIM_TYPE] == category) & (relevant_trials[DataParser.REPLAY] == True), :]
                    if relevant_trials.empty:
                        continue
                    trial_samples = trial_samples.loc[trial_samples[DataParser.TRIAL].isin(list(relevant_trials[TRIAL_NUMBER])), :]
                    trial_samples = calculate_sample_time_in_trial(relevant_trials, trial_samples, [])

                    trial_samples = trial_samples.loc[trial_samples[DataParser.TRIAL] != -1, :]  # ones that are within a TRIAL epoch (not between trials)
                    # collapse across stimulus duration groups, at each timepoint (from epoch start to end) and COUNT SACCADES
                    trial_means = trial_samples.groupby([TIME_IN_EPOCH]).mean(numeric_only=True).reset_index()
                    trial_means = trial_means.loc[:, [TIME_IN_EPOCH, f"{DataParser.REAL_SACC}"]]
                    trial_means.loc[:, SUBJECT] = sub_data[PARAMS]['SubjectName']
                    trial_means.loc[:, QualityChecker.STIM_TYPE] = category
                    trial_means.loc[:, MODALITY] = modality
                    only_mod_dfs_list.append(trial_means)
                # summarize this modality - average across all subjects' averages (and calculate std)
                mod_mean_df = pd.concat(only_mod_dfs_list)
                all_dfs_list.append(mod_mean_df)

            # Now, we do the all subs plot
            all_mean_df = pd.concat(all_dfs_list)
            everything_df_list.append(all_mean_df)

        # Now, we do the all subs plot
        all_mean_df = pd.concat(everything_df_list)
        all_mean_df.to_csv(os.path.join(save_path, f"saccade_replay_target_{f_name}_data.csv"), index=False)
    else:
        all_mean_df = pd.read_csv(os.path.join(save_path, f"saccade_replay_target_{f_name}_data.csv"))

    return all_mean_df


def calculate_fixation_density_va(num_of_bins_x, num_of_bins_y, gaze_x, gaze_y, minimal_dims_va, params):
    """
    This function divides the screen into bins and sums the time during which a gaze was present at each bin.
    NOTE: we assume that regardless of the actual screen size, the visual stimulus was THE SAME SIZE IN VISUAL ANGLES
    for ALL subjects (i.e., larger screens --> subjects sat farther).
    For fixation density, we are interested in a
    histogram showing the fixation distances (in VA) from the screen center and from points of interest in the screen
    (which all should be the SAME DISTANCE IN VA across subjects). Thus, when aggregating across labs with different
    screen sizes, we assume that all screens spanned the same VA SIZE. And so, for density, we want to BIN ALL SCREENS
    to the SAME NUMBER OF BINS across screens. Because each bin spans THE SAME VISUAL ANGLE.
    This function calculates fixation density based on a SET NUMBER OF BINS (num_of_bins_x * num_of_bins_y), which
    is expected to be the same across all subjects and screens.

    :param num_of_bins_x: Number of bins in the X axis (WIDTH) of the screen
    :param num_of_bins_y: Number of bins in the Y axis (HEIGHT) of the screen
    :param screen_dims: screen dims are described as [W, H]
    :param gaze_x: X coordinates of gaze
    :param gaze_y: Y coordinates of gaze
    :return: fixation density array
    """
    screen_rows = minimal_dims_va[1]  # the screen HEIGHT is like "rows" in dataframe
    screen_cols = minimal_dims_va[0]  # the screen WIDTH is like "columns"

    screen_rows_real = params["ScreenResolution"][1] * params["DegreesPerPix"]
    screen_cols_real = params["ScreenResolution"][0] * params["DegreesPerPix"]
    print(f"screen rows(va): {screen_rows}, screen cols(va): {screen_cols}")
    print(f"screen rows real(va): {screen_rows_real}, screen real cols(va): {screen_cols_real}")

    """
    Initialize the fixation density matrix: the bins' order is starting from the TOP LEFT and fills up accordingly!
    NOTE that TL -> BR is also the order of things in Eyelink data we parsed in ET_data_extraction, so this is ok.
    Gaze coordinates (X, Y) in Eyelink are such that (0, 0) is the TOP LEFT corner of the screen!!!
    This means that when gaze goes DOWN --> Y coordinate goes UP! 
    Source: EL1000 User manual 1.5 chapter 4.4.2.3 GAZE
    http://sr-research.jp/support/EyeLink%201000%20User%20Manual%201.5.0.pdf
    """
    # fix_density = np.zeros((int(np.ceil(screen_rows / scale)), int(np.ceil(screen_cols / scale))))
    fix_density = np.zeros((num_of_bins_y, num_of_bins_x))
    scale_x = screen_cols / num_of_bins_x
    scale_y = screen_rows / num_of_bins_y
    x_start = (screen_cols_real - screen_cols) / 2
    y_start = (screen_rows_real - screen_rows) / 2
    # loop through the bins
    L = len(gaze_x)
    for i in range(0, fix_density.shape[1]):  # go over COLUMNS (screen WIDTH)
        for j in range(0, fix_density.shape[0]):  # go over ROWS (screen HEIGHT)
            if L == 0:  # no samples at all
                fix_density[j, i] = 0
            else:  # for each bin, we ask how many points fall in this bin
                bin_sum = np.sum(((gaze_x >= scale_x * i + x_start) & (gaze_x <= scale_x * (i + 1) + x_start)) &
                                 ((gaze_y >= scale_y * j + y_start) & (gaze_y <= scale_y * (j + 1) + y_start)))  # how many times did the gaze points hit this bin
                fix_density[j, i] = bin_sum / L  # normalize the sum
    return fix_density


def fixation_denisity_worker(df_samples, time_window, eye, params, minimal_dims_va):
    """
    Calculates the fixation density across all trials in trial list, by using the samples in df_samples that belong
    to this trial are within the time window, and the dims in screen dims per eye
    :return: 1 df per eye of fixation density
    """
    # take only REAL fixations in the RELEVANT time window
    real_fixations = df_samples.loc[(df_samples[time_window] != -1) & (df_samples[DataParser.REAL_FIX] == True), :]
    sample_count = real_fixations.shape[0]
    if sample_count != 0:
        # LX, LY here are in PIXELS!!!
        x = np.array(real_fixations[f"{eye}X"]) * params['DegreesPerPix']
        y = np.array(real_fixations[f"{eye}Y"]) * params['DegreesPerPix']
        fix_density = calculate_fixation_density_va(FIXATION_DENSITY_XBINS, FIXATION_DENSITY_YBINS, x, y,
                                                    minimal_dims_va,
                                                    params)
    else:
        fix_density = None
    return fix_density, sample_count


def fix_hist_mod(subs_list, modality, time, save_path, minimal_dims_va, max_val=0, phase_name="", plot=False,
                 square=False, im_name="game_orange.png", in_va=False, filter_fix=True, replay=False, task_relevant_only=False, task_irrelevant_only=False, load=False,
                 special_time=False, special_time_start=0, special_time_end=0):
    """
    This is a WEIGHTED average across all subjects. Meaning, from each subject, we take the number of samples where
    a fixation occurred within a specific window. Then, all of these samples' locations are binned and presented
    across ALL subjects.

    As long as the screen resolution is dividable by the bins we chose, this should run.
    :param subs_list:
    :param modality:
    :param save_path:
    :return:
    """

    if not replay:
        game_name = "GAME"
        save_name = f"fix_dens_{time}_GAME_{modality}_{task_relevant_only}_{task_irrelevant_only}_{phase_name}_{special_time}_{special_time_start}_{special_time_end}.npy"
    else:
        game_name = "REPLAY"
        save_name = f"fix_dens_{time}_REPLAY_{modality}_{task_relevant_only}_{task_irrelevant_only}_{phase_name}_{special_time}_{special_time_start}_{special_time_end}.npy"
    samples_counter = 0  # count how many samples are in that window
    density_array = None
    sub_per_lab = dict()

    if not load:
        for sub_data_path in subs_list:
            fl = open(sub_data_path, 'rb')
            sub_data = pickle.load(fl)
            fl.close()
            if phase_name == "all":
                if sub_data[PARAMS][SUBJECT_NAME] in CHOSEN_10:
                    continue
            else:
                if sub_data[PARAMS][SUBJECT_NAME] not in CHOSEN_10:
                    continue
            samples = sub_data[ET_DATA_DICT][DataParser.DF_SAMPLES]
            screen_dims = sub_data[PARAMS]['ScreenResolution']
            if sub_data[PARAMS][SUBJECT_NAME][:2] not in sub_per_lab:
                sub_per_lab[sub_data[PARAMS][SUBJECT_NAME][:2]] = sub_data[PARAMS]
            relevant_trials = sub_data[TRIAL_INFO]
            # Take GAME trials ONLY
            relevant_trials = relevant_trials.loc[(relevant_trials[DataParser.IS_LEVEL_ELIMINATED] == False) & (relevant_trials[DataParser.REPLAY] == replay) &
                                                  (relevant_trials[ET_data_extraction.IS_PROBED] == True), :]
            if task_relevant_only:
                relevant_trials = relevant_trials.loc[~relevant_trials[QualityChecker.STIM_TYPE].str.contains("None")]
            if task_irrelevant_only:
                relevant_trials = relevant_trials.loc[relevant_trials[QualityChecker.STIM_TYPE].str.contains("None")]
            samples = samples.loc[samples[time].isin(list(relevant_trials[TRIAL_NUMBER])), :]

            if special_time:
                samples = calculate_sample_time_in_trial(relevant_trials, samples, [])
                samples = samples[(samples[TIME_IN_EPOCH] >= special_time_start) & (samples[TIME_IN_EPOCH] <= special_time_end)]

            # then, calculate the fixation density bins PER SUBJECT with fixation_denisity_worker
            sub_fix_density, sub_fix_count = fixation_denisity_worker(samples, time, sub_data[PARAMS][DataParser.EYE],
                                                                      sub_data[PARAMS], minimal_dims_va)
            if sub_fix_density is not None:
                if density_array is None:
                    density_array = sub_fix_density * sub_fix_count
                else:
                    density_array += sub_fix_density * sub_fix_count
            samples_counter += sub_fix_count

        density_array /= samples_counter

        with open(os.path.join(save_path, save_name), 'wb') as f:
            np.save(f, density_array)
    else:
        with open(os.path.join(save_path, save_name), 'rb') as f:
            density_array = np.load(f)

    if density_array.max() > max_val:
        max_val = density_array.max()


    # PLOT: density_array is the thing to plot - 2D histogram
    if plot:
        # plot
        plt.gcf()
        plt.figure()
        sns.reset_orig()

        ax = sns.heatmap(density_array, cmap="magma", alpha=0.75, cbar=True,
                         cbar_kws={"shrink": .82, "label": "Density"},
                         xticklabels=[], yticklabels=[],
                         annot=False, square=True, zorder=2, vmin=0, vmax=max_val, mask=density_array == 0)
        # add (0, 0) lines
        ax.axvline(density_array.shape[1] / 2, color="white", lw=1)  # vertical line
        ax.axhline(density_array.shape[0] / 2, color="white", lw=1)  # horizontal line
        # number of ticks: unify for the journal plots s.t. they all have the same number of ticks
        from matplotlib import ticker
        tick_locator = ticker.MaxNLocator(nbins=8)
        ax.collections[0].colorbar.locator = tick_locator
        ax.collections[0].colorbar.update_ticks()

        if square:
            screen_rows = minimal_dims_va[1]  # the screen HEIGHT is like "rows" in dataframe
            screen_cols = minimal_dims_va[0]  # the screen WIDTH is like "columns"

            va_per_bin_x = (screen_cols / FIXATION_DENSITY_XBINS)
            va_per_bin_y = (screen_rows / FIXATION_DENSITY_YBINS)
            print(f"va_per_bin_x: {va_per_bin_x}, va_per_bin_y: {va_per_bin_y}")

            """
            CREATE A 6 X 6 VA RECTANGLE around fixation by using patches.Rectangle 
            https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html

            The size is EXP1_STIM_VA x EXP1_STIM_VA around the screen center -- meaning, it spans (EXP1_STIM_VA/2) from
            the center in each direction
            """
            point_y = (density_array.shape[0] / 2) - ((ET_param_manager.STIM_VA / 2) / va_per_bin_y)
            point_x = (density_array.shape[1] / 2) - ((ET_param_manager.STIM_VA / 2) / va_per_bin_x)
            rect = patches.Rectangle((point_x, point_y), (ET_param_manager.STIM_VA / va_per_bin_x), (ET_param_manager.STIM_VA / va_per_bin_y),
                                     linewidth=2, edgecolor='r', fill=False, alpha=0.8, zorder=3)
            print(
                f"point_x: {point_x}, point_y:{point_y}, xlen: {(ET_param_manager.STIM_VA / va_per_bin_x)}, ylen: {(ET_param_manager.STIM_VA / va_per_bin_y)}")
            ax.add_patch(rect)

        if not (im_name is None):  # background image
            im_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), im_name)
            arr_img = plt.imread(im_path, format='jpg')
            ax.imshow(arr_img, aspect=ax.get_aspect(), extent=ax.get_xlim() + ax.get_ylim(), zorder=1, alpha=0.95)

        plt.title(f"Fixation Density Plot: {time} - {modality}", fontsize=TITLE_SIZE, pad=LABEL_PAD + 5)

        # save plot
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(15, 12)
        if square:
            plt.savefig(os.path.join(save_path, f"fix_dens_{time}_{modality}_{task_relevant_only}_{task_irrelevant_only}_{special_time}_{special_time_start}_{special_time_end}_{phase_name}_square.png"), dpi=1000)
            plt.savefig(os.path.join(save_path, f"fix_dens_{time}_{modality}_{task_relevant_only}_{task_irrelevant_only}_{special_time}_{special_time_start}_{special_time_end}_{phase_name}_square.svg"), format="svg",
                        dpi=1000)
        else:
            plt.savefig(os.path.join(save_path, f"fix_dens_{time}_{game_name}_{modality}_{task_relevant_only}_{task_irrelevant_only}_{special_time}_{special_time_start}_{special_time_end}_{phase_name}.png"), dpi=1000)
            plt.savefig(os.path.join(save_path, f"fix_dens_{time}_{game_name}_{modality}_{task_relevant_only}_{task_irrelevant_only}_{special_time}_{special_time_start}_{special_time_end}_{phase_name}.svg"), format="svg", dpi=1000)

        # save memory
        del figure
        plt.close()
        gc.collect()

    return max_val


def get_minimal_dims_va(subs_list):
    min_va_col = 100000
    min_va_row = 100000
    for sub_data_path in subs_list:
        fl = open(sub_data_path, 'rb')
        sub_data = pickle.load(fl)
        fl.close()

        screen_dims = sub_data[PARAMS]['ScreenResolution']
        screen_rows = screen_dims[1] * sub_data[PARAMS]['DegreesPerPix']  # the screen HEIGHT is like "rows" in dataframe
        screen_cols = screen_dims[0] * sub_data[PARAMS]['DegreesPerPix']  # the screen WIDTH is like "columns"

        if min_va_col > screen_cols:
            min_va_col = screen_cols

        if min_va_row > screen_rows:
            min_va_row = screen_rows

    print(f"mins are col:{min_va_col}, row: {min_va_row}")
    return (min_va_col, min_va_row)


def fixation_plots(subs_dict, save_path, phase_name, is_tau=False, load=False, special_time=False, special_time_start=0, special_time_end=0):
    print("--- Fixation spread (2D histogram) per window ---")
    if ET_qc_manager.FMRI in subs_dict:
        minimal_dims_va = min(get_minimal_dims_va(subs_dict[ET_qc_manager.FMRI]), get_minimal_dims_va(subs_dict[ET_qc_manager.MEG]))
    else:
        minimal_dims_va = get_minimal_dims_va(subs_dict[ET_qc_manager.MEG])

    if load == False:
        n_repeats = 1
    else:
        n_repeats = 2

    max_val = 0
    for i in range(n_repeats):
        # Use same max across all conditions
        for time in [DataParser.TRIAL]: #[DataParser.STIM_DUR, DataParser.PRE_STIM_DUR]:  # DataParser.TRIAL,
            print(time)
            if time == DataParser.PRE_STIM_DUR:
                img = "game_orange.png"
                #img = "seattle_stimOn_orange_cropped.png"
            else:
                #img = "seattle_stimOn_orange_cropped.png"
                img = "game_orange.png"
                #img = "game_sync.png"
            for modality in subs_dict.keys():
                print(modality)
                #max_val = fix_hist_mod(subs_dict[modality], modality, time, save_path, minimal_dims_va, max_val, "all",
                #                       plot=True, square=False, im_name=img, task_relevant_only=True, load=load)
                #max_val = fix_hist_mod(subs_dict[modality], modality, time, save_path, minimal_dims_va, max_val, "all",
                #                       plot=True, square=False, im_name=img, task_irrelevant_only=True, load=load)
                max_val = fix_hist_mod(subs_dict[modality], modality, time, save_path, minimal_dims_va, max_val, "all",
                                       plot=True, square=False, im_name=img, load=load, special_time=special_time,
                                       special_time_start=special_time_start, special_time_end=special_time_end)
                max_val = fix_hist_mod(subs_dict[modality], modality, time, save_path, minimal_dims_va, max_val, "all",
                                       plot=True, square=False, im_name=img, replay=True, load=load, special_time=special_time,
                                       special_time_start=special_time_start, special_time_end=special_time_end)
                if not is_tau and False:
                    max_val = fix_hist_mod(subs_dict[modality], modality, time, save_path, minimal_dims_va, max_val, "chosen10",
                                           plot=True, square=False, im_name=img, load=load)
                    max_val = fix_hist_mod(subs_dict[modality], modality, time, save_path, minimal_dims_va, max_val,
                                           "chosen10", plot=True, square=False, im_name=img, task_relevant_only=True, load=load)
                    max_val = fix_hist_mod(subs_dict[modality], modality, time, save_path, minimal_dims_va, max_val,
                                           "chosen10", plot=True, square=False, im_name=img, task_irrelevant_only=True, load=load)
    return


def et_multivariant_model(subs_dict, save_path, load=False):
    df_all_sub_list = list()

    if not load:
        for modality in subs_dict.keys():
            for sub_data_path in subs_dict[modality]:
                print(sub_data_path)
                fl = open(sub_data_path, 'rb')
                sub_data = pickle.load(fl)
                fl.close()
                sub_name = sub_data[PARAMS][SUBJECT_NAME]

                for is_replay in [False, True]:
                    relevant_trials = sub_data[TRIAL_INFO]
                    relevant_trials = relevant_trials.loc[(relevant_trials[DataParser.IS_LEVEL_ELIMINATED] == False) & (relevant_trials[DataParser.REPLAY] == is_replay) & (relevant_trials[ET_data_extraction.IS_PROBED] == True), :]
                    if relevant_trials.empty:
                        continue

                    for time in [DataParser.STIM_DUR, DataParser.PRE_STIM_DUR, DataParser.TRIAL]:
                        if time != DataParser.TRIAL:
                            relevant_trials2 = relevant_trials[(relevant_trials[DataParser.VIS] == "TruePositive") | (relevant_trials[DataParser.VIS] == "FalseNegative")]
                        else:
                            relevant_trials2 = relevant_trials

                        relevant_info = relevant_trials2.loc[:, ["trialNumber", "stimulusType", "Visibility", "IsReplay", f"Fixation{time}IsInFixationArea", f"{time}bothNumSaccs", f"{time}NumBlinks"]]
                        relevant_info["time"] = time
                        relevant_info["sub"] = sub_name
                        df_all_sub_list.append(relevant_info)

        total_df = pd.concat(df_all_sub_list)
        total_df.to_csv(os.path.join(save_path, f"gaze_params_all.csv"), index=False)
    else:
        total_df = pd.read_csv(os.path.join(save_path, f"gaze_params_all.csv"))

    for time in [DataParser.STIM_DUR, DataParser.PRE_STIM_DUR, DataParser.TRIAL]:
        csv_to_save = total_df[total_df["time"] == time]
        if time != DataParser.TRIAL:
            csv_to_save = csv_to_save[csv_to_save["IsReplay"] == False]
        csv_to_save = csv_to_save[["sub", "trialNumber", "stimulusType", "Visibility", "IsReplay", f"Fixation{time}IsInFixationArea",
                                   f"{time}bothNumSaccs", f"{time}NumBlinks"]]
        csv_to_save.rename(columns={f"{time}bothNumSaccs": "NumSaccs", f"{time}NumBlinks": "NumBlinks", f"Fixation{time}IsInFixationArea": "IsInFixationArea"}, inplace=True)
        csv_to_save["NumSaccs"] = csv_to_save["NumSaccs"].fillna(value=0)

        if time != DataParser.TRIAL:
            dep_variable = "Visibility"
        else:
            dep_variable = "IsReplay"
        csv_means = csv_to_save.groupby(["sub", dep_variable]).mean().reset_index()
        all_means = csv_means.groupby([dep_variable]).mean().reset_index()
        all_stds = csv_means.groupby([dep_variable]).std().reset_index()
        for col in ["NumSaccs", "NumBlinks", "IsInFixationArea"]:
            all_means[f"{col}_std"] = all_stds[col]
        if time != DataParser.TRIAL:
            csv_to_save.to_csv(os.path.join(save_path, f"et_game_params_per_vis_{time}.csv"), index=False)
            all_means.to_csv(os.path.join(save_path, f"et_game_params_per_vis_{time}_descriptives.csv"), index=False)
        else:
            csv_to_save.to_csv(os.path.join(save_path, f"et_params_per_cond_{time}.csv"), index=False)
            all_means.to_csv(os.path.join(save_path, f"et_params_per_cond_{time}_descriptives.csv"), index=False)

    return


def calc_fixation_proportion(subs_dict, save_path):
    df_all_sub_list = list()
    df_all_sub_list2 = list()

    for modality in subs_dict.keys():
        for sub_data_path in subs_dict[modality]:
            print(sub_data_path)
            fl = open(sub_data_path, 'rb')
            sub_data = pickle.load(fl)
            fl.close()
            analyzed_eye = sub_data[PARAMS][DataParser.EYE]
            sub_name = sub_data[PARAMS][SUBJECT_NAME]

            for is_replay in [True, False]:
                relevant_trials = sub_data[TRIAL_INFO]
                relevant_trials = relevant_trials.loc[(relevant_trials[DataParser.IS_LEVEL_ELIMINATED] == False) & (relevant_trials[DataParser.REPLAY] == is_replay) & (relevant_trials[ET_data_extraction.IS_PROBED] == True), :]
                if relevant_trials.empty:
                    continue
                trial_samples = sub_data[ET_DATA_DICT][DataParser.DF_SAMPLES]

                trial_samples = trial_samples.loc[trial_samples[DataParser.TRIAL].isin(list(relevant_trials[TRIAL_NUMBER])), :]

                trial_samples = trial_samples[trial_samples[DataParser.REAL_FIX] == True]

                fix_area_small = REF_ANGLE_RADIUS_SMALL / sub_data[PARAMS]['DegreesPerPix']
                fix_area_big = REF_ANGLE_RADIUS_BIG / sub_data[PARAMS]['DegreesPerPix']

                trial_samples.loc[:, f"IsInFixationArea_1.5"] = trial_samples[f"{analyzed_eye}CenterDistPixels"] <= fix_area_small
                trial_samples.loc[:, f"IsInFixationArea_3"] = trial_samples[f"{analyzed_eye}CenterDistPixels"] <= fix_area_big
                if trial_samples.empty:
                    continue
                for dur in [DataParser.TRIAL, DataParser.STIM_DUR, DataParser.PRE_STIM_DUR]:
                    trial_means = trial_samples.loc[trial_samples[dur] != -1, :]
                    trial_means = trial_means.groupby([dur]).mean(numeric_only=True).reset_index()
                    sub_df = pd.DataFrame([trial_means.mean()])
                    sub_df = sub_df[["IsInFixationArea_3", "IsInFixationArea_1.5"]]
                    sub_df2 = trial_means[["IsInFixationArea_3", "IsInFixationArea_1.5", dur]]
                    sub_df[SUBJECT] = sub_name
                    sub_df[QualityChecker.TIME_WINDOW_NAME] = dur
                    sub_df[DataParser.REPLAY] = is_replay
                    sub_df[MODALITY] = modality
                    sub_df[LAB] = sub_name[:2]
                    sub_df2[SUBJECT] = sub_name
                    sub_df2[QualityChecker.TIME_WINDOW_NAME] = dur
                    sub_df2[DataParser.REPLAY] = is_replay
                    sub_df2[MODALITY] = modality
                    sub_df2[LAB] = sub_name[:2]
                    df_all_sub_list.append(sub_df)
                    df_all_sub_list2.append(sub_df2)

    all_subs_df = pd.concat(df_all_sub_list)
    all_subs_df_no_means = pd.concat(df_all_sub_list2)

    no_chosen_10_all_durs = all_subs_df_no_means[~all_subs_df_no_means[SUBJECT].isin(CHOSEN_10)]
    # things for TAU
    for time in [DataParser.STIM_DUR, DataParser.TRIAL]:
        sub_total_means = no_chosen_10_all_durs.groupby([QualityChecker.TIME_WINDOW_NAME, SUBJECT]).mean(numeric_only=True)
        only_stim_dur = no_chosen_10_all_durs[no_chosen_10_all_durs[QualityChecker.TIME_WINDOW_NAME] == time]
        sub_means = only_stim_dur.groupby([QualityChecker.TIME_WINDOW_NAME, DataParser.REPLAY, SUBJECT, time]).mean(numeric_only=True).reset_index()
        sub_means.to_csv(os.path.join(save_path, f"et_fixation_stability_perTrial_{time}.csv"), index=False)

    no_chosen_10 = all_subs_df[~all_subs_df[SUBJECT].isin(CHOSEN_10)]
    sub_total_means = no_chosen_10_all_durs.groupby([QualityChecker.TIME_WINDOW_NAME, SUBJECT]).mean(numeric_only=True)
    trial_means = no_chosen_10.groupby([QualityChecker.TIME_WINDOW_NAME, DataParser.REPLAY]).mean(numeric_only=True).reset_index()
    trial_stds = no_chosen_10.groupby([QualityChecker.TIME_WINDOW_NAME, DataParser.REPLAY]).std(numeric_only=True).reset_index()
    trial_means["IsInFixationArea_3_std"] = trial_stds["IsInFixationArea_3"]
    trial_means["IsInFixationArea_1.5_std"] = trial_stds["IsInFixationArea_1.5"]
    total_means = sub_total_means.groupby([QualityChecker.TIME_WINDOW_NAME]).mean().reset_index()
    total_std = sub_total_means.groupby([QualityChecker.TIME_WINDOW_NAME]).std().reset_index()
    total_means["IsInFixationArea_3_std"] = total_std["IsInFixationArea_3"]
    total_means["IsInFixationArea_1.5_std"] = total_std["IsInFixationArea_1.5"]
    trial_means = pd.concat([total_means, trial_means])
    trial_means.to_csv(os.path.join(save_path, f"fixation_stability_all_mean.csv"), index=False)

    sub_total_means = no_chosen_10_all_durs.groupby([QualityChecker.TIME_WINDOW_NAME, SUBJECT, MODALITY]).mean(numeric_only=True)
    trial_means = no_chosen_10.groupby([QualityChecker.TIME_WINDOW_NAME, DataParser.REPLAY, MODALITY]).mean(
        numeric_only=True).reset_index()
    trial_stds = no_chosen_10.groupby([QualityChecker.TIME_WINDOW_NAME, DataParser.REPLAY, MODALITY]).std(
        numeric_only=True).reset_index()
    trial_means["IsInFixationArea_3_std"] = trial_stds["IsInFixationArea_3"]
    trial_means["IsInFixationArea_1.5_std"] = trial_stds["IsInFixationArea_1.5"]
    total_means = sub_total_means.groupby([QualityChecker.TIME_WINDOW_NAME, MODALITY]).mean().reset_index()
    total_std = sub_total_means.groupby([QualityChecker.TIME_WINDOW_NAME, MODALITY]).std().reset_index()
    total_means["IsInFixationArea_3_std"] = total_std["IsInFixationArea_3"]
    total_means["IsInFixationArea_1.5_std"] = total_std["IsInFixationArea_1.5"]
    trial_means = pd.concat([total_means, trial_means])
    trial_means.to_csv(os.path.join(save_path, f"fixation_stability_all_mean_per_mod.csv"), index=False)

    sub_total_means = no_chosen_10_all_durs.groupby([QualityChecker.TIME_WINDOW_NAME, SUBJECT, MODALITY, LAB]).mean(numeric_only=True)
    trial_means = no_chosen_10.groupby([QualityChecker.TIME_WINDOW_NAME, DataParser.REPLAY, MODALITY, LAB]).mean(
        numeric_only=True).reset_index()
    trial_stds = no_chosen_10.groupby([QualityChecker.TIME_WINDOW_NAME, DataParser.REPLAY, MODALITY, LAB]).std(
        numeric_only=True).reset_index()
    trial_means["IsInFixationArea_3_std"] = trial_stds["IsInFixationArea_3"]
    trial_means["IsInFixationArea_1.5_std"] = trial_stds["IsInFixationArea_1.5"]
    total_means = sub_total_means.groupby([QualityChecker.TIME_WINDOW_NAME, MODALITY, LAB]).mean().reset_index()
    total_std = sub_total_means.groupby([QualityChecker.TIME_WINDOW_NAME, MODALITY, LAB]).std().reset_index()
    total_means["IsInFixationArea_3_std"] = total_std["IsInFixationArea_3"]
    total_means["IsInFixationArea_1.5_std"] = total_std["IsInFixationArea_1.5"]
    trial_means = pd.concat([total_means, trial_means])
    trial_means.to_csv(os.path.join(save_path, f"fixation_stability_all_mean_per_lab.csv"), index=False)

    trial_means = no_chosen_10.groupby([QualityChecker.TIME_WINDOW_NAME, DataParser.REPLAY]).mean(numeric_only=True).reset_index()
    trial_stds = no_chosen_10.groupby([QualityChecker.TIME_WINDOW_NAME, DataParser.REPLAY]).std(numeric_only=True).reset_index()
    trial_means["IsInFixationArea_3_std"] = trial_stds["IsInFixationArea_3"]
    trial_means["IsInFixationArea_1.5_std"] = trial_stds["IsInFixationArea_1.5"]
    total_means = sub_total_means.groupby([QualityChecker.TIME_WINDOW_NAME]).mean().reset_index()
    total_std = sub_total_means.groupby([QualityChecker.TIME_WINDOW_NAME]).std().reset_index()
    total_means["IsInFixationArea_3_std"] = total_std["IsInFixationArea_3"]
    total_means["IsInFixationArea_1.5_std"] = total_std["IsInFixationArea_1.5"]
    trial_means = pd.concat([total_means, trial_means])
    trial_means.to_csv(os.path.join(save_path, f"fixation_stability_all_mean.csv"), index=False)
    #trial_stds.to_csv(os.path.join(save_path, f"fixation_stability_all_std.csv"), index=False)

    chosen_10 = all_subs_df[all_subs_df[SUBJECT].isin(CHOSEN_10)]
    trial_means = chosen_10.groupby([QualityChecker.TIME_WINDOW_NAME, DataParser.REPLAY]).mean(numeric_only=True).reset_index()
    trial_stds = chosen_10.groupby([QualityChecker.TIME_WINDOW_NAME, DataParser.REPLAY]).std(numeric_only=True).reset_index()
    trial_means.to_csv(os.path.join(save_path, f"fixation_stability_chosen10_mean.csv"), index=False)
    trial_stds.to_csv(os.path.join(save_path, f"fixation_stability_chosen10_std.csv"), index=False)
    return


def probe_time_delta_analysis(subs_dict, save_path):
    all_subs_list = []
    for mod in subs_dict:
        if mod != ET_qc_manager.FMRI:
            continue
        for sub_data_path in subs_dict[mod]:
            fl = open(sub_data_path, 'rb')
            sub_data = pickle.load(fl)
            fl.close()
            sub_name = sub_data[PARAMS][SUBJECT_NAME]
            time_csv = [f for f in os.listdir(sub_data_path[:sub_data_path.rfind('/')]) if "probe_deltas" in f][0]
            time_df = pd.read_csv(os.path.join(sub_data_path[:sub_data_path.rfind('/')], time_csv))
            time_df["sub"] = sub_name
            time_df[MODALITY] = mod
            time_df[LAB] = sub_name[:2]
            all_subs_list.append(time_df)
    all_subs_df = pd.concat(all_subs_list)
    all_subs_df = all_subs_df[all_subs_df["WorldID"].isin(["World 1", "World 2", "World 3", "World 4"])]
    zero_thingy = all_subs_df[all_subs_df["time_delta"] == 0]
    zero_subs = list(zero_thingy["sub"])
    for mod in subs_dict:
        if mod != ET_qc_manager.FMRI:
            continue
        for sub_data_path in subs_dict[mod]:
            if len([x for x in zero_subs if x in sub_data_path]) > 0:
                fl = open(sub_data_path, 'rb')
                sub_data = pickle.load(fl)
                fl.close()
                sub_name = sub_data[PARAMS][SUBJECT_NAME]
                relevant_levels = sub_data["trial_info"][sub_data["trial_info"]["probeOnset"].isin(zero_thingy["time"])]
                if not relevant_levels.empty:
                    zero_thingy.loc[zero_thingy["sub"] == sub_name,"real_level"] = list(relevant_levels["currentLevelID"])
    x = 5

    return


def time_diff_analysis(subs_dict, save_path):
    all_subs_list = []
    for mod in subs_dict:
        for sub_data_path in subs_dict[mod]:
            fl = open(sub_data_path, 'rb')
            sub_data = pickle.load(fl)
            fl.close()
            sub_name = sub_data[PARAMS][SUBJECT_NAME]
            time_csv = [f for f in os.listdir(sub_data_path[:sub_data_path.rfind('/')]) if "timings_report" in f][0]
            time_df = pd.read_csv(os.path.join(sub_data_path[:sub_data_path.rfind('/')], time_csv))
            time_df["sub"] = sub_name
            time_df[MODALITY] = mod
            time_df[LAB] = sub_name[:2]
            time_df["Unity_diff"] = time_df['Unity_times'].diff()
            time_df["Eyelink_real_diff"] = time_df['Eyelink_times'].diff()
            time_df["diff_diff"] = time_df["Eyelink_real_diff"] - time_df["Unity_diff"]
            time_df.reset_index(inplace=True)
            time_df["diff_cumsum"] = time_df.cumsum()["diff_diff"]
            all_subs_list.append(time_df)

    all_subs_df = pd.concat(all_subs_list)
    per_sub_sum = all_subs_df.groupby(["sub", LAB]).sum().reset_index()
    per_sub_sum = per_sub_sum[(per_sub_sum["sub"] != "SB036") & (per_sub_sum["sub"] != "SD188")]
    all_subs_mean = per_sub_sum.mean()
    all_subs_df_means = all_subs_df.groupby(["sub"]).mean()
    all_subs_df_stds = all_subs_df.groupby(["sub"]).std()

    all_subs_df = all_subs_df[(all_subs_df["sub"] != "SB036") & (all_subs_df["sub"] != "SD188")]
    cumsum_mean = all_subs_df.groupby(["index", LAB]).mean().reset_index()
    for lab in cumsum_mean[LAB].unique():
        rel_df = cumsum_mean.loc[cumsum_mean[LAB] == lab, :]
        plt.plot(rel_df["index"], rel_df["diff_cumsum"], label=lab)
    plt.legend()
    plt.show()
    return


def saccade_calc_bins(trial_saccs_dur, bin_size, sub_data, additional_col_name, additional_col_value,
                      modality):
    total_sub_len = trial_saccs_dur.shape[0]
    bin_dfs = pd.DataFrame()  # this contains the amt of saccades in this bin per bin
    bin_dfs[RANGE_START_RAD] = [deg * (math.pi / 180) for deg in range(-180, 180, bin_size)]
    bin_dfs["range_end_rads"] = [(deg + bin_size) * (math.pi / 180) for deg in range(-180, 180, bin_size)]
    bin_count_list = []
    # iterate over degree BINS (notice the step in the for) and find all saccades which are within
    # that range
    for deg in range(-180, 180, bin_size):
        range_start_rads = deg * (math.pi / 180)
        range_end_rads = (deg + bin_size) * (math.pi / 180)
        relevant = trial_saccs_dur.loc[(range_start_rads <= trial_saccs_dur[DataParser.SACC_DIRECTION_RAD]) &
                                       ((range_end_rads > trial_saccs_dur[DataParser.SACC_DIRECTION_RAD])), :]
        # add the number of saccades falling within that bin divided by the total number of saccades in that (sub)condition
        if total_sub_len == 0:
            bin_count_list.append(0)
        else:
            bin_count_list.append(relevant.shape[0])  # proportion
    bin_dfs["bin_count"] = bin_count_list
    if total_sub_len != 0:
        bin_dfs[BIN_PROPORTION] = bin_dfs["bin_count"] / total_sub_len
    else:
        bin_dfs[BIN_PROPORTION] = 0
    bin_dfs[SUBJECT] = sub_data[PARAMS]['SubjectName']
    bin_dfs[MODALITY] = modality
    bin_dfs[LAB] = sub_data[PARAMS]['SubjectName'][:2]
    bin_dfs[additional_col_name] = additional_col_value
    return bin_dfs


def polar_plotter(data_df, hue_col, hue_col_name, hue_names_map, hue_color_map, title_name,
                  save_path, save_name, r_range=[0, 0.2, 0.05]):
    """
    Plot a polar DISTRIBUTION plot (radians).
    :return:
    """

    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')

    """
    Theta should be range_start_rad + range_end_rads / 2 --> This is the MIDDLE between bins in RADIANS
    (we chose the middle between the bin start and end)
    r should be bin_proportion
    """
    for dur in list(data_df[hue_col].unique()):
        dur_df = data_df.loc[data_df[hue_col] == dur, :]
        range_starts_rad = list(dur_df["range_starts_rad"].values)
        range_end_rads = list(dur_df["range_end_rads"].values)
        # matplotlib's polarplot theta is in RADIANS, which is just like our data
        theta = [(range_starts_rad[ind] + range_end_rads[ind]) / 2 for ind in range(len(range_starts_rad))]
        r = list(dur_df["bin_proportion"].values)
        ax.plot(theta, r, c=hue_color_map[dur], linewidth=6, label=hue_names_map[dur])
        ax.grid(linewidth=4)

    ax.set_rmax(r_range[1])
    ax.set_rticks(np.arange(r_range[0], r_range[1], r_range[2]))
    ax.tick_params(axis='both', which='major', labelsize=18)

    plt.title(f"{title_name}", fontsize=TITLE_SIZE, pad=LABEL_PAD + 5)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title=hue_col_name)

    # save
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 12)
    plt.savefig(os.path.join(save_path, f"{save_name}.png"), dpi=1000)
    plt.savefig(os.path.join(save_path, f"{save_name}.svg"), format="svg", dpi=1000)
    # free memory
    del figure
    gc.collect()
    return


def saccade_ciruclar_relevance_plot(subs_dict, save_path, chosen_10=False, load=False):
    """
    This plots a circular (polar) plot of the average saccade direction.
    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    We take all of the saccades of a subject that are in a trial of a certain category, then bin them, and calculate
    MEAN between BINS
    """
    BINS = 30
    bin_size = int(
        2 * 180 / BINS)  # MUST be an integer; if you want to change the number of bins note this number as well

    if chosen_10:
        f_name = "chosen10"
    else:
        f_name = "all"

    if not load:
        all_dfs_list = []
        for modality in subs_dict.keys():
            only_mod_dfs_list = []
            for sub_data_path in subs_dict[modality]:
                fl = open(sub_data_path, 'rb')
                sub_data = pickle.load(fl)
                fl.close()
                trial_saccs = sub_data[ET_DATA_DICT][DataParser.DF_SACC_EK]
                trial_saccs = trial_saccs.loc[trial_saccs[DataParser.HERSHMAN_PAD] == False, :]  # only REAL saccades
                relevant_trials = sub_data[TRIAL_INFO]
                relevant_trials = relevant_trials.loc[relevant_trials[DataParser.IS_LEVEL_ELIMINATED] == False, :]
                all_stim_dur_dfs = []
                for replay in [True, False]:
                    relevant_trials_relevance = relevant_trials.loc[
                                                relevant_trials[DataParser.REPLAY] == replay, :]
                    trial_saccs_dur = trial_saccs.loc[trial_saccs[DataParser.TRIAL].isin(
                        list(relevant_trials_relevance[TRIAL_NUMBER])), :]
                    all_stim_dur_dfs.append(saccade_calc_bins(trial_saccs_dur, bin_size, sub_data,
                                                              DataParser.REPLAY, replay, modality))
                only_mod_dfs_list.append(pd.concat(all_stim_dur_dfs))

            # Calculated everything for all subs in modality (for the category), now calculate the MEAN
            all_mod_bins_df = pd.concat(only_mod_dfs_list)
            all_dfs_list.append(all_mod_bins_df)
        # Now, we do the mean for all mods combined
        all_bins_df = pd.concat(all_dfs_list)
        all_bins_df.to_csv(os.path.join(save_path, f"sacc_dir_polar_relevance_all.csv"), index=False)
    else:
        all_bins_df = pd.read_csv(os.path.join(save_path, f"sacc_dir_polar_relevance_all.csv"))

    if chosen_10:
        all_bins_df = all_bins_df.loc[all_bins_df[SUBJECT].isin(CHOSEN_10), :]
    mean_df = all_bins_df.groupby([RANGE_START_RAD, DataParser.REPLAY, MODALITY]).mean(numeric_only=True).reset_index()
    # PLOT DIRECTION OF SACCADE DISTRIBUTION AVERAGED *** WITHIN A STIMULUS CATEGORY ***
    for is_replay in [True, False]:
        df_cat = mean_df[mean_df[DataParser.REPLAY] == is_replay]
        polar_plotter(data_df=df_cat, hue_col=MODALITY, hue_col_name="Modality",
                      hue_names_map=MODALITY_NAME_MAP, hue_color_map=MODALITY_MAP,
                      title_name=f"Saccadde Direction Density in **{is_replay}** Trials",
                      save_path=save_path, save_name=f"sacc_dir_polar_{is_replay}_tr_{f_name}",
                      r_range=[0, 0.2, 0.05])

    # all
    mean_df = all_bins_df.groupby([RANGE_START_RAD, MODALITY]).mean(numeric_only=True).reset_index()
    # PLOT DIRECTION OF SACCADE DISTRIBUTION AVERAGED *** WITHIN A STIMULUS CATEGORY ***
    polar_plotter(data_df=mean_df, hue_col=MODALITY, hue_col_name="Modality",
                  hue_names_map=MODALITY_NAME_MAP, hue_color_map=MODALITY_MAP,
                  title_name=f"Saccadde Direction Density in ALL Trials",
                  save_path=save_path, save_name=f"sacc_dir_polar_tr_{f_name}",
                  r_range=[0, 0.2, 0.05])
    return


def saccade_ciruclar_replay_stim_plot_temp(subs_dict, save_path, time, chosen_10=False, load=False):
    """
    This plots a circular (polar) plot of the average saccade direction.
    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    We take all of the saccades of a subject that are in a trial of a certain category, then bin them, and calculate
    MEAN between BINS
    """
    BINS = 30
    bin_size = int(
        2 * 180 / BINS)  # MUST be an integer; if you want to change the number of bins note this number as well

    if chosen_10:
        f_name = "chosen10"
    else:
        f_name = "all"

    if not load:
        all_dfs_list = []
        for modality in subs_dict.keys():
            only_mod_dfs_list = []
            for sub_data_path in subs_dict[modality]:
                fl = open(sub_data_path, 'rb')
                sub_data = pickle.load(fl)
                fl.close()
                trial_saccs = sub_data[ET_DATA_DICT][DataParser.DF_SACC_EK]
                trial_saccs = trial_saccs.loc[trial_saccs[DataParser.HERSHMAN_PAD] == False, :]  # only REAL saccades
                relevant_trials = sub_data[TRIAL_INFO]
                relevant_trials.loc[:, TARGET] = relevant_trials[QualityChecker.STIM_TYPE] == relevant_trials["TargetType"]
                relevant_trials = relevant_trials.loc[(relevant_trials[DataParser.IS_LEVEL_ELIMINATED] == False) & (relevant_trials[DataParser.REPLAY] == True) &
                                                      (relevant_trials[ET_data_extraction.IS_PROBED] == True), :]
                all_stim_dur_dfs = []
                for target in [True, False]:
                    relevant_trials_relevance = relevant_trials.loc[
                                                (relevant_trials[TARGET] == target), :]
                    trial_saccs_dur = trial_saccs.loc[trial_saccs[time].isin(list(relevant_trials_relevance[TRIAL_NUMBER])), :]
                    tmp_df = saccade_calc_bins(trial_saccs_dur, bin_size, sub_data, TARGET, target, modality)
                    all_stim_dur_dfs.append(tmp_df)
                only_mod_dfs_list.append(pd.concat(all_stim_dur_dfs))

            # Calculated everything for all subs in modality (for the category), now calculate the MEAN
            all_mod_bins_df = pd.concat(only_mod_dfs_list)
            all_dfs_list.append(all_mod_bins_df)
        # Now, we do the mean for all mods combined
        all_bins_df = pd.concat(all_dfs_list)
        all_bins_df.to_csv(os.path.join(save_path, f"sacc_dir_polar_vis_{time}_replay_all.csv"), index=False)
    else:
        all_bins_df = pd.read_csv(os.path.join(save_path, f"sacc_dir_polar_vis_{time}_replay_all.csv"))

    if chosen_10:
        all_bins_df = all_bins_df.loc[all_bins_df[SUBJECT].isin(CHOSEN_10), :]

    mean_df = all_bins_df.groupby([RANGE_START_RAD, TARGET, MODALITY]).mean(numeric_only=True).reset_index()
    # PLOT DIRECTION OF SACCADE DISTRIBUTION AVERAGED *** WITHIN A STIMULUS CATEGORY ***
    for modality in [ET_qc_manager.FMRI, ET_qc_manager.MEG]:
        df_cat = mean_df[mean_df[MODALITY] == modality]
        polar_plotter(data_df=df_cat, hue_col=TARGET, hue_col_name=TARGET,
                      hue_names_map=TARGET_NAME_MAP,
                      hue_color_map=TARGET_MAP,
                      title_name=f"Saccadde Direction Density in {modality}",
                      save_path=save_path, save_name=f"sacc_dir_polar_vis_{time}_{modality}_{f_name}_replay",
                      r_range=[0, 0.2, 0.05])
    return


def saccade_ciruclar_game_stim_plot_temp(subs_dict, save_path, time, chosen_10=False, load=False):
    """
    This plots a circular (polar) plot of the average saccade direction.
    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    We take all of the saccades of a subject that are in a trial of a certain category, then bin them, and calculate
    MEAN between BINS
    """
    BINS = 30
    bin_size = int(
        2 * 180 / BINS)  # MUST be an integer; if you want to change the number of bins note this number as well

    if chosen_10:
        f_name = "chosen10"
    else:
        f_name = "all"

    if not load:
        all_dfs_list = []
        for modality in subs_dict.keys():
            only_mod_dfs_list = []
            for sub_data_path in subs_dict[modality]:
                fl = open(sub_data_path, 'rb')
                sub_data = pickle.load(fl)
                fl.close()
                trial_saccs = sub_data[ET_DATA_DICT][DataParser.DF_SACC_EK]
                trial_saccs = trial_saccs.loc[trial_saccs[DataParser.HERSHMAN_PAD] == False, :]  # only REAL saccades
                relevant_trials = sub_data[TRIAL_INFO]
                relevant_trials = relevant_trials.loc[(relevant_trials[DataParser.IS_LEVEL_ELIMINATED] == False) & (relevant_trials[DataParser.REPLAY] == False) &
                                                      (relevant_trials[ET_data_extraction.IS_PROBED] == True), :]
                all_stim_dur_dfs = []
                for vis in ["FalseNegative", "TruePositive"]:
                    for stimtype in ["Face", "Object"]:
                        relevant_trials_relevance = relevant_trials.loc[
                                                    (relevant_trials[DataParser.VIS] == vis) & (relevant_trials[QualityChecker.STIM_TYPE] == stimtype), :]
                        trial_saccs_dur = trial_saccs.loc[trial_saccs[time].isin(
                            list(relevant_trials_relevance[TRIAL_NUMBER])), :]
                        tmp_df  = saccade_calc_bins(trial_saccs_dur, bin_size, sub_data,
                                                                  DataParser.VIS, vis, modality)
                        tmp_df.loc[:, QualityChecker.STIM_TYPE] = stimtype + "s"
                        all_stim_dur_dfs.append(tmp_df)
                only_mod_dfs_list.append(pd.concat(all_stim_dur_dfs))

            # Calculated everything for all subs in modality (for the category), now calculate the MEAN
            all_mod_bins_df = pd.concat(only_mod_dfs_list)
            all_dfs_list.append(all_mod_bins_df)
        # Now, we do the mean for all mods combined
        all_bins_df = pd.concat(all_dfs_list)
        all_bins_df.to_csv(os.path.join(save_path, f"sacc_dir_polar_vis_{time}_all.csv"), index=False)
    else:
        all_bins_df = pd.read_csv(os.path.join(save_path, f"sacc_dir_polar_vis_{time}_all.csv"))

    if chosen_10:
        all_bins_df = all_bins_df.loc[all_bins_df[SUBJECT].isin(CHOSEN_10), :]

    all_bins_df.loc[:, DataParser.VIS] = all_bins_df[DataParser.VIS] + "_" + all_bins_df[QualityChecker.STIM_TYPE]
    mean_df = all_bins_df.groupby([RANGE_START_RAD, DataParser.VIS, MODALITY]).mean(numeric_only=True).reset_index()
    # PLOT DIRECTION OF SACCADE DISTRIBUTION AVERAGED *** WITHIN A STIMULUS CATEGORY ***
    for modality in [ET_qc_manager.FMRI, ET_qc_manager.MEG]:
        df_cat = mean_df[mean_df[MODALITY] == modality]
        polar_plotter(data_df=df_cat, hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      hue_names_map=SACC_VISIBILITY_OBJECTS_NAME_MAP,
                      hue_color_map=SACC_VISIBILITY_OBJECTS_MAP,
                      title_name=f"Saccadde Direction Density in {modality}",
                      save_path=save_path, save_name=f"sacc_dir_polar_vis_{time}_{modality}_{f_name}",
                      r_range=[0, 0.2, 0.05])
    return


def wrap_to_2pi(radians):
    """
    Convert angles from [-PI, PI] to [0, 2PI].
    """
    return (radians % (2 * np.pi) + 2 * np.pi) % (2 * np.pi)


def circular_sacc_analysis_thesis(subs_dict, save_path, load=False):
    """
    This module receives saccade information across ALL subjects and saves circular statistics.
    :param all_saccades: the output of get_saccades_data - a dataframe with all of the saccades
    :param sacc_analysis_path: path to save the resulting circular statistics csvs in
    :return: A table with a row per each sub-condition of trials, then columns with stats about each sub-condition:
    number of samples in subcondition (amount of saccades in this subcondition)
    the mean and std of the saccade angle
    the p-value of Rayleigh test
    the p-value of v-test against each stimulus location (1 column per v-test result against each location)
    and, the confidence interval for the population mean value for that v-test: The reason for that is that the v-test
    must always be interpreted in combination with the CI for the population mean value: If the confidence interval
    does not include the pre-specified value, then the researcher should conclude that the data do not support their
    alternative hypothesis, but if the confidence interval is narrow and includes this value, then there is evidence to
    support it. Uncertainty arises, however, if the confidence interval includes the pre-specified value but is
    relatively wide. In this case, the conclusion must be very tentative: The data could be interpreted as being in
    accord with the alternative hypothesis, but it is also in accord with other explanations. In brief, there is a
    price to be paid for the extra power offered by the V-test: Having pre-specified one possible position of
    concentration of the data to gain extra power to test this possibility, the researcher loses the ability to consider
    any other potential departures from uniformity. In fairness, once the V-test has been applied, researchers must
    avoid any speculation about support in the data for alternatives to uniformity other than the one they pre-specify
    in their V-test. We would also emphasize that the mean value to be tested in a V-test should be decided before any
    preliminary inspection of the data; if this value is influenced in any way by inspection of the data and then the
    V-test is applied to that same data, then the type I error rate will be substantially increased.

    see: Landler, L., Ruxton, G. D., & Malkemper, E. P. (2018). Circular data in biology: advice for effectively
    implementing statistical procedures. Behavioral ecology and sociobiology, 72(8), 1-10.
    """

    if not load:
        all_saccs_list = []
        for modality in subs_dict.keys():
            for sub_data_path in subs_dict[modality]:
                print(sub_data_path)
                fl = open(sub_data_path, 'rb')
                sub_data = pickle.load(fl)
                fl.close()
                trial_saccs = sub_data[ET_DATA_DICT][DataParser.DF_SACC_EK]
                trial_saccs = trial_saccs.loc[trial_saccs[DataParser.HERSHMAN_PAD] == False, :]  # only REAL saccades
                relevant_trials = sub_data[TRIAL_INFO]
                relevant_trials = relevant_trials.loc[(relevant_trials[DataParser.IS_LEVEL_ELIMINATED] == False) &
                                                      (relevant_trials[ET_data_extraction.IS_PROBED] == True), :]
                all_trial_saccs = trial_saccs.merge(relevant_trials, left_on=DataParser.TRIAL, right_on=TRIAL_NUMBER, how='inner')
                all_trial_saccs[SUBJECT] = sub_data[PARAMS]['SubjectName']
                all_saccs_list.append(all_trial_saccs)
        all_saccs_df = pd.concat(all_saccs_list)
        all_saccs_df.to_csv(os.path.join(save_path, f"sacc_all_saccs.csv"), index=False)
    else:
        all_saccs_df = pd.read_csv(os.path.join(save_path, f"sacc_all_saccs.csv"))

    summary = list()
    stim_loc_test_names = [
        [f'vtest_pval_{l}', f'vtest_V_{l}', f'vtest_Mtest_H0_reject_{l}', f'vtest_Mtest_pop_mean_{l}',
         f'vtest_Mtest_CI_{l}'] for l in ET_param_manager.STIM_LOCS]
    stim_loc_test_names = [item for sublist in stim_loc_test_names for item in sublist]
    summary_cols = ['time window', 'phase', 'cond', 'subcond', 'samples', 'circ_mean', 'circ_std',
                    'rayleightest_test_pval'] + stim_loc_test_names
    all_saccs_df.loc[:, f"{QualityChecker.STIM_TYPE}_{DataParser.VIS}"] = all_saccs_df[DataParser.VIS] + "_" + all_saccs_df[QualityChecker.STIM_TYPE]

    # Remove saccades with both STIM_DUR and PRE_STIM_DUR markings
    all_saccs_df = all_saccs_df[(all_saccs_df[DataParser.STIM_DUR] != -1) ^ (all_saccs_df[DataParser.PRE_STIM_DUR] != -1)]
    for timewindow in [DataParser.STIM_DUR, DataParser.PRE_STIM_DUR, "Both"]:
        for replay in [False, True]:  # for each condition of the VG
            for cond in [DataParser.VIS, QualityChecker.STIM_TYPE, f"{QualityChecker.STIM_TYPE}_{DataParser.VIS}"]:
                # We have only real saccades in this dataframe, so no need to check for "is_blink"
                if timewindow != "Both":
                    relevant_saccs = all_saccs_df[(all_saccs_df[timewindow] != -1) & (all_saccs_df[DataParser.REPLAY] == replay)]
                else:
                    relevant_saccs = all_saccs_df[(all_saccs_df[DataParser.REPLAY] == replay)]
                for subcond in relevant_saccs[cond].unique():
                    relevant_saccs_subcond = relevant_saccs[relevant_saccs[cond] == subcond]
                    sacc_radians = np.array(relevant_saccs_subcond[DataParser.SACC_DIRECTION_RAD])
                    circ_mean = astropy.stats.circstats.circmean(sacc_radians)
                    circ_std = astropy.stats.circstats.circstd(sacc_radians)
                    # https://docs.astropy.org/en/stable/api/astropy.stats.circstats.rayleightest.html#astropy.stats.circstats.rayleightest
                    circ_uniformity_test_pval = astropy.stats.circstats.rayleightest(sacc_radians)
                    templist = [timewindow, replay, cond, subcond, relevant_saccs_subcond.shape[0], circ_mean, circ_std,
                                circ_uniformity_test_pval]
                    for stim_loc in ET_param_manager.STIM_LOCS:
                        loc_degrees = STIM_LOCS_ANGLES[stim_loc]
                        loc_radians = loc_degrees * (math.pi / 180)
                        """
                        There is no need to turn the [0-2PI] radians of the loc to a [-PI, PI] range - it returns the same result
                        loc_radians = wrap_to_2pi(loc_radians)
                        """
                        # V TEST
                        # https://github.com/circstat/pycircstat/blob/master/pycircstat/tests.py#L256
                        # V-TEST P-VALUE AND THE VALUE OF THE V-STATISTIC : order in result is : (pval, v-val)
                        vtest_pval, vtest_v = pycircstat.vtest(alpha=sacc_radians, mu=loc_radians)
                        # M TEST
                        # https://github.com/circstat/pycircstat/blob/master/pycircstat/tests.py#L592
                        # result here is : (H (0 if H0 canot be rejected, 1 otherwise), mean, CI)
                        try:
                            mtest_h, mtest_m, mtest_CI = pycircstat.mtest(alpha=sacc_radians, dir=loc_radians)
                            templist.extend([vtest_pval, vtest_v, mtest_h, mtest_m, mtest_CI])
                        except UserWarning:
                            print(
                                f"Could not calculate mtest for dur {timewindow}, cond {cond}, subcond {subcond}, stim_loc {stim_loc}, as it does not meet the required concentration of the data around the mean")
                    summary.append(templist)

    eye_dur_summary_df = pd.DataFrame(summary, columns=summary_cols)
    eye_dur_summary_df.to_csv(os.path.join(save_path, f"sacc_circular_stats_all.csv"), index=False)

    """
    An hktest on GAME saccades only, with two variables: 
        1. StimDur/PreStimDur saccades - so we remove saccades that appear in both (they are very sparse)
        2. Visibility - With Seen vs Unseen only
    The dependant variable is the saccade direction (in radians)
    """
    relevant_saccs = all_saccs_df[all_saccs_df[DataParser.REPLAY] == False]
    relevant_saccs = relevant_saccs[(relevant_saccs[DataParser.STIM_DUR] != -1) ^ (relevant_saccs[DataParser.PRE_STIM_DUR] != -1)]
    relevant_saccs = relevant_saccs[(relevant_saccs[DataParser.VIS] == "TruePositive") | (relevant_saccs[DataParser.VIS] == "FalseNegative")]
    relevant_saccs.loc[relevant_saccs[DataParser.STIM_DUR] != -1, "StimDurTime"] = True
    relevant_saccs.loc[relevant_saccs[DataParser.STIM_DUR] == -1, "StimDurTime"] = False
    pvals, hk_result = pycircstat.hktest(alpha=relevant_saccs[DataParser.SACC_DIRECTION_RAD], idp=relevant_saccs[DataParser.VIS],
                          idq=relevant_saccs["StimDurTime"], inter=True, fn=[DataParser.VIS, "TimeWindow"])
    hk_result.to_csv(os.path.join(save_path, f"sacc_circular_hktest.csv"), index=True)
    
    # Extract descriptives of the hk test as well. Since hktest doesnt differ between subjects, these are super-sub stats
    stats_list = []
    for factor in ["StimDurTime", DataParser.VIS]:
        #means_stats2 = relevant_saccs.groupby([factor])[DataParser.SACC_DIRECTION_RAD].apply(lambda x: scipy.stats.circmean(np.array(x), low=-np.pi, high=np.pi))

        means_stats = relevant_saccs.groupby([factor])[DataParser.SACC_DIRECTION_RAD].apply(lambda x: astropy.stats.circstats.circmean(np.array(x)))
        stds_stats = relevant_saccs.groupby([factor])[DataParser.SACC_DIRECTION_RAD].apply(lambda x: astropy.stats.circstats.circstd(np.array(x)))
        means_stats = means_stats.reset_index(name=DataParser.SACC_DIRECTION_RAD)
        stds_stats = stds_stats.reset_index(name=DataParser.SACC_DIRECTION_RAD)
        means_stats[f"{DataParser.SACC_DIRECTION_RAD}_std"] = stds_stats[DataParser.SACC_DIRECTION_RAD]
        stats_list.append(means_stats)
    all_stats = pd.concat(stats_list)
    all_stats = all_stats.loc[:, ["StimDurTime", DataParser.VIS, DataParser.SACC_DIRECTION_RAD,f"{DataParser.SACC_DIRECTION_RAD}_std"]]
    all_stats.to_csv(os.path.join(save_path, f"sacc_circular_hktest_descriptives.csv"), index=False)
    return


def saccade_ciruclar_game_vis(subs_dict, save_path, time, chosen_10=False, load=False, special_time=False):
    """
    This plots a circular (polar) plot of the average saccade direction.
    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    We take all of the saccades of a subject that are in a trial of a certain category, then bin them, and calculate
    MEAN between BINS
    """
    BINS = 30
    bin_size = int(
        2 * 180 / BINS)  # MUST be an integer; if you want to change the number of bins note this number as well

    if chosen_10:
        f_name = "-500_500_chosen10" if special_time else "chosen10"
    else:
        f_name = "-500_500_nochosen10" if special_time else "nochosen10"

    if not load:
        all_dfs_list = []
        for modality in subs_dict.keys():
            only_mod_dfs_list = []
            for sub_data_path in subs_dict[modality]:
                fl = open(sub_data_path, 'rb')
                sub_data = pickle.load(fl)
                fl.close()
                trial_saccs = sub_data[ET_DATA_DICT][DataParser.DF_SACC_EK]
                trial_saccs = trial_saccs.loc[trial_saccs[DataParser.HERSHMAN_PAD] == False, :]  # only REAL saccades
                relevant_trials = sub_data[TRIAL_INFO]
                relevant_trials = relevant_trials.loc[(relevant_trials[DataParser.IS_LEVEL_ELIMINATED] == False) & (relevant_trials[DataParser.REPLAY] == False) &
                                                      (relevant_trials[ET_data_extraction.IS_PROBED] == True), :]
                all_stim_dur_dfs = []
                for vis in ["FalseNegative", "TruePositive"]:
                    relevant_trials_relevance = relevant_trials.loc[(relevant_trials[DataParser.VIS] == vis), :]
                    if special_time:
                        times = [DataParser.STIM_DUR, DataParser.PRE_STIM_DUR]
                        mask = trial_saccs[times].apply(lambda col: col.isin(relevant_trials_relevance[TRIAL_NUMBER]), axis=0).any(axis=1)
                        trial_saccs_dur = trial_saccs[mask]
                    else:
                        trial_saccs_dur = trial_saccs.loc[trial_saccs[time].isin(list(relevant_trials_relevance[TRIAL_NUMBER])), :]
                    tmp_df = saccade_calc_bins(trial_saccs_dur, bin_size, sub_data, DataParser.VIS, vis, modality)
                    all_stim_dur_dfs.append(tmp_df)
                only_mod_dfs_list.append(pd.concat(all_stim_dur_dfs))

            # Calculated everything for all subs in modality (for the category), now calculate the MEAN
            all_mod_bins_df = pd.concat(only_mod_dfs_list)
            all_dfs_list.append(all_mod_bins_df)
        # Now, we do the mean for all mods combined
        all_bins_df = pd.concat(all_dfs_list)
        all_bins_df.to_csv(os.path.join(save_path, f"sacc_dir_polar_vis_vis_{time}_{special_time}_all.csv"), index=False)
    else:
        all_bins_df = pd.read_csv(os.path.join(save_path, f"sacc_dir_polar_vis_vis_{time}_{special_time}_all.csv"))

    if chosen_10:
        all_bins_df = all_bins_df.loc[all_bins_df[SUBJECT].isin(CHOSEN_10), :]

    mean_df = all_bins_df.groupby([RANGE_START_RAD, DataParser.VIS, MODALITY]).mean(numeric_only=True).reset_index()
    # PLOT DIRECTION OF SACCADE DISTRIBUTION AVERAGED *** WITHIN A VISIBILITY ***
    for modality in [ET_qc_manager.FMRI, ET_qc_manager.MEG]:
        df_cat = mean_df[mean_df[MODALITY] == modality]
        polar_plotter(data_df=df_cat, hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      hue_names_map=VISIBILITY_NAME_MAP,
                      hue_color_map=VISIBILITY_MAP,
                      title_name=f"Saccadde Direction Density in {modality}",
                      save_path=save_path, save_name=f"sacc_dir_polar_vis_notype_GAME_{time}_{modality}_{f_name}",
                      r_range=[0, 0.2, 0.05])

    all_bins_df.loc[:, DataParser.VIS] = all_bins_df[DataParser.VIS] + "_" + all_bins_df[MODALITY]
    mean_df = all_bins_df.groupby([RANGE_START_RAD, DataParser.VIS]).mean(numeric_only=True).reset_index()
    # PLOT DIRECTION OF SACCADE DISTRIBUTION AVERAGED *** WITHIN A VISIBILITY ***
    polar_plotter(data_df=mean_df, hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                  hue_names_map=SACC_VISIBILITY_MOD_NAME_MAP,
                  hue_color_map=SACC_VISIBILITY_MOD_MAP,
                  title_name=f"Saccadde Direction Density in {modality}",
                  save_path=save_path, save_name=f"sacc_dir_polar_vis_notype_{time}_{f_name}",
                  r_range=[0, 0.2, 0.05])

    return


def saccade_ciruclar_all_vis(subs_dict, save_path, time, chosen_10=False, load=False, w_micro=True):
    """
    This plots a circular (polar) plot of the average sacca9de direction.
    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    We take all of the saccades of a subject that are in a trial of a certain category, then bin them, and calculate
    MEAN between BINS
    """
    if w_micro:
        w_micro_str = "withmicro"
    else:
        w_micro_str = "womicro"

    BINS = 30
    bin_size = int(
        2 * 180 / BINS)  # MUST be an integer; if you want to change the number of bins note this number as well

    if chosen_10:
        f_name = "chosen10"
    else:
        f_name = "all"

    if not load:
        all_dfs_list = []
        for modality in subs_dict.keys():
            only_mod_dfs_list = []
            for sub_data_path in subs_dict[modality]:
                fl = open(sub_data_path, 'rb')
                sub_data = pickle.load(fl)
                fl.close()
                for replay in [True, False]:
                    trial_saccs = sub_data[ET_DATA_DICT][DataParser.DF_SACC_EK]
                    trial_saccs = trial_saccs.loc[trial_saccs[DataParser.HERSHMAN_PAD] == False, :]  # only REAL saccades
                    if not w_micro:
                        trial_saccs = trial_saccs.loc[trial_saccs["microsaccade"] == False, :]
                    relevant_trials = sub_data[TRIAL_INFO]
                    relevant_trials = relevant_trials.loc[(relevant_trials[DataParser.IS_LEVEL_ELIMINATED] == False) & (relevant_trials[DataParser.REPLAY] == replay) &
                                                          (relevant_trials[ET_data_extraction.IS_PROBED] == True), :]
                    all_stim_dur_dfs = []
                    for vis in ["FalseNegative", "TruePositive"]:
                        relevant_trials_relevance = relevant_trials.loc[(relevant_trials[DataParser.VIS] == vis), :]
                        trial_saccs_dur = trial_saccs.loc[trial_saccs[time].isin(
                            list(relevant_trials_relevance[TRIAL_NUMBER])), :]
                        tmp_df = saccade_calc_bins(trial_saccs_dur, bin_size, sub_data,
                                                   DataParser.VIS, vis, modality)
                        tmp_df[DataParser.REPLAY] = replay
                        all_stim_dur_dfs.append(tmp_df)
                    only_mod_dfs_list.append(pd.concat(all_stim_dur_dfs))
            # Calculated everything for all subs in modality (for the category), now calculate the MEAN
            all_mod_bins_df = pd.concat(only_mod_dfs_list)
            all_dfs_list.append(all_mod_bins_df)
        # Now, we do the mean for all mods combined
        all_bins_df = pd.concat(all_dfs_list)
        all_bins_df.to_csv(os.path.join(save_path, f"sacc_dir_polar_vis_vis_both_{time}_{w_micro_str}_all.csv"), index=False)
    else:
        all_bins_df = pd.read_csv(os.path.join(save_path, f"sacc_dir_polar_vis_vis_both_{time}_{w_micro_str}_all.csv"))

    if chosen_10:
        all_bins_df = all_bins_df.loc[all_bins_df[SUBJECT].isin(CHOSEN_10), :]

    # PLOT REPLAY
    all_replay = all_bins_df[all_bins_df[DataParser.REPLAY] == True]
    all_bins_df2 = all_replay
    mean_df = all_bins_df2.groupby([RANGE_START_RAD, DataParser.VIS]).mean(numeric_only=True).reset_index()
    # PLOT DIRECTION OF SACCADE DISTRIBUTION AVERAGED *** WITHIN A VISIBILITY ***
    polar_plotter(data_df=mean_df, hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                  hue_names_map=VISIBILITY_NAME_MAP,
                  hue_color_map=VISIBILITY_MAP_REPLAY,
                  title_name=f"Saccadde Direction Density in REPLAY",
                  save_path=save_path, save_name=f"sacc_dir_polar_vis_notype_REPLAY_{time}_{w_micro_str}_{f_name}",
                  r_range=[0, 0.2, 0.05])

    mean_df = all_bins_df2.groupby([RANGE_START_RAD, DataParser.VIS, MODALITY]).mean(numeric_only=True).reset_index()
    # PLOT DIRECTION OF SACCADE DISTRIBUTION AVERAGED *** WITHIN A VISIBILITY ***
    for modality in [ET_qc_manager.FMRI, ET_qc_manager.MEG]:
        df_cat = mean_df[mean_df[MODALITY] == modality]
        polar_plotter(data_df=df_cat, hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      hue_names_map=VISIBILITY_NAME_MAP,
                      hue_color_map=VISIBILITY_MAP,
                      title_name=f"Saccadde Direction Density in {modality} REPLAY",
                      save_path=save_path, save_name=f"sacc_dir_polar_vis_notype_REPLAY_{time}_{modality}_{w_micro_str}_{f_name}",
                      r_range=[0, 0.2, 0.05])

    # PLOT GAME
    all_no_replay = all_bins_df[all_bins_df[DataParser.REPLAY] == False]
    all_bins_df2 = all_no_replay
    mean_df = all_bins_df2.groupby([RANGE_START_RAD, DataParser.VIS]).mean(numeric_only=True).reset_index()
    # PLOT DIRECTION OF SACCADE DISTRIBUTION AVERAGED *** WITHIN A VISIBILITY ***
    polar_plotter(data_df=mean_df, hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                  hue_names_map=VISIBILITY_NAME_MAP,
                  hue_color_map=VISIBILITY_MAP_GAME,
                  title_name=f"Saccadde Direction Density in GAME",
                  save_path=save_path, save_name=f"sacc_dir_polar_vis_notype_GAME_{time}_{w_micro_str}_{f_name}",
                  r_range=[0, 0.2, 0.05])

    mean_df = all_bins_df2.groupby([RANGE_START_RAD, DataParser.VIS, MODALITY]).mean(numeric_only=True).reset_index()
    # PLOT DIRECTION OF SACCADE DISTRIBUTION AVERAGED *** WITHIN A VISIBILITY ***
    for modality in [ET_qc_manager.FMRI, ET_qc_manager.MEG]:
        df_cat = mean_df[mean_df[MODALITY] == modality]
        polar_plotter(data_df=df_cat, hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      hue_names_map=VISIBILITY_NAME_MAP,
                      hue_color_map=VISIBILITY_MAP,
                      title_name=f"Saccadde Direction Density in {modality} GAME",
                      save_path=save_path, save_name=f"sacc_dir_polar_vis_notype_GAME_{time}_{modality}_{w_micro_str}_{f_name}",
                      r_range=[0, 0.2, 0.05])

    all_bins_df.loc[all_bins_df[DataParser.REPLAY]==True, DataParser.REPLAY] = "True"
    all_bins_df.loc[all_bins_df[DataParser.REPLAY] == False, DataParser.REPLAY] = "False"
    all_bins_df.loc[:, DataParser.VIS] = all_bins_df[DataParser.VIS] + "_" + all_bins_df[DataParser.REPLAY]
    mean_df = all_bins_df.groupby([RANGE_START_RAD, DataParser.REPLAY, DataParser.VIS]).mean(numeric_only=True).reset_index()
    # PLOT DIRECTION OF SACCADE DISTRIBUTION AVERAGED *** WITHIN A VISIBILITY ***
    polar_plotter(data_df=mean_df, hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                  hue_names_map=SACC_VISIBILITY_GAME_NAME_MAP,
                  hue_color_map=SACC_VISIBILITY_GAME_MAP,
                  title_name=f"Saccadde Direction Density",
                  save_path=save_path, save_name=f"sacc_dir_polar_vis_notype_both_{time}_{w_micro_str}_{f_name}",
                  r_range=[0, 0.2, 0.05])

    return


def saccade_ciruclar_game_stim_plot_location(subs_dict, save_path, time, chosen_10=False, load=False, left_right=True):
    """
    This plots a circular (polar) plot of the average saccade direction.
    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    We take all of the saccades of a subject that are in a trial of a certain category, then bin them, and calculate
    MEAN between BINS
    """
    BINS = 30
    bin_size = int(
        2 * 180 / BINS)  # MUST be an integer; if you want to change the number of bins note this number as well

    if chosen_10:
        f_name = "chosen10"
    else:
        f_name = "all"

    if left_right:
        loc_str = "left_right"
        locations = [LEFT, RIGHT]
    else:
        loc_str = "top_bottom"
        locations = [TOP, BOTTOM]

    if not load:
        all_dfs_list = []
        for modality in subs_dict.keys():
            only_mod_dfs_list = []
            for sub_data_path in subs_dict[modality]:
                fl = open(sub_data_path, 'rb')
                sub_data = pickle.load(fl)
                fl.close()
                trial_saccs = sub_data[ET_DATA_DICT][DataParser.DF_SACC_EK]
                trial_saccs = trial_saccs.loc[trial_saccs[DataParser.HERSHMAN_PAD] == False, :]  # only REAL saccades
                relevant_trials = sub_data[TRIAL_INFO]
                relevant_trials = relevant_trials.loc[(relevant_trials[DataParser.IS_LEVEL_ELIMINATED] == False) & (relevant_trials[DataParser.REPLAY] == False) &
                                                      (relevant_trials[ET_data_extraction.IS_PROBED] == True), :]

                if left_right:
                    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(RIGHT), "location"] = RIGHT
                    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(LEFT), "location"] = LEFT
                else:
                    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(TOP), "location"] = TOP
                    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(BOTTOM), "location"] = BOTTOM
                all_stim_dur_dfs = []
                for vis in ["FalseNegative", "TruePositive"]:
                    for stimtype in locations:
                        relevant_trials_relevance = relevant_trials.loc[
                                                    (relevant_trials[DataParser.VIS] == vis) & (relevant_trials["location"] == stimtype), :]
                        trial_saccs_dur = trial_saccs.loc[trial_saccs[time].isin(
                            list(relevant_trials_relevance[TRIAL_NUMBER])), :]
                        tmp_df = saccade_calc_bins(trial_saccs_dur, bin_size, sub_data,
                                                                  DataParser.VIS, vis, modality)
                        tmp_df.loc[:, "location"] = stimtype
                        all_stim_dur_dfs.append(tmp_df)
                only_mod_dfs_list.append(pd.concat(all_stim_dur_dfs))

            # Calculated everything for all subs in modality (for the category), now calculate the MEAN
            all_mod_bins_df = pd.concat(only_mod_dfs_list)
            all_dfs_list.append(all_mod_bins_df)
        # Now, we do the mean for all mods combined
        all_bins_df = pd.concat(all_dfs_list)
        all_bins_df.to_csv(os.path.join(save_path, f"sacc_dir_polar_location_{loc_str}_vis_{time}_all.csv"), index=False)
    else:
        all_bins_df = pd.read_csv(os.path.join(save_path, f"sacc_dir_polar_location_{loc_str}_vis_{time}_all.csv"))

    if chosen_10:
        all_bins_df = all_bins_df.loc[all_bins_df[SUBJECT].isin(CHOSEN_10), :]

    all_bins_df.loc[:, DataParser.VIS] = all_bins_df[DataParser.VIS] + "_" + all_bins_df["location"]
    mean_df = all_bins_df.groupby([RANGE_START_RAD, DataParser.VIS, MODALITY]).mean(numeric_only=True).reset_index()
    # PLOT DIRECTION OF SACCADE DISTRIBUTION AVERAGED *** WITHIN A STIMULUS CATEGORY ***
    for modality in [ET_qc_manager.FMRI, ET_qc_manager.MEG]:
        df_cat = mean_df[mean_df[MODALITY] == modality]
        polar_plotter(data_df=df_cat, hue_col=DataParser.VIS, hue_col_name=DataParser.VIS,
                      hue_names_map=SACC_VISIBILITY_LOCATION_NAME_MAP,
                      hue_color_map=SACC_VISIBILITY_LOCATION_MAP,
                      title_name=f"Saccadde Direction Density in {modality}",
                      save_path=save_path, save_name=f"sacc_dir_polar_location_{loc_str}_vis_{time}_{modality}_{f_name}",
                      r_range=[0, 0.2, 0.05])
    return


def saccade_vpeak_histogram(subs_dict, save_path, chosen_10=True, load=False):
    """
    This plots a circular (polar) plot of the average saccade direction.
    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    We take all of the saccades of a subject that are in a trial of a certain category, then bin them, and calculate
    MEAN between BINS
    """
    BINS = 30
    bin_size = int(
        2 * 180 / BINS)  # MUST be an integer; if you want to change the number of bins note this number as well

    if chosen_10:
        f_name = "chosen10"
    else:
        f_name = "all"

    if not load:
        all_dfs_list = []
        for modality in subs_dict.keys():
            only_mod_dfs_list = []
            for sub_data_path in subs_dict[modality]:
                fl = open(sub_data_path, 'rb')
                sub_data = pickle.load(fl)
                fl.close()
                trial_saccs = sub_data[ET_DATA_DICT][DataParser.DF_SACC_EK]
                trial_saccs = trial_saccs.loc[trial_saccs[DataParser.HERSHMAN_PAD] == False, :]  # only REAL saccades
                relevant_trials = sub_data[TRIAL_INFO]
                relevant_trials = relevant_trials.loc[(relevant_trials[DataParser.IS_LEVEL_ELIMINATED] == False) & relevant_trials[DataParser.REPLAY] == False, :]
                all_stim_dur_dfs = []
                sub_med = trial_saccs["vPeak"].median()
                #trial_saccs["is_fast"] = trial_saccs["vPeak"] > (sub_med + 500)
                trial_saccs["is_fast"] = trial_saccs["vPeak"] > (sub_med + 500)
                i = 0
                for saccade_type in [True, False]:
                    for stim_type in ["FalseNegative", "TruePositive"]:
                        relevant_trials_relevance = relevant_trials.loc[
                                                    relevant_trials[DataParser.VIS] == stim_type, :]
                        trial_saccs_dur = trial_saccs.loc[(trial_saccs[DataParser.TRIAL].isin(
                            list(relevant_trials_relevance[TRIAL_NUMBER]))) & (trial_saccs["is_fast"] == saccade_type), :]
                        all_stim_dur_dfs.append(saccade_calc_bins(trial_saccs_dur, bin_size, sub_data,
                                                                  DataParser.VIS, stim_type, modality))
                        if saccade_type:
                            all_stim_dur_dfs[i]["sacc_type"] = "Fast"
                        else:
                            all_stim_dur_dfs[i]["sacc_type"] = "Slow"
                        i += 1
                only_mod_dfs_list.append(pd.concat(all_stim_dur_dfs))

            # Calculated everything for all subs in modality (for the category), now calculate the MEAN
            all_mod_bins_df = pd.concat(only_mod_dfs_list)
            all_dfs_list.append(all_mod_bins_df)
        # Now, we do the mean for all mods combined
        all_bins_df = pd.concat(all_dfs_list)
        all_bins_df.to_csv(os.path.join(save_path, f"sacc_dir_polar_vis_speed_all.csv"), index=False)
    else:
        all_bins_df = pd.read_csv(os.path.join(save_path, f"sacc_dir_polar_vis_speed_all.csv"))

    if chosen_10:
        all_bins_df = all_bins_df.loc[all_bins_df[SUBJECT].isin(CHOSEN_10), :]
    mean_df = all_bins_df.groupby([RANGE_START_RAD, DataParser.VIS, "sacc_type", MODALITY]).mean(numeric_only=True).reset_index()
    # PLOT DIRECTION OF SACCADE DISTRIBUTION AVERAGED *** WITHIN A STIMULUS CATEGORY ***
    for modality in [ET_qc_manager.MEG, ET_qc_manager.FMRI]:
        df_cat = mean_df[mean_df[MODALITY] == modality]
        df_cat[f"{DataParser.VIS}_sacc_type"] = df_cat[DataParser.VIS] + df_cat[f"sacc_type"]
        #df_cat = df_cat[df_cat[f"{DataParser.VIS}_sacc_type"] == "FalseNegativeSlow"]
        polar_plotter(data_df=df_cat, hue_col=f"{DataParser.VIS}_sacc_type", hue_col_name="saccade type",
                      hue_names_map={"FalseNegativeFast": "Unseen Fast", "TruePositiveFast": "Seen Fast",
                                     "FalseNegativeSlow": "Unseen Slow", "TruePositiveSlow": "Seen Slow"},
                      hue_color_map=VIS_SPEED_MAP,
                      title_name=f"Saccadde Direction Density in {modality}",
                      save_path=save_path, save_name=f"sacc_dir_polar_vis_speed_{modality}_{f_name}",
                      r_range=[0, 0.2, 0.05])
    return


def modality_dfs(all_subs_df):
    all_subs_df2 = all_subs_df.loc[:, [SUBJECT_NAME, "mod", "trialNumber", "currentLevelID", "activeLevelID", "type", "probed",
                                    "stimulusID", "stimulusType",
                                    "stimulusName", "stimulusLocation", "questionTS", "responseTS", "responseDT", "response",
                                    "Visibility", "IS_LEVEL_ELIMINATED", "TargetType", "IsReplay",
                                    "PreStimNumBlinks", "StimDurationNumBlinks", "PreStimNumSaccs", "PreStimMicroNumSaccs", "StimDurationNumSaccs", "StimDurationMicroNumSaccs"]]
    all_subs_df2[["PreStimNumBlinks", "StimDurationNumBlinks", "PreStimNumSaccs", "PreStimMicroNumSaccs", "StimDurationNumSaccs", "StimDurationMicroNumSaccs"]] = all_subs_df2[["PreStimNumBlinks", "StimDurationNumBlinks", "PreStimNumSaccs", "PreStimMicroNumSaccs", "StimDurationNumSaccs", "StimDurationMicroNumSaccs"]].fillna(value=0)
    all_subs_df2["StimDurationNumSaccs"] = all_subs_df2["StimDurationNumSaccs"] + all_subs_df2["StimDurationMicroNumSaccs"]
    all_subs_df2["PreStimNumSaccs"] = all_subs_df2["PreStimNumSaccs"] + all_subs_df2["PreStimMicroNumSaccs"]
    all_subs_df2.drop(columns=["PreStimMicroNumSaccs", "StimDurationMicroNumSaccs"], inplace=True)

    for mod in all_subs_df2["mod"].unique():
        mod_df = all_subs_df2.loc[all_subs_df2["mod"] == mod, :]
        mod_df.to_csv(os.path.join(f"/mnt/beegfs/XNAT/COGITATE/{mod}/phase_2/processed/bids/derivatives/qcs", f"ses-v2-TrialInfo.csv"), index=False)

    return


def circular_saccade_table_game(subs_dict, time, save_path, load=False, w_micro=True):
    if w_micro:
        w_micro_str = "withmicro"
    else:
        w_micro_str = "womicro"
    if not load:
        all_dfs_list = []
        for modality in subs_dict.keys():
            for vis in ["TruePositive", "FalseNegative"]:
                for sub_data_path in subs_dict[modality]:
                    fl = open(sub_data_path, 'rb')
                    sub_data = pickle.load(fl)
                    fl.close()
                    print(sub_data_path)
                    trial_saccs = sub_data[ET_DATA_DICT][DataParser.DF_SACC_EK]
                    trial_saccs = trial_saccs.loc[trial_saccs[DataParser.HERSHMAN_PAD] == False, :]  # only REAL saccades
                    if not w_micro:
                        trial_saccs = trial_saccs.loc[trial_saccs["microsaccade"] == False, :]
                    relevant_trials = sub_data[TRIAL_INFO]
                    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(TOP), VERTICAL] = TOP
                    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(BOTTOM), VERTICAL] = BOTTOM
                    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(LEFT), HORIZONTAL] = LEFT
                    relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(RIGHT), HORIZONTAL] = RIGHT
                    relevant_trials = relevant_trials.loc[(relevant_trials[DataParser.IS_LEVEL_ELIMINATED] == False) & (relevant_trials[ET_data_extraction.IS_PROBED] == True) &
                                                          (relevant_trials[DataParser.REPLAY] == False) & (relevant_trials[DataParser.VIS] == vis), :]
                    if relevant_trials.empty:
                        continue

                    trial_saccs_dur = trial_saccs.loc[trial_saccs[time].isin(list(relevant_trials[TRIAL_NUMBER])), :]
                    if trial_saccs_dur.empty:
                        continue
                    means_df = trial_saccs_dur.groupby([time]).mean().reset_index()
                    relevant_trials = relevant_trials.loc[relevant_trials[TRIAL_NUMBER].isin(list(means_df[time]))].reset_index(drop=True)
                    relevant_trials.loc[:, DataParser.SACC_DIRECTION_RAD] = means_df[DataParser.SACC_DIRECTION_RAD]
                    relevant_trials.loc[:, "sacc_direction_deg"] = means_df.loc[:, "sacc_direction_deg"]
                    relevant_trials = relevant_trials.loc[:, [QualityChecker.STIM_TYPE, DataParser.VIS, DataParser.SACC_DIRECTION_RAD, "sacc_direction_deg", TRIAL_NUMBER, HORIZONTAL, VERTICAL]]
                    relevant_trials.loc[:, SUBJECT] = sub_data[PARAMS]['SubjectName']
                    relevant_trials.loc[:, LAB] = sub_data[PARAMS]['SubjectName'][:2]
                    relevant_trials.loc[:, MODALITY] = modality
                    print(sub_data_path)
                    all_dfs_list.append(relevant_trials)

        # Now, we do the mean for all mods combined
        all_bins_df = pd.concat(all_dfs_list)
        all_bins_df.to_csv(os.path.join(save_path, f"sacc_circular_trials_game_{time}_{w_micro_str}_all.csv"), index=False)
    else:
        all_bins_df = pd.read_csv(os.path.join(save_path, f"sacc_circular_trials_game_{time}_{w_micro_str}_all.csv"))

    for name in ["nochosen10", "chosen10"]:
        if name == "nochosen10":
            relevant_data = all_bins_df[~all_bins_df[SUBJECT].isin(CHOSEN_10)]
        else:
            relevant_data = all_bins_df[all_bins_df[SUBJECT].isin(CHOSEN_10)]

        list_to_combo_from = [MODALITY, QualityChecker.STIM_TYPE, DataParser.VIS, HORIZONTAL, VERTICAL]
        combos = []
        for r in range(1, len(list_to_combo_from) + 1):
            combos.append(itertools.combinations(list_to_combo_from, r))

        stats_df_list = []
        df_to_mean_list = []
        for combo in combos:
            for x in combo:
                if "mod" not in list(x):
                    df_to_mean = relevant_data.groupby(list(x) + [SUBJECT, MODALITY]).mean().reset_index()
                else:
                    df_to_mean = relevant_data.groupby(list(x) + [SUBJECT]).mean().reset_index()
                mean_df = df_to_mean.groupby(list(x)).mean().reset_index()
                std_df = df_to_mean.groupby(list(x)).std().reset_index()
                for col in ["sacc_direction_rad", "sacc_direction_deg"]:
                    mean_df.loc[:, f"{col}_std"] = std_df[col]
                    mean_df.rename(columns={col: f"{col}_mean"}, inplace=True)
                stats_df_list.append(mean_df)
                df_to_mean_list.append(df_to_mean)

        stat_df = pd.concat(stats_df_list)
        stat_df.to_csv(os.path.join(save_path, f"sacc_circular_trials_game_{time}_{w_micro_str}_{name}_descriptives.csv"),
                       index=False)

    return


def circular_saccade_table_replay(subs_dict, time, save_path):
    all_dfs_list = []
    for modality in subs_dict.keys():
        for vis in ["TruePositive", "FalseNegative"]:
            for sub_data_path in subs_dict[modality]:
                fl = open(sub_data_path, 'rb')
                sub_data = pickle.load(fl)
                fl.close()
                trial_saccs = sub_data[ET_DATA_DICT][DataParser.DF_SACC_EK]
                trial_saccs = trial_saccs.loc[trial_saccs[DataParser.HERSHMAN_PAD] == False, :]  # only REAL saccades
                relevant_trials = sub_data[TRIAL_INFO]
                relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(TOP), VERTICAL] = TOP
                relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(BOTTOM), VERTICAL] = BOTTOM
                relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(LEFT), HORIZONTAL] = LEFT
                relevant_trials.loc[relevant_trials[DataParser.STIM_LOC].str.contains(RIGHT), HORIZONTAL] = RIGHT
                relevant_trials = relevant_trials.loc[(relevant_trials[DataParser.IS_LEVEL_ELIMINATED] == False) &
                                                      (relevant_trials[DataParser.REPLAY] == True) & (relevant_trials[DataParser.VIS] == vis), :]
                if relevant_trials.empty:
                    continue

                trial_saccs_dur = trial_saccs.loc[trial_saccs[time].isin(list(relevant_trials[TRIAL_NUMBER])), :]
                if trial_saccs_dur.empty:
                    continue
                means_df = trial_saccs_dur.groupby([time]).mean().reset_index()
                relevant_trials = relevant_trials.loc[relevant_trials[TRIAL_NUMBER].isin(list(means_df[time]))].reset_index(drop=True)
                relevant_trials.loc[:, DataParser.SACC_DIRECTION_RAD] = means_df[DataParser.SACC_DIRECTION_RAD]
                relevant_trials.loc[:, "sacc_direction_deg"] = means_df.loc[:, "sacc_direction_deg"]
                relevant_trials = relevant_trials.loc[:, [QualityChecker.STIM_TYPE, DataParser.VIS, DataParser.SACC_DIRECTION_RAD, "sacc_direction_deg", TRIAL_NUMBER, VERTICAL, HORIZONTAL]]
                relevant_trials.loc[:, SUBJECT] = sub_data[PARAMS]['SubjectName']
                relevant_trials.loc[:, LAB] = sub_data[PARAMS]['SubjectName'][:2]
                relevant_trials.loc[:, MODALITY] = modality
                all_dfs_list.append(relevant_trials)

    # Now, we do the mean for all mods combined
    all_bins_df = pd.concat(all_dfs_list)
    all_bins_df.to_csv(os.path.join(save_path, f"sacc_circular_trials_replay_{time}.csv"), index=False)
    return


def debug_subject(subs_dict, sub_name):
    for mod in subs_dict:
        for sub in subs_dict[mod]:
            if (sub_name in sub):
                fl = open(sub, 'rb')
                sub_data = pickle.load(fl)
                fl.close()
                x=5
    return

def tau_giant_df(subs_df, save_path):
    relevant_trials_game = subs_df.loc[(subs_df[DataParser.IS_LEVEL_ELIMINATED] == False) & (
                           subs_df[DataParser.REPLAY] == False) & (subs_df[ET_data_extraction.IS_PROBED] == True), :]
    relevant_trials_game = relevant_trials_game.loc[~relevant_trials_game[QualityChecker.STIM_TYPE].str.contains("None")]

    relevant_trials_replay = subs_df[(subs_df[DataParser.REPLAY] == True) & (subs_df[DataParser.IS_LEVEL_ELIMINATED] == False)]
    relevant_trials_replay = relevant_trials_replay.loc[relevant_trials_replay[QualityChecker.STIM_TYPE] == relevant_trials_replay["TargetType"], :]
    relevant_trials_replay = relevant_trials_replay[((relevant_trials_replay["stimulusType"] == "Face") | (relevant_trials_replay["stimulusType"] == "Object"))]

    all_trials = pd.concat([relevant_trials_game, relevant_trials_replay])
    all_trials = all_trials.loc[:, [DataParser.REPLAY, QualityChecker.STIM_TYPE, "SubjectName", "trialNumber", "Visibility",
                                    "StimDurationStimDistDegs", "StimDurationbothNumSaccs", "StimDurationbothsacc_direction_rad",
                                    "StimDurationbothsacc_direction_deg", "StimDurationNumBlinks"]]
    all_trials.to_csv(os.path.join(save_path, f"et_giant_vis_cond.csv"), index=False)
    return


def get_removed_fixations(subs_dict, save_path):
    sub_names = list()
    num_removed = list()
    num_removed_in_trial = list()
    for modality in subs_dict.keys():
        subs_list = subs_dict[modality]
        for sub_data_path in subs_list:
            fl = open(sub_data_path, 'rb')
            sub_data = pickle.load(fl)
            fl.close()

            sub_name = sub_data[PARAMS][SUBJECT_NAME]
            df_fix = sub_data[ET_DATA_DICT][DataParser.DF_FIXAT]
            df_fix = df_fix.loc[(df_fix[f"{DataParser.HERSHMAN_PAD}"] == False) & (df_fix[f"{DataParser.REAL_SACC}"] == False) & (df_fix["duration"] < 100) &
                            (~pd.isna(df_fix[f"is_EyelinkBlink"])), :]
            sub_names.append(sub_name)
            num_removed.append(df_fix.shape[0])
            df_fix = df_fix.loc[(df_fix[DataParser.TRIAL] != -1), :]
            num_removed_in_trial.append(df_fix.shape[0])


    df = pd.DataFrame({'SubName': sub_names, 'NumFixationsRemoved': num_removed, 'NumFixationsRemovedInTrial': num_removed_in_trial})
    df.to_csv(os.path.join(save_path, f"numFixRemoved.csv"), index=False)


def new_100_600_analysis(all_subs_df, save_path):
    analysis_cols = ["FirstFixDur100", "P100P600NumBlinks", "P100P600bothNumSaccs", "FirstbothSaccTimePostOnset100",
                     "P100P600bothsacc_direction_deg", "P100P600bothsacc_direction_rad",
                     "P100P600NumSaccs", "FirstSaccTimePostOnset100",
                     "P100P600sacc_direction_deg", "P100P600sacc_direction_rad"]
    all_subs_df["P100P600NumBlinks"] = all_subs_df["P100P600NumBlinks"].fillna(0)
    all_subs_df["P100P600bothNumSaccs"] = all_subs_df["P100P600bothNumSaccs"].fillna(0)
    all_subs_df["P100P600NumSaccs"] = all_subs_df["P100P600NumSaccs"].fillna(0)
    all_subs_df = all_subs_df[~all_subs_df[SUBJECT_NAME].isin(CHOSEN_10)]
    all_subs_df = all_subs_df[((all_subs_df[QualityChecker.STIM_TYPE] == "Face") | (all_subs_df[QualityChecker.STIM_TYPE] == "Object"))]

    replay_map = {False: "dAT", True: "AT", "FALSE": "dAT", "TRUE": "AT", 0: "dAT", 1: "AT"}
    all_subs_df = all_subs_df.copy()

    for col in analysis_cols:
        df_col = all_subs_df.groupby([SUBJECT_NAME, QualityChecker.STIM_TYPE, DataParser.REPLAY]).mean()
        df_col.reset_index(inplace=True)
        wide = (df_col.assign(replay_tag=df_col[DataParser.REPLAY].map(replay_map))
                              .pivot_table(index=SUBJECT_NAME,
                              columns=["replay_tag", QualityChecker.STIM_TYPE],
                              values=col,
                              aggfunc="first"))

        # flatten columns: (AT, Face) -> "AT face", (dAT, Object) -> "dAT obj"
        wide.columns = [f"{tag} {stim}" for tag, stim in wide.columns]
        wide = wide.reset_index()

        wide.to_csv(os.path.join(save_path, f"additional_analysis_P100P600_{col}_nochosen10.csv"), index=True)

    return


def process_data(subs_dict, beh_data_path, save_path, valid_beh=True, include_invalid=False):
    if include_invalid:
        save_path = save_path+ r"/with_invalid"
        global CHOSEN_10
        CHOSEN_10 = []

    subs_dict = {modality: subs_dict[modality] for modality in list(subs_dict.keys()) if len(subs_dict[modality]) > 0}
    #debug_subject(subs_dict, "SC109")
    all_subs_df = get_all_subs_giant_df(subs_dict, save_path, load=True, rerun_analysis_windows=False)

    if True:
        all_subs_df["FirstbothSaccTimePostOnset"] = np.nanmin(all_subs_df[["FirstSaccTimePostOnset", "FirstMicroSaccTimePostOnset"]].values, axis=1)
    #modality_dfs(all_subs_df)
    # Get only probed trials
    all_subs_df_probed = all_subs_df[(all_subs_df["probed"] == True) & (all_subs_df[DataParser.IS_LEVEL_ELIMINATED] == False)]
    print("actually running stuff")
    # 30-01-26
    circular_saccade_table_game(subs_dict, "P250P500", save_path, load=False)
    #poststim_blink_num_analysis_new(all_subs_df_probed, save_path, is_250_500=True)
    return
    poststim_first_fix_analysis_new(all_subs_df_probed, save_path, is_250_500=True)
    poststim_first_sacc_analysis_new(all_subs_df_probed, save_path, is_250_500=True)
    poststim_first_sacc_analysis_new(all_subs_df_probed, save_path, is_250_500=True, w_micro=False)
    stim_duration_saccade_num_analysis_new(all_subs_df, save_path, is_250_500=True, w_micro=False)
    stim_duration_saccade_num_analysis_new(all_subs_df, save_path, is_250_500=True, w_micro=True)
    #new_100_600_analysis(all_subs_df, save_path)
    # Idk when
    #pre_stim_fixation_analysis(all_subs_df_probed, save_path)
    #poststim_first_fix_analysis_game_vs_replay(all_subs_df_probed, save_path)
    #calc_fixation_proportion(subs_dict, save_path)
    #get_removed_fixations(subs_dict, save_path)
    #calc_fixation_proportion(subs_dict, save_path)
    #fixation_plots(subs_dict, save_path, "phase4", is_tau=False, load=True, special_time=True, special_time_start=-500, special_time_end=500)
    exit()
    """
    # 05-12-2024
    pre_stim_saccade_num_analysis(all_subs_df_probed, save_path)
    pre_stim_blink_num_analysis(all_subs_df_probed, save_path)
    poststim_first_fix_analysis_new(all_subs_df_probed, save_path)
    poststim_first_sacc_analysis_new(all_subs_df_probed, save_path)
    stim_duration_saccade_num_analysis_new(all_subs_df, save_path)
    poststim_blink_num_analysis_new(all_subs_df_probed, save_path)
    circular_saccade_table_game(subs_dict, DataParser.PRE_STIM_DUR, save_path, load=True)
    circular_saccade_table_game(subs_dict, DataParser.STIM_DUR, save_path, load=True)
    poststim_first_fix_analysis_game_vs_replay(all_subs_df_probed, save_path)
    poststim_saccade_num_analysis_game_vs_replay(all_subs_df_probed, save_path)
    poststim_blink_num_analysis_game_vs_replay(all_subs_df_probed, save_path)
    """
    poststim_first_fix_analysis_new(all_subs_df_probed, save_path)
    #fixation_plots(subs_dict, save_path, "phase4", is_tau=False, load=True, special_time=True, special_time_start=-500, special_time_end=500)
    #saccade_plot_trial_visibility_no_split(subs_dict, save_path, load=True, w_micro=True)
    #saccade_ciruclar_game_vis(subs_dict, save_path, DataParser.TRIAL, chosen_10=False, load=False, special_time=True)
    #saccade_ciruclar_game_vis(subs_dict, save_path, DataParser.STIM_DUR, chosen_10=False, load=False)
    #saccade_ciruclar_all_vis(subs_dict, save_path, DataParser.STIM_DUR, chosen_10=False, load=True, w_micro=True)
    #blink_plot_trial_vis(subs_dict, save_path, load=True)
    #saccade_trio(subs_dict, save_path, load=True)
    #blink_plot_trial_relevance_replay(subs_dict, save_path, load=True)
    #saccade_plot_trial_visibility_no_split(subs_dict, save_path, load=True, w_micro=False)
    return
    pre_stim_saccade_num_analysis(all_subs_df_probed, save_path, w_micro=True)
    pre_stim_saccade_num_analysis(all_subs_df_probed, save_path, w_micro=False)
    poststim_first_sacc_analysis_new(all_subs_df_probed, save_path, w_micro=True)
    poststim_first_sacc_analysis_new(all_subs_df_probed, save_path, w_micro=False)
    stim_duration_saccade_num_analysis_new(all_subs_df, save_path, w_micro=True)
    stim_duration_saccade_num_analysis_new(all_subs_df, save_path, w_micro=False)
    circular_saccade_table_game(subs_dict, DataParser.PRE_STIM_DUR, save_path, load=False, w_micro=True)
    circular_saccade_table_game(subs_dict, DataParser.STIM_DUR, save_path, load=False, w_micro=True)
    circular_saccade_table_game(subs_dict, DataParser.PRE_STIM_DUR, save_path, load=False, w_micro=False)
    circular_saccade_table_game(subs_dict, DataParser.STIM_DUR, save_path, load=False, w_micro=False)

    fixation_plots(subs_dict, save_path, "phase4", is_tau=False, load=False)
    saccade_ciruclar_all_vis(subs_dict, save_path, DataParser.STIM_DUR, chosen_10=False, load=False, w_micro=True)
    saccade_ciruclar_all_vis(subs_dict, save_path, DataParser.PRE_STIM_DUR, chosen_10=False, load=False, w_micro=True)
    saccade_ciruclar_all_vis(subs_dict, save_path, DataParser.STIM_DUR, chosen_10=False, load=False, w_micro=False)
    saccade_ciruclar_all_vis(subs_dict, save_path, DataParser.PRE_STIM_DUR, chosen_10=False, load=False, w_micro=False)
    #calc_fixation_proportion(subs_dict, save_path)

    return

    """
    # 08-09-24 things we already ran
    
    pre_stim_blink_num_analysis(all_subs_df_probed, save_path)
    pre_stim_fixation_analysis(all_subs_df_probed, save_path)
    pre_stim_saccade_num_analysis(all_subs_df_probed, save_path)
    
    stim_duration_saccade_num_analysis(all_subs_df_probed, save_path)
    stim_duration_saccade_num_analysis_replay(all_subs_df_probed, save_path)
    
    stim_duration_saccade_num_analysis_left_right(all_subs_df_probed, save_path, left_right=True)
    stim_duration_saccade_num_analysis_left_right_replay(all_subs_df_probed, save_path, left_right=True)

    stim_duration_saccade_num_analysis_left_right(all_subs_df_probed, save_path, left_right=False)
    stim_duration_saccade_num_analysis_left_right_replay(all_subs_df_probed, save_path, left_right=False)
   
    saccade_ciruclar_game_stim_plot_temp(subs_dict, save_path, DataParser.STIM_DUR, chosen_10=False, load=False)
    saccade_ciruclar_game_stim_plot_temp(subs_dict, save_path, DataParser.PRE_STIM_DUR, chosen_10=False, load=False)
 
    saccade_ciruclar_game_stim_plot_location(subs_dict, save_path, DataParser.PRE_STIM_DUR, chosen_10=False, load=False, left_right=True)
    saccade_ciruclar_game_stim_plot_location(subs_dict, save_path, DataParser.STIM_DUR, chosen_10=False, load=False, left_right=True)

    saccade_ciruclar_game_stim_plot_location(subs_dict, save_path, DataParser.PRE_STIM_DUR, chosen_10=False, load=False, left_right=False)
    saccade_ciruclar_game_stim_plot_location(subs_dict, save_path, DataParser.STIM_DUR, chosen_10=False, load=False, left_right=False)
    
    saccade_plot_trial_visibility_no_split_objects(subs_dict, save_path, load=False)
    saccade_plot_trial_visibility_no_split_objects_unprobed(subs_dict, save_path, load=False)

    saccade_plot_trial_visibility_no_split_locations(subs_dict, save_path, load=False, left_right=True)
    saccade_plot_trial_visibility_no_split_locations_unprobed(subs_dict, save_path, load=False, left_right=True)
    
    saccade_plot_trial_visibility_no_split_locations(subs_dict, save_path, load=False, left_right=False)
    saccade_plot_trial_visibility_no_split_locations_unprobed(subs_dict, save_path, load=False, left_right=False)
    
    saccade_ciruclar_replay_stim_plot_temp(subs_dict, save_path, DataParser.STIM_DUR, chosen_10=False, load=False)
    saccade_ciruclar_replay_stim_plot_temp(subs_dict, save_path, DataParser.PRE_STIM_DUR, chosen_10=False, load=False)

    blink_plot_trial_probe_vis(subs_dict, save_path, load=False)
    blink_plot_trial_unprobe_vis(subs_dict, save_path, load=False)
    
    blink_plot_trial_probe_vis_location(subs_dict, save_path, load=False, left_right=True)
    blink_plot_trial_unprobe_vis_location(subs_dict, save_path, load=False, left_right=True)

    blink_plot_trial_probe_vis_location(subs_dict, save_path, load=False, left_right=False)
    blink_plot_trial_unprobe_vis_location(subs_dict, save_path, load=False, left_right=False)
    
    saccade_plot_trial_visibility_no_split(subs_dict, save_path, load=False)
    blink_plot_trial_vis(subs_dict, save_path, load=False)
    blink_plot_trial_replay(subs_dict, save_path, load=False)
    saccade_plot_replay_target(subs_dict, save_path, load=False)
    
    fixation_plots(subs_dict, save_path, "phase4", is_tau=True)
    
    circular_saccade_table_game(subs_dict, DataParser.PRE_STIM_DUR, save_path)
    circular_saccade_table_game(subs_dict, DataParser.STIM_DUR, save_path)
    circular_saccade_table_replay(subs_dict, DataParser.PRE_STIM_DUR, save_path)
    circular_saccade_table_replay(subs_dict, DataParser.STIM_DUR, save_path)

    poststim_first_fix_analysis_new(all_subs_df_probed, save_path)
    poststim_first_fix_analysis_replay_new(all_subs_df_probed, save_path)
    
    poststim_first_sacc_analysis_new(all_subs_df_probed, save_path)
    poststim_first_sacc_analysis_replay_new(all_subs_df_probed, save_path)
    
    poststim_blink_num_analysis_new(all_subs_df_probed, save_path)
    poststim_blink_num_analysis_replay_new(all_subs_df_probed, save_path)
    
    """

    #stim_duration_saccade_num_analysis_new(all_subs_df, save_path)
    #stim_duration_saccade_num_analysis_replay_new(all_subs_df, save_path)



    #poststim_first_sacc_analysis_new(all_subs_df_probed, save_path)
    #saccade_plot_replay_target(subs_dict, save_path, load=True)
    #saccade_plot_trial_visibility_no_split(subs_dict, save_path, load=True)
    #blink_plot_trial_vis(subs_dict, save_path, load=True)
    #poststim_first_fix_analysis_new(all_subs_df_probed, save_path)
    #stim_duration_saccade_num_analysis_new(all_subs_df, save_path)
    #stim_duration_saccade_num_analysis_replay_new(all_subs_df, save_path)

    #et_multivariant_model(subs_dict, save_path, load=True)
    #saccade_ciruclar_all_vis(subs_dict, save_path, DataParser.STIM_DUR, chosen_10=False, load=True)
    #saccade_ciruclar_all_vis(subs_dict, save_path, DataParser.PRE_STIM_DUR, chosen_10=False, load=True)
    #circular_sacc_analysis_thesis(subs_dict, save_path, load=True)

    calc_fixation_proportion(subs_dict, save_path)
    #fixation_plots(subs_dict, save_path, "phase4", is_tau=True, load=True)
    return
    #tau_giant_df(all_subs_df, save_path)
    blink_plot_trial_vis(subs_dict, save_path, load=True)
    saccade_ciruclar_game_vis(subs_dict, save_path, DataParser.STIM_DUR, chosen_10=False, load=True)
    saccade_ciruclar_game_vis(subs_dict, save_path, DataParser.PRE_STIM_DUR, chosen_10=False, load=True)

    saccade_ciruclar_all_vis(subs_dict, save_path, DataParser.STIM_DUR, chosen_10=False, load=True)
    saccade_ciruclar_all_vis(subs_dict, save_path, DataParser.PRE_STIM_DUR, chosen_10=False, load=True)

    return
    poststim_first_sacc_analysis_replay_new(all_subs_df_probed, save_path)

    calc_fixation_proportion(subs_dict, save_path)
    # TODO: SACCADE WANTED FIGURE
    saccade_ciruclar_game_vis(subs_dict, save_path, DataParser.STIM_DUR, chosen_10=False, load=False)
    saccade_ciruclar_game_vis(subs_dict, save_path, DataParser.PRE_STIM_DUR, chosen_10=False, load=False)

    # depends on whever they want the saccade split or not
    #saccade_plot_trial_visibility_all(subs_dict, save_path, load=True)
    return


    # saccade_ciruclar_relevance_plot(subs_dict, save_path, chosen_10=True, load=True)

    # THIS did not run for TAU due to no need for unprobed
    #saccade_plot_trial_unprobed_visibility(subs_dict, save_path, load=False)
    calc_fixation_proportion(subs_dict, save_path)
    #DECPRECATED
    #time_diff_analysis(subs_dict, save_path)
    #probe_time_delta_analysis(subs_dict, save_path)
    #saccade_vpeak_histogram(subs_dict, save_path, chosen_10=False, load=False)
    return

