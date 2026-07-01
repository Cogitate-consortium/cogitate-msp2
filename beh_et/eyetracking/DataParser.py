import os
import numpy as np
import pandas as pd
import warnings
import copy
import ET_param_manager
import datetime
from ET_param_manager import Triggers
from progress.bar import IncrementalBar as Ibar
from based_noise_blinks_detection import based_noise_blinks_detection

""" Data Parsing Module

This module includes all the methods used to extract, parse and filter data.

@authors: RonyHirsch, AbdoSharaf98
"""


# Eyelink logs' line tags
OTHER = "OTHER"
EMPTY = "EMPTY"
COMMENT = 'COMMENT'
SAMPLE = "SAMPLE"
START = "START"
END = "END"
MSG = "MSG"
EFIX = "EFIX"
ESACC = 'ESACC'
EBLINK = "EBLINK"
START_REC_MARKER = '!MODE RECORD'
TRIGGER = 'TRIGGER'  # Video Game triggers line tags

DF_REC = "dfRec"
DF_MSG = "dfMsg"
DF_FIXAT = "dfFix"
DF_SACC = "dfSacc"
DF_SACC_EK = f"{DF_SACC}EK"
DF_BLINK = "dfBlink"
DF_SAMPLES = "dfSamples"
EYE = "Eye"
EYELINK = "Eyelink"
EK = "EK"
HERSHMAN = "Hershman"
HERSHMAN_PAD = "HershmanPad"
REAL_SACC = "RealSacc"
REAL_FIX = "RealFix"
REAL_PUPIL = "RealPupil"

# data columns
ONSET = 'StimOnset'
T_START = 'tStart'
T_END = 'tEnd'
T_SAMPLE = 'tSample'
VPEAK = 'vPeak'
AMP_DEG = 'ampDeg'

# duration windows
STIM_DUR = 'StimDuration'
STIM_LOC = "stimulusLocation"
PRE_STIM_DUR = 'PreStim'
EPOCH = 'Epoch'
TRIAL = "Trial"
REPLAY = "IsReplay"
P200P500 = "P200P500"
P250P500 = "P250P500"
P100P600 = "P100P600"

GAME_PHASE = "GAME"
REPLAY_PHASE = "REPLAY"

LOC = 'Location'
VIS = 'Visibility'
WORLD_G = 'Game World'
WORLD_R = 'Replay World'
WORLD = 'World'
CATEGORY = 'Category'
IS_LEVEL_ELIMINATED = "IS_LEVEL_ELIMINATED"
SACC_DIRECTION_RAD = "sacc_direction_rad"

SAMPLING_FREQ = 'SamplingFrequency'
WINDOW_START = 'WindowStart'
WINDOW_END = 'WindowEnd'


class Error(Exception):
    pass


class InputError(Error):
    def __init__(self, msg):
        self.message = msg


def is_probed(row):
    # PROBED: IF IN GAME LOCATION ENDS WITH 1 || IF IN REPLAY, ANY STIMULUS
    value = int(row["Location"])
    if value % 10:
        return True
    if row["WorldName"].endswith("_A") or row["WorldName"].endswith("_B"):
        return True
    return False


def set_visibility(category, response):
    """
    A function mapping between a stimulus type and the stimulus response AS RECEIVED FROM THE EYELINK TRIGGER MESSAGES.
    :param category:
    :param response:
    :return:
    """
    TP = "True Positive"
    TN = "True Negative"
    FP = "False Positive"
    FN = "False Negative"
    vis_dict = {("Blank", "No"): TN, ("Blank", "Yes"): FP, ("Blank", "NoResp"): TN,  ("Blank", "Resp"): FP,
                ("Face", "No"): FN, ("Object", "No"): FN, ("Face_target", "NoResp"): FN, ("Obj_target", "NoResp"): FN,
                ("Face", "Yes"): TP, ("Object", "Yes"): TP, ("Face_target", "Resp"): TP, ("Obj_target", "Resp"): TP,
                ("Face_non_target", "Resp"): FP, ("Obj_non_target", "Resp"): FP,
                ("Face_non_target", "NoResp"): TN, ("Obj_non_target", "NoResp"): TN,
                ("Obj_target", "No"): FN, ("Object", "Resp"): TP}  # THIS CASE SHOULDN'T EVEN HAPPEN, For some reason it was spotted in SD154 only
    return vis_dict[(category, response)]


def stim_id(row):
    stim_name = row["stimulusName"]
    if row["WorldID"].isdigit():  # game
        if isinstance(stim_name, float):
            return 50  # blank
        if "SF" in stim_name:
            return int(stim_name[2:])  # face
        return int(stim_name[2:]) + 20  # object
    if row["TargetType"] == "Face":
        if isinstance(stim_name, float):
            return 150  # blank
        if "SF" in stim_name:
            return int(stim_name[2:]) + 100  # face
        return int(stim_name[2:]) + 120  # object
    # else, replay + object target
    if isinstance(stim_name, float):
        return 250  # blank
    if "SF" in stim_name:
        return int(stim_name[2:]) + 200  # face
    return int(stim_name[2:]) + 220  # object


def world_id(row):
    if row["currentLevelID"] < 100:
        return str(row["activeWorldID"] + 1)
    if 100 <= row["currentLevelID"] < 104:
        return "A"
    return "B"


def identify_missing_triggers(beh_stim, et_stim):
    # we take BEH as ground truth, as we reach this method only if BEH trials > ET trials
    ind_beh = 0
    ind_et = 0
    missing_trials = list()
    while ind_beh < beh_stim.shape[0] and ind_et < et_stim.shape[0]:
        if beh_stim.iloc[ind_beh, :].equals(et_stim.iloc[ind_et, :]):
            ind_beh += 1
            ind_et += 1
        else:  # there is a missing trial appearing in beh_stim that does not appear in et_stim
            missing_trials.append(ind_beh)
            ind_beh += 1
    # Because of SC196, also add the tail end of beh_stim
    for i in range(ind_beh, beh_stim.shape[0]):
        missing_trials.append(i)
    return missing_trials


def compare_beh_triggers(beh_trial_data, et_trial_data, sub_code, is_probed=True):
    """
    Compare the behavioral log outputs to eyetracking trigger messages to make sure that they agree on all trials'
    locations and stimulus identities. In case there is a mismatch, this is an issue that needs to be raised.
    If there is a match, then we can rely on beh_trial_data and simply add things from et_trial_data to it.
    :param beh_trial_data: output of behavioral data as it was parse in the behavioral QC from the log files
    :param et_trial_data: output of behavioral data as it was parsed by get_trial_info from eyelink trigger messages
    :param sub_code: subject code
    :return: whether they match or not
    """
    error = 0
    # create beh df to test matching to trigger
    beh_trial_data["WorldID"] = beh_trial_data.apply(lambda row: world_id(row), axis=1)
    beh_trial_data["stimID"] = beh_trial_data.apply(lambda row: stim_id(row), axis=1)
    beh_stim = beh_trial_data[["WorldID", "stimulusType", "stimID", STIM_LOC]]
    beh_stim.rename(columns={"stimulusType": "Category", STIM_LOC: "Location"}, inplace=True)
    # create trigger df to test matching to beh log trials
    et_stim = et_trial_data[["WorldID", "Category", "stimID", "Location"]].reset_index(drop=True)
    et_stim.loc[:, "Category"] = et_stim.loc[:, "Category"].replace("Blank", "None")
    et_stim.loc[:, "Category"] = et_stim.loc[:, "Category"].replace({"Face_non_target": "Face", "Face_target": "Face", "Obj_non_target": "Object", "Obj_target": "Object"}, regex=True)
    et_stim.loc[:, "WorldID"] = et_stim.loc[:, "WorldID"].replace({"World ": ""}, regex=True)
    et_stim.loc[:, "stimID"] = pd.to_numeric(et_stim["stimID"])
    comparison = et_stim.equals(beh_stim)
    # add a column in beh that marks for each trial whether it has ET data or not
    beh_trial_data.loc[:, "isTrigger"] = 1  # initialize as true

    if not comparison:
        if et_stim.shape[0] < beh_stim.shape[0]:
            if beh_stim.loc[:et_stim.shape[0] - 1, :].compare(et_stim).empty:
                # This is a case where the ET data is trimmed from the end. This means that the ET and BEH data
                # are identical if the BEH data is trimmed to include the same number of trials as the ET data
                beh_trial_data = beh_trial_data.loc[:et_stim.shape[0] - 1, :]  # trim manually
                print(f"{sub_code} ET DATA INCOMPLETE - TRIMMING BEH DATA TO MATCH")
                error = 1
                return error, beh_trial_data
            else:
                worlds = et_trial_data["WorldID"].unique().tolist()
                if (not "World A" in worlds or not "World B" in worlds) and is_probed:  # no replay data in et files
                    print("ERROR: no localizer events found in the ET data files!!!")
                    beh_trial_data.loc[(beh_trial_data["WorldID"] == "A") | (beh_trial_data["WorldID"] == "B"), "isTrigger"] = 0
                    error = 1
                    return error, beh_trial_data
                else:  # missing triggers for BEH trials; need to locate them
                    print(f"{sub_code} ISSUE WITH TRIAL MAPPING: SOME BEHAVIORAL DATA LOG TRIALS MISSING THEIR CORRESPONDING TRIGGERS")
                    missing_trial_indices = identify_missing_triggers(beh_stim, et_stim)
                    beh_trial_data.loc[missing_trial_indices, "isTrigger"] = 0  # trials missing ET triggers
                    error = 1
                    return error, beh_trial_data

        elif et_stim.shape[0] > beh_stim.shape[0]:
            if (REPLAY in beh_trial_data.columns) and (True not in beh_trial_data[REPLAY].unique().tolist()):  # no replay data in beh files
                print("ERROR: no localizer events found in the BEH data files!!!")
                error = 1
                return error, beh_trial_data
            else:
                print(f"{sub_code} ISSUE WITH TRIAL MAPPING: STIMULI TRIGGERS CONTRADICT THE BEHAVIORAL DATA LOGS: CHECK MANUALLY (B)")
                error = 1
                return error, beh_trial_data
        else:
            diff = et_stim.compare(beh_stim, keep_shape=True, keep_equal=True)  # for debugging, see what's wrong
            print(f"{sub_code} ISSUE WITH TRIAL MAPPING: STIMULI TRIGGERS CONTRADICT THE BEHAVIORAL DATA LOGS: CHECK MANUALLY (C)")
            error = 1
            return error, beh_trial_data
    else:
        print(f"{sub_code} ET triggers match BEH logs")
    return error, beh_trial_data


def set_trial_info(et_trial_data, beh_trial_data, sub_code, save_path, is_probed=True):
    """
    Following the behavioral analysis of the video game, two key changes happen to the behavioral responses:
    1. Responses might be flipped : "negative" responses with a recodgnized delayed button press might have flipped
    to positive responses.
    2. Levels might have been eliminated: levels in which subjects' response is unacceptable (e.g., no button presses
    at all during the entire level) are eliminated from further analysis.
    Thus, the ET analysis should take the corrected behavioral data as the ground truth. We first compare the stimulus
    identity and location with compare_beh_triggers to see that ET triggers matched BEH logs, then we unify the
    information to return a comprehensive, correct, trial df
    :param et_trial_data: output of behavioral data as it was parsed by get_trial_info from eyelink trigger messages
    :param beh_trial_data: output of behavioral data as it was parse in the behavioral QC from the log files
    :return: beh_trial_data with timestamps from ET triggers, after making sure the sources agree
    """

    # Check equality between the stimuli in BEH and ET:
    beh_trial_data.reset_index(inplace=True, drop=True)
    et_trial_data.reset_index(inplace=True, drop=True)
    error, beh_trial_data = compare_beh_triggers(beh_trial_data, et_trial_data, sub_code, is_probed=is_probed)
    # after making sure the dfs are equal in terms of trial information, add the ET derived timings into BEH
    # note that now beh_trial_data has an additional column for each trial, whether or not it has a corresponding ET trigger
    if is_probed:
        beh_trial_data.to_csv(os.path.join(save_path, "trial_data_full.csv"), index=False)
        # as this is ET parsing, trials that for some reason don't have ET data are not interesting, so drop them but make sure to have the correct trial numbers
        beh_trial_data = beh_trial_data.loc[beh_trial_data["isTrigger"] == 1, :].rename_axis('trialNumber').reset_index(drop=False, inplace=False)
        print(f"{beh_trial_data.shape[0]} trials have ET trigger data")
        beh_trial_data.loc[:, ONSET] = et_trial_data.loc[:, ONSET]
        beh_trial_data.loc[:, "stimShrink"] = et_trial_data.loc[:, "stimShrink"]
        beh_trial_data.loc[:, "probeOnset"] = et_trial_data.loc[:, "probeOnset"]
        beh_trial_data.rename(columns={"responseEvaluation": VIS}, inplace=True)
    else:
        # as this is ET parsing, trials that for some reason don't have ET data are not interesting, so drop them but make sure to have the correct trial numbers
        beh_trial_data = beh_trial_data.loc[beh_trial_data["isTrigger"] == 1, :].rename_axis('trialNumber').reset_index(drop=False, inplace=False)
        print(f"{beh_trial_data.shape[0]} trials have ET trigger data")
        beh_trial_data.loc[:, ONSET] = et_trial_data.loc[:, "stimShrink"] - 250
        beh_trial_data.loc[:, "stimShrink"] = et_trial_data.loc[:, "stimShrink"]
        beh_trial_data.to_csv(os.path.join(save_path, "trial_data_unprobed_full.csv"), index=False)
    return beh_trial_data


def get_trial_info(allMsgDF, include_world_0=False):  # params
    """
    This function extracts the timestamps and basic information about trials (trials are basically probed stimuli).
    The information about all stimuli (location, id, whether they are probed, AND RESPONSES TO THE PROBES) is extracted
    directly from the TRIGGERS sent to the eyelink by the video game. The mapping of these triggers is an instance of
    ET_param_manager.Triggers class.
    It outputs a dataframe in which each line is a single trial, and each column is some information about it
    (e.g., world, timestamps of beginning and end of this trial, which stimulus was in this trial etc).
    :param allMsgDF: dataframe containing all of the Eye-Tracker messages (raw data)

    :param include_world_0: whether to include practice levels (World 0) or not. Default is not, which means that they
    are not included in the output dataframe.
    :return: trialInfo
    """
    triggers = Triggers()

    # get the trigger messages
    msgDF = allMsgDF.loc[allMsgDF.text.str.startswith(TRIGGER), :].reset_index(inplace=False, drop=True)
    msgcodes = np.array([f.split(';')[1] for f in msgDF.text])  # all trigger message lines' codes
    msgDF.loc[:, 'text'] = msgcodes

    # get all the stimuli
    allStims = msgDF.loc[msgDF.text.str.contains('_'), :]  # all trigger messages indicating a stimulus ONSET have structure of stimID_stimLoc
    allStims.reset_index(inplace=True, drop=True)
    allStims[['stimID', LOC]] = allStims['text'].str.split('_', expand=True)  # gives a warning, but does the work

    # get all stim probe status
    allStims.loc[:, 'Probe'] = allStims.apply(lambda row: is_probed(row), axis=1)

    # get all stimulus locations
    allStims.loc[:, LOC] = np.array([triggers.Location[f] for f in allStims.Location])

    # get all stimulus categories
    allStims.loc[:, CATEGORY] = np.array([triggers.Stimuli[int(f)] for f in allStims['stimID']])

    # get all the timestamps when the stimulus starts shrinking
    stim_onsets = msgDF.loc[msgDF.text.str.contains('_'), :].index
    stim_shrinks_from_onsets = np.array(stim_onsets) + 1  # +1 is to get the AnimationPeakEnd of a stim, right after its onset trigger
    stim_shrinks_animationpeakend = msgDF.loc[msgDF.text == triggers.AnimationPeakEnd, :].index
    stim_shrinks_animationpeakend = np.array(stim_shrinks_animationpeakend)
    if (len(stim_shrinks_animationpeakend) == len(stim_shrinks_from_onsets)) and \
            ((stim_shrinks_animationpeakend == stim_shrinks_from_onsets).all()):  # all good, we can rely on one of them arbitrarily
        stim_sh = stim_shrinks_animationpeakend
    else:
        if len(stim_shrinks_animationpeakend) > len(stim_shrinks_from_onsets):
            print("More AnimationPeakEnd events than stimuli, meaning this trigger was on for more than stim shrinkage")
            stim_shrinks = list()
            for ind in stim_shrinks_from_onsets:
                if msgDF.iloc[ind]['text'] != triggers.AnimationPeakEnd:
                    print("stimulus shrinkage timestampinconclusive")
                    stim_shrinks.append(np.nan)
                else:
                    stim_shrinks.append(ind)
            stim_sh = stim_shrinks
        elif len(stim_shrinks_animationpeakend) == len(stim_shrinks_from_onsets) and len(stim_shrinks_animationpeakend) == len(stim_onsets):
            print("relying on AnimationPeakEnd as stim shrinkage timestamp")
            stim_sh = stim_shrinks_animationpeakend
        else:
            """a possible case where len(stim_shrinks_animationpeakend) < len(stim_shrinks_from_onsets) is that RIGHT 
            after a stimulus the level ended (so that there is NO shrinkage of the stimulus, and the expected message
            right after the stimulus one is a 'levelEnd' one """
            if len(stim_shrinks_animationpeakend) < len(stim_shrinks_from_onsets):
                missing_shrinks = [x for x in stim_shrinks_from_onsets if x not in stim_shrinks_animationpeakend]
                for elem in missing_shrinks:  # for each animationPeakEnd that 'should've' been there, we'll check
                    # what's actually there instead
                    actual_msg = msgDF.loc[elem, 'text']
                    if actual_msg == triggers.LevelEnd:
                        print("1 Stimulus missing shrink time since it was immediately followed by end-of-level")
                    # instead of putting the time of the SHRINK which we don't have, put the time of stim onset
                    stim_shrinks_from_onsets = [x if x != elem else x-1 for x in stim_shrinks_from_onsets]
                stim_sh = stim_shrinks_from_onsets

    allStims.loc[:, "stimShrink"] = msgDF.loc[stim_sh, 'time'].tolist()
    # now, due to the last edge-case (len(stim_shrinks_animationpeakend) < len(stim_shrinks_from_onsets)),
    # we check lines where stimShrink == time (stim Onset), and turn them to nans to make sure we know there wasn't data
    allStims.loc[allStims["stimShrink"] == allStims['time'], "stimShrink"] = np.nan

    # get all probe onsets
    probe_inds_game = msgDF.loc[msgDF.text == triggers.ProbeOnset, :].index  # in game, identify by probe
    msgDF_replay = msgDF[~msgDF["WorldID"].isin([WORLD + f" {i}" for i in range(5)])]
    msgDF_replay = msgDF_replay[msgDF_replay["WorldID"].isin([WORLD + f" A", WORLD + f" B"])]  # THIS IS NOT A DUPLICATION:
    # apparently due to some kinks (subject SD188), removing all worlds [0, 4] does not leave us with worlds A and B.
    # For some reason, this subject also had a "World E", which is definitely not a replay level
    probe_inds_replay = msgDF_replay.loc[msgDF_replay.text.str.contains('_'), :].index  # in replay, probe onset=stim onset
    probe_game_timestamps = msgDF.iloc[probe_inds_game]['time'].tolist()
    probe_replay_timestamps = msgDF.iloc[probe_inds_replay]['time'].tolist() # this acts on msgDF (not msgDF_replay) as msgDF has all messages (both game and replay)
    all_probe_onsets = probe_game_timestamps + probe_replay_timestamps

    # get all probe responses
    probeResponses_game = msgDF.loc[np.array(probe_inds_game) + 1, :]  # +1 is to get the RESPONSE to probe, right after it
    msgDF["ReplayResp"] = np.array([triggers.ReplayResponse[f] if f in triggers.ReplayResponse else None for f in msgDF.text])
    probeReplayInds = np.array(msgDF[msgDF['ReplayResp'].notnull()].index.tolist())
    probeResponses_replay = msgDF.loc[probeReplayInds, :]
    probeResponses = pd.concat([probeResponses_game, probeResponses_replay])
    probeResponses = probeResponses.sort_values(by='time', axis=0)
    probeResponses.drop('ReplayResp', axis=1, inplace=True)
    probeResponseTexts = list()
    for f in probeResponses.text:
        if f in triggers.Response:
            probeResponseTexts.append(triggers.Response[f])
        elif f in triggers.ReplayResponse:
            probeResponseTexts.append(triggers.ReplayResponse[f])
        else:  # this could happen only for probeResponses_game:
            #In replay, we do a direct extraction of response but in game we 'infer it' based on expecting to have the
            # response sample always be +1 from the probe sample. If this is an MRI subject OR something
            # went wrong in a GAME LEVEL -> then, there was no response for this probe (even though it's GAME)
            probeResponseTexts.append("NoResp")
    probeResponses['Response'] = probeResponseTexts

    # get only probed stimuli
    probedStimuli = allStims.loc[allStims.Probe, :]
    probedStimuli.reset_index(inplace=True, drop=True)
    probedStimuli = probedStimuli.sort_values(by='time', axis=0)
    unprobedStimuli = allStims.loc[~allStims.Probe, :]
    unprobedStimuli.reset_index(inplace=True, drop=True)
    unprobedStimuli = unprobedStimuli.sort_values(by='time', axis=0)


    # find duplicates: meaning, a completely identical stimulus rows which differ ONLY in their timestamp. This means
    # the same stimulus (i.e stimID+World+Location etc) was either presented twice or, the game was stopped and restarted
    # and such.
    if probedStimuli.shape[0] > len(all_probe_onsets):
        print("Found invalid probe stimulus entries: probedStimuli > all_probe_onsets")
        # A stimulus APPEARED and was supposed to be probed. BUT it appeared twice
        # (AS A "PROBED" STIM; 2 probedStimuli lines) and probed only once (only 1 all_probe_onsets entry).
        while probedStimuli.shape[0] > len(all_probe_onsets):  # treat ALL these cases, one by one
            found_problem = False
            temp_all_probe_onsets = all_probe_onsets.copy()
            temp_all_probe_onsets.extend([0] * (probedStimuli.shape[0] - len(all_probe_onsets)))
            probed_stim_time_list = list(probedStimuli["time"])
            # compare timestamps in the same indices to see the differences
            time_diff_list = [int(probed_stim_time_list[i]) - int(temp_all_probe_onsets[i]) for i in range(len(temp_all_probe_onsets))]
            for i in range(len(probed_stim_time_list)):  # for each probedStimuli entry
                time_diff = abs(time_diff_list[i])
                if time_diff >= ET_param_manager.STIM_DUR * 2:
                    # if the difference between the time of one of the duplicate probes and the corresponding (only)
                    # all_probe_onsets entry is LARGER than stimulus duration - Need to remove, this is the problem
                    found_problem = True
                    print(f"Found invalid entry: removing probedStimuli[{i}]")
                    probedStimuli.drop(i, axis=0, inplace=True)
                    probedStimuli.reset_index(inplace=True, drop=True)  # reset so we could continue doing so for all invalid entries
                    break
            if not found_problem:
                print(f"Did not find invalid entry: amt mismatch w/o a clear duplicate entry. Inspect subject manually")
                return None

    probedStimuli['probeOnset'] = all_probe_onsets
    # Special case for SA174 - add missing response
    if len(all_probe_onsets) > len(probeResponseTexts) and len(all_probe_onsets) == len(probeResponseTexts) + 1:
        probedStimuli["type"] = 1
        probeResponses["type"] = 2
        tmp_df = pd.concat([probedStimuli, probeResponses]).sort_values(by=["time"]).reset_index()
        dup_location = np.arange(tmp_df.shape[0])[(tmp_df['type'].diff() == 0).values].tolist()
        probeResponseTexts.insert(dup_location[0]//2, "NoResp")
    probedStimuli['Response'] = probeResponseTexts

    # remove those whose response in the GAME is no answer (meaning, when forced to answer they timed out)
    no_answer_inds = probedStimuli.loc[probedStimuli.Response == 'NoAnswer', :].index
    # AND the ones where errors in the game led to no response being recorded for the stimulus
    probe_game = probedStimuli[(probedStimuli['WorldID'] != 'World A') & (probedStimuli['WorldID'] != 'World B')]
    no_resp_in_game_inds = probe_game.loc[probe_game.Response == 'NoResp', :].index
    all_inds = list(no_answer_inds) + list(no_resp_in_game_inds)
    probedStimuli = probedStimuli.drop(all_inds)
    probedStimuli.reset_index(inplace=True, drop=True)

    # translate Response + Category columns to TP (seen)/ FP / TN / FN (unseen)
    probedStimuli[VIS] = probedStimuli.apply(lambda row: set_visibility(row[CATEGORY], row['Response']), axis=1)

    trial_info = probedStimuli[['WorldID', 'WorldName', 'stimID', CATEGORY, LOC, 'time', 'stimShrink', 'probeOnset', 'Response', VIS]]
    trial_info.rename(columns={'time': ONSET}, inplace=True)
    trial_info_unprobed = unprobedStimuli[['WorldID', 'WorldName', 'stimID', CATEGORY, LOC, 'stimShrink']]

    # take out the practice world (World 0)  - W/O RESETTING THE INDEX
    if not include_world_0:
        trial_info = trial_info[trial_info['WorldID'] != "World 0"]
        trial_info_unprobed = trial_info_unprobed[trial_info_unprobed['WorldID'] != "World 0"]

    return trial_info, trial_info_unprobed


def SequenceEyeData(et_sample_data, trial_info, params):
    """
    This function sequences the eye tracking samples into trials defined by EPOCH_START and EPOCH_END.
    Meaning, based on the trialInfo dataframe (which given information about the start and end of each trial),
    the current function splits the ET data samples df into a list of trials: trialData.
    Each element in trial data represents a single trial (for 4 worlds + 8 replay levels we'll have 400 trials=400
    elements). Each trial in this list is a dataframe, containing all the samples in this trial.
    :param et_sample_data: dataframe containing all ET "SAMPLE" lines (all samples in the experiment)
    :param trial_info: dataframe containing all the TRIAL information in the exp (including timestamps of trial beginning
    and end)
    :param params: the subjects' parameter set, including things like the sampling frequency of the Eye Tracker.
    :return: (1) trialData = a list of dataframes of samples. Each df is a single trial's sample collection.
    (2) trialInfo = the same dataframe as the input one, with extra columns.
    """
    # get the trial window start and end indices for each trial
    stimOnset = np.array(trial_info[ONSET])  # column of stimuli onsets for each trial (=probed stimulus)
    onsetInds = np.array([np.where(et_sample_data.tSample == onset)[0][0] for onset in stimOnset])  # index of eye sample which matches each stimulus onset
    # index of eye sample which matches the beginning of the epoch
    epoch_start_inds = list(np.ceil(onsetInds - (ET_param_manager.EPOCH_START / 1000) * params[SAMPLING_FREQ]).astype(int))
    # index of eye sample which matches the end of the epoch: EPOCH ENDS WITH RELATION TO STIMULUS ONSET!!!
    epoch_end_inds = list((np.ceil(onsetInds + (ET_param_manager.EPOCH_END / 1000) * params[SAMPLING_FREQ]) + 1).astype(int))
    # index of eye sample which matches the prestimulus beginning (end is stimonset)
    prestim_start_inds = list(np.ceil(onsetInds - (ET_param_manager.PRE_STIM_DUR / 1000) * params[SAMPLING_FREQ]).astype(int))
    # index of eye sample which matches the stimulus duration end (beginning is stimonset)
    stimdur_end_inds = list((np.ceil(onsetInds + (ET_param_manager.STIM_DUR / 1000) * params[SAMPLING_FREQ]) + 1).astype(int))

    epoch_begin = np.array(et_sample_data.iloc[epoch_start_inds, 0])
    epoch_end = np.array(et_sample_data.iloc[epoch_end_inds, 0])
    prestim_begin = np.array(et_sample_data.iloc[prestim_start_inds, 0])  # prestim end = stim Onset
    stimdur_end = np.array(et_sample_data.iloc[stimdur_end_inds, 0])  # stimdur begin = stim Onset

    # initialize outputs
    epoch_data = [None] * len(stimOnset)
    epoch_info = trial_info.copy(deep=True)

    epoch_info['EpochWindowStart'] = epoch_begin
    epoch_info['EpochWindowEnd'] = epoch_end
    epoch_info[PRE_STIM_DUR+"WindowStart"] = prestim_begin
    epoch_info[STIM_DUR + "WindowEnd"] = stimdur_end

    print('num trials = ' + str(len(stimOnset)))

    # loop through trials
    # start tracking progress
    bar = Ibar('Sequencing', max=len(stimOnset), suffix='%(percent)d%%')
    for trial in range(0, len(stimOnset)):  # for each (probed) stimulus

        # get the indices for the start and end with which to index the data array
        stIdx = epoch_start_inds[trial]
        endIdx = epoch_end_inds[trial]

        # get the trial's data
        epoch_data[trial] = et_sample_data.iloc[stIdx:endIdx, :]
        epoch_data[trial].reset_index(drop=True, inplace=True)

        # raise an error if the trial data is empty
        if epoch_data[trial].empty:
            raise InputError(f'Trial no. {trial}: Epoch has no samples, check the raw data')

        bar.next()
    bar.finish()

    epoch_info = epoch_info.reset_index(drop=True, inplace=False)  # get the first non-world-0 index

    return epoch_data, epoch_info


def GetBlinks(trialData, TrialInfo, blinkDF, params, method='tracker'):
    """
    This function takes in the sequenced trialData (a list, in which each element is a dataframe containing ET samples
    for a single trial) and gets the blinks in each trial.
    Blink extraction can take place in 1 of the following 2 methods, depending on the "method" parameter:
    - method="tracker": identify blinks based on the eye tracker messages
    - method=hershman: identify blinks based on the blink detection script and algorithm based on
    Hershman et al. 2018 (https://osf.io/jyz43/)

    :param trialData: a list of dataframes where each element corresponds all the samples of a single trial. This
    should be the output of the SequenceEyeData.
    :param TrialInfo: a dataframe containing all trial info (should be the output of SequenceEyeData)
    :param blinkDF:  the blink informtion extracted from the eye tracker messages.
    :param params: the parameters dict of the subject. Should be the output of InitParams/UpdateParams
    :param method: which method to use to detect blinks
    ('tracker': use tracker messages, 'hershman': based_noise fxn)

    :return: TrialDataNoBlinks: an updated version of trialData where the blinks have been removed: meaning,
    a list of dataframes where each element corresponds all the samples of a single trial - W/O BLINKS.
    :return: Blinks: a list of dataframes where each entry corresponds to a trial and holds the information about
    the starting time/index and ending time/index of each blink. the number of rows in a dataframe is the number of
    blinks in that trial.
    :return BlinkArray: an array with rows of zeros and ones with ones at the indices of blinks. Each row in
    the array is a trial and the number of columns is the total time points or number of samples associated with a
    trial.
    """

    # initialize the no blinks trial data
    TrialDataNoBlinks = copy.deepcopy(trialData)

    # initialize the list of blinks
    blink_list = [None] * len(trialData)

    # initialize the blink array
    blink_array = np.zeros([trialData[0].shape[0], len(trialData)])

    # grab the start and end times of blinks corresponding to the eye of interest
    blinkSt = np.array(blinkDF.loc[blinkDF['eye'] == params['Eye'], T_START])
    blinkEnd = np.array(blinkDF.loc[blinkDF['eye'] == params['Eye'], T_END])

    # loop through the trials
    for trial in range(0, len(trialData)):

        if method == 'tracker':

            # get the indices of the blinks within this trial
            invalidBlinkInds = (blinkSt > TrialInfo.EpochWindowEnd[trial]) | (blinkEnd <
                                                                              TrialInfo.EpochWindowStart[trial])
            trialBlinkSt = blinkSt[~invalidBlinkInds]
            trialBlinkEnd = blinkEnd[~invalidBlinkInds]
            trialBlinkStInd = np.zeros(len(trialBlinkSt))
            trialBlinkEndInd = np.zeros(len(trialBlinkEnd))

            # loop through the blinks and get their start and end indices relative to the trial's timescale
            for blnk in range(0, len(trialBlinkSt)):
                blnkst = trialBlinkSt[blnk]
                blnkend = trialBlinkEnd[blnk]

                if blnkst < TrialInfo.EpochWindowStart[trial]:
                    stIndx = 0
                else:
                    stIndx = np.argmin(np.abs(trialData[trial].tSample - blnkst))

                if blnkend > TrialInfo.EpochWindowEnd[trial]:
                    endIndx = len(trialData[trial]) - 1
                else:
                    endIndx = np.argmin(np.abs(trialData[trial].tSample - blnkend))

                trialBlinkStInd[blnk] = stIndx
                trialBlinkEndInd[blnk] = endIndx

                # replace with nans
                TrialDataNoBlinks[trial].iloc[int(stIndx):int(endIndx) + 1, 1:] = np.nan

                # add 1s in the blink array where there is a blink
                blink_array[int(stIndx):int(endIndx) + 1, trial] = 1

            # put into the blinks array
            blink_list[trial] = pd.DataFrame({ONSET: trialBlinkStInd, 'Offset': trialBlinkEndInd})

        elif method == "hershman":

            # get the pupil size array for the eye of interest
            pupilArray = np.array(TrialDataNoBlinks[trial].LPupil) if params['Eye'] == 'L' else \
                np.array(TrialDataNoBlinks[trial].RPupil)

            # replace nan values with zeros
            pupilArray[np.isnan(pupilArray)] = 0

            # detect blinks
            trialBlinks = based_noise_blinks_detection(pupilArray, params['SamplingFrequency'])

            # put into the Blinks array
            blink_list[trial] = pd.DataFrame({ONSET: trialBlinks['blink_onset'], 'Offset': trialBlinks['blink_offset']})

            # loop through the blinks and replace them
            for bOn, bOff in zip(trialBlinks['blink_onset'], trialBlinks['blink_offset']):
                # replace with nans
                TrialDataNoBlinks[trial].iloc[int(bOn):int(bOff) + 1, :] = np.nan

                # add 1s in the blink array where there is a blink
                blink_array[int(bOn):int(bOff) + 1, trial] = 1

        else:
            warnings.warn(r"Invalid blink detection methods. Supported methods are: 'tracker' and 'hershman'")

    return TrialDataNoBlinks, blink_list, blink_array.T


def GetEuclideanDistance(data, trialData_no_blinks, trialInfo, params):
    """
    This function gets the euclidean distance from (1) the fixation cross and (2) the stimulus. Based on that, it
    calculates the fixation proportion and mean distance for each condition.

    :param data: dataframe containing all ET "SAMPLE" lines (all samples) - meaning, this is all the sample data
    from the eye tracker messages
    :param trialData_no_blinks: a list of dfs where each element corresponds all the samples of a single trial
    WITHOUT THE BLINKS (meaning, AFTER filtering out of blinks).
    :param trialInfo: a df containing all trial info (should be the output of SequenceEyeData)
    :param params: the parameters dict of the subject. Should be the output of InitParams/UpdateParams

    :return:
    """
    trial_info_res = pd.DataFrame()
    trial_info_res = pd.concat([trial_info_res, trialInfo])

    refLocs = ['StimReference', 'CenterReference']
    timePeriods = [PRE_STIM_DUR, STIM_DUR, EPOCH]  # the analysis time periods relative to stimulus onset

    xcol = 'LX' if params['Eye'] == 'L' else 'RX'
    ycol = 'LY' if params['Eye'] == 'L' else 'RY'

    # initialize the onset time/index for all trials
    trialOnset = np.array(trialInfo[ONSET])

    # the starting inds for prestim
    # DIV by 1000 is because ET_param_manager.PRE_STIM_DUR is in MILLISECONDS and sampling frequency is the
    # number of samples per SECOND. So this is to convert from ms to sec, which then is converted to number of samples
    prestim_onset = np.array(trialInfo[PRE_STIM_DUR+"WindowStart"])

    # the ending indcs for during stim
    stimdur_offset = np.array(trialInfo[STIM_DUR + "WindowEnd"])

    # area used to determine fixation bounds for center
    refAngle = ET_param_manager.FIX_REF_ANGLE_RADIUS  # the radius of ° of visual angle which was defined as fixation stability
    fixAreaCenter = refAngle / params['DegreesPerPix']

    # get the stimulus locations and coordinates for all trials
    allStimLocs = trialInfo.Location
    allStimCoords = np.array([params['StimulusCoords'][n] for n in allStimLocs])
    allFixAreas = np.array([refAngle / params['DegreesPerPix']] * len(list(allStimLocs)))  # Fixation area in PIXELS

    centerCoords = params['ScreenCenter']

    # initialize outputs
    MeanFixationDist = dict.fromkeys(timePeriods)
    FixationProportion = dict.fromkeys(timePeriods)

    for k in MeanFixationDist.keys():
        MeanFixationDist[k] = dict.fromkeys(refLocs)
        FixationProportion[k] = dict.fromkeys(refLocs)
        for r in refLocs:
            MeanFixationDist[k][r] = []
            FixationProportion[k][r] = []

    # do the epochs first since we have sequenced data
    for tr in range(0, len(trialData_no_blinks)):  # loop over the trials
        epoch = trialData_no_blinks[tr]

        for ky in timePeriods:
            # get the relevant gaze data
            if ky == PRE_STIM_DUR:
                gaze = epoch.iloc[epoch[epoch.tSample == prestim_onset[tr]].index[0]:
                                  epoch[epoch.tSample == trialOnset[tr]].index[0]]
            elif ky == STIM_DUR:
                gaze = epoch.iloc[epoch[epoch.tSample == trialOnset[tr]].index[0]:
                                  epoch[epoch.tSample == stimdur_offset[tr]].index[0]]
            else:
                gaze = epoch

            gazeX = gaze[xcol]
            gazeY = gaze[ycol]

            for rf in refLocs:  # for each reference type
                coords = allStimCoords[tr, :] if rf == 'StimReference' else centerCoords
                fixArea = allFixAreas[tr] if rf == 'StimReference' else fixAreaCenter

                # calculate mean distance from reference, and fixation proportion
                distance = np.sqrt(((gazeX - coords[0]) ** 2) + ((gazeY - coords[1]) ** 2))  # dist in PIXELS!!!
                distance_in_degrees = distance * params['DegreesPerPix']  # convert to degrees
                meanDist = np.nanmean(distance_in_degrees)  # mean dists from target IN DEGREEES
                fixating = np.zeros(len(distance))  # BINARY array indicating: 1 = fixating (in area), 0 = not
                # NOTE: we assume fixArea is defined by RADIUS already so we DON'T divide by 2. If this size is DIAMETER then this needs to be divided by 2
                fixating[distance <= fixArea] = 1  # fixation area in pixels, distance in pixels
                fixProp = np.nansum(fixating) / len(fixating)

                # store
                # MEAN DISTANCES FROM CENTER OF TARGET (STIMULUS/FIX) in DEGREES
                trial_info_res.at[tr, 'DistFrom' + rf + "_" + ky] = meanDist
                # % of timestamps where gaze is within the TARGET RADIUS, 0 = OUTSIDE, 1 = INSIDE
                trial_info_res.at[tr, 'FixProp' + rf + "_" + ky] = fixProp

                if ky == EPOCH:  # when we calculate this for the entire epoch (all samples in trial) - save everything
                    trialData_no_blinks[tr]['DistFrom' + rf] = distance_in_degrees
                    trialData_no_blinks[tr]['IsFixating' + rf] = fixating

    return trialData_no_blinks, trial_info_res


def ParseEyeLinkAsc(elFilename, last_end_time, total_prev_diff):
    """
    This method reads in a single eyelink data file in an .asc file format, and produces readable dataframes for further
    analysis.
    :param elFilename: path to the eyelink data file
    :return: res_dict, which contains:
     -dfRec contains information about recording periods (often trials)
     -dfMsg contains information about messages (usually sent from stimulus software)
     -dfFix contains information about fixations
     -dfSacc contains information about saccades
     -dfBlink contains information about blinks
     -dfSamples contains information about individual samples
    """

    # Read in EyeLink file
    print('Reading in EyeLink file %s' % elFilename)
    f = open(elFilename, 'r')
    fileTxt0 = f.read().splitlines(True)  # split into lines
    fileTxt0 = list(filter(None, fileTxt0))  # remove emptys
    fileTxt0 = np.array(fileTxt0)  # concert to np array for simpler indexing
    f.close()

    # Separate lines into samples and messages
    print('Sorting lines')
    nLines = len(fileTxt0)
    lineType = np.array([OTHER] * nLines, dtype='object')
    iStartRec = list()
    for iLine in range(nLines):
        if fileTxt0[iLine] == "**\n" or fileTxt0[iLine] == "\n":
            lineType[iLine] = EMPTY
        elif fileTxt0[iLine].startswith('*') or fileTxt0[iLine].startswith('>>>>>'):
            lineType[iLine] = COMMENT
        elif bool(len(fileTxt0[iLine][0])) and fileTxt0[iLine][0].isdigit():
            fileTxt0[iLine] = fileTxt0[iLine].replace('.\t', 'NaN\t')
            lineType[iLine] = SAMPLE
        else:  # the type of this line is defined by the first string in the line itself (e.g. START, MSG)
            lineType[iLine] = fileTxt0[iLine].split()[0]
        if START in fileTxt0[iLine]:
            # from EyeLink Programmers Guide: "The "START" line and several following lines mark the start of recording, and encode the recording conditions for the trial."
            iStartRec.append(iLine + 1)

    iStartRec = iStartRec[0]

    # ===== PARSE EYELINK FILE ===== #
    # Trials
    print('Parsing recording markers')
    iNotStart = np.nonzero(lineType != START)[0]
    dfRecStart = pd.read_csv(elFilename, skiprows=iNotStart, header=None, delim_whitespace=True, usecols=[1])
    dfRecStart.columns = [T_START]
    iNotEnd = np.nonzero(lineType != END)[0]
    """
    END lines mark the end of a block of data. The two values following the "RES" keyword are the average resolution
    for the block: if samples are present, it is computed from samples, else it summarizes any resolution data in the
    events. Note that resolution data may be missing: this is represented by a dot (".") instead of a number for the
    resolution.
    """
    dfRecEnd = pd.read_csv(elFilename, skiprows=iNotEnd, header=None, delim_whitespace=True, usecols=[1, 5, 6])
    dfRecEnd.columns = [T_END, 'xRes', 'yRes']
    # combine trial info
    dfRec = pd.concat([dfRecStart, dfRecEnd], axis=1)
    nRec = dfRec.shape[0]
    print('%d recording periods found.' % nRec)

    # Import Messages
    print('Parsing stimulus messages')
    iMsg = np.nonzero(lineType == MSG)[0]
    # set up
    tMsg = []
    txtMsg = []
    for i in range(len(iMsg)):
        # separate MSG prefix and timestamp from rest of message
        info = fileTxt0[iMsg[i]].split()
        # extract info
        tMsg.append(int(info[1]))
        txtMsg.append(' '.join(info[2:]))
    """
    Convert dict to dataframe:
    The "MSG"s in the experiment's Ascii file are of 2 types: at the beginning of the recroding there are a lot of
    messages from EYELINK about the parameters of the ET and so on. Then, After the debugging ends ("---DEBUG END---")
    And the actual experiment starts, The video game SENDS TRIGGERS that are written as "MSG" lines for all types of
    events (which are coded in the Triggers class)
    """
    dfMsg = pd.DataFrame({'time': tMsg, 'text': txtMsg})

    # Import Fixations
    print('Parsing fixations')
    # From Eyelink Programmer's guide: "Fixation end events ("EFIX") are read by asc_read_efix() which fills the
    # variable a_efix with the start and end times, and average gaze position, pupil size,"
    # the information is: eye, start time, end time, duration, X position, Y position, pupil
    iNotEfix = np.nonzero(lineType != EFIX)[0]
    try:
        dfFix = pd.read_csv(elFilename, skiprows=iNotEfix, header=None, delim_whitespace=True, usecols=range(1, 8))
        dfFix.columns = ['eye', T_START, T_END, 'duration', 'xAvg', 'yAvg', 'pupilAvg']
    except Exception:  # meaning, NO FIXATIONS IN THIS FILE AT ALL (i.e., if we were to skip all iNotFIx rows, we were to be left with nothing)
        print(f"No fixations in {elFilename}")
        dfFix = pd.DataFrame(columns=['eye', T_START, T_END, 'duration', 'xAvg', 'yAvg', 'pupilAvg'])

    # Saccades
    print('Parsing saccades')
    # From Eyelink Programmer's guide: "Saccade end events ("ESACC") are read by asc_read_esacc() which fills the
    # variable a_esacc with the start and end times, start and end gaze position, duration, amplitude, and peak velocity."
    # the information is: eye, start time, end time, duration IN MILLISECONDS, start X position, start Y position,
    # end X position, end Y position, amplitude in DEGREES, peak velocity in DEGREES PER SECOND.
    # The total visual angle covered in the saccade is reported by the 'amplitude' parameter,
    # which can be divided by (<dur>/1000) to obtain the average velocity.
    iNotEsacc = np.nonzero(lineType != ESACC)[0]
    dfSacc = pd.read_csv(elFilename, skiprows=iNotEsacc, header=None, delim_whitespace=True, usecols=range(1, 11))
    dfSacc.columns = ['eye', T_START, T_END, 'duration', 'xStart', 'yStart', 'xEnd', 'yEnd', AMP_DEG, VPEAK]

    # Blinks
    print('Parsing blinks')
    # From Eyelink Programmer's guide:
    # The STARTBLINK and ENDBLINK events bracket parts of the eye-position data where the pupil size is very small,
    # or the pupil in the camera image is missing or severely distorted by eyelid occlusion.
    # Only the time of the start and end of the blink are recorded." (4.5.3.5 Blinks, Eyelink 1000 Plus user manual)
    # "Blink end events ("EBLINK") mark the reappearance of the eye pupil. These are
    # read by asc_read_-eblink() which fills the variable a_eblink with the start and end times, and duration. Blink
    # events may be used to label the next "ESACC" event as being part of a blink and not a true saccade."
    # more from EDF2ASC documentation: Blinks are always embedded in saccades, caused by artifactual motion as the
    # eyelids progressively occlude the pupil of the eye. Such artifacts are best eliminated by labeling an
    # SSACC...ESACC pair with one or more SBLINK events between them as a blink, not a saccade. The data contained in
    # the ESACC event will be inaccurate in this case, but the "tStart", "tEnd", and "duration" data will be accurate.
    # It is also useful to eliminate any short (less than 120 millisecond duration) fixations that precede or follow
    # a blink. These may be artificial or be corrupted by the blink.
    # right now we're just parsing everything so order doesn't matter
    iNotEblink = np.nonzero(lineType != EBLINK)[0]
    dfBlink = pd.read_csv(elFilename, skiprows=iNotEblink, header=None, delim_whitespace=True, usecols=range(1, 5))
    dfBlink.columns = ['eye', T_START, T_END, 'duration']

    # determine sample columns based on eyes recorded in file
    #eyesInFile = np.unique(dfFix.eye)
    eyesInFile = dfMsg.iloc[0, 1].split()[-1]
    if len(eyesInFile) == 2:
        print('binocular data detected.')
        cols = [T_SAMPLE, 'LX', 'LY', 'LPupil', 'RX', 'RY', 'RPupil']
    else:
        eye = eyesInFile
        print(f"monocular data detected {eye}")
        cols = [T_SAMPLE, f"{eye}X", f"{eye}Y", f"{eye}Pupil"]
    # Import samples
    print('Parsing samples')
    iNotSample = np.nonzero(np.logical_or(lineType != SAMPLE, np.arange(nLines) < iStartRec))[0]
    dfSamples = pd.read_csv(elFilename, skiprows=iNotSample, header=None, delim_whitespace=True,
                            usecols=range(0, len(cols)))
    dfSamples.columns = cols
    # Convert values to numbers
    for eye in ['L', 'R']:
        if eye in eyesInFile:
            dfSamples[f"{eye}X"] = pd.to_numeric(dfSamples[f"{eye}X"], errors='coerce')
            dfSamples[f"{eye}Y"] = pd.to_numeric(dfSamples[f"{eye}Y"], errors='coerce')
            dfSamples[f"{eye}Pupil"] = pd.to_numeric(dfSamples[f"{eye}Pupil"], errors='coerce')
        else:
            dfSamples[f"{eye}X"] = np.nan
            dfSamples[f"{eye}Y"] = np.nan
            dfSamples[f"{eye}Pupil"] = np.nan

    # These variables are used in the case of a new file which starts BEFORE an older file.
    # In such a case, we need to add the endtime to each file we read from now on.
    is_problematic = False
    next_last_end_time = dfMsg.loc[dfMsg.shape[0] - 1, "time"]

    if dfMsg.loc[0, "time"] < last_end_time:
        is_problematic = True
        total_prev_diff += last_end_time
    # This means that we will add total_prev_diff to each of the dfs.
    # In a regular case, total_prev_diff will stay 0.
    dfRec[T_START] = dfRec[T_START] + total_prev_diff
    dfRec[T_END] = dfRec[T_END] + total_prev_diff

    dfMsg["time"] = dfMsg["time"] + total_prev_diff

    dfFix[T_START] = dfFix[T_START] + total_prev_diff
    dfFix[T_END] = dfFix[T_END] + total_prev_diff

    dfSacc[T_START] = dfSacc[T_START] + total_prev_diff
    dfSacc[T_END] = dfSacc[T_END] + total_prev_diff

    dfBlink[T_START] = dfBlink[T_START] + total_prev_diff
    dfBlink[T_END] = dfBlink[T_END] + total_prev_diff

    dfSamples[T_SAMPLE] = dfSamples[T_SAMPLE] + total_prev_diff

    res_dict = {DF_REC: dfRec, DF_MSG: dfMsg, DF_FIXAT: dfFix, DF_SACC: dfSacc,
                DF_BLINK: dfBlink, DF_SAMPLES: dfSamples}  # EYE: sub_eye
    # Return new compilation dataframe
    return res_dict, is_problematic, next_last_end_time


def check_timings(et_data_dict):
    """
    Extracts from the MSG (=trigger) lines of Eyelink logs the timestamp that Unity estimated in Eyelink for all events.
    Meaning, in a dgMSG , the first column is the Eyelink timestamp, and the last column (part of the message Unity sent
    about an event) is the ESTIMATED Eyelink timestamp.
    We then calculated the difference in all messages between the Real Eyelink timestamp and the Estimated Eyelink
    timestamp.
    RIGHT NOW DOES ***NOTHING*** ELSE
    """
    allMsgDF = et_data_dict[DF_MSG]
    # get the trigger messages
    msgDF = allMsgDF.loc[allMsgDF.text.str.startswith(TRIGGER), :].reset_index(inplace=False, drop=True)

    # Steps 1 and 2 (not exactly, but the information we need)
    message_eyelink_times = np.array([int(f.split(';')[-1]) for f in msgDF.text])  # Unity's estimated Eyelink TS
    eyelink_times = np.array(msgDF["time"])  # real Eyelink TS
    message_unity_times = np.array([int(f.split(';')[-2]) for f in msgDF.text])  # real Unity's TS

    eyelink_message_diff = eyelink_times - message_eyelink_times # real Eyelink time minus estimated Eyelink time
    eyelink_message_unity_diff = eyelink_times - message_unity_times  # real Eyelink time minus Unity time
    norm_eyelink_message_unity_diff = eyelink_message_unity_diff - min(eyelink_message_unity_diff)
    result = pd.DataFrame.from_dict({"Eyelink_minus_estimated": eyelink_message_diff, "Eyelink_minus_Unity_normalized": norm_eyelink_message_unity_diff,
                                     "Unity_times": message_unity_times, "Eyelink_times": eyelink_times, "Eyelink_estimated_times": message_eyelink_times})

    return result


def et_data_mark_Eyelink(et_data_dict):
    """
    Mark blinks AS THEY APPEAR IN EYELINK (i.e., periods of missing data // Eylink-calculated blinks).
    The code  labels each sample in the samples dataframe as a EYELINK fixation or EYELINK saccade or
    EYELINK blink if it is one of these. - ALL SAMPLES MUST BE EITHER A BLINK, A FIXATION, OR A SACCADE.
    IF A SAMPLE IS NOT A BLINK, OR A SACCADE - IT MUST BE A FIXATION.

    This is based on Eyelink guidelines (from programmers' guide and chm file):
    In the Eyelink eye tracker (4.5.3.5. Eyelink 1000 Plus User Manual):
    Blinks are always preceded and followed by partial occlusion of the pupil, causing artificial changes in pupil
    position. These are sensed by the EyeLink 1000 Plus parser, and marked as saccades. The sequence of events produced is always:
    • STARTSACC
    • STARTBLINK
    • ENDBLINK
    • ENDSACC
    Note that the position and velocity data recorded in the ENDSACC event is not valid. All data between the STARTSACC
    and ENDSACC events should be discarded.
    - Labeling an SSACC...ESACC pair with one or more SBLINK events between them as a blink, not a saccade.
    - Eliminating any short (less than 120 millisecond duration) fixations that precede or follow a blink as these may
    be artificial or be corrupted by the blink. The end of fixation events will be marked by EFIX events, and those
    markers will immediately precede the SSACC event marker for the saccade surrounding the blink.  Similarly, the start
    of fixations immediately following blinks will be marked by an SFIX marker immediately after the ESACC marker
    containing the blink.  So, one strategy could be to find the EFIX/SFIX markers before/after each blink event and
    check the duration of those associated fixations to determine whether you want to discard them.


    :param et_data_dict: the dictionary of the subject's entire eye tracker information:
    - samples
    - blinks
    - fixations
    - saccades

    :return: a dictionary that contains
    - samples: each sample line labeled as a fixation/saccade/blink based on cleaned fixations and saccades
    - blinks
    - saccades: cleaned from artificial
    - fixations: cleaned from artificial
    """

    res_et_data = {DF_FIXAT: None, DF_SACC: None, DF_BLINK: et_data_dict[DF_BLINK], DF_SAMPLES: None}

    # mark on Eyelink data guidelines and their stamping of FIX, SACC, BLINK
    fixs = et_data_dict[DF_FIXAT].sort_values(by=[T_START])
    saccs = et_data_dict[DF_SACC].sort_values(by=[T_START])
    bls = et_data_dict[DF_BLINK].sort_values(by=[T_START])
    samps = et_data_dict[DF_SAMPLES]

    for index, blink in bls.iterrows():
        # Labeling an SSACC...ESACC pair with one or more SBLINK events between them as BLINKS
        saccs.loc[(saccs[T_START] <= blink[T_START]) & (saccs[T_END] >= blink[T_START]) & (
                saccs['eye'] == blink['eye']), f"is_{EYELINK}Blink"] = blink[T_START]

    """
    note that fixations CANNOT contain blink events according to Marcus from eyelink, so this check should be satisfactory
    Labeling fixations that precede or follow a blink : "The end of fixation events will immediately precede the 
    SSACC event marker for the saccade surrounding the blink. Similarly, the start of fixations immediately following 
    blinks will be marked by an SFIX marker immediately after the ESACC marker containing the blink."
    - short
    - fixation tEnd immediately before the tStart of a false saccade (that surrounds a blink), or
    fixation tStart immediately after false saccade's tEnd
    """
    only_false_saccs = saccs[~pd.isna(saccs[f"is_{EYELINK}Blink"])]  # only saccades that have blinks in them
    min_fixation_dur = 100  # Based on Eyelink data guidelines
    for index, sacc in only_false_saccs.iterrows():
        fixs.loc[(fixs['duration'] < min_fixation_dur) &
                ((fixs[T_END] == sacc[T_START] - 1) | (fixs[T_START] == sacc[T_END] + 1)) &
                (fixs['eye'] == sacc['eye']), f"is_{EYELINK}Blink"] = sacc[f"is_{EYELINK}Blink"]  # if sacc['is_EyeLinkBlink'] is somehow nan, that would be the case in the fixs as well

    # 19-12-25 IMPORTRANT, actually enforce the min duration overlapping
    fixs = fixs[pd.isna(fixs[f"is_{EYELINK}Blink"])]  # only fixations that were not marked in the previous step
    # put it in result dict
    res_et_data[DF_FIXAT] = fixs
    saccs.loc[:, ['xStart', 'yStart', 'xEnd', 'yEnd', 'ampDeg']] = saccs[['xStart', 'yStart', 'xEnd', 'yEnd', 'ampDeg']].replace("^.$", np.nan, regex=True)
    res_et_data[DF_SACC] = saccs.astype({'xStart': 'float64', 'yStart': 'float64', 'xEnd': 'float64', 'yEnd': 'float64', 'ampDeg': 'float64'})

    # MARK sdSamples as fixations and saccades ONLY samples that are NOT EYELINK BLINKS
    only_saccs = saccs[pd.isna(saccs[f"is_{EYELINK}Blink"])]
    only_fixs = fixs[pd.isna(fixs[f"is_{EYELINK}Blink"])]

    # add the (Eyelink + NOT EYELINK BLINK) saccade/fixation information to the sample data
    print(f"Blink information {datetime.datetime.now()}")
    for blink in bls.itertuples():
        samps.loc[samps[T_SAMPLE].between(blink.tStart, blink.tEnd), f"{blink.eye}{EYELINK}Blink"] = 1

    print(f"Saccade information {datetime.datetime.now()}")
    for sacc in only_saccs.itertuples():
        samps.loc[samps[T_SAMPLE].between(sacc.tStart, sacc.tEnd), f"{sacc.eye}{EYELINK}Sacc"] = 1

    print(f"Adding blink, fixation, and saccade information to samples dataframe {datetime.datetime.now()}")
    for fix in only_fixs.itertuples():
        samps.loc[samps[T_SAMPLE].between(fix.tStart, fix.tEnd), f"{fix.eye}{EYELINK}Fix"] = 1

    res_et_data[DF_SAMPLES] = samps.reset_index(drop=True)
    print(f"Finished all {datetime.datetime.now()}")

    return res_et_data


def et_data_to_trials(et_data_prepro, trial_info, params):
    """
    In-place, change the preprocessed et_data dataframes (FIX, SACC, BLINK, SAMPLES) and trial_info to include columns
     indicating about trials. NOTE: trials are recognized by EPOCH start and end. Meaning, anything that's outside a
     trial's EPOCH is marked in et_data_prepro's dataframes column TRIAL as -1.

    :param et_data_prepro: output of preprocess_et_data method: a dictionary containing:
    {DF_FIXAT: fixation dataframe with "is_blink" column labeling false fixations,
    DF_SACC: saccade dataframe with "is_blink" column labeling false saccade,
    DF_BLINK: et_data_dict[DF_BLINK],
    DF_SAMPLES: sample dataframe with a column per fix/sacc/blink per eye (e.g., Rblink)}
    :param trial_info: information about the trial and subject behavior as derived from the eyetracker TRIGGERs
    :param params: subject's parameters
    :return:
    - et_data_prepro: same dict of 4 ET dataframes, with a column named TRIAL marking the trial the sample belongs to.
    - trial_info: same dataframe with additional columns for the important time windows of each trial
    """

    # get the inds of the eye tracking data samples (DF_SAMPLES) that match stimulus onset
    probed_stim_onsets = np.array(trial_info[ONSET])  # remember, these onsets are derived from eyelink trigger messages
    try:
        #probed_stim_onsets_sample_inds = np.array([np.where(et_data_prepro[DF_SAMPLES][T_SAMPLE] == onset)[0][0] for onset in probed_stim_onsets])
        probed_stim_onsets_sample_inds = np.array([np.where(et_data_prepro[DF_SAMPLES][T_SAMPLE] == onset)[0][0] for onset in probed_stim_onsets])
    except Exception:
        print("An issue with subject, trying to succeed regardless")
        probed_stim_onsets_sample_inds = np.array([np.where(et_data_prepro[DF_SAMPLES][T_SAMPLE] == onset)[0] for onset in probed_stim_onsets])

    # get the inds of the eye tracking data samples that match different interesting events (e.g.epoch beginning)
    # and then add to trial_info the timestamps of the samples that match these events for each trial
    interesting_events = {EPOCH + WINDOW_START: ET_param_manager.EPOCH_START,  # EPOCH start (with respect to stim ONSET)
                          EPOCH + WINDOW_END: ET_param_manager.EPOCH_END,  # EPOCH end (with respect to stim ONSET)
                          P100P600: 100,
                          P100P600 + WINDOW_END: 600,
                          P250P500: 250,
                          P250P500 + WINDOW_END: 500,
                          P200P500: 200,
                          P200P500 + WINDOW_END: 500,
                          PRE_STIM_DUR + WINDOW_START: ET_param_manager.PRE_STIM_DUR,  # pre-stimulus start (end=onset)
                          STIM_DUR + WINDOW_END: ET_param_manager.STIM_DUR}  # stim duration (start=onset)

    for event in interesting_events:
        event_time = interesting_events[event] / 1000  # div by 1000 to turn MILLISECONDS TO SECONDS
        rel = event_time * params[SAMPLING_FREQ] * (-1) if "Start" in event else event_time * params[SAMPLING_FREQ]
        event_sample_inds = list(np.ceil(probed_stim_onsets_sample_inds + rel).astype(int))
        event_samples = np.array(et_data_prepro[DF_SAMPLES].iloc[event_sample_inds, 0])
        trial_info[event] = event_samples

    # prepare a column in all ET dataframes to contain the trial number of samples within each epoch
    time_windows = {TRIAL: (EPOCH + WINDOW_START, EPOCH + WINDOW_END),
                    PRE_STIM_DUR: (PRE_STIM_DUR + WINDOW_START, ONSET),
                    STIM_DUR: (ONSET, STIM_DUR + WINDOW_END),
                    P100P600: (P100P600, P100P600 + WINDOW_END),
                    P250P500: (P250P500, P250P500 + WINDOW_END),
                    P200P500: (P200P500, P200P500 + WINDOW_END)}

    for key in et_data_prepro.keys():
        for window in time_windows:
            et_data_prepro[key][window] = -1

    for index, trial in trial_info.iterrows():  # iterate trials
        for window in time_windows:
            window_start = trial[time_windows[window][0]]
            window_end = trial[time_windows[window][1]]
            for key in [DF_BLINK, DF_FIXAT, DF_SACC, DF_SACC_EK]:
                all_key_data = et_data_prepro[key]
                all_key_data.loc[:, IS_LEVEL_ELIMINATED] = 0  # initialize
                all_key_data.loc[all_key_data[T_START].between(window_start, window_end) |
                                 all_key_data[T_END].between(window_start, window_end), IS_LEVEL_ELIMINATED] = trial[IS_LEVEL_ELIMINATED]
                all_key_data.loc[all_key_data[T_START].between(window_start, window_end) |
                                 all_key_data[T_END].between(window_start, window_end), window] = trial["trialNumber"]
            et_data_prepro[DF_SAMPLES].loc[(et_data_prepro[DF_SAMPLES][T_SAMPLE] <= window_end) & (et_data_prepro[DF_SAMPLES][T_SAMPLE] >= window_start), window] = trial["trialNumber"]
            et_data_prepro[DF_SAMPLES].loc[(et_data_prepro[DF_SAMPLES][T_SAMPLE] <= window_end) & (et_data_prepro[DF_SAMPLES][T_SAMPLE] >= window_start), IS_LEVEL_ELIMINATED] = trial[IS_LEVEL_ELIMINATED]

    return et_data_prepro, trial_info
