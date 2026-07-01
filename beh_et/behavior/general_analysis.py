import os
import pickle
import data_reader
import response_type_analysis
import diff_perf_analysis
import vg_features_analysis
import quality_checks
import stats

"""
This module manages all the general analyses performed on the exp.2 data. It extracts all the relevant data from the 
Subject class instances (data_reader module), and then calls all the general analyses modules:
1. Response Type Analysis: response_type_analysis module - analyzes subjects' response type rates (TP/TN/FA/FN)
AND analyzes difficulty and performance parameters per response type: mean/median of performance/difficulty for each 
subject across the trials from that type.
2. Difficulty and Performance Analysis: diff_perf_analysis module - analyzes difficulty and performance throughout the 
experiment (meaning, across trials, not splitting to trial types by their responses or anything else).
Performance: a calculated sensitivity score (d') which is based on the collisions of the player with good/bad essences.
Difficulty: manipulated based on the subject's performance; decreasing when d' < 0 and increasing when d' > 0.3.
3. Analysis of game effects on visibility: starting with several runs of a GLM trying to predict stimulus visibility 
from the video-game's visual features, using 4 different kernels at a time (for comparison). Then, 2 follow-up tests 
about specific relations between specific game elements and visibility, following the results I got. These are specific, 
but can be easily tweaked to test any params you want. 3a tests relationship between two categorical variables in the
feature table (visibility - stim ID) using a chi square test, and 3b tests relationship between one categorical 2-level
variable (visibility: seen/unseen) and a continuous variable (distance) in a t-test accompanied by BFs. 

@author: RonyHirsch
"""

# names for parameters
IRREL = 'OutsideWindowPress'
IRREL_2 = 'AdditionalPress'
GAME = 'GAME'
REPLAY = 'REPLAY'
REPLAY_TARGETS = 'REPLAY_TARGETS'
REPLAY_TARGET = 'TARGET'
ACTIVE_LEVEL_ID = 'activeLevelID'
REPLAY_LEVEL = 'LevelToReplayID_1Based'
TARGET_TYPE = 'TargetType'
TASK_DICT = {GAME: "dAT", REPLAY: "AT"}
PROBED = 'isProbed'
response_eval = 'responseEvaluation'
FN = 'FalseNegative'
TN = 'TrueNegative'
TP = 'TruePositive'
FP = 'FalsePositive'

# more data column names
DIFF = 'difficulty'
PERF = 'performance'


def match_replay_level_to_target(row, replay_targets):
    level = row[ACTIVE_LEVEL_ID]
    target_row = replay_targets[replay_targets[REPLAY_LEVEL] == level]
    target = target_row[TARGET_TYPE].to_string(header=False, index=False)
    return target


def extract_probe_data_replay(sess: data_reader.Session, sub_id):
    localizer_data, _, _, _ = quality_checks.calculate_replay_responses(sess, sub_id, extra=False)
    replay_targets = sess.replay_targets
    # add column in localizer data denoting the target (face/object) of each replay level
    localizer_data[REPLAY_TARGET] = localizer_data.apply(lambda row: match_replay_level_to_target(row, replay_targets), axis=1)
    # ignore 'out of window' presses i.e only count responses to stimuli/blanks
    localizer_data = localizer_data[localizer_data['responseEvaluation'] != IRREL]
    # ignore extra presses subjects make
    localizer_data = localizer_data[localizer_data['responseEvaluation'] != IRREL_2]
    localizer_data.reset_index(inplace=True, drop=True)

    return localizer_data, replay_targets


def extract_probe_data_game(sess: data_reader.Session):
    probe_data = sess.SessDetails.ProbeDet  # probe data of this session
    probe_data = probe_data[probe_data['activeWorldID'] != -1]  # exclude practice levels from analysis
    probe_data.reset_index(drop=True, inplace=True)  # reset index so that index will accurately represent trial number
    return probe_data


def extract_relevant_data(sub_dict):
    """
    This code goes over the dictionary of subjects {subCode:Subject instance} and extracts the probe information for the
    game and replay parts of the experiment. It returns a new dictionary, in which key=sub code, and
    val={GAME:df, REPLAY:df}, where the dfs are the probe information.
    :param sub_dict: A dictionary of type {subCode:Subject class instance}
    :return: A dictionary of type {subCode:{GAME:df, REPLAY:df}}
    """
    result_dict = dict()
    for sub_code in sub_dict:
        sub = sub_dict[sub_code]
        if hasattr(sub, 'full'):  # if the subject has full experimental data
            sub_probe_data_game = extract_probe_data_game(sub.full)
            sub_probe_data_replay, sub_replay_targets = extract_probe_data_replay(sub.full, sub_id=[sub.lab, sub.id])
            result_dict[sub_code] = dict()
            result_dict[sub_code][GAME] = sub_probe_data_game
            result_dict[sub_code][REPLAY] = sub_probe_data_replay
            result_dict[sub_code][REPLAY_TARGETS] = sub_replay_targets
    return result_dict


def analyze(data_path, plot=True, save=False, save_path=""):
    # subjects is a dictionary in which key=subject code (lab code+subject number), and val=an instance of class Subject
    # all subjects in data_path will be included in the analysis!
    subjects = data_reader.read_data(data_path)
    file_name = "subject_beh.pickle"
    fl = open(os.path.join(data_path, file_name), 'rb')
    data = pickle.load(fl)
    fl.close()

    # generate the dictionary in which {key=sub, val={GAME:probe_df, REPLAY:probe_df}}
    probe_data = extract_relevant_data(subjects)

    # GENERAL ANALYSIS #4 : ANALYSIS OF -STIMULUS ID- EFFECTS ON VISIBILITY IN GAME AND IN REPLAY
    #vg_features_analysis.analyze_vis_per_stim_id(subjects, save_path=save_path)

    # GENERAL ANALYSIS #1 : ANALYSIS OF RESPONSE TYPES, CALC OF MEAN/MEDIAN PERFORMANCE/DIFFICULTY PER RESPONSE TYPE
    response_type_analysis.analyze_response_types(probe_data, plot=plot, save=save, save_path=save_path)

    # GENERAL ANALYSIS #2: ANALYSIS OF DIFFICULTY/PERFORMANCE ACROSS THE GAME
    diff_perf_analysis.analyze_diff_perf(probe_data, save=save, save_path=save_path)

    # GENERAL ANALYSIS #3: ANALYSIS OF -GAME- EFFECTS ON VISIBILITY
    vg_features_analysis.analyze_vg_effects_on_vis(subjects, save_path=save_path, kernel='linear')

    # FOLLOW UP ON # 3 (3a): TEST RELATIONSHIP BETWEEN -GAME- VISIBILITY AND STIMULUS DISTANCE FROM ESSENCES
    sub_df = vg_features_analysis.unify_subs_for_analysis(subjects, filter_out_replay=True)
    # only probed stim are interesting
    sub_df = sub_df[sub_df[PROBED] == True]
    # filter only TRUE POSITIVE VS FALSE NEGATIVE
    sub_df = sub_df[(sub_df[response_eval] == TP) | (sub_df[response_eval] == FN)]
    sub_df = sub_df.drop(columns=["subjectID", 'world', 'level', "stimName", "stimType", 'versionString', 'versionDate',
                                  'versionSettings', 'timeMS',
                                  'timeMS_NoPauses', 'fullLogState', 'indexWithinFullLogs',
                                  'stimID', 'probeIndexWithinDetails', 'isProbed', 'offsetTS',
                                  'offsetTS_NoPauses', 'probeTS', 'probeTS_NoPauses', 'response',
                                  'responseTS', 'responseTS_NoPauses', 'onset_averageDifficulty',
                                  'response_averageDifficulty', 'avgNumFallingEssenses_m1000_p500',
                                  'avgNumOppositeColorEssenses_m1000_p500',
                                  'avgNumSameColorEssenses_m1000_p500',
                                  'numButtonPresses_m1000_p500', 'numSaccades_m2000_p500',
                                  'tempProxOfLastSacada', 'spatialProxFromStimLocToLastSacadaLine',
                                  'averageGaze_m2000_p500', 'numBlinks_m2000_p500',
                                  'tempProxOfLastBlink'])
    
    dependent_var_cols = ["spatialProxFromStimLocToNearestEssense_0",
                          "spatialProxFromStimLocToNearestEssense_p200",
                          "spatialProxFromStimLocToNearestEssense_p400",
                          "spatialProxFromStimLocToClusterMeanEssenses_p200",
                          "spatialProxFromStimLocToClusterMeanEssenses_m400",
                          "averageSpeedFallingEssenses_OverFrames_m1000_p500",
                          "averageSpeedFallingEssenses_OverEssences_m1000_p500"]

    vg_features_analysis.ttest_vis_distance_relations(sub_df, dependent_var_cols=dependent_var_cols, save_path=save_path)
    vg_features_analysis.anova_vis_distance_relations(sub_df, dependent_var_cols, save_path)

    return


if __name__ == "__main__":
    analyze(data_path=r"PATH", plot=True, save=True, save_path=r"PATH")



