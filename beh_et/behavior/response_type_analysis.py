import pandas as pd
import numpy as np
import seaborn as sns
import os
import re
import copy
import stats
import boxplotter
import lineplotter
import data_saver
import data_reader

"""
This module manages all the analyses related to response type rates and to data splitted according to response types. 
Response types are TP/FN/FP/TN, and they are responses to exp.2 awareness probes. 
The "analyze_response_types" method manages all the analyses done in relation to response types. These include:
1. analyze_per_response_type : analyze and save a summary table of response types per subject: response counts, 
response rates, average/median difficulty/performance per response. This is performned separatly for game (dAT) and 
replay (AT) conditions. Also, this module plots a boxplot of RESPONSE TYPE RATE data (across game/replay).
2.  diff_perf_per_response_type : saves a summary table of average/median ONLY for difficulty/performance in the 
appropriate place in the BIDS hierarchy. Also, this module plots a boxplot of average/median difficulty/performance 
per response type. 
3. response_ma: this is used to plot a line plot depicting the MOVING AVG (and SEs) of a specific response type, across
the entire game. Currently it is run only for "hit", so 1 plot depicting the moving average of the hit rate across
GAME trials is plotted and saved. 

@author: RonyHirsch
"""

# names for parameters
IRREL = 'OutsideWindowPress'
GAME = 'GAME'
REPLAY = 'REPLAY'
REPLAY_TARGET = 'TARGET'
REPLAY_TARGETS = 'REPLAY_TARGETS'
LEVEL_ID = "currentLevelID"
LEVELS_GAME = [x for x in range(1, 17)]
LEVELS_REPLAY = [x for x in range(100, 108)]
LEVELS = {GAME: LEVELS_GAME, REPLAY: LEVELS_REPLAY}
TASK_DICT = {GAME: "dAT (Game)", REPLAY: "AT (Replay)"}
TASK_DICT_SAVE = {GAME: "game", REPLAY: "replay"}
CATEGORY = 'stimulusType'
FACE = 'Face'
OBJ = 'Object'
BLANK = 'None'
LOC = 'stimulusLocation'
TR = 'TopRight'
TL = 'TopLeft'
BR = 'BottomRight'
BL = 'BottomLeft'
MOVING_AVG = "moving_avg"
UP_BOTTOM = "updown"
LEFT_RIGHT = "leftright"
INTERACTION = "interaction"

# responseEvaluation types
EVAL = 'responseEvaluation'
TRUEPOSITIVE = 'TruePositive'
FALSEPOSITIVE = 'FalsePositive'
TRUENEGATIVE = 'TrueNegative'
FALSENEGATIVE = 'FalseNegative'
COUNTER_RESPONSE = {TRUEPOSITIVE: FALSENEGATIVE, FALSEPOSITIVE: TRUENEGATIVE,
                    FALSENEGATIVE: TRUEPOSITIVE, TRUENEGATIVE: FALSEPOSITIVE}
# names for plotting these
TP = 'True Positive'
FP = 'False Positive'
TN = 'True Negative'
FN = 'False Negative'

SEEN = 'Seen'
UNSEEN = 'Unseen'
SEEN_BLANK = 'Seen (Blank)'
UNSEEN_BLANK = 'Unseen (Blank)'
HIT = 'Hit'
MISS = 'Miss'
FALSE_ALARM = 'False Alarm'
CORRECT_REJECTION = 'Correct Rejection'


# more data column names
DIFF = 'difficulty'
PERF = 'performance'

# result data table column names
ID_NAMES = ['Lab', 'Subject']
PROBE_NAMES = ['probes', 'tp', 'fp', 'tn', 'fn', 'no_response']
RESPONSE_RATE_NAMES = ['tp_rate', 'fp_rate', 'tn_rate', 'fn_rate']
DIFF_STATS_NAMES = ['diff_avg_tp', 'diff_avg_fp', 'diff_avg_tn', 'diff_avg_fn',
                    'diff_med_tp', 'diff_med_fp', 'diff_med_tn', 'diff_med_fn']
PERF_STATS_NAMES = ['perf_avg_tp', 'perf_avg_fp', 'perf_avg_tn', 'perf_avg_fn',
                    'perf_med_tp', 'perf_med_fp', 'perf_med_tn', 'perf_med_fn']
NAMES = ID_NAMES + PROBE_NAMES + RESPONSE_RATE_NAMES + DIFF_STATS_NAMES + PERF_STATS_NAMES

# names for these
RESPONSE_RATE_NAMES_MAPPING = {'tp_rate': 'Hit Rate', 'fp_rate': 'False Alarm',
                               'tn_rate': 'Correct Rejection', 'fn_rate': 'Miss Rate'}

RESPONSE_RATE_NAMES_TITLE = {'tp_rate': TP, 'fp_rate': FP, 'tn_rate': TN, 'fn_rate': FN}
RESPONSE_RATE_NAMES_PLOT = {'tp_rate': {GAME: SEEN, REPLAY: HIT},
                            'fp_rate': {GAME: SEEN_BLANK, REPLAY: FALSE_ALARM},
                            'tn_rate': {GAME: UNSEEN_BLANK, REPLAY: CORRECT_REJECTION},
                            'fn_rate': {GAME: UNSEEN, REPLAY: MISS}}
RESPONSE_RATE_NAMES_PLOT_2 = {TRUEPOSITIVE: {GAME: SEEN, REPLAY: HIT},
                              FALSEPOSITIVE: {GAME: SEEN_BLANK, REPLAY: FALSE_ALARM},
                              TRUENEGATIVE: {GAME: UNSEEN_BLANK, REPLAY: CORRECT_REJECTION},
                              FALSENEGATIVE: {GAME: UNSEEN, REPLAY: MISS}}

# data for SMA
SMA_WINDOW = 13
SMA_REPLAY_WINDOW = 10
WORLD_ID = "activeWorldID"

# variables to account for diff between ECOG and other methods
WORLDS = [0, 1, 2, 3]
WORLDS_ECOG = [0, 1]
ECOG_LABS = ("SE", "SF", "SX")
NUM_PROBES_PER_WORLD = 50
NUM_PROBES_PER_WORLD_REPLAY = 25

# result df columns
WORLD_COL = 'world'
AVG = 'average'
STD = 'standard deviation'
SE = 'standard error'

# summary_stats columns
STATS_COLS = ['variable', 'statistic', 'statistic_val', 'p_val']
BAYES_COLS = ['T', 'dof', 'tail', 'p-val', 'CI95', 'cohen-d', 'BF10', 'power']

# names for saving
FILENAME = "exp2_response_type_analysis"
SMA_RATE_FILENAME = "exp2_response_rate_moving_average"
ACROSS = "across_game"
PER_RESP = "per_response_type"
BF_TTEST_PAIRED = "BF_ttest_paired"
BF_TTEST_IND = "BF_ttest_ind"
PERM_CLUS_PAIRED = "permutation_cluster_ttest_paired"
RESP_TYPE_RATES = "response_type_rates"
COND_FOLDER = {LOC: "stim_loc", CATEGORY: "stim_type", UP_BOTTOM: "top_bottom", LEFT_RIGHT: "left_right"}


# colors for plots
COLOR_DICT = {"SA": "#274C77", "SB": "#6096BA", "SC": "#B42D4F", "SD": "#D65C7A"}


def create_cond_dict(all_conditions):
    """
    Create a dictionary from the given keys and sub-keys (nested dict structure)
    :param all_conditions: The given dict
    :return: The resulting dictionary
    """
    cond_dict = dict.fromkeys(all_conditions)
    for key in cond_dict.keys():
        cond_dict[key] = dict()
        for subcond in all_conditions[key]:
            cond_dict[key][subcond] = None

    return cond_dict


def probe_data_go_nogo(probe_data):
    stim_present_list = list()
    stim_absent_list = list()
    for active_level in probe_data[LEVEL_ID].unique():
        active_target = probe_data[probe_data[LEVEL_ID] == active_level].iloc[0][REPLAY_TARGET]
        present_level_data = probe_data[(probe_data[LEVEL_ID] == active_level) & (probe_data[CATEGORY] == active_target)]
        stim_present_list.append(present_level_data)
        """
        Stimulus absent in REPLAY (localizer) levels:
        - 1: stimulus from the wrong category (stimulusID=int, stimulusType="Face"/"Object")
        - 2: blank stimulus                   (stimulusID=int, stimulusType="None")
        - 3: filler (non-stimulus)            (stimulusID=nan, stimulusType=nan)
        """
        absent_level_data = probe_data[(probe_data[LEVEL_ID] == active_level) & (probe_data[CATEGORY] != active_target)]
        stim_absent_list.append(absent_level_data)
    stim_present = pd.concat(stim_present_list)
    stim_absent = pd.concat(stim_absent_list)
    # these lines are to make sure we did not miss anything and that stim_present + stim_absent include all probe_data
    processed_df = pd.concat([stim_present, stim_absent])
    difference_df = probe_data.merge(processed_df, indicator=True, how='left').loc[lambda x: x['_merge'] != 'both']
    if not difference_df.empty:
        print('probe_data_go_nogo: stim_present + stim_absent != probe_data')
    non_target_stim_only = stim_absent[(stim_absent[CATEGORY].notnull()) & (stim_absent[CATEGORY] != BLANK)]  # these are all the non-target stimuli, w/o blanks
    return stim_present, stim_absent, non_target_stim_only


def response_type_breakdown(sub_code, subj_probe_data, probe_type):
    """
    This method receives subject probe dataframe, which can either be the original dataframe or a pre-filtered one
    (according to some column). It takes this df, then breaks it down according to the EVAL column (TP/FP/TN/FN),
    and returns a line of information about this subject's response type breakdown in the probe data. This is then
    aggregated across subjects to create a dataframe in which each line is a subject and each column is info about
    a response type.
    :param sub_code: subject code, to be added to the data list
    :param subj_probe_data: the subject's probe dataframe, original or filtered
    :return: list of values
    """
    sub_data = []
    id = [sub_code[0:2], sub_code[2:]]
    sub_data.extend(id)

    if probe_type == REPLAY:
        stim_present, stim_absent, non_target_stim_only = probe_data_go_nogo(subj_probe_data)
        num_stim_nontarget = non_target_stim_only.shape[0]
    else:
        stim_present = subj_probe_data[subj_probe_data[CATEGORY] != BLANK]
        stim_absent = subj_probe_data[subj_probe_data[CATEGORY] == BLANK]

    num_stim_present = stim_present.shape[0]
    num_stim_absent = stim_absent.shape[0]

    tp = subj_probe_data[subj_probe_data[EVAL] == TRUEPOSITIVE]
    fp = subj_probe_data[subj_probe_data[EVAL] == FALSEPOSITIVE]
    tn = subj_probe_data[subj_probe_data[EVAL] == TRUENEGATIVE]
    fn = subj_probe_data[subj_probe_data[EVAL] == FALSENEGATIVE]

    if num_stim_present != (tp.shape[0] + fn.shape[0]) or num_stim_absent != (tn.shape[0] + fp.shape[0]):
        print(f"ERROR: Sanity check of number of target stimuli failed in {probe_type}")

    probes = [subj_probe_data.shape[0], tp.shape[0], fp.shape[0], tn.shape[0], fn.shape[0]]
    sub_data.extend(probes)

    # in MRI, subjects have the options not to respond even to game probes (response timeout)
    no_resp = subj_probe_data.shape[0] - (tp.shape[0] + fp.shape[0] + tn.shape[0] + fn.shape[0])
    sub_data.extend([no_resp])

    # translate to SDT terms : d', criterion, Ad, beta
    sdt_df = stats.SDT(hits=tp.shape[0], misses=fn.shape[0], fas=fp.shape[0], crs=tn.shape[0])

    # add rate information : ['tp_rate', 'fp_rate', 'tn_rate', 'fn_rate']
    tpr = tp.shape[0] / num_stim_present if num_stim_present != 0 else 0
    fpr = fp.shape[0] / num_stim_absent if num_stim_absent != 0 else 0
    tnr = tn.shape[0] / num_stim_absent if num_stim_absent != 0 else 0
    fnr = fn.shape[0] / num_stim_present if num_stim_present != 0 else 0
    rates = [tpr, fpr, tnr, fnr]
    sub_data.extend(rates)
    # add performance and difficulty information per response type
    data_list = [tp, fp, tn, fn]
    diff_means = list()
    diff_medians = list()
    perf_means = list()
    perf_medians = list()
    for df in data_list:
        if not df.empty:
            diff_means.append(df[DIFF].mean())
            diff_medians.append(df[DIFF].median())
            perf_means.append(df[PERF].mean())
            perf_medians.append(df[PERF].median())
        else:
            diff_means.append(np.nan)
            diff_medians.append(np.nan)
            perf_means.append(np.nan)
            perf_medians.append(np.nan)
    sub_data.extend(diff_means)
    sub_data.extend(diff_medians)
    sub_data.extend(perf_means)
    sub_data.extend(perf_medians)
    return sub_data, sdt_df


def analyze_per_response_type(probe_data_dict, probe_type, plot=True, save=False, save_path=""):
    """
    Analyze the response types' frequency (TP/FN/FP/TN) in a part of experiment 2.
    NOTE: the analysis is done on either the GAME (dAT) or the REPLAY (AT) parts of the experiment, depending on
    "probe_type". It generates a csv containing the columns listed in "NAMES", which include response type rates
    as well as avg/median difficulty/performance per response type.
    :param probe_data_dict: a dictionary in which {key=sub, val={GAME:probe_df, REPLAY:probe_df}} where the dfs are
    the "details" files containing the relevant probe information (meaning, RAW FULL DATA).
    :param probe_type: whether twe want to analyze the GAME or the REPLAY
    :param plot: whether to plot the data
    :param save: whether to save the data
    :param save_path: path to the highest folder in the BIDS-compatible data folder format.
    :return: dataframe with the summary of all data, each line is a subject and columns are stats
    """

    save_name = TASK_DICT_SAVE[probe_type] + "_resp_type_rates"
    overall_save_name = "overall_" + save_name

    sub_data_list = []
    sub_sdt_list = []
    for sub_code in probe_data_dict:
        data = probe_data_dict[sub_code][probe_type]  # subject probe dataframe, FILTERED BY PROBE TYPE (game/replay)
        # analyze the probe dataframe to extract info about TP, FP, TN, FN
        sub_data, sub_sdt_summary = response_type_breakdown(sub_code, data, probe_type)
        # add it to the list that is going to be the dataframe
        sub_data_list.append(sub_data)
        sub_sdt_list.append(sub_sdt_summary)
    result_df = pd.DataFrame(sub_data_list, columns=NAMES)
    sdt_concatenated = pd.concat((df for df in sub_sdt_list))
    sdt_mean_df = sdt_concatenated.groupby(sdt_concatenated.index).mean()
    sdt_std_df = sdt_concatenated.groupby(sdt_concatenated.index).std()

    if plot:
        # raincloud of ALL response rates
        colors = sns.color_palette("colorblind", 4)
        color_list = [colors[0], colors[2], colors[1], colors[3]]
        col_order = ['tp_rate', 'fn_rate', 'fp_rate', 'tn_rate']
        col_name_order = {x: RESPONSE_RATE_NAMES_PLOT[x][probe_type] for x in col_order}
        horiz_lines = [{'y': None, 'xmin': 0.7, 'xmax': 2.3, 'label': "Stimulus Present"},
                       {'y': None, 'xmin': 2.7, 'xmax': 4.3, 'label': "Stimulus Absent"}]
        boxplotter.plot(data=result_df, data_col_order=col_order, data_name_order=col_name_order,
                        raincloud=True, scatter=False, sub_line=[list(range(4))],
                        plot_title=f"Overall {TASK_DICT[probe_type]} Response Rates", plot_x_label="Response Type",
                        plot_y_label="Response Type Rate", color_list=color_list, horizontal_lines=horiz_lines,
                        save_name=overall_save_name, save_plot=save, save_path=save_path, sub_folder=RESP_TYPE_RATES)

    if save:
        saver(df=result_df, save_path=save_path, folder_name=RESP_TYPE_RATES, file_name=overall_save_name, type="")
        # create summary stats table and save it
        res_df = result_df.copy()  # create a copy not to hurt the returned result df
        res_df.columns = map(str.upper, res_df.columns)
        saver(df=res_df.describe(), save_path=save_path, folder_name=RESP_TYPE_RATES, file_name=overall_save_name+"_summary", type="")
        # save SDT stuff
        saver(df=sdt_mean_df, save_path=save_path, folder_name=RESP_TYPE_RATES, file_name=overall_save_name+"_SDT_avg", type="")
        saver(df=sdt_std_df, save_path=save_path, folder_name=RESP_TYPE_RATES, file_name=overall_save_name + "_SDT_std", type="")

    return result_df, sub_sdt_list


def diff_perf_per_response_type(summary_table: pd.DataFrame, diff_1_perf_0, save_path):
    """
    This function plots and saves information about the AVG/MEDIAN DIFFICULTY/PERFORMANCE per response type (TP, FP,
    TN, FN)
    :param data_dict: the entire GAME / REPLAY data dictionary, as the summary table does not include all information
    :param summary_table: a summary table of response type amounts, difficulty and performance levels s.t each line is
    a subject. This is the "analyze_per_response_type" output
    :param game_1_replay_0: whether or not the summary table belongs to GAME or REPLAY data --> DEPRECATED AS
    THERE'S NO MEANING TO DIFFICULTY / PERFORMANCE IN REPLAY!
    :param diff_1_perf_0: whether or not we want to analyze response types considering DIFFICULTY or PERFORMANCE
    :param save_path: the general data folder (highest folder in the BIDS-compatible folder structure)
    :return:
    """
    names = ID_NAMES + DIFF_STATS_NAMES if diff_1_perf_0 == 1 else ID_NAMES + PERF_STATS_NAMES
    prefix = "diff_" if diff_1_perf_0 == 1 else "perf_"
    title = DIFF if diff_1_perf_0 == 1 else PERF
    save_name = f"{prefix}per_resp_{TASK_DICT_SAVE[GAME]}"
    avg_save_name = save_name + "_avg"
    median_save_name = save_name + "_med"
    summary_save_name = save_name + "_summary"

    # step 4: plot raincloud plots for mean difficulty/performance per response type : SEEN vs UNSEEN
    colors = ["#1F7A8C", "#90141E"]  # seen, unseen
    boxplotter.plot(data=summary_table, data_col_order=[prefix + 'avg_tp', prefix + 'avg_fn'],
                    data_name_order={prefix + 'avg_tp': SEEN, prefix + 'avg_fn': UNSEEN}, raincloud=True,
                    sub_line=[list(range(len(colors)))], plot_title=f"Average {title} in Seen vs Unseen Stimuli",
                    plot_x_label="Response Type", plot_y_label=f"Average {title}", colors_per_lab=0, color_list=colors,
                    save_plot=True, save_path=save_path, save_name=avg_save_name,
                    sub_folder=os.path.join(title, PER_RESP))

    # then for median difficulty/performance per response type
    """
    boxplotter.plot(data=summary_table, data_col_order=[prefix + 'med_tp', prefix + 'med_fn'],
                    data_name_order={prefix + 'med_tp': "Seen", prefix + 'med_fn': "Unseen"}, raincloud=True,
                    sub_line=[list(range(len(colors)))], plot_title=f"Median {title} in Seen vs Unseen Stimuli",
                    plot_x_label="Response Type", plot_y_label=f"Median {title}", color_list=colors, save_plot=True,
                    save_path=save_path, save_name=median_save_name,
                    sub_folder=os.path.join(title, PER_RESP))
    """

    # step 5: ttest and bayes factors for SEEN vs UNSEEN :
    # COMPARE PERFORMANCE/DIFFICULTY BETWEEN SEEN AND UNSEEN STIMULI IN THE GAME
    # # RELATIONSHIP BETWEEN STIMULUS VISIBILITY TO PERFORMANCE/DIFFICULTY IN GAME
    dat_table = summary_table[names]
    tp_col = [x for x in names if 'tp' in x][0]
    fn_col = [x for x in names if 'fn' in x][0]

    ttest_bayes = stats.paired_t_test_pg(dat_table, col_1=tp_col, col_2=fn_col, df=True)

    # step 6: save data
    saver(df=summary_table, save_path=save_path, folder_name=title, file_name=save_name, type=PER_RESP)

    # save summary of data table
    saver(df=dat_table.describe(), save_path=save_path, folder_name=title, file_name=summary_save_name, type=PER_RESP)

    # save BF
    if ttest_bayes is not None:  # could be None if not enough data for BF
        saver(df=ttest_bayes, save_path=save_path, folder_name=title, file_name=BF_TTEST_PAIRED+save_name, type=PER_RESP)

    return


def response_rolling(eval_data, response, sma_size=SMA_WINDOW):
    """
    This function calculates a moving window of the response-rate for a specific response type (hit/miss/fa/cr)
    :param eval_data: list containing only values for hit/miss/fa/cr
    :param response: the response type we're interested in
    :return: list in which each cell is the response-type-rate of that moving window
    """
    res_list = [np.nan] * len(eval_data)
    counter_response = COUNTER_RESPONSE[response]
    start = sma_size - 1  # we start the moving window in this entry, since before that it's too small for our window
    for i in range(start, len(eval_data)):
        dat_range = eval_data[i - start:i + 1]
        num_res = dat_range.count(response)  # number of times the desired response appeared
        num_countres = dat_range.count(counter_response)
        total = num_res + num_countres
        if total == 0:
            """
            This can happen in 1 of 2 cases:
            1. there were no response+counter in this window. E.g., if we are looking for hits/misses, then only trials 
            where a relevant stimulus was present are interesting. So blanks/irrelevant stim/fillers are not counted.
            2. this level is missing from data as either the subject did not play it (e.g., ECoG patient in game worlds
            3-4 and replay levels 5-8) or was removed when response types were pre-processed due to suspicious behavior
            (see response_type_breakdown)
            """
            rate = np.nan
        else:
            rate = num_res / total  # the "_ rate" for the desired response
        res_list[i] = rate
    return res_list


def correct_for_duplicate_probes(sub_dataframe):
    """
    If for some ungodly reason, a subject has more than NUM_PROBES_PER_WORLD entries, it means the subject had at least
    one aborted+restarted level we need to correct for. Otherwise this subject cannot be analyzed with the rest of the
    data.
    :param sub_dataframe:
    :return:
    """
    sub_dataframe_nofillers = sub_dataframe[sub_dataframe['type'] != 'LOCALIZER_FILLER']
    just_fillers = sub_dataframe[sub_dataframe['type'] == 'LOCALIZER_FILLER']
    just_fillers_nodups = just_fillers.drop_duplicates(subset=['animCycleID'], keep="last", inplace=False)
    sub_dataframe_nofillers_nodups = sub_dataframe_nofillers.drop_duplicates(subset=[LEVEL_ID, WORLD_ID, CATEGORY, 'questionID'], keep="last", inplace=False)
    sub_dataframe_result = pd.concat([sub_dataframe_nofillers_nodups, just_fillers_nodups]).sort_values(by=['stimulusOnsetTS'])
    return sub_dataframe_result


def response_ma_specific(data_dict, condition, response, save, save_path, cond_len, sma_size=SMA_WINDOW, specific_name=""):
    """
    Plot and save a moving average of a specific response type throughout all game AND replay trials.
    :param data_dict: a dictionary in which {key=sub, val={GAME:probe_df, REPLAY:probe_df}}
    :param response: the response we want to calculate the moving avg rate for (TP/FP/FN/TN)
    :param plot: whether to plot the data
    :param save: whether to save the data and plot
    :param save_path: the general data folder (highest folder in the BIDS-compatible folder structure)
    :return:
    """
    res_df = pd.DataFrame()

    for sub_code in data_dict:
        sub_data = correct_for_duplicate_probes(data_dict[sub_code][condition])
        # moving average calculation in this case is manual, as we need to calculate the rate
        sub_eval = list(sub_data[EVAL])
        sub_ma = response_rolling(sub_eval, response, sma_size)
        if len(sub_ma) < cond_len:
            sub_ma.extend([np.nan] * (cond_len - len(sub_ma)))
        if len(sub_ma) > cond_len:
            print(sub_code)
            print(len(sub_ma))
            sub_ma = sub_ma[:cond_len]
        res_df[sub_code] = sub_ma


    # calculate avg std and se
    avg = res_df.iloc[:, :].mean(axis=1)
    std = res_df.iloc[:, :].std(axis=1)
    se = std / np.sqrt(res_df.iloc[:, :].shape[1])
    res_df[AVG] = avg
    res_df[STD] = std
    res_df[SE] = se

    resp = re.sub('([A-Z])', r' \1', response)  # insert a space before every capital letter, for plotting and saving
    # fill nan values with the first non-nan value the column has (1st window):
    res_df_game_filled = res_df.fillna(method='bfill')

    if save:
        specifics = re.sub('([A-Z])', r' \1', specific_name)
        cond_save_name = f"ma_{sma_size}_{specifics}_{condition}_{resp}"
        saver(df=res_df, save_path=save_path, folder_name=RESP_TYPE_RATES, file_name=cond_save_name, type="")

    return res_df_game_filled


def mva_corr_test(res_df_game, res_df_replay, save, save_path, save_name, comp_save_dir):

    corr_method = 'kendall'
    corr_df_game = res_df_game.copy()
    corr_df_replay = res_df_replay.copy()
    corr_dict = {GAME: corr_df_game, REPLAY: corr_df_replay}
    corr_result = {GAME: None, REPLAY: None}
    for data in corr_dict:
        df = corr_dict[data]
        df.dropna(axis=0, subset=[AVG], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df = df[[WORLD_COL, AVG]]
        df_corr = stats.corr_test(data_name=TRUEPOSITIVE, data=df, col1=WORLD_COL, col2=AVG, corr_method=corr_method)
        corr_result[data] = df_corr

    if save:
        saver(df=res_df_game, save_path=save_path, folder_name=RESP_TYPE_RATES, file_name=save_name + "_" + GAME,
              type=os.path.join(comp_save_dir, MOVING_AVG))
        saver(df=res_df_replay, save_path=save_path, folder_name=RESP_TYPE_RATES, file_name=save_name + "_" + REPLAY,
              type=os.path.join(comp_save_dir, MOVING_AVG))

        # save the correlation test
        for vg_type in corr_result:
            data = corr_result[vg_type]
            saver(df=data, save_path=save_path, folder_name=RESP_TYPE_RATES,
                  file_name=f"corr_{corr_method}_{TRUEPOSITIVE}_{WORLD_COL}_{vg_type}",
                  type=os.path.join(comp_save_dir, MOVING_AVG))
    return


def mov_avg_stats(probe_data_dict, save, save_path, plot, colors):
    """
    This method manages all the moving-average-based calculations and plots of the behavioral data, and includes:
    - moving average of the HIT rate in the game vs in the replay: 
        - line plot with 2 lines (1 for game MVA 1 for replay MVA)
        - permutation cluster mass t-test between game and replay MVA's
    - moving average of the FACE HIT rate and OBJECT HIT rate for game and replay:
        - line plot with 4 lines (2 for face+object in game and in replay)
        - permutation cluster mass t-test between FACE AND OBJECT WITHIN THE GAME CONDITION (game, replay separately)
    :param probe_data_dict: 
    :param save: 
    :param save_path: 
    :param plot: 
    :param colors: 
    :return: 
    """
    comp_save_dir = "game_replay_comp"

    # STEP 1: create moving average hit rate data for game and for replay
    w = WORLDS
    sample_subcode = list(probe_data_dict.keys())[0]
    if sample_subcode.startswith(ECOG_LABS):
        w = WORLDS_ECOG
    # list of lists, we'll unpack next
    worlds_lol = [[i] * NUM_PROBES_PER_WORLD for i in w]
    worlds = [elem for world in worlds_lol for elem in world]
    hit_game = response_ma_specific(data_dict=probe_data_dict, condition=GAME, response=TRUEPOSITIVE, save=save,
                                    save_path=save_path, cond_len=len(worlds))
    hit_replay = response_ma_specific(data_dict=probe_data_dict, condition=REPLAY, response=TRUEPOSITIVE, save=save,
                                      save_path=save_path, cond_len=len(worlds))

    # STEP 2: moving average hit rate correlation test between game and replay
    hit_game[WORLD_COL] = worlds
    hit_replay[WORLD_COL] = worlds
    respone_nospace = TRUEPOSITIVE.replace(" ", "")
    save_name = f"ma_{SMA_WINDOW}_{respone_nospace.lower()}"
    """
    DEPRECATED
    mva_corr_test(hit_game, hit_replay, save, save_path, save_name=save_name, comp_save_dir=comp_save_dir)
    """
    # create a compact version of the df and test correlation between game world and response type rate

    # STEP 3: A permutation cluster-mass t-test on the time series of moving average hits
    # dataframes with just subjects in them
    hit_game_subs = hit_game.drop(columns=[AVG, STD, SE], inplace=False)
    hit_replay_subs = hit_replay.drop(columns=[AVG, STD, SE], inplace=False)
    cluster_p_vals, cluster_full_df, cluster_mass_df = stats.permutation_cluster_paired_ttest(data_cond_1=hit_game_subs,
                                                                                              data_cond_2=hit_replay_subs)

    # STEP 4: plot the moving averages, while noting (in horizontal bars) the significant clusters
    if plot:
        resp_list = [hit_game, hit_replay]
        # for the significance bar, take only significant clusters
        game_sig = cluster_p_vals[cluster_p_vals["significant"]]
        game_sig_x = [(x1, x2) for x1, x2 in zip(list(game_sig["cluster_starts"]), list(game_sig["cluster_ends"]))]
        # for the plot not to offset, fill nan values with the first non-nan value the column has (1st window):
        resp_filled_list = [x.fillna(method='bfill') for x in resp_list]
        labels = [TASK_DICT[GAME] + " " + HIT, TASK_DICT[REPLAY] + " " + HIT]
        colors = ['#8B2BE2', '#D55E00']
        plot_title = "Moving Average" + " True Positive" + " Rate"
        lineplotter.plot_avg_line(title=plot_title, trial_df_list=resp_filled_list,
                                  avg_col_list=[AVG, AVG], se_col_list=[SE, SE],
                                  label_list=labels, y_name=f"{TRUEPOSITIVE} Rate",
                                  color_list=colors, significance_bars_dict={"replaylines": {"y": 1.01, "x": game_sig_x}},
                                  save=save, save_path=save_path,
                                  save_name=f"ma_{SMA_WINDOW}_{respone_nospace.lower()}",
                                  sub_folder=os.path.join(RESP_TYPE_RATES, comp_save_dir, MOVING_AVG))

    if save:
        saver(df=cluster_p_vals, save_path=save_path, folder_name=RESP_TYPE_RATES,
              file_name=PERM_CLUS_PAIRED + "_clusters",
              type=os.path.join("game_replay_comp", MOVING_AVG))
        saver(df=cluster_full_df, save_path=save_path, folder_name=RESP_TYPE_RATES,
              file_name=PERM_CLUS_PAIRED + f"_full_cluster_df", type=os.path.join("game_replay_comp", MOVING_AVG))
        saver(df=cluster_mass_df, save_path=save_path, folder_name=RESP_TYPE_RATES,
              file_name=PERM_CLUS_PAIRED + f"_cluster_perm_list", type=os.path.join("game_replay_comp", MOVING_AVG))
    #return

    # STEP 5: DO THE SAME PER STIMULUS (face / object) in GAME AND REPLAY
    if sample_subcode.startswith(ECOG_LABS):
        COND_LEN_DICT = {GAME: 40, REPLAY: 50}
        x_ticks = 5
    else:
        COND_LEN_DICT = {GAME: 80, REPLAY: 100}
        x_ticks = 4
    resp_name_list = [f"{FACE}_{GAME}", f"{FACE}_{REPLAY}", f"{OBJ}_{GAME}", f"{OBJ}_{REPLAY}"]
    resp_dict = {x: None for x in resp_name_list}
    for target in [FACE, OBJ]:
        probe_data_dict_target = cherry_pick(sub_probe_data=probe_data_dict, game_column=CATEGORY, game_column_value=target,
                                           replay_column=REPLAY_TARGET, replay_column_value=target)
        for cond in [GAME, REPLAY]:
            hit_rate = response_ma_specific(data_dict=probe_data_dict_target, condition=cond, response=TRUEPOSITIVE,
                                         save=False, save_path=save_path, cond_len=COND_LEN_DICT[cond],
                                         sma_size=SMA_REPLAY_WINDOW, specific_name=f" {target}")
            resp_dict[f"{target}_{cond}"] = hit_rate

    # save
    for key in resp_dict.keys():
        saver(df=resp_dict[key], save_path=save_path, folder_name=RESP_TYPE_RATES,
              file_name=f"ma_{SMA_REPLAY_WINDOW}_tpr_by_stim_" + key,
              type=os.path.join(COND_FOLDER[CATEGORY], MOVING_AVG))

    # A PERMUTATION CLUSTER PAIRED T-TEST on the time series of moving average hits in faces and objects
    # dataframes with just subjects in them
    resp_dict_trimmed = {x: None for x in resp_name_list}
    for key in resp_dict_trimmed.keys():
        resp_dict_trimmed[key] = resp_dict[key].drop(columns=[AVG, STD, SE], inplace=False)

    # permutation test
    game_cluster_p_vals, game_cluster_full_df, game_cluster_mass_list = stats.permutation_cluster_paired_ttest(
        data_cond_1=resp_dict_trimmed[f"{FACE}_{GAME}"], data_cond_2=resp_dict_trimmed[f"{OBJ}_{GAME}"])

    replay_cluster_p_vals, replay_cluster_full_df, replay_cluster_mass_list = stats.permutation_cluster_paired_ttest(
        data_cond_1=resp_dict_trimmed[f"{FACE}_{REPLAY}"], data_cond_2=resp_dict_trimmed[f"{OBJ}_{REPLAY}"])

    if plot:
        resp_list = list(resp_dict.values())
        # for the significance bar, take only significant clusters
        game_sig = game_cluster_p_vals[game_cluster_p_vals["significant"]]
        game_sig_x = [(x1, x2) for x1, x2 in zip(list(game_sig["cluster_starts"]), list(game_sig["cluster_ends"]))]
        replay_sig = replay_cluster_p_vals[replay_cluster_p_vals["significant"]]
        replay_sig_x = [(x1, x2) for x1, x2 in
                        zip(list(replay_sig["cluster_starts"]), list(replay_sig["cluster_ends"]))]
        # for the plot not to offset, fill nan values with the first non-nan value the column has (1st window):
        resp_filled_list = [x.fillna(method='bfill') for x in resp_list]
        labels = [TASK_DICT[GAME] + " " + FACE, TASK_DICT[REPLAY] + " " + FACE,
                  TASK_DICT[GAME] + " " + OBJ, TASK_DICT[REPLAY] + " " + OBJ]
        colors = ['darkorange', 'tab:blue', 'hotpink', 'mediumpurple']
        plot_title = "Moving Average" + " True Positive" + " Rate" + " by Stimulus"
        lineplotter.plot_avg_line(title=plot_title, trial_df_list=resp_filled_list,
                                  avg_col_list=[AVG, AVG, AVG, AVG], se_col_list=[SE, SE, SE, SE],
                                  label_list=labels, y_name=f"{TRUEPOSITIVE} Rate",
                                  color_list=colors, significance_bars_dict={"gamelines": {"y": 0.22, "x": game_sig_x},
                                                                             "replaylines": {"y": 1.01,
                                                                                             "x": replay_sig_x}},
                                  save=save, save_path=save_path, x_tick_intervals=x_ticks,
                                  save_name=f"ma_{SMA_REPLAY_WINDOW}_tpr_by_stim_face_object",
                                  sub_folder=os.path.join(RESP_TYPE_RATES, COND_FOLDER[CATEGORY], MOVING_AVG))

    if save:
        saver(df=game_cluster_p_vals, save_path=save_path, folder_name=RESP_TYPE_RATES,
              file_name=PERM_CLUS_PAIRED + f"_{GAME}" + "_clusters",
              type=os.path.join(COND_FOLDER[CATEGORY], MOVING_AVG))
        saver(df=game_cluster_full_df, save_path=save_path, folder_name=RESP_TYPE_RATES,
              file_name=PERM_CLUS_PAIRED + f"_{GAME}" + f"_full_cluster_df",
              type=os.path.join(COND_FOLDER[CATEGORY], MOVING_AVG))
        saver(df=game_cluster_mass_list, save_path=save_path, folder_name=RESP_TYPE_RATES,
              file_name=PERM_CLUS_PAIRED + f"_{GAME}" + f"_cluster_perm_list",
              type=os.path.join(COND_FOLDER[CATEGORY], MOVING_AVG))

        saver(df=replay_cluster_p_vals, save_path=save_path, folder_name=RESP_TYPE_RATES,
              file_name=PERM_CLUS_PAIRED + f"_{REPLAY}" + "_clusters",
              type=os.path.join(COND_FOLDER[CATEGORY], MOVING_AVG))
        saver(df=replay_cluster_full_df, save_path=save_path, folder_name=RESP_TYPE_RATES,
              file_name=PERM_CLUS_PAIRED + f"_{REPLAY}" + f"_full_cluster_df",
              type=os.path.join(COND_FOLDER[CATEGORY], MOVING_AVG))
        saver(df=replay_cluster_mass_list, save_path=save_path, folder_name=RESP_TYPE_RATES,
              file_name=PERM_CLUS_PAIRED + f"_{REPLAY}" + f"_cluster_perm_list",
              type=os.path.join(COND_FOLDER[CATEGORY], MOVING_AVG))

    return


def response_rate_comp(game_summary_df, replay_summary_df, save, save_path, seperate_mods=False):
    """
    This function compares response type rate between game and replay. It PLOTS a boxplot comparing game and replay,
    as well as performs t-test (using the "stats" module) for each response type.
    :param game_summary_df: the game dataframe which is an output of the analyze_per_response_type function
    :param replay_summary_df: the replay dataframe which is an output of the analyze_per_response_type function
    :param save: whether to save the calculated data
    :param save_path: the highest folder in the BIDS-compatible file hierarchy
    """
    comp_save_dir = "game_replay_comp"

    summary_bayes_dict = dict()
    data_unified = pd.DataFrame()
    for resp in RESPONSE_RATE_NAMES:
        for id in ID_NAMES:
            data_unified[id] = game_summary_df[id]
        data_unified[resp + "_game"] = game_summary_df[resp]
        data_unified[resp + "_replay"] = replay_summary_df[resp]
        # statistical analysis: t-test and Bayes Factors
        bayes = stats.paired_t_test_pg(data_unified, col_1=resp + "_game", col_2=resp + "_replay")
        if bayes is not None:
            summary_bayes_dict[resp] = bayes

        # plot
        resp_title = RESPONSE_RATE_NAMES_TITLE[resp]
        resp_title_name = resp_title.replace(" ", "_").lower()
        # save
        if save:
            saver(df=data_unified, save_path=save_path, folder_name=RESP_TYPE_RATES,
                  file_name=f"comp_{resp_title_name}", type=comp_save_dir)

    # t-test and Bayes Factors : a csv with comparison between game and replay of all response types
    if bool(summary_bayes_dict):
        summary_bayes = pd.DataFrame.from_dict(summary_bayes_dict, orient='index')
        summary_bayes.columns = BAYES_COLS

    if save:
        if bool(summary_bayes_dict):
            saver(df=summary_bayes, save_path=save_path, folder_name=RESP_TYPE_RATES,
                  file_name=REPLAY + "_" + GAME + "_all_resps_" + BF_TTEST_PAIRED, type=comp_save_dir)

    # plot seen response type for game against replay - meaning, both for True Positive and False alarms, compare
    # game and replay on the same plot
    if not seperate_mods:
        colors = sns.color_palette("colorblind", 2)
        colors = ["#8B2BE2", "#D55E00"]
        boxplotter.plot(data=data_unified,
                        data_col_order=['tp_rate_game', 'tp_rate_replay', 'fp_rate_game', 'fp_rate_replay'],
                        data_name_order={'tp_rate_game': TASK_DICT[GAME],
                                         'tp_rate_replay': TASK_DICT[REPLAY],
                                         'fp_rate_game': TASK_DICT[GAME],
                                         'fp_rate_replay': TASK_DICT[REPLAY]},
                        scatter=False, raincloud=True, sub_line=[[0, 1], [2, 3]],
                        horizontal_lines=[{'y': None, 'xmin': 0.7, 'xmax': 2.3, 'label': "Stimulus Present (Seen/Hit)"},
                                          {'y': None, 'xmin': 2.7, 'xmax': 4.3,
                                           'label': "Stimulus Absent (Seen Blank/False Alarm)"}],
                        color_list=[colors[0], colors[1], colors[0], colors[1]],
                        plot_title=f"Seen Response Rate in Task", plot_x_label="Task",
                        plot_y_label=f"% Seen", save_plot=True, save_path=save_path,
                        save_name="seen_resp_rate_in_task",
                        sub_folder=RESP_TYPE_RATES)

        boxplotter.plot(data=data_unified,
                        data_col_order=['tp_rate_game', 'tp_rate_replay'],
                        data_name_order={'tp_rate_game': TASK_DICT[GAME],
                                         'tp_rate_replay': TASK_DICT[REPLAY]},
                        scatter=False, raincloud=True, sub_line=[[0, 1]],
                        color_list=["#188A8C", "#DB8906"],
                        plot_title=f"Seen Response Rate in Task", plot_x_label="Task",
                        plot_y_label=f"% Seen", save_plot=True, save_path=save_path,
                        save_name="seen_resp_rate_in_task_OnlyHits",
                        sub_folder=RESP_TYPE_RATES)
    else:
        colors = sns.color_palette("colorblind", 8)
        data_unified[data_reader.MODALITY] = data_unified["Lab"].apply(lambda x: data_reader.METHOD[x])
        boxplotter.rain_per_mod(data_unified,
                        col_names=['dAT', 'at', 'dAT', 'AT'],
                        cols=['tp_rate_game', 'tp_rate_replay', 'fp_rate_game', 'fp_rate_replay'],
                        sub_line=[[0, 1], [2, 3]],
                        horizontal_lines=[{'y': None, 'xmin': 0.7, 'xmax': 2.3, 'label': "Stimulus Present (Seen/Hit)"},
                                          {'y': None, 'xmin': 2.7, 'xmax': 4.3,
                                           'label': "Stimulus Absent (Seen Blank/False Alarm)"}],
                        color_dict={data_reader.MEEG: [colors[0], colors[1], colors[0], colors[1]],
                                    data_reader.FMRI: [colors[2], colors[4], colors[2], colors[4]]},
                        plot_title=f"Seen Response Rate in Task", plot_x_label="Task",
                        plot_y_label=f"% Seen", save_plot=True, save_path=save_path,
                        save_name="seen_resp_rate_in_task_per_mod",
                        save_folder=RESP_TYPE_RATES)

    for lab in list(data_unified["Lab"].unique()):
        data_unified_lab = data_unified[data_unified["Lab"] == lab]
        colors = sns.color_palette("colorblind", 2)
        colors = ["#8B2BE2", "#D55E00"]
        boxplotter.plot(data=data_unified_lab,
                        data_col_order=['tp_rate_game', 'tp_rate_replay', 'fp_rate_game', 'fp_rate_replay'],
                        data_name_order={'tp_rate_game': TASK_DICT[GAME],
                                         'tp_rate_replay': TASK_DICT[REPLAY],
                                         'fp_rate_game': TASK_DICT[GAME],
                                         'fp_rate_replay': TASK_DICT[REPLAY]},
                        scatter=False, raincloud=True, sub_line=[[0, 1], [2, 3]],
                        horizontal_lines=[{'y': None, 'xmin': 0.7, 'xmax': 2.3, 'label': "Stimulus Present (Seen/Hit)"},
                                          {'y': None, 'xmin': 2.7, 'xmax': 4.3,
                                           'label': "Stimulus Absent (Seen Blank/False Alarm)"}],
                        color_list=[colors[0], colors[1], colors[0], colors[1]],
                        plot_title=f"Seen Response Rate in Task", plot_x_label="Task",
                        plot_y_label=f"% Seen", save_plot=True, save_path=save_path,
                        save_name=f"seen_resp_rate_in_task_{lab}",
                        sub_folder=RESP_TYPE_RATES)
        """
        (data, cols=data_col_order, col_names=[data_name_order[i] for i in data_col_order],
                     color_list=colors, sub_line=sub_line,
                     plot_title=plot_title, plot_x_label=plot_x_label, plot_y_label=plot_y_label, save_plot=save_plot,
                     save_path=save_path, save_folder=sub_folder, horizontal_lines=horizontal_lines,
                     save_name=f"{plot_savename}", custom_ylim=custom_ylim, skip=skip)
        """

    saver(df=data_unified, save_path=save_path, folder_name=RESP_TYPE_RATES, file_name="seen_resp_rate_in_task",
          type=comp_save_dir)



    return


def analyze_category_data(data_dict, cond_dict, probe_type, plot=True, save=False, save_path=""):
    for cond in cond_dict.keys():
        cond_name = COND_FOLDER[cond]
        category_sub_folder_path = os.path.join(RESP_TYPE_RATES, cond_name)

    category_dict = create_cond_dict(cond_dict)

    for cond in cond_dict.keys():
        print("-----------------")
        print(cond)
        for sub_cond in cond_dict[cond]:
            print(sub_cond)
            cond_sub_data_list = []
            for sub_code in data_dict:
                sub_all = data_dict[sub_code][probe_type]  # subject probe dataframe in GAME/REPLAY
                if cond != LEFT_RIGHT and cond != UP_BOTTOM:
                    data = sub_all[sub_all[cond] == sub_cond]  # subject df FILTERED BY CATEGORY TYPE
                else:
                    data = sub_all[sub_all[LOC].str.contains(sub_cond)]
                # analyze the probe dataframe to extract info about TP, FP, TN, FN
                sub_data, sub_sdt_summary = response_type_breakdown(sub_code, data, probe_type)
                # add it to the list that is going to be the dataframe
                cond_sub_data_list.append(sub_data)
            category_dict[cond][sub_cond] = pd.DataFrame(cond_sub_data_list, columns=NAMES)
            # make sure that in all of the dfs the order is the same
            category_dict[cond][sub_cond] = category_dict[cond][sub_cond].sort_values(by=['Subject'])

    # Now, we want to divide the category_dict data according to responses. Meaning, for each condition (e.g., location)
    # we want to have a response type (e.g., TP), and in a single df compare between different sub conditions
    # (e.g., TP in top right, top left, bottom right, bottom left). So we'll take this information from category_dict
    # and re-organize it in response_rate_cond
    response_rate_cond = dict()
    for cond in category_dict.keys():  # for each condition, e.g., Location
        response_rate_cond[cond] = dict()
        for resp in RESPONSE_RATE_NAMES:  # for each response type, e.g., TP
            response_rate_cond[cond][resp] = pd.DataFrame()
            # have Lab and Subject as columns in this dataframe (take them from an arbitrary df, as the order
            # should be the same)
            response_rate_cond[cond][resp]['Lab'] = next(iter(category_dict[cond].values()))['Lab']
            response_rate_cond[cond][resp]['Subject'] = next(iter(category_dict[cond].values()))['Subject']
            for sub_cond in category_dict[cond]:  # for each sub condition, e.g., top right
                subcond_resp_data = category_dict[cond][sub_cond][resp]
                response_rate_cond[cond][resp][sub_cond] = subcond_resp_data

    resp_dict = {'tp_rate': TP, 'fp_rate': FP, 'tn_rate': TN, 'fn_rate': FN}
    for cond in response_rate_cond.keys():
        cond_name = re.sub(r"(\w)([A-Z])", r"\1 \2", cond)
        cond_stat_dict = dict()
        resp_tukey_dict = dict()
        for resp in response_rate_cond[cond].keys():
            data = response_rate_cond[cond][resp]
            cols = list(data.columns.values)[2:]  # don't take Lab / Subject
            if cond == CATEGORY:  # face/object/none -> handle differently so we won't present None in Tp/Fn etc
                if resp == 'tp_rate' or resp == 'fn_rate':  # true responses don't have None as stimulus type
                    cols = [FACE, OBJ]
                if resp == 'fp_rate' or resp == 'tn_rate':
                    cols = [BLANK]
            if plot:
                cols_names = {x: re.sub(r"(\w)([A-Z])", r"\1 \2", x) for x in cols}  # add space between caps
                # choose colors:
                color_list = ["orangered", "sandybrown", "gold", "steelblue"]
                if len(cols) == len(color_list):
                    colors = color_list
                if len(cols) == 2:
                    colors = [color_list[1], color_list[3]]
                if len(cols) == 1:
                    colors = [color_list[0]]
                boxplotter.plot(data=data, data_col_order=cols, data_name_order=cols_names, raincloud=True,
                                scatter=False, sub_line=[list(range(len(cols)))], color_list=colors,
                                plot_title=f"{resp_dict[resp]} Rates by {cond_name} in {probe_type}",
                                plot_x_label=cond_name, plot_y_label=f"{resp_dict[resp]} Rate", save_plot=save,
                                save_name=f"{resp_dict[resp]}_rate_by_{cond_name}_in_{probe_type}",
                                save_path=save_path, sub_folder=category_sub_folder_path)
                if save:
                    saver(df=data, save_path=save_path, folder_name=RESP_TYPE_RATES,
                          file_name=f"{resp_dict[resp]}_rate_by_{cond_name}_in_{probe_type}", type=COND_FOLDER[cond])

            # perform an ANOVA on condition; for each response type, check if there's a difference in response type
            # rate between different sub conditions
            if len(cols) > 1:  # no comparison if there's only one data group
                if len(cols) > 2:  # ANOVA
                    cond_stat_dict[resp], resp_tukey_dict[resp] = stats.repeated_measures_anova(data_name=resp, data=data, columns=cols)
                    stat = "RM_ONE_WAY_ANOVA"
                elif len(cols) == 2:
                    cond_stat_dict[resp] = stats.paired_t_test_pg(data=data, col_1=cols[0], col_2=cols[1], df=True)
                    stat = BF_TTEST_PAIRED

        cond_stat_df = pd.DataFrame()
        for key, value in cond_stat_dict.items():
            df = value
            df.loc[:, 'response_type'] = key
            cond_stat_df = pd.concat([df, cond_stat_df], 0)
        if save:
            saver(df=cond_stat_df, save_path=save_path, folder_name=RESP_TYPE_RATES,
                  file_name=f"{stat}_resp_rate_by_{COND_FOLDER[cond]}_{probe_type}", type=COND_FOLDER[cond])
            if key in resp_tukey_dict:  # we ran an ANOVA (not a ttest)
                if resp_tukey_dict[key] is not None:  # we ran tukey for this (ANOVA was significant)
                    saver(df=resp_tukey_dict[key], save_path=save_path, folder_name=RESP_TYPE_RATES,
                          file_name=f"{stat}_tukey_{key}_rate_by_{COND_FOLDER[cond]}_{probe_type}", type=COND_FOLDER[cond])

    return response_rate_cond


def cherry_pick(sub_probe_data, game_column, game_column_value, replay_column, replay_column_value):
    result_dict = dict()
    for sub in sub_probe_data:
        result_dict[sub] = dict()
        game_orig = sub_probe_data[sub][GAME]
        replay_orig = sub_probe_data[sub][REPLAY]
        result_dict[sub][GAME] = game_orig[game_orig[game_column] == game_column_value]
        result_dict[sub][REPLAY] = replay_orig[replay_orig[replay_column] == replay_column_value]
        result_dict[sub][REPLAY_TARGETS] = sub_probe_data[sub][REPLAY_TARGETS]
    return result_dict


def compare_SDT_game_replay(game_list, replay_list, save_path, column='d'):
    game_dprime = list()
    replay_dprime = list()
    
    for sub_ind in range(len(game_list)):
        sub_game = game_list[sub_ind].loc[0, column]
        sub_replay = replay_list[sub_ind].loc[0, column]
        game_dprime.append(sub_game)
        replay_dprime.append(sub_replay)

    test_df = pd.DataFrame({f"Game_{column}": game_dprime, f"Replay_{column}": replay_dprime})
    stat = stats.paired_t_test_pg(data=test_df, col_1=f"Game_{column}", col_2=f"Replay_{column}")
    saver(df=stat, save_path=save_path, folder_name=RESP_TYPE_RATES, file_name=f"SDT_{column}_game_replay", type="")
    return


def analyze_loc(probe_data_dict, save, save_path):
    """
    analyze_category_data type returns a dictionary for each condition (e.g. location). In it, for each RESPONSE type
    there's a dataframe with the response rate of that response in the location.
    for example: in game_category_data[LOC][tp_rate], in the column for bottom right, the number represents
    how many OUT of the bottom right stimuli are true positives.
    :param probe_data_dict:
    :param save:
    :param save_path:
    :return:
    """
    conditions = {LOC: [TR, TL, BR, BL], CATEGORY: [FACE, OBJ, BLANK], UP_BOTTOM: ["Bottom", "Top"], LEFT_RIGHT: ["Left", "Right"]}

    # GAME
    game_category_data = analyze_category_data(data_dict=probe_data_dict, cond_dict=conditions, probe_type=GAME,
                                               plot=False, save=save, save_path=save_path)
    # REPLAY
    replay_category_data = analyze_category_data(data_dict=probe_data_dict, cond_dict=conditions, probe_type=REPLAY,
                                                 plot=False, save=save, save_path=save_path)

    # generate plots which describe response rate (TP, FP, TN, FN) in task (GAME/REPLAY) by condition (LOCATION,TYPE)
    # meaning, for 'Bottom Right' , what is the True Positive Rate in that location?
    for cond in conditions:
        game_data = game_category_data[cond]
        replay_data = replay_category_data[cond]
        cond_name = re.sub('([A-Z])', r' \1', cond).title()
        for response_type in game_data.keys():
            if cond == CATEGORY:  # stimulus type: face, object, or blank = in which no stimulus is presented
                if response_type == 'tp_rate' or response_type == 'fn_rate':  # meaning, face/object, no blanks here
                    sub_conds = conditions[cond][:-1]
                else:  # blanks
                    sub_conds = conditions[cond]
            else:  # location
                sub_conds = conditions[cond]
            sub_conds_names = [re.sub('([A-Z])', r' \1', s).title() for s in sub_conds]
            game_resp = game_data[response_type].rename(columns={x: x + f"_{GAME}"
                                                                 for x in game_data[response_type].columns.values[2:]})
            replay_resp = replay_data[response_type].rename(columns={x: x + f"_{REPLAY}"
                                                                     for x in replay_data[response_type].columns.values[2:]})
            non_shared_cols = replay_resp.columns.difference(game_resp.columns)
            unified_df = pd.DataFrame.merge(game_resp, replay_resp[non_shared_cols], left_index=True, right_index=True, how='outer')
            cols_game = [x + f"_{GAME}" for x in sub_conds]
            cols_replay = [x + f"_{REPLAY}" for x in sub_conds]
            cols_order = cols_game + cols_replay
            lines = [list(range(len(sub_conds), len(sub_conds) * 2))]
            if (cond == CATEGORY and (response_type == 'tp_rate' or response_type == 'fn_rate')) or cond != CATEGORY:
                lines.append(list(range(len(sub_conds))))
            colors = ['#003544', '#ad501d', '#003544', '#ad501d']  # face (game), object (game), face (replay), object (replay)
            #colors_game = sns.color_palette("flare", len(sub_conds))
            colors_game = ['#003544' if i%2 == 0 else '#ad501d' for i in range(len(sub_conds))]
            colors_replay = ['#003544' if i%2 == 0 else '#ad501d' for i in range(len(sub_conds))]
            #colors_replay = sns.color_palette("crest", len(sub_conds))
            horiz = [{'y': None, 'xmin': 0.7, 'xmax': len(sub_conds) + 0.3, 'label': TASK_DICT[GAME]},
                     {'y': None, 'xmin': len(sub_conds) + 1 - 0.3,
                      'xmax': len(sub_conds) + len(sub_conds) + 0.3, 'label': TASK_DICT[REPLAY]}]
            boxplotter.plot(data=unified_df, data_col_order=cols_order,
                            data_name_order=dict(zip(cols_order, sub_conds_names + sub_conds_names)),
                            scatter=False, raincloud=True, sub_line=lines, color_list=colors_game + colors_replay,
                            horizontal_lines=horiz,
                            plot_title=f"{RESPONSE_RATE_NAMES_MAPPING[response_type]} Rate in Task by {cond_name}",
                            plot_x_label=f"{cond_name}",
                            plot_y_label=f"% {RESPONSE_RATE_NAMES_MAPPING[response_type]}",# Responses in a Specific {cond_name}",
                            save_plot=True, save_name=f"{RESPONSE_RATE_NAMES_MAPPING[response_type].lower()}_by_{cond_name.lower()}",
                            save_path=save_path, sub_folder=os.path.join(RESP_TYPE_RATES, COND_FOLDER[cond]))

            saver(df=unified_df, save_path=save_path, folder_name=RESP_TYPE_RATES,
                  file_name=f"{RESPONSE_RATE_NAMES_MAPPING[response_type].lower()}_by_{cond_name.lower()}",
                  type=COND_FOLDER[cond])

            saver(df=unified_df.describe(), save_path=save_path, folder_name=RESP_TYPE_RATES,
                  file_name=f"{RESPONSE_RATE_NAMES_MAPPING[response_type].lower()}_by_{cond_name.lower()}_stats",
                  type=COND_FOLDER[cond])
    return


def analyze_response_types(probe_data_dict, plot=True, save=False, save_path="", seperate_mods=False):
    """
    This function manages all the general analyses that have to do with RESPONSE TYPE. These include:
    - General analysis of RESPONSE TYPES (TP/FP/TN/FN): THEIR RATES WITHIN GAME AND REPLAY
    - Analysis of DIFFICULT/PERFORMANCE per RESPONSE TYPE: creating summary tables of AVG/MEDIAN DIFFICULTY/PERFORMANCE
    per RESPONSE TYPE, as well as response types themselves (% of all responses) per subject.
    :param probe_data_dict: a dictionary in which {key=sub, val={GAME:probe_df, REPLAY:probe_df}}
    :param plot: whether to plot the data
    :param save: whether to save the data and the plots
    :param save_path: the general data folder (highest folder in the BIDS-compatible folder structure)
    :return:
    """
    analysis_path = data_saver.create_analysis(save_path)

    # STEP 1: OVERALL RESPONSE TYPE RATES, ACROSS GAME/REPLAY
    # Create a summary table of response type rates, difficulty and performance levels in GAME! Plot as BOXPLOT
    game_response_rate_data, game_SDT_data = analyze_per_response_type(probe_data_dict, GAME, False, save, analysis_path)
    # Create a summary table of response type rates, difficulty and performance levels in REPLAY! Plot as BOXPLOT
    replay_response_rate_data, replay_SDT_data = analyze_per_response_type(probe_data_dict, REPLAY, False, save, analysis_path)

    # compare SDT data between game and replay
    # d prime
    compare_SDT_game_replay(game_list=game_SDT_data, replay_list=replay_SDT_data, save_path=analysis_path, column='d')
    # criterion (beta)
    compare_SDT_game_replay(game_list=game_SDT_data, replay_list=replay_SDT_data, save_path=analysis_path, column='beta')

    # For each response type, COMPARE that response type between condition : AT / dAT across subjects: T-TEST and plot
    response_rate_comp(game_response_rate_data, replay_response_rate_data, save, analysis_path, seperate_mods=seperate_mods)

    # STEP 2: Plot and save analysis of difficulty and performance per response type:
    # GAME + DIFFICULTY
    diff_perf_per_response_type(summary_table=game_response_rate_data, diff_1_perf_0=1, save_path=analysis_path)
    # GAME + PERFORMANCE
    diff_perf_per_response_type(summary_table=game_response_rate_data, diff_1_perf_0=0, save_path=analysis_path)

    # STEP 3: Create a moving average for subjects' hit rate across probes:
    # - for hit rate in GAME vs REPLAY
    # - for hit rate in FACE vs OBJECT for both game and replay (separately)
    # As we only care about HIT rates, fillers in the replay are not interesting and should not be considered.
    probe_data_dict_nofillers = copy.deepcopy(probe_data_dict)
    for sub in probe_data_dict_nofillers.keys():
        with_f = probe_data_dict[sub][REPLAY]
        probe_data_dict_nofillers[sub][REPLAY] = with_f[with_f["type"] != 'LOCALIZER_FILLER']
    mov_avg_stats(probe_data_dict=probe_data_dict_nofillers, save=save, save_path=analysis_path, plot=plot, colors={GAME: 'orange', REPLAY: 'tab:green'})

    # STEP 4: analyze response type rates according to different probe categories
    analyze_loc(probe_data_dict, save, analysis_path)
    return


def saver(df, save_path, folder_name, file_name, type=ACROSS):
    if type != "":
        type_split = os.path.split(type)
        data_saver.create_dir(os.path.join(save_path, folder_name, type_split[0]))
        data_saver.create_dir(os.path.join(save_path, folder_name, type_split[0], type_split[1]))
    data_saver.safe_save(os.path.join(save_path, folder_name, type), file_name + f".csv")
    df.to_csv(os.path.join(os.path.join(save_path, folder_name, type), file_name + f".csv"))


