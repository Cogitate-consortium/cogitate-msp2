import pandas as pd
import numpy as np
import os
import lineplotter
import data_saver
import stats

"""
This module manages all the analyses related to difficulty and performance data across trials. 
analyze_diff_perf method manages all the analyses to be performed:
avg_per_method : calculates and saves data related to average difficulty/performance across the game. The saved 
resulting data is a table in which each row is a trial, and each column is a subject's difficulty/performance in that
trial. Then there are average, std and se columns. Based on this dataframe (one for difficulty and one for performance), 
a plot containing both the average and the moving average is created (by calling the lineplotter module). 

@author: RonyHirsch
"""

WORLDS = [0, 1, 2, 3]
WORLDS_ECOG = [0, 1]
ECOG_LABS = ("SE", "SF", "SX")
NUM_PROBES_PER_WORLD = 50
GAME = 'GAME'

# data for SMA
SMA_WINDOW = 13
SMA_MIN_PERIOD = 1

# data column names
DIFF = 'difficulty'
PERF = 'performance'
WORLDID = 'activeWorldID'
ACROSS_TITLE = "across game"
# result df columns
WORLD_COL = 'world'
AVG = 'average'
STD = 'standard deviation'
SE = 'standard error'
SMA = 'SMA'

# responseEvaluation types
EVAL = 'responseEvaluation'
TRUEPOSITIVE = 'TruePositive'
FALSEPOSITIVE = 'FalsePositive'
TRUENEGATIVE = 'TrueNegative'
FALSENEGATIVE = 'FalseNegative'

# names for saving
# folders
ACROSS = "across_game"
PER_RESP = "per_response_type"
DIFF_V_PERF = "performance_vs_difficulty"


def correct_for_duplicate_probes(sub_dataframe, probes):
    """
    If for some ungodly reason, a subject has more than NUM_PROBES_PER_WORLD entries, it means the subject had at least
    one aborted+restarted level we need to correct for. Otherwise this subject cannot be analyzed with the rest of the
    data.
    :param sub_dataframe:
    :return:
    """
    if probes < sub_dataframe.shape[0]:
        sub_dataframe_result = sub_dataframe.drop_duplicates(subset='questionID', keep="last", inplace=False)
    else:
        sub_dataframe_result = sub_dataframe
    return sub_dataframe_result


def col_vals_per_world(probe_data_dict, worlds, col, ecog: bool, only_seen=False, only_unseen=False):
    """
    Create a dataframe in which each row is a trial, and each column (but the first) is a subjects' "col" value within
    that specific trial. The first column represents the game-world this trial belongs to.
    :param probe_data_dict: dictionary in which {key=sub, val={GAME:probe_df, REPLAY:probe_df}}
    :param worlds: a list of all trials in all worlds, such that for every world there are NUM_PROBES_PER_WORLD entries
    of that world in the list.
    :param col: the name of the column from the probeDetails table we want to extract data from
    :return: the resulting df
    """
    all_worlds = WORLDS_ECOG if ecog else WORLDS
    col_per_trial = pd.DataFrame()
    col_per_trial[WORLD_COL] = worlds
    # iterate subject GAME data
    for sub in probe_data_dict.keys():
        sub_data = probe_data_dict[sub]
        game_data = correct_for_duplicate_probes(sub_data[GAME], len(worlds))
        total_col = list()  # this is where the "col" column of the subject will be
        # count how many trials the subject has in each world
        trials_in_world = game_data[WORLDID].value_counts()  # how many trials the subject has in each world
        for w in all_worlds:
            world_trials = game_data[game_data[WORLDID] == w]
            if only_seen:
                world_trials.loc[~world_trials[EVAL].str.contains(TRUEPOSITIVE), col] = np.nan
            elif only_unseen:
                world_trials.loc[~world_trials[EVAL].str.contains(FALSENEGATIVE), col] = np.nan
            col_in_w = list(world_trials[col])  # "col" values in a specific world
            if w not in trials_in_world:
                n = 0
            else:
                n = trials_in_world[w]
            if n < NUM_PROBES_PER_WORLD:  # if subject has less trials than expected
                col_in_w.extend([np.nan] * (NUM_PROBES_PER_WORLD - n))  # fill missing trials with nans
            total_col.extend(col_in_w)
        col_per_trial[sub] = total_col
    avg = col_per_trial.iloc[:, 1:].mean(axis=1)
    std = col_per_trial.iloc[:, 1:].std(axis=1)
    se = std / np.sqrt(col_per_trial.iloc[:, 1:].shape[1])
    col_per_trial[AVG] = avg
    col_per_trial[STD] = std
    col_per_trial[SE] = se
    return col_per_trial


def calc_rolling_avg(trial_df):
    """
    Calculates a rolling average (standard moving average) for each subject trial data, to create a df in which
    each column is a subject's rolling average data and the last column is average across subjects.
    min_periods = minimum number of observations in window required to have a value (otherwise result is NA).
    min_periods will default to the size of the window unless we specify otherwise.
    :param trial_df: dataframe containing a column for each subject's raw data and summary columns
    :return: dataframe containing a column for each subject's moving average and an average columns
    """
    result_df = pd.DataFrame()
    cols = trial_df.columns.tolist()
    # exclude non-subject columns
    sub_cols = [x for x in cols if x not in [WORLD_COL, AVG, STD, SE]]
    result_df[WORLD_COL] = trial_df[WORLD_COL]
    for col in sub_cols:
        sub_data = trial_df[col]
        rolling_avg_data = sub_data.rolling(SMA_WINDOW, min_periods=SMA_MIN_PERIOD).mean()
        result_df[col] = rolling_avg_data
    avg = result_df.iloc[:, 1:].mean(axis=1)
    result_df[AVG] = avg
    return result_df


def avg_per_method(probe_data_dict, diff_1_perf_0, ecog: bool, plot=True, save=False, save_path="", only_seen=False, only_unseen=False):
    """
     This function manages the entire calculation, plotting and saving of the average difficulty / performance
    across trials (=probes). Depending on ECOG/not, it sends expectations to # of worlds and trials to the analysis
    function. Then it calls plotting and saving methods.
    :param probe_data_dict: dictionary in which {key=sub, val={GAME:probe_df, REPLAY:probe_df}}, containing only ECOG /
    only non-ECOG subjects
    :param diff_1_perf_0: 1 if we want to analyze difficulty data, 0 for performance
    :param ecog: boolean, whether the data in probe_data_dict belongs to ECOG or not
    :param plot: whether to plot the data
    :param save: whether to save the data and plots
    :param save_path: the path where to the highest folder in the BIDS-compatible directory hierarchy, under which
    the data will be appropriately saved.
    :return: the dataframe in which each row is a trial and each column is the diff_1_perf_0 value of a subject in
    that trial (with the 1st column indicating which world this trial is from)
    """
    suffix = "_ECOG" if ecog else "_MEG_fMRI"
    dat_type = DIFF if diff_1_perf_0 == 1 else PERF
    plot_title = dat_type + " " + ACROSS_TITLE
    save_name = dat_type + suffix

    # create a list to be a df column indicating which world a certain trial (probe) belongs to:
    w = WORLDS_ECOG if ecog else WORLDS
    worlds_lol = [[i] * NUM_PROBES_PER_WORLD for i in w]  # list of lists, we'll unpack next
    worlds = [elem for world in worlds_lol for elem in world]

    # create a trial-based difficulty table: each row = trial (GAME PROBE), each col = difficulty/perfrmance
    # in that trial for a single subject
    data_per_trial = col_vals_per_world(probe_data_dict, worlds, dat_type, ecog=ecog, only_seen=only_seen, only_unseen=only_unseen)
    # calculate a df of simple moving averages per subject in which col represents moving average per sub and the
    # average column is the average of the simple moving averages
    sma_per_trial = calc_rolling_avg(trial_df=data_per_trial)

    # create a compact version of the df and test correlation between game world and response type rate
    corr_method = 'kendall'
    corr_df_game = sma_per_trial.copy()
    corr_df_game.reset_index(drop=True, inplace=True)
    corr_df_game = corr_df_game[[WORLD_COL, AVG]]
    df_corr = stats.corr_test(data_name=dat_type, data=corr_df_game, col1=WORLD_COL, col2=AVG, corr_method=corr_method)

    if plot:
        lineplotter.plot_avg_line(title=plot_title, trial_df_list=[data_per_trial, sma_per_trial],
                                  avg_col_list=[AVG, AVG], se_col_list=[SE, None],
                                  color_list=['lightskyblue', 'tab:blue'],
                                  label_list=[AVG, SMA], y_name=f"Average {dat_type}", save=save,
                                  save_name=save_name, save_path=save_path, sub_folder=os.path.join(dat_type, ACROSS))
    if save:
        saver(df=data_per_trial, save_path=save_path, folder_name=dat_type, file_name=save_name, type=ACROSS)
        saver(df_corr, save_path, dat_type, f"corr_{corr_method}_{dat_type}_{WORLD_COL}")
    return data_per_trial, sma_per_trial


def analyze_diff_perf(probe_data_dict, save=False, save_path="", seen_only=False, unseen_only=False):
    """
    Main function in this module, handles all the general analyses concerned with performance and difficulty.
    The first thing it does is separate the subjects to 2: ECOG patients and others, as ECOG have a different game
    (e.g., less worlds and less trials).
    Then, it calls an "avg_per_method" function which calculates and plots difficulty and performance across all
    trials (=PROBES) in the game.
    :param probe_data_dict: dictionary in which {key=sub, val={GAME:probe_df, REPLAY:probe_df}}
    :param plot: whether to plot the resulting average difficulty and performance
    :param save: whether to save the plot and data
    :param save_path: the path where to the highest folder in the BIDS-compatible directory hierarchy, under which
    the data will be appropriately saved.
    """

    analysis_path = data_saver.create_analysis(save_path)

    if seen_only or unseen_only:
        save_diff = (False and save)
    else:
        save_diff = (True and save)

    if seen_only:
        seen_name = "_seen"
    elif unseen_only:
        seen_name = "_unseen"
    else:
        seen_name = ""

    # split probe_data_dict to ECOG and REST as ECOG have a different number of worlds
    probe_data_dict_ECOG = {n: probe_data_dict[n] for n in list(probe_data_dict.keys()) if n.startswith(ECOG_LABS)}
    probe_data_dict_REST = {n: probe_data_dict[n] for n in list(probe_data_dict.keys())
                            if not any(x in n for x in ECOG_LABS)}

    # for non-empty datasets, calculate and plot average difficulty and performance ACROSS trials:
    if bool(probe_data_dict_ECOG):
        diff, diff_sma = avg_per_method(probe_data_dict_ECOG, diff_1_perf_0=1, ecog=True, plot=False, save=save_diff,
                                        save_path=analysis_path, only_seen=seen_only, only_unseen=unseen_only)
        perf, perf_sma = avg_per_method(probe_data_dict_ECOG, diff_1_perf_0=0, ecog=True, plot=False, save=save_diff,
                                        save_path=analysis_path, only_seen=seen_only, only_unseen=unseen_only)
    if bool(probe_data_dict_REST):
        diff, diff_sma = avg_per_method(probe_data_dict_REST, diff_1_perf_0=1, ecog=False, plot=False, save=save_diff,
                                        save_path=analysis_path, only_seen=seen_only, only_unseen=unseen_only)
        perf, perf_sma = avg_per_method(probe_data_dict_REST, diff_1_perf_0=0, ecog=False, plot=False, save=save_diff,
                                        save_path=analysis_path, only_seen=seen_only, only_unseen=unseen_only)

    lineplotter.plot_avg_line(title="Moving Average Difficulty and Performance Across Game",
                              trial_df_list=[diff, diff_sma, perf, perf_sma], avg_col_list=[AVG, AVG, AVG, AVG],
                              se_col_list=[SE, None, SE, None], label_list=["Average Difficulty", "SMA Difficulty",
                                                                            "Average Performance", "SMA Performance"],
                              color_list=["#ef7194", "#e83758", "#899dae", "#486682"],
                              y_name=f"Moving Average", save=save, save_name=f"diff_perf_across_game{seen_name}",
                              save_path=analysis_path, sub_folder=DIFF_V_PERF)

    f_list = [diff, diff_sma, perf, perf_sma]
    f_suffix = ["diff", f"diff_sma_{SMA_WINDOW}", "perf", f"perf_sma_{SMA_WINDOW}"]
    for i in range(len(f_list)):
        saver(df=f_list[i], save_path=analysis_path, folder_name=DIFF_V_PERF, file_name=f"diff_perf_across_game_{f_suffix[i]}{seen_name}", type="")

    return diff, diff_sma, perf, perf_sma


def saver(df, save_path, folder_name, file_name, type=ACROSS):
    if type != "":
        data_saver.create_dir(os.path.join(save_path, folder_name, type))
    data_saver.safe_save(os.path.join(save_path, folder_name, type), file_name + f".csv")
    df.to_csv(os.path.join(os.path.join(save_path, folder_name, type), file_name + f".csv"))
