import plotter
import pycircstat
import itertools
from QualityChecker import *
from ET_data_extraction import *
import ET_qc_manager
import astropy.stats
import gc
import data_saver
from multiprocessing import Process

""" EYE TRACKING DATA ANALYSIS MODULE 

This module manages all the eye-tracking analyses performed on the exp.2 data. 
It extracts all the relevant data and then calls and divides analyses to 3 data types:
1. Fixations (analyze_fixations)
2. Blinks (analyze_blinks)
3. Saccades (analyze_saccades)
All analyses and plots are managed and saved vis the manager. 
NOTE: the quality checks of the data must be run prior to the analysis, as the ET analysis relies on all data being
alreay parsed and preprocessed prior to the beginning of the cross-subejct aggregation and statistics. As such, 
"ET_qc_manager.py" script needs to be completed before running this analysis module. 
Note that data for GLMMs of all 3 datatypes is prepared
and saved here to csv files, but the analyses itself is not included in the module.  

@author: RonyHirsch
"""

FIXATION = 'fixation'
CATEGORY = 'Category'
VIS = 'Visibility'
PUPIL = 'Pupil'
DICT_MAPPING = {data_saver.LOC: LOC, data_saver.VIS: VIS, data_saver.WORLD: WORLD_ID, data_saver.CATEGORY: CATEGORY, data_saver.ALL: "all"}
TARGETS = ["Stim", "Center"]
COUNT = 'count'
MEAN = 'mean'
STD = 'std'
PCNT_FIX = 'pcnt_fixation'
STIM_LOCS_NAMES = ["Top Right", "Top Left", "Bottom Left", "Bottom Right"]
STIM_LOCS_ANGLES = {"Top Right": 45, "Top Left": 135, "Bottom Left": 225, "Bottom Right": 315}
ET_DATA_DICT = "et_data_dict"
TRIAL_INFO = "trial_info"
BINOCULAR = "binocular"
MONOCULAR = "monocular"
EYE_TRANSLATION = {"L": MONOCULAR, "R": MONOCULAR, "LR": BINOCULAR}
TRIAL_NUMBER = "trialNumber"


def is_sub_code(st):
    if st.startswith(("SA", "SB", "SC", "SD", "SE", "SF", "SX", "SZ")):
        if st[2:].isdigit():
            return True
    return False


def check_sub_fix_validity(subs_path):
    sub_list = [s for s in os.listdir(subs_path) if os.path.isdir(os.path.join(subs_path, s))]
    valid_sub_list = list()
    # make sure we only return with valid subs
    for sub in sub_list:
        is_sub = is_sub_code(sub)
        if not is_sub:
            print(f"Folder {sub} is not a valid subject folder; skipping it")
            continue
        sub_fix_path = os.path.join(subs_path, sub, FIXATION)
        if not os.path.exists(sub_fix_path):
            print(f"Subject {sub} does not have fixation data; either no valid ET data or need to run QC code first")
            continue
        valid_sub_list.append(sub)
    return valid_sub_list


def analyze_fixation_densities(sub_names, relevant_subs_paths, subs_analysis_path):
    """
    The goal of this module is to produce plots of heatmaps of fixation for each:
    - duration (epoch/stim duration/pre stimulus period)
    - condition (for each world, for each visibility type, for each location of the target stimulus etc).
    :param subs_qc_path: path to the folder containing the subjects' individual data that will be used to calculate
    cross-subject density
    :param subs_analysis_path: path to which the data (plots and pickles) will be saved
    :return:
    """
    first_sub = True
    aggr_dict = dict()
    for i in range(len(relevant_subs_paths)):  # Aggregate across subjects
        sub_path = relevant_subs_paths[i]
        sub_name = sub_names[i]
        fl = open(os.path.join(sub_path, f"{sub_name}EyeTrackingData.pickle"), 'rb')
        sub_data = pickle.load(fl)
        fl.close()
        sub_eye = sub_data[PARAMS]["Eye"]  # this is the ANALYZED eye, see ET_param_manager
        del sub_data
        if sub_eye == "LR":
            print("Only monocular data is considered: 2022-09-12 consortium decision")
            continue
        monocular_eye = EYE_TRANSLATION[sub_eye]

        fl = open(os.path.join(sub_path, "fix_density.pickle"), 'rb')
        sub_fix_data = pickle.load(fl)
        fl.close()

        # aggr_dict INITIATION
        if first_sub:
            first_sub = False
            eye = MONOCULAR  # Only monocular data is considered: 2022-09-12 consortium decision: see ET_param_manager
            aggr_dict[eye] = dict()
            for time in [DataParser.STIM_DUR, DataParser.PRE_STIM_DUR, DataParser.TRIAL]:
                aggr_dict[eye][time] = dict()
                for phase in [DataParser.REPLAY_PHASE, DataParser.GAME_PHASE]:
                    aggr_dict[eye][time][phase] = dict()
                    for cond in [DataParser.VIS, STIM_TYPE, STIM_TYPE + "_" + DataParser.VIS, DataParser.STIM_LOC]:
                        aggr_dict[eye][time][phase][cond] = dict()
                        subconds = sub_fix_data[sub_eye][time][phase][cond].keys()
                        for subcond in subconds:
                            aggr_dict[eye][time][phase][cond][subcond] = dict()
                            aggr_dict[eye][time][phase][cond][subcond]["density"] = None
                            aggr_dict[eye][time][phase][cond][subcond]["count"] = 0

        # Go over the subject's densities in the different conditions
        for time in [DataParser.STIM_DUR, DataParser.PRE_STIM_DUR, DataParser.TRIAL]:
            for phase in [DataParser.REPLAY_PHASE, DataParser.GAME_PHASE]:
                for cond in [DataParser.VIS, STIM_TYPE, STIM_TYPE + "_" + DataParser.VIS, DataParser.STIM_LOC]:
                    for subcond in sub_fix_data[sub_eye][time][phase][cond].keys():
                        if not isinstance(sub_fix_data[sub_eye][time][phase][cond][subcond]["density"], type(None)):  # We add the WEIGHTED density [density * count]
                            # This means, that to give weigh densities by the number of samples they derive from, we multiply by the number of samples
                            # so that it's not "density" anymore (i.e., matrix not summing to 1), but rather the number of sampled data within that bin.
                            if isinstance(aggr_dict[monocular_eye][time][phase][cond][subcond]["density"], type(None)):  # If none, we cannot += to nothing, so we will put in the value
                                aggr_dict[monocular_eye][time][phase][cond][subcond]["density"] = sub_fix_data[sub_eye][time][phase][cond][subcond]["density"] * sub_fix_data[sub_eye][time][phase][cond][subcond]["count"]
                                aggr_dict[monocular_eye][time][phase][cond][subcond]["count"] = sub_fix_data[sub_eye][time][phase][cond][subcond]["count"]
                            else:
                                aggr_dict[monocular_eye][time][phase][cond][subcond]["density"] += sub_fix_data[sub_eye][time][phase][cond][subcond]["density"] * sub_fix_data[sub_eye][time][phase][cond][subcond]["count"]
                                aggr_dict[monocular_eye][time][phase][cond][subcond]["count"] += sub_fix_data[sub_eye][time][phase][cond][subcond]["count"]

    # Because we have yet to use division, we will add Target now to replay
    eye = MONOCULAR  # Only monocular data is considered: 2022-09-12 consortium decision: see ET_param_manager
    for time in [DataParser.STIM_DUR, DataParser.PRE_STIM_DUR, DataParser.TRIAL]:
        phase = DataParser.REPLAY_PHASE
        cond = "Target"

        print(f"eye {eye} time {time} phase {phase} cond {cond}")
        aggr_dict[eye][time][phase][cond] = dict()
        aggr_dict[eye][time][phase][cond]["Target"] = dict()
        aggr_dict[eye][time][phase][cond]["Target"]["density"] = aggr_dict[eye][time][phase][STIM_TYPE]["Face_target"]["density"] + aggr_dict[eye][time][phase][STIM_TYPE]["Object_target"]["density"]
        aggr_dict[eye][time][phase][cond]["Target"]["count"] = aggr_dict[eye][time][phase][STIM_TYPE]["Face_target"]["count"] + aggr_dict[eye][time][phase][STIM_TYPE]["Object_target"]["count"]

        aggr_dict[eye][time][phase][cond]["Non_target"] = dict()
        aggr_dict[eye][time][phase][cond]["Non_target"]["density"] = aggr_dict[eye][time][phase][STIM_TYPE]["Object_non_target"]["density"] + aggr_dict[eye][time][phase][STIM_TYPE]["Face_non_target"]["density"]
        aggr_dict[eye][time][phase][cond]["Non_target"]["count"] = aggr_dict[eye][time][phase][STIM_TYPE]["Object_non_target"]["count"] + aggr_dict[eye][time][phase][STIM_TYPE]["Face_non_target"]["count"]

    # Aggregation over, define some basic names that we'll use later transform the weighted density (# samples in bin) to real density, by dividing by the number of samples.
    for time in [DataParser.STIM_DUR, DataParser.PRE_STIM_DUR, DataParser.TRIAL]:
        for phase in [DataParser.REPLAY_PHASE, DataParser.GAME_PHASE]:
            for cond in [DataParser.VIS, STIM_TYPE, STIM_TYPE + "_" + DataParser.VIS, "Target", DataParser.STIM_LOC]:
                if phase == DataParser.GAME_PHASE and cond == "Target":
                    continue
                for subcond in aggr_dict[eye][time][phase][cond].keys():
                    if not isinstance(aggr_dict[eye][time][phase][cond][subcond]["density"], type(None)):
                        aggr_dict[eye][time][phase][cond][subcond]["density"] /= aggr_dict[eye][time][phase][cond][subcond]["count"]

    fl = open(os.path.join(subs_analysis_path, "fix_dens.pickle"), 'wb')
    pickle.dump(aggr_dict, fl)
    fl.close()
    return


def plot_fixation_densities(subs_analysis_path):
    """
    The goal of this module is to produce plots of heatmaps of fixation for each:
    - duration (epoch/stim duration/pre stimulus period)
    - condition (for each world, for each visibility type, for each location of the target stimulus etc).
    :param subs_analysis_path: path to which the data (plots and pickles) will be saved
    :return:
    """
    fl = open(os.path.join(subs_analysis_path, "fix_dens.pickle"), 'rb')
    aggr_dict = pickle.load(fl)
    fl.close()

    eye = MONOCULAR  # Only monocular data is considered: 2022-09-12 consortium decision: see ET_param_manager
    for time in [DataParser.STIM_DUR, DataParser.PRE_STIM_DUR, DataParser.TRIAL]:
        for phase in [DataParser.REPLAY_PHASE, DataParser.GAME_PHASE]:
            for cond in [DataParser.VIS, STIM_TYPE, "Target", DataParser.STIM_LOC]:
                if phase == DataParser.GAME_PHASE and cond == "Target":
                    continue
                cond_data = aggr_dict[eye][time][phase][cond]
                if len(cond_data.keys()) > 2:
                    cols = 2
                    rows = 2
                    if len(cond_data.keys()) > 4:
                        cols = 3
                else:
                    rows = 1
                    cols = len(cond_data.keys())
                plotter.heatmaps(heatmap_data=cond_data, x_label="Screen Width", y_label="Screen Height",
                                 title=f"Fixation Density During {time} Across Subs", with_pic="game_orange.png",
                                 save_path=subs_analysis_path, save_name=f"{eye}_{time}_{phase}_{cond}_density.png", n_rows=rows, n_cols=cols, figsize=(20, 10))
    return


def analyze_fixation_stabilities(sub_names, relevant_subs_paths, subs_analysis_path):
    """
    This module gets all the stats about fixation we'd want to report:
    mean fixation proportion (and std), for each trial type (time window * condition (and sub-conditions))
    and the distance (mean, std) between subjects' gaze and target (stimulus/fixation center) across types as well.

    :param subs_qc_path:
    :param subs_analysis_path:
    :return:
    """
    all_subs_df = pd.DataFrame()
    for i in range(len(relevant_subs_paths)):  # Aggregate across subjects
        """For each subject, take the fixation stats dataframe (from the QC stage), 
        then, for each of these stats: % fixation in the predefined area, mean distance of gaze from stimulus, mean
        distance of gaze from the center
        add a column which would be the numerator in a weighted average across all subjects: multiply the stat by the 
        number of samples the subject had
        """
        sub_path = relevant_subs_paths[i]
        sub_name = sub_names[i]
        fl = open(os.path.join(sub_path, f"{sub_name}EyeTrackingData.pickle"), 'rb')
        sub_data = pickle.load(fl)
        fl.close()
        sub_eye = sub_data[PARAMS]["Eye"]
        del sub_data
        if sub_eye == "LR":
            print("Only monocular data is considered: 2022-09-12 consortium decision")
            continue

        file_name = "fix_stats_samp_rate_"
        file = [f for f in os.listdir(sub_path) if f.startswith(file_name)][0]
        sub_fix_stats = pd.read_csv(os.path.join(sub_path, file))
        # In order to remove the eye from the equation, take only relevant eye rows
        sub_fix_stats = sub_fix_stats[sub_fix_stats[EYE] == sub_eye]
        for col in ["pcntIsInFixationArea", "meanStimDistDegs", "meanCenterDistDegs"]:
            sub_fix_stats[col+"_sum"] = sub_fix_stats[col] * sub_fix_stats["sampCount"]
        all_subs_df = pd.concat([all_subs_df, sub_fix_stats], axis=0)  # concatenate and add to an all-sub dataframe

    # replace nans and group by sum : so we have both all the numerators and the denominator
    all_subs_df.drop(columns=["Unnamed: 0", "stdStimDistDegs", "stdCenterDistDegs"], axis=1, inplace=True)
    all_subs_df[["timeWindow", "condition", "subCond", "IsReplay"]] = all_subs_df[["timeWindow", "condition", "subCond", "IsReplay"]].fillna("All")
    all_subs_sum = all_subs_df.groupby(["timeWindow", "condition", "subCond", "IsReplay"]).sum().reset_index()
    # Calculate the weighted average
    for col in ["pcntIsInFixationArea", "meanStimDistDegs", "meanCenterDistDegs"]:
        all_subs_sum[col] = all_subs_sum[col+"_sum"] / all_subs_sum["sampCount"]

    # Calculate the WEIGHTED STD : found no easy function for a weighted std which works with our data
    # we don't need the STD of the averages like np.std(weights=..) gives - we need the std of the subject mean with the weighted mean.
    # i.e., the weighting of the average is across SUBJECTS
    for time_win in all_subs_sum["timeWindow"].unique():
        for condition in all_subs_sum["condition"].unique():
            for sub_cond in (all_subs_sum[all_subs_sum["condition"] == condition])["subCond"].unique():
                for is_replay in all_subs_sum["IsReplay"].unique():
                    sub_view_sum = all_subs_sum[(all_subs_sum["timeWindow"] == time_win) & (all_subs_sum["condition"] == condition) &
                                            (all_subs_sum["subCond"] == sub_cond) & (all_subs_sum["IsReplay"] == is_replay)]
                    sub_view_sum.reset_index(inplace=True)
                    sub_view = all_subs_df[(all_subs_df["timeWindow"] == time_win) & (all_subs_df["condition"] == condition) &
                                           (all_subs_df["subCond"] == sub_cond) & (all_subs_df["IsReplay"] == is_replay)]

                    # If we have NaN here, it means that the subject didn't have any fixation samples in the relevant subcond (count will be 0)
                    sub_view.fillna(0, inplace=True)

                    for col in ["pcntIsInFixationArea", "meanStimDistDegs", "meanCenterDistDegs"]:
                        weighted_avg = list(sub_view_sum[col])
                        if np.array(sub_view["sampCount"]).sum() !=0 :
                            weighted_std = math.sqrt(np.average((np.array(sub_view[col]) - weighted_avg)**2, weights=np.array(sub_view["sampCount"])))
                            all_subs_sum.loc[(all_subs_sum["timeWindow"] == time_win) & (all_subs_sum["condition"] == condition) & (all_subs_sum["subCond"] == sub_cond) & (all_subs_sum["IsReplay"] == is_replay), col+"_std"] = weighted_std
                        else:
                            all_subs_sum.loc[(all_subs_sum["timeWindow"] == time_win) & (all_subs_sum["condition"] == condition) & (all_subs_sum["subCond"] == sub_cond) & (all_subs_sum["IsReplay"] == is_replay), col+"_std"] = 0
    all_subs_sum.drop(columns=["pcntIsInFixationArea_sum", "meanStimDistDegs_sum", "meanCenterDistDegs_sum"], inplace=True)
    all_subs_sum.to_csv(os.path.join(subs_analysis_path, "fix_stab_stats.csv"), index=False)
    return


def glmm_prep_fixations(sub_names, relevant_subs_paths, subs_analysis_path):
    """
    This module gets all the stats about fixation we'd want to report:
    mean fixation proportion (and std), for each trial type (time window * condition (and sub-conditions))
    and the distance (mean, std) between subjects' gaze and target (stimulus/fixation center) across types as well.

    :param subs_qc_path:
    :param subs_analysis_path:
    :return:
    """
    all_subs_df = pd.DataFrame()
    for i in range(len(relevant_subs_paths)):  # Aggregate across subjects
        sub_path = relevant_subs_paths[i]
        sub_name = sub_names[i]
        """For each subject, take the fixation stats dataframe (from the QC stage), 
        then, for each of these stats: % fixation in the predefined area, mean distance of gaze from stimulus, mean
        distance of gaze from the center
        add a column which would be the numerator in a weighted average across all subjects: multiply the stat by the 
        number of samples the subject had
        """
        fl = open(os.path.join(sub_path, f"{sub_name}EyeTrackingData.pickle"), 'rb')
        sub_data = pickle.load(fl)
        fl.close()
        sub_eye = sub_data[PARAMS]["Eye"]
        del sub_data
        if sub_eye == "LR":
            print("Only monocular data is considered: 2022-09-12 consortium decision")
            continue
        file_name = "fix_stats_samp_rate_"
        fix_dir = os.path.join(sub_path)
        file = [f for f in os.listdir(fix_dir) if f.startswith(file_name)][0]
        sub_fix_stats = pd.read_csv(os.path.join(fix_dir, file))
        sub_fix_stats["sub"] = sub_name
        sub_fix_stats = sub_fix_stats[sub_fix_stats[EYE] == sub_eye]
        all_subs_df = pd.concat([all_subs_df, sub_fix_stats], axis=0)  # concatenate and add to an all-sub dataframe

    all_subs_df.to_csv(os.path.join(subs_analysis_path, "fix_glmm.csv"), index=False)
    return


def bin_fix_dist(sub_names, relevant_subs_paths, subs_analysis_path, num_of_bins=100, bin_min=-10.0, bin_max=10.0):
    first_sub = True
    fix_dist = dict()
    fix_position_frequency = dict()
    for i in range(len(relevant_subs_paths)):  # Aggregate across subjects
        sub_path = relevant_subs_paths[i]
        sub_name = sub_names[i]
        fl = open(os.path.join(sub_path, f"{sub_name}EyeTrackingData.pickle"), 'rb')
        sub_data = pickle.load(fl)
        fl.close()
        sub_eye = sub_data[PARAMS]["Eye"]
        if sub_eye == "LR":
            print("Only monocular data is considered: 2022-09-12 consortium decision")
            continue

        sub_data[TRIAL_INFO] = prepare_trial_info(sub_data[TRIAL_INFO])

        if first_sub:
            first_sub = False
            # Go over the subject's densities in the different conditions
            for time in [DataParser.STIM_DUR, DataParser.PRE_STIM_DUR, DataParser.TRIAL]:
                fix_dist[time] = dict()
                fix_position_frequency[time] = dict()
                for phase in [DataParser.REPLAY_PHASE, DataParser.GAME_PHASE]:
                    fix_dist[time][phase] = dict()
                    fix_position_frequency[time][phase] = dict()
                    for cond in [DataParser.VIS, STIM_TYPE, STIM_TYPE + "_" + DataParser.VIS]:
                        fix_dist[time][phase][cond] = dict()
                        fix_position_frequency[time][phase][cond] = dict()
                        if cond == DataParser.VIS:
                            rel_subconds = sub_data[TRIAL_INFO][cond].unique()
                        elif cond == STIM_TYPE:
                            if phase == DataParser.REPLAY_PHASE:
                                rel_subconds = ["Face_target", "Face_non_target", "Object_target", "Object_non_target", "Blank"]
                            else:
                                rel_subconds = ["Face", "Object", "Blank"]
                        else:
                            if phase == DataParser.REPLAY_PHASE:
                                cat_subconds = ["Face_target", "Face_non_target", "Object_target", "Object_non_target", "Blank"]
                            else:
                                cat_subconds = ["Face", "Object", "Blank"]
                            rel_subconds = [f"{x[0]}_{x[1]}" for x in itertools.product(sub_data[TRIAL_INFO][DataParser.VIS].unique(), cat_subconds)]

                        for subcond in rel_subconds:
                            fix_dist[time][phase][cond][subcond] = list()
                            fix_position_frequency[time][phase][cond][subcond] = dict()

        for phase in [DataParser.REPLAY_PHASE, DataParser.GAME_PHASE]:
            for cond in [DataParser.VIS, STIM_TYPE, STIM_TYPE + "_" + DataParser.VIS]:
                for subcond in fix_dist["Trial"][phase][cond]:
                    trial_info = sub_data[TRIAL_INFO]
                    if cond == STIM_TYPE + "_" + DataParser.VIS:
                        seperator = subcond.index('_')
                        trial_info = trial_info[trial_info[WORLD_ID].isin(CONDITIONS[WORLD_ID][phase]) & (trial_info[DataParser.VIS] == subcond[:seperator])
                                                & (trial_info[STIM_TYPE] == subcond[seperator+1:])]
                    else:
                        trial_info = trial_info[trial_info[WORLD_ID].isin(CONDITIONS[WORLD_ID][phase]) & (trial_info[cond] == subcond)]

                    trial_info = trial_info[trial_info[DataParser.IS_LEVEL_ELIMINATED] != 1]  # Filter out eliminated levels
                    rel_trials = list(trial_info[TRIAL_NUMBER])
                    cols = [f'{sub_eye}{c}CenterDistDegsSigned' for c in ['X', 'Y']]
                    cols_no_eye = [f"{c}CenterDistDegsSigned" for c in ['X', 'Y']]
                    for time in [DataParser.STIM_DUR, DataParser.PRE_STIM_DUR, DataParser.TRIAL]:
                        if f"{sub_eye}{DataParser.EYELINK}Fix" not in sub_data[ET_DATA_DICT][DF_SAMPLES]:
                            continue
                        rel_samples = sub_data[ET_DATA_DICT][DF_SAMPLES][(sub_data[ET_DATA_DICT][DF_SAMPLES][time].isin(rel_trials))]
                        if not rel_samples.empty:
                            rel_samples = filter_samples(rel_samples, col_relevant=f"{sub_eye}{DataParser.EYELINK}Fix",
                                                         cols_nonrelevant=[f"{sub_eye}{DataParser.HERSHMAN}",
                                                                           f"{sub_eye}{DataParser.EYELINK}Blink",
                                                                           f"{sub_eye}{DataParser.EYELINK}Sacc"])

                        data = rel_samples[cols]
                        data.rename(columns={f'{sub_eye}{c}CenterDistDegsSigned': f"{c}CenterDistDegsSigned" for c in ['X', 'Y']}, inplace=True)
                        freq_data = pd.DataFrame(columns=['Bin Value'] + cols_no_eye)
                        for col in cols_no_eye:
                            cdat = data[col].to_numpy()
                            cdat = cdat[~np.isnan(cdat)]
                            chist = np.histogram(cdat, bins=num_of_bins, range=(bin_min, bin_max))  # chist[0] = frequency, chist[1] = bin value
                            freq = chist[0]/sum(chist[0])  # now instead of frequency (number) we have the proportion (fraction): 100 values
                            freq_data['Bin Value'] = chist[1]  # this has 101 values (as they define bin limits)
                            freq_data[col] = np.concatenate((np.array([np.nan]), freq), axis=0)

                        fix_dist[time][phase][cond][subcond].append(freq_data)

    # STEP 2: aggregate ACROSS ALL subs: here, this is not a weighted average (each subject has the same weight) as the
    # data we're averaging on in the first place is PROPORTIONS. (fixation proportion).
    for phase in [DataParser.REPLAY_PHASE, DataParser.GAME_PHASE]:
        for cond in [DataParser.VIS, STIM_TYPE, STIM_TYPE + "_" + DataParser.VIS]:
            for subcond in fix_dist["Trial"][phase][cond]:
                for time in [DataParser.STIM_DUR, DataParser.PRE_STIM_DUR, DataParser.TRIAL]:
                    print(f"{phase} {cond} {subcond} {time}")
                    concatenated = pd.concat((df for df in fix_dist[time][phase][cond][subcond]))
                    # groupby the INDEX as the index is the BIN, so for each bin we avg across subs
                    avg = concatenated.groupby(concatenated.index).mean()
                    std = concatenated.groupby(concatenated.index).std()
                    fix_position_frequency[time][phase][cond][subcond][MEAN] = avg
                    fix_position_frequency[time][phase][cond][subcond][STD] = std

    fl = open(os.path.join(subs_analysis_path, "fix_dist_from_center.pickle"), 'wb')
    pickle.dump(fix_position_frequency, fl)
    fl.close()
    return


def get_saccades_data(sub_names, relevant_subs_paths, sacc_analysis_path):
    """
    Gets all the saccade data and stats to an analyzable structure. Creates 2 dataframes saved as csv:
    - sacc_data_stats_within_subs: dict with the following nesting: {eye: {dur: {cond: sub_cond / all..}}} where the
    data is a list. In this list, each element is a single subject's dataframe with statistics about their individual
    saccade data.
    - sacc_data_across_subs: dict with the following nesting: {eye: {dur: {cond: sub_cond / all..}}} where the data
    is a single dataframe which includes ALL subjects' saccades, 1 row per saccade. Each row contains the subject
    information as well.
    In addition, this method also saves CSV files with summary statistics of saccades info across subs.
    These csv files are saved within their respective duration+condition folders, and there's 1 csv per eye and per
    sub-condition.
    :param subs_qc_path: path to the QC sub folders where the relevant pickle files are saved
    :param subs_analysis_path: path where to save the resulting pickle files
    :return: sacc_data_across_subs
    """
    first_sub = True
    sacc_dist = dict()  # dict where the lists of subjects' averaged dfs are contained
    """
    The following two dataframes contain ROW PER CONDITION (e.g., all the replay trials where face was the target and a fact appeared), 
    and the columns are statistics of the ***EYELINK*** SACCADES & ***EK*** MICROSACCADES across all trials that fit this condition. 
    In within_subs, this is across trials of the same subject, and subjects are not aggregated. 
    In between_subs, this is across subjects - all subjects are aggregated together like "one massive subject". 
    """
    all_saccades = pd.DataFrame()
    between_subs = pd.DataFrame()
    cols = ["duration", "ampDeg", DataParser.SACC_DIRECTION_RAD, "sacc_direction_deg", "sacc_dist_dva"]
    within_subs = pd.DataFrame(columns=["timeWindow", "phase", "cond", "subcond", "eye", "durationAvg", "ampDegAvg", "sacc_direction_radAvg", "sacc_direction_degAvg", "sacc_dist_dvaAvg", "numSaccs", "numTrials", "Subject"])

    for i in range(len(relevant_subs_paths)):  # Aggregate across subjects
        sub_path = relevant_subs_paths[i]
        sub_name = sub_names[i]
        fl = open(os.path.join(sub_path, f"{sub_name}EyeTrackingData.pickle"), 'rb')
        sub_data = pickle.load(fl)
        fl.close()
        sub_eye = sub_data[PARAMS]["Eye"]
        if sub_eye == "LR":
            print("Only monocular data is considered: 2022-09-12 consortium decision")
            continue

        sub_data[TRIAL_INFO] = prepare_trial_info(sub_data[TRIAL_INFO])
        # First sub handling
        if first_sub:
            first_sub = False
            # Go over the subject's densities in the different conditions
            for timewindow in TIME_WINDOWS:
                sacc_dist[timewindow] = dict()
                for phase in [DataParser.REPLAY_PHASE, DataParser.GAME_PHASE]:
                    sacc_dist[timewindow][phase] = dict()
                    for cond in [DataParser.VIS, STIM_TYPE, STIM_TYPE + "_" + DataParser.VIS, "TARGET", DataParser.STIM_LOC]:
                        if cond == DataParser.VIS:
                            rel_subconds = sub_data[TRIAL_INFO][cond].unique()
                        elif cond == DataParser.STIM_LOC:
                            rel_subconds = sub_data[TRIAL_INFO][cond].unique()
                        elif cond == STIM_TYPE:
                            if phase == DataParser.REPLAY_PHASE:
                                rel_subconds = ["Face_target", "Face_non_target", "Object_target", "Object_non_target", "Blank"]
                            else:
                                rel_subconds = ["Face", "Object", "Blank"]
                        elif cond == "TARGET":
                            if phase == DataParser.REPLAY_PHASE:
                                rel_subconds = ["Target", "Non_target"]
                            else:
                                continue
                        else:
                            if phase == DataParser.REPLAY_PHASE:
                                cat_subconds = ["Face_target", "Face_non_target", "Object_target", "Object_non_target", "Blank"]
                            else:
                                cat_subconds = ["Face", "Object", "Blank"]
                            rel_subconds = [f"{x[0]}_{x[1]}" for x in itertools.product(sub_data[TRIAL_INFO][DataParser.VIS].unique(), cat_subconds)]
                        sacc_dist[timewindow][phase][cond] = dict()
                        for subcond in rel_subconds:
                            sacc_dist[timewindow][phase][cond][subcond] = pd.DataFrame()
        # Done with first sub handling
        # AT THE SUBJECT LEVEL
        # this is for a single subject at a time
        for phase in [DataParser.REPLAY_PHASE, DataParser.GAME_PHASE]:  # for each condition of the VG
            for timewindow in TIME_WINDOWS:
                for cond in [DataParser.VIS, STIM_TYPE, STIM_TYPE + "_" + DataParser.VIS, "TARGET", DataParser.STIM_LOC]:
                    if cond not in sacc_dist[timewindow][phase]:
                        continue
                    for subcond in sacc_dist[timewindow][phase][cond]:  # so all subjects will have all conditions
                        trial_info = sub_data[TRIAL_INFO]
                        if cond == STIM_TYPE + "_" + DataParser.VIS:
                            seperator = subcond.index('_')
                            # select relevant trials
                            trial_info = trial_info[trial_info[WORLD_ID].isin(CONDITIONS[WORLD_ID][phase]) & (trial_info[DataParser.VIS] == subcond[:seperator])
                                                    & (trial_info[STIM_TYPE] == subcond[seperator+1:])]
                        elif cond == "TARGET":
                            if subcond == "Target":
                                trial_info = trial_info[trial_info[WORLD_ID].isin(CONDITIONS[WORLD_ID][phase]) & (trial_info[STIM_TYPE].isin(["Face_target", "Object_target"]))]
                            else:
                                trial_info = trial_info[trial_info[WORLD_ID].isin(CONDITIONS[WORLD_ID][phase]) & (trial_info[STIM_TYPE].isin(["Face_non_target", "Object_non_target"]))]
                        else:
                            trial_info = trial_info[trial_info[WORLD_ID].isin(CONDITIONS[WORLD_ID][phase]) & (trial_info[cond] == subcond)]
                        trial_info = trial_info[trial_info[DataParser.IS_LEVEL_ELIMINATED] != 1]  # Filter out eliminated levels
                        relevant_trials = list(trial_info[TRIAL_NUMBER])
                        relevant_trial_information = trial_info[[TRIAL_NUMBER, "EpochWindowStart", "EpochWindowEnd"]]  # the epochs are added to the saccade table of the subject in the current sub cond
                        relevant_trial_information.set_index(TRIAL_NUMBER, inplace=True)  # This is needed because join works on the INDEX of the first df
                        """
                        Next, filter samples to include only samples of the relevant trial epochs, relevant eye, 
                        and only saccades. We filter out 'saccades' based on Eyelink blinks, as those blinks might
                        have not been real (=not EK blinks), but they could stem from missing data. In any case, 
                        we know this is not a real saccade if Eyelink identified it as blink, so we reomove those.  
                        """
                        relevant_samples = sub_data[ET_DATA_DICT][DataParser.DF_SACC_EK][(sub_data[ET_DATA_DICT][DataParser.DF_SACC_EK][timewindow].isin(relevant_trials)) &
                                                                             (sub_data[ET_DATA_DICT][DataParser.DF_SACC_EK][DataParser.EYE] == f"{sub_eye}")]
                        relevant_samples = relevant_samples[relevant_samples[f"is_{DataParser.EYELINK}Blink"].isna()]
                        relevant_samples = relevant_samples[relevant_samples[f"is_{DataParser.HERSHMAN}Blink"].isna()]
                        # Obtain epoch start and end for easier calculation in the calculations on other functions over time
                        relevant_samples = relevant_samples.join(relevant_trial_information, on=timewindow)

                        num_of_trials = len(relevant_samples["Trial"].unique())
                        # average (within subject, across trials)
                        data = relevant_samples[cols]
                        sub_avg_data = data.mean()
                        sub_avg_data.rename({x: x + "Avg" for x in cols}, inplace=True)
                        sub_count_data = data.count()[DataParser.SACC_DIRECTION_RAD]

                        full_df = pd.DataFrame()
                        full_df = full_df.append(sub_avg_data.to_frame().T)

                        full_df["timeWindow"] = timewindow
                        full_df["phase"] = phase
                        full_df["cond"] = cond
                        full_df["subcond"] = subcond
                        full_df["numSaccs"] = sub_count_data
                        full_df["numTrials"] = num_of_trials
                        full_df["Subject"] = sub_name
                        full_df[EYE] = sub_eye

                        # This is to hold all saccades in a single DF
                        relevant_samples["timeWindow"] = timewindow
                        relevant_samples["phase"] = phase
                        relevant_samples["cond"] = cond
                        relevant_samples["subcond"] = subcond
                        relevant_samples["Subject"] = sub_name
                        relevant_samples[EYE] = sub_eye
                        all_saccades = pd.concat([all_saccades, relevant_samples])

                        within_subs = within_subs.append(full_df)

    # Now, calculate ACROSS SUBS
    for phase in [DataParser.REPLAY_PHASE, DataParser.GAME_PHASE]:  # for each condition of the VG
        for timewindow in TIME_WINDOWS:
            for cond in [DataParser.VIS, STIM_TYPE, STIM_TYPE + "_" + DataParser.VIS, "TARGET", DataParser.STIM_LOC]:
                if cond not in sacc_dist[timewindow][phase]:
                    continue
                for subcond in sacc_dist[timewindow][phase][cond]:  # so all subjects will have all conditions
                    relevevant_saccs = within_subs[(within_subs["phase"] == phase) & (within_subs["timeWindow"] == timewindow)
                                                   & (within_subs["cond"] == cond) & (within_subs["subcond"] == subcond)]
                    for col in cols:
                        relevevant_saccs[col+"_weighted"] = relevevant_saccs[col+"Avg"] * relevevant_saccs["numSaccs"]

                    averaged = relevevant_saccs.mean()
                    for col in cols:
                        averaged[col] = averaged[col+"_weighted"]/relevevant_saccs["numSaccs"].sum()
                        averaged.drop(col+"Avg", inplace=True)
                        averaged.drop(col+"_weighted", inplace=True)
                    std = relevevant_saccs.std()
                    averaged_df = pd.DataFrame()
                    for col in cols:
                        averaged_df[f"{col}Avg"] = [averaged[col]]
                        averaged_df[col+"Std"] = [std[col+"Avg"]]
                    averaged_df["timeWindow"] = [timewindow]
                    averaged_df["phase"] = [phase]
                    averaged_df["cond"] = [cond]
                    averaged_df["subcond"] = [subcond]

                    between_subs = between_subs.append(averaged_df)

    within_subs.to_csv(os.path.join(sacc_analysis_path, "sacc_stats_within_subs.csv"), index=False)
    between_subs.to_csv(os.path.join(sacc_analysis_path, "sacc_stats_between_subs.csv"), index=False)
    return all_saccades


def blink_stats_in_trial(sub_names, relevant_subs, subs_analysis_path):
    save_path = os.path.join(subs_analysis_path, data_saver.BLINK_NAME)

    # This is the names of the relevant columns in the trial info dataframes
    timings_dict = {DataParser.TRIAL: ["EpochWindowStart", "EpochWindowEnd"], DataParser.PRE_STIM_DUR: ["PreStimWindowStart", "StimOnset"],
                    DataParser.STIM_DUR: ["StimOnset", "StimDurationWindowEnd"]}
    # This is only the durations
    window_durs = {DataParser.TRIAL: ET_param_manager.EPOCH_END + ET_param_manager.EPOCH_START, DataParser.PRE_STIM_DUR: ET_param_manager.PRE_STIM_DUR,
                   DataParser.STIM_DUR: ET_param_manager.STIM_DUR}
    remove_cols_dict = {DataParser.TRIAL: ["PreStimWindowStart", "StimDurationWindowEnd"], DataParser.PRE_STIM_DUR: ["EpochWindowStart", "EpochWindowEnd", "StimDurationWindowEnd"],
                        DataParser.STIM_DUR: ["EpochWindowStart", "EpochWindowEnd", "PreStimWindowStart"]}
    timings = [DataParser.STIM_DUR, DataParser.PRE_STIM_DUR, DataParser.TRIAL]
    dict_dfs = {x: pd.DataFrame() for x in timings}
    stats_df = pd.DataFrame()
    for i in range(len(relevant_subs)):  # Iterate over subjects
        sub_path = relevant_subs[i]
        sub_name = sub_names[i]
        fl = open(os.path.join(sub_path, f"{sub_name}EyeTrackingData.pickle"), 'rb')  # distance data is in this pickle
        sub_data = pickle.load(fl)
        fl.close()

        sub_eye = sub_data[PARAMS]["Eye"]
        if sub_eye == "LR":
            print("Only monocular data is considered: 2022-09-12 consortium decision")
            continue

        sub_data[TRIAL_INFO] = prepare_trial_info(sub_data[TRIAL_INFO])
        sub_data[TRIAL_INFO].set_index(TRIAL_NUMBER, inplace=True)  # This is needed because join works on the INDEX of the first df

        for time in timings:
            blinks_in_time = sub_data[ET_DATA_DICT][DataParser.DF_BLINK][sub_data[ET_DATA_DICT][DataParser.DF_BLINK][time] != -1]
            blinks_in_time = blinks_in_time[blinks_in_time[DataParser.IS_LEVEL_ELIMINATED] == 0]
            blinks_in_time = blinks_in_time[(blinks_in_time[EYE.lower()] == sub_eye) & (blinks_in_time["Hershman"] == True)]
            trial_info = sub_data[TRIAL_INFO]
            blinks_in_time = blinks_in_time.join(trial_info, on=time, lsuffix="_blinks")
            blinks_in_time["onset"] = blinks_in_time["tStart"] - blinks_in_time[timings_dict[time][0]]
            blinks_in_time["offset"] = blinks_in_time["onset"] + blinks_in_time["duration"]
            blinks_in_time["overlap_start"] = blinks_in_time["onset"].apply(lambda x: max(x, 0))
            blinks_in_time["overlap_end"] = blinks_in_time["offset"].apply(lambda x: min(x, window_durs[time]))
            blinks_in_time["overlap"] = blinks_in_time["overlap_end"] - blinks_in_time["overlap_start"]
            blinks_in_time.loc[blinks_in_time["overlap"] < 0, "overlap"] = 0
            # Remove timing related columns which are not related for this time window
            irrelevant_timings = [x for x in timings if x != time]
            irrelevant_timings.extend(remove_cols_dict[time])
            blinks_in_time.drop(columns=irrelevant_timings, inplace=True)
            blinks_in_time["subject"] = sub_name
            dict_dfs[time] = pd.concat([dict_dfs[time], blinks_in_time])

        # This part is intended for the statsDF
        trial_info = sub_data[TRIAL_INFO]

        trial_info_eye = trial_info.copy()
        for time in timings:
            blinks_in_time = sub_data[ET_DATA_DICT][DataParser.DF_BLINK][sub_data[ET_DATA_DICT][DataParser.DF_BLINK][time] != -1]
            blinks_in_time = blinks_in_time[blinks_in_time[DataParser.IS_LEVEL_ELIMINATED] == 0]
            blinks_in_time["counter"] = 1
            blinks_in_eye = blinks_in_time[(blinks_in_time[EYE.lower()] == sub_eye) & (blinks_in_time["Hershman"] == True)]

            blinks_in_time_eye = blinks_in_eye.join(trial_info_eye, on=time, how="right", lsuffix="_blinks")
            blinks_in_time_eye["onset"] = blinks_in_time_eye["tStart"] - blinks_in_time_eye[timings_dict[time][0]]
            blinks_in_time_eye["offset"] = blinks_in_time_eye["onset"] + blinks_in_time_eye["duration"]
            blinks_in_time_eye["overlap_start"] = blinks_in_time_eye["onset"].apply(lambda x: max(x, 0))
            blinks_in_time_eye["overlap_end"] = blinks_in_time_eye["offset"].apply(lambda x: min(x, window_durs[time]))
            blinks_in_time_eye["overlap"] = blinks_in_time_eye["overlap_end"] - blinks_in_time_eye["overlap_start"]
            blinks_in_time_eye = blinks_in_time_eye.groupby(by=[time]).sum().reset_index()
            blinks_in_time_eye.set_index(time, inplace=True)

            trial_info_eye[f"Hershman_num_of_blinks_{time}"] = blinks_in_time_eye["counter"]
            trial_info_eye[f"Hershman_total_overlap_{time}"] = blinks_in_time_eye["overlap"]
            # Release unused dataframes
            del blinks_in_time_eye
            del blinks_in_time

        trial_info_eye["eye"] = sub_eye
        trial_info_eye["subject"] = sub_name
        trial_info_eye["trackedEye"] = sub_data[PARAMS]["Eye"]
        trial_info_eye = trial_info_eye.rename(columns={'index': 'trial'})

        stats_df = pd.concat([stats_df, trial_info_eye])

    for key in dict_dfs.keys():
        save_name = f"blinks_per_blink_{key}.csv"
        dict_dfs[key].to_csv(os.path.join(save_path, save_name), index=False)

    save_name = f"blinks_per_trial.csv"
    stats_df.to_csv(os.path.join(save_path, save_name), index=False)

    return


def blink_stabilities_in_time(sub_names, relevant_subs, subs_analysis_path):
    save_path = os.path.join(subs_analysis_path, data_saver.BLINK_NAME)
    blink_probs = list()

    for i in range(len(relevant_subs)):  # Iterate over subjects
        sub_path = relevant_subs[i]
        sub_name = sub_names[i]
        fl = open(os.path.join(sub_path, f"{sub_name}EyeTrackingData.pickle"), 'rb')  # distance data is in this pickle
        sub_data = pickle.load(fl)
        fl.close()

        sub_eye = sub_data[PARAMS]["Eye"]
        if sub_eye == "LR":
            print("Only monocular data is considered: 2022-09-12 consortium decision")
            continue

        sub_data[TRIAL_INFO] = prepare_trial_info(sub_data[TRIAL_INFO])
        # AT THE SUBJECT LEVEL: calculate per-sub averages of gaze distance over time
        # this is for a single subject at a time
        for phase in [DataParser.REPLAY_PHASE, DataParser.GAME_PHASE]:  # for each condition of the VG
            for cond in [DataParser.VIS, STIM_TYPE]:
                for subcond in get_relevant_subcond(sub_data, cond, phase):  # so all subjects will have all conditions
                    trial_info = sub_data[TRIAL_INFO]
                    if phase == DataParser.REPLAY_PHASE and subcond == "Target":
                        trial_info = trial_info[trial_info[WORLD_ID].isin(CONDITIONS[WORLD_ID][phase]) & (trial_info[cond].isin(["Face_target", "Object_target"]))]
                        cond_name = "TARGET"
                    elif phase == DataParser.REPLAY_PHASE and subcond == "Non_target":
                        trial_info = trial_info[trial_info[WORLD_ID].isin(CONDITIONS[WORLD_ID][phase]) & (trial_info[cond].isin(["Face_non_target", "Object_non_target"]))]
                        cond_name = "TARGET"
                    else:
                        cond_name = cond
                        trial_info = trial_info[trial_info[WORLD_ID].isin(CONDITIONS[WORLD_ID][phase]) & (trial_info[cond] == subcond)]
                    relevant_trials = list(trial_info[TRIAL_NUMBER])
                    cols = [DataParser.T_SAMPLE, f"{sub_eye}Hershman"]
                    if f"{sub_eye}Hershman" not in sub_data[ET_DATA_DICT][DF_SAMPLES]:
                        continue
                    # filter samples to include only samples of the relevant trial epochs
                    relevant_samples = sub_data[ET_DATA_DICT][DF_SAMPLES][(sub_data[ET_DATA_DICT][DF_SAMPLES][DataParser.TRIAL].isin(relevant_trials))]
                    relevant_samples = relevant_samples[relevant_samples[DataParser.IS_LEVEL_ELIMINATED] != 1]

                    for trial_num in relevant_trials:  # for each trial (average per sub is across trials)
                        # align timings for averaging - bring all trials' times to normalized around 0
                        # where 0 = STIM ONSET
                        stim_onset = trial_info.loc[trial_info[TRIAL_NUMBER] == trial_num, ONSET]
                        relevant_samples.loc[relevant_samples[DataParser.TRIAL] == trial_num, DataParser.T_SAMPLE] -= int(stim_onset)

                    # average (within subject, across trials)
                    data = relevant_samples[cols]
                    data[f'{sub_eye}Hershman'].fillna(0, inplace=True)

                    # Average the fixations in regard to the relevant timestamp in epoch
                    sub_avg_data = data.groupby(['tSample']).mean()
                    sub_avg_data.rename(columns={f"{sub_eye}Hershman": "blinkAvg"}, inplace=True)
                    # Count number of entries in each such average
                    sub_count_data = data.groupby(['tSample']).count()
                    sub_count_data.rename(columns={f"{sub_eye}Hershman": f"trialCnt"}, inplace=True)

                    full_df = pd.DataFrame(index=range(-1000, 1001), columns=range(1))  # This DF has all of the epoch times (from -1000 to 1000)
                    full_df = full_df.join(sub_avg_data)
                    full_df = full_df.join(sub_count_data)
                    full_df.drop(columns=[0], inplace=True)
                    full_df[[f"trialCnt"]] = full_df[[f"trialCnt"]].fillna(value=0)
                    full_df.reset_index(inplace=True)
                    full_df.rename(columns={"index": "tSample"}, inplace=True)
                    full_df["Subject"] = sub_name
                    full_df["eye"] = sub_eye
                    full_df["phase"] = phase
                    full_df["cond"] = cond_name
                    full_df["subcond"] = subcond
                    blink_probs.append(full_df)

    all_blink_probs = pd.concat(blink_probs)

    # has_data is 0 in the case where a subject did not have any data in this specific sample.
    all_blink_probs["has_data"] = 0
    all_blink_probs.loc[all_blink_probs[f"trialCnt"] != 0, "has_data"] = 1

    # Now we calculate the avg of all of the relevant precentages. Please note, this is NOT WEIGHTED AVERAGE.
    # This is in order to account for between subject variability

    df_means = all_blink_probs.groupby(['tSample', 'phase', 'cond', 'subcond', 'has_data']).mean()  # the MEAN ACROSS SUBJECTS where what is averaged here is subjects' means
    df_sum = all_blink_probs.groupby(['tSample', 'phase', 'cond', 'subcond']).sum()

    df_means.reset_index(inplace=True)
    df_means["SubjectCount"] = df_sum["has_data"].values  # Amount of subjects who have this trial
    df_means = df_means[df_means["has_data"] == 1]
    df_means.drop(columns=[f"trialCnt", "has_data"], inplace=True)

    # Only df_means is used from now on
    del all_blink_probs
    del df_sum

    blink_mean_dict = dict() # dict where the subjects' averaged dfs are contained
    for phase in [DataParser.REPLAY_PHASE, DataParser.GAME_PHASE]:  # for each condition of the VG
        blink_mean_dict[phase] = dict()
        for cond in [DataParser.VIS, STIM_TYPE, "TARGET"]:
            if phase == DataParser.GAME_PHASE and cond == "TARGET":
                continue
            blink_mean_dict[phase][cond] = df_means[(df_means["phase"] == phase) & (df_means["cond"] == cond)]
            save_name = f"{phase}_{cond}_blinks"
            blink_mean_dict[phase][cond].to_csv(os.path.join(save_path, f"{save_name}.csv"), index=False)
            print(f"{phase} {cond}")
            plotter.err_line_plot_multidf(cond_df=blink_mean_dict[phase][cond],
                                          x_col="tSample", y_col="blinkAvg", subcond_col="subcond", sd_col=None,
                                          subcond_color_dict=None, x_label='Time (ms)',
                                          y_label='Mean number of blinks',
                                          title=f"Mean Number of Blinks Across Subs",
                                          save_path=save_path, save_name=save_name,
                                          vertical_lines={"Stimulus On": 0,
                                                          "Stimulus Off": ET_param_manager.STIM_DUR},
                                          num_of_x_ticks=10, y_min=None,  y_max=None)

    # We inserted all of df_means into fix_mean_dict, we can clean it
    del df_means
    return


def blink_analysis(sub_names, relevant_subs, subs_analysis_path):
    """
    Note that blinks are any missing data in Eyelink; BUT, real blinks are the ones "within" saccade events (which are
    not realy saccades but blinks). Note that the data was prepared accordingly so that we could analyze
    saccades/fixation w/o any blink/missing samples (ignoring all "blinks"), as well as acommodate real-blink analysis
    as they are marked in saccade events: in each subject QC pickle file, within 'et_data_per_trial'['dfSacc'] we have
    the "is_blink" column for each saccade to filter out real from fake saccades.

    For blink analysis purposes, we have the "Hershman" columns, which identify blinks based on pupillometry method
    (i.e, based ONLY on pupil size, see QualityChecker).

    :param subs_qc_path:
    :return:
    """
    print("Calculate blink analysis")
    p = Process(target=blink_stats_in_trial, args=(sub_names, relevant_subs, subs_analysis_path))
    p.start()
    p.join()
    if p.exitcode > 0:  # if the process exited with an error
        exit(1)  # exit with an error as well

    print("Calculate blink stability in time")
    p = Process(target=blink_stabilities_in_time, args=(sub_names, relevant_subs, subs_analysis_path))
    p.start()
    p.join()
    if p.exitcode > 0:  # if the process exited with an error
        exit(1)  # exit with an error as well
    return


def plot_blink_overlap(subs_analysis_path):
    save_path = os.path.join(subs_analysis_path, data_saver.BLINK_NAME)
    save_name = f"blinks_per_trial.csv"
    blink_stats = pd.read_csv(os.path.join(save_path, save_name))
    plotter.hist_plot(data=blink_stats, x_col="Hershman_num_of_blinks_StimDuration",
                      title="Duration Blinks Overlapping with Stimulus Presentation", x_label="Overlap Duration",
                      y_label="Number of Trials", save_path=save_path, save_name="blink_stim_overlap")
    return


def analyze_blinks(sub_names, relevant_subs, subs_analysis_path):

    print(f"Analyzing blinks")
    p = Process(target=blink_analysis, args=(sub_names, relevant_subs, subs_analysis_path))
    p.start()
    p.join()
    if p.exitcode > 0:  # if the process exited with an error
        exit(1)  # exit with an error as well

    print(f"printing blinks")
    p = Process(target=plot_blink_overlap, args=(subs_analysis_path,))  # the comma here is deliberate, it makes args a tuple
    p.start()
    p.join()
    if p.exitcode > 0:  # if the process exited with an error
        exit(1)  # exit with an error as well

    return


def circular_sacc_analysis(all_saccades, sacc_analysis_path):
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
    summary = list()
    stim_loc_test_names = [[f'vtest_pval_{l}', f'vtest_V_{l}', f'vtest_Mtest_H0_reject_{l}', f'vtest_Mtest_pop_mean_{l}', f'vtest_Mtest_CI_{l}'] for l in STIM_LOCS_NAMES]
    stim_loc_test_names = [item for sublist in stim_loc_test_names for item in sublist]
    summary_cols = ['time window', 'phase', 'cond', 'subcond', 'samples', 'circ_mean', 'circ_std', 'rayleightest_test_pval'] + stim_loc_test_names

    for timewindow in TIME_WINDOWS:
        for phase in [DataParser.REPLAY_PHASE, DataParser.GAME_PHASE]:  # for each condition of the VG
            for cond in [DataParser.VIS, STIM_TYPE, STIM_TYPE + "_" + DataParser.VIS]:
                # We have only real saccades in this dataframe, so no need to check for "is_blink"
                relevant_saccs = all_saccades[(all_saccades["timeWindow"] == timewindow) & (all_saccades["phase"] == phase) & (all_saccades["cond"] == cond)]
                for subcond in relevant_saccs["subcond"].unique():
                    relevant_saccs_subcond = relevant_saccs[relevant_saccs["subcond"] == subcond]
                    sacc_radians = np.array(relevant_saccs_subcond[DataParser.SACC_DIRECTION_RAD])
                    circ_mean = astropy.stats.circstats.circmean(sacc_radians)
                    circ_std = astropy.stats.circstats.circstd(sacc_radians)
                    # https://docs.astropy.org/en/stable/api/astropy.stats.circstats.rayleightest.html#astropy.stats.circstats.rayleightest
                    circ_uniformity_test_pval = astropy.stats.circstats.rayleightest(sacc_radians)
                    templist = [timewindow, phase, cond, subcond, relevant_saccs_subcond.shape[0], circ_mean, circ_std, circ_uniformity_test_pval]
                    for stim_loc in STIM_LOCS_NAMES:
                        loc_degrees = STIM_LOCS_ANGLES[stim_loc]
                        loc_radians = loc_degrees * (math.pi / 180)
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
                            print(f"Could not calculate mtest for dur {timewindow}, cond {cond}, subcond {subcond}, stim_loc {stim_loc}, as it does not meet the required concentration of the data around the mean")
                    summary.append(templist)

    eye_dur_summary_df = pd.DataFrame(summary, columns=summary_cols)
    file_path = os.path.join(sacc_analysis_path)
    f_name = f"sacc_circular_stats_all"
    analysis_saver(eye_dur_summary_df, file_path, file_name=f_name)
    return


def plot_sacc_dir_dist(all_saccades, sacc_analysis_path):
    """
    To generate the plots of the directional distribution of saccades, our "Theta" (x) needs to be the angles of the
    directions, while our "rho" (y) needs to be the COUNT, amt of saccades that are directed towards this angle.
    To receive a distribution we BIN the data to bins of angles. We arbitratily chose fo divide the range (360 deg, 2pi)
    to 30 bins (of 12 degs). This is a constant as it is only used for plotting in this function, thus changing it here
    will do nothing but generate new plots (data is the same)
    :param sacc_data_dict:
    :param sacc_analysis_path:
    :return:
    """
    bins = 30
    bin_size = int(2*180/bins)  # MUST be an integer; if you want to change the number of bins note this number as well
    dist_dict = dict()

    for dur in TIME_WINDOWS:
        dist_dict[dur] = dict()
        for phase in [DataParser.REPLAY_PHASE, DataParser.GAME_PHASE]:  # for each condition of the VG
            dist_dict[dur][phase] = dict()
            for cond in [DataParser.VIS, STIM_TYPE, STIM_TYPE + "_" + DataParser.VIS, DataParser.STIM_LOC]:
                dist_dict[dur][phase][cond] = dict()
                relevant_saccs = all_saccades[(all_saccades["timeWindow"] == dur) & (all_saccades["phase"] == phase) & (all_saccades["cond"] == cond)]
                for subcond in relevant_saccs["subcond"].unique():
                    data = relevant_saccs[relevant_saccs["subcond"] == subcond]
                    total_sub_len = data.shape[0]
                    bin_count_list = []  # this contains the amt of saccades in this bin per bin
                    # iterate over degree BINS (notice the step in the for) and find all saccades which are within
                    # that range
                    for deg in range(-180, 180, bin_size):
                        range_start_rads = deg * (math.pi / 180)
                        range_end_rads = (deg + bin_size) * (math.pi / 180)
                        relevant = data[(range_start_rads <= data[DataParser.SACC_DIRECTION_RAD]) & ((range_end_rads > data[DataParser.SACC_DIRECTION_RAD]))]
                        # add the number of saccades falling within that bin divided by the total number of saccades in that (sub)condition
                        if total_sub_len == 0:
                            bin_count_list.append(0)
                            print(f"No saccades in {dur} {cond} { subcond}")
                        else:
                            bin_count_list.append(relevant.shape[0]/total_sub_len)  # proportion
                    dist_dict[dur][phase][cond][subcond] = bin_count_list

    # now, plot
    for dur in dist_dict.keys():
        for phase in dist_dict[dur].keys():
            for cond in dist_dict[dur][phase].keys():
                if cond == 'visibility':
                    del dist_dict[dur][phase][cond]['True Negative']
                    del dist_dict[dur][phase][cond]['False Positive']
                f_name = f"{dur}_{cond}_dir_dist_{phase}"
                plotter.polar(r_data=dist_dict[dur][phase][cond], r_range=[0, 0.2, 0.05],
                              title=f"{dur.capitalize().replace('_', ' ')} Time-Window",
                              save_name=f_name,
                              save_path=os.path.join(sacc_analysis_path))
                # to save memory, let's delete the data after we're done with it
                dist_dict[dur][phase][cond] = None
                gc.collect()
    return


def saccade_stabilities_in_time(all_saccades, sacc_analysis_path):

    # Calculate saccade start time and end time in the trial : WITH RESPECT TO STIMULUS ONSET (which is why we deduct)
    all_saccades["saccadeStartTime"] = all_saccades["tStart"] - all_saccades["EpochWindowStart"] - ET_param_manager.EPOCH_START
    all_saccades["saccadeEndTime"] = all_saccades["tEnd"] - all_saccades["EpochWindowStart"] - ET_param_manager.EPOCH_END

    # Remove unwanted timings from DF
    all_saccades = all_saccades[all_saccades["timeWindow"] == "Trial"]
    samples_list = list()
    prev_phase = None
    prev_cond = None
    prev_sub = None
    prev_subcond = None
    for sacc in all_saccades.itertuples():
        """
        We want each subject to have all samples (-1000, 1000). 
        However, when adding those fillers to eac saccade - we use a lot of memory.
        In order to solve this, we add all samples only ONCE per subcondition + subject.
        That way, we have all values, and the sum stays the same (rows with 0 value fillers are added only ONCE).
        """
        if prev_phase == sacc.phase and prev_cond == sacc.cond and prev_subcond == sacc.subcond and prev_sub == sacc.Subject:
            tmp_df = pd.DataFrame(index=range(int(max(-ET_param_manager.EPOCH_START, sacc.saccadeStartTime)),
                                              int(min(sacc.saccadeEndTime, ET_param_manager.EPOCH_START) + 1)))  # This DF has all of the relevant epoch times
            tmp_df["tSample"] = range(int(max(-ET_param_manager.EPOCH_START, sacc.saccadeStartTime)),
                                      int(min(sacc.saccadeEndTime, ET_param_manager.EPOCH_START) + 1))
            tmp_df["isSacc"] = [max(-ET_param_manager.EPOCH_START, sacc.saccadeStartTime) <= x <= min(sacc.saccadeEndTime, ET_param_manager.EPOCH_START) for x in range(int(max(-ET_param_manager.EPOCH_START, sacc.saccadeStartTime)),
                                      int(min(sacc.saccadeEndTime, ET_param_manager.EPOCH_START) + 1))]
            tmp_df["phase"] = sacc.phase
            tmp_df["cond"] = sacc.cond
            tmp_df["subcond"] = sacc.subcond
            tmp_df["ampDeg"] = sacc.ampDeg
            tmp_df["Subject"] = sacc.Subject
            tmp_df.loc[tmp_df["isSacc"] == 0, "ampDeg"] = 0
            samples_list.append(tmp_df)
        else:
            tmp_df = pd.DataFrame(index=range(-ET_param_manager.EPOCH_START, ET_param_manager.EPOCH_END + 1))  # This DF has all of the relevant epoch times
            tmp_df["tSample"] = range(-ET_param_manager.EPOCH_START, ET_param_manager.EPOCH_END + 1)
            tmp_df["isSacc"] = [max(-ET_param_manager.EPOCH_START, sacc.saccadeStartTime) <= x <= min(sacc.saccadeEndTime, ET_param_manager.EPOCH_START) for x in range(-ET_param_manager.EPOCH_START, ET_param_manager.EPOCH_END + 1)]
            tmp_df["phase"] = sacc.phase
            tmp_df["cond"] = sacc.cond
            tmp_df["subcond"] = sacc.subcond
            tmp_df["ampDeg"] = sacc.ampDeg
            tmp_df["Subject"] = sacc.Subject
            prev_phase = sacc.phase
            prev_cond = sacc.cond
            prev_subcond = sacc.subcond
            prev_sub = sacc.Subject
            tmp_df.loc[tmp_df["isSacc"] == 0, "ampDeg"] = 0
            samples_list.append(tmp_df)

    sacc_df = pd.concat(samples_list)

    df_within_means = sacc_df.groupby(['tSample', 'phase', 'cond', 'subcond', 'Subject']).sum()  # the MEAN WITHIN SUBJECTS
    df_within_means["ampDeg"] = df_within_means["ampDeg"] / df_within_means["isSacc"]
    df_within_means.reset_index(inplace=True)

    # In order to use isSacc again
    df_within_means.loc[df_within_means["isSacc"] != 0, "isSacc"] = 1
    df_means = df_within_means.groupby(['tSample', 'phase', 'cond', 'subcond']).mean()  # the MEAN ACROSS SUBJECTS where what is averaged here is subjects' means
    df_means.drop(columns=["isSacc"], inplace=True)
    df_means.rename(columns={f"ampDeg": f"ampDegAvg"}, inplace=True)
    df_std = df_within_means.groupby(['tSample', 'phase', 'cond', 'subcond']).std()  # STD BETWEEN SUBJECTS' means (not samples!)
    df_std.fillna(value=0, inplace=True)
    df_std.rename(columns={f"ampDeg": f"ampDegStd"}, inplace=True)
    df_std.drop(columns=["isSacc"], inplace=True)

    df_means = df_means.join(df_std)
    df_means.reset_index(inplace=True)
    df_means.fillna(value=0, inplace=True)
    del df_within_means
    del df_std
    del sacc_df

    saccade_mean_dict = dict()  # dict where the subjects' averaged dfs are contained
    for phase in [DataParser.REPLAY_PHASE, DataParser.GAME_PHASE]:  # for each condition of the VG
        saccade_mean_dict[phase] = dict()
        for cond in [DataParser.VIS, STIM_TYPE, "TARGET"]:
            if phase == DataParser.GAME_PHASE and cond == "TARGET":
                continue
            saccade_mean_dict[phase][cond] = df_means[(df_means["phase"] == phase) & (df_means["cond"] == cond)]
            save_name = f"{phase}_{cond}_saccade"
            saccade_mean_dict[phase][cond].to_csv(os.path.join(sacc_analysis_path, f"{save_name}.csv"), index=False)

            y_avg = f"ampDegAvg"
            y_std = f"ampDegStd"
            plotter.err_line_plot_multidf(cond_df=saccade_mean_dict[phase][cond],
                                          x_col="tSample", y_col=y_avg, subcond_col="subcond", sd_col=y_std,
                                          subcond_color_dict=None,
                                          x_label='Time (ms)', y_label='Amplitude (Deg)',
                                          title=f"Mean Saccade Amplitude Across Subs",
                                          save_path=sacc_analysis_path, save_name=save_name,
                                          vertical_lines={"Stimulus On": 0, "Stimulus Off": ET_param_manager.STIM_DUR},
                                          num_of_x_ticks=10, y_min=None,  y_max=None)

    # We inserted all of df_means into fix_mean_dict, we can clean it
    del df_means
    return


def pos_freq_plot_new(data_path, xmin=-10.0, xmax=10.1, xticks=5, ymin=0.0, ymax=0.6, yticks=0.1):
    file_name = "fix_dist_from_center.pickle"
    fl = open(os.path.join(data_path, file_name), 'rb')
    data = pickle.load(fl)
    fl.close()

    column_generic_name = 'CenterDistDegsSigned'
    metric_dict = {'Width': 'X', 'Height': 'Y'}

    plot_save_name = "fix_pos_freq"

    gc.collect()

    for dur in list(data.keys()):
        for game in list(data[dur].keys()):
            conds = list(data[dur][game].keys())
            for cond in conds:
                for metric in metric_dict.keys():
                    cond_name = cond.lower() if cond != WORLD_ID else data_saver.WORLD.lower()
                    s_path = data_path
                    if cond == VIS:  # we only want to show SEEN VS UNSEEN
                        data[dur][game][cond]['True Negative'] = None
                        data[dur][game][cond]['False Positive'] = None
                    gc.collect()
                    """
                    TL --> BR is the order of things in Eyelink data:
                    Gaze coordinates (X, Y) in Eyelink are such that (0, 0) is the TOP LEFT corner of the screen!!!
                    This means that when gaze goes DOWN --> Y coordinate goes UP! (e.g., y=6 is LOWER than y=5)
                    Source: EL1000 User manual 1.5 chapter 4.4.2.3 GAZE
                    http://sr-research.jp/support/EyeLink%201000%20User%20Manual%201.5.0.pdf
                    
                    Because of that, when we deduct the screen center from all coordinates in dist_from_target, 
                    what happens is that now, NEGATIVE Y values mean HIGHER on the screen (UP) and POSITIVE Y values
                    are LOWER on the screen. 
                    """
                    plotter.simple_lineplot(data=data[dur][game][cond], line_key='mean', shadow_key='std',
                                            y_col=f"{metric_dict[metric]}{column_generic_name}", x_col="Bin Value",
                                            x_label=f"{metric} Position (degree)", y_label="Fixation Proportion",
                                            title=f"{metric} Fixation Frequency", save_path=s_path,
                                            save_name=f"{dur}_{cond_name}_{metric.lower()}_" + plot_save_name, colors=None, y_min=ymin, y_max=ymax,
                                            y_ticks=yticks, x_min=xmin, x_max=xmax, x_ticks=xticks)
                # free some space
                del data[dur][game][cond]
                gc.collect()
    return


def get_relevant_subcond(sub_data, cond, phase):
    if cond == DataParser.VIS:
        rel_subconds = sub_data[TRIAL_INFO][cond].unique()
    elif cond == STIM_TYPE:
        if phase == DataParser.REPLAY_PHASE:
            rel_subconds = ["Face_target", "Face_non_target", "Object_target", "Object_non_target", "Blank", "Target", "Non_target"]
        else:
            rel_subconds = ["Face", "Object", "Blank"]
    return rel_subconds


def pupils_stabilities_in_time(sub_names, relevant_subs, subs_analysis_pupil_path):
    pupil_sizes = list()

    for i in range(len(relevant_subs)):  # Iterate over subjects
        sub_path = relevant_subs[i]
        sub_name = sub_names[i]
        fl = open(os.path.join(sub_path, f"{sub_name}EyeTrackingData.pickle"), 'rb')  # distance data is in this pickle
        sub_data = pickle.load(fl)
        fl.close()

        sub_eye = sub_data[PARAMS]["Eye"]
        if sub_eye == "LR":
            print("Only monocular data is considered: 2022-09-12 consortium decision")
            continue

        sub_data[TRIAL_INFO] = prepare_trial_info(sub_data[TRIAL_INFO])
        # AT THE SUBJECT LEVEL: calculate per-sub averages of gaze distance over time
        # this is for a single subject at a time
        for phase in [DataParser.REPLAY_PHASE, DataParser.GAME_PHASE]:  # for each condition of the VG
            for cond in [DataParser.VIS, STIM_TYPE]:
                for subcond in get_relevant_subcond(sub_data, cond, phase):  # so all subjects will have all conditions
                    trial_info = sub_data[TRIAL_INFO]
                    if phase == DataParser.REPLAY_PHASE and subcond == "Target":
                        trial_info = trial_info[trial_info[WORLD_ID].isin(CONDITIONS[WORLD_ID][phase]) & (trial_info[cond].isin(["Face_target", "Object_target"]))]
                        cond_name = "TARGET"
                    elif phase == DataParser.REPLAY_PHASE and subcond == "Non_target":
                        trial_info = trial_info[trial_info[WORLD_ID].isin(CONDITIONS[WORLD_ID][phase]) & (trial_info[cond].isin(["Face_non_target", "Object_non_target"]))]
                        cond_name = "TARGET"
                    else:
                        cond_name = cond
                        trial_info = trial_info[trial_info[WORLD_ID].isin(CONDITIONS[WORLD_ID][phase]) & (trial_info[cond] == subcond)]
                    relevant_trials = list(trial_info[TRIAL_NUMBER])
                    cols = [DataParser.T_SAMPLE, f"{sub_eye}Pupil"]
                    # filter samples to include only samples of the relevant trial epochs
                    relevant_samples = sub_data[ET_DATA_DICT][DF_SAMPLES][(sub_data[ET_DATA_DICT][DF_SAMPLES][DataParser.TRIAL].isin(relevant_trials))]
                    relevant_samples = relevant_samples[relevant_samples[DataParser.IS_LEVEL_ELIMINATED] != 1]

                    for trial_num in relevant_trials:  # for each trial (average per sub is across trials)
                        # align timings for averaging - bring all trials' times to normalized around 0
                        # where 0 = STIM ONSET
                        stim_onset = trial_info.loc[trial_info[TRIAL_NUMBER] == trial_num, ONSET]
                        relevant_samples.loc[relevant_samples[DataParser.TRIAL] == trial_num, DataParser.T_SAMPLE] -= int(stim_onset)

                    # average (within subject, across trials)
                    data = relevant_samples[cols]
                    # Average the fixations in regard to the relevant timestamp in epoch
                    sub_avg_data = data.groupby(['tSample']).mean()
                    sub_avg_data.rename(columns={f"{sub_eye}Pupil": f"PupilAvg"}, inplace=True)
                    # Count number of entries in each such average
                    sub_count_data = data.groupby(['tSample']).count()
                    sub_count_data.rename(columns={f"{sub_eye}Pupil": f"PupilCount"}, inplace=True)

                    full_df = pd.DataFrame(index=range(-1000, 1001), columns=range(1))  # This DF has all of the epoch times (from -1000 to 1000)
                    full_df = full_df.join(sub_avg_data)
                    full_df = full_df.join(sub_count_data)
                    full_df.drop(columns=[0], inplace=True)
                    full_df[[f"PupilCount"]] = full_df[[f"PupilCount"]].fillna(value=0)
                    full_df.reset_index(inplace=True)
                    full_df.rename(columns={"index": "tSample"}, inplace=True)
                    full_df["Subject"] = sub_name
                    full_df["phase"] = phase
                    full_df["cond"] = cond_name
                    full_df["subcond"] = subcond
                    pupil_sizes.append(full_df)

    pupil_dist = pd.concat(pupil_sizes)
    # has_data is 0 in the case where a subject did not have any data in this specific sample.
    pupil_dist["has_data"] = 0
    pupil_dist.loc[pupil_dist[f"PupilCount"] != 0, "has_data"] = 1

    # Now we calculate the avg and the std of each relevant timestamps in regard to the subcondition (=sub-condition dataframe, of all subjects' averages concatenated)
    # The std is of the means of the subjects (per timestamp)
    df_sum = pupil_dist.groupby(['tSample', 'phase', 'cond', 'subcond']).sum()  # in order to count across how many instances was the sample averaged (subjects!)
    pupil_dist.drop(columns=["has_data"], inplace=True)
    df_means = pupil_dist.groupby(['tSample', 'phase', 'cond', 'subcond']).mean()  # the MEAN ACROSS SUBJECTS where what is averaged here is subjects' means
    df_means.drop(columns=[f"PupilCount"], inplace=True)
    df_std = pupil_dist.groupby(['tSample', 'phase', 'cond', 'subcond']).std()  # STD BETWEEN SUBJECTS' means (not samples!)
    df_std.drop(columns=[f"PupilCount"], inplace=True)
    df_std.fillna(value=0, inplace=True)
    df_std.rename(columns={f"PupilAvg": f"PupilStd"}, inplace=True)

    df_means = df_means.join(df_std)
    df_means["SubjectCount"] = df_sum["has_data"]
    df_means.reset_index(inplace=True)
    # Only df_means is used from now on
    del pupil_dist
    del df_std
    del df_sum

    pupil_mean_dict = dict() # dict where the subjects' averaged dfs are contained
    for phase in [DataParser.REPLAY_PHASE, DataParser.GAME_PHASE]:  # for each condition of the VG
        pupil_mean_dict[phase] = dict()
        for cond in [DataParser.VIS, STIM_TYPE, "TARGET"]:
            if phase == DataParser.GAME_PHASE and cond == "TARGET":
                continue
            pupil_mean_dict[phase][cond] = df_means[(df_means["phase"] == phase) & (df_means["cond"] == cond)]
            save_name = f"{phase}_{cond}_pupils"
            pupil_mean_dict[phase][cond].to_csv(os.path.join(subs_analysis_pupil_path, f"{save_name}.csv"), index=False)

            y_avg = f"PupilAvg"
            y_std = f"PupilStd"
            plotter.err_line_plot_multidf(cond_df=pupil_mean_dict[phase][cond],
                                          x_col="tSample", y_col=y_avg, subcond_col="subcond", sd_col=y_std,
                                          subcond_color_dict=None,
                                          x_label='Time (ms)', y_label='Pupil Size',
                                          title=f"Mean Pupil Size Across Subs",
                                          save_path=subs_analysis_pupil_path, save_name=save_name,
                                          vertical_lines={"Stimulus On": 0, "Stimulus Off": ET_param_manager.STIM_DUR},
                                          num_of_x_ticks=10, y_min=None,  y_max=None)

    # We inserted all of df_means into pupil_mean_dict, we can clean it
    del df_means

    return


def fixation_stabilities_in_time(sub_names, relevant_subs_paths, subs_analysis_path):
    """
    This method is for aggregating and plotting the distance between fixation gaze and stim/target over time, across
    all trials of a certain type (trial=entire epoch duration). This is calculated by aggregating all epoch samples
    across all trials (of a certain type) of each subject, averaging them (per subject) and then averaging across
    subjects (so that STD is between subjects' averages).
    :param subs_qc_path:
    :param subs_analysis_path:
    :return:
    """
    fixation_dists = list()

    for i in range(len(relevant_subs_paths)):  # Iterate across subjects
        sub_path = relevant_subs_paths[i]
        sub_name = sub_names[i]
        fl = open(os.path.join(sub_path, f"{sub_name}EyeTrackingData.pickle"), 'rb')  # distance data is in this pickle
        sub_data = pickle.load(fl)
        fl.close()

        sub_eye = sub_data[PARAMS]["Eye"]
        if sub_eye == "LR":
            print("Only monocular data is considered: 2022-09-12 consortium decision")
            continue

        sub_data[TRIAL_INFO] = prepare_trial_info(sub_data[TRIAL_INFO])
        # AT THE SUBJECT LEVEL: calculate per-sub averages of gaze distance over time
        # this is for a single subject at a time
        for phase in [DataParser.REPLAY_PHASE, DataParser.GAME_PHASE]:  # for each condition of the VG
            for cond in [DataParser.VIS, STIM_TYPE]:
                for subcond in get_relevant_subcond(sub_data, cond, phase):  # so all subjects will have all conditions
                    trial_info = sub_data[TRIAL_INFO]
                    if phase == DataParser.REPLAY_PHASE and subcond == "Target":
                        trial_info = trial_info[trial_info[WORLD_ID].isin(CONDITIONS[WORLD_ID][phase]) & (trial_info[cond].isin(["Face_target", "Object_target"]))]
                        cond_name = "TARGET"
                    elif phase == DataParser.REPLAY_PHASE and subcond == "Non_target":
                        trial_info = trial_info[trial_info[WORLD_ID].isin(CONDITIONS[WORLD_ID][phase]) & (trial_info[cond].isin(["Face_non_target", "Object_non_target"]))]
                        cond_name = "TARGET"
                    else:
                        cond_name = cond
                        trial_info = trial_info[trial_info[WORLD_ID].isin(CONDITIONS[WORLD_ID][phase]) & (trial_info[cond] == subcond)]
                    trial_info = trial_info[trial_info[DataParser.IS_LEVEL_ELIMINATED] != 1]  # Filter out eliminated levels
                    relevant_trials = list(trial_info[TRIAL_NUMBER])

                    eye = sub_eye  # Only analyzed eye
                    cols = [DataParser.T_SAMPLE, f"{eye}CenterDistDegs", f"{eye}StimDistDegs"]
                    if f"{eye}{DataParser.EYELINK}Fix" not in sub_data[ET_DATA_DICT][DF_SAMPLES]:
                        continue
                    # filter samples to include only samples of the relevant trial epochs
                    relevant_samples = sub_data[ET_DATA_DICT][DF_SAMPLES][(sub_data[ET_DATA_DICT][DF_SAMPLES][DataParser.TRIAL].isin(relevant_trials))]
                    relevant_samples = filter_samples(relevant_samples, col_relevant=f"{eye}{DataParser.EYELINK}Fix",
                                                      cols_nonrelevant=[f"{eye}{DataParser.HERSHMAN}",
                                                                        f"{eye}{DataParser.EYELINK}Blink",
                                                                        f"{eye}{DataParser.EYELINK}Sacc"])

                    for trial_num in relevant_trials:  # for each trial (average per sub is across trials)
                        # align timings for averaging - bring all trials' times to normalized around 0
                        # where 0 = STIM ONSET
                        stim_onset = int(trial_info.loc[trial_info[TRIAL_NUMBER] == trial_num, ONSET])
                        relevant_samples.loc[relevant_samples[DataParser.TRIAL] == trial_num, DataParser.T_SAMPLE] -= stim_onset

                    # average (within subject, across trials)
                    data = relevant_samples[cols]
                    # Average the fixations in regard to the relevant timestamp in epoch
                    sub_avg_data = data.groupby(['tSample']).mean()
                    sub_avg_data.rename(columns={f"{eye}CenterDistDegs": f"CenterDistDegsAvg", f"{eye}StimDistDegs": f"StimDistDegsAvg"}, inplace=True)
                    # Count number of entries in each such average
                    sub_count_data = data.groupby(['tSample']).count()
                    sub_count_data.rename(columns={f"{eye}CenterDistDegs": f"CenterDistDegsCount", f"{eye}StimDistDegs": f"StimDistDegsCount"}, inplace=True)

                    full_df = pd.DataFrame(index=range(-1000, 1001), columns=range(1))  # This DF has all of the epoch times (from -1000 to 1000)
                    full_df = full_df.join(sub_avg_data)
                    full_df = full_df.join(sub_count_data)
                    full_df.drop(columns=[0], inplace=True)
                    full_df[[f"CenterDistDegsCount", f"StimDistDegsCount"]] = full_df[[f"CenterDistDegsCount", f"StimDistDegsCount"]].fillna(value=0)
                    full_df.reset_index(inplace=True)
                    full_df.rename(columns={"index": "tSample"}, inplace=True)
                    full_df["Subject"] = sub_name
                    full_df["eye"] = eye
                    full_df["phase"] = phase
                    full_df["cond"] = cond_name
                    full_df["subcond"] = subcond
                    fixation_dists.append(full_df)

    fix_dist = pd.concat(fixation_dists)
    # has_data is 0 in the case where a subject did not have any data in this specific sample.
    fix_dist["has_data"] = 0
    fix_dist.loc[fix_dist[f"CenterDistDegsCount"] != 0, "has_data"] = 1

    # Now we calculate the avg and the std of each relevant timestamps in regard to the subcondition (=sub-condition dataframe, of all subjects' averages concatenated)
    # The std is of the means of the subjects (per timestamp)
    df_sum = fix_dist.groupby(['tSample', 'phase', 'cond', 'subcond']).sum()  # in order to count across how many instances was the sample averaged (subjects!)
    fix_dist.drop(columns=["has_data"], inplace=True)
    df_means = fix_dist.groupby(['tSample', 'phase', 'cond', 'subcond']).mean()  # the MEAN ACROSS SUBJECTS where what is averaged here is subjects' means
    df_means.drop(columns=[f"CenterDistDegsCount", f"StimDistDegsCount"], inplace=True)
    df_std = fix_dist.groupby(['tSample', 'phase', 'cond', 'subcond']).std()  # STD BETWEEN SUBJECTS' means (not samples!)
    df_std.drop(columns=[f"CenterDistDegsCount", f"StimDistDegsCount"], inplace=True)
    df_std.fillna(value=0, inplace=True)
    df_std.rename(columns={f"CenterDistDegsAvg": f"CenterDistDegsStd",
                           f"StimDistDegsAvg": f"StimDistDegsStd"}, inplace=True)

    df_means = df_means.join(df_std)
    df_means["SubjectCount"] = df_sum["has_data"]
    df_means.reset_index(inplace=True)
    # Only df_means is used from now on
    del fix_dist
    del df_std
    del df_sum

    fix_mean_dict = dict()  # dict where the subjects' averaged dfs are contained
    for phase in [DataParser.REPLAY_PHASE, DataParser.GAME_PHASE]:  # for each condition of the VG
        fix_mean_dict[phase] = dict()
        for cond in [DataParser.VIS, STIM_TYPE, "TARGET"]:
            if phase == DataParser.GAME_PHASE and cond == "TARGET":
                continue
            fix_mean_dict[phase][cond] = df_means[(df_means["phase"] == phase) & (df_means["cond"] == cond)]
            save_name = f"{phase}_{cond}_fixations"
            fix_mean_dict[phase][cond].to_csv(os.path.join(subs_analysis_path, f"{save_name}.csv"), index=False)

            ys = {"CenterDistDegs": "Center", "StimDistDegs": "Target Stimulus"}

            for y in ys:
                plot_save_name = f"{phase}_{cond}_fixations_{y}"
                y_avg = f"{y}Avg"
                y_std = f"{y}Std"
                plotter.err_line_plot_multidf(cond_df=fix_mean_dict[phase][cond],
                                      x_col="tSample", y_col=y_avg, subcond_col="subcond", sd_col=y_std,
                                              subcond_color_dict=None,
                                      x_label='Time (ms)', y_label='Distance (Deg)',
                                      title=f"Mean Euclidean Distance from {ys[y]} Across Subs",
                                      save_path=subs_analysis_path, save_name=plot_save_name,
                                      vertical_lines={"Stimulus On": 0, "Stimulus Off": ET_param_manager.STIM_DUR},
                                      num_of_x_ticks=10, y_min=None,  y_max=None)

    # We inserted all of df_means into fix_mean_dict, we can clean it
    del df_means

    return


def fixation_stabilities_hist(sub_names, relevant_subs_paths, subs_analysis_path):
    """
    :param subs_qc_path:
    :param subs_analysis_path:
    :return:
    """
    all_subs_df = pd.DataFrame()
    for i in range(len(relevant_subs_paths)):  # Iterate across subjects
        sub_path = relevant_subs_paths[i]
        sub_name = sub_names[i]
        fl = open(os.path.join(sub_path, f"{sub_name}EyeTrackingData.pickle"), 'rb')  # distance data is in this pickle
        sub_data = pickle.load(fl)
        fl.close()

        sub_eye = sub_data[PARAMS]["Eye"]
        if sub_eye == "LR":
            print("Only monocular data is considered: 2022-09-12 consortium decision")
            continue

        sub_data[TRIAL_INFO] = prepare_trial_info(sub_data[TRIAL_INFO])
        trial_info = sub_data[TRIAL_INFO]
        trial_info = trial_info[trial_info[DataParser.IS_LEVEL_ELIMINATED] != 1]  # Filter out eliminated levels
        relevant_trials = list(trial_info[TRIAL_NUMBER])
        relevant_samples = sub_data[ET_DATA_DICT][DF_SAMPLES][(sub_data[ET_DATA_DICT][DF_SAMPLES][DataParser.TRIAL].isin(relevant_trials))]
        relevant_samples = filter_samples(relevant_samples, col_relevant=f"{sub_eye}{DataParser.EYELINK}Fix",
                                          cols_nonrelevant=[f"{sub_eye}{DataParser.HERSHMAN}",
                                                            f"{sub_eye}{DataParser.EYELINK}Blink",
                                                            f"{sub_eye}{DataParser.EYELINK}Sacc"])
        df_sub_means = relevant_samples.groupby([TRIAL]).mean().reset_index()

        df_sub_means.rename(columns={f"{sub_eye}StimDistDegs": f"StimDistDegs", f"{sub_eye}CenterDistDegs": f"CenterDistDegs"}, inplace=True)
        df_sub_means = df_sub_means[[TRIAL, "StimDistDegs", "CenterDistDegs"]]
        df_sub_means["Lab"] = sub_name[0:2]
        df_sub_means["Subject"] = sub_name[2:]

        all_subs_df = pd.concat([all_subs_df, df_sub_means], axis=0)  # concatenate and add to an all-sub df

    all_subs_df.to_csv(os.path.join(subs_analysis_path, f"fix_dist_hist.csv"), index=False)

    # now plot: RONY SPLIT TO LABS
    target_names = {"CenterDistDegs": "center", "StimDistDegs": "stimulus"}
    for target in target_names:
        plotter.hist_plot(data=all_subs_df, x_col=target,
                          title=f"Fixation distance from {target_names[target]}".title(),
                          x_label="Distance (Degrees VA)", y_label="Number of Trials",
                          save_path=subs_analysis_path, save_name=f"fix_dist_hist_{target_names[target]}")
    return


def fixation_position_frequency(sub_names, relevant_subs_paths, res_path):
    """
    This method will bin the fixation position data (its distance from the center) to create distributions per each
    duration and condition across subs. To do so, fix_data_in_trial will bring the data to the right format
    (binned, averaged, cross-subject data, divided per duration and condition), and pos_freq_plot will plot the data.
    Note the the data is not transferred between the functions but saved by fix_data_in_trial to a pickle file, which
    pos_freq_plot loads.
    :param data_path:
    :param res_path:
    :return:
    """
    num_of_bins = 100
    bin_min = -10.0
    bin_max = 10.0

    bin_plot_skips = 5

    prop_plot_min = 0.0
    prop_plot_max = 0.6
    prop_plot_ticks = 0.1

    print("calculating fixation frequency data")
    bin_fix_dist(sub_names, relevant_subs_paths, res_path, num_of_bins, bin_min, bin_max)
    print("plotting fixation frequency data")
    pos_freq_plot_new(res_path, xmin=bin_min, xmax=bin_max, xticks=bin_plot_skips, ymin=prop_plot_min, ymax=prop_plot_max,
                  yticks=prop_plot_ticks)

    return


def saccade_analysis(sub_names, relevant_subs_paths, sacc_analysis_path):

    # 1: get the saccade data for each duration+cond+subcond across subjects (dataframe with Subject column)
    all_saccades = get_saccades_data(sub_names, relevant_subs_paths, sacc_analysis_path)

    # 2: calculate circular statistics on the saccade data and save them to csv files
    print(f"Saccade Circular Statistics")
    # this is Rayleigh and V test
    circular_sacc_analysis(all_saccades, sacc_analysis_path)

    # 3: plot the directional distribution of saccades
    print(f"Directional Distribution of Saccades")
    plot_sacc_dir_dist(all_saccades, sacc_analysis_path)

    # 4: plot and calculate saccades amplitudes in time
    print(f"Mean saccade amplitudes in time")
    saccade_stabilities_in_time(all_saccades, sacc_analysis_path)

    return


def analyze_saccades(sub_names, relevant_subs_paths, subs_analysis_path):
    """
    We do this as a process as we don't rely on python calls to the garbage collector and we must release some of the
    memory that's being in use
    :param subs_qc_path:
    :param subs_analysis_path:
    :return:
    """
    sacc_analysis_path = os.path.join(subs_analysis_path, data_saver.SACC_NAME)

    p = Process(target=saccade_analysis, args=(sub_names, relevant_subs_paths, sacc_analysis_path))
    p.start()
    p.join()
    if p.exitcode > 0:  # if the process exited with an error
        exit(1)  # exit with an error as well
    return


def fixation_analysis(sub_names, relevant_subs_paths, subs_analysis_fixation_path):
    density_path = os.path.join(subs_analysis_fixation_path, data_saver.FIX_DENS)
    stability_path = os.path.join(subs_analysis_fixation_path, data_saver.STAB)

    # 1: FIXATION: DENSITY PICKLE FILE
    print("Fixation Density")
    p = Process(target=analyze_fixation_densities, args=(sub_names, relevant_subs_paths, density_path))
    p.start()
    p.join()
    if p.exitcode > 0:  # if the process exited with an error
        exit(1)  # exit with an error as well

    # 1': FIXATION: DENSITY PLOTS
    print("Fixation Density")
    p = Process(target=plot_fixation_densities, args=(density_path,))  # the comma here is deliberate, it makes args a tuple
    p.start()
    p.join()
    if p.exitcode > 0:  # if the process exited with an error
        exit(1)  # exit with an error as well

    # 2: FIXATION: STABILITY STATS: MEAN DIST FROM TARGET (STIM/CENTER) + PROPORTION OF KEEPING FIXATION
    print("Fixation Stability")
    p = Process(target=analyze_fixation_stabilities, args=(sub_names, relevant_subs_paths, stability_path))
    p.start()
    p.join()
    if p.exitcode > 0:  # if the process exited with an error
        exit(1)  # exit with an error as well

    # 2': FIXATION: STABILITY PLOTS (STATS PER TIMEPOINT): DIST FROM TARGET (STIM/CENTER)
    print("Fixation Stability in Time")
    p = Process(target=fixation_stabilities_in_time, args=(sub_names, relevant_subs_paths, stability_path))
    p.start()
    p.join()
    if p.exitcode > 0:  # if the process exited with an error
        exit(1)  # exit with an error as well

    # 3: FIXATION: STABILITY HISTOGRAMS OVER THE ENTIRE EPOCH (DIST IN VA FROM STIM/CENTER)
    print("Fixation Stability Across Labs")
    p = Process(target=fixation_stabilities_hist, args=(sub_names, relevant_subs_paths, stability_path))
    p.start()
    p.join()
    if p.exitcode > 0:  # if the process exited with an error
        exit(1)  # exit with an error as well

    # 4: FIXATION: FREQUENCY OF GAZE POSITION PER TRIAL TYPE (DUR + CONDITION + TIME WINDOW)
    print("Fixation Frequency")
    p = Process(target=fixation_position_frequency, args=(sub_names, relevant_subs_paths, stability_path))
    p.start()
    p.join()
    if p.exitcode > 0:  # if the process exited with an error
        exit(1)  # exit with an error as well

    # 5 PREPARE AND SAVE DATA FOR GLMMS WHICH WILL BE DONE IN R SOFTWARE
    print("Fixation GLMM preparation")
    p = Process(target=glmm_prep_fixations, args=(sub_names, relevant_subs_paths, stability_path))
    p.start()
    p.join()
    if p.exitcode > 0:  # if the process exited with an error
        exit(1)  # exit with an error as well
    return


def analyze_fixations(sub_names, relevant_subs_paths, subs_analysis_path):
    """
    Manager function for fixation analysis.
    """
    # fixation results will be saved in this folder
    subs_analysis_fixation_path = os.path.join(subs_analysis_path, FIXATION)
    print(f"Analyzing fixations")
    p = Process(target=fixation_analysis, args=(sub_names, relevant_subs_paths, subs_analysis_fixation_path))
    p.start()
    p.join()
    if p.exitcode > 0:  # if the process exited with an error
        exit(1)  # exit with an error as well

    return


def pupil_analysis(sub_names, relevant_subs, subs_analysis_pupil_path):
    # 1: PUPIL: plot and calculate sizes over time
    print("Pupil sizes over time")
    p = Process(target=pupils_stabilities_in_time, args=(sub_names, relevant_subs, subs_analysis_pupil_path))
    p.start()
    p.join()
    if p.exitcode > 0:  # if the process exited with an error
        exit(1)  # exit with an error as well


def analyze_pupils(sub_names, relevant_subs, subs_analysis_path):
    """
    Manager function for pupil analysis.
    """
    # pupil results will be saved in this folder
    subs_analysis_pupil_path = os.path.join(subs_analysis_path, PUPIL)

    print(f"Analyzing pupils")
    p = Process(target=pupil_analysis, args=(sub_names, relevant_subs, subs_analysis_pupil_path))
    p.start()
    p.join()
    if p.exitcode > 0:  # if the process exited with an error
        exit(1)  # exit with an error as well

    return


def analyze(subs_path, save_path):
    """
    Manager of all analyses performed on ET data. Analyses are based on data extracted and generated at the subject level
    during the quality check phase.
    :param subs_path: path to the subject result folder, a product of the QualityChecker module, which contains
    :return:
    """
    # initialize a folder structure for subject results
    subs_analysis_path = data_saver.create_subject_analysis(save_path)
    # assume there's already a QC folder structure to work with: find it
    relevant_subs = []
    sub_names = []
    for mod in ET_qc_manager.MODALITY_MAPPING.keys():
        mod_result_path = os.path.join(subs_path, mod, ET_qc_manager.PHASE, ET_qc_manager.PROCESSED, ET_qc_manager.BIDS, ET_qc_manager.DERIV, ET_qc_manager.QCS)
        if not os.path.exists(mod_result_path):
            print(f"No {mod} data found; moving on to next modality")
            continue
        for sub_dir in os.listdir(mod_result_path):  # iterate over PROCESSED data
            sub_result_path = os.path.join(mod_result_path, sub_dir, ET_qc_manager.SES_V2, ET_qc_manager.ET_RES_FOLD)
            sub_name = sub_dir.replace(ET_qc_manager.SUB_PREFIX, '')
            if not os.path.exists(sub_result_path):
                print(f"{sub_name} has no ET pickle file; please run the pre-processing first")
                continue
            pick_file = [f for f in os.listdir(sub_result_path) if fnmatch.fnmatch(f, f"{sub_name}EyeTrackingData.pickle")]
            if len(pick_file) == 0:
                print(f"{sub_name} has no ET pickle file; please run the pre-processing first")
                continue
            pick_file = [f for f in os.listdir(sub_result_path) if fnmatch.fnmatch(f, "fix_density.pickle")]
            if len(pick_file) == 0:
                print(f"{sub_name} has no fixation density pickle file; please run the QC first")
                continue
            else:
                sub_names.append(sub_name)
                relevant_subs.append(sub_result_path)

    # ANALYZE FIXATION-RELATED DATA
    analyze_fixations(sub_names, relevant_subs, subs_analysis_path)

    # ANALYZE SACCADE-RELATED DATA
    analyze_saccades(sub_names, relevant_subs, subs_analysis_path)

    # ANALYZE BLINK-RELATED DATA
    analyze_blinks(sub_names, relevant_subs, subs_analysis_path)

    # ANALYZE PUPIL-SIZE DATA
    analyze_pupils(sub_names, relevant_subs, subs_analysis_path)

    return


def analysis_saver(df, save_path, file_name, folder_name="", sub_folder_name="", save_type="csv"):
    if sub_folder_name != "":
        sub_split = os.path.split(sub_folder_name)
        data_saver.create_dir(os.path.join(save_path, folder_name, sub_split[0]))
        data_saver.create_dir(os.path.join(save_path, folder_name, sub_split[0], sub_split[1]))
    data_saver.safe_save(os.path.join(save_path, folder_name, sub_folder_name), file_name + f".{save_type}")
    if save_type == "csv":
        df.to_csv(os.path.join(os.path.join(save_path, folder_name, sub_folder_name), file_name + f".{save_type}"), index=False)
    elif save_type == "xlsx":
        with pd.ExcelWriter(os.path.join(os.path.join(save_path, folder_name, sub_folder_name), file_name + f".{save_type}")) as xwriter:
            df.to_excel(xwriter)
    else:
        raise TypeError
    return


if __name__ == '__main__':
    analyze(subs_path=r"",
            save_path=r"")
