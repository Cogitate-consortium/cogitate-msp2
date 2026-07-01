import os
import pandas as pd
import numpy as np
import plotter
import ET_param_manager

CSV = ".csv"
XLS = ".xlsx"

SUBCODE_LEN = 5
CATEGORY = 'Category'
WORLD_GAME = 'GameWorld'
WORLD_REPLAY = 'ReplayWorld'
LOC = 'Location'
VIS = 'Visibility'

# list of all the names of the sub-directories within a single subject QC result dir
all_conditions = {CATEGORY: list(), WORLD_GAME: list(), WORLD_REPLAY: list(), LOC: list(), VIS: list()}
# csv/excel file name prefixes that if a file has them all files with the same name need to be averaged
files_to_avg = ['QC_results_Epoch.xlsx', 'QC_results_PreStim.xlsx', 'QC_results_StimDuration.xlsx',
                'MeanDistFrCenterAll_mean.csv', 'MeanDistFrStimAll_mean.csv', 'FixDensEpochAll.csv',
                'FixDensPreStimAll.csv', 'FixDensStimDurAll.csv']


def safe_save(folder_path, file_name):
    """
    Checks whether the folder path + file name combo already exist in folder. This function is called right before
    saving data to an IDENTICAl combo, so if a file like that is found it's DELETED, to be replaced by the new file
    that we are trying to save.
    Meaning, if the folder path + file name exist - the file is DELETED and then replaced by a file with an identical
    name.
    :param folder_path: path where the file is saved
    :param file_name: file name
    """
    existing_files = [f for f in os.listdir(folder_path)]
    is_duplicate = 0
    if len(existing_files) == 0:
        print(f"No files in directory {folder_path}")
        return
    for f in existing_files:
        if f == file_name:
            is_duplicate = 1
    if is_duplicate == 1:
        print(f"File {f} already exists in {folder_path}. Save action is replacing the old file with a new one.")
        # remove the old file
        os.remove(os.path.join(folder_path, file_name))
    else:
        print(f"File {f} was not found in {folder_path}. Save action will not override anything.")
    return


def filter_subject_folders(sub_dir):
    """
    Given a list of directories which are supposed to represent subjects, verify that it is indeed the case and
    return a list of items which are directory names which represent subjects (filter our non-subject ones)
    :param sub_dir: a list of directories which are supposed to represent subjects
    :return: a list of subject directories
    """
    res = list()
    for s in sub_dir:
        if len(s) == SUBCODE_LEN:
            if s[:2].isalpha() and s[0] == "S":
                if s[2:].isdigit():
                    res.append(s)
    return res


def check_dir_tree(sub_path):
    """
    Given a subject dir path, check it contains all the required subfolders of the QC output data
    :param sub_path:
    :return:
    """
    sub_dirs = [d for d in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, d))]
    if set(sub_dirs) == set(list(all_conditions.keys())):  # subject contains all subfolders
        return True
    return False


def create_folder(folder_path):
    try:
        os.mkdir(folder_path)
    except OSError:
        print(f"DIR {folder_path} ALREADY EXISTS")
    return folder_path


def avg_file_list(file_list, sub_list, qc_summary_dir, subject_sub_dir, save_path):  # ADD CATEGORY FOLDER OPTIONAL TO NESTING
    files_data = {f: {'mean': None, 'sem': None, 'n': 0} for f in file_list}
    for f in files_data:
        file_data_list = list()
        for sub in sub_list:
            sub_folder = os.path.join(qc_summary_dir, sub) if subject_sub_dir is None \
                else os.path.join(qc_summary_dir, sub, subject_sub_dir)
            f_folder = os.path.join(sub_folder, f)
            if f.endswith(CSV):
                f_data = pd.read_csv(f_folder)
            else:
                f_data = pd.read_excel(f_folder)
            file_data_list.append(f_data)
        # average all dfs
        files_data[f]['mean'] = pd.concat(file_data_list).groupby(level=0).mean()
        files_data[f]['sem'] = pd.concat(file_data_list).groupby(level=0).sem()
        files_data[f]['n'] = len(file_data_list)
        # save
        print(f"SAVING {f}")
        prefix = f.split(".")[0]
        suffix = f.split(".")[-1]
        index = True if f.startswith("QC_results") else False
        safe_save(save_path, f"{prefix}_AVG_{files_data[f]['n']}.{suffix}")
        files_data[f]['mean'].to_csv(os.path.join(save_path, f"{prefix}_AVG_{files_data[f]['n']}.csv"),index=index)
        safe_save(save_path, f"{prefix}_SEM_{files_data[f]['n']}.{suffix}")
        files_data[f]['sem'].to_csv(os.path.join(save_path, f"{prefix}_SEM_{files_data[f]['n']}.csv"), index=index)
    return files_data


def ET_qc_plots(stats_dir_path, save_path, sampling_freq=1000):
    target_dict = {"Stim": [5, 9, 11, ["violet"]], "Center": [5, 0, 2, ["mediumturquoise"]]}
    for target in target_dict.keys():
        mean_dist_from_target = pd.read_csv(os.path.join(stats_dir_path, f"MeanDistFr{target}All_mean_AVG_30.csv"))
        mean_dist_from_target.drop(mean_dist_from_target.columns[0], axis=1, inplace=True)
        sem_dist_from_target = pd.read_csv(os.path.join(stats_dir_path, f"MeanDistFr{target}All_mean_SEM_30.csv"))
        sem_dist_from_target.rename(columns={"mean": "sem"}, inplace=True)
        sem_dist_from_target.drop(sem_dist_from_target.columns[0], axis=1, inplace=True)
        dist_from_target = pd.concat([mean_dist_from_target, sem_dist_from_target], axis=1).T.drop_duplicates().T
        # data for plot
        sample_of_stim_start = (ET_param_manager.EPOCH_START / 1000) * sampling_freq
        sample_of_stim_end = ((ET_param_manager.EPOCH_START + ET_param_manager.STIM_DUR) / 1000) * sampling_freq
        plotter.err_line_plot(cond_data=dist_from_target, x_label='Time (ms)', y_label='Distance (Deg)',
                              title=f"Mean Euclidean Distance from {target} Across Subs", save_path=save_path,
                              vertical_lines={"Stimulus On": sample_of_stim_start, "Stimulus Off": sample_of_stim_end},
                              x_conversion=sampling_freq, x_zero=sample_of_stim_start,color_list=target_dict[target][3],
                              num_of_x_ticks=target_dict[target][0], y_min=target_dict[target][1],
                              y_max=target_dict[target][2])

    time_dict = ["PreStim", "StimDur", "Epoch"]
    for t in time_dict:
        mean_fix_dens = pd.read_csv(os.path.join(stats_dir_path, f"FixDens{t}All_AVG_30.csv"))
        mean_fix_dens.drop(mean_fix_dens.columns[0], axis=1, inplace=True)
        plotter.heatmaps(heatmap_data=mean_fix_dens, x_label="Screen Width", y_label="Screen Height",
                         title=f"Fixation Density During {t} Across Subs", with_pic="game_sync.png",
                         save_path=save_path, n_rows=1, n_cols=1, figsize=(20, 10))


def ET_qc_stats(qc_summary_dir, save_path):
    """
    Provide statistical information and analyses for the video-game's eye-tracking quality-check results across subs.
    :param qc_summary_dir: The path to the folder called ET_qc_summary which is the output folder of the
    ET_qc_manager.py script. This folder contains 1 folder per subject, in which there are all the results (plots and
    csv files) of the eye tracking quality checks that were performed. We assume that ALL subject folders within
    qc_summary_dir contain the EXACT same structure, so there is not one subject containing a QC result (file, e.g. csv
    or png) that others don't.
    :param save_path: path to which the results of the current analysis will be saved.
    """
    subjects = [d for d in os.listdir(qc_summary_dir) if os.path.isdir(os.path.join(qc_summary_dir, d))]
    subjects = filter_subject_folders(subjects)
    # check that all subjects are identical in their folder structure:
    valid_subs = list()
    for sub in subjects:
        if not check_dir_tree(os.path.join(qc_summary_dir, sub)):
            print(f"ERROR: SUBJECT {sub} DOES NOT HAVE THE EXPECTED FOLDER STRUCTURE OF ET QC OUTPUT: SKIPPING")
        else:
            valid_subs.append(sub)

    # create result folder
    stats_folder = create_folder(folder_path=os.path.join(save_path, "QC_STATS"))
    # average all relevant files in this level of nesting (subject)
    avg_file_list(file_list=files_to_avg, sub_list=valid_subs, qc_summary_dir=qc_summary_dir,
                  subject_sub_dir=None, save_path=stats_folder)

    # create sub folders
    examplar_sub = os.path.join(qc_summary_dir, valid_subs[0])
    for cond in list(all_conditions.keys()):
        create_folder(folder_path=os.path.join(stats_folder, cond))
        # find all the data-containing files within the condition folder
        cond_file_list = [f for f in os.listdir(os.path.join(examplar_sub, cond)) if f.endswith(CSV)]
        avg_file_list(file_list=cond_file_list, sub_list=valid_subs, qc_summary_dir=qc_summary_dir,
                      subject_sub_dir=cond, save_path=os.path.join(stats_folder, cond))

    ET_qc_plots(stats_folder, stats_folder)
    return

