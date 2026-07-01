import os

"""
This module includes all the data-saving related methods. It creates the directories where the data will be saved 
and has methods for saving files into the appropriate folders. Every module which saves analysis / 
QC data uses this module. 

@author: RonyHirsch
"""

QC_NAME = 'ET_QC_summary'
FIX_NAME = 'fixation'
FIX_DENS = 'density'
STAB = 'stability'
SACC_NAME = 'saccade'
BLINK_NAME = 'blink'
PUPIL_NAME = "pupil"
ANALYSIS_NAME = 'ET_analysis'

# ET QC conditions (QC folders)
LOC = 'location'
VIS = 'visibility'
WORLD_G = 'game_world'
WORLD_R = 'replay_world'
WORLD = 'world'
WORLD_ID = "WorldID"
CATEGORY = 'category'
ALL = 'all'
LEVELS = [LOC, VIS, WORLD, CATEGORY, ALL]  # WORLD_G, WORLD_R

# ET QC duration windows
STIM_DUR = 'stim_duration'
PRE_STIM_DUR = 'pre_stim'
EPOCH = 'epoch'
TIMES = [STIM_DUR, PRE_STIM_DUR, EPOCH]


def create_dir(d):
    """
    This function creates a directory, given a specific path.
    If the directory didn't exist, the function creates it. If it cannot be created,an error message is printed.
    If it already exists, the function does nothing.
    :param d: directory path
    """
    if not (os.path.isdir(d)):
        try:
            os.mkdir(d)
        except Exception as e:
            warning_msg = ' \nCould not create directory ' + str(d)
            print(warning_msg)
            raise e
        else:
            print('Created directory ' + str(d))
    return


def create_subject_analysis(subs_path):
    folder_path = os.path.join(subs_path, ANALYSIS_NAME)
    create_dir(folder_path)
    # fixation
    fix_folder_path = os.path.join(folder_path, FIX_NAME)
    create_dir(fix_folder_path)
    for sf in [FIX_DENS, STAB]:  # sub folder structure
        fp = os.path.join(fix_folder_path, sf)
        create_dir(fp)

    # saccade
    sacc_folder_path = os.path.join(folder_path, SACC_NAME)
    create_dir(sacc_folder_path)

    #blink
    blink_folder_path = os.path.join(folder_path, BLINK_NAME)
    create_dir(blink_folder_path)

    # pupil
    pupil_folder_path = os.path.join(folder_path, PUPIL_NAME)
    create_dir(pupil_folder_path)

    return folder_path


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
    err = None
    if len(existing_files) == 0:
        print(f"No files in directory {folder_path}")
        return
    for f in existing_files:
        if f == file_name:
            is_duplicate = 1
            err = f
    if is_duplicate == 1:
        print(f"File {err} already exists in {folder_path}. Save action is replacing the old file with a new one.")
        # remove the old file
        os.remove(os.path.join(folder_path, file_name))
    else:
        print(f"File {f} was not found in {folder_path}. Save action will not override anything.")
    return
