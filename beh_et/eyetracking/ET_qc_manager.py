import os
import ET_data_extraction
import fnmatch
import pickle
import pandas as pd
import QualityChecker
import ET_data_processing
import for_matthias
from multiprocessing import Process

""" Quality Checks Management Module

This module manages everything related to the quality checks of the eye-tracking output data of experiment 2.
NOTE: The checks assumes XNAT-COMPATIBLE STRUCTURE: meaning, that "sub_dir" is a folder which includes one or more 
subject folders (e.g. "SZ104"). Each subject folder contains 2 sub-folders: /BEH, /ET (e.g., "SZ104/BEH"). 
Each data type (BEH/ET) contains the structure of the resource as it should've been uploaded to XNAT 
(e.g., "SZ104/BEH/SZ104/1"). 
See more here: 
https://twcf-arc.slab.com/posts/2-subject-visit-data-structure-and-naming-conventions-gkibcour

The QC creates a folder under "sub_dir" which is named "QC_summary", in which all the resulting plots and data tables 
are saved, per each subject separately (e.g., "sub_dir/QC_summary/SZ104"). In the result folder, you'll find:
- "QC_results_Epoch" / "QC_results_PreStim" / "QC_results_StimDuration" : 3 files with the same structure. All of them
contain summary information about the % of fixation on the stimulus/center, and the distance of gaze from stimulus/center, 
both across all trials and per trial type (Location, World etc), in a specific time-frame 
(pre-stimulus, stimulus duration, or the entire epoch - all of these windows are defined in "ET_param_manager.py" file). 
- A folder per condition type (stimulus category, game world, replay world, location, visibility), containing plots and 
their corresponding data tables. The plots are:
    - Heatmaps: of gaze distribution during each time-window type (Epoch, Stimulus duration, Pre-stim)
    - Line plots: of mean (and sem) of gaze distance from stimulus/center, across each time-window type.

@authors: RonyHirsch, AbdoSharaf98
"""

BEH = "BEH"
ET = "ET"
ET_RES_FOLD = "et"
FULL_LOGS = 'FullLogs'
FULLLOGNAME = "FullLogLevel"
CSV = 'csv'
ASC = 'asc'
COMPLETED = 'COMPLETED'
SUBLEN = 5
TEST_FOLDER = "test_inputs"
DMT_FOLDER = "QC"
V2 = "v2"

COGITATE_PATH = "/mnt/beegfs/XNAT/COGITATE"
RESOURCES = "RESOURCES"
FMRI = 'fMRI'
MR = "MR"
MEEG = 'MEEG'
MEG = 'MEG'
RAW = "Raw"
PHASE = "phase_2"
PROCESSED = "processed"
BIDS = "bids"
DERIV = "derivatives"
QCS = "qcs"
SUB_PREFIX = "sub-"
SES_V2 = "ses-v2"
PROJ = "projects"
MODALITY_MAPPING = {FMRI: MR, MEG: MEEG}
SESS_INFO_FILE = "_SessionInfo.txt"
BEH_QC_FILE_NAME = "quality_checks.csv"
BEH_QC_SUB_COL = "sub_code"
BEH_QC_VALID_COL = "Is_Valid?"


def is_sub(sub_code: str):
    """
    Checks whether a given string is a subject code
    :param sub_code: string of some name
    :return: True if input string is subcode, False if not
    """
    if len(sub_code) != SUBLEN:
        return False
    if not (sub_code[2:].isdigit()):
        return False
    if not (sub_code[0:2].isupper()):
        return False
    return True


def load_beh_data(beh_data_path):
    file_name = "subject_beh.pickle"
    beh_data_file = os.path.join(beh_data_path, file_name)
    if not os.path.exists(beh_data_file):
        print("ERROR: no behavioral data summary found. Run behavioral QC")
        return None
    fl = open(beh_data_file, 'rb')
    beh_data = pickle.load(fl)
    fl.close()
    print(f"Found behavioral data")
    return beh_data


def qc_ET(beh_data_path, straight_to_qc=False, root_folder=COGITATE_PATH, is_tau=False, include_invalid=False):
    """
    Given a path to a directory where one or more subject folders are, perform QC on each subject separately,
    Save the results and create a pickle to hold subject data for future analysis.
    :param sub_dir: path to where subject directories are.
    NOTE: this code assumes that under each subject folder the BEH and ET data are saved saprately, in accordance with
    the naming and structure conventions:
    https://twcf-arc.slab.com/posts/2-subject-visit-data-structure-and-naming-conventions-gkibcour#eye-tracking-files-experiment-2
    :param save_path: where to save the QC resulting data and figures
    """

    #qc_path = os.path.join(root_folder, DMT_FOLDER, V2, ET_RES_FOLD)
    qc_path = os.path.join(root_folder, DMT_FOLDER, V2, ET_RES_FOLD, "et_30_01_26")

    if not straight_to_qc:
        i = 0
        beh_data_all = load_beh_data(beh_data_path)
        mod_exclusion = dict()
        for mod in MODALITY_MAPPING.keys():
            mod_exclusion[mod] = list()
            mod_result_path = os.path.join(root_folder, mod, PHASE, PROCESSED, BIDS, DERIV, QCS)
            current_path = os.path.join(root_folder, mod, RAW, PROJ, f"CoG_{mod}_PhaseII")
            if not os.path.exists(current_path):
                print(f"No {mod} data found; moving on to next modality")
                continue
            for sub_dir in os.listdir(current_path):
                sub_path = os.path.join(current_path, sub_dir)
                sub_name = sub_dir
                print(f"------------sub_name: {sub_name}: #{i}---------------")
                i += 1

                if not os.path.isdir(sub_path):
                    continue
                if not is_tau:
                    sub_v2 = os.path.join(sub_path, f"{sub_name}_{MODALITY_MAPPING[mod]}_V2")
                else:
                    sub_v2 = sub_path
                if os.path.isdir(sub_v2):
                    if not is_tau:
                        resources_dir = os.path.join(sub_v2, RESOURCES)
                    else:
                        resources_dir = sub_path
                    if os.path.isdir(resources_dir):
                        if not is_tau:
                            et_path = os.path.join(resources_dir, ET, sub_name)
                        else:
                            et_path = sub_path
                        if not os.path.exists(et_path):
                            print(f'Sub {sub_name}: No ET data! Skipped')
                            continue
                        et_sessions = [f for f in os.listdir(et_path) if os.path.isdir(os.path.join(et_path, f)) and f.isdigit()]
                        if len(et_sessions) == 0:
                            print(f'Sub {sub_name}: No ET data! Skipped')
                            continue
                        relevant_sess = max(et_sessions)
                        if not is_tau:
                            sess_path = os.path.join(et_path, relevant_sess)
                            # the session info is where the screen pixel dimensions are
                            session_info_path = os.path.join(resources_dir, BEH, sub_name, relevant_sess, f"{sub_name}{relevant_sess}{SESS_INFO_FILE}")
                        else:
                            sess_path = os.path.join(et_path, relevant_sess, "FullLogs")
                            session_info_path = os.path.join(resources_dir, relevant_sess, f"{sub_name}{relevant_sess}{SESS_INFO_FILE}")
                        # get a list of asc files
                        ascFiles = [fl for fl in os.listdir(sess_path) if os.path.isfile(os.path.join(sess_path, fl)) and fl.endswith(ASC)]
                        if len(ascFiles) == 0:
                            print(f'-------- Sub {sub_name}: No ascii files found! Skipped -------- Sub ')
                            continue
                        ascFiles = sorted(ascFiles)
                        # subject exists and has all data; create a result folder for the ET qc
                        sub_result_path = os.path.join(mod_result_path, f"{SUB_PREFIX}{sub_name}", SES_V2, ET_RES_FOLD)
                        if not is_tau:
                            if not os.path.exists(sub_result_path):
                                os.mkdir(sub_result_path)
                        else:
                            if not os.path.exists(sub_result_path):
                                os.makedirs(sub_result_path)
                        # get subject's behavioral data
                        sub_beh_data = beh_data_all[sub_name]
                        # get subject's behavioral QC in order to see whether they were excluded based on their behavior
                        beh_data_qc_path = os.path.join(beh_data_path, BEH_QC_FILE_NAME)
                        beh_data_qc_file = pd.read_csv(beh_data_qc_path)
                        sub_beh_data_qc = beh_data_qc_file[beh_data_qc_file[BEH_QC_SUB_COL] == sub_name][BEH_QC_VALID_COL]
                        if not(sub_beh_data_qc.tolist()[0]):
                            print(f'-------- Sub {sub_name}: BEH QC failed!! Not a valid subject! Skipped --------')
                            mod_exclusion[mod].append(sub_name)
                            if not include_invalid:
                                continue
                        p = Process(target=ET_data_extraction.extract_data,
                                    args=(sub_name, relevant_sess, sess_path, ascFiles, sub_result_path, session_info_path, sub_beh_data))
                        p.start()  # start the process of QC for this subject
                        p.join()  # this blocks until the process terminates, no new sub will run until the previous sub ended
                        if p.exitcode > 0:  # if the process exited with an error
                            exit(1)  # exit with an error as well
                        #raise Exception("Just so one subject only runs") # TODO:REMOVE

            print(f"{len(mod_exclusion[mod])} subjects were skipped as they were previously excluded in behavioral quality-checks")

    print(f'\n-------- Data Extraction Complete ----------\n')
    #raise Exception("Remove me, just so no QC runs")  # TODO: REMOVE ONCE FINISHING ADJUSTMENTS
    subs = {FMRI: list(), MEG: list()}
    unprocessed_subs = list()  # This is a list of subjects whose ET data IS INVALID so a pickle wasn't even generated for analysis
    unprocessed_subs_reason = list()
    processed_subs = list()
    beh_data_all = load_beh_data(beh_data_path)

    for mod in MODALITY_MAPPING.keys():
        mod_result_path = os.path.join(root_folder, mod, PHASE, PROCESSED, BIDS, DERIV, QCS)
        if not os.path.exists(mod_result_path):
            print(f"No {mod} data found; moving on to next modality")
            continue
        for sub_dir in os.listdir(mod_result_path):  # iterate over PROCESSED data
            sub_result_path = os.path.join(mod_result_path, sub_dir, SES_V2, ET_RES_FOLD)
            if SUB_PREFIX not in sub_dir:
                continue
            sub_name = sub_dir.replace(SUB_PREFIX, '')

            # get subject's behavioral QC in order to see whether they were excluded based on their behavior
            beh_data_qc_path = os.path.join(beh_data_path, BEH_QC_FILE_NAME)
            beh_data_qc_file = pd.read_csv(beh_data_qc_path)
            sub_beh_data_qc = beh_data_qc_file[beh_data_qc_file[BEH_QC_SUB_COL] == sub_name][BEH_QC_VALID_COL]
            if sub_beh_data_qc.empty or (not (sub_beh_data_qc.tolist()[0]) and not include_invalid):
                print(f'-------- Sub {sub_name}: BEH QC failed!! Not a valid subject! Skipped --------')
                #unprocessed_subs.append(sub_name)
                #unprocessed_subs_reason.append("Not valid subject (BEH)")
                continue

            if not os.path.exists(sub_result_path):
                print(f"{sub_name} has no et folder")
                unprocessed_subs.append(sub_name)
                unprocessed_subs_reason.append("ET analysis unsuccessful???")
                continue

            pick_file = [f for f in os.listdir(sub_result_path) if
                         fnmatch.fnmatch(f, f"{sub_name}EyeTrackingDataNew30-01-26.pickle")]
            if len(pick_file) == 0:
                print(f"{sub_name} has no ET pickle file")
                unprocessed_subs.append(sub_name)
                unprocessed_subs_reason.append("ET analysis unsuccessful")
                continue

            processed_subs.append(sub_name)
            print(f"Found subject {sub_name} saved data. Loading now...")
            pick_path = os.path.join(sub_result_path, pick_file[0])
            subs[mod].append(pick_path)

    # save into a file all the subjects that were NOT PROCESSED AT ALL as their ET was too problematic
    unprocessed_subs_df = pd.DataFrame({"subCode": unprocessed_subs, "reason": unprocessed_subs_reason})
    unprocessed_subs_df.to_csv(os.path.join(qc_path, "et_invalid_subs.csv"), index=False)

    processed_subs_df = pd.DataFrame({"subCode": processed_subs})
    processed_subs_df.to_csv(os.path.join(qc_path, f"et_valid_subs.csv"), index=False)

    # now we have all the existing subjects in all modalities
    p = Process(target=ET_data_processing.process_data, args=(subs, beh_data_path, qc_path, False, include_invalid))
    #p = Process(target=for_matthias.process_data, args=(subs, beh_data_path, qc_path, False))
    p.start()
    p.join()
    if p.exitcode > 0:  # if the process exited with an error
        exit(1)  # exit with an error as well
    """
    OLD QC
    for mod in MODALITY_MAPPING.keys():
        mod_result_path = os.path.join(root_folder, mod, PHASE, PROCESSED, BIDS, DERIV, QCS)
        if not os.path.exists(mod_result_path):
            print(f"No {mod} data found; moving on to next modality")
            continue
        for sub_dir in os.listdir(mod_result_path):  # iterate over PROCESSED data
            sub_result_path = os.path.join(mod_result_path, sub_dir, SES_V2, ET_RES_FOLD)
            sub_name = sub_dir.replace(SUB_PREFIX, '')
            pick_file = [f for f in os.listdir(sub_result_path) if fnmatch.fnmatch(f, f"{sub_name}EyeTrackingData.pickle")]
            if len(pick_file) == 0:
                print(f"{sub_name} has no ET pickle file; please run the pre-processing first")
                continue
            print(f"Found subject {sub_name} saved data. Loading now...")
            pick_path = os.path.join(sub_result_path, pick_file[0])
            # again, clear RAM between subjects
            p = Process(target=QualityChecker.check, args=(pick_path, sub_name, sub_result_path))
            p.start()
            p.join()
            if p.exitcode > 0:  # if the process exited with an error
                exit(1)  # exit with an error as well
    """

    return


if __name__ == '__main__':
    #qc_ET(beh_data_path=r"/mnt/beegfs/XNAT/COGITATE/QC/v2/TAU/QC/v2/beh",
    #      straight_to_qc=True, root_folder=r"/mnt/beegfs/XNAT/COGITATE/QC/v2/TAU", is_tau=True)
    print("I am here")
    qc_ET(beh_data_path=r"/mnt/beegfs/XNAT/COGITATE/QC/v2/beh",
          straight_to_qc=True, root_folder=r"/mnt/beegfs/XNAT/COGITATE", include_invalid=False)
