import pandas as pd
import numpy as np
import os
import json

"""
This module uploads subject data and organizes it for future checks and analyses.
The "read_data" function expects to receive a path to a directory containing subject folders in the exp.2 structure.
Each subject directory is then read (and verified) by the "read_sub_data" function, which expects to receive a path
to a subject folder that contains session folders. From session, the extracted data are the "Details" files (within
the "Details" folder) and the Analyzer output for this session - which is expected to be nested right under the session
folder. This information is then used to instantiate class instance Subject, which contains instances of class Session, 
and within each session the Details file class and the analyzer output dataframe.

@author: RonyHirsch
"""

" Constants of the structure of exp.2 : assumptions on subject code names, session code names and file names:"
BEH = "BEH"
SUBLEN = 5
DETAILS = "Details"
SEQ = "Sequence"
STIM_TYPE_MAP = {"Object": "SO", "Face": "SF", 'None': 'None'}
PRESCREEN = "A"
PRESCREEN_ALT = "B"
STIMANALYZER = "AnalyzerOutput"
FULL_LOG_FOLDER = "FullLogs"
FULL_LOG_LEVELMARKER = "_L_"
STIM_ANALYZER_SUFFIX = "Stimulus.csv"
FILLERD = "FillerDetails"
LOCALIZERD = "LocalizerDetails"
PROBED = "ProbeDetails"
STIMD = "StimulusDetails"
SUBD = "SubjectDetails"
# column names as they appear in the raw exp.2 output
# fillerDetails
FILLER_ONSET_TS = "fillerOnsetTS"
FILLER_ONSET_TS_NP = "fillerOnsetTS_NoPauses"
DURING_SAFE_END = "duringSafeEndOfGame"
STIM_LOC = "LOCALIZER_STIMULUS"
STIM_GAME = "GAME_STIMULUS/PROBE"
CURR_LEVEL_ID = "currentLevelID"

# localizerDetails
STIM_ONSET_TS = "stimulusOnsetTS"
STIM_ONSET_TS_NP = "stimulusOnsetTS_NoPauses"
TYPE_ = "type"
EVAL = 'responseEvaluation'
TRUENEGATIVE = 'TrueNegative'
STIM_NO_RESP = "StimNoResponse"
BLANK_NO_RESP = "BlankNoResponse"
RESP = "response"

# localizer sequence
LOC_LEVEL_TITLE = "LevelToReplayID_1Based"

# Paths and dicts for HPC
COGITATE_PATH = "/mnt/beegfs/XNAT/COGITATE"
RESOURCES = "RESOURCES"
FMRI = 'fMRI'
MR = "MR"
MEEG = 'MEEG'
MEG = 'MEG'
ECOG = 'ECOG'
RAW = "Raw"
PROJ = "projects"
MODALITY = "modality"
STIM_TYPE = "stimulusType"
MODALITY_MAPPING = {FMRI: MR, MEG: MEEG}
# module per lab : taken from https://twcf-arc.slab.com/posts/institutional-abbreviations-rsi4obcd
METHOD = {'SA': MEEG, 'SB': MEEG, 'SC': FMRI, 'SD': FMRI, 'SE': ECOG, 'SF': ECOG, 'SG': ECOG, 'SX': ECOG, 'SZ': MEEG}
# replay level dict: the file headers for the full logs of replay levels is _L_x (1<x<8) and their name is 100x
REPLAY_LEVEL_MAPPING = {f"{i}": str(100+(i-1)) for i in range(1, 9)}
REPLAY_LEVEL_ORDER = [str(100+(i-1)) for i in range(1, 9)]


class SessDetails:
    """
    A single session's "Details" folder content. Contains details-files that are loaded here to dataframes.
    """
    def __init__(self, path):
        self.LocalizerDetWithFillers = None
        self.LocalizerDetCalculated = None  # This is the final localizer dataframe used for analysis of replay responses : everything that is excluded was removed
        self.LocalizerDetFullCorrected = None  # This is for ET analyses purposes: this includes all the lines originally in localizer data, with corrected responses, but nothing is removed - instead it is annotated in a column
        details_files = [x for x in os.listdir(path) if x.endswith('csv')]
        for f in details_files:
            if FILLERD in f:
                self.FillerDet = pd.read_csv(os.path.join(path, f), sep=';', na_values="")
                self.FillerDet.rename(columns={FILLER_ONSET_TS: STIM_ONSET_TS, FILLER_ONSET_TS_NP: STIM_ONSET_TS_NP}, inplace=True)
            elif LOCALIZERD in f:
                self.LocalizerDet = pd.read_csv(os.path.join(path, f), sep=';', na_values="")  # original localizer details
            elif PROBED in f:
                self.ProbeDet = pd.read_csv(os.path.join(path, f), sep=';', na_values="")
            elif STIMD in f:
                self.StimulusDet = pd.read_csv(os.path.join(path, f), sep=';', na_values="")
            elif SUBD in f:
                self.SubjectDet = pd.read_csv(os.path.join(path, f), sep=';', na_values="")
        self.update_game()
        self.update_replay()

    def update_game(self):
        """
        In the fMRI modality, the game (dAT) did not wait for the subject response; instead, if 3 seconds have passed and no response was made -
        the game automatically continued, and the trial was logged as "NoResponse" instead of response = No) and "StimNoResponse" under responseEvaluation.
        Notably, this happened only in the fMRI modality, as in the others the game stopped and waited for a response with no time limit.
        However, to fully interpret these incidents, we now convert the timeout trials to either false negative (if there was a face or an object) or true negative (if it was a blank)
        """
        self.ProbeDet.loc[self.ProbeDet[EVAL] == STIM_NO_RESP, EVAL] = "FalseNegative"
        self.ProbeDet.loc[self.ProbeDet[EVAL] == BLANK_NO_RESP, EVAL] = "TrueNegative"
        self.ProbeDet.loc[self.ProbeDet[EVAL] == STIM_NO_RESP, RESP] = "No"
        self.ProbeDet.loc[self.ProbeDet[EVAL] == BLANK_NO_RESP, RESP] = "No"
        
        pass

    def update_replay(self):
        """
        Once all "details" files have been loaded, create a version of the localizerDetails that incorporates all the
        localizer (replay) fillers, as they appear in fillerDetails.
        :return:
        """
        additional_cols = list(self.LocalizerDet.columns.values)
        type_ind = additional_cols.index(TYPE_)
        additional_cols = [additional_cols[i] for i in range(type_ind+1, len(additional_cols))]
        self.LocalizerDet[DURING_SAFE_END] = False
        filler_det_copy = self.FillerDet.copy()
        filler_last_col = filler_det_copy[DURING_SAFE_END]
        filler_det_copy.drop(DURING_SAFE_END, axis=1, inplace=True)
        for col_name in additional_cols:
            filler_det_copy[col_name] = np.nan
        filler_det_copy[DURING_SAFE_END] = filler_last_col
        filler_det_copy = filler_det_copy[filler_det_copy[CURR_LEVEL_ID] != CURR_LEVEL_ID]  # header row for some reason
        filler_det_copy[CURR_LEVEL_ID] = filler_det_copy[CURR_LEVEL_ID].astype(int)
        loc_filler_det_copy = filler_det_copy[filler_det_copy[CURR_LEVEL_ID] >= 100]
        loc_filler_det_copy = loc_filler_det_copy[loc_filler_det_copy[TYPE_] != STIM_LOC]  # just fillers
        self.LocalizerDetWithFillers = pd.concat([self.LocalizerDet, loc_filler_det_copy])
        # remove "false" localizers which occur when the level is ending
        self.LocalizerDetWithFillers = self.LocalizerDetWithFillers[self.LocalizerDetWithFillers[DURING_SAFE_END] == False]
        self.LocalizerDetWithFillers.sort_values(by=[STIM_ONSET_TS], inplace=True)
        # All "nan"s in the responseEvaluation column are filler lines (other lines are either stimulus lines OR outsideWindowPresses lines
        # thus, they are all "correct rejections" as there is no stimulus there and no responses (for the moment)
        self.LocalizerDetWithFillers[[EVAL]] = self.LocalizerDetWithFillers[[EVAL]].fillna(value=TRUENEGATIVE)
        return


def helper_stim_name(row):
    stim_type = row["Type"]
    stim_num = row["ID"]
    if not np.isnan(stim_num):  # nan = blank stim
        stimName = STIM_TYPE_MAP[stim_type] + str(int(stim_num)).zfill(2)
    else:
        stimName = STIM_TYPE_MAP[stim_type]
    return stimName


class Session:
    """
    A single session belonging to a single subject.
    A session can be either a pre-screening session, or a full-game session.
    The class contains basic information about the session, and all the relevant data this session has.
    """
    def __init__(self, path, name, stim_analysis_path=""):
        """
        :param path: the full path to the session folder
        :param name: the session name
        :param stim_analysis_path: the path to the "StimulusAnalysis.csv" file which is an Analyzer software output.
        """
        self.path = path
        self.name = name
        self.config = self.get_config(path)
        self.method = self.get_method(path)
        self.replay_targets = self.get_replay_targets(path)
        self.SessDetails = SessDetails(os.path.join(path, DETAILS))
        self.StimSeq = self.get_stim_seq(path)
        # self.GameRaw, self.ReplayRaw = self.get_full_logs(path)
        # self.ReplayRaw = self.get_replay_raw(path)
        if stim_analysis_path and os.path.isdir(stim_analysis_path):
            self.analyzer_output = self.get_stim_analysis(stim_analysis_path)
        else:
            self.analyzer_output = None

    def get_full_logs(self, path):
        game_levels = self.get_game_raw(path)
        replay_levels = self.get_replay_raw(path)
        return game_levels, replay_levels

    def get_game_raw(self, path):
        game_dict = dict()
        full_logs_path = os.path.join(path, FULL_LOG_FOLDER)
        sub_game = [x for x in os.listdir(full_logs_path) if x.endswith('.csv') and FULL_LOG_LEVELMARKER not in x]  # just game levels
        for game_level in sub_game:
            game_level_name = (int(game_level.split("_")[2]) - 1) * 4 + int(game_level.split("_")[3])
            if game_level_name < 0:  # practice levels (i.e., world 0) are not game levels
                continue
            replay_level = pd.read_csv(os.path.join(full_logs_path, game_level), sep='^', header=None)
            game_dict[game_level_name] = replay_level
        return game_dict

    def get_replay_raw(self, path):
        replay_dict = dict()
        full_logs_path = os.path.join(path, FULL_LOG_FOLDER)
        sub_replays = [x for x in os.listdir(full_logs_path) if x.endswith('.csv') and FULL_LOG_LEVELMARKER in x]
        for replay in sub_replays:
            replay_level_name = replay.split("_")[3]  # subcode_FullLogLevel_L_NUMBER ==> NUMBER is [3], and replay ensured in "sub_replays"
            replay_level_log_name = REPLAY_LEVEL_MAPPING[replay_level_name]
            replay_level = pd.read_csv(os.path.join(full_logs_path, replay), sep='^', header=None)
            replay_dict[replay_level_log_name] = replay_level
        return replay_dict

    def get_config(self, path):
        sub_config = [x for x in os.listdir(path) if x.endswith('config.json')]
        print(os.path.join(path, sub_config[0]))
        with open(os.path.join(path, sub_config[0]), encoding="utf-8") as json_file:
            config = json.load(json_file)
        return config

    def get_method(self, path):
        """
        Extract the method/module this session was recorded in : fMRI/MEEG
        :param path: path to the relevant session folder, from which we'll extract the relevant .txt file
        :return: the method
        """
        sub_info = [x for x in os.listdir(path) if x.endswith('_module.txt')]
        sub_method = open(os.path.join(path, sub_info[0]), 'r').read()
        return sub_method

    def get_replay_targets(self, path):
        """
        Extract the replay level target stimuli for this session
        :param path: path to the relevant session folder, from which we'll extract the relevant .txt file
        :return:
        """
        sub_seq = os.path.join(path, SEQ, "Localizers_Corrected.csv")
        if not os.path.exists(sub_seq):  # backward compatibility
            sub_seq = os.path.join(path, SEQ, "Localizers.csv")
        try:
            sub_targets = pd.read_csv(sub_seq, sep=';')
        except pd.errors.ParserError:  # the session was at some point interrupted and the write to this file was err
            sub_targets = pd.read_csv(sub_seq, sep=';', error_bad_lines=False)  # skip the double-header line
            sub_targets.drop_duplicates(keep='last', inplace=True)
            sub_targets.reset_index(drop=True, inplace=True)
        return sub_targets

    def get_stim_seq(self, path):
        """
        Extract from the stimSequence file of the subject all the (unique) stimulus IDs the subject was presented with.
        This will be used for analyses that look at the data per stimulusID.
        :param path: path to the relevant session folder, from which we'll extract the sequence
        :return:
        """
        sub_seq_stim = os.path.join(path, SEQ, "StimSequence.csv")
        sub_stim = pd.read_csv(sub_seq_stim, sep=';')
        sub_stim["stimulusName"] = sub_stim.apply(lambda row: helper_stim_name(row), axis=1)
        return sub_stim

    def get_stim_analysis(self, path):
        stim_analysis_files = [f for f in os.listdir(path) if f.endswith(STIM_ANALYZER_SUFFIX)]
        stim_analysis_csvs = [pd.read_csv(os.path.join(path, f), sep=';') for f in stim_analysis_files]
        stim_analysis_csv = pd.concat(stim_analysis_csvs)
        return stim_analysis_csv


class Subject:
    """
    A single subjct class, contains all of this subject's session information files and raw data.
    """
    def __init__(self, path, full_run_id="", prescreen_id=""):
        """
        :param path: the full path to the subject folder
        :param full_run_id: the name that was given to this subject's full run session (if there was one)
        :param prescreen_id: the name that was given to this subject's behavioral sreening session (if there was one)
        """
        self.path = path
        self.lab = self.sub_code(path)[0:2]
        self.id = self.sub_code(path)[2:]
        self.mod = METHOD[self.lab]  # modality
        self.add_session(full_run_id, prescreen_id)

    def add_session(self, full_run_id="", prescreen_id=""):
        if prescreen_id:
            self.prescreen = Session(path=os.path.join(self.path, prescreen_id), name=prescreen_id,
                                     stim_analysis_path=os.path.join(self.path, prescreen_id, STIMANALYZER))
        if full_run_id:
            self.full = Session(path=os.path.join(self.path, full_run_id), name=full_run_id,
                                stim_analysis_path=os.path.join(self.path, full_run_id, STIMANALYZER))

    def sub_code(self, path):
        """
        extracts the subject code from the full path
        :param path: the full path to the subject folder
        :return: the full subject code (lab id + sub id)
        """
        npath = os.path.normpath(path)
        split = npath.split(os.sep)
        return split[-1]


def read_sub_data(sub_path):
    """
    given a single subject folder, create an instance of class Subject and initialize its contents with raw subject data
    :param sub_path: the full path to the subject folder
    :return: a Subject class instance containing the subject's info
    """
    sub_sessions = [x for x in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, x))]
    sub_full_run_ids = [x for x in sub_sessions if x.isdigit()]
    sub = Subject(sub_path)
    if len(sub_full_run_ids) > 0:
        sub_full_run = str(max([int(x) for x in sub_full_run_ids]))
        if len(os.listdir(os.path.join(sub_path, sub_full_run))) == 0 or not(os.path.isdir(os.path.join(sub_path, sub_full_run, DETAILS))):
            print(f"empty EXPERIMENTAL session (no data): {os.path.join(sub_path, sub_full_run)}: not added to subject struct")
        else:
            sub.add_session(full_run_id=sub_full_run)
    if PRESCREEN in sub_sessions:
        if len(os.listdir(os.path.join(sub_path, PRESCREEN))) != 0:  # only if the session directory is not empty
            sub.add_session(prescreen_id=PRESCREEN)
        else:
            print(f"empty session (no data): {os.path.join(sub_path, PRESCREEN)}: not added to subject struct")
    if not(PRESCREEN in sub_sessions) and PRESCREEN_ALT in sub_sessions:
        if len(os.listdir(os.path.join(sub_path, PRESCREEN_ALT))) != 0:
            sub.add_session(prescreen_id=PRESCREEN_ALT)
        else:
            print(f"empty session (no data): {os.path.join(sub_path, PRESCREEN_ALT)}: not added to subject struct")
    return sub


def is_subject(s):
    """
    given a string s which is a folder name, check if it represents a subject folder or not
    :param s: folder name
    :return: whether this is a subject or not
    """
    if len(s) != SUBLEN:
        return False
    if not (s[2:].isdigit()):
        return False
    if not (s[0:2].isupper()):
        return False
    return True


def read_data(data_path, valid_subs=list()):
    """
    given a folder filled with exp.2 subject folders, this function goes over all of them and creates a dictionary
    in which key=sub code, val=Subject class instance which contains all the subject information and raw data.
    :param data_path: path where all the subject data files reside
    :param valid_subs: If this parameter is given, these are the names of the only subject we want to analyze. The
    valid suject list will be further filtered, to include only subjects that are in the valid_subs list.
    :return:
    """
    subjects = [x for x in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, x)) and is_subject(x)]
    if len(valid_subs) > 0:  # if we gave specific subjects to load, we'll only load them
        subjects = [s for s in subjects if s in valid_subs]
    subjects_dict = dict()
    for sub in subjects:
        beh_path = os.path.join(os.path.join(data_path, sub), BEH, sub)
        subjects_dict[sub] = read_sub_data(beh_path)
    return subjects_dict


def data_reader_hpc(root_folder=COGITATE_PATH):
    """
    Iterates throughout the raw data folder structure to reach eac subject across all modalities.
    For each subject, it loads its entire behavioral data by calling "read_sub_data"
    :param root_folder:
    :return: a dictionary where key=subject, value=an instance of Subject class with all the behavioral information of that subject
    """
    subjects_dict = dict()
    for mod in MODALITY_MAPPING.keys():
        current_path = os.path.join(root_folder, mod, RAW, PROJ, f"CoG_{mod}_PhaseII")
        for sub_dir in os.listdir(current_path):
            sub_path = os.path.join(current_path, sub_dir)
            sub_name = sub_dir
            if not os.path.isdir(sub_path):
                continue
            sub_v2 = os.path.join(sub_path, f"{sub_name}_{MODALITY_MAPPING[mod]}_V2")
            if os.path.isdir(sub_v2):
                resources_dir = os.path.join(sub_v2, RESOURCES)
                if os.path.isdir(resources_dir):
                    beh_path = os.path.join(resources_dir, BEH, sub_name)
                    subjects_dict[sub_name] = read_sub_data(beh_path)
    return subjects_dict
