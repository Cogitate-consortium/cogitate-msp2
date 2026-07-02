#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 10.07.2020
@author: David Richter (d.richter@donders.ru.nl)

Modified 17.08.2023
@author: Yamil Vidal (hvidaldossantos@gmail.com)

Modified 29.04.2024
@author: George Blackburne

Modified 04.06.2024
@author: Zvi Roth

Modified 25.06.2024
@author: George Blackburne

Extracts events relevant for fMRI analysis from various log files of exp.2.
Performs tests on log files.
Outputs MRI specific tsv log file (bids compliant) per run and various event files for 1st level GLMs.

Inputs:
    - log files from exp.2
    - path to dicoms from exp.2 (optional)

Outputs:
    - one log file checks table with warning flags per run (see: logChecks function)
    - one MRI specific log file (tsv) per run with all information relevant for analysis (bids compliant) (see: createMriLog function)
    - 3 column event txt files for use in fMRI analysis per regressor (timestamp, ev duation, parametric modulator) (see: createEvFile_* functions)

Tested on python v3.7.4, pandas v0.25.2, numpy v1.17.2
"""

# TO DO:
# improve bids compliance of MRI specific log file and add description of columns to json file
# add handling of restarts (I would need restarted log files for this) or do we handle this on the file input/upload side before?
# @Aya: you still need to add Yale specific adjustments for using the high accu input, instead of the serial dump, to get MRI trigger timestamps. On our end the high accu input does not log the MR triggers, so i could not implement or test this (no data).
# adjust to folder/naming convention used on cogitate hpc (when do we know these?)
# double check all event files


# %% Imports & parameters
import pandas as pd
import numpy as np
from glob import glob
import re, os

pd.options.mode.chained_assignment = None  # default='warn'

##### Options #####
# The following options need to be adjusted depending on the MRI scanner/recording side, assuming data is stored in a bids complient fashion in the project root path (/bids)
# 1. MRI and response device settings
# 2. project root path
# (3.) logPathPattern
# 4. subject list (non-bids)
# (5.) session prefixes (new session ID conventions were added on slab, different from pilot data conventions: prescreen=A,practice=B,mri-full-game=F)

# Option to only perform log file checks, without creating MRI specific logs or event files
onlyRunChecks = 0  # set to true (1) to only check log files (no event files are created). Set to false (0) for preparation of real data analysis

# if true runs full check of log files. If false (0) does not perform checks depending on full log files with all MRI triggers; setting this to false is useful for checking log files acquired for testing purposes without running the scanner
logsHaveMriTriggers = 1  # should be TRUE (1) for real data

##### Parameters #####

# session prefixes
prescreenID = 'A'  # session ID letter for prescreening session (outside of scanner)
practiceID = 'B'  # session ID letter for practice session (outside of scanner)
scannerID = '1'  # session ID letter for MRI scanning session

# run design params/labels (should remain untouched)
runOnsetLevels = np.concatenate((np.arange(1, 16, 2), np.arange(100, 108,
                                                                2)))  # numbers of levels starting a run (e.g. levels 1-2 are combined to form run 1, hence level 1 is runOnsetLevel for run1, etc. Replay levels start at lvl id 100)
runByLevelAndWorldCode = ['1_1', '1_3', '2_1', '2_3', '3_1', '3_3', '4_1', '4_3', 'L_1', 'L_3', 'L_5',
                          'L_7']  # labels of world_level combinations corresponding to the runOnset levels (run numbers), as used in full logs
complementaryLevelAndWorldCode = ['1_2', '1_4', '2_2', '2_4', '3_2', '3_4', '4_2', '4_4', 'L_2', 'L_4', 'L_6',
                                  'L_8']  # labels of world_level combinations corresponding to the second level in a run
runLabelsMRIlog = [1, 2, 3, 4, 5, 6, 7, 8, 51, 52, 53, 54]  # run labels used in the extraLogs -> FMRI log

# stimulus & response parameters (relevant primarily for creating event files)
stimulusDuration = 0.25  # duration of stimulus events in seconds
nullEventDuration = 12  # duration of the first null events in seconds
shortNullEventDuration = 1  # duration of the second null events in seconds
relevantLocations = ['Left',
                     'Right']  # relevant stimulus locations (i.e. top left and bottom left will be group if left is a relevant category)
probeResponseTypes = ['TruePositive', 'TrueNegative', 'FalseNegative', 'FalsePositive', 'StimNoResponse',
                      'BlankNoResponse']  # possible response types and mapped labels used for event files

probeResponseLabels = {
    "TruePositive": "Seen",
    "FalseNegative": "Unseen",
    "TrueNegative": "Correctreject",
    "FalsePositive": "Falsealarm",
    "StimNoResponse": "NoResponse",
    "BlankNoResponse": "NoResponse"}

replayResponseLabels = {
    "TruePositive": "Hit",
    "FalseNegative": "Miss",
    "TrueNegative": "Correctreject",
    "FalsePositive": "Falsealarm",
    "StimNoResponse": "NoResponse",
    "BlankNoResponse": "NoResponse"}

# MR trigger params
dummyTriggers = 3  # number of dummy triggers to be discarded (i.e. MRI volumes collected before run onset)
TR = 1.5  # to be used to correct the timing of FSL three column format event files

# MRI and response device settings. These are used in extracting data from serial dump files (these may need to be adjusted depending on the MRI setup)
MRItriggerPort = 'COM3'  # device port (e.g. com port) of MRI triggers
MRItriggerKey = 'a'  # MRI trigger key
ButtonPressPort = 'COM2'  # device port (e.g. com port) of button box

##### Paths #####

# project root path; assumed to contains raw folder and bids folder (following bids specification)
projectRoot = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed'

# output paths
outputLogFilePattern = projectRoot + '/bids/sub-%(sub)s/ses-V2/func/sub-%(sub)s_ses-V2_%(runType)s_events.tsv'
outputErrorFlagPattern = projectRoot + '/bids/derivatives/logfilechecks/ses-V2/sub-%(sub)s_ses-%(ses)s_errorFlags.csv'
outputEventFilePattern = projectRoot + '/bids/derivatives/regressoreventfiles/sub-%(sub)s/ses-V2/sub-%(sub)s_ses-V2_%(runType)s_%(eventType)s_EV.txt'

# input log file paths & file names
logPathPattern_s = '/mnt/beegfs/XNAT/COGITATE/fMRI/Raw/projects/CoG_fMRI_PhaseII/%(sub)s/%(sub)s_MR_V2/RESOURCES/BEH/%(sub)s/'
logPathPattern = '/mnt/beegfs/XNAT/COGITATE/fMRI/Raw/projects/CoG_fMRI_PhaseII/%(sub)s/%(sub)s_MR_V2/RESOURCES/BEH/%(sub)s/%(scannerID)s/'  # path to log files
logProbes = 'Details/%(sub)s%(scannerID)s_ProbeDetails.csv'  # probe details log file for Video Game levels
logReplay = 'Details/%(sub)s%(scannerID)s_LocalizerDetails.csv'  # probe details log file for Replay levels
logStimuli = 'Details/%(sub)s%(scannerID)s_StimulusDetails.csv'  # stimulus details log file
logFull = 'FullLogs/%(sub)s%(scannerID)s_FullLogLevel_'  # full log file (incl. subpaths)
logMRI = 'ExtraLogs/%(sub)s%(scannerID)s_FMRI.txt'  # fmri specific log file (incl. subpaths)

logTriggers_donders = 'ExtraLogs/%(sub)s%(scannerID)s_Serial.txt'  # serial dump used at the Donders Institute (incl. subpaths)
logTriggers_yale = 'ExtraLogs/%(sub)s%(scannerID)s_HighAccuInput.txt'  # Trigger logger used at Yale (incl. subpaths)

# subject list (determines on which subjects scripts are run)
# subject_list = projectRoot + '/bids/derivatives/qcs/ses-v2-optimization-subs.csv'
subject_list = projectRoot + '/bids/derivatives/qcs/ses-v2-analysis-subs.csv'
subj_df = pd.read_csv(subject_list)
# subj_df = pd.read_csv(subject_list, sep="\t")

subjects = subj_df['sub_code'].values
# subjects = subjects[~np.char.startswith(subjects.astype(str), 'SD')] # Used for debugging
print('Number of Subjects:', subjects.size)


# %% Misc. helper functions

# save 3 column event file
def saveEventFile(fname, event):
    if len(event) == 0:
        # event = np.empty((1,3))
        event = np.array([[0, 0, 0]], dtype=object)
    dirname = os.path.dirname(fname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    np.savetxt(fname, event, fmt='%f')


def saveErrorFlags(fname, errLogDf):
    dirname = os.path.dirname(fname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    errLogDf.to_csv(fname, sep=',', index=False, na_rep='Null')


def get_last_session(sub):
    # gets the last MRI session
    path = logPathPattern_s % {'sub': sub}
    sessions = glob(path + '/*')
    sessions = [session.split(os.sep)[-1] for session in sessions]
    sessions = [session for session in sessions if session.isdigit()]
    sessions = [int(session) for session in sessions]
    latest_session = str(max(sessions))
    return latest_session

# get list of saccades for current run, start and end times in seconds
def getEyeEvents(fname, timeStart, timeEnd):
    timeStart = 1000*float(timeStart)
    timeEnd = 1000*float(timeEnd)
    colnames = ['onset', 'duration', 'tStartUnity', 'fixed']
    dataEvents = pd.read_csv(fname, names=colnames, sep=',', low_memory=False)
    dataEvents = dataEvents.iloc[1:]  # first row is column names
    # convert to list of float
    dataEvents['tStartUnity'] = dataEvents['tStartUnity'].astype(float)
    eventTimes = dataEvents['tStartUnity']
    # keep only events during this fMRI run
    runEvents = dataEvents[(eventTimes >= timeStart) & (eventTimes <= timeEnd)]
    # bin this run's events
    runEventTimes = runEvents['tStartUnity']/1000  # in sec
    TR = 1.5   # in sec
    bins = np.arange(timeStart/1000, timeEnd/1000 + TR, TR)  # in sec
    event_bins = np.digitize(runEventTimes, bins)
    # Count the number of datapoints in each bin
    binned_events = np.bincount(event_bins, minlength=len(bins))

    return binned_events, runEvents


# get timestamp of start and complete level event from full log file. Level start events indicate run onsets. Level complete events indicate null event onsets.
def getFullLog(fname, idx):
    TSlevel = [np.nan, np.nan]
    if len(fname) == 0:
        print('!!! WARNING: no full log file found. Check log files !!!')
    if len(fname) > 1:
        print('!!! WARNING: no unique full log file found. Check log files !!!')
        print('using the largest file')
        sizes = [os.path.getsize(f) for f in fname]
        fname = [[fname[sizes.index(max(sizes))]][0]]
        print(' . ' + fname[0])
    colnames = ['frame', 'TS', 'tsGameplay', 'wh', 'moduleType', 'evType', 'evCode', 'h', 'i', 'j', 'k', 'l']
    dataFullLog = pd.read_csv(fname[0], names=colnames, sep=';', low_memory=False)
    # get level start timestamp and check whether there is only 1 level start event
    if np.sum([dataFullLog['evType'] == 'LEVEL_START']) == 1:
        TSlevel[0] = (dataFullLog['TS'][dataFullLog['evType'] == 'LEVEL_START'] / 1000).to_numpy()[0]  # level start TS
    else:
        print('!!! WARNING: Run: ' + str(
            idx + 1) + ' ! number of level_start events does not equal 1. Check log files !!!')
    # get level complete timestamp and check whether there is only 1 level start event
    if np.sum([dataFullLog['evType'] == 'LEVEL_COMPLETE']) == 1:
        TSlevel[1] = (dataFullLog['TS'][dataFullLog['evType'] == 'LEVEL_COMPLETE'] / 1000).to_numpy()[0]  # level end TS
    else:
        print('!!! WARNING: Run: ' + str(
            idx + 1) + ' ! number of level_complete event does not equal 1. Check log files !!!')
    # get MR trigger timestamps
    TSMriTrig = (dataFullLog['TS'][(dataFullLog['moduleType'] == 'TRIGGER_MANAGER_FMRI') & (
                dataFullLog['evType'] == 'TRIGGER_RECEIVED') & (dataFullLog[
                                                                    'evCode'] == 'TRCode')] / 1000).to_numpy()  # TS of MRI triggers
    return TSlevel, TSMriTrig


# get MRI triggers from serial dump. Serial dumps are used to extract MRI trigger timestamps for comparison with those extracted from the full log and to bridge the gap between levels, which are not logged in the full log
def getTriggerTs_times_donders(fname):
    """
    Changes: switched to re for reliably detecting different header patterns
    and added support for different delimiters in the serial dump. Also allows
    for rows with multiple MRItriggerKey now.
    v3: updated lastHeaderPattern to allow different order of com port response
    """
    import re
    # Pattern to match the last header line
    lastHeaderPattern = re.compile(
        r'\[SERIAL_PORT_COM3\]\s+(Opened|Attempting\s+to\s+Open)\s+port\s+COM3\s+\(115200\s+baud.*')
    # last_match = None
    last_match_line_num = None
    with open(fname) as myFile:
        for num, line in enumerate(myFile, 1):
            if lastHeaderPattern.match(line.strip()):
                # last_match = line.strip()
                last_match_line_num = num

    colnames = ['SPort', 'TS', 'port', 'key']
    # Read the file with flexible delimiters
    dataTriggers = pd.read_csv(fname, sep=r'\s+|;|\t', names=colnames, header=None, skiprows=last_match_line_num,
                               engine='python')

    # Filter data to include only the MRI trigger port and keys containing only the MRI trigger key
    dataTriggers = dataTriggers[(dataTriggers['port'] == MRItriggerPort)]
    dataTriggers = dataTriggers[dataTriggers['key'].apply(lambda x: all(char == MRItriggerKey for char in x))]

    # Convert timestamps
    triggerTs_times = (dataTriggers['TS'].to_numpy(dtype=float)) / 1000

    return triggerTs_times


def getTriggerTs_times_donders_for_testing(fname):
    """
    Changes: switched to re for reliably detecting different header patterns
    and added support for different delimiters in the serial dump. Also allows
    for rows with multiple MRItriggerKey now.
    v3: updated lastHeaderPattern to allow different order of com port response
    """
    import re
    # Pattern to match the last header line
    lastHeaderPattern = re.compile(
        r'\[SERIAL_PORT_COM3\]\s+(Opened|Attempting\s+to\s+Open)\s+port\s+COM3\s+\(115200\s+baud.*')
    # last_match = None
    last_match_line_num = None
    with open(fname) as myFile:
        for num, line in enumerate(myFile, 1):
            if lastHeaderPattern.match(line.strip()):
                # last_match = line.strip()
                last_match_line_num = num

    colnames = ['SPort', 'TS', 'port', 'key']
    # Read the file with flexible delimiters
    dataTriggers = pd.read_csv(fname, sep=r'\s+|;|\t', names=colnames, header=None, skiprows=last_match_line_num,
                               engine='python')

    # Filter data to include only the MRI trigger port and keys containing only the MRI trigger key
    dataTriggers = dataTriggers[(dataTriggers['port'] == MRItriggerPort)]
    dataTriggers = dataTriggers[dataTriggers['key'].apply(lambda x: all(char == MRItriggerKey for char in x))]

    # Convert timestamps
    triggerTs_times = (dataTriggers['TS'].to_numpy(dtype=float)) / 1000

    return triggerTs_times, dataTriggers


def test_get_trigger_keys_based_on_timestamps(dataTriggers, triggerTs_times, expected_key):
    # Extract trigger keys
    trigger_keys = []
    for ts in triggerTs_times:
        # Find the row in dataTriggers with the matching timestamp
        matching_row = dataTriggers.loc[dataTriggers['TS'].astype(float) == ts * 1000]
        if not matching_row.empty:
            # Append the key from the matching row
            trigger_keys.append(matching_row['key'].values[0])

    # Extract all rows where the timestamp matches but the key not MRItriggerKey
    pattern = re.compile(f'^{expected_key}+$')
    unexpected_rows = dataTriggers.loc[
        (dataTriggers['TS'].astype(float) / 1000).isin(triggerTs_times) &
        (~dataTriggers['key'].str.match(pattern, na=False))
        ]

    return trigger_keys, unexpected_rows


def run_trigger_test_for_sub(fname):
    # Get fname
    filename_without_suffix = os.path.splitext(os.path.basename(fname))[0]
    # Get triggers
    triggerTs_times, dataTriggers = getTriggerTs_times_donders_for_testing(fname)
    # Calculate time
    timespan = triggerTs_times[-1] - triggerTs_times[0]
    print('n timestamps for: ' + filename_without_suffix + ' = ' + str(
        len(triggerTs_times)) + ' | spanning a duration of: ' + str(np.round(timespan / 60, 1)) + ' minutes')
    # Run test to find expected values
    trigger_keys, unexpected_rows = test_get_trigger_keys_based_on_timestamps(dataTriggers, triggerTs_times,
                                                                              MRItriggerKey)
    if len(unexpected_rows):
        print('. unexpected keys found in extracted triggers: ')
        print(unexpected_rows)
    else:
        print('. no unexpected trigger keys found')


def getTriggerTs_times_yale(fname):
    colnames = ['n1', 'port', 'key', 'n2', 'trigger']
    dataTriggers = pd.read_csv(fname, sep='\s|;', names=colnames, header=None,
                               engine='python')  # serial dump needs some cleaning due to mixed delimiters and null bytes
    # corrected regular expression to include timestamps that are whole numbers, i.e. no decimal point
    triggerTs_times = np.concatenate(
        dataTriggers['n1'][dataTriggers['trigger'] == 'Alpha5'][dataTriggers['key'] == 'KEY_DOWN'].str.extract(
            r'_(\d+\.\d+|\d+)_').to_numpy(dtype=float)) / 1000  # get only scanTriggers timestamps and convert to sec.
    return triggerTs_times


# get MRI specific log file with run onset times, instruction onset times, etc.
def getMRIlog(fname):
    colnames = ['F_TS_TSVG', 'run', 'level', 'evType']
    dataMRI = pd.read_csv(fname, sep=';', names=colnames, skiprows=0, engine='python')
    dataMRI[['frame', 'TS', 'tsGameplay']] = dataMRI['F_TS_TSVG'].str.split('_', expand=True)

    # if participant is SD188, then we need to grab the first 3 levels from session 1
    if 'SD188' in fname:
        fname = fname[:fname.rfind('/')] + '/SD1881_FMRI.txt'
        dataMRI_ = pd.read_csv(fname, sep=';', names=colnames, skiprows=0, engine='python')
        dataMRI_[['frame', 'TS', 'tsGameplay']] = dataMRI_['F_TS_TSVG'].str.split('_', expand=True)
        dataMRI = pd.concat([dataMRI, dataMRI_], axis=0)

    return dataMRI


# count n dicom volumes per run, split into the different run types
def countNDcmVols(logPath):
    # DcmLabels = ['/*VGR*0', '/*eplay*0', '/*DC*inv', '/*EVC*0']
    DcmLabels = ['/' + str(i) for i in range(1, 40)]
    for label in DcmLabels:
        pathList = sorted(glob(logPath[0:-23] + '/SCANS' + label + '/DICOM'))
        nDcmVols = []

        for curPath in pathList:
            dcm_files = [file for file in os.listdir(curPath) if file.endswith(".dcm")]
            nDcmVols.append(len(dcm_files))

        if nDcmVols:
            print('. Found: ' + str(nDcmVols) + ' DICOM images for: ' + label)
        else:
            print('. WARNING: no DICOM images found  for: ' + label + '!!!')


# get timestamp of the last trigger of a run based on the serial dump timings; i.e. the last volume (trigger) should be after the level finished
def getLastTrigTS(triggerTs_times, TSlevel_B_endOfRun_log):
    # find triggers closest to end of run (from log)
    trigLevelComplete = np.argmin(np.abs(triggerTs_times - TSlevel_B_endOfRun_log))
    # find next large delay between triggers (>20sec), indicating the transition to the next run and record timestamp from last trigger as ts of last trigger for current run
    transition = np.where(np.diff(triggerTs_times[trigLevelComplete:]) > 20)[0]
    # if there is no transition, it means we're in the last run and the last trigger is the last trigger in ts serial dump
    if len(transition) == 0:
        transition = len(triggerTs_times) - trigLevelComplete - 1
        trigLastTS = triggerTs_times[trigLevelComplete + transition]
    # if the next trigger with a large delay (indicating the next run) is more than 100 volumes apart from the level complete event, this trigger must be from the 2+ run; i.e. the trigger closest to the log file's last event is already the final trigger
    elif transition[0] > 100:
        transition = 0
        trigLastTS = triggerTs_times[trigLevelComplete + transition]
    else:
        trigLastTS = triggerTs_times[trigLevelComplete + transition][0]
    # check that the last trigger TS is within a reasonable time window (max 1 minute) after the end of level timestamp from the log file
    if np.abs(trigLastTS - TSlevel_B_endOfRun_log) > 60:
        print('. WARNING: last trigger timestamp is more than 1 minute different from level complete timestamp!!!')
    return trigLastTS


# create log file per run with events relevant for MRI analysis
def createMriLog(stimAr, probeAr, stimType, TSlevel_A, TSlevel_B, TSlevel_B_endOfRun_log, trigLastTS,
                 eliminated_trials=None):
    stimAr = stimAr.reset_index(drop=True)
    stimAr['stimulusOnsetTS'] = (stimAr['stimulusOnsetTS'] / 1000) - TSlevel_A[0]
    # add response column
    stimAr['response'] = np.nan
    stimAr['response'].loc[stimAr['type'] == stimType] = np.copy(probeAr['responseEvaluation'])
    # add the 2 null event rows
    stimAr.loc[len(stimAr)] = [TSlevel_A[1] - TSlevel_A[0], 'NullEvent', 'NullEvent', 'NullEvent', 'NullEvent',
                               'NullEvent']
    stimAr.loc[len(stimAr)] = [TSlevel_B[1] - TSlevel_A[0], 'NullEvent', 'NullEvent_short', 'NullEvent', 'NullEvent',
                               'NullEvent']
    # sort by timestamp and reset idx
    stimAr = stimAr.sort_values(by=['stimulusOnsetTS'])
    stimAr = stimAr.reset_index(drop=True)
    # add stimulus duration column
    duration = np.tile(stimulusDuration, len(stimAr))
    stimAr.insert(1, "duration", duration)
    # set different durations for null events
    stimAr['duration'][stimAr['stimulusType'] == 'NullEvent'] = nullEventDuration
    stimAr['duration'][stimAr['stimulusType'] == 'NullEvent_short'] = shortNullEventDuration
    stimAr['stimulusType'][stimAr['stimulusType'] == 'NullEvent_short'] = 'NullEvent'
    # add dummy end of run event to soak up any possible variance associated with game pop-up at end of run, transition to menu, leader board, etc.
    stimAr.loc[len(stimAr)] = [TSlevel_B[1] - TSlevel_A[0] + shortNullEventDuration, 'endOfRun', 'endOfRun', 'endOfRun',
                               'endOfRun', 'endOfRun', 'endOfRun']
    stimAr['duration'][stimAr['stimulusType'] == 'endOfRun'] = 10 + (trigLastTS - TSlevel_A[0]) - \
                                                               stimAr['stimulusOnsetTS'][
                                                                   stimAr['stimulusType'] == 'endOfRun']
    # rename columns to conform to bids
    stimAr.rename(columns={'stimulusOnsetTS': 'onset'}, inplace=True)
    # Add a trial_type column with information needed for decoding
    # stimAr['trial_type'] = stimAr.apply(lambda row: f"{row['stimulusType']}_{row['stimulusLocation']}_{row['response']}", axis=1)
    stimAr['trial_type'] = stimAr['stimulusType']

    if eliminated_trials is not None:
        events_ = stimAr[~stimAr['stimulusType'].isin(['NullEvent', 'endOfRun'])]
        elim_onsets = events_.iloc[np.where(eliminated_trials == 1)[0]]['onset'].to_numpy()
        stimAr = stimAr[~stimAr['onset'].isin(elim_onsets)]

    return stimAr


# %%  Log file checks

# perform various checks on the MRI triggers and event logs
def logChecks(stim, triggerTs, idx, TSlevel_A, TSlevel_B, TSlevel_A_full, triggerTs_times):
    # first & last timestamp of an event
    TSfirstEV = stim['stimulusOnsetTS'].iloc[0] / 1000
    TSlastEV = stim['stimulusOnsetTS'].iloc[-1] / 1000
    # create error flags
    errLogFlag = []
    # 1. check that level start event and first estimated trigger are close enough in time (<20ms; i.e. approx. 1 frame)
    if abs(TSlevel_A[0] - triggerTs[0]) > 0.02:
        print('!!! WARNING: Run: ' + str(
            idx + 1) + ' ! 1st level_start event and first trigger differ by more than 20ms (exactly = ' + str(
            np.round(abs(TSlevel_A[0] - triggerTs[0]),
                     1)) + 'ms). Minor differences indicate stimulus PC hiccups and may not be problematic. Check log files !!!')
        errLogFlag.append(1)
    # 2. check that no event takes places before first trigger
    if TSfirstEV < triggerTs[0]:
        print('!!! WARNING: Run: ' + str(
            idx + 1) + ' ! first stimulus event takes place BEFORE first trigger. Check log files !!!')
        errLogFlag.append(2)
    # 3. check that no event takes place before level start
    if TSfirstEV < TSlevel_A[0]:
        print('!!! WARNING: Run: ' + str(
            idx + 1) + ' ! first stimulus event takes place BEFORE start 1st level event. Check log files !!!')
        errLogFlag.append(3)
    # 4. check that last event takes place before last trigger
    if (TSlastEV > triggerTs[-1]) and logsHaveMriTriggers:
        print('!!! WARNING: Run: ' + str(
            idx + 1) + ' ! last stimulus event takes place AFTER last trigger. Check log files !!!')
        errLogFlag.append(4)
    # 5. check that no events take place after level complete event (except for triggers)
    if TSlastEV > TSlevel_B[1]:
        print('!!! WARNING: Run: ' + str(
            idx + 1) + ' ! last stimulus event takes place AFTER 2nd level complete event. Check log files !!!')
        errLogFlag.append(5)
    # 6. check that level start is before level complete
    if TSlevel_A[0] > TSlevel_A[1] or TSlevel_B[0] > TSlevel_B[1]:
        print('!!! WARNING: Run: ' + str(
            idx + 1) + ' ! level start event takes place AFTER level complete event. Check log files !!!')
        errLogFlag.append(6)
    # 7. check that triggers still come in after level complete
    if (triggerTs[-1] < TSlevel_B[1]) and logsHaveMriTriggers:
        print('!!! WARNING: Run: ' + str(
            idx + 1) + ' ! triggers stop BEFORE 2nd level complete event. Check log files !!!')
        errLogFlag.append(7)
    # 8. check that last trigger and last level complete timestamp differ by a reasonable amount (max. ~1min)
    if (abs(triggerTs[-1] - TSlevel_B[1]) > 60) and logsHaveMriTriggers:
        print('!!! WARNING: Run: ' + str(
            idx + 1) + ' ! last trigger and 2nd level complete event timestamps differ by ' + str(
            np.round(abs(triggerTs[-1] - TSlevel_B[1]), 1)) + 's. Check log files !!!')
        errLogFlag.append(8)
    # 9. check that first level complete event and second level start event differ by less than 30 sec.
    if abs(TSlevel_B[0] - TSlevel_A[1]) > 30:
        print('!!! WARNING: Run: ' + str(
            idx + 1) + ' ! 1st level complete event and 2nd level start event differ by more than 30s. Check log files !!!')
        errLogFlag.append(9)
    # 10. check that the critical level start event (level 1) from full log and mri log match
    if TSlevel_A[0] != TSlevel_A_full[0]:
        print('!!! WARNING: Run: ' + str(
            idx + 1) + ' ! timestamp of run start from full log does NOT match FMRI log. This may be because the game was aborted and restarted. Check log files !!!')
        errLogFlag.append(10)
    # 11. check whether & when MRI triggers are dropped, as evident by larger than 1TR (+50ms) diff between successive triggers. Flags any trigger drops during levels (+100ms margin at end of levels; expected hiccups due to loading/saving)
    if logsHaveMriTriggers:
        droppedTrigsTS = triggerTs_times[np.where(np.diff(triggerTs_times) > 1.55)[0]]
        droppedA = np.logical_and((droppedTrigsTS > TSlevel_A[0]), (droppedTrigsTS + 0.1 < TSlevel_A[1]))
        droppedB = np.logical_and((droppedTrigsTS > TSlevel_B[0]), (droppedTrigsTS + 0.1 < TSlevel_B[1]))
        if any(droppedA) or any(droppedB):
            print('!!! CAUTION: Run: ' + str(idx + 1) + ' Dropped ' + str(
                np.sum(droppedA) + np.sum(droppedB)) + ' trigger(s)! Level 1 [start end]: ' + str(
                np.round(TSlevel_A, 2)) + ' | Level 2 [start end]: ' + str(
                np.round(TSlevel_B, 2)) + ' | Triggers dropped at TS: ' + str(
                np.round(droppedTrigsTS[np.logical_or(droppedA, droppedB)],
                         2)) + '. If dropped triggers are close to level start/end (within 1 frame; i.e. 16.6ms) this can occur.')
            errLogFlag.append(11)
    # note if no error was found
    if not errLogFlag:
        print('... Run: ' + str(idx + 1) + ' passed checks without error.')
    return errLogFlag


# %%  create 3 column event files

# generic event files relevant for most analyses (null events)
def createEvFile_nullAndEndOfRunEV(log, outputEventFilePattern, sub, runType):
    for stimType in log['stimulusType'].unique():
        # null events have no location or probe, hence no need to loop over different responses or types
        if (stimType == 'NullEvent' or stimType == 'endOfRun'):
            #  get onsets, set durations, and parametric mod to create 3 column format event file
            onsets = (log['onset'][log['stimulusType'] == stimType]).to_numpy()
            durations = (log['duration'][log['stimulusType'] == stimType]).to_numpy()
            parametricMod = np.ones((len(onsets)))
            event = np.vstack((onsets, durations, parametricMod)).T
            # save event file
            fname = outputEventFilePattern % {'sub': sub, 'runType': runType, 'eventType': stimType, 'analysisType': ''}
            saveEventFile(fname, event)


# event file splitting into category and side
def createEvFile_categoryOnly(log, stimulusDuration, outputEventFilePattern, sub, runType):
    for stimType in log['stimulusType'].unique():
        if not (stimType == 'NullEvent' or stimType == 'endOfRun'):
            #  get onsets, set durations and parametric mod to create 3 column format event file
            onsets = (log['onset'][log['stimulusType'] == stimType]).to_numpy()
            parametricMod = np.ones((len(onsets)))
            durations = np.tile(stimulusDuration, len(onsets))
            event = np.vstack((onsets, durations, parametricMod)).T
            # save event file
            fname = outputEventFilePattern % {'sub': sub, 'runType': runType, 'eventType': stimType,
                                              'analysisType': 'categoryOnly'}
            saveEventFile(fname, event)


# event file splitting into category and side
def createEvFile_categorySide(log, relevantLocations, stimulusDuration, outputEventFilePattern, sub, runType):
    for stimType in log['stimulusType'].unique():
        if not (stimType == 'NullEvent' or stimType == 'endOfRun'):
            for location in relevantLocations:
                #  get onsets, set durations, and parametric mod to create 3 column format event file
                onsets = (log['onset'][
                    (log['stimulusType'] == stimType) & (log['stimulusLocation'].str.contains(location))]).to_numpy()
                durations = np.tile(stimulusDuration, len(onsets))
                parametricMod = np.ones((len(onsets)))
                event = np.vstack((onsets, durations, parametricMod)).T
                # save event file
                fname = outputEventFilePattern % {'sub': sub, 'runType': runType, 'eventType': stimType + location,
                                                  'analysisType': 'categorySide'}
                saveEventFile(fname, event)


# event file splitting into category and side and seen/unseen
def createEvFile_seenCategorySide(log, relevantLocations, stimulusDuration, outputEventFilePattern, sub, runType):
    for stimType in log['stimulusType'].unique():
        # loop over location (left vs right)
        for location in relevantLocations:
            # loop over probed vs unprobed
            for probed in log['type'][np.logical_and(log['type'] != 'NullEvent', log['type'] != 'endOfRun')].unique():
                # if probed split into responses
                if probed == 'GAME_PROBE' or probed == 'LOCALIZER_STIMULUS':

                    # Pick the right labels depending on task
                    if probed == 'GAME_PROBE':
                        ResponseLabels = probeResponseLabels
                        probeLabel = 'probed'
                    else:
                        ResponseLabels = replayResponseLabels
                        probeLabel = ''

                    for response in probeResponseTypes:
                        # Ignore probed events in the videogame for which there was no response
                        if response == 'StimNoResponse' or response == 'BlankNoResponse':
                            continue
                        # skip impossible probed stimulus type & response combinations
                        # Blanks cannot be falseNegative or truePositive
                        elif ((response == 'FalseNegative') & (stimType == 'None')) | (
                                (response == 'TruePositive') & (stimType == 'None')):
                            continue
                        else:
                            #  get onsets, set durations, and parametric mod to create 3 column format event file
                            onsets = (log['onset'][
                                (log['stimulusType'] == stimType) & (log['stimulusLocation'].str.contains(location)) & (
                                            log['type'] == probed) & (log['response'] == response)]).to_numpy()
                            durations = np.tile(stimulusDuration, len(onsets))
                            parametricMod = np.ones((len(onsets)))
                            event = np.vstack((onsets, durations, parametricMod)).T
                            # save event file

                            fname = outputEventFilePattern % {'sub': sub, 'runType': runType,
                                                              'eventType': probeLabel + ResponseLabels[
                                                                  response] + stimType + location,
                                                              'analysisType': 'seenCategorySide'}
                            saveEventFile(fname, event)
                # if its unprobed no need to split into response types
                else:
                    #  get onsets, set durations, and parametric mod to create 3 column format event file
                    onsets = (log['onset'][
                        (log['stimulusType'] == stimType) & (log['stimulusLocation'].str.contains(location)) & (
                                    log['type'] == probed)]).to_numpy()
                    durations = np.tile(stimulusDuration, len(onsets))
                    parametricMod = np.ones((len(onsets)))
                    event = np.vstack((onsets, durations, parametricMod)).T
                    # save event file
                    probeLabel = 'unprobed'
                    fname = outputEventFilePattern % {'sub': sub, 'runType': runType,
                                                      'eventType': probeLabel + stimType + location,
                                                      'analysisType': 'seenCategorySide'}
                    saveEventFile(fname, event)


# %% Run subject
def runSubject(sub):
    print('========== SUBJECT: ' + sub + ' ==========')

    scannerID = get_last_session(sub)
    print('Using session: ' + scannerID)

    # import relevant log files; to be improved later
    logPath = logPathPattern % {'sub': sub, 'scannerID': scannerID}

    # get stimulus details log
    fname = logPath + logStimuli % {'sub': sub, 'scannerID': scannerID}
    dataStimuli = pd.read_csv(fname, sep=';')

    # get probe details log for Video Game levels
    fname = logPath + logProbes % {'sub': sub, 'scannerID': scannerID}
    dataProbes = pd.read_csv(fname, sep=';')

    # get details log for Replay levels
    fname = logPath + logReplay % {'sub': sub, 'scannerID': scannerID}
    dataReplay = pd.read_csv(fname, sep=';')

    # get FMRI log file
    fname = logPath + logMRI % {'sub': sub, 'scannerID': scannerID}
    dataMRI = getMRIlog(fname)

    # get data monitoring team trial info log
    fname = projectRoot + '/bids/derivatives/qcs/ses-v2-TrialInfo.csv'
    dataDMT = pd.read_csv(fname)
    dataDMT = dataDMT[dataDMT['SubjectName'] == sub]

    # get MR triggers & button presses from log files, depending if data was acqurired at Yale or Donders
    if np.char.startswith(sub, 'SC'):
        fname = logPath + logTriggers_donders % {'sub': sub, 'scannerID': scannerID}
        triggerTs_times = getTriggerTs_times_donders(fname)
        run_trigger_test_for_sub(fname)
    elif np.char.startswith(sub, 'SD'):
        fname = logPath + logTriggers_yale % {'sub': sub, 'scannerID': scannerID}
        triggerTs_times = getTriggerTs_times_yale(fname)

    # Filter dataProbes and dataReplay to keep only relevant responses
    dataProbes = dataProbes[dataProbes['responseEvaluation'].isin(probeResponseTypes)]
    dataReplay = dataReplay[dataReplay['responseEvaluation'].isin(probeResponseTypes)]

    # print number of dicom images
    if logsHaveMriTriggers:
        countNDcmVols(logPath)

    # % loop over runs
    replayCounter = 0
    errLogFlag = [[], []]
    for idx in range(len(runOnsetLevels)):

        # get relevant data for current run from stimuli and probes
        stim = dataStimuli[["stimulusOnsetTS", "type", "stimulusType", "stimulusLocation", "stimulusName"]][
            dataStimuli["currentLevelID"].between(runOnsetLevels[idx], runOnsetLevels[idx] + 1)]
        probe = dataProbes[["responseEvaluation"]][
            dataProbes["currentLevelID"].between(runOnsetLevels[idx], runOnsetLevels[idx] + 1)]
        replay = dataReplay[["responseEvaluation"]][
            dataReplay["currentLevelID"].between(runOnsetLevels[idx], runOnsetLevels[idx] + 1)]

        ##### get exact run (level) onset time from FMRI log
        # level 1 (of a run);
        # if we have an aborted and restarted experiment then take the last instance of the level start event
        TSlevel_A = (dataMRI['TS'][(dataMRI['run'] == runLabelsMRIlog[idx]) & (dataMRI['level'] == '1') & (
                    dataMRI['evType'] == 'LEVEL_BEGIN')].to_numpy(dtype=float)) / 1000
        if len(TSlevel_A) > 1:  # if we have an aborted and restarted experiment then take the last instance of the level start event
            TSlevel_A = TSlevel_A[-1]
        TSlevel_A_ = (dataMRI['TS'][(dataMRI['run'] == runLabelsMRIlog[idx]) & (dataMRI['level'] == '1') & (
                    dataMRI['evType'] == 'LEVEL_END')].to_numpy(dtype=float)) / 1000
        if len(TSlevel_A_) > 1:  # if we have an aborted and restarted experiment then take the last instance of the level end event
            TSlevel_A_ = TSlevel_A_[-1]
        TSlevel_A = np.append(TSlevel_A, TSlevel_A_)
        # level 2 (of a run);
        TSlevel_B = (dataMRI['TS'][(dataMRI['run'] == runLabelsMRIlog[idx]) & (dataMRI['level'] == '2') & (
                    dataMRI['evType'] == 'LEVEL_BEGIN')].to_numpy(dtype=float)) / 1000
        if len(TSlevel_B) > 1:  # if we have an aborted and restarted experiment then take the last instance of the level start event
            TSlevel_B = TSlevel_B[-1]
        TSlevel_B_ = (dataMRI['TS'][(dataMRI['run'] == runLabelsMRIlog[idx]) & (dataMRI['level'] == '2') & (
                    dataMRI['evType'] == 'LEVEL_END')].to_numpy(dtype=float)) / 1000
        if len(TSlevel_B_) > 1:  # if we have an aborted and restarted experiment then take the last instance of the level end event
            TSlevel_B_ = TSlevel_B_[-1]
        TSlevel_B = np.append(TSlevel_B, TSlevel_B_)
        # get timestamp when run ended from MRI log (level complete popup hidden; i.e. onset of leaderboard)
        TSlevel_B_endOfRun_log = dataMRI['TS'][(dataMRI['run'] == runLabelsMRIlog[idx]) & (
                    dataMRI['evType'] == 'LEVEL_COMPLETE_POPUP_HIDDEN')].to_numpy(dtype=float)[0] / 1000

        ##### get exact run onset time and MR triggers timestamps from full logs
        # level 1 (of a run); level start (run start) and level end time (level end times should also correspond to null event onset times; this needs to be checked XXX)

        # SD188 full logs switch from ses1 to ses2 at level 7 i.e runOnsetLevels[3]
        if sub == 'SD188' and idx < 3:
            scannerID_s = '1'
        else:
            scannerID_s = scannerID
        fname = glob(
            logPath + logFull % {'sub': sub, 'scannerID': scannerID_s} + runByLevelAndWorldCode[idx] + '*COMPLETED.csv')
        TSlevel_A_full, TSMriTrig_A = getFullLog(fname, idx)
        # level 2 (of a run); level 2 start and level 2 end time (level end times should also correspond to null event onset times; this needs to be checked XXX)
        fname = glob(logPath + logFull % {'sub': sub, 'scannerID': scannerID_s} + complementaryLevelAndWorldCode[
            idx] + '*COMPLETED.csv')
        TSlevel_B_full, TSMriTrig_B = getFullLog(fname, idx)
        # discard triggers to account for dummy volumes
        triggerTs = np.concatenate((TSMriTrig_A, TSMriTrig_B))
        triggerTs = triggerTs[dummyTriggers::]

        ##### get the last trigger from this run (estimated from serial dump and log file)
        trigLastTS = getLastTrigTS(triggerTs_times, TSlevel_B_endOfRun_log)


        ##### load saccades for this run
        sacc_folder = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/qcs/sub-' + sub + '/ses-v2/et/'
        sacc_filename = os.path.join(sacc_folder, sub+'_saccs_new.csv')
        blinks_filename = os.path.join(sacc_folder, sub+'_blinks_new.csv')
        run_start = TSlevel_A_full[0]
        run_end = TSlevel_B_full[1]
        padding_end = 60    # max number of secs after the end of the 2nd level
        binned_sacc, run_sacc = getEyeEvents(sacc_filename, run_start, run_end+padding_end)  #from start of 1st level to end of 2nd level
        print('loaded saccade data')
        binned_blinks, run_blinks = getEyeEvents(blinks_filename, run_start, run_end+padding_end)  # from start of 1st level to end of 2nd level
        print('loaded blink data')

        ###### load the confound regressor file
        preprocessed_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/fmriprep'
        regressor_dir = os.path.join(preprocessed_dir, 'sub-'+sub, 'ses-V2', 'func')
        if idx<8:
            vg_or_replay = 'VG'
            nrun = idx+1
        else:
            vg_or_replay = 'Replay'
            nrun = idx-7
        regressor_filename = os.path.join(regressor_dir, 'sub-' + sub + '_ses-V2_task-' + vg_or_replay + '_run-'
                                          + str(nrun) + '_desc-confounds_regressors.tsv')
        confound_data = pd.read_csv(regressor_filename, sep='\t')
        num_rows = confound_data.shape[0]  # number of TRs for this run
        binned_sacc = binned_sacc[:num_rows]  # discard saccades that are after the end of the scan
        binned_blinks = binned_blinks[:num_rows]  # discard blinks that are after the end of the scan
        # check whether we already added the eye regressors
        last_column_name = confound_data.columns[-1]
        if (last_column_name != 'saccades') & (last_column_name != 'blinks'):
            # Add the saccade column
            confound_data['saccades'] = binned_sacc
            # Add the blinks column
            confound_data['blinks'] = binned_blinks
            #new_filename = os.path.join(regressor_dir, 'sub-' + sub + '_ses-V2_task-' + vg_or_replay + '_run-' + str(nrun) + '_desc-confounds_regressors_NEW.tsv')
            confound_data.to_csv(regressor_filename, sep='\t', index=False)
            print('Added eye regressors')
        else:
            print('EYE REGRESSORS ALREADY EXIST!!')


        ##### Checks #####
        errLogFlag[0].append('Run%s' % (idx + 1))
        errLogFlag[1].append(logChecks(stim, triggerTs, idx, TSlevel_A, TSlevel_B, TSlevel_A_full, triggerTs_times))

        # get elimination array for current run from DMT log
        dataDMTrun = dataDMT[dataDMT['currentLevelID'].between(runOnsetLevels[idx], runOnsetLevels[idx] + 1)]
        eliminated_trials = dataDMTrun['IS_LEVEL_ELIMINATED'].to_numpy()

        ##### Create log file with relevant events per run #####
        if not onlyRunChecks:
            # Gather responses from different log files depending on whether is a video game run or a replay run
            if runByLevelAndWorldCode[idx][0] == 'L':
                log = createMriLog(stim, replay, 'LOCALIZER_STIMULUS', TSlevel_A, TSlevel_B, TSlevel_B_endOfRun_log,
                                   trigLastTS, eliminated_trials)
                replayCounter += 1

                runType = 'task-Replay_run-' + str(replayCounter)
            else:
                log = createMriLog(stim, probe, 'GAME_PROBE', TSlevel_A, TSlevel_B, TSlevel_B_endOfRun_log, trigLastTS,
                                   eliminated_trials)
                runType = 'task-VG_run-' + str(idx + 1)

            # Adjust timestamps of BIDS events files by adding the time of dummy scans
            log['onset'] = log['onset'] + dummyTriggers * TR

            fname = outputLogFilePattern % {'sub': sub, 'runType': runType}
            log.to_csv(fname, sep='\t', index=False, na_rep='Null')

            # ensure it is saved
            if os.path.exists(fname):
                print('... Run: ' + str(idx + 1) + ' log file created.')
            else:
                print('!!! WARNING: Run: ' + str(idx + 1) + ' log file NOT created. Check log files !!!')

            ##### Create event files for GLM per run#####
            print('... Run: ' + str(idx + 1) + ' creating event files.')

            # Adjust timestamps of FSL events files by removing the time of dummy scans
            log['onset'] = log['onset'] - dummyTriggers * TR

            # if itś a replay run we cannot split into seen/unseen (as there are no probes)
            # if runByLevelAndWorldCode[idx][0]=='L':
            #     createEvFile_nullAndEndOfRunEV(log, outputEventFilePattern, sub, runType)
            #     createEvFile_categoryOnly(log, stimulusDuration, outputEventFilePattern, sub, runType)
            #     createEvFile_categorySide(log, relevantLocations, stimulusDuration, outputEventFilePattern, sub, runType)
            # # if its a VG run we create event files for both split into seens/unseen & without splitting seen/unseen
            # else:
            createEvFile_nullAndEndOfRunEV(log, outputEventFilePattern, sub, runType)
            createEvFile_categoryOnly(log, stimulusDuration, outputEventFilePattern, sub, runType)
            createEvFile_categorySide(log, relevantLocations, stimulusDuration, outputEventFilePattern, sub, runType)
            createEvFile_seenCategorySide(log, relevantLocations, stimulusDuration, outputEventFilePattern, sub,
                                          runType)

    # save log file warning flags
    fname = (outputErrorFlagPattern % {'sub': sub, 'ses': 'mri02'})
    errLogDf = pd.DataFrame({"RunNo": errLogFlag[0], "WarningCode": errLogFlag[1]})
    saveErrorFlags(fname, errLogDf)


# loop over subjects
doneSubjects = ['SC109', 'SC123', 'SD119', 'SD198', 'SD135', 'SD159', 'SC157', 'SC189', 'SC132', 'SD174', 'SC154',
                'SD147',
                'SC173', 'SC196', 'SD101', 'SC168', 'SC124', 'SC110', 'SD107', 'SC145', 'SC170', 'SC191', 'SC194',
                'SC118',
                'SC129', 'SC114', 'SC158', 'SD195', 'SD176', 'SC122', 'SC182', 'SD126', 'SC172', 'SD165', 'SC152',
                'SD193',
                'SC121', 'SD191', 'SD199', 'SD190', 'SC148', 'SC120', 'SD136', 'SC131', 'SD182', 'SC187', 'SC202',
                'SD153',
                'SD131', 'SD130', 'SC159', 'SC140', 'SC108', 'SD168', 'SC143', 'SD171', 'SC142', 'SD201', 'SC136',
                'SD118',
                'SC171', 'SC183', 'SD123', 'SD166', 'SD134', 'SD163', 'SD185', 'SD188', 'SD141', 'SC192', 'SD137',
                'SC160',
                'SC144', 'SD156', 'SD194', 'SD196']

goodSubjects = ['SC109', 'SC123', 'SD119', 'SD159', 'SC157', 'SC132', 'SD174', 'SD147',
                'SC173', 'SC196', 'SD101', 'SC168', 'SC124', 'SC110', 'SD107', 'SC145', 'SC170', 'SC194', 'SC118',
                'SC129', 'SC114', 'SC158', 'SD195', 'SD176', 'SC122', 'SD126', 'SC172', 'SD165', 'SC152', 'SD193',
                'SC121', 'SD191', 'SC148', 'SD136', 'SD182', 'SC202',
                'SD131', 'SC159', 'SC140', 'SC108', 'SD168', 'SC143', 'SD171', 'SC142', 'SD201', 'SC136', 'SD118',
                'SC171', 'SC183', 'SD123', 'SD166', 'SD134', 'SD163', 'SD185', 'SD141', 'SC160',
                'SD156', 'SD194', 'SD196']

badSubjects = ['SD198', 'SD135', 'SC189', 'SC154', 'SC191', 'SC182', 'SD199', 'SD190', 'SC120', 'SC131', 'SC187',
               'SD153',
               'SD130', 'SD188', 'SC192', 'SD137', 'SC144']
goodSubCounter = 0
# for sub in subjects:
#    if (sub not in badSubjects) and (sub not in doneSubjects):
#        runSubject(sub)
#        goodSubCounter = goodSubCounter+1
# print("done. " + str(goodSubCounter) + " good subjects processed")

#runSubject('SD188')
#runSubject('SC109')

# loop over subjects
doneSubjects = ['SC109', 'SC123', 'SD119', 'SD198', 'SD135', 'SD159', 'SC157', 'SC189', 'SC132', 'SD174', 'SC154',
                'SD147', 'SC173', 'SC196', 'SD101', 'SC168', 'SC124', 'SC110', 'SD107', 'SC145', 'SC170', 'SC191',
                'SC194', 'SC118',
                'SC129', 'SC114', 'SC158', 'SD195', 'SC122', 'SC182', 'SD126', 'SC172', 'SD165', 'SC152', 'SD193',
                'SC121', 'SD191', 'SD199', 'SD190', 'SC148', 'SC120', 'SD136', 'SC131', 'SD182', 'SC187', 'SC202', 'SD153',
                'SD131', 'SD130', 'SC159', 'SC140', 'SC108', 'SD168', 'SC143', 'SD171', 'SC142', 'SC136', 'SD118',
                'SC171', 'SC183', 'SD123', 'SD166', 'SD134', 'SD163', 'SD185', 'SD188', 'SD141', 'SC192', 'SD137', 'SC160',
                'SC144', 'SD194', 'SD196']
noeyeSubjects = ['SD176', 'SD201', 'SD156']


''' doneSubjects = ['SC109', 'SC123', 'SD119', 'SD198', 'SD135', 'SD159', 'SC157', 'SC189', 'SC132', 'SD174', 'SC154', 'SD147',
                'SC173', 'SC196', 'SD101', 'SC168', 'SC124', 'SC110', 'SD107', 'SC145', 'SC170', 'SC191', 'SC194', 'SC118',
                'SC129', 'SC114', 'SC158', 'SD195', 'SD176', 'SC122', 'SC182', 'SD126', 'SC172', 'SD165', 'SC152', 'SD193',
                'SC121', 'SD191', 'SD199', 'SD190', 'SC148', 'SC120', 'SD136', 'SC131', 'SD182', 'SC187', 'SC202', 'SD153',
                'SD131', 'SD130', 'SC159', 'SC140', 'SC108', 'SD168', 'SC143', 'SD171', 'SC142', 'SD201', 'SC136', 'SD118',
                'SC171', 'SC183', 'SD123', 'SD166', 'SD134', 'SD163', 'SD185', 'SD188', 'SD141', 'SC192', 'SD137', 'SC160',
                'SC144', 'SD156', 'SD194', 'SD196'] 

goodSubjects = ['SC109', 'SC123', 'SD119', 'SD159', 'SC157', 'SC132', 'SD174', 'SD147',
                'SC173', 'SC196', 'SD101', 'SC168', 'SC124', 'SC110', 'SD107', 'SC145', 'SC170', 'SC194', 'SC118',
                'SC129', 'SC114', 'SC158', 'SD195', 'SD176', 'SC122', 'SD126', 'SC172', 'SD165', 'SC152', 'SD193',
                'SC121', 'SD191', 'SC148', 'SD136', 'SD182', 'SC202',
                'SD131', 'SC159', 'SC140', 'SC108', 'SD168', 'SC143', 'SD171', 'SC142', 'SD201', 'SC136', 'SD118',
                'SC171', 'SC183', 'SD123', 'SD166', 'SD134', 'SD163', 'SD185', 'SD141', 'SC160',
                'SD156', 'SD194', 'SD196']

badSubjects = ['SD198', 'SD135', 'SC189', 'SC154', 'SC191', 'SC182', 'SD199', 'SD190', 'SC120', 'SC131', 'SC187', 'SD153',
               'SD130', 'SD188', 'SC192', 'SD137', 'SC144']
badSubjects = ['SD198', 'SD135', 'SD199', 'SD190', 'SD153', 'SD130', 'SD188', 'SD137', 'SC144']
badSubjects = ['SD137', 'SC144']
badSubjects = ['SD188']

badDondersSubjects = ['SC189', 'SC154', 'SC191', 'SC182', 'SC120', 'SC131', 'SC187', 'SC192']
'''

goodSubCounter=0
#for sub in subjects:
#    if (sub not in badSubjects) and (sub not in doneSubjects):
#        runSubject(sub)
#        goodSubCounter = goodSubCounter+1
#print("done. " + str(goodSubCounter) + " good subjects processed")
#runSubject('SC189')
for sub in doneSubjects:
    runSubject(sub)
    goodSubCounter = goodSubCounter + 1
print("done. " + str(goodSubCounter) + " done subjects processed")