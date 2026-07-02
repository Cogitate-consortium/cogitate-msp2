#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 10.07.2020
@author: David Richter (d.richter@donders.ru.nl)

Modified 17.08.2023
@author: Yamil Vidal (hvidaldossantos@gmail.com)

Modified 07.05.2026
@author: Yamil Vidal (hvidaldossantos@gmail.com)

Extracts events relevant for fMRI analysis from various log files of exp.2.
Performs tests on log files.
Outputs MRI specific tsv log file (bids compliant) per run and various FSL compatible event files for 1st level GLMs.

Inputs:
    - log files from exp.2
    - path to dicoms from exp.2 (optional)

Outputs:
    - one log file checks table with warning flags per run (see: logChecks function)
    - one MRI specific log file (tsv) per run with all information relevant for analysis (bids compliant) (see: createMriLog function)
    - 3 column event txt files for use in fMRI analysis per regressor (timestamp, ev duation, parametric modulator) (see: createEvFile_* functions)

Environment overrides (optional):
    BIDS_ROOT, CODE_PATH, RAW_DIR, SUBJECT_CSV, SUB_CODE

Run one subject (Slurm worker):
    python3 exp2_fMRI_logfile_extraction_and_checks.py --sub-code SC108

Submit one Slurm job per subject:
    bash 01_submit_01_exp2_fMRI_logfile_extraction_and_checks.sh

Tested on python v3.7.4, pandas v0.25.2, numpy v1.17.2
"""

# TO DO:
# improve bids compliance of MRI specific log file and add description of columns to json file


# %% Imports & parameters
import argparse
import os
import re
import sys
from glob import glob

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

LOGFILES_DIR = os.path.dirname(os.path.abspath(__file__))


##### Options #####
# The following options need to be adjusted depending on the MRI scanner/recording side.
# 1. MRI and response device settings
# 2. BIDS_ROOT / CODE_PATH / RAW_DIR (or set via environment on cluster)
# 3. subject list CSV (SUBJECT_CSV)
# (4.) session prefixes (new session ID conventions were added on slab, different from pilot data conventions: prescreen=A,practice=B,mri-full-game=F)

# Option to only perform log file checks, without creating MRI specific logs or event files
onlyRunChecks = 0        # set to true (1) to only check log files (no event files are created). Set to false (0) for preparation of real data analysis

# if true runs full check of log files. If false (0) does not perform checks depending on full log files with all MRI triggers; setting this to false is useful for checking log files acquired for testing purposes without running the scanner
logsHaveMriTriggers = 1  # should be TRUE (1) for real data


##### Parameters #####

# session prefixes
prescreenID = 'A'                   # session ID letter for prescreening session (outside of scanner)
practiceID = 'B'                    # session ID letter for practice session (outside of scanner)


# run design params/labels (should remain untouched)
runOnsetLevels = np.concatenate((np.arange(1,16,2), np.arange(100,108,2)))              # numbers of levels starting a run (e.g. levels 1-2 are combined to form run 1, hence level 1 is runOnsetLevel for run1, etc. Replay levels start at lvl id 100)
runByLevelAndWorldCode = ['1_1', '1_3', '2_1', '2_3', '3_1', '3_3', '4_1', '4_3', 'L_1', 'L_3', 'L_5', 'L_7']           # labels of world_level combinations corresponding to the runOnset levels (run numbers), as used in full logs
complementaryLevelAndWorldCode = ['1_2', '1_4', '2_2', '2_4', '3_2', '3_4', '4_2', '4_4', 'L_2', 'L_4', 'L_6', 'L_8']   # labels of world_level combinations corresponding to the second level in a run
runLabelsMRIlog = [1,2,3,4,5,6,7,8,51,52,53,54]         # run labels used in the extraLogs -> FMRI log

# stimulus & response parameters (relevant primarily for creating event files)
stimulusDuration = 0.25                         # duration of stimulus events in seconds
nullEventDuration = 12                          # duration of the first null events in seconds 
shortNullEventDuration = 1                      # duration of the second null events in seconds
relevantLocations = ['Left','Right']            # relevant stimulus locations (i.e. top left and bottom left will be group if left is a relevant category)
probeResponseTypes = ['TruePositive', 'TrueNegative', 'FalseNegative', 'FalsePositive', 'StimNoResponse', 'BlankNoResponse'] # possible response types and mapped labels used for event files

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
dummyTriggers = 3   # number of dummy triggers to be discarded (i.e. MRI volumes collected before run onset)
TR = 1.5            # to be used to correct the timing of FSL three column format event files

# MRI and response device settings. These are used in extracting data from serial dump files (these may need to be adjusted depending on the MRI setup)
MRItriggerPort = 'COM3'                         # device port (e.g. com port) of MRI triggers
MRItriggerKey = 'a'                             # MRI trigger key
ButtonPressPort = 'COM2'                        # device port (e.g. com port) of button box


##### Paths (override via environment on cluster) #####

BIDS_ROOT = os.environ.get(
    'BIDS_ROOT',
    '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids',
)
CODE_PATH = os.environ.get(
    'CODE_PATH',
    os.path.join(BIDS_ROOT, 'derivatives', 'exclude', 'new', 'fMRI_exp2'),
)
RAW_DIR = os.environ.get(
    'RAW_DIR',
    '/mnt/beegfs/XNAT/COGITATE/fMRI/Raw/projects/CoG_fMRI_PhaseII',
)

# output paths
outputLogFilePattern = os.path.join(
    BIDS_ROOT,
    'sub-%(sub)s',
    'ses-V2',
    'func',
    'sub-%(sub)s_ses-V2_%(runType)s_events.tsv',
)
outputErrorFlagPattern = os.path.join(
    BIDS_ROOT,
    'derivatives',
    'exclude',
    'new',
    'logfilechecks',
    'ses-V2',
    'sub-%(sub)s_ses-%(ses)s_errorFlags.csv',
)
outputErrorFlagPatternFullLogs = os.path.join(
    BIDS_ROOT,
    'derivatives',
    'exclude',
    'new',
    'logfilechecks',
    'ses-V2',
    'sub-%(sub)s_ses-%(ses)s_errorFlagsFullLogs.csv',
)
outputEventFilePattern = os.path.join(
    BIDS_ROOT,
    'derivatives',
    'exclude',
    'new',
    'regressoreventfiles',
    'sub-%(sub)s',
    'ses-V2',
    'sub-%(sub)s_ses-V2_%(runType)s_%(eventType)s_EV.txt',
)

# input log file paths & file names
# Used to define the scannerID (MRI acquisition may have been restarted).
path_check_scannerID = os.path.join(
    RAW_DIR, '%(sub)s', '%(sub)s_MR_V2', 'RESOURCES', 'BEH', '%(sub)s', ''
)
logPathPattern = os.path.join(
    RAW_DIR, '%(sub)s', '%(sub)s_MR_V2', 'RESOURCES', 'BEH', '%(sub)s', '%(scannerID)s', ''
)
logProbes = 'Details/%(sub)s%(scannerID)s_ProbeDetails.csv'           # probe details log file for Video Game levels
logReplay = 'Details/%(sub)s%(scannerID)s_LocalizerDetails.csv'       # probe details log file for Replay levels
logStimuli = 'Details/%(sub)s%(scannerID)s_StimulusDetails.csv'       # stimulus details log file
logFull = 'FullLogs/%(sub)s%(scannerID)s_FullLogLevel_'               # full log file (incl. subpaths)
logMRI = 'ExtraLogs/%(sub)s%(scannerID)s_FMRI.txt'                    # fmri specific log file (incl. subpaths)

logTriggers_donders = 'ExtraLogs/%(sub)s%(scannerID)s_Serial.txt'     # serial dump used at the Donders Institute (incl. subpaths)
logTriggers_yale = 'ExtraLogs/%(sub)s%(scannerID)s_HighAccuInput.txt' # Trigger logger used at Yale (incl. subpaths)


# subject list (determines on which subjects scripts are run)
SUBJECT_CSV = os.environ.get(
    'SUBJECT_CSV',
    os.path.join(CODE_PATH, 'ses-v2-analysis-subs-fmri.csv'),
)

# Subjects excluded from batch processing (known bad/missing data).
EXCLUDED_SUBJECTS = frozenset({'SD188'})


def load_subjects(csv_path=None):
    """Return sub_code values with at least one analysis flag TRUE in the CSV."""
    path = csv_path or SUBJECT_CSV
    subj_df = pd.read_csv(path, sep=None, engine='python')
    if 'sub_code' not in subj_df.columns and len(subj_df.columns) == 1 and ';' in subj_df.columns[0]:
        subj_df = pd.read_csv(path, sep=';')
    if 'sub_code' not in subj_df.columns:
        raise ValueError(f'missing sub_code column in {path}')

    flag_cols = [col for col in subj_df.columns if col not in ('sub_code', 'modality', 'Lab')]
    if flag_cols:
        mask = subj_df[flag_cols].apply(
            lambda s: s.astype(str).str.upper().eq('TRUE')
        ).any(axis=1)
        subj_df = subj_df.loc[mask]

    subjects = [str(s) for s in subj_df['sub_code'].tolist() if str(s) not in EXCLUDED_SUBJECTS]
    return subjects


# %% Misc. helper functions

# save 3 column event file
def saveEventFile(fname,event):
    if len(event) == 0:
        #event = np.empty((1,3))
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

# get timestamp of start and complete level event from full log file. Level start events indicate run onsets. Level complete events indicate null event onsets.
def getFullLog(fname,idx):
    TSlevel = [np.nan,np.nan]
    
    if len(fname)>1:
        print('!!! WARNING: no, or no unique, full log file found for current level !!!')
        # Pick the file with the latest creation (modification) time
        latest_idx = max(range(len(fname)), key=lambda i: os.path.getmtime(fname[i]))
        print('Assuming that the last created (most recent) file is the correct one')
        print(' . ' + fname[latest_idx])
        ck = [9] # Outputs an error flag
        fname = [fname[latest_idx]]
 
    else:
        ck = []
    
    fname = [fname[-1]] # Assuming last log file is the correct log file
    
    colnames=['frame','TS','tsGameplay','wh','moduleType','evType','evCode','h','i','j','k','l']
    dataFullLog = pd.read_csv(fname[-1], names=colnames, sep=';', low_memory=False) 
    # get level start timestamp and check whether there is only 1 level start event
    if np.sum([dataFullLog['evType']=='LEVEL_START'])==1:
        TSlevel[0] = (dataFullLog['TS'][dataFullLog['evType']=='LEVEL_START']/1000).to_numpy()[0] # level start TS
    else:
        print('!!! WARNING: Run: ' + str(idx+1) + ' ! number of level_start events does not equal 1. Check log files !!!')
    # get level complete timestamp and check whether there is only 1 level start event
    if np.sum([dataFullLog['evType']=='LEVEL_COMPLETE'])==1:
        TSlevel[1] = (dataFullLog['TS'][dataFullLog['evType']=='LEVEL_COMPLETE']/1000).to_numpy()[0] # level end TS
    else:
        print('!!! WARNING: Run: ' + str(idx+1) + ' ! number of level_complete event does not equal 1. Check log files !!!')
    # get MR trigger timestamps
    TSMriTrig = (dataFullLog['TS'] [(dataFullLog['moduleType']=='TRIGGER_MANAGER_FMRI') & (dataFullLog['evType']=='TRIGGER_RECEIVED') & (dataFullLog['evCode']=='TRCode')]/1000).to_numpy()  # TS of MRI triggers
         
    return TSlevel, TSMriTrig, ck


# get MRI triggers from serial dump. Serial dumps are used to extract MRI trigger timestamps for comparison with those extracted from the full log and to bridge the gap between levels, which are not logged in the full log
def getTriggerTs_times_donders(fname):
    """
    Changes: switched to re for reliably detecting different header patterns 
    and added support for different delimiters in the serial dump. Also allows 
    for rows with multiple MRItriggerKey now.
    v3: updated lastHeaderPattern to allow different order of com port response
    """
    
    # Pattern to match the last header line
    lastHeaderPattern = re.compile(r'\[SERIAL_PORT_COM3\]\s+(Opened|Attempting\s+to\s+Open)\s+port\s+COM3\s+\(115200\s+baud.*')
    #last_match = None
    last_match_line_num = None
    with open(fname) as myFile:
        for num, line in enumerate(myFile, 1):
            if lastHeaderPattern.match(line.strip()):
                #last_match = line.strip()
                last_match_line_num = num
    
    colnames = ['SPort', 'TS', 'port', 'key']
    # Read the file with flexible delimiters
    dataTriggers = pd.read_csv(fname, sep=r'\s+|;|\t', names=colnames, header=None, skiprows=last_match_line_num, engine='python')
    
    # Filter data to include only the MRI trigger port and keys containing only the MRI trigger key
    dataTriggers = dataTriggers[(dataTriggers['port'] == MRItriggerPort)]
    dataTriggers = dataTriggers[dataTriggers['key'].apply(lambda x: all(char == MRItriggerKey for char in x))]
    
    # Convert timestamps
    triggerTs_times = (dataTriggers['TS'].to_numpy(dtype=float)) / 1000
    
    return triggerTs_times

def getTriggerTs_times_yale(fname):
    colnames=['n1','port','key','n2','trigger']
    dataTriggers = pd.read_csv(fname, sep='\s|;', names=colnames, header=None, engine='python') # serial dump needs some cleaning due to mixed delimiters and null bytes        
    triggerTs_times = np.concatenate(dataTriggers['n1'][dataTriggers['trigger']=='Alpha5'][dataTriggers['key']=='KEY_DOWN'].str.extract(r'_(\d+\.?\d*)_').to_numpy(dtype=float))/1000 # get only scanTriggers timestamps and convert to sec.    
    return triggerTs_times


# get MRI specific log file with run onset times, instruction onset times, etc.
def getMRIlog(fname):
    colnames=['F_TS_TSVG','run','level','evType']
    dataMRI = pd.read_csv(fname, sep=';', names=colnames, skiprows=0, engine='python')
    dataMRI[['frame','TS','tsGameplay']] = dataMRI['F_TS_TSVG'].str.split('_',expand=True)
    return dataMRI


# count n dicom volumes per run, split into the different run types
def countNDcmVols(logPath):
    # DcmLabels = ['/*VGR*0', '/*eplay*0', '/*DC*inv', '/*EVC*0']
    DcmLabels = ['/'+str(i) for i in range(1, 40)]
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
    trigLevelComplete = np.where(np.abs(triggerTs_times - TSlevel_B_endOfRun_log) == np.min(np.abs(triggerTs_times-TSlevel_B_endOfRun_log)))[0]
    # find next large delay between triggers (>20sec), indicating the transition to the next run and record timestamp from last trigger as ts of last trigger for current run
    transition = np.where(np.diff(triggerTs_times[trigLevelComplete[0]::])>20)[0]
    # if  there is no transition, it means we're in the last run and the last trigger is the last trigger in ts serial dump
    if len(transition) == 0:
        transition = len(triggerTs_times)-trigLevelComplete-1
        # if the next trigger with a large delay (indicating the next run) is more than 100 volumes apart from the level complete event, this trigger must be from the 2+ run; i.e. the trigger closest to the log file's last event is already the final trigger
    if transition[0] > 100:
        transition = 0
    trigLastTS = triggerTs_times[trigLevelComplete + transition][0]
    # check that the last trigger TS is within a reasonable time window (max 1 minute) after the end of level timestamp from the log file
    if np.abs(trigLastTS - TSlevel_B_endOfRun_log) > 60:
        print('. WARNING: last trigger timestamp is more than 1 minute different from level complete timestamp!!!')
    return trigLastTS


# create log file per run with events relevant for MRI analysis
def createMriLog(stimAr, probeAr, stimType, TSlevel_A, TSlevel_B, TSlevel_B_endOfRun_log, trigLastTS):
    stimAr = stimAr.reset_index(drop=True)
    stimAr['stimulusOnsetTS'] = (stimAr['stimulusOnsetTS']/1000)-TSlevel_A[0]
    # add response column
    stimAr['response'] = pd.Series([None] * len(stimAr), dtype='object')
    stim_mask = stimAr['type'] == stimType
    stimAr.loc[stim_mask, 'response'] = np.asarray(probeAr['responseEvaluation'])
    # add the 2 null event rows   
    stimAr.loc[len(stimAr)] = [TSlevel_A[1]-TSlevel_A[0], 'NullEvent', 'NullEvent', 'NullEvent', 'NullEvent', 'NullEvent']
    stimAr.loc[len(stimAr)] = [TSlevel_B[1]-TSlevel_A[0], 'NullEvent', 'NullEvent_short', 'NullEvent', 'NullEvent', 'NullEvent']
    # sort by timestamp and reset idx
    stimAr = stimAr.sort_values(by=['stimulusOnsetTS'])
    stimAr = stimAr.reset_index(drop=True)
    # add stimulus duration column
    duration = np.tile(stimulusDuration,len(stimAr))
    stimAr.insert(1, "duration", duration)
    # set different durations for null events
    stimAr.loc[stimAr['stimulusType'] == 'NullEvent', 'duration'] = nullEventDuration
    stimAr.loc[stimAr['stimulusType'] == 'NullEvent_short', 'duration'] = shortNullEventDuration
    stimAr.loc[stimAr['stimulusType'] == 'NullEvent_short', 'stimulusType'] = 'NullEvent'
    # add dummy end of run event to soak up any possible variance associated with game pop-up at end of run, transition to menu, leader board, etc.
    stimAr.loc[len(stimAr)] = [TSlevel_B[1]-TSlevel_A[0]+shortNullEventDuration, 'endOfRun', 'endOfRun', 'endOfRun', 'endOfRun', 'endOfRun', 'endOfRun']
    end_mask = stimAr['stimulusType'] == 'endOfRun'
    stimAr.loc[end_mask, 'duration'] = 10 + (trigLastTS - TSlevel_A[0]) - stimAr.loc[end_mask, 'stimulusOnsetTS']
    # rename columns to conform to bids
    stimAr.rename(columns = {'stimulusOnsetTS': 'onset'}, inplace = True)
    # Add a trial_type column with information needed for decoding
    #stimAr['trial_type'] = stimAr.apply(lambda row: f"{row['stimulusType']}_{row['stimulusLocation']}_{row['response']}", axis=1)
    stimAr['trial_type'] = stimAr['stimulusType']

    return stimAr


# %%  Log file checks 

# perform various checks on the MRI triggers and event logs
def logChecks(stim, triggerTs, idx, TSlevel_A, TSlevel_B, TSlevel_A_full, triggerTs_times):
    # first & last timestamp of an event
    TSfirstEV = stim['stimulusOnsetTS'].iloc[0]/1000
    TSlastEV = stim['stimulusOnsetTS'].iloc[-1]/1000
    # create error flags
    errLogFlag = []
    # 1. check that level start event and first estimated trigger are close enough in time (<33ms; i.e. approx. 2 frame)
    if abs(TSlevel_A[0] - triggerTs[0])>0.033:
        print('!!! WARNING: Run: ' + str(idx+1) + ' ! 1st level_start event and first trigger differ by more than 33ms (exactly = ' + str(np.round(abs(TSlevel_A[0] - triggerTs[0]),3)*1000) + 'ms). Minor differences indicate stimulus PC hiccups and may not be problematic. Check log files !!!')
        errLogFlag.append(1)
    # 2. check that no event takes places before first trigger 
    if TSfirstEV < triggerTs[0]:
        print('!!! WARNING: Run: ' + str(idx+1) + ' ! first stimulus event takes place BEFORE first trigger. Check log files !!!')
        errLogFlag.append(2)
    # 3. check that no event takes place before level start
    if TSfirstEV < TSlevel_A[0]:
        print('!!! WARNING: Run: ' + str(idx+1) + ' ! first stimulus event takes place BEFORE start 1st level event. Check log files !!!')
        errLogFlag.append(3)
    # 4. check that last event takes place before last trigger
    if (TSlastEV > triggerTs[-1]) and logsHaveMriTriggers:
        print('!!! WARNING: Run: ' + str(idx+1) + ' ! last stimulus event takes place AFTER last trigger. Check log files !!!')
        errLogFlag.append(4)
    # 5. check that no events take place after level complete event (except for triggers)
    if TSlastEV > TSlevel_B[1]:
        print('!!! WARNING: Run: ' + str(idx+1) + ' ! last stimulus event takes place AFTER 2nd level complete event. Check log files !!!')
        errLogFlag.append(5)
    # 6. check that level start is before level complete
    if TSlevel_A[0] > TSlevel_A[1] or TSlevel_B[0] > TSlevel_B[1]:
        print('!!! WARNING: Run: ' + str(idx+1) + ' ! level start event takes place AFTER level complete event. Check log files !!!')
        errLogFlag.append(6)
    # 7. check that triggers still come in after level complete
    if (triggerTs[-1] < TSlevel_B[1]) and logsHaveMriTriggers:
        print('!!! WARNING: Run: ' + str(idx+1) + ' ! triggers stop BEFORE 2nd level complete event. Check log files !!!')
        errLogFlag.append(7) 
    # 8. check that last trigger and last level complete timestamp differ by a reasonable amount (max. ~1min)
    if (abs(triggerTs[-1] - TSlevel_B[1]) > 60) and logsHaveMriTriggers:
        print('!!! WARNING: Run: ' + str(idx+1) + ' ! last trigger and 2nd level complete event timestamps differ by ' + str(np.round(abs(triggerTs[-1] - TSlevel_B[1]),1)) + 's. Check log files !!!')
        errLogFlag.append(8) 
    # 9. check that first level complete event and second level start event differ by less than 30 sec.
    if abs(TSlevel_B[0] - TSlevel_A[1]) > 30:
        print('!!! WARNING: Run: ' + str(idx+1) + ' ! 1st level complete event and 2nd level start event differ by more than 30s. Check log files !!!')
        errLogFlag.append(9) 
    # 10. check that the critical level start event (level 1) from full log and mri log match
    if abs(TSlevel_A[0] - TSlevel_A_full[0])>0.033:
        print('!!! WARNING: Run: ' + str(idx+1) + ' ! timestamp of run start from full log does NOT match FMRI log (exactly = ' + str(np.round(abs(TSlevel_A[0] - TSlevel_A_full[0]),3)*1000) + 'ms). Check log files !!!')
        errLogFlag.append(10)
    # 11. check whether & when MRI triggers are dropped, as evident by larger than 1TR (+50ms) diff between successive triggers. Flags any trigger drops during levels (+100ms margin at end of levels; expected hiccups due to loading/saving)
    if logsHaveMriTriggers:
        droppedTrigsTS = triggerTs_times[np.where(np.diff(triggerTs_times) > 1.55)[0]]
        droppedA = np.logical_and((droppedTrigsTS > TSlevel_A[0]), (droppedTrigsTS+0.1 < TSlevel_A[1]))
        droppedB = np.logical_and((droppedTrigsTS > TSlevel_B[0]), (droppedTrigsTS+0.1 < TSlevel_B[1]))
        if any(droppedA) or any(droppedB):
            print('!!! CAUTION: Run: ' + str(idx+1) + ' Dropped '+ str(np.sum(droppedA)+np.sum(droppedB)) + ' trigger(s)! Level 1 [start end]: ' + str(np.round(TSlevel_A,2))  + ' | Level 2 [start end]: ' + str(np.round(TSlevel_B,2)) + ' | Triggers dropped at TS: ' + str(np.round(droppedTrigsTS[np.logical_or(droppedA,droppedB)],2)) + '. If dropped triggers are close to level start/end (within 1 frame; i.e. 16.6ms) this can occur.')
            errLogFlag.append(11)
  
    # note if no error was found
    if not errLogFlag:
        print('... Run: ' + str(idx+1) + ' passed checks without error.')
    return errLogFlag


# %%  create 3 column event files

# generic event files relevant for most analyses (null events)
def createEvFile_nullAndEndOfRunEV(log, outputEventFilePattern, sub, runType):
    for stimType in log['stimulusType'].dropna().unique():
    # null events have no location or probe, hence no need to loop over different responses or types
        if (stimType == 'NullEvent' or stimType == 'endOfRun'):
            #  get onsets, set durations, and parametric mod to create 3 column format event file
            onsets = (log['onset'][log['stimulusType']==stimType]).to_numpy()
            durations = (log['duration'][log['stimulusType']==stimType]).to_numpy()
            parametricMod = np.ones((len(onsets)))
            event = np.vstack((onsets,durations,parametricMod)).T
            # save event file
            fname = outputEventFilePattern%{'sub':sub, 'runType':runType, 'eventType':stimType, 'analysisType':''}
            saveEventFile(fname,event)

# event file splitting into category and side
def createEvFile_categoryOnly(log, stimulusDuration, outputEventFilePattern, sub, runType):
    for stimType in log['stimulusType'].dropna().unique():
        if not (stimType == 'NullEvent' or stimType == 'endOfRun'):
            #  get onsets, set durations and parametric mod to create 3 column format event file
            onsets = (log['onset'][log['stimulusType']==stimType]).to_numpy()
            parametricMod = np.ones((len(onsets)))
            durations = np.tile(stimulusDuration,len(onsets))
            event = np.vstack((onsets,durations,parametricMod)).T
            # save event file
            fname = outputEventFilePattern%{'sub':sub, 'runType':runType, 'eventType':stimType, 'analysisType':'categoryOnly'}
            saveEventFile(fname,event)   

# event file splitting into category and side
def createEvFile_categorySide(log, relevantLocations, stimulusDuration, outputEventFilePattern, sub, runType):
    # Debug check: NaNs in stimulusType will appear as floats (np.nan) and can break string concatenation below.
    if log['stimulusType'].isna().any():
        print(f"!!! WARNING: NaNs found in stimulusType for {sub} {runType}: n_nan={int(log['stimulusType'].isna().sum())}")
    for stimType in log['stimulusType'].dropna().unique():
        if not (stimType == 'NullEvent' or stimType == 'endOfRun'):
            for location in relevantLocations:
                #  get onsets, set durations, and parametric mod to create 3 column format event file
                onsets = (log['onset'][(log['stimulusType']==stimType) & (log['stimulusLocation'].str.contains(location))]).to_numpy()
                durations = np.tile(stimulusDuration,len(onsets))
                parametricMod = np.ones((len(onsets)))
                event = np.vstack((onsets,durations,parametricMod)).T
                # save event file
                fname = outputEventFilePattern%{'sub':sub, 'runType':runType, 'eventType':stimType+location, 'analysisType':'categorySide'}
                saveEventFile(fname,event)

# event file splitting into category and side and seen/unseen
def createEvFile_seenCategorySide(log, relevantLocations, stimulusDuration, outputEventFilePattern, sub, runType):
    
    for stimType in log['stimulusType'].dropna().unique():
        #loop over location (left vs right)
        for location in relevantLocations:
            #loop over probed vs unprobed
            for probed in log['type'][np.logical_and(log['type']!='NullEvent', log['type']!='endOfRun')].unique():
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
                        elif ((response=='FalseNegative') & (stimType=='None')) | ((response=='TruePositive') & (stimType=='None')):
                            continue
                        else:
                            #  get onsets, set durations, and parametric mod to create 3 column format event file
                            onsets = (log['onset'][(log['stimulusType']==stimType) & (log['stimulusLocation'].str.contains(location)) & (log['type']==probed) & (log['response']==response)]).to_numpy()
                            durations = np.tile(stimulusDuration,len(onsets))
                            parametricMod = np.ones((len(onsets)))
                            event = np.vstack((onsets,durations,parametricMod)).T
                            # save event file
                            
                            fname = outputEventFilePattern%{'sub':sub, 'runType':runType, 'eventType':probeLabel+ResponseLabels[response]+stimType+location, 'analysisType':'seenCategorySide'}
                            saveEventFile(fname,event)
                # if its unprobed no need to split into response types
                else:
                    #  get onsets, set durations, and parametric mod to create 3 column format event file
                    onsets = (log['onset'][(log['stimulusType']==stimType) & (log['stimulusLocation'].str.contains(location)) & (log['type']==probed)]).to_numpy()
                    durations = np.tile(stimulusDuration,len(onsets))
                    parametricMod = np.ones((len(onsets)))
                    event = np.vstack((onsets,durations,parametricMod)).T
                    # save event file
                    probeLabel = 'unprobed'
                    fname = outputEventFilePattern%{'sub':sub, 'runType':runType, 'eventType':probeLabel+stimType+location, 'analysisType':'seenCategorySide'}
                    saveEventFile(fname,event)


# %% Run subject
def runSubject(sub):
    print('========== SUBJECT: ' + sub + ' ==========')
    
    # Define the scannerID by checking which is the log folder with the highest number
    the_path = path_check_scannerID%{'sub':sub}    
    folders = [f for f in os.listdir(the_path) if os.path.isdir(os.path.join(the_path, f))]

    # Filter the folders that are numeric
    numeric_folders = [f for f in folders if f.isdigit()]

    # Get the folder with the highest numeral
    scannerID = max(numeric_folders, key=int) # session ID letter for MRI scanning session
    
    
    # import relevant log files; to be improved later
    logPath = logPathPattern%{'sub':sub, 'scannerID':scannerID} 
    
    # get stimulus details log
    fname = logPath + logStimuli%{'sub':sub, 'scannerID':scannerID}
    # Prevent the literal string "None" (a valid stimulusType) from being parsed as missing.
    # Still treat truly empty fields as missing.
    dataStimuli = pd.read_csv(fname, sep=';', keep_default_na=False, na_values=["", "nan", "NaN"])
    
    # get probe details log for Video Game levels
    fname = logPath + logProbes%{'sub':sub, 'scannerID':scannerID}
    dataProbes = pd.read_csv(fname, sep=';', keep_default_na=False, na_values=["", "nan", "NaN"])

    # get details log for Replay levels
    fname = logPath + logReplay%{'sub':sub, 'scannerID':scannerID}
    dataReplay = pd.read_csv(fname, sep=';', keep_default_na=False, na_values=["", "nan", "NaN"])
    
    # get FMRI log file
    fname = logPath + logMRI%{'sub':sub, 'scannerID':scannerID}
    dataMRI = getMRIlog(fname)
    
    # get MR triggers & button presses from log files, depending if data was acqurired at Yale or Donders
    if np.char.startswith(sub, 'SC'):
        fname = logPath + logTriggers_donders%{'sub':sub, 'scannerID':scannerID}
        triggerTs_times = getTriggerTs_times_donders(fname)
    elif np.char.startswith(sub, 'SD'):
        fname = logPath + logTriggers_yale%{'sub':sub, 'scannerID':scannerID}
        triggerTs_times = getTriggerTs_times_yale(fname)
        
    # Filter dataProbes and dataReplay to keep only relevant responses
    dataProbes = dataProbes[dataProbes['responseEvaluation'].isin(probeResponseTypes)]
    dataReplay = dataReplay[dataReplay['responseEvaluation'].isin(probeResponseTypes)]

    
    # print number of dicom images
    if logsHaveMriTriggers:
        countNDcmVols(logPath)
    
    # % loop over runs
    replayCounter = 0
    errLogFlag = [[],[]]
    errLogFlag_full_logs = [[],[]]
    for idx in range(len(runOnsetLevels)):
        
        # get relevant data for current run from stimuli and probes
        stim = dataStimuli[["stimulusOnsetTS","type","stimulusType","stimulusLocation","stimulusName"]][dataStimuli["currentLevelID"].between(runOnsetLevels[idx], runOnsetLevels[idx]+1)]
        probe = dataProbes[["responseEvaluation"]][dataProbes["currentLevelID"].between(runOnsetLevels[idx], runOnsetLevels[idx]+1)]
        replay = dataReplay[["responseEvaluation"]][dataReplay["currentLevelID"].between(runOnsetLevels[idx], runOnsetLevels[idx]+1)]
        
        ##### get exact run (level) onset time from FMRI log
        # level 1 (of a run);
        TSlevel_A = (dataMRI['TS'][(dataMRI['run']==runLabelsMRIlog[idx]) & (dataMRI['level']=='1') & (dataMRI['evType']=='LEVEL_BEGIN')].to_numpy(dtype=float))/1000
        TSlevel_A = np.append(TSlevel_A, (dataMRI['TS'][(dataMRI['run']==runLabelsMRIlog[idx]) & (dataMRI['level']=='1') & (dataMRI['evType']=='LEVEL_END')].to_numpy(dtype=float))/1000)
        # level 2 (of a run);
        TSlevel_B = (dataMRI['TS'][(dataMRI['run']==runLabelsMRIlog[idx]) & (dataMRI['level']=='2') & (dataMRI['evType']=='LEVEL_BEGIN')].to_numpy(dtype=float))/1000
        TSlevel_B = np.append(TSlevel_B, (dataMRI['TS'][(dataMRI['run']==runLabelsMRIlog[idx]) & (dataMRI['level']=='2') & (dataMRI['evType']=='LEVEL_END')].to_numpy(dtype=float))/1000)

        # get timestamp when run ended from MRI log (level complete popup hidden; i.e. onset of leaderboard)
        TSlevel_B_endOfRun_log = dataMRI['TS'][(dataMRI['run']==runLabelsMRIlog[idx]) & (dataMRI['evType']=='LEVEL_COMPLETE_POPUP_HIDDEN')].to_numpy(dtype=float)[0]/1000
        
        
        ##### get exact run onset time and MR triggers timestamps from full logs
        errLogFlag_full_logs[0].append('Run%s' %(idx+1)) # Check if more than one full log is present (e.g. due to restart)
        
        # level 1 (of a run); level start (run start) and level end time (level end times should also correspond to null event onset times; this needs to be checked XXX) 
        fname = glob(logPath + logFull%{'sub':sub, 'scannerID':scannerID} + runByLevelAndWorldCode[idx] + '*COMPLETED.csv')
        TSlevel_A_full, TSMriTrig_A, ck1 = getFullLog(fname,idx)
 
        # level 2 (of a run); level 2 start and level 2 end time (level end times should also correspond to null event onset times; this needs to be checked XXX)
        fname = glob(logPath + logFull%{'sub':sub, 'scannerID':scannerID} + complementaryLevelAndWorldCode[idx] + '*COMPLETED.csv')
        TSlevel_B_full, TSMriTrig_B, ck2 = getFullLog(fname,idx)
        
        errLogFlag_full_logs[1].append(ck1+ck2)
        
        # discard triggers to account for dummy volumes
        triggerTs = np.concatenate((TSMriTrig_A,TSMriTrig_B))
        triggerTs = triggerTs[dummyTriggers::]
        
        
        ##### get the last trigger from this run (estimated from serial dump and log file)
        trigLastTS = getLastTrigTS(triggerTs_times, TSlevel_B_endOfRun_log)
        
        
        ##### Checks #####
        errLogFlag[0].append('Run%s' %(idx+1))
        errLogFlag[1].append(logChecks(stim, triggerTs, idx, TSlevel_A, TSlevel_B, TSlevel_A_full, triggerTs_times))
        
        
        ##### Create log file with relevant events per run #####
        if not onlyRunChecks:
            # Gather responses from different log files depending on whether is a video game run or a replay run
            if runByLevelAndWorldCode[idx][0]=='L':
                log = createMriLog(stim, replay, 'LOCALIZER_STIMULUS', TSlevel_A, TSlevel_B, TSlevel_B_endOfRun_log, trigLastTS)
                replayCounter += 1
                runType = 'task-Replay_run-' + str(replayCounter)
            else:
                log = createMriLog(stim, probe, 'GAME_PROBE', TSlevel_A, TSlevel_B, TSlevel_B_endOfRun_log, trigLastTS)
                runType = 'task-VG_run-' + str(idx+1)
            
            # Adjust timestamps of BIDS events files by adding the time of dummy scans
            log['onset'] = log['onset'] + dummyTriggers*TR
            
            fname = outputLogFilePattern%{'sub':sub, 'runType':runType}
            log.to_csv(fname, sep='\t', index=False, na_rep='Null')
            
            
            ##### Create event files for GLM per run#####
            print('... Run: ' + str(idx+1) + ' creating event files.')
            
            # Adjust timestamps of FSL events files by removing the time of dummy scans
            log['onset'] = log['onset'] - dummyTriggers*TR
            
            createEvFile_nullAndEndOfRunEV(log, outputEventFilePattern, sub, runType)
            createEvFile_categoryOnly(log, stimulusDuration, outputEventFilePattern, sub, runType)
            createEvFile_categorySide(log, relevantLocations, stimulusDuration, outputEventFilePattern, sub, runType)
            createEvFile_seenCategorySide(log, relevantLocations, stimulusDuration, outputEventFilePattern, sub, runType)
            
    # save log file warning flags
    fname = (outputErrorFlagPattern%{'sub':sub,'ses':'mri02'})
    errLogDf = pd.DataFrame({"RunNo": errLogFlag[0], "WarningCode": errLogFlag[1]})

    if any(errLogFlag[1]):
        saveErrorFlags(fname, errLogDf)
    
    # Save error log in case that one full log is present (e.g. due to restart)
    fname = (outputErrorFlagPatternFullLogs%{'sub':sub,'ses':'mri02'})
    errLogDf = pd.DataFrame({"RunNo": errLogFlag_full_logs[0], "WarningCode": errLogFlag_full_logs[1]})
    
    if any(errLogFlag_full_logs[1]):
        saveErrorFlags(fname, errLogDf)
    

def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Extract exp.2 fMRI log files and write BIDS events / FSL EV files for one subject.',
    )
    parser.add_argument(
        '--sub-code',
        default=os.environ.get('SUB_CODE', '').strip() or None,
        help='Subject code (e.g. SC108). May also set env SUB_CODE.',
    )
    args = parser.parse_args(argv)

    if not args.sub_code:
        print(
            'ERROR: pass --sub-code SUB (or set SUB_CODE) for one subject.\n'
            '  Submit all subjects: bash 01_submit_01_exp2_fMRI_logfile_extraction_and_checks.sh',
            file=sys.stderr,
        )
        return 1

    runSubject(args.sub_code)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())