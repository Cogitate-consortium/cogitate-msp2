#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extracts events relevant for fMRI analysis from log file of EVC Localizer, then
creates events.tsv file per subject and run. Also writes associated json
sidecars.
Performs various tests on log files.

Inputs:
    - log files from EVC localizer

Outputs:
    - one log file checks table with warning flags per subject (always written)
    - one MRI specific, bids compliant *_event.tsv file per run

Environment overrides (optional):
    BIDS_ROOT, CODE_PATH, RAW_DIR, SUBJECT_CSV, SUBJECT_CSV_COLUMNS, SUBJECT_CSV_COLUMN

Pipeline: run before EVC/02_create_regressor_txt_files.py

Tested on python v3.7.4, pandas v0.25.2, numpy v1.17.2

Created 12.10.2020
@author: David Richter (d.richter@donders.ru.nl)
@author: Yamil Vidal (hvidaldossantos@gmail.com)
Modified by Yamil Vidal 08/05/2026
"""

import os
import sys
from glob import glob
from shutil import copyfile

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

EVC_DIR = os.path.dirname(os.path.abspath(__file__))

##### Parameters #####
sessionLabel = 'V2'

stimulusDuration = 15.25
nExpectedStimuli = 20
nExpectedNullEVs = 2

TR = 1.5
nDummyVols = 3

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

logPathPattern = os.path.join(
    RAW_DIR, '{sub}', '{sub}_MR_{ses}', 'RESOURCES', 'BEHEVCLoc', ''
)
logFile = '{sub}E{run}_Beh_EVCLoc.csv'

jsonSidecarTemplate = os.path.join(
    EVC_DIR,
    'events-json_file-templates',
    'events-json_file-template_ses-V2_task-EVCLoc.json',
)

outputLogFilePattern = os.path.join(
    BIDS_ROOT,
    'sub-{sub}',
    'ses-{ses}',
    'func',
    'sub-{sub}_ses-{ses}_{runType}_events',
)
outputErrorFlagPattern = os.path.join(
    BIDS_ROOT,
    'derivatives',
    'exclude',
    'new',
    'logfilechecks',
    'sub-{sub}_ses-{ses}-EVCLoc_errorFlags.csv',
)

SUBJECT_CSV = os.environ.get(
    'SUBJECT_CSV',
    os.path.join(CODE_PATH, 'ses-v2-analysis-subs-fmri.csv'),
)
SUBJECT_CSV_COLUMNS = os.environ.get('SUBJECT_CSV_COLUMNS', 'SYNCHRONY_min_seen')
SUBJECT_CSV_COLUMN = os.environ.get('SUBJECT_CSV_COLUMN', '').strip()

if CODE_PATH not in sys.path:
    sys.path.insert(0, CODE_PATH)
from helper_functions_MRI import saveErrorFlags  # noqa: E402


def load_subject_codes(csv_path, columns_csv=None, single_column=None):
    subj_df = pd.read_csv(csv_path, sep=None, engine='python')
    if 'sub_code' not in subj_df.columns and len(subj_df.columns) == 1 and ';' in subj_df.columns[0]:
        subj_df = pd.read_csv(csv_path, sep=';')
    if 'sub_code' not in subj_df.columns:
        raise ValueError(f'missing sub_code column in {csv_path}')

    if single_column:
        cols = [single_column]
    elif columns_csv:
        cols = [c.strip() for c in columns_csv.split(',') if c.strip()]
    else:
        raise ValueError('set SUBJECT_CSV_COLUMNS or SUBJECT_CSV_COLUMN')

    mask = pd.Series(False, index=subj_df.index)
    for col in cols:
        if col not in subj_df.columns:
            raise ValueError(f'missing column {col} in {csv_path}')
        mask = mask | subj_df[col].astype(str).str.upper().eq('TRUE')
    return subj_df.loc[mask, 'sub_code'].values


def write_error_flags(flag_path, sub, status, warning_codes=None, note=''):
    """Always write per-subject QC status (PASS, WARN, SKIP, or ERROR)."""
    codes = warning_codes or []
    err_df = pd.DataFrame(
        [{
            'sub_code': sub,
            'session': sessionLabel,
            'status': status,
            'warning_codes': ','.join(str(c) for c in codes),
            'n_warnings': len(codes),
            'note': note,
        }]
    )
    saveErrorFlags(flag_path, err_df)


def logChecks_EvcLocalizer(dataLog, TR, nExpectedNullEVs, nExpectedStimuli, stimulusDuration):
    """
    Perform various checks on the evc localizer log files.
    Returns a list of warning codes (empty list = all checks passed).
    """
    allEventTS = dataLog['Time'][
        (dataLog['EventID'] == 'TLBR')
        | (dataLog['EventID'] == 'TRBL')
        | (dataLog['EventID'] == 'Null')
    ].to_numpy()
    errLogFlag = []

    if np.sum(dataLog['EventID'] == 'Null') != nExpectedNullEVs:
        print(
            '!!! WARNING: Number of null events does not correspond to expected number of null events. Obs='
            + str(np.sum(dataLog['EventID'] == 'Null'))
            + '. Check log files !!!'
        )
        errLogFlag.append(1)

    if np.sum(dataLog['EventID'] == 'TLBR') != nExpectedStimuli / 2:
        print(
            '!!! WARNING: Number of TLBR events does not correspond to expected number of TLBR events. Obs='
            + str(np.sum(dataLog['EventID'] == 'TLBR'))
            + '. Check log files !!!'
        )
        errLogFlag.append(2)

    if np.sum(dataLog['EventID'] == 'TRBL') != nExpectedStimuli / 2:
        print(
            '!!! WARNING: Number of TRBL events does not correspond to expected number of TRBL events. Obs='
            + str(np.sum(dataLog['EventID'] == 'TRBL'))
            + '. Check log files !!!'
        )
        errLogFlag.append(3)

    if any((abs(np.diff(allEventTS) - stimulusDuration)) > 0.1):
        print(
            '!!! WARNING: Event(s) took place at an unexpected interval. Incorrect event intervals: n='
            + str(np.sum(abs((np.diff(allEventTS) - stimulusDuration)) > 0.1))
            + '. Max divergence in ms='
            + str(int(np.max(abs(np.diff(allEventTS) - stimulusDuration)) * 1000))
            + '  Check log files !!!'
        )
        errLogFlag.append(4)

    if len(allEventTS) == 0 or (allEventTS[0] - TR) > 0.1:
        print(
            '!!! WARNING: First stimulus event is unexpectedly delayed or missing. Check log files !!!'
        )
        errLogFlag.append(5)

    if np.sum(dataLog['EventID'] == 'buttonPress') == 0:
        print('!!! WARNING: No button presses found. Check log files !!!')
        errLogFlag.append(6)

    if not errLogFlag:
        print('... EVC localizer run passed checks without error.')
    return errLogFlag


def createEventsTsv_EvcLocalizer(dataLog, output_pattern, sub, runType):
    """Create events.tsv per run (BIDS: onset relative to first imaging volume)."""
    print('... Creating MRI events.tsv file')
    allStimEvents_noPresses = (
        (dataLog['EventID'] == 'TLBR')
        | (dataLog['EventID'] == 'TRBL')
        | (dataLog['EventID'] == 'Null')
    )
    allEventTS = dataLog['Time'][allStimEvents_noPresses].to_numpy()
    allDurations = np.append(allEventTS[1::] - allEventTS[0:-1], stimulusDuration)
    allEvents = dataLog['EventID'][allStimEvents_noPresses].to_numpy()

    allEventTS = np.append(allEventTS, dataLog['Time'][dataLog['EventID'] == 'buttonPress'].to_numpy())
    allDurations = np.append(
        allDurations,
        np.zeros(np.sum(dataLog['EventID'] == 'buttonPress')),
    )
    allEvents = np.append(allEvents, dataLog['EventID'][dataLog['EventID'] == 'buttonPress'].to_numpy())

    onset = allEventTS + (nDummyVols * TR)
    outputDF = pd.DataFrame({'onset': onset, 'duration': allDurations, 'trial_type': allEvents})
    outputDF = outputDF.sort_values(by=['onset']).reset_index(drop=True)

    fname = output_pattern.format(sub=sub, runType=runType, ses=sessionLabel) + '.tsv'
    outputDF.to_csv(fname, sep='\t', index=False, na_rep='Null')
    return fname


def copy_json_sidecar(output_pattern, sub, runType):
    fname = output_pattern.format(sub=sub, runType=runType, ses=sessionLabel) + '.json'
    if not os.path.isfile(jsonSidecarTemplate):
        raise FileNotFoundError(f'missing JSON sidecar template: {jsonSidecarTemplate}')
    copyfile(jsonSidecarTemplate, fname)
    return fname


if __name__ == '__main__':
    single_col = SUBJECT_CSV_COLUMN or None
    subjects = load_subject_codes(
        SUBJECT_CSV,
        columns_csv=None if single_col else SUBJECT_CSV_COLUMNS,
        single_column=single_col,
    )
    filter_desc = single_col or f'OR({SUBJECT_CSV_COLUMNS})'
    print(f'>> {len(subjects)} subjects ({filter_desc} == TRUE; {SUBJECT_CSV})')

    n_skip = 0
    n_warn = 0
    n_pass = 0
    n_error = 0
    failed_subjects = []

    for sub in subjects:
        print('========== SUBJECT: ' + sub + ' ==========')
        flag_path = outputErrorFlagPattern.format(sub=sub, ses=sessionLabel)
        runType = 'task-EVCLoc_run-1'

        logPath = logPathPattern.format(sub=sub, ses=sessionLabel)
        fname_glob = os.path.join(logPath, logFile.format(sub=sub, run='*'))
        matched_logs = glob(fname_glob)
        nRuns = len(matched_logs)

        if nRuns != 1:
            note = f'expected 1 EVCLoc log file, found {nRuns}: {fname_glob}'
            print(' - CAUTION ! --> ' + note)
            print(' - SKIPPING PARTICIPANT ! - ')
            write_error_flags(flag_path, sub, 'SKIP', note=note)
            n_skip += 1
            failed_subjects.append(sub)
            continue

        try:
            dataLog = pd.read_csv(matched_logs[0])
            warning_codes = logChecks_EvcLocalizer(
                dataLog, TR, nExpectedNullEVs, nExpectedStimuli, stimulusDuration
            )
            events_tsv = createEventsTsv_EvcLocalizer(
                dataLog, outputLogFilePattern, sub, runType
            )
            copy_json_sidecar(outputLogFilePattern, sub, runType)

            if warning_codes:
                write_error_flags(
                    flag_path, sub, 'WARN', warning_codes=warning_codes,
                    note=f'events.tsv written: {events_tsv}',
                )
                n_warn += 1
            else:
                write_error_flags(
                    flag_path, sub, 'PASS', note=f'events.tsv written: {events_tsv}',
                )
                n_pass += 1

        except Exception as exc:
            print(f'ERROR: {sub}: {exc}', file=sys.stderr)
            write_error_flags(flag_path, sub, 'ERROR', note=str(exc))
            n_error += 1
            failed_subjects.append(sub)

    print(
        f'>> summary: pass={n_pass} warn={n_warn} skip={n_skip} error={n_error} '
        f'(total={len(subjects)})'
    )
    if failed_subjects:
        print('>> failed/skipped:', ', '.join(failed_subjects))
        sys.exit(1)
