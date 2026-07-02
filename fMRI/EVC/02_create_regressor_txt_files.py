#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loads BIDS-compliant EVCLoc events.tsv per subject, writes FSL 3-column EV txt
files (TLBR, TRBL, buttonPress) and fmriprep-derived confound regressors.
Requires EVC/01_create_events_tsv.py to have run successfully first.

Environment overrides (optional):
    BIDS_ROOT, CODE_PATH, REGRESSOR_EVENT_ROOT, SUBJECT_CSV,
    SUBJECT_CSV_COLUMNS, CONFOUND_REGRESSOR_SET

Pipeline: run after EVC/01_create_events_tsv.py, before EVC/03_make_fsf_files.sh

Tested on python v3.7.4, pandas v0.25.2, numpy v1.17.2

Created 12.10.2020
@author: David Richter (d.richter@donders.ru.nl)
@author: Yamil Vidal (hvidaldossantos@gmail.com)
Modified by Yamil Vidal 08/05/2026
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

##### Parameters #####
sessionLabel = 'V2'
sessionDir = f'ses-{sessionLabel}'
runType = 'task-EVCLoc_run-1'

relevantEvents = ['TRBL', 'TLBR', 'buttonPress']
stimulusDuration = 15.25
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
_CANONICAL_REGRESSOR_ROOT = os.path.join(
    BIDS_ROOT, 'derivatives', 'exclude', 'new', 'regressoreventfiles',
)
REGRESSOR_EVENT_ROOT = os.environ.get('REGRESSOR_EVENT_ROOT', _CANONICAL_REGRESSOR_ROOT)
FMRIPREP_ROOT = os.path.join(BIDS_ROOT, 'derivatives', 'fmriprep')


def _assert_canonical_regressor_root(root):
    norm = os.path.normpath(root).replace('\\', '/')
    if '/derivatives/regressoreventfiles' not in norm:
        raise ValueError(
            'REGRESSOR_EVENT_ROOT must be under derivatives/regressoreventfiles: '
            f'{root}'
        )
    if '/derivatives/regressoreventfiles' in norm and '/derivatives/regressoreventfiles' not in norm:
        raise ValueError(
            'REGRESSOR_EVENT_ROOT must not use legacy derivatives/regressoreventfiles: '
            f'{root}'
        )


_assert_canonical_regressor_root(REGRESSOR_EVENT_ROOT)

SUBJECT_CSV = os.environ.get(
    'SUBJECT_CSV',
    os.path.join(CODE_PATH, 'ses-v2-analysis-subs-fmri.csv'),
)
SUBJECT_CSV_COLUMNS = os.environ.get(
    'SUBJECT_CSV_COLUMNS',
    'SYNCHRONY_min_seen',
)
SUBJECT_CSV_COLUMN = os.environ.get('SUBJECT_CSV_COLUMN', '').strip()
CONFOUND_REGRESSOR_SET = os.environ.get('CONFOUND_REGRESSOR_SET', '24motion_CSF_WM')

events_tsv_pattern = os.path.join(
    BIDS_ROOT,
    'sub-{sub}',
    'ses-{ses}',
    'func',
    'sub-{sub}_ses-{ses}_{runType}_events.tsv',
)
events_qc_pattern = os.path.join(
    BIDS_ROOT,
    'derivatives',
    'exclude',
    'new',
    'logfilechecks',
    'sub-{sub}_ses-{ses}-EVCLoc_errorFlags.csv',
)
output_event_pattern = os.path.join(
    REGRESSOR_EVENT_ROOT,
    'sub-{sub}',
    'ses-{ses}',
    'sub-{sub}_ses-{ses}_{runType}_{eventType}_EV.txt',
)
confound_output_dir_pattern = os.path.join(
    REGRESSOR_EVENT_ROOT,
    'sub-{sub}',
    'ses-{ses}',
    'confound_event_files',
)
confound_output_pattern = os.path.join(
    confound_output_dir_pattern,
    'sub-{sub}_ses-{ses}_{runType}_confounds.txt',
)
output_status_pattern = os.path.join(
    BIDS_ROOT,
    'derivatives',
    'exclude',
    'new',
    'logfilechecks',
    'sub-{sub}_ses-{ses}-EVCLoc_regressor_status.csv',
)

if CODE_PATH not in sys.path:
    sys.path.insert(0, CODE_PATH)
from helper_functions_MRI import saveErrorFlags, saveEventFile  # noqa: E402


def load_glm_confound_module():
    glm_path = os.path.join(CODE_PATH, 'glm', '01_create_confound_regressor_ev_file.py')
    if not os.path.isfile(glm_path):
        raise FileNotFoundError(f'missing glm confound script: {glm_path}')
    spec = importlib.util.spec_from_file_location('glm_confounds', glm_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        meta_cols = {'sub_code', 'modality', 'Lab'}
        analysis_cols = [c for c in subj_df.columns if c not in meta_cols]
        if not analysis_cols:
            raise ValueError(f'no analysis columns in {csv_path}')
        mask = subj_df[analysis_cols].apply(lambda s: s.astype(str).str.upper().eq('TRUE')).any(axis=1)
        return subj_df.loc[mask, 'sub_code'].values

    mask = pd.Series(False, index=subj_df.index)
    for col in cols:
        if col not in subj_df.columns:
            raise ValueError(f'missing column {col} in {csv_path}')
        mask = mask | subj_df[col].astype(str).str.upper().eq('TRUE')
    return subj_df.loc[mask, 'sub_code'].values


def write_status(flag_path, sub, status, note=''):
    status_df = pd.DataFrame(
        [{
            'sub_code': sub,
            'session': sessionLabel,
            'status': status,
            'note': note,
        }]
    )
    saveErrorFlags(flag_path, status_df)


def read_events_qc(sub):
    """Return (status, note) from EVC/01_create_events_tsv.py QC file, or (None, reason) if missing."""
    qc_path = events_qc_pattern.format(sub=sub, ses=sessionLabel)
    if not os.path.isfile(qc_path):
        return None, f'missing QC file (run EVC/01_create_events_tsv.py first): {qc_path}'
    qc = pd.read_csv(qc_path)
    if qc.empty:
        return None, f'empty QC file: {qc_path}'
    row = qc.iloc[0]
    status = str(row.get('status', '')).upper()
    note = str(row.get('note', ''))
    codes = str(row.get('warning_codes', ''))
    if codes and codes != 'nan':
        note = f'{note}; warning_codes={codes}'.strip('; ')
    return status, note


def create_regressor_ev_files(data_log, sub):
    """Write TLBR, TRBL, buttonPress EV txt files (FEAT timing: minus dummy TRs)."""
    written = []
    all_stim = (
        (data_log['trial_type'] == 'TLBR')
        | (data_log['trial_type'] == 'TRBL')
        | (data_log['trial_type'] == 'Null')
    )
    all_event_ts = data_log['onset'][all_stim].to_numpy()
    all_durations = np.append(all_event_ts[1::] - all_event_ts[0:-1], stimulusDuration)
    all_events = data_log['trial_type'][all_stim].to_numpy()

    for stim_type in relevantEvents:
        if stim_type != 'buttonPress':
            onsets = all_event_ts[all_events == stim_type]
            durations = all_durations[all_events == stim_type]
        else:
            onsets = data_log['onset'][data_log['trial_type'] == 'buttonPress'].to_numpy()
            durations = np.zeros(len(onsets))

        onsets = onsets - (nDummyVols * TR)
        parametric_mod = np.ones(len(onsets))
        event = np.vstack((onsets, durations, parametric_mod)).T

        out_path = output_event_pattern.format(
            sub=sub, ses=sessionLabel, runType=runType, eventType=stim_type,
        )
        saveEventFile(out_path, event)
        written.append(out_path)
    return written


def create_confound_file(sub, glm_conf):
    """Build confound EV file from fmriprep for task-EVCLoc_run-1."""
    subj_id = f'sub-{sub}'
    func_dir = os.path.join(FMRIPREP_ROOT, subj_id, sessionDir, 'func')
    candidates = [
        os.path.join(
            func_dir,
            f'{subj_id}_{sessionDir}_{runType}_desc-confounds_timeseries.tsv',
        ),
        os.path.join(
            func_dir,
            f'{subj_id}_{sessionDir}_{runType}_desc-confounds_regressors.tsv',
        ),
    ]
    input_fname = next((p for p in candidates if os.path.isfile(p)), None)
    if input_fname is None:
        raise FileNotFoundError(
            f'missing fmriprep confounds TSV for {subj_id} {sessionDir} {runType}'
        )

    confound_dir = confound_output_dir_pattern.format(sub=sub, ses=sessionLabel)
    os.makedirs(confound_dir, exist_ok=True)
    output_fname = confound_output_pattern.format(sub=sub, ses=sessionLabel, runType=runType)
    confounds_of_interest = glm_conf.get_confound_regressor_list(CONFOUND_REGRESSOR_SET)
    glm_conf.create_confound_regressor_file(
        input_fname, output_fname, confounds_of_interest, nDummyVols,
    )
    return output_fname


if __name__ == '__main__':
    single_col = SUBJECT_CSV_COLUMN or None
    subjects = load_subject_codes(
        SUBJECT_CSV,
        columns_csv=None if single_col else SUBJECT_CSV_COLUMNS,
        single_column=single_col,
    )
    glm_conf = load_glm_confound_module()

    filter_desc = single_col or f'OR({SUBJECT_CSV_COLUMNS})'
    print(f'>> {len(subjects)} subjects ({filter_desc}); confounds={CONFOUND_REGRESSOR_SET}')

    n_pass = n_warn = n_skip = n_error = 0
    failed = []

    for sub in subjects:
        print('========== SUBJECT: ' + sub + ' ==========')
        status_path = output_status_pattern.format(sub=sub, ses=sessionLabel)
        events_tsv = events_tsv_pattern.format(sub=sub, ses=sessionLabel, runType=runType)

        qc_status, qc_note = read_events_qc(sub)
        if qc_status is None:
            print(f' - SKIP: {qc_note}')
            write_status(status_path, sub, 'SKIP', qc_note)
            n_skip += 1
            failed.append(sub)
            continue
        if qc_status in ('SKIP', 'ERROR'):
            note = f'EVC/01_create_events_tsv.py QC={qc_status}: {qc_note}'
            print(f' - SKIP: {note}')
            write_status(status_path, sub, 'SKIP', note)
            n_skip += 1
            failed.append(sub)
            continue

        if not os.path.isfile(events_tsv):
            note = f'missing events.tsv: {events_tsv}'
            print(f' - ERROR: {note}')
            write_status(status_path, sub, 'ERROR', note)
            n_error += 1
            failed.append(sub)
            continue

        try:
            data_log = pd.read_csv(events_tsv, sep='\t')
            ev_files = create_regressor_ev_files(data_log, sub)
            confound_file = create_confound_file(sub, glm_conf)
            note = (
                f'events={events_tsv}; ev_files={len(ev_files)}; '
                f'confounds={confound_file}'
            )
            if qc_status == 'WARN':
                note = f'EVC01_WARN: {qc_note}; {note}'
                write_status(status_path, sub, 'WARN', note)
                n_warn += 1
            else:
                write_status(status_path, sub, 'PASS', note)
                n_pass += 1
        except Exception as exc:
            print(f'ERROR: {sub}: {exc}', file=sys.stderr)
            write_status(status_path, sub, 'ERROR', str(exc))
            n_error += 1
            failed.append(sub)

    print(
        f'>> summary: pass={n_pass} warn={n_warn} skip={n_skip} error={n_error} '
        f'(total={len(subjects)})'
    )
    if failed:
        print('>> failed/skipped:', ', '.join(failed))
        sys.exit(1)
