#!/usr/bin/env bash

#@author: Yamil Vidal (05/2026)
# Build gPPI 3rd-level inclusion report under gPPI_code/3rd level report/.
# Lists included subjects per analysis/cope and traces missing derivatives for
# exclusions (seeds, anatomical masks, structural images, events, fmriprep, FEAT).
# Called from 10_run_PPI_3rd_level.sh (after job submission).
#
# Usage: bash make_3rd_level_inclusion_report.sh

set -euo pipefail

BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
SUBJECT_CSV="${CODE_PATH}/ses-v2-analysis-subs-fmri.csv"
GPPI_CODE_ROOT="${CODE_PATH}/gPPI_code"
TIMECOURSE_ROOT="${BIDS_ROOT}/derivatives/gppi_timecourse"
REPORT_DIR="${GPPI_CODE_ROOT}/3rd level report"
FSF_FILES_ROOT="${GPPI_CODE_ROOT}/fsf_files/group"
INCLUSION_REPORT="${FSF_FILES_ROOT}/included_subjects_report.txt"
RUNS_REPORT="${GPPI_CODE_ROOT}/job_files/run_PPI_2nd_level/included_runs_report.txt"
SUBMISSION_REPORT="${GPPI_CODE_ROOT}/job_files/run_PPI_3rd_level/submission_inclusion_report.txt"
REPORT_PY="${GPPI_CODE_ROOT}/report_3rd_level_inclusion.py"

if [ ! -f "$REPORT_PY" ]; then
    echo "ERROR: report script not found: $REPORT_PY" >&2
    exit 1
fi

if [ ! -f "$SUBJECT_CSV" ]; then
    echo "ERROR: subject CSV not found: $SUBJECT_CSV" >&2
    exit 1
fi

mkdir -p "$REPORT_DIR"

BIDS_ROOT="$BIDS_ROOT" \
GPPI_CODE_ROOT="$GPPI_CODE_ROOT" \
TIMECOURSE_ROOT="$TIMECOURSE_ROOT" \
SUBJECT_CSV="$SUBJECT_CSV" \
INCLUSION_REPORT="$INCLUSION_REPORT" \
RUNS_REPORT="$RUNS_REPORT" \
SUBMISSION_REPORT="$SUBMISSION_REPORT" \
REPORT_DIR="$REPORT_DIR" \
python3 "$REPORT_PY"
