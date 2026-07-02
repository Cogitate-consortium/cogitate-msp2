#!/usr/bin/env bash

#@author: Yamil Vidal (05/2026)
# Build activation 3rd-level inclusion report under activation/3rd level report/.
# Lists included subjects per contrast and traces missing derivatives for exclusions.
# Called from 07_run_3rd_level.sh (after job submission).
#
# Usage: bash make_3rd_level_inclusion_report.sh

set -euo pipefail

BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
SUBJECT_CSV="${CODE_PATH}/ses-v2-analysis-subs-fmri.csv"
ACTIVATION_CODE_ROOT="${CODE_PATH}/activation"
REPORT_DIR="${ACTIVATION_CODE_ROOT}/3rd level report"
FSF_FILES_ROOT="${ACTIVATION_CODE_ROOT}/fsf_files/group"
INCLUSION_REPORT="${FSF_FILES_ROOT}/included_subjects_report.txt"
RUNS_REPORT="${ACTIVATION_CODE_ROOT}/job_files/run_activation_2nd_level/included_runs_report.txt"
SUBMISSION_REPORT="${ACTIVATION_CODE_ROOT}/job_files/run_activation_3rd_level/submission_inclusion_report.txt"
REPORT_PY="${ACTIVATION_CODE_ROOT}/report_3rd_level_inclusion.py"

declare -a ANALYSES=(
    "seen_face_vs_unseen_face:analysis-2ndGLM_seen_face_vs_unseen_face_space-MNI152NLin2009cAsym:ACTIVATION_min_sf_uf:1:seen_vs_unseen_F"
    "seen_object_vs_unseen_object:analysis-2ndGLM_seen_object_vs_unseen_object_space-MNI152NLin2009cAsym:ACTIVATION_min_so_uo:2:seen_vs_unseen_O"
)
ANALYSES_CSV="$(printf '%s\n' "${ANALYSES[@]}")"

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
ACTIVATION_CODE_ROOT="$ACTIVATION_CODE_ROOT" \
SUBJECT_CSV="$SUBJECT_CSV" \
INCLUSION_REPORT="$INCLUSION_REPORT" \
RUNS_REPORT="$RUNS_REPORT" \
SUBMISSION_REPORT="$SUBMISSION_REPORT" \
REPORT_DIR="$REPORT_DIR" \
ANALYSES_CSV="$ANALYSES_CSV" \
python3 "$REPORT_PY"
