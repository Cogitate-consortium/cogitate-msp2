#!/usr/bin/env bash

#@author: Yamil Vidal (05/2026)
# Build FIR 3rd-level inclusion report under FIR/3rd level report/.
# Called from 09_run_FIR_3rd_level.sh (after job submission).
#
# Usage: bash make_fir_3rd_level_inclusion_report.sh

set -euo pipefail

BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
SUBJECT_CSV="${CODE_PATH}/ses-v2-analysis-subs-fmri.csv"
FIR_CODE_ROOT="${CODE_PATH}/FIR"
REPORT_DIR="${FIR_CODE_ROOT}/3rd level report"
FSF_FILES_ROOT="${FIR_CODE_ROOT}/fsf_files/group"
INCLUSION_REPORT="${FSF_FILES_ROOT}/included_subjects_report.txt"
SUBMISSION_REPORT="${FIR_CODE_ROOT}/job_files/run_FIR_3rd_level/submission_inclusion_report.txt"
REPORT_PY="${FIR_CODE_ROOT}/report_fir_3rd_level_inclusion.py"

# shellcheck source=fir_variants.sh
source "${FIR_CODE_ROOT}/fir_variants.sh"

FIR_VARIANTS_CSV="$(IFS=,; echo "${FIR_ALL_VARIANTS[*]}")"
FIR_3RD_COPES_CSV="$(IFS=,; echo "${FIR_3RD_COPES[*]}")"

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
FIR_CODE_ROOT="$FIR_CODE_ROOT" \
SUBJECT_CSV="$SUBJECT_CSV" \
FIR_SUBJECT_FILTER_COL="$FIR_SUBJECT_FILTER_COL" \
FIR_VARIANTS_CSV="$FIR_VARIANTS_CSV" \
FIR_3RD_COPES_CSV="$FIR_3RD_COPES_CSV" \
INCLUSION_REPORT="$INCLUSION_REPORT" \
SUBMISSION_REPORT="$SUBMISSION_REPORT" \
REPORT_DIR="$REPORT_DIR" \
python3 "$REPORT_PY"
