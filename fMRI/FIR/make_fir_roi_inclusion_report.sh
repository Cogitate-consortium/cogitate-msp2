#!/usr/bin/env bash

#@author: Yamil Vidal (05/2026)
# Build FIR ROI averaging inclusion report under FIR/inclusion report/.
# Lists subjects included in ROI CSV outputs (combined + category-matched) and traces
# missing derivatives for exclusions. Called from 10_average_FIR_in_FFA_and_LOC.sh.
#
# Usage: bash make_fir_roi_inclusion_report.sh

set -euo pipefail

BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"
CODE_PATH="${BIDS_ROOT}/code"
SUBJECT_CSV="${CODE_PATH}/ses-v2-analysis-subs-fmri.csv"
FIR_CODE_ROOT="${CODE_PATH}/FIR"
REPORT_DIR="${FIR_CODE_ROOT}/inclusion report"
FIR_DERIV_ROOT="${BIDS_ROOT}/derivatives/fir"
ERROR_LOG="${FIR_DERIV_ROOT}/fir_averaging_error_log.txt"
REPORT_PY="${FIR_CODE_ROOT}/report_fir_roi_inclusion.py"

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
REPORT_DIR="$REPORT_DIR" \
FIR_DERIV_ROOT="$FIR_DERIV_ROOT" \
ERROR_LOG="$ERROR_LOG" \
python3 "$REPORT_PY"
