#!/usr/bin/env bash
# Python-backed EVC ROI averaging for 2nd-level gPPI (cope + zstat tables).
# Keeps 11_average_gppi_in_evc.sh as the legacy bash implementation.
#
# Usage: bash run_average_gppi_in_evc.sh

set -euo pipefail

module load FSL 2>/dev/null || true

BIDS_ROOT="${BIDS_ROOT:-/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids}"
CODE_PATH="${CODE_PATH:-${BIDS_ROOT}/code}"
GPPI_CODE_ROOT="${CODE_PATH}/gPPI_code"
SUBJECT_CSV="${SUBJECT_CSV:-${CODE_PATH}/ses-v2-analysis-subs-fmri.csv}"
FILTER_COL="${FILTER_COL:-SYNCHRONY_min_seen}"
FSLFEAT_ROOT="${BIDS_ROOT}/derivatives/fslFeat"
EVC_ROIS_ROOT="${BIDS_ROOT}/derivatives/evc_rois"
OUT_DIR="${BIDS_ROOT}/derivatives/gppi"

export PYTHONPATH="${GPPI_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python3 "${GPPI_CODE_ROOT}/average_gppi_in_evc.py" \
    --bids-root "$BIDS_ROOT" \
    --subject-csv "$SUBJECT_CSV" \
    --filter-col "$FILTER_COL" \
    --feat-root "$FSLFEAT_ROOT" \
    --evc-rois-root "$EVC_ROIS_ROOT" \
    --out-dir "$OUT_DIR"
