#!/usr/bin/env bash
# Python-backed EVC ROI averaging for 2nd-level gPPI (cope + zstat tables).
# Keeps 11_average_gppi_in_evc.sh as the legacy bash implementation.
#
# Usage: bash run_average_gppi_in_evc.sh

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_CODE_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../demo_paths.sh
source "${_CODE_ROOT}/demo_paths.sh"

set -euo pipefail

module load FSL 2>/dev/null || true

GPPI_CODE_ROOT="${CODE_PATH}/gPPI_code"
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
