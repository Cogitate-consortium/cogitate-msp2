#!/usr/bin/env bash
# Copy 2nd-level gPPI cope1/zstat1 maps (FFA cope3, LOC cope4) for EVC ROI tables.
#
# Usage:
#   bash run_copy_gppi_2nd_lvl_for_evc.sh
#   bash run_copy_gppi_2nd_lvl_for_evc.sh --dry-run

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_CODE_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../demo_paths.sh
source "${_CODE_ROOT}/demo_paths.sh"

set -euo pipefail

GPPI_CODE_ROOT="${CODE_PATH}/gPPI_code"
FILTER_COL="${FILTER_COL:-SYNCHRONY_min_seen}"
FSLFEAT_ROOT="${BIDS_ROOT}/derivatives/fslFeat"
DEST_DIR="${GPPI_2ND_LVL_DEST:-${BIDS_ROOT}/derivatives/gppi/2nd_lvl}"

export PYTHONPATH="${GPPI_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

extra=()
if [ "${1:-}" = "--dry-run" ]; then
    extra+=(--dry-run)
fi

python3 "${GPPI_CODE_ROOT}/copy_gppi_2nd_lvl_for_evc.py" \
    --bids-root "$BIDS_ROOT" \
    --subject-csv "$SUBJECT_CSV" \
    --filter-col "$FILTER_COL" \
    --feat-root "$FSLFEAT_ROOT" \
    --dest-dir "$DEST_DIR" \
    "${extra[@]}"
