#!/usr/bin/env bash

# Descriptive gPPI conjunction: dAT-seen ∩ AT-seen-go ∩ AT-seen-no-go
# (FFA and LOC seeds; cluster-thresholded 3rd-level thresh_zstat1 maps).
#
# Preregistration ver 5.9 — fMRI psychophysiological interaction analysis:
# after group RFX + GRF cluster correction, a descriptive conjunction tests
# whether similar regions appear across all three seen conditions.
#
# Prerequisites: 3rd-level FEAT for dAT-seen, AT-seen-go, and AT-seen-no-go.
#
# Usage: bash run_gPPI_conjunction.sh

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_CODE_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../../demo_paths.sh
source "${_CODE_ROOT}/demo_paths.sh"

set -euo pipefail

GPPI_CONJ_ROOT="${CODE_PATH}/gPPI_code/gPPI_conjunction_analysis"
FSLFEAT_ROOT="${BIDS_ROOT}/derivatives/fslFeat"
OUT_DIR="${GPPI_CONJ_OUT_DIR:-${BIDS_ROOT}/derivatives/gppi/conjunction}"
MASKS_PATH="${MASKS_PATH:-${BIDS_ROOT}/derivatives/masks/ICBM2009c_asym_nlin}"

export PYTHONPATH="${GPPI_CONJ_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python3 "${GPPI_CONJ_ROOT}/gppi_conjunction.py" \
    --bids-root "$BIDS_ROOT" \
    --feat-root "$FSLFEAT_ROOT" \
    --out-dir "$OUT_DIR" \
    --masks-path "$MASKS_PATH"
