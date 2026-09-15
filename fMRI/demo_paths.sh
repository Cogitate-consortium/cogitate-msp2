#!/usr/bin/env bash
# Shared BIDS / code paths for production and demo mode.
# Source from pipeline scripts:
#   _CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # if in a subfolder
#   source "${_CODE_ROOT}/demo_paths.sh"
# Or from the fMRI repo root:
#   source "$(dirname "${BASH_SOURCE[0]}")/demo_paths.sh"
#
# When DEMO=1 (after demo_setup.sh), keep exported paths. Otherwise default to
# the cluster layout. Never overwrite variables that are already set.

_DEMO_PATHS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_DEFAULT_BIDS_ROOT="/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids"

if [ "${DEMO:-0}" = "1" ] || [ "${DEMO:-}" = "true" ] || [ "${DEMO:-}" = "TRUE" ]; then
    export DEMO=1
    export CODE_PATH="${CODE_PATH:-${_DEMO_PATHS_DIR}}"
    export BIDS_ROOT="${BIDS_ROOT:-${CODE_PATH}/data_demo}"
    export SUBJECT_CSV="${SUBJECT_CSV:-${CODE_PATH}/ses-v2-analysis-subs-fmri_demo.csv}"
else
    export BIDS_ROOT="${BIDS_ROOT:-${_DEFAULT_BIDS_ROOT}}"
    export CODE_PATH="${CODE_PATH:-${BIDS_ROOT}/code}"
    export SUBJECT_CSV="${SUBJECT_CSV:-${CODE_PATH}/ses-v2-analysis-subs-fmri.csv}"
fi

export SESSION="${SESSION:-ses-V2}"
export REGRESSOR_EVENT_ROOT="${REGRESSOR_EVENT_ROOT:-${BIDS_ROOT}/derivatives/regressoreventfiles}"
