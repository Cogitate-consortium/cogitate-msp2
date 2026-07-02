#!/usr/bin/env bash

#@author: Yamil Vidal (05/2026)
# Review EVC ROI outputs: list successful subjects and trace missing derivatives for failures.
# Run after 06_make_evc_roi.sh (or anytime to audit evc_rois/).
#
# Writes: EVC/inclusion report/evc_roi_inclusion_report.txt
#
# Usage: bash 07_review_EVC.sh

set -euo pipefail

EVC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=evc_config.sh
source "${EVC_DIR}/evc_config.sh"

REPORT_DIR="${EVC_CODE_ROOT}/inclusion report"
REPORT_PY="${EVC_CODE_ROOT}/report_evc_roi_inclusion.py"
RAW_DIR="${RAW_DIR:-/mnt/beegfs/XNAT/COGITATE/fMRI/Raw/projects/CoG_fMRI_PhaseII}"

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
CODE_PATH="$CODE_PATH" \
SUBJECT_CSV="$SUBJECT_CSV" \
REPORT_DIR="$REPORT_DIR" \
EVC_ROIS_ROOT="$EVC_ROIS_ROOT" \
RAW_DIR="$RAW_DIR" \
SUBJECT_CSV_COLUMNS="$SUBJECT_CSV_COLUMNS" \
SUBJECT_CSV_COLUMN="$SUBJECT_CSV_COLUMN" \
python3 "$REPORT_PY"

echo ">> inclusion report: ${REPORT_DIR}/evc_roi_inclusion_report.txt"
